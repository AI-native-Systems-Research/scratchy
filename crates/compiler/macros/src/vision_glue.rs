// SPDX-License-Identifier: Apache-2.0
//! Per-variant vision-arch glue codegen.
//!
//! For each `#[vision_forward]` variant module the macro emits, this
//! module produces:
//!
//! - `impl scratchy_target_cuda::VisionArchWeights for Weights` baking
//!   the variant's [`scratchy_vision::VisionConfig`] from
//!   `vision_*` bounds + `vision_norm_eps` scalar, forwarding
//!   `pixel_pack` to the user-supplied free fn, and forwarding
//!   `vision_forward` to the variant's local emitted `forward`.
//! - `fn try_load_mm(...)` — d_model fingerprint (the only
//!   per-variant geometric difference within a VL family),
//!   patch_embed 5D-conv flatten, then `Weights::load(...)` and
//!   wrap in `scratchy_target_cuda::VisionWrapper`.
//! - `inventory::submit!` rows for tp ∈ {1, 2, 4, 8} (2/4/8 under
//!   `#[cfg(feature = "nccl")]`). Vision is replicated per-rank;
//!   the same `try_load_mm` runs at every rank.
//!
//! Net: per-arch crates carry no `MultimodalForward` impl, no
//! `try_load_mm`, no `inventory::submit!` rows. The DSL body and a
//! `pixel_pack = path` attribute arg are the whole arch surface.

use proc_macro2::TokenStream;
use quote::quote;

use crate::config::ModelParams;
use crate::weights_manifest::PadHint;

/// Read a required `vision_*` bound (or `d_model`) by name. Vision
/// configs always carry these (G.5.b's
/// `qwen2_vl_vision_weights_manifest_anchors_on_vision_bounds` pins
/// the manifest); a missing key is a config bug, not a runtime
/// concern.
fn bound(model: &ModelParams, key: &str) -> u64 {
    *model.bounds.get(key).unwrap_or_else(|| {
        panic!(
            "vision config `{}` is missing required bound `{key}`",
            model.source_stem,
        )
    })
}

/// Read the `vision_norm_eps` scalar. Mandatory in every vision
/// config; missing is a config bug.
fn norm_eps(model: &ModelParams) -> f64 {
    *model.scalars.get("vision_norm_eps").unwrap_or_else(|| {
        panic!(
            "vision config `{}` is missing required scalar `vision_norm_eps`",
            model.source_stem,
        )
    })
}

/// Emit the per-variant vision-arch glue. Returned `TokenStream` is
/// inserted into the variant's `pub mod` body alongside the
/// `Weights` struct + emitted `forward` fn.
///
/// `pixel_pack`: `None` to use the trait's default impl
/// (delegates to [`VisionConfig::patches_from_normalized_chw`] —
/// the spatial-merge order shared by Qwen2-VL / Qwen2.5-VL / any
/// arch with the same `patch_size · spatial_merge_size` convention).
/// `Some(path)` to override with `path` taking
/// `(&VisionConfig, &[f32], u32, u32) -> (Vec<u16>, (u32, u32, u32))`
/// — for arches with different patch ordering (SigLIP raster, etc.).
pub fn emit_per_variant(
    model: &ModelParams,
    arch_name: &str,
    pixel_pack: Option<&syn::Path>,
    pad_to_mult8: &[PadHint],
    processor: &syn::Path,
) -> TokenStream {
    let embed_dim = bound(model, "vision_embed_dim") as u32;
    let depth = bound(model, "vision_depth") as u32;
    let num_heads = bound(model, "vision_num_heads") as u32;
    let patch_size = bound(model, "vision_patch_size") as u32;
    let temporal_patch_size = bound(model, "vision_temporal_patch_size") as u32;
    let spatial_merge_size = bound(model, "vision_spatial_merge_size") as u32;
    let in_chans = bound(model, "vision_in_chans") as u32;
    let d_model = bound(model, "d_model") as u32;
    let eps = norm_eps(model) as f32;

    // d_model fingerprint: per-arch on-disk weight + dim that
    // `try_load_mm` reads to discriminate among variants. Default
    // is Qwen's `visual.merger.mlp.2.weight` dim 0 — every existing
    // Qwen2-VL / Qwen2.5-VL config keeps that path byte-identically.
    let fingerprint = model
        .arch
        .fingerprint
        .clone()
        .expect("vision fingerprint enforced at config parse");
    let fp_key_lit = syn::LitStr::new(&fingerprint.key, proc_macro2::Span::call_site());
    let fp_dim_lit = proc_macro2::Literal::usize_unsuffixed(fingerprint.dim);
    // Emit `.first()` for dim 0 (the common case) rather than `.get(0)` — same
    // result, but avoids clippy's get_first on the generated fingerprint.
    let fp_dim_access = if fingerprint.dim == 0 {
        quote! { first() }
    } else {
        quote! { get(#fp_dim_lit) }
    };

    // Patch-embed flatten target: per-arch on-disk conv weight that
    // ships in 4D (SigLIP `[E, C, P, P]`) or 5D (Qwen `[E, C, T, P, P]`)
    // form. We multiply every dim after `leading_dim` to land at a
    // dense 2D `[d_lead, prod_after]`. Default is Qwen's 5D path.
    let flatten = model
        .arch
        .patch_embed_flatten
        .clone()
        .expect("vision patch_embed_flatten enforced at config parse");
    let flatten_key_lit = syn::LitStr::new(&flatten.key, proc_macro2::Span::call_site());
    let flatten_lead_lit = proc_macro2::Literal::usize_unsuffixed(flatten.leading_dim);
    // Precompute `leading_dim + 1` at codegen time so the emitted `.skip(..)`
    // is a plain literal (avoids emitting `.skip(0 + 1)` — a clippy identity_op).
    let flatten_lead_next_lit = proc_macro2::Literal::usize_unsuffixed(flatten.leading_dim + 1);
    let flatten_channels_last = flatten.channels_last;

    // Construct the `VisionWrapper`, attaching the learned positional-
    // embedding table host-side when the arch declares one
    // (`vision_pos_embed_key`). The wrapper then interpolates it
    // host-side per forward (bilinear or bicubic per the declared
    // `POS_EMB_INTERP`) and uploads the result as the `pos_embeds`
    // extern. Other arches emit a plain `VisionWrapper::new`.
    //
    // `num_grid_per_side` derives from ELEMENT COUNT / embed_dim —
    // NOT `shape[0]` — because checkpoints ship the square table both
    // flattened (`[ng², d]`, Qwen3.5-VL `[2304, 1152]`) and as a grid
    // (`[ng, ng, d]`, MoonViT `[64, 64, 1152]`): `sqrt(shape[0])`
    // mis-derived ng=8 for the latter and the interp read a garbage
    // 8×8 view (live pos_embeds absmax 26 vs golden 195).
    let pe_embed_dim_lit = proc_macro2::Literal::u32_unsuffixed(embed_dim);
    let wrapper_ctor: TokenStream = match &model.arch.pos_embed_key {
        ::std::option::Option::Some(key) => {
            let key_lit = syn::LitStr::new(key, proc_macro2::Span::call_site());
            quote! {{
                let mut __vw = crate::__gpu::VisionWrapper::new(weights);
                if let ::std::option::Option::Some(__pe_tbl) = gw.tensor_to_f32(#key_lit) {
                    let __rows = __pe_tbl.len() / (#pe_embed_dim_lit as usize);
                    let __ng = (__rows as f64).sqrt().round() as usize;
                    ::std::assert_eq!(
                        __ng * __ng,
                        __rows,
                        "learned pos-embed table is not a square grid",
                    );
                    __vw = __vw.with_pos_embed_table(__pe_tbl, __ng);
                }
                __vw
            }}
        }
        ::std::option::Option::None => {
            quote! { crate::__gpu::VisionWrapper::new(weights) }
        }
    };

    // Optional style keys (validated at config parse). Absent →
    // Qwen defaults, keeping every existing VL config byte-identical
    // in behavior.
    let rope_style_tokens: TokenStream = match model.arch.rope_style.as_deref() {
        Some("interleaved_xy") => {
            quote! { ::scratchy_vision::VisionRopeStyle::InterleavedXy }
        }
        _ => quote! { ::scratchy_vision::VisionRopeStyle::NeoxHw },
    };
    let pos_emb_interp_tokens: TokenStream = match model.arch.pos_emb_interp.as_deref() {
        Some("bicubic") => quote! { ::scratchy_vision::PosEmbInterp::Bicubic },
        _ => quote! { ::scratchy_vision::PosEmbInterp::Bilinear },
    };

    let embed_dim_lit = proc_macro2::Literal::u32_unsuffixed(embed_dim);
    let depth_lit = proc_macro2::Literal::u32_unsuffixed(depth);
    let num_heads_lit = proc_macro2::Literal::u32_unsuffixed(num_heads);
    let patch_size_lit = proc_macro2::Literal::u32_unsuffixed(patch_size);
    let temporal_patch_size_lit = proc_macro2::Literal::u32_unsuffixed(temporal_patch_size);
    let spatial_merge_size_lit = proc_macro2::Literal::u32_unsuffixed(spatial_merge_size);
    // Optional projector AvgPool2d kernel (gemma3-mm SigLIP: 4). Absent
    // on merge-only towers (Qwen*-VL / MoonViT) → 0 = no pooling.
    let pool_kernel = *model.bounds.get("vision_pool_kernel").unwrap_or(&0) as u32;
    let pool_kernel_lit = proc_macro2::Literal::u32_unsuffixed(pool_kernel);
    let in_chans_lit = proc_macro2::Literal::u32_unsuffixed(in_chans);
    let d_model_lit = proc_macro2::Literal::u32_unsuffixed(d_model);
    let d_model_usize_lit = proc_macro2::Literal::usize_unsuffixed(d_model as usize);
    let eps_lit = proc_macro2::Literal::f32_suffixed(eps);

    // Per-block zero-pad expansion for `__pad_to_mult8__` manifest
    // entries. Each PadHint's stem is treated as a per-block weight
    // suffix (matching the per-block convention shared with
    // `__packed_splits__`); the emitted prelude walks
    // `0..vision_depth` and pads the on-disk `.weight` tensor at CPU
    // side before any `Weights::load` runs. Top-level (non-per-block)
    // pads aren't needed today; add a `top_level: bool` PadHint field
    // when a future arch requires it.
    let layered_prefix = {
        let layout = model
            .arch
            .safetensors
            .clone()
            .expect("vision safetensors layout enforced at config parse");
        format!("{}.{}", layout.default_root, layout.layered_subpath)
    };
    let pad_calls: Vec<TokenStream> = pad_to_mult8
        .iter()
        .map(|hint| {
            let stem = &hint.weight;
            let dim_lit = proc_macro2::Literal::usize_unsuffixed(hint.dim);
            let template = format!("{layered_prefix}.{{l}}.{stem}.weight");
            let template_lit = syn::LitStr::new(&template, proc_macro2::Span::call_site());
            quote! {
                for __l in 0..#depth_lit {
                    let __key = #template_lit.replace("{l}", &__l.to_string());
                    gw.pad_axis_to_mult8(&__key, #dim_lit, 8)?;
                }
            }
        })
        .collect();
    let pad_prelude = if pad_calls.is_empty() {
        quote! {}
    } else {
        quote! {
            // CPU-side weight padding: for each `__pad_to_mult8__`
            // hint, grow the named weight's specified axis to the
            // next multiple of 8 with zero fill. This dodges cuBLAS
            // bf16 GEMM rejecting `K=intermediate_size=3420` on
            // Qwen2.5-VL-3B; the macro's downstream `Linear::load`
            // sees the padded shape directly. No-op if already
            // mult-of-8.
            #(#pad_calls)*
        }
    };

    let arch_name_lit = syn::LitStr::new(arch_name, proc_macro2::Span::call_site());
    let hf_arch_lits: Vec<syn::LitStr> = model
        .architectures
        .iter()
        .map(|s| syn::LitStr::new(s, proc_macro2::Span::call_site()))
        .collect();

    // `pixel_pack` override: emit a method body iff the user
    // supplied an explicit path; otherwise let the trait's default
    // impl run (which calls `VisionConfig::patches_from_normalized_chw`).
    let pixel_pack_method = pixel_pack.map(|path| {
        quote! {
            fn pixel_pack(
                cfg: &::scratchy_vision::VisionConfig,
                pixels: &[f32],
                height: u32,
                width: u32,
            ) -> (::std::vec::Vec<u16>, (u32, u32, u32)) {
                #path(cfg, pixels, height, width)
            }
        }
    });

    // Window-attention dispatch override: arches with a
    // `vision_window_size` bound (Qwen2.5-VL family) override the
    // trait default `None` to return `Some(window_size)`. The
    // VisionWrapper reads this at request time and builds the
    // window-grouped permutation + permutes cos/sin host-side.
    let windowed_attn_method = model.bounds.get("vision_window_size").map(|window_size| {
        let lit = proc_macro2::Literal::u32_unsuffixed(*window_size as u32);
        quote! {
            fn windowed_attn_window_size() -> ::std::option::Option<u32> {
                ::std::option::Option::Some(#lit)
            }
        }
    });

    // Learned positional embedding override (G.7(c.1)): arches with a
    // `vision_num_positions` bound (SigLIP / Gemma3-MM family) override
    // the trait default `None` to return `Some(num_pos)`. The
    // VisionWrapper reads this at request time and uploads
    // `[0..N, 0..N, ...]` as `vision_position_ids` so the body's
    // `pos_embed(...)` call has its index buffer.
    let pos_embed_method = model.bounds.get("vision_num_positions").map(|num_pos| {
        let lit = proc_macro2::Literal::u32_unsuffixed(*num_pos as u32);
        quote! {
            fn vision_num_positions() -> ::std::option::Option<u32> {
                ::std::option::Option::Some(#lit)
            }
        }
    });

    quote! {
        impl crate::__gpu::VisionArchWeights for Weights {
            fn vision_config(&self) -> &'static ::scratchy_vision::VisionConfig {
                static C: ::scratchy_vision::VisionConfig = ::scratchy_vision::VisionConfig {
                    embed_dim: #embed_dim_lit,
                    depth: #depth_lit,
                    num_heads: #num_heads_lit,
                    patch_size: #patch_size_lit,
                    temporal_patch_size: #temporal_patch_size_lit,
                    spatial_merge_size: #spatial_merge_size_lit,
                    pool_kernel: #pool_kernel_lit,
                    in_chans: #in_chans_lit,
                    d_model: #d_model_lit,
                    norm_eps: #eps_lit,
                    rope_style: #rope_style_tokens,
                    pos_emb_interp: #pos_emb_interp_tokens,
                };
                &C
            }

            fn mm_metadata() -> &'static ::scratchy_vision::MmMetadata {
                &#processor
            }

            #pixel_pack_method

            #windowed_attn_method

            #pos_embed_method

            unsafe fn vision_forward(
                &self,
                ctx: &crate::__gpu::ForwardCtx<'_>,
                device: &mut crate::__gpu::GpuDevice,
                num_tokens: u64,
            ) -> crate::__gpu::OwnedTensor {
                unsafe { forward(self, ctx, device, num_tokens) }
            }
        }

        /// Variant try-load. Fingerprints on `d_model` (from on-disk
        /// `visual.merger.mlp.2.weight`'s first dim — the only per-
        /// variant geometric difference within a VL family). Returns
        /// `Ok(None)` when the checkpoint isn't a vision tower or
        /// the d_model doesn't match this variant.
        #[allow(clippy::not_unsafe_ptr_arg_deref)]
        fn try_load_mm(
            gw: &mut crate::__gpu::weights::GpuWeights,
            stream: crate::__gpu::CUstream,
            max_model_len: usize,
            tp_rank: u8,
            _hf: ::scratchy_forward_compiler::HfFingerprint<'_>,
        ) -> ::anyhow::Result<::std::option::Option<
            ::std::boxed::Box<dyn ::scratchy_forward_compiler::MultimodalForward>,
        >> {
            // Sniff the per-arch d_model fingerprint key + dim —
            // discriminates among variants of one vision arch.
            // Default (Qwen): `visual.merger.mlp.2.weight` dim 0.
            let fp_shape = match gw.tensor_shape_any(#fp_key_lit) {
                ::std::option::Option::Some(s) => s,
                ::std::option::Option::None => return ::std::result::Result::Ok(::std::option::Option::None),
            };
            if fp_shape.#fp_dim_access.copied() != ::std::option::Option::Some(#d_model_usize_lit) {
                return ::std::result::Result::Ok(::std::option::Option::None);
            }
            // Patch-embed conv weight ships as 4D (SigLIP) or 5D
            // (Qwen) on disk; flatten dims after `leading_dim` into
            // one row so the macro's `LinearLayer::load_dense_or_ggml`
            // reads a dense 2D `[d_lead, prod_rest]` (no shape-
            // override entry point on the loader). No-op if the
            // tensor is already 2D.
            if let ::std::option::Option::Some(pe) =
                gw.tensor_shape_any(#flatten_key_lit)
                && pe.len() > 2
            {
                    if #flatten_channels_last {
                        // Conv weight (4D or 5D). Two on-disk layouts ship
                        // for the same arch: HF-native PyTorch Conv3d
                        // `[out, in, kt, kh, kw]` (channels-FIRST, e.g.
                        // `Qwen/Qwen3.5-9B`'s
                        // `model.visual.patch_embed.proj.weight` is
                        // `[1152, 3, 2, 16, 16]`) and MLX-converted
                        // checkpoints `[out, kt, kh, kw, in]` (channels-LAST).
                        // Sniff: if dim immediately AFTER the leading dim
                        // equals `in_chans`, the weight is channels-FIRST
                        // and just needs flattening; otherwise apply the
                        // permute. Hard-coding channels_last (the previous
                        // behavior) misperms HF-native weights — the result
                        // has the right total element count but every index
                        // is mispaired, silently corrupting all patches.
                        let in_chans_usize: usize = #in_chans_lit as usize;
                        gw.flatten_conv_weight(
                            #flatten_key_lit,
                            #flatten_lead_lit,
                            ::std::option::Option::Some(in_chans_usize),
                        )?;
                    } else {
                        let lead = pe[#flatten_lead_lit];
                        let rest: usize = pe.iter().skip(#flatten_lead_next_lit).product();
                        gw.reshape_in_place(#flatten_key_lit, &[lead, rest])?;
                    }
            }
            #pad_prelude
            let weights = load(gw, stream, max_model_len, tp_rank)?;
            ::std::result::Result::Ok(::std::option::Option::Some(
                ::std::boxed::Box::new(#wrapper_ctor)
                    as ::std::boxed::Box<dyn ::scratchy_forward_compiler::MultimodalForward>,
            ))
        }

        // The MM registry fn-pointer (`MmTryLoadFn`) names the allocator-erased
        // `GpuWeightsHandle`; this shim recovers the concrete
        // `&mut GpuWeights<BackendAllocator>` the per-arch `try_load_mm` above
        // expects, so the registration rows can reference one fn item that
        // coerces to the pointer.
        #[allow(clippy::not_unsafe_ptr_arg_deref)]
        fn __try_load_mm_erased(
            __gw_handle: ::scratchy_forward_compiler::GpuWeightsHandle<'_>,
            stream: crate::__gpu::CUstream,
            max_model_len: usize,
            tp_rank: u8,
            hf: ::scratchy_forward_compiler::HfFingerprint<'_>,
        ) -> ::anyhow::Result<::std::option::Option<
            ::std::boxed::Box<dyn ::scratchy_forward_compiler::MultimodalForward>,
        >> {
            let gw: &mut crate::__gpu::weights::GpuWeights =
                unsafe { __gw_handle.as_mut() };
            try_load_mm(gw, stream, max_model_len, tp_rank, hf)
        }

        // Vision is replicated per-rank — register at every tp size
        // so the worker's `try_load_mm` finds us regardless of
        // `tp_world_size`. Each rank loads the full `visual.*`
        // weights independently and runs `vision_forward` to
        // produce identical mm_embeds; the post-Embed
        // `SpliceMmEmbeds` D2D-copies them into placeholder rows
        // AFTER the vocab-parallel AllReduce.
        ::scratchy_forward_compiler::inventory::submit! {
            ::scratchy_forward_compiler::ScratchyMmRegistration {
                arch_name: #arch_name_lit,
                hf_arches: &[#(#hf_arch_lits),*],
                gguf_archs: &[],
                tp_world_size: 1,
                try_load_mm: __try_load_mm_erased,
                mm_metadata: #processor,
            }
        }
        #[cfg(feature = "nccl")]
        ::scratchy_forward_compiler::inventory::submit! {
            ::scratchy_forward_compiler::ScratchyMmRegistration {
                arch_name: #arch_name_lit,
                hf_arches: &[#(#hf_arch_lits),*],
                gguf_archs: &[],
                tp_world_size: 2,
                try_load_mm: __try_load_mm_erased,
                mm_metadata: #processor,
            }
        }
        #[cfg(feature = "nccl")]
        ::scratchy_forward_compiler::inventory::submit! {
            ::scratchy_forward_compiler::ScratchyMmRegistration {
                arch_name: #arch_name_lit,
                hf_arches: &[#(#hf_arch_lits),*],
                gguf_archs: &[],
                tp_world_size: 4,
                try_load_mm: __try_load_mm_erased,
                mm_metadata: #processor,
            }
        }
        #[cfg(feature = "nccl")]
        ::scratchy_forward_compiler::inventory::submit! {
            ::scratchy_forward_compiler::ScratchyMmRegistration {
                arch_name: #arch_name_lit,
                hf_arches: &[#(#hf_arch_lits),*],
                gguf_archs: &[],
                tp_world_size: 8,
                try_load_mm: __try_load_mm_erased,
                mm_metadata: #processor,
            }
        }
    }
}
