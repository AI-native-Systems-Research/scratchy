// SPDX-License-Identifier: Apache-2.0
//! Generic glue between a `#[vision_forward]`-emitted per-arch
//! `Weights` and the [`MultimodalForward`] trait the executor calls
//! at request time.
//!
//! Each VL/MM arch's macro expansion emits one [`VisionArchWeights`]
//! impl per variant — baking the variant's [`VisionConfig`] from
//! `configs/<variant>.json::vision_*` bounds + scalars, wiring its
//! `pixel_pack` to the user-supplied free fn, and forwarding to the
//! emitted free `forward(...)`. [`VisionWrapper<W>`] provides a
//! single generic [`MultimodalForward`] impl over any
//! `VisionArchWeights` — pixel pack on host, upload to GPU, build
//! cu_seqlens / cos / sin from `grid_thw`, build [`ForwardCtx`],
//! call the emitted forward, fill MRoPE grid info on each returned
//! [`EmbedPatch`].
//!
//! Net effect: per-arch crates drop their hand-written
//! `MultimodalForward` impl + `try_load_mm` shape-probe + inventory
//! rows. The DSL body becomes the only file in the per-arch crate
//! that carries arch-specific code (modulo the per-arch pixel-pack
//! free fn in `scratchy-vision`).

use crate::DType;
// Top-level re-exports (backend-polymorphic) — the `alloc` / `device`
// submodules are cuda-only; `OwnedTensor` / `GpuDevice` resolve to the
// per-backend type on both cuda and metal.
use crate::GpuDevice;
use crate::OwnedTensor;
use crate::kv_cache::KvCachePool;
use crate::tensor::GpuTensor;
use scratchy_vision::{
    VisionConfig, bf16_slice_as_bytes, build_cu_seqlens_i32, i32_slice_as_bytes,
    permute_rows_block_grouped_bf16, u32_slice_as_bytes,
};

use crate::mm_dispatch::{MultimodalForward, PixelInput};
use crate::{EmbedPatch, ForwardCtx};

/// Per-arch `Weights` glue — the macro emits one impl per variant
/// module. Carries the baked [`VisionConfig`], a pointer to the
/// per-arch host-side pixel-pack fn, and an `unsafe fn vision_forward`
/// that just calls the macro's emitted free `forward(...)`.
///
/// The trait is named `VisionArchWeights` (not just `VisionWeights`)
/// to leave the latter free in per-arch crates if they want a
/// concrete name. In practice they don't — [`VisionWrapper<W>`]
/// covers the whole `MultimodalForward` surface generically.
pub trait VisionArchWeights: Send + Sync + Sized + 'static {
    /// Geometric config, baked at macro-expansion time from the
    /// variant's `vision_*` bounds + `vision_norm_eps` scalar.
    fn vision_config(&self) -> &'static VisionConfig;

    /// CPU pixel pack: CHW pixels → varlen `[T*H*W, C·T·P²]` bf16
    /// patches + `(grid_t, grid_h, grid_w)`. Default delegates to
    /// [`VisionConfig::patches_from_normalized_chw`] — the spatial-
    /// merge order shared by Qwen2-VL / Qwen2.5-VL / any arch with
    /// the same `patch_size · spatial_merge_size` convention. Arches
    /// with different patch ordering (SigLIP raster, etc.) override
    /// via the `pixel_pack = path::to::fn` attribute arg, which the
    /// macro turns into an override of this method.
    fn pixel_pack(
        cfg: &VisionConfig,
        pixels: &[f32],
        height: u32,
        width: u32,
    ) -> (Vec<u16>, (u32, u32, u32)) {
        cfg.patches_from_normalized_chw(pixels, height, width)
    }

    /// Run the encoder body. Macro emits a one-line forwarder to
    /// the variant module's free `forward(...)`.
    ///
    /// # Safety
    /// `ctx` must have `pixels` / `cu_seqlens_q` / `max_seqlen_q` /
    /// `vision_rope_cos` / `vision_rope_sin` populated; `device`
    /// live.
    unsafe fn vision_forward(
        &self,
        ctx: &ForwardCtx<'_>,
        device: &mut GpuDevice,
        num_tokens: u64,
    ) -> OwnedTensor;

    /// Window-attention spatial window edge in pixels, when the body
    /// dispatches per-layer between full-frame and windowed varlen
    /// attention (Qwen2.5-VL family). Default `None` → the wrapper
    /// builds the simple single-cu-seqlens path (Qwen2-VL / SigLIP /
    /// any tower with one attention regime). `Some(window_size)` →
    /// the wrapper builds full + windowed cu_seqlens, the
    /// natural→window-grouped permutation pair, and applies it
    /// host-side to the rope cos/sin tables before upload, populating
    /// the matching `vision_*` fields on [`ForwardCtx`].
    fn windowed_attn_window_size() -> Option<u32> {
        None
    }

    /// Per-arch CPU-side preprocessing metadata declaration. The macro
    /// emits this as `&<crate>::PROCESSOR` so the executor can read
    /// declarative flags (e.g. `mrope_positions`) without naming any
    /// arch.
    fn mm_metadata() -> &'static scratchy_vision::MmMetadata;

    /// Number of learned positional embedding rows the body looks up
    /// via `pos_embed(position_ids, embeddings.position_embedding)`.
    /// `Some(N)` → the wrapper builds `[0..N, 0..N, ...]` u32 per
    /// image and uploads it into [`ForwardCtx::vision_position_ids`];
    /// `None` → no positional embedding (Qwen2-VL / Qwen2.5-VL use 2D
    /// RoPE via `vision_rope` instead). Used by SigLIP / Gemma3-MM.
    /// `N` matches the per-image patch count (image_size/patch_size)²,
    /// fixed at compile time for SigLIP-class arches with a fixed
    /// resize step.
    fn vision_num_positions() -> Option<u32> {
        None
    }
}

/// Generic [`MultimodalForward`] impl over any [`VisionArchWeights`].
/// Owns a `KvCachePool` placeholder because [`ForwardCtx::kv_cache`]
/// is non-optional but vision-body codegen never reads it.
pub struct VisionWrapper<W: VisionArchWeights> {
    pub weights: W,
    kv_placeholder: KvCachePool,
    /// Qwen3.5-VL learned positional-embedding table (the f32
    /// `vision_tower.pos_embed.weight`, `[num_grid², embed_dim]`) plus
    /// `num_grid_per_side`, captured host-side at load (the loader
    /// doesn't expose it as a runtime weight since the DSL never
    /// references it). When present, `vision_forward` runs
    /// `fast_pos_embed_interpolate` over it per forward and uploads the
    /// result as the `pos_embeds` extern. `None` for towers without a
    /// learned pos-embed.
    pos_embed_table: Option<(Vec<f32>, usize)>,
}

impl<W: VisionArchWeights> VisionWrapper<W> {
    pub fn new(weights: W) -> Self {
        Self {
            weights,
            kv_placeholder: KvCachePool::empty_for_vision(),
            pos_embed_table: None,
        }
    }

    /// Attach the host-side learned positional-embedding table (see
    /// [`Self::pos_embed_table`]). `num_grid_per_side` =
    /// `sqrt(num_position_embeddings)`.
    pub fn with_pos_embed_table(mut self, table: Vec<f32>, num_grid_per_side: usize) -> Self {
        self.pos_embed_table = Some((table, num_grid_per_side));
        self
    }
}

impl<W: VisionArchWeights> MultimodalForward for VisionWrapper<W> {
    unsafe fn vision_forward(
        &self,
        pixel_batches: &[PixelInput<'_>],
        placeholders: &[EmbedPatch],
        device: ::scratchy_tensors::ForwardDeviceHandle<'_>,
    ) -> (OwnedTensor, Vec<EmbedPatch>) {
        // Recover the concrete device the neutral trait erased. SAFETY: the MM
        // registry's `try_load_mm` and the worker agree the handle wraps
        // `&mut GpuDevice` (the only type built into a `ForwardDeviceHandle`
        // for the vision-encoder call site).
        let device: &mut crate::GpuDevice = unsafe { device.as_mut() };
        let cfg = self.weights.vision_config();
        let p = cfg.patch_size as usize;
        let t = cfg.temporal_patch_size as usize;
        let c = cfg.in_chans as usize;
        let feat = c * t * p * p;

        let mut all_patches: Vec<u16> = Vec::new();
        let mut grid_thw: Vec<(u32, u32, u32)> = Vec::with_capacity(pixel_batches.len());
        for img in pixel_batches {
            let (mut patch_rows, gthw) = W::pixel_pack(cfg, img.pixels, img.height, img.width);
            all_patches.append(&mut patch_rows);
            grid_thw.push(gthw);
        }
        let total_l: usize = grid_thw
            .iter()
            .map(|&(gt, gh, gw)| (gt as usize) * (gh as usize) * (gw as usize))
            .sum();
        debug_assert_eq!(all_patches.len(), total_l * feat);

        let pixels = device.alloc_gpu_tensor_from_host(
            &[total_l, feat],
            DType::BF16,
            bf16_slice_as_bytes(&all_patches),
        );

        let half_rot = cfg.half_rot();
        // GUARD: the CUDA `vision_rope_apply` kernel hardwires NeoX
        // rotate_half pairing; an interleaved-style tower (MoonViT /
        // LocateAnything) would consume the interleaved cos/sin table
        // with the wrong pairing and produce silently-wrong activations.
        // Only the metal `vision_rope_2d_interleaved` kernel is ported.
        #[cfg(feature = "cuda")]
        assert!(
            cfg.rope_style == scratchy_vision::VisionRopeStyle::NeoxHw,
            "vision rope style {:?} is not ported to the CUDA backend",
            cfg.rope_style,
        );
        let (cos_host, sin_host) = cfg.build_rope_cos_sin_bf16(&grid_thw, total_l);

        let (cu_seqlens_host, max_seqlen) = build_cu_seqlens_i32(&grid_thw);
        let cu_seqlens = device.alloc_gpu_tensor_from_host(
            &[cu_seqlens_host.len()],
            DType::I32,
            i32_slice_as_bytes(&cu_seqlens_host),
        );

        // ── Window-attention dispatch (Qwen2.5-VL family) ──────────
        //
        // For arches whose `windowed_attn_window_size()` returns Some,
        // build the natural→window-grouped permutation host-side,
        // apply it to cos/sin (S² block-grouped permute), upload all
        // four extras (cu_window_seqlens + window_index +
        // reverse_indices + permuted cos/sin). The body's per-layer
        // `varlen_attention(..., cu_seqlens_{full,window}, ...)`
        // calls then read the matching `ForwardCtx` field via the
        // u8 discriminant baked at codegen.
        let window_dispatch = W::windowed_attn_window_size().map(|window_size| {
            scratchy_vision::build_qwen2_5_window_dispatch(&grid_thw, cfg, window_size)
        });

        // cos/sin live on GPU. When windowed, allocate from the
        // permuted host buffers; else from the natural ones. Same
        // shape `[total_l, half_rot]` either way.
        let (cos_to_upload, sin_to_upload);
        let (cos_buf, sin_buf);
        if let Some(wd) = window_dispatch.as_ref() {
            cos_buf = permute_rows_block_grouped_bf16(
                &cos_host,
                total_l,
                half_rot,
                cfg.spatial_merge_size as usize,
                &wd.window_index,
            );
            sin_buf = permute_rows_block_grouped_bf16(
                &sin_host,
                total_l,
                half_rot,
                cfg.spatial_merge_size as usize,
                &wd.window_index,
            );
            cos_to_upload = &cos_buf[..];
            sin_to_upload = &sin_buf[..];
        } else {
            cos_to_upload = &cos_host[..];
            sin_to_upload = &sin_host[..];
        }
        let cos = device.alloc_gpu_tensor_from_host(
            &[total_l, half_rot],
            DType::BF16,
            bf16_slice_as_bytes(cos_to_upload),
        );
        let sin = device.alloc_gpu_tensor_from_host(
            &[total_l, half_rot],
            DType::BF16,
            bf16_slice_as_bytes(sin_to_upload),
        );

        // CUDA consumes the precomputed cos/sin above; the raw f32 `freqs`
        // angle table is a metal-only `vision_rope_2d` input, so it's None here.
        let freqs_buf: Option<GpuTensor> = None;
        let freqs_view = freqs_buf.as_ref().map(|t| unsafe { t.as_view() });

        // ── Qwen3.5-VL learned positional embedding (host interp) ───
        //
        // `fast_pos_embed_interpolate` (4-corner bilinear over a 48×48
        // grid) is far cheaper host-side than as a kernel, so the tower
        // ships the *result* as the `pos_embeds` extern and the DSL just
        // `add(pos_embeds, hidden_states)` after patch_embed. The learned
        // `pos_embed.weight` table is captured host-side at load (the
        // loader doesn't expose it as a runtime weight); when present we
        // interpolate it per forward in spatial-merge token order and
        // upload as bf16. `None` for towers without a learned pos-embed
        // (the DSL then never emits `LoadPosEmbeds`). Validated bit-close
        // to the mlx-vlm golden in the vl crate's green-gate test.
        let pos_embeds_buf: Option<GpuTensor> = self.pos_embed_table.as_ref().map(|(table, ng)| {
            let embed_dim = cfg.embed_dim as usize;
            let pe_f32 = match cfg.pos_emb_interp {
                scratchy_vision::PosEmbInterp::Bilinear => {
                    cfg.fast_pos_embed_interpolate(&grid_thw, *ng, table, total_l)
                }
                scratchy_vision::PosEmbInterp::Bicubic => {
                    cfg.bicubic_pos_embed_interpolate(&grid_thw, *ng, table, total_l)
                }
            };
            let pe_bits: Vec<u16> = pe_f32
                .iter()
                .map(|&x| scratchy_vision::f32_to_bf16(x))
                .collect();
            device.alloc_gpu_tensor_from_host(
                &[total_l, embed_dim],
                DType::BF16,
                bf16_slice_as_bytes(&pe_bits),
            )
        });
        let pos_embeds_view = pos_embeds_buf.as_ref().map(|t| unsafe { t.as_view() });

        // ── Learned positional embeddings (SigLIP / Gemma3-MM) ──────
        //
        // For arches that override `vision_num_positions()` to
        // `Some(N)`, build `[0..N, 0..N, ...]` u32 (one chunk per
        // image) and upload as `vision_position_ids`. The DSL body's
        // `pos_embed(position_ids, weight)` reads this view via
        // `kernels::embedding_gather_masked`. Default `None` arches
        // (Qwen2-VL family) skip the upload — `position_ids_buf`
        // stays None so the GpuTensor isn't allocated.
        //
        // The owned GpuTensor lives in this function's stack frame so
        // the view inside `ctx` doesn't dangle.
        let position_ids_buf = W::vision_num_positions().map(|num_pos| {
            let n = num_pos as usize;
            let mut ids: Vec<u32> = Vec::with_capacity(total_l);
            for _ in 0..pixel_batches.len() {
                ids.extend(0..num_pos);
            }
            debug_assert_eq!(ids.len(), total_l);
            debug_assert_eq!(total_l, pixel_batches.len() * n);
            device.alloc_gpu_tensor_from_host(&[total_l], DType::U32, u32_slice_as_bytes(&ids))
        });
        let position_ids_view = position_ids_buf.as_ref().map(|t| unsafe { t.as_view() });

        let (cu_window_view, window_index_view, reverse_indices_view, max_seqlen_window_opt) =
            if let Some(ref wd) = window_dispatch {
                let l_merged = total_l / (cfg.spatial_merge_size as usize).pow(2);
                let cu_w = device.alloc_gpu_tensor_from_host(
                    &[wd.cu_window_seqlens.len()],
                    DType::I32,
                    i32_slice_as_bytes(&wd.cu_window_seqlens),
                );
                let wi = device.alloc_gpu_tensor_from_host(
                    &[l_merged],
                    DType::U32,
                    u32_slice_as_bytes(&wd.window_index),
                );
                let ri = device.alloc_gpu_tensor_from_host(
                    &[l_merged],
                    DType::U32,
                    u32_slice_as_bytes(&wd.reverse_indices),
                );
                (
                    Some(unsafe { cu_w.as_view() }),
                    Some(unsafe { wi.as_view() }),
                    Some(unsafe { ri.as_view() }),
                    Some(wd.max_seqlen_window),
                )
            } else {
                (None, None, None, None)
            };

        let null_view = unsafe { GpuTensor::null(DType::U32).as_view() };
        let pixels_view = unsafe { pixels.as_view() };
        let cu_view = unsafe { cu_seqlens.as_view() };
        let cos_view = unsafe { cos.as_view() };
        let sin_view = unsafe { sin.as_view() };
        let ctx = ForwardCtx {
            input_ids: null_view,
            positions: null_view,
            slot_mapping: null_view,
            cu_seqlens_q: cu_view,
            seqused_k: null_view,
            block_table: null_view,
            // Vision tower is not paged — uniform (empty group vecs).
            block_tables: Vec::new(),
            slot_mappings: Vec::new(),
            max_seqlen_q: max_seqlen,
            max_seqlen_k: 0,
            kv_cache: &self.kv_placeholder,
            gdn_state: None,
            gdn_state_indices: None,
            gdn_is_fresh: None,
            mm_embeds: None,
            embed_patches: &[],
            vision_rope_cos: Some(cos_view),
            vision_rope_sin: Some(sin_view),
            // METAL: the f32 `freqs` table for `vision_rope_2d` (None on
            // cuda, which uses the cos/sin views above).
            vision_rope_freqs: freqs_view,
            pixels: Some(pixels_view),
            // Qwen3.5-VL host-interpolated learned positional embedding.
            // Computed + uploaded just below for towers that carry a
            // `pos_embed` table; `None` for towers without one (the DSL
            // then never emits `LoadPosEmbeds`).
            pos_embeds: pos_embeds_view,
            // Qwen2.5-VL: `cu_seqlens_full` is the same per-image
            // segmentation as `cu_seqlens_q` above (the window
            // permutation preserves per-image boundaries since
            // `window_index` is built per-image). Reuse the same
            // GpuTensor view; the `cu_seqlens_kind=1` arm reads
            // through this field. `max_seqlen_full` matches.
            vision_cu_seqlens_full: window_dispatch.as_ref().map(|_| cu_view),
            vision_cu_seqlens_window: cu_window_view,
            vision_max_seqlen_full: window_dispatch.as_ref().map(|_| max_seqlen),
            vision_max_seqlen_window: max_seqlen_window_opt,
            vision_window_index: window_index_view,
            vision_reverse_indices: reverse_indices_view,
            vision_position_ids: position_ids_view,
            last_token_indices: None,
            // The vision tape is text-/GDN-free, so the decoder-specific slots
            // are inert (the GDN trio is initialized to None above).
            #[cfg(feature = "nccl")]
            tp_group: None,
        };

        let projected = unsafe { self.weights.vision_forward(&ctx, device, total_l as u64) };

        let sm = cfg.spatial_merge_size;
        debug_assert_eq!(placeholders.len(), grid_thw.len());
        let patches: Vec<EmbedPatch> = placeholders
            .iter()
            .zip(grid_thw.iter())
            .map(|(ph, &(gt, gh, gw))| {
                let mh = gh / sm;
                let mw = gw / sm;
                debug_assert_eq!(ph.length, gt * mh * mw);
                EmbedPatch {
                    token_offset: ph.token_offset,
                    length: ph.length,
                    grid_t: gt,
                    grid_h_merged: mh,
                    grid_w_merged: mw,
                }
            })
            .collect();
        (projected, patches)
    }

    fn mm_metadata(&self) -> &'static scratchy_vision::MmMetadata {
        W::mm_metadata()
    }

    fn embed_patch_grids(&self, pixel_batches: &[PixelInput<'_>]) -> Vec<(u32, u32, u32)> {
        let cfg = self.weights.vision_config();
        let p = cfg.patch_size;
        let sm = cfg.spatial_merge_size;
        pixel_batches
            .iter()
            .map(|img| {
                let grid_h = img.height / p;
                let grid_w = img.width / p;
                (1u32, grid_h / sm, grid_w / sm)
            })
            .collect()
    }
}
