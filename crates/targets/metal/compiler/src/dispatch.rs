// SPDX-License-Identifier: Apache-2.0
//! METAL'S LOADED-MODEL SURFACE FOR ONE ARCH: the dispatching forward entry points over the arch's
//! `Weights` enum, its `MetalWeights` impl, its metal registry rows, and its `scr model info`
//! dump registration.

use proc_macro2::TokenStream;
use quote::quote;
use scratchy_forward_compiler_macro::target::DispatchCx;
use syn::Ident;

/// One compiled model of the arch: its `Weights` variant, its module, its config stem and its
/// tensor-parallel world.
struct Arm {
    variant_ident: Ident,
    model_ident: Ident,
    source_stem: String,
    tp_world_size: u8,
}

/// The arch's loaded-model surface on metal.
pub(crate) fn emit(cx: &DispatchCx<'_>) -> TokenStream {
    let arms: Vec<Arm> = (cx.variants.iter().zip(cx.models))
        .map(
            |((variant_ident, model_ident), (source_stem, tp_world_size))| Arm {
                variant_ident: variant_ident.clone(),
                model_ident: model_ident.clone(),
                source_stem: source_stem.clone(),
                tp_world_size: *tp_world_size,
            },
        )
        .collect();
    let arch_name_lit = proc_macro2::Literal::string(cx.arch_name);
    // Per-variant `ScratchyWeights::kv_codec()` / `kv_cache_layers()` arms:
    // each reads its own module's `CanonicalParams` const, so the dispatcher
    // keeps no second record of it.
    let canonical_const_arms = |konst: proc_macro2::TokenStream| -> Vec<proc_macro2::TokenStream> {
        arms.iter()
            .map(|a| {
                let variant_ident = a.variant_ident.clone();
                let model_ident = &a.model_ident;
                quote! {
                    Weights::#variant_ident(_) => <#model_ident::Weights as
                        ::scratchy_forward_compiler::CanonicalParams>::#konst,
                }
            })
            .collect()
    };
    let kv_codec_arms = canonical_const_arms(quote! { KV_CODEC });
    let kv_cache_layers_arms = canonical_const_arms(quote! { KV_CACHE_LAYERS });

    let forward_arms: Vec<proc_macro2::TokenStream> = arms
        .iter()
        .map(|a| {
            let variant_ident = a.variant_ident.clone();
            let model_ident = &a.model_ident;
            quote! {
                Weights::#variant_ident(w) => unsafe {
                    #model_ident::forward(w, ctx, device, num_tokens)
                },
            }
        })
        .collect();

    // Per-variant dispatch arms for `forward_with_metal_followup`.
    let forward_with_followup_arms: Vec<proc_macro2::TokenStream> = arms
        .iter()
        .map(|a| {
            let variant_ident = a.variant_ident.clone();
            let model_ident = &a.model_ident;
            quote! {
                Weights::#variant_ident(w) => unsafe {
                    #model_ident::forward_with_metal_followup(w, ctx, device, num_tokens, followup)
                },
            }
        })
        .collect();

    // Per-variant dispatch arms for `forward_chain_with_encoder`.
    // Phase 6 spec-decode K-step chain entry — each canonical's
    // metal_emission emits its own `forward_chain_with_encoder` that
    // wraps `pool.with_chain_encoder` with a per-canonical
    // `ChainStepHandle` adapter.
    let forward_chain_arms: Vec<proc_macro2::TokenStream> = arms
        .iter()
        .map(|a| {
            let variant_ident = a.variant_ident.clone();
            let model_ident = &a.model_ident;
            quote! {
                Weights::#variant_ident(w) => unsafe {
                    #model_ident::forward_chain_with_encoder(w, ctx, device, num_tokens, body)
                },
            }
        })
        .collect();

    let metal_rope_rows_arms: Vec<proc_macro2::TokenStream> = arms
        .iter()
        .map(|a| {
            let variant_ident = a.variant_ident.clone();
            let model_ident = &a.model_ident;
            quote! {
                Weights::#variant_ident(_) => #model_ident::metal_rope_rows(positions, positions.len()),
            }
        })
        .collect();

    let forward_onto_arms: Vec<proc_macro2::TokenStream> = arms
        .iter()
        .map(|a| {
            let variant_ident = a.variant_ident.clone();
            let model_ident = &a.model_ident;
            quote! {
                Weights::#variant_ident(w) => unsafe {
                    #model_ident::forward_onto(w, ctx, device, num_tokens, encoder)
                },
            }
        })
        .collect();

    // Per-variant `CanonicalParams::METAL_DTYPE` reads.
    let metal_dtype_arms: Vec<proc_macro2::TokenStream> = arms
        .iter()
        .map(|a| {
            let variant_ident = a.variant_ident.clone();
            let model_ident = &a.model_ident;
            quote! {
                Weights::#variant_ident(_) => <#model_ident::Weights as
                    ::scratchy_forward_compiler::CanonicalParams>::METAL_DTYPE,
            }
        })
        .collect();

    // Per-variant `METAL_OFF_TAPE` reads: each model's baked argmax / grammar
    // mask / sampler kernels.
    let metal_off_tape_arms: Vec<proc_macro2::TokenStream> = arms
        .iter()
        .map(|a| {
            let variant_ident = a.variant_ident.clone();
            let model_ident = &a.model_ident;
            quote! {
                Weights::#variant_ident(_) => &#model_ident::METAL_OFF_TAPE,
            }
        })
        .collect();

    // Per-variant: what each model's resident pool shares with its guest.
    let metal_hosted_arms: Vec<proc_macro2::TokenStream> = arms
        .iter()
        .map(|a| {
            let variant_ident = a.variant_ident.clone();
            quote! {
                Weights::#variant_ident(w) => w.metal_pool.get().map(|p| {
                    ::std::boxed::Box::new(p.hosted())
                        as ::std::boxed::Box<dyn ::core::any::Any>
                }),
            }
        })
        .collect();

    // Per-variant: make each model's resident pool hold a guest's arena, before it is built.
    let metal_host_arms: Vec<proc_macro2::TokenStream> = arms
        .iter()
        .map(|a| {
            let variant_ident = a.variant_ident.clone();
            quote! {
                Weights::#variant_ident(w) => ::scratchy_target_metal::interpreter::metal::host_guest(
                    &w.metal_pool,
                    &w.metal_guest,
                    guest,
                ),
            }
        })
        .collect();

    // Per-variant `METAL_RUNGS` reads: each model's baked tape rungs.
    let metal_rungs_arms: Vec<proc_macro2::TokenStream> = arms
        .iter()
        .map(|a| {
            let variant_ident = a.variant_ident.clone();
            let model_ident = &a.model_ident;
            quote! {
                Weights::#variant_ident(_) => &#model_ident::METAL_RUNGS,
            }
        })
        .collect();

    // Per-variant `(stem, dump_fn)` rows for the backbone-dump
    // registry. Each variant's `mod <model_ident>` emits a
    // `pub fn dump() -> Vec<BucketDump>`; here we name them so a
    // single `BackboneDumpRegistration` per arch can iterate them.
    // Iterates over EVERY (model, tp) tuple — `scr model info`
    // shows each tp variant separately (the per-tp module's `dump()`
    // reflects the sharded canonical's bucket layout).
    let dump_rows: Vec<proc_macro2::TokenStream> = arms
        .iter()
        .map(|a| {
            let model_ident = &a.model_ident;
            let stem_lit = proc_macro2::Literal::string(&a.source_stem);
            let tp_lit = proc_macro2::Literal::u8_unsuffixed(a.tp_world_size);
            quote! {
                ::scratchy_forward_compiler::VariantDump {
                    variant_stem: #stem_lit,
                    tp_world_size: #tp_lit,
                    buckets: #model_ident::dump(),
                }
            }
        })
        .collect();

    // One `MetalArchRegistration` per tensor-parallel world the arch compiled, each wrapping the
    // shared `ArchRegistration` over `dyn MetalWeights`. The submit! block is a static initializer,
    // walked once at runtime by `try_load(arch_hint, tp_world_size)`.
    let inventory_submits: Vec<proc_macro2::TokenStream> =
        (cx.arch_rows)(quote! { ::scratchy_target_metal::MetalWeights })
            .into_iter()
            .map(|metal_row| {
                quote! {
                    #[cfg(scratchy_target = "metal")]
                    ::scratchy_forward_compiler::inventory::submit! {
                        ::scratchy_target_metal::MetalArchRegistration(#metal_row)
                    }
                }
            })
            .collect();

    quote! {
        /// Dispatching forward. Matches the `Weights` variant and
        /// calls the per-model specialized `forward`. Same body and
        /// signature under both backends — the cfg-mutex'd
        /// `GpuDevice` and `OwnedTensor` re-exports resolve to the
        /// matching backend's struct, and per-canonical `forward`
        /// fns now exist in both `cfg(cuda)` and `cfg(metal)` arms.
        ///
        /// # Safety
        /// All tensors in `ctx` must be valid GPU memory; `device`
        /// must be the live backend device.
        #[cfg(scratchy_target = "metal")]
        #[allow(clippy::too_many_arguments)]
        pub unsafe fn forward(
            w: &Weights,
            ctx: &crate::__gpu::ForwardCtx,
            device: &mut crate::__gpu::GpuDevice,
            num_tokens: u64,
        ) -> crate::__gpu::OwnedTensor {
            match w {
                #(#forward_arms)*
            }
        }

        /// Same as [`forward`] but with an MTL4 encoder-tail hook.
        /// See `MetalForwardFollowup` for semantics.
        ///
        /// # Safety
        /// Same as [`forward`].
        #[cfg(scratchy_target = "metal")]
        #[allow(clippy::too_many_arguments)]
        pub unsafe fn forward_with_metal_followup(
            w: &Weights,
            ctx: &crate::__gpu::ForwardCtx,
            device: &mut crate::__gpu::GpuDevice,
            num_tokens: u64,
            followup: ::core::option::Option<::scratchy_target_metal::model::MetalForwardFollowup<'_>>,
        ) -> ::core::result::Result<crate::__gpu::OwnedTensor, ::scratchy_target_metal::model::GpuExecutionFailed> {
            match w {
                #(#forward_with_followup_arms)*
            }
        }

        /// One forward encoded onto another model's command buffer. See
        /// `::scratchy_forward_compiler::ScratchyWeights::metal_forward_onto`.
        ///
        /// # Safety
        /// Same as [`forward`].
        #[cfg(scratchy_target = "metal")]
        pub unsafe fn forward_onto<'w>(
            w: &'w Weights,
            ctx: &crate::__gpu::ForwardCtx,
            device: &mut crate::__gpu::GpuDevice,
            num_tokens: u64,
            encoder: &::scratchy_target_metal::model::ProtocolObject<
                dyn ::scratchy_target_metal::model::MTL4ComputeCommandEncoder,
            >,
        ) -> ::core::result::Result<
            ::std::boxed::Box<dyn ::scratchy_target_metal::model::MetalForwardOnto + 'w>,
            ::std::string::String,
        > {
            match w {
                #(#forward_onto_arms)*
            }
        }

        /// Phase 6 spec-decode K-step chain dispatcher. One MTL4 CB,
        /// caller drives K bucket forwards + per-iter argmax +
        /// chain_advance from inside `body`. See
        /// `::scratchy_target_metal::model::MetalChainBody` for the body signature.
        ///
        /// # Safety
        /// Same as [`forward`]; additionally, `body` must not retain
        /// any references to the encoder past its return.
        #[cfg(scratchy_target = "metal")]
        #[allow(clippy::too_many_arguments)]
        pub unsafe fn forward_chain_with_encoder(
            w: &Weights,
            ctx: &crate::__gpu::ForwardCtx,
            device: &mut crate::__gpu::GpuDevice,
            num_tokens: u64,
            body: ::scratchy_target_metal::model::MetalChainBody<'_>,
        ) -> ::core::result::Result<(), ::std::string::String> {
            match w {
                #(#forward_chain_arms)*
            }
        }

        #[cfg(scratchy_target = "metal")]
        impl ::scratchy_target_metal::MetalWeights for Weights {
            fn kv_codec(&self) -> ::scratchy_forward_compiler::KvCodec {
                match self {
                    #(#kv_codec_arms)*
                }
            }

            fn kv_cache_layers(&self) -> ::core::option::Option<&'static [u32]> {
                match self {
                    #(#kv_cache_layers_arms)*
                }
            }

            // The trait takes the neutral `ForwardCtxHandle` / `ForwardDeviceHandle`; recover
            // the concrete `&ForwardCtx` / `&mut GpuDevice` here before delegating to the
            // per-arch free fns, which keep concrete types.
            unsafe fn forward(
                &self,
                ctx: crate::__gpu::ForwardCtxHandle<'_>,
                device: crate::__gpu::ForwardDeviceHandle<'_>,
                num_tokens: u64,
            ) -> crate::__gpu::OwnedTensor {
                let ctx: &crate::__gpu::ForwardCtx = unsafe { ctx.as_ref() };
                let device: &mut crate::__gpu::GpuDevice = unsafe { device.as_mut() };
                unsafe { forward(self, ctx, device, num_tokens) }
            }

            unsafe fn forward_backbone(
                &self,
                ctx: crate::__gpu::ForwardCtxHandle<'_>,
                device: crate::__gpu::ForwardDeviceHandle<'_>,
                num_tokens: u64,
            ) -> crate::__gpu::OwnedTensor {
                    {
                    let _ = (ctx, device, num_tokens);
                    unimplemented!(
                        "metal forward_backbone — pipeline-parallel intermediate \
                         ranks aren't supported on metal yet (no PP fanout)"
                    )
                }
            }


            fn metal_dtype(&self) -> ::scratchy_target_metal::interpreter::metal::MetalDtype {
                match self {
                    #(#metal_dtype_arms)*
                }
            }

            fn metal_off_tape(
                &self,
            ) -> &'static (dyn ::core::any::Any + ::core::marker::Send + ::core::marker::Sync) {
                match self {
                    #(#metal_off_tape_arms)*
                }
            }

            fn metal_rungs(
                &self,
            ) -> &'static (dyn ::core::any::Any + ::core::marker::Send + ::core::marker::Sync) {
                match self {
                    #(#metal_rungs_arms)*
                }
            }

            fn metal_hosted(&self) -> ::core::option::Option<::std::boxed::Box<dyn ::core::any::Any>> {
                match self {
                    #(#metal_hosted_arms)*
                }
            }

            fn metal_host(
                &self,
                guest: ::std::boxed::Box<dyn ::core::any::Any>,
            ) -> ::core::result::Result<(), ::std::string::String> {
                match self {
                    #(#metal_host_arms)*
                }
            }

            unsafe fn forward_with_metal_followup(
                &self,
                ctx: crate::__gpu::ForwardCtxHandle<'_>,
                device: crate::__gpu::ForwardDeviceHandle<'_>,
                num_tokens: u64,
                followup: ::core::option::Option<::scratchy_target_metal::model::MetalForwardFollowup<'_>>,
            ) -> ::core::result::Result<crate::__gpu::OwnedTensor, ::scratchy_target_metal::model::GpuExecutionFailed> {
                let ctx: &crate::__gpu::ForwardCtx = unsafe { ctx.as_ref() };
                let device: &mut crate::__gpu::GpuDevice = unsafe { device.as_mut() };
                unsafe { forward_with_metal_followup(self, ctx, device, num_tokens, followup) }
            }

            fn metal_rope_rows(&self, positions: &[u32]) -> ::core::option::Option<::std::vec::Vec<u8>> {
                match self {
                    #(#metal_rope_rows_arms)*
                }
            }

            unsafe fn metal_forward_onto<'w>(
                &'w self,
                ctx: crate::__gpu::ForwardCtxHandle<'_>,
                device: crate::__gpu::ForwardDeviceHandle<'_>,
                num_tokens: u64,
                encoder: &::scratchy_target_metal::model::ProtocolObject<
                    dyn ::scratchy_target_metal::model::MTL4ComputeCommandEncoder,
                >,
            ) -> ::core::result::Result<
                ::std::boxed::Box<dyn ::scratchy_target_metal::model::MetalForwardOnto + 'w>,
                ::std::string::String,
            > {
                let ctx: &crate::__gpu::ForwardCtx = unsafe { ctx.as_ref() };
                let device: &mut crate::__gpu::GpuDevice = unsafe { device.as_mut() };
                unsafe { forward_onto(self, ctx, device, num_tokens, encoder) }
            }

            unsafe fn metal_chain_with_encoder(
                &self,
                ctx: crate::__gpu::ForwardCtxHandle<'_>,
                device: crate::__gpu::ForwardDeviceHandle<'_>,
                num_tokens: u64,
                body: ::scratchy_target_metal::model::MetalChainBody<'_>,
            ) -> ::core::result::Result<(), ::std::string::String> {
                let ctx: &crate::__gpu::ForwardCtx = unsafe { ctx.as_ref() };
                let device: &mut crate::__gpu::GpuDevice = unsafe { device.as_mut() };
                unsafe { forward_chain_with_encoder(self, ctx, device, num_tokens, body) }
            }
        }

        // One target registry row per distinct tp value across
        // the compiled (variant, tp) tuples. Each closure passes its
        // own tp constant into `Weights::load(...)` so the matching
        // arm there walks only the tp-N variants. Only tp = 1 is
        // emitted, so this expands to a single submission with
        // `tp_world_size: 1u8`.
        #(#inventory_submits)*

        // One `BackboneDumpRegistration` per arch, fanning out over
        // every (model, tp) variant via `dump_rows`. Independent of
        // the per-tp registry rows above — `scr model
        // info` walks this registry separately, with no runtime GPU
        // or weight loading. Gated on either backend feature so the
        // metal CLI sees compiled-in metal arches too.
        #[cfg(scratchy_target = "metal")]
        ::scratchy_forward_compiler::inventory::submit! {
            ::scratchy_forward_compiler::BackboneDumpRegistration {
                arch_name: #arch_name_lit,
                dump_all: || vec![ #(#dump_rows),* ],
            }
        }
    }
}
