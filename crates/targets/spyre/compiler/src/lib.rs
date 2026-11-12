// SPDX-License-Identifier: Apache-2.0
//! Spyre's compile-time emission. The shared pipeline (`scratchy-forward-compiler-macro`) calls
//! [`Spyre`] through its `TargetEmitter` seam; the per-model bake — the decode/prefill ladders
//! lowered to KTIR and SuperDSC, the launch wiring, the sharded worlds — is `bake`'s, and the
//! baked programs reach the binary as const tokens (`ktir_tokens`).

mod bake;
mod ktir_tokens;
mod weights;

use proc_macro2::TokenStream;
use quote::quote;
use scratchy_forward_compiler_macro::target::{
    BakeCx, DispatchCx, LoadCx, StructCx, TargetEmitter, WeightsStruct,
};

/// The spyre target's emitter.
pub struct Spyre;

impl TargetEmitter for Spyre {
    fn facade(&self) -> TokenStream {
        // The host target re-exports the neutral surface (`weights` defaulted to
        // `SpyreAllocator`, `LoadStream`, `BackendAllocator`) the emitted load/fingerprint code
        // names through `crate::__gpu::…`.
        quote! { ::scratchy_target_spyre }
    }

    fn weights_struct(&self, cx: &StructCx<'_>) -> Option<WeightsStruct> {
        Some(weights::weights_struct(cx))
    }

    fn load_fn(&self, cx: &LoadCx<'_>) -> TokenStream {
        let LoadCx {
            weights_ctor,
            lets,
            field_shorthand,
            packed_splits_prelude,
            marlin_prelude,
            bnb4_prelude,
            fp8_prelude,
            rope_rows,
            rotary_load,
            rotary_local_load,
            rotary_init,
            rotary_local_init,
            rope_context_init,
            ..
        } = cx;
        quote! {
            /// It builds the real `Weights` through `crate::__gpu`'s loader `*Ops`, which
            /// `scratchy-target-spyre` provides (backed by the neutral allocator-generic
            /// loaders + `SpyreAllocator`). `stream` is `()` (`LoadStream`); rotary loads from
            /// `GpuWeights` (no stream).
            #[allow(clippy::too_many_lines, unused_variables)]
            pub fn load(
                gw: &mut crate::__gpu::weights::GpuWeights,
                stream: crate::__gpu::LoadStream,
                max_model_len: usize,
                tp_rank: u8,
            ) -> ::anyhow::Result<Weights> {
                use crate::__gpu::layers::{
                    LinearLayerOps as _, EmbeddingOps as _, RmsNormOps as _, LayerNormOps as _,
                    GatedDeltaNetOps as _,
                };
                use crate::__gpu::layers_quant::{
                    MarlinLoadOps as _, Fp8LoadOps as _, Fp8BlockLoadOps as _, Bnb4bitLoadOps as _,
                    AffineQuantEmbeddingOps as _,
                };
                use crate::__gpu::layers_moe::{
                    DeepSeekV2MoEOps as _, DeepSeekV2Fp8BlockMoEOps as _,
                    DeepSeekV2GgmlMoEOps as _, FusedMoEOps as _, SharedFusedMoEOps as _,
                    Gemma4RouterOps as _, SwitchGluExpertsOps as _,
                };
                use crate::__gpu::rotary::CudaRotaryExt as _;
                #packed_splits_prelude
                // ⛔ THE QUANT PRELUDES, because the `#lets` below are the shared per-accessor
                // load plans and they REFERENCE these bindings (`__fp8_dtype`, `__marlin_ws`,
                // `__bnb_code`). Omitting them does not disable quant — it emits a load body
                // that names an undefined local. Each prelude elides itself when the model has
                // no accessor of that kind, so a dense model is unaffected.
                #marlin_prelude
                #bnb4_prelude
                #fp8_prelude
                #(#lets)*
                #rope_rows
                #rotary_load
                #rotary_local_load
                Ok(#weights_ctor {
                    #(#field_shorthand,)*
                    #rotary_init
                    #rotary_local_init
                    #rope_context_init
                })
            }
        }
    }

    fn shim_items(&self, canonical: &syn::Ident) -> TokenStream {
        // A shim shares its canonical's solve, so the canonical owns the embedded `KTIR_BUNDLE`
        // AND `SENGRAPH_BUNDLE`. Re-export BOTH so the shim's `ktir_bundle()` AND
        // `sengraph_bundle()` arms resolve symmetrically (re-exporting only `KTIR_BUNDLE` left
        // `--target sendnn` unresolvable for a shim).
        quote! { pub use super::#canonical::{KTIR_BUNDLE, SENGRAPH_BUNDLE}; }
    }

    fn loaded_model(&self, cx: &DispatchCx<'_>) -> TokenStream {
        // One arm per compiled model, each naming its own module's item: the ONE matched variant
        // answers every accessor, so the bundle, the wiring and the weights cannot come from two
        // different variants.
        let arms = |body: &dyn Fn(&syn::Ident) -> TokenStream, binds: bool| -> Vec<TokenStream> {
            cx.variants
                .iter()
                .map(|(variant, model)| {
                    let rhs = body(model);
                    if binds {
                        quote! { Weights::#variant(w) => #rhs, }
                    } else {
                        quote! { Weights::#variant(_) => #rhs, }
                    }
                })
                .collect()
        };
        let any = |item: &str| {
            let item = syn::Ident::new(item, proc_macro2::Span::call_site());
            move |model: &syn::Ident| {
                quote! {
                    &#model::#item
                        as &'static (dyn ::core::any::Any + ::core::marker::Send + ::core::marker::Sync)
                }
            }
        };
        let ktir_bundle_arms = arms(&any("KTIR_BUNDLE"), false);
        let sengraph_bundle_arms = arms(&any("SENGRAPH_BUNDLE"), false);
        let superdsc_wiring_arms = arms(&any("SUPERDSC_WIRINGS"), false);
        let superdsc_tp_arms = arms(&any("SUPERDSC_TP"), false);
        // These arms BIND the inner weights: the point is to reach real fields, which is why
        // this cannot be a lookup table in the worker.
        let superdsc_weight_arms = arms(&|model| quote! { #model::superdsc_weights(w) }, true);
        let superdsc_embed_arms = arms(&|model| quote! { #model::superdsc_embed(w) }, true);
        let rows = (cx.arch_rows)(quote! { ::scratchy_target_spyre::SpyreWeights });
        quote! {
            impl ::scratchy_target_spyre::SpyreWeights for Weights {
                fn ktir_bundle(
                    &self,
                ) -> Option<&'static (dyn ::core::any::Any + ::core::marker::Send + ::core::marker::Sync)>
                {
                    Some(match self {
                        #(#ktir_bundle_arms)*
                    })
                }

                fn superdsc_weights(
                    &self,
                ) -> Option<::anyhow::Result<Vec<::scratchy_target_spyre::BoundWeight>>> {
                    Some(match self {
                        #(#superdsc_weight_arms)*
                    })
                }

                fn superdsc_wiring(
                    &self,
                ) -> Option<&'static (dyn ::core::any::Any + ::core::marker::Send + ::core::marker::Sync)>
                {
                    Some(match self {
                        #(#superdsc_wiring_arms)*
                    })
                }

                fn superdsc_tp(
                    &self,
                ) -> Option<&'static (dyn ::core::any::Any + ::core::marker::Send + ::core::marker::Sync)>
                {
                    Some(match self {
                        #(#superdsc_tp_arms)*
                    })
                }

                fn superdsc_embed(
                    &self,
                ) -> Option<::anyhow::Result<::scratchy_target_spyre::GpuTensor>> {
                    Some(match self {
                        #(#superdsc_embed_arms)*
                    })
                }

                fn sengraph_bundle(
                    &self,
                ) -> Option<&'static (dyn ::core::any::Any + ::core::marker::Send + ::core::marker::Sync)>
                {
                    Some(match self {
                        #(#sengraph_bundle_arms)*
                    })
                }
            }

            #(
                ::scratchy_forward_compiler::inventory::submit! {
                    ::scratchy_target_spyre::SpyreArchRegistration(#rows)
                }
            )*
        }
    }

    fn bake_model(&self, cx: &BakeCx<'_>) -> TokenStream {
        let mut out = TokenStream::new();
        match cx.decode_bounds {
            Some(decode_bounds) => bake::dump_wavefront_mega(
                cx.program,
                cx.model,
                cx.facts,
                cx.fuf,
                cx.inferred,
                decode_bounds,
                &mut out,
                cx.dump_dir,
            ),
            None => eprintln!(
                "[wavefront] {}: no decode (num_tokens=1) workload point",
                cx.model.source_stem
            ),
        }
        out
    }
}
