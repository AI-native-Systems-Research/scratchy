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
    fn name(&self) -> &'static str {
        "spyre"
    }

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
        match cx.decode_bounds {
            Some(decode_bounds) => {
                // ⭐⭐⭐ THE POOL LAW'S DESCENT — TO THE FIXED POINT. The KV pool is every
                // whole page the funded budget holds beyond mandatory residency, floored at
                // the model's whole context — stated ONCE in `pool_law`, read HERE. The
                // pool is a baked dimension (a KTIR bundle is static-shape), so the law can
                // only be applied by re-baking: pass 1 lowers at the floor (the model's own
                // context) to measure the census, then the law's pages over that pass's own
                // census become the next pass's capacity. Mandatory residency only GROWS
                // with capacity (the walk buckets' mask statics grow with it), so the law's
                // map is monotone decreasing and its fixed point — the largest page count
                // whose own census funds it — is reached by re-baking in BOTH directions:
                // grow when the law funds more than placed, shrink when it funds fewer
                // (an over-grant is a pool its own census does not fund, and it never
                // ships). A 2-cycle between adjacent counts resolves DOWNWARD (the smaller
                // count is funded with margin; an under-grant is a smaller pool, an
                // over-grant is the OOM the load cannot survive). The descent SEALED: the
                // final pass's placement is asserted funded over that pass's own census, at
                // the bake — a build failure, never a load refusal. A pass bound that runs
                // out is a bake bug, and a budget that cannot fund the model's whole
                // context is a REFUSAL here, at the bake. (The census is the card build's
                // alone: off-card it comes back `None` and the floor stands, exactly as
                // those builds have always sized it.)
                use ktir_core::ddl_binds::{Baked, DeviceGen};
                use scratchy_target_spyre::pool_law;
                let stem = cx.model.source_stem.as_str();
                // The funded budget the law sizes from: the platform table's card bytes ×
                // the utilization default. A card that reports less than the table is
                // caught at load, by the same law over the measured card.
                let funded = pool_law::funded_bytes(
                    <Baked as DeviceGen>::MEMORY_BYTES,
                    pool_law::GpuMemoryUtilization::DEFAULT,
                );
                let page_slots: u32 =
                    scratchy_target_spyre::paged_step::SchedulerBlockTokens::REFERENCE.get();
                let mut capacity: Option<std::num::NonZeroU32> = None;
                let mut out = TokenStream::new();
                // The page counts this descent has baked at — a repeat target is a cycle of
                // the monotone law, resolved DOWNWARD below. The final pass's census and
                // placed pages are carried for the seal after the loop.
                let mut visited: Vec<u32> = Vec::new();
                let mut final_census: Option<bake::PoolCensus> = None;
                let mut final_pages: u32 = 0;
                for pass in 1..=8u32 {
                    let mut stream = TokenStream::new();
                    let census = bake::dump_wavefront_mega(
                        cx.program,
                        cx.model,
                        cx.facts,
                        cx.fuf,
                        cx.inferred,
                        decode_bounds,
                        capacity,
                        &mut stream,
                        cx.dump_dir,
                    );
                    out = stream;
                    let Some(census) = census else {
                        break;
                    };
                    // The law over every baked world — the pool a binary serves is one pool,
                    // so the smallest world's pages bound it. The terms are the census's own
                    // (minted once at the bake): the residency it carried, the stride the
                    // KTIR lowering derived, the floor in whole pages.
                    let mut pages: Option<pool_law::Pages> = None;
                    for w in &census.worlds {
                        let p = pool_law::pool_pages(
                            funded,
                            w.residency.mandatory_bytes(),
                            pool_law::PageStride::baked(w.page_stride),
                            pool_law::Pages::new(census.context_pages),
                        )
                        .unwrap_or_else(|| {
                            panic!(
                                "[spyre] {stem}: the funded budget ({} MB at utilization {:.2}) \
                                 cannot hold the model's own context ({} pages) beside this \
                                 bake's {} MB of mandatory residency — a bake that cannot fund \
                                 the model's whole context refuses",
                                funded.get() / (1024 * 1024),
                                pool_law::GpuMemoryUtilization::DEFAULT.get(),
                                census.context_pages,
                                w.residency.mandatory_bytes().get() / (1024 * 1024),
                            )
                        });
                        pages = Some(pages.map_or(p, |m: pool_law::Pages| m.min(p)));
                    }
                    let pages = pages.expect("a census always names at least one world");
                    let current_pages = capacity
                        .map(|c| c.get() / page_slots)
                        .unwrap_or(census.context_pages);
                    visited.push(current_pages);
                    final_census = Some(census);
                    final_pages = current_pages;
                    // ⭐ THE FIXED POINT, EXACTLY. The ONLY clean exit is equality: the
                    // pass's own census funds exactly the pages that pass placed. A law
                    // that funds FEWER pages than the pass placed is an OVER-GRANT — a
                    // pool its own census does not fund — and the descent re-bakes
                    // SMALLER, never ships it. (The old `pages <= current` break shipped
                    // exactly that: a pool sized by the previous pass's census beside
                    // this pass's larger mandatory residency.)
                    if pages.get() == current_pages {
                        break;
                    }
                    // The cycle of a monotone map: the target is a count this descent has
                    // already baked, so equality will not terminate it. Resolve DOWNWARD:
                    // the SMALLEST count visited is funded by its own census with margin —
                    // it was entered by SHRINK (the monotone law funds everything the
                    // shrunk-from pass funded) or it is the floor, which the law never
                    // undercuts. An under-grant is a smaller pool; an over-grant is the
                    // OOM the load cannot survive.
                    if visited.contains(&pages.get()) {
                        let settle = visited.iter().copied().min().unwrap();
                        if settle == current_pages {
                            break;
                        }
                        let positions: u32 = (u64::from(settle) * u64::from(page_slots))
                            .try_into()
                            .unwrap_or_else(|_| {
                                panic!(
                                    "[spyre] {stem}: the law's pool ({settle} pages × \
                                     {page_slots} slots) exceeds u32 positions"
                                )
                            });
                        eprintln!(
                            "[spyre] {stem}: the capacity law's descent cycles — settling \
                             DOWN at {settle} page(s), funded by its own census with margin \
                             (pass {pass})"
                        );
                        capacity = Some(
                            std::num::NonZeroU32::new(positions)
                                .expect("the law's pages are nonzero pages of nonzero slots"),
                        );
                        continue;
                    }
                    if pass == 8 {
                        panic!(
                            "[spyre] {stem}: the pool descent did not converge in 8 passes — \
                             the law still moves past {} page(s)",
                            pages.get()
                        );
                    }
                    let positions: u32 = (u64::from(pages.get()) * u64::from(page_slots))
                        .try_into()
                        .unwrap_or_else(|_| {
                            panic!(
                                "[spyre] {stem}: the law's pool ({} pages × {page_slots} \
                                 slots) exceeds u32 positions",
                                pages.get()
                            )
                        });
                    eprintln!(
                        "[spyre] {stem}: the capacity law re-sizes the KV pool — pass {pass}: \
                         {} page(s) over the {} placed (the model's own context is {}), \
                         mandatory residency {} MB against the {} MB the card funds; \
                         re-baking at {positions} positions",
                        pages.get(),
                        current_pages,
                        final_census.as_ref().map(|c| c.context_pages).unwrap_or(0),
                        final_census
                            .as_ref()
                            .map(|c| {
                                c.worlds
                                    .iter()
                                    .map(|w| w.residency.mandatory_bytes().get() / (1024 * 1024))
                                    .min()
                                    .unwrap_or(0)
                            })
                            .unwrap_or(0),
                        funded.get() / (1024 * 1024),
                    );
                    capacity = Some(
                        std::num::NonZeroU32::new(positions)
                            .expect("the law's pages are nonzero pages of nonzero slots"),
                    );
                }
                // ⭐⭐ THE SEAL — the bake's own proof over the FINAL census it minted: the
                // pool the last pass placed is funded by that pass's own terms, through the
                // same ONE combination the load checks over the measured card. With the
                // fixed point this holds by construction; the assert turns a descent bug
                // into a BUILD failure — never a load refusal over a fact the bake knew.
                if let Some(census) = final_census {
                    for w in &census.worlds {
                        assert!(
                            pool_law::funded_holds(
                                funded,
                                w.residency.mandatory_bytes(),
                                pool_law::PageStride::baked(w.page_stride),
                                pool_law::Pages::new(final_pages),
                            ),
                            "[spyre] {stem}: the descent shipped a pool its own census does \
                             not fund — {} page(s) placed beside {} MB of mandatory residency \
                             against the {} MB the card funds: a bake bug, not a card fact",
                            final_pages,
                            w.residency.mandatory_bytes().get() / (1024 * 1024),
                            funded.get() / (1024 * 1024),
                        );
                    }
                }
                out
            }
            None => {
                eprintln!(
                    "[wavefront] {}: no decode (num_tokens=1) workload point",
                    cx.model.source_stem
                );
                TokenStream::new()
            }
        }
    }
}
