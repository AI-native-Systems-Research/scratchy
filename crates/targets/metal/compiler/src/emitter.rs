// SPDX-License-Identifier: Apache-2.0
//! ⭐ METAL'S EMITTER: what the shared pipeline asks of the metal target, and what it emits for a
//! model and an arch on it.

use proc_macro2::TokenStream;
use quote::quote;
use scratchy_forward_compiler_macro::QuantMethod;
use scratchy_forward_compiler_macro::target::{
    DispatchCx, ExtTrait, LoadCx, ModelCx, ModelEmission, Named, TargetEmitter, ext_trait_imports,
};

/// The relocated `load*` extension traits a metal loader body calls through path syntax
/// (`LinearLayer::load_dense_or_ggml(...)`, `RmsNorm::load(...)`, …), each with the calls it
/// serves. `load_fn` brings into scope only the ones its body calls.
const LOADER_TRAITS: &[ExtTrait] = &[
    ExtTrait {
        path: "crate::__gpu::layers::LinearLayerOps",
        calls: &[
            "LinearLayer::load_dense",
            "LinearLayer::load_raw",
            "LinearLayer::load_dense_or_ggml",
            "LinearLayer::load_dense_concat_packed",
        ],
    },
    ExtTrait {
        path: "crate::__gpu::layers::EmbeddingOps",
        calls: &["Embedding::load"],
    },
    ExtTrait {
        path: "crate::__gpu::layers::RmsNormOps",
        calls: &["RmsNorm::load"],
    },
    ExtTrait {
        path: "crate::__gpu::layers::LayerNormOps",
        calls: &["LayerNorm::load"],
    },
    ExtTrait {
        path: "crate::__gpu::layers::GatedDeltaNetOps",
        calls: &["GatedDeltaNetLayer::load"],
    },
    ExtTrait {
        path: "crate::__gpu::layers::MetalAffineEmbedOps",
        calls: &["AffineQuantEmbedding::load"],
    },
    ExtTrait {
        path: "crate::__gpu::layers::MetalLinearLayerOps",
        calls: &[
            "LinearLayer::load_affine_quant",
            "LinearLayer::load_affine_quant_rows_concat",
            "LinearLayer::load_nvfp4_quant",
            "LinearLayer::load_nvfp4_quant_concat",
            "LinearLayer::load_affine_dequant_as_dense",
            "LinearLayer::load_affine_dequant_concat_as_dense",
        ],
    },
    ExtTrait {
        path: "crate::__gpu::layers_moe::FusedMoEOps",
        calls: &["FusedMoELayer::load", "FusedMoELayer::load_affine"],
    },
    ExtTrait {
        path: "crate::__gpu::layers_moe::SharedFusedMoEOps",
        calls: &[
            "SharedFusedMoELayer::load",
            "SharedFusedMoELayer::load_affine",
        ],
    },
    ExtTrait {
        path: "crate::__gpu::layers_moe::DeepSeekV2MoEOps",
        calls: &["DeepSeekV2MoELayer::load"],
    },
    ExtTrait {
        path: "crate::__gpu::layers_moe::Gemma4RouterOps",
        calls: &["GemmaRouterLayer::load"],
    },
    ExtTrait {
        path: "crate::__gpu::layers_moe::SwitchGluExpertsOps",
        calls: &["SwitchGluExpertsLayer::load"],
    },
    ExtTrait {
        path: "crate::__gpu::layers_moe::GptOssMoEOps",
        calls: &["GptOssMoELayer::load"],
    },
];

/// The metal target's emitter.
pub struct Metal {
    /// TurboQuant KV compression: every model whose geometry the codec takes stores its KV
    /// cache as TurboQuant codes, the rest stay dense and are named in the build output.
    pub turboquant: bool,
}

impl TargetEmitter for Metal {
    fn name(&self) -> &'static str {
        "metal"
    }

    /// ⛔ DETECTION ONLY, AND THAT IS THE WHOLE POINT. Metal runs no solver, so there is no
    /// profile to build from the probe — only the probe, which fails the build when no Metal
    /// device is present. A compile-time profile: it does NOT require Metal 4 (unlike the
    /// runtime `detect_device()`), so models build on non-Metal-4 hosts.
    fn check_host(&self) -> Result<(), String> {
        scratchy_target_metal::device::detect_metal_profile()
            .map(|_| ())
            .ok_or_else(|| {
                "No Metal device detected. Metal backend requires macOS with Apple Silicon."
                    .to_string()
            })
    }

    /// The MLX affine presets reach the tape as dense codes the kernels dequantize, and `nvfp4`
    /// as metal's own E2M1 qmv/qmm; every other preset's kernels (Marlin, Bnb4, Fp8, Ggml) have
    /// no metal realization.
    fn lowers_preset(&self, preset: &str) -> bool {
        preset.starts_with("mlx-affine-") || preset == "nvfp4"
    }

    /// A checked-in quantized config lowers as its preset would: MLX affine or NVFP4.
    fn lowers_quantized_config(&self, method: &QuantMethod) -> bool {
        matches!(
            method,
            QuantMethod::Affine { .. } | QuantMethod::Nvfp4 { .. }
        )
    }

    /// Every affine kernel takes its own width and group, so a per-module override map lowers.
    fn lowers_mixed_bit_affine(&self) -> bool {
        true
    }

    fn compresses_kv(&self) -> bool {
        self.turboquant
    }

    /// The splice copies the tower's embeddings into the image-placeholder rows
    /// (`SpliceMmEmbeds`); a text-only batch makes it a no-op.
    fn splices_mm_embeds(&self) -> bool {
        true
    }

    fn loads_by_accessor(&self) -> bool {
        true
    }

    /// Metal is one device: a concat is CPU-concatenated and allocated once, never sharded.
    fn loads_concat_packed(&self) -> bool {
        true
    }

    fn facade(&self) -> TokenStream {
        quote! { ::scratchy_target_metal }
    }

    /// The baked kernels, each once for every arch that names it (`crate::__metal_bake::K_<key>`).
    fn shared_root_modules(&self) -> &'static [&'static str] {
        &["__metal_bake"]
    }

    fn weights_fields(&self) -> TokenStream {
        quote! {
            /// Lazy-initialized `MetalWorkerPool<Self>` used by
            /// the metal `forward` body. Constructed on the
            /// first call (caller-passed `ctx.kv_cache` provides
            /// the per-layer KV buffers the runtime_factory
            /// closure captures); after that every forward
            /// reuses the same pool. The `OnceLock` lets the
            /// pool live as a field on `Weights` without an
            /// `Arc`-cycle — it borrows `&Weights` for each
            /// `pool.forward(weights, ...)` call instead.
            pub metal_pool: ::std::sync::OnceLock<
                ::scratchy_target_metal::interpreter::metal::MetalWorkerPool<Self>,
            >,
            /// The arena of the model whose forwards follow this one's on its command buffer
            /// (its MTP head), which `metal_pool` holds in its own: set before the pool is
            /// built (`MetalWeights::metal_host`), a fact of this model, not of the device
            /// it shares with other models' pools.
            pub metal_guest: ::std::sync::OnceLock<
                ::scratchy_target_metal::interpreter::metal::GuestArena,
            >,
        }
    }

    fn load_fn(&self, cx: &LoadCx<'_>) -> TokenStream {
        let LoadCx {
            weights_ctor,
            lets,
            field_shorthand,
            packed_splits_prelude,
            rotary_deferred_init,
            rope_context_init,
            ..
        } = cx;
        // Pin the RMSNorm gain dtype to the kernel's `T_scale` (the
        // `_s_<scale>_` arm fixes it from `SCALE_DTYPE`, not the on-disk
        // dtype). `RmsNorm::load`'s `take_as_dtype` then converts the
        // gain only when a checkpoint ships a different dtype (e.g. a
        // standard HF bf16 export feeding an `_s_f16_` Llama arm) — the
        // mlx zero-copy gain path is preserved on the matching case.
        let body = quote! {
            gw.set_rmsnorm_scale_dtype(
                <Weights as ::scratchy_forward_compiler::CanonicalParams>::FACTS.scale_dtype(),
            );
            #packed_splits_prelude
            #(#lets)*
            Ok(#weights_ctor {
                #(#field_shorthand,)*
                #rotary_deferred_init
                #rope_context_init
                metal_pool: ::std::sync::OnceLock::new(),
                metal_guest: ::std::sync::OnceLock::new(),
            })
        };
        // The relocated `load*` extension traits the body's path calls need, and no other.
        let traits = ext_trait_imports(LOADER_TRAITS, &Named::of(&body));
        quote! {
            /// The `stream` is `()` and unused; the body shares the canonical's `lets`
            /// and `field_shorthand` — `LinearConcat` arms route
            /// through `LinearLayer::load_dense_concat_packed`. The
            /// RoPE tables are not built here but with the pool
            /// (`build_rope_tables`), once the KV cache is sized.
            #[allow(clippy::too_many_lines, unused_variables)]
            pub fn load(
                gw: &mut crate::__gpu::weights::GpuWeights,
                stream: crate::__gpu::LoadStream,
                max_model_len: usize,
                tp_rank: u8,
            ) -> ::anyhow::Result<Weights> {
                #traits
                #body
            }
        }
    }

    /// `Weights::build_rope_tables`: the metal pool builds the RoPE tables before it binds them,
    /// once the KV cache is sized, for the positions its block table reaches — the KV cap rung the
    /// pool runs on, picked from `max_model_len` and the blocks the memory budget funds — rather
    /// than at load for every position the checkpoint names. A launch whose KV cache holds less
    /// than `max_position_embeddings` builds only the rows it can serve.
    fn weights_impl(&self, cx: &LoadCx<'_>) -> TokenStream {
        let LoadCx {
            uses_rotary,
            uses_rotary_local,
            rope_rows,
            rotary_load,
            rotary_local_load,
            ..
        } = *cx;
        let set_rotary = uses_rotary.then(|| {
            quote! {
                if self.rotary.set(rotary).is_err() {
                    ::anyhow::bail!("RoPE table built twice");
                }
            }
        });
        let set_rotary_local = uses_rotary_local.then(|| {
            quote! {
                if self.rotary_local.set(rotary_local).is_err() {
                    ::anyhow::bail!("local RoPE table built twice");
                }
            }
        });
        let body = if uses_rotary || uses_rotary_local {
            quote! {
                let reach = ctx.kv_cache.max_blocks_per_seq * ctx.kv_cache.block_size;
                let max_model_len = ::core::cmp::min(self.max_model_len, reach);
                let gw = &mut crate::__gpu::weights::GpuWeights::empty((*device.allocator).clone());
                #rope_rows
                #rotary_load
                #rotary_local_load
                #set_rotary
                #set_rotary_local
            }
        } else {
            quote! { let _ = (ctx, device); }
        };
        quote! {
            impl Weights {
                #[cfg(scratchy_target = "metal")]
                pub fn build_rope_tables(
                    &self,
                    ctx: &crate::__gpu::ForwardCtx,
                    device: &crate::__gpu::GpuDevice,
                ) -> ::anyhow::Result<()> {
                    #body
                    Ok(())
                }
            }
        }
    }

    /// What a shim shares with its canonical: its forward entry points and its baked tables —
    /// variant-specific differences (quant format, fingerprint) are load-time only, and the
    /// static bucket plans are byte-identical.
    fn shim_items(&self, canonical: &syn::Ident) -> TokenStream {
        quote! {
            // `dump` is the per-variant `scr model info` accessor.
            #[cfg(scratchy_target = "metal")]
            pub use super::#canonical::dump;

            #[cfg(scratchy_target = "metal")]
            pub use super::#canonical::{
                forward, forward_chain_with_encoder, forward_onto, forward_with_metal_followup,
                metal_rope_rows,
                METAL_BUCKETS, METAL_OFF_TAPE, METAL_RUNGS,
                metal_pool,
            };
        }
    }

    fn model(&self, cx: &ModelCx<'_>) -> ModelEmission {
        crate::model::emit(cx)
    }

    fn loaded_model(&self, cx: &DispatchCx<'_>) -> TokenStream {
        crate::dispatch::emit(cx)
    }
}
