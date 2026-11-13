// SPDX-License-Identifier: Apache-2.0
//! ⭐ THE TARGET SEAM. The pipeline is shared; what a target emits for a model is the target's.
//! A target's compiler crate (`crates/targets/<target>/compiler`) implements [`TargetEmitter`],
//! and the build that compiles the models hands it to [`crate::compile_carrier`], which calls it
//! at each point a model's emission is the target's own.

use std::collections::BTreeMap;
use std::path::Path;

use proc_macro2::TokenStream;

use crate::classified::Program;
use crate::config::ModelParams;
use crate::fuf::Fuf;
use crate::shape::Inferred;

/// What a target's model bake reads: the model's program, facts and fused graph, and the bounds
/// at its decode (num_tokens = 1) point.
pub struct BakeCx<'a> {
    pub program: &'a Program,
    pub model: &'a ModelParams,
    /// The model's facts (#1083): the one source of the attention geometry and scale a bake
    /// lowers by.
    pub facts: &'a scratchy_ir::model_facts::ModelFacts,
    pub fuf: &'a Fuf,
    pub inferred: &'a Inferred,
    /// The decode point's bounds; `None` when the model compiles no num_tokens = 1 point.
    pub decode_bounds: Option<&'a BTreeMap<String, u64>>,
    /// Where a bake writes its device inputs when the build asked (`superdsc-dump`); `None`
    /// writes nothing.
    pub dump_dir: Option<&'a Path>,
}

/// What a target's loaded-model surface for one arch reads: the arch's compiled models, and its
/// shared registry rows.
pub struct DispatchCx<'a> {
    /// `(variant, model module)` per compiled model of the arch: `Weights::<variant>` holds the
    /// `<model module>::Weights` of that model.
    pub variants: &'a [(syn::Ident, syn::Ident)],
    /// The arch's shared registry rows (`::scratchy_forward_compiler::ArchRegistration`), one per
    /// tensor-parallel world it compiled, whose loaders box the loaded model as
    /// `dyn <weights_trait>` — the target's loaded-model trait, which `Weights` implements.
    pub arch_rows: &'a dyn Fn(TokenStream) -> Vec<TokenStream>,
}

/// A model's weights struct with the affine rows its fingerprint gate carries.
pub type WeightsStruct = (TokenStream, Vec<(String, u32, u32)>);

/// What a target that derives a model's weights struct itself reads.
pub struct StructCx<'a> {
    pub program: &'a Program,
    pub model: &'a ModelParams,
    pub facts: &'a scratchy_ir::model_facts::ModelFacts,
    pub fuf: &'a Fuf,
    pub inferred: &'a Inferred,
    /// The point the struct is derived at — the decode (num_tokens = 1) point, else the first —
    /// as its bounds and its token count. `None` when the model compiles no point.
    pub decode_point: Option<(&'a BTreeMap<String, u64>, u32)>,
    /// The shared struct emission over an accessor set (`None`: no accessors).
    pub struct_of: &'a dyn Fn(Option<&[crate::weight_vocab::WeightAccessor]>) -> WeightsStruct,
}

/// What a target's canonical `Weights::load` is made of: the shared per-accessor load plans and
/// the pieces of the struct the pipeline derived, which each target assembles behind its own
/// loader traits.
pub struct LoadCx<'a> {
    /// The constructor the body ends in (`Weights`).
    pub weights_ctor: &'a TokenStream,
    /// One `let` per weight field, each the field's load plan.
    pub lets: &'a [TokenStream],
    /// The fields the `lets` bound, in struct order.
    pub field_shorthand: &'a [syn::Ident],
    /// The bindings the `lets` read: packed-split tables, and the quant preludes (each empty
    /// when the model has no accessor of its kind).
    pub packed_splits_prelude: &'a TokenStream,
    pub marlin_prelude: &'a TokenStream,
    pub bnb4_prelude: &'a TokenStream,
    pub fp8_prelude: &'a TokenStream,
    /// The rotary tables' rows and loads, for a target that builds them at load.
    pub rope_rows: &'a TokenStream,
    pub rotary_load: &'a TokenStream,
    pub rotary_local_load: &'a TokenStream,
    /// The rotary fields' initializers: built now, or deferred to the pool.
    pub rotary_init: &'a TokenStream,
    pub rotary_local_init: &'a TokenStream,
    pub rotary_deferred_init: &'a TokenStream,
    pub rope_context_init: &'a TokenStream,
}

/// A target's emission, called by the shared pipeline.
pub trait TargetEmitter: Sync {
    /// The target's name: the value of the `scratchy_target` cfg its models crate compiles
    /// under, which emitted code gates its target's own items on.
    fn name(&self) -> &'static str;

    /// A model's weights struct, for a target that derives it itself; `None` takes the shared
    /// emission.
    fn weights_struct(&self, cx: &StructCx<'_>) -> Option<WeightsStruct> {
        let _ = cx;
        None
    }

    /// The target's own fields on every canonical `Weights` struct, after the weights.
    fn weights_fields(&self) -> TokenStream {
        TokenStream::new()
    }

    /// The canonical `Weights::load(gw, stream, max_model_len, tp_rank)`.
    fn load_fn(&self, cx: &LoadCx<'_>) -> TokenStream;

    /// What a shim module (a quant variant sharing `canonical`'s solve) re-exports of the
    /// canonical's baked items, so the loaded-model arms naming them resolve in the shim too.
    fn shim_items(&self, canonical: &syn::Ident) -> TokenStream {
        let _ = canonical;
        TokenStream::new()
    }

    /// The target's runtime crate: the emitted code names the target's types (the weight store,
    /// its layers and loaders) through it, as `crate::__gpu`.
    fn facade(&self) -> TokenStream;

    /// The modules at the models crate's root that every arch's emitted code names
    /// (`crate::<module>`): items the target bakes once for every arch, which the models build
    /// leaves at the root rather than re-rooting under the arch's module.
    fn shared_root_modules(&self) -> &'static [&'static str] {
        &[]
    }

    /// The items a target bakes for one model at its decode point — spliced into the model's
    /// module. Empty for a target whose program is emitted elsewhere in the pipeline.
    fn bake_model(&self, cx: &BakeCx<'_>) -> TokenStream {
        let _ = cx;
        TokenStream::new()
    }

    /// An arch's loaded-model surface on this target: the `impl` of the target's loaded-model
    /// trait for the arch's `Weights` enum, and the submission of its registry rows.
    fn loaded_model(&self, cx: &DispatchCx<'_>) -> TokenStream {
        let _ = cx;
        TokenStream::new()
    }
}

/// The metal target's emitter, while its emission still lives in this crate's codegen.
pub struct Metal;

impl TargetEmitter for Metal {
    fn name(&self) -> &'static str {
        "metal"
    }

    fn facade(&self) -> TokenStream {
        quote::quote! { ::scratchy_target_metal }
    }

    /// The baked kernels, each once for every arch that names it (`crate::__metal_bake::K_<key>`).
    fn shared_root_modules(&self) -> &'static [&'static str] {
        &["__metal_bake"]
    }

    fn weights_fields(&self) -> TokenStream {
        quote::quote! {
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
        quote::quote! {
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
                // Bring the relocated `load*` extension traits into scope so
                // the path-syntax `LinearLayer::load_dense_or_ggml(...)`,
                // `RmsNorm::load(...)`, `Embedding::load(...)`,
                // `AffineQuantEmbedding::load(...)`, … calls the body
                // emits below resolve.
                use crate::__gpu::layers::{
                    LinearLayerOps as _, EmbeddingOps as _, RmsNormOps as _, LayerNormOps as _,
                    GatedDeltaNetOps as _, MetalAffineEmbedOps as _, MetalLinearLayerOps as _,
                };
                use crate::__gpu::layers_moe::{
                    FusedMoEOps as _, SharedFusedMoEOps as _, DeepSeekV2MoEOps as _,
                    Gemma4RouterOps as _, SwitchGluExpertsOps as _,
                    GptOssMoEOps as _,
                };
                // Pin the RMSNorm gain dtype to the kernel's `T_scale` (the
                // `_s_<scale>_` arm fixes it from `SCALE_DTYPE`, not the on-disk
                // dtype). `RmsNorm::load`'s `take_as_dtype` then converts the
                // gain only when a checkpoint ships a different dtype (e.g. a
                // standard HF bf16 export feeding an `_s_f16_` Llama arm) — the
                // mlx zero-copy gain path is preserved on the matching case.
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
            }
        }
    }
}
