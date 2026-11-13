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
    /// Each variant's config stem and tensor-parallel world, in `variants`' order.
    pub models: &'a [(String, u8)],
    /// The arch's name, as its registry rows state it.
    pub arch_name: &'a str,
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
    /// Whether the struct holds the RoPE table (`rotary`) and the local one (`rotary_local`).
    pub uses_rotary: bool,
    pub uses_rotary_local: bool,
    /// The rotary tables' rows and loads, built at load or after it.
    pub rope_rows: &'a TokenStream,
    pub rotary_load: &'a TokenStream,
    pub rotary_local_load: &'a TokenStream,
    /// The rotary fields' initializers: built now, or deferred to the pool.
    pub rotary_init: &'a TokenStream,
    pub rotary_local_init: &'a TokenStream,
    pub rotary_deferred_init: &'a TokenStream,
    pub rope_context_init: &'a TokenStream,
}

/// What a target's emission of one canonical model reads: the model, its facts and fused graph,
/// the workload points it compiles, and the shared emission of its weights struct and its
/// `CanonicalParams` impl.
pub struct ModelCx<'a> {
    pub program: &'a Program,
    pub model: &'a ModelParams,
    pub facts: &'a scratchy_ir::model_facts::ModelFacts,
    pub fuf: &'a Fuf,
    /// The `(num_tokens, sk_bucket)` points the model compiles.
    pub workloads: &'a crate::workload::WorkloadPoints,
    pub inferred: &'a Inferred,
    pub tp_world_size: u8,
    /// Where a bake writes its device inputs when the build asked (`superdsc-dump`); `None`
    /// writes nothing.
    pub dump_dir: Option<&'a Path>,
    /// The model's `CanonicalParams` impl (its `FACTS`), for a target whose emitted code reads it.
    pub canonical_params_impl: &'a TokenStream,
    /// The shared struct emission over an accessor set (`None`: no accessors) and the dims its
    /// fields' handles are typed by.
    pub struct_with: &'a dyn Fn(
        Option<&[crate::weight_vocab::WeightAccessor]>,
        &crate::codegen::ModelDims,
    ) -> WeightsStruct,
    /// The dims of an accessor set's fields (`None`: no accessors).
    pub dims_of:
        &'a dyn Fn(Option<&[crate::weight_vocab::WeightAccessor]>) -> crate::codegen::ModelDims,
}

impl ModelCx<'_> {
    /// The shared struct emission over `accs`, typed by their own dims.
    pub fn struct_of(&self, accs: Option<&[crate::weight_vocab::WeightAccessor]>) -> WeightsStruct {
        (self.struct_with)(accs, &(self.dims_of)(accs))
    }
}

/// One canonical model's emission on a target: its weights struct, and the items its module holds
/// beside it.
pub struct ModelEmission {
    pub weights: WeightsStruct,
    pub items: TokenStream,
}

/// A target's emission, called by the shared pipeline.
pub trait TargetEmitter: Sync {
    /// The target's name: the value of the `scratchy_target` cfg its models crate compiles
    /// under, which emitted code gates its target's own items on.
    fn name(&self) -> &'static str;

    /// Refuses a build on a host that cannot compile for the target, before any model expands.
    fn check_host(&self) -> Result<(), String> {
        Ok(())
    }

    /// Whether the target lowers the quant preset `preset` (`quant/<preset>`). A preset it does
    /// not lower is no variant of any model.
    fn lowers_preset(&self, preset: &str) -> bool {
        let _ = preset;
        true
    }

    /// Whether the target lowers a checkpoint whose own config states `method` (a checked-in
    /// quantized base config). A config it does not lower is no model.
    fn lowers_quantized_config(&self, method: &crate::quantization::QuantMethod) -> bool {
        let _ = method;
        true
    }

    /// Whether the target lowers an affine checkpoint whose modules differ in width (a
    /// per-module override map). A target that does not refuses the model at its config.
    fn lowers_mixed_bit_affine(&self) -> bool {
        false
    }

    /// Whether a model's KV cache is TurboQuant codes wherever the codec takes its geometry
    /// (`scratchy_ir::codec_for`), rather than dense.
    fn compresses_kv(&self) -> bool {
        false
    }

    /// Whether a decoder splices a multimodal tower's embeddings into its embedded rows on this
    /// target (`OpKind::MmEmbedSplice`, after the embedding).
    fn splices_mm_embeds(&self) -> bool {
        false
    }

    /// Whether the target loads each weight through its accessor's own host load plan, so a
    /// weight's storage must be one its accessor's type loads. A target whose program loads its
    /// weights itself takes any storage under the dense accessor.
    fn loads_by_accessor(&self) -> bool {
        false
    }

    /// Whether a concatenated weight (a fused QKV or gate/up) loads packed, in one allocation
    /// from the concatenated host rows, rather than through the streamed, shardable load.
    fn loads_concat_packed(&self) -> bool {
        false
    }

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

    /// The target's own items on every canonical `Weights`, over the same load pieces as its
    /// `load` (a target that builds its RoPE tables after load, once its KV cache is sized).
    fn weights_impl(&self, cx: &LoadCx<'_>) -> TokenStream {
        let _ = cx;
        TokenStream::new()
    }

    /// One canonical model's emission: its weights struct and the items beside it. The default
    /// is the struct the target derives ([`TargetEmitter::weights_struct`]) or the shared one, and
    /// the program it bakes at the decode point ([`TargetEmitter::bake_model`]).
    fn model(&self, cx: &ModelCx<'_>) -> ModelEmission {
        let decode_wp = (cx.workloads.iter().find(|w| w.num_tokens == 1))
            .or_else(|| cx.workloads.iter().next())
            .copied();
        let decode_point = decode_wp.map(|wp| {
            (
                crate::codegen::bounds_for_wp(cx.model, wp, cx.tp_world_size),
                wp.num_tokens as u32,
            )
        });
        let struct_of = |accs: Option<&[crate::weight_vocab::WeightAccessor]>| cx.struct_of(accs);
        let weights = self
            .weights_struct(&StructCx {
                program: cx.program,
                model: cx.model,
                facts: cx.facts,
                fuf: cx.fuf,
                inferred: cx.inferred,
                decode_point: decode_point.as_ref().map(|(b, m)| (b, *m)),
                struct_of: &struct_of,
            })
            .unwrap_or_else(|| struct_of(None));
        let decode_bounds = (cx.workloads.iter().find(|w| w.num_tokens == 1))
            .map(|wp| crate::codegen::bounds_for_wp(cx.model, *wp, cx.tp_world_size));
        let items = self.bake_model(&BakeCx {
            program: cx.program,
            model: cx.model,
            facts: cx.facts,
            fuf: cx.fuf,
            inferred: cx.inferred,
            decode_bounds: decode_bounds.as_ref(),
            dump_dir: cx.dump_dir,
        });
        ModelEmission { weights, items }
    }

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

/// A target of every shared default: it lowers every preset and checkpoint config and emits
/// nothing of its own — the target the pipeline's own tests load configs and lower models under.
#[cfg(any(test, feature = "test-support"))]
pub struct Shared;

#[cfg(any(test, feature = "test-support"))]
impl TargetEmitter for Shared {
    fn name(&self) -> &'static str {
        "shared"
    }

    fn load_fn(&self, cx: &LoadCx<'_>) -> TokenStream {
        let _ = cx;
        TokenStream::new()
    }

    fn facade(&self) -> TokenStream {
        quote::quote! { ::scratchy_layers }
    }
}
