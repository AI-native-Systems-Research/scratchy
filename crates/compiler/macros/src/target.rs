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

/// A target's emission, called by the shared pipeline.
pub trait TargetEmitter: Sync {
    /// The items a target bakes for one model at its decode point — spliced into the model's
    /// module. Empty for a target whose program is emitted elsewhere in the pipeline.
    fn bake_model(&self, cx: &BakeCx<'_>) -> TokenStream {
        let _ = cx;
        TokenStream::new()
    }
}

/// The metal target's emitter, while its emission still lives in this crate's codegen.
pub struct Metal;

impl TargetEmitter for Metal {}
