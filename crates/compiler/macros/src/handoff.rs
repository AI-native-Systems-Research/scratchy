// SPDX-License-Identifier: Apache-2.0
//! ⭐ THE NEUTRAL MODEL HANDOFF. [`ModelCx::handoff`] builds the [`TargetModelInput`] a target that
//! lowers from the shared canonical reads: the front end's canonical at every workload point (or
//! its refusal), the checkpoint names of every weight and bundle tensor a canonical binds, and the
//! config facts a target reads beside a canonical. Every value is a fact about the model; a target
//! calls it from its own [`crate::target::TargetEmitter::model`], so a target that never calls it
//! runs none of it.
//!
//! [`TargetModelInput`]: scratchy_subtile::handoff::TargetModelInput

use crate::classified::{UnrollIndex, WeightId};
use crate::codegen::{
    BackboneLayout, backbone_layout, bounds_for_wp, lent_weight_prefix, safetensors_prefix,
};
use crate::target::ModelCx;
use crate::workload::WorkloadPoint;

impl ModelCx<'_> {
    /// The model as a target that lowers from the shared canonical takes it.
    pub fn handoff(&self) -> scratchy_subtile::handoff::TargetModelInput {
        target_model_input(self)
    }
}

/// The model as [`ModelCx::handoff`] gives it: the shared front end's canonical at every
/// workload point (or its refusal), and the config facts a target reads beside a canonical.
/// Target-neutral: every value is a fact about the model.
fn target_model_input(cx: &ModelCx<'_>) -> scratchy_subtile::handoff::TargetModelInput {
    let (program, fuf, inferred, workloads, tp_world_size) = (
        cx.program,
        cx.fuf,
        cx.inferred,
        cx.workloads,
        cx.tp_world_size,
    );
    let bridge_model = &crate::to_wavefront::BridgeModel {
        params: cx.model,
        facts: cx.facts,
    };
    use scratchy_subtile::handoff::{TargetBucket, TargetModelInput, TargetPoint};
    let model = bridge_model.params;
    let stem = &model.source_stem;
    let point = |wp: &WorkloadPoint| {
        let fit = |what: &str, v: u64| {
            u32::try_from(v).unwrap_or_else(|_| panic!("[handoff] {stem}: {what} {v}"))
        };
        TargetPoint {
            m: fit("num_tokens", wp.num_tokens),
            sk: fit("sk_bucket", wp.sk_bucket),
        }
    };
    let mut buckets = Vec::new();
    let mut unlowered = Vec::new();
    for wp in workloads.iter() {
        let bounds = bounds_for_wp(model, *wp, tp_world_size);
        let prefix =
            crate::to_wavefront::PrefixCapacity::new(std::num::NonZeroU32::new(8192).unwrap());
        let at = point(wp);
        match crate::to_wavefront::lower_decode_to_wavefront(
            fuf,
            inferred,
            &bounds,
            bridge_model,
            prefix,
            at.m,
            crate::to_wavefront::DenseExperts::Lower,
        ) {
            Ok(lowered) => buckets.push(TargetBucket { point: at, lowered }),
            Err(e) => unlowered.push((at, e.to_string())),
        }
    }
    let roped = (cx.facts.attention()).is_some_and(|a| a.rope().is_some());
    // Every weight a canonical binds, named as the checkpoint names it. A tied lm_head is lent the
    // embedding table's prefix, as every target's loader reads it.
    use scratchy_forward_compiler::LentWeight;
    let lm_head = program.weights.find(&["lm_head"]);
    let weights: std::collections::BTreeSet<(u32, Option<UnrollIndex>)> = (buckets.iter())
        .flat_map(|b| b.lowered.bindings().iter())
        .filter_map(|b| match *b {
            scratchy_subtile::handoff::SourceBinding::Weight { id, index } => Some((id, index)),
            _ => None,
        })
        .collect();
    let weight_prefixes = (weights.into_iter())
        .map(|(id, index)| {
            let own =
                || safetensors_prefix(program.weight_naming(), &model.arch, WeightId(id), index);
            let prefix = match (lm_head, index) {
                (Some(head), None) if head == WeightId(id) => {
                    lent_weight_prefix(program, fuf, model, LentWeight::LmHead).unwrap_or_else(own)
                }
                _ => own(),
            };
            (id, index, prefix)
        })
        .collect();
    // Every bundle tensor a canonical binds that has a checkpoint leaf, named in full: the
    // bundle's own prefix, its suffix rewritten as the shared table says, then the leaf.
    let bundle_tensors: std::collections::BTreeMap<_, _> = (buckets.iter())
        .flat_map(|b| b.lowered.bindings().iter())
        .filter_map(|b| match *b {
            scratchy_subtile::handoff::SourceBinding::BundleTensor { id, index, tensor } => {
                Some(((id, index, tensor.name()), tensor))
            }
            _ => None,
        })
        .collect();
    let bundle_tensor_names = (bundle_tensors.into_iter())
        .filter_map(|((id, index, _), tensor)| {
            use scratchy_subtile::handoff::{BUNDLE_PREFIX_REWRITES, BUNDLE_TENSOR_LEAVES};
            let leaf =
                (BUNDLE_TENSOR_LEAVES.iter()).find_map(|&(t, l)| (t == tensor).then_some(l))?;
            let prefix =
                safetensors_prefix(program.weight_naming(), &model.arch, WeightId(id), index);
            let prefix = (BUNDLE_PREFIX_REWRITES.iter())
                .find_map(|&(from, to)| {
                    let base = prefix.strip_suffix(from)?;
                    (base.is_empty() || base.ends_with('.')).then(|| format!("{base}{to}"))
                })
                .unwrap_or(prefix);
            Some((id, index, tensor, format!("{prefix}.{leaf}")))
        })
        .collect();
    TargetModelInput {
        stem: stem.clone(),
        buckets,
        unlowered,
        weight_prefixes,
        embed_prefix: lent_weight_prefix(program, fuf, model, LentWeight::EmbedTokens),
        bundle_tensor_names,
        rows: match backbone_layout(fuf, program) {
            BackboneLayout::Decoder => scratchy_subtile::sample_rows::ResultRows::Sampled,
            BackboneLayout::Encoder => scratchy_subtile::sample_rows::ResultRows::Every,
        },
        facts: *cx.facts,
        // `None` when the model states no rope: a target with no unroped attention refuses it.
        rope_theta: roped.then(|| model.text.rope_theta.get()),
        // A hybrid arch's sliding class rotates by its own base frequency (the local table).
        rope_local_theta: roped
            .then(|| {
                (model.scalars.get("rope_local_base_freq").copied())
                    .or_else(|| model.bounds.get("rope_local_base_freq").map(|&v| v as f64))
            })
            .flatten(),
        dsl_shared_expert_bases: program.weights.shared_expert_bases(),
    }
}
