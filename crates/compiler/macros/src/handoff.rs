// SPDX-License-Identifier: Apache-2.0
//! ⭐ THE NEUTRAL MODEL HANDOFF. [`ModelCx::handoff`] builds the [`TargetModelInput`] a target that
//! lowers from the shared canonical reads: the front end's canonical at every workload point (or
//! its refusal), the checkpoint names of every weight and bundle tensor a canonical binds, and the
//! config facts a target reads beside a canonical. Every value is a fact about the model; a target
//! calls it from its own [`crate::target::TargetEmitter::model`], so a target that never calls it
//! runs none of it.
//!
//! [`TargetModelInput`]: scratchy_subtile::handoff::TargetModelInput

use crate::classified::{OpKind, UnrollIndex, WeightId};
use crate::codegen::{
    BackboneLayout, backbone_layout, bounds_for_wp, lent_weight_prefix, rms_norm_eps,
    safetensors_prefix,
};
use crate::config::ModelParams;
use crate::target::ModelCx;
use crate::workload::WorkloadPoint;

impl ModelCx<'_> {
    /// The model as a target that lowers from the shared canonical takes it, every workload point
    /// lowered at `context`: the KV context the target's tapes hold (its block table's tokens), the
    /// target's own fact.
    pub fn handoff(
        &self,
        context: crate::to_wavefront::PrefixCapacity,
    ) -> scratchy_subtile::handoff::TargetModelInput {
        target_model_input(self, context)
    }
}

/// The model as [`ModelCx::handoff`] gives it: the shared front end's canonical at every
/// workload point (or its refusal), and the config facts a target reads beside a canonical.
/// Target-neutral: every value is a fact about the model.
fn target_model_input(
    cx: &ModelCx<'_>,
    context: crate::to_wavefront::PrefixCapacity,
) -> scratchy_subtile::handoff::TargetModelInput {
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
    use scratchy_subtile::handoff::{
        TargetBucket, TargetGdnFacts, TargetModelFacts, TargetModelInput, TargetPoint,
    };
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
    let prefix = crate::to_wavefront::KvPool::at_context(context);
    for wp in workloads.iter() {
        let bounds = bounds_for_wp(model, *wp, tp_world_size);
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
    let bound = |k: &str| model.bounds.get(k).map(|&v| v as u32);
    let roped = (cx.facts.attention()).is_some_and(|a| a.rope().is_some());
    let head_dim = bound("head_dim").unwrap_or(0);
    let gdn = fuf
        .nodes
        .iter()
        .any(|n| n.op == OpKind::GatedDeltaNet)
        .then(|| {
            let need = |k: &str| {
                bound(k).unwrap_or_else(|| {
                    panic!("[handoff] {stem}: a Gated-DeltaNet model with no `{k}`")
                })
            };
            TargetGdnFacts {
                num_k_heads: need("linear_num_key_heads"),
                head_k: need("linear_key_head_dim"),
                conv_kernel: need("linear_conv_kernel_dim"),
            }
        });
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
    let sliding = (fuf.nodes.iter()).any(|n| {
        matches!(
            n.op,
            OpKind::SlidingAttention | OpKind::SinkSlidingAttention
        )
    });
    let attn_classes = target_attn_classes(model, tp_world_size, head_dim, sliding);
    // The arch is the config's directory (`configs/<arch>/<stem>.json`), the name the build
    // scopes by; `model.arch` is the declared arch spec, not its name.
    let arch = (model.source_path.parent())
        .and_then(|dir| dir.file_name())
        .and_then(|name| name.to_str())
        .unwrap_or_else(|| panic!("[handoff] {stem}: no arch directory"))
        .to_string();
    TargetModelInput {
        stem: stem.clone(),
        arch,
        buckets,
        unlowered,
        weight_prefixes,
        embed_prefix: lent_weight_prefix(program, fuf, model, LentWeight::EmbedTokens),
        bundle_tensor_names,
        rows: match backbone_layout(fuf, program) {
            BackboneLayout::Decoder => scratchy_subtile::sample_rows::ResultRows::Sampled,
            BackboneLayout::Encoder => scratchy_subtile::sample_rows::ResultRows::Every,
        },
        facts: TargetModelFacts {
            head_dim,
            rot_dim: rot_dim(model, head_dim),
            // `None` when the model states no rope: a target with no unroped attention refuses it.
            rope_theta: roped.then(|| model.text.rope_theta.get()),
            rms_norm_eps: rms_norm_eps(model),
            vocab_size: bound("vocab_size").unwrap_or(0),
            gdn,
            dsl_shared_expert_bases: program.weights.shared_expert_bases(),
            attn_classes,
        },
    }
}

/// The model's attention classes, for the handoff: the base class (its rope base
/// frequency `rope_local_base_freq` on a hybrid arch, which rotates its sliding layers by the
/// local table, else `rope_theta`; its window `sliding_window` when the DSL has a sliding
/// attention, `sliding`), then the global class ([`global_attn_class`]) when its geometry is not
/// the base's.
fn target_attn_classes(
    model: &ModelParams,
    tp_world_size: u8,
    head_dim: u32,
    sliding: bool,
) -> Vec<scratchy_subtile::handoff::TargetAttnClass> {
    use scratchy_subtile::handoff::{RopePairing, TargetAttnClass};
    let tp = u32::from(tp_world_size);
    let kv_heads = model
        .bounds
        .get("num_key_value_heads")
        .map_or(0, |&v| v as u32)
        / tp;
    let base_rot = rot_dim(model, head_dim);
    let global = global_attn_class(model, tp, (head_dim, kv_heads, base_rot));
    let theta = Some(model.text.rope_theta.get());
    let local_theta = (model.scalars.get("rope_local_base_freq").copied())
        .or_else(|| model.bounds.get("rope_local_base_freq").map(|&v| v as f64));
    let window = (model.bounds.get("sliding_window"))
        .and_then(|&w| u32::try_from(w).ok())
        .and_then(std::num::NonZeroU32::new);
    let hybrid =
        (global.head_dim, global.kv_heads, global.rot_dim) != (head_dim, kv_heads, base_rot);
    let base = TargetAttnClass {
        head_dim,
        kv_heads,
        rot_dim: base_rot,
        rope_theta: match hybrid {
            true => local_theta.or(theta),
            false => theta,
        },
        pairing: RopePairing::HalfRotated,
        window: window.filter(|_| sliding),
    };
    let mut classes = vec![base];
    if hybrid {
        classes.push(TargetAttnClass {
            head_dim: global.head_dim,
            kv_heads: global.kv_heads,
            rot_dim: global.rot_dim,
            rope_theta: theta,
            pairing: match global.proportional {
                true => RopePairing::HalfHead,
                false => RopePairing::HalfRotated,
            },
            window: None,
        });
    }
    classes
}

/// The global attention class's geometry: a hybrid sliding/global arch's full-attention layers
/// (`GLOBAL_HEAD_DIM`, `NUM_GLOBAL_KV_HEADS`, `GLOBAL_ROT_DIM`, `ROPE_PROPORTIONAL`), the base
/// class's on a uniform model. Read by the handoff's attention classes.
struct GlobalAttnClass {
    pub head_dim: u32,
    pub kv_heads: u32,
    pub rot_dim: u32,
    /// Proportional rope: lane `i` pairs with `i + head_dim/2`, not `i + rot_dim/2`.
    pub proportional: bool,
}

/// [`GlobalAttnClass`] from the config, the base class's `(head_dim, kv_heads, rot_dim)` beside it
/// (`kv_heads` already divided by `tp`).
///
/// Per-layer-class geometry for hybrid sliding/global arches whose classes differ in dims
/// (Gemma4: sliding 256×8kv full-rope vs global 512×1kv proportional-rope-128). Config keys
/// `global_head_dim` / `num_global_key_value_heads` / `global_partial_rotary_factor` feed it;
/// absent keys fall back to the base values (uniform models unchanged).
fn global_attn_class(
    model: &ModelParams,
    tp: u32,
    (head_dim, num_kv_heads, rot_dim_val): (u32, u32, u32),
) -> GlobalAttnClass {
    let global_head_dim: u32 = model
        .bounds
        .get("global_head_dim")
        .map(|&v| v as u32)
        .unwrap_or(head_dim);
    let num_global_kv_heads: u32 = model
        .bounds
        .get("num_global_key_value_heads")
        .map(|&v| (v as u32) / tp)
        .unwrap_or(num_kv_heads);
    let global_rot_dim: u32 = model
        .scalars
        .get("global_partial_rotary_factor")
        .copied()
        .filter(|&f| (f - 1.0).abs() > 1e-9)
        .map(|f| (f * global_head_dim as f64).round() as u32)
        // Fallback must honor the BASE partial factor on uniform
        // models (global_head_dim == head_dim): Qwen3.5's full-
        // attention layers lower through the is_global arm, and the
        // old `global_head_dim` fallback rotated the FULL 256-dim
        // head against the 64-wide partial cos/sin table — rows past
        // the table read zeros and zeroed Q/K for every token beyond
        // ~4 (the "rambling, input-blind thinker" failure). Hybrid
        // arches with distinct global dims and genuinely full global
        // rope (no factor) keep the old fallback.
        .unwrap_or(if global_head_dim == head_dim {
            rot_dim_val
        } else {
            global_head_dim
        });
    // Proportional rope marker — same predicate that routes the
    // rotary cache builder to `new_proportional_from_gpuweights`.
    // Drives the metal rope kernel's pairing offset (lane i pairs
    // with i + head_dim/2, not i + rot_dim/2).
    let proportional = global_rot_dim != global_head_dim
        && model.scalars.contains_key("global_partial_rotary_factor");
    GlobalAttnClass {
        head_dim: global_head_dim,
        kv_heads: num_global_kv_heads,
        rot_dim: global_rot_dim,
        proportional,
    }
}

/// The rotated lanes of a `head_dim`-wide head: `partial_rotary_factor · head_dim` for a
/// partial-rope arch (Qwen3.5: 0.25 · 256 = 64), else the full `head_dim`.
fn rot_dim(model: &ModelParams, head_dim: u32) -> u32 {
    model
        .scalars
        .get("partial_rotary_factor")
        .copied()
        .filter(|&f| (f - 1.0).abs() > 1e-9)
        .map(|f| (f * head_dim as f64).round() as u32)
        .unwrap_or(head_dim)
}
