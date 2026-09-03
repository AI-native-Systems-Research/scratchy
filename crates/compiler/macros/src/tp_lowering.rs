// SPDX-License-Identifier: Apache-2.0
//! Tensor-parallel FUF-lowering pass.
//!
//! The DSL is shape-agnostic: every `#[forward]` body describes the
//! single-rank computation. Sharding and the collective primitives
//! that follow it are injected here, after `fuf::unroll` constructs
//! the FUF and before the solver runs. At `tp_world_size = 1` the
//! pass is a strict no-op — the FUF is byte-identical to the input
//! and the rest of the pipeline behaves exactly as in single-rank
//! builds.
//!
//! At `tp_world_size > 1` (task #5c) the pass walks every `Gemm`
//! node, reads the per-arch shard-kind table to decide whether the
//! gemm's weight is row- or column-parallel, and inserts a fresh
//! `OpKind::AllReduce` FufNode after each row-parallel gemm whose
//! output flows into the residual stream. Coloring later collapses
//! the AllReduce dst slot onto the gemm output's slot (validated by
//! `coloring_allreduce_collapses_to_input_slot`); the fused-pair
//! comm-boundary guard falls out of the FUF dataflow break
//! (validated by `cutlass_gemm_add_does_not_claim_across_intermediate
//! _node`).

use crate::classified::{OpKind, Program};
use crate::fuf::{Fuf, FufInput, FufNode, TileId};
use crate::shape::Dim;

/// How the loader will slice a tensor across `tp_world_size` ranks.
///
/// `ShardDim0` is column-parallel — output dim is split, each rank
/// owns a contiguous chunk along the first axis. The downstream
/// activation flow is naturally per-rank (silu / mul / next gemm
/// operate on the local shard); no all-reduce needed.
///
/// `ShardDim1` is row-parallel — input dim is split, each rank owns
/// a partial-sum gemm output. An all-reduce-sum across ranks is
/// required before the result re-enters the residual stream — the
/// load-bearing reason this lowering pass exists.
///
/// `Replicate` mirrors the full tensor on every rank — embeddings,
/// `lm_head`, and norm weights stay replicated. No sharding, no
/// communication.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ShardKind {
    ShardDim0,
    ShardDim1,
    Replicate,
}

/// Generic name-pattern matcher matching Python vLLM's standard
/// parallel layer types:
///
/// - `q_proj | k_proj | v_proj | gate_proj | up_proj` → `ShardDim0`
///   (column-parallel — the per-head and gate/up projections split
///   along the output dim so each rank holds a contiguous slice of
///   heads / intermediate channels). Mirrors Python vLLM's
///   `ColumnParallelLinear`.
/// - `o_proj | down_proj` → `ShardDim1` (row-parallel — the
///   collapse projections split along the input dim so each rank's
///   gemm produces a partial sum, requiring all-reduce). Mirrors
///   `RowParallelLinear`.
/// - `embed_tokens | lm_head` → `ShardDim0` (vocab-parallel — the
///   vocab dim is sharded across ranks). Mirrors Python vLLM's
///   `VocabParallelEmbedding` and `ParallelLMHead` (which inherits
///   from VocabParallelEmbedding so both share the same dim-0
///   shard layout — that's how `tie_weights` stays consistent at
///   tp>1, since both layers point at the same sharded tensor).
///   Embed forward needs masking + AllReduce; lm_head output needs
///   AllGather before sampler. Both are wired by `tp_lowering`.
/// - Everything else → `Replicate` (norms, rotary tables,
///   biases on column-parallel layers — Python's
///   `ColumnParallelLinear.bias` shards too, but we treat bias as
///   following its parent weight at the codegen layer; the matcher
///   here is for the weight name itself, not the .bias suffix).
///
/// Matches on the LAST segment of the dotted weight path the
/// `WeightTable` stores. The DSL doesn't ship `.weight` /
/// `.qweight` suffixes in WeightTable paths — those are runtime
/// safetensors-key concerns. So `path == ["self_attn", "o_proj"]`
/// → last segment `o_proj` → `ShardDim1`.
///
/// Per-arch overrides can be layered on later (e.g. an arch with
/// non-standard names), but every arch in the current scratchy tree
/// uses the standard HF naming. Keeping the matcher generic avoids
/// per-arch boilerplate at this stage.
pub(crate) fn shard_kind_for_weight_path(path: &[String]) -> ShardKind {
    let Some(last) = path.last() else {
        return ShardKind::Replicate;
    };
    shard_kind_for_last_segment(last.as_str())
}

/// Shard-kind lookup keyed on the bare last segment of a weight
/// path, e.g. `"o_proj"`, `"embed_tokens"`. Used by codegen, which
/// already has the unstructured prefix string and only needs the
/// terminal name.
pub(crate) fn shard_kind_for_last_segment(last: &str) -> ShardKind {
    match last {
        "q_proj" | "k_proj" | "v_proj" | "gate_proj" | "up_proj" => ShardKind::ShardDim0,
        "o_proj" | "down_proj" => ShardKind::ShardDim1,
        "embed_tokens" | "lm_head" => ShardKind::ShardDim0,
        _ => ShardKind::Replicate,
    }
}

/// Convenience for codegen: given a dotted prefix like
/// `"model.embed_tokens"` or `"model.layers.0.self_attn.o_proj"`,
/// return the shard kind based on the last `.`-separated segment.
/// Returns `Replicate` on the empty string.
///
/// Consumed by `codegen::emit_unindexed_let` /
/// `codegen::emit_layered_load_body` once task #5 wires shard-kind
/// dispatch into the load helpers. Pinned by tests now; the
/// `dead_code` allow is removed on its first non-test caller.
#[allow(dead_code)]
pub(crate) fn shard_kind_for_dotted_prefix(prefix: &str) -> ShardKind {
    let last = prefix.rsplit('.').next().unwrap_or("");
    shard_kind_for_last_segment(last)
}

/// Walk the FUF and inject `OpKind::AllReduce` nodes after every
/// row-parallel gemm AND after every vocab-parallel `Embed`. At
/// `tp_world_size = 1` returns immediately without touching `fuf`.
///
/// At `tp_world_size > 1` the pass:
///
/// 1. Collects every `OpKind::Gemm` node whose `FufInput::Weight`
///    has a `ShardDim1` path (`o_proj` / `down_proj`) — the
///    row-parallel gemm pattern.
/// 2. Collects every `OpKind::Embed` node whose weight has a
///    `ShardDim0` path (`embed_tokens`) — the vocab-parallel embed
///    pattern. The masked-gather kernel produces zeros for tokens
///    outside the rank's vocab slice; the AllReduce-sum across
///    ranks yields exactly one embedding contribution per token,
///    matching Python `VocabParallelEmbedding.forward_native`.
/// 3. For each producer, appends a fresh `OpKind::AllReduce`
///    FufNode reading the producer's output as its single tile
///    input, and rewires every OTHER node's `FufInput::Tile` to
///    read the AllReduce instead. The producer itself never
///    reads its own output and the new AllReduce intentionally
///    reads from the producer, so both are safely skipped.
pub fn insert_all_reduces(fuf: &mut Fuf, program: &Program, tp_world_size: u8) {
    if tp_world_size <= 1 {
        // No-op fast path. Pinned by `lowering_no_allreduce_at_tp_eq_1`:
        // any change here that mutates the FUF at tp=1 must trip the
        // single-rank build's existing test suite (216 macro tests +
        // 17-arch golden subset).
        return;
    }

    let mut producers: Vec<TileId> = Vec::new();
    for node in &fuf.nodes {
        let weight_id = node.inputs.iter().find_map(|i| match i {
            FufInput::Weight { id, .. } => Some(*id),
            _ => None,
        });
        let Some(wid) = weight_id else { continue };
        let path = program.weights.path(wid);
        let kind = shard_kind_for_weight_path(path);
        let is_target = match node.op {
            // Row-parallel collapse projection: `o_proj`, `down_proj`.
            OpKind::Gemm => kind == ShardKind::ShardDim1,
            // Vocab-parallel embedding: `embed_tokens`.
            OpKind::Embed => kind == ShardKind::ShardDim0,
            _ => false,
        };
        if is_target {
            producers.push(node.id);
        }
    }

    for src_id in producers {
        let src_shape = fuf.get(src_id).outputs[0].clone();
        let new_id = TileId(fuf.nodes.len() as u32);
        fuf.nodes.push(FufNode {
            id: new_id,
            op: OpKind::AllReduce,
            inputs: vec![FufInput::Tile {
                id: src_id,
                slot: 0,
            }],
            outputs: vec![src_shape],
        });
        rewire_consumers(fuf, src_id, new_id);
    }
}

/// Walk the FUF and inject `OpKind::AllGather` after the lm_head
/// Gemm at tp>1. lm_head is vocab-parallel (`ShardDim0`): the
/// per-rank matmul produces `[N, vocab/tp]` partial logits, and
/// the AllGather reassembles `[N, vocab]` for the sampler.
/// Mirrors Python `LogitsProcessor`'s
/// `tensor_model_parallel_all_gather(logits)` post-lm_head.
///
/// Identifies lm_head by name (last segment of the gemm's weight
/// path == `"lm_head"`). Only one match per FUF in every current
/// arch — multiple matches would still be handled by inserting one
/// AllGather per occurrence, but no arch produces them.
///
/// At `tp_world_size = 1` returns immediately. The AllGather
/// output's last dim is `tp_world_size` × the input's last dim —
/// per-rank lm_head Gemm produces `[N, vocab/tp]`, the gather
/// assembles `[N, vocab]`. Setting a DIFFERENT FUF shape on the
/// AllGather node is load-bearing: shape-aware coloring otherwise
/// sees same-shape + disjoint-liveness with the lm_head Gemm
/// output and collapses both into the same slot, defeating
/// `AllGatherImpl::output_alias = None`'s intent. Empirically
/// surfaced on Qwen2.5-3B where `AllGather(6, 6)` (in_slot ==
/// out_slot) produced wrong logits.
pub fn insert_lm_head_allgather(fuf: &mut Fuf, program: &Program, tp_world_size: u8) {
    if tp_world_size <= 1 {
        return;
    }

    let mut lm_head_gemms: Vec<TileId> = Vec::new();
    for node in &fuf.nodes {
        if node.op != OpKind::Gemm {
            continue;
        }
        let Some(wid) = node.inputs.iter().find_map(|i| match i {
            FufInput::Weight { id, .. } => Some(*id),
            _ => None,
        }) else {
            continue;
        };
        let path = program.weights.path(wid);
        if path.last().map(String::as_str) == Some("lm_head") {
            lm_head_gemms.push(node.id);
        }
    }

    for src_id in lm_head_gemms {
        let src_shape = fuf.get(src_id).outputs[0].clone();
        // AllGather output's last dim is `tp_world_size` × the
        // input's last dim — the per-rank lm_head Gemm produces
        // `[N, vocab/tp]`, the gather assembles `[N, vocab]`.
        // Setting a DIFFERENT shape on the FUF node is critical:
        // otherwise the shape-aware coloring sees same-shape +
        // disjoint-liveness with the lm_head Gemm output and
        // collapses the AllGather output into the same slot
        // (validated empirically — `AllGather(6, 6)` fired in the
        // pre-fix Qwen2.5 trace, producing wrong logits). The
        // distinct shape forces a separate slot, matching
        // `AllGatherImpl::output_alias = None`'s intent.
        let mut out_shape = src_shape;
        if let Some(last) = out_shape.last_mut() {
            let tp_lit = Dim::Lit(u64::from(tp_world_size));
            *last = match std::mem::replace(last, Dim::Lit(0)) {
                Dim::Mul(mut parts) => {
                    parts.push(tp_lit);
                    Dim::Mul(parts)
                }
                d => Dim::Mul(vec![d, tp_lit]),
            };
        }
        let new_id = TileId(fuf.nodes.len() as u32);
        fuf.nodes.push(FufNode {
            id: new_id,
            op: OpKind::AllGather,
            inputs: vec![FufInput::Tile {
                id: src_id,
                slot: 0,
            }],
            outputs: vec![out_shape],
        });
        rewire_consumers(fuf, src_id, new_id);
    }
}

/// Walk the FUF and inject `OpKind::MmEmbedSplice` after every
/// `OpKind::Embed`. Runs **after** `insert_all_reduces` so at tp>1
/// the insertion sees the already-rewired state: if the Embed had
/// a vocab-parallel AllReduce chained onto it, the splice reads
/// that AllReduce's output. At tp=1 the splice reads the Embed
/// directly.
///
/// Unconditional at every tp (including tp=1) — the splice op is
/// a runtime no-op when `ForwardCtx::embed_patches` is empty, so
/// text-only batches pay one branch per forward pass. This
/// replaces the pre-refactor inline splice inside
/// `Instruction::Embed::eval`, which was wrong at tp>1 (every
/// rank's D2D overwrite got summed by the post-Embed AllReduce,
/// multiplying mm_embeds by `tp_world_size`).
///
/// **Embed-scale handling (Gemma2/Gemma3 family).** The DSL `embed(ids,
/// embed_tokens) * sqrt(hidden_size)` lowers to `Embed → Mul(scalar)`.
/// HF semantics (`Gemma3TextScaledWordEmbedding` + `masked_scatter` at
/// `transformers/models/gemma3/modeling_gemma3.py:907-913`): the scale
/// applies ONLY to text positions; image-feature replacement happens
/// AFTER scaling. To match, walk past any single-consumer scalar `Mul`
/// (`x * scalar` — claimed by `ScalarMulImpl`) before injecting the
/// splice, so the splice writes UN-scaled image features into the
/// already-scaled text-embedding buffer. Without this, image features
/// get multiplied by ~50× through the residual stream — RMSNorm at the
/// next layer divides the scale out for attention input, but every
/// block's residual add re-introduces the over-scaled `hidden_states`,
/// drowning out attention's contribution at image positions. End
/// effect: image content is functionally invisible to the LM, so
/// outputs are independent of which image was sent (R1 red and R2 blue
/// produce byte-identical first-token logits when the bug is live).
#[cfg(any(feature = "cuda", feature = "metal"))]
pub fn insert_mm_splices(fuf: &mut Fuf, _program: &Program) {
    let mut insertions: Vec<TileId> = Vec::new();
    for node in &fuf.nodes {
        if node.op != OpKind::Embed {
            continue;
        }
        let embed_id = node.id;
        let reduce_id = fuf.nodes.iter().find_map(|n| {
            if n.op != OpKind::AllReduce {
                return None;
            }
            match n.inputs.first() {
                Some(FufInput::Tile { id, .. }) if *id == embed_id => Some(n.id),
                _ => None,
            }
        });
        let after_reduce = reduce_id.unwrap_or(embed_id);
        // Skip past an embedding-scale Mul (`x * scalar`) so the
        // splice writes after scaling, matching HF's masked_scatter
        // ordering. See doc comment above for why this matters.
        let after_scale = scalar_mul_consumer(fuf, after_reduce).unwrap_or(after_reduce);
        insertions.push(after_scale);
    }

    for src_id in insertions {
        let src_shape = fuf.get(src_id).outputs[0].clone();
        let new_id = TileId(fuf.nodes.len() as u32);
        fuf.nodes.push(FufNode {
            id: new_id,
            op: OpKind::MmEmbedSplice,
            inputs: vec![FufInput::Tile {
                id: src_id,
                slot: 0,
            }],
            outputs: vec![src_shape],
        });
        rewire_consumers(fuf, src_id, new_id);
    }
}

/// Find a downstream `OpKind::Mul` whose inputs are exactly one Tile
/// from `src_id` and one Scalar — i.e. the `x * scalar` pattern that
/// `ScalarMulImpl` claims. Returns the Mul's TileId so the caller can
/// place a splice after it. Returns `None` if no such consumer exists.
#[cfg(any(feature = "cuda", feature = "metal"))]
fn scalar_mul_consumer(fuf: &Fuf, src_id: TileId) -> Option<TileId> {
    fuf.nodes.iter().find_map(|n| {
        if n.op != OpKind::Mul || n.inputs.len() != 2 {
            return None;
        }
        let mut tile_from_src = false;
        let mut has_scalar = false;
        for inp in &n.inputs {
            match inp {
                FufInput::Tile { id, .. } if *id == src_id => tile_from_src = true,
                FufInput::Scalar(_) => has_scalar = true,
                _ => return None,
            }
        }
        if tile_from_src && has_scalar {
            Some(n.id)
        } else {
            None
        }
    })
}

/// Rewire every consumer of `(old_id, slot 0)` to read `(new_id,
/// slot 0)` instead. Skips `old_id` (no self-reference) and
/// `new_id` (which intentionally reads from `old_id` as the
/// collective's input). Used by both AllReduce and AllGather
/// insertions.
fn rewire_consumers(fuf: &mut Fuf, old_id: TileId, new_id: TileId) {
    for node in &mut fuf.nodes {
        if node.id == new_id || node.id == old_id {
            continue;
        }
        for input in &mut node.inputs {
            if let FufInput::Tile { id, slot } = input
                && *id == old_id
                && *slot == 0
            {
                *id = new_id;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::classified::{ExternKind, OpKind, WeightId, WeightTable};
    use crate::fuf::{Fuf, FufInput, FufNode, TileId};
    use crate::shape::Dim;

    /// Build a tiny synthetic Program with just enough WeightTable
    /// entries to call the pass without panicking. Mirrors the names
    /// the per-arch shard-kind lookup will consult at tp>1.
    fn synthetic_program() -> Program {
        let mut weights = WeightTable::default();
        let _ = weights.intern_str(vec!["self_attn".into(), "o_proj".into()]);
        let _ = weights.intern_str(vec!["mlp".into(), "down_proj".into()]);
        Program {
            statements: Vec::new(),
            locals: Default::default(),
            weights,
            reshape_targets: Default::default(),
            prelude: crate::classified::Prelude::Decoder,
            decoder_safetensors_prefix: None,
            weight_leaf_renames: Vec::new(),
        }
    }

    /// Build a FUF with one Gemm node + one downstream Add. This is
    /// the topology a row-parallel weight (e.g. `o_proj`) produces in
    /// the single-rank trace: `gemm → residual_add`. At tp>1 the
    /// lowering pass would insert AllReduce between them; at tp=1
    /// the output must be identical to the input.
    fn gemm_then_add_fuf() -> Fuf {
        let shape = || vec![Dim::Lit(4), Dim::Lit(16)];
        Fuf {
            nodes: vec![
                FufNode {
                    id: TileId(0),
                    op: OpKind::Embed,
                    inputs: vec![FufInput::Extern {
                        kind: ExternKind::InputIds,
                        index: None,
                    }],
                    outputs: vec![shape()],
                },
                FufNode {
                    id: TileId(1),
                    op: OpKind::Gemm,
                    inputs: vec![
                        FufInput::Tile {
                            id: TileId(0),
                            slot: 0,
                        },
                        FufInput::Weight {
                            id: WeightId(0),
                            index: None,
                            storage: crate::quantization::StorageFormat::Dense,
                        },
                    ],
                    outputs: vec![shape()],
                },
                FufNode {
                    id: TileId(2),
                    op: OpKind::Add,
                    inputs: vec![
                        FufInput::Tile {
                            id: TileId(1),
                            slot: 0,
                        },
                        FufInput::Tile {
                            id: TileId(0),
                            slot: 0,
                        },
                    ],
                    outputs: vec![shape()],
                },
            ],
        }
    }

    /// At `tp_world_size = 1` the pass must be a strict no-op. The
    /// FUF is byte-identical to the input — same node count, same
    /// op kinds, same connectivity. This is the load-bearing
    /// invariant that lets scratchy-models keep building under
    /// `--features cuda` (no nccl) with zero behavior change after
    /// the lowering pass is wired into `compile()` in task #5c.
    #[test]
    fn lowering_no_allreduce_at_tp_eq_1() {
        let mut fuf = gemm_then_add_fuf();
        let original = fuf.clone();
        let program = synthetic_program();

        insert_all_reduces(&mut fuf, &program, 1);

        assert_eq!(
            fuf.nodes.len(),
            original.nodes.len(),
            "tp=1 lowering pass must not change the FUF node count",
        );
        for (after, before) in fuf.nodes.iter().zip(original.nodes.iter()) {
            assert_eq!(
                after.id, before.id,
                "tp=1 lowering pass must preserve TileId numbering",
            );
            assert_eq!(
                after.op, before.op,
                "tp=1 lowering pass must preserve OpKind on every tile",
            );
        }
        assert!(
            !fuf.nodes.iter().any(|n| n.op == OpKind::AllReduce),
            "tp=1 lowering pass must produce zero OpKind::AllReduce nodes",
        );
    }

    /// Same invariant on an empty FUF — the pass must accept the
    /// degenerate case without panicking. Prevents a future
    /// implementation from indexing into nodes[0] without a length
    /// check.
    #[test]
    fn lowering_no_allreduce_at_tp_eq_1_empty_fuf() {
        let mut fuf = Fuf { nodes: Vec::new() };
        let program = synthetic_program();
        insert_all_reduces(&mut fuf, &program, 1);
        assert!(fuf.nodes.is_empty(), "empty FUF must stay empty at tp=1");
    }

    /// Generic shard-kind matcher: HF-standard names route to the
    /// expected `ShardKind`. Pinning the table explicitly so a
    /// future per-arch override that adds new names doesn't silently
    /// break the generic fallback for the standard arches.
    #[test]
    fn shard_kind_table_covers_hf_standard_names() {
        let make = |s: &str| vec![s.to_string()];
        for n in ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"] {
            assert_eq!(
                shard_kind_for_weight_path(&make(n)),
                ShardKind::ShardDim0,
                "{n} must map to column-parallel (ShardDim0)",
            );
        }
        for n in ["o_proj", "down_proj"] {
            assert_eq!(
                shard_kind_for_weight_path(&make(n)),
                ShardKind::ShardDim1,
                "{n} must map to row-parallel (ShardDim1)",
            );
        }
        // Vocab-parallel: matches Python vLLM's VocabParallelEmbedding
        // and ParallelLMHead (which inherits from VocabParallelEmbedding).
        // Both shard along dim 0 (vocab dim) so the tied-weight case
        // stays self-consistent at tp>1.
        for n in ["embed_tokens", "lm_head"] {
            assert_eq!(
                shard_kind_for_weight_path(&make(n)),
                ShardKind::ShardDim0,
                "{n} must map to vocab-parallel (ShardDim0) per Python vLLM",
            );
        }
        // Replicate fallback covers norms, rotary, etc. — anything
        // without a parallel-layer analog in Python vLLM.
        for n in [
            "input_layernorm",
            "norm",
            "post_attention_layernorm",
            "rotary_emb",
        ] {
            assert_eq!(
                shard_kind_for_weight_path(&make(n)),
                ShardKind::Replicate,
                "{n} must default to Replicate",
            );
        }
        // Empty path defaults to Replicate, not panic.
        assert_eq!(
            shard_kind_for_weight_path(&[]),
            ShardKind::Replicate,
            "empty path must default to Replicate",
        );
        // Match on the LAST segment, regardless of leading namespace.
        let dotted = vec!["self_attn".to_string(), "o_proj".to_string()];
        assert_eq!(
            shard_kind_for_weight_path(&dotted),
            ShardKind::ShardDim1,
            "dotted `self_attn.o_proj` matches on last segment",
        );
    }

    /// `shard_kind_for_dotted_prefix` is what codegen will call —
    /// it has the layer-0 prefix string in hand (e.g.
    /// `"model.layers.0.self_attn.o_proj"` or `"model.embed_tokens"`)
    /// and needs the shard kind without first splitting into
    /// segments. Pinning the dotted-prefix variant against the
    /// load-bearing paths in every arch's `default_required_weights`.
    #[test]
    fn shard_kind_for_dotted_prefix_pins_runtime_paths() {
        // Layered (per-layer) weights: column-parallel.
        for p in [
            "model.layers.0.self_attn.q_proj",
            "model.layers.0.self_attn.k_proj",
            "model.layers.0.self_attn.v_proj",
            "model.layers.0.mlp.gate_proj",
            "model.layers.0.mlp.up_proj",
        ] {
            assert_eq!(
                shard_kind_for_dotted_prefix(p),
                ShardKind::ShardDim0,
                "{p} must be column-parallel",
            );
        }
        // Layered row-parallel.
        for p in [
            "model.layers.0.self_attn.o_proj",
            "model.layers.0.mlp.down_proj",
        ] {
            assert_eq!(
                shard_kind_for_dotted_prefix(p),
                ShardKind::ShardDim1,
                "{p} must be row-parallel",
            );
        }
        // Unindexed vocab-parallel: matches Python vLLM
        // VocabParallelEmbedding + ParallelLMHead.
        for p in ["model.embed_tokens", "lm_head"] {
            assert_eq!(
                shard_kind_for_dotted_prefix(p),
                ShardKind::ShardDim0,
                "{p} must be vocab-parallel",
            );
        }
        // Norms / replicates.
        for p in [
            "model.norm",
            "model.layers.0.input_layernorm",
            "model.layers.0.post_attention_layernorm",
        ] {
            assert_eq!(
                shard_kind_for_dotted_prefix(p),
                ShardKind::Replicate,
                "{p} must be replicated",
            );
        }
        // Empty string → Replicate, not panic.
        assert_eq!(
            shard_kind_for_dotted_prefix(""),
            ShardKind::Replicate,
            "empty prefix must default to Replicate",
        );
    }

    /// The load-bearing tp>1 invariant: the lowering pass inserts
    /// exactly one `OpKind::AllReduce` per row-parallel gemm and
    /// rewires every downstream consumer to read the post-reduce
    /// value. Any consumer that still reads the raw gemm output at
    /// tp>1 would mix per-rank pre-reduce values into the residual
    /// stream — silently wrong math.
    ///
    /// Topology: embed → gemm(o_proj) → residual_add(gemm, embed).
    /// After lowering at tp=2: the gemm and embed are unchanged, a
    /// new AllReduce node reads the gemm, and the residual_add now
    /// reads the AllReduce instead of the gemm.
    #[test]
    fn lowering_inserts_allreduce_after_row_parallel_gemm_at_tp_gt_1() {
        let mut fuf = gemm_then_add_fuf();
        let program = synthetic_program();
        // The gemm at TileId(1) reads WeightId(0) which the synthetic
        // program maps to ["self_attn", "o_proj"] → ShardDim1.

        let original_gemm_id = TileId(1);
        let original_add_id = TileId(2);
        let original_node_count = fuf.nodes.len();

        insert_all_reduces(&mut fuf, &program, 2);

        // One node added.
        assert_eq!(
            fuf.nodes.len(),
            original_node_count + 1,
            "expected exactly one AllReduce inserted after the row-parallel gemm",
        );

        // The new node is the AllReduce, with the correct shape
        // and reading the gemm output.
        let ar = fuf.nodes.last().unwrap();
        assert_eq!(ar.op, OpKind::AllReduce);
        assert_eq!(
            ar.id,
            TileId(original_node_count as u32),
            "new AllReduce gets the next sequential TileId",
        );
        match ar.inputs.as_slice() {
            [FufInput::Tile { id, slot: 0 }] => assert_eq!(
                *id, original_gemm_id,
                "AllReduce must read the row-parallel gemm output",
            ),
            other => panic!("AllReduce inputs malformed: {other:?}"),
        }

        // Gemm itself unchanged.
        let gemm = fuf.get(original_gemm_id);
        assert_eq!(gemm.op, OpKind::Gemm);
        match gemm.inputs.first() {
            Some(FufInput::Tile {
                id: TileId(0),
                slot: 0,
            }) => {}
            other => panic!("gemm input 0 (activation) must be unchanged: {other:?}"),
        }

        // residual_add now reads from the AllReduce (TileId N), NOT
        // from the gemm (TileId 1). The other input — the embed
        // residual — stays untouched.
        let add = fuf.get(original_add_id);
        let reads_allreduce = add.inputs.iter().any(|i| {
            matches!(
                i,
                FufInput::Tile { id, slot: 0 } if *id == ar.id,
            )
        });
        let reads_raw_gemm = add.inputs.iter().any(|i| {
            matches!(
                i,
                FufInput::Tile { id, slot: 0 } if *id == original_gemm_id,
            )
        });
        assert!(
            reads_allreduce,
            "residual_add must be rewired to read from AllReduce at tp>1",
        );
        assert!(
            !reads_raw_gemm,
            "residual_add must NOT read raw gemm output at tp>1 — \
             that would mix pre-reduce per-rank values into the stream",
        );
    }

    /// `insert_lm_head_allgather` inserts exactly one
    /// `OpKind::AllGather` after a Gemm whose weight's last segment
    /// is `"lm_head"`, and rewires every consumer of the gemm output
    /// to read the AllGather instead. The AllGather output keeps the
    /// same FUF shape as the input (FUF carries pre-shard symbolic
    /// dims; runtime allocation comes from the kernel side). Skips
    /// non-lm_head gemms (q_proj / o_proj / etc.) and is a no-op at
    /// tp=1.
    #[test]
    fn lowering_inserts_allgather_after_lm_head_gemm_at_tp_gt_1() {
        let shape = || vec![Dim::Lit(4), Dim::Lit(16)];
        let mut fuf = Fuf {
            nodes: vec![
                FufNode {
                    id: TileId(0),
                    op: OpKind::Embed,
                    inputs: vec![FufInput::Extern {
                        kind: ExternKind::InputIds,
                        index: None,
                    }],
                    outputs: vec![shape()],
                },
                // The lm_head Gemm — weight path's last segment is
                // `lm_head`. This is the AllGather target.
                FufNode {
                    id: TileId(1),
                    op: OpKind::Gemm,
                    inputs: vec![
                        FufInput::Tile {
                            id: TileId(0),
                            slot: 0,
                        },
                        FufInput::Weight {
                            id: WeightId(0),
                            index: None,
                            storage: crate::quantization::StorageFormat::Dense,
                        },
                    ],
                    outputs: vec![shape()],
                },
            ],
        };
        let mut weights = WeightTable::default();
        let _ = weights.intern_str(vec!["lm_head".into()]); // id 0
        let program = Program {
            statements: Vec::new(),
            locals: Default::default(),
            weights,
            reshape_targets: Default::default(),
            prelude: crate::classified::Prelude::Decoder,
            decoder_safetensors_prefix: None,
            weight_leaf_renames: Vec::new(),
        };
        let original_count = fuf.nodes.len();

        insert_lm_head_allgather(&mut fuf, &program, 2);

        assert_eq!(
            fuf.nodes.len(),
            original_count + 1,
            "expected exactly one AllGather inserted after the lm_head gemm",
        );
        let ag = fuf.nodes.last().unwrap();
        assert_eq!(ag.op, OpKind::AllGather);
        match ag.inputs.as_slice() {
            [FufInput::Tile { id, slot: 0 }] => assert_eq!(
                *id,
                TileId(1),
                "AllGather must read the lm_head gemm output",
            ),
            other => panic!("AllGather inputs malformed: {other:?}"),
        }
        // Output's last dim must be DIFFERENT from the input's last
        // dim — multiplied by `tp_world_size` so shape-aware coloring
        // can't collapse the AllGather output into the lm_head Gemm's
        // slot. Empirically validated by the Qwen2.5-3B trace where
        // pre-fix `AllGather(6, 6)` (in_slot == out_slot) produced
        // wrong logits.
        let in_shape = &fuf.get(TileId(1)).outputs[0];
        assert_ne!(
            ag.outputs[0], *in_shape,
            "AllGather output shape must NOT equal input shape — same shape \
             lets shape-coloring collapse the slot",
        );
        // The non-last dims should be unchanged.
        let out_shape = &ag.outputs[0];
        assert_eq!(out_shape.len(), in_shape.len());
        for i in 0..out_shape.len() - 1 {
            assert_eq!(
                out_shape[i], in_shape[i],
                "non-last dim must be unchanged at idx {i}",
            );
        }
        // Last dim should be the input's last dim multiplied by tp.
        match out_shape.last().unwrap() {
            Dim::Mul(parts) => {
                assert_eq!(parts.last(), Some(&Dim::Lit(2)));
            }
            other => panic!("AllGather last dim must be Mul (got {other:?})"),
        }
    }

    /// At `tp_world_size = 1` the AllGather pass is a strict no-op.
    /// Pinned alongside the other tp=1 invariants so a future change
    /// can't silently start emitting AllGathers in dense single-rank
    /// builds.
    #[test]
    fn lowering_no_allgather_at_tp_eq_1() {
        let shape = || vec![Dim::Lit(4), Dim::Lit(16)];
        let mut fuf = Fuf {
            nodes: vec![FufNode {
                id: TileId(0),
                op: OpKind::Gemm,
                inputs: vec![
                    FufInput::Extern {
                        kind: ExternKind::InputIds,
                        index: None,
                    },
                    FufInput::Weight {
                        id: WeightId(0),
                        index: None,
                        storage: crate::quantization::StorageFormat::Dense,
                    },
                ],
                outputs: vec![shape()],
            }],
        };
        let mut weights = WeightTable::default();
        let _ = weights.intern_str(vec!["lm_head".into()]);
        let program = Program {
            statements: Vec::new(),
            locals: Default::default(),
            weights,
            reshape_targets: Default::default(),
            prelude: crate::classified::Prelude::Decoder,
            decoder_safetensors_prefix: None,
            weight_leaf_renames: Vec::new(),
        };
        let original_count = fuf.nodes.len();
        insert_lm_head_allgather(&mut fuf, &program, 1);
        assert_eq!(fuf.nodes.len(), original_count);
        assert!(!fuf.nodes.iter().any(|n| n.op == OpKind::AllGather));
    }

    /// Non-lm_head Gemms (q_proj, o_proj, etc.) must NOT trigger
    /// AllGather insertion even at tp>1. AllGather is exclusively
    /// for the vocab-parallel terminal Gemm whose output needs to
    /// be reassembled across ranks for the sampler.
    #[test]
    fn lowering_skips_non_lm_head_gemms_for_allgather() {
        let shape = || vec![Dim::Lit(4), Dim::Lit(16)];
        let mut fuf = Fuf {
            nodes: vec![FufNode {
                id: TileId(0),
                op: OpKind::Gemm,
                inputs: vec![
                    FufInput::Extern {
                        kind: ExternKind::InputIds,
                        index: None,
                    },
                    FufInput::Weight {
                        id: WeightId(0),
                        index: None,
                        storage: crate::quantization::StorageFormat::Dense,
                    },
                ],
                outputs: vec![shape()],
            }],
        };
        let mut weights = WeightTable::default();
        let _ = weights.intern_str(vec!["self_attn".into(), "q_proj".into()]); // id 0
        let program = Program {
            statements: Vec::new(),
            locals: Default::default(),
            weights,
            reshape_targets: Default::default(),
            prelude: crate::classified::Prelude::Decoder,
            decoder_safetensors_prefix: None,
            weight_leaf_renames: Vec::new(),
        };
        let original_count = fuf.nodes.len();
        insert_lm_head_allgather(&mut fuf, &program, 4);
        assert_eq!(
            fuf.nodes.len(),
            original_count,
            "q_proj must not trigger AllGather (only lm_head does)",
        );
        assert!(!fuf.nodes.iter().any(|n| n.op == OpKind::AllGather));
    }

    /// Vocab-parallel `Embed` (weight is `embed_tokens` →
    /// `ShardDim0`) must trigger AllReduce insertion at tp>1.
    /// Mirrors Python `VocabParallelEmbedding.forward_native`: the
    /// per-rank masked-gather produces zeros for tokens outside the
    /// rank's vocab slice; the AllReduce-sum reassembles the global
    /// embedding by summing exactly one non-zero contribution per
    /// token. Without this insertion, ranks 1..N would silently
    /// contribute zero embeddings to the residual stream.
    #[test]
    fn lowering_inserts_allreduce_after_vocab_parallel_embed_at_tp_gt_1() {
        let shape = || vec![Dim::Lit(4), Dim::Lit(16)];
        let mut fuf = Fuf {
            nodes: vec![
                // Embed reading `embed_tokens` (ShardDim0 by name).
                FufNode {
                    id: TileId(0),
                    op: OpKind::Embed,
                    inputs: vec![
                        FufInput::Extern {
                            kind: ExternKind::InputIds,
                            index: None,
                        },
                        FufInput::Weight {
                            id: WeightId(0),
                            index: None,
                            storage: crate::quantization::StorageFormat::Dense,
                        },
                    ],
                    outputs: vec![shape()],
                },
                // Downstream consumer (residual-add into hidden
                // state). Should be rewired to read the AllReduce.
                FufNode {
                    id: TileId(1),
                    op: OpKind::Add,
                    inputs: vec![
                        FufInput::Tile {
                            id: TileId(0),
                            slot: 0,
                        },
                        FufInput::Tile {
                            id: TileId(0),
                            slot: 0,
                        },
                    ],
                    outputs: vec![shape()],
                },
            ],
        };
        let mut weights = WeightTable::default();
        let _ = weights.intern_str(vec!["model".into(), "embed_tokens".into()]); // id 0
        let program = Program {
            statements: Vec::new(),
            locals: Default::default(),
            weights,
            reshape_targets: Default::default(),
            prelude: crate::classified::Prelude::Decoder,
            decoder_safetensors_prefix: None,
            weight_leaf_renames: Vec::new(),
        };

        let original_count = fuf.nodes.len();
        insert_all_reduces(&mut fuf, &program, 2);
        assert_eq!(
            fuf.nodes.len(),
            original_count + 1,
            "expected exactly one AllReduce inserted after the vocab-parallel embed",
        );
        let ar = fuf.nodes.last().unwrap();
        assert_eq!(ar.op, OpKind::AllReduce);
        match ar.inputs.as_slice() {
            [FufInput::Tile { id, slot: 0 }] => {
                assert_eq!(*id, TileId(0), "AllReduce must read the embed output",)
            }
            other => panic!("AllReduce inputs malformed: {other:?}"),
        }
        // Downstream Add must be rewired to read the AllReduce.
        let add = fuf.get(TileId(1));
        let reads_allreduce = add
            .inputs
            .iter()
            .any(|i| matches!(i, FufInput::Tile { id, slot: 0 } if *id == ar.id));
        assert!(
            reads_allreduce,
            "downstream consumer must be rewired to read post-AllReduce embed value at tp>1",
        );
    }

    /// Embed with a non-vocab-parallel weight (e.g. a hypothetical
    /// arch-specific embed that the shard table happens to map to
    /// `Replicate`) must NOT trigger AllReduce insertion. Pins the
    /// shard-kind gate so a future change to the shard table can't
    /// silently start emitting stray AllReduces on Replicate-keyed
    /// embeds.
    #[test]
    fn lowering_skips_replicate_embed_at_tp_gt_1() {
        let shape = || vec![Dim::Lit(4), Dim::Lit(16)];
        let mut fuf = Fuf {
            nodes: vec![FufNode {
                id: TileId(0),
                op: OpKind::Embed,
                inputs: vec![
                    FufInput::Extern {
                        kind: ExternKind::InputIds,
                        index: None,
                    },
                    FufInput::Weight {
                        id: WeightId(0),
                        index: None,
                        storage: crate::quantization::StorageFormat::Dense,
                    },
                ],
                outputs: vec![shape()],
            }],
        };
        let mut weights = WeightTable::default();
        // A non-standard embed name not in the shard table → Replicate.
        let _ = weights.intern_str(vec!["model".into(), "side_embed".into()]); // id 0
        let program = Program {
            statements: Vec::new(),
            locals: Default::default(),
            weights,
            reshape_targets: Default::default(),
            prelude: crate::classified::Prelude::Decoder,
            decoder_safetensors_prefix: None,
            weight_leaf_renames: Vec::new(),
        };
        let original_count = fuf.nodes.len();
        insert_all_reduces(&mut fuf, &program, 4);
        assert_eq!(
            fuf.nodes.len(),
            original_count,
            "Replicate-keyed embed must not trigger AllReduce insertion",
        );
        assert!(!fuf.nodes.iter().any(|n| n.op == OpKind::AllReduce));
    }

    /// Column-parallel weights (`q_proj` / `gate_proj` / `up_proj`)
    /// must NOT trigger AllReduce insertion at tp>1 — their output
    /// is a sharded matrix the next op operates on per-rank, no
    /// communication needed. False insertion would add unnecessary
    /// traffic AND clobber the per-rank shape contract.
    #[test]
    fn lowering_skips_column_parallel_gemms_at_tp_gt_1() {
        let shape = || vec![Dim::Lit(4), Dim::Lit(16)];
        // FUF: embed → gemm(q_proj_weight) → consumer
        // q_proj is ShardDim0 (column-parallel). No AllReduce.
        let mut fuf = Fuf {
            nodes: vec![
                FufNode {
                    id: TileId(0),
                    op: OpKind::Embed,
                    inputs: vec![FufInput::Extern {
                        kind: ExternKind::InputIds,
                        index: None,
                    }],
                    outputs: vec![shape()],
                },
                FufNode {
                    id: TileId(1),
                    op: OpKind::Gemm,
                    inputs: vec![
                        FufInput::Tile {
                            id: TileId(0),
                            slot: 0,
                        },
                        // WeightId(2) — the synthetic_program below
                        // maps it to "self_attn.q_proj".
                        FufInput::Weight {
                            id: WeightId(2),
                            index: None,
                            storage: crate::quantization::StorageFormat::Dense,
                        },
                    ],
                    outputs: vec![shape()],
                },
                FufNode {
                    id: TileId(2),
                    op: OpKind::Add,
                    inputs: vec![
                        FufInput::Tile {
                            id: TileId(1),
                            slot: 0,
                        },
                        FufInput::Tile {
                            id: TileId(0),
                            slot: 0,
                        },
                    ],
                    outputs: vec![shape()],
                },
            ],
        };
        let mut weights = WeightTable::default();
        let _ = weights.intern_str(vec!["self_attn".into(), "o_proj".into()]); // id 0
        let _ = weights.intern_str(vec!["mlp".into(), "down_proj".into()]); // id 1
        let _ = weights.intern_str(vec!["self_attn".into(), "q_proj".into()]); // id 2
        let program = Program {
            statements: Vec::new(),
            locals: Default::default(),
            weights,
            reshape_targets: Default::default(),
            prelude: crate::classified::Prelude::Decoder,
            decoder_safetensors_prefix: None,
            weight_leaf_renames: Vec::new(),
        };

        let original_count = fuf.nodes.len();
        insert_all_reduces(&mut fuf, &program, 2);
        assert_eq!(
            fuf.nodes.len(),
            original_count,
            "column-parallel gemm must not get an AllReduce inserted",
        );
        assert!(
            !fuf.nodes.iter().any(|n| n.op == OpKind::AllReduce),
            "no OpKind::AllReduce nodes for column-parallel weights",
        );
    }
}
