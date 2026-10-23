// SPDX-License-Identifier: Apache-2.0
//! CUDA's per-opcode ABI as DECLARED DATA — the counterpart to metal's `op_abi`.
//!
//! Target-ABI facts are small const tables the shared passes take as *input* — never logic. Here:
//! how an op's kernel treats its buffers ([`cuda_colour_rule`]), which steps CUDA fuses
//! ([`CUDA_FUSIONS`]), which result rows it samples ([`CUDA_SAMPLE_ROWS`]), how wide a graph
//! node is ([`CUDA_COLUMN_BLOCK`]), and where a mixture-of-experts block keeps its runtime data
//! ([`cuda_moe_write`], [`CUDA_GROUPED_EXPERTS`], [`CUDA_SORTED_PAIRS`],
//! [`CUDA_EXPERT_TILE_ROWS`]).
//!
//! ⭐ EVERY ENTRY IS THE CONSERVATIVE ONE. Each fact is the one that is correct for ANY kernel a
//! later issue writes: every op writes a fresh buffer and reads its inputs across the step, no row
//! is sampled, and the folds are the `SiluMul` and `MoeRoute` folds, whose kernels exist. That
//! costs arena space and launches; an opcode issue wins them back one row at a time, by changing
//! only its own op's row.

use std::num::NonZeroU32;

use scratchy_subtile::handoff::SourceBinding;
use scratchy_subtile::ops::SubOpKind;
use scratchy_subtile::sample_rows::SampleRowsFacts;
use scratchy_subtile::subtile_ir::{EwKind, ExpertBundle, ExpertProj, OpStage, RopeForm, SubOp};
use scratchy_subtile::tape_colouring::{ColourFacts, ColourRule, OutputAlias};
use scratchy_subtile::tape_folding::{FoldPattern, FusionTable, GatedKernel};
use scratchy_subtile::tape_steps::OperandIx;

// ── The colour rule ─────────────────────────────────────────────────

/// CUDA's colouring facts: the per-op rule table, and the embedded hidden as colour 0 — the same
/// convention as metal's, where the embedding's output is the first buffer.
pub const CUDA_COLOUR_FACTS: ColourFacts = ColourFacts {
    rule: cuda_colour_rule,
    colour_zero: SourceBinding::EmbeddedHidden,
};

/// How each op's CUDA kernel treats its buffers.
///
/// The match is exhaustive: a new `SubOp` (or `EwKind`) fails to compile here (E0004) until its
/// author states which case it is, rather than silently defaulting to one.
///
/// No row is in place: no CUDA kernel writes over an operand. The routed half of a MoE block is
/// off the arena: its values live in the tape's MoE scratch ([`cuda_moe_write`] says which
/// region), not in a colour, because they are not bf16 `[m, cols]` (indices are i32, scores f32,
/// the sorted rows padded to [`CUDA_EXPERT_TILE_ROWS`]), which the colourer's `rows × cols × 2`
/// sizing cannot hold. The combine, which writes the block's `[m, hidden]` result, is in the
/// arena.
pub fn cuda_colour_rule<F: RopeForm, S: OpStage>(op: &SubOp<F, S>) -> ColourRule {
    use EwKind as E;
    use OutputAlias as A;
    use SubOp as L;
    const FRESH: ColourRule = ColourRule::FRESH;
    match op {
        // Metadata-only view over operand 0 — no kernel runs.
        L::Reshape { .. } => FRESH.output(A::Operand(OperandIx(0))),
        // Two-output ops bind a SECOND buffer (VisionRope's k', GateSplit's gate). A fact about the
        // op, not the kernel: without it the second buffer gets no colour.
        L::GateSplit { .. } | L::VisionRope => FRESH.reads_across().two_outputs(),
        // A fresh buffer, its inputs live through the step: an output never shares a colour with
        // one of its own inputs, whatever the kernel does.
        L::MatmulTile { .. }
        | L::SumReduce { .. }
        | L::Elementwise(
            E::Silu
            | E::Gelu
            | E::Mul
            | E::Add
            | E::QuickGelu
            | E::GeluErf
            | E::Relu2
            | E::Sub
            | E::BiasAdd,
        )
        | L::SiluMul
        | L::ScalarMul { .. }
        | L::RmsNorm { .. }
        | L::RmsNormReduce { .. }
        | L::RmsNormApply { .. }
        | L::RopeRotate { .. }
        | L::RopeAppend { .. }
        | L::KvAppend { .. }
        | L::AttnDecode { .. }
        | L::TanhSoftCap { .. }
        | L::RmsNormUnit { .. }
        | L::ScalarWeightMul
        | L::GateApply
        | L::Concat { .. }
        | L::GateScale
        | L::LoadRows { .. }
        | L::EmbeddingGather { .. }
        | L::VarlenAttention { .. }
        | L::EncoderAttn { .. }
        | L::GatedDeltaNet
        | L::Mamba2 { .. }
        | L::Mean
        | L::RouterNorm { .. }
        | L::RouteSigmoidBias { .. }
        | L::RouteBias { .. }
        | L::RouteScale { .. }
        | L::RouteExpertScale { .. }
        | L::ExpertAct { .. }
        | L::ExpertCombine { .. }
        | L::KvEncode { .. }
        | L::KvStage { .. }
        | L::RotateRows { .. }
        | L::AttnPackedKv
        | L::SampleRowsGather
        | L::SampleRowsScatter
        | L::AllRowsMatmul => FRESH.reads_across(),
        // The routed half of a MoE block: in the MoE scratch ([`cuda_moe_write`]), not a colour.
        L::RouterLogits { .. }
        | L::RouteSoftmax
        | L::RouteArgsort
        | L::RouteTopK { .. }
        | L::RouteGatherScores
        | L::RouteRenorm
        | L::ExpertSort { .. }
        | L::ExpertMatmul { .. }
        | L::ExpertGatedAct { .. }
        | L::ExpertUnsort => FRESH.off_arena(),
    }
}

// ── The sampled rows ────────────────────────────────────────────────

/// CUDA's sampled rows: none. The `lm_head` computes every row (bf16 only, no sliced matmul yet),
/// so `sample_rows::expand_sample_rows` inserts nothing.
pub const CUDA_SAMPLE_ROWS: SampleRowsFacts = SampleRowsFacts { weights: &[] };

// ── The fusions ─────────────────────────────────────────────────────

/// The fused kernel each CUDA fold becomes. A fusion issue adds its variant together with its
/// pattern in [`CUDA_FUSIONS`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CudaFusion {
    /// `silu(gate) · up` in one launch (`kernels/silu_mul_bf16.cu`): the SwiGLU MLP's `Silu`
    /// folded into the `Mul` it gates.
    SiluMul,
    /// A MoE block's routing in one launch (`kernels/moe_route_bf16.cu`): the softmax over the
    /// router's logits and their argsort folded into the top-k, the gathered scores and their
    /// renorm its epilogues.
    MoeRoute,
}

/// CUDA's fusions, in the order the shared fold pass applies them: one sweep of two patterns. The
/// `Silu` a `Mul` reads as its operand 0 folds into that `Mul` (metal's `Gated` row, with its
/// `Silu` activation only). A MoE block's top-k folds in the softmax and the argsort before it,
/// and takes the gathered scores and their renorm as its epilogues (metal's `Route` row, with
/// Qwen's softmax-first routing only).
///
/// `fused` repeats `SiluMul`: CUDA's `ModelFoldFacts` never sets `fold_projections`, so the shared
/// pass always picks `split`, and the gate and up GEMMs stay launches of their own. A `Gated` fold
/// with its projections folded in is refused at lowering (`CanonicalRefusal::FusionShape`); one
/// kernel for the gate and up GEMMs is a later GEMM-epilogue issue.
pub const CUDA_FUSIONS: FusionTable<CudaFusion> = FusionTable {
    counted: &[],
    sweeps: &[&[
        FoldPattern::Gated {
            mul: SubOpKind::Mul,
            activations: &[GatedKernel {
                activation: SubOpKind::Silu,
                split: CudaFusion::SiluMul,
                fused: CudaFusion::SiluMul,
            }],
        },
        FoldPattern::Route {
            top_k: SubOpKind::RouteTopK,
            sort: SubOpKind::RouteArgsort,
            pre: &[SubOpKind::RouteSoftmax],
            gather: SubOpKind::RouteGatherScores,
            tail: &[&[], &[SubOpKind::RouteRenorm], &[], &[]],
            kernel: CudaFusion::MoeRoute,
        },
    ]],
    rows: None,
};

// ── The MoE block ───────────────────────────────────────────────────

/// A region of the tape's MoE scratch (`crate::tape::MoeTape`): CUDA's own, because its regions
/// are not metal's (`MoeRegion`): i32 indices, f32 scores, expert runs padded to
/// [`CUDA_EXPERT_TILE_ROWS`] rows, and no router-norm region.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CudaMoeRegion {
    /// `[m, experts]` bf16, the router's logits.
    Logits,
    /// `[m, top_k]` i32, each token's picks.
    TopKIdx,
    /// `[m, top_k]` f32, each pick's routing weight.
    Scores,
    /// `[R, hidden]` bf16, the pairs' token rows sorted by expert (with the sort's index regions).
    SortedRows,
    /// `[R, inter]` bf16 (gathered: `[P, inter]`), the gate projection.
    Gate,
    /// `[R, inter]` bf16 (gathered: `[P, inter]`), the up projection.
    Up,
    /// `[R, inter]` bf16 (gathered: `[P, inter]`), `silu(gate) · up`.
    Act,
    /// `[R, hidden]` bf16 (gathered: `[P, hidden]`), the down projection.
    Down,
}

/// Where an off-arena MoE op's value is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CudaMoeWrite {
    /// In this region of the MoE scratch.
    Region(CudaMoeRegion),
    /// In its operand's region: rewritten there in place (the renorm, over the scores), or read
    /// through it (the unsort, which the combine reads through the sort's `pos`).
    OverOperand(OperandIx),
    /// Nowhere: a member of the `MoeRoute` fold (the softmax, the argsort) its launch keeps in
    /// registers. No other step reads it.
    InFold,
    /// In the arena: the combine's `[m, hidden]` result.
    Arena,
}

/// CUDA's MoE dataflow: where each op of a block's value is. `None` for an op CUDA lowers no MoE
/// step for (the router norm, the sigmoid and bias routings, the scales, the ungated expert).
pub fn cuda_moe_write<F: RopeForm, S: OpStage>(op: &SubOp<F, S>) -> Option<CudaMoeWrite> {
    use CudaMoeRegion as R;
    use CudaMoeWrite as W;
    use SubOp as L;
    Some(match op {
        L::RouterLogits { .. } => W::Region(R::Logits),
        L::RouteSoftmax | L::RouteArgsort => W::InFold,
        L::RouteTopK { .. } => W::Region(R::TopKIdx),
        L::RouteGatherScores => W::Region(R::Scores),
        L::RouteRenorm | L::ExpertUnsort => W::OverOperand(OperandIx(0)),
        L::ExpertSort { .. } => W::Region(R::SortedRows),
        L::ExpertMatmul { proj, .. } => W::Region(match proj {
            ExpertProj::Gate => R::Gate,
            ExpertProj::Up => R::Up,
            ExpertProj::Down => R::Down,
        }),
        L::ExpertGatedAct { .. } => W::Region(R::Act),
        L::ExpertCombine { .. } => W::Arena,
        _ => return None,
    })
}

/// The expert bundles CUDA has a sorted grouped GEMM and an expert GEMV for: the dense bf16
/// shared-fused bundle (the router, then every expert's fused gate and up, then every expert's
/// down). Every other bundle is refused at expansion.
///
/// Metal's speed gates have no counterpart here, on purpose: `METAL_GROUPED_PAIRS_PER_EXPERT`,
/// `METAL_SORTED_PAIRS_PER_EXPERT`, `METAL_SHARED_EXPERTS_FROM` and
/// `METAL_SHARED_EXPERTS_MAX_PAIRS` choose between kernels by speed, or refine the gathered mode,
/// and speed is not yet CUDA's concern: it has one kernel per mode.
pub const CUDA_GROUPED_EXPERTS: &[ExpertBundle] = &[ExpertBundle::SharedFused];

/// The (token, pick) pairs of a bucket from which its tape sorts the pairs by expert and runs the
/// grouped GEMM; under it each pair's expert GEMV reads the pair's token row directly (gathered).
/// Metal's value; with CUDA's buckets {1, 512, …} only the one-row bucket gathers.
pub const CUDA_SORTED_PAIRS: u32 = 64;

/// Each expert's run of sorted pairs is padded to a multiple of this many rows: the grouped GEMM's
/// row tile (`step::GEMM_BF16_TILE.0`), so every tile belongs to exactly one expert.
pub const CUDA_EXPERT_TILE_ROWS: u32 = 128;

const _: () = assert!(CUDA_EXPERT_TILE_ROWS == crate::step::GEMM_BF16_TILE.0);

// ── Tiling ──────────────────────────────────────────────────────────

/// How many columns one graph node covers: all of them. Tiling is a target fact; spyre tiles to
/// 128-column pages because its substrate requires it, while CUDA's kernels — like metal's — take
/// a whole `RmsNorm` or a whole matmul, so the graph is built from whole ops.
pub const CUDA_COLUMN_BLOCK: NonZeroU32 = NonZeroU32::MAX;

#[cfg(test)]
mod tests {
    use super::*;

    use scratchy_subtile::lower::GemmWeight;
    use scratchy_subtile::model_geometry::HeadDim;
    use scratchy_subtile::subtile_ir::{GainConvention, RowScale};
    use scratchy_subtile::tape_colouring::{Outputs, Reads, Residency};

    /// The tables declare exactly the two fused kernels CUDA has, `SiluMul` and `MoeRoute`, and no
    /// sliced `lm_head`: any other pattern, or a sampled weight, would ask the shared passes for a
    /// kernel CUDA does not have. The routing fold takes Qwen's softmax-first routing only: the
    /// softmax as its one pre, the renorm as its one tail stage (stage 1, the post-softmax slot).
    #[test]
    fn cuda_declares_only_the_silu_mul_and_moe_route_folds_and_no_sampled_rows() {
        assert!(CUDA_FUSIONS.counted.is_empty());
        assert!(CUDA_FUSIONS.rows.is_none());
        assert!(CUDA_SAMPLE_ROWS.weights.is_empty());
        let [sweep] = CUDA_FUSIONS.sweeps else {
            panic!("expected one sweep, got {}", CUDA_FUSIONS.sweeps.len());
        };
        let [
            FoldPattern::Gated { mul, activations },
            FoldPattern::Route {
                top_k,
                sort,
                pre,
                gather,
                tail,
                kernel,
            },
        ] = sweep
        else {
            panic!("expected one Gated and one Route pattern, got {sweep:?}");
        };
        assert_eq!(*mul, SubOpKind::Mul);
        let [act] = activations else {
            panic!("expected one gated activation, got {activations:?}");
        };
        assert_eq!(act.activation, SubOpKind::Silu);
        assert_eq!(act.split, CudaFusion::SiluMul);
        assert_eq!(act.fused, CudaFusion::SiluMul);
        assert_eq!(
            (*top_k, *sort, *gather),
            (
                SubOpKind::RouteTopK,
                SubOpKind::RouteArgsort,
                SubOpKind::RouteGatherScores
            )
        );
        assert_eq!(*pre, [SubOpKind::RouteSoftmax]);
        assert_eq!(**tail, [&[][..], &[SubOpKind::RouteRenorm], &[], &[]]);
        assert_eq!(*kernel, CudaFusion::MoeRoute);
    }

    /// The MoE facts: one dense bundle with a grouped GEMM, sorted from metal's 64 pairs, expert
    /// runs padded to the GEMM's 128-row tile.
    #[test]
    fn the_moe_facts() {
        assert_eq!(CUDA_GROUPED_EXPERTS, [ExpertBundle::SharedFused]);
        assert_eq!(CUDA_SORTED_PAIRS, 64);
        assert_eq!(CUDA_EXPERT_TILE_ROWS, 128);
        assert_eq!(CUDA_EXPERT_TILE_ROWS, crate::step::GEMM_BF16_TILE.0);
    }

    /// Where each MoE op's value is: every off-arena op has a region or rides its operand's, the
    /// fold-only steps none, the combine the arena; an op CUDA lowers no MoE step for has no row.
    #[test]
    fn the_moe_write_of_each_op() {
        use scratchy_subtile::lower::ExpertWeight;
        use scratchy_subtile::subtile_ir::{
            GatedAct, NumExperts, RouterBundle, SharedExpertBound, TopK,
        };
        let nz = |n| NonZeroU32::new(n).expect("non-zero");
        let (experts, k) = (NumExperts::new(nz(256)), TopK::new(nz(8)));
        let bundle = ExpertBundle::SharedFused;
        let matmul = |proj| SubOp::ExpertMatmul {
            proj,
            n: 512,
            k,
            weight: ExpertWeight::Dense,
            bundle,
        };
        use CudaMoeRegion as R;
        use CudaMoeWrite as W;
        let cases: [(SubOp, Option<CudaMoeWrite>); 14] = [
            (
                SubOp::RouterLogits {
                    experts,
                    router: RouterBundle::SharedFused,
                },
                Some(W::Region(R::Logits)),
            ),
            (SubOp::RouteSoftmax, Some(W::InFold)),
            (SubOp::RouteArgsort, Some(W::InFold)),
            (SubOp::RouteTopK { k }, Some(W::Region(R::TopKIdx))),
            (SubOp::RouteGatherScores, Some(W::Region(R::Scores))),
            (SubOp::RouteRenorm, Some(W::OverOperand(OperandIx(0)))),
            (
                SubOp::ExpertSort { experts, k, bundle },
                Some(W::Region(R::SortedRows)),
            ),
            (matmul(ExpertProj::Gate), Some(W::Region(R::Gate))),
            (matmul(ExpertProj::Up), Some(W::Region(R::Up))),
            (matmul(ExpertProj::Down), Some(W::Region(R::Down))),
            (
                SubOp::ExpertGatedAct {
                    act: GatedAct::Silu,
                },
                Some(W::Region(R::Act)),
            ),
            (SubOp::ExpertUnsort, Some(W::OverOperand(OperandIx(0)))),
            (
                SubOp::ExpertCombine {
                    hidden: 2048,
                    shared: SharedExpertBound(None),
                },
                Some(W::Arena),
            ),
            (SubOp::RouteScale { scale: 2.0 }, None),
        ];
        for (op, want) in cases {
            assert_eq!(cuda_moe_write(&op), want, "{}", op.name());
        }
    }

    /// The rule's result for each op the CUDA tapes hold, or the next issues will, field by field.
    #[test]
    fn the_colour_rule_of_each_op() {
        let fresh_across = |outputs| ColourRule {
            output: OutputAlias::Fresh,
            reads: Reads::Across,
            outputs,
            residency: Residency::Arena,
            ..ColourRule::FRESH
        };
        let off_arena = ColourRule {
            output: OutputAlias::Fresh,
            reads: Reads::Pointwise,
            outputs: Outputs::One,
            residency: Residency::OffArena,
            ..ColourRule::FRESH
        };
        let nz = |n| NonZeroU32::new(n).expect("non-zero");
        let (experts, k) = (
            scratchy_subtile::subtile_ir::NumExperts::new(nz(256)),
            scratchy_subtile::subtile_ir::TopK::new(nz(8)),
        );
        let bundle = ExpertBundle::SharedFused;
        let cases: [(SubOp, ColourRule); 23] = [
            (
                SubOp::RouterLogits {
                    experts,
                    router: scratchy_subtile::subtile_ir::RouterBundle::SharedFused,
                },
                off_arena,
            ),
            (SubOp::RouteSoftmax, off_arena),
            (SubOp::RouteArgsort, off_arena),
            (SubOp::RouteTopK { k }, off_arena),
            (SubOp::RouteGatherScores, off_arena),
            (SubOp::RouteRenorm, off_arena),
            (SubOp::ExpertSort { experts, k, bundle }, off_arena),
            (
                SubOp::ExpertMatmul {
                    proj: ExpertProj::Gate,
                    n: 512,
                    k,
                    weight: scratchy_subtile::lower::ExpertWeight::Dense,
                    bundle,
                },
                off_arena,
            ),
            (
                SubOp::ExpertGatedAct {
                    act: scratchy_subtile::subtile_ir::GatedAct::Silu,
                },
                off_arena,
            ),
            (SubOp::ExpertUnsort, off_arena),
            (
                SubOp::ExpertCombine {
                    hidden: 2048,
                    shared: scratchy_subtile::subtile_ir::SharedExpertBound(None),
                },
                fresh_across(Outputs::One),
            ),
            (
                SubOp::RopeRotate {
                    head_dim: HeadDim::new(256),
                    _form: std::marker::PhantomData,
                },
                fresh_across(Outputs::One),
            ),
            (SubOp::GateApply, fresh_across(Outputs::One)),
            (
                SubOp::Reshape {
                    rows: RowScale::Times(NonZeroU32::MIN),
                    cols: 1024,
                },
                ColourRule {
                    output: OutputAlias::Operand(OperandIx(0)),
                    reads: Reads::Pointwise,
                    outputs: Outputs::One,
                    residency: Residency::Arena,
                    ..ColourRule::FRESH
                },
            ),
            (
                SubOp::GateSplit { half_cols: 4096 },
                fresh_across(Outputs::Two),
            ),
            (SubOp::VisionRope, fresh_across(Outputs::Two)),
            (
                SubOp::RmsNorm {
                    eps: 1e-6,
                    gain: GainConvention::OnePlusScale,
                },
                fresh_across(Outputs::One),
            ),
            (SubOp::Elementwise(EwKind::Add), fresh_across(Outputs::One)),
            (SubOp::Elementwise(EwKind::Silu), fresh_across(Outputs::One)),
            (SubOp::Elementwise(EwKind::Mul), fresh_across(Outputs::One)),
            (SubOp::SiluMul, fresh_across(Outputs::One)),
            (
                SubOp::MatmulTile {
                    n: 3584,
                    weight: GemmWeight::Dense,
                },
                fresh_across(Outputs::One),
            ),
            (
                SubOp::RmsNorm {
                    eps: 1e-5,
                    gain: GainConvention::Scale,
                },
                fresh_across(Outputs::One),
            ),
        ];
        for (op, want) in cases {
            let got = cuda_colour_rule(&op);
            let name = op.name();
            assert_eq!(got.output, want.output, "{name}: output");
            assert_eq!(got.reads, want.reads, "{name}: reads");
            assert_eq!(got.outputs, want.outputs, "{name}: outputs");
            assert_eq!(got.residency, want.residency, "{name}: residency");
            assert_eq!(got, want, "{name}: the rest of the rule");
        }
    }
}
