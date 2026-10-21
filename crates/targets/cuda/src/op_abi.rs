// SPDX-License-Identifier: Apache-2.0
//! CUDA's per-opcode ABI as DECLARED DATA — the counterpart to metal's `op_abi`.
//!
//! Target-ABI facts are small const tables the shared passes take as *input* — never logic. Here:
//! how an op's kernel treats its buffers ([`cuda_colour_rule`]), which steps CUDA fuses
//! ([`CUDA_FUSIONS`]), which result rows it samples ([`CUDA_SAMPLE_ROWS`]), and how wide a graph
//! node is ([`CUDA_COLUMN_BLOCK`]).
//!
//! ⭐ EVERY ENTRY IS THE CONSERVATIVE ONE. Each fact is the one that is correct for ANY kernel a
//! later issue writes: every op writes a fresh buffer and reads its inputs across the step, no row
//! is sampled, and the one fusion is the `SiluMul` fold, whose kernel exists. That costs arena
//! space and launches; an opcode issue wins them back one row at a time, by changing only its own
//! op's row.

use std::num::NonZeroU32;

use scratchy_subtile::handoff::SourceBinding;
use scratchy_subtile::ops::SubOpKind;
use scratchy_subtile::sample_rows::SampleRowsFacts;
use scratchy_subtile::subtile_ir::{EwKind, OpStage, RopeForm, SubOp};
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
/// No row is in place and none is off the arena: no CUDA kernel writes over an operand yet, and
/// CUDA has no op scratch of its own (metal keeps MoE routing in its `moe_write` scratch), so its
/// MoE ops are coloured in the arena like everything else.
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
        | L::RouterLogits { .. }
        | L::RouteSoftmax
        | L::RouteSigmoidBias { .. }
        | L::RouteBias { .. }
        | L::RouteArgsort
        | L::RouteTopK { .. }
        | L::RouteGatherScores
        | L::RouteScale { .. }
        | L::RouteRenorm
        | L::RouteExpertScale { .. }
        | L::ExpertSort { .. }
        | L::ExpertMatmul { .. }
        | L::ExpertGatedAct { .. }
        | L::ExpertAct { .. }
        | L::ExpertUnsort
        | L::ExpertCombine { .. }
        | L::KvEncode { .. }
        | L::KvStage { .. }
        | L::RotateRows { .. }
        | L::AttnPackedKv
        | L::SampleRowsGather
        | L::SampleRowsScatter
        | L::AllRowsMatmul => FRESH.reads_across(),
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
}

/// CUDA's fusions, in the order the shared fold pass applies them: one sweep of one pattern, the
/// `Silu` a `Mul` reads as its operand 0 folded into that `Mul` (metal's `Gated` row, with its
/// `Silu` activation only).
///
/// `fused` repeats `SiluMul`: CUDA's `ModelFoldFacts` never sets `fold_projections`, so the shared
/// pass always picks `split`, and the gate and up GEMMs stay launches of their own. A `Gated` fold
/// with its projections folded in is refused at lowering (`CanonicalRefusal::FusionShape`); one
/// kernel for the gate and up GEMMs is a later GEMM-epilogue issue.
pub const CUDA_FUSIONS: FusionTable<CudaFusion> = FusionTable {
    counted: &[],
    sweeps: &[&[FoldPattern::Gated {
        mul: SubOpKind::Mul,
        activations: &[GatedKernel {
            activation: SubOpKind::Silu,
            split: CudaFusion::SiluMul,
            fused: CudaFusion::SiluMul,
        }],
    }]],
    rows: None,
};

// ── Tiling ──────────────────────────────────────────────────────────

/// How many columns one graph node covers: all of them. Tiling is a target fact; spyre tiles to
/// 128-column pages because its substrate requires it, while CUDA's kernels — like metal's — take
/// a whole `RmsNorm` or a whole matmul, so the graph is built from whole ops.
pub const CUDA_COLUMN_BLOCK: NonZeroU32 = NonZeroU32::MAX;

#[cfg(test)]
mod tests {
    use super::*;

    use scratchy_subtile::lower::GemmWeight;
    use scratchy_subtile::subtile_ir::{GainConvention, RowScale};
    use scratchy_subtile::tape_colouring::{Outputs, Reads, Residency};

    /// The tables declare exactly the one fused kernel CUDA has, `SiluMul`, and no sliced
    /// `lm_head`: any other pattern, or a sampled weight, would ask the shared passes for a kernel
    /// CUDA does not have.
    #[test]
    fn cuda_declares_only_the_silu_mul_fold_and_no_sampled_rows() {
        assert!(CUDA_FUSIONS.counted.is_empty());
        assert!(CUDA_FUSIONS.rows.is_none());
        assert!(CUDA_SAMPLE_ROWS.weights.is_empty());
        let [sweep] = CUDA_FUSIONS.sweeps else {
            panic!("expected one sweep, got {}", CUDA_FUSIONS.sweeps.len());
        };
        let [FoldPattern::Gated { mul, activations }] = sweep else {
            panic!("expected one Gated pattern, got {sweep:?}");
        };
        assert_eq!(*mul, SubOpKind::Mul);
        let [act] = activations else {
            panic!("expected one gated activation, got {activations:?}");
        };
        assert_eq!(act.activation, SubOpKind::Silu);
        assert_eq!(act.split, CudaFusion::SiluMul);
        assert_eq!(act.fused, CudaFusion::SiluMul);
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
        let cases: [(SubOp, ColourRule); 10] = [
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
