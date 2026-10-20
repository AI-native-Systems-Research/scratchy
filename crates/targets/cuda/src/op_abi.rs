// SPDX-License-Identifier: Apache-2.0
//! CUDA's per-opcode ABI as DECLARED DATA — the counterpart to metal's `op_abi`.
//!
//! Target-ABI facts are small const tables the shared passes take as *input* — never logic. Here:
//! how an op's kernel treats its buffers ([`cuda_colour_rule`]), which steps CUDA fuses
//! ([`CUDA_FUSIONS`]), which result rows it samples ([`CUDA_SAMPLE_ROWS`]), and how wide a graph
//! node is ([`CUDA_COLUMN_BLOCK`]).
//!
//! ⭐ EVERY ENTRY IS THE CONSERVATIVE ONE. No CUDA kernel exists yet, so each fact is the one that
//! is correct for ANY kernel a later issue writes: every op writes a fresh buffer and reads its
//! inputs across the step, nothing fuses, no row is sampled. That costs arena space and launches;
//! an opcode issue wins them back one row at a time, by changing only its own op's row.

use std::num::NonZeroU32;

use scratchy_subtile::handoff::SourceBinding;
use scratchy_subtile::sample_rows::SampleRowsFacts;
use scratchy_subtile::subtile_ir::{EwKind, OpStage, RopeForm, SubOp};
use scratchy_subtile::tape_colouring::{ColourFacts, ColourRule, OutputAlias};
use scratchy_subtile::tape_folding::FusionTable;
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

/// The fused kernel each CUDA fold becomes. UNINHABITED: CUDA has no fused kernel yet; a fusion
/// issue adds its variant together with its pattern in [`CUDA_FUSIONS`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CudaFusion {}

/// CUDA's fusions, in the order the shared fold pass applies them: none, so every step is its own
/// launch.
pub const CUDA_FUSIONS: FusionTable<CudaFusion> = FusionTable {
    counted: &[],
    sweeps: &[],
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

    /// No fused kernel and no sliced `lm_head` exist yet, so the tables declare none: a pattern or
    /// a sampled weight here would ask the shared passes for a kernel CUDA does not have.
    #[test]
    fn cuda_declares_no_fusion_and_no_sampled_rows() {
        assert!(CUDA_FUSIONS.counted.is_empty());
        assert!(CUDA_FUSIONS.sweeps.is_empty());
        assert!(CUDA_FUSIONS.rows.is_none());
        assert!(CUDA_SAMPLE_ROWS.weights.is_empty());
    }
}
