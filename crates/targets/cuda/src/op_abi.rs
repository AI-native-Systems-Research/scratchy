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
//! only its own op's row. The soft-cap has: its kernel writes over its input.

use std::num::NonZeroU32;

use scratchy_subtile::handoff::SourceBinding;
use scratchy_subtile::lower::GemmWeightKind;
use scratchy_subtile::ops::SubOpKind;
use scratchy_subtile::sample_rows::{ResultRows, SampleRowsFacts};
use scratchy_subtile::subtile_ir::{EwKind, ExpertBundle, ExpertProj, OpStage, RopeForm, SubOp};
use scratchy_subtile::tape_colouring::{ColourFacts, ColourRule, OutputAlias};
use scratchy_subtile::tape_folding::{
    ColsBand, CountedOperands, FoldPattern, FusionTable, GatedKernel,
};
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
/// One row is in place: the soft-cap writes over its operand (`pointwise_bf16.cu`'s `out == x`
/// contract); no other CUDA kernel writes over an operand. The routed half of a MoE block is
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
        // The soft-cap writes over operand 0: `pointwise_bf16.cu` allows `out == x` exactly (each
        // element is read and written by the same thread). Metal's row, with pointwise reads.
        L::TanhSoftCap { .. } => FRESH.output(A::Operand(OperandIx(0))),
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
        | L::RouteSigmoidBias { .. }
        | L::RouteBias { .. }
        | L::ExpertAct { .. }
        | L::ExpertCombine { .. }
        | L::KvEncode { .. }
        | L::KvStage { .. }
        | L::RotateRows { .. }
        | L::AttnPackedKv
        | L::SampleRowsGather
        | L::SampleRowsScatter
        | L::AllRowsMatmul => FRESH.reads_across(),
        // Spyre's token draw and chained head (the spyre bake's mints; CUDA never lowers them):
        // each writes its own fresh buffer, metal's row.
        L::Sample { .. } | L::EmbedRowGather { .. } | L::ChainStage { .. } => FRESH,
        // The collectives (tensor parallelism's; CUDA runs one rank, so the analysis inserts none
        // here): each writes the whole value into a buffer of its own, metal's row.
        L::AllReduce | L::AllGather { .. } => FRESH,
        // A DFlash2 drafter's ops (#1015; metal only, CUDA never lowers them): metal's rows. The
        // convolution and attention read across; the capture, the context commit, the window
        // store and load, and the proposals write over their operand 0.
        L::DraftConv { .. } | L::DraftAttention { .. } => FRESH.reads_across(),
        L::CaptureHidden { .. }
        | L::DraftContextKv { .. }
        | L::DraftWindowStore
        | L::DraftWindowLoad
        | L::DraftSelect => FRESH.output(A::Operand(OperandIx(0))),
        // The routed half of a MoE block: in the MoE scratch ([`cuda_moe_write`]), not a colour.
        // Gemma's router norm is `[m, hidden]` bf16 like a colour, but only its router reads it,
        // so it lives in the scratch's router-norm region with the rest of the routing.
        L::RouterNorm { .. }
        | L::RouteScale { .. }
        | L::RouteExpertScale { .. }
        | L::RouterLogits { .. }
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

/// The rows of a CUDA tape's result its caller reads: every row, in order. The worker downloads
/// the whole result colour (no row is sampled to the front), so the shared sample-rows pass leaves
/// the canonical as it is.
pub const CUDA_RESULT_ROWS: ResultRows = ResultRows::Every;

// ── The fusions ─────────────────────────────────────────────────────

/// The fused kernel each CUDA fold becomes. A fusion issue adds its variant together with its
/// pattern in [`CUDA_FUSIONS`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CudaFusion {
    /// `silu(gate) · up` in one launch (`kernels/silu_mul_bf16.cu`): the SwiGLU MLP's `Silu`
    /// folded into the `Mul` it gates.
    SiluMul,
    /// `gelu_tanh(gate) · up` in one launch: the GeGLU MLP's `Gelu` folded into the `Mul` it
    /// gates, and Gemma's expert activation. Its kernel is #885's 13b.
    GeluMul,
    /// A MoE block's routing in one launch (`kernels/moe_route_bf16.cu`): the softmax over the
    /// router's logits (Qwen) or none (Gemma) and their argsort folded into the top-k; the
    /// gathered scores, their scale, softmax or renorm, and the per-expert scale its epilogues.
    MoeRoute,
    /// Sibling projections off one input in one launch (#1049): the q, k and v projections, a
    /// GDN layer's `in_proj_qkv` and `in_proj_z`, or its `in_proj_a` and `in_proj_b`
    /// (`step::CudaStep::GemmGroup`, `GemmNarrowGroup`, `GemvGroup`).
    SiblingMatmuls,
    /// [`Self::SiblingMatmuls`] at the one-row bucket (FP8 S8 W4b): a narrow-band group (the 35B's
    /// bf16 `in_proj_a` and `in_proj_b`, n = 32 + 32) launches as the GEMV group
    /// (`step::CudaStep::GemvGroup`), not the narrow GEMM's; every other group as at every bucket.
    SiblingMatvecs,
    /// A residual add and the RMSNorm reading it in one launch (#1049, Qwen's `norm(add(..))`;
    /// `step::CudaStep::AddRmsNorm`): the sum and the norm both written.
    AddRmsNorm,
    /// An RMSNorm and the residual add it alone feeds in one launch (#1049, Gemma's
    /// `add(norm(o), residual)`; `step::CudaStep::NormAdd`).
    NormAdd,
    /// SwiGLU's gate and up projections and `silu(gate) · up` in one launch (#1049,
    /// `step::CudaStep::GateUpAct`): the `Gated` fold with its projections folded in.
    GateUpSiluMul,
    /// GeGLU's, `gelu_tanh(gate) · up` (#1049, `step::CudaStep::GateUpAct`).
    GateUpGeluMul,
    /// A routed expert's gate and up GEMVs and its activation in one launch, at the one-row bucket
    /// (#1049, `step::CudaStep::ExpertGateUpAct`).
    ExpertGateUpAct,
    /// The one-row fp8 GEMV with the residual add it alone feeds as its epilogue (FP8 S8 W4a; the
    /// shared `MatvecEpilogue`): the o, GDN-out and down projections write `residual + out`.
    GemvFp8Residual,
    /// [`Self::GemvFp8Residual`] on the bf16 GEMV (the 35B's GDN `out_proj`, kept bf16).
    GemvResidual,
    /// The query rotation folded into its layer's rope and append (the shared `Sibling`, FP8 S8
    /// W4c): one step computes the q rope, the k rope and the K/V append.
    RopeAppendSibling,
    /// Gemma's per-head q and k RMSNorms and the V unit norm folded into that step (the shared
    /// `NormedRope`, FP8 S8 W4c): the step reads the raw projections.
    RopeAppendNormed,
}

/// Whether CUDA folds a gated MLP's gate and up projections into its activation
/// (`ModelFoldFacts::fold_projections`), at every bucket (#1049): the gate/up GEMV and GEMM
/// (`step::CudaStep::GateUpAct`) take every dense SwiGLU and GeGLU shape CUDA lowers.
pub const CUDA_FOLD_PROJECTIONS: bool = true;

/// The fusion table of a bucket of `m` rows (#1049): [`CUDA_FUSIONS`] at every bucket, and at the
/// one-row bucket [`CUDA_FUSIONS_ONE_ROW`], which also folds a routed expert's gate and up GEMVs
/// into its activation. A prefill bucket's experts keep their grouped GEMMs (#1050).
pub const fn cuda_fusions(m: u64) -> &'static FusionTable<CudaFusion> {
    if m == 1 {
        &CUDA_FUSIONS_ONE_ROW
    } else {
        &CUDA_FUSIONS
    }
}

/// The output widths CUDA's sibling-matmul fold groups (#1049), one band per grouped kernel: the
/// tiled GEMM's and the GEMV's (multiples of `step::GEMM_BF16_TILE`'s 256 columns: q, k, v,
/// `in_proj_qkv`, `in_proj_z`), and the narrow GEMM's (`in_proj_a` and `in_proj_b`, multiples of
/// 8 at most `step::GEMM_NARROW_BF16_MAX_N`). A width in neither (Gemma's 64-column ones, a
/// one-column gate) never groups.
pub const CUDA_SIBLING_BANDS: &[ColsBand] = &[
    ColsBand {
        multiple: crate::step::GEMM_BF16_TILE.1,
        max: u32::MAX,
    },
    ColsBand {
        multiple: crate::step::GEMM_NARROW_BF16_MULTIPLE,
        max: crate::step::GEMM_NARROW_BF16_MAX_N,
    },
];

/// CUDA's fusions, in the order the shared fold pass applies them: one sweep of five patterns.
/// Up to three dense matmuls of one band off one input fold into one grouped launch (#1049), but
/// for a gate or up projection, which the gated row leaves its own. The `Silu` or `Gelu` a `Mul`
/// reads as its operand 0 folds into that `Mul` (metal's `Gated` row). An RMSNorm only an add
/// reads folds into that add (metal's `NormAdd`), and an add folds into the first norm reading it
/// (metal's `ResidualNorm`) when no `NormAdd` took it: the sweep meets an add before its readers,
/// so Gemma's three norm-then-add pairs a layer take `NormAdd`, and Qwen's add-then-norm pairs
/// `ResidualNorm` (#1049).
/// A MoE block's top-k folds in the softmax (Qwen's softmax-first routing) and the argsort before
/// it, and takes as its epilogues the gathered scores, Gemma's `hidden^-0.5` scale (tail stage 0),
/// a softmax over the picks (Gemma) or their renorm (Qwen) (stage 1), and Gemma's per-expert scale
/// (stage 2): metal's `Route` row, without its sigmoid and bias routings and its last scale.
///
/// Under [`CUDA_FOLD_PROJECTIONS`] the shared pass picks `fused`: the gate and up GEMMs fold into
/// the activation's launch (#1049). `split` is the activation alone, the kernel a `Gated` fold
/// whose projections stay apart lowers to.
pub const CUDA_FUSIONS: FusionTable<CudaFusion> = FusionTable {
    counted: &[],
    sweeps: &[CUDA_SWEEP],
    rows: None,
};

/// The one-row bucket's table ([`cuda_fusions`]), three sweeps (FP8 S8 W4a). The shared pass visits
/// the steps in tape order, and a fold takes only steps no earlier fold took, so the order is
/// forced:
/// 1. [`CUDA_SWEEP_ONE_ROW_GROUPS`]: the sibling matmuls group, and a projection before a residual
///    add takes the add as its epilogue (before `ResidualNorm` could take it into a norm);
/// 2. [`CUDA_SWEEP`], every bucket's: its siblings row finds the groups taken, its `Gated`,
///    `NormAdd`, `ResidualNorm` and `Route` rows as at every bucket;
/// 3. a routed expert's gate and up GEMVs folded into its activation (metal's `ExpertActivation`
///    row, #1049).
///
/// No norm folds into its matvecs (the shared `NormedMatvecs`): measured on the H100 at 38384fc6, a
/// GEMV that normalises its own row does it in every CTA, 23-63 µs a launch at the 27B shapes, more
/// than the norm's launch it saves (/tmp/d-w4.md §1; the user's decision of 2026-11-09).
pub const CUDA_FUSIONS_ONE_ROW: FusionTable<CudaFusion> = FusionTable {
    // A decode attention's operands from 1 on are no consumer edges (Metal's row): its new K and V
    // (operands 3 and 4) name what its layer's append wrote, the cache its kernel reads. So the V
    // unit norm has one consumer, the append, and `NormedRope` can take it.
    counted: &[CountedOperands {
        kind: SubOpKind::AttnDecode,
        before: OperandIx(1),
    }],
    sweeps: &[
        CUDA_SWEEP_ONE_ROW_GROUPS,
        CUDA_SWEEP,
        CUDA_SWEEP_ONE_ROW_ROPE,
        &[FoldPattern::ExpertActivation {
            act: SubOpKind::ExpertGatedAct,
            matmul: SubOpKind::ExpertMatmul,
            kernel: CudaFusion::ExpertGateUpAct,
        }],
    ],
    rows: None,
};

/// Up to three dense matmuls of one band off one input in one grouped launch (#1049): every
/// bucket's first row.
const F_SIBLINGS: FoldPattern<CudaFusion> = FoldPattern::SiblingMatmuls {
    matmul: SubOpKind::MatmulTile,
    max: 3,
    bands: CUDA_SIBLING_BANDS,
    gated: &[SubOpKind::Silu, SubOpKind::Gelu, SubOpKind::Mul],
    kernel: CudaFusion::SiblingMatmuls,
};

/// [`F_SIBLINGS`]'s groups at the one-row bucket's first row (FP8 S8 W4b): the same bands, cap and
/// exclusions, so the same groups form (the 35B's GDN norm feeds qkv + z and a + b, two groups of
/// two), and each lowers as GEMVs ([`CudaFusion::SiblingMatvecs`]). At W0 the bf16 a + b narrow
/// group took 12.6 µs a launch, 30 a token.
const F_SIBLINGS_ONE_ROW: FoldPattern<CudaFusion> = FoldPattern::SiblingMatmuls {
    matmul: SubOpKind::MatmulTile,
    max: 3,
    bands: CUDA_SIBLING_BANDS,
    gated: &[SubOpKind::Silu, SubOpKind::Gelu, SubOpKind::Mul],
    kernel: CudaFusion::SiblingMatvecs,
};

/// The one-row bucket's first sweep (FP8 S8 W4a): the siblings group (as GEMVs, W4b), and the
/// projection a residual
/// add alone reads (its operand 0, the residual its operand 1) takes the add as its epilogue, on
/// the fp8 GEMV and the bf16 GEMV (the 35B's GDN `out_proj`). Metal's `F_COMBINE_EPILOGUE` is not
/// a row: the 35B's tape computes its routed combine before the shared expert, so the combine's
/// gate scale is not ready there and the row would never fold; its add stays a launch.
const CUDA_SWEEP_ONE_ROW_GROUPS: &[FoldPattern<CudaFusion>] = &[
    F_SIBLINGS_ONE_ROW,
    FoldPattern::MatvecEpilogue {
        anchor: SubOpKind::MatmulTile,
        weights: Some(GemmWeightKind::Fp8Dynamic),
        bias: None,
        scale: None,
        gate_scale: None,
        add: SubOpKind::Add,
        gated: &[SubOpKind::Silu, SubOpKind::Gelu, SubOpKind::Mul],
        kernel: CudaFusion::GemvFp8Residual,
    },
    FoldPattern::MatvecEpilogue {
        anchor: SubOpKind::MatmulTile,
        weights: Some(GemmWeightKind::Dense),
        bias: None,
        scale: None,
        gate_scale: None,
        add: SubOpKind::Add,
        gated: &[SubOpKind::Silu, SubOpKind::Gelu, SubOpKind::Mul],
        kernel: CudaFusion::GemvResidual,
    },
];

/// The one-row bucket's rope sweep (FP8 S8 W4c, the user's decisions of 2026-11-09): the query
/// rotation folds into its layer's rope and append (`Sibling`, every model), then Gemma's per-head
/// q and k norms and V unit norm into that step (`NormedRope`, Metal's row: it fires only where V
/// is a single-consumer unit norm, so on Gemma's sliding layers; its global layers' V reads the
/// same per-head K projection as the k norm, which the pattern does not take).
const CUDA_SWEEP_ONE_ROW_ROPE: &[FoldPattern<CudaFusion>] = &[
    FoldPattern::Sibling {
        driver: SubOpKind::RopeAppend,
        sibling: SubOpKind::RopeRotate,
        kernel: CudaFusion::RopeAppendSibling,
    },
    FoldPattern::NormedRope {
        rope: SubOpKind::RopeAppend,
        gain_norm: SubOpKind::RmsNorm,
        unit_norm: SubOpKind::RmsNormUnit,
        kernel: CudaFusion::RopeAppendNormed,
    },
];

/// The sweep every CUDA bucket folds by ([`CUDA_FUSIONS`]).
const CUDA_SWEEP: &[FoldPattern<CudaFusion>] = &[
    F_SIBLINGS,
    FoldPattern::Gated {
        mul: SubOpKind::Mul,
        activations: &[
            GatedKernel {
                activation: SubOpKind::Silu,
                split: CudaFusion::SiluMul,
                fused: CudaFusion::GateUpSiluMul,
            },
            GatedKernel {
                activation: SubOpKind::Gelu,
                split: CudaFusion::GeluMul,
                fused: CudaFusion::GateUpGeluMul,
            },
        ],
    },
    FoldPattern::NormAdd {
        add: SubOpKind::Add,
        norm: SubOpKind::RmsNorm,
        kernel: CudaFusion::NormAdd,
    },
    FoldPattern::ResidualNorm {
        norm: SubOpKind::RmsNorm,
        add: SubOpKind::Add,
        kernel: CudaFusion::AddRmsNorm,
    },
    FoldPattern::Route {
        top_k: SubOpKind::RouteTopK,
        sort: SubOpKind::RouteArgsort,
        pre: &[SubOpKind::RouteSoftmax],
        gather: SubOpKind::RouteGatherScores,
        tail: &[
            &[SubOpKind::RouteScale],
            &[SubOpKind::RouteSoftmax, SubOpKind::RouteRenorm],
            &[SubOpKind::RouteExpertScale],
            &[],
        ],
        kernel: CudaFusion::MoeRoute,
    },
];

// ── The MoE block ───────────────────────────────────────────────────

/// A region of the tape's MoE scratch (`crate::tape::MoeTape`): CUDA's own, because its regions
/// are not metal's (`MoeRegion`): i32 indices, f32 scores, expert runs padded to
/// [`CUDA_EXPERT_TILE_ROWS`] rows, and Gemma's router norm.
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
    /// `[m, hidden]` bf16, the router's own pre-norm (Gemma's `RouterNorm`), which only the
    /// router reads.
    RouterNorm,
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
/// step for (the sigmoid and bias routings, the ungated expert) or that only rides the
/// `MoeRoute` fold as an epilogue (the scales).
pub fn cuda_moe_write<F: RopeForm, S: OpStage>(op: &SubOp<F, S>) -> Option<CudaMoeWrite> {
    use CudaMoeRegion as R;
    use CudaMoeWrite as W;
    use SubOp as L;
    Some(match op {
        L::RouterNorm { .. } => W::Region(R::RouterNorm),
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
/// shared-fused bundle (Qwen: the router, then every expert's fused gate and up, then every
/// expert's down) and Gemma's dense bf16 SwitchGLU bank (the same three tensors back to back;
/// its router's two gains are weights of their own). Every other bundle is refused at expansion.
///
/// Metal's speed gates have no counterpart here, on purpose: `METAL_GROUPED_PAIRS_PER_EXPERT`,
/// `METAL_SORTED_PAIRS_PER_EXPERT`, `METAL_SHARED_EXPERTS_FROM` and
/// `METAL_SHARED_EXPERTS_MAX_PAIRS` choose between kernels by speed, or refine the gathered mode,
/// and speed is not yet CUDA's concern: it has one kernel per mode.
pub const CUDA_GROUPED_EXPERTS: &[ExpertBundle] =
    &[ExpertBundle::SharedFused, ExpertBundle::SwitchGlu];

/// The expert counts CUDA's MoE steps take: Gemma 4 26B-A4B's 128 and Qwen3.6-35B-A3B's 256.
/// Every other count is refused at expansion (`MoeShapeRule::Experts`). The 128-expert kernels
/// are #885's 13b.
pub const MOE_EXPERT_COUNTS: &[u32] = &[128, 256];

/// The (token, pick) pairs of a bucket from which its tape sorts the pairs by expert and runs the
/// grouped GEMM; under it each pair's expert GEMV reads the pair's token row directly (gathered).
/// Metal's value; with CUDA's buckets {1, 512, …} only the one-row bucket gathers.
pub const CUDA_SORTED_PAIRS: u32 = 64;

/// Each expert's run of sorted pairs is padded to a multiple of this many rows: the grouped GEMM's
/// row tile (`step::GEMM_BF16_TILE.0`), so every tile belongs to exactly one expert.
pub const CUDA_EXPERT_TILE_ROWS: u32 = 128;

const _: () = assert!(CUDA_EXPERT_TILE_ROWS == crate::step::GEMM_BF16_TILE.0);

// ── Tiling ──────────────────────────────────────────────────────────

/// The archs whose tapes CUDA lowers with the 64-column GEMM tile (`step::GEMM_COL64_TILE`):
/// Gemma 4 26B-A4B, whose router (128), expert (704) and dense MLP (2112) widths are multiples of
/// 64 and not of 256. Every other arch's GEMMs take the 256-column tile only, so their rules and
/// refusals are the tiled GEMM's.
pub const CUDA_GEMM_COL64_ARCHS: &[&str] = &["gemma4-moe"];

/// How many columns one graph node covers: all of them. Tiling is a target fact; spyre tiles to
/// 128-column pages because its substrate requires it, while CUDA's kernels — like metal's — take
/// a whole `RmsNorm` or a whole matmul, so the graph is built from whole ops.
pub const CUDA_COLUMN_BLOCK: NonZeroU32 = NonZeroU32::MAX;

#[cfg(test)]
mod tests {
    use super::*;

    use scratchy_ir::model_facts::HeadDim;
    use scratchy_subtile::lower::GemmWeight;
    use scratchy_subtile::subtile_ir::{GainConvention, RowScale};
    use scratchy_subtile::tape_colouring::{Outputs, Reads, Residency};

    /// The tables declare exactly CUDA's fused kernels, the grouped GEMMs (`SiblingMatmuls`, #1049),
    /// `SiluMul`, `GeluMul`, their gate/up forms, the add-norm folds, `MoeRoute`, and at one row
    /// the experts' gate/up (#1049), and no sliced `lm_head`: any other pattern, or a sampled
    /// weight, would ask the shared passes for a kernel CUDA does not have. The sibling fold groups
    /// up to three dense matmuls in `CUDA_SIBLING_BANDS`, leaving a gate or up projection (a matmul
    /// a `Silu`, `Gelu` or `Mul` reads) to the gated fold, which folds it in under
    /// `CUDA_FOLD_PROJECTIONS`. `NormAdd` comes before `ResidualNorm`. The routing fold takes the
    /// softmax as its one pre (Qwen), and as its tail Gemma's scale (stage 0), a softmax or a renorm
    /// (stage 1) and Gemma's per-expert scale (stage 2), nothing at stage 3. The one-row table is
    /// the every-bucket sweep, then the experts' `ExpertActivation`.
    #[test]
    fn cuda_declares_its_folds_per_bucket_and_no_sampled_rows() {
        for table in [&CUDA_FUSIONS, &CUDA_FUSIONS_ONE_ROW] {
            assert!(table.rows.is_none());
        }
        assert!(CUDA_FUSIONS.counted.is_empty());
        // The one-row table counts a decode attention's operand 0 alone (FP8 S8 W4c, Metal's row).
        let [
            CountedOperands {
                kind: SubOpKind::AttnDecode,
                before: OperandIx(1),
            },
        ] = CUDA_FUSIONS_ONE_ROW.counted
        else {
            panic!("the one-row table's counted operands");
        };
        assert!(CUDA_SAMPLE_ROWS.weights.is_empty());
        // The one-row table has the residual epilogues' sweep, the rope sweep (FP8 S8 W4a, W4c) and
        // the experts' sweep; every other bucket's has the one sweep.
        assert_eq!(cuda_fusions(1).sweeps.len(), 4);
        for m in [2, 512, 4096] {
            assert_eq!(cuda_fusions(m).sweeps.len(), 1, "m={m}");
        }
        let [sweep] = CUDA_FUSIONS.sweeps else {
            panic!("expected one sweep, got {}", CUDA_FUSIONS.sweeps.len());
        };
        let [groups, first, rope, experts] = CUDA_FUSIONS_ONE_ROW.sweeps else {
            panic!(
                "expected four one-row sweeps, got {}",
                CUDA_FUSIONS_ONE_ROW.sweeps.len()
            );
        };
        assert!(
            std::ptr::eq(*groups, CUDA_SWEEP_ONE_ROW_GROUPS),
            "the one-row table's first sweep is its residual epilogues'"
        );
        assert!(
            std::ptr::eq(*first, *sweep) && std::ptr::eq(*sweep, CUDA_SWEEP),
            "the one-row table's second sweep is the sweep"
        );
        assert!(
            std::ptr::eq(*rope, CUDA_SWEEP_ONE_ROW_ROPE),
            "the one-row table's third sweep is the rope's"
        );
        assert!(
            matches!(
                (CUDA_SWEEP[0], F_SIBLINGS),
                (
                    FoldPattern::SiblingMatmuls {
                        max: 3,
                        kernel: CudaFusion::SiblingMatmuls,
                        ..
                    },
                    FoldPattern::SiblingMatmuls {
                        max: 3,
                        kernel: CudaFusion::SiblingMatmuls,
                        ..
                    },
                )
            ) && matches!(
                CUDA_SWEEP_ONE_ROW_GROUPS[0],
                FoldPattern::SiblingMatmuls { .. }
            ),
            "both tables group the siblings by the one row"
        );
        // The one-row row is every bucket's but for its kernel (FP8 S8 W4b): the same groups form.
        let (
            FoldPattern::SiblingMatmuls {
                matmul,
                max,
                bands,
                gated,
                kernel: CudaFusion::SiblingMatvecs,
            },
            FoldPattern::SiblingMatmuls {
                matmul: every_matmul,
                max: every_max,
                bands: every_bands,
                gated: every_gated,
                kernel: CudaFusion::SiblingMatmuls,
            },
        ) = (CUDA_SWEEP_ONE_ROW_GROUPS[0], F_SIBLINGS)
        else {
            panic!("the one-row siblings row is the GEMV groups'");
        };
        assert_eq!(
            (matmul, max, bands, gated),
            (every_matmul, every_max, every_bands, every_gated)
        );
        let [
            FoldPattern::ExpertActivation {
                act,
                matmul: expert_matmul,
                kernel: expert_kernel,
            },
        ] = experts
        else {
            panic!("expected one ExpertActivation pattern, got {experts:?}");
        };
        assert_eq!(
            (*act, *expert_matmul, *expert_kernel),
            (
                SubOpKind::ExpertGatedAct,
                SubOpKind::ExpertMatmul,
                CudaFusion::ExpertGateUpAct
            )
        );
        let [
            FoldPattern::SiblingMatmuls {
                matmul,
                max,
                bands,
                gated,
                kernel: siblings,
            },
            FoldPattern::Gated { mul, activations },
            FoldPattern::NormAdd {
                add: na_add,
                norm: na_norm,
                kernel: na_kernel,
            },
            FoldPattern::ResidualNorm {
                norm: rn_norm,
                add: rn_add,
                kernel: rn_kernel,
            },
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
            panic!(
                "expected one SiblingMatmuls, Gated, NormAdd, ResidualNorm and Route pattern, got \
                 {sweep:?}"
            );
        };
        assert_eq!(
            (*na_add, *na_norm, *na_kernel),
            (SubOpKind::Add, SubOpKind::RmsNorm, CudaFusion::NormAdd)
        );
        assert_eq!(
            (*rn_norm, *rn_add, *rn_kernel),
            (SubOpKind::RmsNorm, SubOpKind::Add, CudaFusion::AddRmsNorm)
        );
        assert_eq!((*matmul, *max), (SubOpKind::MatmulTile, 3));
        assert_eq!(
            *bands,
            [
                ColsBand {
                    multiple: 256,
                    max: u32::MAX
                },
                ColsBand {
                    multiple: 8,
                    max: 64
                },
            ]
        );
        assert_eq!(*gated, [SubOpKind::Silu, SubOpKind::Gelu, SubOpKind::Mul]);
        assert_eq!(*siblings, CudaFusion::SiblingMatmuls);
        assert_eq!(*mul, SubOpKind::Mul);
        let [silu, gelu] = activations else {
            panic!("expected two gated activations, got {activations:?}");
        };
        assert_eq!(silu.activation, SubOpKind::Silu);
        assert_eq!(silu.split, CudaFusion::SiluMul);
        assert_eq!(silu.fused, CudaFusion::GateUpSiluMul);
        assert_eq!(gelu.activation, SubOpKind::Gelu);
        assert_eq!(gelu.split, CudaFusion::GeluMul);
        assert_eq!(gelu.fused, CudaFusion::GateUpGeluMul);
        assert_eq!(
            (*top_k, *sort, *gather),
            (
                SubOpKind::RouteTopK,
                SubOpKind::RouteArgsort,
                SubOpKind::RouteGatherScores
            )
        );
        assert_eq!(*pre, [SubOpKind::RouteSoftmax]);
        assert_eq!(
            **tail,
            [
                &[SubOpKind::RouteScale][..],
                &[SubOpKind::RouteSoftmax, SubOpKind::RouteRenorm],
                &[SubOpKind::RouteExpertScale],
                &[],
            ]
        );
        assert_eq!(*kernel, CudaFusion::MoeRoute);
    }

    /// The 64-column GEMM is Gemma 4's alone.
    #[test]
    fn the_64_column_gemm_archs() {
        assert_eq!(CUDA_GEMM_COL64_ARCHS, ["gemma4-moe"]);
    }

    /// The MoE facts: two dense bundles with a grouped GEMM, at 128 or 256 experts, sorted from
    /// metal's 64 pairs, expert runs padded to the GEMM's 128-row tile.
    #[test]
    fn the_moe_facts() {
        assert_eq!(
            CUDA_GROUPED_EXPERTS,
            [ExpertBundle::SharedFused, ExpertBundle::SwitchGlu]
        );
        assert_eq!(MOE_EXPERT_COUNTS, [128, 256]);
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
        let gemma = RouterBundle::Gemma;
        let gemma_cases: [(SubOp, Option<CudaMoeWrite>); 3] = [
            (
                SubOp::RouterNorm {
                    eps: 1e-6,
                    router: gemma,
                },
                Some(W::Region(R::RouterNorm)),
            ),
            (SubOp::RouteExpertScale { router: gemma }, None),
            (
                SubOp::ExpertGatedAct {
                    act: GatedAct::Gelu,
                },
                Some(W::Region(R::Act)),
            ),
        ];
        for (op, want) in gemma_cases {
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
        let gemma = scratchy_subtile::subtile_ir::RouterBundle::Gemma;
        let cases = cases.into_iter().chain([
            (SubOp::ScalarMul { scale: 53.0 }, fresh_across(Outputs::One)),
            (SubOp::RmsNormUnit { eps: 1e-6 }, fresh_across(Outputs::One)),
            (SubOp::ScalarWeightMul, fresh_across(Outputs::One)),
            (
                SubOp::TanhSoftCap { cap: 30.0 },
                ColourRule {
                    output: OutputAlias::Operand(OperandIx(0)),
                    ..ColourRule::FRESH
                },
            ),
            (SubOp::Elementwise(EwKind::Gelu), fresh_across(Outputs::One)),
            (
                SubOp::RouterNorm {
                    eps: 1e-6,
                    router: gemma,
                },
                off_arena,
            ),
            (SubOp::RouteScale { scale: 0.0188 }, off_arena),
            (SubOp::RouteExpertScale { router: gemma }, off_arena),
        ]);
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
