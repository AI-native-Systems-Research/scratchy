// SPDX-License-Identifier: Apache-2.0
//! Metal's per-opcode ABI as DECLARED DATA.
//!
//! Target-ABI facts (in-place kernels, write bindings, barrier
//! classes) are small const tables the shared passes take as *input*
//! — never logic. Here: how an op's kernel treats its buffers
//! ([`metal_colour_rule`]), and which steps metal fuses ([`METAL_FUSIONS`]).
//!
//! That fact has two consumers today and they must agree or the
//! output is garbage: the shared tape colourer (an in-place op's
//! output and its aliased operand need ONE color) and the emitter arm
//! (which passes the same slot as both input and output). Stating it
//! once removes the class of bug where a kernel is made in-place in
//! one place and not the other. Contextual aliasing (fold-dependent,
//! value-dependent) is declared here as a pattern KIND; the shared
//! colourer evaluates it.

use scratchy_ir::{BiasStorage, KvOffset, KvOffsets};
use scratchy_subtile::handoff::{SourceBinding, WeightKind, accessor_slot};
use scratchy_subtile::kv_codec::{
    After, Before, CodecAround, CodecGuard, CodecHeadDims, GuardGates, Guarded, KvCodecFacts,
};
use scratchy_subtile::lower::GemmWeightKind;
use scratchy_subtile::ops::SubOpKind;
use scratchy_subtile::sample_rows::SampleRowsFacts;
use scratchy_subtile::subtile_ir::{
    AttnMask, EwKind, ExpertBundle, ExpertProj, KvOperand, OpStage, RopeForm, RouterBundle, SubOp,
};

use crate::tape::lowered::{RuntimeGate, WeightTensor};
use crate::tape::step::MoeRegion;
use scratchy_subtile::tape_colouring::{ColourFacts, ColourRule, OutputAlias};
use scratchy_subtile::tape_folding::{
    CountedOperands, FoldPattern, FusionTable, GatedKernel, NormReader, RoutedBands, RowFold,
};
use scratchy_subtile::tape_steps::OperandIx;

// ── The KV writer's weight site ─────────────────────────────────────
//
// Two declarations, adjacent: how the step records LAY OUT a `RopeAppend`'s
// weight site, and where the TurboQuant lowering FINDS the biases on it.
// Slot 0 is the rotary table; an operand carrying a projection bias
// ([`KvOffset::LinearBias`]) puts that projection's `LinearLayer` on the
// same site, K's first.

/// A `RopeAppend`'s weight site: `cos_sin`, then each biased operand's
/// projection — `Some` exactly where that operand's offset is `LinearBias`.
pub fn rope_append_weight_site<W>(cos_sin: W, k_bias: Option<W>, v_bias: Option<W>) -> Vec<W> {
    std::iter::once(cos_sin)
        .chain(k_bias)
        .chain(v_bias)
        .collect()
}

/// Where [`rope_append_weight_site`] put the K and V bias projections — their
/// ordinal among the site's `Linear` weights ([`accessor_slot`]), the rule the
/// lowering finds a row's sources by — with the storage each bias is read from.
pub fn rope_append_bias_slots(o: KvOffsets) -> [Option<(BiasStorage, u32)>; 2] {
    let storage = |x: KvOffset| match x {
        KvOffset::LinearBias(s) => Some(s),
        KvOffset::Centered => None,
    };
    let (k, v) = (storage(o.k), storage(o.v));
    let linear = |s: Option<BiasStorage>| s.map(|_| WeightKind::Linear);
    let site = rope_append_weight_site(WeightKind::CosSin, linear(k), linear(v));
    // The site is `[cos_sin, K?, V?]`: K sits right after the rotary table, V after K.
    let k_at = 1;
    let v_at = 1 + usize::from(k.is_some());
    [
        k.map(|s| (s, accessor_slot(&site, k_at))),
        v.map(|s| (s, accessor_slot(&site, v_at))),
    ]
}

/// Metal's colouring facts: the per-op rule table, and the embedded hidden as colour 0 (the
/// step records' head emits the embed at slot 0).
/// The widest bucket metal lays out in wave order (`wave_schedule::wave_order`): a decode-shaped
/// tape — the one-row bucket and the verify-sized ones ([`METAL_VERIFY_ROWS`]) — whose steps are
/// latency-bound; a wider bucket would hold more rows' buffers live at once.
pub const METAL_WAVE_ORDER_ROWS: u64 = METAL_VERIFY_ROWS as u64;

pub const METAL_COLOUR_FACTS: ColourFacts = ColourFacts {
    rule: metal_colour_rule,
    colour_zero: SourceBinding::EmbeddedHidden,
};

/// How each op's metal kernel treats its buffers.
///
/// The match is exhaustive: a new `SubOp` fails to compile here
/// (E0004) until its author states which case it is, rather than
/// silently defaulting to "fresh buffer" and losing a color.
pub fn metal_colour_rule<F: RopeForm, S: OpStage>(op: &SubOp<F, S>) -> ColourRule {
    use EwKind as E;
    use OutputAlias as A;
    use SubOp as L;
    const FRESH: ColourRule = ColourRule::FRESH;
    match op {
        // Metadata-only view over operand 0 — no kernel runs.
        L::Reshape { .. } => FRESH.output(A::Operand(OperandIx(0))).peel_through(0),
        // `add_inplace` mutates the residual (operand 1) and publishes it; the first residual add
        // mutates the embed. Folded into a norm, the norm writes over its delta (operand 0).
        L::Elementwise(E::Add) => FRESH
            .output(A::Operand(OperandIx(1)))
            .pinned_on(1)
            .delta_when_absorbed(0),
        // Elementwise activations rewrite their input buffer.
        L::TanhSoftCap { .. } | L::Elementwise(E::Gelu | E::QuickGelu | E::GeluErf | E::Relu2) => {
            FRESH.output(A::Operand(OperandIx(0)))
        }
        // The fused-norm winner mutates the delta in place (NormDeltaResidual).
        L::RmsNorm { .. } => FRESH.output(A::AbsorbedDelta).peel_through(0),
        L::RmsNormUnit { .. } => FRESH.peel_through(0),
        // Identity multiply is elided; the embed normalizer's `scale_inplace` mutates the embed.
        L::ScalarMul { .. } => FRESH.output(A::UnitScale(OperandIx(0))).pinned_on(0),
        // Rope is in place on the RAW PROJECTION (RopeTripleAlias), which for the fused-norm class
        // (gemma-4 per-head qk-norm) sits behind a Reshape → norm → Reshape chain; aliasing the
        // direct input there put the rope on the norm's color and garbled on device. The unfused
        // rope (qwen3's qk-norm) aliases its direct input; peeling there emitted nothing.
        L::RopeRotate { .. } | L::RopeAppend { .. } => {
            FRESH.output(A::PeeledWhenNormed(OperandIx(0)))
        }
        // The un-rotated writer only reads K and V; its K output IS its K operand.
        L::KvAppend { .. } => FRESH.output(A::Operand(OperandIx(0))),
        L::MatmulTile { .. } | L::VarlenAttention { .. } | L::EncoderAttn { .. } => {
            FRESH.reads_across()
        }
        // Splash's drafter: a convolution reads the row before each row of a block, an attention
        // every row of it, each into a buffer of its own; a capture copies its rows out, a context
        // commit writes the ring and a selection the drafter's runtime outputs, each publishing
        // its operand 0 as it found it.
        L::DraftConv { .. } | L::DraftAttention { .. } => FRESH.reads_across(),
        L::CaptureHidden { .. }
        | L::DraftContextKv { .. }
        | L::DraftWindowStore
        | L::DraftWindowLoad
        | L::DraftSelect => FRESH.output(A::Operand(OperandIx(0))),
        // Two-output ops bind a SECOND buffer (VisionRope's k', GateSplit's gate).
        L::VisionRope => FRESH.reads_across().two_outputs(),
        L::GateSplit { .. } => FRESH.two_outputs(),
        // A fresh buffer (the tile-level decompositions never reach metal's whole-op tape).
        L::ScalarWeightMul
        | L::Elementwise(E::BiasAdd | E::Silu | E::Mul | E::Sub)
        | L::SiluMul
        | L::SumReduce { .. }
        | L::RmsNormReduce { .. }
        | L::RmsNormApply { .. }
        | L::Mean
        | L::GateApply
        | L::Concat { .. }
        | L::GateScale
        | L::GatedDeltaNet
        | L::Mamba2 { .. }
        | L::AttnDecode { .. }
        | L::LoadRows { .. }
        | L::EmbeddingGather { .. }
        | L::ExpertCombine { .. }
        // The greedy pick's idx stick (#603): the kernel reads the widened logits row and the
        // seven staged tables and writes its OWN stick — a fresh buffer, never an alias of any
        // operand. Metal never mints the argmax (the mint is the spyre bake's), but the rule
        // table is exhaustive by law, and the stick's colour is FRESH whichever target runs it.
        | L::ArgMax { .. } => FRESH,
        // The rest of a MoE block lives in the op scratch (`moe_write`), not the arena.
        L::RouterNorm { .. }
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
        | L::ExpertUnsort => FRESH.off_arena(),
        // The codec's packed and staged K/V are runtime stores; a rotation turns its rows in place,
        // and the packed twin writes the output of the attention it replaces.
        L::KvEncode { .. } | L::KvStage { .. } => FRESH.off_arena(),
        L::RotateRows { .. } => FRESH.output(A::Operand(OperandIx(0))),
        L::AttnPackedKv => FRESH.output(A::Operand(OperandIx(1))),
        // The sampled rows move in place — the gather over the matmul's input (the embedded
        // hidden's colour 0 too), the scatter over its output — and the all-rows matmul writes the
        // output it takes over.
        L::SampleRowsGather => FRESH.output(A::Operand(OperandIx(0))).pinned_on(0),
        L::SampleRowsScatter => FRESH.output(A::Operand(OperandIx(0))),
        L::AllRowsMatmul => FRESH.output(A::Operand(OperandIx(1))),
    }
}

// ── The sampled rows ────────────────────────────────────────────────

/// Metal's sampled rows: every result matmul it runs leaves the sampled rows' logits at the front
/// of its output — the MLX-affine one's computed by the lm_head slice's qmv, a dense one's moved
/// there after its GEMM.
pub const METAL_SAMPLE_ROWS: SampleRowsFacts = SampleRowsFacts {
    weights: &[GemmWeightKind::Affine, GemmWeightKind::Dense],
};

// ── The KV codec (TurboQuant) ───────────────────────────────────────

/// Metal's KV codec, TurboQuant: every layer coded on a uniform arch, the global class on a
/// hybrid one, head dims the kernels' widened threadgroup arrays take. A writer's K and V are
/// quantized after it; a decode attention yields to its packed-store twin, any other runs in the
/// codebook's rotated domain (K/V staged, q rotated in, its output rotated back) off decode steps.
pub const METAL_KV_CODEC: KvCodecFacts = {
    use CodecGuard::{Codec, CodecDecode, CodecNotDecode, UnlessCodecDecode};
    const fn g<S>(step: S, guard: CodecGuard) -> Guarded<S> {
        Guarded { step, guard }
    }
    const TWIN: Guarded<After> = g(After::PackedTwin, CodecDecode);
    KvCodecFacts {
        head_dims: CodecHeadDims::PowerOfTwoAtMost(512),
        hybrid_class: AttnMask::Causal,
        after_writer: &[g(KvOperand::K, Codec), g(KvOperand::V, Codec)],
        decode: CodecAround {
            before: &[],
            anchor: UnlessCodecDecode,
            after: &[TWIN],
        },
        prefill: CodecAround {
            before: &[
                g(Before::Stage(KvOperand::K), CodecNotDecode),
                g(Before::Stage(KvOperand::V), CodecNotDecode),
                g(Before::RotateQuery, CodecNotDecode),
            ],
            anchor: UnlessCodecDecode,
            after: &[g(After::RotateOutput, CodecNotDecode), TWIN],
        },
    }
};

/// The most rows a verify-sized bucket holds: a speculative verify step's, each sequence's last
/// token and its drafts (an MTP head's: within its `spec_verify_rows`). Such a bucket's
/// steps are decode-shaped — a few query rows over a whole context — and run decode-shaped
/// kernels: its paged attention is the decode kernel's, one query row per token; a gathered MoE
/// block runs the pairs that share an expert once ([`METAL_SHARED_EXPERTS_FROM`]) while its baked
/// pairs stay few ([`METAL_SHARED_EXPERTS_MAX_PAIRS`]). Larger buckets
/// keep prefill-shaped kernels: a 31-token prompt took 20% longer to its first token at a 32-row
/// verify-sized bucket than at the 64-row prefill one (Qwen3.6-35B-A3B, base M5: 176 vs 147 ms).
///
/// TODO: derive it, and the decoder ladder's verify-sized rungs, from the compiled heads'
/// `spec_verify_rows`: a head budget past 8 rows puts verify steps onto the 64-row
/// prefill-shaped tape.
pub const METAL_VERIFY_ROWS: u32 = 8;

/// Whether a guarded codec step runs at a bake, and under which runtime gate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GuardRun {
    /// Its commands run under the gate (`None`: on every step).
    Gated(Option<RuntimeGate>),
    /// The bake drops it.
    Dropped,
}

/// The runtime gate each codec guard runs under on metal. The codec is the model's, fixed at
/// build (`MetalModelConsts::kv_codec`): a dense model inserts no codec step, so a coded one's
/// guards only ask whether the step is a decode step.
pub const METAL_GUARD_GATES: GuardGates<GuardRun> = GuardGates {
    codec: GuardRun::Gated(None),
    codec_decode: GuardRun::Gated(Some(RuntimeGate::OnlyIfDecodeStep)),
    codec_not_decode: GuardRun::Gated(Some(RuntimeGate::UnlessDecodeStep)),
    unless_codec_decode: GuardRun::Gated(Some(RuntimeGate::UnlessDecodeStep)),
};

/// A verify-sized multi-row bucket's guards ([`METAL_VERIFY_ROWS`]): its coded attention is the
/// packed-store decode twin, one query row per token, on every step; the prefill form, and the
/// staging and rotations around it, are dropped.
pub const METAL_VERIFY_GUARD_GATES: GuardGates<GuardRun> = GuardGates {
    codec: GuardRun::Gated(None),
    codec_decode: GuardRun::Gated(None),
    codec_not_decode: GuardRun::Dropped,
    unless_codec_decode: GuardRun::Dropped,
};

/// The guards' realization at a bucket of `bucket_m` rows.
pub fn metal_guard_gates(bucket_m: u32) -> &'static GuardGates<GuardRun> {
    match bucket_m {
        2..=METAL_VERIFY_ROWS => &METAL_VERIFY_GUARD_GATES,
        _ => &METAL_GUARD_GATES,
    }
}

// `MetalFusion::KvEncoded` folds the codec's encodes into their KV writer, which runs ungated.
const _: () = assert!(
    matches!(METAL_GUARD_GATES.codec, GuardRun::Gated(None))
        && matches!(METAL_VERIFY_GUARD_GATES.codec, GuardRun::Gated(None)),
    "a gated encode cannot fold into its ungated writer"
);

// ── The MoE block ───────────────────────────────────────────────────

/// Where a MoE step writes: a scratch region, over its operand's region (in place), or the arena.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MoeWrite {
    Region(MoeRegion),
    OverOperand(OperandIx),
    Arena,
}

/// Metal's MoE dataflow: where each step of a block writes. Every one of them uses the op scratch.
pub fn moe_write(op: &SubOp) -> Option<MoeWrite> {
    use MoeRegion as R;
    use MoeWrite as W;
    use SubOp as L;
    Some(match op {
        L::RouterNorm { .. } => W::Region(R::RouterNormed),
        L::RouterLogits { .. } => W::Region(R::RouterLogits),
        L::RouteArgsort => W::Region(R::SortedExperts),
        L::RouteTopK { .. } => W::Region(R::TopKIndices),
        L::RouteGatherScores => W::Region(R::TopKScores),
        // Fold-only pre: no command of its own (the biased keys and the unbiased scores cannot
        // share a buffer), but it colours with the logits it rewrites, like the softmax it
        // generalizes.
        L::RouteSigmoidBias { .. } => W::OverOperand(OperandIx(0)),
        // Fold-only pre, gpt-oss's: the biased keys and the gathered scores CAN share a buffer
        // (the fold adds the bias on the fly and reads it back the same way), so like the
        // sigmoid's it runs no command of its own — the routing program's PRE 3 owns its math.
        L::RouteBias { .. } => W::OverOperand(OperandIx(0)),
        L::RouteSoftmax
        | L::RouteScale { .. }
        | L::RouteRenorm
        | L::RouteExpertScale { .. }
        | L::ExpertGatedAct { .. }
        | L::ExpertAct { .. }
        | L::ExpertUnsort => W::OverOperand(OperandIx(0)),
        L::ExpertSort { .. } => W::Region(R::SortedRows),
        L::ExpertMatmul { proj, .. } => W::Region(match proj {
            ExpertProj::Gate => R::ExpertGate,
            ExpertProj::Up => R::ExpertUp,
            ExpertProj::Down => R::ExpertDown,
        }),
        L::ExpertCombine { .. } => W::Arena,
        _ => return None,
    })
}

/// The expert bundles metal has a grouped (sorted-by-expert) GEMM for; the others always gather.
/// GptOss's grouped GEMMs cannot add its per-expert LINEAR biases — the decomposed act and
/// combine commands do, from the GateUpAct/DownCombine decompositions in `tape/lowering.rs`.
pub const METAL_GROUPED_EXPERTS: &[ExpertBundle] = &[
    ExpertBundle::SwitchGlu,
    ExpertBundle::Fused,
    ExpertBundle::SharedFused,
    ExpertBundle::GptOss,
];

/// The (row, pick) pairs from which a bake sorts a grouped bundle's pairs by expert; under them it
/// gathers each row by its picks.
pub const METAL_SORTED_PAIRS: u32 = 64;

/// The pairs per expert from which a bake runs the batched per-expert GEMM (each expert's run
/// padded to its 64-row tile): mlx's own gate for its batched `gather_qmm_rhs`
/// (`rows / experts >= 4`). Under it the padding outweighs the rows.
pub const METAL_GROUPED_PAIRS_PER_EXPERT: f32 = 4.0;

/// The pairs per expert at which a bake sorts the per-pair matvecs' rows by expert, so an expert
/// two rows pick is read once: where it was measured to win (#232: Gemma-4's 128 experts at
/// bucket 8, the full conc-8 decode step). Thicker, the sorted matvecs lose to the gathered ones on
/// a short prompt (Qwen3.6, 2 per expert: 161 ms vs 98 ms).
pub const METAL_SORTED_PAIRS_PER_EXPERT: std::ops::Range<f32> = 0.5..1.0;

/// The bucket at which a bake sorts the per-pair matvecs' rows by expert whatever their density:
/// a lane's verify, Splash's one decode plan (~/git/splash runtime/ops/MoE.cpp:290 `decodePlan`,
/// lanes × SPLASH_TARGET_VERIFY_ROWS rows), which groups every route by expert on 8-row tiles
/// (ops/MoE.hpp `MoeConfig::expertTile`, M8 for every decode plan; `moeGgufPrefillTile` picks only a
/// prefill plan's tile height). Above `METAL_SORTED_PAIRS` pairs alone: below it the sort is mlx's
/// to refuse (gpt-oss's 32 at bucket 8 gather).
pub const METAL_DECODE_GROUPED_ROWS: u32 = METAL_VERIFY_ROWS;

/// The bucket rows from which a gathered bake's gate/up command runs the pairs that share an
/// expert in the expert's first pair (`SharedExperts::InFirstPair`, up to [`METAL_VERIFY_ROWS`]):
/// one row's picks are distinct experts. A bake whose gate/up command routes its tokens
/// (`MetalFusion::MoeRouted`) runs its pairs apart whatever its rows.
pub const METAL_SHARED_EXPERTS_FROM: u32 = 2;

/// The baked (row, pick) pairs at and above which even a verify-sized gathered bake runs its
/// pairs apart: `InFirstPair`'s every threadgroup scans the bucket's every other pair for its
/// expert, so the mode's cost grows with the square of the pair count, while the reads it saves
/// stay the pairs that share an expert — expected `pairs² / (2 · experts)` of them, near none
/// once experts run past pairs. Measured on Nemotron-3 Super 120B (22 of 512 experts per row:
/// 88 baked, 66 live pairs at the 3-row verify bucket) the scan dominates: `MoeGateUpAct`
/// 17.3 ms per step in the first pair vs 10.2 ms apart — apart is the 1-row bucket's cost times
/// the 3× pairs' union, the mode added 7 ms a step. 8-of-256 routing (32 baked pairs at the
/// same bucket) stays under.
pub const METAL_SHARED_EXPERTS_MAX_PAIRS: u32 = 64;

/// The steps whose commands a bake may drop: a gathered block's sort, the unsort (the combine
/// reads through it), and an unsliced bake's sampled rows around its matmul.
pub const METAL_ELIDABLE: &[SubOpKind] = &[
    SubOpKind::ExpertSort,
    SubOpKind::ExpertUnsort,
    SubOpKind::SampleRowsGather,
    SubOpKind::SampleRowsScatter,
    SubOpKind::AllRowsMatmul,
];

/// The router projection's tensor in each router bundle.
pub const fn router_gate(router: RouterBundle) -> WeightTensor {
    match router {
        RouterBundle::Gemma => WeightTensor::GemmaRouterGate,
        RouterBundle::Fused | RouterBundle::SharedFused | RouterBundle::GptOss => {
            WeightTensor::MoeRouterGate
        }
    }
}

/// An expert projection's `[weight, scales, biases]` in every expert bundle.
pub const fn expert_tensors(proj: ExpertProj) -> [WeightTensor; 3] {
    use WeightTensor as T;
    match proj {
        ExpertProj::Gate => [T::MoeExpertGateW, T::MoeExpertGateS, T::MoeExpertGateB],
        ExpertProj::Up => [T::MoeExpertUpW, T::MoeExpertUpS, T::MoeExpertUpB],
        ExpertProj::Down => [T::MoeExpertDownW, T::MoeExpertDownS, T::MoeExpertDownB],
    }
}

/// The fused command each metal fold becomes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MetalFusion {
    RopeAppend,
    MeanSubRmsNorm,
    MeanSubRmsNormBiasAdd,
    FusedAddRmsNorm,
    SiluMul,
    GeluMul,
    FusedGateUpSiluMul,
    FusedGateUpGeluMul,
    RopeAppendNormed,
    NormAddScalarMul,
    MoeGateUpAct,
    /// An ungated expert's one in-projection and its activation.
    MoeUpAct,
    MoeDownCombine,
    MoeRoute,
    /// A KV writer that encodes the rows it writes into the codec's packed store too.
    KvEncoded,
    /// A one-row MLX-affine matvec that normalizes its input as it loads it.
    NormedQmv,
    /// A one-row MLX-affine matvec that adds its bias, scales its rows and adds them into the
    /// residual stream as it stores them.
    QmvEpilogue,
    /// A one-row router's logits that normalize its input as they load it.
    NormedRouter,
    /// A gathered MoE block's gated expert command, up to verify-sized rows, that routes each of
    /// its tokens itself, from the logits, and stores the picks and scores the later commands read.
    MoeRouted,
    /// `residual + rmsnorm(delta)`.
    NormAdd,
    /// A group of row-wise steps over one width.
    RowProgram,
    /// A one-row step's decode attention that runs its KV writer: it ropes its query, writes the
    /// new row (and encodes it) and reads the step's own key itself.
    RopedAttention,
    /// A one-row step's gathered expert combine that adds a shared expert's gated rows and the
    /// residual as it stores each row.
    CombineEpilogue,
}

/// The KV writers a decode attention runs itself (`MetalFusion::RopedAttention`): the plain rope
/// writer and its folded-in encode, not the normed one.
const ROPED_ATTENTION_WRITERS: &[MetalFusion] = &[MetalFusion::RopeAppend, MetalFusion::KvEncoded];

/// Metal's fusions, in the order the shared fold pass applies them. An attention reads its new K/V
/// (operands 1..) out of the cache its rope wrote, so only its query is a consumer edge.
///
/// Two tables share every pattern but one: the routed-experts fold's band. The wide band
/// ([`METAL_FUSIONS`]) folds through the verify-sized rows on Max-scale M5 silicon, where the
/// fold's per-threadgroup routing has the cores to hide behind (measured, M5 Max 40-core:
/// −4.1% TPOT at conc-1 MTP-on). The narrow band ([`METAL_FUSIONS_NARROW`]) keeps the fold's
/// one-row origin — main's decision at every bucket — because on 10-core silicon the
/// per-threadgroup routing costs more than the dispatch it saves (measured, base M5:
/// +4.7% verify ITL, PR #847's review); every chip but the wide class lowers under it, and
/// an unmeasured chip joins the wide set only when benched.
const SWEEP_ELEMENTWISE: &[FoldPattern<MetalFusion>] = &[
    FoldPattern::Sibling {
        driver: SubOpKind::RopeAppend,
        sibling: SubOpKind::RopeRotate,
        kernel: MetalFusion::RopeAppend,
    },
    FoldPattern::CentredNorm {
        norm: SubOpKind::RmsNorm,
        sub: SubOpKind::Sub,
        mean: SubOpKind::Mean,
        bias: SubOpKind::BiasAdd,
        kernel: MetalFusion::MeanSubRmsNorm,
        biased: MetalFusion::MeanSubRmsNormBiasAdd,
    },
    FoldPattern::MatvecEpilogue {
        anchor: SubOpKind::MatmulTile,
        weights: Some(GemmWeightKind::Affine),
        bias: Some(SubOpKind::BiasAdd),
        scale: Some(SubOpKind::ScalarMul),
        gate_scale: None,
        add: SubOpKind::Add,
        gated: &[SubOpKind::Silu, SubOpKind::Gelu, SubOpKind::Mul],
        kernel: MetalFusion::QmvEpilogue,
    },
    // A MoE block's input norm too: its router's logits, its shared expert's matvecs
    // and its gathered experts each normalize the rows as they load them.
    FoldPattern::NormedMatvecs {
        norm: SubOpKind::RmsNorm,
        readers: &[
            NormReader::Matvec {
                kind: SubOpKind::MatmulTile,
                weights: Some(GemmWeightKind::Affine),
            },
            NormReader::Matvec {
                kind: SubOpKind::RouterLogits,
                weights: None,
            },
            NormReader::Gathered {
                sort: SubOpKind::ExpertSort,
                gathered_below: METAL_SORTED_PAIRS,
            },
        ],
        kernel: MetalFusion::NormedQmv,
    },
    FoldPattern::NormedMatvecs {
        norm: SubOpKind::RouterNorm,
        readers: &[NormReader::Matvec {
            kind: SubOpKind::RouterLogits,
            weights: None,
        }],
        kernel: MetalFusion::NormedRouter,
    },
    FoldPattern::Gated {
        mul: SubOpKind::Mul,
        activations: &[
            GatedKernel {
                activation: SubOpKind::Silu,
                split: MetalFusion::SiluMul,
                fused: MetalFusion::FusedGateUpSiluMul,
            },
            GatedKernel {
                activation: SubOpKind::Gelu,
                split: MetalFusion::GeluMul,
                fused: MetalFusion::FusedGateUpGeluMul,
            },
        ],
    },
];

const SWEEP_EXPERTS: &[FoldPattern<MetalFusion>] = &[
    // After the row programs: what they left of the residual adds and their norms.
    FoldPattern::ResidualNorm {
        norm: SubOpKind::RmsNorm,
        add: SubOpKind::Add,
        kernel: MetalFusion::FusedAddRmsNorm,
    },
    FoldPattern::NormedRope {
        rope: SubOpKind::RopeAppend,
        gain_norm: SubOpKind::RmsNorm,
        unit_norm: SubOpKind::RmsNormUnit,
        kernel: MetalFusion::RopeAppendNormed,
    },
    FoldPattern::NormAddScale {
        scale: SubOpKind::ScalarWeightMul,
        add: SubOpKind::Add,
        norm: SubOpKind::RmsNorm,
        kernel: MetalFusion::NormAddScalarMul,
    },
    FoldPattern::ExpertActivation {
        act: SubOpKind::ExpertGatedAct,
        matmul: SubOpKind::ExpertMatmul,
        kernel: MetalFusion::MoeGateUpAct,
    },
    FoldPattern::ExpertActivation {
        act: SubOpKind::ExpertAct,
        matmul: SubOpKind::ExpertMatmul,
        kernel: MetalFusion::MoeUpAct,
    },
    FoldPattern::ExpertCombined {
        combine: SubOpKind::ExpertCombine,
        unsort: SubOpKind::ExpertUnsort,
        matmul: SubOpKind::ExpertMatmul,
        kernel: MetalFusion::MoeDownCombine,
    },
    // `moe_route.metal`'s program: a softmax over every expert first (Qwen) or the
    // F32 sigmoid + correction bias GLM-4's noaux_tc routes by, then the scores
    // scaled, softmaxed or renormalized, scaled per expert, and scaled last
    // (Nemotron-H's routed scaling factor, after the renorm).
    FoldPattern::Route {
        top_k: SubOpKind::RouteTopK,
        sort: SubOpKind::RouteArgsort,
        pre: &[
            SubOpKind::RouteSoftmax,
            SubOpKind::RouteSigmoidBias,
            SubOpKind::RouteBias,
        ],
        gather: SubOpKind::RouteGatherScores,
        tail: &[
            &[SubOpKind::RouteScale],
            &[SubOpKind::RouteSoftmax, SubOpKind::RouteRenorm],
            &[SubOpKind::RouteExpertScale],
            &[SubOpKind::RouteScale],
        ],
        kernel: MetalFusion::MoeRoute,
    },
];

// The sweep after the rope folds: it extends the writer command they made. A norm folds into the
// add it feeds only when no fold took either with more. After the expert and routing
// folds, the routing into the expert commands: under the grouped bake's pairs, they
// read each token's row by its picks.
const F_KV_ENCODED_ROPE: FoldPattern<MetalFusion> = FoldPattern::Encoded {
    writer: SubOpKind::RopeAppend,
    encode: SubOpKind::KvEncode,
    kernel: MetalFusion::KvEncoded,
};
const F_KV_ENCODED_APPEND: FoldPattern<MetalFusion> = FoldPattern::Encoded {
    writer: SubOpKind::KvAppend,
    encode: SubOpKind::KvEncode,
    kernel: MetalFusion::KvEncoded,
};
const F_NORM_ADD: FoldPattern<MetalFusion> = FoldPattern::NormAdd {
    add: SubOpKind::Add,
    norm: SubOpKind::RmsNorm,
    kernel: MetalFusion::NormAdd,
};
const F_ROPED_DECODE: FoldPattern<MetalFusion> = FoldPattern::RopedAttention {
    attention: SubOpKind::AttnDecode,
    rope: SubOpKind::RopeAppend,
    writers: ROPED_ATTENTION_WRITERS,
    kernel: MetalFusion::RopedAttention,
};
const F_ROPED_PACKED: FoldPattern<MetalFusion> = FoldPattern::RopedAttention {
    attention: SubOpKind::AttnPackedKv,
    rope: SubOpKind::RopeAppend,
    writers: ROPED_ATTENTION_WRITERS,
    kernel: MetalFusion::RopedAttention,
};
// After the expert folds: the combine's command is the down's and the unsort's.
const F_COMBINE_EPILOGUE: FoldPattern<MetalFusion> = FoldPattern::MatvecEpilogue {
    anchor: SubOpKind::ExpertCombine,
    weights: None,
    bias: None,
    scale: None,
    gate_scale: Some(SubOpKind::GateScale),
    add: SubOpKind::Add,
    gated: &[],
    kernel: MetalFusion::CombineEpilogue,
};

/// The wide band's routed-experts fold: through the verify-sized rows.
const F_ROUTED_WIDE: FoldPattern<MetalFusion> = FoldPattern::RoutedExperts {
    top_k: SubOpKind::RouteTopK,
    sort: SubOpKind::ExpertSort,
    bands: &RoutedBands {
        rows_through: METAL_VERIFY_ROWS,
        grouped_pairs: METAL_SORTED_PAIRS,
        grouped_bundles: METAL_GROUPED_EXPERTS,
        sorted_per_expert: METAL_SORTED_PAIRS_PER_EXPERT,
        grouped_per_expert: METAL_GROUPED_PAIRS_PER_EXPERT,
        decode_rows: METAL_DECODE_GROUPED_ROWS,
    },
    kernel: MetalFusion::MoeRouted,
};

/// The narrow band's: the fold's one-row origin, main's decision at every bucket.
const F_ROUTED_NARROW: FoldPattern<MetalFusion> = FoldPattern::RoutedExperts {
    top_k: SubOpKind::RouteTopK,
    sort: SubOpKind::ExpertSort,
    bands: &RoutedBands {
        rows_through: 1,
        grouped_pairs: METAL_SORTED_PAIRS,
        grouped_bundles: METAL_GROUPED_EXPERTS,
        sorted_per_expert: METAL_SORTED_PAIRS_PER_EXPERT,
        grouped_per_expert: METAL_GROUPED_PAIRS_PER_EXPERT,
        decode_rows: METAL_DECODE_GROUPED_ROWS,
    },
    kernel: MetalFusion::MoeRouted,
};

const SWEEP_AFTER_ROPE_WIDE: &[FoldPattern<MetalFusion>] = &[
    F_KV_ENCODED_ROPE,
    F_KV_ENCODED_APPEND,
    F_NORM_ADD,
    F_ROUTED_WIDE,
    // `attention_via_cache_v2`'s `ATTN_FOLD`: the plain writer, encoding or not.
    F_ROPED_DECODE,
    F_ROPED_PACKED,
    F_COMBINE_EPILOGUE,
];

const SWEEP_AFTER_ROPE_NARROW: &[FoldPattern<MetalFusion>] = &[
    F_KV_ENCODED_ROPE,
    F_KV_ENCODED_APPEND,
    F_NORM_ADD,
    F_ROUTED_NARROW,
    // `attention_via_cache_v2`'s `ATTN_FOLD`: the plain writer, encoding or not.
    F_ROPED_DECODE,
    F_ROPED_PACKED,
    F_COMBINE_EPILOGUE,
];

pub const METAL_FUSIONS: FusionTable<MetalFusion> = FusionTable {
    counted: &[CountedOperands {
        kind: SubOpKind::AttnDecode,
        before: OperandIx(1),
    }],
    sweeps: &[SWEEP_ELEMENTWISE, SWEEP_EXPERTS, SWEEP_AFTER_ROPE_WIDE],
    // `row_program.metal`'s limits: 16 instructions hold 8 steps with their loads and stores.
    // After the matvecs took their ends.
    rows: Some(RowFold {
        after_sweep: 1,
        kinds: &[
            SubOpKind::RmsNorm,
            SubOpKind::Add,
            SubOpKind::ScalarWeightMul,
            SubOpKind::ScalarMul,
        ],
        steps: 8,
        inputs: 4,
        outputs: 3,
        kernel: MetalFusion::RowProgram,
    }),
};

/// Metal's fusions under the narrow routed band — every pattern but the routed-experts fold
/// shared with [`METAL_FUSIONS`] by name. The band every chip but the wide-M5 class lowers
/// under; see the table pair's doc above.
pub const METAL_FUSIONS_NARROW: FusionTable<MetalFusion> = FusionTable {
    counted: METAL_FUSIONS.counted,
    sweeps: &[SWEEP_ELEMENTWISE, SWEEP_EXPERTS, SWEEP_AFTER_ROPE_NARROW],
    rows: METAL_FUSIONS.rows,
};
