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
    CountedOperands, FoldPattern, FusionTable, GatedKernel, NormReader, RowFold,
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
/// The widest bucket metal lays out in wave order (`wave_schedule::wave_order`): a one-row tape,
/// whose steps are latency-bound — a wider bucket would hold more rows' buffers live at once.
pub const METAL_WAVE_ORDER_ROWS: u64 = 1;

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
        L::TanhSoftCap | L::Elementwise(E::Gelu | E::QuickGelu | E::GeluErf) => {
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
        L::MatmulTile { .. } | L::VarlenAttention { .. } | L::EncoderAttn { .. } => {
            FRESH.reads_across()
        }
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
        | L::AttnDecode { .. }
        | L::LoadRows { .. }
        | L::EmbeddingGather { .. }
        | L::ExpertCombine { .. } => FRESH,
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

/// Metal's sampled rows: the MLX-affine result matmul, whose sampled rows the lm_head slice
/// computes with qmv.
pub const METAL_SAMPLE_ROWS: SampleRowsFacts = SampleRowsFacts {
    weights: &[GemmWeightKind::Affine],
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
/// token and its drafts (an MTP head's: `spec_max_seqs * (spec_drafts + 1)`, 3). Such a bucket's
/// steps are decode-shaped — a few query rows over a whole context — and run decode-shaped
/// kernels: its paged attention is the decode kernel's, one query row per token; a gathered MoE
/// block runs the pairs that share an expert once ([`METAL_SHARED_EXPERTS_FROM`]). Larger buckets
/// keep prefill-shaped kernels: a 31-token prompt took 20% longer to its first token at a 32-row
/// verify-sized bucket than at the 64-row prefill one (Qwen3.6-35B-A3B, base M5: 176 vs 147 ms).
///
/// TODO: derive it, and the decoder ladder's verify-sized rungs, from the compiled heads'
/// `spec_max_seqs * (spec_drafts + 1)`: a gate above 2 sequences puts verify steps past 8 rows,
/// onto the 64-row prefill-shaped tape.
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
pub const METAL_GROUPED_EXPERTS: &[ExpertBundle] = &[
    ExpertBundle::SwitchGlu,
    ExpertBundle::Fused,
    ExpertBundle::SharedFused,
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
/// bucket 8, the full conc-8 decode step). The sorted matvecs run the bucket's every pair, so
/// thicker, they lose to the gathered ones on a short prompt (Qwen3.6, 2 per expert: 161 ms vs
/// 98 ms).
pub const METAL_SORTED_PAIRS_PER_EXPERT: std::ops::Range<f32> = 0.5..1.0;

/// The bucket rows from which a gathered bake's gate/up command runs the pairs that share an
/// expert in the expert's first pair (`SharedExperts::InFirstPair`, up to [`METAL_VERIFY_ROWS`]):
/// one row's picks are distinct experts.
pub const METAL_SHARED_EXPERTS_FROM: u32 = 2;

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
    /// A one-row MoE block's gated expert command that routes its token itself, from the logits,
    /// and stores the picks and scores the later ones read.
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
pub const METAL_FUSIONS: FusionTable<MetalFusion> = {
    use MetalFusion as F;
    use SubOpKind as K;
    FusionTable {
        counted: &[CountedOperands {
            kind: K::AttnDecode,
            before: OperandIx(1),
        }],
        sweeps: &[
            &[
                FoldPattern::Sibling {
                    driver: K::RopeAppend,
                    sibling: K::RopeRotate,
                    kernel: F::RopeAppend,
                },
                FoldPattern::CentredNorm {
                    norm: K::RmsNorm,
                    sub: K::Sub,
                    mean: K::Mean,
                    bias: K::BiasAdd,
                    kernel: F::MeanSubRmsNorm,
                    biased: F::MeanSubRmsNormBiasAdd,
                },
                FoldPattern::MatvecEpilogue {
                    anchor: K::MatmulTile,
                    weights: Some(GemmWeightKind::Affine),
                    bias: Some(K::BiasAdd),
                    scale: Some(K::ScalarMul),
                    gate_scale: None,
                    add: K::Add,
                    gated: &[K::Silu, K::Gelu, K::Mul],
                    kernel: F::QmvEpilogue,
                },
                // A MoE block's input norm too: its router's logits, its shared expert's matvecs
                // and its gathered experts each normalize the rows as they load them.
                FoldPattern::NormedMatvecs {
                    norm: K::RmsNorm,
                    readers: &[
                        NormReader::Matvec {
                            kind: K::MatmulTile,
                            weights: Some(GemmWeightKind::Affine),
                        },
                        NormReader::Matvec {
                            kind: K::RouterLogits,
                            weights: None,
                        },
                        NormReader::Gathered {
                            sort: K::ExpertSort,
                            gathered_below: METAL_SORTED_PAIRS,
                        },
                    ],
                    kernel: F::NormedQmv,
                },
                FoldPattern::NormedMatvecs {
                    norm: K::RouterNorm,
                    readers: &[NormReader::Matvec {
                        kind: K::RouterLogits,
                        weights: None,
                    }],
                    kernel: F::NormedRouter,
                },
                FoldPattern::Gated {
                    mul: K::Mul,
                    activations: &[
                        GatedKernel {
                            activation: K::Silu,
                            split: F::SiluMul,
                            fused: F::FusedGateUpSiluMul,
                        },
                        GatedKernel {
                            activation: K::Gelu,
                            split: F::GeluMul,
                            fused: F::FusedGateUpGeluMul,
                        },
                    ],
                },
            ],
            // After the row programs: what they left of the residual adds and their norms.
            &[
                FoldPattern::ResidualNorm {
                    norm: K::RmsNorm,
                    add: K::Add,
                    kernel: F::FusedAddRmsNorm,
                },
                FoldPattern::NormedRope {
                    rope: K::RopeAppend,
                    gain_norm: K::RmsNorm,
                    unit_norm: K::RmsNormUnit,
                    kernel: F::RopeAppendNormed,
                },
                FoldPattern::NormAddScale {
                    scale: K::ScalarWeightMul,
                    add: K::Add,
                    norm: K::RmsNorm,
                    kernel: F::NormAddScalarMul,
                },
                FoldPattern::ExpertGated {
                    act: K::ExpertGatedAct,
                    matmul: K::ExpertMatmul,
                    kernel: F::MoeGateUpAct,
                },
                FoldPattern::ExpertCombined {
                    combine: K::ExpertCombine,
                    unsort: K::ExpertUnsort,
                    matmul: K::ExpertMatmul,
                    kernel: F::MoeDownCombine,
                },
                // `moe_route.metal`'s program: a softmax over every expert first (Qwen) or the
                // F32 sigmoid + correction bias GLM-4's noaux_tc routes by, then the scores
                // scaled, softmaxed or renormalized, and scaled per expert.
                FoldPattern::Route {
                    top_k: K::RouteTopK,
                    sort: K::RouteArgsort,
                    pre: &[K::RouteSoftmax, K::RouteSigmoidBias, K::RouteBias],
                    gather: K::RouteGatherScores,
                    tail: &[
                        &[K::RouteScale],
                        &[K::RouteSoftmax, K::RouteRenorm],
                        &[K::RouteExpertScale],
                    ],
                    kernel: F::MoeRoute,
                },
            ],
            // After the rope folds: it extends the writer command they made. A norm folds into the
            // add it feeds only when no fold took either with more. After the expert and routing
            // folds, the routing into the expert commands: under the grouped bake's pairs, they
            // read each token's row by its picks.
            &[
                FoldPattern::Encoded {
                    writer: K::RopeAppend,
                    encode: K::KvEncode,
                    kernel: F::KvEncoded,
                },
                FoldPattern::NormAdd {
                    add: K::Add,
                    norm: K::RmsNorm,
                    kernel: F::NormAdd,
                },
                FoldPattern::RoutedExperts {
                    top_k: K::RouteTopK,
                    sort: K::ExpertSort,
                    gathered_below: METAL_SORTED_PAIRS,
                    shared_from: METAL_SHARED_EXPERTS_FROM,
                    kernel: F::MoeRouted,
                },
                // `attention_via_cache_v2`'s `ATTN_FOLD`: the plain writer, encoding or not.
                FoldPattern::RopedAttention {
                    attention: K::AttnDecode,
                    rope: K::RopeAppend,
                    writers: ROPED_ATTENTION_WRITERS,
                    kernel: F::RopedAttention,
                },
                FoldPattern::RopedAttention {
                    attention: K::AttnPackedKv,
                    rope: K::RopeAppend,
                    writers: ROPED_ATTENTION_WRITERS,
                    kernel: F::RopedAttention,
                },
                // After the expert folds: the combine's command is the down's and the unsort's.
                FoldPattern::MatvecEpilogue {
                    anchor: K::ExpertCombine,
                    weights: None,
                    bias: None,
                    scale: None,
                    gate_scale: Some(K::GateScale),
                    add: K::Add,
                    gated: &[],
                    kernel: F::CombineEpilogue,
                },
            ],
        ],
        // `row_program.metal`'s limits: 16 instructions hold 8 steps with their loads and stores.
        // After the matvecs took their ends.
        rows: Some(RowFold {
            after_sweep: 1,
            kinds: &[K::RmsNorm, K::Add, K::ScalarWeightMul, K::ScalarMul],
            steps: 8,
            inputs: 4,
            outputs: 3,
            kernel: F::RowProgram,
        }),
    }
};
