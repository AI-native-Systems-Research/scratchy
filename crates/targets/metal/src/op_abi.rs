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
    AttnMask, EwKind, ExpertBundle, ExpertProj, KvOperand, RouterBundle, SubOp,
};

use crate::tape::lowered::{RuntimeGate, WeightTensor};
use crate::tape::step::MoeRegion;
use scratchy_subtile::tape_colouring::{ColourFacts, ColourRule, OutputAlias};
use scratchy_subtile::tape_folding::{CountedOperands, FoldPattern, FusionTable, GatedKernel};
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
pub const METAL_COLOUR_FACTS: ColourFacts = ColourFacts {
    rule: metal_colour_rule,
    colour_zero: SourceBinding::EmbeddedHidden,
};

/// How each op's metal kernel treats its buffers.
///
/// The match is exhaustive: a new `SubOp` fails to compile here
/// (E0004) until its author states which case it is, rather than
/// silently defaulting to "fresh buffer" and losing a color.
pub fn metal_colour_rule(op: &SubOp) -> ColourRule {
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
        | L::GateScale
        | L::GatedDeltaNet
        | L::AttnDecode { .. }
        | L::LoadPixels { .. }
        | L::LoadPosEmbeds { .. }
        | L::EmbeddingGather { .. }
        | L::ExpertCombine { .. } => FRESH,
        // The rest of a MoE block lives in the op scratch (`moe_write`), not the arena.
        L::RouterNorm { .. }
        | L::RouterLogits { .. }
        | L::RouteSoftmax
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

/// The runtime gate each codec guard runs under on metal. The codec is the model's, fixed at
/// build (`MetalModelConsts::kv_codec`): a dense model inserts no codec step, so a coded one's
/// guards only ask whether the step is a decode step.
pub const METAL_GUARD_GATES: GuardGates<Option<RuntimeGate>> = GuardGates {
    codec: None,
    codec_decode: Some(RuntimeGate::OnlyIfDecodeStep),
    codec_not_decode: Some(RuntimeGate::UnlessDecodeStep),
    unless_codec_decode: Some(RuntimeGate::UnlessDecodeStep),
};

// `MetalFusion::KvEncoded` folds the codec's encodes into their KV writer, which runs ungated.
const _: () = assert!(
    METAL_GUARD_GATES.codec.is_none(),
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
        L::RouteSoftmax
        | L::RouteScale { .. }
        | L::RouteRenorm
        | L::RouteExpertScale { .. }
        | L::ExpertGatedAct { .. } => W::OverOperand(OperandIx(0)),
        L::ExpertSort { .. } => W::Region(R::SortedRows),
        L::ExpertMatmul { proj, .. } => W::Region(match proj {
            ExpertProj::Gate => R::ExpertGate,
            ExpertProj::Up => R::ExpertUp,
            ExpertProj::Down => R::ExpertDown,
        }),
        L::ExpertUnsort => W::Region(R::TokenRows),
        L::ExpertCombine { .. } => W::Arena,
        _ => return None,
    })
}

/// The expert bundles metal has a grouped (sorted-by-expert) GEMM for; the others always gather.
pub const METAL_GROUPED_EXPERTS: &[ExpertBundle] = &[ExpertBundle::SwitchGlu];

/// The steps whose commands a bake may drop: a gathered block's sort and unsort, and an unsliced
/// bake's sampled rows around its matmul.
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
        RouterBundle::Fused | RouterBundle::SharedFused => WeightTensor::MoeRouterGate,
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
    /// `residual + rmsnorm(delta)`.
    NormAdd,
}

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
                    matmul: K::MatmulTile,
                    weights: GemmWeightKind::Affine,
                    bias: K::BiasAdd,
                    scale: K::ScalarMul,
                    add: K::Add,
                    gated: &[K::Silu, K::Gelu, K::Mul],
                    kernel: F::QmvEpilogue,
                },
                FoldPattern::NormedMatvecs {
                    norm: K::RmsNorm,
                    matmul: K::MatmulTile,
                    weights: GemmWeightKind::Affine,
                    kernel: F::NormedQmv,
                },
                FoldPattern::ResidualNorm {
                    norm: K::RmsNorm,
                    add: K::Add,
                    kernel: F::FusedAddRmsNorm,
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
            &[
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
                // `moe_route.metal`'s program: a softmax over every expert first, then the scores
                // scaled, softmaxed or renormalized, and scaled per expert.
                FoldPattern::Route {
                    top_k: K::RouteTopK,
                    sort: K::RouteArgsort,
                    pre: K::RouteSoftmax,
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
            // add it feeds only when no fold took either with more.
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
            ],
        ],
    }
};
