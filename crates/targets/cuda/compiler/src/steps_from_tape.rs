// SPDX-License-Identifier: Apache-2.0
//! CUDA'S OPCODE LOWERING: one tape step to one [`CudaStep`], its operands resolved to arena
//! offsets and weight indices.
//!
//! A kept step lowers through [`lower_step`], a step driving a fold through [`lower_fused`]; an
//! absorbed step emits nothing (`canonical::lower_canonical` asks the fold role first).
//!
//! The match lists every `SubOp` variant, with no wildcard, so a new `SubOp` fails to compile here
//! (E0004) as it does in the colour rule. Each opcode issue moves exactly one variant out of the
//! refusing arm into an arm of its own.

use std::num::NonZeroU32;

use scratchy_subtile::handoff::{SourceBinding, UnrollIndex};
use scratchy_subtile::lower::{ExpertWeight, GemmWeight};
use scratchy_subtile::model_geometry::ModelAttnGeometry;
use scratchy_subtile::subtile_ir::{
    AttnMask, EwKind, ExpertBundle, ExpertProj, GainConvention, GatedAct, RopeFormTag,
    RouterBundle, SubOp, SubtileIR, SubtileNode,
};
use scratchy_subtile::subtile_ir::{TensorRegion, TensorShape};
use scratchy_subtile::subtile_tape::{ComputeInput, ComputeInputs, SlotId};
use scratchy_subtile::tape_colouring::{Colour, TapeColouring};
use scratchy_subtile::tape_folding::{FusedShape, Fusion};
use scratchy_target_cuda::op_abi::{CUDA_GROUPED_EXPERTS, CudaFusion, MOE_EXPERT_COUNTS};
use scratchy_target_cuda::step::{
    ADD_BF16_COLS, ATTN_BF16_HEAD_DIMS, ATTN_DECODE_BF16_MAX_GROUP, ATTN_PREFILL_BF16_ROWS,
    ATTN_WINDOW_MULTIPLE, AddBf16, AttnPrefillBf16, CudaStep, EMBED_BF16_COLS, EmbedBf16,
    ExpertCombineBf16, ExpertGemmBf16, ExpertGemvBf16, ExpertProjection, ExpertSortBf16,
    GATE_APPLY_BF16_COLS, GATE_SCALE_BF16_COLS, GATE_SPLIT_BF16_HEAD_DIM,
    GATE_SPLIT_BF16_HEAD_MULTIPLE, GDN_CONV_KERNEL_RANGE, GDN_HEAD_K, GDN_HEAD_V_MAX,
    GDN_HEAD_V_MULTIPLE, GEMM_BF16_TILE, GEMM_COL64_TILE, GEMM_NARROW_BF16_MAX_N,
    GEMM_NARROW_BF16_MULTIPLE, GEMM_NARROW_BF16_ONE, GEMV_BF16_MULTIPLE, GateApplyBf16,
    GateScaleBf16, GateSplitBf16, GatedDeltaNetBf16, GemmBf16, LayerScaleBf16, MOE_ROUTE_MAX_TOP_K,
    MatmulKernel, MoeRouteBf16, MoeWeight, PairRows, RMS_NORM_BF16_COLS, ROPE_BF16_HEAD_MULTIPLE,
    ROPE_BF16_ROT_MULTIPLE, RmsNormBf16, RmsNormUnitBf16, RopeAppendBf16, RopeBf16, RouteProgram,
    RouterLogitsBf16, SCALE_BF16_COLS, SILU_MUL_BF16_COLS, SOFT_CAP_BF16_COLS, ScaleBf16,
    SiluMulBf16, SoftCapBf16,
};
use scratchy_target_cuda::tape::{ArenaOffset, MoeTape, WeightIx};

use crate::canonical::{
    ArenaRefusal, AttnShapeRule, CanonicalRefusal, CudaAttnClass, CudaModelFacts, EmbedShapeRule,
    GateShapeRule, GdnRegions, GdnShapeRule, GdnWidths, GemmColTile, GemmShapeRule, KvCacheRefusal,
    KvPool, KvRegions, MoeShapeRule, OperandRefusal, ProducerRefusal, RopeShapeRule, RowShapeRule,
    SharedExpertOwner, ViewRefusal,
};

/// What a step's lowering reads besides the step itself: the planned canonical, the model facts,
/// and where each colour and the rotary table live in the arena.
pub struct StepCx<'a> {
    pub graph: &'a SubtileIR,
    /// Parallel to the graph's sources.
    pub bindings: &'a [SourceBinding],
    pub colouring: &'a TapeColouring,
    pub facts: CudaModelFacts,
    /// The bucket's token count (the embedded hidden's rows), when one source is bound to it.
    pub tokens: Option<u32>,
    /// Each colour's byte offset, or why the arena has no layout.
    pub offsets: Result<&'a [u64], ArenaRefusal>,
    /// The rotary table's offset, when the arena has one.
    pub rotary: Option<ArenaOffset>,
    /// The local rotary table's offset (a two-class model's base class), when the arena has one.
    pub rotary_local: Option<ArenaOffset>,
    /// The KV pool the `PrefixK` / `PrefixV` sources describe (`None` when there are none), or
    /// why they cannot be one.
    pub kv: Result<Option<&'a KvPool>, KvCacheRefusal>,
    /// The KV regions' spans, when the arena has them.
    pub kv_regions: Option<KvRegions>,
    /// The model layer ids of the canonical's GDN bundles, ascending: a GDN step's `layer` is its
    /// index here.
    pub gdn_layers: &'a [u32],
    /// The GDN regions' spans, when the arena has them.
    pub gdn_regions: Option<GdnRegions>,
    /// The MoE scratch, when the arena has one.
    pub moe: Option<MoeTape>,
}

/// One `Compute` of the tape: its position among the `Compute`s, its node, the slot it writes and
/// its positional inputs.
pub struct TapeStep<'a> {
    pub step: usize,
    pub node: &'a SubtileNode,
    pub writes: SlotId,
    pub inputs: &'a ComputeInputs,
}

/// Lower the op of one kept `Compute` on the tape to its CUDA step, or to none for a view.
/// `computes` is every `Compute` of the tape, in which a `GateApply` finds its gate's `GateSplit`
/// and an attention its layer's `RopeAppend`.
pub fn lower_step(
    cx: &StepCx<'_>,
    s: &TapeStep<'_>,
    computes: &[TapeStep<'_>],
) -> Result<Option<CudaStep>, CanonicalRefusal> {
    use EwKind as E;
    use SubOp as L;
    let op = &s.node.op;
    let step = match op {
        L::MatmulTile { n, weight } => gemm(cx, s, *n, *weight),
        L::RmsNorm { eps, gain } => rms_norm(cx, s, *eps, *gain).map(CudaStep::RmsNorm),
        L::Elementwise(E::Add) => add(cx, s).map(CudaStep::Add),
        L::Reshape { .. } => return view(cx, s).map(|()| None),
        L::RopeRotate { head_dim, .. } => rope(cx, s, head_dim.get()).map(CudaStep::Rope),
        L::GateSplit { half_cols } => gate_split(cx, s, *half_cols).map(CudaStep::GateSplit),
        L::GateApply => gate_apply(cx, s, computes).map(CudaStep::GateApply),
        L::RopeAppend {
            head_dim,
            layer,
            attn,
            pairing,
            ..
        } => rope_append(cx, s, head_dim.get(), *layer, *attn, *pairing).map(CudaStep::RopeAppend),
        L::AttnDecode {
            geom, scale, mask, ..
        } => attention(cx, s, *geom, *scale, *mask, computes),
        L::GatedDeltaNet => gated_delta_net(cx, s).map(CudaStep::GatedDeltaNet),
        L::RouterLogits { experts, router } => {
            router_logits(cx, s, experts.get(), *router, computes).map(CudaStep::RouterLogits)
        }
        L::RouterNorm { eps, router } => {
            router_norm(cx, s, *eps, *router).map(CudaStep::RouterNorm)
        }
        L::ExpertSort { experts, k, bundle } => {
            return expert_sort(cx, s, (experts.get(), k.get()), *bundle, computes)
                .map(|e| e.map(CudaStep::ExpertSort));
        }
        L::ExpertMatmul {
            proj,
            n,
            k,
            weight,
            bundle,
        } => expert_matmul(cx, s, (*proj, *n, k.get()), (*weight, *bundle), computes),
        L::ExpertGatedAct { act } => expert_gated_act(cx, s, *act, computes),
        L::ExpertUnsort => return expert_unsort(cx, s, computes).map(|()| None),
        L::ExpertCombine { hidden, shared } => {
            expert_combine(cx, s, *hidden, shared.0.is_some(), computes)
                .map(CudaStep::ExpertCombine)
        }
        L::GateScale => gate_scale(cx, s).map(CudaStep::GateScale),
        L::ScalarMul { scale } => scalar_mul(cx, s, *scale).map(CudaStep::Scale),
        L::RmsNormUnit { eps } => rms_norm_unit(cx, s, *eps).map(CudaStep::RmsNormUnit),
        L::ScalarWeightMul => layer_scale(cx, s).map(CudaStep::LayerScale),
        L::TanhSoftCap { cap } => soft_cap(cx, s, *cap).map(CudaStep::SoftCap),
        // A routing runs only as the `MoeRoute` fold its top-k drives: one the fold left (a
        // softmax or argsort it did not absorb, a top-k driving none, a gather, scale, renorm or
        // per-expert scale no top-k took as its epilogue) has no kernel.
        L::RouteSoftmax
        | L::RouteArgsort
        | L::RouteTopK { .. }
        | L::RouteGatherScores
        | L::RouteScale { .. }
        | L::RouteExpertScale { .. }
        | L::RouteRenorm => Err(CanonicalRefusal::MoeShape {
            step: s.step,
            op: op.name(),
            rule: MoeShapeRule::Routing,
        }),
        // SwiGLU runs only as the `SiluMul` fold: a `Silu` the fold left, a `Mul` driving none,
        // and the pre-fused op (which only spyre's `fuse_silu_mul` makes) have no kernel. GeGLU
        // runs only as the `GeluMul` fold: a `Gelu` it left is refused below, as not lowered.
        L::Elementwise(E::Silu | E::Mul) | L::SiluMul => Err(CanonicalRefusal::Unfused {
            step: s.step,
            op: op.name(),
        }),
        L::SumReduce { .. }
        | L::Elementwise(E::Gelu | E::QuickGelu | E::GeluErf | E::Relu2 | E::Sub | E::BiasAdd)
        | L::RmsNormReduce { .. }
        | L::RmsNormApply { .. }
        | L::KvAppend { .. }
        | L::Concat { .. }
        | L::LoadRows { .. }
        | L::EmbeddingGather { .. }
        | L::VisionRope
        | L::VarlenAttention { .. }
        | L::EncoderAttn { .. }
        | L::Mamba2 { .. }
        | L::Mean
        | L::RouteSigmoidBias { .. }
        | L::RouteBias { .. }
        | L::ExpertAct { .. }
        | L::KvEncode { .. }
        | L::KvStage { .. }
        | L::RotateRows { .. }
        | L::AttnPackedKv
        | L::SampleRowsGather
        | L::SampleRowsScatter
        | L::AllRowsMatmul => Err(CanonicalRefusal::NotLowered {
            step: s.step,
            op: op.name(),
        }),
    };
    step.map(Some)
}

/// The head's embed: `out [rows, hidden]` (colour 0, the embedded hidden) gathered from the
/// `[vocab, hidden]` bf16 `table` by the `rows` u32 token ids at `tokens`. Not lowered from a tape
/// op: the shared tape's first ops read the embedded hidden as a source, which this step fills.
/// The shape rule is the launcher's ([`EMBED_BF16_COLS`]), stated at expansion.
pub fn embed(
    tokens: ArenaOffset,
    table: WeightIx,
    out: ArenaOffset,
    (rows, hidden, vocab): (u32, u32, u32),
) -> Result<CudaStep, CanonicalRefusal> {
    let refuse = |rule| CanonicalRefusal::EmbedShape {
        rows,
        hidden,
        vocab,
        rule,
    };
    for (dim, value) in [("rows", rows), ("hidden", hidden), ("vocab", vocab)] {
        if value == 0 {
            return Err(refuse(EmbedShapeRule::Empty { dim }));
        }
        if value > i32::MAX as u32 {
            return Err(refuse(EmbedShapeRule::AboveIntMax { dim, value }));
        }
    }
    if !hidden.is_multiple_of(EMBED_BF16_COLS) {
        return Err(refuse(EmbedShapeRule::OffVector {
            multiple: EMBED_BF16_COLS,
        }));
    }
    let elements = u64::from(rows) * u64::from(hidden);
    if elements > i32::MAX as u64 {
        return Err(refuse(EmbedShapeRule::TooManyElements { elements }));
    }
    Ok(CudaStep::Embed(EmbedBf16 {
        tokens,
        table,
        out,
        rows,
        hidden,
        vocab,
    }))
}

/// Lower the `Compute` driving `fusion` to the fused kernel's CUDA step. `computes` is every
/// `Compute` of the tape, in which a fold's absorbed steps are found by the slots its shape names.
pub fn lower_fused(
    cx: &StepCx<'_>,
    s: &TapeStep<'_>,
    fusion: &Fusion<CudaFusion>,
    computes: &[TapeStep<'_>],
) -> Result<CudaStep, CanonicalRefusal> {
    match fusion.kernel {
        CudaFusion::SiluMul => gated_mul(cx, s, fusion.shape, computes).map(CudaStep::SiluMul),
        CudaFusion::GeluMul => gated_mul(cx, s, fusion.shape, computes).map(CudaStep::GeluMul),
        CudaFusion::MoeRoute => moe_route(cx, s, fusion.shape, computes).map(CudaStep::MoeRoute),
    }
}

/// `MatmulTile` as a bf16 GEMM `C[m, n] = A[m, k] · B[n, k]ᵀ`: the tiled GEMM when `n` is a
/// multiple of its tile's, else the narrow GEMM when `n` is at most [`GEMM_NARROW_BF16_MAX_N`]
/// (the gated delta net's `a` and `b` projections, one column per value head). The choice is the
/// shape rule's ([`gemm_kernel`]), made here at expansion.
///
/// The IR states the weight as `[K, N]`; the bytes are the checkpoint's `[N, K]`, which is what
/// both kernels take (metal's `gemm.metal` the same). So `k` is the weight's IR rows (as metal's
/// `steps_from_tape::matmul` reads it), `n` the op's, and `m` the output's rows.
fn gemm(
    cx: &StepCx<'_>,
    s: &TapeStep<'_>,
    n: u32,
    weight: GemmWeight,
) -> Result<CudaStep, CanonicalRefusal> {
    let (step, node) = (s.step, s.node);
    let (act, wt) = match node.inputs.as_slice() {
        [act, wt] => (act, wt),
        [_] | [] => return Err(operand(s, 1, OperandRefusal::Missing)),
        [_, _, ..] => return Err(operand(s, 2, OperandRefusal::Extra)),
    };
    let m = cx.graph.shape(node.output.tensor).rows;
    let k = cx.graph.shape(wt.tensor).rows;
    let kernel = gemm_kernel(m, n, k, cx.facts.gemm_col_tile).map_err(|rule| {
        CanonicalRefusal::GemmShape {
            step,
            m,
            n,
            k,
            rule,
        }
    })?;
    if !matches!(weight, GemmWeight::Dense) {
        return Err(CanonicalRefusal::GemmWeight {
            step,
            weight: weight.kind(),
        });
    }
    whole_tensors(cx, s, &[(0, act), (1, wt)])?;
    let at = arena(cx)?;
    let mut inputs = s.inputs.iter();
    let a = activation(cx, s, 0, inputs.next())?;
    let b = weight_operand(cx, s, 1, inputs.next())?;
    let gemm = GemmBf16 {
        a: at(a),
        b,
        c: at(colour_of(cx, s.writes)),
        m,
        n,
        k,
    };
    Ok(kernel(gemm))
}

/// `RmsNorm { eps, gain }` as the bf16 RMSNorm: `x` (operand 0) a computed slot or the embedded
/// hidden, `gain` (operand 1) a `[1, cols]` weight, `rows` and `cols` the output's.
fn rms_norm(
    cx: &StepCx<'_>,
    s: &TapeStep<'_>,
    eps: f32,
    convention: GainConvention,
) -> Result<RmsNormBf16, CanonicalRefusal> {
    let (step, node) = (s.step, s.node);
    let (x, g) = match node.inputs.as_slice() {
        [x, g] => (x, g),
        _ => return Err(operand(s, 1, OperandRefusal::Missing)),
    };
    let TensorShape { rows, cols } = cx.graph.shape(node.output.tensor);
    row_shape(s, rows, cols, RMS_NORM_BF16_COLS)?;
    if !(eps.is_finite() && eps > 0.0) {
        return Err(CanonicalRefusal::NormEps { step, eps });
    }
    let gain = cx.graph.shape(g.tensor);
    if (gain.rows, gain.cols) != (1, cols) {
        return Err(CanonicalRefusal::GainShape {
            step,
            rows: gain.rows,
            cols: gain.cols,
            want: cols,
        });
    }
    same_shape(cx, s, 0, x, rows, cols)?;
    whole_tensors(cx, s, &[(0, x), (1, g)])?;
    let at = arena(cx)?;
    let mut inputs = s.inputs.iter();
    let xc = activation(cx, s, 0, inputs.next())?;
    let gain = weight_operand(cx, s, 1, inputs.next())?;
    Ok(RmsNormBf16 {
        x: at(xc),
        gain,
        y: at(colour_of(cx, s.writes)),
        rows,
        cols,
        eps_bits: eps.to_bits(),
        convention,
    })
}

/// `Elementwise(Add)` as the bf16 add: both operands computed slots or the embedded hidden, each
/// of the output's shape.
fn add(cx: &StepCx<'_>, s: &TapeStep<'_>) -> Result<AddBf16, CanonicalRefusal> {
    let node = s.node;
    let (a, b) = match node.inputs.as_slice() {
        [a, b] => (a, b),
        _ => return Err(operand(s, 1, OperandRefusal::Missing)),
    };
    let TensorShape { rows, cols } = cx.graph.shape(node.output.tensor);
    row_shape(s, rows, cols, ADD_BF16_COLS)?;
    same_shape(cx, s, 0, a, rows, cols)?;
    same_shape(cx, s, 1, b, rows, cols)?;
    whole_tensors(cx, s, &[(0, a), (1, b)])?;
    let at = arena(cx)?;
    let mut inputs = s.inputs.iter();
    let ac = activation(cx, s, 0, inputs.next())?;
    let bc = activation(cx, s, 1, inputs.next())?;
    Ok(AddBf16 {
        a: at(ac),
        b: at(bc),
        out: at(colour_of(cx, s.writes)),
        rows,
        cols,
    })
}

/// The `SiluMul` fold of `mul(silu(gate), up)`, or the `GeluMul` fold of `mul(gelu(gate), up)`,
/// as one bf16 launch: `gate` is the absorbed activation's operand 0, `up` the driving `Mul`'s
/// operand 1, `out` the `Mul`'s colour. The activation's own colour is never written.
fn gated_mul(
    cx: &StepCx<'_>,
    s: &TapeStep<'_>,
    shape: FusedShape,
    computes: &[TapeStep<'_>],
) -> Result<SiluMulBf16, CanonicalRefusal> {
    let step = s.step;
    let FusedShape::Gated {
        activation: silu,
        gate: None,
        up: None,
    } = shape
    else {
        return Err(CanonicalRefusal::FusionShape { step });
    };
    let act =
        (computes.iter().find(|c| c.writes == silu)).ok_or(CanonicalRefusal::ProducerMissing {
            step,
            op: s.node.op.name(),
        })?;
    let (silu_in, up) = match (act.node.inputs.as_slice(), s.node.inputs.as_slice()) {
        ([g], [_, u]) => (g, u),
        ([_], _) => return Err(operand(s, 1, OperandRefusal::Missing)),
        _ => return Err(operand(act, 0, OperandRefusal::Missing)),
    };
    let TensorShape { rows, cols } = cx.graph.shape(s.node.output.tensor);
    row_shape(s, rows, cols, SILU_MUL_BF16_COLS)?;
    same_shape(cx, act, 0, silu_in, rows, cols)?;
    same_shape(cx, s, 1, up, rows, cols)?;
    whole_tensors(cx, act, &[(0, silu_in)])?;
    whole_tensors(cx, s, &[(1, up)])?;
    let at = arena(cx)?;
    let gate = activation(cx, act, 0, act.inputs.iter().next())?;
    let up = activation(cx, s, 1, s.inputs.iter().nth(1))?;
    Ok(SiluMulBf16 {
        gate: at(gate),
        up: at(up),
        out: at(colour_of(cx, s.writes)),
        rows,
        cols,
    })
}

/// `Reshape` as a view: no launch, and its output is its input's buffer. CUDA's colour rule
/// aliases every `Reshape` to operand 0; this checks that the alias holds rather than trusting it.
/// Row-major, `[m, heads·hd]` and `[m·heads, hd]` are the same bytes in the same order, so both
/// row scales (`Times` and `Over`) are views.
fn view(cx: &StepCx<'_>, s: &TapeStep<'_>) -> Result<(), CanonicalRefusal> {
    let step = s.step;
    let refuse = |why| CanonicalRefusal::View { step, why };
    let [input] = s.node.inputs.as_slice() else {
        return Err(operand(s, 0, OperandRefusal::Missing));
    };
    let slot = match s.inputs.iter().next() {
        Some(ComputeInput::Computed(slots)) => match slots.as_slice() {
            [slot] => *slot,
            _ => return Err(operand(s, 0, OperandRefusal::SplitWriters(slots.len()))),
        },
        Some(ComputeInput::External { .. }) => return Err(refuse(ViewRefusal::OfASource)),
        None => return Err(operand(s, 0, OperandRefusal::Missing)),
    };
    let count = |t: TensorShape| u64::from(t.rows) * u64::from(t.cols);
    let (i, o) = (
        cx.graph.shape(input.tensor),
        cx.graph.shape(s.node.output.tensor),
    );
    if count(i) != count(o) {
        return Err(refuse(ViewRefusal::ElementCount {
            input: count(i),
            output: count(o),
        }));
    }
    whole_tensors(cx, s, &[(0, input)])?;
    // Unreachable under CUDA's rule (`op_abi::cuda_colour_rule` aliases a `Reshape` to operand 0,
    // and (a) above made that operand a step): kept so a rule change cannot make a view silently
    // read a buffer nobody wrote.
    let (ic, oc) = (colour_of(cx, slot), colour_of(cx, s.writes));
    if ic != oc {
        return Err(refuse(ViewRefusal::Colour {
            input: ic.index(),
            output: oc.index(),
        }));
    }
    Ok(())
}

/// `RopeRotate { head_dim }` as the bf16 partial rope: `x` (operand 0) a computed slot or the
/// embedded hidden; operands 1 and 2 a rotary `cos` and `sin`, which CUDA does not upload as
/// sources but reads as the tape's rotary table of their class. The rotated width and the pairing
/// are that class's (`CudaModelFacts::table_class`); the op carries only `head_dim`.
fn rope(cx: &StepCx<'_>, s: &TapeStep<'_>, head_dim: u32) -> Result<RopeBf16, CanonicalRefusal> {
    let (step, node) = (s.step, s.node);
    let [x, ..] = node.inputs.as_slice() else {
        return Err(operand(s, 0, OperandRefusal::Missing));
    };
    let class = cx.facts.table_class(table_is_local(cx, s));
    let class = class.expect("the global table's class is every model's");
    let rot_dim = class.rot_dim;
    let TensorShape { rows, cols } = cx.graph.shape(node.output.tensor);
    rope_shape(rows, cols, head_dim, rot_dim, cx.tokens).map_err(|rule| {
        CanonicalRefusal::RopeShape {
            step,
            op: node.op.name(),
            rows,
            cols,
            head_dim,
            rot_dim,
            rule,
        }
    })?;
    same_shape(cx, s, 0, x, rows, cols)?;
    whole_tensors(cx, s, &[(0, x)])?;
    let xc = activation(cx, s, 0, s.inputs.iter().next())?;
    let table = rotary_table(cx, s, table_is_local(cx, s))?;
    let at = arena(cx)?;
    Ok(RopeBf16 {
        x: at(xc),
        table,
        out: at(colour_of(cx, s.writes)),
        rows,
        cols,
        head_dim,
        rot_dim,
        pairing: class.pairing,
    })
}

/// Whether a rope's operand 1 is the local rotary `cos` of a model that has a local table (a
/// two-class model's base class). Any other operand 1 reads as the global table's, which
/// [`rotary_table`] then checks.
fn table_is_local(cx: &StepCx<'_>, s: &TapeStep<'_>) -> bool {
    let local = match s.inputs.iter().nth(1) {
        Some(ComputeInput::External { tensor, .. }) => {
            cx.bindings[tensor.index()] == SourceBinding::Cos { local: true }
        }
        _ => false,
    };
    local && cx.facts.table_class(true).is_some()
}

/// The attention class whose window is `mask`'s, and whether it reads the local rotary table:
/// a layer's class is its attention ops' own fact. A uniform model's one class has no window.
fn attn_class(cx: &StepCx<'_>, mask: AttnMask) -> Option<(CudaAttnClass, bool)> {
    let window = mask.window().map(NonZeroU32::get);
    [true, false]
        .into_iter()
        .filter_map(|local| cx.facts.table_class(local).map(|c| (c, local)))
        .find(|(c, _)| c.window == window)
}

/// Operands 1 and 2 of a rope (`RopeRotate` or `RopeAppend`): the rotary `cos` and `sin` of the
/// `local` or the global table, which CUDA does not upload as sources but reads as the tape's
/// rotary table of that class. Its offset.
fn rotary_table(
    cx: &StepCx<'_>,
    s: &TapeStep<'_>,
    local: bool,
) -> Result<ArenaOffset, CanonicalRefusal> {
    let (step, op) = (s.step, s.node.op.name());
    let mut inputs = s.inputs.iter().skip(1);
    for (k, want) in [
        (1, SourceBinding::Cos { local }),
        (2, SourceBinding::Sin { local }),
    ] {
        let binding = match inputs.next() {
            Some(ComputeInput::External { tensor, .. }) => {
                Some(cx.bindings[tensor.index()].clone())
            }
            Some(ComputeInput::Computed(_)) => None,
            None => return Err(operand(s, k, OperandRefusal::Missing)),
        };
        if binding.as_ref() != Some(&want) {
            return Err(CanonicalRefusal::RotaryOperand {
                step,
                op,
                operand: k,
                binding,
            });
        }
    }
    // Operand 1 is the table's `cos`, so the arena placed the table, unless the arena has no
    // layout at all (refused below, by the caller's `arena`).
    let table = match local {
        true => cx.rotary_local,
        false => cx.rotary,
    };
    match cx.offsets {
        Ok(_) => table.ok_or(CanonicalRefusal::RotaryOperand {
            step,
            op,
            operand: 1,
            binding: None,
        }),
        Err(e) => Err(CanonicalRefusal::Arena(e)),
    }
}

/// Operands `k` and `k + 1` of a KV step: layer `layer`'s `PrefixK` and `PrefixV` sources.
fn prefix_operands(
    s: &TapeStep<'_>,
    cx: &StepCx<'_>,
    k: u8,
    layer: u64,
) -> Result<(), CanonicalRefusal> {
    let mut inputs = s.inputs.iter().skip(usize::from(k));
    for (k, name, want) in [
        (k, "PrefixK", SourceBinding::PrefixK { layer }),
        (k + 1, "PrefixV", SourceBinding::PrefixV { layer }),
    ] {
        let binding = match inputs.next() {
            Some(ComputeInput::External { tensor, .. }) => {
                Some(cx.bindings[tensor.index()].clone())
            }
            Some(ComputeInput::Computed(_)) => None,
            None => return Err(operand(s, k, OperandRefusal::Missing)),
        };
        if binding.as_ref() != Some(&want) {
            return Err(CanonicalRefusal::KvOperand {
                step: s.step,
                op: s.node.op.name(),
                operand: k,
                layer,
                want: name,
                binding,
            });
        }
    }
    Ok(())
}

/// The canonical's KV pool, the regions the arena placed for it, and the pool index of model
/// layer `layer`. Called after [`prefix_operands`] found `layer`'s `PrefixK` among the step's
/// operands, so `kv_cache` found a pool holding `layer`, and the arena (when it has a layout)
/// placed the regions.
fn kv_pool<'a>(
    cx: &StepCx<'a>,
    layer: u32,
) -> Result<(&'a KvPool, KvRegions, u32), CanonicalRefusal> {
    let pool = cx.kv.map_err(CanonicalRefusal::KvCache)?;
    cx.offsets.map_err(CanonicalRefusal::Arena)?;
    let pool = pool.expect("a PrefixK operand is bound, so kv_cache found a pool");
    let regions =
        (cx.kv_regions).expect("a PrefixK source is bound, so the arena placed the KV regions");
    let index = (pool.index_of(u64::from(layer)))
        .expect("the pool's layers are the PrefixK sources', and the operand is this layer's");
    Ok((pool, regions, index))
}

/// A computed-slot operand `k` of `s`: its one slot.
fn slot_operand(cx: &StepCx<'_>, s: &TapeStep<'_>, k: u8) -> Result<SlotId, CanonicalRefusal> {
    match s.inputs.iter().nth(usize::from(k)) {
        Some(ComputeInput::Computed(slots)) => match slots.as_slice() {
            [slot] => Ok(*slot),
            _ => Err(operand(s, k, OperandRefusal::SplitWriters(slots.len()))),
        },
        Some(ComputeInput::External { tensor, .. }) => Err(operand(
            s,
            k,
            OperandRefusal::NotASlot {
                source: tensor.index(),
                binding: cx.bindings[tensor.index()].clone(),
            },
        )),
        None => Err(operand(s, k, OperandRefusal::Missing)),
    }
}

/// `RopeAppend { head_dim, layer }` as two launches: K (operand 0, a computed slot) rotated by
/// the tape's rotary table of its class (operands 1 and 2, as a `RopeRotate`'s) into this step's
/// colour, then the rotated K and V (operand 3, a computed slot) written into the KV cache of
/// `layer`, whose `PrefixK` and `PrefixV` are operands 4 and 5. Its class is the one of its mask's
/// window ([`attn_class`]), the same as its attention's. The rope's shape rule is K's
/// `[rows, cols]`.
fn rope_append(
    cx: &StepCx<'_>,
    s: &TapeStep<'_>,
    head_dim: u32,
    layer: u32,
    attn: AttnMask,
    pairing: RopeFormTag,
) -> Result<RopeAppendBf16, CanonicalRefusal> {
    let (step, node) = (s.step, s.node);
    let op = node.op.name();
    if pairing == RopeFormTag::Interleaved {
        return Err(CanonicalRefusal::RopePairing { step });
    }
    let (class, local) =
        attn_class(cx, attn).ok_or(CanonicalRefusal::SlidingWindow { step, op })?;
    window_rule(s, rows_of(cx, s), head_dim, class.window)?;
    if head_dim != class.head_dim {
        return Err(CanonicalRefusal::KvHeadDim {
            step,
            op,
            head_dim,
            want: class.head_dim,
        });
    }
    let rot_dim = class.rot_dim;
    let (k, v) = match node.inputs.as_slice() {
        [k, _, _, v, _, _] => (k, v),
        _ => return Err(operand(s, 5, OperandRefusal::Missing)),
    };
    let TensorShape { rows, cols } = cx.graph.shape(node.output.tensor);
    rope_shape(rows, cols, head_dim, rot_dim, cx.tokens).map_err(|rule| {
        CanonicalRefusal::RopeShape {
            step,
            op,
            rows,
            cols,
            head_dim,
            rot_dim,
            rule,
        }
    })?;
    same_shape(cx, s, 0, k, rows, cols)?;
    same_shape(cx, s, 3, v, rows, cols)?;
    whole_tensors(cx, s, &[(0, k), (3, v)])?;
    let kc = colour_of(cx, slot_operand(cx, s, 0)?);
    let vc = colour_of(cx, slot_operand(cx, s, 3)?);
    let table = rotary_table(cx, s, local)?;
    prefix_operands(s, cx, 4, u64::from(layer))?;
    let (pool, regions, index) = kv_pool(cx, layer)?;
    let cache = pool.class_of(index).expect("every pool layer has a class");
    if cols != cache.kv_heads * cache.head_dim || cache.head_dim != head_dim {
        return Err(CanonicalRefusal::KvGeometry {
            step,
            op,
            kv_heads: cache.kv_heads,
            cols,
        });
    }
    let at = arena(cx)?;
    Ok(RopeAppendBf16 {
        k: at(kc),
        v: at(vc),
        table,
        slot_mapping: regions.slot_mapping.offset,
        k_out: at(colour_of(cx, s.writes)),
        layer: index,
        rows,
        cols,
        head_dim,
        rot_dim,
        pairing: class.pairing,
    })
}

/// The rows of `s`'s output.
fn rows_of(cx: &StepCx<'_>, s: &TapeStep<'_>) -> u32 {
    cx.graph.shape(s.node.output.tensor).rows
}

/// Refuse a window that is not a non-zero multiple of [`ATTN_WINDOW_MULTIPLE`] keys: the
/// windowed kernels skip whole KV tiles below the window, so a tile never sits across its edge.
fn window_rule(
    s: &TapeStep<'_>,
    rows: u32,
    head_dim: u32,
    window: Option<u32>,
) -> Result<(), CanonicalRefusal> {
    match window {
        Some(w) if w == 0 || !w.is_multiple_of(ATTN_WINDOW_MULTIPLE) => {
            Err(CanonicalRefusal::AttnShape {
                step: s.step,
                rows,
                head_dim,
                rule: AttnShapeRule::WindowOffBlock {
                    window: w,
                    multiple: ATTN_WINDOW_MULTIPLE,
                },
            })
        }
        _ => Ok(()),
    }
}

/// `AttnDecode` as attention over the paged KV cache: the bf16 decode attention for a one-row
/// bucket, the bf16 causal prefill attention for a bucket of a multiple of
/// [`ATTN_PREFILL_BF16_ROWS`] rows, chosen here at expansion; any other row count is refused, and
/// so, last, is a one-row bucket whose GQA group is above [`ATTN_DECODE_BF16_MAX_GROUP`]. The
/// layer's attention class is the one of the mask's window ([`attn_class`]): its head dim one of
/// [`ATTN_BF16_HEAD_DIMS`], its window (the step's) a multiple of [`ATTN_WINDOW_MULTIPLE`].
/// `q` (operand 0) is a computed slot `[rows, q_heads · head_dim]`; operands 1 and 2 the layer's
/// `PrefixK` and `PrefixV`; operand 3 that layer's `RopeAppend` and operand 4 the V it read. Both
/// kernels read K and V from the cache the `RopeAppend` wrote, not from operands 3 and 4, which
/// stay live up to here because the IR says the attention reads them.
fn attention(
    cx: &StepCx<'_>,
    s: &TapeStep<'_>,
    geom: ModelAttnGeometry,
    scale: f32,
    mask: AttnMask,
    computes: &[TapeStep<'_>],
) -> Result<CudaStep, CanonicalRefusal> {
    let (step, node) = (s.step, s.node);
    let op = node.op.name();
    let rows = cx.graph.shape(node.output.tensor).rows;
    let head_dim = geom.hd().get();
    let kernel: fn(AttnPrefillBf16) -> CudaStep = match rows {
        1 => CudaStep::AttnDecode,
        _ => CudaStep::AttnPrefill,
    };
    let (class, _) = attn_class(cx, mask).ok_or(CanonicalRefusal::SlidingWindow { step, op })?;
    let q = match node.inputs.as_slice() {
        [q, _, _, _, _] => q,
        [_, _, _, _, _, _] => return Err(CanonicalRefusal::AttnSinks { step }),
        _ => return Err(operand(s, 4, OperandRefusal::Missing)),
    };
    if !ATTN_BF16_HEAD_DIMS.contains(&head_dim) {
        return Err(CanonicalRefusal::AttnShape {
            step,
            rows,
            head_dim,
            rule: AttnShapeRule::HeadDim {
                want: class.head_dim,
            },
        });
    }
    if rows != 1 && (rows == 0 || !rows.is_multiple_of(ATTN_PREFILL_BF16_ROWS)) {
        return Err(CanonicalRefusal::AttnShape {
            step,
            rows,
            head_dim,
            rule: AttnShapeRule::RowsOffTile {
                multiple: ATTN_PREFILL_BF16_ROWS,
            },
        });
    }
    window_rule(s, rows, head_dim, class.window)?;
    if head_dim != class.head_dim {
        return Err(CanonicalRefusal::KvHeadDim {
            step,
            op,
            head_dim,
            want: class.head_dim,
        });
    }
    let q_width = geom.q_width();
    same_shape(cx, s, 0, q, rows, q_width)?;
    whole_tensors(cx, s, &[(0, q)])?;
    let qc = colour_of(cx, slot_operand(cx, s, 0)?);
    // Operand 3 is its layer's `RopeAppend`, operand 4 the V that append read.
    let new_k = slot_operand(cx, s, 3)?;
    let append = (computes.iter().find(|c| c.writes == new_k))
        .ok_or(CanonicalRefusal::ProducerMissing { step, op })?;
    let SubOp::RopeAppend { layer, .. } = append.node.op else {
        return Err(CanonicalRefusal::AttnProducer {
            step,
            operand: 3,
            why: ProducerRefusal::NotARopeAppend,
        });
    };
    let new_v = slot_operand(cx, s, 4)?;
    if slot_operand(cx, append, 3)? != new_v {
        return Err(CanonicalRefusal::AttnProducer {
            step,
            operand: 4,
            why: ProducerRefusal::NotItsV,
        });
    }
    let prefix_layer = match s.inputs.iter().nth(1) {
        Some(ComputeInput::External { tensor, .. }) => match cx.bindings[tensor.index()] {
            SourceBinding::PrefixK { layer } => Some(layer),
            _ => None,
        },
        _ => None,
    };
    if let Some(prefix) = prefix_layer
        && prefix != u64::from(layer)
    {
        return Err(CanonicalRefusal::AttnProducer {
            step,
            operand: 3,
            why: ProducerRefusal::OtherLayer { layer },
        });
    }
    prefix_operands(s, cx, 1, u64::from(layer))?;
    let (pool, regions, index) = kv_pool(cx, layer)?;
    let (q_heads, kv_heads) = (geom.nqh().get(), geom.nkvh().get());
    let cache = pool.class_of(index).expect("every pool layer has a class");
    if kv_heads != cache.kv_heads || head_dim != cache.head_dim {
        return Err(CanonicalRefusal::KvGeometry {
            step,
            op,
            kv_heads,
            cols: geom.kv_width(),
        });
    }
    let group = geom.gqa().get();
    if rows == 1 && group > ATTN_DECODE_BF16_MAX_GROUP {
        return Err(CanonicalRefusal::AttnShape {
            step,
            rows,
            head_dim,
            rule: AttnShapeRule::GroupAboveMax {
                group,
                max: ATTN_DECODE_BF16_MAX_GROUP,
            },
        });
    }
    let at = arena(cx)?;
    Ok(kernel(AttnPrefillBf16 {
        q: at(qc),
        block_table: regions.block_table.offset,
        sequence: regions.sequence.offset,
        out: at(colour_of(cx, s.writes)),
        layer: index,
        rows,
        q_heads,
        kv_heads,
        head_dim,
        scale_bits: scale.to_bits(),
        window: class.window,
    }))
}

/// The model layer of a `GatedDeltaNet`'s weight bundle (operand 4): a `SourceBinding::Weight`
/// with an unroll index that fits a `u32`. `None` for any other operand.
pub fn gdn_bundle_layer(bindings: &[SourceBinding], s: &TapeStep<'_>) -> Option<u32> {
    match s.inputs.iter().nth(4) {
        Some(ComputeInput::External { tensor, .. }) => match bindings[tensor.index()] {
            SourceBinding::Weight {
                index: Some(UnrollIndex(layer)),
                ..
            } => u32::try_from(layer).ok(),
            _ => None,
        },
        _ => None,
    }
}

/// `GatedDeltaNet(qkv, z, a, b, bundle)` as one prefill chunk of the Gated-DeltaNet layer: four
/// launches (conv1d, gating, recurrent scan, gated norm). Operands 0-3 are computed slots or the
/// embedded hidden, `qkv [rows, conv_dim]`, `z [rows, value_dim]`, `a` and `b [rows, nv]`;
/// operand 4 is the layer's weight bundle, a `[1, 1]` source bound to a model layer's weight,
/// which the device holds packed as `gdn_bundle`. The op carries no geometry: the value heads
/// and their dim are the operands', the key heads, their dim, the conv kernel and `eps` the
/// model's (`CudaModelFacts::gdn`).
fn gated_delta_net(
    cx: &StepCx<'_>,
    s: &TapeStep<'_>,
) -> Result<GatedDeltaNetBf16, CanonicalRefusal> {
    let (step, node) = (s.step, s.node);
    let (qkv, z, a, b, bundle) = match node.inputs.as_slice() {
        [qkv, z, a, b, bundle] => (qkv, z, a, b, bundle),
        [_, _, _, _, _, _, ..] => return Err(operand(s, 5, OperandRefusal::Extra)),
        _ => return Err(operand(s, 4, OperandRefusal::Missing)),
    };
    let facts = cx.facts.gdn.ok_or(CanonicalRefusal::GdnFacts { step })?;
    let refuse = |rule| CanonicalRefusal::GdnShape { step, rule };
    let shape = |r: &TensorRegion| cx.graph.shape(r.tensor);
    let TensorShape {
        rows,
        cols: out_cols,
    } = shape(&node.output);
    let (conv_dim, value_dim, nv) = (shape(qkv).cols, shape(z).cols, shape(a).cols);
    let (nk, head_k, conv_kernel) = (facts.num_k_heads, facts.head_k, facts.conv_kernel);
    if head_k != GDN_HEAD_K {
        return Err(refuse(GdnShapeRule::HeadK { head_k }));
    }
    if nk == 0 || nv == 0 || !nv.is_multiple_of(nk) || !value_dim.is_multiple_of(nv) {
        return Err(refuse(GdnShapeRule::HeadsNotGrouped { nk, nv }));
    }
    let head_v = value_dim / nv;
    if head_v == 0 || !head_v.is_multiple_of(GDN_HEAD_V_MULTIPLE) || head_v > GDN_HEAD_V_MAX {
        return Err(refuse(GdnShapeRule::HeadV { head_v }));
    }
    let (kmin, kmax) = GDN_CONV_KERNEL_RANGE;
    if !(kmin..=kmax).contains(&conv_kernel) {
        return Err(refuse(GdnShapeRule::ConvKernel { k: conv_kernel }));
    }
    // `nk · head_k ≤ nv · 128 ≤ value_dim`, so `want` is at most `3 · u32::MAX`; in u64 it fits.
    let want = 2 * u64::from(nk) * u64::from(head_k) + u64::from(value_dim);
    if u64::from(conv_dim) != want {
        let want = u32::try_from(want).unwrap_or(u32::MAX);
        return Err(refuse(GdnShapeRule::Widths { conv_dim, want }));
    }
    // The scratch is sized at the bucket's rows, so the op's rows must be those too.
    let operand_rows = [qkv, z, a, b].map(|r| shape(r).rows);
    if operand_rows.iter().any(|&r| r != rows) || cx.tokens.is_some_and(|t| t != rows) {
        return Err(refuse(GdnShapeRule::RowsDisagree));
    }
    let elements = u64::from(rows) * u64::from(conv_dim);
    if rows == 0 || elements > i32::MAX as u64 {
        return Err(refuse(GdnShapeRule::TooManyElements { elements }));
    }
    same_shape(cx, s, 1, z, rows, out_cols)?;
    same_shape(cx, s, 3, b, rows, nv)?;
    whole_tensors(cx, s, &[(0, qkv), (1, z), (2, a), (3, b), (4, bundle)])?;
    let mut inputs = s.inputs.iter();
    let [qc, zc, ac, bc] = [0, 1, 2, 3].map(|k| activation(cx, s, k, inputs.next()));
    let (qc, zc, ac, bc) = (qc?, zc?, ac?, bc?);
    let bundle = weight_operand(cx, s, 4, inputs.next())?;
    let layer = gdn_bundle_layer(cx.bindings, s).ok_or(CanonicalRefusal::GdnLayer { step })?;
    let at = arena(cx)?;
    // The arena sized the scratch, and the worker sizes the state pool, by the canonical's first
    // GDN op's geometry; one of another geometry would need another scratch and another pool.
    let regions = (cx.gdn_regions).ok_or(CanonicalRefusal::GdnPool { step })?;
    let geometry = GdnWidths {
        conv_dim,
        value_dim,
        nv,
    };
    if geometry != regions.geometry {
        return Err(CanonicalRefusal::GdnPool { step });
    }
    let index = (cx.gdn_layers.iter().position(|&l| l == layer))
        .ok_or(CanonicalRefusal::GdnLayer { step })?;
    Ok(GatedDeltaNetBf16 {
        qkv: at(qc),
        z: at(zc),
        a: at(ac),
        b: at(bc),
        bundle,
        out: at(colour_of(cx, s.writes)),
        scratch: regions.scratch.offset,
        sequence: regions.sequence.offset,
        layer: index as u32,
        rows,
        num_k_heads: nk,
        num_v_heads: nv,
        head_k,
        head_v,
        conv_kernel,
        eps_bits: facts.eps.to_bits(),
    })
}

/// `GateSplit { half_cols }` as the bf16 per-head split: `qg` (operand 0) `[rows, 2·half_cols]`,
/// the query this step's colour and the gate its second colour. `head_dim` is the model's: the op
/// carries only `half_cols`.
fn gate_split(
    cx: &StepCx<'_>,
    s: &TapeStep<'_>,
    half_cols: u32,
) -> Result<GateSplitBf16, CanonicalRefusal> {
    let (step, node) = (s.step, s.node);
    let head_dim = cx.facts.head_dim;
    let [qg] = node.inputs.as_slice() else {
        return Err(operand(s, 0, OperandRefusal::Missing));
    };
    let rows = cx.graph.shape(node.output.tensor).rows;
    gate_shape(rows, half_cols, head_dim).map_err(|rule| CanonicalRefusal::GateShape {
        step,
        rows,
        half_cols,
        head_dim,
        rule,
    })?;
    same_shape(cx, s, 0, qg, rows, 2 * half_cols)?;
    whole_tensors(cx, s, &[(0, qg)])?;
    let gate =
        (cx.colouring.second_colour_of(s.writes)).ok_or(CanonicalRefusal::SecondOutput { step })?;
    let qgc = activation(cx, s, 0, s.inputs.iter().next())?;
    let at = arena(cx)?;
    Ok(GateSplitBf16 {
        qg: at(qgc),
        q: at(colour_of(cx, s.writes)),
        gate: at(gate),
        rows,
        half_cols,
        head_dim,
    })
}

/// `GateApply` as the bf16 `attn · sigmoid(gate)`: `attn` (operand 0) a computed slot or the
/// embedded hidden. Operand 1 names a `GateSplit`'s slot, and means its SECOND output, the gate:
/// the shared graph has one tensor for both, so read naively it would gate by the query. As metal
/// does (`metal/compiler/src/steps_from_tape.rs`), the gate binds the producer's second colour.
fn gate_apply(
    cx: &StepCx<'_>,
    s: &TapeStep<'_>,
    computes: &[TapeStep<'_>],
) -> Result<GateApplyBf16, CanonicalRefusal> {
    let (step, node) = (s.step, s.node);
    let (a, g) = match node.inputs.as_slice() {
        [a, g] => (a, g),
        _ => return Err(operand(s, 1, OperandRefusal::Missing)),
    };
    let TensorShape { rows, cols } = cx.graph.shape(node.output.tensor);
    row_shape(s, rows, cols, GATE_APPLY_BF16_COLS)?;
    same_shape(cx, s, 0, a, rows, cols)?;
    same_shape(cx, s, 1, g, rows, cols)?;
    whole_tensors(cx, s, &[(0, a), (1, g)])?;
    let mut inputs = s.inputs.iter();
    let ac = activation(cx, s, 0, inputs.next())?;
    let refuse = CanonicalRefusal::GateOperand { step };
    let split = match inputs.next() {
        Some(ComputeInput::Computed(slots)) => match slots.as_slice() {
            [slot] => computes.iter().find(|c| c.writes == *slot),
            _ => None,
        },
        _ => None,
    }
    .filter(|c| matches!(c.node.op, SubOp::GateSplit { .. }))
    .ok_or(refuse)?;
    let gate = (cx.colouring.second_colour_of(split.writes))
        .ok_or(CanonicalRefusal::SecondOutput { step: split.step })?;
    let at = arena(cx)?;
    Ok(GateApplyBf16 {
        attn: at(ac),
        gate: at(gate),
        out: at(colour_of(cx, s.writes)),
        rows,
        cols,
    })
}

/// The one activation operand (operand 0) of a unary row op, `[rows, cols]` as its output, and
/// that output's shape under the row rule of `multiple`.
fn unary_row(
    cx: &StepCx<'_>,
    s: &TapeStep<'_>,
    multiple: u32,
) -> Result<(Colour, u32, u32), CanonicalRefusal> {
    let x = match s.node.inputs.as_slice() {
        [x] => x,
        [] => return Err(operand(s, 0, OperandRefusal::Missing)),
        [_, ..] => return Err(operand(s, 1, OperandRefusal::Extra)),
    };
    let TensorShape { rows, cols } = cx.graph.shape(s.node.output.tensor);
    row_shape(s, rows, cols, multiple)?;
    same_shape(cx, s, 0, x, rows, cols)?;
    whole_tensors(cx, s, &[(0, x)])?;
    let xc = activation(cx, s, 0, s.inputs.iter().next())?;
    Ok((xc, rows, cols))
}

/// Refuse a compile-time scalar of a row op unless it is finite and, when `positive`, above 0.
fn finite_scalar(s: &TapeStep<'_>, value: f32, positive: bool) -> Result<(), CanonicalRefusal> {
    if value.is_finite() && (!positive || value > 0.0) {
        return Ok(());
    }
    Err(CanonicalRefusal::Scalar {
        step: s.step,
        op: s.node.op.name(),
        value,
    })
}

/// `ScalarMul { scale }` as the bf16 scale (Gemma's embedding scale): `x` (operand 0) a computed
/// slot or the embedded hidden, times the op's `scale`.
fn scalar_mul(
    cx: &StepCx<'_>,
    s: &TapeStep<'_>,
    scale: f32,
) -> Result<ScaleBf16, CanonicalRefusal> {
    let (xc, rows, cols) = unary_row(cx, s, SCALE_BF16_COLS)?;
    finite_scalar(s, scale, false)?;
    let at = arena(cx)?;
    Ok(ScaleBf16 {
        x: at(xc),
        out: at(colour_of(cx, s.writes)),
        rows,
        cols,
        scale_bits: scale.to_bits(),
    })
}

/// `RmsNormUnit { eps }` as the bf16 RMSNorm with no gain (Gemma's `v_norm`): `x` (operand 0) a
/// computed slot or the embedded hidden. The norm is per KV head: when the row is one attention
/// class's `kv_heads · head_dim` columns, the step normalises each `head_dim`-wide head on its own,
/// as `[rows · kv_heads, head_dim]` (the row-major `[rows, kv_heads · head_dim]` buffer viewed with
/// one head per row). Gemma's sliding v is 8 heads of 256, `(8m, 256)`; its global v reaches the
/// op already viewed per head (one head of 512, the `k_eq_v` reshape), `(2m, 512)`. Any other row
/// is normalised whole. The same rule as Metal's (`RmsNormUnitImpl::fan_out`: sliding 256 × 8,
/// global 512 × 1) and the repo's HF reference (`golden_gen_carrier.py`'s `rmsnorm_unit`).
fn rms_norm_unit(
    cx: &StepCx<'_>,
    s: &TapeStep<'_>,
    eps: f32,
) -> Result<RmsNormUnitBf16, CanonicalRefusal> {
    let (xc, rows, cols) = unary_row(cx, s, RMS_NORM_BF16_COLS)?;
    if !(eps.is_finite() && eps > 0.0) {
        return Err(CanonicalRefusal::NormEps { step: s.step, eps });
    }
    let (rows, cols) = per_head_rows(cx, rows, cols);
    let at = arena(cx)?;
    Ok(RmsNormUnitBf16 {
        x: at(xc),
        y: at(colour_of(cx, s.writes)),
        rows,
        cols,
        eps_bits: eps.to_bits(),
    })
}

/// `(rows, cols)` as one row per KV head, `(rows · kv_heads, head_dim)`, when `cols` is one of the
/// model's attention classes' `kv_heads · head_dim` with more than one head; else as it is. A
/// uniform model has no classes, so its rows stay whole. The two classes' widths differ (Gemma's
/// 2048 and 1024), so a row matches at most one. `rows · cols` is unchanged, within `unary_row`'s
/// bound.
fn per_head_rows(cx: &StepCx<'_>, rows: u32, cols: u32) -> (u32, u32) {
    let Some(classes) = cx.facts.classes else {
        return (rows, cols);
    };
    for class in [classes.base, classes.global] {
        let heads = class.kv_heads;
        if heads > 1 && class.head_dim.checked_mul(heads) == Some(cols) {
            return (rows * heads, class.head_dim);
        }
    }
    (rows, cols)
}

/// `ScalarWeightMul` as the bf16 layer scale (Gemma's `layer_scalar`): `x` (operand 0) a computed
/// slot or the embedded hidden, times operand 1, a `[1, 1]` model weight.
fn layer_scale(cx: &StepCx<'_>, s: &TapeStep<'_>) -> Result<LayerScaleBf16, CanonicalRefusal> {
    let (x, w) = match s.node.inputs.as_slice() {
        [x, w] => (x, w),
        [_] | [] => return Err(operand(s, 1, OperandRefusal::Missing)),
        [_, _, ..] => return Err(operand(s, 2, OperandRefusal::Extra)),
    };
    let TensorShape { rows, cols } = cx.graph.shape(s.node.output.tensor);
    row_shape(s, rows, cols, SCALE_BF16_COLS)?;
    same_shape(cx, s, 0, x, rows, cols)?;
    same_shape(cx, s, 1, w, 1, 1)?;
    whole_tensors(cx, s, &[(0, x), (1, w)])?;
    let at = arena(cx)?;
    let mut inputs = s.inputs.iter();
    let xc = activation(cx, s, 0, inputs.next())?;
    let scale = weight_operand(cx, s, 1, inputs.next())?;
    Ok(LayerScaleBf16 {
        x: at(xc),
        scale,
        out: at(colour_of(cx, s.writes)),
        rows,
        cols,
    })
}

/// `TanhSoftCap { cap }` as the bf16 soft-cap (Gemma's final logits): `x` (operand 0) a computed
/// slot or the embedded hidden, `cap` finite and above 0.
fn soft_cap(cx: &StepCx<'_>, s: &TapeStep<'_>, cap: f32) -> Result<SoftCapBf16, CanonicalRefusal> {
    let (xc, rows, cols) = unary_row(cx, s, SOFT_CAP_BF16_COLS)?;
    finite_scalar(s, cap, true)?;
    let at = arena(cx)?;
    Ok(SoftCapBf16 {
        x: at(xc),
        out: at(colour_of(cx, s.writes)),
        rows,
        cols,
        cap_bits: cap.to_bits(),
    })
}

// ── The MoE block ───────────────────────────────────────────────────

/// A MoE op's refusal.
fn moe_refusal(s: &TapeStep<'_>, rule: MoeShapeRule) -> CanonicalRefusal {
    CanonicalRefusal::MoeShape {
        step: s.step,
        op: s.node.op.name(),
        rule,
    }
}

/// The tape's MoE scratch, which the arena placed because the canonical has a MoE op; or the
/// arena's refusal.
fn moe_tape(cx: &StepCx<'_>, s: &TapeStep<'_>) -> Result<MoeTape, CanonicalRefusal> {
    cx.offsets.map_err(CanonicalRefusal::Arena)?;
    cx.moe.ok_or(moe_refusal(s, MoeShapeRule::Geometry))
}

/// The step writing operand `k` of `s`, which must be a computed slot written by a step whose op
/// `is` accepts: the MoE dataflow `op_abi::cuda_moe_write` states, checked rather than trusted.
fn moe_producer<'c, 'a>(
    cx: &StepCx<'_>,
    s: &TapeStep<'_>,
    k: u8,
    computes: &'c [TapeStep<'a>],
    is: impl Fn(&SubOp) -> bool,
) -> Result<&'c TapeStep<'a>, CanonicalRefusal> {
    let slot = slot_operand(cx, s, k)?;
    (computes.iter().find(|c| c.writes == slot))
        .filter(|c| is(&c.node.op))
        .ok_or(CanonicalRefusal::MoeOperand {
            step: s.step,
            op: s.node.op.name(),
            operand: k,
        })
}

/// The bucket's MoE geometry checks: `experts` and `top_k` the scratch's, the routing kernel's
/// expert count and top-k, and `rows` the bucket's tokens.
fn moe_counts(
    cx: &StepCx<'_>,
    s: &TapeStep<'_>,
    moe: &MoeTape,
    (experts, top_k): (u32, u32),
) -> Result<(), CanonicalRefusal> {
    if !MOE_EXPERT_COUNTS.contains(&experts) || top_k == 0 || top_k > MOE_ROUTE_MAX_TOP_K {
        return Err(moe_refusal(s, MoeShapeRule::Experts { experts, top_k }));
    }
    if (experts, top_k) != (moe.experts, moe.top_k) {
        return Err(moe_refusal(s, MoeShapeRule::Geometry));
    }
    let rows = cx.graph.shape(s.node.output.tensor).rows;
    if cx.tokens != Some(rows) {
        return Err(moe_refusal(s, MoeShapeRule::Rows));
    }
    Ok(())
}

/// The expert bundle of a router, and the buffer its weight is: Qwen's shared-fused router reads
/// its bundle, Gemma's its own projection beside the SwitchGLU bank. `None` for a router CUDA has
/// no MoE step for.
fn router_weight(router: RouterBundle) -> Option<(ExpertBundle, MoeWeight)> {
    match router {
        RouterBundle::SharedFused => Some((ExpertBundle::SharedFused, MoeWeight::Bundle)),
        RouterBundle::Gemma => Some((ExpertBundle::SwitchGlu, MoeWeight::Own)),
        RouterBundle::Fused | RouterBundle::GptOss => None,
    }
}

/// `RouterLogits { experts, router }` as the bf16 GEMM over the router: at byte 0 of the layer's
/// MoE bundle (Qwen), or Gemma's own `[experts, hidden]` projection. `x` (operand 0) is `[m,
/// hidden]`: a computed slot or the embedded hidden, or (Gemma) the `RouterNorm` it reads, in the
/// MoE scratch's router-norm rows. The weight (operand 1) is a source bound to the bundle or the
/// projection. The tiled GEMM, or the GEMV at one row, as [`gemm_kernel`] chooses for a
/// `MatmulTile` (the router's `n` is its experts, never narrow). Only a router whose bundle
/// `op_abi::CUDA_GROUPED_EXPERTS` names.
fn router_logits(
    cx: &StepCx<'_>,
    s: &TapeStep<'_>,
    experts: u32,
    router: RouterBundle,
    computes: &[TapeStep<'_>],
) -> Result<RouterLogitsBf16, CanonicalRefusal> {
    let (step, node) = (s.step, s.node);
    let weight = match router_weight(router) {
        Some((bundle, weight)) if CUDA_GROUPED_EXPERTS.contains(&bundle) => weight,
        _ => return Err(moe_refusal(s, MoeShapeRule::Bundle)),
    };
    let (x, w) = match node.inputs.as_slice() {
        [x, w] => (x, w),
        _ => return Err(operand(s, 1, OperandRefusal::Missing)),
    };
    let moe = moe_tape(cx, s)?;
    moe_counts(cx, s, &moe, (experts, moe.top_k))?;
    let TensorShape { rows: m, cols } = cx.graph.shape(node.output.tensor);
    let hidden = cx.graph.shape(x.tensor).cols;
    if cols != experts || hidden != moe.hidden {
        return Err(moe_refusal(s, MoeShapeRule::Geometry));
    }
    same_shape(cx, s, 0, x, m, hidden)?;
    whole_tensors(cx, s, &[(0, x), (1, w)])?;
    if weight == MoeWeight::Own {
        same_shape(cx, s, 1, w, hidden, experts)?;
    }
    let col64 =
        cx.facts.gemm_col_tile == GemmColTile::Col64 && !experts.is_multiple_of(GEMM_BF16_TILE.1);
    let kernel = match m {
        1 => gemv_shape(experts, hidden).map(|()| MatmulKernel::Gemv),
        _ if col64 => gemm_col64_shape(m, experts, hidden).map(|()| MatmulKernel::Col64),
        _ => gemm_shape(m, experts, hidden).map(|()| MatmulKernel::Tiled),
    }
    .map_err(|rule| CanonicalRefusal::GemmShape {
        step,
        m,
        n: experts,
        k: hidden,
        rule,
    })?;
    let at = arena(cx)?;
    let mut inputs = s.inputs.iter();
    // Gemma's router reads its own pre-norm, which the scratch holds; Qwen's reads the hidden.
    let x = match router {
        RouterBundle::Gemma => {
            moe_producer(cx, s, 0, computes, |op| {
                matches!(op, SubOp::RouterNorm { .. })
            })?;
            inputs.next();
            let region = moe
                .router_norm
                .ok_or(moe_refusal(s, MoeShapeRule::Geometry))?;
            region.offset
        }
        _ => at(activation(cx, s, 0, inputs.next())?),
    };
    let bundle = weight_operand(cx, s, 1, inputs.next())?;
    Ok(RouterLogitsBf16 {
        x,
        bundle,
        weight,
        logits: moe.logits.offset,
        m,
        experts,
        hidden,
        inter: moe.inter,
        kernel,
    })
}

/// The `MoeRoute` fold its `RouteTopK { k }` drives, as the one routing launch, in one of two
/// programs:
/// - Qwen's: the folded softmax (the fold's `pre`) over the `RouterLogits` it reads, its argsort
///   (the fold's `sort`), the top-k, the gathered scores and, when the fold took one as tail stage
///   1, their renorm;
/// - Gemma's: no `pre`, the argsort and top-k over the raw logits, the gathered scores times the
///   `RouteScale` (tail stage 0), their softmax (stage 1), and the `RouteExpertScale` by the
///   router's per-expert scale (stage 2, its operand 2 a `[1, experts]` weight).
///
/// No other pre or tail stage.
fn moe_route(
    cx: &StepCx<'_>,
    s: &TapeStep<'_>,
    shape: FusedShape,
    computes: &[TapeStep<'_>],
) -> Result<MoeRouteBf16, CanonicalRefusal> {
    let routing = || moe_refusal(s, MoeShapeRule::Routing);
    let FusedShape::Route {
        sort, pre, tail, ..
    } = shape
    else {
        return Err(routing());
    };
    let SubOp::RouteTopK { k } = s.node.op else {
        return Err(routing());
    };
    let at = |slot: SlotId| computes.iter().find(|c| c.writes == slot);
    let of = |slot: SlotId, want: fn(&SubOp) -> bool| at(slot).filter(|c| want(&c.node.op));
    // The step the logits are read through: the softmax (Qwen), else the argsort (Gemma).
    let (reader, program, renorm, expert_scale) = match (pre, tail) {
        (Some(pre), [None, renorm, None, None]) => {
            let softmax = of(pre, |op| matches!(op, SubOp::RouteSoftmax)).ok_or_else(routing)?;
            if renorm.is_some_and(|r| of(r, |op| matches!(op, SubOp::RouteRenorm)).is_none()) {
                return Err(routing());
            }
            (softmax, RouteProgram::QwenSoftmax, renorm.is_some(), None)
        }
        (None, [Some(scale), Some(softmax), Some(per_expert), None]) => {
            let scale = at(scale).ok_or_else(routing)?;
            let SubOp::RouteScale { scale: by } = scale.node.op else {
                return Err(routing());
            };
            of(softmax, |op| matches!(op, SubOp::RouteSoftmax)).ok_or_else(routing)?;
            let per_expert = of(per_expert, |op| {
                matches!(op, SubOp::RouteExpertScale { .. })
            })
            .ok_or_else(routing)?;
            finite_scalar(scale, by, true)?;
            let argsort = of(sort, |op| matches!(op, SubOp::RouteArgsort)).ok_or_else(routing)?;
            let program = RouteProgram::GemmaScaledSoftmax {
                scale_bits: by.to_bits(),
            };
            (argsort, program, false, Some(per_expert))
        }
        _ => return Err(routing()),
    };
    let logits = moe_producer(cx, reader, 0, computes, |op| {
        matches!(op, SubOp::RouterLogits { .. })
    })?;
    let SubOp::RouterLogits { experts, .. } = logits.node.op else {
        return Err(routing());
    };
    let moe = moe_tape(cx, s)?;
    moe_counts(cx, s, &moe, (experts.get(), k.get()))?;
    let expert_scale = match expert_scale {
        Some(per_expert) => {
            let [_, _, w] = per_expert.node.inputs.as_slice() else {
                return Err(operand(per_expert, 2, OperandRefusal::Missing));
            };
            same_shape(cx, per_expert, 2, w, 1, moe.experts)?;
            let input = per_expert.inputs.iter().nth(2);
            Some(weight_operand(cx, per_expert, 2, input)?)
        }
        None => None,
    };
    Ok(MoeRouteBf16 {
        logits: moe.logits.offset,
        topk_idx: moe.topk_idx.offset,
        scores: moe.scores.offset,
        m: cx.tokens.unwrap_or(0),
        experts: moe.experts,
        top_k: moe.top_k,
        renorm,
        program,
        expert_scale,
    })
}

/// `RouterNorm { eps, router }` as the bf16 RMSNorm of Gemma's router input into the MoE
/// scratch's router-norm rows: `x` (operand 0) `[m, hidden]` a computed slot or the embedded
/// hidden, its gain (operand 1) the router's `[1, hidden]` `router.scale`. Only Gemma's router has
/// a pre-norm.
fn router_norm(
    cx: &StepCx<'_>,
    s: &TapeStep<'_>,
    eps: f32,
    router: RouterBundle,
) -> Result<RmsNormBf16, CanonicalRefusal> {
    if router != RouterBundle::Gemma {
        return Err(moe_refusal(s, MoeShapeRule::Bundle));
    }
    let (x, g) = match s.node.inputs.as_slice() {
        [x, g] => (x, g),
        _ => return Err(operand(s, 1, OperandRefusal::Missing)),
    };
    let moe = moe_tape(cx, s)?;
    let region = moe
        .router_norm
        .ok_or(moe_refusal(s, MoeShapeRule::Geometry))?;
    let TensorShape { rows, cols } = cx.graph.shape(s.node.output.tensor);
    if cx.tokens != Some(rows) {
        return Err(moe_refusal(s, MoeShapeRule::Rows));
    }
    if cols != moe.hidden {
        return Err(moe_refusal(s, MoeShapeRule::Geometry));
    }
    row_shape(s, rows, cols, RMS_NORM_BF16_COLS)?;
    if !(eps.is_finite() && eps > 0.0) {
        return Err(CanonicalRefusal::NormEps { step: s.step, eps });
    }
    same_shape(cx, s, 0, x, rows, cols)?;
    same_shape(cx, s, 1, g, 1, cols)?;
    whole_tensors(cx, s, &[(0, x), (1, g)])?;
    let at = arena(cx)?;
    let mut inputs = s.inputs.iter();
    let xc = activation(cx, s, 0, inputs.next())?;
    let gain = weight_operand(cx, s, 1, inputs.next())?;
    Ok(RmsNormBf16 {
        x: at(xc),
        gain,
        y: region.offset,
        rows,
        cols,
        eps_bits: eps.to_bits(),
        convention: GainConvention::Scale,
    })
}

/// The bundle and geometry checks every expert op shares: its bundle is one
/// `op_abi::CUDA_GROUPED_EXPERTS` names, and its experts and top-k the scratch's.
fn expert_bundle(
    cx: &StepCx<'_>,
    s: &TapeStep<'_>,
    bundle: ExpertBundle,
    (experts, top_k): (u32, u32),
) -> Result<MoeTape, CanonicalRefusal> {
    if !CUDA_GROUPED_EXPERTS.contains(&bundle) {
        return Err(moe_refusal(s, MoeShapeRule::Bundle));
    }
    let moe = moe_tape(cx, s)?;
    moe_counts(cx, s, &moe, (experts, top_k))?;
    Ok(moe)
}

/// `ExpertSort { experts, k, bundle }`: the pairs of `x` (operand 0, `[m, hidden]`, a computed
/// slot or the embedded hidden) sorted by the experts the routing (operand 1, the `RouteTopK`)
/// picked. A sorted bucket's stable counting sort; a gathered bucket's no launch, its expert GEMVs
/// reading each pair's token row of `x` directly.
fn expert_sort(
    cx: &StepCx<'_>,
    s: &TapeStep<'_>,
    (experts, top_k): (u32, u32),
    bundle: ExpertBundle,
    computes: &[TapeStep<'_>],
) -> Result<Option<ExpertSortBf16>, CanonicalRefusal> {
    let moe = expert_bundle(cx, s, bundle, (experts, top_k))?;
    let [x, _] = s.node.inputs.as_slice() else {
        return Err(operand(s, 1, OperandRefusal::Missing));
    };
    moe_producer(cx, s, 1, computes, |op| {
        matches!(op, SubOp::RouteTopK { .. })
    })?;
    let m = cx.tokens.unwrap_or(0);
    same_shape(cx, s, 0, x, m, moe.hidden)?;
    whole_tensors(cx, s, &[(0, x)])?;
    let xc = activation(cx, s, 0, s.inputs.iter().next())?;
    let Some(sort) = moe.sort else {
        return Ok(None);
    };
    let at = arena(cx)?;
    Ok(Some(ExpertSortBf16 {
        x: at(xc),
        topk_idx: moe.topk_idx.offset,
        chunk_hist: sort.chunk_hist.offset,
        counts: sort.counts.offset,
        offsets: sort.offsets.offset,
        tile_expert: sort.tile_expert.offset,
        pos: sort.pos.offset,
        rows: sort.rows.offset,
        m,
        experts,
        top_k,
        hidden: moe.hidden,
        max_tiles: moe.max_tiles,
    }))
}

/// `ExpertMatmul { proj, n, k, weight, bundle }` over a dense bundle: every pair's `proj`
/// projection by its expert's `[n, k]` slab of the bundle (operand 2). The gate and up read the
/// sort's rows (operand 0, the `ExpertSort`), the down the activation's (operand 0, the
/// `ExpertGatedAct`); operand 1 is the sort, whose picks name each pair's expert. A sorted
/// bucket's grouped GEMM over the expert-padded tiles; a gathered bucket's expert GEMV, the gate
/// and up reading each pair's token row of the sort's `x`.
fn expert_matmul(
    cx: &StepCx<'_>,
    s: &TapeStep<'_>,
    (proj, n, top_k): (ExpertProj, u32, u32),
    (weight, bundle): (ExpertWeight, ExpertBundle),
    computes: &[TapeStep<'_>],
) -> Result<CudaStep, CanonicalRefusal> {
    if let ExpertWeight::Affine(_) = weight {
        return Err(moe_refusal(s, MoeShapeRule::Affine));
    }
    let sort_op = |op: &SubOp| matches!(op, SubOp::ExpertSort { .. });
    let sort = moe_producer(cx, s, 1, computes, sort_op)?;
    let SubOp::ExpertSort { experts, .. } = sort.node.op else {
        return Err(moe_refusal(s, MoeShapeRule::Routing));
    };
    let moe = expert_bundle(cx, s, bundle, (experts.get(), top_k))?;
    // Gemma's SwitchGLU bank is its own expert bank, with no router in it.
    let layout = match bundle {
        ExpertBundle::SwitchGlu => MoeWeight::Own,
        _ => MoeWeight::Bundle,
    };
    let (projection, reads) = match proj {
        ExpertProj::Gate => (ExpertProjection::Gate, PairRows::Token),
        ExpertProj::Up => (ExpertProjection::Up, PairRows::Token),
        ExpertProj::Down => (ExpertProjection::Down, PairRows::Pair),
    };
    let (want_n, k) = projection.n_k(moe.hidden, moe.inter);
    if n != want_n {
        return Err(moe_refusal(s, MoeShapeRule::Geometry));
    }
    let rows = match proj {
        ExpertProj::Gate | ExpertProj::Up => moe_producer(cx, s, 0, computes, sort_op)?,
        ExpertProj::Down => moe_producer(cx, s, 0, computes, |op| {
            matches!(op, SubOp::ExpertGatedAct { .. })
        })?,
    };
    let [_, _, w] = s.node.inputs.as_slice() else {
        return Err(operand(s, 3, OperandRefusal::Extra));
    };
    whole_tensors(cx, s, &[(2, w)])?;
    let bundle = weight_operand(cx, s, 2, s.inputs.iter().nth(2))?;
    let (_, out) = moe.projection_regions(projection, moe.sort.map(|t| t.rows));
    let Some(sorted) = moe.sort else {
        if !k.is_multiple_of(GEMV_BF16_MULTIPLE) {
            return Err(moe_refusal(s, MoeShapeRule::OffTile { dim: "k", value: k }));
        }
        // Gathered: the gate and up read each pair's token row, the sort's `x`; the down its own
        // row of the activation.
        let a = match reads {
            PairRows::Token => {
                let at = arena(cx)?;
                at(activation(cx, rows, 0, rows.inputs.iter().next())?)
            }
            PairRows::Pair => moe.act.offset,
        };
        return Ok(CudaStep::ExpertGemv(ExpertGemvBf16 {
            a,
            rows: reads,
            bundle,
            weight: layout,
            proj: projection,
            topk_idx: moe.topk_idx.offset,
            c: out.1.offset,
            m: cx.tokens.unwrap_or(0),
            top_k,
            experts: moe.experts,
            hidden: moe.hidden,
            inter: moe.inter,
        }));
    };
    let (_, tn, tk) = GEMM_BF16_TILE;
    // A 64-column-tile model's expert GEMM takes 64-column tiles when `n` is off the 256 one.
    let tn = match cx.facts.gemm_col_tile {
        GemmColTile::Col64 if !n.is_multiple_of(tn) => GEMM_COL64_TILE.1,
        _ => tn,
    };
    for (dim, value, tile) in [("n", n, tn), ("k", k, tk)] {
        if !value.is_multiple_of(tile) {
            return Err(moe_refusal(s, MoeShapeRule::OffTile { dim, value }));
        }
    }
    let a = match reads {
        PairRows::Token => sorted.rows.offset,
        PairRows::Pair => moe.act.offset,
    };
    Ok(CudaStep::ExpertGemm(ExpertGemmBf16 {
        a,
        bundle,
        weight: layout,
        proj: projection,
        offsets: sorted.offsets.offset,
        tile_expert: sorted.tile_expert.offset,
        c: out.1.offset,
        max_tiles: moe.max_tiles,
        experts: moe.experts,
        hidden: moe.hidden,
        inter: moe.inter,
        n_tile: tn,
    }))
}

/// `ExpertGatedAct { act }` as the bf16 `SiluMul` (Qwen's SiLU) or `GeluMul` (Gemma's GELU) over
/// the MoE scratch's rows: `act(gate) · up` from the gate and up projections (operands 0 and 1)
/// into the activation rows, every row of the regions (a sorted bucket's padding rows hold `+0.0`,
/// its rows past the live tiles stale data no step reads). Any other activation is refused.
fn expert_gated_act(
    cx: &StepCx<'_>,
    s: &TapeStep<'_>,
    act: GatedAct,
    computes: &[TapeStep<'_>],
) -> Result<CudaStep, CanonicalRefusal> {
    let kernel: fn(SiluMulBf16) -> CudaStep = match act {
        GatedAct::Silu => CudaStep::SiluMul,
        GatedAct::Gelu => CudaStep::GeluMul,
        _ => return Err(moe_refusal(s, MoeShapeRule::Activation)),
    };
    let proj_is = |p: ExpertProj| move |op: &SubOp| matches!(op, SubOp::ExpertMatmul { proj, .. } if *proj == p);
    moe_producer(cx, s, 0, computes, proj_is(ExpertProj::Gate))?;
    moe_producer(cx, s, 1, computes, proj_is(ExpertProj::Up))?;
    let moe = moe_tape(cx, s)?;
    let rows = moe.expert_rows(cx.tokens.unwrap_or(0));
    row_shape(s, rows, moe.inter, SILU_MUL_BF16_COLS)?;
    Ok(kernel(SiluMulBf16 {
        gate: moe.gate.offset,
        up: moe.up.offset,
        out: moe.act.offset,
        rows,
        cols: moe.inter,
    }))
}

/// `ExpertUnsort`: no launch. The combine reads each pair's down row through the sort's `pos`
/// (sorted) or at the pair's own row (gathered), so the rows are never put back in token order;
/// metal elides it the same way. Its operands are checked: the down projection (operand 0) and
/// the sort (operand 1).
fn expert_unsort(
    cx: &StepCx<'_>,
    s: &TapeStep<'_>,
    computes: &[TapeStep<'_>],
) -> Result<(), CanonicalRefusal> {
    moe_producer(cx, s, 0, computes, |op| {
        matches!(
            op,
            SubOp::ExpertMatmul {
                proj: ExpertProj::Down,
                ..
            }
        )
    })?;
    moe_producer(cx, s, 1, computes, |op| {
        matches!(op, SubOp::ExpertSort { .. })
    })?;
    moe_tape(cx, s).map(|_| ())
}

/// `ExpertCombine { hidden, shared }`: each token's `top_k` down rows (operand 0, the
/// `ExpertUnsort`) summed by its routing weights (operand 1, the gathered scores or their renorm)
/// into this step's `[m, hidden]` colour. It never adds a shared expert: one the bundle has must
/// be the DSL's own ops (the single-owner rule, metal's `shared_expert_owned`), or it is refused.
fn expert_combine(
    cx: &StepCx<'_>,
    s: &TapeStep<'_>,
    hidden: u32,
    shared: bool,
    computes: &[TapeStep<'_>],
) -> Result<ExpertCombineBf16, CanonicalRefusal> {
    if shared && cx.facts.shared_expert != SharedExpertOwner::Dsl {
        return Err(CanonicalRefusal::UndeclaredSharedExpert { step: s.step });
    }
    moe_producer(cx, s, 0, computes, |op| matches!(op, SubOp::ExpertUnsort))?;
    moe_producer(cx, s, 1, computes, |op| {
        matches!(
            op,
            SubOp::RouteGatherScores | SubOp::RouteRenorm | SubOp::RouteExpertScale { .. }
        )
    })?;
    let moe = moe_tape(cx, s)?;
    let m = cx.tokens.unwrap_or(0);
    let TensorShape { rows, cols } = cx.graph.shape(s.node.output.tensor);
    if (rows, cols) != (m, hidden) || hidden != moe.hidden {
        return Err(moe_refusal(s, MoeShapeRule::Geometry));
    }
    row_shape(s, rows, cols, ADD_BF16_COLS)?;
    let at = arena(cx)?;
    Ok(ExpertCombineBf16 {
        down: moe.down.offset,
        pos: moe.sort.map(|t| t.pos.offset),
        scores: moe.scores.offset,
        out: at(colour_of(cx, s.writes)),
        m,
        top_k: moe.top_k,
        hidden,
        down_rows: moe.expert_rows(m),
    })
}

/// `GateScale` as the bf16 `routed + shared · sigmoid(g)`: `routed` and `shared` (operands 0 and
/// 1) `[rows, cols]`, `g` (operand 2) `[rows, 1]`, each a computed slot or the embedded hidden.
fn gate_scale(cx: &StepCx<'_>, s: &TapeStep<'_>) -> Result<GateScaleBf16, CanonicalRefusal> {
    let (r, sh, g) = match s.node.inputs.as_slice() {
        [r, sh, g] => (r, sh, g),
        _ => return Err(operand(s, 2, OperandRefusal::Missing)),
    };
    let TensorShape { rows, cols } = cx.graph.shape(s.node.output.tensor);
    row_shape(s, rows, cols, GATE_SCALE_BF16_COLS)?;
    same_shape(cx, s, 0, r, rows, cols)?;
    same_shape(cx, s, 1, sh, rows, cols)?;
    same_shape(cx, s, 2, g, rows, 1)?;
    whole_tensors(cx, s, &[(0, r), (1, sh), (2, g)])?;
    let at = arena(cx)?;
    let mut inputs = s.inputs.iter();
    let rc = activation(cx, s, 0, inputs.next())?;
    let sc = activation(cx, s, 1, inputs.next())?;
    let gc = activation(cx, s, 2, inputs.next())?;
    Ok(GateScaleBf16 {
        routed: at(rc),
        shared: at(sc),
        g: at(gc),
        out: at(colour_of(cx, s.writes)),
        rows,
        cols,
    })
}

/// The bf16 rope's shape rule, the one its launcher applies, and the table's: `rows`, `cols`,
/// `head_dim` and `rot_dim` non-zero; `rot_dim` a multiple of [`ROPE_BF16_ROT_MULTIPLE`] and at
/// most `head_dim`; `head_dim` a multiple of [`ROPE_BF16_HEAD_MULTIPLE`]; `cols` a whole number
/// of heads; `rows · cols` at most `i32::MAX`; and `rows` the bucket's `tokens`, when known.
fn rope_shape(
    rows: u32,
    cols: u32,
    head_dim: u32,
    rot_dim: u32,
    tokens: Option<u32>,
) -> Result<(), RopeShapeRule> {
    for (dim, value) in [
        ("rows", rows),
        ("cols", cols),
        ("head_dim", head_dim),
        ("rot_dim", rot_dim),
    ] {
        if value == 0 {
            return Err(RopeShapeRule::Empty { dim });
        }
    }
    if !rot_dim.is_multiple_of(ROPE_BF16_ROT_MULTIPLE) {
        return Err(RopeShapeRule::RotOffVector {
            multiple: ROPE_BF16_ROT_MULTIPLE,
        });
    }
    if rot_dim > head_dim {
        return Err(RopeShapeRule::RotAboveHead);
    }
    if !head_dim.is_multiple_of(ROPE_BF16_HEAD_MULTIPLE) {
        return Err(RopeShapeRule::HeadOffVector {
            multiple: ROPE_BF16_HEAD_MULTIPLE,
        });
    }
    if !cols.is_multiple_of(head_dim) {
        return Err(RopeShapeRule::ColsOffHead);
    }
    let elements = u64::from(rows) * u64::from(cols);
    if elements > i32::MAX as u64 {
        return Err(RopeShapeRule::TooManyElements { elements });
    }
    if let Some(tokens) = tokens
        && rows != tokens
    {
        return Err(RopeShapeRule::RowsNotTokens { tokens });
    }
    Ok(())
}

/// The bf16 gate split's shape rule, the one its launcher applies: `rows`, `half_cols` and
/// `head_dim` non-zero; `head_dim` a multiple of [`GATE_SPLIT_BF16_HEAD_MULTIPLE`]; `half_cols` a
/// whole number of heads; `rows · 2·half_cols` at most `i32::MAX`; and, last, `head_dim`
/// [`GATE_SPLIT_BF16_HEAD_DIM`], the one the kernel is instantiated for.
fn gate_shape(rows: u32, half_cols: u32, head_dim: u32) -> Result<(), GateShapeRule> {
    for (dim, value) in [
        ("rows", rows),
        ("half_cols", half_cols),
        ("head_dim", head_dim),
    ] {
        if value == 0 {
            return Err(GateShapeRule::Empty { dim });
        }
    }
    if !head_dim.is_multiple_of(GATE_SPLIT_BF16_HEAD_MULTIPLE) {
        return Err(GateShapeRule::HeadOffVector {
            multiple: GATE_SPLIT_BF16_HEAD_MULTIPLE,
        });
    }
    if !half_cols.is_multiple_of(head_dim) {
        return Err(GateShapeRule::HalfOffHead);
    }
    let elements = u64::from(rows) * 2 * u64::from(half_cols);
    if elements > i32::MAX as u64 {
        return Err(GateShapeRule::TooManyElements { elements });
    }
    if head_dim != GATE_SPLIT_BF16_HEAD_DIM {
        return Err(GateShapeRule::HeadNotInstantiated {
            want: GATE_SPLIT_BF16_HEAD_DIM,
        });
    }
    Ok(())
}

/// Which bf16 GEMM takes `(m, n, k)`, as the step constructor: an `n` of at most
/// [`GEMM_NARROW_BF16_MAX_N`] that is not a multiple of the tile's goes to the narrow GEMM under
/// its rule ([`gemm_narrow_shape`]) at every `m`, one row included, since its bits for a row do
/// not depend on where the row sits; then, for a [`GemmColTile::Col64`] model, an `m` of 2 or
/// more and an `n` that is a multiple of 64 and not of 256 goes to the 64-column GEMM under its
/// rule ([`gemm_col64_shape`]); otherwise `m = 1` goes to the GEMV under its rule
/// ([`gemv_shape`]), and any other `m` to the tiled GEMM under its rule ([`gemm_shape`]).
fn gemm_kernel(
    m: u32,
    n: u32,
    k: u32,
    col_tile: GemmColTile,
) -> Result<fn(GemmBf16) -> CudaStep, GemmShapeRule> {
    if n != 0 && n <= GEMM_NARROW_BF16_MAX_N && !n.is_multiple_of(GEMM_BF16_TILE.1) {
        gemm_narrow_shape(m, n, k)?;
        return Ok(CudaStep::GemmNarrow);
    }
    if col_tile == GemmColTile::Col64
        && m >= 2
        && n > GEMM_NARROW_BF16_MAX_N
        && n.is_multiple_of(GEMM_COL64_TILE.1)
        && !n.is_multiple_of(GEMM_BF16_TILE.1)
    {
        gemm_col64_shape(m, n, k)?;
        return Ok(CudaStep::GemmCol64);
    }
    if m == 1 {
        gemv_shape(n, k)?;
        return Ok(CudaStep::Gemv);
    }
    gemm_shape(m, n, k)?;
    Ok(CudaStep::Gemm)
}

/// The bf16 GEMV's shape rule (`m = 1`, which the caller chose it by): `n` non-zero, `k` a
/// non-zero multiple of [`GEMV_BF16_MULTIPLE`], and both at most `i32::MAX`.
fn gemv_shape(n: u32, k: u32) -> Result<(), GemmShapeRule> {
    if n == 0 {
        return Err(GemmShapeRule::OffTile {
            dim: "n",
            value: n,
            tile: 1,
        });
    }
    if k == 0 || !k.is_multiple_of(GEMV_BF16_MULTIPLE) {
        return Err(GemmShapeRule::GemvOffVector {
            value: k,
            multiple: GEMV_BF16_MULTIPLE,
        });
    }
    for (dim, value) in [("n", n), ("k", k)] {
        if i32::try_from(value).is_err() {
            return Err(GemmShapeRule::AboveIntMax { dim, value });
        }
    }
    Ok(())
}

/// The narrow bf16 GEMM's shape rule: `m` non-zero, `k` a non-zero multiple of
/// [`GEMM_NARROW_BF16_MULTIPLE`], `n` [`GEMM_NARROW_BF16_ONE`] or a non-zero multiple of it (`n`
/// at most [`GEMM_NARROW_BF16_MAX_N`], which the caller chose it by), and `m` and `k` at most
/// `i32::MAX`.
fn gemm_narrow_shape(m: u32, n: u32, k: u32) -> Result<(), GemmShapeRule> {
    if m == 0 {
        return Err(GemmShapeRule::OffTile {
            dim: "m",
            value: m,
            tile: 1,
        });
    }
    for (dim, value) in [("n", n), ("k", k)] {
        if dim == "n" && value == GEMM_NARROW_BF16_ONE {
            continue;
        }
        if value == 0 || !value.is_multiple_of(GEMM_NARROW_BF16_MULTIPLE) {
            return Err(GemmShapeRule::NarrowOffVector {
                dim,
                value,
                multiple: GEMM_NARROW_BF16_MULTIPLE,
            });
        }
    }
    for (dim, value) in [("m", m), ("k", k)] {
        if i32::try_from(value).is_err() {
            return Err(GemmShapeRule::AboveIntMax { dim, value });
        }
    }
    Ok(())
}

/// The bf16 GEMM's shape rule: `m`, `n` and `k` non-zero multiples of [`GEMM_BF16_TILE`], and
/// each at most `i32::MAX` (ThunderKittens' `gl` stores dimensions as `int`).
fn gemm_shape(m: u32, n: u32, k: u32) -> Result<(), GemmShapeRule> {
    let (tm, tn, tk) = GEMM_BF16_TILE;
    for (dim, value, tile) in [("m", m, tm), ("n", n, tn), ("k", k, tk)] {
        if value == 0 || !value.is_multiple_of(tile) {
            return Err(GemmShapeRule::OffTile { dim, value, tile });
        }
    }
    for (dim, value) in [("m", m), ("n", n), ("k", k)] {
        if i32::try_from(value).is_err() {
            return Err(GemmShapeRule::AboveIntMax { dim, value });
        }
    }
    Ok(())
}

/// The 64-column bf16 GEMM's shape rule: `m`, `n` and `k` non-zero multiples of
/// [`GEMM_COL64_TILE`], and each at most `i32::MAX`.
fn gemm_col64_shape(m: u32, n: u32, k: u32) -> Result<(), GemmShapeRule> {
    let (tm, tn, tk) = GEMM_COL64_TILE;
    for (dim, value, tile) in [("m", m, tm), ("n", n, tn), ("k", k, tk)] {
        if value == 0 || !value.is_multiple_of(tile) {
            return Err(GemmShapeRule::Col64OffTile { dim, value, tile });
        }
    }
    for (dim, value) in [("m", m), ("n", n), ("k", k)] {
        if i32::try_from(value).is_err() {
            return Err(GemmShapeRule::AboveIntMax { dim, value });
        }
    }
    Ok(())
}

/// The row kernels' shape rule, the one their launchers apply: `rows` and `cols` non-zero, `cols`
/// a multiple of the kernel's vector width `multiple`, and `rows · cols` at most `i32::MAX`.
fn row_shape(
    s: &TapeStep<'_>,
    rows: u32,
    cols: u32,
    multiple: u32,
) -> Result<(), CanonicalRefusal> {
    let refuse = |rule| CanonicalRefusal::RowShape {
        step: s.step,
        op: s.node.op.name(),
        rows,
        cols,
        rule,
    };
    for (dim, value) in [("rows", rows), ("cols", cols)] {
        if value == 0 {
            return Err(refuse(RowShapeRule::Empty { dim }));
        }
    }
    if !cols.is_multiple_of(multiple) {
        return Err(refuse(RowShapeRule::OffVector { cols, multiple }));
    }
    let elements = u64::from(rows) * u64::from(cols);
    if elements > i32::MAX as u64 {
        return Err(refuse(RowShapeRule::TooManyElements { elements }));
    }
    Ok(())
}

/// Refuse operand `k` of a row kernel unless it is `[rows, cols]`: these kernels broadcast
/// nothing.
fn same_shape(
    cx: &StepCx<'_>,
    s: &TapeStep<'_>,
    k: u8,
    r: &TensorRegion,
    rows: u32,
    cols: u32,
) -> Result<(), CanonicalRefusal> {
    let got = cx.graph.shape(r.tensor);
    if (got.rows, got.cols) == (rows, cols) {
        return Ok(());
    }
    Err(CanonicalRefusal::OperandShape {
        step: s.step,
        op: s.node.op.name(),
        operand: k,
        rows: got.rows,
        cols: got.cols,
        want_rows: rows,
        want_cols: cols,
    })
}

/// Refuse a step unless each operand `(k, region)` of `operands` and its output cover their whole
/// tensors: a kernel binds whole buffers.
fn whole_tensors(
    cx: &StepCx<'_>,
    s: &TapeStep<'_>,
    operands: &[(u8, &TensorRegion)],
) -> Result<(), CanonicalRefusal> {
    for &(k, region) in operands {
        if !whole(cx.graph, region) {
            return Err(operand(s, k, OperandRefusal::PartialRead));
        }
    }
    if !whole(cx.graph, &s.node.output) {
        return Err(CanonicalRefusal::PartialWrite {
            step: s.step,
            op: s.node.op.name(),
        });
    }
    Ok(())
}

/// Each colour's byte offset, or the arena's refusal.
fn arena<'a>(cx: &StepCx<'a>) -> Result<impl Fn(Colour) -> ArenaOffset + 'a, CanonicalRefusal> {
    let offsets = cx.offsets.map_err(CanonicalRefusal::Arena)?;
    Ok(move |c: Colour| ArenaOffset(offsets[c.index() as usize]))
}

/// An activation operand: operand `k` of `s`, a computed slot (its colour) or the embedded hidden
/// (colour 0).
fn activation(
    cx: &StepCx<'_>,
    s: &TapeStep<'_>,
    k: u8,
    input: Option<&ComputeInput>,
) -> Result<Colour, CanonicalRefusal> {
    match input {
        Some(ComputeInput::Computed(slots)) => match slots.as_slice() {
            [slot] => Ok(colour_of(cx, *slot)),
            _ => Err(operand(s, k, OperandRefusal::SplitWriters(slots.len()))),
        },
        Some(ComputeInput::External {
            tensor, per_layer, ..
        }) => {
            if !per_layer.is_empty() {
                return Err(operand(s, k, OperandRefusal::PerLayer));
            }
            match &cx.bindings[tensor.index()] {
                SourceBinding::EmbeddedHidden => Ok(Colour::ZERO),
                binding => Err(operand(
                    s,
                    k,
                    OperandRefusal::NotHiddenOrSlot {
                        source: tensor.index(),
                        binding: binding.clone(),
                    },
                )),
            }
        }
        None => Err(operand(s, k, OperandRefusal::Missing)),
    }
}

/// A weight operand: operand `k` of `s`, a source bound to a model weight or to one tensor of a
/// weight bundle (its index): a bundle tensor is one tensor, so one CUDA weight, as a weight is.
fn weight_operand(
    cx: &StepCx<'_>,
    s: &TapeStep<'_>,
    k: u8,
    input: Option<&ComputeInput>,
) -> Result<WeightIx, CanonicalRefusal> {
    match input {
        Some(ComputeInput::External {
            tensor, per_layer, ..
        }) => {
            if !per_layer.is_empty() {
                return Err(operand(s, k, OperandRefusal::PerLayer));
            }
            match &cx.bindings[tensor.index()] {
                SourceBinding::Weight { .. } | SourceBinding::BundleTensor { .. } => {
                    Ok(WeightIx(tensor.index() as u32))
                }
                binding => Err(operand(
                    s,
                    k,
                    OperandRefusal::NotAWeight {
                        source: Some((tensor.index(), binding.clone())),
                    },
                )),
            }
        }
        Some(ComputeInput::Computed(_)) => {
            Err(operand(s, k, OperandRefusal::NotAWeight { source: None }))
        }
        None => Err(operand(s, k, OperandRefusal::Missing)),
    }
}

/// The colour of a slot the tape writes.
fn colour_of(cx: &StepCx<'_>, slot: SlotId) -> Colour {
    cx.colouring
        .colour_of(slot)
        .expect("the colourer colours every arena step, and every CUDA step is on the arena")
}

/// Whether `r` covers its whole tensor.
fn whole(graph: &SubtileIR, r: &TensorRegion) -> bool {
    r.region == graph.shape(r.tensor).whole()
}

fn operand(s: &TapeStep<'_>, operand: u8, why: OperandRefusal) -> CanonicalRefusal {
    CanonicalRefusal::Operand {
        step: s.step,
        op: s.node.op.name(),
        operand,
        why,
    }
}
