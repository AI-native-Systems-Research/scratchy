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

use scratchy_subtile::handoff::{SourceBinding, UnrollIndex};
use scratchy_subtile::lower::GemmWeight;
use scratchy_subtile::model_geometry::ModelAttnGeometry;
use scratchy_subtile::subtile_ir::{
    AttnMask, EwKind, GainConvention, RopeFormTag, SubOp, SubtileIR, SubtileNode,
};
use scratchy_subtile::subtile_ir::{TensorRegion, TensorShape};
use scratchy_subtile::subtile_tape::{ComputeInput, ComputeInputs, SlotId};
use scratchy_subtile::tape_colouring::{Colour, TapeColouring};
use scratchy_subtile::tape_folding::{FusedShape, Fusion};
use scratchy_target_cuda::op_abi::CudaFusion;
use scratchy_target_cuda::step::{
    ADD_BF16_COLS, ATTN_DECODE_BF16_HEAD_DIM, ATTN_DECODE_BF16_MAX_GROUP,
    ATTN_PREFILL_BF16_HEAD_DIM, ATTN_PREFILL_BF16_ROWS, AddBf16, AttnPrefillBf16, CudaStep,
    EMBED_BF16_COLS, EmbedBf16, GATE_APPLY_BF16_COLS, GATE_SPLIT_BF16_HEAD_DIM,
    GATE_SPLIT_BF16_HEAD_MULTIPLE, GDN_CONV_KERNEL_RANGE, GDN_HEAD_K, GDN_HEAD_V_MAX,
    GDN_HEAD_V_MULTIPLE, GEMM_BF16_TILE, GEMM_NARROW_BF16_MAX_N, GEMM_NARROW_BF16_MULTIPLE,
    GEMV_BF16_MULTIPLE, GateApplyBf16, GateSplitBf16, GatedDeltaNetBf16, GemmBf16,
    RMS_NORM_BF16_COLS, ROPE_BF16_HEAD_MULTIPLE, ROPE_BF16_ROT_MULTIPLE, RmsNormBf16,
    RopeAppendBf16, RopeBf16, SILU_MUL_BF16_COLS, SiluMulBf16,
};
use scratchy_target_cuda::tape::{ArenaOffset, WeightIx};

use crate::canonical::{
    ArenaRefusal, AttnShapeRule, CanonicalRefusal, CudaModelFacts, EmbedShapeRule, GateShapeRule,
    GdnRegions, GdnShapeRule, GdnWidths, GemmShapeRule, KvCacheRefusal, KvPool, KvRegions,
    OperandRefusal, ProducerRefusal, RopeShapeRule, RowShapeRule, ViewRefusal,
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
        // SwiGLU runs only as the `SiluMul` fold: a `Silu` the fold left, a `Mul` driving none,
        // and the pre-fused op (which only spyre's `fuse_silu_mul` makes) have no kernel.
        L::Elementwise(E::Silu | E::Mul) | L::SiluMul => Err(CanonicalRefusal::Unfused {
            step: s.step,
            op: op.name(),
        }),
        L::SumReduce { .. }
        | L::Elementwise(E::Gelu | E::QuickGelu | E::GeluErf | E::Relu2 | E::Sub | E::BiasAdd)
        | L::ScalarMul { .. }
        | L::RmsNormReduce { .. }
        | L::RmsNormApply { .. }
        | L::KvAppend { .. }
        | L::TanhSoftCap { .. }
        | L::RmsNormUnit { .. }
        | L::ScalarWeightMul
        | L::Concat { .. }
        | L::GateScale
        | L::LoadRows { .. }
        | L::EmbeddingGather { .. }
        | L::VisionRope
        | L::VarlenAttention { .. }
        | L::EncoderAttn { .. }
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
        CudaFusion::SiluMul => silu_mul(cx, s, fusion.shape, computes).map(CudaStep::SiluMul),
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
    let kernel = gemm_kernel(m, n, k).map_err(|rule| CanonicalRefusal::GemmShape {
        step,
        m,
        n,
        k,
        rule,
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

/// The `SiluMul` fold of `mul(silu(gate), up)` as one bf16 launch: `gate` is the absorbed
/// `Silu`'s operand 0, `up` the driving `Mul`'s operand 1, `out` the `Mul`'s colour. The `Silu`'s
/// own colour is never written.
fn silu_mul(
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
/// embedded hidden; operands 1 and 2 the global rotary `cos` and `sin`, which CUDA does not upload
/// as sources but reads as the tape's one rotary table. The rotated width is the model's
/// (`CudaModelFacts::rot_dim`); the op carries only `head_dim`.
fn rope(cx: &StepCx<'_>, s: &TapeStep<'_>, head_dim: u32) -> Result<RopeBf16, CanonicalRefusal> {
    let (step, node) = (s.step, s.node);
    let rot_dim = cx.facts.rot_dim;
    let [x, ..] = node.inputs.as_slice() else {
        return Err(operand(s, 0, OperandRefusal::Missing));
    };
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
    let table = rotary_table(cx, s)?;
    let at = arena(cx)?;
    Ok(RopeBf16 {
        x: at(xc),
        table,
        out: at(colour_of(cx, s.writes)),
        rows,
        cols,
        head_dim,
        rot_dim,
    })
}

/// Operands 1 and 2 of a rope (`RopeRotate` or `RopeAppend`): the global rotary `cos` and
/// `sin`, which CUDA does not upload as sources but reads as the tape's one rotary table. Its
/// offset.
fn rotary_table(cx: &StepCx<'_>, s: &TapeStep<'_>) -> Result<ArenaOffset, CanonicalRefusal> {
    let (step, op) = (s.step, s.node.op.name());
    let mut inputs = s.inputs.iter().skip(1);
    for (k, want) in [
        (1, SourceBinding::Cos { local: false }),
        (2, SourceBinding::Sin { local: false }),
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
    // Operand 1 is the global `cos`, so the arena placed the table, unless the arena has no
    // layout at all (refused below, by the caller's `arena`).
    match cx.offsets {
        Ok(_) => cx.rotary.ok_or(CanonicalRefusal::RotaryOperand {
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
/// the tape's rotary table (operands 1 and 2, as a `RopeRotate`'s) into this step's colour, then
/// the rotated K and V (operand 3, a computed slot) written into the KV cache of `layer`, whose
/// `PrefixK` and `PrefixV` are operands 4 and 5. The rope's shape rule is K's `[rows, cols]`.
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
    if attn.window().is_some() {
        return Err(CanonicalRefusal::SlidingWindow { step, op });
    }
    if head_dim != cx.facts.head_dim {
        return Err(CanonicalRefusal::KvHeadDim {
            step,
            op,
            head_dim,
            want: cx.facts.head_dim,
        });
    }
    let rot_dim = cx.facts.rot_dim;
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
    let table = rotary_table(cx, s)?;
    prefix_operands(s, cx, 4, u64::from(layer))?;
    let (pool, regions, index) = kv_pool(cx, layer)?;
    if cols != pool.kv_heads * pool.head_dim {
        return Err(CanonicalRefusal::KvGeometry {
            step,
            op,
            kv_heads: pool.kv_heads,
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
    })
}

/// `AttnDecode` as attention over the paged KV cache: the bf16 decode attention for a one-row
/// bucket, the bf16 causal prefill attention for a bucket of a multiple of
/// [`ATTN_PREFILL_BF16_ROWS`] rows, chosen here at expansion; any other row count is refused, and
/// so, last, is a one-row bucket whose GQA group is above [`ATTN_DECODE_BF16_MAX_GROUP`].
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
    let (kernel, want_head_dim): (fn(AttnPrefillBf16) -> CudaStep, u32) = if rows == 1 {
        (CudaStep::AttnDecode, ATTN_DECODE_BF16_HEAD_DIM)
    } else {
        (CudaStep::AttnPrefill, ATTN_PREFILL_BF16_HEAD_DIM)
    };
    if mask.window().is_some() {
        return Err(CanonicalRefusal::SlidingWindow { step, op });
    }
    let q = match node.inputs.as_slice() {
        [q, _, _, _, _] => q,
        [_, _, _, _, _, _] => return Err(CanonicalRefusal::AttnSinks { step }),
        _ => return Err(operand(s, 4, OperandRefusal::Missing)),
    };
    if head_dim != want_head_dim {
        return Err(CanonicalRefusal::AttnShape {
            step,
            rows,
            head_dim,
            rule: AttnShapeRule::HeadDim {
                want: want_head_dim,
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
    if head_dim != cx.facts.head_dim {
        return Err(CanonicalRefusal::KvHeadDim {
            step,
            op,
            head_dim,
            want: cx.facts.head_dim,
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
    if kv_heads != pool.kv_heads {
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
/// not depend on where the row sits; otherwise `m = 1` goes to the GEMV under its rule
/// ([`gemv_shape`]), and any other `m` to the tiled GEMM under its rule ([`gemm_shape`]).
fn gemm_kernel(m: u32, n: u32, k: u32) -> Result<fn(GemmBf16) -> CudaStep, GemmShapeRule> {
    if n != 0 && n <= GEMM_NARROW_BF16_MAX_N && !n.is_multiple_of(GEMM_BF16_TILE.1) {
        gemm_narrow_shape(m, n, k)?;
        return Ok(CudaStep::GemmNarrow);
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

/// The narrow bf16 GEMM's shape rule: `m` non-zero, `n` and `k` non-zero multiples of
/// [`GEMM_NARROW_BF16_MULTIPLE`] (`n` at most [`GEMM_NARROW_BF16_MAX_N`], which the caller
/// chose it by), and `m` and `k` at most `i32::MAX`.
fn gemm_narrow_shape(m: u32, n: u32, k: u32) -> Result<(), GemmShapeRule> {
    if m == 0 {
        return Err(GemmShapeRule::OffTile {
            dim: "m",
            value: m,
            tile: 1,
        });
    }
    for (dim, value) in [("n", n), ("k", k)] {
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

/// A weight operand: operand `k` of `s`, a source bound to a model weight (its index).
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
                SourceBinding::Weight { .. } => Ok(WeightIx(tensor.index() as u32)),
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
