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

use scratchy_subtile::handoff::SourceBinding;
use scratchy_subtile::lower::GemmWeight;
use scratchy_subtile::subtile_ir::{EwKind, GainConvention, SubOp, SubtileIR, SubtileNode};
use scratchy_subtile::subtile_ir::{TensorRegion, TensorShape};
use scratchy_subtile::subtile_tape::{ComputeInput, ComputeInputs, SlotId};
use scratchy_subtile::tape_colouring::{Colour, TapeColouring};
use scratchy_subtile::tape_folding::{FusedShape, Fusion};
use scratchy_target_cuda::op_abi::CudaFusion;
use scratchy_target_cuda::step::{
    ADD_BF16_COLS, AddBf16, CudaStep, GEMM_BF16_TILE, GemmBf16, RMS_NORM_BF16_COLS, RmsNormBf16,
    SILU_MUL_BF16_COLS, SiluMulBf16,
};
use scratchy_target_cuda::tape::{ArenaOffset, WeightIx};

use crate::canonical::{
    ArenaRefusal, CanonicalRefusal, GemmShapeRule, OperandRefusal, RowShapeRule,
};

/// What a step's lowering reads besides the step itself: the planned canonical, and where each
/// colour lives in the arena.
pub struct StepCx<'a> {
    pub graph: &'a SubtileIR,
    /// Parallel to the graph's sources.
    pub bindings: &'a [SourceBinding],
    pub colouring: &'a TapeColouring,
    /// Each colour's byte offset, or why the arena has no layout.
    pub offsets: Result<&'a [u64], ArenaRefusal>,
}

/// One `Compute` of the tape: its position among the `Compute`s, its node, the slot it writes and
/// its positional inputs.
pub struct TapeStep<'a> {
    pub step: usize,
    pub node: &'a SubtileNode,
    pub writes: SlotId,
    pub inputs: &'a ComputeInputs,
}

/// Lower the op of one kept `Compute` on the tape to its CUDA step.
pub fn lower_step(cx: &StepCx<'_>, s: &TapeStep<'_>) -> Result<CudaStep, CanonicalRefusal> {
    use EwKind as E;
    use SubOp as L;
    let op = &s.node.op;
    match op {
        L::MatmulTile { n, weight } => gemm(cx, s, *n, *weight).map(CudaStep::Gemm),
        L::RmsNorm { eps, gain } => rms_norm(cx, s, *eps, *gain).map(CudaStep::RmsNorm),
        L::Elementwise(E::Add) => add(cx, s).map(CudaStep::Add),
        // SwiGLU runs only as the `SiluMul` fold: a `Silu` the fold left, a `Mul` driving none,
        // and the pre-fused op (which only spyre's `fuse_silu_mul` makes) have no kernel.
        L::Elementwise(E::Silu | E::Mul) | L::SiluMul => Err(CanonicalRefusal::Unfused {
            step: s.step,
            op: op.name(),
        }),
        L::SumReduce { .. }
        | L::Reshape { .. }
        | L::Elementwise(E::Gelu | E::QuickGelu | E::GeluErf | E::Relu2 | E::Sub | E::BiasAdd)
        | L::ScalarMul { .. }
        | L::RmsNormReduce { .. }
        | L::RmsNormApply { .. }
        | L::RopeRotate { .. }
        | L::RopeAppend { .. }
        | L::KvAppend { .. }
        | L::AttnDecode { .. }
        | L::TanhSoftCap { .. }
        | L::RmsNormUnit { .. }
        | L::ScalarWeightMul
        | L::GateSplit { .. }
        | L::GateApply
        | L::Concat { .. }
        | L::GateScale
        | L::LoadRows { .. }
        | L::EmbeddingGather { .. }
        | L::VisionRope
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
        | L::AllRowsMatmul => Err(CanonicalRefusal::NotLowered {
            step: s.step,
            op: op.name(),
        }),
    }
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

/// `MatmulTile` as the bf16 GEMM `C[m, n] = A[m, k] · B[n, k]ᵀ`.
///
/// The IR states the weight as `[K, N]`; the bytes are the checkpoint's `[N, K]`, which is what
/// the kernel takes (metal's `gemm.metal` the same). So `k` is the weight's IR rows (as metal's
/// `steps_from_tape::matmul` reads it), `n` the op's, and `m` the output's rows.
fn gemm(
    cx: &StepCx<'_>,
    s: &TapeStep<'_>,
    n: u32,
    weight: GemmWeight,
) -> Result<GemmBf16, CanonicalRefusal> {
    let (step, node) = (s.step, s.node);
    let (act, wt) = match node.inputs.as_slice() {
        [act, wt, ..] => (act, wt),
        _ => return Err(operand(s, 1, OperandRefusal::Missing)),
    };
    let m = cx.graph.shape(node.output.tensor).rows;
    let k = cx.graph.shape(wt.tensor).rows;
    gemm_shape(m, n, k).map_err(|rule| CanonicalRefusal::GemmShape {
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
    Ok(GemmBf16 {
        a: at(a),
        b,
        c: at(colour_of(cx, s.writes)),
        m,
        n,
        k,
    })
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
    let act = (computes.iter().find(|c| c.writes == silu))
        .expect("the fold pass names an activation slot a Compute of this tape writes");
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

/// The bf16 GEMM's shape rule: `m`, `n` and `k` non-zero multiples of [`GEMM_BF16_TILE`], and
/// each at most `i32::MAX` (ThunderKittens' `gl` stores dimensions as `int`).
fn gemm_shape(m: u32, n: u32, k: u32) -> Result<(), GemmShapeRule> {
    if m == 1 {
        return Err(GemmShapeRule::Decode);
    }
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
