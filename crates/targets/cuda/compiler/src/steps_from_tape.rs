// SPDX-License-Identifier: Apache-2.0
//! CUDA'S OPCODE LOWERING: one tape step to one [`CudaStep`], its operands resolved to arena
//! offsets and weight indices.
//!
//! The match lists every `SubOp` variant, with no wildcard, so a new `SubOp` fails to compile here
//! (E0004) as it does in the colour rule. Each opcode issue moves exactly one variant out of the
//! refusing arm into an arm of its own; `MatmulTile` is the first.

use scratchy_subtile::handoff::SourceBinding;
use scratchy_subtile::lower::GemmWeight;
use scratchy_subtile::subtile_ir::{SubOp, SubtileIR, SubtileNode, TensorRegion};
use scratchy_subtile::subtile_tape::{ComputeInput, ComputeInputs, SlotId};
use scratchy_subtile::tape_colouring::{Colour, TapeColouring};
use scratchy_target_cuda::step::{CudaStep, GEMM_BF16_TILE, GemmBf16};
use scratchy_target_cuda::tape::{ArenaOffset, WeightIx};

use crate::canonical::{ArenaRefusal, CanonicalRefusal, GemmShapeRule, OperandRefusal};

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

/// Lower the op of one `Compute` on the tape to its CUDA step.
pub fn lower_step(cx: &StepCx<'_>, s: &TapeStep<'_>) -> Result<CudaStep, CanonicalRefusal> {
    use SubOp as L;
    let op = &s.node.op;
    match op {
        L::MatmulTile { n, weight } => gemm(cx, s, *n, *weight).map(CudaStep::Gemm),
        L::SumReduce { .. }
        | L::Reshape { .. }
        | L::Elementwise(_)
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
    for (k, region) in [(0, act), (1, wt)] {
        if !whole(cx.graph, region) {
            return Err(operand(s, k, OperandRefusal::PartialRead));
        }
    }
    if !whole(cx.graph, &node.output) {
        return Err(CanonicalRefusal::PartialWrite {
            step,
            op: s.node.op.name(),
        });
    }
    let offsets = cx.offsets.map_err(CanonicalRefusal::Arena)?;
    let at = |c: Colour| ArenaOffset(offsets[c.index() as usize]);
    let mut inputs = s.inputs.iter();
    let a = match inputs.next() {
        Some(ComputeInput::Computed(slots)) => match slots.as_slice() {
            [slot] => cx.colouring.colour_of(*slot).expect(
                "the colourer colours every arena step, and every CUDA step is on the arena",
            ),
            _ => {
                return Err(operand(s, 0, OperandRefusal::SplitWriters(slots.len())));
            }
        },
        Some(ComputeInput::External {
            tensor, per_layer, ..
        }) => {
            if !per_layer.is_empty() {
                return Err(operand(s, 0, OperandRefusal::PerLayer));
            }
            match &cx.bindings[tensor.index()] {
                SourceBinding::EmbeddedHidden => Colour::ZERO,
                binding => {
                    return Err(operand(
                        s,
                        0,
                        OperandRefusal::NotHiddenOrSlot {
                            source: tensor.index(),
                            binding: binding.clone(),
                        },
                    ));
                }
            }
        }
        None => return Err(operand(s, 0, OperandRefusal::Missing)),
    };
    let b = match inputs.next() {
        Some(ComputeInput::External {
            tensor, per_layer, ..
        }) => {
            if !per_layer.is_empty() {
                return Err(operand(s, 1, OperandRefusal::PerLayer));
            }
            match &cx.bindings[tensor.index()] {
                SourceBinding::Weight { .. } => WeightIx(tensor.index() as u32),
                binding => {
                    return Err(operand(
                        s,
                        1,
                        OperandRefusal::NotAWeight {
                            source: Some((tensor.index(), binding.clone())),
                        },
                    ));
                }
            }
        }
        Some(ComputeInput::Computed(_)) => {
            return Err(operand(s, 1, OperandRefusal::NotAWeight { source: None }));
        }
        None => return Err(operand(s, 1, OperandRefusal::Missing)),
    };
    let c = cx
        .colouring
        .colour_of(s.writes)
        .expect("the colourer colours every arena step, and every CUDA step is on the arena");
    Ok(GemmBf16 {
        a: at(a),
        b,
        c: at(c),
        m,
        n,
        k,
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
