// SPDX-License-Identifier: Apache-2.0
//! `LoweringInput` — flat, topologically-ordered, backend-agnostic
//! description of a solved forward.
//!
//! One [`OpDesc`] per op, each naming its op — THE op vocabulary [`SubOp`], at the [`Arch`]
//! stage — row count, and inputs (either a prior op's output or an external source — a
//! weight/activation/extern). It is deliberately a *plain data* mirror
//! of the macro crate's internal `Fuf` + solver `Assignment`: the
//! macro crate (which alone can see those types) does the trivial
//! structural translation and hands the result to
//! [`crate::subtile_ir::lower_region`], so all the decomposition logic
//! lives in this Mac-testable crate.
//!
//! [`fuse_silu_mul`] is a pre-pass that fuses adjacent `Silu` → `Mul`
//! into [`SubOp::SiluMul`] before lowering — the GPU has only the
//! fused arm, so fusion must happen before scheduling places the pair
//! on workers.

use crate::subtile_ir::{Arch, EwKind, NeoX, OpStage, RopeForm, SourceShape, SubOp};

/// One input edge of an op: a prior op's output, or an external source.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InputRef {
    /// Output of op `ops[idx]` (must be `< this op's index`).
    Op(usize),
    /// External source `sources[idx]` — a weight, activation, or extern.
    Ext(usize),
}

/// How a [`SubOp::MatmulTile`]'s WEIGHT is stored and matmul'd — the weight's quantization scheme AS A
/// TYPE. Deliberately NOT a bool: a bool can only say fp8-or-not, and the moment int4 weights land
/// ("we will have int4 weights soon") a second `weight_int4: bool` makes `weight_fp8 && weight_int4` a
/// nonsensical-but-constructible state. The `#[forward]` macro maps this at EXPANSION time from the
/// weight's `StorageFormat` (the scheme is already known at compile time — no need to launder it through
/// a runtime bool), and the SDSC emitter matches it EXHAUSTIVELY: adding a scheme is a `cargo build`
/// error at every lowering site until it is handled (guard-driven), never a silent misread-as-dense.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GemmWeight {
    /// Dense bf16/fp16 weight. Matmul inputs: `[act, weight]` (arity-2). The path every dense model takes.
    Dense,
    /// compressed-tensors fp8-dynamic: per-CHANNEL static fp8 weight + per-TOKEN dynamic fp8 activation.
    /// Matmul inputs: `[act, weight_fp8, weight_scale]` (arity-3). matmulfp8 (fp8×fp8→fp16) + dequant by
    /// `w_scale[n]·a_scale[m]`. Lowered by the fp8 branch in `lower_subtile_tape_to_superdsc`.
    Fp8Dynamic,
    /// MLX-affine packed weight (`bits` codes per element, one scale+bias per
    /// `group_size` elements of K). Matmul inputs stay arity-2 `[act, weight]`: the
    /// scales and biases resolve from the SAME weight source under different tensor
    /// roles, so they are not separate IR operands. Metal realizes it with the qmv
    /// family at decode and qmm_t at prefill. See [`AffineInt4`] for what is checked.
    Affine { affine: AffineInt4 },
}

/// A [`GemmWeight`]'s scheme without its parameters — how a target's declared table names one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GemmWeightKind {
    Dense,
    Fp8Dynamic,
    Affine,
}

impl GemmWeight {
    pub const fn kind(self) -> GemmWeightKind {
        match self {
            Self::Dense => GemmWeightKind::Dense,
            Self::Fp8Dynamic => GemmWeightKind::Fp8Dynamic,
            Self::Affine { .. } => GemmWeightKind::Affine,
        }
    }
}

/// Bits per packed weight element. Minted only for widths a kernel family exists for,
/// so a checkpoint declaring an unsupported width is refused at lowering rather than
/// unpacked with the wrong stride.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct QuantBits(u32);

impl QuantBits {
    pub const fn get(self) -> u32 {
        self.0
    }
}

/// Elements of K sharing one scale/bias pair.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct GroupSize(u32);

impl GroupSize {
    pub const fn get(self) -> u32 {
        self.0
    }
}

/// ⭐ AN MLX-AFFINE INT4 WEIGHT'S QUANTIZATION GEOMETRY — minted only by
/// [`AffineInt4::mint`], the ONE place `bits`, `group_size` and the matmul's `k` are
/// checked against each other.
///
/// ⛔ THE GROUP MUST TILE K, AND NOTHING DOWNSTREAM RE-CHECKS IT. The qmv kernels index
/// `scales[n][k / group_size]`; a `k` that is not a whole number of groups reads one
/// group past the end of the last row — which lands inside the NEXT output channel's
/// scales, so it dequantizes with a plausible wrong scale and produces fluent, wrong
/// output rather than faulting. `bits` is checked here for the same reason: the packing
/// stride is `32 / bits` elements per word, and a width no kernel exists for would
/// silently unpack at the wrong stride.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AffineInt4 {
    bits: QuantBits,
    group: GroupSize,
}

impl AffineInt4 {
    /// The packed widths a metal qmv kernel family exists for.
    const SUPPORTED_BITS: [u32; 2] = [4, 8];

    /// `Some` iff `bits` names a kernel family and `group_size` tiles `k` exactly.
    pub const fn mint(bits: u32, group_size: u32, k: u32) -> Option<Self> {
        if group_size == 0 || k == 0 || !k.is_multiple_of(group_size) {
            return None;
        }
        // `const fn` cannot iterate a slice; state the membership directly.
        if bits != Self::SUPPORTED_BITS[0] && bits != Self::SUPPORTED_BITS[1] {
            return None;
        }
        Some(Self {
            bits: QuantBits(bits),
            group: GroupSize(group_size),
        })
    }

    pub const fn bits(self) -> QuantBits {
        self.bits
    }

    pub const fn group(self) -> GroupSize {
        self.group
    }
}

/// An MoE expert bundle's MLX-affine storage, AS THE CHECKPOINT DECLARES IT.
///
/// ⚠️ NOT an [`AffineInt4`]: nothing checks the group against the projection's `k` or the width
/// against a kernel family. The expert kernels take the declared pair verbatim, as they always
/// have; minting would refuse configs that bake today.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExpertQuant {
    group: GroupSize,
    bits: QuantBits,
}

impl ExpertQuant {
    pub const fn declared(group_size: u32, bits: u32) -> Self {
        Self {
            group: GroupSize(group_size),
            bits: QuantBits(bits),
        }
    }

    pub const fn group(self) -> GroupSize {
        self.group
    }

    pub const fn bits(self) -> QuantBits {
        self.bits
    }
}

/// An op as the front end states it: THE op vocabulary [`SubOp`] at its [`Arch`] stage, before
/// [`crate::subtile_ir::lower_region`] binds the KV-cache witnesses.
pub type ArchOp = SubOp<NeoX, Arch>;

/// The learned constant an op's output carries on top of the input-dependent
/// part — what a per-vector-norm KV codec must remove from a cached K/V before
/// quantizing, or the constant's norm sets its error instead of the signal's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdditiveOffset {
    /// None: a projection, or a norm (which rescales per token, adding nothing).
    Absent,
    /// The op adds its bias source (operand 1) to operand 0.
    Bias,
    /// A view: operand 0's offset, unchanged.
    OfOperand0,
}

impl<F: RopeForm, S: OpStage> SubOp<F, S> {
    /// This op's [`AdditiveOffset`], or `None` for an op a cached K/V is never
    /// produced by. Exhaustive: a new op states its rule before it compiles.
    pub fn additive_offset(&self) -> Option<AdditiveOffset> {
        use SubOp as O;
        match self {
            O::MatmulTile { .. }
            | O::SumReduce { .. }
            | O::RmsNorm { .. }
            | O::RmsNormApply { .. }
            | O::RmsNormUnit { .. } => Some(AdditiveOffset::Absent),
            O::Elementwise(EwKind::BiasAdd) => Some(AdditiveOffset::Bias),
            O::Reshape { .. } => Some(AdditiveOffset::OfOperand0),
            O::Elementwise(
                EwKind::Silu
                | EwKind::Gelu
                | EwKind::Mul
                | EwKind::Add
                | EwKind::QuickGelu
                | EwKind::GeluErf
                | EwKind::Sub,
            )
            | O::RmsNormReduce { .. }
            | O::TanhSoftCap
            | O::ScalarWeightMul
            | O::GateSplit { .. }
            | O::GateApply
            | O::GateScale
            | O::LoadPixels { .. }
            | O::EmbeddingGather { .. }
            | O::LoadPosEmbeds { .. }
            | O::VisionRope
            | O::VarlenAttention { .. }
            | O::EncoderAttn { .. }
            | O::GatedDeltaNet
            | expansion_ops!()
            | O::ScalarMul { .. }
            | O::SiluMul
            | O::Mean
            | O::RopeRotate { .. }
            | O::RopeAppend { .. }
            | O::AttnDecode { .. } => None,
        }
    }
}

/// The projection — its op index and weight — whose bias the KV operand
/// produced by `ops[op]` carries, or `None` if it carries none: views are
/// followed back to the producer, which [`SubOp::additive_offset`] decides.
/// `Err` names a producer with no rule, or a bias on something not a projection.
pub fn kv_operand_bias(ops: &[OpDesc], op: usize) -> Result<Option<(usize, GemmWeight)>, String> {
    let source = || match ops[op].inputs.first() {
        Some(InputRef::Op(s)) => Ok(*s),
        _ => Err(format!(
            "op {op} ({:?}): operand 0 is not an op",
            ops[op].op
        )),
    };
    match ops[op].op.additive_offset() {
        Some(AdditiveOffset::Absent) => Ok(None),
        Some(AdditiveOffset::OfOperand0) => kv_operand_bias(ops, source()?),
        Some(AdditiveOffset::Bias) => match ops[source()?].op {
            SubOp::MatmulTile { weight, .. } => Ok(Some((source()?, weight))),
            other => Err(format!("op {op}: a KV bias on {other:?}, not a projection")),
        },
        None => Err(format!(
            "op {op} ({:?}) feeds a KV writer but has no additive-offset rule",
            ops[op].op
        )),
    }
}

#[cfg(test)]
mod additive_offset_tests {
    use super::{AdditiveOffset, ArchOp, GemmWeight};
    use crate::subtile_ir::{EwKind, GainConvention, RowScale, SubOp};

    /// Every producer a KV writer's operand has across the arches: Qwen2's
    /// `bias_add(gemm)` carries the bias — marked `Absent`, TurboQuant codes
    /// it and Qwen2 decodes garbage — while projections and the qk-norms
    /// (Qwen3, Gemma) add nothing, and a per-head view carries its source's.
    #[test]
    fn kv_operand_producers_declare_their_offset() {
        let gemm: ArchOp = SubOp::MatmulTile {
            n: 512,
            weight: GemmWeight::Dense,
        };
        let bias: ArchOp = SubOp::Elementwise(EwKind::BiasAdd);
        assert_eq!(bias.additive_offset(), Some(AdditiveOffset::Bias));
        for centered in [
            gemm,
            SubOp::RmsNorm {
                eps: 1e-6,
                gain: GainConvention::Scale,
            },
            SubOp::RmsNormUnit { eps: 1e-6 },
        ] {
            assert_eq!(centered.additive_offset(), Some(AdditiveOffset::Absent));
        }
        let view: ArchOp = SubOp::Reshape {
            rows: RowScale::Times(std::num::NonZeroU32::new(4).unwrap()),
            cols: 128,
        };
        assert_eq!(view.additive_offset(), Some(AdditiveOffset::OfOperand0));
        let add: ArchOp = SubOp::Elementwise(EwKind::Add);
        assert_eq!(add.additive_offset(), None);
    }
}

/// One op in the forward.
#[derive(Clone, Debug)]
pub struct OpDesc {
    pub op: ArchOp,
    /// Row count (M); decode is 1.
    pub m: u32,
    pub inputs: Vec<InputRef>,
}

/// A whole forward, ready to lower.
#[derive(Clone, Debug)]
pub struct LoweringInput {
    /// External tensors (weights / activations / externs), indexed by
    /// `InputRef::Ext` and by `SourceId` in the produced graph.
    pub sources: Vec<SourceShape>,
    /// Ops in topological order; op `i` may reference ops `< i`.
    pub ops: Vec<OpDesc>,
    /// Which op's output is the forward's result.
    pub result: usize,
}

// ── Silu + Mul → SiluMul fusion ─────────────────────────────────────

/// Fuse each adjacent `Silu` → `Mul(silu, up)` into one
/// [`SubOp::SiluMul`] (SwiGLU). The GPU has only a *fused* `silu_mul`
/// arm (no standalone silu), and fusion must happen **before scheduling**
/// so the pair lands on one worker — the serializer otherwise rejects a
/// standalone `Silu`. A `Silu` is fused only when its output feeds exactly
/// one consumer (that `Mul`) and is not the forward result; everything else
/// is left untouched. Bit-exact: `SiluMul` computes `silu(gate) * up`,
/// identical to the two separate ops, so `eval_dag` is unchanged.
///
/// `Mul` is commutative, so the `Silu`-producing operand becomes `gate` and
/// the other becomes `up` regardless of input order. Op references are
/// re-indexed after the dropped `Silu`s are removed.
pub fn fuse_silu_mul(input: &LoweringInput) -> LoweringInput {
    use std::collections::HashMap;
    let n = input.ops.len();

    // Count consumers of each op output (op inputs across the forward + the
    // result) so a multiply-consumed `Silu` is never duplicated by fusion.
    let mut uses = vec![0u32; n];
    for od in &input.ops {
        for r in &od.inputs {
            if let InputRef::Op(j) = r {
                uses[*j] += 1;
            }
        }
    }
    uses[input.result] += 1;

    // Decide fusions: mul index → (silu index, gate ref, up ref).
    let is_silu = |i: usize| matches!(input.ops[i].op, SubOp::Elementwise(EwKind::Silu));
    let mut fuse_at: HashMap<usize, (usize, InputRef, InputRef)> = HashMap::new();
    let mut dropped = vec![false; n];
    for (j, od) in input.ops.iter().enumerate() {
        if !matches!(od.op, SubOp::Elementwise(EwKind::Mul)) || od.inputs.len() != 2 {
            continue;
        }
        // A fusable operand is a single-use, non-result `Silu` output.
        let fusable = |r: InputRef, dropped: &[bool]| -> Option<usize> {
            if let InputRef::Op(i) = r
                && is_silu(i)
                && uses[i] == 1
                && input.result != i
                && !dropped[i]
            {
                return Some(i);
            }
            None
        };
        let (a, b) = (od.inputs[0], od.inputs[1]);
        let pick = fusable(a, &dropped)
            .map(|si| (si, b))
            .or_else(|| fusable(b, &dropped).map(|si| (si, a)));
        if let Some((si, up)) = pick {
            let gate = input.ops[si].inputs[0];
            fuse_at.insert(j, (si, gate, up));
            dropped[si] = true;
        }
    }
    if fuse_at.is_empty() {
        return input.clone();
    }

    // Old op index → new index, with the dropped `Silu`s removed.
    let mut new_idx = vec![usize::MAX; n];
    let mut next = 0usize;
    for (i, d) in dropped.iter().enumerate() {
        if !d {
            new_idx[i] = next;
            next += 1;
        }
    }
    let remap = |r: InputRef| -> InputRef {
        match r {
            InputRef::Op(j) => {
                debug_assert_ne!(
                    new_idx[j],
                    usize::MAX,
                    "ref to a dropped Silu survived fusion"
                );
                InputRef::Op(new_idx[j])
            }
            InputRef::Ext(e) => InputRef::Ext(e),
        }
    };

    let mut ops: Vec<OpDesc> = Vec::with_capacity(next);
    for (i, od) in input.ops.iter().enumerate() {
        if dropped[i] {
            continue;
        }
        if let Some((_, gate, up)) = fuse_at.get(&i) {
            ops.push(OpDesc {
                op: SubOp::SiluMul,
                m: od.m,
                inputs: vec![remap(*gate), remap(*up)],
            });
        } else {
            ops.push(OpDesc {
                op: od.op,
                m: od.m,
                inputs: od.inputs.iter().map(|r| remap(*r)).collect(),
            });
        }
    }
    LoweringInput {
        sources: input.sources.clone(),
        ops,
        result: new_idx[input.result],
    }
}
