// SPDX-License-Identifier: Apache-2.0
//! Metal's per-opcode ABI as DECLARED DATA.
//!
//! Target-ABI facts (in-place kernels, write bindings, barrier
//! classes) are small const tables the shared passes take as *input*
//! — never logic. This module holds the first such fact: WHICH
//! OPERAND an op's kernel writes over.
//!
//! That fact has two consumers today and they must agree or the
//! output is garbage: the slot colorer (an in-place op's output and
//! its aliased operand need ONE color) and the emitter arm (which
//! passes the same slot as both input and output). Stating it once
//! removes the class of bug where a kernel is made in-place in one
//! place and not the other.
//!
//! Ops whose aliasing is CONTEXTUAL are absent by design and keep
//! their code: a fused `RmsNorm` aliases the Add it folded (which
//! Add depends on the fold plan), a normed rope peels back through
//! its norm to the raw projection, and `ScalarMul` aliases only when
//! its scale is exactly 1.0 — a value predicate, not an opcode fact.

use scratchy_ir::{BiasStorage, Instruction, KvOffset, KvOffsets};
use scratchy_subtile::handoff::{WeightKind, accessor_slot};
use scratchy_subtile::lower::LoweredOp;

// ── The KV writer's weight site ─────────────────────────────────────
//
// Two declarations, adjacent: how the bridge LAYS OUT a `RopeAppend`'s
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
/// accessor slots, by the rule the per-arch `WeightAccessors` are emitted with
/// ([`accessor_slot`]) — with the storage each bias is read from.
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

/// The operand index this op's metal kernel writes OVER, if any.
///
/// `Some(k)` means the kernel mutates operand `k` in place and
/// publishes it — the colorer must give the output and operand `k`
/// one color. `None` means a fresh output buffer, OR that the
/// aliasing is contextual and its owner decides (see module docs).
///
/// The match is exhaustive: a new `LoweredOp` fails to compile here
/// (E0004) until its author states which case it is, rather than
/// silently defaulting to "fresh buffer" and losing a color.
pub fn metal_alias_operand(op: &LoweredOp) -> Option<u8> {
    use LoweredOp as L;
    match op {
        // Metadata-only view over operand 0 — no kernel runs.
        L::Reshape { .. } => Some(0),
        // `add_inplace` mutates the residual (operand 1) and
        // publishes it.
        L::Add => Some(1),
        // Elementwise activations rewrite their input buffer.
        L::TanhSoftCap | L::Gelu | L::QuickGelu | L::GeluErf => Some(0),

        // Fresh output, or contextual (decided by the fold plan /
        // value predicates — see module docs).
        L::Gemm { .. }
        | L::RmsNorm { .. }
        | L::RmsNormUnit { .. }
        | L::ScalarMul { .. }
        | L::ScalarWeightMul
        | L::BiasAdd
        | L::Silu
        | L::SiluMul
        | L::Mul
        | L::Sub
        | L::Mean
        | L::Moe { .. }
        | L::GemmaMoe { .. }
        | L::GateSplit { .. }
        | L::GateApply
        | L::GateScale
        | L::GatedDeltaNet
        | L::RopeRotate { .. }
        | L::RopeAppend { .. }
        | L::AttnDecode { .. }
        | L::EncoderAttn { .. }
        | L::VarlenAttention { .. }
        | L::VisionRope
        | L::LoadPixels { .. }
        | L::LoadPosEmbeds { .. }
        | L::EmbeddingGather { .. } => None,
    }
}

// ── The elementwise-unary class ─────────────────────────────────────
//
// Two declarations, adjacent, in THIS file: the pattern the bridge
// matches, and the constructor each op emits. Adding an activation is
// two lines here and NOTHING in the bridge — the arm that used to be
// written per-op is now written once, over the class.
//
// A macro in PATTERN position is what makes that possible while
// keeping the bridge's match exhaustive. The obvious alternative —
// an `if let ... continue` before the match — does NOT work: the
// compiler still demands every variant textually, so the arms come
// back and the site is not absorbed. Exhaustiveness is not
// negotiable here; it is the gate that stops a new opcode being
// silently forgotten (`cc62c55a8`).

/// The ops whose entire emission is "read operand 0's slot, write my
/// own". Used in pattern position by the bridge's single arm.
#[macro_export]
macro_rules! metal_unary_pat {
    () => {
        LoweredOp::Gelu | LoweredOp::QuickGelu | LoweredOp::GeluErf | LoweredOp::TanhSoftCap
    };
}

/// The `Instruction` each member of that class emits. Kept beside the
/// pattern so the two cannot drift: a name in one and not the other
/// is a compile error at the `expect` below, not a silent miss.
pub fn metal_unary_ctor(op: &LoweredOp) -> Option<fn(u32, u32) -> Instruction> {
    use LoweredOp as L;
    Some(match op {
        L::Gelu => Instruction::Gelu,
        L::QuickGelu => Instruction::QuickGelu,
        L::GeluErf => Instruction::GeluErf,
        L::TanhSoftCap => Instruction::TanhSoftCap,
        _ => return None,
    })
}

// ── The source class ────────────────────────────────────────────────
//
// Zero-input ops whose emission is "write my own slot": the host
// stages the data (pixels, interpolated position embeddings) and the
// command copies it in. Same shape as the unary class one level
// down — a pattern the bridge matches once, and the constructor row
// beside it.

/// Ops that read NO operand and write only their own slot.
#[macro_export]
macro_rules! metal_source_pat {
    () => {
        LoweredOp::LoadPixels { .. } | LoweredOp::LoadPosEmbeds { .. }
    };
}

/// The `Instruction` each source op emits.
pub fn metal_source_ctor(op: &LoweredOp) -> Option<fn(u32) -> Instruction> {
    use LoweredOp as L;
    Some(match op {
        L::LoadPixels { .. } => Instruction::LoadPixels,
        L::LoadPosEmbeds { .. } => Instruction::LoadPosEmbeds,
        _ => return None,
    })
}

// ── The one-in/one-out class, WITH fields ───────────────────────────
//
// The answer to "can a field-reading op be declared data?" is yes,
// for the ops that bind no weights: the row carries a BUILD closure
// that reads the op's own fields, and the bridge supplies the
// resolved slots. `EmbeddingGather` needs its `indices_kind`, and
// that is all that separated it from the unary class.
//
// The boundary is weights and fusion, not fields. An op that pushes
// `WeightSlot`s (Gemm, RmsNorm, BiasAdd, the MoE family) needs the
// accessor machinery the bridge owns, and an op whose shape depends
// on the FOLD PLAN (fused norms, normed rope) is not a function of
// its own fields at all. Those keep per-op arms and always will.

/// N inputs, one output, no weights, fields read from the op.
#[macro_export]
macro_rules! metal_in_out_pat {
    () => {
        LoweredOp::EmbeddingGather { .. } | LoweredOp::GateScale | LoweredOp::Reshape { .. }
    };
}

/// How many operand slots the bridge resolves for a member of the
/// in/out class, and whether its hazard signature is METADATA-only
/// (a view that runs no kernel, so it fences nothing).
pub fn metal_in_out_shape(op: &LoweredOp) -> Option<(u8, bool)> {
    use LoweredOp as L;
    Some(match op {
        L::EmbeddingGather { .. } => (1, false),
        L::GateScale => (3, false),
        // A reshape publishes a VIEW: no kernel runs, so the barrier
        // walk must not fence on it.
        L::Reshape { .. } => (1, true),
        _ => return None,
    })
}

/// Build the instruction from the op's fields plus the resolved
/// operand slots and output slot.
pub fn metal_in_out_build(op: &LoweredOp, ins: &[u32], out: u32) -> Option<Instruction> {
    use LoweredOp as L;
    Some(match op {
        L::EmbeddingGather { indices_kind } => {
            Instruction::EmbeddingGather(ins[0], out, *indices_kind)
        }
        L::GateScale => Instruction::GateScale(ins[0], ins[1], ins[2], out),
        L::Reshape {
            rows_mult,
            rows_div,
            cols,
        } => {
            let mut dims = [1u32; 4];
            dims[0] = *rows_mult;
            dims[1] = *cols;
            Instruction::Reshape(ins[0], out, dims, [1, 0, 0, 0], [*rows_div, 1, 1, 1], 2)
        }
        _ => return None,
    })
}
