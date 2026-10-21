// SPDX-License-Identifier: Apache-2.0
//! qwen3.5-0.8b's full-attention prelude and output gate: the canonical the attention-gate tape
//! tests lower, and its `static` tapes written out by hand from the issue's numbers, which
//! `tape_layout` checks against what the compiler lowers.
//!
//! The ops, in the order and at the shapes the macro emits them for the query side (the per-head
//! Reshape pair around `q_norm` inserted by shape inference): `n = RmsNorm(x)`, `qg = n · Wq`,
//! `q (+ gate) = GateSplit(qg)`, `qh = Reshape(q)`, `qn = RmsNorm(qh, q_norm)`, `qf =
//! Reshape(qn)`, `qr = RopeRotate(qf, cos, sin)`, then `ga = GateApply(·, gate)`, `o = ga · Wo`
//! and `y = o + x`. There is no attention yet, so `GateApply` gates the rotated query in place of
//! the attention output; every other op is the real tape's.
//!
//! Shapes: hidden `H = 1024`, 8 query heads of `HD = 256`, rotated width `R = 64`, `rms_norm_eps`
//! 1e-6, the zero-centred gain. Every weight is declared in the IR as `[K, N]`; its bytes on the
//! device are `[N, K]`. Source 0 is the activation, rebound to the embedded hidden; sources 4 and
//! 5 are the rotary `cos` and `sin`, rebound to the global table.
//!
//! Colours: 0 `x`, 1 `n` (then `o`), 2 `qg`, 3 `q` (and the view `qh`), 4 `qn` (and the view
//! `qf`), 5 `qr`, 6 `ga`, 7 `y`, 8 the gate. The rotary table follows the colours.

use std::num::NonZeroU32;

use scratchy_subtile::handoff::{LoweredDecode, SourceBinding};
use scratchy_subtile::lower::{GemmWeight, InputRef, LoweringInput, OpDesc};
use scratchy_subtile::model_geometry::HeadDim;
use scratchy_subtile::subtile_ir::{EwKind, GainConvention, RowScale, SourceShape, SubOp};
use scratchy_target_cuda::step::{
    AddBf16, CudaStep, GateApplyBf16, GateSplitBf16, GemmBf16, RmsNormBf16, RopeBf16,
};
use scratchy_target_cuda::tape::{
    ArenaLayout, ArenaOffset, ArenaSpan, CudaTape, RotaryTable, WeightIx,
};

use super::chains::with_hidden;

/// qwen3.5-0.8b's hidden size.
pub const H: u32 = 1024;
/// qwen3.5's head dim.
pub const HD: u32 = 256;
/// qwen3.5-0.8b's query heads.
pub const HEADS: u32 = 8;
/// The query width, `HEADS · HD`.
pub const Q: u32 = HEADS * HD;
/// qwen3.5's rotated width, `0.25 · HD`.
pub const R: u32 = 64;
/// qwen3.5's `rope_theta`.
pub const THETA: f64 = 1e7;
/// qwen3.5's `rms_norm_eps`.
pub const EPS: f32 = 1e-6;

/// The block's ops at `m` rows. Sources: 0 `x [m, H]`, 1 `g_in [1, H]`, 2 `Wq [H, 2Q]`,
/// 3 `q_norm [1, HD]`, 4 `cos [m, Q]`, 5 `sin [m, Q]`, 6 `Wo [Q, H]`.
pub fn attn_gate_input(m: u32) -> LoweringInput {
    use InputRef::{Ext, Op};
    let norm = SubOp::RmsNorm {
        eps: EPS,
        gain: GainConvention::OnePlusScale,
    };
    let matmul = |n| SubOp::MatmulTile {
        n,
        weight: GemmWeight::Dense,
    };
    let op = |op, inputs| OpDesc { op, m, inputs };
    let shape = |rows, cols| SourceShape { rows, cols };
    LoweringInput {
        exported: None,
        sources: vec![
            shape(m, H),
            shape(1, H),
            shape(H, 2 * Q),
            shape(1, HD),
            shape(m, Q),
            shape(m, Q),
            shape(Q, H),
        ],
        ops: vec![
            // 0: n = RmsNorm(x, g_in)
            op(norm, vec![Ext(0), Ext(1)]),
            // 1: qg = n · Wq
            op(matmul(2 * Q), vec![Op(0), Ext(2)]),
            // 2: q = GateSplit(qg), and the gate, its second output
            op(SubOp::GateSplit { half_cols: Q }, vec![Op(1)]),
            // 3: qh = the per-head view [8m, 256]
            op(
                SubOp::Reshape {
                    rows: RowScale::Times(NonZeroU32::new(HEADS).expect("HEADS is non-zero")),
                    cols: HD,
                },
                vec![Op(2)],
            ),
            // 4: qn = RmsNorm(qh, q_norm), one norm per head
            op(norm, vec![Op(3), Ext(3)]),
            // 5: qf = the flatten-back view [m, 2048]
            op(
                SubOp::Reshape {
                    rows: RowScale::Times(NonZeroU32::MIN),
                    cols: Q,
                },
                vec![Op(4)],
            ),
            // 6: qr = RopeRotate(qf, cos, sin)
            op(
                SubOp::rope_rotate(HeadDim::new(HD)),
                vec![Op(5), Ext(4), Ext(5)],
            ),
            // 7: ga = GateApply(qr, gate): operand 1 names op 2 and means its second output
            op(SubOp::GateApply, vec![Op(6), Op(2)]),
            // 8: o = ga · Wo
            op(matmul(H), vec![Op(7), Ext(6)]),
            // 9: y = o + x, the result
            op(SubOp::Elementwise(EwKind::Add), vec![Op(8), Ext(0)]),
        ],
        result: 9,
    }
}

/// [`attn_gate_input`] as the front end binds it, with `x` rebound to the embedded hidden and
/// `cos`, `sin` to the global rotary table.
pub fn attn_gate(m: u32) -> LoweredDecode {
    let mut l = with_hidden(attn_gate_input(m));
    l.bindings[4] = SourceBinding::Cos { local: false };
    l.bindings[5] = SourceBinding::Sin { local: false };
    l
}

const fn span(offset: u64, bytes: u64) -> ArenaSpan {
    ArenaSpan {
        offset: ArenaOffset(offset),
        bytes,
    }
}

const fn gemm(a: u64, b: u32, c: u64, m: u32, n: u32, k: u32) -> CudaStep {
    CudaStep::Gemm(GemmBf16 {
        a: ArenaOffset(a),
        b: WeightIx(b),
        c: ArenaOffset(c),
        m,
        n,
        k,
    })
}

const fn norm(x: u64, gain: u32, y: u64, rows: u32, cols: u32) -> CudaStep {
    CudaStep::RmsNorm(RmsNormBf16 {
        x: ArenaOffset(x),
        gain: WeightIx(gain),
        y: ArenaOffset(y),
        rows,
        cols,
        eps_bits: EPS.to_bits(),
        convention: GainConvention::OnePlusScale,
    })
}

const fn gate_split(qg: u64, q: u64, gate: u64, rows: u32) -> CudaStep {
    CudaStep::GateSplit(GateSplitBf16 {
        qg: ArenaOffset(qg),
        q: ArenaOffset(q),
        gate: ArenaOffset(gate),
        rows,
        half_cols: Q,
        head_dim: HD,
    })
}

const fn rope(x: u64, table: u64, out: u64, rows: u32) -> CudaStep {
    CudaStep::Rope(RopeBf16 {
        x: ArenaOffset(x),
        table: ArenaOffset(table),
        out: ArenaOffset(out),
        rows,
        cols: Q,
        head_dim: HD,
        rot_dim: R,
    })
}

const fn gate_apply(attn: u64, gate: u64, out: u64, rows: u32) -> CudaStep {
    CudaStep::GateApply(GateApplyBf16 {
        attn: ArenaOffset(attn),
        gate: ArenaOffset(gate),
        out: ArenaOffset(out),
        rows,
        cols: Q,
    })
}

const fn add(a: u64, b: u64, out: u64, rows: u32) -> CudaStep {
    CudaStep::Add(AddBf16 {
        a: ArenaOffset(a),
        b: ArenaOffset(b),
        out: ArenaOffset(out),
        rows,
        cols: H,
    })
}

const fn rotary(offset: u64, bytes: u64) -> Option<RotaryTable> {
    Some(RotaryTable {
        span: span(offset, bytes),
        rot_dim: R,
        theta_bits: THETA.to_bits(),
    })
}

/// `g_in` is `H` bf16, `Wq` `2Q × H`, `q_norm` `HD`, `Wo` `H × Q`; `cos` and `sin` are not
/// uploaded (every rope reads the rotary table).
const ATTN_GATE_WEIGHT_BYTES: &[u64] = &[0, 2048, 8388608, 512, 0, 0, 4194304];

/// [`attn_gate`]`(128)`: every offset of [`ATTN_GATE_M256`] halved, and the per-head norm's rows
/// with them.
pub static ATTN_GATE_M128: CudaTape = CudaTape {
    bucket_m: 128,
    arena: ArenaLayout {
        colour_bytes: &[
            262144, 262144, 1048576, 524288, 524288, 524288, 524288, 262144, 524288,
        ],
        colour_offsets: &[
            0, 262144, 524288, 1572864, 2097152, 2621440, 3145728, 3670016, 3932160,
        ],
        total_bytes: 4472832,
    },
    input: span(0, 262144),
    result: span(3670016, 262144),
    weight_bytes: ATTN_GATE_WEIGHT_BYTES,
    steps: &[
        norm(0, 1, 262144, 128, H),
        gemm(262144, 2, 524288, 128, 2 * Q, H),
        gate_split(524288, 1572864, 3932160, 128),
        norm(1572864, 3, 2097152, 1024, HD),
        rope(2097152, 4456448, 2621440, 128),
        gate_apply(2621440, 3932160, 3145728, 128),
        gemm(3145728, 6, 262144, 128, H, Q),
        add(262144, 0, 3670016, 128),
    ],
    rotary: rotary(4456448, 16384),
};

/// [`attn_gate`]`(256)`: `x` 0, `n` (then `o`) 524288, `qg` 1048576, `q` 3145728, `qn` 4194304,
/// `qr` 5242880, `ga` 6291456, `y` 7340032, the gate 7864320; the rotary table at 8912896.
pub static ATTN_GATE_M256: CudaTape = CudaTape {
    bucket_m: 256,
    arena: ArenaLayout {
        colour_bytes: &[
            524288, 524288, 2097152, 1048576, 1048576, 1048576, 1048576, 524288, 1048576,
        ],
        colour_offsets: &[
            0, 524288, 1048576, 3145728, 4194304, 5242880, 6291456, 7340032, 7864320,
        ],
        total_bytes: 8945664,
    },
    input: span(0, 524288),
    result: span(7340032, 524288),
    weight_bytes: ATTN_GATE_WEIGHT_BYTES,
    steps: &[
        norm(0, 1, 524288, 256, H),
        gemm(524288, 2, 1048576, 256, 2 * Q, H),
        gate_split(1048576, 3145728, 7864320, 256),
        norm(3145728, 3, 4194304, 2048, HD),
        rope(4194304, 8912896, 5242880, 256),
        gate_apply(5242880, 7864320, 6291456, 256),
        gemm(6291456, 6, 524288, 256, H, Q),
        add(524288, 0, 7340032, 256),
    ],
    rotary: rotary(8912896, 32768),
};
