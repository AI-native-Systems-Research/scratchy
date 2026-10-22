// SPDX-License-Identifier: Apache-2.0
//! One qwen3.5-0.8b MLP block, `y = x + down(silu(gate(n)) · up(n))` with `n = RmsNorm(x, g)`: the
//! canonical the row-op tape tests lower, and its `static` tapes written out by hand from the
//! issue's numbers, which `tape_layout` checks against what the compiler lowers.
//!
//! Shapes are qwen3.5-0.8b's: hidden `H = 1024`, intermediate `I = 3584`, `rms_norm_eps` 1e-6,
//! and the zero-centred gain (`OnePlusScale`). Every weight is declared in the IR as `[K, N]`; its
//! bytes on the device are `[N, K]`. Source 0 is the activation, rebound to the embedded hidden:
//! colour 0, the forward's input.
//!
//! Colours: 0 `x`, 1 `n`, 2 `gate`, 3 `up`, 4 `s` (the absorbed `Silu`'s, which no step writes or
//! reads), 5 `a`, 6 `y`. `d` reuses colour 1: `n` is dead after the up GEMM.

use scratchy_subtile::handoff::LoweredDecode;
use scratchy_subtile::lower::{GemmWeight, InputRef, LoweringInput, OpDesc};
use scratchy_subtile::subtile_ir::{EwKind, GainConvention, SourceShape, SubOp};
use scratchy_target_cuda::step::{AddBf16, CudaStep, GemmBf16, RmsNormBf16, SiluMulBf16};
use scratchy_target_cuda::tape::{
    ArenaLayout, ArenaOffset, ArenaSpan, CudaTape, TapeInput, WeightIx,
};

use super::chains::with_hidden;

/// qwen3.5-0.8b's hidden size.
pub const H: u32 = 1024;
/// qwen3.5-0.8b's MLP intermediate size.
pub const I: u32 = 3584;
/// qwen3.5's `rms_norm_eps`.
pub const EPS: f32 = 1e-6;

/// The block's ops as the macro emits them for qwen3.5 (`Silu` and `Mul` apart, the residual add
/// as `add(delta, residual)`), at `m` rows. Sources: 0 `x [m, H]`, 1 `g [1, H]`, 2 `Wg [H, I]`,
/// 3 `Wu [H, I]`, 4 `Wd [I, H]`.
pub fn mlp_input(m: u32) -> LoweringInput {
    let matmul = |n, act, w| OpDesc {
        op: SubOp::MatmulTile {
            n,
            weight: GemmWeight::Dense,
        },
        m,
        inputs: vec![act, InputRef::Ext(w)],
    };
    let ew = |kind, inputs| OpDesc {
        op: SubOp::Elementwise(kind),
        m,
        inputs,
    };
    LoweringInput {
        exported: None,
        sources: vec![
            SourceShape { rows: m, cols: H },
            SourceShape { rows: 1, cols: H },
            SourceShape { rows: H, cols: I },
            SourceShape { rows: H, cols: I },
            SourceShape { rows: I, cols: H },
        ],
        ops: vec![
            // 0: n = RmsNorm(x, g)
            OpDesc {
                op: SubOp::RmsNorm {
                    eps: EPS,
                    gain: GainConvention::OnePlusScale,
                },
                m,
                inputs: vec![InputRef::Ext(0), InputRef::Ext(1)],
            },
            // 1: gate = n · Wg
            matmul(I, InputRef::Op(0), 2),
            // 2: up = n · Wu
            matmul(I, InputRef::Op(0), 3),
            // 3: s = silu(gate), absorbed into op 4
            ew(EwKind::Silu, vec![InputRef::Op(1)]),
            // 4: a = s · up, the SiluMul fold's driver
            ew(EwKind::Mul, vec![InputRef::Op(3), InputRef::Op(2)]),
            // 5: d = a · Wd
            matmul(H, InputRef::Op(4), 4),
            // 6: y = d + x, the result
            ew(EwKind::Add, vec![InputRef::Op(5), InputRef::Ext(0)]),
        ],
        result: 6,
    }
}

/// [`mlp_input`] as the front end binds it, with `x` rebound to the embedded hidden.
pub fn mlp(m: u32) -> LoweredDecode {
    with_hidden(mlp_input(m))
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

const fn norm(x: u64, y: u64, rows: u32) -> CudaStep {
    CudaStep::RmsNorm(RmsNormBf16 {
        x: ArenaOffset(x),
        gain: WeightIx(1),
        y: ArenaOffset(y),
        rows,
        cols: H,
        eps_bits: EPS.to_bits(),
        convention: GainConvention::OnePlusScale,
    })
}

const fn silu_mul(gate: u64, up: u64, out: u64, rows: u32) -> CudaStep {
    CudaStep::SiluMul(SiluMulBf16 {
        gate: ArenaOffset(gate),
        up: ArenaOffset(up),
        out: ArenaOffset(out),
        rows,
        cols: I,
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

/// `g` is `H` bf16; each weight matrix `H × I` bf16.
const MLP_WEIGHT_BYTES: &[u64] = &[0, 2048, 7340032, 7340032, 7340032];

/// [`mlp`]`(128)`: every offset of [`MLP_M256`], halved.
pub static MLP_M128: CudaTape = CudaTape {
    bucket_m: 128,
    arena: ArenaLayout {
        colour_bytes: &[262144, 262144, 917504, 917504, 917504, 917504, 262144],
        colour_offsets: &[0, 262144, 524288, 1441792, 2359296, 3276800, 4194304],
        total_bytes: 4456448,
    },
    input: TapeInput::Hidden(span(0, 262144)),
    result: span(4194304, 262144),
    weight_bytes: MLP_WEIGHT_BYTES,
    steps: &[
        norm(0, 262144, 128),
        gemm(262144, 2, 524288, 128, I, H),
        gemm(262144, 3, 1441792, 128, I, H),
        silu_mul(524288, 1441792, 3276800, 128),
        gemm(3276800, 4, 262144, 128, H, I),
        add(262144, 0, 4194304, 128),
    ],
    rotary: None,
    kv: None,
    gdn: None,
};

/// [`mlp`]`(256)`: `x` 0, `n` 524288, `gate` 1048576, `up` 2883584, `s` 4718592 (unused),
/// `a` 6553600, `y` 8388608; `d` over `n` at 524288.
pub static MLP_M256: CudaTape = CudaTape {
    bucket_m: 256,
    arena: ArenaLayout {
        colour_bytes: &[524288, 524288, 1835008, 1835008, 1835008, 1835008, 524288],
        colour_offsets: &[0, 524288, 1048576, 2883584, 4718592, 6553600, 8388608],
        total_bytes: 8912896,
    },
    input: TapeInput::Hidden(span(0, 524288)),
    result: span(8388608, 524288),
    weight_bytes: MLP_WEIGHT_BYTES,
    steps: &[
        norm(0, 524288, 256),
        gemm(524288, 2, 1048576, 256, I, H),
        gemm(524288, 3, 2883584, 256, I, H),
        silu_mul(1048576, 2883584, 6553600, 256),
        gemm(6553600, 4, 524288, 256, H, I),
        add(524288, 0, 8388608, 256),
    ],
    rotary: None,
    kv: None,
    gdn: None,
};
