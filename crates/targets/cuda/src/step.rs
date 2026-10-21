// SPDX-License-Identifier: Apache-2.0
//! CUDA's step: one kernel launch of a lowered tape, the counterpart of metal's `MetalStep`.

use scratchy_subtile::subtile_ir::GainConvention;

use crate::tape::{ArenaOffset, ArenaSpan, CUDA_ACT_BYTES, WeightIx};

/// One launch-emitting step of a lowered tape, its operands already resolved.
///
/// It gains one variant per opcode CUDA lowers, each added by the issue that writes that opcode's
/// kernel, beside that kernel's registry row and its arm of [`CudaStep::extents`]. The tape
/// player matches it exhaustively, so a new variant does not compile until the player launches
/// it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CudaStep {
    /// The bf16 GEMM, lowered from `SubOp::MatmulTile` (`kernels/gemm_bf16.cu`).
    Gemm(GemmBf16),
    /// The bf16 RMSNorm, lowered from `SubOp::RmsNorm` (`kernels/rms_norm_bf16.cu`).
    RmsNorm(RmsNormBf16),
    /// The bf16 elementwise add, lowered from `Elementwise(Add)` (`kernels/add_bf16.cu`).
    Add(AddBf16),
    /// `silu(gate) · up` in one launch, lowered from the `SiluMul` fold of an `Elementwise(Silu)`
    /// into the `Elementwise(Mul)` it gates (`kernels/silu_mul_bf16.cu`).
    SiluMul(SiluMulBf16),
}

/// The bf16 GEMM's tile `(m, n, k)`: every `m`, `n` and `k` it takes is a non-zero multiple of
/// these (`kernels/gemm_bf16.cu`: two 64-row consumer warpgroups, four 64-column B tiles, one
/// 64-wide K step).
pub const GEMM_BF16_TILE: (u32, u32, u32) = (128, 256, 64);

/// `C[m, n] = A[m, k] · B[n, k]ᵀ`, bf16 in and out, accumulated in f32.
///
/// `A` and `C` are arena slots; `B` is a weight buffer in the checkpoint's `[N, K]` storage order
/// (the IR states it as `[K, N]`; the bytes are the checkpoint's). `C` overlaps neither operand:
/// CUDA's colour rule for a matmul reads its inputs across the step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GemmBf16 {
    pub a: ArenaOffset,
    pub b: WeightIx,
    pub c: ArenaOffset,
    pub m: u32,
    pub n: u32,
    pub k: u32,
}

/// The RMSNorm's `cols` is a non-zero multiple of this: each thread loads and stores 16-byte
/// vectors of 8 bf16 (`kernels/rms_norm_bf16.cu`). `rows` is non-zero and `rows · cols` at most
/// `i32::MAX`.
pub const RMS_NORM_BF16_COLS: u32 = 8;

/// `y[r, c] = x[r, c] / rms(x[r, :]) · (offset + gain[c])`, `rms(x) = sqrt(mean(x²) + eps)`,
/// bf16 in and out, computed in f32 and rounded once.
///
/// `x` and `y` are `[rows, cols]` arena slots; `gain` is a `[cols]` bf16 weight. `offset` is the
/// convention's ([`GainConvention::offset`]): 0 for a stored gain, 1 for a zero-centred one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RmsNormBf16 {
    pub x: ArenaOffset,
    pub gain: WeightIx,
    pub y: ArenaOffset,
    pub rows: u32,
    pub cols: u32,
    /// `eps` as its bits (`f32::to_bits`), so the step is `Eq`; [`Self::eps`] reads it back.
    pub eps_bits: u32,
    pub convention: GainConvention,
}

impl RmsNormBf16 {
    /// The norm's `eps`.
    pub const fn eps(&self) -> f32 {
        f32::from_bits(self.eps_bits)
    }
}

/// The add's `cols` is a non-zero multiple of this (`kernels/add_bf16.cu`: one 8-wide bf16
/// vector per thread), `rows` non-zero, and `rows · cols` at most `i32::MAX`.
pub const ADD_BF16_COLS: u32 = 8;

/// `out = a + b` over `[rows, cols]`, bf16 in and out: one f32 add, one rounding. No broadcast:
/// all three are arena slots of the same shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AddBf16 {
    pub a: ArenaOffset,
    pub b: ArenaOffset,
    pub out: ArenaOffset,
    pub rows: u32,
    pub cols: u32,
}

/// The SiluMul's `cols` is a non-zero multiple of this (`kernels/silu_mul_bf16.cu`: one 8-wide
/// bf16 vector per thread), `rows` non-zero, and `rows · cols` at most `i32::MAX`.
pub const SILU_MUL_BF16_COLS: u32 = 8;

/// `out = gate / (1 + e^(−gate)) · up` over `[rows, cols]`, bf16 in and out, computed in f32 and
/// rounded once. No broadcast: all three are arena slots of the same shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SiluMulBf16 {
    pub gate: ArenaOffset,
    pub up: ArenaOffset,
    pub out: ArenaOffset,
    pub rows: u32,
    pub cols: u32,
}

/// A weight a step reads, and the bytes it reads of it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WeightRead {
    pub ix: WeightIx,
    pub bytes: u64,
}

/// What one step touches: the arena spans it reads, in operand order, the span it writes, and
/// the weight it reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StepExtents {
    pub reads: [Option<ArenaSpan>; 2],
    pub writes: ArenaSpan,
    pub weight: Option<WeightRead>,
}

impl CudaStep {
    /// What the step's kernel reads and writes, in bytes; `None` when a byte count overflows
    /// `u64`. Each arm states its own kernel's operands, next to that kernel's registry row:
    /// `CudaTape::validate` checks a tape against it.
    pub const fn extents(&self) -> Option<StepExtents> {
        match *self {
            Self::Gemm(GemmBf16 { a, b, c, m, n, k }) => {
                let (Some(ab), Some(bb), Some(cb)) =
                    (bf16_bytes(m, k), bf16_bytes(n, k), bf16_bytes(m, n))
                else {
                    return None;
                };
                Some(StepExtents {
                    reads: [Some(span(a, ab)), None],
                    writes: span(c, cb),
                    weight: Some(WeightRead { ix: b, bytes: bb }),
                })
            }
            Self::RmsNorm(RmsNormBf16 {
                x,
                gain,
                y,
                rows,
                cols,
                ..
            }) => {
                let (Some(xb), Some(gb)) = (bf16_bytes(rows, cols), bf16_bytes(1, cols)) else {
                    return None;
                };
                Some(StepExtents {
                    reads: [Some(span(x, xb)), None],
                    writes: span(y, xb),
                    weight: Some(WeightRead {
                        ix: gain,
                        bytes: gb,
                    }),
                })
            }
            Self::Add(AddBf16 {
                a,
                b,
                out,
                rows,
                cols,
            })
            | Self::SiluMul(SiluMulBf16 {
                gate: a,
                up: b,
                out,
                rows,
                cols,
            }) => {
                let Some(bytes) = bf16_bytes(rows, cols) else {
                    return None;
                };
                Some(StepExtents {
                    reads: [Some(span(a, bytes)), Some(span(b, bytes))],
                    writes: span(out, bytes),
                    weight: None,
                })
            }
        }
    }
}

/// `rows × cols` bf16 elements' bytes, or `None` on overflow.
const fn bf16_bytes(rows: u32, cols: u32) -> Option<u64> {
    (rows as u64 * cols as u64).checked_mul(CUDA_ACT_BYTES)
}

const fn span(offset: ArenaOffset, bytes: u64) -> ArenaSpan {
    ArenaSpan { offset, bytes }
}
