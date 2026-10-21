// SPDX-License-Identifier: Apache-2.0
//! CUDA's step: one kernel launch of a lowered tape, the counterpart of metal's `MetalStep`.

use crate::tape::{ArenaOffset, WeightIx};

/// One launch-emitting step of a lowered tape, its operands already resolved.
///
/// It gains one variant per opcode CUDA lowers, each added by the issue that writes that opcode's
/// kernel, beside that kernel's registry row. The tape player matches it exhaustively, so a new
/// variant does not compile until the player launches it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CudaStep {
    /// The bf16 GEMM, lowered from `SubOp::MatmulTile` (`kernels/gemm_bf16.cu`).
    Gemm(GemmBf16),
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
