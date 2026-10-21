// SPDX-License-Identifier: Apache-2.0
//! ONE BUCKET'S TAPE, AS STATIC DATA — the counterpart of metal's `LoweredMetalTape`.
//!
//! A tape is what `scratchy-target-cuda-compiler` lowers one canonical (one bucket) to: the
//! arena's layout, where the forward's input and result live in it, how large each weight must
//! be, and the steps in launch order with every operand already resolved to a byte offset or a
//! weight index. Every field is a constant built at expansion time; the runtime reads it and
//! analyses nothing. Plain `Copy` data with no pointer, so it builds in a `const` context and on a
//! host with no CUDA toolchain.

use crate::step::CudaStep;

/// Bytes per activation element: CUDA's activations are bf16.
pub const CUDA_ACT_BYTES: u64 = 2;

/// Every colour's arena offset is a multiple of this. TMA needs 16-byte-aligned global addresses
/// and `cudaMalloc` returns 256-byte-aligned memory, so every slot pointer meets the launchers'
/// alignment contract with margin.
pub const CUDA_ARENA_ALIGN: u64 = 256;

/// A byte offset into the worker's arena.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArenaOffset(pub u64);

/// A weight, by its source's index in the canonical: the index into the worker's weight buffers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WeightIx(pub u32);

/// One colour's bytes in the arena.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArenaSpan {
    pub offset: ArenaOffset,
    pub bytes: u64,
}

/// Where every colour lives in one tape's arena.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArenaLayout {
    /// Per colour: its value's `rows × cols × CUDA_ACT_BYTES`.
    pub colour_bytes: &'static [u64],
    /// Per colour: the sum of the [`CUDA_ARENA_ALIGN`]-aligned sizes of the colours before it.
    pub colour_offsets: &'static [u64],
    /// The sum of every colour's aligned size: the bytes this tape addresses.
    pub total_bytes: u64,
}

/// One bucket's forward.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CudaTape {
    /// The token count `M` the tape is compiled for.
    pub bucket_m: u32,
    pub arena: ArenaLayout,
    /// Colour 0: the embedded hidden, `[bucket_m, hidden]` bf16, which the forward uploads.
    pub input: ArenaSpan,
    /// The colourer's result colour, which the forward downloads.
    pub result: ArenaSpan,
    /// Per source index: the bytes its weight buffer must hold; 0 for a source that is not a
    /// dense weight.
    pub weight_bytes: &'static [u64],
    /// One step per launch, in launch order.
    pub steps: &'static [CudaStep],
}
