// SPDX-License-Identifier: Apache-2.0
//! Numeric newtypes for the Metal interpreter.
//!
//! Distinguish kinds of integers that share a primitive type but are
//! not interchangeable. Each newtype is
//! `Copy + Clone + Eq + Hash + Debug` plus a `From<inner>` impl so adding
//! one to an existing call site is a one-line wrap, not a refactor.
//!
//! The goal is compile-time prevention of bug classes already hit on
//! scratchy-target-metal; each newtype guards against one of them.

/// Per-layer index into the model's transformer stack.
///
/// Distinct from `ArenaSlotIdx` (a colored tile-arena slot), from
/// `PhysicalBlockIdx` / `LogicalBlockIdx` (paged-cache block numbers),
/// and from `SeqIdx` (per-batch sequence id). The inner width is `u32`
/// to match the existing `RuntimeBindingKind::KvCacheK { layer: u32 }`
/// arithmetic (`*layer + layer_offset` during loop unrolling).
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug, serde::Serialize)]
pub struct LayerId(pub u32);

impl From<u32> for LayerId {
    fn from(v: u32) -> Self {
        Self(v)
    }
}

impl LayerId {
    pub fn get(self) -> u32 {
        self.0
    }
}

/// Colored arena slot id (the worker's per-shape-class tile arena).
///
/// Distinct from `LayerId` and from raw kernel binding indices. The
/// post-coloring linear-scan reg allocator (`colored_slot_map()` in
/// `scratchy-forward-compiler-macro/src/interpreter_codegen.rs`) emits these.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug, serde::Serialize)]
pub struct ArenaSlotIdx(pub u32);

impl From<u32> for ArenaSlotIdx {
    fn from(v: u32) -> Self {
        Self(v)
    }
}

impl ArenaSlotIdx {
    pub fn get(self) -> u32 {
        self.0
    }
}

/// Physical block index inside the paged KV cache pool.
///
/// Distinct from `LogicalBlockIdx`: lookup goes
/// `block_table[seq][logical] → physical`. Confusing the two silently
/// reads or writes the wrong sequence's cache memory.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct PhysicalBlockIdx(pub u32);

impl From<u32> for PhysicalBlockIdx {
    fn from(v: u32) -> Self {
        Self(v)
    }
}

/// Per-sequence logical block index (0..max_blocks_per_seq).
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct LogicalBlockIdx(pub u32);

impl From<u32> for LogicalBlockIdx {
    fn from(v: u32) -> Self {
        Self(v)
    }
}

/// Token index inside one paged block (`[0, BLOCK_SIZE)`).
///
/// `u16` is plenty — block sizes are 16/32/64 in practice.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct SlotInBlock(pub u16);

impl From<u16> for SlotInBlock {
    fn from(v: u16) -> Self {
        Self(v)
    }
}

/// Per-batch sequence id (0..num_seqs in the forward batch).
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct SeqIdx(pub u32);

impl From<u32> for SeqIdx {
    fn from(v: u32) -> Self {
        Self(v)
    }
}

/// Q-token index inside a forward call (`[0, total_q)`).
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct QTokenIdx(pub u32);

impl From<u32> for QTokenIdx {
    fn from(v: u32) -> Self {
        Self(v)
    }
}

/// The static dispatch upper bound (set per bucket).
///
/// Distinct from `NumTokens` so the `ceil(baseline * n / bucket_m)`
/// axis-scaling math (see
/// `worker::scale_tg_for_num_tokens`) can't reverse its arguments.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug, serde::Serialize)]
pub struct BucketM(pub u32);

impl From<u32> for BucketM {
    fn from(v: u32) -> Self {
        Self(v)
    }
}

impl BucketM {
    pub fn get(self) -> u32 {
        self.0
    }
}

/// The actual M of the in-flight forward.
///
/// Always `<= bucket_m` of the active bucket (the bucket picker
/// guarantees this).
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct NumTokens(pub u32);

impl From<u32> for NumTokens {
    fn from(v: u32) -> Self {
        Self(v)
    }
}

impl NumTokens {
    pub fn get(self) -> u32 {
        self.0
    }
}

/// `[[buffer(N)]]` binding index on a kernel function signature.
///
/// Distinct from `ConstSlot` (function-constant index, a different
/// Metal-level concept). Distinct from `ArenaSlotIdx` (worker-arena
/// slot id used to resolve the buffer pointer).
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct BindingIdx(pub u8);

impl From<u8> for BindingIdx {
    fn from(v: u8) -> Self {
        Self(v)
    }
}

impl BindingIdx {
    pub fn get(self) -> u8 {
        self.0
    }
}

/// Function-constant slot id (`[[function_constant(N)]]`).
///
/// Re-exported from `scratchy-target-metal` so the type lives next to
/// the `ConstantValue` constructor it parameterizes.
pub use crate::tape::constants::ConstSlot;

// ── Per-dim newtypes for kernel function-constant fields ────────────
//
// Each is a transparent newtype wrapping the primitive Metal expects
// (`u32` for `[[function_constant(N)]] constant uint`,
//  `i32` for `… int`,
//  `f32` for `… float`).
//
// Phase 2's per-kernel constants structs use these so swapping
// `head_dim` and `num_q_heads` at a call site requires a visually
// obvious mistake (`HeadDim(W::NUM_Q_HEADS)`) rather than silent
// reordering of two interchangeable `u32`s.

macro_rules! u32_newtype {
    ($($(#[$m:meta])* $name:ident),* $(,)?) => {$(
        $(#[$m])*
        #[derive(Copy, Clone, Eq, PartialEq, Hash, Debug, serde::Serialize)]
        pub struct $name(pub u32);
        impl From<u32> for $name { fn from(v: u32) -> Self { Self(v) } }
        impl $name { pub fn get(self) -> u32 { self.0 } }
    )*}
}

macro_rules! i32_newtype {
    ($($(#[$m:meta])* $name:ident),* $(,)?) => {$(
        $(#[$m])*
        #[derive(Copy, Clone, Eq, PartialEq, Hash, Debug, serde::Serialize)]
        pub struct $name(pub i32);
        impl From<i32> for $name { fn from(v: i32) -> Self { Self(v) } }
        impl $name { pub fn get(self) -> i32 { self.0 } }
    )*}
}

macro_rules! f32_newtype {
    ($($(#[$m:meta])* $name:ident),* $(,)?) => {$(
        $(#[$m])*
        #[derive(Copy, Clone, PartialEq, Debug, serde::Serialize)]
        pub struct $name(pub f32);
        impl From<f32> for $name { fn from(v: f32) -> Self { Self(v) } }
        impl $name { pub fn get(self) -> f32 { self.0 } }
    )*}
}

u32_newtype!(
    /// Per-head feature width (`W::HEAD_DIM`).
    HeadDim,
    /// Number of query heads (`W::NUM_Q_HEADS`).
    NumQHeads,
    /// Number of key/value heads (`W::NUM_KV_HEADS`), pre-GQA fan-out.
    NumKvHeads,
    /// Number of `head_dim` positions touched by RoPE (`W::ROT_DIM`).
    /// Equal to `head_dim` for full RoPE, smaller for partial RoPE.
    RotDim,
    /// Element-pairing offset for the RoPE rotation: lane `d <
    /// rot_dim/2` rotates the `(d, d + pair_off)` pair. Standard NeoX
    /// (full and HF-partial) pairs within the rot window (`pair_off =
    /// rot_dim/2`); Gemma4's proportional rope pairs across the full
    /// head's halves (`pair_off = head_dim/2` — mlx `ProportionalRoPE`
    /// rotates the first rot_dim/2 lanes of EACH head half).
    RopePairOff,
    /// Tokens per paged KV-cache block (`W::BLOCK_SIZE`).
    BlockSize,
    /// Reactive (chunked) KV pool granularity — paged blocks backed by
    /// one physical chunk buffer (`crate::BLOCKS_PER_CHUNK`).
    /// The KV cache bindings are per-layer chunk-address tables (device
    /// uint64 gpuAddresses); a physical block id `pb` derefs
    /// `table[pb / BLOCKS_PER_CHUNK]` and addresses `pb % BLOCKS_PER_CHUNK`
    /// within that chunk. Same value baked into the `SynthPreAttn` MSL.
    BlocksPerChunk,
    /// Block-table fanout per sequence (`W::MAX_BLOCKS_PER_SEQ`).
    MaxBlocksPerSeq,
    /// Hidden / Q-projection size (`W::Q_SIZE` — `num_q_heads * head_dim`).
    QSize,
    /// MLP intermediate size (`W::INTERMEDIATE_SIZE`).
    IntermediateSize,
    /// Generic hidden size in elements (used by elementwise kernels —
    /// `AffineEmbed.hidden_size`, `GatherLastToken.row_stride`).
    HiddenSize,
    /// MLX-affine packed-K (input cols) for a qmv / qmm_t dispatch.
    KDim,
    /// MLX-affine output cols / weight rows (qmv / qmm_t `n_v`).
    NDim,
    /// Split-K partition count (for `affine_qmm_t_splitk`).
    SplitK,
    /// Steel attention `[[function_constant(99)]]` debug-mode toggle.
    /// Bound to `0` for production; `>0` selects diagnostic paths.
    AttnDebugMode,
    /// TurboQuant codebook width, bits per packed code
    /// (`tq_bits(W::TQ_KV_BITS)`).
    TqCodeBits,
    /// Query heads one TurboQuant decode threadgroup covers
    /// (`attention.metal` `ATTN_TQ_HEADS`, slot 16) — see [`TqDecodeHeads::for_group`].
    TqDecodeHeads,
);

impl TqDecodeHeads {
    /// Every head count the decode kernel serves for this geometry, ascending:
    /// the divisors of the GQA group whose per-lane state fits
    /// (`heads * head_dim / 32 <= 32`, at most 8 heads).
    pub fn candidates(
        head_dim: HeadDim,
        num_q_heads: NumQHeads,
        num_kv_heads: NumKvHeads,
    ) -> impl Iterator<Item = Self> {
        let group = num_q_heads.get() / num_kv_heads.get().max(1);
        let lanes = head_dim.get() / 32;
        (1..=group.min(8))
            .filter(move |&h| group.is_multiple_of(h) && h * lanes <= 32)
            .map(Self)
    }

    /// The most heads that still leave `4/5` of the GPU's cores a threadgroup
    /// each (one per `heads` query heads). The threadgroup decodes each packed
    /// key once for all of them, and TurboQuant decode is ALU-bound on that
    /// decode — but too few threadgroups idle cores. Measured on a 10-core M5
    /// (Llama-3.2-3B, Llama-3.1-8B, Qwen2.5-3B / -7B geometries at 8k): the
    /// best count left 8 threadgroups in every case.
    pub fn for_group(
        head_dim: HeadDim,
        num_q_heads: NumQHeads,
        num_kv_heads: NumKvHeads,
        gpu_cores: u32,
    ) -> Self {
        let min_threadgroups = gpu_cores * 4 / 5;
        Self::candidates(head_dim, num_q_heads, num_kv_heads)
            .filter(|h| num_q_heads.get() / h.get() >= min_threadgroups)
            .last()
            .unwrap_or(Self(1))
    }
}

i32_newtype!(
    /// Sliding-window width in tokens (`W::SLIDING_WINDOW`) for the
    /// `attention.metal` kernels' `ATTN_WINDOW [[function_constant(7)]]`.
    /// `0` disables the window (full attention); declared as signed
    /// `int` to match the MSL declaration.
    AttnWindow,
    /// MLX-affine `[[function_constant]] constant int` packed-K size.
    /// Same semantic as `KDim`, but the qmv / qmm_t MLX-port shaders
    /// declare the constant as signed `int`.
    KDimI32,
    /// MLX-affine `[[function_constant]] constant int` output cols.
    NDimI32,
    /// MLX-affine `[[function_constant]] constant int` per-call M (=
    /// bucket_m). Distinct from [`BucketM`] because the qmm_t shader
    /// declares the constant as `int`.
    MDimI32,
    /// MLX-affine `[[function_constant]] constant int` split-K
    /// partition stride (`affine_qmm_t_splitk` only).
    KPartitionSizeI32,
);

f32_newtype!(
    /// Pre-softmax scale applied per Q*K dot product
    /// (`W::ATTN_SCALE`, typically `1/sqrt(head_dim)`).
    AttnScale,
    /// RMSNorm epsilon (`W::RMS_NORM_EPS`).
    RmsNormEps,
);
