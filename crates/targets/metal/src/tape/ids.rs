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

/// The rows one step samples (its `SampleIndices`' length): the count the chain step's trait
/// carries ([`scratchy_forward_compiler::ChainStepHandle`]), read by a runtime gate and a
/// [`crate::tape::lowered::SetAxis`].
pub use scratchy_forward_compiler::SampledRowCount;

macro_rules! u32_newtype {
    ($($(#[$m:meta])* $name:ident),* $(,)?) => {$(
        $(#[$m])*
        #[derive(Copy, Clone, Eq, PartialEq, Hash, Debug, serde::Serialize)]
        pub struct $name(pub u32);
        impl From<u32> for $name { fn from(v: u32) -> Self { Self(v) } }
        impl $name { pub fn get(self) -> u32 { self.0 } }
    )*}
}

/// A command's position in its tape's command table ([`crate::tape::lowered::TapeCommands`]):
/// two bytes per command a tape runs, where a reference would be eight plus a load-time fixup. A
/// table too large for it is refused ([`Self::of`]), never truncated.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug, serde::Serialize)]
pub struct CommandIx(pub u16);

impl CommandIx {
    /// The index of the command at `position` of its table; `None` past the last index.
    pub fn of(position: usize) -> Option<Self> {
        u16::try_from(position).ok().map(Self)
    }

    pub fn get(self) -> usize {
        usize::from(self.0)
    }
}

/// A position in the command table's part array of `T`s ([`PartArray`] of
/// [`crate::tape::lowered::CommandTable`]): two bytes per command field, where the field itself
/// would be up to thirty-two plus load-time fixups. A table holds no more distinct parts of one
/// kind than distinct commands, so whatever a [`CommandIx`] indexes, a part index does too.
///
/// The array's element type is the index's: a constant set's index reads no binding list.
///
/// ```compile_fail,E0308
/// # use scratchy_target_metal::tape::ids::{PartArray, PartIx};
/// static SYMBOLS: [(&str, &str); 1] = [("lib", "fn")];
/// static GATES: [Option<u8>; 1] = [None];
/// let gate: PartIx<Option<u8>> = PartIx::at(0);
/// let _ = PartArray::of(&SYMBOLS).get(gate);
/// ```
pub struct PartIx<T>(u16, std::marker::PhantomData<fn() -> T>);

impl<T> PartIx<T> {
    /// The index of the part at `position` of its array; `None` past the last index.
    pub fn of(position: usize) -> Option<Self> {
        u16::try_from(position).ok().map(Self::at)
    }

    /// The part at `position`, as a table's literal names it.
    pub const fn at(position: u16) -> Self {
        Self(position, std::marker::PhantomData)
    }

    pub fn get(self) -> usize {
        usize::from(self.0)
    }

    /// Its position, as a table's literal names it.
    pub fn raw(self) -> u16 {
        self.0
    }
}

impl<T> Copy for PartIx<T> {}

impl<T> Clone for PartIx<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> PartialEq for PartIx<T> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<T> Eq for PartIx<T> {}

impl<T> std::hash::Hash for PartIx<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.hash(state)
    }
}

impl<T> std::fmt::Debug for PartIx<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PartIx({})", self.0)
    }
}

impl<T> serde::Serialize for PartIx<T> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_u16(self.0)
    }
}

/// A command table's array of `T`s, read only at a [`PartIx<T>`].
pub struct PartArray<T: 'static>(pub &'static [T]);

impl<T> PartArray<T> {
    pub const fn of(parts: &'static [T]) -> Self {
        Self(parts)
    }

    /// The part at `ix`.
    pub fn get(&self, ix: PartIx<T>) -> &'static T {
        &self.0[ix.get()]
    }

    /// Every part, in order.
    pub fn all(&self) -> &'static [T] {
        self.0
    }
}

impl<T> Copy for PartArray<T> {}

impl<T> Clone for PartArray<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: serde::Serialize> serde::Serialize for PartArray<T> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_newtype_struct("PartArray", &self.0)
    }
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

/// Whether `table` holds `v`: a kernel's instantiation table as a `const fn` law.
const fn instantiated(table: &[u32], v: u32) -> bool {
    let mut i = 0;
    while i < table.len() {
        if table[i] == v {
            return true;
        }
        i += 1;
    }
    false
}

pub use super::typed::{AFFINE_BITS, AFFINE_GROUP_SIZES, INT4_BITS, NAX_GROUP_SIZES};

/// A value an instantiated kernel takes, refined by the table of the values it is instantiated
/// for — each table the marker types of [`super::typed`] its kernels are instantiated for: the
/// table is the type's constructor, so no command is built with a value its kernel lacks, and
/// nothing downstream re-checks it.
macro_rules! refined_u32 {
    ($($(#[$m:meta])* $name:ident: $table:ident;)*) => {$(
        $(#[$m])*
        #[derive(Copy, Clone, Eq, PartialEq, Hash, Debug, serde::Serialize)]
        pub struct $name(u32);
        impl $name {
            #[doc = concat!("`v` as a [`", stringify!($name), "`]; `None` for a value no kernel is instantiated for.")]
            pub const fn new(v: u32) -> Option<Self> {
                match instantiated($table, v) {
                    true => Some(Self(v)),
                    false => None,
                }
            }
            pub const fn get(self) -> u32 {
                self.0
            }
            #[doc = concat!("`v` as a [`", stringify!($name), "`], for a const context: a value no kernel is instantiated for fails the build there.")]
            pub const fn of(v: u32) -> Self {
                match Self::new(v) {
                    Some(x) => x,
                    None => panic!(concat!("no kernel is instantiated for this ", stringify!($name))),
                }
            }
        }
    )*};
}

refined_u32! {
    /// MLX-affine bits per quantized weight: the widths the affine kernels are instantiated for
    /// (`_b_{2,3,4,8}` in `quantized*.metal`; the packing stride is `32 / bits` elements a word).
    AffineBits: AFFINE_BITS;
    /// MLX-affine quantization group size: the groups the affine kernels are instantiated for
    /// (`_gs_{32,64,128}`).
    AffineGroupSize: AFFINE_GROUP_SIZES;
    /// The group size of an MLX-affine kernel on the matrix unit (`quantized_qmm_nax`: the W4A8
    /// GEMMs, the NAX gather GEMM, a planes layout): those are instantiated for 64 and 128 only.
    NaxGroupSize: NAX_GROUP_SIZES;
    /// The width of an MLX-affine kernel instantiated for int4 codes alone (`qmm_t_splitk`, `qmm_n`,
    /// `qvm`): 4.
    Int4Bits: INT4_BITS;
}

impl AffineBits {
    /// Four bits a weight: the width the int4-only kernels (the small-M tiles, W4A8, split-K) take.
    pub const B4: Self = Self(4);

    /// The width as the int4-only kernels take it; `None` for any other.
    pub const fn int4(self) -> Option<Int4Bits> {
        Int4Bits::new(self.0)
    }
}

impl Int4Bits {
    /// The width as every affine kernel takes it.
    pub const fn affine(self) -> AffineBits {
        AffineBits(self.0)
    }
}

impl AffineGroupSize {
    /// The group as the matrix-unit kernels take it; `None` for a group they lack.
    pub const fn nax(self) -> Option<NaxGroupSize> {
        NaxGroupSize::new(self.0)
    }
}

/// A head width `RopeAppendNormed` stages whole in threadgroup memory: at most
/// [`Self::MAX`], the staging its kernel is sized for.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug, serde::Serialize)]
pub struct StagedHeadDim(u32);

impl StagedHeadDim {
    /// The widest head the kernel's threadgroup staging holds.
    pub const MAX: u32 = 512;

    /// `head_dim` as a staged width; `None` past [`Self::MAX`] (or zero).
    pub const fn new(head_dim: u32) -> Option<Self> {
        match head_dim > 0 && head_dim <= Self::MAX {
            true => Some(Self(head_dim)),
            false => None,
        }
    }

    pub const fn get(self) -> u32 {
        self.0
    }

    /// The width as the kernels' head-dim constant.
    pub const fn head_dim(self) -> HeadDim {
        HeadDim(self.0)
    }
}

/// The token block a GDN conv thread owns (`gdn_conv1d_varlen`'s baked constant slot 6): the
/// grid's Z is the step's rows over it, each thread walking its block with the causal window in
/// registers. Minted against the model it runs: it covers the seed window (`kernel - 1` entries)
/// and a verify step's rows (`drafts + 1`), so block 0 is the one thread that reads the seed ring
/// and the one that writes the new ring, blocks past a sequence's head seed from `x` alone, and
/// only block 0 touches the draft lanes.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug, serde::Serialize)]
pub struct GdnConvBlock(u32);

impl GdnConvBlock {
    /// The tokens a thread owns: the kernel's one instantiation.
    pub const TOKENS: u32 = 32;

    /// The block for a conv of `kernel` taps under `drafts` drafted tokens a step; `None` when
    /// the block covers either too few.
    pub const fn covering(kernel: u32, drafts: u32) -> Option<Self> {
        match kernel <= Self::TOKENS + 1 && drafts < Self::TOKENS {
            true => Some(Self(Self::TOKENS)),
            false => None,
        }
    }

    pub const fn get(self) -> u32 {
        self.0
    }
}

u32_newtype!(
    /// A model tensor family (a `(WeightKind, accessor)` bundle, every layer) in the model's
    /// source manifest — what a `Binding::Source` names; the macro's generated resolver maps it.
    SourceIx,
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
    /// within that chunk.
    BlocksPerChunk,
    /// Block-table fanout per sequence: the KV cap rung a tape is baked for.
    #[derive(PartialOrd, Ord)]
    MaxBlocksPerSeq,
    /// The positions a model attends over (`max_position_embeddings`): the top KV cap rung.
    MaxPositions,
    /// Hidden / Q-projection size (`W::Q_SIZE` — `num_q_heads * head_dim`).
    QSize,
    /// MLP intermediate size (`W::INTERMEDIATE_SIZE`).
    IntermediateSize,
    /// Generic hidden size in elements (used by elementwise kernels —
    /// `AffineEmbed.hidden_size`).
    HiddenSize,
    /// Elements between one row of a buffer and the next, that a row-moving kernel swaps whole:
    /// the hidden size (the lm_head's input) or the vocabulary (its logits).
    RowStride,
    /// MLX-affine packed-K (input cols) for a qmv / qmm_t dispatch.
    KDim,
    /// MLX-affine output cols / weight rows (qmv / qmm_t `n_v`).
    NDim,
    /// Split-K partition count (for `affine_qmm_t_splitk`).
    SplitK,
    /// An MLX-affine GEMM's matvec/matmul boundary (`get_qmv_batch_limit`).
    QmvBatchLimit,
    /// Routed experts of a MoE layer.
    NumExperts,
    /// Experts each token routes to.
    TopK,
    /// Rows a per-row op runs per token (a per-head norm's head count; 1 on the
    /// residual stream).
    RowsPerToken,
    /// Elements an elementwise kernel's buffer holds: its threads past it write nothing.
    ElementCount,
    /// A reshape's row divisor (`num_tokens / d` rows — the vision merger's factor).
    RowsDivisor,
    /// Iterations of a rolled loop.
    LoopIters,
    /// Rows in a rolled loop's body.
    BodyLen,
    /// Layers one iteration of a rolled loop advances.
    LayerStride,
    /// Steel attention `[[function_constant(99)]]` debug-mode toggle.
    /// Bound to `0` for production; `>0` selects diagnostic paths.
    AttnDebugMode,
    /// TurboQuant codebook width, bits per packed code (`W::KV_CODEC`'s `TqBits`).
    TqCodeBits,
    /// Query heads one TurboQuant decode threadgroup covers
    /// (`attention.metal` `ATTN_TQ_HEADS`, slot 16) — see [`TqDecodeHeads::for_group`].
    TqDecodeHeads,
    /// The GPU cores of the device a tape runs on, read at load
    /// ([`crate::device::gpu_cores`]). No baked profile can stand in for it:
    /// one chip name ships with several core counts (an M1 Max has 24 or 32).
    GpuCores,
    /// The most rows a step samples, and a decode or verify step brings — one a sequence of the
    /// serving engine's `max_num_seqs`, a verify step's drafts too — read at load: the rows a
    /// decode-gated TurboQuant decode split's partials are reserved for
    /// ([`crate::tape::lowered::LoweredMetalTape::scratch_bytes`]), and the logits buffer holds
    /// ([`crate::tape::lowered::LoweredMetalTape::logits_rows`]).
    DecodeRows,
    /// Logits per row a model's forward leaves (`METAL_VOCAB_SIZE`): the width the off-tape
    /// kernels (argmax, grammar mask, sampler) read ([`crate::off_tape`]).
    LogitsWidth,
    /// `u32` words per row of a grammar allow-bitset (one bit per token).
    BitsetWords,
    /// Drafts a multi-token-prediction head makes a speculative step (its `spec_drafts`).
    NumDrafts,
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
        gpu_cores: GpuCores,
    ) -> Self {
        let min_threadgroups = gpu_cores.get() * 4 / 5;
        Self::candidates(head_dim, num_q_heads, num_kv_heads)
            .filter(|h| num_q_heads.get() / h.get() >= min_threadgroups)
            .last()
            .unwrap_or(Self(1))
    }

    /// The most heads a threadgroup serves for this geometry, baked for the
    /// plain (non-TurboQuant) decode: the largest divisor of the GQA group
    /// that fits the per-lane state. Unlike [`Self::for_group`] there is no
    /// occupancy floor — that filter is tuned for TurboQuant's ALU-bound
    /// decode, while the plain walk is bound by the K/V reads, which the
    /// largest group reads the fewest times over.
    pub fn largest_group(
        head_dim: HeadDim,
        num_q_heads: NumQHeads,
        num_kv_heads: NumKvHeads,
    ) -> Self {
        Self::candidates(head_dim, num_q_heads, num_kv_heads)
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

#[cfg(test)]
mod refined_tests {
    use super::*;

    /// Every `u32` a refined constructor could be handed, in the kernels' range and past it: each
    /// value up to twice the widest instantiation, and the type's edges.
    fn candidates(widest: u32) -> impl Iterator<Item = u32> {
        (0..=2 * widest + 1).chain([u32::MAX - 1, u32::MAX])
    }

    /// The values a constructor accepts among `candidates`.
    fn accepted(widest: u32, new: impl Fn(u32) -> Option<u32>) -> Vec<u32> {
        candidates(widest).filter(|&v| new(v).is_some()).collect()
    }

    /// `AffineBits` is minted for exactly the widths the affine kernels are instantiated for, and
    /// refuses every other.
    #[test]
    fn affine_bits_refuses_each_width_no_kernel_is_instantiated_for() {
        assert_eq!(AFFINE_BITS, &[2, 3, 4, 8]);
        assert_eq!(
            accepted(8, |v| AffineBits::new(v).map(AffineBits::get)),
            AFFINE_BITS
        );
        assert_eq!(AffineBits::B4.get(), 4);
    }

    /// `Int4Bits` takes 4 alone: every other affine width is one the int4-only kernels lack,
    /// refused by the constructor and by [`AffineBits::int4`].
    #[test]
    fn int4_bits_refuses_each_width_the_int4_kernels_lack() {
        assert_eq!(INT4_BITS, &[4]);
        assert_eq!(
            accepted(8, |v| Int4Bits::new(v).map(Int4Bits::get)),
            INT4_BITS
        );
        for &bits in AFFINE_BITS {
            let int4 = AffineBits::new(bits).and_then(AffineBits::int4);
            assert_eq!(
                int4.map(Int4Bits::get),
                (bits == 4).then_some(4),
                "{bits} bits"
            );
        }
        assert_eq!(Int4Bits::of(4).affine(), AffineBits::B4);
    }

    /// `of`, the const-context constructor, mints what `new` does and refuses what it refuses.
    #[test]
    fn of_mints_what_new_mints_and_refuses_the_rest() {
        assert_eq!(AffineBits::of(8), AffineBits::new(8).unwrap());
        assert!(std::panic::catch_unwind(|| AffineBits::of(5)).is_err());
        assert!(std::panic::catch_unwind(|| AffineGroupSize::of(16)).is_err());
    }

    /// `AffineGroupSize` is minted for exactly the groups the affine kernels are instantiated for.
    #[test]
    fn affine_group_size_refuses_each_group_no_kernel_is_instantiated_for() {
        assert_eq!(AFFINE_GROUP_SIZES, &[32, 64, 128]);
        assert_eq!(
            accepted(128, |v| AffineGroupSize::new(v).map(AffineGroupSize::get)),
            AFFINE_GROUP_SIZES
        );
    }

    /// `NaxGroupSize` takes 64 and 128 alone: a group of 32 is an affine group the matrix-unit
    /// kernels lack, refused by the constructor and by [`AffineGroupSize::nax`].
    #[test]
    fn nax_group_size_refuses_each_group_the_matrix_unit_kernels_lack() {
        assert_eq!(NAX_GROUP_SIZES, &[64, 128]);
        assert_eq!(
            accepted(128, |v| NaxGroupSize::new(v).map(NaxGroupSize::get)),
            NAX_GROUP_SIZES
        );
        let nax = |g| AffineGroupSize::new(g).and_then(AffineGroupSize::nax);
        assert_eq!(nax(32), None);
        assert_eq!(nax(64).map(NaxGroupSize::get), Some(64));
        assert_eq!(nax(128).map(NaxGroupSize::get), Some(128));
    }

    /// `StagedHeadDim` holds every head width up to the staging's 512 and refuses each wider one
    /// (and a zero-wide head).
    #[test]
    fn staged_head_dim_refuses_each_width_past_the_staging() {
        let held = accepted(StagedHeadDim::MAX, |v| {
            StagedHeadDim::new(v).map(StagedHeadDim::get)
        });
        assert_eq!(held, (1..=512).collect::<Vec<_>>());
        assert_eq!(
            StagedHeadDim::new(256).map(|h| h.head_dim()),
            Some(HeadDim(256))
        );
    }

    /// `GdnConvBlock` covers a conv of up to 33 taps (its 32-token block holds the 32-entry seed
    /// window) and up to 31 drafted tokens a step (32 verify rows), and refuses each conv or
    /// draft count past either.
    #[test]
    fn gdn_conv_block_refuses_each_conv_or_draft_count_it_does_not_cover() {
        let block = GdnConvBlock::TOKENS;
        for kernel in candidates(block + 1) {
            for drafts in candidates(block) {
                let covers = kernel <= block + 1 && drafts < block;
                assert_eq!(
                    GdnConvBlock::covering(kernel, drafts).is_some(),
                    covers,
                    "kernel {kernel}, drafts {drafts}"
                );
            }
        }
        assert_eq!(
            GdnConvBlock::covering(4, 3).map(GdnConvBlock::get),
            Some(32)
        );
    }
}
