// SPDX-License-Identifier: Apache-2.0
//! Typed per-kernel function-constants structs.
//!
//! Each kernel family that takes `[[function_constant(N)]]` slots gets
//! one struct. Constructing the struct lists every slot the kernel
//! reads — so adding a slot to a `.metal` source without adding a
//! field here, or vice versa, breaks every call site at compile time.
//!
//! Catches bug class #1 (the `ATTN_PAGED_DEBUG_MODE` slot-99 omission
//! that turned production output into `" pr formal formal ..."`,
//! fixed in commit `b3ddb3b46`).
//!
//! Convention: each struct provides
//! `impl From<Self> for Vec<ConstantValue>` that emits the slots in
//! their declared order — same order as the matching `.metal` header.
//! Lowering arms construct the struct, then `.into()` for assignment
//! to `LoweredCommand::constants`.

use crate::tape::constants::{ConstSlot, ConstantValue};

use crate::tape::ids::{
    AttnDebugMode, AttnScale, AttnWindow, BlockSize, BlocksPerChunk, BucketM, DecodeHeads,
    ElementCount, HeadDim, HiddenSize, IntermediateSize, KDim, KDimI32, KPartitionSizeI32, MDimI32,
    NDim, NDimI32, NumExperts, NumKvHeads, NumQHeads, QSize, RmsNormEps, RopePairOff, RotDim,
    RowStride, SplitK, StagedHeadDim, TopK,
};
use crate::tape::lowered::ActivationWidth;

/// Append the spans rope-on-read function constants (slot 8 = rotary
/// dim, slot 9 = NeoX pairing offset, slot 10 = 0/1 master switch) when
/// present. Shared by the attention + rope_append constant builders. When
/// the fields are `None` (every non-spans dispatch) nothing is pushed, so
/// the emitted `Vec<ConstantValue>` — and thus the pipeline cache key —
/// is byte-identical to the pre-spans path. The kernels read these via
/// `is_function_constant_defined`, so omitting them dead-eliminates the
/// in-shader rotation entirely.
fn push_rope_on_read_consts(
    v: &mut Vec<ConstantValue>,
    rot_dim: Option<RotDim>,
    pair_off: Option<RopePairOff>,
    rope_on_read: Option<u32>,
) {
    if let Some(rd) = rot_dim {
        v.push(ConstantValue::uint(ConstSlot(8), rd.get()));
    }
    if let Some(po) = pair_off {
        v.push(ConstantValue::uint(ConstSlot(9), po.get()));
    }
    if let Some(ror) = rope_on_read {
        v.push(ConstantValue::uint(ConstSlot(10), ror));
    }
}

// ── Embed (token gather) ───────────────────────────────────────────

/// `KernelId::Embed` (`embed_<dtype>_specialized`).
pub struct EmbedConstants {
    pub bucket_m: BucketM,
    pub q_size: QSize,
}

impl From<EmbedConstants> for Vec<ConstantValue> {
    fn from(c: EmbedConstants) -> Self {
        vec![
            ConstantValue::uint(ConstSlot(0), c.bucket_m.get()),
            ConstantValue::uint(ConstSlot(1), c.q_size.get()),
        ]
    }
}

// ── RmsNorm / FusedAddRmsNorm ──────────────────────────────────────

/// `KernelId::RmsNorm` (`rmsnorm_<T_act>_s_<T_scale>_specialized`)
/// and `KernelId::FusedAddRmsNorm` (`fused_add_rmsnorm_<...>`).
/// Constants are identical across the two.
pub struct RmsNormConstants {
    pub bucket_m: BucketM,
    pub q_size: QSize,
    pub rms_norm_eps: RmsNormEps,
    /// Zero-centered (Gemma / Qwen3.5) gain offset: effective gain =
    /// `weight + weight_offset`. `1.0` for `(1 + weight)` arches, `0.0`
    /// for plain RMSNorm. Function constant 3 in `rmsnorm.metal` /
    /// `fused_add_rmsnorm.metal`.
    pub weight_offset: f32,
}

/// The most threads an RMSNorm row's threadgroup runs: a row program's, and a norm's over rows at
/// least this wide.
pub const NORM_THREADS: u32 = 1024;

/// The threads of an RMSNorm row's threadgroup over `width`-wide rows: the power of two covering a
/// row, up to [`NORM_THREADS`]. Its kernel takes them compiled in (slot 4), so each thread's share
/// of the row is a count the compiler knows. Over a row no wider than them, any covering count sums
/// in the same order (a thread holds at most one element; `row_sum`'s extra levels add zeros): a
/// per-head norm's 256-wide rows run 256 threads, the same bits as 1024.
pub fn norm_threads(width: QSize) -> u32 {
    width.get().next_power_of_two().clamp(32, NORM_THREADS)
}

impl From<RmsNormConstants> for Vec<ConstantValue> {
    fn from(c: RmsNormConstants) -> Self {
        vec![
            ConstantValue::uint(ConstSlot(0), c.bucket_m.get()),
            ConstantValue::uint(ConstSlot(1), c.q_size.get()),
            ConstantValue::float(ConstSlot(2), c.rms_norm_eps.get()),
            ConstantValue::float(ConstSlot(3), c.weight_offset),
            ConstantValue::uint(ConstSlot(4), norm_threads(c.q_size)),
        ]
    }
}

// ── ScalarMul ──────────────────────────────────────────────────────

/// `KernelId::ScalarMul` (`scalar_mul_<T>_specialized`, `elementwise.metal`): the scale (slot 2;
/// slots 0 / 1 are the file's other kernels') and the elements the buffer holds (slot 3) — its
/// dispatch rounds up to whole threadgroups, and a thread past the buffer would scale whatever
/// lies beyond it.
pub struct ScalarMulConstants {
    pub scale: f32,
    pub elements: ElementCount,
}

impl From<ScalarMulConstants> for Vec<ConstantValue> {
    fn from(c: ScalarMulConstants) -> Self {
        vec![
            ConstantValue::float(ConstSlot(2), c.scale),
            ConstantValue::uint(ConstSlot(3), c.elements.get()),
        ]
    }
}

/// `KernelId::LoadRows` (`copy_rows_<T>`, `elementwise.metal` slot 4): the elements the
/// staged buffer holds.
pub struct CopyRowsConstants {
    pub elements: ElementCount,
}

impl From<CopyRowsConstants> for Vec<ConstantValue> {
    fn from(c: CopyRowsConstants) -> Self {
        vec![ConstantValue::uint(ConstSlot(4), c.elements.get())]
    }
}

/// `KernelId::ConcatRows` (`concat_rows_<T>`, `elementwise.metal` slots 6 and 7): the elements
/// the output holds and the width of each operand's rows.
pub struct ConcatRowsConstants {
    pub elements: ElementCount,
    pub width: ActivationWidth,
}

impl From<ConcatRowsConstants> for Vec<ConstantValue> {
    fn from(c: ConcatRowsConstants) -> Self {
        vec![
            ConstantValue::uint(ConstSlot(6), c.elements.get()),
            ConstantValue::uint(ConstSlot(7), c.width.get()),
        ]
    }
}

/// `KernelId::MmEmbedSplice` (`mm_embed_splice_<T>`, `elementwise.metal` slot 5): the width of
/// the embedding rows it scatters.
pub struct MmEmbedSpliceConstants {
    pub hidden: HiddenSize,
}

impl From<MmEmbedSpliceConstants> for Vec<ConstantValue> {
    fn from(c: MmEmbedSpliceConstants) -> Self {
        vec![ConstantValue::uint(ConstSlot(5), c.hidden.get())]
    }
}

/// `KernelId::VisionGelu` (`gelu_tanh_<T>` / `gelu_erf_<T>` / `quick_gelu_<T>`,
/// `activation.metal`): the elements the buffer holds.
pub struct GeluConstants {
    pub elements: ElementCount,
}

impl From<GeluConstants> for Vec<ConstantValue> {
    fn from(c: GeluConstants) -> Self {
        vec![ConstantValue::uint(ConstSlot(0), c.elements.get())]
    }
}

/// `KernelId::EmbeddingGather` (`embedding_gather_rows_<T>`): the elements the output holds and
/// the width of the rows it permutes.
pub struct EmbeddingGatherConstants {
    pub elements: ElementCount,
    pub width: ActivationWidth,
}

impl From<EmbeddingGatherConstants> for Vec<ConstantValue> {
    fn from(c: EmbeddingGatherConstants) -> Self {
        vec![
            ConstantValue::uint(ConstSlot(0), c.elements.get()),
            ConstantValue::uint(ConstSlot(1), c.width.get()),
        ]
    }
}

// ── RopeAppendNormed ───────────────────────────────────────────────

/// `KernelId::RopeAppendNormed` (`rope_append_normed_<act>_s_<scale>_
/// specialized`). Slots 0..6 identical to [`RopeAppendConstants`];
/// 7 = norm eps, 8 = norm gain offset (Gemma4 full gains -> 0.0).
pub struct RopeAppendNormedConstants {
    /// Staged whole in threadgroup memory: no wider than the staging holds.
    pub head_dim: StagedHeadDim,
    pub num_q_heads: NumQHeads,
    pub num_kv_heads: NumKvHeads,
    pub rot_dim: RotDim,
    pub block_size: BlockSize,
    pub blocks_per_chunk: BlocksPerChunk,
    pub pair_off: RopePairOff,
    pub rms_norm_eps: RmsNormEps,
    pub weight_offset: f32,
    /// Spans rope-on-read (slot 9): when `Some(1)`, K-rotation is skipped
    /// for blocks flagged unrotated (store normed-but-unrotated). `None`
    /// → byte-identical emitted Vec (rope.metal folds the gate away).
    pub rope_on_read: Option<u32>,
    /// The int8 encode folded in (slot 10), if the writer writes the int8 store.
    pub int8: Option<KvInt8Constants>,
}

/// A KV writer's folded-in int8 encode (`rope.metal`'s `ROPE_KV_INT8`, `kv_append.metal`'s
/// `KVA_KV_INT8`, slot 10): the writer also encodes its rows into the layer's int8 store.
#[derive(Clone, Copy)]
pub struct KvInt8Constants;

impl KvInt8Constants {
    fn push(int8: Option<Self>, v: &mut Vec<ConstantValue>) {
        v.extend(int8.map(|_| ConstantValue::uint(ConstSlot(10), 1)));
    }
}

impl From<RopeAppendNormedConstants> for Vec<ConstantValue> {
    fn from(c: RopeAppendNormedConstants) -> Self {
        let mut v = vec![
            ConstantValue::uint(ConstSlot(0), c.head_dim.get()),
            ConstantValue::uint(ConstSlot(1), c.num_q_heads.get()),
            ConstantValue::uint(ConstSlot(2), c.num_kv_heads.get()),
            ConstantValue::uint(ConstSlot(3), c.rot_dim.get()),
            ConstantValue::uint(ConstSlot(4), c.block_size.get()),
            ConstantValue::uint(ConstSlot(5), c.blocks_per_chunk.get()),
            ConstantValue::uint(ConstSlot(6), c.pair_off.get()),
            ConstantValue::float(ConstSlot(7), c.rms_norm_eps.get()),
            ConstantValue::float(ConstSlot(8), c.weight_offset),
        ];
        if let Some(ror) = c.rope_on_read {
            v.push(ConstantValue::uint(ConstSlot(9), ror));
        }
        KvInt8Constants::push(c.int8, &mut v);
        v
    }
}

// ── RopeAppend ─────────────────────────────────────────────────────

/// `KernelId::RopeAppend` (`rope_append_<dtype>_specialized`).
pub struct RopeAppendConstants {
    pub head_dim: HeadDim,
    pub num_q_heads: NumQHeads,
    pub num_kv_heads: NumKvHeads,
    pub rot_dim: RotDim,
    pub block_size: BlockSize,
    pub blocks_per_chunk: BlocksPerChunk,
    /// Rotation pairing offset (see [`RopePairOff`]): `rot_dim/2` for
    /// standard NeoX; `head_dim/2` for Gemma4's proportional rope.
    pub pair_off: RopePairOff,
    /// Spans rope-on-read (slot 9): see [`RopeAppendNormedConstants`].
    pub rope_on_read: Option<u32>,
    /// The int8 encode folded in: see [`RopeAppendNormedConstants`].
    pub int8: Option<KvInt8Constants>,
    /// The query alone (slot 15, `ROPE_Q_ONLY`): no K/V rotation or cache write.
    pub query_only: bool,
}

impl From<RopeAppendConstants> for Vec<ConstantValue> {
    fn from(c: RopeAppendConstants) -> Self {
        let mut v = vec![
            ConstantValue::uint(ConstSlot(0), c.head_dim.get()),
            ConstantValue::uint(ConstSlot(1), c.num_q_heads.get()),
            ConstantValue::uint(ConstSlot(2), c.num_kv_heads.get()),
            ConstantValue::uint(ConstSlot(3), c.rot_dim.get()),
            ConstantValue::uint(ConstSlot(4), c.block_size.get()),
            ConstantValue::uint(ConstSlot(5), c.blocks_per_chunk.get()),
            ConstantValue::uint(ConstSlot(6), c.pair_off.get()),
        ];
        if let Some(ror) = c.rope_on_read {
            v.push(ConstantValue::uint(ConstSlot(9), ror));
        }
        KvInt8Constants::push(c.int8, &mut v);
        if c.query_only {
            v.push(ConstantValue::uint(ConstSlot(15), 1));
        }
        v
    }
}

/// `KernelId::KvAppend` (`kv_append_<dtype>`, `kv_append.metal`).
pub struct KvAppendConstants {
    pub head_dim: HeadDim,
    pub num_kv_heads: NumKvHeads,
    pub block_size: BlockSize,
    pub blocks_per_chunk: BlocksPerChunk,
    /// The folded-in int8 encode, at the rope writer's slot 10.
    pub int8: Option<KvInt8Constants>,
}

impl From<KvAppendConstants> for Vec<ConstantValue> {
    fn from(c: KvAppendConstants) -> Self {
        let mut v = vec![
            ConstantValue::uint(ConstSlot(0), c.head_dim.get()),
            ConstantValue::uint(ConstSlot(1), c.num_kv_heads.get()),
            ConstantValue::uint(ConstSlot(2), c.block_size.get()),
            ConstantValue::uint(ConstSlot(3), c.blocks_per_chunk.get()),
        ];
        KvInt8Constants::push(c.int8, &mut v);
        v
    }
}

// ── AttentionViaCache (decode) ─────────────────────────────────────

/// `KernelId::AttentionViaCache`
/// (`attention_via_cache_v2_<dtype>_specialized`).
pub struct AttentionViaCacheConstants {
    pub cache: DecodeCache,
    /// gpt-oss attention sinks (slot 21, `ATTN_SINKS`): the layer's per-head sink logits are an
    /// extra softmax column, bound at buffer 16.
    pub sinks: SinkColumn,
    /// Query heads one threadgroup serves (`ATTN_DECODE_HEADS`, slot 16): the largest divisor of
    /// the GQA group that fits the per-lane state, so the walk reads each key once for the whole
    /// group. The dispatch counts `num_q_heads / heads` threadgroups.
    pub heads: DecodeHeads,
    /// What a threadgroup row queries (slot 22, `ATTN_ROW_QUERIES`).
    pub query_rows: QueryRows,
}

impl AttentionViaCacheConstants {
    /// `ATTN_SINKS` (slot 21): the sink column's switch ([`Self::sinks`]).
    pub const SINKS: ConstSlot = ConstSlot(21);
    /// `ATTN_DECODE_HEADS` (slot 16): the query heads one threadgroup serves ([`Self::heads`]).
    pub const HEADS: ConstSlot = ConstSlot(16);
    /// `ATTN_KV_INT8` (slot 13): the command reads the layer's int8 store.
    pub const INT8: ConstSlot = ConstSlot(13);
}

/// What the decode attention's threadgroup rows query.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QueryRows {
    /// One row per sequence: a decode bucket's single token.
    PerSequence,
    /// One row per token, under its sequence's keys up to its own position: a bucket of several
    /// rows, whose steps run several sequences' tokens (decode), or a few of one sequence's (a
    /// verify step's last token and drafts, a short chunk) — every row attends as decoding would.
    PerToken,
}

impl QueryRows {
    /// A bucket of `bucket_m` rows' rows.
    pub fn for_bucket(bucket_m: u32) -> Self {
        match bucket_m {
            1 => Self::PerSequence,
            _ => Self::PerToken,
        }
    }
}

impl From<AttentionViaCacheConstants> for Vec<ConstantValue> {
    fn from(c: AttentionViaCacheConstants) -> Self {
        let mut v = Vec::new();
        c.cache.push(&mut v);
        if c.sinks == SinkColumn::Present {
            v.push(ConstantValue::uint(AttentionViaCacheConstants::SINKS, 1));
        }
        v.push(ConstantValue::uint(
            AttentionViaCacheConstants::HEADS,
            c.heads.get(),
        ));
        push_query_rows(&mut v, c.query_rows);
        v
    }
}

/// How a decode lane holds the rope pairs it re-ropes on read (`ATTN_PAIR_CORESIDENT`, slot 12).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PairLanes {
    /// Each lane owns its pairs `{d, d + head_dim / 2}`: the rope is in-lane.
    Coresident,
    /// A lane's elements are a contiguous slice, its pairs met by shuffles.
    Shuffled,
}

/// A spans model's rope on read (slots 8/9/10, and 12): the class's rotated dims, its pairs'
/// offset and how a decode lane holds them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RopeOnRead {
    pub rot_dim: RotDim,
    pub pair_off: RopePairOff,
    pub lanes: PairLanes,
}

/// What a decode attention reads of one class's paged cache (slots 0..=12): the class's
/// geometry, its window, the pool's chunking, and the rope it re-applies on read.
#[derive(Clone, Copy, Debug)]
pub struct DecodeCache {
    pub head_dim: HeadDim,
    pub num_q_heads: NumQHeads,
    pub num_kv_heads: NumKvHeads,
    pub attn_scale: AttnScale,
    pub block_size: BlockSize,
    pub blocks_per_chunk: BlocksPerChunk,
    /// `ATTN_WINDOW` (slot 7): the class's sliding window, `0` for a full class.
    pub window: AttnWindow,
    pub rope_on_read: Option<RopeOnRead>,
}

impl DecodeCache {
    fn push(self, v: &mut Vec<ConstantValue>) {
        v.extend([
            ConstantValue::uint(ConstSlot(0), self.head_dim.get()),
            ConstantValue::uint(ConstSlot(1), self.num_q_heads.get()),
            ConstantValue::uint(ConstSlot(2), self.num_kv_heads.get()),
            ConstantValue::float(ConstSlot(3), self.attn_scale.get()),
            ConstantValue::uint(ConstSlot(4), self.block_size.get()),
            ConstantValue::kv_cap(ConstSlot(5)),
            ConstantValue::uint(ConstSlot(6), self.blocks_per_chunk.get()),
            ConstantValue::int(ConstSlot(7), self.window.get()),
        ]);
        if let Some(r) = self.rope_on_read {
            push_rope_on_read_consts(v, Some(r.rot_dim), Some(r.pair_off), Some(1));
            if r.lanes == PairLanes::Coresident {
                v.push(ConstantValue::uint(ConstSlot(12), 1));
            }
        }
    }
}

/// `ATTN_ROW_QUERIES` (slot 22), set for a bucket whose rows are tokens.
fn push_query_rows(v: &mut Vec<ConstantValue>, rows: QueryRows) {
    if rows == QueryRows::PerToken {
        v.push(ConstantValue::uint(ConstSlot(22), 1));
    }
}

/// `ATTN_KV_INT8` (13), `ATTN_SPLITS` (18) and `ATTN_COMBINE_GROUPS` (25): the int8 decode's and
/// its combine's shared part of the shape.
fn push_int8_split(v: &mut Vec<ConstantValue>, shape: crate::kv_int8::Int8DecodeShape) {
    v.extend([
        ConstantValue::uint(AttentionViaCacheConstants::INT8, 1),
        ConstantValue::uint(ConstSlot(18), shape.splits()),
        ConstantValue::uint(ConstSlot(25), shape.combine_groups()),
    ]);
}

/// `KernelId::AttentionDecodeInt8` (`attention_decode_gqa_int8`): one split of a KV head's
/// query-head group's int8 decode.
pub struct AttentionDecodeInt8Constants {
    pub cache: DecodeCache,
    pub query_rows: QueryRows,
    pub shape: crate::kv_int8::Int8DecodeShape,
}

impl From<AttentionDecodeInt8Constants> for Vec<ConstantValue> {
    fn from(c: AttentionDecodeInt8Constants) -> Self {
        let mut v = Vec::new();
        c.cache.push(&mut v);
        push_query_rows(&mut v, c.query_rows);
        push_int8_split(&mut v, c.shape);
        v.extend([
            ConstantValue::uint(AttentionViaCacheConstants::HEADS, c.shape.heads()),
            ConstantValue::uint(ConstSlot(24), c.shape.key_group()),
        ]);
        v
    }
}

/// `KernelId::AttentionDecodeCombine` (`attention_via_cache_v2_combine`): the merge of an int8
/// decode's splits, the sink column (`ATTN_SINKS`, slot 21) with them on a sinks layer.
pub struct AttentionDecodeCombineConstants {
    pub cache: DecodeCache,
    pub query_rows: QueryRows,
    pub shape: crate::kv_int8::Int8DecodeShape,
    pub sinks: SinkColumn,
}

/// Whether a layer's softmax carries an attention-sinks column (gpt-oss).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SinkColumn {
    Absent,
    Present,
}

impl From<AttentionDecodeCombineConstants> for Vec<ConstantValue> {
    fn from(c: AttentionDecodeCombineConstants) -> Self {
        let mut v = Vec::new();
        c.cache.push(&mut v);
        if c.sinks == SinkColumn::Present {
            v.push(ConstantValue::uint(AttentionViaCacheConstants::SINKS, 1));
        }
        push_query_rows(&mut v, c.query_rows);
        push_int8_split(&mut v, c.shape);
        v
    }
}

/// `KernelId::KvStage` (`kv_stage_<dtype>`, attention.metal slots): the layer's KV geometry.
pub struct KvStageConstants {
    pub head_dim: HeadDim,
    pub num_kv_heads: NumKvHeads,
    pub block_size: BlockSize,
    pub blocks_per_chunk: BlocksPerChunk,
    /// The layer's sliding window (`ATTN_WINDOW`, slot 7): 0 for a full-attention layer.
    pub window: AttnWindow,
}

impl From<KvStageConstants> for Vec<ConstantValue> {
    fn from(c: KvStageConstants) -> Self {
        vec![
            ConstantValue::uint(ConstSlot(0), c.head_dim.get()),
            ConstantValue::uint(ConstSlot(2), c.num_kv_heads.get()),
            ConstantValue::uint(ConstSlot(4), c.block_size.get()),
            ConstantValue::kv_cap(ConstSlot(5)),
            ConstantValue::uint(ConstSlot(6), c.blocks_per_chunk.get()),
            ConstantValue::int(ConstSlot(7), c.window.get()),
            ConstantValue::uint(AttentionViaCacheConstants::INT8, 1),
        ]
    }
}

// ── AttentionPrefillSdpaPaged (sdpa + steel) ───────────────────────

/// `KernelId::AttentionPrefillSdpaPaged` for both the
/// `attention_prefill_sdpa_v2_paged_*` and `attention_steel_paged_*`
/// kernel families.
///
/// Steel kernel reads slot 99 (`ATTN_PAGED_DEBUG_MODE`); the
/// sdpa_vector kernel ignores it. The lowering arm sets
/// `debug_mode = Some(AttnDebugMode(0))` for steel and `None` for sdpa
/// to keep the typed surface honest — Metal pipeline build tolerates
/// extra constants but failing to bind a declared slot is exactly the
/// `b3ddb3b46` regression.
#[derive(Copy, Clone)]
pub struct AttentionPrefillPagedConstants {
    pub head_dim: HeadDim,
    pub num_q_heads: NumQHeads,
    pub num_kv_heads: NumKvHeads,
    pub attn_scale: AttnScale,
    pub block_size: BlockSize,
    pub blocks_per_chunk: BlocksPerChunk,
    /// Sliding-window width (`ATTN_WINDOW` / `ATTN_PAGED_WINDOW`,
    /// slot 7). `0` = disabled. Read by BOTH the sdpa_vector paged
    /// kernel and the steel paged kernel (which additionally SKIPS
    /// K-tiles entirely older than the window — O(T·window) prefill).
    pub window: AttnWindow,
    /// `Some(0)` for the steel kernel (production), `None` for the
    /// sdpa_vector kernel (declares no slot 99).
    pub debug_mode: Option<AttnDebugMode>,
    /// Spans rope-on-read (slots 8/9/10) — shared by the sdpa-paged,
    /// gqa_shared, and steel kernels (steel reads them as ATTN_PAGED_*).
    /// `None` on non-spans dispatch → byte-identical emitted Vec.
    pub rot_dim: Option<RotDim>,
    pub pair_off: Option<RopePairOff>,
    pub rope_on_read: Option<u32>,
    /// Spans rope-once-to-scratch (slot 11, `ATTN_K_SCRATCH`) — gqa_shared
    /// only. `Some(1)` makes the gqa_shared kernel read PRE-ROPED K from the
    /// dense scratch (written by `rope_once_gqa_shared`) instead of re-roping
    /// each staged K tile in smem. `None` → no slot 11 emitted (steel/NAX use
    /// their own `ATTN_PAGED_ROR` scratch path; the sdpa-paged + in-kernel-rope
    /// gqa_shared keep the cos_sin path; non-spans is byte-identical).
    pub k_scratch: Option<u32>,
    /// Self-only span masking (slot 14, `ATTN_PAGED_SELFONLY`) — DECOUPLED from
    /// rope-on-read. `Some(1)` makes the steel paged kernel clamp a Relocatable
    /// span query's kb-loop to the span's own first block (block-diagonal), so
    /// the span's K/V is a pure function of its own bytes — the property the
    /// spans design requires for content-addressed reuse — WITHOUT touching the
    /// K-source/rope path. The sliding spans arm needs this because it runs
    /// with `rope_on_read: None` (ROR=0), which had silently gated the seek
    /// off. `None` → slot 14 unset → byte-identical (the seek folds away).
    pub self_only: Option<u32>,
    /// gpt-oss attention sinks (slot 21): `Some(1)` binds the layer's
    /// `[num_heads]` sink-logit tensor (buffer 9) and turns the sink math
    /// on in every kernel of this family (sdpa-paged, gqa_shared, steel,
    /// NAX — each reads the slot under its own `ATTN_PAGED_SINKS` /
    /// `NAXP_SINKS` name). `None` → no slot 21, byte-identical.
    pub sinks: Option<u32>,
    /// Resolved KV geometry — the compile-time continuation/span witness. Its
    /// type is only constructible via
    /// [`KvGeometry::resolve`](crate::tape::continuation_witness::KvGeometry::resolve),
    /// so no lowering arm can build a paged-attention dispatch without going
    /// through the single geometry resolver (a missing kernel migration is then
    /// a build error, not silent garbage).
    pub geom: crate::tape::continuation_witness::KvGeometry,
}

impl From<AttentionPrefillPagedConstants> for Vec<ConstantValue> {
    fn from(c: AttentionPrefillPagedConstants) -> Self {
        let mut v = vec![
            ConstantValue::uint(ConstSlot(0), c.head_dim.get()),
            ConstantValue::uint(ConstSlot(1), c.num_q_heads.get()),
            ConstantValue::uint(ConstSlot(2), c.num_kv_heads.get()),
            ConstantValue::float(ConstSlot(3), c.attn_scale.get()),
            ConstantValue::uint(ConstSlot(4), c.block_size.get()),
            ConstantValue::kv_cap(ConstSlot(5)),
            ConstantValue::uint(ConstSlot(6), c.blocks_per_chunk.get()),
            ConstantValue::int(ConstSlot(7), c.window.get()),
        ];
        push_rope_on_read_consts(&mut v, c.rot_dim, c.pair_off, c.rope_on_read);
        if let Some(ks) = c.k_scratch {
            v.push(ConstantValue::uint(ConstSlot(11), ks));
        }
        if let Some(so) = c.self_only {
            v.push(ConstantValue::uint(ConstSlot(14), so));
        }
        if let Some(sinks) = c.sinks {
            v.push(ConstantValue::uint(ConstSlot(21), sinks));
        }
        // Per-token `span_ids` index divisor from the KV-geometry witness
        // (slot 13). `== 1` = the per-token contract. Emitted for every
        // paged-attention dispatch; kernels that don't declare slot 13 tolerate
        // the extra constant.
        v.push(ConstantValue::uint(ConstSlot(13), c.geom.span_index_unit()));
        if let Some(dm) = c.debug_mode {
            v.push(ConstantValue::uint(ConstSlot(99), dm.get()));
        }
        v
    }
}

// ── MLX-affine code storage ──────────────────────────────────────

/// How the MLX-affine packed codes a command reads are stored. On a
/// [`GenClass::stores_affine_b4_offset8`](crate::tape::lowered::GenClass::stores_affine_b4_offset8)
/// target every 4-bit weight's codes are stored XOR 0x88; every kernel
/// that reads them carries this, and on `Offset8` gets `AFFINE_CODES_OFFSET8`
/// (slot 5) to XOR them back. `Planes` is offset-8 codes regrouped a 64-input chunk at a
/// time (`affine_codes.h`: chunk `c`'s 32 bytes of every row, row after row), which a kernel
/// reads under `AFFINE_CODES_PLANES` (slot 24) as well; `k` is the rows' input count, which
/// the store needs to regroup them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AffineCodes {
    AsWritten,
    Offset8,
    Planes { k: u32 },
}

/// The bytes of a 64-input chunk of one 4-bit row: the unit [`AffineCodes::Planes`] regroups.
pub const AFFINE_PLANE_ROW_BYTES: usize = 32;

/// `AFFINE_CODES_PLANES`' constant slot (`affine_codes.h`).
pub const AFFINE_CODES_PLANES_SLOT: ConstSlot = ConstSlot(24);

impl AffineCodes {
    pub fn of(profile: Option<&crate::targets::MetalTargetProfile>, bits: u32) -> Self {
        AffineCodesTarget::of(profile).for_bits(bits)
    }

    /// Slot 5, set on `Offset8` and `Planes`, so every other pipeline keeps its key.
    pub fn constant(self) -> Option<ConstantValue> {
        match self {
            AffineCodes::Offset8 | AffineCodes::Planes { .. } => {
                Some(ConstantValue::boolean(ConstSlot(5), true))
            }
            AffineCodes::AsWritten => None,
        }
    }

    /// Every constant a kernel reading codes stored this way takes: slot 5 on offset-8 storage,
    /// and slot 24 besides on planes.
    pub fn constants(self) -> impl Iterator<Item = ConstantValue> {
        let planes = matches!(self, AffineCodes::Planes { .. })
            .then(|| ConstantValue::boolean(AFFINE_CODES_PLANES_SLOT, true));
        self.constant().into_iter().chain(planes)
    }

    /// Signed (q - 8) codes: offset-8, as planes or not — what the matrix-unit kernels read.
    pub fn signed(self) -> bool {
        matches!(self, AffineCodes::Offset8 | AffineCodes::Planes { .. })
    }

    /// The same codes regrouped into planes, for rows of `k` inputs: offset-8 storage only
    /// (the planes kernels read signed codes), whole 64-input chunks only.
    pub fn planes(self, k: u32) -> Self {
        match self {
            AffineCodes::Offset8 if k.is_multiple_of(64) => AffineCodes::Planes { k },
            codes => codes,
        }
    }

    /// What a command whose kernel reads codes declares, from its baked
    /// constants (slot 5 is `bool` only on such kernels; slot 24 likewise, with its input
    /// count at slot 0, `QMM_K` / `IN_VEC_SIZE`).
    pub fn of_constants(constants: &[ConstantValue]) -> Self {
        let has = |c: ConstantValue| constants.contains(&c);
        let planes = ConstantValue::boolean(AFFINE_CODES_PLANES_SLOT, true);
        match AffineCodes::Offset8.constant() {
            Some(c) if has(c) && has(planes) => {
                let k = (constants.iter()).find(|v| v.index == 0).map(|v| v.bits);
                AffineCodes::Planes {
                    k: k.expect("a planes reader names its input count at slot 0"),
                }
            }
            Some(c) if has(c) => AffineCodes::Offset8,
            _ => AffineCodes::AsWritten,
        }
    }
}

/// A target's code storage, before a weight's width picks its [`AffineCodes`]
/// (only 4-bit codes are ever stored offset-8).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AffineCodesTarget {
    offset8_b4: bool,
}

impl AffineCodesTarget {
    pub fn of(profile: Option<&crate::targets::MetalTargetProfile>) -> Self {
        Self {
            offset8_b4: profile.is_some_and(|t| {
                crate::tape::lowered::GenClass::of(t.generation).stores_affine_b4_offset8()
            }),
        }
    }

    pub fn for_bits(self, bits: u32) -> AffineCodes {
        if self.offset8_b4 && bits == 4 {
            AffineCodes::Offset8
        } else {
            AffineCodes::AsWritten
        }
    }
}

// ── MLX-affine matmul ABI ─────────────────────────────────────────

use crate::tape::lowered::KernelId;

/// Where every MLX-affine matmul kernel ([`AFFINE_MATMUL_KERNELS`]) takes its weight's K.
pub const AFFINE_K: ConstSlot = ConstSlot(0);
/// Where every MLX-affine matmul kernel takes its weight's N.
pub const AFFINE_N: ConstSlot = ConstSlot(1);

/// The kernels an MLX-affine matmul (`AffineMatmul`) lowers to that read its weight: each takes
/// the weight's K and N at [`AFFINE_K`] and [`AFFINE_N`], signed.
pub const AFFINE_MATMUL_KERNELS: &[KernelId] = &[
    KernelId::AffineQmvQuad,
    KernelId::AffineQmvFast,
    KernelId::AffineQmv,
    KernelId::AffineQmvTiny,
    KernelId::AffineQmvWide,
    KernelId::AffineQmmT,
    KernelId::AffineQmmTNax,
    KernelId::AffineQmmTSplitK,
    KernelId::AffineQmmW4a8,
];

/// Of [`AFFINE_MATMUL_KERNELS`], those instantiated for int4 codes alone.
pub const INT4_MATMUL_KERNELS: &[KernelId] = &[KernelId::AffineQmmTSplitK, KernelId::AffineQmmW4a8];

/// Of [`AFFINE_MATMUL_KERNELS`], those on the matrix unit (`quantized_qmm_nax`), instantiated for
/// its groups alone.
pub const NAX_MATMUL_KERNELS: &[KernelId] = &[KernelId::AffineQmmTNax, KernelId::AffineQmmW4a8];

// ── MLX-affine QMV (decode matvec) ────────────────────────────────

/// `KernelId::AffineQmvQuad` / `AffineQmvFast` / `AffineQmv`
/// (`quantized_qmv` library; constants declared as signed `int`).
pub struct AffineQmvConstants {
    pub k: KDimI32,
    pub n: NDimI32,
    pub codes: AffineCodes,
}

impl From<AffineQmvConstants> for Vec<ConstantValue> {
    fn from(c: AffineQmvConstants) -> Self {
        let mut v = vec![
            ConstantValue::int(AFFINE_K, c.k.get()),
            ConstantValue::int(AFFINE_N, c.n.get()),
        ];
        v.extend(c.codes.constants());
        v
    }
}

/// `KernelId::RowProgram` (`row_program.metal`): the width (slot 0), its threads (1), the
/// instruction count (2), and instruction k's word (3 + 3k: op, dst, a, b, weight in 4-bit fields)
/// and floats (4 + 3k an epsilon or a scale, 5 + 3k a gain offset).
impl From<&crate::tape::step::RowProgram> for Vec<ConstantValue> {
    fn from(r: &crate::tape::step::RowProgram) -> Self {
        use crate::tape::step::RowInstr as R;
        let word = |op: u32, dst: u8, a: u8, b: u8, w: u8| {
            op | u32::from(dst) << 4 | u32::from(a) << 8 | u32::from(b) << 12 | u32::from(w) << 16
        };
        let mut v = vec![
            ConstantValue::uint(ConstSlot(0), r.width.get()),
            ConstantValue::uint(ConstSlot(1), NORM_THREADS),
        ];
        let instrs: Vec<R> = r.instrs.iter().flatten().copied().collect();
        v.push(ConstantValue::uint(ConstSlot(2), instrs.len() as u32));
        for (k, instr) in instrs.iter().enumerate() {
            let (w, f, g) = match *instr {
                R::Load { dst, input } => (word(0, dst, input, 0, 0), 0.0, 0.0),
                R::Add { dst, a, b } => (word(1, dst, a, b, 0), 0.0, 0.0),
                R::Norm {
                    dst,
                    a,
                    gain,
                    eps,
                    offset,
                } => (word(2, dst, a, 0, gain), eps.0, offset.0),
                R::ScaleWeight { dst, a, scalar } => (word(3, dst, a, 0, scalar), 0.0, 0.0),
                R::Scale { dst, a, scale } => (word(4, dst, a, 0, 0), scale.0, 0.0),
                R::Store { output, a } => (word(5, output, a, 0, 0), 0.0, 0.0),
            };
            let slot = 3 + 3 * k as u16;
            v.extend([
                ConstantValue::uint(ConstSlot(slot), w),
                ConstantValue::float(ConstSlot(slot + 1), f),
                ConstantValue::float(ConstSlot(slot + 2), g),
            ]);
        }
        v
    }
}

/// The norm a matvec applies to its input as it loads it (`quantized_qmv.metal`'s `QMV_NORMED`):
/// its epsilon (slot 8) and gain offset (9).
impl From<crate::tape::step::RowNorm> for Vec<ConstantValue> {
    fn from(n: crate::tape::step::RowNorm) -> Self {
        vec![
            ConstantValue::float(ConstSlot(8), n.eps.0),
            ConstantValue::float(ConstSlot(9), n.offset.0),
        ]
    }
}

/// A one-row matvec's [`QmvEnds`](crate::tape::step::QmvEnds), compiled in: its norm's
/// ([`RowNorm`](crate::tape::step::RowNorm)'s slots 8 and 9), whether its rows add into the
/// residual (10), take a bias (11), and their scale (12).
impl From<crate::tape::step::QmvEnds> for Vec<ConstantValue> {
    fn from(e: crate::tape::step::QmvEnds) -> Self {
        let norm = e.norm.into_iter().flat_map(Vec::from);
        let residual = e
            .residual
            .then(|| ConstantValue::boolean(ConstSlot(10), true));
        let bias = e.bias.map(|_| ConstantValue::boolean(ConstSlot(11), true));
        let scale = e.scale.map(|s| ConstantValue::float(ConstSlot(12), s.0));
        norm.chain(residual).chain(bias).chain(scale).collect()
    }
}

/// `KernelId::NormedGemv` (`gemv_normed_<T>_s_<G>`, `gemm.metal`): the GEMM's one row (slots 0 /
/// 1 / 2: M, N, K) and its folded norm's epsilon (5) and gain offset (6).
pub struct NormedGemvConstants {
    pub n: NDim,
    pub k: KDim,
    pub eps: crate::tape::step::Eps,
    pub offset: crate::tape::step::GainOffset,
}

impl From<NormedGemvConstants> for Vec<ConstantValue> {
    fn from(c: NormedGemvConstants) -> Self {
        vec![
            ConstantValue::uint(ConstSlot(0), 1),
            ConstantValue::uint(ConstSlot(1), c.n.get()),
            ConstantValue::uint(ConstSlot(2), c.k.get()),
            ConstantValue::float(ConstSlot(5), c.eps.0),
            ConstantValue::float(ConstSlot(6), c.offset.0),
        ]
    }
}

/// `KernelId::AffineGatherQmvFast` / `AffineGatherQmv` (`affine_gather_qmv[_fast]_*`): the
/// expert matvec's [`AffineQmvConstants`] and which rows it reads (slot 2: the output rows one
/// input row feeds).
pub struct AffineGatherQmvConstants {
    pub qmv: AffineQmvConstants,
    pub rows: GatherRows,
}

/// The rows a MoE gather matvec reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GatherRows {
    /// Each token's row, once per chosen expert.
    Tokens(TopK),
    /// Each (token, expert) pair's own row.
    Pairs,
}

impl From<AffineGatherQmvConstants> for Vec<ConstantValue> {
    fn from(c: AffineGatherQmvConstants) -> Self {
        let per_row = match c.rows {
            GatherRows::Tokens(k) => k.get(),
            GatherRows::Pairs => 1,
        };
        let mut v: Vec<ConstantValue> = c.qmv.into();
        v.push(ConstantValue::int(ConstSlot(2), per_row as i32));
        v
    }
}

/// `KernelId::MoeGateUpAct` (`affine_gather_qmv_gated[_fast]_*`, `Q` =
/// [`GatedGatherQmvConstants`]) and `KernelId::AffineQmvGated` (`affine_qmv_gated[_fast]_*`,
/// `Q` = [`AffineQmvConstants`]): the gate projection's matvec constants (the up's are the same)
/// and the activation (slot 3: 0 SiLU, 1 GELU, 2 SwiGLU-OAI, 3 an ungated expert's relu² of its
/// one projection — the gather form only, binding no up). The gather form additionally takes
/// the gpt-oss gate/up LINEAR biases: the lowering binds them at the kernel's buffer 14 and sets
/// slot 4 (`GATED_LINEAR_BIAS`), which this struct deliberately does NOT own — it is a MoE-bundle
/// fact, not a matvec one, and only the gathered command ever carries it.
pub struct AffineGatedQmvConstants<Q> {
    pub qmv: Q,
    pub act: scratchy_subtile::subtile_ir::ExpertBody,
}

impl<Q: Into<Vec<ConstantValue>>> From<AffineGatedQmvConstants<Q>> for Vec<ConstantValue> {
    fn from(c: AffineGatedQmvConstants<Q>) -> Self {
        use crate::tape::step::{GatedAct, UngatedAct};
        use scratchy_subtile::subtile_ir::ExpertBody;
        let act = match c.act {
            ExpertBody::Gated(GatedAct::Silu) => 0,
            ExpertBody::Gated(GatedAct::Gelu) => 1,
            // gpt-oss SwiGLU-OAI: `y = (clamp(u, ±7)+1) · min(g,7) ·
            // sigmoid(1.702·g)` — the ±7 limits are baked into the kernel's
            // `swiglu_oai_mul_f`, asserted equal to the config's
            // `swiglu_limit` at the bridge.
            ExpertBody::Gated(GatedAct::SwigluOai) => 2,
            ExpertBody::Ungated(UngatedAct::Relu2) => 3,
        };
        let mut v: Vec<ConstantValue> = c.qmv.into();
        v.push(ConstantValue::int(ConstSlot(3), act));
        v
    }
}

/// The gated MoE gather's matvec constants: the gather's ([`AffineGatherQmvConstants`]) and where
/// its pairs that share an expert run (slot 21).
pub struct GatedGatherQmvConstants {
    pub gather: AffineGatherQmvConstants,
    pub shared: SharedExperts,
}

impl From<GatedGatherQmvConstants> for Vec<ConstantValue> {
    fn from(c: GatedGatherQmvConstants) -> Self {
        let mut v: Vec<ConstantValue> = c.gather.into();
        v.push(c.shared.constant());
        v
    }
}

/// Where the gated MoE gather runs the (row, expert) pairs that share an expert.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SharedExperts {
    /// Each pair in its own threadgroup: a bucket of many rows, whose experts take many pairs.
    /// An expert two pairs read costs two reads, but no pair scans the bucket's others.
    Apart,
    /// An expert's pairs one after another in the threadgroup of its first pair, which reads the
    /// expert's weights once: a bucket of a speculative verify step's few rows. Every threadgroup
    /// scans the bucket's every other pair for its expert, so the mode's cost grows with the
    /// square of the pair count while the reads it saves stay the few pairs that share one.
    InFirstPair,
}

impl SharedExperts {
    /// A bucket of `bucket_m` rows baking `pairs` (row, pick) pairs' choice: in the first pair at
    /// a verify-sized bucket of more than one row
    /// (`op_abi::METAL_SHARED_EXPERTS_FROM..=METAL_VERIFY_ROWS`) whose pairs are few enough for
    /// the mode's square-cost scan (`op_abi::METAL_SHARED_EXPERTS_MAX_PAIRS`).
    pub fn for_pairs(bucket_m: u32, pairs: u32) -> Self {
        use crate::op_abi::{
            METAL_SHARED_EXPERTS_FROM, METAL_SHARED_EXPERTS_MAX_PAIRS, METAL_VERIFY_ROWS,
        };
        match (METAL_SHARED_EXPERTS_FROM..=METAL_VERIFY_ROWS).contains(&bucket_m)
            && pairs <= METAL_SHARED_EXPERTS_MAX_PAIRS
        {
            true => Self::InFirstPair,
            false => Self::Apart,
        }
    }

    /// Slot 21: 0 apart, 1 in the first pair.
    pub fn constant(self) -> ConstantValue {
        ConstantValue::int(ConstSlot(21), i32::from(self == Self::InFirstPair))
    }
}

/// `KernelId::MoeDownCombine` (`affine_gather_qmv_combine[_fast]_*`): the down projection's
/// [`AffineQmvConstants`] and the experts each token chose (slot 2), whose rows it combines. The
/// gpt-oss down LINEAR biases ride the same rule as the gated kernel's: the lowering binds them
/// at buffer 10 and sets slot 6 (`COMBINE_LINEAR_BIAS`), outside this struct.
pub struct AffineCombineQmvConstants {
    pub qmv: AffineQmvConstants,
    pub top_k: TopK,
    /// What it computes as it stores each row: slot 18 a shared expert's gated rows, slot 19 the
    /// residual add (`COMBINE_GATE_SCALE` / `COMBINE_RESIDUAL`). Unset when it stores the sum alone.
    pub gate_scale: bool,
    pub residual: bool,
    /// Whether each pair's down input row is where a sorted bake's sort put it, read through its
    /// `pos` at buffer 11 (slot 22, `COMBINE_ROWS_SORTED`).
    pub sorted_rows: bool,
}

impl From<AffineCombineQmvConstants> for Vec<ConstantValue> {
    fn from(c: AffineCombineQmvConstants) -> Self {
        let mut v: Vec<ConstantValue> = c.qmv.into();
        v.push(ConstantValue::int(ConstSlot(2), c.top_k.get() as i32));
        v.extend(
            c.gate_scale
                .then(|| ConstantValue::boolean(ConstSlot(18), true)),
        );
        v.extend(
            c.residual
                .then(|| ConstantValue::boolean(ConstSlot(19), true)),
        );
        v.extend(
            c.sorted_rows
                .then(|| ConstantValue::boolean(ConstSlot(22), true)),
        );
        v
    }
}

/// `KernelId::AffineQmvWide` — the small-M band matvec. Same K/N
/// slots as the other qmv kernels plus the bucket's M at slot 7
/// (`QMV_WIDE_M`), baked like K/N.
pub struct AffineQmvWideConstants {
    pub k: KDimI32,
    pub n: NDimI32,
    pub m: MDimI32,
    pub codes: AffineCodes,
}

impl From<AffineQmvWideConstants> for Vec<ConstantValue> {
    fn from(c: AffineQmvWideConstants) -> Self {
        let mut v = vec![
            ConstantValue::int(AFFINE_K, c.k.get()),
            ConstantValue::int(AFFINE_N, c.n.get()),
            ConstantValue::int(ConstSlot(7), c.m.get()),
        ];
        v.extend(c.codes.constants());
        v
    }
}

// ── MoE routing ───────────────────────────────────────────────────

/// `KernelId::Softmax` (`block_softmax_precise_<T>`, `topk_renorm_<T>`, `softmax.metal`): the
/// scores a row holds.
pub struct SoftmaxConstants {
    pub row: ScoresRow,
}

/// A MoE score row: every expert's, or the chosen top-k's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScoresRow {
    Experts(NumExperts),
    TopK(TopK),
}

impl From<SoftmaxConstants> for Vec<ConstantValue> {
    fn from(c: SoftmaxConstants) -> Self {
        let width = match c.row {
            ScoresRow::Experts(e) => e.get(),
            ScoresRow::TopK(k) => k.get(),
        };
        vec![ConstantValue::int(ConstSlot(0), width as i32)]
    }
}

/// `KernelId::ArgPartitionTopK` (`c_arg_block_sort_*`): MLX `block_sort` over contiguous rows of
/// the router's experts — slot 0 the sorted axis's size, 1 / 2 its in / out stride, 3 / 4 the row
/// strides.
pub struct ArgsortConstants {
    pub experts: NumExperts,
}

impl From<ArgsortConstants> for Vec<ConstantValue> {
    fn from(c: ArgsortConstants) -> Self {
        let e = c.experts.get() as i32;
        [e, 1, 1, e, e]
            .into_iter()
            .zip(0..)
            .map(|(v, slot)| ConstantValue::int(ConstSlot(slot), v))
            .collect()
    }
}

/// `KernelId::SliceTrailingColsU32` (`slice_trailing_cols_u32`) and `KernelId::TakeAlongAxis`
/// (`take_along_axis_2d_contig_<T>`): the router's experts (slot 0) and the top-k it keeps of
/// them (slot 1).
pub struct MoeTopKConstants {
    pub experts: NumExperts,
    pub top_k: TopK,
}

impl From<MoeTopKConstants> for Vec<ConstantValue> {
    fn from(c: MoeTopKConstants) -> Self {
        vec![
            ConstantValue::int(ConstSlot(0), c.experts.get() as i32),
            ConstantValue::int(ConstSlot(1), c.top_k.get() as i32),
        ]
    }
}

/// `KernelId::MoeRoute` (`moe_route_<T>_bn<bn>`, `moe_route.metal`): the experts a router scores
/// (slot 0), the top-k it keeps (slot 1), and its program — the pre over the experts (slot 2:
/// 0 none, 1 softmax, 2 F32 sigmoid + correction bias, 3 F32 logit + linear bias), the scores'
/// scale (slot 3, set only when they scale), what follows (slot 4: 0 nothing, 1 softmax, 2
/// renorm), the per-expert scale (slot 5), and the last scale (slot 6, set only when they take
/// one).
pub struct MoeRouteConstants {
    pub experts: NumExperts,
    pub top_k: TopK,
    pub program: crate::tape::step::RouteProgram,
}

/// A routing program's pre and post codes, as both routing kernels read them.
fn route_codes(p: &crate::tape::step::RouteProgram) -> (i32, i32) {
    use crate::tape::step::{RoutePost, RoutePre};
    let pre = match p.pre {
        RoutePre::None => 0,
        RoutePre::Softmax => 1,
        RoutePre::SigmoidBias(_) => 2,
        // gpt-oss: the F32 linear router bias added on the fly — the biased values order the
        // top-k picks AND are read back as the gathered scores.
        RoutePre::Bias(_) => 3,
    };
    let post = match p.post {
        RoutePost::None => 0,
        RoutePost::Softmax => 1,
        RoutePost::Renorm => 2,
    };
    (pre, post)
}

impl From<MoeRouteConstants> for Vec<ConstantValue> {
    fn from(c: MoeRouteConstants) -> Self {
        let p = c.program;
        let (pre, post) = route_codes(&p);
        let mut v = vec![
            ConstantValue::int(ConstSlot(0), c.experts.get() as i32),
            ConstantValue::int(ConstSlot(1), c.top_k.get() as i32),
            ConstantValue::int(ConstSlot(2), pre),
        ];
        v.extend(p.scale.map(|s| ConstantValue::float(ConstSlot(3), s.0)));
        v.push(ConstantValue::int(ConstSlot(4), post));
        v.push(ConstantValue::int(
            ConstSlot(5),
            i32::from(p.expert_scale.is_some()),
        ));
        v.extend(
            p.post_scale
                .map(|s| ConstantValue::float(ConstSlot(6), s.0)),
        );
        v
    }
}

/// A routed expert kernel's routing (`MetalFusion::MoeRouted`, `quantized_qmv.metal` slots
/// 13-17 and 20): the experts, and the [`MoeRouteConstants`] program — the pre over the experts
/// (0 none, 1 softmax, 2 F32 sigmoid + correction bias, 3 F32 logit + linear bias), the scores'
/// scale, their last step, the per-expert scale, the last scale.
pub struct RoutedConstants {
    pub experts: NumExperts,
    pub program: crate::tape::step::RouteProgram,
}

impl From<RoutedConstants> for Vec<ConstantValue> {
    fn from(c: RoutedConstants) -> Self {
        let p = c.program;
        let (pre, post) = route_codes(&p);
        let mut v = vec![
            ConstantValue::int(ConstSlot(13), c.experts.get() as i32),
            ConstantValue::int(ConstSlot(14), pre),
        ];
        v.extend(p.scale.map(|s| ConstantValue::float(ConstSlot(15), s.0)));
        v.push(ConstantValue::int(ConstSlot(16), post));
        v.push(ConstantValue::boolean(
            ConstSlot(17),
            p.expert_scale.is_some(),
        ));
        v.extend(
            p.post_scale
                .map(|s| ConstantValue::float(ConstSlot(20), s.0)),
        );
        v
    }
}

// ── MLX-affine QMM_T (prefill matmul) ─────────────────────────────

/// `KernelId::AffineQmmT` / `AffineQmmTNax`
/// (`quantized_qmm.metal` / `quantized_qmm_nax.metal`; constants
/// declared as signed `int`).
pub struct AffineQmmTConstants {
    pub k: KDimI32,
    pub n: NDimI32,
    pub m: MDimI32,
    pub codes: AffineCodes,
}

impl From<AffineQmmTConstants> for Vec<ConstantValue> {
    fn from(c: AffineQmmTConstants) -> Self {
        let mut v = vec![
            ConstantValue::int(AFFINE_K, c.k.get()),
            ConstantValue::int(AFFINE_N, c.n.get()),
            ConstantValue::int(ConstSlot(2), c.m.get()),
        ];
        v.extend(c.codes.constants());
        v
    }
}

// ── MLX-affine QMM_T SplitK ───────────────────────────────────────

/// `KernelId::AffineQmmTSplitK`. The `k_partition_size` slot was the
/// historical mistake — omitting it leaves the partition stride
/// undefined and every layer's prefill output is garbage. Exhaustive
/// struct so it can't be omitted.
pub struct AffineQmmTSplitKConstants {
    pub k: KDimI32,
    pub n: NDimI32,
    pub m: MDimI32,
    pub k_partition_size: KPartitionSizeI32,
    pub codes: AffineCodes,
}

impl From<AffineQmmTSplitKConstants> for Vec<ConstantValue> {
    fn from(c: AffineQmmTSplitKConstants) -> Self {
        let mut v = vec![
            ConstantValue::int(AFFINE_K, c.k.get()),
            ConstantValue::int(AFFINE_N, c.n.get()),
            ConstantValue::int(ConstSlot(2), c.m.get()),
            ConstantValue::int(ConstSlot(3), c.k_partition_size.get()),
        ];
        v.extend(c.codes.constants());
        v
    }
}

// ── SplitKReduceSum ────────────────────────────────────────────────

/// `KernelId::SplitKReduceSum`
/// (`quantized_splitk_reduce.metal::splitk_reduce_sum_<dtype>`).
pub struct SplitKReduceSumConstants {
    pub bucket_m: BucketM,
    pub n: NDim,
    pub split_k: SplitK,
}

impl From<SplitKReduceSumConstants> for Vec<ConstantValue> {
    fn from(c: SplitKReduceSumConstants) -> Self {
        vec![
            ConstantValue::uint(ConstSlot(0), c.bucket_m.get()),
            ConstantValue::uint(ConstSlot(1), c.n.get()),
            ConstantValue::uint(ConstSlot(2), c.split_k.get()),
        ]
    }
}

// ── SiluMul ────────────────────────────────────────────────────────

/// `KernelId::SiluMul` (`silu_mul.metal::silu_mul_<dtype>`).
/// `n = M * intermediate_size` — total output elements.
pub struct SiluMulConstants {
    pub n: HiddenSize,
}

impl From<SiluMulConstants> for Vec<ConstantValue> {
    fn from(c: SiluMulConstants) -> Self {
        vec![ConstantValue::uint(ConstSlot(0), c.n.get())]
    }
}

// ── Multi-head latent attention, its column window and its per-head matmul ─────

/// `KernelId::SliceCols` (`slice_cols_<dtype>`, `slice_cols.metal`).
pub struct SliceColsConstants {
    pub in_cols: ActivationWidth,
    pub offset: crate::tape::step::ColOffset,
    pub cols: ActivationWidth,
    pub elements: ElementCount,
}

impl From<SliceColsConstants> for Vec<ConstantValue> {
    fn from(c: SliceColsConstants) -> Self {
        vec![
            ConstantValue::uint(ConstSlot(0), c.in_cols.get()),
            ConstantValue::uint(ConstSlot(1), c.offset.0),
            ConstantValue::uint(ConstSlot(2), c.cols.get()),
            ConstantValue::uint(ConstSlot(3), c.elements.get()),
        ]
    }
}

/// `KernelId::HeadMatmul` (`head_qmv_*` / `head_mv_*`, `head_matmul.metal`): the heads, the
/// lanes each row dots, a head's output lanes and input part, and an affine weight's group and
/// code width (slots 4 / 5, set for the affine kernels alone).
pub struct HeadMatmulConstants {
    pub heads: NumQHeads,
    pub k: KDim,
    pub n_head: NDim,
    pub in_stride: RowStride,
    /// A threadgroup's simdgroups (slot 6) and the output rows each computes (slot 7).
    pub simdgroups: crate::tape::ids::SimdgroupCount,
    pub rows_per_simdgroup: crate::tape::ids::RowsPerSimdgroup,
    pub affine: Option<(
        crate::tape::ids::AffineGroupSize,
        crate::tape::ids::AffineBits,
    )>,
}

impl From<HeadMatmulConstants> for Vec<ConstantValue> {
    fn from(c: HeadMatmulConstants) -> Self {
        let mut v = vec![
            ConstantValue::uint(ConstSlot(0), c.heads.get()),
            ConstantValue::uint(ConstSlot(1), c.k.get()),
            ConstantValue::uint(ConstSlot(2), c.n_head.get()),
            ConstantValue::uint(ConstSlot(3), c.in_stride.get()),
            ConstantValue::uint(ConstSlot(6), c.simdgroups.get()),
            ConstantValue::uint(ConstSlot(7), c.rows_per_simdgroup.get()),
        ];
        if let Some((group, bits)) = c.affine {
            v.push(ConstantValue::uint(ConstSlot(4), group.get()));
            v.push(ConstantValue::uint(ConstSlot(5), bits.get()));
        }
        v
    }
}

/// The paged cache an MLA kernel reads or writes: its block's rows, the chunks it is held in, and
/// the lanes of a row its K store holds (`mla.metal` slots 4 to 6 and 14; slot 5, the block
/// table's stride, is the tape's KV cap rung).
#[derive(Clone, Copy)]
pub struct MlaCacheConstants {
    pub block_size: BlockSize,
    pub blocks_per_chunk: BlocksPerChunk,
    pub half_row: HeadDim,
}

/// How an MLA attention spreads over threadgroups: the query heads one serves, and its
/// simdgroups.
#[derive(Clone, Copy)]
pub struct MlaAttendTiling {
    pub heads_per_tg: NumQHeads,
    pub simdgroups: crate::tape::ids::SimdgroupCount,
    /// The partitions each sequence's cached rows split into: one writes the output, more write
    /// partials (each `partial` elements) a merge of `merge_threads` threads a head combines.
    pub parts: crate::tape::ids::KvPartitions,
    pub partial: crate::tape::ids::PartialStride,
    pub merge_threads: crate::tape::ids::ThreadCount,
}

/// `KernelId::MlaRopeQ` / `MlaAppend` / `MlaAttend` (`mla.metal`): the geometry (slots 0 to 3),
/// the cache (4 to 6, the writer's and the attention's), and the attention's and its merge's: the
/// softmax scale (7) and the tiling (8 to 12); a row kernel's, its threads (13).
pub struct MlaConstants {
    pub mla: crate::tape::step::Mla,
    pub cache: Option<MlaCacheConstants>,
    pub attend: Option<MlaAttendTiling>,
    /// A row kernel's threads (slot 13): the query rotation's and the cache write's.
    pub row_threads: Option<crate::tape::ids::ThreadCount>,
}

impl From<MlaConstants> for Vec<ConstantValue> {
    fn from(c: MlaConstants) -> Self {
        let m = c.mla;
        let mut v = vec![
            ConstantValue::uint(ConstSlot(0), m.heads().get()),
            ConstantValue::uint(ConstSlot(1), m.latent().get()),
            ConstantValue::uint(ConstSlot(2), m.nope().get()),
            ConstantValue::uint(ConstSlot(3), m.rope().get()),
        ];
        if let Some(cache) = c.cache {
            v.push(ConstantValue::uint(ConstSlot(4), cache.block_size.get()));
            v.push(ConstantValue::kv_cap(ConstSlot(5)));
            v.push(ConstantValue::uint(
                ConstSlot(6),
                cache.blocks_per_chunk.get(),
            ));
            v.push(ConstantValue::uint(ConstSlot(14), cache.half_row.get()));
        }
        if let Some(t) = c.attend {
            v.push(ConstantValue::float(ConstSlot(7), m.scale().get()));
            v.push(ConstantValue::uint(ConstSlot(8), t.heads_per_tg.get()));
            v.push(ConstantValue::uint(ConstSlot(9), t.simdgroups.get()));
            v.push(ConstantValue::uint(ConstSlot(10), t.parts.get()));
            v.push(ConstantValue::uint(ConstSlot(11), t.partial.get()));
            v.push(ConstantValue::uint(ConstSlot(12), t.merge_threads.get()));
        }
        if let Some(threads) = c.row_threads {
            v.push(ConstantValue::uint(ConstSlot(13), threads.get()));
        }
        v
    }
}

// ── GateApply / GateSplit (Qwen3.5 attention output gate) ─────────

/// `KernelId::GateApply` (`gate_apply.metal::gate_apply_<dtype>`).
/// `n = M * num_heads * head_dim` — total output elements. Same single
/// `n` constant as `SiluMulConstants`, kept distinct for clarity.
pub type GateApplyConstants = SiluMulConstants;

/// `KernelId::GateSplit` (`gate_split.metal::gate_split_<dtype>`).
/// `n` = per-output element count (`M * num_heads * head_dim`);
/// `head_dim` / `num_heads` drive the per-head interleaved source index.
pub struct GateSplitConstants {
    pub n: HiddenSize,
    pub head_dim: u32,
    pub num_heads: u32,
}

impl From<GateSplitConstants> for Vec<ConstantValue> {
    fn from(c: GateSplitConstants) -> Self {
        vec![
            ConstantValue::uint(ConstSlot(0), c.n.get()),
            ConstantValue::uint(ConstSlot(1), c.head_dim),
            ConstantValue::uint(ConstSlot(2), c.num_heads),
        ]
    }
}

/// `KernelId::GateScale` (`gate_scale.metal::gate_scale_<dtype>`).
/// `n` = total output elements (`M * hidden_size`); `cols` =
/// `hidden_size` — the gate's row index for the `[T, 1]` broadcast is
/// `gid / cols`.
pub struct GateScaleConstants {
    pub n: HiddenSize,
    pub cols: u32,
}

impl From<GateScaleConstants> for Vec<ConstantValue> {
    fn from(c: GateScaleConstants) -> Self {
        vec![
            ConstantValue::uint(ConstSlot(0), c.n.get()),
            ConstantValue::uint(ConstSlot(1), c.cols),
        ]
    }
}

// ── AffineEmbed (MLX-affine int4 embedding lookup) ────────────────

/// `KernelId::AffineEmbed`
/// (`quantized_dequantize.metal::affine_embed_<dtype>_gs_<gs>_b_4`).
pub struct AffineEmbedConstants {
    pub hidden_size: HiddenSize,
    pub codes: AffineCodes,
}

impl From<AffineEmbedConstants> for Vec<ConstantValue> {
    fn from(c: AffineEmbedConstants) -> Self {
        let mut v = vec![ConstantValue::uint(ConstSlot(0), c.hidden_size.get())];
        v.extend(c.codes.constants());
        v
    }
}

// ── GatherSampledRows / PutSampledRowsBack ───────────────────────

/// `KernelId::GatherSampledRows` and `KernelId::PutSampledRowsBack`
/// share the single-`row_stride` constant layout
/// (`gather_sampled_rows_<dtype>_specialized` and
/// `put_sampled_rows_back_<dtype>_specialized`).
pub struct SampleRowsConstants {
    pub row_stride: RowStride,
}

impl From<SampleRowsConstants> for Vec<ConstantValue> {
    fn from(c: SampleRowsConstants) -> Self {
        vec![ConstantValue::uint(ConstSlot(0), c.row_stride.get())]
    }
}

// ── FusedGateUpSiluMul (decode + prefill) ─────────────────────────

/// `KernelId::FusedGateUpSiluMul` decode branch
/// (`fused_gate_up_silu_mul_..._specialized`, decode `bucket_m == 1`).
/// Slots are 3/4/5 — the prefill branch uses 6/7/8, so the two are
/// distinct struct types.
pub struct FusedGateUpSiluMulDecodeConstants {
    pub bucket_m: BucketM,
    pub intermediate_size: IntermediateSize,
    pub q_size: QSize,
    /// GELU (Gemma GeGLU) vs SiLU (SwiGLU) activation. Slot 9; default
    /// false keeps existing SwiGLU dispatches bit-identical.
    pub is_gelu: bool,
}

impl From<FusedGateUpSiluMulDecodeConstants> for Vec<ConstantValue> {
    fn from(c: FusedGateUpSiluMulDecodeConstants) -> Self {
        vec![
            ConstantValue::uint(ConstSlot(3), c.bucket_m.get()),
            ConstantValue::uint(ConstSlot(4), c.intermediate_size.get()),
            ConstantValue::uint(ConstSlot(5), c.q_size.get()),
            ConstantValue::boolean(ConstSlot(9), c.is_gelu),
        ]
    }
}

/// `KernelId::FusedGateUpSiluMul` prefill branch
/// (`fused_gate_up_silu_mul_gemm_steel_..._specialized`).
pub struct FusedGateUpSiluMulPrefillConstants {
    pub bucket_m: BucketM,
    pub intermediate_size: IntermediateSize,
    pub q_size: QSize,
    /// GELU vs SiLU activation. Slot 10; default false.
    pub is_gelu: bool,
}

impl From<FusedGateUpSiluMulPrefillConstants> for Vec<ConstantValue> {
    fn from(c: FusedGateUpSiluMulPrefillConstants) -> Self {
        vec![
            ConstantValue::uint(ConstSlot(6), c.bucket_m.get()),
            ConstantValue::uint(ConstSlot(7), c.intermediate_size.get()),
            ConstantValue::uint(ConstSlot(8), c.q_size.get()),
            ConstantValue::boolean(ConstSlot(10), c.is_gelu),
        ]
    }
}

// ── Off-tape logits kernels ────────────────────────────────────────

/// `argmax.metal` slot 0: the logits per row.
pub struct ArgmaxConstants {
    pub vocab: crate::tape::ids::LogitsWidth,
}

impl From<ArgmaxConstants> for Vec<ConstantValue> {
    fn from(c: ArgmaxConstants) -> Self {
        vec![ConstantValue::uint(ConstSlot(0), c.vocab.get())]
    }
}

/// `logprobs.metal` slot 0: the logits per row.
pub struct LogprobsConstants {
    pub vocab: crate::tape::ids::LogitsWidth,
}

impl From<LogprobsConstants> for Vec<ConstantValue> {
    fn from(c: LogprobsConstants) -> Self {
        vec![ConstantValue::uint(ConstSlot(0), c.vocab.get())]
    }
}

/// `grammar_mask.metal`: slot 0 the logits per row, slot 1 the allow-bitset row stride.
pub struct GrammarMaskConstants {
    pub vocab: crate::tape::ids::LogitsWidth,
    pub words_per_row: crate::tape::ids::BitsetWords,
}

impl From<GrammarMaskConstants> for Vec<ConstantValue> {
    fn from(c: GrammarMaskConstants) -> Self {
        vec![
            ConstantValue::uint(ConstSlot(0), c.vocab.get()),
            ConstantValue::uint(ConstSlot(1), c.words_per_row.get()),
        ]
    }
}

/// `sampling.metal`: slot 0 the logits per row, slot 1 whether the telemetry spill is compiled
/// in (`sample_softmax_materialize` / `sample_finalize` only).
pub struct SamplerConstants {
    pub vocab: crate::tape::ids::LogitsWidth,
    pub telemetry: Option<bool>,
}

impl From<SamplerConstants> for Vec<ConstantValue> {
    fn from(c: SamplerConstants) -> Self {
        let vocab = ConstantValue::uint(ConstSlot(0), c.vocab.get());
        let telemetry = c.telemetry.map(|t| ConstantValue::boolean(ConstSlot(1), t));
        std::iter::once(vocab).chain(telemetry).collect()
    }
}

/// `mtp_chain.metal`: slot 0 a multi-token-prediction head's hidden width, slot 1 its drafts a
/// step, slot 2 the width of its rotary cos/sin rows (0: its rope reads positions).
#[derive(Clone, Copy)]
pub struct MtpChainConstants {
    pub hidden: HiddenSize,
    pub drafts: crate::tape::ids::NumDrafts,
    pub rope: crate::tape::ids::RotDim,
}

impl From<MtpChainConstants> for Vec<ConstantValue> {
    fn from(c: MtpChainConstants) -> Self {
        vec![
            ConstantValue::uint(ConstSlot(0), c.hidden.get()),
            ConstantValue::uint(ConstSlot(1), c.drafts.get()),
            ConstantValue::uint(ConstSlot(2), c.rope.get()),
        ]
    }
}

/// `chain_advance.metal` slot 0: the model's KV block size.
pub struct ChainAdvanceConstants {
    pub block_size: BlockSize,
}

impl From<ChainAdvanceConstants> for Vec<ConstantValue> {
    fn from(c: ChainAdvanceConstants) -> Self {
        vec![ConstantValue::uint(ConstSlot(0), c.block_size.get())]
    }
}

// ── Splash's DFlash2 drafter (`dflash2.metal`, `dflash2_select.metal`) ─────────────────────────

/// `dflash2_target_sampling.metal`'s constants (`TSAMPLE_*`, slots 0-1): a target's logits width
/// and its drafter's block.
pub struct TargetSamplingConstants {
    pub vocab: crate::tape::ids::LogitsWidth,
    pub rows: u32,
}

impl From<TargetSamplingConstants> for Vec<ConstantValue> {
    fn from(c: TargetSamplingConstants) -> Self {
        vec![
            ConstantValue::uint(ConstSlot(0), c.vocab.get()),
            ConstantValue::uint(ConstSlot(1), c.rows),
        ]
    }
}

/// `dflash2_accept.metal`'s constants (`ACCEPT_*`, slots 0-1): a drafter's block and the token
/// its proposal rows embed.
#[derive(Clone, Copy, Debug)]
pub struct DraftAcceptConstants {
    pub rows: u32,
    pub mask: u32,
}

impl From<DraftAcceptConstants> for Vec<ConstantValue> {
    fn from(c: DraftAcceptConstants) -> Self {
        vec![
            ConstantValue::uint(ConstSlot(0), c.rows),
            ConstantValue::uint(ConstSlot(1), c.mask),
        ]
    }
}

/// `dflash2_capture.metal`'s constants (`CAPTURE_*`, slots 0-3): a target layer's output rows
/// into block `slot` of `slots` of the captured rows.
pub struct CaptureConstants {
    pub hidden: u32,
    pub slot: u32,
    pub slots: u32,
    /// The bucket's elements: its rows times `hidden`.
    pub elements: u32,
}

impl From<CaptureConstants> for Vec<ConstantValue> {
    fn from(c: CaptureConstants) -> Self {
        vec![
            ConstantValue::uint(ConstSlot(0), c.hidden),
            ConstantValue::uint(ConstSlot(1), c.slot),
            ConstantValue::uint(ConstSlot(2), c.slots),
            ConstantValue::uint(ConstSlot(3), c.elements),
        ]
    }
}

/// Every `dflash2.metal` kernel's constants: the drafter's geometry and the convolution stage a
/// `draft_conv` runs (`DRAFT_*`, slots 0-10; every kernel of the shader sets every slot).
pub struct DraftConstants {
    pub rows: u32,
    pub query_heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub window: u32,
    pub splits: u32,
    pub hidden: u32,
    pub conv_group: u32,
    /// 0: the convolution's preparing stage; 1: its finishing stage (and every other kernel).
    pub conv_finish: u32,
    pub rms_eps: f32,
    pub scale: f32,
}

impl From<DraftConstants> for Vec<ConstantValue> {
    fn from(c: DraftConstants) -> Self {
        vec![
            ConstantValue::uint(ConstSlot(0), c.rows),
            ConstantValue::uint(ConstSlot(1), c.query_heads),
            ConstantValue::uint(ConstSlot(2), c.kv_heads),
            ConstantValue::uint(ConstSlot(3), c.head_dim),
            ConstantValue::uint(ConstSlot(4), c.window),
            ConstantValue::uint(ConstSlot(5), c.splits),
            ConstantValue::uint(ConstSlot(6), c.hidden),
            ConstantValue::uint(ConstSlot(7), c.conv_group),
            ConstantValue::uint(ConstSlot(8), c.conv_finish),
            ConstantValue::float(ConstSlot(9), c.rms_eps),
            ConstantValue::float(ConstSlot(10), c.scale),
        ]
    }
}

/// Every `dflash2_select.metal` kernel's constants (`SELECT_*`, slots 0-31).
pub struct DraftSelectConstants {
    pub vocab: u32,
    pub rows: u32,
    pub positions: u32,
    pub candidates: u32,
    pub rank: u32,
    pub shards: u32,
    pub verify: crate::dflash2::VerifyRows,
    pub lanes: u32,
}

impl From<DraftSelectConstants> for Vec<ConstantValue> {
    fn from(c: DraftSelectConstants) -> Self {
        use crate::dflash2::{PROPOSALS, VERIFY_COST_SEED, VerifyRows};
        // Every row leaves the budget's slots unread.
        let (every_row, slope, intercepts, cost) = match c.verify {
            VerifyRows::EveryRow => (true, 0.0, [0.0; PROPOSALS as usize], VERIFY_COST_SEED),
            VerifyRows::Budget {
                slope,
                intercepts,
                cost,
            } => (false, slope, intercepts, cost),
        };
        let mut v = vec![
            ConstantValue::uint(ConstSlot(0), c.vocab),
            ConstantValue::uint(ConstSlot(1), c.rows),
            ConstantValue::uint(ConstSlot(2), c.positions),
            ConstantValue::uint(ConstSlot(3), c.candidates),
            ConstantValue::uint(ConstSlot(4), c.rank),
            ConstantValue::uint(ConstSlot(5), c.shards),
            ConstantValue::float(ConstSlot(6), slope),
        ];
        v.extend(
            (7u16..)
                .zip(intercepts)
                .map(|(s, x)| ConstantValue::float(ConstSlot(s), x)),
        );
        v.push(ConstantValue::uint(ConstSlot(14), c.lanes));
        v.push(ConstantValue::boolean(ConstSlot(15), every_row));
        v.extend(
            (16u16..)
                .zip(cost.0.into_iter().flatten())
                .map(|(s, x)| ConstantValue::float(ConstSlot(s), x)),
        );
        v
    }
}
