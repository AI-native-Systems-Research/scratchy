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
    AttnDebugMode, AttnScale, AttnWindow, BlockSize, BlocksPerChunk, BucketM, ElementCount,
    HeadDim, HiddenSize, IntermediateSize, KDim, KDimI32, KPartitionSizeI32, MDimI32, NDim, NDimI32,
    NumExperts, NumKvHeads, NumQHeads, QSize, RmsNormEps, RopePairOff, RotDim, SplitK, TopK,
    TqCodeBits,
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

/// The threads of an RMSNorm row's threadgroup: every norm command dispatches this many, and its
/// kernel takes it compiled in (slot 4), so each thread's share of the row is a count the compiler
/// knows.
pub const NORM_THREADS: u32 = 1024;

impl From<RmsNormConstants> for Vec<ConstantValue> {
    fn from(c: RmsNormConstants) -> Self {
        vec![
            ConstantValue::uint(ConstSlot(0), c.bucket_m.get()),
            ConstantValue::uint(ConstSlot(1), c.q_size.get()),
            ConstantValue::float(ConstSlot(2), c.rms_norm_eps.get()),
            ConstantValue::float(ConstSlot(3), c.weight_offset),
            ConstantValue::uint(ConstSlot(4), NORM_THREADS),
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

/// `KernelId::VisionLoadPixels` (`copy_rows_<T>`, `elementwise.metal` slot 4): the elements the
/// staged buffer holds.
pub struct CopyRowsConstants {
    pub elements: ElementCount,
}

impl From<CopyRowsConstants> for Vec<ConstantValue> {
    fn from(c: CopyRowsConstants) -> Self {
        vec![ConstantValue::uint(ConstSlot(4), c.elements.get())]
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
    pub head_dim: HeadDim,
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
    /// The TurboQuant encode folded in (slots 10..=14), if the writer writes the packed store.
    pub tq: Option<RopeTqConstants>,
}

/// A KV writer's folded-in TurboQuant encode (`rope.metal`'s `ROPE_TQ_*`): the codebook's bits,
/// the packing from them as [`TqCompressConstants`] takes it, and each operand's offset.
#[derive(Clone, Copy)]
pub struct RopeTqConstants {
    pub bits: TqCodeBits,
    pub k_offset: TqOffset,
    pub v_offset: TqOffset,
}

impl RopeTqConstants {
    fn push(tq: Option<Self>, head_dim: HeadDim, v: &mut Vec<ConstantValue>) {
        use scratchy_layers::turboquant::{packed_dim, vals_per_word};
        let Some(t) = tq else {
            return;
        };
        let (dim, bits) = (head_dim.get() as usize, t.bits.get());
        v.extend([
            ConstantValue::uint(ConstSlot(10), bits),
            ConstantValue::uint(ConstSlot(11), vals_per_word(bits) as u32),
            ConstantValue::uint(ConstSlot(12), packed_dim(dim, bits) as u32),
            ConstantValue::uint(ConstSlot(13), t.k_offset as u32),
            ConstantValue::uint(ConstSlot(14), t.v_offset as u32),
        ]);
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
        RopeTqConstants::push(c.tq, c.head_dim, &mut v);
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
    /// The TurboQuant encode folded in: see [`RopeAppendNormedConstants`].
    pub tq: Option<RopeTqConstants>,
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
        RopeTqConstants::push(c.tq, c.head_dim, &mut v);
        v
    }
}

// ── AttentionViaCache (decode) ─────────────────────────────────────

/// `KernelId::AttentionViaCache`
/// (`attention_via_cache_v2_<dtype>_specialized`).
pub struct AttentionViaCacheConstants {
    pub head_dim: HeadDim,
    pub num_q_heads: NumQHeads,
    pub num_kv_heads: NumKvHeads,
    pub attn_scale: AttnScale,
    pub block_size: BlockSize,
    pub blocks_per_chunk: BlocksPerChunk,
    /// Sliding-window width (`ATTN_WINDOW`, slot 7). `0` = disabled
    /// (full attention); the sliding lowering arm passes
    /// `W::SLIDING_WINDOW`.
    pub window: AttnWindow,
    /// Spans rope-on-read (slots 8/9/10): rotary dim, NeoX pairing
    /// offset, and the master 0/1 switch. `None` on every non-spans
    /// dispatch → the emitted Vec is byte-identical to today (8 consts)
    /// and the in-shader `is_function_constant_defined` guard folds the
    /// rotation away. `Some` only when `W::ROPE_ON_READ`.
    pub rot_dim: Option<RotDim>,
    pub pair_off: Option<RopePairOff>,
    pub rope_on_read: Option<u32>,
    /// Co-resident NeoX-pair lane layout for the decode rope-on-read path
    /// (slot 12, `ATTN_PAIR_CORESIDENT`). When `Some(1)` each lane owns its
    /// NeoX pairs `{d, d+half_dim}` so the on-read rope is in-lane (no
    /// `simd_shuffle`, no `k_pair[]` staging array). Only valid for full
    /// NeoX rope (`rot_dim == head_dim`); the lowering sets it only then.
    /// `None` (every non-spans dispatch) → byte-identical emitted Vec and the
    /// shader keeps the contiguous-slice + shuffle path.
    pub pair_coresident: Option<u32>,
}

impl From<AttentionViaCacheConstants> for Vec<ConstantValue> {
    fn from(c: AttentionViaCacheConstants) -> Self {
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
        if let Some(pc) = c.pair_coresident {
            v.push(ConstantValue::uint(ConstSlot(12), pc));
        }
        v
    }
}

/// `KernelId::AttentionViaCacheTq`: the constants its `AttentionViaCache`
/// twin's set gains — `ATTN_TQ_BITS` (slot 13), which switches the kernel to
/// reading the TurboQuant packed store, and `ATTN_TQ_K_BIAS` /
/// `ATTN_TQ_V_BIAS` (slots 14 / 15), set when the codes hold that operand
/// minus its projection bias (bound at buffers 14 / 15).
pub struct AttentionViaCacheTqConstants {
    pub bits: TqCodeBits,
    pub k_bias: bool,
    pub v_bias: bool,
}

impl AttentionViaCacheTqConstants {
    /// `ATTN_TQ_HEADS`: the query heads one threadgroup serves, the tape variant's.
    pub const HEADS: ConstSlot = ConstSlot(16);
}

impl From<AttentionViaCacheTqConstants> for Vec<ConstantValue> {
    fn from(c: AttentionViaCacheTqConstants) -> Self {
        let mut v = vec![ConstantValue::uint(ConstSlot(13), c.bits.get())];
        v.extend(c.k_bias.then(|| ConstantValue::uint(ConstSlot(14), 1)));
        v.extend(c.v_bias.then(|| ConstantValue::uint(ConstSlot(15), 1)));
        v.push(ConstantValue::tq_heads(AttentionViaCacheTqConstants::HEADS));
        v
    }
}

/// `KernelId::TqStageRotated` (`tq_stage_rotated_<dtype>`, attention.metal
/// slots): the layer's KV geometry, the codebook width, and — for K under
/// rope-on-read — the span re-rope geometry (slots 8/9/10).
pub struct TqStageConstants {
    pub head_dim: HeadDim,
    pub num_kv_heads: NumKvHeads,
    pub block_size: BlockSize,
    pub blocks_per_chunk: BlocksPerChunk,
    pub bits: TqCodeBits,
    pub rot_dim: Option<RotDim>,
    pub pair_off: Option<RopePairOff>,
    pub rope_on_read: Option<u32>,
    /// `ATTN_TQ_K_BIAS` / `ATTN_TQ_V_BIAS` (slots 14 / 15): the staged
    /// operand's codes hold it minus its projection bias (bound at buffer 10).
    pub k_bias: bool,
    pub v_bias: bool,
    /// `ATTN_TQ_STAGE_PASS` (slot 17): which rows this dispatch stages.
    pub pass: TqStagePass,
}

/// The rows one `tq_stage_rotated` dispatch stages. A step stages its new rows,
/// then its cached ones: a row new for one sequence can be a prefix-cache hit
/// for another in the same step, and both rewrite it in place.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TqStagePass {
    /// The step's new rows, read from the cache their writer just filled.
    New = 1,
    /// Cached rows, decoded from the packed store.
    Cached = 2,
}

impl From<TqStageConstants> for Vec<ConstantValue> {
    fn from(c: TqStageConstants) -> Self {
        let mut v = vec![
            ConstantValue::uint(ConstSlot(0), c.head_dim.get()),
            ConstantValue::uint(ConstSlot(2), c.num_kv_heads.get()),
            ConstantValue::uint(ConstSlot(4), c.block_size.get()),
            ConstantValue::kv_cap(ConstSlot(5)),
            ConstantValue::uint(ConstSlot(6), c.blocks_per_chunk.get()),
        ];
        push_rope_on_read_consts(&mut v, c.rot_dim, c.pair_off, c.rope_on_read);
        v.push(ConstantValue::uint(ConstSlot(13), c.bits.get()));
        v.extend(c.k_bias.then(|| ConstantValue::uint(ConstSlot(14), 1)));
        v.extend(c.v_bias.then(|| ConstantValue::uint(ConstSlot(15), 1)));
        v.push(ConstantValue::uint(ConstSlot(17), c.pass as u32));
        v
    }
}

/// `tq_compress_paged[_bf16]` (turboquant.metal), the standalone encode the kernel tests fill
/// packed stores with: the KV geometry of the layer it quantizes, the codebook width (and from it
/// the packing), and the offset it removes first.
pub struct TqCompressConstants {
    pub head_dim: HeadDim,
    pub bits: TqCodeBits,
    pub num_kv_heads: NumKvHeads,
    pub block_size: BlockSize,
    pub blocks_per_chunk: BlocksPerChunk,
    pub writeback: TqWriteback,
    pub offset: TqOffset,
    /// The rotated offset's rope geometry ([`TqOffset::RotatedBias`]); zero otherwise.
    pub rot_dim: RotDim,
    pub pair_off: RopePairOff,
}

/// Whether `tq_compress_paged` writes its lossy dequant back into the pool.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TqWriteback {
    /// The pool keeps the raw vector.
    Raw = 0,
    Dequantized = 1,
}

/// The offset `tq_compress_paged` removes before quantizing (`turboquant_offset.h`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TqOffset {
    None = 0,
    /// The operand's projection bias.
    Bias = 1,
    /// The bias rotated to the key's position.
    RotatedBias = 2,
}

impl From<TqCompressConstants> for Vec<ConstantValue> {
    fn from(c: TqCompressConstants) -> Self {
        use scratchy_layers::turboquant::{packed_dim, vals_per_word};
        let (dim, bits) = (c.head_dim.get(), c.bits.get());
        vec![
            ConstantValue::uint(ConstSlot(0), dim),
            ConstantValue::uint(ConstSlot(1), bits),
            ConstantValue::uint(ConstSlot(2), vals_per_word(bits) as u32),
            ConstantValue::uint(ConstSlot(3), packed_dim(dim as usize, bits) as u32),
            ConstantValue::uint(ConstSlot(4), 1 << bits),
            ConstantValue::float(ConstSlot(5), 1.0 / (dim as f32).sqrt()),
            ConstantValue::uint(ConstSlot(6), c.num_kv_heads.get()),
            ConstantValue::uint(ConstSlot(7), c.block_size.get()),
            ConstantValue::uint(ConstSlot(8), c.blocks_per_chunk.get()),
            ConstantValue::uint(ConstSlot(9), c.writeback as u32),
            ConstantValue::uint(ConstSlot(10), c.offset as u32),
            ConstantValue::uint(ConstSlot(11), c.rot_dim.get()),
            ConstantValue::uint(ConstSlot(12), c.pair_off.get()),
        ]
    }
}

/// `KernelId::TqRotateRows` (`tq_{rotate,unrotate}_rows_<dtype>`).
pub struct TqRotateRowsConstants {
    pub head_dim: HeadDim,
    pub num_q_heads: NumQHeads,
}

impl From<TqRotateRowsConstants> for Vec<ConstantValue> {
    fn from(c: TqRotateRowsConstants) -> Self {
        vec![
            ConstantValue::uint(ConstSlot(0), c.head_dim.get()),
            ConstantValue::uint(ConstSlot(1), c.num_q_heads.get()),
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
/// (slot 5) to XOR them back.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AffineCodes {
    AsWritten,
    Offset8,
}

impl AffineCodes {
    pub fn of(profile: Option<&crate::targets::MetalTargetProfile>, bits: u32) -> Self {
        AffineCodesTarget::of(profile).for_bits(bits)
    }

    /// Slot 5, set only on `Offset8` so every other pipeline keeps its key.
    pub fn constant(self) -> Option<ConstantValue> {
        match self {
            AffineCodes::Offset8 => Some(ConstantValue::boolean(ConstSlot(5), true)),
            AffineCodes::AsWritten => None,
        }
    }

    /// What a command whose kernel reads codes declares, from its baked
    /// constants (slot 5 is `bool` only on such kernels).
    pub fn of_constants(constants: &[ConstantValue]) -> Self {
        match AffineCodes::Offset8.constant() {
            Some(c) if constants.contains(&c) => AffineCodes::Offset8,
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
            ConstantValue::int(ConstSlot(0), c.k.get()),
            ConstantValue::int(ConstSlot(1), c.n.get()),
        ];
        v.extend(c.codes.constant());
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

/// A one-row matvec's [`QmvEnds`](crate::tape::step::QmvEnds), compiled in: its norm's epsilon
/// (slot 8) and gain offset (9), whether its rows add into the residual (10), take a bias (11), and
/// their scale (12).
impl From<crate::tape::step::QmvEnds> for Vec<ConstantValue> {
    fn from(e: crate::tape::step::QmvEnds) -> Self {
        let norm = e.norm.into_iter().flat_map(|n| {
            [
                ConstantValue::float(ConstSlot(8), n.eps.0),
                ConstantValue::float(ConstSlot(9), n.offset.0),
            ]
        });
        let residual = e
            .residual
            .then(|| ConstantValue::boolean(ConstSlot(10), true));
        let bias = e.bias.map(|_| ConstantValue::boolean(ConstSlot(11), true));
        let scale = e.scale.map(|s| ConstantValue::float(ConstSlot(12), s.0));
        norm.chain(residual).chain(bias).chain(scale).collect()
    }
}

/// `KernelId::NormedGemv` (`gemv_normed_<T>_s_<G>`, `gemm.metal`): the GEMM's one row (slots 0 /
/// 1 / 2: M, N, K) and its folded norm's epsilon (5).
pub struct NormedGemvConstants {
    pub n: NDim,
    pub k: KDim,
    pub eps: crate::tape::step::Eps,
}

impl From<NormedGemvConstants> for Vec<ConstantValue> {
    fn from(c: NormedGemvConstants) -> Self {
        vec![
            ConstantValue::uint(ConstSlot(0), 1),
            ConstantValue::uint(ConstSlot(1), c.n.get()),
            ConstantValue::uint(ConstSlot(2), c.k.get()),
            ConstantValue::float(ConstSlot(5), c.eps.0),
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
/// [`AffineGatherQmvConstants`]) and `KernelId::AffineQmvGated` (`affine_qmv_gated[_fast]_*`,
/// `Q` = [`AffineQmvConstants`]): the gate projection's matvec constants (the up's are the same)
/// and the activation (slot 3: 0 SiLU, 1 GELU).
pub struct AffineGatedQmvConstants<Q> {
    pub qmv: Q,
    pub act: crate::tape::step::GatedAct,
}

impl<Q: Into<Vec<ConstantValue>>> From<AffineGatedQmvConstants<Q>> for Vec<ConstantValue> {
    fn from(c: AffineGatedQmvConstants<Q>) -> Self {
        use crate::tape::step::GatedAct;
        let act = match c.act {
            GatedAct::Silu => 0,
            GatedAct::Gelu => 1,
        };
        let mut v: Vec<ConstantValue> = c.qmv.into();
        v.push(ConstantValue::int(ConstSlot(3), act));
        v
    }
}

/// `KernelId::MoeDownCombine` (`affine_gather_qmv_combine[_fast]_*`): the down projection's
/// [`AffineQmvConstants`] and the experts each token chose (slot 2), whose rows it combines.
pub struct AffineCombineQmvConstants {
    pub qmv: AffineQmvConstants,
    pub top_k: TopK,
}

impl From<AffineCombineQmvConstants> for Vec<ConstantValue> {
    fn from(c: AffineCombineQmvConstants) -> Self {
        let mut v: Vec<ConstantValue> = c.qmv.into();
        v.push(ConstantValue::int(ConstSlot(2), c.top_k.get() as i32));
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
            ConstantValue::int(ConstSlot(0), c.k.get()),
            ConstantValue::int(ConstSlot(1), c.n.get()),
            ConstantValue::int(ConstSlot(7), c.m.get()),
        ];
        v.extend(c.codes.constant());
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
/// (slot 0), the top-k it keeps (slot 1), and its program — a softmax first (slot 2), the scores'
/// scale (slot 3, set only when they scale), what follows (slot 4: 0 nothing, 1 softmax,
/// 2 renorm), and the per-expert scale (slot 5).
pub struct MoeRouteConstants {
    pub experts: NumExperts,
    pub top_k: TopK,
    pub program: crate::tape::step::RouteProgram,
}

impl From<MoeRouteConstants> for Vec<ConstantValue> {
    fn from(c: MoeRouteConstants) -> Self {
        use crate::tape::step::RoutePost;
        let p = c.program;
        let post = match p.post {
            RoutePost::None => 0,
            RoutePost::Softmax => 1,
            RoutePost::Renorm => 2,
        };
        let mut v = vec![
            ConstantValue::int(ConstSlot(0), c.experts.get() as i32),
            ConstantValue::int(ConstSlot(1), c.top_k.get() as i32),
            ConstantValue::int(ConstSlot(2), i32::from(p.pre_softmax)),
        ];
        v.extend(p.scale.map(|s| ConstantValue::float(ConstSlot(3), s.0)));
        v.push(ConstantValue::int(ConstSlot(4), post));
        v.push(ConstantValue::int(
            ConstSlot(5),
            i32::from(p.expert_scale.is_some()),
        ));
        v
    }
}

/// A routed expert kernel's routing (`MetalFusion::MoeRouted`, `quantized_qmv.metal` slots
/// 13-17): the experts, and the [`MoeRouteConstants`] program — a softmax over the experts
/// first, the scores' scale, their last step, the per-expert scale.
pub struct RoutedConstants {
    pub experts: NumExperts,
    pub program: crate::tape::step::RouteProgram,
}

impl From<RoutedConstants> for Vec<ConstantValue> {
    fn from(c: RoutedConstants) -> Self {
        use crate::tape::step::RoutePost;
        let p = c.program;
        let post = match p.post {
            RoutePost::None => 0,
            RoutePost::Softmax => 1,
            RoutePost::Renorm => 2,
        };
        let mut v = vec![
            ConstantValue::int(ConstSlot(13), c.experts.get() as i32),
            ConstantValue::boolean(ConstSlot(14), p.pre_softmax),
        ];
        v.extend(p.scale.map(|s| ConstantValue::float(ConstSlot(15), s.0)));
        v.push(ConstantValue::int(ConstSlot(16), post));
        v.push(ConstantValue::boolean(
            ConstSlot(17),
            p.expert_scale.is_some(),
        ));
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
            ConstantValue::int(ConstSlot(0), c.k.get()),
            ConstantValue::int(ConstSlot(1), c.n.get()),
            ConstantValue::int(ConstSlot(2), c.m.get()),
        ];
        v.extend(c.codes.constant());
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
            ConstantValue::int(ConstSlot(0), c.k.get()),
            ConstantValue::int(ConstSlot(1), c.n.get()),
            ConstantValue::int(ConstSlot(2), c.m.get()),
            ConstantValue::int(ConstSlot(3), c.k_partition_size.get()),
        ];
        v.extend(c.codes.constant());
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
        v.extend(c.codes.constant());
        v
    }
}

// ── GatherLastToken / ScatterFirstToLastRow ───────────────────────

/// `KernelId::GatherLastToken` and `KernelId::ScatterFirstToLastRow`
/// share the single-`row_stride` constant layout
/// (`gather_last_token_<dtype>_specialized` and
/// `scatter_first_to_last_row_<dtype>_specialized`).
pub struct GatherLastTokenConstants {
    pub row_stride: HiddenSize,
}

impl From<GatherLastTokenConstants> for Vec<ConstantValue> {
    fn from(c: GatherLastTokenConstants) -> Self {
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

/// `chain_advance.metal` slot 0: the model's KV block size.
pub struct ChainAdvanceConstants {
    pub block_size: BlockSize,
}

impl From<ChainAdvanceConstants> for Vec<ConstantValue> {
    fn from(c: ChainAdvanceConstants) -> Self {
        vec![ConstantValue::uint(ConstSlot(0), c.block_size.get())]
    }
}
