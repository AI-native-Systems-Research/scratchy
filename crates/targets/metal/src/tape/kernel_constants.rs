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
    AttnDebugMode, AttnScale, AttnWindow, BlockSize, BlocksPerChunk, BucketM, HeadDim, HiddenSize,
    IntermediateSize, KDim, KDimI32, KPartitionSizeI32, MDimI32, MaxBlocksPerSeq, NDim, NDimI32,
    NumKvHeads, NumQHeads, QSize, RmsNormEps, RopePairOff, RotDim, SplitK, TqCodeBits,
};

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

impl From<RmsNormConstants> for Vec<ConstantValue> {
    fn from(c: RmsNormConstants) -> Self {
        vec![
            ConstantValue::uint(ConstSlot(0), c.bucket_m.get()),
            ConstantValue::uint(ConstSlot(1), c.q_size.get()),
            ConstantValue::float(ConstSlot(2), c.rms_norm_eps.get()),
            ConstantValue::float(ConstSlot(3), c.weight_offset),
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
        v
    }
}

// ── FusedQkvRopeCache (dense BF16/F16) ─────────────────────────────

/// `KernelId::FusedQkvRopeCache`
/// (`fused_qkv_rope_cache_<dtype>_specialized`).
pub struct FusedQkvRopeCacheConstants {
    pub q_size: QSize,
    pub num_q_heads: NumQHeads,
    pub num_kv_heads: NumKvHeads,
    pub head_dim: HeadDim,
    pub rot_dim: RotDim,
    pub block_size: BlockSize,
    pub bucket_m: BucketM,
    pub blocks_per_chunk: BlocksPerChunk,
    /// Spans rope-on-read: `Some(1)` sets FQRC_ROPE_ON_READ (slot 8) so the
    /// fused kernel stores K unrotated for slot_mapping-bit-31 blocks. `None`
    /// omits the const → default-off (byte-identical non-spans).
    pub rope_on_read: Option<u32>,
}

impl From<FusedQkvRopeCacheConstants> for Vec<ConstantValue> {
    fn from(c: FusedQkvRopeCacheConstants) -> Self {
        let mut v = vec![
            ConstantValue::uint(ConstSlot(0), c.q_size.get()),
            ConstantValue::uint(ConstSlot(1), c.num_q_heads.get()),
            ConstantValue::uint(ConstSlot(2), c.num_kv_heads.get()),
            ConstantValue::uint(ConstSlot(3), c.head_dim.get()),
            ConstantValue::uint(ConstSlot(4), c.rot_dim.get()),
            ConstantValue::uint(ConstSlot(5), c.block_size.get()),
            ConstantValue::uint(ConstSlot(6), c.bucket_m.get()),
            ConstantValue::uint(ConstSlot(7), c.blocks_per_chunk.get()),
        ];
        if let Some(ror) = c.rope_on_read {
            v.push(ConstantValue::uint(ConstSlot(8), ror));
        }
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
    pub max_blocks: MaxBlocksPerSeq,
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
    /// NeoX rope (`rot_dim == head_dim`); the lowering sets it only then,
    /// and the env knob `SPANS_CORESIDENT=0` forces it off for A/B. `None`
    /// (every non-spans dispatch, and the A/B-off case) → byte-identical
    /// emitted Vec and the shader keeps the contiguous-slice + shuffle path.
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
            ConstantValue::uint(ConstSlot(5), c.max_blocks.get()),
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

/// `KernelId::AttentionViaCacheTq`: the constant its `AttentionViaCache`
/// twin's set gains — `ATTN_TQ_BITS` (slot 13), which switches the kernel to
/// reading the TurboQuant packed store.
pub struct AttentionViaCacheTqConstants {
    pub bits: TqCodeBits,
}

impl From<AttentionViaCacheTqConstants> for Vec<ConstantValue> {
    fn from(c: AttentionViaCacheTqConstants) -> Self {
        vec![ConstantValue::uint(ConstSlot(13), c.bits.get())]
    }
}

/// `KernelId::TqStageRotated` (`tq_stage_rotated_<dtype>`, attention.metal
/// slots): the layer's KV geometry, the codebook width, and — for K under
/// rope-on-read — the span re-rope geometry (slots 8/9/10).
pub struct TqStageConstants {
    pub head_dim: HeadDim,
    pub num_kv_heads: NumKvHeads,
    pub block_size: BlockSize,
    pub max_blocks: MaxBlocksPerSeq,
    pub blocks_per_chunk: BlocksPerChunk,
    pub bits: TqCodeBits,
    pub rot_dim: Option<RotDim>,
    pub pair_off: Option<RopePairOff>,
    pub rope_on_read: Option<u32>,
}

impl From<TqStageConstants> for Vec<ConstantValue> {
    fn from(c: TqStageConstants) -> Self {
        let mut v = vec![
            ConstantValue::uint(ConstSlot(0), c.head_dim.get()),
            ConstantValue::uint(ConstSlot(2), c.num_kv_heads.get()),
            ConstantValue::uint(ConstSlot(4), c.block_size.get()),
            ConstantValue::uint(ConstSlot(5), c.max_blocks.get()),
            ConstantValue::uint(ConstSlot(6), c.blocks_per_chunk.get()),
        ];
        push_rope_on_read_consts(&mut v, c.rot_dim, c.pair_off, c.rope_on_read);
        v.push(ConstantValue::uint(ConstSlot(13), c.bits.get()));
        v
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
    pub max_blocks: MaxBlocksPerSeq,
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
            ConstantValue::uint(ConstSlot(5), c.max_blocks.get()),
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

// ── MLX-affine QMV (decode matvec) ────────────────────────────────

/// `KernelId::AffineQmvQuad` / `AffineQmvFast` / `AffineQmv`
/// (`quantized_qmv` library; constants declared as signed `int`).
pub struct AffineQmvConstants {
    pub k: KDimI32,
    pub n: NDimI32,
}

impl From<AffineQmvConstants> for Vec<ConstantValue> {
    fn from(c: AffineQmvConstants) -> Self {
        vec![
            ConstantValue::int(ConstSlot(0), c.k.get()),
            ConstantValue::int(ConstSlot(1), c.n.get()),
        ]
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
}

impl From<AffineQmmTConstants> for Vec<ConstantValue> {
    fn from(c: AffineQmmTConstants) -> Self {
        vec![
            ConstantValue::int(ConstSlot(0), c.k.get()),
            ConstantValue::int(ConstSlot(1), c.n.get()),
            ConstantValue::int(ConstSlot(2), c.m.get()),
        ]
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
}

impl From<AffineQmmTSplitKConstants> for Vec<ConstantValue> {
    fn from(c: AffineQmmTSplitKConstants) -> Self {
        vec![
            ConstantValue::int(ConstSlot(0), c.k.get()),
            ConstantValue::int(ConstSlot(1), c.n.get()),
            ConstantValue::int(ConstSlot(2), c.m.get()),
            ConstantValue::int(ConstSlot(3), c.k_partition_size.get()),
        ]
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
}

impl From<AffineEmbedConstants> for Vec<ConstantValue> {
    fn from(c: AffineEmbedConstants) -> Self {
        vec![ConstantValue::uint(ConstSlot(0), c.hidden_size.get())]
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

// ── Synth-* (compiler-emitted megakernels) ────────────────────────

/// `KernelId::SynthPreAttn` / `KernelId::SynthMlpPreDown` /
/// `KernelId::SynthGateUpSiluMul` — the macro-emitted megakernels.
///
/// All three bake every dim into MSL `constant constexpr` literals at
/// synth time. Only the per-bucket `M` stays a function constant
/// (slot 0).
pub struct SynthMegakernelConstants {
    pub bucket_m: BucketM,
}

impl From<SynthMegakernelConstants> for Vec<ConstantValue> {
    fn from(c: SynthMegakernelConstants) -> Self {
        vec![ConstantValue::uint(ConstSlot(0), c.bucket_m.get())]
    }
}

// Keep `KDim` / `NDim` re-exported even though the int32 siblings
// (`KDimI32`/`NDimI32`) cover the qmv/qmm_t shaders today. SplitK
// reduce and the synth-* family need the unsigned form.
#[allow(dead_code)]
const _: fn() = || {
    let _ = std::marker::PhantomData::<KDim>;
    let _ = std::marker::PhantomData::<NDim>;
};
