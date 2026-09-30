// SPDX-License-Identifier: Apache-2.0
//! Buffer-pointer-free Metal tape produced by the lowering pass.
//!
//! `LoweredMetalTape` is computed once per `(model variant, bucket)` and
//! shared across every `MetalWorker` in the pool via `Arc`. It carries
//! pipeline keys, dispatch shapes, scalar constants, and slot ids only —
//! never raw `metal::Buffer` pointers. Each `MetalWorker` instantiates
//! the tape against its own arena at worker init time, resolving
//! `Binding::ArenaSlot` against `arena[slot]` and recording the result
//! into a per-bucket MTL4 dispatch tape.
//!
//! The lowered tape is structurally a sequence of `LoweredCommand`s,
//! with `Loop` instructions already statically unrolled by the lowering
//! pass (CUDA's runtime `Loop` interpreter has no analogue on Metal —
//! the per-bucket dispatch is fully baked).

use crate::tape::constants::{ConstSlot, ConstantType, ConstantValue};
use crate::tape::ids::{BucketM, LayerId, NumTokens, SourceIx};

/// One-of identifier for the kernel a `LoweredCommand` invokes.
///
/// The `MetalWorker` resolves `(KernelId, bucket)` against a
/// `SpecializedPipelineCache` (Phase 5.B) to find the right
/// `MTLComputePipelineState`. Each variant corresponds to one Metal
/// kernel under `crates/targets/metal/shaders/` and one
/// recorder under `crates/targets/metal/src/instruction_executor/`.
///
/// The set is closed and small on purpose: the TinyLlama-1.1B critical
/// path covers ~13 of these, with more added as additional models come
/// online.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize)]
pub enum KernelId {
    /// Token embedding gather: rows of `embed_tokens.weight` indexed
    /// by `input_ids`. Output `[num_tokens, hidden_size]`.
    Embed,
    /// Standalone RMSNorm: `out = weight * x / sqrt(mean(x²) + eps)`.
    RmsNorm,
    /// Unit-gain RMSNorm (no weight — mlx `RMSNormNoScale`, Gemma4
    /// `v_norm`). Maps to `rmsnorm_unit_<dtype>_specialized` in
    /// `rmsnorm.metallib`.
    RmsNormUnit,
    /// Multiply by a loaded `[1]`-shaped weight (Gemma4
    /// `layer_scalar`). Maps to `scalar_weight_mul_<dtype>_specialized`
    /// in `elementwise.metallib`.
    ScalarWeightMul,
    NormAddScalarMul,
    RopeAppendNormed,
    /// Fused residual-add + RMSNorm: writes `residual += delta` and
    /// publishes `weight * residual / sqrt(mean(residual²) + eps)`.
    FusedAddRmsNorm,
    /// Generic dense GEMM: `y = x @ W^T`. Backed by Metal Performance
    /// Shaders' `matmul2d` (or a hand-rolled tile shader once the
    /// fused kernels land).
    Gemm,
    /// Fused gate-up SwiGLU MLP: `silu(gate) * up` after a single GEMM
    /// produces the stacked `[gate; up]` activation. TinyLlama-1.1B's
    /// MLP path.
    FusedGateUpSiluMul,
    /// Apply RoPE to a fresh QKV projection and write K/V to the
    /// paged KV cache at the per-request slot. Output: rotated Q
    /// only (K/V are sunk into cache).
    RopeAppend,
    /// Decode-bucket attention reading from the paged KV cache.
    /// Single-query-token-per-sequence path.
    AttentionViaCache,
    /// Prefill-bucket attention reading from the paged KV cache.
    /// Faithful MLX `sdpa_vector` port: 1 Q per threadgroup,
    /// online softmax + per-simdgroup K-axis split (same outer
    /// structure as the decode kernel `AttentionViaCache`), with K/V
    /// access through `block_table` indirection. K-axis covers the
    /// FULL `seqused_k[seq]` (prefix + new), and the per-Q causal
    /// mask shifts by `(seqused_k[seq] - new_q_for_seq)` to account
    /// for prior cached prefix. Required for chunked prefill,
    /// prefix caching, mixed prefill/decode batches, and multi-turn
    /// chat continuation. The only prefill kernel emitted by metal
    /// post-Phase B (the legacy contiguous and non-paged sdpa
    /// kernels were retired once `Instruction::AttentionPrefillPaged`
    /// became the universal metal prefill emitter).
    AttentionPrefillSdpaPaged,
    /// Spans rope-on-read (NAX prefill): ropes one request+layer's K from
    /// the paged cache ONCE into the shared `Binding::RopedKScratch` buffer,
    /// so the NAX attention that follows reads pre-roped K with no per-tile
    /// rotation. Emitted only when `use_nax && W::ROPE_ON_READ`. Maps to
    /// `rope_once_nax_<dtype>_bd<head_dim>_bs16` in the `attention_steel_nax_paged`
    /// library.
    RopeOnceNax,
    /// Spans rope-on-read (SIMDGROUP steel prefill): the simdgroup twin of
    /// `RopeOnceNax`. Ropes one request+layer's K from the paged cache ONCE
    /// into the shared `Binding::RopedKScratch` buffer, so the
    /// `attention_steel_paged` attention that follows reads pre-roped K with no
    /// per-tile rotation. Emitted only when `use_steel && !use_nax &&
    /// W::ROPE_ON_READ`. Maps to `rope_once_steel_<dtype>_bd<head_dim>_bs16` in
    /// the `attention_steel_paged` library (covers the simdgroup head dims
    /// 64/96/128/256, incl. SmolLM's hd64).
    RopeOnceSteel,
    /// Spans rope-on-read (GQA-COOPERATIVE shared prefill): the gqa_shared twin
    /// of `RopeOnceNax`/`RopeOnceSteel`. Ropes one request+layer's K from the
    /// paged cache ONCE into the shared `Binding::RopedKScratch` buffer, so the
    /// `attention_prefill_sdpa_gqa_shared_*` attention that follows reads
    /// pre-roped K (no per-tile smem rotation). Emitted only when
    /// `use_gqa_shared && W::ROPE_ON_READ`. Covers the head dims with no steel
    /// instantiation that fall to gqa_shared (gemma4 global: hd 512, bs 32) —
    /// the symbol reads head_dim/block_size from function constants, so one
    /// pair (`rope_once_gqa_shared_<dtype>_specialized`, in the `attention`
    /// library) covers any shape the gqa_shared kernel takes.
    RopeOnceGqaShared,
    /// hd512 unfused attention (env `SCRATCHY_HD512_UNFUSED`): kv-head-major
    /// gather of K (rope-on-read) into the attn-unfused scratch, STATIC row
    /// stride (max_blocks*block_size) so per-head GEMM offsets are bakeable.
    AttnGatherKRope,
    /// hd512 unfused: kv-head-major TRANSPOSED gather of V → [kv, head_dim, kv_len]
    /// so PV's `C = A @ B^T` gets B = V^T directly.
    AttnGatherVCopyT,
    /// hd512 unfused: Q `[total_q, nh, hd]` (token-major) → `[nh, Lq, hd]`
    /// (head-major) bf16, so each head's QK^T A-operand is contiguous.
    AttnQConvert,
    /// hd512 unfused: O `[nh, Lq, hd]` (head-major) → `[total_q, nh, hd]`
    /// (token-major) bf16 — the layout o_proj expects (same as gqa_shared).
    AttnOConvert,
    /// hd512 unfused: fused scale + chunked-prefill causal mask + row softmax
    /// over scores `[Lq, kv_len]` (one head at a time), kv_len from seq_used.
    AttnCausalSoftmax,
    /// hd512 unfused: per-head QK^T via the plain bf16 NAX GEMM
    /// (`gemm_nax_bf16_qk`); N = kv_len read from seq_used at runtime.
    AttnGemmQk,
    /// hd512 unfused: per-head PV via the plain bf16 NAX GEMM
    /// (`gemm_nax_bf16_pv`); K = kv_len read from seq_used at runtime.
    AttnGemmPv,
    /// Pure scalar broadcast multiply: `out = x * scale`.
    ScalarMul,
    /// Final logit softcapping: `out = cap * tanh(x / cap)` with the
    /// cap baked from `W::FINAL_LOGIT_SOFTCAPPING` as function
    /// constant 0 (Gemma2/Gemma4). Maps to
    /// `tanh_soft_cap_{f16,bf16}_specialized` in `elementwise.metallib`.
    TanhSoftCap,
    /// Elementwise residual add: `lhs += rhs`. Output is the lhs slot
    /// rebound (in-place semantics).
    Add,
    /// Per-row bias broadcast add: `out[m, n] = in[m, n] + bias[n]`
    /// (Qwen2/Qwen2.5 QKV biases). Maps to
    /// `bias_add_{f16,bf16}_specialized` in `elementwise.metallib`;
    /// `num_cols` rides on `function_constant(0)`.
    BiasAdd,
    /// Static reshape: rebinds a slot to a fresh logical shape; does
    /// not touch device memory. Lowering treats this as a metadata
    /// op — no `LoweredCommand` is emitted, only the slot's logical
    /// shape registers in the dispatcher.
    /// (Present in this enum for symmetry / future zero-copy ops.)
    Reshape,
    /// MLX-affine int4 decode matvec, K∈{64,128} ∧ pow2 bits.
    /// Maps to `affine_qmv_quad_<dtype>_gs_<gs>_b_4_d_<K>_batch_<batched>`
    /// in `quantized_qmv.metallib`. Faithful port of MLX's
    /// `affine_qmv_quad` (`quantized.h:1444`).
    AffineQmvQuad,
    /// MLX-affine int4 decode matvec, `N % 8 == 0 ∧ K % 512 == 0`.
    /// Maps to `affine_qmv_fast_<dtype>_gs_<gs>_b_4_batch_<batched>`.
    /// Faithful port of MLX's `affine_qmv_fast` (`quantized.h:1496`).
    AffineQmvFast,
    /// MLX-affine int4 decode matvec, generic shape fallback.
    /// Maps to `affine_qmv_<dtype>_gs_<gs>_b_4_batch_<batched>`.
    /// Faithful port of MLX's `affine_qmv` (`quantized.h:1548`).
    AffineQmv,
    /// MLX-affine int4 prefill matmul, transpose=true. Maps to
    /// `affine_qmm_t_<dtype>_gs_<gs>_b_4_alN_<bool>_batch_0` in
    /// `quantized_qmm.metallib`. Faithful port of MLX's
    /// `affine_qmm_t` (`quantized.h:1707`).
    AffineQmmT,
    /// MoE grouped expert GEMM (mlx `affine_gather_qmm_rhs`). Same as
    /// `AffineQmmT` but selects the per-expert weight slab via an
    /// `indices` buffer (the padded, expert-sorted per-row expert id);
    /// trailing sentinel tiles (`expert == QMM_NUM_EXPERTS`) skip. Maps
    /// to `affine_gather_qmm_t_<dtype>_s_<scale>_gs_<gs>_b_4_alN_<bool>
    /// _batch_0` in `quantized_qmm.metallib`. Bindings add `indices @ 5`;
    /// constants reuse `QMM_K/N/M` (0/1/2) + `QMM_NUM_EXPERTS` (4).
    AffineGatherQmmT,
    /// MoE grouped expert GEMM on the M5 matrix accelerator (NAX) — the
    /// fast path for the MoE prefill (the steel `AffineGatherQmmT` runs
    /// ~5-10x slower per call). BM=64 tile. Maps to
    /// `affine_gather_qmm_t_nax_<dtype>_s_<scale>_gs_<gs>_b_4_alN_<bool>
    /// _batch_0` in `quantized_qmm_nax.metallib`. Same bindings/constants
    /// as `AffineGatherQmmT`. Dispatched when `is_nax_capable` + N%64==0
    /// + gs in {64,128}.
    AffineGatherQmmTNax,
    /// MLX-affine int4 prefill matmul, transpose=true, split-K
    /// variant for small-M / B=1 shapes. Maps to
    /// `affine_qmm_t_splitk_<dtype>_gs_<gs>_b_4_alN_<bool>`. Faithful
    /// port of MLX's `affine_qmm_t_splitk` (`quantized.h:1780`).
    /// Downstream sum-reduce across the split_k partition axis is
    /// emitted by the lowering pass (mirroring
    /// `quantized.cpp:861 strided_reduce_general_dispatch`).
    AffineQmmTSplitK,
    /// NAX (Apple9 / M4+) prefill matmul — 64×64×64 MPP matmul2d tile.
    /// Maps to `affine_qmm_t_nax_<dtype>_gs_<gs>_b_4_alN_<bool>_batch_0`
    /// in `quantized_qmm_nax.metallib`. Only dispatched when
    /// `is_nax_capable(profile.generation)` and `K % 64 == 0`.
    AffineQmmTNax,
    /// NAX decode-batch matmul: the 4-bit codes as the MPP `matmul2d`
    /// operand, one threadgroup's rows covering the batch, so each weight is
    /// read once per 8 or 16 rows. Maps to `affine_qmm_small_m_*` in
    /// `quantized_qmm_nax.metallib`; runs on `SMALL_M_TOKENS` steps only.
    AffineQmmSmallM,
    /// W4A8 pre-pass: quantizes a GEMM's activations to int8 per (row,
    /// 64-chunk) into the shared scratch. `affine_w4a8_quant_<dtype>`.
    AffineW4a8Quant,
    /// W4A8 GEMM on the matrix unit's int8 lane: the pre-pass's int8
    /// activations x the offset-8 4-bit codes.
    /// `affine_qmm_w4a8_<dtype>_s_<scale>_gs_<gs>_tn_<tn>_nsg_<nsg>`.
    AffineQmmW4a8,
    /// The MoE grouped W4A8 pre-pass over the padded rows (sentinel rows
    /// skipped). `affine_gather_w4a8_quant_<dtype>`.
    AffineGatherW4a8Quant,
    /// The MoE grouped expert GEMM on the int8 lane: each 32-row tile's
    /// expert slab x its int8 rows.
    /// `affine_gather_qmm_w4a8_<dtype>_s_<scale>_gs_<gs>_tn_<tn>_nsg_<nsg>`.
    AffineGatherQmmW4a8,
    /// NVFP4 int4 decode matvec (generic). Maps to
    /// `nvfp4_qmv_<dtype>_s_<scale>_gs_16_b_4_batch_0` in the
    /// `quantized_qmv.metallib` (nvfp4 kernels share that library with
    /// the affine `qmv` ones). Same structure as `AffineQmv`; the only
    /// difference is the E2M1-LUT weight decode (no per-group bias).
    /// NVFP4 int4 prefill matmul (transpose=true, standard tile). Maps
    /// to `nvfp4_qmm_t_<dtype>_s_<scale>_gs_16_b_4_alN_<bool>_batch_0`
    /// in `quantized_qmm.metallib`. Mirrors `AffineQmmT`.
    /// NVFP4 int4 prefill matmul on NAX (Apple9 / M4+) — 64×64 MPP
    /// matmul2d tile. Maps to
    /// `nvfp4_qmm_t_nax_<dtype>_s_<scale>_gs_16_b_4_alN_<bool>_batch_0`
    /// in `quantized_qmm_nax.metallib`. Dispatched (in place of
    /// `K % 64 == 0`. Mirrors `AffineQmmTNax`.
    /// Fused `silu(gate) * up` for the decomposed q-MLP path. The
    /// macro emits this after a pair of `AffineQmm` GEMMs when the
    /// gate/up Linears are MLX-affine quantized (plan P12 branch
    /// (i)). Maps to `silu_mul_<dtype>` in `silu_mul.metallib`.
    SiluMul,
    /// GELU (tanh approx) sibling of [`KernelId::SiluMul`] for the
    /// decomposed GeGLU q-MLP path (Gemma2/3/4). Maps to
    /// `gelu_mul_<dtype>` in `silu_mul.metallib`.
    GeluMul,
    /// Qwen3.5 attention output gate `out = attn * sigmoid(gate)`. Maps
    /// to `gate_apply_<dtype>` in `gate_apply.metallib`.
    GateApply,
    /// Qwen3.5-MoE shared-expert combine `out = routed + shared_y *
    /// sigmoid(g)` with `g` `[T, 1]` row-broadcast. Maps to
    /// `gate_scale_<dtype>` in `gate_scale.metallib`.
    GateScale,
    /// Qwen3.5 attention output-gate split: per-head deinterleave of the
    /// doubled `q_proj` output into `query` + `gate`. Maps to
    /// `gate_split_<dtype>` in `gate_split.metallib`.
    GateSplit,
    /// Qwen3.5 / Qwen3-Next Gated-DeltaNet linear-attention core. Maps to
    /// the `gated_delta_*` kernels in `gated_delta.metallib`.
    GatedDeltaNet,
    /// Sum-along-axis-0 reduce for the `[split_k, M, N]` intermediate
    /// `AffineQmmTSplitK` produces. Maps to
    /// `splitk_reduce_sum_<dtype>` in `quantized_splitk_reduce.metallib`.
    /// Lowered alongside `AffineQmmTSplitK` so the worker sees
    /// (qmm_t_splitk → scratch, reduce → out) as adjacent commands.
    SplitKReduceSum,
    /// MLX-affine int4 quantized embedding lookup: gather + dequant
    /// in one pass. Maps to
    /// `affine_embed_<dtype>_gs_<gs>_b_4` in
    /// `quantized_dequantize.metallib`. Faithful port of MLX's
    /// `nn.QuantizedEmbedding.__call__`
    /// (`python/mlx/nn/layers/quantized.py:144`).
    AffineEmbed,
    /// Slice the last-token row of a `[num_tokens, hidden]` activation
    /// to row 0 of the same buffer, in place. Inserted by the lowering
    /// pass before the lm_head GEMM so the GEMM runs at M=1 instead of
    /// M=num_tokens — only the last token's logits are ever consumed
    /// by the sampler. Maps to `gather_last_token_{f16,bf16}_specialized`
    /// in `gather_last_token.metallib`. Reads num_tokens from a 4-byte
    /// runtime buffer the worker re-writes each forward() — see
    /// `RuntimeBindingKind::NumTokensU32`.
    GatherLastToken,
    /// Paired with [`KernelId::GatherLastToken`]: copies row 0 of a
    /// `[num_tokens, vocab]` logits tensor BACK to row `num_tokens-1`
    /// in place, after the lm_head GEMM ran on its shrunk
    /// 1-m-tile grid. The worker's downstream
    /// `embedding_gather(logits, last_token_indices=[num_tokens-1])`
    /// then reads valid logits at row `num_tokens-1` without needing
    /// to know about the slice. Reads num_tokens from the same 4-byte
    /// runtime buffer the gather does
    /// (`RuntimeBindingKind::NumTokensU32`). Maps to
    /// `scatter_first_to_last_row_{f16,bf16}_specialized` in
    /// `gather_last_token.metallib`.
    ScatterFirstToLastRow,
    /// Row-wise precise softmax (MoE router prerequisite). Faithful
    /// port of MLX `softmax_single_row` from
    /// `mlx/backend/metal/kernels/softmax.h:10-98`. Bindings:
    /// `(in @ 0, out @ 1, axis_size_i32 inline @ 2)`. Dispatch shape:
    /// `(rows, 1, 1)` threadgroups × `(256, 1, 1)` threads. Maps to
    /// `block_softmax_precise_{float16,bfloat16}` in `softmax.metallib`.
    Softmax,
    /// Row-wise full ascending argsort. Used as the "argpartition+
    /// trailing-k slice" equivalent in the MoE router lowering for
    /// the small router widths (E ≤ 128) we target. Bindings:
    /// `(in @ 0, out_u32 @ 1, axis @ 2, one @ 3, one @ 4, stride_in @ 5,
    /// stride_out @ 6)`. Dispatch: `(1, rows, 1)` threadgroups ×
    /// `(bn, 1, 1)` threads where `bn ∈ {32,64}` per
    /// `argpartition::pick_pipeline_shape`. Symbol:
    /// `c_arg_block_sort_<dtype>_uint32_bn<bn>_tn4` in
    /// `argpartition.metallib`. Lowering pairs this with
    /// `SliceTrailingColsU32` to recover top-k indices.
    ArgPartitionTopK,
    /// 2-D contiguous take-along-axis gather: pulls the `[top_k]`
    /// scores per row from the `[num_experts]` softmax output via
    /// the `[num_tokens, top_k]` top-k index buffer. Faithful port
    /// of MLX `take_along_axis_2d_contig` (gather_axis.h). Bindings:
    /// `(src @ 0, idx_u32 @ 1, out @ 2, src_axis_i32 @ 3,
    /// idx_axis_i32 @ 4)`. Dispatch (converted to threadgroup form):
    /// `(ceil(idx_axis/tg_x), rows, 1)` × `(min(32, idx_axis), 1, 1)`.
    /// Symbol: `take_along_axis_2d_contig_{float16,bfloat16}` in
    /// `take_along_axis.metallib`.
    TakeAlongAxis,
    /// Per-row "drop everything but the trailing `top_k` columns"
    /// u32 slicer. Sits between [`KernelId::ArgPartitionTopK`] and
    /// [`KernelId::TakeAlongAxis`] to convert the full sorted-ascending
    /// `[rows, num_experts]` index tensor into `[rows, top_k]`.
    /// Bindings: `(src_u32 @ 0, dst_u32 @ 1, axis_size_i32 @ 2,
    /// top_k_i32 @ 3)`. Dispatch (threads-form converted to tg):
    /// `(ceil(top_k/tg_x), rows, 1)` × `(min(32, top_k), 1, 1)`.
    /// Symbol: `slice_trailing_cols_u32` in
    /// `slice_trailing_cols.metallib`.
    SliceTrailingColsU32,
    /// MoE per-expert gather-matvec, fast variant
    /// (`N % 8 == 0 && K % 512 == 0`). Faithful port of MLX
    /// `affine_gather_qmv_fast` (`quantized.h:1899`). Used for the
    /// 3× SwitchGLU gate/up/down projections inside one MoE block.
    /// Bindings: `(packed_w @ 0, scales @ 1, biases @ 2, x @ 3,
    /// rhs_indices @ 4, y @ 5, top_k_i32 inline @ 6)`. Function
    /// constants 0/1 carry K/N respectively. Dispatch:
    /// `(1, N/8, num_tokens*top_k)` threadgroups × `(32, 2, 1)` threads.
    /// Symbol: `affine_gather_qmv_fast_<dtype>_s_<sdtype>_gs_<gs>_b_4`
    /// in `quantized_qmv.metallib`.
    AffineGatherQmvFast,
    /// MLX-affine int4 gather-matvec generic-shape fallback. Same
    /// bindings + dispatch as [`KernelId::AffineGatherQmvFast`].
    /// Symbol: `affine_gather_qmv_<dtype>_s_<sdtype>_gs_<gs>_b_4`.
    AffineGatherQmv,
    /// `out[n, d] = Σ_k expert[n, k, d] * scores[n, k]` — the final
    /// MoE reduction. Function-constant specialization on top_k
    /// (constant 0) and hidden (constant 1). Bindings:
    /// `(expert_out @ 0, scores @ 1, out @ 2)`. Dispatch:
    /// `(ceil(hidden/tg_x), num_tokens, 1)` × `(min(64, hidden), 1, 1)`.
    /// Symbol: `moe_weighted_sum_{float16,bfloat16}` in
    /// `moe_weighted_sum.metallib`.
    MoeWeightedSum,
    /// MoE grouped-GEMM prefill: histogram `topk_inds` + padded
    /// exclusive-scan offsets. Single threadgroup. Function constants
    /// `MG_M` (0) / `MG_NUM_EXPERTS` (1). Bindings `(topk_inds @ 0,
    /// count @ 1, offset @ 2, total @ 3)`. Symbol `moe_group_offsets`
    /// in `moe_group.metallib`.
    MoeGroupOffsets,
    /// MoE grouped-GEMM prefill: sentinel-fill `indices_pad` + zero the
    /// `fill` counters. Function constants `MG_NUM_EXPERTS` (1) /
    /// `MG_MPAD_MAX` (2). Bindings `(indices_pad @ 0, fill @ 1)`.
    /// Symbol `moe_group_init`.
    MoeGroupInit,
    /// MoE grouped-GEMM prefill: scatter each (token,expert) row to its
    /// padded slot, recording `pos`, `indices_pad`, and the gathered
    /// `x_pad` row. Function constants `MG_M`(0)/`MG_NUM_EXPERTS`(1)/
    /// `MG_TOP_K`(3)/`MG_K`(4). Bindings `(topk_inds @ 0, offset @ 1,
    /// x @ 2, fill @ 3, pos @ 4, indices_pad @ 5, x_pad @ 6)`. Symbol
    /// `moe_group_scatter_{float16,bfloat16}`.
    MoeGroupScatter,
    /// `MoeGroupScatter` for the W4A8 grouped GEMM: scatters the tokens'
    /// int8 rows + per-64-chunk scales (quantized once, before the scatter)
    /// into the padded layout. Also `MG_MPAD_MAX` (2). Symbol
    /// `moe_group_scatter_q8`.
    MoeGroupScatterQ8,
    /// MoE grouped-GEMM prefill: un-scatter the padded down output back
    /// to token order — `out[i,:] = src[pos[i],:]`. Function constant
    /// `MG_W` (5). Bindings `(src @ 0, pos @ 1, out @ 2)`. Symbol
    /// `moe_group_gather_{float16,bfloat16}`.
    MoeGroupGather,
    /// Gemma-4 per-expert score scale (the `gemma_moe` op): in place,
    /// `topk_scores[m, j] *= per_expert_scale[topk_inds[m, j]]` over the
    /// `[M, top_k]` gathered top-k scores. Bindings: `(topk_scores @ 0
    /// in/out f16/bf16, topk_inds @ 1 u32, per_expert_scale @ 2 f16/bf16,
    /// top_k inline @ 3)`. Dispatch one thread per `(m, j)`. Symbol:
    /// `moe_per_expert_scale_{float16,bfloat16}` in
    /// `moe_per_expert_scale.metallib`.
    MoePerExpertScale,
    /// Vision-tower LayerNorm-with-bias (Qwen3.5-VL ViT norm1/norm2/
    /// merger.norm). Maps to `vision_layernorm_{f16,bf16}` in
    /// `vision_layernorm.metallib`. Bindings: `(out @ 0, in @ 1,
    /// weight @ 2, bias @ 3)`; function constants LN_M/LN_HIDDEN/LN_EPS.
    VisionLayerNorm,
    /// Vision-tower 2D NeoX RoPE (rotate_half), applied independently
    /// to q and k. Maps to `vision_rope_2d_{f16,bf16}` in
    /// `vision_rope_2d.metallib`. Bindings: `(out @ 0, x @ 1, freqs @ 2)`;
    /// function constants VR_HEAD_DIM/VR_NUM_HEADS/VR_N_ELEMS.
    VisionRope,
    /// Vision-tower bidirectional varlen SDPA. Maps to
    /// `vision_varlen_attn_{f16,bf16}` in `vision_varlen_attn.metallib`.
    /// Bindings: `(out @ 0, q @ 1, k @ 2, v @ 3, cu_seqlens @ 4)`;
    /// function constants VA_HEAD_DIM/VA_NUM_HEADS/VA_NUM_SEGS/
    /// VA_N_TOKENS/VA_SCALE.
    VisionVarlenAttn,
    /// Row gather by a runtime u32 index buffer (`embedding_gather.metal`)
    /// — Qwen2.5-VL window permutation / inverse.
    EmbeddingGather,
    /// Standalone tanh-approx GELU (Qwen3.5-VL ViT MLP / merger MLP).
    /// Maps to `gelu_tanh_{f16,bf16}` in `activation.metallib`. Bindings:
    /// `(out @ 0, in @ 1, n inline @ 2)`.
    VisionGelu,
    /// Copy the vision `pixels` runtime extern into an arena slot
    /// (materialized by `vision_lowering::materialize_pixels`). Maps to
    /// `copy_rows_{f16,bf16}` in `elementwise.metallib`.
    VisionLoadPixels,
    /// Multimodal embed splice — scatter the projected vision embeddings
    /// (`MmEmbeds`) into the text embedding stream at the placeholder
    /// rows (`MmDstRows`). Maps to `mm_embed_splice_{f16,bf16}` in
    /// `elementwise.metallib`. Bindings: `(embed @ 0 in/out, mm @ 1,
    /// dst_rows @ 2, hidden inline @ 3)`.
    MmEmbedSplice,
    /// TurboQuant prefill: write a layer's K (or V) for the step's sequences
    /// into the reused fp16 scratch in the codebook's ROTATED domain (R·k),
    /// right before that layer's prefill attention — a table lookup per cached
    /// key, a Walsh-Hadamard transform per new key. Maps to
    /// `tq_stage_rotated_{f16,bf16}` in `attention.metallib`. The packed store
    /// (canonical, ~4.7x smaller) is the only persistent KV; the scratch holds
    /// one layer's image for the attention read, then is reused.
    TqStageRotated,
    /// TurboQuant prefill: rotate the attention's q rows into the codebook
    /// domain (R·q) before it and its output rows back (Rᵀ·o) after it, in
    /// place. Maps to `tq_{rotate,unrotate}_rows_{f16,bf16}` in
    /// `attention.metallib`.
    TqRotateRows,
    /// TurboQuant: quantize a layer's newly-written KV (in the fp16 scratch)
    /// into the PACKED store right after that layer's KV writer. Maps to
    /// `tq_compress_paged[_bf16]` in `turboquant.metallib`.
    TqQuantizeToPacked,
    /// TurboQuant decode attention: `AttentionViaCache` reading the packed
    /// store directly in the codebook domain (function constant 13), so a
    /// decode step never dequantizes the context. Same symbol as
    /// `AttentionViaCache`.
    AttentionViaCacheTq,
}

/// Which sequences of a step a command computes correctly.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SeqScope {
    /// Every sequence: the kernel works token by token, or finds each
    /// sequence's rows through `cu_seqlens_q` and its own block-table row.
    AllRows,
    /// Batch row 0 only: the kernel, or a scratch it binds, holds one
    /// sequence's state (`seq_used[0]`, `cu_seqlens_q[1] - cu_seqlens_q[0]`,
    /// or keys staged from row 0). The lowering runs such a command on
    /// single-sequence steps only, beside a per-row twin for the rest.
    RowZero,
}

impl KernelId {
    /// No wildcard arm: a new kernel must say whether it reads every sequence.
    pub const fn seq_scope(self) -> SeqScope {
        match self {
            Self::RopeOnceNax
            | Self::RopeOnceSteel
            | Self::RopeOnceGqaShared
            | Self::AttnGatherKRope
            | Self::AttnGatherVCopyT
            | Self::AttnCausalSoftmax
            | Self::AttnGemmQk
            | Self::AttnGemmPv => SeqScope::RowZero,
            Self::Embed
            | Self::RmsNorm
            | Self::RmsNormUnit
            | Self::ScalarWeightMul
            | Self::NormAddScalarMul
            | Self::RopeAppendNormed
            | Self::FusedAddRmsNorm
            | Self::Gemm
            | Self::FusedGateUpSiluMul
            | Self::RopeAppend
            | Self::AttentionViaCache
            | Self::AttentionPrefillSdpaPaged
            | Self::AttnQConvert
            | Self::AttnOConvert
            | Self::ScalarMul
            | Self::TanhSoftCap
            | Self::Add
            | Self::BiasAdd
            | Self::Reshape
            | Self::AffineQmvQuad
            | Self::AffineQmvFast
            | Self::AffineQmv
            | Self::AffineQmmT
            | Self::AffineGatherQmmT
            | Self::AffineGatherQmmTNax
            | Self::AffineQmmTSplitK
            | Self::AffineQmmTNax
            | Self::AffineQmmSmallM
            | Self::AffineW4a8Quant
            | Self::AffineQmmW4a8
            | Self::AffineGatherW4a8Quant
            | Self::AffineGatherQmmW4a8
            | Self::SiluMul
            | Self::GeluMul
            | Self::GateApply
            | Self::GateScale
            | Self::GateSplit
            | Self::GatedDeltaNet
            | Self::SplitKReduceSum
            | Self::AffineEmbed
            | Self::GatherLastToken
            | Self::ScatterFirstToLastRow
            | Self::Softmax
            | Self::ArgPartitionTopK
            | Self::TakeAlongAxis
            | Self::SliceTrailingColsU32
            | Self::AffineGatherQmvFast
            | Self::AffineGatherQmv
            | Self::MoeWeightedSum
            | Self::MoeGroupOffsets
            | Self::MoeGroupInit
            | Self::MoeGroupScatter
            | Self::MoeGroupScatterQ8
            | Self::MoeGroupGather
            | Self::MoePerExpertScale
            | Self::VisionLayerNorm
            | Self::VisionRope
            | Self::VisionVarlenAttn
            | Self::EmbeddingGather
            | Self::VisionGelu
            | Self::VisionLoadPixels
            | Self::MmEmbedSplice
            | Self::TqStageRotated
            | Self::TqRotateRows
            | Self::TqQuantizeToPacked
            | Self::AttentionViaCacheTq => SeqScope::AllRows,
        }
    }
}

/// `MetalDtype` relocated to the cfg-free `scratchy-tensors` core so
/// the `scratchy-ir` `CanonicalParams::METAL_DTYPE` const can name
/// it. Re-exported here so `crate::tape::MetalDtype` (the
/// ~54 `W::METAL_DTYPE` reads + the `mod.rs` re-export) keeps resolving.
pub use scratchy_tensors::MetalDtype;

/// Per-axis dispatch grid: threadgroup count + threads per group.
///
/// The lowering pass computes both from `(bucket M, kernel-specific
/// tile dims)`. Kept as `(u32, u32, u32)` rather than Metal's `MTLSize`
/// so this type stays available without the `metal` crate (lowering is
/// pure CPU code).
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub struct DispatchShape {
    /// (x, y, z) threadgroup count — baseline computed against
    /// `bucket_m` at lowering time.
    pub threadgroups: (u32, u32, u32),
    /// (x, y, z) threads per threadgroup.
    pub threads_per_threadgroup: (u32, u32, u32),
    /// If `Some`, the runtime rewrites the m-axis count using
    /// `actual_num_tokens` instead of `bucket_m`. The baked
    /// `threadgroups` field still holds the bucket_m-based count;
    /// runtime computes `axis_count = num_tokens.div_ceil(tile)` and
    /// patches `tg.{axis}` immediately before `dispatchThreadgroups`.
    ///
    /// Without this, GEMM-class kernels over-dispatch by up to 8×
    /// when actual_M sits at the low end of a wide bucket
    /// (e.g. bucket_m=4096 servicing num_tokens=1024) — every spare
    /// threadgroup pays the full per-tile compute cost on garbage
    /// rows in the arena past `num_tokens`.
    pub m_scaling: Option<MScaling>,
}

/// Tells the runtime how to shrink the dispatch grid for the actual
/// `num_tokens` of this forward pass. `axis` is 0/1/2 for x/y/z;
/// `bucket_m` is the bucket-M the baseline `threadgroups` count was
/// computed against.
///
/// Runtime formula:
///
/// ```text
/// new_count = ceil(baseline_count * num_tokens / bucket_m)
/// ```
///
/// Equivalent to "scale this axis proportionally with M". Works for
/// every kernel — tile-based GEMM grids
/// (`(n_tiles, ceil(M/tile), …)`), per-row dispatches
/// (`(M, n_heads, …)`), and 1D-over-`M*K` elementwise kernels alike
/// — because each is linear in M and the baseline is just that
/// linear function evaluated at `bucket_m`.
/// Axis of a threadgroup dispatch grid (`(x, y, z)`). Replaces the
/// historical `u8` field on [`MScaling`]: writing `axis: 3` would have
/// silently fallen through `worker::scale_tg_for_num_tokens`'s match
/// and returned the un-scaled grid; the enum variant set forces one
/// of the three legal options.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize)]
pub enum MScaleAxis {
    X,
    Y,
    Z,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub struct MScaling {
    pub axis: MScaleAxis,
    pub bucket_m: BucketM,
    /// When `Some(ax)`, the worker SETS `threadgroups.{ax}` to the live
    /// `num_seqs` at dispatch (not a proportional scale — an exact set).
    /// Used by the steel paged prefill attention kernel, whose grid is
    /// `(Q-blocks-per-seq, num_q_heads, num_seqs)`: it tiles queries in
    /// BQ-sized blocks that must not straddle a sequence boundary, so it
    /// needs one grid-Z layer per sequence (`tid.z = seq_idx`, indexing
    /// `cu_seqlens_q` / `block_table`). `axis` still scales the Q-block
    /// dimension by `num_tokens`; over-dispatched blocks early-out in the
    /// kernel. `None` (every other kernel, incl. SDPA which is per-query-
    /// token and self-attributes via `cu_seqlens_q`) leaves Z untouched.
    pub seq_axis: Option<MScaleAxis>,
}

/// Runtime gate evaluated per dispatch — when `Some`, the worker
/// skips the dispatch unless the live `num_seqs` matches the gate.
/// `None` (the common case, used by every kernel except the
/// lm_head slice / fallback pair) means "always dispatch."
///
/// Mirrors the way `barrier_before` is a per-command parallel Vec
/// on [`LoweredMetalTape`]: cheap to encode, cheap to check at
/// dispatch time, no impact on the hot path for the 99% of
/// commands that aren't gated.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize)]
pub enum RuntimeGate {
    /// Run only when `!has_spec_tokens`. Used by the lm_head slice
    /// trio (gather/qmv/scatter) — the slice writes exactly the
    /// `num_sample_rows` rows pointed at by `last_token_indices`,
    /// which is correct for single-seq + multi-seq prefill and for
    /// decode (num_sample_rows == num_seqs), but wrong for spec
    /// verify where rejection sampling needs every row of logits.
    OnlyIfNoSpec,
    /// Run only when `has_spec_tokens`. Used by the full
    /// `M=bucket_m` lm_head fallback that populates every row of
    /// logits for greedy rejection sampling at spec-decode verify.
    OnlyIfSpec,
    /// Run only on a decode step — every sequence contributes exactly one
    /// token (`num_tokens == num_seqs`): a TurboQuant model's
    /// `AttentionViaCacheTq`, reading the packed store directly.
    OnlyIfDecodeStep,
    /// Run unless this is a decode step: the attention an
    /// `AttentionViaCacheTq` twin replaces there, and the rotated-domain K/V
    /// staging and q/output rotation around it.
    UnlessDecodeStep,
    /// Run only on a step whose token count is in
    /// `quantized::SMALL_M_TOKENS`: the `AffineQmmSmallM` twin of a GEMM.
    OnlyIfSmallMTokens,
    /// Run unless the step's token count is in `quantized::SMALL_M_TOKENS`:
    /// the GEMM an `AffineQmmSmallM` twin replaces there.
    UnlessSmallMTokens,
    /// Run only on a step with one sequence: the prefill attention that reads
    /// K from the rope-once scratch, which holds one sequence's keys.
    OnlyIfOneSequence,
    /// Run only on a step with several sequences: the attention twins that
    /// read each sequence's K through its own block-table row.
    UnlessOneSequence,
    /// Run only on a step whose block tables hold an unrotated (bit-31, span)
    /// block: the per-row attention twin that re-ropes K as it reads it.
    OnlyIfUnrotatedBlocks,
    /// Run unless the step's block tables hold an unrotated block: the
    /// per-row attention twin that reads the cache's already-roped K as is.
    UnlessUnrotatedBlocks,
    /// Run only when every gate in the list matches.
    All(&'static [RuntimeGate]),
}

impl RuntimeGate {
    /// `gate` narrowed by `and`: a command that already carries a gate keeps
    /// it, and must now also match every gate in `and`.
    pub fn and(gate: Option<RuntimeGate>, and: &[RuntimeGate]) -> RuntimeGate {
        let mut all: Vec<RuntimeGate> = match gate {
            None => Vec::new(),
            Some(RuntimeGate::All(gs)) => gs.to_vec(),
            Some(g) => vec![g],
        };
        all.extend_from_slice(and);
        match all.as_slice() {
            [only] => *only,
            _ => RuntimeGate::All(baked(all)),
        }
    }
}

/// The live facts a [`RuntimeGate`] reads, for one forward.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GateCtx {
    pub num_tokens: u32,
    pub num_seqs: u32,
    pub has_spec_tokens: bool,
    /// Some sequence's block table has an unrotated (bit-31, span) block.
    pub unrotated_blocks: bool,
}

impl GateCtx {
    /// One sequence decoding one token, no speculative tokens: the contexts a bucket-1
    /// [`MegakernelTape`] is planned under, where every gate is a constant — the bake checks
    /// each of its gates reads the same whether or not the sequence holds an unrotated block.
    pub const fn decode_one(unrotated_blocks: bool) -> Self {
        Self {
            num_tokens: 1,
            num_seqs: 1,
            has_spec_tokens: false,
            unrotated_blocks,
        }
    }
}

impl RuntimeGate {
    /// Whether a command under this gate runs in `ctx` — ONE predicate: the worker asks it per
    /// dispatch, the megakernel bake per planned context.
    pub fn admits(self, ctx: GateCtx) -> bool {
        // lm_head slice (`OnlyIfNoSpec`) fires only when there are EXTRA
        // tokens to drop (prefill / chunked-prefill / mixed batches);
        // steady-state decode has `num_tokens == num_seqs` and slicing
        // would just add 2 kernel launches with no GEMM-work savings
        // (qmv at M=num_seqs == qmm at M=num_seqs). Verified: gating
        // unconditionally on multi-seq regressed c=4 TPOT by +5% on
        // Llama-1B; gating on `num_tokens > num_seqs` keeps the prefill
        // win without hurting decode.
        let GateCtx {
            num_tokens,
            num_seqs,
            has_spec_tokens,
            unrotated_blocks,
        } = ctx;
        let decode_step = num_tokens == num_seqs;
        match self {
            Self::OnlyIfNoSpec => !has_spec_tokens && num_tokens > num_seqs,
            Self::OnlyIfSpec => has_spec_tokens,
            Self::OnlyIfDecodeStep => decode_step,
            Self::UnlessDecodeStep => !decode_step,
            Self::OnlyIfSmallMTokens => crate::quantized::SMALL_M_TOKENS.contains(&num_tokens),
            Self::UnlessSmallMTokens => !crate::quantized::SMALL_M_TOKENS.contains(&num_tokens),
            Self::OnlyIfOneSequence => num_seqs == 1,
            Self::UnlessOneSequence => num_seqs > 1,
            Self::OnlyIfUnrotatedBlocks => unrotated_blocks,
            Self::UnlessUnrotatedBlocks => !unrotated_blocks,
            Self::All(gates) => gates.iter().all(|g| g.admits(ctx)),
        }
    }
}

/// How the worker plays a baked tape.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TapePlay {
    /// One MTL4 dispatch per command.
    Dispatch,
    /// Every megakernel run the tape carries as one persistent kernel; the commands outside the
    /// runs are dispatched as [`Self::Dispatch`] dispatches them.
    #[default]
    Megakernel,
}

impl DispatchShape {
    /// 1D dispatch helper: `total_threads` rounded up by
    /// `threads_per_group`.
    pub fn dispatch_1d(total_threads: u32, threads_per_group: u32) -> Self {
        let groups = total_threads.div_ceil(threads_per_group);
        Self {
            threadgroups: (groups, 1, 1),
            threads_per_threadgroup: (threads_per_group, 1, 1),
            m_scaling: None,
        }
    }

    /// Typed elementwise-broadcast dispatch for per-token activation
    /// kernels (`ScalarMul`, `TanhSoftCap`, the lm_head narrow softcap,
    /// vision `Add` / `Gelu*` / `LayerNorm` / `EmbeddingGather`).
    ///
    /// Builds the `eff_m * width` 1D grid plus the standard X-axis
    /// `m_scaling` (full `bucket_m`) the runtime uses to rescale to the
    /// live `num_tokens`.
    ///
    /// `width` is an [`ActivationWidth`] — a value that can ONLY be
    /// produced by a real activation source (a Gemm/AffineQmm output-N, a
    /// reshape width, or the residual-stream `HIDDEN_SIZE`). It is a
    /// COMPILE error to pass a head-geometry constant such as `W::Q_SIZE`
    /// here. That confusion is exactly the granite logits-corruption bug:
    /// a partial `eff_m * Q_SIZE` multiply over a `[tokens, vocab]` buffer
    /// leaves each row's tail unscaled and is NOT argmax-invariant. The
    /// type makes that unrepresentable rather than caught at runtime.
    pub fn activation_broadcast(eff_m: u32, width: ActivationWidth, bucket_m: BucketM) -> Self {
        // Threadgroup width matches lowering.rs's THREADS_PER_GROUP (256);
        // lowered.rs is the lower-level module and must not depend upward
        // on lowering.rs, so the value is spelled here.
        let mut d = Self::dispatch_1d(eff_m * width.get(), 256);
        d.m_scaling = Some(MScaling {
            seq_axis: None,
            axis: MScaleAxis::X,
            bucket_m,
        });
        d
    }

    /// The grid dispatched for a forward of `num_tokens` tokens in `num_seqs` sequences.
    pub fn threadgroups_at(&self, num_tokens: NumTokens, num_seqs: u32) -> (u32, u32, u32) {
        let (x, y, z) = self.threadgroups;
        let Some(s) = self.m_scaling else {
            return self.threadgroups;
        };
        let [x, y, z] = s.scale([x, y, z].map(u64::from), num_tokens, num_seqs);
        let narrow = |v: u64| u32::try_from(v).unwrap_or(u32::MAX);
        (narrow(x), narrow(y), narrow(z))
    }
}

impl MScaling {
    /// `grid` (threadgroups per axis) rescaled for a forward of `num_tokens` tokens in
    /// `num_seqs` sequences: `axis` becomes `ceil(baseline · n / bucket_m)` with `n` clamped to
    /// `[1, bucket_m]` (never above the baked grid), and `seq_axis`, when set, becomes the live
    /// `num_seqs`.
    pub fn scale(self, mut grid: [u64; 3], num_tokens: NumTokens, num_seqs: u32) -> [u64; 3] {
        let bm = self.bucket_m.get().max(1) as u64;
        let n = (num_tokens.get().max(1) as u64).min(bm);
        let slot = &mut grid[self.axis.index()];
        *slot = slot.saturating_mul(n).div_ceil(bm);
        // `seq_axis`: SET (not scale) the chosen axis to the live num_seqs. The steel paged
        // prefill kernel needs one grid-Z layer per sequence (`tid.z = seq_idx`) so a BQ-block
        // tile never straddles a sequence boundary; over-dispatched (seq, q-block) pairs
        // early-out in the kernel.
        if let Some(seq_ax) = self.seq_axis {
            grid[seq_ax.index()] = num_seqs.max(1) as u64;
        }
        grid
    }
}

impl MScaleAxis {
    /// The axis of `(x, y, z)` index `i`.
    pub const fn of_index(i: u8) -> Option<Self> {
        match i {
            0 => Some(Self::X),
            1 => Some(Self::Y),
            2 => Some(Self::Z),
            _ => None,
        }
    }

    pub const fn index(self) -> usize {
        match self {
            Self::X => 0,
            Self::Y => 1,
            Self::Z => 2,
        }
    }
}

/// Width (in elements) of one activation row at a point in the
/// instruction stream — the per-token row stride an elementwise/broadcast
/// kernel must cover.
///
/// The inner field is PRIVATE and the only constructors are legitimate
/// activation producers:
///   * [`ActivationWidth::residual_stream`] = `HIDDEN_SIZE` (the width in
///     force before the first Gemm publishes one — e.g. granite's embed
///     `* embedding_multiplier`),
///   * [`ActivationWidth::from_gemm_n`] = a dense `Gemm` / quantized
///     `AffineQmm` output-N (vocab after lm_head, intermediate after the
///     MLP, hidden after o_proj, …),
///   * [`ActivationWidth::from_reshape_width`] = a reshape's static
///     (num_tokens-independent) column dim (the vision merger).
///
/// There is deliberately NO constructor from `W::Q_SIZE` / head geometry,
/// so [`DispatchShape::activation_broadcast`] can never be handed the
/// head-projection width by mistake. That confusion was the granite
/// `ScalarMul` corruption bug — now a compile error instead of corrupted
/// argmax at runtime.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize)]
pub struct ActivationWidth(u32);

impl ActivationWidth {
    /// The residual-stream width (`HIDDEN_SIZE`). The initial width before
    /// the first Gemm publishes one — i.e. the embed-time scalar multiply
    /// (granite's `* embedding_multiplier`), which precedes the first
    /// Gemm. Distinct from `Q_SIZE` (`num_q_heads * head_dim`), which it
    /// differs from whenever `head_dim != hidden/num_heads` (Qwen3.5
    /// head_dim=256).
    pub fn residual_stream(p: &crate::tape::model_consts::MetalModelConsts) -> Self {
        Self(p.hidden_size as u32)
    }

    /// A dense `Gemm` / quantized `AffineQmm` output-N: the `[*, n]`
    /// activation that op publishes (vocab after lm_head, etc.).
    pub fn from_gemm_n(n: u32) -> Self {
        Self(n)
    }

    /// A reshape's static (num_tokens-independent) column dim — the new
    /// activation width after the vision merger reshape.
    pub fn from_reshape_width(width: u32) -> Self {
        Self(width)
    }

    /// The width in elements.
    pub fn get(self) -> u32 {
        self.0
    }
}

/// Where the worker should source the buffer for a binding at worker
/// init time.
///
/// A model tensor is a [`Binding::Source`]: a family of the model's source
/// manifest, resolved ONCE at load through the macro-generated
/// [`ModelSources`] impl. No fn pointers live here, so `Binding` is fully
/// backend-neutral.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
pub enum Binding {
    /// `MetalWorker.arena[slot]` — the worker's private tile-arena
    /// buffer for this slot. The arena is sized for the colored
    /// `NUM_TILES` post-FUF coloring (linear-scan reg allocation
    /// performed by `colored_slot_map()` in
    /// `scratchy-forward-compiler-macro/src/interpreter_codegen.rs`).
    ArenaSlot { slot: u32, binding_index: u8 },
    /// A model tensor: `which` tensor of source family `ix` at `layer`. The family is the
    /// model's source manifest entry (a weight bundle, every layer); the pool resolves each
    /// distinct `(ix, which, layer)` ONCE at load through the macro-generated [`ModelSources`]
    /// impl — no locator, no accessor table, no per-dispatch lookup. A rolled loop's later
    /// iterations advance only `layer` ([`Binding::bump_layer`]).
    Source {
        ix: SourceIx,
        which: WeightTensor,
        layer: LayerId,
        binding_index: u8,
    },
    /// A buffer drawn from `ForwardCtx`-equivalent runtime state at
    /// `forward()` time (input_ids, positions, KV cache pages,
    /// cu_seqlens, etc.). The worker's bucket-selection layer
    /// rebinds these on every forward — the MTL4 compute encoder
    /// does not re-record, so runtime bindings reach the GPU via a
    /// pre-the MTL4 compute encoder `setBuffer` on the encoder.
    Runtime {
        kind: RuntimeBindingKind,
        binding_index: u8,
    },
    /// The worker's shared SplitK scratch buffer — sized to
    /// `LoweredMetalTape::splitk_scratch_bytes` at worker init, used
    /// by the two-command lowering for `Instruction::AffineQmm` when
    /// the bucket picks `QmmTKernel::SplitK`. The first command
    /// (`affine_qmm_t_splitk_*`) writes the `[split_k, M, N]`
    /// partial here; the second (`splitk_reduce_sum_*`) reads it and
    /// reduces to `[M, N]` in the AffineQmm's arena slot.
    ///
    /// Only one scratch buffer is needed even when multiple AffineQmm
    /// tiles pick SplitK: dispatch commands inside a single encoder are
    /// serialized, so the writer/reader pair fully completes before
    /// the next AffineQmm overwrites the scratch.
    Scratch { binding_index: u8 },
    /// The worker's shared roped-K scratch buffer (spans rope-on-read on
    /// NAX) — sized to `LoweredMetalTape::roped_k_scratch_bytes` at worker
    /// init. The `KernelId::RopeOnceNax` command ropes one request+layer's
    /// K from the paged cache into this dense buffer (indexed by logical
    /// block); the following NAX attention reads PRE-ROPED K from here at
    /// buffer slot 7 (no per-tile rotation). One buffer suffices: rope-once
    /// and its attention run serially per layer, and the scratch is
    /// overwritten each layer (the cache stays the persistent artifact).
    RopedKScratch { binding_index: u8 },
    /// The worker's shared hd512-unfused-attention scratch buffer — sized to
    /// `LoweredMetalTape::attn_unfused_scratch_bytes`. Packs the q_head /
    /// Kdense / Vdense_T / scores / out_head regions at baked byte `offset`s
    /// (one buffer, serial commands per layer; overwritten each layer).
    AttnUnfusedScratch { offset: u32, binding_index: u8 },
    /// `setBytes_length_atIndex` of a `u32` immediate at the argument
    /// table slot `binding_index`. Used by the MoE lowering arms to
    /// pass scalar shape parameters (axis_size, top_k, etc.) that
    /// match each kernel's `constant int& [[buffer(N)]]` declaration.
    /// The worker writes the 4 bytes onto the encoder; no device
    /// buffer is allocated.
    Inline { binding_index: u8, value: u32 },
    /// A bound sub-region of the worker's shared MoE scratch buffer
    /// (`MetalWorker.moe_scratch`, sized to
    /// `LoweredMetalTape::moe_scratch_bytes`). Each lowered MoE
    /// command picks the named region it operates on by passing
    /// `byte_offset` into `setBuffer_offset_atIndex`. Sub-regions are
    /// 256-byte aligned by the lowering pass per Apple Silicon's
    /// `MTLBuffer.offset` alignment rule.
    ///
    /// Inside one bucket the regions are: router_logits, sorted_full
    /// (`[M, num_experts]` u32), topk_inds (`[M, top_k]` u32),
    /// topk_scores (`[M, top_k]` act), gate_up_out (`[M, top_k,
    /// intermediate]` act), down_out (`[M, top_k, hidden]` act),
    /// plus optional shared_expert scratch (gate_up / act / out /
    /// gate_logit) when the variant is SharedFusedMoe with
    /// `shared_expert_intermediate_size > 0`. Layout is computed at
    /// lowering time and stamped into the byte_offset field; the
    /// worker only sees opaque offsets.
    MoeScratch { binding_index: u8, byte_offset: u32 },
}

/// A model's weight bundle, as its source manifest names it — what [`ModelSources::source`]
/// hands back and [`SourceRef::tensor`] picks a [`WeightTensor`] out of.
#[derive(Clone, Copy)]
pub enum SourceRef<'w> {
    Embedding(&'w scratchy_layers::Embedding),
    /// MLX-affine int4 quantized embedding.
    AffineQuantEmbedding(&'w scratchy_quantizations::AffineQuantEmbedding),
    RmsNorm(&'w scratchy_layers::RmsNorm),
    /// Torch-style LayerNorm with bias (vision towers).
    LayerNorm(&'w scratchy_layers::LayerNorm),
    Linear(&'w scratchy_layers::LinearLayer),
    /// A RoPE table's packed cos/sin cache (layer-independent).
    CosSin(scratchy_tensors::tensor::GpuTensor),
    /// Mixtral-style MoE: dense router + packed per-expert slabs.
    FusedMoe(&'w scratchy_layers::layers_moe::FusedMoELayer),
    /// Qwen-style MoE with an optional shared expert.
    SharedFusedMoe(&'w scratchy_layers::layers_moe::SharedFusedMoELayer),
    /// Gemma-4 router (dense gate, per-expert scale, norm gain).
    GemmaRouter(&'w scratchy_layers::layers_moe::GemmaRouterLayer),
    /// Gemma-4 SwitchGLU experts (router-less).
    GemmaSwitchGlu(&'w scratchy_layers::layers_moe::SwitchGluExpertsLayer),
    /// Qwen3.5 Gated-DeltaNet per-layer weights.
    GatedDeltaNet(&'w scratchy_layers::GatedDeltaNetLayer),
}

impl SourceRef<'_> {
    /// Whether the `which` tensor of this bundle is MLX-affine packed codes — what a kernel reads
    /// as written or offset-8 ([`crate::tape::kernel_constants::AffineCodes`]).
    pub fn is_affine_codes(self, which: WeightTensor) -> bool {
        use WeightTensor as T;
        match self {
            Self::Linear(scratchy_layers::LinearLayer::AffineQuant(_))
            | Self::AffineQuantEmbedding(_) => which == T::Weight,
            Self::FusedMoe(_) | Self::SharedFusedMoe(_) | Self::GemmaSwitchGlu(_) => matches!(
                which,
                T::MoeExpertGateW
                    | T::MoeExpertUpW
                    | T::MoeExpertDownW
                    | T::MoeSharedGateUpW
                    | T::MoeSharedDownW
            ),
            _ => false,
        }
    }

    /// The `which` tensor of this bundle; `None` when the bundle holds no such tensor (a `which`
    /// of another kind, an absent optional bias or shared expert, a Dense MoE on metal).
    pub fn tensor(self, which: WeightTensor) -> Option<scratchy_tensors::tensor::GpuTensor> {
        use WeightTensor as T;
        use scratchy_layers::LinearLayer as L;
        use scratchy_layers::layers_moe::{FusedMoELayer as F, SharedFusedMoELayer as Sh};
        let routed = |r: &scratchy_layers::layers_moe::AffineFusedMoELayer| match which {
            T::MoeRouterGate => Some(r.router_gate),
            T::MoeExpertGateW => Some(r.expert_gate_w),
            T::MoeExpertGateS => Some(r.expert_gate_scales),
            T::MoeExpertGateB => Some(r.expert_gate_biases),
            T::MoeExpertUpW => Some(r.expert_up_w),
            T::MoeExpertUpS => Some(r.expert_up_scales),
            T::MoeExpertUpB => Some(r.expert_up_biases),
            T::MoeExpertDownW => Some(r.expert_down_w),
            T::MoeExpertDownS => Some(r.expert_down_scales),
            T::MoeExpertDownB => Some(r.expert_down_biases),
            _ => None,
        };
        match (self, which) {
            (Self::Embedding(e), T::Weight) => Some(e.weight),
            (Self::AffineQuantEmbedding(e), T::Weight) => Some(e.weight),
            (Self::AffineQuantEmbedding(e), T::AffineScales) => Some(e.scales),
            (Self::AffineQuantEmbedding(e), T::AffineBiases) => Some(e.affine_biases),
            (Self::RmsNorm(n), T::Weight) => Some(n.weight),
            (Self::LayerNorm(n), T::Weight) => Some(n.weight),
            (Self::LayerNorm(n), T::Bias) => n.bias,
            (Self::CosSin(t), T::Weight) => Some(t),
            (Self::Linear(l @ L::Dense(_)), T::Weight) => Some(l.dense_weight()),
            (Self::Linear(l @ L::Dense(_)), T::Bias) => l.dense_bias(),
            (Self::Linear(l @ L::AffineQuant(_)), T::Weight) => Some(l.affine_weight()),
            (Self::Linear(l @ L::AffineQuant(_)), T::AffineScales) => Some(l.affine_scales()),
            (Self::Linear(l @ L::AffineQuant(_)), T::AffineBiases) => Some(l.affine_biases()),
            (Self::Linear(l @ L::AffineQuant(_)), T::AffineLinearBias) => l.affine_linear_bias(),
            (Self::Linear(l @ L::Nvfp4(_)), T::Weight) => Some(l.nvfp4_weight()),
            (Self::GatedDeltaNet(g), T::GdnConv1d) => Some(g.conv1d),
            (Self::GatedDeltaNet(g), T::GdnALog) => Some(g.a_log),
            (Self::GatedDeltaNet(g), T::GdnDtBias) => Some(g.dt_bias),
            (Self::GatedDeltaNet(g), T::GdnNorm) => Some(g.norm),
            (Self::GemmaRouter(r), T::GemmaRouterGate) => Some(r.gate),
            (Self::GemmaRouter(r), T::GemmaPerExpertScale) => Some(r.per_expert_scale),
            (Self::GemmaRouter(r), T::GemmaRouterScale) => Some(r.scale),
            (Self::GemmaSwitchGlu(e), T::MoeExpertGateW) => Some(e.expert_gate_w),
            (Self::GemmaSwitchGlu(e), T::MoeExpertGateS) => Some(e.expert_gate_scales),
            (Self::GemmaSwitchGlu(e), T::MoeExpertGateB) => Some(e.expert_gate_biases),
            (Self::GemmaSwitchGlu(e), T::MoeExpertUpW) => Some(e.expert_up_w),
            (Self::GemmaSwitchGlu(e), T::MoeExpertUpS) => Some(e.expert_up_scales),
            (Self::GemmaSwitchGlu(e), T::MoeExpertUpB) => Some(e.expert_up_biases),
            (Self::GemmaSwitchGlu(e), T::MoeExpertDownW) => Some(e.expert_down_w),
            (Self::GemmaSwitchGlu(e), T::MoeExpertDownS) => Some(e.expert_down_scales),
            (Self::GemmaSwitchGlu(e), T::MoeExpertDownB) => Some(e.expert_down_biases),
            (Self::FusedMoe(F::Affine(a)), _) => routed(a),
            (Self::SharedFusedMoe(Sh::Affine(s)), _) => match which {
                T::MoeSharedGateUpW => s.shared_gate_up_w,
                T::MoeSharedGateUpS => s.shared_gate_up_scales,
                T::MoeSharedGateUpB => s.shared_gate_up_biases,
                T::MoeSharedDownW => s.shared_down_w,
                T::MoeSharedDownS => s.shared_down_scales,
                T::MoeSharedDownB => s.shared_down_biases,
                T::MoeSharedExpertGate => s.shared_expert_gate,
                _ => routed(&s.routed),
            },
            _ => None,
        }
    }
}

/// A model's tensors by source index — implemented per model by the `#[forward]` macro from
/// the tapes' source manifest (one arm per family), and read ONCE at load by the pool.
pub trait ModelSources {
    /// Each source family's accessor, by index — for error messages only, never a key.
    const SOURCES: &'static [&'static str];
    /// Source family `ix` at `layer`; `None` when the manifest has no such family.
    fn source(&self, ix: SourceIx, layer: LayerId) -> Option<SourceRef<'_>>;
}

/// Which tensor inside a multi-tensor weight bundle this binding
/// references. Most bundles have a single weight tensor (`Weight`);
/// MLX-affine `LinearLayer::AffineQuant` carries four (packed weight,
/// per-group scales, per-group affine offsets, optional fp linear bias).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize)]
pub enum WeightTensor {
    /// The bundle's primary weight (`RmsNorm.weight`,
    /// `LinearLayer::Dense(Linear { weight, .. })`,
    /// `LinearLayer::AffineQuant(AffineQuantLinear { weight, .. })`
    /// — the U32 packed weights, in the affine case —
    /// `Embedding.weight`).
    Weight,
    /// The bundle's bias, if present (`Linear.bias`,
    /// `LayerNorm.bias`). The worker treats absent bias as
    /// `LoweringError::MissingBias` if a binding asks for it.
    Bias,
    /// Per-group scales on an MLX-affine LinearLayer
    /// (`AffineQuantLinear.scales`, `[N, K / group_size]` F16).
    /// Only valid against `LinearLayer::AffineQuant`.
    AffineScales,
    /// Per-group affine offsets on an MLX-affine LinearLayer
    /// (`AffineQuantLinear.affine_biases`, `[N, K / group_size]` F16).
    /// MLX terminology calls these "biases" — they are NOT the
    /// linear-layer bias. Only valid against `LinearLayer::AffineQuant`.
    AffineBiases,
    /// Optional fp linear-layer bias on an MLX-affine LinearLayer
    /// (`AffineQuantLinear.linear_bias`, `[N]` in activation dtype).
    /// Worker reports `MissingBias` if the layer's `linear_bias` is
    /// `None`. Only valid against `LinearLayer::AffineQuant`.
    AffineLinearBias,
    // ── MoE bundle tensors ──────────────────────────────────────────
    //
    // Valid only against the `SourceRef::{FusedMoe, SharedFusedMoe}` bundles (the expert slabs
    // also against `GemmaSwitchGlu`). Each names a concrete tensor; the worker reads its
    // arena-backed pointer and stamps it onto the encoder.
    /// `[num_experts, hidden_size]` dense router gate weight (BF16 /
    /// F16 — not quantized). Output of `Gemm(x, router_gate)` produces
    /// `[num_tokens, num_experts]` router logits.
    MoeRouterGate,
    /// Gemma-4 router bundle: dequant'd dense `[num_experts, hidden]`
    /// router projection (`router.proj` 8-bit → dense). Same role as
    /// `MoeRouterGate` but resolved through the `GemmaRouter` bundle.
    GemmaRouterGate,
    /// Gemma-4 router bundle: `[num_experts]` bf16 per-expert score scale
    /// (`router.per_expert_scale`). Read by the `moe_per_expert_scale`
    /// gather-multiply after the softmax.
    GemmaPerExpertScale,
    /// Gemma-4 router bundle: `[hidden]` bf16 RMSNorm gain (`router.scale`),
    /// applied to the router input before the router GEMM.
    GemmaRouterScale,
    /// `[num_experts, intermediate_size, hidden_size / 8]` packed
    /// per-expert gate_proj (gate half of SwitchGLU). U32 storage of
    /// int4 elements.
    MoeExpertGateW,
    /// `[num_experts, intermediate_size, hidden_size / group_size]`
    /// per-expert gate_proj scales (act-dtype).
    MoeExpertGateS,
    /// `[num_experts, intermediate_size, hidden_size / group_size]`
    /// per-expert gate_proj affine biases (act-dtype).
    MoeExpertGateB,
    /// Packed per-expert up_proj (`up` half of SwitchGLU).
    MoeExpertUpW,
    MoeExpertUpS,
    MoeExpertUpB,
    /// Packed per-expert down_proj. Reads `[num_tokens, top_k,
    /// intermediate_size]` × `[num_experts, hidden_size,
    /// intermediate_size]` → `[num_tokens, top_k, hidden_size]`.
    MoeExpertDownW,
    MoeExpertDownS,
    MoeExpertDownB,
    /// Shared-expert `gate_up` packed weight (`[2*shared_intermediate,
    /// hidden / 8]` U32) — present only when
    /// `shared_expert_intermediate_size > 0`. Lowering arm reads this
    /// through `LinearLayer::AffineQuant` semantics: one AffineQmm
    /// emits both halves stacked, then `SiluMul` splits.
    MoeSharedGateUpW,
    MoeSharedGateUpS,
    MoeSharedGateUpB,
    /// Shared-expert `down_proj` packed weight (`[hidden,
    /// shared_intermediate / 8]` U32).
    MoeSharedDownW,
    MoeSharedDownS,
    MoeSharedDownB,
    /// Dense `[1, hidden_size]` sigmoid gate that scales the shared-
    /// expert output. Stored as a `Linear` (not quantized) in MLX
    /// safetensors.
    MoeSharedExpertGate,
    // ── Gated-DeltaNet bundle tensors ───────────────────────────────
    //
    // Valid only against the `SourceRef::GatedDeltaNet` bundle.
    /// Causal depthwise conv1d weight `[conv_dim, 1, kernel]` (on-disk dtype).
    GdnConv1d,
    /// Per-value-head log-decay base `A_log` `[num_v_heads]`.
    GdnALog,
    /// Per-value-head softplus bias `dt_bias` `[num_v_heads]`.
    GdnDtBias,
    /// Gated-RMSNorm weight `[head_v_dim]`.
    GdnNorm,
}

/// Categories of buffers the worker rebinds per forward call.
///
/// These map 1:1 to fields on the runtime context the engine threads
/// into the Metal forward (the Metal analogue of `ForwardCtx`). The
/// worker's `forward()` consults `RuntimeBindingKind` to know which
/// runtime buffer to bind at which encoder slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize)]
pub enum RuntimeBindingKind {
    /// `[num_tokens]` u32 — tokens to embed.
    InputIds,
    /// `[num_tokens]` u32 — RoPE position per token (1D path) or
    /// `[3, num_tokens]` for MRoPE arches.
    Positions,
    /// `[num_tokens]` u32 — paged-cache slot per token for `layer`'s
    /// KV-cache GROUP. vLLM's group-shared hybrid layout has one slot_mapping
    /// per group (full + N sliding); the worker resolves
    /// `slot_mappings[layer_to_group[layer]]`. Uniform models have one group,
    /// so every layer resolves to `slot_mappings[0]` (byte-identical).
    SlotMapping {
        layer: LayerId,
    },
    /// `[batch+1]` u32 — prefill-only sequence boundaries.
    CuSeqlensQ,
    /// `[batch]` u32 — current K-axis used length per sequence.
    SeqUsedK,
    /// `[num_logical_blocks]` u32 — per-block SPAN LABEL for block-diagonal
    /// span attention (`0` = shared/query, `k+1` = k-th Relocatable span).
    /// Bound to the prefill attention only on rope-on-read arches; all-zero ⇒
    /// the span mask is inert ⇒ byte-identical to the non-spans path.
    SpanIds,
    /// `[batch, max_blocks]` u32 — per-sequence block table for `layer`'s
    /// KV-cache GROUP. Resolved as `block_tables[layer_to_group[layer]]`; one
    /// group on uniform models (every layer → `block_tables[0]`).
    BlockTable {
        layer: LayerId,
    },
    /// Paged KV cache pool (the worker resolves to the K and V
    /// halves at the right layer offset based on `layer`).
    KvCacheK {
        layer: LayerId,
    },
    KvCacheV {
        layer: LayerId,
    },
    /// `[num_physical_blocks]` u8 — spans / rope-on-read: `1` = this
    /// physical block's K is stored UNROTATED (a `BlockKind::Relocatable`
    /// span block) so the attention / rope_append kernels re-rope it on
    /// read. Per-LAYER (indexed directly by `layer.get()`, NOT through
    /// `layer_to_group`), parallel to `KvCacheK/V`, so hybrid arches get
    /// one flag mirror per layer. Bound only when `W::ROPE_ON_READ`; an
    /// all-zero buffer (every non-span request) makes the kernel's
    /// per-block rotation branch a no-op (parity).
    BlockUnrotatedFlags {
        layer: LayerId,
    },
    /// `[1]` u32 — actual `num_tokens` of this forward, written by
    /// the worker at `forward()` entry. Consumed by
    /// `KernelId::GatherLastToken` so the kernel can compute the
    /// source row index `num_tokens - 1` at runtime without needing
    /// a function constant (M varies per call).
    NumTokensU32,
    /// `[1]` u32 — actual `num_seqs` of the in-flight forward
    /// (= `cu_seqlens_q.len() - 1`). Mirrors `NumTokensU32` but for
    /// the per-seq-sample-row count the lm_head slice trio uses to
    /// gate its inner loop (gather copies `num_seqs` rows, qmm runs
    /// at M = num_seqs, scatter writes back `num_seqs` rows).
    NumSeqsU32,
    /// `[num_seqs]` u32 — per-sequence sample-row index
    /// (`cu_seqlens_q[i+1] - 1` for seq `i`). Mirrors Python vLLM's
    /// `logits_indices` and the CUDA path's `ForwardCtx.last_token_indices`.
    /// Read by the index-driven `gather_last_token`/`scatter_first_to_last_row`
    /// kernels — one row per sequence, in cu_seqlens_q order.
    SampleIndices,
    /// Gated-DeltaNet conv-state ring for a linear-attention layer
    /// (persistent f32 pool, mutated in place across forwards — the
    /// non-paged sibling of `KvCacheK/V`). Resolved to the per-layer
    /// `GdnStatePool` conv buffer.
    GdnConvState {
        layer: LayerId,
    },
    /// Gated-DeltaNet recurrent (ssm) state for a linear-attention layer
    /// (persistent f32 pool, mutated in place across forwards).
    GdnSsmState {
        layer: LayerId,
    },
    /// `[num_seqs]` i32 — GDN state-pool slot id per batched sequence
    /// (cu_seqlens order). Written per forward by the worker.
    GdnStateIndices,
    /// `[num_seqs]` u32 — 1 when the sequence is on its first (fresh)
    /// forward; the GDN conv1d/scan kernels then treat the slot's state
    /// as zero instead of reading stale recurrent state (the
    /// "degeneration after N requests" guard). Written per forward.
    GdnIsFresh,
    /// `[total_L, vision_head_dim/2]` f32 — the vision 2D-RoPE per-token
    /// rotary angle table (`freqs`). The `vision_rope_2d` kernel reads
    /// it at `buffer(2)` and computes cos/sin internally. Built host-side
    /// from `grid_thw` and uploaded by the vision wrapper, written per
    /// forward by the worker. Carried on `ForwardCtx::vision_rope_freqs`.
    VisionRopeFreqs,
    /// `[num_tokens, vision_in_features]` model-dtype — the vision patch
    /// pixel rows. `Instruction::LoadPixels` copies it into an arena slot.
    /// Carried on `ForwardCtx::pixels`, written per forward by the worker.
    Pixels,
    /// `[num_tokens, vision_embed_dim]` model-dtype — the Qwen3.5-VL
    /// host-interpolated learned positional embedding.
    /// `Instruction::LoadPosEmbeds` copies it into an arena slot.
    /// Carried on `ForwardCtx::pos_embeds`, written per forward by the
    /// worker.
    VisionPosEmbeds,
    /// `[num_tokens, hidden]` model-dtype — the projected vision-encoder
    /// embeddings (`ForwardCtx::mm_embeds`), copied row-blockwise into
    /// the text embedding stream at the image-placeholder rows by
    /// `Instruction::SpliceMmEmbeds`. Only the first `total_mm` rows are
    /// live; padding rows are never read (guarded by `MmDstRows`).
    MmEmbeds,
    /// `[num_tokens]` u32 — for each source `mm_embeds` row, the
    /// destination row in the text embedding stream, or `u32::MAX` for
    /// padding / text-only rows (the splice kernel skips those). Built
    /// host-side from `ForwardCtx::embed_patches`.
    MmDstRows,
    /// `[num_tokens, ROT_DIM]` model-dtype — per-token RoPE cos/sin
    /// override for MRoPE text decoders (Qwen3.5-VL). Each row `t` is the
    /// band-split (`mrope_section` T/H/W) cos/sin for token `t`; combined
    /// with identity positions (`positions[t] = t`) it lets the unmodified
    /// 1D `rope_append_*`/fused-qkv-rope kernels read the correct row
    /// without an in-kernel band-split (option (b)). The lowering binds it in
    /// place of the static cos/sin source when `MetalModelConsts::mrope`; the macro
    /// forward builds + writes it per forward via `ForwardInputs`. 16-byte
    /// placeholder on 1D-rope arches (never bound).
    MropeCosSin,
    /// i32 cu_seqlens bound by `VarlenAttention(cu_seqlens_kind = 1)`
    /// — Qwen2.5-VL full-attention layers. Per-image boundaries.
    VisionCuSeqlensFull,
    /// i32 cu_seqlens bound by `VarlenAttention(cu_seqlens_kind = 2)`
    /// — Qwen2.5-VL window-attention layers. Per-window boundaries.
    VisionCuSeqlensWindow,
    /// u32 merged-row permutation read by `EmbeddingGather(kind = 0)`
    /// (natural → window-grouped order).
    VisionWindowIndex,
    /// u32 inverse permutation read by `EmbeddingGather(kind = 1)`
    /// (window-grouped → natural order at the merger output).
    VisionReverseIndices,
    /// TurboQuant per-layer PACKED key/value code store (canonical KV, ~4.7x
    /// smaller than fp16). Source for `TqStageRotated` and
    /// `AttentionViaCacheTq`, dest for `TqQuantizeToPacked`. Worker resolves
    /// to `tq_packed_k/v[layer]`.
    TqPackedK {
        layer: LayerId,
    },
    TqPackedV {
        layer: LayerId,
    },
    /// TurboQuant per-layer f32 norm store (one norm per quantized vector).
    TqNormsK {
        layer: LayerId,
    },
    TqNormsV {
        layer: LayerId,
    },
    /// TurboQuant codebook buffers (shared across layers): the random ±1 sign
    /// vector, the Lloyd-Max boundaries, and the N(0,1) centroids.
    TqSigns,
    TqBoundaries,
    TqCentroids,
}

pub fn baked<T>(v: Vec<T>) -> &'static [T] {
    Box::leak(v.into_boxed_slice())
}

/// `X.into_baked()` = `baked(Vec::<ConstantValue>::from(X))` — the
/// constants-struct construction sites read the same as before with one
/// suffix swap.
pub trait IntoBaked<E> {
    fn into_baked(self) -> &'static [E];
}
impl<E, T: Into<Vec<E>>> IntoBaked<E> for T {
    fn into_baked(self) -> &'static [E] {
        baked(self.into())
    }
}

#[derive(Clone, Copy, PartialEq, serde::Serialize)]
pub struct LoweredCommand {
    pub kernel: KernelId,
    /// Compiled-metallib name the kernel symbol lives in (matches the
    /// `&'static str` keys [`SpecializedPipelineCache::with_standard_shaders`]
    /// registers).
    ///
    /// [`SpecializedPipelineCache::with_standard_shaders`]: crate::specialized_pipeline_cache::SpecializedPipelineCache::with_standard_shaders
    pub library: &'static str,
    /// MSL `kernel void` symbol the pipeline binds.
    pub function: &'static str,
    /// `[[function_constant(N)]]` bag the pipeline specializes on.
    /// Pre-baked at lowering time from `W::*` + bucket_m so the worker
    /// never reaches back into `CanonicalParams`. Empty is valid
    /// (e.g. `Add`, `ScalarMul`).
    pub constants: &'static [ConstantValue],
    pub dispatch: DispatchShape,
    pub bindings: &'static [Binding],
}

impl LoweredCommand {
    /// Construct from a [`MetalKernel`] ZST. The trait carries
    /// `LIBRARY`, `FUNCTION`, and `KERNEL_ID` so the trio can't drift
    /// out of sync. The typed Constants / BindingSet structs from
    /// Phases 2 and 3 lower to the existing `Vec` wire formats via
    /// `Into`.
    ///
    /// Use this for kernels whose symbol name is a single `&'static
    /// str` — attention, etc. Kernels whose symbol is composed at
    /// lowering time (qmv / qmm_t) keep the struct-literal
    /// `LoweredCommand { kernel, library, function, ... }` form.
    pub fn for_kernel<K: crate::tape::kernel_identity::MetalKernel>(
        constants: K::Constants,
        bindings: K::BindingSet,
        dispatch: DispatchShape,
    ) -> Self {
        Self {
            kernel: K::KERNEL_ID,
            library: K::LIBRARY,
            function: K::FUNCTION,
            constants: baked(constants.into()),
            dispatch,
            bindings: baked(bindings.into()),
        }
    }

    /// Row zero if its kernel or any buffer it binds is.
    pub fn seq_scope(&self) -> SeqScope {
        match self
            .bindings
            .iter()
            .map(|b| b.seq_scope())
            .find(|s| *s == SeqScope::RowZero)
        {
            Some(row_zero) => row_zero,
            None => self.kernel.seq_scope(),
        }
    }
}

/// A lowered command fused with its runtime gate.
///
/// The gate rides ON the command rather than in a `Vec<Option<RuntimeGate>>`
/// running parallel to `commands`. That makes a length/contents mismatch
/// *unrepresentable*: there is no separate gate array to drop, re-initialize,
/// or let drift out of sync with the commands. `gate == None` (the common
/// case) means "always dispatch."
#[derive(Clone, Copy, PartialEq, serde::Serialize)]
pub struct GatedCommand {
    pub command: LoweredCommand,
    pub gate: Option<RuntimeGate>,
}

impl GatedCommand {
    /// This command as it appears in loop iteration `by`: every layer it names, advanced.
    pub fn with_layer_bumped(&self, by: u32) -> GatedCommand {
        if by == 0 {
            return *self;
        }
        let mut c = *self;
        c.command.bindings = baked(
            c.command
                .bindings
                .iter()
                .map(|b| b.bump_layer(by))
                .collect::<Vec<_>>(),
        );
        c
    }

    /// An ungated command — always dispatches (the common case).
    pub fn ungated(command: LoweredCommand) -> Self {
        Self {
            command,
            gate: None,
        }
    }

    /// Apply a gate to a command.
    pub fn gated(command: LoweredCommand, gate: RuntimeGate) -> Self {
        Self {
            command,
            gate: Some(gate),
        }
    }
}

impl From<LoweredCommand> for GatedCommand {
    fn from(command: LoweredCommand) -> Self {
        Self::ungated(command)
    }
}

/// A dense GEMM's shape: `out = in @ weight^T` for the canonical row-major Linear layer —
/// `in: [m, k]`, `weight: [n, k]`, `out: [m, n]`. The lowering bakes it into the command's
/// function constants and grid (`lowering::gemm_command`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GemmDims {
    /// Rows of the activation / output (= bucket_m).
    pub m: u32,
    /// Output columns (= weight rows; from `Instruction::Gemm`'s `n`).
    pub n: u32,
    /// Inner dimension (= weight columns = activation columns).
    pub k: u32,
}

impl LayerId {
    /// This layer, `by` iterations further into a re-rolled layer loop.
    pub const fn advance(self, by: u32) -> Self {
        Self(self.0 + by)
    }
}

impl RuntimeBindingKind {
    /// Advance every layer this binding names by `by` loop iterations.
    ///
    /// ⛔ EXHAUSTIVE, NEVER `_`. A new layer-bearing variant that fell into a wildcard would
    /// keep iteration 0's layer on every iteration — the whole model would run on layer 0's KV
    /// cache and norms and still produce plausible-looking tokens. E0004 here instead.
    pub fn bump_layer(self, by: u32) -> Self {
        match self {
            Self::SlotMapping { layer } => Self::SlotMapping {
                layer: layer.advance(by),
            },
            Self::BlockTable { layer } => Self::BlockTable {
                layer: layer.advance(by),
            },
            Self::KvCacheK { layer } => Self::KvCacheK {
                layer: layer.advance(by),
            },
            Self::KvCacheV { layer } => Self::KvCacheV {
                layer: layer.advance(by),
            },
            Self::BlockUnrotatedFlags { layer } => Self::BlockUnrotatedFlags {
                layer: layer.advance(by),
            },
            Self::GdnConvState { layer } => Self::GdnConvState {
                layer: layer.advance(by),
            },
            Self::GdnSsmState { layer } => Self::GdnSsmState {
                layer: layer.advance(by),
            },
            Self::TqPackedK { layer } => Self::TqPackedK {
                layer: layer.advance(by),
            },
            Self::TqPackedV { layer } => Self::TqPackedV {
                layer: layer.advance(by),
            },
            Self::TqNormsK { layer } => Self::TqNormsK {
                layer: layer.advance(by),
            },
            Self::TqNormsV { layer } => Self::TqNormsV {
                layer: layer.advance(by),
            },
            Self::InputIds
            | Self::Positions
            | Self::CuSeqlensQ
            | Self::SeqUsedK
            | Self::SpanIds
            | Self::NumTokensU32
            | Self::NumSeqsU32
            | Self::SampleIndices
            | Self::GdnStateIndices
            | Self::GdnIsFresh
            | Self::VisionRopeFreqs
            | Self::Pixels
            | Self::VisionPosEmbeds
            | Self::MmEmbeds
            | Self::MmDstRows
            | Self::MropeCosSin
            | Self::VisionCuSeqlensFull
            | Self::VisionCuSeqlensWindow
            | Self::VisionWindowIndex
            | Self::VisionReverseIndices
            | Self::TqSigns
            | Self::TqBoundaries
            | Self::TqCentroids => self,
        }
    }
}

impl Binding {
    /// The argument-table index this binding fills.
    pub fn binding_index(&self) -> u8 {
        match *self {
            Self::ArenaSlot { binding_index, .. }
            | Self::Source { binding_index, .. }
            | Self::Runtime { binding_index, .. }
            | Self::Scratch { binding_index }
            | Self::RopedKScratch { binding_index }
            | Self::AttnUnfusedScratch { binding_index, .. }
            | Self::Inline { binding_index, .. }
            | Self::MoeScratch { binding_index, .. } => binding_index,
        }
    }

    /// Advance every layer this binding names by `by` loop iterations.
    pub fn bump_layer(self, by: u32) -> Self {
        match self {
            Self::Source {
                ix,
                which,
                layer,
                binding_index,
            } => Self::Source {
                ix,
                which,
                layer: layer.advance(by),
                binding_index,
            },
            Self::Runtime {
                kind,
                binding_index,
            } => Self::Runtime {
                kind: kind.bump_layer(by),
                binding_index,
            },
            Self::ArenaSlot { .. }
            | Self::Scratch { .. }
            | Self::RopedKScratch { .. }
            | Self::AttnUnfusedScratch { .. }
            | Self::Inline { .. }
            | Self::MoeScratch { .. } => self,
        }
    }

    /// The rope-once and unfused-attention scratches hold one sequence's keys.
    pub const fn seq_scope(self) -> SeqScope {
        match self {
            Self::RopedKScratch { .. } | Self::AttnUnfusedScratch { .. } => SeqScope::RowZero,
            Self::ArenaSlot { .. }
            | Self::Source { .. }
            | Self::Runtime { .. }
            | Self::Scratch { .. }
            | Self::Inline { .. }
            | Self::MoeScratch { .. } => SeqScope::AllRows,
        }
    }
}

/// ⭐ THE LAYER LOOP, KEPT IN THE BAKED TAPE INSTEAD OF UNROLLED INTO IT.
///
/// `commands[start .. start + period]` is ONE copy of the body; it runs `iters` times, and the
/// only thing that differs between iterations is the layer — `lower_one`'s `layer_offset` is
/// used in exactly one pattern, `LayerId(literal + layer_offset)`, and a weight's source family
/// is the same every iteration (the roll proof checks it). So iteration `i` is the baked
/// body with every [`LayerId`] advanced by `i`, which is what [`Binding::bump_layer`] does.
///
/// Unrolling at bake time is what made the emitted tape enormous: five llama configs came to
/// 14.3M lines of `const` struct literals, 328k of them `LayerId`s, and rustc's single-threaded
/// front end spent ~46s on the file. The loop was already found — `apply_loop_compression`
/// hands the lowering an `Instruction::Loop` — and the lowering then expanded it right back out.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub struct TapeLoop {
    /// First command of the body, in `commands`.
    pub start: u32,
    /// Body length in commands.
    pub period: u32,
    /// Total iterations, including the baked one.
    pub iters: u32,
    /// How far the LAYER advances per iteration.
    ///
    /// ⛔ NOT ALWAYS 1, AND ASSUMING SO SILENTLY MIS-ROLLS A MODEL. A one-layer body advances
    /// the layer by one, which was every model until gemma-4: its layers run `SSSSSG` — five
    /// sliding-window, one global — so the repeating unit is a SIX-layer cell and iteration `i`
    /// begins at layer `6i`. The commands are periodic; the layer numbering is not.
    pub layer_stride: u32,
}

/// An expanded command's origin: its position in the baked `commands` and how many layers its
/// enclosing loops advance it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CommandOrigin {
    pub baked: usize,
    pub layers: u32,
}

impl LoweredMetalTape {
    /// The tape's commands with the layer loop played out: prologue, then the body once per
    /// iteration with every `LayerId` advanced, then the tail.
    ///
    /// ⭐ THIS IS WHERE THE UNROLL MOVED TO. It used to happen at BAKE time, so the emitted
    /// `const` carried one materialised copy of the body per layer — 14.3M lines of struct
    /// literals for five llama configs, and rustc's single-threaded front end sitting on the
    /// file for ~46s. Doing it here costs one `Vec` per bucket at load.
    ///
    /// Exact because the only per-iteration input to the lowering is `layer_offset`, consumed
    /// in exactly one pattern (`LayerId(literal + layer_offset)`); a weight's source family
    /// does not vary.
    pub fn commands_expanded(&self) -> Vec<GatedCommand> {
        let cmds = self.commands;
        let origins = self.expanded_origins().into_iter();
        origins
            .map(|o| cmds[o.baked].with_layer_bumped(o.layers))
            .collect()
    }

    /// Where each command of [`Self::commands_expanded`] comes from — the same walk.
    pub fn expanded_origins(&self) -> Vec<CommandOrigin> {
        let mut out = Vec::with_capacity(self.commands.len());
        Self::walk(
            self.loops,
            0..self.commands.len(),
            0,
            &mut |baked, layers| out.push(CommandOrigin { baked, layers }),
        );
        out
    }

    /// `barrier_before`, expanded in lockstep with [`Self::commands_expanded`].
    ///
    /// The body is byte-equivalent across iterations apart from the layer, so every iteration
    /// fences exactly as the baked one does — the same walk, reading the flag at each baked
    /// position instead of the command.
    pub fn barriers_expanded(&self) -> Vec<bool> {
        let mut out = Vec::with_capacity(self.barrier_before.len());
        let flags = self.barrier_before;
        Self::walk(self.loops, 0..self.commands.len(), 0, &mut |pos, _| {
            out.push(flags.get(pos).copied().unwrap_or(true))
        });
        out
    }

    /// THE ONE WALK. Emitted rows, in order, each with the layer offset every enclosing loop
    /// contributes. Loops are outermost-first and nest by containment of `[start, start+period)`.
    fn walk(
        loops: &[TapeLoop],
        range: std::ops::Range<usize>,
        base: u32,
        f: &mut impl FnMut(usize, u32),
    ) {
        let mut pos = range.start;
        while pos < range.end {
            match loops.iter().enumerate().find(|(_, l)| {
                l.start as usize == pos && (l.start + l.period) as usize <= range.end
            }) {
                Some((i, l)) => {
                    let body = pos..pos + l.period as usize;
                    for it in 0..l.iters {
                        Self::walk(&loops[i + 1..], body.clone(), base + it * l.layer_stride, f);
                    }
                    pos = body.end;
                }
                None => {
                    f(pos, base);
                    pos += 1;
                }
            }
        }
    }
}

/// One bucket's lowered tape — the input the `MetalWorker` walks at
/// init time to record its per-bucket dispatch.
#[derive(Clone, Copy, PartialEq, serde::Serialize)]
pub struct LoweredMetalTape {
    /// Bucket M (number of tokens this tape was specialized for).
    /// Used by the worker to pick the right specialized pipeline
    /// (Phase 5.B) and the right runtime-buffer shapes.
    pub bucket_m: u32,
    /// Number of arena slots the tape references (post-coloring tile
    /// count). The worker allocates exactly this many arena buffers
    /// per shape class.
    pub num_arena_slots: u32,
    pub commands: &'static [GatedCommand],
    /// MTL4 encoder barrier-before flag per command, mirroring
    /// `commands.len()`. Sourced from the macro-emitted
    /// `MetalBucketSpec::{backbone,lm_head}_barriers` slice (one
    /// bool per `Instruction`) and expanded through loop
    /// unrolling — the macro's loop-compression body has the same
    /// barrier pattern across iterations (byte-equivalence is the
    /// compression precondition), so iteration N's body row i
    /// reuses iteration 0's flag at the same position. The bake
    /// pass propagates this into `Mtl4Step.barrier_before`; the
    /// runtime never re-derives the analysis.
    pub barrier_before: &'static [bool],
    /// The layer loops the bake kept ROLLED, OUTERMOST FIRST: `commands[start..start+period]`
    /// is one body copy that runs `iters` times, iteration `i` being that body with every
    /// `LayerId` advanced by `i * layer_stride` ([`Binding::bump_layer`]). Empty = straight-line.
    ///
    /// ⛔ A LIST BECAUSE LOOPS NEST. gemma-4's repeating unit is a six-layer `SSSSSG` cell, and
    /// the five sliding layers inside it are themselves a loop — so a row's layer is the SUM of
    /// every enclosing loop's contribution. A single `Option` could hold one or the other.
    pub loops: &'static [TapeLoop],
    /// Byte size of the shared SplitK scratch buffer the worker
    /// allocates if any `Instruction::AffineQmm` in this tape was
    /// lowered to the SplitK two-command form. Computed as
    /// `max(split_k * bucket_m * N * elem_size)` across all such
    /// instructions. Zero when no AffineQmm picked SplitK (in which
    /// case `Binding::Scratch` never appears and the worker skips
    /// the buffer allocation).
    pub splitk_scratch_bytes: u32,
    /// Byte size of the shared MoE scratch buffer the worker allocates
    /// if any `Instruction::{FusedMoe, SharedFusedMoe}` in this tape
    /// was lowered. The lowering pass packs router_logits / sorted_inds
    /// / topk_inds / topk_scores / gate_out / up_out / down_out (and
    /// shared-expert intermediates when present) into a single buffer
    /// region, each 256-byte aligned. Computed as the max
    /// per-MoE-block scratch footprint across all `I::FusedMoe` /
    /// `I::SharedFusedMoe` lowerings in this tape (MoE blocks within
    /// one bucket execute serially through the dispatch, so they can share
    /// scratch). Zero when no MoE instruction was lowered.
    pub moe_scratch_bytes: u32,
    /// Byte size of the shared roped-K scratch buffer the worker
    /// allocates if any spans rope-on-read NAX prefill (`KernelId::
    /// RopeOnceNax` + `Binding::RopedKScratch`) was lowered. Computed as
    /// `max(num_pages * num_kv_heads * BLOCK_SIZE * head_dim * elem_size)`
    /// across all such attention layers (one buffer, overwritten per
    /// layer — the cache, not the scratch, is the persistent artifact).
    /// Zero when no NAX spans prefill was lowered (the binding never
    /// appears and the worker skips the allocation).
    pub roped_k_scratch_bytes: u32,
    /// Byte size of the shared hd512-unfused-attention scratch
    /// (`Binding::AttnUnfusedScratch`). Max across layers of the packed
    /// q_head + Kdense + Vdense_T + scores + out_head regions. Zero when no
    /// hd512-unfused attention was lowered.
    pub attn_unfused_scratch_bytes: u32,
}

/// Errors produced by the lowering pass. Every step kind lowers (the
/// match over `MetalStep` is exhaustive), so these are malformed tapes.
#[derive(Debug)]
pub enum LoweringError {
    /// A `StepRow::Loop` whose body overran the step tape: the body
    /// extended past the end of the slice.
    /// Indicates malformed codegen — every tape the macro produces
    /// has been validated by the time it reaches lowering.
    MalformedLoop {
        index: usize,
        count: u32,
        body_len: u32,
        remaining: usize,
    },
    /// A TurboQuant codec command could not bind the additive offset of the
    /// KV operand it compresses — quantizing without removing it would let the
    /// offset's norm, not the signal's, set the codec's error: a K bias, with no
    /// rotary table to rotate it to each key (rope-on-read off).
    TurboQuantOffsetUnbound { index: usize },
    /// A step at tape index `index` binds the `slot`-th `kind` weight of its site, and its
    /// site holds no such source — the step records and the opcode lowering disagree.
    NoSource {
        index: usize,
        kind: scratchy_subtile::handoff::WeightKind,
        slot: u32,
    },
    /// A rope-on-read attention binds its layer class's rotary table, and the model has none.
    NoRotaryTable { index: usize },
    /// A gated row whose realization already gates its commands: one command, two gates.
    DoubleGate { index: usize },
    /// A command computes batch row 0 only ([`SeqScope::RowZero`]) and its
    /// step has no per-row twin re-roping span blocks, so some step with
    /// several sequences would run no attention, or row 0's for every sequence.
    RowZeroWithoutPerRowTwin { index: usize, kernel: KernelId },
    /// The model's KV codec is TurboQuant, but its backbone binds the KV cache
    /// without compressing any of it: the factory would provision packed stores
    /// nothing writes, and the worker would stop growing a pool the tape reads.
    TurboQuantCompressesNothing,
    /// A KV codec step reached the lowering of a model whose KV codec is dense: the codec pass
    /// runs only on a TurboQuant model.
    CodecStepOnDenseModel,
}

impl std::fmt::Display for LoweringError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MalformedLoop {
                index,
                count,
                body_len,
                remaining,
            } => write!(
                f,
                "lowering: malformed Loop({count}, {body_len}) at tape index {index} \
                 — body extends past tape end (only {remaining} rows remain)"
            ),
            Self::TurboQuantOffsetUnbound { index } => write!(
                f,
                "lowering: the TurboQuant command at tape index {index} cannot bind its KV \
                 operand's additive offset (no rotary table rotates its K bias)"
            ),
            Self::NoSource { index, kind, slot } => write!(
                f,
                "lowering: the step at tape index {index} binds {kind:?} weight #{slot} of its \
                 site, which has no such source"
            ),
            Self::NoRotaryTable { index } => write!(
                f,
                "lowering: the attention at tape index {index} re-ropes on read, and the model \
                 has no rotary table"
            ),
            Self::DoubleGate { index } => write!(
                f,
                "lowering: the row at tape index {index} is gated, and its realization gates a \
                 command of its own"
            ),
            Self::RowZeroWithoutPerRowTwin { index, kernel } => write!(
                f,
                "lowering: `{kernel:?}` at tape index {index} computes sequence 0 only, and \
                 its step has no per-row twin re-roping span blocks for steps with \
                 several sequences"
            ),
            Self::TurboQuantCompressesNothing => f.write_str(
                "lowering: the model's KV codec is TurboQuant, but no KV writer in its \
                 backbone is one the codec compresses",
            ),
            Self::CodecStepOnDenseModel => f.write_str(
                "lowering: a KV codec step in the tape of a model whose KV cache is dense",
            ),
        }
    }
}

impl std::error::Error for LoweringError {}

// ── Static-tape classing (macro-baked, load-selected) ───────────────

/// Device-generation class a baked tape variant targets. The lowering's
/// only device-profile dependence is `AppleSiliconGen` (NAX capability
/// and the M1 bf16-simdgroup slow path), so three classes cover every
/// chip; variants that lower identically are deduped at expansion.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum GenClass {
    /// M1 family (no NAX, bf16 simdgroup emulated).
    M1,
    /// M2/M3/M4 (no NAX, native bf16 simdgroup).
    Mid,
    /// M5+ (NAX hardware MMA).
    M5,
}

impl GenClass {
    pub fn of(generation: crate::tape::targets::AppleSiliconGen) -> Self {
        use crate::tape::targets::AppleSiliconGen as G;
        match generation {
            G::M1 => GenClass::M1,
            G::M2 | G::M3 | G::M4 => GenClass::Mid,
            G::M5 => GenClass::M5,
        }
    }

    /// Whether this class stores MLX-affine 4-bit weight codes XOR 0x88
    /// (signed q - 8): the matrix unit's int8 x int4 lane reads them as
    /// stored (the W4A8 GEMM). The weight loader flips them on load and the
    /// lowering tells every other reader to flip them back
    /// ([`AffineCodes`](super::kernel_constants::AffineCodes)) — one fact,
    /// read by both.
    pub fn stores_affine_b4_offset8(self) -> bool {
        matches!(self, GenClass::M5)
    }
}

/// Which scalar of a command a [`CapPatch`] rewrites at load.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum PatchTarget {
    /// `constants[i].bits`.
    Constant(u32),
    /// `dispatch.threadgroups.{0,1,2}`.
    Threadgroups(u8),
    /// `dispatch.threads_per_threadgroup.{0,1,2}`.
    ThreadsPerThreadgroup(u8),
    /// `dispatch.m_scaling.bucket_m` (the rescale base — e.g. the
    /// rope-once `num_pages`).
    MScalingBucketM,
    /// `bindings[i]`'s `AttnUnfusedScratch { offset }` — the hd512
    /// unfused-attention scratch regions pack at byte offsets that
    /// scale with the KV capacity.
    AttnScratchOffset(u32),
}

/// One baked command scalar's dependence on the runtime block-table
/// capacity: `value(cap) = max(floor, base + (num·cap)/den)` (floor or
/// ceiling division per `round_up`). Derived at expansion by probing
/// the lowering at three capacities and verified at a fourth — a shape
/// the model can't fit refuses the bake.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
pub struct CapPatch {
    /// Index into `LoweredMetalTape::commands`.
    pub cmd_idx: u32,
    pub target: PatchTarget,
    pub floor: u32,
    pub base: i64,
    pub num: i64,
    pub den: u32,
    pub round_up: bool,
}

/// Which scratch-size field of [`LoweredMetalTape`] a [`ScratchPatch`]
/// re-derives at load.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
pub enum ScratchField {
    SplitK,
    Moe,
    RopedK,
    AttnUnfused,
}

/// A scratch-size field's dependence on the runtime block-table
/// capacity — same rational model as [`CapPatch`].
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
pub struct ScratchPatch {
    pub field: ScratchField,
    pub floor: u32,
    pub base: i64,
    pub num: i64,
    pub den: u32,
    pub round_up: bool,
}

/// One baked tape variant: the full [`LoweredMetalTape`] lowered at
/// expansion for a `(generation class, chunked addressing)` pair, with
/// block-capacity dependence expressed as patches. The pool picks the
/// matching variant at load and [`Self::materialize`]s it with the
/// runtime capacity and the device's TurboQuant decode heads — selection
/// and substitution only, no analysis.
#[derive(Clone, Copy, PartialEq, serde::Serialize)]
pub struct ClassedTape {
    pub gen_class: GenClass,
    /// `true` = the chunked-addressing attention variant (spec-decode
    /// forces it via `force_chunked_attention_addressing`).
    pub chunked: bool,
    /// Baked at capacity 0, so every patched location holds its floor.
    pub tape: LoweredMetalTape,
    pub const_patches: &'static [CapPatch],
    pub scratch_patches: &'static [ScratchPatch],
    /// The decode megakernel, one per KV mode — baked for the bucket-1, M5, direct-addressing
    /// variant only; empty everywhere else.
    pub megakernel: &'static [MegakernelTape],
}

fn patched(floor: u32, base: i64, num: i64, den: u32, round_up: bool, cap: u32) -> u32 {
    let prod = num * cap as i64;
    let den = den.max(1) as i64;
    let q = if round_up {
        prod.div_euclid(den) + if prod.rem_euclid(den) != 0 { 1 } else { 0 }
    } else {
        prod.div_euclid(den)
    };
    (floor as i64).max(base + q).try_into().unwrap_or(u32::MAX)
}

impl ClassedTape {
    /// Substitute the runtime block-table capacity, and the query heads
    /// each TurboQuant decode threadgroup serves on this device
    /// ([`crate::tape::lowering::serve_tq_decode_heads`]), into the baked
    /// tape. The ONE permitted load-time `baked` site: the commands are
    /// copied once per model load.
    pub fn materialize(
        &self,
        cap: u32,
        tq_heads: crate::tape::ids::TqDecodeHeads,
    ) -> LoweredMetalTape {
        let mut tape = self.tape;
        let mut commands: Vec<GatedCommand> = self.tape.commands.to_vec();
        for p in self.const_patches {
            let cmd = &mut commands[p.cmd_idx as usize];
            let v = patched(p.floor, p.base, p.num, p.den, p.round_up, cap);
            match p.target {
                PatchTarget::Constant(i) => {
                    let mut consts = cmd.command.constants.to_vec();
                    consts[i as usize].bits = v;
                    cmd.command.constants = baked(consts);
                }
                PatchTarget::Threadgroups(ax) => {
                    let tg = &mut cmd.command.dispatch.threadgroups;
                    match ax {
                        0 => tg.0 = v,
                        1 => tg.1 = v,
                        _ => tg.2 = v,
                    }
                }
                PatchTarget::ThreadsPerThreadgroup(ax) => {
                    let t = &mut cmd.command.dispatch.threads_per_threadgroup;
                    match ax {
                        0 => t.0 = v,
                        1 => t.1 = v,
                        _ => t.2 = v,
                    }
                }
                PatchTarget::MScalingBucketM => {
                    let ms = cmd
                        .command
                        .dispatch
                        .m_scaling
                        .as_mut()
                        .expect("MScalingBucketM patch on a command without m_scaling");
                    ms.bucket_m = crate::tape::ids::BucketM(v);
                }
                PatchTarget::AttnScratchOffset(bi) => {
                    let mut binds = cmd.command.bindings.to_vec();
                    match &mut binds[bi as usize] {
                        Binding::AttnUnfusedScratch { offset, .. } => *offset = v,
                        other => {
                            panic!("AttnScratchOffset patch on non-scratch binding {other:?}")
                        }
                    }
                    cmd.command.bindings = baked(binds);
                }
            }
        }
        for cmd in &mut commands {
            crate::tape::lowering::serve_tq_decode_heads(&mut cmd.command, tq_heads);
        }
        tape.commands = baked(commands);
        for p in self.scratch_patches {
            let v = patched(p.floor, p.base, p.num, p.den, p.round_up, cap);
            match p.field {
                ScratchField::SplitK => tape.splitk_scratch_bytes = v,
                ScratchField::Moe => tape.moe_scratch_bytes = v,
                ScratchField::RopedK => tape.roped_k_scratch_bytes = v,
                ScratchField::AttnUnfused => tape.attn_unfused_scratch_bytes = v,
            }
        }
        tape
    }
}

// ── The decode megakernel (compiled per bucket-1 tape at expansion, launched by the worker) ────

/// A span of [`LoweredMetalTape::commands_expanded`]: `[start, end)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub struct CommandSpan {
    pub start: u32,
    pub end: u32,
}

/// The decode megakernel of one bucket-1 tape, planned under [`GateCtx::decode_one`]: the WHOLE
/// decode forward as ONE generated kernel — every command the gates admit, as straight-line
/// adapter calls with every constant a literal, the tape's layer loops kept rolled — launched once
/// per forward in place of `commands`.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
pub struct MegakernelTape {
    /// Names the library `source` compiles to (a content hash): workers of one load share it.
    pub library: &'static str,
    /// The generated kernel; the worker appends it to the adapters' bodies at load.
    pub source: &'static str,
    /// Its `[[kernel]]` host name.
    pub kernel: &'static str,
    /// The expanded commands the launch replaces (the admitted ones run inside it).
    pub commands: CommandSpan,
    /// Every baked command the kernel plays, in tape order.
    pub steps: &'static [MkKernelStep],
    /// Addresses (`u64`) the kernel's address table holds.
    pub table_len: u32,
    /// Grid barriers one launch crosses, loops played out.
    pub grid_barriers: u32,
    /// The kernel's function constants the load supplies.
    pub load_constants: &'static [MkLoadConstant],
}

/// A baked command the kernel plays: `baked` indexes [`LoweredMetalTape::commands`]; its `k`-th
/// expanded instance (loop iteration) reads binding `i`'s address at table entry
/// `table_at + k · row_len + i`. Named, so the load checks it plays what the bake saw.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub struct MkKernelStep {
    pub baked: u32,
    pub function: &'static str,
    pub table_at: u32,
    pub row_len: u32,
}

/// A scalar the load decides — the KV capacity's patches, the device's TurboQuant decode heads —
/// read from the materialized command at `baked` into function constant `index` of the kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub struct MkLoadConstant {
    pub index: ConstSlot,
    pub baked: u32,
    pub source: MkLoadSource,
}

/// Where a [`MkLoadConstant`] is read.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum MkLoadSource {
    /// The command's function constant at this slot.
    Constant(ConstSlot),
    /// The command's threadgroups along this axis.
    Threadgroups(MScaleAxis),
}

impl LoweredCommand {
    /// What `materialize` sets from the device, beyond the capacity patches: the TurboQuant
    /// decode heads ([`crate::tape::lowering::serve_tq_decode_heads`]) — a constant and the grid's
    /// y axis.
    pub fn device_served(&self) -> &'static [MkLoadSource] {
        const HEADS: &[MkLoadSource] = &[
            MkLoadSource::Constant(
                crate::tape::kernel_constants::AttentionViaCacheTqConstants::HEADS,
            ),
            MkLoadSource::Threadgroups(MScaleAxis::Y),
        ];
        if self.kernel == KernelId::AttentionViaCacheTq {
            HEADS
        } else {
            &[]
        }
    }
}

/// Threadgroup memory one virtual threadgroup of an adapter owns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VtgBytes(pub u32);

/// Binding indices, one bit each.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BindingMask(pub u32);

impl BindingMask {
    pub fn contains(self, binding_index: u8) -> bool {
        binding_index < 32 && self.0 & (1 << binding_index) != 0
    }
}

/// A function constant of a dispatch kernel, as its adapter's policy names it.
#[derive(Clone, Copy, Debug)]
pub struct MkConst {
    pub slot: ConstSlot,
    pub ty: ConstantType,
    /// The policy's accessor (`C::name()`).
    pub name: &'static str,
}

/// A kernel a megakernel can call: one `MK_ADAPTER` line of a normalized shader, enumerated by
/// build.rs from the shader's own instantiation macros (`MK_ADAPTERS`, generated).
#[derive(Clone, Copy, Debug)]
pub struct MkAdapter {
    /// The dispatch kernel's library and host name — what a command names.
    pub library: &'static str,
    pub function: &'static str,
    pub tg_bytes: VtgBytes,
    /// Bindings the adapter reads or writes device-coherently.
    pub coherent: BindingMask,
    /// Threads one item plays at most: [`MK_THREADS`], or fewer for a streaming body that streams
    /// its weights faster with fewer live threads per core (`MK_STREAM` in the shader, measured).
    pub item_threads: u32,
    /// Bindings the adapter may write (every one, unless its `MK_STREAM` line says which).
    pub writes: BindingMask,
    /// An elementwise adapter (`MK_TAIL`): a step of a few of its items may be played whole by one
    /// threadgroup.
    pub tail: bool,
    /// A streaming body whose fastest width depends on its row length (`MK_STREAM_ROWS`).
    pub short_rows: Option<ShortRows>,
    /// The dispatch kernel's function constants.
    pub constants: &'static [MkConst],
    /// The adapter, `MK_C` standing for the step's constant policy.
    pub call: &'static str,
}

/// The width a streaming body plays SHORT rows at: a step whose constant at `row` is below `below`
/// plays items of up to `item_threads` (few passes per row: more rows in flight hide each row's
/// latency).
#[derive(Clone, Copy, Debug)]
pub struct ShortRows {
    pub item_threads: u32,
    pub row: ConstSlot,
    pub below: u32,
}

include!(concat!(env!("OUT_DIR"), "/mk_adapters.rs"));

impl MkAdapter {
    /// Threads one item of a step with `constants` plays at most: [`Self::item_threads`], or the
    /// [`ShortRows`] width for a short-row step.
    pub fn item_threads_for(&self, constants: &[ConstantValue]) -> u32 {
        let short = self.short_rows.filter(|r| {
            constants
                .iter()
                .any(|c| c.index == r.row.get() && c.bits < r.below)
        });
        short.map_or(self.item_threads, |r| r.item_threads)
    }

    /// The adapter of the kernel a command names, if a megakernel can call it.
    pub fn of(library: &str, function: &str) -> Option<&'static MkAdapter> {
        MK_ADAPTERS
            .iter()
            .find(|a| (a.library, a.function) == (library, function))
    }
}

/// Threads of one persistent (physical) threadgroup of the megakernel.
pub const MK_THREADS: u32 = 1024;
/// Virtual threadgroups start on simdgroup boundaries.
pub const MK_SIMD_WIDTH: u32 = 32;
/// Threadgroup memory the steps of one persistent threadgroup may use: the 32 KiB a threadgroup
/// has, less the 16 bytes the generated kernel keeps for its barrier flag.
pub const MK_TG_MEMORY: u32 = 32 * 1024 - 16;
/// The function constants of every generated kernel (`megakernel.metal`): the persistent
/// threadgroups `MK_P` and the grid-barrier spin bound `MK_SPIN_LIMIT`; the load constants
/// ([`MkLoadConstant`]) follow from `MK_FC_LOAD`.
pub const MK_FC_P: ConstSlot = ConstSlot(4096);
pub const MK_FC_SPIN_LIMIT: ConstSlot = ConstSlot(4097);
pub const MK_FC_LOAD: ConstSlot = ConstSlot(4200);

/// How a step packs into items of at most [`MK_THREADS`] threads (its adapter's `item_threads`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MkGeometry {
    /// Threads per virtual threadgroup, rounded up to a simdgroup.
    pub vtg_stride: u32,
    /// Virtual threadgroups one item plays.
    pub vtgs_per_item: u32,
    /// Items of the step.
    pub items: scratchy_subtile::megakernel_plan::Items,
    /// Threadgroup memory one item needs: a region per virtual threadgroup, plus the sink the
    /// lanes past the last one share.
    pub tg_memory: u32,
}

/// The geometry of a step dispatched as `grid` threadgroups of `tpg` threads, each owning
/// `tg_bytes` of threadgroup memory, played in items of at most `item_threads` threads (at least
/// one virtual threadgroup).
pub fn mk_geometry(
    grid: (u32, u32, u32),
    tpg: (u32, u32, u32),
    tg_bytes: VtgBytes,
    item_threads: u32,
) -> Result<MkGeometry, MegakernelError> {
    let threads = tpg.0 * tpg.1 * tpg.2;
    if threads == 0 || threads > MK_THREADS {
        return Err(MegakernelError::ThreadCap {
            needed: threads,
            cap: MK_THREADS,
        });
    }
    let vtg_stride = threads.next_multiple_of(MK_SIMD_WIDTH);
    let regions = |k: u32| {
        if k * vtg_stride < MK_THREADS {
            k + 1
        } else {
            k
        }
    };
    let fits = |k: &u32| regions(*k) * tg_bytes.0 <= MK_TG_MEMORY;
    let live = item_threads.clamp(vtg_stride, MK_THREADS);
    let k = (1..=live / vtg_stride).rev().find(fits);
    let k = k.ok_or(MegakernelError::ThreadgroupMemory {
        needed: 2 * tg_bytes.0,
        budget: MK_TG_MEMORY,
    })?;
    let vtgs = u64::from(grid.0) * u64::from(grid.1) * u64::from(grid.2);
    let items = u32::try_from(vtgs.div_ceil(u64::from(k))).ok();
    let items = items.and_then(std::num::NonZeroU32::new);
    let items = items.ok_or(MegakernelError::EmptyGrid)?;
    Ok(MkGeometry {
        vtg_stride,
        vtgs_per_item: k,
        items: scratchy_subtile::megakernel_plan::Items(items),
        tg_memory: regions(k) * tg_bytes.0,
    })
}

/// Why the megakernel cannot bake, load or finish — never a silent fallback: a bucket-1 decode
/// tape the megakernel cannot play fails the build.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MegakernelError {
    /// A decode command whose kernel has no megakernel adapter (`MK_ADAPTER` line).
    NoAdapter {
        library: &'static str,
        function: &'static str,
    },
    /// A decode command whose kernel is classed as never running in a bucket-1 decode forward.
    NotAtDecode {
        kernel: KernelId,
        function: &'static str,
        class: &'static str,
    },
    /// A command whose gate at a one-token decode reads whether the sequence holds an unrotated
    /// span block: the kernel plays the same either way, so no gate there may.
    SpanDependentGate { function: &'static str },
    /// A decode command's load patch the kernel cannot take as a function constant.
    LoadPatch {
        symbol: &'static str,
        target: PatchTarget,
    },
    /// A location the plan shares between threadgroups is bound by an adapter that does not
    /// access it device-coherently.
    IncoherentShared { symbol: &'static str, binding: u8 },
    /// A command binds a location its step row's dataflow does not state.
    UnstatedLocation { symbol: &'static str, binding: u8 },
    /// Two commands touch the KV codec's shared scratch with neither ordered after the other.
    UnorderedScratch {
        first: &'static str,
        second: &'static str,
    },
    /// A step's threadgroup, or the persistent threadgroup the pipeline can launch.
    ThreadCap { needed: u32, cap: u32 },
    /// Threadgroup memory beyond what a persistent threadgroup has.
    ThreadgroupMemory { needed: u32, budget: u32 },
    /// A command's constant disagrees in type with the adapter's declared constant.
    ConstantType {
        symbol: &'static str,
        index: u16,
        declared: ConstantType,
        given: ConstantType,
    },
    /// A step dispatches no threadgroup.
    EmptyGrid,
    /// A command the kernel plays is not the kernel the bake generated the step for.
    AdapterMismatch {
        expected: &'static str,
        found: &'static str,
    },
    /// A load constant's source is not on its materialized command.
    LoadConstant { symbol: &'static str },
    /// The generated library failed to compile at load.
    Compile(String),
    /// Not every persistent threadgroup of the launch checked in at its first grid barrier (site
    /// `site`): the GPU did not run all `of` of them at once (`arrived` had).
    CoResidency { site: u32, arrived: u32, of: u32 },
    /// The launch's `ordinal`-th grid barrier (site `site` of the kernel text) gave up waiting
    /// (`arrived` of `of`).
    Stall {
        site: u32,
        ordinal: u32,
        arrived: u32,
        of: u32,
    },
}

impl std::fmt::Display for MegakernelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoAdapter { library, function } => write!(
                f,
                "megakernel: `{library}::{function}` has no megakernel adapter (MK_ADAPTER)"
            ),
            Self::NotAtDecode {
                kernel,
                function,
                class,
            } => write!(
                f,
                "megakernel: `{function}` ({kernel:?}) runs in the bucket-1 decode forward, but \
                 its kernel is classed {class}"
            ),
            Self::SpanDependentGate { function } => write!(
                f,
                "megakernel: `{function}`'s gate at a one-token decode depends on whether the \
                 sequence holds an unrotated span block"
            ),
            Self::LoadPatch { symbol, target } => write!(
                f,
                "megakernel: `{symbol}`: load patch {target:?} is not a function constant"
            ),
            Self::IncoherentShared { symbol, binding } => write!(
                f,
                "megakernel: `{symbol}` binds a location threadgroups share at {binding}, which \
                 its adapter does not access device-coherently"
            ),
            Self::UnstatedLocation { symbol, binding } => write!(
                f,
                "megakernel: `{symbol}` binds a location at {binding} its step row does not state"
            ),
            Self::UnorderedScratch { first, second } => write!(
                f,
                "megakernel: `{first}` and `{second}` touch the KV codec's shared scratch unordered"
            ),
            Self::ThreadCap { needed, cap } => {
                write!(f, "megakernel: {needed} threads per threadgroup, cap {cap}")
            }
            Self::ThreadgroupMemory { needed, budget } => write!(
                f,
                "megakernel: {needed} bytes of threadgroup memory, budget {budget}"
            ),
            Self::ConstantType {
                symbol,
                index,
                declared,
                given,
            } => write!(
                f,
                "megakernel: `{symbol}` constant {index} is {given:?}, its adapter declares \
                 {declared:?}"
            ),
            Self::EmptyGrid => write!(f, "megakernel: a step dispatches no threadgroup"),
            Self::AdapterMismatch { expected, found } => write!(
                f,
                "megakernel: a kernel step generated for `{expected}` would play `{found}`"
            ),
            Self::LoadConstant { symbol } => write!(
                f,
                "megakernel: a load constant of `{symbol}` is not on its materialized command"
            ),
            Self::Compile(e) => write!(f, "megakernel: the generated library: {e}"),
            Self::CoResidency { site, arrived, of } => write!(
                f,
                "megakernel: only {arrived} of {of} persistent threadgroups checked in at the \
                 first grid barrier (site {site}): not co-resident"
            ),
            Self::Stall {
                site,
                ordinal,
                arrived,
                of,
            } => write!(
                f,
                "megakernel: grid barrier {ordinal} (site {site}) gave up ({arrived} of {of} \
                 threadgroups arrived)"
            ),
        }
    }
}

impl std::error::Error for MegakernelError {}

#[cfg(test)]
mod megakernel_tests {
    use super::*;

    fn g(grid: (u32, u32, u32), tpg: (u32, u32, u32), bytes: u32) -> (u32, u32, u32, u32) {
        streamed(grid, tpg, bytes, MK_THREADS)
    }

    fn streamed(
        grid: (u32, u32, u32),
        tpg: (u32, u32, u32),
        bytes: u32,
        item_threads: u32,
    ) -> (u32, u32, u32, u32) {
        let g = mk_geometry(grid, tpg, VtgBytes(bytes), item_threads).expect("fits");
        (g.vtg_stride, g.vtgs_per_item, g.items.0.get(), g.tg_memory)
    }

    /// A qmv step: 64-thread threadgroups, sixteen to an item, no threadgroup memory.
    #[test]
    fn small_threadgroups_pack_on_simdgroup_boundaries_into_one_item() {
        assert_eq!(g((1, 384, 1), (32, 2, 1), 0), (64, 16, 24, 0));
        // 96 threads: ten virtual threadgroups, the 64 lanes over share a sink region.
        assert_eq!(g((25, 1, 1), (96, 1, 1), 1024), (96, 10, 3, 11 * 1024));
    }

    /// A streaming body's items play at most its measured width, never less than one virtual
    /// threadgroup.
    #[test]
    fn a_streaming_step_packs_to_its_item_width() {
        assert_eq!(streamed((1, 384, 1), (32, 2, 1), 0, 256), (64, 4, 96, 0));
        assert_eq!(streamed((1, 384, 1), (32, 2, 1), 0, 512), (64, 8, 48, 0));
        assert_eq!(streamed((4, 1, 1), (512, 1, 1), 0, 256), (512, 1, 4, 0));
    }

    /// A norm owns its `shared_sum[1024]` per virtual threadgroup; a budget that cannot hold
    /// every region packs fewer threadgroups per item and keeps the sink.
    #[test]
    fn threadgroup_memory_bounds_the_virtual_threadgroups_of_an_item() {
        assert_eq!(g((1, 1, 1), (256, 1, 1), 4096), (256, 4, 1, 4 * 4096));
        assert_eq!(g((8, 1, 1), (128, 1, 1), 6144), (128, 4, 2, 5 * 6144));
        let too_big = mk_geometry((1, 1, 1), (1024, 1, 1), VtgBytes(40 * 1024), MK_THREADS);
        assert!(matches!(
            too_big,
            Err(MegakernelError::ThreadgroupMemory { .. })
        ));
        let wide = mk_geometry((1, 1, 1), (1025, 1, 1), VtgBytes(0), MK_THREADS);
        assert!(matches!(wide, Err(MegakernelError::ThreadCap { .. })));
        let empty = mk_geometry((0, 1, 1), (64, 1, 1), VtgBytes(0), MK_THREADS);
        assert_eq!(empty, Err(MegakernelError::EmptyGrid));
    }

    /// At a one-token decode of one sequence every gate but the span one is a constant.
    #[test]
    fn a_one_token_decode_admits_by_the_gate_alone() {
        use RuntimeGate as G;
        for ctx in [false, true].map(GateCtx::decode_one) {
            assert!(G::OnlyIfDecodeStep.admits(ctx) && !G::UnlessDecodeStep.admits(ctx));
            assert!(!G::OnlyIfNoSpec.admits(ctx) && !G::OnlyIfSpec.admits(ctx));
            assert!(G::OnlyIfOneSequence.admits(ctx) && !G::UnlessOneSequence.admits(ctx));
        }
    }
}
