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

use crate::tape::constants::ConstantValue;
use crate::tape::ids::{BucketM, LayerId};

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
/// online. Variants the lowering pass cannot yet produce surface as a
/// `LoweringError::UnsupportedVariant` rather than appearing here as a
/// stub.
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
    /// Fused QKV matmul + NeoX-style RoPE + paged KV-cache write in
    /// one kernel. Replaces the four-dispatch
    /// `Q_proj + K_proj + V_proj + RopeAppend` chain on the dense
    /// (BF16 / F16) path. Affine-int4 / prefill variants land
    /// separately. Maps to
    /// `fused_qkv_rope_cache_<dtype>_specialized` in
    /// `fused_qkv_rope_cache.metallib`.
    FusedQkvRopeCache,
    /// Affine-int4 sibling of [`KernelId::FusedQkvRopeCache`]. Reads
    /// packed `u32` weights + per-group F16 scales/biases (mlx-community
    /// 4bit layout) for Q/K/V concatenated along the output axis, fuses
    /// the dequant→matmul→RoPE→paged-cache-write chain in a single
    /// launch. Maps to
    /// `fused_affine_qkv_rope_cache_<dtype>_s_<scale_dtype>_b_4_specialized`
    /// in `fused_affine_qkv_rope_cache.metallib`. Group size rides on
    /// function constant 7.
    FusedAffineQkvRopeCache,
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
    /// `rope_once_nax_<dtype>_bd<head_dim>_bs16` in the runtime-compiled
    /// `attention_steel_nax_paged` library.
    RopeOnceNax,
    /// Spans rope-on-read (SIMDGROUP steel prefill): the simdgroup twin of
    /// `RopeOnceNax`. Ropes one request+layer's K from the paged cache ONCE
    /// into the shared `Binding::RopedKScratch` buffer, so the
    /// `attention_steel_paged` attention that follows reads pre-roped K with no
    /// per-tile rotation. Emitted only when `use_steel && !use_nax &&
    /// W::ROPE_ON_READ`. Maps to `rope_once_steel_<dtype>_bd<head_dim>_bs16` in
    /// the runtime-compiled `attention_steel_paged` library (covers the
    /// simdgroup head dims 64/96/128/256, incl. SmolLM's hd64).
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
    /// Per-row bias broadcast add: `out[m, n] = in[m, n] + bias[n]`.
    /// Singleton claim used by `MetalBiasAddImpl` for Qwen2/Qwen2.5
    /// (and any other) QKV biases that the synth megakernel doesn't
    /// absorb at the current bucket M (e.g. M ≥ 2 prefill where the
    /// solver's cost CSV picks unfused). Maps to
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
    /// NVFP4 int4 decode matvec (generic). Maps to
    /// `nvfp4_qmv_<dtype>_s_<scale>_gs_16_b_4_batch_0` in the
    /// `quantized_qmv.metallib` (nvfp4 kernels share that library with
    /// the affine `qmv` ones). Same structure as `AffineQmv`; the only
    /// difference is the E2M1-LUT weight decode (no per-group bias).
    Nvfp4Qmv,
    /// NVFP4 int4 prefill matmul (transpose=true, standard tile). Maps
    /// to `nvfp4_qmm_t_<dtype>_s_<scale>_gs_16_b_4_alN_<bool>_batch_0`
    /// in `quantized_qmm.metallib`. Mirrors `AffineQmmT`.
    Nvfp4QmmT,
    /// NVFP4 int4 prefill matmul on NAX (Apple9 / M4+) — 64×64 MPP
    /// matmul2d tile. Maps to
    /// `nvfp4_qmm_t_nax_<dtype>_s_<scale>_gs_16_b_4_alN_<bool>_batch_0`
    /// in `quantized_qmm_nax.metallib`. Dispatched (in place of
    /// `Nvfp4QmmT`) when `is_nax_capable(profile.generation)` and
    /// `K % 64 == 0`. Mirrors `AffineQmmTNax`.
    Nvfp4QmmTNax,
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
    /// Compiler-synthesized pre-attention megakernel. Symbol resolves
    /// against a per-arch source-compiled library registered at worker
    /// init via `SpecializedPipelineCache::register_source_library`.
    /// Kernel body is generated at macro-expansion time by
    /// `scratchy-forward-compiler-macro::fuse_pass`.
    SynthPreAttn,
    /// Compiler-synthesized MLP pre-down megakernel. Symbol resolves
    /// against a per-arch source-compiled library registered at worker
    /// init via `SpecializedPipelineCache::register_source_library`.
    /// Kernel body is generated at macro-expansion time by
    /// `scratchy-forward-compiler-macro::fuse_pass::synthesize_mlp_pre_down_chunk`.
    /// Fuses `FusedAddRmsNorm + gate AffineQmv + up AffineQmv + SiluMul`
    /// into one dispatch; the standalone `AffineQmm` down_proj
    /// instruction follows immediately and consumes the device-buffer
    /// `silu_mul` output.
    SynthMlpPreDown,
    /// Fused gate+up GEMM + SiluMul large-M prefill kernel.
    SynthGateUpSiluMul,
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
    /// 2-D non-overlapping average pool (`avg_pool_2d.metal`) — the
    /// Gemma3-MM SigLIP→text projector's k×k spatial collapse.
    AvgPool2d,
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
    /// TurboQuant: dequant a layer's PACKED KV codes into the reused fp16
    /// scratch (one layer at a time) right BEFORE that layer's attention.
    /// Maps to `tq_dequant_paged[_bf16]` in `turboquant.metallib`. The packed
    /// store (canonical, ~4.7x smaller) is the only persistent KV; the scratch
    /// holds one layer's fp16 for the attention read, then is reused.
    TqDequantToScratch,
    /// TurboQuant: quantize a layer's newly-written KV (in the fp16 scratch)
    /// into the PACKED store right AFTER that layer's rope-append-cache write.
    /// Maps to `tq_compress_paged[_bf16]` in `turboquant.metallib`.
    TqQuantizeToPacked,
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
    /// Run only when the KV cache is TurboQuant-compressed. The per-layer
    /// dequant/quantize commands are emitted unconditionally by the lowering
    /// but only fire when `kv_cache_dtype == turboquant` (the worker resolves
    /// the tq buffers + the KV scratch only then). No-op on every other run.
    OnlyIfTurboquant,
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
/// Weight bindings carry a `(bucket, op_idx, slot, kind)` locator the
/// worker passes into the per-arch [`scratchy_ir::WeightAccessors`] impl at
/// dispatch-record time to recover the `&Layer` struct (`linear_at`,
/// `rms_norm_at`, etc.). The resolved tensor's pointer is then looked
/// up against the `MetalAllocator`'s arena registry. No fn pointers
/// live here, so `Binding` is fully backend-neutral.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
pub enum Binding {
    /// `MetalWorker.arena[slot]` — the worker's private tile-arena
    /// buffer for this slot. The arena is sized for the colored
    /// `NUM_TILES` post-FUF coloring (linear-scan reg allocation
    /// performed by `colored_slot_map()` in
    /// `scratchy-forward-compiler-macro/src/interpreter_codegen.rs`).
    ArenaSlot { slot: u32, binding_index: u8 },
    /// A model tensor named by its SOURCE INDEX in the macro-emitted
    /// static tape. The player resolves it ONCE at load through the
    /// tape's generated resolver fn (the macro knows every source's
    /// field path at expansion) — there is no locator, no accessor
    /// table, and no per-dispatch lookup on this path. The
    /// accessor-based `Weight` variant above serves only the legacy
    /// runtime-lowered tapes and dies with them.
    Source { ix: u32, binding_index: u8 },
    /// A weight bundle resolved at dispatch-record time through the per-arch
    /// [`scratchy_ir::WeightAccessors`] impl. `locator` is the
    /// `(bucket, op_idx, slot)` triple the macro baked at codegen time;
    /// the trait method picked by `kind` returns the named field on
    /// `Weights`. `which` then selects which tensor inside the bundle
    /// to bind (e.g. weight vs bias vs affine scales).
    Weight {
        kind: WeightBundleKind,
        which: WeightTensor,
        layer: LayerId,
        locator: WeightLocator,
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

/// Per-bundle locator for the macro-emitted `WeightAccessors` impl.
/// The worker invokes the matching `<kind>_at(bucket, op_idx, slot,
/// layer)` method to recover the `&Layer` reference.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub struct WeightLocator {
    /// Bucket id baked at codegen — matches the bucket axis of the
    /// per-arch `WeightAccessors` match arm.
    pub bucket: u32,
    /// Flat tape position of the source `Instruction` (post loop
    /// unrolling for backbone slices; absolute index in the lm_head
    /// slice for lm_head).
    pub op_idx: u32,
    /// Sub-position within the same `(bucket, op_idx)` for variants
    /// that resolve multiple accessors of the same kind — e.g.
    /// `SynthPreAttn` consumes 3 `LinearLayer`s (Q/K/V at slots 0/1/2).
    pub slot: u32,
}

/// Discriminator selecting which per-arch [`scratchy_ir::WeightAccessors`]
/// method the worker should call to resolve this binding.
///
/// Pure tag — fn pointers are gone after the lift; weight resolution
/// lives at the tape level via `(bucket, op_idx, slot)` keys.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum WeightBundleKind {
    /// Calls `WeightAccessors::embedding_at` → `&Embedding`.
    Embedding,
    /// Calls `WeightAccessors::rms_norm_at` → `&RmsNorm`.
    RmsNorm,
    /// Calls `WeightAccessors::linear_at` → `&LinearLayer`.
    LinearLayer,
    /// Calls `WeightAccessors::cos_sin_at` → `GpuTensor` (per-layer
    /// RoPE table; no struct wrapper).
    CosSin,
    /// Spans rope-on-read: the cos/sin table the paged attention
    /// kernels re-rope cached K with. Resolved by LAYER CLASS via
    /// `WeightAccessors::rope_on_read_cos_sin(is_global, layer)`
    /// (`is_global` = global/full-attention `rotary` vs sliding
    /// `rotary_local`), not by an op operand — attention ops carry no
    /// cos_sin weight slot. The binding's `locator` is unused. Only
    /// emitted when `W::ROPE_ON_READ`.
    RopeOnReadCosSin { is_global: bool },
    /// MLX-affine int4 quantized embedding (Metal-only). Calls
    /// `WeightAccessors::affine_quant_embedding_at` →
    /// `&AffineQuantEmbedding`. P6 macro emission decides between this
    /// and `Embedding` per-layer based on safetensors layout (U32
    /// weight ⇒ AffineQuantEmbedding, else Embedding).
    AffineQuantEmbedding,
    /// Mixtral-style fused MoE bundle (no shared expert). Metal-only —
    /// resolves through `WeightAccessors::fused_moe_at(...)` to a
    /// `&FusedMoELayer` whose Metal arm carries packed per-expert
    /// {gate, up, down} weight slabs in MLX-affine int4 layout plus
    /// the dense router gate. The Metal worker treats this as the
    /// arbitrator for the 19-ish `WeightTensor::Moe*` variants below.
    FusedMoe,
    /// Qwen-MoE-style fused MoE + shared expert bundle. Metal-only —
    /// resolves through `WeightAccessors::shared_fused_moe_at(...)`.
    /// Carries the same per-expert slabs as `FusedMoe` plus the
    /// shared-expert {gate_up, down} affine slabs + the dense
    /// `shared_expert_gate` (`[1, hidden]` sigmoid gate). On variants
    /// where `shared_expert_intermediate_size == 0` (modern
    /// Qwen3-MoE-30B-A3B), the shared-expert tensors are absent and
    /// the lowering arm skips the shared-expert tail.
    SharedFusedMoe,
    /// Gemma-4 router bundle (`GemmaMoe` op). Resolves through
    /// `WeightAccessors::gemma_router_at(...)` to a `&GemmaRouterLayer`
    /// carrying the dequant'd dense router gate, the per-expert score scale,
    /// and the RMSNorm gain. The `WeightTensor::GemmaRouter{Gate,PerExpertScale,Scale}`
    /// variants select which sub-tensor.
    GemmaRouter,
    /// Gemma-4 SwitchGLU experts bundle (`GemmaMoe` op). Resolves through
    /// `WeightAccessors::gemma_switch_glu_at(...)` to a
    /// `&SwitchGluExpertsLayer` carrying the stacked 4-bit per-expert
    /// gate/up/down. Reuses the `WeightTensor::MoeExpert{Gate,Up,Down}{W,S,B}`
    /// variants (same expert-major affine layout as the routed FusedMoe path).
    GemmaSwitchGlu,
    /// Qwen3.5 / Qwen3-Next Gated-DeltaNet per-layer weight bundle.
    /// Resolves through `WeightAccessors::gated_delta_net_at(...)` to a
    /// `&GatedDeltaNetLayer` (conv1d / A_log / dt_bias / norm); the
    /// `WeightTensor::Gdn*` variants select which sub-tensor.
    GatedDeltaNet,
    /// Torch-style LayerNorm-with-bias bundle (vision towers:
    /// Qwen3.5-VL norm1/norm2/merger.norm). Resolves through
    /// `WeightAccessors::layer_norm_at(...)` to a `&LayerNorm`
    /// (`.weight` + `.bias`); `WeightTensor::Weight` selects the gain
    /// and `WeightTensor::Bias` the (mandatory) bias.
    LayerNorm,
}

/// Which tensor inside a multi-tensor weight bundle this binding
/// references. Most bundles have a single weight tensor (`Weight`);
/// MLX-affine `LinearLayer::AffineQuant` carries four (packed weight,
/// per-group scales, per-group affine offsets, optional fp linear bias).
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
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
    /// Per-group folded scales on an NVFP4 LinearLayer
    /// (`Nvfp4Linear.scales`, `[N, K/group_size]` F16 —
    /// `e4m3_to_f32(weight_scale) / weight_global_scale`). Only valid
    /// against `LinearLayer::Nvfp4`. NVFP4 has no per-group bias, so
    /// there is no `Nvfp4Biases` counterpart; the packed weight is
    /// fetched via the shared `WeightTensor::Weight`.
    Nvfp4Scales,
    // ── MoE bundle tensors ──────────────────────────────────────────
    //
    // Valid only against `WeightBundleKind::{FusedMoe, SharedFusedMoe}`.
    // The Metal worker resolves these against the MetalSwitchGluMoeWeights
    // payload on the FusedMoELayer / SharedFusedMoELayer struct. Each
    // names a concrete tensor; the worker reads its arena-backed pointer
    // and stamps it onto the encoder.
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
    // Valid only against `WeightBundleKind::GatedDeltaNet`. The worker
    // resolves these against the `GatedDeltaNetLayer` struct returned by
    // `WeightAccessors::gated_delta_net_at`.
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
    /// without an in-kernel band-split (option (b)). The worker
    /// `resolve_bindings` redirects the baked `WeightBundleKind::CosSin`
    /// pointer to this buffer when `W::MROPE_SECTION.is_some()`; the macro
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
    /// u32 SigLIP positional-embedding indices read by `PosEmbed`
    /// (`[0..vision_num_positions]` per image). Gemma3-MM.
    VisionPositionIds,
    /// TurboQuant per-layer PACKED key/value code store (canonical KV, ~4.7x
    /// smaller than fp16). Source for `TqDequantToScratch`, dest for
    /// `TqQuantizeToPacked`. Worker resolves to `tq_packed_k/v[layer]`.
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
    /// registers). Empty for `KernelId::Gemm` (no entry; routed
    /// out-of-band).
    ///
    /// [`SpecializedPipelineCache::with_standard_shaders`]: crate::specialized_pipeline_cache::SpecializedPipelineCache::with_standard_shaders
    pub library: &'static str,
    /// MSL `kernel void` symbol the pipeline binds. Empty for
    /// `KernelId::Gemm`.
    pub function: &'static str,
    /// `[[function_constant(N)]]` bag the pipeline specializes on.
    /// Pre-baked at lowering time from `W::*` + bucket_m so the worker
    /// never reaches back into `CanonicalParams`. Empty Vec is valid
    /// (e.g. `Add`, `ScalarMul`); empty constants AND empty function
    /// name signals "this is the opaque GEMM path."
    pub constants: &'static [ConstantValue],
    pub dispatch: DispatchShape,
    pub bindings: &'static [Binding],
    /// Dense-GEMM dimensions when `kernel == KernelId::Gemm`; `None`
    /// for every other kernel. The worker reads `(m, n, k)` from
    /// here when encoding the MPS dispatch (5.C.5 routing).
    /// Carried on the lowered command rather than baked into
    /// `DispatchShape` because MPS does not consume threadgroup
    /// counts — the dimensions are the actual API parameters.
    pub gemm_dims: Option<GemmDims>,
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
    /// lowering time (qmv / qmm_t / synth_*) keep the struct-literal
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
            gemm_dims: None,
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

/// Dense-GEMM dimensions for `KernelId::Gemm`.
///
/// `out = in @ weight^T` for the canonical row-major Linear layer:
/// `in: [m, k]`, `weight: [n, k]`, `out: [m, n]`. Future quantized
/// or transposed variants get sibling structs once they land.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
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
            | Self::VisionPositionIds
            | Self::TqSigns
            | Self::TqBoundaries
            | Self::TqCentroids => self,
        }
    }
}

impl Binding {
    /// Advance every layer this binding names by `by` loop iterations.
    pub fn bump_layer(self, by: u32) -> Self {
        match self {
            Self::Weight {
                locator,
                layer,
                kind,
                which,
                binding_index,
            } => Self::Weight {
                locator,
                layer: layer.advance(by),
                kind,
                which,
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
            | Self::Source { .. }
            | Self::Scratch { .. }
            | Self::RopedKScratch { .. }
            | Self::AttnUnfusedScratch { .. }
            | Self::Inline { .. }
            | Self::MoeScratch { .. } => self,
        }
    }
}

/// ⭐ THE LAYER LOOP, KEPT IN THE BAKED TAPE INSTEAD OF UNROLLED INTO IT.
///
/// `commands[start .. start + period]` is ONE copy of the body; it runs `iters` times, and the
/// only thing that differs between iterations is the layer — `lower_one`'s `layer_offset` is
/// used in exactly one pattern, `LayerId(literal + layer_offset)`, and the weight locator's
/// `op_idx` is the BODY position, identical across iterations. So iteration `i` is the baked
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
    /// in exactly one pattern (`LayerId(literal + layer_offset)`); the weight locator's
    /// `op_idx` is the body position and does not vary.
    pub fn commands_expanded(&self) -> Vec<GatedCommand> {
        let mut out = Vec::with_capacity(self.commands.len());
        let cmds = self.commands;
        Self::walk(self.loops, 0..cmds.len(), 0, &mut |pos, off| {
            out.push(cmds[pos].with_layer_bumped(off))
        });
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

/// Errors produced by the lowering pass.
///
/// `LoweringError::UnsupportedVariant` is the most common case during
/// Phase 5.A: each model adds new `Instruction<W>` variants that the
/// TinyLlama-only lowering doesn't yet handle. The variant name and
/// the `Instruction<W>` index are surfaced so the model author knows
/// exactly which instruction to add support for.
#[derive(Debug)]
pub enum LoweringError {
    /// The lowering pass doesn't yet handle this `Instruction<W>`
    /// variant. Add a match arm in `lowering::lower_one()` and a
    /// matching kernel in `KernelId`.
    UnsupportedVariant {
        /// Index of the offending instruction in the source tape.
        index: usize,
        /// `std::any::type_name_of_val()` of the offending variant
        /// — gives the variant name without requiring `Debug`.
        variant_type: &'static str,
    },
    /// A `Loop(count, body_len)` instruction overran the source tape
    /// when unrolling: the body extended past the end of the slice.
    /// Indicates malformed codegen — every tape the macro produces
    /// has been validated by the time it reaches lowering.
    MalformedLoop {
        index: usize,
        count: u32,
        body_len: u32,
        remaining: usize,
    },
}

impl std::fmt::Display for LoweringError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedVariant {
                index,
                variant_type,
            } => write!(
                f,
                "lowering: unsupported instruction variant `{variant_type}` at tape index {index} \
                 (TinyLlama-1.1B critical path is the only set covered in Phase 5.A; \
                 add a match arm in `interpreter::metal::lowering::lower_one`)"
            ),
            Self::MalformedLoop {
                index,
                count,
                body_len,
                remaining,
            } => write!(
                f,
                "lowering: malformed Loop({count}, {body_len}) at tape index {index} \
                 — body extends past tape end (only {remaining} instructions remain)"
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
}

/// Which scalar of a command a [`CapPatch`] rewrites at load.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
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
/// runtime capacity — selection and substitution only, no analysis.
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
    /// Substitute the runtime block-table capacity into the baked tape.
    /// The ONE permitted load-time `baked` site: commands
    /// whose constants carry a capacity patch are copied once per model
    /// load; everything else stays the macro-emitted static.
    pub fn materialize(&self, cap: u32) -> LoweredMetalTape {
        let mut tape = self.tape;
        if !self.const_patches.is_empty() {
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
            tape.commands = baked(commands);
        }
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
