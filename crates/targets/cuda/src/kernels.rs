// SPDX-License-Identifier: Apache-2.0
//! Direct kernel dispatch for `GpuTensor`.
//!
//! These wrap the same CUDA FFI functions from `vllm-kernels/csrc/` but
//! dispatch from `GpuTensor::as_ptr()` instead of extracting raw pointers
//! Here it's one line.

use core::ffi::{c_int, c_void};

use crate::alloc::{CachingAllocator, OwnedTensor};
use crate::dtype::DType;
use crate::tensor::GpuTensor;

// ---------------------------------------------------------------------------
// FFI declarations (same C symbols as vllm-kernels, linked from libscratchy_serving_cuda.a)
// ---------------------------------------------------------------------------

type CUstream = cudarc::driver::sys::CUstream;

unsafe extern "C" {
    // RMS norm. `weight_offset` is added to each weight element in
    // fp32 before the multiply — pass 0.0 for the standard
    // `y = x * w / rms(x)` (Llama, Qwen2), 1.0 for Gemma's
    // `y = x * (1 + w) / rms(x)`. See
    // `crates/scratchy-serving-cuda/csrc/layernorm_kernels.cu::rms_norm_kernel`.
    fn rms_norm_f16(
        out: *mut u16,
        input: *const u16,
        weight: *const u16,
        epsilon: f32,
        weight_offset: f32,
        num_tokens: i32,
        hidden_size: i32,
        stream: CUstream,
    );
    fn rms_norm_bf16(
        out: *mut u16,
        input: *const u16,
        weight: *const u16,
        epsilon: f32,
        weight_offset: f32,
        num_tokens: i32,
        hidden_size: i32,
        stream: CUstream,
    );
    fn rms_norm_f32(
        out: *mut f32,
        input: *const f32,
        weight: *const f32,
        epsilon: f32,
        weight_offset: f32,
        num_tokens: i32,
        hidden_size: i32,
        stream: CUstream,
    );

    // Fused add + RMS norm (in-place: residual += input, then norm).
    // Same `weight_offset` semantics as `rms_norm_*`.
    fn fused_add_rms_norm_f16(
        input: *mut u16,
        residual: *mut u16,
        weight: *const u16,
        epsilon: f32,
        weight_offset: f32,
        num_tokens: i32,
        hidden_size: i32,
        stream: CUstream,
    );
    fn fused_add_rms_norm_bf16(
        input: *mut u16,
        residual: *mut u16,
        weight: *const u16,
        epsilon: f32,
        weight_offset: f32,
        num_tokens: i32,
        hidden_size: i32,
        stream: CUstream,
    );
    fn fused_add_rms_norm_f32(
        input: *mut f32,
        residual: *mut f32,
        weight: *const f32,
        epsilon: f32,
        weight_offset: f32,
        num_tokens: i32,
        hidden_size: i32,
        stream: CUstream,
    );

    // Cohere LayerNorm (full LayerNorm with mean subtraction, weight only)
    fn cohere_layer_norm_f16(
        out: *mut u16,
        input: *const u16,
        weight: *const u16,
        epsilon: f32,
        num_tokens: i32,
        hidden_size: i32,
        stream: CUstream,
    );
    fn cohere_layer_norm_bf16(
        out: *mut u16,
        input: *const u16,
        weight: *const u16,
        epsilon: f32,
        num_tokens: i32,
        hidden_size: i32,
        stream: CUstream,
    );
    fn cohere_layer_norm_f32(
        out: *mut f32,
        input: *const f32,
        weight: *const f32,
        epsilon: f32,
        num_tokens: i32,
        hidden_size: i32,
        stream: CUstream,
    );

    // Fused add + Cohere LayerNorm (in-place)
    fn fused_add_cohere_layer_norm_f16(
        input: *mut u16,
        residual: *mut u16,
        weight: *const u16,
        epsilon: f32,
        num_tokens: i32,
        hidden_size: i32,
        stream: CUstream,
    );
    fn fused_add_cohere_layer_norm_bf16(
        input: *mut u16,
        residual: *mut u16,
        weight: *const u16,
        epsilon: f32,
        num_tokens: i32,
        hidden_size: i32,
        stream: CUstream,
    );
    fn fused_add_cohere_layer_norm_f32(
        input: *mut f32,
        residual: *mut f32,
        weight: *const f32,
        epsilon: f32,
        num_tokens: i32,
        hidden_size: i32,
        stream: CUstream,
    );

    // LayerNorm with bias (standard nn.LayerNorm)
    fn layer_norm_bias_f16(
        out: *mut u16,
        input: *const u16,
        weight: *const u16,
        bias: *const u16,
        epsilon: f32,
        num_tokens: i32,
        hidden_size: i32,
        stream: CUstream,
    );
    fn layer_norm_bias_bf16(
        out: *mut u16,
        input: *const u16,
        weight: *const u16,
        bias: *const u16,
        epsilon: f32,
        num_tokens: i32,
        hidden_size: i32,
        stream: CUstream,
    );
    fn layer_norm_bias_f32(
        out: *mut f32,
        input: *const f32,
        weight: *const f32,
        bias: *const f32,
        epsilon: f32,
        num_tokens: i32,
        hidden_size: i32,
        stream: CUstream,
    );

    // Fused add + LayerNorm with bias (in-place)
    fn fused_add_layer_norm_bias_f16(
        input: *mut u16,
        residual: *mut u16,
        weight: *const u16,
        bias: *const u16,
        epsilon: f32,
        num_tokens: i32,
        hidden_size: i32,
        stream: CUstream,
    );
    fn fused_add_layer_norm_bias_bf16(
        input: *mut u16,
        residual: *mut u16,
        weight: *const u16,
        bias: *const u16,
        epsilon: f32,
        num_tokens: i32,
        hidden_size: i32,
        stream: CUstream,
    );
    fn fused_add_layer_norm_bias_f32(
        input: *mut f32,
        residual: *mut f32,
        weight: *const f32,
        bias: *const f32,
        epsilon: f32,
        num_tokens: i32,
        hidden_size: i32,
        stream: CUstream,
    );

    // Fused QKV split + interleaved RoPE (Cohere convention: pairs at 2i, 2i+1)
    fn fused_qkv_interleaved_rope_f16(
        q: *mut u16,
        k: *mut u16,
        v: *mut u16,
        qkv: *const u16,
        positions: *const u32,
        cos_sin_cache: *const u16,
        q_size: i32,
        kv_size: i32,
        total_dim: i32,
        rotary_dim: i32,
        head_size: i32,
        num_tokens: i32,
        stream: CUstream,
    );
    fn fused_qkv_interleaved_rope_bf16(
        q: *mut u16,
        k: *mut u16,
        v: *mut u16,
        qkv: *const u16,
        positions: *const u32,
        cos_sin_cache: *const u16,
        q_size: i32,
        kv_size: i32,
        total_dim: i32,
        rotary_dim: i32,
        head_size: i32,
        num_tokens: i32,
        stream: CUstream,
    );
    fn fused_qkv_interleaved_rope_f32(
        q: *mut f32,
        k: *mut f32,
        v: *mut f32,
        qkv: *const f32,
        positions: *const u32,
        cos_sin_cache: *const f32,
        q_size: i32,
        kv_size: i32,
        total_dim: i32,
        rotary_dim: i32,
        head_size: i32,
        num_tokens: i32,
        stream: CUstream,
    );

    // Fused SiLU(gate) * up from combined [num_tokens, 2*d]
    fn silu_and_mul_fused_f16(
        out: *mut u16,
        gate_up: *const u16,
        num_tokens: i32,
        d: i32,
        stream: CUstream,
    );
    fn silu_and_mul_fused_bf16(
        out: *mut u16,
        gate_up: *const u16,
        num_tokens: i32,
        d: i32,
        stream: CUstream,
    );
    fn silu_and_mul_fused_f32(
        out: *mut f32,
        gate_up: *const f32,
        num_tokens: i32,
        d: i32,
        stream: CUstream,
    );

    // Fused GELU(tanh)(gate) * up from combined [num_tokens, 2*d]
    fn gelu_and_mul_fused_f16(
        out: *mut u16,
        gate_up: *const u16,
        num_tokens: i32,
        d: i32,
        stream: CUstream,
    );
    fn gelu_and_mul_fused_bf16(
        out: *mut u16,
        gate_up: *const u16,
        num_tokens: i32,
        d: i32,
        stream: CUstream,
    );
    fn gelu_and_mul_fused_f32(
        out: *mut f32,
        gate_up: *const f32,
        num_tokens: i32,
        d: i32,
        stream: CUstream,
    );

    // Non-fused GELU(tanh)(gate) * up from SEPARATE gate / up [num_tokens, d].
    fn gelu_and_mul_f16(
        out: *mut u16,
        gate: *const u16,
        up: *const u16,
        num_tokens: i32,
        d: i32,
        stream: CUstream,
    );
    fn gelu_and_mul_bf16(
        out: *mut u16,
        gate: *const u16,
        up: *const u16,
        num_tokens: i32,
        d: i32,
        stream: CUstream,
    );
    fn gelu_and_mul_f32(
        out: *mut f32,
        gate: *const f32,
        up: *const f32,
        num_tokens: i32,
        d: i32,
        stream: CUstream,
    );

    // QuickGELU (in-place pointwise): x[i] = x[i] * sigmoid(1.702 * x[i]).
    // Used by Qwen2-VL's vision MLP. NOT the same as gelu_tanh/gelu_erf;
    // the 1.702 multiplier is the OpenAI/CLIP approximation.
    fn quick_gelu_inplace_f16(x: *mut u16, n_elements: c_int, stream: CUstream);
    fn quick_gelu_inplace_bf16(x: *mut u16, n_elements: c_int, stream: CUstream);
    fn quick_gelu_inplace_f32(x: *mut f32, n_elements: c_int, stream: CUstream);

    // GELU (erf form, PyTorch nn.GELU() default): x * 0.5 * (1 + erf(x / sqrt(2))).
    // Used by Qwen2-VL's PatchMerger between its two Linear layers.
    fn gelu_erf_inplace_f16(x: *mut u16, n_elements: c_int, stream: CUstream);
    fn gelu_erf_inplace_bf16(x: *mut u16, n_elements: c_int, stream: CUstream);
    fn gelu_erf_inplace_f32(x: *mut f32, n_elements: c_int, stream: CUstream);

    // GELU (tanh approximation, PyTorch nn.GELU(approximate="tanh"),
    // HuggingFace `gelu_pytorch_tanh`):
    //   0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))).
    // Used by SigLIP / Gemma3-MM vision MLP. Distinct from
    // `gelu_erf_inplace` and `quick_gelu_inplace` — the three differ in
    // numerical form.
    fn gelu_tanh_inplace_f16(x: *mut u16, n_elements: c_int, stream: CUstream);
    fn gelu_tanh_inplace_bf16(x: *mut u16, n_elements: c_int, stream: CUstream);
    fn gelu_tanh_inplace_f32(x: *mut f32, n_elements: c_int, stream: CUstream);

    // 2-D non-overlapping average pool over a flat patch grid.
    // Input  [ph * ph, e] → output [(ph/k) * (ph/k), e]. Stride == kernel.
    // Used by Gemma3-MM's SigLIP→text projector (Phase G.7(b)).
    fn avg_pool_2d_f16(
        out: *mut u16,
        x: *const u16,
        ph: c_int,
        k: c_int,
        e: c_int,
        stream: CUstream,
    );
    fn avg_pool_2d_bf16(
        out: *mut u16,
        x: *const u16,
        ph: c_int,
        k: c_int,
        e: c_int,
        stream: CUstream,
    );
    fn avg_pool_2d_f32(
        out: *mut f32,
        x: *const f32,
        ph: c_int,
        k: c_int,
        e: c_int,
        stream: CUstream,
    );

    // Tanh softcap inplace: x[i] = cap * tanh(x[i] / cap)
    fn tanh_softcap_inplace_f16(x: *mut u16, cap: f32, n: c_int, stream: CUstream);
    fn tanh_softcap_inplace_bf16(x: *mut u16, cap: f32, n: c_int, stream: CUstream);
    fn tanh_softcap_inplace_f32(x: *mut f32, cap: f32, n: c_int, stream: CUstream);

    // Broadcast multiply inplace: x[row, col] *= scale[col]
    fn broadcast_mul_inplace_f16(
        x: *mut u16,
        scale: *const u16,
        num_rows: i32,
        d: i32,
        stream: CUstream,
    );
    fn broadcast_mul_inplace_bf16(
        x: *mut u16,
        scale: *const u16,
        num_rows: i32,
        d: i32,
        stream: CUstream,
    );
    fn broadcast_mul_inplace_f32(
        x: *mut f32,
        scale: *const f32,
        num_rows: i32,
        d: i32,
        stream: CUstream,
    );

    // Rotary embedding (in-place on q and k)
    fn rotary_embedding_f16(
        positions: *const u32,
        query: *mut u16,
        key: *mut u16,
        cos_sin_cache: *const u16,
        rotary_dim: i32,
        total_q_dim: i32,
        total_k_dim: i32,
        head_size: i32,
        num_tokens: i32,
        stream: CUstream,
    );
    fn rotary_embedding_bf16(
        positions: *const u32,
        query: *mut u16,
        key: *mut u16,
        cos_sin_cache: *const u16,
        rotary_dim: i32,
        total_q_dim: i32,
        total_k_dim: i32,
        head_size: i32,
        num_tokens: i32,
        stream: CUstream,
    );
    fn rotary_embedding_f32(
        positions: *const u32,
        query: *mut f32,
        key: *mut f32,
        cos_sin_cache: *const f32,
        rotary_dim: i32,
        total_q_dim: i32,
        total_k_dim: i32,
        head_size: i32,
        num_tokens: i32,
        stream: CUstream,
    );

    // Interleaved rotary embedding (in-place on q and k, DeepSeek MLA style)
    fn rotary_embedding_interleaved_f16(
        positions: *const u32,
        query: *mut u16,
        key: *mut u16,
        cos_sin_cache: *const u16,
        rotary_dim: i32,
        total_q_dim: i32,
        total_k_dim: i32,
        head_size: i32,
        num_tokens: i32,
        stream: CUstream,
    );
    fn rotary_embedding_interleaved_bf16(
        positions: *const u32,
        query: *mut u16,
        key: *mut u16,
        cos_sin_cache: *const u16,
        rotary_dim: i32,
        total_q_dim: i32,
        total_k_dim: i32,
        head_size: i32,
        num_tokens: i32,
        stream: CUstream,
    );
    fn rotary_embedding_interleaved_f32(
        positions: *const u32,
        query: *mut f32,
        key: *mut f32,
        cos_sin_cache: *const f32,
        rotary_dim: i32,
        total_q_dim: i32,
        total_k_dim: i32,
        head_size: i32,
        num_tokens: i32,
        stream: CUstream,
    );

    // Paged KV cache RoPE (for spans: rotate unrotated K in paged cache)
    fn rotary_paged_k_cache_f16(
        k_cache: *mut u16,
        cos_sin_cache: *const u16,
        block_table: *const i32,
        seqused_k: *const i32,
        block_flags: *const u8, // per-physical-block flag, or null for all blocks
        batch_size: i32,
        max_seqlen_k: i32,
        max_blocks_per_seq: i32,
        page_block_size: i32,
        num_kv_heads: i32,
        head_dim: i32,
        rotary_dim: i32,
        inverse: i32,
        stream: CUstream,
    );
    fn rotary_paged_k_cache_bf16(
        k_cache: *mut u16,
        cos_sin_cache: *const u16,
        block_table: *const i32,
        seqused_k: *const i32,
        block_flags: *const u8,
        batch_size: i32,
        max_seqlen_k: i32,
        max_blocks_per_seq: i32,
        page_block_size: i32,
        num_kv_heads: i32,
        head_dim: i32,
        rotary_dim: i32,
        inverse: i32,
        stream: CUstream,
    );

    // Embedding gather with vocab-shard masking. At tp=1 callers pass
    // (vocab_offset=0, vocab_per_rank=weight.dim(0)) and the kernel
    // degrades to a plain gather. At tp>1 each rank's `weight` is the
    // [vocab_per_rank, hidden] slice; tokens outside [vocab_offset,
    // vocab_offset+vocab_per_rank) get zero-filled output rows so the
    // post-Embed AllReduce-sum produces exactly one contribution per
    // token from the rank that owns its embedding. Mirrors Python
    // VocabParallelEmbedding's masked-input + masked_fill_ pattern.
    fn embedding_gather_f16(
        out: *mut u16,
        weight: *const u16,
        ids: *const u32,
        hidden_size: i32,
        num_tokens: i32,
        vocab_offset: u32,
        vocab_per_rank: u32,
        stream: CUstream,
    );
    fn embedding_gather_bf16(
        out: *mut u16,
        weight: *const u16,
        ids: *const u32,
        hidden_size: i32,
        num_tokens: i32,
        vocab_offset: u32,
        vocab_per_rank: u32,
        stream: CUstream,
    );
    fn embedding_gather_f32(
        out: *mut f32,
        weight: *const f32,
        ids: *const u32,
        hidden_size: i32,
        num_tokens: i32,
        vocab_offset: u32,
        vocab_per_rank: u32,
        stream: CUstream,
    );

    // Update decode metadata in-place on GPU
    fn update_decode_metadata(
        positions: *mut u32,
        slot_mapping: *mut i64,
        seqused_k: *mut i32,
        block_table: *const i32,
        num_reqs: c_int,
        block_size: c_int,
        max_blocks_per_seq: c_int,
        stream: CUstream,
    );

    // Prefix sum of seqused_k → cu_seqlens_k on GPU (single thread, batch ≤512)
    fn prefix_sum_seqused_k_gpu(
        seqused_k: *const i32,
        cu_seqlens_k: *mut i32,
        num_reqs: c_int,
        stream: CUstream,
    );

    // Split fused QKV
    fn split_qkv_f16(
        q: *mut u16,
        k: *mut u16,
        v: *mut u16,
        qkv: *const u16,
        q_size: i32,
        kv_size: i32,
        total_dim: i32,
        num_tokens: i32,
        stream: CUstream,
    );
    fn split_qkv_bf16(
        q: *mut u16,
        k: *mut u16,
        v: *mut u16,
        qkv: *const u16,
        q_size: i32,
        kv_size: i32,
        total_dim: i32,
        num_tokens: i32,
        stream: CUstream,
    );
    fn split_qkv_f32(
        q: *mut f32,
        k: *mut f32,
        v: *mut f32,
        qkv: *const f32,
        q_size: i32,
        kv_size: i32,
        total_dim: i32,
        num_tokens: i32,
        stream: CUstream,
    );

    // Bias add: out[i,j] += bias[j]
    fn bias_add_f16(out: *mut c_void, bias: *const c_void, m: c_int, n: c_int, stream: CUstream);
    fn bias_add_bf16(out: *mut c_void, bias: *const c_void, m: c_int, n: c_int, stream: CUstream);
    fn bias_add_f32(out: *mut c_void, bias: *const c_void, m: c_int, n: c_int, stream: CUstream);

    // Fused QKV split + RoPE (replaces split_qkv + rotary_embedding).
    // MRoPE (Qwen2-VL): positions_stride0 = n_tokens for [3, n_tokens]
    // positions; mrope_sec0/sec01 are cumulative section boundaries.
    // Legacy 1D path: positions_stride0 = 0 (mrope_sec0/sec01 ignored).
    fn fused_qkv_rope_f16(
        q: *mut u16,
        k: *mut u16,
        v: *mut u16,
        qkv: *const u16,
        positions: *const u32,
        cos_sin_cache: *const u16,
        q_size: i32,
        kv_size: i32,
        total_dim: i32,
        rotary_dim: i32,
        head_size: i32,
        num_tokens: i32,
        positions_stride0: i32,
        mrope_sec0: i32,
        mrope_sec01: i32,
        stream: CUstream,
    );
    fn fused_qkv_rope_bf16(
        q: *mut u16,
        k: *mut u16,
        v: *mut u16,
        qkv: *const u16,
        positions: *const u32,
        cos_sin_cache: *const u16,
        q_size: i32,
        kv_size: i32,
        total_dim: i32,
        rotary_dim: i32,
        head_size: i32,
        num_tokens: i32,
        positions_stride0: i32,
        mrope_sec0: i32,
        mrope_sec01: i32,
        stream: CUstream,
    );
    fn fused_qkv_rope_f32(
        q: *mut f32,
        k: *mut f32,
        v: *mut f32,
        qkv: *const f32,
        positions: *const u32,
        cos_sin_cache: *const f32,
        q_size: i32,
        kv_size: i32,
        total_dim: i32,
        rotary_dim: i32,
        head_size: i32,
        num_tokens: i32,
        positions_stride0: i32,
        mrope_sec0: i32,
        mrope_sec01: i32,
        stream: CUstream,
    );

    // Fused QKV split + RoPE + reshape_and_cache (decode path).
    // MRoPE params have the same semantics as fused_qkv_rope_*.
    fn fused_qkv_rope_cache_f16(
        q: *mut u16,
        key_cache: *mut u16,
        value_cache: *mut u16,
        qkv: *const u16,
        positions: *const u32,
        cos_sin_cache: *const u16,
        slot_mapping: *const i64,
        q_size: i32,
        kv_size: i32,
        total_dim: i32,
        rotary_dim: i32,
        head_size: i32,
        num_tokens: i32,
        positions_stride0: i32,
        mrope_sec0: i32,
        mrope_sec01: i32,
        stream: CUstream,
    );
    fn fused_qkv_rope_cache_bf16(
        q: *mut u16,
        key_cache: *mut u16,
        value_cache: *mut u16,
        qkv: *const u16,
        positions: *const u32,
        cos_sin_cache: *const u16,
        slot_mapping: *const i64,
        q_size: i32,
        kv_size: i32,
        total_dim: i32,
        rotary_dim: i32,
        head_size: i32,
        num_tokens: i32,
        positions_stride0: i32,
        mrope_sec0: i32,
        mrope_sec01: i32,
        stream: CUstream,
    );
    fn fused_qkv_rope_cache_f32(
        q: *mut f32,
        key_cache: *mut f32,
        value_cache: *mut f32,
        qkv: *const f32,
        positions: *const u32,
        cos_sin_cache: *const f32,
        slot_mapping: *const i64,
        q_size: i32,
        kv_size: i32,
        total_dim: i32,
        rotary_dim: i32,
        head_size: i32,
        num_tokens: i32,
        positions_stride0: i32,
        mrope_sec0: i32,
        mrope_sec01: i32,
        stream: CUstream,
    );

    // Vision 2D RoPE apply (in-place). Used by Qwen2-VL ViT — applies
    // neox-style rotary to a `[seq_len, num_heads, head_size]` buffer
    // using precomputed `cos`/`sin` of shape `[seq_len, head_size/2]`.
    // Caller invokes once for Q and once for K (no packed QKV here).
    fn vision_rope_apply_f16(
        x: *mut u16,
        cos: *const u16,
        sin: *const u16,
        seq_len: c_int,
        num_heads: c_int,
        head_size: c_int,
        stream: CUstream,
    );
    fn vision_rope_apply_bf16(
        x: *mut u16,
        cos: *const u16,
        sin: *const u16,
        seq_len: c_int,
        num_heads: c_int,
        head_size: c_int,
        stream: CUstream,
    );
    fn vision_rope_apply_f32(
        x: *mut f32,
        cos: *const f32,
        sin: *const f32,
        seq_len: c_int,
        num_heads: c_int,
        head_size: c_int,
        stream: CUstream,
    );

    // Fused interleaved QKV split + RoPE + reshape_and_cache (decode, Cohere/CommandR)
    fn fused_qkv_interleaved_rope_cache_f16(
        q: *mut u16,
        key_cache: *mut u16,
        value_cache: *mut u16,
        qkv: *const u16,
        positions: *const u32,
        cos_sin_cache: *const u16,
        slot_mapping: *const i64,
        q_size: i32,
        kv_size: i32,
        total_dim: i32,
        rotary_dim: i32,
        head_size: i32,
        num_tokens: i32,
        stream: CUstream,
    );
    fn fused_qkv_interleaved_rope_cache_bf16(
        q: *mut u16,
        key_cache: *mut u16,
        value_cache: *mut u16,
        qkv: *const u16,
        positions: *const u32,
        cos_sin_cache: *const u16,
        slot_mapping: *const i64,
        q_size: i32,
        kv_size: i32,
        total_dim: i32,
        rotary_dim: i32,
        head_size: i32,
        num_tokens: i32,
        stream: CUstream,
    );
    fn fused_qkv_interleaved_rope_cache_f32(
        q: *mut f32,
        key_cache: *mut f32,
        value_cache: *mut f32,
        qkv: *const f32,
        positions: *const u32,
        cos_sin_cache: *const f32,
        slot_mapping: *const i64,
        q_size: i32,
        kv_size: i32,
        total_dim: i32,
        rotary_dim: i32,
        head_size: i32,
        num_tokens: i32,
        stream: CUstream,
    );

    // Fused QKV split + RoPE + FP8 quantize + cache write (BF16→FP8)
    fn fused_qkv_rope_cache_fp8_bf16(
        q: *mut u16,
        key_cache: *mut u8,
        value_cache: *mut u8,
        qkv: *const u16,
        positions: *const u32,
        cos_sin_cache: *const u16,
        slot_mapping: *const i64,
        k_scale: *const f32,
        v_scale: *const f32,
        q_size: i32,
        kv_size: i32,
        total_dim: i32,
        rotary_dim: i32,
        head_size: i32,
        num_tokens: i32,
        stream: CUstream,
    );
    fn fused_qkv_interleaved_rope_cache_fp8_bf16(
        q: *mut u16,
        key_cache: *mut u8,
        value_cache: *mut u8,
        qkv: *const u16,
        positions: *const u32,
        cos_sin_cache: *const u16,
        slot_mapping: *const i64,
        k_scale: *const f32,
        v_scale: *const f32,
        q_size: i32,
        kv_size: i32,
        total_dim: i32,
        rotary_dim: i32,
        head_size: i32,
        num_tokens: i32,
        stream: CUstream,
    );

    // MoE top-k softmax
    fn topk_softmax_f32(
        topk_weights: *mut f32,
        topk_ids: *mut i32,
        workspace: *mut f32,
        gating_output: *const f32,
        num_tokens: c_int,
        num_experts: c_int,
        topk: c_int,
        renormalize: c_int,
        stream: CUstream,
    );
    fn topk_softmax_bf16(
        topk_weights: *mut f32,
        topk_ids: *mut i32,
        workspace: *mut f32,
        gating_output: *const c_void,
        num_tokens: c_int,
        num_experts: c_int,
        topk: c_int,
        renormalize: c_int,
        stream: CUstream,
    );
    fn topk_softmax_f16(
        topk_weights: *mut f32,
        topk_ids: *mut i32,
        workspace: *mut f32,
        gating_output: *const c_void,
        num_tokens: c_int,
        num_experts: c_int,
        topk: c_int,
        renormalize: c_int,
        stream: CUstream,
    );

    // MoE sigmoid top-k with e_score_correction_bias (DeepSeek V3 / Kimi K2)
    // Same workspace convention as topk_softmax_*; bias may be null.
    fn topk_sigmoid_f32(
        topk_weights: *mut f32,
        topk_ids: *mut i32,
        workspace: *mut f32,
        bias: *const f32,
        gating_output: *const f32,
        num_tokens: c_int,
        num_experts: c_int,
        topk: c_int,
        renormalize: c_int,
        stream: CUstream,
    );
    fn topk_sigmoid_bf16(
        topk_weights: *mut f32,
        topk_ids: *mut i32,
        workspace: *mut f32,
        bias: *const f32,
        gating_output: *const c_void,
        num_tokens: c_int,
        num_experts: c_int,
        topk: c_int,
        renormalize: c_int,
        stream: CUstream,
    );
    fn topk_sigmoid_f16(
        topk_weights: *mut f32,
        topk_ids: *mut i32,
        workspace: *mut f32,
        bias: *const f32,
        gating_output: *const c_void,
        num_tokens: c_int,
        num_experts: c_int,
        topk: c_int,
        renormalize: c_int,
        stream: CUstream,
    );

    // Grouped noaux_tc top-k (DeepSeek V3 / Kimi K2): sigmoid + group selection
    fn topk_noaux_tc_f32_f32(
        scores: *mut c_void,
        topk_values: *mut f32,
        topk_indices: *mut i32,
        bias: *const c_void,
        num_tokens: i64,
        num_experts: i64,
        n_group: i64,
        topk_group: i64,
        topk: i64,
        renormalize: c_int,
        routed_scaling_factor: f64,
        stream: CUstream,
    );
    fn topk_noaux_tc_bf16_f32(
        scores: *mut c_void,
        topk_values: *mut f32,
        topk_indices: *mut i32,
        bias: *const c_void,
        num_tokens: i64,
        num_experts: i64,
        n_group: i64,
        topk_group: i64,
        topk: i64,
        renormalize: c_int,
        routed_scaling_factor: f64,
        stream: CUstream,
    );
    fn topk_noaux_tc_f16_f32(
        scores: *mut c_void,
        topk_values: *mut f32,
        topk_indices: *mut i32,
        bias: *const c_void,
        num_tokens: i64,
        num_experts: i64,
        n_group: i64,
        topk_group: i64,
        topk: i64,
        renormalize: c_int,
        routed_scaling_factor: f64,
        stream: CUstream,
    );

    // MoE sum reduction
    fn moe_sum_f32(
        out: *mut f32,
        input: *const f32,
        num_tokens: c_int,
        hidden_size: c_int,
        topk: c_int,
        stream: CUstream,
    );
    fn moe_sum_f16(
        out: *mut c_void,
        input: *const c_void,
        num_tokens: c_int,
        hidden_size: c_int,
        topk: c_int,
        stream: CUstream,
    );
    fn moe_sum_bf16(
        out: *mut c_void,
        input: *const c_void,
        num_tokens: c_int,
        hidden_size: c_int,
        topk: c_int,
        stream: CUstream,
    );

    // MoE align block size
    fn moe_align_block_size_i32(
        topk_ids: *const i32,
        sorted_token_ids: *mut i32,
        expert_ids: *mut i32,
        total_tokens_post_pad: *mut i32,
        num_experts: c_int,
        block_size: c_int,
        numel: c_int,
        max_num_tokens_padded: c_int,
        stream: CUstream,
    );

    // Fused MoE GEMM
    fn fused_moe_gemm_bf16(
        output: *mut c_void,
        input: *const c_void,
        weights: *const c_void,
        topk_weights: *const f32,
        sorted_token_ids: *const i32,
        expert_ids: *const i32,
        num_tokens_post_padded: *const i32,
        num_valid_tokens: c_int,
        in_features: c_int,
        out_features: c_int,
        top_k: c_int,
        num_tokens_padded_total: c_int,
        apply_weights: c_int,
        block_m: c_int,
        stream: CUstream,
    );
    fn fused_moe_gemm_f16(
        output: *mut c_void,
        input: *const c_void,
        weights: *const c_void,
        topk_weights: *const f32,
        sorted_token_ids: *const i32,
        expert_ids: *const i32,
        num_tokens_post_padded: *const i32,
        num_valid_tokens: c_int,
        in_features: c_int,
        out_features: c_int,
        top_k: c_int,
        num_tokens_padded_total: c_int,
        apply_weights: c_int,
        block_m: c_int,
        stream: CUstream,
    );

    // CUTLASS grouped MoE GEMM glue + arch launchers (cutlass_moe_grouped.cu).
    // Permute (gather A), arg-marshal (per-group ptr/problem/stride arrays),
    // scatter (token-indexed write), and the Sm80 / Sm90 grouped GEMM cores.
    fn scratchy_moe_permute_gather_a(
        a_permuted: *mut c_void,
        input: *const c_void,
        sorted_token_ids: *const i32,
        num_tokens_post_padded: *const i32,
        total_padded_rows_cap: c_int,
        num_valid_tokens: c_int,
        top_k: c_int,
        k: c_int,
        stream: CUstream,
    );
    fn scratchy_moe_marshal_groups(
        problem_sizes: *mut i32,
        ptr_a: *mut *mut c_void,
        ptr_b: *mut *mut c_void,
        ptr_c: *mut *mut c_void,
        lda: *mut i64,
        ldb: *mut i64,
        ldc: *mut i64,
        expert_ids: *const i32,
        num_tokens_post_padded: *const i32,
        a_permuted: *mut c_void,
        weights: *const c_void,
        c_permuted: *mut c_void,
        num_experts: c_int,
        block_m: c_int,
        n: c_int,
        k: c_int,
        stream: CUstream,
    );
    fn scratchy_moe_scatter_c(
        output: *mut c_void,
        c_permuted: *const c_void,
        topk_weights: *const f32,
        sorted_token_ids: *const i32,
        num_tokens_post_padded: *const i32,
        total_padded_rows_cap: c_int,
        num_valid_tokens: c_int,
        apply_weights: c_int,
        n: c_int,
        stream: CUstream,
    );
    fn cutlass_moe_grouped_bf16_sm80(
        problem_sizes: *const c_void,
        ptr_a: *mut *mut c_void,
        ptr_b: *mut *mut c_void,
        ptr_c: *mut *mut c_void,
        lda: *mut i64,
        ldb: *mut i64,
        ldc: *mut i64,
        num_experts: c_int,
        threadblock_count: c_int,
        workspace: *mut c_void,
        stream: CUstream,
    ) -> c_int;
    fn cutlass_moe_grouped_bf16_sm80_threadblock_count() -> c_int;
    fn cutlass_moe_grouped_bf16_sm90(
        problem_sizes: *const c_void,
        ptr_a: *mut *mut c_void,
        ptr_b: *mut *mut c_void,
        ptr_c: *mut *mut c_void,
        stride_a: *mut c_void,
        stride_b: *mut c_void,
        stride_c: *mut c_void,
        stride_d: *mut c_void,
        num_experts: c_int,
        sm_count: c_int,
        workspace: *mut c_void,
        stream: CUstream,
    ) -> c_int;
    fn cutlass_moe_grouped_sm90_fill_strides(
        stride_a: *mut c_void,
        stride_b: *mut c_void,
        stride_c: *mut c_void,
        stride_d: *mut c_void,
        problem_sizes: *const i32,
        num_experts: c_int,
        stream: CUstream,
    );
    fn cutlass_moe_grouped_sm90_stride_bytes() -> c_int;

    // MoE expert gather kernels (graph-capture-safe unfused path)
    fn moe_expert_gather_bf16(
        output: *mut c_void,
        all_expert_out: *const c_void,
        topk_ids: *const i32,
        num_tokens: c_int,
        top_k: c_int,
        out_features: c_int,
        stream: CUstream,
    );
    fn moe_expert_gather_weighted_sum_bf16(
        output: *mut c_void,
        all_expert_out: *const c_void,
        topk_ids: *const i32,
        topk_weights: *const f32,
        num_tokens: c_int,
        top_k: c_int,
        out_features: c_int,
        rows_per_expert: c_int,
        stream: CUstream,
    );

    // Fused MoE GEMM — FP8 E4M3 (SM89+ true FP8 tensor cores)
    // TODO: fix PTX fragment layout bug (currently produces 0.5x output)
    #[allow(dead_code)]
    fn fused_moe_fp8_gemm_sm89(
        output: *mut c_void,
        input: *const c_void,
        weights: *const c_void,
        a_scales: *const f32,
        w_scales: *const f32,
        topk_weights: *const f32,
        sorted_token_ids: *const i32,
        expert_ids: *const i32,
        num_tokens_post_padded: *const i32,
        num_valid_tokens: c_int,
        in_features: c_int,
        out_features: c_int,
        top_k: c_int,
        apply_weights: c_int,
        stream: CUstream,
    );

    // Fused MoE GEMM — FP8 E4M3 (SM80+ dequant fallback)
    fn fused_moe_fp8_gemm_dequant(
        output: *mut c_void,
        input: *const c_void,
        weights: *const c_void,
        a_scales: *const f32,
        w_scales: *const f32,
        topk_weights: *const f32,
        sorted_token_ids: *const i32,
        expert_ids: *const i32,
        num_tokens_post_padded: *const i32,
        num_valid_tokens: c_int,
        in_features: c_int,
        out_features: c_int,
        top_k: c_int,
        apply_weights: c_int,
        block_m: c_int,
        stream: CUstream,
    );

    // Fused MoE GEMM — FP8 block-quantized (per-block weight scales applied during dequant)
    fn fused_moe_fp8_block_gemm_dequant(
        output: *mut c_void,
        input: *const c_void,
        weights: *const c_void,
        a_scales: *const f32,
        w_scale_inv: *const f32,
        topk_weights: *const f32,
        sorted_token_ids: *const i32,
        expert_ids: *const i32,
        num_tokens_post_padded: *const i32,
        num_valid_tokens: c_int,
        in_features: c_int,
        out_features: c_int,
        top_k: c_int,
        apply_weights: c_int,
        block_m: c_int,
        quant_block_n: c_int,
        quant_block_k: c_int,
        scale_stride_e: c_int,
        scale_stride_n: c_int,
        stream: CUstream,
    );

    // Fused sigmoid_mul_add: out = a + sigmoid(gate) * b
    fn sigmoid_mul_add_bf16(
        out: *mut c_void,
        a: *const c_void,
        b: *const c_void,
        gate: *const c_void,
        num_tokens: c_int,
        hidden_size: c_int,
        stream: CUstream,
    );
    fn sigmoid_mul_add_f16(
        out: *mut c_void,
        a: *const c_void,
        b: *const c_void,
        gate: *const c_void,
        num_tokens: c_int,
        hidden_size: c_int,
        stream: CUstream,
    );
    fn sigmoid_mul_add_f32(
        out: *mut c_void,
        a: *const c_void,
        b: *const c_void,
        gate: *const c_void,
        num_tokens: c_int,
        hidden_size: c_int,
        stream: CUstream,
    );

    // In-place add: a += b
    fn add_inplace_bf16(
        a: *mut c_void,
        b: *const c_void,
        num_tokens: c_int,
        hidden_size: c_int,
        stream: CUstream,
    );
    fn add_inplace_f16(
        a: *mut c_void,
        b: *const c_void,
        num_tokens: c_int,
        hidden_size: c_int,
        stream: CUstream,
    );
    fn add_inplace_f32(
        a: *mut c_void,
        b: *const c_void,
        num_tokens: c_int,
        hidden_size: c_int,
        stream: CUstream,
    );

    fn qk_norm_rope_f32(
        query: *mut f32,
        key: *mut f32,
        q_weight: *const f32,
        k_weight: *const f32,
        cos_cache: *const f32,
        sin_cache: *const f32,
        positions: *const u32,
        epsilon: f32,
        q_weight_offset: f32,
        k_weight_offset: f32,
        num_q_heads: c_int,
        num_kv_heads: c_int,
        head_dim: c_int,
        num_tokens: c_int,
        stream: CUstream,
    );
    fn qk_norm_rope_f16(
        query: *mut u16,
        key: *mut u16,
        q_weight: *const u16,
        k_weight: *const u16,
        cos_cache: *const u16,
        sin_cache: *const u16,
        positions: *const u32,
        epsilon: f32,
        q_weight_offset: f32,
        k_weight_offset: f32,
        num_q_heads: c_int,
        num_kv_heads: c_int,
        head_dim: c_int,
        num_tokens: c_int,
        stream: CUstream,
    );
    fn qk_norm_rope_bf16(
        query: *mut u16,
        key: *mut u16,
        q_weight: *const u16,
        k_weight: *const u16,
        cos_cache: *const u16,
        sin_cache: *const u16,
        positions: *const u32,
        epsilon: f32,
        q_weight_offset: f32,
        k_weight_offset: f32,
        num_q_heads: c_int,
        num_kv_heads: c_int,
        head_dim: c_int,
        num_tokens: c_int,
        stream: CUstream,
    );

    // MLA data movement kernels (DeepSeek V2/V3)
    fn mla_split_kv_a_f16(
        src: *const c_void,
        dst_latent: *mut c_void,
        dst_k_pe: *mut c_void,
        num_tokens: c_int,
        kv_lora_rank: c_int,
        rope_dim: c_int,
        stream: CUstream,
    );
    fn mla_split_kv_a_bf16(
        src: *const c_void,
        dst_latent: *mut c_void,
        dst_k_pe: *mut c_void,
        num_tokens: c_int,
        kv_lora_rank: c_int,
        rope_dim: c_int,
        stream: CUstream,
    );
    fn mla_split_kv_a_f32(
        src: *const c_void,
        dst_latent: *mut c_void,
        dst_k_pe: *mut c_void,
        num_tokens: c_int,
        kv_lora_rank: c_int,
        rope_dim: c_int,
        stream: CUstream,
    );

    fn mla_extract_q_pe_f16(
        src: *const c_void,
        dst: *mut c_void,
        num_tokens: c_int,
        num_heads: c_int,
        qk_head_dim: c_int,
        qk_nope_head_dim: c_int,
        rope_dim: c_int,
        stream: CUstream,
    );
    fn mla_extract_q_pe_bf16(
        src: *const c_void,
        dst: *mut c_void,
        num_tokens: c_int,
        num_heads: c_int,
        qk_head_dim: c_int,
        qk_nope_head_dim: c_int,
        rope_dim: c_int,
        stream: CUstream,
    );
    fn mla_extract_q_pe_f32(
        src: *const c_void,
        dst: *mut c_void,
        num_tokens: c_int,
        num_heads: c_int,
        qk_head_dim: c_int,
        qk_nope_head_dim: c_int,
        rope_dim: c_int,
        stream: CUstream,
    );

    fn mla_write_q_pe_f16(
        src: *const c_void,
        dst: *mut c_void,
        num_tokens: c_int,
        num_heads: c_int,
        qk_head_dim: c_int,
        qk_nope_head_dim: c_int,
        rope_dim: c_int,
        stream: CUstream,
    );
    fn mla_write_q_pe_bf16(
        src: *const c_void,
        dst: *mut c_void,
        num_tokens: c_int,
        num_heads: c_int,
        qk_head_dim: c_int,
        qk_nope_head_dim: c_int,
        rope_dim: c_int,
        stream: CUstream,
    );
    fn mla_write_q_pe_f32(
        src: *const c_void,
        dst: *mut c_void,
        num_tokens: c_int,
        num_heads: c_int,
        qk_head_dim: c_int,
        qk_nope_head_dim: c_int,
        rope_dim: c_int,
        stream: CUstream,
    );

    fn mla_assemble_k_f16(
        kv_b: *const c_void,
        k_pe: *const c_void,
        dst: *mut c_void,
        num_tokens: c_int,
        num_heads: c_int,
        qk_nope_head_dim: c_int,
        qk_rope_head_dim: c_int,
        v_head_dim: c_int,
        qk_head_dim: c_int,
        stream: CUstream,
    );
    fn mla_assemble_k_bf16(
        kv_b: *const c_void,
        k_pe: *const c_void,
        dst: *mut c_void,
        num_tokens: c_int,
        num_heads: c_int,
        qk_nope_head_dim: c_int,
        qk_rope_head_dim: c_int,
        v_head_dim: c_int,
        qk_head_dim: c_int,
        stream: CUstream,
    );
    fn mla_assemble_k_f32(
        kv_b: *const c_void,
        k_pe: *const c_void,
        dst: *mut c_void,
        num_tokens: c_int,
        num_heads: c_int,
        qk_nope_head_dim: c_int,
        qk_rope_head_dim: c_int,
        v_head_dim: c_int,
        qk_head_dim: c_int,
        stream: CUstream,
    );

    fn mla_assemble_v_f16(
        kv_b: *const c_void,
        dst: *mut c_void,
        num_tokens: c_int,
        num_heads: c_int,
        qk_nope_head_dim: c_int,
        v_head_dim: c_int,
        qk_head_dim: c_int,
        stream: CUstream,
    );
    fn mla_assemble_v_bf16(
        kv_b: *const c_void,
        dst: *mut c_void,
        num_tokens: c_int,
        num_heads: c_int,
        qk_nope_head_dim: c_int,
        v_head_dim: c_int,
        qk_head_dim: c_int,
        stream: CUstream,
    );
    fn mla_assemble_v_f32(
        kv_b: *const c_void,
        dst: *mut c_void,
        num_tokens: c_int,
        num_heads: c_int,
        qk_nope_head_dim: c_int,
        v_head_dim: c_int,
        qk_head_dim: c_int,
        stream: CUstream,
    );

    fn mla_slice_attn_output_f16(
        src: *const c_void,
        dst: *mut c_void,
        num_tokens: c_int,
        num_heads: c_int,
        qk_head_dim: c_int,
        v_head_dim: c_int,
        stream: CUstream,
    );
    fn mla_slice_attn_output_bf16(
        src: *const c_void,
        dst: *mut c_void,
        num_tokens: c_int,
        num_heads: c_int,
        qk_head_dim: c_int,
        v_head_dim: c_int,
        stream: CUstream,
    );
    fn mla_slice_attn_output_f32(
        src: *const c_void,
        dst: *mut c_void,
        num_tokens: c_int,
        num_heads: c_int,
        qk_head_dim: c_int,
        v_head_dim: c_int,
        stream: CUstream,
    );
}

// ---------------------------------------------------------------------------
// RMS Norm
// ---------------------------------------------------------------------------

/// RMS normalization: `out = input / rms(input) * weight`.
///
/// * `input`: `[num_tokens, hidden_size]`
/// * `weight`: `[hidden_size]`
/// * Returns: `[num_tokens, hidden_size]` allocated from arena.
///
/// For Gemma-style `y = x * (1 + w) / rms(x)` use the
/// [`rms_norm_with_offset`] variant that passes the offset to the
/// kernel.
pub unsafe fn rms_norm(
    input: GpuTensor,
    weight: GpuTensor,
    eps: f32,
    alloc: &mut CachingAllocator,
    stream: CUstream,
) -> OwnedTensor {
    rms_norm_with_offset(input, weight, eps, 0.0, alloc, stream)
}

/// RMS normalization with an `(weight + weight_offset)` fold,
/// applied in fp32 inside the kernel with cast-last semantics.
///
/// * `weight_offset`: added to each weight element before the
///   multiply. `0.0` ⇒ standard `rms_norm`; `1.0` ⇒ Gemma-style
///   `y = x * (1 + w) / rms(x)`.
pub unsafe fn rms_norm_with_offset(
    input: GpuTensor,
    weight: GpuTensor,
    eps: f32,
    weight_offset: f32,
    alloc: &mut CachingAllocator,
    stream: CUstream,
) -> OwnedTensor {
    let num_tokens = input.dim(0) as i32;
    let hidden_size = input.dim(1) as i32;
    // GUARD: the kernel reads `weight` with the SAME element type it dispatches on
    // (`input.dtype()`). A mismatched gain dtype (e.g. an FP16 RMSNorm gain fed to
    // the BF16 path) is reinterpreted bit-for-bit and silently corrupts the result
    // with no crash — exactly the Gemma-4 router degeneracy (FP16 gain read as BF16
    // ≈ 7e9 collapsed the top-k softmax). Fail loudly instead.
    assert_eq!(
        weight.dtype(),
        input.dtype(),
        "rms_norm: gain dtype {:?} != input dtype {:?}; the kernel reads the gain as \
         input.dtype() and would misread a mismatched gain.",
        weight.dtype(),
        input.dtype()
    );
    let out = alloc.alloc_tensor(&[num_tokens as usize, hidden_size as usize], input.dtype());

    match input.dtype() {
        DType::F16 => rms_norm_f16(
            out.as_mut_ptr(),
            input.as_ptr(),
            weight.as_ptr(),
            eps,
            weight_offset,
            num_tokens,
            hidden_size,
            stream,
        ),
        DType::BF16 => rms_norm_bf16(
            out.as_mut_ptr(),
            input.as_ptr(),
            weight.as_ptr(),
            eps,
            weight_offset,
            num_tokens,
            hidden_size,
            stream,
        ),
        DType::F32 => rms_norm_f32(
            out.as_mut_ptr(),
            input.as_ptr(),
            weight.as_ptr(),
            eps,
            weight_offset,
            num_tokens,
            hidden_size,
            stream,
        ),
        _ => panic!("rms_norm: unsupported dtype {:?}", input.dtype()),
    }
    out
}

// ---------------------------------------------------------------------------
// Fused Add + RMS Norm
// ---------------------------------------------------------------------------

/// Fused add + RMS norm: `residual += input; normed = rms_norm(residual) * weight`
///
/// The CUDA kernel writes normed output into `input` (in-place) and updates
/// `residual` in-place. We allocate a fresh output from the arena and copy
/// the input there first so the caller's input isn't corrupted.
///
/// Actually — the vllm-kernels fused_add_rms_norm writes:
///   residual[i] += input[i]       (in-place on residual)
///   input[i] = norm(residual[i])  (in-place on input, which becomes the output)
///
/// So `input` becomes the normed output and `residual` is the updated residual.
/// Both are mutated in-place.
///
/// For arena-based flow: we allocate a copy of `input` for the normed output,
/// and the caller must own the residual buffer.
///
/// * `input`: `[num_tokens, hidden_size]` — will contain normed output after call
/// * `residual`: `[num_tokens, hidden_size]` — will be updated in-place (res += input)
/// * `weight`: `[hidden_size]`
/// * `eps`: normalization epsilon
///
/// Returns: `(normed, residual)` — normed is the same buffer as `input` (now overwritten),
/// residual is the same buffer (updated in-place).
pub unsafe fn fused_add_rms_norm_inplace(
    input: GpuTensor,
    residual: GpuTensor,
    weight: GpuTensor,
    eps: f32,
    stream: CUstream,
) -> (GpuTensor, GpuTensor) {
    fused_add_rms_norm_inplace_with_offset(input, residual, weight, eps, 0.0, stream)
}

/// Same as [`fused_add_rms_norm_inplace`] but with a
/// `(weight + weight_offset)` fold for Gemma-style `(1+w)` rmsnorm.
pub unsafe fn fused_add_rms_norm_inplace_with_offset(
    input: GpuTensor,
    residual: GpuTensor,
    weight: GpuTensor,
    eps: f32,
    weight_offset: f32,
    stream: CUstream,
) -> (GpuTensor, GpuTensor) {
    let num_tokens = input.dim(0) as i32;
    let hidden_size = input.dim(1) as i32;

    match input.dtype() {
        DType::F16 => fused_add_rms_norm_f16(
            input.as_mut_ptr(),
            residual.as_mut_ptr(),
            weight.as_ptr(),
            eps,
            weight_offset,
            num_tokens,
            hidden_size,
            stream,
        ),
        DType::BF16 => fused_add_rms_norm_bf16(
            input.as_mut_ptr(),
            residual.as_mut_ptr(),
            weight.as_ptr(),
            eps,
            weight_offset,
            num_tokens,
            hidden_size,
            stream,
        ),
        DType::F32 => fused_add_rms_norm_f32(
            input.as_mut_ptr(),
            residual.as_mut_ptr(),
            weight.as_ptr(),
            eps,
            weight_offset,
            num_tokens,
            hidden_size,
            stream,
        ),
        _ => panic!("fused_add_rms_norm: unsupported dtype {:?}", input.dtype()),
    }

    // input now contains normed output, residual is updated.
    (input, residual)
}

/// Fused add + RMS norm with arena-allocated output.
///
/// Copies `input` to an arena buffer first (so the original `input` is not mutated),
/// then runs the in-place fused kernel.
///
/// Returns `(normed, residual)` where normed is arena-allocated.
pub unsafe fn fused_add_rms_norm(
    input: GpuTensor,
    residual: GpuTensor,
    weight: GpuTensor,
    eps: f32,
    alloc: &mut CachingAllocator,
    stream: cudarc::driver::sys::CUstream,
) -> (OwnedTensor, GpuTensor) {
    // Allocate a copy of input for the normed output.
    let normed_buf = alloc.alloc_tensor(&[input.dim(0), input.dim(1)], input.dtype());
    crate::driver::memcpy_dtod_async(
        normed_buf.as_gpu_tensor().raw_ptr(),
        input.raw_ptr() as *const u8,
        input.size_bytes(),
        stream,
    )
    .expect("fused_add_rms_norm: D2D copy failed");

    // Run in-place kernel on normed_buf (which is a copy of input).
    let normed_gpu = normed_buf.as_gpu_tensor();
    fused_add_rms_norm_inplace(normed_gpu, residual, weight, eps, stream);

    // normed_buf now contains normed output, residual is updated in-place.
    (normed_buf, residual)
}

// ---------------------------------------------------------------------------
// Cohere LayerNorm
// ---------------------------------------------------------------------------

/// Cohere LayerNorm: `out = weight * (input - mean(input)) / sqrt(var(input) + eps)`
///
/// Full LayerNorm with mean subtraction, weight only (no bias).
/// Used by Command R (CohereForCausalLM).
///
/// * `input`: `[num_tokens, hidden_size]`
/// * `weight`: `[hidden_size]`
/// * Returns: `[num_tokens, hidden_size]` allocated from arena.
pub unsafe fn cohere_layer_norm(
    input: GpuTensor,
    weight: GpuTensor,
    eps: f32,
    alloc: &mut CachingAllocator,
    stream: CUstream,
) -> OwnedTensor {
    let num_tokens = input.dim(0) as i32;
    let hidden_size = input.dim(1) as i32;
    // GUARD: the kernel reads `weight` with the SAME element type it dispatches on
    // (`input.dtype()`). A mismatched gain dtype (e.g. an FP16 RMSNorm gain fed to
    // the BF16 path) is reinterpreted bit-for-bit and silently corrupts the result
    // with no crash — exactly the Gemma-4 router degeneracy (FP16 gain read as BF16
    // ≈ 7e9 collapsed the top-k softmax). Fail loudly instead.
    assert_eq!(
        weight.dtype(),
        input.dtype(),
        "rms_norm: gain dtype {:?} != input dtype {:?}; the kernel reads the gain as \
         input.dtype() and would misread a mismatched gain.",
        weight.dtype(),
        input.dtype()
    );
    let out = alloc.alloc_tensor(&[num_tokens as usize, hidden_size as usize], input.dtype());

    match input.dtype() {
        DType::F16 => cohere_layer_norm_f16(
            out.as_mut_ptr(),
            input.as_ptr(),
            weight.as_ptr(),
            eps,
            num_tokens,
            hidden_size,
            stream,
        ),
        DType::BF16 => cohere_layer_norm_bf16(
            out.as_mut_ptr(),
            input.as_ptr(),
            weight.as_ptr(),
            eps,
            num_tokens,
            hidden_size,
            stream,
        ),
        DType::F32 => cohere_layer_norm_f32(
            out.as_mut_ptr(),
            input.as_ptr(),
            weight.as_ptr(),
            eps,
            num_tokens,
            hidden_size,
            stream,
        ),
        _ => panic!("cohere_layer_norm: unsupported dtype {:?}", input.dtype()),
    }
    out
}

/// Fused add + Cohere LayerNorm: `residual += input; normed = layernorm(residual) * weight`
///
/// Mutates both `input` (becomes normed output) and `residual` (updated in-place).
pub unsafe fn fused_add_cohere_layer_norm_inplace(
    input: GpuTensor,
    residual: GpuTensor,
    weight: GpuTensor,
    eps: f32,
    stream: CUstream,
) -> (GpuTensor, GpuTensor) {
    let num_tokens = input.dim(0) as i32;
    let hidden_size = input.dim(1) as i32;

    match input.dtype() {
        DType::F16 => fused_add_cohere_layer_norm_f16(
            input.as_mut_ptr(),
            residual.as_mut_ptr(),
            weight.as_ptr(),
            eps,
            num_tokens,
            hidden_size,
            stream,
        ),
        DType::BF16 => fused_add_cohere_layer_norm_bf16(
            input.as_mut_ptr(),
            residual.as_mut_ptr(),
            weight.as_ptr(),
            eps,
            num_tokens,
            hidden_size,
            stream,
        ),
        DType::F32 => fused_add_cohere_layer_norm_f32(
            input.as_mut_ptr(),
            residual.as_mut_ptr(),
            weight.as_ptr(),
            eps,
            num_tokens,
            hidden_size,
            stream,
        ),
        _ => panic!(
            "fused_add_cohere_layer_norm: unsupported dtype {:?}",
            input.dtype()
        ),
    }

    (input, residual)
}

// ---------------------------------------------------------------------------
// LayerNorm with bias
// ---------------------------------------------------------------------------

/// Full LayerNorm with mean subtraction, weight and optional bias.
///
/// * `input`: `[num_tokens, hidden_size]`
/// * `weight`: `[hidden_size]`
/// * `bias`: `[hidden_size]`
/// * Returns: `[num_tokens, hidden_size]` allocated from arena.
pub unsafe fn layer_norm_bias(
    input: GpuTensor,
    weight: GpuTensor,
    bias: GpuTensor,
    eps: f32,
    alloc: &mut CachingAllocator,
    stream: CUstream,
) -> OwnedTensor {
    let num_tokens = input.dim(0) as i32;
    let hidden_size = input.dim(1) as i32;
    // GUARD: the kernel reads `weight` with the SAME element type it dispatches on
    // (`input.dtype()`). A mismatched gain dtype (e.g. an FP16 RMSNorm gain fed to
    // the BF16 path) is reinterpreted bit-for-bit and silently corrupts the result
    // with no crash — exactly the Gemma-4 router degeneracy (FP16 gain read as BF16
    // ≈ 7e9 collapsed the top-k softmax). Fail loudly instead.
    assert_eq!(
        weight.dtype(),
        input.dtype(),
        "rms_norm: gain dtype {:?} != input dtype {:?}; the kernel reads the gain as \
         input.dtype() and would misread a mismatched gain.",
        weight.dtype(),
        input.dtype()
    );
    let out = alloc.alloc_tensor(&[num_tokens as usize, hidden_size as usize], input.dtype());

    match input.dtype() {
        DType::F16 => layer_norm_bias_f16(
            out.as_mut_ptr(),
            input.as_ptr(),
            weight.as_ptr(),
            bias.as_ptr(),
            eps,
            num_tokens,
            hidden_size,
            stream,
        ),
        DType::BF16 => layer_norm_bias_bf16(
            out.as_mut_ptr(),
            input.as_ptr(),
            weight.as_ptr(),
            bias.as_ptr(),
            eps,
            num_tokens,
            hidden_size,
            stream,
        ),
        DType::F32 => layer_norm_bias_f32(
            out.as_mut_ptr() as *mut f32,
            input.as_ptr() as *const f32,
            weight.as_ptr() as *const f32,
            bias.as_ptr() as *const f32,
            eps,
            num_tokens,
            hidden_size,
            stream,
        ),
        _ => panic!("layer_norm_bias: unsupported dtype {:?}", input.dtype()),
    }
    out
}

/// Fused add + LayerNorm with bias: `residual += input; normed = layernorm(residual) * weight + bias`
///
/// Mutates both `input` (becomes normed output) and `residual` (updated in-place).
pub unsafe fn fused_add_layer_norm_bias_inplace(
    input: GpuTensor,
    residual: GpuTensor,
    weight: GpuTensor,
    bias: GpuTensor,
    eps: f32,
    stream: CUstream,
) -> (GpuTensor, GpuTensor) {
    let num_tokens = input.dim(0) as i32;
    let hidden_size = input.dim(1) as i32;

    match input.dtype() {
        DType::F16 => fused_add_layer_norm_bias_f16(
            input.as_mut_ptr(),
            residual.as_mut_ptr(),
            weight.as_ptr(),
            bias.as_ptr(),
            eps,
            num_tokens,
            hidden_size,
            stream,
        ),
        DType::BF16 => fused_add_layer_norm_bias_bf16(
            input.as_mut_ptr(),
            residual.as_mut_ptr(),
            weight.as_ptr(),
            bias.as_ptr(),
            eps,
            num_tokens,
            hidden_size,
            stream,
        ),
        DType::F32 => fused_add_layer_norm_bias_f32(
            input.as_mut_ptr() as *mut f32,
            residual.as_mut_ptr() as *mut f32,
            weight.as_ptr() as *const f32,
            bias.as_ptr() as *const f32,
            eps,
            num_tokens,
            hidden_size,
            stream,
        ),
        _ => panic!(
            "fused_add_layer_norm_bias: unsupported dtype {:?}",
            input.dtype()
        ),
    }

    (input, residual)
}

// ---------------------------------------------------------------------------
// Fused QKV split + interleaved RoPE (Cohere convention)
// ---------------------------------------------------------------------------

/// Fused QKV split + interleaved RoPE.
///
/// Like `fused_qkv_rope` but pairs adjacent elements (2i, 2i+1) for rotation
/// instead of NeoX-style (i, i+half). Used by Command R (CohereForCausalLM).
///
/// * `qkv`: `[num_tokens, q_size + 2*kv_size]` — fused QKV GEMM output
/// * `positions`: `[num_tokens]` u32
/// * `cos_sin_cache`: `[max_pos, rotary_dim]`
/// * Returns: `(q, k, v)` where q is `[num_tokens, num_q_heads, head_dim]`,
///   k and v are `[num_tokens, num_kv_heads, head_dim]`.
pub unsafe fn fused_qkv_interleaved_rope(
    qkv: GpuTensor,
    positions: GpuTensor,
    cos_sin_cache: GpuTensor,
    q_size: usize,
    kv_size: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    alloc: &mut CachingAllocator,
    stream: CUstream,
) -> (OwnedTensor, OwnedTensor, OwnedTensor) {
    let num_tokens = qkv.dim(0);
    let total_dim = qkv.dim(1);
    let rotary_dim = cos_sin_cache.dim(1) as i32;

    debug_assert_eq!(total_dim, q_size + 2 * kv_size);

    let q = alloc.alloc_tensor(&[num_tokens, num_q_heads, head_dim], qkv.dtype());
    let k = alloc.alloc_tensor(&[num_tokens, num_kv_heads, head_dim], qkv.dtype());
    let v = alloc.alloc_tensor(&[num_tokens, num_kv_heads, head_dim], qkv.dtype());

    match qkv.dtype() {
        DType::F16 => fused_qkv_interleaved_rope_f16(
            q.as_mut_ptr(),
            k.as_mut_ptr(),
            v.as_mut_ptr(),
            qkv.as_ptr(),
            positions.as_ptr(),
            cos_sin_cache.as_ptr(),
            q_size as i32,
            kv_size as i32,
            total_dim as i32,
            rotary_dim,
            head_dim as i32,
            num_tokens as i32,
            stream,
        ),
        DType::BF16 => fused_qkv_interleaved_rope_bf16(
            q.as_mut_ptr(),
            k.as_mut_ptr(),
            v.as_mut_ptr(),
            qkv.as_ptr(),
            positions.as_ptr(),
            cos_sin_cache.as_ptr(),
            q_size as i32,
            kv_size as i32,
            total_dim as i32,
            rotary_dim,
            head_dim as i32,
            num_tokens as i32,
            stream,
        ),
        DType::F32 => fused_qkv_interleaved_rope_f32(
            q.as_mut_ptr(),
            k.as_mut_ptr(),
            v.as_mut_ptr(),
            qkv.as_ptr(),
            positions.as_ptr(),
            cos_sin_cache.as_ptr(),
            q_size as i32,
            kv_size as i32,
            total_dim as i32,
            rotary_dim,
            head_dim as i32,
            num_tokens as i32,
            stream,
        ),
        _ => panic!(
            "fused_qkv_interleaved_rope: unsupported dtype {:?}",
            qkv.dtype()
        ),
    }

    (q, k, v)
}

// ---------------------------------------------------------------------------
// Fused SiLU-and-Mul
// ---------------------------------------------------------------------------

/// Fused SiLU(gate) * up from combined gate_up tensor.
///
/// * `gate_up`: `[num_tokens, 2 * intermediate_size]`
/// * `intermediate_size`: the "d" dimension
/// * Returns: `[num_tokens, intermediate_size]` from arena.
pub unsafe fn silu_and_mul_fused(
    gate_up: GpuTensor,
    intermediate_size: usize,
    alloc: &mut CachingAllocator,
    stream: CUstream,
) -> OwnedTensor {
    let num_tokens = gate_up.dim(0) as i32;
    let d = intermediate_size as i32;
    let out = alloc.alloc_tensor(&[num_tokens as usize, intermediate_size], gate_up.dtype());

    match gate_up.dtype() {
        DType::F16 => {
            silu_and_mul_fused_f16(out.as_mut_ptr(), gate_up.as_ptr(), num_tokens, d, stream)
        }
        DType::BF16 => {
            silu_and_mul_fused_bf16(out.as_mut_ptr(), gate_up.as_ptr(), num_tokens, d, stream)
        }
        DType::F32 => {
            silu_and_mul_fused_f32(out.as_mut_ptr(), gate_up.as_ptr(), num_tokens, d, stream)
        }
        _ => panic!(
            "silu_and_mul_fused: unsupported dtype {:?}",
            gate_up.dtype()
        ),
    }
    out
}

// ---------------------------------------------------------------------------
// Fused GELU-and-Mul
// ---------------------------------------------------------------------------

/// Fused GELU(tanh)(gate) * up from combined gate_up tensor.
///
/// * `gate_up`: `[num_tokens, 2 * intermediate_size]`
/// * `intermediate_size`: the "d" dimension
/// * Returns: `[num_tokens, intermediate_size]` from arena.
pub unsafe fn gelu_and_mul_fused(
    gate_up: GpuTensor,
    intermediate_size: usize,
    alloc: &mut CachingAllocator,
    stream: CUstream,
) -> OwnedTensor {
    let num_tokens = gate_up.dim(0) as i32;
    let d = intermediate_size as i32;
    let out = alloc.alloc_tensor(&[num_tokens as usize, intermediate_size], gate_up.dtype());

    match gate_up.dtype() {
        DType::F16 => {
            gelu_and_mul_fused_f16(out.as_mut_ptr(), gate_up.as_ptr(), num_tokens, d, stream)
        }
        DType::BF16 => {
            gelu_and_mul_fused_bf16(out.as_mut_ptr(), gate_up.as_ptr(), num_tokens, d, stream)
        }
        DType::F32 => {
            gelu_and_mul_fused_f32(out.as_mut_ptr(), gate_up.as_ptr(), num_tokens, d, stream)
        }
        _ => panic!(
            "gelu_and_mul_fused: unsupported dtype {:?}",
            gate_up.dtype()
        ),
    }
    out
}

/// Non-fused GELU(tanh)(gate) * up from SEPARATE `gate` / `up` `[M, d]` tensors.
///
/// `gelu_pytorch_tanh(gate) * up` — the Gemma GeGLU activation. Used by the
/// Gemma-4 SwitchGLU expert path, whose gate and up come from two separate
/// `fused_moe_gemm` calls (so the interleaved `gelu_and_mul_fused` doesn't fit).
/// Returns `[M, d]` in the input dtype.
pub unsafe fn gelu_and_mul(
    gate: GpuTensor,
    up: GpuTensor,
    alloc: &mut CachingAllocator,
    stream: CUstream,
) -> OwnedTensor {
    debug_assert_eq!(gate.ndim(), 2);
    debug_assert_eq!(gate.dim(0), up.dim(0));
    debug_assert_eq!(gate.dim(1), up.dim(1));
    let num_tokens = gate.dim(0) as i32;
    let d = gate.dim(1) as i32;
    let out = alloc.alloc_tensor(&[gate.dim(0), gate.dim(1)], gate.dtype());

    match gate.dtype() {
        DType::F16 => gelu_and_mul_f16(
            out.as_mut_ptr(),
            gate.as_ptr(),
            up.as_ptr(),
            num_tokens,
            d,
            stream,
        ),
        DType::BF16 => gelu_and_mul_bf16(
            out.as_mut_ptr(),
            gate.as_ptr(),
            up.as_ptr(),
            num_tokens,
            d,
            stream,
        ),
        DType::F32 => gelu_and_mul_f32(
            out.as_mut_ptr(),
            gate.as_ptr(),
            up.as_ptr(),
            num_tokens,
            d,
            stream,
        ),
        _ => panic!("gelu_and_mul: unsupported dtype {:?}", gate.dtype()),
    }
    out
}

// ---------------------------------------------------------------------------
// QuickGELU (in-place pointwise)
// ---------------------------------------------------------------------------

/// In-place QuickGELU: `x[i] = x[i] * sigmoid(1.702 * x[i])`.
///
/// Used by Qwen2-VL's vision MLP (the OpenAI/CLIP approximation —
/// NOT the same as `gelu_and_mul_fused`'s tanh-form approximation
/// or `gelu_erf`). The 1.702 multiplier is tuned to approximate
/// erf-GELU more cheaply.
///
/// # Safety
/// `x` must be valid GPU memory; in-place write semantics — caller
/// serializes against any other kernel touching the buffer.
pub unsafe fn quick_gelu_inplace(x: GpuTensor, stream: CUstream) {
    let n = x.numel() as c_int;
    if n == 0 {
        return;
    }
    match x.dtype() {
        DType::F16 => quick_gelu_inplace_f16(x.as_mut_ptr(), n, stream),
        DType::BF16 => quick_gelu_inplace_bf16(x.as_mut_ptr(), n, stream),
        DType::F32 => quick_gelu_inplace_f32(x.as_mut_ptr(), n, stream),
        _ => panic!("quick_gelu_inplace: unsupported dtype {:?}", x.dtype()),
    }
}

/// In-place GELU (erf form): `x[i] = x[i] * 0.5 * (1 + erf(x[i] / sqrt(2)))`.
///
/// Matches PyTorch `nn.GELU()` default (`approximate='none'`). Used by
/// Qwen2-VL's `Qwen2VisionPatchMerger.mlp[1]`. Distinct from
/// `quick_gelu_inplace` (1.702 sigmoid approximation) and the tanh-form
/// `gelu_and_mul_fused`.
///
/// # Safety
/// Same as [`quick_gelu_inplace`].
pub unsafe fn gelu_erf_inplace(x: GpuTensor, stream: CUstream) {
    let n = x.numel() as c_int;
    if n == 0 {
        return;
    }
    match x.dtype() {
        DType::F16 => gelu_erf_inplace_f16(x.as_mut_ptr(), n, stream),
        DType::BF16 => gelu_erf_inplace_bf16(x.as_mut_ptr(), n, stream),
        DType::F32 => gelu_erf_inplace_f32(x.as_mut_ptr(), n, stream),
        _ => panic!("gelu_erf_inplace: unsupported dtype {:?}", x.dtype()),
    }
}

/// In-place GELU (tanh approximation):
/// `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`.
///
/// Matches PyTorch `nn.GELU(approximate="tanh")` / HuggingFace
/// `gelu_pytorch_tanh`. Used by SigLIP / Gemma3-MM vision MLP.
/// Distinct from `quick_gelu_inplace` (1.702 sigmoid) and
/// `gelu_erf_inplace` (exact erf).
///
/// # Safety
/// Same as [`quick_gelu_inplace`].
pub unsafe fn gelu_tanh_inplace(x: GpuTensor, stream: CUstream) {
    let n = x.numel() as c_int;
    if n == 0 {
        return;
    }
    match x.dtype() {
        DType::F16 => gelu_tanh_inplace_f16(x.as_mut_ptr(), n, stream),
        DType::BF16 => gelu_tanh_inplace_bf16(x.as_mut_ptr(), n, stream),
        DType::F32 => gelu_tanh_inplace_f32(x.as_mut_ptr(), n, stream),
        _ => panic!("gelu_tanh_inplace: unsupported dtype {:?}", x.dtype()),
    }
}

// ---------------------------------------------------------------------------
// AvgPool2d (non-overlapping) over a flat patch grid
// ---------------------------------------------------------------------------

/// 2-D non-overlapping average pool over a square patch grid.
///
/// Treats `x: [ph * ph, e]` as `[ph, ph, e]`, applies a stride-k k×k
/// average pool (stride == kernel — non-overlapping), and flattens back
/// to `[(ph/k) * (ph/k), e]`. Used by Gemma3-MM's SigLIP→text projector
/// (Phase G.7(b)).
///
/// `ph` must be divisible by `k`. Allocates a fresh output tensor from
/// the caching allocator.
///
/// # Safety
/// `x` must be a valid `[ph * ph, e]` GPU tensor; the caller serializes
/// against other kernel writes to `x`.
pub unsafe fn avg_pool_2d(
    x: GpuTensor,
    ph: u32,
    k: u32,
    alloc: &mut CachingAllocator,
    stream: CUstream,
) -> OwnedTensor {
    let l = x.dim(0);
    let e = x.dim(1);
    let ph_us = ph as usize;
    let k_us = k as usize;
    assert!(
        ph_us > 0 && k_us > 0 && ph_us.is_multiple_of(k_us),
        "avg_pool_2d: ph ({ph}) must be a positive multiple of k ({k})"
    );
    assert_eq!(
        l,
        ph_us * ph_us,
        "avg_pool_2d: input leading dim {l} must equal ph² ({})",
        ph_us * ph_us,
    );
    let ph_out = ph_us / k_us;
    let l_out = ph_out * ph_out;
    let out = alloc.alloc_tensor(&[l_out, e], x.dtype());
    let ph_c = ph as c_int;
    let k_c = k as c_int;
    let e_c = e as c_int;
    match x.dtype() {
        DType::F16 => avg_pool_2d_f16(out.as_mut_ptr(), x.as_ptr(), ph_c, k_c, e_c, stream),
        DType::BF16 => avg_pool_2d_bf16(out.as_mut_ptr(), x.as_ptr(), ph_c, k_c, e_c, stream),
        DType::F32 => avg_pool_2d_f32(out.as_mut_ptr(), x.as_ptr(), ph_c, k_c, e_c, stream),
        _ => panic!("avg_pool_2d: unsupported dtype {:?}", x.dtype()),
    }
    out
}

// ---------------------------------------------------------------------------
// Broadcast Multiply Inplace
// ---------------------------------------------------------------------------

/// `x[row, col] *= scale[col]` — in-place broadcast multiply.
///
/// * `x`: `[num_rows, d]` — mutated in-place
/// * `scale`: `[d]`
#[cfg(feature = "cuda")]
pub unsafe fn broadcast_mul_inplace(x: GpuTensor, scale: GpuTensor, stream: CUstream) {
    let num_rows = x.dim(0) as i32;
    let d = x.dim(x.ndim() - 1) as i32;
    match x.dtype() {
        DType::F16 => {
            broadcast_mul_inplace_f16(x.as_mut_ptr(), scale.as_ptr(), num_rows, d, stream)
        }
        DType::BF16 => {
            broadcast_mul_inplace_bf16(x.as_mut_ptr(), scale.as_ptr(), num_rows, d, stream)
        }
        DType::F32 => broadcast_mul_inplace_f32(
            x.as_mut_ptr() as *mut f32,
            scale.as_ptr() as *const f32,
            num_rows,
            d,
            stream,
        ),
        _ => panic!("broadcast_mul_inplace: unsupported dtype {:?}", x.dtype()),
    }
}

// ---------------------------------------------------------------------------
// ScalarWeightMul: out = in * w[0] (Gemma4 layer_scalar)
// ---------------------------------------------------------------------------

unsafe extern "C" {
    fn scalar_weight_mul_f16(
        out: *mut u16,
        input: *const u16,
        w: *const u16,
        n: c_int,
        s: CUstream,
    );
    fn scalar_weight_mul_bf16(
        out: *mut u16,
        input: *const u16,
        w: *const u16,
        n: c_int,
        s: CUstream,
    );
    fn scalar_weight_mul_f32(
        out: *mut f32,
        input: *const f32,
        w: *const f32,
        n: c_int,
        s: CUstream,
    );
}

/// `out = input * weight[0]` — multiply every element of `input` by a single
/// `[1]`-shaped loaded scalar weight (Gemma-4 `layer_scalar`). Allocates a fresh
/// output of the same shape/dtype.
pub unsafe fn scalar_weight_mul(
    input: GpuTensor,
    weight: GpuTensor,
    alloc: &mut CachingAllocator,
    stream: CUstream,
) -> OwnedTensor {
    let shape: Vec<usize> = input.shape().iter().map(|&d| d as usize).collect();
    let out = alloc.alloc_tensor(&shape, input.dtype());
    let n = input.numel() as c_int;
    match input.dtype() {
        DType::F16 => {
            scalar_weight_mul_f16(out.as_mut_ptr(), input.as_ptr(), weight.as_ptr(), n, stream)
        }
        DType::BF16 => {
            scalar_weight_mul_bf16(out.as_mut_ptr(), input.as_ptr(), weight.as_ptr(), n, stream)
        }
        DType::F32 => scalar_weight_mul_f32(
            out.as_mut_ptr() as *mut f32,
            input.as_ptr() as *const f32,
            weight.as_ptr() as *const f32,
            n,
            stream,
        ),
        dt => panic!("scalar_weight_mul: unsupported dtype {dt:?}"),
    }
    out
}

// ---------------------------------------------------------------------------
// Tanh Softcap (in-place)
// ---------------------------------------------------------------------------

/// Apply `x[i] = cap * tanh(x[i] / cap)` in-place.
/// Used for Gemma 2 final logit softcapping.
pub unsafe fn tanh_softcap_inplace(x: GpuTensor, cap: f32, stream: CUstream) {
    let n = x.numel() as c_int;
    match x.dtype() {
        DType::F16 => tanh_softcap_inplace_f16(x.as_mut_ptr(), cap, n, stream),
        DType::BF16 => tanh_softcap_inplace_bf16(x.as_mut_ptr(), cap, n, stream),
        DType::F32 => tanh_softcap_inplace_f32(x.as_mut_ptr() as *mut f32, cap, n, stream),
        _ => panic!("tanh_softcap_inplace: unsupported dtype {:?}", x.dtype()),
    }
}

// ---------------------------------------------------------------------------
// Rotary Embedding
// ---------------------------------------------------------------------------

/// Fused rotary embedding applied in-place to Q and K.
///
/// * `q`: `[num_tokens, num_q_heads * head_dim]` — mutated in-place
/// * `k`: `[num_tokens, num_kv_heads * head_dim]` — mutated in-place
/// * `positions`: `[num_tokens]` (U32)
/// * `cos_sin_cache`: `[max_pos, rotary_dim]`
/// * `head_dim`: dimension per head
pub unsafe fn rotary_embedding_inplace(
    q: GpuTensor,
    k: GpuTensor,
    positions: GpuTensor,
    cos_sin_cache: GpuTensor,
    head_dim: usize,
    stream: CUstream,
) {
    let num_tokens = q.dim(0) as i32;
    let total_q_dim = q.dim(1) as i32;
    let total_k_dim = k.dim(1) as i32;
    let rotary_dim = cos_sin_cache.dim(1) as i32;
    let head_size = head_dim as i32;

    match q.dtype() {
        DType::F16 => rotary_embedding_f16(
            positions.as_ptr(),
            q.as_mut_ptr(),
            k.as_mut_ptr(),
            cos_sin_cache.as_ptr(),
            rotary_dim,
            total_q_dim,
            total_k_dim,
            head_size,
            num_tokens,
            stream,
        ),
        DType::BF16 => rotary_embedding_bf16(
            positions.as_ptr(),
            q.as_mut_ptr(),
            k.as_mut_ptr(),
            cos_sin_cache.as_ptr(),
            rotary_dim,
            total_q_dim,
            total_k_dim,
            head_size,
            num_tokens,
            stream,
        ),
        DType::F32 => rotary_embedding_f32(
            positions.as_ptr(),
            q.as_mut_ptr(),
            k.as_mut_ptr(),
            cos_sin_cache.as_ptr(),
            rotary_dim,
            total_q_dim,
            total_k_dim,
            head_size,
            num_tokens,
            stream,
        ),
        _ => panic!("rotary_embedding: unsupported dtype {:?}", q.dtype()),
    }
}

unsafe extern "C" {
    fn rotary_embedding_proportional_bf16(
        positions: *const u32,
        query: *mut u16,
        key: *mut u16,
        cos_sin_cache: *const u16,
        rotary_dim: c_int,
        total_q_dim: c_int,
        total_k_dim: c_int,
        head_size: c_int,
        pair_off: c_int,
        num_tokens: c_int,
        stream: CUstream,
    );
}

/// Gemma-4 GLOBAL proportional RoPE (mlx ProportionalRoPE): pairs lane `i` with
/// lane `i + head_dim/2`, rotating only `rotary_dim/2` lanes (rot_dim < head_dim).
/// In-place on Q `[nt, num_q_heads*head_dim]` and K `[nt, num_kv_heads*head_dim]`.
/// `cos_sin_cache` is `[max_pos, rotary_dim]` (the global theta=1e6 cache).
///
/// # Safety
/// BF16 Q/K, valid positions (u32) + cache; head_dim even.
pub unsafe fn rotary_embedding_proportional_inplace(
    q: GpuTensor,
    k: GpuTensor,
    positions: GpuTensor,
    cos_sin_cache: GpuTensor,
    head_dim: usize,
    stream: CUstream,
) {
    let num_tokens = q.dim(0) as c_int;
    let total_q_dim = q.dim(1) as c_int;
    let total_k_dim = k.dim(1) as c_int;
    let rotary_dim = cos_sin_cache.dim(1) as c_int;
    assert!(
        matches!(q.dtype(), DType::BF16),
        "rotary_embedding_proportional: only BF16 (got {:?})",
        q.dtype()
    );
    rotary_embedding_proportional_bf16(
        positions.as_ptr(),
        q.as_mut_ptr(),
        k.as_mut_ptr(),
        cos_sin_cache.as_ptr(),
        rotary_dim,
        total_q_dim,
        total_k_dim,
        head_dim as c_int,
        (head_dim / 2) as c_int,
        num_tokens,
        stream,
    );
}

// ---------------------------------------------------------------------------
// Interleaved Rotary Embedding (in-place, DeepSeek MLA style)
// ---------------------------------------------------------------------------

/// Interleaved RoPE in-place on Q and K.
///
/// Like `rotary_embedding_inplace` but uses interleaved pair layout
/// (pairs at [2i, 2i+1]) instead of NeoX layout (pairs at [i, i+half]).
/// Used by DeepSeek V2/V3 MLA attention.
///
/// * `q`: `[num_tokens, total_q_dim]` — modified in place
/// * `k`: `[num_tokens, total_k_dim]` — modified in place
/// * `positions`: `[num_tokens]` (U32)
/// * `cos_sin_cache`: `[max_pos, rotary_dim]`
/// * `head_dim`: dimension per head (for stride computation)
pub unsafe fn rotary_embedding_interleaved_inplace(
    q: GpuTensor,
    k: GpuTensor,
    positions: GpuTensor,
    cos_sin_cache: GpuTensor,
    head_dim: usize,
    stream: CUstream,
) {
    let num_tokens = q.dim(0) as i32;
    let total_q_dim = q.dim(1) as i32;
    let total_k_dim = k.dim(1) as i32;
    let rotary_dim = cos_sin_cache.dim(1) as i32;
    let head_size = head_dim as i32;

    match q.dtype() {
        DType::F16 => rotary_embedding_interleaved_f16(
            positions.as_ptr(),
            q.as_mut_ptr(),
            k.as_mut_ptr(),
            cos_sin_cache.as_ptr(),
            rotary_dim,
            total_q_dim,
            total_k_dim,
            head_size,
            num_tokens,
            stream,
        ),
        DType::BF16 => rotary_embedding_interleaved_bf16(
            positions.as_ptr(),
            q.as_mut_ptr(),
            k.as_mut_ptr(),
            cos_sin_cache.as_ptr(),
            rotary_dim,
            total_q_dim,
            total_k_dim,
            head_size,
            num_tokens,
            stream,
        ),
        DType::F32 => rotary_embedding_interleaved_f32(
            positions.as_ptr(),
            q.as_mut_ptr(),
            k.as_mut_ptr(),
            cos_sin_cache.as_ptr(),
            rotary_dim,
            total_q_dim,
            total_k_dim,
            head_size,
            num_tokens,
            stream,
        ),
        _ => panic!(
            "rotary_embedding_interleaved: unsupported dtype {:?}",
            q.dtype()
        ),
    }
}

// ---------------------------------------------------------------------------
// MLA Data Movement (DeepSeek V2/V3)
// ---------------------------------------------------------------------------

/// Split kv_a output into latent and k_pe.
/// * `src`: `[num_tokens, kv_lora_rank + rope_dim]`
/// * `dst_latent`: `[num_tokens, kv_lora_rank]`
/// * `dst_k_pe`: `[num_tokens, rope_dim]`
pub unsafe fn mla_split_kv_a(
    src: GpuTensor,
    dst_latent: GpuTensor,
    dst_k_pe: GpuTensor,
    kv_lora_rank: usize,
    rope_dim: usize,
    stream: CUstream,
) {
    let num_tokens = src.dim(0) as c_int;
    let s = src.raw_ptr() as *const c_void;
    let dl = dst_latent.raw_ptr() as *mut c_void;
    let dk = dst_k_pe.raw_ptr() as *mut c_void;
    match src.dtype() {
        DType::F16 => mla_split_kv_a_f16(
            s,
            dl,
            dk,
            num_tokens,
            kv_lora_rank as c_int,
            rope_dim as c_int,
            stream,
        ),
        DType::BF16 => mla_split_kv_a_bf16(
            s,
            dl,
            dk,
            num_tokens,
            kv_lora_rank as c_int,
            rope_dim as c_int,
            stream,
        ),
        DType::F32 => mla_split_kv_a_f32(
            s,
            dl,
            dk,
            num_tokens,
            kv_lora_rank as c_int,
            rope_dim as c_int,
            stream,
        ),
        _ => panic!("mla_split_kv_a: unsupported dtype {:?}", src.dtype()),
    }
}

/// Extract q_pe (rope portion) from Q projection output.
/// * `src`: `[num_tokens, num_heads * qk_head_dim]`
/// * `dst`: `[num_tokens, num_heads * rope_dim]`
pub unsafe fn mla_extract_q_pe(
    src: GpuTensor,
    dst: GpuTensor,
    num_heads: usize,
    qk_head_dim: usize,
    qk_nope_head_dim: usize,
    rope_dim: usize,
    stream: CUstream,
) {
    let num_tokens = src.dim(0) as c_int;
    let s = src.raw_ptr() as *const c_void;
    let d = dst.raw_ptr() as *mut c_void;
    match src.dtype() {
        DType::F16 => mla_extract_q_pe_f16(
            s,
            d,
            num_tokens,
            num_heads as c_int,
            qk_head_dim as c_int,
            qk_nope_head_dim as c_int,
            rope_dim as c_int,
            stream,
        ),
        DType::BF16 => mla_extract_q_pe_bf16(
            s,
            d,
            num_tokens,
            num_heads as c_int,
            qk_head_dim as c_int,
            qk_nope_head_dim as c_int,
            rope_dim as c_int,
            stream,
        ),
        DType::F32 => mla_extract_q_pe_f32(
            s,
            d,
            num_tokens,
            num_heads as c_int,
            qk_head_dim as c_int,
            qk_nope_head_dim as c_int,
            rope_dim as c_int,
            stream,
        ),
        _ => panic!("mla_extract_q_pe: unsupported dtype {:?}", src.dtype()),
    }
}

/// Write q_pe back into Q projection output after RoPE.
/// * `src`: `[num_tokens, num_heads * rope_dim]` (RoPE'd q_pe)
/// * `dst`: `[num_tokens, num_heads * qk_head_dim]` (Q to write into)
pub unsafe fn mla_write_q_pe(
    src: GpuTensor,
    dst: GpuTensor,
    num_heads: usize,
    qk_head_dim: usize,
    qk_nope_head_dim: usize,
    rope_dim: usize,
    stream: CUstream,
) {
    let num_tokens = src.dim(0) as c_int;
    let s = src.raw_ptr() as *const c_void;
    let d = dst.raw_ptr() as *mut c_void;
    match src.dtype() {
        DType::F16 => mla_write_q_pe_f16(
            s,
            d,
            num_tokens,
            num_heads as c_int,
            qk_head_dim as c_int,
            qk_nope_head_dim as c_int,
            rope_dim as c_int,
            stream,
        ),
        DType::BF16 => mla_write_q_pe_bf16(
            s,
            d,
            num_tokens,
            num_heads as c_int,
            qk_head_dim as c_int,
            qk_nope_head_dim as c_int,
            rope_dim as c_int,
            stream,
        ),
        DType::F32 => mla_write_q_pe_f32(
            s,
            d,
            num_tokens,
            num_heads as c_int,
            qk_head_dim as c_int,
            qk_nope_head_dim as c_int,
            rope_dim as c_int,
            stream,
        ),
        _ => panic!("mla_write_q_pe: unsupported dtype {:?}", src.dtype()),
    }
}

/// Assemble K from k_nope (in kv_b) + broadcast k_pe.
/// * `kv_b`: `[num_tokens, num_heads * (nope_dim + v_head_dim)]`
/// * `k_pe`: `[num_tokens, rope_dim]` (single head, broadcast)
/// * `dst`: `[num_tokens, num_heads * qk_head_dim]`
pub unsafe fn mla_assemble_k(
    kv_b: GpuTensor,
    k_pe: GpuTensor,
    dst: GpuTensor,
    num_heads: usize,
    qk_nope_head_dim: usize,
    qk_rope_head_dim: usize,
    v_head_dim: usize,
    qk_head_dim: usize,
    stream: CUstream,
) {
    let num_tokens = kv_b.dim(0) as c_int;
    let kb = kv_b.raw_ptr() as *const c_void;
    let kp = k_pe.raw_ptr() as *const c_void;
    let d = dst.raw_ptr() as *mut c_void;
    match kv_b.dtype() {
        DType::F16 => mla_assemble_k_f16(
            kb,
            kp,
            d,
            num_tokens,
            num_heads as c_int,
            qk_nope_head_dim as c_int,
            qk_rope_head_dim as c_int,
            v_head_dim as c_int,
            qk_head_dim as c_int,
            stream,
        ),
        DType::BF16 => mla_assemble_k_bf16(
            kb,
            kp,
            d,
            num_tokens,
            num_heads as c_int,
            qk_nope_head_dim as c_int,
            qk_rope_head_dim as c_int,
            v_head_dim as c_int,
            qk_head_dim as c_int,
            stream,
        ),
        DType::F32 => mla_assemble_k_f32(
            kb,
            kp,
            d,
            num_tokens,
            num_heads as c_int,
            qk_nope_head_dim as c_int,
            qk_rope_head_dim as c_int,
            v_head_dim as c_int,
            qk_head_dim as c_int,
            stream,
        ),
        _ => panic!("mla_assemble_k: unsupported dtype {:?}", kv_b.dtype()),
    }
}

/// Assemble V: copy v_head_dim from kv_b into zero-padded buffer.
/// * `kv_b`: `[num_tokens, num_heads * (nope_dim + v_head_dim)]`
/// * `dst`: `[num_tokens, num_heads * qk_head_dim]` (must be pre-zeroed)
pub unsafe fn mla_assemble_v(
    kv_b: GpuTensor,
    dst: GpuTensor,
    num_heads: usize,
    qk_nope_head_dim: usize,
    v_head_dim: usize,
    qk_head_dim: usize,
    stream: CUstream,
) {
    let num_tokens = kv_b.dim(0) as c_int;
    let kb = kv_b.raw_ptr() as *const c_void;
    let d = dst.raw_ptr() as *mut c_void;
    match kv_b.dtype() {
        DType::F16 => mla_assemble_v_f16(
            kb,
            d,
            num_tokens,
            num_heads as c_int,
            qk_nope_head_dim as c_int,
            v_head_dim as c_int,
            qk_head_dim as c_int,
            stream,
        ),
        DType::BF16 => mla_assemble_v_bf16(
            kb,
            d,
            num_tokens,
            num_heads as c_int,
            qk_nope_head_dim as c_int,
            v_head_dim as c_int,
            qk_head_dim as c_int,
            stream,
        ),
        DType::F32 => mla_assemble_v_f32(
            kb,
            d,
            num_tokens,
            num_heads as c_int,
            qk_nope_head_dim as c_int,
            v_head_dim as c_int,
            qk_head_dim as c_int,
            stream,
        ),
        _ => panic!("mla_assemble_v: unsupported dtype {:?}", kv_b.dtype()),
    }
}

/// Slice attention output from qk_head_dim to v_head_dim per head.
/// * `src`: `[num_tokens, num_heads * qk_head_dim]`
/// * `dst`: `[num_tokens, num_heads * v_head_dim]`
pub unsafe fn mla_slice_attn_output(
    src: GpuTensor,
    dst: GpuTensor,
    num_heads: usize,
    qk_head_dim: usize,
    v_head_dim: usize,
    stream: CUstream,
) {
    let num_tokens = src.dim(0) as c_int;
    let s = src.raw_ptr() as *const c_void;
    let d = dst.raw_ptr() as *mut c_void;
    match src.dtype() {
        DType::F16 => mla_slice_attn_output_f16(
            s,
            d,
            num_tokens,
            num_heads as c_int,
            qk_head_dim as c_int,
            v_head_dim as c_int,
            stream,
        ),
        DType::BF16 => mla_slice_attn_output_bf16(
            s,
            d,
            num_tokens,
            num_heads as c_int,
            qk_head_dim as c_int,
            v_head_dim as c_int,
            stream,
        ),
        DType::F32 => mla_slice_attn_output_f32(
            s,
            d,
            num_tokens,
            num_heads as c_int,
            qk_head_dim as c_int,
            v_head_dim as c_int,
            stream,
        ),
        _ => panic!("mla_slice_attn_output: unsupported dtype {:?}", src.dtype()),
    }
}

// ---------------------------------------------------------------------------
// Embedding Gather
// ---------------------------------------------------------------------------

/// Embedding gather: `out[i] = weight[input_ids[i]]`. Convenience
/// wrapper for the unsharded (tp=1, hand-written models, etc.) case
/// — delegates to [`embedding_gather_masked`] with
/// `(vocab_offset = 0, vocab_per_rank = weight.dim(0))`. Existing
/// call sites in scratchy-serving-cuda hand-written models stay byte-equivalent
/// to before tensor-parallel landed.
///
/// * `weight`: `[vocab_size, hidden_size]`
/// * `input_ids`: `[num_tokens]` (U32)
/// * Returns: `[num_tokens, hidden_size]` from arena.
pub unsafe fn embedding_gather(
    weight: GpuTensor,
    input_ids: GpuTensor,
    alloc: &mut CachingAllocator,
    stream: CUstream,
) -> OwnedTensor {
    let vocab_per_rank = weight.dim(0) as u32;
    unsafe { embedding_gather_masked(weight, input_ids, 0, vocab_per_rank, alloc, stream) }
}

/// Embedding gather with vocab-shard masking, used by scratchy at
/// tp>1. Each rank's `weight` is the `[vocab_size / world, hidden]`
/// slice; tokens outside `[vocab_offset, vocab_offset + vocab_per_rank)`
/// get zero-filled output rows so the cross-rank AllReduce-sum (which
/// the lowering pass injects when the embed weight's shard-kind is
/// `ShardDim0`) yields exactly one embedding contribution per token
/// from the rank that owns it. At tp=1 (`vocab_offset = 0`,
/// `vocab_per_rank = weight.dim(0)`) the mask never trips and the
/// output is byte-identical to `embedding_gather`. Mirrors Python
/// vLLM's `VocabParallelEmbedding.forward_native`'s masked-input +
/// `masked_fill_(mask, 0)` pattern.
pub unsafe fn embedding_gather_masked(
    weight: GpuTensor,
    input_ids: GpuTensor,
    vocab_offset: u32,
    vocab_per_rank: u32,
    alloc: &mut CachingAllocator,
    stream: CUstream,
) -> OwnedTensor {
    let num_tokens = input_ids.dim(0);
    let hidden_size = weight.dim(1);
    let out = alloc.alloc_tensor(&[num_tokens, hidden_size], weight.dtype());

    match weight.dtype() {
        DType::F16 => embedding_gather_f16(
            out.as_mut_ptr(),
            weight.as_ptr(),
            input_ids.as_ptr(),
            hidden_size as i32,
            num_tokens as i32,
            vocab_offset,
            vocab_per_rank,
            stream,
        ),
        DType::BF16 => embedding_gather_bf16(
            out.as_mut_ptr(),
            weight.as_ptr(),
            input_ids.as_ptr(),
            hidden_size as i32,
            num_tokens as i32,
            vocab_offset,
            vocab_per_rank,
            stream,
        ),
        DType::F32 => embedding_gather_f32(
            out.as_mut_ptr(),
            weight.as_ptr(),
            input_ids.as_ptr(),
            hidden_size as i32,
            num_tokens as i32,
            vocab_offset,
            vocab_per_rank,
            stream,
        ),
        _ => panic!("embedding_gather: unsupported dtype {:?}", weight.dtype()),
    }
    out
}

// ---------------------------------------------------------------------------
// Update decode metadata on GPU (persistent buffers)
// ---------------------------------------------------------------------------

/// Increment positions, recompute slot_mapping from block_table, and
/// increment seqused_k — all in one kernel launch on the GPU.
///
/// This replaces 3 CPU Vec builds + 3 H2D copies per decode step.
///
/// # Safety
/// All pointers must be valid GPU memory. `positions` and `slot_mapping`
/// must have at least `num_reqs` elements. `seqused_k` must have
/// `num_reqs` elements. `block_table` must be `[num_reqs, max_blocks_per_seq]`.
pub unsafe fn update_decode_metadata_gpu(
    positions: *mut u8,
    slot_mapping: *mut u8,
    seqused_k: *mut u8,
    block_table: *const u8,
    num_reqs: usize,
    block_size: usize,
    max_blocks_per_seq: usize,
    stream: CUstream,
) {
    update_decode_metadata(
        positions as *mut u32,
        slot_mapping as *mut i64,
        seqused_k as *mut i32,
        block_table as *const i32,
        num_reqs as c_int,
        block_size as c_int,
        max_blocks_per_seq as c_int,
        stream,
    );
}

/// Compute cu_seqlens_k (prefix sum) from seqused_k on GPU.
///
/// `cu_seqlens_k` must have at least `num_reqs + 1` i32 elements.
/// Single-thread kernel — batch sizes ≤512, sub-microsecond.
///
/// # Safety
/// Pointers must be valid GPU memory. CUDA context must be current.
pub unsafe fn compute_cu_seqlens_k_gpu(
    seqused_k: *const u8,
    cu_seqlens_k: *mut u8,
    num_reqs: usize,
    stream: CUstream,
) {
    prefix_sum_seqused_k_gpu(
        seqused_k as *const i32,
        cu_seqlens_k as *mut i32,
        num_reqs as c_int,
        stream,
    );
}

// ---------------------------------------------------------------------------
// Reshape and Cache (write new K/V tokens into paged KV cache)
// ---------------------------------------------------------------------------

unsafe extern "C" {
    fn reshape_and_cache_f16(
        key: *const u16,
        value: *const u16,
        key_cache: *mut u16,
        value_cache: *mut u16,
        slot_mapping: *const i64,
        num_tokens: i32,
        num_heads: i32,
        head_dim: i32,
        block_size: i32,
        stream: CUstream,
    );
    fn reshape_and_cache_bf16(
        key: *const u16,
        value: *const u16,
        key_cache: *mut u16,
        value_cache: *mut u16,
        slot_mapping: *const i64,
        num_tokens: i32,
        num_heads: i32,
        head_dim: i32,
        block_size: i32,
        stream: CUstream,
    );
    fn reshape_and_cache_f32(
        key: *const f32,
        value: *const f32,
        key_cache: *mut f32,
        value_cache: *mut f32,
        slot_mapping: *const i64,
        num_tokens: i32,
        num_heads: i32,
        head_dim: i32,
        block_size: i32,
        stream: CUstream,
    );
}

/// Write new K/V tokens into the paged KV cache at the given slot positions.
///
/// * `key`: `[num_tokens, num_kv_heads, head_dim]`
/// * `value`: `[num_tokens, num_kv_heads, head_dim]`
/// * `key_cache`: `[num_blocks, block_size, num_kv_heads, head_dim]`
/// * `value_cache`: same layout
/// * `slot_mapping`: `[num_tokens]` (I64) — absolute slot index for each token
pub unsafe fn reshape_and_cache(
    key: GpuTensor,
    value: GpuTensor,
    key_cache: GpuTensor,
    value_cache: GpuTensor,
    slot_mapping: GpuTensor,
    block_size: usize,
    stream: CUstream,
) {
    let num_tokens = key.dim(0) as i32;
    let num_heads = key.dim(1) as i32;
    let head_dim = key.dim(2) as i32;
    let bs = block_size as i32;

    match key.dtype() {
        DType::F16 => reshape_and_cache_f16(
            key.as_ptr(),
            value.as_ptr(),
            key_cache.as_mut_ptr(),
            value_cache.as_mut_ptr(),
            slot_mapping.as_ptr(),
            num_tokens,
            num_heads,
            head_dim,
            bs,
            stream,
        ),
        DType::BF16 => reshape_and_cache_bf16(
            key.as_ptr(),
            value.as_ptr(),
            key_cache.as_mut_ptr(),
            value_cache.as_mut_ptr(),
            slot_mapping.as_ptr(),
            num_tokens,
            num_heads,
            head_dim,
            bs,
            stream,
        ),
        DType::F32 => reshape_and_cache_f32(
            key.as_ptr(),
            value.as_ptr(),
            key_cache.as_mut_ptr(),
            value_cache.as_mut_ptr(),
            slot_mapping.as_ptr(),
            num_tokens,
            num_heads,
            head_dim,
            bs,
            stream,
        ),
        _ => panic!("reshape_and_cache: unsupported dtype {:?}", key.dtype()),
    }
}

// ---------------------------------------------------------------------------
// FP8 KV Cache: reshape_and_cache + dequant_gather + scale computation
// ---------------------------------------------------------------------------

unsafe extern "C" {
    fn reshape_and_cache_fp8_bf16(
        key: *const u16,
        value: *const u16,
        key_cache: *mut u8,
        value_cache: *mut u8,
        slot_mapping: *const i64,
        k_scale: *const f32,
        v_scale: *const f32,
        num_tokens: i32,
        num_heads: i32,
        head_dim: i32,
        block_size: i32,
        stream: CUstream,
    );
    fn reshape_and_cache_fp8_f16(
        key: *const u16,
        value: *const u16,
        key_cache: *mut u8,
        value_cache: *mut u8,
        slot_mapping: *const i64,
        k_scale: *const f32,
        v_scale: *const f32,
        num_tokens: i32,
        num_heads: i32,
        head_dim: i32,
        block_size: i32,
        stream: CUstream,
    );
    fn dequant_gather_pages_bf16(
        cache: *const u8,
        block_table: *const i32,
        cu_seqlens_k: *const i32,
        scale: f32,
        total_kv_tokens: i32,
        num_heads: i32,
        head_dim: i32,
        block_size: i32,
        max_pages_per_seq: i32,
        batch_size: i32,
        output: *mut u16,
        stream: CUstream,
    );
    fn dequant_gather_pages_f16(
        cache: *const u8,
        block_table: *const i32,
        cu_seqlens_k: *const i32,
        scale: f32,
        total_kv_tokens: i32,
        num_heads: i32,
        head_dim: i32,
        block_size: i32,
        max_pages_per_seq: i32,
        batch_size: i32,
        output: *mut u16,
        stream: CUstream,
    );
    fn gather_pages_bf16(
        cache: *const u16,
        block_table: *const i32,
        total_kv: i32,
        num_heads: i32,
        head_dim: i32,
        block_size: i32,
        output: *mut u16,
        stream: CUstream,
    );
    fn compute_abs_max_and_scale_bf16(
        tensor: *const u16,
        num_elements: i32,
        divisor: f32,
        scale_out: *mut f32,
        stream: CUstream,
    );
}

/// Write BF16/F16 K/V tokens into an FP8 E4M3 paged KV cache.
///
/// * `key`, `value`: `[num_tokens, num_kv_heads, head_dim]` (BF16 or F16)
/// * `key_cache`, `value_cache`: `[num_blocks, block_size, num_kv_heads, head_dim]` (FP8)
/// * `k_scale`, `v_scale`: GPU f32 scalar pointers
#[allow(clippy::too_many_arguments)]
pub unsafe fn reshape_and_cache_fp8(
    key: GpuTensor,
    value: GpuTensor,
    key_cache: GpuTensor,
    value_cache: GpuTensor,
    slot_mapping: GpuTensor,
    k_scale: *const f32,
    v_scale: *const f32,
    block_size: usize,
    stream: CUstream,
) {
    let num_tokens = key.dim(0) as i32;
    let num_heads = key.dim(1) as i32;
    let head_dim = key.dim(2) as i32;
    let bs = block_size as i32;

    match key.dtype() {
        DType::BF16 => reshape_and_cache_fp8_bf16(
            key.as_ptr(),
            value.as_ptr(),
            key_cache.as_mut_ptr(),
            value_cache.as_mut_ptr(),
            slot_mapping.as_ptr(),
            k_scale,
            v_scale,
            num_tokens,
            num_heads,
            head_dim,
            bs,
            stream,
        ),
        DType::F16 => reshape_and_cache_fp8_f16(
            key.as_ptr(),
            value.as_ptr(),
            key_cache.as_mut_ptr(),
            value_cache.as_mut_ptr(),
            slot_mapping.as_ptr(),
            k_scale,
            v_scale,
            num_tokens,
            num_heads,
            head_dim,
            bs,
            stream,
        ),
        _ => panic!(
            "reshape_and_cache_fp8: input must be BF16 or F16, got {:?}",
            key.dtype()
        ),
    }
}

/// Dequantize FP8 pages from KV cache into a contiguous BF16/F16 tensor.
///
/// * `cache`: `[num_blocks, block_size, num_heads, head_dim]` (FP8)
/// * `block_table`: `[batch_size, max_pages_per_seq]` (I32) on GPU
/// * `cu_seqlens_k`: `[batch_size + 1]` (I32) prefix sum on GPU
/// * `scale`: scale factor (host float)
/// * Returns: `[total_kv_tokens, num_heads, head_dim]` in `output_dtype`
#[allow(clippy::too_many_arguments)]
pub unsafe fn dequant_gather_pages(
    cache: GpuTensor,
    block_table: GpuTensor,
    cu_seqlens_k: GpuTensor,
    scale: f32,
    total_kv_tokens: usize,
    num_heads: usize,
    head_dim: usize,
    block_size: usize,
    output_dtype: DType,
    alloc: &mut CachingAllocator,
    stream: CUstream,
) -> OwnedTensor {
    let out = alloc.alloc_tensor(&[total_kv_tokens, num_heads, head_dim], output_dtype);
    let batch_size = cu_seqlens_k.dim(0) - 1;
    let max_pages = block_table.dim(1);

    match output_dtype {
        DType::BF16 => dequant_gather_pages_bf16(
            cache.raw_ptr() as *const u8,
            block_table.as_ptr(),
            cu_seqlens_k.as_ptr(),
            scale,
            total_kv_tokens as i32,
            num_heads as i32,
            head_dim as i32,
            block_size as i32,
            max_pages as i32,
            batch_size as i32,
            out.as_gpu_tensor().as_mut_ptr(),
            stream,
        ),
        DType::F16 => dequant_gather_pages_f16(
            cache.raw_ptr() as *const u8,
            block_table.as_ptr(),
            cu_seqlens_k.as_ptr(),
            scale,
            total_kv_tokens as i32,
            num_heads as i32,
            head_dim as i32,
            block_size as i32,
            max_pages as i32,
            batch_size as i32,
            out.as_gpu_tensor().as_mut_ptr(),
            stream,
        ),
        _ => panic!(
            "dequant_gather_pages: output must be BF16 or F16, got {:?}",
            output_dtype
        ),
    }

    out
}

/// On-device BF16 paged gather (no dequant) for a SINGLE sequence — Gemma-4
/// GLOBAL attention (head_dim 512 > flash cap). Gathers `total_kv` tokens from
/// `cache` (`[num_blocks, block_size, num_heads, head_dim]` BF16) via
/// `block_table` (`[1, max_pages]` I32, on GPU) into `out`
/// (`[total_kv, num_heads, head_dim]` BF16). Replaces the host-driven
/// D2H + per-block D2D loop; no host readback → CUDA-graph-capture-safe.
///
/// # Safety
/// `cache` / `block_table` / `out` are live GPU tensors; `stream` is current.
#[allow(clippy::too_many_arguments)]
pub unsafe fn gather_pages_bf16_paged(
    cache: GpuTensor,
    block_table: GpuTensor,
    total_kv: usize,
    num_heads: usize,
    head_dim: usize,
    block_size: usize,
    out: GpuTensor,
    stream: CUstream,
) {
    gather_pages_bf16(
        cache.raw_ptr() as *const u16,
        block_table.as_ptr(),
        total_kv as i32,
        num_heads as i32,
        head_dim as i32,
        block_size as i32,
        out.raw_ptr() as *mut u16,
        stream,
    );
}

/// Like [`dequant_gather_pages`] but writes into a caller-supplied output buffer
/// instead of allocating. Used for CUDA graph capture where buffer addresses must
/// be fixed.
///
/// `grid_total_kv` is the grid launch size (may exceed actual `total_kv_tokens`
/// from `cu_seqlens_k` — the kernel bounds-checks via `seq_idx >= batch_size`).
#[allow(clippy::too_many_arguments)]
pub unsafe fn dequant_gather_pages_into(
    cache: GpuTensor,
    block_table: GpuTensor,
    cu_seqlens_k: GpuTensor,
    scale: f32,
    grid_total_kv: usize,
    num_heads: usize,
    head_dim: usize,
    block_size: usize,
    output_dtype: DType,
    output_ptr: *mut u8,
    stream: CUstream,
) {
    if grid_total_kv == 0 {
        return;
    }
    let batch_size = cu_seqlens_k.dim(0) - 1;
    let max_pages = block_table.dim(1);

    match output_dtype {
        DType::BF16 => dequant_gather_pages_bf16(
            cache.raw_ptr() as *const u8,
            block_table.as_ptr(),
            cu_seqlens_k.as_ptr(),
            scale,
            grid_total_kv as i32,
            num_heads as i32,
            head_dim as i32,
            block_size as i32,
            max_pages as i32,
            batch_size as i32,
            output_ptr as *mut u16,
            stream,
        ),
        DType::F16 => dequant_gather_pages_f16(
            cache.raw_ptr() as *const u8,
            block_table.as_ptr(),
            cu_seqlens_k.as_ptr(),
            scale,
            grid_total_kv as i32,
            num_heads as i32,
            head_dim as i32,
            block_size as i32,
            max_pages as i32,
            batch_size as i32,
            output_ptr as *mut u16,
            stream,
        ),
        _ => panic!(
            "dequant_gather_pages_into: output must be BF16 or F16, got {:?}",
            output_dtype
        ),
    }
}

/// Compute the abs-max of a BF16 tensor and write `scale = abs_max / divisor`.
///
/// * `tensor`: flat BF16 data on GPU
/// * `num_elements`: total number of BF16 elements
/// * `divisor`: FP8 E4M3 max (448.0) or user-provided constant
/// * `scale_out`: GPU f32 scalar — will contain the computed scale
pub unsafe fn compute_kv_scale(
    tensor: GpuTensor,
    num_elements: usize,
    divisor: f32,
    scale_out: *mut f32,
    stream: CUstream,
) {
    compute_abs_max_and_scale_bf16(
        tensor.as_ptr(),
        num_elements as i32,
        divisor,
        scale_out,
        stream,
    );
}

// ---------------------------------------------------------------------------
// FP8 Quantization Kernels
// ---------------------------------------------------------------------------

unsafe extern "C" {
    fn scaled_fp8_quant_dynamic_bf16(
        input: *const u16,
        output: *mut u8,
        scales: *mut f32,
        num_tokens: i32,
        hidden_dim: i32,
        stream: CUstream,
    );
    fn scaled_fp8_quant_dynamic_f16(
        input: *const u16,
        output: *mut u8,
        scales: *mut f32,
        num_tokens: i32,
        hidden_dim: i32,
        stream: CUstream,
    );
    fn scaled_fp8_quant_static_bf16(
        input: *const u16,
        output: *mut u8,
        scale: *const f32,
        num_elements: i32,
        stream: CUstream,
    );
    fn scaled_fp8_quant_static_f16(
        input: *const u16,
        output: *mut u8,
        scale: *const f32,
        num_elements: i32,
        stream: CUstream,
    );
    fn fp8_quantize_weight_bf16(
        weight: *const u16,
        output: *mut u8,
        scale_out: *mut f32,
        num_elements: i32,
        stream: CUstream,
    );
}

// ---------------------------------------------------------------------------
// FP8 Post-GEMM Scale + Re-quantization Kernels
// ---------------------------------------------------------------------------

unsafe extern "C" {
    fn fp8_row_scale_multiply_bf16(
        output: *mut u16,
        scales: *const f32,
        m: i32,
        n: i32,
        stream: CUstream,
    );
    fn fp8_row_scale_multiply_f16(
        output: *mut u16,
        scales: *const f32,
        m: i32,
        n: i32,
        stream: CUstream,
    );
    fn fp8_row_scale_multiply_f32(
        output: *mut f32,
        scales: *const f32,
        m: i32,
        n: i32,
        stream: CUstream,
    );
    fn fp8_requantize_rows(
        weight: *mut u8,
        k: i32,
        start_row: i32,
        num_rows: i32,
        scale_ratio: f32,
        stream: CUstream,
    );
}

/// Apply per-row activation scales to GEMM output (in-place).
///
/// `output[i, :] *= scales[i]`
///
/// Used after cublasLt FP8 GEMM with scalar scale to apply per-token
/// activation scales. Matches the behavior of CUTLASS cutlass_scaled_mm
/// which fuses per-row scale_a into the GEMM kernel.
pub unsafe fn fp8_post_scale_multiply(output: GpuTensor, scales: GpuTensor, stream: CUstream) {
    debug_assert_eq!(output.ndim(), 2);
    debug_assert_eq!(scales.ndim(), 1);
    debug_assert_eq!(output.dim(0), scales.dim(0));

    let m = output.dim(0);
    let n = output.dim(1);

    match output.dtype() {
        DType::BF16 => fp8_row_scale_multiply_bf16(
            output.as_mut_ptr(),
            scales.as_ptr() as *const f32,
            m as i32,
            n as i32,
            stream,
        ),
        DType::F16 => fp8_row_scale_multiply_f16(
            output.as_mut_ptr(),
            scales.as_ptr() as *const f32,
            m as i32,
            n as i32,
            stream,
        ),
        DType::F32 => fp8_row_scale_multiply_f32(
            output.as_mut_ptr() as *mut f32,
            scales.as_ptr() as *const f32,
            m as i32,
            n as i32,
            stream,
        ),
        other => panic!("fp8_post_scale_multiply: unsupported dtype {other}"),
    }
}

/// Re-quantize FP8 weight rows in-place with a new unified scale.
///
/// For fused module scale merging: each shard was quantized with its own
/// per-tensor scale. After concatenation, all shards must use the max scale.
/// This re-quantizes rows that had a smaller original scale:
///   new_fp8[i] = quantize_fp8(old_fp8[i] * old_scale / new_scale)
///
/// Matches Python's `requantize_with_max_scale()`.
pub unsafe fn fp8_requantize_weight_rows(
    weight: GpuTensor,
    k: usize,
    start_row: usize,
    num_rows: usize,
    old_scale: f32,
    new_scale: f32,
    stream: CUstream,
) {
    debug_assert_eq!(weight.dtype(), DType::Fp8E4m3);
    if num_rows == 0 || (old_scale - new_scale).abs() < 1e-12 {
        return; // Same scale, no re-quantization needed
    }
    let scale_ratio = old_scale / new_scale;
    fp8_requantize_rows(
        weight.as_mut_ptr(),
        k as i32,
        start_row as i32,
        num_rows as i32,
        scale_ratio,
        stream,
    );
}

/// Dynamic per-token FP8 quantization: BF16/F16 → FP8 E4M3 + per-token scales.
///
/// * `input`: `[num_tokens, hidden_dim]` BF16 or F16 on GPU
/// * Returns: `(output_fp8, scales)` — FP8 `[num_tokens, hidden_dim]` + f32 `[num_tokens]`
///
/// Each token row gets its own scale derived from absmax / 448.0 (FP8_E4M3_MAX).
/// Matches Python vLLM's `ops.scaled_fp8_quant(input, scale=None)`.
pub unsafe fn scaled_fp8_quant_dynamic(
    input: GpuTensor,
    alloc: &mut crate::alloc::CachingAllocator,
    stream: CUstream,
) -> (crate::alloc::OwnedTensor, crate::alloc::OwnedTensor) {
    debug_assert_eq!(input.ndim(), 2);
    let num_tokens = input.dim(0);
    let hidden_dim = input.dim(1);

    let output = alloc.alloc_tensor(&[num_tokens, hidden_dim], DType::Fp8E4m3);
    let scales = alloc.alloc_tensor(&[num_tokens], DType::F32);

    match input.dtype() {
        DType::BF16 => scaled_fp8_quant_dynamic_bf16(
            input.as_ptr(),
            output.as_gpu_tensor().as_mut_ptr(),
            scales.as_gpu_tensor().as_mut_ptr() as *mut f32,
            num_tokens as i32,
            hidden_dim as i32,
            stream,
        ),
        DType::F16 => scaled_fp8_quant_dynamic_f16(
            input.as_ptr(),
            output.as_gpu_tensor().as_mut_ptr(),
            scales.as_gpu_tensor().as_mut_ptr() as *mut f32,
            num_tokens as i32,
            hidden_dim as i32,
            stream,
        ),
        dt => panic!("scaled_fp8_quant_dynamic: unsupported input dtype {dt}"),
    }

    (output, scales)
}

/// Static FP8 quantization: BF16/F16 → FP8 E4M3 with pre-calibrated scale.
///
/// * `input`: `[num_tokens, hidden_dim]` BF16 or F16 on GPU
/// * `scale`: GPU f32 scalar pointer (pre-calibrated input_scale)
/// * Returns: FP8 `[num_tokens, hidden_dim]`
///
/// Matches Python vLLM's `ops.scaled_fp8_quant(input, scale=input_scale)`.
pub unsafe fn scaled_fp8_quant_static(
    input: GpuTensor,
    scale: *const f32,
    alloc: &mut crate::alloc::CachingAllocator,
    stream: CUstream,
) -> crate::alloc::OwnedTensor {
    debug_assert_eq!(input.ndim(), 2);
    let num_elements = input.dim(0) * input.dim(1);

    let output = alloc.alloc_tensor(&[input.dim(0), input.dim(1)], DType::Fp8E4m3);

    match input.dtype() {
        DType::BF16 => scaled_fp8_quant_static_bf16(
            input.as_ptr(),
            output.as_gpu_tensor().as_mut_ptr(),
            scale,
            num_elements as i32,
            stream,
        ),
        DType::F16 => scaled_fp8_quant_static_f16(
            input.as_ptr(),
            output.as_gpu_tensor().as_mut_ptr(),
            scale,
            num_elements as i32,
            stream,
        ),
        dt => panic!("scaled_fp8_quant_static: unsupported input dtype {dt}"),
    }

    output
}

/// Online weight FP8 quantization: BF16 weight → FP8 E4M3 + per-tensor scale.
///
/// * `weight`: `[N, K]` BF16 on GPU
/// * `scale_out`: GPU f32 scalar (will hold computed scale = absmax / 448.0)
/// * Returns: FP8 `[N, K]`
///
/// Used for online FP8 quantization when checkpoint is BF16 but model
/// wants FP8 weight format.
pub unsafe fn fp8_quantize_weight(
    weight: GpuTensor,
    scale_out: *mut f32,
    alloc: &mut crate::alloc::CachingAllocator,
    stream: CUstream,
) -> crate::alloc::OwnedTensor {
    debug_assert_eq!(weight.ndim(), 2);
    debug_assert_eq!(
        weight.dtype(),
        DType::BF16,
        "online FP8 weight quant requires BF16 input"
    );
    let num_elements = weight.dim(0) * weight.dim(1);

    let output = alloc.alloc_tensor(&[weight.dim(0), weight.dim(1)], DType::Fp8E4m3);

    fp8_quantize_weight_bf16(
        weight.as_ptr(),
        output.as_gpu_tensor().as_mut_ptr(),
        scale_out,
        num_elements as i32,
        stream,
    );

    output
}

/// Raw FFI wrapper for `fp8_quantize_weight_bf16` — used during weight loading
/// when we don't have a `CachingAllocator` (pre-allocated buffers passed in).
pub unsafe fn fp8_quantize_weight_bf16_raw(
    weight: *const u16,
    output: *mut u8,
    scale_out: *mut f32,
    num_elements: i32,
    stream: CUstream,
) {
    fp8_quantize_weight_bf16(weight, output, scale_out, num_elements, stream);
}

// ---------------------------------------------------------------------------
// FP8 Block Dequantization Kernels
// ---------------------------------------------------------------------------

unsafe extern "C" {
    fn fp8_block_dequant_bf16(
        weight: *const u8,
        scale_inv: *const f32,
        output: *mut u16,
        n: i32,
        k: i32,
        block_n: i32,
        block_k: i32,
        stream: CUstream,
    );
    fn fp8_block_dequant_f16(
        weight: *const u8,
        scale_inv: *const f32,
        output: *mut u16,
        n: i32,
        k: i32,
        block_n: i32,
        block_k: i32,
        stream: CUstream,
    );
}

/// Dequantize FP8 block-quantized weight to BF16/F16.
///
/// * `weight`: `[N, K]` FP8 E4M3
/// * `scale_inv`: `[ceil(N/block_n), ceil(K/block_k)]` f32
/// * `block_size`: `[block_n, block_k]`
/// * Returns: `[N, K]` in `output_dtype`
pub unsafe fn fp8_block_dequant(
    weight: GpuTensor,
    scale_inv: GpuTensor,
    block_size: [usize; 2],
    output_dtype: DType,
    alloc: &mut crate::alloc::CachingAllocator,
    stream: CUstream,
) -> crate::alloc::OwnedTensor {
    debug_assert_eq!(weight.ndim(), 2);
    debug_assert_eq!(weight.dtype(), DType::Fp8E4m3);
    let n = weight.dim(0);
    let k = weight.dim(1);

    let out = alloc.alloc_tensor(&[n, k], output_dtype);

    match output_dtype {
        DType::BF16 => fp8_block_dequant_bf16(
            weight.as_ptr(),
            scale_inv.as_ptr(),
            out.as_gpu_tensor().as_mut_ptr(),
            n as i32,
            k as i32,
            block_size[0] as i32,
            block_size[1] as i32,
            stream,
        ),
        DType::F16 => fp8_block_dequant_f16(
            weight.as_ptr(),
            scale_inv.as_ptr(),
            out.as_gpu_tensor().as_mut_ptr(),
            n as i32,
            k as i32,
            block_size[0] as i32,
            block_size[1] as i32,
            stream,
        ),
        dt => panic!("fp8_block_dequant: output must be BF16 or F16, got {dt}"),
    }

    out
}

// ---------------------------------------------------------------------------
// MLX-affine int4 / int8 dequant (Gemma-4 MoE: w = scale*q + bias, bf16 s/b)
// ---------------------------------------------------------------------------

unsafe extern "C" {
    fn affine_dequant_b4_bf16(
        w: *const u8,
        scales: *const u16,
        biases: *const u16,
        out: *mut u16,
        rows: i32,
        cols: i32,
        group_size: i32,
        stream: CUstream,
    );
    fn affine_dequant_b8_bf16(
        w: *const u8,
        scales: *const u16,
        biases: *const u16,
        out: *mut u16,
        rows: i32,
        cols: i32,
        group_size: i32,
        stream: CUstream,
    );
}

/// Dequantize an MLX-affine int4 / int8 weight to BF16 in place into `out`.
///
/// * `w`: packed weight bytes. For `bits=4` the buffer is `[rows, cols/2]`
///   bytes (two nibbles per byte, low-nibble-even); for `bits=8` it is
///   `[rows, cols]` bytes. The on-disk tensor is typed `DType::U32` — pass
///   its byte pointer.
/// * `scales` / `biases`: BF16, `[rows, cols/group_size]`.
/// * `out`: BF16 `[rows, cols]` destination (caller-owned).
/// * Formula: `out = scale * q + bias` (no zero-point, no interleave).
///
/// # Safety
/// All pointers must be valid GPU memory for the stated shapes; `out` must be
/// sized `rows * cols` BF16 elements. Valid CUDA context + stream.
#[allow(clippy::too_many_arguments)]
pub unsafe fn affine_dequant_into(
    w: *const u8,
    scales: *const u16,
    biases: *const u16,
    out: *mut u16,
    rows: usize,
    cols: usize,
    bits: u32,
    group_size: usize,
    stream: CUstream,
) {
    match bits {
        4 => affine_dequant_b4_bf16(
            w,
            scales,
            biases,
            out,
            rows as i32,
            cols as i32,
            group_size as i32,
            stream,
        ),
        8 => affine_dequant_b8_bf16(
            w,
            scales,
            biases,
            out,
            rows as i32,
            cols as i32,
            group_size as i32,
            stream,
        ),
        b => panic!("affine_dequant_into: only bits 4/8 supported, got {b}"),
    }
}

// ---------------------------------------------------------------------------
// CUTLASS Scaled Matmul (Fused FP8 GEMM with per-row scale epilogue)
// ---------------------------------------------------------------------------

unsafe extern "C" {
    /// CUTLASS 2.x FP8 scaled matmul on SM89 (Ada Lovelace).
    /// Fuses per-row activation scale and per-tensor weight scale into the GEMM
    /// epilogue — single kernel launch, matching Python vLLM's cutlass_scaled_mm.
    fn cutlass_scaled_mm_sm89(
        c: *mut u8,           // [M, N] output (BF16 or F16)
        a: *const u8,         // [M, K] FP8 E4M3 row-major
        b: *const u8,         // [N, K] FP8 E4M3 row-major (= [K,N] col-major)
        a_scales: *const f32, // [M] per-token or [1] per-tensor
        a_scales_numel: i32,  // M or 1
        b_scales: *const f32, // [N] per-channel or [1] per-tensor
        b_scales_numel: i32,  // N or 1
        m: i32,
        n: i32,
        k: i32,
        out_dtype: i32, // 0 = BF16, 1 = F16
        stream: CUstream,
    );

    /// CUTLASS 2.x FP8 scaled matmul on SM89 with bias.
    fn cutlass_scaled_mm_bias_sm89(
        c: *mut u8,
        a: *const u8,
        b: *const u8,
        a_scales: *const f32,
        a_scales_numel: i32,
        b_scales: *const f32,
        b_scales_numel: i32,
        bias: *const u8, // [N] same dtype as output
        m: i32,
        n: i32,
        k: i32,
        out_dtype: i32,
        stream: CUstream,
    );

    /// CUTLASS 3.x FP8 scaled matmul on SM90 (Hopper). Same ABI as the SM89
    /// entry; selects a Hopper TMA warp-specialized tile config by (M, N).
    /// Matches Python vLLM's `cutlass_scaled_mm_sm90` (the C3X path).
    fn cutlass_scaled_mm_sm90(
        c: *mut u8,
        a: *const u8,
        b: *const u8,
        a_scales: *const f32,
        a_scales_numel: i32,
        b_scales: *const f32,
        b_scales_numel: i32,
        m: i32,
        n: i32,
        k: i32,
        out_dtype: i32,
        stream: CUstream,
    );

    /// CUTLASS 3.x FP8 scaled matmul on SM90 (Hopper) with bias.
    fn cutlass_scaled_mm_bias_sm90(
        c: *mut u8,
        a: *const u8,
        b: *const u8,
        a_scales: *const f32,
        a_scales_numel: i32,
        b_scales: *const f32,
        b_scales_numel: i32,
        bias: *const u8,
        m: i32,
        n: i32,
        k: i32,
        out_dtype: i32,
        stream: CUstream,
    );
}

/// Cached device compute capability (major*10 + minor, e.g. 89 / 90).
///
/// Routes FP8 `cutlass_scaled_mm` to the correct CUTLASS kernel: the SM89 (Ada)
/// C2X kernel and the SM90 (Hopper) C3X kernel are arch-guarded at the device
/// level and trap if launched on the wrong SM (this is the bug that produced
/// "This kernel only supports sm[89, 90)." on H100). Compute capability is fixed
/// for the process, so query the current device once.
fn device_sm_version() -> u32 {
    use std::sync::OnceLock;
    static SM_VERSION: OnceLock<u32> = OnceLock::new();
    *SM_VERSION.get_or_init(|| unsafe {
        let dev = crate::driver::current_device()
            .expect("cutlass_scaled_mm: no current CUDA device/context");
        crate::driver::device_get_sm_version(dev)
            .expect("cutlass_scaled_mm: failed to query device compute capability")
    })
}

/// Fused CUTLASS FP8 GEMM with per-row activation scales.
///
/// `output[M, N] = diag(a_scales) @ (A_fp8 @ B_fp8^T) * b_scale`
///
/// Single kernel launch — fuses the scale multiply into the GEMM epilogue.
/// This matches Python vLLM's `cutlass_scaled_mm` exactly.
///
/// * `a`: `[M, K]` FP8 E4M3 activations (row-major)
/// * `b`: `[N, K]` FP8 E4M3 weights (row-major, treated as col-major [K,N])
/// * `a_scales`: `[M]` f32 per-token scales, or `[1]` for per-tensor
/// * `b_scales`: `[1]` f32 per-tensor weight scale (or `[N]` per-channel)
/// * `output_dtype`: BF16 or F16
///
/// Returns: `[M, N]` tensor in `output_dtype`
pub unsafe fn cutlass_scaled_mm(
    a: GpuTensor,
    b: GpuTensor,
    a_scales: GpuTensor,
    b_scales: GpuTensor,
    output_dtype: DType,
    alloc: &mut crate::alloc::CachingAllocator,
    stream: CUstream,
) -> crate::alloc::OwnedTensor {
    debug_assert_eq!(a.ndim(), 2);
    debug_assert_eq!(b.ndim(), 2);
    debug_assert_eq!(a.dtype(), DType::Fp8E4m3);
    debug_assert_eq!(b.dtype(), DType::Fp8E4m3);
    debug_assert_eq!(a.dim(1), b.dim(1), "A [M,K] and B [N,K] must share K dim");

    let m = a.dim(0);
    let k = a.dim(1);
    let n = b.dim(0);

    let output = alloc.alloc_tensor(&[m, n], output_dtype);
    let out_dtype_code = match output_dtype {
        DType::BF16 => 0,
        DType::F16 => 1,
        dt => panic!("cutlass_scaled_mm: output must be BF16 or F16, got {dt}"),
    };

    // Route by device compute capability. The SM89 (Ada) and SM90 (Hopper)
    // kernels are each arch-guarded at the device level; calling the wrong one
    // traps with "This kernel only supports sm[..]". Hopper (90..100) needs the
    // CUTLASS 3.x C3X kernel; Ada (89) uses the C2X kernel.
    let sm = device_sm_version();
    let c = output.as_gpu_tensor().as_mut_ptr();
    if (90..100).contains(&sm) {
        cutlass_scaled_mm_sm90(
            c,
            a.as_ptr(),
            b.as_ptr(),
            a_scales.as_ptr() as *const f32,
            a_scales.numel() as i32,
            b_scales.as_ptr() as *const f32,
            b_scales.numel() as i32,
            m as i32,
            n as i32,
            k as i32,
            out_dtype_code,
            stream,
        );
    } else if sm == 89 {
        cutlass_scaled_mm_sm89(
            c,
            a.as_ptr(),
            b.as_ptr(),
            a_scales.as_ptr() as *const f32,
            a_scales.numel() as i32,
            b_scales.as_ptr() as *const f32,
            b_scales.numel() as i32,
            m as i32,
            n as i32,
            k as i32,
            out_dtype_code,
            stream,
        );
    } else {
        panic!(
            "cutlass_scaled_mm: FP8 scaled_mm requires SM89 (Ada) or SM90 (Hopper), \
             got SM{sm}. No compiled CUTLASS FP8 kernel for this compute capability."
        );
    }

    output
}

/// Fused CUTLASS FP8 GEMM with per-row scales and bias.
pub unsafe fn cutlass_scaled_mm_with_bias(
    a: GpuTensor,
    b: GpuTensor,
    a_scales: GpuTensor,
    b_scales: GpuTensor,
    bias: GpuTensor,
    output_dtype: DType,
    alloc: &mut crate::alloc::CachingAllocator,
    stream: CUstream,
) -> crate::alloc::OwnedTensor {
    debug_assert_eq!(a.ndim(), 2);
    debug_assert_eq!(b.ndim(), 2);
    debug_assert_eq!(a.dtype(), DType::Fp8E4m3);
    debug_assert_eq!(b.dtype(), DType::Fp8E4m3);
    debug_assert_eq!(a.dim(1), b.dim(1));

    let m = a.dim(0);
    let k = a.dim(1);
    let n = b.dim(0);

    let output = alloc.alloc_tensor(&[m, n], output_dtype);
    let out_dtype_code = match output_dtype {
        DType::BF16 => 0,
        DType::F16 => 1,
        dt => panic!("cutlass_scaled_mm_with_bias: output must be BF16 or F16, got {dt}"),
    };

    let sm = device_sm_version();
    let c = output.as_gpu_tensor().as_mut_ptr();
    if (90..100).contains(&sm) {
        cutlass_scaled_mm_bias_sm90(
            c,
            a.as_ptr(),
            b.as_ptr(),
            a_scales.as_ptr() as *const f32,
            a_scales.numel() as i32,
            b_scales.as_ptr() as *const f32,
            b_scales.numel() as i32,
            bias.as_ptr(),
            m as i32,
            n as i32,
            k as i32,
            out_dtype_code,
            stream,
        );
    } else if sm == 89 {
        cutlass_scaled_mm_bias_sm89(
            c,
            a.as_ptr(),
            b.as_ptr(),
            a_scales.as_ptr() as *const f32,
            a_scales.numel() as i32,
            b_scales.as_ptr() as *const f32,
            b_scales.numel() as i32,
            bias.as_ptr(),
            m as i32,
            n as i32,
            k as i32,
            out_dtype_code,
            stream,
        );
    } else {
        panic!(
            "cutlass_scaled_mm_with_bias: FP8 scaled_mm requires SM89 (Ada) or SM90 \
             (Hopper), got SM{sm}. No compiled CUTLASS FP8 kernel for this capability."
        );
    }

    output
}

// ---------------------------------------------------------------------------
// FlashAttention-2 Paged (raw FFI)
// ---------------------------------------------------------------------------

unsafe extern "C" {
    /// FFI to mha_varlen_fwd in flash-attn-shim/ffi_shim.cu.
    /// Mirrors upstream vllm-project/flash-attention mha_varlen_fwd() logic.
    /// Handles both paged and non-paged KV, forces splitkv kernel for paged.
    fn mha_varlen_fwd(
        q_ptr: *mut c_void,
        k_ptr: *mut c_void,
        v_ptr: *mut c_void,
        out_ptr: *mut c_void,
        softmax_lse_ptr: *mut c_void,

        cu_seqlens_q: *const i32,
        cu_seqlens_k: *const i32,
        seqused_k: *const i32,

        block_table: *const i32,
        block_table_batch_stride: i32,

        batch_size: i32,
        max_seqlen_q: i32,
        max_seqlen_k: i32,
        num_heads: i32,
        num_heads_k: i32,
        head_size: i32,
        page_block_size: i32,

        q_row_stride: i64,
        q_head_stride: i64,
        k_batch_stride: i64,
        k_row_stride: i64,
        k_head_stride: i64,
        o_row_stride: i64,
        o_head_stride: i64,

        softmax_scale: f32,
        is_causal: i32,
        window_size_left: i32,
        window_size_right: i32,
        softcap: f32,
        is_bf16: i32,
        num_splits: i32,

        softmax_lse_accum_ptr: *mut c_void,
        out_accum_ptr: *mut c_void,
        seqlenq_ngroups_swapped: i32,
        total_q: i32,

        // Spans: fused RoPE for cached K reads.
        rotary_cos_ptr: *const c_void,
        rotary_sin_ptr: *const c_void,
        rotary_dim: i32,
        rotate_cached_k: i32,
        is_rotary_interleaved: i32,

        // Per-block rotation flags (nullptr = use rotate_cached_k for all).
        block_unrotated_flags: *const u8,

        stream: CUstream,
    );
}

unsafe extern "C" {
    /// Transpose Q from [B, H, D] to [B*ngroups, Hk, D] for seqlenq_ngroups_swapped.
    fn ngroups_transpose_q(
        src: *const c_void,
        dst: *mut c_void,
        batch_size: i32,
        num_heads: i32,
        num_heads_k: i32,
        ngroups: i32,
        head_dim: i32,
        stream: CUstream,
    );
    /// Reverse transpose: [B*ngroups, Hk, D] back to [B, H, D].
    fn ngroups_untranspose_o(
        src: *const c_void,
        dst: *mut c_void,
        batch_size: i32,
        num_heads: i32,
        num_heads_k: i32,
        ngroups: i32,
        head_dim: i32,
        stream: CUstream,
    );
}

/// Exact port of num_splits_heuristic from flash_api.cpp lines 262-296.
/// Two-pass: find max efficiency, then find smallest split ≥ 85% of max.
fn num_splits_heuristic(
    batch_nheads_mblocks: usize,
    num_sm: usize,
    num_n_blocks: usize,
    max_splits: usize,
) -> usize {
    // Python line 264: if batch_nheads_mblocks >= 0.8f * num_SMs, return 1
    if batch_nheads_mblocks as f32 >= 0.8 * num_sm as f32 {
        return 1;
    }

    let max_splits = max_splits.min(num_sm).min(num_n_blocks);

    // Python's is_split_eligible: ceildiv(n, s) != ceildiv(n, s-1)
    let is_split_eligible =
        |s: usize| -> bool { s == 1 || num_n_blocks.div_ceil(s) != num_n_blocks.div_ceil(s - 1) };

    // Pass 1: compute efficiencies, find max
    let mut max_efficiency = 0.0_f32;
    let mut efficiencies = Vec::with_capacity(max_splits);
    for s in 1..=max_splits {
        if !is_split_eligible(s) {
            efficiencies.push(0.0_f32);
        } else {
            let n_waves = (batch_nheads_mblocks * s) as f32 / num_sm as f32;
            let eff = n_waves / n_waves.ceil();
            if eff > max_efficiency {
                max_efficiency = eff;
            }
            efficiencies.push(eff);
        }
    }

    // Pass 2: find smallest split ≥ 85% of max efficiency
    for s in 1..=max_splits {
        if !is_split_eligible(s) {
            continue;
        }
        if efficiencies[s - 1] >= 0.85 * max_efficiency {
            return s;
        }
    }
    1
}

fn round_multiple(x: usize, m: usize) -> usize {
    x.div_ceil(m) * m
}

// Non-paged FlashAttention-2 forward pass (contiguous K/V).
//
// Used for **prefill** where K/V come directly from the current forward pass
// (not from the paged block cache). Matches Python vLLM's prefill attention.
//
// * `q`: `[total_q_tokens, num_heads, head_dim]`
// * `k`: `[total_k_tokens, num_kv_heads, head_dim]`
// * `v`: `[total_k_tokens, num_kv_heads, head_dim]`
// * `cu_seqlens_q/k`: `[batch_size + 1]` cumulative sequence lengths
//
// Returns: `[total_q_tokens, num_heads, head_dim]` output tensor from arena.
#[allow(clippy::too_many_arguments)]
unsafe extern "C" {
    fn sdpa_naive_bf16(
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        out: *mut c_void,
        num_q: c_int,
        num_k: c_int,
        num_q_heads: c_int,
        num_kv_heads: c_int,
        head_dim: c_int,
        scale: f32,
        softcap: f32,
        window: c_int,
        q_pos_base: c_int,
        stream: CUstream,
    );

    // Split-KV flash-decode (fast path for small num_q). Scratch buffers are
    // f32 and caller-allocated: partial_out [num_q*num_q_heads, num_splits,
    // head_dim], partial_m / partial_l [num_q*num_q_heads, num_splits].
    #[allow(clippy::too_many_arguments)]
    fn sdpa_flash_decode_bf16(
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        out: *mut c_void,
        partial_out: *mut f32,
        partial_m: *mut f32,
        partial_l: *mut f32,
        num_q: c_int,
        num_k: c_int,
        num_q_heads: c_int,
        num_kv_heads: c_int,
        head_dim: c_int,
        scale: f32,
        softcap: f32,
        window: c_int,
        q_pos_base: c_int,
        num_splits: c_int,
        stream: CUstream,
    );
}

/// Naive (non-flash) SDPA for ARBITRARY head_dim — Gemma-4 GLOBAL attention
/// (head_dim 512 > the flash-attn cap of 256). Contiguous Q/K/V; causal with
/// optional sliding `window` (<0 = none) and `softcap` (0 = none). GQA via
/// `num_q_heads / num_kv_heads`. `q_pos_base` = absolute position of query
/// token 0 (0 for a fresh prefill). Returns `[num_q, num_q_heads, head_dim]`.
///
/// # Safety
/// q `[num_q, num_q_heads, head_dim]`, k/v `[num_k, num_kv_heads, head_dim]`,
/// all BF16 contiguous GPU tensors; head_dim must be a power of two ≤ 1024.
#[allow(clippy::too_many_arguments)]
pub unsafe fn sdpa_naive(
    q: GpuTensor,
    k: GpuTensor,
    v: GpuTensor,
    scale: f32,
    softcap: f32,
    window: i32,
    q_pos_base: i32,
    alloc: &mut CachingAllocator,
    stream: CUstream,
) -> OwnedTensor {
    let num_q = q.dim(0);
    let num_q_heads = q.dim(1);
    let head_dim = q.dim(2);
    let num_k = k.dim(0);
    let num_kv_heads = k.dim(1);
    assert!(
        matches!(q.dtype(), DType::BF16),
        "sdpa_naive: only BF16 supported (got {:?})",
        q.dtype()
    );
    assert!(
        head_dim.is_power_of_two() && head_dim <= 1024,
        "sdpa_naive: head_dim={head_dim} must be a power of two ≤ 1024"
    );
    let out = alloc.alloc_tensor(&[num_q, num_q_heads, head_dim], q.dtype());

    // Fast path: small query count (decode, or a short continuation) — the
    // naive kernel launches only `num_q * num_q_heads` blocks and streams all
    // keys serially, which dominates decode wall-clock. Split the KV dimension
    // across `num_splits` blocks per (query, head) and combine the partials.
    // Large `num_q` (prefill) already saturates the grid via the (qi, qh) axes,
    // so it keeps the single-pass kernel below.
    if num_q <= SDPA_FLASH_DECODE_MAX_Q && num_k > 0 && sdpa_flash_decode_enabled() {
        // ~256 keys per split, clamped so the grid stays well-occupied without
        // blowing up the combine cost or scratch footprint.
        let num_splits = num_k.div_ceil(256).clamp(1, 64);
        let nqh = num_q * num_q_heads;
        let partial_out = alloc.alloc_tensor(&[nqh * num_splits, head_dim], DType::F32);
        let partial_m = alloc.alloc_tensor(&[nqh, num_splits], DType::F32);
        let partial_l = alloc.alloc_tensor(&[nqh, num_splits], DType::F32);
        sdpa_flash_decode_bf16(
            q.raw_ptr() as *const c_void,
            k.raw_ptr() as *const c_void,
            v.raw_ptr() as *const c_void,
            out.raw_ptr() as *mut c_void,
            partial_out.raw_ptr() as *mut f32,
            partial_m.raw_ptr() as *mut f32,
            partial_l.raw_ptr() as *mut f32,
            num_q as c_int,
            num_k as c_int,
            num_q_heads as c_int,
            num_kv_heads as c_int,
            head_dim as c_int,
            scale,
            softcap,
            window,
            q_pos_base,
            num_splits as c_int,
            stream,
        );
        return out;
    }

    sdpa_naive_bf16(
        q.raw_ptr() as *const c_void,
        k.raw_ptr() as *const c_void,
        v.raw_ptr() as *const c_void,
        out.raw_ptr() as *mut c_void,
        num_q as c_int,
        num_k as c_int,
        num_q_heads as c_int,
        num_kv_heads as c_int,
        head_dim as c_int,
        scale,
        softcap,
        window,
        q_pos_base,
        stream,
    );
    out
}

/// Query-count threshold below which [`sdpa_naive`] routes to the split-KV
/// flash-decode kernels instead of the single-pass naive kernel. Decode is
/// `num_q == 1`; a small band covers short prefill continuations too.
const SDPA_FLASH_DECODE_MAX_Q: usize = 8;

/// `SDPA_FLASH_DECODE=0` forces the single-pass kernel even at small `num_q`
/// (diagnostic A/B against the split-KV flash decode; default on). Read once.
fn sdpa_flash_decode_enabled() -> bool {
    static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *E.get_or_init(|| {
        std::env::var("SDPA_FLASH_DECODE")
            .map(|v| v != "0")
            .unwrap_or(true)
    })
}

pub unsafe fn flash_attn_contiguous(
    q: GpuTensor,
    k: GpuTensor,
    v: GpuTensor,
    cu_seqlens_q: GpuTensor,
    cu_seqlens_k: GpuTensor,
    max_seqlen_q: usize,
    max_seqlen_k: usize,
    softmax_scale: f32,
    is_causal: bool,
    softcap: f32,
    window_size_left: i32,
    alloc: &mut CachingAllocator,
    stream: CUstream,
    cos_sin_cache_ptr: *const u8,
    rotary_dim: usize,
    is_rotary_interleaved: bool,
) -> OwnedTensor {
    let total_q = q.dim(0);
    let num_heads = q.dim(1);
    let head_dim = q.dim(2);
    let num_kv_heads = k.dim(1);

    // GUARD: flash-attn instantiates head_dim ≤ 256 only (HEADDIM_SWITCH in
    // static_switch.h); head_dim > 256 falls through → illegal memory access.
    // Gemma-4 global attention uses head_dim 512 (no cuda flash kernel yet —
    // gemma4 was metal-only). Hard-stop with a clear message, not UB.
    assert!(
        head_dim <= 256,
        "flash_attn_contiguous: head_dim={head_dim} exceeds the flash-attn cap of 256. \
         Gemma-4 global attention head_dim=512 needs a non-flash cuda attention path \
         (flashinfer / split-KV / naive) — unimplemented."
    );

    let batch_size = cu_seqlens_q.dim(0) - 1;

    let _head_size_rounded = round_multiple(head_dim, 32);
    let _seqlen_q_rounded = round_multiple(max_seqlen_q, 128);
    let _seqlen_k_rounded = round_multiple(max_seqlen_k, 128);

    // Allocate output and softmax_lse from arena.
    let out = alloc.alloc_tensor(&[total_q, num_heads, head_dim], q.dtype());
    let softmax_lse = alloc.alloc_tensor(&[num_heads * total_q], DType::F32);

    // Use mha_varlen_fwd with null block_table for contiguous (non-paged) path
    mha_varlen_fwd(
        q.raw_ptr() as *mut c_void,
        k.raw_ptr() as *mut c_void,
        v.raw_ptr() as *mut c_void,
        out.raw_ptr() as *mut c_void,
        softmax_lse.raw_ptr() as *mut c_void,
        cu_seqlens_q.as_ptr::<i32>() as *const i32,
        cu_seqlens_k.as_ptr::<i32>() as *const i32,
        std::ptr::null(), // seqused_k = null for contiguous
        std::ptr::null(), // block_table = null for contiguous
        0,                // block_table_batch_stride
        batch_size as i32,
        max_seqlen_q as i32,
        max_seqlen_k as i32,
        num_heads as i32,
        num_kv_heads as i32,
        head_dim as i32,
        0,                                // page_block_size (unused for contiguous)
        (num_heads * head_dim) as i64,    // q_row_stride
        head_dim as i64,                  // q_head_stride
        0,                                // k_batch_stride
        (num_kv_heads * head_dim) as i64, // k_row_stride
        head_dim as i64,                  // k_head_stride
        (num_heads * head_dim) as i64,    // o_row_stride
        head_dim as i64,                  // o_head_stride
        softmax_scale,
        if is_causal { 1 } else { 0 },
        window_size_left,
        if is_causal { 0 } else { -1 },
        softcap,
        if q.dtype() == DType::BF16 { 1 } else { 0 },
        1, // num_splits
        std::ptr::null_mut(),
        std::ptr::null_mut(),
        0, // seqlenq_ngroups_swapped = false
        total_q as i32,
        cos_sin_cache_ptr as *const c_void, // rotary_cos_ptr (spans)
        std::ptr::null(),                   // rotary_sin_ptr (spans)
        rotary_dim as i32,                  // rotary_dim (spans)
        if cos_sin_cache_ptr.is_null() { 0 } else { 1 }, // rotate_cached_k (spans)
        if is_rotary_interleaved { 1 } else { 0 }, // is_rotary_interleaved
        std::ptr::null(), // block_unrotated_flags (contiguous path, no paged blocks)
        stream,
    );

    out
}

/// Paged FlashAttention-2 forward pass.
///
/// Uses upstream vllm-project/flash-attention kernel via `mha_varlen_fwd` shim.
/// For paged KV, forces the splitkv kernel which correctly handles block_table.
///
/// * `q`: `[total_q_tokens, num_heads, head_dim]` (contiguous)
/// * `k_cache`: `[num_blocks, block_size, num_kv_heads, head_dim]`
/// * `v_cache`: same layout
/// * `cu_seqlens_q`: `[batch_size + 1]` (I32 on GPU, cumulative seq lengths for Q)
/// * `seqused_k`: `[batch_size]` (I32 on GPU, actual K length per sequence)
/// * `block_table`: `[batch_size, max_blocks_per_seq]` (I32 on GPU)
#[allow(clippy::too_many_arguments)]
pub unsafe fn flash_attn_paged(
    q: GpuTensor,
    k_cache: GpuTensor,
    v_cache: GpuTensor,
    cu_seqlens_q: GpuTensor,
    seqused_k: GpuTensor,
    block_table: GpuTensor,
    max_seqlen_q: usize,
    max_seqlen_k: usize,
    softmax_scale: f32,
    is_causal: bool,
    block_size: usize,
    num_sm: i32,
    alloc: &mut CachingAllocator,
    _stream: CUstream,
) -> OwnedTensor {
    flash_attn_paged_ext(
        q,
        k_cache,
        v_cache,
        cu_seqlens_q,
        seqused_k,
        block_table,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        is_causal,
        0.0,
        -1,
        block_size,
        num_sm,
        alloc,
        _stream,
        std::ptr::null(),
        0,
        false,
        std::ptr::null(),
    )
}

/// Paged FlashAttention-2 with softcap and sliding window support.
///
/// Calls upstream mha_varlen_fwd which forces the splitkv kernel for paged KV.
/// Matches Python vLLM's flash_attn_varlen_func calling convention exactly.
#[allow(clippy::too_many_arguments)]
/// `cos_sin_cache_ptr`: when non-null, pointer to `[max_pos, rotary_dim]` cos/sin
///   cache. Used with `rotary_dim > 0` to apply fused RoPE to cached K during
///   attention (for spans / relocatable KV blocks).
pub unsafe fn flash_attn_paged_ext(
    q: GpuTensor,
    k_cache: GpuTensor,
    v_cache: GpuTensor,
    cu_seqlens_q: GpuTensor,
    seqused_k: GpuTensor,
    block_table: GpuTensor,
    max_seqlen_q: usize,
    max_seqlen_k: usize,
    softmax_scale: f32,
    is_causal: bool,
    softcap: f32,
    window_size_left: i32,
    block_size: usize,
    num_sm: i32,
    alloc: &mut CachingAllocator,
    _stream: CUstream,
    cos_sin_cache_ptr: *const u8,
    rotary_dim: usize,
    is_rotary_interleaved: bool,
    block_unrotated_flags: *const u8,
) -> OwnedTensor {
    let total_q = q.dim(0);
    let num_heads_orig = q.dim(1);
    let head_dim = q.dim(2);
    let num_kv_heads = k_cache.dim(2);

    // GUARD: vllm-flash-attn only instantiates head_dim ∈ {32,64,96,128,192,256}
    // (see HEADDIM_SWITCH in static_switch.h). For head_dim > 256 the switch
    // falls through and the kernel launches with a stale/undefined kHeadDim →
    // ILLEGAL MEMORY ACCESS. Gemma-4 GLOBAL attention uses head_dim 512, which
    // has no cuda flash kernel yet (gemma4 was metal-only until now). Fail with
    // a clear message instead of corrupting memory.
    assert!(
        head_dim <= 256,
        "flash_attn_paged_ext: head_dim={head_dim} exceeds the flash-attn cap of 256 \
         (HEADDIM_SWITCH instantiates up to 256). Gemma-4 global attention head_dim=512 \
         needs a non-flash cuda attention path (e.g. flashinfer or a split-KV kernel) — \
         unimplemented. This is a hard stop, NOT an illegal-memory-access."
    );

    let batch_size = cu_seqlens_q.dim(0) - 1;
    let max_blocks_per_seq = if block_table.ndim() == 2 {
        block_table.dim(1)
    } else {
        0
    };

    // Python line 587: override is_causal FIRST (before window_size_right)
    let is_causal = if max_seqlen_q == 1 { false } else { is_causal };
    // Python line 588: then compute window_size_right
    let window_size_right = if is_causal { 0 } else { -1_i32 };

    // Detect if we're in CUDA graph capture mode.
    // During capture, we disable seqlenq_ngroups_swapped and force num_splits=1
    // to match Python vLLM's behavior — Python doesn't use do_swap at all and
    // uses a fixed max_num_splits=1 during graph capture.
    let is_graph_capture = {
        let mut status = cudarc::driver::sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE;
        let ret = cudarc::driver::sys::cuStreamIsCapturing(_stream, &mut status);
        ret == cudarc::driver::sys::CUresult::CUDA_SUCCESS
            && status == cudarc::driver::sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_ACTIVE
    };

    // --- seqlenq_ngroups_swapped (matching Python flash_api.cpp lines 594-601) ---
    // GQA decode optimization: reshape Q from [B, H, D] to [B*ngroups, Hk, D]
    // so FA2 processes fewer heads with longer "sequences". The transpose
    // kernels and caching-allocator temporaries are graph-capture-safe.
    let ngroups = num_heads_orig / num_kv_heads;
    let do_swap = max_seqlen_q == 1
        && num_heads_orig > num_kv_heads
        && window_size_left < 0
        && window_size_right < 0
        && softcap == 0.0
        && head_dim.is_multiple_of(8);
    let (eff_max_seqlen_q, eff_num_heads, eff_total_q) = if do_swap {
        (ngroups, num_kv_heads, batch_size * ngroups)
    } else {
        (max_seqlen_q, num_heads_orig, total_q)
    };

    // --- Pad Q for the splitkv kernel ---
    // The splitkv kernel (always used for paged KV) reads kBlockM rows (up to 128)
    // per M-tile via unconditional vectorized loads, even when seqlen_q < kBlockM.
    // Only the output writes are masked to seqlen_q. With tight allocations from
    // our caching allocator, the reads go OOB into unmapped memory.
    // Fix: copy Q into a buffer with 128 extra rows of zero padding.
    // (PyTorch's allocator over-allocates so Python never hits this.)
    const Q_PAD_ROWS: usize = 128; // >= max kBlockM across all FA kernel configs
    let q_padded = alloc.alloc_tensor(
        &[eff_total_q + Q_PAD_ROWS, eff_num_heads, head_dim],
        q.dtype(),
    );
    // Zero the entire padded buffer, then copy actual Q data.
    crate::driver::memset_d8(q_padded.raw_ptr(), 0, q_padded.size_bytes(), _stream)
        .expect("memset q_padded");
    crate::driver::memcpy_dtod_async(
        q_padded.raw_ptr(),
        q.raw_ptr() as *const u8,
        eff_total_q * eff_num_heads * head_dim * q.dtype().size_bytes(),
        _stream,
    )
    .expect("copy Q into padded buffer");

    // --- Allocate all temporaries up front so they stay alive past mha_varlen_fwd ---

    let final_out = alloc.alloc_tensor(&[total_q, num_heads_orig, head_dim], q.dtype());

    let kernel_out_buf = if do_swap {
        Some(alloc.alloc_tensor(&[eff_total_q, eff_num_heads, head_dim], q.dtype()))
    } else {
        None
    };
    let out_ptr = match &kernel_out_buf {
        Some(buf) => buf.raw_ptr() as *mut c_void,
        None => final_out.raw_ptr() as *mut c_void,
    };

    let softmax_lse = alloc.alloc_tensor(&[eff_num_heads * eff_total_q], DType::F32);

    // Transpose Q if swapped: [B, H, D] -> [B*ngroups, Hk, D]
    // Python: q.reshape({B, Hk, ngroups, D}).transpose(1,2).reshape({B*ngroups, Hk, D})
    // The reshape after transpose triggers a clone (contiguous copy).
    let q_swapped_buf = if do_swap {
        let buf = alloc.alloc_tensor(&[eff_total_q, eff_num_heads, head_dim], q.dtype());
        ngroups_transpose_q(
            q.raw_ptr() as *const c_void,
            buf.raw_ptr() as *mut c_void,
            batch_size as i32,
            num_heads_orig as i32,
            num_kv_heads as i32,
            ngroups as i32,
            head_dim as i32,
            _stream,
        );
        Some(buf)
    } else {
        None
    };

    let (q_ptr, q_row_stride, q_head_stride) = match &q_swapped_buf {
        Some(buf) => (
            buf.raw_ptr() as *mut c_void,
            (num_kv_heads * head_dim) as i64, // contiguous [B*ngroups, Hk, D]
            head_dim as i64,
        ),
        None => (
            q_padded.raw_ptr() as *mut c_void,
            (num_heads_orig * head_dim) as i64,
            head_dim as i64,
        ),
    };

    let o_row_stride = if do_swap {
        (eff_num_heads * head_dim) as i64
    } else {
        (num_heads_orig * head_dim) as i64
    };
    let o_head_stride = head_dim as i64;

    // K/V cache: [num_blocks, block_size, num_kv_heads, head_dim]
    let kv_block_stride = (block_size * num_kv_heads * head_dim) as i64;
    let kv_row_stride = (num_kv_heads * head_dim) as i64;
    let kv_head_stride = head_dim as i64;

    // Python passes cu_seqlens_k.data_ptr() (line 683). We use a zero buffer
    // since our caller doesn't provide cu_seqlens_k for the paged path.
    let dummy_cu_seqlens_k = alloc.alloc_tensor(&[batch_size + 1], DType::I32);
    crate::driver::memset_d8(
        dummy_cu_seqlens_k.raw_ptr(),
        0,
        (batch_size + 1) * 4,
        _stream,
    )
    .expect("memset dummy_cu_seqlens_k");

    // Python: cu_seqlens_q_d = nullptr when swapped (line 600).
    // The C shim then uses q_batch_stride instead of cu_seqlens_q.
    let cu_seqlens_q_ptr: *const i32 = if do_swap {
        std::ptr::null()
    } else {
        cu_seqlens_q.as_ptr::<i32>()
    };

    // --- num_splits heuristic (only when swapped, matching Python line 705-710) ---
    let head_size_rounded = round_multiple(head_dim, if head_dim <= 128 { 32 } else { 64 });
    let block_n = if head_dim <= 64 {
        256
    } else if head_dim <= 128 {
        128
    } else {
        64
    };
    // Python: (max_seqlen_k + block_n - 1) / block_n (line 307)
    let num_n_blocks = max_seqlen_k.div_ceil(block_n);
    // Python: (max_seqlen_q + 64 - 1) / 64 (line 310)
    let num_m_blocks = eff_max_seqlen_q.div_ceil(64);

    // Split-K heuristic: parallelize K blocks across SMs when there aren't
    // enough CTAs to fill the GPU.
    // During graph capture, force num_splits=1 to match Python vLLM which uses
    // flash_attn_max_num_splits_for_cuda_graph=1 during capture. The dynamic
    // heuristic varies by batch size, causing different-sized split-K buffers
    // per capture which wastes memory and can trigger allocation failures.
    let num_splits = if is_graph_capture {
        1
    } else if num_sm > 0 {
        num_splits_heuristic(
            batch_size * eff_num_heads * num_m_blocks,
            (num_sm as usize) * 2,
            num_n_blocks,
            128,
        )
    } else {
        1
    };

    // Allocate split-K accum buffers — keep alive past mha_varlen_fwd.
    // Must use seqlen_q_rounded (not eff_max_seqlen_q) because the splitkv
    // kernel writes kBlockM rows per tile, and kBlockM is rounded up to 128.
    // With seqlenq_ngroups_swapped, eff_max_seqlen_q can be as small as 4
    // (= ngroups) while kBlockM = 128, causing massive buffer overflow.
    let seqlen_q_rounded = round_multiple(eff_max_seqlen_q, 128);
    let lse_accum_buf = if num_splits > 1 {
        Some(alloc.alloc_tensor(
            &[num_splits * batch_size * eff_num_heads * seqlen_q_rounded],
            DType::F32,
        ))
    } else {
        None
    };
    let out_accum_buf = if num_splits > 1 {
        Some(alloc.alloc_tensor(
            &[num_splits * batch_size * eff_num_heads * seqlen_q_rounded * head_size_rounded],
            DType::F32,
        ))
    } else {
        None
    };
    let lse_accum_ptr = lse_accum_buf
        .as_ref()
        .map_or(std::ptr::null_mut(), |t| t.raw_ptr() as *mut c_void);
    let out_accum_ptr = out_accum_buf
        .as_ref()
        .map_or(std::ptr::null_mut(), |t| t.raw_ptr() as *mut c_void);

    mha_varlen_fwd(
        q_ptr,
        k_cache.raw_ptr() as *mut c_void,
        v_cache.raw_ptr() as *mut c_void,
        out_ptr,
        softmax_lse.raw_ptr() as *mut c_void,
        cu_seqlens_q_ptr,
        dummy_cu_seqlens_k.as_ptr::<i32>(),
        seqused_k.as_ptr::<i32>(),
        block_table.as_ptr::<i32>(),
        max_blocks_per_seq as i32,
        batch_size as i32,
        eff_max_seqlen_q as i32,
        max_seqlen_k as i32,
        eff_num_heads as i32,
        num_kv_heads as i32,
        head_dim as i32,
        block_size as i32,
        q_row_stride,
        q_head_stride,
        kv_block_stride,
        kv_row_stride,
        kv_head_stride,
        o_row_stride,
        o_head_stride,
        softmax_scale,
        if is_causal { 1 } else { 0 },
        window_size_left,
        window_size_right,
        softcap,
        if q.dtype() == DType::BF16 { 1 } else { 0 },
        num_splits as i32,
        lse_accum_ptr,
        out_accum_ptr,
        if do_swap { 1 } else { 0 },
        eff_total_q as i32,
        // Spans: fused RoPE for cached K reads (combined cos|sin cache).
        // cos_sin_cache is always passed so per-block rotation can use it.
        cos_sin_cache_ptr as *const c_void,
        std::ptr::null(), // sin_ptr unused — rotate_k_smem_contiguous reads combined layout
        rotary_dim as i32,
        // rotate_cached_k: fallback when block_unrotated_flags is null.
        // When per-block flags are provided, they override this — set to 0
        // since K is stored rotated by default. When flags are null (no span
        // machinery), fall back to old behavior based on cos_sin_cache_ptr.
        if block_unrotated_flags.is_null() && !cos_sin_cache_ptr.is_null() && rotary_dim != 0 {
            1
        } else {
            0
        }, // rotate_cached_k
        if is_rotary_interleaved { 1 } else { 0 }, // is_rotary_interleaved
        block_unrotated_flags,
        _stream,
    );

    // Ensure temporaries stay alive past the kernel call
    drop(lse_accum_buf);
    drop(out_accum_buf);
    drop(q_swapped_buf);
    drop(q_padded);

    // Untranspose output if swapped: [B*ngroups, Hk, D] -> [B, H, D]
    // Python: out.reshape({B, ngroups, Hk, D}).transpose(1,2) then copy_ to original out
    if do_swap && let Some(ref kout) = kernel_out_buf {
        ngroups_untranspose_o(
            kout.raw_ptr() as *const c_void,
            final_out.raw_ptr() as *mut c_void,
            batch_size as i32,
            num_heads_orig as i32,
            num_kv_heads as i32,
            ngroups as i32,
            head_dim as i32,
            _stream,
        );
    }
    drop(kernel_out_buf);
    drop(dummy_cu_seqlens_k);

    final_out
}

// ---------------------------------------------------------------------------
// Scalar multiply (in-place via cuBLAS)
// ---------------------------------------------------------------------------

unsafe extern "C" {
    fn cublasScalEx(
        handle: cudarc::cublas::sys::cublasHandle_t,
        n: c_int,
        alpha: *const c_void,
        alphaType: cudarc::cublas::sys::cudaDataType_t,
        x: *mut c_void,
        xType: cudarc::cublas::sys::cudaDataType_t,
        incx: c_int,
        executionType: cudarc::cublas::sys::cudaDataType_t,
    ) -> cudarc::cublas::sys::cublasStatus_t;
}

/// Multiply every element of a tensor by a scalar, in-place.
///
/// * `x`: any contiguous tensor (F16, BF16, or F32)
/// * `scale`: the scalar multiplier (always f32)
/// * `cublas`: cuBLAS handle on the compute stream
pub unsafe fn scale_inplace(x: GpuTensor, scale: f32, cublas: &crate::cublas::CublasHandle) {
    use cudarc::cublas::sys::cudaDataType_t;
    let n = x.numel() as c_int;
    let x_type = match x.dtype() {
        DType::F16 => cudaDataType_t::CUDA_R_16F,
        DType::BF16 => cudaDataType_t::CUDA_R_16BF,
        DType::F32 => cudaDataType_t::CUDA_R_32F,
        _ => panic!("scale_inplace: unsupported dtype {:?}", x.dtype()),
    };
    let status = cublasScalEx(
        cublas.raw_handle(),
        n,
        &scale as *const f32 as *const c_void,
        cudaDataType_t::CUDA_R_32F,
        x.raw_ptr() as *mut c_void,
        x_type,
        1,
        cudaDataType_t::CUDA_R_32F,
    );
    assert_eq!(
        status,
        cudarc::cublas::sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS
    );
}

// ---------------------------------------------------------------------------
// Embedding pooling (GPU-side)
// ---------------------------------------------------------------------------

/// Extract a single row from `[num_tokens, hidden_size]` → `[hidden_size]` on GPU.
/// Used for Last-token and CLS pooling. Just a D2D memcpy of one row.
///
/// # Safety
/// `hidden_states` must be a valid 2D GPU tensor. `row_idx < num_tokens`.
pub unsafe fn pool_select_row(
    hidden_states: GpuTensor,
    row_idx: usize,
    alloc: &mut CachingAllocator,
    stream: CUstream,
) -> OwnedTensor {
    let hidden_size = hidden_states.dim(1);
    let elem_bytes = hidden_states.dtype().size_bytes();
    let row_bytes = hidden_size * elem_bytes;

    let out = alloc.alloc_tensor(&[hidden_size], hidden_states.dtype());
    let src = hidden_states.raw_ptr().add(row_idx * row_bytes);
    crate::driver::memcpy_dtod_async(out.raw_ptr(), src, row_bytes, stream)
        .expect("pool_select_row: D2D copy failed");
    out
}

/// Mean pool: average all rows of `[num_tokens, hidden_size]` → `[hidden_size]`.
///
/// Uses cuBLAS gemv: `out = (1/N) * A^T * ones_vec` where A = `[N, H]` (row-major).
/// The ones vector is allocated from the caching allocator and filled via memset.
///
/// # Safety
/// `hidden_states` must be a valid 2D GPU tensor with dtype F32.
/// For bf16/f16 inputs, caller must cast to f32 first.
pub unsafe fn pool_mean_f32(
    hidden_states: GpuTensor,
    cublas: &crate::cublas::CublasHandle,
    alloc: &mut CachingAllocator,
    stream: CUstream,
) -> OwnedTensor {
    use cudarc::cublas::sys::cublasOperation_t;

    assert_eq!(
        hidden_states.dtype(),
        DType::F32,
        "pool_mean_f32 requires f32 input"
    );
    let num_tokens = hidden_states.dim(0);
    let hidden_size = hidden_states.dim(1);

    if num_tokens == 1 {
        // Single token: just copy the row.
        return pool_select_row(hidden_states, 0, alloc, stream);
    }

    // Allocate ones vector [num_tokens] filled with 1.0f32.
    let ones = alloc.alloc_tensor(&[num_tokens], DType::F32);
    // Fill with 1.0f32 (bit pattern 0x3F800000). Use a kernel-free approach:
    // set all bytes to 0, then use cuBLAS to set to 1.0 would be circular.
    // Instead, write 1.0f32 via a small H2D.
    let ones_host: Vec<f32> = vec![1.0f32; num_tokens];
    crate::driver::memcpy_htod_async(
        ones.raw_ptr(),
        ones_host.as_ptr() as *const u8,
        num_tokens * 4,
        stream,
    )
    .expect("pool_mean: H2D ones vector");

    // Allocate output [hidden_size].
    let out = alloc.alloc_tensor(&[hidden_size], DType::F32);

    // cuBLAS gemv: out = alpha * A^T * x + beta * out
    // A is [N, H] in row-major = [H, N] in column-major.
    // We want A^T * x = [H, N]^T * [N] = [H] (sum of rows).
    // In column-major: A is [H, N], op=N means no-transpose, so y = A * x = [H].
    let alpha = 1.0f32 / num_tokens as f32;
    let beta = 0.0f32;

    unsafe extern "C" {
        fn cublasSgemv_v2(
            handle: cudarc::cublas::sys::cublasHandle_t,
            trans: cublasOperation_t,
            m: c_int,
            n: c_int,
            alpha: *const f32,
            a: *const f32,
            lda: c_int,
            x: *const f32,
            incx: c_int,
            beta: *const f32,
            y: *mut f32,
            incy: c_int,
        ) -> cudarc::cublas::sys::cublasStatus_t;
    }

    // Row-major [N, H] is column-major [H, N]. We want sum of rows = A^T * ones
    // In column-major: A = [H, N], trans=T → A^T * x = [N, H] * [N] — wrong dims.
    // Actually: row-major [N, H] stored as contiguous memory.
    // In cuBLAS column-major convention, this is a [H, N] matrix (columns are rows).
    // We want: out[h] = sum_n A[n, h] / N = (1/N) * A^T_colmajor * ones
    // A_colmajor = [H, N], transpose it → [N, H], multiply by ones[N] → [N] — wrong.
    // No: A_colmajor = [H, N], no transpose: y = A * x = [H, N] * [N, 1] = [H, 1]. Correct!
    let status = cublasSgemv_v2(
        cublas.raw_handle(),
        cublasOperation_t::CUBLAS_OP_N, // no transpose
        hidden_size as c_int,           // m = H
        num_tokens as c_int,            // n = N
        &alpha,
        hidden_states.as_ptr::<f32>(),
        hidden_size as c_int, // lda = H (column-major leading dim)
        ones.as_ptr::<f32>(),
        1,
        &beta,
        out.as_mut_ptr::<f32>(),
        1,
    );
    assert_eq!(
        status,
        cudarc::cublas::sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS,
        "pool_mean_f32: cublasSgemv failed"
    );

    out
}

// ---------------------------------------------------------------------------
// Bias Add
// ---------------------------------------------------------------------------

/// Add bias to a 2D tensor in-place: out[i,j] += bias[j]
///
/// * `out`: `[M, N]` — modified in-place
/// * `bias`: `[N]`
pub unsafe fn bias_add_inplace(out: GpuTensor, bias: GpuTensor, stream: CUstream) {
    debug_assert_eq!(out.ndim(), 2);
    debug_assert_eq!(bias.ndim(), 1);
    debug_assert_eq!(out.dim(1), bias.dim(0));
    let m = out.dim(0) as c_int;
    let n = out.dim(1) as c_int;
    match out.dtype() {
        DType::F16 => bias_add_f16(
            out.raw_ptr() as *mut _,
            bias.raw_ptr() as *const _,
            m,
            n,
            stream,
        ),
        DType::BF16 => bias_add_bf16(
            out.raw_ptr() as *mut _,
            bias.raw_ptr() as *const _,
            m,
            n,
            stream,
        ),
        DType::F32 => bias_add_f32(
            out.raw_ptr() as *mut _,
            bias.raw_ptr() as *const _,
            m,
            n,
            stream,
        ),
        _ => panic!("bias_add: unsupported dtype {:?}", out.dtype()),
    }
}

// ---------------------------------------------------------------------------
// QKV Split (zero-copy on dim 1 via pointer arithmetic + D2D copy)
// ---------------------------------------------------------------------------

/// Split fused QKV output into separate Q, K, V tensors via CUDA kernel.
///
/// * `qkv`: `[num_tokens, q_size + 2 * kv_size]` (contiguous)
/// * Returns: `(q, k, v)` each `[num_tokens, heads, head_dim]`
///
/// Uses a single kernel launch instead of per-token D2D memcpy calls,
/// eliminating O(num_tokens * 3) driver API overhead per layer.
pub unsafe fn split_qkv(
    qkv: GpuTensor,
    q_size: usize,
    kv_size: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    alloc: &mut CachingAllocator,
    stream: cudarc::driver::sys::CUstream,
) -> (OwnedTensor, OwnedTensor, OwnedTensor) {
    let num_tokens = qkv.dim(0);
    let total_dim = qkv.dim(1);

    debug_assert_eq!(total_dim, q_size + 2 * kv_size);

    let q = alloc.alloc_tensor(&[num_tokens, num_q_heads, head_dim], qkv.dtype());
    let k = alloc.alloc_tensor(&[num_tokens, num_kv_heads, head_dim], qkv.dtype());
    let v = alloc.alloc_tensor(&[num_tokens, num_kv_heads, head_dim], qkv.dtype());

    match qkv.dtype() {
        DType::F16 => split_qkv_f16(
            q.as_mut_ptr(),
            k.as_mut_ptr(),
            v.as_mut_ptr(),
            qkv.as_ptr(),
            q_size as i32,
            kv_size as i32,
            total_dim as i32,
            num_tokens as i32,
            stream,
        ),
        DType::BF16 => split_qkv_bf16(
            q.as_mut_ptr(),
            k.as_mut_ptr(),
            v.as_mut_ptr(),
            qkv.as_ptr(),
            q_size as i32,
            kv_size as i32,
            total_dim as i32,
            num_tokens as i32,
            stream,
        ),
        DType::F32 => split_qkv_f32(
            q.as_mut_ptr(),
            k.as_mut_ptr(),
            v.as_mut_ptr(),
            qkv.as_ptr(),
            q_size as i32,
            kv_size as i32,
            total_dim as i32,
            num_tokens as i32,
            stream,
        ),
        _ => panic!("split_qkv: unsupported dtype {:?}", qkv.dtype()),
    }

    (q, k, v)
}

/// Apply RoPE in-place to Q only (K is left unrotated).
///
/// Used by spans: keys are stored without positional encoding so that KV cache
/// blocks are relocatable. Q still needs rotation for correct attention scores.
///
/// * `q`: `[num_tokens, num_q_heads, head_dim]` — rotated in-place
/// * `positions`: `[num_tokens]` u32
/// * `cos_sin_cache`: `[max_pos, rotary_dim]`
pub unsafe fn rotary_embedding_q_only(
    q: GpuTensor,
    positions: GpuTensor,
    cos_sin_cache: GpuTensor,
    num_q_heads: usize,
    head_dim: usize,
    stream: CUstream,
) {
    let num_tokens = q.dim(0);
    let total_q_dim = (num_q_heads * head_dim) as i32;
    let rotary_dim = cos_sin_cache.dim(1) as i32;

    // Call the standard RoPE kernel with total_k_dim=0 so it skips K rotation.
    // The key pointer is set to the query pointer (unused when total_k_dim=0).
    match q.dtype() {
        DType::F16 => rotary_embedding_f16(
            positions.as_ptr(),
            q.as_mut_ptr(),
            q.as_mut_ptr(), // dummy K ptr (not used when total_k_dim=0)
            cos_sin_cache.as_ptr(),
            rotary_dim,
            total_q_dim,
            0, // total_k_dim = 0 → skip K rotation
            head_dim as i32,
            num_tokens as i32,
            stream,
        ),
        DType::BF16 => rotary_embedding_bf16(
            positions.as_ptr(),
            q.as_mut_ptr(),
            q.as_mut_ptr(),
            cos_sin_cache.as_ptr(),
            rotary_dim,
            total_q_dim,
            0,
            head_dim as i32,
            num_tokens as i32,
            stream,
        ),
        DType::F32 => rotary_embedding_f32(
            positions.as_ptr(),
            q.as_mut_ptr() as *mut f32,
            q.as_mut_ptr() as *mut f32,
            cos_sin_cache.as_ptr() as *const f32,
            rotary_dim,
            total_q_dim,
            0,
            head_dim as i32,
            num_tokens as i32,
            stream,
        ),
        _ => panic!("rotary_embedding_q_only: unsupported dtype {:?}", q.dtype()),
    }
}

/// Apply interleaved RoPE in-place to Q only (K is left unrotated).
///
/// Same as [`rotary_embedding_q_only`] but for interleaved RoPE layout
/// (pairs at adjacent positions 2i, 2i+1 instead of NeoX d, d+half_dim).
///
/// * `q`: `[num_tokens, num_q_heads * head_dim]` — rotated in-place
/// * `positions`: `[num_tokens]` u32
/// * `cos_sin_cache`: `[max_pos, rotary_dim]`
pub unsafe fn rotary_embedding_interleaved_q_only(
    q: GpuTensor,
    positions: GpuTensor,
    cos_sin_cache: GpuTensor,
    num_q_heads: usize,
    head_dim: usize,
    stream: CUstream,
) {
    let num_tokens = q.dim(0);
    let total_q_dim = (num_q_heads * head_dim) as i32;
    let rotary_dim = cos_sin_cache.dim(1) as i32;

    match q.dtype() {
        DType::F16 => rotary_embedding_interleaved_f16(
            positions.as_ptr(),
            q.as_mut_ptr(),
            q.as_mut_ptr(), // dummy K ptr (not used when total_k_dim=0)
            cos_sin_cache.as_ptr(),
            rotary_dim,
            total_q_dim,
            0, // total_k_dim = 0 → skip K rotation
            head_dim as i32,
            num_tokens as i32,
            stream,
        ),
        DType::BF16 => rotary_embedding_interleaved_bf16(
            positions.as_ptr(),
            q.as_mut_ptr(),
            q.as_mut_ptr(),
            cos_sin_cache.as_ptr(),
            rotary_dim,
            total_q_dim,
            0,
            head_dim as i32,
            num_tokens as i32,
            stream,
        ),
        DType::F32 => rotary_embedding_interleaved_f32(
            positions.as_ptr(),
            q.as_mut_ptr() as *mut f32,
            q.as_mut_ptr() as *mut f32,
            cos_sin_cache.as_ptr() as *const f32,
            rotary_dim,
            total_q_dim,
            0,
            head_dim as i32,
            num_tokens as i32,
            stream,
        ),
        _ => panic!(
            "rotary_embedding_interleaved_q_only: unsupported dtype {:?}",
            q.dtype()
        ),
    }
}

/// Apply RoPE in-place to K tokens in a paged KV cache.
///
/// For spans (relocatable KV cache blocks): keys are stored without positional
/// encoding. This kernel rotates cached K using each token's actual sequence
/// position, so that FA2 reads correctly rotated keys.
///
/// Use with `inverse=false` before attention (rotate), then `inverse=true`
/// after attention (un-rotate) to restore the cache to its unrotated state.
///
/// * `k_cache`: `[num_blocks, block_size, num_kv_heads, head_dim]`
/// * `cos_sin_cache`: `[max_pos, rotary_dim]`
/// * `block_table`: `[batch_size, max_blocks_per_seq]` (I32)
/// * `seqused_k`: `[batch_size]` (I32) — actual K lengths per sequence
/// * `block_flags`: optional GPU `[num_physical_blocks]` u8 array. When
///   provided, only blocks with `flag != 0` are rotated (span blocks).
///   Pass `std::ptr::null()` to rotate all blocks.
/// * `inverse`: if true, apply inverse rotation (un-rotate)
#[allow(clippy::too_many_arguments)]
pub unsafe fn rotary_paged_k_cache(
    k_cache: GpuTensor,
    cos_sin_cache: GpuTensor,
    block_table: GpuTensor,
    seqused_k: GpuTensor,
    block_flags: *const u8,
    batch_size: usize,
    max_seqlen_k: usize,
    max_blocks_per_seq: usize,
    page_block_size: usize,
    num_kv_heads: usize,
    head_dim: usize,
    inverse: bool,
    stream: CUstream,
) {
    let rotary_dim = cos_sin_cache.dim(1) as i32;
    let inv = if inverse { 1i32 } else { 0i32 };

    match k_cache.dtype() {
        DType::F16 => rotary_paged_k_cache_f16(
            k_cache.as_mut_ptr(),
            cos_sin_cache.as_ptr(),
            block_table.as_ptr() as *const i32,
            seqused_k.as_ptr() as *const i32,
            block_flags,
            batch_size as i32,
            max_seqlen_k as i32,
            max_blocks_per_seq as i32,
            page_block_size as i32,
            num_kv_heads as i32,
            head_dim as i32,
            rotary_dim,
            inv,
            stream,
        ),
        DType::BF16 => rotary_paged_k_cache_bf16(
            k_cache.as_mut_ptr(),
            cos_sin_cache.as_ptr(),
            block_table.as_ptr() as *const i32,
            seqused_k.as_ptr() as *const i32,
            block_flags,
            batch_size as i32,
            max_seqlen_k as i32,
            max_blocks_per_seq as i32,
            page_block_size as i32,
            num_kv_heads as i32,
            head_dim as i32,
            rotary_dim,
            inv,
            stream,
        ),
        _ => panic!(
            "rotary_paged_k_cache: unsupported dtype {:?}",
            k_cache.dtype()
        ),
    }
}

/// Fused QKV split + RoPE: reads from the fused QKV GEMM output, applies
/// rotary position embedding to Q and K, copies V, and writes contiguous
/// outputs. Replaces separate `split_qkv` + `rotary_embedding_inplace` calls,
/// saving 1 kernel launch per layer per step.
///
/// * `qkv`: `[num_tokens, q_size + 2*kv_size]` — fused QKV GEMM output
/// * `positions`: `[num_tokens]` u32 (1D, broadcast across MRoPE axes when
///   `mrope_section.is_some()`) or `[3, num_tokens]` u32 (per-axis values for
///   image-bearing batches on MRoPE arches). Shape is read at runtime; the
///   kernel takes its broadcast path via `positions_stride0 = 0` when 1D.
/// * `cos_sin_cache`: `[max_pos, rotary_dim]`
/// * `mrope_section`: `None` for text arches; `Some([T,H,W])` for MRoPE arches
///   (Qwen2-VL / Qwen2.5-VL). Cumulative boundaries computed inside. The
///   `Some + 1D positions` combo matches Python vLLM's
///   `arange(n).expand(3, -1)` view: kernel reads `positions[token_idx]` for
///   every axis, dispatch by section is moot because all axes return the same
///   value, so the result is numerically identical to standard 1D rope on the
///   same `inv_freq` table.
/// * Returns: `(q, k, v)` where q is `[num_tokens, num_q_heads, head_dim]`,
///   k and v are `[num_tokens, num_kv_heads, head_dim]`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn fused_qkv_rope(
    qkv: GpuTensor,
    positions: GpuTensor,
    cos_sin_cache: GpuTensor,
    q_size: usize,
    kv_size: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    mrope_section: Option<[u32; 3]>,
    alloc: &mut CachingAllocator,
    stream: CUstream,
) -> (OwnedTensor, OwnedTensor, OwnedTensor) {
    let num_tokens = qkv.dim(0);
    let total_dim = qkv.dim(1);
    let rotary_dim = cos_sin_cache.dim(1) as i32;

    debug_assert_eq!(total_dim, q_size + 2 * kv_size);

    let q = alloc.alloc_tensor(&[num_tokens, num_q_heads, head_dim], qkv.dtype());
    let k = alloc.alloc_tensor(&[num_tokens, num_kv_heads, head_dim], qkv.dtype());
    let v = alloc.alloc_tensor(&[num_tokens, num_kv_heads, head_dim], qkv.dtype());

    // MRoPE: positions tensor shape determines stride0 — `[3, n_tokens]`
    // (real per-axis values for image-bearing batches) → stride0 = n_tokens;
    // `[n_tokens]` (1D, every axis reads the same broadcast value, including
    // text-only Qwen2-VL) → stride0 = 0. The kernel uses
    // `positions_stride0 != 0` as the mrope flag, so 1D positions on an
    // mrope arch take the broadcast path with sec0/sec01 collapsed —
    // numerically identical to standard 1D rope on the same inv_freq table.
    // Mirrors Python vLLM's `arange(n).expand(3, -1)` stride-0 view.
    let (positions_stride0, mrope_sec0, mrope_sec01) = match mrope_section {
        Some([t, h, _w]) => {
            let stride0 = if positions.ndim() == 2 && positions.dim(0) == 3 {
                num_tokens as i32
            } else {
                0
            };
            (stride0, t as i32, (t + h) as i32)
        }
        None => (0, 0, 0),
    };

    match qkv.dtype() {
        DType::F16 => fused_qkv_rope_f16(
            q.as_mut_ptr(),
            k.as_mut_ptr(),
            v.as_mut_ptr(),
            qkv.as_ptr(),
            positions.as_ptr(),
            cos_sin_cache.as_ptr(),
            q_size as i32,
            kv_size as i32,
            total_dim as i32,
            rotary_dim,
            head_dim as i32,
            num_tokens as i32,
            positions_stride0,
            mrope_sec0,
            mrope_sec01,
            stream,
        ),
        DType::BF16 => fused_qkv_rope_bf16(
            q.as_mut_ptr(),
            k.as_mut_ptr(),
            v.as_mut_ptr(),
            qkv.as_ptr(),
            positions.as_ptr(),
            cos_sin_cache.as_ptr(),
            q_size as i32,
            kv_size as i32,
            total_dim as i32,
            rotary_dim,
            head_dim as i32,
            num_tokens as i32,
            positions_stride0,
            mrope_sec0,
            mrope_sec01,
            stream,
        ),
        DType::F32 => fused_qkv_rope_f32(
            q.as_mut_ptr(),
            k.as_mut_ptr(),
            v.as_mut_ptr(),
            qkv.as_ptr(),
            positions.as_ptr(),
            cos_sin_cache.as_ptr(),
            q_size as i32,
            kv_size as i32,
            total_dim as i32,
            rotary_dim,
            head_dim as i32,
            num_tokens as i32,
            positions_stride0,
            mrope_sec0,
            mrope_sec01,
            stream,
        ),
        _ => panic!("fused_qkv_rope: unsupported dtype {:?}", qkv.dtype()),
    }

    (q, k, v)
}

/// Fused QKV split + RoPE + reshape_and_cache (decode path).
///
/// Reads the fused QKV GEMM output, applies RoPE, writes Q to a contiguous
/// output buffer, and writes K/V directly into the paged KV cache. Returns
/// only Q — no intermediate K/V allocations needed.
///
/// `mrope_section`: see [`fused_qkv_rope`] — same shape-driven stride0 logic.
/// `None` for text arches; `Some([T,H,W])` for MRoPE arches with stride0
/// chosen at runtime from `positions.dims()` so a text-only batch on an
/// MRoPE arch reuses the kernel's broadcast path (Python parity with
/// `arange(n).expand(3, -1)`).
#[allow(clippy::too_many_arguments)]
pub unsafe fn fused_qkv_rope_cache(
    qkv: GpuTensor,
    positions: GpuTensor,
    cos_sin_cache: GpuTensor,
    slot_mapping: GpuTensor,
    key_cache: GpuTensor,
    value_cache: GpuTensor,
    q_size: usize,
    kv_size: usize,
    num_q_heads: usize,
    head_dim: usize,
    mrope_section: Option<[u32; 3]>,
    alloc: &mut CachingAllocator,
    stream: CUstream,
) -> OwnedTensor {
    let num_tokens = qkv.dim(0);
    let total_dim = qkv.dim(1);
    let rotary_dim = cos_sin_cache.dim(1) as i32;

    debug_assert_eq!(total_dim, q_size + 2 * kv_size);

    let q = alloc.alloc_tensor(&[num_tokens, num_q_heads, head_dim], qkv.dtype());

    let (positions_stride0, mrope_sec0, mrope_sec01) = match mrope_section {
        Some([t, h, _w]) => {
            let stride0 = if positions.ndim() == 2 && positions.dim(0) == 3 {
                num_tokens as i32
            } else {
                0
            };
            (stride0, t as i32, (t + h) as i32)
        }
        None => (0, 0, 0),
    };

    match qkv.dtype() {
        DType::F16 => fused_qkv_rope_cache_f16(
            q.as_mut_ptr(),
            key_cache.raw_ptr() as *mut u16,
            value_cache.raw_ptr() as *mut u16,
            qkv.as_ptr(),
            positions.as_ptr(),
            cos_sin_cache.as_ptr(),
            slot_mapping.as_ptr() as *const i64,
            q_size as i32,
            kv_size as i32,
            total_dim as i32,
            rotary_dim,
            head_dim as i32,
            num_tokens as i32,
            positions_stride0,
            mrope_sec0,
            mrope_sec01,
            stream,
        ),
        DType::BF16 => fused_qkv_rope_cache_bf16(
            q.as_mut_ptr(),
            key_cache.raw_ptr() as *mut u16,
            value_cache.raw_ptr() as *mut u16,
            qkv.as_ptr(),
            positions.as_ptr(),
            cos_sin_cache.as_ptr(),
            slot_mapping.as_ptr() as *const i64,
            q_size as i32,
            kv_size as i32,
            total_dim as i32,
            rotary_dim,
            head_dim as i32,
            num_tokens as i32,
            positions_stride0,
            mrope_sec0,
            mrope_sec01,
            stream,
        ),
        DType::F32 => fused_qkv_rope_cache_f32(
            q.as_mut_ptr(),
            key_cache.raw_ptr() as *mut f32,
            value_cache.raw_ptr() as *mut f32,
            qkv.as_ptr(),
            positions.as_ptr(),
            cos_sin_cache.as_ptr(),
            slot_mapping.as_ptr() as *const i64,
            q_size as i32,
            kv_size as i32,
            total_dim as i32,
            rotary_dim,
            head_dim as i32,
            num_tokens as i32,
            positions_stride0,
            mrope_sec0,
            mrope_sec01,
            stream,
        ),
        _ => panic!("fused_qkv_rope_cache: unsupported dtype {:?}", qkv.dtype()),
    }

    q
}

/// Apply 2D vision RoPE (neox-style halves) to `x` in-place.
///
/// Used by Qwen2-VL's vision encoder. `x` is `[seq_len, num_heads, head_size]`,
/// `cos` and `sin` are `[seq_len, head_size/2]`. Math matches Python vLLM's
/// `ApplyRotaryEmb.forward_static` neox path:
///
/// ```text
/// x1, x2 = chunk(x, 2, dim=-1)
/// out = cat([x1*cos - x2*sin, x2*cos + x1*sin], dim=-1)
/// ```
///
/// Distinct from the text-side rope kernels: no packed QKV, no KV-cache
/// writeback, no positions table — caller indexes cos/sin per the per-image
/// grid before the call. Invoke once for Q and once for K.
///
/// # Safety
/// All tensor pointers must be valid GPU memory; shapes must match
/// `[seq_len, num_heads, head_size]` for `x` and `[seq_len, head_size/2]`
/// for `cos`/`sin`. Caller serializes against any other kernel using these
/// tensors (this kernel writes `x` in-place).
pub unsafe fn vision_rope_apply(x: GpuTensor, cos: GpuTensor, sin: GpuTensor, stream: CUstream) {
    debug_assert_eq!(x.dtype(), cos.dtype());
    debug_assert_eq!(x.dtype(), sin.dtype());
    debug_assert_eq!(x.ndim(), 3);
    debug_assert_eq!(cos.ndim(), 2);
    debug_assert_eq!(sin.ndim(), 2);

    let seq_len = x.dim(0) as i32;
    let num_heads = x.dim(1) as i32;
    let head_size = x.dim(2) as i32;
    debug_assert_eq!(cos.dim(0), x.dim(0));
    debug_assert_eq!(sin.dim(0), x.dim(0));
    debug_assert_eq!(cos.dim(1) * 2, x.dim(2));
    debug_assert_eq!(sin.dim(1) * 2, x.dim(2));

    match x.dtype() {
        DType::F16 => vision_rope_apply_f16(
            x.as_mut_ptr(),
            cos.as_ptr(),
            sin.as_ptr(),
            seq_len,
            num_heads,
            head_size,
            stream,
        ),
        DType::BF16 => vision_rope_apply_bf16(
            x.as_mut_ptr(),
            cos.as_ptr(),
            sin.as_ptr(),
            seq_len,
            num_heads,
            head_size,
            stream,
        ),
        DType::F32 => vision_rope_apply_f32(
            x.as_mut_ptr(),
            cos.as_ptr(),
            sin.as_ptr(),
            seq_len,
            num_heads,
            head_size,
            stream,
        ),
        _ => panic!("vision_rope_apply: unsupported dtype {:?}", x.dtype()),
    }
}

/// Fused interleaved QKV split + RoPE + reshape_and_cache (decode path, Cohere/CommandR).
///
/// Same as [`fused_qkv_rope_cache`] but uses interleaved RoPE pairing (2i, 2i+1).
#[allow(clippy::too_many_arguments)]
pub unsafe fn fused_qkv_interleaved_rope_cache(
    qkv: GpuTensor,
    positions: GpuTensor,
    cos_sin_cache: GpuTensor,
    slot_mapping: GpuTensor,
    key_cache: GpuTensor,
    value_cache: GpuTensor,
    q_size: usize,
    kv_size: usize,
    num_q_heads: usize,
    head_dim: usize,
    alloc: &mut CachingAllocator,
    stream: CUstream,
) -> OwnedTensor {
    let num_tokens = qkv.dim(0);
    let total_dim = qkv.dim(1);
    let rotary_dim = cos_sin_cache.dim(1) as i32;

    debug_assert_eq!(total_dim, q_size + 2 * kv_size);

    let q = alloc.alloc_tensor(&[num_tokens, num_q_heads, head_dim], qkv.dtype());

    match qkv.dtype() {
        DType::F16 => fused_qkv_interleaved_rope_cache_f16(
            q.as_mut_ptr(),
            key_cache.raw_ptr() as *mut u16,
            value_cache.raw_ptr() as *mut u16,
            qkv.as_ptr(),
            positions.as_ptr(),
            cos_sin_cache.as_ptr(),
            slot_mapping.as_ptr() as *const i64,
            q_size as i32,
            kv_size as i32,
            total_dim as i32,
            rotary_dim,
            head_dim as i32,
            num_tokens as i32,
            stream,
        ),
        DType::BF16 => fused_qkv_interleaved_rope_cache_bf16(
            q.as_mut_ptr(),
            key_cache.raw_ptr() as *mut u16,
            value_cache.raw_ptr() as *mut u16,
            qkv.as_ptr(),
            positions.as_ptr(),
            cos_sin_cache.as_ptr(),
            slot_mapping.as_ptr() as *const i64,
            q_size as i32,
            kv_size as i32,
            total_dim as i32,
            rotary_dim,
            head_dim as i32,
            num_tokens as i32,
            stream,
        ),
        DType::F32 => fused_qkv_interleaved_rope_cache_f32(
            q.as_mut_ptr(),
            key_cache.raw_ptr() as *mut f32,
            value_cache.raw_ptr() as *mut f32,
            qkv.as_ptr(),
            positions.as_ptr(),
            cos_sin_cache.as_ptr(),
            slot_mapping.as_ptr() as *const i64,
            q_size as i32,
            kv_size as i32,
            total_dim as i32,
            rotary_dim,
            head_dim as i32,
            num_tokens as i32,
            stream,
        ),
        _ => panic!(
            "fused_qkv_interleaved_rope_cache: unsupported dtype {:?}",
            qkv.dtype()
        ),
    }

    q
}

/// Fused QKV split + RoPE + FP8 quantize + cache write (decode, NeoX RoPE).
///
/// Q is written as BF16, K/V are quantized to FP8 E4M3 and written directly to cache.
#[allow(clippy::too_many_arguments)]
pub unsafe fn fused_qkv_rope_cache_fp8(
    qkv: GpuTensor,
    positions: GpuTensor,
    cos_sin_cache: GpuTensor,
    slot_mapping: GpuTensor,
    key_cache: GpuTensor,
    value_cache: GpuTensor,
    k_scale: *const f32,
    v_scale: *const f32,
    q_size: usize,
    kv_size: usize,
    num_q_heads: usize,
    head_dim: usize,
    alloc: &mut CachingAllocator,
    stream: CUstream,
) -> OwnedTensor {
    let num_tokens = qkv.dim(0);
    let total_dim = qkv.dim(1);
    let rotary_dim = cos_sin_cache.dim(1) as i32;

    debug_assert_eq!(total_dim, q_size + 2 * kv_size);
    debug_assert!(
        qkv.dtype() == DType::BF16,
        "FP8 cache fusion only supports BF16 input"
    );

    let q = alloc.alloc_tensor(&[num_tokens, num_q_heads, head_dim], DType::BF16);

    fused_qkv_rope_cache_fp8_bf16(
        q.as_mut_ptr(),
        key_cache.raw_ptr() as *mut u8,
        value_cache.raw_ptr() as *mut u8,
        qkv.as_ptr(),
        positions.as_ptr(),
        cos_sin_cache.as_ptr(),
        slot_mapping.as_ptr() as *const i64,
        k_scale,
        v_scale,
        q_size as i32,
        kv_size as i32,
        total_dim as i32,
        rotary_dim,
        head_dim as i32,
        num_tokens as i32,
        stream,
    );

    q
}

/// Fused interleaved QKV split + RoPE + FP8 quantize + cache write (decode, Cohere/CommandR).
#[allow(clippy::too_many_arguments)]
pub unsafe fn fused_qkv_interleaved_rope_cache_fp8(
    qkv: GpuTensor,
    positions: GpuTensor,
    cos_sin_cache: GpuTensor,
    slot_mapping: GpuTensor,
    key_cache: GpuTensor,
    value_cache: GpuTensor,
    k_scale: *const f32,
    v_scale: *const f32,
    q_size: usize,
    kv_size: usize,
    num_q_heads: usize,
    head_dim: usize,
    alloc: &mut CachingAllocator,
    stream: CUstream,
) -> OwnedTensor {
    let num_tokens = qkv.dim(0);
    let total_dim = qkv.dim(1);
    let rotary_dim = cos_sin_cache.dim(1) as i32;

    debug_assert_eq!(total_dim, q_size + 2 * kv_size);
    debug_assert!(
        qkv.dtype() == DType::BF16,
        "FP8 cache fusion only supports BF16 input"
    );

    let q = alloc.alloc_tensor(&[num_tokens, num_q_heads, head_dim], DType::BF16);

    fused_qkv_interleaved_rope_cache_fp8_bf16(
        q.as_mut_ptr(),
        key_cache.raw_ptr() as *mut u8,
        value_cache.raw_ptr() as *mut u8,
        qkv.as_ptr(),
        positions.as_ptr(),
        cos_sin_cache.as_ptr(),
        slot_mapping.as_ptr() as *const i64,
        k_scale,
        v_scale,
        q_size as i32,
        kv_size as i32,
        total_dim as i32,
        rotary_dim,
        head_dim as i32,
        num_tokens as i32,
        stream,
    );

    q
}

// ---------------------------------------------------------------------------
// Batched argmax (GPU greedy sampling — avoids D2H of full logits)
// ---------------------------------------------------------------------------

unsafe extern "C" {
    fn argmax_batched_f16(
        output: *mut u32,
        logits: *const u16,
        vocab_size: c_int,
        batch_size: c_int,
        stream: CUstream,
    );
    fn argmax_batched_bf16(
        output: *mut u32,
        logits: *const u16,
        vocab_size: c_int,
        batch_size: c_int,
        stream: CUstream,
    );
    fn argmax_batched_f32(
        output: *mut u32,
        logits: *const f32,
        vocab_size: c_int,
        batch_size: c_int,
        stream: CUstream,
    );

    fn sample_gumbel_batched_f16(
        output: *mut u32,
        logits: *const u16,
        vocab_size: c_int,
        batch_size: c_int,
        temperatures: *const f32,
        seeds: *const u32,
        scratch_vals: *mut f32,
        scratch_indices: *mut c_int,
        stream: CUstream,
    );
    fn sample_gumbel_batched_bf16(
        output: *mut u32,
        logits: *const u16,
        vocab_size: c_int,
        batch_size: c_int,
        temperatures: *const f32,
        seeds: *const u32,
        scratch_vals: *mut f32,
        scratch_indices: *mut c_int,
        stream: CUstream,
    );
    fn sample_gumbel_batched_f32(
        output: *mut u32,
        logits: *const f32,
        vocab_size: c_int,
        batch_size: c_int,
        temperatures: *const f32,
        seeds: *const u32,
        scratch_vals: *mut f32,
        scratch_indices: *mut c_int,
        stream: CUstream,
    );

    fn sample_batched_f16(
        output: *mut u32,
        logits: *const u16,
        vocab_size: c_int,
        batch_size: c_int,
        temperatures: *const f32,
        top_ks: *const c_int,
        top_ps: *const f32,
        min_ps: *const f32,
        uniform_randoms: *const f32,
        stream: CUstream,
    );
    fn sample_batched_bf16(
        output: *mut u32,
        logits: *const u16,
        vocab_size: c_int,
        batch_size: c_int,
        temperatures: *const f32,
        top_ks: *const c_int,
        top_ps: *const f32,
        min_ps: *const f32,
        uniform_randoms: *const f32,
        stream: CUstream,
    );
    fn sample_batched_f32(
        output: *mut u32,
        logits: *const f32,
        vocab_size: c_int,
        batch_size: c_int,
        temperatures: *const f32,
        top_ks: *const c_int,
        top_ps: *const f32,
        min_ps: *const f32,
        uniform_randoms: *const f32,
        stream: CUstream,
    );
}

/// Batched argmax over logits `[batch_size, vocab_size]`.
///
/// Returns `[batch_size]` u32 tensor of token IDs, allocated from arena.
/// Only copies `batch_size * 4` bytes D2H instead of `batch_size * vocab_size * dtype_size`.
pub unsafe fn argmax_batched(
    logits: GpuTensor,
    alloc: &mut CachingAllocator,
    stream: CUstream,
) -> OwnedTensor {
    let batch_size = logits.dim(0) as c_int;
    let vocab_size = logits.dim(1) as c_int;
    let out = alloc.alloc_tensor(&[batch_size as usize], DType::U32);

    match logits.dtype() {
        DType::F16 => argmax_batched_f16(
            out.as_mut_ptr() as *mut u32,
            logits.as_ptr() as *const u16,
            vocab_size,
            batch_size,
            stream,
        ),
        DType::BF16 => argmax_batched_bf16(
            out.as_mut_ptr() as *mut u32,
            logits.as_ptr() as *const u16,
            vocab_size,
            batch_size,
            stream,
        ),
        DType::F32 => argmax_batched_f32(
            out.as_mut_ptr() as *mut u32,
            logits.as_ptr() as *const f32,
            vocab_size,
            batch_size,
            stream,
        ),
        _ => panic!("argmax_batched: unsupported dtype {:?}", logits.dtype()),
    }
    out
}

/// Fast batched sampling via the Gumbel-max trick (multi-block).
///
/// Matches Python vLLM's Triton implementation: 2D grid with 1024 threads per
/// block, ceil(vocab/1024) blocks per request. Phase 1 computes per-block
/// local argmax, phase 2 reduces across blocks. Uses Philox4x32-10 RNG for
/// per-element noise (matching Triton's `tl.rand`).
///
/// * `logits`: `[batch_size, vocab_size]`
/// * `temperatures`: `[batch_size]` (F32, on GPU)
/// * `seeds`: `[batch_size]` (U32, on GPU) — Philox RNG seeds per request
///
/// Returns `[batch_size]` u32 tensor of sampled token IDs, allocated from arena.
pub unsafe fn sample_gumbel_batched(
    logits: GpuTensor,
    temperatures: GpuTensor,
    seeds: GpuTensor,
    alloc: &mut CachingAllocator,
    stream: CUstream,
) -> OwnedTensor {
    let batch_size = logits.dim(0) as c_int;
    let vocab_size = logits.dim(1) as c_int;
    let out = alloc.alloc_tensor(&[batch_size as usize], DType::U32);

    // Scratch for multi-block reduction: [batch_size, num_blocks] for vals and indices.
    let num_blocks = (vocab_size as usize).div_ceil(1024);
    let scratch_elems = batch_size as usize * num_blocks;
    let scratch_vals = alloc.alloc_tensor(&[scratch_elems], DType::F32);
    let scratch_indices = alloc.alloc_tensor(&[scratch_elems], DType::I32);

    match logits.dtype() {
        DType::F16 => sample_gumbel_batched_f16(
            out.as_mut_ptr() as *mut u32,
            logits.as_ptr() as *const u16,
            vocab_size,
            batch_size,
            temperatures.as_ptr(),
            seeds.as_ptr() as *const u32,
            scratch_vals.as_mut_ptr() as *mut f32,
            scratch_indices.as_mut_ptr() as *mut c_int,
            stream,
        ),
        DType::BF16 => sample_gumbel_batched_bf16(
            out.as_mut_ptr() as *mut u32,
            logits.as_ptr() as *const u16,
            vocab_size,
            batch_size,
            temperatures.as_ptr(),
            seeds.as_ptr() as *const u32,
            scratch_vals.as_mut_ptr() as *mut f32,
            scratch_indices.as_mut_ptr() as *mut c_int,
            stream,
        ),
        DType::F32 => sample_gumbel_batched_f32(
            out.as_mut_ptr() as *mut u32,
            logits.as_ptr() as *const f32,
            vocab_size,
            batch_size,
            temperatures.as_ptr(),
            seeds.as_ptr() as *const u32,
            scratch_vals.as_mut_ptr() as *mut f32,
            scratch_indices.as_mut_ptr() as *mut c_int,
            stream,
        ),
        _ => panic!(
            "sample_gumbel_batched: unsupported dtype {:?}",
            logits.dtype()
        ),
    }
    out
}

/// Batched GPU sampling with top-k/top-p/min-p over logits `[batch_size, vocab_size]`.
///
/// * `logits`: `[batch_size, vocab_size]`
/// * `temperatures`: `[batch_size]` (F32, on GPU)
/// * `top_ks`: `[batch_size]` (I32, on GPU)
/// * `top_ps`: `[batch_size]` (F32, on GPU)
/// * `min_ps`: `[batch_size]` (F32, on GPU)
/// * `uniform_randoms`: `[batch_size]` (F32, on GPU) — pre-generated U[0,1) random values
///
/// Returns `[batch_size]` u32 tensor of sampled token IDs, allocated from arena.
#[allow(clippy::too_many_arguments)]
pub unsafe fn sample_batched(
    logits: GpuTensor,
    temperatures: GpuTensor,
    top_ks: GpuTensor,
    top_ps: GpuTensor,
    min_ps: GpuTensor,
    uniform_randoms: GpuTensor,
    alloc: &mut CachingAllocator,
    stream: CUstream,
) -> OwnedTensor {
    let batch_size = logits.dim(0) as c_int;
    let vocab_size = logits.dim(1) as c_int;
    let out = alloc.alloc_tensor(&[batch_size as usize], DType::U32);

    match logits.dtype() {
        DType::F16 => sample_batched_f16(
            out.as_mut_ptr() as *mut u32,
            logits.as_ptr() as *const u16,
            vocab_size,
            batch_size,
            temperatures.as_ptr(),
            top_ks.as_ptr(),
            top_ps.as_ptr(),
            min_ps.as_ptr(),
            uniform_randoms.as_ptr(),
            stream,
        ),
        DType::BF16 => sample_batched_bf16(
            out.as_mut_ptr() as *mut u32,
            logits.as_ptr() as *const u16,
            vocab_size,
            batch_size,
            temperatures.as_ptr(),
            top_ks.as_ptr(),
            top_ps.as_ptr(),
            min_ps.as_ptr(),
            uniform_randoms.as_ptr(),
            stream,
        ),
        DType::F32 => sample_batched_f32(
            out.as_mut_ptr() as *mut u32,
            logits.as_ptr() as *const f32,
            vocab_size,
            batch_size,
            temperatures.as_ptr(),
            top_ks.as_ptr(),
            top_ps.as_ptr(),
            min_ps.as_ptr(),
            uniform_randoms.as_ptr(),
            stream,
        ),
        _ => panic!("sample_batched: unsupported dtype {:?}", logits.dtype()),
    }
    out
}

// ---------------------------------------------------------------------------
// Cast to f32 (for logit modification kernels that require f32)
// ---------------------------------------------------------------------------

unsafe extern "C" {
    fn cast_to_f32_f16(output: *mut f32, input: *const u16, n: c_int, stream: CUstream);
    fn cast_to_f32_bf16(output: *mut f32, input: *const u16, n: c_int, stream: CUstream);
}

/// Cast logits from any dtype to f32 on GPU.
/// If already f32, returns a view (no copy). Otherwise allocates from arena.
pub unsafe fn cast_logits_to_f32(
    logits: GpuTensor,
    alloc: &mut CachingAllocator,
    stream: CUstream,
) -> OwnedTensor {
    let n = logits.numel() as c_int;
    let shape: Vec<usize> = logits.shape().iter().map(|&d| d as usize).collect();
    let out = alloc.alloc_tensor(&shape, DType::F32);
    match logits.dtype() {
        DType::F32 => {
            // Just D2D copy.
            crate::driver::memcpy_dtod_async(
                out.as_gpu_tensor().raw_ptr(),
                logits.raw_ptr() as *const u8,
                logits.size_bytes(),
                stream,
            )
            .expect("cast_logits_to_f32: D2D copy failed");
        }
        DType::F16 => cast_to_f32_f16(
            out.as_mut_ptr() as *mut f32,
            logits.as_ptr() as *const u16,
            n,
            stream,
        ),
        DType::BF16 => cast_to_f32_bf16(
            out.as_mut_ptr() as *mut f32,
            logits.as_ptr() as *const u16,
            n,
            stream,
        ),
        _ => panic!("cast_logits_to_f32: unsupported dtype {:?}", logits.dtype()),
    }
    out
}

/// Cast a bias tensor in-place into a pre-allocated F32 GpuTensor.
/// `src` may be F16, BF16, or F32; `dst` must be F32 with the same numel.
/// Used by `DeepSeekV2MoELayer::load` to normalize the checkpoint dtype.
pub unsafe fn cast_bias_to_f32(src: GpuTensor, dst: GpuTensor, stream: CUstream) {
    let n = src.numel() as c_int;
    match src.dtype() {
        DType::F32 => crate::driver::memcpy_dtod_async(
            dst.raw_ptr(),
            src.as_ptr::<u8>(),
            src.size_bytes(),
            stream,
        )
        .expect("cast_bias_to_f32: D2D copy failed"),
        DType::F16 => cast_to_f32_f16(
            dst.as_ptr::<u8>() as *mut f32,
            src.as_ptr::<u16>(),
            n,
            stream,
        ),
        DType::BF16 => cast_to_f32_bf16(
            dst.as_ptr::<u8>() as *mut f32,
            src.as_ptr::<u16>(),
            n,
            stream,
        ),
        _ => panic!("cast_bias_to_f32: unsupported dtype {:?}", src.dtype()),
    }
}

// ---------------------------------------------------------------------------
// Cast from f32 (for GGML matmul output → model dtype)
// ---------------------------------------------------------------------------

unsafe extern "C" {
    fn cast_from_f32_f16(output: *mut u16, input: *const f32, n: c_int, stream: CUstream);
    fn cast_from_f32_bf16(output: *mut u16, input: *const f32, n: c_int, stream: CUstream);
}

/// Cast f32 tensor to target dtype on GPU. If target is f32, does a D2D copy.
pub unsafe fn cast_from_f32(
    input: GpuTensor,
    target_dtype: DType,
    alloc: &mut CachingAllocator,
    stream: CUstream,
) -> OwnedTensor {
    assert_eq!(
        input.dtype(),
        DType::F32,
        "cast_from_f32: input must be f32"
    );
    let n = input.numel() as c_int;
    let shape: Vec<usize> = input.shape().iter().map(|&d| d as usize).collect();
    let out = alloc.alloc_tensor(&shape, target_dtype);
    match target_dtype {
        DType::F32 => {
            crate::driver::memcpy_dtod_async(
                out.as_gpu_tensor().raw_ptr(),
                input.raw_ptr() as *const u8,
                input.size_bytes(),
                stream,
            )
            .expect("cast_from_f32: D2D copy failed");
        }
        DType::F16 => cast_from_f32_f16(
            out.as_gpu_tensor().raw_ptr() as *mut u16,
            input.as_ptr::<f32>() as *const f32,
            n,
            stream,
        ),
        DType::BF16 => cast_from_f32_bf16(
            out.as_gpu_tensor().raw_ptr() as *mut u16,
            input.as_ptr::<f32>() as *const f32,
            n,
            stream,
        ),
        _ => panic!("cast_from_f32: unsupported target dtype {:?}", target_dtype),
    }
    out
}

/// Cast f32 data into a pre-allocated buffer of target dtype.
///
/// # Safety
/// `input` must point to `n` f32 values. `output` must have room for `n` elements of `target_dtype`.
pub unsafe fn cast_from_f32_into(
    input: *const f32,
    output: *mut u8,
    target_dtype: DType,
    n: usize,
    stream: CUstream,
) {
    let n_i = n as c_int;
    match target_dtype {
        DType::F32 => {
            crate::driver::memcpy_dtod_async(output, input as *const u8, n * 4, stream)
                .expect("cast_from_f32_into: D2D copy failed");
        }
        DType::F16 => cast_from_f32_f16(output as *mut u16, input, n_i, stream),
        DType::BF16 => cast_from_f32_bf16(output as *mut u16, input, n_i, stream),
        _ => panic!(
            "cast_from_f32_into: unsupported target dtype {:?}",
            target_dtype
        ),
    }
}

// ---------------------------------------------------------------------------
// MoE top-k softmax
// ---------------------------------------------------------------------------

/// Top-k softmax gating for MoE.
///
/// * `gating_output`: `[num_tokens, num_experts]` — raw gate logits
/// * `topk`: number of experts to select per token
/// * `renormalize`: if true, renormalize selected weights to sum to 1
///
/// Returns `(topk_weights, topk_ids)`:
/// * `topk_weights`: `[num_tokens, topk]` (F32)
/// * `topk_ids`: `[num_tokens, topk]` (I32)
pub unsafe fn topk_softmax(
    gating_output: GpuTensor,
    topk: usize,
    renormalize: bool,
    alloc: &mut CachingAllocator,
    stream: CUstream,
) -> (OwnedTensor, OwnedTensor) {
    let num_tokens = gating_output.dim(0);
    let num_experts = gating_output.dim(1);

    let weights_out = alloc.alloc_tensor(&[num_tokens, topk], DType::F32);
    let ids_out = alloc.alloc_tensor(&[num_tokens, topk], DType::I32);
    // Workspace for fallback path (separate softmax + topK)
    let workspace = alloc.alloc_tensor(&[num_tokens, num_experts], DType::F32);

    match gating_output.dtype() {
        DType::F32 => topk_softmax_f32(
            weights_out.as_mut_ptr() as *mut f32,
            ids_out.as_mut_ptr() as *mut i32,
            workspace.as_mut_ptr() as *mut f32,
            gating_output.as_ptr() as *const f32,
            num_tokens as c_int,
            num_experts as c_int,
            topk as c_int,
            renormalize as c_int,
            stream,
        ),
        DType::BF16 => topk_softmax_bf16(
            weights_out.as_mut_ptr() as *mut f32,
            ids_out.as_mut_ptr() as *mut i32,
            workspace.as_mut_ptr() as *mut f32,
            gating_output.as_ptr() as *const c_void,
            num_tokens as c_int,
            num_experts as c_int,
            topk as c_int,
            renormalize as c_int,
            stream,
        ),
        DType::F16 => topk_softmax_f16(
            weights_out.as_mut_ptr() as *mut f32,
            ids_out.as_mut_ptr() as *mut i32,
            workspace.as_mut_ptr() as *mut f32,
            gating_output.as_ptr() as *const c_void,
            num_tokens as c_int,
            num_experts as c_int,
            topk as c_int,
            renormalize as c_int,
            stream,
        ),
        _ => panic!(
            "topk_softmax: unsupported dtype {:?}",
            gating_output.dtype()
        ),
    }
    drop(workspace);
    (weights_out, ids_out)
}

/// Sigmoid top-k with e_score_correction_bias — DeepSeek V3 / Kimi K2 routing.
///
/// Selection: `argmax(sigmoid(logit[e]) + bias[e])` (biased)
/// Output weight: `sigmoid(logit[e])` (unbiased), optionally renormalized.
///
/// * `gating_output`: `[num_tokens, num_experts]`
/// * `bias`: `[num_experts]` F32 correction bias
/// * `topk`, `renormalize`: same as `topk_softmax`
///
/// Returns `(topk_weights, topk_ids)`:
/// * `topk_weights`: `[num_tokens, topk]` F32 — unbiased sigmoid scores
/// * `topk_ids`: `[num_tokens, topk]` I32
pub unsafe fn topk_sigmoid_with_bias(
    gating_output: GpuTensor,
    bias: GpuTensor,
    topk: usize,
    renormalize: bool,
    alloc: &mut CachingAllocator,
    stream: CUstream,
) -> (OwnedTensor, OwnedTensor) {
    let num_tokens = gating_output.dim(0);
    let num_experts = gating_output.dim(1);

    let weights_out = alloc.alloc_tensor(&[num_tokens, topk], DType::F32);
    let ids_out = alloc.alloc_tensor(&[num_tokens, topk], DType::I32);
    let workspace = alloc.alloc_tensor(&[num_tokens, num_experts], DType::F32);

    let bias_ptr = bias.as_ptr() as *const f32;
    match gating_output.dtype() {
        DType::F32 => topk_sigmoid_f32(
            weights_out.as_mut_ptr() as *mut f32,
            ids_out.as_mut_ptr() as *mut i32,
            workspace.as_mut_ptr() as *mut f32,
            bias_ptr,
            gating_output.as_ptr() as *const f32,
            num_tokens as c_int,
            num_experts as c_int,
            topk as c_int,
            renormalize as c_int,
            stream,
        ),
        DType::BF16 => topk_sigmoid_bf16(
            weights_out.as_mut_ptr() as *mut f32,
            ids_out.as_mut_ptr() as *mut i32,
            workspace.as_mut_ptr() as *mut f32,
            bias_ptr,
            gating_output.as_ptr() as *const c_void,
            num_tokens as c_int,
            num_experts as c_int,
            topk as c_int,
            renormalize as c_int,
            stream,
        ),
        DType::F16 => topk_sigmoid_f16(
            weights_out.as_mut_ptr() as *mut f32,
            ids_out.as_mut_ptr() as *mut i32,
            workspace.as_mut_ptr() as *mut f32,
            bias_ptr,
            gating_output.as_ptr() as *const c_void,
            num_tokens as c_int,
            num_experts as c_int,
            topk as c_int,
            renormalize as c_int,
            stream,
        ),
        _ => panic!(
            "topk_sigmoid_with_bias: unsupported dtype {:?}",
            gating_output.dtype()
        ),
    }
    drop(workspace);
    (weights_out, ids_out)
}

// ---------------------------------------------------------------------------
// Grouped noaux_tc top-k (DeepSeek V3 / Kimi K2)
// ---------------------------------------------------------------------------

/// Grouped sigmoid top-k with `noaux_tc` algorithm (DeepSeek V3 / Kimi K2).
///
/// Implements group-based expert selection:
///   1. Apply sigmoid to gating logits.
///   2. Per group: compute group score = sum of top-2 (sigmoid + bias) values.
///   3. Select top `topk_group` groups.
///   4. Within selected groups, select top `topk` experts by (sigmoid + bias).
///   5. Output weights = unbiased sigmoid scores, optionally renormalized and
///      scaled by `routed_scaling_factor`.
///
/// * `gating_output`: `[num_tokens, num_experts]` BF16/F16/F32
/// * `bias`: `[num_experts]` F32 e_score_correction_bias
/// * `n_expert_group`: number of expert groups (e.g. 8 for V3)
/// * `topk_group`: number of groups to select (e.g. 4 for V3)
///
/// Returns `(topk_weights [num_tokens, topk] F32, topk_ids [num_tokens, topk] I32)`.
pub unsafe fn topk_noaux_tc(
    gating_output: GpuTensor,
    bias: GpuTensor,
    topk: usize,
    n_expert_group: usize,
    topk_group: usize,
    renormalize: bool,
    routed_scaling_factor: f64,
    alloc: &mut CachingAllocator,
    stream: CUstream,
) -> (OwnedTensor, OwnedTensor) {
    let num_tokens = gating_output.dim(0);
    let num_experts = gating_output.dim(1);

    let weights_out = alloc.alloc_tensor(&[num_tokens, topk], DType::F32);
    let ids_out = alloc.alloc_tensor(&[num_tokens, topk], DType::I32);

    // Bias must be F32; cast on the fly if the checkpoint stored it in BF16/F16.
    let bias_f32_tmp: OwnedTensor;
    let bias_f32: GpuTensor = if bias.dtype() == DType::F32 {
        bias
    } else {
        let n = bias.numel();
        bias_f32_tmp = alloc.alloc_tensor(&[n], DType::F32);
        cast_bias_to_f32(bias, bias_f32_tmp.as_gpu_tensor(), stream);
        bias_f32_tmp.as_gpu_tensor()
    };

    let scores_ptr = gating_output.as_ptr::<u8>() as *mut c_void;
    let bias_ptr = bias_f32.as_ptr::<u8>() as *const c_void;

    let weights_ptr = weights_out.as_mut_ptr::<f32>();
    let ids_ptr = ids_out.as_mut_ptr::<i32>();

    match gating_output.dtype() {
        DType::F32 => topk_noaux_tc_f32_f32(
            scores_ptr,
            weights_ptr,
            ids_ptr,
            bias_ptr,
            num_tokens as i64,
            num_experts as i64,
            n_expert_group as i64,
            topk_group as i64,
            topk as i64,
            renormalize as c_int,
            routed_scaling_factor,
            stream,
        ),
        DType::BF16 => topk_noaux_tc_bf16_f32(
            scores_ptr,
            weights_ptr,
            ids_ptr,
            bias_ptr,
            num_tokens as i64,
            num_experts as i64,
            n_expert_group as i64,
            topk_group as i64,
            topk as i64,
            renormalize as c_int,
            routed_scaling_factor,
            stream,
        ),
        DType::F16 => topk_noaux_tc_f16_f32(
            scores_ptr,
            weights_ptr,
            ids_ptr,
            bias_ptr,
            num_tokens as i64,
            num_experts as i64,
            n_expert_group as i64,
            topk_group as i64,
            topk as i64,
            renormalize as c_int,
            routed_scaling_factor,
            stream,
        ),
        _ => panic!(
            "topk_noaux_tc: unsupported dtype {:?}",
            gating_output.dtype()
        ),
    }

    (weights_out, ids_out)
}

// ---------------------------------------------------------------------------
// MoE sum reduction
// ---------------------------------------------------------------------------

/// Reduce expert outputs: `[num_tokens, topk, hidden] → [num_tokens, hidden]`.
///
/// * `input`: `[num_tokens, topk, hidden_size]`
/// * `topk`: number of experts per token
///
/// Returns `[num_tokens, hidden_size]`.
pub unsafe fn moe_sum(
    input: GpuTensor,
    num_tokens: usize,
    hidden_size: usize,
    topk: usize,
    alloc: &mut CachingAllocator,
    stream: CUstream,
) -> OwnedTensor {
    let out = alloc.alloc_tensor(&[num_tokens, hidden_size], input.dtype());

    match input.dtype() {
        DType::F32 => moe_sum_f32(
            out.as_mut_ptr() as *mut f32,
            input.as_ptr() as *const f32,
            num_tokens as c_int,
            hidden_size as c_int,
            topk as c_int,
            stream,
        ),
        DType::F16 => moe_sum_f16(
            out.as_mut_ptr() as *mut c_void,
            input.as_ptr() as *const c_void,
            num_tokens as c_int,
            hidden_size as c_int,
            topk as c_int,
            stream,
        ),
        DType::BF16 => moe_sum_bf16(
            out.as_mut_ptr() as *mut c_void,
            input.as_ptr() as *const c_void,
            num_tokens as c_int,
            hidden_size as c_int,
            topk as c_int,
            stream,
        ),
        _ => panic!("moe_sum: unsupported dtype {:?}", input.dtype()),
    }
    out
}

// ---------------------------------------------------------------------------
// Penalties / Logit bias / Grammar mask / Log-softmax top-k
// ---------------------------------------------------------------------------

unsafe extern "C" {
    fn apply_penalties_inplace(
        logits: *mut f32,
        output_token_ids: *const c_int,
        prompt_token_ids: *const c_int,
        rep_penalties: *const f32,
        freq_penalties: *const f32,
        pres_penalties: *const f32,
        vocab_size: c_int,
        batch_size: c_int,
        max_output_len: c_int,
        max_prompt_len: c_int,
        stream: CUstream,
    );

    fn apply_logit_bias_inplace(
        logits: *mut f32,
        bias_token_ids: *const c_int,
        bias_values: *const f32,
        bias_offsets: *const c_int,
        vocab_size: c_int,
        batch_size: c_int,
        stream: CUstream,
    );

    fn apply_grammar_mask_inplace(
        logits: *mut f32,
        logits_backup: *const f32,
        allowed_ids: *const c_int,
        allowed_offsets: *const c_int,
        req_indices: *const c_int,
        vocab_size: c_int,
        num_grammar_reqs: c_int,
        stream: CUstream,
    );

    fn log_softmax_topk(
        logits: *const f32,
        sampled_ids: *const u32,
        vocab_size: c_int,
        batch_size: c_int,
        num_logprobs: c_int,
        out_logprobs: *mut f32,
        out_indices: *mut c_int,
        out_ranks: *mut u32,
        stream: CUstream,
    );

    fn apply_min_tokens_inplace(
        logits: *mut f32,
        req_indices: *const c_int,
        token_ids: *const c_int,
        count: c_int,
        vocab_size: c_int,
        stream: CUstream,
    );
}

/// Apply repetition, frequency, and presence penalties to f32 logits in-place.
///
/// * `logits`: `[batch_size, vocab_size]` f32 on GPU — modified in-place
/// * `output_token_ids`: `[batch_size, max_output_len]` i32 on GPU, padded with `vocab_size`
/// * `prompt_token_ids`: `[batch_size, max_prompt_len]` i32 on GPU, padded with `vocab_size`
/// * `rep_penalties`, `freq_penalties`, `pres_penalties`: `[batch_size]` f32 on GPU
#[allow(clippy::too_many_arguments)]
pub unsafe fn apply_penalties(
    logits: GpuTensor,
    output_token_ids: GpuTensor,
    prompt_token_ids: GpuTensor,
    rep_penalties: GpuTensor,
    freq_penalties: GpuTensor,
    pres_penalties: GpuTensor,
    stream: CUstream,
) {
    let batch_size = logits.dim(0) as c_int;
    let vocab_size = logits.dim(1) as c_int;
    let max_output_len = output_token_ids.dim(1) as c_int;
    let max_prompt_len = prompt_token_ids.dim(1) as c_int;

    apply_penalties_inplace(
        logits.as_mut_ptr() as *mut f32,
        output_token_ids.as_ptr() as *const c_int,
        prompt_token_ids.as_ptr() as *const c_int,
        rep_penalties.as_ptr(),
        freq_penalties.as_ptr(),
        pres_penalties.as_ptr(),
        vocab_size,
        batch_size,
        max_output_len,
        max_prompt_len,
        stream,
    );
}

/// Apply sparse logit bias via CSR-packed scatter-add on f32 logits in-place.
///
/// * `logits`: `[batch_size, vocab_size]` f32 on GPU
/// * `bias_token_ids`: `[total_biases]` i32 on GPU
/// * `bias_values`: `[total_biases]` f32 on GPU
/// * `bias_offsets`: `[batch_size + 1]` i32 on GPU (CSR offsets)
pub unsafe fn apply_logit_bias(
    logits: GpuTensor,
    bias_token_ids: GpuTensor,
    bias_values: GpuTensor,
    bias_offsets: GpuTensor,
    stream: CUstream,
) {
    let batch_size = logits.dim(0) as c_int;
    let vocab_size = logits.dim(1) as c_int;

    apply_logit_bias_inplace(
        logits.as_mut_ptr() as *mut f32,
        bias_token_ids.as_ptr() as *const c_int,
        bias_values.as_ptr(),
        bias_offsets.as_ptr() as *const c_int,
        vocab_size,
        batch_size,
        stream,
    );
}

/// Apply grammar mask: set disallowed tokens to -inf using CSR-packed allow-list.
///
/// * `logits`: `[batch_size, vocab_size]` f32 on GPU — modified in-place
/// * `logits_backup`: `[batch_size, vocab_size]` f32 on GPU — pristine copy for restoring allowed tokens
/// * `allowed_ids`: `[total_allowed]` i32 on GPU
/// * `allowed_offsets`: `[num_grammar_reqs + 1]` i32 on GPU
/// * `req_indices`: `[num_grammar_reqs]` i32 on GPU — maps grammar index to batch row
pub unsafe fn apply_grammar_mask(
    logits: GpuTensor,
    logits_backup: GpuTensor,
    allowed_ids: GpuTensor,
    allowed_offsets: GpuTensor,
    req_indices: GpuTensor,
    stream: CUstream,
) {
    let vocab_size = logits.dim(1) as c_int;
    let num_grammar_reqs = req_indices.dim(0) as c_int;

    apply_grammar_mask_inplace(
        logits.as_mut_ptr() as *mut f32,
        logits_backup.as_ptr(),
        allowed_ids.as_ptr() as *const c_int,
        allowed_offsets.as_ptr() as *const c_int,
        req_indices.as_ptr() as *const c_int,
        vocab_size,
        num_grammar_reqs,
        stream,
    );
}

/// Suppress stop tokens for requests below min_tokens by setting logits to -inf.
///
/// * `logits`: `[batch_size, vocab_size]` f32 on GPU — modified in-place
/// * `req_indices`: `[count]` i32 on GPU — batch index for each (req, stop_token) pair
/// * `token_ids`: `[count]` i32 on GPU — token ID to suppress for each pair
pub unsafe fn apply_min_tokens(
    logits: GpuTensor,
    req_indices: GpuTensor,
    token_ids: GpuTensor,
    stream: CUstream,
) {
    let count = req_indices.dim(0) as c_int;
    let vocab_size = logits.dim(1) as c_int;

    apply_min_tokens_inplace(
        logits.as_mut_ptr() as *mut f32,
        req_indices.as_ptr() as *const c_int,
        token_ids.as_ptr() as *const c_int,
        count,
        vocab_size,
        stream,
    );
}

/// Fused log-softmax + top-k on f32 logits. Returns per-request top-k logprobs.
///
/// * `logits`: `[batch_size, vocab_size]` f32 on GPU (raw logits before penalties)
/// * `sampled_ids`: `[batch_size]` u32 on GPU
/// * `num_logprobs`: number of top logprobs to return (slot 0 = sampled token)
///
/// Returns `(logprobs, indices, ranks)`:
/// * `logprobs`: `[batch_size, num_logprobs + 1]` f32
/// * `indices`: `[batch_size, num_logprobs + 1]` i32
/// * `ranks`: `[batch_size, num_logprobs + 1]` u32
pub unsafe fn log_softmax_topk_gather(
    logits: GpuTensor,
    sampled_ids: GpuTensor,
    num_logprobs: usize,
    alloc: &mut CachingAllocator,
    stream: CUstream,
) -> (OwnedTensor, OwnedTensor, OwnedTensor) {
    let batch_size = logits.dim(0);
    let vocab_size = logits.dim(1) as c_int;
    let k = num_logprobs + 1;

    let out_lp = alloc.alloc_tensor(&[batch_size, k], DType::F32);
    let out_idx = alloc.alloc_tensor(&[batch_size, k], DType::U32);
    let out_ranks = alloc.alloc_tensor(&[batch_size, k], DType::U32);

    log_softmax_topk(
        logits.as_ptr(),
        sampled_ids.as_ptr() as *const u32,
        vocab_size,
        batch_size as c_int,
        num_logprobs as c_int,
        out_lp.as_mut_ptr() as *mut f32,
        out_idx.as_mut_ptr() as *mut c_int,
        out_ranks.as_mut_ptr() as *mut u32,
        stream,
    );

    (out_lp, out_idx, out_ranks)
}

// ---------------------------------------------------------------------------
// Gemma-4 routing: softmax-over-topk-with-temp + per_expert_scale
// ---------------------------------------------------------------------------

unsafe extern "C" {
    fn moe_softmax_topk_temp_bf16(
        scores: *mut f32,
        logits: *const c_void,
        ids: *const i32,
        num_tokens: c_int,
        num_experts: c_int,
        top_k: c_int,
        temp: f32,
        stream: CUstream,
    );
    fn moe_softmax_topk_temp_f16(
        scores: *mut f32,
        logits: *const c_void,
        ids: *const i32,
        num_tokens: c_int,
        num_experts: c_int,
        top_k: c_int,
        temp: f32,
        stream: CUstream,
    );
    fn moe_softmax_topk_temp_f32(
        scores: *mut f32,
        logits: *const c_void,
        ids: *const i32,
        num_tokens: c_int,
        num_experts: c_int,
        top_k: c_int,
        temp: f32,
        stream: CUstream,
    );
    fn moe_per_expert_scale_f32(
        scores: *mut f32,
        ids: *const i32,
        per_expert_scale: *const c_void,
        n: c_int,
        stream: CUstream,
    );
}

/// Gemma-4 routing softmax: softmax over the gathered top-k logits with a
/// `temp` multiply folded in before exp (`temp = hidden^-0.5`).
///
/// * `logits`: `[num_tokens, num_experts]` (bf16/f16/f32) — the raw router GEMM
///   output.
/// * `ids`: `[num_tokens, top_k]` I32 — the selected expert ids.
/// * Returns `[num_tokens, top_k]` F32 routing scores (pre per-expert-scale).
pub unsafe fn moe_softmax_topk_temp(
    logits: GpuTensor,
    ids: GpuTensor,
    top_k: usize,
    temp: f32,
    alloc: &mut CachingAllocator,
    stream: CUstream,
) -> OwnedTensor {
    let num_tokens = logits.dim(0);
    let num_experts = logits.dim(1);
    let scores = alloc.alloc_tensor(&[num_tokens, top_k], DType::F32);
    let f = match logits.dtype() {
        DType::BF16 => moe_softmax_topk_temp_bf16,
        DType::F16 => moe_softmax_topk_temp_f16,
        DType::F32 => moe_softmax_topk_temp_f32,
        dt => panic!("moe_softmax_topk_temp: unsupported logits dtype {dt:?}"),
    };
    f(
        scores.as_mut_ptr() as *mut f32,
        logits.as_ptr() as *const c_void,
        ids.as_ptr() as *const i32,
        num_tokens as c_int,
        num_experts as c_int,
        top_k as c_int,
        temp,
        stream,
    );
    scores
}

/// Gemma-4 per-expert score scale: `scores[g] *= per_expert_scale[ids[g]]`,
/// applied in place over the flat `[num_tokens * top_k]` routing scores.
///
/// * `scores`: `[num_tokens, top_k]` F32 (in place).
/// * `ids`: `[num_tokens, top_k]` I32.
/// * `per_expert_scale`: `[num_experts]` BF16.
pub unsafe fn moe_per_expert_scale(
    scores: GpuTensor,
    ids: GpuTensor,
    per_expert_scale: GpuTensor,
    stream: CUstream,
) {
    // GUARD (hard, release-included): the kernel reads `per_expert_scale` as BF16.
    // A mismatched dtype (e.g. the FP16 the cyankiwi checkpoint ships) would be
    // reinterpreted bit-for-bit and silently corrupt every routed score with no
    // crash — `debug_assert` would miss it in the release builds we actually run.
    assert_eq!(
        per_expert_scale.dtype(),
        DType::BF16,
        "moe_per_expert_scale: per_expert_scale must be BF16 (got {:?}); the kernel \
         reads it as BF16 and would misread any other dtype.",
        per_expert_scale.dtype()
    );
    let n = scores.numel();
    moe_per_expert_scale_f32(
        scores.as_mut_ptr() as *mut f32,
        ids.as_ptr() as *const i32,
        per_expert_scale.as_ptr() as *const c_void,
        n as c_int,
        stream,
    );
}

// ---------------------------------------------------------------------------
// MoE align block size
// ---------------------------------------------------------------------------

/// Align MoE token assignments to GEMM block boundaries.
///
/// Sorts tokens by expert and pads to `block_size` alignment.
///
/// * `topk_ids`: `[num_tokens * topk]` (I32) — expert assignments
/// * `num_experts`: total number of experts
/// * `block_size`: GEMM tile size (e.g. 128)
///
/// Returns `(sorted_token_ids, expert_ids, num_tokens_post_padded)`.
pub unsafe fn moe_align_block_size(
    topk_ids: GpuTensor,
    num_experts: usize,
    block_size: usize,
    alloc: &mut CachingAllocator,
    stream: CUstream,
) -> (OwnedTensor, OwnedTensor, OwnedTensor) {
    let numel = topk_ids.numel();
    // Max padded size: each expert's tokens padded to block_size
    let max_num_tokens_padded = numel + num_experts * block_size;
    let max_num_m_blocks = max_num_tokens_padded.div_ceil(block_size);

    let sorted_token_ids = alloc.alloc_tensor(&[max_num_tokens_padded], DType::I32);
    let expert_ids = alloc.alloc_tensor(&[max_num_m_blocks], DType::I32);
    let num_tokens_post_pad = alloc.alloc_tensor(&[1], DType::I32);

    moe_align_block_size_i32(
        topk_ids.as_ptr() as *const i32,
        sorted_token_ids.as_mut_ptr() as *mut i32,
        expert_ids.as_mut_ptr() as *mut i32,
        num_tokens_post_pad.as_mut_ptr() as *mut i32,
        num_experts as c_int,
        block_size as c_int,
        numel as c_int,
        max_num_tokens_padded as c_int,
        stream,
    );

    (sorted_token_ids, expert_ids, num_tokens_post_pad)
}

// ---------------------------------------------------------------------------
// Fused MoE GEMM
// ---------------------------------------------------------------------------

/// Fused MoE GEMM: expert-indexed matrix multiplication.
///
/// * `input`: `[num_tokens, in_features]` — hidden states
/// * `weights`: `[num_experts, out_features, in_features]` — stacked expert weights
/// * `topk_weights`: `[num_tokens, top_k]` (F32) — routing weights
/// * `sorted_token_ids`: from `moe_align_block_size`
/// * `expert_ids`: from `moe_align_block_size`
/// * `num_tokens_post_padded`: from `moe_align_block_size`
/// * `block_m`: GEMM M-tile size (16/32/64/128) — must match the block_size used in `moe_align_block_size`
/// * `apply_weights`: if true, multiply output by routing weight
///
/// Returns `[num_tokens * top_k, out_features]`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn fused_moe_gemm(
    input: GpuTensor,
    weights: GpuTensor,
    topk_weights: GpuTensor,
    sorted_token_ids: GpuTensor,
    expert_ids: GpuTensor,
    num_tokens_post_padded: GpuTensor,
    num_tokens: usize,
    top_k: usize,
    block_m: usize,
    apply_weights: bool,
    alloc: &mut CachingAllocator,
    stream: CUstream,
) -> OwnedTensor {
    let in_features = input.dim(1);
    let out_features = weights.dim(1);
    let num_valid_tokens = num_tokens * top_k;
    let num_tokens_padded_total = sorted_token_ids.numel();

    let out = alloc.alloc_tensor(&[num_valid_tokens, out_features], input.dtype());

    match input.dtype() {
        DType::BF16 => fused_moe_gemm_bf16(
            out.as_mut_ptr() as *mut c_void,
            input.as_ptr() as *const c_void,
            weights.as_ptr() as *const c_void,
            topk_weights.as_ptr() as *const f32,
            sorted_token_ids.as_ptr() as *const i32,
            expert_ids.as_ptr() as *const i32,
            num_tokens_post_padded.as_ptr() as *const i32,
            num_valid_tokens as c_int,
            in_features as c_int,
            out_features as c_int,
            top_k as c_int,
            num_tokens_padded_total as c_int,
            apply_weights as c_int,
            block_m as c_int,
            stream,
        ),
        DType::F16 => fused_moe_gemm_f16(
            out.as_mut_ptr() as *mut c_void,
            input.as_ptr() as *const c_void,
            weights.as_ptr() as *const c_void,
            topk_weights.as_ptr() as *const f32,
            sorted_token_ids.as_ptr() as *const i32,
            expert_ids.as_ptr() as *const i32,
            num_tokens_post_padded.as_ptr() as *const i32,
            num_valid_tokens as c_int,
            in_features as c_int,
            out_features as c_int,
            top_k as c_int,
            num_tokens_padded_total as c_int,
            apply_weights as c_int,
            block_m as c_int,
            stream,
        ),
        _ => panic!("fused_moe_gemm: unsupported dtype {:?}", input.dtype()),
    }
    out
}

// ---------------------------------------------------------------------------
// CUTLASS grouped MoE GEMM (arch-optimal: Sm90 wgmma+TMA / Sm80 mma.sync)
// ---------------------------------------------------------------------------

/// CUTLASS grouped expert GEMM — drop-in replacement for `fused_moe_gemm`.
///
/// Same contract as `fused_moe_gemm`:
/// * `input`: `[num_tokens, in_features]` BF16
/// * `weights`: `[num_experts, out_features, in_features]` BF16 (dequant'd)
/// * `topk_weights`: `[num_valid_tokens]` F32 routing weights
/// * the `sorted_token_ids`/`expert_ids`/`num_tokens_post_padded` triple from
///   `moe_align_block_size` (block_size == `block_m`)
/// * `apply_weights`: multiply each output row by its routing weight
///
/// Returns `[num_tokens * top_k, out_features]` BF16, indexed by token slot —
/// byte-identical layout to `fused_moe_gemm`, so the caller's `gelu_and_mul` /
/// `moe_sum` stay unchanged.
///
/// Pipeline: permute (gather A grouped) → marshal (per-expert ptr/problem/stride
/// arrays) → grouped GEMM (arch-dispatched) → scatter (token-indexed write).
///
/// # Safety
/// All tensors must be valid GPU memory; an active CUDA context/stream.
#[allow(clippy::too_many_arguments)]
pub unsafe fn cutlass_moe_grouped(
    input: GpuTensor,
    weights: GpuTensor,
    topk_weights: GpuTensor,
    sorted_token_ids: GpuTensor,
    expert_ids: GpuTensor,
    num_tokens_post_padded: GpuTensor,
    num_experts: usize,
    num_tokens: usize,
    top_k: usize,
    block_m: usize,
    apply_weights: bool,
    alloc: &mut CachingAllocator,
    stream: CUstream,
) -> OwnedTensor {
    debug_assert_eq!(input.dtype(), DType::BF16, "cutlass_moe_grouped: BF16 only");
    let in_features = input.dim(1); // K
    let out_features = weights.dim(1); // N
    let num_valid_tokens = num_tokens * top_k;
    // Worst-case padded rows = sorted_token_ids length (grid cap; the kernels
    // early-out past the device-read total_tokens_post_padded).
    let total_padded_cap = sorted_token_ids.numel();

    // Output: token-indexed [num_valid_tokens, N] (same as fused_moe_gemm).
    let out = alloc.alloc_tensor(&[num_valid_tokens, out_features], DType::BF16);

    // Contiguous expert-grouped A_permuted and C_permuted scratch.
    let a_permuted = alloc.alloc_tensor(&[total_padded_cap, in_features], DType::BF16);
    let c_permuted = alloc.alloc_tensor(&[total_padded_cap, out_features], DType::BF16);

    // Per-group (per-expert) marshaling arrays.
    let problem_sizes = alloc.alloc_tensor(&[3 * num_experts], DType::I32);
    let ptr_a = alloc.alloc_tensor(&[num_experts], DType::I64);
    let ptr_b = alloc.alloc_tensor(&[num_experts], DType::I64);
    let ptr_c = alloc.alloc_tensor(&[num_experts], DType::I64);
    let lda = alloc.alloc_tensor(&[num_experts], DType::I64);
    let ldb = alloc.alloc_tensor(&[num_experts], DType::I64);
    let ldc = alloc.alloc_tensor(&[num_experts], DType::I64);

    // 1. Gather A_permuted (expert-grouped, padding rows zeroed).
    scratchy_moe_permute_gather_a(
        a_permuted.as_mut_ptr() as *mut c_void,
        input.as_ptr() as *const c_void,
        sorted_token_ids.as_ptr() as *const i32,
        num_tokens_post_padded.as_ptr() as *const i32,
        total_padded_cap as c_int,
        num_valid_tokens as c_int,
        top_k as c_int,
        in_features as c_int,
        stream,
    );

    // 2. Build per-expert ptr/problem/stride arrays.
    scratchy_moe_marshal_groups(
        problem_sizes.as_mut_ptr() as *mut i32,
        ptr_a.as_mut_ptr() as *mut *mut c_void,
        ptr_b.as_mut_ptr() as *mut *mut c_void,
        ptr_c.as_mut_ptr() as *mut *mut c_void,
        lda.as_mut_ptr() as *mut i64,
        ldb.as_mut_ptr() as *mut i64,
        ldc.as_mut_ptr() as *mut i64,
        expert_ids.as_ptr() as *const i32,
        num_tokens_post_padded.as_ptr() as *const i32,
        a_permuted.as_mut_ptr() as *mut c_void,
        weights.as_ptr() as *const c_void,
        c_permuted.as_mut_ptr() as *mut c_void,
        num_experts as c_int,
        block_m as c_int,
        out_features as c_int,
        in_features as c_int,
        stream,
    );

    // 3. Arch-dispatched grouped GEMM into C_permuted.
    let sm = device_sm_version();
    if (90..100).contains(&sm) {
        // Sm90: build cute packed strides, then ptr-array warp-specialized GEMM.
        let stride_bytes = cutlass_moe_grouped_sm90_stride_bytes() as usize;
        let stride_a = alloc.alloc_tensor(&[num_experts * stride_bytes], DType::U8);
        let stride_b = alloc.alloc_tensor(&[num_experts * stride_bytes], DType::U8);
        let stride_c = alloc.alloc_tensor(&[num_experts * stride_bytes], DType::U8);
        let stride_d = alloc.alloc_tensor(&[num_experts * stride_bytes], DType::U8);
        cutlass_moe_grouped_sm90_fill_strides(
            stride_a.as_mut_ptr() as *mut c_void,
            stride_b.as_mut_ptr() as *mut c_void,
            stride_c.as_mut_ptr() as *mut c_void,
            stride_d.as_mut_ptr() as *mut c_void,
            problem_sizes.as_ptr() as *const i32,
            num_experts as c_int,
            stream,
        );
        let sm_count = crate::driver::current_device()
            .and_then(|d| crate::driver::device_get_num_sm(d))
            .unwrap_or(0);
        let rc = cutlass_moe_grouped_bf16_sm90(
            problem_sizes.as_ptr() as *const c_void,
            ptr_a.as_mut_ptr() as *mut *mut c_void,
            ptr_b.as_mut_ptr() as *mut *mut c_void,
            ptr_c.as_mut_ptr() as *mut *mut c_void,
            stride_a.as_mut_ptr() as *mut c_void,
            stride_b.as_mut_ptr() as *mut c_void,
            stride_c.as_mut_ptr() as *mut c_void,
            stride_d.as_mut_ptr() as *mut c_void,
            num_experts as c_int,
            sm_count as c_int,
            std::ptr::null_mut(),
            stream,
        );
        assert_eq!(rc, 0, "cutlass_moe_grouped_bf16_sm90 failed (rc={rc})");
    } else if sm == 89 || sm == 80 || sm == 86 {
        let tb = cutlass_moe_grouped_bf16_sm80_threadblock_count();
        let rc = cutlass_moe_grouped_bf16_sm80(
            problem_sizes.as_ptr() as *const c_void,
            ptr_a.as_mut_ptr() as *mut *mut c_void,
            ptr_b.as_mut_ptr() as *mut *mut c_void,
            ptr_c.as_mut_ptr() as *mut *mut c_void,
            lda.as_mut_ptr() as *mut i64,
            ldb.as_mut_ptr() as *mut i64,
            ldc.as_mut_ptr() as *mut i64,
            num_experts as c_int,
            tb,
            std::ptr::null_mut(),
            stream,
        );
        assert_eq!(rc, 0, "cutlass_moe_grouped_bf16_sm80 failed (rc={rc})");
    } else {
        panic!(
            "cutlass_moe_grouped: no compiled grouped GEMM for SM{sm} \
             (need SM80/86/89 or SM90)"
        );
    }

    // 4. Scatter C_permuted to token-indexed output (apply routing weights).
    scratchy_moe_scatter_c(
        out.as_mut_ptr() as *mut c_void,
        c_permuted.as_ptr() as *const c_void,
        topk_weights.as_ptr() as *const f32,
        sorted_token_ids.as_ptr() as *const i32,
        num_tokens_post_padded.as_ptr() as *const i32,
        total_padded_cap as c_int,
        num_valid_tokens as c_int,
        apply_weights as c_int,
        out_features as c_int,
        stream,
    );

    out
}

// ---------------------------------------------------------------------------
// MoE expert gather (graph-capture-safe unfused path)
// ---------------------------------------------------------------------------

/// Gather top_k experts' outputs from an all-expert buffer.
///
/// * `all_expert_out`: `[num_experts * num_tokens, out_features]` (BF16)
/// * `topk_ids`: `[num_tokens, top_k]` (I32) — selected expert IDs per token
///
/// Returns `[num_tokens * top_k, out_features]` (BF16).
#[allow(clippy::too_many_arguments)]
pub unsafe fn moe_expert_gather(
    all_expert_out: GpuTensor,
    topk_ids: GpuTensor,
    num_tokens: usize,
    num_experts: usize,
    top_k: usize,
    out_features: usize,
    alloc: &mut CachingAllocator,
    stream: CUstream,
) -> OwnedTensor {
    debug_assert_eq!(all_expert_out.dim(0), num_experts * num_tokens);
    debug_assert_eq!(all_expert_out.dim(1), out_features);
    let out = alloc.alloc_tensor(&[num_tokens * top_k, out_features], DType::BF16);
    moe_expert_gather_bf16(
        out.raw_ptr() as *mut c_void,
        all_expert_out.raw_ptr() as *const c_void,
        topk_ids.as_ptr::<i32>(),
        num_tokens as c_int,
        top_k as c_int,
        out_features as c_int,
        stream,
    );
    out
}

/// Gather top_k experts' w2 outputs, apply routing weights, and sum across top_k.
///
/// * `all_expert_out`: `[num_experts * rows_per_expert, out_features]` (BF16)
/// * `topk_ids`: `[num_tokens * top_k]` (I32) — flat
/// * `topk_weights`: `[num_tokens * top_k]` (F32) — flat
///
/// Returns `[num_tokens, out_features]` (BF16).
#[allow(clippy::too_many_arguments)]
pub unsafe fn moe_expert_gather_weighted_sum(
    all_expert_out: GpuTensor,
    topk_ids: GpuTensor,
    topk_weights: GpuTensor,
    num_tokens: usize,
    top_k: usize,
    out_features: usize,
    rows_per_expert: usize,
    alloc: &mut CachingAllocator,
    stream: CUstream,
) -> OwnedTensor {
    let out = alloc.alloc_tensor(&[num_tokens, out_features], DType::BF16);
    moe_expert_gather_weighted_sum_bf16(
        out.raw_ptr() as *mut c_void,
        all_expert_out.raw_ptr() as *const c_void,
        topk_ids.as_ptr::<i32>(),
        topk_weights.as_ptr::<f32>(),
        num_tokens as c_int,
        top_k as c_int,
        out_features as c_int,
        rows_per_expert as c_int,
        stream,
    );
    out
}

// ---------------------------------------------------------------------------
// FP8 Fused MoE GEMM
// ---------------------------------------------------------------------------

/// FP8 E4M3 fused MoE GEMM with per-token/expert scale application.
///
/// * `input`: `[num_tokens, in_features]` — FP8 E4M3 hidden states (already quantized)
/// * `weights`: `[num_experts, out_features, in_features]` — FP8 E4M3 stacked expert weights
/// * `a_scales`: `[num_tokens]` — f32 per-token activation scales
/// * `w_scales`: `[num_experts]` — f32 per-expert weight scales
/// * `topk_weights`: `[num_tokens, top_k]` (F32) — routing weights
/// * `sorted_token_ids`, `expert_ids`, `num_tokens_post_padded`: from `moe_align_block_size`
/// * `apply_weights`: if true, multiply output by routing weight
/// * `sm_version`: GPU SM version (e.g. 89 for L40S). >= 89 uses FP8 tensor cores.
///
/// Returns `[num_tokens * top_k, out_features]` BF16.
#[allow(clippy::too_many_arguments)]
pub unsafe fn fused_moe_fp8_gemm(
    input: GpuTensor,    // FP8 E4M3
    weights: GpuTensor,  // FP8 E4M3
    a_scales: GpuTensor, // f32
    w_scales: GpuTensor, // f32
    topk_weights: GpuTensor,
    sorted_token_ids: GpuTensor,
    expert_ids: GpuTensor,
    num_tokens_post_padded: GpuTensor,
    num_tokens: usize,
    top_k: usize,
    block_m: usize,
    apply_weights: bool,
    sm_version: u32,
    alloc: &mut CachingAllocator,
    stream: CUstream,
) -> OwnedTensor {
    let in_features = input.dim(1);
    let out_features = weights.dim(1);

    let out = alloc.alloc_tensor(&[num_tokens * top_k, out_features], DType::BF16);

    // Use dequant path for all SM versions.
    // SM89+ PTX FP8 tensor core path has a fragment layout bug (0.5x output);
    // dequant path gives correct results and still gets 2x bandwidth from FP8 storage.
    // TODO: fix SM89 PTX path for additional compute throughput.
    let _ = sm_version;
    let ffi_fn = fused_moe_fp8_gemm_dequant;

    ffi_fn(
        out.as_mut_ptr() as *mut c_void,
        input.as_ptr() as *const c_void,
        weights.as_ptr() as *const c_void,
        a_scales.as_ptr() as *const f32,
        w_scales.as_ptr() as *const f32,
        topk_weights.as_ptr() as *const f32,
        sorted_token_ids.as_ptr() as *const i32,
        expert_ids.as_ptr() as *const i32,
        num_tokens_post_padded.as_ptr() as *const i32,
        num_tokens as c_int,
        in_features as c_int,
        out_features as c_int,
        top_k as c_int,
        apply_weights as c_int,
        block_m as c_int,
        stream,
    );
    out
}

// ---------------------------------------------------------------------------
// FP8 Block-Quantized Fused MoE GEMM
// ---------------------------------------------------------------------------

/// FP8 block-quantized fused MoE GEMM with per-block weight scales.
///
/// Same as `fused_moe_fp8_gemm` but weight scales are 3D block scales
/// `[E, ceil(N/block_n), ceil(K/block_k)]` applied during FP8→BF16 dequant.
#[allow(clippy::too_many_arguments)]
pub unsafe fn fused_moe_fp8_block_gemm(
    input: GpuTensor,       // FP8 E4M3
    weights: GpuTensor,     // FP8 E4M3
    a_scales: GpuTensor,    // f32 per-token
    w_scale_inv: GpuTensor, // f32 [E, ceil(N/bn), ceil(K/bk)]
    topk_weights: GpuTensor,
    sorted_token_ids: GpuTensor,
    expert_ids: GpuTensor,
    num_tokens_post_padded: GpuTensor,
    num_tokens: usize,
    top_k: usize,
    block_m: usize,
    apply_weights: bool,
    block_size: [usize; 2], // [block_n, block_k]
    alloc: &mut CachingAllocator,
    stream: CUstream,
) -> OwnedTensor {
    let in_features = input.dim(1);
    let out_features = weights.dim(1);

    let out = alloc.alloc_tensor(&[num_tokens * top_k, out_features], DType::BF16);

    let block_n = block_size[0];
    let block_k = block_size[1];
    let scale_cols = in_features.div_ceil(block_k);
    let scale_rows = out_features.div_ceil(block_n);
    let scale_stride_e = scale_rows * scale_cols;
    let scale_stride_n = scale_cols;

    fused_moe_fp8_block_gemm_dequant(
        out.as_mut_ptr() as *mut c_void,
        input.as_ptr() as *const c_void,
        weights.as_ptr() as *const c_void,
        a_scales.as_ptr() as *const f32,
        w_scale_inv.as_ptr() as *const f32,
        topk_weights.as_ptr() as *const f32,
        sorted_token_ids.as_ptr() as *const i32,
        expert_ids.as_ptr() as *const i32,
        num_tokens_post_padded.as_ptr() as *const i32,
        num_tokens as c_int,
        in_features as c_int,
        out_features as c_int,
        top_k as c_int,
        apply_weights as c_int,
        block_m as c_int,
        block_n as c_int,
        block_k as c_int,
        scale_stride_e as c_int,
        scale_stride_n as c_int,
        stream,
    );
    out
}

// ---------------------------------------------------------------------------
// Sigmoid-gated add: out = a + sigmoid(gate) * b
// ---------------------------------------------------------------------------

/// Fused sigmoid-gated addition for shared expert output.
///
/// `out = a + sigmoid(gate) * b`
///
/// * `a`: `[num_tokens, hidden_size]` — MoE output
/// * `b`: `[num_tokens, hidden_size]` — shared expert output
/// * `gate`: `[num_tokens, 1]` — shared expert gate logits (pre-sigmoid)
///
/// Returns `[num_tokens, hidden_size]` (writes into a new buffer).
pub unsafe fn sigmoid_mul_add(
    a: GpuTensor,
    b: GpuTensor,
    gate: GpuTensor,
    alloc: &mut CachingAllocator,
    stream: CUstream,
) -> OwnedTensor {
    let num_tokens = a.dim(0);
    let hidden_size = a.dim(1);
    let out = alloc.alloc_tensor(&[num_tokens, hidden_size], a.dtype());

    match a.dtype() {
        DType::BF16 => sigmoid_mul_add_bf16(
            out.as_mut_ptr() as *mut c_void,
            a.as_ptr() as *const c_void,
            b.as_ptr() as *const c_void,
            gate.as_ptr() as *const c_void,
            num_tokens as c_int,
            hidden_size as c_int,
            stream,
        ),
        DType::F16 => sigmoid_mul_add_f16(
            out.as_mut_ptr() as *mut c_void,
            a.as_ptr() as *const c_void,
            b.as_ptr() as *const c_void,
            gate.as_ptr() as *const c_void,
            num_tokens as c_int,
            hidden_size as c_int,
            stream,
        ),
        DType::F32 => sigmoid_mul_add_f32(
            out.as_mut_ptr() as *mut c_void,
            a.as_ptr() as *const c_void,
            b.as_ptr() as *const c_void,
            gate.as_ptr() as *const c_void,
            num_tokens as c_int,
            hidden_size as c_int,
            stream,
        ),
        _ => panic!("sigmoid_mul_add: unsupported dtype {:?}", a.dtype()),
    }
    out
}

// ---------------------------------------------------------------------------
// In-place add: a += b
// ---------------------------------------------------------------------------

/// Element-wise in-place addition: `a += b`.
///
/// * `a`: `[num_tokens, hidden_size]` — modified in place
/// * `b`: `[num_tokens, hidden_size]`
pub unsafe fn add_inplace(a: GpuTensor, b: GpuTensor, stream: CUstream) {
    let num_tokens = a.dim(0);
    let hidden_size = a.dim(1);

    match a.dtype() {
        DType::BF16 => add_inplace_bf16(
            a.as_mut_ptr() as *mut c_void,
            b.as_ptr() as *const c_void,
            num_tokens as c_int,
            hidden_size as c_int,
            stream,
        ),
        DType::F16 => add_inplace_f16(
            a.as_mut_ptr() as *mut c_void,
            b.as_ptr() as *const c_void,
            num_tokens as c_int,
            hidden_size as c_int,
            stream,
        ),
        DType::F32 => add_inplace_f32(
            a.as_mut_ptr() as *mut c_void,
            b.as_ptr() as *const c_void,
            num_tokens as c_int,
            hidden_size as c_int,
            stream,
        ),
        _ => panic!("add_inplace: unsupported dtype {:?}", a.dtype()),
    }
}

// ---------------------------------------------------------------------------
// Fused QK-norm + RoPE (per-head RMS norm + rotation)
// ---------------------------------------------------------------------------

/// Fused per-head QK RMS normalization + RoPE, in-place on Q and K.
///
/// Used by Qwen3 (and Gemma3) which apply per-head RMS norm before RoPE.
///
/// * `query`: `[num_tokens, num_q_heads * head_dim]` — modified in-place
/// * `key`: `[num_tokens, num_kv_heads * head_dim]` — modified in-place
/// * `q_norm_weight`: `[head_dim]` — per-head Q norm weight
/// * `k_norm_weight`: `[head_dim]` — per-head K norm weight
/// * `cos_sin_cache`: `[max_pos, head_dim]` — combined cos|sin cache (first half cos, second half sin)
/// * `positions`: `[num_tokens]` (U32)
/// * `epsilon`: norm epsilon (e.g. 1e-6)
#[allow(clippy::too_many_arguments)]
pub unsafe fn qk_norm_rope_inplace(
    query: GpuTensor,
    key: GpuTensor,
    q_norm_weight: GpuTensor,
    k_norm_weight: GpuTensor,
    cos_sin_cache: GpuTensor,
    positions: GpuTensor,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    epsilon: f32,
    q_weight_offset: f32,
    k_weight_offset: f32,
    stream: CUstream,
) {
    let num_tokens = positions.dim(0);
    let half_dim = head_dim / 2;
    let elem_size = query.dtype().size_bytes();

    let cos_ptr = cos_sin_cache.raw_ptr();
    let sin_ptr = cos_ptr.add(half_dim * elem_size);

    match query.dtype() {
        DType::BF16 => qk_norm_rope_bf16(
            query.as_mut_ptr() as *mut u16,
            key.as_mut_ptr() as *mut u16,
            q_norm_weight.as_ptr() as *const u16,
            k_norm_weight.as_ptr() as *const u16,
            cos_ptr as *const u16,
            sin_ptr as *const u16,
            positions.as_ptr() as *const u32,
            epsilon,
            q_weight_offset,
            k_weight_offset,
            num_q_heads as c_int,
            num_kv_heads as c_int,
            head_dim as c_int,
            num_tokens as c_int,
            stream,
        ),
        DType::F16 => qk_norm_rope_f16(
            query.as_mut_ptr() as *mut u16,
            key.as_mut_ptr() as *mut u16,
            q_norm_weight.as_ptr() as *const u16,
            k_norm_weight.as_ptr() as *const u16,
            cos_ptr as *const u16,
            sin_ptr as *const u16,
            positions.as_ptr() as *const u32,
            epsilon,
            q_weight_offset,
            k_weight_offset,
            num_q_heads as c_int,
            num_kv_heads as c_int,
            head_dim as c_int,
            num_tokens as c_int,
            stream,
        ),
        DType::F32 => qk_norm_rope_f32(
            query.as_mut_ptr() as *mut f32,
            key.as_mut_ptr() as *mut f32,
            q_norm_weight.as_ptr() as *const f32,
            k_norm_weight.as_ptr() as *const f32,
            cos_ptr as *const f32,
            sin_ptr as *const f32,
            positions.as_ptr() as *const u32,
            epsilon,
            q_weight_offset,
            k_weight_offset,
            num_q_heads as c_int,
            num_kv_heads as c_int,
            head_dim as c_int,
            num_tokens as c_int,
            stream,
        ),
        _ => panic!("qk_norm_rope: unsupported dtype {:?}", query.dtype()),
    }
}

// ---------------------------------------------------------------------------
// Tests for FlashAttention-2 FFI shim (mha_varlen_fwd)
// ---------------------------------------------------------------------------

#[cfg(test)]
#[cfg(feature = "cuda")]
mod tests_flash_attn {
    use super::*;
    use crate::driver;

    /// Initialize CUDA context for test. Returns a stream.
    unsafe fn test_init() -> cudarc::driver::sys::CUstream {
        driver::init().expect("CUDA init");
        let dev = driver::device_get(0).expect("device 0");
        let _ctx = driver::ctx_create(dev).expect("ctx_create");
        driver::stream_create().expect("stream_create")
    }

    /// Upload a &[T] to GPU, return raw pointer.
    unsafe fn upload<T: Copy>(data: &[T], stream: cudarc::driver::sys::CUstream) -> *mut u8 {
        let bytes = std::mem::size_of_val(data);
        let ptr = driver::mem_alloc(bytes).expect("mem_alloc");
        driver::memcpy_htod_async(ptr, data.as_ptr() as *const u8, bytes, stream)
            .expect("memcpy_htod");
        driver::stream_synchronize(stream).expect("sync");
        ptr
    }

    /// Download GPU data to host Vec<T>.
    unsafe fn download<T: Copy + Default>(
        ptr: *mut u8,
        count: usize,
        stream: cudarc::driver::sys::CUstream,
    ) -> Vec<T> {
        let bytes = count * std::mem::size_of::<T>();
        let mut host = vec![T::default(); count];
        driver::memcpy_dtoh_async(host.as_mut_ptr() as *mut u8, ptr, bytes, stream)
            .expect("memcpy_dtoh");
        driver::stream_synchronize(stream).expect("sync");
        host
    }

    /// Non-paged (contiguous) basic test: Q=K=V=0.1, verify output ~0.1.
    #[test]
    #[ignore] // requires CUDA GPU
    fn test_mha_varlen_fwd_contiguous_basic() {
        unsafe {
            let stream = test_init();
            let (batch, seqlen, heads, head_dim) = (1, 4, 2, 64);
            let total_q = batch * seqlen;
            let nelems = total_q * heads * head_dim;
            let val = half::bf16::from_f32(0.1);
            let data: Vec<u16> = vec![val.to_bits(); nelems];

            let q_ptr = upload(&data, stream);
            let k_ptr = upload(&data, stream);
            let v_ptr = upload(&data, stream);
            let out_ptr = driver::mem_alloc(nelems * 2).expect("out alloc");
            driver::memset_d8(out_ptr, 0, nelems * 2, stream).expect("memset");
            let lse_ptr = driver::mem_alloc(heads * total_q * 4).expect("lse alloc");
            let cu_q_ptr = upload(&[0i32, seqlen as i32], stream);
            let cu_k_ptr = upload(&[0i32, seqlen as i32], stream);

            mha_varlen_fwd(
                q_ptr as *mut c_void,
                k_ptr as *mut c_void,
                v_ptr as *mut c_void,
                out_ptr as *mut c_void,
                lse_ptr as *mut c_void,
                cu_q_ptr as *const i32,
                cu_k_ptr as *const i32,
                std::ptr::null(), // seqused_k
                std::ptr::null(), // block_table (non-paged)
                0,
                batch as i32,
                seqlen as i32,
                seqlen as i32,
                heads as i32,
                heads as i32,
                head_dim as i32,
                1,
                (heads * head_dim) as i64,
                head_dim as i64,
                0,
                (heads * head_dim) as i64,
                head_dim as i64,
                (heads * head_dim) as i64,
                head_dim as i64,
                1.0 / (head_dim as f32).sqrt(),
                1,
                -1,
                0,
                0.0,
                1,
                0,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                0, // seqlenq_ngroups_swapped = false
                total_q as i32,
                std::ptr::null(), // rotary_cos_ptr (spans)
                std::ptr::null(), // rotary_sin_ptr (spans)
                0,                // rotary_dim (spans)
                0,                // rotate_cached_k (spans)
                0,                // is_rotary_interleaved
                std::ptr::null(), // block_unrotated_flags
                stream,
            );
            driver::stream_synchronize(stream).expect("sync");

            let out = download::<u16>(out_ptr, nelems, stream);
            for (i, &bits) in out.iter().enumerate() {
                let f = half::bf16::from_bits(bits).to_f32();
                assert!(
                    (f - 0.1).abs() < 0.05,
                    "contiguous out[{}]={} (want ~0.1)",
                    i,
                    f
                );
            }
            for p in [q_ptr, k_ptr, v_ptr, out_ptr, lse_ptr, cu_q_ptr, cu_k_ptr] {
                let _ = driver::mem_free(p);
            }
        }
    }

    /// Paged basic test: single block, single sequence.
    #[test]
    #[ignore]
    fn test_mha_varlen_fwd_paged_basic() {
        unsafe {
            let stream = test_init();
            let (batch, q_len, kv_len, heads, head_dim, block_size) = (1, 1, 4, 2, 64, 16);
            let val = half::bf16::from_f32(0.1);

            let q_ptr = upload(&vec![val.to_bits(); q_len * heads * head_dim], stream);
            let cache_elems = block_size * heads * head_dim;
            let k_ptr = upload(&vec![val.to_bits(); cache_elems], stream);
            let v_ptr = upload(&vec![val.to_bits(); cache_elems], stream);
            let out_elems = q_len * heads * head_dim;
            let out_ptr = driver::mem_alloc(out_elems * 2).expect("out");
            let lse_ptr = driver::mem_alloc(heads * q_len * 4).expect("lse");
            let cu_q_ptr = upload(&[0i32, 1], stream);
            let cu_k_ptr = upload(&[0i32, 0], stream); // dummy
            let seqused_ptr = upload(&[kv_len as i32], stream);
            let bt_ptr = upload(&[0i32], stream);

            mha_varlen_fwd(
                q_ptr as *mut c_void,
                k_ptr as *mut c_void,
                v_ptr as *mut c_void,
                out_ptr as *mut c_void,
                lse_ptr as *mut c_void,
                cu_q_ptr as *const i32,
                cu_k_ptr as *const i32,
                seqused_ptr as *const i32,
                bt_ptr as *const i32,
                1,
                batch as i32,
                q_len as i32,
                kv_len as i32,
                heads as i32,
                heads as i32,
                head_dim as i32,
                block_size as i32,
                (heads * head_dim) as i64,
                head_dim as i64,
                (block_size * heads * head_dim) as i64,
                (heads * head_dim) as i64,
                head_dim as i64,
                (heads * head_dim) as i64,
                head_dim as i64,
                1.0 / (head_dim as f32).sqrt(),
                1,
                -1,
                0,
                0.0,
                1,
                1, // num_splits=1 (paged requires explicit value)
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                0, // seqlenq_ngroups_swapped = false
                batch as i32,
                std::ptr::null(), // rotary_cos_ptr (spans)
                std::ptr::null(), // rotary_sin_ptr (spans)
                0,                // rotary_dim (spans)
                0,                // rotate_cached_k (spans)
                0,                // is_rotary_interleaved
                std::ptr::null(), // block_unrotated_flags
                stream,
            );
            driver::stream_synchronize(stream).expect("sync");

            let out = download::<u16>(out_ptr, out_elems, stream);
            for (i, &bits) in out.iter().enumerate() {
                let f = half::bf16::from_bits(bits).to_f32();
                assert!((f - 0.1).abs() < 0.05, "paged out[{}]={} (want ~0.1)", i, f);
            }
            for p in [
                q_ptr,
                k_ptr,
                v_ptr,
                out_ptr,
                lse_ptr,
                cu_q_ptr,
                cu_k_ptr,
                seqused_ptr,
                bt_ptr,
            ] {
                let _ = driver::mem_free(p);
            }
        }
    }

    /// THE critical test: non-contiguous block allocation.
    /// block_table=[2, 0] — first tokens in block 2, later tokens in block 0.
    /// Block 1 is garbage. The old standard kernel would read linearly and hit garbage.
    #[test]
    #[ignore]
    fn test_mha_varlen_fwd_paged_noncontiguous_blocks() {
        unsafe {
            let stream = test_init();
            // block_size must be multiple of 16 (upstream requirement)
            let (batch, q_len, kv_len, heads, head_dim, block_size) = (1, 1, 32, 2, 64, 16);
            let num_blocks = 3;
            let val_a = half::bf16::from_f32(0.2);
            let val_b = half::bf16::from_f32(0.4);

            let q_ptr = upload(
                &vec![half::bf16::from_f32(0.1).to_bits(); q_len * heads * head_dim],
                stream,
            );

            // Build cache: block 0=val_b, block 1=garbage, block 2=val_a
            let epb = block_size * heads * head_dim;
            let mut cache: Vec<u16> = vec![0u16; num_blocks * epb];
            cache[..epb].fill(val_b.to_bits()); // block 0
            cache[epb..2 * epb].fill(0xDEAD); // block 1 (garbage)
            cache[2 * epb..3 * epb].fill(val_a.to_bits()); // block 2
            let k_ptr = upload(&cache, stream);
            let v_ptr = upload(&cache, stream);

            let out_elems = q_len * heads * head_dim;
            let out_ptr = driver::mem_alloc(out_elems * 2).expect("out");
            let lse_ptr = driver::mem_alloc(heads * q_len * 4).expect("lse");
            let cu_q_ptr = upload(&[0i32, 1], stream);
            let cu_k_ptr = upload(&[0i32, 0], stream);
            let seqused_ptr = upload(&[kv_len as i32], stream);
            // block_table = [2, 0]: tokens 0-15 in block 2, tokens 16-31 in block 0
            let bt_ptr = upload(&[2i32, 0], stream);

            mha_varlen_fwd(
                q_ptr as *mut c_void,
                k_ptr as *mut c_void,
                v_ptr as *mut c_void,
                out_ptr as *mut c_void,
                lse_ptr as *mut c_void,
                cu_q_ptr as *const i32,
                cu_k_ptr as *const i32,
                seqused_ptr as *const i32,
                bt_ptr as *const i32,
                2,
                batch as i32,
                q_len as i32,
                kv_len as i32,
                heads as i32,
                heads as i32,
                head_dim as i32,
                block_size as i32,
                (heads * head_dim) as i64,
                head_dim as i64,
                (block_size * heads * head_dim) as i64,
                (heads * head_dim) as i64,
                head_dim as i64,
                (heads * head_dim) as i64,
                head_dim as i64,
                1.0 / (head_dim as f32).sqrt(),
                1,
                -1,
                0,
                0.0,
                1,
                1, // num_splits=1 (paged requires explicit value)
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                0, // seqlenq_ngroups_swapped = false
                batch as i32,
                std::ptr::null(), // rotary_cos_ptr (spans)
                std::ptr::null(), // rotary_sin_ptr (spans)
                0,                // rotary_dim (spans)
                0,                // rotate_cached_k (spans)
                0,                // is_rotary_interleaved
                std::ptr::null(), // block_unrotated_flags
                stream,
            );
            driver::stream_synchronize(stream).expect("sync");

            // Expected: uniform attention over 4*0.2 + 4*0.4 = avg 0.3
            let out = download::<u16>(out_ptr, out_elems, stream);
            for (i, &bits) in out.iter().enumerate() {
                let f = half::bf16::from_bits(bits).to_f32();
                assert!(
                    (f - 0.3).abs() < 0.1,
                    "noncontig out[{}]={} (want ~0.3)",
                    i,
                    f
                );
            }
            for p in [
                q_ptr,
                k_ptr,
                v_ptr,
                out_ptr,
                lse_ptr,
                cu_q_ptr,
                cu_k_ptr,
                seqused_ptr,
                bt_ptr,
            ] {
                let _ = driver::mem_free(p);
            }
        }
    }

    /// Paged with batch_size=2 — verifies second batch element isn't corrupted.
    #[test]
    #[ignore]
    fn test_mha_varlen_fwd_paged_batch2() {
        unsafe {
            let stream = test_init();
            let (batch, heads, head_dim, block_size) = (2, 2, 64, 16);
            let kv_lens = [4i32, 6i32];
            let total_q = 2;
            let val = half::bf16::from_f32(0.1);

            let q_ptr = upload(&vec![val.to_bits(); total_q * heads * head_dim], stream);
            let num_blocks = 2;
            let epb = block_size * heads * head_dim;
            let k_ptr = upload(&vec![val.to_bits(); num_blocks * epb], stream);
            let v_ptr = upload(&vec![val.to_bits(); num_blocks * epb], stream);

            let out_elems = total_q * heads * head_dim;
            let out_ptr = driver::mem_alloc(out_elems * 2).expect("out");
            let lse_ptr = driver::mem_alloc(heads * total_q * 4).expect("lse");
            let cu_q_ptr = upload(&[0i32, 1, 2], stream);
            let cu_k_ptr = upload(&[0i32, 0, 0], stream);
            let seqused_ptr = upload(&kv_lens, stream);
            let bt_ptr = upload(&[0i32, -1, 1, -1], stream); // seq0→block0, seq1→block1

            mha_varlen_fwd(
                q_ptr as *mut c_void,
                k_ptr as *mut c_void,
                v_ptr as *mut c_void,
                out_ptr as *mut c_void,
                lse_ptr as *mut c_void,
                cu_q_ptr as *const i32,
                cu_k_ptr as *const i32,
                seqused_ptr as *const i32,
                bt_ptr as *const i32,
                2,
                batch as i32,
                1,
                6, // max_seqlen_q=1, max_seqlen_k=6
                heads as i32,
                heads as i32,
                head_dim as i32,
                block_size as i32,
                (heads * head_dim) as i64,
                head_dim as i64,
                (block_size * heads * head_dim) as i64,
                (heads * head_dim) as i64,
                head_dim as i64,
                (heads * head_dim) as i64,
                head_dim as i64,
                1.0 / (head_dim as f32).sqrt(),
                1,
                -1,
                0,
                0.0,
                1,
                1, // num_splits=1 (paged requires explicit value)
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                0, // seqlenq_ngroups_swapped = false
                total_q as i32,
                std::ptr::null(), // rotary_cos_ptr (spans)
                std::ptr::null(), // rotary_sin_ptr (spans)
                0,                // rotary_dim (spans)
                0,                // rotate_cached_k (spans)
                0,                // is_rotary_interleaved
                std::ptr::null(), // block_unrotated_flags
                stream,
            );
            driver::stream_synchronize(stream).expect("sync");

            let out = download::<u16>(out_ptr, out_elems, stream);
            for (i, &bits) in out.iter().enumerate() {
                let f = half::bf16::from_bits(bits).to_f32();
                assert!(
                    (f - 0.1).abs() < 0.05,
                    "batch2 out[{}]={} (want ~0.1)",
                    i,
                    f
                );
            }
            for p in [
                q_ptr,
                k_ptr,
                v_ptr,
                out_ptr,
                lse_ptr,
                cu_q_ptr,
                cu_k_ptr,
                seqused_ptr,
                bt_ptr,
            ] {
                let _ = driver::mem_free(p);
            }
        }
    }

    /// GQA test: num_heads=4, num_kv_heads=2.
    #[test]
    #[ignore]
    fn test_mha_varlen_fwd_paged_gqa() {
        unsafe {
            let stream = test_init();
            let (batch, q_len, kv_len, num_heads, num_kv_heads, head_dim, block_size) =
                (1, 1, 4, 4, 2, 64, 16);
            let val = half::bf16::from_f32(0.1);

            let q_ptr = upload(&vec![val.to_bits(); q_len * num_heads * head_dim], stream);
            let epb = block_size * num_kv_heads * head_dim;
            let k_ptr = upload(&vec![val.to_bits(); epb], stream);
            let v_ptr = upload(&vec![val.to_bits(); epb], stream);

            let out_elems = q_len * num_heads * head_dim;
            let out_ptr = driver::mem_alloc(out_elems * 2).expect("out");
            let lse_ptr = driver::mem_alloc(num_heads * q_len * 4).expect("lse");
            let cu_q_ptr = upload(&[0i32, 1], stream);
            let cu_k_ptr = upload(&[0i32, 0], stream);
            let seqused_ptr = upload(&[kv_len as i32], stream);
            let bt_ptr = upload(&[0i32], stream);

            mha_varlen_fwd(
                q_ptr as *mut c_void,
                k_ptr as *mut c_void,
                v_ptr as *mut c_void,
                out_ptr as *mut c_void,
                lse_ptr as *mut c_void,
                cu_q_ptr as *const i32,
                cu_k_ptr as *const i32,
                seqused_ptr as *const i32,
                bt_ptr as *const i32,
                1,
                batch as i32,
                q_len as i32,
                kv_len as i32,
                num_heads as i32,
                num_kv_heads as i32,
                head_dim as i32,
                block_size as i32,
                (num_heads * head_dim) as i64,
                head_dim as i64,
                (block_size * num_kv_heads * head_dim) as i64,
                (num_kv_heads * head_dim) as i64,
                head_dim as i64,
                (num_heads * head_dim) as i64,
                head_dim as i64,
                1.0 / (head_dim as f32).sqrt(),
                1,
                -1,
                0,
                0.0,
                1,
                1, // num_splits=1 (paged requires explicit value)
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                0, // seqlenq_ngroups_swapped = false
                batch as i32,
                std::ptr::null(), // rotary_cos_ptr (spans)
                std::ptr::null(), // rotary_sin_ptr (spans)
                0,                // rotary_dim (spans)
                0,                // rotate_cached_k (spans)
                0,                // is_rotary_interleaved
                std::ptr::null(), // block_unrotated_flags
                stream,
            );
            driver::stream_synchronize(stream).expect("sync");

            let out = download::<u16>(out_ptr, out_elems, stream);
            for (i, &bits) in out.iter().enumerate() {
                let f = half::bf16::from_bits(bits).to_f32();
                assert!((f - 0.1).abs() < 0.05, "GQA out[{}]={} (want ~0.1)", i, f);
            }
            for p in [
                q_ptr,
                k_ptr,
                v_ptr,
                out_ptr,
                lse_ptr,
                cu_q_ptr,
                cu_k_ptr,
                seqused_ptr,
                bt_ptr,
            ] {
                let _ = driver::mem_free(p);
            }
        }
    }

    /// Compare paged FA2 with contiguous block_table [0,1,2] vs shuffled [2,0,1].
    /// Same KV data in blocks, just different block_table mapping.
    /// If outputs differ, the paged kernel mishandles non-contiguous block mappings.
    #[test]
    #[ignore]
    fn test_paged_vs_contiguous_gqa() {
        unsafe {
            let stream = test_init();
            // Realistic GQA config matching Qwen2.5-0.5B
            let (num_heads, num_kv_heads, head_dim, block_size) = (14, 2, 64, 16);
            let kv_len = 37; // spans 3 blocks (16+16+5)
            let scale = 1.0 / (head_dim as f32).sqrt();

            // Generate deterministic pseudo-random bf16 data
            let mut rng_state: u32 = 42;
            let mut next_bf16 = || -> u16 {
                rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
                let f = (rng_state >> 16) as f32 / 65536.0 - 0.5;
                half::bf16::from_f32(f).to_bits()
            };

            // Q: [1, num_heads, head_dim]
            let q_data: Vec<u16> = (0..num_heads * head_dim).map(|_| next_bf16()).collect();
            let q_ptr = upload(&q_data, stream);

            let row_bytes = num_kv_heads * head_dim; // elements per row
            let block_elems = block_size * row_bytes;

            // Generate 3 blocks of KV data (virtual blocks 0, 1, 2)
            let kv0: Vec<u16> = (0..block_elems).map(|_| next_bf16()).collect();
            let kv1: Vec<u16> = (0..block_elems).map(|_| next_bf16()).collect();
            let kv2: Vec<u16> = (0..block_elems).map(|_| next_bf16()).collect();
            let v0: Vec<u16> = (0..block_elems).map(|_| next_bf16()).collect();
            let v1: Vec<u16> = (0..block_elems).map(|_| next_bf16()).collect();
            let v2: Vec<u16> = (0..block_elems).map(|_| next_bf16()).collect();

            // --- Layout A: physical [0,1,2] = virtual [0,1,2], block_table=[0,1,2] ---
            let mut k_a = Vec::with_capacity(3 * block_elems);
            k_a.extend_from_slice(&kv0);
            k_a.extend_from_slice(&kv1);
            k_a.extend_from_slice(&kv2);
            let mut v_a = Vec::with_capacity(3 * block_elems);
            v_a.extend_from_slice(&v0);
            v_a.extend_from_slice(&v1);
            v_a.extend_from_slice(&v2);
            let k_a_ptr = upload(&k_a, stream);
            let v_a_ptr = upload(&v_a, stream);
            let bt_a = upload(&[0i32, 1, 2], stream);

            // --- Layout B: physical [0,1,2] = virtual [2,0,1], block_table=[1,2,0] ---
            // Physical 0 has virtual 2's data, physical 1 has virtual 0's, physical 2 has virtual 1's
            let mut k_b = Vec::with_capacity(3 * block_elems);
            k_b.extend_from_slice(&kv2); // physical 0 = virtual 2
            k_b.extend_from_slice(&kv0); // physical 1 = virtual 0
            k_b.extend_from_slice(&kv1); // physical 2 = virtual 1
            let mut v_b = Vec::with_capacity(3 * block_elems);
            v_b.extend_from_slice(&v2);
            v_b.extend_from_slice(&v0);
            v_b.extend_from_slice(&v1);
            let k_b_ptr = upload(&k_b, stream);
            let v_b_ptr = upload(&v_b, stream);
            let bt_b = upload(&[1i32, 2, 0], stream); // virtual 0→phys 1, virtual 1→phys 2, virtual 2→phys 0

            // Shared metadata
            let cu_q = upload(&[0i32, 1], stream);
            let cu_k_dummy = upload(&[0i32, 0], stream);
            let seqused = upload(&[kv_len as i32], stream);

            let kv_block_stride = (block_size * num_kv_heads * head_dim) as i64;
            let kv_row_stride = (num_kv_heads * head_dim) as i64;
            let kv_head_stride = head_dim as i64;

            let call_paged = |k_ptr, v_ptr, bt_ptr, out_ptr, lse_ptr| {
                mha_varlen_fwd(
                    q_ptr as *mut c_void,
                    k_ptr as *mut c_void,
                    v_ptr as *mut c_void,
                    out_ptr as *mut c_void,
                    lse_ptr as *mut c_void,
                    cu_q as *const i32,
                    cu_k_dummy as *const i32,
                    seqused as *const i32,
                    bt_ptr as *const i32,
                    3, // block_table_batch_stride
                    1, // batch_size
                    1, // max_seqlen_q
                    kv_len as i32,
                    num_heads as i32,
                    num_kv_heads as i32,
                    head_dim as i32,
                    block_size as i32,
                    (num_heads * head_dim) as i64,
                    head_dim as i64,
                    kv_block_stride,
                    kv_row_stride,
                    kv_head_stride,
                    (num_heads * head_dim) as i64,
                    head_dim as i64,
                    scale,
                    1,
                    -1,
                    0,
                    0.0,
                    1,
                    1, // num_splits
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    0,                // seqlenq_ngroups_swapped = false
                    1,                // total_q
                    std::ptr::null(), // rotary_cos_ptr (spans)
                    std::ptr::null(), // rotary_sin_ptr (spans)
                    0,                // rotary_dim (spans)
                    0,                // rotate_cached_k (spans)
                    0,                // is_rotary_interleaved
                    std::ptr::null(), // block_unrotated_flags
                    stream,
                );
            };

            let out_a = driver::mem_alloc(num_heads * head_dim * 2).expect("out a");
            let lse_a = driver::mem_alloc(num_heads * 4).expect("lse a");
            call_paged(k_a_ptr, v_a_ptr, bt_a, out_a, lse_a);
            driver::stream_synchronize(stream).expect("sync a");

            let out_b = driver::mem_alloc(num_heads * head_dim * 2).expect("out b");
            let lse_b = driver::mem_alloc(num_heads * 4).expect("lse b");
            call_paged(k_b_ptr, v_b_ptr, bt_b, out_b, lse_b);
            driver::stream_synchronize(stream).expect("sync b");

            // --- Compare ---
            let res_a = download::<u16>(out_a, num_heads * head_dim, stream);
            let res_b = download::<u16>(out_b, num_heads * head_dim, stream);

            let mut max_diff: f32 = 0.0;
            let mut num_bad = 0;
            for i in 0..res_a.len() {
                let a = half::bf16::from_bits(res_a[i]).to_f32();
                let b = half::bf16::from_bits(res_b[i]).to_f32();
                let diff = (a - b).abs();
                if diff > max_diff {
                    max_diff = diff;
                }
                if diff > 0.01 {
                    num_bad += 1;
                }
            }
            eprintln!(
                "contiguous [0,1,2] vs shuffled [1,2,0]: max_diff={:.6}, num_bad={}/{} (threshold=0.01)",
                max_diff,
                num_bad,
                res_a.len()
            );

            // Also print first few values from each
            for i in 0..5.min(res_a.len()) {
                let a = half::bf16::from_bits(res_a[i]).to_f32();
                let b = half::bf16::from_bits(res_b[i]).to_f32();
                eprintln!("  [{i}] a={a:.6} b={b:.6} diff={:.6}", (a - b).abs());
            }

            assert!(
                max_diff < 0.02,
                "paged FA2 shuffled blocks diverge from contiguous: max_diff={:.6}, bad={}/{}",
                max_diff,
                num_bad,
                res_a.len()
            );

            // Cleanup
            for p in [
                q_ptr, k_a_ptr, v_a_ptr, k_b_ptr, v_b_ptr, bt_a, bt_b, out_a, lse_a, out_b, lse_b,
                cu_q, cu_k_dummy, seqused,
            ] {
                let _ = driver::mem_free(p);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Parameterized paged FA2 correctness: contiguous vs shuffled blocks.
    // Covers various GQA ratios, head dims, KV lengths, and batch sizes.
    // -----------------------------------------------------------------------

    /// Helper: run paged FA2 with contiguous block mapping [0,1,2,...] and a
    /// shuffled mapping, then assert outputs match within tolerance.
    unsafe fn assert_paged_shuffle_equivalent(
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        block_size: usize,
        kv_len: usize,
        batch_size: usize,
        label: &str,
    ) {
        let stream = test_init();
        let scale = 1.0 / (head_dim as f32).sqrt();
        let num_blocks_per_seq = kv_len.div_ceil(block_size);
        let total_physical_blocks = num_blocks_per_seq + 1; // +1 for shuffle room

        let mut rng_state: u32 = 0xBEEF_u32
            .wrapping_mul(num_heads as u32 + 1)
            .wrapping_add(kv_len as u32)
            .wrapping_add(batch_size as u32);
        let mut next_bf16 = || -> u16 {
            rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
            let f = (rng_state >> 16) as f32 / 65536.0 - 0.5;
            half::bf16::from_f32(f).to_bits()
        };

        let row_elems = num_kv_heads * head_dim;
        let block_elems = block_size * row_elems;
        let total_cache_blocks = total_physical_blocks * batch_size;

        // Allocate Q for the batch: [batch_size, num_heads, head_dim]
        let q_elems = batch_size * num_heads * head_dim;
        let q_data: Vec<u16> = (0..q_elems).map(|_| next_bf16()).collect();
        let q_ptr = upload(&q_data, stream);

        // Allocate KV blocks: enough physical blocks for both layouts
        let cache_elems = total_cache_blocks * block_elems;
        let k_data: Vec<u16> = (0..cache_elems).map(|_| next_bf16()).collect();
        let v_data: Vec<u16> = (0..cache_elems).map(|_| next_bf16()).collect();

        // Layout A: block_table = [0, 1, 2, ...] per batch
        let k_a_ptr = upload(&k_data, stream);
        let v_a_ptr = upload(&v_data, stream);
        let bt_a_data: Vec<i32> = (0..batch_size)
            .flat_map(|b| {
                let base = (b * total_physical_blocks) as i32;
                (0..num_blocks_per_seq).map(move |i| base + i as i32)
            })
            .collect();
        let bt_a = upload(&bt_a_data, stream);

        // Layout B: shuffled — rotate blocks by 1 within each batch element
        // Physical block mapping: virtual[i] -> physical[(i+1) % (num_blocks+1)]
        let mut k_b_data = vec![0u16; cache_elems];
        let mut v_b_data = vec![0u16; cache_elems];
        let mut bt_b_data = Vec::with_capacity(batch_size * num_blocks_per_seq);

        for b in 0..batch_size {
            let phys_base = b * total_physical_blocks;
            for virt_idx in 0..num_blocks_per_seq {
                let phys_a = phys_base + virt_idx;
                let phys_b = phys_base + ((virt_idx + 1) % total_physical_blocks);
                // Copy block data from A's physical location to B's
                let src_off = phys_a * block_elems;
                let dst_off = phys_b * block_elems;
                k_b_data[dst_off..dst_off + block_elems]
                    .copy_from_slice(&k_data[src_off..src_off + block_elems]);
                v_b_data[dst_off..dst_off + block_elems]
                    .copy_from_slice(&v_data[src_off..src_off + block_elems]);
                bt_b_data.push(phys_b as i32);
            }
        }
        let k_b_ptr = upload(&k_b_data, stream);
        let v_b_ptr = upload(&v_b_data, stream);
        let bt_b = upload(&bt_b_data, stream);

        // Sequence metadata
        let cu_q_data: Vec<i32> = (0..=batch_size as i32).collect();
        let cu_q = upload(&cu_q_data, stream);
        let cu_k_dummy: Vec<i32> = vec![0i32; batch_size + 1];
        let cu_k = upload(&cu_k_dummy, stream);
        let seqused_data: Vec<i32> = vec![kv_len as i32; batch_size];
        let seqused = upload(&seqused_data, stream);

        let kv_block_stride = (block_size * num_kv_heads * head_dim) as i64;
        let kv_row_stride = (num_kv_heads * head_dim) as i64;
        let kv_head_stride = head_dim as i64;
        let q_row_stride = (num_heads * head_dim) as i64;
        let q_head_stride = head_dim as i64;

        let call = |k_ptr, v_ptr, bt_ptr, out_ptr, lse_ptr| {
            mha_varlen_fwd(
                q_ptr as *mut c_void,
                k_ptr as *mut c_void,
                v_ptr as *mut c_void,
                out_ptr as *mut c_void,
                lse_ptr as *mut c_void,
                cu_q as *const i32,
                cu_k as *const i32,
                seqused as *const i32,
                bt_ptr as *const i32,
                num_blocks_per_seq as i32,
                batch_size as i32,
                1, // max_seqlen_q
                kv_len as i32,
                num_heads as i32,
                num_kv_heads as i32,
                head_dim as i32,
                block_size as i32,
                q_row_stride,
                q_head_stride,
                kv_block_stride,
                kv_row_stride,
                kv_head_stride,
                q_row_stride, // o_row_stride = q_row_stride
                q_head_stride,
                scale,
                1,
                -1,
                0,
                0.0,
                1,
                1, // num_splits
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                0, // seqlenq_ngroups_swapped = false
                batch_size as i32,
                std::ptr::null(), // rotary_cos_ptr (spans)
                std::ptr::null(), // rotary_sin_ptr (spans)
                0,                // rotary_dim (spans)
                0,                // rotate_cached_k (spans)
                0,                // is_rotary_interleaved
                std::ptr::null(), // block_unrotated_flags
                stream,
            );
        };

        let out_elems = batch_size * num_heads * head_dim;
        let out_a = driver::mem_alloc(out_elems * 2).expect("out a");
        let lse_a = driver::mem_alloc(batch_size * num_heads * 4).expect("lse a");
        call(k_a_ptr, v_a_ptr, bt_a, out_a, lse_a);
        driver::stream_synchronize(stream).expect("sync a");

        let out_b = driver::mem_alloc(out_elems * 2).expect("out b");
        let lse_b = driver::mem_alloc(batch_size * num_heads * 4).expect("lse b");
        call(k_b_ptr, v_b_ptr, bt_b, out_b, lse_b);
        driver::stream_synchronize(stream).expect("sync b");

        let res_a = download::<u16>(out_a, out_elems, stream);
        let res_b = download::<u16>(out_b, out_elems, stream);

        let mut max_diff: f32 = 0.0;
        let mut num_bad = 0;
        for i in 0..res_a.len() {
            let a = half::bf16::from_bits(res_a[i]).to_f32();
            let b = half::bf16::from_bits(res_b[i]).to_f32();
            let diff = (a - b).abs();
            if diff > max_diff {
                max_diff = diff;
            }
            if diff > 0.01 {
                num_bad += 1;
            }
        }
        eprintln!(
            "[{label}] h={num_heads} hk={num_kv_heads} d={head_dim} bs={block_size} kv={kv_len} batch={batch_size}: max_diff={max_diff:.6}, bad={num_bad}/{}",
            res_a.len()
        );
        assert!(
            max_diff < 0.02,
            "[{label}] paged shuffle diverges: max_diff={max_diff:.6}, bad={num_bad}/{}",
            res_a.len()
        );

        for p in [
            q_ptr, k_a_ptr, v_a_ptr, k_b_ptr, v_b_ptr, bt_a, bt_b, out_a, lse_a, out_b, lse_b,
            cu_q, cu_k, seqused,
        ] {
            let _ = driver::mem_free(p);
        }
    }

    /// GQA ratio 7:1 (Qwen2.5-0.5B config) — the exact bug scenario
    #[test]
    #[ignore]
    fn test_paged_shuffle_gqa_7to1_hdim64() {
        unsafe {
            assert_paged_shuffle_equivalent(14, 2, 64, 16, 37, 1, "gqa_7to1_d64");
        }
    }

    /// GQA ratio 4:1 (common config)
    #[test]
    #[ignore]
    fn test_paged_shuffle_gqa_4to1_hdim128() {
        unsafe {
            assert_paged_shuffle_equivalent(32, 8, 128, 16, 50, 1, "gqa_4to1_d128");
        }
    }

    /// MHA (no GQA, ratio 1:1) — should also work with splitkv
    #[test]
    #[ignore]
    fn test_paged_shuffle_mha_hdim64() {
        unsafe {
            assert_paged_shuffle_equivalent(8, 8, 64, 16, 48, 1, "mha_d64");
        }
    }

    /// Small head_dim=32
    #[test]
    #[ignore]
    fn test_paged_shuffle_hdim32() {
        unsafe {
            assert_paged_shuffle_equivalent(8, 2, 32, 16, 64, 1, "gqa_d32");
        }
    }

    /// Large head_dim=128 with GQA
    #[test]
    #[ignore]
    fn test_paged_shuffle_hdim128_gqa() {
        unsafe {
            assert_paged_shuffle_equivalent(32, 4, 128, 16, 80, 1, "gqa_d128");
        }
    }

    /// Single block — all tokens fit in one page
    #[test]
    #[ignore]
    fn test_paged_shuffle_single_block() {
        unsafe {
            assert_paged_shuffle_equivalent(14, 2, 64, 16, 16, 1, "single_block");
        }
    }

    /// Many blocks (5 blocks, kv_len=80)
    #[test]
    #[ignore]
    fn test_paged_shuffle_many_blocks() {
        unsafe {
            assert_paged_shuffle_equivalent(14, 2, 64, 16, 80, 1, "many_blocks");
        }
    }

    /// Large kv_len spanning multiple kBlockN tiles (kBlockN=128 for hdim=64)
    #[test]
    #[ignore]
    fn test_paged_shuffle_multi_tile() {
        unsafe {
            // 200 tokens = 13 blocks of 16, spans 2 kBlockN tiles
            assert_paged_shuffle_equivalent(14, 2, 64, 16, 200, 1, "multi_tile");
        }
    }

    /// Batch size > 1 with shuffled blocks
    #[test]
    #[ignore]
    fn test_paged_shuffle_batch4() {
        unsafe {
            assert_paged_shuffle_equivalent(14, 2, 64, 16, 48, 4, "batch4");
        }
    }

    /// Batch size 2 with large GQA ratio and hdim=128
    #[test]
    #[ignore]
    fn test_paged_shuffle_batch2_gqa_hdim128() {
        unsafe {
            assert_paged_shuffle_equivalent(32, 4, 128, 16, 64, 2, "batch2_gqa_d128");
        }
    }

    /// Contiguous path (non-paged) through mha_varlen_fwd with null block_table
    #[test]
    #[ignore]
    fn test_contiguous_varlen_fwd() {
        unsafe {
            let stream = test_init();
            let (num_heads, num_kv_heads, head_dim) = (8, 2, 64);
            let kv_len = 32;
            let scale = 1.0 / (head_dim as f32).sqrt();

            let mut rng_state: u32 = 12345;
            let mut next_bf16 = || -> u16 {
                rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
                half::bf16::from_f32((rng_state >> 16) as f32 / 65536.0 - 0.5).to_bits()
            };

            let q: Vec<u16> = (0..num_heads * head_dim).map(|_| next_bf16()).collect();
            let k: Vec<u16> = (0..kv_len * num_kv_heads * head_dim)
                .map(|_| next_bf16())
                .collect();
            let v: Vec<u16> = (0..kv_len * num_kv_heads * head_dim)
                .map(|_| next_bf16())
                .collect();

            let q_ptr = upload(&q, stream);
            let k_ptr = upload(&k, stream);
            let v_ptr = upload(&v, stream);
            let out_ptr = driver::mem_alloc(num_heads * head_dim * 2).expect("out");
            let lse_ptr = driver::mem_alloc(num_heads * 4).expect("lse");
            let cu_q = upload(&[0i32, 1], stream);
            let cu_k = upload(&[0i32, kv_len as i32], stream);

            mha_varlen_fwd(
                q_ptr as *mut c_void,
                k_ptr as *mut c_void,
                v_ptr as *mut c_void,
                out_ptr as *mut c_void,
                lse_ptr as *mut c_void,
                cu_q as *const i32,
                cu_k as *const i32,
                std::ptr::null(), // seqused_k = null
                std::ptr::null(), // block_table = null (contiguous)
                0,
                1,
                1,
                kv_len as i32,
                num_heads as i32,
                num_kv_heads as i32,
                head_dim as i32,
                0, // page_block_size unused
                (num_heads * head_dim) as i64,
                head_dim as i64,
                0, // k_batch_stride unused for contiguous
                (num_kv_heads * head_dim) as i64,
                head_dim as i64,
                (num_heads * head_dim) as i64,
                head_dim as i64,
                scale,
                1,
                -1,
                0,
                0.0,
                1,
                1,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                0,                // seqlenq_ngroups_swapped = false
                1,                // total_q = 1 (single query token)
                std::ptr::null(), // rotary_cos_ptr (spans)
                std::ptr::null(), // rotary_sin_ptr (spans)
                0,                // rotary_dim (spans)
                0,                // rotate_cached_k (spans)
                0,                // is_rotary_interleaved
                std::ptr::null(), // block_unrotated_flags
                stream,
            );
            driver::stream_synchronize(stream).expect("sync");

            let out = download::<u16>(out_ptr, num_heads * head_dim, stream);
            // Verify output is not all zeros or NaN
            let mut has_nonzero = false;
            for &bits in &out {
                let f = half::bf16::from_bits(bits).to_f32();
                assert!(!f.is_nan(), "contiguous FA2 produced NaN");
                if f.abs() > 1e-6 {
                    has_nonzero = true;
                }
            }
            assert!(has_nonzero, "contiguous FA2 output is all zeros");

            for p in [q_ptr, k_ptr, v_ptr, out_ptr, lse_ptr, cu_q, cu_k] {
                let _ = driver::mem_free(p);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests: GPU pooling kernels
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Sampling kernel tests (logit_bias, penalties, min_tokens)
// ---------------------------------------------------------------------------

#[cfg(test)]
#[cfg(feature = "cuda")]
mod tests_sampling {
    use super::*;
    use crate::alloc::CachingAllocator;
    use crate::driver;

    unsafe fn test_init() -> (CachingAllocator, cudarc::driver::sys::CUstream) {
        driver::init().expect("CUDA init");
        let dev = driver::device_get(0).expect("device 0");
        let _ctx = driver::ctx_create(dev).expect("ctx_create");
        let stream = driver::stream_create().expect("stream_create");
        let alloc = CachingAllocator::new();
        (alloc, stream)
    }

    unsafe fn upload_f32(data: &[f32], stream: cudarc::driver::sys::CUstream) -> *mut u8 {
        let bytes = data.len() * 4;
        let ptr = driver::mem_alloc(bytes).expect("mem_alloc");
        driver::memcpy_htod_async(ptr, data.as_ptr() as *const u8, bytes, stream).expect("htod");
        driver::stream_synchronize(stream).expect("sync");
        ptr
    }

    unsafe fn upload_i32(data: &[i32], stream: cudarc::driver::sys::CUstream) -> *mut u8 {
        let bytes = data.len() * 4;
        let ptr = driver::mem_alloc(bytes).expect("mem_alloc");
        driver::memcpy_htod_async(ptr, data.as_ptr() as *const u8, bytes, stream).expect("htod");
        driver::stream_synchronize(stream).expect("sync");
        ptr
    }

    unsafe fn download_f32(
        ptr: *const u8,
        count: usize,
        stream: cudarc::driver::sys::CUstream,
    ) -> Vec<f32> {
        let mut host = vec![0.0f32; count];
        driver::memcpy_dtoh_async(host.as_mut_ptr() as *mut u8, ptr, count * 4, stream)
            .expect("dtoh");
        driver::stream_synchronize(stream).expect("sync");
        host
    }

    /// Test apply_min_tokens: suppress specific tokens for specific requests.
    #[test]
    #[ignore] // requires CUDA GPU
    fn test_cuda_apply_min_tokens() {
        unsafe {
            let (_alloc, stream) = test_init();
            // 2 requests, vocab_size=5.
            // logits = [[1.0, 2.0, 3.0, 4.0, 5.0],
            //           [5.0, 4.0, 3.0, 2.0, 1.0]]
            let vocab_size = 5;
            let logits_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 4.0, 3.0, 2.0, 1.0];
            let logits_ptr = upload_f32(&logits_data, stream);
            let logits = GpuTensor::new(logits_ptr, &[2, vocab_size], DType::F32);

            // Suppress token 4 for request 0, token 0 for request 1.
            let req_indices = [0i32, 1];
            let token_ids = [4i32, 0];
            let req_ptr = upload_i32(&req_indices, stream);
            let tok_ptr = upload_i32(&token_ids, stream);
            let gpu_req = GpuTensor::new(req_ptr, &[2], DType::U32);
            let gpu_tok = GpuTensor::new(tok_ptr, &[2], DType::U32);

            apply_min_tokens(logits, gpu_req, gpu_tok, stream);
            driver::stream_synchronize(stream).expect("sync");

            let result = download_f32(logits_ptr, 10, stream);
            // Request 0: token 4 (index 4) should be -inf.
            assert_eq!(result[0], 1.0);
            assert_eq!(result[1], 2.0);
            assert_eq!(result[2], 3.0);
            assert_eq!(result[3], 4.0);
            assert!(
                result[4].is_infinite() && result[4] < 0.0,
                "token 4 should be -inf"
            );
            // Request 1: token 0 (index 0) should be -inf.
            assert!(
                result[5].is_infinite() && result[5] < 0.0,
                "token 0 should be -inf"
            );
            assert_eq!(result[6], 4.0);
            assert_eq!(result[7], 3.0);
            assert_eq!(result[8], 2.0);
            assert_eq!(result[9], 1.0);

            let _ = driver::mem_free(logits_ptr);
            let _ = driver::mem_free(req_ptr);
            let _ = driver::mem_free(tok_ptr);
        }
    }

    /// Test apply_logit_bias: CSR-packed sparse bias addition.
    #[test]
    #[ignore] // requires CUDA GPU
    fn test_cuda_apply_logit_bias() {
        unsafe {
            let (_alloc, stream) = test_init();
            // 2 requests, vocab_size=4.
            let vocab_size = 4;
            let logits_data: Vec<f32> = vec![0.0; 8]; // all zeros
            let logits_ptr = upload_f32(&logits_data, stream);
            let logits = GpuTensor::new(logits_ptr, &[2, vocab_size], DType::F32);

            // Request 0: token 1 += 10.0, token 3 += -5.0
            // Request 1: token 0 += 3.0
            let bias_ids = [1i32, 3, 0];
            let bias_vals = [10.0f32, -5.0, 3.0];
            let bias_offsets = [0i32, 2, 3]; // CSR: req0=[0..2), req1=[2..3)
            let ids_ptr = upload_i32(&bias_ids, stream);
            let vals_ptr = upload_f32(&bias_vals, stream);
            let off_ptr = upload_i32(&bias_offsets, stream);
            let gpu_ids = GpuTensor::new(ids_ptr, &[3], DType::U32);
            let gpu_vals = GpuTensor::new(vals_ptr, &[3], DType::F32);
            let gpu_off = GpuTensor::new(off_ptr, &[3], DType::U32);

            apply_logit_bias(logits, gpu_ids, gpu_vals, gpu_off, stream);
            driver::stream_synchronize(stream).expect("sync");

            let result = download_f32(logits_ptr, 8, stream);
            // Request 0: [0, 10, 0, -5]
            assert_eq!(result[0], 0.0);
            assert_eq!(result[1], 10.0);
            assert_eq!(result[2], 0.0);
            assert_eq!(result[3], -5.0);
            // Request 1: [3, 0, 0, 0]
            assert_eq!(result[4], 3.0);
            assert_eq!(result[5], 0.0);
            assert_eq!(result[6], 0.0);
            assert_eq!(result[7], 0.0);

            let _ = driver::mem_free(logits_ptr);
            let _ = driver::mem_free(ids_ptr);
            let _ = driver::mem_free(vals_ptr);
            let _ = driver::mem_free(off_ptr);
        }
    }

    /// Test apply_penalties: repetition penalty on output tokens.
    #[test]
    #[ignore] // requires CUDA GPU
    fn test_cuda_apply_penalties_repetition() {
        unsafe {
            let (_alloc, stream) = test_init();
            // 1 request, vocab_size=4, output tokens = [0, 1].
            let vocab_size = 4usize;
            let logits_data: Vec<f32> = vec![2.0, -1.0, 0.5, 3.0];
            let logits_ptr = upload_f32(&logits_data, stream);
            let logits = GpuTensor::new(logits_ptr, &[1, vocab_size], DType::F32);

            // Output token IDs: [0, 1], padded with vocab_size sentinel.
            let out_ids = [0i32, 1, vocab_size as i32]; // 3 cols, last is padding
            let prompt_ids = [vocab_size as i32]; // 1 col, all padding
            let out_ptr = upload_i32(&out_ids, stream);
            let prompt_ptr = upload_i32(&prompt_ids, stream);
            let gpu_out = GpuTensor::new(out_ptr, &[1, 3], DType::U32);
            let gpu_prompt = GpuTensor::new(prompt_ptr, &[1, 1], DType::U32);

            // rep_penalty=2.0, freq=0, pres=0.
            let rep = [2.0f32];
            let freq = [0.0f32];
            let pres = [0.0f32];
            let rep_ptr = upload_f32(&rep, stream);
            let freq_ptr = upload_f32(&freq, stream);
            let pres_ptr = upload_f32(&pres, stream);
            let gpu_rep = GpuTensor::new(rep_ptr, &[1], DType::F32);
            let gpu_freq = GpuTensor::new(freq_ptr, &[1], DType::F32);
            let gpu_pres = GpuTensor::new(pres_ptr, &[1], DType::F32);

            apply_penalties(
                logits, gpu_out, gpu_prompt, gpu_rep, gpu_freq, gpu_pres, stream,
            );
            driver::stream_synchronize(stream).expect("sync");

            let result = download_f32(logits_ptr, 4, stream);
            // Token 0 (logit=2.0, positive): 2.0 / 2.0 = 1.0.
            assert!((result[0] - 1.0).abs() < 1e-5, "got {}", result[0]);
            // Token 1 (logit=-1.0, negative): -1.0 * 2.0 = -2.0.
            assert!((result[1] - (-2.0)).abs() < 1e-5, "got {}", result[1]);
            // Token 2: unmodified (not in output), 0.5.
            assert!((result[2] - 0.5).abs() < 1e-5, "got {}", result[2]);
            // Token 3: unmodified, 3.0.
            assert!((result[3] - 3.0).abs() < 1e-5, "got {}", result[3]);

            let _ = driver::mem_free(logits_ptr);
            let _ = driver::mem_free(out_ptr);
            let _ = driver::mem_free(prompt_ptr);
            let _ = driver::mem_free(rep_ptr);
            let _ = driver::mem_free(freq_ptr);
            let _ = driver::mem_free(pres_ptr);
        }
    }

    /// Test apply_min_tokens with zero count (no-op).
    #[test]
    #[ignore] // requires CUDA GPU
    fn test_cuda_apply_min_tokens_noop() {
        unsafe {
            let (_alloc, stream) = test_init();
            let logits_data: Vec<f32> = vec![1.0, 2.0, 3.0];
            let logits_ptr = upload_f32(&logits_data, stream);
            let logits = GpuTensor::new(logits_ptr, &[1, 3], DType::F32);

            // Empty req_indices and token_ids — kernel should be a no-op.
            let empty_ptr = driver::mem_alloc(4).expect("mem_alloc");
            let gpu_req = GpuTensor::new(empty_ptr, &[0], DType::U32);
            let gpu_tok = GpuTensor::new(empty_ptr, &[0], DType::U32);

            apply_min_tokens(logits, gpu_req, gpu_tok, stream);
            driver::stream_synchronize(stream).expect("sync");

            let result = download_f32(logits_ptr, 3, stream);
            assert_eq!(result, vec![1.0, 2.0, 3.0]);

            let _ = driver::mem_free(logits_ptr);
            let _ = driver::mem_free(empty_ptr);
        }
    }
}

#[cfg(test)]
#[cfg(feature = "cuda")]
mod tests_pooling {
    use super::*;
    use crate::alloc::CachingAllocator;
    use crate::driver;

    unsafe fn test_init() -> cudarc::driver::sys::CUstream {
        driver::init().expect("CUDA init");
        let dev = driver::device_get(0).expect("device 0");
        let _ctx = driver::ctx_create(dev).expect("ctx_create");
        driver::stream_create().expect("stream_create")
    }

    unsafe fn upload_f32(data: &[f32], stream: cudarc::driver::sys::CUstream) -> *mut u8 {
        let bytes = data.len() * 4;
        let ptr = driver::mem_alloc(bytes).expect("mem_alloc");
        driver::memcpy_htod_async(ptr, data.as_ptr() as *const u8, bytes, stream).expect("H2D");
        ptr
    }

    unsafe fn download_f32(
        ptr: *mut u8,
        count: usize,
        stream: cudarc::driver::sys::CUstream,
    ) -> Vec<f32> {
        let mut buf = vec![0.0f32; count];
        driver::memcpy_dtoh_async(buf.as_mut_ptr() as *mut u8, ptr, count * 4, stream)
            .expect("D2H");
        driver::stream_synchronize(stream).expect("sync");
        buf
    }

    // -- pool_select_row tests --

    #[test]
    #[ignore]
    fn test_pool_select_row_first() {
        unsafe {
            let stream = test_init();
            let mut alloc = CachingAllocator::new();

            // 3 tokens, hidden_size=4
            let data: Vec<f32> = vec![
                1.0, 2.0, 3.0, 4.0, // row 0
                5.0, 6.0, 7.0, 8.0, // row 1
                9.0, 10.0, 11.0, 12.0, // row 2
            ];
            let ptr = upload_f32(&data, stream);
            let hs = GpuTensor::new(ptr, &[3, 4], DType::F32);

            let out = pool_select_row(hs, 0, &mut alloc, stream);
            let result = download_f32(out.as_gpu_tensor().raw_ptr(), 4, stream);
            assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0]);

            let _ = driver::mem_free(ptr);
        }
    }

    #[test]
    #[ignore]
    fn test_pool_select_row_last() {
        unsafe {
            let stream = test_init();
            let mut alloc = CachingAllocator::new();

            let data: Vec<f32> = vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ];
            let ptr = upload_f32(&data, stream);
            let hs = GpuTensor::new(ptr, &[3, 4], DType::F32);

            let out = pool_select_row(hs, 2, &mut alloc, stream);
            let result = download_f32(out.as_gpu_tensor().raw_ptr(), 4, stream);
            assert_eq!(result, vec![9.0, 10.0, 11.0, 12.0]);

            let _ = driver::mem_free(ptr);
        }
    }

    #[test]
    #[ignore]
    fn test_pool_select_row_middle() {
        unsafe {
            let stream = test_init();
            let mut alloc = CachingAllocator::new();

            let data: Vec<f32> = vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ];
            let ptr = upload_f32(&data, stream);
            let hs = GpuTensor::new(ptr, &[3, 4], DType::F32);

            let out = pool_select_row(hs, 1, &mut alloc, stream);
            let result = download_f32(out.as_gpu_tensor().raw_ptr(), 4, stream);
            assert_eq!(result, vec![5.0, 6.0, 7.0, 8.0]);

            let _ = driver::mem_free(ptr);
        }
    }

    #[test]
    #[ignore]
    fn test_pool_select_row_bf16() {
        unsafe {
            let stream = test_init();
            let mut alloc = CachingAllocator::new();

            // 2 tokens, hidden_size=3, bf16
            let row0: Vec<u16> = [1.0f32, 2.0, 3.0]
                .iter()
                .map(|&v| half::bf16::from_f32(v).to_bits())
                .collect();
            let row1: Vec<u16> = [4.0f32, 5.0, 6.0]
                .iter()
                .map(|&v| half::bf16::from_f32(v).to_bits())
                .collect();
            let data: Vec<u16> = [row0, row1].concat();
            let bytes = data.len() * 2;
            let ptr = driver::mem_alloc(bytes).expect("alloc");
            driver::memcpy_htod_async(ptr, data.as_ptr() as *const u8, bytes, stream).expect("H2D");

            let hs = GpuTensor::new(ptr, &[2, 3], DType::BF16);
            let out = pool_select_row(hs, 1, &mut alloc, stream);

            let mut buf = vec![0u16; 3];
            driver::memcpy_dtoh_async(
                buf.as_mut_ptr() as *mut u8,
                out.as_gpu_tensor().raw_ptr(),
                6,
                stream,
            )
            .expect("D2H");
            driver::stream_synchronize(stream).expect("sync");
            let vals: Vec<f32> = buf
                .iter()
                .map(|&b| half::bf16::from_bits(b).to_f32())
                .collect();
            assert!((vals[0] - 4.0).abs() < 0.1);
            assert!((vals[1] - 5.0).abs() < 0.1);
            assert!((vals[2] - 6.0).abs() < 0.1);

            let _ = driver::mem_free(ptr);
        }
    }

    // -- pool_mean_f32 tests --

    #[test]
    #[ignore]
    fn test_pool_mean_basic() {
        unsafe {
            let stream = test_init();
            let mut alloc = CachingAllocator::new();
            let cublas = crate::cublas::CublasHandle::new(stream, &mut alloc).expect("cublas");

            // 3 tokens, hidden_size=3
            // row0=[1,2,3], row1=[3,4,5], row2=[5,6,7] → mean=[3,4,5]
            let data: Vec<f32> = vec![1.0, 2.0, 3.0, 3.0, 4.0, 5.0, 5.0, 6.0, 7.0];
            let ptr = upload_f32(&data, stream);
            let hs = GpuTensor::new(ptr, &[3, 3], DType::F32);

            let out = pool_mean_f32(hs, &cublas, &mut alloc, stream);
            driver::stream_synchronize(stream).expect("sync");
            let result = download_f32(out.as_gpu_tensor().raw_ptr(), 3, stream);
            assert!((result[0] - 3.0).abs() < 1e-5, "got {}", result[0]);
            assert!((result[1] - 4.0).abs() < 1e-5, "got {}", result[1]);
            assert!((result[2] - 5.0).abs() < 1e-5, "got {}", result[2]);

            let _ = driver::mem_free(ptr);
        }
    }

    #[test]
    #[ignore]
    fn test_pool_mean_single_row() {
        unsafe {
            let stream = test_init();
            let mut alloc = CachingAllocator::new();
            let cublas = crate::cublas::CublasHandle::new(stream, &mut alloc).expect("cublas");

            let data: Vec<f32> = vec![7.0, 8.0, 9.0, 10.0];
            let ptr = upload_f32(&data, stream);
            let hs = GpuTensor::new(ptr, &[1, 4], DType::F32);

            let out = pool_mean_f32(hs, &cublas, &mut alloc, stream);
            driver::stream_synchronize(stream).expect("sync");
            let result = download_f32(out.as_gpu_tensor().raw_ptr(), 4, stream);
            assert_eq!(result, vec![7.0, 8.0, 9.0, 10.0]);

            let _ = driver::mem_free(ptr);
        }
    }

    #[test]
    #[ignore]
    fn test_pool_mean_two_rows() {
        unsafe {
            let stream = test_init();
            let mut alloc = CachingAllocator::new();
            let cublas = crate::cublas::CublasHandle::new(stream, &mut alloc).expect("cublas");

            // [0, 4] and [2, 6] → mean [1, 5]
            let data: Vec<f32> = vec![0.0, 4.0, 2.0, 6.0];
            let ptr = upload_f32(&data, stream);
            let hs = GpuTensor::new(ptr, &[2, 2], DType::F32);

            let out = pool_mean_f32(hs, &cublas, &mut alloc, stream);
            driver::stream_synchronize(stream).expect("sync");
            let result = download_f32(out.as_gpu_tensor().raw_ptr(), 2, stream);
            assert!((result[0] - 1.0).abs() < 1e-5);
            assert!((result[1] - 5.0).abs() < 1e-5);

            let _ = driver::mem_free(ptr);
        }
    }
}

// ---------------------------------------------------------------------------
// Tensor concatenation along dim 1
// ---------------------------------------------------------------------------

/// Concatenate two 2D tensors along dimension 1 (column-wise).
///
/// * `a`: `[M, Na]`
/// * `b`: `[M, Nb]`
/// * Returns: `[M, Na + Nb]` from caching allocator.
///
/// Uses row-by-row D2D copies. For the quantized MLP path (separate gate + up
/// projections), this replaces the fused gate_up dense GEMM approach.
pub unsafe fn concat_dim1(
    a: GpuTensor,
    b: GpuTensor,
    alloc: &mut CachingAllocator,
    stream: CUstream,
) -> OwnedTensor {
    let m = a.dim(0);
    assert_eq!(m, b.dim(0), "concat_dim1: row count mismatch");
    let na = a.dim(1);
    let nb = b.dim(1);
    let n_out = na + nb;
    let dtype = a.dtype();
    let elem = dtype.size_bytes();

    let out = alloc.alloc_tensor(&[m, n_out], dtype);

    // Copy row by row: for each row i, copy a[i] then b[i] into out[i]
    for i in 0..m {
        let dst_base = out.raw_ptr().add(i * n_out * elem);
        let src_a = (a.raw_ptr() as *const u8).add(i * na * elem);
        let src_b = (b.raw_ptr() as *const u8).add(i * nb * elem);
        crate::driver::memcpy_dtod_async(dst_base, src_a, na * elem, stream)
            .expect("concat_dim1: D2D copy a");
        crate::driver::memcpy_dtod_async(dst_base.add(na * elem), src_b, nb * elem, stream)
            .expect("concat_dim1: D2D copy b");
    }

    out
}

// ---------------------------------------------------------------------------
// Marlin INT4 GEMM (AWQ/GPTQ → Marlin format)
// ---------------------------------------------------------------------------

unsafe extern "C" {
    fn marlin_gemm_f16(
        a: *const c_void,
        b_q_weight: *const c_void,
        c: *mut c_void,
        b_scales: *const c_void,
        b_zeros: *const c_void,
        g_idx: *const c_void,
        perm: *const c_void,
        b_bias: *const c_void,
        workspace: *mut c_void,
        c_tmp: *mut c_void,
        a_tmp: *mut c_void,
        size_m: c_int,
        size_n: c_int,
        size_k: c_int,
        lda: c_int,
        num_groups: c_int,
        group_size: c_int,
        has_act_order: bool,
        is_k_full: bool,
        has_zp: bool,
        is_zp_float: bool,
        use_fp32_reduce: bool,
        has_bias: bool,
        b_type_id: c_int,
        stream: CUstream,
        device_id: c_int,
    );

    fn marlin_gemm_bf16(
        a: *const c_void,
        b_q_weight: *const c_void,
        c: *mut c_void,
        b_scales: *const c_void,
        b_zeros: *const c_void,
        g_idx: *const c_void,
        perm: *const c_void,
        b_bias: *const c_void,
        workspace: *mut c_void,
        c_tmp: *mut c_void,
        a_tmp: *mut c_void,
        size_m: c_int,
        size_n: c_int,
        size_k: c_int,
        lda: c_int,
        num_groups: c_int,
        group_size: c_int,
        has_act_order: bool,
        is_k_full: bool,
        has_zp: bool,
        is_zp_float: bool,
        use_fp32_reduce: bool,
        has_bias: bool,
        b_type_id: c_int,
        stream: CUstream,
        device_id: c_int,
    );

    fn awq_marlin_repack_4bit(
        b_q_weight: *const u32,
        out: *mut u32,
        size_k: c_int,
        size_n: c_int,
        stream: CUstream,
        device_id: c_int,
    );

    fn gptq_marlin_repack_4bit(
        b_q_weight: *const u32,
        perm: *const u32,
        out: *mut u32,
        size_k: c_int,
        size_n: c_int,
        has_perm: bool,
        stream: CUstream,
        device_id: c_int,
    );
}

/// Marlin INT4×FP16→FP16 fused GEMM.
///
/// * `a`: `[M, K]` activation tensor (F16 or BF16)
/// * `b_q_weight`: Marlin-tiled packed INT4 weights
/// * `b_scales`: `[num_groups, N]` scales (same dtype as `a`)
/// * `b_zeros`: packed zero points (or null tensor for symmetric)
/// * `g_idx`: group index for act_order (or null tensor)
/// * `perm`: permutation for act_order (or null tensor)
/// * `workspace`: `[num_sms]` i32 workspace buffer
/// * Returns: `[M, N]` output from caching allocator
#[allow(clippy::too_many_arguments)]
pub unsafe fn marlin_gemm(
    a: GpuTensor,
    b_q_weight: GpuTensor,
    b_scales: GpuTensor,
    b_zeros: Option<GpuTensor>,
    g_idx: Option<GpuTensor>,
    perm: Option<GpuTensor>,
    b_bias: Option<GpuTensor>,
    workspace: GpuTensor,
    size_m: usize,
    size_n: usize,
    size_k: usize,
    num_groups: usize,
    group_size: usize,
    has_act_order: bool,
    has_zp: bool,
    b_type_id: i32,
    device_id: i32,
    alloc: &mut CachingAllocator,
    stream: CUstream,
) -> OwnedTensor {
    let out = alloc.alloc_tensor(&[size_m, size_n], a.dtype());

    // FP32 reduction buffer — matches Python vLLM's USE_FP32_REDUCE_DEFAULT=True.
    // Partial sums across K-splits are accumulated in f32 for numerical
    // accuracy. The kernel indexes via `locks_off * c_size` where
    // `locks_off` can reach `sms` and `c_size = tb_m * tb_n * f32` per
    // compilation unit, so the allocation has to reach the worst-case
    // `sms * max_m_block_size * max_thread_n` floats (Python vLLM's
    // `marlin.cu` uses this exact formula; `max_thread_n = 256` matches
    // `marlin.cuh`). Undersizing this buffer silently corrupts reductions
    // and produces NaN logits — which is exactly the AWQ/GPTQ bug we had
    // until this line ran `sms * max_m_block_size * 256`.
    const MAX_THREAD_N: usize = 256;
    let sms = unsafe {
        crate::driver::device_get_num_sm(device_id as cudarc::driver::sys::CUdevice)
            .expect("cuDeviceGetAttribute(MULTIPROCESSOR_COUNT) succeeds") as usize
    };
    let max_m_block_size = (size_m.div_ceil(16) * 16).min(64);
    let c_tmp_size = sms * max_m_block_size * MAX_THREAD_N;
    let c_tmp = alloc.alloc_tensor(&[c_tmp_size], DType::F32);

    // When has_act_order, the C++ permute_cols_kernel writes column-permuted
    // activations into a_tmp before GEMM. Same shape/dtype as input `a`.
    let a_tmp_ptr: *mut c_void = if has_act_order {
        let a_tmp = alloc.alloc_tensor(&[size_m, size_k], a.dtype());
        a_tmp.raw_ptr() as *mut c_void
    } else {
        std::ptr::null_mut()
    };

    let zeros_ptr = b_zeros.map_or(std::ptr::null(), |t| t.raw_ptr() as *const c_void);
    let g_idx_ptr = g_idx.map_or(std::ptr::null(), |t| t.raw_ptr() as *const c_void);
    let perm_ptr = perm.map_or(std::ptr::null(), |t| t.raw_ptr() as *const c_void);
    let bias_ptr = b_bias.map_or(std::ptr::null(), |t| t.raw_ptr() as *const c_void);
    let has_bias = b_bias.is_some();

    // is_k_full = true: we always process the complete K dimension.
    // (is_k_full=false is only for expert-parallelism K-slicing, which we don't do.)
    let is_k_full = true;

    match a.dtype() {
        DType::F16 => marlin_gemm_f16(
            a.raw_ptr() as *const c_void,
            b_q_weight.raw_ptr() as *const c_void,
            out.raw_ptr() as *mut c_void,
            b_scales.raw_ptr() as *const c_void,
            zeros_ptr,
            g_idx_ptr,
            perm_ptr,
            bias_ptr,
            workspace.raw_ptr() as *mut c_void,
            c_tmp.raw_ptr() as *mut c_void,
            a_tmp_ptr,
            size_m as c_int,
            size_n as c_int,
            size_k as c_int,
            size_k as c_int, // lda = size_k for row-major
            num_groups as c_int,
            group_size as c_int,
            has_act_order,
            is_k_full,
            has_zp,
            false, // is_zp_float
            true,  // use_fp32_reduce (matches Python vLLM default)
            has_bias,
            b_type_id as c_int,
            stream,
            device_id,
        ),
        DType::BF16 => marlin_gemm_bf16(
            a.raw_ptr() as *const c_void,
            b_q_weight.raw_ptr() as *const c_void,
            out.raw_ptr() as *mut c_void,
            b_scales.raw_ptr() as *const c_void,
            zeros_ptr,
            g_idx_ptr,
            perm_ptr,
            bias_ptr,
            workspace.raw_ptr() as *mut c_void,
            c_tmp.raw_ptr() as *mut c_void,
            a_tmp_ptr,
            size_m as c_int,
            size_n as c_int,
            size_k as c_int,
            size_k as c_int,
            num_groups as c_int,
            group_size as c_int,
            has_act_order,
            is_k_full,
            has_zp,
            false, // is_zp_float
            true,  // use_fp32_reduce (matches Python vLLM default)
            has_bias,
            b_type_id as c_int,
            stream,
            device_id,
        ),
        _ => panic!("marlin_gemm: unsupported dtype {:?}", a.dtype()),
    }

    out
}

/// Repack AWQ INT4 weights to Marlin tiled layout (GPU kernel).
///
/// * `b_q_weight`: `[K, N/8]` packed AWQ weights (u32, on GPU)
/// * `size_k`: number of input features
/// * `size_n`: number of output features
/// * Returns: repacked weights from caching allocator
///   Repack AWQ INT4 weights into a pre-allocated buffer.
pub unsafe fn awq_repack_into(
    b_q_weight: GpuTensor,
    out_ptr: *mut u8,
    size_k: usize,
    size_n: usize,
    device_id: i32,
    stream: CUstream,
) {
    awq_marlin_repack_4bit(
        b_q_weight.as_ptr::<u32>(),
        out_ptr as *mut u32,
        size_k as c_int,
        size_n as c_int,
        stream,
        device_id,
    );
}

pub unsafe fn awq_repack(
    b_q_weight: GpuTensor,
    size_k: usize,
    size_n: usize,
    device_id: i32,
    alloc: &mut CachingAllocator,
    stream: CUstream,
) -> OwnedTensor {
    let num_u32 = size_k * size_n / 8;
    let out = alloc.alloc_tensor(&[num_u32], DType::U32);

    awq_marlin_repack_4bit(
        b_q_weight.as_ptr::<u32>(),
        out.as_mut_ptr::<u32>(),
        size_k as c_int,
        size_n as c_int,
        stream,
        device_id,
    );

    out
}

/// Repack GPTQ INT4 weights to Marlin tiled layout into a pre-allocated buffer.
///
/// Use this for model weights (persistent allocations that must survive
/// `free_leaked_blocks`). The caller allocates via `driver::mem_alloc`.
pub unsafe fn gptq_repack_into(
    b_q_weight: GpuTensor,
    perm: Option<GpuTensor>,
    out_ptr: *mut u8,
    size_k: usize,
    size_n: usize,
    device_id: i32,
    stream: CUstream,
) {
    let (perm_ptr, has_perm) = match perm {
        Some(p) => (p.as_ptr::<u32>(), true),
        None => (std::ptr::null(), false),
    };

    gptq_marlin_repack_4bit(
        b_q_weight.as_ptr::<u32>(),
        perm_ptr,
        out_ptr as *mut u32,
        size_k as c_int,
        size_n as c_int,
        has_perm,
        stream,
        device_id,
    );
}

/// Repack GPTQ INT4 weights to Marlin tiled layout (GPU kernel).
///
/// * `b_q_weight`: `[K/8, N]` packed GPTQ weights (u32, on GPU)
/// * `perm`: optional `[K]` permutation (for act_order)
/// * `size_k`: number of input features
/// * `size_n`: number of output features
/// * Returns: repacked weights from caching allocator
pub unsafe fn gptq_repack(
    b_q_weight: GpuTensor,
    perm: Option<GpuTensor>,
    size_k: usize,
    size_n: usize,
    device_id: i32,
    alloc: &mut CachingAllocator,
    stream: CUstream,
) -> OwnedTensor {
    let num_u32 = size_k * size_n / 8;
    let out = alloc.alloc_tensor(&[num_u32], DType::U32);

    let (perm_ptr, has_perm) = match perm {
        Some(p) => (p.as_ptr::<u32>(), true),
        None => (std::ptr::null(), false),
    };

    gptq_marlin_repack_4bit(
        b_q_weight.as_ptr::<u32>(),
        perm_ptr,
        out.as_mut_ptr::<u32>(),
        size_k as c_int,
        size_n as c_int,
        has_perm,
        stream,
        device_id,
    );

    out
}

// ---------------------------------------------------------------------------
// Marlin MoE INT4 GEMM (AWQ/GPTQ MoE → Marlin format)
// ---------------------------------------------------------------------------

unsafe extern "C" {
    fn marlin_moe_gemm_bf16(
        a: *const c_void,
        c: *mut c_void,
        c_tmp: *mut c_void,
        b_q_weight: *const c_void,
        b_scales: *const c_void,
        b_zeros: *const c_void,
        g_idx: *const c_void,
        perm: *const c_void,
        a_tmp: *mut c_void,
        workspace: *mut c_void,
        sorted_token_ids: *const c_void,
        expert_ids: *const c_void,
        num_tokens_past_padded: *const c_void,
        topk_weights: *const c_void,
        moe_block_size: c_int,
        num_experts: c_int,
        top_k: c_int,
        mul_topk_weights: bool,
        size_m: c_int,
        size_n: c_int,
        size_k: c_int,
        num_groups: c_int,
        group_size: c_int,
        has_act_order: bool,
        is_k_full: bool,
        has_zp: bool,
        is_zp_float: bool,
        use_fp32_reduce: bool,
        b_type_id: c_int,
        stream: CUstream,
        device_id: c_int,
    );

    fn marlin_moe_gemm_f16(
        a: *const c_void,
        c: *mut c_void,
        c_tmp: *mut c_void,
        b_q_weight: *const c_void,
        b_scales: *const c_void,
        b_zeros: *const c_void,
        g_idx: *const c_void,
        perm: *const c_void,
        a_tmp: *mut c_void,
        workspace: *mut c_void,
        sorted_token_ids: *const c_void,
        expert_ids: *const c_void,
        num_tokens_past_padded: *const c_void,
        topk_weights: *const c_void,
        moe_block_size: c_int,
        num_experts: c_int,
        top_k: c_int,
        mul_topk_weights: bool,
        size_m: c_int,
        size_n: c_int,
        size_k: c_int,
        num_groups: c_int,
        group_size: c_int,
        has_act_order: bool,
        is_k_full: bool,
        has_zp: bool,
        is_zp_float: bool,
        use_fp32_reduce: bool,
        b_type_id: c_int,
        stream: CUstream,
        device_id: c_int,
    );
}

/// Marlin MoE fused INT4 GEMM — routes tokens to experts via sorted IDs.
///
/// * `a`:            `[M, K]` activation tensor (BF16 or F16)
/// * `b_q_weight`:   `[E, K/tile, N*tile]` Marlin-packed expert weights
/// * `b_scales`:     `[E, num_groups, N]` scales
/// * `b_zeros`:      `[E, num_groups, N/8]` zero points (AWQ) or None
/// * `workspace`:    barrier locks, at least `[sms * 4]` i32
/// * `sorted_token_ids`, `expert_ids`, `num_tokens_past_padded`: from `moe_align_block_size`
/// * `topk_weights`: `[M, top_k]` f32
/// * Returns: `[M * top_k, N]` output tensor
#[allow(clippy::too_many_arguments)]
pub unsafe fn marlin_moe_gemm(
    a: GpuTensor,
    b_q_weight: GpuTensor,
    b_scales: GpuTensor,
    b_zeros: Option<GpuTensor>,
    g_idx: Option<GpuTensor>,
    perm: Option<GpuTensor>,
    workspace: GpuTensor,
    sorted_token_ids: GpuTensor,
    expert_ids: GpuTensor,
    num_tokens_past_padded: GpuTensor,
    topk_weights: GpuTensor,
    moe_block_size: usize,
    num_experts: usize,
    top_k: usize,
    mul_topk_weights: bool,
    size_m: usize,
    size_n: usize,
    size_k: usize,
    num_groups: usize,
    group_size: usize,
    has_act_order: bool,
    has_zp: bool,
    b_type_id: i32,
    device_id: i32,
    alloc: &mut CachingAllocator,
    stream: CUstream,
) -> OwnedTensor {
    // GUARD: the Marlin kernel reads `b_scales` as the SAME scalar type it
    // dispatches on (`a.dtype()`). F16 on-disk scales handed to the bf16 kernel
    // are reinterpreted bit-for-bit (a real ~0.15 scale → ~1e-9), silently
    // killing every expert. Convert scales to the activation dtype at load
    // (see load_gemma_marlin_experts); this fails loudly if that is missed.
    assert_eq!(
        b_scales.dtype(),
        a.dtype(),
        "marlin_moe_gemm: scale dtype {:?} != activation dtype {:?}; the kernel \
         reads scales as the activation type and would misread mismatched scales \
         (dead experts).",
        b_scales.dtype(),
        a.dtype()
    );
    let out = alloc.alloc_tensor(&[size_m * top_k, size_n], a.dtype());

    // FP32 reduction buffer for global_reduce_fp32 — sized to match Python vLLM's
    // ops.cu: min(size_n * sorted_token_ids.size(0), sms * 4 * moe_block_size * max_thread_n)
    // Each threadblock slice needs its own c_tmp region indexed by locks_off.
    let sorted_token_count = sorted_token_ids.dim(0);
    let sms = unsafe { crate::driver::device_get_num_sm(device_id) }.unwrap_or(128) as usize;
    const MAX_THREAD_N: usize = 256; // from marlin.cuh
    let max_c_tmp_size = std::cmp::min(
        size_n * sorted_token_count,
        sms * 4 * moe_block_size * MAX_THREAD_N,
    );
    let max_c_tmp_size = if moe_block_size == 8 {
        max_c_tmp_size * 2
    } else {
        max_c_tmp_size
    };
    let c_tmp = alloc.alloc_tensor(&[max_c_tmp_size], DType::F32);

    // Temp buffer for act_order column permutation
    let a_tmp_ptr: *mut c_void = if has_act_order {
        let a_tmp = alloc.alloc_tensor(&[size_m * top_k, size_k], a.dtype());
        a_tmp.raw_ptr() as *mut c_void
    } else {
        std::ptr::null_mut()
    };

    let zeros_ptr = b_zeros.map_or(std::ptr::null(), |t| t.raw_ptr() as *const c_void);
    let g_idx_ptr = g_idx.map_or(std::ptr::null(), |t| t.raw_ptr() as *const c_void);
    let perm_ptr = perm.map_or(std::ptr::null(), |t| t.raw_ptr() as *const c_void);

    let is_k_full = true;

    match a.dtype() {
        DType::BF16 => marlin_moe_gemm_bf16(
            a.raw_ptr() as *const c_void,
            out.raw_ptr() as *mut c_void,
            c_tmp.raw_ptr() as *mut c_void,
            b_q_weight.raw_ptr() as *const c_void,
            b_scales.raw_ptr() as *const c_void,
            zeros_ptr,
            g_idx_ptr,
            perm_ptr,
            a_tmp_ptr,
            workspace.raw_ptr() as *mut c_void,
            sorted_token_ids.raw_ptr() as *const c_void,
            expert_ids.raw_ptr() as *const c_void,
            num_tokens_past_padded.raw_ptr() as *const c_void,
            topk_weights.raw_ptr() as *const c_void,
            moe_block_size as c_int,
            num_experts as c_int,
            top_k as c_int,
            mul_topk_weights,
            size_m as c_int,
            size_n as c_int,
            size_k as c_int,
            num_groups as c_int,
            group_size as c_int,
            has_act_order,
            is_k_full,
            has_zp,
            false, // is_zp_float
            true,  // use_fp32_reduce
            b_type_id as c_int,
            stream,
            device_id,
        ),
        DType::F16 => marlin_moe_gemm_f16(
            a.raw_ptr() as *const c_void,
            out.raw_ptr() as *mut c_void,
            c_tmp.raw_ptr() as *mut c_void,
            b_q_weight.raw_ptr() as *const c_void,
            b_scales.raw_ptr() as *const c_void,
            zeros_ptr,
            g_idx_ptr,
            perm_ptr,
            a_tmp_ptr,
            workspace.raw_ptr() as *mut c_void,
            sorted_token_ids.raw_ptr() as *const c_void,
            expert_ids.raw_ptr() as *const c_void,
            num_tokens_past_padded.raw_ptr() as *const c_void,
            topk_weights.raw_ptr() as *const c_void,
            moe_block_size as c_int,
            num_experts as c_int,
            top_k as c_int,
            mul_topk_weights,
            size_m as c_int,
            size_n as c_int,
            size_k as c_int,
            num_groups as c_int,
            group_size as c_int,
            has_act_order,
            is_k_full,
            has_zp,
            false, // is_zp_float
            true,  // use_fp32_reduce
            b_type_id as c_int,
            stream,
            device_id,
        ),
        _ => panic!("marlin_moe_gemm: unsupported dtype {:?}", a.dtype()),
    }

    out
}

// ---------------------------------------------------------------------------
// BitsAndBytes NF4/FP4 dequantization
// ---------------------------------------------------------------------------

unsafe extern "C" {
    fn dequantize_nf4_bf16(
        packed: *const u8,
        absmax: *const f32,
        code: *const f32,
        out: *mut u8, // actually bf16
        num_packed: i64,
        blocksize: c_int,
        stream: CUstream,
    );
    fn dequantize_nf4_f16(
        packed: *const u8,
        absmax: *const f32,
        code: *const f32,
        out: *mut u8, // actually f16
        num_packed: i64,
        blocksize: c_int,
        stream: CUstream,
    );
}

/// Dequantize NF4/FP4 packed weights to BF16 or F16.
///
/// * `packed` — `[num_packed]` U8 tensor (2 nibbles per byte)
/// * `absmax` — `[num_blocks]` F32 per-block scale factors
/// * `code` — `[16]` F32 lookup table (NF4 or FP4)
/// * `out` — pre-allocated `[num_elements]` BF16 or F16 tensor
/// * `blocksize` — elements per quantization block (typically 64)
pub unsafe fn dequantize_bnb4bit(
    packed: GpuTensor,
    absmax: GpuTensor,
    code: GpuTensor,
    out: GpuTensor,
    blocksize: usize,
    stream: CUstream,
) {
    let num_packed = packed.numel() as i64;
    match out.dtype() {
        DType::BF16 => dequantize_nf4_bf16(
            packed.raw_ptr(),
            absmax.raw_ptr() as *const f32,
            code.raw_ptr() as *const f32,
            out.raw_ptr() as *mut u8,
            num_packed,
            blocksize as c_int,
            stream,
        ),
        DType::F16 => dequantize_nf4_f16(
            packed.raw_ptr(),
            absmax.raw_ptr() as *const f32,
            code.raw_ptr() as *const f32,
            out.raw_ptr() as *mut u8,
            num_packed,
            blocksize as c_int,
            stream,
        ),
        other => panic!("dequantize_bnb4bit: unsupported output dtype {other}"),
    }
}

// ---------------------------------------------------------------------------
// QK-norm only (no RoPE) — for partial-RoPE models like Qwen3-Next
// ---------------------------------------------------------------------------

unsafe extern "C" {
    fn qk_norm_f32(
        query: *mut f32,
        key: *mut f32,
        q_weight: *const f32,
        k_weight: *const f32,
        epsilon: f32,
        q_weight_offset: f32,
        k_weight_offset: f32,
        num_q_heads: c_int,
        num_kv_heads: c_int,
        head_dim: c_int,
        num_tokens: c_int,
        stream: CUstream,
    );
    fn qk_norm_f16(
        query: *mut u16,
        key: *mut u16,
        q_weight: *const u16,
        k_weight: *const u16,
        epsilon: f32,
        q_weight_offset: f32,
        k_weight_offset: f32,
        num_q_heads: c_int,
        num_kv_heads: c_int,
        head_dim: c_int,
        num_tokens: c_int,
        stream: CUstream,
    );
    fn qk_norm_bf16(
        query: *mut u16,
        key: *mut u16,
        q_weight: *const u16,
        k_weight: *const u16,
        epsilon: f32,
        q_weight_offset: f32,
        k_weight_offset: f32,
        num_q_heads: c_int,
        num_kv_heads: c_int,
        head_dim: c_int,
        num_tokens: c_int,
        stream: CUstream,
    );
}

/// Apply per-head QK RMS norm in-place (no RoPE).
///
/// Q: `[num_tokens, num_q_heads, head_dim]`
/// K: `[num_tokens, num_kv_heads, head_dim]`
/// Weights use GemmaRMSNorm convention: `weight_offset` is added to
/// each weight element (`0.0` for plain weights, `1.0` for Gemma's
/// `(1+w)` convention).
#[allow(clippy::too_many_arguments)]
pub unsafe fn qk_norm_inplace(
    query: GpuTensor,
    key: GpuTensor,
    q_weight: GpuTensor,
    k_weight: GpuTensor,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    epsilon: f32,
    q_weight_offset: f32,
    k_weight_offset: f32,
    stream: CUstream,
) {
    let num_tokens = query.dim(0) as c_int;
    match query.dtype() {
        DType::F32 => qk_norm_f32(
            query.as_mut_ptr(),
            key.as_mut_ptr(),
            q_weight.as_ptr(),
            k_weight.as_ptr(),
            epsilon,
            q_weight_offset,
            k_weight_offset,
            num_q_heads as c_int,
            num_kv_heads as c_int,
            head_dim as c_int,
            num_tokens,
            stream,
        ),
        DType::F16 => qk_norm_f16(
            query.as_mut_ptr() as *mut u16,
            key.as_mut_ptr() as *mut u16,
            q_weight.as_ptr() as *const u16,
            k_weight.as_ptr() as *const u16,
            epsilon,
            q_weight_offset,
            k_weight_offset,
            num_q_heads as c_int,
            num_kv_heads as c_int,
            head_dim as c_int,
            num_tokens,
            stream,
        ),
        DType::BF16 => qk_norm_bf16(
            query.as_mut_ptr() as *mut u16,
            key.as_mut_ptr() as *mut u16,
            q_weight.as_ptr() as *const u16,
            k_weight.as_ptr() as *const u16,
            epsilon,
            q_weight_offset,
            k_weight_offset,
            num_q_heads as c_int,
            num_kv_heads as c_int,
            head_dim as c_int,
            num_tokens,
            stream,
        ),
        _ => panic!("qk_norm: unsupported dtype {:?}", query.dtype()),
    }
}

/// Apply RoPE in-place to separate Q and K tensors (partial RoPE supported).
///
/// Q: `[num_tokens, num_q_heads * head_dim]` or `[num_tokens, num_q_heads, head_dim]`
/// K: `[num_tokens, num_kv_heads * head_dim]` or `[num_tokens, num_kv_heads, head_dim]`
///
/// This is a thin wrapper over `rotary_embedding_inplace`.
pub unsafe fn apply_rope_qk_inplace(
    q: GpuTensor,
    k: GpuTensor,
    cos_sin_cache: GpuTensor,
    positions: GpuTensor,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    stream: CUstream,
) {
    // Reshape to flat [num_tokens, total_dim] for rotary_embedding_inplace.
    let num_tokens = q.dim(0);
    let q_flat = q.reshape(&[num_tokens, num_q_heads * head_dim]);
    let k_flat = k.reshape(&[num_tokens, num_kv_heads * head_dim]);
    rotary_embedding_inplace(q_flat, k_flat, positions, cos_sin_cache, head_dim, stream);
}

// ---------------------------------------------------------------------------
// Sigmoid-mul: out = sigmoid(gate) * input (in-place on input)
// ---------------------------------------------------------------------------

unsafe extern "C" {
    fn sigmoid_mul_f32(input: *mut f32, gate: *const f32, numel: c_int, stream: CUstream);
    fn sigmoid_mul_f16(input: *mut u16, gate: *const u16, numel: c_int, stream: CUstream);
    fn sigmoid_mul_bf16(input: *mut u16, gate: *const u16, numel: c_int, stream: CUstream);
}

/// Apply `input = sigmoid(gate) * input` element-wise in-place.
pub unsafe fn sigmoid_mul_inplace(
    input: GpuTensor,
    gate: GpuTensor,
    _alloc: &mut CachingAllocator,
    stream: CUstream,
) {
    let numel = input.numel() as c_int;
    match input.dtype() {
        DType::F32 => sigmoid_mul_f32(input.as_mut_ptr(), gate.as_ptr(), numel, stream),
        DType::F16 => sigmoid_mul_f16(
            input.as_mut_ptr() as *mut u16,
            gate.as_ptr() as *const u16,
            numel,
            stream,
        ),
        DType::BF16 => sigmoid_mul_bf16(
            input.as_mut_ptr() as *mut u16,
            gate.as_ptr() as *const u16,
            numel,
            stream,
        ),
        _ => panic!("sigmoid_mul: unsupported dtype {:?}", input.dtype()),
    }
}

// ---------------------------------------------------------------------------
// Sigmoid-rowgate-combine: out = routed + sh * sigmoid(gate[row]) —
// Qwen3.5-MoE shared-expert combine. `gate` is [rows, 1], row-broadcast
// across cols. 4-buffer form mirroring gate_scale.metal: the caller binds
// the routed input and out output directly (no staging copy); aliasing
// out == routed degrades to the in-place accumulate.
// ---------------------------------------------------------------------------

unsafe extern "C" {
    fn sigmoid_rowgate_combine_f32(
        out: *mut f32,
        routed: *const f32,
        sh: *const f32,
        gate: *const f32,
        numel: c_int,
        cols: c_int,
        stream: CUstream,
    );
    fn sigmoid_rowgate_combine_f16(
        out: *mut u16,
        routed: *const u16,
        sh: *const u16,
        gate: *const u16,
        numel: c_int,
        cols: c_int,
        stream: CUstream,
    );
    fn sigmoid_rowgate_combine_bf16(
        out: *mut u16,
        routed: *const u16,
        sh: *const u16,
        gate: *const u16,
        numel: c_int,
        cols: c_int,
        stream: CUstream,
    );
}

/// `out = routed + sh * sigmoid(gate[row])` over `[T, H]` tiles, with
/// `gate` `[T, 1]` row-broadcast across the hidden axis.
pub unsafe fn sigmoid_rowgate_combine(
    out: GpuTensor,
    routed: GpuTensor,
    sh: GpuTensor,
    gate: GpuTensor,
    stream: CUstream,
) {
    let numel = out.numel() as c_int;
    let cols = out.dim(1) as c_int;
    match out.dtype() {
        DType::F32 => sigmoid_rowgate_combine_f32(
            out.as_mut_ptr(),
            routed.as_ptr(),
            sh.as_ptr(),
            gate.as_ptr(),
            numel,
            cols,
            stream,
        ),
        DType::F16 => sigmoid_rowgate_combine_f16(
            out.as_mut_ptr() as *mut u16,
            routed.as_ptr() as *const u16,
            sh.as_ptr() as *const u16,
            gate.as_ptr() as *const u16,
            numel,
            cols,
            stream,
        ),
        DType::BF16 => sigmoid_rowgate_combine_bf16(
            out.as_mut_ptr() as *mut u16,
            routed.as_ptr() as *const u16,
            sh.as_ptr() as *const u16,
            gate.as_ptr() as *const u16,
            numel,
            cols,
            stream,
        ),
        _ => panic!(
            "sigmoid_rowgate_combine: unsupported dtype {:?}",
            out.dtype()
        ),
    }
}

// ---------------------------------------------------------------------------
// GDN kernels (Gated Delta Net for Qwen3-Next)
// ---------------------------------------------------------------------------

unsafe extern "C" {
    fn fused_gdn_gating(
        g_out: *mut f32,
        beta_out: *mut f32,
        A_log: *const f32,
        a: *const f32,
        b: *const f32,
        dt_bias: *const f32,
        num_heads: c_int,
        batch_size: c_int,
        stream: CUstream,
    );

    fn causal_conv1d_update(
        conv_state: *mut f32,
        x: *const f32,
        w: *const f32,
        output: *mut f32,
        state_indices: *const c_int,
        conv_dim: c_int,
        kernel_size: c_int,
        batch_size: c_int,
        stream: CUstream,
    );

    fn causal_conv1d_prefill(
        conv_state: *mut f32,
        x: *const f32,
        w: *const f32,
        output: *mut f32,
        slot_idx: c_int,
        conv_dim: c_int,
        kernel_size: c_int,
        num_tokens: c_int,
        stream: CUstream,
    );

    fn fused_recurrent_gdn_fwd(
        q: *const f32,
        k: *const f32,
        v: *const f32,
        g: *const f32,
        beta: *const f32,
        o: *mut f32,
        ssm_state: *mut f32,
        state_indices: *const c_int,
        cu_seqlens: *const c_int,
        scale: f32,
        N: c_int,
        T: c_int,
        H: c_int,
        HV: c_int,
        K: c_int,
        V_dim: c_int,
        stream: CUstream,
    );

    fn rms_norm_gated(
        x: *const f32,
        z: *const f32,
        weight: *const f32,
        out: *mut f32,
        eps: f32,
        head_v_dim: c_int,
        total_rows: c_int,
        stream: CUstream,
    );

    // GDN QKVZ split kernel
    fn gdn_qkvz_split_bf16(
        qkvz: *const u16,
        ba: *const u16,
        q: *mut f32,
        k: *mut f32,
        v: *mut f32,
        z: *mut f32,
        a: *mut f32,
        b: *mut f32,
        mixed: *mut f32,
        num_tokens: c_int,
        num_k_heads: c_int,
        num_v_heads: c_int,
        head_k_dim: c_int,
        head_v_dim: c_int,
        v_per_k: c_int,
        key_dim: c_int,
        value_dim: c_int,
        qkvz_dim: c_int,
        conv_dim: c_int,
        stream: CUstream,
    );
    fn gdn_qkvz_split_f16(
        qkvz: *const u16,
        ba: *const u16,
        q: *mut f32,
        k: *mut f32,
        v: *mut f32,
        z: *mut f32,
        a: *mut f32,
        b: *mut f32,
        mixed: *mut f32,
        num_tokens: c_int,
        num_k_heads: c_int,
        num_v_heads: c_int,
        head_k_dim: c_int,
        head_v_dim: c_int,
        v_per_k: c_int,
        key_dim: c_int,
        value_dim: c_int,
        qkvz_dim: c_int,
        conv_dim: c_int,
        stream: CUstream,
    );
    fn gdn_qkvz_split_f32(
        qkvz: *const f32,
        ba: *const f32,
        q: *mut f32,
        k: *mut f32,
        v: *mut f32,
        z: *mut f32,
        a: *mut f32,
        b: *mut f32,
        mixed: *mut f32,
        num_tokens: c_int,
        num_k_heads: c_int,
        num_v_heads: c_int,
        head_k_dim: c_int,
        head_v_dim: c_int,
        v_per_k: c_int,
        key_dim: c_int,
        value_dim: c_int,
        qkvz_dim: c_int,
        conv_dim: c_int,
        stream: CUstream,
    );

    // GDN conv output split kernel
    fn gdn_conv_output_split(
        conv_out: *const f32,
        q: *mut f32,
        k: *mut f32,
        v: *mut f32,
        num_tokens: c_int,
        key_dim: c_int,
        value_dim: c_int,
        conv_dim: c_int,
        stream: CUstream,
    );

    // Qwen3.5 attention output-gate split: per-head deinterleave of doubled
    // q_proj output `qg[T, num_heads*2*head_dim]` into `q[T, num_heads*head_dim]`
    // and `gate[T, num_heads*head_dim]`. Mirrors gate_split.metal.
    fn gate_split_bf16(
        q_out: *mut u16,
        gate_out: *mut u16,
        qg: *const u16,
        num_tokens: c_int,
        num_heads: c_int,
        head_dim: c_int,
        stream: CUstream,
    );
    fn gate_split_f16(
        q_out: *mut u16,
        gate_out: *mut u16,
        qg: *const u16,
        num_tokens: c_int,
        num_heads: c_int,
        head_dim: c_int,
        stream: CUstream,
    );
}

/// Fused GDN gating computation (all f32 on GPU).
///
/// Computes `g = -exp(A_log) * softplus(a + dt_bias)` and `beta = sigmoid(b)`.
pub unsafe fn gdn_gating(
    g_out: GpuTensor,
    beta_out: GpuTensor,
    a_log: GpuTensor,
    a: GpuTensor,
    b: GpuTensor,
    dt_bias: GpuTensor,
    num_heads: usize,
    batch_size: usize,
    stream: CUstream,
) {
    fused_gdn_gating(
        g_out.as_mut_ptr(),
        beta_out.as_mut_ptr(),
        a_log.as_ptr(),
        a.as_ptr(),
        b.as_ptr(),
        dt_bias.as_ptr(),
        num_heads as c_int,
        batch_size as c_int,
        stream,
    );
}

/// Causal conv1d single-token update (decode).
pub unsafe fn gdn_conv1d_update(
    conv_state: GpuTensor,
    x: GpuTensor,
    w: GpuTensor,
    output: GpuTensor,
    state_indices: GpuTensor,
    conv_dim: usize,
    kernel_size: usize,
    batch_size: usize,
    stream: CUstream,
) {
    causal_conv1d_update(
        conv_state.as_mut_ptr(),
        x.as_ptr(),
        w.as_ptr(),
        output.as_mut_ptr(),
        state_indices.as_ptr() as *const c_int,
        conv_dim as c_int,
        kernel_size as c_int,
        batch_size as c_int,
        stream,
    );
}

/// Causal conv1d multi-token prefill.
pub unsafe fn gdn_conv1d_prefill(
    conv_state: GpuTensor,
    x: GpuTensor,
    w: GpuTensor,
    output: GpuTensor,
    slot_idx: usize,
    conv_dim: usize,
    kernel_size: usize,
    num_tokens: usize,
    stream: CUstream,
) {
    causal_conv1d_prefill(
        conv_state.as_mut_ptr(),
        x.as_ptr(),
        w.as_ptr(),
        output.as_mut_ptr(),
        slot_idx as c_int,
        conv_dim as c_int,
        kernel_size as c_int,
        num_tokens as c_int,
        stream,
    );
}

/// Fused recurrent GDN forward pass.
///
/// Processes token-by-token recurrence on GPU with L2 norm, decay, delta update.
#[allow(clippy::too_many_arguments)]
pub unsafe fn gdn_recurrent_fwd(
    q: GpuTensor,
    k: GpuTensor,
    v: GpuTensor,
    g: GpuTensor,
    beta: GpuTensor,
    o: GpuTensor,
    ssm_state: GpuTensor,
    state_indices: GpuTensor,
    cu_seqlens: GpuTensor,
    scale: f32,
    num_seqs: usize,
    total_tokens: usize,
    num_k_heads: usize,
    num_v_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    stream: CUstream,
) {
    fused_recurrent_gdn_fwd(
        q.as_ptr(),
        k.as_ptr(),
        v.as_ptr(),
        g.as_ptr(),
        beta.as_ptr(),
        o.as_mut_ptr(),
        ssm_state.as_mut_ptr(),
        state_indices.as_ptr() as *const c_int,
        cu_seqlens.as_ptr() as *const c_int,
        scale,
        num_seqs as c_int,
        total_tokens as c_int,
        num_k_heads as c_int,
        num_v_heads as c_int,
        head_k_dim as c_int,
        head_v_dim as c_int,
        stream,
    );
}

/// RMS norm with SiLU gating: `out = rms_norm(x) * weight * silu(z)` where
/// `silu(z) = z * sigmoid(z)` (norm_before_gate; matches Python `RMSNormGated`
/// and the cpu_golden oracle — NOT plain sigmoid).
pub unsafe fn gdn_rms_norm_gated(
    x: GpuTensor,
    z: GpuTensor,
    weight: GpuTensor,
    out: GpuTensor,
    eps: f32,
    head_v_dim: usize,
    total_rows: usize,
    stream: CUstream,
) {
    rms_norm_gated(
        x.as_ptr(),
        z.as_ptr(),
        weight.as_ptr(),
        out.as_mut_ptr(),
        eps,
        head_v_dim as c_int,
        total_rows as c_int,
        stream,
    );
}

// ---------------------------------------------------------------------------
// GDN QKVZ Split (GPU)
// ---------------------------------------------------------------------------

/// Split QKVZ and BA projection outputs into individual f32 tensors on GPU.
///
/// Also produces concatenated Q||K||V for conv1d input.
///
/// * `qkvz`: `[T, 2*key_dim + 2*value_dim]` in model dtype
/// * `ba`:   `[T, 2*num_v_heads]` in model dtype
///
/// Returns: (q, k, v, z, a, b, mixed_qkv) all f32 on GPU.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn gdn_qkvz_split(
    qkvz: GpuTensor,
    ba: GpuTensor,
    num_tokens: usize,
    num_k_heads: usize,
    num_v_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    key_dim: usize,
    value_dim: usize,
    conv_dim: usize,
    caching: &mut crate::alloc::CachingAllocator,
    stream: CUstream,
) -> (
    crate::alloc::OwnedTensor, // q [T, key_dim]
    crate::alloc::OwnedTensor, // k [T, key_dim]
    crate::alloc::OwnedTensor, // v [T, value_dim]
    crate::alloc::OwnedTensor, // z [T, value_dim]
    crate::alloc::OwnedTensor, // a [T, num_v_heads]
    crate::alloc::OwnedTensor, // b [T, num_v_heads]
    crate::alloc::OwnedTensor, // mixed_qkv [T, conv_dim]
) {
    let q = caching.alloc_tensor(&[num_tokens, key_dim], DType::F32);
    let k = caching.alloc_tensor(&[num_tokens, key_dim], DType::F32);
    let v = caching.alloc_tensor(&[num_tokens, value_dim], DType::F32);
    let z = caching.alloc_tensor(&[num_tokens, value_dim], DType::F32);
    let a = caching.alloc_tensor(&[num_tokens, num_v_heads], DType::F32);
    let b = caching.alloc_tensor(&[num_tokens, num_v_heads], DType::F32);
    let mixed = caching.alloc_tensor(&[num_tokens, conv_dim], DType::F32);

    let v_per_k = (num_v_heads / num_k_heads) as c_int;
    let qkvz_dim = (2 * key_dim + 2 * value_dim) as c_int;

    match qkvz.dtype() {
        DType::BF16 => gdn_qkvz_split_bf16(
            qkvz.as_ptr(),
            ba.as_ptr(),
            q.as_gpu_tensor().as_mut_ptr(),
            k.as_gpu_tensor().as_mut_ptr(),
            v.as_gpu_tensor().as_mut_ptr(),
            z.as_gpu_tensor().as_mut_ptr(),
            a.as_gpu_tensor().as_mut_ptr(),
            b.as_gpu_tensor().as_mut_ptr(),
            mixed.as_gpu_tensor().as_mut_ptr(),
            num_tokens as c_int,
            num_k_heads as c_int,
            num_v_heads as c_int,
            head_k_dim as c_int,
            head_v_dim as c_int,
            v_per_k,
            key_dim as c_int,
            value_dim as c_int,
            qkvz_dim,
            conv_dim as c_int,
            stream,
        ),
        DType::F16 => gdn_qkvz_split_f16(
            qkvz.as_ptr(),
            ba.as_ptr(),
            q.as_gpu_tensor().as_mut_ptr(),
            k.as_gpu_tensor().as_mut_ptr(),
            v.as_gpu_tensor().as_mut_ptr(),
            z.as_gpu_tensor().as_mut_ptr(),
            a.as_gpu_tensor().as_mut_ptr(),
            b.as_gpu_tensor().as_mut_ptr(),
            mixed.as_gpu_tensor().as_mut_ptr(),
            num_tokens as c_int,
            num_k_heads as c_int,
            num_v_heads as c_int,
            head_k_dim as c_int,
            head_v_dim as c_int,
            v_per_k,
            key_dim as c_int,
            value_dim as c_int,
            qkvz_dim,
            conv_dim as c_int,
            stream,
        ),
        DType::F32 => gdn_qkvz_split_f32(
            qkvz.as_ptr(),
            ba.as_ptr(),
            q.as_gpu_tensor().as_mut_ptr(),
            k.as_gpu_tensor().as_mut_ptr(),
            v.as_gpu_tensor().as_mut_ptr(),
            z.as_gpu_tensor().as_mut_ptr(),
            a.as_gpu_tensor().as_mut_ptr(),
            b.as_gpu_tensor().as_mut_ptr(),
            mixed.as_gpu_tensor().as_mut_ptr(),
            num_tokens as c_int,
            num_k_heads as c_int,
            num_v_heads as c_int,
            head_k_dim as c_int,
            head_v_dim as c_int,
            v_per_k,
            key_dim as c_int,
            value_dim as c_int,
            qkvz_dim,
            conv_dim as c_int,
            stream,
        ),
        _ => panic!("gdn_qkvz_split: unsupported dtype {:?}", qkvz.dtype()),
    }

    (q, k, v, z, a, b, mixed)
}

/// Split conv1d output [T, conv_dim] into Q, K, V on GPU.
///
/// conv_dim = 2*key_dim + value_dim, layout is Q||K||V flat.
#[cfg(feature = "cuda")]
pub unsafe fn gdn_conv_split(
    conv_out: GpuTensor,
    num_tokens: usize,
    num_k_heads: usize,
    num_v_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    key_dim: usize,
    value_dim: usize,
    conv_dim: usize,
    caching: &mut crate::alloc::CachingAllocator,
    stream: CUstream,
) -> (
    crate::alloc::OwnedTensor, // q [T, num_k_heads, head_k_dim]
    crate::alloc::OwnedTensor, // k [T, num_k_heads, head_k_dim]
    crate::alloc::OwnedTensor, // v [T, num_v_heads, head_v_dim]
) {
    let q = caching.alloc_tensor(&[num_tokens, num_k_heads, head_k_dim], DType::F32);
    let k = caching.alloc_tensor(&[num_tokens, num_k_heads, head_k_dim], DType::F32);
    let v = caching.alloc_tensor(&[num_tokens, num_v_heads, head_v_dim], DType::F32);

    gdn_conv_output_split(
        conv_out.as_ptr(),
        q.as_gpu_tensor().as_mut_ptr(),
        k.as_gpu_tensor().as_mut_ptr(),
        v.as_gpu_tensor().as_mut_ptr(),
        num_tokens as c_int,
        key_dim as c_int,
        value_dim as c_int,
        conv_dim as c_int,
        stream,
    );

    (q, k, v)
}

/// Qwen3.5 attention output-gate split (CUDA mirror of gate_split.metal).
///
/// Per-head deinterleave of the doubled q_proj output:
///   `qg[T, num_heads*2*head_dim]` → `q[T, num_heads*head_dim]`, `gate[T, num_heads*head_dim]`.
/// Each head's `2*head_dim` block in `qg` is laid out `[query | gate]`.
///
/// Allocates the two outputs from `alloc` with `qg.dtype()`. Supports BF16/F16
/// only (matches the metal kernel's host-name instantiations).
#[cfg(feature = "cuda")]
pub unsafe fn gate_split(
    qg: GpuTensor,
    num_heads: usize,
    head_dim: usize,
    alloc: &mut crate::alloc::CachingAllocator,
    stream: CUstream,
) -> (
    crate::alloc::OwnedTensor, // q    [num_tokens, num_heads * head_dim]
    crate::alloc::OwnedTensor, // gate [num_tokens, num_heads * head_dim]
) {
    let num_tokens = qg.dim(0);
    debug_assert_eq!(
        qg.dim(1),
        num_heads * 2 * head_dim,
        "gate_split: qg.dim(1)={} but num_heads*2*head_dim={}",
        qg.dim(1),
        num_heads * 2 * head_dim,
    );
    let cols = num_heads * head_dim;
    let dt = qg.dtype();
    let q_out = alloc.alloc_tensor(&[num_tokens, cols], dt);
    let gate_out = alloc.alloc_tensor(&[num_tokens, cols], dt);

    match dt {
        DType::BF16 => gate_split_bf16(
            q_out.as_gpu_tensor().as_mut_ptr() as *mut u16,
            gate_out.as_gpu_tensor().as_mut_ptr() as *mut u16,
            qg.as_ptr() as *const u16,
            num_tokens as c_int,
            num_heads as c_int,
            head_dim as c_int,
            stream,
        ),
        DType::F16 => gate_split_f16(
            q_out.as_gpu_tensor().as_mut_ptr() as *mut u16,
            gate_out.as_gpu_tensor().as_mut_ptr() as *mut u16,
            qg.as_ptr() as *const u16,
            num_tokens as c_int,
            num_heads as c_int,
            head_dim as c_int,
            stream,
        ),
        _ => panic!("gate_split: unsupported dtype {:?} (bf16/f16 only)", dt),
    }

    (q_out, gate_out)
}

// ---------------------------------------------------------------------------
// Tests for GDN kernels (gating, conv1d, recurrence, rms_norm_gated)
// ---------------------------------------------------------------------------

#[cfg(test)]
#[cfg(feature = "cuda")]
mod tests_gdn {
    use super::*;
    use crate::driver;
    use crate::tensor::GpuTensor;

    unsafe fn test_init() -> CUstream {
        driver::init().expect("CUDA init");
        let dev = driver::device_get(0).expect("device 0");
        let _ctx = driver::ctx_create(dev).expect("ctx_create");
        driver::stream_create().expect("stream_create")
    }

    unsafe fn upload_f32(data: &[f32], stream: CUstream) -> GpuTensor {
        let bytes = data.len() * 4;
        let ptr = driver::mem_alloc(bytes).expect("mem_alloc");
        driver::memcpy_htod_async(ptr, data.as_ptr() as *const u8, bytes, stream)
            .expect("memcpy_htod");
        driver::stream_synchronize(stream).expect("sync");
        GpuTensor::new(ptr, &[data.len()], crate::dtype::DType::F32)
    }

    unsafe fn upload_i32(data: &[i32], stream: CUstream) -> GpuTensor {
        let bytes = data.len() * 4;
        let ptr = driver::mem_alloc(bytes).expect("mem_alloc");
        driver::memcpy_htod_async(ptr, data.as_ptr() as *const u8, bytes, stream)
            .expect("memcpy_htod");
        driver::stream_synchronize(stream).expect("sync");
        GpuTensor::new(ptr, &[data.len()], crate::dtype::DType::I32)
    }

    unsafe fn alloc_f32(n: usize, stream: CUstream) -> GpuTensor {
        let bytes = n * 4;
        let ptr = driver::mem_alloc(bytes).expect("mem_alloc");
        driver::memset_d8(ptr, 0, bytes, stream).expect("memset");
        GpuTensor::new(ptr, &[n], crate::dtype::DType::F32)
    }

    unsafe fn download_f32(t: GpuTensor, stream: CUstream) -> Vec<f32> {
        let n = t.numel();
        let bytes = n * 4;
        let mut host = vec![0.0f32; n];
        driver::memcpy_dtoh_async(
            host.as_mut_ptr() as *mut u8,
            t.as_ptr::<u8>(),
            bytes,
            stream,
        )
        .expect("dtoh");
        driver::stream_synchronize(stream).expect("sync");
        host
    }

    /// Test fused GDN gating: g = -exp(A_log) * softplus(a + dt_bias), beta = sigmoid(b).
    #[test]
    #[ignore]
    fn test_cuda_gdn_gating() {
        unsafe {
            let stream = test_init();
            let num_heads = 4;
            let batch = 2;
            let n = batch * num_heads;

            // A_log = [0.0; 4] => exp(0) = 1
            let a_log = upload_f32(&vec![0.0; num_heads], stream);
            // dt_bias = [0.0; 4]
            let dt_bias = upload_f32(&vec![0.0; num_heads], stream);
            // a = [1.0; n] => softplus(1.0 + 0.0) = ln(1 + e) ≈ 1.3133
            let a = upload_f32(&vec![1.0; n], stream);
            // b = [0.0; n] => sigmoid(0) = 0.5
            let b = upload_f32(&vec![0.0; n], stream);

            let g_out = alloc_f32(n, stream);
            let beta_out = alloc_f32(n, stream);

            gdn_gating(
                g_out, beta_out, a_log, a, b, dt_bias, num_heads, batch, stream,
            );

            let g = download_f32(g_out, stream);
            let beta = download_f32(beta_out, stream);

            // g = -1.0 * softplus(1.0) ≈ -1.3133
            for &v in &g {
                assert!((v - (-1.3133)).abs() < 0.01, "g={v}, expected ~-1.3133");
            }
            // beta = sigmoid(0) = 0.5
            for &v in &beta {
                assert!((v - 0.5).abs() < 0.01, "beta={v}, expected 0.5");
            }
        }
    }

    /// Test causal conv1d single-token update.
    #[test]
    #[ignore]
    fn test_cuda_gdn_conv1d_update() {
        unsafe {
            let stream = test_init();
            let conv_dim = 4;
            let kernel_size = 4;
            let state_len = kernel_size - 1;
            let batch = 1;
            let num_slots = 1;

            // conv_state: [num_slots, conv_dim, state_len] = [1, 4, 3] — all zeros
            let conv_state = alloc_f32(num_slots * conv_dim * state_len, stream);
            // x: [batch, conv_dim] = [1, 4] — all ones
            let x = upload_f32(&vec![1.0; batch * conv_dim], stream);
            // w: [conv_dim, kernel_size] = [4, 4] — all ones
            let w = upload_f32(&vec![1.0; conv_dim * kernel_size], stream);
            let output = alloc_f32(batch * conv_dim, stream);
            let state_indices = upload_i32(&[0], stream);

            gdn_conv1d_update(
                conv_state,
                x,
                w,
                output,
                state_indices,
                conv_dim,
                kernel_size,
                batch,
                stream,
            );

            let out = download_f32(output, stream);
            // With zero state and input=1, conv = 0*1 + 0*1 + 0*1 + 1*1 = 1.
            // SiLU(1) = 1/(1+exp(-1)) ≈ 0.7311
            for &v in &out {
                assert!(
                    (v - 0.7311).abs() < 0.01,
                    "conv1d output={v}, expected ~0.7311"
                );
            }

            // Verify state was updated: last position should be 1.0.
            let state = download_f32(conv_state, stream);
            // State layout: [conv_dim, state_len]. Last element of each dim should be 1.0.
            for d in 0..conv_dim {
                let last = state[d * state_len + state_len - 1];
                assert!(
                    (last - 1.0).abs() < 1e-6,
                    "state[{d}][last]={last}, expected 1.0"
                );
            }
        }
    }

    /// Test RMS norm gated: out = rms_norm(x) * weight * silu(z),
    /// silu(z) = z * sigmoid(z).
    #[test]
    #[ignore]
    fn test_cuda_gdn_rms_norm_gated() {
        unsafe {
            let stream = test_init();
            let dim = 4;
            let rows = 2;

            // x = [1, 1, 1, 1, 2, 2, 2, 2]
            let x_data: Vec<f32> = (0..rows).flat_map(|r| vec![(r + 1) as f32; dim]).collect();
            let x = upload_f32(&x_data, stream);
            // z = [1, 1, ...] => silu(1) = 1 * sigmoid(1) = 0.7310586
            let z = upload_f32(&vec![1.0; rows * dim], stream);
            // weight = [1, 1, 1, 1]
            let weight = upload_f32(&vec![1.0; dim], stream);
            let out = alloc_f32(rows * dim, stream);

            gdn_rms_norm_gated(x, z, weight, out, 1e-6, dim, rows, stream);

            let result = download_f32(out, stream);
            let silu1 = 1.0f32 / (1.0 + (-1.0f32).exp()); // silu(1) = 1*sigmoid(1)
            // Row 0: x=[1,1,1,1], rms = 1, normed = 1, * weight 1, * silu(1).
            for (i, &val) in result[..dim].iter().enumerate() {
                assert!(
                    (val - silu1).abs() < 0.01,
                    "row0[{i}]={val}, expected {silu1}",
                );
            }
            // Row 1: x=[2,2,2,2], rms = 1/2, normed = 2*0.5 = 1, * weight 1, * silu(1).
            for (i, &val) in result[dim..2 * dim].iter().enumerate() {
                assert!(
                    (val - silu1).abs() < 0.01,
                    "row1[{}]={val}, expected {silu1}",
                    i + dim
                );
            }
        }
    }

    /// Test fused recurrent GDN forward: simple single-token, single-sequence case.
    #[test]
    #[ignore]
    fn test_cuda_gdn_recurrent_fwd_basic() {
        unsafe {
            let stream = test_init();
            let num_seqs = 1;
            let total_tokens = 1;
            let hk = 4; // head_k_dim
            let hv_dim = 2; // head_v_dim
            let n_k_heads = 1;
            let n_v_heads = 1;
            let num_slots = 1;

            // q, k: [T=1, H=1, K=4] — unit vectors
            let q_data = vec![1.0f32, 0.0, 0.0, 0.0];
            let k_data = vec![0.0f32, 1.0, 0.0, 0.0];
            let q = upload_f32(&q_data, stream);
            let k = upload_f32(&k_data, stream);

            // v: [T=1, HV=1, V=2]
            let v_data = vec![1.0f32, 2.0];
            let v = upload_f32(&v_data, stream);

            // g: [T=1, HV=1] = 0.0 => decay = exp(0) = 1
            let g = upload_f32(&[0.0f32], stream);
            // beta: [T=1, HV=1] = 1.0
            let beta = upload_f32(&[1.0f32], stream);

            // ssm_state: [num_slots, HV, V, K] = [1, 1, 2, 4] — zeros
            let ssm_state = alloc_f32(num_slots * n_v_heads * hv_dim * hk, stream);
            let o = alloc_f32(total_tokens * n_v_heads * hv_dim, stream);

            let state_indices = upload_i32(&[0], stream);
            let cu_seqlens = upload_i32(&[0, 1], stream);

            let scale = 1.0f32;

            gdn_recurrent_fwd(
                q,
                k,
                v,
                g,
                beta,
                o,
                ssm_state,
                state_indices,
                cu_seqlens,
                scale,
                num_seqs,
                total_tokens,
                n_k_heads,
                n_v_heads,
                hk,
                hv_dim,
                stream,
            );

            let output = download_f32(o, stream);
            // With zero state:
            // q_norm = [1,0,0,0] (already unit), k_norm = [0,1,0,0]
            // decay = 1, S stays zero
            // delta = v - dot(S=0, k) = v = [1, 2]
            // beta = 1, so v_delta = [1, 2]
            // S += v_delta * k = [[0,1,0,0], [0,2,0,0]]
            // o = dot(S, q*scale) where q_scaled = [1,0,0,0]
            // o[0] = S[0][0]*1 = 0, o[1] = S[1][0]*1 = 0
            // So output should be [0, 0] because S has weight only on k-dim 1, but q on k-dim 0.
            assert!(output[0].abs() < 0.01, "o[0]={}, expected ~0", output[0]);
            assert!(output[1].abs() < 0.01, "o[1]={}, expected ~0", output[1]);

            // Verify state was updated.
            let state = download_f32(ssm_state, stream);
            // S[v=0, k=1] = 1.0, S[v=1, k=1] = 2.0
            assert!(
                (state[1] - 1.0).abs() < 0.01,
                "state[0,1]={}, expected 1.0",
                state[1]
            );
            assert!(
                (state[hk + 1] - 2.0).abs() < 0.01,
                "state[1,1]={}, expected 2.0",
                state[hk + 1]
            );
        }
    }
}

// ---------------------------------------------------------------------------
// FP8 KV cache kernel tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[cfg(feature = "cuda")]
mod tests_fp8_kv {
    use super::*;
    use crate::driver;

    type CUstream = cudarc::driver::sys::CUstream;

    unsafe fn test_init() -> (CachingAllocator, CUstream) {
        driver::init().expect("CUDA init");
        let dev = driver::device_get(0).expect("device 0");
        let _ctx = driver::ctx_create(dev).expect("ctx_create");
        let stream = driver::stream_create().expect("stream_create");
        let alloc = CachingAllocator::new();
        (alloc, stream)
    }

    unsafe fn upload_bf16(data: &[u16], stream: CUstream) -> *mut u8 {
        let bytes = data.len() * 2;
        let ptr = driver::mem_alloc(bytes).expect("mem_alloc");
        driver::memcpy_htod_async(ptr, data.as_ptr() as *const u8, bytes, stream).expect("H2D");
        ptr
    }

    unsafe fn upload_i64(data: &[i64], stream: CUstream) -> *mut u8 {
        let bytes = data.len() * 8;
        let ptr = driver::mem_alloc(bytes).expect("mem_alloc");
        driver::memcpy_htod_async(ptr, data.as_ptr() as *const u8, bytes, stream).expect("H2D");
        ptr
    }

    unsafe fn upload_f32(data: &[f32], stream: CUstream) -> *mut u8 {
        let bytes = data.len() * 4;
        let ptr = driver::mem_alloc(bytes).expect("mem_alloc");
        driver::memcpy_htod_async(ptr, data.as_ptr() as *const u8, bytes, stream).expect("H2D");
        ptr
    }

    unsafe fn upload_i32(data: &[i32], stream: CUstream) -> *mut u8 {
        let bytes = data.len() * 4;
        let ptr = driver::mem_alloc(bytes).expect("mem_alloc");
        driver::memcpy_htod_async(ptr, data.as_ptr() as *const u8, bytes, stream).expect("H2D");
        ptr
    }

    fn f32_to_bf16(val: f32) -> u16 {
        half::bf16::from_f32(val).to_bits()
    }

    fn bf16_to_f32(bits: u16) -> f32 {
        half::bf16::from_bits(bits).to_f32()
    }

    /// Write BF16 → FP8 cache → dequant_gather → compare to original.
    /// Max error ≤ FP8 quantization step (~0.03 for values near 1.0).
    #[test]
    #[ignore] // requires CUDA GPU
    fn test_cuda_fp8_reshape_round_trip() {
        unsafe {
            let (mut alloc, stream) = test_init();

            let num_tokens = 2;
            let num_heads = 2;
            let head_dim = 8;
            let block_size = 16;
            let num_blocks = 1;

            // BF16 input: values in range [-2, 2]
            let input_f32: Vec<f32> = (0..num_tokens * num_heads * head_dim)
                .map(|i| (i as f32 - 16.0) * 0.125) // range [-2, 1.875]
                .collect();
            let input_bf16: Vec<u16> = input_f32.iter().map(|&v| f32_to_bf16(v)).collect();

            let key_ptr = upload_bf16(&input_bf16, stream);
            let value_ptr = upload_bf16(&input_bf16, stream);
            let key = GpuTensor::new(key_ptr, &[num_tokens, num_heads, head_dim], DType::BF16);
            let value = GpuTensor::new(value_ptr, &[num_tokens, num_heads, head_dim], DType::BF16);

            // FP8 cache
            let cache_elems = num_blocks * block_size * num_heads * head_dim;
            let k_cache_ptr = driver::mem_alloc(cache_elems).expect("alloc k_cache");
            let v_cache_ptr = driver::mem_alloc(cache_elems).expect("alloc v_cache");
            let k_cache = GpuTensor::new(
                k_cache_ptr,
                &[num_blocks, block_size, num_heads, head_dim],
                DType::Fp8E4m3,
            );
            let v_cache = GpuTensor::new(
                v_cache_ptr,
                &[num_blocks, block_size, num_heads, head_dim],
                DType::Fp8E4m3,
            );

            // Scale = 1.0 (on GPU)
            let scale_ptr = upload_f32(&[1.0_f32], stream);

            // Slot mapping: tokens go to slots 0, 1
            let slots = [0i64, 1];
            let slot_ptr = upload_i64(&slots, stream);
            let slot_mapping = GpuTensor::new(slot_ptr, &[num_tokens], DType::I64);

            // Write BF16 → FP8 cache
            reshape_and_cache_fp8(
                key,
                value,
                k_cache,
                v_cache,
                slot_mapping,
                scale_ptr as *const f32,
                scale_ptr as *const f32,
                block_size,
                stream,
            );

            // Dequant+gather K from FP8 cache
            let block_table_data = [0i32]; // single block
            let block_table_ptr = upload_i32(&block_table_data, stream);
            let block_table = GpuTensor::new(block_table_ptr, &[1, 1], DType::I32);

            let cu_seqlens = [0i32, num_tokens as i32];
            let cu_seqlens_ptr = upload_i32(&cu_seqlens, stream);
            let cu_seqlens_t = GpuTensor::new(cu_seqlens_ptr, &[2], DType::I32);

            let k_out = dequant_gather_pages(
                k_cache,
                block_table,
                cu_seqlens_t,
                1.0,
                num_tokens,
                num_heads,
                head_dim,
                block_size,
                DType::BF16,
                &mut alloc,
                stream,
            );

            // D2H and compare
            let out_bytes = num_tokens * num_heads * head_dim * 2;
            let mut out_bf16 = vec![0u16; num_tokens * num_heads * head_dim];
            driver::memcpy_dtoh_async(
                out_bf16.as_mut_ptr() as *mut u8,
                k_out.as_gpu_tensor().raw_ptr() as *const u8,
                out_bytes,
                stream,
            )
            .expect("D2H");
            driver::stream_synchronize(stream).expect("sync");

            for i in 0..input_f32.len() {
                let original = input_f32[i];
                let recovered = bf16_to_f32(out_bf16[i]);
                let err = (original - recovered).abs();
                // FP8 E4M3 has ~0.03 precision for values near 1.0, worse for larger values
                let tolerance = original.abs() * 0.15 + 0.05; // relative + absolute tolerance
                assert!(
                    err <= tolerance,
                    "element {i}: original={original}, recovered={recovered}, err={err}, tol={tolerance}"
                );
            }
        }
    }

    /// Verify that slot=-1 (padding) doesn't corrupt the FP8 cache.
    #[test]
    #[ignore] // requires CUDA GPU
    fn test_cuda_fp8_reshape_slot_neg1_skipped() {
        unsafe {
            let (_alloc, stream) = test_init();

            let num_tokens = 2;
            let num_heads = 1;
            let head_dim = 8;
            let block_size = 16;
            let num_blocks = 1;
            let n_elems = num_heads * head_dim;

            // Fill cache with 0xFF sentinel
            let cache_bytes = num_blocks * block_size * n_elems;
            let k_cache_ptr = driver::mem_alloc(cache_bytes).expect("alloc");
            let sentinel = vec![0xFFu8; cache_bytes];
            driver::memcpy_htod_async(k_cache_ptr, sentinel.as_ptr(), cache_bytes, stream)
                .expect("H2D");
            let v_cache_ptr = driver::mem_alloc(cache_bytes).expect("alloc");
            driver::memcpy_htod_async(v_cache_ptr, sentinel.as_ptr(), cache_bytes, stream)
                .expect("H2D");

            let k_cache = GpuTensor::new(
                k_cache_ptr,
                &[num_blocks, block_size, num_heads, head_dim],
                DType::Fp8E4m3,
            );
            let v_cache = GpuTensor::new(
                v_cache_ptr,
                &[num_blocks, block_size, num_heads, head_dim],
                DType::Fp8E4m3,
            );

            // Input data
            let input_bf16: Vec<u16> = (0..num_tokens * n_elems)
                .map(|_| f32_to_bf16(1.0))
                .collect();
            let key_ptr = upload_bf16(&input_bf16, stream);
            let value_ptr = upload_bf16(&input_bf16, stream);
            let key = GpuTensor::new(key_ptr, &[num_tokens, num_heads, head_dim], DType::BF16);
            let value = GpuTensor::new(value_ptr, &[num_tokens, num_heads, head_dim], DType::BF16);

            // Slot mapping: first token → slot 0, second → slot -1 (padding)
            let slots = [0i64, -1i64];
            let slot_ptr = upload_i64(&slots, stream);
            let slot_mapping = GpuTensor::new(slot_ptr, &[num_tokens], DType::I64);

            let scale_ptr = upload_f32(&[1.0_f32], stream);

            reshape_and_cache_fp8(
                key,
                value,
                k_cache,
                v_cache,
                slot_mapping,
                scale_ptr as *const f32,
                scale_ptr as *const f32,
                block_size,
                stream,
            );

            // Verify: slot 0 was written (not sentinel), slot 1 is still sentinel
            let mut cache_host = vec![0u8; cache_bytes];
            driver::memcpy_dtoh_async(
                cache_host.as_mut_ptr(),
                k_cache_ptr as *const u8,
                cache_bytes,
                stream,
            )
            .expect("D2H");
            driver::stream_synchronize(stream).expect("sync");

            // Slot 0 (first n_elems bytes) should NOT be all 0xFF
            let slot0_all_ff = cache_host[..n_elems].iter().all(|&b| b == 0xFF);
            assert!(!slot0_all_ff, "slot 0 should have been written");

            // Slot 1 (next n_elems bytes) should still be all 0xFF (padding skipped)
            let slot1_all_ff = cache_host[n_elems..2 * n_elems].iter().all(|&b| b == 0xFF);
            assert!(
                slot1_all_ff,
                "slot 1 (padding) should not have been written"
            );
        }
    }

    /// Verify KvCachePool FP8 allocates half the memory of BF16.
    #[test]
    #[ignore] // requires CUDA GPU
    fn test_cuda_kv_pool_fp8_half_memory() {
        unsafe {
            let (_alloc, _stream) = test_init();

            let num_layers = 2;
            let num_blocks = 64;
            let block_size = 16;
            let num_kv_heads = 4;
            let head_dim = 64;

            let fp8_pool = crate::kv_cache::KvCachePool::new(
                num_layers,
                num_blocks,
                block_size,
                num_kv_heads,
                head_dim,
                num_blocks, // max_blocks_per_seq
                None,
                DType::Fp8E4m3,
                |bytes| unsafe {
                    let ptr = crate::driver::mem_alloc(bytes)?;
                    Ok(crate::raw_cuda(ptr, bytes))
                },
                |scale_mem| {
                    let one: f32 = 1.0;
                    crate::driver::memcpy_htod_async(
                        scale_mem.ptr(),
                        &one as *const f32 as *const u8,
                        4,
                        std::ptr::null_mut(),
                    )?;
                    Ok(())
                },
            )
            .expect("FP8 pool");

            assert!(fp8_pool.is_fp8());
            assert_eq!(fp8_pool.cache_dtype(), DType::Fp8E4m3);

            // Verify we can access scale pointers
            let k_scale = fp8_pool.k_scale_ptr(0);
            assert!(!k_scale.is_null());

            // Read back default scale — should be 1.0
            let mut scale_val: f32 = 0.0;
            driver::memcpy_dtoh_async(
                &mut scale_val as *mut f32 as *mut u8,
                k_scale as *const u8,
                4,
                _stream,
            )
            .expect("D2H scale");
            driver::stream_synchronize(_stream).expect("sync");
            assert_eq!(scale_val, 1.0, "default K scale should be 1.0");
        }
    }

    /// Test compute_abs_max_and_scale: tensor with known max.
    #[test]
    #[ignore] // requires CUDA GPU
    fn test_cuda_compute_kv_scale() {
        unsafe {
            let (_alloc, stream) = test_init();

            // BF16 tensor with max abs value = 1000.0
            let data: Vec<u16> = vec![
                f32_to_bf16(100.0),
                f32_to_bf16(-500.0),
                f32_to_bf16(1000.0),
                f32_to_bf16(0.1),
                f32_to_bf16(-200.0),
                f32_to_bf16(50.0),
                f32_to_bf16(0.0),
                f32_to_bf16(-1000.0),
            ];
            let tensor_ptr = upload_bf16(&data, stream);
            let tensor = GpuTensor::new(tensor_ptr, &[8], DType::BF16);

            let scale_ptr = driver::mem_alloc(4).expect("alloc scale") as *mut f32;

            // divisor = 200.0, so scale = 1000.0 / 200.0 = 5.0
            compute_kv_scale(tensor, 8, 200.0, scale_ptr, stream);

            let mut scale_val: f32 = 0.0;
            driver::memcpy_dtoh_async(
                &mut scale_val as *mut f32 as *mut u8,
                scale_ptr as *const u8,
                4,
                stream,
            )
            .expect("D2H");
            driver::stream_synchronize(stream).expect("sync");

            assert!(
                (scale_val - 5.0).abs() < 0.1,
                "expected scale ≈ 5.0, got {scale_val}"
            );
        }
    }

    /// Dequant+gather with multiple sequences of different lengths.
    #[test]
    #[ignore] // requires CUDA GPU
    fn test_cuda_dequant_gather_multi_seq() {
        unsafe {
            let (mut alloc, stream) = test_init();

            let num_heads = 1;
            let head_dim = 8;
            let block_size = 4;
            let num_blocks = 4;
            let n_elems = num_heads * head_dim;

            // Allocate FP8 cache and fill with known values.
            let cache_bytes = num_blocks * block_size * n_elems;
            let cache_ptr = driver::mem_alloc(cache_bytes).expect("alloc");

            // Fill each slot with its slot index cast to FP8 (via BF16→FP8 on host)
            // For simplicity, fill the entire cache with a known byte pattern.
            // FP8 E4M3: 0x38 = 1.0, 0x3C = 1.5, 0x40 = 2.0, 0x00 = 0.0
            let cache_data = vec![0x38u8; cache_bytes]; // 1.0 in FP8 E4M3
            driver::memcpy_htod_async(cache_ptr, cache_data.as_ptr(), cache_bytes, stream)
                .expect("H2D cache");
            let cache = GpuTensor::new(
                cache_ptr,
                &[num_blocks, block_size, num_heads, head_dim],
                DType::Fp8E4m3,
            );

            // Two sequences: seq0 has 3 tokens, seq1 has 2 tokens.
            // Block table: seq0 uses block 0, seq1 uses block 1.
            let block_table_data = [0i32, 0, 1, 0]; // [2, 2] padded
            let block_table_ptr = upload_i32(&block_table_data, stream);
            let block_table = GpuTensor::new(block_table_ptr, &[2, 2], DType::I32);

            let cu_seqlens = [0i32, 3, 5]; // seq0: 3 tokens, seq1: 2 tokens
            let cu_seqlens_ptr = upload_i32(&cu_seqlens, stream);
            let cu_seqlens_t = GpuTensor::new(cu_seqlens_ptr, &[3], DType::I32);

            let total_kv = 5;
            let out = dequant_gather_pages(
                cache,
                block_table,
                cu_seqlens_t,
                1.0, // scale
                total_kv,
                num_heads,
                head_dim,
                block_size,
                DType::BF16,
                &mut alloc,
                stream,
            );

            // All values should dequant to ~1.0
            let out_count = total_kv * n_elems;
            let mut out_bf16 = vec![0u16; out_count];
            driver::memcpy_dtoh_async(
                out_bf16.as_mut_ptr() as *mut u8,
                out.as_gpu_tensor().raw_ptr() as *const u8,
                out_count * 2,
                stream,
            )
            .expect("D2H");
            driver::stream_synchronize(stream).expect("sync");

            for (i, &raw) in out_bf16[..out_count].iter().enumerate() {
                let val = bf16_to_f32(raw);
                assert!(
                    (val - 1.0).abs() < 0.1,
                    "element {i}: expected ~1.0, got {val}"
                );
            }
        }
    }

    /// Set K/V scale on pool and verify via D2H.
    #[test]
    #[ignore] // requires CUDA GPU
    #[allow(clippy::approx_constant)]
    fn test_cuda_kv_pool_fp8_set_scale() {
        use crate::kv_cache::CudaKvCacheExt;
        unsafe {
            let (_alloc, stream) = test_init();

            let pool = crate::kv_cache::KvCachePool::new(
                1,
                16,
                16,
                2,
                64,
                16, // max_blocks_per_seq
                None,
                DType::Fp8E4m3,
                |bytes| unsafe {
                    let ptr = crate::driver::mem_alloc(bytes)?;
                    Ok(crate::raw_cuda(ptr, bytes))
                },
                |scale_mem| {
                    let one: f32 = 1.0;
                    crate::driver::memcpy_htod_async(
                        scale_mem.ptr(),
                        &one as *const f32 as *const u8,
                        4,
                        std::ptr::null_mut(),
                    )?;
                    Ok(())
                },
            )
            .expect("FP8 pool");

            pool.set_k_scale(0, 3.14, stream);
            pool.set_v_scale(0, 2.71, stream);

            let mut k_scale: f32 = 0.0;
            let mut v_scale: f32 = 0.0;
            driver::memcpy_dtoh_async(
                &mut k_scale as *mut f32 as *mut u8,
                pool.k_scale_ptr(0) as *const u8,
                4,
                stream,
            )
            .expect("D2H k_scale");
            driver::memcpy_dtoh_async(
                &mut v_scale as *mut f32 as *mut u8,
                pool.v_scale_ptr(0) as *const u8,
                4,
                stream,
            )
            .expect("D2H v_scale");
            driver::stream_synchronize(stream).expect("sync");

            assert!((k_scale - 3.14).abs() < 0.001, "k_scale={k_scale}");
            assert!((v_scale - 2.71).abs() < 0.001, "v_scale={v_scale}");
        }
    }

    /// `dequant_gather_pages_into` writes to a pre-allocated output buffer.
    #[test]
    #[ignore] // requires CUDA GPU
    fn test_cuda_dequant_gather_pages_into() {
        unsafe {
            let (_alloc, stream) = test_init();

            let num_heads = 1;
            let head_dim = 8;
            let block_size = 4;
            let num_blocks = 2;
            let n_elems = num_heads * head_dim;

            // FP8 cache filled with 0x38 (1.0 in E4M3).
            let cache_bytes = num_blocks * block_size * n_elems;
            let cache_ptr = driver::mem_alloc(cache_bytes).expect("alloc");
            let cache_data = vec![0x38u8; cache_bytes];
            driver::memcpy_htod_async(cache_ptr, cache_data.as_ptr(), cache_bytes, stream)
                .expect("H2D");
            let cache = GpuTensor::new(
                cache_ptr,
                &[num_blocks, block_size, num_heads, head_dim],
                DType::Fp8E4m3,
            );

            // 1 sequence, 3 tokens → block 0.
            let bt = [0i32, 0];
            let bt_ptr = upload_i32(&bt, stream);
            let block_table = GpuTensor::new(bt_ptr, &[1, 2], DType::I32);

            let cu = [0i32, 3];
            let cu_ptr = upload_i32(&cu, stream);
            let cu_t = GpuTensor::new(cu_ptr, &[2], DType::I32);

            // Pre-allocate output buffer larger than needed (simulating graph capture).
            let max_total = 8; // capacity > actual 3
            let out_ptr = driver::mem_alloc(max_total * n_elems * 2).expect("alloc out");
            // Zero it to detect writes.
            driver::memset_d8(out_ptr, 0, max_total * n_elems * 2, stream).expect("memset");

            dequant_gather_pages_into(
                cache,
                block_table,
                cu_t,
                1.0,
                max_total, // over-sized grid
                num_heads,
                head_dim,
                block_size,
                DType::BF16,
                out_ptr,
                stream,
            );

            let out_count = max_total * n_elems;
            let mut out_bf16 = vec![0u16; out_count];
            driver::memcpy_dtoh_async(
                out_bf16.as_mut_ptr() as *mut u8,
                out_ptr as *const u8,
                out_count * 2,
                stream,
            )
            .expect("D2H");
            driver::stream_synchronize(stream).expect("sync");

            // First 3 tokens (24 elements) should be ~1.0.
            for (i, &raw) in out_bf16[..(3 * n_elems)].iter().enumerate() {
                let val = bf16_to_f32(raw);
                assert!(
                    (val - 1.0).abs() < 0.1,
                    "element {i}: expected ~1.0, got {val}"
                );
            }
            // Elements beyond actual total should remain zero (bounds check worked).
            for (i, &raw) in out_bf16[(3 * n_elems)..(max_total * n_elems)]
                .iter()
                .enumerate()
            {
                let val = bf16_to_f32(raw);
                assert!(
                    val.abs() < 0.001,
                    "element {} beyond total: expected ~0.0, got {val}",
                    i + 3 * n_elems
                );
            }

            let _ = driver::mem_free(cache_ptr);
            let _ = driver::mem_free(bt_ptr);
            let _ = driver::mem_free(cu_ptr);
            let _ = driver::mem_free(out_ptr);
        }
    }

    /// `prefix_sum_seqused_k_gpu` computes correct cu_seqlens_k on GPU.
    #[test]
    #[ignore] // requires CUDA GPU
    fn test_cuda_prefix_sum_seqused_k() {
        unsafe {
            let (_alloc, stream) = test_init();

            let seqused = [5i32, 3, 8, 1];
            let num_reqs = seqused.len();
            let seqused_ptr = upload_i32(&seqused, stream);

            let cu_ptr = driver::mem_alloc((num_reqs + 1) * 4).expect("alloc cu");
            driver::memset_d8(cu_ptr, 0xFF, (num_reqs + 1) * 4, stream).expect("memset");

            compute_cu_seqlens_k_gpu(seqused_ptr as *const u8, cu_ptr, num_reqs, stream);

            let mut result = vec![0i32; num_reqs + 1];
            driver::memcpy_dtoh_async(
                result.as_mut_ptr() as *mut u8,
                cu_ptr as *const u8,
                (num_reqs + 1) * 4,
                stream,
            )
            .expect("D2H");
            driver::stream_synchronize(stream).expect("sync");

            assert_eq!(result, [0, 5, 8, 16, 17], "prefix sum mismatch: {result:?}");

            let _ = driver::mem_free(seqused_ptr);
            let _ = driver::mem_free(cu_ptr);
        }
    }

    /// `prefix_sum_seqused_k_gpu` with batch_size=1.
    #[test]
    #[ignore] // requires CUDA GPU
    fn test_cuda_prefix_sum_seqused_k_single() {
        unsafe {
            let (_alloc, stream) = test_init();

            let seqused = [42i32];
            let seqused_ptr = upload_i32(&seqused, stream);
            let cu_ptr = driver::mem_alloc(2 * 4).expect("alloc");

            compute_cu_seqlens_k_gpu(seqused_ptr as *const u8, cu_ptr, 1, stream);

            let mut result = vec![0i32; 2];
            driver::memcpy_dtoh_async(
                result.as_mut_ptr() as *mut u8,
                cu_ptr as *const u8,
                8,
                stream,
            )
            .expect("D2H");
            driver::stream_synchronize(stream).expect("sync");

            assert_eq!(result, [0, 42]);

            let _ = driver::mem_free(seqused_ptr);
            let _ = driver::mem_free(cu_ptr);
        }
    }
}

// ---------------------------------------------------------------------------
// FP8 quantization kernel tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[cfg(feature = "cuda")]
mod tests_fp8_quant {
    use super::*;
    use crate::driver;
    type CUstream = cudarc::driver::sys::CUstream;

    unsafe fn test_init() -> (CachingAllocator, CUstream) {
        driver::init().expect("CUDA init");
        let dev = driver::device_get(0).expect("device 0");
        let _ctx = driver::ctx_create(dev).expect("ctx_create");
        let stream = driver::stream_create().expect("stream_create");
        let alloc = CachingAllocator::new();
        (alloc, stream)
    }

    unsafe fn upload_bf16(data: &[half::bf16], stream: CUstream) -> GpuTensor {
        let bytes = data.len() * 2;
        let ptr = driver::mem_alloc(bytes).expect("mem_alloc");
        driver::memcpy_htod_async(ptr, data.as_ptr() as *const u8, bytes, stream).expect("H2D");
        GpuTensor::new(ptr, &[data.len()], DType::BF16)
    }

    unsafe fn download_u8(tensor: GpuTensor, stream: CUstream) -> Vec<u8> {
        let count = tensor.numel();
        let host = driver::mem_alloc_host(count).expect("host alloc");
        driver::memcpy_dtoh_async(host, tensor.raw_ptr(), count, stream).expect("D2H");
        driver::stream_synchronize(stream).expect("sync");
        let result = std::slice::from_raw_parts(host, count).to_vec();
        driver::mem_free_host(host).expect("free");
        result
    }

    unsafe fn download_f32(ptr: *mut u8, count: usize, stream: CUstream) -> Vec<f32> {
        let bytes = count * 4;
        let host = driver::mem_alloc_host(bytes).expect("host alloc");
        driver::memcpy_dtoh_async(host, ptr, bytes, stream).expect("D2H");
        driver::stream_synchronize(stream).expect("sync");
        let result = std::slice::from_raw_parts(host as *const f32, count).to_vec();
        driver::mem_free_host(host).expect("free");
        result
    }

    #[test]
    fn test_scaled_fp8_quant_dynamic_bf16() {
        // Dynamic per-token FP8 quantization: BF16 → FP8 E4M3
        // Input: 2 tokens, 4 hidden dims
        // Token 0: [1.0, 2.0, -3.0, 4.0] → absmax=4.0, scale=4.0/448.0
        // Token 1: [0.5, -1.0, 1.5, 0.0] → absmax=1.5, scale=1.5/448.0
        unsafe {
            let (mut alloc, stream) = test_init();
            let num_tokens = 2;
            let hidden_dim = 4;

            let data: Vec<half::bf16> = [1.0f32, 2.0, -3.0, 4.0, 0.5, -1.0, 1.5, 0.0]
                .iter()
                .map(|&v| half::bf16::from_f32(v))
                .collect();

            let input = upload_bf16(&data, stream);
            let input_2d = input.reshape(&[num_tokens, hidden_dim]);

            let (output, scales) = scaled_fp8_quant_dynamic(input_2d, &mut alloc, stream);

            driver::stream_synchronize(stream).expect("sync");

            // Check output shape
            assert_eq!(output.as_gpu_tensor().dim(0), num_tokens);
            assert_eq!(output.as_gpu_tensor().dim(1), hidden_dim);
            assert_eq!(output.as_gpu_tensor().dtype(), DType::Fp8E4m3);

            // Check scales shape
            assert_eq!(scales.as_gpu_tensor().dim(0), num_tokens);

            // Download scales and verify
            let host_scales = download_f32(scales.as_gpu_tensor().raw_ptr(), num_tokens, stream);
            // scale = absmax / 448.0
            let expected_scale_0 = 4.0 / 448.0;
            let expected_scale_1 = 1.5 / 448.0;
            assert!(
                (host_scales[0] - expected_scale_0).abs() < 0.001,
                "scale[0]={}, expected ~{expected_scale_0}",
                host_scales[0]
            );
            assert!(
                (host_scales[1] - expected_scale_1).abs() < 0.001,
                "scale[1]={}, expected ~{expected_scale_1}",
                host_scales[1]
            );

            // Download FP8 output and verify roundtrip
            let fp8_bytes = download_u8(output.as_gpu_tensor(), stream);
            assert_eq!(fp8_bytes.len(), num_tokens * hidden_dim);
            // All bytes should be non-zero (except the 0.0 in token 1)
            // Token 1, index 3 should be 0x00 (FP8 zero)
            assert_eq!(fp8_bytes[7], 0, "0.0 should quantize to FP8 zero");

            driver::stream_destroy(stream).expect("destroy");
        }
    }

    #[test]
    fn test_scaled_fp8_quant_static_bf16() {
        // Static FP8 quantization with a given scale.
        unsafe {
            let (mut alloc, stream) = test_init();
            let num_tokens = 2;
            let hidden_dim = 4;

            let data: Vec<half::bf16> = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
                .iter()
                .map(|&v| half::bf16::from_f32(v))
                .collect();

            let input = upload_bf16(&data, stream);
            let input_2d = input.reshape(&[num_tokens, hidden_dim]);

            // Use scale = 1.0/56.0 (so max representable = 448 * (1/56) = 8.0)
            let scale_val = 1.0f32 / 56.0;
            let scale_ptr = driver::mem_alloc(4).unwrap();
            driver::memcpy_htod_async(scale_ptr, &scale_val as *const f32 as *const u8, 4, stream)
                .unwrap();
            let output =
                scaled_fp8_quant_static(input_2d, scale_ptr as *const f32, &mut alloc, stream);

            driver::stream_synchronize(stream).expect("sync");

            assert_eq!(output.as_gpu_tensor().dim(0), num_tokens);
            assert_eq!(output.as_gpu_tensor().dim(1), hidden_dim);
            assert_eq!(output.as_gpu_tensor().dtype(), DType::Fp8E4m3);

            // All elements should be non-zero
            let fp8_bytes = download_u8(output.as_gpu_tensor(), stream);
            for (i, &b) in fp8_bytes.iter().enumerate() {
                assert_ne!(b, 0, "element {i} should not be zero");
            }

            driver::mem_free(scale_ptr).unwrap();
            driver::stream_destroy(stream).expect("destroy");
        }
    }
}

// ---------------------------------------------------------------------------
// FP8 Fused MoE GEMM tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[cfg(feature = "cuda")]
mod tests_fp8_moe_gemm {
    use super::*;
    use crate::driver;

    unsafe fn test_init() -> (CachingAllocator, cudarc::driver::sys::CUstream, u32) {
        driver::init().expect("CUDA init");
        let dev = driver::device_get(0).expect("device 0");
        let _ctx = driver::ctx_create(dev).expect("ctx_create");
        let stream = driver::stream_create().expect("stream_create");
        let sm = driver::device_get_sm_version(dev).expect("sm version");
        (CachingAllocator::new(), stream, sm)
    }

    unsafe fn upload_slice<T: Copy>(data: &[T], stream: cudarc::driver::sys::CUstream) -> *mut u8 {
        let bytes = std::mem::size_of_val(data);
        let ptr = driver::mem_alloc(bytes).expect("alloc");
        driver::memcpy_htod_async(ptr, data.as_ptr() as *const u8, bytes, stream).expect("h2d");
        driver::stream_synchronize(stream).expect("sync");
        ptr
    }

    unsafe fn download_bf16(
        tensor: GpuTensor,
        stream: cudarc::driver::sys::CUstream,
    ) -> Vec<half::bf16> {
        let count = tensor.numel();
        let bytes = count * 2;
        let mut host = vec![half::bf16::ZERO; count];
        driver::memcpy_dtoh_async(host.as_mut_ptr() as *mut u8, tensor.as_ptr(), bytes, stream)
            .expect("d2h");
        driver::stream_synchronize(stream).expect("sync");
        host
    }

    /// Test FP8 dequant fallback path (SM80) — same setup as basic test.
    #[test]
    #[ignore]
    fn test_cuda_fused_moe_fp8_gemm_dequant() {
        unsafe {
            let (mut alloc, stream, _sm) = test_init();

            let num_tokens: usize = 4;
            let top_k: usize = 1;
            let num_experts: usize = 2;
            let k: usize = 128;
            let n: usize = 64;
            let block_size: usize = 128;

            let fp8_one: u8 = 0x38; // 1.0 in FP8 E4M3
            let fp8_half: u8 = 0x30; // 0.5 in FP8 E4M3

            let input_ptr = upload_slice(&vec![fp8_one; num_tokens * k], stream);
            let input = GpuTensor::new(input_ptr, &[num_tokens, k], DType::Fp8E4m3);

            let weight_ptr = upload_slice(&vec![fp8_half; num_experts * n * k], stream);
            let weights = GpuTensor::new(weight_ptr, &[num_experts, n, k], DType::Fp8E4m3);

            let a_scales_ptr = upload_slice(&vec![1.0f32; num_tokens], stream);
            let a_scales = GpuTensor::new(a_scales_ptr, &[num_tokens], DType::F32);
            let w_scales_ptr = upload_slice(&vec![1.0f32; num_experts], stream);
            let w_scales = GpuTensor::new(w_scales_ptr, &[num_experts], DType::F32);
            let topk_weights_ptr = upload_slice(&vec![1.0f32; num_tokens * top_k], stream);
            let topk_weights = GpuTensor::new(topk_weights_ptr, &[num_tokens, top_k], DType::F32);

            let topk_ids_ptr = upload_slice(&[0i32, 0, 1, 1], stream);
            let topk_ids = GpuTensor::new(topk_ids_ptr, &[num_tokens, top_k], DType::I32);

            let (sorted, experts, ntpp) =
                moe_align_block_size(topk_ids, num_experts, block_size, &mut alloc, stream);

            // Force dequant path by passing sm_version=80
            let output = fused_moe_fp8_gemm(
                input,
                weights,
                a_scales,
                w_scales,
                topk_weights,
                sorted.as_gpu_tensor(),
                experts.as_gpu_tensor(),
                ntpp.as_gpu_tensor(),
                num_tokens,
                top_k,
                block_size,
                false,
                80,
                &mut alloc,
                stream,
            );

            let result = download_bf16(output.as_gpu_tensor(), stream);
            let expected = 64.0f32; // 1.0 * 0.5 * 128 = 64.0
            for (i, &val) in result.iter().enumerate() {
                let v = val.to_f32();
                assert!(
                    (v - expected).abs() < 2.0,
                    "dequant element {i}: got {v}, expected {expected}"
                );
            }

            driver::stream_destroy(stream).expect("destroy");
        }
    }

    /// Test FP8 fused MoE GEMM kernel.
    ///
    /// Setup: 4 tokens, 2 experts, top_k=1, K=128, N=128.
    /// All input values = 1.0 (FP8), all weight values = 0.5 (FP8).
    /// a_scales = 1.0, w_scales = 1.0 (identity).
    /// Expected output = 1.0 * 0.5 * 128 = 64.0 per element.
    #[test]
    #[ignore] // requires CUDA GPU
    fn test_cuda_fused_moe_fp8_gemm_basic() {
        unsafe {
            let (mut alloc, stream, sm) = test_init();

            let num_tokens: usize = 4;
            let top_k: usize = 1;
            let num_experts: usize = 2;
            let k: usize = 128;
            let n: usize = 128;
            let block_size: usize = 128;

            // FP8 E4M3 encoding: 0.5 = 0x38, 0.25 = 0x30
            let fp8_one: u8 = 0x38; // 1.0 in FP8 E4M3
            let fp8_half: u8 = 0x30; // 0.5 in FP8 E4M3

            // Input: [num_tokens, K] FP8
            let input_data = vec![fp8_one; num_tokens * k];
            let input_ptr = upload_slice(&input_data, stream);
            let input = GpuTensor::new(input_ptr, &[num_tokens, k], DType::Fp8E4m3);

            // Weights: [num_experts, N, K] FP8
            let weight_data = vec![fp8_half; num_experts * n * k];
            let weight_ptr = upload_slice(&weight_data, stream);
            let weights = GpuTensor::new(weight_ptr, &[num_experts, n, k], DType::Fp8E4m3);

            // a_scales: [num_tokens] = 1.0
            let a_scales_data = vec![1.0f32; num_tokens];
            let a_scales_ptr = upload_slice(&a_scales_data, stream);
            let a_scales = GpuTensor::new(a_scales_ptr, &[num_tokens], DType::F32);

            // w_scales: [num_experts] = 1.0
            let w_scales_data = vec![1.0f32; num_experts];
            let w_scales_ptr = upload_slice(&w_scales_data, stream);
            let w_scales = GpuTensor::new(w_scales_ptr, &[num_experts], DType::F32);

            // topk_weights: [num_tokens, top_k] = 1.0
            let topk_weights_data = vec![1.0f32; num_tokens * top_k];
            let topk_weights_ptr = upload_slice(&topk_weights_data, stream);
            let topk_weights = GpuTensor::new(topk_weights_ptr, &[num_tokens, top_k], DType::F32);

            // topk_ids: tokens 0,1 → expert 0; tokens 2,3 → expert 1
            let topk_ids_data: Vec<i32> = vec![0, 0, 1, 1];
            let topk_ids_ptr = upload_slice(&topk_ids_data, stream);
            let topk_ids = GpuTensor::new(topk_ids_ptr, &[num_tokens, top_k], DType::I32);

            // moe_align_block_size
            let (sorted_token_ids, expert_ids, num_tokens_post_padded) =
                moe_align_block_size(topk_ids, num_experts, block_size, &mut alloc, stream);

            // Run FP8 GEMM
            let output = fused_moe_fp8_gemm(
                input,
                weights,
                a_scales,
                w_scales,
                topk_weights,
                sorted_token_ids.as_gpu_tensor(),
                expert_ids.as_gpu_tensor(),
                num_tokens_post_padded.as_gpu_tensor(),
                num_tokens,
                top_k,
                block_size,
                false, // don't apply routing weights
                sm,
                &mut alloc,
                stream,
            );

            assert_eq!(
                output.as_gpu_tensor().shape(),
                &[(num_tokens * top_k) as u32, n as u32]
            );
            assert_eq!(output.as_gpu_tensor().dtype(), DType::BF16);

            let result = download_bf16(output.as_gpu_tensor(), stream);

            // Expected: 1.0 * 0.5 * 128 = 64.0
            let expected = 64.0f32;
            for (i, &val) in result.iter().enumerate() {
                let v = val.to_f32();
                assert!(
                    (v - expected).abs() < 2.0,
                    "element {i}: got {v}, expected {expected}"
                );
            }

            driver::stream_destroy(stream).expect("destroy");
        }
    }

    /// Test FP8 MoE GEMM with scale application.
    /// a_scale=2.0, w_scale=3.0 → output should be 6x the identity-scale result.
    #[test]
    #[ignore]
    fn test_cuda_fused_moe_fp8_gemm_scales() {
        unsafe {
            let (mut alloc, stream, sm) = test_init();

            let num_tokens: usize = 2;
            let top_k: usize = 1;
            let num_experts: usize = 1;
            let k: usize = 128;
            let n: usize = 64;
            let block_size: usize = 128;

            let fp8_one: u8 = 0x38; // 1.0 in FP8 E4M3
            let fp8_half: u8 = 0x30; // 0.5 in FP8 E4M3

            let input_ptr = upload_slice(&vec![fp8_one; num_tokens * k], stream);
            let input = GpuTensor::new(input_ptr, &[num_tokens, k], DType::Fp8E4m3);

            let weight_ptr = upload_slice(&vec![fp8_half; num_experts * n * k], stream);
            let weights = GpuTensor::new(weight_ptr, &[num_experts, n, k], DType::Fp8E4m3);

            // a_scale = 2.0 per token, w_scale = 3.0 per expert
            let a_scales_ptr = upload_slice(&vec![2.0f32; num_tokens], stream);
            let a_scales = GpuTensor::new(a_scales_ptr, &[num_tokens], DType::F32);

            let w_scales_ptr = upload_slice(&vec![3.0f32; num_experts], stream);
            let w_scales = GpuTensor::new(w_scales_ptr, &[num_experts], DType::F32);

            let topk_weights_ptr = upload_slice(&vec![1.0f32; num_tokens * top_k], stream);
            let topk_weights = GpuTensor::new(topk_weights_ptr, &[num_tokens, top_k], DType::F32);

            let topk_ids_ptr = upload_slice(&vec![0i32; num_tokens], stream);
            let topk_ids = GpuTensor::new(topk_ids_ptr, &[num_tokens, top_k], DType::I32);

            let (sorted, experts, ntpp) =
                moe_align_block_size(topk_ids, num_experts, block_size, &mut alloc, stream);

            let output = fused_moe_fp8_gemm(
                input,
                weights,
                a_scales,
                w_scales,
                topk_weights,
                sorted.as_gpu_tensor(),
                experts.as_gpu_tensor(),
                ntpp.as_gpu_tensor(),
                num_tokens,
                top_k,
                block_size,
                false,
                sm,
                &mut alloc,
                stream,
            );

            let result = download_bf16(output.as_gpu_tensor(), stream);

            // Base = 1.0 * 0.5 * 128 = 64.0, scaled = 64.0 * 2.0 * 3.0 = 384.0
            let expected = 384.0f32;
            for (i, &val) in result.iter().enumerate() {
                let v = val.to_f32();
                assert!(
                    (v - expected).abs() < 4.0,
                    "element {i}: got {v}, expected {expected}"
                );
            }

            driver::stream_destroy(stream).expect("destroy");
        }
    }

    /// Test full FP8 MoE pipeline: quant → align → gemm1 → silu_and_mul → quant → gemm2 → sum.
    /// Uses random-ish BF16 input, quantizes to FP8, runs through the full pipeline.
    /// Checks that the output has the right shape and non-zero values.
    #[test]
    #[ignore]
    fn test_cuda_fused_moe_fp8_pipeline() {
        unsafe {
            let (mut alloc, stream, sm) = test_init();

            let num_tokens: usize = 4;
            let top_k: usize = 2;
            let num_experts: usize = 4;
            let hidden: usize = 128;
            let inter: usize = 64;
            let block_size: usize = 128;

            // BF16 input: small positive values
            let input_bf16: Vec<u16> = (0..num_tokens * hidden)
                .map(|i| half::bf16::from_f32(0.01 * ((i % 100) as f32 + 1.0)).to_bits())
                .collect();
            let input_ptr = upload_slice(&input_bf16, stream);
            let input = GpuTensor::new(input_ptr, &[num_tokens, hidden], DType::BF16);

            // FP8 expert weights: small values
            let fp8_val: u8 = 0x20; // ~0.0625 in FP8 E4M3
            let w1_data = vec![fp8_val; num_experts * 2 * inter * hidden];
            let w1_ptr = upload_slice(&w1_data, stream);
            let w1 = GpuTensor::new(w1_ptr, &[num_experts, 2 * inter, hidden], DType::Fp8E4m3);

            let w2_data = vec![fp8_val; num_experts * hidden * inter];
            let w2_ptr = upload_slice(&w2_data, stream);
            let w2 = GpuTensor::new(w2_ptr, &[num_experts, hidden, inter], DType::Fp8E4m3);

            // Scales = 1.0
            let w1_scale_ptr = upload_slice(&vec![1.0f32; num_experts], stream);
            let w1_scale = GpuTensor::new(w1_scale_ptr, &[num_experts], DType::F32);
            let w2_scale_ptr = upload_slice(&vec![1.0f32; num_experts], stream);
            let w2_scale = GpuTensor::new(w2_scale_ptr, &[num_experts], DType::F32);

            // Gate output: [num_tokens, num_experts] — BF16, make expert 0 and 1 highest
            let mut gate_data = vec![half::bf16::from_f32(0.1).to_bits(); num_tokens * num_experts];
            for t in 0..num_tokens {
                gate_data[t * num_experts] = half::bf16::from_f32(2.0).to_bits();
                gate_data[t * num_experts + 1] = half::bf16::from_f32(1.5).to_bits();
            }
            let gate_ptr = upload_slice(&gate_data, stream);
            let gate_output = GpuTensor::new(gate_ptr, &[num_tokens, num_experts], DType::BF16);

            // Step 1: topk_softmax
            let (topk_weights, topk_ids) =
                topk_softmax(gate_output, top_k, true, &mut alloc, stream);

            // Step 2: scaled_fp8_quant_dynamic
            let (fp8_input, a1_scales) = scaled_fp8_quant_dynamic(input, &mut alloc, stream);

            // Step 3: moe_align_block_size
            let (sorted, experts, ntpp) = moe_align_block_size(
                topk_ids.as_gpu_tensor(),
                num_experts,
                block_size,
                &mut alloc,
                stream,
            );

            // Step 4: GEMM 1 (no routing weights)
            let intermediate = fused_moe_fp8_gemm(
                fp8_input.as_gpu_tensor(),
                w1,
                a1_scales.as_gpu_tensor(),
                w1_scale,
                topk_weights.as_gpu_tensor(),
                sorted.as_gpu_tensor(),
                experts.as_gpu_tensor(),
                ntpp.as_gpu_tensor(),
                num_tokens,
                top_k,
                block_size,
                false,
                sm,
                &mut alloc,
                stream,
            );
            drop(fp8_input);
            drop(a1_scales);

            // Step 5: silu_and_mul
            let activated =
                silu_and_mul_fused(intermediate.as_gpu_tensor(), inter, &mut alloc, stream);
            drop(intermediate);

            // Step 6: quant activated
            let (fp8_act, a2_scales) =
                scaled_fp8_quant_dynamic(activated.as_gpu_tensor(), &mut alloc, stream);
            drop(activated);

            // Step 7: GEMM 2 (with routing weights, top_k=1 for moe_sum compat)
            // Re-align for the activated tensor shape
            let (sorted2, experts2, ntpp2) = moe_align_block_size(
                topk_ids.as_gpu_tensor(),
                num_experts,
                block_size,
                &mut alloc,
                stream,
            );

            let output2 = fused_moe_fp8_gemm(
                fp8_act.as_gpu_tensor(),
                w2,
                a2_scales.as_gpu_tensor(),
                w2_scale,
                topk_weights.as_gpu_tensor(),
                sorted2.as_gpu_tensor(),
                experts2.as_gpu_tensor(),
                ntpp2.as_gpu_tensor(),
                num_tokens,
                top_k,
                block_size,
                true, // apply routing weights
                sm,
                &mut alloc,
                stream,
            );
            drop(fp8_act);
            drop(a2_scales);

            // Step 8: moe_sum
            let final_out = moe_sum(
                output2.as_gpu_tensor(),
                num_tokens,
                hidden,
                top_k,
                &mut alloc,
                stream,
            );

            assert_eq!(
                final_out.as_gpu_tensor().shape(),
                &[num_tokens as u32, hidden as u32]
            );
            assert_eq!(final_out.as_gpu_tensor().dtype(), DType::BF16);

            let result = download_bf16(final_out.as_gpu_tensor(), stream);
            // Just check non-NaN and finite
            for (i, &val) in result.iter().enumerate() {
                let v = val.to_f32();
                assert!(v.is_finite(), "element {i} is not finite: {v}");
            }

            driver::stream_destroy(stream).expect("destroy");
        }
    }
}

// ---------------------------------------------------------------------------
// Tests: BF16 fused MoE GEMM (variable BLOCK_M + GROUP_SIZE_M)
// ---------------------------------------------------------------------------

#[cfg(test)]
#[cfg(feature = "cuda")]
mod tests_fused_moe_gemm_bf16 {
    use super::*;
    use crate::driver;

    type CUstream = cudarc::driver::sys::CUstream;

    unsafe fn test_init() -> (CachingAllocator, CUstream) {
        driver::init().expect("CUDA init");
        let dev = driver::device_get(0).expect("device 0");
        let _ctx = driver::ctx_create(dev).expect("ctx_create");
        let stream = driver::stream_create().expect("stream_create");
        (CachingAllocator::new(), stream)
    }

    unsafe fn upload_bf16_const(val: f32, count: usize, stream: CUstream) -> *mut u8 {
        let data: Vec<u16> = vec![half::bf16::from_f32(val).to_bits(); count];
        let bytes = count * 2;
        let ptr = driver::mem_alloc(bytes).expect("alloc");
        driver::memcpy_htod_async(ptr, data.as_ptr() as *const u8, bytes, stream).expect("h2d");
        driver::stream_synchronize(stream).expect("sync");
        ptr
    }

    unsafe fn upload_i32(data: &[i32], stream: CUstream) -> *mut u8 {
        let bytes = data.len() * 4;
        let ptr = driver::mem_alloc(bytes).expect("alloc");
        driver::memcpy_htod_async(ptr, data.as_ptr() as *const u8, bytes, stream).expect("h2d");
        driver::stream_synchronize(stream).expect("sync");
        ptr
    }

    unsafe fn download_bf16(tensor: GpuTensor, stream: CUstream) -> Vec<half::bf16> {
        let count = tensor.numel();
        let bytes = count * 2;
        let mut host = vec![half::bf16::ZERO; count];
        driver::memcpy_dtoh_async(host.as_mut_ptr() as *mut u8, tensor.as_ptr(), bytes, stream)
            .expect("d2h");
        driver::stream_synchronize(stream).expect("sync");
        host
    }

    /// Core test: all-ones input × all-0.5 weights, verifying output = K * 0.5.
    ///
    /// Tests a specific block_m size with given num_tokens and num_experts.
    /// Each token is assigned to expert 0 (top_k=1).
    unsafe fn run_bf16_gemm_test(
        num_tokens: usize,
        num_experts: usize,
        top_k: usize,
        k: usize,
        n: usize,
        block_m: usize,
    ) {
        let (mut alloc, stream) = test_init();

        // Input: all 1.0 BF16
        let input_ptr = upload_bf16_const(1.0, num_tokens * k, stream);
        let input = GpuTensor::new(input_ptr, &[num_tokens, k], DType::BF16);

        // Weights: all 0.5 BF16, [num_experts, N, K]
        let weight_ptr = upload_bf16_const(0.5, num_experts * n * k, stream);
        let weights = GpuTensor::new(weight_ptr, &[num_experts, n, k], DType::BF16);

        // Topk weights: all 1.0
        let tw_ptr = upload_bf16_const(1.0, num_tokens * top_k, stream);
        // topk_weights is F32, not BF16 — re-upload as f32
        driver::mem_free(tw_ptr).ok();
        let tw_data: Vec<f32> = vec![1.0; num_tokens * top_k];
        let tw_bytes = tw_data.len() * 4;
        let tw_ptr = driver::mem_alloc(tw_bytes).expect("alloc");
        driver::memcpy_htod_async(tw_ptr, tw_data.as_ptr() as *const u8, tw_bytes, stream)
            .expect("h2d");
        let topk_weights = GpuTensor::new(tw_ptr, &[num_tokens, top_k], DType::F32);

        // Topk IDs: assign each token to experts round-robin
        let topk_ids_data: Vec<i32> = (0..num_tokens * top_k)
            .map(|i| (i % num_experts) as i32)
            .collect();
        let topk_ids_ptr = upload_i32(&topk_ids_data, stream);
        let topk_ids = GpuTensor::new(topk_ids_ptr, &[num_tokens, top_k], DType::I32);

        // moe_align_block_size
        let (sorted, experts, ntpp) =
            moe_align_block_size(topk_ids, num_experts, block_m, &mut alloc, stream);

        // fused_moe_gemm (BF16)
        let output = fused_moe_gemm(
            input,
            weights,
            topk_weights,
            sorted.as_gpu_tensor(),
            experts.as_gpu_tensor(),
            ntpp.as_gpu_tensor(),
            num_tokens,
            top_k,
            block_m,
            false, // don't apply routing weights
            &mut alloc,
            stream,
        );

        assert_eq!(
            output.as_gpu_tensor().shape(),
            &[num_tokens as u32 * top_k as u32, n as u32]
        );

        let result = download_bf16(output.as_gpu_tensor(), stream);
        // Expected: each element = sum(1.0 * 0.5, K times) = K * 0.5
        let expected = k as f32 * 0.5;
        for (i, &val) in result.iter().enumerate() {
            let v = val.to_f32();
            assert!(
                (v - expected).abs() < 2.0,
                "block_m={block_m}, tokens={num_tokens}, experts={num_experts}: \
                 element {i}: got {v}, expected {expected}"
            );
        }

        driver::stream_destroy(stream).expect("destroy");
    }

    // --- BLOCK_M=128 (original tile size) ---

    #[test]
    #[ignore]
    fn test_bf16_moe_gemm_block_m_128() {
        // 8 tokens, 2 experts, top_k=1 → tokens_per_expert=4 → block_m=128 forced
        unsafe { run_bf16_gemm_test(8, 2, 1, 128, 128, 128) }
    }

    // --- BLOCK_M=64 ---

    #[test]
    #[ignore]
    fn test_bf16_moe_gemm_block_m_64() {
        unsafe { run_bf16_gemm_test(8, 2, 1, 128, 128, 64) }
    }

    // --- BLOCK_M=32 ---

    #[test]
    #[ignore]
    fn test_bf16_moe_gemm_block_m_32() {
        unsafe { run_bf16_gemm_test(8, 2, 1, 128, 128, 32) }
    }

    // --- BLOCK_M=16 (decode-critical, 8 warps × 1 N-tile each) ---

    #[test]
    #[ignore]
    fn test_bf16_moe_gemm_block_m_16() {
        unsafe { run_bf16_gemm_test(8, 2, 1, 128, 128, 16) }
    }

    // --- Single token (decode) at each block_m ---

    #[test]
    #[ignore]
    fn test_bf16_moe_gemm_single_token_block_m_16() {
        unsafe { run_bf16_gemm_test(1, 2, 1, 128, 128, 16) }
    }

    #[test]
    #[ignore]
    fn test_bf16_moe_gemm_single_token_block_m_128() {
        unsafe { run_bf16_gemm_test(1, 2, 1, 128, 128, 128) }
    }

    // --- Multi-expert top_k > 1 ---

    #[test]
    #[ignore]
    fn test_bf16_moe_gemm_topk_2_block_m_16() {
        unsafe { run_bf16_gemm_test(4, 4, 2, 128, 128, 16) }
    }

    #[test]
    #[ignore]
    fn test_bf16_moe_gemm_topk_2_block_m_128() {
        unsafe { run_bf16_gemm_test(4, 4, 2, 128, 128, 128) }
    }

    // --- Routing weight application ---

    #[test]
    #[ignore]
    fn test_bf16_moe_gemm_apply_weights() {
        unsafe {
            let (mut alloc, stream) = test_init();
            let num_tokens: usize = 4;
            let num_experts: usize = 2;
            let top_k: usize = 1;
            let k: usize = 128;
            let n: usize = 128;
            let block_m: usize = 16;

            let input_ptr = upload_bf16_const(1.0, num_tokens * k, stream);
            let input = GpuTensor::new(input_ptr, &[num_tokens, k], DType::BF16);

            let weight_ptr = upload_bf16_const(0.5, num_experts * n * k, stream);
            let weights = GpuTensor::new(weight_ptr, &[num_experts, n, k], DType::BF16);

            // Routing weights: 0.25 for all tokens
            let tw_data: Vec<f32> = vec![0.25; num_tokens * top_k];
            let tw_bytes = tw_data.len() * 4;
            let tw_ptr = driver::mem_alloc(tw_bytes).expect("alloc");
            driver::memcpy_htod_async(tw_ptr, tw_data.as_ptr() as *const u8, tw_bytes, stream)
                .expect("h2d");
            let topk_weights = GpuTensor::new(tw_ptr, &[num_tokens, top_k], DType::F32);

            let topk_ids_data: Vec<i32> = vec![0, 0, 1, 1];
            let topk_ids_ptr = upload_i32(&topk_ids_data, stream);
            let topk_ids = GpuTensor::new(topk_ids_ptr, &[num_tokens, top_k], DType::I32);

            let (sorted, experts, ntpp) =
                moe_align_block_size(topk_ids, num_experts, block_m, &mut alloc, stream);

            let output = fused_moe_gemm(
                input,
                weights,
                topk_weights,
                sorted.as_gpu_tensor(),
                experts.as_gpu_tensor(),
                ntpp.as_gpu_tensor(),
                num_tokens,
                top_k,
                block_m,
                true, // apply routing weights
                &mut alloc,
                stream,
            );

            let result = download_bf16(output.as_gpu_tensor(), stream);
            // Expected: K * 0.5 * 0.25 = 128 * 0.5 * 0.25 = 16.0
            let expected = 16.0f32;
            for (i, &val) in result.iter().enumerate() {
                let v = val.to_f32();
                assert!(
                    (v - expected).abs() < 1.0,
                    "apply_weights element {i}: got {v}, expected {expected}"
                );
            }

            driver::stream_destroy(stream).expect("destroy");
        }
    }

    // --- select_moe_block_m heuristic ---

    #[test]
    fn test_select_moe_block_m_heuristic() {
        use crate::layers_moe::select_moe_block_m;
        // 1 token, 64 experts, top_k=8 → tpe=0 → 16
        assert_eq!(select_moe_block_m(1, 8, 64), 16);
        // 10 tokens, 8 experts, top_k=2 → tpe=2 → 16
        assert_eq!(select_moe_block_m(10, 2, 8), 16);
        // 100 tokens, 8 experts, top_k=2 → tpe=25 → 32
        assert_eq!(select_moe_block_m(100, 2, 8), 32);
        // 500 tokens, 8 experts, top_k=2 → tpe=125 → 128
        assert_eq!(select_moe_block_m(500, 2, 8), 128);
        // Boundary: tpe=16 → 16, tpe=17 → 32
        assert_eq!(select_moe_block_m(16, 1, 1), 16);
        assert_eq!(select_moe_block_m(17, 1, 1), 32);
        // Boundary: tpe=64 → 64, tpe=65 → 128
        assert_eq!(select_moe_block_m(64, 1, 1), 64);
        assert_eq!(select_moe_block_m(65, 1, 1), 128);
    }
}

// ---------------------------------------------------------------------------
// Tests: fused_qkv_rope_cache (NeoX + interleaved)
// ---------------------------------------------------------------------------

#[cfg(test)]
#[cfg(feature = "cuda")]
mod tests_fused_qkv_rope_cache {
    use super::*;
    use crate::driver;

    type CUstream = cudarc::driver::sys::CUstream;

    unsafe fn test_init() -> (CachingAllocator, CUstream) {
        driver::init().expect("CUDA init");
        let dev = driver::device_get(0).expect("device 0");
        let _ctx = driver::ctx_create(dev).expect("ctx_create");
        let stream = driver::stream_create().expect("stream_create");
        let alloc = CachingAllocator::new();
        (alloc, stream)
    }

    fn f32_to_bf16(val: f32) -> u16 {
        half::bf16::from_f32(val).to_bits()
    }

    fn bf16_to_f32(bits: u16) -> f32 {
        half::bf16::from_bits(bits).to_f32()
    }

    unsafe fn upload_bf16(data: &[u16], stream: CUstream) -> *mut u8 {
        let bytes = data.len() * 2;
        let ptr = driver::mem_alloc(bytes).expect("mem_alloc");
        driver::memcpy_htod_async(ptr, data.as_ptr() as *const u8, bytes, stream).expect("H2D");
        ptr
    }

    unsafe fn upload_u32(data: &[u32], stream: CUstream) -> *mut u8 {
        let bytes = data.len() * 4;
        let ptr = driver::mem_alloc(bytes).expect("mem_alloc");
        driver::memcpy_htod_async(ptr, data.as_ptr() as *const u8, bytes, stream).expect("H2D");
        ptr
    }

    unsafe fn upload_i64(data: &[i64], stream: CUstream) -> *mut u8 {
        let bytes = data.len() * 8;
        let ptr = driver::mem_alloc(bytes).expect("mem_alloc");
        driver::memcpy_htod_async(ptr, data.as_ptr() as *const u8, bytes, stream).expect("H2D");
        ptr
    }

    unsafe fn download_bf16_raw(ptr: *mut u8, count: usize, stream: CUstream) -> Vec<u16> {
        let mut out = vec![0u16; count];
        driver::memcpy_dtoh_async(
            out.as_mut_ptr() as *mut u8,
            ptr as *const u8,
            count * 2,
            stream,
        )
        .expect("D2H");
        driver::stream_synchronize(stream).expect("sync");
        out
    }

    /// Build a cos_sin_cache for positions 0..max_pos.
    /// cache[pos, 0..half_rot] = cos, cache[pos, half_rot..rotary_dim] = sin.
    fn build_cos_sin_cache(max_pos: usize, rotary_dim: usize, base: f32) -> Vec<u16> {
        let half = rotary_dim / 2;
        let mut cache = vec![0u16; max_pos * rotary_dim];
        for pos in 0..max_pos {
            for i in 0..half {
                let freq = 1.0 / base.powf(2.0 * i as f32 / rotary_dim as f32);
                let angle = pos as f32 * freq;
                cache[pos * rotary_dim + i] = f32_to_bf16(angle.cos());
                cache[pos * rotary_dim + half + i] = f32_to_bf16(angle.sin());
            }
        }
        cache
    }

    /// Compare fused_qkv_rope_cache against separate fused_qkv_rope + reshape_and_cache.
    /// Verifies Q output matches and K/V in cache match.
    #[test]
    #[ignore] // requires CUDA GPU
    fn test_cuda_fused_qkv_rope_cache_neox() {
        unsafe {
            let (mut alloc, stream) = test_init();

            let num_tokens = 4;
            let num_q_heads = 4;
            let num_kv_heads = 2;
            let head_dim = 8;
            let rotary_dim = 8;
            let q_size = num_q_heads * head_dim;
            let kv_size = num_kv_heads * head_dim;
            let total_dim = q_size + 2 * kv_size;
            let block_size = 16;
            let num_blocks = 1;

            // Random-ish QKV input
            let qkv_f32: Vec<f32> = (0..num_tokens * total_dim)
                .map(|i| (i as f32 * 0.37 + 0.13).sin() * 2.0)
                .collect();
            let qkv_bf16: Vec<u16> = qkv_f32.iter().map(|&v| f32_to_bf16(v)).collect();

            let positions: Vec<u32> = (0..num_tokens as u32).collect();
            let slot_mapping: Vec<i64> = (0..num_tokens as i64).collect();
            let cos_sin_cache = build_cos_sin_cache(num_tokens + 4, rotary_dim, 10000.0);
            let max_pos = num_tokens + 4;

            // Upload two copies of everything for reference vs fused
            let qkv_ptr = upload_bf16(&qkv_bf16, stream);
            let qkv_ptr2 = upload_bf16(&qkv_bf16, stream);
            let pos_ptr = upload_u32(&positions, stream);
            let pos_ptr2 = upload_u32(&positions, stream);
            let slot_ptr = upload_i64(&slot_mapping, stream);
            let slot_ptr2 = upload_i64(&slot_mapping, stream);
            let cache_ptr = upload_bf16(&cos_sin_cache, stream);
            let cache_ptr2 = upload_bf16(&cos_sin_cache, stream);

            // --- Reference path: fused_qkv_rope + reshape_and_cache ---
            let (ref_q, ref_k, ref_v) = fused_qkv_rope(
                GpuTensor::new(qkv_ptr2, &[num_tokens, total_dim], DType::BF16),
                GpuTensor::new(pos_ptr2, &[num_tokens], DType::U32),
                GpuTensor::new(cache_ptr2, &[max_pos, rotary_dim], DType::BF16),
                q_size,
                kv_size,
                num_q_heads,
                num_kv_heads,
                head_dim,
                None,
                &mut alloc,
                stream,
            );

            let cache_elems = num_blocks * block_size * num_kv_heads * head_dim;
            let ref_kcache_ptr = upload_bf16(&vec![0u16; cache_elems], stream);
            let ref_vcache_ptr = upload_bf16(&vec![0u16; cache_elems], stream);

            reshape_and_cache(
                ref_k.as_gpu_tensor(),
                ref_v.as_gpu_tensor(),
                GpuTensor::new(
                    ref_kcache_ptr,
                    &[num_blocks, block_size, num_kv_heads, head_dim],
                    DType::BF16,
                ),
                GpuTensor::new(
                    ref_vcache_ptr,
                    &[num_blocks, block_size, num_kv_heads, head_dim],
                    DType::BF16,
                ),
                GpuTensor::new(slot_ptr2, &[num_tokens], DType::I64),
                block_size,
                stream,
            );

            // --- Fused path: fused_qkv_rope_cache ---
            let fused_kcache_ptr = upload_bf16(&vec![0u16; cache_elems], stream);
            let fused_vcache_ptr = upload_bf16(&vec![0u16; cache_elems], stream);

            let fused_q = fused_qkv_rope_cache(
                GpuTensor::new(qkv_ptr, &[num_tokens, total_dim], DType::BF16),
                GpuTensor::new(pos_ptr, &[num_tokens], DType::U32),
                GpuTensor::new(cache_ptr, &[max_pos, rotary_dim], DType::BF16),
                GpuTensor::new(slot_ptr, &[num_tokens], DType::I64),
                GpuTensor::new(
                    fused_kcache_ptr,
                    &[num_blocks, block_size, num_kv_heads, head_dim],
                    DType::BF16,
                ),
                GpuTensor::new(
                    fused_vcache_ptr,
                    &[num_blocks, block_size, num_kv_heads, head_dim],
                    DType::BF16,
                ),
                q_size,
                kv_size,
                num_q_heads,
                head_dim,
                None,
                &mut alloc,
                stream,
            );

            // --- Compare Q ---
            let ref_q_data = download_bf16_raw(
                ref_q.as_gpu_tensor().raw_ptr(),
                num_tokens * num_q_heads * head_dim,
                stream,
            );
            let fused_q_data = download_bf16_raw(
                fused_q.as_gpu_tensor().raw_ptr(),
                num_tokens * num_q_heads * head_dim,
                stream,
            );
            for i in 0..ref_q_data.len() {
                let r = bf16_to_f32(ref_q_data[i]);
                let f = bf16_to_f32(fused_q_data[i]);
                assert!((r - f).abs() < 1e-3, "Q mismatch at {i}: ref={r} fused={f}");
            }

            // --- Compare K cache ---
            let ref_kc = download_bf16_raw(ref_kcache_ptr, cache_elems, stream);
            let fused_kc = download_bf16_raw(fused_kcache_ptr, cache_elems, stream);
            for i in 0..cache_elems {
                let r = bf16_to_f32(ref_kc[i]);
                let f = bf16_to_f32(fused_kc[i]);
                assert!(
                    (r - f).abs() < 1e-3,
                    "K cache mismatch at {i}: ref={r} fused={f}"
                );
            }

            // --- Compare V cache ---
            let ref_vc = download_bf16_raw(ref_vcache_ptr, cache_elems, stream);
            let fused_vc = download_bf16_raw(fused_vcache_ptr, cache_elems, stream);
            for i in 0..cache_elems {
                let r = bf16_to_f32(ref_vc[i]);
                let f = bf16_to_f32(fused_vc[i]);
                assert!(
                    (r - f).abs() < 1e-3,
                    "V cache mismatch at {i}: ref={r} fused={f}"
                );
            }

            driver::stream_destroy(stream).expect("destroy");
        }
    }

    /// MRoPE (Qwen2-VL): positions tensor is `[3, num_tokens]` and
    /// `mrope_section = Some([T,H,W])` partitions the half_rot rotary
    /// pairs across three position axes. Verify the fused_qkv_rope
    /// kernel against an independent CPU reference computation, AND
    /// verify the degenerate case (all three position rows equal)
    /// matches the legacy 1D path bit-for-bit.
    ///
    /// **Currently gated out**: `cargo test -p scratchy-target-cuda` cannot
    /// link the test binary because it pulls in CUDA `.a` symbols
    /// (`mha_varlen_fwd` etc.) that only resolve when `scratchy-cli`
    /// consumes both scratchy-target-cuda and scratchy-serving-cuda. The MRoPE math
    /// will be exercised end-to-end at Phase B (`scr chat --image`
    /// on Qwen2-VL with `MROPE_SECTION = Some([16,24,24])`); a Python
    /// vLLM byte-for-byte greedy match is the real verifier. The
    /// CPU reference below stays in tree as the ground-truth spec
    /// for that future verifier.
    #[test]
    #[ignore]
    #[cfg(any())] // disabled — see docstring above
    fn test_cuda_fused_qkv_rope_mrope() {
        unsafe {
            let (mut alloc, stream) = test_init();

            let num_tokens: usize = 4;
            let num_q_heads: usize = 2;
            let num_kv_heads: usize = 1;
            let head_dim: usize = 8;
            let rotary_dim: usize = 8;
            let half_rot: usize = rotary_dim / 2; // 4
            let q_size = num_q_heads * head_dim;
            let kv_size = num_kv_heads * head_dim;
            let total_dim = q_size + 2 * kv_size;

            // mrope_section [2, 1, 1] sums to half_rot = 4. VEC-aligned
            // not strictly required at this small size (scalar tail
            // handles unaligned remainder).
            let mrope_section: [u32; 3] = [2, 1, 1];

            // Positions [3, num_tokens] in row-major: T row, H row, W row.
            let pos_t: [u32; 4] = [0, 1, 2, 3];
            let pos_h: [u32; 4] = [2, 3, 4, 5];
            let pos_w: [u32; 4] = [5, 6, 7, 8];
            let mut positions_3d: Vec<u32> = Vec::with_capacity(3 * num_tokens);
            positions_3d.extend_from_slice(&pos_t);
            positions_3d.extend_from_slice(&pos_h);
            positions_3d.extend_from_slice(&pos_w);

            let qkv_f32: Vec<f32> = (0..num_tokens * total_dim)
                .map(|i| (i as f32 * 0.37 + 0.13).sin() * 2.0)
                .collect();
            let qkv_bf16: Vec<u16> = qkv_f32.iter().map(|&v| f32_to_bf16(v)).collect();

            let max_pos: usize = 16;
            let cos_sin_cache = build_cos_sin_cache(max_pos, rotary_dim, 10000.0);

            // ---------- Path A: MRoPE on GPU ----------
            let qkv_ptr = upload_bf16(&qkv_bf16, stream);
            let pos_ptr = upload_u32(&positions_3d, stream);
            let cache_ptr = upload_bf16(&cos_sin_cache, stream);

            let (q_gpu, k_gpu, _v_gpu) = fused_qkv_rope(
                GpuTensor::new(qkv_ptr, &[num_tokens, total_dim], DType::BF16),
                GpuTensor::new(pos_ptr, &[3, num_tokens], DType::U32),
                GpuTensor::new(cache_ptr, &[max_pos, rotary_dim], DType::BF16),
                q_size,
                kv_size,
                num_q_heads,
                num_kv_heads,
                head_dim,
                Some(mrope_section),
                &mut alloc,
                stream,
            );
            let q_out_bits =
                download_bf16_raw(q_gpu.as_gpu_tensor().raw_ptr(), num_tokens * q_size, stream);
            let k_out_bits = download_bf16_raw(
                k_gpu.as_gpu_tensor().raw_ptr(),
                num_tokens * kv_size,
                stream,
            );

            // ---------- Path B: CPU reference ----------
            // For each token t, head h, rotary index r in [0, half_rot):
            //   axis = (r < sec0) ? 0 : (r < sec01) ? 1 : 2
            //   pos = positions[axis, t]
            //   c = cache[pos, r], s = cache[pos, half_rot + r]
            //   x = qkv[t, h_off + r], y = qkv[t, h_off + half_rot + r]
            //   out[h_off + r]            = x*c - y*s
            //   out[h_off + half_rot + r] = y*c + x*s
            let sec0 = mrope_section[0] as usize;
            let sec01 = sec0 + mrope_section[1] as usize;
            let pick_pos = |t: usize, r: usize| -> u32 {
                let axis = if r < sec0 {
                    0
                } else if r < sec01 {
                    1
                } else {
                    2
                };
                positions_3d[axis * num_tokens + t]
            };
            let mut q_ref = vec![0u16; num_tokens * q_size];
            let mut k_ref = vec![0u16; num_tokens * kv_size];
            for t in 0..num_tokens {
                for h in 0..num_q_heads {
                    let h_off_in = t * total_dim + h * head_dim;
                    let h_off_out = t * q_size + h * head_dim;
                    for r in 0..half_rot {
                        let pos = pick_pos(t, r) as usize;
                        let c = bf16_to_f32(cos_sin_cache[pos * rotary_dim + r]);
                        let s = bf16_to_f32(cos_sin_cache[pos * rotary_dim + half_rot + r]);
                        let x = bf16_to_f32(qkv_bf16[h_off_in + r]);
                        let y = bf16_to_f32(qkv_bf16[h_off_in + half_rot + r]);
                        q_ref[h_off_out + r] = f32_to_bf16(x * c - y * s);
                        q_ref[h_off_out + half_rot + r] = f32_to_bf16(y * c + x * s);
                    }
                }
                for h in 0..num_kv_heads {
                    let h_off_in = t * total_dim + q_size + h * head_dim;
                    let h_off_out = t * kv_size + h * head_dim;
                    for r in 0..half_rot {
                        let pos = pick_pos(t, r) as usize;
                        let c = bf16_to_f32(cos_sin_cache[pos * rotary_dim + r]);
                        let s = bf16_to_f32(cos_sin_cache[pos * rotary_dim + half_rot + r]);
                        let x = bf16_to_f32(qkv_bf16[h_off_in + r]);
                        let y = bf16_to_f32(qkv_bf16[h_off_in + half_rot + r]);
                        k_ref[h_off_out + r] = f32_to_bf16(x * c - y * s);
                        k_ref[h_off_out + half_rot + r] = f32_to_bf16(y * c + x * s);
                    }
                }
            }

            // Compare with bf16 tolerance.
            for (i, (&g, &r)) in q_out_bits.iter().zip(q_ref.iter()).enumerate() {
                let dg = bf16_to_f32(g);
                let dr = bf16_to_f32(r);
                let diff = (dg - dr).abs();
                let scale = dg.abs().max(dr.abs()).max(1.0);
                assert!(
                    diff / scale < 2e-2,
                    "Q[{i}] MRoPE mismatch: gpu={dg} ref={dr} (rel_diff={})",
                    diff / scale
                );
            }
            for (i, (&g, &r)) in k_out_bits.iter().zip(k_ref.iter()).enumerate() {
                let dg = bf16_to_f32(g);
                let dr = bf16_to_f32(r);
                let diff = (dg - dr).abs();
                let scale = dg.abs().max(dr.abs()).max(1.0);
                assert!(
                    diff / scale < 2e-2,
                    "K[{i}] MRoPE mismatch: gpu={dg} ref={dr} (rel_diff={})",
                    diff / scale
                );
            }

            // ---------- Degenerate consistency: all three rows equal ----------
            // Fill positions_3d so all three axes carry the same row →
            // MRoPE output must equal the legacy 1D output bit-for-bit.
            let positions_1d: Vec<u32> = pos_t.to_vec();
            let mut positions_3d_eq = Vec::with_capacity(3 * num_tokens);
            positions_3d_eq.extend_from_slice(&positions_1d);
            positions_3d_eq.extend_from_slice(&positions_1d);
            positions_3d_eq.extend_from_slice(&positions_1d);

            let qkv_ptr_a = upload_bf16(&qkv_bf16, stream);
            let pos_ptr_a = upload_u32(&positions_3d_eq, stream);
            let cache_ptr_a = upload_bf16(&cos_sin_cache, stream);
            let (q_a, _k_a, _v_a) = fused_qkv_rope(
                GpuTensor::new(qkv_ptr_a, &[num_tokens, total_dim], DType::BF16),
                GpuTensor::new(pos_ptr_a, &[3, num_tokens], DType::U32),
                GpuTensor::new(cache_ptr_a, &[max_pos, rotary_dim], DType::BF16),
                q_size,
                kv_size,
                num_q_heads,
                num_kv_heads,
                head_dim,
                Some(mrope_section),
                &mut alloc,
                stream,
            );
            let q_a_bits =
                download_bf16_raw(q_a.as_gpu_tensor().raw_ptr(), num_tokens * q_size, stream);

            let qkv_ptr_b = upload_bf16(&qkv_bf16, stream);
            let pos_ptr_b = upload_u32(&positions_1d, stream);
            let cache_ptr_b = upload_bf16(&cos_sin_cache, stream);
            let (q_b, _k_b, _v_b) = fused_qkv_rope(
                GpuTensor::new(qkv_ptr_b, &[num_tokens, total_dim], DType::BF16),
                GpuTensor::new(pos_ptr_b, &[num_tokens], DType::U32),
                GpuTensor::new(cache_ptr_b, &[max_pos, rotary_dim], DType::BF16),
                q_size,
                kv_size,
                num_q_heads,
                num_kv_heads,
                head_dim,
                None,
                &mut alloc,
                stream,
            );
            let q_b_bits =
                download_bf16_raw(q_b.as_gpu_tensor().raw_ptr(), num_tokens * q_size, stream);

            for (i, (&a, &b)) in q_a_bits.iter().zip(q_b_bits.iter()).enumerate() {
                assert_eq!(
                    a, b,
                    "Degenerate MRoPE Q[{i}] differs from 1D path: mrope=0x{a:04x} 1d=0x{b:04x}"
                );
            }

            driver::stream_destroy(stream).expect("destroy");
        }
    }

    /// Test with negative slot_mapping (padding tokens) — cache should be untouched.
    #[test]
    #[ignore] // requires CUDA GPU
    fn test_cuda_fused_qkv_rope_cache_padding() {
        unsafe {
            let (mut alloc, stream) = test_init();

            let num_tokens = 2;
            let num_q_heads = 2;
            let num_kv_heads = 1;
            let head_dim = 8;
            let rotary_dim = 8;
            let q_size = num_q_heads * head_dim;
            let kv_size = num_kv_heads * head_dim;
            let total_dim = q_size + 2 * kv_size;
            let block_size = 16;
            let num_blocks = 1;

            let qkv_f32: Vec<f32> = (0..num_tokens * total_dim)
                .map(|i| (i as f32 * 0.5).sin())
                .collect();
            let qkv_bf16: Vec<u16> = qkv_f32.iter().map(|&v| f32_to_bf16(v)).collect();

            let positions: Vec<u32> = vec![0, 1];
            // slot 0 valid, slot 1 is padding (-1)
            let slot_mapping: Vec<i64> = vec![0, -1];
            let cos_sin_cache = build_cos_sin_cache(8, rotary_dim, 10000.0);

            let qkv_ptr = upload_bf16(&qkv_bf16, stream);
            let pos_ptr = upload_u32(&positions, stream);
            let slot_ptr = upload_i64(&slot_mapping, stream);
            let cache_ptr = upload_bf16(&cos_sin_cache, stream);

            let cache_elems = num_blocks * block_size * num_kv_heads * head_dim;
            // Fill cache with sentinel value
            let sentinel = vec![0xBEEFu16; cache_elems];
            let kcache_ptr = upload_bf16(&sentinel, stream);
            let vcache_ptr = upload_bf16(&sentinel, stream);

            let _q = fused_qkv_rope_cache(
                GpuTensor::new(qkv_ptr, &[num_tokens, total_dim], DType::BF16),
                GpuTensor::new(pos_ptr, &[num_tokens], DType::U32),
                GpuTensor::new(cache_ptr, &[8, rotary_dim], DType::BF16),
                GpuTensor::new(slot_ptr, &[num_tokens], DType::I64),
                GpuTensor::new(
                    kcache_ptr,
                    &[num_blocks, block_size, num_kv_heads, head_dim],
                    DType::BF16,
                ),
                GpuTensor::new(
                    vcache_ptr,
                    &[num_blocks, block_size, num_kv_heads, head_dim],
                    DType::BF16,
                ),
                q_size,
                kv_size,
                num_q_heads,
                head_dim,
                None,
                &mut alloc,
                stream,
            );

            let kc = download_bf16_raw(kcache_ptr, cache_elems, stream);

            // Slot 1 region: offset = 1 * kv_size = 8 elements — should be sentinel.
            for (j, &val) in kc[kv_size..(2 * kv_size)].iter().enumerate() {
                assert_eq!(
                    val,
                    0xBEEF,
                    "Padding slot should be untouched, but slot 1 K[{}] = 0x{:04X}",
                    j + kv_size,
                    val
                );
            }

            // Slot 0 should NOT be sentinel (it was written).
            let any_written = (0..kv_size).any(|i| kc[i] != 0xBEEF);
            assert!(any_written, "Slot 0 K cache should have been written");

            driver::stream_destroy(stream).expect("destroy");
        }
    }

    /// Compare fused_qkv_interleaved_rope_cache against separate
    /// fused_qkv_interleaved_rope + reshape_and_cache.
    #[test]
    #[ignore] // requires CUDA GPU
    fn test_cuda_fused_qkv_interleaved_rope_cache() {
        unsafe {
            let (mut alloc, stream) = test_init();

            let num_tokens = 3;
            let num_q_heads = 4;
            let num_kv_heads = 2;
            let head_dim = 8;
            let rotary_dim = 8;
            let q_size = num_q_heads * head_dim;
            let kv_size = num_kv_heads * head_dim;
            let total_dim = q_size + 2 * kv_size;
            let block_size = 16;
            let num_blocks = 1;

            let qkv_f32: Vec<f32> = (0..num_tokens * total_dim)
                .map(|i| (i as f32 * 0.23 + 0.7).cos() * 1.5)
                .collect();
            let qkv_bf16: Vec<u16> = qkv_f32.iter().map(|&v| f32_to_bf16(v)).collect();

            let positions: Vec<u32> = (0..num_tokens as u32).collect();
            let slot_mapping: Vec<i64> = (0..num_tokens as i64).collect();
            let cos_sin_cache = build_cos_sin_cache(num_tokens + 4, rotary_dim, 10000.0);
            let max_pos = num_tokens + 4;

            // Upload two copies for reference vs fused
            let qkv_ptr = upload_bf16(&qkv_bf16, stream);
            let qkv_ptr2 = upload_bf16(&qkv_bf16, stream);
            let pos_ptr = upload_u32(&positions, stream);
            let pos_ptr2 = upload_u32(&positions, stream);
            let slot_ptr = upload_i64(&slot_mapping, stream);
            let slot_ptr2 = upload_i64(&slot_mapping, stream);
            let cache_ptr = upload_bf16(&cos_sin_cache, stream);
            let cache_ptr2 = upload_bf16(&cos_sin_cache, stream);

            // --- Reference: fused_qkv_interleaved_rope + reshape_and_cache ---
            let (ref_q, ref_k, ref_v) = fused_qkv_interleaved_rope(
                GpuTensor::new(qkv_ptr2, &[num_tokens, total_dim], DType::BF16),
                GpuTensor::new(pos_ptr2, &[num_tokens], DType::U32),
                GpuTensor::new(cache_ptr2, &[max_pos, rotary_dim], DType::BF16),
                q_size,
                kv_size,
                num_q_heads,
                num_kv_heads,
                head_dim,
                &mut alloc,
                stream,
            );

            let cache_elems = num_blocks * block_size * num_kv_heads * head_dim;
            let ref_kcache_ptr = upload_bf16(&vec![0u16; cache_elems], stream);
            let ref_vcache_ptr = upload_bf16(&vec![0u16; cache_elems], stream);

            reshape_and_cache(
                ref_k.as_gpu_tensor(),
                ref_v.as_gpu_tensor(),
                GpuTensor::new(
                    ref_kcache_ptr,
                    &[num_blocks, block_size, num_kv_heads, head_dim],
                    DType::BF16,
                ),
                GpuTensor::new(
                    ref_vcache_ptr,
                    &[num_blocks, block_size, num_kv_heads, head_dim],
                    DType::BF16,
                ),
                GpuTensor::new(slot_ptr2, &[num_tokens], DType::I64),
                block_size,
                stream,
            );

            // --- Fused path ---
            let fused_kcache_ptr = upload_bf16(&vec![0u16; cache_elems], stream);
            let fused_vcache_ptr = upload_bf16(&vec![0u16; cache_elems], stream);

            let fused_q = fused_qkv_interleaved_rope_cache(
                GpuTensor::new(qkv_ptr, &[num_tokens, total_dim], DType::BF16),
                GpuTensor::new(pos_ptr, &[num_tokens], DType::U32),
                GpuTensor::new(cache_ptr, &[max_pos, rotary_dim], DType::BF16),
                GpuTensor::new(slot_ptr, &[num_tokens], DType::I64),
                GpuTensor::new(
                    fused_kcache_ptr,
                    &[num_blocks, block_size, num_kv_heads, head_dim],
                    DType::BF16,
                ),
                GpuTensor::new(
                    fused_vcache_ptr,
                    &[num_blocks, block_size, num_kv_heads, head_dim],
                    DType::BF16,
                ),
                q_size,
                kv_size,
                num_q_heads,
                head_dim,
                None,
                &mut alloc,
                stream,
            );

            // --- Compare Q ---
            let ref_q_data = download_bf16_raw(
                ref_q.as_gpu_tensor().raw_ptr(),
                num_tokens * num_q_heads * head_dim,
                stream,
            );
            let fused_q_data = download_bf16_raw(
                fused_q.as_gpu_tensor().raw_ptr(),
                num_tokens * num_q_heads * head_dim,
                stream,
            );
            for i in 0..ref_q_data.len() {
                let r = bf16_to_f32(ref_q_data[i]);
                let f = bf16_to_f32(fused_q_data[i]);
                assert!((r - f).abs() < 1e-3, "Q mismatch at {i}: ref={r} fused={f}");
            }

            // --- Compare K cache ---
            let ref_kc = download_bf16_raw(ref_kcache_ptr, cache_elems, stream);
            let fused_kc = download_bf16_raw(fused_kcache_ptr, cache_elems, stream);
            for i in 0..cache_elems {
                let r = bf16_to_f32(ref_kc[i]);
                let f = bf16_to_f32(fused_kc[i]);
                assert!(
                    (r - f).abs() < 1e-3,
                    "K cache mismatch at {i}: ref={r} fused={f}"
                );
            }

            // --- Compare V cache ---
            let ref_vc = download_bf16_raw(ref_vcache_ptr, cache_elems, stream);
            let fused_vc = download_bf16_raw(fused_vcache_ptr, cache_elems, stream);
            for i in 0..cache_elems {
                let r = bf16_to_f32(ref_vc[i]);
                let f = bf16_to_f32(fused_vc[i]);
                assert!(
                    (r - f).abs() < 1e-3,
                    "V cache mismatch at {i}: ref={r} fused={f}"
                );
            }

            driver::stream_destroy(stream).expect("destroy");
        }
    }

    unsafe fn upload_f32_scalar(val: f32, stream: CUstream) -> *mut u8 {
        let ptr = driver::mem_alloc(4).expect("mem_alloc");
        driver::memcpy_htod_async(ptr, &val as *const f32 as *const u8, 4, stream).expect("H2D");
        ptr
    }

    unsafe fn download_u8_raw(ptr: *mut u8, count: usize, stream: CUstream) -> Vec<u8> {
        let mut out = vec![0u8; count];
        driver::memcpy_dtoh_async(out.as_mut_ptr(), ptr as *const u8, count, stream).expect("D2H");
        driver::stream_synchronize(stream).expect("sync");
        out
    }

    /// Compare fused_qkv_rope_cache_fp8 against separate fused_qkv_rope + reshape_and_cache_fp8.
    #[test]
    #[ignore]
    fn test_cuda_fused_qkv_rope_cache_fp8_neox() {
        unsafe {
            let (mut alloc, stream) = test_init();

            let num_tokens = 4;
            let num_q_heads = 4;
            let num_kv_heads = 2;
            let head_dim = 8;
            let rotary_dim = 8;
            let q_size = num_q_heads * head_dim;
            let kv_size = num_kv_heads * head_dim;
            let total_dim = q_size + 2 * kv_size;
            let block_size = 16;
            let num_blocks = 1;

            let qkv_f32: Vec<f32> = (0..num_tokens * total_dim)
                .map(|i| (i as f32 * 0.37 + 0.13).sin() * 2.0)
                .collect();
            let qkv_bf16: Vec<u16> = qkv_f32.iter().map(|&v| f32_to_bf16(v)).collect();

            let positions: Vec<u32> = (0..num_tokens as u32).collect();
            let slot_mapping: Vec<i64> = (0..num_tokens as i64).collect();
            let cos_sin_cache = build_cos_sin_cache(num_tokens + 4, rotary_dim, 10000.0);
            let max_pos = num_tokens + 4;

            let k_scale_ptr = upload_f32_scalar(1.0, stream);
            let v_scale_ptr = upload_f32_scalar(1.0, stream);
            let k_scale_ptr2 = upload_f32_scalar(1.0, stream);
            let v_scale_ptr2 = upload_f32_scalar(1.0, stream);

            let qkv_ptr = upload_bf16(&qkv_bf16, stream);
            let qkv_ptr2 = upload_bf16(&qkv_bf16, stream);
            let pos_ptr = upload_u32(&positions, stream);
            let pos_ptr2 = upload_u32(&positions, stream);
            let slot_ptr = upload_i64(&slot_mapping, stream);
            let slot_ptr2 = upload_i64(&slot_mapping, stream);
            let cache_ptr = upload_bf16(&cos_sin_cache, stream);
            let cache_ptr2 = upload_bf16(&cos_sin_cache, stream);

            // --- Reference: fused_qkv_rope + reshape_and_cache_fp8 ---
            let (ref_q, ref_k, ref_v) = fused_qkv_rope(
                GpuTensor::new(qkv_ptr2, &[num_tokens, total_dim], DType::BF16),
                GpuTensor::new(pos_ptr2, &[num_tokens], DType::U32),
                GpuTensor::new(cache_ptr2, &[max_pos, rotary_dim], DType::BF16),
                q_size,
                kv_size,
                num_q_heads,
                num_kv_heads,
                head_dim,
                None,
                &mut alloc,
                stream,
            );

            let cache_elems = num_blocks * block_size * num_kv_heads * head_dim;
            let ref_kcache_ptr = driver::mem_alloc(cache_elems).expect("alloc");
            let ref_vcache_ptr = driver::mem_alloc(cache_elems).expect("alloc");
            driver::memset_d8(ref_kcache_ptr, 0, cache_elems, stream).expect("memset");
            driver::memset_d8(ref_vcache_ptr, 0, cache_elems, stream).expect("memset");

            reshape_and_cache_fp8(
                ref_k.as_gpu_tensor(),
                ref_v.as_gpu_tensor(),
                GpuTensor::new(
                    ref_kcache_ptr,
                    &[num_blocks, block_size, num_kv_heads, head_dim],
                    DType::Fp8E4m3,
                ),
                GpuTensor::new(
                    ref_vcache_ptr,
                    &[num_blocks, block_size, num_kv_heads, head_dim],
                    DType::Fp8E4m3,
                ),
                GpuTensor::new(slot_ptr2, &[num_tokens], DType::I64),
                k_scale_ptr2 as *const f32,
                v_scale_ptr2 as *const f32,
                block_size,
                stream,
            );

            // --- Fused path ---
            let fused_kcache_ptr = driver::mem_alloc(cache_elems).expect("alloc");
            let fused_vcache_ptr = driver::mem_alloc(cache_elems).expect("alloc");
            driver::memset_d8(fused_kcache_ptr, 0, cache_elems, stream).expect("memset");
            driver::memset_d8(fused_vcache_ptr, 0, cache_elems, stream).expect("memset");

            let fused_q = fused_qkv_rope_cache_fp8(
                GpuTensor::new(qkv_ptr, &[num_tokens, total_dim], DType::BF16),
                GpuTensor::new(pos_ptr, &[num_tokens], DType::U32),
                GpuTensor::new(cache_ptr, &[max_pos, rotary_dim], DType::BF16),
                GpuTensor::new(slot_ptr, &[num_tokens], DType::I64),
                GpuTensor::new(
                    fused_kcache_ptr,
                    &[num_blocks, block_size, num_kv_heads, head_dim],
                    DType::Fp8E4m3,
                ),
                GpuTensor::new(
                    fused_vcache_ptr,
                    &[num_blocks, block_size, num_kv_heads, head_dim],
                    DType::Fp8E4m3,
                ),
                k_scale_ptr as *const f32,
                v_scale_ptr as *const f32,
                q_size,
                kv_size,
                num_q_heads,
                head_dim,
                None,
                &mut alloc,
                stream,
            );

            // Compare Q (BF16)
            let ref_q_data = download_bf16_raw(
                ref_q.as_gpu_tensor().raw_ptr(),
                num_tokens * num_q_heads * head_dim,
                stream,
            );
            let fused_q_data = download_bf16_raw(
                fused_q.as_gpu_tensor().raw_ptr(),
                num_tokens * num_q_heads * head_dim,
                stream,
            );
            for i in 0..ref_q_data.len() {
                let r = bf16_to_f32(ref_q_data[i]);
                let f = bf16_to_f32(fused_q_data[i]);
                assert!(
                    (r - f).abs() < 1e-3,
                    "FP8 Q mismatch at {i}: ref={r} fused={f}"
                );
            }

            // Compare K cache (FP8 bytes) — allow ±1 for rounding difference
            // (fused does f32→FP8 directly, reference does f32→BF16→FP8)
            let ref_kc = download_u8_raw(ref_kcache_ptr, cache_elems, stream);
            let fused_kc = download_u8_raw(fused_kcache_ptr, cache_elems, stream);
            for i in 0..cache_elems {
                let diff = (ref_kc[i] as i16 - fused_kc[i] as i16).unsigned_abs();
                assert!(
                    diff <= 1,
                    "FP8 K cache mismatch at {i}: ref=0x{:02X} fused=0x{:02X} (diff={diff})",
                    ref_kc[i],
                    fused_kc[i]
                );
            }

            // Compare V cache (FP8 bytes) — allow ±1
            let ref_vc = download_u8_raw(ref_vcache_ptr, cache_elems, stream);
            let fused_vc = download_u8_raw(fused_vcache_ptr, cache_elems, stream);
            for i in 0..cache_elems {
                let diff = (ref_vc[i] as i16 - fused_vc[i] as i16).unsigned_abs();
                assert!(
                    diff <= 1,
                    "FP8 V cache mismatch at {i}: ref=0x{:02X} fused=0x{:02X} (diff={diff})",
                    ref_vc[i],
                    fused_vc[i]
                );
            }

            driver::stream_destroy(stream).expect("destroy");
        }
    }

    /// Compare fused_qkv_interleaved_rope_cache_fp8 against separate path.
    #[test]
    #[ignore]
    fn test_cuda_fused_qkv_interleaved_rope_cache_fp8() {
        unsafe {
            let (mut alloc, stream) = test_init();

            let num_tokens = 3;
            let num_q_heads = 4;
            let num_kv_heads = 2;
            let head_dim = 8;
            let rotary_dim = 8;
            let q_size = num_q_heads * head_dim;
            let kv_size = num_kv_heads * head_dim;
            let total_dim = q_size + 2 * kv_size;
            let block_size = 16;
            let num_blocks = 1;

            let qkv_f32: Vec<f32> = (0..num_tokens * total_dim)
                .map(|i| (i as f32 * 0.23 + 0.7).cos() * 1.5)
                .collect();
            let qkv_bf16: Vec<u16> = qkv_f32.iter().map(|&v| f32_to_bf16(v)).collect();

            let positions: Vec<u32> = (0..num_tokens as u32).collect();
            let slot_mapping: Vec<i64> = (0..num_tokens as i64).collect();
            let cos_sin_cache = build_cos_sin_cache(num_tokens + 4, rotary_dim, 10000.0);
            let max_pos = num_tokens + 4;

            let k_scale_ptr = upload_f32_scalar(1.0, stream);
            let v_scale_ptr = upload_f32_scalar(1.0, stream);
            let k_scale_ptr2 = upload_f32_scalar(1.0, stream);
            let v_scale_ptr2 = upload_f32_scalar(1.0, stream);

            let qkv_ptr = upload_bf16(&qkv_bf16, stream);
            let qkv_ptr2 = upload_bf16(&qkv_bf16, stream);
            let pos_ptr = upload_u32(&positions, stream);
            let pos_ptr2 = upload_u32(&positions, stream);
            let slot_ptr = upload_i64(&slot_mapping, stream);
            let slot_ptr2 = upload_i64(&slot_mapping, stream);
            let cache_ptr = upload_bf16(&cos_sin_cache, stream);
            let cache_ptr2 = upload_bf16(&cos_sin_cache, stream);

            // --- Reference ---
            let (ref_q, ref_k, ref_v) = fused_qkv_interleaved_rope(
                GpuTensor::new(qkv_ptr2, &[num_tokens, total_dim], DType::BF16),
                GpuTensor::new(pos_ptr2, &[num_tokens], DType::U32),
                GpuTensor::new(cache_ptr2, &[max_pos, rotary_dim], DType::BF16),
                q_size,
                kv_size,
                num_q_heads,
                num_kv_heads,
                head_dim,
                &mut alloc,
                stream,
            );

            let cache_elems = num_blocks * block_size * num_kv_heads * head_dim;
            let ref_kcache_ptr = driver::mem_alloc(cache_elems).expect("alloc");
            let ref_vcache_ptr = driver::mem_alloc(cache_elems).expect("alloc");
            driver::memset_d8(ref_kcache_ptr, 0, cache_elems, stream).expect("memset");
            driver::memset_d8(ref_vcache_ptr, 0, cache_elems, stream).expect("memset");

            reshape_and_cache_fp8(
                ref_k.as_gpu_tensor(),
                ref_v.as_gpu_tensor(),
                GpuTensor::new(
                    ref_kcache_ptr,
                    &[num_blocks, block_size, num_kv_heads, head_dim],
                    DType::Fp8E4m3,
                ),
                GpuTensor::new(
                    ref_vcache_ptr,
                    &[num_blocks, block_size, num_kv_heads, head_dim],
                    DType::Fp8E4m3,
                ),
                GpuTensor::new(slot_ptr2, &[num_tokens], DType::I64),
                k_scale_ptr2 as *const f32,
                v_scale_ptr2 as *const f32,
                block_size,
                stream,
            );

            // --- Fused path ---
            let fused_kcache_ptr = driver::mem_alloc(cache_elems).expect("alloc");
            let fused_vcache_ptr = driver::mem_alloc(cache_elems).expect("alloc");
            driver::memset_d8(fused_kcache_ptr, 0, cache_elems, stream).expect("memset");
            driver::memset_d8(fused_vcache_ptr, 0, cache_elems, stream).expect("memset");

            let fused_q = fused_qkv_interleaved_rope_cache_fp8(
                GpuTensor::new(qkv_ptr, &[num_tokens, total_dim], DType::BF16),
                GpuTensor::new(pos_ptr, &[num_tokens], DType::U32),
                GpuTensor::new(cache_ptr, &[max_pos, rotary_dim], DType::BF16),
                GpuTensor::new(slot_ptr, &[num_tokens], DType::I64),
                GpuTensor::new(
                    fused_kcache_ptr,
                    &[num_blocks, block_size, num_kv_heads, head_dim],
                    DType::Fp8E4m3,
                ),
                GpuTensor::new(
                    fused_vcache_ptr,
                    &[num_blocks, block_size, num_kv_heads, head_dim],
                    DType::Fp8E4m3,
                ),
                k_scale_ptr as *const f32,
                v_scale_ptr as *const f32,
                q_size,
                kv_size,
                num_q_heads,
                head_dim,
                None,
                &mut alloc,
                stream,
            );

            // Compare Q
            let ref_q_data = download_bf16_raw(
                ref_q.as_gpu_tensor().raw_ptr(),
                num_tokens * num_q_heads * head_dim,
                stream,
            );
            let fused_q_data = download_bf16_raw(
                fused_q.as_gpu_tensor().raw_ptr(),
                num_tokens * num_q_heads * head_dim,
                stream,
            );
            for i in 0..ref_q_data.len() {
                let r = bf16_to_f32(ref_q_data[i]);
                let f = bf16_to_f32(fused_q_data[i]);
                assert!(
                    (r - f).abs() < 1e-3,
                    "FP8 interleaved Q mismatch at {i}: ref={r} fused={f}"
                );
            }

            // Compare K cache (FP8 bytes) — allow ±1 for rounding
            let ref_kc = download_u8_raw(ref_kcache_ptr, cache_elems, stream);
            let fused_kc = download_u8_raw(fused_kcache_ptr, cache_elems, stream);
            for i in 0..cache_elems {
                let diff = (ref_kc[i] as i16 - fused_kc[i] as i16).unsigned_abs();
                assert!(
                    diff <= 1,
                    "FP8 interleaved K cache mismatch at {i}: ref=0x{:02X} fused=0x{:02X}",
                    ref_kc[i],
                    fused_kc[i]
                );
            }

            // Compare V cache (FP8 bytes) — allow ±1 for rounding
            let ref_vc = download_u8_raw(ref_vcache_ptr, cache_elems, stream);
            let fused_vc = download_u8_raw(fused_vcache_ptr, cache_elems, stream);
            for i in 0..cache_elems {
                let diff = (ref_vc[i] as i16 - fused_vc[i] as i16).unsigned_abs();
                assert!(
                    diff <= 1,
                    "FP8 interleaved V cache mismatch at {i}: ref=0x{:02X} fused=0x{:02X}",
                    ref_vc[i],
                    fused_vc[i]
                );
            }

            driver::stream_destroy(stream).expect("destroy");
        }
    }
}

// ---------------------------------------------------------------------------
// LayerNorm bias kernel tests
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "cuda"))]
mod layer_norm_bias_tests {
    use super::*;
    use crate::driver;

    unsafe fn test_init() -> (CachingAllocator, CUstream) {
        driver::init().expect("init");
        let dev = driver::device_get(0).expect("dev");
        let ctx = driver::ctx_create(dev).expect("ctx");
        let stream = driver::stream_create().expect("stream");
        let _ = ctx;
        let alloc = CachingAllocator::new();
        (alloc, stream)
    }

    unsafe fn upload_f32(data: &[f32], stream: CUstream) -> *mut u8 {
        let nbytes = data.len() * 4;
        let ptr = driver::mem_alloc(nbytes).expect("alloc");
        driver::memcpy_htod_async(ptr, data.as_ptr() as *const u8, nbytes, stream).expect("htod");
        ptr
    }

    unsafe fn download_f32(ptr: *mut u8, count: usize, stream: CUstream) -> Vec<f32> {
        let mut host = vec![0.0f32; count];
        driver::memcpy_dtoh_async(host.as_mut_ptr() as *mut u8, ptr, count * 4, stream)
            .expect("dtoh");
        driver::stream_synchronize(stream).expect("sync");
        host
    }

    /// Test layer_norm_bias: y = (x - mean) / sqrt(var + eps) * weight + bias
    #[test]
    #[ignore] // requires CUDA GPU
    fn test_cuda_layer_norm_bias_f32() {
        unsafe {
            let (mut alloc, stream) = test_init();

            // 2 tokens, hidden_size=4
            let input_data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let weight_data = [1.0f32, 1.0, 1.0, 1.0];
            let bias_data = [0.1f32, 0.2, 0.3, 0.4];
            let eps = 1e-5f32;

            let input_ptr = upload_f32(&input_data, stream);
            let weight_ptr = upload_f32(&weight_data, stream);
            let bias_ptr = upload_f32(&bias_data, stream);
            driver::stream_synchronize(stream).expect("sync");

            let input = GpuTensor::new(input_ptr, &[2, 4], DType::F32);
            let weight = GpuTensor::new(weight_ptr, &[4], DType::F32);
            let bias = GpuTensor::new(bias_ptr, &[4], DType::F32);

            let out = layer_norm_bias(input, weight, bias, eps, &mut alloc, stream);

            let result = download_f32(out.raw_ptr(), 8, stream);

            // Row 0: [1,2,3,4], mean=2.5, var=1.25, std=sqrt(1.25+eps)
            let mean0 = 2.5f32;
            let var0 = ((1.0 - mean0).powi(2)
                + (2.0 - mean0).powi(2)
                + (3.0 - mean0).powi(2)
                + (4.0 - mean0).powi(2))
                / 4.0;
            let std0 = (var0 + eps).sqrt();
            for i in 0..4 {
                let expected = (input_data[i] - mean0) / std0 * weight_data[i] + bias_data[i];
                assert!(
                    (result[i] - expected).abs() < 1e-4,
                    "row 0 elem {i}: got {}, expected {expected}",
                    result[i]
                );
            }

            // Row 1: [5,6,7,8], mean=6.5
            let mean1 = 6.5f32;
            let var1 = ((5.0 - mean1).powi(2)
                + (6.0 - mean1).powi(2)
                + (7.0 - mean1).powi(2)
                + (8.0 - mean1).powi(2))
                / 4.0;
            let std1 = (var1 + eps).sqrt();
            for i in 0..4 {
                let expected = (input_data[4 + i] - mean1) / std1 * weight_data[i] + bias_data[i];
                assert!(
                    (result[4 + i] - expected).abs() < 1e-4,
                    "row 1 elem {i}: got {}, expected {expected}",
                    result[4 + i]
                );
            }

            driver::stream_destroy(stream).expect("destroy");
        }
    }

    /// Test fused_add_layer_norm_bias: residual += input; normed = LN(residual) * w + b
    #[test]
    #[ignore] // requires CUDA GPU
    fn test_cuda_fused_add_layer_norm_bias_f32() {
        unsafe {
            let (_alloc, stream) = test_init();

            // 1 token, hidden_size=4
            let input_data = [1.0f32, 2.0, 3.0, 4.0];
            let residual_data = [0.5f32, 0.5, 0.5, 0.5];
            let weight_data = [2.0f32, 2.0, 2.0, 2.0];
            let bias_data = [0.1f32, 0.2, 0.3, 0.4];
            let eps = 1e-5f32;

            let input_ptr = upload_f32(&input_data, stream);
            let residual_ptr = upload_f32(&residual_data, stream);
            let weight_ptr = upload_f32(&weight_data, stream);
            let bias_ptr = upload_f32(&bias_data, stream);
            driver::stream_synchronize(stream).expect("sync");

            let input = GpuTensor::new(input_ptr, &[1, 4], DType::F32);
            let residual = GpuTensor::new(residual_ptr, &[1, 4], DType::F32);
            let weight = GpuTensor::new(weight_ptr, &[4], DType::F32);
            let bias = GpuTensor::new(bias_ptr, &[4], DType::F32);

            let (normed, updated_res) =
                fused_add_layer_norm_bias_inplace(input, residual, weight, bias, eps, stream);

            // residual should be [1.5, 2.5, 3.5, 4.5]
            let res_result = download_f32(updated_res.raw_ptr(), 4, stream);
            for i in 0..4 {
                let expected = input_data[i] + residual_data[i];
                assert!(
                    (res_result[i] - expected).abs() < 1e-5,
                    "residual {i}: got {}, expected {expected}",
                    res_result[i]
                );
            }

            // normed should be LN([1.5, 2.5, 3.5, 4.5]) * 2.0 + bias
            let mean: f32 = (1.5 + 2.5 + 3.5 + 4.5) / 4.0;
            let var: f32 = ((1.5_f32 - mean).powi(2)
                + (2.5_f32 - mean).powi(2)
                + (3.5_f32 - mean).powi(2)
                + (4.5_f32 - mean).powi(2))
                / 4.0;
            let std = (var + eps).sqrt();
            let normed_result = download_f32(normed.raw_ptr(), 4, stream);
            let res_vals = [1.5f32, 2.5, 3.5, 4.5];
            for i in 0..4 {
                let expected = (res_vals[i] - mean) / std * weight_data[i] + bias_data[i];
                assert!(
                    (normed_result[i] - expected).abs() < 1e-4,
                    "normed {i}: got {}, expected {expected}",
                    normed_result[i]
                );
            }

            driver::stream_destroy(stream).expect("destroy");
        }
    }

    /// Test layer_norm_bias with zero bias gives pure LN (zero-mean, unit-variance).
    #[test]
    #[ignore] // requires CUDA GPU
    fn test_cuda_layer_norm_bias_zero_bias() {
        unsafe {
            let (mut alloc, stream) = test_init();

            let input_data: Vec<f32> = (0..8).map(|i| (i as f32) * 0.5 + 1.0).collect();
            let weight_data = vec![1.0f32; 8];
            let bias_data = vec![0.0f32; 8];

            let input_ptr = upload_f32(&input_data, stream);
            let weight_ptr = upload_f32(&weight_data, stream);
            let bias_ptr = upload_f32(&bias_data, stream);
            driver::stream_synchronize(stream).expect("sync");

            let input = GpuTensor::new(input_ptr, &[1, 8], DType::F32);
            let weight = GpuTensor::new(weight_ptr, &[8], DType::F32);
            let bias = GpuTensor::new(bias_ptr, &[8], DType::F32);

            let out = layer_norm_bias(input, weight, bias, 1e-5, &mut alloc, stream);
            let result = download_f32(out.raw_ptr(), 8, stream);

            let mean: f32 = result.iter().sum::<f32>() / 8.0;
            assert!(
                mean.abs() < 1e-5,
                "LN with zero bias should have ~zero mean, got {mean}"
            );

            let var: f32 = result.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / 8.0;
            assert!(
                (var - 1.0).abs() < 0.01,
                "LN output variance should be ~1.0, got {var}"
            );

            driver::stream_destroy(stream).expect("destroy");
        }
    }
}
