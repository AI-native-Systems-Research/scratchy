// Thin FFI shim for FlashAttention-2.
//
// Sets up Flash_fwd_params identically to vllm-flash-attn/kernels/flash_api.cu
// (run_mha and run_mha_paged), but takes raw GPU pointers instead of at::Tensor.
//
// IMPORTANT: We do NOT include flash_fwd_launch_template.h here. That header
// contains full template definitions with <<<kernel launch>>> syntax, which
// would cause nvcc to instantiate ALL kernel variants in this TU (~28 min).
// Instead we include only flash.h (which has the template DECLARATIONS) and
// static_switch.h (dispatch macros). The template bodies are compiled in the
// per-headdim .cu files; we resolve them at link time.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>

#include "namespace_config.h"
#include "flash.h"
#include "static_switch.h"

#include <cutlass/numeric_types.h>

using namespace FLASH_NAMESPACE;

// ---------------------------------------------------------------------------
// Dispatch — matches vllm-flash-attn/csrc/flash_attn/flash_api.cpp
// ---------------------------------------------------------------------------

static void run_mha_fwd(Flash_fwd_params &params, cudaStream_t stream, bool force_split_kernel = false) {
    FP16_SWITCH(!params.is_bf16, [&] {
        HEADDIM_SWITCH(params.d, [&] {
            BOOL_SWITCH(params.is_causal, Is_causal, [&] {
                if (params.num_splits <= 1 && !force_split_kernel) {
                    run_mha_fwd_<elem_type, kHeadDim, Is_causal>(params, stream);
                } else {
                    run_mha_fwd_splitkv_dispatch<elem_type, kHeadDim, Is_causal>(params, stream);
                }
            });
        });
    });
}

static inline int round_up(int x, int m) { return (x + m - 1) / m * m; }

// ---------------------------------------------------------------------------
// Q transpose for seqlenq_ngroups_swapped.
//
// Transposes Q from [B, H, D] (varlen with seqlen_q=1) to [B*ngroups, Hk, D].
// Python: q.reshape(B, Hk, ngroups, D).transpose(1,2).reshape(B*ngroups, Hk, D)
// This is equivalent to: for head h_orig = hk*ngroups+g, copy to position
// (b*ngroups+g)*Hk*D + hk*D.
// ---------------------------------------------------------------------------

__global__ void ngroups_transpose_kernel(
    const uint16_t* __restrict__ src,  // [B, H, D]
    uint16_t* __restrict__ dst,        // [B*ngroups, Hk, D]
    int B, int H, int Hk, int ngroups, int D
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * H * D;
    if (idx >= total) return;

    int d = idx % D;
    int h = (idx / D) % H;
    int b = idx / (H * D);

    int hk = h / ngroups;
    int g  = h % ngroups;

    int dst_idx = (b * ngroups + g) * Hk * D + hk * D + d;
    dst[dst_idx] = src[idx];
}

// Reverse: from [B*ngroups, Hk, D] back to [B, H, D]
__global__ void ngroups_untranspose_kernel(
    const uint16_t* __restrict__ src,  // [B*ngroups, Hk, D]
    uint16_t* __restrict__ dst,        // [B, H, D]
    int B, int H, int Hk, int ngroups, int D
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * H * D;
    if (idx >= total) return;

    int d = idx % D;
    int h = (idx / D) % H;
    int b = idx / (H * D);

    int hk = h / ngroups;
    int g  = h % ngroups;

    int src_idx = (b * ngroups + g) * Hk * D + hk * D + d;
    dst[idx] = src[src_idx];
}

extern "C" void ngroups_transpose_q(
    const void *src, void *dst,
    int batch_size, int num_heads, int num_heads_k, int ngroups, int head_dim,
    cudaStream_t stream
) {
    int total = batch_size * num_heads * head_dim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    ngroups_transpose_kernel<<<blocks, threads, 0, stream>>>(
        (const uint16_t*)src, (uint16_t*)dst,
        batch_size, num_heads, num_heads_k, ngroups, head_dim
    );
}

extern "C" void ngroups_untranspose_o(
    const void *src, void *dst,
    int batch_size, int num_heads, int num_heads_k, int ngroups, int head_dim,
    cudaStream_t stream
) {
    int total = batch_size * num_heads * head_dim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    ngroups_untranspose_kernel<<<blocks, threads, 0, stream>>>(
        (const uint16_t*)src, (uint16_t*)dst,
        batch_size, num_heads, num_heads_k, ngroups, head_dim
    );
}

// ---------------------------------------------------------------------------
// mha_varlen_fwd — unified entry point for both contiguous and paged KV.
//
// Parameter setup matches vllm-flash-attn/kernels/flash_api.cu exactly.
// For paged KV: block_table != nullptr, pass seqused_k for actual K lengths.
// For contiguous: block_table == nullptr.
// ---------------------------------------------------------------------------

extern "C" void mha_varlen_fwd(
    // Tensor pointers (device)
    void *q_ptr,
    void *k_ptr,
    void *v_ptr,
    void *out_ptr,
    void *softmax_lse_ptr,

    // Sequence metadata (device)
    int32_t *cu_seqlens_q,
    int32_t *cu_seqlens_k,
    int32_t *seqused_k,       // [batch_size] actual K lengths (paged), or nullptr

    // Paged KV cache (device), or nullptr for non-paged
    int32_t *block_table,
    int32_t block_table_batch_stride,

    // Dimensions
    int32_t batch_size,
    int32_t max_seqlen_q,
    int32_t max_seqlen_k,
    int32_t num_heads,
    int32_t num_heads_k,
    int32_t head_size,
    int32_t page_block_size,

    // Q strides (elements, not bytes)
    int64_t q_row_stride,
    int64_t q_head_stride,
    // K/V strides
    int64_t k_batch_stride,
    int64_t k_row_stride,
    int64_t k_head_stride,
    // O strides
    int64_t o_row_stride,
    int64_t o_head_stride,

    // Config
    float softmax_scale,
    int32_t is_causal,
    int32_t window_size_left,
    int32_t window_size_right,
    float softcap,
    int32_t is_bf16,
    int32_t num_splits,

    // Split-K accumulators (required when num_splits > 1)
    void *softmax_lse_accum_ptr,
    void *out_accum_ptr,

    // seqlenq_ngroups_swapped flag (set by Rust caller)
    int32_t seqlenq_ngroups_swapped,

    // total number of Q tokens (Python: q.sizes()[0])
    int32_t total_q,

    // Spans: fused RoPE for cached K reads.
    // When rotate_cached_k != 0 and rotary_cos/sin are non-null,
    // apply rotary embedding to K in shared memory during attention.
    void *rotary_cos_ptr,
    void *rotary_sin_ptr,
    int32_t rotary_dim,
    int32_t rotate_cached_k,
    int32_t is_rotary_interleaved,

    // Per-block rotation flags: block_unrotated_flags[phys_block] != 0 means
    // K is stored unrotated and needs RoPE on read. nullptr = use rotate_cached_k.
    const uint8_t *block_unrotated_flags,

    cudaStream_t stream
) {
    // --- Parameter setup matches vllm-flash-attn flash_api.cu exactly ---

    const int head_size_rounded = round_up(head_size, head_size <= 128 ? 32 : 64);
    const int seqlen_q_rounded = round_up(max_seqlen_q, 128);
    const int seqlen_k_rounded = round_up(max_seqlen_k, 128);

    Flash_fwd_params params = {};

    params.q_ptr = q_ptr;
    params.k_ptr = k_ptr;
    params.v_ptr = v_ptr;
    params.o_ptr = out_ptr;

    params.softmax_lse_ptr = softmax_lse_ptr;
    params.alibi_slopes_ptr = nullptr;

    // When seqlenq_ngroups_swapped, cu_seqlens_q is nullptr and we use
    // batch strides instead — matching Python's set_params_fprop exactly.
    if (seqlenq_ngroups_swapped && cu_seqlens_q == nullptr) {
        // q is [B*ngroups, Hk, D] contiguous. q.stride(0) = Hk * D = q_row_stride.
        // Python multiplies by seqlen_q (= ngroups).
        params.q_batch_stride = q_row_stride * max_seqlen_q;
        params.o_batch_stride = o_row_stride * max_seqlen_q;
    } else {
        params.q_batch_stride = 0;
        params.o_batch_stride = 0;
    }
    params.k_batch_stride = k_batch_stride;
    params.v_batch_stride = k_batch_stride;
    params.alibi_slopes_batch_stride = 0;

    params.q_row_stride = q_row_stride;
    params.k_row_stride = k_row_stride;
    params.v_row_stride = k_row_stride;
    params.o_row_stride = o_row_stride;
    params.q_head_stride = q_head_stride;
    params.k_head_stride = k_head_stride;
    params.v_head_stride = k_head_stride;
    params.o_head_stride = o_head_stride;

    params.b = batch_size;
    params.h = num_heads;
    params.h_k = num_heads_k;
    params.h_h_k_ratio = num_heads / num_heads_k;
    params.seqlen_q = max_seqlen_q;
    params.seqlen_k = max_seqlen_k;
    params.seqlen_q_rounded = seqlen_q_rounded;
    params.seqlen_k_rounded = seqlen_k_rounded;
    params.d = head_size;
    params.d_rounded = head_size_rounded;

    if (softcap > 0.0f) {
        params.softcap = softmax_scale / softcap;
        params.scale_softmax = softcap;
        params.scale_softmax_log2 = softcap * (float)M_LOG2E;
    } else {
        params.softcap = 0.0f;
        params.scale_softmax = softmax_scale;
        params.scale_softmax_log2 = softmax_scale * (float)M_LOG2E;
    }

    params.p_dropout = 1.0f;
    params.p_dropout_in_uint8_t = uint8_t(std::floor(1.0f * 255.0));
    params.rp_dropout = 1.0f;
    params.scale_softmax_rp_dropout = params.scale_softmax;
    params.is_bf16 = is_bf16;

    params.cu_seqlens_q = cu_seqlens_q;
    params.cu_seqlens_k = cu_seqlens_k;
    params.seqused_k = seqused_k;
    params.p_ptr = nullptr;

    // Causal / window — matching set_params_fprop lines 139-144 exactly.
    // is_causal is recomputed from window sizes (not from the input flag).
    params.is_causal = window_size_left < 0 && window_size_right == 0;

    int wsl = window_size_left;
    int wsr = window_size_right;
    if (wsl < 0 && wsr >= 0) { wsl = max_seqlen_k; }
    if (wsl >= 0 && wsr < 0) { wsr = max_seqlen_k; }
    params.window_size_left = wsl;
    params.window_size_right = wsr;

    params.is_seqlens_k_cumulative = true;
    params.unpadded_lse = 1;
    params.num_splits = num_splits;
    params.total_q = total_q;
    params.seqlenq_ngroups_swapped = seqlenq_ngroups_swapped;

    // Paged KV
    const bool paged = (block_table != nullptr);
    if (paged) {
        params.block_table = block_table;
        params.block_table_batch_stride = block_table_batch_stride;
    }
    params.page_block_size = page_block_size;

    // Spans: fused RoPE for cached K reads.
    // rotary_cos_ptr points to the combined [max_pos, rotary_dim] cos|sin cache.
    // rotate_k_smem_contiguous reads cos/sin from this combined layout directly.
    params.rotary_cos_ptr = rotary_cos_ptr;
    params.rotary_sin_ptr = nullptr;  // unused by rotate_k_smem_contiguous
    params.rotary_dim = rotary_dim;
    params.rotate_cached_k = (rotate_cached_k != 0);
    params.is_rotary_interleaved = (is_rotary_interleaved != 0);
    params.block_unrotated_flags = block_unrotated_flags;

    // Split-K accumulators (for num_splits > 1)
    params.softmax_lseaccum_ptr = softmax_lse_accum_ptr;
    params.oaccum_ptr = out_accum_ptr;

    // Launch — paged KV MUST use the splitkv kernel (the standard kernel
    // compute_attn_1rowblock has no block_table support).
    if (max_seqlen_k > 0) {
        run_mha_fwd(params, stream, /*force_split_kernel=*/paged);
    }
}
