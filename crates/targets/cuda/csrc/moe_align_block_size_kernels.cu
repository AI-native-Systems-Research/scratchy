/*
 * MoE align block size kernel, ported from Python vLLM's
 * csrc/moe/moe_align_sum_kernels.cu.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Stripped PyTorch/ATen dependencies, added extern "C" launcher for Rust FFI.
 * Two paths:
 * - Small batch (numel < 1024, ≤64 experts): single threadblock, shared-memory counting
 * - Large batch: alignment kernel (2 TBs) + separate sort kernel
 */

#include <cub/cub.cuh>
#include <stdint.h>

#define CEILDIV(x, y) (((x) + (y) - 1) / (y))

namespace vllm {
namespace moe {

// =========================================================================
// Small-batch kernel: single threadblock, shared-memory counting + sorting.
// Handles ≤1024 tokens and ≤64 experts.
// =========================================================================

template <int32_t fill_threads>
__global__ void moe_align_block_size_small_batch_kernel(
    const int32_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids,
    int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ total_tokens_post_pad,
    int32_t num_experts,
    int32_t block_size,
    int32_t numel,
    int32_t max_num_tokens_padded)
{
    const int32_t max_num_m_blocks = CEILDIV(max_num_tokens_padded, block_size);

    // First fill_threads threads initialize sorted_token_ids with sentinel.
    if (threadIdx.x < fill_threads) {
        for (int32_t it = threadIdx.x; it < max_num_tokens_padded; it += fill_threads) {
            sorted_token_ids[it] = numel;
        }
        __syncthreads();
        __syncthreads();
        __syncthreads();
        return;
    }

    const int32_t tid = threadIdx.x - fill_threads;
    const int32_t stride = blockDim.x - fill_threads;

    extern __shared__ int32_t shared_mem[];
    int32_t* cumsum = shared_mem;
    int32_t* tokens_cnts = shared_mem + num_experts + 1;

    // Initialize per-thread expert counts.
    for (int i = 0; i < num_experts; ++i) {
        tokens_cnts[(tid + 1) * num_experts + i] = 0;
    }

    // Count tokens per expert per thread.
    for (int32_t i = tid; i < numel; i += stride) {
        int32_t expert_id = topk_ids[i];
        if (expert_id >= 0 && expert_id < num_experts) {
            tokens_cnts[(tid + 1) * num_experts + expert_id] += 1;
        }
    }

    __syncthreads();

    // Reduce per-thread counts into cumulative per-expert counts.
    if (tid < num_experts) {
        tokens_cnts[tid] = 0;
        for (int i = 1; i <= stride; ++i) {
            tokens_cnts[i * num_experts + tid] +=
                tokens_cnts[(i - 1) * num_experts + tid];
        }
    }

    __syncthreads();

    // Compute prefix sum over padded expert counts.
    if (tid == 0) {
        cumsum[0] = 0;
        for (int i = 1; i <= num_experts; ++i) {
            cumsum[i] =
                cumsum[i - 1] +
                CEILDIV(tokens_cnts[stride * num_experts + i - 1], block_size) *
                    block_size;
        }
        *total_tokens_post_pad = cumsum[num_experts];
    }

    __syncthreads();

    // Fill expert_ids.
    if (tid < num_experts) {
        for (int i = cumsum[tid]; i < cumsum[tid + 1]; i += block_size) {
            expert_ids[i / block_size] = tid;
        }
    }

    // Fill remaining expert_ids with -1.
    const int32_t fill_start = cumsum[num_experts] / block_size + tid;
    for (int32_t i = fill_start; i < max_num_m_blocks; i += stride) {
        expert_ids[i] = -1;
    }

    // Sort tokens into expert-grouped order.
    for (int32_t i = tid; i < numel; i += stride) {
        int32_t expert_id = topk_ids[i];
        if (expert_id >= 0 && expert_id < num_experts) {
            int32_t rank = tokens_cnts[tid * num_experts + expert_id] + cumsum[expert_id];
            sorted_token_ids[rank] = i;
            ++tokens_cnts[tid * num_experts + expert_id];
        }
    }
}

// =========================================================================
// Large-batch path: two kernels.
// Kernel 1: count + align + fill expert_ids (2 threadblocks).
// Kernel 2: sort tokens using atomicAdd on cumsum buffer.
// =========================================================================

// Kernel 1a (odd blocks): fill sorted_token_ids with sentinel.
// Kernel 1b (even blocks): count, prefix sum, fill expert_ids, write cumsum.
__global__ void moe_align_kernel(
    const int32_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids,
    int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ total_tokens_post_pad,
    int32_t* __restrict__ cumsum_buffer,  // [num_experts + 1] for sort kernel
    int32_t num_experts,
    int32_t padded_num_experts,
    int32_t experts_per_warp,
    int32_t block_size,
    int32_t numel,
    int32_t max_num_tokens_padded)
{
    const int32_t max_num_m_blocks = CEILDIV(max_num_tokens_padded, block_size);

    extern __shared__ int32_t shared_counts[];

    // Odd blocks fill sorted_token_ids with sentinel.
    if (blockIdx.x % 2) {
        for (int32_t it = threadIdx.x; it < max_num_tokens_padded; it += blockDim.x) {
            sorted_token_ids[it] = numel;
        }
        return;
    }

    const int warp_id = threadIdx.x / 32;
    const int my_expert_start = warp_id * experts_per_warp;

    for (int i = 0; i < experts_per_warp; ++i) {
        if (my_expert_start + i < padded_num_experts) {
            shared_counts[warp_id * experts_per_warp + i] = 0;
        }
    }

    __syncthreads();

    // Count tokens per expert using atomics in shared memory.
    for (int32_t i = threadIdx.x; i < numel; i += blockDim.x) {
        int32_t expert_id = topk_ids[i];
        if (expert_id >= 0 && expert_id < num_experts) {
            int warp_idx = expert_id / experts_per_warp;
            int expert_offset = expert_id % experts_per_warp;
            atomicAdd(&shared_counts[warp_idx * experts_per_warp + expert_offset], 1);
        }
    }

    __syncthreads();

    // Prefix sum over padded expert counts using CUB BlockScan.
    using BlockScan = cub::BlockScan<int32_t, 1024>;
    __shared__ typename BlockScan::TempStorage temp_storage;

    int expert_count = 0;
    int expert_id = threadIdx.x;
    if (expert_id < num_experts) {
        int warp_idx = expert_id / experts_per_warp;
        int expert_offset = expert_id % experts_per_warp;
        expert_count = shared_counts[warp_idx * experts_per_warp + expert_offset];
        expert_count = CEILDIV(expert_count, block_size) * block_size;
    }

    int cumsum_val;
    BlockScan(temp_storage).ExclusiveSum(expert_count, cumsum_val);

    // Write cumsum to global memory for the sort kernel.
    if (expert_id <= num_experts) {
        cumsum_buffer[expert_id] = cumsum_val;
    }

    if (expert_id == num_experts) {
        *total_tokens_post_pad = cumsum_val;
    }

    __syncthreads();

    // Fill expert_ids.
    if (threadIdx.x < num_experts) {
        for (int i = cumsum_buffer[threadIdx.x]; i < cumsum_buffer[threadIdx.x + 1]; i += block_size) {
            expert_ids[i / block_size] = threadIdx.x;
        }
    }

    // Fill remaining with -1.
    const int32_t fill_start = cumsum_buffer[num_experts] / block_size + threadIdx.x;
    for (int32_t i = fill_start; i < max_num_m_blocks; i += blockDim.x) {
        expert_ids[i] = -1;
    }
}

// Kernel 2: sort tokens into sorted_token_ids using atomicAdd on cumsum_buffer.
// Runs after moe_align_kernel completes. Uses cumsum_buffer as write offsets.
__global__ void moe_sort_tokens_kernel(
    const int32_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids,
    int32_t* __restrict__ cumsum_buffer,  // atomically incremented
    int32_t num_experts,
    int32_t numel)
{
    const int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = blockDim.x * gridDim.x;

    for (int32_t i = tid; i < numel; i += stride) {
        int32_t expert_id = topk_ids[i];
        if (expert_id >= 0 && expert_id < num_experts) {
            int32_t rank = atomicAdd(&cumsum_buffer[expert_id], 1);
            sorted_token_ids[rank] = i;
        }
    }
}

} // namespace moe
} // namespace vllm

// =====================================================================
// extern "C" launcher for Rust FFI
// =====================================================================

extern "C" void moe_align_block_size_i32(
    const int32_t* topk_ids,
    int32_t* sorted_token_ids,
    int32_t* expert_ids,
    int32_t* total_tokens_post_pad,
    int num_experts,
    int block_size,
    int numel,
    int max_num_tokens_padded,
    cudaStream_t stream)
{
    // Small-batch: single threadblock, shared-memory counting + sorting.
    // Threshold: numel < 1024 AND shared memory fits (stride+1)*num_experts ints.
    // With 1024 threads, fill_threads=256, stride=768: shared = (769*E + E+1)*4 bytes.
    // For E=64: 197KB — too much. For E=8: 24.6KB — fine.
    // So: use small-batch for numel < 1024 AND num_experts <= 32.
    bool use_small = (numel < 1024 && num_experts <= 32);
    if (use_small) {
        constexpr int fill_threads = 256;
        int total_threads = 1024;
        int stride_count = total_threads - fill_threads;
        int shared_bytes = ((stride_count + 1) * num_experts + num_experts + 1) * sizeof(int32_t);
        // Request extended shared memory if needed (default limit is 48 KB).
        if (shared_bytes > 48 * 1024) {
            cudaFuncSetAttribute(
                vllm::moe::moe_align_block_size_small_batch_kernel<fill_threads>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                shared_bytes);
        }
        vllm::moe::moe_align_block_size_small_batch_kernel<fill_threads>
            <<<1, total_threads, shared_bytes, stream>>>(
                topk_ids, sorted_token_ids, expert_ids, total_tokens_post_pad,
                num_experts, block_size, numel, max_num_tokens_padded);
    } else {
        // Large-batch: alignment kernel + sort kernel.
        // Allocate cumsum_buffer in the padding space at end of sorted_token_ids.
        // sorted_token_ids is [max_num_tokens_padded] ints — we need (num_experts+1)
        // extra ints for cumsum. The buffer is large enough (max_num_tokens_padded
        // = numel + num_experts * block_size, which is much larger than num_experts+1).
        // Use the end of sorted_token_ids as scratch for cumsum.
        int32_t* cumsum_buffer = sorted_token_ids + max_num_tokens_padded - (num_experts + 1);

        int padded_num_experts = CEILDIV(num_experts, 32) * 32;
        int experts_per_warp = padded_num_experts / (1024 / 32);
        if (experts_per_warp < 1) experts_per_warp = 1;
        padded_num_experts = experts_per_warp * (1024 / 32);
        int shared_bytes = padded_num_experts * sizeof(int32_t);

        // Kernel 1: align + fill expert_ids + init sorted_token_ids.
        vllm::moe::moe_align_kernel
            <<<2, 1024, shared_bytes, stream>>>(
                topk_ids, sorted_token_ids, expert_ids, total_tokens_post_pad,
                cumsum_buffer, num_experts, padded_num_experts, experts_per_warp,
                block_size, numel, max_num_tokens_padded);

        // Kernel 2: sort tokens into sorted_token_ids.
        int sort_threads = 256;
        int sort_blocks = CEILDIV(numel, sort_threads);
        if (sort_blocks > 128) sort_blocks = 128;
        vllm::moe::moe_sort_tokens_kernel
            <<<sort_blocks, sort_threads, 0, stream>>>(
                topk_ids, sorted_token_ids, cumsum_buffer, num_experts, numel);
    }
}
