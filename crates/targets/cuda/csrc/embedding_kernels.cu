// SPDX-License-Identifier: Apache-2.0
// Embedding gather kernel with tensor-parallel vocab-shard masking.
//
// At tp=1 the shard covers the full vocab: vocab_offset = 0,
// vocab_per_rank = vocab_size; the mask never trips and behavior is
// identical to a plain gather. At tp>1 each rank's `weight` tensor
// is a [vocab_size/tp, hidden_size] slice; tokens outside this rank's
// [vocab_offset, vocab_offset + vocab_per_rank) range get zero-filled
// rows so the cross-rank AllReduce-sum produces exactly one
// embedding contribution per token (from the rank that owns it).
// Mirrors Python vLLM's VocabParallelEmbedding.forward_native — see
// `get_masked_input_and_mask` + `output_parallel.masked_fill_(mask, 0)`.

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>

template<typename T>
__global__ void embedding_gather_kernel(
    T* __restrict__ out,           // [num_tokens, hidden_size]
    const T* __restrict__ weight,  // [vocab_per_rank, hidden_size]
    const uint32_t* __restrict__ ids,  // [num_tokens] (global ids)
    int hidden_size,
    int num_tokens,
    uint32_t vocab_offset,
    uint32_t vocab_per_rank
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_tokens * hidden_size;
    if (idx >= total) return;

    int token = idx / hidden_size;
    int dim = idx % hidden_size;
    uint32_t id = ids[token];
    // Underflow on `id < vocab_offset` is fine — `local_id` becomes a
    // huge unsigned value that fails the `< vocab_per_rank` check.
    uint32_t local_id = id - vocab_offset;

    if (id < vocab_offset || local_id >= vocab_per_rank) {
        out[token * hidden_size + dim] = T(0);
    } else {
        out[token * hidden_size + dim] = weight[local_id * hidden_size + dim];
    }
}

// Vectorized version: 128-bit loads (8 x f16/bf16 or 4 x f32)
template<typename T, int VEC_SIZE>
__global__ void embedding_gather_vec_kernel(
    T* __restrict__ out,
    const T* __restrict__ weight,
    const uint32_t* __restrict__ ids,
    int hidden_size,
    int num_tokens,
    uint32_t vocab_offset,
    uint32_t vocab_per_rank
) {
    using VecT = typename std::conditional<sizeof(T) * VEC_SIZE == 16, int4,
                  typename std::conditional<sizeof(T) * VEC_SIZE == 8, int2, int>::type>::type;

    int vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int vec_hidden = hidden_size / VEC_SIZE;
    int total_vecs = num_tokens * vec_hidden;
    if (vec_idx >= total_vecs) return;

    int token = vec_idx / vec_hidden;
    int dim_vec = vec_idx % vec_hidden;
    uint32_t id = ids[token];
    uint32_t local_id = id - vocab_offset;

    VecT* o_vec = reinterpret_cast<VecT*>(out + token * hidden_size);
    if (id < vocab_offset || local_id >= vocab_per_rank) {
        // Zero-fill this rank's contribution for an out-of-range token.
        VecT zero;
        // Zero-init via memset on the vector type — works for int / int2 / int4.
        for (int b = 0; b < (int)sizeof(VecT) / 4; b++) {
            reinterpret_cast<int*>(&zero)[b] = 0;
        }
        o_vec[dim_vec] = zero;
    } else {
        const VecT* w_vec = reinterpret_cast<const VecT*>(weight + local_id * hidden_size);
        o_vec[dim_vec] = w_vec[dim_vec];
    }
}

#define LAUNCH_GATHER(T, VEC)                                                  \
    do {                                                                        \
        int vec_hidden = hidden_size / (VEC);                                   \
        int total = num_tokens * vec_hidden;                                    \
        int threads = 256;                                                      \
        int blocks = (total + threads - 1) / threads;                           \
        if (hidden_size % (VEC) == 0) {                                         \
            embedding_gather_vec_kernel<T, VEC><<<blocks, threads, 0, stream>>>( \
                (T*)out, (const T*)weight, ids, hidden_size, num_tokens,         \
                vocab_offset, vocab_per_rank);                                   \
        } else {                                                                \
            total = num_tokens * hidden_size;                                    \
            blocks = (total + threads - 1) / threads;                           \
            embedding_gather_kernel<T><<<blocks, threads, 0, stream>>>(          \
                (T*)out, (const T*)weight, ids, hidden_size, num_tokens,         \
                vocab_offset, vocab_per_rank);                                   \
        }                                                                       \
    } while (0)

extern "C" {

void embedding_gather_f16(
    void* out, const void* weight, const uint32_t* ids,
    int hidden_size, int num_tokens,
    uint32_t vocab_offset, uint32_t vocab_per_rank,
    cudaStream_t stream
) {
    LAUNCH_GATHER(__half, 8);  // 8 x f16 = 128 bits
}

void embedding_gather_bf16(
    void* out, const void* weight, const uint32_t* ids,
    int hidden_size, int num_tokens,
    uint32_t vocab_offset, uint32_t vocab_per_rank,
    cudaStream_t stream
) {
    LAUNCH_GATHER(__nv_bfloat16, 8);
}

void embedding_gather_f32(
    void* out, const void* weight, const uint32_t* ids,
    int hidden_size, int num_tokens,
    uint32_t vocab_offset, uint32_t vocab_per_rank,
    cudaStream_t stream
) {
    LAUNCH_GATHER(float, 4);  // 4 x f32 = 128 bits
}

}  // extern "C"

// ---------------------------------------------------------------------------
// update_decode_metadata: GPU-side metadata update for decode steps.
//
// In one kernel launch, this:
// 1. Increments positions[i] += 1
// 2. Computes slot_mapping[i] from new position + block_table
// 3. Increments seqused_k[i] += 1 (per-sequence K length)
//
// Replaces 3 CPU Vec builds + 3 H2D copies per decode step.
// ---------------------------------------------------------------------------

__global__ void update_decode_metadata_kernel(
    uint32_t* __restrict__ positions,       // [num_reqs] — incremented in place
    int64_t* __restrict__ slot_mapping,     // [num_reqs] — recomputed from new pos
    int32_t* __restrict__ seqused_k,        // [num_reqs] — per-seq K lengths, incremented
    const int32_t* __restrict__ block_table, // [num_reqs, max_blocks_per_seq]
    int num_reqs,
    int block_size,
    int max_blocks_per_seq
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_reqs) return;

    // 1. Increment position.
    uint32_t new_pos = positions[i] + 1;
    positions[i] = new_pos;

    // 2. Compute slot_mapping from new position + block_table.
    int block_idx = new_pos / block_size;
    int offset = new_pos % block_size;
    if (block_idx < max_blocks_per_seq) {
        int32_t block_id = block_table[i * max_blocks_per_seq + block_idx];
        slot_mapping[i] = (int64_t)(block_id * block_size + offset);
    } else {
        slot_mapping[i] = -1;
    }

    // 3. Increment seqused_k[i] (per-sequence K length).
    seqused_k[i] += 1;
}

extern "C" {

void update_decode_metadata(
    uint32_t* positions,
    int64_t* slot_mapping,
    int32_t* seqused_k,
    const int32_t* block_table,
    int num_reqs,
    int block_size,
    int max_blocks_per_seq,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (num_reqs + threads - 1) / threads;
    update_decode_metadata_kernel<<<blocks, threads, 0, stream>>>(
        positions, slot_mapping, seqused_k, block_table,
        num_reqs, block_size, max_blocks_per_seq);
}

}  // extern "C"

// ---------------------------------------------------------------------------
// Prefix sum of seqused_k → cu_seqlens_k on GPU.
// Single thread — batch sizes ≤512, so this is sub-microsecond.
// ---------------------------------------------------------------------------

__global__ void prefix_sum_seqused_k_kernel(
    const int32_t* __restrict__ seqused_k,
    int32_t* __restrict__ cu_seqlens_k,
    int num_reqs
) {
    if (threadIdx.x != 0) return;
    cu_seqlens_k[0] = 0;
    for (int i = 0; i < num_reqs; i++) {
        cu_seqlens_k[i + 1] = cu_seqlens_k[i] + seqused_k[i];
    }
}

extern "C" {

void prefix_sum_seqused_k_gpu(
    const int32_t* seqused_k,
    int32_t* cu_seqlens_k,
    int num_reqs,
    cudaStream_t stream
) {
    if (num_reqs == 0) return;
    prefix_sum_seqused_k_kernel<<<1, 1, 0, stream>>>(
        seqused_k, cu_seqlens_k, num_reqs);
}

}  // extern "C"

// ---------------------------------------------------------------------------
// Split fused QKV: [num_tokens, q_size + 2*kv_size] → Q, K, V contiguous
// One thread per element.
// ---------------------------------------------------------------------------

template<typename T>
__global__ void split_qkv_kernel(
    T* __restrict__ q_out,     // [num_tokens, q_size]
    T* __restrict__ k_out,     // [num_tokens, kv_size]
    T* __restrict__ v_out,     // [num_tokens, kv_size]
    const T* __restrict__ qkv, // [num_tokens, total_dim]
    int q_size,
    int kv_size,
    int total_dim,
    int num_tokens
) {
    // Grid covers all elements across Q, K, V outputs.
    // We map each thread to one element in one of the three outputs.
    int total_out = num_tokens * (q_size + 2 * kv_size);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_out) return;

    int token = idx / (q_size + 2 * kv_size);
    int within = idx % (q_size + 2 * kv_size);

    const T* src_row = qkv + token * total_dim;

    if (within < q_size) {
        q_out[token * q_size + within] = src_row[within];
    } else if (within < q_size + kv_size) {
        int k_offset = within - q_size;
        k_out[token * kv_size + k_offset] = src_row[q_size + k_offset];
    } else {
        int v_offset = within - q_size - kv_size;
        v_out[token * kv_size + v_offset] = src_row[q_size + kv_size + v_offset];
    }
}

#define LAUNCH_SPLIT_QKV(T)                                                    \
    do {                                                                        \
        int total = num_tokens * (q_size + 2 * kv_size);                       \
        int threads = 256;                                                      \
        int blocks = (total + threads - 1) / threads;                           \
        split_qkv_kernel<T><<<blocks, threads, 0, stream>>>(                   \
            (T*)q_out, (T*)k_out, (T*)v_out, (const T*)qkv,                   \
            q_size, kv_size, total_dim, num_tokens);                           \
    } while (0)

extern "C" {

void split_qkv_f16(
    void* q_out, void* k_out, void* v_out, const void* qkv,
    int q_size, int kv_size, int total_dim, int num_tokens,
    cudaStream_t stream
) {
    LAUNCH_SPLIT_QKV(__half);
}

void split_qkv_bf16(
    void* q_out, void* k_out, void* v_out, const void* qkv,
    int q_size, int kv_size, int total_dim, int num_tokens,
    cudaStream_t stream
) {
    LAUNCH_SPLIT_QKV(__nv_bfloat16);
}

void split_qkv_f32(
    void* q_out, void* k_out, void* v_out, const void* qkv,
    int q_size, int kv_size, int total_dim, int num_tokens,
    cudaStream_t stream
) {
    LAUNCH_SPLIT_QKV(float);
}

}  // extern "C"

// ---------------------------------------------------------------------------
// Bias add: out[i,j] += bias[j]  for out [M, N] and bias [N]
// ---------------------------------------------------------------------------

template<typename T>
__global__ void bias_add_kernel(
    T* __restrict__ out,         // [M, N]
    const T* __restrict__ bias,  // [N]
    int M, int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;
    int j = idx % N;
    out[idx] = __hadd(out[idx], bias[j]);
}

template<>
__global__ void bias_add_kernel<float>(
    float* __restrict__ out,
    const float* __restrict__ bias,
    int M, int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;
    int j = idx % N;
    out[idx] = out[idx] + bias[j];
}

extern "C" {

void bias_add_f16(void* out, const void* bias, int M, int N, cudaStream_t stream) {
    int total = M * N;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    bias_add_kernel<__half><<<blocks, threads, 0, stream>>>(
        (__half*)out, (const __half*)bias, M, N);
}

void bias_add_bf16(void* out, const void* bias, int M, int N, cudaStream_t stream) {
    int total = M * N;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    bias_add_kernel<__nv_bfloat16><<<blocks, threads, 0, stream>>>(
        (__nv_bfloat16*)out, (const __nv_bfloat16*)bias, M, N);
}

void bias_add_f32(void* out, const void* bias, int M, int N, cudaStream_t stream) {
    int total = M * N;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    bias_add_kernel<float><<<blocks, threads, 0, stream>>>(
        (float*)out, (const float*)bias, M, N);
}

}  // extern "C"
