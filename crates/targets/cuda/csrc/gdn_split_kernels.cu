// SPDX-License-Identifier: Apache-2.0
// GDN QKVZ+BA split and concat kernels for Qwen3-Next.
//
// The QKVZ projection output has grouped layout:
//   [T, num_k_heads * (head_k + head_k + v_per_k*head_v + v_per_k*head_v)]
// where each group of size `per_group` contains: Q_k, K_k, V_v0..V_vn, Z_v0..Z_vn
//
// This kernel splits into flat Q, K, V, Z tensors and also produces
// the concatenated Q||K||V for conv1d input.
//
// BA projection output: [T, num_k_heads * 2 * v_per_k]
// Split into B[T, num_v_heads] and A[T, num_v_heads].

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// ---------------------------------------------------------------------------
// Device helpers (outside extern "C" — templates need C++ linkage)
// ---------------------------------------------------------------------------

template <typename T>
__device__ __forceinline__ float to_f32(T v);

template <>
__device__ __forceinline__ float to_f32<float>(float v) { return v; }

template <>
__device__ __forceinline__ float to_f32<__half>(__half v) { return __half2float(v); }

template <>
__device__ __forceinline__ float to_f32<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }

template <typename InT>
__global__ void gdn_qkvz_split_kernel(
    const InT* __restrict__ qkvz,   // [T, qkvz_dim]
    const InT* __restrict__ ba,     // [T, 2*num_v_heads]
    float* __restrict__ out_q,      // [T, key_dim]
    float* __restrict__ out_k,      // [T, key_dim]
    float* __restrict__ out_v,      // [T, value_dim]
    float* __restrict__ out_z,      // [T, value_dim]
    float* __restrict__ out_a,      // [T, num_v_heads]
    float* __restrict__ out_b,      // [T, num_v_heads]
    float* __restrict__ out_mixed,  // [T, conv_dim] = [T, 2*key_dim + value_dim]
    int num_k_heads,
    int num_v_heads,
    int head_k_dim,
    int head_v_dim,
    int v_per_k,
    int key_dim,        // num_k_heads * head_k_dim
    int value_dim,      // num_v_heads * head_v_dim
    int qkvz_dim,       // 2*key_dim + 2*value_dim
    int conv_dim)       // 2*key_dim + value_dim
{
    const int t = blockIdx.x;
    const int per_group = head_k_dim + head_k_dim + v_per_k * head_v_dim + v_per_k * head_v_dim;
    const InT* qkvz_row = qkvz + t * qkvz_dim;
    const InT* ba_row = ba + t * 2 * num_v_heads;
    float* q_row = out_q + t * key_dim;
    float* k_row = out_k + t * key_dim;
    float* v_row = out_v + t * value_dim;
    float* z_row = out_z + t * value_dim;
    float* a_row = out_a + t * num_v_heads;
    float* b_row = out_b + t * num_v_heads;
    float* mixed_row = out_mixed + t * conv_dim;

    // Process QKVZ: iterate over flat output indices.
    for (int i = threadIdx.x; i < qkvz_dim; i += blockDim.x) {
        float val = to_f32(qkvz_row[i]);

        int g = i / per_group;
        int off = i % per_group;

        if (off < head_k_dim) {
            int q_idx = g * head_k_dim + off;
            q_row[q_idx] = val;
            mixed_row[q_idx] = val;
        } else if (off < 2 * head_k_dim) {
            int k_idx = g * head_k_dim + (off - head_k_dim);
            k_row[k_idx] = val;
            mixed_row[key_dim + k_idx] = val;
        } else {
            int vz_off = off - 2 * head_k_dim;
            int total_v = v_per_k * head_v_dim;
            if (vz_off < total_v) {
                int vv = vz_off / head_v_dim;
                int d = vz_off % head_v_dim;
                int vh = g * v_per_k + vv;
                int v_idx = vh * head_v_dim + d;
                v_row[v_idx] = val;
                mixed_row[2 * key_dim + v_idx] = val;
            } else {
                int zz_off = vz_off - total_v;
                int vv = zz_off / head_v_dim;
                int d = zz_off % head_v_dim;
                int vh = g * v_per_k + vv;
                z_row[vh * head_v_dim + d] = val;
            }
        }
    }

    // Process BA: grouped layout [num_k_heads, 2*v_per_k] -> B[num_v_heads], A[num_v_heads]
    for (int i = threadIdx.x; i < 2 * num_v_heads; i += blockDim.x) {
        float val = to_f32(ba_row[i]);
        int g = i / (2 * v_per_k);
        int off = i % (2 * v_per_k);
        int vh = g * v_per_k + (off % v_per_k);
        if (off < v_per_k) {
            b_row[vh] = val;
        } else {
            a_row[vh] = val;
        }
    }
}

__global__ void gdn_conv_split_kernel(
    const float* __restrict__ conv_out,  // [T, conv_dim]
    float* __restrict__ q,               // [T, key_dim]
    float* __restrict__ k,               // [T, key_dim]
    float* __restrict__ v,               // [T, value_dim]
    int key_dim,
    int value_dim,
    int conv_dim)
{
    const int t = blockIdx.x;
    const float* row = conv_out + t * conv_dim;
    float* q_row = q + t * key_dim;
    float* k_row = k + t * key_dim;
    float* v_row = v + t * value_dim;

    for (int i = threadIdx.x; i < key_dim; i += blockDim.x) {
        q_row[i] = row[i];
    }
    for (int i = threadIdx.x; i < key_dim; i += blockDim.x) {
        k_row[i] = row[key_dim + i];
    }
    for (int i = threadIdx.x; i < value_dim; i += blockDim.x) {
        v_row[i] = row[2 * key_dim + i];
    }
}

// ---------------------------------------------------------------------------
// C entry points
// ---------------------------------------------------------------------------

extern "C" {

void gdn_qkvz_split_bf16(
    const void* qkvz, const void* ba,
    float* q, float* k, float* v, float* z,
    float* a, float* b, float* mixed,
    int num_tokens, int num_k_heads, int num_v_heads,
    int head_k_dim, int head_v_dim, int v_per_k,
    int key_dim, int value_dim, int qkvz_dim, int conv_dim,
    cudaStream_t stream)
{
    int max_dim = qkvz_dim;
    if (2 * num_v_heads > max_dim) max_dim = 2 * num_v_heads;
    int threads = (max_dim < 1024) ? max_dim : 1024;
    gdn_qkvz_split_kernel<__nv_bfloat16><<<num_tokens, threads, 0, stream>>>(
        (const __nv_bfloat16*)qkvz, (const __nv_bfloat16*)ba,
        q, k, v, z, a, b, mixed,
        num_k_heads, num_v_heads, head_k_dim, head_v_dim, v_per_k,
        key_dim, value_dim, qkvz_dim, conv_dim);
}

void gdn_qkvz_split_f16(
    const void* qkvz, const void* ba,
    float* q, float* k, float* v, float* z,
    float* a, float* b, float* mixed,
    int num_tokens, int num_k_heads, int num_v_heads,
    int head_k_dim, int head_v_dim, int v_per_k,
    int key_dim, int value_dim, int qkvz_dim, int conv_dim,
    cudaStream_t stream)
{
    int max_dim = qkvz_dim;
    if (2 * num_v_heads > max_dim) max_dim = 2 * num_v_heads;
    int threads = (max_dim < 1024) ? max_dim : 1024;
    gdn_qkvz_split_kernel<__half><<<num_tokens, threads, 0, stream>>>(
        (const __half*)qkvz, (const __half*)ba,
        q, k, v, z, a, b, mixed,
        num_k_heads, num_v_heads, head_k_dim, head_v_dim, v_per_k,
        key_dim, value_dim, qkvz_dim, conv_dim);
}

void gdn_qkvz_split_f32(
    const void* qkvz, const void* ba,
    float* q, float* k, float* v, float* z,
    float* a, float* b, float* mixed,
    int num_tokens, int num_k_heads, int num_v_heads,
    int head_k_dim, int head_v_dim, int v_per_k,
    int key_dim, int value_dim, int qkvz_dim, int conv_dim,
    cudaStream_t stream)
{
    int max_dim = qkvz_dim;
    if (2 * num_v_heads > max_dim) max_dim = 2 * num_v_heads;
    int threads = (max_dim < 1024) ? max_dim : 1024;
    gdn_qkvz_split_kernel<float><<<num_tokens, threads, 0, stream>>>(
        (const float*)qkvz, (const float*)ba,
        q, k, v, z, a, b, mixed,
        num_k_heads, num_v_heads, head_k_dim, head_v_dim, v_per_k,
        key_dim, value_dim, qkvz_dim, conv_dim);
}

void gdn_conv_output_split(
    const float* conv_out, float* q, float* k, float* v,
    int num_tokens, int key_dim, int value_dim, int conv_dim,
    cudaStream_t stream)
{
    int threads = (conv_dim < 1024) ? conv_dim : 1024;
    gdn_conv_split_kernel<<<num_tokens, threads, 0, stream>>>(
        conv_out, q, k, v, key_dim, value_dim, conv_dim);
}

} // extern "C"
