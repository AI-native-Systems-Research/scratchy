// SPDX-License-Identifier: Apache-2.0
// Causal conv1d update CUDA kernel for GDN (Qwen3-Next).
//
// For decode (single token per sequence):
// 1. Compute dot product of [conv_state..., new_token] with conv weights
// 2. Apply SiLU activation
// 3. Shift conv_state left, insert new token
//
// Port of: _causal_conv1d_update_kernel (Triton) from causal_conv1d.py

#include <cstdint>
#include <cmath>
#include <cuda_runtime.h>

__device__ __forceinline__ float silu_f(float x) {
    return x / (1.0f + expf(-x));
}

// Single-token update kernel.
// conv_state: [num_slots, conv_dim, state_len]  (state_len = kernel_size - 1)
// x:          [batch, conv_dim]
// w:          [conv_dim, kernel_size]
// output:     [batch, conv_dim]
// state_indices: [batch]
__global__ void causal_conv1d_update_kernel(
    float* __restrict__ conv_state,
    const float* __restrict__ x,
    const float* __restrict__ w,
    float* __restrict__ output,
    const int* __restrict__ state_indices,
    int conv_dim,
    int kernel_size,
    int batch_size
) {
    int i_b = blockIdx.x;
    int d = blockIdx.y * blockDim.x + threadIdx.x;

    if (i_b >= batch_size || d >= conv_dim) return;

    int slot = state_indices[i_b];
    int state_len = kernel_size - 1;
    float* state_ptr = conv_state + (slot * conv_dim + d) * state_len;
    const float* w_ptr = w + d * kernel_size;

    // Convolution
    float sum = 0.0f;
    for (int ki = 0; ki < state_len; ki++) {
        sum += state_ptr[ki] * w_ptr[ki];
    }
    float new_val = x[i_b * conv_dim + d];
    sum += new_val * w_ptr[state_len];

    output[i_b * conv_dim + d] = silu_f(sum);

    // Shift state left, insert new token
    for (int ki = 0; ki < state_len - 1; ki++) {
        state_ptr[ki] = state_ptr[ki + 1];
    }
    if (state_len > 0) {
        state_ptr[state_len - 1] = new_val;
    }
}

// Multi-token prefill kernel.
// x:      [num_tokens, conv_dim]
// output: [num_tokens, conv_dim]
__global__ void causal_conv1d_prefill_kernel(
    float* __restrict__ conv_state,
    const float* __restrict__ x,
    const float* __restrict__ w,
    float* __restrict__ output,
    int slot_idx,
    int conv_dim,
    int kernel_size,
    int num_tokens
) {
    int t = blockIdx.x;
    int d = blockIdx.y * blockDim.x + threadIdx.x;

    if (t >= num_tokens || d >= conv_dim) return;

    int state_len = kernel_size - 1;
    float* state_ptr = conv_state + (slot_idx * conv_dim + d) * state_len;
    const float* w_ptr = w + d * kernel_size;

    float sum = 0.0f;
    for (int ki = 0; ki < kernel_size; ki++) {
        int src_t = t - state_len + ki;
        float val;
        if (src_t < 0) {
            int state_pos = state_len + src_t;
            val = (state_pos >= 0) ? state_ptr[state_pos] : 0.0f;
        } else {
            val = x[src_t * conv_dim + d];
        }
        sum += val * w_ptr[ki];
    }

    output[t * conv_dim + d] = silu_f(sum);

    // Last token updates state
    if (t == num_tokens - 1) {
        for (int ki = 0; ki < state_len; ki++) {
            int src_t = num_tokens - state_len + ki;
            if (src_t >= 0) {
                state_ptr[ki] = x[src_t * conv_dim + d];
            }
            // If src_t < 0, keep old state (already there from shifted perspective)
        }
    }
}

// C entry points
extern "C" void causal_conv1d_update(
    float* conv_state, const float* x, const float* w, float* output,
    const int* state_indices,
    int conv_dim, int kernel_size, int batch_size,
    cudaStream_t stream)
{
    int threads = (conv_dim < 256) ? conv_dim : 256;
    dim3 grid(batch_size, (conv_dim + threads - 1) / threads);
    causal_conv1d_update_kernel<<<grid, threads, 0, stream>>>(
        conv_state, x, w, output, state_indices,
        conv_dim, kernel_size, batch_size);
}

extern "C" void causal_conv1d_prefill(
    float* conv_state, const float* x, const float* w, float* output,
    int slot_idx, int conv_dim, int kernel_size, int num_tokens,
    cudaStream_t stream)
{
    int threads = (conv_dim < 256) ? conv_dim : 256;
    dim3 grid(num_tokens, (conv_dim + threads - 1) / threads);
    causal_conv1d_prefill_kernel<<<grid, threads, 0, stream>>>(
        conv_state, x, w, output, slot_idx,
        conv_dim, kernel_size, num_tokens);
}
