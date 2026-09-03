// SPDX-License-Identifier: Apache-2.0
// Fused recurrent Gated Delta Rule CUDA kernel for Qwen3-Next.
//
// Port of: fused_recurrent_gated_delta_rule_fwd_kernel (Triton)
//
// The recurrence (per v-head, per token):
//   q, k = L2_normalize(q), L2_normalize(k)
//   q *= scale
//   S *= exp(g)                           // decay
//   v_delta = v - dot(S, k)               // delta
//   v_delta *= beta                       // gating
//   S += v_delta * k                      // update (outer product row)
//   o = dot(S, q)                         // output
//
// Tensor layouts (matching Python):
//   q, k: [T, H, K]  (H = num_k_heads, K = head_k_dim)
//   v:    [T, HV, V]  (HV = num_v_heads, V = head_v_dim)
//   g:    [T, HV]     (scalar per v-head per token)
//   beta: [T, HV]     (scalar per v-head per token)
//   ssm_state: [num_slots, HV, V, K]
//   o:    [T, HV, V]
//
// Grid: (cdiv(V, 1), N * HV)
// Each thread handles one element of V for one v-head for one sequence.
// K is processed entirely in registers (assertion: K <= 128).

#include <cstdint>
#include <cmath>
#include <cuda_runtime.h>

#define MAX_HEAD_K_DIM 128

__global__ void fused_recurrent_gdn_fwd_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ g,
    const float* __restrict__ beta,
    float* __restrict__ o,
    float* __restrict__ ssm_state,
    const int* __restrict__ state_indices,
    const int* __restrict__ cu_seqlens,
    float scale,
    int N,         // num sequences
    int T,         // total tokens
    int H,         // num_k_heads
    int HV,        // num_v_heads
    int K,         // head_k_dim
    int V_dim      // head_v_dim
) {
    int i_nh = blockIdx.y;
    int i_n = i_nh / HV;
    int i_hv = i_nh % HV;
    int i_h = i_hv / (HV / H);

    if (i_n >= N) return;

    int i_v = blockIdx.x * blockDim.x + threadIdx.x;
    if (i_v >= V_dim) return;

    int bos = cu_seqlens[i_n];
    int eos = cu_seqlens[i_n + 1];
    int seq_len = eos - bos;
    if (seq_len <= 0) return;

    int slot = state_indices[i_n];
    if (slot < 0) return;

    // Load state row into registers
    float* state_row = ssm_state + ((slot * HV + i_hv) * V_dim + i_v) * K;
    float b_h[MAX_HEAD_K_DIM];
    for (int ki = 0; ki < K; ki++) {
        b_h[ki] = state_row[ki];
    }

    for (int i_t = 0; i_t < seq_len; i_t++) {
        int t = bos + i_t;

        // Load and L2-normalize q and k
        const float* q_ptr = q + (t * H + i_h) * K;
        const float* k_ptr = k + (t * H + i_h) * K;

        float q_local[MAX_HEAD_K_DIM], k_local[MAX_HEAD_K_DIM];
        float q_sq = 0.0f, k_sq = 0.0f;
        for (int ki = 0; ki < K; ki++) {
            q_local[ki] = q_ptr[ki];
            k_local[ki] = k_ptr[ki];
            q_sq += q_local[ki] * q_local[ki];
            k_sq += k_local[ki] * k_local[ki];
        }
        float q_inv = rsqrtf(q_sq + 1e-6f);
        float k_inv = rsqrtf(k_sq + 1e-6f);
        for (int ki = 0; ki < K; ki++) {
            q_local[ki] *= q_inv * scale;
            k_local[ki] *= k_inv;
        }

        // Decay state
        float b_g = g[t * HV + i_hv];
        float decay = expf(b_g);
        for (int ki = 0; ki < K; ki++) {
            b_h[ki] *= decay;
        }

        // Delta: v - dot(b_h, k)
        float b_v = v[(t * HV + i_hv) * V_dim + i_v];
        float dot_hk = 0.0f;
        for (int ki = 0; ki < K; ki++) {
            dot_hk += b_h[ki] * k_local[ki];
        }
        b_v -= dot_hk;

        // Beta gating
        float b_beta = beta[t * HV + i_hv];
        b_v *= b_beta;

        // State update
        for (int ki = 0; ki < K; ki++) {
            b_h[ki] += b_v * k_local[ki];
        }

        // Output
        float b_o = 0.0f;
        for (int ki = 0; ki < K; ki++) {
            b_o += b_h[ki] * q_local[ki];
        }
        o[(t * HV + i_hv) * V_dim + i_v] = b_o;
    }

    // Store final state
    for (int ki = 0; ki < K; ki++) {
        state_row[ki] = b_h[ki];
    }
}

// RMSNormGated: out = rms_norm(x) * weight * silu(z)   (norm_before_gate)
// The gate is SiLU(z) = z * sigmoid(z), NOT plain sigmoid — matches Python
// vLLM `RMSNormGated` / the cpu_golden `gdn_rms_norm_gated` oracle.
// x, z: [total_rows, head_v_dim] where total_rows = num_tokens * num_v_heads
// One block per row.
__global__ void rms_norm_gated_kernel(
    const float* __restrict__ x,
    const float* __restrict__ z,
    const float* __restrict__ weight,
    float* __restrict__ out,
    float eps,
    int head_v_dim,
    int total_rows
) {
    int row = blockIdx.x;
    if (row >= total_rows) return;

    const float* x_row = x + row * head_v_dim;
    const float* z_row = z + row * head_v_dim;
    float* o_row = out + row * head_v_dim;

    extern __shared__ float sdata[];

    float local_ss = 0.0f;
    for (int i = threadIdx.x; i < head_v_dim; i += blockDim.x) {
        float val = x_row[i];
        local_ss += val * val;
    }

    // Warp reduction
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        local_ss += __shfl_xor_sync(0xffffffff, local_ss, offset);
    }

    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;
    if (lane == 0) sdata[warp_id] = local_ss;
    __syncthreads();

    if (warp_id == 0) {
        local_ss = (lane < (blockDim.x + warpSize - 1) / warpSize) ? sdata[lane] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            local_ss += __shfl_xor_sync(0xffffffff, local_ss, offset);
        }
    }

    if (threadIdx.x == 0) sdata[0] = local_ss;
    __syncthreads();
    float rms = rsqrtf(sdata[0] / (float)head_v_dim + eps);

    for (int i = threadIdx.x; i < head_v_dim; i += blockDim.x) {
        float normed = x_row[i] * rms * weight[i];
        float z_val = z_row[i];
        // SiLU gate: silu(z) = z * sigmoid(z) (NOT plain sigmoid).
        float silu = z_val / (1.0f + expf(-z_val));
        o_row[i] = normed * silu;
    }
}

// C entry points
extern "C" void fused_recurrent_gdn_fwd(
    const float* q, const float* k, const float* v,
    const float* g, const float* beta,
    float* o, float* ssm_state,
    const int* state_indices, const int* cu_seqlens,
    float scale,
    int N, int T, int H, int HV, int K, int V_dim,
    cudaStream_t stream)
{
    // One thread per V element, one block-row per (seq, v_head)
    int threads = (V_dim < 256) ? V_dim : 256;
    dim3 grid((V_dim + threads - 1) / threads, N * HV);
    fused_recurrent_gdn_fwd_kernel<<<grid, threads, 0, stream>>>(
        q, k, v, g, beta, o, ssm_state,
        state_indices, cu_seqlens,
        scale, N, T, H, HV, K, V_dim);
}

extern "C" void rms_norm_gated(
    const float* x, const float* z, const float* weight,
    float* out, float eps, int head_v_dim, int total_rows,
    cudaStream_t stream)
{
    int threads = (head_v_dim < 256) ? head_v_dim : 256;
    int smem = ((threads + 31) / 32) * sizeof(float);
    rms_norm_gated_kernel<<<total_rows, threads, smem, stream>>>(
        x, z, weight, out, eps, head_v_dim, total_rows);
}
