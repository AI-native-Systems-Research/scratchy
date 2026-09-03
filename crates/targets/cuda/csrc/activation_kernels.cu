// SPDX-License-Identifier: Apache-2.0
// Fused activation + element-wise multiply CUDA kernels for vLLM Rust.
//
// Port of csrc/activation_kernels.cu from Python vLLM.
// One thread block per row (token). Each block processes the gate and up
// projections element-wise: out[i] = act(gate[i]) * up[i].
//
// Uses vectorized 128-bit loads/stores for throughput.

#include <cstdint>
#include <cmath>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "vec_utils.cuh"

// ---------------------------------------------------------------------------
// Activation functions (scalar, in float)
// ---------------------------------------------------------------------------

// SiLU: x * sigmoid(x) = x / (1 + exp(-x))
__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + expf(-x));
}

// GELU (exact / erf): x * 0.5 * (1 + erf(x / sqrt(2)))
__device__ __forceinline__ float gelu_erf(float x) {
    constexpr float ALPHA = 0.7071067811865476f; // 1/sqrt(2)
    return x * 0.5f * (1.0f + erff(x * ALPHA));
}

// GELU (tanh approx): 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
__device__ __forceinline__ float gelu_tanh(float x) {
    constexpr float BETA = 0.7978845608028654f;  // sqrt(2/pi)
    constexpr float KAPPA = 0.044715f;
    float x3 = x * x * x;
    float inner = BETA * (x + KAPPA * x3);
    return 0.5f * x * (1.0f + tanhf(inner));
}

// ---------------------------------------------------------------------------
// Fused act-and-mul kernel (vectorized): out[i] = act(gate[i]) * up[i]
// ---------------------------------------------------------------------------

template <float (*ACT_FN)(float), typename T>
__global__ void act_and_mul_kernel(
    T* __restrict__ out,
    const T* __restrict__ gate,
    const T* __restrict__ up,
    int d)
{
    constexpr int VEC_SIZE = VecType<T>::SIZE;

    // Use 64-bit row offset to compute pointers — `(int)row * d` overflows
    // int32 for M ≥ 2^31/d (e.g. M ≥ 131072 at d=18944), wrapping to a
    // negative offset that lands in unrelated GPU mappings (silent
    // illegal-address fault that surfaces at the next launch).
    const size_t row = blockIdx.x;
    const size_t row_stride = (size_t)d;
    const T* g = gate + row * row_stride;
    const T* u = up + row * row_stride;
    T* o = out + row * row_stride;

    const int num_vecs = d / VEC_SIZE;
    const int tail_start = num_vecs * VEC_SIZE;

    // Vectorized loop.
    for (int vi = threadIdx.x; vi < num_vecs; vi += blockDim.x) {
        float gbuf[VEC_SIZE], ubuf[VEC_SIZE], obuf[VEC_SIZE];
        unpack_vec<T>(vec_load(&g[vi * VEC_SIZE]), gbuf);
        unpack_vec<T>(vec_load(&u[vi * VEC_SIZE]), ubuf);
        #pragma unroll
        for (int j = 0; j < VEC_SIZE; j++) {
            obuf[j] = ACT_FN(gbuf[j]) * ubuf[j];
        }
        vec_store(&o[vi * VEC_SIZE], pack_vec<T>(obuf));
    }

    // Scalar tail.
    for (int i = tail_start + threadIdx.x; i < d; i += blockDim.x) {
        float gv = ACT_FN(static_cast<float>(g[i]));
        float uv = static_cast<float>(u[i]);
        o[i] = static_cast<T>(gv * uv);
    }
}

// ---------------------------------------------------------------------------
// C entry points: silu_and_mul
// ---------------------------------------------------------------------------

extern "C" {

void silu_and_mul_f32(
    float* out, const float* gate, const float* up,
    int num_tokens, int d, cudaStream_t stream)
{
    int threads = (d < 1024) ? d : 1024;
    act_and_mul_kernel<silu, float><<<num_tokens, threads, 0, stream>>>(out, gate, up, d);
}

void silu_and_mul_f16(
    __half* out, const __half* gate, const __half* up,
    int num_tokens, int d, cudaStream_t stream)
{
    int threads = (d < 1024) ? d : 1024;
    act_and_mul_kernel<silu, __half><<<num_tokens, threads, 0, stream>>>(out, gate, up, d);
}

void silu_and_mul_bf16(
    __nv_bfloat16* out, const __nv_bfloat16* gate, const __nv_bfloat16* up,
    int num_tokens, int d, cudaStream_t stream)
{
    int threads = (d < 1024) ? d : 1024;
    act_and_mul_kernel<silu, __nv_bfloat16><<<num_tokens, threads, 0, stream>>>(out, gate, up, d);
}

// ---------------------------------------------------------------------------
// C entry points: gelu_and_mul (tanh approximation)
// ---------------------------------------------------------------------------

void gelu_and_mul_f32(
    float* out, const float* gate, const float* up,
    int num_tokens, int d, cudaStream_t stream)
{
    int threads = (d < 1024) ? d : 1024;
    act_and_mul_kernel<gelu_tanh, float><<<num_tokens, threads, 0, stream>>>(out, gate, up, d);
}

void gelu_and_mul_f16(
    __half* out, const __half* gate, const __half* up,
    int num_tokens, int d, cudaStream_t stream)
{
    int threads = (d < 1024) ? d : 1024;
    act_and_mul_kernel<gelu_tanh, __half><<<num_tokens, threads, 0, stream>>>(out, gate, up, d);
}

void gelu_and_mul_bf16(
    __nv_bfloat16* out, const __nv_bfloat16* gate, const __nv_bfloat16* up,
    int num_tokens, int d, cudaStream_t stream)
{
    int threads = (d < 1024) ? d : 1024;
    act_and_mul_kernel<gelu_tanh, __nv_bfloat16><<<num_tokens, threads, 0, stream>>>(out, gate, up, d);
}

// ---------------------------------------------------------------------------
// C entry points: gelu_new_and_mul (exact / erf)
// ---------------------------------------------------------------------------

void gelu_new_and_mul_f32(
    float* out, const float* gate, const float* up,
    int num_tokens, int d, cudaStream_t stream)
{
    int threads = (d < 1024) ? d : 1024;
    act_and_mul_kernel<gelu_erf, float><<<num_tokens, threads, 0, stream>>>(out, gate, up, d);
}

void gelu_new_and_mul_f16(
    __half* out, const __half* gate, const __half* up,
    int num_tokens, int d, cudaStream_t stream)
{
    int threads = (d < 1024) ? d : 1024;
    act_and_mul_kernel<gelu_erf, __half><<<num_tokens, threads, 0, stream>>>(out, gate, up, d);
}

void gelu_new_and_mul_bf16(
    __nv_bfloat16* out, const __nv_bfloat16* gate, const __nv_bfloat16* up,
    int num_tokens, int d, cudaStream_t stream)
{
    int threads = (d < 1024) ? d : 1024;
    act_and_mul_kernel<gelu_erf, __nv_bfloat16><<<num_tokens, threads, 0, stream>>>(out, gate, up, d);
}

} // extern "C"

// ---------------------------------------------------------------------------
// Fused act-and-mul from combined gate_up tensor [num_tokens, 2*d]
// Avoids two contiguous copy kernels + allocations per layer.
// ---------------------------------------------------------------------------

template <float (*ACT_FN)(float), typename T>
__global__ void act_and_mul_fused_kernel(
    T* __restrict__ out,
    const T* __restrict__ gate_up,
    int d)
{
    constexpr int VEC_SIZE = VecType<T>::SIZE;

    // Use 64-bit row offset. The fused variant uses stride = 2 * d which
    // doubles the multiplication, so overflow hits at half the M of the
    // non-fused kernel: at d=18944 (Qwen2.5-7B intermediate), row=57389
    // causes (int)row * 2*d to wrap negative on M=65536 prefill — the
    // exact failure block compute-sanitizer flagged ("block (57390, 0,
    // 0)"). Cast to `size_t` so the offset is computed in 64-bit and any
    // M that fits in gridDim.x is safe.
    const size_t row = blockIdx.x;
    const size_t stride = 2 * (size_t)d;
    const size_t d_off = (size_t)d;
    const T* g = gate_up + row * stride;          // gate half
    const T* u = gate_up + row * stride + d_off;  // up half
    T* o = out + row * d_off;

    const int num_vecs = d / VEC_SIZE;
    const int tail_start = num_vecs * VEC_SIZE;

    for (int vi = threadIdx.x; vi < num_vecs; vi += blockDim.x) {
        float gbuf[VEC_SIZE], ubuf[VEC_SIZE], obuf[VEC_SIZE];
        unpack_vec<T>(vec_load(&g[vi * VEC_SIZE]), gbuf);
        unpack_vec<T>(vec_load(&u[vi * VEC_SIZE]), ubuf);
        #pragma unroll
        for (int j = 0; j < VEC_SIZE; j++) {
            obuf[j] = ACT_FN(gbuf[j]) * ubuf[j];
        }
        vec_store(&o[vi * VEC_SIZE], pack_vec<T>(obuf));
    }

    for (int i = tail_start + threadIdx.x; i < d; i += blockDim.x) {
        float gv = ACT_FN(static_cast<float>(g[i]));
        float uv = static_cast<float>(u[i]);
        o[i] = static_cast<T>(gv * uv);
    }
}

extern "C" {

// C entry points: silu_and_mul_fused (takes [num_tokens, 2*d])
void silu_and_mul_fused_f32(
    float* out, const float* gate_up,
    int num_tokens, int d, cudaStream_t stream)
{
    int threads = (d < 1024) ? d : 1024;
    act_and_mul_fused_kernel<silu, float><<<num_tokens, threads, 0, stream>>>(out, gate_up, d);
}

void silu_and_mul_fused_f16(
    __half* out, const __half* gate_up,
    int num_tokens, int d, cudaStream_t stream)
{
    int threads = (d < 1024) ? d : 1024;
    act_and_mul_fused_kernel<silu, __half><<<num_tokens, threads, 0, stream>>>(out, gate_up, d);
}

void silu_and_mul_fused_bf16(
    __nv_bfloat16* out, const __nv_bfloat16* gate_up,
    int num_tokens, int d, cudaStream_t stream)
{
    int threads = (d < 1024) ? d : 1024;
    act_and_mul_fused_kernel<silu, __nv_bfloat16><<<num_tokens, threads, 0, stream>>>(out, gate_up, d);
}

// C entry points: gelu_and_mul_fused (takes [num_tokens, 2*d])
void gelu_and_mul_fused_f32(
    float* out, const float* gate_up,
    int num_tokens, int d, cudaStream_t stream)
{
    int threads = (d < 1024) ? d : 1024;
    act_and_mul_fused_kernel<gelu_tanh, float><<<num_tokens, threads, 0, stream>>>(out, gate_up, d);
}

void gelu_and_mul_fused_f16(
    __half* out, const __half* gate_up,
    int num_tokens, int d, cudaStream_t stream)
{
    int threads = (d < 1024) ? d : 1024;
    act_and_mul_fused_kernel<gelu_tanh, __half><<<num_tokens, threads, 0, stream>>>(out, gate_up, d);
}

void gelu_and_mul_fused_bf16(
    __nv_bfloat16* out, const __nv_bfloat16* gate_up,
    int num_tokens, int d, cudaStream_t stream)
{
    int threads = (d < 1024) ? d : 1024;
    act_and_mul_fused_kernel<gelu_tanh, __nv_bfloat16><<<num_tokens, threads, 0, stream>>>(out, gate_up, d);
}

} // extern "C"

// ---------------------------------------------------------------------------
// Tanh softcap inplace: x[i] = cap * tanh(x[i] / cap)
// Used for Gemma 2 final logit softcapping.
// x: [n] total elements, cap: scalar float.
// ---------------------------------------------------------------------------

template <typename T>
__global__ void tanh_softcap_inplace_kernel(
    T* __restrict__ x,
    float inv_cap,
    float cap,
    int n)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        float v = static_cast<float>(x[i]) * inv_cap;
        x[i] = static_cast<T>(cap * tanhf(v));
    }
}

extern "C" {

void tanh_softcap_inplace_f32(
    float* x, float cap, int n, cudaStream_t stream)
{
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    tanh_softcap_inplace_kernel<float><<<blocks, threads, 0, stream>>>(
        x, 1.0f / cap, cap, n);
}

void tanh_softcap_inplace_f16(
    __half* x, float cap, int n, cudaStream_t stream)
{
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    tanh_softcap_inplace_kernel<__half><<<blocks, threads, 0, stream>>>(
        x, 1.0f / cap, cap, n);
}

void tanh_softcap_inplace_bf16(
    __nv_bfloat16* x, float cap, int n, cudaStream_t stream)
{
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    tanh_softcap_inplace_kernel<__nv_bfloat16><<<blocks, threads, 0, stream>>>(
        x, 1.0f / cap, cap, n);
}

} // extern "C"

// ---------------------------------------------------------------------------
// Broadcast multiply inplace: x[row, col] *= scale[col]
// x: [num_rows, d], scale: [d] — both same dtype.
// One block per row.
// ---------------------------------------------------------------------------

template <typename T>
__global__ void broadcast_mul_inplace_kernel(
    T* __restrict__ x,
    const T* __restrict__ scale,
    int d)
{
    const int row = blockIdx.x;
    T* xr = x + row * d;

    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        float xv = (float)xr[i];
        float sv = (float)scale[i];
        xr[i] = (T)(xv * sv);
    }
}

extern "C" {

void broadcast_mul_inplace_f32(
    float* x, const float* scale,
    int num_rows, int d, cudaStream_t stream)
{
    int threads = (d < 1024) ? d : 1024;
    broadcast_mul_inplace_kernel<float><<<num_rows, threads, 0, stream>>>(x, scale, d);
}

void broadcast_mul_inplace_f16(
    __half* x, const __half* scale,
    int num_rows, int d, cudaStream_t stream)
{
    int threads = (d < 1024) ? d : 1024;
    broadcast_mul_inplace_kernel<__half><<<num_rows, threads, 0, stream>>>(x, scale, d);
}

void broadcast_mul_inplace_bf16(
    __nv_bfloat16* x, const __nv_bfloat16* scale,
    int num_rows, int d, cudaStream_t stream)
{
    int threads = (d < 1024) ? d : 1024;
    broadcast_mul_inplace_kernel<__nv_bfloat16><<<num_rows, threads, 0, stream>>>(x, scale, d);
}

} // extern "C"

// ---------------------------------------------------------------------------
// ScalarWeightMul: out = in * w[0] — multiply every element by a single
// [1]-shaped loaded weight (Gemma4 `layer_scalar[layer]`). Unlike
// broadcast_mul_inplace (per-channel scale[i]), the scale here is one scalar
// broadcast across the whole tensor.
// ---------------------------------------------------------------------------

template <typename T>
__global__ void scalar_weight_mul_kernel(
    T* __restrict__ out, const T* __restrict__ in, const T* __restrict__ w, int n)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const float s = (float)w[0];
    out[idx] = (T)((float)in[idx] * s);
}

extern "C" {

void scalar_weight_mul_f32(
    float* out, const float* in, const float* w, int n, cudaStream_t stream)
{
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    scalar_weight_mul_kernel<float><<<blocks, threads, 0, stream>>>(out, in, w, n);
}

void scalar_weight_mul_f16(
    __half* out, const __half* in, const __half* w, int n, cudaStream_t stream)
{
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    scalar_weight_mul_kernel<__half><<<blocks, threads, 0, stream>>>(out, in, w, n);
}

void scalar_weight_mul_bf16(
    __nv_bfloat16* out, const __nv_bfloat16* in, const __nv_bfloat16* w, int n, cudaStream_t stream)
{
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    scalar_weight_mul_kernel<__nv_bfloat16><<<blocks, threads, 0, stream>>>(out, in, w, n);
}

} // extern "C"

// ---------------------------------------------------------------------------
// QuickGELU activation (in-place pointwise)
//
// QuickGELU = x * sigmoid(1.702 * x). This is the OpenAI/CLIP-style
// approximation used by Qwen2-VL's vision MLP (and the classic
// HuggingFace `quick_gelu` activation). NOT the same as `gelu_tanh`
// or `gelu_erf` — the multiplier 1.702 is tuned to approximate
// erf-GELU more cheaply than the tanh form.
//
// Vectorized 128-bit loads/stores; one thread block per row, threads
// stride across the d-dim.
// ---------------------------------------------------------------------------

__device__ __forceinline__ float quick_gelu(float x) {
    return x / (1.0f + expf(-1.702f * x));
}

template <typename T>
__global__ void quick_gelu_inplace_kernel(T* __restrict__ x, int n_elements) {
    constexpr int VEC_SIZE = VecType<T>::SIZE;

    const int total_vecs = n_elements / VEC_SIZE;
    const int tail_start = total_vecs * VEC_SIZE;

    // Vectorized loop: one warp-stride per vec.
    for (int vi = blockIdx.x * blockDim.x + threadIdx.x;
         vi < total_vecs;
         vi += blockDim.x * gridDim.x)
    {
        T* slot = x + vi * VEC_SIZE;
        float buf[VEC_SIZE];
        unpack_vec<T>(vec_load(slot), buf);
        #pragma unroll
        for (int j = 0; j < VEC_SIZE; j++) {
            buf[j] = quick_gelu(buf[j]);
        }
        vec_store(slot, pack_vec<T>(buf));
    }

    // Scalar tail (only block 0's first warp handles it; trivial).
    if (blockIdx.x == 0) {
        for (int i = tail_start + threadIdx.x; i < n_elements; i += blockDim.x) {
            x[i] = static_cast<T>(quick_gelu(static_cast<float>(x[i])));
        }
    }
}

extern "C" {

#define LAUNCH_QUICK_GELU(T)                                                   \
    do {                                                                       \
        if (n_elements <= 0) return;                                           \
        const int VEC = VecType<T>::SIZE;                                      \
        const int total_vecs = n_elements / VEC;                               \
        const int threads = 256;                                               \
        const int blocks = (total_vecs > 0) ? min(1024, (total_vecs + threads - 1) / threads) : 1; \
        quick_gelu_inplace_kernel<T><<<blocks, threads, 0, stream>>>(          \
            reinterpret_cast<T*>(x), n_elements);                              \
    } while (0)

void quick_gelu_inplace_f32(void* x, int n_elements, cudaStream_t stream)
{ LAUNCH_QUICK_GELU(float); }

void quick_gelu_inplace_f16(void* x, int n_elements, cudaStream_t stream)
{ LAUNCH_QUICK_GELU(__half); }

void quick_gelu_inplace_bf16(void* x, int n_elements, cudaStream_t stream)
{ LAUNCH_QUICK_GELU(__nv_bfloat16); }

} // extern "C"

// ---------------------------------------------------------------------------
// GELU (erf form) — in-place pointwise
//
// Standard PyTorch `nn.GELU()` (default `approximate='none'`):
//   x * 0.5 * (1 + erf(x / sqrt(2)))
//
// Used by Qwen2-VL's `Qwen2VisionPatchMerger.mlp[1]` (sandwiched between
// the two Linear layers). NOT the same as `gelu_and_mul_fused` (which
// uses the tanh approximation) or `quick_gelu_inplace`.
// ---------------------------------------------------------------------------

template <typename T>
__global__ void gelu_erf_inplace_kernel(T* __restrict__ x, int n_elements) {
    constexpr int VEC_SIZE = VecType<T>::SIZE;

    const int total_vecs = n_elements / VEC_SIZE;
    const int tail_start = total_vecs * VEC_SIZE;

    for (int vi = blockIdx.x * blockDim.x + threadIdx.x;
         vi < total_vecs;
         vi += blockDim.x * gridDim.x)
    {
        T* slot = x + vi * VEC_SIZE;
        float buf[VEC_SIZE];
        unpack_vec<T>(vec_load(slot), buf);
        #pragma unroll
        for (int j = 0; j < VEC_SIZE; j++) {
            buf[j] = gelu_erf(buf[j]);
        }
        vec_store(slot, pack_vec<T>(buf));
    }

    if (blockIdx.x == 0) {
        for (int i = tail_start + threadIdx.x; i < n_elements; i += blockDim.x) {
            x[i] = static_cast<T>(gelu_erf(static_cast<float>(x[i])));
        }
    }
}

extern "C" {

#define LAUNCH_GELU_ERF(T)                                                     \
    do {                                                                       \
        if (n_elements <= 0) return;                                           \
        const int VEC = VecType<T>::SIZE;                                      \
        const int total_vecs = n_elements / VEC;                               \
        const int threads = 256;                                               \
        const int blocks = (total_vecs > 0) ? min(1024, (total_vecs + threads - 1) / threads) : 1; \
        gelu_erf_inplace_kernel<T><<<blocks, threads, 0, stream>>>(            \
            reinterpret_cast<T*>(x), n_elements);                              \
    } while (0)

void gelu_erf_inplace_f32(void* x, int n_elements, cudaStream_t stream)
{ LAUNCH_GELU_ERF(float); }

void gelu_erf_inplace_f16(void* x, int n_elements, cudaStream_t stream)
{ LAUNCH_GELU_ERF(__half); }

void gelu_erf_inplace_bf16(void* x, int n_elements, cudaStream_t stream)
{ LAUNCH_GELU_ERF(__nv_bfloat16); }

} // extern "C"

// ---------------------------------------------------------------------------
// GELU (tanh form) — in-place pointwise
//
// PyTorch `nn.GELU(approximate="tanh")` (HuggingFace activation key
// `gelu_pytorch_tanh`):
//   0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
//
// Used by SigLIP / Gemma3-MM vision MLP and other towers calibrated for
// tanh-GELU. Mirrors `quick_gelu_inplace` / `gelu_erf_inplace`; same
// vector-load/store + scalar-tail pattern.
// ---------------------------------------------------------------------------

template <typename T>
__global__ void gelu_tanh_inplace_kernel(T* __restrict__ x, int n_elements) {
    constexpr int VEC_SIZE = VecType<T>::SIZE;

    const int total_vecs = n_elements / VEC_SIZE;
    const int tail_start = total_vecs * VEC_SIZE;

    for (int vi = blockIdx.x * blockDim.x + threadIdx.x;
         vi < total_vecs;
         vi += blockDim.x * gridDim.x)
    {
        T* slot = x + vi * VEC_SIZE;
        float buf[VEC_SIZE];
        unpack_vec<T>(vec_load(slot), buf);
        #pragma unroll
        for (int j = 0; j < VEC_SIZE; j++) {
            buf[j] = gelu_tanh(buf[j]);
        }
        vec_store(slot, pack_vec<T>(buf));
    }

    if (blockIdx.x == 0) {
        for (int i = tail_start + threadIdx.x; i < n_elements; i += blockDim.x) {
            x[i] = static_cast<T>(gelu_tanh(static_cast<float>(x[i])));
        }
    }
}

extern "C" {

#define LAUNCH_GELU_TANH(T)                                                    \
    do {                                                                       \
        if (n_elements <= 0) return;                                           \
        const int VEC = VecType<T>::SIZE;                                      \
        const int total_vecs = n_elements / VEC;                               \
        const int threads = 256;                                               \
        const int blocks = (total_vecs > 0) ? min(1024, (total_vecs + threads - 1) / threads) : 1; \
        gelu_tanh_inplace_kernel<T><<<blocks, threads, 0, stream>>>(           \
            reinterpret_cast<T*>(x), n_elements);                              \
    } while (0)

void gelu_tanh_inplace_f32(void* x, int n_elements, cudaStream_t stream)
{ LAUNCH_GELU_TANH(float); }

void gelu_tanh_inplace_f16(void* x, int n_elements, cudaStream_t stream)
{ LAUNCH_GELU_TANH(__half); }

void gelu_tanh_inplace_bf16(void* x, int n_elements, cudaStream_t stream)
{ LAUNCH_GELU_TANH(__nv_bfloat16); }

} // extern "C"

// ---------------------------------------------------------------------------
// AvgPool2d (non-overlapping) on a flat patch grid
//
// Models the Gemma3-MM SigLIP→text projector's average-pool stage. The
// flat input `[L = ph * ph, e]` is treated as a `ph × ph × e` grid; we
// fold each `k × k` cell (stride == kernel) into one output row of an
// `[(ph/k) * (ph/k), e]` output. ph and k come from the per-arch
// `CanonicalParams` bake (`vision_patch_grid_side` and
// `vision_pool_kernel`), so the kernel sees them as plain ints.
//
// One thread per `(output_row, embed_dim)` cell — total ≈ 295k threads
// for SigLIP @ 896² (ph=64, k=4, e=1152, → 256 × 1152). Each thread
// reads k² source elements from non-contiguous rows (separated by
// `ph * e`) and writes one bf16 — bandwidth-bound. No vectorization
// (pool side is too small to make 128-bit loads worthwhile).

template <typename T>
__global__ void avg_pool_2d_kernel(
    T* __restrict__ out,         // [(ph/k) * (ph/k), e]
    const T* __restrict__ in,    // [ph * ph, e]
    int ph,
    int k,
    int e
) {
    const int ph_out = ph / k;
    const int total_out = ph_out * ph_out * e;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_out) return;

    int dim = idx % e;
    int out_row = idx / e;
    int out_r = out_row / ph_out;
    int out_c = out_row % ph_out;
    int in_r0 = out_r * k;
    int in_c0 = out_c * k;

    float acc = 0.0f;
    const float inv = 1.0f / static_cast<float>(k * k);
    for (int dr = 0; dr < k; dr++) {
        int in_r = in_r0 + dr;
        const T* row = in + (in_r * ph + in_c0) * e + dim;
        for (int dc = 0; dc < k; dc++) {
            acc += static_cast<float>(row[dc * e]);
        }
    }
    out[idx] = static_cast<T>(acc * inv);
}

extern "C" {

#define LAUNCH_AVG_POOL_2D(T)                                                  \
    do {                                                                       \
        const int ph_out = ph / k;                                             \
        const int total = ph_out * ph_out * e;                                 \
        if (total <= 0) return;                                                \
        const int threads = 256;                                               \
        const int blocks = (total + threads - 1) / threads;                    \
        avg_pool_2d_kernel<T><<<blocks, threads, 0, stream>>>(                 \
            reinterpret_cast<T*>(out),                                         \
            reinterpret_cast<const T*>(in),                                    \
            ph, k, e);                                                         \
    } while (0)

void avg_pool_2d_f32(void* out, const void* in, int ph, int k, int e, cudaStream_t stream)
{ LAUNCH_AVG_POOL_2D(float); }

void avg_pool_2d_f16(void* out, const void* in, int ph, int k, int e, cudaStream_t stream)
{ LAUNCH_AVG_POOL_2D(__half); }

void avg_pool_2d_bf16(void* out, const void* in, int ph, int k, int e, cudaStream_t stream)
{ LAUNCH_AVG_POOL_2D(__nv_bfloat16); }

} // extern "C"
