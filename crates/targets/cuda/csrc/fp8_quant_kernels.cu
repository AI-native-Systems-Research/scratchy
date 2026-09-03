// SPDX-License-Identifier: Apache-2.0
//
// FP8 E4M3 quantization kernels — verbatim port of upstream vllm's
// `csrc/quantization/w8a8/fp8/common.cu`. Every functional line of
// `dynamic_per_token_scaled_fp8_quant_kernel_strided` and its helpers
// is preserved bit-for-bit; only the surrounding torch dependencies
// are stripped (raw pointers + cudaStream_t in place of torch::Tensor).
//
// Sources merged:
//   csrc/quantization/vectorization.cuh        → vec_n_t
//   csrc/quantization/vectorization_utils.cuh  → vectorize_(read_)with_alignment
//   csrc/quantization/utils.cuh                → quant_type_max_v, min_scaling_factor
//   csrc/quantization/w8a8/fp8/common.cuh      → scaled_fp8_conversion, atomicMaxFloat
//   csrc/quantization/w8a8/fp8/nvidia/quant_utils.cuh → fp8::vec_conversion
//   csrc/quantization/w8a8/fp8/common.cu       → kernels + dispatchers

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cub/cub.cuh>
#include <cmath>

// ---------------------------------------------------------------------------
// vllm/csrc/quantization/vectorization.cuh
// ---------------------------------------------------------------------------

namespace vllm {

template <typename scalar_t, size_t vec_size>
struct __align__(vec_size * sizeof(scalar_t)) vec_n_t {
  scalar_t val[vec_size];
};

}  // namespace vllm

// ---------------------------------------------------------------------------
// vllm/csrc/quantization/utils.cuh
// (specialised to c10::Float8_e4m3fn → raw u8 storage; FP8_E4M3_MAX is the
// same 448.0 c10 returns from numeric_limits<Float8_e4m3fn>::max())
// ---------------------------------------------------------------------------

static constexpr float FP8_E4M3_MAX = 448.0f;

template <typename T>
struct quant_type_max {
  static constexpr float val() { return FP8_E4M3_MAX; }
};

template <typename T>
__device__ __host__ static constexpr float quant_type_max_v = quant_type_max<T>::val();

template <typename T>
struct min_scaling_factor {
  __device__ __forceinline__ static float val() {
    return 1.0f / (quant_type_max_v<T> * 512.0f);
  }
};

// ---------------------------------------------------------------------------
// vllm/csrc/quantization/w8a8/fp8/common.cuh
// ---------------------------------------------------------------------------

namespace vllm {

__device__ __forceinline__ float atomicMaxFloat(float* addr, float value) {
  float old;
  old = (value >= 0)
            ? __int_as_float(atomicMax((int*)addr, __float_as_int(value)))
            : __uint_as_float(
                  atomicMin((unsigned int*)addr, __float_as_uint(value)));

  return old;
}

template <bool is_scale_inverted, typename fp8_type>
__device__ __forceinline__ fp8_type scaled_fp8_conversion(float const val,
                                                          float const scale) {
  float x = 0.0f;
  if constexpr (is_scale_inverted) {
    x = val * scale;
  } else {
    x = val / scale;
  }

  float r =
      fmaxf(-quant_type_max_v<fp8_type>, fminf(x, quant_type_max_v<fp8_type>));
  // Hardware cvt — c10::Float8_e4m3fn(uint8_t, from_bits()) is just a
  // typed wrapper around the byte returned by __nv_cvt_float_to_fp8;
  // scratchy stores fp8 as raw uint8_t directly.
  return __nv_cvt_float_to_fp8(r, __NV_SATFINITE, __NV_E4M3);
}

}  // namespace vllm

// ---------------------------------------------------------------------------
// vllm/csrc/quantization/vectorization_utils.cuh
// ---------------------------------------------------------------------------

namespace vllm {

template <int VEC_SIZE, typename InT, typename OutT, typename ScaOp>
struct DefaultVecOp {
  ScaOp scalar_op;

  __device__ __forceinline__ void operator()(
      vec_n_t<OutT, VEC_SIZE>& dst, const vec_n_t<InT, VEC_SIZE>& src) const {
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
      scalar_op(dst.val[i], src.val[i]);
    }
  }
};

template <int VEC_SIZE, typename InT, typename OutT, typename VecOp,
          typename ScaOp>
__device__ inline void vectorize_with_alignment(
    const InT* in, OutT* out, int len, int tid, int stride,
    VecOp&& vec_op, ScaOp&& scalar_op) {
  static_assert(VEC_SIZE > 0 && (VEC_SIZE & (VEC_SIZE - 1)) == 0,
                "VEC_SIZE must be a positive power-of-two");
  constexpr int WIDTH = VEC_SIZE * sizeof(InT);
  uintptr_t addr = reinterpret_cast<uintptr_t>(in);

  bool can_vec = ((addr & (WIDTH - 1)) == 0) && ((len & (VEC_SIZE - 1)) == 0);
  if (can_vec) {
    int num_vec = len / VEC_SIZE;

    using vin_t = vec_n_t<InT, VEC_SIZE>;
    using vout_t = vec_n_t<OutT, VEC_SIZE>;
    auto* v_in = reinterpret_cast<const vin_t*>(in);
    auto* v_out = reinterpret_cast<vout_t*>(out);

    for (int i = tid; i < num_vec; i += stride) {
      vout_t tmp;
      vin_t src = v_in[i];
      vec_op(tmp, src);
      v_out[i] = tmp;
    }
    return;
  }

  int misalignment_offset = addr & (WIDTH - 1);
  int alignment_bytes = WIDTH - misalignment_offset;
  int prefix_elems = alignment_bytes & (WIDTH - 1);
  prefix_elems /= sizeof(InT);
  prefix_elems = min(prefix_elems, len);

  for (int i = tid; i < prefix_elems; i += stride) {
    scalar_op(out[i], in[i]);
  }

  in += prefix_elems;
  out += prefix_elems;
  len -= prefix_elems;

  int num_vec = len / VEC_SIZE;
  using vin_t = vec_n_t<InT, VEC_SIZE>;
  using vout_t = vec_n_t<OutT, VEC_SIZE>;
  auto* v_in = reinterpret_cast<const vin_t*>(in);
  auto* v_out = reinterpret_cast<vout_t*>(out);

  for (int i = tid; i < num_vec; i += stride) {
    vout_t tmp;
    vin_t src = v_in[i];
    vec_op(tmp, src);
    v_out[i] = tmp;
  }

  int tail_start = num_vec * VEC_SIZE;
  for (int i = tid + tail_start; i < len; i += stride) {
    scalar_op(out[i], in[i]);
  }
}

template <int VEC_SIZE, typename InT, typename OutT, typename ScaOp>
__device__ __forceinline__ void vectorize_with_alignment(const InT* in,
                                                         OutT* out, int len,
                                                         int tid, int stride,
                                                         ScaOp&& scalar_op) {
  using Vec = DefaultVecOp<VEC_SIZE, InT, OutT, std::decay_t<ScaOp>>;
  vectorize_with_alignment<VEC_SIZE>(in, out, len, tid, stride, Vec{scalar_op},
                                     std::forward<ScaOp>(scalar_op));
}

template <int VEC_SIZE, typename InT, typename ScaOp>
struct DefaultReadVecOp {
  ScaOp scalar_op;

  __device__ __forceinline__ void operator()(
      const vec_n_t<InT, VEC_SIZE>& src) const {
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
      scalar_op(src.val[i]);
    }
  }
};

template <int VEC_SIZE, typename InT, typename VecOp, typename ScaOp>
__device__ inline void vectorize_read_with_alignment(const InT* in, int len,
                                                     int tid, int stride,
                                                     VecOp&& vec_op,
                                                     ScaOp&& scalar_op) {
  static_assert(VEC_SIZE > 0 && (VEC_SIZE & (VEC_SIZE - 1)) == 0,
                "VEC_SIZE must be a positive power-of-two");
  constexpr int WIDTH = VEC_SIZE * sizeof(InT);
  uintptr_t addr = reinterpret_cast<uintptr_t>(in);

  bool can_vec = ((addr & (WIDTH - 1)) == 0) && ((len & (VEC_SIZE - 1)) == 0);
  if (can_vec) {
    int num_vec = len / VEC_SIZE;

    using vin_t = vec_n_t<InT, VEC_SIZE>;
    auto* v_in = reinterpret_cast<const vin_t*>(in);

    for (int i = tid; i < num_vec; i += stride) {
      vin_t tmp = v_in[i];
      vec_op(tmp);
    }
    return;
  }

  int misalignment_offset = addr & (WIDTH - 1);
  int alignment_bytes = WIDTH - misalignment_offset;
  int prefix_elems = alignment_bytes & (WIDTH - 1);
  prefix_elems /= sizeof(InT);
  prefix_elems = min(prefix_elems, len);

  for (int i = tid; i < prefix_elems; i += stride) {
    scalar_op(in[i]);
  }

  in += prefix_elems;
  len -= prefix_elems;

  int num_vec = len / VEC_SIZE;
  using vin_t = vec_n_t<InT, VEC_SIZE>;
  auto* v_in = reinterpret_cast<const vin_t*>(in);

  for (int i = tid; i < num_vec; i += stride) {
    vec_op(v_in[i]);
  }

  int tail_start = num_vec * VEC_SIZE;
  for (int i = tid + tail_start; i < len; i += stride) {
    scalar_op(in[i]);
  }
}

template <int VEC_SIZE, typename InT, typename ScaOp>
__device__ __forceinline__ void vectorize_read_with_alignment(
    const InT* in, int len, int tid, int stride, ScaOp&& scalar_op) {
  using Vec = DefaultReadVecOp<VEC_SIZE, InT, std::decay_t<ScaOp>>;
  vectorize_read_with_alignment<VEC_SIZE>(in, len, tid, stride, Vec{scalar_op},
                                          std::forward<ScaOp>(scalar_op));
}

}  // namespace vllm

// ---------------------------------------------------------------------------
// vllm/csrc/cub_helpers.h
// ---------------------------------------------------------------------------

#if CUB_VERSION >= 200800
  #include <cuda/std/functional>
using CubMaxOp = cuda::maximum<>;
#else
using CubMaxOp = cub::Max;
#endif

// ---------------------------------------------------------------------------
// vllm/csrc/quantization/w8a8/fp8/common.cu — kernel + segmented_max_reduction
// ---------------------------------------------------------------------------

namespace vllm {

template <typename scalar_t, typename fp8_type>
__global__ void segmented_max_reduction_strided(
    float* __restrict__ scale, const scalar_t* __restrict__ input,
    int hidden_size, int64_t in_row_stride, int64_t num_tokens) {
  __shared__ float cache[256];
  const int tid = threadIdx.x;
  int64_t token_idx = blockIdx.x;

  if (token_idx >= num_tokens) {
    return;
  }

  const scalar_t* row_ptr = input + token_idx * in_row_stride;

  float thread_max = 0.0f;
  for (int e = tid; e < hidden_size; e += blockDim.x) {
    float v = fabsf(static_cast<float>(row_ptr[e]));
    thread_max = fmaxf(thread_max, v);
  }

  cache[tid] = thread_max;
  __syncthreads();

  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (tid < offset) {
      cache[tid] = fmaxf(cache[tid], cache[tid + offset]);
    }
    __syncthreads();
  }

  if (tid == 0) {
    atomicMaxFloat(scale, cache[0] / quant_type_max_v<fp8_type>);
  }
}

template <typename scalar_t, typename fp8_type>
__global__ void scaled_fp8_quant_kernel_strided_dynamic(
    fp8_type* __restrict__ out, const scalar_t* __restrict__ input,
    const float* __restrict__ scale, int hidden_size, int64_t in_row_stride,
    int64_t out_row_stride) {
  const int64_t token_idx = blockIdx.x;
  const int tid = threadIdx.x;

  const scalar_t* token_in = input + token_idx * in_row_stride;
  fp8_type* token_out = out + token_idx * out_row_stride;

  const float reciprocal_scale = 1.0f / (*scale);
  vectorize_with_alignment<16>(
      token_in, token_out, hidden_size, tid, blockDim.x,
      [=] __device__(fp8_type & dst, const scalar_t& src) {
        dst = scaled_fp8_conversion<true, fp8_type>(static_cast<float>(src),
                                                    reciprocal_scale);
      });
}

template <typename scalar_t, typename fp8_type>
__global__ void dynamic_per_token_scaled_fp8_quant_kernel_strided(
    fp8_type* __restrict__ out, float* __restrict__ scale,
    const scalar_t* __restrict__ input, const float* __restrict__ scale_ub,
    int hidden_size, int64_t in_row_stride, int64_t out_row_stride) {
  const int64_t token_idx = blockIdx.x;
  const int tid = threadIdx.x;

  int64_t in_offset = static_cast<int64_t>(token_idx) * in_row_stride;
  int64_t out_offset = static_cast<int64_t>(token_idx) * out_row_stride;
  const scalar_t* token_in = input + in_offset;
  fp8_type* token_out = out + out_offset;

  // 1) per-token absmax
  float absmax_val = 0.f;
  vectorize_read_with_alignment<16>(
      token_in, hidden_size, tid, blockDim.x, [&] __device__(scalar_t v) {
        absmax_val = fmaxf(absmax_val, fabsf(static_cast<float>(v)));
      });

  using BlockReduce = cub::BlockReduce<float, 256>;
  __shared__ typename BlockReduce::TempStorage tmp;
  const float block_max =
      BlockReduce(tmp).Reduce(absmax_val, CubMaxOp{}, blockDim.x);

  __shared__ float token_scale;
  if (tid == 0) {
    token_scale = scale_ub ? fminf(block_max, *scale_ub) : block_max;
    token_scale = fmaxf(token_scale / quant_type_max_v<fp8_type>,
                        min_scaling_factor<fp8_type>::val());
    scale[token_idx] = token_scale;
  }
  __syncthreads();

  // 2) quantize
  vectorize_with_alignment<16>(
      token_in, token_out, hidden_size, tid, blockDim.x,
      [=] __device__(fp8_type & dst, const scalar_t& src) {
        dst = scaled_fp8_conversion<false, fp8_type>(static_cast<float>(src),
                                                     token_scale);
      });
}

}  // namespace vllm

// ---------------------------------------------------------------------------
// Operator overloads for converting our raw scalar types into float in the
// `static_cast<float>(v)` calls inside the kernel — bf16/f16 don't have an
// implicit cast to float, but vllm relies on c10's overloads. We add the
// equivalent overloads for __nv_bfloat16 / __half so the kernel body stays
// byte-identical to vllm's source.
// ---------------------------------------------------------------------------

// __nv_bfloat16 → float and __half → float already exist as intrinsic
// conversions; the explicit `static_cast<float>` invokes
// `__bfloat162float`/`__half2float` automatically.

// ---------------------------------------------------------------------------
// Kernel 2: Static FP8 quantization (kept from the original port — used by
// `Fp8Linear::forward` static-input-scale path).
// ---------------------------------------------------------------------------

__global__ void scaled_fp8_quant_static_bf16_kernel(
    const uint16_t* __restrict__ input,
    uint8_t* __restrict__ output,
    const float* __restrict__ scale_ptr,
    int num_elements)
{
    float scale = *scale_ptr;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int vec_elems = 8;
    const int total_vecs = num_elements / vec_elems;
    const int remainder_start = total_vecs * vec_elems;

    for (int vi = tid; vi < total_vecs; vi += blockDim.x * gridDim.x) {
        int base = vi * vec_elems;
        uint4 in_vec = *reinterpret_cast<const uint4*>(&input[base]);
        const uint16_t* vals = reinterpret_cast<const uint16_t*>(&in_vec);

        uint8_t fp8_bytes[8];
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            float fval = __bfloat162float(
                *reinterpret_cast<const __nv_bfloat16*>(&vals[j]));
            fval = fval / scale;
            fval = fmaxf(-FP8_E4M3_MAX, fminf(fval, FP8_E4M3_MAX));
            fp8_bytes[j] = __nv_cvt_float_to_fp8(fval, __NV_SATFINITE, __NV_E4M3);
        }

        uint2 out_vec;
        out_vec.x = *reinterpret_cast<uint32_t*>(&fp8_bytes[0]);
        out_vec.y = *reinterpret_cast<uint32_t*>(&fp8_bytes[4]);
        *reinterpret_cast<uint2*>(&output[base]) = out_vec;
    }

    for (int i = remainder_start + tid; i < num_elements; i += blockDim.x * gridDim.x) {
        float fval = __bfloat162float(
            *reinterpret_cast<const __nv_bfloat16*>(&input[i]));
        fval = fval / scale;
        fval = fmaxf(-FP8_E4M3_MAX, fminf(fval, FP8_E4M3_MAX));
        output[i] = __nv_cvt_float_to_fp8(fval, __NV_SATFINITE, __NV_E4M3);
    }
}

__global__ void scaled_fp8_quant_static_f16_kernel(
    const uint16_t* __restrict__ input,
    uint8_t* __restrict__ output,
    const float* __restrict__ scale_ptr,
    int num_elements)
{
    float scale = *scale_ptr;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int vec_elems = 8;
    const int total_vecs = num_elements / vec_elems;
    const int remainder_start = total_vecs * vec_elems;

    for (int vi = tid; vi < total_vecs; vi += blockDim.x * gridDim.x) {
        int base = vi * vec_elems;
        uint4 in_vec = *reinterpret_cast<const uint4*>(&input[base]);
        const uint16_t* vals = reinterpret_cast<const uint16_t*>(&in_vec);

        uint8_t fp8_bytes[8];
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            float fval = __half2float(*reinterpret_cast<const __half*>(&vals[j]));
            fval = fval / scale;
            fval = fmaxf(-FP8_E4M3_MAX, fminf(fval, FP8_E4M3_MAX));
            fp8_bytes[j] = __nv_cvt_float_to_fp8(fval, __NV_SATFINITE, __NV_E4M3);
        }

        uint2 out_vec;
        out_vec.x = *reinterpret_cast<uint32_t*>(&fp8_bytes[0]);
        out_vec.y = *reinterpret_cast<uint32_t*>(&fp8_bytes[4]);
        *reinterpret_cast<uint2*>(&output[base]) = out_vec;
    }

    for (int i = remainder_start + tid; i < num_elements; i += blockDim.x * gridDim.x) {
        float fval = __half2float(*reinterpret_cast<const __half*>(&input[i]));
        fval = fval / scale;
        fval = fmaxf(-FP8_E4M3_MAX, fminf(fval, FP8_E4M3_MAX));
        output[i] = __nv_cvt_float_to_fp8(fval, __NV_SATFINITE, __NV_E4M3);
    }
}

// ---------------------------------------------------------------------------
// Online weight quantization (kept from the original port — used by
// `Fp8Linear::load` for BF16 checkpoints).
// ---------------------------------------------------------------------------

__device__ __forceinline__ void atomicMaxFloatLocal(float* addr, float val) {
    if (val >= 0.0f) {
        atomicMax(reinterpret_cast<unsigned int*>(addr), __float_as_uint(val));
    }
}

__global__ void weight_absmax_bf16_kernel(
    const uint16_t* __restrict__ tensor,
    int num_elements,
    float* __restrict__ abs_max_out)
{
    __shared__ float smem[256];
    const int tid = threadIdx.x;
    float local_max = 0.0f;

    for (int i = blockIdx.x * blockDim.x + tid; i < num_elements;
         i += blockDim.x * gridDim.x) {
        float val = __bfloat162float(
            *reinterpret_cast<const __nv_bfloat16*>(&tensor[i]));
        float abs_val = fabsf(val);
        if (abs_val > local_max) local_max = abs_val;
    }

    smem[tid] = local_max;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && smem[tid + s] > smem[tid]) {
            smem[tid] = smem[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicMaxFloatLocal(abs_max_out, smem[0]);
    }
}

__global__ void finalize_fp8_weight_scale_kernel(
    float* __restrict__ abs_max_and_scale)
{
    float abs_max = *abs_max_and_scale;
    float scale = (abs_max > 0.0f) ? (abs_max / FP8_E4M3_MAX) : 1.0f;
    *abs_max_and_scale = scale;
}

// ---------------------------------------------------------------------------
// C entry points
// ---------------------------------------------------------------------------

extern "C" {

void scaled_fp8_quant_dynamic_bf16(
    const uint16_t* input,
    uint8_t* output,
    float* scales,
    int num_tokens,
    int hidden_dim,
    cudaStream_t stream)
{
    // Match vllm dispatch (csrc/quantization/w8a8/fp8/common.cu:382):
    //   block(std::min(hidden_size, 256))
    //   in_row_stride = input.stride(-2)  → hidden_dim for contiguous
    //   out_row_stride = out.stride(-2)   → hidden_dim for contiguous
    //   scale_ub = nullptr (None in our path)
    const int block_size = 256;
    dim3 grid(num_tokens);
    dim3 block(min(hidden_dim, block_size));
    vllm::dynamic_per_token_scaled_fp8_quant_kernel_strided<__nv_bfloat16, uint8_t>
        <<<grid, block, 0, stream>>>(
            output, scales, reinterpret_cast<const __nv_bfloat16*>(input),
            /*scale_ub=*/nullptr,
            hidden_dim, hidden_dim, hidden_dim);
}

void scaled_fp8_quant_dynamic_f16(
    const uint16_t* input,
    uint8_t* output,
    float* scales,
    int num_tokens,
    int hidden_dim,
    cudaStream_t stream)
{
    const int block_size = 256;
    dim3 grid(num_tokens);
    dim3 block(min(hidden_dim, block_size));
    vllm::dynamic_per_token_scaled_fp8_quant_kernel_strided<__half, uint8_t>
        <<<grid, block, 0, stream>>>(
            output, scales, reinterpret_cast<const __half*>(input),
            /*scale_ub=*/nullptr,
            hidden_dim, hidden_dim, hidden_dim);
}

void scaled_fp8_quant_static_bf16(
    const uint16_t* input,
    uint8_t* output,
    const float* scale,
    int num_elements,
    cudaStream_t stream)
{
    const int threads = 256;
    const int blocks = min((num_elements / 8 + threads - 1) / threads, 1024);
    scaled_fp8_quant_static_bf16_kernel<<<blocks, threads, 0, stream>>>(
        input, output, scale, num_elements);
}

void scaled_fp8_quant_static_f16(
    const uint16_t* input,
    uint8_t* output,
    const float* scale,
    int num_elements,
    cudaStream_t stream)
{
    const int threads = 256;
    const int blocks = min((num_elements / 8 + threads - 1) / threads, 1024);
    scaled_fp8_quant_static_f16_kernel<<<blocks, threads, 0, stream>>>(
        input, output, scale, num_elements);
}

void fp8_quantize_weight_bf16(
    const uint16_t* weight,
    uint8_t* output,
    float* scale_out,
    int num_elements,
    cudaStream_t stream)
{
    cudaMemsetAsync(scale_out, 0, sizeof(float), stream);
    const int threads = 256;
    const int blocks = min((num_elements + threads - 1) / threads, 1024);
    weight_absmax_bf16_kernel<<<blocks, threads, 0, stream>>>(
        weight, num_elements, scale_out);
    finalize_fp8_weight_scale_kernel<<<1, 1, 0, stream>>>(scale_out);
    const int quant_blocks = min((num_elements / 8 + threads - 1) / threads, 1024);
    scaled_fp8_quant_static_bf16_kernel<<<quant_blocks, threads, 0, stream>>>(
        weight, output, scale_out, num_elements);
}

}  // extern "C"
