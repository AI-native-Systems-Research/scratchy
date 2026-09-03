// SPDX-License-Identifier: Apache-2.0
// BitsAndBytes NF4/FP4 dequantization kernels.
//
// Each packed U8 byte contains two 4-bit values (lo nibble = even index,
// hi nibble = odd index). Dequantization: out[i] = code[nibble] * absmax[i / blocksize]

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

// One thread per packed byte → produces 2 output elements.
template <typename OutT>
__global__ void dequantize_nf4_kernel(
    const uint8_t* __restrict__ packed,
    const float* __restrict__ absmax,
    const float* __restrict__ code,
    OutT* __restrict__ out,
    int64_t num_packed,
    int blocksize) {
  int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_packed) return;

  uint8_t byte = packed[idx];
  int lo = byte & 0x0F;
  int hi = (byte >> 4) & 0x0F;

  // BNB convention: hi nibble = first element, lo nibble = second element.
  int64_t out_idx = idx * 2;
  int block_0 = (int)(out_idx / blocksize);
  int block_1 = (int)((out_idx + 1) / blocksize);

  float val_0 = code[hi] * absmax[block_0];
  float val_1 = code[lo] * absmax[block_1];

  out[out_idx] = (OutT)val_0;
  out[out_idx + 1] = (OutT)val_1;
}

extern "C" void dequantize_nf4_bf16(
    const uint8_t* packed,
    const float* absmax,
    const float* code,
    __nv_bfloat16* out,
    int64_t num_packed,
    int blocksize,
    cudaStream_t stream) {
  int threads = 256;
  int blocks = (int)((num_packed + threads - 1) / threads);
  dequantize_nf4_kernel<<<blocks, threads, 0, stream>>>(
      packed, absmax, code, out, num_packed, blocksize);
}

extern "C" void dequantize_nf4_f16(
    const uint8_t* packed,
    const float* absmax,
    const float* code,
    __half* out,
    int64_t num_packed,
    int blocksize,
    cudaStream_t stream) {
  int threads = 256;
  int blocks = (int)((num_packed + threads - 1) / threads);
  dequantize_nf4_kernel<<<blocks, threads, 0, stream>>>(
      packed, absmax, code, out, num_packed, blocksize);
}
