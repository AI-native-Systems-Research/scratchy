// SPDX-License-Identifier: Apache-2.0
// MLX-affine int4 / int8 → BF16 dequant-at-load for Gemma-4-26B-A4B MoE.
//
// Formula (mlx_lm quantized_dequantize): w = scale * q + bias.
//   * NO zero-point (the bias absorbs it).
//   * NO AWQ column-interleave — a flat byte walk, low-nibble-is-even.
//   * scales / biases ship BF16 on disk; group_size groups run along the
//     IN (K) dimension of each [out, in] row-major weight matrix.
//
// On-disk packing (little-endian U32, viewed as bytes):
//   4-bit: one byte holds two nibbles → 2 output elems / byte.
//          out[2i+0] = scale*(byte & 0xF)       + bias
//          out[2i+1] = scale*((byte >> 4) & 0xF) + bias
//   8-bit: one byte holds one unsigned weight → 1 output elem / byte.
//
// Stacked MoE experts arrive as `[E, out, in]` row-major; flattening
// `rows = E*out, cols = in` folds the expert axis into `row`, so the flat
// `gindex = (row*cols + col)/group_size` indexes the equally-stacked
// `[E, out, in/group_size]` scale/bias buffers with no per-expert offset
// (group_size divides `in`, so groups never straddle a row boundary).
//
// Modeled on fp8_block_dequant_kernels.cu (flat 1-D grid, uint16_t bf16 store).

#include <cstdint>
#include <cuda_bf16.h>

// ---------------------------------------------------------------------------
// 4-bit: one thread per packed BYTE (emits 2 output elements).
// ---------------------------------------------------------------------------

__global__ void affine_dequant_b4_bf16_kernel(
    const uint8_t* __restrict__ w,        // packed nibbles, [rows, cols/2] bytes
    const uint16_t* __restrict__ scales,  // bf16, [rows, cols/group_size]
    const uint16_t* __restrict__ biases,  // bf16, same shape as scales
    uint16_t* __restrict__ out,           // bf16, [rows, cols]
    int rows, int cols, int group_size)
{
    const int bytes_per_row = cols >> 1;          // cols / 2
    const int total_bytes = rows * bytes_per_row;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total_bytes) return;

    const int row = i / bytes_per_row;
    const int col2 = (i - row * bytes_per_row) * 2;   // even output col

    // group index along the flattened [rows, cols] buffer. group_size is even
    // (64), so a byte's two nibbles (cols col2, col2+1) share one group.
    const int gcols = cols / group_size;
    const int gindex = row * gcols + (col2 / group_size);
    const __nv_bfloat16 s_bf = *reinterpret_cast<const __nv_bfloat16*>(&scales[gindex]);
    const __nv_bfloat16 b_bf = *reinterpret_cast<const __nv_bfloat16*>(&biases[gindex]);
    const float s = __bfloat162float(s_bf);
    const float b = __bfloat162float(b_bf);

    const uint8_t v = w[i];
    const int base = row * cols + col2;

    __nv_bfloat16 o0 = __float2bfloat16(s * float(v & 0x0F) + b);
    __nv_bfloat16 o1 = __float2bfloat16(s * float((v >> 4) & 0x0F) + b);
    out[base + 0] = *reinterpret_cast<uint16_t*>(&o0);
    out[base + 1] = *reinterpret_cast<uint16_t*>(&o1);
}

// ---------------------------------------------------------------------------
// 8-bit: one thread per element (one byte = one unsigned weight).
// ---------------------------------------------------------------------------

__global__ void affine_dequant_b8_bf16_kernel(
    const uint8_t* __restrict__ w,        // raw bytes, [rows, cols]
    const uint16_t* __restrict__ scales,  // bf16, [rows, cols/group_size]
    const uint16_t* __restrict__ biases,  // bf16, same shape as scales
    uint16_t* __restrict__ out,           // bf16, [rows, cols]
    int rows, int cols, int group_size)
{
    const int total = rows * cols;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    const int row = idx / cols;
    const int col = idx - row * cols;

    const int gcols = cols / group_size;
    const int gindex = row * gcols + (col / group_size);
    const __nv_bfloat16 s_bf = *reinterpret_cast<const __nv_bfloat16*>(&scales[gindex]);
    const __nv_bfloat16 b_bf = *reinterpret_cast<const __nv_bfloat16*>(&biases[gindex]);
    const float s = __bfloat162float(s_bf);
    const float b = __bfloat162float(b_bf);

    __nv_bfloat16 o = __float2bfloat16(s * float(w[idx]) + b);
    out[idx] = *reinterpret_cast<uint16_t*>(&o);
}

// ---------------------------------------------------------------------------
// C entry points
// ---------------------------------------------------------------------------

extern "C" {

void affine_dequant_b4_bf16(
    const uint8_t* w,
    const uint16_t* scales,
    const uint16_t* biases,
    uint16_t* out,
    int rows, int cols, int group_size,
    cudaStream_t stream)
{
    const int total_bytes = rows * (cols >> 1);
    const int threads = 256;
    const int blocks = (total_bytes + threads - 1) / threads;
    affine_dequant_b4_bf16_kernel<<<blocks, threads, 0, stream>>>(
        w, scales, biases, out, rows, cols, group_size);
}

void affine_dequant_b8_bf16(
    const uint8_t* w,
    const uint16_t* scales,
    const uint16_t* biases,
    uint16_t* out,
    int rows, int cols, int group_size,
    cudaStream_t stream)
{
    const int total = rows * cols;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    affine_dequant_b8_bf16_kernel<<<blocks, threads, 0, stream>>>(
        w, scales, biases, out, rows, cols, group_size);
}

} // extern "C"
