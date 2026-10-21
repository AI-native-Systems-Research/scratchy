// SPDX-License-Identifier: Apache-2.0
//
// bf16 partial rotary embedding (RoPE) for the H100, over x and out [rows, cols], cols = heads ·
// head_dim, all row-major bf16. Only the first `rot_dim` lanes of each head are rotated, in NeoX
// pairing within that width: lane d pairs with lane d + rot_dim/2, for d < rot_dim/2, and
//
//     out[d]             = x[d] · c − x[d + R/2] · s
//     out[d + R/2]       = x[d + R/2] · c + x[d] · s
//
// with c = table[row, d] and s = table[row, R/2 + d], computed in f32 and rounded once to bf16
// (round-to-nearest-even). The lanes rot_dim..head_dim pass through bit for bit: `out` is a fresh
// buffer (CUDA's colour rule), so they are copied, not left in place. qwen3.5 rotates 64 of each
// head's 256 lanes. Metal's counterpart is `shaders/rope.metal`'s pair at `(d, d + pair_off)`.
//
// `table` is the tape's rotary table, [rows, rot_dim] bf16, row r laid out [cos(0..R/2) |
// sin(0..R/2)] for token r's position (built on the host, `src/rotary.rs`).
//
// Plain CUDA, not ThunderKittens: an elementwise map. ThunderKittens' `kernels/rotary/rotary.cu`
// (at the vendored 67845f5f) rotates the full head on [B, H, N, D] tiles, with no partial width
// and no row-major [M, heads·hd] layout.
//
// One thread per 16-byte vector of 8 lanes of one head: a rotated thread (v < R/16) loads the
// 8 lanes of each half of its pairs, and their 8 table entries of cos and of sin, and stores both
// halves; a pass-through thread (R/8 <= v < head_dim/8) copies its 8 lanes as one uint4. Threads
// R/16 <= v < R/8 have nothing to do (their lanes are the second halves the rotated threads
// write). No atomics, no shared memory, a fixed order: the same input gives the same bits.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <climits>
#include <cstdint>

// The launcher's return code that is not a `cudaError_t` (which is non-negative). `src/kernels.rs`
// mirrors it.
#define SCRATCHY_CUDA_ERR_SHAPE (-1)

namespace {

// bf16 values per 16-byte vector: every load and store is one `uint4`.
constexpr int VEC = 8;
constexpr int THREADS = 256;
// `rot_dim` is a multiple of this: each half (R/2 lanes) is whole vectors (`step::ROPE_BF16_ROT_MULTIPLE`).
constexpr uint32_t ROT_MULTIPLE = 2 * VEC;

__device__ __forceinline__ void load8(const __nv_bfloat16* p, float (&f)[VEC]) {
    const uint4 raw = *reinterpret_cast<const uint4*>(p);
    const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&raw);
#pragma unroll
    for (int i = 0; i < VEC / 2; ++i) {
        const float2 v = __bfloat1622float2(h[i]);
        f[2 * i] = v.x;
        f[2 * i + 1] = v.y;
    }
}

__device__ __forceinline__ void store8(__nv_bfloat16* p, const float (&f)[VEC]) {
    uint4 raw;
    __nv_bfloat162* h = reinterpret_cast<__nv_bfloat162*>(&raw);
#pragma unroll
    for (int i = 0; i < VEC / 2; ++i) {
        h[i] = __floats2bfloat162_rn(f[2 * i], f[2 * i + 1]);
    }
    *reinterpret_cast<uint4*>(p) = raw;
}

// `vecs` = rows · cols / 8: one thread per vector of the output. Each head is `head_vecs` vectors,
// the rotated pairs' first halves the first `half_vecs` of them.
__global__ void __launch_bounds__(THREADS)
rope_bf16_kernel(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ table,
                 __nv_bfloat16* __restrict__ out, int vecs, int cols, int head_vecs,
                 int half_vecs, int rot_dim) {
    const int g = blockIdx.x * THREADS + threadIdx.x;
    if (g >= vecs) {
        return;
    }
    const int v = g % head_vecs;  // the vector's place within its head
    const size_t at = size_t(g) * VEC;
    if (v >= 2 * half_vecs) {
        // A pass-through lane group: copied bit for bit.
        *reinterpret_cast<uint4*>(out + at) = *reinterpret_cast<const uint4*>(x + at);
        return;
    }
    if (v >= half_vecs) {
        // A second half: the thread at `v - half_vecs` writes it.
        return;
    }
    const int half = rot_dim / 2;
    const size_t row = at / size_t(cols);
    const __nv_bfloat16* trow = table + row * size_t(rot_dim) + size_t(v) * VEC;
    float x0[VEC], x1[VEC], c[VEC], s[VEC];
    load8(x + at, x0);
    load8(x + at + half, x1);
    load8(trow, c);
    load8(trow + half, s);
    float y0[VEC], y1[VEC];
#pragma unroll
    for (int i = 0; i < VEC; ++i) {
        y0[i] = x0[i] * c[i] - x1[i] * s[i];
        y1[i] = x1[i] * c[i] + x0[i] * s[i];
    }
    store8(out + at, y0);
    store8(out + at + half, y1);
}

}  // namespace

// 0 = launched; SCRATCHY_CUDA_ERR_SHAPE = shape refused, nothing launched; any other value = the
// cudaError_t cudaLaunchKernel returned for THIS launch. Nothing here reads or clears the
// runtime's last-error state.
//
// Refused before any pointer is touched: `rows`, `cols`, `head_dim` or `rot_dim` zero; `rot_dim`
// not a multiple of 16 or above `head_dim`; `head_dim` not a multiple of 8; `cols` not a multiple
// of `head_dim`; `rows · cols` above INT_MAX. `x`, `table` and `out` must be 16-byte-aligned
// device pointers to `rows·cols`, `rows·rot_dim` and `rows·cols` bf16, `out` overlapping neither
// input. No shared memory, so no one-time setup. The launch is asynchronous on `stream`; the
// caller owns it and synchronizes.
extern "C" int scratchy_cuda_rope_bf16(const void* x, const void* table, void* out, uint32_t rows,
                                       uint32_t cols, uint32_t head_dim, uint32_t rot_dim,
                                       void* stream /* cudaStream_t; null = default stream */) {
    if (rows == 0 || cols == 0 || head_dim == 0 || rot_dim == 0 || rot_dim % ROT_MULTIPLE != 0 ||
        rot_dim > head_dim || head_dim % VEC != 0 || cols % head_dim != 0 ||
        uint64_t(rows) * uint64_t(cols) > uint64_t(INT_MAX)) {
        return SCRATCHY_CUDA_ERR_SHAPE;
    }
    const __nv_bfloat16* xp = static_cast<const __nv_bfloat16*>(x);
    const __nv_bfloat16* tp = static_cast<const __nv_bfloat16*>(table);
    __nv_bfloat16* op = static_cast<__nv_bfloat16*>(out);
    int vecs = static_cast<int>(uint64_t(rows) * uint64_t(cols) / VEC);
    int icols = static_cast<int>(cols);
    int head_vecs = static_cast<int>(head_dim / VEC);
    int half_vecs = static_cast<int>(rot_dim / ROT_MULTIPLE);
    int irot = static_cast<int>(rot_dim);
    void* args[] = {&xp, &tp, &op, &vecs, &icols, &head_vecs, &half_vecs, &irot};
    const dim3 grid((vecs + THREADS - 1) / THREADS), block(THREADS);
    return static_cast<int>(cudaLaunchKernel(rope_bf16_kernel, grid, block, args, 0,
                                             static_cast<cudaStream_t>(stream)));
}
