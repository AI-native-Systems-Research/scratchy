// SPDX-License-Identifier: Apache-2.0
//
// bf16 attention output-gate split for the H100 (qwen3.5's `attn_output_gate`): q_proj is twice as
// wide as the query, each head's 2·head_dim outputs laid out [query(head_dim) | gate(head_dim)].
// For head h and lane d, over qg [rows, 2·half_cols] and q, gate [rows, half_cols]:
//
//     q[r, h·hd + d]    = qg[r, 2h·hd + d]
//     gate[r, h·hd + d] = qg[r, 2h·hd + hd + d]
//
// Pure copies, so bit-exact. Metal's counterpart is `shaders/gate_split.metal`; transformers'
// `Qwen3_5Attention` is `torch.chunk(q_proj(x).view(..., head_dim*2), 2, dim=-1)`.
//
// Plain CUDA, not ThunderKittens: a per-element copy. One thread per 16-byte vector of 8 lanes of
// one output row; `head_dim` is a multiple of 8, so a vector never straddles a head, and each
// thread copies one vector of the query and one of the gate.

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

// `vecs` = rows · half_cols / 8, one thread per vector of `q` (and of `gate`).
__global__ void __launch_bounds__(THREADS)
gate_split_bf16_kernel(const uint4* __restrict__ qg, uint4* __restrict__ q,
                       uint4* __restrict__ gate, int vecs, int row_vecs, int head_vecs) {
    const int g = blockIdx.x * THREADS + threadIdx.x;
    if (g >= vecs) {
        return;
    }
    const int row = g / row_vecs;
    const int c = g % row_vecs;  // the vector's place within the output row
    const int h = c / head_vecs;
    const int d = c % head_vecs;
    // The input row is twice as wide, and head h's query starts at 2h·hd.
    const size_t base = size_t(row) * size_t(2 * row_vecs) + size_t(2 * h * head_vecs + d);
    q[g] = qg[base];
    gate[g] = qg[base + head_vecs];
}

}  // namespace

// 0 = launched; SCRATCHY_CUDA_ERR_SHAPE = shape refused, nothing launched; any other value = the
// cudaError_t cudaLaunchKernel returned for THIS launch. Nothing here reads or clears the
// runtime's last-error state.
//
// Refused before any pointer is touched: `rows`, `half_cols` or `head_dim` zero; `head_dim` not a
// multiple of 8; `half_cols` not a multiple of `head_dim`; `rows · 2·half_cols` above INT_MAX.
// `qg`, `q` and `gate` must be 16-byte-aligned device pointers to `rows·2·half_cols`,
// `rows·half_cols` and `rows·half_cols` bf16, no two overlapping. No shared memory, so no one-time
// setup. The launch is asynchronous on `stream`; the caller owns it and synchronizes.
extern "C" int scratchy_cuda_gate_split_bf16(const void* qg, void* q, void* gate, uint32_t rows,
                                             uint32_t half_cols, uint32_t head_dim,
                                             void* stream /* cudaStream_t; null = default */) {
    if (rows == 0 || half_cols == 0 || head_dim == 0 || head_dim % VEC != 0 ||
        half_cols % head_dim != 0 ||
        uint64_t(rows) * 2 * uint64_t(half_cols) > uint64_t(INT_MAX)) {
        return SCRATCHY_CUDA_ERR_SHAPE;
    }
    const uint4* qgp = static_cast<const uint4*>(qg);
    uint4* qp = static_cast<uint4*>(q);
    uint4* gp = static_cast<uint4*>(gate);
    int vecs = static_cast<int>(uint64_t(rows) * uint64_t(half_cols) / VEC);
    int row_vecs = static_cast<int>(half_cols / VEC);
    int head_vecs = static_cast<int>(head_dim / VEC);
    void* args[] = {&qgp, &qp, &gp, &vecs, &row_vecs, &head_vecs};
    const dim3 grid((vecs + THREADS - 1) / THREADS), block(THREADS);
    return static_cast<int>(cudaLaunchKernel(gate_split_bf16_kernel, grid, block, args, 0,
                                             static_cast<cudaStream_t>(stream)));
}
