// SPDX-License-Identifier: Apache-2.0
// Rearrange NCCL all-gather output from dim=0 to last-dim layout.
//
// NCCL all-gather concatenates along dim=0:
//   input [N, S] per rank → gathered [world_size * N, S]
//
// Python does: reshape to [W, N, S] → movedim(0, 1) → reshape [N, W*S]
// This kernel fuses that into a single pass.

#include <cstdint>

template <typename T>
__global__ void gather_last_dim_kernel(
    const T* __restrict__ src,   // [world_size * N, S]
    T* __restrict__ dst,         // [N, world_size * S]
    int N,
    int S,                       // shard size per rank
    int world_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * world_size * S;
    if (idx >= total) return;

    // Output layout: [N, world_size * S]
    int vocab = world_size * S;
    int row = idx / vocab;          // request index
    int col = idx % vocab;          // position in full vocab
    int rank = col / S;
    int s = col % S;                // position within shard

    // Source layout: [world_size * N, S] — rank r's block starts at row r*N.
    int src_idx = (rank * N + row) * S + s;
    dst[idx] = src[src_idx];
}

extern "C" {

void gather_last_dim_f16(
    const void* src, void* dst,
    int N, int S, int world_size, void* stream)
{
    int total = N * world_size * S;
    int block = 256;
    int grid = (total + block - 1) / block;
    gather_last_dim_kernel<uint16_t><<<grid, block, 0, (cudaStream_t)stream>>>(
        (const uint16_t*)src, (uint16_t*)dst, N, S, world_size);
}

void gather_last_dim_bf16(
    const void* src, void* dst,
    int N, int S, int world_size, void* stream)
{
    int total = N * world_size * S;
    int block = 256;
    int grid = (total + block - 1) / block;
    gather_last_dim_kernel<uint16_t><<<grid, block, 0, (cudaStream_t)stream>>>(
        (const uint16_t*)src, (uint16_t*)dst, N, S, world_size);
}

void gather_last_dim_f32(
    const void* src, void* dst,
    int N, int S, int world_size, void* stream)
{
    int total = N * world_size * S;
    int block = 256;
    int grid = (total + block - 1) / block;
    gather_last_dim_kernel<float><<<grid, block, 0, (cudaStream_t)stream>>>(
        (const float*)src, (float*)dst, N, S, world_size);
}

}  // extern "C"
