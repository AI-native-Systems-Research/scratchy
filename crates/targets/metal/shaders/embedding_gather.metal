// SPDX-License-Identifier: Apache-2.0
// Row gather by a runtime u32 index buffer:
//
//   out[i, :] = src[indices[i], :]
//
// over a `[rows, width]` activation tile. Mirrors the cuda
// `kernels::embedding_gather` row semantics (scratchy-target-cuda
// kernels.rs) for `Instruction::EmbeddingGather` — Qwen2.5-VL permutes
// merged-token groups into window-contiguous order on encoder entry
// (indices = `vision_window_index`) and unpermutes the merger output
// (indices = `vision_reverse_indices`). The DSL reshapes so each row
// is one whole merge group; this kernel never sees the grouping.
//
// One thread per output ELEMENT: gid → (row = gid / width,
// col = gid % width). `n` = rows·width (m-scaled to the live token
// count by the dispatcher; the indices buffer holds exactly the live
// row count, so no guard beyond `gid >= n` is needed).

#include <metal_stdlib>
using namespace metal;

kernel void embedding_gather_rows_f16(
    device half* output [[buffer(0)]],
    device const half* input [[buffer(1)]],
    device const uint* indices [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    constant uint& width [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;

    uint row = gid / width;
    uint col = gid % width;
    output[gid] = input[indices[row] * width + col];
}

kernel void embedding_gather_rows_bf16(
    device bfloat* output [[buffer(0)]],
    device const bfloat* input [[buffer(1)]],
    device const uint* indices [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    constant uint& width [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;

    uint row = gid / width;
    uint col = gid % width;
    output[gid] = input[indices[row] * width + col];
}

kernel void embedding_gather_rows_f32(
    device float* output [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device const uint* indices [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    constant uint& width [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;

    uint row = gid / width;
    uint col = gid % width;
    output[gid] = input[indices[row] * width + col];
}
