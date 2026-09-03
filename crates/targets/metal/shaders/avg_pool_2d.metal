// SPDX-License-Identifier: Apache-2.0
// 2-D non-overlapping average pool over a square patch grid:
//
//   in  : [ph*ph, e]  (row-major patches on a ph×ph grid)
//   out : [(ph/k)*(ph/k), e]
//   out[or,oc,:] = mean over the k×k cell in[or*k .. or*k+k, oc*k .. oc*k+k, :]
//
// Gemma3-MM's SigLIP→text projector collapses the 64×64 patch grid by
// k=4 to 16×16 = 256 tokens. Faithful to the cuda `avg_pool_2d_kernel`
// (vllm-cuda/csrc/activation_kernels.cu) and the host eval in
// scratchy-target-cuda kernels.rs: one thread per OUTPUT element.
//
// ph / k / e arrive as inline `constant uint&` args (the lowering bakes
// W::VISION_PATCH_GRID_SIDE / W::VISION_POOL_KERNEL / vision embed dim).

#include <metal_stdlib>
using namespace metal;

#define SCRATCHY_AVG_POOL_2D(NAME, T)                                            \
kernel void NAME(                                                              \
    device T* output            [[buffer(0)]],                                \
    device const T* input       [[buffer(1)]],                                \
    constant uint& ph           [[buffer(2)]],                                \
    constant uint& k            [[buffer(3)]],                                \
    constant uint& e            [[buffer(4)]],                                \
    uint gid [[thread_position_in_grid]]                                      \
) {                                                                            \
    uint ph_out = ph / k;                                                      \
    uint total = ph_out * ph_out * e;                                          \
    if (gid >= total) return;                                                  \
    uint dim     = gid % e;                                                    \
    uint out_row = gid / e;                                                    \
    uint out_r   = out_row / ph_out;                                           \
    uint out_c   = out_row % ph_out;                                           \
    uint in_r0   = out_r * k;                                                  \
    uint in_c0   = out_c * k;                                                  \
    float acc = 0.0f;                                                          \
    for (uint dr = 0; dr < k; dr++) {                                          \
        uint base = ((in_r0 + dr) * ph + in_c0) * e + dim;                     \
        for (uint dc = 0; dc < k; dc++) {                                      \
            acc += float(input[base + dc * e]);                               \
        }                                                                      \
    }                                                                          \
    output[gid] = T(acc / float(k * k));                                       \
}

SCRATCHY_AVG_POOL_2D(avg_pool_2d_f16,  half)
SCRATCHY_AVG_POOL_2D(avg_pool_2d_bf16, bfloat)
SCRATCHY_AVG_POOL_2D(avg_pool_2d_f32,  float)
