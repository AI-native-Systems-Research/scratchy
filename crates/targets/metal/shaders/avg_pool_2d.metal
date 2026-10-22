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
// Compiled in: the patch grid's side `AVG_POOL_PH` (W::VISION_PATCH_GRID_SIDE), the pool's
// `AVG_POOL_K` (W::VISION_POOL_KERNEL) and the embed dim `AVG_POOL_E`.

#include <metal_stdlib>
#include "baked.h"
using namespace metal;

SCRATCHY_CONSTANT(uint, AVG_POOL_PH, 0);
SCRATCHY_CONSTANT(uint, AVG_POOL_K, 1);
SCRATCHY_CONSTANT(uint, AVG_POOL_E, 2);
static_assert(AVG_POOL_K != 0u && AVG_POOL_PH % AVG_POOL_K == 0u,
              "the pool tiles the patch grid");

template <typename T>
[[kernel]] void avg_pool_2d(
    device T* output      [[buffer(0)]],
    device const T* input [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    constexpr uint ph = AVG_POOL_PH, k = AVG_POOL_K, e = AVG_POOL_E;
    constexpr uint ph_out = ph / k;
    constexpr uint total = ph_out * ph_out * e;
    if (gid >= total) return;
    uint dim     = gid % e;
    uint out_row = gid / e;
    uint out_r   = out_row / ph_out;
    uint out_c   = out_row % ph_out;
    uint in_r0   = out_r * k;
    uint in_c0   = out_c * k;
    float acc = 0.0f;
    for (uint dr = 0; dr < k; dr++) {
        uint base = ((in_r0 + dr) * ph + in_c0) * e + dim;
        for (uint dc = 0; dc < k; dc++) {
            acc += float(input[base + dc * e]);
        }
    }
    output[gid] = T(acc / float(k * k));
}

SCRATCHY_KERNEL(avg_pool_2d_f16, avg_pool_2d<half>)
SCRATCHY_KERNEL(avg_pool_2d_bf16, avg_pool_2d<bfloat>)
SCRATCHY_KERNEL(avg_pool_2d_f32, avg_pool_2d<float>)
