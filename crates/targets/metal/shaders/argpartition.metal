// SPDX-License-Identifier: Apache-2.0
//
// Faithful port of MLX's `block_sort` argsort entry point from
// `mlx/backend/metal/kernels/sort.h` (lines 22-364) — sorts each
// row of a 2D contiguous tensor ascending and returns the sorted
// indices as uint32.
//
// Why "argpartition" in the file name: MoE router top-k is
// expressed in MLX-LM as
//   inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]   (qwen3_moe)
//   inds = mx.argpartition(-gates, kth=k-1, axis=-1)[..., :k]  (mixtral / qwen2_moe)
// Both reduce to "give me the indices of the top-k entries by
// gate value, sorted." A full ascending sort with a trailing slice
// of size `top_k` satisfies both forms (positions are sorted, NaN-
// padded slots — when N_PER_BLOCK > num_experts — land at the
// extreme right per the LessThan-NaN convention from MLX sort.h:46).
//
// Symbol naming follows MLX's instantiation macro
// `instantiate_block_sort` with the c-prefix (contiguous) and
// `arg_block_sort_<itname>_<otname>_bn<bn>_tn<tn>`:
//
//   c_arg_block_sort_bfloat16_uint32_bn32_tn4
//   c_arg_block_sort_float16_uint32_bn32_tn4
//   c_arg_block_sort_float32_uint32_bn32_tn4
//
// bn=32 tn=4 ⇒ N_PER_BLOCK=128 covers Mixtral (E=8), Qwen2-MoE
// (E=60), Qwen3-MoE (E=128); bn=64 tn=4 covers E=256.

#include <metal_simdgroup>
#include <metal_stdlib>
#include "baked.h"
#include "block_sort.h"

using namespace metal;

// `ArgsortConstants`, compiled in: the sorted axis's size and the strides
// MLX's `block_sort` takes (contiguous rows).
SCRATCHY_CONSTANT(int, SORT_SIZE, 0);
SCRATCHY_CONSTANT(int, SORT_IN_STRIDE, 1);
SCRATCHY_CONSTANT(int, SORT_OUT_STRIDE, 2);
SCRATCHY_CONSTANT(int, SORT_IN_SEGMENT_STRIDE, 3);
SCRATCHY_CONSTANT(int, SORT_OUT_SEGMENT_STRIDE, 4);

using namespace mlx_sort;

template <
    typename T,
    typename U,
    bool ARG_SORT,
    short BLOCK_THREADS,
    short N_PER_THREAD>
[[kernel, max_total_threads_per_threadgroup(BLOCK_THREADS)]] void
block_sort(
    const device T* inp [[buffer(0)]],
    device U* out [[buffer(1)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) {
  using sort_kernel =
      KernelMergeSort<T, U, ARG_SORT, BLOCK_THREADS, N_PER_THREAD>;
  using ValT = typename sort_kernel::ValT;
  using IdxT = typename sort_kernel::IdxT;

  threadgroup ValT tgp_vals[sort_kernel::N_PER_BLOCK];
  if (ARG_SORT) {
    threadgroup IdxT tgp_idxs[sort_kernel::N_PER_BLOCK];
    sort_kernel::block_sort_impl(
        inp,
        out,
        SORT_SIZE,
        SORT_IN_STRIDE,
        SORT_OUT_STRIDE,
        SORT_IN_SEGMENT_STRIDE,
        SORT_OUT_SEGMENT_STRIDE,
        tgp_vals,
        tgp_idxs,
        tid,
        lid);
  } else {
    sort_kernel::block_sort_impl(
        inp,
        out,
        SORT_SIZE,
        SORT_IN_STRIDE,
        SORT_OUT_STRIDE,
        SORT_IN_SEGMENT_STRIDE,
        SORT_OUT_SEGMENT_STRIDE,
        tgp_vals,
        nullptr,
        tid,
        lid);
  }
}

// `bfloat` lacks numeric_limits<>::quiet_NaN() / lowest() under the
// current Metal toolchain. Cast to bf16's u16 bit pattern is the
// MLX dance (bf16.h:Limits): for now this kernel handles bf16 by
// upcasting load → float, sort as float, store back as bf16 — done
// by an outer wrapper layer in the lowering arm.

#define INSTANTIATE_ARG_SORT(itname, itype, bn, tn)                 \
  SCRATCHY_KERNEL(c_arg_block_sort_##itname##_uint32_bn##bn##_tn##tn, \
                  block_sort<itype, uint, true, bn, tn>)

INSTANTIATE_ARG_SORT(float32, float, 32, 4)
INSTANTIATE_ARG_SORT(float16, half, 32, 4)
INSTANTIATE_ARG_SORT(bfloat16, bfloat, 32, 4)
INSTANTIATE_ARG_SORT(float32, float, 64, 4)
// bn=64 tn=4 ⇒ N_PER_BLOCK=256: Qwen3.5-MoE router (E=256). The bn=32
// variants above cap at 128 experts.
INSTANTIATE_ARG_SORT(float16, half, 64, 4)
INSTANTIATE_ARG_SORT(bfloat16, bfloat, 64, 4)
