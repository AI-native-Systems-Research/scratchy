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
//   c_arg_block_sort_uint32_uint32_bn32_tn4   (used by _gather_sort
//                                              expert-token reorder)
//
// bn=32 tn=4 ⇒ N_PER_BLOCK=128 covers Mixtral (E=8), Qwen2-MoE
// (E=60), Qwen3-MoE (E=128). Larger MoE configs would need a
// bn=64 tn=4 instantiation.

#include <metal_simdgroup>
#include <metal_stdlib>

using namespace metal;

#define MLX_MTL_CONST static constant constexpr const
#define MLX_MTL_LOOP_UNROLL _Pragma("clang loop unroll(full)")

template <typename T>
struct Limits {
  static constant constexpr const T max = numeric_limits<T>::max();
};

// `bfloat` is missing `numeric_limits<>::max() / quiet_NaN()` under
// the current Metal toolchain; specialize using known bit patterns
// reinterpreted through `as_type`. `0x7f7f` is max-finite bf16
// (S=0 E=0xfe M=0x7f); `0x7fc0` is quiet NaN (S=0 E=0xff M=0x40).
// MLX uses the same pair in its `bf16.h::Limits<bfloat16_t>`.
template <>
struct Limits<bfloat> {
  static constant constexpr const bfloat max = as_type<bfloat>(ushort(0x7f7f));
};

template <typename T, typename = void>
struct Init {
  static constant constexpr const T v = Limits<T>::max;
};

template <typename T>
struct Init<T, metal::enable_if_t<metal::is_floating_point_v<T>>> {
  static constant constexpr const T v = numeric_limits<T>::quiet_NaN();
};

// bfloat specialization: same NaN bit pattern as the float SFINAE
// branch but constructed via `as_type` so it doesn't depend on
// `numeric_limits<bfloat>`.
template <>
struct Init<bfloat, void> {
  static constant constexpr const bfloat v = as_type<bfloat>(ushort(0x7fc0));
};

// `LessThan<T>` — sort.h:39-58. NaN treated as "greater than
// anything finite" so NaN-padding lands at the end of the sorted
// row (right-tail), leaving the trailing top_k indices free of
// pad contamination.
template <typename T>
struct LessThan {
  static constant constexpr const T init = Init<T>::v;
  METAL_FUNC bool operator()(T a, T b) const {
    if constexpr (metal::is_floating_point_v<T>) {
      bool an = metal::isnan(a);
      bool bn = metal::isnan(b);
      if (an | bn) {
        return (!an) & bn;
      }
    }
    return a < b;
  }
};

template <typename T>
METAL_FUNC void thread_swap(thread T& a, thread T& b) {
  T w = a;
  a = b;
  b = w;
}

template <
    typename ValT,
    typename IdxT,
    bool ARG_SORT,
    short N_PER_THREAD,
    typename CompareOp>
struct ThreadSort {
  static METAL_FUNC void sort(
      thread ValT (&vals)[N_PER_THREAD],
      thread IdxT (&idxs)[N_PER_THREAD]) {
    CompareOp op;
    MLX_MTL_LOOP_UNROLL
    for (short i = 0; i < N_PER_THREAD; ++i) {
      MLX_MTL_LOOP_UNROLL
      for (short j = i & 1; j < N_PER_THREAD - 1; j += 2) {
        if (op(vals[j + 1], vals[j])) {
          thread_swap(vals[j + 1], vals[j]);
          if (ARG_SORT) {
            thread_swap(idxs[j + 1], idxs[j]);
          }
        }
      }
    }
  }
};

template <
    typename ValT,
    typename IdxT,
    bool ARG_SORT,
    short BLOCK_THREADS,
    short N_PER_THREAD,
    typename CompareOp>
struct BlockMergeSort {
  using thread_sort_t = ThreadSort<ValT, IdxT, ARG_SORT, N_PER_THREAD, CompareOp>;
  static METAL_FUNC int merge_partition(
      const threadgroup ValT* As,
      const threadgroup ValT* Bs,
      short A_sz,
      short B_sz,
      short sort_md) {
    CompareOp op;
    short A_st = max(0, sort_md - B_sz);
    short A_ed = min(sort_md, A_sz);
    while (A_st < A_ed) {
      short md = A_st + (A_ed - A_st) / 2;
      auto a = As[md];
      auto b = Bs[sort_md - 1 - md];
      if (op(b, a)) {
        A_ed = md;
      } else {
        A_st = md + 1;
      }
    }
    return A_ed;
  }

  static METAL_FUNC void merge_step(
      const threadgroup ValT* As,
      const threadgroup ValT* Bs,
      const threadgroup IdxT* As_idx,
      const threadgroup IdxT* Bs_idx,
      short A_sz,
      short B_sz,
      thread ValT (&vals)[N_PER_THREAD],
      thread IdxT (&idxs)[N_PER_THREAD]) {
    CompareOp op;
    short a_idx = 0;
    short b_idx = 0;
    for (int i = 0; i < N_PER_THREAD; ++i) {
      auto a = (a_idx < A_sz) ? As[a_idx] : ValT(CompareOp::init);
      auto b = (b_idx < B_sz) ? Bs[b_idx] : ValT(CompareOp::init);
      bool pred = (b_idx < B_sz) && (a_idx >= A_sz || op(b, a));
      vals[i] = pred ? b : a;
      if (ARG_SORT) {
        if (pred) {
          idxs[i] = Bs_idx[b_idx];
        } else {
          idxs[i] = (a_idx < A_sz) ? As_idx[a_idx] : IdxT(0);
        }
      }
      b_idx += short(pred);
      a_idx += short(!pred);
    }
  }

  static METAL_FUNC void sort(
      threadgroup ValT* tgp_vals,
      threadgroup IdxT* tgp_idxs,
      int size_sorted_axis,
      uint3 lid) {
    int idx = lid.x * N_PER_THREAD;

    thread ValT thread_vals[N_PER_THREAD];
    thread IdxT thread_idxs[N_PER_THREAD];
    for (int i = 0; i < N_PER_THREAD; ++i) {
      thread_vals[i] = tgp_vals[idx + i];
      if (ARG_SORT) {
        thread_idxs[i] = tgp_idxs[idx + i];
      }
    }

    if (idx < size_sorted_axis) {
      thread_sort_t::sort(thread_vals, thread_idxs);
    }

    for (int merge_threads = 2; merge_threads <= BLOCK_THREADS;
         merge_threads *= 2) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (int i = 0; i < N_PER_THREAD; ++i) {
        tgp_vals[idx + i] = thread_vals[i];
        if (ARG_SORT) {
          tgp_idxs[idx + i] = thread_idxs[i];
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      int merge_group = lid.x / merge_threads;
      int merge_lane = lid.x % merge_threads;

      int sort_sz = N_PER_THREAD * merge_threads;
      int sort_st = N_PER_THREAD * merge_threads * merge_group;

      int A_st = sort_st;
      int A_ed = sort_st + sort_sz / 2;
      int B_st = sort_st + sort_sz / 2;
      int B_ed = sort_st + sort_sz;

      const threadgroup ValT* As = tgp_vals + A_st;
      const threadgroup ValT* Bs = tgp_vals + B_st;
      int A_sz = A_ed - A_st;
      int B_sz = B_ed - B_st;

      int sort_md = N_PER_THREAD * merge_lane;
      int partition = merge_partition(As, Bs, A_sz, B_sz, sort_md);

      As += partition;
      Bs += sort_md - partition;

      A_sz -= partition;
      B_sz -= sort_md - partition;

      const threadgroup IdxT* As_idx =
          ARG_SORT ? tgp_idxs + A_st + partition : nullptr;
      const threadgroup IdxT* Bs_idx =
          ARG_SORT ? tgp_idxs + B_st + sort_md - partition : nullptr;

      merge_step(As, Bs, As_idx, Bs_idx, A_sz, B_sz, thread_vals, thread_idxs);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = 0; i < N_PER_THREAD; ++i) {
      tgp_vals[idx + i] = thread_vals[i];
      if (ARG_SORT) {
        tgp_idxs[idx + i] = thread_idxs[i];
      }
    }
  }
};

template <
    typename T,
    typename U,
    bool ARG_SORT,
    short BLOCK_THREADS,
    short N_PER_THREAD,
    typename CompareOp = LessThan<T>>
struct KernelMergeSort {
  using ValT = T;
  using IdxT = uint;
  using block_merge_sort_t = BlockMergeSort<
      ValT,
      IdxT,
      ARG_SORT,
      BLOCK_THREADS,
      N_PER_THREAD,
      CompareOp>;

  MLX_MTL_CONST short N_PER_BLOCK = BLOCK_THREADS * N_PER_THREAD;

  static METAL_FUNC void block_sort_impl(
      const device T* inp,
      device U* out,
      const constant int& size_sorted_axis,
      const constant int& in_stride_sorted_axis,
      const constant int& out_stride_sorted_axis,
      const constant int& in_stride_segment_axis,
      const constant int& out_stride_segment_axis,
      threadgroup ValT* tgp_vals,
      threadgroup IdxT* tgp_idxs,
      uint3 tid,
      uint3 lid) {
    inp += tid.y * in_stride_segment_axis;
    out += tid.y * out_stride_segment_axis;

    for (short i = lid.x; i < N_PER_BLOCK; i += BLOCK_THREADS) {
      tgp_vals[i] = i < size_sorted_axis ? inp[i * in_stride_sorted_axis]
                                         : ValT(CompareOp::init);
      if (ARG_SORT) {
        tgp_idxs[i] = i;
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    block_merge_sort_t::sort(tgp_vals, tgp_idxs, size_sorted_axis, lid);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int i = lid.x; i < size_sorted_axis; i += BLOCK_THREADS) {
      if (ARG_SORT) {
        out[i * out_stride_sorted_axis] = tgp_idxs[i];
      } else {
        out[i * out_stride_sorted_axis] = tgp_vals[i];
      }
    }
  }
};

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
    const constant int& size_sorted_axis [[buffer(2)]],
    const constant int& in_stride_sorted_axis [[buffer(3)]],
    const constant int& out_stride_sorted_axis [[buffer(4)]],
    const constant int& in_stride_segment_axis [[buffer(5)]],
    const constant int& out_stride_segment_axis [[buffer(6)]],
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
        size_sorted_axis,
        in_stride_sorted_axis,
        out_stride_sorted_axis,
        in_stride_segment_axis,
        out_stride_segment_axis,
        tgp_vals,
        tgp_idxs,
        tid,
        lid);
  } else {
    sort_kernel::block_sort_impl(
        inp,
        out,
        size_sorted_axis,
        in_stride_sorted_axis,
        out_stride_sorted_axis,
        in_stride_segment_axis,
        out_stride_segment_axis,
        tgp_vals,
        nullptr,
        tid,
        lid);
  }
}

// `uint32` keys lack simd comparisons in MSL but the LessThan
// branch above only uses simple `<` so the integer variant works
// untouched. It's needed by `_gather_sort` (switch_layers.py:12) —
// the prefill path argsorts the flattened expert-id list to bring
// every token's experts into contiguous-by-expert order for the
// per-expert GEMM batch.
//
// `bfloat` lacks numeric_limits<>::quiet_NaN() / lowest() under the
// current Metal toolchain. Cast to bf16's u16 bit pattern is the
// MLX dance (bf16.h:Limits): for now this kernel handles bf16 by
// upcasting load → float, sort as float, store back as bf16 — done
// by an outer wrapper layer in the lowering arm.

#define INSTANTIATE_ARG_SORT(itname, itype, bn, tn)                    \
  template [[host_name("c_arg_block_sort_" #itname "_uint32_bn" #bn    \
                       "_tn" #tn)]] [[kernel]] void                    \
  block_sort<itype, uint, true, bn, tn>(                               \
      const device itype* inp [[buffer(0)]],                           \
      device uint* out [[buffer(1)]],                                  \
      const constant int& size_sorted_axis [[buffer(2)]],              \
      const constant int& in_stride_sorted_axis [[buffer(3)]],         \
      const constant int& out_stride_sorted_axis [[buffer(4)]],        \
      const constant int& in_stride_segment_axis [[buffer(5)]],        \
      const constant int& out_stride_segment_axis [[buffer(6)]],       \
      uint3 tid [[threadgroup_position_in_grid]],                      \
      uint3 lid [[thread_position_in_threadgroup]]);

INSTANTIATE_ARG_SORT(float32, float, 32, 4)
INSTANTIATE_ARG_SORT(float16, half, 32, 4)
INSTANTIATE_ARG_SORT(bfloat16, bfloat, 32, 4)
INSTANTIATE_ARG_SORT(uint32, uint, 32, 4)
INSTANTIATE_ARG_SORT(float32, float, 64, 4)
INSTANTIATE_ARG_SORT(uint32, uint, 64, 4)
// bn=64 tn=4 ⇒ N_PER_BLOCK=256: Qwen3.5-MoE router (E=256). The bn=32
// variants above cap at 128 experts.
INSTANTIATE_ARG_SORT(float16, half, 64, 4)
INSTANTIATE_ARG_SORT(bfloat16, bfloat, 64, 4)
