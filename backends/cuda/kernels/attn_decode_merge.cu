// SPDX-License-Identifier: Apache-2.0
//
// The merge of the split fp8kv decode (FP8 S8 W6b, flash-decoding's second launch): the partials
// `attn_decode_fp8kv.cu` and `attn_decode_d512_fp8kv.cu` write when more than one split of a kv
// head has keys, merged into the decode's bf16 output row.
//
// One CTA per kv head, 4 warps, warp w its `HEAD_DIM / 4` columns of the head's 16 rows (rows at
// or past `hr` are computed and not stored, as in the decode). It recomputes the decode's split
// rule from the sequence, the window, `splits` and `min_tiles` (`used` splits with keys; one used
// split means the decode wrote `out` itself, and the CTA leaves at once). Then, per row, in split
// order 0 .. used − 1, with no atomics:
//   - `M` the max of the splits' maxima (`warp::max`, split 0 first);
//   - each split's rescale `r_s = exp2((m_s − M) · scale_log2)`, exactly 1 where `m_s = M`
//     (`rescale_exact1`, the decode's own rescale);
//   - `l = Σ l_s · r_s` and `o = Σ o_s · r_s`, each added in split order;
//   - `o / l`, rounded once to bf16, as the decode's own merge ends.
// So the output is a function of the partials and `(n, S, min_tiles)` alone.

#include "kittens.cuh"
#include "scratchy_ops.cuh"

#include <climits>
#include <cmath>
#include <cstdint>

using namespace kittens;

// Launcher return codes that are not a `cudaError_t`. `src/kernels.rs` mirrors these two values.
#define SCRATCHY_CUDA_ERR_SHAPE (-1)
#define SCRATCHY_CUDA_ERR_HOST (-2)

namespace {

// The decodes' rows per kv head (`step::ATTN_DECODE_BF16_MAX_GROUP`), K/V rows per tile, and
// `step::CUDA_MAX_BLOCKS_PER_SEQ`.
constexpr int MAX_GROUP = 16;
constexpr int KV_HEIGHT = 64;
constexpr uint32_t BLOCK_SIZE = 64;
constexpr uint32_t MAX_BLOCKS_PER_SEQ = 128;
// `step::ATTN_DECODE_MAX_SPLITS`.
constexpr uint32_t MAX_SPLITS = 64;
constexpr int MERGE_WARPS = 4;
constexpr int MERGE_THREADS = MERGE_WARPS * kittens::WARP_THREADS;
// log2(e): the decodes' softmax runs in base 2 with the scale folded in.
constexpr float LOG2E = 1.44269504089f;

using ml_gl = gl<float, -1, -1, 2, MAX_GROUP>;
using out_gl = gl<bf16, 1, 1, -1, -1>;
using index_gl = gl<int, 1, 1, -1, -1>;

template <int HEAD_DIM>
struct merge_globals {
    gl<float, -1, -1, MAX_GROUP, HEAD_DIM> o_split;
    ml_gl ml_split;
    out_gl out;
    // `[1, 2]`: `[P, L]`.
    index_gl sequence;
    // `scale · log2(e)`, the decode's.
    float scale_log2;
    int hr;
    int window;
    int splits;
    int min_tiles;
};

template <int HEAD_DIM>
__global__ __launch_bounds__(MERGE_THREADS, 1) void attn_decode_merge_ker(
    const __grid_constant__ merge_globals<HEAD_DIM> g) {
    constexpr int COLS = HEAD_DIM / MERGE_WARPS;
    using part_rt = rt_fl<MAX_GROUP, COLS>;
    using rows_rv = col_vec<part_rt>;
    // Each warp's merged columns, plain, so each row's are contiguous for the bulk store.
    __shared__ alignas(128) st_bf<MAX_GROUP, COLS, false> o_bf[MERGE_WARPS];

    // The decode's split rule, its integers (`attn_decode_fp8kv.cu`).
    const int kv_head = blockIdx.x;
    const int kv_len = g.sequence[{0, 0, 0, 0}] + g.sequence[{0, 0, 0, 1}];
    const int kv_iters = (kv_len + KV_HEIGHT - 1) / KV_HEIGHT;
    const int kv_first = g.window > 0 ? max(0, kv_len - g.window) / KV_HEIGHT : 0;
    const int n_tiles = kv_iters - kv_first;
    const int per = max(g.min_tiles, (n_tiles + g.splits - 1) / g.splits);
    const int used = (n_tiles + per - 1) / per;
    // One used split: the decode stored the row itself. Every thread of the CTA leaves alike.
    if (used <= 1) {
        return;
    }
    const int w = kittens::warpid();

    rows_rv big_m, m, l, l_s, r;
    part_rt o, o_s;
    warp::load(big_m, g.ml_split, {0, kv_head, 0, 0});
    for (int s = 1; s < used; s++) {
        warp::load(m, g.ml_split, {s, kv_head, 0, 0});
        warp::max(big_m, big_m, m);
    }
    warp::zero(l);
    warp::zero(o);
    for (int s = 0; s < used; s++) {
        warp::load(m, g.ml_split, {s, kv_head, 0, 0});
        warp::load(l_s, g.ml_split, {s, kv_head, 1, 0});
        scratchy::ops::rv_ternary_map<scratchy::ops::rescale_exact1>(r, m, big_m, g.scale_log2);
        warp::mul(l_s, l_s, r);
        warp::add(l, l, l_s);
        warp::load(o_s, g.o_split, {s, kv_head, 0, w});
        warp::mul_row(o_s, o_s, r);
        warp::add(o, o, o_s);
    }
    warp::div_row(o, o, l);
    rt_bf<MAX_GROUP, COLS> o16;
    warp::copy(o16, o);
    warp::store(o_bf[w], o16);
    // The register-to-shared store and the bulk copy's reads split the tile across lanes
    // differently: a warp barrier between them. Each warp stores its own columns of rows
    // `0 .. hr` (kv head g's query heads), so only this warp is here.
    __syncwarp();
    const uint32_t bytes = COLS * sizeof(bf16);
    for (int row = 0; row < g.hr; row++) {
        const int at = (kv_head * g.hr + row) * HEAD_DIM + w * COLS;
        warp::tma::store_async(g.out[{0, 0, 0, at}], o_bf[w][row * COLS], bytes);
    }
    warp::tma::store_async_wait();
}

template <int HEAD_DIM>
int launch_merge(const void* o_split, const void* ml_split, const void* sequence, void* out,
                 uint32_t q_heads, uint32_t kv_heads, float scale, uint32_t window,
                 uint32_t splits, uint32_t min_tiles, void* stream) {
    using split_gl = gl<float, -1, -1, MAX_GROUP, HEAD_DIM>;
    split_gl osg{static_cast<float*>(const_cast<void*>(o_split)), size_t(splits),
                 size_t(kv_heads), nullptr, nullptr};
    ml_gl mlg{static_cast<float*>(const_cast<void*>(ml_split)), size_t(splits), size_t(kv_heads),
              nullptr, nullptr};
    out_gl og{static_cast<bf16*>(out), nullptr, nullptr, size_t(1), size_t(q_heads) * HEAD_DIM};
    index_gl sg{static_cast<int*>(const_cast<void*>(sequence)), nullptr, nullptr, size_t(1),
                size_t(2)};
    merge_globals<HEAD_DIM> G{osg,         mlg,          og,
                              sg,          scale * LOG2E, int(q_heads / kv_heads),
                              int(window), int(splits),   int(min_tiles)};
    void* args[] = {&G};
    const dim3 grid{kv_heads}, block{uint32_t(MERGE_THREADS)};
    return static_cast<int>(cudaLaunchKernel(attn_decode_merge_ker<HEAD_DIM>, grid, block, args,
                                             0, static_cast<cudaStream_t>(stream)));
}

}  // namespace

// The split decode's merge: `out` (`[1, q_heads · head_dim]` bf16) from the partials
// `scratchy_cuda_attn_decode_fp8kv_split` (head_dim 256) or `..._d512_fp8kv_split` (512) wrote,
// launched after it on the same stream with the same `q_heads`, `kv_heads`, `scale`, `window`,
// `splits`, `min_tiles` and `sequence`. A kv head whose decode used one split is left as the decode
// wrote it.
//
// 0 = launched; SCRATCHY_CUDA_ERR_SHAPE = refused, nothing launched: `head_dim` not 256 or 512;
// `q_heads` or `kv_heads` zero, `q_heads` not a multiple of `kv_heads`, or a group above 16;
// `scale` not finite or not above 0; `window` not a multiple of 64; `splits` zero or above 64;
// `min_tiles` zero or above 128; a dimension or the window above INT_MAX; a null pointer.
// SCRATCHY_CUDA_ERR_HOST = a C++ exception was caught; any other value = the cudaError_t of the
// launch. Asynchronous on `stream`.
extern "C" int scratchy_cuda_attn_decode_merge(const void* o_split, const void* ml_split,
                                               const void* sequence, void* out, uint32_t q_heads,
                                               uint32_t kv_heads, uint32_t head_dim, float scale,
                                               uint32_t window, uint32_t splits,
                                               uint32_t min_tiles,
                                               void* stream /* cudaStream_t; null = default */) {
    if ((head_dim != 256 && head_dim != 512) || q_heads == 0 || kv_heads == 0 ||
        q_heads % kv_heads != 0 || q_heads / kv_heads > uint32_t(MAX_GROUP) ||
        !std::isfinite(scale) || !(scale > 0.0f) || window % BLOCK_SIZE != 0 || splits == 0 ||
        splits > MAX_SPLITS || min_tiles == 0 || min_tiles > MAX_BLOCKS_PER_SEQ ||
        q_heads > uint32_t(INT_MAX) || window > uint32_t(INT_MAX) || o_split == nullptr ||
        ml_split == nullptr || sequence == nullptr || out == nullptr) {
        return SCRATCHY_CUDA_ERR_SHAPE;
    }
    // ThunderKittens' host code can throw; an exception must never unwind into Rust.
    try {
        if (head_dim == 256) {
            return launch_merge<256>(o_split, ml_split, sequence, out, q_heads, kv_heads, scale,
                                     window, splits, min_tiles, stream);
        }
        return launch_merge<512>(o_split, ml_split, sequence, out, q_heads, kv_heads, scale,
                                 window, splits, min_tiles, stream);
    } catch (...) {
        return SCRATCHY_CUDA_ERR_HOST;
    }
}
