// SPDX-License-Identifier: Apache-2.0
//
// The element operations ThunderKittens lacks, written once, the way ThunderKittens writes its
// own: an op struct with `template<typename T> static __device__ inline T op(...)` and its
// `float`/`float2` forms (`thunderkittens/include/common/base_ops.cuh`, e.g. `base_ops::exp2`),
// applied through TK's maps. A kernel never writes such an operation inline.
//
// The rule (`tests/kernels_use_thunderkittens.rs`): math intrinsics (`exp2f`, `__expf`, `fmaf`,
// ...) may appear here and in no `.cu`. A helper that fills an operation TK has no map for sits
// between `// TK-GAP(<name>) begin` and `// TK-GAP(<name>) end`, and its name is on the test's
// allowlist. This header defines no kernel.

#pragma once

#include "kittens.cuh"

namespace scratchy {
namespace ops {

// The online softmax's rescale of one tile, per row: `exp2((m_old − m_new) · c)`, or exactly 1
// when `m_new == m_old` (the tile raised no score above the running max, as for a tile masked out
// for the row). `(m_old − m_new) · c` has no product feeding a sum, so nothing contracts into an
// FMA under `--use_fast_math`; the select makes the unchanged case exact without relying on
// `exp2(0)`'s rounding. `m_old = −inf` (the first tile) gives `exp2(−inf) = 0`.
//
// The `float2` form is the one `kernels/attn_prefill_bf16.cu` applies: its vectors are
// `col_vec<rt_fl<16, 64>>`, a row-layout tile's column vector, whose layout is `ortho`
// (`rt_base::col_vec_layout`), so `inner_dim` is 1 and `dtype` is the packed `float2`
// (`rv::dtype`: `is_naive` is false). Each `[i][0]` holds two rows of subtile `i`: `.x` the top
// row and `.y` the bottom one, as `row_reduce` writes them (`accum_packed.x` from the top row,
// `.y` from the bottom) and `row_map` reads them (`row_values[i][0].x` scales the top row, `.y`
// the bottom). So the select and the exp2 run on `.x` and `.y` separately.
struct rescale_exact1 {
    template <typename T>
    static __device__ inline T op(const T& o, const T& n, const float& c) {
        return o == n ? T(1.0f) : exp2f((o - n) * c);
    }
};
template <>
__device__ inline float rescale_exact1::op<float>(const float& o, const float& n, const float& c) {
    return o == n ? 1.0f : exp2f((o - n) * c);
}
template <>
__device__ inline float2 rescale_exact1::op<float2>(const float2& o, const float2& n,
                                                    const float& c) {
    return float2{rescale_exact1::op<float>(o.x, n.x, c), rescale_exact1::op<float>(o.y, n.y, c)};
}

// TK-GAP(rv_ternary_map) begin
// A register-vector map over two vectors and one scalar, `dst[i][j] = op(a[i][j], b[i][j], c)`:
// TK's own vector `bin_op` (`ops/group/register/vec/maps.cuh`) takes one vector and a scalar, or
// two vectors, never both. Written as TK writes `bin_op`: loops over `outer_dim` and `inner_dim`,
// calling `op::template op<typename V::dtype>`. Packed `float2` column vectors only, so an op's
// `float2` form is the one that runs (the #812 lesson: a `float` form over `.x` alone would leave
// every bottom row unscaled).
template <typename op, kittens::ducks::rv::all V>
__device__ static inline void rv_ternary_map(V& dst, const V& a, const V& b, const float& c) {
    static_assert(std::is_same_v<typename V::dtype, float2>, "a packed f32 column vector");
#pragma unroll
    for (int i = 0; i < V::outer_dim; i++) {
#pragma unroll
        for (int j = 0; j < V::inner_dim; j++) {
            dst[i][j] = op::template op<typename V::dtype>(a[i][j], b[i][j], c);
        }
    }
}
// TK-GAP(rv_ternary_map) end

}  // namespace ops
}  // namespace scratchy
