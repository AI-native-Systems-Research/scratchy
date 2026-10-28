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

// The RMSNorm's inverse root of one row's mean square plus eps: `rsqrtf(m + eps)`, with `m` the
// row's sum of squares already divided by `cols` (`warp::div`), so the two together are today's
// `rsqrtf(total / float(cols) + eps)`. A division then an add: nothing contracts into an FMA.
// The `float2` form serves the packed column vector of a row-layout tile (see `rescale_exact1`).
struct rms_inv {
    template <typename T>
    static __device__ inline T op(const T& m, const T& eps) {
        return rsqrtf(m + eps);
    }
};
template <>
__device__ inline float rms_inv::op<float>(const float& m, const float& eps) {
    return rsqrtf(m + eps);
}
template <>
__device__ inline float2 rms_inv::op<float2>(const float2& m, const float2& eps) {
    return float2{rms_inv::op<float>(m.x, eps.x), rms_inv::op<float>(m.y, eps.y)};
}

// SwiGLU's activation: `g / (1 + e^(−g)) · u`, the expression the plain-CUDA kernel computed. For
// `g ≤ −88`, `__expf(−g)` overflows to +inf and the result is a signed zero, never NaN.
struct silu_mul {
    template <typename T>
    static __device__ inline T op(const T& g, const T& u) {
        return g / (1.0f + __expf(-g)) * u;
    }
};
template <>
__device__ inline float silu_mul::op<float>(const float& g, const float& u) {
    return g / (1.0f + __expf(-g)) * u;
}
template <>
__device__ inline float2 silu_mul::op<float2>(const float2& g, const float2& u) {
    return float2{silu_mul::op<float>(g.x, u.x), silu_mul::op<float>(g.y, u.y)};
}

// Gemma's GeGLU activation: `0.5 · g · (1 + tanh(√(2/π) · (g + 0.044715 · g³))) · u`, HF's
// `gelu_pytorch_tanh` of the gate times `up` (TK's `flux_gelu.cu` uses the same two constants).
// Under `--use_fast_math` (build.rs) `tanhf` is the intrinsic `__tanhf` (the CUDA programming
// guide's table of functions `--use_fast_math` affects), whose maximum relative error is 2⁻¹¹.
// No clamp: `tanh(±inf) = ±1`, so a finite gate whose cube overflows still gives a finite
// result; `g = −inf` gives NaN, as HF does.
struct gelu_tanh_mul {
    static constexpr float SQRT_2_OVER_PI = 0.7978845608f;
    static constexpr float CUBIC = 0.044715f;
    template <typename T>
    static __device__ inline T op(const T& g, const T& u) {
        return 0.5f * g * (1.0f + tanhf(SQRT_2_OVER_PI * (g + CUBIC * g * g * g))) * u;
    }
};
template <>
__device__ inline float gelu_tanh_mul::op<float>(const float& g, const float& u) {
    return 0.5f * g * (1.0f + tanhf(SQRT_2_OVER_PI * (g + CUBIC * g * g * g))) * u;
}
template <>
__device__ inline float2 gelu_tanh_mul::op<float2>(const float2& g, const float2& u) {
    return float2{gelu_tanh_mul::op<float>(g.x, u.x), gelu_tanh_mul::op<float>(g.y, u.y)};
}

// Gemma's final-logit soft-cap: `c · tanh(x / c)`, `c` the cap (30). `x / c` is the fast division
// and `tanhf` the intrinsic `__tanhf` under `--use_fast_math` (see `gelu_tanh_mul`). `±0 → ±0`,
// `±inf → ±c`, NaN → NaN. The `float2` form is the one TK's tile-and-scalar `bin_map` runs: it
// packs `c` into both halves of a `float2`.
struct soft_cap {
    template <typename T>
    static __device__ inline T op(const T& x, const T& c) {
        return c * tanhf(x / c);
    }
};
template <>
__device__ inline float soft_cap::op<float>(const float& x, const float& c) {
    return c * tanhf(x / c);
}
template <>
__device__ inline float2 soft_cap::op<float2>(const float2& x, const float2& c) {
    return float2{soft_cap::op<float>(x.x, c.x), soft_cap::op<float>(x.y, c.y)};
}

// The attention output gate: `a / (1 + e^(−g))`, the expression the plain-CUDA kernel computed.
struct gate_apply {
    template <typename T>
    static __device__ inline T op(const T& a, const T& g) {
        return a / (1.0f + __expf(-g));
    }
};
template <>
__device__ inline float gate_apply::op<float>(const float& a, const float& g) {
    return a / (1.0f + __expf(-g));
}
template <>
__device__ inline float2 gate_apply::op<float2>(const float2& a, const float2& g) {
    return float2{gate_apply::op<float>(a.x, g.x), gate_apply::op<float>(a.y, g.y)};
}

// The gated delta-rule scan's inverse L2 norm of one q or k vector: `rsqrtf(x + 1e-6)`, `x` the
// vector's sum of squares (`warp::sum`), the expression the plain-CUDA scan computed (metal's and
// fla's eps, inside the sqrt). The scan multiplies q's by `scale` after.
struct l2_inv {
    static constexpr float EPS = 1e-6f;
    template <typename T>
    static __device__ inline T op(const T& x) {
        return rsqrtf(x + EPS);
    }
};
template <>
__device__ inline float l2_inv::op<float>(const float& x) {
    return rsqrtf(x + EPS);
}
template <>
__device__ inline float2 l2_inv::op<float2>(const float2& x) {
    return float2{l2_inv::op<float>(x.x), l2_inv::op<float>(x.y)};
}

// The gated delta-rule scan's decay of one token: `expf(g)`, `g` the gating's log decay, the
// expression the plain-CUDA scan computed (`__expf` under `--use_fast_math`).
struct exp_of {
    template <typename T>
    static __device__ inline T op(const T& x) {
        return expf(x);
    }
};
template <>
__device__ inline float exp_of::op<float>(const float& x) {
    return expf(x);
}
template <>
__device__ inline float2 exp_of::op<float2>(const float2& x) {
    return float2{exp_of::op<float>(x.x), exp_of::op<float>(x.y)};
}

// One tap of the GDN conv1d's sum: `fmaf(a, b, c)`, `a·b + c` with one rounding, as the plain-CUDA
// kernel's `fmaf(w, x, sum)`. Applied through the `rv_fma` gap below (three vector operands).
struct fma_acc {
    template <typename T>
    static __device__ inline T op(const T& a, const T& b, const T& c) {
        return fmaf(a, b, c);
    }
};
template <>
__device__ inline float fma_acc::op<float>(const float& a, const float& b, const float& c) {
    return fmaf(a, b, c);
}
template <>
__device__ inline float2 fma_acc::op<float2>(const float2& a, const float2& b, const float2& c) {
    return float2{fma_acc::op<float>(a.x, b.x, c.x), fma_acc::op<float>(a.y, b.y, c.y)};
}

// SiLU: `x / (1 + e^(−x))`, the GDN conv1d's and gated norm's expression (`expf`, `__expf` under
// `--use_fast_math`). For `x ≤ −88`, `e^(−x)` overflows to +inf and the result is a signed zero.
struct silu {
    template <typename T>
    static __device__ inline T op(const T& x) {
        return x / (1.0f + expf(-x));
    }
};
template <>
__device__ inline float silu::op<float>(const float& x) {
    return x / (1.0f + expf(-x));
}
template <>
__device__ inline float2 silu::op<float2>(const float2& x) {
    return float2{silu::op<float>(x.x), silu::op<float>(x.y)};
}

// The sigmoid: `1 / (1 + e^(−x))`, the GDN gating's β, its expression.
struct sigmoid {
    template <typename T>
    static __device__ inline T op(const T& x) {
        return 1.0f / (1.0f + expf(-x));
    }
};
template <>
__device__ inline float sigmoid::op<float>(const float& x) {
    return 1.0f / (1.0f + expf(-x));
}
template <>
__device__ inline float2 sigmoid::op<float2>(const float2& x) {
    return float2{sigmoid::op<float>(x.x), sigmoid::op<float>(x.y)};
}

// The GDN gating's log decay: `−exp(a_log) · softplus(x)`, with `x = a + dt_bias` and
// `softplus(x) = log1p(exp(x))` for `x ≤ 20`, else `x` (metal's threshold, and torch's default).
// The whole expression is one op, the plain-CUDA kernel's, so its evaluation order and rounding
// are kept exactly.
struct gdn_g {
    static constexpr float SOFTPLUS_THRESHOLD = 20.0f;
    template <typename T>
    static __device__ inline T op(const T& x, const T& a_log) {
        return -expf(a_log) * (x <= SOFTPLUS_THRESHOLD ? log1pf(expf(x)) : x);
    }
};
template <>
__device__ inline float gdn_g::op<float>(const float& x, const float& a_log) {
    return -expf(a_log) * (x <= SOFTPLUS_THRESHOLD ? log1pf(expf(x)) : x);
}
template <>
__device__ inline float2 gdn_g::op<float2>(const float2& x, const float2& a_log) {
    return float2{gdn_g::op<float>(x.x, a_log.x), gdn_g::op<float>(x.y, a_log.y)};
}

// The MoE route's top-k key: a NaN probability as −inf, every other value as it is. TK's
// `base_ops::max` is `::max`, which returns the non-NaN operand, so a NaN key would never be the
// round's max; as −inf it is the lowest, and the round's index pick still takes it in turn.
struct nan_to_neg_inf {
    template <typename T>
    static __device__ inline T op(const T& x) {
        return x != x ? kittens::base_types::constants<T>::neg_infty() : x;
    }
};
template <>
__device__ inline float nan_to_neg_inf::op<float>(const float& x) {
    return x != x ? kittens::base_types::constants<float>::neg_infty() : x;
}
template <>
__device__ inline float2 nan_to_neg_inf::op<float2>(const float2& x) {
    return float2{nan_to_neg_inf::op<float>(x.x), nan_to_neg_inf::op<float>(x.y)};
}

// The windowed online softmax's running max, before it is scaled: −inf as +0, every other value as
// it is. With a window, a warp's first KV tile can lie wholly below every one of its rows'
// windows, so its row max is −inf and `sub_row` would form `−inf − (−inf) = NaN`. Through this op
// the tile's `p = exp2(−inf − 0)` is exactly 0 instead, and a finite max is unchanged. The
// `float2` form serves the packed column vector of a row-layout tile (see `rescale_exact1`).
struct neg_inf_to_zero {
    template <typename T>
    static __device__ inline T op(const T& x) {
        return x == kittens::base_types::constants<T>::neg_infty() ? T(0.0f) : x;
    }
};
template <>
__device__ inline float neg_inf_to_zero::op<float>(const float& x) {
    return x == kittens::base_types::constants<float>::neg_infty() ? 0.0f : x;
}
template <>
__device__ inline float2 neg_inf_to_zero::op<float2>(const float2& x) {
    return float2{neg_inf_to_zero::op<float>(x.x), neg_inf_to_zero::op<float>(x.y)};
}

// The windowed attention's softmax sum before the final division: 0 as 1, every other value as it
// is. A padding row of a prefill bucket at least `window − 1` rows past the sequence's last key has
// no key in its window, so its sum and its output are both 0 and `div_row` would form
// `0 / 0 = NaN`; through this op it gives `0 / 1 = +0`, finite as the causal kernel's padding rows
// are. A row with any allowed key has a sum of at least 1 (its max key's `p` is exactly 1), so this
// is the identity there. The `float2` form as `neg_inf_to_zero`'s.
struct zero_to_one {
    template <typename T>
    static __device__ inline T op(const T& x) {
        return x == T(0.0f) ? T(1.0f) : x;
    }
};
template <>
__device__ inline float zero_to_one::op<float>(const float& x) {
    return x == 0.0f ? 1.0f : x;
}
template <>
__device__ inline float2 zero_to_one::op<float2>(const float2& x) {
    return float2{zero_to_one::op<float>(x.x), zero_to_one::op<float>(x.y)};
}

// The MoE sort's run length: an expert's pair count rounded up to whole 128-row tiles of the
// grouped GEMM (`op_abi::CUDA_EXPERT_TILE_ROWS`), so a tile belongs to one expert. Integer, exact.
struct tile_pad {
    static constexpr int ROWS = 128;
    template <typename T>
    static __device__ inline T op(const T& n) {
        return (n + (ROWS - 1)) / ROWS * ROWS;
    }
};

// TK-GAP(exclusive_scan_sv) begin
// An exclusive prefix sum over a shared vector, in place, each element first mapped by `op`:
// `v[i] = Σ_{j < i} op(v[j])`, returning the total `Σ_j op(v[j])`. TK has reductions to one value
// and maps, never a scan. One lane runs it, serially in index order, so the sums are the same on
// every run; the caller syncs its warp before (the vector's last write) and after (its next read),
// as for any TK operation on a shared vector. Written over `sv::operator[]`, as TK's own
// shared-vector maps (`ops/group/shared/vec/maps.cuh`) index it. The MoE sort's `offsets` go
// through it, padded by `tile_pad`.
template <typename op, kittens::ducks::sv::all SV>
__device__ static inline typename SV::dtype exclusive_scan_sv(SV& v) {
    typename SV::dtype run = kittens::base_types::constants<typename SV::dtype>::zero();
    for (int i = 0; i < SV::length; i++) {
        const typename SV::dtype x = v[i];
        v[i] = run;
        run = run + op::template op<typename SV::dtype>(x);
    }
    return run;
}
// TK-GAP(exclusive_scan_sv) end

// TK-GAP(rv_fma) begin
// A register-vector map over three vectors, `dst[i][j] = op(a[i][j], b[i][j], c[i][j])`: TK's own
// vector `bin_op` (`ops/group/register/vec/maps.cuh`) takes one vector and a scalar, or two
// vectors, never three. Written as TK writes `bin_op`: loops over `outer_dim` and `inner_dim`,
// calling `op::template op<typename V::dtype>`. The GDN conv1d's `fma_acc` taps go through it, so
// each tap is one `fmaf`, one rounding.
template <typename op, kittens::ducks::rv::all V>
__device__ static inline void rv_fma(V& dst, const V& a, const V& b, const V& c) {
#pragma unroll
    for (int i = 0; i < V::outer_dim; i++) {
#pragma unroll
        for (int j = 0; j < V::inner_dim; j++) {
            dst[i][j] = op::template op<typename V::dtype>(a[i][j], b[i][j], c[i][j]);
        }
    }
}
// TK-GAP(rv_fma) end

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
