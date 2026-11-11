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

// The one-row RMSNorm's `inv` (`kernels/rms_norm_bf16.cu`'s `rms_norm_one_row_ker`), written out
// as the 16-row kernel's SASS computes `rms_inv(total / cols, eps)` in all four forms on the H100:
// `rcp(cols)` (one MUFU.RCP: `warp::div` by a scalar under `--use_fast_math`), then `total · rcp +
// eps` contracted into one FFMA, then MUFU.RSQ. Left to ptxas in the one-row `ADD_NORM` instance,
// the bottom packed row's product and `+ eps` stayed apart (an FMUL, then an FADD past the warps'
// branch); with the FFMA explicit here (`fmaf`, one rounding) both rows are that FFMA. `rcp` is
// the caller's, `1 / cols` from `warp::div` of ones, the same MUFU.RCP. Applied through
// `rv_ternary_map`, so its `float2` form runs on both packed rows.
struct rms_inv_fused {
    template <typename T>
    static __device__ inline T op(const T& total, const T& rcp, const float& eps) {
        return rsqrtf(fmaf(total, rcp, eps));
    }
};
template <>
__device__ inline float rms_inv_fused::op<float>(const float& total, const float& rcp,
                                                 const float& eps) {
    return rsqrtf(fmaf(total, rcp, eps));
}
template <>
__device__ inline float2 rms_inv_fused::op<float2>(const float2& total, const float2& rcp,
                                                   const float& eps) {
    return float2{rms_inv_fused::op<float>(total.x, rcp.x, eps),
                  rms_inv_fused::op<float>(total.y, rcp.y, eps)};
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

// The greedy argmax's maps (`kernels/argmax_bf16.cu`), over naive f32 vectors. `nan_flag`: 1 for
// a NaN logit, else 0 (a `max` reduce says whether a row holds one). `equals_key`: 1 for a logit
// equal to the row's max `key`, else 0 (−0 equals +0, as the host's comparison has it).
// `index_if`: index `idx` (an exact f32 below 2^24) where `flag` is set, else +inf, so a `min`
// reduce gives the lowest index of the flagged set.
struct nan_flag {
    template <typename T>
    static __device__ inline T op(const T& x) {
        return x != x ? T(1) : T(0);
    }
};
struct equals_key {
    template <typename T>
    static __device__ inline T op(const T& x, const T& key) {
        return x == key ? T(1) : T(0);
    }
};
struct index_if {
    template <typename T>
    static __device__ inline T op(const T& flag, const T& idx) {
        return flag != T(0) ? idx : kittens::base_types::constants<T>::pos_infty();
    }
};

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

// The per-token e4m3 scale of one row (FP8 S3, `quant_rows_fp8.cu`): `max(amax / 448, 1 / (448 ·
// 512))`, vLLM v0.31.0's `dynamic_per_token_scaled_fp8_quant_kernel_strided` (`csrc/libtorch_stable/
// quantization/w8a8/fp8/common.cu`, its `token_scale`, with `min_scaling_factor` from `csrc/
// quantization/utils.cuh`). vLLM's division is IEEE; under `--use_fast_math` `/` is not, so it is
// `__fdiv_rn`. The floor is a constant of the type, rounded once at compile time, as vLLM's
// `1.0f / (448.0f * 512.0f)` is. `amax` is never NaN (TK's `max` is `fmaxf`'s, which drops one).
struct fp8_scale {
    static constexpr float E4M3_MAX = 448.0f;
    static constexpr float MIN_SCALE = 1.0f / (448.0f * 512.0f);
    template <typename T>
    static __device__ inline T op(const T& amax) {
        return fmaxf(__fdiv_rn(amax, E4M3_MAX), MIN_SCALE);
    }
};
template <>
__device__ inline float fp8_scale::op<float>(const float& amax) {
    return fmaxf(__fdiv_rn(amax, E4M3_MAX), MIN_SCALE);
}
template <>
__device__ inline float2 fp8_scale::op<float2>(const float2& amax) {
    return float2{fp8_scale::op<float>(amax.x), fp8_scale::op<float>(amax.y)};
}

// One element before its e4m3 cast (FP8 S3): `clamp(x / scale, −448, 448)`, vLLM v0.31.0's
// `scaled_fp8_conversion<false>` (`csrc/quantization/w8a8/fp8/common.cuh`): the per-token kernel
// DIVIDES by the scale (`is_scale_inverted = false`), it does not multiply by a reciprocal. The
// division is `__fdiv_rn` (see `fp8_scale`); the clamp is vLLM's `fmaxf(−max, fminf(x, max))`, so
// a NaN quotient becomes +448, as in vLLM. TK's `warp::copy` to an fp8 tile then rounds once to
// nearest even (`__nv_fp8x4_e4m3`, saturating to finite), as vLLM's `__nv_cvt_float_to_fp8`.
// The `float2` form serves a row-layout tile's packed pairs (`warp::row_map`).
struct fp8_quant_pre {
    template <typename T>
    static __device__ inline T op(const T& x, const T& scale) {
        return fmaxf(-fp8_scale::E4M3_MAX, fminf(__fdiv_rn(x, scale), fp8_scale::E4M3_MAX));
    }
};
template <>
__device__ inline float fp8_quant_pre::op<float>(const float& x, const float& scale) {
    return fmaxf(-fp8_scale::E4M3_MAX, fminf(__fdiv_rn(x, scale), fp8_scale::E4M3_MAX));
}
template <>
__device__ inline float2 fp8_quant_pre::op<float2>(const float2& x, const float2& scale) {
    return float2{fp8_quant_pre::op<float>(x.x, scale.x), fp8_quant_pre::op<float>(x.y, scale.y)};
}

// `fp8_quant_pre` that leaves a ±0 undivided (FP8 S8 F3, `quant_rows_fp8.cu`): `±0 / scale` is
// ±0 and the clamp keeps it, so the value is `fp8_quant_pre`'s, but `__fdiv_rn` of a zero takes its
// slow path (SASS `FCHK` then a `CALL`), which a prefill bucket's padding rows hit on every element.
struct fp8_quant_pre_nz {
    template <typename T>
    static __device__ inline T op(const T& x, const T& scale) {
        return x == T(0) ? x : fp8_quant_pre::op<T>(x, scale);
    }
};
template <>
__device__ inline float2 fp8_quant_pre_nz::op<float2>(const float2& x, const float2& scale) {
    return float2{fp8_quant_pre_nz::op<float>(x.x, scale.x),
                  fp8_quant_pre_nz::op<float>(x.y, scale.y)};
}

// One quantized element as the fp8 GEMV computes with it (FP8 S4, `gemv_fp8.cu`): `x` rounded to
// e4m3 and widened back, `float(e4m3_satfinite(x))`. The cast is TK's scalar convertor
// (`__nv_fp8_e4m3(float)`: round to nearest even, saturating to finite), the one byte S3's
// `warp::copy` to an fp8 tile writes for the same `x`; the widening is exact. `x` is
// `fp8_quant_pre`'s output, so the pair gives `quant_rows_fp8.cu`'s byte, as its value.
struct fp8_round {
    template <typename T>
    static __device__ inline T op(const T& x) {
        return kittens::base_types::convertor<float, kittens::fp8e4m3>::convert(
            kittens::base_types::convertor<kittens::fp8e4m3, float>::convert(x));
    }
};

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

// The residual epilogue (FP8 S8 W4a): `bf16(bf16(v) + r)`, `v` rounded as the GEMV stores it, then
// `add_bf16.cu`'s one f32 add of the two bf16 values and its one rounding.
__device__ inline kittens::bf16 with_residual(float v, kittens::bf16 r) {
    using kittens::base_types::convertor;
    const float stored = convertor<float, kittens::bf16>::convert(convertor<kittens::bf16, float>::convert(v));
    const float sum = kittens::base_ops::sum::op<float>(stored, convertor<float, kittens::bf16>::convert(r));
    return convertor<kittens::bf16, float>::convert(sum);
}

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

// TK-GAP(sv_fp8_to_rv_fl) begin
// `warp::load` of an e4m3 shared vector into a naive f32 register vector (FP8 S4's decode GEMV,
// approved 2026-11-06): TK at the pin has no `rv<fp8e4m3>` (`types/register/rv.cuh` static_asserts
// it) and no `move<fp8e4m3>::lds`, so its `load` (`ops/group/memory/vec/shared_to_register.cuh`)
// does not compile for an fp8 source. The layout is that `load`'s naive one: lane `l`'s slot `w`
// is element `32w + l`. Slots `w` and `w + 1` widen together, `cvt.rn.f16x2.e4m3x2` on the pair
// then f16 to f32 (`__nv_cvt_fp8x2_to_halfraw2`, then `__half22float2`, as TK's own fp4 convertor
// in `common/base_types.cuh` writes it): exact, as every e4m3 value is an f16 and an f32. The low
// byte is `.x`. One warp, as TK's `load`.
template <kittens::ducks::rv::naive_layout RV, kittens::ducks::sv::all SV>
__device__ static inline void sv_fp8_to_rv_fl(RV& dst, const SV& src) {
    static_assert(std::is_same_v<typename SV::dtype, kittens::fp8e4m3>, "an e4m3 shared vector");
    static_assert(std::is_same_v<typename RV::dtype, float>, "a naive f32 register vector");
    static_assert(SV::length == RV::length && RV::length % 64 == 0, "whole pairs of 32-wide slots");
    const int lane = kittens::laneid();
    __syncwarp();
#pragma unroll
    for (int w = 0; w < RV::outer_dim; w += 2) {
        const __nv_fp8x2_storage_t lo = src[32 * w + lane].__x;
        const __nv_fp8x2_storage_t hi = src[32 * (w + 1) + lane].__x;
        const __nv_fp8x2_storage_t pair = __nv_fp8x2_storage_t(lo | (hi << 8));
        const float2 f = __half22float2(__half2(__nv_cvt_fp8x2_to_halfraw2(pair, __NV_E4M3)));
        dst[w][0] = f.x;
        dst[w + 1][0] = f.y;
    }
}
// TK-GAP(sv_fp8_to_rv_fl) end

}  // namespace ops
}  // namespace scratchy
