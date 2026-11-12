// SPDX-License-Identifier: Apache-2.0
//
// The decode RMSNorm in one CTA a row, fused with the next GEMM's per-token FP8 quantization (FP8
// S8 N3; every one-row norm of a tape plays it, `step::norm_row_takes`). Per row `r` of `x [rows,
// cols]` bf16:
//
//     y[r, c] = bf16(x[r, c] · inv · (offset + gain[c])),  inv = rsqrt(Σ_c x[r, c]² / cols + eps)
//
// and the two residual folds of `rms_norm_bf16.cu`: `ADD_NORM` norms `s = bf16(a + b)` and stores
// `s` (Qwen's `norm(add(a, b))`), `NORM_ADD` stores `bf16(bf16(y) + residual)` (Gemma's
// `add(norm(x), residual)`). With `QUANT` (the norm and the add-norm, whose outputs feed the T2
// tensor-core GEMVs), also the row's e4m3 bytes and scale: `quant_rows_fp8` of the bf16 row `y`,
// bit for bit (its `amax` of `|y|`, `fp8_scale`, `fp8_quant_pre`, TK's e4m3 convertor).
//
// THE SUMMING ORDER IS ITS OWN (user decision 2026-11-12: the one-row norm's order may change,
// gated by an f64 bound and coherence end to end). The row kernel (`rms_norm_row_ker`, N1) keeps the
// 16-row kernel's order, so every one of its `cols / 256` CTAs re-reads the whole row; here:
//   - one CTA of 16 warps a row; warp `w` holds the row's 256-column chunks `w` and `w + 16` (at
//     most 8,192 columns) as naive `rv_fl<256>`, loaded straight from the `gl` with the gain's
//     chunks (and `b` or the residual) in one burst, so every load is in flight at once;
//   - the warp's partial: `warp::sum` of `warp::mul(x, x)` per chunk, the chunks added in order;
//     the 16 partials through a shared `sv_fl<16>`, then `warp::sum` of them in every warp (TK's
//     shuffle tree), so every warp holds the same total; `inv` by `rms_inv_fused` (`rsqrt(total ·
//     (1/cols) + eps)`, the row kernel's expression);
//   - each element then exactly as the row kernel: `warp::mul` by `inv`, `warp::mul` by
//     `offset + gain` (`warp::add` first), rounded once to bf16 (`warp::copy` to `rv_bf`), so only
//     `inv`'s bits move.
//   - `QUANT`: each warp's `amax` of `|y|` (`warp::abs`, elementwise `warp::max`, then the
//     reduction), the 16 through a second `sv_fl<16>` and `warp::max` (exact in any order),
//     `fp8_scale`, then `warp::bin_op<fp8_quant_pre>` on the widened bf16 `y` and a converting
//     `warp::store` to the e4m3 row (TK's convertor, round to nearest even, saturating).
// Two barriers a launch; no scratch; repeats are bit-identical. Built with `--use_fast_math`, as
// every norm: `rsqrtf` is the approximate reciprocal square root, subnormals flush to zero.

#include "kittens.cuh"
#include "scratchy_ops.cuh"

#include <climits>
#include <cmath>
#include <cstdint>

using namespace kittens;

// The launcher's return codes that are not a `cudaError_t`. `src/kernels.rs` mirrors them.
#define SCRATCHY_CUDA_ERR_SHAPE (-1)
#define SCRATCHY_CUDA_ERR_HOST (-2)

namespace {

constexpr int CHUNK = 256;
constexpr int WARPS = 16;
constexpr int NUM_THREADS = WARPS * kittens::WARP_THREADS;
constexpr int WARP_CHUNKS = 2;
// The widest row: 16 warps × 2 chunks × 256 columns.
constexpr uint32_t MAX_COLS = uint32_t(WARPS * WARP_CHUNKS * CHUNK);

// The forms, the launcher's codes.
constexpr int NORM = 0;
constexpr int ADD_NORM = 1;
constexpr int NORM_ADD = 2;

using chunk_rv = rv_fl<CHUNK>;
using chunk_bf = rv_bf<CHUNK>;
using warps_rv = rv_fl<WARPS>;
using row_gl = gl<bf16, 1, 1, -1, -1>;
using gain_gl = gl<bf16, 1, 1, 1, -1>;
using q_gl = gl<fp8e4m3, 1, 1, -1, -1>;
using scale_gl = gl<float, 1, 1, 1, -1>;

struct n3_globals {
    // The row (`ADD_NORM`: `a`), and `ADD_NORM`'s `b` or `NORM_ADD`'s residual (else `x`, unread).
    row_gl x;
    row_gl other;
    gain_gl gain;
    row_gl y;
    // `ADD_NORM`'s sum (else `y`, unwritten).
    row_gl sum;
    // `QUANT`'s e4m3 rows and scales (else unwritten views).
    q_gl q;
    scale_gl scale;
    int chunks;
    float eps;
    // `1 / cols`.
    float rcp;
    float offset;
};

template <int FORM, bool QUANT>
__global__ __launch_bounds__(NUM_THREADS, 1) void rms_norm_quant_row_ker(
    const __grid_constant__ n3_globals g) {
    __shared__ sv_fl<WARPS> parts;
    __shared__ sv_fl<WARPS> amaxes;
    const int w = kittens::warpid();
    const int r = blockIdx.x;

    // 1. The warp's chunks, the gain's and the other input's, every load at once; `ADD_NORM`'s sum
    //    added, rounded to bf16 (the add's one rounding) and stored.
    chunk_rv xs[WARP_CHUNKS], gs[WARP_CHUNKS], os[WARP_CHUNKS];
#pragma unroll
    for (int i = 0; i < WARP_CHUNKS; i++) {
        const int c = w + i * WARPS;
        if (c < g.chunks) {
            warp::load(xs[i], g.x, {0, 0, r, c});
            warp::load(gs[i], g.gain, {0, 0, 0, c});
            if constexpr (FORM != NORM) {
                warp::load(os[i], g.other, {0, 0, r, c});
            }
        }
    }
    if constexpr (FORM == ADD_NORM) {
#pragma unroll
        for (int i = 0; i < WARP_CHUNKS; i++) {
            const int c = w + i * WARPS;
            if (c < g.chunks) {
                warp::add(xs[i], xs[i], os[i]);
                chunk_bf s;
                warp::copy(s, xs[i]);
                warp::copy(xs[i], s);
                warp::store(g.sum, s, {0, 0, r, c});
            }
        }
    }

    // 2. The sum of squares: the warp's chunks in order, then the 16 warps' tree.
    float part = 0.0f;
#pragma unroll
    for (int i = 0; i < WARP_CHUNKS; i++) {
        if (w + i * WARPS < g.chunks) {
            chunk_rv sq;
            warp::mul(sq, xs[i], xs[i]);
            part += warp::sum(sq);
        }
    }
    if (warp::laneid() == 0) {
        parts[w] = part;
    }
    __syncthreads();
    warps_rv all;
    warp::load(all, parts);
    const float inv = scratchy::ops::rms_inv_fused::op<float>(warp::sum(all), g.rcp, g.eps);

    // 3. The normed row, the row kernel's per-element ops; `QUANT`'s amax of the bf16 row.
    chunk_rv m;
    warp::zero(m);
#pragma unroll
    for (int i = 0; i < WARP_CHUNKS; i++) {
        const int c = w + i * WARPS;
        if (c < g.chunks) {
            warp::add(gs[i], gs[i], g.offset);
            warp::mul(xs[i], xs[i], inv);
            warp::mul(xs[i], xs[i], gs[i]);
            chunk_bf yb;
            warp::copy(yb, xs[i]);
            if constexpr (FORM == NORM_ADD) {
                warp::copy(xs[i], yb);
                warp::add(xs[i], xs[i], os[i]);
                warp::store(g.y, xs[i], {0, 0, r, c});
            } else {
                warp::store(g.y, yb, {0, 0, r, c});
                if constexpr (QUANT) {
                    // The bf16 row, widened back exactly: what `quant_rows_fp8` reads.
                    warp::copy(xs[i], yb);
                    chunk_rv a;
                    warp::abs(a, xs[i]);
                    warp::max(m, m, a);
                }
            }
        }
    }

    // 4. The row's e4m3 bytes and scale.
    if constexpr (QUANT) {
        const float wmax = warp::max(m);
        if (warp::laneid() == 0) {
            amaxes[w] = wmax;
        }
        __syncthreads();
        warps_rv am;
        warp::load(am, amaxes);
        const float scale = scratchy::ops::fp8_scale::op<float>(warp::max(am));
        if (threadIdx.x == 0) {
            g.scale[{0, 0, 0, r}] = scale;
        }
#pragma unroll
        for (int i = 0; i < WARP_CHUNKS; i++) {
            const int c = w + i * WARPS;
            if (c < g.chunks) {
                warp::bin_op<scratchy::ops::fp8_quant_pre>(xs[i], xs[i], scale);
                warp::store(g.q, xs[i], {0, 0, r, c});
            }
        }
    }
}

template <int FORM, bool QUANT>
int launch(const n3_globals& G, uint32_t rows, cudaStream_t s) {
    n3_globals g = G;
    void* args[] = {&g};
    return static_cast<int>(cudaLaunchKernel(rms_norm_quant_row_ker<FORM, QUANT>, dim3(rows),
                                             dim3(NUM_THREADS), args, 0, s));
}

}  // namespace

// The kernel's one-time setup, run once per device by `kernels::prepare_all`: query each
// instance's attributes, which loads its module, so a graph capture never loads one mid-capture.
// 0 = done; any other value = the first failing cudaError_t.
extern "C" int scratchy_cuda_rms_norm_quant_row_prepare(void) {
    cudaFuncAttributes attr;
    for (const cudaError_t e : {
             cudaFuncGetAttributes(&attr, rms_norm_quant_row_ker<NORM, false>),
             cudaFuncGetAttributes(&attr, rms_norm_quant_row_ker<NORM, true>),
             cudaFuncGetAttributes(&attr, rms_norm_quant_row_ker<ADD_NORM, false>),
             cudaFuncGetAttributes(&attr, rms_norm_quant_row_ker<ADD_NORM, true>),
             cudaFuncGetAttributes(&attr, rms_norm_quant_row_ker<NORM_ADD, false>),
         }) {
        if (e != cudaSuccess) {
            return static_cast<int>(e);
        }
    }
    return 0;
}

// 0 = launched; SCRATCHY_CUDA_ERR_SHAPE = refused, nothing launched; SCRATCHY_CUDA_ERR_HOST = a
// C++ exception was caught; any other value = the launch's cudaError_t.
//
// Per row of `x [rows, cols]` bf16: `form` 0 (`y = norm(x)`), 1 (`sum = bf16(x + other)`, `y =
// norm(sum)`) or 2 (`y = bf16(bf16(norm(x)) + other)`), `norm(v) = v · rsqrt(mean(v²) + eps) ·
// (gain_offset + gain)` rounded once to bf16; with `q` (forms 0 and 1), also `q [rows, cols]` e4m3
// and `scale [rows]` f32: `quant_rows_fp8` of `y`, bit for bit. Refused before any pointer is
// touched: `rows` zero or above 65,535; `cols` zero, not a multiple of 256 or above 8,192; `form`
// above 2; `eps` not finite and positive; `gain_offset` not finite; `x`, `gain` or `y` null;
// `other` null at forms 1 and 2, `sum` null at form 1; exactly one of `q` and `scale` null; `q` at
// form 2. Pointers 16-byte-aligned device memory, the outputs overlapping no input. Asynchronous on
// `stream`.
extern "C" int scratchy_cuda_rms_norm_quant_row(const void* x, const void* other, const void* gain,
                                                void* y, void* sum, void* q, void* scale,
                                                uint32_t rows, uint32_t cols, uint32_t form,
                                                float eps, float gain_offset,
                                                void* stream /* cudaStream_t; null = default */) {
    const bool quant = q != nullptr;
    if (rows == 0 || rows > 65535 || cols == 0 || cols % uint32_t(CHUNK) != 0 || cols > MAX_COLS ||
        form > uint32_t(NORM_ADD) || !std::isfinite(eps) || !(eps > 0.0f) ||
        !std::isfinite(gain_offset) || x == nullptr || gain == nullptr || y == nullptr ||
        (form != uint32_t(NORM) && other == nullptr) ||
        (form == uint32_t(ADD_NORM) && sum == nullptr) || quant != (scale != nullptr) ||
        (quant && form == uint32_t(NORM_ADD))) {
        return SCRATCHY_CUDA_ERR_SHAPE;
    }
    // ThunderKittens' host code can throw; an exception must never unwind into Rust.
    try {
        const auto bf = [](const void* p) { return static_cast<bf16*>(const_cast<void*>(p)); };
        // Views a form does not read or write point at `x` or `y`, valid addresses never touched.
        const void* o = other != nullptr ? other : x;
        void* s = sum != nullptr ? sum : y;
        n3_globals G{
            row_gl{bf(x), nullptr, nullptr, size_t(rows), size_t(cols)},
            row_gl{bf(o), nullptr, nullptr, size_t(rows), size_t(cols)},
            gain_gl{bf(gain), nullptr, nullptr, nullptr, size_t(cols)},
            row_gl{bf(y), nullptr, nullptr, size_t(rows), size_t(cols)},
            row_gl{bf(s), nullptr, nullptr, size_t(rows), size_t(cols)},
            q_gl{static_cast<fp8e4m3*>(quant ? q : y), nullptr, nullptr, size_t(rows),
                 size_t(cols)},
            scale_gl{static_cast<float*>(quant ? scale : y), nullptr, nullptr, nullptr,
                     size_t(rows)},
            int(cols / uint32_t(CHUNK)),
            eps,
            1.0f / float(cols),
            gain_offset};
        const cudaStream_t st = static_cast<cudaStream_t>(stream);
        switch (form) {
            case uint32_t(NORM):
                return quant ? launch<NORM, true>(G, rows, st) : launch<NORM, false>(G, rows, st);
            case uint32_t(ADD_NORM):
                return quant ? launch<ADD_NORM, true>(G, rows, st)
                             : launch<ADD_NORM, false>(G, rows, st);
            default:
                return launch<NORM_ADD, false>(G, rows, st);
        }
    } catch (...) {
        return SCRATCHY_CUDA_ERR_HOST;
    }
}
