// SPDX-License-Identifier: Apache-2.0
//
// Gated-DeltaNet gating for the H100, one prefill chunk of one sequence: launch 2 of the four a
// `GatedDeltaNet` step makes (`src/step.rs`). For each real token `t < q_len` and value head
// `h < nv`, from `a` and `b` `[rows, nv]` (bf16) and the bundle's f32 `a_log [nv]` and `dt_bias
// [nv]`:
//
//     g[t, h]    = −exp(a_log[h]) · softplus(a[t, h] + dt_bias[h])     (the log of the decay α)
//     beta[t, h] = 1 / (1 + exp(−b[t, h]))                              (β = sigmoid(b))
//
// with `softplus(x) = log1p(exp(x))` for `x ≤ 20`, else `x` (metal's `gdn_gating.metal`, its
// threshold 20). Both are f32, into the step's scratch.
//
// ThunderKittens' warp vector (PLAN-tk100 §2 S4, §3.13): one warp per real token row, 8 warps per
// CTA, instantiated for `nv` 16 and 32 (`step::GDN_GATING_NV_MULTIPLE`). The row's `a` and `b` are
// loaded as naive `rv_fl<NV>` (`warp::load`, bf16 to f32, exact; at `NV` 16 lanes 16..31 hold
// nothing), `a_log` and `dt_bias` once per warp the same way. Then `x = a + dt_bias`
// (`warp::add`), `g = bin_op<scratchy::ops::gdn_g>(x, a_log)` (the plain-CUDA kernel's whole
// expression in one op, so its order and rounding are kept), `β = unary_op<scratchy::ops::sigmoid>(b)`,
// and both stored f32 (`warp::store`). No reduction: each element's result depends on its own
// inputs only. A warp at or past `q_len` returns before reading anything, so padding rows are
// never read nor written.
//
// Built with `--use_fast_math` (build.rs): `expf` is `__expf` and the division approximate.
// `log1pf` has no fast-math intrinsic and stays the library function. The tests' bound
// (2⁻¹⁶·|ref| + 2⁻²² for `g` and for `β`) allows for both.

#include "kittens.cuh"
#include "scratchy_ops.cuh"

#include <climits>
#include <cstdint>

using namespace kittens;

// Launcher return codes that are not a `cudaError_t`. `src/kernels.rs` mirrors these two values.
#define SCRATCHY_CUDA_ERR_SHAPE (-1)
#define SCRATCHY_CUDA_ERR_HOST (-2)

namespace {

constexpr int WARPS = 8;
constexpr int NUM_THREADS = WARPS * kittens::WARP_THREADS;
// `step::GDN_GATING_NV_MULTIPLE`: `nv` is 16 or 32, one naive vector of TK's 16-element tiles.
constexpr uint32_t NV_MULTIPLE = 16;

using in_gl = gl<bf16, 1, 1, -1, -1>;
using vec_gl = gl<float, 1, 1, 1, -1>;
using out_gl = gl<float, 1, 1, -1, -1>;
// `[slot, q_len, start, 0]`.
using seq_gl = gl<int, 1, 1, 1, -1>;

struct gating_globals {
    // `[rows, nv]` each.
    in_gl a;
    in_gl b;
    // `[nv]` each.
    vec_gl a_log;
    vec_gl dt_bias;
    seq_gl sequence;
    // `[rows, nv]` each.
    out_gl g;
    out_gl beta;
    int rows;
};

template <int NV>
__global__ __launch_bounds__(NUM_THREADS) void gdn_gating_ker(const __grid_constant__ gating_globals g) {
    const int t = blockIdx.x * WARPS + kittens::warpid();
    const int q_len = g.sequence[{0, 0, 0, 1}];
    if (t >= g.rows || t >= q_len) {
        return;
    }
    rv_fl<NV> av, bv, a_log, dt_bias;
    warp::load(av, g.a, {0, 0, t, 0});
    warp::load(bv, g.b, {0, 0, t, 0});
    warp::load(a_log, g.a_log, {0, 0, 0, 0});
    warp::load(dt_bias, g.dt_bias, {0, 0, 0, 0});
    warp::add(av, av, dt_bias);
    warp::bin_op<scratchy::ops::gdn_g>(av, av, a_log);
    warp::unary_op<scratchy::ops::sigmoid>(bv, bv);
    warp::store(g.g, av, {0, 0, t, 0});
    warp::store(g.beta, bv, {0, 0, t, 0});
}

template <int NV>
int launch(const gating_globals& G, uint32_t rows, cudaStream_t stream) {
    void* args[] = {const_cast<gating_globals*>(&G)};
    const dim3 grid((rows + uint32_t(WARPS) - 1) / uint32_t(WARPS)), block(NUM_THREADS);
    return static_cast<int>(cudaLaunchKernel(gdn_gating_ker<NV>, grid, block, args, 0, stream));
}

}  // namespace

// 0 = launched; SCRATCHY_CUDA_ERR_SHAPE = shape refused, nothing launched;
// SCRATCHY_CUDA_ERR_HOST = a C++ exception was caught; any other value = the cudaError_t
// cudaLaunchKernel returned for THIS launch. Nothing here reads or clears the runtime's last-error
// state.
//
// `rows` and `nv` must be non-zero, `nv` 16 or 32 (a multiple of 16, `step::GDN_GATING_NV_MULTIPLE`,
// the widths the kernel is instantiated for), and `rows · nv` at most INT_MAX; checked before any
// pointer is touched. `a` and `b` are `[rows, nv]` bf16, `a_log` and `dt_bias` `[nv]` f32, `g` and
// `beta` `[rows, nv]` f32, `sequence` `[slot, q_len, start, 0]` u32 with `1 ≤ q_len ≤ rows`; `g`
// and `beta` overlap nothing else. No shared memory, so no one-time setup. The launch is
// asynchronous on `stream`.
extern "C" int scratchy_cuda_gdn_gating(const void* a, const void* b, const void* a_log,
                                        const void* dt_bias, const void* sequence, void* g,
                                        void* beta, uint32_t rows, uint32_t nv,
                                        void* stream /* cudaStream_t; null = default stream */) {
    if (rows == 0 || nv == 0 || nv % NV_MULTIPLE != 0 || nv > 2 * NV_MULTIPLE ||
        uint64_t(rows) * uint64_t(nv) > uint64_t(INT_MAX)) {
        return SCRATCHY_CUDA_ERR_SHAPE;
    }
    // ThunderKittens' host code can throw; an exception must never unwind into Rust.
    try {
        // No view carries a TMA type, so building them encodes nothing: they are the pointers and
        // the dimensions the kernel indexes with.
        in_gl ag{static_cast<bf16*>(const_cast<void*>(a)), nullptr, nullptr, size_t(rows),
                 size_t(nv)};
        in_gl bg{static_cast<bf16*>(const_cast<void*>(b)), nullptr, nullptr, size_t(rows),
                 size_t(nv)};
        vec_gl lg{static_cast<float*>(const_cast<void*>(a_log)), nullptr, nullptr, nullptr,
                  size_t(nv)};
        vec_gl dg{static_cast<float*>(const_cast<void*>(dt_bias)), nullptr, nullptr, nullptr,
                  size_t(nv)};
        seq_gl qg{static_cast<int*>(const_cast<void*>(sequence)), nullptr, nullptr, nullptr,
                  size_t(4)};
        out_gl gg{static_cast<float*>(g), nullptr, nullptr, size_t(rows), size_t(nv)};
        out_gl btg{static_cast<float*>(beta), nullptr, nullptr, size_t(rows), size_t(nv)};
        gating_globals G{ag, bg, lg, dg, qg, gg, btg, int(rows)};
        const cudaStream_t s = static_cast<cudaStream_t>(stream);
        return nv == NV_MULTIPLE ? launch<16>(G, rows, s) : launch<32>(G, rows, s);
    } catch (...) {
        return SCRATCHY_CUDA_ERR_HOST;
    }
}
