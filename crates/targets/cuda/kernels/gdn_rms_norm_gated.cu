// SPDX-License-Identifier: Apache-2.0
//
// Gated-DeltaNet gated RMSNorm for the H100, one prefill chunk of one sequence: launch 4 of the
// four a `GatedDeltaNet` step makes (`src/step.rs`). Per token `t` and value head `h`, over the
// head's `head_v` lanes of the scan's f32 `o [rows, value_dim]` and the gate `z [rows, value_dim]`
// (bf16), with the bundle's f32 gain `w [head_v]`:
//
//     out[t, h·head_v + i] = o[i] / sqrt(mean(o²) + eps) · w[i] · SiLU(z[i])     (bf16)
//
// in f32, rounded once (metal's `gdn_rms_norm_gated.metal`). The gate is SiLU, not sigmoid, and
// the gain is plain, not the zero-centred `1 + w` of qwen3.5's other norms.
//
// ThunderKittens' warp vector (PLAN-tk100 §2 S4, §3.15): one warp per `(t, h)` row, 8 warps per
// CTA, instantiated for `head_v` 32, 64, …, 256 (`HV`). The row's `o` is loaded as a naive
// `rv_fl<HV>` (`warp::load` from the f32 `gl`, viewed `[rows · nv, head_v]`), its squares by
// `warp::mul` and their sum by `warp::sum` (each lane's elements in order, then a fixed shuffle
// tree down 16, 8, 4, 2, 1 into lane 0 and a broadcast, `ops/group/register/vec/reductions.cuh`),
// so every lane holds the same sum and the result does not depend on the launch shape. The
// inverse root is the existing op `scratchy::ops::rms_inv` (`rsqrtf(m + eps)`) after the scalar
// `/ head_v`. The gate `z` is loaded (bf16 to f32, exact) and mapped by
// `unary_op<scratchy::ops::silu>`, the gain loaded once, and `o · inv · w · SiLU(z)` taken in that
// order by `warp::mul`, then stored with one round-to-nearest-even (`warp::store` to a bf16 `gl`).
//
// THE ORDER, a new fixed one: each square is rounded by `warp::mul`, then summed by `warp::sum`'s
// tree. The plain-CUDA kernel fused each square into its lane's sum with `fmaf`, then an xor
// butterfly. Different bits, the same bound (`2⁻⁸·|ref| + 2⁻²⁴`).
//
// PADDING ROWS (`t ≥ q_len`) ARE WRITTEN `+0.0` WITHOUT READING `o` OR `z` (`warp::zero` then
// `warp::store`): the scan wrote no `o` there and `z`'s padding rows may hold NaN, which `0 · NaN`
// would carry. So the step's whole output is defined, and nothing of a padding row reaches it.
//
// Built with `--use_fast_math` (build.rs): `rsqrtf`, `expf` and the division are the approximate
// ones; the row kernels' bound (2⁻⁸·|ref| + 2⁻²⁴) allows for them.

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

constexpr int WARPS = 8;
constexpr int NUM_THREADS = WARPS * kittens::WARP_THREADS;
// `step::GDN_HEAD_V_MULTIPLE` and `step::GDN_HEAD_V_MAX`.
constexpr uint32_t HEAD_V_MULTIPLE = 32;
constexpr uint32_t HEAD_V_MAX = 256;

// `o`, `z` and `out` viewed `[rows · nv, head_v]`: one `(t, h)` row each.
using o_gl = gl<float, 1, 1, -1, -1>;
using z_gl = gl<bf16, 1, 1, -1, -1>;
using w_gl = gl<float, 1, 1, 1, -1>;
// `[slot, q_len, start, 0]`.
using seq_gl = gl<int, 1, 1, 1, -1>;

struct norm_globals {
    o_gl o;
    z_gl z;
    w_gl w;
    seq_gl sequence;
    z_gl out;
    // `rows · nv`, the `(t, h)` rows.
    int head_rows;
    int nv;
    int head_v;
    float eps;
};

template <int HV>
__global__ __launch_bounds__(NUM_THREADS) void gdn_rms_norm_gated_ker(
    const __grid_constant__ norm_globals g) {
    const int row = blockIdx.x * WARPS + kittens::warpid();  // (t, h) as t · nv + h
    if (row >= g.head_rows) {
        return;
    }
    const int t = row / g.nv;
    rv_fl<HV> v;
    if (t >= g.sequence[{0, 0, 0, 1}]) {
        // A padding row: written, never read.
        warp::zero(v);
        warp::store(g.out, v, {0, 0, row, 0});
        return;
    }
    rv_fl<HV> sq, z, w;
    warp::load(v, g.o, {0, 0, row, 0});
    warp::mul(sq, v, v);
    const float m = warp::sum(sq) / float(g.head_v);
    const float inv = scratchy::ops::rms_inv::op<float>(m, g.eps);
    warp::load(z, g.z, {0, 0, row, 0});
    warp::unary_op<scratchy::ops::silu>(z, z);
    warp::load(w, g.w, {0, 0, 0, 0});
    warp::mul(v, v, inv);
    warp::mul(v, v, w);
    warp::mul(v, v, z);
    warp::store(g.out, v, {0, 0, row, 0});
}

template <int HV>
int launch(const norm_globals& G, cudaStream_t stream) {
    void* args[] = {const_cast<norm_globals*>(&G)};
    const dim3 grid((uint32_t(G.head_rows) + uint32_t(WARPS) - 1) / uint32_t(WARPS)),
        block(NUM_THREADS);
    return static_cast<int>(
        cudaLaunchKernel(gdn_rms_norm_gated_ker<HV>, grid, block, args, 0, stream));
}

}  // namespace

// 0 = launched; SCRATCHY_CUDA_ERR_SHAPE = shape refused, nothing launched;
// SCRATCHY_CUDA_ERR_HOST = a C++ exception was caught; any other value = the cudaError_t
// cudaLaunchKernel returned for THIS launch. Nothing here reads or clears the runtime's last-error
// state.
//
// `rows` and `nv` must be non-zero, `head_v` a non-zero multiple of 32 and at most 256,
// `rows · nv · head_v` at most INT_MAX, and `eps` finite and above 0; all are checked before any
// pointer is touched. `o` is `[rows, nv · head_v]` f32, `z` and `out` the same shape in bf16, `w`
// `[head_v]` f32, `sequence` `[slot, q_len, start, 0]` u32 with `q_len ≤ rows`; `out` overlaps no
// input. No shared memory, so no one-time setup. The launch is asynchronous on `stream`.
extern "C" int scratchy_cuda_gdn_rms_norm_gated(const void* o, const void* z, const void* w,
                                                const void* sequence, void* out, uint32_t rows,
                                                uint32_t nv, uint32_t head_v, float eps,
                                                void* stream /* cudaStream_t; null = default */) {
    if (rows == 0 || nv == 0 || head_v == 0 || head_v % HEAD_V_MULTIPLE != 0 ||
        head_v > HEAD_V_MAX ||
        uint64_t(rows) * uint64_t(nv) * uint64_t(head_v) > uint64_t(INT_MAX) ||
        !std::isfinite(eps) || !(eps > 0.0f)) {
        return SCRATCHY_CUDA_ERR_SHAPE;
    }
    // ThunderKittens' host code can throw; an exception must never unwind into Rust.
    try {
        const size_t head_rows = size_t(rows) * nv;
        // No view carries a TMA type, so building them encodes nothing: they are the pointers and
        // the dimensions the kernel indexes with.
        o_gl og{static_cast<float*>(const_cast<void*>(o)), nullptr, nullptr, head_rows,
                size_t(head_v)};
        z_gl zg{static_cast<bf16*>(const_cast<void*>(z)), nullptr, nullptr, head_rows,
                size_t(head_v)};
        w_gl wg{static_cast<float*>(const_cast<void*>(w)), nullptr, nullptr, nullptr,
                size_t(head_v)};
        seq_gl qg{static_cast<int*>(const_cast<void*>(sequence)), nullptr, nullptr, nullptr,
                  size_t(4)};
        z_gl outg{static_cast<bf16*>(out), nullptr, nullptr, head_rows, size_t(head_v)};
        norm_globals G{og, zg, wg, qg, outg, int(head_rows), int(nv), int(head_v), eps};
        const cudaStream_t s = static_cast<cudaStream_t>(stream);
        switch (head_v) {
            case 32: return launch<32>(G, s);
            case 64: return launch<64>(G, s);
            case 96: return launch<96>(G, s);
            case 128: return launch<128>(G, s);
            case 160: return launch<160>(G, s);
            case 192: return launch<192>(G, s);
            case 224: return launch<224>(G, s);
            default: return launch<256>(G, s);
        }
    } catch (...) {
        return SCRATCHY_CUDA_ERR_HOST;
    }
}
