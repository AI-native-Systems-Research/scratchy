// SPDX-License-Identifier: Apache-2.0
//
// The one-row activation and its quantization (FP8 S8 T1, /tmp/a-tc-plan.md §2 option C; no tape
// launches it yet): for each row `r` of the gate/up GEMV's f32 outputs `gate`, `up [rows, n]` (the
// tensor-core GEMV's f32 store, `gemv_fp8_tc.cu`'s B8),
//
//     act[r, c] = bf16(OP(gate[r, c], up[r, c]))         OP = silu_mul or gelu_tanh_mul
//     scale[r]  = quant_rows_fp8's scale of act[r, :]
//     q[r, c]   = quant_rows_fp8's e4m3 byte of act[r, c]
//
// `OP` in f32 on the two scaled sums and one rounding to bf16: today's gate/up fold
// (`gemv_fp8.cu`'s `gate_up_gemv_fp8_ker`), whose one rounding the f32 store preserves. Then that
// bf16 row's per-token quantization, bit for bit `quant_rows_fp8`'s:
//   - `amax`, a max of absolute values, exact in any order; `scratchy::ops::fp8_scale` its scale;
//   - each element `fp8_quant_pre` (vLLM's division and clamp), then TK's e4m3 convertor (round to
//     nearest even, saturating), the conversion `quant_rows_fp8`'s `warp::copy` to `rt_fp8e4m3`
//     and the fp8 GEMV's `fp8_round` both make.
//
// `quant_rows_fp8.cu`'s one-row form (Q1): one CTA of 8 warps per 256-column chunk of a row, CTA
// `(t, r)` owning chunk `t` of row `r`:
//   1. Every CTA computes the whole row's `act` and `amax`: warp `w` takes the row's chunks
//      `w, w + 8, …` (at most 9: `n` at most 17,408, the 27B's intermediate). Its gate and up
//      chunks are TMA-loaded into shared memory in one burst (TK's `sv_fl<256>` boxes: the `n`
//      tail is TMA's zero fill, and `OP(0, 0) = +0` adds nothing to the max), `warp::load`ed into
//      naive `rv_fl<256>`, `OP`'d with `warp::bin_op` and rounded with `round_bf16`
//      (`with_residual`'s first rounding: bf16, widened back exactly), and `warp::store`d back over
//      the gate chunk. The row's `amax` goes through an `sv_fl<16>` (each warp's twice), every warp
//      reduces the 16.
//   2. Each of the CTA's 256 threads quantizes one element of the owned chunk, read back from
//      shared memory (so the same `act` the max saw), and writes its byte and, when `act` is given,
//      its bf16 through the `gl` accessor, columns past `n` skipped.
// (The earlier form, one CTA a row with each lane quantizing up to 72 elements, took ~25 µs at
// 17,408: Q1's ~64-dependent-divisions slow path.)
//
// Shared memory: 8 warps × 9 chunks × 2 × 1 KiB, plus 1,024 bytes of alignment slack, set once by
// `scratchy_cuda_act_quant_fp8_prepare`. The semaphores and the amax vector are static.

#include "kittens.cuh"
#include "scratchy_ops.cuh"

#include <climits>
#include <cstdint>

using namespace kittens;

// The launcher's return codes that are not a `cudaError_t`. `src/kernels.rs` mirrors them.
#define SCRATCHY_CUDA_ERR_SHAPE (-1)
#define SCRATCHY_CUDA_ERR_HOST (-2)

namespace {

// `step::GEMM_FP8_K_MULTIPLE`: `n` is a multiple of it, so every row stride is a multiple of 16
// bytes, as the tensor maps need.
constexpr uint32_t COL_MULTIPLE = 64;
constexpr int CHUNK = 256;
constexpr int WARPS = 8;
constexpr int THREADS = WARPS * kittens::WARP_THREADS;
static_assert(THREADS == CHUNK, "one element of the owned chunk a thread");
// The longest row's chunks (17,408 / 256), and a warp's share of them.
constexpr int ROW_CHUNKS = 68;
constexpr int WARP_CHUNKS = (ROW_CHUNKS + WARPS - 1) / WARPS;
// `step::GatedOp`'s codes.
constexpr uint32_t ACT_SILU = 0;
constexpr uint32_t ACT_GELU = 1;

using chunk_sv = sv_fl<CHUNK>;
using chunk_rv = rv_fl<CHUNK>;
using amax_rv = rv_fl<16>;

using in_gl = gl<float, 1, 1, -1, -1, chunk_sv>;
using q_gl = gl<fp8e4m3, 1, 1, -1, -1>;
using act_gl = gl<bf16, 1, 1, -1, -1>;
using scale_gl = gl<float, 1, 1, 1, -1>;

struct act_quant_globals {
    in_gl gate;
    in_gl up;
    q_gl q;
    scale_gl scale;
    act_gl act;
    int n;
    int chunks;
    // Whether `act` is written.
    int store_act;
};

constexpr size_t SMEM_USED = size_t(WARPS) * WARP_CHUNKS * 2 * sizeof(chunk_sv);
constexpr int SMEM_BYTES = int(SMEM_USED) + 1024;
static_assert(SMEM_BYTES <= kittens::MAX_SHARED_MEMORY, "fits the H100's 227 KiB");

template <typename OP>
__global__ __launch_bounds__(THREADS, 1) void act_quant_fp8_ker(
    const __grid_constant__ act_quant_globals g) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);
    chunk_sv(&gate_s)[WARPS][WARP_CHUNKS] = al.allocate<chunk_sv, WARPS, WARP_CHUNKS>();
    chunk_sv(&up_s)[WARPS][WARP_CHUNKS] = al.allocate<chunk_sv, WARPS, WARP_CHUNKS>();
    __shared__ kittens::semaphore arrived[WARPS];
    __shared__ sv_fl<16> amax_s;
    const int w = kittens::warpid();
    const int t_own = blockIdx.x;
    const int r = blockIdx.y;
    const int mine = (g.chunks - w + WARPS - 1) / WARPS;
    if (warp::laneid() == 0) {
        init_semaphore(arrived[w], 0, 1);
    }
    __syncwarp();
    warp::tma::expect_bytes(arrived[w], uint32_t(2 * mine * sizeof(chunk_sv)));
    for (int i = 0; i < mine; i++) {
        warp::tma::load_async(gate_s[w][i], g.gate, {0, 0, r, w + i * WARPS}, arrived[w]);
        warp::tma::load_async(up_s[w][i], g.up, {0, 0, r, w + i * WARPS}, arrived[w]);
    }
    chunk_rv a, u, m;
    warp::zero(m);
    if (mine > 0) {
        wait(arrived[w], 0);
    }
    // Pass 1: this warp's chunks' `act`, kept in shared memory over the gate, and their max.
    for (int i = 0; i < mine; i++) {
        warp::load(a, gate_s[w][i]);
        warp::load(u, up_s[w][i]);
        warp::bin_op<OP>(a, a, u);
        warp::unary_op<scratchy::ops::round_bf16>(a, a);
        warp::store(gate_s[w][i], a);
        warp::abs(u, a);
        warp::max(m, m, u);
    }
    const float part = warp::max(m);
    if (warp::laneid() == 0) {
        amax_s[w] = part;
        amax_s[w + WARPS] = part;
    }
    // Also orders every warp's `act` store before pass 2 reads the owned chunk.
    __syncthreads();
    amax_rv all;
    warp::load(all, amax_s);
    const float scale = scratchy::ops::fp8_scale::op<float>(warp::max(all));
    if (t_own == 0 && threadIdx.x == 0) {
        g.scale[{0, 0, 0, r}] = scale;
    }
    // Pass 2, one element a thread: the owned chunk is warp `t_own % 8`'s load `t_own / 8`.
    const int col = t_own * CHUNK + int(threadIdx.x);
    if (col < g.n) {
        const float v = gate_s[t_own % WARPS][t_own / WARPS][int(threadIdx.x)];
        if (g.store_act) {
            g.act[{0, 0, r, col}] = base_types::convertor<bf16, float>::convert(v);
        }
        g.q[{0, 0, r, col}] = base_types::convertor<fp8e4m3, float>::convert(
            scratchy::ops::fp8_quant_pre::op<float>(v, scale));
    }
}

}  // namespace

// The kernel's one-time setup: allow both instances `SMEM_BYTES` of dynamic shared memory.
// 0 = done; any other value = the first failing cudaError_t.
extern "C" int scratchy_cuda_act_quant_fp8_prepare(void) {
    cudaError_t e = cudaFuncSetAttribute(act_quant_fp8_ker<scratchy::ops::silu_mul>,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
    if (e != cudaSuccess) {
        return static_cast<int>(e);
    }
    return static_cast<int>(cudaFuncSetAttribute(act_quant_fp8_ker<scratchy::ops::gelu_tanh_mul>,
                                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                 SMEM_BYTES));
}

// 0 = launched; SCRATCHY_CUDA_ERR_SHAPE = refused, nothing launched; SCRATCHY_CUDA_ERR_HOST = a
// C++ exception was caught; any other value = the launch's cudaError_t.
//
// `rows` rows of `act = bf16(OP(gate, up))` from `gate`, `up [rows, n]` f32, then their
// `quant_rows_fp8` bytes `q [rows, n]` e4m3 and scales `scale [rows]` f32, bit for bit; `act [rows,
// n]` bf16 written too when it is not null. `op` 0 SwiGLU, 1 GeGLU (`step::GatedOp`). Refused:
// `rows` zero or above 65,535; `n` zero, not a multiple of 64 or above 17,408; `op` another value;
// a size above INT_MAX; a null pointer but `act`. Pointers 16-byte-aligned device memory, no output
// overlapping an input. `scratchy_cuda_act_quant_fp8_prepare` must have succeeded. Asynchronous.
extern "C" int scratchy_cuda_act_quant_fp8(const void* gate, const void* up, void* q, void* scale,
                                           void* act, uint32_t rows, uint32_t n, uint32_t op,
                                           void* stream /* cudaStream_t; null = default */) {
    if (rows == 0 || rows > 65535 || n == 0 || n % COL_MULTIPLE != 0 ||
        n > uint32_t(ROW_CHUNKS * CHUNK) || (op != ACT_SILU && op != ACT_GELU) ||
        uint64_t(rows) * n > uint64_t(INT_MAX) || gate == nullptr || up == nullptr ||
        q == nullptr || scale == nullptr) {
        return SCRATCHY_CUDA_ERR_SHAPE;
    }
    // ThunderKittens' host code can throw; an exception must never unwind into Rust.
    try {
        const auto f = [](const void* p) { return static_cast<float*>(const_cast<void*>(p)); };
        act_quant_globals G{
            in_gl{f(gate), nullptr, nullptr, size_t(rows), size_t(n)},
            in_gl{f(up), nullptr, nullptr, size_t(rows), size_t(n)},
            q_gl{static_cast<fp8e4m3*>(q), nullptr, nullptr, size_t(rows), size_t(n)},
            scale_gl{static_cast<float*>(scale), nullptr, nullptr, nullptr, size_t(rows)},
            // With no `act` the view is never written; `q` is a valid address for it.
            act_gl{static_cast<bf16*>(act != nullptr ? act : q), nullptr, nullptr, size_t(rows),
                   size_t(n)},
            int(n),
            int((n + uint32_t(CHUNK) - 1) / uint32_t(CHUNK)),
            act != nullptr ? 1 : 0};
        void* args[] = {&G};
        const dim3 grid{(n + uint32_t(CHUNK) - 1) / uint32_t(CHUNK), rows}, block{uint32_t(THREADS)};
        const auto ker = op == ACT_SILU ? act_quant_fp8_ker<scratchy::ops::silu_mul>
                                        : act_quant_fp8_ker<scratchy::ops::gelu_tanh_mul>;
        return static_cast<int>(cudaLaunchKernel(ker, grid, block, args, SMEM_BYTES,
                                                 static_cast<cudaStream_t>(stream)));
    } catch (...) {
        return SCRATCHY_CUDA_ERR_HOST;
    }
}

// ── P1: the MoE prefill's act + quantize over live rows (FP8 S8 P1) ──
//
// The expert down's input at prefill: for each row `r` below the sort's live rows (`offsets[E]`,
// its padded total) of the gate's and the up's bf16 outputs `gate`, `up [rows, cols]` (the expert
// GEMMs'),
//
//     act[r, c] = bf16(OP(gate[r, c], up[r, c]))       `silu_mul_bf16` / `gelu_mul_bf16`'s value
//     q[r, :], scale[r] = quant_rows_fp8 of act[r, :]
//
// `OP` on the two widened bf16 values (exact) in f32, rounded once to bf16 (`round_bf16`, RNE, the
// rounding the act kernel's store makes), the op struct `silu_mul_bf16_ker`'s map applies per
// element (its float2 form is the float one on each half). Then `quant_rows_fp8`'s ops: the row's
// max of absolute values (exact in any order), `fp8_scale`, and each element's `fp8_quant_pre`
// with TK's scalar e4m3 convertor (Q1's). So the bytes and scales of every live row (padding rows
// included: their gate and up are +0, so their act is +0) are today's act then quantize; a row at
// or past `offsets[E]` is never written, and a CTA whose first row is past it exits at once.
//
// One warp per row, 8 rows per CTA; lane `l` takes columns `l, l + 32, …` through the `gl`
// accessors (a row is at most 704 wide: 22 a lane), the row's max met through a per-warp
// `sv_fl<32>` reduced by `warp::max`. The act is formed twice, for the max and for the bytes, from
// the same two loads' values. No dynamic shared memory: its one-time setup only loads it.

namespace {

using expert_act_gl = gl<bf16, 1, 1, -1, -1>;
using expert_idx_gl = gl<int, 1, 1, 1, -1>;

struct expert_act_quant_globals {
    expert_act_gl gate;
    expert_act_gl up;
    q_gl q;
    scale_gl scale;
    // The sort's `[1, E + 1]` offsets: entry `E` is the live rows.
    expert_idx_gl offsets;
    int rows;
    int cols;
    int experts;
};

template <typename OP>
__global__ __launch_bounds__(THREADS, 1) void expert_act_quant_fp8_ker(
    const __grid_constant__ expert_act_quant_globals g) {
    __shared__ sv_fl<kittens::WARP_THREADS> amax_s[WARPS];
    const int live = min(g.offsets[{0, 0, 0, g.experts}], g.rows);
    const int w = kittens::warpid(), lane = warp::laneid();
    const int r = int(blockIdx.x) * WARPS + w;
    if (r >= live) {
        return;
    }
    const auto act = [&](int c) {
        const float gv = base_types::convertor<float, bf16>::convert(g.gate[{0, 0, r, c}]);
        const float uv = base_types::convertor<float, bf16>::convert(g.up[{0, 0, r, c}]);
        return scratchy::ops::round_bf16::op<float>(OP::template op<float>(gv, uv));
    };
    float amax = 0.0f;
    for (int c = lane; c < g.cols; c += kittens::WARP_THREADS) {
        amax = base_ops::max::op<float>(amax, base_ops::abs::op<float>(act(c)));
    }
    amax_s[w][lane] = amax;
    __syncwarp();
    rv_fl<kittens::WARP_THREADS> all;
    warp::load(all, amax_s[w]);
    const float scale = scratchy::ops::fp8_scale::op<float>(warp::max(all));
    if (lane == 0) {
        g.scale[{0, 0, 0, r}] = scale;
    }
    for (int c = lane; c < g.cols; c += kittens::WARP_THREADS) {
        g.q[{0, 0, r, c}] = base_types::convertor<fp8e4m3, float>::convert(
            scratchy::ops::fp8_quant_pre::op<float>(act(c), scale));
    }
}

}  // namespace

// The expert act + quantize's one-time setup: query both instances' attributes, which loads their
// module. 0 = done; any other value = the first failing cudaError_t.
extern "C" int scratchy_cuda_expert_act_quant_fp8_prepare(void) {
    cudaFuncAttributes attr;
    cudaError_t e = cudaFuncGetAttributes(&attr, expert_act_quant_fp8_ker<scratchy::ops::silu_mul>);
    if (e != cudaSuccess) {
        return static_cast<int>(e);
    }
    return static_cast<int>(
        cudaFuncGetAttributes(&attr, expert_act_quant_fp8_ker<scratchy::ops::gelu_tanh_mul>));
}

// 0 = launched; SCRATCHY_CUDA_ERR_SHAPE = refused, nothing launched; SCRATCHY_CUDA_ERR_HOST = a
// C++ exception was caught; any other value = the launch's cudaError_t.
//
// FP8 S8 P1: rows `r < offsets[experts]` of `act = bf16(OP(gate, up))` from `gate`, `up [rows,
// cols]` bf16, quantized into `q [rows, cols]` e4m3 and `scale [rows]` f32, bit for bit
// `silu_mul_bf16` / `gelu_mul_bf16` then `quant_rows_fp8` of those rows; the rows at or past it
// are not written. `op` 0 SwiGLU, 1 GeGLU (`step::GatedOp`). Refused: `rows` zero, `cols` zero or
// not a multiple of 64, `experts` zero, `op` another value, `rows · cols` above INT_MAX, a null
// pointer. `gate`, `up` 2-byte aligned, `q` any, `scale` and `offsets` (`[experts + 1]` i32) 4-byte
// aligned device pointers; no output overlapping an input. Asynchronous.
extern "C" int scratchy_cuda_expert_act_quant_fp8(const void* gate, const void* up, void* q,
                                                  void* scale, const void* offsets, uint32_t rows,
                                                  uint32_t cols, uint32_t experts, uint32_t op,
                                                  void* stream /* cudaStream_t; null = default */) {
    if (rows == 0 || cols == 0 || cols % COL_MULTIPLE != 0 || experts == 0 ||
        (op != ACT_SILU && op != ACT_GELU) || uint64_t(rows) * cols > uint64_t(INT_MAX) ||
        gate == nullptr || up == nullptr || q == nullptr || scale == nullptr ||
        offsets == nullptr) {
        return SCRATCHY_CUDA_ERR_SHAPE;
    }
    // ThunderKittens' host code can throw; an exception must never unwind into Rust.
    try {
        const auto b = [](const void* p) { return static_cast<bf16*>(const_cast<void*>(p)); };
        // No view carries a TMA type, so building them encodes nothing.
        expert_act_quant_globals G{
            expert_act_gl{b(gate), nullptr, nullptr, size_t(rows), size_t(cols)},
            expert_act_gl{b(up), nullptr, nullptr, size_t(rows), size_t(cols)},
            q_gl{static_cast<fp8e4m3*>(q), nullptr, nullptr, size_t(rows), size_t(cols)},
            scale_gl{static_cast<float*>(scale), nullptr, nullptr, nullptr, size_t(rows)},
            expert_idx_gl{static_cast<int*>(const_cast<void*>(offsets)), nullptr, nullptr, nullptr,
                          size_t(experts) + 1},
            int(rows),
            int(cols),
            int(experts)};
        void* args[] = {&G};
        const dim3 grid{(rows + uint32_t(WARPS) - 1) / uint32_t(WARPS)}, block{uint32_t(THREADS)};
        const auto ker = op == ACT_SILU ? expert_act_quant_fp8_ker<scratchy::ops::silu_mul>
                                        : expert_act_quant_fp8_ker<scratchy::ops::gelu_tanh_mul>;
        return static_cast<int>(
            cudaLaunchKernel(ker, grid, block, args, 0, static_cast<cudaStream_t>(stream)));
    } catch (...) {
        return SCRATCHY_CUDA_ERR_HOST;
    }
}
