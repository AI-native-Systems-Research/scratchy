// SPDX-License-Identifier: Apache-2.0
//
// B8, FP8 S8's tensor-core decode GEMV for the H100 (T1, T2: the tapes' `GemvTcFp8`,
// `GemvGroupTcFp8`, `GateUpTcFp8`, `ExpertGemvTcFp8` and `ExpertGateUpTcFp8`): for up to three
// members off a quantized activation,
//
//     c_i[p, j] = bf16(((Σ_k q[p / row_div, k] · b_i[j, k]) · b_scale_i[j]) · a_scale[p / row_div])
//
// `q` e4m3 and `a_scale` f32 are `quant_rows_fp8`'s rows and scales (the activation is quantized
// before this launch); `b_i` e4m3 and `b_scale_i` f32 the checkpoint's weight and its per-channel
// scale. The epilogue's order is `gemm_fp8.cu`'s (`mul_col` by the weight scale, then by the token
// scale), rounded once.
//
// vLLM's geometry (its `cutlass_3x_gemm_sm90_fp8` at TileShape 64x16x256) at the smallest fp8 N TK
// has at the pin (32): the weight's 64-row, 256-wide tile is the WGMMA's M side
// (`warpgroup::mma_ABt`, e4m3 × e4m3, as `gemm_fp8.cu`) and the activation's `st_fp8e4m3<32, 256>`
// box its N side (1 real row, 31 rows of TMA zero fill, TK-1's R1 rule), so `D [64, 32]`'s column 0
// is the GEMV (`warp::row_sum`: the other columns are exact zeros). One CTA per 64-row tile over the
// whole K, no stream-K, its epilogue stored directly; a producer warp TMA-loads both boxes through a
// 4-stage ring, one consumer warpgroup runs the WGMMAs; two CTAs per SM (2 × 97 KB). On top of it:
//   - **The expert form** (`EXPERT`): grid `(tiles, pairs)`; pair `p` reads weight depth
//     `topk_idx[p]` of `[experts, expert_rows, k]` and activation row `p / row_div`
//     (`gemv_fp8.cu`'s `EXPERT` contract). The dense form is `experts` 1, `topk_idx` unread. A
//     pair is the dense GEMV on its slice, bit for bit: the same CTA on the same bytes.
//   - **The activation as `[rows, 1, k]`**: each row its own depth, so every box's rows past 0 are
//     TMA's zero fill even when the activation holds several rows (the expert pairs' 8). Without
//     it a box would read the next rows into `D`'s other columns, which the row sum adds.
//   - **The K tail**: `⌈k / 256⌉` steps. Both boxes are 128-byte swizzled, and TK's tensor map for
//     a swizzled tile bounds K in whole 128-byte atoms (TK-1's R1 rule): past `k`, a whole atom is
//     TMA's zero fill, but the rest of a partial one is read from memory past the row (the next
//     row's bytes). So when `k % 128 ≠ 0` the consumer zeroes those columns, `[tail_from, tail_to)`
//     of the last step's two tiles (64 a row: `k` is a multiple of 64), through the tile accessor,
//     before its WGMMAs (`b8_zero_tail`); a zero adds 0 exactly (`tests/gemv_fp8_b8.rs`'s
//     swizzled-box probe, which applies the same zeroing).
//   - **The residual epilogue** (`add`): `round_bf16`, the residual row widened, one f32 add, the
//     store's one rounding: `scratchy::ops::with_residual`'s bits (W4a's `MatvecEpilogue`).
//   - **The f32 store** (`F32`): the scaled f32 sums, unrounded, for the gate/up members, so the
//     act + quantize step after them rounds the activation once (`act_quant_fp8.cu`).
//   - Up to 3 members: each tile's member by `start1`/`start2`; each member is its own
//     launch's bits (no split: a tile's CTA does not depend on the other members).
// Bits are a function of the shape alone. They are not `gemv_fp8.cu`'s (WGMMA's fp8 accumulation
// and K grouping): the approved kind, against S3's f64 bound.
//
// Shared memory: 4 stages × (a 16 KB weight tile and an 8 KB activation tile) plus 1,024 bytes of
// alignment slack, set by `scratchy_cuda_gemv_fp8_tc_prepare`.

#include "kittens.cuh"
#include "scratchy_ops.cuh"

#include <climits>
#include <cstdint>
#include <type_traits>

using namespace kittens;

// The launcher's return codes that are not a `cudaError_t`. `src/kernels.rs` mirrors them.
#define SCRATCHY_CUDA_ERR_SHAPE (-1)
#define SCRATCHY_CUDA_ERR_HOST (-2)

namespace {

constexpr int NUM_THREADS = 2 * kittens::WARPGROUP_THREADS;
// `n` and `k` multiples of it: the scales' reads stay inside the arrays.
constexpr uint32_t N_MULTIPLE = 64;

// B8's instance: its K step, ring, row tile and CTAs per SM, and its two boxes.
struct tc_b8 {
    static constexpr int k_step = 256;
    static constexpr int stages = 4;
    static constexpr int tile = 64;
    static constexpr int min_blocks = 2;
    using a_tile = st_fp8e4m3<32, k_step>;
    using b_tile = st_fp8e4m3<tile, k_step>;
    static constexpr size_t smem = stages * (sizeof(a_tile) + sizeof(b_tile)) + 1024;
    static_assert(min_blocks * smem <= kittens::MAX_SHARED_MEMORY, "the CTAs per SM fit");
};

int tiles_of(uint32_t n, int tile) {
    return int((n + uint32_t(tile) - 1) / uint32_t(tile));
}

using b8_x_tile = tc_b8::a_tile;
using b8_w_tile = tc_b8::b_tile;
// The activation `[rows, 1, k]` e4m3, and a member's weight `[depth, rows, k]` e4m3.
using b8_x_gl = gl<fp8e4m3, 1, -1, 1, -1, b8_x_tile>;
using b8_w_gl = gl<fp8e4m3, 1, -1, -1, -1, b8_w_tile>;
// No tensor map: the weight scales `[depth, rows]`, the activation's `[rows]`, the outputs and
// residuals `[pairs, n]`, the experts `[pairs]`.
using b8_scale_gl = gl<float, 1, 1, -1, -1>;
using b8_ascale_gl = gl<float, 1, 1, 1, -1>;
template <bool F32>
using b8_out_gl = gl<std::conditional_t<F32, float, bf16>, 1, 1, -1, -1>;
using b8_res_gl = gl<bf16, 1, 1, -1, -1>;
using b8_idx_gl = gl<int, 1, 1, 1, -1>;

template <bool F32>
struct b8_globals {
    b8_x_gl x;
    b8_ascale_gl a_scale;
    b8_w_gl b0, b1, b2;
    b8_scale_gl s0, s1, s2;
    b8_out_gl<F32> c0, c1, c2;
    b8_res_gl r0, r1, r2;
    b8_idx_gl topk_idx;
    // The first tile of members 1 and 2 (the tile count for an absent one), the K steps, the
    // activation row of pair `p` (`p / row_div`), and the residual epilogue's switch.
    int start1, start2, k_steps, row_div, add;
    // The last step's columns read past `k` inside its last 128-byte atom (`tail_from` =
    // `tail_to` when there are none).
    int tail_from, tail_to;
};

// The K tail's columns `[from, to)` of a stage's two tiles zeroed by the `threads` threads of a
// group, thread `tid` of it: every row (rows past the arrays are TMA's zero fill already).
__device__ inline void b8_zero_tail(b8_w_tile& w, b8_x_tile& x, int from, int to, int tid,
                                    int threads) {
    const fp8e4m3 zero = base_types::convertor<fp8e4m3, float>::convert(0.0f);
    const int width = to - from;
    constexpr int ROWS = b8_w_tile::rows + b8_x_tile::rows;
    for (int i = tid; i < ROWS * width; i += threads) {
        const int r = i / width, c = from + i % width;
        if (r < b8_w_tile::rows) {
            w[{r, c}] = zero;
        } else {
            x[{r - b8_w_tile::rows, c}] = zero;
        }
    }
}

// The K tail of a `k`-wide row at B8's 256-wide steps: `[from, to)` of the last step's columns, empty
// when `k` is a multiple of 128 (a whole atom past `k` is TMA's zero fill).
inline void b8_tail(uint32_t k, int& from, int& to) {
    const int rem = int(k - (k - 1) / uint32_t(tc_b8::k_step) * uint32_t(tc_b8::k_step));
    from = rem;
    to = (rem + 127) / 128 * 128;
}

template <bool EXPERT, bool F32>
__global__ __launch_bounds__(NUM_THREADS, tc_b8::min_blocks) void gemv_fp8_b8_ker(
    const __grid_constant__ b8_globals<F32> g) {
    using acc32 = rt_fl<16, 32>;
    constexpr int STAGES = tc_b8::stages;
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);
    b8_w_tile(&w_s)[STAGES] = al.allocate<b8_w_tile, STAGES>();
    b8_x_tile(&x_s)[STAGES] = al.allocate<b8_x_tile, STAGES>();
    __shared__ semaphore arrived[STAGES], done[STAGES];
    const int t = blockIdx.x, p = blockIdx.y;
    const int m = t >= g.start2 ? 2 : t >= g.start1 ? 1 : 0;
    const int local = t - (m == 2 ? g.start2 : m == 1 ? g.start1 : 0);
    const int xrow = p / g.row_div;
    int depth = 0;
    if constexpr (EXPERT) {
        depth = g.topk_idx[{0, 0, 0, p}];
    }
    if (threadIdx.x == 0) {
        for (int s = 0; s < STAGES; s++) {
            init_semaphore(arrived[s], 0, 1);
            init_semaphore(done[s], 1, 0);
        }
    }
    __syncthreads();
    if (warpgroup::groupid() == 1) {
        // The producer: warp 0 of the second warpgroup.
        if (warpgroup::warpid() == 0) {
            const b8_w_gl& b = m == 2 ? g.b2 : m == 1 ? g.b1 : g.b0;
            for (int j = 0; j < g.k_steps; j++) {
                const int s = j % STAGES;
                if (j >= STAGES) {
                    wait(done[s], ((j - STAGES) / STAGES) % 2);
                }
                warp::tma::expect_bytes(arrived[s], sizeof(b8_x_tile) + sizeof(b8_w_tile));
                warp::tma::load_async(x_s[s], g.x, {0, xrow, 0, j}, arrived[s]);
                warp::tma::load_async(w_s[s], b, {0, depth, local, j}, arrived[s]);
            }
        }
    } else {
        acc32 acc;
        warp::zero(acc);
        for (int j = 0; j < g.k_steps; j++) {
            const int s = j % STAGES;
            wait(arrived[s], (j / STAGES) % 2);
            if (j == g.k_steps - 1 && g.tail_from < g.tail_to) {
                // The K tail read past `k`: zeroed by the warpgroup before its WGMMAs read it;
                // each writer's async-proxy fence (`mma_fence`) precedes the barrier.
                b8_zero_tail(w_s[s], x_s[s], g.tail_from, g.tail_to, warpgroup::laneid(),
                             kittens::WARPGROUP_THREADS);
                warpgroup::mma_fence(acc);
                warpgroup::sync(1);
            }
            warpgroup::mma_ABt(acc, w_s[s], x_s[s]);
            warpgroup::mma_async_wait();
            if (warpgroup::laneid() == 0) {
                arrive(done[s], 1);
            }
        }
        // Warp `w` holds weight rows `16w .. 16w + 16` of the tile; column 0 is theirs.
        const int w = warpgroup::warpid();
        const int row = local * tc_b8::tile + 16 * w;
        typename acc32::col_vec v, side;
        warp::row_sum(v, acc);
        warp::load(side, m == 2 ? g.s2 : m == 1 ? g.s1 : g.s0, {0, 0, depth, row / 16});
        warp::mul(v, v, side);
        warp::mul(v, v, g.a_scale[{0, 0, 0, xrow}]);
        if (g.add) {
            warp::unary_op<scratchy::ops::round_bf16>(v, v);
            warp::load(side, m == 2 ? g.r2 : m == 1 ? g.r1 : g.r0, {0, 0, p, row / 16});
            warp::add(v, v, side);
        }
        warp::store(m == 2 ? g.c2 : m == 1 ? g.c1 : g.c0, v, {0, 0, p, row / 16});
    }
}

// The swizzled-box probe (test support, `tests/gemv_fp8_b8.rs`, TK-1's R1 on B8's 128-byte-swizzled
// fp8 boxes): each box first TMA-loaded with all-NaN `poison`, then B8's own box at K step `j`,
// activation row `row` (the `[rows, 1, k]` view) and weight tile `tile`, then both TMA-stored to
// exact-fit dumps. Every byte past `k`, past the activation's row 0 or past the weight's rows must
// come back 0, not NaN.
using b8_dump_x_gl = gl<fp8e4m3, 1, 1, -1, -1, b8_x_tile>;
using b8_dump_w_gl = gl<fp8e4m3, 1, 1, -1, -1, b8_w_tile>;

struct b8_probe_globals {
    b8_x_gl x;
    b8_w_gl w;
    b8_dump_x_gl poison_x, dump_x;
    b8_dump_w_gl poison_w, dump_w;
    int row, tile, j;
    // B8's K tail zeroing at step `j` (`tail_from` = `tail_to` but at the last step of a `k` with a
    // partial last atom).
    int tail_from, tail_to;
};

__global__ void gemv_fp8_b8_box_probe_ker(const __grid_constant__ b8_probe_globals g) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);
    b8_x_tile& xs = al.allocate<b8_x_tile>();
    b8_w_tile& ws = al.allocate<b8_w_tile>();
    __shared__ semaphore poisoned, loaded;
    if (warp::laneid() == 0) {
        init_semaphore(poisoned, 0, 1);
        init_semaphore(loaded, 0, 1);
    }
    __syncwarp();
    warp::tma::expect_bytes(poisoned, sizeof(b8_x_tile) + sizeof(b8_w_tile));
    warp::tma::load_async(xs, g.poison_x, {0, 0, 0, 0}, poisoned);
    warp::tma::load_async(ws, g.poison_w, {0, 0, 0, 0}, poisoned);
    wait(poisoned, 0);
    warp::tma::expect_bytes(loaded, sizeof(b8_x_tile) + sizeof(b8_w_tile));
    warp::tma::load_async(xs, g.x, {0, g.row, 0, g.j}, loaded);
    warp::tma::load_async(ws, g.w, {0, 0, g.tile, g.j}, loaded);
    wait(loaded, 0);
    // The boxes as B8 reads them: its K tail zeroing, by this one warp.
    if (g.tail_from < g.tail_to) {
        b8_zero_tail(ws, xs, g.tail_from, g.tail_to, warp::laneid(), kittens::WARP_THREADS);
        __syncwarp();
    }
    warp::tma::store_async(g.dump_x, xs, {0, 0, 0, 0});
    warp::tma::store_async(g.dump_w, ws, {0, 0, 0, 0});
    warp::tma::store_async_wait();
}

constexpr int B8_PROBE_SMEM = int(sizeof(b8_x_tile) + sizeof(b8_w_tile)) + 1024;

// B8's production instances.
template <bool EXPERT, bool F32>
cudaError_t allow_b8() {
    return cudaFuncSetAttribute(gemv_fp8_b8_ker<EXPERT, F32>,
                                cudaFuncAttributeMaxDynamicSharedMemorySize, int(tc_b8::smem));
}

// A B8 launch's operands, as the launcher was given them.
struct b8_args {
    const void* q = nullptr;
    const void* a_scale = nullptr;
    uint32_t a_rows = 0;
    const void* const* b = nullptr;
    const void* const* b_scale = nullptr;
    void* const* c = nullptr;
    const void* const* residual = nullptr;
    const uint32_t* n = nullptr;
    uint32_t count = 0;
    uint32_t k = 0;
    uint32_t experts = 0;
    uint32_t expert_rows = 0;
    const void* topk_idx = nullptr;
    uint32_t pairs = 0;
    uint32_t row_div = 0;
};

// The views and counts of `a`'s launch, then the launch on `stream`. Can throw (TK's host code).
template <bool EXPERT, bool F32>
int launch_b8(const b8_args& a, void* stream) {
    using out_t = std::conditional_t<F32, float, bf16>;
    const uint32_t last = a.count - 1;
    const auto at = [&](uint32_t i) { return i < a.count ? i : last; };
    const auto rows = [&](uint32_t i) { return EXPERT ? a.expert_rows : a.n[at(i)]; };
    const auto wv = [&](uint32_t i) {
        return b8_w_gl{static_cast<fp8e4m3*>(const_cast<void*>(a.b[at(i)])), nullptr,
                       size_t(a.experts), size_t(rows(i)), size_t(a.k)};
    };
    const auto sv = [&](uint32_t i) {
        return b8_scale_gl{static_cast<float*>(const_cast<void*>(a.b_scale[at(i)])), nullptr,
                           nullptr, size_t(a.experts), size_t(rows(i))};
    };
    const auto cv = [&](uint32_t i) {
        return b8_out_gl<F32>{static_cast<out_t*>(a.c[at(i)]), nullptr, nullptr,
                              size_t(a.pairs), size_t(a.n[at(i)])};
    };
    // A member with no residual views its own output (never read: `add` is 0 for the launch).
    const auto rv = [&](uint32_t i) {
        const void* r = a.residual != nullptr ? a.residual[at(i)] : a.c[at(i)];
        return b8_res_gl{static_cast<bf16*>(const_cast<void*>(r)), nullptr, nullptr,
                         size_t(a.pairs), size_t(a.n[at(i)])};
    };
    int tiles[3] = {0, 0, 0};
    for (uint32_t i = 0; i < a.count; i++) {
        tiles[i] = tiles_of(a.n[i], tc_b8::tile);
    }
    const int all = tiles[0] + tiles[1] + tiles[2];
    // With no experts the view is never read; `a_scale` is a valid address for it.
    const void* idx = EXPERT ? a.topk_idx : a.a_scale;
    b8_globals<F32> G{
        b8_x_gl{static_cast<fp8e4m3*>(const_cast<void*>(a.q)), nullptr, size_t(a.a_rows), nullptr,
                size_t(a.k)},
        b8_ascale_gl{static_cast<float*>(const_cast<void*>(a.a_scale)), nullptr, nullptr, nullptr,
                     size_t(a.a_rows)},
        wv(0), wv(1), wv(2),
        sv(0), sv(1), sv(2),
        cv(0), cv(1), cv(2),
        rv(0), rv(1), rv(2),
        b8_idx_gl{static_cast<int*>(const_cast<void*>(idx)), nullptr, nullptr, nullptr,
                  size_t(EXPERT ? a.pairs : 1)},
        a.count > 1 ? tiles[0] : all,
        a.count > 2 ? tiles[0] + tiles[1] : all,
        int((a.k + uint32_t(tc_b8::k_step) - 1) / uint32_t(tc_b8::k_step)),
        int(a.row_div),
        a.residual != nullptr ? 1 : 0,
        0,
        0};
    b8_tail(a.k, G.tail_from, G.tail_to);
    void* args[] = {&G};
    const dim3 grid{uint32_t(all), a.pairs}, block{uint32_t(NUM_THREADS)};
    return static_cast<int>(cudaLaunchKernel(gemv_fp8_b8_ker<EXPERT, F32>, grid, block, args,
                                             tc_b8::smem, static_cast<cudaStream_t>(stream)));
}

// Whether `a` is refused (`scratchy_cuda_gemv_fp8_b8`'s list).
bool b8_refused(const b8_args& a, bool f32) {
    constexpr uint64_t LIMIT = uint64_t(INT_MAX);
    const bool expert = a.topk_idx != nullptr;
    if (a.count < 1 || a.count > 3 || a.k == 0 || a.k % N_MULTIPLE != 0 || a.pairs == 0 ||
        a.pairs > 65535 || a.row_div == 0 || a.a_rows == 0 ||
        uint64_t(a.pairs - 1) / a.row_div >= a.a_rows || a.experts == 0 ||
        (!expert && a.experts != 1) || (f32 && a.residual != nullptr) || a.q == nullptr ||
        a.a_scale == nullptr || uint64_t(a.a_rows) * a.k > LIMIT) {
        return true;
    }
    for (uint32_t i = 0; i < a.count; i++) {
        const uint64_t rows = expert ? a.expert_rows : a.n[i];
        if (a.n[i] == 0 || a.n[i] % N_MULTIPLE != 0 || rows < a.n[i] ||
            uint64_t(a.experts) * rows * a.k > LIMIT || uint64_t(a.pairs) * a.n[i] > LIMIT ||
            a.b[i] == nullptr || a.b_scale[i] == nullptr || a.c[i] == nullptr ||
            (a.residual != nullptr && a.residual[i] == nullptr)) {
            return true;
        }
    }
    return false;
}

}  // namespace

// The kernel's one-time setup: allow each B8 instance its dynamic shared memory on the current
// device. 0 = done; any other value = the first failing cudaError_t.
extern "C" int scratchy_cuda_gemv_fp8_tc_prepare(void) {
    cudaError_t e = cudaSuccess;
    for (const cudaError_t r : {allow_b8<false, false>(), allow_b8<false, true>(),
                                allow_b8<true, false>(), allow_b8<true, true>()}) {
        if (e == cudaSuccess) {
            e = r;
        }
    }
    return static_cast<int>(e);
}

// B8 (the tapes' tensor-core decode GEMVs). 0 = launched; SCRATCHY_CUDA_ERR_SHAPE = refused,
// nothing launched; SCRATCHY_CUDA_ERR_HOST = a C++ exception was caught; any other
// value = the launch's cudaError_t. One launch, asynchronous on `stream`.
//
// `count` (1 to 3) members off the e4m3 activation `q [a_rows, k]` and its per-row `a_scale
// [a_rows]` f32 (`quant_rows_fp8`'s bytes): pair `p` (`0 .. pairs`) of member `i` is
//
//     c[i][p, j] = (((q[p / row_div] · b_e[j, :]) · b_scale_e[j]) · a_scale[p / row_div])
//
// over the member's `n[i]` rows, rounded once to bf16 (`out_f32` 0), with `residual[i][p, j]`
// added as `scratchy::ops::with_residual` does when `residual` is not null; or stored as f32
// (`out_f32` 1, no residual). Dense (`topk_idx` null): `b[i]` is `[n[i], k]` and `b_scale[i]
// [n[i]]`, `experts` 1, `expert_rows` unread, `e` 0. Expert: `b[i]` is the bank's `[experts,
// expert_rows, k]` and `b_scale[i] [experts, expert_rows]`, `e = topk_idx[p]`, each member's first
// `n[i]` rows of its expert (`gemv_fp8.cu`'s `EXPERT` contract).
//
// Refused before any pointer is touched: `count` not 1..3; `k` or an `n[i]` zero or not a multiple
// of 64; `pairs` 0 or above 65,535; `row_div` 0; a pair's row at or past `a_rows`; `experts` 0,
// or not 1 for the dense form; `expert_rows` below an `n[i]` (expert); a residual with `out_f32`;
// a size above INT_MAX; a null pointer. Every pointer 16-byte-aligned device memory, no output
// overlapping an input. `scratchy_cuda_gemv_fp8_tc_prepare` must have succeeded.
extern "C" int scratchy_cuda_gemv_fp8_b8(
    const void* q, const void* a_scale, uint32_t a_rows, const void* const* b,
    const void* const* b_scale, void* const* c, const void* const* residual, const uint32_t* n,
    uint32_t count, uint32_t k, uint32_t experts, uint32_t expert_rows, const void* topk_idx,
    uint32_t pairs, uint32_t row_div, uint32_t out_f32,
    void* stream /* cudaStream_t; null = default */) {
    const b8_args a{q,     a_scale, a_rows, b,           b_scale,  c,     residual, n,
                    count, k,       experts, expert_rows, topk_idx, pairs, row_div};
    if (b8_refused(a, out_f32 != 0)) {
        return SCRATCHY_CUDA_ERR_SHAPE;
    }
    // ThunderKittens' host code can throw; an exception must never unwind into Rust.
    try {
        const bool expert = topk_idx != nullptr;
        if (out_f32 != 0) {
            return expert ? launch_b8<true, true>(a, stream) : launch_b8<false, true>(a, stream);
        }
        return expert ? launch_b8<true, false>(a, stream) : launch_b8<false, false>(a, stream);
    } catch (...) {
        return SCRATCHY_CUDA_ERR_HOST;
    }
}

// Test support (`tests/gemv_fp8_b8.rs`): `gemv_fp8_b8_box_probe_ker` on the activation `x [rows, 1,
// k]` at row `row` and the weight `w [n, k]` at tile `tile`, K step `j`; `poison_x [32, 256]`,
// `poison_w [64, 256]` all-NaN e4m3, `dump_x`, `dump_w` the same shapes. Refused: a zero or
// misaligned size (`k` a multiple of 16), a coordinate past its array, a null pointer.
extern "C" int scratchy_cuda_gemv_fp8_b8_box_probe(const void* x, uint32_t rows, const void* w,
                                                   uint32_t n, uint32_t k, uint32_t row,
                                                   uint32_t tile, uint32_t j, const void* poison_x,
                                                   const void* poison_w, void* dump_x,
                                                   void* dump_w,
                                                   void* stream /* cudaStream_t; null = default */) {
    if (rows == 0 || n == 0 || k == 0 || k % 16 != 0 || row >= rows || tile * 64 >= n ||
        uint64_t(j) * 256 >= k || x == nullptr || w == nullptr || poison_x == nullptr ||
        poison_w == nullptr || dump_x == nullptr || dump_w == nullptr ||
        uint64_t(rows) * k > uint64_t(INT_MAX) || uint64_t(n) * k > uint64_t(INT_MAX)) {
        return SCRATCHY_CUDA_ERR_SHAPE;
    }
    // ThunderKittens' host code can throw; an exception must never unwind into Rust.
    try {
        const auto e4 = [](const void* p) { return static_cast<fp8e4m3*>(const_cast<void*>(p)); };
        b8_probe_globals G{
            b8_x_gl{e4(x), nullptr, size_t(rows), nullptr, size_t(k)},
            b8_w_gl{e4(w), nullptr, size_t(1), size_t(n), size_t(k)},
            b8_dump_x_gl{e4(poison_x), nullptr, nullptr, size_t(32), size_t(256)},
            b8_dump_x_gl{e4(dump_x), nullptr, nullptr, size_t(32), size_t(256)},
            b8_dump_w_gl{e4(poison_w), nullptr, nullptr, size_t(64), size_t(256)},
            b8_dump_w_gl{e4(dump_w), nullptr, nullptr, size_t(64), size_t(256)},
            int(row), int(tile), int(j), 0, 0};
        if (uint64_t(j + 1) * 256 >= k) {
            b8_tail(k, G.tail_from, G.tail_to);
        }
        void* args[] = {&G};
        const dim3 grid{1u}, block{uint32_t(kittens::WARP_THREADS)};
        return static_cast<int>(cudaLaunchKernel(gemv_fp8_b8_box_probe_ker, grid, block, args,
                                                 B8_PROBE_SMEM, static_cast<cudaStream_t>(stream)));
    } catch (...) {
        return SCRATCHY_CUDA_ERR_HOST;
    }
}
