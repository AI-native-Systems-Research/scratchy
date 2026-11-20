// SPDX-License-Identifier: Apache-2.0
//
// The MoE router's routing for the H100 (MoE step M1 of #857). The softmax-top-k program
// (`SOFTMAX_TOPK`, `RouteProgram::SoftmaxTopK`, qwen3-5-moe's): per token row `t` of the router
// logits `[m, 256]` bf16, `p = softmax(logits[t])` in f32, its `top_k` largest picked, and
// their probabilities (divided by their sum when `renorm`) stored as f32 `scores [m, top_k]`, their
// expert indices as i32 `topk_idx [m, top_k]`. The picks are stored ascending: slot `top_k − 1` is
// the best, as Metal writes `inds[K-1-r]` (`metal/shaders/moe_route.h`).
//
// ThunderKittens' warp vector: one warp per token row, 8 warps per CTA;
// a warp at or past `m` returns before reading anything. The row is one naive `rv_fl<256>`
// (`warp::load`, bf16 to f32, exact), 8 logits per lane.
//   - Softmax: `warp::max`, then `warp::sub` of the max BEFORE `warp::exp`, so the max element is
//     exactly `exp(0) = 1` under `--use_fast_math` (the #812 lesson); `warp::sum` and `warp::div`.
//   - Top-k (TK has no argmax or top-k; this is the gap, filled with TK reductions): the keys are
//     `unary_op<scratchy::ops::nan_to_neg_inf>` of `p`, and `taken` a vector of 0s. Each of
//     `top_k` rounds takes `v = warp::max` of the untaken keys (`apply`: taken ones as −inf), then
//     `idx = warp::max` of the untaken indices whose key equals `v` (`apply`: `float(i)`, else −1),
//     and marks `taken[idx]`. Every `apply` reads only the lane's own elements of `key` and
//     `taken`, by the index TK passes, so no lane reads another's.
//   - Tie rule: equal probabilities rank by the HIGHER expert index (the round takes the max
//     index), #841's rule and Metal's packed key. Indices up to 255 are exact in f32.
//   - Precision (#1283), both programs: the router logits are bf16, as HF transformers' Gemma 4
//     router; vLLM computes them in f32 and breaks exact ties on the lower expert index, so a
//     near-tie at the k-th pick can route a token differently. Measured on Gemma 4: no effect on
//     the output.
//   - NaN: TK's `max` returns the non-NaN operand, so a NaN probability is keyed −inf, the lowest.
//     With every logit NaN, every key is −inf and every round still picks the highest untaken
//     index: `top_k` distinct indices in [0, 256), and NaN scores.
//   - Renorm: the picked probabilities, summed in slot order, then each divided (scalar arithmetic
//     on reduction results). Lane 0 stores both rows through `gl::operator[]`.
//
// The scaled program (`TOPK_SCALED_SOFTMAX`, `RouteProgram::TopKScaledSoftmax`, Gemma 4's; Metal's
// `moe_route.h` score program and `moe_per_expert_scale.metal`), at 128 experts (Gemma 4 26B-A4B)
// or 256, 4 or 8 logits per lane:
//   - The picks are the same top-k rounds over `nan_to_neg_inf` of the RAW logits, not a softmax:
//     the inputs are exact bf16, so nothing is rounded before the picks. Ties by the higher index.
//   - `s = logits · scale` (`warp::mul`, `scale` = `hidden^-0.5`), every unpicked entry set to −inf
//     (`apply` reading `taken`); `warp::max`, `warp::sub` of it BEFORE `warp::exp`, so the best pick
//     is exactly 1; every unpicked entry then set to exactly 0 (`apply`, never relying on
//     `exp(−inf)`); `warp::sum` and `warp::div`: the softmax over the picks.
//   - `s · per_expert_scale` (`warp::load` of the `[experts]` bf16 weight, widened exactly, then
//     `warp::mul` vector by vector), each pick's score the one element at its index (`apply`, then
//     `warp::sum`, as the softmax-top-k program's).
//   - NaN: all-NaN logits pick `top_k` distinct indices (the highest), and every score is NaN.
// The template is instantiated for the softmax-top-k program at 256 experts (its body unchanged)
// and for the scaled one at 128 and 256 (`op_abi::MOE_EXPERT_COUNTS`).
//
// No atomics, a fixed order: the same row gives the same bits wherever it sits.

#include "kittens.cuh"
#include "scratchy_ops.cuh"

#include <climits>
#include <cstdint>
#include <type_traits>

using namespace kittens;

// The launcher's shape refusal, a return code no `cudaError_t` is (those are non-negative);
// `src/kernels.rs` mirrors it. Its host-exception code is the shared `scratchy::CUDA_ERR_HOST`.
#define SCRATCHY_CUDA_ERR_SHAPE (-1)

namespace {

// `step::MOE_ROUTE_EXPERTS`: the softmax-top-k program's one count, one naive `rv_fl<256>` per
// row.
constexpr int SOFTMAX_TOPK_EXPERTS = scratchy::MOE_ROUTE_EXPERTS;
// `op_abi::MOE_EXPERT_COUNTS`: the scaled program's counts.
constexpr int SCALED_EXPERTS_SMALL = scratchy::MOE_EXPERT_COUNTS[0];
constexpr int SCALED_EXPERTS_LARGE = scratchy::MOE_EXPERT_COUNTS[1];
// The launcher's `program` (`kernels.rs`'s `ROUTE_SOFTMAX_TOPK`, `ROUTE_TOPK_SCALED_SOFTMAX`).
constexpr uint32_t SOFTMAX_TOPK = scratchy::ROUTE_SOFTMAX_TOPK;
constexpr uint32_t TOPK_SCALED_SOFTMAX = scratchy::ROUTE_TOPK_SCALED_SOFTMAX;
// `step::MOE_ROUTE_MAX_TOP_K`.
constexpr uint32_t MAX_TOP_K = scratchy::MOE_ROUTE_MAX_TOP_K;
constexpr int WARPS = 8;
constexpr int NUM_THREADS = WARPS * kittens::WARP_THREADS;

using idx_gl = gl<int, 1, 1, -1, -1>;
using score_gl = gl<float, 1, 1, -1, -1>;

// The softmax-top-k program's arguments.
struct softmax_topk_globals {
    gl<bf16, 1, 1, -1, SOFTMAX_TOPK_EXPERTS> logits;
    idx_gl topk_idx;
    score_gl scores;
    int m;
    int top_k;
    bool renorm;
};

// The scaled program's: no `renorm`, and its scale and per-expert scale.
template <int EXPERTS>
struct scaled_route_globals {
    gl<bf16, 1, 1, -1, EXPERTS> logits;
    idx_gl topk_idx;
    score_gl scores;
    // `[experts]` bf16, `router.per_expert_scale`.
    gl<bf16, 1, 1, 1, EXPERTS> expert_scale;
    int m;
    int top_k;
    // `hidden^-0.5`.
    float scale;
};

template <int EXPERTS, bool SCALED>
using globals_of = std::conditional_t<SCALED, scaled_route_globals<EXPERTS>, softmax_topk_globals>;

template <int EXPERTS, bool SCALED>
__global__ __launch_bounds__(NUM_THREADS) void moe_route_bf16_ker(
    const __grid_constant__ globals_of<EXPERTS, SCALED> g) {
    static_assert(SCALED || EXPERTS == SOFTMAX_TOPK_EXPERTS,
                  "the softmax-top-k program is at 256 only");
    using expert_vec = rv_fl<EXPERTS>;
    const int t = blockIdx.x * WARPS + kittens::warpid();
    if (t >= g.m) {
        return;
    }
    constexpr float NEG_INF = kittens::base_types::constants<float>::neg_infty();
    // `p`: the softmax-top-k program's probabilities, or the scaled program's raw logits.
    expert_vec p, key, taken, probe;
    warp::load(p, g.logits, {0, 0, t, 0});
    if constexpr (!SCALED) {
        const float mx = warp::max(p);
        warp::sub(p, p, mx);
        warp::exp(p, p);
        const float total = warp::sum(p);
        warp::div(p, p, total);
    }
    warp::unary_op<scratchy::ops::nan_to_neg_inf>(key, p);
    warp::zero(taken);

    int picked[MAX_TOP_K];
    float prob[MAX_TOP_K];
#pragma unroll
    for (int j = 0; j < int(MAX_TOP_K); ++j) {
        if (j >= g.top_k) {
            break;
        }
        warp::apply(probe, key, [&](int i, float k) {
            return taken[i / kittens::WARP_THREADS][0] != 0.0f ? NEG_INF : k;
        });
        const float v = warp::max(probe);
        warp::apply(probe, key, [&](int i, float k) {
            return (taken[i / kittens::WARP_THREADS][0] == 0.0f && k == v) ? float(i) : -1.0f;
        });
        const int idx = int(warp::max(probe));
        warp::apply(taken, taken, [&](int i, float was) { return i == idx ? 1.0f : was; });
        picked[j] = idx;
        if constexpr (!SCALED) {
            // The pick's probability: the one element of `p` at `idx`, every other lane's 0.
            warp::apply(probe, p, [&](int i, float x) { return i == idx ? x : 0.0f; });
            prob[j] = warp::sum(probe);
        }
    }
    if constexpr (SCALED) {
        // The scaled logits of the picks, a softmax over them, then the per-expert scale.
        expert_vec s, pes;
        warp::mul(s, p, g.scale);
        warp::apply(s, s, [&](int i, float x) {
            return taken[i / kittens::WARP_THREADS][0] != 0.0f ? x : NEG_INF;
        });
        const float mx = warp::max(s);
        warp::sub(s, s, mx);
        warp::exp(s, s);
        warp::apply(s, s, [&](int i, float x) {
            return taken[i / kittens::WARP_THREADS][0] != 0.0f ? x : 0.0f;
        });
        const float total = warp::sum(s);
        warp::div(s, s, total);
        warp::load(pes, g.expert_scale, {0, 0, 0, 0});
        warp::mul(s, s, pes);
#pragma unroll
        for (int j = 0; j < int(MAX_TOP_K); ++j) {
            if (j >= g.top_k) {
                break;
            }
            const int idx = picked[j];
            // The pick's score: the one element of `s` at `idx`, every other lane's 0.
            warp::apply(probe, s, [&](int i, float x) { return i == idx ? x : 0.0f; });
            prob[j] = warp::sum(probe);
        }
    }

    float sum = 0.0f;
    bool renorm = false;
    if constexpr (!SCALED) {
        renorm = g.renorm;
    }
    if (renorm) {
#pragma unroll
        for (int j = 0; j < int(MAX_TOP_K); ++j) {
            if (j < g.top_k) {
                sum += prob[j];
            }
        }
    }
    if (warp::laneid() == 0) {
#pragma unroll
        for (int j = 0; j < int(MAX_TOP_K); ++j) {
            if (j < g.top_k) {
                // Round j is the j-th best: slot `top_k − 1 − j`, so the slots ascend.
                const int slot = g.top_k - 1 - j;
                g.topk_idx[{0, 0, t, slot}] = picked[j];
                g.scores[{0, 0, t, slot}] = renorm ? prob[j] / sum : prob[j];
            }
        }
    }
}

// Launch the program with its globals `G` over `m` rows.
template <int EXPERTS, bool SCALED>
int launch(const globals_of<EXPERTS, SCALED>& G, uint32_t m, cudaStream_t stream) {
    void* args[] = {const_cast<globals_of<EXPERTS, SCALED>*>(&G)};
    const dim3 grid((m + uint32_t(WARPS) - 1) / uint32_t(WARPS)), block(NUM_THREADS);
    return static_cast<int>(cudaLaunchKernel(moe_route_bf16_ker<EXPERTS, SCALED>, grid,
                                             block, args, 0, stream));
}

}  // namespace

// The kernel's one-time setup, run once per device by `kernels::prepare_all` (`CudaDevice::open`):
// query its attributes (`cudaFuncGetAttributes`), which loads its module, so a worker's graph
// capture (`CudaWorker::new`) never loads a kernel mid-capture. It changes nothing on the device.
//
// 0 = done; any other value = the cudaError_t of the first query that failed.
extern "C" int scratchy_cuda_moe_route_bf16_prepare(void) {
    cudaFuncAttributes attr;
    if (const cudaError_t e = cudaFuncGetAttributes(&attr, moe_route_bf16_ker<SOFTMAX_TOPK_EXPERTS, false>); e != cudaSuccess) {
        return static_cast<int>(e);
    }
    if (const cudaError_t e = cudaFuncGetAttributes(&attr, moe_route_bf16_ker<SCALED_EXPERTS_SMALL, true>); e != cudaSuccess) {
        return static_cast<int>(e);
    }
    if (const cudaError_t e = cudaFuncGetAttributes(&attr, moe_route_bf16_ker<SCALED_EXPERTS_LARGE, true>); e != cudaSuccess) {
        return static_cast<int>(e);
    }
    return 0;
}

// 0 = launched; SCRATCHY_CUDA_ERR_SHAPE = shape refused, nothing launched;
// scratchy::CUDA_ERR_HOST = a C++ exception was caught; any other value = the cudaError_t
// cudaLaunchKernel returned for THIS launch. Nothing here reads or clears the runtime's last-error
// state.
//
// `m` must be non-zero and at most INT_MAX, `top_k` in 1..=32 (`step::MOE_ROUTE_MAX_TOP_K`), and
// `program` 0 (softmax-top-k) with `experts` 256 (`step::MOE_ROUTE_EXPERTS`), or 1 (scaled) with
// `experts` 128 or 256 (`op_abi::MOE_EXPERT_COUNTS`) and a non-null `expert_scale`; all are checked
// before any pointer is touched. `logits` is `[m, experts]` bf16, `expert_scale` (the scaled
// program's; null for the other) `[experts]` bf16, `topk_idx` `[m, top_k]` i32 and `scores`
// `[m, top_k]` f32, the two outputs overlapping nothing. `renorm` is the softmax-top-k program's;
// `scale` (`hidden^-0.5`) the scaled program's.
// No shared memory, so its one-time setup (`_prepare`, in `PREPARE`) only loads it. The launch is
// asynchronous on `stream`.
extern "C" int scratchy_cuda_moe_route_bf16(const void* logits, const void* expert_scale,
                                            void* topk_idx, void* scores, uint32_t m,
                                            uint32_t experts, uint32_t top_k, uint32_t renorm,
                                            uint32_t program, float scale,
                                            void* stream /* cudaStream_t; null = default */) {
    const bool softmax_topk = program == SOFTMAX_TOPK && experts == uint32_t(SOFTMAX_TOPK_EXPERTS);
    const bool scaled = program == TOPK_SCALED_SOFTMAX && expert_scale != nullptr &&
                        (experts == uint32_t(SCALED_EXPERTS_SMALL) ||
                         experts == uint32_t(SCALED_EXPERTS_LARGE));
    if (m == 0 || m > uint32_t(INT_MAX) || !(softmax_topk || scaled) || top_k == 0 ||
        top_k > MAX_TOP_K) {
        return SCRATCHY_CUDA_ERR_SHAPE;
    }
    // ThunderKittens' host code can throw; an exception must never unwind into Rust.
    try {
        const cudaStream_t s = static_cast<cudaStream_t>(stream);
        bf16* lp = static_cast<bf16*>(const_cast<void*>(logits));
        bf16* ep = static_cast<bf16*>(const_cast<void*>(expert_scale));
        // No view carries a TMA type, so building them encodes nothing.
        idx_gl ig{static_cast<int*>(topk_idx), nullptr, nullptr, size_t(m), size_t(top_k)};
        score_gl sg{static_cast<float*>(scores), nullptr, nullptr, size_t(m), size_t(top_k)};
        if (softmax_topk) {
            softmax_topk_globals G{{lp, nullptr, nullptr, size_t(m), nullptr}, ig, sg, int(m),
                            int(top_k), renorm != 0};
            return launch<SOFTMAX_TOPK_EXPERTS, false>(G, m, s);
        }
        if (experts == uint32_t(SCALED_EXPERTS_SMALL)) {
            scaled_route_globals<SCALED_EXPERTS_SMALL> G{
                {lp, nullptr, nullptr, size_t(m), nullptr}, ig, sg,
                {ep, nullptr, nullptr, nullptr, nullptr}, int(m), int(top_k), scale};
            return launch<SCALED_EXPERTS_SMALL, true>(G, m, s);
        }
        scaled_route_globals<SCALED_EXPERTS_LARGE> G{
            {lp, nullptr, nullptr, size_t(m), nullptr}, ig, sg,
            {ep, nullptr, nullptr, nullptr, nullptr}, int(m), int(top_k), scale};
        return launch<SCALED_EXPERTS_LARGE, true>(G, m, s);
    } catch (...) {
        return scratchy::CUDA_ERR_HOST;
    }
}
