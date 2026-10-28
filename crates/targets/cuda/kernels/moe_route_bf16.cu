// SPDX-License-Identifier: Apache-2.0
//
// The qwen3-5-moe router's routing for the H100 (MoE step M1 of #857): per token row `t` of the
// router logits `[m, 256]` bf16, `p = softmax(logits[t])` in f32, its `top_k` largest picked, and
// their probabilities (divided by their sum when `renorm`) stored as f32 `scores [m, top_k]`, their
// expert indices as i32 `topk_idx [m, top_k]`. The picks are stored ascending: slot `top_k − 1` is
// the best, as Metal writes `inds[K-1-r]` (`metal/shaders/moe_route.h`).
//
// ThunderKittens' warp vector (PLAN-tk100 §2 S4, §3.M M1): one warp per token row, 8 warps per CTA;
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
//   - NaN: TK's `max` returns the non-NaN operand, so a NaN probability is keyed −inf, the lowest.
//     With every logit NaN, every key is −inf and every round still picks the highest untaken
//     index: `top_k` distinct indices in [0, 256), and NaN scores.
//   - Renorm: the picked probabilities, summed in slot order, then each divided (scalar arithmetic
//     on reduction results). Lane 0 stores both rows through `gl::operator[]`.
//
// Gemma 4's program (`GEMMA`, `RouteProgram::GemmaScaledSoftmax`, Metal's `moe_route.h` score program
// and `moe_per_expert_scale.metal`), at 128 experts (Gemma 4 26B-A4B) or 256, 4 or 8 logits per
// lane:
//   - The picks are the same top-k rounds over `nan_to_neg_inf` of the RAW logits, not a softmax:
//     the inputs are exact bf16, so nothing is rounded before the picks. Ties by the higher index.
//   - `s = logits · scale` (`warp::mul`, `scale` = `hidden^-0.5`), every unpicked entry set to −inf
//     (`apply` reading `taken`); `warp::max`, `warp::sub` of it BEFORE `warp::exp`, so the best pick
//     is exactly 1; every unpicked entry then set to exactly 0 (`apply`, never relying on
//     `exp(−inf)`); `warp::sum` and `warp::div`: the softmax over the picks.
//   - `s · per_expert_scale` (`warp::load` of the `[experts]` bf16 weight, widened exactly, then
//     `warp::mul` vector by vector), each pick's score the one element at its index (`apply`, then
//     `warp::sum`, as the Qwen program's).
//   - NaN: all-NaN logits pick `top_k` distinct indices (the highest), and every score is NaN.
// The template is instantiated for the Qwen program at 256 experts (its body unchanged) and for
// Gemma's at 128 and 256 (`op_abi::MOE_EXPERT_COUNTS`).
//
// No atomics, a fixed order: the same row gives the same bits wherever it sits.

#include "kittens.cuh"
#include "scratchy_ops.cuh"

#include <climits>
#include <cstdint>
#include <type_traits>

using namespace kittens;

// Launcher return codes that are not a `cudaError_t`. `src/kernels.rs` mirrors these two values.
#define SCRATCHY_CUDA_ERR_SHAPE (-1)
#define SCRATCHY_CUDA_ERR_HOST (-2)

namespace {

// `step::MOE_ROUTE_EXPERTS`: the Qwen program's one count, one naive `rv_fl<256>` per row.
constexpr int QWEN_EXPERTS = 256;
// `op_abi::MOE_EXPERT_COUNTS`: the Gemma program's counts.
constexpr int GEMMA_EXPERTS_SMALL = 128;
constexpr int GEMMA_EXPERTS_LARGE = 256;
// The launcher's `program`.
constexpr uint32_t QWEN = 0;
constexpr uint32_t GEMMA = 1;
// `step::MOE_ROUTE_MAX_TOP_K`.
constexpr uint32_t MAX_TOP_K = 32;
constexpr int WARPS = 8;
constexpr int NUM_THREADS = WARPS * kittens::WARP_THREADS;

using idx_gl = gl<int, 1, 1, -1, -1>;
using score_gl = gl<float, 1, 1, -1, -1>;

// The Qwen program's arguments, as before Gemma's program was added.
struct route_globals {
    gl<bf16, 1, 1, -1, QWEN_EXPERTS> logits;
    idx_gl topk_idx;
    score_gl scores;
    int m;
    int top_k;
    bool renorm;
};

// The Gemma program's: no `renorm`, and its scale and per-expert scale.
template <int EXPERTS>
struct gemma_route_globals {
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

template <int EXPERTS, bool GEMMA_PROGRAM>
using globals_of = std::conditional_t<GEMMA_PROGRAM, gemma_route_globals<EXPERTS>, route_globals>;

template <int EXPERTS, bool GEMMA_PROGRAM>
__global__ __launch_bounds__(NUM_THREADS) void moe_route_bf16_ker(
    const __grid_constant__ globals_of<EXPERTS, GEMMA_PROGRAM> g) {
    static_assert(GEMMA_PROGRAM || EXPERTS == QWEN_EXPERTS, "the Qwen program is at 256 only");
    using expert_vec = rv_fl<EXPERTS>;
    const int t = blockIdx.x * WARPS + kittens::warpid();
    if (t >= g.m) {
        return;
    }
    constexpr float NEG_INF = kittens::base_types::constants<float>::neg_infty();
    // `p`: Qwen's probabilities, or Gemma's raw logits.
    expert_vec p, key, taken, probe;
    warp::load(p, g.logits, {0, 0, t, 0});
    if constexpr (!GEMMA_PROGRAM) {
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
        if constexpr (!GEMMA_PROGRAM) {
            // The pick's probability: the one element of `p` at `idx`, every other lane's 0.
            warp::apply(probe, p, [&](int i, float x) { return i == idx ? x : 0.0f; });
            prob[j] = warp::sum(probe);
        }
    }
    if constexpr (GEMMA_PROGRAM) {
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
    if constexpr (!GEMMA_PROGRAM) {
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
template <int EXPERTS, bool GEMMA_PROGRAM>
int launch(const globals_of<EXPERTS, GEMMA_PROGRAM>& G, uint32_t m, cudaStream_t stream) {
    void* args[] = {const_cast<globals_of<EXPERTS, GEMMA_PROGRAM>*>(&G)};
    const dim3 grid((m + uint32_t(WARPS) - 1) / uint32_t(WARPS)), block(NUM_THREADS);
    return static_cast<int>(cudaLaunchKernel(moe_route_bf16_ker<EXPERTS, GEMMA_PROGRAM>, grid,
                                             block, args, 0, stream));
}

}  // namespace

// 0 = launched; SCRATCHY_CUDA_ERR_SHAPE = shape refused, nothing launched;
// SCRATCHY_CUDA_ERR_HOST = a C++ exception was caught; any other value = the cudaError_t
// cudaLaunchKernel returned for THIS launch. Nothing here reads or clears the runtime's last-error
// state.
//
// `m` must be non-zero and at most INT_MAX, `top_k` in 1..=32 (`step::MOE_ROUTE_MAX_TOP_K`), and
// `program` 0 (Qwen) with `experts` 256 (`step::MOE_ROUTE_EXPERTS`), or 1 (Gemma) with `experts`
// 128 or 256 (`op_abi::MOE_EXPERT_COUNTS`) and a non-null `expert_scale`; all are checked before
// any pointer is touched. `logits` is `[m, experts]` bf16, `expert_scale` (Gemma's; null for Qwen)
// `[experts]` bf16, `topk_idx` `[m, top_k]` i32 and `scores` `[m, top_k]` f32, the two outputs
// overlapping nothing. `renorm` is the Qwen program's; `scale` (`hidden^-0.5`) the Gemma
// program's. No shared memory, so no one-time setup. The launch is asynchronous on `stream`.
extern "C" int scratchy_cuda_moe_route_bf16(const void* logits, const void* expert_scale,
                                            void* topk_idx, void* scores, uint32_t m,
                                            uint32_t experts, uint32_t top_k, uint32_t renorm,
                                            uint32_t program, float scale,
                                            void* stream /* cudaStream_t; null = default */) {
    const bool qwen = program == QWEN && experts == uint32_t(QWEN_EXPERTS);
    const bool gemma = program == GEMMA && expert_scale != nullptr &&
                       (experts == uint32_t(GEMMA_EXPERTS_SMALL) ||
                        experts == uint32_t(GEMMA_EXPERTS_LARGE));
    if (m == 0 || m > uint32_t(INT_MAX) || !(qwen || gemma) || top_k == 0 || top_k > MAX_TOP_K) {
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
        if (qwen) {
            route_globals G{{lp, nullptr, nullptr, size_t(m), nullptr}, ig, sg, int(m),
                            int(top_k), renorm != 0};
            return launch<QWEN_EXPERTS, false>(G, m, s);
        }
        if (experts == uint32_t(GEMMA_EXPERTS_SMALL)) {
            gemma_route_globals<GEMMA_EXPERTS_SMALL> G{
                {lp, nullptr, nullptr, size_t(m), nullptr}, ig, sg,
                {ep, nullptr, nullptr, nullptr, nullptr}, int(m), int(top_k), scale};
            return launch<GEMMA_EXPERTS_SMALL, true>(G, m, s);
        }
        gemma_route_globals<GEMMA_EXPERTS_LARGE> G{
            {lp, nullptr, nullptr, size_t(m), nullptr}, ig, sg,
            {ep, nullptr, nullptr, nullptr, nullptr}, int(m), int(top_k), scale};
        return launch<GEMMA_EXPERTS_LARGE, true>(G, m, s);
    } catch (...) {
        return SCRATCHY_CUDA_ERR_HOST;
    }
}
