#include <metal_stdlib>
#include "baked.h"
using namespace metal;

// `GeluConstants`, compiled in: the elements the buffer holds (the dispatch
// rounds up to whole threadgroups; a thread past them writes nothing).
SCRATCHY_CONSTANT(uint, ACTIVATION_N, 0);

// ============================================================================
// GELU, tanh approximation, computed in `A` (half for f16, float otherwise):
// GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// ============================================================================

template <typename T, typename A>
[[kernel]] void gelu_tanh(
    device T* output [[buffer(0)]],
    device const T* input [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= ACTIVATION_N) return;

    A x = A(input[gid]);
    const A BETA = A(0.7978845608f);  // sqrt(2/pi)
    const A KAPPA = A(0.044715f);
    A x_cube = x * x * x;
    // Clamp the tanh argument: Metal's relaxed-math `tanh` evaluates via
    // `exp(2*inner)`, which overflows to Inf (→ NaN) for large `inner`.
    // `tanh` is already saturated to ±1 well before ±15, so this is
    // bit-exact in f16/bf16/f32 while killing the overflow. (Qwen3.5-VL's
    // ViT MLP drives fc1 activations to ~17 → inner ~189 — the first
    // kernel to hit it; text MLPs stay well within range, so it's a no-op
    // there.)
    A inner = clamp(BETA * (x + KAPPA * x_cube), A(-15.0f), A(15.0f));
    output[gid] = T(A(0.5f) * x * (A(1.0f) + tanh(inner)));
}

SCRATCHY_KERNEL(gelu_tanh_f16, gelu_tanh<half, half>)
SCRATCHY_KERNEL(gelu_tanh_bf16, gelu_tanh<bfloat, float>)

// ============================================================================
// GELU Erf (exact): 0.5 * x * (1 + erf(x / sqrt(2)))
//
// MSL has no built-in erf; this is the faithfully-rounded rational
// approximation MLX ships (mlx/backend/metal/kernels/erf.h, itself the
// well-known Norbert Juffa implementation; max error < 1 ulp). One
// deviation: MLX's upper branch ends `-expm1f(r)`; we use
// `1.0f - exp(r)` — there r ≤ −0.86 so exp(r) ≤ 0.42 and the
// subtraction has no cancellation; the sub-ulp f32 difference vanishes
// entirely in bf16/f16 outputs.
//
// Used by the `gelu_erf` DSL op (LocateAnything projector; the MoonViT
// block MLP uses `gelu` = tanh — the two flavors are numerically
// distinct and MUST NOT be conflated).
// ============================================================================

inline float scratchy_erf(float a) {
    float r, s, t, u;
    t = metal::abs(a);
    s = a * a;
    if (t > 0.927734375f) {
        // maximum error 0.99527 ulp
        r = metal::fma(-1.72853470e-5f, t, 3.83197126e-4f);
        u = metal::fma(-3.88396438e-3f, t, 2.42546219e-2f);
        r = metal::fma(r, s, u);
        r = metal::fma(r, t, -1.06777877e-1f);
        r = metal::fma(r, t, -6.34846687e-1f);
        r = metal::fma(r, t, -1.28717512e-1f);
        r = metal::fma(r, t, -t);
        r = 1.0f - metal::exp(r);
        r = metal::copysign(r, a);
    } else {
        // maximum error 0.98929 ulp
        r = -5.96761703e-4f;
        r = metal::fma(r, s, 4.99119423e-3f);
        r = metal::fma(r, s, -2.67681349e-2f);
        r = metal::fma(r, s, 1.12819925e-1f);
        r = metal::fma(r, s, -3.76125336e-1f);
        r = metal::fma(r, s, 1.28379166e-1f);
        r = metal::fma(r, a, a);
    }
    return r;
}

template <typename T>
[[kernel]] void gelu_erf(
    device T* output [[buffer(0)]],
    device const T* input [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= ACTIVATION_N) return;

    float x = float(input[gid]);
    constexpr float INV_SQRT2 = 0.70710678118654752440f;
    output[gid] = T(0.5f * x * (1.0f + scratchy_erf(x * INV_SQRT2)));
}

SCRATCHY_KERNEL(gelu_erf_f16, gelu_erf<half>)
SCRATCHY_KERNEL(gelu_erf_bf16, gelu_erf<bfloat>)

// ============================================================================
// QuickGELU: x * sigmoid(1.702 * x) — the CLIP/Qwen2-VL block-MLP
// activation (mlx_vlm models/qwen2_vl/vision.py nn.quick_gelu). A
// sigmoid approximation of GELU, numerically DISTINCT from both the
// tanh and erf flavors above — Qwen2-VL uses quick_gelu in tower
// blocks and gelu_erf in the patch merger, so all three coexist.
// ============================================================================

template <typename T>
[[kernel]] void quick_gelu(
    device T* output [[buffer(0)]],
    device const T* input [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= ACTIVATION_N) return;

    float x = float(input[gid]);
    output[gid] = T(x / (1.0f + metal::exp(-1.702f * x)));
}

SCRATCHY_KERNEL(quick_gelu_f16, quick_gelu<half>)
SCRATCHY_KERNEL(quick_gelu_bf16, quick_gelu<bfloat>)
