#include <metal_stdlib>
using namespace metal;

// ============================================================================
// SiLU (Swish) Activation: x * sigmoid(x)
// ============================================================================

kernel void silu_f16(
    device half* output [[buffer(0)]],
    device const half* input [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    
    half x = input[gid];
    // SiLU: x * sigmoid(x) = x / (1 + exp(-x))
    output[gid] = x / (1.0h + exp(-x));
}

kernel void silu_bf16(
    device bfloat* output [[buffer(0)]],
    device const bfloat* input [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    
    float x = float(input[gid]);
    // SiLU: x * sigmoid(x) = x / (1 + exp(-x))
    output[gid] = bfloat(x / (1.0f + exp(-x)));
}

kernel void silu_f32(
    device float* output [[buffer(0)]],
    device const float* input [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    
    float x = input[gid];
    // SiLU: x * sigmoid(x) = x / (1 + exp(-x))
    output[gid] = x / (1.0f + exp(-x));
}

// Vectorized SiLU (4 elements at a time)
kernel void silu_vec4_f16(
    device half4* output [[buffer(0)]],
    device const half4* input [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    
    half4 x = input[gid];
    // SiLU: x * sigmoid(x) = x / (1 + exp(-x))
    output[gid] = x / (1.0h + exp(-x));
}

// ============================================================================
// GELU Activation (Tanh Approximation)
// Note: Metal doesn't have erf(), so we use tanh approximation
// GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// ============================================================================

kernel void gelu_f16(
    device half* output [[buffer(0)]],
    device const half* input [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    
    half x = input[gid];
    // GELU tanh approximation
    constexpr half BETA = 0.7978845608h;  // sqrt(2/pi)
    constexpr half KAPPA = 0.044715h;
    half x_cube = x * x * x;
    half inner = clamp(BETA * (x + KAPPA * x_cube), -15.0h, 15.0h);
    output[gid] = 0.5h * x * (1.0h + tanh(inner));
}

kernel void gelu_bf16(
    device bfloat* output [[buffer(0)]],
    device const bfloat* input [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    
    float x = float(input[gid]);
    // GELU tanh approximation
    constexpr float BETA = 0.7978845608f;  // sqrt(2/pi)
    constexpr float KAPPA = 0.044715f;
    float x_cube = x * x * x;
    // Clamp the tanh argument: Metal's relaxed-math `tanh` evaluates via
    // `exp(2*inner)`, which overflows to Inf (→ NaN) for large `inner`.
    // `tanh` is already saturated to ±1 well before ±15, so this is
    // bit-exact in f16/bf16/f32 while killing the overflow. (Qwen3.5-VL's
    // ViT MLP drives fc1 activations to ~17 → inner ~189 — the first
    // kernel to hit it; text MLPs stay well within range, so it's a no-op
    // there.)
    float inner = clamp(BETA * (x + KAPPA * x_cube), -15.0f, 15.0f);
    output[gid] = bfloat(0.5f * x * (1.0f + tanh(inner)));
}

kernel void gelu_f32(
    device float* output [[buffer(0)]],
    device const float* input [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    
    float x = input[gid];
    // GELU tanh approximation
    constexpr float BETA = 0.7978845608f;  // sqrt(2/pi)
    constexpr float KAPPA = 0.044715f;
    float x_cube = x * x * x;
    // Clamp the tanh argument: Metal's relaxed-math `tanh` evaluates via
    // `exp(2*inner)`, which overflows to Inf (→ NaN) for large `inner`.
    // `tanh` is already saturated to ±1 well before ±15, so this is
    // bit-exact in f16/bf16/f32 while killing the overflow. (Qwen3.5-VL's
    // ViT MLP drives fc1 activations to ~17 → inner ~189 — the first
    // kernel to hit it; text MLPs stay well within range, so it's a no-op
    // there.)
    float inner = clamp(BETA * (x + KAPPA * x_cube), -15.0f, 15.0f);
    output[gid] = 0.5f * x * (1.0f + tanh(inner));
}

// ============================================================================
// GELU Tanh (explicit name for compatibility)
// ============================================================================

kernel void gelu_tanh_f16(
    device half* output [[buffer(0)]],
    device const half* input [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    
    half x = input[gid];
    constexpr half BETA = 0.7978845608h;
    constexpr half KAPPA = 0.044715h;
    half x_cube = x * x * x;
    half inner = clamp(BETA * (x + KAPPA * x_cube), -15.0h, 15.0h);
    output[gid] = 0.5h * x * (1.0h + tanh(inner));
}

kernel void gelu_tanh_bf16(
    device bfloat* output [[buffer(0)]],
    device const bfloat* input [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    
    float x = float(input[gid]);
    constexpr float BETA = 0.7978845608f;
    constexpr float KAPPA = 0.044715f;
    float x_cube = x * x * x;
    // Clamp the tanh argument: Metal's relaxed-math `tanh` evaluates via
    // `exp(2*inner)`, which overflows to Inf (→ NaN) for large `inner`.
    // `tanh` is already saturated to ±1 well before ±15, so this is
    // bit-exact in f16/bf16/f32 while killing the overflow. (Qwen3.5-VL's
    // ViT MLP drives fc1 activations to ~17 → inner ~189 — the first
    // kernel to hit it; text MLPs stay well within range, so it's a no-op
    // there.)
    float inner = clamp(BETA * (x + KAPPA * x_cube), -15.0f, 15.0f);
    output[gid] = bfloat(0.5f * x * (1.0f + tanh(inner)));
}

kernel void gelu_tanh_f32(
    device float* output [[buffer(0)]],
    device const float* input [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    
    float x = input[gid];
    constexpr float BETA = 0.7978845608f;
    constexpr float KAPPA = 0.044715f;
    float x_cube = x * x * x;
    // Clamp the tanh argument: Metal's relaxed-math `tanh` evaluates via
    // `exp(2*inner)`, which overflows to Inf (→ NaN) for large `inner`.
    // `tanh` is already saturated to ±1 well before ±15, so this is
    // bit-exact in f16/bf16/f32 while killing the overflow. (Qwen3.5-VL's
    // ViT MLP drives fc1 activations to ~17 → inner ~189 — the first
    // kernel to hit it; text MLPs stay well within range, so it's a no-op
    // there.)
    float inner = clamp(BETA * (x + KAPPA * x_cube), -15.0f, 15.0f);
    output[gid] = 0.5f * x * (1.0f + tanh(inner));
}

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

kernel void gelu_erf_f16(
    device half* output [[buffer(0)]],
    device const half* input [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;

    float x = float(input[gid]);
    constexpr float INV_SQRT2 = 0.70710678118654752440f;
    output[gid] = half(0.5f * x * (1.0f + scratchy_erf(x * INV_SQRT2)));
}

kernel void gelu_erf_bf16(
    device bfloat* output [[buffer(0)]],
    device const bfloat* input [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;

    float x = float(input[gid]);
    constexpr float INV_SQRT2 = 0.70710678118654752440f;
    output[gid] = bfloat(0.5f * x * (1.0f + scratchy_erf(x * INV_SQRT2)));
}

kernel void gelu_erf_f32(
    device float* output [[buffer(0)]],
    device const float* input [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;

    float x = input[gid];
    constexpr float INV_SQRT2 = 0.70710678118654752440f;
    output[gid] = 0.5f * x * (1.0f + scratchy_erf(x * INV_SQRT2));
}

// ============================================================================
// GELU Quick Approximation: x * sigmoid(1.702 * x)
// ============================================================================

kernel void gelu_quick_f16(
    device half* output [[buffer(0)]],
    device const half* input [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    
    half x = input[gid];
    // GELU quick: x * sigmoid(1.702 * x)
    constexpr half ALPHA = 1.702h;
    output[gid] = x / (1.0h + exp(-ALPHA * x));
}

kernel void gelu_quick_bf16(
    device bfloat* output [[buffer(0)]],
    device const bfloat* input [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    
    float x = float(input[gid]);
    // GELU quick: x * sigmoid(1.702 * x)
    constexpr float ALPHA = 1.702f;
    output[gid] = bfloat(x / (1.0f + exp(-ALPHA * x)));
}

kernel void gelu_quick_f32(
    device float* output [[buffer(0)]],
    device const float* input [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    
    float x = input[gid];
    // GELU quick: x * sigmoid(1.702 * x)
    constexpr float ALPHA = 1.702f;
    output[gid] = x / (1.0f + exp(-ALPHA * x));
}

// ============================================================================
// FatReLU: max(0, x) with threshold
// ============================================================================

kernel void fatrelu_f16(
    device half* output [[buffer(0)]],
    device const half* input [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    constant float& threshold [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    
    half x = input[gid];
    half t = half(threshold);
    output[gid] = (x > t) ? x : 0.0h;
}

kernel void fatrelu_bf16(
    device bfloat* output [[buffer(0)]],
    device const bfloat* input [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    constant float& threshold [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    
    float x = float(input[gid]);
    output[gid] = bfloat((x > threshold) ? x : 0.0f);
}

kernel void fatrelu_f32(
    device float* output [[buffer(0)]],
    device const float* input [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    constant float& threshold [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    
    float x = input[gid];
    output[gid] = (x > threshold) ? x : 0.0f;
}
// ============================================================================
// QuickGELU: x * sigmoid(1.702 * x) — the CLIP/Qwen2-VL block-MLP
// activation (mlx_vlm models/qwen2_vl/vision.py nn.quick_gelu). A
// sigmoid approximation of GELU, numerically DISTINCT from both the
// tanh and erf flavors above — Qwen2-VL uses quick_gelu in tower
// blocks and gelu_erf in the patch merger, so all three coexist.
// ============================================================================

kernel void quick_gelu_f16(
    device half* output [[buffer(0)]],
    device const half* input [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;

    float x = float(input[gid]);
    output[gid] = half(x / (1.0f + metal::exp(-1.702f * x)));
}

kernel void quick_gelu_bf16(
    device bfloat* output [[buffer(0)]],
    device const bfloat* input [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;

    float x = float(input[gid]);
    output[gid] = bfloat(x / (1.0f + metal::exp(-1.702f * x)));
}

kernel void quick_gelu_f32(
    device float* output [[buffer(0)]],
    device const float* input [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;

    float x = input[gid];
    output[gid] = x / (1.0f + metal::exp(-1.702f * x));
}
