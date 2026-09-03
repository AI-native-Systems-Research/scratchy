// Stub — MLX's full complex.h pulls in host-side `bfloat16_t` which
// the Metal compiler doesn't have. We never instantiate qmm_t over
// complex inputs but mma.h has a `BlockMMA<complex64_t, ...>`
// partial specialization that needs the type to exist with a
// 2-arg constructor (used inside the unreachable code path).
#pragma once
#include <metal_stdlib>
using namespace metal;
struct complex64_t {
  float real = 0.0f;
  float imag = 0.0f;
  complex64_t() thread = default;
  complex64_t(float r, float i) thread : real(r), imag(i) {}
};
