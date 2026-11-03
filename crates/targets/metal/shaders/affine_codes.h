// SPDX-License-Identifier: Apache-2.0
//
// How an MLX-affine 4-bit weight's packed codes are laid out in memory
// (`scratchy_target_metal::tape::kernel_constants::AffineCodes`), the half every kernel reading
// them shares with the allocator that stores them.
//
// As written (MLX's checkpoint layout): row `n`'s codes are one run of `K / 2` bytes, the input
// elements in order — byte `(n * K + k) / 2`.
//
// Planes (`AFFINE_CODES_PLANES`, constant slot 24): the same bytes regrouped a 64-input chunk
// at a time — chunk `c`'s 32 bytes of every row, row after row, then chunk `c + 1`'s — byte
// `(c * N + n) * 32 + (k % 64) / 2` for `c = k / 64`. A GEMM tile's step over one chunk then reads
// one contiguous span of its columns' codes instead of a 32-byte piece every `K / 2` bytes. The
// byte values are unchanged (XOR 0x88 on top under `AFFINE_CODES_OFFSET8`), so the scales and
// biases, row-major `[N][K / G]`, are untouched.

#pragma once

#include <metal_stdlib>

// The bytes of a 64-input chunk of one row.
constant constexpr int AFFINE_PLANE_ROW_BYTES = 32;

// The address of the codes of input `k` (a multiple of 2) of row `n`, of `N` rows of `K` inputs,
// stored as planes. A shader includes this header once for every kernel the bake batches into one
// compile (`#pragma once`, hoisted above every kernel's constants), so the header holds no baked
// constant: each shader declares `AFFINE_CODES_PLANES` itself (`AFFINE_CODES_PLANES_DECL`) and
// picks between this and its own row-major address.
inline const device uchar* affine_planes_at(
    const device uchar* w, int64_t n, int64_t k, int64_t N) {
  return w + ((k / 64) * N + n) * AFFINE_PLANE_ROW_BYTES + (k % 64) / 2;
}

// The declaration every shader reading planes codes makes, after `baked.h`: slot 24.
#define AFFINE_CODES_PLANES_DECL SCRATCHY_CONSTANT_OPTIONAL(bool, AFFINE_CODES_PLANES, 24)

// The address of the codes of input `k` of row `n`: planes under `AFFINE_CODES_PLANES`, else row
// major. A macro, so it reads the including kernel's own constant.
#define affine_codes_at(w, n, k, N, K)                                                   \
  (AFFINE_CODES_PLANES ? affine_planes_at((w), (n), (k), (N))                           \
                       : (w) + (int64_t(n) * int64_t(K) + int64_t(k)) / 2)
