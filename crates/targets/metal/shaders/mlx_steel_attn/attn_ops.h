// SPDX-License-Identifier: Apache-2.0
//
// The steel attentions' element ops: the row reductions and the bin ops their online softmax
// applies to MMA tiles (`MMATile::row_reduce` / `row_bin_op`). Shared by the paged attentions
// (`steel_attention_paged_kernel.h` and the kernels that include it) and the latent attention's
// tiled prefill (`mla.metal`).

#pragma once

#include <metal_stdlib>

struct MaxOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return metal::max(x, y);
  }
};

struct SumOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return x + y;
  }
};

struct MulOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return x * y;
  }
};

struct SubOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return x - y;
  }
};

struct ExpSubOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return metal::fast::exp2(x - y);
  }
};

struct DivOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return x / y;
  }
};
