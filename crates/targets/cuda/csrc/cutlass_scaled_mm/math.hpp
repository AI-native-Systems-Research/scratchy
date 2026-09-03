// SPDX-License-Identifier: Apache-2.0
// Math utilities for CUTLASS kernel dispatch.
#pragma once

#include <climits>

inline constexpr uint32_t next_pow_2(uint32_t const num) {
  if (num <= 1) return num;
  return 1 << (CHAR_BIT * sizeof(num) - __builtin_clz(num - 1));
}
