// SPDX-License-Identifier: Apache-2.0
//
// A target's capture of a layer's output rows for its DFlash2 drafter (~/git/splash @ 1482f6bb
// runtime/model/QwenTarget.cpp `addCapture`, `ops::RowCopy`): each row of the layer's output into
// block CAPTURE_SLOT of the captured rows, CAPTURE_SLOTS blocks of CAPTURE_HIDDEN values a row.
//
// Baked constants:
//   CAPTURE_HIDDEN — the hidden width (a block's)
//   CAPTURE_SLOT   — the layer's block
//   CAPTURE_SLOTS  — the blocks of a captured row
//   CAPTURE_N      — the bucket's elements (rows · hidden)
//
// Dispatch: one thread per element, scaled with the step's rows.

#include <metal_stdlib>
#include "baked.h"

using namespace metal;

SCRATCHY_CONSTANT(uint, CAPTURE_HIDDEN, 0);
SCRATCHY_CONSTANT(uint, CAPTURE_SLOT, 1);
SCRATCHY_CONSTANT(uint, CAPTURE_SLOTS, 2);
SCRATCHY_CONSTANT(uint, CAPTURE_N, 3);

template <typename T>
[[kernel]] void capture_rows(device const T *input [[buffer(0)]],
                             device T *captured [[buffer(1)]],
                             uint gid [[thread_position_in_grid]]) {
  if (gid >= CAPTURE_N)
    return;
  const uint row = gid / CAPTURE_HIDDEN, column = gid % CAPTURE_HIDDEN;
  captured[(ulong(row) * CAPTURE_SLOTS + CAPTURE_SLOT) * CAPTURE_HIDDEN + column] = input[gid];
}

SCRATCHY_KERNEL(capture_rows_f16, capture_rows<half>)
SCRATCHY_KERNEL(capture_rows_bf16, capture_rows<bfloat>)
