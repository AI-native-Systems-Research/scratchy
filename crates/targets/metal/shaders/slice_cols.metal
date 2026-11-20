// SPDX-License-Identifier: Apache-2.0
//
// Each row's column window: `out[t, c] = in[t, OFFSET + c]` for `c < COLS`
// (`MetalStep::SliceCols`; MLA's split of its compressed projection into the
// latent and the shared rope key).
//
// Constants (compiled in; `SliceColsConstants`):
//   0 = IN_COLS   the input row's width
//   1 = OFFSET    the window's first column
//   2 = COLS      the window's width
//   3 = ELEMENTS  the bucket's output elements (rows * COLS)
//
// Bindings: buffer(0) = in [rows, IN_COLS], buffer(1) = out [rows, COLS].
// Dispatch: one thread per output element.

#include <metal_stdlib>
#include "baked.h"

using namespace metal;

SCRATCHY_CONSTANT(uint, SC_IN_COLS, 0);
SCRATCHY_CONSTANT(uint, SC_OFFSET, 1);
SCRATCHY_CONSTANT(uint, SC_COLS, 2);
SCRATCHY_CONSTANT(uint, SC_ELEMENTS, 3);

template <typename T>
[[kernel]] void slice_cols(
    device const T* in  [[buffer(0)]],
    device       T* out [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= SC_ELEMENTS) {
        return;
    }
    const uint t = gid / SC_COLS;
    const uint c = gid % SC_COLS;
    out[gid] = in[t * SC_IN_COLS + SC_OFFSET + c];
}

SCRATCHY_KERNEL(slice_cols_f16, slice_cols<half>)
SCRATCHY_KERNEL(slice_cols_bf16, slice_cols<bfloat>)
