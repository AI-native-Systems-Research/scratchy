// SPDX-License-Identifier: Apache-2.0
//
// select_rows — a speculative step's inputs that depend on how many drafts the step before it kept,
// picked on the device: the host lays the step out while that one still runs, so it writes every
// outcome and the device copies the one that happened, at the head of the step's command buffer.
//
// One threadgroup per op: `len` words of its source from word `*sel * stride` — variants `stride`
// words apart, overlapping where `stride < len` — to `dst`. The source is an earlier step's output
// (`src`), or a table in the step's staged region, `src_at` bytes in (`src` null); `src`, `dst` and
// `sel` are addresses of buffers the step's residency holds.
//
// Bindings: 0 ops [n_ops], 1 the staged region.
// Dispatch: n_ops threadgroups of 64 threads.

#include <metal_stdlib>
#include "baked.h"
using namespace metal;

struct SelectOp {
  uint src_at;
  uint stride;
  uint len;
  uint pad;
  device const uint* src;
  device uint* dst;
  device const uint* sel;
};

kernel void select_rows(
    device const SelectOp* ops [[buffer(0)]],
    device const uchar* region [[buffer(1)]],
    uint op  [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg  [[threads_per_threadgroup]])
{
  const SelectOp o = ops[op];
  device const uint* base = o.src ? o.src : (device const uint*)(region + o.src_at);
  device const uint* from = base + ulong(*o.sel) * o.stride;
  for (uint i = tid; i < o.len; i += tg) {
    o.dst[i] = from[i];
  }
}
