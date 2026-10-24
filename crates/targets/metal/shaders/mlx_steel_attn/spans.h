// SPDX-License-Identifier: Apache-2.0

#pragma once

// Block-diagonal span attention, shared by the paged prefill kernels (simdgroup steel, its wide
// twin, and both NAX kernels).
//
// The host's per-token label buffer (`span_ids_per_token`) holds, at each absolute token position,
// its Relocatable span's first token + 1, or 0 for the shared prefix and the query, which attend
// every key. A query inside a span attends only the keys of its own span: those from the span's
// first token up to itself. So a span's K/V is a function of its own tokens alone, which is what
// lets a span cached in one prompt be reused in another (`BlockKind::Relocatable`).
//
// The kernels mask that bound per row, inside every K-tile that starts below it. Their tile
// seek (start the kb-loop at a span-uniform Q tile's first span tile) only skips whole tiles, so
// alone it isolates a span only when the span is aligned to the kernel's Q and K tiles.

#include <metal_stdlib>

using namespace metal;

namespace mlx {
namespace steel {

// The first key the query at absolute position `pos` attends: its span's first token, or 0
// outside a span.
METAL_FUNC int span_first_key(const device uint* span_ids, int pos) {
  const uint label = span_ids[uint(pos)];
  return label != 0u ? int(label) - 1 : 0;
}

} // namespace steel
} // namespace mlx
