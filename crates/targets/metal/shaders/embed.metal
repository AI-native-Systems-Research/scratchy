// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

#include <metal_stdlib>
#include "baked.h"
using namespace metal;

// ============================================================================
// Embed: Embedding lookup operation
// out[i, :] = table[indices[i], :]
// ============================================================================

/// `EmbedConstants`, compiled in:
///   0 = M (bucket_m, = num_tokens for this bucket)
///   1 = HIDDEN_SIZE (the embedding row width)
SCRATCHY_CONSTANT(uint, EMBED_M, 0);
SCRATCHY_CONSTANT(uint, EMBED_HIDDEN_SIZE, 1);

/// Pure gather, no reductions, no casts.
template <typename T>
[[kernel]] void embed_specialized(
    device       T* out          [[buffer(0)]],   // [num_tokens, hidden_size]
    device const T* table        [[buffer(1)]],   // [vocab_size, hidden_size]
    device const uint* indices   [[buffer(2)]],   // [num_tokens]
    uint tid [[thread_position_in_grid]]
) {
    // Dispatch is `(ceil(M/threads_per_group), 1, 1)` × `(threads_per_group, 1, 1)`,
    // so the trailing partial group's threads have `tid >= M` and must
    // short-circuit before touching `indices` / `table`. Without this,
    // out-of-bounds reads cause a GPU command-buffer hang.
    if (tid >= EMBED_M) return;
    uint idx = indices[tid];
    device const T* src = table + idx * EMBED_HIDDEN_SIZE;
    device       T* dst = out   + tid * EMBED_HIDDEN_SIZE;
    for (uint i = 0; i < EMBED_HIDDEN_SIZE; i++) {
        dst[i] = src[i];
    }
}

SCRATCHY_KERNEL(embed_f16_specialized, embed_specialized<half>)
SCRATCHY_KERNEL(embed_bf16_specialized, embed_specialized<bfloat>)
