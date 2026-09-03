// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Embed: Embedding lookup operation
// out[i, :] = table[indices[i], :]
// ============================================================================

kernel void embed_f16(
    device const half* table [[buffer(0)]],      // [vocab_size, hidden_size]
    device const int* indices [[buffer(1)]],     // [num_tokens]
    device half* out [[buffer(2)]],              // [num_tokens, hidden_size]
    constant uint& hidden_size [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    // Each thread processes one token (copies one row)
    int idx = indices[tid];
    device const half* src = table + idx * hidden_size;
    device half* dst = out + tid * hidden_size;
    
    // Copy the entire row
    for (uint i = 0; i < hidden_size; i++) {
        dst[i] = src[i];
    }
}

kernel void embed_bf16(
    device const bfloat* table [[buffer(0)]],    // [vocab_size, hidden_size]
    device const int* indices [[buffer(1)]],     // [num_tokens]
    device bfloat* out [[buffer(2)]],            // [num_tokens, hidden_size]
    constant uint& hidden_size [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    // Each thread processes one token (copies one row)
    int idx = indices[tid];
    device const bfloat* src = table + idx * hidden_size;
    device bfloat* dst = out + tid * hidden_size;

    // Copy the entire row
    for (uint i = 0; i < hidden_size; i++) {
        dst[i] = src[i];
    }
}

/// Phase 5.B specialized variant: bucket_m + hidden_size baked in via
/// `[[function_constant(N)]]`. Index assignment must match
/// `scratchy-target-metal::interpreter::metal::pipelines::constants_for(Embed)`:
///   0 = M (bucket_m, = num_tokens for this bucket)
///   1 = HIDDEN_SIZE (= W::Q_SIZE)
constant uint EMBED_M           [[function_constant(0)]];
constant uint EMBED_HIDDEN_SIZE [[function_constant(1)]];

kernel void embed_f16_specialized(
    device       half* out     [[buffer(0)]],   // [num_tokens, hidden_size]
    device const half* table   [[buffer(1)]],   // [vocab_size, hidden_size]
    device const uint* indices [[buffer(2)]],   // [num_tokens]
    uint tid [[thread_position_in_grid]]
) {
    // Dispatch is `(ceil(M/threads_per_group), 1, 1)` × `(threads_per_group, 1, 1)`,
    // so the trailing partial group's threads have `tid >= M` and must
    // short-circuit before touching `indices` / `table`. Without this,
    // out-of-bounds reads cause a GPU command-buffer hang.
    if (tid >= EMBED_M) return;
    uint idx = indices[tid];
    device const half* src = table + idx * EMBED_HIDDEN_SIZE;
    device       half* dst = out   + tid * EMBED_HIDDEN_SIZE;
    for (uint i = 0; i < EMBED_HIDDEN_SIZE; i++) {
        dst[i] = src[i];
    }
}

/// BF16 specialized variant — pure gather, no reductions, no casts;
/// the only difference from the f16 path is binding type.
kernel void embed_bf16_specialized(
    device       bfloat* out     [[buffer(0)]],   // [num_tokens, hidden_size]
    device const bfloat* table   [[buffer(1)]],   // [vocab_size, hidden_size]
    device const uint*   indices [[buffer(2)]],   // [num_tokens]
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= EMBED_M) return;
    uint idx = indices[tid];
    device const bfloat* src = table + idx * EMBED_HIDDEN_SIZE;
    device       bfloat* dst = out   + tid * EMBED_HIDDEN_SIZE;
    for (uint i = 0; i < EMBED_HIDDEN_SIZE; i++) {
        dst[i] = src[i];
    }
}
