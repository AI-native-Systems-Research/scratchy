// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

#include <metal_stdlib>
#include "megakernel/mk_common.h"
using namespace metal;

#ifndef MK_BODIES_ONLY
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
#endif // MK_BODIES_ONLY
#define EMBED_CONSTS(X) X(uint, m, EMBED_M, 0) X(uint, hidden, EMBED_HIDDEN_SIZE, 1)

// Megakernel adapter: the rows the dispatch threads of one virtual threadgroup gather (token
// `tg_pos.x * tpg.x + t` for thread `t`), each row copied by all of its threads — a copy, so the
// split writes the same bytes.
template <typename T, typename C>
MK_FUNC void mk_embed(thread const MkStep& s, MkLane l, threadgroup uchar*) {
  if (!l.live) return;
  mk_ptr<T> out = (mk_ptr<T>)s.addr[0];
  const device T* table = (const device T*)s.addr[1];
  const device uint* indices = (const device uint*)s.addr[2];
  const uint first = l.tg_pos.x * l.tpg.x;
  const uint last = min(first + l.tpg.x, C::m());
  for (uint tok = first; tok < last; ++tok) {
    const uint idx = indices[tok];
    for (uint i = l.tid; i < C::hidden(); i += l.tpg.x) {
      out[tok * C::hidden() + i] = table[idx * C::hidden() + i];
    }
  }
}

#ifndef MK_BODIES_ONLY
EMBED_CONSTS(MK_FC_DECLARE)

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
#else
MK_TAIL(embed_f16_specialized, (mk_embed<half, MK_C>), EMBED_CONSTS)
MK_TAIL(embed_bf16_specialized, (mk_embed<bfloat, MK_C>), EMBED_CONSTS)
#endif // MK_BODIES_ONLY
