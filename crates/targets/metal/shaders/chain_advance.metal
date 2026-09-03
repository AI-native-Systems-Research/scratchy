// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// chain_advance — Phase 6 primitive
//
// Between K-step draft iterations, advances per-req `runtime.positions`,
// `runtime.slot_mapping`, and `runtime.seqused_k` so the next iter's
// forward dispatches see the correct values without a host roundtrip.
//
// For each req i (one thread per req):
//   new_pos        = positions[i] + 1
//   block_idx      = new_pos / block_size
//   offset         = new_pos % block_size
//   new_slot       = block_table[i * block_table_stride + block_idx]
//                    * block_size + offset
//   new_seqused_k  = new_pos + 1
//
// All writes are to the SAME buffers the inputs were read from
// (in-place update). The next forward dispatch on the same compute
// encoder reads the updated values; Metal's intra-encoder write→read
// hazard tracking serializes correctly.
//
// If block_idx >= number of valid blocks for the req (out of bounds),
// write u32::MAX as the slot — the rope_append kernel checks for
// the sentinel and skips the cache write. (We can't easily get
// "number of valid blocks" per req here without additional metadata,
// so this kernel assumes the caller's block_table is sized to cover
// the K-step horizon — which the scheduler reserves
// `num_lookahead_tokens = K` blocks per req for.)
//
// Bindings (must match `chain_advance::dispatch_chain_advance`):
//   buffer(0) = positions    [num_reqs]                              read+write u32
//   buffer(1) = slot_mapping [num_reqs]                              write u32
//   buffer(2) = seqused_k    [num_reqs]                              write u32
//   buffer(3) = block_table  [num_reqs * block_table_stride]         read u32
//   buffer(4) = block_size           constant uint
//   buffer(5) = block_table_stride   constant uint
//   buffer(6) = num_reqs             constant uint
//
// Dispatch: one threadgroup, num_reqs threads. Cheap (~1 us).
// ---------------------------------------------------------------------------
kernel void chain_advance(
    device       uint*  positions          [[buffer(0)]],
    device       uint*  slot_mapping       [[buffer(1)]],
    device       uint*  seqused_k          [[buffer(2)]],
    device const uint*  block_table        [[buffer(3)]],
    constant     uint&  block_size         [[buffer(4)]],
    constant     uint&  block_table_stride [[buffer(5)]],
    constant     uint&  num_reqs           [[buffer(6)]],
    uint  tid [[thread_position_in_threadgroup]])
{
    if (tid >= num_reqs) return;
    uint new_pos = positions[tid] + 1u;
    uint block_idx = new_pos / block_size;
    uint offset    = new_pos - block_idx * block_size;
    // Block-table row for this req starts at tid * block_table_stride.
    uint block_id = block_table[tid * block_table_stride + block_idx];
    uint new_slot = block_id * block_size + offset;
    positions[tid]    = new_pos;
    slot_mapping[tid] = new_slot;
    seqused_k[tid]    = new_pos + 1u;
}
