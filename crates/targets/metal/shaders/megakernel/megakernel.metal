// SPDX-License-Identifier: Apache-2.0
// THE DECODE MEGAKERNEL'S BODIES: the translation unit the macro-generated kernel completes.
//
// `megakernel_bake.rs` compiles the whole bucket-1 decode forward into ONE
// `[[kernel]] void mk_forward(...)` — the tape's steps as straight-line calls to the adapters
// below inside its rolled layer loops, every constant a literal, work split statically over the
// `MK_P` persistent threadgroups, a grid barrier (`mk_grid_sync`) exactly where the dataflow
// crosses threadgroups. build.rs inlines this file's local includes into one self-contained text
// (`mk_bodies.metal`); the worker appends the tape's generated kernel and compiles the library
// once at load.
//
// build.rs also preprocesses this file with `-DMK_ENUMERATE`: each `MK_ADAPTER` prints a marker,
// and the markers become `MK_ADAPTERS`, the adapters the bake may call.
#define MK_BODIES_ONLY 1
#include "megakernel/mk_common.h"

#define MK_LIB embed
#include "embed.metal"
#undef MK_LIB
#define MK_LIB rmsnorm
#include "rmsnorm.metal"
#undef MK_LIB
#define MK_LIB vision_layernorm
#include "vision_layernorm.metal"
#undef MK_LIB
#define MK_LIB fused_add_rmsnorm
#include "fused_add_rmsnorm.metal"
#undef MK_LIB
#define MK_LIB quantized_qmv
#include "quantized_qmv.metal"
#undef MK_LIB
#define MK_LIB silu_mul
#include "silu_mul.metal"
#undef MK_LIB
#define MK_LIB fused_gate_up_silu_mul
#include "fused_gate_up_silu_mul.metal"
#undef MK_LIB
#define MK_LIB gate_split
#include "gate_split.metal"
#undef MK_LIB
#define MK_LIB gate_apply
#include "gate_apply.metal"
#undef MK_LIB
#define MK_LIB gate_scale
#include "gate_scale.metal"
#undef MK_LIB
#define MK_LIB gdn_conv1d_varlen
#include "gdn_conv1d_varlen.metal"
#undef MK_LIB
#define MK_LIB gdn_gating
#include "gdn_gating.metal"
#undef MK_LIB
#define MK_LIB gdn_scan_varlen
#include "gdn_scan_varlen.metal"
#undef MK_LIB
#define MK_LIB gdn_rms_norm_gated
#include "gdn_rms_norm_gated.metal"
#undef MK_LIB
#define MK_LIB elementwise
#include "elementwise.metal"
#undef MK_LIB
#define MK_LIB quantized_dequantize
#include "quantized_dequantize.metal"
#undef MK_LIB
#define MK_LIB rope
#include "rope.metal"
#undef MK_LIB
#define MK_LIB turboquant
#include "turboquant.metal"
#undef MK_LIB
#define MK_LIB gemm
#include "gemm.metal"
#undef MK_LIB
#define MK_LIB argpartition
#include "argpartition.metal"
#undef MK_LIB
#define MK_LIB slice_trailing_cols
#include "slice_trailing_cols.metal"
#undef MK_LIB
#define MK_LIB take_along_axis
#include "take_along_axis.metal"
#undef MK_LIB
#define MK_LIB softmax
#include "softmax.metal"
#undef MK_LIB
#define MK_LIB moe_per_expert_scale
#include "moe_per_expert_scale.metal"
#undef MK_LIB
#define MK_LIB moe_weighted_sum
#include "moe_weighted_sum.metal"
#undef MK_LIB
#define MK_LIB attention
#include "attention.metal"
#undef MK_LIB

#ifndef MK_ENUMERATE
// The threadgroups one launch asks for — the GPU's cores, known at load. Those that start in
// time to check in (`mk_check_in`) run the token: the generated work split reads their count
// `mk_p`, item `i` of a spread step at phase cursor `c` running on participant `(c + i) mod mk_p`,
// a pinned unit of lane `l` on participant `l`.
constant uint MK_P [[function_constant(4096)]];
// Polls one grid-barrier wait may spend before it records a stall and gives up. The participants
// are all running, so a wait lasts as long as the slowest of them takes to arrive; there is no
// forward-progress primitive to wait on instead (Apple10: `atomic_wait*`, `critical_section` and
// `yield_simdgroup` fail pipeline creation), so a wait spins — boundedly.
constant uint MK_SPIN_LIMIT [[function_constant(4097)]];
// Polls the first threadgroups to check in wait for the others before the launch goes ahead
// without them. The GPU starts a launch's threadgroups together, unless other work holds some of
// its cores; then one can start 100+ ms late (an M1 Max, measured), and the token runs without it.
constant uint MK_CHECKIN_POLLS [[function_constant(4098)]];

// The launch's synchronization block: the stall word, then the check-in's ticket counter and its
// close word, alone on a 128-byte line; one 128-byte line per participant follows it
// (`mk_arrived`). The host zeroes everything after the stall word before every launch. Mirrors
// `MkSyncBlock` (interpreter/metal/megakernel.rs).
struct MkSync {
  atomic_uint stall[4]; // [0] = 1 + the barrier site that gave up (0 = healthy), [1] = arrivals
                        // it saw, [2] = the barriers the launch had passed, [3] = the
                        // participants it waited for
  atomic_uint checkin;  // tickets taken this launch
  atomic_uint closed;   // 0 while the check-in is open, then 1 + the participants
  uint line[26];
};

// The grid barriers participant `idx` has arrived at this launch: its own line after the block.
METAL_FUNC device atomic_uint* mk_arrived(device MkSync* sync, uint idx) {
  return (device atomic_uint*)((device uchar*)(sync + 1) + idx * 128u);
}

// THE CHECK-IN, before any work: thread 0 takes a ticket and waits (at most `MK_CHECKIN_POLLS`)
// for every threadgroup of the launch to take one; then the first to see the wait over closes
// the check-in at the tickets taken so far — the participants. Returns false for a threadgroup
// whose ticket came after the close: it exits having touched nothing. `ok` holds [0] the verdict
// the barriers read (a launch after an unreported stall waits at none), [1] the ticket, [2] the
// participants — written by thread 0 alone and read by every thread after a threadgroup barrier,
// so every thread of a threadgroup takes the same path out of it.
METAL_FUNC bool mk_check_in(device MkSync* sync, uint t, threadgroup uint* ok, thread uint& idx,
                            thread uint& p) {
  if (t == 0) {
    const uint ticket = atomic_fetch_add_explicit(&sync->checkin, 1u, memory_order_relaxed);
    uint closed = atomic_load_explicit(&sync->closed, memory_order_relaxed);
    for (uint polls = 0; closed == 0u; ++polls) {
      const uint taken = atomic_load_explicit(&sync->checkin, memory_order_relaxed);
      if (taken >= MK_P || polls >= MK_CHECKIN_POLLS) {
        uint open = 0u;
        atomic_compare_exchange_weak_explicit(&sync->closed, &open, 1u + min(taken, MK_P),
                                              memory_order_relaxed, memory_order_relaxed);
      }
      closed = atomic_load_explicit(&sync->closed, memory_order_relaxed);
    }
    const bool healthy = atomic_load_explicit(&sync->stall[0], memory_order_relaxed) == 0u;
    ok[0] = healthy ? 1u : 0u;
    ok[1] = ticket;
    ok[2] = closed - 1u;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  idx = ok[1];
  p = ok[2];
  return idx < p;
}

// The grid barrier's fences: release before a threadgroup publishes its arrival, acquire after
// it sees every other's. The acquire and release orders are Metal 4.1's; the standard this
// library compiles with (the build toolchain's, as for every shader) is older where the
// toolchain is (Xcode 26: 4.0), and there the barrier fences sequentially consistently — at least
// the order it needs, in every standard from 3.2 on.
#if __METAL_VERSION__ >= 410
#define MK_FENCE_RELEASE memory_order_release
#define MK_FENCE_ACQUIRE memory_order_acquire
#else
#define MK_FENCE_RELEASE memory_order_seq_cst
#define MK_FENCE_ACQUIRE memory_order_seq_cst
#endif

// A GRID BARRIER among the launch's `p` participants: every participant's device writes before
// it are visible to every participant after it. Release; thread 0 publishes this participant's
// arrival on its own line; threads 0..p-1 each spin (bounded) until one participant's line shows
// the same barrier; acquire. No read-modify-write and no second hop: the last arrival's store IS
// what the others wait for. `site` numbers the barrier in the kernel text; `gen` counts the
// barriers passed this launch. Once a wait gave up (`ok[0]` = 0) no later barrier waits and the
// launch runs to its end, the host reporting the stall: the kernel never exits mid-way. An exit
// on a verdict read from threadgroup memory while other threads may still write it puts every
// later barrier under control flow that is not uniform — so compiled, a threadgroup of the
// Gemma-4 kernel hung at a layer loop's exit.
METAL_FUNC void mk_grid_sync(device MkSync* sync, uint idx, uint p, uint t, thread uint& gen,
                             threadgroup uint* ok, uint site) {
  const uint mine = gen + 1u;
  atomic_thread_fence(mem_flags::mem_device, MK_FENCE_RELEASE, thread_scope_device);
  threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
  if (t == 0) atomic_store_explicit(mk_arrived(sync, idx), mine, memory_order_relaxed);
  if (t < p && ok[0] != 0u) {
    device atomic_uint* line = mk_arrived(sync, t);
    for (uint spins = 0;; ++spins) {
      if (atomic_load_explicit(line, memory_order_relaxed) >= mine) break;
      const bool check = (spins & 1023u) == 1023u;
      if (check && atomic_load_explicit(&sync->stall[0], memory_order_relaxed) != 0u) {
        ok[0] = 0u;
        break;
      }
      if (spins == MK_SPIN_LIMIT) {
        uint expected = 0u;
        if (atomic_compare_exchange_weak_explicit(&sync->stall[0], &expected, 1u + site,
                                                  memory_order_relaxed, memory_order_relaxed)) {
          uint seen = 0u;
          for (uint l = 0; l < p; ++l) {
            seen += atomic_load_explicit(mk_arrived(sync, l), memory_order_relaxed) >= mine;
          }
          atomic_store_explicit(&sync->stall[1], seen, memory_order_relaxed);
          atomic_store_explicit(&sync->stall[2], mine - 1u, memory_order_relaxed);
          atomic_store_explicit(&sync->stall[3], p, memory_order_relaxed);
        }
        ok[0] = 0u;
        break;
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  atomic_thread_fence(mem_flags::mem_device, MK_FENCE_ACQUIRE, thread_scope_device);
  gen = mine;
}

// The first work item of a spread step on participant `idx` of `p`, the phase cursor at `c`.
METAL_FUNC uint mk_first(uint idx, uint p, uint c) { return (idx + p - c % p) % p; }

// The work items of a grid the load sizes (`vpi` virtual threadgroups per item).
METAL_FUNC uint mk_items(uint3 grid, uint vpi) { return (grid.x * grid.y * grid.z + vpi - 1u) / vpi; }
#endif // MK_ENUMERATE
