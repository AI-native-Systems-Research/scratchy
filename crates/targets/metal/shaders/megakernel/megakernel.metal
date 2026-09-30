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
// The persistent threadgroups one launch runs — the GPU's cores, known at load. Every generated
// work split reads it: item `i` of a spread step at phase cursor `c` runs on threadgroup
// `(c + i) mod MK_P`, a pinned unit of lane `l` on threadgroup `l mod MK_P`.
constant uint MK_P [[function_constant(4096)]];
// Polls one grid-barrier wait may spend before it records a stall and gives up. Apple10 offers
// no forward-progress primitive (`atomic_wait*`, `critical_section` and `yield_simdgroup` fail
// pipeline creation), so a wait spins — boundedly.
constant uint MK_SPIN_LIMIT [[function_constant(4097)]];

// The launch's synchronization block, the stall word alone on its 128-byte line; one 128-byte line
// per persistent threadgroup follows it (`mk_arrived`). Mirrors `MkSyncBlock`
// (interpreter/metal/megakernel.rs).
struct MkSync {
  atomic_uint stall[4]; // [0] = 1 + the barrier site that gave up (0 = healthy), [1] = arrivals
                        // it saw, [2] = the barriers the launch had passed, [3] = the P it
                        // waited for
  uint line[28];
};

// The grid barriers threadgroup `tg` has arrived at, ever (wraps): its own line after the block.
METAL_FUNC device atomic_uint* mk_arrived(device MkSync* sync, uint tg) {
  return (device atomic_uint*)((device uchar*)(sync + 1) + tg * 128u);
}

// Stall word first, then this threadgroup's arrivals, which every thread keeps (`gen`): thread 0
// at kernel entry. `ok` holds [0] the verdict, [1] the arrivals. A launch after a stall (the host
// turns the stall word into a typed error and resets the block) waits at no barrier.
METAL_FUNC void mk_enter(device MkSync* sync, uint tg, uint t, thread uint& gen,
                         threadgroup uint* ok) {
  if (t == 0) {
    const bool healthy = atomic_load_explicit(&sync->stall[0], memory_order_relaxed) == 0u;
    ok[1] = atomic_load_explicit(mk_arrived(sync, tg), memory_order_relaxed);
    ok[0] = healthy ? 1u : 0u;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  gen = ok[1];
}

// A GRID BARRIER: every threadgroup's device writes before it are visible to every threadgroup
// after it. Release; thread 0 publishes this threadgroup's arrival on its own line; threads
// 0..P-1 each spin (bounded) until one threadgroup's line shows the same barrier; acquire. No
// read-modify-write and no second hop: the last arrival's store IS what the others wait for.
// `site` numbers the barrier in the kernel text; `gen0` is the arrivals at entry, so
// `gen - gen0` counts the barriers passed — the first is also the co-residency check-in (a
// threadgroup the GPU never scheduled alongside the others shows as a stall there). Once a wait
// gave up (`ok[0]` = 0) no later barrier waits and the launch runs to its end, the host reporting
// the stall: the kernel never exits early. An exit on a verdict read from threadgroup memory puts
// every later barrier under control flow the compiler cannot prove uniform — so compiled, a
// threadgroup of the Gemma-4 kernel hung at a layer loop's exit.
METAL_FUNC void mk_grid_sync(device MkSync* sync, uint tg, uint t, thread uint& gen, uint gen0,
                             threadgroup uint* ok, uint site) {
  const uint mine = gen + 1u;
  atomic_thread_fence(mem_flags::mem_device, memory_order_release, thread_scope_device);
  threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
  if (t == 0) atomic_store_explicit(mk_arrived(sync, tg), mine, memory_order_relaxed);
  if (t < MK_P && ok[0] != 0u) {
    device atomic_uint* line = mk_arrived(sync, t);
    for (uint spins = 0;; ++spins) {
      if (int(atomic_load_explicit(line, memory_order_relaxed) - mine) >= 0) break;
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
          for (uint l = 0; l < MK_P; ++l) {
            const uint at = atomic_load_explicit(mk_arrived(sync, l), memory_order_relaxed);
            seen += int(at - mine) >= 0 ? 1u : 0u;
          }
          atomic_store_explicit(&sync->stall[1], seen, memory_order_relaxed);
          atomic_store_explicit(&sync->stall[2], mine - 1u - gen0, memory_order_relaxed);
          atomic_store_explicit(&sync->stall[3], MK_P, memory_order_relaxed);
        }
        ok[0] = 0u;
        break;
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  atomic_thread_fence(mem_flags::mem_device, memory_order_acquire, thread_scope_device);
  gen = mine;
}

// The first work item of a spread step on threadgroup `tg`, the phase cursor at `c`.
METAL_FUNC uint mk_first(uint tg, uint c) { return (tg + MK_P - c % MK_P) % MK_P; }

// The work items of a grid the load sizes (`vpi` virtual threadgroups per item).
METAL_FUNC uint mk_items(uint3 grid, uint vpi) { return (grid.x * grid.y * grid.z + vpi - 1u) / vpi; }
#endif // MK_ENUMERATE
