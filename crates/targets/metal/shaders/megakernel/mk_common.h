// SPDX-License-Identifier: Apache-2.0
// What every normalized shader and the compiled decode megakernels (`megakernel.metal`) share.
//
// A normalized shader keeps its `[[kernel]]` as a thin wrapper over a body function; the body
// also backs a megakernel ADAPTER, which plays one virtual threadgroup of a tape step inside a
// 1024-thread persistent threadgroup. The macro generates each megakernel as straight-line calls
// to the adapters: an `MkStep` is one step as the generated code spells it (literal geometry, the
// step's row of the run's address table); an `MkLane` is one thread's view of the virtual
// threadgroup it plays, presented exactly as the dispatch builtins present it.
#pragma once
#include <metal_stdlib>
using namespace metal;

// Adapters inline into the megakernel. An adapter whose rounding depends on the function around
// it — which of its products the backend fuses into which adds — is compiled as a function of its
// own (`MK_FUNC_ALONE`), as its dispatch kernel is.
#define MK_FUNC METAL_FUNC
#define MK_FUNC_ALONE __attribute__((noinline))

// A shader's function constants are ONE list, `<FILE>_CONSTS(X)` of `X(type, name, NAME, index)`:
// declared here for the dispatch kernels, and spelled as literals by the generated megakernel's
// constant policies (`megakernel_bake.rs`) from the same list, so both read the same values.
#define MK_FC_DECLARE(ty, name, NAME, i) constant ty NAME [[function_constant(i)]];
#define MK_FC_ACCESSOR(ty, name, NAME, i) \
  static METAL_FUNC ty name() { return NAME; }
// The list of a kernel without function constants.
#define MK_NO_CONSTS(X)
// Inside `template <typename G> struct …`: the listed constants of the generated policy `G` (a
// kernel whose bodies read DERIVED constants wraps `G` and adds those).
#define MK_FC_FORWARD(ty, name, NAME, i) \
  static METAL_FUNC ty name() { return G::name(); }

// `MK_ADAPTER(symbol, tg_bytes, coherent, call, CONSTS)`: the megakernel form of one
// instantiation line — `symbol` is the dispatch kernel's host name, `tg_bytes` the threadgroup
// memory one virtual threadgroup owns, `coherent` the mask of binding indices the adapter reads
// or writes device-coherently, `call` the adapter with `MK_C` standing for the step's constant
// policy. Under `MK_ENUMERATE` it prints a marker build.rs parses into `MK_ADAPTERS`; otherwise it
// is empty. An item plays up to all 1024 threads of the persistent threadgroup; every binding of
// the step's row counts as written.
// `MK_STREAM(symbol, item_threads, coherent, writes, call, CONSTS)`: a STREAMING adapter (no
// threadgroup memory) whose items play at most `item_threads` threads, the rest of the persistent
// threadgroup idling through them — the width its body streams weights fastest at
// (`MkAdapter::item_threads`) — and that writes only the bindings of the mask `writes`.
// `MK_STREAM_ROWS(symbol, item_threads, short_threads, row, short_below, coherent, writes, call,
// CONSTS)`: an `MK_STREAM` whose fastest width depends on its row length — a step whose constant
// at slot `row` is below `short_below` plays items of up to `short_threads` (few passes per row:
// more rows in flight hide each row's latency) (`MkAdapter::short_rows`).
// `MK_TAIL(symbol, coherent, call, CONSTS)`: an ELEMENTWISE adapter (a few loads and stores per
// thread, no threadgroup memory) — a step of a few such items may be played whole by the
// threadgroup that ran what it waits on instead of spreading after a grid barrier
// (`MkAdapter::tail`); a heavier step never is.
#ifdef MK_ENUMERATE
#define MK_ENUM_C(ty, name, NAME, i) ty name i ;
#define MK_ADAPTER(sym, tg_bytes, coherent, call, CONSTS) \
  @@MK MK_LIB sym tg_bytes coherent 1024 0xffffffff 0 0 0 0 @@C CONSTS(MK_ENUM_C) @@CALL call @@END
#define MK_STREAM(sym, item_threads, coherent, writes, call, CONSTS) \
  @@MK MK_LIB sym 0 coherent item_threads writes 0 0 0 0 @@C CONSTS(MK_ENUM_C) @@CALL call @@END
#define MK_STREAM_ROWS(sym, item_threads, short_threads, row, short_below, coherent, writes, call, \
                       CONSTS)                                                                    \
  @@MK MK_LIB sym 0 coherent item_threads writes 0 short_threads row short_below @@C              \
      CONSTS(MK_ENUM_C) @@CALL call @@END
#define MK_TAIL(sym, coherent, call, CONSTS) \
  @@MK MK_LIB sym 0 coherent 1024 0xffffffff 1 0 0 0 @@C CONSTS(MK_ENUM_C) @@CALL call @@END
#else
#define MK_ADAPTER(sym, tg_bytes, coherent, call, CONSTS)
#define MK_STREAM(sym, item_threads, coherent, writes, call, CONSTS)
#define MK_STREAM_ROWS(sym, item_threads, short_threads, row, short_below, coherent, writes, call, \
                       CONSTS)
#define MK_TAIL(sym, coherent, call, CONSTS)
#endif

// One tape step as the generated megakernel spells it: every field but `addr` is a literal, so
// the adapter's geometry folds; `addr` is the step's row of the run's address table (binding
// index → buffer address, filled at load).
struct MkStep {
  const constant ulong* addr; // argument addresses, by binding index
  uint3 grid;                 // threadgroups per grid
  uint3 tpg;                  // threads per threadgroup
  uint vtgs_per_item;         // virtual threadgroups one physical threadgroup plays per item
  uint vtg_stride;            // roundup(threads per threadgroup, 32): vTGs start on simdgroups
  uint tg_bytes;              // threadgroup bytes one virtual threadgroup owns
};

// One thread's view of the virtual threadgroup it plays.
struct MkLane {
  uint3 tg_pos;          // [[threadgroup_position_in_grid]] of the virtual TG
  uint3 tid3;            // [[thread_position_in_threadgroup]]
  uint3 tpg;             // [[threads_per_threadgroup]]
  uint tid;              // linear index in the virtual TG
  uint simd_gid;         // [[simdgroup_index_in_threadgroup]]
  uint simd_lid;         // [[thread_index_in_simdgroup]]
  uint vtg_local;        // which virtual TG of this physical TG
  bool live;             // a real thread of a real (in-grid) virtual TG
};

METAL_FUNC MkLane mk_lane(thread const MkStep& s, uint item, uint t) {
  MkLane l;
  const uint S = s.tpg.x * s.tpg.y * s.tpg.z;
  const uint V = s.grid.x * s.grid.y * s.grid.z;
  l.vtg_local = t / s.vtg_stride;
  const uint lin = t % s.vtg_stride;
  const uint vtg = item * s.vtgs_per_item + l.vtg_local;
  l.live = l.vtg_local < s.vtgs_per_item && vtg < V && lin < S;
  l.tpg = s.tpg;
  l.tg_pos = uint3(vtg % s.grid.x, (vtg / s.grid.x) % s.grid.y, vtg / (s.grid.x * s.grid.y));
  l.tid3 = uint3(lin % s.tpg.x, (lin / s.tpg.x) % s.tpg.y, lin / (s.tpg.x * s.tpg.y));
  l.tid = lin;
  l.simd_gid = lin / 32u;
  l.simd_lid = lin % 32u;
  return l;
}

// [[threads_per_grid]] of the step.
METAL_FUNC uint3 mk_threads_per_grid(thread const MkStep& s) { return s.grid * s.tpg; }

// [[thread_position_in_grid]] of the lane.
METAL_FUNC uint3 mk_thread_in_grid(MkLane l) { return l.tg_pos * l.tpg + l.tid3; }

// The threadgroup memory the lane's virtual TG owns. Lanes past the last virtual TG an item plays
// share one sink region (the bake sizes it): their writes land where no live lane reads.
METAL_FUNC threadgroup uchar* mk_region(thread const MkStep& s, MkLane l, threadgroup uchar* tg) {
  return tg + min(l.vtg_local, s.vtgs_per_item) * s.tg_bytes;
}

// The megakernel's memory: activations, KV pages and codec stores are written by one threadgroup
// and read by another inside one dispatch, so every such pointer is device-coherent. Weights and
// host-written runtime inputs keep the plain qualifier.
template <typename T> using mk_ptr = coherent(device) device T*;
template <typename T> using mk_cptr = const coherent(device) device T*;
