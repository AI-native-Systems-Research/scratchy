// SPDX-License-Identifier: Apache-2.0
// What every normalized shader and the compiled decode megakernels (`megakernel.metal`) share.
//
// A normalized shader keeps its `[[kernel]]` as a thin wrapper over a body function; the body
// also backs a megakernel ADAPTER, which plays one virtual threadgroup of a tape step inside a
// 1024-thread threadgroup of a segment kernel's launch. The macro generates each segment kernel as
// straight-line calls to the adapters: an `MkStep` is one step as the generated code spells it
// (literal geometry, the step's row of the launch's address-table block); an `MkLane` is one
// thread's view of the virtual threadgroup it plays, presented exactly as the dispatch builtins
// present it.
#pragma once
#include <metal_stdlib>
using namespace metal;

// Adapters inline into the segment kernels. An adapter whose rounding depends on the function
// around it — which of its products the backend fuses into which adds — is compiled as a function
// of its own (`MK_FUNC_ALONE`), as its dispatch kernel is.
#define MK_FUNC METAL_FUNC
#define MK_FUNC_ALONE __attribute__((noinline))

// A shader's function constants are ONE list, `<FILE>_CONSTS(X)` of `X(type, name, NAME, index)`:
// declared here for the dispatch kernels, and spelled as literals by the generated kernels'
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

// `MK_ADAPTER(symbol, tg_bytes, call, CONSTS)`: the megakernel form of one instantiation line —
// `symbol` is the dispatch kernel's host name, `tg_bytes` the threadgroup memory one virtual
// threadgroup owns, `call` the adapter with `MK_C` standing for the step's constant policy. Under
// `MK_ENUMERATE` it prints a marker build.rs parses into `MK_ADAPTERS`; otherwise it is empty.
// Every binding of the step's row counts as written.
// `MK_STREAMING(symbol, tg_bytes, rows, bits, gs, scale_bytes, k, n, w, s, b, x, y, writes, call,
// calibrate, CONSTS)`: a STREAMING adapter — each virtual threadgroup reads `rows` rows of weights,
// a row the step's constant at slot `k` values of `bits` bits plus a scale and a bias of
// `scale_bytes` per `gs` values, of the constant at slot `n` rows (`MkAdapter::stream`) — that
// writes only the bindings of the mask `writes`. Its bindings: the weights at `w`, the scales and
// biases at `s` and `b` (255: none), the input vector at `x`, the output at `y`. `calibrate` is an
// adapter that streams the same rows the same way through the same bindings: the load measures
// the device with it on synthetic weights. How many threads an item plays is the launch's work
// split, solved from those measurements.
// `MK_STREAM(symbol, rows, bits, gs, scale_bytes, k, n, writes, call, calibrate, CONSTS)`: an
// `MK_STREAMING` matvec of no threadgroup memory, bound `w, scales, biases, x, y`.
// `MK_TAIL(symbol, call, CONSTS)`: an ELEMENTWISE adapter (a few loads and stores per thread, no
// threadgroup memory) — a step of such items may be played whole by the threadgroup that ran what
// it waits on (`MkAdapter::tail`), a run the split weighs against spreading it after a launch
// boundary of its own.
#ifdef MK_ENUMERATE
#define MK_ENUM_C(ty, name, NAME, i) ty name i ;
#define MK_ADAPTER(sym, tg_bytes, call, CONSTS) \
  @@MK MK_LIB sym tg_bytes 0xffffffff 0 0 0 0 0 0 0 0 0 0 0 0 @@C CONSTS(MK_ENUM_C) @@CALL call @@END
#define MK_STREAMING(sym, tg_bytes, rows, bits, gs, scale_bytes, k, n, w, s, b, x, y, writes, call,  \
                     calibrate, CONSTS)                                                         \
  @@MK MK_LIB sym tg_bytes writes 0 rows bits gs scale_bytes k n w s b x y @@C CONSTS(MK_ENUM_C) \
      @@CALL call @@CALIB calibrate @@END
#define MK_TAIL(sym, call, CONSTS) \
  @@MK MK_LIB sym 0 0xffffffff 1 0 0 0 0 0 0 0 0 0 0 0 @@C CONSTS(MK_ENUM_C) @@CALL call @@END
#else
#define MK_ADAPTER(sym, tg_bytes, call, CONSTS)
#define MK_STREAMING(sym, tg_bytes, rows, bits, gs, scale_bytes, k, n, w, s, b, x, y, writes, call,  \
                     calibrate, CONSTS)
#define MK_TAIL(sym, call, CONSTS)
#endif
#define MK_STREAM(sym, rows, bits, gs, scale_bytes, k, n, writes, call, calibrate, CONSTS)       \
  MK_STREAMING(sym, 0, rows, bits, gs, scale_bytes, k, n, 0, 1, 2, 3, 4, writes, call, calibrate, \
               CONSTS)

// One tape step as a generated segment kernel spells it: every field but `addr` is a literal, so
// the adapter's geometry folds; `addr` is the step's row of the address-table block its launch
// binds (binding index → buffer address, filled at load).
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

// The pointers the adapters pass the tape's own memory by — activations, KV pages, codec stores:
// plain device memory, as in the dispatch kernels. Inside one launch no threadgroup depends on
// another's writes (a segment orders every wait on one threadgroup; where every threadgroup plays
// a unit, each reads its own copy's writes, all of the same values), and the barrier between two
// launches makes one's writes visible to the next.
template <typename T> using mk_ptr = device T*;
template <typename T> using mk_cptr = const device T*;
