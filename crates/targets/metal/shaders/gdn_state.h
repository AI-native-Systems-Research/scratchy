// SPDX-License-Identifier: Apache-2.0
//
// One sequence's record in the `state_indices` buffer every GDN and Mamba-2 kernel reads
// (`gdn_conv1d_varlen`, `gdn_scan_varlen`, `gdn_scan_simd`, `gdn_scan_pipelined`, `gdn_decode`):
// `scratchy_layers::gdn_state::GdnSeqSlots`, the same 24 bytes (`[slot, source, r0, s0, r1, s1]`).
//
//   slot   — the sequence's own slot: every state the step writes lands there.
//   source — `0`, or `s + 1`: a step that starts from its slot (`GdnStep` start 0) reads its start
//            state from slot `s` instead — a prefix-cache snapshot it resumes from. Every other
//            start ignores it.
//   saves  — up to two `(rows_through, snapshot)` stores: once the step has run the sequence's first
//            `rows_through` rows (its own rows from `cu_seqlens[seq]`, never a replayed one), the
//            state after them is ALSO stored into slot `snapshot`'s state entry. `0` rows: none.
//
// Snapshots are slots past the live ones in the same per-layer buffers, laid out like any slot, so a
// kernel reaches one by its index alone. All zeros is a plain step on slot 0.

#pragma once

#include <metal_stdlib>

struct GdnSeqSlots {
  int slot;
  int source;
  int2 saves[2];
};

static_assert(sizeof(GdnSeqSlots) == 24, "GdnSeqSlots is the host's 6-word record");

// A sequence's saves.
constant constexpr uint GDN_SAVES = 2u;

// The slot a step reads its start state from: a slot start's snapshot when it restores one, else
// the sequence's own (whose checkpoints a checkpoint start reads).
inline int gdn_start_slot(GdnSeqSlots s, uint start) {
  return start == 0u && s.source > 0 ? s.source - 1 : s.slot;
}

// The rows each of a sequence's saves stores the state after (`0`: no store).
inline int2 gdn_saves_through(GdnSeqSlots s) {
  return int2(s.saves[0].x, s.saves[1].x);
}

// Whether save `j` stores the state after step row `row`: the step's own rows count from 0, and a
// replayed row (negative) never stores — nor does a save of `0` rows through.
inline bool gdn_saves_after(int2 through, uint j, int row) {
  return row >= 0 && through[j] == row + 1;
}

// Where a block of rows starting at `t0` and running to `t1` ends instead: at the first save's
// row inside it, so the state is stored between blocks and the per-token loop carries no save
// (`gdn_scan_pipelined`).
inline int gdn_cut_at_save(int2 through, int t0, int t1) {
  for (uint j = 0; j < GDN_SAVES; j++) {
    if (through[j] > t0 && through[j] < t1) {
      t1 = through[j];
    }
  }
  return t1;
}

// Whether a sequence saves nothing this step — decode, and every prefill chunk short of a save
// point: `gdn_scan_pipelined` then cuts no block at a save.
inline bool gdn_saves_none(int2 through) {
  return through.x <= 0 && through.y <= 0;
}

// A scan whose rows run one at a time runs a saving step in `GDN_SAVES + 1` stretches, a save's
// store after each but the last, so its per-row loop carries no save: stretch `k` ends at the row
// its `k`-th save stores after (saves in row order, one of `0` rows last), the last at the step's
// end.

// The save a scan reaches `k`-th.
inline uint gdn_save_at(int2 through, uint k) {
  const uint first = through.y > 0 && (through.x <= 0 || through.y < through.x) ? 1u : 0u;
  return k == 0u ? first : 1u - first;
}

// Where stretch `k` of a step of `rows` rows ends: the step's rows before it has run.
inline int gdn_stretch_end(int2 through, uint k, int rows) {
  const int r = k < GDN_SAVES ? through[gdn_save_at(through, k)] : 0;
  return r > 0 ? metal::min(r, rows) : rows;
}

// The save stored once stretch `k` of a step of `rows` rows has run, or -1: none.
inline int gdn_stretch_save(int2 through, uint k, int rows) {
  const uint j = gdn_save_at(through, k);
  return k < GDN_SAVES && through[j] > 0 && through[j] <= rows ? int(j) : -1;
}
