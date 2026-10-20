# SPDX-License-Identifier: Apache-2.0
"""THE DOUBLE-UNROLL COUNT TWIN — attention's M=1 one-pass body at two grid depths.

This is `kernels/attn.py`'s paged-decode arm with exactly the machinery the count wall
does not need removed: no mask (`HAS_MASK` and its descriptor are gone), no segment
corners, no Kᵀ/V pooling beyond the one sweep. What is kept verbatim is everything the
wall lives in:

* the THREE-AXIS grid `(NUM_SEQS, NKVH, GQA)` and the corner arithmetic it induces —
  `q`'s row `pid_s*NQH + pid_h*GQA + pid_g`, `kt`'s row block `(pid_s*NKVH + pid_h)*HD`,
  the id table's `pid_s*(NKVH*W) + pid_h*W`. The div/mod delinearisation of the three
  pids is what makes these corners NON-uniform, which is what fires
  `unroll_grid_positions`' gate: an affine corner would be claimed by
  `uniform_grid_corners` and the pass would never run.
* the ONE GATHER PAIR — the V pool through ids LOADED from the row-id table, never
  computed in-register (the one-index-per-descriptor law; the ids are i32 and the table
  is `[NUM_SEQS*NKVH*W]`, one `[W]` block per position).
* the TWO DOTS at M=1 (the score leg against the whole-sweep `[HD, W]` Kᵀ window, the
  value leg against the gathered `[W, HD]` rows) — a MULTI-dot body, so
  `dot_to_linalg`'s single-dot canonical verifiers (which demand a block equal to the
  descriptor's full extent, offsets at `[0, 0]`) never gate it. That is the same
  exemption the production kernel rides.
* the ONE-PASS SOFTMAX and its EPILOGUE DIVIDE — `max`/`exp2`/`sum` at rows == 1 and
  `o / l[:, None]`, the RealDiv whose extent check is the refusal the count wall ends
  at.

THE TWO ROWS THAT BAKE THIS FIXTURE differ only in `NUM_SEQS` (and the grid that
follows it): `(1, 8, 4)` = 32 positions — exactly the 32-core fold bound, the path
`attn_paged_decode_batch_fold` proved door-legal (every position's windows
materialized onto its own clone) — and `(2, 8, 4)` = 64 positions, TWO TRIPS per core:
`fold_grid_work_loop`'s unroll arm regrids to `[num_cores]` while the body carries two
copies per landmark value, and `unroll_grid_positions`' partitioned-parameter count
must then count `n * trips` tiles, not `n`. Before the count fix the 64-position row
refuses at the door; after it, it bakes at exactly twice the 32-position row's op
count. The row pair is the `to_ktir.rs` count fix's gate.

No `inputs`/`reference` harness: like the production attention rows, these
configurations exercise the DOOR, and the numeric legs report `Missing` for them.
"""

import triton
import triton.language as tl


@triton.jit
def banded_attn_fwd(desc_q, desc_o, desc_kt, desc_v, desc_v_ids,  #
                    NUM_SEQS: tl.constexpr, NQH: tl.constexpr,  #
                    NKVH: tl.constexpr, HD: tl.constexpr,  #
                    GQA: tl.constexpr, PAGES: tl.constexpr,  #
                    PAGE_SLOTS: tl.constexpr, SCALE: tl.constexpr):
    """The M=1 one-pass attention body — the module doc's contract. ONE program per
    (sequence, KV head, q head): one q row, the whole sweep in one pass, Kᵀ direct,
    V gathered through the id table, the softmax one-pass at rows == 1."""
    pid_s = tl.program_id(0)   # the sequence
    pid_h = tl.program_id(1)   # the KV head
    pid_g = tl.program_id(2)   # the q head within the KV head's group
    width: tl.constexpr = PAGES * PAGE_SLOTS
    q_desc = tl.make_tensor_descriptor(desc_q, shape=[NUM_SEQS * NQH, HD],
                                       strides=[HD, 1], block_shape=[1, HD])
    o_desc = tl.make_tensor_descriptor(desc_o, shape=[NUM_SEQS * NQH, HD],
                                       strides=[HD, 1], block_shape=[1, HD])
    # The Kᵀ plane, logical page order per (sequence, head): ONE [HD, W] window for
    # the whole sweep (non-pow2 blocks are legal — the one-pass arm's own precedent).
    kt_desc = tl.make_tensor_descriptor(
        desc_kt, shape=[NUM_SEQS * NKVH * HD, PAGES * PAGE_SLOTS],
        strides=[PAGES * PAGE_SLOTS, 1], block_shape=[HD, width])
    # The V pool, physical pages head-major: the descriptor's block is [1, HD] — the
    # gathered row count comes from the index vector, so a [W]-wide id vector gathers
    # [W, HD] rows.
    v_desc = tl.make_tensor_descriptor(
        desc_v, shape=[NKVH * PAGES * PAGE_SLOTS, HD],
        strides=[HD, 1], block_shape=[1, HD])
    # The V row-id table, 1D i32 at row-id granularity: the whole sweep's ids in one
    # [W] block per position, head base riding the table content.
    v_ids_desc = tl.make_tensor_descriptor(
        desc_v_ids, shape=[NUM_SEQS * NKVH * PAGES * PAGE_SLOTS], strides=[1],
        block_shape=[width])

    # The program's one q row: [1, HD] at row pid_s*NQH + pid_h*GQA + pid_g.
    q = q_desc.load([pid_s * NQH + pid_h * GQA + pid_g, 0])  # [1, HD]
    # The whole sweep, ONE pass: Kᵀ direct, V through the id table's ids.
    kt = kt_desc.load([(pid_s * NKVH + pid_h) * HD, 0])      # [HD, W]
    v_ids = v_ids_desc.load([pid_s * (NKVH * PAGES * PAGE_SLOTS)
                             + pid_h * (PAGES * PAGE_SLOTS)])  # [W] i32
    v_rows = v_desc.gather(v_ids, 0)                          # [W, HD]
    # The score dot — M=1, the production decode arm's own form. The scale rides
    # log2(e) as an immediate multiply on the dot's result.
    qk = tl.dot(q, kt, out_dtype=tl.float16)                  # [1, W]
    qk = qk * (SCALE * 1.44269504)
    # One-pass softmax at rows=1 — no seeds, no recurrence.
    m = tl.max(qk, 1)                                        # [1]
    p = tl.math.exp2((qk - m[:, None]).to(tl.float32)).to(tl.float16)
    l = tl.sum(p, 1)                                         # [1]
    o = tl.dot(p, v_rows, out_dtype=tl.float16)              # [1, HD]
    o = o / l[:, None]
    o_desc.store([pid_s * NQH + pid_h * GQA + pid_g, 0], o)
