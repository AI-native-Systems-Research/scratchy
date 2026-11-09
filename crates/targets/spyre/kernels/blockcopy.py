# SPDX-License-Identifier: Apache-2.0
"""THE BLOCK COPY'S KERNEL — `dst[:, COL_OFF : COL_OFF + C_SRC] = src`: a whole `[R, C_SRC]`
source placed as a column block of an `[R, C_DST]` destination. An all-gather consume's leg: one
rank's block (its own, or its peer's out of the receive window) landing at that rank's columns.

BODY PROVENANCE: `elementwise.py`'s row-blocked whole-width form — `N_BLOCKS` full `[BLOCK_M,
C_SRC]` tiles plus a `[TAIL_H, C_SRC]` tail, each one load and one store — with the destination
column shifted by `COL_OFF`. The door (`lower_ktir_to_superdsc::block_copy`) reads the block off
the program's own first tiles (the source at its corner, the destination at `[0, COL_OFF]`) and
the views' extents, and lowers the whole program to ONE identity over the run of stick columns,
which stick-major layout makes contiguous at both ends.

⛔ WHOLE-WIDTH ROW BLOCKS, NOT ONE COPY PER STICK. The per-stick walk (`R · C_SRC / STK` copies)
unrolled into ~`12 · R · C_SRC / STK` KTIR ops — 196,616 for a `[512, 2048]` copy, 313 MB of
build-script memory for that one program (MEASURED, crate-only, nickm5 2026-11-08) — and a bake
mints one per (rank, gather, rung). The row blocks are the elementwise LX law's (`rows_per_block`),
so the program is `O(R / BLOCK_M)` ops at any width (8 ops, 10 MB for the same copy), and the door
emits the same one descriptor.

THE SPLICE'S OWN CONTRACT (what `scratchy-triton-splice` states about this kernel):

* PARAMETERS, IN ORDER: `desc_src`, `desc_dst` — the source, then the destination.
* CONSTEXPRS: `R`, `C_SRC`, `C_DST`, `COL_OFF` (elements), `STK` (the operand's stick in
  elements), `BLOCK_M`, `N_BLOCKS`, `TAIL_H` (the row blocks: `N_BLOCKS · BLOCK_M + TAIL_H = R`) —
  stated by the splice from the two regions and the LX law; the door refuses unless `C_SRC`,
  `C_DST` and `COL_OFF` are whole sticks and the block fits.
* GRID: `[1]`.

⭐ THE TRIP COUNT IS CONSTEXPR, so `to_ktir` unrolls the walk into straight-line copies — the form
the door's region walk requires, exactly as for `elementwise.py`.
"""

import triton
import triton.language as tl


@triton.jit
def blockcopy_fwd(desc_src, desc_dst,  #
                  R: tl.constexpr, C_SRC: tl.constexpr, C_DST: tl.constexpr,  #
                  COL_OFF: tl.constexpr, STK: tl.constexpr,  #
                  BLOCK_M: tl.constexpr, N_BLOCKS: tl.constexpr, TAIL_H: tl.constexpr,  #
                  ):
    # The column block is whole sticks of the operand's own stick (`STK` elements).
    tl.static_assert(C_SRC % STK == 0)
    tl.static_assert(COL_OFF % STK == 0)
    src_desc = tl.make_tensor_descriptor(desc_src, shape=[R, C_SRC],
                                         strides=[C_SRC, 1],
                                         block_shape=[BLOCK_M, C_SRC])
    dst_desc = tl.make_tensor_descriptor(desc_dst, shape=[R, C_DST],
                                         strides=[C_DST, 1],
                                         block_shape=[BLOCK_M, C_SRC])
    for blk in tl.range(0, N_BLOCKS, 1):
        v = src_desc.load([blk * BLOCK_M, 0])
        dst_desc.store([blk * BLOCK_M, COL_OFF], v)
    if TAIL_H > 0:
        t_src = tl.make_tensor_descriptor(desc_src, shape=[R, C_SRC],
                                          strides=[C_SRC, 1],
                                          block_shape=[TAIL_H, C_SRC])
        t_dst = tl.make_tensor_descriptor(desc_dst, shape=[R, C_DST],
                                          strides=[C_DST, 1],
                                          block_shape=[TAIL_H, C_SRC])
        v = t_src.load([N_BLOCKS * BLOCK_M, 0])
        t_dst.store([N_BLOCKS * BLOCK_M, COL_OFF], v)
