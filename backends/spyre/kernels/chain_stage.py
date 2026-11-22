# SPDX-License-Identifier: Apache-2.0
"""Triton: the chained head's stage — each request row's drawn token decomposed, on the card, into
the next step's gather indices.

The pick `t` is word 0 of the row's idx stick (the sampler's output). The decompose is two table
rows at `t`: `s_tab[t]`, the slab `t >> 6` in every word, and `l_tab[t]`, the lane's one-hot (the
zero row past the pad, and past this rank's block of a split table). They are written into the
persistent s and l sticks the next step's embedding gather reads, and the s row is the stage's own
output.

Parameters:
* `pick_ptr` `[B, PICK]` int32 — the idx stick, each row's token in word 0;
* `s_ptr` `[B, IDX]` int32, `l_ptr` `[B, LANES]` fp16 — the persistent sticks, written;
* `s_tab_ptr` `[TABLE_ROWS, IDX]` int32, `l_tab_ptr` `[TABLE_ROWS, LANES]` fp16 — the decompose,
  one row per answerable pick, the sampler's sentinel the last;
* `out_ptr` `[B, IDX]` int32.

One program walks the request rows (the pick is a scalar read, at an offset of the row loop's own
induction variable).
"""

import triton
import triton.language as tl


@triton.jit
def chain_stage_fwd(pick_ptr, s_ptr, l_ptr, s_tab_ptr, l_tab_ptr, out_ptr,  #
                    TABLE_ROWS: tl.constexpr, IDX: tl.constexpr, LANES: tl.constexpr,  #
                    PICK: tl.constexpr, B: tl.constexpr):
    s_tab = tl.make_tensor_descriptor(s_tab_ptr, shape=[TABLE_ROWS, IDX], strides=[IDX, 1],
                                      block_shape=[1, IDX])
    l_tab = tl.make_tensor_descriptor(l_tab_ptr, shape=[TABLE_ROWS, LANES], strides=[LANES, 1],
                                      block_shape=[1, LANES])
    s_desc = tl.make_tensor_descriptor(s_ptr, shape=[B, IDX], strides=[IDX, 1],
                                       block_shape=[1, IDX])
    l_desc = tl.make_tensor_descriptor(l_ptr, shape=[B, LANES], strides=[LANES, 1],
                                       block_shape=[1, LANES])
    o_desc = tl.make_tensor_descriptor(out_ptr, shape=[B, IDX], strides=[IDX, 1],
                                       block_shape=[1, IDX])
    for r in range(B):
        t = tl.load(pick_ptr + r * PICK)
        s_row = s_tab.load([t, 0])
        l_row = l_tab.load([t, 0])
        s_desc.store([r, 0], s_row)
        l_desc.store([r, 0], l_row)
        o_desc.store([r, 0], s_row)
