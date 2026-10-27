# SPDX-License-Identifier: Apache-2.0
"""THE HEAD RELAYOUT — the copy between the token stream and the attention
kernels' per-head row frame, at every head dim.

A `[R, H*HD]` activation (the rope's q, the o projection's input) is
stick-blocked: element `(r, c)` sits at `(c/STK)*R*STK + r*STK + c%STK`, so head
`h` slab `s` row `r` is stick `(h*SLABS + s)*R + r`. The attention kernels read
q and write their output through a `[H*R, HD]` view (row `h*R + r`), whose
stick-blocked bytes put the same element at stick `s*H*R + h*R + r`. The two
frames are the SAME bytes exactly when `SLABS == 1` (one stick per head) — the
head dim 64 models — and a permutation of whole `[R, STK]` sticks blocks at
every wider head dim (128, 256, 512). These two kernels are that permutation:
one `[R, STK]` copy per (head, slab), each side addressed at its own view's
corner, so the bytes the attention kernel reads are the bytes its view states.

THE SPLICE'S OWN CONTRACT:

* PARAMETERS, IN ORDER: `desc_src`, `desc_dst`.
* CONSTEXPRS: `R` (rows per head — the chunk's query rows, or the batch's
  requests), `H` (heads), `HD` (head dim), `STK` (the stick).
* GRID: `[1]`. The trip counts are constexpr, so `to_ktir` unrolls both loops
  into straight-line load/store pairs at constant corners.
"""

import triton
import triton.language as tl


@triton.jit
def to_head_rows_fwd(desc_src, desc_dst,  #
                     R: tl.constexpr, H: tl.constexpr,  #
                     HD: tl.constexpr, STK: tl.constexpr):
    """`[R, H*HD]` token stream → `[H*R, HD]` per-head rows."""
    tl.static_assert(HD % STK == 0, "head_relayout: the head dim is whole sticks")
    SLABS: tl.constexpr = HD // STK
    src = tl.make_tensor_descriptor(desc_src, shape=[R, H * HD], strides=[H * HD, 1],
                                    block_shape=[R, STK])
    dst = tl.make_tensor_descriptor(desc_dst, shape=[H * R, HD], strides=[HD, 1],
                                    block_shape=[R, STK])
    for h in tl.static_range(H):
        for s in tl.static_range(SLABS):
            dst.store([h * R, s * STK], src.load([0, h * HD + s * STK]))


@triton.jit
def from_head_rows_fwd(desc_src, desc_dst,  #
                       R: tl.constexpr, H: tl.constexpr,  #
                       HD: tl.constexpr, STK: tl.constexpr):
    """`[H*R, HD]` per-head rows → `[R, H*HD]` token stream."""
    tl.static_assert(HD % STK == 0, "head_relayout: the head dim is whole sticks")
    SLABS: tl.constexpr = HD // STK
    src = tl.make_tensor_descriptor(desc_src, shape=[H * R, HD], strides=[HD, 1],
                                    block_shape=[R, STK])
    dst = tl.make_tensor_descriptor(desc_dst, shape=[R, H * HD], strides=[H * HD, 1],
                                    block_shape=[R, STK])
    for h in tl.static_range(H):
        for s in tl.static_range(SLABS):
            dst.store([0, h * HD + s * STK], src.load([h * R, s * STK]))
