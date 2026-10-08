# SPDX-License-Identifier: Apache-2.0
"""Scratchy's Triton last-row-extraction kernel — the prefill lm-head fold's first half.

`lmlast_fwd` copies the LAST prompt token's hidden row from the final-norm output
`[MQ, HIDDEN]` into the `[1, HIDDEN]` reserved staging `LAST_HIDDEN_TID`, so the
vocab-wide lm_head matmul re-lowers at m=1 over a tensor whose row 0 is the last
prompt token — `mq`-times less work for the one row of logits the first generated
token reads.

BODY PROVENANCE: main's `KtirFunc::last_row_extract` (the builder control
`lower_prefill_lm_head_at_m1`'s own extraction body, from main), stated as a Triton kernel —
`hidden/64` single-stick copies, one `[1, 64]` tile per stick-group. That per-stick
form is not a style choice: the source buffer is DEVICE-TILED `[hidden/64, mq, 64]`
(stick-major), so row `mq-1`'s stick-group `j` is 64 CONTIGUOUS elements, and no
wider window over the row is representable (`selector_lastrow_col`'s doc: a strided
read cannot be expressed at fp16; a one-hot selector needs `k = mq` to be a whole
stick, which it never is).

THE SPLICE'S OWN CONTRACT (what `scratchy-triton-splice` states about this kernel):

* PARAMETERS, IN ORDER: `desc_src`, `desc_dst` — the final-norm output, then the
  reserved staging. The registry does not permute.
* CONSTEXPRS: `MQ`, `HIDDEN`, `ROW`, `N_STICKS`. `ROW = selector_lastrow_col(mq) =
  mq - 1`, stated by the splice from the node's own activation region — the SINGLE
  SOURCE OF TRUTH the Kani proof `selector_lastrow_picks_last_row` pins.
* GRID: `[1]`.

⛔ THE SOURCE VIEW IS THE BUFFER ITSELF — `[MQ, HIDDEN]`, strides `[HIDDEN, 1]`. The
door's `lmlast` arm reads `mq` and `hidden` off this view (`mq` is a TERM of the copy
address: row `mq-1`'s stick-group `j` sits at `j·mq + row` in the stick plane), so a
view narrowed to the row would mis-address every copy.

⛔ EVERY WINDOW IS ONE SINGLE STICK — `[1, 64]`. The door's `lmlast` arm checks
exactly this (`r_len == 1 && c_len == stk` on both operands' first tiles): each copy
is one stick-group, the only form a row offset is representable in.

⛔ THE DESTINATION VIEW IS `[1, HIDDEN]` — the reserved staging's own extent, whose
window is `[1, 64]` at column `j·64` per copy. The destination buffer must be
whole-row even though each window is one stick, and `regions()` reads the buffer off
the view, the window off the first tile.
"""

import triton
import triton.language as tl


@triton.jit
def lmlast_fwd(desc_src, desc_dst,  #
               MQ: tl.constexpr,  #
               HIDDEN: tl.constexpr,  #
               ROW: tl.constexpr,  #
               N_STICKS: tl.constexpr,  #
               ):
    # THE SOURCE AS THE BUFFER IT IS — `[mq, hidden]`. The door reads `mq` and
    # `hidden` off this view.
    src_desc = tl.make_tensor_descriptor(desc_src, shape=[MQ, HIDDEN],
                                         strides=[HIDDEN, 1],
                                         block_shape=[1, 64])
    # THE RESERVED STAGING — `[1, hidden]`, whole-row.
    dst_desc = tl.make_tensor_descriptor(desc_dst, shape=[1, HIDDEN],
                                         strides=[HIDDEN, 1],
                                         block_shape=[1, 64])

    # ONE SINGLE-STICK COPY PER STICK-GROUP — the trip count is constexpr, so
    # `to_ktir` unrolls it and the program is straight-line with one `[1, 64]` tile
    # per stick, the form the door's `lmlast` arm checks.
    for j in tl.range(0, N_STICKS, 1):
        v = src_desc.load([ROW, j * 64])
        dst_desc.store([0, j * 64], v)
