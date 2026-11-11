# SPDX-License-Identifier: Apache-2.0
"""Scratchy's Triton last-row-extraction kernel — the prefill lm-head fold's first half.

`lmlast_fwd` copies each request window's LAST prompt-token hidden row from the
final-norm output `[MQ, HIDDEN]` into the `[S, HIDDEN]` reserved staging
`LAST_HIDDEN_TID` — request `r`'s answer row `r·W + W-1`, one row per window of the
rung's `(W, S)` packing — so the vocab-wide lm_head matmul re-lowers at m=S over a
tensor whose row `r` is request `r`'s last prompt token: `mq`-times less work for the
`S` rows of logits the first generated token of each request reads.

BODY PROVENANCE: main's `KtirFunc::last_row_extract` (the builder control
`lower_prefill_lm_head_at_m1`'s own extraction body, from main), stated as a Triton kernel —
`S · hidden/64` single-stick copies, one `[1, 64]` tile per stick-group. That per-stick
form is not a style choice: the source buffer is DEVICE-TILED `[hidden/64, mq, 64]`
(stick-major), so row `r·W + W-1`'s stick-group `j` is 64 CONTIGUOUS elements, and no
wider window over the row is representable (`selector_lastrow_col`'s doc: a strided
read cannot be expressed at fp16; a one-hot selector needs `k = mq` to be a whole
stick, which it never is). At `S = 1` (`W = MQ`) the one row is
`selector_lastrow_col(mq) = mq - 1` — exactly main's single-row law, and every tile
and corner is byte-identical to it.

THE SPLICE'S OWN CONTRACT (what `scratchy-triton-splice` states about this kernel):

* PARAMETERS, IN ORDER: `desc_src`, `desc_dst` — the final-norm output, then the
  reserved staging. The registry does not permute.
* CONSTEXPRS: `MQ`, `HIDDEN`, `W`, `S`, `N_STICKS`. `W` and `S` are the rung's
  `(per-request width, request capacity)` — the same `QueryExtent` pair the attention
  reads, minted once at the bake; each request's row is `r·W + W-1`, the
  `AnswerRows` law the splice states on the node and the door reads back.
* GRID: `[1]`. Both loops are constexpr-trip `tl.range`s, unrolled by the ladder
  (`to_ktir::unroll_constant_trip_loops`), so the program is straight-line with one
  `[1, 64]` tile per `(r, j)` — the single-work-item law, no grid axis.

⛔ THE SOURCE VIEW IS THE BUFFER ITSELF — `[MQ, HIDDEN]`, strides `[HIDDEN, 1]`. The
door's `lmlast` arm reads `mq` and `hidden` off this view (`mq` is a TERM of the copy
address: row `r`'s stick-group `j` sits at `j·mq + row` in the stick plane), so a
view narrowed to the row would mis-address every copy.

⛔ EVERY WINDOW IS ONE SINGLE STICK — `[1, 64]`. The door's `lmlast` arm checks
exactly this (`r_len == 1 && c_len == stk` on both operands' first tiles): each copy
is one stick-group, the only form a row offset is representable in.

⛔ THE DESTINATION VIEW IS `[S, HIDDEN]` — the reserved staging's own extent, whose
window is `[1, 64]` at `(r, j·64)` per copy. The destination buffer must be
whole-row even though each window is one stick, and `regions()` reads the buffer off
the view, the window off the first tile.
"""

import triton
import triton.language as tl


@triton.jit
def lmlast_fwd(desc_src, desc_dst,  #
               MQ: tl.constexpr,  #
               HIDDEN: tl.constexpr,  #
               W: tl.constexpr,  #
               S: tl.constexpr,  #
               N_STICKS: tl.constexpr,  #
               ):
    # THE SOURCE AS THE BUFFER IT IS — `[mq, hidden]`. The door reads `mq` and
    # `hidden` off this view.
    src_desc = tl.make_tensor_descriptor(desc_src, shape=[MQ, HIDDEN],
                                         strides=[HIDDEN, 1],
                                         block_shape=[1, 64])
    # THE RESERVED STAGING — `[S, hidden]`, whole-row.
    dst_desc = tl.make_tensor_descriptor(desc_dst, shape=[S, HIDDEN],
                                         strides=[HIDDEN, 1],
                                         block_shape=[1, 64])

    # ONE SINGLE-STICK COPY PER (REQUEST, STICK-GROUP) — both trip counts are
    # constexpr, so `to_ktir` unrolls them and the program is straight-line with one
    # `[1, 64]` tile per `(r, j)`, the form the door's `lmlast` arm checks. Request
    # `r`'s answer row is `r·W + W-1` — its window's last BAKED row.
    for r in tl.range(0, S, 1):
        # `(W - 1)` parenthesised so the subtraction is a CONSTEXPR fold in the
        # frontend — bound to the runtime sum instead, it lowers to an
        # `arith.subi` the TTIR adapter has no form for (and should not need:
        # the row law's offset is a bake-time constant).
        row = r * W + (W - 1)
        for j in tl.range(0, N_STICKS, 1):
            v = src_desc.load([row, j * 64])
            dst_desc.store([r, j * 64], v)
