# SPDX-License-Identifier: Apache-2.0
"""Triton: the chained head's input rows — each request row's token embedding read on the card out
of the tied lm-head weight, at the token the previous step drew, so the token never leaves the card.

The tied weight is the lm-head's own operand as it is held in memory: the presented `[PAD, HIDDEN]`
weight, token `t`'s embedding its row `t` (the matmul reads it transposed). The previous step's
chain stage decomposed the pick at the 64-row slab, `t = 64 * s + l`: row `r` of the s stick
carries the slab `s` (word 0 is read), row `r` of the l stick is the lane's one-hot. The row is the
slab's rows weighted by the one-hot,

    out[r, :] = sum_j W[64 * s + j, :] * onehot[j] = W[t, :],

one nonzero term, so the sum is exact — stated as the one-hot times the slab, a
`[1, LANES] x [LANES, HIDDEN]` product. The zero one-hot — a pick past the pad (the sampler's
sentinel) or outside this rank's block of a split table — gathers a zero row, read inside the
matrix. The slab is the unit the weight's held layout keeps contiguous (a stick column of the
lm-head's kernel layout), so the read is one block per row however the weight is held.

Parameters:
* `s_ptr` `[B, IDX]` int32 — each row's slab in word 0;
* `l_ptr` `[B, LANES]` fp16 — each row's lane one-hot;
* `w_ptr` `[PAD, HIDDEN]` fp16 — the tied weight, the vocabulary widened to the pick's pad (one
  rank's block of it when the projection is split);
* `out_ptr` `[B, HIDDEN]` fp16.

One program walks the request rows (the slab id is a scalar read, at an offset of the row loop's
own induction variable).
"""

import triton
import triton.language as tl


@triton.jit
def embed_row_gather_fwd(s_ptr, l_ptr, w_ptr, out_ptr,  #
                         HIDDEN: tl.constexpr, PAD: tl.constexpr, LANES: tl.constexpr,  #
                         IDX: tl.constexpr, B: tl.constexpr):
    w = tl.make_tensor_descriptor(w_ptr, shape=[PAD, HIDDEN], strides=[HIDDEN, 1],
                                  block_shape=[LANES, HIDDEN])
    l_desc = tl.make_tensor_descriptor(l_ptr, shape=[B, LANES], strides=[LANES, 1],
                                       block_shape=[1, LANES])
    o_desc = tl.make_tensor_descriptor(out_ptr, shape=[B, HIDDEN], strides=[HIDDEN, 1],
                                       block_shape=[1, HIDDEN])
    for r in range(B):
        slab = tl.load(s_ptr + r * IDX)
        rows = w.load([slab * LANES, 0])
        onehot = l_desc.load([r, 0])
        row = tl.dot(tl.reshape(onehot, [1, 1, LANES]), tl.reshape(rows, [1, LANES, HIDDEN]),
                     out_dtype=tl.float16)
        o_desc.store([r, 0], tl.reshape(row, [1, HIDDEN]))
