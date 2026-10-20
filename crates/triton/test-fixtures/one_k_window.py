# SPDX-License-Identifier: Apache-2.0
"""THE ONE-K MULF AT A COLUMN WINDOW — attn.py's decode value leg, in miniature.

The decode arm's weighted-V product is `o_d = tl.dot(w_d, vd)` with `w_d` the
`[1, 1]` per-row softmax weight and `vd` head h's `[1, HD]` window at column
`kvh·HD` of the `[NEW_LEN, NKVH·HD]` V buffer. `K = 1` is not a multiple of the
64-fp16 stick, so the ladder's `one_k_dot_to_mul` rewrites the dot into
`broadcast(w_d) · vd` — a Col-broadcast operand beside a DENSE operand at a
column corner. That is the shape this fixture exists to hold: the whole-function
walk recognises the broadcast chain, and the elementwise door's broadcast arm
places the dense operand (and the output) at their own windows — the dense
arm's per-operand forms, not the buffer base.

THE SHAPES (granite-3.1-2b decode, GQA=4, the first head whose V window leaves
column 0 — kvh=1):

* `desc_q` `[1, HD]`, `desc_k` `[HD, 1]` — the new-token score leg, plain-B
  against the transposed staging form (the restickify scratch's shape class).
  Their dot is `[1, 1]`, the K=1 mulf's A operand once the ladder widens it.
* `desc_w` `[1, SWEPT]`, `desc_vc` `[SWEPT, HD]` — the PREFIX value leg
  (`o_c = tl.dot(w_c, vc)`, `K = SWEPT`). It stands in for the softmax weight
  chain as a plain load, and it keeps the kernel MULTI-DOT: `one_k_dot_to_mul`
  rewrites only the K=1 dot, and a kernel whose every other dot is gone is the
  SINGLE-dot shape `dot_to_linalg`'s canonical verifier refuses ("only one
  matmul feeding one store") — the real attention always keeps its score and
  prefix dots beside the value legs, so the fixture does too.
* `desc_v` `[1, V_COLS]` with a `[1, HD]` block loaded at column `HD` — the
  kvh=1 V window. `V_COLS = NKVH·HD = 512`.
* `desc_o` the store target, the same window — the epilogue addf
  (`o_desc.store([0, h*HD], o + o_d)`) takes two minted corner-0 intermediates
  and writes the window: DENSE elementwise with per-operand corners, the
  card-proven rotate-half class.

Everything the score leg needs is real: the `[HD, 1]` block is the ktn
scratch's decode block (`[HD, NEW_LEN]` at `NEW_LEN=1`), and the `[1, 1]` dot
result is the class every swept decode head's `qk_d` takes.
"""

import triton
import triton.language as tl


@triton.jit
def one_k_window(desc_q, desc_k, desc_w, desc_vc, desc_v, desc_o,  #
                 HD: tl.constexpr, V_COLS: tl.constexpr, SWEPT: tl.constexpr):
    q_desc = tl.make_tensor_descriptor(desc_q, shape=[1, HD],
                                       strides=[HD, 1],
                                       block_shape=[1, HD])
    k_desc = tl.make_tensor_descriptor(desc_k, shape=[HD, 1],
                                       strides=[1, 1],
                                       block_shape=[HD, 1])
    w_desc = tl.make_tensor_descriptor(desc_w, shape=[1, SWEPT],
                                       strides=[SWEPT, 1],
                                       block_shape=[1, SWEPT])
    vc_desc = tl.make_tensor_descriptor(desc_vc, shape=[SWEPT, HD],
                                        strides=[HD, 1],
                                        block_shape=[SWEPT, HD])
    v_desc = tl.make_tensor_descriptor(desc_v, shape=[1, V_COLS],
                                       strides=[V_COLS, 1],
                                       block_shape=[1, HD])
    o_desc = tl.make_tensor_descriptor(desc_o, shape=[1, V_COLS],
                                       strides=[V_COLS, 1],
                                       block_shape=[1, HD])
    q = q_desc.load([0, 0])  # [1, HD]
    k = k_desc.load([0, 0])  # [HD, 1] — the new block's Kᵀ, plain-B
    s = tl.dot(q, k, out_dtype=tl.float16)  # [1, 1] — the new-token score
    w = w_desc.load([0, 0])  # [1, SWEPT] — the prefix weights
    vc = vc_desc.load([0, 0])  # [SWEPT, HD] — the prefix V rows
    o_c = tl.dot(w, vc, out_dtype=tl.float16)  # [1, HD] — K=SWEPT, kept a dot
    v = v_desc.load([0, HD])  # [1, HD] — kvh=1's V window at column HD
    o_d = tl.dot(s, v, out_dtype=tl.float16)  # K=1 → broadcast(s) · v
    o_desc.store([0, HD], o_c + o_d)
