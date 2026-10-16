# SPDX-License-Identifier: Apache-2.0
"""THE EXPERT-COMBINE SPLICE'S KERNEL — the MoE output combine:
`out[n, d] = Σ_k rows[n, k·w + d] · scores[n, k]`, the weighted sum metal's
`moe_weighted_sum` computes.

BODY PROVENANCE: main's `KtirFunc::expert_combine` — each slot `j`'s `[m, w]`
block of the pair rows times its OWN score column, accumulated across slots with
one narrowing at the very end. The score column is loaded as a `[m, 1]` tile at
column `j` (main's `TensorExtractSlice` of the `[m, k]` scores, restated as a
one-column descriptor window — the door's own `In::col_at` read is the same
column), and the `[m, 1]` broadcasts across the block's width exactly as main's
`LinalgBroadcast` sprayed the sliced column.

THE SPLICE'S OWN CONTRACT (what `scratchy-triton-splice` states about this kernel):

* PARAMETERS, IN ORDER: `desc_rows`, `desc_scores`, `desc_o` — the node's operand
  order (pair rows, scores) then the output.
* CONSTEXPRS: `M`, `W`, `K` — stated by the splice from the node's own regions
  (`M =` rows, `W =` the output width, `K =` the scores width; the producer
  refuses unless the rows width is exactly `K·W`).
* GRID: `[1]`.

THE DOOR'S OWN FACTS (`ktir_superdsc::expert_combine`): it resolves the two
operands BY EXTENT — the scores' `v_cols` is `k`, the rows' `v_cols` is `k·w` —
so the descriptors state the STORAGE extents exactly: rows `[M, K·W]` with a
one-slot `[M, W]` block, scores `[M, K]` with a ONE-COLUMN `[M, 1]` block, out
`[M, W]` whole. The door emits the `2k−1` op chain itself (mul, then mul+add per
slot); this body is what the EMULATOR runs, and it is the same `2k−1` shape.

⛔ THE `[M, 1]` SCORE BLOCK IS LEGAL WHERE A GATHER'S IS NOT — the front end's
16-byte last-dim floor is `descriptor_gather`'s own assert (`semantic.py:1143`'s
five guards); `descriptor_load` has none, which is also how `rmsnorm.py` loads
its 1-D weight.
"""

import triton
import triton.language as tl


@triton.jit
def expertcombine_fwd(desc_rows, desc_scores, desc_o,  #
                      M: tl.constexpr, W: tl.constexpr, K: tl.constexpr,  #
                      ):
    rows_desc = tl.make_tensor_descriptor(desc_rows, shape=[M, K * W], strides=[K * W, 1],
                                           block_shape=[M, W])
    # The scores `[m, k]`: one column per slot, loaded as its own `[m, 1]` window
    # at the slot's column corner.
    scores_desc = tl.make_tensor_descriptor(desc_scores, shape=[M, K], strides=[K, 1],
                                             block_shape=[M, 1])
    o_desc = tl.make_tensor_descriptor(desc_o, shape=[M, W], strides=[W, 1],
                                        block_shape=[M, W])
    # Slot 0's product seeds the accumulator; every later slot is a product leg
    # then an add leg — `2k−1` ops, the chain the door emits and the emu runs.
    # The f32 island matches main's f32-accumulated `acc` (one narrowing at the
    # end); LegalizeTypes collapses it to the device's f16 the way it collapses
    # every stability-widening island.
    b0 = rows_desc.load([0, 0]).to(tl.float32)
    s0 = scores_desc.load([0, 0]).to(tl.float32)
    acc = b0 * s0
    for j in tl.range(1, K, 1):
        b = rows_desc.load([0, j * W]).to(tl.float32)
        s = scores_desc.load([0, j]).to(tl.float32)
        acc = acc + b * s
    o_desc.store([0, 0], acc.to(tl.float16))
