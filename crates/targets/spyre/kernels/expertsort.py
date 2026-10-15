# SPDX-License-Identifier: Apache-2.0
"""THE EXPERT-SORT SPLICE'S KERNEL — the MoE dispatch sort's body: the `[m, w]`
expert rows replicated into the `[m, k·w]` (token, slot) pair layout, slot `j`'s
block at column `j·w`.

BODY PROVENANCE: main's `KtirFunc::expert_sort` — the whole `[m, w]` tile loads
once and stores `k` times (the LX holds `w` elements per block row and the copy is
pointwise, so one load feeding `k` stores is the natural blocking for a copy whose
source is one tile). No arithmetic; `k` stores of the same value.

THE SPLICE'S OWN CONTRACT (what `scratchy-triton-splice` states about this kernel):

* PARAMETERS, IN ORDER: `desc_x`, `desc_o` — ⭐ INPUT 0 AND THE OUTPUT ONLY. The
  node carries `[x, indices]`, but the sort's index tensor rides along in the
  GRAPH (the door's `split_out(..., 1)` and its identity legs never read it), so
  the program binds what the body touches: main's `KtirFunc::expert_sort` took
  `(x_r, out_r, k)` and never touched the indices either.
* CONSTEXPRS: `M`, `W`, `K` — stated by the splice from the node's own regions
  (`M =` rows, `W =` the x region's width, `K =` the output width over `W`, which
  the producer refuses unless it is a whole `k`).
* GRID: `[1]`.

THE DOOR'S OWN FACTS (`ktir_superdsc::expert_sort`): `w =` the x view's
`v_cols`, `full =` the output view's `v_cols`, `k = full / w` — refused unless
whole — so the descriptors state the STORAGE extents exactly: x `[M, W]`, out
`[M, K·W]`, while the out BLOCK is one slot's `[M, W]` window at the slot's own
column corner. The door emits the `k` identity legs itself at `col_of(j·w)`; this
body is what the EMULATOR runs.
"""

import triton
import triton.language as tl


@triton.jit
def expertsort_fwd(desc_x, desc_o,  #
                   M: tl.constexpr, W: tl.constexpr, K: tl.constexpr,  #
                   ):
    x_desc = tl.make_tensor_descriptor(desc_x, shape=[M, W], strides=[W, 1],
                                       block_shape=[M, W])
    o_desc = tl.make_tensor_descriptor(desc_o, shape=[M, K * W], strides=[K * W, 1],
                                       block_shape=[M, W])
    # ONE LOAD, K STORES — slot `j`'s block is the same `[m, w]` tile again, at
    # column `j·w`. The trip count is constexpr, so `to_ktir` unrolls it and each
    # store corner is a constant the door reads.
    x = x_desc.load([0, 0])
    for j in tl.range(0, K, 1):
        o_desc.store([0, j * W], x)
