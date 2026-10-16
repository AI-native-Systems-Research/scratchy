# SPDX-License-Identifier: Apache-2.0
"""THE EXPERT-UNSORT SPLICE'S KERNEL — the identity copy out of the (token, slot)
pair layout: all `m·k·w` elements moved once, no arithmetic, no permutation.

BODY PROVENANCE: main's `KtirFunc::expert_unsort` — `load_region` then
`store_region`, nothing else. Metal's gathered bake emits nothing here either
(`S::Unsort => vec![]`) — the combine reads the pair rows where they lie — so this
materializes the same non-move.

THE SPLICE'S OWN CONTRACT (what `scratchy-triton-splice` states about this kernel):

* PARAMETERS, IN ORDER: `desc_rows`, `desc_o` — the node's one input, then the
  output. The registry does not permute.
* CONSTEXPRS: `M`, `COLS` — stated by the splice from the node's own regions
  (`M =` rows, `COLS =` the region's width, which the producer refuses unless it
  equals both tensors' storage width).
* GRID: `[1]`.

THE DOOR'S OWN FACTS (`ktir_superdsc::expert_unsort`): `cols = out.c_len` — the
FIRST access tile's width — and it emits ONE identity, so the kernel states one
whole-region tile at corner 0 on both sides.
"""

import triton
import triton.language as tl


@triton.jit
def expertunsort_fwd(desc_rows, desc_o,  #
                     M: tl.constexpr, COLS: tl.constexpr,  #
                     ):
    rows_desc = tl.make_tensor_descriptor(desc_rows, shape=[M, COLS], strides=[COLS, 1],
                                           block_shape=[M, COLS])
    o_desc = tl.make_tensor_descriptor(desc_o, shape=[M, COLS], strides=[COLS, 1],
                                        block_shape=[M, COLS])
    x = rows_desc.load([0, 0])
    o_desc.store([0, 0], x)
