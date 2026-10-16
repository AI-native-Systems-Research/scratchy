# SPDX-License-Identifier: Apache-2.0
"""THE ROUTE-RENORM SPLICE'S KERNEL — mixtral's router renorm over `[m, e]`:
`x / rowsum(x)` — the softmax chain minus the stability subtract and the exp.

BODY PROVENANCE: main's `KtirFunc::route_renorm` — the row sum, the broadcast
back over the row, the terminal divide. The f32 island matches main's all-f32
chain; LegalizeTypes collapses it as it collapses every stability-widening
island.

THE SPLICE'S OWN CONTRACT (what `scratchy-triton-splice` states about this kernel):

* PARAMETERS, IN ORDER: `desc_x`, `desc_o` — the node's one input, then the
  output.
* CONSTEXPRS: `M`, `N` — stated by the splice from the node's own region.
* GRID: `[1]`.

THE DOOR'S OWN FACTS (`ktir_superdsc::route_renorm`): like the softmax's, it
reads its width off the output view's own `v_cols` and states its own op chain,
so the kernel states one whole-region `[M, N]` tile at corner 0 on both sides.
"""

import triton
import triton.language as tl


@triton.jit
def routerenorm_fwd(desc_x, desc_o,  #
                    M: tl.constexpr, N: tl.constexpr,  #
                    ):
    x_desc = tl.make_tensor_descriptor(desc_x, shape=[M, N], strides=[N, 1],
                                        block_shape=[M, N])
    o_desc = tl.make_tensor_descriptor(desc_o, shape=[M, N], strides=[N, 1],
                                        block_shape=[M, N])
    x = x_desc.load([0, 0]).to(tl.float32)
    s = tl.sum(x, 1)
    # `tl.fdiv`, not `/`: `/` upcasts f16→f32 (the silumul delta-6 law).
    y = tl.fdiv(x, s[:, None])
    o_desc.store([0, 0], y.to(tl.float16))
