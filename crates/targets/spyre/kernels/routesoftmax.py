# SPDX-License-Identifier: Apache-2.0
"""THE ROUTE-SOFTMAX SPLICE'S KERNEL — the MoE router's row softmax over
`[m, e]`, written longhand as the stability chain
`exp(x − rowmax) / rowsum(exp(x − rowmax))`.

BODY PROVENANCE: main's `KtirFunc::route_softmax` — the structure is
`rmsnorm_unit`'s (a per-row reduce, a broadcast back over the row, a terminal
divide) with `max`/`exp`/`sum`/`divf` in place of the norm's primitives, because
a row softmax and a row rms are the same shape of computation: one scalar per
row, sprayed back over it. The f32 island matches main's all-f32 chain
(`f32_splat`/`f32_binop`); LegalizeTypes collapses it as it collapses every
stability-widening island.

THE SPLICE'S OWN CONTRACT (what `scratchy-triton-splice` states about this kernel):

* PARAMETERS, IN ORDER: `desc_x`, `desc_o` — the node's one input, then the
  output.
* CONSTEXPRS: `M`, `N` — stated by the splice from the node's own region
  (`M =` rows, `N =` the router's expert count, the region's width).
* GRID: `[1]`.

THE DOOR'S OWN FACTS (`ktir_superdsc::route_softmax`): it reads `k` off the
output view's own `v_cols` — the LOGICAL expert count, not a padded stick — and
pads the device width itself, so the kernel states one whole-region `[M, N]`
tile at corner 0 on both sides.
"""

import triton
import triton.language as tl


@triton.jit
def routesoftmax_fwd(desc_x, desc_o,  #
                     M: tl.constexpr, N: tl.constexpr,  #
                     ):
    x_desc = tl.make_tensor_descriptor(desc_x, shape=[M, N], strides=[N, 1],
                                        block_shape=[M, N])
    o_desc = tl.make_tensor_descriptor(desc_o, shape=[M, N], strides=[N, 1],
                                        block_shape=[M, N])
    x = x_desc.load([0, 0]).to(tl.float32)
    # 1. rowmax(x) — the stability subtractend, `[m]`.
    rowmax = tl.max(x, 1)
    # 2. exp(x − rowmax). ONLY the `[m]` rowmax takes the `[:, None]` broadcast —
    #    spraying each row's scalar back over its columns; `x` is already `[m, n]`
    #    full-rank (main's broadcast-axis law).
    e = tl.exp(x - rowmax[:, None])
    # 3. rowsum(exp) — the denominator, `[m]`.
    s = tl.sum(e, 1)
    # 4. exp / rowsum, the denominator broadcast back over the row. `tl.fdiv`,
    #    not `/`: `/` upcasts f16→f32 (the silumul delta-6 law).
    y = tl.fdiv(e, s[:, None])
    o_desc.store([0, 0], y.to(tl.float16))
