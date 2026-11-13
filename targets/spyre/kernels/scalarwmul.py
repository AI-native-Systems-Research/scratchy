# SPDX-License-Identifier: Apache-2.0
"""THE SCALAR-WEIGHT-MUL SPLICE'S KERNEL — `out = x · w` with `w` the ONE loaded
weight: the routed expert's shared output scale, a `[1]` tensor sprayed over
every row.

BODY PROVENANCE: main's `KtirFunc::scalar_weight_mul` — the `[1]` weight
sprayed over every row, one multiply. The weight crosses as a RUNTIME fp16
SCALAR argument (rung 3 of the address-provenance ladder, `mul_scale.py`'s own
shape): the value is staged from the checkpoint, so it is not a constexpr, and
it is not a DESCRIPTOR either — a `[1]`-region load would have to broadcast
its `[1, 1]`-widened tile over `[M, N]`, an all-unit source `to_ktir` refuses
(it would need a rank-0 source), which is exactly the wall the raw-graph
wiring walk hit at prefill `m > 1`. A rank-0 argument needs no broadcast at
all: the frontend SPLATS it (`broadcast_impl_value`), and the splat re-points
at the `[1, 1]` load the rung-3 rewrite mints over the argument.

THE SPLICE'S OWN CONTRACT (what `scratchy-triton-splice` states about this kernel):

* PARAMETERS, IN ORDER: `desc_x`, `w`, `desc_o` — the node's operand order
  (x, weight) then the output; `w` is a scalar (`fp16`), the base address of
  the one-element buffer the launch binds.
* CONSTEXPRS: `M`, `N` — stated by the splice from the node's own region.
* GRID: `[1]`.

THE DOOR'S OWN FACTS (`ktir_superdsc::scalarweightmul`): it reads the operands'
NAMES (the weight is the second, broadcast `In::scalar`-style), so the rung-3
`[1, 1]` view is the weight's whole stated storage — the door's one-stick `rb`
read, the card's lane-0 spray, and the emulator's splat (which takes the load
tile's FIRST flat element) all read the one staged value.
"""

import triton
import triton.language as tl


@triton.jit
def scalarwmul_fwd(desc_x, w, desc_o,  #
                   M: tl.constexpr, N: tl.constexpr,  #
                   ):
    x_desc = tl.make_tensor_descriptor(desc_x, shape=[M, N], strides=[N, 1],
                                        block_shape=[M, N])
    o_desc = tl.make_tensor_descriptor(desc_o, shape=[M, N], strides=[N, 1],
                                        block_shape=[M, N])
    x = x_desc.load([0, 0])
    o_desc.store([0, 0], x * w)
