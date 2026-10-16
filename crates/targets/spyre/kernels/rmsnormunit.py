# SPDX-License-Identifier: Apache-2.0
"""THE RMSNORM-UNIT SPLICE'S KERNEL — the UN-GAINED rmsnorm
`out = x / rms(x)`: `rmsnorm.py`'s own chain minus the gamma multiply, the form
the head-norm sandwich's un-fused fallback runs (and deep-norm inner units).

BODY PROVENANCE: main's `KtirFunc::rmsnorm_unit` — the mean-of-squares
reduce, the rsqrt, the broadcast multiply; no gain tensor, no offset. The
reduce is stated PLAIN for the same reason `rmsnorm.py` states its own
(the module doc there is the law: the reduce is f16-typed — no widening
survives LegalizeTypes — but its ACCUMULATOR is the card's DLF16, which
the emulator models, so no pre-scale is needed or allowed).

THE SPLICE'S OWN CONTRACT (what `scratchy-triton-splice` states about this kernel):

* PARAMETERS, IN ORDER: `desc_x`, `desc_o` — the node's one input, then the
  output.
* CONSTEXPRS: `M`, `D_MODEL`, `BLOCK_M`, `EPS`, `INV_D` — stated by the splice
  from the node's own region and the tape's epsilon (`INV_D = 1/D_MODEL` folded
  on the host), exactly `rmsnorm.py`'s set minus the weight.
* GRID: `[1]`.

THE DOOR'S OWN FACTS (`ktir_superdsc::rmsnorm_unit`): `cols = out.c_len` and the
epsilon off the program (`program_rmsnorm_eps`: ONE sqrt/rsqrt root feeding the
addf that carries the eps splat), so the kernel states one whole-region
`[M, D_MODEL]` tile at corner 0 and the chain the reader walks.
"""

import triton
import triton.language as tl


@triton.jit
def rmsnormunit_fwd(desc_x, desc_o,  #
                    M: tl.constexpr, D_MODEL: tl.constexpr,  #
                    BLOCK_M: tl.constexpr,  #
                    EPS: tl.constexpr,  #
                    INV_D: tl.constexpr,  #
                    ):
    start_m = tl.program_id(0)
    x_desc = tl.make_tensor_descriptor(desc_x, shape=[M, D_MODEL],
                                       strides=[D_MODEL, 1],
                                       block_shape=[BLOCK_M, D_MODEL])
    o_desc = tl.make_tensor_descriptor(desc_o, shape=[M, D_MODEL],
                                        strides=[D_MODEL, 1],
                                        block_shape=[BLOCK_M, D_MODEL])
    offs_m = start_m * BLOCK_M
    x = x_desc.load([offs_m, 0])
    # EXX2_ZEROMEAN, no mean subtracted — rmsnorm.py's own plain reduce (its module
    # doc's law); no gain follows it.
    ms = tl.sum(x * x, 1) * INV_D
    r = tl.rsqrt((ms + EPS).to(tl.float32)).to(tl.float16)
    o_desc.store([offs_m, 0], x * r[:, None])
