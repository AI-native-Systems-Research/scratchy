# SPDX-License-Identifier: Apache-2.0
"""Scratchy's Triton silu-mul kernel — the SECOND SPLICED KERNEL.

BODY PROVENANCE: derived from `crates/triton/test-fixtures/swiglu_mlp.py`'s activation
body (deltas 5/6/8 there) — the same four-op chain (`e = exp(-g)` → `s = g / (1+e)` →
`h = s * u`) that fixture card-validated through this repo's KTIR→SuperDSC lowering,
stated standalone over whole-region tiles the way the builder's own `KtirFunc::silu_mul`
states them. It is NOT one tl.dot of that fixture: `SubOp::SiluMul` is the activation
alone, with the projections' matmuls their own nodes.

THE SPLICE'S OWN CONTRACT (what `scratchy-triton-splice` states about this kernel):

* PARAMETERS, IN ORDER: `desc_g`, `desc_u`, `desc_o` — the node's operand order
  (gate, up) then the output. The registry does not permute.
* CONSTEXPRS: `M`, `N`, `BLOCK_M`, `BLOCK_N` — stated by the splice from the node's
  own region (`M = BLOCK_M =` the region's row count, `N = BLOCK_N =` its width).
* GRID: `[1]`.

⛔ NO ROW BLOCKING, deliberately — the same law `KtirFunc::silu_mul`'s own comment
records (the builder blocks only when a whole `[mq, intermediate]` region would not fit
a core's LX at prefill; the kernel loads the full `[M, N]` and the layout pass owns
when that must break). `swiglu_mlp.py`'s BLOCK_N=64 stick-width note does NOT apply
here: that is a `tl.dot` tile constraint, and this kernel has no `tl.dot`.
"""

import triton
import triton.language as tl


@triton.jit
def silumul_fwd(desc_g, desc_u, desc_o,  #
                M: tl.constexpr, N: tl.constexpr,  #
                BLOCK_M: tl.constexpr,  #
                BLOCK_N: tl.constexpr,  #
                ):
    start_m = tl.program_id(0)
    g_desc = tl.make_tensor_descriptor(desc_g, shape=[M, N], strides=[N, 1],
                                       block_shape=[BLOCK_M, BLOCK_N])
    u_desc = tl.make_tensor_descriptor(desc_u, shape=[M, N], strides=[N, 1],
                                       block_shape=[BLOCK_M, BLOCK_N])
    o_desc = tl.make_tensor_descriptor(desc_o, shape=[M, N], strides=[N, 1],
                                       block_shape=[BLOCK_M, BLOCK_N])

    offs_m = start_m * BLOCK_M
    g = g_desc.load([offs_m, 0])
    u = u_desc.load([offs_m, 0])
    # silu(g) = g / (1 + exp(-g)) — the same chain `KtirFunc::silu_mul` writes.
    # `tl.sigmoid` refuses an f16 tensor outright (swiglu delta 5, `math.exp` is
    # @_check_dtype(["fp32","fp64"])), and the ladder's LegalizeTypes collapses the
    # widen/truncate island this spells, so the emitted KTIR carries `math.exp` on
    # an f16 tile with no extf/truncf — the exact four ops the builder emits.
    e = tl.exp((-g).to(tl.float32)).to(tl.float16)
    s = tl.fdiv(g, 1.0 + e)  # tl.fdiv, not `/`: `/` upcasts f16→f32 (swiglu delta 6)
    o_desc.store([offs_m, 0], s * u)
