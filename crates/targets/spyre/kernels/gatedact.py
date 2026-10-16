# SPDX-License-Identifier: Apache-2.0
"""THE GATED-ACT SPLICE'S KERNEL — the expert MLP's gated activation
`act(gate) · up` over the pair rows: `silumul.py`'s own silu chain for the silu
act, `elementwise.py`'s own gelu chain for the gelu act — the two bodies the
door's `gated_act` arm already spans (`silu`/`gelu` OpFuncs, silumul's
two-op body with the act's primitive in place of `"silu"`).

BODY PROVENANCE: main's `KtirFunc::gated_act` — `silu_mul`'s computation with
the act the block declares. The pair rows fold into `[m, k·w]` columns, so the
computation is POINTWISE over whatever layout the sort produced — no pair
structure is read here, which is what makes this op lowerable before the
sort/unsort pair is. The gelu body is `elementwise.py`'s `gelu_fwd` VERBATIM
(the tanh approximation the DDL's `gelu` primitive itself makes, spelled
through the sigmoid identity `gelu(x) = x/(1+exp(-2v))`,
`v = 2·√(2/π)·(x + 0.044715·x³)` — the cubic in the f32 island, where an f16
`x³` would overflow at |x| ≥ 41).

THE SPLICE'S OWN CONTRACT (what `scratchy-triton-splice` states about this kernel):

* PARAMETERS, IN ORDER: `desc_gate`, `desc_up`, `desc_o` — the node's operand
  order (gate, up) then the output.
* CONSTEXPRS: `M`, `N`, `BLOCK_M`, `BLOCK_N`, `ACT`, `N_BLOCKS`, `TAIL_H` —
  stated by the splice from the node's own region (`M =` the region's row count,
  `N = BLOCK_N =` its width) and its `SubOp::ExpertGatedAct { act }` payload
  (`ACT =` 0 for silu, 1 for gelu; `if` on a constexpr folds at codegen, so a
  program carries exactly one act's chain). The front end's column chunker does
  not tile this op (its chunk set is the elementwise/scalarmul/silumul family),
  so the region is always whole and every block states corner 0.
* GRID: `[1]`.

⭐ ROW BLOCKING, THE BUILDER'S OWN LAW (main's `KtirFunc::gated_act`): eight
tiles are live here in the worst case (gate, up, the act's temporaries, the
result — `SILU_MUL_LIVE_TILES`' own count), so the kernel takes `N_BLOCKS` full
`[BLOCK_M, N]` tiles plus a `[TAIL_H, N]` tail tile. The trip count is constexpr,
so `to_ktir` unrolls it and the door folds the tiles through `Region::r_cover`
into ONE descriptor.
"""

import triton
import triton.language as tl


@triton.jit
def gatedact_fwd(desc_gate, desc_up, desc_o,  #
                 M: tl.constexpr, N: tl.constexpr,  #
                 BLOCK_M: tl.constexpr,  #
                 BLOCK_N: tl.constexpr,  #
                 ACT: tl.constexpr,  #
                 N_BLOCKS: tl.constexpr,  #
                 TAIL_H: tl.constexpr,  #
                 ):
    g_desc = tl.make_tensor_descriptor(desc_gate, shape=[M, N], strides=[N, 1],
                                       block_shape=[BLOCK_M, BLOCK_N])
    u_desc = tl.make_tensor_descriptor(desc_up, shape=[M, N], strides=[N, 1],
                                       block_shape=[BLOCK_M, BLOCK_N])
    o_desc = tl.make_tensor_descriptor(desc_o, shape=[M, N], strides=[N, 1],
                                       block_shape=[BLOCK_M, BLOCK_N])
    for blk in tl.range(0, N_BLOCKS, 1):
        offs_m = blk * BLOCK_M
        g = g_desc.load([offs_m, 0])
        u = u_desc.load([offs_m, 0])
        if ACT == 0:
            # silu(g) = g / (1 + exp(-g)) — silumul.py's own chain.
            e = tl.exp((-g).to(tl.float32)).to(tl.float16)
            a = tl.fdiv(g, 1.0 + e)
        else:
            # gelu(g), elementwise.py's gelu_fwd body verbatim — the tanh form
            # the DDL's `gelu` primitive approximates, through the sigmoid
            # identity; the cubic in the f32 island (an f16 x³ overflows at
            # |x| >= 41).
            g32 = g.to(tl.float32)
            v = 1.5957691216057308 * (g32 + 0.044715 * g32 * g32 * g32)
            e = tl.exp(-v).to(tl.float16)
            a = tl.fdiv(g, 1.0 + e)
        o_desc.store([offs_m, 0], a * u)
    if TAIL_H > 0:
        t_g = tl.make_tensor_descriptor(desc_gate, shape=[M, N], strides=[N, 1],
                                        block_shape=[TAIL_H, BLOCK_N])
        t_u = tl.make_tensor_descriptor(desc_up, shape=[M, N], strides=[N, 1],
                                        block_shape=[TAIL_H, BLOCK_N])
        t_o = tl.make_tensor_descriptor(desc_o, shape=[M, N], strides=[N, 1],
                                        block_shape=[TAIL_H, BLOCK_N])
        offs_m = N_BLOCKS * BLOCK_M
        g = t_g.load([offs_m, 0])
        u = t_u.load([offs_m, 0])
        if ACT == 0:
            e = tl.exp((-g).to(tl.float32)).to(tl.float16)
            a = tl.fdiv(g, 1.0 + e)
        else:
            g32 = g.to(tl.float32)
            v = 1.5957691216057308 * (g32 + 0.044715 * g32 * g32 * g32)
            e = tl.exp(-v).to(tl.float16)
            a = tl.fdiv(g, 1.0 + e)
        t_o.store([offs_m, 0], a * u)
