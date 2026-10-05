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
* CONSTEXPRS: `M`, `N`, `BLOCK_M`, `BLOCK_N`, `N_TOTAL`, `C_START`, `N_BLOCKS`,
  `TAIL_H` — stated by the splice from the node's own region (`M =` the region's row
  count, `N = BLOCK_N =` its width) PLUS the STORAGE the region windows (`N_TOTAL =`
  the tensor's full width, `C_START =` the region's column corner). A whole-region
  node states `N_TOTAL = N`, `C_START = 0`. See `scalarmul.py`'s window law: the
  descriptor names the storage, the load names the window — granite-8b's 12800-wide
  intermediate is TWO chunks sharing one tensor, and chunk 1 must read and write
  columns 8192.. of that tensor, not its base.
* GRID: `[1]`.

⭐ ROW BLOCKING, THE BUILDER'S OWN LAW (`KtirFunc::silu_mul`): eight tiles are live
here (gate, up, neg, exp, the splat, denom, silu, y), so a whole `[mq, intermediate]`
region does not fit a core's LX at prefill — the widest live set in the model. The
kernel takes `N_BLOCKS` full `[BLOCK_M, N]` tiles (BLOCK_M = the builder's own
`rows_per_block(cols, 8)`) plus a `[TAIL_H, N]` tail tile when the region does not
divide evenly. The trip count is constexpr, so `to_ktir` unrolls it and the spliced
program is straight-line with one constant-corner tile per block — the door folds
those through `Region::r_cover` and ONE descriptor spans all of them, exactly as it
spans the builder's row-blocked program. `swiglu_mlp.py`'s BLOCK_N=64 stick-width
note does NOT apply here: that is a `tl.dot` tile constraint, and this kernel has no
`tl.dot`.
"""

import triton
import triton.language as tl


@triton.jit
def silumul_fwd(desc_g, desc_u, desc_o,  #
                M: tl.constexpr, N: tl.constexpr,  #
                BLOCK_M: tl.constexpr,  #
                BLOCK_N: tl.constexpr,  #
                N_TOTAL: tl.constexpr,  #
                C_START: tl.constexpr,  #
                N_BLOCKS: tl.constexpr,  #
                TAIL_H: tl.constexpr,  #
                ):
    g_desc = tl.make_tensor_descriptor(desc_g, shape=[M, N_TOTAL], strides=[N_TOTAL, 1],
                                       block_shape=[BLOCK_M, BLOCK_N])
    u_desc = tl.make_tensor_descriptor(desc_u, shape=[M, N_TOTAL], strides=[N_TOTAL, 1],
                                       block_shape=[BLOCK_M, BLOCK_N])
    o_desc = tl.make_tensor_descriptor(desc_o, shape=[M, N_TOTAL], strides=[N_TOTAL, 1],
                                       block_shape=[BLOCK_M, BLOCK_N])

    # THE FULL BLOCKS — one `[BLOCK_M, N]` tile per trip, at the trip's own row
    # corner. Unrolled by the ladder, so each corner is a constant the door reads.
    for blk in tl.range(0, N_BLOCKS, 1):
        offs_m = blk * BLOCK_M
        g = g_desc.load([offs_m, C_START])
        u = u_desc.load([offs_m, C_START])
        # silu(g) = g / (1 + exp(-g)) — the same chain `KtirFunc::silu_mul` writes.
        # `tl.sigmoid` refuses an f16 tensor outright (swiglu delta 5, `math.exp` is
        # @_check_dtype(["fp32","fp64"])), and the ladder's LegalizeTypes collapses
        # the widen/truncate island this spells, so the emitted KTIR carries
        # `math.exp` on an f16 tile with no extf/truncf — the exact four ops the
        # builder emits.
        e = tl.exp((-g).to(tl.float32)).to(tl.float16)
        s = tl.fdiv(g, 1.0 + e)  # tl.fdiv, not `/`: `/` upcasts f16→f32 (swiglu delta 6)
        o_desc.store([offs_m, C_START], s * u)

    # THE TAIL — its own `[TAIL_H, N]` tile at the row corner the full blocks stop
    # at. `if` on a constexpr folds at codegen (only the taken branch is visited),
    # so an even-divide region emits no tail at all.
    if TAIL_H > 0:
        t_g = tl.make_tensor_descriptor(desc_g, shape=[M, N_TOTAL], strides=[N_TOTAL, 1],
                                        block_shape=[TAIL_H, BLOCK_N])
        t_u = tl.make_tensor_descriptor(desc_u, shape=[M, N_TOTAL], strides=[N_TOTAL, 1],
                                        block_shape=[TAIL_H, BLOCK_N])
        t_o = tl.make_tensor_descriptor(desc_o, shape=[M, N_TOTAL], strides=[N_TOTAL, 1],
                                        block_shape=[TAIL_H, BLOCK_N])
        offs_m = N_BLOCKS * BLOCK_M
        g = t_g.load([offs_m, C_START])
        u = t_u.load([offs_m, C_START])
        e = tl.exp((-g).to(tl.float32)).to(tl.float16)
        s = tl.fdiv(g, 1.0 + e)
        t_o.store([offs_m, C_START], s * u)
