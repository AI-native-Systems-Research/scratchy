# SPDX-License-Identifier: Apache-2.0
"""Scratchy's Triton scalarmul kernel — `out = x * SCALE` over a whole-region tile.

BODY PROVENANCE: derived from `crates/targets/spyre/kernels/elementwise.py`'s form —
descriptor-loaded whole-region tiles, one arithmetic chain, one store — with the
multiplier a `tl.constexpr`, which is the whole difference from `mul_fwd`: the scale
is a monomorphisation constant, not a second descriptor operand.

THE SPLICE'S OWN CONTRACT (what `scratchy-triton-splice` states about this kernel):

* PARAMETERS, IN ORDER: `desc_x`, `desc_o` — the node's one input, then the output.
  The registry does not permute.
* CONSTEXPRS: `M`, `N`, `BLOCK_M`, `BLOCK_N`, `SCALE`, `N_TOTAL`, `C_START`,
  `N_BLOCKS`, `TAIL_H` — stated by the splice from the node's own region (`M =` rows,
  `N = BLOCK_N =` the region's DEVICE width, see below) and the node's own
  `SubOp::ScalarMul { scale }` payload, plus the STORAGE the region windows
  (`N_TOTAL =` the tensor's full width, `C_START =` the region's column corner). A
  whole-region node states `N_TOTAL = N`, `C_START = 0` — the same constants the
  one-tile form implied.
* GRID: `[1]`.

⛔ THE DESCRIPTOR NAMES THE STORAGE, THE LOAD NAMES THE WINDOW — the front end's
column chunking (`n_blocks(out_cols, nb)`, production `nb = 8192`) splits one wide
pointwise op into chunks that share one output tensor, and the builder's own program
states each chunk's access-tile corner (`KtirFunc::load_region` honors
`region.cols.start`). The descriptor's shape/strides are therefore the TENSOR's
(`[M, N_TOTAL]` / `[N_TOTAL, 1]`), never the chunk's, and the load/store corner is
`C_START` — stating the chunk's width as the descriptor's would stride the store
wrong (the 8b splice defect: chunk 1 of a 12800-wide intermediate read and wrote
chunk 0's columns).

⛔ THE WINDOW IS THE DEVICE WIDTH, NOT THE LOGICAL ONE — the door's own law for this
family (`scalarmul_scaled` pads through `DeviceWidth::for_pointwise`, because a
ScalarMul on the padded logits must use the width its producer matmul emitted), and
the Triton front end refuses a block whose last dim is under 16 bytes, so granite's
3-wide logits tail chunk is not spellable at its logical width. `for_pointwise` is
IDEMPOTENT, so the door re-derives the same width from this program as from the
builder's logical one.

⛔ THE SCALE MUST REACH THE PROGRAM AS ONE SPLAT FEEDING ONE `arith.mulf` PER BLOCK —
the consumer's `program_scalarmul_scale` reads the multiplier OFF THE PROGRAM (every
`arith.mulf` in the body must multiply by the same splat) and then looks its value up
in `BundleLayout::scalarmul_scales`, the registry the TAPE-side layout pass fills from
the node's own `scale` payload. So `x * SCALE` — a constexpr multiply, lowered as
`splat(SCALE)` then one `arith.mulf` per tile — is the shape the door requires;
folding the multiply away, or spelling the scale a second way, would desync the
program from the registry slot the descriptor reads.

⭐ ROW BLOCKING, THE BUILDER'S OWN LAW (`lower_scalarmul_node`'s `by_row`): a whole
`[rows, cols]` region whose three-tile live set (`rows × cols × 3`) exceeds the
`EW_LX_ELEMS` budget is ROW-BLOCKED, not refused — `N_BLOCKS` full `[BLOCK_M, N]`
tiles (BLOCK_M = the builder's own `rows_per_block(cols, 3)`) plus a `[TAIL_H, N]`
tail tile when the region does not divide evenly. The trip count is constexpr, so
`to_ktir` unrolls it and the spliced program is straight-line with one
constant-corner tile per block — the door folds those through `Region::r_cover` and
ONE descriptor spans all of them, exactly as it spans the builder's row-blocked
program.
"""

import triton
import triton.language as tl


@triton.jit
def scalarmul_fwd(desc_x, desc_o,  #
                  M: tl.constexpr, N: tl.constexpr,  #
                  BLOCK_M: tl.constexpr,  #
                  BLOCK_N: tl.constexpr,  #
                  SCALE: tl.constexpr,  #
                  N_TOTAL: tl.constexpr,  #
                  C_START: tl.constexpr,  #
                  N_BLOCKS: tl.constexpr,  #
                  TAIL_H: tl.constexpr,  #
                  ):
    x_desc = tl.make_tensor_descriptor(desc_x, shape=[M, N_TOTAL], strides=[N_TOTAL, 1],
                                       block_shape=[BLOCK_M, BLOCK_N])
    o_desc = tl.make_tensor_descriptor(desc_o, shape=[M, N_TOTAL], strides=[N_TOTAL, 1],
                                       block_shape=[BLOCK_M, BLOCK_N])
    for blk in tl.range(0, N_BLOCKS, 1):
        offs_m = blk * BLOCK_M
        x = x_desc.load([offs_m, C_START])
        # ONE splat, ONE multiply — the shape `program_scalarmul_scale` reads the
        # scale off.
        o_desc.store([offs_m, C_START], x * SCALE)
    if TAIL_H > 0:
        t_x = tl.make_tensor_descriptor(desc_x, shape=[M, N_TOTAL], strides=[N_TOTAL, 1],
                                        block_shape=[TAIL_H, BLOCK_N])
        t_o = tl.make_tensor_descriptor(desc_o, shape=[M, N_TOTAL], strides=[N_TOTAL, 1],
                                        block_shape=[TAIL_H, BLOCK_N])
        offs_m = N_BLOCKS * BLOCK_M
        x = t_x.load([offs_m, C_START])
        t_o.store([offs_m, C_START], x * SCALE)
