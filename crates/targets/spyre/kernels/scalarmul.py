# SPDX-License-Identifier: Apache-2.0
"""Scratchy's Triton scalarmul kernel — `out = x * SCALE` over a whole-region tile.

BODY PROVENANCE: derived from `crates/targets/spyre/kernels/elementwise.py`'s form —
descriptor-loaded whole-region tiles, one arithmetic chain, one store — with the
multiplier a `tl.constexpr`, which is the whole difference from `mul_fwd`: the scale
is a monomorphisation constant, not a second descriptor operand.

THE SPLICE'S OWN CONTRACT (what `scratchy-triton-splice` states about this kernel):

* PARAMETERS, IN ORDER: `desc_x`, `desc_o` — the node's one input, then the output.
  The registry does not permute.
* CONSTEXPRS: `M`, `N`, `BLOCK_M`, `BLOCK_N`, `SCALE`, `N_TOTAL`, `C_START` — stated
  by the splice from the node's own region (`M = BLOCK_M =` rows, `N = BLOCK_N =` the
  region's width) and the node's own `SubOp::ScalarMul { scale }` payload, plus the
  STORAGE the region windows (`N_TOTAL =` the tensor's full width, `C_START =` the
  region's column corner). A whole-region node states `N_TOTAL = N`, `C_START = 0` —
  the same constants the one-tile form implied.
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

⛔ THE SCALE MUST REACH THE PROGRAM AS ONE SPLAT FEEDING ONE `arith.mulf` — the
consumer's `program_scalarmul_scale` reads the multiplier OFF THE PROGRAM (every
`arith.mulf` in the body must multiply by the same splat) and then looks its value up
in `BundleLayout::scalarmul_scales`, the registry the TAPE-side layout pass fills from
the node's own `scale` payload. So `x * SCALE` — a constexpr multiply, lowered as
`splat(SCALE)` then one `arith.mulf` — is the shape the door requires; folding the
multiply away, or spelling the scale a second way, would desync the program from the
registry slot the descriptor reads.

⛔ NO ROW BLOCKING, deliberately — the same law `elementwise.py` states. The builder's
old arm row-blocked a whole `[rows, cols]` region whose three-tile live set exceeded
the LX budget; a region that wide is refused loudly by the splice until its kernel
states the block shape.
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
                  ):
    start_m = tl.program_id(0)
    x_desc = tl.make_tensor_descriptor(desc_x, shape=[M, N_TOTAL], strides=[N_TOTAL, 1],
                                       block_shape=[BLOCK_M, BLOCK_N])
    o_desc = tl.make_tensor_descriptor(desc_o, shape=[M, N_TOTAL], strides=[N_TOTAL, 1],
                                       block_shape=[BLOCK_M, BLOCK_N])
    offs_m = start_m * BLOCK_M
    x = x_desc.load([offs_m, C_START])
    # ONE splat, ONE multiply — the shape `program_scalarmul_scale` reads the scale off.
    o_desc.store([offs_m, C_START], x * SCALE)
