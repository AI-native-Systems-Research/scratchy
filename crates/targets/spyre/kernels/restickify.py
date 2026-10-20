# SPDX-License-Identifier: Apache-2.0
"""THE NEW-BLOCK Kᵀ RESTICKIFY — the relayout the attention's plain-B score dot needs.

The attention kernel's score leg contracts `q [·, HD]` against K with the CONTRACTION
AXIS AS B's ROW AXIS (the door's own prescription — `transposed_b_is_placeable`'s
refusal text): a `.T` over a buffer written n-major has no agent that spends the
transposition, so the new block's K arrives ALREADY transposed, in a scratch this
kernel writes, and the score dot stays PLAIN. The resident prefix gets its Kᵀ the
same way from the Kᵀ PLANE (the pool's third plane, written transposed by its own
writer); the new block has no plane — its Kᵀ is THIS program's output.

THE SPLICE'S CONTRACT (what `scratchy-triton-splice` states about this kernel):

* PARAMETERS, IN ORDER: `desc_k` (the new-K staging tensor, `[ROWS, W]` wide-form)
  and `desc_kt` (the Kᵀ scratch, `[W, ROWS]`). ONE input, ONE output — the
  whole-function walk's `Program::Transpose` shape.
* VIEWS: `desc_k` states the staging tensor's PADDED wide form `[ROWS, W]` —
  `ROWS` is the stick cover `pad_of_mq(new_len)` (the SAME pad the placement
  already sizes `new_k` at: the staging tensor is re-placed `[mq_pad, device
  cols]` precisely so a padded read never aliases the next tensor), and `W` is
  the DEVICE width the producing proj matmul writes (`DeviceWidth::for_pointwise`
  of `nkvh·hd` — the placement's own width law, so the view never leaves the
  buffer). `desc_kt` states the transposed `[W, ROWS]` over the splice-minted
  scratch (a reserved per-layer tid placed at fresh seg3 high-water).
* ONE ACCESS TILE PER SIDE — the tile IS the view: the door's FOLDED basis, its
  original whole-tensor form (one relayout descriptor at zero corners, the core
  split blocking both axes on card). A trip-loop form would state per-trip
  windows, and the transpose door takes exactly ONE window per side — the fold
  is the only single-window shape that covers the tensor.
* ⛔ BOTH EXTENTS ARE WHOLE STICKS BY CONSTRUCTION, and must be: the door's stick
  law refuses a sub-stick extent because the dxp scheduler refuses it
  (L3DlOpsScheduler:1040) and the on-card ReStickify asked for a sub-stick
  transpose anyway wrote the SECOND transposed stick wrong and garbled decode
  past 64 tokens. `ROWS` is a 64-multiple by the pad law; `W` is a 64-multiple
  because every head dim this crate serves is one and the device-width bump
  only adds whole sticks.
* GRID: `[1]` — one work item; the SuperDSC core split parallelises the
  descriptor on card (`transpose_core_split` cuts BOTH axes at the block).

⛔ THE PAD ROWS ARE WRITTEN, NEVER READ. `ROWS > new_len` whenever the block is
not stick-aligned (decode's single row; the ladder's bottom rungs), so the fold
transposes the staging tensor's pad rows into the scratch's pad columns. Those
columns hold whatever the placement's unwritten tail held — and NOTHING reads
them: the attention's B window is the EXACT `[HD, new_len]` at column 0, and the
score tile's own logical width is `new_len`, so the pad columns never enter any
reduce. This is the same contract the staging tensor's placement comment states
("pad rows are masked") — the pads are dead weight the stick law makes
unavoidable, and every consumer bounds itself to the logical extent.
"""

import triton
import triton.language as tl


@triton.jit
def kt_restick(desc_k, desc_kt,  #
               ROWS: tl.constexpr, W: tl.constexpr):
    # ONE TILE PER SIDE — the tile is the view entire (the folded basis). The
    # transpose door places one relayout descriptor at zero corners over both
    # views; the card's core split blocks it. A block smaller than the view
    # would state a window per trip, and the door takes exactly one window per
    # side — several disjoint rectangles are a program shape it refuses by name.
    k_desc = tl.make_tensor_descriptor(desc_k, shape=[ROWS, W],
                                       strides=[W, 1],
                                       block_shape=[ROWS, W])
    kt_desc = tl.make_tensor_descriptor(desc_kt, shape=[W, ROWS],
                                        strides=[ROWS, 1],
                                        block_shape=[W, ROWS])
    # `.T`, never `tl.trans`: the attribute form is the census's own in-scope
    # spelling (`attention_flash.py`'s `q.T`/`k.T` — an `Attribute` node, not a
    # call target), and it lowers to the same `tt.trans {order = [1, 0]}` the
    # ladder's `convert_trans` mints `linalg.transpose` from — the op the
    # whole-function walk classifies as `Program::Transpose`.
    k = k_desc.load([0, 0])  # [ROWS, W]
    kt_desc.store([0, 0], k.T)  # [W, ROWS]
