# SPDX-License-Identifier: Apache-2.0
"""THE ROPE SPLICE'S KERNEL — Granite's split-half NeoX `rotate_half`, adapted from the
numerically validated fixture `crates/triton/test-fixtures/rope.py` (rope_q32/rope_kv8
in `triton-numeric`'s executed configs).

    q_embed = q * cos + rotate_half(q) * sin
    rotate_half(x) = cat(-x[..., d//2:], x[..., :d//2])

which, written per half (x1 = x[..., :d//2], x2 = x[..., d//2:]), is

    out1 = x1 * cos - x2 * sin
    out2 = x2 * cos + x1 * sin

THE SPLICE'S CONTRACT — the door (`ktir_superdsc_door::rope`) derives the node's facts
from the program's own views and access tiles, so the kernel states them:

* the x/out descriptors are `[MQ * H, HEAD_DIM]` — the WORKER's staging. Main's
  `KtirFunc::rope` views x as `[rows*heads, hd]` over the same bytes, so the view extents
  agree.
* ONE ACCESS TILE PER POSITION, `[H, HALF]` — the door reads `heads` off the FIRST access
  tile's row extent (`Region::r_len`) and `mq` as `v_rows / heads`, so a whole-tensor
  tile would state `heads = MQ*H` and mis-derive everything. Main's builder program
  takes exactly these tiles (main's `KtirFunc::rope`'s `tile(x_view, ri*heads, 0, heads, half)`).
* ONE WORK ITEM (grid `[1]`) with the position loop UNROLLED by the ladder
  (`to_ktir::unroll_constant_trip_loops`): constant-trip `tl.range(0, MQ, 1)`, each trip
  a copy of the per-position body. The spliced program is straight-line, like the
  builder's.
* cos/sin: the worker stages per-position head-tilled rows (`spyre_forward.rs`'s `tile`
  over `rope_cos_sin`), so element (p, h*hd+d) sits at `p*total + h*hd + d` — the
  `[MQ*H, HEAD_DIM]` head-replicated view, read at the position's OWN row base. The
  fixture's `stage_table` (delta 8) is the same bytes; the door's "NOT THE COS VIEW"
  trap does not bite here because `rope_at` derives its extents from `mq`/`total`, not
  from the cos parameter's view.

THE ROW ORDER, kept from the fixture (delta 7, measured): row = position * H + head,
TOKEN-MAJOR, which is `addr_eq`'s same-bytes reshape of the worker's `[mq, heads*hd]`
row-major staging — the arrangement the attention that consumes this tensor reads.

⭐ THE ROW WINDOW (per-row rope): the real front end's per-position decomposition
produces rope nodes whose regions are ONE row's `[1, qw]` slice of the shared q
plane, and a row slice is expressed by the ACCESS TILE's row corner, never by a
taller view (main's `view_rows` law: the offset rides the tile's row index, which
is the only form legible to both the emulator and the GEMM offload). The kernel
therefore states `Y_TOTAL` — the x TENSOR's full re-tiled height `rows·H`, so the
tile corner is in bounds — and `R_START` — the region's row corner `ri·H`, baked
into every load/store row. `MQ` is then the REGION's row count: a whole-region
node states `R_START = 0`, `Y_TOTAL = MQ·H`, which is byte-identical to the one
form this kernel used to state; a per-row node states `MQ = 1` at `R_START =
ri·H`. cos/sin take the same corner: position `p`'s angles are rows
`[p·H, (p+1)·H)` of the head-replicated table, so the region's rows are exactly
`[R_START, R_START + MQ·H)`.
"""

import triton
import triton.language as tl


@triton.jit
def rope_fwd(desc_x, desc_cos, desc_sin, desc_o,
             H: tl.constexpr, MQ: tl.constexpr,
             HEAD_DIM: tl.constexpr, HALF: tl.constexpr,
             Y_TOTAL: tl.constexpr, R_START: tl.constexpr):
    tl.static_assert(HALF + HALF == HEAD_DIM)
    # The x/out views over the worker-staged plane, `[rows*heads, hd]` — the full
    # TENSOR re-tiled (Y_TOTAL), so the row-windowed tile corners stay in bounds.
    # The cos/sin views are the same-bytes head-replicated tables, `[tbl_rows*H,
    # hd]` — their extents agree with x's when the tables are per-position rows.
    x_desc = tl.make_tensor_descriptor(desc_x, shape=[Y_TOTAL, HEAD_DIM],
                                       strides=[HEAD_DIM, 1],
                                       block_shape=[H, HALF])
    o_desc = tl.make_tensor_descriptor(desc_o, shape=[Y_TOTAL, HEAD_DIM],
                                       strides=[HEAD_DIM, 1],
                                       block_shape=[H, HALF])
    # THE WORKER-STAGED cos/sin, `[mq*heads, hd]` head-replicated (spyre_forward.rs's
    # `tile`): position p's angles are H consecutive rows starting at p*H — read at x's
    # own row base, so cos rows line up with x's H head-rows row for row.
    cos_desc = tl.make_tensor_descriptor(desc_cos, shape=[Y_TOTAL, HEAD_DIM],
                                         strides=[HEAD_DIM, 1],
                                         block_shape=[H, HALF])
    sin_desc = tl.make_tensor_descriptor(desc_sin, shape=[Y_TOTAL, HEAD_DIM],
                                         strides=[HEAD_DIM, 1],
                                         block_shape=[H, HALF])

    # ONE ACCESS TILE PER POSITION (`[H, HALF]` at row base `R_START + pos * H`,
    # column offsets 0 and HALF). The trip count is constexpr, so `to_ktir`
    # unrolls it and the spliced program is straight-line with one tile per
    # position. A whole-region node states `R_START = 0`, so its tiles sit at
    # `pos * H` exactly as before; a per-row node states `MQ = 1` and its single
    # trip's corner is the region's own row.
    for pos in tl.range(0, MQ, 1):
        offs_m = R_START + pos * H
        x1 = x_desc.load([offs_m, 0])
        x2 = x_desc.load([offs_m, HALF])
        c = cos_desc.load([offs_m, 0])
        s = sin_desc.load([offs_m, 0])
        # rotate_half, per half: the MINUS is on the second half's contribution to
        # the first, which is `cat(-x2, x1)` written out.
        o_desc.store([offs_m, 0], x1 * c - x2 * s)
        o_desc.store([offs_m, HALF], x2 * c + x1 * s)
