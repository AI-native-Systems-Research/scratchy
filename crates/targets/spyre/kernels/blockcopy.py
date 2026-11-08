# SPDX-License-Identifier: Apache-2.0
"""THE BLOCK COPY'S KERNEL — `dst[:, COL_OFF : COL_OFF + C_SRC] = src`: a whole `[R, C_SRC]`
source placed as a column block of an `[R, C_DST]` destination. An all-gather consume's leg: one
rank's block (its own, or its peer's out of the receive window) landing at that rank's columns.

BODY PROVENANCE: `reshape.py`'s stick walk — ONE `[1, 64]` identity copy per stick, the `lmlast`
mechanism — with the destination column shifted by `COL_OFF`. The door
(`lower_ktir_to_superdsc::block_copy`) reads the block off the program's own first tiles (the
source at its corner, the destination at `[0, COL_OFF]`) and lowers the whole walk to ONE identity
over the run of stick columns, which stick-major layout makes contiguous at both ends.

THE SPLICE'S OWN CONTRACT (what `scratchy-triton-splice` states about this kernel):

* PARAMETERS, IN ORDER: `desc_src`, `desc_dst` — the source, then the destination.
* CONSTEXPRS: `R`, `C_SRC`, `C_DST`, `COL_OFF`, `N_STICKS` (= `R · C_SRC / 64`) — stated by the
  splice from the two regions; the door refuses unless `C_SRC`, `C_DST` and `COL_OFF` are whole
  64-element sticks and the block fits.
* GRID: `[1]`.

⭐ THE TRIP COUNT IS CONSTEXPR, so `to_ktir` unrolls the walk into straight-line copies — the form
the door's region walk requires, exactly as for `reshape.py`.
"""

import triton
import triton.language as tl


@triton.jit
def blockcopy_fwd(desc_src, desc_dst,  #
                  R: tl.constexpr, C_SRC: tl.constexpr, C_DST: tl.constexpr,  #
                  COL_OFF: tl.constexpr, N_STICKS: tl.constexpr,  #
                  ):
    src_desc = tl.make_tensor_descriptor(desc_src, shape=[R, C_SRC],
                                         strides=[C_SRC, 1],
                                         block_shape=[1, 64])
    dst_desc = tl.make_tensor_descriptor(desc_dst, shape=[R, C_DST],
                                         strides=[C_DST, 1],
                                         block_shape=[1, 64])
    SRC_STICKS: tl.constexpr = C_SRC // 64
    # ONE SINGLE-STICK COPY PER SOURCE STICK, landing `COL_OFF` columns over.
    for q in tl.range(0, N_STICKS, 1):
        v = src_desc.load([q // SRC_STICKS, (q % SRC_STICKS) * 64])
        dst_desc.store([q // SRC_STICKS, COL_OFF + (q % SRC_STICKS) * 64], v)
