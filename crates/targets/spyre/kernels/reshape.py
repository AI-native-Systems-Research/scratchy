# SPDX-License-Identifier: Apache-2.0
"""THE RESHAPE SPLICE'S KERNEL — the pure relayout `out = reshape(x)`: the same
element sequence, different `[rows, cols]` extents.

BODY PROVENANCE: main's `KtirFunc::reshape` lowered through the door's own
decomposition — ONE `identity` COPY PER STICK, the `lmlast` mechanism. A device
tensor is stick-blocked, and (because both column extents are 64-multiples —
the door's own stick law) flat elements `64q..64q+64` are EXACTLY one `[1, 64]`
tile of each side's own view: input row `q // (C_IN/64)` at column
`(q % (C_IN/64))·64`, output row `q // (C_OUT/64)` at column
`(q % (C_OUT/64))·64`. Walking the flat sequence a stick at a time is therefore
the one form that is simultaneously (a) a legal load/store pair for the front
end (every block fits its view), (b) a flat copy for the emulator, and (c) the
same op count the door itself emits.

THE SPLICE'S OWN CONTRACT (what `scratchy-triton-splice` states about this kernel):

* PARAMETERS, IN ORDER: `desc_src`, `desc_dst` — the node's one input, then the
  output.
* CONSTEXPRS: `R_IN`, `C_IN`, `R_OUT`, `C_OUT`, `N_STICKS` — stated by the splice
  from the node's own regions and tensors; the producer refuses unless
  `R_IN·C_IN == R_OUT·C_OUT` (the element-count law) and both regions are whole
  tensors.
* GRID: `[1]`.

⭐ THE TRIP COUNT IS CONSTEXPR, so `to_ktir` unrolls the stick loop and the
spliced program is straight-line — one `[1, 64]` load/store pair per stick, all
at top level, which is the straight-line form the door's region walk requires
(an `scf.for`-borne window is refused there by name). The stick count is the
buffer's own (`R·C/64`), the same count of identity legs the door emits, so the
emulator's per-tile work and the card's per-op work agree on the cost shape.

⛔ THE COLUMN EXTENTS MUST BE WHOLE 64-ELEMENT STICKS — the door's own law for
both sides, re-stated here because the walk's coordinates assume it. A
sub-stick width is a build `Err` at the door naming it; the splice's count law
fires first for a malformed node.
"""

import triton
import triton.language as tl


@triton.jit
def reshape_fwd(desc_src, desc_dst,  #
                R_IN: tl.constexpr, C_IN: tl.constexpr,  #
                R_OUT: tl.constexpr, C_OUT: tl.constexpr,  #
                N_STICKS: tl.constexpr,  #
                ):
    # Both sides at their TRUE extents — the door reads `v_rows`/`v_cols` off
    # these views and derives its own stick walk from them.
    src_desc = tl.make_tensor_descriptor(desc_src, shape=[R_IN, C_IN],
                                         strides=[C_IN, 1],
                                         block_shape=[1, 64])
    dst_desc = tl.make_tensor_descriptor(desc_dst, shape=[R_OUT, C_OUT],
                                         strides=[C_OUT, 1],
                                         block_shape=[1, 64])
    # Sticks per row on each side (the extents are 64-multiples, the door's law).
    IN_STICKS: tl.constexpr = C_IN // 64
    OUT_STICKS: tl.constexpr = C_OUT // 64
    # ONE SINGLE-STICK COPY PER FLAT STICK — the `lmlast` form. Each stick's
    # coordinates on both sides are whole `//`/`%` of its own view's extents.
    for q in tl.range(0, N_STICKS, 1):
        v = src_desc.load([q // IN_STICKS, (q % IN_STICKS) * 64])
        dst_desc.store([q // OUT_STICKS, (q % OUT_STICKS) * 64], v)
