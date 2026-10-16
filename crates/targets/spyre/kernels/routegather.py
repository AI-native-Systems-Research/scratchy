# SPDX-License-Identifier: Apache-2.0
"""Scratchy's Triton router score-gather kernel — take_along_axis at −1.

BODY PROVENANCE: the ladder's own column-scan gather, designed against the
door's contract (`lower_ktir_to_superdsc::route_gather_scores`) and the emu
golden (`moe_router_chain`'s `route_gather_scores_takes_along_the_last_axis`),
not ported from a fixture. The door never walks these ops (it re-derives its
`6k − 2` one-hot select chain from the minted views), so this body states the
same selection function the door's chain computes, for the ladder's own
executed proof (`triton-numeric`'s `routegather.rs`):

    out[i, j] = src[i, idx[i, j]]

the plain `take_along_axis` at the last axis — a TRUE gather over whatever
indices the node states, not a top-k re-derivation (the chain fixture feeds
hand-set indices that are not any top-k's output, so a value-recomputing
substitute would fail it by construction).

THE SPLICE'S OWN CONTRACT (what `scratchy-triton-splice` states about this
kernel):

* PARAMETERS, IN ORDER: `desc_x`, `desc_i`, `desc_o` — the node's own operand
  order (the source, the indices, the `[m, k]` output), the plain
  `[input, input, output]` binding.
* CONSTEXPRS: `M`, `E`, `K` — stated by the splice from the node's own
  regions: `M` the output's rows, `K` the output's cols, `E` the source's
  cols, guarded so the indices' `[m, k]` matches the output's and every view
  shares one row count (the door's own extent laws).
* GRID: `[1]`.

⛔ THE GATHER IS LANE-FREE, AND THAT IS A MEASURED LAW, NOT A TASTE. The
natural Triton spelling of "select lane `idx[i, j]`" is an `arange`-keyed
one-hot compare — and `tl.arange` is REFUSED by the front end at the chain
fixture's `E = 5` (a power-of-two length is required), the same law
`routeargsort.py`'s header states. The column-scan form needs no lane vector:
per slot `j`, each source column `h`'s `[M, 1]` tile is compared against the
slot's loaded index tile, and the matching column's value is added — a real
index is a single lane, so exactly one column matches per row and the sum IS
the gathered value.

⛔ THE VALUE IS f16, THE DOOR'S OWN DTYPE, AND EVERY SELECT TAKES THE TENSOR
AS ITS LEFT OPERAND. The two measured laws `routeargsort.py`'s header states
hold here verbatim: `LegalizeTypes` (SP-E2-03) collapses every f32 island to
f16 (no `.to(tl.float32)` anywhere — the gathered value is the source's own
bits, one add of zeros away), and `tl.where` types its pair WITHOUT promotion
— so the source tile sits on the select's LEFT with the `0.0` literal
collapsing to its f16, and the lane constant rides
`tl.full([M, 1], float(h), tl.float16)` (an f16 constant from birth). The
compare is tensor-vs-tensor f16 (`h_col == ij`, both tiles — the M3-proven
form), never tensor-vs-scalar.

NUMERICS, THE GOLDEN'S OWN: the gathered value is the source column's own
f16 bits and the sum adds zeros elsewhere — EXACT, no rounding at all. The
indices are integers below `E` and exact in f16 (integers are exact to 2048;
the real routers' `E = 128`).

⛔ EVERY LOOP IS `tl.static_range` — the ladder's own law (see
`expertmatmul.py`): the induction variable is a PYTHON INT at trace time, so
every load corner (`[0, j]`, `[0, h]`) and every `tl.full` value (`float(h)`)
is constant at trace. The cost is the door's own `O(k·E)` compare count — an
emulator-only cost the card never pays, because the card runs the door's
`6k − 2` chain over the same views.
"""

import triton
import triton.language as tl


@triton.jit
def routegather_fwd(desc_x, desc_i, desc_o,  #
                    M: tl.constexpr, E: tl.constexpr, K: tl.constexpr):
    # Descriptors name STORAGE (the door's own views): the `[M, E]` source
    # and the `[M, K]` indices each re-viewed as `[M, 1]` columns (the door's
    # `In::col_at` reads), and the `[M, K]` output as `[M, 1]` columns so
    # each slot lands in its own lane (the door's identity-row accumulate
    # shape — the store windows tile the view's rows, which is what the
    # door's `node_rows` law reads). ⛔ THE `regions()` WALK TAKES THE FIRST
    # VIEW PER PARAMETER: `x_desc`'s shape `[M, E]` is what the door reads as
    # `E`, `i_desc`'s `[M, K]` and `o_desc`'s as `k` — the door's own extent
    # laws, and the door's `split_out(.., 2)` counts exactly these two
    # non-output parameters.
    x_desc = tl.make_tensor_descriptor(desc_x, shape=[M, E], strides=[E, 1],
                                       block_shape=[M, 1])
    i_desc = tl.make_tensor_descriptor(desc_i, shape=[M, K], strides=[K, 1],
                                       block_shape=[M, 1])
    o_desc = tl.make_tensor_descriptor(desc_o, shape=[M, K], strides=[K, 1],
                                       block_shape=[M, 1])
    for j in tl.static_range(0, K):
        # The slot's per-row index, loaded as its own [M, 1] tile (the door's
        # `col_at` read of lane j).
        ij = i_desc.load([0, j])
        v = tl.zeros([M, 1], dtype=tl.float16)
        for h in tl.static_range(0, E):
            # Source column h against the slot's index — a real index names
            # one lane, so exactly one column matches per row. The source
            # tile is the select's LEFT operand (the `0.0` splat
            # island-collapses to its f16), and the lane constant rides a
            # constant f16 tile.
            s = x_desc.load([0, h])
            h_col = tl.full([M, 1], float(h), tl.float16)
            v = v + tl.where(h_col == ij, s, 0.0)
        # Slot j's gathered value lands in lane j of the output — the
        # source's own f16 bits.
        o_desc.store([0, j], v)
