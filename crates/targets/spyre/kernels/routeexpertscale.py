# SPDX-License-Identifier: Apache-2.0
"""Scratchy's Triton router expert-scale kernel — each score times its
expert's learned scale.

BODY PROVENANCE: the ladder's own column-scan scale-select, designed against
the door's contract (`lower_ktir_to_superdsc::route_expert_scale`) and the emu
golden (`moe_router_chain`'s
`route_expert_scale_multiplies_the_per_expert_scale`), not ported from a
fixture. The door never walks these ops (it re-derives its `7k − 2` one-hot
select chain from the minted views), so this body states the same function the
door's chain computes, for the ladder's own executed proof
(`triton-numeric`'s `routeexpertscale.rs`):

    out[i, j] = scores[i, j] · per_expert_scale[idx[i, j]]

gemma's `moe_per_expert_scale` — the same one-hot select as the gather, with
the LOADED `[1, E]` scale row as the gather parent (a real weight parameter
the worker stages, not a bake-time const).

THE SPLICE'S OWN CONTRACT (what `scratchy-triton-splice` states about this
kernel):

* PARAMETERS, IN ORDER: `desc_p`, `desc_i`, `desc_s`, `desc_o` — ⛔ NOT the
  node's operand order. The node's inputs are `[scores, idx, scale]`, but the
  door reads its parameters positionally as `[scale, idx, scores, out]` —
  `tensors[0]` IS THE SCALE ROW (the door's own ⛔ note: the old door read
  them as `[scores, idx, scale]` and every descriptor the m=1 bake shipped
  computed the WRONG function — the score rode the scale row's lanes — passing
  at m ≤ 4 only because the mis-read fit the bumped footprint, until the
  placement guard refused at m=8). The splice's binding row REORDERS to the
  door's mint order: `regions()` walks the ARGUMENTS, and argument 0 must be
  the scale row's `[1, E]` view.
* CONSTEXPRS: `M`, `K`, `E` — stated by the splice from the node's own
  regions: `M` the output's rows, `K` the output's cols, `E` the scale row's
  cols, guarded so the scores' and indices' `[m, k]` match the output's and
  the scale row is one row (the door's own extent laws).
* GRID: `[1]`.

⛔ THE SCALE-SELECT IS LANE-FREE, AND THAT IS A MEASURED LAW, NOT A TASTE.
The natural Triton spelling of "scale[idx[i, j]]" is an `arange`-keyed one-hot
compare — and `tl.arange` is REFUSED by the front end at the chain fixture's
`E = 4`/`E = 5` geometries (a power-of-two length is required), the same law
`routeargsort.py`'s header states. The column-scan form needs no lane vector:
per slot `j`, each scale column `h`'s `[1, 1]` tile is matched against the
slot's loaded index tile, and the matched column's value is accumulated — a
real index is a single lane, so exactly one column matches per row and the
sum IS the selected scale.

⛔ THE SCALE RIDES THE MULTIPLY, NOT THE SELECT. `tl.where` broadcasts its
CONDITION up to the value pair's shape — never the values down to the cond's
— so a `[1, 1]` scale tile cannot sit in the select against an `[M, 1]`
compare. The match tile (`where(h_col == ij, one, 0.0)`, the `[M, 1]` f16
`one` on the select's LEFT per the M3 law) multiplies the `[1, 1]` scale
tile instead — the `[M, 1] × [1, 1]` broadcast multiply `scalarwmul.py`'s
header already states ("the weight loads to `[1, 1]`, which the multiply
broadcasts over the `[M, N]` tile"), the axis-0 spray the door's own
`In::mb(&scale)` leg names.

NUMERICS, THE GOLDEN'S OWN: the selected scale is the weight's own f16 bits,
the accumulation adds exact zeros elsewhere, and the one terminal multiply is
the score's — EXACT whenever the product is f16-representable (the chain
fixture's are: every value a power-of-two multiple). The indices are integers
below `E` and exact in f16 (integers are exact to 2048; the real routers'
`E = 128`). ⚠️ The door's own doc owns the rounding question for scores whose
products are NOT representable: the emu's f32 intermediate (one rounding) is
not replicated on this first landing — the tiny26 EMU-vs-card parity gate
owns that upgrade.

⛔ EVERY LOOP IS `tl.static_range` — the ladder's own law (see
`expertmatmul.py`): the induction variable is a PYTHON INT at trace time, so
every load corner (`[0, j]`, `[0, h]`) and every `tl.full` value (`float(h)`)
is constant at trace. The cost is the door's own `O(k·E)` compare count — an
emulator-only cost the card never pays, because the card runs the door's
`7k − 2` chain over the same views.
"""

import triton
import triton.language as tl


@triton.jit
def routeexpertscale_fwd(desc_p, desc_i, desc_s, desc_o,  #
                         M: tl.constexpr, K: tl.constexpr, E: tl.constexpr):
    # Descriptors name STORAGE (the door's own views), ⛔ IN THE DOOR'S MINT
    # ORDER: the `[1, E]` scale row is PARAMETER 0 (the door's `tensors[0]`
    # law — the splice's binding row reordered the node's [scores, idx,
    # scale] to feed this order), then the `[M, K]` indices and scores as
    # `[M, 1]` columns (the door's `In::col_at` reads), and the `[M, K]`
    # output as `[M, 1]` columns so each slot lands in its own lane (the
    # door's identity-row accumulate shape). The `regions()` walk takes the
    # FIRST view per parameter: `p_desc`'s shape `[1, E]` is what the door
    # reads as `E`, `o_desc`'s `[M, K]` as `rows` and `k`.
    p_desc = tl.make_tensor_descriptor(desc_p, shape=[1, E], strides=[E, 1],
                                       block_shape=[1, 1])
    i_desc = tl.make_tensor_descriptor(desc_i, shape=[M, K], strides=[K, 1],
                                       block_shape=[M, 1])
    s_desc = tl.make_tensor_descriptor(desc_s, shape=[M, K], strides=[K, 1],
                                       block_shape=[M, 1])
    o_desc = tl.make_tensor_descriptor(desc_o, shape=[M, K], strides=[K, 1],
                                       block_shape=[M, 1])
    # The select's LEFT operand — an [M, 1] f16 constant tile, so the `0.0`
    # splat island-collapses to its f16 (the M3 law).
    one = tl.full([M, 1], 1.0, tl.float16)
    for j in tl.static_range(0, K):
        # The slot's per-row index and score, each loaded as its own [M, 1]
        # tile (the door's `col_at` read of lane j).
        ij = i_desc.load([0, j])
        sv = s_desc.load([0, j])
        v = tl.zeros([M, 1], dtype=tl.float16)
        for h in tl.static_range(0, E):
            # Scale column h's [1, 1] tile against the slot's index — a real
            # index names one lane, so exactly one column matches per row.
            # The match tile is [M, 1] (the `one` tile on the select's LEFT),
            # and the [1, 1] scale rides the MULTIPLY's broadcast — the
            # scalarwmul law, not the select (the cond only broadcasts UP).
            ph = p_desc.load([0, h])
            h_col = tl.full([M, 1], float(h), tl.float16)
            m_h = tl.where(h_col == ij, one, 0.0)
            v = v + m_h * ph
        # Slot j's scaled score lands in lane j of the output — the score
        # times its expert's scale, one f16 multiply.
        o_desc.store([0, j], sv * v)
