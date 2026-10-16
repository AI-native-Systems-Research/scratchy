# SPDX-License-Identifier: Apache-2.0
"""Scratchy's Triton router argsort kernel — the stable ascending rank vector.

BODY PROVENANCE: the ladder's own count form, designed against the door's
contract (`lower_ktir_to_superdsc::route_argsort`) and the emu golden
(`moe_router_chain`'s `route_argsort_matches_the_stable_ascending_rank`), not
ported from a fixture. The door never walks these ops (it re-derives its
`9W + 1` compare/reduce chain from the minted views), so this body states the
same rank function the door's chain computes, for the ladder's own executed
proof (`triton-numeric`'s `routeargsort.rs`):

    rank[i, j] = |{h : x[i,h] < x[i,j]}| + |{h < j : x[i,h] == x[i,j]}|

the stable ascending argsort (MLX `block_sort`'s ordering, ties by index),
after the NaN→+inf sanitization that makes NaN sort last.

THE SPLICE'S OWN CONTRACT (what `scratchy-triton-splice` states about this
kernel):

* PARAMETERS, IN ORDER: `desc_x`, `desc_o` — the node's own operand order
  (the scores, then the rank vector), the plain `[input, output]` binding.
* CONSTEXPRS: `M`, `E` — stated by the splice from the node's own regions:
  `M` the output's rows, `E` the output's cols, guarded shape-preserving
  against the scores' own `[M, E]` (the door's law: out is `[m, E]` like x).
* GRID: `[1]`.

⛔ THE COUNT FORM IS LANE-FREE, AND THAT IS A MEASURED LAW, NOT A TASTE. The
door's own card form fixes the ranked expert `j` and compares every lane
against it (`lesserthan(xsan, xsan[:, j])`); the natural Triton spelling of
that compare is a `[M, E]` tile against a `[M, 1]` column — which needs NO
lane vector. The `tl.arange`-keyed form (rank as a sum of one-hot products
over precomputed lanes) is REFUSED by the front end at the pinned geometry
anyway (`tl.arange` requires a power-of-two length, and the chain fixture's
`E = 5` is not one), so the count form is the only spelling that states the
function at every E — the door's per-`j` group and this body share it.

⛔ THE COUNT IS f16, THE DOOR'S OWN DTYPE, AND EVERY SELECT TAKES THE TENSOR
AS ITS LEFT OPERAND. Two measured laws force this shape:

* `LegalizeTypes` (SP-E2-03) REMOVES every `arith.extf` and collapses the
  whole f32 stability-widening island back to f16 — there are NO f32 compute
  islands on this ladder, and any f32 the collapse cannot reach is a named
  RED-stop, not a silent degrade. So a count spelled `.to(tl.float32)` runs
  in f16 anyway after the collapse — this kernel simply spells what it means.
* `tl.where` types its pair WITHOUT promotion (a select chooses between its
  operands, it does not compute with them): a float literal splats to the
  pair at ITS OWN dtype — f32 — while the result takes the LEFT operand's
  type after the splat. The TENSOR must therefore be the left operand
  (`where(x == x, x, inf)`, NOT `where(x != x, inf, x)` — the latter's f32
  splat on the left makes the select f32-typed), and the literal splat then
  island-collapses to the tensor's f16. `ones` rides `tl.full` so it is f16
  from birth — never derived from the scores (a score-derived anchor is NaN
  at an inf/NaN lane) and never an f32 constant the collapse must chase.

NUMERICS, THE GOLDEN'S OWN: the rank is a count of lanes, so every value is
an integer below `E` and exact in f16 (integers are exact to 2048; the real
routers' `E = 128`). The tie term is the door's own stable tie-break — a
running count of prior equal columns, the `identity`-seeded accumulator's
ladder spelling — and the NaN sanitize (`x == x` picking `x`, else +inf) is
the golden's NaN-as-greater law. ⛔ The card's first landing does not
replicate that select (the door's `maximum` lets NaN propagate, winning
neither compare); the asymmetry is the door's own doc's, owned by the tiny26
EMU-vs-card parity gate.

⛔ EVERY LOOP IS `tl.static_range` — the ladder's own law (see
`expertmatmul.py`): the induction variable is a PYTHON INT at trace time, so
every load corner (`[0, j]`, `[0, jj]`) is constant at trace and the inner
bound `j` is the outer induction variable itself. The cost is the door's
`O(E²)` compare count over prior columns — an emulator-only cost the card
never pays, because the card runs the door's `9W + 1` chain over the same
views.
"""

import triton
import triton.language as tl


@triton.jit
def routeargsort_fwd(desc_x, desc_o,  #
                     M: tl.constexpr, E: tl.constexpr):
    # Descriptors name STORAGE (the door's own views): the `[M, E]` scores and
    # the `[M, E]` rank vector. `x_desc` reads the whole tile (the door's
    # `lesserthan` operand); `j_desc` re-views the SAME storage as `[M, 1]`
    # columns for the per-j reads (`In::col_at`, the combine door's RedStick
    # law); `o_desc` views the output as `[M, 1]` columns so each rank lands
    # in its own lane (the door's `rank_j · onehot_j` accumulate shape — the
    # store windows tile the view's rows, which is what the door's
    # `node_rows` law reads). ⛔ THE `regions()` WALK TAKES THE FIRST VIEW
    # PER PARAMETER, so `x_desc` (the full `[M, E]` view, declared first) is
    # what the door sees; `j_desc` (second, over the same base) exists for
    # this body alone.
    x_desc = tl.make_tensor_descriptor(desc_x, shape=[M, E], strides=[E, 1],
                                       block_shape=[M, E])
    j_desc = tl.make_tensor_descriptor(desc_x, shape=[M, E], strides=[E, 1],
                                       block_shape=[M, 1])
    o_desc = tl.make_tensor_descriptor(desc_o, shape=[M, E], strides=[E, 1],
                                       block_shape=[M, 1])
    # The sanitize: NaN → +inf (sorts last, the golden's NaN-as-greater). The
    # TENSOR is the left operand so the select is f16-typed and the `inf`
    # splat island-collapses to it (see the header's second law).
    x = x_desc.load([0, 0])
    xsan = tl.where(x == x, x, float("inf"))
    # The count form's ones — f16 splats from birth (`tl.full`), one per
    # select shape: the full-tile count and the per-column tie count.
    ones = tl.full([M, E], 1.0, tl.float16)
    ones_col = tl.full([M, 1], 1.0, tl.float16)
    for j in tl.static_range(0, E):
        # Ranked expert j's column, sanitized the same way — two NaNs tie by
        # index, so the tie compare must see +inf == +inf, not NaN == NaN.
        col = j_desc.load([0, j])
        colsan = tl.where(col == col, col, float("inf"))
        # |{h : xsan[h] < xsan[j]}| — the full-tile compare the door's
        # `lesserthan` states, reduced over the expert axis. The ones are
        # the LEFT operand (the f16 tensor); the 0.0 splat collapses to it.
        lt = tl.sum(tl.where(xsan < colsan, ones, 0.0), 1)
        # |{h < j : xsan[h] == xsan[j]}| — the stable tie-break: a running
        # count of prior equal columns, the door's identity-seeded
        # accumulator spelled as trace-time reassignment.
        t = tl.zeros([M, 1], dtype=tl.float16)
        for jj in tl.static_range(0, j):
            prior = j_desc.load([0, jj])
            psan = tl.where(prior == prior, prior, float("inf"))
            t += tl.where(psan == colsan, ones_col, 0.0)
        # rank_j lands in lane j of the output — an exact integer count,
        # already f16.
        o_desc.store([0, j], lt[:, None] + t)
