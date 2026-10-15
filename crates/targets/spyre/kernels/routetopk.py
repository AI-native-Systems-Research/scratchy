# SPDX-License-Identifier: Apache-2.0
"""Scratchy's Triton router top-k kernel — the vendor pair's selection, as one
descending insertion ripple over the raw logits.

BODY PROVENANCE: `topk.ddl`'s `topkindex`/`topkvalue` pair, mirrored op for op
(the k_loop's ripple at :164-198), designed against the door's contract
(`lower_ktir_to_superdsc::route_topk` — the fused chain that lands the pair's
`[slots, tokens]` outputs token-major). The door never walks these ops (it
emits the vendor pair's descriptors over the Kᵀ-restickified logits), so this
body states the same selection function the pair computes, for the ladder's
own executed proof (`triton-numeric`'s `routetopk.rs`):

    slot 0 = the maximum; slot j the (j+1)-th largest — DESCENDING, the
    pair's own order (`topk.ddl`'s output k_loop stores slot j's running
    register verbatim), NOT the old chain's ascending trailing slice.

THE RIPPLE, THE DDL'S OWN LAW (:168-198): K running `(value, index)` registers
initialized `(-inf, -1)`; per candidate `h` (the scan position — the index the
pair's counter mints), per slot `j`:

    state      = run_val[j] > cand          # STRICT GREATERTHAN
    run_val[j] = where(state, run_val[j], cand)      # the bigger stays
    run_idx[j] = where(state, run_idx[j], cand_h)
    cand       = where(state, cand, run_val_old[j])  # the displaced pair
    cand_h     = where(state, cand_h, run_idx_old[j])# ripples DOWN

Ties and NaN: `>` is FALSE, so the NEWCOMER takes the slot and the old pair
ripples down — the DDL's SELECT semantics, verbatim. The ids are scan
positions `0..E-1`, UNIQUE BY CONSTRUCTION (each candidate enters the ripple
exactly once) — the property whose absence (the argsort's dup ranks summing
expert ids past the bank) faulted the 26b's card FETCH.

⛔ THE K REGISTERS LIVE IN THE OUTPUT BUFFERS, AND THAT IS THE GRAMMAR'S OWN
LAW. The frontend's assignment targets are NAMES AND TUPLES ONLY (no
subscript store, no list mutation, no comprehension — `py/ast.rs`'s
`AssignTarget`), so a Python list of running tiles cannot be built or
updated. The descriptors can: `v_desc`/`o_desc` are random-access `[M, 1]`
column windows, so slot `j`'s register is `v_desc`'s column `j` — one init
store per slot, then the ripple's per-slot step reads the register back
(`v_desc.load([0, j])`), selects, and stores the survivor. The DDL's
`current_max`/`current_idx` internal tensors and this buffer round-trip are
the same K registers; the card never pays the round-trip (its pair keeps them
in SFP LRFs — the descriptor form the door emits).

THE SPLICE'S OWN CONTRACT (what `scratchy-triton-splice` states about this
kernel):

* PARAMETERS, IN ORDER: `desc_l`, `desc_v`, `desc_o` — the ROUTER CHAIN's
  binding order (the logits, the values output, the ids output), resolved by
  `bindings_for`'s RouteTopK walk: `desc_l` is the ARGSORT's producer (the
  RouterLogits output — the argsort and its rank vector are folded away),
  `desc_v` is the GATHER consumer's output (the gather is folded away too —
  this kernel IS its values), `desc_o` is the node's own `[m, k]` output.
  THREE pointer parameters where the node states ONE input: the fold re-points
  the top-k at the head of its own chain, and the values landing (the old
  gather's job) rides with it.
* CONSTEXPRS: `M`, `E`, `K` — stated by the splice from the walked regions:
  `M` the rows every view shares, `K` the outputs' cols, `E` the logits'
  cols, guarded `k ≤ E` and row-consistent.
* GRID: `[1]`.

⛔ LANE-FREE, THE FAMILY'S OWN MEASURED LAW (`routeargsort.py`'s header):
`tl.arange` is refused by the front end at non-power-of-two lengths, so every
tile here is an `[M, 1]` column window — one per candidate, one per slot —
and every loop is `tl.static_range` (trace-time ints, the `expertmatmul.py`
loop law).

⛔ EVERY VALUE IS f16 AND EVERY SELECT IS TILE-ON-TILE. The loads and the
`-inf`/`-1`/`h` initializers are f16 from birth (`tl.full` with
`tl.float16`), and every `tl.where` takes loaded tiles on BOTH sides — the
`LegalizeTypes` collapse law (SP-E2-03) and the left-operand typing law
`routeargsort.py`'s header states, both satisfied by construction here (no
f32 splat ever enters a select).

NUMERICS: the values are raw router logits (f16, the card's own dtype — the
card's pair compares them in SFP f16 too); the ids are integers below `E`
(exact in f16 to 2048; the real routers' `E = 128`). E·K ripple steps — the
same order as the old chain's O(K·E) column scan, an emulator-only cost the
card never pays: the card runs the vendor pair's own descriptor over the same
logits.
"""

import triton
import triton.language as tl


@triton.jit
def routetopk_fwd(desc_l, desc_v, desc_o,  #
                  M: tl.constexpr, E: tl.constexpr, K: tl.constexpr):
    # Descriptors name STORAGE (the door's own views): the `[M, E]` logits
    # re-viewed as `[M, 1]` columns (one candidate tile per scan position),
    # and the `[M, K]` values and ids as `[M, 1]` columns so each slot lands
    # in its own lane. ⛔ THE `regions()` WALK TAKES THE FIRST VIEW PER
    # PARAMETER: `l_desc`'s shape `[M, E]` is what the door reads as `E`, and
    # `v_desc`/`o_desc`'s `[M, K]` as `k` — the door's own extent laws.
    l_desc = tl.make_tensor_descriptor(desc_l, shape=[M, E], strides=[E, 1],
                                       block_shape=[M, 1])
    v_desc = tl.make_tensor_descriptor(desc_v, shape=[M, K], strides=[K, 1],
                                       block_shape=[M, 1])
    o_desc = tl.make_tensor_descriptor(desc_o, shape=[M, K], strides=[K, 1],
                                       block_shape=[M, 1])
    # The DDL's initialization (`:116-126`): every running register to
    # `(-inf, -1)` — only the first K candidates can meet an un-displaced
    # slot, and E ≥ K is the splice's own guard.
    for j in tl.static_range(0, K):
        v_desc.store([0, j], tl.full([M, 1], float("-inf"), tl.float16))
        o_desc.store([0, j], tl.full([M, 1], -1.0, tl.float16))
    for h in tl.static_range(0, E):
        # The candidate: scan position h's logit column, and its index as an
        # f16-from-birth constant tile (the pair's counter, minted here).
        cand = l_desc.load([0, h])
        cand_h = tl.full([M, 1], float(h), tl.float16)
        # The insertion ripple — slot 0 first (the running maximum), the
        # displaced pair passing down to the next slot, the DDL's k_loop. The
        # registers are read back BEFORE the survivor is stored, so the
        # displaced pair is the OLD one (the DDL's tmp_val/tmp_idx copies).
        for j in tl.static_range(0, K):
            run_val = v_desc.load([0, j])
            run_idx = o_desc.load([0, j])
            state = run_val > cand
            v_desc.store([0, j], tl.where(state, run_val, cand))
            o_desc.store([0, j], tl.where(state, run_idx, cand_h))
            # The displaced pair ripples down: at `state` the candidate
            # continues unchanged; otherwise the slot's OLD pair becomes the
            # candidate for slot j+1.
            cand = tl.where(state, cand, run_val)
            cand_h = tl.where(state, cand_h, run_idx)
