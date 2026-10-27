# SPDX-License-Identifier: Apache-2.0
"""PROBE-ONLY (front-b-noise, auditor order 2026-10-27, ruling (a)): the
PRODUCTION NOISE ISLAND at the batched rung's own rows, its `g` and `score`
read OFF THE CARD — the sampler's noise-and-score section of
`sample_fwd.py` VERBATIM (op for op, constant for constant, staging class for
staging class), the selection DROPPED (the draw's argmax is the chat leg's own
measurement; a wide f32 reduce through this door is the recorded
production-selection problem, not a probe need — `hash_noise_probe`'s own
law), and the three f32 values the auditor named STORED as outputs: `u`
(bit-exact on the host — the whole integer chain's free check), `g` (the
Gumbel draw) and `score` (the draw's input to the selection).

THE DEFECT THIS PROBE MEASURES: at `rows > 1` the ten uniform `[rows, 1]`
tables (the four i32 hash/fold constants, the five f32 fold constants, the
fp16 FEST clamp) were staged DENSE by the bake — `rows` words against a
placement that reserves a whole stick footprint per row — so every row past 0
read ZERO at its stick head. Zero multipliers collapse the hash to a
lane-CONSTANT, `u` goes to a constant, `noise` goes to a constant, and
`argmax(x32 * invt + const) = argmax(x32)` — the pick-equals-argmax signature
the chat leg measured. The fix (`d99c3d750f`) stages the constant at EVERY
row's stick head; this probe reads the noise the card actually computes at
every row, so a still-wrong head lands on a per-row, per-lane mismatch —
`u`'s bit-check names the first wrong (row, lane, card word, host word), and
a still-constant noise shows as `argmax(score) == argmax(x)` on every row
with near-tie logits staged for exactly that question.

FIDELITY — every op is the production noise section's own, verbatim from
`sample_fwd.py`: the fp16 logits row loaded `[1, PAD_V]` at `[pid, 0]` of a
`[B, PAD_V]` view and widened exactly; the per-row temperature reciprocal and
the seed QUINTUPLE `(seed_lo, seed_hi, step, greedy_gate, row)` as FIVE
`[B, 1]` i32 tables read through `[1, 1]` tiles at `[pid, 0]` (the stick
head); the lane as its own `[1, PAD_V]` f32 table (the no-arange law); the
ten uniform constants in the seeds' OWN class — `[B, 1]` tables at the rung's
own row count, the same word at every row's stick head, broadcast by their
consuming ops (the 2026-10-27 ruling); the counter hash in wrapping i32; the
compare-free wrap-fold; the strict-interior uniform; the Gumbel draw through
the card's only log (the fp16 FEST bind via `LegalizeTypes`' own
widening-island collapse, both entries clamped at 2^-14); `gate` from the
quintuple's greedy_gate word; the three-rounding score; and the pad law
applied last (`padoff`, 0.0 on the real lanes, -FLT_MAX on the pads). What is
NOT here: the two-level ramp selection, the one-hot, and the idx stick.

THE STAGED `sgate` IS 1 ON EVERY ROW — the SAMPLED spelling, the defect's own
case (a greedy row's gate 0 makes `score = x32` exactly, and pick == argmax is
then CORRECT, not a defect; the solo chat-leg readback line that showed
`eq=true` was the greedy stage).

THE ISLAND'S EXITS, `hash_noise_probe`'s own law: `LegalizeTypes` refuses f32
compute that does not exit through an `arith.fptosi`, and a merely-STORED f32
is genuine f32 by the pass's own definition — so each of the three stored f32
rows (`u`, `g`, `score`) closes through a REAL `fp32toint32` whose result is
stored as the probe's own i32 outputs (`ui`, `gi`, `si` — truncated toward
zero, numerically inert, read only as a liveness witness). No pass touched,
nothing degraded, and every f32 component exits through production's own
mechanism.

PARAMETERS, IN ORDER: the fp16 `[B, PAD_V]` logits row; the quintuple's five
`[B, 1]` i32 word tables; the `[B, 1]` f32 temperature reciprocals; the
`[B, 1]` f32 ones word (the score terminal's widen — see the body); the
`[1, PAD_V]` f32 pad-law offset row; the four `[B, 1]` i32 hash/fold
constants; the `[1, PAD_V]` f32 lane table; the five `[B, 1]` f32 fold
constants; the `[B, 1]` fp16 FEST clamp; then the six outs — `u`, `g`,
`score` (`[B, PAD_V]` f32) and `ui`, `gi`, `si` (`[B, PAD_V]` i32, the
island's exits). CONSTEXPRS: `PAD_V` (the padded row) and `B` (the grid's
width — 2 the intmr-proven control width, 8 the batched defect geometry; at
B=1 the `[B, 1]` views are `[1, 1]`, which this door sizes as a 2 B
launcher-bound scalar, so the solo case is the chat leg's own seat readback,
not this probe's). GRID: `[B]`.

THE VERDICT is the stats script's (`sn_stats.py`): `u` BIT-EXACT per row and
lane against the host emulation given the staged seed words (every op through
`u` is exact by representability — a wrong stick-head word, a wrong row's
seed, a mis-staged constant all land on a named mismatch); `g` non-constant
per row and its moments against the Gumbel values; `score`'s per-row argmax
against the logits row's own (near-tie logits staged so a LIVE noise moves it
— a constant noise cannot); and the per-lane `g`/`score` diffs against the
host fp16/f32 emulation (the FEST log is the only unemulable op, ~2^-11
relative — reported, never gated bit-for-bit).
"""

import triton
import triton.language as tl


@triton.jit
def sample_noise_probe(logits_ptr, slo_ptr, shi_ptr, stp_ptr, sgate_ptr, srow_ptr,  #
                       invt_ptr, ones_ptr, padoff_ptr,  #
                       hc1_ptr, hc2_ptr, hc3_ptr, c256_ptr,  #
                       lane_ptr, fm8_ptr, ftop_ptr, fzero_ptr, fhalf_ptr, fm22_ptr,  #
                       cmin_ptr,  #
                       u_ptr, g_ptr, score_ptr, ui_ptr, gi_ptr, si_ptr,  #
                       PAD_V: tl.constexpr, B: tl.constexpr):
    # ONE PROGRAM PER REQUEST ROW — `sample_fwd.py`'s own grid law: the grid is
    # [B], every shape rank 2 with a one-row block, the per-row operands at
    # [pid, 0], the shared tables at [0, 0]. No loops, no scalars, no splats.
    pid = tl.program_id(0)
    # THE LOGITS ROW, the card's f16 lm-head output, widened exactly (every f16
    # value is exactly representable in f32) — the score's left term.
    l_desc = tl.make_tensor_descriptor(logits_ptr, shape=[B, PAD_V], strides=[PAD_V, 1],
                                       block_shape=[1, PAD_V])
    x = l_desc.load([pid, 0])
    x32 = x.to(tl.float32)
    # THE TEMPERATURE'S RECIPROCAL, ONE PER ROW — the [B, 1] f32 table at this
    # program's own row, broadcast over the whole padded row by the multiply.
    i_desc = tl.make_tensor_descriptor(invt_ptr, shape=[B, 1], strides=[1, 1],
                                       block_shape=[1, 1])
    invt = i_desc.load([pid, 0])
    # THE SEED WORDS, ONE TABLE PER MEMBER OF THE QUINTUPLE — five [B, 1] i32
    # tables, each loaded corner-free at this program's own row (`sample_fwd`'s
    # own spelling; the host writes one word per row per table, every step).
    slo_d = tl.make_tensor_descriptor(slo_ptr, shape=[B, 1], strides=[1, 1],
                                      block_shape=[1, 1])
    slo = slo_d.load([pid, 0])
    shi_d = tl.make_tensor_descriptor(shi_ptr, shape=[B, 1], strides=[1, 1],
                                      block_shape=[1, 1])
    shi = shi_d.load([pid, 0])
    stp_d = tl.make_tensor_descriptor(stp_ptr, shape=[B, 1], strides=[1, 1],
                                      block_shape=[1, 1])
    stp = stp_d.load([pid, 0])
    sgate_d = tl.make_tensor_descriptor(sgate_ptr, shape=[B, 1], strides=[1, 1],
                                        block_shape=[1, 1])
    sgate = sgate_d.load([pid, 0])
    srow_d = tl.make_tensor_descriptor(srow_ptr, shape=[B, 1], strides=[1, 1],
                                       block_shape=[1, 1])
    srow = srow_d.load([pid, 0])
    # THE LANE'S OWN INDEX, staged as its own f32 table (0..PAD_V-1, exact in
    # f32 below 2^24) and ingested by the index-cast law — the no-arange law's
    # own reading, the shared row every program reads.
    la_desc = tl.make_tensor_descriptor(lane_ptr, shape=[1, PAD_V], strides=[PAD_V, 1],
                                        block_shape=[1, PAD_V])
    lanef = la_desc.load([0, 0])
    lane = lanef.to(tl.int32)
    # THE NOISE SECTION'S CONSTANTS — [B, 1] tables, the SEED WORDS' OWN CLASS
    # (the auditor's 2026-10-27 ruling): a [1, 1] block loaded at [pid, 0] —
    # this row's stick head, exactly where a seed word sits — and broadcast by
    # its consuming op, never materialized at row width. The hash multipliers
    # ride the int-bcast operand mode; the f32 constants the same broadcast
    # class `invt` itself uses; the fp16 clamp stays on the fp16 stick.
    hc1_d = tl.make_tensor_descriptor(hc1_ptr, shape=[B, 1], strides=[1, 1],
                                      block_shape=[1, 1])
    hc1 = hc1_d.load([pid, 0])
    hc2_d = tl.make_tensor_descriptor(hc2_ptr, shape=[B, 1], strides=[1, 1],
                                      block_shape=[1, 1])
    hc2 = hc2_d.load([pid, 0])
    hc3_d = tl.make_tensor_descriptor(hc3_ptr, shape=[B, 1], strides=[1, 1],
                                      block_shape=[1, 1])
    hc3 = hc3_d.load([pid, 0])
    c256_d = tl.make_tensor_descriptor(c256_ptr, shape=[B, 1], strides=[1, 1],
                                       block_shape=[1, 1])
    c256 = c256_d.load([pid, 0])
    fm8_d = tl.make_tensor_descriptor(fm8_ptr, shape=[B, 1], strides=[1, 1],
                                      block_shape=[1, 1])
    fm8 = fm8_d.load([pid, 0])
    ftop_d = tl.make_tensor_descriptor(ftop_ptr, shape=[B, 1], strides=[1, 1],
                                       block_shape=[1, 1])
    ftop = ftop_d.load([pid, 0])
    fzero_d = tl.make_tensor_descriptor(fzero_ptr, shape=[B, 1], strides=[1, 1],
                                        block_shape=[1, 1])
    fzero = fzero_d.load([pid, 0])
    fhalf_d = tl.make_tensor_descriptor(fhalf_ptr, shape=[B, 1], strides=[1, 1],
                                        block_shape=[1, 1])
    fhalf = fhalf_d.load([pid, 0])
    fm22_d = tl.make_tensor_descriptor(fm22_ptr, shape=[B, 1], strides=[1, 1],
                                      block_shape=[1, 1])
    fm22 = fm22_d.load([pid, 0])
    cmin_d = tl.make_tensor_descriptor(cmin_ptr, shape=[B, 1], strides=[1, 1],
                                       block_shape=[1, 1])
    cmin = cmin_d.load([pid, 0])

    # THE COUNTER HASH — i32 mul/add ONLY, wrapping (the card's bit-exact
    # integer class, card leg B1). Key: the row's seed quintuple and the lane.
    h = lane * hc1 + slo
    h = h * hc2 + shi + stp + srow
    h = h * h
    h = h * hc1 + slo
    h = h * h
    h = h * hc3 + shi + stp
    h = h * h
    # THE LANE RE-INJECTION — heals the square's low-bit non-uniformity by
    # construction (`sample_fwd.py`'s own comment).
    h = h * hc1 + lane * hc2 + slo
    # EXTRACT THE LOW 24 BITS WITHOUT A COMPARE: multiply by 2^8 in wrapping
    # i32, then int32tofp32 (exact — every attainable r is a multiple of 2^8
    # with at most 24 significant bits).
    r = h * c256
    rf = r.to(tl.float32)
    # THE ABS FOLD: bit 31 of r is bit 23 of h — a 2-to-1 fold of the window.
    a = tl.abs(rf)
    n = a * fm8
    # THE STRICT-INTERIOR FOLDS: min against (2^23 - 1) - n, max against 0.
    t = ftop - n
    nm = tl.minimum(n, t)
    np = tl.maximum(nm, fzero)
    # THE UNIFORM, STRICTLY INSIDE (0, 1): (np + 0.5) * 2^-22 — every value an
    # exact f32, so u is bit-emulable on the host given the seed quintuple.
    u = (np + fhalf) * fm22
    # THE GUMBEL DRAW through the card's ONLY log — the fp16 FEST bind, via
    # `LegalizeTypes`' own widening-island collapse; the arithmetic between
    # the logs is fp16 negation (exact sign-flips), the clamps at 2^-14 guard
    # the FEST log's subnormal-input behavior on both entries, and the final
    # widen is the exit island's own protected funnel.
    u16 = u.to(tl.float16)
    c1 = tl.maximum(u16, cmin)
    lu = tl.log(c1.to(tl.float32)).to(tl.float16)
    v = -lu
    c2 = tl.maximum(v, cmin)
    lt = tl.log(c2.to(tl.float32)).to(tl.float16)
    g16 = -lt
    noise = g16.to(tl.float32)
    # score = x32 * invt + noise * gate, THREE roundings before the sum — the
    # exact sequence a reference must state; nothing fuses mul+add into an FMA.
    gate = sgate.to(tl.float32)
    score = x32 * invt + noise * gate
    # THE PAD LAW, APPLIED LAST: 0.0 on the real lanes (the rounding sequence
    # exactly unchanged), f32::MIN on the pads — never IEEE -inf (an
    # exponent-all-ones lane reads as NaN on the card).
    p_desc = tl.make_tensor_descriptor(padoff_ptr, shape=[1, PAD_V], strides=[PAD_V, 1],
                                       block_shape=[1, PAD_V])
    padoff = p_desc.load([0, 0])
    score = score + padoff
    # THE TERMINAL WIDEN through a loaded `[B, 1]` ones word — production's own
    # ones-table law (`sample_fwd.py`'s `g64 = gstar * ones`, staged for exactly
    # this door-shape reason): the whole-function door's row-tiled family fold
    # demands a CONSTANT row corner for an all-dense pointwise terminal that is
    # STORED, and the sampler's `[pid, 0]` corner is not one at `B > 1`; a
    # terminal whose operand is a `[1, 1]` broadcast word is gated out of the
    # fold and lowers per-op (the intmr cells' own measured form). `x * 1.0` is
    # EXACT — the pad law's rounding sequence is unchanged bit for bit.
    o_desc = tl.make_tensor_descriptor(ones_ptr, shape=[B, 1], strides=[1, 1],
                                       block_shape=[1, 1])
    ones = o_desc.load([pid, 0])
    score = score * ones
    # THE ISLAND'S EXITS — a real fp32toint32 over each stored f32 row
    # (`hash_noise_probe`'s own law): the values truncate toward zero and are
    # read only as a liveness witness; the f32 rows above are the probe's data.
    ui = u.to(tl.int32)
    gi = noise.to(tl.int32)
    si = score.to(tl.int32)
    # THE SIX OUTPUTS, each this program's own [1, PAD_V] row of a [B, PAD_V]
    # view — u and g and score as f32, the exits as i32.
    u_desc = tl.make_tensor_descriptor(u_ptr, shape=[B, PAD_V], strides=[PAD_V, 1],
                                       block_shape=[1, PAD_V])
    u_desc.store([pid, 0], u)
    g_desc = tl.make_tensor_descriptor(g_ptr, shape=[B, PAD_V], strides=[PAD_V, 1],
                                       block_shape=[1, PAD_V])
    g_desc.store([pid, 0], noise)
    s_desc = tl.make_tensor_descriptor(score_ptr, shape=[B, PAD_V], strides=[PAD_V, 1],
                                       block_shape=[1, PAD_V])
    s_desc.store([pid, 0], score)
    ui_desc = tl.make_tensor_descriptor(ui_ptr, shape=[B, PAD_V], strides=[PAD_V, 1],
                                        block_shape=[1, PAD_V])
    ui_desc.store([pid, 0], ui)
    gi_desc = tl.make_tensor_descriptor(gi_ptr, shape=[B, PAD_V], strides=[PAD_V, 1],
                                        block_shape=[1, PAD_V])
    gi_desc.store([pid, 0], gi)
    si_desc = tl.make_tensor_descriptor(si_ptr, shape=[B, PAD_V], strides=[PAD_V, 1],
                                        block_shape=[1, PAD_V])
    si_desc.store([pid, 0], si)
