# SPDX-License-Identifier: Apache-2.0
"""PROBE-ONLY (front-b-noise, auditor order 2026-10-26): the on-card noise
hash STANDALONE — the sampler's noise section VERBATIM, its outputs exposed
as tensors instead of folded into the draw.

The production body (`crates/targets/spyre/kernels/sample_fwd.py`) computes
`u` and `g = -log(-log(u))` and consumes them in the same kernel; this probe
STORES them, so the card run can answer the statistical questions the
sampler cannot: the u-histogram's chi-square over 256 bins and g's mean and
variance against the Gumbel values (Euler-Mascheroni 0.5772…, π²/6
1.6449…). It also makes `u` bit-comparable on the host: every op up to and
including the uniform is exact by representability in wrapping i32 / IEEE
f32 (card leg B1 proved i32 mul/add bit-exact and int32tofp32 exact — the
wrap-fold's operands are multiples of 2^8 with ≤24 significant bits, exact
even at magnitude ~2^31), so a host recomputation of `u` checks the whole
chain — leaving the card's `log` as the only unemulable op, which is the
point.

FIDELITY: every op here is the production noise section's own — the seed
quad `(seed_lo, seed_hi, step, row)` (the probe drops only the greedy-gate
lane, which the hash never reads), the lane, the i32 mul/add mix with
squarings, the compare-free wrap-fold extraction (`·2^8` → int32tofp32 →
abs → folds — the device has no lane-wise integer compare, so the
conditional-subtract ladder it replaced cannot lower), the strict-interior
uniform, and `g` through the card's log. What is NOT here: the logits, the
temperature, the pad law, and the two-level ramp selection — not under
test.

EVERY CONSTANT IS A STAGED TABLE, the production kernel's own law
(`sample_fwd.py` stages `ones`/`sixtyfour`/`padoff` for exactly this
reason): a splatted constant routes its consumer to the scalar-multiply
door, whose registry and descriptor are fp16 — wrong for an f32 fold step
and unpronounceable for an i32 hash constant. So the hash constants
(C1/C2/C3/2^8), the seed quad, the lane, and the fold's five f32 constants
all ride loaded `[1, PAD_V]` tables, and every op in the kernel is the
dense two-tensor (or convert/reduce) form the whole-function door lowers.
The LANE is staged directly as an f32 table (0..PAD_V-1, exact below 2^24)
rather than reconstructed `g*4096 + b*64 + l` — the no-arange law's own
reading (a per-lane fact is host-staged), and it keeps the probe's f32
surface to the fold and the tail alone.

ONE PROGRAM, ONE WHOLE ROW: B=1, PAD_V=2^20 — 2^20 lanes in a single
`[1, 2^20]` row, the sampler's per-program whole-row width class (gemma-4's
PAD_V is 2^19; the probe is one power up). A B>1 grid is NOT this door's
shape: per-program `[1, PAD_V]` tiles over `[B, PAD_V]` views mismatch the
band/extent laws (the band cover equals the whole view, so the
intermediates never lift to the band), so the probe takes its lanes in
width, not in rows.

THE ISLAND'S EXITS, and why the probe has them at all: `LegalizeTypes`
(SP-E2-03) refuses f32 compute that does not exit through an `arith.fptosi`
-- genuine f32 is an owner decision, and the pass is right to refuse. The
production sampler's every f32 (score, noise, u, the constants) is legal
ONLY because it funnels into the selection's fptosi, the device's one f32
recipe (`dl16tofp32` -> same-stick f32 -> `fp32toint32`). A probe whose
u and g merely STORE would be genuine f32 by the pass's own definition, so
the probe closes the island the same way production does: a REAL
`fp32toint32` over each of u and g, its result stored as the probe's own
i32 outputs (`ui`, `gi` -- truncated toward zero, numerically inert, never
read; they exist so the pass sees the exit and so DCE cannot remove it).
No pass touched, nothing degraded, and both f32 components (the fold chain
behind u, the log chain behind g) exit through production's own mechanism.

The probe deliberately carries NO selection (no argmax): the walk's Reduce
arm cannot lower a wide f32 reduce -- one 2^20-lane f32 row is 4 MB of
per-core LX residency, the tiler time-tiles it, and the partials law
(#603's `tiled_trips_alias` guard) refuses the seeded form (the walk mints
the accum at its logical one-stick width, not the partials width). A wide
f32 reduce through this door needs the per-slice-plus-combine split (the
fp8 qspt path's own pattern) -- a production-selection problem, recorded
for the port-back, not a probe need: the auditor's statistics (the
u-histogram and g's moments) are host-side reads of the stored rows.

THE GUMBEL DRAW THROUGH THE CARD'S ONLY LOG — THE FP16 FEST BIND. The vendor
DDLs bind `log` to `[%type_fp16]` ALONE (unary_parallel.ddl — FEST-based,
like fp16 tanh/gelu, and unlike them carrying no fp32 binding), so an f32
`tl.log` can never map: dxp refuses the op ("Scheduler failed to find a
suitable op mapping" — measured on this probe's own card leg). But Triton's
`tl.log` refuses fp16 operands (upstream `@_check_dtype`), so the kernel
cannot spell the fp16 log directly either. The way through is the shape
`LegalizeTypes` exists to collapse: each log is written as the
stability-widening island `tl.log(x16.to(tl.float32)).to(tl.float16)`, and
the pass removes the extf/truncf pair and retypes the log to its operand's
fp16 — the emitted op is the one fp16 `log` bind the device carries. The
collapsed chain:

    u16 = u.to(tl.float16)         # fp32todl16 (the narrow-back convert)
    c1  = max(u16, 2^-14)          # subnormal/FTZ guard (fp16 maximum)
    lu  = log(c1)                  # the FEST log, fp16 after the collapse
    v   = -lu                      # fp16 neg — an exact sign-flip
    c2  = max(v, 2^-14)            # second guard
    lt  = log(c2)                  # the FEST log, fp16
    g   = -lt.to(tl.float32)       # neg + the protected dl16tofp32 widen

PRECISION, and why nothing here degrades the moments: an fp16 value's
rounding error is RELATIVE (~2^-11), the negations are exact sign-flips, and
the clamps are exact, so the only error the logs leave is ~2^-11 RELATIVE on
each log's output — which g = -log(-log(u)) turns into a ~2^-11·|log(v)|
ABSOLUTE error on g, at most ~1e-3 per lane and uniformly spread, invisible
at the 2^20-lane moments (σ_mean ≈ 1.25e-3 dwarfs the averaged contribution).
What is deliberately NOT done: any f32 arithmetic between the logs that
cancels two same-magnitude fp16 values (e.g. the rescaled `12·ln2 -
log(u·2^12)` with its subtraction kept in fp16) — the rescale puts every
inner log output at magnitude ~8.3 where the fp16 ULP is ~0.004 ABSOLUTE,
and cancelling that against 12·ln2 leaves v with a 0.004 ABSOLUTE error that
blows up as dv/v at small v (~0.4% of lanes poisoned, v ≤ 0 → NaN). The
clamps at 2^-14 guard the FEST log's subnormal-input behavior (whether it
flushes is a vendor fact the probe does not assume): the inner clamp fires
on ~50 lanes of 2^20 (u < 2^-14), the outer on ~270 — fp16 rounds the top
2^-12 of u to exactly 1.0, so log(1) = 0 and v = 0 there (measured on the
host emulation; the stats script reports both counts). Pinning those lanes
to the boundary biases the mean by < 1e-3 and the variance by < 0.003 —
both under the 2^20-lane sampling noise, and both reported by the stats
script as clamped-lane counts, never gated.

PARAMETERS, IN ORDER: the seed quad `slo`, `shi`, `stp`, `srow` and the
hash constants `c1`, `c2`, `c3`, `c256` (all `[1, PAD_V]` i32, the seed
quad's value duplicated down every lane -- one program, one seed); the f32
tables `lane`, `fm8` (2^-8), `ftop` (8388607.0), `fzero` (0.0), `fhalf`
(0.5), `fm22` (2^-22), and the fp16 table `cmin` (2^-14 — the FEST-domain
clamp, fp16 so the maximum never leaves the fp16 stick); then the four outs
`u`, `g` (`[1, PAD_V]` f32) and `ui`, `gi` (`[1, PAD_V]` i32, the island's
exits).
CONSTEXPRS: `PAD_V` (lanes per row), `B` (the grid's width, always 1
here). GRID: `[1]`.
"""

import triton
import triton.language as tl


@triton.jit
def hash_noise_probe(slo_ptr, shi_ptr, stp_ptr, srow_ptr,  #
                     c1_ptr, c2_ptr, c3_ptr, c256_ptr,  #
                     lane_ptr, fm8_ptr, ftop_ptr, fzero_ptr, fhalf_ptr, fm22_ptr,  #
                     cmin_ptr,  #
                     u_ptr, g_ptr, ui_ptr, gi_ptr,  #
                     PAD_V: tl.constexpr, B: tl.constexpr):
    # THE STAGED TABLES, all [1, PAD_V] whole-row blocks at corner (0, 0) --
    # the shared-table idiom inside a one-program grid. Every per-lane fact
    # is host-staged (the no-arange law): the seed quad, the hash constants,
    # the lane, and the fold's constants. Nothing is a splat, so nothing
    # routes to the fp16 scalar-multiply door.
    slo_d = tl.make_tensor_descriptor(slo_ptr, shape=[1, PAD_V], strides=[PAD_V, 1],
                                      block_shape=[1, PAD_V])
    slo = slo_d.load([0, 0])
    shi_d = tl.make_tensor_descriptor(shi_ptr, shape=[1, PAD_V], strides=[PAD_V, 1],
                                      block_shape=[1, PAD_V])
    shi = shi_d.load([0, 0])
    stp_d = tl.make_tensor_descriptor(stp_ptr, shape=[1, PAD_V], strides=[PAD_V, 1],
                                      block_shape=[1, PAD_V])
    stp = stp_d.load([0, 0])
    srow_d = tl.make_tensor_descriptor(srow_ptr, shape=[1, PAD_V], strides=[PAD_V, 1],
                                       block_shape=[1, PAD_V])
    srow = srow_d.load([0, 0])
    c1_d = tl.make_tensor_descriptor(c1_ptr, shape=[1, PAD_V], strides=[PAD_V, 1],
                                     block_shape=[1, PAD_V])
    c1 = c1_d.load([0, 0])
    c2_d = tl.make_tensor_descriptor(c2_ptr, shape=[1, PAD_V], strides=[PAD_V, 1],
                                     block_shape=[1, PAD_V])
    c2 = c2_d.load([0, 0])
    c3_d = tl.make_tensor_descriptor(c3_ptr, shape=[1, PAD_V], strides=[PAD_V, 1],
                                     block_shape=[1, PAD_V])
    c3 = c3_d.load([0, 0])
    c256_d = tl.make_tensor_descriptor(c256_ptr, shape=[1, PAD_V], strides=[PAD_V, 1],
                                       block_shape=[1, PAD_V])
    c256 = c256_d.load([0, 0])
    # THE LANE, staged as an f32 table (0..PAD_V-1, exact in f32 below 2^24)
    # and ingested by the index-cast law: the f32 value is exact, so the
    # fptosi recovers the lane bit for bit.
    lane_d = tl.make_tensor_descriptor(lane_ptr, shape=[1, PAD_V], strides=[PAD_V, 1],
                                       block_shape=[1, PAD_V])
    lanef = lane_d.load([0, 0])
    lane = lanef.to(tl.int32)
    # THE FOLD'S CONSTANTS, staged f32 rows: 2^-8 (the window scale),
    # 8388607.0 (2^23 - 1, the reflect), 0.0 (the lower clamp), 0.5 and
    # 2^-22 (the strict-interior shift and scale).
    fm8_d = tl.make_tensor_descriptor(fm8_ptr, shape=[1, PAD_V], strides=[PAD_V, 1],
                                      block_shape=[1, PAD_V])
    fm8 = fm8_d.load([0, 0])
    ftop_d = tl.make_tensor_descriptor(ftop_ptr, shape=[1, PAD_V], strides=[PAD_V, 1],
                                       block_shape=[1, PAD_V])
    ftop = ftop_d.load([0, 0])
    fzero_d = tl.make_tensor_descriptor(fzero_ptr, shape=[1, PAD_V], strides=[PAD_V, 1],
                                        block_shape=[1, PAD_V])
    fzero = fzero_d.load([0, 0])
    fhalf_d = tl.make_tensor_descriptor(fhalf_ptr, shape=[1, PAD_V], strides=[PAD_V, 1],
                                        block_shape=[1, PAD_V])
    fhalf = fhalf_d.load([0, 0])
    fm22_d = tl.make_tensor_descriptor(fm22_ptr, shape=[1, PAD_V], strides=[PAD_V, 1],
                                       block_shape=[1, PAD_V])
    fm22 = fm22_d.load([0, 0])
    # THE FEST-DOMAIN CLAMP, staged as an fp16 row (2^-14, fp16's smallest
    # normal): both log inputs are guarded with it, so neither the fp32todl16
    # convert's subnormal output nor the FEST log's subnormal-input behavior
    # (flush or not — a vendor fact the probe does not assume) can put a zero
    # or a flush into a log. fp16 so the `maximum` stays on the fp16 stick.
    cmin_d = tl.make_tensor_descriptor(cmin_ptr, shape=[1, PAD_V], strides=[PAD_V, 1],
                                       block_shape=[1, PAD_V])
    cmin = cmin_d.load([0, 0])

    # THE COUNTER HASH -- i32 mul/add ONLY, wrapping (the card's bit-exact
    # integer class, card leg B1: muli32toi32/addi32toi32 over SENUINT32).
    # Key: the row's seed quad and the lane. Multiplication by an odd
    # constant is a bijection mod 2^32 and each square breaks the linear
    # structure an xor-shift mix would break on a richer vocabulary; the
    # constants are the staged spellings of 0x9E3779B1 / 0x85EBCA77 /
    # 0xC2B2AE3D.
    h = lane * c1 + slo
    h = h * c2 + shi + stp + srow
    h = h * h
    h = h * c1 + slo
    h = h * h
    h = h * c3 + shi + stp
    h = h * h
    # ⭐ THE LANE RE-INJECTION, and why the hash does not end on the square:
    # a square's low bits are structurally non-uniform (x² mod 8 ∈ {0, 1, 4},
    # and every mul/add finalizer is a BIJECTION mod 8, so no chain of them
    # can heal it). Adding a term that varies per lane AFTER the last square
    # does heal it -- the sum of any distribution with a uniform is uniform,
    # at every bit position the varying term reaches. One mul-add, and the
    # low-bit defect is gone by construction rather than by measurement.
    h = h * c1 + lane * c2 + slo
    # ⭐ EXTRACT THE LOW 24 BITS WITHOUT A COMPARE. The device has no
    # lane-wise integer compare (the attention fixtures' own law -- their
    # causal masks are host-prepared for exactly this reason), so the
    # conditional-subtract ladder cannot lower. The wrap-fold needs no
    # compare at all: multiply by 2^8 in wrapping i32 -- r = h·2^8 mod 2^32
    # = (h mod 2^24)·2^8, the low 24 bits shifted to the top, the high 8
    # dropped. Every attainable r is a multiple of 2^8 with at most 24
    # significant bits, so int32tofp32 is EXACT on it even at magnitude
    # ~2^31 (card leg B1: exact below 2^24, ties-up above -- and an exactly
    # representable value rounds to itself under any nearest-mode).
    r = h * c256
    rf = r.to(tl.float32)
    # THE ABS FOLD: bit 31 of r is bit 23 of h, so r is negative exactly on
    # the upper half of the 24-bit window, and |r| = L·2^8 there becomes
    # (2^24 - L)·2^8 -- a FOLD of the window, 2-to-1 on the interior. A fold
    # of a uniform is uniform, and the sign case's magnitude (2^24 - L)·2^8
    # has at most 23 significant bits, so it too converts exactly.
    a = tl.abs(rf)
    n = a * fm8
    # THE STRICT-INTERIOR FOLDS: n lands in [0, 2^23] with the top point at
    # weight 2^-24 (L = 2^23 exactly), and u = n·2^-23 + half-ulp would sit
    # ABOVE 1 there -- log(u) > 0, -log(u) < 0, log of a negative: NaN. Two
    # folds close it: min against (2^23 - 1) - n reflects the upper half
    # (the -1 keeps the fixed point at 2^22 - 1, strictly below 2^22), and
    # the max against 0 catches the one negative the -1 lets through at
    # n = 2^23. Both are card elementwise ops (`minimum`, `maximum`), both
    # operands are exact integers, and the fold of a uniform is uniform --
    # every grid point ends at full weight including the ends.
    t = ftop - n
    nm = tl.minimum(n, t)
    np = tl.maximum(nm, fzero)
    # THE UNIFORM, STRICTLY INSIDE (0, 1): (np + 0.5)·2^-22 over
    # np ∈ [0, 2^22 - 1] -- every value an exact f32 (np + 0.5 has at most
    # 23 significant bits), every grid point an odd multiple of 2^-23, off
    # both singularity ends of the Gumbel transform. Every op so far is
    # exact by representability, so u is bit-emulable on the host given the
    # seed quad -- the probe's free bit-check of the whole integer chain.
    u = (np + fhalf) * fm22
    # THE GUMBEL DRAW through the card's ONLY log — the fp16 FEST bind, reached
    # through `LegalizeTypes`' own widening-island collapse. Triton's `tl.log`
    # refuses fp16 operands (the frontend's `@_check_dtype`, same as upstream), so
    # each log is spelled as the stability-widening island —
    # `tl.log(x16.to(tl.float32)).to(tl.float16)` — which the pass collapses to a
    # BARE fp16 log (extf removed, result retyped, truncf removed): the emitted op
    # is `log` at fp16, the one bind the vendor DDL carries. The arithmetic
    # between the logs is fp16 negation — an exact sign-flip, no rounding — and
    # the clamps at 2^-14 guard the FEST log's subnormal-input behavior on both
    # entries (see the docstring's precision section). The final widen
    # `g16.to(tl.float32)` is the exit island's own protected funnel (f16 must
    # stage through `dl16tofp32` on the way to an integer — the DDL refuses the
    # direct fp16→u32 pairing), so it survives the collapse by the pass's own
    # protection law and `g` lands as f32 for the fptosi exit.
    u16 = u.to(tl.float16)
    c1 = tl.maximum(u16, cmin)
    lu = tl.log(c1.to(tl.float32)).to(tl.float16)
    v = -lu
    c2 = tl.maximum(v, cmin)
    lt = tl.log(c2.to(tl.float32)).to(tl.float16)
    g16 = -lt
    g = g16.to(tl.float32)
    # THE ISLAND'S EXITS — production's own funnel (see the docstring): a real
    # `fp32toint32` over each f32 row, stored as the probe's i32 outputs so
    # `LegalizeTypes` sees the exit and DCE cannot remove it. The integer
    # values are truncated toward zero and never read; the u and g rows above
    # are the probe's data.
    ui = u.to(tl.int32)
    gi = g.to(tl.int32)
    # THE FOUR OUTPUTS: u and g as the [1, PAD_V] f32 rows (u bit-checkable
    # on the host, g the card's log in the loop), ui and gi as the island's
    # [1, PAD_V] i32 exits.
    u_desc = tl.make_tensor_descriptor(u_ptr, shape=[B, PAD_V], strides=[PAD_V, 1],
                                       block_shape=[1, PAD_V])
    u_desc.store([0, 0], u)
    g_desc = tl.make_tensor_descriptor(g_ptr, shape=[B, PAD_V], strides=[PAD_V, 1],
                                       block_shape=[1, PAD_V])
    g_desc.store([0, 0], g)
    ui_desc = tl.make_tensor_descriptor(ui_ptr, shape=[B, PAD_V], strides=[PAD_V, 1],
                                        block_shape=[1, PAD_V])
    ui_desc.store([0, 0], ui)
    gi_desc = tl.make_tensor_descriptor(gi_ptr, shape=[B, PAD_V], strides=[PAD_V, 1],
                                        block_shape=[1, PAD_V])
    gi_desc.store([0, 0], gi)
