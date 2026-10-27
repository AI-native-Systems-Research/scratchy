# SPDX-License-Identifier: Apache-2.0
"""Scratchy's Triton sampling kernel — the token draw, Gumbel-max over the
padded row, FIRST-MAX by arithmetic, as one kernel per request row.

BODY PROVENANCE: #813's `kernels/argmax.py` (the whole-row equality-selection
body that fixture baked through this repo's KTIR→SuperDSC lowering and that
the triton-numeric legs pinned against a derived first-max reference), with
three deltas the sampler needs: the grid is `[B]` (one program per request
row — the solo step is `B = 1`, a batched step `B = 2..32`), the selection
runs IN F32 (the score is a sum `x·inv_t + g`; f16's 10-bit mantissa would
round near-ties into ties the reference never sees), and the per-row seed
table and the temperature's reciprocal join the row (the noise itself is
computed on the card, below).

THE DRAW: `idx = first-max over (logits·inv_T + g)` with `g ~ Gumbel(0, 1)`,
which samples exactly `softmax(logits/T)`. THE NOISE IS COMPUTED ON THE CARD:
a counter hash over `(seed_lo, seed_hi, step, row, lane)` in the card's
bit-exact integer class — i32 mul/add ONLY (card leg B1; the vocabulary has
no xor and no shift, `arch_enums.cpp` `opFuncsToString`: 180 entries, neither
op present — so the mix is odd-constant multiplies and squarings, built to
never need them), closed by a LANE RE-INJECTION after the last square (a
square's low bits are structurally non-uniform — x² mod 8 ∈ {0, 1, 4}, and
every mul/add finalizer is a bijection mod 8, so no chain of them heals it;
a per-lane term after the square does, by construction). The hash lands a
uniform `u` STRICTLY INSIDE `(0, 1)` by the WRAP-FOLD, the compare-free
extraction the front-b probe proved bit-exact on card (0/2²⁰ lanes):
multiply by 2⁸ in wrapping i32 — `r = h·2⁸ mod 2³² = (h mod 2²⁴)·2⁸`, the
low 24 bits shifted to the top, every attainable `r` a multiple of 2⁸ with
at most 24 significant bits, so int32tofp32 is EXACT on it even at
magnitude ~2³¹ — then fold the sign (`|r|`: bit 31 of `r` is bit 23 of
`h`, a 2-to-1 fold of the window; a fold of a uniform is uniform), scale
`n = |r|·2⁻⁸ ∈ [0, 2²³]`, and close both ends with the two
strict-interior folds — `min(n, (2²³−1)−n)` reflects the upper half (the
−1 keeps the fixed point strictly below 2²²) and `max(·, 0)` catches the
one negative the −1 lets through — both card elementwise ops
(`minimum`/`maximum`), needed because the device has NO lane-wise integer
compare, so the conditional-subtract ladder this replaces cannot lower.
`u = (np + 0.5)·2⁻²²` over `np ∈ [0, 2²²−1]`: every grid point an odd
multiple of 2⁻²³, exact in f32, off both singularity ends of the Gumbel
transform, and every op up to and including `u` is exact by
representability — the whole integer chain is host bit-emulable given the
seed table (the probe's free bit-check).

`g = -log(-log(u))` through the card's ONLY log — the FP16 FEST bind: the
vendor DDLs bind `log` to `[%type_fp16]` ALONE (unlike fp16 tanh/gelu it
carries no fp32 binding, so an f32 `tl.log` can never map — dxp refuses
the op; measured on the front-b probe's own card leg), and Triton's
`tl.log` refuses fp16 operands, so each log is spelled as the
stability-widening island `tl.log(x16.to(tl.float32)).to(tl.float16)`,
which LegalizeTypes collapses to the BARE fp16 log. The clamps at 2⁻¹⁴
(fp16's smallest normal, the staged `cmin` table) guard subnormal/FTZ on
both log entries, and the arithmetic between the logs is fp16 negation —
an exact sign-flip — so the island's error stays RELATIVE (~5e-4 absolute
on g; the probe measured the card's g at mean 0.576039 / var 1.632352
against Gumbel 0.5772157 / 1.6449341, with the FEST log costing ~0.0018
mean / ~0.015 var against the fp16-ideal). The host's `SplitMix64`
staging is gone: what rides the old noise operand is a FIVE-TABLE PER-ROW
SEED — five `[B, 1]` i32 word tables, one per member of the quintuple
`(seed_lo, seed_hi, step, greedy_gate, row)`, each loaded corner-free at
`[pid, 0]` (the Col operand mode reads the stick head alone — the
int-bcast leg's measured law) — one tiny per-row word write where
`[B, PAD_V]` f32 noise rows rode before. Temperature 0
is the exact greedy draw: the gate lane is 0 and `inv_t = 1`, so
`score = x·1 + g·0 = x` exact — byte-identical to the first-max, never a
separate path (`g·0 = 0` and `g·1 = g` are both exact; the gate is a per-row
multiply, the `invt` broadcast's own class).

THE SPLICE'S OWN CONTRACT (what `scratchy-triton-splice` states about this
kernel):

* PARAMETERS, IN ORDER: `logits`, the FIVE SEED WORD TABLES `slo`, `shi`,
  `stp`, `sgate`, `srow` (each `[B, 1]` i32 — one word per row, one per
  member of the seed quintuple), `invt`, `bramp`, `lramp`, `binc`, `linc`,
  `ones`, `sixtyfour`, `padoff`, `gramp`, `ginc`, then the noise section's
  own tables — `hc1`, `hc2`, `hc3`, `c256` (i32: the hash multipliers, the
  signed spellings of 0x9E3779B1 / 0x85EBCA77 / 0xC2B2AE3D, and the
  wrap-fold's 2⁸), `lane` (f32: the per-lane index), `fm8`, `ftop`, `fzero`,
  `fhalf`, `fm22` (f32: the fold's 2⁻⁸ / 2²³−1 / 0 / 0.5 / 2⁻²²), `cmin`
  (fp16: 2⁻¹⁴, the FEST-domain clamp) — then `idx`, the node's operand
  order then the output. The registry does not permute. The first six are
  PER-ROW (the row, its seed quintuple, its temperature); the twenty
  tables after them are STAGED buffers the host owns, constant across
  decode steps, functions of the vocab alone. ⭐ TEN OF THE TWENTY ARE
  `[1, 1]` ONE-WORD TABLES (the wall-5 measured law): the hash multipliers
  `hc1`/`hc2`/`hc3` and the wrap-fold's `c256` (i32), the fold's `fm8`,
  `ftop`, `fzero`, `fhalf`, `fm22` (f32: 2⁻⁸ / 2²³−1 / 0 / 0.5 / 2⁻²²) and
  the FEST-domain clamp `cmin` (fp16: 2⁻¹⁴) are UNIFORM values — one word
  broadcast by the consuming op, `invt`'s own class — never whole-row
  splats: a `[1, PAD_V]` splat burns PAD_V·4 B of local store to carry one
  repeated word, and ten of them tip the 2 MiB LX (the wall-5 measurement).
  The GENUINELY per-lane tables stay whole-row `[1, PAD_V]`: `lane` (f32:
  the per-lane index), `padoff`, and the cascade's ramps and index tables —
  their values vary by lane, and only a load can state them (the no-arange
  law). ⭐ THE SEED WORDS
  RIDE FIVE SEPARATE `[B, 1]` TABLES, NOT ONE `[B, 5]` TABLE: the Col
  operand mode that broadcasts a per-row word reads the STICK HEAD ALONE
  (the int-bcast card leg's measured law — a word at stick word k would be
  mis-read as word 0), so each word rides its own one-stick-per-row table
  loaded corner-free at `[pid, 0]`, `invt`'s own class — the auditor's
  wall-4 ruling (a); the fold's first spelling (one `[B, 5]` table read
  through five `[1, 1]` windows) carried the column corners the integer
  door refuses by name. The worker writes the five words of each row at
  the five tables' stick heads and nothing else.
* CONSTEXPRS: `V` (the model's real vocab), `PAD_V` (V rounded up to a
  power-of-two whole row, STRICTLY above V — the descriptor block's law and
  the sentinel's), `NB = PAD_V // BLOCK`, `BLOCK = 64`, `B` (the row count
  this rung serves — the grid's own width) — stated by the splice from the
  node's vocab and rows, never hardcoded to a model: a different vocab bakes
  its own tables, a different batch width its own grid.
* GRID: `[B]` — one program per request row, at every rung (the solo m=1
  rung and every batched rung alike; a batched rung without the sampler does
  not build).
* THE PAD LAW RIDES THE STAGED OFFSET TABLE (`padoff`), APPLIED LAST:
  `score = (x·inv_t + g·gate) + padoff`. 0.0 on `[0, V)` — adding 0.0 last
  keeps the two-op score's rounding sequence exactly unchanged — and f32::MIN
  (−FLT_MAX, 0xFF7FFFFF) on `[V, PAD_V)`, NEVER IEEE −inf: the hardware's
  fp32 domain does not use IEEE inf (the fp32 reduce template's own "negInf"
  seed is 0xFF7FFFFF, a FINITE number — `summeanmaxexx2_fp32.ddl:71`), and
  an exponent-all-ones lane reads as NaN on the card, poisoning the max into
  NaN. The kernel uses the padded `score` throughout, so the row's own pad
  bytes are irrelevant — the lm-head's zero-pad columns write 0.0 into them
  every step, and `anything + −FLT_MAX` never wins a max a finite lane
  contests. A real lane is untouched (`+ 0.0` exact), NaN rides through (`NaN +
  0.0 = NaN`, the all-NaN sentinel law intact), and an all-real-−FLT_MAX row
  still answers lane 0 — every lane ties and the staged ramps pick the first,
  agreeing with the host reference's first-max. This is why the offset
  table and not a V-constexpr mask: the pad law is a HOST-STAGED fact like
  every other table this kernel reads, never an in-kernel arange.
* THE NOISE IS FINITE ON THE PAD LANES (the hash computes a finite `g` on
  every lane, `u` strictly inside `(0, 1)`), so `(0·inv_t + g·gate) +
  −FLT_MAX` stays FINITE on the pads — no NaN to poison the max, and the pad
  lane loses every contest a real lane enters at ANY temperature (at the
  greedy `inv_t = 1`, gate 0 the pad lane's score is exactly −FLT_MAX, below
  every real finite score — the guards never disagreed, the staged one is
  gone, the surviving one is exact).

⚠ THE ISLAND'S WIDEN CARRIES THE dl16tofp32 STAGGER. The card's
`dl16tofp32` output permutes within each 64-lane block — Front A's probe-8
measurement on the row's own widen (`x32`): the sixteen 4-lane groups
permute by 1-bit right rotation of the 4-bit group index (logical lane
4g+r lands at buffer 4·ror(g)+r; lanes 0-3 and 60-63 fixed). The island's
final `g16.to(tl.float32)` is the SAME op and must carry the SAME
arrangement type. Two MEASURED facts and one planned leg: (1) the
dl16tofp32 stagger above, measured on the row's widen; (2) the island's
composed converts land STANDARD at the noise — the front-b probe's card
leg, elementwise against the host fp16 emulation, mean |diff| 0.0042.
What is NOT stated is the composition's decomposition: "the `fp32todl16`
at the island's entrance composes as the inverse of the stagger" is an
INFERENCE from those two facts, never a fact itself — the arrangement
table records the composed (measured) landing and the stagger's own
measured permutation, and a one-op `fp32todl16` card leg is PLANNED to
measure the entrance directly (later, not now). No consumer assumes
either spelling. The score sum `x·invt + g·gate` adds two
identically-arranged operands (lane-consistent either way); everything
that meets a STANDARD-arranged staged table (the ramps, the index tables)
goes through the compile-time arrangement table argmax-coder is landing
(Front A's fix). This kernel states the requirement; it does not code
around it.

THE ALGORITHM (equality selection over a TWO-LEVEL ramp, no inversions): the
block index is split GROUP-major — `G` groups of 64 blocks (4096 lanes) —
because a single flat decreasing block ramp would need NB in a lane at
gemma-4's PAD_V 524288, past fp16's exact-integer range (2048). The staged
DECREASING ramps (gramp group-major, bramp periodic within each group, lramp
lane-minor) make each `max` land on the FIRST group / FIRST block / FIRST
lane — #631's first-max law by monotonicity — and the staged INCREASING
index tables (ginc, binc, linc) read the selected numbers back out through
the one-hot. Every staged lane value stays ≤ max(64, G) with G ≤ 2048 (the
admission gate's bound) — an exact integer in f32 at ANY admitted vocab, the
same numbers the f16 tables carried (the selection widened to f32, the tables
re-stated at the same values in the wider dtype). The wide combine `g*·4096
+ b*·64 + l*` stays in f32 (exact below 2^24); everything in this kernel is
f32 but the loaded row itself.

⚠ THE MULTIPLY AND THE ADD ARE TWO OPS, TWO ROUNDINGS — `score = x·inv_t +
g·gate` rounds the products, then the sum, exactly the sequence the host
reference states. Nothing in this repo's Triton frontend or KTIR lowering
fuses `mul`+`add` into an FMA (no fma anywhere in either crate), so the card
and the reference agree bit for bit on near-ties — the draw-for-draw bar's
own precondition.

THE SENTINEL LAW (all-NaN row): a NaN lane never wins an ordered max and
never satisfies an ordered equality, so an all-NaN row leaves the one-hot on
the PAD lanes, where the host stages ginc = G, binc = 64 and linc = BLOCK —
idx = G·4096 + 64·64 + 64 = PAD_V + 4160, past the last real lane. The
consumer treats idx > V as the loud refusal (the host argmax answers token 0
over the same row; that divergence is #666, recorded, not reconciled). A
finite row can never reach the pad lanes.
"""

import triton
import triton.language as tl


@triton.jit
def sample_fwd(logits_ptr, slo_ptr, shi_ptr, stp_ptr, sgate_ptr, srow_ptr,  #
               invt_ptr,  #
               bramp_ptr, lramp_ptr, binc_ptr, linc_ptr,  #
               ones_ptr, sixtyfour_ptr, padoff_ptr, gramp_ptr, ginc_ptr,  #
               hc1_ptr, hc2_ptr, hc3_ptr, c256_ptr,  #
               lane_ptr, fm8_ptr, ftop_ptr, fzero_ptr, fhalf_ptr, fm22_ptr,  #
               cmin_ptr,  #
               idx_ptr,  #
               V: tl.constexpr, PAD_V: tl.constexpr,  #
               NB: tl.constexpr, BLOCK: tl.constexpr, B: tl.constexpr):
    # ONE PROGRAM PER REQUEST ROW: the grid is [B] — the solo step's single
    # row and every row of a batched step alike. Every shape below is rank 2
    # with a one-row block; the per-row operands load at [pid, 0], the shared
    # tables at [0, 0] (the same row every program reads — the rmsnorm weight
    # load's own idiom inside a multi-program grid). Every reduce is the
    # golden rank-2 -> rank-1 form over the tile's second axis; no scalar, no
    # reduce-to-scalar, and no splat of an interior value exists anywhere in
    # the kernel. No loops at all.
    pid = tl.program_id(0)
    l_desc = tl.make_tensor_descriptor(logits_ptr, shape=[B, PAD_V], strides=[PAD_V, 1],
                                       block_shape=[1, PAD_V])
    x = l_desc.load([pid, 0])
    # THE WHOLE SELECTION IS F32: the score is a sum, and f16 would round
    # near-ties into ties the host reference never sees. The row itself is
    # the card's f16 lm-head output, widened exactly (every f16 value is
    # exactly representable in f32); every staged table below is f32.
    x32 = x.to(tl.float32)
    # THE TEMPERATURE'S RECIPROCAL, ONE PER ROW: a [1, 1] block loaded at
    # this program's own row, broadcast over the whole padded row by the
    # multiply -- the rmsnorm `x * r[:, None]` broadcast class. The greedy
    # row stages 1.0 (x * 1 exact), a sampled row its own 1/T; a mixed batch
    # works per row.
    i_desc = tl.make_tensor_descriptor(invt_ptr, shape=[B, 1], strides=[1, 1],
                                       block_shape=[1, 1])
    invt = i_desc.load([pid, 0])
    # THE SEED WORDS, ONE TABLE PER MEMBER OF THE QUINTUPLE: five `[B, 1]`
    # i32 tables, each `invt`'s own idiom — a `[1, 1]` block loaded at this
    # program's own row, corner-free. (The fold's first spelling read all
    # five words through `[1, 1]` windows of one `[B, 5]` view; a window at
    # column k carries a corner, and the Col operand mode that broadcasts
    # the word reads the STICK HEAD ALONE — the int-bcast card leg's
    # measured law — so the cornered spelling is unpronounceable on this
    # path. The auditor's wall-4 ruling (a): one word, one table, at the
    # stick head.) The host writes one word per row per table, every step:
    # the step lane is what makes the hash a fresh draw per decode step
    # without restaging anything else, and the row lane is the row's own key
    # fact, STAGED -- the program id is a load offset, never a value (a
    # tt.splat of an interior scalar has no contract on this path).
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

    # THE LANE'S OWN INDEX, staged as its own f32 table (0..PAD_V-1, exact
    # in f32 below 2^24) and ingested by the index-cast law -- the f32 value
    # is exact, so the fptosi recovers the lane bit for bit. The no-arange
    # law's own reading (a per-lane fact is host-staged): the reconstruction
    # g*4096 + b*64 + l would need the 4096/64 scales in a lane, and a
    # splatted constant routes its consumer to the fp16 scalar-multiply door
    # -- unpronounceable for an i32 hash constant, wrong for an f32 multiply
    # -- so the lane rides its own table, the exact shape the front-b probe
    # proved on card (bit-exact through u, 0/2^20 lanes). The ginc/binc/linc
    # tables stay where they belong, at the readback's first use below.
    la_desc = tl.make_tensor_descriptor(lane_ptr, shape=[1, PAD_V], strides=[PAD_V, 1],
                                        block_shape=[1, PAD_V])
    lanef = la_desc.load([0, 0])
    lane = lanef.to(tl.int32)
    # THE NOISE SECTION'S CONSTANTS — `[B, 1]` tables, the SEED WORDS' OWN
    # CLASS (the auditor's 2026-10-27 ruling): a `[1, 1]` block loaded at
    # `[pid, 0]` -- this row's stick head, exactly where a seed word sits --
    # and broadcast by its consuming op (the operand mode), never materialized
    # at row width. The bake stages the same word at every row, so the worker
    # writes nothing for them. ⚠ THE WALL-5 MEASURED LAW still holds: a
    # uniform constant staged as a whole-row `[1, PAD_V]` splat burns PAD_V·4 B
    # of local store to carry ONE repeated word -- at PAD_V=65536 that is
    # 256 KiB per constant, and ten of them (the three hash multipliers and
    # the wrap-fold's 2^8 in i32, the fold's five f32 constants, the
    # FEST-domain clamp in fp16) are 2.5 MiB against the 2 MiB LX, tipped
    # over at the seventh charge. The `[B, 1]` spelling keeps every value
    # bit-identical -- the broadcast IS the value -- and the hash multipliers
    # ride the int-bcast operand mode the int-bcast card leg measured (word 0
    # of the stick); the f32 constants ride the same broadcast class `invt`
    # itself uses. Per-lane tables (the lane index, the pad law's offset row,
    # the cascade's ramps) stay whole-row: their values genuinely vary by
    # lane, and only a load can state them (the no-arange law).
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

    # THE COUNTER HASH -- i32 mul/add ONLY, wrapping (the card's bit-exact
    # integer class, card leg B1). Key: the row's seed table (seed_lo,
    # seed_hi, step, row) and the lane. Multiplication by an odd constant
    # is a bijection mod 2^32 and each square breaks the linear structure
    # an xor-shift mix would break on a richer vocabulary; the constants
    # are the staged spellings of 0x9E3779B1 / 0x85EBCA77 / 0xC2B2AE3D.
    # seed_hi enters additively -- high bits of a product cannot reach low
    # positions with mul/add alone, so the low-bit extraction below is what
    # bounds the effective seed width, a stated fact for the auditor, not a
    # hidden one.
    h = lane * hc1 + slo
    h = h * hc2 + shi + stp + srow
    h = h * h
    h = h * hc1 + slo
    h = h * h
    h = h * hc3 + shi + stp
    h = h * h
    # THE LANE RE-INJECTION, and why the hash does not end on the square:
    # a square's low bits are structurally non-uniform (x² mod 8 ∈ {0, 1, 4},
    # and every mul/add finalizer is a BIJECTION mod 8, so no chain of them
    # can heal it). Adding a term that varies per lane AFTER the last square
    # does heal it -- the sum of any distribution with a uniform is uniform,
    # at every bit position the varying term reaches. One mul-add, and the
    # low-bit defect is gone by construction rather than by measurement.
    h = h * hc1 + lane * hc2 + slo
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
    # seed table -- the probe's free bit-check of the whole integer chain.
    u = (np + fhalf) * fm22
    # THE GUMBEL DRAW through the card's ONLY log -- the fp16 FEST bind,
    # reached through `LegalizeTypes`' own widening-island collapse (the
    # f32 spelling is refused at dxp: the vendor DDLs bind `log` to fp16
    # alone, measured on the probe's card leg). Each log is spelled as the
    # stability-widening island -- `tl.log(x16.to(tl.float32)).to(tl.float16)`
    # -- which the pass collapses to a BARE fp16 log: the emitted op is
    # `log` at fp16, the one bind the vendor DDL carries. The arithmetic
    # between the logs is fp16 negation -- an exact sign-flip, no rounding --
    # and the clamps at 2^-14 (fp16's smallest normal, the staged `cmin`
    # table) guard the FEST log's subnormal-input behavior on both entries.
    # The final widen `g16.to(tl.float32)` is the exit island's own
    # protected funnel (f16 must stage through `dl16tofp32` on the way to an
    # integer -- the DDL refuses the direct fp16→u32 pairing), so it
    # survives the collapse by the pass's own protection law and the noise
    # lands as f32 for the score. ⚠ The widen's output carries the
    # dl16tofp32 STAGGER -- the docstring's arrangement note; the seat's
    # arrangement table states the composed fact, no consumer assumes it.
    u16 = u.to(tl.float16)
    c1 = tl.maximum(u16, cmin)
    lu = tl.log(c1.to(tl.float32)).to(tl.float16)
    v = -lu
    c2 = tl.maximum(v, cmin)
    lt = tl.log(c2.to(tl.float32)).to(tl.float16)
    g16 = -lt
    noise = g16.to(tl.float32)
    # score = x32 * invt + noise * gate: the gate is the greedy row's zero
    # (g * 0 = 0 exact, score = x32 * 1 = x32 exactly -- byte-identical to
    # the first-max, never a separate path) and 1 on a sampled row
    # (g * 1 = g exact). THREE roundings before the sum (x32*invt, noise*gate,
    # then the add) -- the exact sequence a reference must state; nothing in
    # this repo's frontend or lowering fuses mul+add into an FMA.
    gate = sgate.to(tl.float32)
    score = x32 * invt + noise * gate
    # THE PAD LAW, APPLIED LAST: score = (x32 * invt + noise * gate) + padoff,
    # and the padded score throughout. padoff is 0.0 on the real lanes (adding
    # 0.0 last keeps the rounding sequence of the two-op score EXACTLY
    # unchanged) and f32::MIN (−FLT_MAX, 0xFF7FFFFF) on the pads, so a pad
    # lane can never win a max or an equality the real lanes contest WHATEVER
    # the row's pad bytes hold. ⭐ NEVER IEEE −inf: the card's fp32 domain does
    # not use IEEE inf -- the fp32 reduce template's own "negInf" is
    # 0xFF7FFFFF, a FINITE number (summeanmaxexx2_fp32.ddl:71), and an
    # exponent-all-ones lane reads as NaN on the card, poisoning the max into
    # NaN. −FLT_MAX loses every max a finite lane contests, with no NaN to
    # ride through (NaN rides the REAL lanes only: NaN + 0.0 = NaN, the
    # all-NaN sentinel law below intact). A two-tensor add of two loaded
    # tiles -- the same Elementwise(Add) every spliced add kernel is.
    p_desc = tl.make_tensor_descriptor(padoff_ptr, shape=[1, PAD_V], strides=[PAD_V, 1],
                                       block_shape=[1, PAD_V])
    padoff = p_desc.load([0, 0])
    score = score + padoff
    # The global max — in f32, NOT narrowed back: the draw's comparisons are
    # the f32 comparisons the host reference makes, so widening the max would
    # re-introduce exactly the rounding the f32 selection exists to avoid.
    m = tl.max(score, axis=1)
    mm = tl.expand_dims(m, 1)

    # LEVEL 0 -- the FIRST GROUP holding the maximum. gramp is the staged
    # table (host contract): strictly decreasing in the group index, CONSTANT
    # across each group's 4096 lanes, 0 on the pad lanes, which can never win.
    # mask * gramp <= G <= 2048: an exact small-integer f32 product (the
    # admission gate's bound -- a single flat block ramp would need NB in a
    # lane, past fp16's exact range at wide vocabs).
    gr_desc = tl.make_tensor_descriptor(gramp_ptr, shape=[1, PAD_V], strides=[PAD_V, 1],
                                        block_shape=[1, PAD_V])
    gramp = gr_desc.load([0, 0])
    mask = (score == mm).to(tl.float32)
    w0 = mask * gramp
    jG = tl.max(w0, axis=1)
    # The equality inverts the ramp: gramp is CONSTANT per group, so comparing
    # it against its own value at the first maximal group selects that group
    # and nothing else. No subtraction anywhere in the kernel.
    selG = (gramp == tl.expand_dims(jG, 1)).to(tl.float32)

    # LEVEL 1 -- the FIRST 64-block holding the maximum WITHIN group g*. bramp
    # is the staged table (host contract): strictly decreasing in the block
    # index within each group, PERIODIC across groups (the same 64 values in
    # every group), 0 on the pad lanes, which can never win. mask * selG *
    # bramp <= 64: an exact small-integer f32 product.
    br_desc = tl.make_tensor_descriptor(bramp_ptr, shape=[1, PAD_V], strides=[PAD_V, 1],
                                        block_shape=[1, PAD_V])
    bramp = br_desc.load([0, 0])
    w1 = mask * selG * bramp
    jA = tl.max(w1, axis=1)
    # The equality inverts the ramp: bramp is CONSTANT per block, so comparing
    # it against its own value at the first maximal block selects that block
    # and nothing else. No subtraction anywhere in the kernel.
    sel = (bramp == tl.expand_dims(jA, 1)).to(tl.float32)

    # LEVEL 2 -- the FIRST lane within block b*. lramp is the staged per-lane
    # table, strictly decreasing within each block; mask * selG * sel * lramp
    # <= 64.
    lr_desc = tl.make_tensor_descriptor(lramp_ptr, shape=[1, PAD_V], strides=[PAD_V, 1],
                                        block_shape=[1, PAD_V])
    lramp = lr_desc.load([0, 0])
    w2 = mask * selG * sel * lramp
    jB = tl.max(w2, axis=1)
    # lramp is PERIODIC (the same 64 values in every block), so this equality
    # selects lane l* of EVERY block -- and intersected with selG and sel
    # (group g*, block b*) it is the one-hot at the first maximal lane.
    selL = (lramp == tl.expand_dims(jB, 1)).to(tl.float32)
    onehot = selG * sel * selL

    # READ THE NUMBERS OUT through the one-hot: ginc, binc and linc are the
    # staged INCREASING 0-based index tables (staged here at first use --
    # the hash reads its own `lane` table now, so these belong to the
    # readback alone), so the max of the masked product
    # IS the selected group / block / lane number -- including the g* == 0
    # case, where an all-zero product row's max is 0 and 0 is the answer.
    gi_desc = tl.make_tensor_descriptor(ginc_ptr, shape=[1, PAD_V], strides=[PAD_V, 1],
                                        block_shape=[1, PAD_V])
    ginc = gi_desc.load([0, 0])
    bi_desc = tl.make_tensor_descriptor(binc_ptr, shape=[1, PAD_V], strides=[PAD_V, 1],
                                        block_shape=[1, PAD_V])
    binc = bi_desc.load([0, 0])
    li_desc = tl.make_tensor_descriptor(linc_ptr, shape=[1, PAD_V], strides=[PAD_V, 1],
                                        block_shape=[1, PAD_V])
    linc = li_desc.load([0, 0])
    wG = onehot * ginc
    gstar = tl.max(wG, axis=1)
    wB = onehot * binc
    bstar = tl.max(wB, axis=1)
    wL = onehot * linc
    lstar = tl.max(wL, axis=1)

    # WIDEN TO THE OUTPUT STICK through a LOADED ones table: the [1]s
    # broadcast to [1, BLOCK] as an operand mode of a two-tensor multiply (a
    # `tl.full` constant would splat and route the op to the scalar-multiply
    # door; a staged row keeps it `Elementwise(Mul)` -- and the frontend folds
    # a multiply-by-ones constant away entirely, so the table is load-bearing).
    # The selection is f32, so the widen tables are too.
    o_desc = tl.make_tensor_descriptor(ones_ptr, shape=[1, BLOCK], strides=[BLOCK, 1],
                                       block_shape=[1, BLOCK])
    ones = o_desc.load([0, 0])
    g64 = tl.expand_dims(gstar, 1) * ones
    b64 = tl.expand_dims(bstar, 1) * ones
    l64 = tl.expand_dims(lstar, 1) * ones

    # THE WIDE INTEGER, ALREADY IN F32: g* * 4096 + b* * 64 + l* reaches past
    # fp16's exact range, and the f32 selection carries it natively -- the
    # multiply-adds stay f32 (exact below 2^24) against the staged f32 scale
    # table (used twice: once for the block term, once squared for the group
    # term's * 4096), then cast to i32: the index-ingestion law.
    sf_desc = tl.make_tensor_descriptor(sixtyfour_ptr, shape=[1, BLOCK], strides=[BLOCK, 1],
                                        block_shape=[1, BLOCK])
    sf = sf_desc.load([0, 0])
    idx = (g64 * sf * sf + b64 * sf + l64).to(tl.int32)
    # THE OUTPUT STICK, ONE PER ROW: [B, BLOCK] i32, every lane equal, stored
    # at this program's own row. The host reads ONE word per row.
    i2_desc = tl.make_tensor_descriptor(idx_ptr, shape=[B, BLOCK], strides=[BLOCK, 1],
                                        block_shape=[1, BLOCK])
    i2_desc.store([pid, 0], idx)
