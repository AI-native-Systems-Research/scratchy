# SPDX-License-Identifier: Apache-2.0
"""Scratchy's Triton sampling kernel — the token draw, Gumbel-max over the
padded row, FIRST-MAX by arithmetic, as one kernel per request row.

BODY PROVENANCE: #813's `kernels/argmax.py` (the whole-row equality-selection
body that fixture baked through this repo's KTIR→SuperDSC lowering and that
the triton-numeric legs pinned against a derived first-max reference), with
three deltas the sampler needs: the grid is `[B]` (one program per request
row — the solo step is `B = 1`, a batched step `B = 2..32`), the selection
runs IN F32 (the score is a sum `x·inv_t + g`; f16's 10-bit mantissa would
round near-ties into ties the reference never sees), and two per-row operands
join the row (the staged noise row and the temperature's reciprocal).

THE DRAW: `idx = first-max over (logits·inv_T + g)` with `g ~ Gumbel(0, 1)`,
which samples exactly `softmax(logits/T)`. The noise is NOT computed here —
it is the HOST'S staging (`SplitMix64(seed, i)`, f64 internal, ONE f32 cast
per lane), because the device's op vocabulary has no xor and no shift
(`arch_enums.cpp` `opFuncsToString`: 180 entries, neither op present), so the
counter hash cannot run on the card. Temperature 0 is the exact greedy draw:
the host stages a noise row of zeros and `inv_t = 1`, so `score = x + 0`
exact — byte-identical to the first-max, never a separate path.

THE SPLICE'S OWN CONTRACT (what `scratchy-triton-splice` states about this
kernel):

* PARAMETERS, IN ORDER: `logits`, `noise`, `invt`, `bramp`, `lramp`, `binc`,
  `linc`, `ones`, `sixtyfour`, `padoff`, `gramp`, `ginc`, then `idx` — the
  node's operand order then the output. The registry does not permute. The
  first three are PER-ROW (the row, its noise, its temperature); the nine
  tables after them are STAGED buffers the host owns, constant across decode
  steps, functions of the vocab alone.
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
  `score = (x32 * inv_t + noise) + padoff`. 0.0 on `[0, V)` — adding 0.0
  last keeps the two-op score's rounding sequence exactly unchanged — and
  f32::MIN (−FLT_MAX, 0xFF7FFFFF) on `[V, PAD_V)`, NEVER IEEE −inf: the
  hardware's fp32 domain does not use IEEE inf (the fp32 reduce template's
  own "negInf" seed is 0xFF7FFFFF, a FINITE number —
  `summeanmaxexx2_fp32.ddl:71`), and an exponent-all-ones lane reads as NaN
  on the card, poisoning the max into NaN. The kernel uses the padded
  `score` throughout, so the row's own pad bytes are irrelevant — the
  lm-head's zero-pad columns write 0.0 into them every step, and
  `anything + −FLT_MAX` never wins a max a finite lane contests. A real lane
  is untouched (`+ 0.0` exact), NaN rides through (`NaN + 0.0 = NaN`, the
  all-NaN sentinel law intact), and an all-real-−FLT_MAX row still answers
  lane 0 — every lane ties and the staged ramps pick the first, agreeing
  with the host reference's first-max. This is why the offset table and not
  a V-constexpr mask: the pad law is a HOST-STAGED fact like every other
  table this kernel reads, never an in-kernel arange.
* THE NOISE ROW'S PAD LANES ARE 0.0 (the host stages them so): the pads are
  ruled by `padoff` alone, and no IEEE inf may reach the card from ANY
  staged buffer.

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

⚠ THE MULTIPLY AND THE ADD ARE TWO OPS, TWO ROUNDINGS — `score = xm·inv_t +
noise` rounds the product, then the sum, exactly the sequence the host
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
def sample_fwd(logits_ptr, noise_ptr, invt_ptr,  #
               bramp_ptr, lramp_ptr, binc_ptr, linc_ptr,  #
               ones_ptr, sixtyfour_ptr, padoff_ptr, gramp_ptr, ginc_ptr, idx_ptr,  #
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
    # THE NOISE ROW, ONE PER ROW: the host's Gumbel draws (SplitMix64(seed,
    # i), f64 internal, ONE f32 cast per lane -- the reference's own law),
    # 0.0 on the pad lanes (the pads are ruled by `padoff` alone -- see the
    # pad law below). score = x32 * invt + noise: TWO ops, TWO roundings,
    # the exact sequence the host reference states (nothing in this repo's
    # frontend or lowering fuses mul+add into an FMA). The greedy row stages
    # zeros and invt = 1, so score = x32 exactly.
    n_desc = tl.make_tensor_descriptor(noise_ptr, shape=[B, PAD_V], strides=[PAD_V, 1],
                                       block_shape=[1, PAD_V])
    noise = n_desc.load([pid, 0])
    score = x32 * invt + noise
    # THE PAD LAW, APPLIED LAST: score = (x32 * invt + noise) + padoff, and
    # the padded score throughout. padoff is 0.0 on the real lanes (adding
    # 0.0 last keeps the rounding sequence of the two-op score EXACTLY
    # unchanged) and f32::MIN (−FLT_MAX, 0xFF7FFFFF) on the pads, so a pad
    # lane can never win a max or an equality the real lanes contest WHATEVER
    # the row's pad bytes hold. ⭐ NEVER IEEE −inf: the card's fp32 domain
    # does not use IEEE inf -- the fp32 reduce template's own "negInf" is
    # 0xFF7FFFFF, a FINITE number (summeanmaxexx2_fp32.ddl:71), and an
    # exponent-all-ones lane reads as NaN on the card, poisoning the max
    # into NaN. −FLT_MAX loses every max a finite lane contests, with no NaN
    # to ride through (NaN rides the REAL lanes only: NaN + 0.0 = NaN, the
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
    # staged INCREASING 0-based index tables, so the max of the masked product
    # IS the selected group / block / lane number -- including the g* == 0
    # case, where an all-zero product row's max is 0 and 0 is the answer.
    gi_desc = tl.make_tensor_descriptor(ginc_ptr, shape=[1, PAD_V], strides=[PAD_V, 1],
                                        block_shape=[1, PAD_V])
    ginc = gi_desc.load([0, 0])
    wG = onehot * ginc
    gstar = tl.max(wG, axis=1)
    bi_desc = tl.make_tensor_descriptor(binc_ptr, shape=[1, PAD_V], strides=[PAD_V, 1],
                                        block_shape=[1, PAD_V])
    binc = bi_desc.load([0, 0])
    wB = onehot * binc
    bstar = tl.max(wB, axis=1)
    li_desc = tl.make_tensor_descriptor(linc_ptr, shape=[1, PAD_V], strides=[PAD_V, 1],
                                        block_shape=[1, PAD_V])
    linc = li_desc.load([0, 0])
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
