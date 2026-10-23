# SPDX-License-Identifier: Apache-2.0
"""argmax_first_max fixture: the greedy sampler's vocab scan, FIRST-MAX by
arithmetic, as one kernel.

This is the ladder form of loop closure's device sampler (#603/#500): the
algorithm that used to live in a hand-minted 8-link SuperDDC composite
(#698, deleted by #714) is stated HERE, in the kernel, over generic ops only
-- whole-row `tt.reduce` maxima, equality compares, staged-table multiplies,
the fp32 island for the one wide integer. Nothing argmax-specific exists
below the kernel: whatever lowering the splice emits for `tt.reduce`/
`arith.cmpf`/`arith.mulf` is the lowering this kernel gets, which is the
whole point.

THE ALGORITHM (equality selection over one tile, no inversions)
----------------------------------------------------------------
The block index is split GROUP-major -- G groups of 64 blocks (4096 lanes) --
because a single flat decreasing block ramp would need NB in a lane (8192 at
a 524288-wide row), past fp16's exact-integer range (2048). Three levels:

    m    = max of the logits row (whole row, one reduce)
    mask = (x == m)                     true at EVERY lane holding the max
    w0   = mask * gramp                 gramp_i = G - (i // 4096)  (first-GROUP)
    jG   = max of w0                    gramp's value at the first maximal group
    selG = (gramp == jG)                one on group g* ONLY (gramp is
                                        group-constant, so the equality selects
                                        the group and nothing else)
    w1   = mask * selG * bramp          bramp_i = 64 - ((i // 64) % 64),
                                        PERIODIC across groups (first-BLOCK)
    jA   = max of w1                    bramp's value at the first maximal
                                        block within group g*
    sel  = (bramp == jA)                one at block offset b* of EVERY group
                                        (bramp is periodic), so `selG * sel`
                                        is group g*'s block b* and nothing else
    w2   = mask * selG * sel * lramp    lramp_i = 64 - (i % 64)   (first-LANE)
    jB   = max of w2                    lramp's value at the first maximal lane
                                        within block b*
    selL = (lramp == jB)                one at lane l* of EVERY block (lramp is
                                        periodic), so the triple product is one
                                        at (g*, b*, l*) and nowhere else
    onehot = selG * sel * selL          THE first maximal lane, exactly one
    g*   = max of (onehot * ginc)       ginc_i = i // 4096  (increasing, 0-based)
    b*   = max of (onehot * binc)       binc_i = (i // 64) % 64  (periodic)
    l*   = max of (onehot * linc)       linc_i = i % 64   (increasing, 0-based)
    idx  = g* * 4096 + b* * 64 + l*     (computed in the fp32 island)

FIRST-MAX BY ARITHMETIC, not by a tie-break rule. The selection never
INVERTS anything: the decreasing ramps (gramp, bramp, lramp) make each `max`
land on the FIRST group / FIRST block / FIRST lane (a strictly decreasing
weight is #631's first-max law made structural), and the INCREASING ramps
(ginc, binc, linc) read the selected numbers back OUT through the one-hot --
`max` of `onehot * ginc` is g* because the one-hot lane carries g* and every
other lane carries 0, including the g* == 0 case (an all-zero product row
whose max is 0 IS the answer). The v5 form inverted by subtraction
(`b = NB - jA`, `l = BLOCK - jB`); those `[1]`-shaped `splat - tensor` subs
have no door arm and no whole-stick emission, while an equality has one
(`equal`, the vendor's own broadcast op), so the inversion rides the
equality selection instead. Same tie law, same sentinel, no subtractions.

THE SENTINEL LAW (all-NaN row), derived not patched: a NaN lane never wins
an ordered max and never satisfies an ordered equality, so an all-NaN row
leaves w0 identically zero: jG = 0, and `gramp == 0` holds exactly on the PAD
lanes (gramp is 0 there); likewise jA = 0 and `bramp == 0`, jB = 0 and
`lramp == 0` select the pad lanes. `onehot` is then the whole pad region,
where the host stages ginc = G, binc = 64 and linc = BLOCK -- so idx =
G * 4096 + 64 * 64 + 64 = PAD_V + 4160, past the last real lane under either
NaN semantics (maxnumf-style combines leave m at the -inf of the pad lanes,
whose ramps are 0; NaN-propagating combines leave m itself NaN; w0 is zero
both ways). The caller treats idx > V as the loud refusal (the host argmax
answers token 0 over the same row; that divergence is recorded, not
reconciled here). A finite row can never reach the pad lanes: its jG >= 1,
jA >= 1 and jB >= 1 (the first real group/block/lane's ramps), and no pad
lane holds a nonzero ramp.

WHY THE STAGED NUMBERS RIDE f16: every STAGED lane value is a whole number
<= max(64, G) -- the periodic block and lane tables top out at 64, and the
group ramp's ceiling G is bounded by the admission gate at 2048 -- and every
product along the way (mask * gramp <= G, onehot * ginc <= G - 1,
onehot * binc <= 63, onehot * linc <= 63) is an exact small-integer fp16
product. This is moe.py's own law -- "the lane ids ride f16 (exact to 2048;
the real router's E = 128)" -- restated at vocab width, where a single flat
id cannot ride f16 but the (group, block, lane) TRIPLE always can.

THE WIDE INTEGER RIDES THE FP32 ISLAND, ONCE: the final combine
`g* * 4096 + b* * 64 + l*` reaches past fp16's exact range, so it is
computed as `(g64.to(f32) * sf * sf + b64.to(f32) * sf + l64.to(f32))
.to(int32)` -- the f16 -> f32 widens, f32 multiply-adds against the staged
scale table (twice for the group term: 64 * 64 = 4096), and the f32 -> i32
cast. This is the exact island moe.py's gelu runs (``g32 = g.to(tl.float32)``
for a cubic that overflows f16) and rmsnorm.py's mean runs; on the device it
is the KSPLIT pair's own recipe (`dl16tofp32` -> same-stick f32
multiply-add -> `fp32toint32`), and the values are whole numbers below 2^24,
exact in every one of those formats. Everything ELSE in the kernel stays in
its native fp16 domain end to end -- the maxima, the equality compares, the
ramp multiplies never leave it.

DEVICE LAWS THIS SHAPE OBEYS
----------------------------
* THE WHOLE ROW IS ONE TILE: the row is presented (and loaded) as [1, PAD_V]
  in ONE descriptor block, exactly as rmsnorm.py loads [BLOCK_M, D_MODEL] --
  the descriptor's iteration space carries the width, so there is NO
  per-stick loop to unroll and the door receives a straight-line program of
  about twenty-five ops (seven reduces, the compare/multiply chain, the
  island, the store). A per-stick-loop form would splice to ~2300 trips and
  tens of thousands of descriptors; this form is the door-lowerable one.
* EACH `tt.reduce` IS the golden rank-2 -> rank-1 form over the tile's second
  axis, and `tl.max` of an f16 tile reduces in f32 (upstream's own promotion)
  and is narrowed straight back, moe.py's spelling -- the widening island
  LegalizeTypes collapses.
* NO LOOPS AT ALL: not even `tl.static_range` -- every offset is the one
  [0, 0] load corner. (moe.py's law is that every loop is static; this kernel
  has none to keep static.)
* NO `[1]`-SHAPED COMPUTE: every reduce result is consumed either by an
  equality (through the [1] -> [1, PAD_V] broadcast the compare materialises)
  or by the `* ones` widen (through the [1] -> [1, BLOCK] broadcast the
  multiply materialises) -- both are operand modes of a [1, n] op, never ops
  of their own. A `[1]`-shaped pointwise op is sub-stick on this device and
  the door refuses it; this kernel never states one.
* NO SPLAT OPERAND ON ANY COMPUTE OP: the widen multiplies read a LOADED
  [1, BLOCK] table of ones and the island's scale a LOADED [1, BLOCK] f32
  table of 64.0 -- both host-staged, neither a `tensor.splat`, so every
  multiply in the kernel is a two-tensor `Elementwise(Mul)` (a splatted
  operand would route to the scalar-multiply door, which addresses one
  registry slot, not a sprayed row).
* V, PAD_V, NB, BLOCK ARE constexpr (SPYRE-SPECIFIC): the emitter derives
  work division from descriptor extents, and Spyre compiles one kernel per
  shape. Same law as every other fixture.
* ONE PROGRAM (grid [1]): the sampler serves the solo greedy decode step's
  single [1, V] row. Batched/sampled steps keep the host sampler -- they need
  the whole distribution, not one index.

HOST CONTRACT (the part that is not in this file)
-------------------------------------------------
NINE staged tables, all host obligations for the same reason embedding.py
pads nothing in-kernel and attention_sdpa.py's causal mask is
host-precomputed: the buffers are staging rows the host already owns, and
`other=`-masked loads are not in this frontend's vocabulary. The first eight
are CONSTANT across decode steps (functions of V alone), so under the
pooled-substrate regime they live in stable slots and every step's CB reads
the same addresses:

* THE LOGITS ROW is presented as [PAD_V] fp16, PAD_V = the vocab rounded UP
  to a power-of-two whole row, STRICTLY above V (a power-of-two vocab still
  needs pad lanes for the sentinel), with the pad lanes holding 0.0 -- the
  PRODUCTION shape: the lm-head's zero-pad weight columns write 0.0 there
  every decode step. The row's pad bytes are deliberately load-bearing
  garbage to the sampler: the kernel adds the offset table before its first
  max, so it never maxes the raw pads. (On device a durable -inf tail past
  the device width also survives -- the offset makes both shapes the same
  answer.)
* THE BLOCK RAMP is [PAD_V] fp16 with bramp_i = 64 - ((i // 64) % 64) on the
  real lanes and 0 on the pad lanes -- strictly decreasing in the block index
  within each group, PERIODIC across groups, so the first-block law is its
  monotonicity and its ceiling is 64 at any vocab.
* THE LANE RAMP is [PAD_V] fp16 with lramp_i = 64 - (i % 64) on the real
  lanes and 0 on the pad lanes -- strictly decreasing in the lane index
  within each block, so the first-lane law is its monotonicity. It is the
  same 64-value pattern in every block, staged expanded because a [1, 64]
  stick cannot broadcast against a [1, PAD_V] tile.
* THE BLOCK INDEX TABLE is [PAD_V] fp16 with binc_i = (i // 64) % 64 on the
  real lanes and 64 on the pad lanes -- 0-based, increasing, PERIODIC across
  groups (the block-within-group number), so the one-hot reads b* straight
  out; the PAD value is the SENTINEL's block half (an all-NaN row's onehot
  lands on the pad lanes, so b* = 64 there).
* THE LANE INDEX TABLE is [PAD_V] fp16 with linc_i = i % 64 on the real
  lanes and BLOCK on the pad lanes -- the lane half of the same law.
* THE WIDEN TABLES are [1, BLOCK] fp16 of 1.0 (the `* ones` widen, staging
  what a `tl.full` constant would splat, so the multiply stays a two-tensor
  op) and [1, BLOCK] fp32 of 64.0 (the island's scale, in the island's own
  dtype so the f32 multiply is also two-tensor; used twice -- once for the
  block term, once squared for the group term's 4096).
* THE OFFSET TABLE is [PAD_V] fp16 with 0.0 on `[0, V)` and -inf on the pads
  -- THE PAD LAW, added to the row before the kernel's first max (`xm = x +
  padoff`, xm throughout). Real lanes are untouched (`x + 0` exact); the pads
  become -inf whatever the row staged there (0.0 from the lm-head's zero
  columns, a durable -inf tail: `0 + -inf = -inf` and `-inf + -inf = -inf`
  alike); NaN rides through (`NaN + 0 = NaN`), so the all-NaN sentinel law
  survives; and an all-(-inf) row still answers lane 0 (every lane ties at
  -inf and the ramps pick the first). Staging the law as a table -- never a
  V-constexpr arange mask -- keeps it on the same host contract as every
  other constant this kernel reads.
* THE GROUP RAMP is [PAD_V] fp16 with gramp_i = G - (i // 4096) on the real
  lanes and 0 on the pad lanes -- strictly decreasing in the group index,
  CONSTANT across each group's 4096 lanes. Its ceiling G is what the
  admission gate bounds (<= 2048, fp16-exact), which is the whole reason the
  ramp is two-level.
* THE GROUP INDEX TABLE is [PAD_V] fp16 with ginc_i = i // 4096 on the real
  lanes and G on the pad lanes -- the sentinel's group half.

Computing the ramps in-kernel would need `V - (start + tl.arange(...))`-style
arithmetic -- a runtime scalar added to an arange in the tile domain, a form
no fixture carries (see the pass-2 comment in the v4 history and attention's
aranges, which are dead code in every golden).

The output is ONE i32 value (the gather ingestion law: index tables are
SENUINT32 at wordLength 4), stored as a [1, 64] i32 stick with every lane
equal so the store is a whole stick like every other fixture's; the consumer
reads word 0.

`reference()` is native PyTorch with the tie law stated EXPLICITLY (mask,
then the first true index) rather than `torch.argmax`, whose tie behavior is
a backend promise rather than an arithmetic fact -- this fixture exists to
pin first-max, so the reference must derive it, not assume it.
"""

import torch

import triton
import triton.language as tl

V = 49155  # granite-3.1's vocab: NOT a power of two -- PAD_V carries the pad lanes
PAD_V = 65536  # V rounded up to a power-of-two (the descriptor block's law)
BLOCK = 64  # the stick: the two-level split's lane width, and the output stick
NB = PAD_V // BLOCK  # the block count: 1024
GROUP_BLOCKS = 64  # blocks per group: the periodic block ramp's period
G = NB // GROUP_BLOCKS  # the group count: 16, gramp's ceiling, fp16-exact

NEG_INF = float("-inf")
DTYPE = torch.float16


@triton.jit
def argmax_first_max_kernel(logits_ptr, bramp_ptr, lramp_ptr, binc_ptr, linc_ptr,
                            ones_ptr, sixtyfour_ptr, padoff_ptr, gramp_ptr, ginc_ptr,
                            idx_ptr,
                            V: tl.constexpr, PAD_V: tl.constexpr,
                            NB: tl.constexpr, BLOCK: tl.constexpr):
    # THE WHOLE ROW IS ONE TILE: one load, [1, PAD_V], the rmsnorm fixture's own
    # whole-row block (its [BLOCK_M, D_MODEL] is larger than this). Every shape
    # below is rank 2; every reduce is the golden rank-2 -> rank-1 form over the
    # tile's second axis; no scalar, no reduce-to-scalar, and no splat of an
    # interior value exists anywhere in the kernel.
    l_desc = tl.make_tensor_descriptor(logits_ptr, shape=[1, PAD_V], strides=[PAD_V, 1],
                                       block_shape=[1, PAD_V])
    x = l_desc.load([0, 0])
    # THE PAD LAW, BEFORE THE FIRST MAX: xm = x + padoff, and xm throughout.
    # padoff is 0.0 on the real lanes (x + 0 exact) and -inf on the pads, so a
    # pad lane can never win a max or an equality the real lanes contest
    # WHATEVER the row's pad bytes hold (this fixture stages them 0.0 -- the
    # production shape, the lm-head's zero-pad columns; 0 + -inf = -inf). NaN
    # rides through (NaN + 0 = NaN), so the all-NaN sentinel law is intact. A
    # two-tensor add of two loaded tiles -- the same Elementwise(Add) every
    # spliced add kernel is.
    p_desc = tl.make_tensor_descriptor(padoff_ptr, shape=[1, PAD_V], strides=[PAD_V, 1],
                                       block_shape=[1, PAD_V])
    padoff = p_desc.load([0, 0])
    xm = x + padoff
    # The global max. `tl.max` of an f16 tile reduces in f32 (upstream's own
    # promotion) and is narrowed straight back, moe.py's spelling: the max of
    # f16 values is exactly representable in f16, so the round trip is exact.
    m = tl.max(xm, axis=1).to(tl.float16)
    mm = tl.expand_dims(m, 1)

    # LEVEL 0 -- the FIRST GROUP holding the maximum. gramp is the staged
    # table (host contract): strictly decreasing in the group index, CONSTANT
    # across each group's 4096 lanes, 0 on the pad lanes, which can never win.
    # mask * gramp <= G = 16: an exact small-integer fp16 product (the
    # docstring's two-level law -- a single flat block ramp would need NB in a
    # lane, past fp16's exact range at wide vocabs).
    gr_desc = tl.make_tensor_descriptor(gramp_ptr, shape=[1, PAD_V], strides=[PAD_V, 1],
                                        block_shape=[1, PAD_V])
    gramp = gr_desc.load([0, 0])
    mask = (xm == mm).to(tl.float16)
    w0 = mask * gramp
    jG = tl.max(w0, axis=1).to(tl.float16)
    # The equality inverts the ramp: gramp is CONSTANT per group, so comparing
    # it against its own value at the first maximal group selects that group
    # and nothing else. No subtraction anywhere in the kernel.
    selG = (gramp == tl.expand_dims(jG, 1)).to(tl.float16)

    # LEVEL 1 -- the FIRST 64-block holding the maximum WITHIN group g*. bramp
    # is the staged table (host contract): strictly decreasing in the block
    # index within each group, PERIODIC across groups (the same 64 values in
    # every group), 0 on the pad lanes, which can never win. mask * selG *
    # bramp <= 64: an exact small-integer fp16 product.
    br_desc = tl.make_tensor_descriptor(bramp_ptr, shape=[1, PAD_V], strides=[PAD_V, 1],
                                        block_shape=[1, PAD_V])
    bramp = br_desc.load([0, 0])
    w1 = mask * selG * bramp
    jA = tl.max(w1, axis=1).to(tl.float16)
    # The equality inverts the ramp: bramp is CONSTANT per block, so comparing
    # it against its own value at the first maximal block selects that block
    # and nothing else. No subtraction anywhere in the kernel.
    sel = (bramp == tl.expand_dims(jA, 1)).to(tl.float16)

    # LEVEL 2 -- the FIRST lane within block b*. lramp is the staged per-lane
    # table, strictly decreasing within each block; mask * selG * sel * lramp
    # <= 64.
    lr_desc = tl.make_tensor_descriptor(lramp_ptr, shape=[1, PAD_V], strides=[PAD_V, 1],
                                        block_shape=[1, PAD_V])
    lramp = lr_desc.load([0, 0])
    w2 = mask * selG * sel * lramp
    jB = tl.max(w2, axis=1).to(tl.float16)
    # lramp is PERIODIC (the same 64 values in every block), so this equality
    # selects lane l* of EVERY block -- and intersected with selG and sel
    # (group g*, block b*) it is the one-hot at the first maximal lane.
    selL = (lramp == tl.expand_dims(jB, 1)).to(tl.float16)
    onehot = selG * sel * selL

    # READ THE NUMBERS OUT through the one-hot: ginc, binc and linc are the
    # staged INCREASING 0-based index tables, so the max of the masked product
    # IS the selected group / block / lane number -- including the g* == 0
    # case, where an all-zero product row's max is 0 and 0 is the answer.
    gi_desc = tl.make_tensor_descriptor(ginc_ptr, shape=[1, PAD_V], strides=[PAD_V, 1],
                                        block_shape=[1, PAD_V])
    ginc = gi_desc.load([0, 0])
    wG = onehot * ginc
    gstar = tl.max(wG, axis=1).to(tl.float16)
    bi_desc = tl.make_tensor_descriptor(binc_ptr, shape=[1, PAD_V], strides=[PAD_V, 1],
                                        block_shape=[1, PAD_V])
    binc = bi_desc.load([0, 0])
    wB = onehot * binc
    bstar = tl.max(wB, axis=1).to(tl.float16)
    li_desc = tl.make_tensor_descriptor(linc_ptr, shape=[1, PAD_V], strides=[PAD_V, 1],
                                        block_shape=[1, PAD_V])
    linc = li_desc.load([0, 0])
    wL = onehot * linc
    lstar = tl.max(wL, axis=1).to(tl.float16)

    # WIDEN TO THE OUTPUT STICK through a LOADED ones table: the [1]s broadcast
    # to [1, BLOCK] as an operand mode of a two-tensor multiply (a `tl.full`
    # constant would splat and route the op to the scalar-multiply door; a
    # staged row keeps it `Elementwise(Mul)` -- and the frontend folds a
    # multiply-by-ones constant away entirely, so the table is load-bearing).
    o_desc = tl.make_tensor_descriptor(ones_ptr, shape=[1, BLOCK], strides=[BLOCK, 1],
                                       block_shape=[1, BLOCK])
    ones = o_desc.load([0, 0])
    g64 = tl.expand_dims(gstar, 1) * ones
    b64 = tl.expand_dims(bstar, 1) * ones
    l64 = tl.expand_dims(lstar, 1) * ones

    # THE WIDE INTEGER RIDES THE FP32 ISLAND, ONCE: g* * 4096 + b* * 64 + l*
    # reaches past fp16's exact range, so the combine widens and
    # multiply-adds in f32 (exact below 2^24) against the staged f32 scale
    # table -- twice for the group term (64 * 64 = 4096), once for the block
    # -- and casts to i32: moe.py's gelu island and the KSPLIT pair's own
    # device recipe.
    sf_desc = tl.make_tensor_descriptor(sixtyfour_ptr, shape=[1, BLOCK], strides=[BLOCK, 1],
                                        block_shape=[1, BLOCK])
    sf = sf_desc.load([0, 0])
    idx = (g64.to(tl.float32) * sf * sf + b64.to(tl.float32) * sf
           + l64.to(tl.float32)).to(tl.int32)
    i_desc = tl.make_tensor_descriptor(idx_ptr, shape=[1, BLOCK], strides=[BLOCK, 1],
                                       block_shape=[1, BLOCK])
    i_desc.store([0, 0], idx)


def inputs(seed: int = 0):
    """The nine staged tables (host contract) plus the logits row, whose pads
    hold 0.0 -- the PRODUCTION shape, the lm-head's zero-pad columns -- so the
    pad law lives entirely in the `padoff` table. The row carries a deliberate
    TIE at the max -- lanes 40017 and 48000 both hold the row's largest value,
    so the reference pins FIRST-max against a real tie, not a vacuous one --
    and its real lanes sit BELOW ZERO on purpose (the padoff control): with
    0.0-staged pads, a kernel that maxes the RAW row answers a pad lane, so
    only `x + padoff` (pads -inf) keeps the answer on the real lanes. A
    positive-max stimulus cannot see a missing offset table at all -- the pads
    never win either way -- which is why this row is shifted.

    The tie lanes are HIGH on purpose, and that is a second pin: the answer
    40017 = 9*4096 + 49*64 + 17 spans all three levels of the two-level ramp
    (group 9, block 49, lane 17) and is NOT representable in fp16 (the
    format's ULP is 32 at [32768, 65536)), so a kernel whose final combine
    degraded to f16 answers 40032 or 40016 and the exact comparison sees it.
    A tie at low lanes (the original 100/5000) cannot discriminate that: 100
    = 0*4096 + 1*64 + 36 sits inside fp16's exact-integer range, so a
    degraded combine still answers it exactly and the defect hides."""
    g = torch.Generator().manual_seed(seed)
    # -8 keeps every real lane below the 0.0 pads (randn's max over 49k lanes
    # is ~4.3), so the tie maximum is negative too and the padoff control
    # discriminates: raw-row maxing answers a pad lane, never 40017.
    x = torch.randn(V, generator=g, dtype=DTYPE) - 8.0
    # ONE ABOVE the natural maximum: a 49k-lane fp16 randn row ties its own
    # maximum naturally (the format's tail is coarse), so pinning the tie AT
    # the natural max would leave an earlier natural lane winning. +1 is exact
    # in fp16 at these magnitudes and strictly above every natural lane.
    top = x.max() + 1.0
    x[40017] = top
    x[48000] = top
    row = torch.zeros(PAD_V, dtype=DTYPE)
    row[:V] = x
    # THE OFFSET TABLE: 0.0 on the real lanes, -inf on the pads -- the pad law
    # the kernel adds before its first max. Staging it here (not as an
    # in-kernel arange mask) is the same host contract as every other table.
    padoff = torch.zeros(PAD_V, dtype=DTYPE)
    padoff[V:] = NEG_INF
    bramp = torch.zeros(PAD_V, dtype=DTYPE)
    lramp = torch.zeros(PAD_V, dtype=DTYPE)
    binc = torch.full((PAD_V,), GROUP_BLOCKS, dtype=DTYPE)
    linc = torch.full((PAD_V,), BLOCK, dtype=DTYPE)
    gramp = torch.zeros(PAD_V, dtype=DTYPE)
    ginc = torch.full((PAD_V,), G, dtype=DTYPE)
    for i in range(V):
        bramp[i] = GROUP_BLOCKS - (i // BLOCK) % GROUP_BLOCKS
        lramp[i] = BLOCK - (i % BLOCK)
        binc[i] = (i // BLOCK) % GROUP_BLOCKS
        linc[i] = i % BLOCK
        gramp[i] = G - (i // (BLOCK * GROUP_BLOCKS))
        ginc[i] = i // (BLOCK * GROUP_BLOCKS)
    ones = torch.ones((1, BLOCK), dtype=DTYPE)
    sixtyfour = torch.full((1, BLOCK), 64.0, dtype=torch.float32)
    return row, bramp, lramp, binc, linc, ones, sixtyfour, padoff, gramp, ginc


def reference(row: torch.Tensor) -> torch.Tensor:
    """First-max argmax, derived: the equality mask's first true index, over
    the REAL lanes alone (the pads are unread -- the kernel's own pad law adds
    -inf to them, and the reference states the same predicate over [0, V)).
    An all-NaN real row leaves the mask empty and answers the sentinel (the
    pad lanes' staged ginc/binc/linc: G*4096 + 64*64 + 64, one past the padded
    row's block span), matching the kernel's arithmetic. Returned as a
    1-lane int32 tensor (the ingestion dtype)."""
    x = row[:V]
    m = x.max()
    mask = x == m
    hit = torch.nonzero(mask)
    idx = hit[0].item() if hit.numel() > 0 else PAD_V + GROUP_BLOCKS * BLOCK + BLOCK
    return torch.tensor([idx], dtype=torch.int32)
