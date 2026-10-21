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
    m    = max of the logits row (whole row, one reduce)
    mask = (x == m)                     true at EVERY lane holding the max
    w1   = mask * bramp                 bramp_i = NB - (i // 64)  (first-BLOCK)
    jA   = max of w1                    bramp's value at the first maximal block
    sel  = (bramp == jA)                one on block b* ONLY (bramp is
                                        block-constant, so the equality selects
                                        the block and nothing else)
    w2   = mask * sel * lramp           lramp_i = 64 - (i % 64)   (first-LANE)
    jB   = max of w2                    lramp's value at the first maximal lane
                                        within block b*
    selL = (lramp == jB)                one at lane l* of EVERY block (lramp is
                                        periodic), so `sel * selL` is one at
                                        (b*, l*) and nowhere else
    onehot = sel * selL                 THE first maximal lane, exactly one
    b*   = max of (onehot * binc)       binc_i = i // 64   (increasing, 0-based)
    l*   = max of (onehot * linc)       linc_i = i % 64   (increasing, 0-based)
    idx  = b* * 64 + l*                 (computed in the fp32 island)

FIRST-MAX BY ARITHMETIC, not by a tie-break rule. The selection never
INVERTS anything: the decreasing ramps (bramp, lramp) make each `max` land on
the FIRST block / FIRST lane (a strictly decreasing weight is #631's
first-max law made structural), and the INCREASING ramps (binc, linc) read
the selected block and lane numbers back OUT through the one-hot -- `max` of
`onehot * binc` is b* because the one-hot lane carries b* and every other
lane carries 0, including the b* == 0 case (an all-zero product row whose
max is 0 IS the answer). The v5 form inverted by subtraction
(`b = NB - jA`, `l = BLOCK - jB`); those `[1]`-shaped `splat - tensor` subs
have no door arm and no whole-stick emission, while an equality has one
(`equal`, the vendor's own broadcast op), so the inversion rides the
equality selection instead. Same tie law, same sentinel, no subtractions.

THE SENTINEL LAW (all-NaN row), derived not patched: a NaN lane never wins an
ordered max and never satisfies an ordered equality, so an all-NaN row leaves
w1 identically zero: jA = 0, and `bramp == 0` holds exactly on the PAD lanes
(bramp is 0 there); likewise jB = 0 and `lramp == 0` selects the pad lanes.
`onehot` is then the whole pad region, where the host stages binc = NB and
linc = BLOCK -- so b* = NB, l* = BLOCK and idx = NB * 64 + 64 = PAD_V + 64,
past the last real lane under either NaN semantics (maxnumf-style combines
leave m at the -inf of the pad lanes, whose ramps are 0; NaN-propagating
combines leave m itself NaN; w1 is zero both ways). The caller treats idx > V
as the loud refusal (the host argmax answers token 0 over the same row; that
divergence is recorded, not reconciled here). A finite row can never reach
the pad lanes: its jA >= 1 (the first real block's bramp is NB or less but at
least NB - 767) and its jB >= 1, and no pad lane holds a nonzero ramp.

WHY THE BLOCK/LANE NUMBERS RIDE f16: every STAGED lane value is a whole
number <= 1024 (NB = 1024, the ramps' ceiling), and every product along the
way (mask * bramp <= 1024, onehot * binc <= 1023, onehot * linc <= 63) is an
exact small-integer fp16 product. This is moe.py's own law -- "the lane ids
ride f16 (exact to 2048; the real router's E = 128)" -- restated at vocab
width, where a single flat id cannot ride f16 but the (block, lane) PAIR
always can.

THE WIDE INTEGER RIDES THE FP32 ISLAND, ONCE: the final combine
`b* * 64 + l*` reaches 49154, past fp16's exact range, so it is computed as
`(b64.to(f32) * sf + l64.to(f32)).to(int32)` -- the f16 -> f32 widens, one f32
multiply-add against staged tables, and the f32 -> i32 cast. This is the
exact island moe.py's gelu runs (``g32 = g.to(tl.float32)`` for a cubic that
overflows f16) and rmsnorm.py's mean runs; on the device it is the KSPLIT
pair's own recipe (`dl16tofp32` -> same-stick f32 multiply-add ->
`fp32toint32`), and the values are whole numbers below 2^24, exact in every
one of those formats. Everything ELSE in the kernel stays in its native fp16
domain end to end -- the maxima, the equality compares, the ramp multiplies
never leave it.

DEVICE LAWS THIS SHAPE OBEYS
----------------------------
* THE WHOLE ROW IS ONE TILE: the row is presented (and loaded) as [1, PAD_V]
  in ONE descriptor block, exactly as rmsnorm.py loads [BLOCK_M, D_MODEL] --
  the descriptor's iteration space carries the width, so there is NO
  per-stick loop to unroll and the door receives a straight-line program of
  about twenty ops (five reduces, the compare/multiply chain, the island,
  the store). A per-stick-loop form would splice to ~2300 trips and tens of
  thousands of descriptors; this form is the door-lowerable one.
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
SIX staged buffers, all host obligations for the same reason embedding.py
pads nothing in-kernel and attention_sdpa.py's causal mask is
host-precomputed: the buffers are staging rows the host already owns, and
`other=`-masked loads are not in this frontend's vocabulary. The first five
are CONSTANT across decode steps (functions of V alone), so under the
pooled-substrate regime they live in stable slots and every step's CB reads
the same addresses:

* THE LOGITS ROW is presented as [PAD_V] fp16, PAD_V = the vocab rounded UP
  to a power-of-two whole row (the descriptor block must be a power of two),
  with the pad lanes already holding -inf (the fp16 floor). Real lanes are
  untouched: adding -inf past the row cannot change the max, and a pad lane
  can never win the equality against a finite m.
* THE BLOCK RAMP is [PAD_V] fp16 with bramp_i = NB - (i // 64) on the real
  lanes and 0 on the pad lanes -- strictly decreasing in the block index, so
  the first-block law is its monotonicity.
* THE LANE RAMP is [PAD_V] fp16 with lramp_i = 64 - (i % 64) on the real
  lanes and 0 on the pad lanes -- strictly decreasing in the lane index
  within each block, so the first-lane law is its monotonicity. It is the
  same 64-value pattern in every block, staged expanded because a [1, 64]
  stick cannot broadcast against a [1, PAD_V] tile.
* THE BLOCK INDEX TABLE is [PAD_V] fp16 with binc_i = i // 64 on the real
  lanes and NB on the pad lanes -- 0-based and increasing, so the one-hot
  reads b* straight out; the PAD value is the SENTINEL's block half (an
  all-NaN row's onehot lands on the pad lanes, so b* = NB there).
* THE LANE INDEX TABLE is [PAD_V] fp16 with linc_i = i % 64 on the real
  lanes and BLOCK on the pad lanes -- the lane half of the same law.
* THE WIDEN TABLES are [1, BLOCK] fp16 of 1.0 (the `* ones` widen, staging
  what a `tl.full` constant would splat, so the multiply stays a two-tensor
  op) and [1, BLOCK] fp32 of 64.0 (the island's scale, in the island's own
  dtype so the f32 multiply is also two-tensor).

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
PAD_V = 65536  # V rounded up to a power of two (the descriptor block's law)
BLOCK = 64  # the stick: the two-level split's lane width, and the output stick
NB = PAD_V // BLOCK  # the block count: bramp's ceiling, fp16-exact at 1024

NEG_INF = float("-inf")
DTYPE = torch.float16


@triton.jit
def argmax_first_max_kernel(logits_ptr, bramp_ptr, lramp_ptr, binc_ptr, linc_ptr,
                            ones_ptr, sixtyfour_ptr, idx_ptr,
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
    # The global max. `tl.max` of an f16 tile reduces in f32 (upstream's own
    # promotion) and is narrowed straight back, moe.py's spelling: the max of
    # f16 values is exactly representable in f16, so the round trip is exact.
    m = tl.max(x, axis=1).to(tl.float16)
    mm = tl.expand_dims(m, 1)

    # LEVEL 1 -- the FIRST 64-block holding the maximum. bramp is the staged
    # table (host contract): strictly decreasing in the block index, 0 on the
    # pad lanes, which can never win. mask * bramp <= NB = 1024: an exact
    # small-integer fp16 product (the docstring's two-level law -- a single
    # V - i ramp would need 49155 in a lane, past fp16's exact range).
    br_desc = tl.make_tensor_descriptor(bramp_ptr, shape=[1, PAD_V], strides=[PAD_V, 1],
                                        block_shape=[1, PAD_V])
    bramp = br_desc.load([0, 0])
    mask = (x == mm).to(tl.float16)
    w1 = mask * bramp
    jA = tl.max(w1, axis=1).to(tl.float16)
    # The equality inverts the ramp: bramp is CONSTANT per block, so comparing
    # it against its own value at the first maximal block selects that block
    # and nothing else. No subtraction anywhere in the kernel.
    sel = (bramp == tl.expand_dims(jA, 1)).to(tl.float16)

    # LEVEL 2 -- the FIRST lane within block b*. lramp is the staged per-lane
    # table, strictly decreasing within each block; mask * sel * lramp <= 64.
    lr_desc = tl.make_tensor_descriptor(lramp_ptr, shape=[1, PAD_V], strides=[PAD_V, 1],
                                        block_shape=[1, PAD_V])
    lramp = lr_desc.load([0, 0])
    w2 = mask * sel * lramp
    jB = tl.max(w2, axis=1).to(tl.float16)
    # lramp is PERIODIC (the same 64 values in every block), so this equality
    # selects lane l* of EVERY block -- and intersected with sel (block b*
    # only) it is the one-hot at the first maximal lane. The mask is already
    # carried by jA and jB, so it is not needed here.
    selL = (lramp == tl.expand_dims(jB, 1)).to(tl.float16)
    onehot = sel * selL

    # READ THE NUMBERS OUT through the one-hot: binc and linc are the staged
    # INCREASING 0-based index tables, so the max of the masked product IS the
    # selected block / lane number -- including the b* == 0 case, where an
    # all-zero product row's max is 0 and 0 is the answer.
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
    b64 = tl.expand_dims(bstar, 1) * ones
    l64 = tl.expand_dims(lstar, 1) * ones

    # THE WIDE INTEGER RIDES THE FP32 ISLAND, ONCE: b* * 64 + l* reaches 49154,
    # past fp16's exact range, so the combine widens, multiplies-adds in f32
    # (exact below 2^24) against a staged f32 scale table, and casts to i32 --
    # moe.py's gelu island and the KSPLIT pair's own device recipe.
    sf_desc = tl.make_tensor_descriptor(sixtyfour_ptr, shape=[1, BLOCK], strides=[BLOCK, 1],
                                        block_shape=[1, BLOCK])
    sf = sf_desc.load([0, 0])
    idx = (b64.to(tl.float32) * sf + l64.to(tl.float32)).to(tl.int32)
    i_desc = tl.make_tensor_descriptor(idx_ptr, shape=[1, BLOCK], strides=[BLOCK, 1],
                                       block_shape=[1, BLOCK])
    i_desc.store([0, 0], idx)


def inputs(seed: int = 0):
    """The six staged buffers (host contract): one fp16 logits row, pad
    lanes -inf, with a deliberate TIE at the max -- lanes 40017 and 48000 both
    hold the row's largest value, so the reference pins FIRST-max against a
    real tie, not a vacuous one -- plus the four constant index tables and the
    two widen tables.

    The tie lanes are HIGH on purpose, and that is a second pin: the answer
    40017 = 625*64 + 17 is NOT representable in fp16 (the format's ULP is 32
    at [32768, 65536)), so a kernel whose final combine degraded to f16
    answers 40032 or 40016 and the exact comparison sees it. A tie at low
    lanes (the original 100/5000) cannot discriminate that: 100 = 1*64 + 36
    sits inside fp16's exact-integer range, so a degraded combine still
    answers it exactly and the defect hides."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(V, generator=g, dtype=DTYPE)
    # ONE ABOVE the natural maximum: a 49k-lane fp16 randn row ties its own
    # maximum naturally (the format's tail is coarse), so pinning the tie AT
    # the natural max would leave an earlier natural lane winning. +1 is exact
    # in fp16 at randn magnitudes and strictly above every natural lane.
    top = x.max() + 1.0
    x[40017] = top
    x[48000] = top
    row = torch.full((PAD_V,), NEG_INF, dtype=DTYPE)
    row[:V] = x
    bramp = torch.zeros(PAD_V, dtype=DTYPE)
    lramp = torch.zeros(PAD_V, dtype=DTYPE)
    binc = torch.full((PAD_V,), NB, dtype=DTYPE)
    linc = torch.full((PAD_V,), BLOCK, dtype=DTYPE)
    for i in range(V):
        bramp[i] = NB - (i // BLOCK)
        lramp[i] = BLOCK - (i % BLOCK)
        binc[i] = i // BLOCK
        linc[i] = i % BLOCK
    ones = torch.ones((1, BLOCK), dtype=DTYPE)
    sixtyfour = torch.full((1, BLOCK), 64.0, dtype=torch.float32)
    return row, bramp, lramp, binc, linc, ones, sixtyfour


def reference(row: torch.Tensor) -> torch.Tensor:
    """First-max argmax, derived: the equality mask's first true index. Pad
    lanes are -inf and can never tie a finite max; an all-NaN real row leaves
    the mask empty and answers the sentinel (one past the padded row),
    matching the kernel's arithmetic. Returned as a 1-lane int32 tensor (the
    ingestion dtype)."""
    x = row[:V]
    m = x.max()
    mask = x == m
    hit = torch.nonzero(mask)
    idx = hit[0].item() if hit.numel() > 0 else PAD_V + BLOCK
    return torch.tensor([idx], dtype=torch.int32)
