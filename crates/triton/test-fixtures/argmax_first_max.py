# SPDX-License-Identifier: Apache-2.0
"""argmax_first_max fixture: the greedy sampler's vocab scan, FIRST-MAX by
arithmetic, as one kernel.

This is the ladder form of loop closure's device sampler (#603/#500): the
algorithm that used to live in a hand-minted 8-link SuperDDC composite
(#698, deleted by #714) is stated HERE, in the kernel, over generic ops only
-- whole-row `tt.reduce` max, an equality compare, staged-table multiplies,
the fp32 island for the one wide integer. Nothing argmax-specific exists
below the kernel: whatever lowering the splice emits for `tt.reduce`/
`arith.cmpf`/`arith.mulf` is the lowering this kernel gets, which is the
whole point.

THE ALGORITHM (two-level selection over one tile)
-------------------------------------------------
    m    = max of the logits row (whole row, one reduce)
    w1   = (x == m) * bramp          bramp_i = NB - (i // 64)   (first-BLOCK)
    jA   = max of w1                 b* = NB - jA  -- the FIRST 64-block of the row
                                     holding the maximum
    sel  = (bramp == b*_indicator)   one-hot on block b* only
    w2   = (x == m) * sel * lramp    lramp_i = 64 - (i % 64)    (first-LANE)
    jB   = max of w2                 l* = 64 - jB  -- the FIRST lane within block b*
                                     holding the maximum
    idx  = b* * 64 + l*              (computed in the fp32 island)

FIRST-MAX BY ARITHMETIC, not by a tie-break rule: the equality mask is true at
EVERY lane holding the maximum; bramp is STRICTLY DECREASING in the block
index, so the max of w1 lands on the SMALLEST block containing the maximum;
lramp is STRICTLY DECREASING in the lane index, so the max of w2 lands on the
SMALLEST lane within that block. No comparator chain, no index-carrying
reduce pair -- the tie law is the two ramps' monotonicity. This is #631's
first-max law (host argmax, metal, and the deleted composite all agree on it)
made structural.

WHY TWO LEVELS AND NOT ONE RAMP: a single ramp V - i would need values up to
49155 in a lane, and fp16 exact integers stop at 2048. The two-level split
keeps EVERY lane value in the fp16-exact range: bramp <= NB = 1024 and
lramp <= 64, and every product along the way (mask * bramp <= 1024,
mask * sel * lramp <= 64) is an exact small-integer fp16 product. This is
moe.py's own law -- "the lane ids ride f16 (exact to 2048; the real router's
E = 128)" -- restated at vocab width, where a single id cannot ride f16 but
the (block, lane) PAIR always can.

THE WIDE INTEGER RIDES THE FP32 ISLAND, ONCE: the final combine
`b* * 64 + l*` reaches 49154, past fp16's exact range, so it is computed as
`(b.to(f32) * 64 + l.to(f32)).to(int32)` -- the f16 -> f32 widen, one f32
multiply-add, and the f32 -> i32 cast. This is the exact island moe.py's
gelu runs (``g32 = g.to(tl.float32)`` for a cubic that overflows f16) and
rmsnorm.py's mean runs; on the device it is the KSPLIT pair's own recipe
(`dl16tofp32` -> same-stick f32 adds -> `fp32todl16`/`fp32toint32`), and the
values are whole numbers below 2^24, exact in every one of those formats.
Everything ELSE in the kernel stays in its native fp16 domain end to end --
the max, the equality compares, the ramp multiplies never leave it.

THE SENTINEL LAW (all-NaN row), derived not patched: a NaN lane never wins an
ordered max and never satisfies an ordered equality, so an all-NaN row leaves
the mask identically zero: jA = 0 (b* = NB), jB = 0 (l* = 64), and
idx = NB * 64 + 64 = PAD_V + 64 -- past the last real lane under either NaN
semantics (maxnumf-style combines leave m at the -inf of the pad lanes, whose
ramps are 0; NaN-propagating combines leave m itself NaN; the mask is zero
both ways). The caller treats idx > V as the loud refusal (the host argmax
answers token 0 over the same row; that divergence is recorded, not
reconciled here).

DEVICE LAWS THIS SHAPE OBEYS
----------------------------
* THE WHOLE ROW IS ONE TILE: the row is presented (and loaded) as [1, PAD_V]
  in ONE descriptor block, exactly as rmsnorm.py loads [BLOCK_M, D_MODEL] --
  the descriptor's iteration space carries the width, so there is NO
  per-stick loop to unroll and the door receives a straight-line program of
  about fifteen ops (three reduces, the compare/multiply chain, the island,
  the store). A per-stick-loop form would splice to ~2300 trips and tens of
  thousands of descriptors; this form is the door-lowerable one.
* EACH `tt.reduce` IS the golden rank-2 -> rank-1 form over the tile's second
  axis, and `tl.max` of an f16 tile reduces in f32 (upstream's own promotion)
  and is narrowed straight back, moe.py's spelling -- the widening island
  LegalizeTypes collapses.
* NO LOOPS AT ALL: not even `tl.static_range` -- every offset is the one
  [0, 0] load corner. (moe.py's law is that every loop is static; this kernel
  has none to keep static.)
* V, PAD_V, NB, BLOCK ARE constexpr (SPYRE-SPECIFIC): the emitter derives
  work division from descriptor extents, and Spyre compiles one kernel per
  shape. Same law as every other fixture.
* ONE PROGRAM (grid [1]): the sampler serves the solo greedy decode step's
  single [1, V] row. Batched/sampled steps keep the host sampler -- they need
  the whole distribution, not one index.

HOST CONTRACT (the part that is not in this file)
-------------------------------------------------
THREE staged buffers, all host obligations for the same reason embedding.py
pads nothing in-kernel and attention_sdpa.py's causal mask is
host-precomputed: the buffers are staging rows the host already owns, and
`other=`-masked loads are not in this frontend's vocabulary.

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
Both ramps are CONSTANT across decode steps (functions of V alone), so the
host stages them ONCE; under the pooled-substrate regime they live in
stable slots and every step's CB reads the same addresses. Computing them
in-kernel would need `V - (start + tl.arange(...))`-style arithmetic -- a
runtime scalar added to an arange in the tile domain, a form no fixture
carries (see the pass-2 comment in the v4 history and attention's aranges,
which are dead code in every golden).

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
def argmax_first_max_kernel(logits_ptr, bramp_ptr, lramp_ptr, idx_ptr,
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
    b = NB - jA  # the first maximal block, [1]

    # LEVEL 2 -- the FIRST lane within block b*. sel is one exactly on block
    # b*'s lanes (bramp is constant per block, so bramp == the block's own
    # indicator selects it and nothing else; the pad lanes' 0 never matches a
    # b* that a real lane won). lramp is the staged per-lane table, strictly
    # decreasing within each block; mask * sel * lramp <= 64.
    sel = (bramp == tl.expand_dims(jA, 1)).to(tl.float16)
    lr_desc = tl.make_tensor_descriptor(lramp_ptr, shape=[1, PAD_V], strides=[PAD_V, 1],
                                        block_shape=[1, PAD_V])
    lramp = lr_desc.load([0, 0])
    w2 = mask * sel * lramp
    jB = tl.max(w2, axis=1).to(tl.float16)
    l = BLOCK - jB  # the first maximal lane within b*, [1]

    # THE WIDE INTEGER RIDES THE FP32 ISLAND, ONCE: b * 64 + l reaches 49154,
    # past fp16's exact range, so the combine widens, multiplies-adds in f32
    # (exact below 2^24), and casts to i32 -- moe.py's gelu island and the
    # KSPLIT pair's own device recipe. The [1]s broadcast to the [1, BLOCK]
    # output stick through the unit-dim multiply (the tl.maximum broadcast
    # law: no tl.broadcast_to, no splat of an interior value).
    ones = tl.full([1, BLOCK], 1.0, tl.float16)
    b64 = tl.expand_dims(b, 1) * ones
    l64 = tl.expand_dims(l, 1) * ones
    idx = (b64.to(tl.float32) * 64.0 + l64.to(tl.float32)).to(tl.int32)
    o_desc = tl.make_tensor_descriptor(idx_ptr, shape=[1, BLOCK], strides=[BLOCK, 1],
                                       block_shape=[1, BLOCK])
    o_desc.store([0, 0], idx)


def inputs(seed: int = 0):
    """The three staged buffers (host contract): one fp16 logits row, pad
    lanes -inf, with a deliberate TIE at the max -- lanes 40017 and 48000 both
    hold the row's largest value, so the reference pins FIRST-max against a
    real tie, not a vacuous one -- plus the two constant ramps.

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
    for i in range(V):
        bramp[i] = NB - (i // BLOCK)
        lramp[i] = BLOCK - (i % BLOCK)
    return row, bramp, lramp


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
