# SPDX-License-Identifier: Apache-2.0
"""Scratchy's Triton greedy-argmax kernel — the vocab scan, FIRST-MAX by
arithmetic, as one kernel.

BODY PROVENANCE: `crates/triton/test-fixtures/argmax_first_max.py` verbatim — the
same whole-row equality-selection body that fixture baked through this repo's
KTIR→SuperDSC lowering (ops=21, `dxp_standalone -b sentient` clean) and that the
triton-numeric legs pin against a derived first-max reference. It is the Triton
form of the algorithm the deleted #698 composite hand-minted (max → equal →
ramp-mul → max → V-sub → fp32toint32; #714 deleted that emission because the
card path runs only what lowering Triton kernels produces): the same math,
stated in the kernel, over generic ops only.

THE SPLICE'S OWN CONTRACT (what `scratchy-triton-splice` states about this kernel):

* PARAMETERS, IN ORDER: `logits`, `bramp`, `lramp`, `binc`, `linc`, `ones`,
  `sixtyfour`, `padoff`, `gramp`, `ginc`, then `idx` — the node's operand order
  then the output. The registry does not permute. The nine tables after the
  logits row are STAGED buffers the host owns (the fixture's own host
  contract): constant across decode steps, functions of the vocab alone.
* CONSTEXPRS: `V` (the model's real vocab), `PAD_V` (V rounded up to a
  power-of-two whole row, STRICTLY above V — the descriptor block's law and
  the sentinel's), `NB = PAD_V // BLOCK`, `BLOCK = 64` — stated by the splice
  from the node's vocab, never hardcoded to a model: a different vocab bakes
  its own tables.
* GRID: `[1]` — the sampler serves the solo greedy decode step's single row.
* THE PAD LAW RIDES THE STAGED OFFSET TABLE (`padoff`): 0.0 on `[0, V)`, −inf
  on `[V, PAD_V)`. The kernel computes `xm = x + padoff` before the first max
  and uses `xm` throughout, so the row's own pad bytes are irrelevant — the
  lm-head's zero-pad columns write 0.0 into them every step, and `0 + −inf =
  −inf` all the same (a durable −inf tail survives too: `−inf + −inf = −inf`).
  A real lane is untouched (`x + 0` exact), NaN rides through (`NaN + 0 =
  NaN`, the all-NaN sentinel law intact), and an all-−inf row still answers
  lane 0 — every lane ties at −inf and the staged ramps pick the first,
  agreeing with the host reference's first-max. This is why the offset table
  and not a V-constexpr mask: the pad law is a HOST-STAGED fact like every
  other table this kernel reads, never an in-kernel arange.

THE ALGORITHM (equality selection over a TWO-LEVEL ramp, no inversions): the
block index is split GROUP-major — `G` groups of 64 blocks (4096 lanes) —
because a single flat decreasing block ramp would need NB = 8192 in a lane at
gemma-4's PAD_V 524288, past fp16's exact-integer range (2048). The staged
DECREASING ramps (gramp group-major, bramp periodic within each group, lramp
lane-minor) make each `max` land on the FIRST group / FIRST block / FIRST
lane — #631's first-max law by monotonicity — and the staged INCREASING
index tables (ginc, binc, linc) read the selected numbers back out through
the one-hot. Every staged lane value stays ≤ max(64, G) with G ≤ 2048 (the
admission gate's bound), an exact fp16 integer at any admitted vocab. The
wide combine `g*·4096 + b*·64 + l*` rides the fp32 island once (past fp16's
exact range at vocab width; exact in f32 below 2^24); everything else stays
f16 end to end. Full derivation and device laws: the fixture's module
docstring, which this body is.

THE SENTINEL LAW (all-NaN row): a NaN lane never wins an ordered max and never
satisfies an ordered equality, so an all-NaN row leaves the one-hot on the PAD
lanes, where the host stages ginc = G, binc = 64 and linc = BLOCK — idx =
G·4096 + 64·64 + 64 = PAD_V + 4160, past the last real lane. The consumer
treats idx > V as the loud refusal (the host argmax answers token 0 over the
same row; that divergence is #666, recorded, not reconciled). A finite row can
never reach the pad lanes.
"""

import triton
import triton.language as tl


@triton.jit
def argmax_fwd(logits_ptr, bramp_ptr, lramp_ptr, binc_ptr, linc_ptr,  #
               ones_ptr, sixtyfour_ptr, padoff_ptr, gramp_ptr, ginc_ptr, idx_ptr,  #
               V: tl.constexpr, PAD_V: tl.constexpr,  #
               NB: tl.constexpr, BLOCK: tl.constexpr):
    # THE WHOLE ROW IS ONE TILE: one load, [1, PAD_V], the rmsnorm fixture's own
    # whole-row block. Every shape below is rank 2; every reduce is the golden
    # rank-2 -> rank-1 form over the tile's second axis; no scalar, no
    # reduce-to-scalar, and no splat of an interior value exists anywhere in
    # the kernel. No loops at all — every offset is the one [0, 0] load corner.
    l_desc = tl.make_tensor_descriptor(logits_ptr, shape=[1, PAD_V], strides=[PAD_V, 1],
                                       block_shape=[1, PAD_V])
    x = l_desc.load([0, 0])
    # THE PAD LAW, BEFORE THE FIRST MAX: xm = x + padoff, and xm throughout.
    # padoff is 0.0 on the real lanes (x + 0 exact) and -inf on the pads, so a
    # pad lane can never win a max or an equality the real lanes contest
    # WHATEVER the row's pad bytes hold (the lm-head's zero-pad columns write
    # 0.0 there every step; 0 + -inf = -inf, and a durable -inf tail survives
    # as -inf + -inf = -inf). NaN rides through (NaN + 0 = NaN), so the
    # all-NaN sentinel law below is intact. A two-tensor add of two loaded
    # tiles -- the same Elementwise(Add) every spliced add kernel is.
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
    # mask * gramp <= G <= 2048: an exact small-integer fp16 product (the
    # admission gate's bound -- a single flat block ramp would need NB in a
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
