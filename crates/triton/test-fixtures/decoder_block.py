# Copyright (c) 2026 IBM Corporation. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""A WHOLE Granite decoder block in one kernel -- and the SAME block fused TWICE.

    x = x + rm * (attn(rmsnorm(x) -> q,k,v -> rope -> sdpa -> Wo))
    x = x + rm * (mlp(rmsnorm(x)))

`_decoder_layer` below is that block. Two kernels call it:

    decoder_layer_fwd        calls it ONCE   <- the per-layer kernel
    decoder_two_layers_fwd   calls it TWICE  <- the fusion experiment

Both compile from this one source, so the difference between them is exactly one call and
nothing else. Structure per this repo's own
`journal/artifacts/granite-block/2026-07-10-granite-block-plan.md`, whose constants come from
`ibm-granite/granite-3.3-8b-instruct`'s config.json: `residual_multiplier = 0.22` on BOTH
residuals, `attention_multiplier = 0.0078125` (= 1/128, the SDPA scale, NOT 1/sqrt(d_h)),
`rms_norm_eps = 1e-05`, and `attention_bias = mlp_bias = false` -- so every projection is a
pure matmul with no bias term.

===============================================================================
FOUR FINDINGS THIS FIXTURE PRODUCED. Two are structural facts about fusing at all.
===============================================================================

**1. RoPE FUSED WITH THE PROJECTION NEEDS NO SLICE, BECAUSE THE SPLIT MOVES ONTO THE
WEIGHT.** `rope.py` gets its two halves as two descriptor loads at column offsets 0 and
HALF. That works because q is IN MEMORY there. Here q is a `tl.dot` result -- an in-register
tile -- and taking half of one of those is a bounded slice, which Triton refuses outright
(`unsupported tensor index`; measured, see `rope.py`'s docstring). The fix is not a
construct, it is a factorisation:

    q1 = dot(h, Wq[:, :HALF])       two dots against two COLUMN BLOCKS of the weight
    q2 = dot(h, Wq[:, HALF:])       -- and a weight IS in memory, so those are descriptor
                                       loads at offsets 0 and HALF, exactly like rope.py

so the rotation `q1*cos - q2*sin` / `q2*cos + q1*sin` never needs a slice. The contraction
that follows is then done in two pieces, `dot(q1r, k1r.T)` accumulated with
`dot(q2r, k2r.T)`, which is the same sum over head_dim taken in two halves.

**2. FUSING TWO LAYERS CONSTRAINS THE GRID, AND THAT IS A DATAFLOW FACT, NOT A COMPILER
LIMIT.** Layer 2's attention reads layer 1's output AT EVERY KV POSITION. A kernel whose
program holds only one query block does not have those values -- they are in other programs'
registers -- and Triton has no cross-program barrier. So a fused multi-layer kernel is
well-defined ONLY when one program holds every position it will attend to. This fixture
therefore sets `M = N_TOK`: one program, the whole (short) sequence. Anything longer must
either write the intermediate hidden state to memory and start a new kernel -- which is
one-kernel-per-layer -- or recompute layer 1 for every KV block inside layer 2's loop.
The choice is forced by the dataflow; it is not a property of this bridge.

**3. ONE HEAD, DELIBERATELY, AND THE REASON IS ALSO A MISSING CONSTRUCT.** `H = 1`, so
`head_dim == D_MODEL` and the attention output feeds Wo directly. With more heads each head
produces a `[M, head_dim]` tile and they have to be assembled into one `[M, D_MODEL]` tile
before Wo -- and there is NO CONCATENATE in this front end (`rope.py` hit the same wall from
the other side). The two spellings are a scratch round-trip (store each head's columns, load
the whole tile back) or `tl.cat`/`tl.join`, none of which the bridge has. That is the next
construct a multi-head fused block needs, named here so it is not rediscovered.

**4. THE SOFTMAX IS ONE PASS, NOT A FLASH RECURRENCE, and that follows from finding 2.**
With every position in one program the score tile is the whole `[M, M]`, so `max`, `exp2`,
`sum` and the divide are exact in one go: no running maximum, no `alpha` correction, no
loop-carried `l_i`. `attention_flash.py` keeps the recurrence because it tiles the KV axis;
here there is nothing to tile. This kernel is therefore NOT a substitute for that one -- it
is what fusion looks like at a length that fits.

THE DELTA against the HF block, and the justification for each item
===================================================================

1. THE CAUSAL MASK IS A LOADED f16 TILE, additive, position-independent -- verbatim
   `attention_flash.py`'s delta 10, for its reason: there is no lane-wise integer compare on
   this device, so the predicate is precomputed on the host and arrives as `qk + mask`. With
   one program covering all positions the tile is the full `[M, M]` lower triangle, 0 on and
   below the diagonal and -1.0e4 above it (finite, per that fixture's delta 3: -1e6 overflows
   f16 to -inf and an all-masked row then yields NaN).

2. EVERY SCALAR IS constexpr (SPYRE-SPECIFIC): the SDPA scale, the residual multiplier, eps,
   and the norm's reciprocal. `attention_flash.py`'s delta 9 has the measurement -- a compute
   group can only read values it LOADS, so a scalar reaching a compute op as an operand has to
   be routed through memory as a caller-seeded splat, and that routing needs the value at
   compile time. These are all config constants, so nothing is lost.

3. f16 THROUGHOUT, with f32 ISLANDS only around `exp2` and `rsqrt` (delta 2 of
   `attention_flash.py`, delta 1 of `rmsnorm.py`). `tl.dot` carries `out_dtype=tl.float16`
   because the device MAC rounds to DL16 at every step.

4. `tl.fdiv` FOR THE SOFTMAX DIVIDE, not `/` -- `swiglu_mlp.py`'s delta 6. A bare `/` on f16
   is promoted to f32 by Triton for a reason that is a PTX fact and false on this device, and
   the promotion would then make the downstream `tl.dot` illegal. `tl.fdiv` is the same
   `create_fdiv` with the arithmetic check off.

5. THE MLP IS `swiglu_mlp.py`'s BODY, including its delta 5 (the sigmoid written as a widen /
   exp / truncate island, because `tl.sigmoid` cannot be called on f16) and delta 8
   (`silu(gate) * up`, not upstream's `s * (up + 1)`). One loop over d_ff, so the
   `[M, D_FF]` intermediate never exists.

6. NOT DONE: GQA. `H = 1` (finding 3), so there is no kv-head sharing to express. Granite is
   32 query heads over 8 kv heads; that is `attention_flash.py`'s delta 4 and it belongs to
   the multi-head form this fixture explicitly does not have.

8. EVERY PROJECTION WEIGHT IS n-MAJOR AND EVERY PROJECTION DOT OPERAND CARRIES `.T`, so
   every one of those `linalg.matmul`s states `indexing_maps = [(d0,d2), (d1,d2), (d0,d1)]`
   rather than relying on MLIR's default. Seven buffers, nine loads: `wq` and `wk` (two row
   blocks each), `wv`, `wo`, `wg`, `wu`, `wd`.

   WHY, and it is device evidence rather than a preference. An ABSENT `indexing_maps` IS the
   default `[(d0,d2), (d2,d1), (d0,d1)]`, i.e. W as `[k, n]`; the explicit form above says W
   as `[n, k]`. The DDLs cannot settle which the device wants: `bmm.ddl`'s layout is unreachable
   from V1 (the dataflow scheduler has no contraction op -- `KTIRLegalityCheck.cpp:103`, "V1 only
   supports add/mul/sub compute ops"), what is emitted is a multiply plus a fold tree governed by
   the ELEMENTWISE templates, and `broadcast_ops.ddl:6-7` fixes no axis order. The only pairing
   with device evidence behind it is scratchy's -- `[n, k]` bytes read with transpose-B maps,
   Granite at 41 tok/s.

   ⛔ THE SENTENCE THAT USED TO STAND HERE -- "nothing in `ktir-superdsc` reads the maps, so both
   forms get the SAME descriptor and one of them must therefore be computing the transposed
   contraction" -- IS HALF RIGHT AND THE OTHER HALF IS BACKWARDS, and the correction is the whole
   point of the delta. The maps ARE read now (`whole_function::matmul_b_orientation`), and what
   they decide is which of the W REGION's two extents is K. The DEVICE statement really is the
   same for both forms, and it is `[k, n]`: `matmul` declares the kernel operand
   `StickLayout::kernel(k_in, n_out)`, whose address law is `dev_off_stk`'s rank-2 stick-blocked
   form with `dims[0] = in = K` as the ROW count. `sdsc_abstract::vcache_write_offset` states it
   for the hardware-proven attention -- "the V cache ... the VALUE bmm reads as a `[cap, hd]`
   KERNEL sticked on `hd` (`out`) ... the kernel is `[k=cap, n=hd]`" -- and its twin
   `kcache_kt_write_offset` says the score leg's kernel is `[hd, cap]`, in-rows again, which is
   WHY the K cache is written already-transposed. So the PLAIN form is the one the slot reads
   directly, and a transpose-B weight is the one whose bytes somebody has to place in that order.

   THE HOST CONTRACT, and it is a REAL change even where the numbers do not move: `wq`, `wk`,
   `wv` and `wo` are square `[D_MODEL, D_MODEL]`, so their declared `shape` and `strides` are
   textually identical before and after while their MEANING transposes from `[in, out]` to
   `[out, in]`. A host that keeps feeding the old bytes gets a well-formed kernel computing the
   transposed projection, with nothing to catch it. `wg`/`wu` become `[D_FF, D_MODEL]` and `wd`
   becomes `[D_MODEL, D_FF]`. All seven are therefore `[out_features, in_features]` -- exactly
   `torch.nn.Linear.weight`, so the host owes no transpose.

   TWO MATMULS ARE DELIBERATELY NOT TOUCHED, and neither is an omission:
     * the two SCORE matmuls (`q1r @ k1r.T`, `q2r @ k2r.T`) were ALREADY transpose-B, because
       `k1r`/`k2r` are in-register RoPE results and `.T` is how the head_dim contraction is
       spelled. They are the shape this delta moves everything else TO.

       ⛔ AND THEY ARE THE ONE PLACE WHERE THE `.T` IS A REAL DATA MOVEMENT RATHER THAN A LABEL.
       A transpose-B B is `[n, k]` bytes read by a slot that addresses `[k, n]`, so the
       transposition is spent OUTSIDE the descriptor: for a presented weight, once, by the host
       stage (`stage_2d(&StickLayout::kernel(k, n), ..)`); for a CACHE, by the device writing it
       already transposed (`kcache_kt_write_offset`). `k1r`/`k2r` are NEITHER -- they are
       in-register values with no host and no cache write -- and nothing in `ktir-superdsc` reads
       an access tile's `CoordinateOrder`, so a LABEL has nowhere to be spent here.

       ⛔⛔⛔ THIS USED TO SAY "so these two contract as `q1r @ k1r`", AND THAT WAS A SILENT WRONG
       ANSWER, NOW FIXED AT THE PASS THAT CAUSED IT. `dot_to_linalg` folded EVERY `.T` on a dot's
       B into `indexing_maps`, including these two, and the fold DROPPED them -- invisible to every
       extent guard because `[M, HALF]` is square here (64x64). It now folds ONLY when the
       transposed value is a direct `tt.descriptor_load` (the one form the host stage places), so
       these two keep their `tt.trans`, `to_ktir::convert_trans` makes it a real
       `linalg.transpose`, and each score matmul states NO maps -- MLIR's plain `[k, n]` -- over a
       buffer that is physically `[k, n]`. MEASURED on the `.py` path: `decoder_layer_one_flat`
       goes 50 -> 52 programs (`22_transpose_o31` / `23_matmul_o32`, `24_transpose_o33` /
       `25_matmul_o34`) and the 22 programs before them are byte-identical.

       ⛔ THE PRICE IS AN ARCH FEATURE, AND IT IS A DDL-TEMPLATE FACT RATHER THAN A MYSTERY.
       deeptools picks a DDL per opFunc per ISA (`ddc/ddl/ddl_conversion.h`): `BATCHMATMUL_FWD` has
       a `{"bmm_dd1.ddl", MPW4_ISA}` row carrying no `ddl.implicit_sync`, which is why every matmul
       schedules -- while both relayouts point MPW4 at a template that has one
       (`restickify.ddl:80`, `inter_slice_transpose.ddl:75`, both unconditional) and
       `Ddc::finalizeOps` refuses any implicit sync below `RCUDD1A_ISA` (`ddcv1.cpp:3416`). There
       is no `restickify_dd1.ddl`. So these two configurations now REFUSE at `dxp_standalone` on
       MPW4 instead of compiling the wrong contraction, which is the better of the two.

       ⛔⛔⛔ AND READ THAT REFUSAL FOR WHAT IT IS: NOT A REGRESSION, AND NOT OUR DESCRIPTOR.
       MEASURED, `SENARCH=MPW4 -b sentient`, control `test_gather_1core` exit 0 / 125568 B in the
       same session:

           decoder_layer_one_flat   ops=52   dxp exit 1   no init_binary.bin
           decoder_two_layers_flat  ops=104  dxp exit 1   no init_binary.bin
           sbf-ddc: DtException: Implicit syncs not available for architectures prior to RCUDD1A,
             ddcv1.cpp line 3416
           sbf-run-scheduler-on-sdsc: failed on program 'sdsc_22'

       `sdsc_22` IS `22_transpose_o31` -- the RELAYOUT, the one op that spends the transposition.
       dxp refuses the DDL template that op selects on this ISA; it does not object to any matmul,
       any descriptor, or anything else this path emits, and the whole regression bar is
       byte-identical in the same run. So the program is now the CORRECT one and the blocker is a
       device feature with a name and a file:line. It replaced a bake that reported nine successes
       while two of them contracted `q1r @ k1r` where this source says `q1r @ k1r.T` -- a number
       nobody could act on. Seven of nine with two refusals that name `ddcv1.cpp:3416` is the
       better tree; do not "restore" the nine by putting the fold back.

       ⛔ THE ACTIVATION SLOT IS NOT AN ESCAPE FROM IT EITHER, and this is the trap to read before
       proposing one. `ddl.layout`'s `is_order_fixed` governs the ORDER of the dims LISTED in the op
       (`ddc/ddl/Dialect/DdlOps.td:106-116`); MEMBERSHIP is the list itself and is never negotiable.
       The `is_order_fixed=false` at `bmm_dd1.ddl:17` is the int8/fp8/int4 activation's SLICE layout
       over `(%in, %wrd#0, %wrdpd)` -- `mb` is not in that list -- while THIS kernel is f16 (delta 3),
       so its activation uses `:18` (`ddl.layout(%in)`, TRUE) with stick `:19` (`ddl.layout(%in)`,
       TRUE). The activation's stick is `%in` = the CONTRACTED axis in `bmm.ddl`, `bmm_dd1.ddl` and
       `bmm_sen1p5.ddl` for every dtype (and `bmm_sen1p5.ddl:18`'s own comment says why: "only allow
       IN channel in the stick until we implement strided load in L0LU"). Both native-contraction
       slots therefore pin an axis -- activation stick `%in`, kernel stick `%out` -- so moving K into
       the activation slot would hand it `[k, m]` bytes it reads as `[m, k]`, which is the same
       silent wrong answer from the other side.
     * `a = tl.dot(p, v)` NEEDS NO MOVE AT ALL, and that is the other half of the same law. Its
       second operand is not a presented weight -- `v` is this kernel's own V-projection result,
       a `[k, n]` tile -- and `[k, n]` IS what the kernel slot addresses. `p @ v` is exactly the
       contraction the shipped attention runs against its V cache in place at 41 tok/s. So this
       one `linalg.matmul` carries no maps because the plain form is the RIGHT one here, not
       because it is a form that had to be tolerated.

7. NOT DONE: the embedding and the lm_head. Those are the model boundaries
   (`embedding_multiplier = 12.0`, `logits_scaling = 16.0`), not the block --
   `embedding.py` is the first of them.
"""

import torch

import triton
import triton.language as tl


@triton.jit
def _decoder_layer(x,  #
                   desc_n1, desc_wq, desc_wk, desc_wv, desc_wo, desc_mask,  #
                   desc_cos, desc_sin,  #
                   desc_n2, desc_wg, desc_wu, desc_wd,  #
                   M: tl.constexpr, D_MODEL: tl.constexpr, D_FF: tl.constexpr,  #
                   BLOCK_N: tl.constexpr, HALF: tl.constexpr,  #
                   EPS: tl.constexpr, INV_D: tl.constexpr,  # SPYRE: delta 2
                   QK_SCALE: tl.constexpr, RM: tl.constexpr,  # SPYRE: delta 2
                   ):
    # ---- pre-attention RMSNorm (rmsnorm.py's body) -----------------------------
    n1 = desc_n1.load([0])
    ms = tl.sum(x * x, 1) * INV_D
    # SPYRE: delta 3 -- the f32 island is FORCED: `tl.rsqrt` is `@_check_dtype(["fp32",
    # "fp64"])`, so it cannot be called on f16 at all.
    r = tl.rsqrt((ms + EPS).to(tl.float32)).to(tl.float16)
    h = x * r[:, None] * n1[None, :]

    # ---- q / k / v projections, SPLIT INTO HALVES ON THE WEIGHT (finding 1) -----
    # SPYRE: delta 8 -- the weights are n-major, so each of these is a [HALF, D_MODEL] ROW block
    # of a [D_MODEL, D_MODEL] weight and the halves still come from descriptor OFFSETS with no
    # tensor ever sliced. `.T` makes the dot operand [D_MODEL, HALF] = [k, n].
    # SPYRE: delta 3 -- `out_dtype=tl.float16` on every dot: the device MAC rounds to DL16 at
    # every step, so there is no wide accumulator to keep.
    q1 = tl.dot(h, desc_wq.load([0, 0]).T, out_dtype=tl.float16)
    q2 = tl.dot(h, desc_wq.load([HALF, 0]).T, out_dtype=tl.float16)
    k1 = tl.dot(h, desc_wk.load([0, 0]).T, out_dtype=tl.float16)
    k2 = tl.dot(h, desc_wk.load([HALF, 0]).T, out_dtype=tl.float16)

    # ---- RoPE, split-half (rope.py's arithmetic, on in-register halves) ---------
    c = desc_cos.load([0, 0])
    s = desc_sin.load([0, 0])
    q1r = q1 * c - q2 * s
    q2r = q2 * c + q1 * s
    k1r = k1 * c - k2 * s
    k2r = k2 * c + k1 * s

    # ---- scores: the head_dim contraction in TWO PIECES (finding 1) -------------
    qk = tl.dot(q1r, k1r.T, out_dtype=tl.float16)
    qk = tl.dot(q2r, k2r.T, qk)
    # SPYRE: delta 1 -- the causal mask is a LOADED additive f16 tile, because there is no
    # lane-wise integer compare on this device; and delta 2 -- the SDPA scale is constexpr.
    # It is position-independent, so one tile serves every launch and both layers.
    qk = qk * QK_SCALE + desc_mask.load([0, 0])

    # ---- one-pass softmax (finding 4) ------------------------------------------
    m = tl.max(qk, 1).to(tl.float16)
    p = tl.math.exp2((qk - m[:, None]).to(tl.float32)).to(tl.float16)
    l = tl.sum(p, 1).to(tl.float16)
    # The v PROJECTION is a full-width dot; only q and k are split, because only they are
    # rotated (finding 1).
    v = tl.dot(h, desc_wv.load([0, 0]).T, out_dtype=tl.float16)  # SPYRE: delta 8
    a = tl.dot(p, v, out_dtype=tl.float16)
    # `(p @ v) / l` rather than `(p / l) @ v`: `l` is per ROW, so the two are the same value
    # and this one divides M*D_MODEL elements instead of M*M. `tl.fdiv`, not `/` (delta 4).
    a = tl.fdiv(a, l[:, None])  # SPYRE: delta 4

    # ---- output projection and the FIRST residual ------------------------------
    o = tl.dot(a, desc_wo.load([0, 0]).T, out_dtype=tl.float16)  # SPYRE: delta 8
    x = x + o * RM

    # ---- post-attention RMSNorm ------------------------------------------------
    n2 = desc_n2.load([0])
    ms2 = tl.sum(x * x, 1) * INV_D
    r2 = tl.rsqrt((ms2 + EPS).to(tl.float32)).to(tl.float16)
    h2 = x * r2[:, None] * n2[None, :]

    # ---- SwiGLU MLP -----------------------------------------------------------
    # SPYRE: delta 5 -- `swiglu_mlp.py`'s body verbatim, including ITS delta 5 (the sigmoid
    # written out as a widen / exp / truncate island, because `tl.sigmoid` cannot be called
    # on an f16 tensor at all) and its delta 8 (`silu(gate) * up`).
    acc = tl.zeros([M, D_MODEL], dtype=tl.float16)
    one = tl.full([M, BLOCK_N], 1.0, tl.float16)
    for n in tl.range(0, D_FF, BLOCK_N):
        # SPYRE: delta 8 -- n-major weights, `.T` on every dot operand.
        g = tl.dot(h2, desc_wg.load([n, 0]).T, out_dtype=tl.float16)
        u = tl.dot(h2, desc_wu.load([n, 0]).T, out_dtype=tl.float16)
        e = tl.exp((-g).to(tl.float32)).to(tl.float16)
        sig = tl.fdiv(g, one + e)
        acc = tl.dot(sig * u, desc_wd.load([0, n]).T, acc)

    # ---- the SECOND residual --------------------------------------------------
    return x + acc * RM


@triton.jit
def decoder_layer_fwd(desc_x, desc_o,  #
                      desc_n1, desc_wq, desc_wk, desc_wv, desc_wo, desc_mask,  #
                      desc_cos, desc_sin,  #
                      desc_n2, desc_wg, desc_wu, desc_wd,  #
                      M: tl.constexpr, D_MODEL: tl.constexpr, D_FF: tl.constexpr,  #
                      BLOCK_N: tl.constexpr, HALF: tl.constexpr,  #
                      EPS: tl.constexpr, INV_D: tl.constexpr,  #
                      QK_SCALE: tl.constexpr, RM: tl.constexpr,  #
                      ):
    """ONE layer. This is the per-layer kernel: the intermediate hidden state crosses the
    kernel boundary through memory, which is what makes any sequence length expressible
    (finding 2)."""
    x_desc = tl.make_tensor_descriptor(desc_x, shape=[M, D_MODEL], strides=[D_MODEL, 1],
                                       block_shape=[M, D_MODEL])
    o_desc = tl.make_tensor_descriptor(desc_o, shape=[M, D_MODEL], strides=[D_MODEL, 1],
                                       block_shape=[M, D_MODEL])
    n1_desc = tl.make_tensor_descriptor(desc_n1, shape=[D_MODEL], strides=[1],
                                        block_shape=[D_MODEL])
    n2_desc = tl.make_tensor_descriptor(desc_n2, shape=[D_MODEL], strides=[1],
                                        block_shape=[D_MODEL])
    wq_desc = tl.make_tensor_descriptor(desc_wq, shape=[D_MODEL, D_MODEL],
                                        strides=[D_MODEL, 1], block_shape=[HALF, D_MODEL])
    wk_desc = tl.make_tensor_descriptor(desc_wk, shape=[D_MODEL, D_MODEL],
                                        strides=[D_MODEL, 1], block_shape=[HALF, D_MODEL])
    wv_desc = tl.make_tensor_descriptor(desc_wv, shape=[D_MODEL, D_MODEL],
                                        strides=[D_MODEL, 1], block_shape=[D_MODEL, D_MODEL])
    wo_desc = tl.make_tensor_descriptor(desc_wo, shape=[D_MODEL, D_MODEL],
                                        strides=[D_MODEL, 1], block_shape=[D_MODEL, D_MODEL])
    mask_desc = tl.make_tensor_descriptor(desc_mask, shape=[M, M], strides=[M, 1],
                                          block_shape=[M, M])
    cos_desc = tl.make_tensor_descriptor(desc_cos, shape=[M, HALF], strides=[HALF, 1],
                                         block_shape=[M, HALF])
    sin_desc = tl.make_tensor_descriptor(desc_sin, shape=[M, HALF], strides=[HALF, 1],
                                         block_shape=[M, HALF])
    wg_desc = tl.make_tensor_descriptor(desc_wg, shape=[D_FF, D_MODEL], strides=[D_MODEL, 1],
                                        block_shape=[BLOCK_N, D_MODEL])
    wu_desc = tl.make_tensor_descriptor(desc_wu, shape=[D_FF, D_MODEL], strides=[D_MODEL, 1],
                                        block_shape=[BLOCK_N, D_MODEL])
    wd_desc = tl.make_tensor_descriptor(desc_wd, shape=[D_MODEL, D_FF], strides=[D_FF, 1],
                                        block_shape=[D_MODEL, BLOCK_N])

    x = x_desc.load([0, 0])
    x = _decoder_layer(x, n1_desc, wq_desc, wk_desc, wv_desc, wo_desc, mask_desc,
                       cos_desc, sin_desc, n2_desc, wg_desc, wu_desc, wd_desc,
                       M, D_MODEL, D_FF, BLOCK_N, HALF, EPS, INV_D, QK_SCALE, RM)
    o_desc.store([0, 0], x)


@triton.jit
def decoder_two_layers_fwd(desc_x, desc_o,  #
                           desc_n1a, desc_wqa, desc_wka, desc_wva, desc_woa,  #
                           desc_n2a, desc_wga, desc_wua, desc_wda,  #
                           desc_n1b, desc_wqb, desc_wkb, desc_wvb, desc_wob,  #
                           desc_n2b, desc_wgb, desc_wub, desc_wdb,  #
                           desc_mask, desc_cos, desc_sin,  #
                           M: tl.constexpr, D_MODEL: tl.constexpr, D_FF: tl.constexpr,  #
                           BLOCK_N: tl.constexpr, HALF: tl.constexpr,  #
                           EPS: tl.constexpr, INV_D: tl.constexpr,  #
                           QK_SCALE: tl.constexpr, RM: tl.constexpr,  #
                           ):
    """TWO layers, fused. The hidden state between them NEVER TOUCHES MEMORY -- which is the
    whole point, and also why `M = N_TOK` is forced (finding 2).

    The mask and the rope tables are SHARED: the mask is position-independent and the tables
    are position-only, so one of each serves both layers.
    """
    x_desc = tl.make_tensor_descriptor(desc_x, shape=[M, D_MODEL], strides=[D_MODEL, 1],
                                       block_shape=[M, D_MODEL])
    o_desc = tl.make_tensor_descriptor(desc_o, shape=[M, D_MODEL], strides=[D_MODEL, 1],
                                       block_shape=[M, D_MODEL])
    mask_desc = tl.make_tensor_descriptor(desc_mask, shape=[M, M], strides=[M, 1],
                                          block_shape=[M, M])
    cos_desc = tl.make_tensor_descriptor(desc_cos, shape=[M, HALF], strides=[HALF, 1],
                                         block_shape=[M, HALF])
    sin_desc = tl.make_tensor_descriptor(desc_sin, shape=[M, HALF], strides=[HALF, 1],
                                         block_shape=[M, HALF])

    n1a = tl.make_tensor_descriptor(desc_n1a, shape=[D_MODEL], strides=[1],
                                    block_shape=[D_MODEL])
    n2a = tl.make_tensor_descriptor(desc_n2a, shape=[D_MODEL], strides=[1],
                                    block_shape=[D_MODEL])
    wqa = tl.make_tensor_descriptor(desc_wqa, shape=[D_MODEL, D_MODEL], strides=[D_MODEL, 1],
                                    block_shape=[HALF, D_MODEL])
    wka = tl.make_tensor_descriptor(desc_wka, shape=[D_MODEL, D_MODEL], strides=[D_MODEL, 1],
                                    block_shape=[HALF, D_MODEL])
    wva = tl.make_tensor_descriptor(desc_wva, shape=[D_MODEL, D_MODEL], strides=[D_MODEL, 1],
                                    block_shape=[D_MODEL, D_MODEL])
    woa = tl.make_tensor_descriptor(desc_woa, shape=[D_MODEL, D_MODEL], strides=[D_MODEL, 1],
                                    block_shape=[D_MODEL, D_MODEL])
    wga = tl.make_tensor_descriptor(desc_wga, shape=[D_FF, D_MODEL], strides=[D_MODEL, 1],
                                    block_shape=[BLOCK_N, D_MODEL])
    wua = tl.make_tensor_descriptor(desc_wua, shape=[D_FF, D_MODEL], strides=[D_MODEL, 1],
                                    block_shape=[BLOCK_N, D_MODEL])
    wda = tl.make_tensor_descriptor(desc_wda, shape=[D_MODEL, D_FF], strides=[D_FF, 1],
                                    block_shape=[D_MODEL, BLOCK_N])

    n1b = tl.make_tensor_descriptor(desc_n1b, shape=[D_MODEL], strides=[1],
                                    block_shape=[D_MODEL])
    n2b = tl.make_tensor_descriptor(desc_n2b, shape=[D_MODEL], strides=[1],
                                    block_shape=[D_MODEL])
    wqb = tl.make_tensor_descriptor(desc_wqb, shape=[D_MODEL, D_MODEL], strides=[D_MODEL, 1],
                                    block_shape=[HALF, D_MODEL])
    wkb = tl.make_tensor_descriptor(desc_wkb, shape=[D_MODEL, D_MODEL], strides=[D_MODEL, 1],
                                    block_shape=[HALF, D_MODEL])
    wvb = tl.make_tensor_descriptor(desc_wvb, shape=[D_MODEL, D_MODEL], strides=[D_MODEL, 1],
                                    block_shape=[D_MODEL, D_MODEL])
    wob = tl.make_tensor_descriptor(desc_wob, shape=[D_MODEL, D_MODEL], strides=[D_MODEL, 1],
                                    block_shape=[D_MODEL, D_MODEL])
    wgb = tl.make_tensor_descriptor(desc_wgb, shape=[D_FF, D_MODEL], strides=[D_MODEL, 1],
                                    block_shape=[BLOCK_N, D_MODEL])
    wub = tl.make_tensor_descriptor(desc_wub, shape=[D_FF, D_MODEL], strides=[D_MODEL, 1],
                                    block_shape=[BLOCK_N, D_MODEL])
    wdb = tl.make_tensor_descriptor(desc_wdb, shape=[D_MODEL, D_FF], strides=[D_FF, 1],
                                    block_shape=[D_MODEL, BLOCK_N])

    x = x_desc.load([0, 0])
    x = _decoder_layer(x, n1a, wqa, wka, wva, woa, mask_desc, cos_desc, sin_desc,
                       n2a, wga, wua, wda,
                       M, D_MODEL, D_FF, BLOCK_N, HALF, EPS, INV_D, QK_SCALE, RM)
    x = _decoder_layer(x, n1b, wqb, wkb, wvb, wob, mask_desc, cos_desc, sin_desc,
                       n2b, wgb, wub, wdb,
                       M, D_MODEL, D_FF, BLOCK_N, HALF, EPS, INV_D, QK_SCALE, RM)
    o_desc.store([0, 0], x)


# --- the Spyre configuration ---------------------------------------------------
# Granite-3.3 8B's block constants, from its config.json.
RMS_NORM_EPS = 1e-05
RESIDUAL_MULTIPLIER = 0.22
ATTENTION_MULTIPLIER = 0.0078125  # 1/128, the SDPA scale -- NOT 1/sqrt(head_dim)
LOG2E = 1.44269504  # folded into the scale so `exp2` needs no extra multiply
BLOCK_N = 64
MASK_FILL = -1.0e4  # finite, per attention_flash.py's delta 3

ONE_LAYER_SIGNATURE = {
    "desc_x": "*fp16", "desc_o": "*fp16",
    "desc_n1": "*fp16", "desc_wq": "*fp16", "desc_wk": "*fp16", "desc_wv": "*fp16",
    "desc_wo": "*fp16", "desc_mask": "*fp16",
    "desc_cos": "*fp16", "desc_sin": "*fp16",
    "desc_n2": "*fp16", "desc_wg": "*fp16", "desc_wu": "*fp16", "desc_wd": "*fp16",
    "M": "constexpr", "D_MODEL": "constexpr", "D_FF": "constexpr",
    "BLOCK_N": "constexpr", "HALF": "constexpr",
    "EPS": "constexpr", "INV_D": "constexpr", "QK_SCALE": "constexpr", "RM": "constexpr",
}

TWO_LAYER_SIGNATURE = {
    "desc_x": "*fp16", "desc_o": "*fp16",
    "desc_n1a": "*fp16", "desc_wqa": "*fp16", "desc_wka": "*fp16", "desc_wva": "*fp16",
    "desc_woa": "*fp16", "desc_n2a": "*fp16", "desc_wga": "*fp16", "desc_wua": "*fp16",
    "desc_wda": "*fp16",
    "desc_n1b": "*fp16", "desc_wqb": "*fp16", "desc_wkb": "*fp16", "desc_wvb": "*fp16",
    "desc_wob": "*fp16", "desc_n2b": "*fp16", "desc_wgb": "*fp16", "desc_wub": "*fp16",
    "desc_wdb": "*fp16",
    "desc_mask": "*fp16", "desc_cos": "*fp16", "desc_sin": "*fp16",
    "M": "constexpr", "D_MODEL": "constexpr", "D_FF": "constexpr",
    "BLOCK_N": "constexpr", "HALF": "constexpr",
    "EPS": "constexpr", "INV_D": "constexpr", "QK_SCALE": "constexpr", "RM": "constexpr",
}


def constexprs(m=64, d_model=128, d_ff=256):
    """One configuration. `M` is BOTH the block and the sequence length (finding 2)."""
    return {"M": m, "D_MODEL": d_model, "D_FF": d_ff,
            "BLOCK_N": BLOCK_N, "HALF": d_model // 2,
            "EPS": RMS_NORM_EPS, "INV_D": 1.0 / d_model,
            "QK_SCALE": ATTENTION_MULTIPLIER * LOG2E,
            "RM": RESIDUAL_MULTIPLIER}


def causal_mask(m: int, fill: float = MASK_FILL) -> torch.Tensor:
    """The additive causal mask the host owes the kernel (delta 1): 0 on and below the
    diagonal, `fill` above it. Position-independent, so ONE tile serves every launch and both
    layers."""
    keep = torch.tril(torch.ones(m, m, dtype=torch.float16))
    return torch.where(keep.bool(), torch.zeros(1, dtype=torch.float16),
                       torch.full((1,), fill, dtype=torch.float16))


def layer_weights(seed: int, d_model: int, d_ff: int):
    """One layer's seven weight tensors, in the [in, out] orientation the kernel reads."""
    g = torch.Generator().manual_seed(seed)
    scale = 1.0 / (d_model ** 0.5)
    return dict(
        n1=torch.ones(d_model, dtype=torch.float16),
        wq=(torch.randn(d_model, d_model, generator=g, dtype=torch.float16) * scale),
        wk=(torch.randn(d_model, d_model, generator=g, dtype=torch.float16) * scale),
        wv=(torch.randn(d_model, d_model, generator=g, dtype=torch.float16) * scale),
        wo=(torch.randn(d_model, d_model, generator=g, dtype=torch.float16) * scale),
        n2=torch.ones(d_model, dtype=torch.float16),
        wg=(torch.randn(d_model, d_ff, generator=g, dtype=torch.float16) * scale),
        wu=(torch.randn(d_model, d_ff, generator=g, dtype=torch.float16) * scale),
        wd=(torch.randn(d_ff, d_model, generator=g, dtype=torch.float16) * scale),
    )


def inputs(seed: int = 0, m=64, d_model=128, d_ff=256, layers=2, theta=1e7):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(m, d_model, generator=g, dtype=torch.float16)
    half = d_model // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, half, dtype=torch.float32) / half))
    freqs = torch.outer(torch.arange(m, dtype=torch.float32), inv_freq)
    return dict(x=x, mask=causal_mask(m),
                cos=freqs.cos().to(torch.float16), sin=freqs.sin().to(torch.float16),
                layers=[layer_weights(seed + 1 + i, d_model, d_ff) for i in range(layers)])


def reference(inp: dict, eps: float = RMS_NORM_EPS, rm: float = RESIDUAL_MULTIPLIER,
              scale: float = ATTENTION_MULTIPLIER) -> torch.Tensor:
    """Native PyTorch reference for the WHOLE block, once per layer.

    Written from HF's `GraniteDecoderLayer` in f32 -- `rmsnorm`, split-half rope, one-head
    SDPA with the additive causal mask and `attention_multiplier`, `silu(gate) * up`, and
    `residual_multiplier` on both residuals. NOT a re-implementation of the kernel's
    factorisation: the halves are taken with real slices here, so the kernel's
    weight-column split (finding 1) is compared against the plain form rather than against
    itself.
    """
    def rms(t, w):
        v = t.pow(2).mean(-1, keepdim=True)
        return (t * torch.rsqrt(v + eps)) * w

    def rope(t, cos, sin):
        d = t.shape[-1] // 2
        t1, t2 = t[..., :d], t[..., d:]
        return torch.cat((t1 * cos - t2 * sin, t2 * cos + t1 * sin), dim=-1)

    x = inp["x"].to(torch.float32)
    cos = inp["cos"].to(torch.float32)
    sin = inp["sin"].to(torch.float32)
    mask = inp["mask"].to(torch.float32)
    for w in inp["layers"]:
        f = {k: v.to(torch.float32) for k, v in w.items()}
        h = rms(x, f["n1"])
        q = rope(h @ f["wq"], cos, sin)
        k = rope(h @ f["wk"], cos, sin)
        v = h @ f["wv"]
        s = (q @ k.transpose(-1, -2)) * scale + mask
        p = torch.softmax(s, dim=-1)
        x = x + (p @ v) @ f["wo"] * rm
        h2 = rms(x, f["n2"])
        g = h2 @ f["wg"]
        u = h2 @ f["wu"]
        x = x + (torch.sigmoid(g) * g * u) @ f["wd"] * rm
    return x.to(torch.float16)
