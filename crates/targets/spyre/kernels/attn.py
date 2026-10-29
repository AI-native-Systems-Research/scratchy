# SPDX-License-Identifier: Apache-2.0
"""Scratchy's Triton attention kernel: TORCH-SPYRE'S SDPA DECOMPOSITION over the two KV
segments the forward hands it.

THE ALGORITHM is torch-spyre's `spyre__sdpa_overrideable`
(torch_spyre/_inductor/decompositions.py:1960, at a9fa997b) -- the same port
`crates/triton/test-fixtures/sdpa.py` verifies at torch-spyre's own layout and tilings:

  1. Q is scaled ONCE, before any KV block (:2175).
  2. ONE KV block: the stable softmax, `out = exp(s - max) . v / sum` (:2178-2200).
  3. SEVERAL: the online-softmax scan, carry (running max -inf, denominator 0, output 0) and
     per block (:2220-2256)
         block_max  = max(s)
         correction = exp(min(running_max - block_max, 0))   -- before the new max
         new_max    = maximum(running_max, block_max)
         p          = exp(s - new_max)
         denom      = denom * correction + sum(p)
         output     = output * correction + p . v            -- the matmul, then the add
     and `output / denom` after the last (:2270).
  4. Masks are ADDITIVE tiles the host stages (:2191, :2240); nothing is compared on device.
  5. GQA reads KV head `h // GQA`; K/V are never repeated (:2012).
  6. Everything is f16; `tl.exp` takes f32, so each exp is widened and truncated straight back.

THE KV BLOCKS ARE THE NODE'S TWO SEGMENTS: the NEW block (this step's K/V, `desc_kd`/`desc_vd`)
and the RESIDENT PREFIX (the cache, `desc_kc`/`desc_vc`, `SWEPT` rows). A shape with no prefix
(`SWEPT == 0`, a first prompt chunk) is torch-spyre's one-block arm; every other shape is the
two-block scan. The prefix is ONE block of `SWEPT` rows -- the single-block decode plan
torch-spyre's planner returns up to 2048 keys -- and it must be: the door reads the swept rung
back off this tile's row extent (`param_read_rows`).

THE NEW BLOCK IS SCANNED FIRST, the one ordering choice that is not torch-spyre's. torch-spyre
scans keys in order and never meets an empty first block, because its first block holds key 0,
which every causal row attends. Here the first PREFIX block can be empty for a row -- a decode
step's length mask drives every column past the valid length to -inf -- and an empty first
block makes the correction `exp(min(-inf - -inf, 0))`, a NaN. The new block is never empty (a
decode step's own key is unmasked; the causal triangle keeps its diagonal), so scanning it
first leaves the running max finite before any prefix column is read, and a fully masked
prefix then contributes `exp(-inf) = 0` with correction 1. Softmax is invariant under a
permutation of its keys, so the order changes rounding, not the function.

THE LAUNCH CONTRACT is the splice's (`scratchy-triton-splice::lower_attn`), which reads every
fact the door needs off this program -- unchanged from the kernel this one replaces:

* PARAMETERS, IN ORDER: `desc_q`, `desc_o`, `desc_kc`, `desc_kd`, `desc_vc`, `desc_vd`,
  `desc_mask` -- the door's positional order. A shape that reads no mask never addresses
  `desc_mask` (the splice drops the parameter), and a shape with no prefix never addresses
  `desc_kc`/`desc_vc` (the splice injects their views).
* VIEWS: q and out are `[MQ, NQH*HD]`, the cache `[CAP, NKVH*HD]`, the new block
  `[NEW_LEN, NKVH*HD]`. Q's first tile is `[MQ, HD]` (the door reads the head dim off it), the
  prefix tile `[SWEPT, HD]` (the door reads the swept rung off it), one Q tile per head (the
  door refuses a program that grows with its rows).
* THE MASK, when bound, is `[MQ, MASK_W]` with the prefix's columns first: `[1, SWEPT]` at
  decode (the runtime length mask), `[MQ, MQ]` for a first prompt chunk (the causal triangle),
  `[MQ, SWEPT + MQ]` for a chunk with a prefix (prefix mask, then the new block's).
* GRID: `[NQH]` when `MQ > 1` (the program id is the head), `[1]` at decode with a trace-time
  head loop -- the splice's own split.
* THE SCALE is the Q multiply (torch-spyre's own spelling); the door reads it there, and a
  scale of 1.0 (folded into W_q/W_k at bake) is a multiply the pipeline folds away.
"""

import triton
import triton.language as tl


@triton.jit
def _scan_block(running_max, denominator, output, scores, v):
    """One KV block of torch-spyre's scan (`sdpa_lk_body`, decompositions.py:2242-2256), on
    the block's already-masked `[MQ, keys]` scores and its `[keys, HD]` values."""
    block_max = tl.max(scores, 1)
    # :2246 -- the correction reads the OLD max and the BLOCK max only.
    correction = tl.exp(
        tl.minimum(running_max - block_max, 0.0).to(tl.float32)).to(tl.float16)
    new_max = tl.maximum(running_max, block_max)
    exp_scores = tl.exp((scores - new_max[:, None]).to(tl.float32)).to(tl.float16)
    denominator = denominator * correction + tl.sum(exp_scores, 1).to(tl.float16)
    weighted = tl.dot(exp_scores, v, out_dtype=tl.float16)
    output = output * correction[:, None] + weighted
    return new_max, denominator, output


@triton.jit
def _attn_head(h, q_desc, o_desc, kc_desc, kd_desc, vc_desc, vd_desc, mask_c, mask_d,  #
               HD: tl.constexpr, GQA: tl.constexpr, MQ: tl.constexpr,  #
               SWEPT: tl.constexpr, KC_ROW: tl.constexpr, KC_COL: tl.constexpr,  #
               KD_ROW: tl.constexpr, KD_COL: tl.constexpr, SCALE: tl.constexpr,  #
               HAS_MASK: tl.constexpr):
    """One query head, all `MQ` rows. A descriptor an arm never builds arrives as its raw
    pointer and is never touched (the guards below are the same constexprs that built it)."""
    kvh = h // GQA
    q = q_desc.load([0, h * HD])  # [MQ, HD]
    q_scaled = q * SCALE  # :2175
    # THE NEW BLOCK, first (see the module doc).
    kd = kd_desc.load([KD_ROW, KD_COL + kvh * HD])
    scores_d = tl.dot(q_scaled, kd.T, out_dtype=tl.float16)
    if MQ > 1:
        scores_d = scores_d + mask_d.load([0, SWEPT])
    vd = vd_desc.load([KD_ROW, KD_COL + kvh * HD])
    if SWEPT > 0:
        # TWO KV BLOCKS: the scan (:2204-2270).
        running_max = tl.full([MQ], float("-inf"), tl.float16)
        denominator = tl.full([MQ], 0.0, tl.float16)
        output = tl.full([MQ, HD], 0.0, tl.float16)
        running_max, denominator, output = _scan_block(running_max, denominator, output,
                                                       scores_d, vd)
        kc = kc_desc.load([KC_ROW, KC_COL + kvh * HD])  # [SWEPT, HD]
        scores_c = tl.dot(q_scaled, kc.T, out_dtype=tl.float16)
        if HAS_MASK:
            scores_c = scores_c + mask_c.load([0, 0])
        vc = vc_desc.load([KC_ROW, KC_COL + kvh * HD])  # [SWEPT, HD]
        running_max, denominator, output = _scan_block(running_max, denominator, output,
                                                       scores_c, vc)
        output = output / denominator[:, None]
    else:
        # ONE KV BLOCK: the stable softmax (:2178-2200).
        block_max = tl.max(scores_d, 1)
        exp_scores = tl.exp((scores_d - block_max[:, None]).to(tl.float32)).to(tl.float16)
        denominator = tl.sum(exp_scores, 1).to(tl.float16)
        output = tl.dot(exp_scores, vd, out_dtype=tl.float16)
        output = output / denominator[:, None]
    o_desc.store([0, h * HD], output)


@triton.jit
def attn_fwd(desc_q, desc_o,  #
             desc_kc, desc_kd, desc_vc, desc_vd,  #
             desc_mask,  #
             NQH: tl.constexpr, NKVH: tl.constexpr, HD: tl.constexpr,  #
             GQA: tl.constexpr, MQ: tl.constexpr, CAP: tl.constexpr,  #
             SWEPT: tl.constexpr, NEW_LEN: tl.constexpr,  #
             KC_ROW: tl.constexpr, KC_COL: tl.constexpr,  #
             KD_ROW: tl.constexpr, KD_COL: tl.constexpr,  #
             SCALE: tl.constexpr, HAS_MASK: tl.constexpr,  #
             ONE_PASS: tl.constexpr):
    # `ONE_PASS` is the splice's statement of the no-prefix chunk; the kernel's own arm
    # selector is `SWEPT`, and the two must name the same shape.
    if ONE_PASS:
        tl.static_assert(SWEPT == 0)
        tl.static_assert(MQ > 1)
    q_width: tl.constexpr = NQH * HD
    kv_width: tl.constexpr = NKVH * HD
    q_desc = tl.make_tensor_descriptor(desc_q, shape=[MQ, q_width], strides=[q_width, 1],
                                       block_shape=[MQ, HD])
    o_desc = tl.make_tensor_descriptor(desc_o, shape=[MQ, q_width], strides=[q_width, 1],
                                       block_shape=[MQ, HD])
    # The new block: the whole chunk's rows when `MQ > 1`, the step's own row(s) at decode.
    if MQ > 1:
        kd_desc = tl.make_tensor_descriptor(desc_kd, shape=[NEW_LEN, kv_width],
                                            strides=[kv_width, 1], block_shape=[MQ, HD])
        vd_desc = tl.make_tensor_descriptor(desc_vd, shape=[NEW_LEN, kv_width],
                                            strides=[kv_width, 1], block_shape=[MQ, HD])
    else:
        kd_desc = tl.make_tensor_descriptor(desc_kd, shape=[NEW_LEN, kv_width],
                                            strides=[kv_width, 1], block_shape=[NEW_LEN, HD])
        vd_desc = tl.make_tensor_descriptor(desc_vd, shape=[NEW_LEN, kv_width],
                                            strides=[kv_width, 1], block_shape=[NEW_LEN, HD])
    # The resident prefix: built only when swept (`SWEPT == 0` leaves the cache parameters
    # addressed nowhere, the dead prefix the splice states with injected views).
    kc_desc = desc_kc
    vc_desc = desc_vc
    if SWEPT > 0:
        kc_desc = tl.make_tensor_descriptor(desc_kc, shape=[CAP, kv_width],
                                            strides=[kv_width, 1], block_shape=[SWEPT, HD])
        vc_desc = tl.make_tensor_descriptor(desc_vc, shape=[CAP, kv_width],
                                            strides=[kv_width, 1], block_shape=[SWEPT, HD])
    # The additive mask, prefix columns first; built only where a block reads it, so a
    # maskless shape leaves `desc_mask` addressed nowhere.
    mask_c = desc_mask
    mask_d = desc_mask
    if MQ > 1:
        # Every multi-row shape masks its new block (the causal triangle, or a decode batch's
        # diagonal); an unmasked one would let a row read the keys after it.
        tl.static_assert(HAS_MASK)
        mask_w: tl.constexpr = SWEPT + MQ
        mask_d = tl.make_tensor_descriptor(desc_mask, shape=[MQ, mask_w], strides=[mask_w, 1],
                                           block_shape=[MQ, MQ])
        if HAS_MASK:
            if SWEPT > 0:
                mask_c = tl.make_tensor_descriptor(desc_mask, shape=[MQ, mask_w],
                                                   strides=[mask_w, 1],
                                                   block_shape=[MQ, SWEPT])
    else:
        if HAS_MASK:
            mask_c = tl.make_tensor_descriptor(desc_mask, shape=[1, SWEPT], strides=[SWEPT, 1],
                                               block_shape=[1, SWEPT])
    if MQ > 1:
        # One head per program: the grid is `[NQH]`.
        _attn_head(tl.program_id(0), q_desc, o_desc, kc_desc, kd_desc, vc_desc, vd_desc,
                   mask_c, mask_d, HD, GQA, MQ, SWEPT, KC_ROW, KC_COL, KD_ROW, KD_COL,
                   SCALE, HAS_MASK)
    else:
        for h in tl.static_range(0, NQH, 1):
            _attn_head(h, q_desc, o_desc, kc_desc, kd_desc, vc_desc, vd_desc,
                       mask_c, mask_d, HD, GQA, MQ, SWEPT, KC_ROW, KC_COL, KD_ROW, KD_COL,
                       SCALE, HAS_MASK)
