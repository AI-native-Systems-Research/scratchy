# SPDX-License-Identifier: Apache-2.0
"""Scratchy's Triton attention kernel — the flash-attention decode/prefill program
over the paged KV cache.

BODY PROVENANCE: `crates/triton/test-fixtures/attention_flash.py`'s numerics (the
tutorial-derived flash form this tree card-validated), restated in the SPLICE'S own
launch contract — which is NOT the fixture's delta-8 layout. The fixture's
transposed/splat-source staging belongs to the old ladder path; the splice presents
the tensors the WAY THE BUILDER'S OWN PROGRAM VIEWS THEM (main's `KtirFunc::attn`), because
the door reads every fact — the q/out `[mq, nqh·hd]` views, the kv `[cap, nkvh·hd]`
views, q's first tile's `hd` width, the cache's swept row extent — off the program itself.

THE SPLICE'S OWN CONTRACT (what `scratchy-triton-splice` states about this kernel):

* PARAMETERS, IN ORDER: `desc_q`, `desc_o`, then the four segment buffers in the
  door's positional order — the Kᵀ PLANE (`desc_kt`, the pool's third plane, in
  the resident-K slot), the new block's Kᵀ scratch (`desc_ktn`, in the new-K
  slot), resident V (`desc_vc`), new V (`desc_vd`) — then `desc_mask`. The door
  reads `r[0]`/`r[1]` as q/out and the next four as the segment pairs, skipping
  the mask by its bound tid (`KtirNode.mask`).
* ⛔ K IS READ ONLY IN ITS TRANSPOSED FORM — the door's own law
  (`transposed_b_is_placeable`): a `.T` over a buffer written n-major has no
  agent that spends the transposition, so the score dot states the PLAIN `[k, n]`
  form with the contraction axis as B's row axis. The resident prefix's Kᵀ comes
  from the PLANE (written transposed by its own writer — the rope append's second
  store on card, the host's transposed staging on the emulator and the gate); the
  new block's comes from `desc_ktn`, the `restickify.py` program's output, which
  the splice emits immediately before this one.
* VIEWS: q and out are `[MQ, NQH*HD]`; `desc_kt` is `[HD, NKVH*CAP]` — the Kᵀ
  plane's TRUE arrangement (per head `[hd, PAGE_SLOTS]` slot-sticked, heads
  concatenated along the out axis: `kernel(HD, NKVH*CAP).dev_off(0, kvh*CAP + s)`
  IS `kt_write_off(kvh, s, 0)`, exact at `CAP == PAGE_SLOTS`); `desc_ktn` is
  `[KTN_W, KTN_ROWS]` — the restickify's transposed staging form, per head read
  at the ROW corner `kvh*HD` (the B operand's in-corner, which the door places
  without alignment needs and which is stick-aligned anyway); the V params are
  `[CAP, NKVH*HD]` / `[NEW_LEN, NKVH*HD]`; the mask is `[1, SWEPT]` (decode) or
  `[MQ, MQ]` (the causal one-pass). Every extent is a constexpr — one kernel per
  shape, the fixture's delta-7 law.
* GRID: `[1]` at every shape — the head loop is TRACE-TIME (`tl.static_range`),
  the decode arm's own form. The builder's grid-`(nq, 1)` one-head-per-program
  split is DEFUNCT here, and MEASURED so: `to_ktir`'s clone protocol
  (`unroll_grid_positions`) fires on every mq>1 attention — the V corner
  `kvh*HD` is computed and feeds a matmul — and re-materializes every grid
  position as a body copy anyway (the grid resets to `[1]`), so the split bought
  no op count; and its per-position corners walk the protocol's partition
  census, which re-partitioned the Kᵀ scratch's head-offset ROW corner into 28
  per-position POINTER ARGUMENTS (MEASURED at granite-2b mq=31: `attn_fwd`
  compiled to 35 parameters against the door's 7). The trace-time loop states
  the same 32 head bodies with CONSTANT corners — the form the decode arm ships
  and the door resolves window-by-window. The trace unrolls ONE head's body per
  trip because the per-row causal extent `qi + 1` is a constexpr only inside it,
  rope.py's law.
* CONSTEXPRS: `NQH, NKVH, HD, GQA, MQ, CAP, SWEPT, SCALE` plus the segment
  corners (`KC_ROW/KC_COL/KD_ROW/KD_COL`), `NEW_LEN`, the new block's scratch
  extents (`KTN_ROWS` the stick cover, `KTN_W` the device width), `HAS_MASK`,
  `ONE_PASS` — all stated by the splice from the node's own payload and regions,
  never inferred.

⛔ THE Kᵀ PLANE'S STICK LAW. The plane is sticked on SLOTS (64 consecutive slots
per stick), so every `desc_kt` load corner must be 64-aligned and every load
width a 64 multiple — `KC_ROW` is 0 at the masked prefix (the builder's own law)
and `SWEPT` the rung's swept extent (a ladder bound, itself a 64 multiple), so
both hold by construction; the head's own corner `kvh*CAP` is whole pages. The
V plane needs no such alignment (its stick axis is the head dim).

⛔ THE NEW BLOCK'S B WINDOW IS EXACT. `desc_ktn`'s block is `[HD, NEW_LEN]` at
`[kvh*HD, 0]` — the LOGICAL new-block extent, never the stick cover: the scratch's
pad columns (the transposed staging pad rows, written by the restickify's fold and
never zeroed) must not enter the score, and at `NEW_LEN` they cannot — the dot's
logical n is `NEW_LEN`, the score tile's own width is `NEW_LEN`, and every reduce
below bounds itself to it. The util-floor pad the matmul door applies to a
sub-stick n lands in the score tile's own placement pad (the `cols_entire` law —
the same exception the value leg's `[1,1]`-contraction dots ride).

⛔ THE THREE SHAPES, exactly the builder's own arms (main's `KtirFunc::attn`):

1. DECODE (`MQ == 1`): per head, the two live segments (resident prefix swept
   `SWEPT` rows + the one new row) scored, online-softmax-combined across them —
   the builder's 3-pass row loop at its single row.
2. PREFILL ONE-PASS (`ONE_PASS`): the first prompt chunk — no resident prefix
   (`SWEPT == 0`, the bundle baked `ActiveCap::NONE`) attends only its own `MQ`
   tokens causally: whole-chunk scores `[MQ, MQ]` plus the additive triangle mask,
   ONE row-wise softmax. The builder's own one-pass arm, at every `MQ > 1` — the
   front end's TMA 16-byte floor (a GPU law) is gone, so the ladder's bottom rung
   (MQ=7) takes this arm exactly as the builder's did.
3. PREFILL CONTINUATION (`MQ > 1`, `SWEPT > 0`): per row, per head, the 3-pass
   loop over both segments with the new segment's causal extent `qi + 1` — and,
   at `SWEPT == 0`, over the new segment alone, the
   prefix guards keeping the swept descriptors unaddressed exactly as the decode
   arm's do.

⛔ THE DEAD PREFIX IS A SPLICE-INJECTED VIEW, NOT A KERNEL LOAD. At `SWEPT == 0`
the `if SWEPT > 0` guards below leave `desc_kt`/`desc_vc` addressed NOWHERE, and no
Triton program can state "a parameter with a view and no access tile" — an
unconsumed descriptor is DCE'd and the handoff refuses the hole (the fixture's
delta-11 law). The builder keeps the dead segment's VIEW so the cache tensor stays
a program parameter (its identity is what the card's resident KV and the emulator's
host threading bind). The SPLICE therefore injects the two dead
`ktdp.construct_memory_view` ops into the compiled module post-hoc — a stated tape
fact, the same law as `lmlast`'s `node_out_tid`.

⛔ THE SCALE IS AN IMMEDIATE MULTIPLY ON THE DOT'S RESULT (`qk * SCALE`), the
builder's own spelling (`arith.mulf` of the matmul against the constant) — the
door's `program_score_scale` resolves the `[1,1]` registry const FROM that value,
and the fixture's delta-9 law (a compute operand must be a compile-time constant)
is why `SCALE` is a constexpr.
"""

import triton
import triton.language as tl


@triton.jit
def _attn_one_pass_head(h, desc_q, desc_o, desc_ktn, desc_vd, desc_mask,  #
                        NQH: tl.constexpr, NKVH: tl.constexpr, HD: tl.constexpr,  #
                        GQA: tl.constexpr, MQ: tl.constexpr,  #
                        NEW_LEN: tl.constexpr, KTN_ROWS: tl.constexpr,  #
                        KTN_W: tl.constexpr, KD_ROW: tl.constexpr,  #
                        KD_COL: tl.constexpr, SCALE: tl.constexpr):
    """One head's causal one-pass body — one trip of the trace-time head loop (and the
    form the decode arm's own loop takes; `h` is a trace-time constant, so every
    corner below is constant). Descriptors are constructed HERE because
    guard-conditional names cannot be passed as arguments (the actual argument
    evaluates at trace time), and the `[MQ, HD]`-block descriptors are this arm's
    own (rope.py's second-descriptor law)."""
    q_width = NQH * HD
    kv_width = NKVH * HD
    o_wide = tl.make_tensor_descriptor(desc_o, shape=[MQ, q_width],
                                       strides=[q_width, 1],
                                       block_shape=[MQ, HD])
    q_wide = tl.make_tensor_descriptor(desc_q, shape=[MQ, q_width],
                                       strides=[q_width, 1],
                                       block_shape=[MQ, HD])
    # The new block's Kᵀ — the restickify's transposed staging form, this head's
    # [HD, NEW_LEN] window at the ROW corner kvh*HD (the B operand's in-corner).
    # The block is the LOGICAL extent: the scratch's stick-cover pad columns
    # must not enter the score (see the module doc's EXACT-window law).
    ktn_desc = tl.make_tensor_descriptor(desc_ktn, shape=[KTN_W, KTN_ROWS],
                                         strides=[KTN_ROWS, 1],
                                         block_shape=[HD, NEW_LEN])
    vd_desc = tl.make_tensor_descriptor(desc_vd, shape=[NEW_LEN, kv_width],
                                        strides=[kv_width, 1],
                                        block_shape=[MQ, HD])
    mask_desc = tl.make_tensor_descriptor(desc_mask, shape=[MQ, MQ],
                                          strides=[MQ, 1],
                                          block_shape=[MQ, MQ])
    kvh = h // GQA
    q = q_wide.load([0, h * HD])  # [MQ, HD]
    ktn = ktn_desc.load([kvh * HD, 0])  # [HD, MQ]
    qk = tl.dot(q, ktn, out_dtype=tl.float16)  # [MQ, MQ]
    qk = qk * SCALE
    qk = qk + mask_desc.load([0, 0])
    # Row-wise softmax, one reduce per axis.
    mx = tl.max(qk, 1)  # [MQ]
    sh = qk - mx[:, None]
    e = tl.exp(sh.to(tl.float32)).to(tl.float16)
    su = tl.sum(e, 1)  # [MQ]
    v = vd_desc.load([KD_ROW, KD_COL + kvh * HD])  # [MQ, HD]
    o = tl.dot(e, v, out_dtype=tl.float16)  # [MQ, HD]
    o = o / su[:, None]
    o_wide.store([0, h * HD], o)


@triton.jit
def _attn_cont_head(h, desc_q, desc_o, desc_kt, desc_vc, desc_ktn, desc_vd,  #
                    desc_mask,  #
                    NQH: tl.constexpr, NKVH: tl.constexpr, HD: tl.constexpr,  #
                    GQA: tl.constexpr, MQ: tl.constexpr, SWEPT: tl.constexpr,  #
                    CAP: tl.constexpr, NEW_LEN: tl.constexpr,  #
                    KTN_ROWS: tl.constexpr, KTN_W: tl.constexpr,  #
                    KC_ROW: tl.constexpr, KC_COL: tl.constexpr,  #
                    KD_ROW: tl.constexpr, KD_COL: tl.constexpr,  #
                    SCALE: tl.constexpr):
    """One head's continuation body, WHOLE-CHUNK: the resident prefix and the chunk's own
    new block scored for all `MQ` rows at once, combined by one two-segment softmax —
    one trip of the trace-time head loop (`h` is a trace-time constant, so every
    corner below is constant).

    ⛔ IT WAS A PER-ROW UNROLL, AND THAT WAS THE BAKE'S WIDEST COMPILE. The row loop was
    `tl.static_range(0, MQ)` with each row's own `[qi + 1, HD]` causal tiles, so the
    program held `MQ` row bodies whose new-block tiles summed to `MQ(MQ+1)/2` rows — its
    size grew with `MQ²`. MEASURED, granite-2b: 94.7 s to lower at 256 rows, and at 512 a
    build-script thread in `to_ktir_emit` for over 25 min at 102 GB without finishing.

    The causality now rides the mask instead of the tile extents, exactly as the one-pass
    arm's does: the mask parameter is ONE `[MQ, SWEPT + MQ]` additive tile, the prefix's
    runtime length mask in its first `SWEPT` columns and the new block's row mask in the
    rest — the causal triangle for a prompt chunk, the diagonal for a decode batch, staged
    by the host either way (the device has no lane-wise integer compare to derive one,
    attention_flash delta 10). The program's size is now flat in `MQ`.

    Every fact the door reads off the program is the old arm's: q's first tile is still
    `HD` columns wide (`[MQ, HD]`, the one-pass arm's own q tile), the cache tiles are
    `[SWEPT, HD]` (the swept extent the door reads back), and the scale is the same
    immediate multiply on the score dot."""
    # A continuation always has a live prefix: a chunk with none is the one-pass arm.
    tl.static_assert(SWEPT > 0)
    q_width = NQH * HD
    kv_width = NKVH * HD
    q_wide = tl.make_tensor_descriptor(desc_q, shape=[MQ, q_width],
                                       strides=[q_width, 1],
                                       block_shape=[MQ, HD])
    o_wide = tl.make_tensor_descriptor(desc_o, shape=[MQ, q_width],
                                       strides=[q_width, 1],
                                       block_shape=[MQ, HD])
    # The resident prefix's Kᵀ — the PLANE's [HD, NKVH*CAP] fused view, this head's
    # [HD, SWEPT] window at the slot corner kvh*CAP + KC_ROW (0 at the masked prefix).
    kt_desc = tl.make_tensor_descriptor(desc_kt, shape=[HD, NKVH * CAP],
                                        strides=[NKVH * CAP, 1],
                                        block_shape=[HD, SWEPT])
    vc_desc = tl.make_tensor_descriptor(desc_vc, shape=[CAP, kv_width],
                                        strides=[kv_width, 1],
                                        block_shape=[SWEPT, HD])
    ktn_desc = tl.make_tensor_descriptor(desc_ktn, shape=[KTN_W, KTN_ROWS],
                                         strides=[KTN_ROWS, 1],
                                         block_shape=[HD, NEW_LEN])
    vd_desc = tl.make_tensor_descriptor(desc_vd, shape=[NEW_LEN, kv_width],
                                        strides=[kv_width, 1],
                                        block_shape=[MQ, HD])
    width: tl.constexpr = SWEPT + MQ
    mask_c = tl.make_tensor_descriptor(desc_mask, shape=[MQ, width],
                                       strides=[width, 1],
                                       block_shape=[MQ, SWEPT])
    mask_d = tl.make_tensor_descriptor(desc_mask, shape=[MQ, width],
                                       strides=[width, 1],
                                       block_shape=[MQ, MQ])
    kvh = h // GQA
    q = q_wide.load([0, h * HD])  # [MQ, HD]
    # Pass 1: both segments' scores for every row, and the per-row global max.
    kt = kt_desc.load([0, kvh * CAP + KC_ROW])  # [HD, SWEPT]
    qk_c = tl.dot(q, kt, out_dtype=tl.float16)  # [MQ, SWEPT]
    qk_c = qk_c * SCALE
    qk_c = qk_c + mask_c.load([0, 0])
    ktn = ktn_desc.load([kvh * HD, 0])  # [HD, MQ]
    qk_d = tl.dot(q, ktn, out_dtype=tl.float16)  # [MQ, MQ]
    qk_d = qk_d * SCALE
    qk_d = qk_d + mask_d.load([0, SWEPT])
    gmax = tl.maximum(tl.max(qk_c, 1), tl.max(qk_d, 1))  # [MQ]
    # Pass 2: exp(score - gmax) and the per-row global sum.
    e_c = tl.exp((qk_c - gmax[:, None]).to(tl.float32)).to(tl.float16)
    e_d = tl.exp((qk_d - gmax[:, None]).to(tl.float32)).to(tl.float16)
    gsum = tl.sum(e_c, 1) + tl.sum(e_d, 1)  # [MQ]
    # Pass 3: weighted V, accumulated across the two segments.
    vc = vc_desc.load([KC_ROW, KC_COL + kvh * HD])  # [SWEPT, HD]
    vd = vd_desc.load([KD_ROW, KD_COL + kvh * HD])  # [MQ, HD]
    o = tl.dot(e_c / gsum[:, None], vc, out_dtype=tl.float16)  # [MQ, HD]
    o_d = tl.dot(e_d / gsum[:, None], vd, out_dtype=tl.float16)  # [MQ, HD]
    o_wide.store([0, h * HD], o + o_d)


@triton.jit
def attn_fwd(desc_q, desc_o,  #
             desc_kt, desc_ktn, desc_vc, desc_vd,  #
             desc_mask,  #
             NQH: tl.constexpr, NKVH: tl.constexpr, HD: tl.constexpr,  #
             GQA: tl.constexpr, MQ: tl.constexpr, CAP: tl.constexpr,  #
             SWEPT: tl.constexpr, NEW_LEN: tl.constexpr,  #
             KTN_ROWS: tl.constexpr, KTN_W: tl.constexpr,  #
             KC_ROW: tl.constexpr, KC_COL: tl.constexpr,  #
             KD_ROW: tl.constexpr, KD_COL: tl.constexpr,  #
             SCALE: tl.constexpr, HAS_MASK: tl.constexpr,  #
             ONE_PASS: tl.constexpr):
    q_width = NQH * HD
    kv_width = NKVH * HD
    q_desc = tl.make_tensor_descriptor(desc_q, shape=[MQ, q_width],
                                       strides=[q_width, 1],
                                       block_shape=[1, HD])
    # TWO DESCRIPTORS OVER THE OUT POINTER — rope.py's law. The door reads q's
    # first access tile (`[1, HD]`) for the `hd` width, so the LOAD tile must be
    # the builder's per-head row; the one-pass computes a whole-chunk `[MQ, HD]`
    # output per head, which stores through its own `[MQ, HD]`-block descriptor
    # over the same pointer. `o_desc` below must stay `[1, HD]` to match the
    # decode/continuation arms' single-row stores.
    o_desc = tl.make_tensor_descriptor(desc_o, shape=[MQ, q_width],
                                       strides=[q_width, 1],
                                       block_shape=[1, HD])
    # THE SWEPT RESIDENT SEGMENT — the Kᵀ plane and the V plane, `SWEPT` rows at
    # the segment's own corners. The door reads the rung back off exactly these
    # tiles, so the splice states the RESOLVED active cap here, never the full
    # capacity. Guarded: at `SWEPT == 0` the views are the splice's to inject
    # (see the module doc).
    if SWEPT > 0:
        kt_desc = tl.make_tensor_descriptor(desc_kt, shape=[HD, NKVH * CAP],
                                            strides=[NKVH * CAP, 1],
                                            block_shape=[HD, SWEPT])
        vc_desc = tl.make_tensor_descriptor(desc_vc, shape=[CAP, kv_width],
                                            strides=[kv_width, 1],
                                            block_shape=[SWEPT, HD])
    # The new-token block's descriptors are the DECODE arm's (its [1, HD] loads
    # ride the whole-region blocks); the one-pass and continuation helpers
    # construct their own per-arm blocks from the same raw pointers.
    if not ONE_PASS:
        if MQ == 1:
            ktn_desc = tl.make_tensor_descriptor(desc_ktn, shape=[KTN_W, KTN_ROWS],
                                                 strides=[KTN_ROWS, 1],
                                                 block_shape=[HD, NEW_LEN])
            vd_desc = tl.make_tensor_descriptor(desc_vd, shape=[NEW_LEN, kv_width],
                                                strides=[kv_width, 1],
                                                block_shape=[NEW_LEN, HD])
    # The runtime length mask: `[1, SWEPT]` bounding the resident prefix to the rows
    # valid this step (decode). The one-pass `[MQ, MQ]` causal triangle descriptor is
    # constructed inside `_attn_one_pass_head` (its arm's own). Built only when a
    # segment consumes it — an unguarded descriptor no load reads leaves its parameter
    # unaddressed (delta 11). Nested constexpr ifs, not `and`: the parser refuses
    # Python `BoolOp` inside a @triton.jit kernel, and both selectors are constexpr so
    # each `if` resolves at trace time anyway.
    if HAS_MASK:
        if MQ == 1:
            mask_desc = tl.make_tensor_descriptor(desc_mask, shape=[1, SWEPT],
                                                  strides=[SWEPT, 1],
                                                  block_shape=[1, SWEPT])

    # ⭐ CASE 2 — THE CAUSAL ONE-PASS (the builder's own arm): one live segment, the
    # whole chunk in one pass, per head. The whole-chunk `[MQ, HD]` output stores
    # through a `[MQ, HD]`-block descriptor over the same out pointer — rope.py's
    # second-descriptor law (the base `o_desc` stays `[1, HD]`-blocked for the other
    # arms' single-row stores). The `[MQ, MQ]` mask descriptor above has NO
    # minimum width: the front end's TMA 16-byte floor was a GPU law this target
    # does not have, and it is gone — the ladder's bottom rung (MQ=7) takes this
    # arm exactly as the builder's did.
    # ⭐ THE HEAD LOOP IS TRACE-TIME at every MQ, the decode arm's own form. The
    # builder's grid-`(nq, 1)` one-head-per-program split is defunct here, MEASURED:
    # `to_ktir`'s clone protocol fires on every mq>1 attention (the V corner is
    # computed and feeds a matmul) and re-materializes every grid position as a body
    # copy anyway, so the split bought no op count — while its per-position corners
    # walk the protocol's partition census, which re-partitioned the Kᵀ scratch's
    # head-offset ROW corner into per-position POINTER ARGUMENTS (granite-2b mq=31:
    # 35 parameters against the door's 7). The trace-time loop states the same 32
    # head bodies with CONSTANT corners, `h` a trace-time constant per trip.
    if ONE_PASS:
        for h in tl.static_range(0, NQH, 1):
            _attn_one_pass_head(h, desc_q, desc_o, desc_ktn, desc_vd, desc_mask,
                                NQH, NKVH, HD, GQA, MQ, NEW_LEN, KTN_ROWS,
                                KTN_W, KD_ROW, KD_COL, SCALE)
    elif MQ == 1:
        for h in tl.static_range(0, NQH, 1):
            kvh = h // GQA
            q = q_desc.load([0, h * HD])  # [1, HD]
            # Pass 1: per-segment scores and the global max.
            if SWEPT > 0:
                kt = kt_desc.load([0, kvh * CAP + KC_ROW])  # [HD, SWEPT]
                qk_c = tl.dot(q, kt, out_dtype=tl.float16)  # [1, SWEPT]
                qk_c = qk_c * SCALE
                if HAS_MASK:
                    qk_c = qk_c + mask_desc.load([0, 0])
                m_c = tl.max(qk_c, 1)  # [1]
            ktn = ktn_desc.load([kvh * HD, 0])  # [HD, NEW_LEN]
            qk_d = tl.dot(q, ktn, out_dtype=tl.float16)  # [1, NEW_LEN]
            qk_d = qk_d * SCALE
            m_d = tl.max(qk_d, 1)  # [1]
            if SWEPT > 0:
                gmax = tl.maximum(m_c, m_d)
            else:
                gmax = m_d
            # Pass 2: exp(score - gmax) and the global sum.
            if SWEPT > 0:
                e_c = tl.exp((qk_c - gmax[:, None]).to(tl.float32)).to(tl.float16)
                s_c = tl.sum(e_c, 1)  # [1]
            e_d = tl.exp((qk_d - gmax[:, None]).to(tl.float32)).to(tl.float16)
            s_d = tl.sum(e_d, 1)  # [1]
            if SWEPT > 0:
                gsum = s_c + s_d
            else:
                gsum = s_d
            # Pass 3: weighted V, accumulated across segments.
            if SWEPT > 0:
                w_c = e_c / gsum[:, None]
                vc = vc_desc.load([KC_ROW, KC_COL + kvh * HD])  # [SWEPT, HD]
                o = tl.dot(w_c, vc, out_dtype=tl.float16)  # [1, HD]
            w_d = e_d / gsum[:, None]
            vd = vd_desc.load([KD_ROW, KD_COL + kvh * HD])  # [1, HD]
            o_d = tl.dot(w_d, vd, out_dtype=tl.float16)  # [1, HD]
            if SWEPT > 0:
                o_desc.store([0, h * HD], o + o_d)
            else:
                o_desc.store([0, h * HD], o_d)
    else:
        # ⭐ CASE 3 — PREFILL CONTINUATION (and a decode batch): both segments, every
        # row at once, one two-segment softmax — see `_attn_cont_head`. The splice routes
        # here only with a live prefix and its mask bound (`SWEPT > 0`, `HAS_MASK`); any
        # other `MQ > 1` shape is the one-pass arm or a refusal.
        #
        # ⭐ THE HEAD LOOP IS TRACE-TIME here too — one trip per head, `h` a trace-time
        # constant, the decode arm's own form (see the one-pass arm's comment for the
        # measured law that retired the grid-`(nq, 1)` per-program head).
        for h in tl.static_range(0, NQH, 1):
            _attn_cont_head(h, desc_q, desc_o, desc_kt, desc_vc, desc_ktn, desc_vd,
                            desc_mask, NQH, NKVH, HD, GQA, MQ, SWEPT, CAP, NEW_LEN,
                            KTN_ROWS, KTN_W, KC_ROW, KC_COL, KD_ROW, KD_COL, SCALE)
