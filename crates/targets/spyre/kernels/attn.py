# SPDX-License-Identifier: Apache-2.0
"""Scratchy's Triton attention kernel — the flash-attention decode/prefill program
over the paged KV cache.

BODY PROVENANCE: `crates/triton/test-fixtures/attention_flash.py`'s numerics (the
tutorial-derived flash form this tree card-validated), restated in the SPLICE's own
launch contract — which is NOT the fixture's delta-8 layout. The fixture's
transposed/splat-source staging belongs to the old ladder path; the splice presents
the tensors the WAY THE BUILDER'S OWN PROGRAM VIEWS THEM (main's `KtirFunc::attn`), because
the door (`ktir_superdsc_door::attn` → `attn_operands`) reads every fact — the q/out
`[mq, nqh·hd]` views, the kv `[cap, nkvh·hd]` views, q's first tile's `hd` width, the
cache's swept row extent — off the program itself.

THE SPLICE'S OWN CONTRACT (what `scratchy-triton-splice` states about this kernel):

* PARAMETERS, IN ORDER: `desc_q`, `desc_o`, then the four segment buffers in the
  door's positional order — resident K (`desc_kc`), new K (`desc_kd`), resident V
  (`desc_vc`), new V (`desc_vd`) — then `desc_mask`. The door reads `r[0]`/`r[1]` as
  q/out and the next four as (kc, new_k, vc, new_v), skipping the mask by its bound
  tid (`KtirNode.mask`); the builder's own `arg_for` order is the same.
* VIEWS: q and out are `[MQ, NQH*HD]`; the cache params are `[CAP, NKVH*HD]`; the new
  params are `[NEW_LEN, NKVH*HD]`; the mask is `[1, SWEPT]` (decode) or `[MQ, MQ]`
  (the causal one-pass). Every extent is a constexpr — one kernel per
  shape, the fixture's delta-7 law.
* GRID: `[NQH]` at `MQ > 1`, `[1]` at decode — the builder's own head split: one
  head per program instance (`KtdpGetComputeTileId`, grid `(nq, 1)`), the same
  form every NQH-tile kernel in this tree follows. The head is `tl.program_id(0)`
  in both mq > 1 arms; the trace unrolls ONE head's body (the row loop stays
  trace-time because the per-row causal extent `qi + 1` is a constexpr only
  inside it, rope.py's law). Decode keeps grid `[1]` and the trace-time head
  loop — the builder loops all heads at mq == 1 too.
* CONSTEXPRS: `NQH, NKVH, HD, GQA, MQ, CAP, SWEPT, SCALE` plus the segment corners
  (`KC_ROW/KC_COL/KD_ROW/KD_COL`), `NEW_LEN`, `HAS_MASK`, `ONE_PASS` — all stated
  by the splice from the node's own payload and regions, never inferred.

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
the `if SWEPT > 0` guards below leave `desc_kc`/`desc_vc` addressed NOWHERE, and no
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
def _attn_one_pass_head(h, desc_q, desc_o, desc_kd, desc_vd, desc_mask,  #
                        NQH: tl.constexpr, NKVH: tl.constexpr, HD: tl.constexpr,  #
                        GQA: tl.constexpr, MQ: tl.constexpr,  #
                        NEW_LEN: tl.constexpr, KD_ROW: tl.constexpr,  #
                        KD_COL: tl.constexpr, SCALE: tl.constexpr):
    """One head's causal one-pass body — shared by the decode head loop and the
    mq > 1 per-program head (the builder's `head_pid` form). Descriptors are
    constructed HERE because guard-conditional names cannot be passed as arguments
    (the actual argument evaluates at trace time), and the `[MQ, HD]`-block
    descriptors are this arm's own (rope.py's second-descriptor law)."""
    q_width = NQH * HD
    kv_width = NKVH * HD
    o_wide = tl.make_tensor_descriptor(desc_o, shape=[MQ, q_width],
                                       strides=[q_width, 1],
                                       block_shape=[MQ, HD])
    q_wide = tl.make_tensor_descriptor(desc_q, shape=[MQ, q_width],
                                       strides=[q_width, 1],
                                       block_shape=[MQ, HD])
    kd_desc = tl.make_tensor_descriptor(desc_kd, shape=[NEW_LEN, kv_width],
                                        strides=[kv_width, 1],
                                        block_shape=[MQ, HD])
    vd_desc = tl.make_tensor_descriptor(desc_vd, shape=[NEW_LEN, kv_width],
                                        strides=[kv_width, 1],
                                        block_shape=[MQ, HD])
    mask_desc = tl.make_tensor_descriptor(desc_mask, shape=[MQ, MQ],
                                          strides=[MQ, 1],
                                          block_shape=[MQ, MQ])
    kvh = h // GQA
    q = q_wide.load([0, h * HD])  # [MQ, HD]
    k = kd_desc.load([KD_ROW, KD_COL + kvh * HD])  # [MQ, HD]
    kt = k.T  # [HD, MQ]
    qk = tl.dot(q, kt, out_dtype=tl.float16)  # [MQ, MQ]
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
def _attn_cont_row(qi, h, desc_kc, desc_vc, desc_kd, desc_vd, desc_mask,  #
                   q_desc, o_desc,  #
                   NQH: tl.constexpr, NKVH: tl.constexpr, HD: tl.constexpr,  #
                   GQA: tl.constexpr, MQ: tl.constexpr, SWEPT: tl.constexpr,  #
                   CAP: tl.constexpr, NEW_LEN: tl.constexpr,  #
                   HAS_MASK: tl.constexpr, KC_ROW: tl.constexpr,  #
                   KC_COL: tl.constexpr, KD_ROW: tl.constexpr,  #
                   KD_COL: tl.constexpr, SCALE: tl.constexpr):
    """One row of one head's continuation arm — the builder's per-(row, head) body,
    called with `h = program_id(0)` at mq > 1 (one head per program) and from the
    static head loop at decode. The per-row descriptors are constructed HERE, under
    the same constexpr guards the inline form had, because a descriptor whose block
    shape depends on the trace-time row `qi` is constructible only where `qi` is in
    scope — and passing an undefined guard-conditional name as an argument would
    evaluate it at trace time, so the raw pointers are the parameters instead."""
    kv_width = NKVH * HD
    if SWEPT > 0:
        kc_desc = tl.make_tensor_descriptor(desc_kc, shape=[CAP, kv_width],
                                            strides=[kv_width, 1],
                                            block_shape=[SWEPT, HD])
        vc_desc = tl.make_tensor_descriptor(desc_vc, shape=[CAP, kv_width],
                                            strides=[kv_width, 1],
                                            block_shape=[SWEPT, HD])
    if HAS_MASK:
        mask_desc = tl.make_tensor_descriptor(desc_mask, shape=[1, SWEPT],
                                              strides=[SWEPT, 1],
                                              block_shape=[1, SWEPT])
    slen: tl.constexpr = qi + 1
    kd_r = tl.make_tensor_descriptor(desc_kd, shape=[NEW_LEN, kv_width],
                                     strides=[kv_width, 1],
                                     block_shape=[slen, HD])
    vd_r = tl.make_tensor_descriptor(desc_vd, shape=[NEW_LEN, kv_width],
                                     strides=[kv_width, 1],
                                     block_shape=[slen, HD])
    kvh = h // GQA
    q = q_desc.load([qi, h * HD])  # [1, HD]
    # Pass 1.
    if SWEPT > 0:
        kc = kc_desc.load([KC_ROW, KC_COL + kvh * HD])  # [SWEPT, HD]
        qk_c = tl.dot(q, kc.T, out_dtype=tl.float16)  # [1, SWEPT]
        qk_c = qk_c * SCALE
        if HAS_MASK:
            qk_c = qk_c + mask_desc.load([0, 0])
        m_c = tl.max(qk_c, 1)  # [1]
    kd = kd_r.load([KD_ROW, KD_COL + kvh * HD])  # [slen, HD]
    qk_d = tl.dot(q, kd.T, out_dtype=tl.float16)  # [1, slen]
    qk_d = qk_d * SCALE
    m_d = tl.max(qk_d, 1)  # [1] over the causal rows only
    if SWEPT > 0:
        gmax = tl.maximum(m_c, m_d)
    else:
        gmax = m_d
    # Pass 2.
    if SWEPT > 0:
        e_c = tl.exp((qk_c - gmax[:, None]).to(tl.float32)).to(tl.float16)
        s_c = tl.sum(e_c, 1)
    e_d = tl.exp((qk_d - gmax[:, None]).to(tl.float32)).to(tl.float16)
    s_d = tl.sum(e_d, 1)
    if SWEPT > 0:
        gsum = s_c + s_d
    else:
        gsum = s_d
    # Pass 3.
    if SWEPT > 0:
        w_c = e_c / gsum[:, None]
        vc = vc_desc.load([KC_ROW, KC_COL + kvh * HD])  # [SWEPT, HD]
        o = tl.dot(w_c, vc, out_dtype=tl.float16)  # [1, HD]
    w_d = e_d / gsum[:, None]
    vd = vd_r.load([KD_ROW, KD_COL + kvh * HD])  # [slen, HD]
    o_d = tl.dot(w_d, vd, out_dtype=tl.float16)  # [1, HD]
    if SWEPT > 0:
        o_desc.store([qi, h * HD], o + o_d)
    else:
        o_desc.store([qi, h * HD], o_d)


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
    # THE SWEPT RESIDENT CACHE — `SWEPT` rows at the segment's own corner. The door
    # reads the rung back off exactly this tile (`param_read_rows`), so the splice
    # states the RESOLVED active cap here, never the full capacity. Guarded: at
    # `SWEPT == 0` the views are the splice's to inject (see the module doc).
    if SWEPT > 0:
        kc_desc = tl.make_tensor_descriptor(desc_kc, shape=[CAP, kv_width],
                                            strides=[kv_width, 1],
                                            block_shape=[SWEPT, HD])
        vc_desc = tl.make_tensor_descriptor(desc_vc, shape=[CAP, kv_width],
                                            strides=[kv_width, 1],
                                            block_shape=[SWEPT, HD])
    # The new-token block's [NEW_LEN, HD] descriptors are the DECODE arm's (its
    # [1, HD] loads ride the whole-region blocks); the one-pass and continuation
    # helpers construct their own per-arm blocks from the same raw pointers.
    if not ONE_PASS:
        if MQ == 1:
            kd_desc = tl.make_tensor_descriptor(desc_kd, shape=[NEW_LEN, kv_width],
                                                strides=[kv_width, 1],
                                                block_shape=[NEW_LEN, HD])
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
        if not ONE_PASS:
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
    # ⭐ ONE HEAD PER PROGRAM at mq > 1 — the builder's own structure. `KtirFunc::attn`
    # runs `KtdpGetComputeTileId` with grid `(nq, 1)` when mq > 1 (one head per core,
    # `nh = 1` in its head loop), and loops all heads inside the program only at decode.
    # The splice states the same shape: the program id IS the head, and the head loops
    # below stay trace-time only where the builder keeps them (MQ == 1). This is the
    # compile-time law, not a tuning choice: a static_range head loop unrolls NQH bodies
    # into ONE grid-[1] program (granite: 32 heads = 32x the ops of the builder's form,
    # and the ladder's passes each walk every op), while `program_id(0)` emits one body
    # and lets the grid N-fold it.
    pid = tl.program_id(0)
    # ONE-PASS (the builder's own arm): the program id is the head directly. ONE_PASS
    # implies MQ > 1 (the splice's condition), so this is always the per-program head.
    if ONE_PASS:
        _attn_one_pass_head(pid, desc_q, desc_o, desc_kd, desc_vd, desc_mask,
                            NQH, NKVH, HD, GQA, MQ, NEW_LEN, KD_ROW,
                            KD_COL, SCALE)
    elif MQ == 1:
        for h in tl.static_range(0, NQH, 1):
            kvh = h // GQA
            q = q_desc.load([0, h * HD])  # [1, HD]
            # Pass 1: per-segment scores and the global max.
            if SWEPT > 0:
                kc = kc_desc.load([KC_ROW, KC_COL + kvh * HD])  # [SWEPT, HD]
                qk_c = tl.dot(q, kc.T, out_dtype=tl.float16)  # [1, SWEPT]
                qk_c = qk_c * SCALE
                if HAS_MASK:
                    qk_c = qk_c + mask_desc.load([0, 0])
                m_c = tl.max(qk_c, 1)  # [1]
            kd = kd_desc.load([KD_ROW, KD_COL + kvh * HD])  # [1, HD]
            qk_d = tl.dot(q, kd.T, out_dtype=tl.float16)  # [1, 1]
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
        # ⭐ CASE 3 — PREFILL CONTINUATION: per row, per head, the 3-pass loop over
        # both segments, the new segment's causal extent `qi + 1`. Per-row causal
        # tiles are CONSTRUCTIBLE at trace time — `tl.static_range` makes `qi` a
        # compile-time int — so the row's own `[qi+1, HD]` descriptors spell the
        # causal extent exactly as the builder does (`slen = qi + 1`), never a
        # whole-row max.
        #
        # ⛔ THE PREFIX SEGMENT IS CONDITIONAL — the `if SWEPT > 0` guards exist for
        # the `SWEPT == 0` shape (a first chunk baked `ActiveCap::NONE`): the swept
        # descriptors stay addressed-nowhere and the row's loop over the one new
        # segment is the causal extent in full. The builder's row loop is the same
        # shape: its `live` segment list skips the dead prefix, and causality comes
        # from `qi + 1` either way.
        #
        # ⭐ THE HEAD AXIS IS THE PROGRAM ID — the builder's `head_pid` form: one
        # head per program instance at mq > 1, so the op count is ONE head's body
        # times the row count, not NQH heads' (granite: 32x fewer ops per program,
        # which is the compile-time law this kernel follows). Decode keeps the
        # static head loop — the builder loops all heads at mq == 1 too.
        for qi in tl.static_range(0, MQ, 1):
            _attn_cont_row(qi, pid, desc_kc, desc_vc, desc_kd, desc_vd, desc_mask,
                           q_desc, o_desc, NQH, NKVH, HD, GQA, MQ, SWEPT, CAP,
                           NEW_LEN, HAS_MASK, KC_ROW, KC_COL, KD_ROW, KD_COL, SCALE)
