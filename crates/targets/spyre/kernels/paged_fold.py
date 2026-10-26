# SPDX-License-Identifier: Apache-2.0
"""The Spyre-native paged fold: the per-page attention body the runtime
re-launches once per resident page.

`batched_prefill_hm` walks every page in ONE launch (`tl.static_range` over
`PAGES`), so its context is a BAKED extent — the #438 cliff: past the widest
chunk nothing serves. This file is the other shape: ONE page per launch, the
online-softmax carry (max, sum, accumulator) in DEVICE MEMORY instead of
registers, so context is a RUNTIME page count. Three kernels, one contract:

  * `paged_fold_seed_hm` — writes the carry's seed values (max = the finite
    -65504 sentinel, sum = 0, accumulator = 0). Runs ONCE per step, BEFORE
    the first pass. A seeding PASS would reset the running state
    mid-accumulation, so the seed is a separate program, not a flag.
  * `paged_fold_hm` — ONE page of the walk: loads the carry, gathers this
    launch's page through the pool's index unit, folds the page into the
    carry, stores the carry. NO finalize, NO output: the launch is pure
    accumulation, and repeating it IS the loop.
  * `paged_fold_finalize_hm` — `out = acc / l`, ONCE, after the last pass.

The runtime contract this family states (and `fold_plan` enforces — the
launch-side arithmetic is single-sourced there):

  * A GATHERED FOLD TAKES NO KV SHIFT. The index entries are GLOBAL
    stick-block numbers (`addr = idx * skip_addr + base`), so the pool's
    segment base never moves; the page a pass reads is named by the entries
    the host staged for it.
  * THE INDEX TABLES AND THE MASK RIDE THE ACTIVATION SEGMENT, PAGE-MAJOR.
    Each pass shifts that segment's base by one page's worth of bytes —
    512 at this geometry (`W4 * KTB * 4` for a table, `PAGE_SLOTS * 2` for
    the mask row; the same number, which is why one delta moves all three
    tensors correctly). The descriptors below state the WHOLE region and
    load at THIS page's window, so the shifted base lands inside the
    region for every pass.
  * THE CARRY RIDES THE INTERMEDIATE SEGMENT, UNSHIFTED. A request's pages
    all fold into the SAME rows — that is what makes the online softmax a
    running state — so nothing about the carry may move per pass.
  * THE CARRY RIDES THE ACCUMULATOR'S OWN FRAME — `[BATCH*MQ, 1]`, ONE
    VALUE PER STICK AT LANE 0 (`strides [64, 1]`, the pad lanes inert).
    The walk's reduce family writes that frame and no other: a max or sum
    over the score tile's last axis lands one value per stick at lane 0
    (the MEASURED device law, wall 11), and the door's re-row law refuses
    every bridge to the packed `[BATCH, MQ]` frame — a reshape verdict
    runs off the SOURCE'S OWN TYPE, so `[B, MQ]` at an `MQ_PAD` row stride
    is a DIFFERENT BYTE LAYOUT, not a respelling. MEASURED, wall 19,
    build 10: the fold rung's `Maximum` refused the carried `[32, 256]` m
    against its `[8192, 1]` output — correctly, packed element (i, j) at
    byte `(i*MQ_PAD + j)*2` is not stick-per-value at byte `j*128` — and
    the rung was dropped. So the carry STATES the accumulator frame, and
    the body SPELLS its softmax chain there: the masked score tile lent
    once to `[B*MQ, 1, 64]` (the admitted flat identity; the mask add
    itself stays at `[B, MQ, 64]`, where the `[1, MQ, 64]` mask broadcast
    is the row-split mode the door admits), every reduce over the last
    axis landing `[B*MQ, 1]` with no lens, `p` lent back to
    `[B, MQ, 64]` for the score dot, and the FMA run at the batched
    spelling with the rescale lent to `[B, MQ, 1]` — the broadcast's
    own proof: a rank-2 Col spray from a COMPUTED source states no
    vector fact (the door admits an `expand_shape`'s degenerate dim or
    a one-column LOAD, and `rescale` is exp2's result), while the
    rank-3 flat form's count proof — one source element per result
    row — is exactly the fact the accumulator frame states (MEASURED,
    wall 20, build 11: the rank-2 spelling was refused at that proof).
    `acc` lands FLAT `[B*MQ, HD]` at the store (the admitted flat
    identity). At MQ == 1 the frame is byte-identical to the b833dd2f2e
    `[BATCH, MQ]` law (`MQ_PAD` was 64 then, so `[32, 1]` at
    `strides [64, 1]` either way — the card-proven decode family is
    untouched); this law lifts that guarantee to every MQ by MOVING the
    carry to the frame the device writes, never by bridging frames.

The per-trip laws are `batched_prefill_hm`'s own (one full id stick per
page-window trip, the V block contiguous only at HD 64, the exp2 softmax),
restated at ONE page per launch. ⭐ THE TABLES STATE THE MASK'S OWN PITCH:
the per-pass shift is ONE Activation delta, and a mask page block is
`W4 * MQ` rows (`512 * MQ` bytes at this geometry) while a table's
meaningful page block is one stick per window (`512` bytes) — so the
tables are stated PADDED at the mask's pitch, ONE ID STICK PER MASK ROW
(`[REGION_PAGES * W4 * MQ * KTB]` entries, the first `W4 * KTB` of each
page's block meaningful, the tail unread and inert). At MQ == 1 the padded
form is byte-identical to the natural one (`W4 * KTB == W4 * MQ * KTB`),
which is why the two pitches used to agree "only at one query row" — the
padding law is what lifts that to every MQ, and `KTB == 32` (`128 B` of
entries per `128 B` of mask row) is what makes the two pitches EXACTLY
equal, asserted below beside the id-stick law that forces it.

⛔ THE CARRY'S MEMORY IMAGE IS f16 — the SAME grading the one-shot kernel's
LEGALIZED program carries, so the fold is numerically IDENTICAL to the
body it replaces, page for page. `batched_prefill_hm` SPELLS its max/sum
at f32 in Triton, but `LegalizeTypes` collapses that island to f16 before
the door ever sees it (the widening casts around the f16 score tile are
removed and the chain retypes), so an f16 carry here is parity, not a
degradation. An f32-GRADED carry in device memory is a different campaign
— it needs a transfer-boundary anchor in `LegalizeTypes` (a genuine-f32
value persisted between f32 descriptor views must not collapse) plus the
door's fp32 allowlist extended past the merge chain — and is NOT this
family's ask. The -65504 sentinel is f16-exact (the format's own max),
so the seed loses nothing by the narrowing.
"""


@triton.jit
def paged_fold_seed_hm(desc_m, desc_l, desc_acc,  #
                       MQ: tl.constexpr, NKVH: tl.constexpr,  #
                       GQA: tl.constexpr, HD: tl.constexpr):
    """Seed the fold's carry: max at the finite sentinel, sum and
    accumulator at zero. Once per step, before the first pass. The carry's
    memory image is f16 (the module doc's law) and the sentinel is f16-exact."""
    BATCH: tl.constexpr = NKVH * GQA
    # The finite sentinel, not -inf: a masked-everywhere row still
    # normalizes with weight 0, never NaN (torch's own clamp trick).
    # [BATCH*MQ, 1] at strides [64, 1] — the module doc's frame law: the
    # accumulator's own frame, one value per stick at lane 0, the shape
    # the body's reduce family writes, so no load or store ever re-rows.
    m_desc = tl.make_tensor_descriptor(
        desc_m, shape=[BATCH * MQ, 1], strides=[64, 1],
        block_shape=[BATCH * MQ, 1])
    l_desc = tl.make_tensor_descriptor(
        desc_l, shape=[BATCH * MQ, 1], strides=[64, 1],
        block_shape=[BATCH * MQ, 1])
    acc_desc = tl.make_tensor_descriptor(
        desc_acc, shape=[BATCH * MQ, HD], strides=[HD, 1],
        block_shape=[BATCH * MQ, HD])
    m_desc.store([0, 0], tl.full([BATCH * MQ, 1], -65504.0, tl.float16))
    l_desc.store([0, 0], tl.zeros([BATCH * MQ, 1], tl.float16))
    acc_desc.store([0, 0], tl.zeros([BATCH * MQ, HD], tl.float16))


@triton.jit
def paged_fold_hm(desc_q, desc_m, desc_l, desc_acc,  #
                  desc_kt, desc_v, desc_ids_kt, desc_ids_v, desc_mask,  #
                  MQ: tl.constexpr, NKVH: tl.constexpr,  #
                  GQA: tl.constexpr, HD: tl.constexpr,  #
                  PAGE_SLOTS: tl.constexpr, REGION_PAGES: tl.constexpr,  #
                  SCALE: tl.constexpr):
    """ONE page of the walk, folded into the carried online-softmax state.
    This launch's page is the one the host's per-pass segment shift lands
    on; the loads below read that page's own window."""
    BATCH: tl.constexpr = NKVH * GQA
    W4: tl.constexpr = PAGE_SLOTS // 64
    LOG2E: tl.constexpr = 1.44269504
    ID_STICK: tl.constexpr = 32
    tl.static_assert(
        NKVH * GQA == ID_STICK,
        "paged_fold_hm: one full id stick per page-window trip requires NKVH * GQA == 32; other geometries need the PAGES_PER_TRIP generalization (the pending worklist item)")
    tl.static_assert(HD == 64, "paged_fold_hm: the V plane's [64, HD] block is one contiguous stick block only at HD 64; a wider head needs the per-(window, slab) V gather this kernel does not state yet")
    tl.static_assert(PAGE_SLOTS % 64 == 0, "paged_fold_hm: a page must be a whole number of 64-slot windows")
    # ⛔ THE REGION, NOT THE PAGE: every table below states the WHOLE
    # page-major region the host stages, and every load reads THIS launch's
    # page-window at its own offsets — the runtime's per-pass segment shift
    # is what moves the base from page to page. A view stated at one page
    # would place a 512 B buffer the shift walks right off.
    tl.static_assert(REGION_PAGES >= 1, "paged_fold_hm: the table region holds at least one page")

    # The chunk's q, in torch's [kvh, g, qlen, hd] layout — reloaded every
    # pass; the carry is what persists, not the query.
    q_desc = tl.make_tensor_descriptor(
        desc_q, shape=[BATCH * MQ, HD], strides=[HD, 1],
        block_shape=[BATCH * MQ, HD])
    # The carried online-softmax state, per (kv head, group, query row):
    # torch's carry, at the f16 memory image the module doc's law states
    # (the one-shot's own legalized grading — parity, not degradation).
    # [BATCH*MQ, 1] at strides [64, 1] — the module doc's frame law: the
    # accumulator's own frame, one value per stick at lane 0, the shape
    # the walk's reduce family writes, so the carry never meets a re-row.
    m_desc = tl.make_tensor_descriptor(
        desc_m, shape=[BATCH * MQ, 1], strides=[64, 1],
        block_shape=[BATCH * MQ, 1])
    l_desc = tl.make_tensor_descriptor(
        desc_l, shape=[BATCH * MQ, 1], strides=[64, 1],
        block_shape=[BATCH * MQ, 1])
    acc_desc = tl.make_tensor_descriptor(
        desc_acc, shape=[BATCH * MQ, HD], strides=[HD, 1],
        block_shape=[BATCH * MQ, HD])
    # The pool planes at BLOCK granularity over the WHOLE region: the
    # entries are global stick-block numbers, so a pass may name any page
    # the pool holds.
    kt_blk_desc = tl.make_tensor_descriptor(
        desc_kt, shape=[REGION_PAGES * NKVH * W4, HD * 64],
        strides=[HD * 64, 1], block_shape=[1, HD * 64])
    v_blk_desc = tl.make_tensor_descriptor(
        desc_v, shape=[REGION_PAGES * NKVH * W4, 64 * HD],
        strides=[64 * HD, 1], block_shape=[1, 64 * HD])
    # The block-entry tables, page-major over the whole region at the
    # MASK'S OWN PITCH — one id stick per mask row (the module doc's pitch
    # law): this launch's page is the base's own, so the per-window loads
    # read [w * KTB] — never a page term, which the shift already added.
    # The entries past each page's first W4 * KTB are PADDING, unread and
    # inert: the per-pass shift is one Activation delta, and at MQ > 1 a
    # table's natural page block (512 B) is shorter than a mask page block
    # (512 * MQ B) — the padding is what makes the two pitches one number.
    KTB: tl.constexpr = NKVH * GQA
    tl.static_assert(
        KTB * 4 == 64 * 2,
        "paged_fold_hm: one id stick per mask row requires KTB * 4 bytes of entries per 64-lane fp16 mask row — the geometry broke the pitch convergence the one per-pass shift needs")
    kt_ids_desc = tl.make_tensor_descriptor(
        desc_ids_kt, shape=[REGION_PAGES * W4 * MQ * KTB], strides=[1],
        block_shape=[KTB])
    v_ids_desc = tl.make_tensor_descriptor(
        desc_ids_v, shape=[REGION_PAGES * W4 * MQ * KTB], strides=[1],
        block_shape=[KTB])
    # The additive validity mask, window-major within each page, one row
    # per query row over the whole region — the per-pass shift walks it.
    mask_desc = tl.make_tensor_descriptor(
        desc_mask, shape=[REGION_PAGES * W4 * MQ, 64],
        strides=[64, 1], block_shape=[MQ, 64])

    q = tl.reshape(q_desc.load([0, 0]), [BATCH, MQ, HD])
    m = m_desc.load([0, 0])                             # [B*MQ, 1]
    l = l_desc.load([0, 0])                             # [B*MQ, 1]
    # acc rides the BATCHED spelling inside the loop — the FMA's rescale
    # broadcast is provable only in the rank-3 frame (the FMA's own
    # comment states the law) — and lands flat at the store.
    acc = tl.reshape(acc_desc.load([0, 0]), [BATCH, MQ, HD])

    for w in tl.static_range(W4):
        # The block gathers: one stick block per index entry, both planes,
        # this window's own g-replicated id stick, the ids loaded from the
        # memory-resident tables (never computed in-register).
        kt_ids = kt_ids_desc.load([w * KTB])             # [KTB] i32
        v_ids = v_ids_desc.load([w * KTB])               # [KTB] i32
        kt_blk = tl.reshape(kt_blk_desc.gather(kt_ids, 0),
                            [BATCH, HD, 64])
        v_blk = tl.reshape(v_blk_desc.gather(v_ids, 0),
                           [BATCH, 64, HD])
        qk = tl.dot(q, kt_blk, out_dtype=tl.float16)     # [B, MQ, 64]
        qk = qk * (SCALE * LOG2E)
        # The per-window mask rows, added at the score tile as a leading
        # stride-0 broadcast.
        qk = qk + tl.reshape(mask_desc.load([w * MQ, 0]),
                             [1, MQ, 64])
        # The score tile LENT to the accumulator's own frame — the module
        # doc's frame law: a flat identity ([B,MQ,64] and [B*MQ,1,64] both
        # flat to [B*MQ rows, 64]), so the lens is admitted, and the mask
        # add above stays at [B, MQ, 64] where the [1, MQ, 64] broadcast
        # is the row-split mode the door admits.
        qk_f = tl.reshape(qk, [BATCH * MQ, 1, 64])
        # The online update, folded per trip onto the CARRIED state: every
        # reduce over the last axis lands [B*MQ, 1] — the accumulator
        # frame — with no lens anywhere in the carry chain.
        m_new = tl.maximum(m, tl.max(qk_f, 2))           # [B*MQ, 1]
        rescale = tl.math.exp2(m - m_new).to(tl.float16)
        p = tl.math.exp2(qk_f - m_new[:, :, None]).to(tl.float16)
        l = l * rescale + tl.sum(p, 2)                   # [B*MQ, 1]
        # p lent back to the batched spelling for the score dot (a flat
        # identity again). ⭐ THE FMA'S RESCALE BROADCAST IS PROVABLE ONLY
        # IN THE RANK-3 FRAME: a rank-2 Col spray ([B*MQ, HD] times
        # [B*MQ, 1]) from a COMPUTED source states no vector fact — the
        # door admits an expand_shape's degenerate dim or a one-column
        # LOAD, and rescale is exp2's result — so the rescale is lent to
        # [B, MQ, 1], the rank-3 flat form whose count proof (one source
        # element per result row) is exactly the accumulator frame's own
        # fact. MEASURED, wall 20, build 11: the rank-2 spelling was
        # refused at exactly that proof.
        pv = tl.dot(tl.reshape(p, [BATCH, MQ, 64]), v_blk,
                    out_dtype=tl.float16)                # [B, MQ, HD]
        acc = acc * tl.reshape(rescale, [BATCH, MQ, 1]) + pv
        m = m_new

    m_desc.store([0, 0], m)
    l_desc.store([0, 0], l)
    acc_desc.store([0, 0], tl.reshape(acc, [BATCH * MQ, HD]))


@triton.jit
def paged_fold_finalize_hm(desc_o, desc_l, desc_acc,  #
                           MQ: tl.constexpr, NKVH: tl.constexpr,  #
                           GQA: tl.constexpr, HD: tl.constexpr):
    """The one normalize: `out = acc / l`, ONCE, after the last pass, at the
    carry's own f16 grading. The max is spent by the last pass — nothing
    reads it here."""
    BATCH: tl.constexpr = NKVH * GQA
    o_desc = tl.make_tensor_descriptor(
        desc_o, shape=[BATCH * MQ, HD], strides=[HD, 1],
        block_shape=[BATCH * MQ, HD])
    # [BATCH*MQ, 1] at strides [64, 1] — the module doc's frame law: the
    # accumulator's own frame, one value per stick at lane 0.
    l_desc = tl.make_tensor_descriptor(
        desc_l, shape=[BATCH * MQ, 1], strides=[64, 1],
        block_shape=[BATCH * MQ, 1])
    acc_desc = tl.make_tensor_descriptor(
        desc_acc, shape=[BATCH * MQ, HD], strides=[HD, 1],
        block_shape=[BATCH * MQ, HD])
    l = l_desc.load([0, 0])                              # [B*MQ, 1]
    # acc stays FLAT: the division broadcasts the per-row l over the HD
    # lanes, so the whole normalize runs at the flat frame, no lens.
    acc = acc_desc.load([0, 0])                          # [B*MQ, HD]
    o = (acc / l).to(tl.float16)                         # [B*MQ, HD]
    o_desc.store([0, 0], o)
