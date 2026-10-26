# SPDX-License-Identifier: Apache-2.0
"""The Spyre-native batched prefill attention kernel.

torch-spyre's chunked-prefill attention (`spyre_inference/v1/attention/ops/
page_attn_head_major_prefill.py`), stated in Triton over the access forms this
ladder lowers: descriptor windows and index-table block gathers. NO masked
pointer loads anywhere.

torch's algorithm, cited:
  * q gathered by row index and laid out `[kvh, g, qlen, hd]` —
    `page_attn_head_major_prefill.py:59-64`;
  * one page per walk trip, `matmul(q, k_page^T) * scale` — `:76-80`;
  * the mask added at the score tile — `:85`;
  * the online softmax carry (max, rescale, sum, output) — `:97-105`;
  * one normalize, then the head-major store — `:123-129`.

The contract (the module doc of `attn.py` states the shared laws; this file
follows them):

  * ONE program, grid `[1, 1]` — the whole chunk in one kernel. All
    descriptor corners are compile-time constants (the page and window loops
    are `tl.static_range`, unrolled), so nothing materializes per grid
    position.
  * THE POOL IS GATHERED ONE STICK BLOCK PER INDEX ENTRY. A block is the
    pool's own own unit — `HD * 64` elements: the K^T block `[HD, 64]` and the
    V block `[64, HD]`. The index values are the host's `GatherEntry` bytes —
    page-relative stick-block numbers composed on the host through the pool's
    ONE door — so this kernel never spells a page stride, a block-in-page
    offset, or a layer term: the descriptor's base is the plane's own
    placement, and every (page, kv head, window) fact rides the index VALUE
    the host staged.
  * ONE FULL ID STICK PER (PAGE, WINDOW) TRIP: a trip's id stick is
    `NKVH * GQA` block-ids — every kv head's block in this window, named once
    per GQA group (the group replication rides the index VALUES, the pool's
    own door, never a kernel-side spray) — which must fill the SenUint32 stick
    exactly. The gather is forced to stick width twice over: Triton refuses a
    gather with fewer than 8 indices, and the door's work division spends its
    cores on the entry axis (`mb`) alone — a gather with fewer entries than
    cores splits `out`, which dxp's `skip_addr` cannot address.
  * K and V ride their OWN ids tables — the two planes' block numberings
    differ, so each names its own blocks.
  * ONE WINDOW PER TRIP: the walk is a page loop over a window loop, and each
    (page, window) trip dots its gathered blocks against q at the batch the
    g-replicated stick states — `(kvh, g)`, q's own row order — and folds the
    trip into the running softmax IMMEDIATELY. The online merge is
    associative, so a window needs no state axis of its own and no spray of q
    over one. Every operand of every dot is then a load or a gather result at
    its natural shape: no permute, no relayout, no broadcast feeding a dot.
  * q is loaded ONCE as one window and reused by every trip's dot — the
    one-program form pays the page transfer over every query row, which is
    torch's own stated trade (`page_attn_head_major_prefill.py:17-20`).
  * The scale rides log2(e) as an immediate multiply on the dot's result, so
    the softmax is `exp2` end to end (the flash fixtures' own law).
  * The causal mask is host-staged at a compile-time-constant shape
    (torch's `mask_stack` law at `:51`, re-ordered window-major) and added at
    the score tile as a leading stride-0 broadcast — the row-split operand
    mode the door admits, unamplified over the batch.

Staged by the host, per chunk:

  * `desc_q`       — the chunk's q in torch's own layout: row
                     `(kvh*GQA + g)*MQ + t` = `Q[q_start + t, kvh*GQA + g, :]`,
                     padded rows repeated from row 0 and masked.
  * `desc_kt`      — the resident K^T plane; this kernel reads it as
                     `[PAGES*NKVH*W4, HD*64]` — one stick block a row.
  * `desc_v`       — the resident V plane; read as `[PAGES*NKVH*W4, 64*HD]`.
  * `desc_ids_kt`  — the K^T plane's block-entry table, `[REGION_PAGES * W4
                     * MQ * NKVH*GQA]` i32, page-major at the MASK'S OWN
                     PITCH (one id stick per mask row: the per-page block
                     is `W4 * MQ` sticks, the first `W4` meaningful, the
                     tail padding — the pitch the per-page fold's ONE
                     per-pass Activation shift walks both tensors by, so
                     the two must state one number; at MQ == 1 the padded
                     form is byte-identical to the natural one),
                     window-major within a page, (kv head, group)-major
                     within a window — the group replication rides the
                     values. Every value is a `GatherEntry` the host
                     minted for the K^T plane's own block numbering.
  * `desc_ids_v`   — the V plane's own block-entry table, same shape and
                     order, the V plane's own block numbering.
  * `desc_mask`    — additive fp16 mask `[REGION_PAGES * W4*MQ, 64]`, window-major:
                     row `(s*W4 + w)*MQ + t`, `0.0` where the slot is visible
                     to that query row (within the sequence and causally),
                     `-65504.0` where it is not, or where the query row is
                     padding. The finite sentinel is torch's own clamp trick:
                     a masked-everywhere row still normalizes with weight 0,
                     never NaN.
  * `desc_o`       — output, the chunk's attention at the same row law as q.
"""


@triton.jit
def batched_prefill_hm(desc_q, desc_o,  #
                       desc_kt, desc_v, desc_ids_kt, desc_ids_v, desc_mask,  #
                       MQ: tl.constexpr, NKVH: tl.constexpr,  #
                       GQA: tl.constexpr, HD: tl.constexpr,  #
                       PAGE_SLOTS: tl.constexpr, PAGES: tl.constexpr,  #
                       REGION_PAGES: tl.constexpr,  #
                       SCALE: tl.constexpr):
    """One prefill chunk for the whole sequence: a per-(page, window) walk
    over the cached pages through the pool's own index unit, an online
    softmax folded per trip, one store.

    `REGION_PAGES` is the extent the two id tables and the mask STATE — the
    page-major region the host stages and a fold's per-pass segment shift
    walks — and it is NOT this kernel's walk count (`PAGES` is). On a
    chunked ladder every rung states the SAME region so their step-input
    placements agree (`ladder_stages_one_layout`), and the per-page fold
    body the ladder also bakes reads the same region one page per launch;
    an unchunked bundle passes `REGION_PAGES = PAGES`, which leaves every
    descriptor byte for byte what a region-less statement would say."""
    # torch's own batch: the KV heads AND the GQA group ride the score dot's
    # batch axis (the g-replicated id stick states it); M is the chunk's
    # query rows.
    BATCH: tl.constexpr = NKVH * GQA
    # 64-slot windows per kv head's page — the pool's own stick geometry.
    W4: tl.constexpr = PAGE_SLOTS // 64
    LOG2E: tl.constexpr = 1.44269504
    # The id stick's width is the door's own (one SenUint32 stick = 32
    # entries), so one (page, window) trip's g-replicated blocks must fill it
    # exactly. A geometry with NKVH*GQA != 32 refuses HERE, at compile time,
    # instead of silently reading a partial stick; the pending
    # PAGES_PER_TRIP/WINDOWS_PER_TRIP generalization lifts this.
    ID_STICK: tl.constexpr = 32
    tl.static_assert(
        NKVH * GQA == ID_STICK,
        "batched_prefill_hm: one full id stick per page-window trip requires NKVH * GQA == 32; other geometries need the PAGES_PER_TRIP generalization (the pending worklist item)")
    # ⛔ THE V BLOCK IS ONE CONTIGUOUS STICK BLOCK ONLY AT HD == 64. The V
    # plane lays a head out slab-major in 64-feature sticks, so a wider head's
    # [64, HD] window is several runs apart and the V side must gather per
    # (window, slab) — a form this kernel does not state yet. Refused HERE,
    # loudly, by the kernel's own contract — never silently misaddressed.
    tl.static_assert(HD == 64, "batched_prefill_hm: the V plane's [64, HD] block is one contiguous stick block only at HD 64; a wider head needs the per-(window, slab) V gather this kernel does not state yet")
    tl.static_assert(PAGE_SLOTS % 64 == 0, "batched_prefill_hm: a page must be a whole number of 64-slot windows")

    # The chunk's q, in torch's [kvh, g, qlen, hd] layout, ONE window at base
    # reused by every trip's dot — the dot's own A operand, at its natural
    # shape.
    q_desc = tl.make_tensor_descriptor(
        desc_q, shape=[BATCH * MQ, HD], strides=[HD, 1],
        block_shape=[BATCH * MQ, HD])
    o_desc = tl.make_tensor_descriptor(
        desc_o, shape=[BATCH * MQ, HD], strides=[HD, 1],
        block_shape=[BATCH * MQ, HD])
    # The pool planes at BLOCK granularity: ONE row = ONE stick block
    # (`HD * 64` elements), the row an index entry names. The row count is the
    # plane's own block space; the runtime re-bases the plane per launch, and
    # the pool a value can reach is bounded where the value is BORN (the
    # host's GatherEntry door), never re-derived here.
    kt_blk_desc = tl.make_tensor_descriptor(
        desc_kt, shape=[PAGES * NKVH * W4, HD * 64],
        strides=[HD * 64, 1], block_shape=[1, HD * 64])
    v_blk_desc = tl.make_tensor_descriptor(
        desc_v, shape=[PAGES * NKVH * W4, 64 * HD],
        strides=[64 * HD, 1], block_shape=[1, 64 * HD])
    # The memory-resident block-entry tables (the host's GatherEntry bytes),
    # one whole stick of ids per (page, window) trip. ⛔ STATED AT THE
    # REGION **AND AT THE MASK'S OWN PITCH**, NOT THE WALK'S: the extent the
    # host stages is the ladder's own region (every rung of a chunked ladder
    # states the same one, and a fold's per-pass shift walks it), and the
    # per-page block is padded to ONE ID STICK PER MASK ROW (`W4 * MQ`
    # sticks, the first `W4` meaningful, the tail unread) so the table and
    # the mask state ONE pitch — the fold's per-pass shift is a single
    # Activation delta and cannot move them by two different numbers. At
    # MQ == 1 the padded form is byte-identical to the natural one. The
    # loads below stay inside this kernel's own `PAGES`-page walk.
    KTB: tl.constexpr = NKVH * GQA
    tl.static_assert(
        KTB * 4 == 64 * 2,
        "batched_prefill_hm: one id stick per mask row requires KTB * 4 bytes of entries per 64-lane fp16 mask row — the geometry broke the pitch convergence the fold's one per-pass shift needs")
    kt_ids_desc = tl.make_tensor_descriptor(
        desc_ids_kt, shape=[REGION_PAGES * W4 * MQ * KTB], strides=[1],
        block_shape=[KTB])
    v_ids_desc = tl.make_tensor_descriptor(
        desc_ids_v, shape=[REGION_PAGES * W4 * MQ * KTB], strides=[1],
        block_shape=[KTB])
    # The causal mask, window-major within each page, added at the score tile
    # as a leading stride-0 broadcast — stated at the same region for the
    # same reason the tables are.
    mask_desc = tl.make_tensor_descriptor(
        desc_mask, shape=[REGION_PAGES * W4 * MQ, 64],
        strides=[64, 1], block_shape=[MQ, 64])

    # The running softmax across the walk, per (kv head, group, query row):
    # torch's carry, folded once per (page, window) trip.
    q = tl.reshape(q_desc.load([0, 0]), [BATCH, MQ, HD])
    m = tl.full([BATCH, MQ], -65504.0, tl.float32)
    # ⭐ THE DENOMINATOR IS f32, LIKE THE MAX IT PAIRS WITH (wall 13): `m` and `l`
    # are one online-softmax carry — the same per-trip recurrence updates both —
    # so an f16 `l` rounds every rescale at 2⁻¹¹ while its twin `m` keeps f32
    # bits, and the final `acc / l` divides an f32-graded sum by a
    # coarser-graded one. `o`'s landing stays f16 (the store's own dtype); only
    # the carry widens.
    l = tl.zeros([BATCH, MQ], tl.float32)
    acc = tl.zeros([BATCH, MQ, HD], tl.float16)

    for s in tl.static_range(PAGES):
        for w in tl.static_range(W4):
            # The block gathers: one stick block per index entry, both
            # planes, this trip's own g-replicated id stick, the ids loaded
            # from the memory-resident tables (never computed in-register).
            kt_ids = kt_ids_desc.load([(s * W4 * MQ + w) * KTB])  # [KTB] i32
            v_ids = v_ids_desc.load([(s * W4 * MQ + w) * KTB])    # [KTB] i32
            kt_blk = tl.reshape(kt_blk_desc.gather(kt_ids, 0),
                                [BATCH, HD, 64])
            v_blk = tl.reshape(v_blk_desc.gather(v_ids, 0),
                               [BATCH, 64, HD])
            # The score dot at the g-replicated stick's own rows: q natural on
            # the A side (its row order IS (kvh, g)), the gathered blocks on
            # the B side, the chunk's query rows the dot's M axis — never a
            # new descriptor, never a permute, never a spray.
            qk = tl.dot(q, kt_blk, out_dtype=tl.float16)      # [B, MQ, 64]
            qk = qk * (SCALE * LOG2E)
            # The causal mask, host-staged window-major, added at the score
            # tile as a leading stride-0 broadcast — the row-split operand
            # mode, every batch row reading the same mask row.
            qk = qk + tl.reshape(mask_desc.load([(s * W4 + w) * MQ, 0]),
                                 [1, MQ, 64])
            # The online update, folded per trip: running max over the 64
            # score lanes (a LAST-axis reduction), rescaled sum, rescaled
            # accumulator — the max spray over the lanes is a per-row scalar,
            # the one broadcast mode a pointwise kernel states natively.
            m_new = tl.maximum(m, tl.max(qk, 2))              # [B, MQ] f32
            rescale = tl.math.exp2(m - m_new).to(tl.float16)
            p = tl.math.exp2(qk - m_new[:, :, None]).to(tl.float16)
            l = l * rescale + tl.sum(p, 2)
            pv = tl.dot(p, v_blk, out_dtype=tl.float16)       # [B, MQ, HD]
            acc = acc * rescale[:, :, None] + pv
            m = m_new

    # ⭐ THE LANDING IS f16 (the store's dtype); the division itself rides the
    # f32 carry above.
    o = (acc / l[:, :, None]).to(tl.float16)                 # [B, MQ, HD]
    o_desc.store([0, 0], tl.reshape(o, [BATCH * MQ, HD]))
