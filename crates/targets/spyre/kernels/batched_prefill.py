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

The contract (the module doc of `batched_decode.py` states the shared laws;
this file follows them):

  * ONE program, grid `[1, 1]` — the whole chunk in one kernel. All
    descriptor corners are compile-time constants (the page and window loops
    are `tl.static_range`, unrolled), so nothing materializes per grid
    position.
  * ⭐ THE GEOMETRY AND POOL FACTS ARE BOUND, NEVER DERIVED: `SLABS` is
    `AttnGeometry::slabs()`, `STK` the pool's own stick lane count, `W4` its
    windows per page, `KTB` the geometry's own id-stick width — each a
    `tl.constexpr` PARAMETER the splice binds from the typed const, exactly
    as it binds `HD`/`NKVH`/`GQA`. The batch axis is the same number as the
    id stick (`BATCH = KTB`: the g-replicated stick states the dot's (kvh, g)
    rows), derived from the bound const, never recomputed from the head
    counts. A value the kernel recomputes is a value it can disagree with
    the type that sized the tables, so there is none (the stick-multiple law
    fires at the SPLICE, where `geom.slabs()` is spent — a sub-stick head is
    the declaring model's own build failure, never a kernel assert).
  * THE POOL IS GATHERED ONE SLAB BLOCK PER INDEX ENTRY. A block is the
    FIXED `[STK, STK]` fp16 unit — STK*STK elements at EVERY head dim: the
    unit both planes' windows divide into (Kᵀ's (window, slab) sub-runs are
    contiguous STK*STK at every hd; the V plane is feature-slab-major, its
    slab runs STK*STK by the plane's own stick law). At `SLABS == 1` the
    slab block IS the old stick block (`HD == STK`, one slab the whole
    head), so a stick-width head is byte-identical by construction. The
    index values are the host's `GatherEntry` bytes — page-relative
    slab-block numbers composed on the host through the pool's ONE door —
    so this kernel never spells a page stride, a block-in-page offset, or a
    layer term: the descriptor's base is the plane's own placement, and
    every (page, kv head, window, slab) fact rides the index VALUE the host
    staged.
  * ONE WINDOW PER TRIP: the walk is a page loop over a window loop, and
    each (page, window) trip dots its gathered blocks against q at the
    batch the g-replicated stick states — `(kvh, g)`, q's own row order —
    and folds the trip into the running softmax IMMEDIATELY. The online
    merge is associative, so a window needs no state axis of its own and no
    spray of q over one. The two tables number their trips the SAME way —
    SLAB-MAJOR, slab `s`'s `KTB` entries at `s*KTB` within the trip's
    contiguous `SLABS*KTB` entries, one `[KTB]` load per (trip, slab) —
    because BOTH sides dot per slab: the score's partial dots pair q's
    slab-s column window with that slab's gathered blocks, summing `SLABS`
    partials into the one whole-HD score tile, and the V side's PV partials
    land per slab the same way. The two tables differ only in their VALUES
    — each plane's own slab-block numbering, composed on the host through
    the pool's ONE door.
  * ⭐ THE TABLES AND THE MASK STATE ONE COMMON PITCH —
    `AttnGeometry::table_pitch()`, the least common multiple of the id
    table's own row (`KTB` entries) and the mask's own row (one `STK`-lane
    stick). The id tables are ONE FLAT 1D REGION per plane whose load
    offsets carry the pitch — the per-trip stride is `SLABS*PITCH` bytes
    (the trip's `SLABS*KTB` entries contiguous at its head, the pad
    unread), the per-page block `W4*MQ*SLABS*PITCH` bytes with the first
    `W4` trip-blocks meaningful and the tail padding — and the mask gives
    each trip `SLABS * PITCH / (2*STK)` whole `STK`-lane rows per query row
    (its `MQ` query rows first, the rest pad rows, the view DENSE so every
    reader addresses it at its own row stride) — so both tensors'
    per-page blocks are the SAME byte count at
    every geometry, which is what the per-page fold's ONE per-pass
    Activation shift walks them by (`paged_fold.py` states the shift side;
    every rung of a chunked ladder stages this same region,
    `ladder_stages_one_layout`). The 1D form is not a preference: the
    gather's index vector must be 1D (Triton's own assert) and the KTDP
    trace recovers an indirect read's view only from a RAW descriptor
    load, so the ids are loaded at exactly the shape the gather consumes.
    At `KTB == 32` the two natural rows coincide, the pad is empty, and
    the layout is byte-identical to the natural unpadded one.
  * ⭐ THE ACCUMULATORS ARE ONE TUPLE, ONE SLOT PER SLAB — `[BATCH, MQ,
    STK]` each, `(zeros,) * SLABS` — because the V plane's blocks are
    per-slab and no transpose, gather order or reshape can fold them into
    one `[B, MQ, HD]` tensor (the V plane's block is [slot, feat']
    row-major, so a per-trip V gather's flat order is (slab, slot, feat')
    while `[B, MQ, HD]` wants (slot, slab, feat') — slot is INSIDE the
    block, so no id ordering fixes it). Each trip's PV partial for slab s
    updates ITS OWN tuple slot — `accs[s] = accs[s] * rescale + pv`, a
    compile-time sequence-slot assignment over the static unroll — and the
    tail stores each slot through its own column window of `o` at
    `s*STK`. At `SLABS == 1` this is the old single `[BATCH, MQ, HD]`
    accumulator and single whole-row store exactly.
  * q is loaded ONCE — one column window per slab, before the trip loop —
    and reused by every trip's partial dots, the one-program form paying
    the page transfer over every query row, which is torch's own stated
    trade (`page_attn_head_major_prefill.py:17-20`).
  * The scale rides log2(e) as an immediate multiply on the dot's result, so
    the softmax is `exp2` end to end (the flash fixtures' own law).
  * The causal mask is host-staged at a compile-time-constant shape
    (torch's `mask_stack` law at `:51`, re-ordered window-major) and added
    at the score tile as a leading stride-0 broadcast — the row-split
    operand mode the door admits, unamplified over the batch.

Staged by the host, per chunk:

  * `desc_q`       — the chunk's q in torch's own layout: row
                     `(kvh*GQA + g)*MQ + t` = `Q[q_start + t, kvh*GQA + g, :]`,
                     padded rows repeated from row 0 and masked.
  * `desc_o`       — output, the chunk's attention at the same row law as q.
  * `desc_kt`      — the resident Kᵀ plane; this kernel reads it as
                     `[PAGES*NKVH*W4*SLABS, STK*STK]` — one slab block a
                     row, the plane's whole slab-block space.
  * `desc_v`       — the resident V plane; read the same way —
                     `[PAGES*NKVH*W4*SLABS, STK*STK]`, the V plane's own
                     slab-block numbering.
  * `desc_ids_kt`  — the Kᵀ plane's slab-block table, ONE FLAT 1D region
                     of `[REGION_PAGES * W4 * MQ * SLABS * PITCH // 4]`
                     i32 entries: page-major, the per-trip stride
                     `SLABS*PITCH//4` (the first `W4` trip-blocks of each
                     page's block meaningful, the tail padding),
                     ENTRY-MAJOR, SLAB-MINOR within a trip (the trip's
                     `SLABS*KTB` entries contiguous, in the gather's own
                     consumption order) — the group replication rides the
                     values. Every value is a `GatherEntry` the host
                     minted for the Kᵀ plane's own slab-block numbering.
  * `desc_ids_v`   — the V plane's own slab-block table, the same 1D
                     extent and pitch, SLAB-MAJOR within a trip (slab s's
                     `KTB` entries at `s*KTB`), the V plane's own
                     slab-block numbering.
  * `desc_mask`    — additive fp16 mask, `[REGION_PAGES * W4*MROWS*MQ,
                     STK]` dense, `MROWS = SLABS*PITCH/(2*STK)` (the
                     common-pitch law above: a trip's `MROWS*MQ` rows, its
                     `MQ` query rows first, the rest whole pad rows),
                     window-major: row `(p*W4 + w)*MROWS*MQ + t`,
                     `0.0` where the slot is visible to that query row
                     (within the sequence and causally), `-65504.0` where
                     it is not, or where the query row is padding. The
                     finite sentinel is torch's own clamp trick: a
                     masked-everywhere row still normalizes with weight 0,
                     never NaN.
"""


@triton.jit
def batched_prefill_hm(desc_q, desc_o,  #
                       desc_kt, desc_v, desc_ids_kt, desc_ids_v, desc_mask,  #
                       MQ: tl.constexpr, NKVH: tl.constexpr,  #
                       GQA: tl.constexpr, HD: tl.constexpr,  #
                       SLABS: tl.constexpr, STK: tl.constexpr, W4: tl.constexpr,  #
                       KTB: tl.constexpr, PITCH: tl.constexpr,  #
                       PAGES: tl.constexpr,  #
                       REGION_PAGES: tl.constexpr,  #
                       SCALE: tl.constexpr):
    """One prefill chunk for the whole sequence: a per-(page, window) walk
    over the cached pages through the pool's own index unit, an online
    softmax folded per trip, one store per slab.

    ⭐ THE GEOMETRY AND POOL CONSTEXPRS ARE BOUND, NEVER DERIVED: SLABS is
    AttnGeometry::slabs() (the splice spends the stick-multiple law there —
    a sub-stick head is the declaring model's build failure, not this
    kernel's), STK is the pool's own stick lane count, W4 its windows per
    page, KTB the geometry's own id-stick width, PITCH its common table
    pitch. A value this kernel recomputes is a value it can disagree with
    the type that sized the tables, so there is none."""
    # torch's own batch: the KV heads AND the GQA group ride the score dot's
    # batch axis — the g-replicated id stick states it — and that stick IS
    # the bound KTB, so the batch axis derives from it, never from the head
    # counts again. M is the chunk's query rows.
    BATCH: tl.constexpr = KTB
    LOG2E: tl.constexpr = 1.44269504

    # The chunk's q, in torch's [kvh, g, qlen, hd] layout, ONE COLUMN WINDOW
    # PER SLAB — the score dot is per slab (the door's batched gathered
    # contraction frame: one gathered row per batch row), so each slab's
    # partial dot pairs the slab's own STK features of q with that slab's
    # gathered blocks. The windows are loaded ONCE, before the trip loop,
    # into one tuple slot per slab — q does not vary per trip. The output is
    # the same column-window form, stored per slab at `s*STK`.
    q_desc = tl.make_tensor_descriptor(
        desc_q, shape=[BATCH * MQ, HD], strides=[HD, 1],
        block_shape=[BATCH * MQ, STK])
    o_desc = tl.make_tensor_descriptor(
        desc_o, shape=[BATCH * MQ, HD], strides=[HD, 1],
        block_shape=[BATCH * MQ, STK])
    # The pool planes at SLAB-BLOCK granularity: ONE row = ONE [STK, STK]
    # fp16 block (STK*STK elements), FIXED at every head dim — the unit both
    # planes' windows divide into (Kᵗ's (window, slab) sub-runs are
    # contiguous STK*STK at every hd; V's slab runs are STK*STK by the
    # plane's own stick law). The row count is the plane's own slab-block
    # space; the runtime re-bases the plane per launch, and the pool a value
    # can reach is bounded where the value is BORN (the host's GatherEntry
    # door), never re-derived here. At SLABS == 1 the slab block IS the
    # stick block, so this is the old descriptor byte for byte.
    kt_blk_desc = tl.make_tensor_descriptor(
        desc_kt, shape=[PAGES * NKVH * W4 * SLABS, STK * STK],
        strides=[STK * STK, 1], block_shape=[1, STK * STK])
    v_blk_desc = tl.make_tensor_descriptor(
        desc_v, shape=[PAGES * NKVH * W4 * SLABS, STK * STK],
        strides=[STK * STK, 1], block_shape=[1, STK * STK])
    # The memory-resident block-entry tables (the host's GatherEntry bytes),
    # ONE FLAT 1D REGION PER PLANE, STATED AT THE REGION **AND AT THE COMMON
    # PITCH**: the extent the host stages is the ladder's own region (every
    # rung of a chunked ladder states the same one, and a fold's per-pass
    # shift walks it), and the pitch rides the LOAD OFFSETS — the per-trip
    # stride is `SLABS*PITCH//4` entries and the per-page block is
    # `W4*MQ*SLABS*PITCH` bytes, the same byte count the mask's own per-page
    # block states (the common-pitch law), with the first `W4` trip-blocks of
    # each page's block meaningful (the trip's `SLABS*KTB` entries contiguous
    # at its head, the pad unread) and the tail padding. ⛔ THE LOADS ARE 1D
    # AT THE GATHER'S OWN CONSUMED SHAPE — no reshape between the ids load and
    # the gather: the KTDP trace recovers an indirect read's memory view only
    # from a RAW descriptor load, and the gather's index vector must be 1D
    # (Triton's own `descriptor_gather` asserts), so a pitched 2D stick view
    # cannot feed one. Both tables number a trip's blocks SLAB-MAJOR — slab
    # s's `KTB` entries at `s*KTB`, one `[KTB]` load per (trip, slab), the
    # door's own one-gathered-row-per-batch-row frame — because BOTH sides
    # dot per slab (the score's partial dots and the V side's PV partials);
    # the two tables differ only in their VALUES (each plane's own
    # slab-block numbering).
    kt_ids_desc = tl.make_tensor_descriptor(
        desc_ids_kt, shape=[REGION_PAGES * W4 * MQ * SLABS * PITCH // 4],
        strides=[1], block_shape=[KTB])
    v_ids_desc = tl.make_tensor_descriptor(
        desc_ids_v, shape=[REGION_PAGES * W4 * MQ * SLABS * PITCH // 4],
        strides=[1], block_shape=[KTB])
    # The causal mask, window-major within each page, added at the score tile
    # as a leading stride-0 broadcast — stated at the same region and the
    # same common pitch, DENSE: each (page, window) trip owns MROWS*MQ rows of
    # STK lanes, its MQ query rows first and the rest whole pad rows, so one
    # trip's mask is SLABS*PITCH*MQ bytes — the tables' own per-trip bytes —
    # and the view's dense row stride is the one every reader addresses.
    MROWS: tl.constexpr = SLABS * PITCH // (2 * STK)
    mask_desc = tl.make_tensor_descriptor(
        desc_mask, shape=[REGION_PAGES * W4 * MROWS * MQ, STK],
        strides=[STK, 1], block_shape=[MQ, STK])

    # The running softmax across the walk, per (kv head, group, query row):
    # torch's carry, folded once per (page, window) trip.
    # ⭐ Q IS ONE TUPLE, ONE COLUMN WINDOW PER SLAB — `[BATCH, MQ, STK]`
    # each, loaded once before the trip loop — because the score dot is per
    # slab (the door's batched gathered contraction frame: one gathered row
    # per batch row), each partial pairing the slab's own STK features of q
    # with that slab's gathered blocks. At `SLABS == 1` the one window at
    # column 0 IS the old whole-row load exactly.
    qs = (tl.zeros([BATCH, MQ, STK], tl.float16),) * SLABS
    for s in tl.static_range(SLABS):
        qs[s] = tl.reshape(q_desc.load([0, s * STK]), [BATCH, MQ, STK])
    m = tl.full([BATCH, MQ], -65504.0, tl.float32)
    # ⭐ THE DENOMINATOR IS f32, LIKE THE MAX IT PAIRS WITH (wall 13): `m` and `l`
    # are one online-softmax carry — the same per-trip recurrence updates both —
    # so an f16 `l` rounds every rescale at 2⁻¹¹ while its twin `m` keeps f32
    # bits, and the final `acc / l` divides an f32-graded sum by a
    # coarser-graded one. `o`'s landing stays f16 (the store's own dtype); only
    # the carry widens.
    l = tl.zeros([BATCH, MQ], tl.float32)
    # ⭐ THE ACCUMULATORS ARE ONE TUPLE, ONE SLOT PER SLAB — [BATCH, MQ, STK]
    # each, `(zeros,) * SLABS` — because the V plane's blocks are per-slab and
    # no transpose, gather order or reshape can fold them into one [B, MQ, HD]
    # tensor (the V plane's block is [slot, feat'] row-major, so a per-trip V
    # gather's flat order is (slab, slot, feat') while [B, MQ, HD] wants
    # (slot, slab, feat') — slot is INSIDE the block, so no id ordering fixes
    # it). Each trip's PV partial for slab s updates ITS OWN tuple slot
    # (`accs[s] = accs[s] * rescale + pv`, a compile-time sequence-slot
    # assignment over the static unroll — one [B, MQ, STK] read-modify-write
    # per slab, never a where-select over a slabbled accumulator), and the
    # tail below stores each slot through its own column window of the
    # output. At SLABS == 1 this is the old [BATCH, MQ, HD] accumulator
    # exactly (the tuple is compile-time; its one slot IS the old register).
    accs = (tl.zeros([BATCH, MQ, STK], tl.float16),) * SLABS

    for p in tl.static_range(PAGES):
        for w in tl.static_range(W4):
            # The trip's base in the flat 1D table: page p's block plus w
            # trip-blocks — the per-trip stride SLABS*PITCH//4 entries is the
            # common pitch riding the offset, its pad unread.
            trip = (p * W4 * MQ + w) * (SLABS * PITCH // 4)
            # ⭐ THE SCORE DOT IS PER SLAB, THE PARTIALS SUMMED — the door's
            # own batched gathered contraction frame (one gathered row per
            # batch row: each slab's [KTB] stick is its own gather, its
            # partial dot contracting that slab's STK features). By the
            # pool's own layout law (`PagedKvPool::addr`, sdsc_abstract.rs
            # `KvPlane::Kt` arm): a Kᵗ window is FEAT-MAJOR — element
            # (feat, slot) of window w at `w*hd*STK + feat*STK + slot` — so
            # slab s's contiguous STK*STK sub-run is a row-major
            # [STK feat, STK slot] block, and the per-slab id set (slab s's
            # KTB entries at `trip + s*KTB` in the slab-major table) gathers
            # to a [BATCH, STK, STK] block in the plane's own (feat, slot)
            # order, the partial contracting q's slab-s features against it.
            # SLABS partial dots summing to the one [B, MQ, STK] score tile
            # the whole-HD contraction states; `static_range(1, SLABS)` is
            # EMPTY at `SLABS == 1`, so slab 0's dot IS the old single score
            # dot exactly. The scatter (`kv_cache_write`) addresses the same
            # window through the same `addr` law, so the write and this
            # gather cannot disagree about where a slab lives.
            kt_blk = tl.reshape(
                kt_blk_desc.gather(kt_ids_desc.load([trip]), 0),
                [BATCH, STK, STK])
            qk = tl.dot(qs[0], kt_blk, out_dtype=tl.float16)  # [B, MQ, STK]
            for s in tl.static_range(1, SLABS):
                kt_blk = tl.reshape(
                    kt_blk_desc.gather(kt_ids_desc.load([trip + s * KTB]), 0),
                    [BATCH, STK, STK])
                qk = qk + tl.dot(qs[s], kt_blk, out_dtype=tl.float16)
            qk = qk * (SCALE * LOG2E)
            # The causal mask, host-staged window-major, added at the score
            # tile as a leading stride-0 broadcast — the row-split operand
            # mode, every batch row reading the same mask row.
            qk = qk + tl.reshape(mask_desc.load([(p * W4 + w) * MROWS * MQ, 0]),
                                 [1, MQ, STK])
            # The online update, folded per trip: running max over the STK
            # score lanes (a LAST-axis reduction), rescaled sum, rescaled
            # accumulator — the max spray over the lanes is a per-row scalar,
            # the one broadcast mode a pointwise kernel states natively.
            m_new = tl.maximum(m, tl.max(qk, 2))              # [B, MQ] f32
            rescale = tl.math.exp2(m - m_new).to(tl.float16)
            p_tile = tl.math.exp2(qk - m_new[:, :, None]).to(tl.float16)
            l = l * rescale + tl.sum(p_tile, 2)
            # The V gather and PV dot, PER SLAB: the V table's slab-major
            # slice is one [KTB] 1D load at `s*KTB` — the gather's own
            # consumed shape, no reshape between load and gather — and, by
            # the pool's own layout law (`PagedKvPool::addr`, `KvPlane::V` arm: a V window is
            # FEATURE-SLAB-MAJOR, `(feat/STK)*(PLANE_SLOTS*STK) + slot*STK +
            # feat%STK`, so slab s's run is a row-major [STK slot, STK feat]
            # block) — the gathered block is [BATCH, STK, STK] in the
            # plane's own (slot, lane) order, and the partial product updates
            # ITS OWN tuple slot. The contraction is unchanged: `p`'s STK
            # slots against each slab's STK lanes, the rescale applied once
            # per trip per slot (the slab loop covers every slot), SLABS
            # partial dots summing to the one [B, MQ, HD] product.
            for s in tl.static_range(SLABS):
                v_ids = v_ids_desc.load([trip + s * KTB])
                v_blk = tl.reshape(v_blk_desc.gather(v_ids, 0),
                                   [BATCH, STK, STK])
                pv = tl.dot(p_tile, v_blk, out_dtype=tl.float16)  # [B, MQ, STK]
                accs[s] = accs[s] * rescale[:, :, None] + pv
            m = m_new

    # ⭐ THE LANDING IS f16 (the store's dtype); the division itself rides the
    # f32 carry above. The merge weights are slab-free (m and l carry no
    # feature axis), so the normalize is stated ONCE and every accumulator
    # slot lands through its own column window of the output — the flat
    # (slab, lane) pair IS the natural feature order, slab s storing at
    # column s*STK. At SLABS == 1 this is the old single normalize and single
    # whole-row store exactly (the one column window IS the whole row).
    for s in tl.static_range(SLABS):
        o = (accs[s] / l[:, :, None]).to(tl.float16)          # [B, MQ, STK]
        o_desc.store([0, s * STK], tl.reshape(o, [BATCH * MQ, STK]))
