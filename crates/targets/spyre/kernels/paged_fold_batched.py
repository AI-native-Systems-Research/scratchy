# SPDX-License-Identifier: Apache-2.0
"""The Spyre-native batched paged fold: the per-page attention body the
runtime re-launches once per resident page, for EVERY sequence of a batched
decode step at once.

`batched_decode_hm` walks every page in ONE launch (`tl.static_range` over
`PAGES`), so its context is a BAKED extent — past the widest chunk nothing
served, and the ladder-completeness gate refused the batched ladder for
exactly that. This file is the other shape, `paged_fold.py`'s own contract
restated at the batched fold stick: ONE page per launch, the online-softmax
carry (max, sum, accumulator) in DEVICE MEMORY instead of registers, so
context is a RUNTIME page count. Three kernels, one contract:

  * `paged_fold_batched_seed_hm` — writes the carry's seed values (max = the
    finite -65504 sentinel, sum = 0, accumulator = 0). Runs ONCE per step,
    BEFORE the first pass. A seeding PASS would reset the running state
    mid-accumulation, so the seed is a separate program, not a flag.
  * `paged_fold_batched_hm` — ONE page of the walk for the whole batch:
    loads the carry, gathers this launch's page through the pool's index
    unit, folds the page into the carry, stores the carry. NO finalize, NO
    output: the launch is pure accumulation, and repeating it IS the loop.
  * `paged_fold_batched_finalize_hm` — `out = acc / l`, ONCE, after the
    last pass.

The runtime contract this family states (and `fold_plan` enforces — the
launch-side arithmetic is single-sourced there):

  * A GATHERED FOLD TAKES NO KV SHIFT. The index entries are GLOBAL
    slab-block numbers (`addr = idx * skip_addr + base`), so the pool's
    segment base never moves; the page a pass reads is named by the entries
    the host staged for it.
  * ⭐ THE INDEX TABLES AND THE MASK RIDE THE ACTIVATION SEGMENT, PAGE-MAJOR,
    AT ONE COMMON PITCH — `batched_decode_hm`'s own table law (the module
    doc there states it in full): the pitch is the least common multiple of
    the id table's own row (`KTB = NUM_SEQS*NKVH` entries, the fold stick)
    and the mask's own row (one `STK`-lane stick), the ONE typed item both
    tensors' per-page blocks and the fold's single per-pass Activation
    shift read. Each pass advances the segment base by one page's block —
    `W4*PG*STK*2` bytes, `PG = KTB*GQA` the score rows — and the two
    tensors' blocks are the SAME byte count by construction: the tables are
    ONE FLAT 1D REGION per plane whose load offsets carry the pitch (the
    per-trip stride `SLABS*PITCH` bytes, the trip's slabs spaced
    `KTB_PAD` entries apart — the whole-stick spacing the gather's index
    runs start at — within the trip's `SLABS*PITCH//4`-entry slot, the first
    `W4` trip-blocks of each page's
    `W4*MQ*SLABS*PITCH`-byte block meaningful, the tail padding) — 1D
    because the gather's index vector must be 1D and the KTDP trace
    recovers an indirect read's view only from a RAW descriptor load — and
    the mask is `W4*PG` dense rows of `STK` lanes (each window's PG score
    rows — one per (sequence, kv head, group) — `MROWS*MQ = PG`, none pad).
    The descriptors below state the WHOLE region and load at THIS page's
    window, so the shifted base lands inside the region for every pass.
  * THE CARRY RIDES THE INTERMEDIATE SEGMENT, UNSHIFTED. Every sequence's
    pages all fold into the SAME rows — that is what makes the online
    softmax a running state — so nothing about the carry may move per pass.
    The batched fold's rows are the WHOLE BATCH's score rows (`PG` of them,
    `fold_rows: WholeBatch`): request `r`'s intermediate term is 0, the
    worker's own `IntRepStride` law at that regime.
  * THE CARRY RIDES THE ACCUMULATOR'S OWN FRAME — `[PG, 1]`, ONE VALUE PER
    STICK AT LANE 0 (`strides [STK, 1]`, the pad lanes inert) — the
    `paged_fold.py` frame law (walls 11/19) at the batched row count: the
    walk's reduce family writes that frame and no other, and the door's
    re-row law refuses every bridge to the packed `[KTB, GQA]` frame. So
    the carry STATES the accumulator frame, and the body SPELLS its softmax
    chain there: the masked score tile lent once to `[PG, 1, STK]` (the
    admitted flat identity; the mask add itself stays at `[KTB, GQA, STK]`,
    elementwise — every score row owns its validity row), every reduce over
    the last axis landing `[PG, 1]` with no lens, `p` lent back to
    `[KTB, GQA, STK]` for the PV dot, and the FMA run at the batched
    spelling with the rescale lent to `[KTB, GQA, 1]` — the rank-3 flat
    form's count proof, one source element per result row, is exactly the
    fact the accumulator frame states (MEASURED, wall 20, build 11).
  * ⭐ THE ACCUMULATOR CARRY IS ONE TUPLE, ONE SLOT PER SLAB — the
    `batched_decode.py` law: the V plane's blocks are per-slab and no
    transpose, gather order or reshape folds them into one `[.., HD]`
    tensor, so the carried accumulator is SLABS slots of `[KTB, GQA, STK]`,
    each loaded from and stored to ITS OWN column window of the carry
    (`[0, s*STK]`, block `[PG, STK]`) — the same bytes a whole-row
    `[PG, HD]` access touched, cut at the slab boundaries. The seed writes
    the whole row (one store, all columns zero); the finalize normalizes
    per slab through the same column windows. At `SLABS == 1` every one of
    these is the old single whole-row access exactly.

The per-trip laws are `batched_decode_hm`'s own (both tables SLAB-MAJOR,
one `[KTB]` gather and one partial score dot per slab, the partials summed;
the exp2 softmax; no merge tail — the fold IS the J == 1 merge), restated
at ONE page per launch.

⛔ THE CARRY'S MEMORY IMAGE IS f16 — the SAME grading the one-shot kernel's
LEGALIZED program carries, so the fold is numerically IDENTICAL to the body
it replaces, page for page (`paged_fold.py`'s own law: `LegalizeTypes`
collapses the one-shot's f32 island to f16 before the door ever sees it,
so an f16 carry here is parity, not a degradation). The -65504 sentinel is
f16-exact (the format's own max), so the seed loses nothing by the
narrowing.
"""


@triton.jit
def paged_fold_batched_seed_hm(desc_m, desc_l, desc_acc,  #
                               NUM_SEQS: tl.constexpr, NKVH: tl.constexpr,  #
                               GQA: tl.constexpr, HD: tl.constexpr,  #
                               STK: tl.constexpr):
    """Seed the batched fold's carry: max at the finite sentinel, sum and
    accumulator at zero. Once per step, before the first pass. The carry's
    memory image is f16 (the module doc's law) and the sentinel is f16-exact.
    The accumulator seed is ONE whole-row store — every column zero — so the
    per-slab loads the body states read zeros whichever window they cut."""
    KTB: tl.constexpr = NUM_SEQS * NKVH
    PG: tl.constexpr = KTB * GQA
    # The finite sentinel, not -inf: a masked-everywhere row still
    # normalizes with weight 0, never NaN (torch's own clamp trick).
    # [PG, 1] at strides [STK, 1] — the module doc's frame law: the
    # accumulator's own frame, one value per stick at lane 0, the shape
    # the body's reduce family writes, so no load or store ever re-rows.
    m_desc = tl.make_tensor_descriptor(
        desc_m, shape=[PG, 1], strides=[STK, 1],
        block_shape=[PG, 1])
    l_desc = tl.make_tensor_descriptor(
        desc_l, shape=[PG, 1], strides=[STK, 1],
        block_shape=[PG, 1])
    acc_desc = tl.make_tensor_descriptor(
        desc_acc, shape=[PG, HD], strides=[HD, 1],
        block_shape=[PG, HD])
    m_desc.store([0, 0], tl.full([PG, 1], -65504.0, tl.float16))
    l_desc.store([0, 0], tl.zeros([PG, 1], tl.float16))
    acc_desc.store([0, 0], tl.zeros([PG, HD], tl.float16))


@triton.jit
def paged_fold_batched_hm(desc_q, desc_m, desc_l, desc_acc,  #
                          desc_kt, desc_v, desc_ids_kt, desc_ids_v, desc_mask,  #
                          NUM_SEQS: tl.constexpr, NKVH: tl.constexpr,  #
                          GQA: tl.constexpr, HD: tl.constexpr,  #
                          SLABS: tl.constexpr, STK: tl.constexpr, W4: tl.constexpr,  #
                          PITCH: tl.constexpr,  #
                          REGION_PAGES: tl.constexpr,  #
                          SCALE: tl.constexpr):
    """ONE page of the walk for the whole batch, folded into the carried
    online-softmax state. This launch's page is the one the host's per-pass
    segment shift lands on; the loads below read that page's own window.

    ⭐ THE GEOMETRY AND POOL CONSTEXPRS ARE BOUND, NEVER DERIVED: SLABS is
    AttnGeometry::slabs() (the splice spends the stick-multiple law there —
    a sub-stick head is the declaring model's build failure, not this
    kernel's), STK is the pool's own stick lane count, W4 its windows per
    page, PITCH its common table pitch — the one typed item the tables, the
    mask and the fold's per-pass shift all read."""
    # The fold stick, the score rows and the pitch law's trip-slot count —
    # pure arithmetic on BOUND params, the same arithmetic
    # `batched_decode_hm` states (the splice refuses a non-integral MQ, so
    # a geometry the pitch cannot serve never reaches this kernel).
    KTB: tl.constexpr = NUM_SEQS * NKVH
    PG: tl.constexpr = KTB * GQA
    MQ: tl.constexpr = (PG * STK * 2) // (SLABS * PITCH)
    # The per-slab entry SPACING, padded to a whole 32-entry SenUint32
    # stick: each slab's gather run starts at `trip + s*KTB_PAD`, and the
    # IBR is loaded one 32-entry stick at a time, so a run start that is
    # not a whole stick hands the core words of two different runs — a
    # clean bake, a wrong address. `PITCH//4` is itself a multiple of 32
    # at least `KTB` (PITCH is the lcm of `KTB*4` and 128), so the padded
    # spacing always fits inside the trip slot: a law of the pitch, never
    # a runtime check.
    KTB_PAD: tl.constexpr = (KTB + 31) // 32 * 32
    LOG2E: tl.constexpr = 1.44269504
    # ⛔ THE REGION, NOT THE PAGE: every table below states the WHOLE
    # page-major region the host stages, and every load reads THIS launch's
    # page-window at its own offsets — the runtime's per-pass segment shift
    # is what moves the base from page to page. A view stated at one page
    # would place a one-page buffer the shift walks right off.
    tl.static_assert(REGION_PAGES >= 1, "paged_fold_batched_hm: the table region holds at least one page")

    # The step's q, at its natural `[PG, HD]` rows — reloaded every pass;
    # the carry is what persists, not the query. ⛔ THE SCORE'S FLAT ROW IS
    # Q'S NATURAL ROW ONLY AT ONE REQUEST: the score's rows are
    # request-major (`j = s·NQH + h`), while the bytes a `[PG, HD]` view
    # over the `[R, nqh·HD]` token stream reads are the stream's own
    # STICK-MAJOR residency — head `h` request `s` at stick `h·S + s`,
    # head-major — an order that coincides with the request-major score
    # rows only at `S == 1` (the only width this family has been card-run
    # at; 10-27). At `S > 1` this view needs the same score-row →
    # head-major bridge the one-shot family's repetition table states,
    # which this family does not yet carry — a latent row-order defect at
    # multi-request fold bodies, never reached by a rung these legs select.
    # ⭐ ONE COLUMN WINDOW PER SLAB, loaded once into a tuple: the score
    # dot is per slab (the door's batched gathered contraction frame),
    # each partial pairing the slab's own STK features of q with that
    # slab's gathered blocks — the acc_desc window form. At `SLABS == 1`
    # the one window at column 0 IS the old whole-row load exactly.
    q_desc = tl.make_tensor_descriptor(
        desc_q, shape=[PG, HD], strides=[HD, 1],
        block_shape=[PG, STK])
    # The carried online-softmax state, per (sequence, kv head, group):
    # torch's carry, at the f16 memory image the module doc's law states
    # (the one-shot's own legalized grading — parity, not degradation).
    # [PG, 1] at strides [STK, 1] — the module doc's frame law: the
    # accumulator's own frame, one value per stick at lane 0, the shape
    # the walk's reduce family writes, so the carry never meets a re-row.
    m_desc = tl.make_tensor_descriptor(
        desc_m, shape=[PG, 1], strides=[STK, 1],
        block_shape=[PG, 1])
    l_desc = tl.make_tensor_descriptor(
        desc_l, shape=[PG, 1], strides=[STK, 1],
        block_shape=[PG, 1])
    # The accumulator carry, per slab through its own column window — the
    # module doc's tuple law. The descriptor states the whole row extent
    # ([PG, HD]); every load and store below cuts a [PG, STK] window at
    # column s*STK, the same bytes the old whole-row access touched.
    acc_desc = tl.make_tensor_descriptor(
        desc_acc, shape=[PG, HD], strides=[HD, 1],
        block_shape=[PG, STK])
    # The pool planes at SLAB-BLOCK granularity over the WHOLE region: the
    # entries are global slab-block numbers, so a pass may name any page
    # the pool holds. ONE row = ONE [STK, STK] fp16 block, FIXED at every
    # head dim (the shared pool law `batched_decode.py` states).
    kt_blk_desc = tl.make_tensor_descriptor(
        desc_kt, shape=[REGION_PAGES * NKVH * W4 * SLABS, STK * STK],
        strides=[STK * STK, 1], block_shape=[1, STK * STK])
    v_blk_desc = tl.make_tensor_descriptor(
        desc_v, shape=[REGION_PAGES * NKVH * W4 * SLABS, STK * STK],
        strides=[STK * STK, 1], block_shape=[1, STK * STK])
    # The block-entry tables, ONE FLAT 1D REGION PER PLANE over the whole
    # region at the COMMON PITCH — `batched_decode_hm`'s own table law:
    # the pitch riding the LOAD OFFSETS, the per-trip stride
    # `SLABS*PITCH//4` entries and the per-page block `W4*MQ*SLABS*PITCH`
    # bytes (the same byte count the mask's own per-page block states):
    # this launch's page is the base's own, so the per-window loads read
    # `w * ...` — never a page term, which the shift already added. ⛔ 1D
    # AT THE GATHER'S OWN CONSUMED SHAPE, no reshape between load and
    # gather (the KTDP trace recovers the view only from a raw load, and
    # the gather's index vector must be 1D). ⛔ THE TWO TABLES NUMBER THEIR
    # TRIPS THE SAME WAY — SLAB-MAJOR, slab s's `KTB` entries at `s*KTB_PAD`
    # (the whole-stick spacing the gather's index runs start at),
    # one `[KTB]` load per (trip, slab) — because both sides dot per slab;
    # the two tables differ only in their VALUES (each plane's own
    # slab-block numbering). The entries past each page's meaningful block
    # are PADDING, unread and inert.
    kt_ids_desc = tl.make_tensor_descriptor(
        desc_ids_kt, shape=[REGION_PAGES * W4 * MQ * SLABS * PITCH // 4],
        strides=[1], block_shape=[KTB])
    v_ids_desc = tl.make_tensor_descriptor(
        desc_ids_v, shape=[REGION_PAGES * W4 * MQ * SLABS * PITCH // 4],
        strides=[1], block_shape=[KTB])
    # The additive validity mask, window-major within each page — at the
    # same region and the same common pitch, DENSE: each window owns PG
    # rows of STK lanes, one per score row (every (sequence, kv head,
    # group) owns its validity), so its per-page block is the tables' own
    # byte count and the view's dense row stride is the one every reader
    # addresses.
    mask_desc = tl.make_tensor_descriptor(
        desc_mask, shape=[REGION_PAGES * W4 * PG, STK],
        strides=[STK, 1], block_shape=[PG, STK])

    # ⭐ Q IS ONE TUPLE, ONE COLUMN WINDOW PER SLAB — the score dot is per
    # slab (the door's batched gathered contraction frame), each partial
    # pairing the slab's own STK features of q with that slab's gathered
    # blocks. At `SLABS == 1` the one window at column 0 IS the old
    # whole-row load exactly.
    qs = (tl.zeros([KTB, GQA, STK], tl.float16),) * SLABS
    for s in tl.static_range(SLABS):
        qs[s] = tl.reshape(q_desc.load([0, s * STK]), [KTB, GQA, STK])
    m = m_desc.load([0, 0])                                # [PG, 1]
    l = l_desc.load([0, 0])                                # [PG, 1]
    # ⭐ THE ACCUMULATOR CARRY IS ONE TUPLE, ONE SLOT PER SLAB — the V
    # plane's blocks are per-slab (the module doc's law), so each slot is
    # loaded from and stored to its own column window of the carry. The
    # slots ride the BATCHED spelling inside the loop — the FMA's rescale
    # broadcast is provable only in the rank-3 frame — and land flat at
    # the store.
    accs = (tl.zeros([KTB, GQA, STK], tl.float16),) * SLABS
    for s in tl.static_range(SLABS):
        accs[s] = tl.reshape(acc_desc.load([0, s * STK]), [KTB, GQA, STK])

    for w in tl.static_range(W4):
        # The trip's base in the flat 1D table — the per-trip stride
        # SLABS*PITCH//4 entries is the common pitch riding the offset.
        trip = w * (SLABS * PITCH // 4)
        # ⭐ THE SCORE DOT IS PER SLAB, THE PARTIALS SUMMED — the door's
        # own batched gathered contraction frame (one gathered row per
        # batch row: each slab's [KTB] stick is its own gather, its partial
        # dot contracting that slab's STK features). By the pool's own
        # layout law (`PagedKvPool::addr`, sdsc_abstract.rs `KvPlane::Kt`
        # arm: a Kᵗ window is FEAT-MAJOR — element (feat, slot) of window
        # w at `w*hd*STK + feat*STK + slot` — so slab s's contiguous
        # STK*STK sub-run is a row-major [STK feat, STK slot] block), the
        # per-slab id set (slab s's KTB entries at `trip + s*KTB_PAD`)
        # gathers to a [KTB, STK, STK] block in the plane's own (feat,
        # slot) order, the partial contracting q's slab-s features against
        # it. SLABS partial dots summing to the one whole-HD score tile;
        # `static_range(1, SLABS)` is EMPTY at `SLABS == 1`, so slab 0's
        # dot IS the old single score dot exactly.
        kt_blk = tl.reshape(
            kt_blk_desc.gather(kt_ids_desc.load([trip]), 0),
            [KTB, STK, STK])
        qk = tl.dot(qs[0], kt_blk, out_dtype=tl.float16)     # [K,GQA,STK]
        for s in tl.static_range(1, SLABS):
            kt_blk = tl.reshape(
                kt_blk_desc.gather(kt_ids_desc.load([trip + s * KTB_PAD]), 0),
                [KTB, STK, STK])
            qk = qk + tl.dot(qs[s], kt_blk, out_dtype=tl.float16)
        qk = qk * (SCALE * LOG2E)
        # The per-window mask rows, added at the score tile — elementwise,
        # every (sequence, kv head, group) row owning its validity row: a
        # pure flat reshape, no broadcast.
        qk = qk + tl.reshape(mask_desc.load([w * PG, 0]),
                             [KTB, GQA, STK])
        # The score tile LENT to the accumulator's own frame — the module
        # doc's frame law: a flat identity ([KTB,GQA,STK] and [PG,1,STK]
        # both flat to [PG rows, STK]), so the lens is admitted, and the
        # mask add above stays at [KTB, GQA, STK] where it is elementwise.
        qk_f = tl.reshape(qk, [PG, 1, STK])
        # The online update, folded per trip onto the CARRIED state: every
        # reduce over the last axis lands [PG, 1] — the accumulator frame
        # — with no lens anywhere in the carry chain.
        m_new = tl.maximum(m, tl.max(qk_f, 2))             # [PG, 1]
        rescale = tl.math.exp2(m - m_new).to(tl.float16)
        p = tl.math.exp2(qk_f - m_new[:, :, None]).to(tl.float16)
        l = l * rescale + tl.sum(p, 2)                     # [PG, 1]
        # p lent back to the batched spelling for the PV dot (a flat
        # identity again). ⭐ THE FMA'S RESCALE BROADCAST IS PROVABLE ONLY
        # IN THE RANK-3 FRAME: a rank-2 Col spray ([PG, STK] times
        # [PG, 1]) from a COMPUTED source states no vector fact — the
        # door admits an expand_shape's degenerate dim or a one-column
        # LOAD, and rescale is exp2's result — so the rescale is lent to
        # [KTB, GQA, 1], the rank-3 flat form whose count proof (one
        # source element per result row) is exactly the accumulator
        # frame's own fact. MEASURED, wall 20, build 11: the rank-2
        # spelling was refused at exactly that proof.
        # The V gather and PV dot, PER SLAB — the V table's slab-major
        # slice is one [KTB] 1D load at `s*KTB_PAD` — the gather's own
        # consumed shape, no reshape between load and gather — and the
        # gathered block is [KTB, STK, STK] in the plane's own (slot,
        # lane) order (`PagedKvPool::addr`, `KvPlane::V` arm:
        # feature-slab-major), the partial product updating ITS OWN tuple
        # slot.
        p_b = tl.reshape(p, [KTB, GQA, STK])
        for s in tl.static_range(SLABS):
            v_ids = v_ids_desc.load([trip + s * KTB_PAD])
            v_blk = tl.reshape(v_blk_desc.gather(v_ids, 0),
                               [KTB, STK, STK])
            pv = tl.dot(p_b, v_blk, out_dtype=tl.float16)  # [K,GQA,STK]
            accs[s] = accs[s] * tl.reshape(rescale, [KTB, GQA, 1]) + pv
        m = m_new

    m_desc.store([0, 0], m)
    l_desc.store([0, 0], l)
    for s in tl.static_range(SLABS):
        acc_desc.store([0, s * STK],
                       tl.reshape(accs[s], [PG, STK]))


@triton.jit
def paged_fold_batched_finalize_hm(desc_o, desc_l, desc_acc,  #
                                   NUM_SEQS: tl.constexpr, NKVH: tl.constexpr,  #
                                   GQA: tl.constexpr, HD: tl.constexpr,  #
                                   SLABS: tl.constexpr, STK: tl.constexpr):
    """The one normalize: `out = acc / l`, ONCE, after the last pass, at the
    carry's own f16 grading — per slab, through the same column windows the
    body stores. The max is spent by the last pass — nothing reads it here."""
    KTB: tl.constexpr = NUM_SEQS * NKVH
    PG: tl.constexpr = KTB * GQA
    o_desc = tl.make_tensor_descriptor(
        desc_o, shape=[PG, HD], strides=[HD, 1],
        block_shape=[PG, STK])
    # [PG, 1] at strides [STK, 1] — the module doc's frame law: the
    # accumulator's own frame, one value per stick at lane 0.
    l_desc = tl.make_tensor_descriptor(
        desc_l, shape=[PG, 1], strides=[STK, 1],
        block_shape=[PG, 1])
    acc_desc = tl.make_tensor_descriptor(
        desc_acc, shape=[PG, HD], strides=[HD, 1],
        block_shape=[PG, STK])
    l = l_desc.load([0, 0])                                # [PG, 1]
    # acc stays FLAT: the division broadcasts the per-row l over the slot's
    # STK lanes, so the whole normalize runs at the flat frame, no lens —
    # one column window per slab, the same bytes the old whole-row access
    # touched.
    for s in tl.static_range(SLABS):
        acc = acc_desc.load([0, s * STK])                  # [PG, STK]
        o = (acc / l).to(tl.float16)                       # [PG, STK]
        o_desc.store([0, s * STK], o)
