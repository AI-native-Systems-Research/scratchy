# SPDX-License-Identifier: Apache-2.0
"""The Spyre-native batched paged-decode attention kernel.

torch-spyre's `batched_decode_head_major` algorithm (`spyre_inference/v1/
attention/ops/batched_decode_head_major.py`), stated in Triton over the access
forms this ladder lowers: descriptor windows and index-table block gathers.
NO masked pointer loads anywhere.

The contract (the shared paged-pool laws, stated here and followed by
`batched_prefill.py`):

  * ONE program, grid `[1, 1]` — every sequence in one kernel. All descriptor
    corners are compile-time constants (the page and window loops are
    `tl.static_range`, unrolled), so nothing materializes per grid position.
  * ⭐ THE GEOMETRY AND POOL FACTS ARE BOUND, NEVER DERIVED: `SLABS` is
    `AttnGeometry::slabs()`, `STK` the pool's own stick lane count, `W4` its
    windows per page, `PITCH` the common table pitch — each a `tl.constexpr`
    PARAMETER the splice binds from the typed const, exactly as it binds
    `HD`/`NKVH`/`GQA`. A value the kernel recomputes is a value it can
    disagree with the type that sized the tables, so there is none (the
    stick-multiple law fires at the SPLICE, where `geom.slabs()` is spent —
    a sub-stick head is the declaring model's own build failure, never a
    kernel assert).
  * ⭐ THE ID STICK IS THE FOLD STICK — `KTB = NUM_SEQS*NKVH`, ONE REAL
    ENTRY PER (sequence, kv head), NO chunk slots and NO J-parallelism.
    torch's own chunking law (`blocks_per_chunk = 32 // num_seqs`, one
    running-softmax state per chunk-slot, one `_merge_entry_local` at the
    end) is DELETED here, because the ladder this kernel heads also carries
    the batched fold family (`paged_fold_batched.py`): past the widest
    chunk the walk must fold ONE PAGE-ORDINAL PER LAUNCH into per-
    (sequence, kv head, group) state, and a fold's tables are the pages'
    own — a chunk of BPC pages is a different table law, a different
    accumulator law, and a merge no rung could share. One trip = one page =
    one window = one fold step at EVERY rung, so the one-shot and the fold
    body state the SAME trip and the ladder stages one layout
    (`ladder_stages_one_layout`). The per-trip work is unchanged: the
    chunk family's `[ENTRIES*NKVH]` stick covered BPC pages' worth of
    (sequence, kv head) pairs, and BPC page-sticks cover it back.
  * ⭐ THE SCORE'S ROWS ARE REQUEST-MAJOR; Q AND O RIDE HEAD-MAJOR ROWS —
    two orders that meet as one identity ONLY at `NUM_SEQS == 1`. The
    id tables and the mask number the score tile's flat row
    `(s*NKVH + kvh)*GQA + g` — request-major, the host's own mint —
    while the bytes q arrives in and o leaves in are the token stream's
    stick-blocked residency: head `h` request `s` at row `h*NUM_SEQS +
    s`, `[nqh, S, HD]` dense at one slab (the same head-major order the
    head relayout's row frame states at every head dim, stick
    `s*PG + h*R + r`). The REPETITION TABLE is the bridge, staged by the
    host: ONE `[SLABS*PG]` i32 table, slab-major slices of `PG`, slice
    `s`'s entry `r` = `s*PG + rep[r]` where `rep[r] = (r % NQH)*NUM_SEQS +
    (r // NQH)` is score row `r`'s head-major row — the STICK the gather
    reads and the scatter writes: at one slab the view reads the token
    stream's own sticks (head-major, the residency law above), and at
    every wider head dim the relayout wrap's `[nqh*R, HD]` row frame
    holds head-major row `k` slab `s` at stick `s*nqh*R + k` with
    `nqh*R == PG` — so the entry is the slab's stick base `s*PG` plus the
    head-major row in BOTH cases. ⛔ NOT `rep[r]*SLABS + s`: that spells
    the frame FLAT (`row*HD + slab*STK`), a byte order no stick-blocked
    residency states — the two spellings agree only at `SLABS == 1`, which
    is why every head-dim-64 width stayed coherent while granite-8b
    batched (slabs 2) answered word salad with no fault. CONSTANT per
    geometry (it names the batch's own shape, never a runtime fact; the
    identity table at `NUM_SEQS == 1`). The q read gathers its rows
    through the slice and the output scatters its rows through the SAME
    slice: both edges map score row → head-major stick, so one table
    serves both, and the permutation lives at the EDGES only — the
    score frame, the mask and the id tables are untouched. torch's own
    `rep_row_ids` was this table's ancestor; it died with the chunk-slot
    entry order and is reborn head-major here.
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
  * ONE WINDOW PER TRIP: the walk is a page loop over a window loop
    (`batched_prefill_hm`'s own walk), and each (page, window) trip dots
    its gathered blocks against q at the batch the fold stick states —
    `(sequence, kv head)` × GQA group, q's own row order — and folds the
    trip into the running softmax IMMEDIATELY. The online merge is
    associative, so a window needs no state axis of its own and no spray
    of q over one. The two tables number their trips the SAME way —
    SLAB-MAJOR, slab `s`'s `KTB` entries at `s*KTB_PAD` (the per-slab
    spacing padded to a whole 32-entry SenUint32 stick, the unit the IBR
    loads index runs in) within the trip's `SLABS*PITCH//4`-entry slot —
    because BOTH sides dot per slab: the
    score's partial dots pair q's slab-s column window with that slab's
    gathered blocks, summing `SLABS` partials into the one whole-HD score
    tile, and the V side's PV partials land per slab the same way. The two
    tables differ only in their VALUES — each plane's own slab-block
    numbering.
  * ⭐ THE TABLES AND THE MASK STATE ONE COMMON PITCH — the
    `batched_prefill.py` law, restated at the fold stick: `PITCH` is the
    least common multiple of the id table's own row (`KTB` entries) and
    the mask's own row (one `STK`-lane stick). The id tables are ONE FLAT
    1D REGION per plane whose load offsets carry the pitch — the per-trip
    stride is `SLABS*PITCH` bytes (the trip's slabs spaced `KTB_PAD`
    entries apart — the whole-stick spacing the gather's index runs start
    at — within the trip's `SLABS*PITCH//4`-entry slot, the pad unread), the per-page block
    `W4*MQ*SLABS*PITCH` bytes with the first `W4` trip-blocks meaningful
    and the tail padding — and the mask gives each window `PG =
    KTB*GQA` whole `STK`-lane rows (one per score row: every
    (sequence, kv head, group) owns its validity), `MROWS*MQ = PG` rows
    with none pad — so both tensors' per-page blocks are the SAME byte
    count `W4*PG*STK*2` at every geometry, which is what the fold's ONE
    per-pass Activation shift walks them by (`paged_fold_batched.py`
    states the shift side; every rung of the batched ladder stages this
    same region, `ladder_stages_one_layout`). `MQ = PG*STK*2 //
    (SLABS*PITCH)` is the trip-slot count per window the pitch law
    states; the SPLICE refuses a non-integral one (a geometry the pitch
    cannot serve is the declaring model's own build failure, never a
    kernel assert). The 1D form is not a preference: the gather's index
    vector must be 1D (Triton's own assert) and the KTDP trace recovers
    an indirect read's view only from a RAW descriptor load, so the ids
    are loaded at exactly the shape the gather consumes.
  * ⭐ THE ACCUMULATORS ARE ONE TUPLE, ONE SLOT PER SLAB — `[KTB, GQA,
    STK]` each, `(zeros,) * SLABS` — because the V plane's blocks are
    per-slab and no transpose or gather order can fold them into one
    `[.., HD]` tensor (the V block is [slot, feat'] row-major, so slot
    sits INSIDE the block and the flat orders cannot agree). Each trip's
    PV partial for slab s updates its own tuple slot — `accs[s] =
    accs[s] * rescale + pv`, a compile-time sequence-slot assignment over
    the static unroll, one `[KTB, GQA, STK]` read-modify-write per slab —
    and the landing stores each slot through its own column window of `o`
    at `s*STK`. At `SLABS == 1` this is the old single `[BATCH, GQA, HD]`
    accumulator and single whole-row store exactly.
  * NO MERGE TAIL: every trip folds into per-(sequence, kv head, group)
    state — the state torch's `_merge_entry_local` would merge over J
    chunk-slots, at J == 1. The landing is that merge's identity spelling:
    the weights are exp2(m − m) = 1 exactly, the denominator is the carry
    itself, and each accumulator slot normalizes through its own column
    window. The J > 1 merge arm is DELETED with the chunking (no rung of
    this ladder states it, and a fold cannot share a ladder with it).
  * The scale rides log2(e) as an immediate multiply on the dot's result, so
    the softmax is `exp2` end to end (the flash fixtures' own law).
  * The mask is staged at the SCORE TILE'S OWN ROWS — one row per
    (sequence, kv head, group) of STK lanes per window trip — so the add
    is a pure flat reshape and an elementwise add. No broadcast.

Staged by the host, per step:

  * `desc_q`       — Q at the token stream's own bytes, read through a
                     `[SLABS*PG, STK]` row view whose row `t` is the
                     head-major row `t % PG` at slab `t // PG` (row `t` IS
                     stick `t`, and the sticks are the frame's own — the
                     stream's head-major residency at one slab, the
                     relayout row frame's `s*PG + k` at every wider head
                     dim) — GATHERED per slab through the repetition
                     table's slice (the module doc's edge law), never
                     loaded straight: at `NUM_SEQS > 1` the stream's rows
                     and the score's rows are different orders.
  * `desc_o`       — output, the same `[SLABS*PG, STK]` row law as q
                     (`NQH == NKVH*GQA`), written per slab through the
                     SAME repetition-table slice — scattered, so each
                     score row's result lands at its head-major row.
  * `desc_kt`      — the resident K^T plane; this kernel reads it as
                     `[PAGES*NKVH*W4*SLABS, STK*STK]` — one slab block a
                     row, the plane's whole slab-block space over the
                     walk's own pages.
  * `desc_v`       — the resident V plane; read the same way —
                     `[PAGES*NKVH*W4*SLABS, STK*STK]`, the V plane's own
                     slab-block numbering.
  * `desc_ids_kt`  — the K^T plane's slab-block table, ONE FLAT 1D region
                     of `[REGION_PAGES * W4 * MQ * SLABS * PITCH // 4]`
                     i32 entries: page-major, the per-trip stride
                     `SLABS*PITCH//4` (the first `W4` trip-blocks of each
                     page's block meaningful, the tail padding), the
                     trip's entries ENTRY-MAJOR, SLAB-MINOR (slab s's
                     `KTB` entries at `s*KTB_PAD`, the whole-stick
                     spacing the gather's index runs start at; the
                     gather's own consumption order). Every value is a
                     `GatherEntry` the host minted for the K^T plane's
                     own slab-block numbering. Pad entries name a WRITTEN
                     block and are masked (never read unwritten bytes).
  * `desc_ids_v`   — the V plane's own slab-block table, the same 1D
                     extent and pitch, SLAB-MAJOR within a trip, the V
                     plane's own slab-block numbering.
  * `desc_mask`    — additive fp16 mask `[REGION_PAGES * W4 * PG, STK]`,
                     dense, window-major: row `(p*W4 + w)*PG + (b*GQA +
                     g)` — the score tile's own flat row order — `0.0`
                     where the slot is valid for that (sequence, kv head,
                     group), `-65504.0` where it is past the sequence's
                     length or a padded window. The finite sentinel is
                     torch's own clamp trick: a slot masked in every trip
                     still normalizes with weight exp2(-65504 - m) == 0,
                     never NaN.
  * `desc_rep`     — the repetition table, `[SLABS*PG]` i32 entries,
                     slab-major slices of `PG`: slice `s`'s entry `r` =
                     `s*PG + rep[r]`, the stick of the `[SLABS*PG, STK]`
                     view holding score row `r`'s q/o bytes at slab `s`
                     (the module doc's edge law). CONSTANT per geometry,
                     staged by the host with the tables and mask.
"""


@triton.jit
def batched_decode_hm(desc_q, desc_o,  #
                      desc_kt, desc_v, desc_ids_kt, desc_ids_v, desc_mask,  #
                      desc_rep,  #
                      NUM_SEQS: tl.constexpr, NKVH: tl.constexpr,  #
                      GQA: tl.constexpr, HD: tl.constexpr,  #
                      SLABS: tl.constexpr, STK: tl.constexpr, W4: tl.constexpr,  #
                      PITCH: tl.constexpr,  #
                      PAGES: tl.constexpr,  #
                      REGION_PAGES: tl.constexpr,  #
                      SCALE: tl.constexpr):
    """One decode step for every sequence: a per-(page, window) walk over the
    cached pages through the pool's own index unit, an online softmax folded
    per trip into per-(sequence, kv head, group) state, one normalize per
    slab — `batched_prefill_hm`'s own walk at the fold stick.

    ⭐ THE GEOMETRY AND POOL CONSTEXPRS ARE BOUND, NEVER DERIVED: SLABS is
    AttnGeometry::slabs() (the splice spends the stick-multiple law there —
    a sub-stick head is the declaring model's build failure, not this
    kernel's), STK is the pool's own stick lane count, W4 its windows per
    page, PITCH its common table pitch. A value this kernel recomputes is a
    value it can disagree with the type that sized the tables, so there is
    none."""
    # The fold stick and the score rows — pure arithmetic on BOUND params
    # (the batch's own width and the bound head counts, no geometry in
    # them): the one arithmetic this kernel states, and the pitch law's MQ
    # with it (the splice refuses a non-integral MQ — a geometry the pitch
    # cannot serve never reaches this kernel).
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

    # Q AND O AT THE HEAD-MAJOR ROW VIEW — `[SLABS*PG, STK]`, row `t` the
    # head-major row `t % PG` at slab `t // PG` (row `t` IS stick `t`), ONE
    # WHOLE-STICK ROW PER GATHER/SCATTER (the block laws both edges state:
    # the row count comes from the index vector, so the block is `[1, STK]`).
    # The bytes are q's own — the token stream's stick-blocked residency at
    # one slab (stick `t` = head-major row `t`), the head relayout's row
    # frame at every wider head dim (its own stick law puts head-major row
    # `k` slab `s` at stick `s*nqh*R + k`, and `nqh*R == PG`, so the
    # frame's sticks are the same head-major order per slab) — the SAME
    # order both sides of the relayout wrap state, which is why one table
    # serves every head dim. The ROWS ARE GATHERED AND SCATTERED through
    # the repetition table's per-slab slice (the module doc's edge law),
    # never addressed straight: the score's flat row is request-major and
    # the stream's rows are head-major, orders that coincide only at
    # `NUM_SEQS == 1`.
    q_desc = tl.make_tensor_descriptor(desc_q, shape=[SLABS * PG, STK],
                                       strides=[STK, 1],
                                       block_shape=[1, STK])
    o_desc = tl.make_tensor_descriptor(desc_o, shape=[SLABS * PG, STK],
                                       strides=[STK, 1],
                                       block_shape=[1, STK])
    # The repetition table — ONE FLAT 1D REGION of `[SLABS*PG]` i32
    # entries, slab-major slices of `PG`, loaded RAW per slab (the
    # gather's and the scatter's index vectors must be raw descriptor
    # loads — the KTDP trace law; a computed index vector is refused).
    # The values are the host's constant geometry mapping (the module
    # doc's law), the identity table at `NUM_SEQS == 1`.
    rep_desc = tl.make_tensor_descriptor(desc_rep, shape=[SLABS * PG],
                                         strides=[1], block_shape=[PG])
    # The pool planes at SLAB-BLOCK granularity: ONE row = ONE [STK, STK]
    # fp16 block (STK*STK elements), FIXED at every head dim — the unit both
    # planes' windows divide into (Kᵗ's (window, slab) sub-runs are
    # contiguous STK*STK at every hd; V's slab runs are STK*STK by the
    # plane's own stick law). The row count is the plane's own slab-block
    # space over the WALK's own pages; the runtime re-bases the plane per
    # launch, and the pool a value can reach is bounded where the value is
    # BORN (the host's GatherEntry door), never re-derived here. At
    # `SLABS == 1` the slab block IS the stick block, so this is the old
    # descriptor byte for byte.
    kt_blk_desc = tl.make_tensor_descriptor(
        desc_kt, shape=[PAGES * NKVH * W4 * SLABS, STK * STK],
        strides=[STK * STK, 1], block_shape=[1, STK * STK])
    v_blk_desc = tl.make_tensor_descriptor(
        desc_v, shape=[PAGES * NKVH * W4 * SLABS, STK * STK],
        strides=[STK * STK, 1], block_shape=[1, STK * STK])
    # The memory-resident block-entry tables (the host's GatherEntry bytes),
    # ONE FLAT 1D REGION PER PLANE, STATED AT THE REGION **AND AT THE COMMON
    # PITCH** — `batched_prefill_hm`'s own table law, at the fold stick: the
    # extent the host stages is the ladder's own region (every rung of the
    # batched ladder states the same one, and the fold's per-pass shift walks
    # it), and the pitch rides the LOAD OFFSETS — the per-trip stride is
    # `SLABS*PITCH//4` entries and the per-page block is `W4*MQ*SLABS*PITCH`
    # bytes, the same byte count the mask's own per-page block states (the
    # common-pitch law), with the first `W4` trip-blocks of each page's
    # block meaningful (the trip's `SLABS*KTB` entries contiguous at its
    # head, the pad unread) and the tail padding. ⛔ THE LOADS ARE 1D AT THE
    # GATHER'S OWN CONSUMED SHAPE — no reshape between the ids load and the
    # gather: the KTDP trace recovers an indirect read's memory view only
    # from a RAW descriptor load, and the gather's index vector must be 1D
    # (Triton's own `descriptor_gather` asserts). Both tables number a
    # trip's blocks SLAB-MAJOR — slab s's `KTB` entries at `s*KTB_PAD`, one
    # `[KTB]` load per (trip, slab) — because BOTH sides dot per slab; the
    # two tables differ only in their VALUES (each plane's own slab-block
    # numbering).
    kt_ids_desc = tl.make_tensor_descriptor(
        desc_ids_kt, shape=[REGION_PAGES * W4 * MQ * SLABS * PITCH // 4],
        strides=[1], block_shape=[KTB])
    v_ids_desc = tl.make_tensor_descriptor(
        desc_ids_v, shape=[REGION_PAGES * W4 * MQ * SLABS * PITCH // 4],
        strides=[1], block_shape=[KTB])
    # The additive validity mask, window-major within each page — stated at
    # the same region and the same common pitch, DENSE: each (page, window)
    # trip owns PG rows of STK lanes — one per score row, every (sequence,
    # kv head, group) owning its own validity — so one trip's mask is
    # `PG*STK*2` bytes, the tables' own per-trip bytes (`MROWS*MQ = PG`, the
    # common-pitch law), and the view's dense row stride is the one every
    # reader addresses.
    mask_desc = tl.make_tensor_descriptor(
        desc_mask, shape=[REGION_PAGES * W4 * PG, STK],
        strides=[STK, 1], block_shape=[PG, STK])

    # The running softmax, per (sequence, kv head, group): torch's
    # `_entry_local_update` carry at J == 1, folded once per (page, window)
    # trip.
    # ⭐ Q IS ONE TUPLE, ONE GATHERED SLAB SLICE PER SLAB — `[KTB, GQA,
    # STK]` each, gathered once before the trip loop (q does not vary per
    # trip) — because the score dot is per slab (the door's batched
    # gathered contraction frame: one gathered row per batch row), each
    # partial pairing the slab's own STK features of q with that slab's
    # gathered blocks. The gather runs through the repetition table's
    # slice `s` — score row `r` reads the head-major row the slice names
    # (the module doc's edge law) — so the reshape to `[KTB, GQA, STK]`
    # lands each gathered row at its request-major score row. At
    # `NUM_SEQS == 1` the table is the identity and the gather is the
    # old whole-row load, byte for byte.
    qs = (tl.zeros([KTB, GQA, STK], tl.float16),) * SLABS
    for s in tl.static_range(SLABS):
        qs[s] = tl.reshape(q_desc.gather(rep_desc.load([s * PG]), 0),
                           [KTB, GQA, STK])
    # ⭐ THE CARRY RIDES THE ACCUMULATOR'S OWN FLAT FRAME — `[PG, 1]`, one
    # value per stick at lane 0 — `paged_fold.py`'s own carry law (walls
    # 11/19): the walk's reduce family writes that frame and no other, and
    # the door's re-row law refuses every bridge to the packed
    # `[KTB, GQA]` frame (packed element (b, g) at lane g of stick b is a
    # different byte layout than lane 0 of stick b*GQA+g, not a respelling),
    # so the carry STATES the flat frame — stated type == minted frame — and
    # every view the landing takes (`[PG, 1, STK]`-shaped, flat) is an
    # admitted identity lens over it.
    m = tl.full([PG, 1], -65504.0, tl.float32)
    # ⭐ THE DENOMINATOR IS f32, LIKE THE MAX IT PAIRS WITH (wall 13): the same
    # per-trip recurrence updates `m` and `l`, so an f16 `l` rounds every
    # rescale at 2⁻¹¹ while its twin keeps f32 bits, and the final `acc / l`
    # divides an f32-graded sum by a coarser-graded one. `o`'s landing stays
    # f16 (the store's own dtype); only the carry widens. (Same law as
    # `batched_prefill.py`'s carry.)
    l = tl.zeros([PG, 1], tl.float32)
    # ⭐ THE ACCUMULATORS ARE ONE TUPLE, ONE SLOT PER SLAB — [KTB, GQA, STK]
    # each, `(zeros,) * SLABS` — because the V plane's blocks are per-slab and
    # no transpose, gather order or reshape can fold them into one [.., HD]
    # tensor (the V plane's block is [slot, feat'] row-major, so a per-trip
    # V gather's flat order is (slab, slot, feat') while [B, STK, HD] wants
    # (slot, slab, feat') — slot is INSIDE the block, so no id ordering fixes
    # it). Each trip's PV partial for slab s updates ITS OWN tuple slot
    # (`accs[s] = accs[s] * rescale + pv`, a compile-time sequence-slot
    # assignment over the static unroll — one [KTB, GQA, STK]
    # read-modify-write per slab, never a where-select over a slabbled
    # accumulator), and the landing below scatters each slot through the
    # repetition table's own slab slice. At `SLABS == 1` this is the old
    # [BATCH, GQA, HD] accumulator exactly (the tuple is compile-time; its
    # one slot IS the old register).
    accs = (tl.zeros([KTB, GQA, STK], tl.float16),) * SLABS

    for p in tl.static_range(PAGES):
        for w in tl.static_range(W4):
            # The trip's base in the flat 1D table: page p's block plus w
            # trip-blocks — the per-trip stride SLABS*PITCH//4 entries is the
            # common pitch riding the offset, its pad unread.
            trip = (p * W4 * MQ + w) * (SLABS * PITCH // 4)
            # ⭐ THE SCORE DOT IS PER SLAB, THE PARTIALS SUMMED — the
            # door's own batched gathered contraction frame (one gathered
            # row per batch row: each slab's [KTB] stick is its own
            # gather, its partial dot contracting that slab's STK
            # features). By the pool's own layout law
            # (`PagedKvPool::addr`, sdsc_abstract.rs `KvPlane::Kt` arm):
            # a Kᵗ window is FEAT-MAJOR — element (feat, slot) of window
            # w at `w*hd*STK + feat*STK + slot` — so slab s's contiguous
            # STK*STK sub-run is a row-major [STK feat, STK slot] block,
            # and the slab-major id slice (slab s's KTB entries at
            # `trip + s*KTB_PAD`) gathers to a [KTB, STK, STK] block in the
            # plane's own (feat, slot) order, the partial contracting
            # q's slab-s features against it. SLABS partial dots summing
            # to the one whole-HD score tile; `static_range(1, SLABS)` is
            # EMPTY at `SLABS == 1`, so slab 0's dot IS the old single
            # score dot exactly. The scatter (`kv_cache_write`)
            # addresses the same window through the same `addr` law, so
            # the write and this gather cannot disagree about where a
            # slab lives.
            kt_blk = tl.reshape(
                kt_blk_desc.gather(kt_ids_desc.load([trip]), 0),
                [KTB, STK, STK])
            qk = tl.dot(qs[0], kt_blk, out_dtype=tl.float16)  # [K,GQA,STK]
            for s in tl.static_range(1, SLABS):
                kt_blk = tl.reshape(
                    kt_blk_desc.gather(kt_ids_desc.load([trip + s * KTB_PAD]), 0),
                    [KTB, STK, STK])
                qk = qk + tl.dot(qs[s], kt_blk, out_dtype=tl.float16)
            qk = qk * (SCALE * LOG2E)
            # The mask, staged at the score tile's own rows: a pure flat
            # reshape, an elementwise add. No broadcast — every (sequence,
            # kv head, group) row owns its validity row.
            qk = qk + tl.reshape(
                mask_desc.load([(p * W4 + w) * PG, 0]),
                [KTB, GQA, STK])
            # The online update, folded per trip: running max over the STK
            # score lanes (a LAST-axis reduction), rescaled sum, rescaled
            # accumulator — the max spray over the lanes is a per-row scalar,
            # the one broadcast mode a pointwise kernel states natively.
            # ⭐ THE SOFTMAX CHAIN RUNS AT THE CARRY'S OWN FLAT FRAME — the
            # score tile lent to `[PG, 1, STK]` (a flat identity: both
            # spellings flat to [PG rows, STK] — the lens `paged_fold.py`
            # states for the same lend), every reduce over the last axis
            # landing `[PG, 1]` with no lens anywhere in the carry chain,
            # `p` lent back to the batched spelling for the PV dot, and
            # the FMA's rescale broadcast stated at the rank-3 flat form
            # `[KTB, GQA, 1]` — one source element per result row, wall
            # 20's fixed form. This is `paged_fold.py`'s own softmax body,
            # the frame law its module doc states, restated at this
            # kernel's per-trip fold.
            qk_f = tl.reshape(qk, [PG, 1, STK])
            m_new = tl.maximum(m, tl.max(qk_f, 2))           # [PG, 1] f32
            rescale = tl.math.exp2(m - m_new).to(tl.float16)
            p_tile = tl.math.exp2(qk_f - m_new[:, :, None]).to(tl.float16)
            l = l * rescale + tl.sum(p_tile, 2)              # [PG, 1]
            # The V gather and PV dot, PER SLAB: the V table's slab-major
            # slice is one [KTB] 1D load at `s*KTB_PAD` — the gather's own
            # consumed shape, no reshape between load and gather — and, by
            # the pool's own layout law again (`PagedKvPool::addr`,
            # `KvPlane::V` arm: a V window is FEATURE-SLAB-MAJOR,
            # `(feat/STK)*(PLANE_SLOTS*STK) + slot*STK + feat%STK`, so
            # slab s's run is a row-major [STK slot, STK feat] block) — the
            # gathered block is [KTB, STK, STK] in the plane's own (slot,
            # lane) order, and the partial product updates ITS OWN tuple
            # slot. The contraction is unchanged: `p`'s STK slots against
            # each slab's STK lanes, the rescale applied once per trip per
            # slot (the slab loop covers every slot), SLABS partial dots
            # summing to the one [KTB, GQA, HD] product.
            p_b = tl.reshape(p_tile, [KTB, GQA, STK])
            for s in tl.static_range(SLABS):
                v_ids = v_ids_desc.load([trip + s * KTB_PAD])
                v_blk = tl.reshape(v_blk_desc.gather(v_ids, 0),
                                   [KTB, STK, STK])
                pv = tl.dot(p_b, v_blk, out_dtype=tl.float16)  # [K,GQA,STK]
                accs[s] = accs[s] * tl.reshape(rescale, [KTB, GQA, 1]) + pv
            m = m_new

    # ⭐ NO MERGE TAIL — the landing IS torch's `_merge_entry_local` at
    # J == 1, stated as the identity it is: the weights are exp2(m − m) = 1
    # exactly (a value minus itself is exactly zero, exp2(0) is exactly
    # one), so merged = m, the denominator is the carry itself, and each
    # accumulator slot is its own merge. Spelling the reduces anyway would
    # state sums over a singleton leading axis — in_numel == out_numel, a
    # reduce that removes nothing — and no reduce arm owns that shape with
    # a wide out: the nonstick arm declines reduced-1 by design (its
    # geometry sees the op remove nothing) and the row-reduce body writes
    # ONE value per row, refusing a `[PG, STK]` out. The identity is the
    # honest spelling. The normalize is the rank-3 flat division — one
    # source element per result row — and the weights are slab-free (m and
    # l carry no feature axis), so every accumulator slot lands against
    # them, each SCATTERED through its own repetition-table slice — the
    # score row's result to its head-major stick (the module doc's edge
    # law), slab `s`'s slice naming the stick `s*PG + rep[r]`. At
    # `SLABS == 1` with `NUM_SEQS == 1` this is the old single normalize
    # and single whole-row store, byte for byte (the table is the
    # identity).
    for s in tl.static_range(SLABS):
        o = (tl.reshape(accs[s], [PG, 1, STK]) /
             tl.reshape(l, [PG, 1, 1])).to(tl.float16)
        o_desc.scatter(tl.reshape(o, [PG, STK]), rep_desc.load([s * PG]), 0)
