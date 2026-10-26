# SPDX-License-Identifier: Apache-2.0
"""The Spyre-native batched paged-decode attention kernel.

torch-spyre's `batched_decode_head_major` algorithm (`spyre_inference/v1/
attention/ops/batched_decode_head_major.py`), stated in Triton over the access
forms this ladder lowers: descriptor windows and index-table block gathers.
NO masked pointer loads anywhere.

The contract (the shared paged-pool laws, stated here and followed by
`batched_prefill.py`):

  * ONE program, grid `[1, 1]` — every sequence in one kernel. All descriptor
    corners are compile-time constants (the chunk and window loops are
    `tl.static_range`, unrolled), so nothing materializes per grid position.
  * ⭐ THE GEOMETRY AND POOL FACTS ARE BOUND, NEVER DERIVED: `SLABS` is
    `AttnGeometry::slabs()`, `STK` the pool's own stick lane count, `W4` its
    windows per page — each a `tl.constexpr` PARAMETER the splice binds from
    the typed const, exactly as it binds `HD`/`NKVH`/`GQA`. A value the
    kernel recomputes is a value it can disagree with the type that sized
    the tables, so there is none (the stick-multiple law fires at the
    SPLICE, where `geom.slabs()` is spent — a sub-stick head is the
    declaring model's own build failure, never a kernel assert).
  * THE POOL IS GATHERED ONE SLAB BLOCK PER INDEX ENTRY. A block is the
    FIXED `[STK, STK]` fp16 unit — STK*STK elements at EVERY head dim: the unit
    both planes' windows divide into (Kᵀ's (window, slab) sub-runs are
    contiguous STK*STK at every hd; the V plane is feature-slab-major, its slab
    runs STK*STK by the plane's own stick law). At `SLABS == 1` the slab block
    IS the old stick block (`HD == STK`, one slab the whole head), so a
    stick-width head is byte-identical by construction. The index values are the host's `GatherEntry` bytes —
    page-relative slab-block numbers composed on the host through the pool's
    ONE door — so this kernel never spells a page stride, a block-in-page
    offset, or a layer term: the descriptor's base is the plane's own
    placement, and every (page, kv head, window, slab) fact rides the index
    VALUE the host staged.
  * ONE WINDOW PER TRIP: the walk is a chunk loop over a window loop, and
    each (chunk, window) trip dots its gathered blocks against q at the
    batch the pool's own rows state, and folds the trip into the running
    softmax IMMEDIATELY — the online merge is associative, so a window needs
    no state axis of its own and no spray of q over one. The two tables
    number their trips the SAME way — SLAB-MAJOR
    (`[(trip*SLABS + s)*KTB + e]`, one `[KTB]` load per (trip, slab)) —
    because BOTH sides dot per slab (the score's partial dots pair q's
    slab-s column window with that slab's gathered blocks, summing SLABS
    partials into the one whole-HD score tile; the V side's PV partials
    land per slab the same way). The two tables differ only in their
    VALUES — each plane's own slab-block numbering.
  * THE ACCUMULATORS ARE ONE TUPLE, ONE SLOT PER SLAB — `[BATCH, GQA, STK]`
    each, `(zeros,) * SLABS` — because the V plane's blocks are per-slab and
    no transpose or gather order can fold them into one `[.., HD]` tensor
    (the V block is [slot, feat'] row-major, so slot sits INSIDE the block
    and the flat orders cannot agree). Each trip's PV partial for slab s
    updates its own tuple slot — `accs[s] = accs[s] * rescale + pv`, a
    compile-time sequence-slot assignment over the static unroll, one
    `[B, GQA, STK]` read-modify-write per slab — and the merge tail stores
    each slot through its own column window of `o` at `s*STK`. At
    `SLABS == 1` this is the old single `[BATCH, GQA, HD]` accumulator and
    single whole-row store exactly.
  * Entry order is torch's own: entry `j*B + s` is block slot `j` of sequence
    `s`, so a sequence's J chunk-slots keep SEPARATE running softmax state and
    merge once at the end (`_entry_local_update` / `_merge_entry_local`).
  * The scale rides log2(e) as an immediate multiply on the dot's result, so
    the softmax is `exp2` end to end (the flash fixtures' own law).
  * The mask is staged at the SCORE TILE'S OWN ROWS — pre-replicated xGQA on
    the host, one row per (entry, kv head, gqa) group of STK lanes per window
    trip — so the add is a pure flat reshape and an elementwise add. No
    broadcast.

Staged by the host, per step:

  * `desc_q`       — Q at its NATURAL shape `[NUM_SEQS*NKVH*GQA, HD]`
                     (the graph tensor's own bytes); the kernel gathers its
                     rows in entry order through `desc_rep_ids` below — a
                     descriptor window plus an index-table gather, the two
                     access forms this ladder lowers. NO pre-repeated fp16
                     staging: torch's `batched_decode_head_major` repeats Q
                     host-side (`rep_row_ids` `index_select`); this kernel
                     states that same data movement as the gather, so the
                     card moves the bytes the reference moves host-side and
                     the splice binds the un-repeated source directly.
  * `desc_o`       — output, head-major `[NUM_SEQS*NQH, HD]`.
  * `desc_kt`      — the resident K^T plane; this kernel reads it as
                     `[PAGES*NKVH*W4*SLABS, STK*STK]` — one slab block a
                     row, the plane's whole slab-block space.
  * `desc_v`       — the resident V plane; read the same way —
                     `[PAGES*NKVH*W4*SLABS, STK*STK]`, the V plane's own
                     slab-block numbering.
  * `desc_rep_ids` — Q's row-repetition table, `[BATCH*GQA]` i32: entry `e`
                     names the Q row the pre-repeated buffer's row `e` held,
                     `(j*B+s)*NKVH*GQA + kvh*GQA + g` → `s*NKVH*GQA +
                     kvh*GQA + g` (torch's own `rep_row_ids` values, the
                     index_select's index vector — the SAME bytes the host
                     reference stages, at i32 width instead of fp16 rows).
  * `desc_ids_kt`  — the K^T plane's slab-block table,
                     `[NUM_CHUNKS * W4 * SLABS * KTB]` i32, chunk-major,
                     window-major within a chunk, SLAB-MAJOR within a
                     window (slab, then entry) — the SAME layout as the V
                     table. Every value is a `GatherEntry` the host minted
                     for the K^T plane's own slab-block numbering. Pad
                     entries name a WRITTEN block and are masked (never
                     read unwritten bytes).
  * `desc_ids_v`   — the V plane's own slab-block table,
                     `[NUM_CHUNKS * W4 * SLABS * KTB]` i32, SLAB-MAJOR
                     within a window (slab, then entry), the V plane's own
                     slab-block numbering.
  * `desc_mask`    — additive fp16 mask `[NUM_CHUNKS * W4*ENTRIES*NKVH*GQA,
                     STK]`, pre-replicated xGQA to the score tile's own rows
                     in the trip's own (entry, kv head, gqa) order: 0 where
                     the slot is valid, -65504 where it is past the sequence's
                     length or a padded window. The finite sentinel is torch's
                     own clamp trick: a slot masked in every chunk still
                     merges with weight exp2(-65504 - m) == 0, never NaN.
"""


@triton.jit
def batched_decode_hm(desc_q, desc_o,  #
                      desc_kt, desc_v, desc_rep_ids, desc_ids_kt, desc_ids_v, desc_mask,  #
                      NUM_SEQS: tl.constexpr, NKVH: tl.constexpr,  #
                      GQA: tl.constexpr, HD: tl.constexpr,  #
                      SLABS: tl.constexpr, STK: tl.constexpr, W4: tl.constexpr,  #
                      PAGES: tl.constexpr,  #
                      BLOCKS_PER_CHUNK: tl.constexpr,  #
                      NUM_CHUNKS: tl.constexpr, SCALE: tl.constexpr):
    """One decode step for every sequence: per-(chunk, window) block gathers
    through the pool's own index unit, an online softmax folded per trip, one
    J-slot merge, one store.

    ⭐ THE GEOMETRY AND POOL CONSTEXPRS ARE BOUND, NEVER DERIVED: SLABS is
    AttnGeometry::slabs() (the splice spends the stick-multiple law there —
    a sub-stick head is the declaring model's build failure, not this
    kernel's), STK is the pool's own stick lane count, W4 its windows per
    page. A value this kernel recomputes is a value it can disagree with the
    type that sized the tables, so there is none."""
    # torch's chunking law at this batch: blocks_per_chunk = 32 // num_seqs,
    # so the chunk's entry slots fill the 32-entry stick exactly. ENTRIES and
    # KTB derive from BOUND params (the batch's own chunking, no geometry in
    # them) — the one arithmetic this kernel states.
    ENTRIES: tl.constexpr = NUM_SEQS * BLOCKS_PER_CHUNK
    BATCH: tl.constexpr = ENTRIES * NKVH
    KTB: tl.constexpr = ENTRIES * NKVH
    LOG2E: tl.constexpr = 1.44269504

    # Q at its NATURAL shape — the graph tensor's own `[NUM_SEQS*NKVH*GQA, HD]`
    # bytes — gathered one row per index entry (the same access form the pool
    # planes take: a descriptor window plus an index-table gather). The
    # entry-order repetition torch stages host-side (`rep_row_ids`
    # `index_select`) is stated HERE as the gather, so the dot's A operand is
    # a load at its natural gathered shape: no pre-repeated staging buffer, no
    # permute, no broadcast.
    q_desc = tl.make_tensor_descriptor(desc_q, shape=[NUM_SEQS * NKVH * GQA, HD],
                                       strides=[HD, 1],
                                       block_shape=[1, STK])
    rep_ids_desc = tl.make_tensor_descriptor(
        desc_rep_ids, shape=[BATCH * GQA], strides=[1],
        block_shape=[BATCH * GQA])
    o_desc = tl.make_tensor_descriptor(desc_o, shape=[NUM_SEQS * NKVH * GQA, HD],
                                       strides=[HD, 1],
                                       block_shape=[NUM_SEQS * NKVH * GQA, STK])
    # The pool planes at SLAB-BLOCK granularity: ONE row = ONE [STK, STK]
    # fp16 block (STK*STK elements), FIXED at every head dim — the unit both
    # planes' windows divide into (Kᵗ's (window, slab) sub-runs are contiguous
    # STK*STK at every hd; V's slab runs are STK*STK by the plane's own stick
    # law). The row count is the plane's own slab-block space (every page of the
    # baked extent, every head, every window, every slab); the runtime
    # re-bases the plane per launch, and the pool a value can reach is
    # bounded where the value is BORN (the host's GatherEntry door), never
    # re-derived here. At SLABS == 1 the slab block IS the stick block, so
    # this is the old descriptor byte for byte.
    kt_blk_desc = tl.make_tensor_descriptor(
        desc_kt, shape=[PAGES * NKVH * W4 * SLABS, STK * STK],
        strides=[STK * STK, 1], block_shape=[1, STK * STK])
    v_blk_desc = tl.make_tensor_descriptor(
        desc_v, shape=[PAGES * NKVH * W4 * SLABS, STK * STK],
        strides=[STK * STK, 1], block_shape=[1, STK * STK])
    # The memory-resident block-entry tables (the host's GatherEntry bytes),
    # one whole stick of ids per (chunk, window) trip, and the
    # pre-replicated additive mask at the score tile's own rows. THE TWO
    # TABLES NUMBER THEIR TRIPS THE SAME WAY — SLAB-MAJOR
    # (`[(trip * SLABS + s) * KTB + e]`, one `[KTB]` load per (trip, slab))
    # — because both sides dot per slab; the two tables differ only in
    # their VALUES (each plane's own slab-block numbering).
    kt_ids_desc = tl.make_tensor_descriptor(
        desc_ids_kt, shape=[NUM_CHUNKS * W4 * SLABS * KTB], strides=[1],
        block_shape=[KTB])
    v_ids_desc = tl.make_tensor_descriptor(
        desc_ids_v, shape=[NUM_CHUNKS * W4 * SLABS * KTB], strides=[1],
        block_shape=[KTB])
    mask_desc = tl.make_tensor_descriptor(
        desc_mask, shape=[NUM_CHUNKS * W4 * BATCH * GQA, STK],
        strides=[STK, 1], block_shape=[BATCH * GQA, STK])

    # The running softmax, per (entry-slot, kv head, gqa): torch's
    # `_entry_local_update` carry, folded once per (chunk, window) trip.
    # The Q gather: one row per rep-table entry, taken once before the trip
    # loop (Q does not vary per trip) — ⭐ ONE COLUMN WINDOW PER SLAB into a
    # tuple, `q_desc.gather(rep_ids, s*STK)` with block `[1, STK]`, because
    # the score dot is per slab (the door's batched gathered contraction
    # frame), each partial pairing the slab's own STK features of q with
    # that slab's gathered blocks. At `SLABS == 1` the one window at column
    # 0 IS the old whole-row gather exactly.
    rep_ids = rep_ids_desc.load([0])                    # [BATCH*GQA] i32
    qs = (tl.zeros([BATCH, GQA, STK], tl.float16),) * SLABS
    for s in tl.static_range(SLABS):
        qs[s] = tl.reshape(q_desc.gather(rep_ids, s * STK), [BATCH, GQA, STK])
    m = tl.full([BATCH, GQA], -65504.0, tl.float32)
    # ⭐ THE DENOMINATOR IS f32, LIKE THE MAX IT PAIRS WITH (wall 13): the same
    # per-trip recurrence updates `m` and `l`, so an f16 `l` rounds every
    # rescale at 2⁻¹¹ while its twin keeps f32 bits, and the final `acc / l`
    # divides an f32-graded sum by a coarser-graded one. `o`'s landing stays
    # f16 (the store's own dtype); only the carry widens. (Same law as
    # `batched_prefill.py`'s carry.)
    l = tl.zeros([BATCH, GQA], tl.float32)
    # ⭐ THE ACCUMULATORS ARE ONE TUPLE, ONE SLOT PER SLAB — [BATCH, GQA, STK]
    # each, `(zeros,) * SLABS` — because the V plane's blocks are per-slab and
    # no transpose, gather order or reshape can fold them into one [.., HD]
    # tensor (the V plane's block is [slot, feat'] row-major, so a per-trip
    # V gather's flat order is (slab, slot, feat') while [B, STK, HD] wants
    # (slot, slab, feat') — slot is INSIDE the block, so no id ordering fixes
    # it). Each trip's PV partial for slab s updates ITS OWN tuple slot
    # (`accs[s] = accs[s] * rescale + pv`, a compile-time sequence-slot
    # assignment over the static unroll — one [B, GQA, STK] read-modify-write
    # per slab, never a where-select over a slabbled accumulator), and the
    # merge tail below stores each slot through its own column window of o.
    # At SLABS == 1 this is the old [BATCH, GQA, HD] accumulator exactly (the
    # tuple is compile-time; its one slot IS the old register).
    accs = (tl.zeros([BATCH, GQA, STK], tl.float16),) * SLABS

    for c in tl.static_range(NUM_CHUNKS):
        for w in tl.static_range(W4):
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
            # `[(trip*SLABS + s)*KTB]`) gathers to a [BATCH, STK, STK]
            # block in the plane's own (feat, slot) order, the partial
            # contracting q's slab-s features against it. SLABS partial
            # dots summing to the one whole-HD score tile;
            # `static_range(1, SLABS)` is EMPTY at `SLABS == 1`, so slab
            # 0's dot IS the old single score dot exactly. The scatter
            # (`kv_cache_write`) addresses the same window through the
            # same `addr` law, so the write and this gather cannot
            # disagree about where a slab lives.
            kt_blk = tl.reshape(
                kt_blk_desc.gather(
                    kt_ids_desc.load([(c * W4 + w) * SLABS * KTB]), 0),
                [BATCH, STK, STK])
            qk = tl.dot(qs[0], kt_blk, out_dtype=tl.float16)  # [B, GQA, STK]
            for s in tl.static_range(1, SLABS):
                kt_blk = tl.reshape(
                    kt_blk_desc.gather(
                        kt_ids_desc.load(
                            [((c * W4 + w) * SLABS + s) * KTB]), 0),
                    [BATCH, STK, STK])
                qk = qk + tl.dot(qs[s], kt_blk, out_dtype=tl.float16)
            qk = qk * (SCALE * LOG2E)
            # The mask, staged at the score tile's own rows: a pure flat
            # reshape, an elementwise add. No broadcast.
            qk = qk + tl.reshape(
                mask_desc.load([(c * W4 + w) * BATCH * GQA, 0]),
                [BATCH, GQA, STK])
            # The online update, folded per trip: running max over the STK
            # score lanes (a LAST-axis reduction), rescaled sum, rescaled
            # accumulator — the max spray over the lanes is a per-row scalar,
            # the one broadcast mode a pointwise kernel states natively.
            m_new = tl.maximum(m, tl.max(qk, 2))             # [B, GQA] f32
            rescale = tl.math.exp2(m - m_new).to(tl.float16)
            p = tl.math.exp2(qk - m_new[:, :, None]).to(tl.float16)
            l = l * rescale + tl.sum(p, 2)
            # The V gather and PV dot, PER SLAB: the V table's slab-major
            # slice is one [KTB] load, and — by the pool's own layout law
            # again (`PagedKvPool::addr`, `KvPlane::V` arm: a V window is
            # FEATURE-SLAB-MAJOR, `(feat/STK)*(PLANE_SLOTS*STK) + slot*STK +
            # feat%STK`, so slab s's run is a row-major [STK slot, STK feat]
            # block) — the gathered block is [BATCH, STK, STK] in the
            # plane's own (slot, lane) order, and the partial product updates
            # ITS OWN tuple slot. The contraction is unchanged: `p`'s STK
            # slots against each slab's STK lanes, the rescale applied once
            # per trip per slot (the slab loop covers every slot), SLABS
            # partial dots summing to the one [B, GQA, HD] product.
            for s in tl.static_range(SLABS):
                v_ids = v_ids_desc.load([((c * W4 + w) * SLABS + s) * KTB])
                v_blk = tl.reshape(v_blk_desc.gather(v_ids, 0),
                                   [BATCH, STK, STK])
                pv = tl.dot(p, v_blk, out_dtype=tl.float16)  # [B, GQA, STK]
                accs[s] = accs[s] * rescale[:, :, None] + pv
            m = m_new

    # The slot merge, torch's `_merge_entry_local`: entry order is j*B + s, so
    # the state reshapes [J, B*NKVH] and the J slots of one sequence merge
    # against their common maximum, then normalize once. The merged maximum's
    # spray along the leading J axis is the row-split broadcast the door
    # admits (`[J, B*NKVH, GQA]`, the source's own rows as the merged axis).
    # The merge weights are slab-free (m and l carry no feature axis), so
    # they are stated ONCE and every accumulator slot merges against them,
    # each through its own column window of the output — the flat (slab,
    # lane) pair IS the natural feature order, slab s storing at column
    # s*STK. At SLABS == 1 this is the old single merge and single store
    # exactly (the one column window IS the whole row).
    mm = tl.reshape(m, [BLOCKS_PER_CHUNK, NUM_SEQS * NKVH, GQA])
    lm = tl.reshape(l, [BLOCKS_PER_CHUNK, NUM_SEQS * NKVH, GQA])
    merged = tl.max(mm, 0)                                  # [B*NKVH, GQA]
    w = tl.math.exp2(mm - merged[None, :, :]).to(tl.float16)
    l = tl.sum(w * lm, 0)
    for s in tl.static_range(SLABS):
        am = tl.reshape(accs[s],
                        [BLOCKS_PER_CHUNK, NUM_SEQS * NKVH, GQA, STK])
        acc = tl.sum(w[:, :, :, None] * am, 0)              # [B*NKVH, GQA, STK]
        # Rank 3 over rank 2 — the denominator sprays over the slot's STK
        # lanes with ONE None.
        o = (acc / l[:, :, None]).to(tl.float16)
        o_desc.store([0, s * STK],
                     tl.reshape(o, [NUM_SEQS * NKVH * GQA, STK]))
