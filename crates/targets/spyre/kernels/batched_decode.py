# SPDX-License-Identifier: Apache-2.0
"""The Spyre-native batched paged-decode attention kernel.

torch-spyre's `batched_decode_head_major` algorithm (`spyre_inference/v1/
attention/ops/batched_decode_head_major.py`), stated in Triton over the access
forms this ladder lowers: descriptor windows and index-table block gathers.
NO masked pointer loads anywhere.

The contract (the module doc of `attn.py` states the shared laws; this file
follows them):

  * ONE program, grid `[1, 1]` — every sequence in one kernel. All descriptor
    corners are compile-time constants (the chunk and window loops are
    `tl.static_range`, unrolled), so nothing materializes per grid position.
  * THE POOL IS GATHERED ONE STICK BLOCK PER INDEX ENTRY. A block is the
    pool's own unit — `HD * 64` elements: the K^T block `[HD, 64]` (64 slots,
    every feature) and the V block `[64, HD]` (64 slots, every feature). The
    index values are the host's `GatherEntry` bytes — page-relative stick-block
    numbers composed on the host through the pool's ONE door — so this kernel
    never spells a page stride, a block-in-page offset, or a layer term: the
    descriptor's base is the plane's own placement, and every (page, kv head,
    window) fact rides the index VALUE the host staged.
  * ONE WINDOW PER TRIP: the walk is a chunk loop over a window loop, and
    each (chunk, window) trip gathers its OWN id stick of `ENTRIES*NKVH`
    block-ids, dots it against q at the batch the pool's own rows state, and
    folds the trip into the running softmax IMMEDIATELY — the online merge is
    associative, so a window needs no state axis of its own and no spray of q
    over one. Every operand of every dot is then a load or a gather result at
    its natural shape: no permute, no relayout, no broadcast anywhere in the
    kernel.
  * Entry order is torch's own: entry `j*B + s` is block slot `j` of sequence
    `s`, so a sequence's J chunk-slots keep SEPARATE running softmax state and
    merge once at the end (`_entry_local_update` / `_merge_entry_local`).
  * The scale rides log2(e) as an immediate multiply on the dot's result, so
    the softmax is `exp2` end to end (the flash fixtures' own law).
  * The mask is staged at the SCORE TILE'S OWN ROWS — pre-replicated xGQA on
    the host (attn.py's own law), one row per (entry, kv head, gqa) group of
    64 lanes per window trip — so the add is a pure flat reshape and an
    elementwise add. No broadcast.

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
                     `[PAGES*NKVH*W4, HD*64]` — one stick block a row.
  * `desc_v`       — the resident V plane; read as `[PAGES*NKVH*W4, 64*HD]`.
  * `desc_rep_ids` — Q's row-repetition table, `[BATCH*GQA]` i32: entry `e`
                     names the Q row the pre-repeated buffer's row `e` held,
                     `(j*B+s)*NKVH*GQA + kvh*GQA + g` → `s*NKVH*GQA +
                     kvh*GQA + g` (torch's own `rep_row_ids` values, the
                     index_select's index vector — the SAME bytes the host
                     reference stages, at i32 width instead of fp16 rows).
  * `desc_ids_kt`  — the K^T plane's block-entry table, `[NUM_CHUNKS *
                     W4*ENTRIES*NKVH]` i32, chunk-major, window-major within
                     a chunk, entry-major within a window (entry, then kv
                     head). Every value is a `GatherEntry` the host minted for
                     the K^T plane's own block numbering. Pad entries name a
                     WRITTEN block and are masked (never read unwritten
                     bytes).
  * `desc_ids_v`   — the V plane's own block-entry table, same shape and
                     order, the V plane's own block numbering.
  * `desc_mask`    — additive fp16 mask `[NUM_CHUNKS * W4*ENTRIES*NKVH*GQA,
                     64]`, pre-replicated xGQA to the score tile's own rows
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
                      PAGE_SLOTS: tl.constexpr, PAGES: tl.constexpr,  #
                      BLOCKS_PER_CHUNK: tl.constexpr,  #
                      NUM_CHUNKS: tl.constexpr, SCALE: tl.constexpr):
    """One decode step for every sequence: per-(chunk, window) block gathers
    through the pool's own index unit, an online softmax folded per trip, one
    J-slot merge, one store."""
    # torch's chunking law at this batch: blocks_per_chunk = 32 // num_seqs,
    # so the chunk's entry slots fill the 32-entry stick exactly.
    ENTRIES: tl.constexpr = NUM_SEQS * BLOCKS_PER_CHUNK
    BATCH: tl.constexpr = ENTRIES * NKVH
    # 64-slot windows per kv head's page — the pool's own stick geometry.
    W4: tl.constexpr = PAGE_SLOTS // 64
    LOG2E: tl.constexpr = 1.44269504

    # ⛔ THE V BLOCK IS ONE CONTIGUOUS STICK BLOCK ONLY AT HD == 64. The V
    # plane lays a head out slab-major in 64-feature sticks, so a wider head's
    # [64, HD] window is several runs apart and the V side must gather per
    # (window, slab) — a form this kernel does not state yet. Refused HERE,
    # loudly, by the kernel's own contract — never silently misaddressed.
    tl.static_assert(HD == 64, "batched_decode_hm: the V plane's [64, HD] block is one contiguous stick block only at HD 64; a wider head needs the per-(window, slab) V gather this kernel does not state yet")
    tl.static_assert(PAGE_SLOTS % 64 == 0, "batched_decode_hm: a page must be a whole number of 64-slot windows")

    # Q at its NATURAL shape — the graph tensor's own `[NUM_SEQS*NKVH*GQA, HD]`
    # bytes — gathered one row per index entry (the same access form the pool
    # planes take: a descriptor window plus an index-table gather). The
    # entry-order repetition torch stages host-side (`rep_row_ids`
    # `index_select`) is stated HERE as the gather, so the dot's A operand is
    # a load at its natural gathered shape: no pre-repeated staging buffer, no
    # permute, no broadcast.
    q_desc = tl.make_tensor_descriptor(desc_q, shape=[NUM_SEQS * NKVH * GQA, HD],
                                       strides=[HD, 1],
                                       block_shape=[1, HD])
    rep_ids_desc = tl.make_tensor_descriptor(
        desc_rep_ids, shape=[BATCH * GQA], strides=[1],
        block_shape=[BATCH * GQA])
    o_desc = tl.make_tensor_descriptor(desc_o, shape=[NUM_SEQS * NKVH * GQA, HD],
                                       strides=[HD, 1],
                                       block_shape=[NUM_SEQS * NKVH * GQA, HD])
    # The pool planes at BLOCK granularity: ONE row = ONE stick block
    # (`HD * 64` elements), the row an index entry names. The row count is the
    # plane's own block space (every page of the baked extent, every head,
    # every window); the runtime re-bases the plane per launch, and the pool
    # a value can reach is bounded where the value is BORN (the host's
    # GatherEntry door), never re-derived here.
    kt_blk_desc = tl.make_tensor_descriptor(
        desc_kt, shape=[PAGES * NKVH * W4, HD * 64],
        strides=[HD * 64, 1], block_shape=[1, HD * 64])
    v_blk_desc = tl.make_tensor_descriptor(
        desc_v, shape=[PAGES * NKVH * W4, 64 * HD],
        strides=[64 * HD, 1], block_shape=[1, 64 * HD])
    # The memory-resident block-entry tables (the host's GatherEntry bytes),
    # one whole stick of ids per (chunk, window) trip, and the
    # pre-replicated additive mask at the score tile's own rows.
    KTB: tl.constexpr = ENTRIES * NKVH
    kt_ids_desc = tl.make_tensor_descriptor(
        desc_ids_kt, shape=[NUM_CHUNKS * W4 * KTB], strides=[1],
        block_shape=[KTB])
    v_ids_desc = tl.make_tensor_descriptor(
        desc_ids_v, shape=[NUM_CHUNKS * W4 * KTB], strides=[1],
        block_shape=[KTB])
    mask_desc = tl.make_tensor_descriptor(
        desc_mask, shape=[NUM_CHUNKS * W4 * BATCH * GQA, 64],
        strides=[64, 1], block_shape=[BATCH * GQA, 64])

    # The running softmax, per (entry-slot, kv head, gqa): torch's
    # `_entry_local_update` carry, folded once per (chunk, window) trip.
    # The Q gather: one row per rep-table entry, the whole `[BATCH*GQA, HD]`
    # frame taken once before the trip loop (Q does not vary per trip).
    rep_ids = rep_ids_desc.load([0])                    # [BATCH*GQA] i32
    q = tl.reshape(q_desc.gather(rep_ids, 0), [BATCH, GQA, HD])
    m = tl.full([BATCH, GQA], -65504.0, tl.float32)
    # ⭐ THE DENOMINATOR IS f32, LIKE THE MAX IT PAIRS WITH (wall 13): the same
    # per-trip recurrence updates `m` and `l`, so an f16 `l` rounds every
    # rescale at 2⁻¹¹ while its twin keeps f32 bits, and the final `acc / l`
    # divides an f32-graded sum by a coarser-graded one. `o`'s landing stays
    # f16 (the store's own dtype); only the carry widens. (Same law as
    # `batched_prefill.py`'s carry.)
    l = tl.zeros([BATCH, GQA], tl.float32)
    acc = tl.zeros([BATCH, GQA, HD], tl.float16)

    for c in tl.static_range(NUM_CHUNKS):
        for w in tl.static_range(W4):
            # The block gathers: one stick block per index entry, both
            # planes, this trip's own id stick, the ids loaded from the
            # memory-resident tables (never computed in-register).
            kt_ids = kt_ids_desc.load([(c * W4 + w) * KTB])  # [KTB] i32
            v_ids = v_ids_desc.load([(c * W4 + w) * KTB])    # [KTB] i32
            kt_blk = tl.reshape(kt_blk_desc.gather(kt_ids, 0),
                                [BATCH, HD, 64])
            v_blk = tl.reshape(v_blk_desc.gather(v_ids, 0),
                               [BATCH, 64, HD])
            # The score dot at the pool's own rows: q natural on the A side,
            # the gathered blocks on the B side, GQA the dot's M axis —
            # never a new descriptor, never a permute, never a spray.
            qk = tl.dot(q, kt_blk, out_dtype=tl.float16)     # [B, GQA, 64]
            qk = qk * (SCALE * LOG2E)
            # The mask, staged at the score tile's own rows: a pure flat
            # reshape, an elementwise add. No broadcast.
            qk = qk + tl.reshape(
                mask_desc.load([(c * W4 + w) * BATCH * GQA, 0]),
                [BATCH, GQA, 64])
            # The online update, folded per trip: running max over the 64
            # score lanes (a LAST-axis reduction), rescaled sum, rescaled
            # accumulator — the max spray over the lanes is a per-row scalar,
            # the one broadcast mode a pointwise kernel states natively.
            m_new = tl.maximum(m, tl.max(qk, 2))             # [B, GQA] f32
            rescale = tl.math.exp2(m - m_new).to(tl.float16)
            p = tl.math.exp2(qk - m_new[:, :, None]).to(tl.float16)
            l = l * rescale + tl.sum(p, 2)
            pv = tl.dot(p, v_blk, out_dtype=tl.float16)      # [B, GQA, HD]
            acc = acc * rescale[:, :, None] + pv
            m = m_new

    # The slot merge, torch's `_merge_entry_local`: entry order is j*B + s, so
    # the state reshapes [J, B*NKVH] and the J slots of one sequence merge
    # against their common maximum, then normalize once. The merged maximum's
    # spray along the leading J axis is the row-split broadcast the door
    # admits (`[J, B*NKVH, GQA]`, the source's own rows as the merged axis).
    mm = tl.reshape(m, [BLOCKS_PER_CHUNK, NUM_SEQS * NKVH, GQA])
    lm = tl.reshape(l, [BLOCKS_PER_CHUNK, NUM_SEQS * NKVH, GQA])
    am = tl.reshape(acc, [BLOCKS_PER_CHUNK, NUM_SEQS * NKVH, GQA, HD])
    merged = tl.max(mm, 0)                                  # [B*NKVH, GQA]
    w = tl.math.exp2(mm - merged[None, :, :]).to(tl.float16)
    l = tl.sum(w * lm, 0)
    acc = tl.sum(w[:, :, :, None] * am, 0)                  # [B*NKVH, GQA, HD]
    o = (acc / l[:, :, None]).to(tl.float16)
    o_desc.store([0, 0], tl.reshape(o, [NUM_SEQS * NKVH * GQA, HD]))
