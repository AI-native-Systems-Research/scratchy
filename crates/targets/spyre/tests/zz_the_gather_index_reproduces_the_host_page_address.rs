// SPDX-License-Identifier: Apache-2.0
//! ⭐⭐⭐⭐⭐ THE GATHER REPRODUCES `page_base_bytes` EXACTLY — so batching needs no pool relayout.
//!
//! ## The constraint, in the codebase's own words
//! `SessionKv::fold_requests`: *"ONE LAUNCHED OP HAS ONE PAGE BASE, so B requests cannot share a
//! pass: the fold runs `pages x requests` times, and each pass names both."*
//!
//! That is the whole batch-scaling cost. The base is a per-launch SCALAR, so B requests need B
//! launches of work that is already fully core-parallel, with the mask discarding all but one row's
//! worth. Flash attention's premise — each row reads its own KV — is exactly what a per-launch scalar
//! cannot express.
//!
//! ## Why the hardware gather is a drop-in for it
//! The host resolves a page like this (`fold_plan::page_base_bytes`):
//! ```text
//! phys = block_tables[row][logical_page]
//! addr = phys * page_stride_bytes
//! ```
//! and dxp's index-to-address conversion (`ConvertData_gather_idx`) computes:
//! ```text
//! addr = idx * skip_addr + base_addr
//! ```
//! ⭐ THE SAME AFFINE MAP. `page_base_bytes` is already `idx * stride` with `idx` the physical page
//! number the host looked up. So the free-list nature of the pool is IRRELEVANT to the gather: pages
//! are uniformly spaced by construction (`phys * page_stride_bytes`), and the index is what selects
//! among them. Nothing needs to be made contiguous, because nothing was ever non-uniform.
//!
//! ## The one wrinkle, and why it is free
//! `page_stride_bytes = iters * kv_stride` — one page spans EVERY layer (`superdsc_exec.rs:712`).
//! A single score op's value tensor is ONE layer, so its natural entry is `kv_stride`, not the whole
//! page. That is not a mismatch to engineer around: the index tensor is data WE build on the host, so
//! layer `L` of logical page `p` is simply index `phys(p) * iters + L` at unit stride `kv_stride`.
//! The layer term is absorbed into the index, and the host already has `phys` in `block_tables`.
//!
//! These tests pin that identity against the REAL `page_base_bytes`, so if either side's arithmetic
//! is ever changed the correspondence fails here rather than on the card.

use scratchy_target_spyre::fold_plan::{
    Bytes, PageSlots, RowCursors, RowIdx, RowPage, SessionKv, page_base_bytes,
};

const ITERS: i64 = 40; // layers in the rolled loop
const KV_STRIDE: u64 = 4096; // bytes of ONE layer's page

/// A paged session whose block table is deliberately NON-MONOTONIC and sparse — a free list that has
/// been churned. If the gather's correspondence held only for tidy tables it would be worthless.
fn churned_session() -> SessionKv {
    SessionKv {
        paged: true,
        fold_pages: 0,
        page_slots: PageSlots::per_page(256),
        page_stride_bytes: Bytes(ITERS as u64 * KV_STRIDE),
        // row 0 -> physical 17, 3, 200 ; row 1 -> 4, 199, 18. Interleaved and out of order, which is
        // what a real free list under churn looks like.
        block_tables: vec![vec![17, 3, 200], vec![4, 199, 18]],
        request_positions: RowCursors::bound(vec![0, 300]),
        fold_requests: 2,
        mask_rep_stride_bytes: 0,
        mask_blocks: 0,
        int_rep_stride_bytes: 0,
    }
}

/// The index value the host would put in the gather's index tensor for (row, logical page, layer).
/// This is the WHOLE host-side computation the gather needs, and it is one multiply and one add over
/// data the session already holds.
fn gather_index(s: &SessionKv, row: usize, logical_page: usize, layer: i64) -> i64 {
    s.block_tables[row][logical_page] * ITERS + layer
}

/// dxp's address computation, verbatim from `ConvertData_gather_idx`.
fn gather_addr(idx: i64, skip_addr: u64, base: u64) -> u64 {
    (idx as u64) * skip_addr + base
}

/// ⭐⭐⭐ THE IDENTITY: for every row, every logical page, and every layer, the gather's affine map
/// lands on exactly the byte the host's page lookup would have chosen.
#[test]
fn the_gather_index_lands_exactly_where_the_host_would_have_pointed() {
    let s = churned_session();
    let pool_base = 0x4000_0000u64; // the symbolic base the bundle passes in

    for row in 0..s.block_tables.len() {
        for lp in 0..s.block_tables[row].len() {
            // What the host computes today, per launch: one base for this (row, page).
            let rp = RowPage::of(&s, RowIdx::from_launch_row(row as u64), lp as i64)
                .expect("the row holds this page");
            let host_page_base = page_base_bytes(&s, rp).0;

            for layer in 0..ITERS {
                // Today: page base + the layer's offset within the page.
                let host = pool_base + host_page_base + layer as u64 * KV_STRIDE;
                // With a gather: one index, unit stride of one layer's page.
                let via_gather =
                    gather_addr(gather_index(&s, row, lp, layer), KV_STRIDE, pool_base);
                assert_eq!(
                    via_gather, host,
                    "row {row} logical page {lp} layer {layer}: the gather must address the SAME \
                     byte the host's page table chose (phys {})",
                    s.block_tables[row][lp]
                );
            }
        }
    }
}

/// ⛔ AND THE CORRESPONDENCE IS WHAT MAKES BATCHING POSSIBLE, so state that directly: two DIFFERENT
/// rows' pages are reachable from ONE base with only the index differing.
///
/// This is the property a per-launch scalar base cannot have and is the entire reason the fold costs
/// `pages x requests`. Asserted rather than argued, because "the gather removes the per-launch base"
/// is the claim the whole change rests on.
#[test]
fn two_rows_are_reachable_from_one_base_by_index_alone() {
    let s = churned_session();
    let base = 0x8000_0000u64;
    let layer = 7;

    let addr_of =
        |row: usize, lp: usize| gather_addr(gather_index(&s, row, lp, layer), KV_STRIDE, base);

    // Row 0's page 0 and row 1's page 0 are different physical pages (17 vs 4)...
    assert_ne!(s.block_tables[0][0], s.block_tables[1][0]);
    // ...and both are addressed from the SAME base, differing only in the index.
    let a = addr_of(0, 0);
    let b = addr_of(1, 0);
    assert_ne!(a, b, "distinct rows must land on distinct bytes");

    // Each still matches its own host-computed address, so reaching both from one base costs no
    // correctness.
    for (row, addr) in [(0usize, a), (1usize, b)] {
        let rp = RowPage::of(&s, RowIdx::from_launch_row(row as u64), 0).expect("page 0");
        assert_eq!(
            addr,
            base + page_base_bytes(&s, rp).0 + layer as u64 * KV_STRIDE,
            "row {row} read from the shared base must equal its own host address"
        );
    }
}

/// ⭐⭐⭐ WHAT `skip_addr` MUST EQUAL, AND THEREFORE WHAT THE VALUE OPERAND MUST SPAN.
///
/// This is the one remaining constraint between here and a working gather, so it is written down as
/// arithmetic rather than left to be rediscovered at the emitter.
///
/// `skip_addr` is NOT freely declarable. dxp derives it from the value tensor's own per-dim
/// capacities (`getBufferCapacityForNodePerDim`): an UNBOUNDED dim contributes its per-core datastage
/// extent, and the PINNED dim is clamped to the page. With the pin at 1, that product is exactly the
/// per-core size of everything except the indexed axis — i.e. `skip_addr` == ONE ENTRY'S SIZE.
///
/// So the requirement is an equation, not a preference:
/// ```text
///   skip_addr  ==  kv_stride        (one layer's page, the unit our index steps)
///   skip_addr  ==  product of the value operand's other per-core extents
///   =>  the value operand must SPAN one layer-page: (nkvh, hd, page_slots)
/// ```
/// ⛔ A score op today reads ONE kv head's window, which is far smaller than a layer-page. So the
/// operand cannot be left as-is: it must DECLARE (via `device_extent`, whose whole purpose is "this op
/// iterates a window of a larger allocation") that it is a window into the full layer-page. Declaring
/// less makes `skip_addr` smaller than `kv_stride`, and the index then walks a fraction of a page per
/// step — fluent output built from the wrong keys, with a clean bake.
///
/// That is the next mechanical step, and it needs a card to validate, so it is pinned here rather
/// than half-built: this test states the target number so the wiring is checked against it instead of
/// against a guess.
#[test]
fn skip_addr_must_equal_one_layer_page_so_the_operand_must_span_one() {
    // The pool geometry, in the terms the emitter has to satisfy.
    const NKVH: u64 = 8;
    const HD: u64 = 64;
    const PAGE_SLOTS: u64 = 256;
    const BYTES_PER_ELEM: u64 = 2; // fp16 KV

    // One layer's page, as the pool lays it out.
    let layer_page_bytes = NKVH * HD * PAGE_SLOTS * BYTES_PER_ELEM;

    // A whole page across every layer — what `page_stride_bytes` is.
    let page_stride = ITERS as u64 * layer_page_bytes;

    // The index unit our committed identity uses is ONE LAYER-PAGE, so that is what `skip_addr` has to
    // be. Stated as the relationship rather than a literal, so changing `iters` or the pool shape
    // cannot leave this test asserting a stale constant.
    assert_eq!(
        page_stride / ITERS as u64,
        layer_page_bytes,
        "one index step is one layer-page, by construction of `page_stride = iters * kv_stride`"
    );

    // ⛔ AND THE OPERAND MUST SPAN IT. A single kv head's Kᵀ window is this much smaller:
    let one_kv_head_window = HD * PAGE_SLOTS * BYTES_PER_ELEM;
    assert_eq!(
        layer_page_bytes / one_kv_head_window,
        NKVH,
        "a score op's own operand covers 1/nkvh of a layer-page, so it must DECLARE the full span \
         via device_extent — otherwise skip_addr is {NKVH}x too small and every index step lands a \
         fraction of a page short"
    );
}

/// ⭐ THE POOL'S FREE LIST IS NOT AN OBSTACLE, stated as its own law: page addresses are affine in the
/// PHYSICAL page number regardless of what order the allocator handed them out.
///
/// Recorded because the opposite was believed for most of a session — that a free-list pool would have
/// to be relaid out so gathered entries were contiguous. `page_base_bytes` is `phys * stride`, so the
/// spacing is uniform no matter how scrambled the table is; only the INDEX is scrambled, and a
/// scrambled index is precisely what an index tensor is for.
#[test]
fn page_addresses_are_uniformly_spaced_however_scrambled_the_table_is() {
    let s = churned_session();
    let stride = s.page_stride_bytes.0;

    // Every entry in the (deliberately churned) table obeys `addr == phys * stride`.
    for row in 0..s.block_tables.len() {
        for lp in 0..s.block_tables[row].len() {
            let phys = s.block_tables[row][lp];
            let rp = RowPage::of(&s, RowIdx::from_launch_row(row as u64), lp as i64).unwrap();
            assert_eq!(
                page_base_bytes(&s, rp).0,
                phys as u64 * stride,
                "page addressing must stay affine in the physical page number"
            );
        }
    }

    // And consecutive PHYSICAL pages are exactly one stride apart, which is the only spacing property
    // the gather's fixed `skip_addr` depends on.
    let ident = SessionKv {
        block_tables: vec![vec![0, 1, 2, 3]],
        ..churned_session()
    };
    for p in 1..4i64 {
        let hi = page_base_bytes(
            &ident,
            RowPage::of(&ident, RowIdx::from_launch_row(0), p).unwrap(),
        );
        let lo = page_base_bytes(
            &ident,
            RowPage::of(&ident, RowIdx::from_launch_row(0), p - 1).unwrap(),
        );
        assert_eq!(hi.0 - lo.0, stride, "physical pages are one stride apart");
    }
}

// ══════════════════════════════════════════════════════════════════════════════════════════════════
//  ⛔ THE ENTRY UNIT ABOVE IS SUPERSEDED — the tests below use the pool's OWN geometry
// ══════════════════════════════════════════════════════════════════════════════════════════════════
//
// `KV_STRIDE = 4096` above is a SYNTHETIC number: at nkvh=8/hd=64 a real layer-page is
// `3 * nkvh * hd * PLANE_SLOTS * 2 = 786,432` bytes (THREE planes — Kᵗ, V and natural K), which
// `skip_addr_must_equal_one_layer_page_so_the_operand_must_span_one` above also gets wrong by omitting
// the ×3. Those tests are still true of the AFFINE STRUCTURE, which is all they were written to pin,
// and they stay; what they must no longer be read as is a statement of the entry unit.
//
// The entry unit the emitter actually declares is ONE STICK BLOCK — `PageExtent::of_one_stick()` on the
// slot axis, `hd * 64` elements, MEASURED at 4096 elements / 8192 bytes at hd=64 by
// `zz_the_gathered_kv_operands_page_geometry.rs`. It is a divisor of a layer-page, not equal to one, and
// the emitter's own doc records why: "`skip_addr` only has to DIVIDE the address stride we need to
// express". So `skip_addr == one layer-page` was never a requirement, and the operand does NOT have to
// be widened to span a layer.

/// ⭐⭐⭐⭐⭐ THE ENTRY FACTOR, AGAINST THE REAL POOL AND THE REAL `page_base_bytes`.
///
/// This is the number a wrong value of which is the single most dangerous outcome available in this
/// work: `addr = idx * skip_addr + base` with a short `idx` bakes clean, faults nothing, moves no
/// counter, and assembles fluent text out of another request's keys. So it is asserted as an IDENTITY
/// between the two sides — the host's `phys * page_stride_bytes` and the card's `idx * stick_block` —
/// over a deliberately churned table, at both baked head widths.
#[test]
fn the_entry_factor_makes_the_two_addressings_the_same_byte() {
    use scratchy_subtile::sdsc_abstract::{PagedKvPool, gather_entries_per_page};

    // BOTH baked head widths: the factor carries `hd` (a stick block is `hd * 64`), so a test at one
    // width would pass on a hardcoded 4096 — which is exactly what every doc comment in the tree says.
    for (nkvh, hd) in [(8usize, 64usize), (8, 128)] {
        let pool = PagedKvPool::new(nkvh, hd);
        let layer_page_bytes = pool.layer_stride() as u64 * 2;
        let page_stride = ITERS as u64 * layer_page_bytes;

        let s = SessionKv {
            page_stride_bytes: Bytes(page_stride),
            ..churned_session()
        };
        let per_page = gather_entries_per_page(page_stride, pool)
            .unwrap_or_else(|| panic!("nkvh={nkvh} hd={hd}: a page must be whole stick blocks"));

        // ⭐ THE FACTOR IS NOT A CONSTANT. Stated so a future hardcoded 4096 fails here.
        assert_eq!(
            per_page.get() as u64 * pool.stick_block_bytes(),
            page_stride,
            "nkvh={nkvh} hd={hd}: entries-per-page times the entry size must be exactly one page"
        );

        // ⛔ AND THE IDENTITY ITSELF, over the churned table: for every row and every logical page the
        // gather's address equals the host's, from the same base.
        let base = 0x4000_0000u64;
        for row in 0..s.block_tables.len() {
            for lp in 0..s.block_tables[row].len() {
                let rp = RowPage::of(&s, RowIdx::from_launch_row(row as u64), lp as i64)
                    .expect("the row holds this page");
                let host = base + page_base_bytes(&s, rp).0;
                let idx = s.block_tables[row][lp] * per_page.get();
                let via_gather = gather_addr(idx, pool.stick_block_bytes(), base);
                assert_eq!(
                    via_gather, host,
                    "nkvh={nkvh} hd={hd} row {row} page {lp} (phys {}): the gather must land on the \
                     byte the host's page table chose",
                    s.block_tables[row][lp]
                );
            }
        }

        // ⛔ AND A PAGE NUMBER STAGED RAW IS WRONG BY THIS WHOLE FACTOR — the failure this test's
        // existence is aimed at. Named so the magnitude is on record: hundreds, not a rounding.
        assert!(
            per_page.get() > 1,
            "nkvh={nkvh} hd={hd}: if a page were ONE entry, staging phys directly would be correct \
             and this whole factor would be untestable"
        );
        let raw_page_number_addr = gather_addr(s.block_tables[0][2], pool.stick_block_bytes(), 0);
        let correct = gather_addr(
            s.block_tables[0][2] * per_page.get(),
            pool.stick_block_bytes(),
            0,
        );
        assert_ne!(
            raw_page_number_addr, correct,
            "nkvh={nkvh} hd={hd}: staging a raw page number must NOT coincide with the right address"
        );
    }
}

/// ⛔ A PAGE THAT IS NOT A WHOLE NUMBER OF ENTRIES HAS NO INDEX, so the factor REFUSES.
///
/// Not a rounded answer: `idx * entry` would then land inside the previous page for every page but the
/// first, uniformly, silently. There is no safe default, so there is a `None`.
#[test]
fn a_page_that_is_not_whole_entries_has_no_index_at_all() {
    use scratchy_subtile::sdsc_abstract::{PagedKvPool, gather_entries_per_page};
    let pool = PagedKvPool::new(8, 64);
    assert!(
        gather_entries_per_page(pool.stick_block_bytes() + 1, pool).is_none(),
        "a page one byte past a whole entry cannot be named by any integer index"
    );
    assert!(
        gather_entries_per_page(0, pool).is_none(),
        "an unpaged bundle has no pages to index"
    );
}

/// ⛔⛔⛔ THE TABLE IS A **FLAT ARRAY OVER THE COPY'S OWN `mb` ROWS**, ONE PASS-BLOCK PER FOLD PASS —
/// and the row order is `GatherScratch::row_of`'s, request-MINOR.
///
/// ## What changed and why the old layout does not transfer
/// This asserted "one 32-entry stick per launch row", which was right for the MATMUL as the gathered
/// operand: its index dims were the value's two pinned axes `["out","mb"] = [256, 8]`, paged on `out`, so
/// `mb` was a second index axis and dxp rounded the inner paged extent up to a whole 32-entry stick per
/// row. The gather now lives on the KERNEL-less copy (deeptools refuses one on an op with a `KERNEL`,
/// measured twice on the card), whose gather dim is `mb` with NO second pinned axis — the index is rank-1
/// and `dsc2.cpp:4001-4008`'s product has a single factor. There is no per-row stick to pad to, because
/// the ROW *is* the entry.
///
/// ## The two things this pins
/// * **The row order.** The copy writes scratch row `i` from entry `i`, and the score/value matmuls read
///   row `i` at `i * cols` with `y` stepping ONE row — so the order here IS where request `r`'s keys land.
///   Request-minor is forced: `y` strides one block, so putting the request anywhere else lands on another
///   kv head's or another window's block.
/// * **The pass pitch.** Pass `p`'s entries sit `pass_stride` entries in, because the index rides the
///   MASK's segment shift (it is an `Activation`, and `SegRole::Activation` IS `SEG_MASK`), whose step is
///   one mask block. A pitch that disagrees gathers pass `p` from another pass's entries.
#[test]
fn the_index_is_flat_over_the_scratch_rows_one_block_per_pass() {
    use scratchy_subtile::sdsc_abstract::{
        GatherScratch, PAGE_MASK_COLS, POOL_STICK, PagedKvPool, PrefixMaskShape, QueryRowCount,
        SlotCount, SlotWindow, gather_entries_per_page, gather_index_table,
    };
    // Two kv heads and one 64-slot window keeps the arithmetic readable; the request axis is what the
    // order is about, so it is the axis with two values.
    let pool = PagedKvPool::new(2, 64);
    let scratch = GatherScratch::of_fold_pass(
        pool,
        SlotWindow::count_in(SlotCount::new(64)),
        QueryRowCount::of_mq(2),
    )
    .expect("this geometry admits the flat copy");
    assert_eq!(scratch.rows(), 4, "2 kv heads x 1 window x 2 requests");

    // Row 0 holds two pages, row 1 holds one — the ragged batch that is the ordinary case.
    let tables = vec![vec![17i64, 3], vec![4]];
    // ⭐ BOTH FACTORS THROUGH THEIR ONE DOOR. `per_page` was `10i64` and `pass_stride` was `8usize`
    // — hand-written numbers for quantities that are now [`EntriesPerPage`] (only
    // `gather_entries_per_page` mints one, from the pool's own entry size) and [`PassStride`] (only
    // the prefix mask mints one, because the index rides the mask's shift). Ten pages' worth of
    // stick blocks is a real page stride for this pool; the mask's own block gives the pitch.
    let per_page = gather_entries_per_page(10 * pool.stick_block_bytes(), pool)
        .expect("a page that is a whole number of stick blocks");
    let pass_stride = PrefixMaskShape::<POOL_STICK, PAGE_MASK_COLS>::new(
        1,
        scratchy_subtile::sdsc_abstract::RungWidth::of_baked_rows(2).expect("2 is a baked rung"),
    )
    .expect("one head is a shape")
    .pass_stride();
    assert!(
        pass_stride.get() >= scratch.rows() as usize,
        "the mask's own block must hold one pass's entries (stride {} vs {} rows)",
        pass_stride.get(),
        scratch.rows()
    );
    let t = gather_index_table(scratch, &tables, 2, per_page, pass_stride).expect("a table");
    assert_eq!(
        t.len(),
        2 * pass_stride.get(),
        "one pass block per fold pass"
    );

    // ⭐ EVERY ENTRY, CHECKED AGAINST THE TWO LAWS IT IS COMPOSED OF — the page term from the block
    // table and the block-in-page term from the pool's own two sizes. Recomputed here rather than
    // typed as literals, so this is a check of the COMPOSITION and not of one geometry's numbers.
    for p in 0..2usize {
        for r in 0..2u32 {
            let row_pages = &tables[r as usize];
            let phys = *row_pages.get(p).unwrap_or(row_pages.last().unwrap());
            for kvh in 0..2u32 {
                let i = scratch.row_of(kvh, 0, r).expect("in range") as usize;
                // The composition spelled the LONG way — `phys * per_page + block` over the raw
                // factor — and compared against what the table built through `GatherEntry`. This is
                // the one place the arithmetic may be written out, because writing it out
                // INDEPENDENTLY is what makes the comparison evidence.
                let want =
                    phys * per_page.get() + scratch.block_in_page(kvh, 0).expect("in range") as i64;
                assert_eq!(
                    t[p * pass_stride.get() + i].as_i32() as i64,
                    want,
                    "pass {p}, kv head {kvh}, request {r} at flat row {i}"
                );
            }
        }
    }
    // ⛔ REQUEST-MINOR, STATED AS THE PROPERTY THE MATMUL RELIES ON: adjacent requests of one
    // (kv head, window) are ADJACENT rows, so a `y` step of one row is a step of one request.
    assert_eq!(
        scratch.row_of(1, 0, 1).unwrap() - scratch.row_of(1, 0, 0).unwrap(),
        1,
        "a `y` step must move ONE request, which is one scratch row"
    );
    // ⛔ A SHORT ROW REPEATS ITS OWN LAST PAGE, never page 0 — its own keys, which its prefix mask
    // already invalidates for those columns, where page 0 is another row's.
    let i = scratch.row_of(0, 0, 1).expect("in range") as usize;
    assert_eq!(
        t[pass_stride.get() + i].as_i32() as i64,
        4 * per_page.get(),
        "request 1 holds one page, so its pass-1 entry repeats its OWN page 4"
    );
    // ⛔ AND THE REFUSALS, each for a shape that has no honest answer.
    //
    // ⭐ ONE OF THEM IS GONE BECAUSE IT BECAME UNREPRESENTABLE, NOT BECAUSE IT STOPPED MATTERING:
    // "a zero entry factor would make every page the same address" passed `0` for the factor, and
    // [`EntriesPerPage`] has no such value — its only constructor divides a NONZERO page stride by
    // the pool's entry size, so the quotient is at least 1. A runtime assertion for a state the type
    // system now excludes is a test of dead code, so it is deleted rather than reworded.
    assert!(
        gather_index_table(scratch, &[vec![17], vec![]], 2, per_page, pass_stride).is_none(),
        "a launch row with no pages cannot be given one"
    );
    assert!(
        gather_index_table(scratch, &tables[..1], 2, per_page, pass_stride).is_none(),
        "fewer installed page maps than the scratch's requests is a bind for rows nobody installed"
    );
}

/// ⭐⭐⭐⭐⭐ THE HOST STAGES FOR THE **CEILING** BODY AND THE LAUNCH RUNS AN **INTERIOR** ONE — and the
/// row of every coordinate they share must be the SAME NUMBER.
///
/// ## The defect this is the gate on
/// A paged decode rung bakes a LADDER of bodies (`ActiveCap::decode_ladder`: interior rungs 64 / 128 …
/// plus the `PAGE_SLOTS` ceiling), and `Executor::select_body_paged` picks one from the LIVE context
/// length. The host cannot know which: `DecodeRung::swept` is filled by `codegen` from the CEILING body
/// ("`cap` IS this body's swept extent: `bfp` is the CEILING rung"), one value per BATCH WIDTH. So the
/// index table is built for `PAGE_SLOTS / 64` windows while the body that reads it sweeps
/// `active_cap / 64` — at a short prompt, 4 against 1.
///
/// With the window as a MINOR row coordinate that mismatch moved every row but kv head 0's: the host
/// wrote kv head 1's page block at row `4 * mq` and the body read it at row `1 * mq`, so heads above the
/// first gathered another head's — or an entirely UNWRITTEN — page block, underneath a valid mask.
/// `rc=0`, no fence, a plausible first token (the new-token block is ungathered) and then collapse, for
/// EVERY request including row 0.
///
/// ## What is asserted
/// Not "the two counts agree" — they do not and cannot. That the ROW LAW cannot see the count: a table
/// built at the ceiling is a strict SUPERSET of the narrow body's, at identical indices.
#[test]
fn the_row_of_a_coordinate_does_not_depend_on_how_many_windows_the_scratch_holds() {
    use scratchy_subtile::sdsc_abstract::{
        GatherScratch, PAGE_MASK_COLS, POOL_STICK, PagedKvPool, PrefixMaskShape, QueryRowCount,
        SlotCount, SlotWindow, gather_entries_per_page, gather_index_table,
    };
    let pool = PagedKvPool::new(8, 64);
    let mq = QueryRowCount::of_mq(2);
    // THE HOST's scratch — the rung's CEILING sweep, which is what `DecodeRung::swept` carries.
    let ceiling = GatherScratch::of_fold_pass(
        pool,
        SlotWindow::count_in(SlotCount::new(PagedKvPool::PAGE_SLOTS as u32)),
        mq,
    )
    .expect("the ceiling sweep admits the flat copy");
    // THE LAUNCHED BODY's scratch, at each interior ladder rung a short context selects.
    for cap in [64u32, 128, PagedKvPool::PAGE_SLOTS as u32] {
        let body = GatherScratch::of_fold_pass(pool, SlotWindow::count_in(SlotCount::new(cap)), mq)
            .expect("an interior rung admits the flat copy");
        assert!(
            body.windows() <= ceiling.windows(),
            "an interior rung cannot sweep more windows than the ceiling"
        );
        // ⭐ THE PROPERTY, STATED DIRECTLY: the two scratches disagree about the window COUNT and agree
        // about the ROW LAW, because the law has no field for the count. Everything below is that one
        // fact spelled out per coordinate.
        assert_eq!(
            body.row_law(),
            ceiling.row_law(),
            "the emitter's scratch and the host's must BE the same row law; they differ only in the \
             window EXTENT ({} vs {})",
            body.windows(),
            ceiling.windows()
        );
        for kvh in 0..body.nkvh() {
            for w in 0..body.windows() {
                for r in 0..body.mq() {
                    assert_eq!(
                        body.row_of(kvh, w, r),
                        ceiling.row_of(kvh, w, r),
                        "active_cap {cap}: the body reads (kv head {kvh}, window {w}, request {r}) \
                         at a DIFFERENT row than the host staged it at — that row holds another \
                         head's page block, or none at all, and the mask over it is valid"
                    );
                }
            }
        }
        // ⭐ AND THE TABLE ITSELF, read the way the body's ops read it. The host builds it once, at the
        // ceiling; every entry the narrow body gathers must be the block that body's coordinate names.
        let tables = vec![vec![17i64, 3], vec![4i64, 9]];
        let per_page = gather_entries_per_page(10 * pool.stick_block_bytes(), pool)
            .expect("a page that is a whole number of stick blocks");
        let pass_stride = PrefixMaskShape::<POOL_STICK, PAGE_MASK_COLS>::new(
            32,
            scratchy_subtile::sdsc_abstract::RungWidth::of_baked_rows(2).expect("a baked rung"),
        )
        .expect("a shape")
        .pass_stride();
        let t = gather_index_table(ceiling, &tables, 2, per_page, pass_stride)
            .expect("the host's table, built at the ceiling");
        for p in 0..2usize {
            for r in 0..body.mq() {
                let phys = tables[r as usize][p];
                for kvh in 0..body.nkvh() {
                    for w in 0..body.windows() {
                        let i = body.row_of(kvh, w, r).expect("in range") as usize;
                        let want = phys * per_page.get()
                            + body.block_in_page(kvh, w).expect("in range") as i64;
                        assert_eq!(
                            t[p * pass_stride.get() + i].as_i32() as i64,
                            want,
                            "active_cap {cap}, pass {p}: the body's (kv head {kvh}, window {w}, \
                             request {r}) gathers the wrong page block"
                        );
                    }
                }
            }
        }
    }
}
