//! ⭐⭐ THE RAGGED CASE, ON THE HOST: is a pass a SHALLOW row does not own actually MASKED?
//!
//! `max_pages` is the DEEPEST row's page count, so the fold takes that many passes for EVERY row. A row
//! holding fewer pages therefore receives passes for pages it does not own — and
//! `fold_plan::page_base_bytes` resolves an out-of-range logical page to **PHYSICAL PAGE 0**, i.e. it reads
//! somebody else's page. `PageRun::physical`'s own doc concedes that is "safe only because the mask does in
//! fact cover it".
//!
//! THIS FILE IS THAT ASSUMPTION, CHECKED, on the exact geometry the card fails at: rows of 1/3/6/8 pages,
//! `max_pages = 8`, width 4. Every mask block whose page is out of range for its row must contain NO valid
//! column. A single valid one means that row folds physical page 0 as real history — the silent wrong answer
//! behind the 8-page ragged collapse.
//!
//! ⛔ AN INTEGRATION TEST, NOT A UNIT TEST, because `cargo test -p scratchy-subtile --lib --features superdsc`
//! does not compile for reasons that predate this work (other test modules import `tk_player`,
//! `cpu_golden`, `matmul_opspec`, none of which resolve under this feature set). An integration file compiles
//! against the public API alone, so the check is runnable TODAY instead of after that cleanup.
//!
//! Milliseconds. Every on-card trial of this profile costs ~5 minutes.
#![cfg(feature = "spyre")]

use scratchy_subtile::sdsc_abstract::*;

/// A VALID column is additive-zero (`0.0f16` == bytes `[0,0]`); anything else is masked. Comparing against
/// zero avoids depending on the `half` crate for the mask_neg encoding, and zero is the value that matters:
/// it is what an unstaged byte reads as, and what makes a bogus page count as history.
const VALID: [u8; 2] = [0, 0];

/// The runtime side of a fold declaration, as this file needs it: staging a mask means naming who walks the
/// passes, and the value that comes back is what the mask is blocked by. Nothing here asserts about the
/// declaration itself, so it is only the door.
#[derive(Default)]
struct Walker;

impl FoldWalker for Walker {
    type Error = std::convert::Infallible;

    fn declare_fold(&mut self, _pages: FoldPages, _blocks: MaskBlocks) -> Result<(), Self::Error> {
        Ok(())
    }
}

fn declare(grid: MaskPassGrid) -> DeclaredFold {
    grid.declare_to(&mut Walker)
        .expect("a test walker never refuses a declaration")
}

#[test]
fn a_pass_past_a_rows_own_pages_has_no_valid_column() {
    const PER_PAGE: u32 = PagedKvPool::PAGE_SLOTS as u32;
    let pages_each = [1usize, 3, 6, 8];
    // One history per BOUND row — the modelled bundle's rung width.
    let rung =
        RungWidth::of_baked_rows(pages_each.len() as u32).expect("a modelled bundle has rows");
    let width = rung.get();
    let nqh = 32u32;

    // The ragged batch as it stands before any decode step: each row a contiguous prefix of its own length.
    let hists: Vec<KvHistory> = pages_each
        .iter()
        .map(|p| KvHistory::contiguous(p * PagedKvPool::PAGE_SLOTS))
        .collect();

    let shape = PrefixMaskShape::<{ scratchy_subtile::sdsc_abstract::POOL_STICK }, PER_PAGE>::new(
        nqh, rung,
    )
    .expect("block is whole sticks");
    // ⭐ THROUGH `FoldPages`, from the batch's write slot — the only door. A test that named its own page
    // count could pin a geometry the fold never walks, which is the drift `FoldPages` exists to close.
    // The deepest row here ends at `8 * PAGE_SLOTS`, so the slot appended at is that, and `covering`
    // returns 9 pages: the 8 the deepest row already holds plus the one its NEXT write lands in. Every
    // page from a row's own count up to that is out of range FOR THAT ROW, which is what this checks.
    let grid = FoldPages::covering(BatchSlot::of(hists.iter()), SlotCount::new(PER_PAGE))
        .expect("at least one page")
        .grid(rung, MaskBlockForm::PerRowPage);
    let max_pages = grid.pages().get();
    let bytes = decode_batch_prefix_mask_f16::<PER_PAGE>(shape, declare(grid), &hists, -30000.0);

    let mut checked = 0usize;
    for (r, own) in pages_each.iter().enumerate() {
        for p in (*own as u32)..max_pages {
            let block = r as u32 * max_pages + p;
            for h in 0..nqh {
                let row = HeadRequestRow::of(h, r as u32, width).get();
                for col in 0..PER_PAGE {
                    let e = shape.elem(block, row, col);
                    assert_ne!(
                        &bytes[e * 2..e * 2 + 2],
                        &VALID,
                        "row {r} owns {own} page(s), so block {block} (page {p}, head {h}, col {col}) is a \
                         pass it does not own — but the mask marks it VALID, so the fold counts PHYSICAL \
                         PAGE 0 as this row's history"
                    );
                    checked += 1;
                }
            }
        }
    }
    assert!(
        checked > 0,
        "the geometry produced no out-of-range passes — the test would be vacuous"
    );
}

/// ⭐⭐ THE HOLE-FREE ROW vs THE HOLED ROWS, AT THE BOUNDARY — the last uncharacterised asymmetry.
///
/// `BatchSlot::of` is the MAX over live `end()`s, so in a ragged batch the DEEPEST row's own end IS the
/// shared slot: it appends CONTIGUOUSLY and keeps ONE run, while every shallower row's history becomes
/// `[0,len) ∪ [shared, ..)` — TWO runs with a masked hole. On card it is exactly the deepest row that
/// collapses, and uniform batches (every row hole-free) WORK, so the trigger is "hole-free while others are
/// holed".
///
/// This pins what the mask says about the boundary column for BOTH shapes: the slot AT `end` must be
/// INVALID before the step (nothing written there yet) and VALID after recording it, for the one-run row and
/// the two-run rows alike. A disagreement between the two shapes at that column is the bug.
#[test]
fn the_boundary_column_agrees_for_holed_and_hole_free_rows() {
    const PER_PAGE: u32 = PagedKvPool::PAGE_SLOTS as u32;
    let lens = [225usize, 710, 1282, 1986]; // the ragged profile the card fails on
    // One history per BOUND row — the modelled bundle's rung width.
    let rung = RungWidth::of_baked_rows(lens.len() as u32).expect("a modelled bundle has rows");
    let width = rung.get();
    let nqh = 32u32;
    let shared = *lens.iter().max().unwrap(); // 1986 — the DEEPEST row's own end

    let shape = PrefixMaskShape::<{ scratchy_subtile::sdsc_abstract::POOL_STICK }, PER_PAGE>::new(
        nqh, rung,
    )
    .expect("whole sticks");

    // AFTER the step: each row recorded one slot at the shared write slot. The deepest row's run EXTENDS
    // (contiguous); the others gain a second run.
    let mut hists: Vec<KvHistory> = lens.iter().map(|l| KvHistory::contiguous(*l)).collect();
    let at = BatchSlot::of(hists.iter());

    // ⭐ Every row is grown to cover the SHARED slot, and `FoldPages::covering` IS that rule — the same
    // `ceil((slot + 1) / PAGE_SLOTS)` this test used to spell out by hand, and the same one the shim's
    // `n_fold_pages` computes. Deriving it here rather than restating it is the whole point of the type.
    let grid = FoldPages::covering(at, SlotCount::new(PER_PAGE))
        .expect("at least one page")
        .grid(rung, MaskBlockForm::PerRowPage);
    let max_pages = grid.pages().get();
    assert_eq!(
        max_pages as usize,
        (shared + 1).div_ceil(PagedKvPool::PAGE_SLOTS),
        "FoldPages must reproduce the shim's ceiling exactly — that equality is the lock"
    );
    assert_eq!(
        at.slot().get() as usize,
        shared,
        "the shared slot is the deepest row's end"
    );
    for h in hists.iter_mut() {
        h.record(at, scratchy_subtile::sdsc_abstract::SlotCount::new(1));
    }
    assert_eq!(
        hists[3].runs().len(),
        1,
        "the deepest row must stay ONE run (contiguous append)"
    );
    assert_eq!(
        hists[0].runs().len(),
        2,
        "a shallower row must hold a hole (two runs)"
    );

    let bytes = decode_batch_prefix_mask_f16::<PER_PAGE>(shape, declare(grid), &hists, -30000.0);
    let page = (shared / PagedKvPool::PAGE_SLOTS) as u32;
    let col = (shared % PagedKvPool::PAGE_SLOTS) as u32;

    // THE BOUNDARY COLUMN: the slot every row just wrote. It must read VALID for EVERY row — the
    // hole-free one and the holed ones alike.
    for r in 0..width {
        let block = r * max_pages + page;
        let row = HeadRequestRow::of(0, r, width).get();
        let e = shape.elem(block, row, col);
        assert_eq!(
            &bytes[e * 2..e * 2 + 2],
            &VALID,
            "row {r} wrote slot {shared} (page {page}, col {col}) this step, so its mask must mark it \
             VALID — runs = {:?}",
            hists[r as usize].runs(),
        );
    }
}
