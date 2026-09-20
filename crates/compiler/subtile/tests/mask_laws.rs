// SPDX-License-Identifier: Apache-2.0
#![cfg(feature = "spyre")]
//! THE NEW-BLOCK MASK LAWS, and the fact that they are not interchangeable.
//!
//! Lives here rather than beside the functions as an integration test, which is also what keeps it
//! honest about the PUBLIC surface: every law below is reached through `sdsc_abstract`'s exported
//! types, so a lock cannot quietly come to depend on something the crate does not expose.

const PER_PAGE: u32 = scratchy_subtile::sdsc_abstract::PagedKvPool::PAGE_SLOTS as u32;

use scratchy_subtile::sdsc_abstract::{
    BatchSlot, DeclaredFold, FoldPages, FoldWalker, HeadRequestRow, KvHistory, KvSlot,
    MaskBlockForm, MaskBlocks, MaskPassGrid, PrefixMaskShape, RungWidth, SlotCount, SlotRun,
    decode_batch_causal_col_valid, decode_batch_prefix_mask_f16, decode_prefix_col_valid,
    prefill_causal_col_valid,
};

/// THE RUNTIME SIDE OF A FOLD DECLARATION, as a test can hold it: it KEEPS what it was told, because the
/// law is that the blocks the mask is staged with and the passes the walker steps are one number, and a
/// sink that discarded the declaration could not state it.
///
/// Infallible, so nothing here has to model a device that refuses — `declare_to` carries the walker's own
/// error type precisely so a test walker and an FFI session can each keep theirs.
#[derive(Default)]
struct Walker {
    pages: Option<FoldPages>,
    blocks: Option<MaskBlocks>,
}

impl FoldWalker for Walker {
    type Error = std::convert::Infallible;

    fn declare_fold(&mut self, pages: FoldPages, blocks: MaskBlocks) -> Result<(), Self::Error> {
        self.pages = Some(pages);
        self.blocks = Some(blocks);
        Ok(())
    }
}

impl Walker {
    /// The pages the walker will divide a pass index by — `fold_plan::fold_pass`'s `rep / pages`.
    fn pages(&self) -> u32 {
        self.pages.expect("a fold was declared").get().get()
    }
    /// The passes it will step, which is the block count it was told the mask has.
    fn blocks(&self) -> u32 {
        self.blocks.expect("a fold was declared").get()
    }
}

/// The prefix mask exactly as it was staged before it was narrowed at the source: build `f32`, then let
/// the bind loop's one narrowing (`half::f16::from_f32(v).to_le_bytes()`) turn it into device bytes.
///
/// Transcribed from the f32 fill this replaced, and kept as the SPEC. The point of the f16 staging is
/// that it is not a new mask, only the same bytes without the round trip — a claim that means nothing
/// unless the old path is still here to be compared against.
fn reference_f32_then_narrow<const COLS: u32>(
    nqh: usize,
    cap: usize,
    hists: &[KvHistory],
    mask_neg: f32,
) -> Vec<u8> {
    let page_slots = COLS as usize;
    let _sh = shape::<COLS>(nqh, hists.len());
    let mq = hists.len().max(1);
    let per_page = page_slots.max(1);
    let pages = cap.div_ceil(per_page).max(1);
    let rows = nqh * mq;
    // ⭐ PASS-MAJOR: one block per (ROW, page), block `r * pages + p`, and ONLY row `r` is written in it.
    //
    // The fold takes a pass per (row, page) — `fold_plan::fold_pass` maps `rep = r * pages + p` — and steps
    // the mask one block per pass, so a pass must find a block describing ITS row alone. This reference was
    // page-major (`p` outermost, every row written into every block), which agrees with the pass order only
    // at one row; at two rows pass 1 wanted row 0's page 1 and got a block holding both rows' page 1.
    let mut m = vec![mask_neg; pages * mq * rows * per_page];
    for (r, h_r) in hists.iter().enumerate() {
        for p in 0..pages {
            let block = (r * pages + p) * rows * per_page;
            for h in 0..nqh {
                let row = h * mq + r;
                for col in 0..per_page {
                    if h_r.contains(p * per_page + col) {
                        m[block + (col / 64) * (rows * 64) + row * 64 + (col % 64)] = 0.0;
                    }
                }
            }
        }
    }
    m.iter()
        .flat_map(|&v| half::f16::from_f32(v).to_le_bytes())
        .collect()
}

/// The shape both the emitter and the worker build — mirrored here so the tests exercise the same law
/// rather than a third derivation of it.
/// GENERIC OVER THE BLOCK WIDTH, because the width is a const generic now and these tests deliberately
/// cover more than one: a 64-column block is small enough to reason about by hand, a 256-column block is
/// what ships. A runtime `page_slots` cannot feed a const generic, and that is the point — the shape
/// cannot be built with a width nobody named.
fn shape<const COLS: u32>(
    nqh: usize,
    mq: usize,
) -> PrefixMaskShape<{ scratchy_subtile::sdsc_abstract::POOL_STICK }, COLS> {
    // `mq` here is the width the modelled bundle was baked at — the histories these tests stage are
    // always one per bound row, padding included, exactly as the worker's staging supplies them.
    let width = scratchy_subtile::sdsc_abstract::RungWidth::of_baked_rows(mq as u32)
        .expect("a modelled bundle has rows");
    PrefixMaskShape::<{ scratchy_subtile::sdsc_abstract::POOL_STICK }, COLS>::new(nqh as u32, width)
        .expect("nonzero heads")
}

/// The PASS GRID — pages per row and rows — which is the staging's to know, not the shape's. The block
/// count derives from it (`MaskBlocks::of`), so a test cannot stage a count the fold would not walk.
/// ⭐ THROUGH `FoldPages`, THE ONLY DOOR. There is no `MaskPassGrid::new` any more: the page count must
/// come from a [`BatchSlot`], because that is the quantity the shim's fold divides by. A test that could
/// name its own page count could pin a layout the fold never walks — which is how the mask and the fold
/// came to be blocked by two different numbers in the first place.
///
/// These tests parameterise coverage by `cap`, so the slot named here is `cap - 1`: the deepest slot such
/// a window holds. `covering` then returns `ceil(cap / COLS)` — exactly the `(cap / COLS).max(1)` this
/// helper used to compute by hand, so no assertion in this file changes meaning.
fn grid<const COLS: u32>(cap: usize, rows: usize) -> MaskPassGrid {
    FoldPages::covering(
        BatchSlot::solo(&KvHistory::contiguous(cap.max(1) - 1)),
        SlotCount::new(COLS),
    )
    .expect("at least one page")
    // The grid's row axis is the RUNG width — these tests hand it the same row count the shape was
    // built with, one history per bound row.
    .grid(
        RungWidth::of_baked_rows(rows.max(1) as u32).expect("at least one row"),
        // ⛔ `PerRowPage` — every assertion in this file is about the UNCOLLAPSED fold's layout, where a
        // pass reads ONE request's page and its block marks only that request's rows valid. The
        // collapsed form's `PerPage` blocks are pinned separately; handing that form here would make
        // these row-major block-index checks vacuous rather than failing.
        MaskBlockForm::PerRowPage,
    )
}

/// The grid AS THE STAGING CAN RECEIVE IT — declared to a walker, which is the only way to a
/// `DeclaredFold`. A test that wants to stage a mask has to name who will walk the passes, exactly as a
/// launch does; the walker here is discarded because these cases assert about the BYTES, and the ones
/// that assert about the declaration keep their own.
fn declared<const COLS: u32>(cap: usize, rows: usize) -> DeclaredFold {
    grid::<COLS>(cap, rows)
        .declare_to(&mut Walker::default())
        .expect("a test walker never refuses a declaration")
}

/// ⭐⭐⭐⭐⭐ THE COLLAPSED FOLD'S MASK: ONE BLOCK PER **PAGE**, AND EVERY ROW'S OWN VALIDITY INSIDE IT.
///
/// ⛔ WHY THIS HAS TO BE ITS OWN LOCK. `MaskBlockForm` is read by THREE consumers — the block COUNT
/// (`MaskBlocks::of`), the block INDEX the fill composes, and `fold_plan::reps` — and every pairwise
/// agreement with the third one wrong is silent, because an additive mask byte the host never wrote reads
/// as ZERO, which means VALID. Under `PerPage` a fold pass serves the WHOLE batch (the score kernel's `y`
/// axis walks the requests over the gathered scratch), so row `h*mq + r` of pass `p` must find request
/// `r`'s own validity in block `p` — not in block `r*pages + p`, which is where `PerRowPage` puts it and
/// which under this form is some other page's block entirely.
///
/// So this asserts BOTH halves against histories that can tell them apart: the count is `pages` (not
/// `pages * rows`), and for every (request, page) the valid columns in block `p` are exactly that
/// request's own history — checked through `decode_prefix_col_valid`, the same law the broadcast path uses.
#[test]
fn the_collapsed_form_blocks_by_page_and_holds_every_rows_own_history() {
    const COLS: u32 = 256;
    let nqh = 4usize;
    // RAGGED, and with a hole: three prompts of different depths then one shared decode step. A uniform
    // batch cannot tell the two forms apart for the rows it happens to line up.
    let hs = batch(&[300usize, 40, 570], 1);
    let sh = shape::<COLS>(nqh, hs.len());
    let deepest = hs.iter().map(|h| h.end().get()).max().unwrap_or(0) as usize;
    let rung = RungWidth::of_baked_rows(hs.len() as u32).expect("at least one row");
    let per_row_page = FoldPages::covering(
        BatchSlot::solo(&KvHistory::contiguous(deepest)),
        SlotCount::new(COLS),
    )
    .expect("at least one page")
    .grid(rung, MaskBlockForm::PerRowPage);
    let per_page = FoldPages::covering(
        BatchSlot::solo(&KvHistory::contiguous(deepest)),
        SlotCount::new(COLS),
    )
    .expect("at least one page")
    .grid(rung, MaskBlockForm::PerPage);
    let pages = per_page.pages().get();
    assert!(pages > 1, "a single page cannot distinguish the two forms");

    // ⭐ THE COUNT IS `mq`x SMALLER, which is the same factor as the launch saving — the mask's shape and
    // the fold's pass count are one decision.
    assert_eq!(MaskBlocks::of(per_page).get(), pages);
    assert_eq!(
        MaskBlocks::of(per_row_page).get(),
        pages * hs.len() as u32,
        "and the uncollapsed form still costs a block per (request, page)"
    );
    assert_eq!(
        per_page.passes(),
        pages,
        "`passes` must agree with the block count — they are the same number by definition"
    );

    let declared = per_page
        .declare_to(&mut Walker::default())
        .expect("a test walker never refuses a declaration");
    let bytes = decode_batch_prefix_mask_f16::<COLS>(sh, declared, &hs, f32::NEG_INFINITY);
    let zero = half::f16::from_f32(0.0).to_le_bytes();
    assert_eq!(
        bytes.len(),
        sh.elems(MaskBlocks::of(per_page)) * 2,
        "the buffer is exactly the collapsed block count"
    );
    for (r, h) in hs.iter().enumerate() {
        for p in 0..pages {
            for head in 0..nqh as u32 {
                let row = HeadRequestRow::of(head, r as u32, sh.mq()).get();
                for c in 0..COLS {
                    let slot = (p * COLS + c) as usize;
                    let e = sh.elem(p, row, c);
                    let got = bytes[e * 2..e * 2 + 2] == zero;
                    assert_eq!(
                        got,
                        h.contains(slot),
                        "block {p} row {row} (head {head}, request {r}) column {c} = slot {slot}: \
                         a PerPage block must hold EVERY row's own history, so this cell is valid \
                         exactly when request {r} holds that slot"
                    );
                }
            }
        }
    }
}

/// The batch shapes the mask has to survive: a list of prompt lengths, then `steps` decode tokens all
/// appended at the batch's ONE shared slot, exactly as the worker will do it.
///
/// Ragged lengths plus a shared slot is what puts a HOLE in every request but the longest — the shape a
/// length cannot describe, and the reason these tests take histories at all.
fn batch(lens: &[usize], steps: usize) -> Vec<KvHistory> {
    let mut hs: Vec<KvHistory> = lens.iter().map(|&l| KvHistory::contiguous(l)).collect();
    for _ in 0..steps {
        let slot = BatchSlot::of(hs.iter());
        for h in &mut hs {
            h.record(slot, scratchy_subtile::sdsc_abstract::SlotCount::new(1));
        }
    }
    hs
}

/// THE f16 STAGING IS THE SAME BYTES, not a new mask.
///
/// The staged mask is the largest thing crossing the bus in a batched decode step and it is quadratic
/// in the request count, so it is staged in the device's format directly instead of being built as f32
/// and narrowed per element. That is only safe because both values it can hold are exactly
/// representable; if either ever stopped being (a different sentinel, a scaled mask), the narrowing
/// would round and this is where it would be caught.
///
/// RAGGED starts, because equal lengths is the one case that cannot distinguish a per-request mask from
/// a broadcast one — the shape this whole buffer exists for. And GAPPED, because a shared write slot is
/// the reason the buffer stopped being describable by a length.
#[test]
fn staged_f16_prefix_mask_is_byte_identical_to_the_f32_path() {
    // A page as granite bakes it, and a 64-column block that is small enough to check by hand. Two
    // instantiations because the width is a const generic — a runtime loop cannot supply one, which is
    // exactly what stops a shape being built at a width nobody named.
    identical::<256>(&[
        (32, 256, vec![64], 0),
        (32, 256, vec![64, 64], 0),
        (32, 256, vec![130, 129, 65, 3], 0),
        (32, 256, vec![130, 129, 65, 3], 1),
        (32, 256, vec![130, 129, 65, 3], 8),
        (32, 256, vec![64, 64], 4),
    ]);
    identical::<64>(&[
        (4, 128, vec![100, 7, 0, 64], 0),
        (2, 64, vec![63, 1], 0),
        (4, 128, vec![100, 7, 0, 64], 5),
        (2, 64, vec![63, 1], 1),
    ]);
}

fn identical<const COLS: u32>(cases: &[(usize, usize, Vec<usize>, usize)]) {
    let mask_neg = -(half::f16::MAX.to_f32() / 2.0);
    for (nqh, cap, lens, steps) in cases.iter().cloned() {
        let hists = batch(&lens, steps);
        assert_eq!(
            decode_batch_prefix_mask_f16(
                shape::<COLS>(nqh, hists.len()),
                declared::<COLS>(cap, hists.len()),
                &hists,
                mask_neg
            ),
            reference_f32_then_narrow::<COLS>(nqh, cap, &hists, mask_neg),
            "nqh={nqh} COLS={COLS} cap={cap} lens={lens:?} steps={steps}"
        );
    }
}

/// ⛔ THE MASK A LENGTH WOULD HAVE PRODUCED IS WRONG, not merely coarser — pinned here so the gapped
/// cases above cannot be dismissed as a shape nobody builds.
///
/// One shared write slot per launch is what turns 8 cache-write launches into 1 (~26 ms of a bs=8 step
/// at the ~93 µs launch floor). The price is that a shorter request's keys are `[0, len) ∪ [shared, …)`,
/// and `col < end` calls the hole between them VALID — so that request attends whatever the pool holds
/// at slots it never wrote, which is another request's keys. Fluent, wrong, and invisible downstream.
#[test]
fn a_length_would_mark_the_shared_slot_hole_valid() {
    const COLS: u32 = 64;
    let (nqh, page_slots, cap) = (2usize, 64usize, 64usize);
    let mask_neg = -(half::f16::MAX.to_f32() / 2.0);
    let hists = batch(&[40, 9], 1); // request 1: [0,9) then {40}
    let short = &hists[1];
    assert_eq!(
        short.runs(),
        &[
            SlotRun::new(KvSlot::ZERO, SlotCount::new(9)),
            SlotRun::new(KvSlot::new(40), SlotCount::ONE),
        ],
        "a hole a length cannot describe"
    );

    // Every column of the hole is one the length law admits and the history denies.
    let mut hole = 0;
    for col in 9..40usize {
        assert!(
            decode_prefix_col_valid(col, short.end().get() as usize),
            "col {col}"
        );
        assert!(!short.contains(col), "col {col}");
        hole += 1;
    }
    assert_eq!(hole, 31, "31 columns of someone else's keys");

    // And the staged bytes follow the history, not the length: request 1's rows are masked there.
    let m = decode_batch_prefix_mask_f16(
        shape::<COLS>(nqh, hists.len()),
        declared::<COLS>(cap, hists.len()),
        &hists,
        mask_neg,
    );
    let zero = half::f16::from_f32(0.0).to_le_bytes();
    let (mq, rows) = (hists.len(), nqh * hists.len());
    // ⭐ THE BLOCK IS THE PASS: `(row, page)`, index `r * pages + p`. Request 1's page 0 is block
    // `1 * pages + 0` — a pass carries ONE row, so its own block is the only place its history is stated.
    let pages_here = cap.div_ceil(page_slots).max(1);
    let block = pages_here * (nqh * hists.len()) * page_slots;
    for h in 0..nqh {
        let row = h * mq + 1;
        for col in 9..40usize {
            let e = block + (col / 64) * (rows * 64) + row * 64 + (col % 64);
            assert_ne!(
                m[e * 2..e * 2 + 2],
                zero,
                "row {row} col {col} must be masked"
            );
        }
        for col in [0usize, 8, 40] {
            let e = block + (col / 64) * (rows * 64) + row * 64 + (col % 64);
            assert_eq!(
                m[e * 2..e * 2 + 2],
                zero,
                "row {row} col {col} is its own key"
            );
        }
    }
}

/// The mask must actually SAY what the fold contract needs, not merely agree with an old copy of
/// itself: a pass is ONE PAGE and it carries every request, so EVERY row of the block is valid exactly
/// on the slots its own request holds.
///
/// That is the whole content of the fold collapse. A pass used to read one request's page and mask off
/// every row that was not its — `pages × requests` launches to keep one request's worth of each. Now
/// the score and value kernels step a request per unit of their batch axis, so one pass per page serves
/// the batch and the mask has to describe the batch. Pinned independently of the reference above
/// because both could be wrong the same way, and the failure mode is a request answering out of
/// another's history: fluent, and invisible to anything downstream.
#[test]
fn a_fold_pass_carries_every_request_masked_to_its_own_history() {
    let mask_neg = -(half::f16::MAX.to_f32() / 2.0);
    const COLS: u32 = 64;
    let (nqh, page_slots, cap) = (4usize, 64usize, 128usize);
    // Ragged AND decoded, so three of the four requests carry a hole: the contract has to hold on the
    // rows of a request whose own history is not an interval.
    let hists = batch(&[70usize, 5, 0, 64], 3);
    let mq = hists.len();
    let per_page = page_slots;
    let pages = cap.div_ceil(per_page);
    // A block is ONE request's `nqh` rows now: the request is the block's identity, not a coordinate
    // inside it, so every row of block (r, p) belongs to request r and the only question left is
    // whether the column is inside that request's own history.
    let rows = nqh * mq;
    let m = decode_batch_prefix_mask_f16(
        shape::<COLS>(nqh, hists.len()),
        declared::<COLS>(cap, hists.len()),
        &hists,
        mask_neg,
    );
    let zero = half::f16::from_f32(0.0).to_le_bytes();
    // ONE BLOCK PER PASS: `pages * mq` of them, each `(nqh*mq)` rows deep.
    assert_eq!(
        m.len(),
        pages * mq * rows * per_page * 2,
        "(pages x rows) blocks x (nqh*mq) x page, f16"
    );

    for r in 0..mq {
        for p in 0..pages {
            let block = (r * pages + p) * rows * per_page;
            for row in 0..rows {
                for col in 0..per_page {
                    let e = block + (col / 64) * (rows * 64) + row * 64 + (col % 64);
                    let valid = m[e * 2..e * 2 + 2] == zero;
                    // ⭐ A PASS CARRIES EXACTLY ONE ROW. Block `(r, p)` states row `r`'s history and masks
                    // every other row OFF — that is what makes pass `r` contribute only request `r`, and
                    // it is where a sequence's identity lives now that no address carries one.
                    let owner = row % mq;
                    let want = owner == r && hists[owner].contains(p * per_page + col);
                    assert_eq!(
                        valid,
                        want,
                        "pass (row {r}, page {p}) row {row} (owner {owner}) col {col}: runs={:?}",
                        hists[owner].runs()
                    );
                }
            }
        }
    }
}

/// A prefill chunk's rows are consecutive positions of ONE prompt, so row `r` attends every column
/// up to and including itself. A decode batch's rows are INDEPENDENT REQUESTS, so row `i` attends
/// only its own column.
///
/// Reusing the prefill law for a batch lets request 1 read request 0's token: fluent output built
/// partly from someone else's conversation, with no crash, no shape error, and nothing downstream
/// that would notice. Every off-diagonal lower cell below is exactly where that leaks.
#[test]
fn decode_batch_mask_is_the_diagonal_not_the_triangle() {
    for row in 0..8usize {
        for col in 0..8usize {
            assert_eq!(
                decode_batch_causal_col_valid(col, row),
                col == row,
                "a decode batch row must attend its own column and no other ({row},{col})"
            );
            if col < row {
                assert!(
                    prefill_causal_col_valid(col, row) && !decode_batch_causal_col_valid(col, row),
                    "col {col} < row {row}: prefill attends it and a batch must not"
                );
            }
        }
    }
}

/// At ONE row the two laws coincide — which is why unbatched decode never needed the distinction,
/// and why introducing it cannot move any bundle that exists today.
///
/// Coinciding is NOT the same as being unnecessary, and the difference matters here. The new-token
/// block is `mq_pad` (64) columns wide even for a single token, and columns 1.. are the padding rows.
/// Per ATTN_ZERO_TID's own comment the zeros land PAST the packed data, so the Kt restickify reads
/// the NEXT HEAD's real K/V in those columns. The mask is the only thing stopping one token attending
/// 63 columns of another head's keys, so the property to pin at one row is not "the laws agree" — it
/// is "everything but column 0 is masked", under BOTH.
#[test]
fn a_single_row_attends_only_itself_under_either_law() {
    const MQ_PAD: usize = 64;
    for col in 0..MQ_PAD {
        let want = col == 0;
        assert_eq!(
            prefill_causal_col_valid(col, 0),
            want,
            "row 0 col {col}: the padding columns hold another head's K/V, not zeros"
        );
        assert_eq!(
            decode_batch_causal_col_valid(col, 0),
            want,
            "row 0 col {col}"
        );
    }
}

/// ⭐⭐⭐⭐⭐ A PADDING ROW'S WHOLE FORWARD IS LIVE 0'S — ITS NEW-BLOCK MASK ROW **AND** ITS PREFIX-MASK
/// ROW, BYTE FOR BYTE — BECAUSE ITS CACHE WRITE RACES LIVE 0'S AT ONE POOL CELL.
///
/// ⛔ THE DEFECT. A launch slot past the live count is padding. It borrows live 0's PAGE MAP (every slot's
/// table must exist), the on-card paged cachewr address has NO row term inside a page — a pool cell is
/// (plane, kv-head, slot-in-page, feature) — and the shim elides the pipeline barrier between consecutive
/// slot writes (`pipeline_barrier = !next_is_slot_write`) on a disjointness premise padding breaks. So the
/// padding row's one cache write and live 0's land on the SAME cell, UNORDERED, and the only sound
/// arrangement is that they carry IDENTICAL BYTES.
///
/// Those bytes are the padding row's WHOLE FORWARD, not its input, so every input has to be live 0's. The
/// token and the rotation were mirrored and the two MASKS were not: the padding row took the prompt-chunk
/// clamp (column `real - 1`, the LAST live request's new token, where live 0's column is 0) and a
/// nothing-valid prefix row (where live 0 has its real history). Two different key sets ⇒ a different
/// softmax ⇒ a different output ⇒ different K/V into live 0's cell, at every layer from 1 on. Fluent, no
/// fault, and only from bs >= 3: a full rung has no padding row at all.
///
/// ⛔ WHY "NOTHING VALID" WAS THE WRONG SAFETY. It is right that a padding row must not CLAIM history it
/// was not assigned. It is wrong as a way to get there, because it makes the row a DIFFERENT COMPUTATION
/// from live 0, which is the one thing the racing write forbids. Padding still owns nothing — no request,
/// no pages, no place in the live list, and its logits are never read. It is a second evaluation of live 0
/// whose result is discarded.
///
/// TWO RAGGED SHAPES ON THE 4-ROW RUNG, because one padding row cannot distinguish "replicates live 0"
/// from "replicates whatever row 0 happens to be": at `real = 2` there are TWO padding rows and they must
/// both be live 0, not live 0 and live 1.
#[test]
fn a_padding_row_replicates_live_0_in_both_masks() {
    for real in [3usize, 2] {
        padding_row_replicates_live_0(real);
    }
}

fn padding_row_replicates_live_0(real: usize) {
    use scratchy_subtile::sdsc_abstract::{BatchRow, SeqPos, SlotMap};
    const COLS: u32 = 64;
    let nqh = 4usize;
    let width = RungWidth::of_baked_rows(4).expect("a 4-row rung"); // the RUNG's baked width
    let mq = width.count();
    let cap = 128usize;
    let mask_neg = -(half::f16::MAX.to_f32() / 2.0);

    // RAGGED live rows that have all decoded once, so each carries a hole at the shared write slot — the
    // shape where a history and a length stop agreeing. Truncated to `real`, exactly as a step with fewer
    // live requests than the rung is wide supplies them.
    let hists = batch(&[25usize, 90, 60][..real], 2);
    // Live tokens that are all DIFFERENT, so "the pad row took live 0's token" is distinguishable from
    // "it took the last live token" (the prompt-chunk clamp) at every `real`.
    let toks: Vec<usize> = (0..real).map(|i| 101 + 100 * i).collect();

    let slots = SlotMap::of_live(real, width).expect("live rows fit the rung");
    let live = slots
        .live_rows(|i| {
            Ok::<_, ()>(BatchRow {
                // Positions that differ per row AND differ from every launch position, so a pad row
                // rotated anywhere but live 0's own position is visible.
                rope_pos: SeqPos::new(10 + i as u32),
                hist: hists[i].clone(),
            })
        })
        .expect("the builder never fails here");
    let rows = live
        .launch_rows(&toks, width)
        .expect("a live batch lays into its rung");
    assert_eq!(
        rows.live(),
        real,
        "the fixture must run fewer live rows than the rung is wide"
    );
    assert!(real < mq, "a case with no padding row proves nothing here");

    // ── THE FOUR ASPECTS, FROM THE ONE VALUE ──────────────────────────────────────────────────────
    let live0 = rows.row(0).expect("live 0");
    for pad in real..mq {
        let p = rows.row(pad).expect("a bound slot");
        assert_eq!(p.token(), live0.token(), "pad row {pad}: token");
        assert_eq!(p.rope_pos(), live0.rope_pos(), "pad row {pad}: rotation");
        assert_eq!(
            p.new_block_col().index(),
            live0.new_block_col().index(),
            "pad row {pad}: new-block column — the prompt-chunk clamp would give it {}, the LAST live \
             request's new token",
            real - 1
        );
        assert_eq!(p.hist(), live0.hist(), "pad row {pad}: prefix history");
        // And live 0's own aspects are not degenerate — otherwise every equality above is vacuous.
        assert_eq!(live0.token().get() as usize, toks[0]);
        assert_eq!(live0.new_block_col().index(), 0);
        assert!(
            !live0.hist().runs().is_empty(),
            "live 0 must actually hold slots"
        );
    }
    // The live rows keep their OWN aspects: replication is for padding only.
    for i in 1..real {
        let r = rows.row(i).expect("a live slot");
        assert_eq!(
            r.token().get() as usize,
            toks[i],
            "live row {i} keeps its own token"
        );
        assert_eq!(
            r.new_block_col().index(),
            i,
            "live row {i} attends its own column"
        );
        assert_eq!(r.hist(), &hists[i], "live row {i} keeps its own history");
    }

    // ── THE NEW-BLOCK (CAUSAL) MASK ROW ───────────────────────────────────────────────────────────
    // `[mq, mq_pad]` row-major, exactly as the worker stages it and the emitter reads it.
    let cmask = rows.new_block_mask(mask_neg);
    let mq_pad = mq.div_ceil(64) * 64;
    assert_eq!(cmask.len(), mq * mq_pad, "one row per bound slot");
    for pad in real..mq {
        assert_eq!(
            &cmask[pad * mq_pad..(pad + 1) * mq_pad],
            &cmask[0..mq_pad],
            "pad row {pad}'s causal mask row must be LIVE 0's, byte for byte — under the prompt-chunk \
             clamp it attends column {} instead of column 0, which is a different softmax and therefore \
             different K/V into the cell its write shares with live 0's",
            real - 1
        );
    }
    // Not vacuous: live 0's row really does single out column 0, and a live row past 0 differs from it.
    assert_eq!(cmask[0], 0.0, "live 0 attends its own column");
    assert!(
        cmask[1..mq_pad].iter().all(|&v| v == mask_neg),
        "and no other"
    );
    assert_ne!(
        &cmask[mq_pad..2 * mq_pad],
        &cmask[0..mq_pad],
        "live row 1 is not live 0"
    );

    // ── THE PREFIX-MASK ROW ───────────────────────────────────────────────────────────────────────
    // Blocked per (row, page), so a row's own validity lives in its own blocks: compare pad row `r`'s
    // entries in block `(r, p)` against live 0's in block `(0, p)`, over every head and column.
    let sh = shape::<COLS>(nqh, mq);
    let gr = declared::<COLS>(cap, mq);
    let m = decode_batch_prefix_mask_f16(sh, gr, &rows.mask_histories(), mask_neg);
    let pages = (cap as u32).div_ceil(COLS);
    let zero = half::f16::from_f32(0.0).to_le_bytes();
    let mut valid_seen = 0usize;
    for pad in real..mq {
        for p in 0..pages {
            for h in 0..nqh as u32 {
                let pad_row = HeadRequestRow::of(h, pad as u32, mq as u32).get();
                let live_row = HeadRequestRow::of(h, 0, mq as u32).get();
                for col in 0..COLS {
                    let e = sh.elem(pad as u32 * pages + p, pad_row, col);
                    let l = sh.elem(p, live_row, col);
                    assert_eq!(
                        &m[e * 2..e * 2 + 2],
                        &m[l * 2..l * 2 + 2],
                        "pad row {pad} (page {p}, head {h}, col {col}) must see exactly live 0's key set \
                         — a nothing-valid row is a DIFFERENT softmax over a different set, which is the \
                         same corruption by another route"
                    );
                    if m[l * 2..l * 2 + 2] == zero {
                        valid_seen += 1;
                    }
                }
            }
        }
    }
    assert!(
        valid_seen > 0,
        "live 0 must have valid columns for the comparison to mean anything (real={real})"
    );
}

/// ⭐ A POSITION IS NOT A SLOT, AND THE TYPE SYSTEM NOW SAYS SO.
///
/// This is a compile-time property, so the test's job is to state the LAW and the reason — the enforcement is
/// that `SeqPos` has no conversion to or from `KvSlot`/`BatchSlot`, which no runtime assertion can check.
///
/// ⛔ WHY IT MATTERS ONLY WHEN RAGGED. While every request appends at its own position, a request's token count
/// and its newest slot are THE SAME NUMBER, so confusing them is invisible. Once a batch shares one write slot,
/// a short request's slot count EXCEEDS its token count by the size of the hole — and rotating it at its slot
/// instead of its position reads as that request drifting toward the longest request's topic. Fluent, no crash.
#[test]
fn a_sequence_position_and_a_kv_slot_diverge_exactly_when_a_batch_shares_a_write_slot() {
    use scratchy_subtile::sdsc_abstract::SeqPos;

    // A short request: 225 tokens of its own, then it joins a batch whose shared write slot is 1449.
    let own_tokens = 225u32;
    let shared_write_slot = 1449u32;
    let pos = SeqPos::new(own_tokens);

    // Its history is its own prompt PLUS the shared slot — the union with a hole between.
    let mut hist = KvHistory::contiguous(own_tokens as usize);
    hist.record(
        BatchSlot::of(std::iter::once(&hist)),
        scratchy_subtile::sdsc_abstract::SlotCount::new(1),
    );
    let _ = &hist;

    // The two numbers are now different, which is the whole point: the position advances by one token,
    // the slot jumped past the hole.
    assert_eq!(
        pos.advanced().get(),
        226,
        "a step advances the POSITION by one token"
    );
    assert_ne!(
        pos.advanced().get(),
        shared_write_slot + 1,
        "and it is NOT the write slot + 1 — that is the number the rotary op must not use, and the gap is the \
         masked hole a shared write slot creates"
    );

    // `SeqPos::ZERO` is the only position that needs no derivation, and `advanced()` is the only arithmetic —
    // there is deliberately no `+`, because "position plus one" and "slot plus one" are the same expression on
    // two quantities that must not be interchanged.
    assert_eq!(SeqPos::ZERO.advanced(), SeqPos::new(1));
}

/// ⛔⛔⛔ DOES EVERY SOFTMAX ROW OF EVERY BLOCK SEE AT LEAST ONE VALID COLUMN?
///
/// This is not a style question. `ibm-forbids-what-our-fold-does` records three things IBM's compiler
/// refuses to emit, and one of them is a FULLY-MASKED SOFTMAX ROW. The fill's own comment says block
/// `r*pages + p` "describes ROW `r`'s page `p` AND NOTHING ELSE. Every other row in that block stays at
/// the sentinel" — so if a pass's block is consumed by ops that compute all `nqh*mq` rows, then every row
/// not belonging to that pass is exactly the case IBM refuses, on every pass.
///
/// ⚠️ AND THE FUNCTION'S OWN HEADER DOC DESCRIBES THE OPPOSITE LAYOUT: "a block describes the whole batch
/// and there is one per page", which is the row-axis version that is NOT the live path
/// (`kv_batched_requests` measures false on every op). One of the two is stale, and they disagree about
/// precisely the thing this test measures.
///
/// So: report, per block, how many of its `nqh*mq` rows are entirely sentinel. This makes no claim about
/// whether it is the defect — it establishes the number, which nothing currently does.
#[test]
fn report_fully_masked_rows_per_block() {
    const COLS: u32 = 256;
    let nqh = 32usize;
    let lens = vec![225usize, 1099, 709, 225]; // the ragged probe's real depths
    let hs = batch(&lens, 1);
    let sh = shape::<COLS>(nqh, hs.len());
    let deepest = hs.iter().map(|h| h.end().get()).max().unwrap_or(0) as usize;
    let g = declared::<COLS>(deepest + 1, hs.len());
    let bytes = decode_batch_prefix_mask_f16::<COLS>(sh, g, &hs, f32::NEG_INFINITY);
    let zero = half::f16::from_f32(0.0).to_le_bytes();
    let rows = sh.nqh() * sh.mq();
    let blocks = MaskBlocks::of(g.grid()).get();
    let mut all_masked = 0u32;
    let mut per_block: Vec<(u32, u32)> = Vec::new();
    for b in 0..blocks {
        let mut masked_here = 0u32;
        for r in 0..rows {
            let any = (0..sh.cols().get()).any(|c| {
                let e = sh.elem(b, r, c);
                bytes[e * 2..e * 2 + 2] == zero
            });
            if !any {
                masked_here += 1;
                all_masked += 1;
            }
        }
        per_block.push((b, masked_here));
    }
    println!(
        "mq={} nqh={} pages/row={} blocks={blocks} rows/block={rows}",
        sh.mq(),
        sh.nqh(),
        g.grid().pages().get()
    );
    for (b, n) in per_block.iter() {
        let row = b / g.grid().pages().get();
        let page = b % g.grid().pages().get();
        println!(
            "  block {b} (row {row} page {page}): {n}/{rows} rows entirely sentinel{}",
            if *n == rows {
                "   <== NO ROW OF THIS PASS HAS A VALID COLUMN"
            } else {
                ""
            }
        );
    }
    // And what each row's own history actually spans, which is what makes the holes.
    for (i, h) in hs.iter().enumerate() {
        let runs: Vec<String> = h
            .runs()
            .iter()
            .map(|r| format!("[{},{})", r.start().get(), r.end().get()))
            .collect();
        println!("  row {i} history: {}", runs.join(" u "));
    }
    println!(
        "TOTAL fully-masked (block,row) pairs: {all_masked} of {}",
        blocks * rows
    );
}

/// ⭐ A PADDING SLOT HAS NO REQUEST, AND NONE CAN BE INVENTED FOR IT. `SlotMap::live_rows` is the one
/// constructor of `LiveBatch`, and it walks the map's OWN live slots — so the builder is asked exactly
/// the live indices, in slot order, and a padding slot has no `BatchRow` of its own.
///
/// What it RUNS is the separate question, and the answer is not "nothing": it is live 0, complete
/// (`LaunchRows::row`, pinned by `a_padding_row_replicates_live_0_in_both_masks`). Owning no request and
/// replicating one are not in tension — the padding slot is handed no request, no pages of its own and
/// no place in the live list, and it re-evaluates live 0 so that the cache write it cannot avoid making
/// is the write live 0 was already making into that cell.
#[test]
fn a_padding_slot_has_no_request_of_its_own_and_runs_live_0s() {
    use scratchy_subtile::sdsc_abstract::{BatchRow, SeqPos, SlotMap};
    // 3 live requests on the 4-row rung — the exact shape the corruption ships under.
    let width = RungWidth::of_baked_rows(4).expect("a 4-row rung");
    let slots = SlotMap::of_live(3, width).expect("3 live rows fit a 4-row rung");
    let mut asked: Vec<usize> = Vec::new();
    let live = slots
        .live_rows(|i| {
            asked.push(i);
            Ok::<_, ()>(BatchRow {
                rope_pos: SeqPos::new(10 + i as u32),
                hist: KvHistory::contiguous(5 + i),
            })
        })
        .expect("the builder never fails here");
    assert_eq!(
        asked,
        vec![0, 1, 2],
        "the builder is asked exactly the live slots, in slot order"
    );
    assert_eq!(
        live.live(),
        3,
        "the live count is the LIVE count, not the rung width"
    );
    for r in 0..3 {
        let row = live.row(r).expect("a live slot's row");
        assert_eq!(
            row.rope_pos,
            SeqPos::new(10 + r as u32),
            "slot {r} carries its own request's position"
        );
        assert!(
            row.hist.contains(5 + r - 1),
            "slot {r} carries its own request's history"
        );
    }
    assert!(
        live.row(3).is_none(),
        "the padding slot holds no request — nothing here invents one for it"
    );

    // ── AND WHAT IT RUNS ──────────────────────────────────────────────────────────────────────────
    // Live 0's token (101) differs from the LAST live token (303), and every rope_pos differs from any
    // launch position, so "took live 0's" is distinguishable from every wrong answer that has shipped.
    let toks = [101usize, 202, 303];
    let rows = live
        .launch_rows(&toks, width)
        .expect("a live batch lays into its rung");
    let pad = rows.row(3).expect("every bound slot runs something");
    let live0 = rows.row(0).expect("live 0");
    assert_eq!(
        pad.token().get(),
        101,
        "the pad row's token is LIVE 0's, not the last live token"
    );
    assert_eq!(
        pad.rope_pos(),
        SeqPos::new(10),
        "it rotates at LIVE 0's position"
    );
    assert_eq!(
        pad.new_block_col().index(),
        0,
        "it attends LIVE 0's new-block column"
    );
    assert_eq!(pad.hist(), live0.hist(), "and it reads LIVE 0's prefix");

    // ⭐ THE DOOR REFUSES A PARTIAL OR MISPAIRED LAUNCH. One token per LIVE row, never per bound row: a
    // rung-width list is how a padding row came to be paired with some other slot's token.
    assert!(
        live.launch_rows(&[101, 202, 303, 303], width).is_none(),
        "a rung-width token list cannot lay a live batch into its rung"
    );
    assert!(
        live.launch_rows(&[101, 202], width).is_none(),
        "nor can a list shorter than the live rows"
    );
    // Nor can more requests be laid in than the rung binds.
    let narrow = RungWidth::of_baked_rows(2).expect("a 2-row rung");
    assert!(
        live.launch_rows(&toks, narrow).is_none(),
        "3 live rows do not lay into a 2-row rung"
    );
}

/// ⭐⭐⭐⭐⭐ THE BLOCKS THE MASK IS STAGED WITH AND THE PASSES THE FOLD WALKS ARE **ONE NUMBER**.
///
/// ⛔ TWO POPULATIONS ANSWER IT DIFFERENTLY, AND BOTH ANSWERS LOOK LIKE "how many pages".
///
/// * The LAUNCH's is over EVERY LIVE REQUEST — the set the batch's shared write slot is taken over, the
///   set `ensure_pages` grows the pool for, and the set the shim's own ceiling
///   (`ceil(seq_pos / page_slots)`) is computed from. A request still PREFILLING is in it, and it may be
///   deeper than every decoding row.
/// * The STAGING's is over the DECODING ROWS ONLY, because those are the histories the mask fill walks —
///   plus the padding rows, whose history is live 0's.
///
/// They agree on every step where no prefilling request is the deepest, which is most of them, and at
/// `mq = 2` a single page usually covers both. They separate exactly when a live request that is not
/// decoding this step reaches into a page the decoders have not, and then:
///
/// * the fill composes block `row * pages_staged + page`, while `fold_plan::fold_pass` inverts the pass
///   index as `(rep / pages_walked, rep % pages_walked)` — so every row but row 0 is fed ANOTHER ROW'S
///   PAGE, and
/// * the passes past the staged bytes read a buffer the host never wrote, which in an ADDITIVE mask is
///   ZERO, which is VALID: those rows attend whatever the pool holds.
///
/// Fluent, wrong, no fault. So the number is DECLARED to the walker and comes back as the only value the
/// staging accepts, and the divergence needs two walkers — two runtimes — which one launch does not have.
#[test]
fn the_mask_is_blocked_by_the_passes_the_fold_was_declared_to_walk() {
    const COLS: u32 = PER_PAGE;
    let nqh = 4u32;
    let mask_neg = -(half::f16::MAX.to_f32() / 2.0);
    // A 4-row rung running 3 decoders and one padding slot — the shape a ragged batch actually runs.
    let rung = RungWidth::of_baked_rows(4).expect("a 4-row rung");
    let sh = shape::<COLS>(nqh as usize, rung.count());

    // THE DECODING ROWS: three shallow histories, then the padding slot's — which is LIVE 0's, because
    // its cache write races live 0's at one pool cell and only identical bytes commute there.
    let mut rows: Vec<KvHistory> = [100usize, 200, 300]
        .iter()
        .map(|&l| KvHistory::contiguous(l))
        .collect();
    rows.push(rows[0].clone());
    // AND THE REQUEST THAT IS NOT DECODING THIS STEP — still prefilling, and deeper than all of them.
    // Its keys are in the pool and the batch appends past them, which is why the launch counts it.
    let prefilling = KvHistory::contiguous(900);

    let per_page = SlotCount::new(COLS);
    let from_rows = FoldPages::covering(BatchSlot::of(rows.iter()), per_page)
        .expect("a row holds pages")
        .grid(rung, MaskBlockForm::PerRowPage);
    let from_live = FoldPages::covering(
        BatchSlot::of(rows.iter().chain(std::iter::once(&prefilling))),
        per_page,
    )
    .expect("a row holds pages")
    .grid(rung, MaskBlockForm::PerRowPage);

    // ⛔ THE TWO POPULATIONS, DISAGREEING. This is the whole content of the law: without it every
    // assertion below is trivially satisfied by either derivation.
    assert_eq!(
        from_rows.pages().get(),
        2,
        "the decoding rows reach 301 slots — two pages"
    );
    assert_eq!(
        from_live.pages().get(),
        4,
        "a live request reaches 901 — four pages"
    );
    assert_ne!(
        MaskBlocks::of(from_rows),
        MaskBlocks::of(from_live),
        "8 blocks against 16: one of these is the number of passes, and the mask must be the same one"
    );

    // ── THE LAUNCH ────────────────────────────────────────────────────────────────────────────────
    // It declares the LIVE population, because that is what the fold walks, and stages with what comes
    // back. There is no second number available to it.
    let mut walker = Walker::default();
    let fold = from_live
        .declare_to(&mut walker)
        .expect("the walker accepts");
    let bytes = decode_batch_prefix_mask_f16::<COLS>(sh, fold, &rows, mask_neg);

    assert_eq!(
        walker.pages(),
        from_live.pages().get(),
        "the walker divides pass indices by this"
    );
    assert_eq!(
        walker.blocks(),
        from_live.passes(),
        "and steps this many blocks"
    );
    assert_eq!(
        bytes.len() / 2,
        sh.elems(MaskBlocks::of(fold.grid())),
        "the staged buffer is exactly the declared blocks — no pass reads past what the host wrote"
    );

    // EVERY PASS FINDS ITS OWN ROW'S PAGE. `rep` is inverted the way the runtime inverts it, using the
    // number the runtime was told, and the block it lands on must state that row's history over that
    // page and NOTHING for any other row.
    let zero = half::f16::from_f32(0.0).to_le_bytes();
    let valid = |bytes: &[u8], block: u32, row: u32, col: u32| {
        let e = sh.elem(block, row, col);
        bytes[e * 2..e * 2 + 2] == zero
    };
    for rep in 0..walker.blocks() {
        let (r, p) = (rep / walker.pages(), rep % walker.pages());
        for h in 0..nqh {
            for other in 0..rung.get() {
                let row = HeadRequestRow::of(h, other, rung.get()).get();
                for col in 0..COLS {
                    let want = other == r && rows[r as usize].contains((p * COLS + col) as usize);
                    assert_eq!(
                        valid(&bytes, rep, row, col),
                        want,
                        "pass {rep} is row {r}'s page {p}: row {other} head {h} col {col}"
                    );
                }
            }
        }
    }

    // ── THE ARRANGEMENT THIS LAW FORBIDS ──────────────────────────────────────────────────────────
    // Staging the DECODING-ROW population while the fold walks the LIVE one. It takes a SECOND walker to
    // express at all — a launch has one runtime, and telling that runtime the smaller number would make
    // it the number walked, which is the two being one again. Here it is, built out of two, and here is
    // what it does.
    let mut elsewhere = Walker::default();
    let mismatched = from_rows
        .declare_to(&mut elsewhere)
        .expect("the walker accepts");
    let short = decode_batch_prefix_mask_f16::<COLS>(sh, mismatched, &rows, mask_neg);
    assert!(
        short.len() < bytes.len(),
        "the smaller population stages fewer blocks than the fold steps — the tail passes read bytes \
         nobody wrote, and an unwritten additive-mask byte is ZERO, which is VALID"
    );
    // The pass index a (row, page) pair composes to — the fill's own `row * pages + page`, and what
    // `fold_plan::fold_pass` inverts. Row 1's page 0 was WRITTEN at 2 (two pages) and is READ at 4 (four),
    // so block 4 of the short buffer is row 2's page 0: the pass meant for row 1 carries row 2's history
    // and row 1 sees nothing of its own.
    let pass_of = |row: u32, page: u32, pages: u32| row * pages + page;
    assert_eq!(
        pass_of(1, 0, elsewhere.pages()),
        2,
        "where row 1's page 0 was written"
    );
    let rep = pass_of(1, 0, walker.pages());
    assert_eq!(rep, 4, "and where the fold reads it");
    let row1 = HeadRequestRow::of(0, 1, rung.get()).get();
    let row2 = HeadRequestRow::of(0, 2, rung.get()).get();
    assert!(
        !valid(&short, rep, row1, 0),
        "the pass the fold calls row 1's page 0 says nothing valid for row 1"
    );
    assert!(
        valid(&short, rep, row2, 0),
        "because the block it lands on is ROW 2's page 0 — one request reading another's history"
    );
    // And the same pass, staged under the declared geometry, is row 1's own.
    assert!(valid(&bytes, rep, row1, 0), "row 1's first key is its own");
    assert!(
        !valid(&bytes, rep, row2, 0),
        "and row 2 contributes nothing to a pass it does not own"
    );
}
