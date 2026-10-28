// SPDX-License-Identifier: Apache-2.0
//! ⭐⭐⭐⭐⭐ A ONE-REQUEST BODY READS ITS PREFIX MASK AS ONE BROADCAST ROW — never the fold's blocks.
//!
//! The W=1 text defect (auditor ruling 2026-10-27), pinned at the layout law it broke: the owner
//! session's bodies are ONE-REQUEST bodies, and the door rules their gathered reads broadcast-form
//! (`BundleAttnParams::gathers_kv`: "a one-request body gathers only as a chunk, whose read IS the
//! broadcast row"). The deleted solo loop staged that row — a FLAT `[cap]` f16 buffer, valid where
//! the request's history holds the slot. The W=1 arm staged the fold's BLOCKS instead
//! (`decode_batch_prefix_mask_f16`, stick-blocked per pass), and the fused body read them FLAT:
//!
//! The blocks buffer is a `[nqh*mq, page_slots]` stick-blocked nest — stick `j` at flat offset
//! `j·rows·64`, row `r`'s slot in a stick at `r·64`. Read FLAT as the broadcast row, column `c`
//! for `c < 64` is row 0's own first stick (TRUE bytes), and column `c ≥ 64` is ROW `c/64`'s
//! first stick — the HEAD ROWS' own validity, mostly VALID. A step whose sweep stays inside one
//! stick is therefore byte-identical either way (the defect hid there for the whole 64-column
//! rung), and the FIRST STEP PAST IT — the rung 64→128 switch — attends ~64 phantom columns of
//! unwritten keys: a small deterministic logits perturbation that flips the first near-tie. On
//! card, granite-3.1-2b fp8: 3/3 prompts, one token each, at valid_len 67/72/68, deterministic.
//!
//! `BatchPrefix::OneRequest` is the fix — this file pins the law it exists to carry, so staging
//! the fold's blocks for a one-request body can never again look like a layout question.

use ktir_superdsc::sdsc_abstract::{
    BatchSlot, DeclaredFold, FoldPages, FoldWalker, KvHistory, MaskBlockForm, MaskBlocks,
    PagedKvPool, PrefixMaskShape, RungWidth, SlotCount, decode_batch_prefix_mask_f16,
};

const POOL_STICK: u32 = ktir_superdsc::sdsc_abstract::POOL_STICK;
const PER_PAGE: u32 = PagedKvPool::PAGE_SLOTS as u32;
/// `-(f16::MAX / 2)` = −32752.0 — the mask's −∞, spelled as a literal (`f16::to_f32` is not const).
const MASK_NEG: f32 = -32752.0;

/// An infallible sink for `declare_to` — the test only needs the [`DeclaredFold`].
struct Sink;
impl FoldWalker for Sink {
    type Error = core::convert::Infallible;
    fn declare_fold(&mut self, _: FoldPages, _: MaskBlocks) -> Result<(), Self::Error> {
        Ok(())
    }
}

/// The fold's own grid for one request at `pos` on a one-row rung — the geometry the W=1 arm
/// declared when it staged the blocks.
fn one_request_fold(pos: u32) -> DeclaredFold {
    let slot = BatchSlot::solo(&KvHistory::contiguous(pos as usize));
    let pages = FoldPages::covering(slot, SlotCount::new(PER_PAGE)).expect("pos admits pages");
    pages
        .grid(
            RungWidth::of_baked_rows(1).expect("1 is a baked width"),
            MaskBlockForm::PerRowPage,
        )
        .declare_to(&mut Sink)
        .expect("the sink is infallible")
}

/// Read one f16 out of the staged bytes.
fn f16_at(buf: &[u8], elem: usize) -> f32 {
    half::f16::from_le_bytes([buf[elem * 2], buf[elem * 2 + 1]]).to_f32()
}

/// ⭐ THE FIRST STICK IS ROW 0'S OWN — the defect's hiding place. Every column of the flat
/// broadcast read below 64 carries the TRUE validity, which is why a 64-column sweep could not
/// see the wrong form at all.
#[test]
fn the_flat_read_of_the_first_stick_is_true_validity() {
    let (nqh, p) = (8u32, 58usize);
    let shape = PrefixMaskShape::<{ POOL_STICK }, PER_PAGE>::new(
        nqh,
        RungWidth::of_baked_rows(1).expect("1 is a baked width"),
    )
    .expect("nonzero heads");
    let hist = KvHistory::contiguous(p);
    let buf = decode_batch_prefix_mask_f16(shape, one_request_fold(p as u32), &[hist], MASK_NEG);
    for c in 0..64 {
        let want = if c < p { 0.0 } else { MASK_NEG };
        assert_eq!(
            f16_at(&buf, c),
            want,
            "flat column {c}: the first stick is row 0's own bytes"
        );
    }
}

/// ⭐⭐⭐ THE PHANTOM COLUMNS — in the stick-blocked nest `dev_off(r,c) = (c/64)·(rows·64) + r·64 + c%64`,
/// flat column `C ≥ 64` of a row-0 read is HEAD ROW `C/64`'s slot `C%64` of stick 0 — the head
/// row's OWN history validity, so it reads VALID wherever `C%64 < p`. The broadcast law demands
/// −∞ there (the slot is past the history), so a one-request body sweeping past one stick attends
/// keys the mask was never told to silence — `p` phantom columns per extra stick. This is the
/// flipped near-tie, at the layout.
#[test]
fn the_flat_read_past_the_first_stick_is_phantom_validity() {
    let (nqh, p) = (8u32, 58usize);
    let shape = PrefixMaskShape::<{ POOL_STICK }, PER_PAGE>::new(
        nqh,
        RungWidth::of_baked_rows(1).expect("1 is a baked width"),
    )
    .expect("nonzero heads");
    let hist = KvHistory::contiguous(p);
    let buf = decode_batch_prefix_mask_f16(shape, one_request_fold(p as u32), &[hist], MASK_NEG);
    let rows = (nqh * 1) as usize; // mq = 1: one request, `nqh` head rows
    for c in 64..rows * 64 {
        // Flat column `c` is head row `c/64`'s slot `c%64` of stick 0: the head row's own
        // validity, so it is VALID (0.0) wherever `c%64 < p`, exactly where the broadcast law
        // demands −∞ (the slot is past the single live row's history).
        let head_row_own = if c % 64 < p { 0.0 } else { MASK_NEG };
        assert_eq!(
            f16_at(&buf, c),
            head_row_own,
            "flat column {c} is head row {}'s slot {} of stick 0 — the history's own validity",
            c / 64,
            c % 64
        );
    }
    // …and the phantom block (columns 64..64+p) reaches into a 128-column sweep — the rung the
    // card switched to at valid_len 65 — so the phantom columns are ones the body really attends.
    assert!(rows * 64 >= 128);
    assert!(p > 0, "a nonempty history leaves phantom columns past 64");
}

/// ⭐ AND THE BROADCAST ROW ITSELF IS NOT THE BLOCKS' SIZE — the two spellings are different
/// BUFFERS, not two fills of one: a `[cap]` row against `blocks × rows × page_slots`. The bind
/// guard admits the smaller into the larger's placement (an under-bind is "legitimate and
/// common"), which is the silence the phantom columns rode in on.
#[test]
fn the_broadcast_row_is_a_different_buffer_than_the_blocks() {
    let (nqh, p) = (8u32, 58usize);
    let shape = PrefixMaskShape::<{ POOL_STICK }, PER_PAGE>::new(
        nqh,
        RungWidth::of_baked_rows(1).expect("1 is a baked width"),
    )
    .expect("nonzero heads");
    let fold = one_request_fold(p as u32);
    let hist = KvHistory::contiguous(p);
    let buf = decode_batch_prefix_mask_f16(shape, fold, &[hist], MASK_NEG);
    let blocks = fold.grid().passes() as usize;
    assert_eq!(
        buf.len(),
        blocks * shape.block_elems() * 2,
        "the blocks buffer is blocks × rows × page_slots f16"
    );
    assert!(
        buf.len() > PER_PAGE as usize * 2,
        "the blocks buffer is larger than one broadcast row of the page"
    );
}
