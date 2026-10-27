// SPDX-License-Identifier: Apache-2.0
//! ⭐⭐⭐⭐⭐ THE V KERNEL'S ROW EXTENT NAMES ITS PLANE — as a type, not a number. The value leg's
//! shared kernel reads one of four V planes: the resident page plane (and the gathered scratch's
//! byte-for-byte copy of it), the new-token block's unrestickified wide stream, the same block's
//! CHUNK form (the whole-block `[mq_pad, hd]` contraction of `new_v`), or a chunk part's own
//! region in the gathered scratch. The row extent used to travel as a bare `u32` fed the HEAD DIM
//! at the new-block call — a head-dim fact standing in a plane-rows slot, coincident only where
//! `hd` is one stick.
//!
//! THE LAW'S TEETH: the gathered value leg refuses every plane but the page (attn.rs's arm door),
//! because its scratch rows ARE page rows — a window's, a stream's, a block's, or a part's extent
//! there once declared a stick-group stride the buffer does not have. These pins hold the doors
//! themselves.

use ktir_superdsc::sdsc_abstract::{ChunkScratch, PaddedMq, PagedKvPool, VKernelRows};

#[test]
fn the_page_planes_rows_are_the_pools_own_page_slots() {
    assert_eq!(
        VKernelRows::Page.rows(),
        PagedKvPool::PAGE_SLOTS as u32,
        "the resident V plane is `[PAGE_SLOTS, hd]` — its rows are the pool's own fact"
    );
}

#[test]
fn the_new_block_stream_carries_the_head_dim_it_declares() {
    for hd in [64u32, 128] {
        assert_eq!(
            VKernelRows::of_new_block_stream(hd).rows(),
            hd,
            "the wide stream's per-head slice is declared `hd` rows, at every head dim"
        );
    }
}

/// ⭐ THE IMPOSTOR SEAT THE ENUM REMOVES: a new-block stream whose row count HAPPENS to equal the
/// page's slots is still not the page plane. As bare `u32`s the two pairings were indistinguishable
/// — 256 == 256 — which is exactly the shape of every swap defect on this path: equal at every
/// value the tests pass, divergent at the first geometry that differs.
#[test]
fn a_stream_of_the_pages_row_count_is_still_not_the_page_plane() {
    assert_ne!(
        VKernelRows::Page,
        VKernelRows::of_new_block_stream(PagedKvPool::PAGE_SLOTS as u32),
        "the plane a kernel reads is a fact about WHICH buffer, not how many rows it happens to have"
    );
    // And each door reads back its own fact from its own slot.
    assert_eq!(
        VKernelRows::of_new_block_stream(PagedKvPool::PAGE_SLOTS as u32).rows(),
        PagedKvPool::PAGE_SLOTS as u32
    );
}

/// ⭐ THE CHUNK FORM CONTRACTS THE WHOLE PADDED BLOCK — `new_v`'s kv-head blocks are `[mq_pad, hd]`
/// (RoPE packs them `[nkvh][mq][hd]`), so the kernel's rows are the pad's own row extent. The door
/// demands the block's `SlotExtent`, minted only by `MqPad::slots`, so no other plane's number can
/// fill the slot.
#[test]
fn the_new_block_chunk_contracts_the_whole_padded_block() {
    for mq in [2u32, 5, 64, 256] {
        let pm = PaddedMq::of_bundle(mq, false).expect("a prefill chunk width the law pads");
        assert_eq!(
            VKernelRows::of_new_block_chunk(pm.pad().slots()).rows(),
            pm.pad().rows().row_axis_extent(),
            "the chunk form contracts the whole [mq_pad, hd] block — its rows are the pad's own"
        );
    }
    // The impostor seat, the chunk form's own: a block padded to the page's slot count is still
    // not the page plane.
    let pm = PaddedMq::of_bundle(PagedKvPool::PAGE_SLOTS as u32, false)
        .expect("a prefill chunk width the law pads");
    assert_ne!(
        VKernelRows::Page,
        VKernelRows::of_new_block_chunk(pm.pad().slots()),
        "the plane is a fact about WHICH buffer, not how many rows it happens to have"
    );
}

/// ⭐ A CHUNK PART'S V REGION IS THE PART'S OWN SLOTS — never the chunk's and never a page's. The
/// door demands the `ChunkPart` itself, minted only by `ChunkScratch::slot_parts`, so the extent
/// cannot arrive from any other plane's arithmetic.
#[test]
fn a_chunk_parts_rows_are_the_parts_own_slots() {
    let chunk = ChunkScratch::of_chunk(PagedKvPool::new(8, 64), 2).expect("a chunk granite admits");
    let part = chunk.slot_parts(1).next().expect("two pages split");
    assert_eq!(
        VKernelRows::of_chunk_part(part).rows(),
        part.slots(),
        "the kernel's rows are the part's own extent — the scratch is read at its rows"
    );
    // The impostor seat, the part form's: a full part holds exactly the page's own slot count and
    // is still not the page plane.
    assert_eq!(part.slots(), PagedKvPool::PAGE_SLOTS as u32);
    assert_ne!(
        VKernelRows::Page,
        VKernelRows::of_chunk_part(part),
        "the plane is a fact about WHICH buffer, not how many rows it happens to have"
    );
}
