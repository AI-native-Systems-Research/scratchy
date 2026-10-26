// SPDX-License-Identifier: Apache-2.0
//! ⭐ WHAT THE SPLIT-KV CHUNK's GATHER LAYS DOWN, element by element, at every head dim it admits: each
//! kv head's Kᵗ as RowBlocked `[hd, chunk slots]` and its V as RowBlocked `[chunk slots, hd]` — the two
//! kernels the score and value legs read — out of the pool's page planes, through the entry orders the
//! copies and the host's index table share (`ChunkScratch::entry_coord` / `value_entry_coord`).

use ktir_superdsc::sdsc_abstract::{ChunkScratch, PagedKvPool};

const NKVH: usize = 8;
const STICK: u64 = 64;
const PAGE_SLOTS: u64 = PagedKvPool::PAGE_SLOTS as u64;

/// For destination element `(kvh, dest_in_head)` of a gather whose entry order is `coord`, the
/// `(page, element in that page's plane)` it is copied from: the entry it falls in names its source
/// block, and the offset inside the entry carries over.
fn source_of(
    chunk: ChunkScratch,
    coord: impl Fn(u32) -> (u32, u32, u32),
    kvh: u64,
    dest_in_head: u64,
) -> (u64, u64) {
    let ee = chunk.plane().entry_elems();
    let e = kvh * chunk.entries_per_row() as u64 + dest_in_head / ee;
    let (k2, page, q) = coord(e as u32);
    assert_eq!(
        k2 as u64, kvh,
        "entry {e} of kv head {kvh}'s run names kv head {k2}"
    );
    let block = chunk.source_block(k2, q).expect("a block inside the head") as u64;
    (page as u64, block * ee + dest_in_head % ee)
}

#[test]
fn each_kv_head_gathers_whole_chunk_kernels_at_every_admitted_head_dim() {
    for hd in [64u64, 128, 256] {
        let chunk = ChunkScratch::of_chunk(PagedKvPool::new(NKVH as u32, hd as u32), 2)
            .unwrap_or_else(|| panic!("hd {hd}: a two-page chunk is admitted"));
        let slots = chunk.slots() as u64;
        let hbe = chunk.plane().head_block_elems();
        for kvh in 0..NKVH as u64 {
            for sl in 0..slots {
                let (page, s) = (sl / PAGE_SLOTS, sl % PAGE_SLOTS);
                for d in 0..hd {
                    // Kᵗ: slot sticks outermost, in the chunk and in the page alike.
                    let k_dest = (sl / STICK) * (hd * STICK) + d * STICK + sl % STICK;
                    let k_src = kvh * hbe + (s / STICK) * (hd * STICK) + d * STICK + s % STICK;
                    assert_eq!(
                        source_of(chunk, |e| chunk.entry_coord(e), kvh, k_dest),
                        (page, k_src),
                        "hd {hd}: K of kv head {kvh}, slot {sl}, feature {d}"
                    );
                    // V: feature slabs outermost — `[slab][slot][stick]` in the page, and slab-major
                    // over the WHOLE chunk in the scratch.
                    let v_dest = (d / STICK) * (slots * STICK) + sl * STICK + d % STICK;
                    let v_src =
                        kvh * hbe + (d / STICK) * (PAGE_SLOTS * STICK) + s * STICK + d % STICK;
                    assert_eq!(
                        source_of(chunk, |e| chunk.value_entry_coord(e), kvh, v_dest),
                        (page, v_src),
                        "hd {hd}: V of kv head {kvh}, slot {sl}, feature {d}"
                    );
                }
            }
        }
    }
}

/// At ONE slab V's order IS K's, so the two gathers share one index run and the emission is the
/// one-slab form; above it V reads its own run after K's, and the host stages both.
#[test]
fn v_reads_its_own_index_run_only_above_one_slab() {
    let at =
        |hd: usize| ChunkScratch::of_chunk(PagedKvPool::new(NKVH as u32, hd as u32), 4).unwrap();
    let one = at(64);
    assert_eq!(one.value_index_base(), 0);
    assert_eq!(one.table_sticks(), one.index_sticks());
    assert!((0..one.entries()).all(|e| one.value_entry_coord(e) == one.entry_coord(e)));
    let two = at(128);
    assert_eq!(two.value_index_base(), two.index_sticks());
    assert_eq!(two.table_sticks(), 2 * two.index_sticks());
    let sticks: Vec<u32> = two.value_copies().map(|c| c.stick()).collect();
    assert_eq!(
        sticks,
        (two.index_sticks()..two.table_sticks()).collect::<Vec<_>>()
    );
}

/// A head dim whose stick blocks straddle V's slabs has no slab-major order, so the chunk is refused —
/// the model keeps the per-page fold, as a build-time fact.
#[test]
fn a_head_dim_past_the_page_is_refused() {
    assert!(ChunkScratch::of_chunk(PagedKvPool::new(NKVH as u32, 512), 2).is_none());
}
