// SPDX-License-Identifier: Apache-2.0
//! ⭐ granite-3.1-8b's ONE-REQUEST DECODE CHUNK KEEPS ITS GATHERED Kᵗ AND V IN LX, at every rung the
//! frontend's LX window holds (the query-group form the 8b decode runs).
//!
//! At head dim 128 a gather entry is one 16 KiB stick block. It is a 64-slot stick of Kᵗ's
//! `[128, slots]` (both slabs), and 128 slots of ONE slab of V's slab-major `[slab][slot][stick]`. So
//! the two legs have different parts:
//!   * Kᵗ: 8 pages a part, one copy per (part, kv head). The part's score splits its columns, one slot
//!     stick per core.
//!   * V: 16 pages a part, one copy per (part, kv head, slab). The part's per-slab PV splits its
//!     contraction, 128 slots per core.
//!
//! Every pin is one 16 KiB entry per core. All of them must fit `[LX_PIN_BASE, LX_PIN_END)`, the
//! frontend's share of LX (`FRONTEND_LX_END`), which refuses the 48-, 56- and 64-page rungs.
//!
//! (The emitter-side half of this file — op-local pin slots against a minted chunk, and a trailing
//! norm's slot bounding the chunk before it — was pinned through the deleted hand-written attention
//! emitter and went with it; when the live spliced kernels need the same pin, mint it from the
//! splice, not from a copy of the old emitter.)

use ktir_superdsc::placement::{LX_PIN_BASE, LX_PIN_END};
use ktir_superdsc::sdsc_abstract::{ChunkRowKind, ChunkScratch, PagedKvPool};

const NKVH: u32 = 8;
const HD: u32 = 128;
const NSLAB: u32 = 2;
const ENTRY_BYTES: u64 = HD as u64 * 64 * 2;

fn pool() -> PagedKvPool {
    PagedKvPool::new(NKVH, HD)
}

#[test]
fn the_hd128_lx_form_covers_exactly_the_rungs_its_pins_fit() {
    let window = LX_PIN_END - LX_PIN_BASE;
    assert_eq!(window, 1_494_272);
    let mut admitted = Vec::new();
    for &j in &ChunkScratch::LADDER {
        let chunk = ChunkScratch::of_chunk(pool(), j).unwrap();
        // Both budget terms 0: this ladder is the GATHER law alone (which rungs have a whole leg
        // and what the gather pins weigh). The full core-local budget — gathers plus the
        // probability-part pins — went with the old emitter's op-local declaration.
        assert!(
            chunk.lx_form(ChunkRowKind::Prompt, 0, 0).is_none(),
            "{j} page(s), prompt"
        );
        // A decode batch's chunk takes the same LX form: the law reads only the chunk's pages.
        assert_eq!(
            chunk.lx_form(ChunkRowKind::Batch, 0, 0),
            chunk.lx_form(ChunkRowKind::OneRequest, 0, 0),
            "{j} page(s), batch"
        );
        // Kᵗ: one 16 KiB tensor per (8-page part, kv head). V: one per (16-page part, kv head, slab).
        // Both come to `pages * nkvh / 8` entries a core.
        let k_pages = if j % 8 == 0 { j } else { 0 };
        let v_pages = j / 16 * 16;
        let pinned = u64::from(k_pages + v_pages) * u64::from(NKVH) * ENTRY_BYTES / 8;
        let fits = (v_pages > 0 || k_pages > 0) && pinned <= window;
        let Some(lx) = chunk.lx_form(ChunkRowKind::OneRequest, 0, 0) else {
            assert!(!fits, "{j} page(s): {pinned} B of pins fit, but no LX form");
            continue;
        };
        assert!(fits, "{j} page(s): an LX form pinning {pinned} B");
        admitted.push(j);
        assert_eq!(lx.part_pages(), 16, "{j} page(s): a V part");
        assert_eq!(lx.lx_parts() * 16 + lx.tail_pages(), j);
        assert_eq!(lx.lx_tensors(), lx.lx_parts() * NKVH * NSLAB);
        assert_eq!(lx.per_core_bytes(), ENTRY_BYTES);
        assert_eq!(lx.pinned_bytes_per_core(), pinned, "{j} page(s)");
        assert_eq!(lx.keys().is_some(), k_pages > 0, "{j} page(s)");
        if let Some(k) = lx.keys() {
            assert_eq!(k.part_pages(), 8);
            assert_eq!(k.tensors(), j / 8 * NKVH);
        }
    }
    assert_eq!(admitted, [8, 16, 20, 24, 28, 32, 36, 40, 44, 52, 60]);
}

/// ⭐⭐ THE LX RUN LAW: every leg's entries are one part-major run over `(kv head, page, block)`,
/// V's run follows Kᵗ's in the index sticks, and the host's gather index table is the same run.
/// With one slab of V per entry and four Kᵗ blocks a page, V entry `e` of part `part` names kv head
/// `h`, V's source block `slab * 2 + c % 2` of page `16 p + c / 2`. Both runs cover every (kv
/// head, page, block) exactly once, and each copy reads its own stick of the table.
#[test]
fn the_hd128_lx_table_is_each_legs_part_major_run() {
    // The pool's own ceiling — 64 pages of budget, the ladder's own top (PAGES_MAX), so every
    // rung below names only pages it holds.
    let ceiling = ktir_superdsc::sdsc_abstract::gather_entry_ceiling(
        64,
        32 * pool().stick_block_bytes(),
        pool(),
    )
    .unwrap();
    for &j in &ChunkScratch::LADDER {
        let chunk = ChunkScratch::of_chunk(pool(), j).unwrap();
        let Some(lx) = chunk.lx_form(ChunkRowKind::OneRequest, 0, 0) else {
            continue;
        };
        let n = chunk.entries();
        assert_eq!(
            lx.index_base(),
            chunk.index_sticks(),
            "{j} page(s): V's run follows Kᵗ's"
        );
        // V's run, entry by entry.
        let mut seen = std::collections::BTreeSet::new();
        for e in 0..n {
            let (kvh, page, block) = lx.entry_coord(e);
            assert!(
                kvh < NKVH && page < j && block < 4,
                "{j} page(s), V entry {e}"
            );
            assert!(
                seen.insert((kvh, page, block)),
                "{j} page(s): V entry {e} repeats"
            );
            if e < lx.lx_tensors() * 32 {
                let (op, c) = (e / 32, e % 32);
                let (head_part, slab) = (op / NSLAB, op % NSLAB);
                let (part, h) = (head_part / NKVH, head_part % NKVH);
                assert_eq!(lx.lx_tensor(part, h, slab), op);
                assert_eq!(
                    (kvh, page, block),
                    (h, part * 16 + c / 2, slab * 2 + c % 2),
                    "{j} page(s), V entry {e}"
                );
            }
        }
        assert_eq!(seen.len(), n as usize);
        // Kᵗ's run, where Kᵗ stays in LX.
        if let Some(k) = lx.keys() {
            let mut seen = std::collections::BTreeSet::new();
            for e in 0..n {
                let (kvh, page, block) = k.entry_coord(e);
                let (op, c) = (e / 32, e % 32);
                let (part, h) = (op / NKVH, op % NKVH);
                assert_eq!(k.tensor(part, h), op);
                assert_eq!(
                    (kvh, page, block),
                    (h, part * 8 + c / 4, c % 4),
                    "{j}: Kᵗ entry {e}"
                );
                assert!(seen.insert((kvh, page, block)));
            }
            assert_eq!(seen.len(), n as usize);
            let sticks: Vec<(u32, u32)> = k.copies().map(|(i, cp)| (i, cp.stick())).collect();
            assert_eq!(
                sticks,
                (0..chunk.index_sticks())
                    .map(|s| (s, s))
                    .collect::<Vec<_>>()
            );
        }
        let v_sticks: Vec<u32> = lx.copies().map(|(_, cp)| cp.stick()).collect();
        assert_eq!(
            v_sticks,
            (chunk.index_sticks()..2 * chunk.index_sticks()).collect::<Vec<_>>(),
            "{j} page(s)"
        );
        // The host's table is Kᵗ's run, then V's. Both budget terms 0: the run law in isolation.
        let pages: Vec<i64> = (0..j as i64).collect();
        let table = ktir_superdsc::sdsc_abstract::chunk_gather_index_table(
            chunk,
            ChunkRowKind::OneRequest,
            &pages,
            ceiling,
            0,
            0,
        )
        .unwrap();
        assert_eq!(table.len() as u32, 2 * n, "{j} page(s)");
        let word = |(kvh, page, block): (u32, u32, u32)| {
            ktir_superdsc::sdsc_abstract::GatherEntry::of_page_block(
                ceiling,
                i64::from(page),
                chunk.source_block(kvh, block).unwrap(),
            )
            .unwrap()
        };
        for e in 0..n {
            let key = match lx.keys() {
                Some(k) => k.entry_coord(e),
                None => chunk.entry_coord(e),
            };
            assert_eq!(table[e as usize], word(key), "{j} page(s): Kᵗ word {e}");
            assert_eq!(
                table[(n + e) as usize],
                word(lx.entry_coord(e)),
                "{j} page(s): V word {e}"
            );
        }
    }
}

/// ⭐ A V TAIL PART READS ITS OWN SLAB OF THE TAIL SCRATCH. The tail scratch holds each kv head's tail
/// pages slab-major, laid out by V's run (`ChunkLx::entry_coord` past the whole parts), and each tail
/// PV's kernel corner is that kv head's slab (`ChunkLx::tail_corner`). Read here as one law: the
/// destination element where tail entry `t` lands is the corner of its (kv head, slab) plus its run's
/// offset inside that slab.
#[test]
fn a_hd128_value_tail_part_reads_its_own_slab_of_the_tail_scratch() {
    for &j in &ChunkScratch::LADDER {
        let chunk = ChunkScratch::of_chunk(pool(), j).unwrap();
        let Some(lx) = chunk.lx_form(ChunkRowKind::OneRequest, 0, 0) else {
            continue;
        };
        if lx.tail_pages() == 0 {
            assert!(lx.tail_corner(0, 0).is_none());
            continue;
        }
        let tail_slots = lx.tail_pages() * 256;
        let per_head = lx.tail_pages() * 4;
        let entry_elems = HD * 64;
        for t in 0..NKVH * per_head {
            let e = lx.lx_tensors() * 32 + t;
            let (kvh, page, block) = lx.entry_coord(e);
            let (slab, run) = (block / 2, block % 2);
            // Where the copy writes entry `t`: the tail scratch's flat entry space.
            let dest = t * entry_elems;
            // Where the PV of (kvh, slab) reads that run: its corner, plus the run's slots in the slab.
            let corner = lx.tail_corner(kvh, slab).unwrap();
            let tail_page = page - lx.lx_parts() * 16;
            let slot = tail_page * 256 + run * 128;
            assert!(slot < tail_slots);
            assert_eq!(
                dest,
                corner.elems() + slot * 64,
                "{j} page(s), tail entry {t}"
            );
        }
    }
}
