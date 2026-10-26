// SPDX-License-Identifier: Apache-2.0
//! The split-KV chunk's LX run law and its doors, at granite-3.1-2b's geometry.
//!
//! (The emitter-side half of this file — the chunk body's op counts, shapes, splits and LX pins,
//! minted through the deleted hand-written attention emitter — went with it; when the live spliced
//! kernels need the same pins, mint them from the splice, not from a copy of the old emitter.)

use ktir_superdsc::sdsc_abstract::{ChunkRowKind, ChunkScratch, PagedKvPool};

const NKVH: u32 = 8;
const HD: u32 = 64;

/// ⭐ THE RUN's ORDER IS ONE LAW FOR BOTH LEGS' COPIES AND THE HOST. Word `c` of a whole part's stick
/// names slot run `c` of that part, the run its score and PV both hand core `c`. The tail is the flat
/// chunk order over the tail's pages, and the run covers every (kv head, page, block) of the chunk
/// exactly once. The host's table is this one run, and each copy reads exactly its own stick of it.
#[test]
fn the_lx_form_run_covers_the_chunk_once_in_the_order_each_part_pv_reads() {
    let pool = PagedKvPool::new(NKVH, HD);
    for &j in ChunkScratch::LADDER.iter().filter(|&&j| j >= 8) {
        let chunk = ChunkScratch::of_chunk(pool, j).unwrap();
        let vlx = chunk.lx_form(ChunkRowKind::OneRequest, 0, 0).unwrap();
        let eph = chunk.entries_per_head_page();
        let mut seen = std::collections::BTreeSet::new();
        for e in 0..chunk.entries() {
            let (kvh, page, block) = vlx.entry_coord(e);
            assert!(
                kvh < NKVH && page < j && block < eph,
                "{j} page(s), entry {e}"
            );
            assert!(
                seen.insert((kvh, page, block)),
                "{j} page(s): entry {e} repeats"
            );
            if e < vlx.lx_tensors() * 32 {
                let (op, c) = (e / 32, e % 32);
                let (part, h) = (op / NKVH, op % NKVH);
                assert_eq!(vlx.lx_tensor(part, h, 0), op);
                assert_eq!(
                    (kvh, page * eph + block),
                    (h, part * 8 * eph + c),
                    "{j} page(s)"
                );
            }
        }
        assert_eq!(seen.len(), chunk.entries() as usize);
        // Each LX copy reads its own stick of the LX run: the ONE run both legs share when Kᵗ stays in
        // LX, else V's run after Kᵗ's own head-major one.
        let base = if vlx.keys().is_some() {
            0
        } else {
            chunk.index_sticks()
        };
        let sticks: Vec<u32> = vlx.copies().map(|(_, cp)| cp.stick()).collect();
        let want: Vec<u32> = (base..base + chunk.index_sticks()).collect();
        assert_eq!(sticks, want, "{j} page(s)");
        // Kᵗ's own copies (with its witness) read the same sticks and write tensor `stick`.
        if let Some(keys) = vlx.keys() {
            let k: Vec<(u32, u32)> = keys.copies().map(|(i, cp)| (i, cp.stick())).collect();
            let want: Vec<(u32, u32)> = (0..chunk.index_sticks()).map(|j| (j, j)).collect();
            assert_eq!(k, want, "{j} page(s)");
        }
        // And the host's table is that run (after Kᵗ's own when Kᵗ is in HBM): the prompt chunk's
        // same entries, part-major.
        let pages: Vec<i64> = (0..j as i64).collect();
        // A one-layer pool's page is `nkvh * PAGE_SLOTS / POOL_STICK` stick blocks; the ceiling
        // carries the ladder's own top (PAGES_MAX = 64) as the page budget.
        let ceiling = ktir_superdsc::sdsc_abstract::gather_entry_ceiling(
            64,
            32 * pool.stick_block_bytes(),
            pool,
        )
        .unwrap();
        let one = ktir_superdsc::sdsc_abstract::chunk_gather_index_table(
            chunk,
            ChunkRowKind::OneRequest,
            &pages,
            ceiling,
            0,
            0,
        )
        .unwrap();
        let prompt = ktir_superdsc::sdsc_abstract::chunk_gather_index_table(
            chunk,
            ChunkRowKind::Prompt,
            &pages,
            ceiling,
            0,
            0,
        )
        .unwrap();
        assert_eq!(prompt.len() as u32, chunk.entries(), "{j} page(s)");
        let n = chunk.entries() as usize;
        let one = if vlx.keys().is_some() {
            assert_eq!(one.len(), n, "{j} page(s): one run");
            one
        } else {
            assert_eq!(one.len(), 2 * n, "{j} page(s): Kᵗ's run, then V's");
            assert_eq!(
                &one[..n],
                &prompt[..],
                "{j} page(s): Kᵗ's run is head-major"
            );
            one[n..].to_vec()
        };
        let (mut a, mut b) = (one.clone(), prompt.clone());
        a.sort();
        b.sort();
        assert_eq!(a, b, "{j} page(s): the same entries, reordered");
        // One whole part and no tail is the one width where part-major IS head-major.
        if vlx.lx_parts() > 1 || vlx.tail_pages() > 0 {
            assert_ne!(one, prompt, "{j} page(s): part-major, not head-major");
        } else {
            assert_eq!(one, prompt, "{j} page(s): one part is the head-major order");
        }
    }
}

/// ⛔ A Kᵗ LX FORM WITH A TAIL IS UNREPRESENTABLE — the bake-time lockout of the measured card fault.
/// granite-2b's 12- and 20-page rungs (a 4-page tail) faulted with Kᵗ in LX (`HMI=STORE`, see
/// `ChunkLx::keys`, which also says why that measurement is confounded by the session's allocation and
/// not yet re-run). The only door to the Kᵗ LX form (`ChunkKeysLx`) refuses
/// exactly the tail rungs, at every rung of both shipped granite pools.
#[test]
fn a_tail_rung_cannot_mint_keys_in_lx() {
    let pool = PagedKvPool::new(NKVH, HD);
    for &j in &ChunkScratch::LADDER {
        let Some(lx) =
            ChunkScratch::of_chunk(pool, j)
                .unwrap()
                .lx_form(ChunkRowKind::OneRequest, 0, 0)
        else {
            continue;
        };
        assert_eq!(
            lx.keys().is_some(),
            lx.tail_pages() == 0,
            "{j} page(s): tail {}",
            lx.tail_pages()
        );
    }
}

/// ⛔ A CORE-LOCAL TENSOR TAKES ONE WORK DIVISION: the first op's is recorded, the same one again stands,
/// and any other is refused — the build-time half of "each core holds only what its own producer wrote".
#[test]
fn a_core_local_tensor_refuses_a_second_division() {
    use ktir_superdsc::place::PlaceId;
    use ktir_superdsc::placement::BundleLayout;
    let l = BundleLayout::default();
    l.synth_lx_core_local(PlaceId::Act(7), 4096);
    let name = PlaceId::Act(7).to_string();
    assert!(l.lx_core_local(&name));
    assert_eq!(l.lx_pin(&name).map(|(_, fp)| fp), Some(4096));
    assert_eq!(
        l.record_lx_division(&name, "esub", "split {out: 32}".into()),
        Ok(())
    );
    assert_eq!(
        l.record_lx_division(&name, "exp", "split {out: 32}".into()),
        Ok(())
    );
    assert!(
        l.record_lx_division(&name, "sum", "split {mb: 32}".into())
            .is_err()
    );
}
