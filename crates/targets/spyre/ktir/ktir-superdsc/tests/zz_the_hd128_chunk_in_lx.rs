// SPDX-License-Identifier: Apache-2.0
//! ⭐⭐⭐ granite-3.1-8b's ONE-REQUEST DECODE CHUNK KEEPS ITS GATHERED Kᵗ AND V IN LX, at every rung the
//! frontend's LX window holds, from the shipped `assemble_attn` against a real layout whose attention
//! class registry names hd 128 (the query-group form the 8b decode runs).
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

use ktir_superdsc::ir::bridge::tiled_op_sdsc_op::assemble_attn;
use ktir_superdsc::place::PlaceId;
use ktir_superdsc::placement::{BundleLayout, LX_PIN_BASE, LX_PIN_END, SegRole, TensorPlacement};
use ktir_superdsc::sdsc_abstract::{
    AttnGeometry, ChunkRowKind, ChunkScratch, PagedKvPool, attn_bundle_rows,
};

const NQH: u32 = 32;
const NKVH: u32 = 8;
const HD: u32 = 128;
const NSLAB: u32 = 2;
const CAP: u32 = 256;
const ENTRY_BYTES: u64 = HD as u64 * 64 * 2;

fn pool() -> PagedKvPool {
    PagedKvPool::new(NKVH as usize, HD as usize)
}

fn emit(pages: u32) -> Vec<(String, serde_json::Value)> {
    let mut layout = BundleLayout {
        attn_class_hds: vec![HD],
        ..BundleLayout::default()
    };
    let mut names = Vec::new();
    let ident = ktir_superdsc::reserved_tids::identity_class_tid(0);
    for (i, tid) in (1..=8u32).chain([900, ident]).enumerate() {
        let name = PlaceId::Act(tid).to_string();
        layout
            .ids
            .borrow_mut()
            .insert(name.clone(), PlaceId::Act(tid));
        names.push(name);
        layout.placements.insert(
            tid,
            TensorPlacement {
                tid,
                role: SegRole::Activation,
                segment: 3,
                bank: 0,
                offset: (i as u64) << 30,
                size: 1 << 30,
            },
        );
    }
    let geom = AttnGeometry::<NQH, NKVH, HD>::minted();
    let rows = attn_bundle_rows(geom, 1, true).expect("the one-request decode rung");
    let chunk = ChunkScratch::of_chunk(pool(), pages).expect("a chunk granite-8b admits");
    let mut sym = 0i64;
    assemble_attn(
        0,
        geom,
        rows,
        CAP,
        CAP,
        &names[0],
        &names[1],
        &names[2],
        &names[3],
        &names[4],
        &names[5],
        &names[6],
        Some(&names[7]),
        Some(chunk),
        PlaceId::Act(900),
        false,
        false,
        &mut sym,
        Some(&layout),
    )
    .unwrap_or_else(|e| panic!("pages={pages}: assemble_attn refused: {}", e.0))
    .iter()
    .map(|e| {
        let v = serde_json::to_value(&e.op).unwrap();
        let name = v["dscs_"][0]
            .as_object()
            .and_then(|m| m.keys().next().cloned())
            .expect("one named dsc per emitted op");
        (name, v)
    })
    .collect()
}

fn lx_only(v: &serde_json::Value) -> bool {
    v["memOrg_"]
        .as_object()
        .is_some_and(|m| m.len() == 1 && m.contains_key("lx"))
}

/// The one start address an operand's allocate node gives every core, or `None` if they differ.
fn pin(body: &serde_json::Value, lds: &serde_json::Value) -> Option<u64> {
    let idx = lds["ldsIdx_"].as_u64();
    let alloc = body["scheduleTree_"]
        .as_array()?
        .iter()
        .find(|n| n["nodeType_"] == "allocate" && n["ldsIdx_"].as_u64() == idx)?;
    assert_eq!(alloc["component_"], "lx");
    let starts: std::collections::BTreeSet<u64> = alloc["startAddressCoreCorelet_"]["data_"]
        .as_object()?
        .values()
        .map(|a| a.as_str().map_or_else(|| a.as_u64(), |s| s.parse().ok()))
        .collect::<Option<_>>()?;
    (starts.len() == 1).then(|| starts.into_iter().next().unwrap())
}

/// ⭐ WHICH RUNGS KEEP WHICH LEG IN LX, from the form's own law. Kᵗ is in LX on every rung that is a whole
/// number of 8-page parts, V on every rung with at least one 16-page part whose tail is whole sticks, and
/// both only while their pins fit the frontend's window. The 48-, 56- and 64-page rungs would pin 1.5,
/// 1.75 and 2 MiB; the window is 1,494,272 B.
#[test]
fn the_hd128_lx_form_covers_exactly_the_rungs_its_pins_fit() {
    let window = LX_PIN_END - LX_PIN_BASE;
    assert_eq!(window, 1_494_272);
    let mut admitted = Vec::new();
    for &j in &ChunkScratch::LADDER {
        let chunk = ChunkScratch::of_chunk(pool(), j).unwrap();
        for rows in [ChunkRowKind::Prompt, ChunkRowKind::Batch] {
            assert!(chunk.lx_form(rows).is_none(), "{j} page(s), {rows:?}");
        }
        // Kᵗ: one 16 KiB tensor per (8-page part, kv head). V: one per (16-page part, kv head, slab).
        // Both come to `pages * nkvh / 8` entries a core.
        let k_pages = if j % 8 == 0 { j } else { 0 };
        let v_pages = j / 16 * 16;
        let pinned = u64::from(k_pages + v_pages) * u64::from(NKVH) * ENTRY_BYTES / 8;
        let fits = v_pages > 0 && pinned <= window;
        let Some(lx) = chunk.lx_form(ChunkRowKind::OneRequest) else {
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
    assert_eq!(admitted, [16, 20, 24, 28, 32, 36, 40, 44, 52, 60]);
}

/// ⭐⭐⭐ EVERY LX COPY AND ITS CONSUMER MEET AT ONE PIN, read off the emitted descriptors at every rung.
/// Each Kᵗ copy writes one (part, kv head) tensor that exactly one score part reads, and each V copy one
/// (part, kv head, slab) tensor that exactly one PV reads. Every pin is one 16 KiB entry per core, every
/// pin lies inside the frontend's window, and no two tensors share one. What stays in HBM is exactly
/// what the form says: Kᵗ whole on a rung with a Kᵗ tail, and V's tail pages. That every rung emits at
/// all is the build-time proof each consumer divides its work as its copy did (the core-local read
/// law).
#[test]
fn every_hd128_lx_copy_and_its_consumer_meet_at_one_pin() {
    for &j in &ChunkScratch::LADDER {
        let lx = ChunkScratch::of_chunk(pool(), j)
            .unwrap()
            .lx_form(ChunkRowKind::OneRequest);
        let ops = emit(j);
        let mut all_pins = std::collections::BTreeSet::new();
        for (copy, consumer) in [("attn_cgkt_", "attn_csc_g"), ("attn_cgv_", "attn_cov_g")] {
            let keys = copy == "attn_cgkt_";
            let (mut copy_pins, mut hbm_copies) = (Vec::new(), 0usize);
            for (name, v) in ops.iter().filter(|(n, _)| n.starts_with(copy)) {
                let body = &v["dscs_"][0][name];
                let out = body["labeledDs_"].as_array().unwrap().last().unwrap();
                if lx_only(out) {
                    copy_pins.push(pin(body, out).expect("one pin on every core"));
                } else {
                    hbm_copies += 1;
                }
            }
            let (mut read_pins, mut hbm_parts) = (Vec::new(), 0usize);
            for (name, v) in ops.iter().filter(|(n, _)| n.starts_with(consumer)) {
                let body = &v["dscs_"][0][name];
                let kernel = body["labeledDs_"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .find(|l| l["dsType_"] == "KERNEL")
                    .expect("the consumer's kernel");
                if lx_only(kernel) {
                    read_pins.push(pin(body, kernel).expect("one pin on every core"));
                } else {
                    hbm_parts += 1;
                }
            }
            let (lx_tensors, want_hbm_copies, want_hbm_parts) = match (lx, keys) {
                (None, _) => (
                    0,
                    j as usize,
                    if keys { NKVH } else { NKVH * NSLAB } as usize,
                ),
                (Some(lx), true) => match lx.keys() {
                    Some(k) => (k.tensors() as usize, 0, 0),
                    None => (0, j as usize, NKVH as usize),
                },
                (Some(lx), false) => {
                    let tail = lx.tail_pages();
                    (
                        lx.lx_tensors() as usize,
                        (tail * 4 * NKVH / 32) as usize,
                        if tail > 0 { (NKVH * NSLAB) as usize } else { 0 },
                    )
                }
            };
            assert_eq!(copy_pins.len(), lx_tensors, "{copy} at {j} page(s)");
            assert_eq!(read_pins.len(), lx_tensors, "{consumer} at {j} page(s)");
            assert_eq!(hbm_copies, want_hbm_copies, "{copy} HBM at {j} page(s)");
            assert_eq!(hbm_parts, want_hbm_parts, "{consumer} HBM at {j} page(s)");
            let (mut c, mut r) = (copy_pins.clone(), read_pins.clone());
            c.sort();
            r.sort();
            assert_eq!(
                c, r,
                "{copy} at {j} page(s): copy and consumer meet at one pin"
            );
            for p in copy_pins {
                assert!(
                    (LX_PIN_BASE..LX_PIN_END - ENTRY_BYTES + 1).contains(&p),
                    "{copy} at {j} page(s): pin {p} leaves the window"
                );
                assert!(all_pins.insert(p), "{j} page(s): pin {p} shared");
            }
        }
        // The pins are bump-allocated back to back, 16 KiB each, so their span IS the form's byte count.
        if let (Some(lx), Some(&lo), Some(&hi)) =
            (lx, all_pins.iter().next(), all_pins.iter().next_back())
        {
            assert_eq!(
                hi + ENTRY_BYTES - lo,
                lx.pinned_bytes_per_core(),
                "{j} page(s)"
            );
        }
    }
}

/// ⭐ THE HOST's TABLE IS THE COPIES' RUNS. Kᵗ's run is part-major when Kᵗ stays in LX (stick
/// `p * nkvh + h`, word `c` = slot stick `c` of kv head `h`'s part `p`), head-major when it does not. V's
/// run follows it, part-major by (part, kv head, slab): word `c` of a whole part's stick is run `c` of
/// that slab, V's source block `slab * 2 + c % 2` of page `16 p + c / 2`. Both runs cover every (kv
/// head, page, block) exactly once, and each copy reads its own stick of the table.
#[test]
fn the_hd128_lx_table_is_each_legs_part_major_run() {
    let per_page = ktir_superdsc::sdsc_abstract::gather_entries_per_page(
        32 * pool().stick_block_bytes(),
        pool(),
    )
    .unwrap();
    for &j in &ChunkScratch::LADDER {
        let chunk = ChunkScratch::of_chunk(pool(), j).unwrap();
        let Some(lx) = chunk.lx_form(ChunkRowKind::OneRequest) else {
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
        // The host's table is Kᵗ's run, then V's.
        let pages: Vec<i64> = (0..j as i64).collect();
        let table = ktir_superdsc::sdsc_abstract::chunk_gather_index_table(
            chunk,
            ChunkRowKind::OneRequest,
            &pages,
            per_page,
        )
        .unwrap();
        assert_eq!(table.len() as u32, 2 * n, "{j} page(s)");
        let word = |(kvh, page, block): (u32, u32, u32)| {
            ktir_superdsc::sdsc_abstract::GatherEntry::of_page_block(
                per_page,
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
        let Some(lx) = chunk.lx_form(ChunkRowKind::OneRequest) else {
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
