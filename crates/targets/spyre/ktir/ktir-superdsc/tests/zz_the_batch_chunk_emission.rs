// SPDX-License-Identifier: Apache-2.0
//! The split-KV chunk over a decode BATCH (`BatchChunk`): the emission from the shipped `assemble_attn`
//! at granite-3.1-2b's and granite-3.1-8b's geometries, and the host law its mask and index are staged by.

use ktir_superdsc::ir::bridge::tiled_op_sdsc_op::{AttnFacts, assemble_attn};
use ktir_superdsc::sdsc_abstract::{
    AttnGeometry, BatchChunk, ChunkScratch, KvExtent, KvHistory, PagedKvPool, SlotCount,
    StickLayout, attn_bundle_rows, batch_chunk_empty_sweep_row, batch_chunk_mask_f16,
};

const CAP: u32 = 256;
/// The storage/sweep pair on the boundary's one carrier — the sweep is the full cap (no rung).
const EXTENT: KvExtent = KvExtent::new(SlotCount::new(CAP), SlotCount::new(CAP));

fn named(ops: Vec<ktir_superdsc::emit::EmittedOp>) -> Vec<(String, serde_json::Value)> {
    ops.iter()
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

/// One batch rung's body at `seqs` request rows over a `pages`-page chunk (`seqs × J` pages).
fn emit_batch<const NQH: u32, const NKVH: u32, const HD: u32>(
    seqs: u32,
    pages: u32,
) -> Vec<ktir_superdsc::emit::EmittedOp> {
    let geom = AttnGeometry::<NQH, NKVH, HD>::minted();
    let rows = attn_bundle_rows(geom, seqs, true).expect("a decode batch rung");
    let chunk = ChunkScratch::of_chunk(PagedKvPool::new(NKVH, HD), pages)
        .expect("a chunk the geometry admits");
    let mut sym = 0i64;
    assemble_attn(
        AttnFacts {
            t: 0,
            geom,
            bundle_rows: rows,
            extent: EXTENT,
            qs: "t_qs",
            new_k_scaled: "t_new_k",
            new_v: "t_new_v",
            kct: "t_kct",
            vc: "t_vc",
            pmask: "t_pmask",
            cmask: "t_cmask",
            kv_block_index: Some("t_kvbi"),
            chunk: Some(chunk),
            out_id: ktir_superdsc::place::PlaceId::Act(900),
            rows_are_requests: true,
            headmajor_handoff: false,
        },
        &mut sym,
        None,
    )
    .unwrap_or_else(|e| panic!("seqs={seqs} pages={pages}: assemble_attn refused: {}", e.0))
}

fn pool_2b() -> PagedKvPool {
    PagedKvPool::new(8, 64)
}

/// The per-dim `scale_` of the score op's fused epilogue operand — the mask. It is inserted just BEFORE
/// the output (`OpSpec::attach_fused_epilogue`), so it is the second-to-last labeled
/// operand; a dim the read BROADCASTS carries `-1` (`Scale::RedNonStick`), a dim it walks `1`.
fn mask_scale(v: &serde_json::Value, name: &str) -> Vec<i64> {
    let body = &v["dscs_"][0][name];
    let lds = body["labeledDs_"].as_array().expect("labeled operands");
    assert!(lds.len() >= 2, "{name}: no epilogue operand");
    let m = &lds[lds.len() - 2];
    assert_eq!(
        m["dsType_"], "OUTPUT",
        "{name}: the epilogue operand is {m}"
    );
    m["scale_"]
        .as_array()
        .expect("a scale per dim")
        .iter()
        .map(|s| s.as_i64().unwrap_or(0))
        .collect()
}

/// ⭐ THE BATCH LADDER IS `B × J` PAGES AND ITS SWEPT KEY IS `J` PAGES — and the host's inverse of that
/// key is the chunk the bake attended, at every rung of every batch width.
#[test]
fn the_batch_ladder_round_trips_through_its_swept_key() {
    let pool = pool_2b();
    assert_eq!(
        ChunkScratch::ladder_for(pool, 1),
        ChunkScratch::LADDER.to_vec(),
        "one request keeps the solo ladder"
    );
    for seqs in PagedKvPool::BATCH_RUNGS {
        let ladder = ChunkScratch::ladder_for(pool, seqs);
        assert!(
            ladder
                .iter()
                .all(|&t| t.is_multiple_of(seqs) && t / seqs >= 2 && t <= ChunkScratch::PAGES_MAX),
            "seqs={seqs}: {ladder:?}"
        );
        assert!(
            ladder.windows(2).all(|w| w[0] < w[1]),
            "seqs={seqs}: ascending"
        );
        for &t in &ladder {
            let chunk = ChunkScratch::of_chunk(pool, t).expect("admitted");
            let swept = chunk.swept_per_request(seqs);
            assert_eq!(swept, (t / seqs) * 256, "seqs={seqs} t={t}");
            assert_eq!(
                ChunkScratch::of_batch(pool, seqs, swept),
                Some(chunk),
                "seqs={seqs} t={t}: the host's chunk is the bake's"
            );
        }
    }
    // The canonical batched cell — conc-8 at ~1.05k-token prompts, 256 out — crosses 5 → 6 pages per
    // request; both are exact rungs, so no request pads a page.
    assert!(ChunkScratch::ladder_for(pool, 8).contains(&40));
    assert!(ChunkScratch::ladder_for(pool, 8).contains(&48));
}

/// The slots of each score part and each value part a batch chunk of `pages` pages emits: one whole chunk
/// part each, unless its gather stays in LX (`ChunkScratch::lx_form` at `ChunkRowKind::Batch`). Then each
/// leg is cut into its own parts (8 pages at one slab), plus V's tail.
fn batch_part_slots(pool: PagedKvPool, pages: u32) -> (Vec<u32>, Vec<u32>) {
    let chunk = ChunkScratch::of_chunk(pool, pages).unwrap();
    let cut = |per: u32| -> Vec<u32> {
        (0..pages.div_ceil(per))
            .map(|i| per.min(pages - i * per) * 256)
            .collect()
    };
    match chunk.lx_form(
        ktir_superdsc::sdsc_abstract::ChunkRowKind::Batch,
        /* no op-local pins: geometry only */ 0,
        /* no probability-part pins: geometry only */ 0,
    ) {
        None => (vec![pages * 256], vec![pages * 256]),
        Some(lx) => (
            lx.keys().map_or(vec![pages * 256], |k| cut(k.part_pages())),
            cut(lx.part_pages()),
        ),
    }
}

/// ⭐⭐⭐ THE BATCH CHUNK IS THE SOLO CHUNK's OPS AT `mq = B`: one score per kv head (and Kᵗ part, where
/// the gather stays in LX) over its group's `gqa·B` contiguous head-major rows, the mask read PER ROW (no
/// broadcast scale on the epilogue operand, where the shared-history chunk's mask is the broadcast row),
/// the value leg's contraction divided across the cores (one part, or one per V part in LX), and nothing a
/// fold op.
#[test]
fn a_batch_chunk_scores_every_request_row_against_the_whole_chunk_with_a_per_row_mask() {
    for seqs in PagedKvPool::BATCH_RUNGS {
        for t in ChunkScratch::ladder_for(pool_2b(), seqs) {
            let (k_parts, v_parts) = batch_part_slots(pool_2b(), t);
            let ops = emit_batch::<32, 8, 64>(seqs, t);
            for e in &ops {
                assert!(
                    !e.kv_page_fold,
                    "{} seqs={seqs} t={t}: a fold op",
                    e.op_name
                );
                let copy =
                    e.op_name.starts_with("attn_cgkt_") || e.op_name.starts_with("attn_cgv_");
                assert_eq!(e.kv_gathered, copy, "{} seqs={seqs} t={t}", e.op_name);
            }
            let ops = named(ops);
            let rows = u64::from(4 * seqs);
            let part_of = |name: &str| -> usize {
                name.split("_k")
                    .nth(1)
                    .and_then(|s| s.split('_').next())
                    .map_or(0, |p| p.parse().unwrap())
            };
            let sc: Vec<&(String, serde_json::Value)> = ops
                .iter()
                .filter(|(n, _)| n.starts_with("attn_csc_g"))
                .collect();
            assert_eq!(
                sc.len(),
                8 * k_parts.len(),
                "seqs={seqs} t={t}: one score per (kv head, Kᵗ part)"
            );
            for (name, v) in sc {
                let n = &v["dscs_"][0][name]["N_"];
                assert_eq!(
                    (n["mb_"].as_u64(), n["out_"].as_u64()),
                    (Some(rows), Some(u64::from(k_parts[part_of(name)]))),
                    "{name} seqs={seqs} t={t}: N_ = {n}"
                );
                assert_eq!(
                    mask_scale(v, name),
                    vec![1, 1],
                    "{name} seqs={seqs} t={t}: the mask is not read per row"
                );
            }
            let pv: Vec<&(String, serde_json::Value)> = ops
                .iter()
                .filter(|(n, _)| n.starts_with("attn_cov_g"))
                .collect();
            assert_eq!(
                pv.len(),
                8 * v_parts.len(),
                "seqs={seqs} t={t}: one value op per (kv head, V part)"
            );
            for (name, v) in pv {
                let sticks = u64::from(v_parts[part_of(name)] / 64);
                let most = (1..=32u64.min(sticks))
                    .rev()
                    .find(|s| sticks.is_multiple_of(*s))
                    .unwrap_or(1);
                let s = &v["numWkSlicesPerDim_"];
                assert_eq!(
                    (s["in"].as_u64().unwrap_or(1), s["mb"].as_u64().unwrap_or(1)),
                    (most, 1),
                    "{name} seqs={seqs} t={t}: split {s}"
                );
            }
        }
    }
}

/// ⛔ THE SHARED-HISTORY CHUNK STILL READS ITS MASK BROADCAST — the per-row read is the batch's alone.
#[test]
fn the_one_request_chunk_keeps_its_broadcast_mask_row() {
    let geom = AttnGeometry::<32, 8, 64>::minted();
    let rows = attn_bundle_rows(geom, 1, true).expect("the one-request decode rung");
    let chunk = ChunkScratch::of_chunk(pool_2b(), 8).expect("admitted");
    let mut sym = 0i64;
    let ops = named(
        assemble_attn(
            AttnFacts {
                t: 0,
                geom,
                bundle_rows: rows,
                extent: EXTENT,
                qs: "t_qs",
                new_k_scaled: "t_new_k",
                new_v: "t_new_v",
                kct: "t_kct",
                vc: "t_vc",
                pmask: "t_pmask",
                cmask: "t_cmask",
                kv_block_index: Some("t_kvbi"),
                chunk: Some(chunk),
                out_id: ktir_superdsc::place::PlaceId::Act(900),
                rows_are_requests: false,
                headmajor_handoff: false,
            },
            &mut sym,
            None,
        )
        .expect("the solo chunk"),
    );
    for (name, v) in ops.iter().filter(|(n, _)| n.starts_with("attn_csc_g")) {
        assert_eq!(
            mask_scale(v, name),
            vec![-1, 1],
            "{name}: the solo chunk's mask is no longer the broadcast row"
        );
    }
}

/// ⭐ AT hd=128 WITHOUT THE CLASS REGISTRY (an emission fixture, no identity table to re-lay with) THE
/// BATCH CHUNK's ROW BLOCKS ARE QUERY HEADS — one score per head over its `B` rows, one value op per
/// (head, slab) — and every one emits at every rung of every batch width. With the registry the batch
/// takes the query-group form instead (`zz_the_hd128_query_group_chunk`).
#[test]
fn an_hd128_batch_chunk_scores_per_query_head_over_its_request_rows() {
    let pool = PagedKvPool::new(8, 128);
    for seqs in PagedKvPool::BATCH_RUNGS {
        for t in ChunkScratch::ladder_for(pool, seqs) {
            let (k_parts, v_parts) = batch_part_slots(pool, t);
            let ops = named(emit_batch::<32, 8, 128>(seqs, t));
            let part_of = |name: &str| -> usize {
                name.split("_k")
                    .nth(1)
                    .and_then(|s| s.split('_').next())
                    .map_or(0, |p| p.parse().unwrap())
            };
            let sc: Vec<&(String, serde_json::Value)> = ops
                .iter()
                .filter(|(n, _)| n.starts_with("attn_csc_h"))
                .collect();
            assert_eq!(
                sc.len(),
                32 * k_parts.len(),
                "seqs={seqs} t={t}: one score per (query head, Kᵗ part)"
            );
            for (name, v) in sc {
                let n = &v["dscs_"][0][name]["N_"];
                assert_eq!(
                    (n["mb_"].as_u64(), n["out_"].as_u64()),
                    (
                        Some(u64::from(seqs)),
                        Some(u64::from(k_parts[part_of(name)]))
                    ),
                    "{name} seqs={seqs} t={t}: N_ = {n}"
                );
            }
            assert_eq!(
                ops.iter()
                    .filter(|(n, _)| n.starts_with("attn_cov_h"))
                    .count(),
                64 * v_parts.len(),
                "seqs={seqs} t={t}: one value op per (query head, slab, V part)"
            );
        }
    }
}

/// ⭐⭐⭐ THE MASK LAW: element `(row, slot)` is the score block's own RowBlocked offset, row `i` is
/// request `i % B`'s, and a row is valid EXACTLY on its own request's run where that request's history
/// holds the slot — never another request's run, never its own padded tail, never a shared-slot hole.
#[test]
fn the_batch_mask_validates_each_row_on_its_own_requests_history_alone() {
    let (seqs, j) = (4u32, 3u32);
    let chunk = ChunkScratch::of_chunk(pool_2b(), seqs * j).expect("admitted");
    let bc = BatchChunk::of(chunk, seqs, 4).expect("a whole number of pages per request");
    assert_eq!(bc.mask_rows(), 4 * seqs);
    assert_eq!(bc.mask_cols(), seqs * j * 256);
    let layout = StickLayout::row_blocked(bc.mask_rows() as usize, bc.mask_cols() as usize);
    for row in [0, 1, 5, 15] {
        for slot in [0, 63, 64, 700, bc.mask_cols() - 1] {
            assert_eq!(
                bc.mask_elem(row, slot),
                Some(layout.dev_off(row as usize, slot as usize)),
                "({row}, {slot})"
            );
        }
    }
    assert_eq!(bc.mask_elem(bc.mask_rows(), 0), None);
    // Request 0: 10 keys. Request 1: a whole page. Request 2: 5 keys, a hole, then one key a batch
    // appended at its shared write slot (300, past a 300-key batch-mate). Request 3: the padding row,
    // live 0's history.
    let hists = vec![
        KvHistory::contiguous(10),
        KvHistory::contiguous(256),
        {
            let mut h = KvHistory::contiguous(5);
            let deep = [KvHistory::contiguous(300)];
            h.record(
                ktir_superdsc::sdsc_abstract::BatchSlot::of(deep.iter()),
                ktir_superdsc::sdsc_abstract::SlotCount::new(1),
            );
            assert!(h.contains(300) && !h.contains(5) && !h.contains(299));
            h
        },
        KvHistory::contiguous(10),
    ];
    let mask = batch_chunk_mask_f16(bc, &hists, -100.0).expect("one history per request row");
    let m = mask.wire_bytes();
    assert_eq!(m.len(), bc.mask_elems() * 2);
    let valid = |row: u32, slot: u32| {
        let e = bc.mask_elem(row, slot).expect("in the block");
        half::f16::from_le_bytes([m[e * 2], m[e * 2 + 1]]).to_f32() == 0.0
    };
    let per = j * 256;
    // ⭐ THE TALLIES ARE THE BYTES' OWN VALIDITY — derived independently of the minter's runs
    // walk, at the minter's own (row, 64-lane slab) pitch; every count must agree.
    let slabs = bc.mask_cols() as usize / 64;
    let mut tallies = vec![0u32; bc.mask_rows() as usize * slabs];
    for row in 0..bc.mask_rows() {
        let r = row % seqs;
        for slot in 0..bc.mask_cols() {
            let (owner, local) = bc.slot_of(slot).expect("inside the chunk");
            let want = owner == r && hists[r as usize].contains(local as usize);
            assert_eq!(
                valid(row, slot),
                want,
                "row {row} (request {r}) at slot {slot} (request {owner}, local {local})"
            );
            assert!(local < per);
            if want {
                tallies[row as usize * slabs + slot as usize / 64] += 1;
            }
        }
    }
    for (flat, &t) in tallies.iter().enumerate() {
        assert_eq!(
            mask.window_valid(flat / slabs, flat % slabs).get(),
            t,
            "flat window-row {flat}: the count must be the bytes' own validity, minted by the \
             same runs walk that wrote them"
        );
    }
    assert!(batch_chunk_mask_f16(bc, &hists[..3], -100.0).is_none());
}

/// ⭐⭐⭐ THE SWEEP READING at the chunk family's own door: a row whose whole slab sweep is empty
/// is a request that holds none of its own run's slots — the never-prefilled desync the fold's
/// `l` would normalize, because the row's `m` never rises past the sentinel seed to expunge it.
/// A healthy batch sweeps one on every row; the named row maps to its request by the chunk's
/// own law door.
#[test]
fn the_chunk_family_names_an_empty_sweep_row() {
    let (seqs, j) = (4u32, 3u32);
    let chunk = ChunkScratch::of_chunk(pool_2b(), seqs * j).expect("admitted");
    let bc = BatchChunk::of(chunk, seqs, 4).expect("a whole number of pages per request");
    let healthy = vec![
        KvHistory::contiguous(10),
        KvHistory::contiguous(256),
        KvHistory::contiguous(5),
        KvHistory::contiguous(10),
    ];
    let m = batch_chunk_mask_f16(bc, &healthy, -100.0).expect("one history per request row");
    assert_eq!(
        batch_chunk_empty_sweep_row(&m),
        None,
        "a batch whose every request holds a slot must sweep one on every row"
    );
    // Request 1's history is EMPTY — no run, no shared slot — so every row `i % B == 1`
    // sweeps nothing, and the first of them is named.
    let starved = vec![
        KvHistory::contiguous(10),
        KvHistory::contiguous(0),
        KvHistory::contiguous(5),
        KvHistory::contiguous(10),
    ];
    let m = batch_chunk_mask_f16(bc, &starved, -100.0).expect("one history per request row");
    let row = batch_chunk_empty_sweep_row(&m).expect("request 1's rows sweep nothing");
    assert_eq!(
        bc.row_request(row),
        1,
        "the named row {row} must be request 1's"
    );
}

/// ⭐ THE PAGE LIST LAYS EACH REQUEST's OWN MAP AT `J` PAGES, padding a short map with ITS OWN last page
/// — never another request's — and refuses a map past `J` (a body the selector should not have chosen).
#[test]
fn the_batch_page_list_pads_each_request_with_its_own_last_page() {
    let chunk = ChunkScratch::of_chunk(pool_2b(), 6).expect("admitted");
    let bc = BatchChunk::of(chunk, 2, 4).expect("three pages per request");
    assert_eq!(bc.pages_per_request(), 3);
    assert_eq!(
        bc.page_list(&[vec![7, 9], vec![4, 5, 6]]),
        Some(vec![7, 9, 9, 4, 5, 6])
    );
    assert_eq!(bc.page_list(&[vec![7, 9, 1, 2], vec![4]]), None);
    assert_eq!(bc.page_list(&[vec![7]]), None);
    assert_eq!(bc.page_list(&[vec![], vec![4]]), None);
    assert!(
        BatchChunk::of(chunk, 4, 4).is_none(),
        "6 pages is no whole number per request at 4"
    );
}

/// ⛔ A BATCH CHUNK WHOSE MASK BLOCK IS PAST THE MASK's PLACEMENT DOES NOT BAKE. Every row block reads the
/// block at offset 0; past the placement it would read the next activation's bytes as validity, and in an
/// additive mask an unstaged zero is VALID. The same body bakes against a placement that holds it.
#[test]
fn a_batch_chunk_mask_past_its_placement_is_refused_at_bake() {
    use ktir_superdsc::place::PlaceId;
    use ktir_superdsc::placement::{BundleLayout, SegRole, TensorPlacement};
    let (seqs, pages) = (4u32, 16u32);
    let chunk = ChunkScratch::of_chunk(pool_2b(), pages).expect("admitted");
    let need = BatchChunk::of(chunk, seqs, 4)
        .expect("four pages per request")
        .mask_elems() as u64
        * 2;
    let emit = |mask_bytes: u64| {
        let mut layout = BundleLayout::default();
        let mut names = Vec::new();
        for (i, tid) in (1..=8u32).chain([900]).enumerate() {
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
                    // tid 6 is the prefix mask (`names[5]` below).
                    size: if tid == 6 { mask_bytes } else { 1 << 30 },
                    cover: ktir_superdsc::placement::BindCover::Full,
                },
            );
        }
        let geom = AttnGeometry::<32, 8, 64>::minted();
        let rows = attn_bundle_rows(geom, seqs, true).expect("a decode batch rung");
        let mut sym = 0i64;
        assemble_attn(
            AttnFacts {
                t: 0,
                geom,
                bundle_rows: rows,
                extent: EXTENT,
                qs: &names[0],
                new_k_scaled: &names[1],
                new_v: &names[2],
                kct: &names[3],
                vc: &names[4],
                pmask: &names[5],
                cmask: &names[6],
                kv_block_index: Some(&names[7]),
                chunk: Some(chunk),
                out_id: PlaceId::Act(900),
                rows_are_requests: true,
                headmajor_handoff: false,
            },
            &mut sym,
            Some(&layout),
        )
        .map(|_| ())
        .map_err(|e| e.0)
    };
    assert_eq!(emit(need), Ok(()));
    let refused = emit(need - 128).expect_err("a mask block past its placement must not bake");
    assert!(refused.contains("past the mask"), "{refused}");
}
