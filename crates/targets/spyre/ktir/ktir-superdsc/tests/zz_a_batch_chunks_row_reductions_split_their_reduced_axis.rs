// SPDX-License-Identifier: Apache-2.0
//! ⭐⭐⭐ A DECODE BATCH's ROW MAX AND ROW SUM SPLIT THE AXIS THEY REDUCE — every core reduces a run of whole
//! sticks of every row, and the reduce template's own ring merges the partials (`ReduceDivision::
//! ReducedAxis`). Every other chunk keeps rows first, so its bundles are unchanged.
//!
//! Emitted from the shipped `assemble_attn` against a real layout, at granite-3.1-2b's geometry (hd 64) and
//! granite-3.1-8b's (hd 128, the batch's query-group blocks), on every rung of every batch width.

use ktir_superdsc::ir::bridge::tiled_op_sdsc_op::{AttnFacts, assemble_attn};
use ktir_superdsc::place::PlaceId;
use ktir_superdsc::placement::{BundleLayout, SegRole, TensorPlacement};
use ktir_superdsc::sdsc_abstract::{
    AttnGeometry, ChunkScratch, KvExtent, PagedKvPool, SlotCount, attn_bundle_rows,
};

const NQH: u32 = 32;
const NKVH: u32 = 8;
const CAP: u32 = 256;
/// The storage/sweep pair on the boundary's one carrier — the sweep is the full cap (no rung).
const EXTENT: KvExtent = KvExtent::new(SlotCount::new(CAP), SlotCount::new(CAP));

#[derive(Clone, Copy, Debug)]
enum Rows {
    One,
    Batch(u32),
    Prompt(u32),
}

fn emit<const HD: u32>(pages: u32, rows: Rows) -> Vec<(String, serde_json::Value)> {
    let mut layout = BundleLayout {
        attn_class_hds: vec![HD],
        ..BundleLayout::default()
    };
    let ident = ktir_superdsc::reserved_tids::identity_class_tid(0);
    let mut names = Vec::new();
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
                cover: ktir_superdsc::placement::BindCover::Full,
            },
        );
    }
    let geom = AttnGeometry::<NQH, NKVH, HD>::minted();
    let (bundle_rows, requests) = match rows {
        Rows::One => (attn_bundle_rows(geom, 1, true), false),
        Rows::Batch(n) => (attn_bundle_rows(geom, n, true), true),
        Rows::Prompt(n) => (attn_bundle_rows(geom, n, false), false),
    };
    let bundle_rows = bundle_rows.unwrap_or_else(|| panic!("{rows:?}: no rung"));
    let chunk =
        ChunkScratch::of_chunk(PagedKvPool::new(NKVH, HD), pages).expect("a chunk granite admits");
    let mut sym = 0i64;
    assemble_attn(
        AttnFacts {
            t: 0,
            geom,
            bundle_rows,
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
            rows_are_requests: requests,
            headmajor_handoff: false,
        },
        &mut sym,
        Some(&layout),
    )
    .unwrap_or_else(|e| {
        panic!(
            "hd {HD}, {pages} pages, {rows:?}: assemble_attn refused: {}",
            e.0
        )
    })
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

/// `(mb, out)` splits and the reduced extent `N_.out_` of one emitted reduce.
fn split(name: &str, v: &serde_json::Value) -> ((u64, u64), u64, u64) {
    let s = &v["numWkSlicesPerDim_"];
    (
        (
            s["mb"].as_u64().unwrap_or(1),
            s["out"].as_u64().unwrap_or(1),
        ),
        v["dscs_"][0][name]["N_"]["mb_"].as_u64().unwrap_or(0),
        v["dscs_"][0][name]["N_"]["out_"].as_u64().unwrap_or(0),
    )
}

fn row_reductions(ops: &[(String, serde_json::Value)]) -> Vec<&(String, serde_json::Value)> {
    ops.iter()
        .filter(|(n, _)| n.starts_with("attn_cbmax_") || n.starts_with("attn_cbsum_"))
        .collect()
}

/// The most cores a reduced-axis split takes: the largest divisor of the stick count within 32.
fn most(sticks: u64) -> u64 {
    (1..=sticks.min(32))
        .rev()
        .find(|s| sticks.is_multiple_of(*s))
        .unwrap()
}

/// ⭐ AT EVERY RUNG OF EVERY BATCH WIDTH, AT BOTH GEOMETRIES, the row max and row sum split only `out`, as far
/// as whole sticks allow, with every row whole.
#[test]
fn every_batch_rung_splits_its_row_reductions_reduced_axis() {
    fn check<const HD: u32>() {
        let pool = PagedKvPool::new(NKVH, HD);
        let mut cases = 0usize;
        for seqs in PagedKvPool::BATCH_RUNGS {
            for pages in ChunkScratch::ladder_for(pool, seqs) {
                let ops = emit::<HD>(pages, Rows::Batch(seqs));
                let rs = row_reductions(&ops);
                assert!(
                    !rs.is_empty(),
                    "hd {HD} B={seqs} {pages}p: no row reductions"
                );
                for (name, v) in rs {
                    let ((sm, so), _rows, cols) = split(name, v);
                    assert_eq!(
                        (sm, so),
                        (1, most(cols / 64)),
                        "hd {HD} B={seqs} {pages}p {name}: over {cols} slots"
                    );
                }
                cases += 1;
            }
        }
        assert!(cases >= 10, "hd {HD}: only {cases} batch rungs");
    }
    check::<64>();
    check::<128>();
}

/// The canonical conc-8 cell, by value: 2b at 40 pages, `[32, 10240]`, `{mb:1, out:32}`.
#[test]
fn the_canonical_conc8_cell_runs_out32() {
    let ops = emit::<64>(40, Rows::Batch(8));
    for (name, v) in row_reductions(&ops) {
        assert_eq!(split(name, v), ((1, 32), 32, 10240), "{name}");
    }
}

/// ⛔ EVERY OTHER CHUNK KEEPS ROWS FIRST — the reduced axis is never split: the one-request decode at both
/// geometries and the shipped prompt chunk width, on every rung.
#[test]
fn the_one_request_and_prompt_chunks_keep_rows_first() {
    fn check<const HD: u32>(rows: Rows) {
        for &pages in &ChunkScratch::LADDER {
            let ops = emit::<HD>(pages, rows);
            let rs = row_reductions(&ops);
            assert!(
                !rs.is_empty(),
                "hd {HD} {rows:?} {pages}p: no row reductions"
            );
            for (name, v) in rs {
                let ((_, so), _, _) = split(name, v);
                assert_eq!(so, 1, "hd {HD} {rows:?} {pages}p {name}: out split {so}");
            }
        }
    }
    check::<64>(Rows::One);
    check::<128>(Rows::One);
    check::<64>(Rows::Prompt(PagedKvPool::PREFILL_CHUNK_SLOTS as u32));
}
