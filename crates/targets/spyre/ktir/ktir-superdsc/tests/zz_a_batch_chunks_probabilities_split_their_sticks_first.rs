// SPDX-License-Identifier: Apache-2.0
//! ⭐⭐⭐ A DECODE BATCH's `esub` AND `exp` DIVIDE THEIR WORK STICKS FIRST — every core takes all of a row
//! block's rows over a run of whole sticks, and the core-local `cesub` between the two ops is laid out under
//! that same division. Every other chunk keeps the law's rows-first division, so its bundles are unchanged.
//!
//! Emitted from the shipped `assemble_attn` against a REAL layout (the core-local pin reaches the
//! descriptors only through one), at granite-3.1-2b's geometry (hd 64, per-kv-head blocks) and granite-3.1-8b's
//! (hd 128, the batch's query-group blocks), on every rung of every batch width. That the ladder emits at
//! all is the build-time proof that each `exp` reads exactly the slice its `esub` wrote: the emitter's
//! core-local read law refuses any op that divides a core-local tensor differently from its first op.

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

/// The rows a chunk's rows are: one request's decode, a decode batch of `n`, or a prompt chunk of `n`.
#[derive(Clone, Copy, Debug)]
enum Rows {
    One,
    Batch(u32),
    Prompt(u32),
}

/// The chunk at `pages` pages and head dim `HD`, emitted against a real layout whose attention class
/// registry names `HD` (so an hd-128 batch takes the query-group blocks it bakes with).
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

/// `(mb, out)` splits, cores used and `N_` `(mb, out)` of one emitted op.
fn division(name: &str, v: &serde_json::Value) -> ((u64, u64), u64, (u64, u64)) {
    let s = &v["numWkSlicesPerDim_"];
    let n = &v["dscs_"][0][name]["N_"];
    (
        (
            s["mb"].as_u64().unwrap_or(1),
            s["out"].as_u64().unwrap_or(1),
        ),
        v["numCoresUsed_"].as_u64().unwrap_or(0),
        (
            n["mb_"].as_u64().unwrap_or(0),
            n["out_"].as_u64().unwrap_or(0),
        ),
    )
}

/// The sticks-first plan for a `[rows, cols]` fp16 op, by the definition `distribute_cores_sticks_first`
/// is pinned against: of every legal `(out, mb)` split, the most cores, and the widest `out` on a tie.
fn sticks_first(rows: u64, cols: u64) -> (u64, u64) {
    let sticks = cols / 64;
    let (cores, out) = (1..=sticks.min(32))
        .filter(|c| sticks.is_multiple_of(*c))
        .flat_map(|c| {
            (1..=(32 / c).min(rows))
                .filter(move |m| rows.is_multiple_of(*m))
                .map(move |m| (c * m, c))
        })
        .max()
        .unwrap();
    (cores / out, out)
}

/// Every operand of `name` that is a core-local LX tensor: `memOrg_` LX-only, one start address on every
/// core. Returns how many there are.
fn core_local_operands(name: &str, v: &serde_json::Value) -> usize {
    let body = &v["dscs_"][0][name];
    let lds = body["labeledDs_"].as_array().expect("labeled operands");
    lds.iter()
        .filter(|d| {
            d["memOrg_"]
                .as_object()
                .is_some_and(|m| m.len() == 1 && m.contains_key("lx"))
        })
        .count()
}

/// The probability ops of a chunk, `(name, value)`, `esub` then `exp` in emission order.
fn probs(ops: &[(String, serde_json::Value)]) -> Vec<&(String, serde_json::Value)> {
    ops.iter()
        .filter(|(n, _)| n.starts_with("attn_cesub_") || n.starts_with("attn_cexp_"))
        .collect()
}

/// ⭐⭐⭐ AT EVERY RUNG OF EVERY BATCH WIDTH, AT BOTH GEOMETRIES: each `esub` and `exp` is divided sticks first
/// over its `[rows, part]` block, both name the one core-local `cesub`, and the parts still cover each row
/// block's whole chunk exactly once.
#[test]
fn every_batch_rung_divides_its_probabilities_sticks_first() {
    fn check<const HD: u32>() {
        let pool = PagedKvPool::new(NKVH, HD);
        let mut cases = 0usize;
        for seqs in PagedKvPool::BATCH_RUNGS {
            for pages in ChunkScratch::ladder_for(pool, seqs) {
                let ops = emit::<HD>(pages, Rows::Batch(seqs));
                let ps = probs(&ops);
                assert!(!ps.is_empty(), "hd {HD} B={seqs} {pages}p: no esub/exp");
                let esubs = ps
                    .iter()
                    .filter(|(n, _)| n.starts_with("attn_cesub_"))
                    .count();
                assert_eq!(
                    esubs * 2,
                    ps.len(),
                    "hd {HD} B={seqs} {pages}p: one exp per esub"
                );
                let mut covered = 0u64;
                let mut blocks = std::collections::BTreeSet::new();
                for (name, v) in ps {
                    let ((sm, so), cores, (rows, cols)) = division(name, v);
                    assert_eq!(
                        (sm, so),
                        sticks_first(rows, cols),
                        "hd {HD} B={seqs} {pages}p {name}: [{rows}, {cols}] split mb:{sm} out:{so}"
                    );
                    assert_eq!(cores, sm * so, "hd {HD} B={seqs} {pages}p {name}: cores");
                    assert_eq!(
                        core_local_operands(name, v),
                        1,
                        "hd {HD} B={seqs} {pages}p {name}: cesub is its one core-local operand"
                    );
                    if let Some(rest) = name.strip_prefix("attn_cesub_") {
                        covered += cols;
                        // The row block the op is named for: `g{kvh}` or `h{qh}`, before the next `_`.
                        blocks.insert(rest.split('_').next().unwrap_or_default().to_string());
                    }
                }
                // Every row block covers the whole chunk once.
                let n_blocks = blocks.len() as u64;
                assert!(n_blocks > 0, "hd {HD} B={seqs} {pages}p: no row block");
                assert_eq!(
                    covered,
                    n_blocks * u64::from(pages) * 256,
                    "hd {HD} B={seqs} {pages}p: the esub parts cover {covered} slots over {n_blocks} blocks"
                );
                cases += 1;
            }
        }
        assert!(cases >= 10, "hd {HD}: only {cases} batch rungs");
    }
    check::<64>();
    check::<128>();
}

/// The canonical cell's shape, pinned by value: granite-2b's conc-8 batch at 5 pages a request (40 pages,
/// `[32, 10240]` per kv head) runs each `esub`/`exp` as `out:32` with every row on every core. The probe
/// that motivated this measured main's `mb:32` division there at about 32 GB/s per pass.
#[test]
fn the_canonical_conc8_cell_runs_out32() {
    let ops = emit::<64>(40, Rows::Batch(8));
    for (name, v) in probs(&ops) {
        assert_eq!(
            division(name, v),
            ((1, 32), 32, (32, 10240)),
            "{name}: the conc-8 cell's split"
        );
    }
}

/// ⛔ EVERY OTHER CHUNK KEEPS THE LAW'S ROWS-FIRST DIVISION: the one-request decode at both geometries and
/// the shipped prompt chunk width, on every rung. That is the division main emits, so their bundles are
/// unchanged by the batch's.
#[test]
fn the_one_request_and_prompt_chunks_keep_rows_first() {
    fn rows_first(rows: u64, cols: u64) -> (u64, u64) {
        let sm = (1..=rows.min(32))
            .rev()
            .find(|m| rows.is_multiple_of(*m))
            .unwrap();
        let sticks = cols / 64;
        let rest = 32 / sm;
        let so = (1..=sticks.min(rest))
            .rev()
            .find(|o| sticks.is_multiple_of(*o))
            .unwrap();
        (sm, so)
    }
    fn check<const HD: u32>(rows: Rows) {
        for &pages in &ChunkScratch::LADDER {
            let ops = emit::<HD>(pages, rows);
            let ps = probs(&ops);
            assert!(!ps.is_empty(), "hd {HD} {rows:?} {pages}p: no esub/exp");
            for (name, v) in ps {
                let ((sm, so), _, (r, c)) = division(name, v);
                assert_eq!(
                    (sm, so),
                    rows_first(r, c),
                    "hd {HD} {rows:?} {pages}p {name}: [{r}, {c}] split mb:{sm} out:{so}"
                );
            }
        }
    }
    check::<64>(Rows::One);
    check::<128>(Rows::One);
    check::<64>(Rows::Prompt(PagedKvPool::PREFILL_CHUNK_SLOTS as u32));
}
