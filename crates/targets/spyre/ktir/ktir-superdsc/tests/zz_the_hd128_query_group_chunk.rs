// SPDX-License-Identifier: Apache-2.0
//! ⭐ THE hd=128 ONE-REQUEST DECODE CHUNK ATTENDS EACH KV HEAD's WHOLE GQA GROUP — granite-3.1-8b's geometry,
//! from the shipped `assemble_attn` against a real layout whose attention class registry names hd 128.
//!
//! With the query re-laid group-major (`QGroupBlock`), the per-layer chunk is:
//!   * 16 re-layout copies, one per (kv head, slab);
//!   * 8 scores, one per kv head, each `[gqa, hd]·[hd, slots]` — the whole head dim, unbatched;
//!   * one row max, `esub`, `exp` and row sum per kv head over its `[gqa, slots]` block;
//!   * 16 value ops, one per (kv head, slab), each `gqa` rows × one stick, the contraction across the cores.
//!
//! The per-query-head form (32 scores, 64 value ops) is what every chunk without the registry, or with more
//! than one row per head, still emits — pinned by `zz_the_hd128_chunk_emission`.

use ktir_superdsc::ir::bridge::tiled_op_sdsc_op::assemble_attn;
use ktir_superdsc::place::PlaceId;
use ktir_superdsc::placement::{BundleLayout, SegRole, TensorPlacement};
use ktir_superdsc::sdsc_abstract::{
    AttnGeometry, ChunkRowKind, ChunkScratch, PagedKvPool, attn_bundle_rows,
};

const NQH: u32 = 32;
const NKVH: u32 = 8;
const GQA: u32 = NQH / NKVH;
const HD: u32 = 128;
const NSLAB: u32 = 2;
const CAP: u32 = 256;

fn emit(pages: u32, mq: u32, classes: &[u32]) -> Vec<(String, serde_json::Value)> {
    let mut layout = BundleLayout {
        attn_class_hds: classes.to_vec(),
        ..BundleLayout::default()
    };
    let mut names = Vec::new();
    // Every caller-bound operand placed, plus the identity table the re-layout reads.
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
    let rows = if mq == 1 {
        attn_bundle_rows(geom, 1, true).expect("the one-request decode rung")
    } else {
        attn_bundle_rows(geom, mq, false).expect("a prompt chunk's rows")
    };
    let chunk = ChunkScratch::of_chunk(PagedKvPool::new(NKVH as usize, HD as usize), pages)
        .expect("a chunk granite-8b admits");
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
    .unwrap_or_else(|e| panic!("pages={pages} mq={mq}: assemble_attn refused: {}", e.0))
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

fn count(ops: &[(String, serde_json::Value)], stem: &str) -> u32 {
    ops.iter().filter(|(n, _)| n.starts_with(stem)).count() as u32
}

/// The slots of each score part and each value part of a `pages`-page one-request decode chunk: one whole
/// chunk part each, unless its gather stays in LX (`ChunkScratch::lx_form`). Then Kᵗ is cut into 8-page
/// parts when it has no tail of its own, and V into 16-page parts plus its tail.
fn part_slots(pages: u32) -> (Vec<u32>, Vec<u32>) {
    let chunk =
        ChunkScratch::of_chunk(PagedKvPool::new(NKVH as usize, HD as usize), pages).unwrap();
    let cut = |per: u32| -> Vec<u32> {
        (0..pages.div_ceil(per))
            .map(|i| per.min(pages - i * per) * 256)
            .collect()
    };
    match chunk.lx_form(ChunkRowKind::OneRequest) {
        None => (vec![pages * 256], vec![pages * 256]),
        Some(lx) => (
            lx.keys().map_or(vec![pages * 256], |k| cut(k.part_pages())),
            cut(lx.part_pages()),
        ),
    }
}

/// The part an op's `_k{p}` suffix names, part 0 when its leg is one part.
fn part_of(name: &str) -> usize {
    name.split("_k")
        .nth(1)
        .and_then(|s| s.split('_').next())
        .map_or(0, |p| p.parse().unwrap())
}

fn n_of<'a>(name: &str, v: &'a serde_json::Value) -> &'a serde_json::Value {
    &v["dscs_"][0][name]["N_"]
}

/// ⭐ THE OP COUNTS PER LAYER, AT EVERY RUNG: 16 copies, one score per (kv head, Kᵗ part), one softmax op of
/// each kind per kv head, one value op per (kv head, slab, V part) — and no per-query-head op anywhere in
/// the chunk. A rung whose gather stays in HBM is one part of each: 8 scores and 16 value ops.
#[test]
fn an_hd128_decode_chunk_attends_each_kv_heads_whole_group() {
    for &j in &ChunkScratch::LADDER {
        let ops = emit(j, 1, &[HD]);
        let (k_parts, v_parts) = part_slots(j);
        assert_eq!(
            count(&ops, "attn_cqg_g"),
            NKVH * NSLAB,
            "{j} page(s): copies"
        );
        for stem in [
            "attn_csc_h",
            "attn_cbmax_h",
            "attn_cesub_h",
            "attn_cexp_h",
            "attn_cbsum_h",
            "attn_cov_h",
        ] {
            assert_eq!(
                count(&ops, stem),
                0,
                "{j} page(s): no per-query-head {stem}*"
            );
        }
        assert_eq!(
            count(&ops, "attn_csc_g"),
            NKVH * k_parts.len() as u32,
            "{j} page(s): one score per (kv head, Kᵗ part)"
        );
        assert_eq!(
            count(&ops, "attn_cbmax_g"),
            NKVH,
            "{j} page(s): one row max per kv head"
        );
        assert_eq!(
            count(&ops, "attn_cbsum_g"),
            NKVH,
            "{j} page(s): one row sum per kv head"
        );
        let sm_parts = count(&ops, "attn_cesub_g0");
        assert!(sm_parts >= 1, "{j} page(s): kv head 0 has an esub");
        assert_eq!(
            count(&ops, "attn_cesub_g"),
            NKVH * sm_parts,
            "{j} page(s): esub parts"
        );
        assert_eq!(
            count(&ops, "attn_cexp_g"),
            NKVH * sm_parts,
            "{j} page(s): exp parts"
        );
        assert_eq!(
            count(&ops, "attn_cov_g"),
            NKVH * NSLAB * v_parts.len() as u32,
            "{j} page(s): one value op per (kv head, slab, V part)"
        );
    }
}

/// The score contracts the WHOLE head dim over the group's rows and its Kᵗ part (the whole chunk where Kᵗ
/// stays in HBM), unbatched; each value op writes the group's `gqa` rows one stick wide with its V part's
/// contraction divided across the cores.
#[test]
fn the_group_score_and_value_ops_have_the_reference_shape() {
    for &j in &ChunkScratch::LADDER {
        let ops = emit(j, 1, &[HD]);
        let (k_parts, v_parts) = part_slots(j);
        for (name, v) in ops.iter().filter(|(n, _)| n.starts_with("attn_csc_g")) {
            let n = n_of(name, v);
            assert_eq!(
                (
                    n["mb_"].as_u64(),
                    n["in_"].as_u64(),
                    n["out_"].as_u64(),
                    n["y_"].as_u64()
                ),
                (
                    Some(u64::from(GQA)),
                    Some(u64::from(HD)),
                    Some(u64::from(k_parts[part_of(name)])),
                    None
                ),
                "{name} at {j} page(s): N_ = {n}"
            );
        }
        for (name, v) in ops.iter().filter(|(n, _)| n.starts_with("attn_cov_g")) {
            let slots = v_parts[part_of(name)];
            let sticks = u64::from(slots / 64);
            let most = (1..=32u64.min(sticks))
                .rev()
                .find(|s| sticks.is_multiple_of(*s))
                .unwrap_or(1);
            let n = n_of(name, v);
            assert_eq!(
                (n["mb_"].as_u64(), n["out_"].as_u64(), n["in_"].as_u64()),
                (Some(u64::from(GQA)), Some(64), Some(u64::from(slots))),
                "{name} at {j} page(s): N_ = {n}"
            );
            assert_eq!(
                (
                    v["numWkSlicesPerDim_"]["in"].as_u64().unwrap_or(1),
                    v["numCoresUsed_"].as_u64()
                ),
                (most, Some(most)),
                "{name} at {j} page(s): split {}",
                v["numWkSlicesPerDim_"]
            );
        }
        for (name, v) in ops.iter().filter(|(n, _)| n.starts_with("attn_cqg_g")) {
            let n = n_of(name, v);
            assert_eq!(
                (
                    n["y_"].as_u64(),
                    n["mb_"].as_u64(),
                    n["in_"].as_u64(),
                    n["out_"].as_u64()
                ),
                (Some(u64::from(GQA)), Some(1), Some(64), Some(64)),
                "{name}: N_ = {n}"
            );
        }
    }
}

/// ⭐ THE COPIES PRECEDE EVERY READ OF THE RE-LAID QUERY, and the scores read nothing but it: each score's
/// activation is the `cqg` block, at the kv head's own block.
#[test]
fn every_group_score_reads_the_relaid_query_after_its_copies() {
    let ops = emit(8, 1, &[HD]);
    let pos = |stem: &str| ops.iter().position(|(n, _)| n.starts_with(stem));
    let last_copy = ops
        .iter()
        .rposition(|(n, _)| n.starts_with("attn_cqg_g"))
        .expect("copies");
    let first_score = pos("attn_csc_g").expect("scores");
    assert!(
        last_copy < first_score,
        "every copy precedes the first score"
    );
}

/// Without the class registry (an emission fixture) or above one row per head, the chunk keeps the
/// per-query-head form exactly — no copy, no group score.
#[test]
fn the_group_form_needs_the_registry_and_a_one_request_decode() {
    let no_registry = emit(8, 1, &[]);
    assert_eq!(count(&no_registry, "attn_cqg_g"), 0);
    assert_eq!(count(&no_registry, "attn_csc_h"), NQH);
    let prompt = emit(8, PagedKvPool::PREFILL_CHUNK_SLOTS as u32, &[HD]);
    assert_eq!(count(&prompt, "attn_cqg_g"), 0);
    assert_eq!(count(&prompt, "attn_csc_h"), NQH);
}
