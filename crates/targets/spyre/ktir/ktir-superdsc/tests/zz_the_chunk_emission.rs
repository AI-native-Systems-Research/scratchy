// SPDX-License-Identifier: Apache-2.0
//! The split-KV chunk body's emission, from the shipped `assemble_attn` at granite-3.1-2b's geometry.

use ktir_superdsc::ir::bridge::tiled_op_sdsc_op::assemble_attn;
use ktir_superdsc::sdsc_abstract::{AttnGeometry, ChunkScratch, PagedKvPool, attn_bundle_rows};

const NQH: u32 = 32;
const NKVH: u32 = 8;
const HD: u32 = 64;
const CAP: u32 = 256;

fn emit_chunk(pages: u32) -> Vec<(String, serde_json::Value)> {
    emit_chunk_ops(pages)
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

fn emit_chunk_ops(pages: u32) -> Vec<ktir_superdsc::emit::EmittedOp> {
    let geom = AttnGeometry::<NQH, NKVH, HD>::minted();
    let rows = attn_bundle_rows(geom, 1, true).expect("the one-request decode rung");
    let chunk = ChunkScratch::of_chunk(PagedKvPool::new(NKVH as usize, HD as usize), pages)
        .expect("a chunk granite-2b admits");
    let mut sym = 0i64;
    assemble_attn(
        0,
        geom,
        rows,
        CAP,
        CAP,
        "t_qs",
        "t_new_k",
        "t_new_v",
        "t_kct",
        "t_vc",
        "t_pmask",
        "t_cmask",
        Some("t_kvbi"),
        Some(chunk),
        ktir_superdsc::place::PlaceId::Act(900),
        false,
        false,
        &mut sym,
        None,
    )
    .unwrap_or_else(|e| panic!("pages={pages}: assemble_attn refused: {}", e.0))
}

/// ⭐ THE WHOLE POINT: one launch attends every page, with the SAME compute ops at every chunk width —
/// only the gather's copies (one per index stick) grow with it.
#[test]
fn the_chunk_op_count_is_independent_of_its_width() {
    let counts: Vec<(u32, usize)> = ChunkScratch::LADDER
        .iter()
        .map(|&j| {
            let compute = emit_chunk(j)
                .iter()
                .filter(|(n, _)| !n.starts_with("attn_cgkt_") && !n.starts_with("attn_cgv_"))
                .count();
            (j, compute)
        })
        .collect();
    assert!(
        counts.windows(2).all(|w| w[0].1 == w[1].1),
        "chunk compute ops vary with the chunk width: {counts:?}"
    );
}

/// ⛔ THE CHUNK'S TWO REDUCTIONS ARE PER-ROW, ON STICK-BLOCKED OPERANDS, AT EVERY WIDTH: each row's max
/// (the stabilizer — a group max underflows `o`/`l` in fp16) and each row's denominator are stick-major
/// `[gqa, slots]` reduces, one per kv head. Both data operands walk `-1` (dxp reconstructs the
/// stick-blocked layout from the global shape) — every per-lane page-axis form with a DECLARED row-major
/// walk mis-walked part of each core's rows on card.
#[test]
fn the_chunk_reductions_are_per_row_on_stick_blocked_operands() {
    for &j in &ChunkScratch::LADDER {
        let slots = u64::from(j * 256);
        for (stem, rows, cols) in [("attn_cbmax_g", 4, slots), ("attn_cbsum_g", 4, slots)] {
            let ops: Vec<(String, serde_json::Value)> = emit_chunk(j)
                .into_iter()
                .filter(|(n, _)| n.starts_with(stem))
                .collect();
            assert_eq!(ops.len(), 8, "{stem}* at {j} page(s): one per kv head");
            for (name, v) in &ops {
                let body = &v["dscs_"][0][name];
                let n = &body["N_"];
                assert_eq!(
                    (n["mb_"].as_u64(), n["out_"].as_u64()),
                    (Some(rows), Some(cols)),
                    "{name} at {j} page(s): N_ = {n}"
                );
                let walk: Vec<i64> = body["scheduleTree_"][0]["maxDimSizes_"]
                    .as_array()
                    .map(|a| a.iter().map(|d| d.as_i64().unwrap_or(0)).collect())
                    .unwrap_or_default();
                assert!(
                    walk.iter().all(|&d| d == -1),
                    "{name} at {j} page(s): the data walks {walk:?}, not the reconstructed layout"
                );
            }
        }
    }
}

/// ⛔ EXACTLY THE GATHER COPIES ARE DECLARED GATHERED, AND NOTHING IS A FOLD OP. A gathered op groups on
/// its own (`GroupKind::Gathered`): its launch takes no KV page shift — the entries are absolute — and its
/// body is the one the host stages the index for. An unmarked copy lands in an ordinary group, whose
/// write-page base offsets every absolute entry; on card that was a refusal to run at all ("KvBlockIndex
/// step on a forward with no block table"). And the chunk body launches ONCE, so no op may carry the
/// per-page relaunch.
#[test]
fn only_the_gather_copies_are_declared_gathered_and_nothing_folds() {
    for &j in &ChunkScratch::LADDER {
        for e in emit_chunk_ops(j) {
            let copy = e.op_name.starts_with("attn_cgkt_") || e.op_name.starts_with("attn_cgv_");
            assert_eq!(e.kv_gathered, copy, "{} at {j} page(s)", e.op_name);
            assert!(
                !e.kv_page_fold,
                "{} at {j} page(s) is a fold op — the chunk body launches once",
                e.op_name
            );
        }
    }
}
