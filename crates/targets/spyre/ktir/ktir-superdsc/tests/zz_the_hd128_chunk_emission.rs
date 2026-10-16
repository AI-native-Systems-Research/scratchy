// SPDX-License-Identifier: Apache-2.0
//! The split-KV chunk body at head dim 128 (granite-3.1-8b's geometry), from the shipped `assemble_attn`.

use ktir_superdsc::ir::bridge::tiled_op_sdsc_op::assemble_attn;
use ktir_superdsc::sdsc_abstract::{AttnGeometry, ChunkScratch, PagedKvPool, attn_bundle_rows};

const NQH: u32 = 32;
const NKVH: u32 = 8;
const HD: u32 = 128;
const CAP: u32 = 256;
const NSLAB: u32 = 2;

fn emit(pages: u32, mq: u32) -> Vec<(String, serde_json::Value)> {
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

/// ⭐ ABOVE ONE SLAB THE CHUNK IS CUT PER QUERY HEAD, AND THE VALUE LEG PER (HEAD, SLAB), AT EVERY RUNG.
/// A GQA group's rows interleave the slabs in the token stream, so each query head scores its own
/// `[mq, hd]` (the whole head dim, unbatched), owns a whole `[mq, slots]` score block, and contracts its
/// probabilities against one slab of the slab-major V at a time — one stick of `out` per op, the fold's
/// card-proven hd=128 value form. Nothing is a fold op.
#[test]
fn an_hd128_chunk_scores_per_head_and_values_per_head_slab() {
    for mq in [1, PagedKvPool::PREFILL_CHUNK_SLOTS as u32] {
        for &j in &ChunkScratch::LADDER {
            let ops = emit(j, mq);
            let count = |stem: &str| ops.iter().filter(|(n, _)| n.starts_with(stem)).count() as u32;
            for stem in ["attn_csc_h", "attn_cbmax_h", "attn_cbsum_h"] {
                assert_eq!(
                    count(stem),
                    NQH,
                    "{stem}* at {j} page(s), mq {mq}: one per query head"
                );
            }
            assert_eq!(
                count("attn_csc_g"),
                0,
                "no per-kv-head score above one slab"
            );
            let parts = if mq > 1 {
                j.div_ceil(ChunkScratch::VALUE_PART_PAGES)
            } else {
                1
            };
            assert_eq!(
                count("attn_cov_h"),
                NQH * NSLAB * parts,
                "{j} page(s), mq {mq}: one value op per (query head, slab, part)"
            );
            for (name, v) in ops.iter().filter(|(n, _)| n.starts_with("attn_cov_h")) {
                let n = &v["dscs_"][0][name]["N_"];
                assert_eq!(
                    (n["mb_"].as_u64(), n["out_"].as_u64()),
                    (Some(u64::from(mq)), Some(64)),
                    "{name} at {j} page(s): one head's rows, one stick of out — N_ = {n}"
                );
            }
            for (name, v) in ops.iter().filter(|(n, _)| n.starts_with("attn_csc_h")) {
                let n = &v["dscs_"][0][name]["N_"];
                assert_eq!(
                    (n["mb_"].as_u64(), n["in_"].as_u64(), n["out_"].as_u64()),
                    (
                        Some(u64::from(mq)),
                        Some(u64::from(HD)),
                        Some(u64::from(j * 256))
                    ),
                    "{name} at {j} page(s): one head, the whole head dim, the whole chunk — N_ = {n}"
                );
            }
        }
    }
}

/// ⛔ V's COPIES READ V's OWN INDEX RUN: the slab-major order lives in the sticks after K's, and both
/// gathers stay one copy per stick.
#[test]
fn the_hd128_value_gather_reads_its_own_index_sticks() {
    let chunk = ChunkScratch::of_chunk(PagedKvPool::new(NKVH as usize, HD as usize), 8).unwrap();
    let ops = emit(8, 1);
    let sticks = |stem: &str| -> Vec<u32> {
        ops.iter()
            .filter_map(|(n, _)| n.strip_prefix(stem))
            .filter_map(|rest| rest.split("_s").nth(1))
            .filter_map(|s| s.parse().ok())
            .collect()
    };
    assert_eq!(
        sticks("attn_cgkt_o"),
        (0..chunk.index_sticks()).collect::<Vec<_>>()
    );
    assert_eq!(
        sticks("attn_cgv_o"),
        (chunk.index_sticks()..chunk.table_sticks()).collect::<Vec<_>>()
    );
}
