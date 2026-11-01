// SPDX-License-Identifier: Apache-2.0
//! A two-rank world's prompt-chunk attention, kv head by kv head, is the one-rank world's.
//!
//! Tensor parallelism splits granite-3.1-2b's attention by head: each rank holds 16 query heads over 4
//! kv heads where the one-rank forward holds 32 over 8. Every GQA group is the same `gqa = 4` heads of
//! `hd = 64`, so each kv head's share of a prompt chunk must be the same computation in both worlds:
//! the same ops, each with the same work division. Only the ops that span every head at once see the
//! head count.

use ktir_superdsc::ir::bridge::tiled_op_sdsc_op::assemble_attn;
use ktir_superdsc::sdsc_abstract::{AttnGeometry, ChunkScratch, PagedKvPool, attn_bundle_rows};

const CAP: u32 = 256;

fn emit<const NQH: u32, const NKVH: u32, const HD: u32>(
    mq: u32,
    pages: u32,
) -> Vec<(String, serde_json::Value)> {
    let geom = AttnGeometry::<NQH, NKVH, HD>::minted();
    let rows = attn_bundle_rows(geom, mq, false).expect("a prompt chunk's rows");
    let chunk = ChunkScratch::of_chunk(PagedKvPool::new(NKVH as usize, HD as usize), pages)
        .expect("a chunk the geometry admits");
    let mut sym = 0i64;
    let ops = assemble_attn(
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
    .unwrap_or_else(|e| panic!("assemble_attn refused: {}", e.0));
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

/// The work-division facts of one op: how many cores, which slice each takes, the op's own extents
/// and per-core stage, where each operand lives (HBM, LX), and its schedule.
fn division(name: &str, v: &serde_json::Value) -> [serde_json::Value; 9] {
    [
        v["numCoresUsed_"].clone(),
        v["numCoreletsUsed_"].clone(),
        v["coreIdToWkSlice_"].clone(),
        v["dscs_"][0][name]["N_"].clone(),
        v["dscs_"][0][name]["dataStageParam_"].clone(),
        v["dscs_"][0][name]["computeOp_"].clone(),
        v["dscs_"][0][name]["labeledDs_"].clone(),
        v["dscs_"][0][name]["primaryDsInfo_"].clone(),
        v["dscs_"][0][name]["scheduleTree_"].clone(),
    ]
}

/// Every op of the two-rank chunk that the one-rank chunk also emits under the same name — each kv
/// head's (`…{k}…` for the two-rank world's 4 heads, a subset of the one-rank world's 8) and the
/// gathers — divides its work exactly as the one-rank op does, at the widest prompt chunk over a
/// two-page prefix. The all-head ops are the ones that may differ, and only in their row extent.
#[test]
fn each_kv_heads_ops_divide_their_work_as_at_one_rank() {
    let one = emit::<32, 8, 64>(512, 2);
    let two = emit::<16, 4, 64>(512, 2);
    let mut shared = 0;
    let mut all_head: Vec<&str> = Vec::new();
    for (n, v2) in &two {
        let (_, v1) = one
            .iter()
            .find(|(m, _)| m == n)
            .unwrap_or_else(|| panic!("the two-rank chunk emits {n}, which the one-rank does not"));
        let (d1, d2) = (division(n, v1), division(n, v2));
        if d1 == d2 {
            shared += 1;
            continue;
        }
        // An op that differs must span every head: its rows are the heads' rows, halved.
        let rows = |v: &serde_json::Value| v["dscs_"][0][n.as_str()]["N_"]["mb_"].as_u64();
        assert_eq!(
            rows(v1).map(|r| r / 2),
            rows(v2),
            "{n} differs between the worlds in more than its row extent:\n one {d1:?}\n two {d2:?}"
        );
        all_head.push(n);
    }
    assert!(shared > 0, "the worlds share no op");
    assert_eq!(
        all_head,
        [
            "attn_cnewm_o0",
            "attn_ccorrsub_o0",
            "attn_ccorre_o0",
            "attn_cmset_o0",
            "attn_cocorr_o0",
            "attn_clcorr_o0",
            "attn_cladd_o0",
        ],
        "the ops that see the head count"
    );
}
