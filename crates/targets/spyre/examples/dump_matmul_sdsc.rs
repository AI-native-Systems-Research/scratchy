// SPDX-License-Identifier: Apache-2.0
//! Dump the emitted matmul SDSC for the v_proj shape at mb=8 (prefill, BROKEN on-card:
//! only even output rows land) vs mb=1 (decode, works) to find why mb>1 writes only the
//! first mb row of each core's tile. Bypasses the bit-rotted #[cfg(test)] module.
//!
//!   cargo run -p scratchy-subtile --features spyre --example dump_matmul_sdsc

use scratchy_subtile::sdsc_abstract::{KernelTag, RowBlockedTag, StickLayout, Stk};
use scratchy_target_spyre::lower_subtile_tape_to_superdsc::{
    In, assemble_matmul, assemble_pointwise_broadcast,
};
// Typed handles (dims annotation-only for the pointwise input path; matmul reads its own shape).
fn rbh(n: &str, r: u32, c: u32) -> Stk<RowBlockedTag> {
    Stk::<RowBlockedTag>::new(n, StickLayout::row_blocked(r as usize, c as usize)).unwrap()
}
fn kh(n: &str, k: u32, no: u32) -> Stk<KernelTag> {
    Stk::<KernelTag>::kernel(k as usize, no as usize, n)
}

fn dump(tag: &str, j: &serde_json::Value) {
    println!("===================== {tag} =====================");
    println!("numCoresUsed_ = {}", j["numCoresUsed_"]);
    println!("numWkSlicesPerDim_ = {}", j["numWkSlicesPerDim_"]);
    let key = j["dscs_"][0]
        .as_object()
        .unwrap()
        .keys()
        .next()
        .unwrap()
        .clone();
    let dsc = &j["dscs_"][0][&key];
    println!("N_ = {}", dsc["N_"]);
    println!("dataStageParam_ = {}", dsc["dataStageParam_"]);
    if let Some(sched) = dsc["scheduleTree_"].as_array() {
        for node in sched {
            // last tensor = output
            if node["component_"] == serde_json::json!("hbm") {
                println!(
                    "  ldsIdx {} startAddr = {}",
                    node["ldsIdx_"], node["startAddressCoreCorelet_"]["data_"]
                );
            }
        }
    }
    println!();
}

fn full(tag: &str, j: &serde_json::Value) {
    let key = j["dscs_"][0]
        .as_object()
        .unwrap()
        .keys()
        .next()
        .unwrap()
        .clone();
    let dsc = &j["dscs_"][0][&key];
    println!("========== FULL {tag} ==========");
    // top-level + dsc, minus the giant scheduleTree (print node types/keys only)
    println!(
        "numCoresUsed_={} numWkSlicesPerDim_={} coreletFoldProp_={}",
        j["numCoresUsed_"], j["numWkSlicesPerDim_"], j["coreletFoldProp_"]
    );
    println!("N_={}", dsc["N_"]);
    println!(
        "dataStageParam_={}",
        serde_json::to_string_pretty(&dsc["dataStageParam_"]).unwrap()
    );
    println!(
        "primaryDsInfo_={}",
        serde_json::to_string_pretty(&dsc["primaryDsInfo_"]).unwrap()
    );
    if let Some(sched) = dsc["scheduleTree_"].as_array() {
        for n in sched {
            println!(
                "  sched node: nodeType_={} ldsIdx_={} component_={} keys={:?}",
                n["nodeType_"],
                n["ldsIdx_"],
                n["component_"],
                n.as_object()
                    .map(|o| o.keys().cloned().collect::<Vec<_>>())
                    .unwrap_or_default()
            );
        }
    }
    // computeOp
    println!("computeOp_={}", dsc["computeOp_"]);
    println!();
}

fn main() {
    if std::env::args().any(|a| a == "--full") {
        let e = assemble_matmul(
            "MM",
            8,
            512,
            2048,
            1,
            &rbh("Tensor0", 8, 2048),
            &kh("Tensor1", 2048, 512),
            &rbh("Tensor2", 8, 512),
            None,
        );
        full(
            "MINE matmul mb=8 v_proj",
            &serde_json::to_value(&e.op).unwrap(),
        );
        return;
    }
    if std::env::args().any(|a| a == "--complete") {
        // COMPLETE dsc JSON (every field) for the field-by-field diff vs the reference.
        let e = assemble_matmul(
            "MM",
            8,
            512,
            2048,
            1,
            &rbh("Tensor0", 8, 2048),
            &kh("Tensor1", 2048, 512),
            &rbh("Tensor2", 8, 512),
            None,
        );
        let j = serde_json::to_value(&e.op).unwrap();
        let key = j["dscs_"][0]
            .as_object()
            .unwrap()
            .keys()
            .next()
            .unwrap()
            .clone();
        println!(
            "{}",
            serde_json::to_string_pretty(&j["dscs_"][0][&key]).unwrap()
        );
        return;
    }
    // POINTWISE multiply [8, 2048] (like rmsnorm rmgamma — WORKS on-card, 8 rows) — its per-core
    // OUT-split addresses tell us whether the working path uses FLAT or STICKED out addressing.
    {
        let (a0, a1) = (rbh("Tensor0", 8, 2048), rbh("Tensor1", 8, 2048));
        let ops = [In::full(&a0).ew(), In::full(&a1).ew()];
        let mut sib: i64 = 0;
        let e = assemble_pointwise_broadcast(
            "PW",
            "multiply",
            scratchy_subtile::sdsc_abstract::RowCount::of_token_rows(8),
            scratchy_subtile::sdsc_abstract::BlockCols::of_feature_cols(2048),
            &ops,
            &rbh("Tensor2", 8, 2048),
            &mut sib,
            None,
        );
        let j = serde_json::to_value(&e.op).expect("json");
        dump("POINTWISE multiply [8,2048]", &j);
    }
    for m in [1u32, 8] {
        // v_proj: A[m, 2048] · W[2048, 512] -> O[m, 512]  (out=512=8 sticks → mb SPLIT to fill cores)
        let e = assemble_matmul(
            "MM",
            m,
            512,
            2048,
            1,
            &rbh("Tensor0", m, 2048),
            &kh("Tensor1", 2048, 512),
            &rbh("Tensor2", m, 512),
            None,
        );
        dump(
            &format!("MATMUL v_proj mb={m} [m,512] k=2048"),
            &serde_json::to_value(&e.op).unwrap(),
        );
    }
    // q_proj: out=2048=32 sticks → mb UNSPLIT (row_cores=1) like the pointwise. Does it still fail?
    let e = assemble_matmul(
        "MM",
        8,
        2048,
        2048,
        1,
        &rbh("Tensor0", 8, 2048),
        &kh("Tensor1", 2048, 2048),
        &rbh("Tensor2", 8, 2048),
        None,
    );
    dump(
        "MATMUL q_proj mb=8 [8,2048] k=2048 (mb UNSPLIT)",
        &serde_json::to_value(&e.op).unwrap(),
    );
}
