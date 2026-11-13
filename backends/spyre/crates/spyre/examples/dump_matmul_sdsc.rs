// SPDX-License-Identifier: Apache-2.0
//! Dump the emitted matmul SDSC for the v_proj shape at mb=8 (prefill, BROKEN on-card:
//! only even output rows land) vs mb=1 (decode, works) to find why mb>1 writes only the
//! first mb row of each core's tile. Bypasses the bit-rotted #[cfg(test)] module.
//!
//!   cargo run -p scratchy-subtile --features spyre --example dump_matmul_sdsc

use ktir_superdsc::emit::{In, assemble_pointwise_broadcast};
use ktir_superdsc::ir::bridge::tiled_op_sdsc_op::assemble_matmul;
use ktir_superdsc::sdsc_abstract::{KernelTag, RowBlockedTag, StickLayout, Stk};

/// The example's one error type: every builder's refusal, and the dump's own reads of the JSON it
/// is handed, end the run with their message instead of panicking at a named `expect`.
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

// Typed handles (dims annotation-only for the pointwise input path; matmul reads its own shape).
fn rbh(n: &str, r: u32, c: u32) -> Result<Stk<RowBlockedTag>> {
    Stk::<RowBlockedTag>::new(n, StickLayout::row_blocked(r as usize, c as usize))
        .map_err(|e| format!("handle {n}: {e:?}").into())
}
fn kh(n: &str, k: u32, no: u32) -> Stk<KernelTag> {
    Stk::<KernelTag>::kernel(k as usize, no as usize, n)
}

/// The descriptor's one op name — the key of its `dscs_[0]` map.
fn op_key(j: &serde_json::Value) -> Result<String> {
    j["dscs_"][0]
        .as_object()
        .and_then(|o| o.keys().next())
        .cloned()
        .ok_or_else(|| "the descriptor has no `dscs_[0]` op".into())
}

fn dump(tag: &str, j: &serde_json::Value) -> Result<()> {
    println!("===================== {tag} =====================");
    println!("numCoresUsed_ = {}", j["numCoresUsed_"]);
    println!("numWkSlicesPerDim_ = {}", j["numWkSlicesPerDim_"]);
    let key = op_key(j)?;
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
    Ok(())
}

fn full(tag: &str, j: &serde_json::Value) -> Result<()> {
    let key = op_key(j)?;
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
        serde_json::to_string_pretty(&dsc["dataStageParam_"])?
    );
    println!(
        "primaryDsInfo_={}",
        serde_json::to_string_pretty(&dsc["primaryDsInfo_"])?
    );
    if let Some(sched) = dsc["scheduleTree_"].as_array() {
        for n in sched {
            println!(
                "  sched node: nodeType_={} ldsIdx_={} component_={} keys={:?}",
                n["nodeType_"],
                n["ldsIdx_"],
                n["component_"],
                n.as_object()
                    .into_iter()
                    .flat_map(|o| o.keys())
                    .collect::<Vec<_>>()
            );
        }
    }
    // computeOp
    println!("computeOp_={}", dsc["computeOp_"]);
    println!();
    Ok(())
}

fn main() -> Result<()> {
    if std::env::args().any(|a| a == "--full") {
        let e = assemble_matmul::<ktir_superdsc::collective::TpWorld<1>>(
            "MM",
            8,
            512,
            2048,
            1,
            &rbh("Tensor0", 8, 2048)?,
            &kh("Tensor1", 2048, 512),
            &rbh("Tensor2", 8, 512)?,
            None,
        )?;
        full("MINE matmul mb=8 v_proj", &serde_json::to_value(&e.op)?)?;
        return Ok(());
    }
    if std::env::args().any(|a| a == "--complete") {
        // COMPLETE dsc JSON (every field) for the field-by-field diff vs the reference.
        let e = assemble_matmul::<ktir_superdsc::collective::TpWorld<1>>(
            "MM",
            8,
            512,
            2048,
            1,
            &rbh("Tensor0", 8, 2048)?,
            &kh("Tensor1", 2048, 512),
            &rbh("Tensor2", 8, 512)?,
            None,
        )?;
        let j = serde_json::to_value(&e.op)?;
        let key = op_key(&j)?;
        println!("{}", serde_json::to_string_pretty(&j["dscs_"][0][&key])?);
        return Ok(());
    }
    // POINTWISE multiply [8, 2048] (like rmsnorm rmgamma — WORKS on-card, 8 rows) — its per-core
    // OUT-split addresses tell us whether the working path uses FLAT or STICKED out addressing.
    {
        let (a0, a1) = (rbh("Tensor0", 8, 2048)?, rbh("Tensor1", 8, 2048)?);
        let ops = [In::full(&a0).ew(), In::full(&a1).ew()];
        let mut sib: i64 = 0;
        let e = assemble_pointwise_broadcast::<ktir_superdsc::collective::TpWorld<1>>(
            "PW",
            "multiply",
            ktir_superdsc::sdsc_abstract::RowCount::of_token_rows(8),
            ktir_superdsc::sdsc_abstract::BlockCols::of_feature_cols(2048),
            &ops,
            &rbh("Tensor2", 8, 2048)?,
            &mut sib,
            None,
        )?;
        let j = serde_json::to_value(&e.op)?;
        dump("POINTWISE multiply [8,2048]", &j)?;
    }
    for m in [1u32, 8] {
        // v_proj: A[m, 2048] · W[2048, 512] -> O[m, 512]  (out=512=8 sticks → mb SPLIT to fill cores)
        let e = assemble_matmul::<ktir_superdsc::collective::TpWorld<1>>(
            "MM",
            m,
            512,
            2048,
            1,
            &rbh("Tensor0", m, 2048)?,
            &kh("Tensor1", 2048, 512),
            &rbh("Tensor2", m, 512)?,
            None,
        )?;
        dump(
            &format!("MATMUL v_proj mb={m} [m,512] k=2048"),
            &serde_json::to_value(&e.op)?,
        )?;
    }
    // q_proj: out=2048=32 sticks → mb UNSPLIT (row_cores=1) like the pointwise. Does it still fail?
    let e = assemble_matmul::<ktir_superdsc::collective::TpWorld<1>>(
        "MM",
        8,
        2048,
        2048,
        1,
        &rbh("Tensor0", 8, 2048)?,
        &kh("Tensor1", 2048, 2048),
        &rbh("Tensor2", 8, 2048)?,
        None,
    )?;
    dump(
        "MATMUL q_proj mb=8 [8,2048] k=2048 (mb UNSPLIT)",
        &serde_json::to_value(&e.op)?,
    )?;
    Ok(())
}
