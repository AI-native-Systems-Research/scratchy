// SPDX-License-Identifier: Apache-2.0
//! ⭐ A SEEDED REDUCE SPLIT ALONG THE AXIS IT REDUCES (`ReduceDivision::ReducedAxis`): every core reduces a
//! whole-stick run of every row, and `summeanmaxexx2.ddl`'s own `genericpartialreduction` ring combines
//! the partials into the one accum.
//!
//! Pinned here, for the decode batch chunk's row max and row sum at granite's conc-8 cell
//! (`[32, 10240]`, stick-major):
//!   * the descriptor splits `out` 32 ways, rows whole, and every core's DATA start is its own run of 5
//!     sticks (`5 · 32 · 64` elements apart), while every core's ACCUM start is the same (the ring's shared
//!     output);
//!   * the emitter's disjoint-output guard admits the shared accum only for an op whose template carries
//!     the combine: a `mean` (absent from the table) is refused at the builder.
//!
//! The pod's `dxp_standalone -d <dir>` compile of the staged groups is the next gate (`stage_*`): the
//! reduced-axis pair, and the rows-first pair for contrast.

use ktir_superdsc::emit::{EmittedOp, emit_sdsc_tiled};
use ktir_superdsc::ir::bridge::tiled_op_sdsc_op::{
    ReduceDivision, ReduceOffSpec, reduce_opspec_off_divided,
};
use ktir_superdsc::superdsc_opspec::{OpFunc, OpSpec, SdscFoldSet};

const ROWS: u32 = 32;
const COLS: u32 = 10240;

/// The conc-8 cell's `[32, 10240]` score block reduced into a one-stick-per-row accum, at offset 0.
fn spec(op: OpFunc, division: ReduceDivision) -> Result<OpSpec, String> {
    reduce_opspec_off_divided(ReduceOffSpec {
        op,
        rows: ROWS,
        cols: COLS,
        data_name: "t_sc",
        accum_name: "t_bmax",
        data_off: 0,
        accum_off: 0,
        head_major: false,
        division,
    })
}

fn emit_divided(op: OpFunc, name: &str, division: ReduceDivision) -> EmittedOp {
    let spec = spec(op, division).unwrap_or_else(|e| panic!("{name}: {e}"));
    let folds = SdscFoldSet::new(spec.iter.cores_used());
    let mut sym = 0i64;
    emit_sdsc_tiled(name, &spec, &folds, &mut sym, None)
        .unwrap_or_else(|e| panic!("{name}: {}", e.0))
}

fn emit(op: OpFunc, name: &str) -> EmittedOp {
    emit_divided(op, name, ReduceDivision::ReducedAxis)
}

fn json(e: &EmittedOp) -> serde_json::Value {
    serde_json::to_value(&e.op).unwrap()
}

/// Per-core start address of operand `idx` (0 = data, 1 = accum), read off its allocate node.
fn starts(v: &serde_json::Value, name: &str, idx: usize) -> Vec<i64> {
    let tree = v["dscs_"][0][name]["scheduleTree_"]
        .as_array()
        .expect("schedule tree");
    let node = tree
        .iter()
        .find(|n| n["ldsIdx_"].as_u64() == Some(idx as u64) && n["component_"] == "hbm")
        .expect("an hbm allocate");
    let data = node["startAddressCoreCorelet_"]["data_"]
        .as_object()
        .expect("starts");
    let mut by_core: Vec<(u32, i64)> = data
        .iter()
        .map(|(k, s)| {
            let core: u32 = k
                .trim_start_matches('[')
                .split(',')
                .next()
                .unwrap()
                .trim()
                .parse()
                .unwrap();
            (core, s.as_str().unwrap().parse().unwrap())
        })
        .collect();
    by_core.sort();
    by_core.into_iter().map(|(_, a)| a).collect()
}

#[test]
fn the_row_max_splits_its_reduced_axis_and_shares_its_accum() {
    for (op, name) in [
        (OpFunc::Max, "attn_cbmax_g0_o0"),
        (OpFunc::Sum, "attn_cbsum_g0_o0"),
    ] {
        let e = emit(op, name);
        let v = json(&e);
        let s = &v["numWkSlicesPerDim_"];
        assert_eq!(
            (s["out"].as_u64(), s["mb"].as_u64().unwrap_or(1)),
            (Some(32), 1),
            "{name}: split {s}"
        );
        assert_eq!(v["numCoresUsed_"].as_u64(), Some(32), "{name}: cores");
        // Data: core c's run starts 5 sticks of all 32 rows after core c-1's (stick-major), i.e. 10240 B.
        let d = starts(&v, name, 0);
        assert_eq!(d.len(), 32, "{name}: one data start per core");
        let step = (COLS / 32) * ROWS * 2;
        for w in d.windows(2) {
            assert_eq!(w[1] - w[0], step as i64, "{name}: data starts {d:?}");
        }
        // Accum: every core addresses the same one-stick-per-row accum — the ring's shared output.
        let a = starts(&v, name, 1);
        assert!(
            a.windows(2).all(|w| w[0] == w[1]),
            "{name}: accum starts are not shared: {a:?}"
        );
    }
}

#[test]
fn a_reduce_whose_template_cannot_combine_is_refused() {
    let err = spec(OpFunc::Mean, ReduceDivision::ReducedAxis)
        .expect_err("a mean's template is not in the combine table");
    assert!(err.contains("combines the cores' partials"), "{err}");
}

#[test]
fn the_row_division_is_unchanged() {
    let spec = spec(OpFunc::Max, ReduceDivision::Rows).unwrap();
    let plain = ktir_superdsc::ir::bridge::tiled_op_sdsc_op::reduce_opspec_off(
        OpFunc::Max,
        ROWS,
        COLS,
        "t_sc",
        "t_bmax",
        0,
        0,
        false,
    )
    .unwrap();
    assert_eq!(spec.iter.splits(), plain.iter.splits());
    assert_eq!(spec.iter.split_of("mb"), 32);
}

/// One staged group for `dxp_standalone -d <dir>` (the dump's own dir contract).
fn stage(dir_name: &str, division: ReduceDivision) {
    let dir = std::env::temp_dir().join(dir_name);
    std::fs::create_dir_all(&dir).expect("probe dir");
    let mut bundle = String::from("module {\n  func.func @sdsc_bundle() {\n");
    for (i, (op, name)) in [
        (OpFunc::Max, "attn_cbmax_g0_o0"),
        (OpFunc::Sum, "attn_cbsum_g0_o0"),
    ]
    .into_iter()
    .enumerate()
    {
        let v = json(&emit_divided(op, name, division));
        let mut wrapped = serde_json::Map::new();
        wrapped.insert(name.to_string(), v);
        std::fs::write(
            dir.join(format!("sdsc_{i}.json")),
            serde_json::to_vec_pretty(&serde_json::Value::Object(wrapped)).unwrap(),
        )
        .unwrap();
        bundle.push_str(&format!(
            "    sdscbundle.sdsc_execute () {{sdsc_filename=\"sdsc_{i}.json\"}}\n"
        ));
    }
    bundle.push_str("    return\n  }\n}\n");
    std::fs::write(dir.join("bundle.mlir"), bundle).unwrap();
    eprintln!("probe group {dir_name} staged at {}", dir.display());
}

/// Stage the form-A probe group for the pod (the reduced-axis pair) and the rows-first pair as its
/// control: the two compiled programs are compared on the pod.
#[test]
fn stage_the_reduced_axis_probe_for_the_pod() {
    stage("reduced-axis-probe", ReduceDivision::ReducedAxis);
    stage("rows-first-probe", ReduceDivision::Rows);
}
