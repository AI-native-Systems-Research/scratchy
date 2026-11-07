// SPDX-License-Identifier: Apache-2.0
//! ⭐ THE KTIR DOOR LOWERS THE WIDE MULTI-ROW MAX — the refusal is deleted and the flip is
//! pinned here, because no bake case reaches it (every existing attention/swiglu fixture pins
//! `BLOCK_N = 64`, which dodges the conjunction the guard used to refuse).
//!
//! The guard this test replaces was built on a misread diagnostic: the one measured `mxp = 0`
//! cell (2026-07-28) carried this crate's own row-mixing defect (fixed with `group_stride`,
//! 2026-09-16) AND its unmasked stick padding — the template's own DDL seeds the reduce identity
//! internally (`summeanmaxexx2.ddl:69-72`), so a seed-win would read −65504, not 0. The full
//! corrected law, with the DDL citations and the card gate that arbitrates (dxp_standalone
//! executes no arithmetic), is `lower_ktir_to_superdsc::reduce`'s own doc.
//!
//! What this file pins, per the historical cell's exact shape family:
//!   * `Max` at `[rows>1, cols>64]` — the REFUSED cell — now lowers as ONE op, in the
//!     stick-major rank-2 form (the addressing that was the real defect, since fixed);
//!   * the SAME shape at an UNALIGNED width composes with the samv coordinate masking
//!     (the pad lanes masked with the identity — see
//!     `zz_the_padded_reduce_masks_its_pad_lanes`);
//!   * the two always-legal controls (`Max` at `rows == 1`; `Sum` multi-row wide) still lower.

use ktir_superdsc::emit::EmittedOp;
use ktir_superdsc::emit::lower_ktir_to_superdsc::{Region, reduce};
use ktir_superdsc::ktir_node::ReduceKind;
use ktir_superdsc::superdsc_opspec::Df;

/// The historical cell's shape family: 4 query-head rows × a 512-wide score row (8 sticks).
const ROWS: u32 = 4;
const WIDE: u32 = 512;

fn region(tid: u32, rows: u32, cols: u32, is_out: bool) -> Region {
    Region {
        tid,
        v_rows: rows,
        v_cols: cols,
        r_start: 0,
        c_start: 0,
        r_len: rows,
        c_len: cols,
        r_cover: (0, rows),
        c_cover: (0, cols),
        is_out,
        is_fp8: false,
        band: None,
    }
}

fn lower(
    kind: ReduceKind,
    rows: u32,
    cols: u32,
) -> Vec<EmittedOp<ktir_superdsc::collective::TpWorld<1>>> {
    let data = region(7, rows, cols, false);
    let out = region(8, rows, 1, true);
    let mut sym = 0i64;
    // `Df::Fp16` — the reduce's own proven basis (the fp16 path `reduce`'s doc names
    // byte-identical to its pre-`df` form, the one this file's pinned numbers state).
    reduce("attn_bmax_g0", kind, Df::Fp16, &[data, out], &mut sym, None).unwrap_or_else(|e| {
        panic!(
            "{kind:?} [{rows}, {cols}]: the door must lower it: {}",
            e.message
        )
    })
}

fn dsc(
    ops: &[EmittedOp<ktir_superdsc::collective::TpWorld<1>>],
    op_name: &str,
) -> serde_json::Value {
    assert_eq!(ops.len(), 1, "one reduce, one op");
    serde_json::to_value(&ops[0].op).unwrap()["dscs_"][0][op_name].clone()
}

#[test]
fn the_wide_multi_row_max_lowers() {
    let v = dsc(&lower(ReduceKind::Max, ROWS, WIDE), "max_o8");
    // The reduce itself: a max over data + accum, writing the accum.
    let co = &v["computeOp_"][0];
    assert_eq!(co["opFuncName"], "max");
    assert_eq!(co["inputLabeledDs"].as_array().map(|a| a.len()), Some(2));
    // The iteration space walks the full (aligned) width, and NO mask fields fire.
    assert_eq!(v["N_"]["out_"].as_u64(), Some(WIDE as u64));
    assert_eq!(v["maskingConstId_"].as_i64(), Some(-1));
    assert_eq!(v["coordinateMasking_"].to_string(), "{}");
    // The DATA operand is the stick-major rank-2 form — the addressing whose absence was the
    // real defect behind the measured cell, since fixed by `group_stride`/`head_major`. The
    // rank is the `scale_` vector's length (rank-2: [Active, Active]; the accum: [Active, -2]).
    let data_lds = &v["labeledDs_"][0];
    assert_eq!(
        data_lds["scale_"].as_array().map(|a| a.len()),
        Some(2),
        "the wide multi-row reduce reads its data stick-major rank-2: {}",
        data_lds["scale_"]
    );
    assert_eq!(v["labeledDs_"][1]["scale_"].to_string(), "[1,-2]");
}

#[test]
fn the_padded_wide_max_composes_with_the_mask() {
    // The same cell at the fixture's own unaligned width: the door does not refuse it — the
    // assembler pads up to the stick extent and the pad lanes are masked with the identity.
    let v = dsc(&lower(ReduceKind::Max, ROWS, 514), "max_o8");
    assert_eq!(v["N_"]["out_"].as_u64(), Some(576));
    assert_eq!(v["coordinateMasking_"].to_string(), r#"{"out":[[514,62]]}"#);
    assert_eq!(v["maskingConstId_"].as_i64(), Some(1));
    assert_eq!(v["constantInfo_"]["1"]["name_"], "samv-maskvalue");
    assert_eq!(
        v["constantInfo_"]["1"]["data_"]["data_"]["[0, 0, 0]"][0],
        "65534"
    );
}

#[test]
fn the_controls_still_lower() {
    // `Max` at `rows == 1` (any width) and `Sum` multi-row wide were never refused.
    let v = dsc(&lower(ReduceKind::Max, 1, WIDE), "max_o8");
    assert_eq!(v["computeOp_"][0]["opFuncName"], "max");
    let v = dsc(&lower(ReduceKind::Sum, ROWS, WIDE), "sum_o8");
    assert_eq!(v["computeOp_"][0]["opFuncName"], "sum");
}

/// ⭐⭐ THE PAST-THE-BUDGET REDUCE SPLITS THROUGH THE DOOR ITSELF — the long-reduce split
/// body (`assemble_reduce_split`) is wired into this door, on the door's own facts: a reduce
/// whose full width bills past the usable LX lowers as `time` single-shot slice reduces
/// (slice 0 into the caller's own accum, the rest into their own partial synthetics) plus
/// `time − 1` running combines, in place on the caller's accum. The probe states the door's
/// own single-shot opspec — the same op, rows, cols, df and core division — so a reduce that
/// FITS returns `Ok(None)` inside the door and emits its ONE descriptor byte-identical,
/// which is what every test above this one still pins (the handles' names and kind tags are
/// all the single shot reads; the placement is sized from the tape, never from a door's
/// handles).
///
/// One row, 1,048,545 wide at fp16: the full-width residency bills past the 1,677,721-B
/// budget, half of it fits, so the tiler answers `time = 2` (the same geometry the split
/// body's own lib fixtures pin, `long_reduce_split`).
#[test]
fn the_past_the_budget_reduce_splits_through_the_door() {
    for (kind, combine) in [(ReduceKind::Sum, "add"), (ReduceKind::Max, "maximum")] {
        let ops = lower(kind, 1, 1_048_545);
        assert_eq!(
            ops.len(),
            3,
            "{kind:?}: two slice reduces and their combine"
        );
        let name = format!("{}_o8", kind.op_func());
        let d0 = dsc_of(&ops, &format!("{name}_p0_op"));
        let d1 = dsc_of(&ops, &format!("{name}_p1_op"));
        // Both slices walk their own whole-stick run (524,288 = 8,192 sticks)...
        assert_eq!(
            d0["N_"]["out_"].as_u64(),
            Some(524_288),
            "{kind:?}: slice 0"
        );
        assert_eq!(
            d1["N_"]["out_"].as_u64(),
            Some(524_288),
            "{kind:?}: slice 1"
        );
        // ...the raw remainder (524,257, 31 lanes short) rides the LAST slice, masked with
        // the reduce's own identity — the same samv law the single-shot padded reduce lives by.
        assert_eq!(
            d0["coordinateMasking_"].to_string(),
            "{}",
            "{kind:?}: slice 0 is stick-aligned"
        );
        assert_eq!(
            d1["coordinateMasking_"].to_string(),
            r#"{"out":[[524257,31]]}"#,
            "{kind:?}: the tail pad rides the last slice, masked"
        );
        // THE COMBINE, one per reduce kind — never per shape: `max` takes the maximum
        // (exactly reassociable), every sum-class kind adds.
        let c = dsc_of(&ops, &format!("{name}_c1_op"));
        assert_eq!(
            c["computeOp_"][0]["opFuncName"], combine,
            "{kind:?}: the combine op"
        );
        // The combine runs IN PLACE on the caller's own accum, at one stick per row.
        assert_eq!(
            c["N_"]["out_"].as_u64(),
            Some(64),
            "{kind:?}: the combine's width"
        );
    }
}

/// `dsc` for a door that emitted MORE than one op — keyed by the op's emitted name.
fn dsc_of(ops: &[EmittedOp<ktir_superdsc::collective::TpWorld<1>>], op_name: &str) -> serde_json::Value {
    ops.iter()
        .map(|e| serde_json::to_value(&e.op).unwrap()["dscs_"][0][op_name].clone())
        .find(|v| !v.is_null())
        .unwrap_or_else(|| panic!("no op named {op_name} among the door's {} ops", ops.len()))
}
