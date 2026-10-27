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
use ktir_superdsc::ktir_node::{ReduceAxes, ReduceKind};
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
        // Wall 13's f32 rows live in their own door file; every shape here is the f16
        // emission's own control, byte-identical to the pre-wall-13 basis.
        df: Df::Fp16,
    }
}

fn lower(kind: ReduceKind, rows: u32, cols: u32) -> Vec<EmittedOp> {
    let data = region(7, rows, cols, false);
    let out = region(8, rows, 1, true);
    let mut sym = 0i64;
    // The TRAILING block this shape always was: a rank-2 `[rows, cols]` reduced over its last
    // axis (`Dimensions = [1]`) — the row-reduce form's own statement (wall 12 threaded the
    // op's axes to the door; the classification must keep this path byte-identical).
    let axes = ReduceAxes {
        dims: [1, 0, 0, 0],
        n_dims: 1,
        in_rank: 2,
    };
    reduce("attn_bmax_g0", kind, axes, &[data, out], &mut sym, None).unwrap_or_else(|e| {
        panic!(
            "{kind:?} [{rows}, {cols}]: the door must lower it: {}",
            e.message
        )
    })
}

fn dsc(ops: &[EmittedOp], op_name: &str) -> serde_json::Value {
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
