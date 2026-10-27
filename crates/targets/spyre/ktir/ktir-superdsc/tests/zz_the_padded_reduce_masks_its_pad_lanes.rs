// SPDX-License-Identifier: Apache-2.0
//! ⭐ A PADDED REDUCE MASKS ITS PAD LANES — the vendor's SAMV coordinate masking, mirrored
//! field for field from the backend's own multi-row max fixture.
//!
//! A reduce over a width that is not a stick multiple still walks the STICK-PADDED extent
//! (the allocator hands out whole sticks), so the pad lanes ride the reduction as whatever
//! the buffer holds — for `max` over non-positive data that is a zeros-read
//! (`max(−65504, 0, x) = 0`, the measured `mxp=0` signature). The device's fix is the
//! `coordinateMasking_` / `samv-maskvalue` pair: every coordinate past `valid` is fed the
//! mask word — the reduce's IDENTITY — instead of a memory read, so the padding contributes
//! nothing.
//!
//! Pinned here, against `ddc/ddl_templates/test/sdsc_max.json` (the backend's own
//! `[mb=32, 514]` max, stick extent 576):
//!   * `coordinateMasking_: {"out": [[514, 62]]}` — the reduced dim's [valid, pad] in
//!     ELEMENTS (valid + pad = the stick-padded extent);
//!   * the `samv-maskvalue` constant is APPENDED after the op's own constants at the next
//!     insertion index — the vendor's own rule (`constants["samv-maskvalue"] =
//!     _get_mask_value(op)` then `masking_const_id = len(constants)`,
//!     codegen/superdsc.py:2357-2365) — `1` here, after `scaling_factor` at "0" (the
//!     fixture's own id 8 counts ITS eight preceding constants);
//!   * the const word is the RAW `0xFFFE` ("65534") — the DDL's own negInf pattern
//!     (`summeanmaxexx2.ddl:69-72`), NOT the value-true sen169 encoding of −65504
//!     (`0xDE00`). `sum` masks with 0, the additive identity;
//!   * a STICK-ALIGNED reduce emits NO mask fields at all — `coordinateMasking_: {}`,
//!     `maskingConstId_: -1`, `constantInfo_` unchanged — byte-identical to before the
//!     field existed;
//!   * an fp32 padded reduce is REFUSED at the builder: SAMV masking is a ≤16-bit-element
//!     mechanism (the vendor's own element-bit guard in `_get_coordinate_mask`), so no
//!     faithful fp32 descriptor exists — garbage refuses at bake.

use ktir_superdsc::emit::{EmittedOp, emit_sdsc_tiled};
use ktir_superdsc::ir::bridge::tiled_op_sdsc_op::{reduce_opspec, reduce_opspec_df};
use ktir_superdsc::superdsc_opspec::{Df, OpFunc, SdscFoldSet};

const ROWS: u32 = 32;
/// The fixture's own width: 514 logical → 576 stick-padded (pad 62).
const COLS: u32 = 514;

fn emit(op: OpFunc, cols: u32, name: &str) -> EmittedOp {
    let spec = reduce_opspec(op, ROWS, cols, "t_sc", "t_bmax", false)
        .unwrap_or_else(|e| panic!("{name}: {e}"));
    let folds = SdscFoldSet::new(spec.iter.cores_used());
    let mut sym = 0i64;
    emit_sdsc_tiled(name, &spec, &folds, &mut sym, None)
        .unwrap_or_else(|e| panic!("{name}: {}", e.0))
}

fn json(e: &EmittedOp) -> serde_json::Value {
    serde_json::to_value(&e.op).unwrap()
}

/// The op's one `dscs_` entry, keyed by name — where the DSC-level fields live.
fn dsc(e: &EmittedOp, name: &str) -> serde_json::Value {
    json(e)["dscs_"][0][name].clone()
}

#[test]
fn the_padded_max_carries_the_fixture_mask() {
    let name = "attn_bmax_g0_o0";
    let v = dsc(&emit(OpFunc::Max, COLS, name), name);
    // The iteration space walks the PADDED whole-stick extent (the fixture's out_ = 576)...
    assert_eq!(v["N_"]["out_"].as_u64(), Some(576), "padded N_.out_");
    assert_eq!(
        v["coordinateMasking_"].to_string(),
        r#"{"out":[[514,62]]}"#,
        "coordinateMasking_"
    );
    assert_eq!(v["maskingConstId_"].as_i64(), Some(1), "maskingConstId_");
    // The op's own constant still leads at "0"; the mask const follows at the NEXT index.
    assert_eq!(v["constantInfo_"]["0"]["name_"], "scaling_factor");
    let samv = &v["constantInfo_"]["1"];
    assert_eq!(samv["name_"], "samv-maskvalue");
    assert_eq!(samv["dataFormat_"], "SEN169_FP16");
    assert_eq!(samv["data_"]["data_"]["[0, 0, 0]"][0], "65534");
}

#[test]
fn a_sum_masks_with_the_additive_identity() {
    let name = "attn_bsum_g0_o0";
    let v = dsc(&emit(OpFunc::Sum, COLS, name), name);
    assert_eq!(v["maskingConstId_"].as_i64(), Some(1));
    assert_eq!(v["constantInfo_"]["1"]["name_"], "samv-maskvalue");
    assert_eq!(
        v["constantInfo_"]["1"]["data_"]["data_"]["[0, 0, 0]"][0],
        "0"
    );
}

#[test]
fn an_aligned_reduce_emits_no_mask() {
    let name = "attn_bmax_g0_o0";
    let v = dsc(&emit(OpFunc::Max, 512, name), name);
    assert_eq!(v["coordinateMasking_"].to_string(), "{}");
    assert_eq!(v["maskingConstId_"].as_i64(), Some(-1));
    // constantInfo_ carries only the op's own scaling_factor — no samv entry appended.
    assert_eq!(v["constantInfo_"]["0"]["name_"], "scaling_factor");
    assert_eq!(v["constantInfo_"].as_object().map(|o| o.len()), Some(1));
}

#[test]
fn an_fp32_padded_reduce_is_refused() {
    let err = reduce_opspec_df(OpFunc::Mean, ROWS, COLS, "t_sc", "t_bmean", false, Df::Fp32)
        .expect_err("an fp32 padded reduce has no faithful descriptor");
    assert!(
        err.contains("SAMV coordinate masking is fp16-only"),
        "{err}"
    );
}
