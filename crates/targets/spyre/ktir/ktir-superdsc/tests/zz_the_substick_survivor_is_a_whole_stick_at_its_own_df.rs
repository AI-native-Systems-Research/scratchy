// SPDX-License-Identifier: Apache-2.0
//! ⭐⭐⭐⭐⭐ WALL 13 — THE STICK BASIS IS THE DATA'S OWN `df`, pinned at the DOOR.
//!
//! The online-softmax `m`/`l` merge geometry — `[J, 32]` reduced over its leading axis to
//! `[32]` — was wall 13's refusal: 32 survivor lanes, HALF an fp16 stick. The kill + ruling
//! (attention-coder, 2026-10-22) dissolved it: a 128-byte stick is 64 `Fp16` lanes OR 32
//! `Fp32` lanes (`DataFormat::ELEMS_PER_STICK`), the merge input `m` IS f32 in the kernel
//! (`batched_decode.py:157`, `tl.full` f32; the f16 cast is after the merge), and the vendor
//! carries the FULL nonstick family at IEEE_FP32 (`summeanmaxexx2_fp32.ddl:40/:43/:46/:49/:55/:57`
//! — `sumnonstick`/`meannonstick`/`maxnonstick`/`minnonstick`/`absmaxnonstick`/`prodnonstick`,
//! on the SAME nonstick layouts as the f16 template, only `%type_fp32` differs).
//!
//! What was an fp16-stick hardcode in `nonstick_geometry` is now the data's own `df`, so the
//! 32 survivors are EXACTLY one whole fp32 stick — zero data movement, the `[J,32]` f32
//! row-major IS the nonstick layout.
//!
//! # WHAT THESE DOOR-LEVEL TESTS PIN (and why door-level)
//!
//! The walk still mints `Df::Fp16` on every region until wall 13's island commit threads each
//! value's own element type (`Region::df`'s doc states the order); a walk-level f32 case
//! cannot exist yet. So the BASIS LAW is pinned here at the door, with hand-stated regions —
//! the same discipline `zz_the_ktir_door_lowers_the_wide_multi_row_max` applies to its shape:
//!
//! * THE PAIR (the auditor's condition): the SAME `[8, 32] → [32]` numels LOWER at
//!   `Df::Fp32` (one whole fp32 stick) and REFUSE BY NAME at `Df::Fp16` (half a stick) —
//!   the basis is the data's own, never a knob.
//! * THE FP32 STRUCTURE vs the vendor's fp32 DDL binds: `maxnonstick`/`sumnonstick` wire
//!   names, both operands `IEEE_FP32` (`wordLength` 4), the survivor stick 32 lanes, and the
//!   `scaling_factor` const at RAW IEEE_FP32 bits (`0x3F800000` = 1.0) — the reduce.rs
//!   mixed-const law (a SEN169 const feeding an fp32 op is a MIXED op → DD2 "Unsupported
//!   result precision conversion").
//! * NO `row_blocked` on the fp32 NONSTICK form (the addendum law): the fp32 template's
//!   nonstick binds are layout-identical to the f16 family's — `row_blocked` is the fp32
//!   STICK form's property (`reduce_opspec_df`), and a nonstick operand here keeps the
//!   element arrangement the f16 nonstick emission carries.
//! * THE f16 CONTROL (no-drift): the same door at `Df::Fp16` over `[8, 64] → [64]` keeps the
//!   f16 emission — `SEN169_FP16`, `wordLength` 2, one whole 64-lane stick — so the df
//!   parameter moved ONLY the df-derived fields.
//!
//! The ddc leg (`matchDdl2Dsc` for real) and the numerics close at the BAKE on card, the
//! same closure law every reduce pin records.

use ktir_superdsc::emit::EmittedOp;
use ktir_superdsc::emit::lower_ktir_to_superdsc::{Region, reduce};
use ktir_superdsc::ktir_node::{ReduceAxes, ReduceKind};
use ktir_superdsc::superdsc_opspec::Df;

/// The `m`/`l` merge's own column count — `S = B·NKVH·GQA = 32` at B=1 (the geometry every
/// decode row of the golden shapes states).
const SURVIVORS: u32 = 32;
/// The merge's row count at B=1: `J = 32/B` slot blocks, each contributing one candidate row.
const J: u32 = 8;

fn region(tid: u32, rows: u32, cols: u32, is_out: bool, df: Df) -> Region {
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
        df,
    }
}

/// One leading-axis reduce through the door: data `[rows, cols]`, output `[1, cols]` — the
/// numels are what the geometry law reads (frame-invariant, wall 11), and the axes state the
/// leading prefix (`[0]` of a rank-2 frame).
fn lower(kind: ReduceKind, rows: u32, cols: u32, df: Df) -> Result<Vec<EmittedOp>, String> {
    let data = region(7, rows, cols, false, df);
    let out = region(8, 1, cols, true, df);
    let mut sym = 0i64;
    let axes = ReduceAxes {
        dims: [0, 0, 0, 0],
        n_dims: 1,
        in_rank: 2,
    };
    reduce("attn_merge", kind, axes, &[data, out], &mut sym, None).map_err(|e| e.message)
}

fn dsc(ops: &[EmittedOp], op_name: &str) -> serde_json::Value {
    assert_eq!(ops.len(), 1, "one reduce, one op");
    serde_json::to_value(&ops[0].op).unwrap()["dscs_"][0][op_name].clone()
}

/// ⭐ THE PAIR'S LOWERING HALF: `[8, 32] → [32]` at the data's own `Df::Fp32` — 32 survivors,
/// EXACTLY one whole fp32 stick — lowers as ONE nonstick descriptor in the vendor's own fp32
/// family, with every df-derived field stating IEEE_FP32.
#[test]
fn the_m_merge_lowers_at_the_datas_own_fp32_stick() {
    let v = dsc(
        &lower(ReduceKind::Sum, J, SURVIVORS, Df::Fp32).unwrap_or_else(|e| {
            panic!("the f32 m-merge geometry must lower at its own stick, refused: {e}")
        }),
        "sumnonstick_o8",
    );
    let c = &v["computeOp_"][0];
    assert_eq!(
        c["opFuncName"], "sumnonstick",
        "the fp32 family's wire name"
    );
    assert_eq!(
        c["exUnit"], "sfp",
        "RCUDD1A has no fp32 PE — fp32 lives on SFP"
    );
    // The geometry: J reduced, 32 survivor lanes = ONE whole fp32 stick.
    assert_eq!(
        v["N_"]["out_"].as_u64(),
        Some(J as u64),
        "the reduced extent"
    );
    assert_eq!(
        v["N_"]["y_"].as_u64(),
        Some(SURVIVORS as u64),
        "the survivor lanes (one whole fp32 stick)"
    );
    // Every operand states the data's own format — the mixed-df op is unwritable by type.
    let body = &v["labeledDs_"];
    for lds in body.as_array().expect("the labeledDs list") {
        assert_eq!(
            lds["dataFormat_"], "IEEE_FP32",
            "a nonstick f32 operand's dataFormat"
        );
        assert_eq!(
            lds["wordLength"].as_u64(),
            Some(4),
            "4 bytes per f32 element"
        );
    }
    // The scaling const at RAW IEEE_FP32 bits (1.0f32 = 0x3F800000) — the reduce.rs
    // mixed-const law: the const's element type MUST match the op's data_format.
    let scale = &v["constantInfo_"]["0"];
    assert_eq!(scale["name_"], "scaling_factor", "the family's own const");
    assert_eq!(
        scale["dataFormat_"], "IEEE_FP32",
        "the const is an fp32 word"
    );
    assert_eq!(
        scale["data_"]["data_"]["[0, 0, 0]"][0], "1065353216",
        "raw f32 bits of 1.0 (0x3F800000), never a SEN169 encoding"
    );
}

/// ⭐ THE PAIR'S MAX TWIN: the same geometry under `max` rides `maxnonstick` — the fp32
/// template's own bind (`summeanmaxexx2_fp32.ddl:46`).
#[test]
fn the_m_merge_max_twin_lowers_fp32() {
    let v = dsc(
        &lower(ReduceKind::Max, J, SURVIVORS, Df::Fp32)
            .unwrap_or_else(|e| panic!("the f32 max twin must lower, refused: {e}")),
        "maxnonstick_o8",
    );
    assert_eq!(
        v["computeOp_"][0]["opFuncName"], "maxnonstick",
        "the fp32 max twin's wire name"
    );
}

/// ⛔ THE PAIR'S REFUSAL HALF: the SAME `[8, 32] → [32]` numels at `Df::Fp16` still refuse BY
/// NAME — 32 lanes are HALF an fp16 stick. This is the auditor's condition pin: the basis is
/// the data's own df, and the same shape fires at one df and refuses at the other. The fix for
/// a real f32 value hitting this refusal is the walk threading its true `df` (`Region::df`),
/// never a wider vocabulary or a re-layout.
#[test]
fn the_same_shape_refuses_at_the_foreign_fp16_basis() {
    let e = match lower(ReduceKind::Sum, J, SURVIVORS, Df::Fp16) {
        Err(e) => e,
        Ok(ops) => panic!(
            "the 32-survivor shape must refuse at the fp16 basis, not lower {} op(s)",
            ops.len()
        ),
    };
    assert!(
        e.contains("SUB-STICK"),
        "the refusal names the sub-stick law, got: {e}"
    );
    assert!(
        e.contains("32 survivor lanes"),
        "the refusal names the survivor width, got: {e}"
    );
    assert!(
        e.contains("SEN169_FP16 stick is 64 lanes"),
        "the refusal names the DATA'S OWN basis it measured against, got: {e}"
    );
}

/// ⛔ NO-DRIFT f16 CONTROL: the door at `Df::Fp16` over `[8, 64] → [64]` — one whole fp16
/// stick — keeps the f16 nonstick emission: `SEN169_FP16`, `wordLength` 2, and the SEN169
/// scaling const. The df parameter moved ONLY the df-derived fields.
#[test]
fn the_f16_control_keeps_the_f16_emission() {
    let v = dsc(
        &lower(ReduceKind::Sum, J, 64, Df::Fp16)
            .unwrap_or_else(|e| panic!("the f16 one-stick control must lower, refused: {e}")),
        "sumnonstick_o8",
    );
    assert_eq!(v["N_"]["y_"].as_u64(), Some(64), "one whole fp16 stick");
    for lds in v["labeledDs_"].as_array().expect("the labeledDs list") {
        assert_eq!(
            lds["dataFormat_"], "SEN169_FP16",
            "the f16 control's format"
        );
        assert_eq!(
            lds["wordLength"].as_u64(),
            Some(2),
            "2 bytes per f16 element"
        );
    }
    assert_eq!(
        v["constantInfo_"]["0"]["dataFormat_"], "SEN169_FP16",
        "the f16 const keeps its SEN169 word"
    );
}
