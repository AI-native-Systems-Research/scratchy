// SPDX-License-Identifier: Apache-2.0
//! ⭐ THE fp32 BROADCAST OPERAND LOWERS — the run-5 wall is deleted. `elementwise_df`'s non-fp16
//! gate used to refuse a broadcast-flagged fp32 op by name ("a BROADCAST operand on an fp32 op");
//! it now routes it to [`elementwise_broadcast`] with the `df`, whose whole stick basis (the Col
//! storage width, the per-row one-stick fold, the Y-mode frame, the tile's out dim) is stated at
//! `df.elems_per_stick()`/`df` instead of the fp16 constants. The operand modes themselves
//! (`In::col`/`In::mb`) are broadcast FLAGS plus a column offset, df-agnostic, so the fp32 form is
//! the fp16 arm's generalization, not a new arm — this is the sampler's noise/inv_t chain
//! (`sample_s690`, the op the refusal named) getting onto the device through the ladder.
//!
//! What this file pins, at the sampler's own shape family (`Mul` of a dense `[rows, cols]` fp32
//! tile by a `[rows, 1]` per-row vector — `noise · inv_t`-shaped):
//!   * the fp32 broadcast lowers as ONE op whose labeledDs_ state the ACTUAL dtype residency
//!     (`wordLength` 4, `dataFormat_` IEEE_FP32, `stickSize_` 32 — the 32-element fp32 stick),
//!     whose Col operand is read RedStick (one stick per row — the `-2` scale), and whose
//!     iteration space is the full tile;
//!   * the fp16 CONTROL at the same shape keeps every fp16 number (`wordLength` 2,
//!     `SEN169_FP16`, `stickSize_` 64) — the generalization is byte-identical at `Df::Fp16`,
//!     because every `df`-stated expression reduces to the fp16 constant it replaced;
//!   * the refusals that STAY, by name: a per-row one-stick fp32 output (the one-stick chain
//!     rides lane 0 of a 64-element fp16 stick), a corner on any operand of a broadcast arm
//!     (any df — the broadcast mode addresses at the buffer base or the operand's own extent,
//!     neither of which can place a window), and a format that is neither Fp16 nor Fp32.

use ktir_superdsc::emit::lower_ktir_to_superdsc::{BcastAxis, Region, elementwise_df};
use ktir_superdsc::ktir_node::Elementwise;
use ktir_superdsc::superdsc_opspec::Df;

/// The sampler's shape family: 4 request rows × a 256-wide fp32 slice (8 whole fp32 sticks —
/// 4× the fp16 stick count, so a basis error cannot alias onto a passing number).
const ROWS: u32 = 4;
const COLS: u32 = 256;

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

/// The run-5 shape: dense `[ROWS, COLS]` × per-row `[ROWS, 1]` (a `Col` broadcast), at `df`.
fn lower_s690(df: Df) -> Vec<ktir_superdsc::emit::EmittedOp> {
    let dense = region(1, ROWS, COLS, false);
    let col = region(2, ROWS, 1, false);
    let out = region(3, ROWS, COLS, true);
    let mut sym = 0i64;
    elementwise_df(
        "sample_s690",
        Elementwise::Mul,
        &[dense, col, out],
        &[None, Some(BcastAxis::Col)],
        df,
        &mut sym,
        None,
    )
    .unwrap_or_else(|e| {
        panic!(
            "Mul [dense, Col] at {df:?}: the door must lower it: {}",
            e.message
        )
    })
}

fn dsc(ops: &[ktir_superdsc::emit::EmittedOp], op_name: &str) -> serde_json::Value {
    assert_eq!(ops.len(), 1, "one broadcast mul, one op");
    serde_json::to_value(&ops[0].op).unwrap()["dscs_"][0][op_name].clone()
}

#[test]
fn the_fp32_broadcast_operand_lowers() {
    // The dsc key is `{op_func}_o{tid}` with `elementwise_op_func`'s own spelling (`multiply` for
    // `Mul`), while the compute op's `opFuncName` is the dxp wire name (`OpFunc::Multiply.name()`
    // = `mul`) — the vocabulary round-trip `zz_the_compare_select_vocabulary` pins.
    let v = dsc(&lower_s690(Df::Fp32), "multiply_o3");
    // The multiply itself: two input labeledDs, one compute op.
    assert_eq!(v["computeOp_"][0]["opFuncName"], "mul");
    assert_eq!(
        v["computeOp_"][0]["inputLabeledDs"]
            .as_array()
            .map(|a| a.len()),
        Some(2)
    );
    // The iteration space walks the full tile.
    assert_eq!(v["N_"]["mb_"].as_u64(), Some(ROWS as u64));
    assert_eq!(v["N_"]["out_"].as_u64(), Some(COLS as u64));
    // ⭐ EVERY dataspace states the fp32 residency — `wordLength` 4 / IEEE_FP32 — because the `df`
    // rides the `TileOp` into every `TensorArg` (the whole point of the arm: an fp16-annotated
    // descriptor over fp32 buffers would mis-stripe every stick, right shape, wrong
    // `dataFormat_`, silently).
    for lds in v["labeledDs_"].as_array().expect("labeledDs_") {
        assert_eq!(
            lds["wordLength"].as_u64(),
            Some(4),
            "fp32 word length: {lds}"
        );
        assert_eq!(lds["dataFormat_"], "IEEE_FP32", "fp32 data format: {lds}");
    }
    // And the stick basis is the fp32 one (32 elements, not 64) — `primaryDsInfo_` carries the
    // OUTPUT role's layout (the EwOperand builder emits every operand under the OUTPUT dsType,
    // the same at fp16, so INPUT has no entry of its own to read).
    assert_eq!(
        v["primaryDsInfo_"]["OUTPUT"]["stickSize_"].to_string(),
        "[32]"
    );
    // ⭐ THE Col OPERAND IS READ REDSTICK (scale `-2`): one value per row, swept at its own
    // 32-element stick — the operand mode is a broadcast flag plus a column offset, df-agnostic,
    // so the mode survives the generalization verbatim.
    let col_lds = &v["labeledDs_"][1];
    assert!(
        col_lds["scale_"]
            .as_array()
            .expect("scale_")
            .iter()
            .any(|s| s.as_i64() == Some(-2)),
        "the Col operand's stick dim carries RedStick (-2): {}",
        col_lds["scale_"]
    );
    // The dense operand stays Active on every dim.
    assert!(
        v["labeledDs_"][0]["scale_"]
            .as_array()
            .expect("scale_")
            .iter()
            .all(|s| s.as_i64() == Some(1)),
        "the dense operand is Active everywhere: {}",
        v["labeledDs_"][0]["scale_"]
    );
}

#[test]
fn the_fp16_control_keeps_the_fp16_numbers() {
    // The same shape at `Df::Fp16` — the proven arm verbatim: 2-byte words, SEN169_FP16, the
    // 64-element stick. This is the byte-identity half of the generalization: every `df`-stated
    // expression in the arm reduces to the fp16 constant it replaced.
    let v = dsc(&lower_s690(Df::Fp16), "multiply_o3");
    for lds in v["labeledDs_"].as_array().expect("labeledDs_") {
        assert_eq!(
            lds["wordLength"].as_u64(),
            Some(2),
            "fp16 word length: {lds}"
        );
        assert_eq!(lds["dataFormat_"], "SEN169_FP16", "fp16 data format: {lds}");
    }
    assert_eq!(
        v["primaryDsInfo_"]["OUTPUT"]["stickSize_"].to_string(),
        "[64]"
    );
    assert!(
        v["labeledDs_"][1]["scale_"]
            .as_array()
            .expect("scale_")
            .iter()
            .any(|s| s.as_i64() == Some(-2)),
        "the fp16 Col operand is RedStick too: {}",
        v["labeledDs_"][1]["scale_"]
    );
}

#[test]
fn the_fp32_per_row_output_is_still_refused() {
    // A `[rows, 1]` fp32 OUTPUT is refused by name at the gate, before the broadcast route: the
    // per-row one-stick chain rides lane 0 of a 64-element fp16 stick, and the broadcast arm's
    // one-stick fold would state the width at the fp32 stick over a form this door has not proven.
    let dense = region(1, ROWS, 1, false);
    let col = region(2, ROWS, 1, false);
    let out = region(3, ROWS, 1, true);
    let mut sym = 0i64;
    let e = match elementwise_df(
        "sample_per_row",
        Elementwise::Mul,
        &[dense, col, out],
        &[None, Some(BcastAxis::Col)],
        Df::Fp32,
        &mut sym,
        None,
    ) {
        Err(e) => e,
        // `EmittedOp` is not `Debug`, so the refusal is matched rather than `expect_err`-ed.
        Ok(ops) => panic!(
            "the fp32 per-row output must be refused, not emitted ({} ops)",
            ops.len()
        ),
    };
    assert!(
        e.message.contains("per-row one-stick fp32 output"),
        "the refusal names the unproven form: {}",
        e.message
    );
}

#[test]
fn a_corner_on_a_broadcast_operand_is_refused_at_any_df() {
    // The corner guard fires BEFORE the non-fp16 gate, so a broadcast arm with a column corner is
    // refused at fp16 exactly as at fp32 — the broadcast mode addresses its operands at the buffer
    // base or their own extent, neither of which can place a window.
    for df in [Df::Fp16, Df::Fp32] {
        let dense = region(1, ROWS, COLS, false);
        let mut col = region(2, ROWS, 1, false);
        col.c_start = 64;
        let out = region(3, ROWS, COLS, true);
        let mut sym = 0i64;
        let e = match elementwise_df(
            "sample_cornered",
            Elementwise::Mul,
            &[dense, col, out],
            &[None, Some(BcastAxis::Col)],
            df,
            &mut sym,
            None,
        ) {
            Err(e) => e,
            // `EmittedOp` is not `Debug`, so the refusal is matched rather than `expect_err`-ed.
            Ok(ops) => panic!(
                "a corner on a broadcast arm must be refused, not emitted ({} ops)",
                ops.len()
            ),
        };
        assert!(
            e.message.contains("states column corner"),
            "the refusal names the corner: {}",
            e.message
        );
    }
}

#[test]
fn a_format_that_is_neither_fp16_nor_fp32_is_refused() {
    let dense = region(1, ROWS, COLS, false);
    let col = region(2, ROWS, 1, false);
    let out = region(3, ROWS, COLS, true);
    let mut sym = 0i64;
    let e = match elementwise_df(
        "sample_fp8",
        Elementwise::Mul,
        &[dense, col, out],
        &[None, Some(BcastAxis::Col)],
        Df::Fp8,
        &mut sym,
        None,
    ) {
        Err(e) => e,
        // `EmittedOp` is not `Debug`, so the refusal is matched rather than `expect_err`-ed.
        Ok(ops) => panic!(
            "an fp8 pointwise must be refused, not emitted ({} ops)",
            ops.len()
        ),
    };
    assert!(
        e.message.contains("Fp16 and Fp32 only"),
        "the refusal names the two admitted formats: {}",
        e.message
    );
}
