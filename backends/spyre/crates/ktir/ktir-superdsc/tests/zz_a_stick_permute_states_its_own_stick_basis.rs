// SPDX-License-Identifier: Apache-2.0

//! ⭐ A STICK PERMUTE STATES ITS OWN STICK BASIS — the copy forms' relayout tile is stated at the
//! `df` they were handed: the `out` dim is the format's own stick (fp16 = 64 elements, fp8 = 128),
//! never a borrowed fp16 basis. The float8 arm exists for the scatter's relayout
//! (`scatter_run` → [`assemble_stick_permute`]): a paged KV at fp8 scatters rows of 2 sticks when a
//! head is 256 wide, where a 128-wide head's single-stick rows short-circuit the relayout, so no
//! fp8 page ever reached this seam before the fp8 KV cache. Its fixtures:
//!
//! * the match: a float8 permute of 8 rows × 2 sticks emits an `identity` whose `out` dim is the
//!   128-element float8 stick;
//! * the neighbour it must not disturb: the same geometry at fp16 keeps the proven 64-element
//!   basis, op for op.
//!
//! ARITHMETIC elementwise at fp8 is a different door and stays refused BY NAME at
//! `elementwise_df`'s non-fp16 gate — the tile arm here admits the byte-moving forms only. The
//! `zz_` prefix keeps this door at the END of the crate's test list.

use ktir_superdsc::collective::TpWorld;
use ktir_superdsc::emit::EmittedOp;
use ktir_superdsc::ir::bridge::tiled_op_sdsc_op::{
    StickPermute, StickPermuteSpec, assemble_stick_permute,
};
use ktir_superdsc::superdsc_opspec::{DataFormat, Df, Fp16, Fp8, OpFunc};

/// One permute of `rows` rows of `sticks` sticks at `df`, stick-major to row-major, emitted.
fn permute(rows: u32, sticks: u32, df: Df) -> Result<EmittedOp<TpWorld<1>>, String> {
    let mut sid = 0i64;
    assemble_stick_permute::<TpWorld<1>>(
        "permute_fixture",
        "src",
        "dst",
        StickPermuteSpec {
            rows,
            sticks,
            df,
            direction: StickPermute::StickToRow,
            src_elems: 0,
            dst_elems: 0,
        },
        &mut sid,
        None,
    )
}

/// The op's one dsc, as JSON.
fn dsc(op: &EmittedOp<TpWorld<1>>) -> serde_json::Value {
    serde_json::to_value(op.op.as_ref().expect("a descriptor"))
        .expect("json")["dscs_"][0]
        .as_object()
        .and_then(|m| m.values().next())
        .expect("one dsc")
        .clone()
}

#[test]
fn a_float8_stick_permute_states_its_own_stick_basis() {
    let op = permute(8, 2, Df::Fp8).unwrap_or_else(|e| panic!("the float8 permute lowers: {e}"));
    let dsc = dsc(&op);
    assert_eq!(
        dsc["computeOp_"][0]["opFuncName"],
        OpFunc::Identity.name(),
        "the permute is a copy, never a compute"
    );
    assert_eq!(
        dsc["N_"]["out_"].as_u64(),
        Some(Fp8::ELEMS_PER_STICK as u64),
        "the out dim is the float8 stick (128 elements), not a borrowed fp16 basis"
    );
    assert_eq!(
        dsc["N_"]["mb_"].as_u64(),
        Some(8),
        "the permute's rows are the stated rows"
    );
    assert_eq!(
        dsc["N_"]["y_"].as_u64(),
        Some(2),
        "and its sticks the stated sticks"
    );
}

#[test]
fn the_sixteen_bit_permute_keeps_its_own_stick_basis() {
    let op = permute(8, 2, Df::Fp16).unwrap_or_else(|e| panic!("the fp16 permute lowers: {e}"));
    let dsc = dsc(&op);
    assert_eq!(
        dsc["computeOp_"][0]["opFuncName"],
        OpFunc::Identity.name(),
        "the fp16 permute stays the copy it always was"
    );
    assert_eq!(
        dsc["N_"]["out_"].as_u64(),
        Some(Fp16::ELEMS_PER_STICK as u64),
        "the fp16 basis is unchanged by the float8 arm — 64 elements, as proven"
    );
}
