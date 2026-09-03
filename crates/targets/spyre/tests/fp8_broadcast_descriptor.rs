// SPDX-License-Identifier: Apache-2.0
//! SETTLING EVIDENCE (mq>1 fp8 KV=0 investigation): the WIDE `EwOperand::scalar`
//! (mb+out broadcast, e.g. the old `fq_pre_op`'s `act·scalar(1/448)`) emits a
//! broadcast-operand descriptor whose `out` (stick) fold is BYTE-IDENTICAL to the
//! proven-working wide `col` (out_broadcast alone) and narrow `scalar`, and whose
//! `mb` fold is byte-identical to the proven wide `mb_broadcast` (rmg gamma). i.e.
//! the emitter does NOT make the const advance/read-OOB across the 32 output sticks —
//! refuting the "wide scalar reads 0 for m>1" descriptor hypothesis. The on-card
//! fp16-stickmajor-R2 behavior of `out=RedStick` is a separate (bake-only) question.

use scratchy_subtile::sdsc_abstract::{
    BlockCols, Lanes, RowBlockedTag, RowCount, StickLayout, Stk,
};
use scratchy_target_spyre::lower_subtile_tape_to_superdsc as superdsc;
use superdsc::In;
fn h(n: &str) -> Stk<RowBlockedTag> {
    Stk::<RowBlockedTag>::new(n, StickLayout::row_blocked(1, 64)).unwrap()
}

/// The 2nd input (the broadcast operand) of a 2-input pointwise op is `scheduleTree_[1]`.
/// Return its per-dim coordInfo (`{mb:…, out:…}`).
fn broadcast_operand_coordinfo(e: &superdsc::EmittedOp, op_name: &str) -> serde_json::Value {
    let v = serde_json::to_value(&e.op).unwrap();
    v["dscs_"][0][op_name]["scheduleTree_"][1]["coordinates_"]["coordInfo"].clone()
}

#[test]
fn wide_scalar_broadcast_descriptor_matches_proven_forms() {
    // The residual-stream shapes the fp8 quant chain really emits: token rows, feature cols,
    // and the ONE-STICK-wide per-row scale tensors.
    let (m, k) = (
        RowCount::of_token_rows(31),
        BlockCols::of_feature_cols(2048),
    );
    let stk = BlockCols::of_one_stick(Lanes::FP16);

    let mut s = 0i64;
    let wide_scalar = superdsc::assemble_pointwise_broadcast(
        "wide_scalar",
        "mul",
        m,
        k,
        &[In::full(&h("t_act")).ew(), In::scalar(&h("t_inv448")).ew()],
        &h("t_pre"),
        &mut s,
        None,
    );
    let mut s = 0i64;
    let wide_col = superdsc::assemble_pointwise_broadcast(
        "wide_col",
        "mul",
        m,
        k,
        &[In::full(&h("t_act")).ew(), In::col(&h("t_invs")).ew()],
        &h("t_sc"),
        &mut s,
        None,
    );
    let mut s = 0i64;
    let wide_mb = superdsc::assemble_pointwise_broadcast(
        "wide_mb",
        "mul",
        m,
        k,
        &[In::full(&h("t_xn")).ew(), In::mb(&h("t_gamma")).ew()],
        &h("t_out"),
        &mut s,
        None,
    );
    let mut s = 0i64;
    let narrow_scalar = superdsc::assemble_pointwise_broadcast(
        "narrow_scalar",
        "mul",
        m,
        stk,
        &[
            In::full(&h("t_msum")).ew(),
            In::scalar(&h("t_invcols")).ew(),
        ],
        &h("t_mean"),
        &mut s,
        None,
    );

    let ws = broadcast_operand_coordinfo(&wide_scalar, "wide_scalar");
    let wc = broadcast_operand_coordinfo(&wide_col, "wide_col");
    let wm = broadcast_operand_coordinfo(&wide_mb, "wide_mb");
    let ns = broadcast_operand_coordinfo(&narrow_scalar, "narrow_scalar");

    // The `out` (stick) fold — the axis the "reads OOB across 32 sticks" hypothesis was about — is
    // IDENTICAL for the wide scalar-both, the proven wide col, and the narrow scalar (RedStick, one stick,
    // trailing Affine alpha_=0 = no advance). No 32-stick walk is baked into the const.
    assert_eq!(
        ws["out"], wc["out"],
        "wide scalar `out` fold must equal proven wide-col `out` fold"
    );
    assert_eq!(
        ws["out"], ns["out"],
        "wide scalar `out` fold must equal proven narrow-scalar `out` fold"
    );
    // The `mb` fold of the wide scalar-both (RedNonStick, size-1 broadcast) equals the proven wide
    // mb_broadcast (rmg gamma) `mb` fold.
    assert_eq!(
        ws["mb"], wm["mb"],
        "wide scalar `mb` fold must equal proven wide-mb `mb` fold"
    );
    // And it DIFFERS from the wide-col `mb` (Active, 31 rows) — the only axis on which scalar≠col.
    assert_ne!(
        ws["mb"], wc["mb"],
        "wide scalar `mb` (broadcast) must differ from wide-col `mb` (active)"
    );
}
