// SPDX-License-Identifier: Apache-2.0
//! THE BATCH-DECODE WALK ORDER, pinned off the emitted DESCRIPTOR FIELDS — `layoutDimOrder_`,
//! `maxDimSizes_` and the per-core start addresses keyed by each core's own declared work slice.
//!
//! The defect class this pins against: the shared-kernel batched score/value matmul used to declare
//! `[mb, y, in]` / `[mb, y, out]`, whose row-major provenance gives `mb` the batch BLOCK `y·in` and
//! `y` one stick — SWAPPED against the head-major tensors (row pitch `in`, head stride `mb·in`),
//! coincident only at `mb == 1`. The per-core START addresses cannot see it (they were a strict
//! subset of the per-head set), so these tests assert the declared-walk fields themselves: put the
//! swapped order back and the `layoutDimOrder_` pins here go RED at mq 2/4/8.
//!
//! And the SCOPING is pinned with it: the corrected head-outermost walk belongs ONLY to
//! rows-are-requests rungs at mq>1. The mq==1 solo bundle — the one proven on hardware — must keep
//! the batch-inner order and the `-1`-reconstructed walk byte-for-byte, and a non-request row kind
//! must never be handed the corrected order. Only the card can accept the mq>1 emit; these pins keep
//! the proven emit out of its blast radius.

use scratchy_subtile::addr::DevOff;
use scratchy_subtile::sdsc_abstract::{
    BlockCols, KernelTag, MaskRows, MatK, MatM, MatN, MatY, OperandPlacement, PaddedMq,
    PerRequestRows, QueryRowCount, RowBlockedTag, RungWidth, SlotWindow, StickLayout, Stk,
};
use scratchy_target_spyre::ir::bridge::tiled_op_sdsc_op::{
    SharedKernelBmmForm, assemble_attn, assemble_matmul_off,
};
use scratchy_target_spyre::lower_subtile_tape_to_superdsc as superdsc;

const HD: u32 = 64; // the g-form exists only where a head is one stick
const GQA: u32 = 4;
const NQH: u32 = 32;
const PAGE_SLOTS: u32 = 256;

fn act(n: &str, rows: u32, cols: u32) -> Stk<RowBlockedTag> {
    Stk::<RowBlockedTag>::new(n, StickLayout::row_blocked(rows as usize, cols as usize)).unwrap()
}
fn ker(n: &str, rows: u32, cols: u32) -> Stk<KernelTag> {
    Stk::<KernelTag>::new(n, StickLayout::kernel(rows as usize, cols as usize)).unwrap()
}

/// The operand's `layoutDimOrder_` — the field that IS the stride assignment.
fn layout_dim_order(e: &superdsc::EmittedOp, op_name: &str, arg: usize) -> Vec<String> {
    let v = serde_json::to_value(&e.op).unwrap();
    v["dscs_"][0][op_name]["scheduleTree_"][arg]["layoutDimOrder_"]
        .as_array()
        .expect("declared order")
        .iter()
        .map(|x| x.as_str().unwrap().to_string())
        .collect()
}

/// The operand's `maxDimSizes_`: `[-1; rank]` = "reconstruct the stick-blocked walk", pinned
/// extents = "walk row-major over exactly these extents, in `layoutDimOrder_` order".
fn max_dim_sizes(e: &superdsc::EmittedOp, op_name: &str, arg: usize) -> Vec<i64> {
    let v = serde_json::to_value(&e.op).unwrap();
    v["dscs_"][0][op_name]["scheduleTree_"][arg]["maxDimSizes_"]
        .as_array()
        .expect("declared walk")
        .iter()
        .map(|x| x.as_i64().unwrap())
        .collect()
}

/// Per-core start offsets keyed by the core's OWN declared `(y, mb)` slice (`coreIdToWkSlice_`) —
/// never an assumed enumeration order, which once transposed the grid and cancelled the swapped
/// strides into "correct" numbers.
fn per_core_by_slice(
    e: &superdsc::EmittedOp,
    op_name: &str,
    arg: usize,
) -> std::collections::BTreeMap<(u32, u32), i64> {
    let v = serde_json::to_value(&e.op).unwrap();
    let wk = v["coreIdToWkSlice_"]
        .as_object()
        .expect("declared work slices");
    let d = &v["dscs_"][0][op_name]["scheduleTree_"][arg]["startAddressCoreCorelet_"]["data_"];
    let mut rows: Vec<(u32, i64)> = d
        .as_object()
        .expect("per-core start addresses")
        .iter()
        .map(|(k, val)| {
            let core: u32 = k
                .trim_start_matches('[')
                .split(',')
                .next()
                .unwrap()
                .trim()
                .parse()
                .unwrap();
            (core, val.as_str().unwrap().parse::<i64>().unwrap())
        })
        .collect();
    rows.sort();
    let base = rows[0].1;
    let mut offs = std::collections::BTreeMap::new();
    for (core, a) in &rows {
        let slices = wk[&core.to_string()]
            .as_object()
            .expect("a core's slice map");
        let slice = |axis: &str| slices.get(axis).and_then(|x| x.as_u64()).unwrap_or(0) as u32;
        offs.insert((slice("y"), slice("mb")), (a - base) / 2);
    }
    offs
}

/// The g-form score op exactly as `assemble_attn_block` emits it for one kv-head group, with the
/// form the given row kind + count thread through the emitter's boundary.
fn score_op(mq: u32, rows_are_requests: bool) -> superdsc::EmittedOp {
    let mut s = 0i64;
    assemble_matmul_off(
        "sc_pin",
        MatM::of_query_rows(QueryRowCount::of_mq(mq)),
        MatN::of_kv_window(BlockCols::of_slot_window(SlotWindow::SLOTS)),
        MatK::of_head_dim(HD),
        // The batch axis now CARRIES both operands' head strides, so a `y`-batch cannot be asked for
        // without stating where its heads are — see `MatY::of_gqa_group`. The score leg reads the token
        // stream (`qs`, heads `mq*hd` apart) and writes the head-major score buffer (`sc`, heads
        // `mq*stick` apart), and the builder refuses if the declared walk strides by neither.
        MatY::of_gqa_group(
            GQA,
            OperandPlacement::of_token_stream(QueryRowCount::of_mq(mq), NQH, HD, 0, 0),
            OperandPlacement::of_head_major_rows(
                0,
                QueryRowCount::of_mq(mq),
                MaskRows::new(
                    NQH,
                    RungWidth::of_baked_rows(mq).expect("a nonzero rung width"),
                ),
                BlockCols::of_slot_window(SlotWindow::SLOTS),
            ),
        ),
        SharedKernelBmmForm::of_attn_rows(rows_are_requests, QueryRowCount::of_mq(mq)),
        &act("t_qs", mq, HD),
        DevOff::ZERO,
        &ker("t_kct", HD, PAGE_SLOTS),
        DevOff::ZERO,
        &act("t_sc", NQH * mq, 64),
        DevOff::ZERO,
        &mut s,
        None,
    )
}

/// THE CORRECTED WALK at the batch-decode widths: batch OUTERMOST on both rank-3 operands, and the
/// per-`y` step equal to the head-major head pitch `mq·hd` — at every rung, not just where `mb`
/// happens to be 1. Re-swap the order and the first assert goes RED at every width here.
#[test]
fn a_request_batch_declares_y_outermost_with_the_head_major_pitch() {
    for mq in [2u32, 4, 8] {
        let e = score_op(mq, true);
        assert_eq!(
            layout_dim_order(&e, "sc_pin", 0),
            vec!["y", "mb", "in"],
            "mq={mq}: the input walk is batch-OUTERMOST"
        );
        assert_eq!(
            layout_dim_order(&e, "sc_pin", 2),
            vec!["y", "mb", "out"],
            "mq={mq}: the output walk is batch-OUTERMOST"
        );
        // Row-major provenance over `[y, mb, in]`: y's stride is the product of what follows it —
        // `mb·in = mq·hd`, one head plane. The addresses each core declares must realize exactly
        // that, keyed by the core's own (y, mb) slice.
        let qs = per_core_by_slice(&e, "sc_pin", 0);
        let sc = per_core_by_slice(&e, "sc_pin", 2);
        for (pc, name) in [(&qs, "qs"), (&sc, "sc")] {
            let y_step = pc[&(1, 0)] - pc[&(0, 0)];
            assert_eq!(
                y_step,
                i64::from(mq * HD),
                "mq={mq}: {name}'s per-y step is the head-major head pitch mq·hd"
            );
            let mb_step = pc[&(0, 1)] - pc[&(0, 0)];
            assert_eq!(
                mb_step,
                i64::from(HD),
                "mq={mq}: {name}'s per-mb step is one row"
            );
        }
        // The pinned extents follow the declared order: one gqa group of `mq`-row head planes.
        assert_eq!(
            max_dim_sizes(&e, "sc_pin", 0),
            vec![i64::from(GQA), i64::from(mq), i64::from(HD)],
            "mq={mq}: the pinned walk states the head-outermost extents"
        );
    }
}

/// THE SCOPING: mq==1 — the solo decode bundle proven on hardware — keeps the batch-inner order and
/// the `-1`-reconstructed walk, exactly the descriptors the card has accepted. The corrected order
/// must not reach it.
#[test]
fn the_mq1_proven_bundle_keeps_the_batch_inner_reconstructed_walk() {
    let e = score_op(1, true);
    assert_eq!(
        layout_dim_order(&e, "sc_pin", 0),
        vec!["mb", "y", "in"],
        "mq==1 keeps the on-hardware-proven batch-inner order"
    );
    assert_eq!(
        layout_dim_order(&e, "sc_pin", 2),
        vec!["mb", "y", "out"],
        "mq==1 output likewise"
    );
    assert_eq!(
        max_dim_sizes(&e, "sc_pin", 0),
        vec![-1, -1, -1],
        "mq==1's walk is the stick-blocked reconstruction, as the proven descriptors carry"
    );
}

/// AND A NON-REQUEST ROW KIND CANNOT TAKE THE CORRECTED WALK: the boundary hands a prompt-shaped
/// caller the proven form whatever its width. (The live prefill never emits the g-form at all —
/// `assemble_attn`'s gate keeps mq>1 prompts on the per-head path — so this is the boundary's own
/// pin, not a claim about a shipped prompt bundle.)
/// ⭐ STILL AT `mq = 8`, WHICH IS THE ONLY WIDTH WHERE THIS CLAIM HAS TEETH — the two walks coincide at
/// `mq == 1`, so pinning the order there asserts almost nothing.
///
/// What changed is the OPERANDS, not the width. `MatY::of_gqa_group` now carries both operands' head
/// strides and the builder refuses a walk that strides `y` by neither, so a token-stream activation
/// (heads `mq*hd` apart) can no longer be paired with the batch-inner walk (`y` one stick) at `mq > 1`
/// — it refused this very call: "strides `y` by 64 elems … but this operand's placement law puts
/// adjacent heads 512 elems apart". That pairing was always wrong and the live emitter never built it
/// (`assemble_attn` keeps `mq>1` prompts on the per-head path); only this test did.
///
/// So the op gets operands the batch-inner walk DOES fit — the request-major law, where adjacent heads
/// are ONE ROW apart and a row is one stick. The subject here is which FORM the boundary hands back for
/// a prompt row kind, and that is unchanged at width 8.
#[test]
fn a_prompt_row_kind_is_refused_the_corrected_walk() {
    let mut s = 0i64;
    let one_row_per_head = |width: BlockCols| {
        OperandPlacement::of_request_major_rows(
            0,
            0,
            PerRequestRows::of_one_request_heads(NQH),
            width,
        )
    };
    let e = assemble_matmul_off(
        "sc_pin",
        MatM::of_query_rows(QueryRowCount::of_mq(8)),
        MatN::of_kv_window(BlockCols::of_slot_window(SlotWindow::SLOTS)),
        MatK::of_head_dim(HD),
        MatY::of_gqa_group(
            GQA,
            one_row_per_head(BlockCols::of_head_dim(HD)),
            one_row_per_head(BlockCols::of_slot_window(SlotWindow::SLOTS)),
        ),
        SharedKernelBmmForm::of_attn_rows(false, QueryRowCount::of_mq(8)),
        &act("t_qs", 8, HD),
        DevOff::ZERO,
        &ker("t_kct", HD, PAGE_SLOTS),
        DevOff::ZERO,
        &act("t_sc", NQH * 8, 64),
        DevOff::ZERO,
        &mut s,
        None,
    );
    assert_eq!(layout_dim_order(&e, "sc_pin", 0), vec!["mb", "y", "in"]);
    assert_eq!(layout_dim_order(&e, "sc_pin", 2), vec!["mb", "y", "out"]);
}

/// THE WHOLE THREAD, through the real emitter: `assemble_attn` names the form once from
/// `(rows_are_requests, mq)` and every g-form score/value op of a batch rung comes out
/// batch-outermost while the mq==1 bundle's come out batch-inner. A prompt chunk emits no g-form op
/// at all.
#[test]
fn assemble_attn_threads_the_form_to_every_gform_op() {
    let cap = 256u32;
    let emit = |mq: u32, rar: bool| {
        let mut sym = 0i64;
        // The width crosses the same parse boundary the emitter's own entry uses: a request-kind
        // width must be a baked ladder rung, a prompt width takes the runtime law.
        let geom =
            scratchy_subtile::sdsc_abstract::AttnGeometry::<NQH, { NQH / GQA }, HD>::minted();
        let width = scratchy_subtile::sdsc_abstract::attn_bundle_rows(geom, mq, rar)
            .expect("a width the emitter can bake");
        assemble_attn(
            7,
            geom,
            width,
            cap,
            cap,
            "qs",
            "nks",
            "nvr",
            "kct",
            "vc",
            "pmask",
            "cmask",
            scratchy_target_spyre::bundle_code::PlaceId::Act(7),
            rar,
            &mut sym,
            None,
        )
        .expect("attention must emit")
    };
    for (mq, rar, want) in [
        (4u32, true, vec!["y", "mb", "in"]),
        (8, true, vec!["y", "mb", "in"]),
        (1, true, vec!["mb", "y", "in"]),
    ] {
        let ops = emit(mq, rar);
        let gform: Vec<_> = ops.iter().filter(|e| e.op_name.contains("sc_g")).collect();
        assert!(
            !gform.is_empty(),
            "mq={mq}: the batched g-form must be live at hd=64"
        );
        for e in gform {
            assert_eq!(
                layout_dim_order(e, &e.op_name, 0),
                want,
                "mq={mq} rows_are_requests={rar}: {}",
                e.op_name
            );
        }
    }
    // A prompt chunk keeps the per-head path — no g-form op exists for the form to matter to.
    let prompt = emit(31, false);
    assert!(
        prompt.iter().all(|e| !e.op_name.contains("sc_g")),
        "an mq>1 prompt must not emit the batched g-form"
    );
}
