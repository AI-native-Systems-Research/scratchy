// SPDX-License-Identifier: Apache-2.0
//! WHAT A REQUEST AXIS ON THE FOLD'S MATMULS ADDRESSES — read off the emitted per-core addresses
//! rather than argued from the source.
//!
//! Collapsing the prefix fold from `pages × requests` launches to `pages` needs the score and value
//! KERNELS to step one request per unit of the batch axis, i.e. a per-`y` step of exactly
//! `PagedKvPool::request_stride` = `hd * PAGE_SLOTS`. Two readings of the stride rule agree on every
//! op that ships today and differ here, so this settles it from the generated addresses: guessing
//! wrong makes every request read a fraction of the way into request 0's page, which is fluent text
//! built from another conversation's keys and nothing downstream that notices.
//!
//! ⛔ THE FOLD SWEEPS ONE 64-SLOT BLOCK AT A TIME (`width = stick`, `nb = active_cap/stick` blocks per
//! launch), so each kernel's SWEPT extent is 64 while its physical block is the whole page. The
//! `device_extent` override is what states the storage the op is a window into, and without it `y`
//! steps a quarter of a page. That is the content of this file: which dim to declare, and that
//! declaring it works.

use scratchy_subtile::sdsc_abstract::{
    BlockCols, KernelTag, MatK, MatM, MatN, MatY, OperandPlacement, PagedKvPool, PerRequestRows,
    QueryRowCount, RowBlockedTag, SlotWindow, StickLayout, Stk,
};
use scratchy_target_spyre::ir::bridge::tiled_op_sdsc_op::{
    SharedKernelBmmForm, assemble_matmul_batched_off, assemble_matmul_off_phys_m,
};
use scratchy_target_spyre::lower_subtile_tape_to_superdsc as superdsc;

const HD: u32 = 64;
const MQ: u32 = 8; // requests = the batch axis
const STICK: u32 = 64; // the fold's swept width: one 64-slot block
const PAGE_SLOTS: u32 = PagedKvPool::PAGE_SLOTS as u32;

fn act(n: &str, rows: u32, cols: u32) -> Stk<RowBlockedTag> {
    Stk::<RowBlockedTag>::new(n, StickLayout::row_blocked(rows as usize, cols as usize)).unwrap()
}
fn ker(n: &str, rows: u32, cols: u32) -> Stk<KernelTag> {
    Stk::<KernelTag>::new(n, StickLayout::kernel(rows as usize, cols as usize)).unwrap()
}

/// The per-core start addresses of operand `arg` (0=input, 1=kernel, 2=output), in ELEMENTS from the
/// operand's own base, indexed by core.
fn per_core_elems(e: &superdsc::EmittedOp, op_name: &str, arg: usize) -> Vec<i64> {
    let v = serde_json::to_value(&e.op).unwrap();
    let data = &v["dscs_"][0][op_name]["scheduleTree_"][arg]["startAddressCoreCorelet_"]["data_"];
    let mut rows: Vec<(u32, i64)> = data
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
    rows.iter().map(|(_, a)| (a - base) / 2).collect()
}

/// THE DISTINCT OFFSETS, ascending — the addresses the batch axis reaches. Other dims may also be
/// split across cores (the `out` split, when the op has one), so the distinct values are what
/// identify the request steps rather than "every nth core".
fn distinct_ascending(offs: &[i64], want: usize) -> Vec<i64> {
    let mut v: Vec<i64> = offs.to_vec();
    v.sort_unstable();
    v.dedup();
    assert!(
        v.len() >= want,
        "expected at least {want} distinct offsets, got {v:?}"
    );
    v
}

/// THE SCORE LEG: `qs · Kᵀ`, one op per query head, `y` = request. The kernel is `[hd, PAGE_SLOTS]`
/// per request and the op sweeps a 64-slot window of it, so `out` is the dim to declare.
#[test]
fn the_score_kernel_steps_one_request_per_y() {
    let mut s = 0i64;
    let sc = assemble_matmul_batched_off(
        "sc",
        MatM::single_row(),
        MatN::of_kv_window(BlockCols::of_slot_window(SlotWindow::SLOTS)),
        MatK::of_head_dim(HD),
        MatY::of_requests(MQ),
        &act("t_qs", MQ, HD),
        0,
        &ker("t_kct", HD, PAGE_SLOTS),
        0,
        Some(("out", PAGE_SLOTS)),
        &act("t_sc", MQ * 32, STICK),
        0,
        &mut s,
        None,
    );
    let req = (HD * PAGE_SLOTS) as i64; // PagedKvPool::request_stride, in elements
    let ys = distinct_ascending(&per_core_elems(&sc, "sc", 1), MQ as usize);
    for (r, off) in ys.iter().take(MQ as usize).enumerate() {
        assert_eq!(
            *off,
            r as i64 * req,
            "request {r}'s Kᵀ block (offsets {ys:?})"
        );
    }
}

/// THE VALUE LEG: `probs · V`, one op per query head, `y` = request. V is `[PAGE_SLOTS, hd]` per
/// request and the op sweeps 64 of its ROWS, so `in` is the dim to declare — the mirror of the score
/// leg, and the reason the two overrides name different dims.
#[test]
fn the_value_kernel_steps_one_request_per_y() {
    let mut s = 0i64;
    let ov = assemble_matmul_batched_off(
        "ov",
        MatM::single_row(),
        MatN::of_head_dim(HD),
        MatK::of_kv_window(BlockCols::of_slot_window(SlotWindow::SLOTS)),
        MatY::of_requests(MQ),
        &act("t_expb", MQ * 32, STICK),
        0,
        &ker("t_vc", PAGE_SLOTS, HD),
        0,
        Some(("in", PAGE_SLOTS)),
        &act("t_ov", MQ * 32, HD),
        0,
        &mut s,
        None,
    );
    let req = (HD * PAGE_SLOTS) as i64;
    let ys = distinct_ascending(&per_core_elems(&ov, "ov", 1), MQ as usize);
    for (r, off) in ys.iter().take(MQ as usize).enumerate() {
        assert_eq!(
            *off,
            r as i64 * req,
            "request {r}'s V block (offsets {ys:?})"
        );
    }
}

/// ⛔ AND WITHOUT THE DECLARATION IT IS WRONG, not merely different: `y` steps `in * 64`, a QUARTER of
/// a page, so request 1 reads 64 slots into request 0's keys. Pinned so the override cannot be dropped
/// as redundant — this is the failure the two readings of the stride rule differ by.
#[test]
fn an_undeclared_kernel_extent_steps_a_quarter_of_a_page() {
    let mut s = 0i64;
    let sc = assemble_matmul_batched_off(
        "sc_bad",
        MatM::single_row(),
        MatN::of_kv_window(BlockCols::of_slot_window(SlotWindow::SLOTS)),
        MatK::of_head_dim(HD),
        MatY::of_requests(MQ),
        &act("t_qs", MQ, HD),
        0,
        &ker("t_kct", HD, PAGE_SLOTS),
        0,
        None,
        &act("t_sc", MQ * 32, STICK),
        0,
        &mut s,
        None,
    );
    let req = (HD * PAGE_SLOTS) as i64;
    let ys = distinct_ascending(&per_core_elems(&sc, "sc_bad", 1), 2);
    assert_eq!(
        ys[1],
        (HD * STICK) as i64,
        "undeclared: one swept window, not one page"
    );
    assert_eq!(ys[1] * 4, req, "…exactly a quarter of the request stride");
}

// ── THE PER-REQUEST GQA VALUE LEG (`assemble_matmul_off_phys_m`) — one op per kv-head group, `y` =
//    the group's query heads, the request carried by each op's own base offset. The activation's
//    `OperandPlacement` declares its base offset and its row packing in one law-minted value, so this
//    pins both the per-core addresses and the declared walk, exactly as the request-axis tests above
//    pin `kernel_device_extent`. ──

const NQH: u32 = 32; // the per-request framing: one request's nqh rows are the contiguous ones
const GQA: u32 = 4;

/// The input operand's DECLARED on-card walk (`maxDimSizes_`): `[-1; rank]` = "reconstruct the
/// stick-blocked walk from N_/layoutDimOrder_/stickSize_", pinned extents = "walk row-major over a
/// buffer this many rows deep".
fn max_dim_sizes(e: &superdsc::EmittedOp, op_name: &str, arg: usize) -> Vec<i64> {
    let v = serde_json::to_value(&e.op).unwrap();
    v["dscs_"][0][op_name]["scheduleTree_"][arg]["maxDimSizes_"]
        .as_array()
        .expect("declared walk")
        .iter()
        .map(|x| x.as_i64().unwrap())
        .collect()
}

/// The operand's `layoutDimOrder_` — WHICH axis owns which stride is set by nothing but this order,
/// so the walk pin above is only meaningful together with it.
fn layout_dim_order(e: &superdsc::EmittedOp, op_name: &str, arg: usize) -> Vec<String> {
    let v = serde_json::to_value(&e.op).unwrap();
    v["dscs_"][0][op_name]["scheduleTree_"][arg]["layoutDimOrder_"]
        .as_array()
        .expect("declared order")
        .iter()
        .map(|x| x.as_str().unwrap().to_string())
        .collect()
}

/// THE PER-REQUEST VALUE LEG's activation is `expb`, whose head pitch under the request-major row law
/// is ONE ROW: the buffer is one stick wide and adjacent heads sit one row (`STICK` elements) apart, so
/// the placement's pitch — the activation's rows per stick plane — is the law's own head pitch, 1.
///
/// The chunk's `mq` is the pitch of the OTHER activation (`qs`, the token-stream law). Declared here it
/// would claim `expb` packs `mq` rows per plane — the per-core starts still land (the `mb` corner never
/// moves at `mb = 1`), but the head (`y`) plane deepens to `mq` rows and every head past the first is
/// walked `mq`x too far in. That pairing has no spelling: the pitch rides inside the placement the
/// request-major law mints with the offset, so an offset from one law cannot carry another law's pitch.
/// The law's pitch declares exactly the rows the op sweeps, so the DECLARED WALK is the head-outermost
/// row-major sweep `[y, mb, in]` pinned at `[GQA, 1, 64]`: each head plane is the ONE row the law
/// places, and `y` steps one stick per head — the same strides the per-core addresses below realize.
#[test]
fn the_per_request_value_activation_steps_one_row_per_head() {
    let mut s = 0i64;
    let ov = assemble_matmul_off_phys_m(
        "ov_req",
        MatM::single_row(),
        MatN::of_head_dim(HD),
        MatK::of_kv_window(BlockCols::of_slot_window(SlotWindow::SLOTS)),
        // The batch axis carries BOTH operands' head strides now, so the builder can refuse a walk
        // that strides `y` by neither. Under the request-major law adjacent heads are ONE ROW apart,
        // and a row is one stick — which is exactly the `mb_dev * out` this walk derives.
        MatY::of_gqa_group(
            GQA,
            OperandPlacement::of_request_major_rows(
                0,
                0,
                PerRequestRows::of_one_request_heads(NQH),
                BlockCols::of_slot_window(SlotWindow::SLOTS),
            ),
            OperandPlacement::of_request_major_rows(
                0,
                0,
                PerRequestRows::of_one_request_heads(NQH),
                BlockCols::of_head_dim(HD),
            ),
        ),
        // The per-request fold leg exists only for a decode batch, so it carries the batch form —
        // the head-outermost walk the pins below state.
        SharedKernelBmmForm::of_attn_rows(true, QueryRowCount::of_mq(MQ)),
        &act("t_expb", NQH, STICK),
        // Request 0, head 0 of the request-major framing — offset 0 (the law's first row) and the
        // law's one-row head pitch, minted together.
        OperandPlacement::of_request_major_rows(
            0,
            0,
            PerRequestRows::of_one_request_heads(NQH),
            BlockCols::of_slot_window(SlotWindow::SLOTS),
        ),
        &ker("t_vc", PAGE_SLOTS, HD),
        scratchy_subtile::addr::DevOff::ZERO,
        &act("t_ov", NQH, HD),
        scratchy_subtile::addr::DevOff::ZERO,
        &mut s,
        None,
    );
    assert_eq!(
        layout_dim_order(&ov, "ov_req", 0),
        vec!["y", "mb", "in"],
        "the value activation's walk is batch-OUTERMOST — `y` owns the head plane, `mb` the row"
    );
    assert_eq!(
        max_dim_sizes(&ov, "ov_req", 0),
        vec![i64::from(GQA), 1, i64::from(STICK)],
        "the value activation's declared walk pins the law's own extents: a one-row head plane \
         per `y` step, never a plane deeper than the one row per head the law places"
    );
    let ys = distinct_ascending(&per_core_elems(&ov, "ov_req", 0), GQA as usize);
    for (g, off) in ys.iter().take(GQA as usize).enumerate() {
        assert_eq!(
            *off,
            (g as u32 * STICK) as i64,
            "head {g}'s probability row (offsets {ys:?})"
        );
    }
    // The OUTPUT side steps one head-major row per head too (`mb*out = hd` at mb=1) — the pair
    // states the whole per-request relation, both sides one row per head.
    let os = distinct_ascending(&per_core_elems(&ov, "ov_req", 2), GQA as usize);
    for (g, off) in os.iter().take(GQA as usize).enumerate() {
        assert_eq!(
            *off,
            (g as u32 * HD) as i64,
            "head {g}'s output row (offsets {os:?})"
        );
    }
}

/// AND THE ACTIVATIONS LINE UP WITHOUT ANY DECLARATION, because a request is one ROW of a
/// stick-blocked buffer and a row is one stick: `qs` steps `hd` per request within head `h`'s plane,
/// `sc` steps 64. Both are `mb * <stick>` at `mb = 1`, so the truthful extents already produce them.
#[test]
fn the_activations_step_one_row_per_request() {
    let mut s = 0i64;
    let sc = assemble_matmul_batched_off(
        "sc_act",
        MatM::single_row(),
        MatN::of_kv_window(BlockCols::of_slot_window(SlotWindow::SLOTS)),
        MatK::of_head_dim(HD),
        MatY::of_requests(MQ),
        &act("t_qs", MQ, HD),
        0,
        &ker("t_kct", HD, PAGE_SLOTS),
        0,
        Some(("out", PAGE_SLOTS)),
        &act("t_sc", MQ * 32, STICK),
        0,
        &mut s,
        None,
    );
    let qs = distinct_ascending(&per_core_elems(&sc, "sc_act", 0), MQ as usize);
    let out = distinct_ascending(&per_core_elems(&sc, "sc_act", 2), MQ as usize);
    for r in 0..MQ as usize {
        assert_eq!(
            qs[r],
            r as i64 * HD as i64,
            "request {r}'s query row (offsets {qs:?})"
        );
        assert_eq!(
            out[r],
            r as i64 * STICK as i64,
            "request {r}'s score row (offsets {out:?})"
        );
    }
}
