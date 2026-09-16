// SPDX-License-Identifier: Apache-2.0
//! WHAT THE SHIPPED FOLD OP ADDRESSES — calibrating the stride model against an op that is KNOWN
//! CORRECT on the card.
//!
//! Two attempts at collapsing the prefix fold were built on `attn.rs`'s claim that the batched score
//! leg's `y`-stride is `mb*in = gqa*mq*hd`, and both corrupted the output. A probe of the per-core
//! addresses then said the per-`y` step is 64 — one stick row. Those cannot both describe reality,
//! because the op in this file is EXACTLY the one a coherent bs=8 decode runs today.
//!
//! So this is a calibration, not a proof: emit the shipped op at bs=8 and print what it addresses.
//! Whatever it does IS correct by construction, and the collapse has to be built on that rather than
//! on either of my two readings. Recording the numbers here so the next attempt starts from measured
//! semantics instead of a comment.

use ktir_superdsc::emit;
use ktir_superdsc::ir::bridge::tiled_op_sdsc_op::{SharedKernelBmmForm, assemble_matmul_off};
use scratchy_subtile::addr::DevOff;
use scratchy_subtile::sdsc_abstract::{
    BlockCols, KernelTag, MaskRows, MatK, MatM, MatN, MatY, OperandPlacement, QueryRowCount,
    RowBlockedTag, RungWidth, SlotWindow, StickLayout, Stk,
};

const HD: u32 = 64;
const STICK: u32 = 64; // the fold sweeps one 64-slot block at a time
const NQH: u32 = 32;
const MQ: u32 = 8; // a bs=8 decode batch: the rows ARE requests
const GQA: u32 = 4;
const PAGE_SLOTS: u32 = 256;

fn act(n: &str, rows: u32, cols: u32) -> Stk<RowBlockedTag> {
    Stk::<RowBlockedTag>::new(n, StickLayout::row_blocked(rows as usize, cols as usize)).unwrap()
}
fn ker(n: &str, rows: u32, cols: u32) -> Stk<KernelTag> {
    Stk::<KernelTag>::new(n, StickLayout::kernel(rows as usize, cols as usize)).unwrap()
}

/// An element offset inside one operand, measured from that operand's own base. NOT interchangeable
/// with a byte address, a row index, or a core index — the whole point of this file is that those got
/// confused.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
struct Elems(i64);

/// A batched matmul's per-core start offsets, KEYED BY THE AXES THEY VARY — never flattened.
///
/// ⛔ FLATTENING IS THE BUG THIS FILE DOCUMENTS. Read as one ascending `Vec`, the offsets of a
/// `(y, mb)` grid look like a single stride, and I took the `mb` step for the `y` step — which sent two
/// attempts at the fold collapse at the wrong fix. Asking for `at(y, mb)` makes the question "which
/// axis's stride is this?" impossible to skip, and `y_stride`/`mb_stride` can then only mean one thing
/// each.
///
/// ⛔ AND AN ASSUMED CORE ORDER IS THE SAME FLATTENING ONE LEVEL UP. Which `(y, mb)` a core owns is
/// the op's OWN declaration (`coreIdToWkSlice_`), not an enumeration convention: a hand-assumed
/// `core = y·mb_extent + mb` here silently TRANSPOSED the grid, and against the batch-inner walk's
/// equally swapped strides the two errors cancelled — this file printed "correct" numbers for a walk
/// the card faults on. The grid is keyed by the declared slices, so there is no order to assume.
struct PerCore {
    /// `(y_slice, mb_slice)` → offset, exactly as each core declares its own slices.
    offs: std::collections::BTreeMap<(u32, u32), Elems>,
    y_extent: u32,
    mb_extent: u32,
}

impl PerCore {
    /// Read operand `arg` (0=input, 1=kernel, 2=output) of an emitted op.
    fn of(e: &emit::EmittedOp, op: &str, arg: usize, y_extent: u32, mb_extent: u32) -> PerCore {
        let v = serde_json::to_value(&e.op).unwrap();
        let wk = v["coreIdToWkSlice_"]
            .as_object()
            .expect("declared work slices");
        let d = &v["dscs_"][0][op]["scheduleTree_"][arg]["startAddressCoreCorelet_"]["data_"];
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
            let prev = offs.insert((slice("y"), slice("mb")), Elems((a - base) / 2));
            assert!(prev.is_none(), "two cores declare the same (y, mb) slice");
        }
        PerCore {
            offs,
            y_extent,
            mb_extent,
        }
    }

    fn at(&self, y: u32, mb: u32) -> Elems {
        assert!(
            y < self.y_extent && mb < self.mb_extent,
            "({y},{mb}) outside the grid"
        );
        self.offs[&(y, mb)]
    }

    /// The step from one `y` to the next, at a fixed `mb`. `None` when `y_extent == 1`.
    fn y_stride(&self) -> Option<Elems> {
        (self.y_extent > 1).then(|| Elems(self.at(1, 0).0 - self.at(0, 0).0))
    }

    /// The step from one `mb` to the next, at a fixed `y`. `None` when `mb_extent == 1`.
    fn mb_stride(&self) -> Option<Elems> {
        (self.mb_extent > 1).then(|| Elems(self.at(0, 1).0 - self.at(0, 0).0))
    }

    /// Every core reads the same address — a SHARED operand, like a GQA group's kv-head kernel.
    fn is_shared(&self) -> bool {
        self.offs.values().all(|o| *o == Elems(0))
    }
}

/// THE SHIPPED BATCHED SCORE LEG at bs=8, printed. `m = mq` (it computes every row), `y = gqa`.
///
/// The row order it writes into is `h*mq + r`, so within a GQA group the four heads are `mq` ROWS
/// apart — `mq * 64` elements in a stick-blocked buffer — and the requests inside a head are one row
/// apart. Whether the emitted addresses show that, or something else, is the whole question.
#[test]
fn print_what_the_shipped_fold_score_op_addresses() {
    let mut s = 0i64;
    // Offsets exactly as `assemble_attn_block` builds them for kv-head 0 (query heads 0..3), and
    // the same FORM a bs=8 bundle threads: rows are requests at mq>1, so this op carries the
    // corrected head-outermost walk. The per-core START addresses asserted below are identical
    // under both forms — the strict-subset fact that let the swapped walk hide — so the
    // calibration stands; the declared-walk fields are pinned in `batched_decode_walk_order.rs`.
    let e = assemble_matmul_off(
        "sc_shipped",
        MatM::of_query_rows(QueryRowCount::of_mq(MQ)),
        MatN::of_kv_window(BlockCols::of_slot_window(SlotWindow::SLOTS)),
        MatK::of_head_dim(HD),
        // The batch axis carries both operands' head strides, so the head-outermost walk it derives
        // (`mb_dev * in` for the read, `mb_dev * out` for the write) is checked against where the heads
        // really are: `qs` is the token stream (`mq*hd` apart) and `sc` is head-major (`mq*stick`).
        MatY::of_gqa_group(
            GQA,
            OperandPlacement::of_token_stream(QueryRowCount::of_mq(MQ), NQH, HD, 0, 0),
            OperandPlacement::of_head_major_rows(
                0,
                QueryRowCount::of_mq(MQ),
                MaskRows::new(
                    NQH,
                    RungWidth::of_baked_rows(MQ).expect("a nonzero rung width"),
                ),
                BlockCols::of_slot_window(SlotWindow::SLOTS),
            ),
        ),
        SharedKernelBmmForm::of_attn_rows(true, QueryRowCount::of_mq(MQ)),
        &act("t_qs", MQ, HD),
        DevOff::ZERO,
        &ker("t_kct", HD, PAGE_SLOTS),
        DevOff::ZERO,
        &act("t_sc", NQH * MQ, STICK),
        DevOff::ZERO,
        &mut s,
        None,
    );
    for (name, arg) in [("input qs", 0usize), ("kernel kct", 1), ("output sc", 2)] {
        let pc = PerCore::of(&e, "sc_shipped", arg, GQA, MQ);
        println!(
            "{name}: y_stride={:?} mb_stride={:?} shared={}",
            pc.y_stride(),
            pc.mb_stride(),
            pc.is_shared()
        );
    }
    println!(
        "one row = {STICK} elems; mb_extent rows = {} elems",
        MQ * STICK
    );

    // ⭐ THE CALIBRATION. `mb` (the request) steps ONE ROW; `y` (the head) steps `mb_extent` ROWS.
    // That is `attn.rs`'s "y-stride = mb*out", and it is what makes the shipped row order `h*mq + r`.
    let qs = PerCore::of(&e, "sc_shipped", 0, GQA, MQ);
    let sc = PerCore::of(&e, "sc_shipped", 2, GQA, MQ);
    for pc in [&qs, &sc] {
        assert_eq!(
            pc.mb_stride(),
            Some(Elems(STICK as i64)),
            "a request is one row"
        );
        assert_eq!(
            pc.y_stride(),
            Some(Elems((MQ * STICK) as i64)),
            "a head is mb_extent rows — NOT one row, which is the misreading that cost two bakes"
        );
        // And the grid is exactly the `gqa*mq` consecutive rows the shipped order expects.
        for y in 0..GQA {
            for mb in 0..MQ {
                assert_eq!(
                    pc.at(y, mb),
                    Elems(((y * MQ + mb) * STICK) as i64),
                    "row h*mq+r"
                );
            }
        }
    }
    // THE KERNEL IS SHARED: a GQA group reads ONE kv head's Kᵀ. That is why this form can never carry
    // a request axis — the requests inside a head do not share a block.
    assert!(PerCore::of(&e, "sc_shipped", 1, GQA, MQ).is_shared());
}
