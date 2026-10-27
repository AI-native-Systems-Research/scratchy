// SPDX-License-Identifier: Apache-2.0

//! ⭐⭐⭐⭐⭐ THE ROW GATHER READS ITS OWN REQUEST'S SLABT WINDOW — the 3b-2
//! ruling's source base, pinned at the emission.
//!
//! `embed_row_gather`'s leg 3 (the row gather) reads the transposed slab each
//! request's leg 2 wrote at `slabt_request_base(r)` — `(stk + 1) · hidden`
//! elements past the buffer's start. The l-ramp's index values are sticks
//! RELATIVE TO THAT WINDOW (`64·j + l`, max `hidden − 1`), so a run whose
//! source base stays ZERO gathers REQUEST 0's window for every row: W = 1
//! bakes and runs exact (r = 0, base 0) while every batched rung hands
//! request r ≥ 1 request 0's slab at row lᵣ.
//!
//! ⛔ THAT IS NOT A HYPOTHESIS — MEASURED on the card, granite-3.1-2b fp8, the
//! b2 cell: row 0's gathered row byte-exact against the host embedding, row
//! 1's gathered row byte-exact against the WRONG token's row —
//! `s₀·64 + l₁`, request 0's slab read at row 1's lane — at every step, with
//! the seeds staged correctly. A clean bake, no fault.
//!
//! What this pins, off the emitted wire (the same reading
//! `every_run_of_the_cut_pass_addresses_its_own_entries_and_rows` takes):
//!
//! * leg 3's GATHERED SOURCE (the descriptor's operand 0) for request 1 sits
//!   exactly `(stk + 1) · hidden` elements — one request's window, sentinel
//!   row included — past request 0's;
//! * leg 2's transpose WRITES request 1's slabt window at the SAME delta —
//!   the write and the read cannot disagree, because both spell the base
//!   through the one `slabt_request_base` the door states.

use ktir_core::arena::Arena;
use ktir_core::ir::IRFunction;
use ktir_superdsc::emit::lower_ktir_to_superdsc::{embed_row_gather, Region};
use ktir_superdsc::ktir_node::{BufferId, KtirNode, Program};
use ktir_superdsc::place::PlaceId;
use ktir_superdsc::placement::{BundleLayout, SegRole, TensorPlacement};

/// The b2 geometry at the smallest shape that lowers: `hidden = 128` (two
/// one-stick row-gather entries, `l_sticks = 1`), `vocab = 512` (eight slabs),
/// `k = 4` slab entries, and WIDTH 2 — the first width whose requests do not
/// share a window.
const W: u32 = 2;
const HIDDEN: u32 = 128;
const VOCAB: u32 = 512;
const K: u32 = 4;

/// One whole-view region of `[rows, cols]` at `tid`.
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

/// The node: an `EmbedRowGather` payload stating the entry count. The door
/// reads the program kind and the regions; the function body is inert here.
fn node() -> KtirNode {
    let a: &'static Arena = Arena::global();
    KtirNode {
        func: IRFunction {
            name: "erg",
            arguments: a.args(vec![]),
            operations: a.ops(vec![]),
            grid: (1, 1, 1),
            return_type: None,
        },
        program: Program::EmbedRowGather { k: K },
        bindings: vec![BufferId::new(301), BufferId::new(302), BufferId::new(303), BufferId::new(304)],
        mask: None,
        node_out_tid: None,
        minted: Default::default(),
    }
}

/// A layout that places the four parameters at distinct offsets of one
/// activation segment, so an address names its tensor.
fn layout() -> BundleLayout {
    let mut l = BundleLayout::default();
    for (i, (tid, bytes)) in [
        (301u32, (W * 32 * 4) as u64), // s stick [W, 32] u32
        (302, (W * 32 * 4) as u64),    // l stick [W·l_sticks, 32] u32
        (303, (HIDDEN * VOCAB * 2) as u64), // weight [hidden, vocab] fp16
        (304, (W * HIDDEN * 2) as u64), // out [W, hidden] fp16
    ]
    .into_iter()
    .enumerate()
    {
        l.ids.borrow_mut().insert(
            ktir_superdsc::place::act_name(tid),
            PlaceId::Act(tid),
        );
        l.placements.insert(
            tid,
            TensorPlacement {
                tid,
                role: SegRole::Activation,
                segment: 3,
                bank: 0,
                offset: 0x10000 * (i as u64 + 1),
                size: bytes,
            },
        );
    }
    l
}

/// The smallest per-core start address of a schedule-tree node — dxp's own
/// idx2addr takes the minimum across cores as the run's base.
fn min_start(node: &serde_json::Value) -> Option<i64> {
    node.get("startAddressCoreCorelet_")?
        .get("data_")?
        .as_object()?
        .values()
        .filter_map(|v| {
            v.as_str()
                .and_then(|s| s.parse::<i64>().ok())
                .or(v.as_i64())
        })
        .min()
}

/// The start address the descriptor gives `labeledDs_[i]`'s operand — the
/// allocate node carrying that operand's lds index.
fn operand_start(dsc: &serde_json::Value, i: usize) -> Option<i64> {
    let lds = dsc.get("labeledDs_")?.as_array()?.get(i)?.get("ldsIdx_")?;
    let lds = lds.as_u64().or_else(|| lds.as_i64().map(|x| x as u64))?;
    dsc.get("scheduleTree_")?
        .as_array()?
        .iter()
        .filter(|n| n.get("nodeType_").and_then(|v| v.as_str()) == Some("allocate"))
        .find(|n| n.get("ldsIdx_").and_then(|v| v.as_u64()) == Some(lds))
        .and_then(|n| min_start(n))
}

/// Every `(name, dsc)` the emitted ops carry.
fn dscs_of(
    ops: &[ktir_superdsc::emit::EmittedOp],
) -> Vec<(String, serde_json::Value)> {
    let mut out = Vec::new();
    for op in ops {
        let j = serde_json::to_value(&op.op).expect("serializes");
        for d in j["dscs_"].as_array().into_iter().flatten() {
            for (name, dsc) in d.as_object().into_iter().flatten() {
                out.push((name.clone(), dsc.clone()));
            }
        }
    }
    out
}

#[test]
fn the_row_gather_reads_its_own_requests_window() {
    let regions = [
        region(301, W, 32, false),          // s stick
        region(302, W, 32, false),          // l stick: [W·l_sticks, 32] at l_sticks = 1
        region(303, HIDDEN, VOCAB, false),  // weight
        region(304, W, HIDDEN, true),       // out
    ];
    let l = layout();
    let mut sym = 0i64;
    let ops = embed_row_gather("erg", &node(), &regions, &mut sym, Some(&l))
        .expect("the two-request chained head lowers");

    // ⭐ THE CUT: at W = 2 and l_sticks = 1 the row gather is exactly two ops,
    // one per request, each named for its request.
    let rowg: Vec<(String, serde_json::Value)> = dscs_of(&ops)
        .into_iter()
        .filter(|(n, _)| n.contains("rowg"))
        .collect();
    assert_eq!(
        rowg.len(),
        2,
        "W = 2 and l_sticks = 1 is one row-gather op per request, got {:?}",
        rowg.iter().map(|(n, _)| n).collect::<Vec<_>>()
    );
    let by_req = |r: char| {
        rowg.iter()
            .find(|(n, _)| n.contains(&format!("rowg{r}x")))
            .unwrap_or_else(|| panic!("no rowg op for request {r}: {rowg:?}"))
            .1
            .clone()
    };

    // ⛔ THE PIN: leg 3's GATHERED SOURCE — the descriptor's operand 0 — for
    // request 1 sits exactly one request's window, `(stk + 1) · hidden`
    // elements = `(64 + 1) · 128 · 2` bytes, past request 0's. A delta of
    // ZERO is the measured defect: both requests gather request 0's window.
    let want_delta: i64 = ((64 + 1) * HIDDEN as i64) * 2;
    let src0 = operand_start(&by_req('0'), 0).expect("rowg0 names a gathered source");
    let src1 = operand_start(&by_req('1'), 0).expect("rowg1 names a gathered source");
    assert_eq!(
        src1 - src0,
        want_delta,
        "request 1's row gather must read its OWN slabt window, one \
         (stk+1)·hidden-element window past request 0's — a delta of {} means \
         every request gathers request 0's window (the measured b2 defect: \
         row 1's gathered row is request 0's slab at row l₁)",
        src1 - src0
    );

    // ⭐ AND THE WRITE IT READS: leg 2's transpose puts request 1's slabt
    // window at the SAME delta — the write and the read state one base, so
    // they cannot disagree. The transpose's slabt side is its OUTPUT, the
    // schedule tree's last node.
    let xpose: Vec<(String, serde_json::Value)> = dscs_of(&ops)
        .into_iter()
        .filter(|(n, _)| n.contains("xpose"))
        .collect();
    assert_eq!(
        xpose.len(),
        W as usize,
        "one transpose per request, got {:?}",
        xpose.iter().map(|(n, _)| n).collect::<Vec<_>>()
    );
    let xpose_out = |i: usize| -> i64 {
        let dsc = &xpose[i].1;
        let tree = dsc.get("scheduleTree_").and_then(|t| t.as_array()).expect("a tree");
        let last = tree.last().expect("the output is the tree's last node");
        min_start(last).expect("the output node carries a start address")
    };
    assert_eq!(
        xpose_out(1) - xpose_out(0),
        want_delta,
        "leg 2's transpose writes request 1's slabt window at the same \
         per-request base leg 3 reads — a disagreement here is a window the \
         read names but the write never filled"
    );
}
