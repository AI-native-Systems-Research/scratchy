// SPDX-License-Identifier: Apache-2.0

//! ⭐⭐⭐⭐⭐ THE CHAIN OUT IS MINTED STICK-BLOCKED — the memcpy leg's walk law,
//! pinned at the emission against the body's read law.
//!
//! The chained embedding head's leg 4 (`{name}_rowflat`, a pointwise identity)
//! is the HOMOGENEOUS stick-major copy: leg 3 wrote the `[W, hidden]` ChainRow
//! synth AT ITS OWN stick-blocked scatter (`GatherDest::Scatter` — the same
//! bytes `StickLayout::row_blocked(W, hidden)` places), so the copy presents
//! BOTH operands under that one walk and the OUT's RowBlocked placement is the
//! same law the body's first op reads. One fact states all three sides.
//!
//! ⛔ THE DEFECT CLASS THIS PIN STANDS OVER was measured on the card
//! (granite-3.1-2b fp8, the b2 cell): when the gather wrote the ChainRow
//! ROW-MAJOR, the identity could not spell both walks at once — presenting the
//! flat input at the op's own stick-major walk made it a MEMCPY of row-major
//! bytes (the body's read scrambled both rows), and the mixed-rank/mixed-walk
//! forms that kept the input's flat walk were unsound or unspelleable (the
//! 10-27 ruling: the scatter moved to the WRITE side, where it is one law).
//!
//! What this pins, off the emitted wire of the memcpy leg:
//!
//! * BOTH operands walk rank-2 `["mb", "out"]` with `maxDimSizes_ = [-1, -1]`
//!   (dxp reconstructs the stick-blocked walk) — the input taking rank-2 at
//!   all is the rb declaration at work: a flat-declared input in a
//!   stick-major op has no rank-2 spelling anymore, so a rank-2 input IS the
//!   ChainRow's scatter being read under its own law;
//! * every core's input and output starts PAIR — the same corner, the same
//!   law, byte for byte: the copy moves sticks, never re-lays-out lanes;
//! * at W = 1 the memcpy mints NOTHING (the elision: the laws coincide, leg 3
//!   wrote the OUT's own name) — solo stays byte-identical.

use ktir_core::arena::Arena;
use ktir_core::ir::IRFunction;
use ktir_superdsc::emit::lower_ktir_to_superdsc::{Region, embed_row_gather};
use ktir_superdsc::ktir_node::{BufferId, KtirNode, Program};
use ktir_superdsc::place::PlaceId;
use ktir_superdsc::placement::{BundleLayout, SegRole, TensorPlacement};
use ktir_superdsc::sdsc_abstract::StickLayout;

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

/// The node: an `EmbedRowGather` payload stating the entry count.
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
        bindings: vec![
            BufferId::new(301),
            BufferId::new(302),
            BufferId::new(303),
            BufferId::new(304),
        ],
        mask: None,
        node_out_tid: None,
        minted: Default::default(),
        lx_plan: Default::default(),
    }
}

/// A layout that places the four parameters at distinct offsets of one activation
/// segment, so an address names its tensor.
fn layout(w: u32) -> BundleLayout {
    let mut l = BundleLayout::default();
    for (i, (tid, bytes)) in [
        (301u32, (w * 32 * 4) as u64),      // s stick [W, 32] u32
        (302, (w * 32 * 4) as u64),         // l stick [W·l_sticks, 32] u32
        (303, (HIDDEN * VOCAB * 2) as u64), // weight [hidden, vocab] fp16
        (304, (w * HIDDEN * 2) as u64),     // out [W, hidden] fp16
    ]
    .into_iter()
    .enumerate()
    {
        l.ids
            .borrow_mut()
            .insert(ktir_superdsc::place::act_name(tid), PlaceId::Act(tid));
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

/// Every `(name, dsc)` the emitted ops carry.
fn dscs_of(ops: &[ktir_superdsc::emit::EmittedOp]) -> Vec<(String, serde_json::Value)> {
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

/// The relayout leg's dsc, by name.
fn rowflat_dsc(ops: &[ktir_superdsc::emit::EmittedOp]) -> Option<serde_json::Value> {
    dscs_of(ops)
        .into_iter()
        .find(|(n, _)| n.contains("rowflat"))
        .map(|(_, d)| d)
}

/// One operand's allocate node (by `labeledDs_` index) — its per-core starts,
/// keyed by the core coordinate the dsc spells (`"[c, cl, t]"`).
fn core_starts(dsc: &serde_json::Value, i: usize) -> std::collections::BTreeMap<String, i64> {
    let lds = dsc
        .get("labeledDs_")
        .and_then(|a| a.as_array())
        .and_then(|a| a.get(i))
        .and_then(|v| v.get("ldsIdx_"))
        .and_then(|v| v.as_u64())
        .unwrap_or_else(|| panic!("operand {i} names an ldsIdx"));
    let node = dsc
        .get("scheduleTree_")
        .and_then(|t| t.as_array())
        .expect("a tree")
        .iter()
        .find(|n| {
            n.get("nodeType_").and_then(|v| v.as_str()) == Some("allocate")
                && n.get("ldsIdx_").and_then(|v| v.as_u64()) == Some(lds)
        })
        .unwrap_or_else(|| panic!("operand {i} has an allocate node"));
    let mut out = std::collections::BTreeMap::new();
    for (k, v) in node
        .get("startAddressCoreCorelet_")
        .and_then(|s| s.get("data_"))
        .and_then(|d| d.as_object())
        .expect("per-core starts")
    {
        let a = v
            .as_str()
            .and_then(|s| s.parse::<i64>().ok())
            .or(v.as_i64());
        if let Some(a) = a {
            out.insert(k.clone(), a);
        }
    }
    out
}

/// THE PIN, at one width: the relayout's output walks `row_blocked(W, HIDDEN)` at
/// exactly the corners its input's flat walk names.
fn the_out_is_minted_stick_blocked(w: u32) {
    let regions = [
        region(301, w, 32, false),         // s stick
        region(302, w, 32, false),         // l stick: [W·l_sticks, 32] at l_sticks = 1
        region(303, HIDDEN, VOCAB, false), // weight
        region(304, w, HIDDEN, true),      // out
    ];
    let l = layout(w);
    let mut sym = 0i64;
    let ops = embed_row_gather("erg", &node(), &regions, &mut sym, Some(&l))
        .unwrap_or_else(|e| panic!("the {w}-request chained head lowers: {e:?}"));

    let dsc = rowflat_dsc(&ops).unwrap_or_else(|| {
        panic!("W = {w}: the relayout leg exists (the laws do not coincide at W > 1)")
    });

    // ⭐ BOTH OPERANDS WALK RANK-2 STICK-BLOCKED: `["mb", "out"]` with
    // `maxDimSizes_ = [-1, -1]` — dxp reconstructs the stick-blocked walk. The
    // INPUT's rank-2 spelling is itself the pin's subject: a flat-declared
    // input in a stick-major op has no rank-2 form (it would be the mixed-rank
    // op dxp refuses), so a rank-2 input states the ChainRow's scatter being
    // read under its own law.
    fn as_strs(v: &serde_json::Value) -> Vec<&str> {
        v.as_array()
            .expect("an array")
            .iter()
            .map(|x| x.as_str().expect("a string"))
            .collect()
    }
    for operand in [0usize, 1usize] {
        let order = dsc
            .get("scheduleTree_")
            .and_then(|t| t.as_array())
            .expect("a tree")
            .iter()
            .find(|n| {
                n.get("nodeType_").and_then(|v| v.as_str()) == Some("allocate")
                    && n.get("ldsIdx_").and_then(|v| v.as_u64())
                        == dsc["labeledDs_"][operand]["ldsIdx_"].as_u64()
            })
            .unwrap_or_else(|| panic!("operand {operand}'s allocate"));
        assert_eq!(
            as_strs(order.get("layoutDimOrder_").expect("an order")),
            vec!["mb", "out"],
            "W = {w}: operand {operand}'s layout is the rank-2 stick-major \
             spelling — one rank and one walk for the whole op (a rank-3 operand \
             beside a rank-2 one is the mixed-rank op dxp refuses)"
        );
        let max_dims = order
            .get("maxDimSizes_")
            .and_then(|v| v.as_array())
            .expect("maxDimSizes_");
        assert_eq!(
            max_dims.len(),
            2,
            "W = {w}: operand {operand}'s maxDimSizes_ is rank-2"
        );
        assert!(
            max_dims.iter().all(|x| x.as_i64() == Some(-1)),
            "W = {w}: operand {operand}'s maxDimSizes_ is -1 per dim — dxp reconstructs the stick-blocked walk"
        );
    }

    // ⭐⭐ THE PER-CORE PAIRING — the copy moves sticks, never re-lays-out
    // lanes. Both operands walk the SAME `StickLayout::row_blocked(W, hidden)`
    // (the input's rank-2 spelling IS the scatter's own law — see the header),
    // so for every core the input's start delta and the output's start delta
    // name the SAME corner under the SAME law: byte for byte equal. A
    // disagreement is a re-laid-out copy — the measured defect class, in
    // which the input's walk and the output's walk spelled different laws.
    let rb = StickLayout::row_blocked(w as usize, HIDDEN as usize);
    let flat = StickLayout::flat(w as usize, HIDDEN as usize);
    assert!(
        !rb.addr_eq(&flat),
        "W = {w}: the pin must discriminate — the two laws must disagree at this width"
    );
    let inp = core_starts(&dsc, 0);
    let out = core_starts(&dsc, 1);
    assert!(
        !inp.is_empty() && !out.is_empty(),
        "both operands carry starts"
    );
    let inp_min = *inp.values().min().unwrap();
    let out_min = *out.values().min().unwrap();
    for (core, &i0) in &inp {
        let o0 = *out
            .get(core)
            .unwrap_or_else(|| panic!("core {core} starts both operands"));
        let d_in = i0 - inp_min;
        let d_out = o0 - out_min;
        assert_eq!(
            d_in, d_out,
            "W = {w}, core {core}: the memcpy pairs the SAME corner under the SAME \
             walk on both sides (input delta {d_in} B, output delta {d_out} B) — a \
             disagreement is a copy that re-lays-out lanes, the defect class this \
             leg's homogeneous form exists to make unspelleable"
        );
        // And the corner is a WHOLE STICK of the rb law — a core's start is
        // always a stick boundary, never mid-stick.
        assert_eq!(
            d_in % (64 * 2),
            0,
            "W = {w}, core {core}: a stick-major walk starts every core at a stick \
             boundary — {d_in} B is mid-stick"
        );
    }
}

#[test]
fn the_out_is_minted_stick_blocked_at_w2() {
    the_out_is_minted_stick_blocked(2);
}

#[test]
fn the_out_is_minted_stick_blocked_at_w4() {
    the_out_is_minted_stick_blocked(4);
}

#[test]
fn the_relayout_mints_nothing_at_w1() {
    // The elision: at W = 1 the flat and row_blocked laws COINCIDE (one row —
    // nothing to interleave), so leg 3 wrote the OUT's own name and the relayout
    // mints NOTHING. Solo stays byte-identical because nothing re-lays-out.
    let regions = [
        region(301, 1, 32, false),
        region(302, 1, 32, false),
        region(303, HIDDEN, VOCAB, false),
        region(304, 1, HIDDEN, true),
    ];
    let l = layout(1);
    let mut sym = 0i64;
    let ops = embed_row_gather("erg", &node(), &regions, &mut sym, Some(&l))
        .expect("the solo chained head lowers");
    assert!(
        rowflat_dsc(&ops).is_none(),
        "W = 1: the laws coincide, so the relayout leg must be elided — an op here \
         would re-lay-out a one-row tensor whose laws already agree"
    );
    assert!(
        StickLayout::row_blocked(1, HIDDEN as usize)
            .addr_eq(&StickLayout::flat(1, HIDDEN as usize)),
        "W = 1: the elision's own premise — the OUT's row_blocked walk equals the \
         gather's row-major walk byte for byte, so solo is byte-identical"
    );
}
