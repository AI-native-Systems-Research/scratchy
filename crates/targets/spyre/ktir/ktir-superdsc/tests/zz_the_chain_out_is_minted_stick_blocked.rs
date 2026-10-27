// SPDX-License-Identifier: Apache-2.0

//! ⭐⭐⭐⭐⭐ THE CHAIN OUT IS MINTED STICK-BLOCKED — the relayout leg's write law,
//! pinned at the emission against the body's read law.
//!
//! The chained embedding head's leg 4 (`{name}_rowflat`, a pointwise identity) is
//! the device's own restickify: it carries the gather's FLAT `[W, hidden]` ChainRow
//! synth into the OUT's RowBlocked placement — the layout the body's first op
//! reads. One fact states both sides: the OUT's `StickLayout::row_blocked(W,
//! hidden)` states the WRITE, and the body states the READ.
//!
//! ⛔ THAT IS NOT A HYPOTHESIS — the defect it pins was measured on the card
//! (granite-3.1-2b fp8, the b2 cell): the identity presented BOTH operands at the
//! op's own rank-2 stick-major walk, so the copy was a MEMCPY — the gather's
//! row-major bytes landed in the OUT unchanged (needle-exact: the full two-row
//! row-major embedding found in BOTH the ChainRow synth and the OUT, nothing
//! anywhere scattered), and the body's stick-major read scrambled BOTH rows'
//! logits. W = 1 byte-identical (the laws coincide), which is how it hid.
//!
//! What this pins, off the emitted wire of the relayout leg:
//!
//! * the OUTPUT operand walks `StickLayout::row_blocked(W, hidden)`: rank-2
//!   `["mb", "out"]`, `maxDimSizes_ = [-1, -1]` (dxp reconstructs the
//!   stick-blocked walk), and every core's start sits at the row_blocked law of
//!   ITS OWN corner — at W = 2 and W = 4;
//! * the INPUT operand keeps the gather's flat law (its declared `fl` handle):
//!   the flat walk's corner is recoverable from the input's own start, and the
//!   output's start for THAT SAME CORE must be the row_blocked spelling of the
//!   SAME corner — the per-core pairing a memcpy breaks;
//! * at W = 1 the relayout mints NOTHING (the elision: the laws coincide, leg 3
//!   wrote the OUT's own name) — solo stays byte-identical.

use ktir_core::arena::Arena;
use ktir_core::ir::IRFunction;
use ktir_superdsc::emit::lower_ktir_to_superdsc::{embed_row_gather, Region};
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
        bindings: vec![BufferId::new(301), BufferId::new(302), BufferId::new(303), BufferId::new(304)],
        mask: None,
        node_out_tid: None,
        minted: Default::default(),
    }
}

/// A layout that places the four parameters at distinct offsets of one activation
/// segment, so an address names its tensor.
fn layout(w: u32) -> BundleLayout {
    let mut l = BundleLayout::default();
    for (i, (tid, bytes)) in [
        (301u32, (w * 32 * 4) as u64), // s stick [W, 32] u32
        (302, (w * 32 * 4) as u64),    // l stick [W·l_sticks, 32] u32
        (303, (HIDDEN * VOCAB * 2) as u64), // weight [hidden, vocab] fp16
        (304, (w * HIDDEN * 2) as u64), // out [W, hidden] fp16
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
        let a = v.as_str().and_then(|s| s.parse::<i64>().ok()).or(v.as_i64());
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
        region(301, w, 32, false),           // s stick
        region(302, w, 32, false),           // l stick: [W·l_sticks, 32] at l_sticks = 1
        region(303, HIDDEN, VOCAB, false),   // weight
        region(304, w, HIDDEN, true),        // out
    ];
    let l = layout(w);
    let mut sym = 0i64;
    let ops = embed_row_gather("erg", &node(), &regions, &mut sym, Some(&l))
        .unwrap_or_else(|e| panic!("the {w}-request chained head lowers: {e:?}"));

    let dsc = rowflat_dsc(&ops).unwrap_or_else(|| {
        panic!("W = {w}: the relayout leg exists (the laws do not coincide at W > 1)")
    });

    // ⭐ THE OUTPUT'S WALK IS RANK-2 STICK-BLOCKED: `["mb", "out"]` with
    // `maxDimSizes_ = [-1, -1]` — dxp reconstructs the stick-blocked walk.
    let out_order = dsc
        .get("scheduleTree_")
        .and_then(|t| t.as_array())
        .expect("a tree")
        .iter()
        .find(|n| {
            n.get("nodeType_").and_then(|v| v.as_str()) == Some("allocate")
                && n.get("ldsIdx_").and_then(|v| v.as_u64())
                    == dsc["labeledDs_"][1]["ldsIdx_"].as_u64()
        })
        .expect("the output's allocate");
    fn as_strs(v: &serde_json::Value) -> Vec<&str> {
        v.as_array()
            .expect("an array")
            .iter()
            .map(|x| x.as_str().expect("a string"))
            .collect()
    }
    assert_eq!(
        as_strs(out_order.get("layoutDimOrder_").expect("an order")),
        vec!["mb", "out", "y"],
        "W = {w}: the OUT's layout is the rank-3 trailing-y spelling (one rank for the whole \
         op — dxp refuses a mixed-rank op) whose trailing unit dim folds to the SAME \
         RowBlocked classification the rank-2 form carries"
    );
    let max_dims = out_order
        .get("maxDimSizes_")
        .and_then(|v| v.as_array())
        .expect("maxDimSizes_");
    assert_eq!(max_dims.len(), 3, "W = {w}: the OUT's maxDimSizes_ is rank-3");
    assert!(
        max_dims.iter().all(|x| x.as_i64() == Some(-1)),
        "W = {w}: the OUT's maxDimSizes_ is -1 per dim — dxp reconstructs the stick-blocked walk"
    );

    // ⭐⭐ THE PER-CORE PAIRING — the check a memcpy breaks. The input walks FLAT
    // (the gather's own write law), and a flat start is INJECTIVE over the core
    // grid: element offset `r · hidden + c` names the corner (r, c) outright. The
    // output's start for THAT SAME CORE must be the row_blocked spelling of the
    // SAME corner — `(c / 64) · (W · 64) + r · 64`, one `StickLayout::dev_off`.
    let rb = StickLayout::row_blocked(w as usize, HIDDEN as usize);
    let flat = StickLayout::flat(w as usize, HIDDEN as usize);
    assert!(
        !rb.addr_eq(&flat),
        "W = {w}: the pin must discriminate — the two laws must disagree at this width"
    );
    let inp = core_starts(&dsc, 0);
    let out = core_starts(&dsc, 1);
    assert!(!inp.is_empty() && !out.is_empty(), "both operands carry starts");
    let inp_min = *inp.values().min().unwrap();
    let out_min = *out.values().min().unwrap();
    for (core, &i0) in &inp {
        let o0 = *out
            .get(core)
            .unwrap_or_else(|| panic!("core {core} starts both operands"));
        // The input's flat delta names the corner — elements, then bytes (wordLength 2).
        let d_in = ((i0 - inp_min) / 2) as usize;
        let (r, c) = (d_in / HIDDEN as usize, d_in % HIDDEN as usize);
        let want_out = rb.dev_off(r, c) as i64 * 2;
        let got_out = o0 - out_min;
        assert_eq!(
            got_out, want_out,
            "W = {w}, core {core}: the OUT's start must be the row_blocked law \
             ({rb:?}) of the corner ({r}, {c}) the input's own flat start names — \
             got {got_out} B, want {want_out} B. A start equal to the FLAT spelling \
             ({}) is the measured b2 defect: the identity memcpy'd the gather's \
             row-major bytes into the OUT and the body's stick-major read scrambled \
             both rows",
            flat.dev_off(r, c) as i64 * 2
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
