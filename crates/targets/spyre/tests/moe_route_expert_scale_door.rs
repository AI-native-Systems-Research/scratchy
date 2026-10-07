// SPDX-License-Identifier: Apache-2.0
//! THE ROUTE EXPERT-SCALE DOOR — the per-expert scale select's acceptance
//! gate.
//!
//! `Program::RouteExpertScale` was refused by name in the door; this file
//! drives the REAL path (`lower_graph_to_superdsc` → the door, with the
//! bundle layout) and pins what the door now emits:
//!
//! * **The chain shape**: per slot `j` the 6/7-op group
//!   `equal`/`multiply`(scale)/`multiply`(score)/`sum`-reduce/`multiply`
//!   (identity-row) + accumulate (`identity` seed at j=0, `add` after) —
//!   `7k−2` ops, independent of the token count.
//! * **The select masks the SCALE ROW** (a loaded weight, never a bake-time
//!   const): the second leg's second operand is the scale tensor, the
//!   mb-broadcast `[1,W]` parent every row gathers from.
//! * **The score rides the matched lane**: the third leg's second operand
//!   reads lane j of the scores buffer — j·2 bytes past the row base.
//!
//! The NUMERIC gate (tiny26 EMU-vs-card tensordump parity) runs on the pod.

use scratchy_subtile::subtile_ir::{
    NeoX, NumExperts, RouterBundle, SubOp, SubtileId, SubtileIR, SubtileNode, TensorId,
    TensorRegion, TensorShape, TopK,
};

/// Stick-legal geometry: E = 8 experts (padded width W = 64, one stick),
/// k = 2, m = 5.
const E: u32 = 8;
const K: u32 = 2;
const M: u32 = 5;

/// RouterLogits → RouteArgsort → RouteTopK → RouteGatherScores →
/// RouteExpertScale — the smallest tape that reaches the scale door with
/// every const placement minted.
fn scale_ir() -> SubtileIR<NeoX> {
    let tensors = vec![
        TensorShape { rows: M, cols: 64 }, // t0 x (the matmul's input)
        TensorShape { rows: E, cols: 64 }, // t1 the [E, hidden] gate buffer
        TensorShape { rows: M, cols: E },  // t2 logits (argsort's input)
        TensorShape { rows: M, cols: E },  // t3 rank vector (argsort out)
        TensorShape { rows: M, cols: K },  // t4 top-k indices
        TensorShape { rows: M, cols: K },  // t5 gathered scores
        TensorShape { rows: 1, cols: E },  // t6 the [1,E] scale row
        TensorShape { rows: M, cols: K },  // t7 scaled scores (out)
    ];
    let tr = |t: usize| TensorRegion {
        tensor: TensorId::from_index(t),
        region: tensors[t].whole(),
    };
    let nodes = vec![
        SubtileNode {
            id: SubtileId::from_index(0),
            op: SubOp::RouterLogits {
                experts: NumExperts::new(std::num::NonZeroU32::new(E).unwrap()),
                router: RouterBundle::Gemma,
            },
            inputs: vec![tr(0), tr(1)],
            output: tr(2),
        },
        SubtileNode {
            id: SubtileId::from_index(1),
            op: SubOp::RouteArgsort,
            inputs: vec![tr(2)],
            output: tr(3),
        },
        SubtileNode {
            id: SubtileId::from_index(2),
            op: SubOp::RouteTopK {
                k: TopK::new(std::num::NonZeroU32::new(K).unwrap()),
            },
            inputs: vec![tr(3)],
            output: tr(4),
        },
        SubtileNode {
            id: SubtileId::from_index(3),
            op: SubOp::RouteGatherScores,
            inputs: vec![tr(2), tr(4)],
            output: tr(5),
        },
        SubtileNode {
            id: SubtileId::from_index(4),
            op: SubOp::RouteExpertScale {
                router: RouterBundle::Gemma,
            },
            inputs: vec![tr(5), tr(4), tr(6)],
            output: tr(7),
        },
    ];
    // Every tensor is a source here (the e2e chain test's own form): the
    // scale row t6 is a LOADED weight in the real tape, and a source
    // placement is how a weight-shaped input reaches this door in a test.
    let num_sources = tensors.len() as u32;
    SubtileIR {
        result: TensorId::from_index(7),
        tensors,
        num_sources,
        nodes,
        op_output: Vec::new(),
    }
}

/// Lower through the REAL door path and return the emitted ops.
fn door_ops() -> Vec<ktir_superdsc::emit::EmittedOp> {
    let weight_ids = std::collections::HashSet::new();
    let (ops, _layout) =
        scratchy_target_spyre::lower_subtile_tape_to_ktir::lower_graph_to_superdsc(
            &scale_ir(),
            &weight_ids,
            ktir_superdsc::ktir_node::ActiveCap::FULL,
            false,
        )
        .unwrap_or_else(|e| panic!("lower through the door: {e:?}"));
    ops
}

/// The emitted `opFuncName`s of the door's ops, in emission order.
fn opfunc_names(ops: &[ktir_superdsc::emit::EmittedOp]) -> Vec<String> {
    ops.iter()
        .map(|o| {
            let dsc = o.dsc();
            let d = dsc
                .dscs_
                .first()
                .and_then(|m| m.values().next())
                .expect("an emitted op has at least one dsc");
            d.computeOp_
                .first()
                .map(|c| c.opFuncName.clone())
                .expect("a dsc has a computeOp")
        })
        .collect()
}

/// An op's `which`-th input operand's core-0 start HBM bytes — the combine
/// door test's own read pattern.
fn input_start_bytes(o: &ktir_superdsc::emit::EmittedOp, which: usize) -> u64 {
    let dsc = o.dsc();
    let d = dsc
        .dscs_
        .first()
        .and_then(|m| m.values().next())
        .expect("an emitted op has at least one dsc");
    let in_lds = d
        .computeOp_
        .first()
        .and_then(|c| c.inputLabeledDs.get(which))
        .cloned()
        .unwrap_or_default();
    let idx: u32 = in_lds
        .rsplit("idx")
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or(u32::MAX);
    let alloc = d
        .scheduleTree_
        .iter()
        .find(|n| n.nodeType_ == "allocate" && n.component_ == "hbm" && n.ldsIdx_ == idx)
        .unwrap_or_else(|| panic!("the input's hbm allocate (lds {idx}, from {in_lds:?})"));
    alloc
        .startAddressCoreCorelet_
        .data_
        .get("[0, 0, 0]")
        .and_then(|s| s.parse().ok())
        .expect("core 0's start address")
}

/// ⭐ THE CHAIN: `7k−2` ops — per slot j the compare, the scale-select
/// product, the score product, the reduce, the lane-j product, and the
/// accumulate leg (`identity` seed at j=0, `add` after).
#[test]
fn the_scale_emits_the_seven_k_chain() {
    let ops = door_ops();
    let names = opfunc_names(&ops);
    let expect: Vec<&str> = (0..K)
        .flat_map(|j| {
            let acc = if j == 0 { "identity" } else { "add" };
            ["equal", "mul", "mul", "sum", "mul", acc]
        })
        .collect();
    let tail: Vec<&str> = names
        .iter()
        .skip(names.len() - expect.len())
        .map(|n| n.as_str())
        .collect();
    assert_eq!(tail, expect, "the scale chain, in emission order");
}

/// ⭐ THE SELECT MASKS THE SCALE ROW AND THE SCORE RIDES LANE j: the per-slot
/// scale-select legs read the scale tensor (the same base every slot — one
/// shared row), and the score-product legs' score operand advances j·2 bytes
/// per slot (lane j of the scores buffer).
#[test]
fn the_select_reads_the_scale_row_and_lane_j_of_the_scores() {
    let ops = door_ops();
    let names = opfunc_names(&ops);
    // The LAST K mul-mul pairs are the scale door's select+score legs: find
    // the tail 7k-2 window, then index within it.
    let tail_len = (7 * K - 2) as usize;
    let start = names.len() - tail_len;
    let tail = &names[start..];
    let mut scale_legs = Vec::new();
    let mut score_legs = Vec::new();
    let mut i = 0;
    while i + 6 <= tail.len() {
        if tail[i..i + 6] == ["equal", "mul", "mul", "sum", "mul", if i == 0 { "identity" } else { "add" }] {
            scale_legs.push(start + i + 1); // the first mul: the scale select
            score_legs.push(start + i + 2); // the second mul: the score product
            i += 6;
        } else {
            i += 1;
        }
    }
    assert_eq!(scale_legs.len(), K as usize, "one select pair per slot");
    // The scale operand (input 1 of the select leg) is the SAME base every
    // slot — the one shared [1,W] row.
    let bases: std::collections::HashSet<u64> = scale_legs
        .iter()
        .map(|&i| input_start_bytes(&ops[i], 1))
        .collect();
    assert_eq!(bases.len(), 1, "every slot's select reads the same scale row");
    // The score operand (input 1 of the score leg) advances one lane per slot.
    let score_starts: Vec<u64> = score_legs
        .iter()
        .map(|&i| input_start_bytes(&ops[i], 1))
        .collect();
    for (j, s) in score_starts.iter().enumerate() {
        assert_eq!(
            s - score_starts[0],
            j as u64 * 2,
            "score leg {j} reads lane {j} of the scores buffer"
        );
    }
}
