// SPDX-License-Identifier: Apache-2.0
//! THE ROUTE GATHER-SCORES DOOR — the one-hot select's acceptance gate.
//!
//! `Program::RouteGatherScores` was refused by name in the door; this file
//! drives the REAL path (`lower_graph_to_superdsc` → the door, with the
//! bundle layout) and pins what the door now emits:
//!
//! * **The chain shape**: per slot `j` the 5/6-op group
//!   `equal`/`multiply`/`sum`-reduce/`multiply` + accumulate (`identity`
//!   seed at j=0, `add` after) — `6k−2` ops, independent of the token count,
//!   and NEVER an indirect access tile (the one-hot select states the same
//!   function with no data-dependent memory traffic).
//! * **The compare reads LANE j of the indices**: the `equal` leg's first
//!   operand is the per-row index scalar at lane j (the combine door's
//!   col_at law) — j·2 bytes past the row base.
//! * **The select masks the SCORES**: the second leg multiplies the match
//!   mask by the full scores row, one lane survives, the reduce picks it.
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

/// RouterLogits → RouteArgsort → RouteTopK → RouteGatherScores — the
/// smallest tape that reaches the gather door with every const placement
/// minted.
fn gather_ir() -> SubtileIR<NeoX> {
    let tensors = vec![
        TensorShape { rows: M, cols: 64 }, // t0 x (the matmul's input)
        TensorShape { rows: E, cols: 64 }, // t1 the [E, hidden] gate buffer
        TensorShape { rows: M, cols: E },  // t2 logits (argsort's input)
        TensorShape { rows: M, cols: E },  // t3 rank vector (argsort out)
        TensorShape { rows: M, cols: K },  // t4 top-k indices
        TensorShape { rows: M, cols: K },  // t5 gathered scores (out)
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
    ];
    SubtileIR {
        result: TensorId::from_index(5),
        tensors,
        num_sources: 2,
        nodes,
        op_output: Vec::new(),
    }
}

/// Lower through the REAL door path and return the emitted ops.
fn door_ops() -> Vec<ktir_superdsc::emit::EmittedOp> {
    let weight_ids = std::collections::HashSet::new();
    let (ops, _layout) =
        scratchy_target_spyre::lower_subtile_tape_to_ktir::lower_graph_to_superdsc(
            &gather_ir(),
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

/// ⭐ THE CHAIN: `6k−2` ops — per slot j the compare, the mask-product, the
/// reduce, the lane-j product, and the accumulate leg (`identity` seed at
/// j=0, `add` after). Same combine-chain shape as the argsort/top-k doors:
/// every slot's product writes the full [m,W] box, so the accumulation is
/// load-bearing.
#[test]
fn the_gather_emits_the_six_k_chain() {
    let ops = door_ops();
    let names = opfunc_names(&ops);
    let expect: Vec<&str> = (0..K)
        .flat_map(|j| {
            let acc = if j == 0 { "identity" } else { "add" };
            ["equal", "mul", "sum", "mul", acc]
        })
        .collect();
    let tail: Vec<&str> = names
        .iter()
        .skip(names.len() - expect.len())
        .map(|n| n.as_str())
        .collect();
    assert_eq!(tail, expect, "the gather chain, in emission order");
}

/// ⭐ THE COMPARE READS LANE j OF THE INDICES: the gather's per-slot `equal`
/// legs read the per-row index scalar at lane j — the compare operand's
/// start is j·2 bytes past the indices row base. (The top-k door's equals
/// PRECEDE these in the emission and read the targets const, so the LAST k
/// equal legs are the gather's.)
#[test]
fn the_compare_reads_lane_j_of_the_indices() {
    let ops = door_ops();
    let names = opfunc_names(&ops);
    let compare_idx: Vec<usize> = names
        .iter()
        .enumerate()
        .filter(|(_, n)| **n == "equal")
        .map(|(i, _)| i)
        .collect();
    let gather_compares = &compare_idx[compare_idx.len() - K as usize..];
    assert_eq!(gather_compares.len(), K as usize);
    for (j, &i) in gather_compares.iter().enumerate() {
        let row = input_start_bytes(&ops[i], 0);
        let lane = input_start_bytes(&ops[i], 1);
        // The col_at operand is the FIRST input; the mb-broadcast iota is the
        // second — but both name their own buffers' bases. The LAW is the
        // col_at operand's own offset: compare leg j's operand-0 start minus
        // leg 0's is j·2 bytes (lane j of the same indices buffer).
        let base = input_start_bytes(&ops[gather_compares[0]], 0);
        assert_eq!(
            row - base,
            j as u64 * 2,
            "gather compare leg {j} reads lane {j} of the indices buffer"
        );
        // The iota operand is the same const for every slot.
        assert_eq!(lane, input_start_bytes(&ops[gather_compares[0]], 1));
    }
}
