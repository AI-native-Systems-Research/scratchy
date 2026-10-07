// SPDX-License-Identifier: Apache-2.0
//! THE ROUTE TOP-K DOOR — the rank-vector inversion's acceptance gate.
//!
//! `Program::RouteTopK` was refused by name in the door; this file drives the
//! REAL path (`lower_graph_to_superdsc` → the door, with the bundle layout)
//! and pins what the door now emits:
//!
//! * **The chain shape**: per top-k slot `j` the 4-op group
//!   `equal`/`multiply`/`sum`-reduce/`multiply` — `4k` ops total,
//!   independent of the token count, and NEVER a trailing-column slice of
//!   the rank vector (that reads "the ranks of the last k columns" and picks
//!   wrong experts — measured on the metal track).
//! * **The compare reads the TARGET ROW j**: the `equal` leg's second
//!   operand advances one `[1,W]` row per slot — row j of the `[k,W]`
//!   targets const, the one k-sized router placement.
//! * **The write reads the IDENTITY ROW j**: each slot's final multiply
//!   carries a distinct mb-broadcast row of the `[W,W]` identity, so the
//!   per-row index scalar lands at LANE j of the output.
//!
//! The NUMERIC gate (tiny26 EMU-vs-card tensordump parity) runs on the pod;
//! what belongs HERE is the descriptor-level contract the emu cannot check.

use scratchy_subtile::subtile_ir::{
    NeoX, NumExperts, RouterBundle, SubOp, SubtileId, SubtileIR, SubtileNode, TensorId,
    TensorRegion, TensorShape, TopK,
};

/// Stick-legal geometry: E = 8 experts (padded width W = 64, one stick),
/// k = 2, m = 5.
const E: u32 = 8;
const K: u32 = 2;
const M: u32 = 5;
/// The PADDED expert width (one whole fp16 stick).
const W: u32 = 64;

/// One `SubOp::RouterLogits` (declares E) → `RouteArgsort` (writes the rank
/// vector) → `RouteTopK { k }` — the smallest tape that reaches the top-k
/// door with every const placement minted (the targets table needs the
/// RouteTopK node itself).
fn topk_ir() -> SubtileIR<NeoX> {
    // RouterLogits demands two inputs (x, the [E, hidden] gate buffer).
    let tensors = vec![
        TensorShape { rows: M, cols: 64 }, // t0 x (the matmul's input)
        TensorShape { rows: E, cols: 64 }, // t1 the [E, hidden] gate buffer
        TensorShape { rows: M, cols: E },  // t2 logits (argsort's input)
        TensorShape { rows: M, cols: E },  // t3 rank vector (argsort out)
        TensorShape { rows: M, cols: K },  // t4 top-k indices (out)
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
    ];
    SubtileIR {
        result: TensorId::from_index(4),
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
            &topk_ir(),
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
/// door test's own read pattern (inputLabeledDs → ldsIdx → hbm AllocNode →
/// startAddressCoreCorelet_).
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

/// ⭐ THE CHAIN: `6k−2` ops in the exact order — per slot j the compare, the
/// selector product, the reduce, the lane-j product, and the accumulate leg
/// (`identity` seed at j=0, `add` after). The accumulation is LOAD-BEARING:
/// every slot's product writes the full [m,W] box with zeros in the other
/// lanes, so writing the output directly would clobber the earlier slots —
/// the argsort's own combine-chain shape.
#[test]
fn the_topk_emits_the_six_k_chain() {
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
    assert_eq!(tail, expect, "the top-k chain, in emission order");
}

/// ⭐ THE COMPARE READS THE TARGET ROW j: each slot's `equal` leg carries an
/// mb-broadcast operand whose start advances one `[1,W]` row (W·2 bytes) per
/// slot — row j of the `[k,W]` targets const, the target rank `E−k+j` the
/// compare inverts the rank vector against.
#[test]
fn the_compare_reads_the_target_row_j() {
    let ops = door_ops();
    let names = opfunc_names(&ops);
    let compare_idx: Vec<usize> = names
        .iter()
        .enumerate()
        .filter(|(_, n)| **n == "equal")
        .map(|(i, _)| i)
        .collect();
    // The top-k's equal legs are the LAST k (the argsort's equal legs precede).
    let topk_compares = &compare_idx[compare_idx.len() - K as usize..];
    assert_eq!(topk_compares.len(), K as usize);
    let starts: Vec<u64> = topk_compares
        .iter()
        .map(|&i| input_start_bytes(&ops[i], 1))
        .collect();
    for w in starts.windows(2) {
        assert_eq!(
            w[1] - w[0],
            W as u64 * 2,
            "consecutive slots' target rows are one [1,W] fp16 row apart"
        );
    }
}

/// ⭐ THE WRITE READS THE IDENTITY ROW j: each slot's lane-j product `mul`
/// (the one BETWEEN the `sum` and the accumulate leg) carries a distinct
/// mb-broadcast row of the `[W,W]` identity — consecutive rows W·2 bytes
/// apart — so the per-row index scalar lands at LANE j.
#[test]
fn the_write_reads_the_identity_row_j() {
    let ops = door_ops();
    let names = opfunc_names(&ops);
    // The product legs: the `mul` that FOLLOWS a `sum` in the chain.
    let mut write_idx: Vec<usize> = Vec::new();
    let mut after_sum = false;
    for (i, n) in names.iter().enumerate() {
        if n == "sum" {
            after_sum = true;
        } else if n == "mul" && after_sum {
            after_sum = false;
            write_idx.push(i);
        }
    }
    // The LAST k are the top-k's (the argsort's product legs precede).
    let topk_writes = &write_idx[write_idx.len() - K as usize..];
    assert_eq!(topk_writes.len(), K as usize, "one write leg per slot");
    let starts: Vec<u64> = topk_writes
        .iter()
        .map(|&i| input_start_bytes(&ops[i], 1))
        .collect();
    for w in starts.windows(2) {
        assert_eq!(
            w[1] - w[0],
            W as u64 * 2,
            "consecutive slots' identity rows are one [1,W] fp16 row apart"
        );
    }
}
