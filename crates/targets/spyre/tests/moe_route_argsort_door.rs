// SPDX-License-Identifier: Apache-2.0
//! THE ROUTE ARGSORT DOOR — the rank-vector lowering's acceptance gate.
//!
//! `Program::RouteArgsort` was refused by name in the door; this file drives
//! the REAL path (`lower_graph_to_superdsc` → the door, with the bundle
//! layout) and pins what the door now emits:
//!
//! * **The chain shape**: 1 sanitize `maximum` + per ranked expert `j` the
//!   6-op group `lesserthan`/`equal`/`multiply`/`add`/`sum`-reduce/`multiply`
//!   plus the accumulate leg (`identity` seed at j=0, `add` after) —
//!   `7W + 1` ops total (W = the padded expert width), independent of the token count.
//! * **The pad-mask sanitize leads**: the FIRST op is `maximum(x, mask)` —
//!   the leg that makes the producer's zero-padded lanes sort last. A door
//!   without it computes ranks that count pad lanes among the smallest.
//! * **The per-j compare reads the score at LANE j**: the `col_at` RedStick
//!   read, the combine door's own card-proven law — pinned here as distinct,
//!   monotonically advancing per-j operand offsets.
//! * **The one-hot product writes LANE j**: each slot's final multiply reads
//!   a DISTINCT one-hot const row, so the accumulated output has rank_j in
//!   lane j and zeros elsewhere.
//!
//! The NUMERIC gate (tiny26 EMU-vs-card tensordump parity, and the emu's own
//! `route_argsort_matches_the_stable_ascending_rank`) runs elsewhere; what
//! belongs HERE is the descriptor-level contract the emu cannot check.

use scratchy_subtile::subtile_ir::{
    NeoX, NumExperts, RouterBundle, SubOp, SubtileId, SubtileIR, SubtileNode, TensorId,
    TensorRegion, TensorShape,
};

/// Stick-legal geometry: E = 8 experts (padded width W = 64, one stick), m = 5.
const E: u32 = 8;
const M: u32 = 5;
/// The PADDED expert width — the loop bound the door runs to (the top-k's
/// full-stick compare reads the pad lanes' ranks, so they must be written).
const W: u32 = 64;

/// One `SubOp::RouterLogits` (declares E, places the consts) feeding one
/// `SubOp::RouteArgsort` — the smallest tape that reaches the door with the
/// const placements minted.
fn argsort_ir() -> SubtileIR<NeoX> {
    // RouterLogits demands two inputs (x, the [E, hidden] gate buffer rides in
    // the weights — but the door only needs the arity, so it is fed from a
    // source region here like the e2e chain test feeds it).
    let tensors = vec![
        TensorShape { rows: M, cols: 64 }, // t0 x (the matmul's input)
        TensorShape { rows: E, cols: 64 }, // t1 the [E, hidden] gate buffer
        TensorShape { rows: M, cols: E },  // t2 logits (argsort's input)
        TensorShape { rows: M, cols: E },  // t3 rank vector (out)
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
    ];
    SubtileIR {
        result: TensorId::from_index(3),
        tensors,
        num_sources: 2,
        nodes,
        op_output: Vec::new(),
    }
}

/// Lower through the REAL door path and return the argsort door's ops (the
/// RouterLogits node lowers to a Matmul program with no SuperDSC ops of its
/// own here, so the emitted pointwise chain is the argsort's).
fn door_ops() -> Vec<ktir_superdsc::emit::EmittedOp> {
    let weight_ids = std::collections::HashSet::new();
    let (ops, _layout) =
        scratchy_target_spyre::lower_subtile_tape_to_ktir::lower_graph_to_superdsc(
            &argsort_ir(),
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

/// ⭐ THE CHAIN: `7W + 1` ops in the exact order — the sanitize leads, then
/// per ranked expert j the compare group, the reduce, the one-hot product,
/// and the accumulate. A door that emitted the compare without the sanitize,
/// or the product without the accumulate, bakes clean and computes the wrong
/// function.
#[test]
fn the_argsort_emits_the_seven_e_plus_one_chain() {
    let names = opfunc_names(&door_ops());
    // The RouterLogits node's OWN program leads (a batchmatmul for the logits
    // themselves); the argsort's chain is everything from its sanitize on.
    let chain: Vec<&str> = names
        .iter()
        .skip_while(|n| *n != "maximum")
        .map(|n| n.as_str())
        .collect();
    let expect: Vec<&str> = std::iter::once("maximum")
        .chain((0..W).flat_map(|j| {
            let acc = if j == 0 { "identity" } else { "add" };
            ["lesserthan", "equal", "mul", "add", "sum", "mul", acc]
        }))
        .collect();
    assert_eq!(chain, expect, "the argsort chain, in emission order");
}

/// The `computeOp_`'s `inputLabeledDs` names → the FIRST input's hbm allocate
/// node → its per-core start bytes (the combine door test's own helper, the
/// same read pattern).
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

/// ⭐ THE COMPARE READS LANE j: the per-j `lesserthan` and `equal` legs each
/// read the SAME sanitized-scores buffer (input 0, the full row) as their
/// first input, and their SECOND input is the col_at broadcast of the same
/// buffer at lane j — whose per-core start is exactly j·2 bytes past input
/// 0's (the RedStick read = base + lane offset, no mb advance). Pinned as:
/// leg (j, kind)'s input-1 start − its input-0 start == j·2.
#[test]
fn the_compare_reads_the_score_at_lane_j() {
    let ops = door_ops();
    let names = opfunc_names(&ops);
    let compare_idx: Vec<usize> = names
        .iter()
        .enumerate()
        .filter(|(_, n)| **n == "lesserthan" || **n == "equal")
        .map(|(i, _)| i)
        .collect();
    assert_eq!(compare_idx.len(), 2 * W as usize, "two compare legs per j");
    for (pair, &i) in compare_idx.iter().enumerate() {
        let j = (pair / 2) as u64;
        let row = input_start_bytes(&ops[i], 0);
        let lane = input_start_bytes(&ops[i], 1);
        assert_eq!(
            lane - row,
            j * 2,
            "compare leg {pair} (expert {j}): the col operand starts j·2 bytes past the row — \
             the lane-{j} single-element read"
        );
    }
}

/// ⭐ THE ONE-HOT PRODUCT READS A DISTINCT ROW PER SLOT: each slot's final
/// `multiply` carries a different mb-broadcast const operand (one-hot row j),
/// so the products land in DISTINCT output lanes. Pinned as: the product
/// legs' input-1 starts are E distinct addresses, consecutive rows one
/// [1, W] fp16 row (W·2 bytes) apart in the router-const placement block.
#[test]
fn the_onehot_products_read_distinct_rows() {
    let ops = door_ops();
    let names = opfunc_names(&ops);
    // The product legs: the multiplies that FOLLOW a sum-reduce in the chain.
    let mut product_idx: Vec<usize> = Vec::new();
    let mut after_sum = false;
    for (i, n) in names.iter().enumerate() {
        if n == "sum" {
            after_sum = true;
        } else if n == "mul" && after_sum {
            after_sum = false;
            product_idx.push(i);
        }
    }
    assert_eq!(product_idx.len(), W as usize, "one product leg per j");
    let mut starts: Vec<u64> = product_idx
        .iter()
        .map(|&i| input_start_bytes(&ops[i], 1))
        .collect();
    let distinct: std::collections::HashSet<u64> = starts.iter().copied().collect();
    assert_eq!(
        distinct.len(),
        W as usize,
        "each slot's one-hot row is a distinct buffer address"
    );
    starts.sort_unstable();
    starts.dedup();
    for w in starts.windows(2) {
        assert_eq!(
            w[1] - w[0],
            64 * 2,
            "consecutive one-hot rows are one [1,64] fp16 row apart"
        );
    }
}
