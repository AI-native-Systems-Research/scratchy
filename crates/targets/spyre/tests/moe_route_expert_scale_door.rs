// SPDX-License-Identifier: Apache-2.0
//! THE ROUTE EXPERT-SCALE DOOR — the per-expert scale select's acceptance
//! gate.
//!
//! `Program::RouteExpertScale` was refused by name in the door; this file
//! drives the REAL path (`lower_graph_to_superdsc` → the door, with the
//! bundle layout) and pins what the door now emits:
//!
//! * **The chain shape**: per slot `j` the 6-op group
//!   `equal`/`multiply`(scale)/`multiply`(score)/`sum`-reduce/`multiply`
//!   (identity-row) + accumulate (`identity` seed at j=0, `add` after) —
//!   `6k` ops, independent of the token count.
//! * **The select masks the SCALE ROW** (a loaded weight, never a bake-time
//!   const): the second leg's second operand is the scale tensor, the
//!   mb-broadcast `[1,W]` parent every row gathers from.
//! * **The score rides the matched lane's REPLICA-RUN HEAD**: the third
//!   leg's second operand reads lane j·rep of the scores buffer (rep =
//!   64/k — the pair replicates each result 64/k lanes wide, so the head is
//!   the one lane per result that holds result j) — j·rep·2 bytes past the
//!   row base.
//!
//! ⛔ THE CHAIN IS THE WAVEFRONT'S OWN, NOT THE SHORTCUT: the gemma arm
//! feeds the door the SOFTMAX's output (`gather → scale → softmax →
//! expert-scale`, `to_wavefront`'s own wiring) — a score-side tensor on the
//! `router_score_bytes` carve, NOT the gather's folded landing. This file's
//! first tape fed the gather's output directly (the landing carve, whose
//! pad rows absorb the lane walk) and passed while the REAL tape refused at
//! m=1 (`reprd_j1_o203`: `2B + 128B > 128B` — the score carve had no
//! `col_at` tail). The honest wiring is the regression pin for that tail:
//! the door's lane-j reads walk `(k−1) + m·64` elements into the softmax
//! output's reservation, and a carve without the tail refuses HERE first.
//!
//! The NUMERIC gate (tiny26 EMU-vs-card tensordump parity) runs on the pod.

use scratchy_subtile::subtile_ir::{
    NeoX, NumExperts, RouterBundle, SubOp, SubtileIR, SubtileId, SubtileNode, TensorId,
    TensorRegion, TensorShape, TopK,
};

/// Stick-legal geometry: E = 8 experts (padded width W = 64, one stick),
/// k = 8 — the splice rows' own floor (a score-side block whose last dim is
/// under 16 bytes is refused by the Triton front end, so the honest chain's
/// scale/softmax legs are not spellable at k = 2), m = 5.
const E: u32 = 8;
const K: u32 = 8;
const M: u32 = 5;

/// RouterLogits → RouteArgsort → RouteTopK → RouteGatherScores →
/// RouteScale → RouteSoftmax → RouteExpertScale — the wavefront's own gemma
/// chain (`to_wavefront`'s MoE arm verbatim): the smallest tape that reaches
/// the scale door through the honest wiring, with every const placement
/// minted. t8/t9 sit AFTER the result so every `_o{tid}` pin keeps its id.
/// `k` is the top-k width — the honest chain's own K for the existing arms,
/// 1 for the top-1 arm (the Triton front end refuses a `[·,1]` score-side
/// block — the 16-byte row floor — so this per-op wiring is the drive that
/// reaches the chain at top-1).
fn scale_ir(k: u32) -> SubtileIR<NeoX> {
    let tensors = vec![
        TensorShape { rows: M, cols: 64 }, // t0 x (the matmul's input)
        TensorShape { rows: E, cols: 64 }, // t1 the [E, hidden] gate buffer
        TensorShape { rows: M, cols: E },  // t2 logits (argsort's input)
        TensorShape { rows: M, cols: E },  // t3 rank vector (argsort out)
        TensorShape { rows: M, cols: k },  // t4 top-k indices
        TensorShape { rows: M, cols: k },  // t5 gathered scores (the pair's vals landing)
        TensorShape { rows: 1, cols: E },  // t6 the [1,E] scale row
        TensorShape { rows: M, cols: k },  // t7 scaled scores (out)
        TensorShape { rows: M, cols: k },  // t8 the temperature-scaled scores
        TensorShape { rows: M, cols: k },  // t9 the softmax's output (the door's scores)
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
                k: TopK::new(std::num::NonZeroU32::new(k).unwrap()),
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
        // The honest chain's two score-side legs — the wavefront's own order
        // and payloads (`to_wavefront`: the hidden^-0.5 temperature, then the
        // softmax over the chosen k). They exist here so the door's scores
        // operand is the SOFTMAX's output — the score-carved tensor the real
        // tape feeds it — not the gather's folded landing.
        SubtileNode {
            id: SubtileId::from_index(5),
            op: SubOp::RouteScale {
                scale: (64f32).powf(-0.5),
            },
            inputs: vec![tr(5)],
            output: tr(8),
        },
        SubtileNode {
            id: SubtileId::from_index(6),
            op: SubOp::RouteSoftmax,
            inputs: vec![tr(8)],
            output: tr(9),
        },
        SubtileNode {
            id: SubtileId::from_index(4),
            op: SubOp::RouteExpertScale {
                router: RouterBundle::Gemma,
            },
            inputs: vec![tr(9), tr(4), tr(6)],
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
fn door_ops(k: u32) -> Vec<ktir_superdsc::emit::EmittedOp> {
    let weight_ids = std::collections::HashSet::new();
    let (ops, _layout) =
        scratchy_target_spyre::lower_subtile_tape_to_ktir::lower_graph_to_superdsc(
            &scale_ir(k),
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

/// ⭐ THE CHAIN: 6 ops per slot (`6k` — the count coincides with `7k−2`
/// only at k=2, the count this file first ran at) — per slot j the compare,
/// the scale-select product, the score product, the reduce, the lane-j
/// product, and the accumulate leg (`identity` seed at j=0, `add` after).
#[test]
fn the_scale_emits_the_seven_k_chain() {
    let ops = door_ops(K);
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
    let ops = door_ops(K);
    let names = opfunc_names(&ops);
    // The LAST 6k ops are the scale door's chain: find the tail 6k window,
    // then index within it.
    let tail_len = (6 * K) as usize;
    let start = names.len() - tail_len;
    let tail = &names[start..];
    let mut scale_legs = Vec::new();
    let mut score_legs = Vec::new();
    let mut i = 0;
    while i + 6 <= tail.len() {
        if tail[i..i + 6]
            == [
                "equal",
                "mul",
                "mul",
                "sum",
                "mul",
                if i == 0 { "identity" } else { "add" },
            ]
        {
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
    assert_eq!(
        bases.len(),
        1,
        "every slot's select reads the same scale row"
    );
    // The score operand (input 1 of the score leg) advances ONE REPLICA RUN
    // per slot — lane j·rep of the landing's 64-slot rows (rep = 64/k: the
    // pair writes result r at lanes [r·rep, (r+1)·rep), so the HEAD lane
    // j·rep is the one lane per result that holds result j; the old lane-j
    // read result j/rep's replica, which at k=8 made every leg read
    // result 0).
    let score_starts: Vec<u64> = score_legs
        .iter()
        .map(|&i| input_start_bytes(&ops[i], 1))
        .collect();
    let rep = 64 / K;
    for (j, s) in score_starts.iter().enumerate() {
        assert_eq!(
            s - score_starts[0],
            (j as u64) * (rep as u64) * 2,
            "score leg {j} reads lane {j}·{rep} of the scores buffer — the head of result {j}"
        );
    }
}

/// ⭐ TOP-1: at k=1 the seed's product IS the result — the chain writes its
/// output at the OUT tensor and no accumulating add follows. The unfixed
/// chain seeded the accumulator and never wrote `out` at all (top-1 routing
/// is a real geometry — Llama-4-class and Switch-style MoEs — and 1 divides
/// the landing's 64 slots, so only the write, never the shape, was wrong).
/// The seed's name carries the out tid (`reseed_o{tid}`); its output binding
/// must BE that tensor, not the accumulator synth. The gather one node
/// upstream never runs in this chain at ANY k — `lower_all` folds it into
/// the fused top-k pair whenever the pair's logits are the gather's source
/// (the wall-11 fold; `rtkvl_o4` IS the gather) — so this arm pins the fold,
/// and the gather door's own top-1 write is pinned one test down, at the
/// fold's leftover corner.
#[test]
fn the_chain_writes_its_output_at_top_1() {
    let ops = door_ops(1);
    let seed = ops
        .iter()
        .find(|o| o.op_name.starts_with("reseed_"))
        .unwrap_or_else(|| {
            panic!(
                "one seed leg at top-1: {:?}",
                ops.iter().map(|o| &o.op_name).collect::<Vec<_>>()
            )
        });
    assert!(
        !ops.iter().any(|o| o.op_name.starts_with("reacc_")),
        "no accumulating adds at top-1"
    );
    assert!(
        !ops.iter()
            .any(|o| o.op_name.starts_with("rgseed_") || o.op_name.starts_with("rgacc_")),
        "the gather is folded into the fused pair at every k — rtkvl IS the gather"
    );
    let tid: u32 = seed
        .op_name
        .strip_prefix("reseed_o")
        .and_then(|s| s.parse().ok())
        .expect("the seed's name carries the out tid");
    let out_binding = seed
        .arg_bindings
        .iter()
        .find(|b| !b.is_input)
        .expect("the seed states an output");
    assert_eq!(
        out_binding.buffer,
        ktir_superdsc::place::act_name(tid),
        "the top-1 seed writes the out tensor itself, not the accumulator"
    );
}

/// The gather door's LEFTOVER CORNER — a gather whose indices arrive from
/// anywhere but a RouteTopK over the same source (the only shape that
/// reaches `route_gather_scores`; in the router chain the fused pair lands
/// the values itself and this door never runs). The indices are a SOURCE,
/// and the RouterLogits head exists to gate the router-const block (the
/// identity table and the iota row the door reads are placed only for a
/// tape that carries a logits node — a source-only tape would refuse at the
/// undeclared-synth guard).
fn gather_ir(k: u32) -> SubtileIR<NeoX> {
    let tensors = vec![
        TensorShape { rows: M, cols: 64 }, // t0 x (the logits matmul's input)
        TensorShape { rows: E, cols: 64 }, // t1 the [E, hidden] gate buffer
        TensorShape { rows: M, cols: E },  // t2 logits (the gather's scores)
        TensorShape { rows: M, cols: k },  // t3 the indices (a source — not a top-k landing)
        TensorShape { rows: M, cols: k },  // t4 the gather's out
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
            op: SubOp::RouteGatherScores,
            inputs: vec![tr(2), tr(3)],
            output: tr(4),
        },
    ];
    let num_sources = tensors.len() as u32;
    SubtileIR {
        result: TensorId::from_index(4),
        tensors,
        num_sources,
        nodes,
        op_output: Vec::new(),
    }
}

/// ⭐ THE GATHER'S OWN TOP-1 LAW — the same fix's other door: at k=1 the
/// gather's seed writes the OUT tensor (`rgseed_o{tid}`'s output binding),
/// no accumulating add follows, and the unfixed chain never wrote `out` at
/// all.
#[test]
fn the_gather_writes_its_output_at_top_1() {
    let weight_ids = std::collections::HashSet::new();
    let (ops, _layout) =
        scratchy_target_spyre::lower_subtile_tape_to_ktir::lower_graph_to_superdsc(
            &gather_ir(1),
            &weight_ids,
            ktir_superdsc::ktir_node::ActiveCap::FULL,
            false,
        )
        .unwrap_or_else(|e| panic!("lower the leftover-corner gather at k=1: {e:?}"));
    let seed = ops
        .iter()
        .find(|o| o.op_name.starts_with("rgseed_"))
        .unwrap_or_else(|| {
            panic!(
                "one gather seed leg at top-1: {:?}",
                ops.iter().map(|o| &o.op_name).collect::<Vec<_>>()
            )
        });
    assert!(
        !ops.iter().any(|o| o.op_name.starts_with("rgacc_")),
        "no gather accumulating adds at top-1"
    );
    let tid: u32 = seed
        .op_name
        .strip_prefix("rgseed_o")
        .and_then(|s| s.parse().ok())
        .expect("the gather seed's name carries the out tid");
    let out_binding = seed
        .arg_bindings
        .iter()
        .find(|b| !b.is_input)
        .expect("the gather seed states an output");
    assert_eq!(
        out_binding.buffer,
        ktir_superdsc::place::act_name(tid),
        "the top-1 gather seed writes the out tensor itself, not the accumulator"
    );
}
