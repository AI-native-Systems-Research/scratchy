// SPDX-License-Identifier: Apache-2.0
//! THE ROUTE ARGSORT DOOR — the rank-vector lowering's acceptance gate.
//!
//! `Program::RouteArgsort` was refused by name in the door; this file drives
//! the REAL path (`lower_graph_to_superdsc` → the door, with the bundle
//! layout) and pins what the door now emits:
//!
//! * **The chain shape**: 1 sanitize `maximum` + per ranked expert `j` the
//!   8-op compare group led by the ONE-HOT MASK-REDUCE that lands `x[i, j]`
//!   (`multiply`/`sum`, then `lesserthan`/`equal`/`multiply`/`add`/`sum`/
//!   `multiply`) plus the accumulate leg (`identity` seed at j=0, `add`
//!   after) — `9W + 1` ops total (W = the padded expert width), independent
//!   of the token count.
//! * **The pad-mask sanitize leads**: the FIRST op is `maximum(x, mask)` —
//!   the leg that makes the producer's zero-padded lanes sort last. A door
//!   without it computes ranks that count pad lanes among the smallest.
//! * **The compare's B operand is the b_j REDUCE ACCUM, never a scores lane**
//!   (the D1 law): the per-row scalar `x[i, j]` lands at LANE 0 of the
//!   `[m, stick]` accum via the one-hot mask-reduce, and both compare legs
//!   of every slot read that one fixed buffer — pinned here as a CONSTANT
//!   operand delta, the exact signature the convicted mid-stick `col_at`
//!   form lacked (its delta advanced `j·2` per slot: every `j` in a region
//!   compared against the SAME constant, measured on-card).
//! * **The one-hot legs read LANE j's row**: each slot's mask-reduce product
//!   AND its final output product read the SAME one-hot const row `j`, so
//!   the accumulated output has rank_j in lane j and zeros elsewhere.
//!
//! The NUMERIC gate (tiny26 EMU-vs-card tensordump parity, and the emu's own
//! `route_argsort_matches_the_stable_ascending_rank`) runs elsewhere; what
//! belongs HERE is the descriptor-level contract the emu cannot check.

use scratchy_subtile::subtile_ir::{
    NeoX, NumExperts, RouterBundle, SubOp, SubtileIR, SubtileId, SubtileNode, TensorId,
    TensorRegion, TensorShape,
};

/// Stick-legal geometry: E = 8 experts (padded width W = 64, one stick), k =
/// 2 (the bare gather's slots), m = 5.
const E: u32 = 8;
const K: u32 = 2;
const M: u32 = 5;
/// The PADDED expert width — the loop bound the door runs to (the top-k's
/// full-stick compare reads the pad lanes' ranks, so they must be written).
const W: u32 = 64;
/// The rank vector's tensor index — every one of the argsort door's op
/// names carries it as its suffix (`_o{tid}`), so the census filters below
/// are the door's own naming law.
const OUT_TID: u32 = 4;

/// One `SubOp::RouterLogits` (declares E, places the consts) feeding one
/// `SubOp::RouteArgsort` whose ranks feed a bare `RouteGatherScores` with
/// HAND-FED ids — the BARE-ARGSORT corner. ⛔ THE CONSUMER IS LOAD-BEARING:
/// the splice folds an argsort whose every consumer is a RouteTopK into the
/// fused pair (which reads the raw logits and needs no ranks), and an
/// argsort with NO consumer folds vacuously — only a non-topk consumer
/// keeps this door running, which is exactly the tape class the row stays
/// registered for. The gather's ids come from a SOURCE tensor (no
/// producer), so the gather keeps its own program too — the bare corner's
/// other half. ⛔ AND THE IDS MUST SIT INSIDE THE SOURCE PREFIX (t2, under
/// `num_sources`): a hand-fed ids at a COMPUTED slot has no writer, is
/// never placed, and the gather door's operand falls through to the
/// synth-declaration law's panic — so the numbering puts every hand-fed
/// tensor before the chain's first computed one.
fn argsort_ir() -> SubtileIR<NeoX> {
    // RouterLogits demands two inputs (x, the [E, hidden] gate buffer rides in
    // the weights — but the door only needs the arity, so it is fed from a
    // source region here like the e2e chain test feeds it).
    let tensors = vec![
        TensorShape { rows: M, cols: 64 }, // t0 x (the matmul's input)
        TensorShape { rows: E, cols: 64 }, // t1 the [E, hidden] gate buffer
        TensorShape { rows: M, cols: K },  // t2 the gather's hand-fed ids (a source)
        TensorShape { rows: M, cols: E },  // t3 logits (argsort's input)
        TensorShape { rows: M, cols: E },  // t4 rank vector (out)
        TensorShape { rows: M, cols: K },  // t5 the gather's output
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
            output: tr(3),
        },
        SubtileNode {
            id: SubtileId::from_index(1),
            op: SubOp::RouteArgsort,
            inputs: vec![tr(3)],
            output: tr(4),
        },
        // The non-RouteTopK consumer that makes the argsort BARE (keeps its
        // program — and its consts — alive past the fold).
        SubtileNode {
            id: SubtileId::from_index(2),
            op: SubOp::RouteGatherScores,
            inputs: vec![tr(4), tr(2)],
            output: tr(5),
        },
    ];
    SubtileIR {
        result: TensorId::from_index(5),
        tensors,
        num_sources: 3,
        nodes,
        op_output: Vec::new(),
    }
}

/// Lower through the REAL door path and return the ARGSORT door's own ops —
/// the `_o{OUT_TID}`-suffixed ones, by the door's naming law. The
/// RouterLogits node lowers to a Matmul program and the bare gather to its
/// own chain; the suffix filter is robust to whatever either emits.
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
    ops.into_iter()
        .filter(|o| {
            o.op_name
                .rsplit_once("_o")
                .is_some_and(|(_, tid)| tid == OUT_TID.to_string())
        })
        .collect()
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

/// ⭐ THE CHAIN: `9W + 1` ops in the exact order — the sanitize leads, then
/// per ranked expert `j` the one-hot mask-reduce that lands `x[i, j]`, the
/// compare group, the reduce, the one-hot product, and the accumulate. A door
/// that emitted the compare without the sanitize, or the product without the
/// accumulate, bakes clean and computes the wrong function.
#[test]
fn the_argsort_emits_the_nine_w_plus_one_chain() {
    let names = opfunc_names(&door_ops());
    // The filtered set IS the argsort's chain, whole: the sanitize leads and
    // every ranked expert j contributes its 9-op group.
    let expect: Vec<&str> = std::iter::once("maximum")
        .chain((0..W).flat_map(|j| {
            let acc = if j == 0 { "identity" } else { "add" };
            [
                "mul",
                "sum",
                "lesserthan",
                "equal",
                "mul",
                "add",
                "sum",
                "mul",
                acc,
            ]
        }))
        .collect();
    let got: Vec<&str> = names.iter().map(|n| n.as_str()).collect();
    assert_eq!(got, expect, "the argsort chain, in emission order");
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

/// ⭐ THE COMPARE READS THE b_j REDUCE ACCUM, NEVER A SCORES LANE (the D1
/// law): both compare legs of every slot read the SAME fixed buffer — the
/// `[m, stick]` accum the one-hot mask-reduce landed `x[i, j]` at LANE 0 of —
/// so every leg's input-1 minus input-0 delta is ONE CONSTANT. The convicted
/// mid-stick `col_at` form this replaces read the scores row itself at lane
/// `j`, a delta that advanced `j·2` per slot (and on the card selected no
/// lane at all — every `j` in a region compared against the same constant).
#[test]
fn the_compare_reads_the_b_j_accum_not_a_scores_lane() {
    let ops = door_ops();
    let names = opfunc_names(&ops);
    let compare_idx: Vec<usize> = names
        .iter()
        .enumerate()
        .filter(|(_, n)| **n == "lesserthan" || **n == "equal")
        .map(|(i, _)| i)
        .collect();
    assert_eq!(compare_idx.len(), 2 * W as usize, "two compare legs per j");
    let mut deltas: Vec<u64> = Vec::with_capacity(compare_idx.len());
    for &i in &compare_idx {
        let row = input_start_bytes(&ops[i], 0);
        let lane = input_start_bytes(&ops[i], 1);
        assert!(
            lane >= row,
            "the compare's B operand starts before the scores row — a mis-bound operand"
        );
        deltas.push(lane - row);
    }
    let distinct: std::collections::HashSet<u64> = deltas.iter().copied().collect();
    assert_eq!(
        distinct.len(),
        1,
        "every compare leg's B operand is ONE fixed buffer (the b_j reduce accum at \
         lane 0), so the input-1 − input-0 delta is a single constant — got {} distinct \
         deltas, the advancing-per-slot signature of the convicted mid-stick col_at form \
         (every j in a region compared against the SAME constant, measured on-card)",
        distinct.len()
    );
}

/// ⭐ THE ONE-HOT LEGS READ LANE j's ROW: each slot's mask-reduce product
/// (the group's first `mul`) AND its final output product (the group's last
/// `mul`) read the SAME one-hot const row `j` of the `[W, W]` identity — the
/// reduce masks the scores down to lane j and the product writes rank_j back
/// into lane j — so the rows advance one `[1, W]` fp16 row per slot and are
/// W distinct addresses.
#[test]
fn the_onehot_legs_read_lane_js_row() {
    let ops = door_ops();
    let names = opfunc_names(&ops);
    // The chain starts at the sanitize; each 9-op group after it is one
    // ranked expert j (the census test above pins the exact sequence).
    let start = names
        .iter()
        .position(|n| n == "maximum")
        .expect("the sanitize leads the chain");
    let group = [
        "mul",
        "sum",
        "lesserthan",
        "equal",
        "mul",
        "add",
        "sum",
        "mul",
        "add",
    ];
    let mut mask_legs = Vec::with_capacity(W as usize);
    let mut prod_legs = Vec::with_capacity(W as usize);
    for j in 0..W as usize {
        let base = start + 1 + j * group.len();
        let got: Vec<&str> = names[base..base + group.len()]
            .iter()
            .map(|n| n.as_str())
            .collect();
        let expect: Vec<&str> = group
            .iter()
            .copied()
            .enumerate()
            .map(|(i, op)| {
                if i == group.len() - 1 && j == 0 {
                    "identity"
                } else {
                    op
                }
            })
            .collect();
        assert_eq!(got, expect, "slot {j}'s 9-op group, in emission order");
        // Leg 0 is the mask-reduce product; leg 7 the output product — both
        // read the identity's row j as their B operand.
        mask_legs.push(input_start_bytes(&ops[base], 1));
        prod_legs.push(input_start_bytes(&ops[base + 7], 1));
    }
    // Both one-hot legs of a slot read the SAME row j.
    for (j, (&m, &p)) in mask_legs.iter().zip(&prod_legs).enumerate() {
        assert_eq!(
            m, p,
            "slot {j}'s mask-reduce and output products read the same one-hot row"
        );
    }
    // The rows are W distinct addresses, one [1, W] fp16 row apart.
    let distinct: std::collections::HashSet<u64> = prod_legs.iter().copied().collect();
    assert_eq!(
        distinct.len(),
        W as usize,
        "each slot's one-hot row is a distinct buffer address"
    );
    let mut starts = prod_legs.clone();
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
