// SPDX-License-Identifier: Apache-2.0
//! THE FUSED TOP-K DOOR — the wall-11 fix's acceptance gate.
//!
//! `Program::RouteTopK` is the fused vendor pair: the splice folds the
//! argsort and the score gather into it (`lower_all`'s two router folds),
//! binds its parameters in the MINT ORDER `[lg, vals, ids]`, and this file
//! drives the REAL path (`lower_graph_to_superdsc` → the door, with the
//! bundle layout) and pins what the door emits:
//!
//! * **The fold**: the chain's argsort and gather emit NOTHING — no
//!   `maximum` sanitize, no `lesserthan`/`equal` compare legs, anywhere in
//!   the emission. The old one-hot chains they would have run are the
//!   wall-11 culprit (the argsort's rank targets could DUPLICATE at ties
//!   and sum expert ids past the bank, faulting the 26b card's FETCH).
//! * **The chain shape**: `rtkstg` (the stage to `[m_pad, W]`, an `add`
//!   against the pad row at `W > E`) → `rtkkt` (the Kᵀ restickify) →
//!   `rtkix`/`rtkvl` (the vendor `topkindex`/`topkvalue` pair) →
//!   `rtkfl`/`rtkxf`/`rtkxi` (the flat-u32 index converts, at EVERY `m`) →
//!   `rtkti`/`rtktv` (the landing transposes, token-major `[m_pad, 64]`).
//! * **The wiring**: the pair reads ONE Kᵀ table; the flat restickify and
//!   the ids landing read the SAME pair index output; the landings write
//!   the chain's own ids/values tensors at their placed bases; the stage's
//!   sanitize operand IS the pad-row const's placement.
//! * **The m=1 regime**: the convert trio runs there TOO — one owner. The
//!   decode arm of `expert_matmul` used to mint its own staged `[1, W]`
//!   pair over the landing, a form with two card-measured defects at the
//!   26b (the staged widening halved the element stream, and the `[1, W]`
//!   window read REPLICA RUNS, not compact ids — together they staged the
//!   top-2 experts ×4); the trio at every `m` is what replaced it, so the
//!   door never shares the index scratch with a second mint.
//!
//! The NUMERIC gate (tiny26 EMU-vs-card tensordump parity) runs on the pod;
//! what belongs HERE is the descriptor-level contract the emu cannot check.

use scratchy_subtile::subtile_ir::{
    NeoX, NumExperts, RouterBundle, SubOp, SubtileIR, SubtileId, SubtileNode, TensorId,
    TensorRegion, TensorShape, TopK,
};

/// Stick-legal geometry: E = 8 experts (padded width W = 64, one stick — so
/// `W > E` and the stage's add-form sanitize is REAL), k = 2, m = 5. The
/// width itself is the placement's own fact (the byte pins read it off the
/// layout), so no `W` const — the geometry lives in the shapes below.
const E: u32 = 8;
const K: u32 = 2;
const M: u32 = 5;
/// The ids output's tensor index — every `rtk*` op name carries it as its
/// suffix (`_o{tid}`), so the census below is the door's own naming law.
const IDS_TID: u32 = 4;

/// The REAL router chain: `RouterLogits` (declares E) → `RouteArgsort`
/// (folded — its only consumer is the top-k) → `RouteTopK { k }` →
/// `RouteGatherScores` (folded — its ids are the top-k's and its source is
/// the same logits). This is the shape every MoE model lowers to, and the
/// binding walk refuses anything less: `desc_l` is the argsort's producer's
/// input, `desc_v` is the gather's output, `desc_o` is the node's own.
fn chain_ir(m: u32) -> SubtileIR<NeoX> {
    // RouterLogits demands two inputs (x, the [E, hidden] gate buffer).
    let tensors = vec![
        TensorShape { rows: m, cols: 64 }, // t0 x (the matmul's input)
        TensorShape { rows: E, cols: 64 }, // t1 the [E, hidden] gate buffer
        TensorShape { rows: m, cols: E },  // t2 logits (the pair's input)
        TensorShape { rows: m, cols: E },  // t3 rank vector (argsort out)
        TensorShape { rows: m, cols: K },  // t4 top-k indices (ids landing)
        TensorShape { rows: m, cols: K },  // t5 gathered scores (values landing)
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

/// Lower through the REAL door path and return the emitted ops WITH the
/// layout (the byte-level pins below read both).
fn through_the_door(
    m: u32,
) -> (
    Vec<ktir_superdsc::emit::EmittedOp>,
    ktir_superdsc::placement::BundleLayout,
) {
    let weight_ids = std::collections::HashSet::new();
    scratchy_target_spyre::lower_subtile_tape_to_ktir::lower_graph_to_superdsc(
        &chain_ir(m),
        &weight_ids,
        ktir_superdsc::ktir_node::ActiveCap::FULL,
        false,
    )
    .unwrap_or_else(|e| panic!("lower through the door: {e:?}"))
}

/// The door's own ops — the `rtk*`-named ones, in emission order. The
/// RouterLogits node's own program (the logits matmul) leads the emission;
/// filtering by the door's naming law (`_o{ids_tid}`) is robust to whatever
/// it emits.
fn rtk_ops(ops: &[ktir_superdsc::emit::EmittedOp]) -> Vec<ktir_superdsc::emit::EmittedOp> {
    ops.iter()
        .filter(|o| o.op_name.ends_with(&format!("_o{IDS_TID}")) && o.op_name.starts_with("rtk"))
        .cloned()
        .collect()
}

/// The emitted `opFuncName`s of the given ops, in emission order.
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
/// `startAddressCoreCorelet_`).
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

/// The op's OUTPUT tensor's core-0 start HBM bytes — the sort door test's
/// own read pattern (the LAST `outputLabeledDs` names the output's
/// allocate).
fn output_start_bytes(o: &ktir_superdsc::emit::EmittedOp) -> u64 {
    let dsc = o.dsc();
    let d = dsc
        .dscs_
        .first()
        .and_then(|m| m.values().next())
        .expect("an emitted op has at least one dsc");
    let out_lds = d
        .computeOp_
        .first()
        .and_then(|c| c.outputLabeledDs.last())
        .cloned()
        .unwrap_or_default();
    let idx: u32 = out_lds
        .rsplit("idx")
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or(u32::MAX);
    let alloc = d
        .scheduleTree_
        .iter()
        .find(|n| n.nodeType_ == "allocate" && n.component_ == "hbm" && n.ldsIdx_ == idx)
        .unwrap_or_else(|| panic!("the output's hbm allocate (lds {idx}, from {out_lds:?})"));
    alloc
        .startAddressCoreCorelet_
        .data_
        .get("[0, 0, 0]")
        .and_then(|s| s.parse().ok())
        .expect("core 0's start address")
}

/// ⭐ THE FOLD: the argsort and the score gather emit NOTHING. The old
/// one-hot chains are the wall-11 culprit — the argsort's rank targets
/// could DUPLICATE at ties and sum expert ids past the bank — so a chain
/// that still emitted `maximum`/`lesserthan` (the argsort's sanitize and
/// compare legs) or `equal` (both chains' compare legs) would be running
/// the convicted form under the new name.
#[test]
fn the_folded_argsort_and_gather_emit_nothing() {
    let (ops, _layout) = through_the_door(M);
    let names = opfunc_names(&ops);
    for convicted in ["maximum", "lesserthan", "equal"] {
        assert!(
            !names.iter().any(|n| n == convicted),
            "the folded chain must not emit `{convicted}` — that is the old one-hot \
             compare form whose duplicate rank targets faulted the 26b card's FETCH \
             (wall 11); got {names:?}"
        );
    }
}

/// ⭐ THE CHAIN: the door's NINE ops at m>1, by the door's own naming law
/// and in order — the stage (`add`, the add-form pad sanitize this
/// geometry demands: `W > E`), the Kᵀ restickify, the vendor pair, the
/// flat-u32 convert trio (m>1 only), and the two landing transposes. The
/// opFunc names pin the VENDOR ops themselves (`topkindex`/`topkvalue` and
/// the restickify/convert forms) — the pair the registry rows bind.
#[test]
fn the_door_emits_the_fused_chain_in_order() {
    let (ops, _layout) = through_the_door(M);
    let door = rtk_ops(&ops);
    let names: Vec<&str> = door.iter().map(|o| o.op_name.as_str()).collect();
    let expect = [
        format!("rtkstg_g0_o{IDS_TID}"),
        format!("rtkkt_o{IDS_TID}"),
        format!("rtkix_o{IDS_TID}"),
        format!("rtkvl_o{IDS_TID}"),
        format!("rtkfl_o{IDS_TID}"),
        format!("rtkxf_o{IDS_TID}"),
        format!("rtkxi_o{IDS_TID}"),
        format!("rtkti_o{IDS_TID}"),
        format!("rtktv_o{IDS_TID}"),
    ];
    assert_eq!(
        names, expect,
        "the fused chain's op names, in emission order (E = {E} ⇒ one stage group)"
    );
    let funcs = opfunc_names(&door);
    let expect_funcs = [
        "add",
        "ReStickifyOpHBM",
        "topkindex",
        "topkvalue",
        "ReStickifyOpHBM",
        "dl16tofp32",
        "fp32toint32",
        "ReStickifyOpHBM",
        "ReStickifyOpHBM",
    ];
    let got: Vec<&str> = funcs.iter().map(|f| f.as_str()).collect();
    assert_eq!(got, expect_funcs, "the fused chain's opFunc names");
}

/// ⭐ THE WIRING: each leg reads exactly the buffer the leg before it
/// wrote, and the landings write the chain's own tensors at their placed
/// bases — the fold's whole point is that the pair's outputs ARE the
/// chain's ids and values, so a mis-bound leg would corrupt the expert
/// gather's indices silently.
#[test]
fn the_legs_read_what_the_previous_leg_wrote() {
    let (ops, layout) = through_the_door(M);
    let door = rtk_ops(&ops);
    assert_eq!(door.len(), 9, "the m>1 chain (see the census test)");
    // The stage reads the logits at their placed base and sanitizes with
    // the pad row at ITS placed base (the add-form sanitize's own const).
    // ⛔ THE PIN IS THE FULL ADDRESS — `segment_base(segment) + offset`: an
    // operand's core-0 start carries its segment's HBM base, and the router
    // consts ride their OWN segment (the activation happens to sit on the
    // base-zero segment, so a raw-offset pin works there BY the base being
    // zero — the pad row does not: its segment's base is `0xC_0000_0000`,
    // and a raw-offset comparison reads a 48-GiB disagreement out of the
    // segment law, not out of the wiring).
    let placements = &layout.placements;
    let full_address = |tid: u32| {
        let p = placements
            .get(&tid)
            .unwrap_or_else(|| panic!("t{tid} is placed"));
        ktir_superdsc::wire::segment_base(p.segment)
            .unwrap_or_else(|e| panic!("t{tid}'s segment base: {e:?}"))
            + p.offset
    };
    assert_eq!(
        input_start_bytes(&door[0], 0),
        full_address(2),
        "the stage reads the logits tensor at its placed base"
    );
    assert_eq!(
        input_start_bytes(&door[0], 1),
        full_address(ktir_superdsc::reserved_tids::router_topk_pad_tid()),
        "the stage's sanitize operand IS the add-form pad row's placement"
    );
    // The pair reads ONE Kᵀ table: both members' inputs are the restickify's
    // output.
    let kt_out = output_start_bytes(&door[1]);
    assert_eq!(
        input_start_bytes(&door[2], 0),
        kt_out,
        "topkindex reads the Kᵀ restickify's output"
    );
    assert_eq!(
        input_start_bytes(&door[3], 0),
        kt_out,
        "topkvalue reads the SAME Kᵀ table"
    );
    // The index landing and the flat restickify both read the pair's INDEX
    // output; the value landing reads the pair's VALUE output.
    let six_out = output_start_bytes(&door[2]);
    let svl_out = output_start_bytes(&door[3]);
    assert_ne!(
        six_out, svl_out,
        "the pair's two outputs are distinct buffers"
    );
    assert_eq!(
        input_start_bytes(&door[4], 0),
        six_out,
        "the flat restickify reads the pair's index output"
    );
    assert_eq!(
        input_start_bytes(&door[7], 0),
        six_out,
        "the ids landing transpose reads the same pair index output"
    );
    assert_eq!(
        input_start_bytes(&door[8], 0),
        svl_out,
        "the values landing transpose reads the pair's value output"
    );
    // The converts chain: widen reads the flat restickify's output, narrow
    // reads the widen's.
    assert_eq!(
        input_start_bytes(&door[5], 0),
        output_start_bytes(&door[4]),
        "the widen convert reads the flat restickify's output"
    );
    assert_eq!(
        input_start_bytes(&door[6], 0),
        output_start_bytes(&door[5]),
        "the narrow convert reads the widen's output"
    );
    // The landings write the chain's own tensors at their placed bases —
    // the ids t4 and the values t5 (the gather's output, the fold's law).
    // Same FULL-ADDRESS law as above (these two ride the base-zero
    // activation segment, so the raw offsets happen to match — the pin
    // states the address, not the accident).
    assert_eq!(
        output_start_bytes(&door[7]),
        full_address(4),
        "the ids landing writes t4 at its placed base"
    );
    assert_eq!(
        output_start_bytes(&door[8]),
        full_address(5),
        "the values landing writes t5 (the gather's output) at its placed base"
    );
}

/// ⭐ THE PAIR'S DATASTAGE MINIMUM ON OUT IS ONE SLOT — not the per-core slice
/// (slots/ring). A slice-stated minimum (8 at this geometry's 64-slot landing,
/// ring 8) makes the vendor DDC's `allocAllMem` minimum-datastage check refuse
/// the whole bundle at bake ("Cannot allocate even the smallest size",
/// ddcv1.cpp:1351 — the 26b decode bundle's group_7 refusal, card-mutation-
/// proven: changing ONLY `ss_`/`el_` `out_` 8→1 compiles the same dsc clean).
/// ONE is the vendor fixtures' own per-core out geometry, and the DDC
/// maximizes the datastage upward itself once the min-check passes — so `N_`,
/// the 8-way out work split, and the landing bytes are unchanged. The rest of
/// the minimum stays structural: the never-split `in` scan and the token
/// stick.
#[test]
fn the_pair_datastage_minimum_on_out_is_one_slot() {
    let (ops, _layout) = through_the_door(M);
    let door = rtk_ops(&ops);
    for (i, member) in [(2usize, "rtkix"), (3, "rtkvl")] {
        let d = door[i]
            .dsc()
            .dscs_
            .first()
            .and_then(|m| m.values().next())
            .unwrap_or_else(|| panic!("{member}'s dsc"));
        let stage = d
            .dataStageParam_
            .get("0")
            .unwrap_or_else(|| panic!("{member}'s core stage (exactly one stage)"));
        assert_eq!(
            stage.ss_.out_, 1,
            "{member}: ss_ out streams ONE slot — the slice-stated minimum refused the \
             26b decode bundle at bake (allocAllMem, ddcv1.cpp:1351)"
        );
        assert_eq!(
            stage.el_.out_, 1,
            "{member}: el_ out streams ONE slot (the same law on the element stage)"
        );
        assert_eq!(
            stage.ss_.in_, 64,
            "{member}: the in scan stays whole at its extent (the reduction never splits)"
        );
    }
}

/// ⭐ THE m=1 REGIME RUNS THE CONVERT TRIO TOO — ONE OWNER: the decode arm
/// of `expert_matmul` used to mint its own staged `[1, W]` index converts,
/// and that form was the 26b's staging defect (the staged widening halved
/// the element stream and the `[1, W]` window read replica runs — the top-2
/// experts ×4, scale-fingerprint-proven on the card). The trio now mints at
/// EVERY `m` over the pair's own flat `[64, m_pad]`, so the m=1 chain is the
/// m>1 chain verbatim: stage, Kᵀ, pair, flat restickify, two converts, two
/// landings — nine ops, and no second mint can collide on the index scratch.
#[test]
fn the_m1_regime_runs_the_convert_trio() {
    let (ops, _layout) = through_the_door(1);
    let door = rtk_ops(&ops);
    let names: Vec<&str> = door.iter().map(|o| o.op_name.as_str()).collect();
    let expect = [
        format!("rtkstg_g0_o{IDS_TID}"),
        format!("rtkkt_o{IDS_TID}"),
        format!("rtkix_o{IDS_TID}"),
        format!("rtkvl_o{IDS_TID}"),
        format!("rtkfl_o{IDS_TID}"),
        format!("rtkxf_o{IDS_TID}"),
        format!("rtkxi_o{IDS_TID}"),
        format!("rtkti_o{IDS_TID}"),
        format!("rtktv_o{IDS_TID}"),
    ];
    assert_eq!(
        names, expect,
        "the m=1 chain is the m>1 chain verbatim — the trio is the only index mint"
    );
}
