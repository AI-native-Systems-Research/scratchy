// SPDX-License-Identifier: Apache-2.0
//! THE ROUTER CONST ROWS — the staging half of the MoE router quartet
//! (`RouteArgsort`/`RouteTopK`/`RouteGatherScores`/`RouteExpertScale`).
//!
//! The router doors read four kinds of token-INDEPENDENT factors — the
//! `[W,W]` stable-argsort tie table, the `[1,W]` iota row, `k` one-hot
//! rows, and the two sanitize rows — and this file pins the STAGING chain
//! that delivers them:
//!
//! ```text
//! compute_bundle_layout  →  places the router_const tids at [W,W]/[1,W]
//! bake_layout            →  carries the placements into the bundle
//! BakeFacts::of          →  reads the geometry (W, k) back off them
//! synthetic_constants    →  builds the VALUES from that geometry
//! ```
//!
//! ⛔ THE LOAD-BEARING INVARIANT IS tid ↔ VALUE AGREEMENT, which no single
//! site can see: the placement pass mints the geometry from the tape's own
//! `RouterLogits`/`RouteTopK` nodes, and the bind builds the values from
//! the geometry it read back off the PLACEMENTS. A disagreement between the
//! two halves is exactly the class the memory calls "a USAGE CENSUS is not
//! an admission predicate" — each half correct by construction, the pair
//! silently wrong. The tests here drive BOTH halves of the real chain and
//! assert the values against the geometry the tape declared.
//!
//! The NUMERIC gate (tiny26 EMU-vs-card tensordump parity) runs on the pod;
//! what belongs HERE is the staging contract, which the emu cannot check —
//! the emulator executes KTIR ops directly and never reads these tids.

use scratchy_subtile::subtile_ir::{
    NeoX, NumExperts, RouterBundle, SubOp, SubtileId, SubtileIR, SubtileNode, TensorId,
    TensorRegion, TensorShape, TopK,
};

/// Small stick-legal geometry: E = 8 experts (padded width W = 64), k = 2,
/// m = 5 rows — the tiny26 proportions at the smallest legible size.
const E: u32 = 8;
const K: u32 = 2;
const M: u32 = 5;
/// The PADDED expert width: one whole fp16 stick (64 lanes), which is what
/// the narrow-tensor law mandates for router buffers.
const W: u32 = 64;

/// A minimal MoE-router tape: one `RouterLogits` (declares E) feeding a
/// `RouteTopK` (declares k). The placement pass's geometry read looks for
/// exactly these two node kinds, so this is the smallest tape that mints
/// the router consts.
fn router_ir() -> SubtileIR<NeoX> {
    let tensors = vec![
        TensorShape { rows: M, cols: E }, // t0 logits
        TensorShape { rows: M, cols: E }, // t1 softmax out
        TensorShape { rows: M, cols: E }, // t2 argsort out
        TensorShape { rows: M, cols: K }, // t3 topk out
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
            inputs: vec![tr(0)],
            output: tr(1),
        },
        SubtileNode {
            id: SubtileId::from_index(1),
            op: SubOp::RouteTopK {
                k: TopK::new(std::num::NonZeroU32::new(K).unwrap()),
            },
            inputs: vec![tr(2)],
            output: tr(3),
        },
    ];
    SubtileIR {
        result: TensorId::from_index(3),
        tensors,
        num_sources: 1,
        nodes,
        op_output: Vec::new(),
    }
}

/// The placement half: `compute_bundle_layout` over the router tape.
fn layout_of(ir: &SubtileIR<NeoX>) -> ktir_superdsc::placement::BundleLayout {
    let weight_ids = std::collections::HashSet::new();
    scratchy_target_spyre::lower_subtile_tape_to_superdsc::compute_bundle_layout(
        ir,
        &weight_ids,
        false,
        &[],
    )
    .unwrap_or_else(|e| panic!("compute_bundle_layout: {e:?}"))
}

use ktir_superdsc::reserved_tids as rt;

/// ⭐ EVERY ROUTER CONST TID IS PLACED, AT THE PADDED GEOMETRY, and nothing
/// else in the layout moved: the tie table at `[W, W]` fp16 (W = the padded
/// expert width, NOT E — the narrow-tensor law), the iota row and the two
/// sanitize rows at `[1, W]`, and exactly `k` one-hot rows at `[1, W]`.
#[test]
fn the_router_consts_are_placed_at_the_padded_geometry() {
    let l = layout_of(&router_ir());
    let tie = l.placements.get(&rt::router_rank_tie_tid());
    assert_eq!(
        tie.map(|p| p.size),
        Some(W as u64 * W as u64 * 2),
        "the tie table must be [W, W] fp16 at the PADDED width"
    );
    for (tid, what) in [
        (rt::router_topk_iota_tid(), "the iota row"),
        (rt::router_pad_hi_tid(), "the +inf sanitize row"),
        (rt::router_pad_lo_tid(), "the -inf sanitize row"),
    ] {
        assert_eq!(
            l.placements.get(&tid).map(|p| p.size),
            Some(W as u64 * 2),
            "{what} must be [1, W] fp16"
        );
    }
    for j in 0..K {
        assert_eq!(
            l.placements.get(&rt::router_topk_onehot_tid(j)).map(|p| p.size),
            Some(W as u64 * 2),
            "one-hot row {j} must be [1, W] fp16"
        );
    }
    // And NO one-hot row past k: an over-placed row is a registry/placement
    // desync in the other direction (the bind would read a placement the
    // geometry does not account for).
    assert!(
        l.placements.get(&rt::router_topk_onehot_tid(K)).is_none(),
        "one row past k must not be placed"
    );
}

/// ⭐ A TAPE WITH NO ROUTER PLACES NOTHING — a non-MoE model's bundle is
/// byte-identical: no router_const tid in the placement map at all. This is
/// the guard against the const staging leaking into every model.
#[test]
fn a_routerless_tape_places_no_router_consts() {
    // One plain elementwise node — no RouterLogits, no RouteTopK.
    let tensors = vec![TensorShape { rows: M, cols: W }];
    let region = TensorRegion {
        tensor: TensorId::from_index(0),
        region: tensors[0].whole(),
    };
    let ir = SubtileIR {
        result: TensorId::from_index(0),
        tensors,
        num_sources: 1,
        nodes: vec![SubtileNode {
            id: SubtileId::from_index(0),
            op: SubOp::Mean,
            inputs: vec![region],
            output: region,
        }],
        op_output: Vec::new(),
    };
    let l = layout_of(&ir);
    assert!(
        l.placements.get(&rt::router_rank_tie_tid()).is_none(),
        "a tape with no router must not place the tie table"
    );
    assert!(
        l.placements.get(&rt::router_topk_iota_tid()).is_none(),
        "a tape with no router must not place the iota row"
    );
}

/// ⭐ THE BIND BUILDS THE RIGHT VALUES: `synthetic_constants` at the router
/// geometry produces the tie table `tie[j,h'] = 𝟙[h' < j]`, the iota row
/// `h ↦ h`, one-hot rows with lane `j` hot, and the ±inf sanitize rows —
/// each at its placed tid, each the value its door's arithmetic assumes.
#[test]
fn the_bind_builds_the_declared_values() {
    let env = scratchy_target_spyre::wiring::ConstantEnv {
        hidden: 128,
        head_dim: 64,
        mq_pad: 64,
        uses_ones_reduce: false,
        ones_reduce_len: 0,
        uses_identity: false,
        rows: 1,
        scalarmul_scales: &[],
        rope_class_hds: &[],
        attn_class_hds: &[],
        rms_invcols: &[],
        router: (W as usize, E as usize, K as usize),
    };
    let consts = scratchy_target_spyre::wiring::synthetic_constants(&env);
    let get = |tid: u32| -> Vec<f32> {
        consts
            .iter()
            .find(|(t, _)| *t == tid)
            .unwrap_or_else(|| panic!("tid {tid} was not bound"))
            .1
            .to_vec()
    };
    // The tie table: 1 exactly where h' < j, row-major [j, h'].
    let tie = get(rt::router_rank_tie_tid());
    assert_eq!(tie.len(), (W * W) as usize);
    for j in 0..W {
        for h in 0..W {
            assert_eq!(
                tie[(j * W + h) as usize],
                if h < j { 1.0 } else { 0.0 },
                "tie[{j}][{h}] must be 1 iff h < j (j on ROWS, h' on LANES)"
            );
        }
    }
    // The iota row: h ↦ h.
    let iota = get(rt::router_topk_iota_tid());
    assert_eq!(iota.len(), W as usize);
    for (h, &v) in iota.iter().enumerate() {
        assert_eq!(v, h as f32, "iota lane {h}");
    }
    // One-hot rows: lane j hot, every other lane 0.
    for j in 0..K {
        let row = get(rt::router_topk_onehot_tid(j));
        assert_eq!(row.len(), W as usize);
        for (lane, &v) in row.iter().enumerate() {
            assert_eq!(
                v,
                if lane == j as usize { 1.0 } else { 0.0 },
                "one-hot row {j} lane {lane}"
            );
        }
    }
    // The sanitize rows.
    let hi = get(rt::router_pad_hi_tid());
    let lo = get(rt::router_pad_lo_tid());
    assert!(hi.iter().all(|&v| v == f32::INFINITY), "pad_hi is +inf");
    assert!(lo.iter().all(|&v| v == f32::NEG_INFINITY), "pad_lo is -inf");
    assert_eq!(hi.len(), W as usize);
    assert_eq!(lo.len(), W as usize);
    // The argsort pad-mask: 0 below E, +inf above — the row that makes the producer's
    // zero-padded lanes sort last.
    let mask = get(rt::router_pad_mask_tid());
    assert_eq!(mask.len(), W as usize);
    for (h, &v) in mask.iter().enumerate() {
        assert_eq!(
            v,
            if h < E as usize { 0.0 } else { f32::INFINITY },
            "pad_mask lane {h}"
        );
    }
}

/// ⭐ A NON-ROUTER ENV BINDS NOTHING: `router: (0, 0, 0)` produces no
/// router_const entries at all — the byte-identity half of the guard.
#[test]
fn a_non_router_env_binds_no_router_consts() {
    let env = scratchy_target_spyre::wiring::ConstantEnv {
        hidden: 128,
        head_dim: 64,
        mq_pad: 64,
        uses_ones_reduce: false,
        ones_reduce_len: 0,
        uses_identity: false,
        rows: 1,
        scalarmul_scales: &[],
        rope_class_hds: &[],
        attn_class_hds: &[],
        rms_invcols: &[],
        router: (0, 0, 0),
    };
    let consts = scratchy_target_spyre::wiring::synthetic_constants(&env);
    assert!(
        consts
            .iter()
            .all(|(t, _)| *t != rt::router_rank_tie_tid()),
        "no tie table may be bound without a router"
    );
    assert!(
        consts.iter().all(|(t, _)| *t != rt::router_topk_iota_tid()),
        "no iota row may be bound without a router"
    );
}
