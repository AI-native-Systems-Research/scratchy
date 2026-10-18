// SPDX-License-Identifier: Apache-2.0
//! THE ROUTER CONST ROWS — the staging half of the MoE router quartet
//! (`RouteArgsort`/`RouteTopK`/`RouteGatherScores`/`RouteExpertScale`).
//!
//! The router doors read five kinds of token-INDEPENDENT factors — the
//! `[W,W]` stable-argsort tie table and the `[1,W]` argsort pad-mask (both
//! BARE-ARGSORT-CORNER only: the splice folds every all-topk argsort into
//! the fused vendor pair, which reads the raw logits and needs no ranks),
//! the `[W,W]` identity table (row j = the one-hot row with lane j hot —
//! ONE table serves the argsort's per-expert lane writes, the bare
//! gather's, and the expert-scale door's per-slot writes), the `[1,W]`
//! iota row, the `[1,W]` softmax k-boundary mask, and the fused top-k's
//! `[1,W]` add-form pad row — and this file pins the STAGING chain that
//! delivers them:
//!
//! ```text
//! compute_bundle_layout  →  places the router_const tids, fold-aware
//! BakeFacts::of          →  reads the geometry (W, E, k) and the two
//!                           corner gates back off the placements
//! ConstantEnv            →  carries them into the bind
//! synthetic_constants    →  builds the VALUES from that geometry
//! ```
//!
//! ⛔ THE LOAD-BEARING INVARIANT IS tid ↔ VALUE AGREEMENT, which no single
//! site can see: the placement pass mints the geometry from the tape's own
//! `RouterLogits`/`RouteTopK`/`RouteArgsort` nodes, and the bind builds the
//! values from the geometry it read back off the PLACEMENTS. A disagreement
//! between the two halves is exactly the class the memory calls "a USAGE
//! CENSUS is not an admission predicate" — each half correct by
//! construction, the pair silently wrong. The tests here drive BOTH halves
//! of the real chain and assert the values against the geometry the tape
//! declared.
//!
//! ⛔ AND THE FOLD IS PART OF THE CONTRACT: the splice silences an argsort
//! whose every consumer is a RouteTopK, so the placement pass places the
//! tie/pad-mask rows ONLY for a BARE argsort (some consumer that is not a
//! RouteTopK) — one question asked of the same tape the fold asked, so the
//! two halves cannot desync into a placed-but-unread table or a
//! read-but-unplaced one.
//!
//! The NUMERIC gate (tiny26 EMU-vs-card tensordump parity) runs on the pod;
//! what belongs HERE is the staging contract, which the emu cannot check —
//! the emulator executes KTIR ops directly and never reads these tids.

use scratchy_subtile::subtile_ir::{
    NeoX, NumExperts, RouterBundle, SubOp, SubtileIR, SubtileId, SubtileNode, TensorId,
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

/// The REAL router chain: one `RouterLogits` (declares E) feeding a
/// `RouteArgsort` whose ONLY consumer is the `RouteTopK` (declares k) — the
/// folded shape every MoE model lowers to, and the smallest tape that mints
/// the fused pair's own consts (the pad row) while folding the argsort's
/// away.
fn router_ir() -> SubtileIR<NeoX> {
    let tensors = vec![
        TensorShape { rows: M, cols: E }, // t0 logits
        TensorShape { rows: M, cols: E }, // t1 softmax out
        TensorShape { rows: M, cols: E }, // t2 rank vector
        TensorShape { rows: M, cols: K }, // t3 topk ids
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
            op: SubOp::RouteArgsort,
            inputs: vec![tr(1)],
            output: tr(2),
        },
        SubtileNode {
            id: SubtileId::from_index(2),
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
        false,
        &[],
        // No composite argmax: the router tape is neither tied nor one-request.
        None,
    )
    .unwrap_or_else(|e| panic!("compute_bundle_layout: {e:?}"))
}

use ktir_superdsc::reserved_tids as rt;

/// The burned slots — the old one-hot top-k's `[k,W]` targets table (3) and
/// its uniform `pad_hi`/`pad_lo` sanitize rows (4, 5), dead the day the
/// fused vendor pair replaced the rank-inversion chain. Named through the
/// region's own `at` so the burn is pinned MECHANICALLY: a reused slot would
/// place (or bind) a tid this file asserts stays absent.
fn retired_slot(idx: u32) -> u32 {
    rt::reserved_region("router_const").at(idx)
}

/// ⭐ THE FOLDED CHAIN PLACES THE FUSED PAIR'S OWN CONSTS, AND NOTHING THE
/// FOLD REMOVED: the identity table at `[W, W]` fp16 (W = the padded expert
/// width, NOT E — the narrow-tensor law), the iota row and the softmax's
/// k-boundary mask at `[1, W]`, and — because this geometry has a REAL pad
/// (`W > E`) — the top-k's add-form pad row at `[1, W]`. The argsort's tie
/// table and pad-mask are NOT placed (the argsort's only consumer is the
/// RouteTopK, so the splice folds it and it emits nothing), and the three
/// retired slots stay absent.
#[test]
fn the_folded_chain_places_the_fused_pairs_consts() {
    let l = layout_of(&router_ir());
    assert_eq!(
        l.placements.get(&rt::router_identity_tid()).map(|p| p.size),
        Some(W as u64 * W as u64 * 2),
        "the identity table must be [W, W] fp16 at the PADDED width"
    );
    for (tid, what) in [
        (rt::router_topk_iota_tid(), "the iota row"),
        (rt::router_sm_mask_tid(), "the softmax k-boundary mask"),
        (rt::router_topk_pad_tid(), "the fused top-k's pad row"),
    ] {
        assert_eq!(
            l.placements.get(&tid).map(|p| p.size),
            Some(W as u64 * 2),
            "{what} must be [1, W] fp16"
        );
    }
    for (tid, what) in [
        (rt::router_rank_tie_tid(), "the tie table"),
        (rt::router_pad_mask_tid(), "the argsort pad-mask row"),
    ] {
        assert!(
            !l.placements.contains_key(&tid),
            "{what} must NOT be placed for a folded argsort (its only consumer is the \
             RouteTopK, so the splice silences it and no door reads the table)"
        );
    }
    for idx in [3u32, 4, 5] {
        assert!(
            !l.placements.contains_key(&retired_slot(idx)),
            "retired router_const slot {idx} must stay unplaced — burned, never reused"
        );
    }
}

/// ⭐ A BARE ARGSORT STILL PLACES ITS CONSTS: the fold silences an argsort
/// only when EVERY consumer of its rank vector is a RouteTopK, so an
/// argsort feeding anything else (here a plain `Mean`) keeps its program —
/// and its `[W,W]` tie table and `[1,W]` maximum-form pad-mask with it.
/// This is the defect the door's own first run surfaced: a k-gated placement
/// left an argsort-only tape with NO consts and the bake panicked on an
/// undeclared tid. The fold-aware gate must not recreate it in the other
/// direction (folding an argsort a door still reads).
#[test]
fn a_bare_argsort_tape_places_the_argsort_consts() {
    let tensors = vec![
        TensorShape { rows: M, cols: E }, // t0 logits
        TensorShape { rows: M, cols: E }, // t1 softmax out
        TensorShape { rows: M, cols: E }, // t2 rank vector
        TensorShape { rows: M, cols: E }, // t3 mean out
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
            op: SubOp::RouteArgsort,
            inputs: vec![tr(1)],
            output: tr(2),
        },
        // The non-RouteTopK consumer that makes the argsort BARE.
        SubtileNode {
            id: SubtileId::from_index(2),
            op: SubOp::Mean,
            inputs: vec![tr(2)],
            output: tr(3),
        },
    ];
    let ir = SubtileIR {
        result: TensorId::from_index(3),
        tensors,
        num_sources: 1,
        nodes,
        op_output: Vec::new(),
    };
    let l = layout_of(&ir);
    assert_eq!(
        l.placements.get(&rt::router_rank_tie_tid()).map(|p| p.size),
        Some(W as u64 * W as u64 * 2),
        "a bare argsort (ranks feeding a non-topk consumer) places the tie table"
    );
    assert_eq!(
        l.placements.get(&rt::router_pad_mask_tid()).map(|p| p.size),
        Some(W as u64 * 2),
        "a bare argsort places the maximum-form pad-mask row"
    );
    assert_eq!(
        l.placements.get(&rt::router_identity_tid()).map(|p| p.size),
        Some(W as u64 * W as u64 * 2),
        "every router tape places the identity table"
    );
    // And the pair's own rows are NOT placed: this tape has no RouteTopK, so
    // there is no k for the softmax mask and no pad row to sanitize.
    for (tid, what) in [
        (rt::router_sm_mask_tid(), "the softmax k-boundary mask"),
        (rt::router_topk_pad_tid(), "the fused top-k's pad row"),
    ] {
        assert!(
            !l.placements.contains_key(&tid),
            "{what} must not be placed without a RouteTopK node"
        );
    }
}

/// ⭐ A REAL-WIDTH ROUTER PLACES NO PAD ROW: at every real router the logits
/// are written at full stick width (`W == E`, E a whole stick), so the
/// producer's pad lanes do not exist and the add-form sanitize is inert —
/// the row is placed only when `W > E`, which keeps every real model's wire
/// byte-identical. This pins that gate from the placement half.
#[test]
fn a_real_width_router_places_no_pad_row() {
    let e = W; // E = 64: a whole stick, so W == E and the pad is empty.
    let tensors = vec![
        TensorShape { rows: M, cols: e }, // t0 logits
        TensorShape { rows: M, cols: e }, // t1 softmax out
        TensorShape { rows: M, cols: e }, // t2 rank vector
        TensorShape { rows: M, cols: K }, // t3 topk ids
    ];
    let tr = |t: usize| TensorRegion {
        tensor: TensorId::from_index(t),
        region: tensors[t].whole(),
    };
    let nodes = vec![
        SubtileNode {
            id: SubtileId::from_index(0),
            op: SubOp::RouterLogits {
                experts: NumExperts::new(std::num::NonZeroU32::new(e).unwrap()),
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
            inputs: vec![tr(1)],
            output: tr(3),
        },
    ];
    let ir = SubtileIR {
        result: TensorId::from_index(3),
        tensors,
        num_sources: 1,
        nodes,
        op_output: Vec::new(),
    };
    let l = layout_of(&ir);
    assert!(
        !l.placements.contains_key(&rt::router_topk_pad_tid()),
        "W == E means no pad lanes exist — the pad row must not be placed"
    );
    assert_eq!(
        l.placements.get(&rt::router_identity_tid()).map(|p| p.size),
        Some(W as u64 * W as u64 * 2),
        "the identity table is placed at the same [W, W] either way"
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
    for (tid, what) in [
        (rt::router_rank_tie_tid(), "the tie table"),
        (rt::router_identity_tid(), "the identity table"),
        (rt::router_topk_iota_tid(), "the iota row"),
        (rt::router_pad_mask_tid(), "the argsort pad-mask"),
        (rt::router_sm_mask_tid(), "the softmax mask"),
        (rt::router_topk_pad_tid(), "the fused top-k's pad row"),
    ] {
        assert!(
            !l.placements.contains_key(&tid),
            "a tape with no router must not place {what}"
        );
    }
}

/// ⭐ THE BIND BUILDS THE RIGHT VALUES: `synthetic_constants` at the router
/// geometry produces the tie table `tie[j,h'] = 𝟙[h' < j]`, the iota row
/// `h ↦ h`, one-hot rows with lane `j` hot, the softmax's 0/−inf
/// k-boundary, the argsort's −inf/+inf maximum-form mask, and the fused
/// top-k's 0/−inf ADD-form pad row — each at its placed tid, each the value
/// its door's arithmetic assumes.
#[test]
fn the_bind_builds_the_declared_values() {
    let env = scratchy_target_spyre::wiring::ConstantEnv {
        hidden: 128,
        head_dim: 64,
        mq_pad: 64,
        uses_ones_reduce: false,
        ones_reduce_len: 0,
        uses_identity: false,
        rope_class_hds: &[],
        attn_class_hds: &[],
        rms_invcols: &[],
        rows: 1,
        scalarmul_scales: &[],
        router: (W as usize, E as usize),
        router_k: K as usize,
        router_argsort: true,
        router_topk_pad: true,
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
    // The SOFTMAX sanitize row: 0 at the top-k landing's REPLICA-RUN HEADS —
    // lane r·(64/k) for r < k, the one lane per result the chain keeps (`x +
    // 0` bit-identical there) — and −inf everywhere else (every other replica
    // and every pad lane: `exp(−inf)=0`, so only the heads contribute to the
    // denominator). The old content (0 at lanes [0, k)) kept k replicas of
    // result 0 and made every softmax row uniform — card-measured at the 26b.
    let sm_mask = get(rt::router_sm_mask_tid());
    assert_eq!(sm_mask.len(), W as usize);
    let rep = 64 / K;
    for (h, &v) in sm_mask.iter().enumerate() {
        assert_eq!(
            v,
            if h < 64 && h % rep as usize == 0 {
                0.0
            } else {
                f32::NEG_INFINITY
            },
            "sm_mask lane {h} (head lanes at r·{rep})"
        );
    }
    // The identity table: row j is the one-hot row with lane j hot — ONE
    // [W,W] table, E-sized (row per ranked expert), never k-sized.
    let ident = get(rt::router_identity_tid());
    assert_eq!(ident.len(), (W * W) as usize);
    for j in 0..W {
        for (lane, &v) in ident[(j * W) as usize..((j + 1) * W) as usize]
            .iter()
            .enumerate()
        {
            assert_eq!(
                v,
                if lane == j as usize { 1.0 } else { 0.0 },
                "identity row {j} lane {lane}"
            );
        }
    }
    // The argsort pad-mask: −inf below E (the real experts — `maximum(v, −inf)
    // = v` for every non-NaN v, NEGATIVES INCLUDED), +inf above (the producer's
    // zero-padded lanes sort last). ⛔ The real-lane value MUST be −inf, never
    // 0: `maximum(v, 0)` clamps every negative router logit to exactly 0 and
    // collapses the ranks into tie-runs (measured on-card, the 26b wall-11
    // predecessor) — this assertion pins that law.
    let mask = get(rt::router_pad_mask_tid());
    assert_eq!(mask.len(), W as usize);
    for (h, &v) in mask.iter().enumerate() {
        assert_eq!(
            v,
            if h < E as usize {
                f32::NEG_INFINITY
            } else {
                f32::INFINITY
            },
            "pad_mask lane {h}"
        );
    }
    // The fused top-k's ADD-form pad row: 0 below E (the real experts —
    // `x + 0 = x` bit-identical, negatives included), −inf above (the
    // producer's zero-padded lanes: the pair's scan runs to W, and a zero
    // pad lane would OUTRANK every negative router logit and steal a slot,
    // emitting the out-of-range id whose expert-page fetch faults the card —
    // wall-11's own root). ADD-form, not the argsort's maximum-form:
    // `maximum` can only RAISE, so no maximum row can force a zero pad lane
    // DOWN to −inf.
    let pad = get(rt::router_topk_pad_tid());
    assert_eq!(pad.len(), W as usize);
    for (h, &v) in pad.iter().enumerate() {
        assert_eq!(
            v,
            if h < E as usize {
                0.0
            } else {
                f32::NEG_INFINITY
            },
            "topk_pad lane {h}"
        );
    }
    // The retired slots bind nothing — burned, never reused.
    for idx in [3u32, 4, 5] {
        assert!(
            consts.iter().all(|(t, _)| *t != retired_slot(idx)),
            "retired router_const slot {idx} must stay unbound"
        );
    }
}

/// ⭐ THE TWO CORNER ROWS GATE ON THEIR OWN FACTS, NOT THE GEOMETRY: with
/// the same `(W, E, k)` but no bare argsort and no `W > E` pad, the tie
/// table, the argsort pad-mask, and the top-k pad row bind NOTHING — while
/// the geometry-derived rows (identity, iota, softmax mask) still do. The
/// bind's gates are the placement's own questions read back off the
/// artifact, so a bundle that placed nothing for a corner binds nothing
/// for it.
#[test]
fn the_corner_rows_gate_on_their_own_facts() {
    let env = scratchy_target_spyre::wiring::ConstantEnv {
        hidden: 128,
        head_dim: 64,
        mq_pad: 64,
        uses_ones_reduce: false,
        ones_reduce_len: 0,
        uses_identity: false,
        rope_class_hds: &[],
        attn_class_hds: &[],
        rms_invcols: &[],
        rows: 1,
        scalarmul_scales: &[],
        router: (W as usize, E as usize),
        router_k: K as usize,
        router_argsort: false,
        router_topk_pad: false,
    };
    let consts = scratchy_target_spyre::wiring::synthetic_constants(&env);
    let bound = |tid: u32| consts.iter().any(|(t, _)| *t == tid);
    for (tid, what) in [
        (rt::router_rank_tie_tid(), "the tie table"),
        (rt::router_pad_mask_tid(), "the argsort pad-mask"),
        (rt::router_topk_pad_tid(), "the fused top-k's pad row"),
    ] {
        assert!(
            !bound(tid),
            "{what} must not bind without its placement fact (bare argsort / W > E)"
        );
    }
    for (tid, what) in [
        (rt::router_identity_tid(), "the identity table"),
        (rt::router_topk_iota_tid(), "the iota row"),
        (rt::router_sm_mask_tid(), "the softmax mask"),
    ] {
        assert!(bound(tid), "{what} binds from the geometry alone");
    }
}

/// ⭐ A NON-ROUTER ENV BINDS NOTHING: `router: (0, 0)` produces no
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
        rope_class_hds: &[],
        attn_class_hds: &[],
        rms_invcols: &[],
        rows: 1,
        scalarmul_scales: &[],
        router: (0, 0),
        router_k: 0,
        router_argsort: false,
        router_topk_pad: false,
    };
    let consts = scratchy_target_spyre::wiring::synthetic_constants(&env);
    for (tid, what) in [
        (rt::router_rank_tie_tid(), "the tie table"),
        (rt::router_identity_tid(), "the identity table"),
        (rt::router_topk_iota_tid(), "the iota row"),
        (rt::router_pad_mask_tid(), "the argsort pad-mask"),
        (rt::router_sm_mask_tid(), "the softmax mask"),
        (rt::router_topk_pad_tid(), "the fused top-k's pad row"),
    ] {
        assert!(
            consts.iter().all(|(t, _)| *t != tid),
            "no {what} may be bound without a router"
        );
    }
}
