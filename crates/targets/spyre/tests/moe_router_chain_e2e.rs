// SPDX-License-Identifier: Apache-2.0
// ⛔ RUN THIS FILE WITH `SCRATCHY_PLAN_ONLY_BAKE=1` — the door side of these gates
// runs the KTIR→SuperDSC lowering, whose fixtures must not demand a device compiler.
//
//     SCRATCHY_PLAN_ONLY_BAKE=1 cargo test -p scratchy-target-spyre \
//         --features spyre-emu --test moe_router_chain_e2e
//
// (See `superdsc_time_tile.rs`'s header for why the variable is not set from
// inside the test.)
#![cfg(feature = "spyre-emu")]
//! THE FULL gemma-4 ROUTER CHAIN, MULTI-NODE, THROUGH THE REAL SERVING PATH.
//!
//! `moe_router_chain.rs` proves each router op's SINGLE-node program against a
//! hand golden. The 26b E2E run dies INSIDE a FUSED segment of the whole
//! program (`seg10 fused @fused: KtdpLoad: indirect index -154 from index_view
//! 0 is negative`), which none of those tests can see: the failure needs the
//! chain's nodes fused into ONE segment, the way `SpyreSession::new_multi` →
//! `plan_segments_budgeted` runs them in serving. This file builds exactly the
//! gemma-4 wavefront's router chain (`RouterNorm → RouterLogits →
//! RouteArgsort → RouteTopK → RouteGatherScores → RouteScale → RouteSoftmax →
//! RouteExpertScale`) as one SubtileIR and drives it through the same
//! `lower_graph_to_ktir` → `ktir_groups` → `SpyreSession::new_multi` →
//! `run_step` path, at gemma-4's own geometry (E=128 experts, k=8).
//!
//! The golden is computed HERE on the host, stating metal's semantics op by op
//! (the same goldens `moe_router_chain.rs` states per-op: the rank-vector
//! argsort with NaN-as-greater, the trailing-k slice, take_along_axis, the
//! hidden^-0.5 temperature, the row softmax, the per-expert scale multiply).

use scratchy_subtile::subtile_ir::{
    NeoX, NumExperts, RouterBundle, SubOp, SubtileId, SubtileIR, SubtileNode, TensorId,
    TensorRegion, TensorShape, TopK,
};
use scratchy_target_spyre::lower_subtile_tape_to_superdsc as superdsc;

/// The gemma-4 router chain as one graph, at the 26b's own geometry.
///
/// `t0` = x `[m, hidden]` (source), `t1` = router gain `[1, hidden]` (source),
/// `t2` = router gate `[hidden, E]` (source), `t3` = per-expert scale `[1, E]`
/// (source), then one tensor per chain node. The chain ends at
/// RouteExpertScale's `[m, k]` output — the tensor the expert half consumes.
fn router_chain_ir(m: u32, e: u32, k: u32, hidden: u32) -> SubtileIR<NeoX> {
    let nz = |n| std::num::NonZeroU32::new(n).unwrap();
    let mut tensors = vec![
        TensorShape { rows: m, cols: hidden }, // t0 x (source)
        TensorShape { rows: 1, cols: hidden }, // t1 router gain (source)
        TensorShape { rows: hidden, cols: e }, // t2 router gate W (source)
        TensorShape { rows: 1, cols: e },     // t3 per-expert scale (source)
    ];
    let num_sources = tensors.len() as u32;
    let mut nodes = Vec::new();
    let mut next = tensors.len();
    let mut push = |op: SubOp, inputs: Vec<usize>, tensors: &mut Vec<TensorShape>| -> usize {
        // The shape table in ops.rs: output rows = in0's rows; cols per op.
        let (rows, cols) = match op {
            SubOp::RouterNorm { .. } => (m, hidden),
            SubOp::RouterLogits { .. } | SubOp::RouteArgsort => (m, e),
            SubOp::RouteTopK { k: kk } => (m, kk.get()),
            SubOp::RouteGatherScores
            | SubOp::RouteScale { .. }
            | SubOp::RouteSoftmax
            | SubOp::RouteRenorm
            | SubOp::RouteExpertScale { .. } => (m, k),
            _ => unreachable!("router chain op"),
        };
        let out = next;
        tensors.push(TensorShape { rows, cols });
        next += 1;
        let tr = |t: usize| TensorRegion {
            tensor: TensorId::from_index(t),
            region: tensors[t].whole(),
        };
        nodes.push(SubtileNode {
            id: SubtileId::from_index(nodes.len()),
            op,
            inputs: inputs.iter().map(|&t| tr(t)).collect(),
            output: tr(out),
        });
        out
    };
    let gemma = RouterBundle::Gemma;
    let xr = push(
        SubOp::RouterNorm { eps: 1e-6, router: gemma },
        vec![0, 1],
        &mut tensors,
    );
    let lg = push(
        SubOp::RouterLogits { experts: NumExperts::new(nz(e)), router: gemma },
        vec![xr, 2],
        &mut tensors,
    );
    let sorted = push(SubOp::RouteArgsort, vec![lg], &mut tensors);
    let indices = push(SubOp::RouteTopK { k: TopK::new(nz(k)) }, vec![sorted], &mut tensors);
    let scores = push(SubOp::RouteGatherScores, vec![lg, indices], &mut tensors);
    let scaled = push(SubOp::RouteScale { scale: (hidden as f32).powf(-0.5) }, vec![scores], &mut tensors);
    let soft = push(SubOp::RouteSoftmax, vec![scaled], &mut tensors);
    let fin = push(
        SubOp::RouteExpertScale { router: gemma },
        vec![soft, indices, 3],
        &mut tensors,
    );
    SubtileIR {
        result: TensorId::from_index(fin),
        tensors,
        num_sources,
        nodes,
        op_output: Vec::new(),
    }
}

/// The host golden for the whole chain, stating metal's op semantics exactly.
fn golden_chain(
    src: (&[f32], &[f32], &[f32], &[f32]),
    geom: (usize, usize, usize, usize),
) -> Vec<f32> {
    let (x, gain, w, per_expert) = src;
    let (m, hidden, e, k) = geom;
    // RouterNorm: rmsnorm(x, gain), eps 1e-6 — gemma's scale convention (no +1).
    let norm: Vec<f32> = (0..m)
        .flat_map(|r| {
            let row = &x[r * hidden..(r + 1) * hidden];
            let ss: f32 = row.iter().map(|v| v * v).sum();
            let inv = 1.0 / (ss / hidden as f32 + 1e-6).sqrt();
            row.iter().zip(gain).map(|(&v, &g)| v * inv * g).collect::<Vec<_>>()
        })
        .collect();
    // RouterLogits: norm · W (`[m, hidden] · [hidden, E]`).
    let logits: Vec<f32> = (0..m)
        .flat_map(|r| {
            (0..e)
                .map(|c| {
                    let row = &norm[r * hidden..(r + 1) * hidden];
                    let col: Vec<f32> = (0..hidden).map(|h| w[h * e + c]).collect();
                    row.iter().zip(&col).map(|(&a, &b)| a * b).sum()
                })
                .collect::<Vec<_>>()
        })
        .collect();
    // RouteArgsort (rank vector) → RouteTopK (trailing k) → the top-k INDICES.
    let indices: Vec<Vec<usize>> = (0..m)
        .map(|r| {
            let row = &logits[r * e..(r + 1) * e];
            let mut idx: Vec<usize> = (0..e).collect();
            idx.sort_by(|&a, &b| {
                row[a]
                    .partial_cmp(&row[b])
                    .unwrap_or(std::cmp::Ordering::Greater)
            });
            idx[e - k..].to_vec()
        })
        .collect();
    // GatherScores at the indices, then the hidden^-0.5 temperature.
    let temp = (hidden as f32).powf(-0.5);
    let scaled: Vec<f32> = (0..m)
        .flat_map(|r| {
            indices[r]
                .iter()
                .map(|&j| logits[r * e + j] * temp)
                .collect::<Vec<_>>()
        })
        .collect();
    // Row softmax over the k scaled scores.
    let soft: Vec<f32> = (0..m)
        .flat_map(|r| {
            let row = &scaled[r * k..(r + 1) * k];
            let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = row.iter().map(|v| (v - mx).exp()).collect();
            let s: f32 = exps.iter().sum();
            exps.iter().map(|v| v / s).collect::<Vec<_>>()
        })
        .collect();
    // × per-expert scale at the indices.
    (0..m)
        .flat_map(|r| {
            indices[r]
                .iter()
                .enumerate()
                .map(|(j, &ei)| soft[r * k + j] * per_expert[ei])
                .collect::<Vec<_>>()
        })
        .collect()
}

/// Drive the chain through the real serving path and return the final tensor.
fn run_chain(ir: &SubtileIR<NeoX>, sources: Vec<(u64, Vec<f32>)>) -> Vec<f32> {
    let weight_ids = std::collections::HashSet::new();
    let (ops, _layout) = superdsc::lower_graph_to_ktir(
        ir,
        &weight_ids,
        superdsc::ActiveCap::FULL,
        false,
    )
    .unwrap_or_else(|e| panic!("lower the router chain: {e:?}"));
    let out_tid = ir.result.index() as u64;
    let groups = superdsc::ktir_groups(&ops, superdsc::FoldGrouping::Split)
        .expect("bind the router chain's programs");
    let mut session = scratchy_target_spyre::runner::SpyreSession::new_multi(
        &[(&groups, &[out_tid])],
        Vec::new(),
    )
    .expect("build the router-chain session");
    let out = session
        .run_step(
            0,
            sources
                .into_iter()
                .map(|(id, data)| {
                    (
                        id,
                        data,
                        vec![
                            ir.tensors[id as usize].rows as usize,
                            ir.tensors[id as usize].cols as usize,
                        ],
                    )
                })
                .collect(),
            &[(out_tid, 0)],
        )
        .expect("run the router chain");
    out[&out_tid].clone()
}

/// The E2E geometry (a small hidden; E and k are gemma-4's own 128/8) with
/// LOGIT-SCALE values — the real router's regime, where a misread index shows
/// up as a large negative integer rather than an in-range one.
#[test]
fn the_whole_router_chain_runs_fused_and_matches_the_host_golden() {
    let (m, e, k, hidden) = (2u32, 128u32, 8u32, 64u32);
    run_chain_case(m, e, k, hidden);
}

/// The DECODE row count at the model's own hidden — the 26b's first-step
/// regime, where the E2E run died (`seg10 fused @fused: KtdpLoad: indirect
/// index -154 ... is negative`).
#[test]
fn the_chain_runs_at_the_decode_geometry_too() {
    let (m, e, k, hidden) = (1u32, 128u32, 8u32, 2816u32);
    run_chain_case(m, e, k, hidden);
}

fn run_chain_case(m: u32, e: u32, k: u32, hidden: u32) {
    let ir = router_chain_ir(m, e, k, hidden);
    // Deterministic pseudo-random sources with realistic magnitudes: x ~ N(0,1),
    // gain ~ 1, W ~ N(0, 1/sqrt(hidden)), per-expert ~ N(1, 0.25).
    let mut seed = 0x5eed_1234u64;
    let mut rnd = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((seed >> 33) as f64 / (1u64 << 31) as f64) - 1.0
    };
    let x: Vec<f32> = (0..m * hidden).map(|_| rnd() as f32).collect();
    let gain: Vec<f32> = (0..hidden).map(|_| 1.0 + 0.1 * rnd() as f32).collect();
    let w: Vec<f32> = (0..hidden * e)
        .map(|_| (rnd() as f32) / (hidden as f32).sqrt())
        .collect();
    let per_expert: Vec<f32> = (0..e).map(|_| 1.0 + 0.25 * rnd() as f32).collect();
    let got = run_chain(
        &ir,
        vec![(0, x.clone()), (1, gain.clone()), (2, w.clone()), (3, per_expert.clone())],
    );
    let golden = golden_chain(
        (&x, &gain, &w, &per_expert),
        (m as usize, hidden as usize, e as usize, k as usize),
    );
    assert_eq!(got.len(), golden.len(), "chain output width");
    for (i, (g, w)) in golden.iter().zip(&got).enumerate() {
        assert!(
            (g - w).abs() < 5e-2,
            "chain[{i}] golden={g} got={w}"
        );
    }
}
