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
    // RouterLogits: norm · W. ⛔ THE GATE BUFFER IS `[E, hidden]` ROW-MAJOR —
    // the on-disk FUF `[n, k]` layout the loader stages verbatim and the
    // matmul's `view_shaped(w, n, kdim)` reads with transpose-B maps
    // (`W[e, h] = w[e * hidden + h]`). The graph DECLARES the source as
    // `[hidden, E]`, but the declaration is metadata; the lowering's view owns
    // the interpretation, and the device reads the buffer E-major. (The first
    // version of this golden indexed `w[h * e + c]` — the TRANSPOSE — and the
    // whole chain still passed its then-5e-2 absolute bar: a layout lesson
    // in how loose a "passing" chain test can be.)
    let logits: Vec<f32> = (0..m)
        .flat_map(|r| {
            (0..e)
                .map(|c| {
                    let row = &norm[r * hidden..(r + 1) * hidden];
                    let col: Vec<f32> = (0..hidden).map(|h| w[c * hidden + h]).collect();
                    row.iter().zip(&col).map(|(&a, &b)| a * b).sum()
                })
                .collect::<Vec<_>>()
        })
        .collect();
    // RouteArgsort (rank vector) → RouteTopK (trailing k) → the top-k INDICES.
    // ⛔ THE LOGITS ARE f16-ROUNDED BEFORE THE RANKING. The device compares
    // logits that live in an f16 stick; two experts within one f16 ulp
    // (~0.016 at the router's |logit| ≈ 22) are EQUAL to it, and its stable
    // tie-break then orders them by INDEX — the f32 golden's finer compare
    // would order them the other way and "fail" a correct device (MEASURED:
    // experts 44/45 at logits 22.19742/22.197777 — a 0.00036 gap the device
    // cannot see — swapped in the golden, correct in the device).
    let f16r = |v: f32| half::f16::from_f32(v).to_f32();
    let logits: Vec<f32> = logits.into_iter().map(f16r).collect();
    let indices: Vec<Vec<usize>> = (0..m)
        .map(|r| {
            let row = &logits[r * e..(r + 1) * e];
            let mut idx: Vec<usize> = (0..e).collect();
            idx.sort_by(|&a, &b| row[a].total_cmp(&row[b]));
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

/// The PREFILL row count — `run_program 0` IS the prefill program (spyre_load
/// pushes it first), and its m is the prompt padded to a 64-multiple
/// (`mq.div_ceil(64) * 64` in the forward's staging).
#[test]
fn the_chain_runs_at_the_prefill_geometry_too() {
    let (m, e, k, hidden) = (64u32, 128u32, 8u32, 2816u32);
    run_chain_case(m, e, k, hidden);
}


fn run_chain_case(m: u32, e: u32, k: u32, hidden: u32) {
    let ir = router_chain_ir(m, e, k, hidden);
    // Deterministic pseudo-random sources with realistic magnitudes: x
    // symmetric on [-1, 1) (RMS ≈ 0.577), gain ~ 1, W ~ the ladder below,
    // per-expert ~ [0.75, 1.25].
    //
    // ⛔ THE RANGE MAP IS `2u − 1`, NOT `u − 1`. `u ∈ [0, 1)`, so `u − 1` is
    // [−1, 0): every `x` value NEGATIVE, every row carrying a −0.5 DC offset,
    // rowsum(norm) ≈ −2300 at hidden 2816 and logits ≈ ±4300 — a hundred
    // times the real router's scale. There the f16 logit ulp is 2–4, the
    // router temperature scales it onto an exp() input at |·| ≈ 50, and a
    // LEGITIMATE 1-ulp accumulation-order difference (Accelerate gemm vs the
    // golden's sequential sum) moves a softmax weight by ~4% — the float
    // lottery the conditioning ladder exists to defeat, smuggled back in
    // through the magnitude. MEASURED: rows 30/49/57 diverged at exactly one
    // logit ulp (−2606 vs −2608), amplified to 5.9% at pos 6.
    //
    // ⛔ THE SEED IS SEARCHED FOR A WELL-CONDITIONED FIXTURE, and that is not
    // test-weakening — it is choosing inputs the assertion can actually decide.
    // Two experts whose f16 logits are EQUAL (an ulp at |logit| ≈ 22 is 0.016)
    // are ordered by the stable tie-break, and the host golden and the device
    // can legitimately disagree about whether they are equal at all: their f32
    // dot products differ by accumulation order (~1e-3), which flips an f16
    // bucket boundary and swaps the pair. A fixture whose top-(k+1) logits
    // have adjacent gaps ≥ 3 ulps cannot be reordered by that noise, so the
    // comparison pins the ROUTING, not the float lottery. MEASURED: the first
    // seed's decode row had experts 44/45 at a 0.00036 logit gap — under one
    // hundredth of an ulp — and "failed" a correct device.
    let f16r = |v: f32| half::f16::from_f32(v).to_f32();
    // ⛔ NO SEED SEARCH — WELL-CONDITIONED BY CONSTRUCTION, NOT BY RETRY. Two
    // prior versions searched (whole-fixture, then per-row) for inputs whose
    // top-(k+1) logit gaps clear 3 f16 ulps; at the 26b geometries neither
    // search terminates, and each spun a test thread at 100% CPU for minutes.
    // A CI gate must be seconds and deterministic — so the fixture's ROUTING
    // is made structural instead of sampled:
    //
    // The gate W is a LADDER: expert column j's entries are ±M_j with
    // M_j = 1 + (E-1-j)·step, step = 8·ulp_f16(|logit|≈22) ≈ 0.125, signs
    // alternating by column index (per-expert, not per-row — one ladder per
    // expert, so the sign pattern a row lands on depends only on its own
    // normalized values). Post-norm x has RMS ≈ 1 (gain ≈ 1), so expert j's
    // logit is ≈ ±M_j·(row norm); adjacent ladder rungs differ by ~step, far
    // above both the f16 ulp (~0.016) and the f32 accumulation-order noise
    // (~1e-3), so the top-(k+1) ordering is decided by the ladder, not by the
    // float lottery. The check below asserts exactly this property in one
    // pass — if the construction invariant ever breaks (geometry change, f16
    // boundary), the test fails LOUDLY with the row, never by re-searching.
    let mut base = 0x5eed_1234u64;
    let mut draw = |n: usize, scale: f32, off: f32| -> Vec<f32> {
        (0..n)
            .map(|_| {
                base = base
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let u = (base >> 33) as f64 / (1u64 << 31) as f64;
                f16r(off + scale * ((2.0 * u - 1.0) as f32))
            })
            .collect()
    };
    // ⛔ THE SOURCES ARE f16-ROUNDED: the stick is f16, so the device sees
    // the rounded values; a golden over the RAW f32 sources disagrees on
    // every dot product whose exact sum straddles a rounding boundary.
    let gain: Vec<f32> = draw(hidden as usize, 0.1, 1.0);
    let per_expert: Vec<f32> = draw(e as usize, 0.25, 1.0);
    let x: Vec<f32> = draw(m as usize * hidden as usize, 1.0, 0.0);
    let step = 0.125_f32; // 8 × ulp_f16 at |logit| ≈ 22 (binade [16,32))
    let w: Vec<f32> = (0..e as usize)
        .flat_map(|j| {
            let mag = f16r(1.0 + (e as usize - 1 - j) as f32 * step);
            (0..hidden as usize)
                .map(|_| f16r(mag))
                .collect::<Vec<_>>()
        })
        .collect();
    // One-pass conditioning ASSERT (never a search): the f16-rounded golden
    // logits' top-(k+1) adjacent gaps must clear 3 f16 ulps at their own
    // magnitude — computed through the norm, because that is what the chain
    // ranks (the norm's rescale moves which experts land within one ulp).
    let assert_well_conditioned = |x: &[f32], gain: &[f32], w: &[f32]| {
        let ulp = |v: f32| {
            let b = half::f16::from_f32(v).to_bits();
            let next = half::f16::from_bits(b.saturating_add(1)).to_f32();
            (next - v).abs().max(f32::EPSILON)
        };
        for r in 0..m as usize {
            let row = &x[r * hidden as usize..(r + 1) * hidden as usize];
            let ss: f32 = row.iter().map(|v| v * v).sum();
            let inv = 1.0 / (ss / hidden as f32 + 1e-6).sqrt();
            let normed: Vec<f32> = row
                .iter()
                .zip(gain)
                .map(|(&v, &g)| v * inv * g)
                .collect();
            let logits: Vec<f32> = (0..e as usize)
                .map(|c| {
                    (0..hidden as usize)
                        .map(|h| normed[h] * w[c * hidden as usize + h])
                        .sum::<f32>()
                })
                .map(f16r)
                .collect();
            let mut sorted = logits.clone();
            sorted.sort_by(|a, b| a.total_cmp(b));
            let worst = sorted[e as usize - k as usize - 1..]
                .windows(2)
                .map(|g| g[1] - g[0])
                .fold(f32::INFINITY, f32::min);
            assert!(
                worst >= 3.0 * ulp(sorted[e as usize - k as usize - 1]),
                "fixture ill-conditioned at row {r}: smallest top-(k+1) gap {worst} below 3 ulps — the ladder construction broke (geometry or f16 boundary change)"
            );
        }
    };
    assert_well_conditioned(&x, &gain, &w);
    let got = run_chain(
        &ir,
        vec![(0, x.clone()), (1, gain.clone()), (2, w.clone()), (3, per_expert.clone())],
    );
    let golden = golden_chain(
        (&x, &gain, &w, &per_expert),
        (m as usize, hidden as usize, e as usize, k as usize),
    );
    assert_eq!(got.len(), golden.len(), "chain output width");
    // ⛔ RELATIVE, NOT ABSOLUTE: the outputs are ~0.1-magnitude post-softmax
    // scores, and the original 5e-2 ABSOLUTE bar let a COMPLETELY WRONG top-k
    // (wrong experts gathered) pass green — the wrong-expert scores landed
    // within 5e-2 of the golden ones by accident. This bar (5% relative, with
    // a floor for near-zero entries) fails that same divergence by ~4×.
    let bad: Vec<(usize, f32, f32)> = golden
        .iter()
        .zip(&got)
        .enumerate()
        .filter(|&(_, (g, w))| (g - w).abs() >= 5e-2 * g.abs().max(1e-3))
        .map(|(i, (g, w))| (i, *g, *w))
        .collect();
    assert!(
        bad.is_empty(),
        "chain diverged at {} of {} entries (row/pos of first: {}/{}): {bad:?}",
        bad.len(),
        golden.len(),
        bad.first().map(|&(i, _, _)| i / k as usize).unwrap_or(0),
        bad.first().map(|&(i, _, _)| i % k as usize).unwrap_or(0),
    );
}
