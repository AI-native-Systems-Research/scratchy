// SPDX-License-Identifier: Apache-2.0
// ⛔ RUN THIS FILE WITH `SCRATCHY_PLAN_ONLY_BAKE=1` — the door side of these gates
// runs the KTIR→SuperDSC lowering, whose fixtures must not demand a device compiler.
//
//     SCRATCHY_PLAN_ONLY_BAKE=1 cargo test -p scratchy-target-spyre \
//         --features spyre-emu --test moe_expert_chain_e2e
//
// (See `superdsc_time_tile.rs`'s header for why the variable is not set from
// inside the test.)
#![cfg(feature = "spyre-emu")]
//! THE WHOLE gemma-4 EXPERT HALF, FUSED, THROUGH THE REAL SERVING PATH.
//!
//! `moe_router_chain.rs` proves each expert op's SINGLE-node program against a
//! hand golden (sort, unsort, combine, the gathered fp8 matmul unblocked and
//! blocked). What none of those can see is the CHAIN — `ExpertSort →
//! ExpertMatmul(gate) → ExpertMatmul(up) → ExpertGatedAct(GeGLU) →
//! ExpertMatmul(down) → ExpertUnsort → ExpertCombine` as ONE SubtileIR through
//! the same `lower_graph_to_ktir` → `ktir_groups` → `SpyreSession::new_multi` →
//! `run_step` path serving runs, where the FUSED-SEGMENT PLANNER (not the
//! per-node oracle) owns the op order, the LX liveness, and the GPU offloads.
//! `ExpertGatedAct` — the GeGLU `gelu_tanh(gate) · up` — has no test at all;
//! this file is its first.
//!
//! The banks are staged exactly as the loader stages them (fp8 codes 1-byte
//! verbatim, scales bf16→f16), the indices/scores arrive as f16 tiles the way
//! `RouteTopK`/`RouteExpertScale` emit them, and the golden states the
//! emulator's op-by-op numerics: the f32 gathered code-dot, ONE f16 round at
//! the scale mulf, the f16 gelu-tanh polynomial (clamped ±15, each op rounded
//! to f16), the f16 `act·up`, the f32 down-projection code-dot, and the
//! combine's f32 fma chain with one f16 narrowing.

use scratchy_subtile::subtile_ir::{
    ExpertBundle, ExpertProj, GatedAct, NeoX, NumExperts, SubOp, SubtileId, SubtileIR,
    SubtileNode, TensorId, TensorRegion, TensorShape, TopK,
};
use scratchy_target_spyre::lower_subtile_tape_to_superdsc as superdsc;

/// The gemma-4 expert half as one graph, at a small geometry of the 26b's own
/// proportions (stacked banks, k slots, GeGLU).
///
/// Sources: `t0` = x `[m, hidden]`, `t1` = indices `[m, k]`, `t2` = scores
/// `[m, k]`, then each projection's codes `[E·out, in]` and scales `[E·out, 1]`.
fn expert_chain_ir(m: u32, e: u32, k: u32, hidden: u32, inter: u32) -> SubtileIR<NeoX> {
    let mut tensors = vec![
        TensorShape { rows: m, cols: hidden }, // t0 x (source)
        TensorShape { rows: m, cols: k },     // t1 indices (source)
        TensorShape { rows: m, cols: k },     // t2 scores (source)
        TensorShape { rows: e * inter, cols: hidden }, // t3 gate codes (source)
        TensorShape { rows: e * inter, cols: 1 },     // t4 gate scales (source)
        TensorShape { rows: e * inter, cols: hidden }, // t5 up codes (source)
        TensorShape { rows: e * inter, cols: 1 },     // t6 up scales (source)
        TensorShape { rows: e * hidden, cols: inter }, // t7 down codes (source)
        TensorShape { rows: e * hidden, cols: 1 },     // t8 down scales (source)
    ];
    let num_sources = tensors.len() as u32;
    let mut nodes = Vec::new();
    let mut next = tensors.len();
    let nz = |n: u32| std::num::NonZeroU32::new(n).unwrap();
    let experts = NumExperts::new(nz(e));
    let topk = TopK::new(nz(k));
    let mut push = |op: SubOp, inputs: Vec<usize>, tensors: &mut Vec<TensorShape>| -> usize {
        let (rows, cols) = match &op {
            SubOp::ExpertSort { .. } => (m, hidden * k),
            SubOp::ExpertMatmul { proj, .. } => {
                let w = match proj {
                    ExpertProj::Gate | ExpertProj::Up => inter,
                    ExpertProj::Down => hidden,
                };
                (m, w * k)
            }
            SubOp::ExpertGatedAct { .. } => (m, inter * k),
            SubOp::ExpertUnsort => (m, hidden * k),
            SubOp::ExpertCombine { .. } => (m, hidden),
            _ => unreachable!("expert chain op"),
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
    let bundle = ExpertBundle::SwitchGlu;
    let quant = scratchy_subtile::lower::ExpertQuant::declared(64, 8);
    let pairs = push(
        SubOp::ExpertSort { experts, k: topk, bundle },
        vec![0, 1],
        &mut tensors,
    );
    let gate = push(
        SubOp::ExpertMatmul { proj: ExpertProj::Gate, n: inter, k: topk, quant, bundle },
        vec![pairs, pairs, 3, 4],
        &mut tensors,
    );
    let up = push(
        SubOp::ExpertMatmul { proj: ExpertProj::Up, n: inter, k: topk, quant, bundle },
        vec![pairs, pairs, 5, 6],
        &mut tensors,
    );
    let act = push(SubOp::ExpertGatedAct { act: GatedAct::Gelu }, vec![gate, up], &mut tensors);
    let down = push(
        SubOp::ExpertMatmul { proj: ExpertProj::Down, n: hidden, k: topk, quant, bundle },
        vec![act, pairs, 7, 8],
        &mut tensors,
    );
    let tokens = push(SubOp::ExpertUnsort, vec![down, pairs], &mut tensors);
    let out = push(
        SubOp::ExpertCombine {
            hidden,
            shared: scratchy_subtile::subtile_ir::SharedExpertBound(None),
        },
        vec![tokens, 2],
        &mut tensors,
    );
    SubtileIR {
        result: TensorId::from_index(out),
        tensors,
        num_sources,
        nodes,
        op_output: Vec::new(),
    }
}

/// f16 round-trip through the emulator's storage: the value an f16 element
/// holds after `Tile::compute`'s single rounding.
fn f16(v: f32) -> f32 {
    let bits = ktir_emulator::codec::encode(&[v], ktir_emulator::dtypes::DType::F16);
    ktir_emulator::codec::decode(&bits, 1, ktir_emulator::dtypes::DType::F16)[0]
}

/// The device's gelu-tanh polynomial, op by op in f16 — `KtirFunc::gelu`'s
/// exact chain (`binop`/`unop` round each result to the tile's f16).
fn gelu_f16(x: f32) -> f32 {
    let k = 0.797_884_56f32; // sqrt(2/pi), f16-splatted
    let c = 0.044_715f32;
    let x2 = f16(x * x);
    let x3 = f16(x2 * x);
    let cx3 = f16(c * x3);
    let inner0 = f16(x + cx3);
    let inner = f16(k * inner0);
    let clamped = inner.clamp(-15.0, 15.0);
    let t = f16(clamped.tanh());
    let half = 0.5f32;
    let hx = f16(half * x);
    let ht = f16(hx * t);
    f16(hx + ht)
}

/// Drive the chain through the real serving path and return the final tensor.
fn run_chain(
    ir: &SubtileIR<NeoX>,
    sources: Vec<(u64, Vec<f32>)>,
    weights: Vec<(usize, Vec<u8>, scratchy_tensors::DType, Vec<usize>)>,
) -> Vec<f32> {
    let weight_ids = std::collections::HashSet::new();
    let (ops, _layout) = superdsc::lower_graph_to_ktir(
        ir,
        &weight_ids,
        superdsc::ActiveCap::FULL,
        false,
    )
    .unwrap_or_else(|e| panic!("lower the expert chain: {e:?}"));
    let out_tid = ir.result.index() as u64;
    let groups = superdsc::ktir_groups(&ops, superdsc::FoldGrouping::Split)
        .expect("bind the expert chain's programs");
    let mut session = scratchy_target_spyre::runner::SpyreSession::new_multi(
        &[(&groups, &[out_tid])],
        weights,
    )
    .expect("build the expert-chain session");
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
        .expect("run the expert chain");
    out[&out_tid].clone()
}

/// Build one projection's bank: deterministic codes (a safe e4m3 range) and
/// per-channel scales in [0.75, 1.25], staged the way the loader stages them.
fn bank(e: u32, out: u32, inp: u32, salt: u64) -> (Vec<u8>, Vec<u8>, Vec<f32>) {
    let rows = (e * out) as usize;
    let cols = inp as usize;
    let mut base = salt;
    let mut draw = || {
        base = base
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (base >> 33) as f64 / (1u64 << 31) as f64
    };
    let codes: Vec<f32> = (0..rows * cols)
        .map(|_| {
            let u = draw();
            // Small codes keep every INTERMEDIATE inside f16: the gate/up dots
            // are ~±5, the GeGLU act ~±25, the down dot ~±600 — the real
            // model's regime (an act value of ~500 would make the down dot
            // ~2e5 = f16 inf, and the comparison would pin an overflow, not
            // the routing).
            ((2.0 * u - 1.0) as f32) * 1.5
        })
        .collect();
    let scales: Vec<f32> = (0..rows)
        .map(|_| {
            let u = draw();
            (0.75 + 0.5 * u) as f32
        })
        .collect();
    let w_bytes: Vec<u8> = codes.iter().map(|&v| ktir_core::codec::f32_to_e4m3(v)).collect();
    let s_bytes = ktir_emulator::codec::encode(&scales, ktir_emulator::dtypes::DType::F16);
    (w_bytes, s_bytes, scales)
}

fn decode(b: u8) -> f32 {
    ktir_core::codec::e4m3_to_f32(b)
}

/// One (geometry, fixtures, golden, compare) case at the given row count.
///
/// ⛔ NO SEED SEARCH — WELL-CONDITIONED BY CONSTRUCTION. The codes are bounded
/// (|code| ≤ 24), the scales ≈ 1, and the dots are over ≤ 64-deep rows, so the
/// f32 code-dots stay ~1e2 — two f16 ulps at that magnitude are ~0.06, far
/// under the 5% bar. A mis-routed pair reads a DIFFERENT EXPERT'S slab, which
/// moves the output by O(1) — 20× the bar.
fn run_case(m: u32) {
    let (e, k, hidden, inter) = (4u32, 2u32, 32u32, 16u32);
    let ir = expert_chain_ir(m, e, k, hidden, inter);
    // x symmetric on [-1, 1); indices in range; scores a softmax-like simplex.
    let mut base = 0x5eed_beefu64;
    let mut draw = || {
        base = base
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (base >> 33) as f64 / (1u64 << 31) as f64
    };
    let x: Vec<f32> = (0..m * hidden)
        .map(|_| ((2.0 * draw() - 1.0) as f32))
        .collect();
    // DISTINCT experts per (row, slot) — a wrong slab is visible everywhere.
    let idx: Vec<f32> = (0..m)
        .flat_map(|r| {
            let e0 = ((r as usize + 1) % e as usize) as f32;
            let e1 = ((r as usize + 3) % e as usize) as f32;
            vec![e0, e1]
        })
        .collect();
    let scores: Vec<f32> = (0..m)
        .flat_map(|_| vec![0.6f32, 0.4f32])
        .collect();
    let (gw, gs, gs_v) = bank(e, inter, hidden, 0x1234_5678);
    let (uw, us, us_v) = bank(e, inter, hidden, 0x9abc_def0);
    let (dw, ds, ds_v) = bank(e, hidden, inter, 0x1357_9bdf);
    let weights = vec![
        (3, gw.clone(), scratchy_tensors::DType::Fp8E4m3, vec![(e * inter) as usize, hidden as usize]),
        (4, gs.clone(), scratchy_tensors::DType::F16, vec![(e * inter) as usize]),
        (5, uw.clone(), scratchy_tensors::DType::Fp8E4m3, vec![(e * inter) as usize, hidden as usize]),
        (6, us.clone(), scratchy_tensors::DType::F16, vec![(e * inter) as usize]),
        (7, dw.clone(), scratchy_tensors::DType::Fp8E4m3, vec![(e * hidden) as usize, inter as usize]),
        (8, ds.clone(), scratchy_tensors::DType::F16, vec![(e * hidden) as usize]),
    ];
    let got = run_chain(
        &ir,
        vec![(0, x.clone()), (1, idx.clone()), (2, scores.clone())],
        weights,
    );

    // ── the golden: each op of the chain, stated element by element ──
    let (m, k, hidden, inter) = (m as usize, k as usize, hidden as usize, inter as usize);
    let golden: Vec<f32> = (0..m)
        .flat_map(|n| {
            // The combine: acc = f32 fma over slots, one f16 narrowing.
            let mut acc = vec![0f32; hidden];
            for slot in 0..k {
                let e_id = idx[n * k + slot] as usize;
                let x_row: Vec<f32> = (0..hidden).map(|j| f16(x[n * hidden + j])).collect();
                // gate/up projections: f32 code-dot, ONE f16 round at the mulf.
                let proj = |bank_w: &Vec<u8>, bank_s: &Vec<f32>, row: &Vec<f32>, out: usize| -> Vec<f32> {
                    let inp = row.len();
                    (0..out)
                        .map(|c| {
                            let slab_row = e_id * out + c;
                            let mut dot = 0f32;
                            for (j, &xv) in row.iter().enumerate() {
                                let wv = decode(bank_w[slab_row * inp + j]);
                                dot += xv * wv;
                            }
                            f16(f16(dot) * f16(bank_s[slab_row]))
                        })
                        .collect()
                };
                let gate = proj(&gw, &gs_v, &x_row, inter);
                let up = proj(&uw, &us_v, &x_row, inter);
                // GeGLU: gelu_tanh(gate) · up, each op f16.
                let act: Vec<f32> = (0..inter)
                    .map(|c| f16(gelu_f16(gate[c]) * up[c]))
                    .collect();
                // down projection back to hidden, over the ACT row.
                let down = proj(&dw, &ds_v, &act, hidden);
                // combine: acc = fma(down, score, acc) in f32 (the tile is
                // F32, so the fma's round_to is the identity).
                let score = f16(scores[n * k + slot]);
                for d in 0..hidden {
                    acc[d] += f16(down[d]) * score;
                }
            }
            acc.iter().map(|&v| f16(v)).collect::<Vec<_>>()
        })
        .collect();

    assert_eq!(got.len(), golden.len(), "chain output width");
    let bad: Vec<(usize, f32, f32)> = golden
        .iter()
        .zip(&got)
        .enumerate()
        .filter(|&(_, (g, w))| (g - w).abs() >= 5e-2 * g.abs().max(1e-3))
        .map(|(i, (g, w))| (i, *g, *w))
        .collect();
    assert!(
        bad.is_empty(),
        "expert chain diverged at {} of {} entries (first: {:?})",
        bad.len(),
        golden.len(),
        bad.first(),
    );
}

/// The DECODE row count — the 26b's first-step regime.
#[test]
fn the_whole_expert_chain_runs_fused_and_matches_the_host_golden() {
    run_case(1);
}

/// The PREFILL row count — the fused-segment planner's multi-row regime.
#[test]
fn the_expert_chain_runs_at_the_prefill_row_count_too() {
    run_case(3);
}
