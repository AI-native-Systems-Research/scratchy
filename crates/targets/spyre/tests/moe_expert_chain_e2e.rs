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
    AttnMask, ExpertBundle, ExpertProj, GatedAct, KvCacheLayout, NeoX, NumExperts, RopeFormTag,
    RouterBundle, SubOp, SubtileIR, SubtileId, SubtileNode, TensorId, TensorRegion, TensorShape,
    TopK,
};
use scratchy_target_spyre::lower_subtile_tape_to_superdsc as superdsc;

/// The gemma-4 expert half as one graph, at a small geometry of the 26b's own
/// proportions (stacked banks, k slots, GeGLU).
///
/// Sources: `t0` = x `[m, hidden]`, `t1` = the router logits `[m, e]` (the
/// honest router chain — argsort, topk, gather — runs over them; the argsort
/// and the gather both FOLD into the fused pair, whose descending ripple
/// lands the expert indices), `t2` = scores `[m, k]`, then each projection's
/// codes `[E·out, in]` and scales `[E·out, 1]`.
fn expert_chain_ir(m: u32, e: u32, k: u32, hidden: u32, inter: u32) -> SubtileIR<NeoX> {
    let mut tensors = vec![
        TensorShape {
            rows: m,
            cols: hidden,
        }, // t0 x (source)
        TensorShape { rows: m, cols: e }, // t1 rank vector (source)
        TensorShape { rows: m, cols: k }, // t2 scores (source)
        TensorShape {
            rows: e * inter,
            cols: hidden,
        }, // t3 gate codes (source)
        TensorShape {
            rows: e * inter,
            cols: 1,
        }, // t4 gate scales (source)
        TensorShape {
            rows: e * inter,
            cols: hidden,
        }, // t5 up codes (source)
        TensorShape {
            rows: e * inter,
            cols: 1,
        }, // t6 up scales (source)
        TensorShape {
            rows: e * hidden,
            cols: inter,
        }, // t7 down codes (source)
        TensorShape {
            rows: e * hidden,
            cols: 1,
        }, // t8 down scales (source)
    ];
    let num_sources = tensors.len() as u32;
    let mut nodes = Vec::new();
    let mut next = tensors.len();
    let nz = |n: u32| std::num::NonZeroU32::new(n).unwrap();
    let experts = NumExperts::new(nz(e));
    let topk = TopK::new(nz(k));
    let mut push = |op: SubOp, inputs: Vec<usize>, tensors: &mut Vec<TensorShape>| -> usize {
        let (rows, cols) = match &op {
            SubOp::RouteArgsort => (m, e),
            SubOp::RouteTopK { .. } => (m, k),
            SubOp::RouteGatherScores => (m, k),
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
    // The honest router chain FIRST (P1's provenance law: the sort's ids
    // must be a `RouteTopK` output, the only mint of in-range ids): the
    // argsort over the raw logits t1, the topk whose descending ripple lands
    // the desired idx, and the gather that closes the chain so the pair's
    // binding walk binds. The argsort and the gather both FOLD into the
    // fused pair, so only the topk program mints — and the sort routes
    // through its indices.
    let ranks = push(SubOp::RouteArgsort, vec![1], &mut tensors);
    let indices = push(SubOp::RouteTopK { k: topk }, vec![ranks], &mut tensors);
    let _scores = push(SubOp::RouteGatherScores, vec![1, indices], &mut tensors);
    let pairs = push(
        SubOp::ExpertSort {
            experts,
            k: topk,
            bundle,
        },
        vec![0, indices],
        &mut tensors,
    );
    let gate = push(
        SubOp::ExpertMatmul {
            proj: ExpertProj::Gate,
            n: inter,
            k: topk,
            quant,
            bundle,
        },
        vec![pairs, pairs, 3, 4],
        &mut tensors,
    );
    let up = push(
        SubOp::ExpertMatmul {
            proj: ExpertProj::Up,
            n: inter,
            k: topk,
            quant,
            bundle,
        },
        vec![pairs, pairs, 5, 6],
        &mut tensors,
    );
    let act = push(
        SubOp::ExpertGatedAct {
            act: GatedAct::Gelu,
        },
        vec![gate, up],
        &mut tensors,
    );
    let down = push(
        SubOp::ExpertMatmul {
            proj: ExpertProj::Down,
            n: hidden,
            k: topk,
            quant,
            bundle,
        },
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

/// The router logits whose descending top-k IS `idx` (the fused pair's own
/// selection law: the ripple keeps the k highest, ties by later scan
/// position). Slot `j`'s expert carries logit `k−j` and the unchosen lanes
/// take distinct negatives below −1 — every value f16-exact, the whole row
/// tie-free, so the ripple's slot order is exactly `idx`'s.
fn logits_for(idx: &[f32], m: u32, k: u32, e: u32) -> Vec<f32> {
    (0..m as usize)
        .flat_map(|r| {
            let row = &idx[r * k as usize..(r + 1) * k as usize];
            let mut lg = vec![0f32; e as usize];
            let mut next_low = -1f32;
            for (h, slot) in lg.iter_mut().enumerate() {
                match row.iter().position(|&id| id as usize == h) {
                    Some(j) => *slot = (k as usize - j) as f32,
                    None => {
                        *slot = next_low;
                        next_low -= 1.0;
                    }
                }
            }
            lg
        })
        .collect()
}

/// The device's gelu-tanh polynomial, op by op in f16 — `KtirFunc::gelu`'s
/// exact chain (`binop`/`unop` round each result to the tile's f16).
fn gelu_f16(x: f32) -> f32 {
    let k = 0.797_884_6f32; // sqrt(2/pi), f16-splatted — the golden's own spelling
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
    let (ops, _layout) =
        superdsc::lower_graph_to_ktir(ir, &weight_ids, superdsc::ActiveCap::FULL, false)
            .unwrap_or_else(|e| panic!("lower the expert chain: {e:?}"));
    let out_tid = ir.result.index() as u64;
    let groups = superdsc::ktir_groups(&ops, superdsc::FoldGrouping::Split)
        .expect("bind the expert chain's programs");
    let mut session =
        scratchy_target_spyre::runner::SpyreSession::new_multi(&[(&groups, &[out_tid])], weights)
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
    let w_bytes: Vec<u8> = codes
        .iter()
        .map(|&v| ktir_core::codec::f32_to_e4m3(v))
        .collect();
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
        .map(|_| (2.0 * draw() - 1.0) as f32)
        .collect();
    // DISTINCT experts per (row, slot) — a wrong slab is visible everywhere.
    let idx: Vec<f32> = (0..m)
        .flat_map(|r| {
            let e0 = ((r as usize + 1) % e as usize) as f32;
            let e1 = ((r as usize + 3) % e as usize) as f32;
            vec![e0, e1]
        })
        .collect();
    // The router logits whose descending top-k IS that idx — what the tape
    // feeds through its honest router chain (the fused pair's ripple).
    let logits = logits_for(&idx, m, k, e);
    let scores: Vec<f32> = (0..m).flat_map(|_| vec![0.6f32, 0.4f32]).collect();
    let (gw, gs, gs_v) = bank(e, inter, hidden, 0x1234_5678);
    let (uw, us, us_v) = bank(e, inter, hidden, 0x9abc_def0);
    let (dw, ds, ds_v) = bank(e, hidden, inter, 0x1357_9bdf);
    let weights = vec![
        (
            3,
            gw.clone(),
            scratchy_tensors::DType::Fp8E4m3,
            vec![(e * inter) as usize, hidden as usize],
        ),
        (
            4,
            gs.clone(),
            scratchy_tensors::DType::F16,
            vec![(e * inter) as usize],
        ),
        (
            5,
            uw.clone(),
            scratchy_tensors::DType::Fp8E4m3,
            vec![(e * inter) as usize, hidden as usize],
        ),
        (
            6,
            us.clone(),
            scratchy_tensors::DType::F16,
            vec![(e * inter) as usize],
        ),
        (
            7,
            dw.clone(),
            scratchy_tensors::DType::Fp8E4m3,
            vec![(e * hidden) as usize, inter as usize],
        ),
        (
            8,
            ds.clone(),
            scratchy_tensors::DType::F16,
            vec![(e * hidden) as usize],
        ),
    ];
    let got = run_chain(
        &ir,
        vec![(0, x.clone()), (1, logits.clone()), (2, scores.clone())],
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
                let proj =
                    |bank_w: &Vec<u8>, bank_s: &Vec<f32>, row: &Vec<f32>, out: usize| -> Vec<f32> {
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
                let act: Vec<f32> = (0..inter).map(|c| f16(gelu_f16(gate[c]) * up[c])).collect();
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

// ── W29 COVERAGE PIN ─────────────────────────────────────────────────────────
//
// ⛔ WHY THE TWO CASES ABOVE COULD NOT CATCH THE ROUTER-W STAGING CLASS, AND
// THIS TEST EXISTS. The 26b card defect: `SubOp::RouterLogits` was EXCLUDED
// from the kernel-weight manifest's collection (only `MatmulTile` was
// gathered), so the shim's `None` arm staged the router's `[E,hidden]`
// projection VERBATIM — the checkpoint's row-major bytes — into the
// stick-tiled slot the gemm addresses. The card computed `A @ perm(W)`: right
// ranking form, wrong experts, junk. Every emulator-level suite stayed green
// through it, for three structural reasons, each one a way this file is blind:
//
//   1. THIS FILE'S IR MINTS NO `RouterLogits` NODE — its "router chain" starts
//      at `RouteArgsort` over a fed-in logits SOURCE (`t1`), so the router's
//      own GEMM and its weight never exist here.
//   2. THE WEIGHTS ARE HAND-STAGED — `run_chain` binds caller bytes straight
//      into `SpyreSession::new_multi` (the EMULATOR session). No manifest, no
//      re-tile, no worker pad: the production staging path this defect lived
//      in is never executed.
//   3. THE EMULATOR CANNOT SEE A STAGING LAYOUT AT ALL — it reads a weight
//      operand through the program's row-major transpose-B `indexing_maps`
//      (see `spyre_load.rs`'s emu path), not through the card's per-core stick
//      addressing, so a layout-only defect is numerically invisible to it.
//   4. THE UNROLLED ENTRY BUILDS NO MANIFEST — `lower_graph_to_ktir` (what
//      `run_chain` calls, and what its `let (ops, _layout)` then discards)
//      never runs the re-rolled walk that collects `kernel0` and writes
//      `kernel_weights`; only `lower_subtile_tape_to_superdsc` — the tape
//      entry the `#[forward]` bake calls — does. A manifest law can only be
//      pinned at that entry.
//
// The pin closes the gap at the MANIFEST — the one artifact both the card
// staging and the worker's zero-pad key on: a `RouterLogits` node's weight
// operand MUST land in `kernel_weights` with the device tile layout
// `[for_pointwise(E)/64, k, 64]` — the UNCONDITIONAL pointwise-floor bump, the
// same law the slot reservation sizes every gemm weight at, NOT the
// m-conditional `for_output` (the router is the one real weight below the
// util floor at m=1: E=128, k=2816 → 360K macs, so `for_output` would leave
// it at 2 sticks while its slot and every m≥3 gemm read 8).

/// The gemma router HEAD as a ROLLABLE LAYER, chained four deep: per layer ℓ,
/// `RopeAppend(xℓ, cos, sin, Vℓ, Kcℓ, Vcℓ)` (the KV writer that names the layer —
/// "a layer is named by its KV writer") → `RouterLogits(xℓ, W_routerℓ)` → the honest
/// argsort → top-k → the top-k's one `RouteGatherScores` consumer → `MatmulTile(xℓ,
/// W_denseℓ)` (the residual producer). `x0` is a source; `xℓ = mm(ℓ-1)`'s output for
/// ℓ ≥ 1, so the chain carries the residual layer-to-layer exactly as a real model
/// does, and each layer's weights/KV/V are that layer's OWN sources (the per-layer
/// weight tables the roll gathers).
///
/// ⛔ WHY FOUR LAYERS, AND WHY THIS NODE ORDER — the roll's own laws, each one a way a
/// simpler head-only graph cannot even lower:
///   1. THE RUN DETECTOR NEEDS ≥2 IDENTICAL COPIES, and `reroll_fingerprint` hashes
///      the input KINDS (External ≠ Computed): layer 0 reads the `x0` SOURCE and can
///      never be in the run, so layers 1..3 are the three fingerprint-identical copies
///      and layer 0 is the prefix. A head-only graph has NO second copy — its best run
///      is a spurious alloc/free micro-run with no `Compute`, which the residual
///      pre-pass panics on ("a loop body holds at least one Compute").
///   2. EVERY LOOP BODY NEEDS A `RopeAppend` — `bodies[bi].layer_ids` is filled only
///      from the body's rope's `per_layer_out` (`layer_index()` is `RopeAppend`-only),
///      and the walk refuses a body whose layer ids do not cover its iterations.
///   3. THE RESIDUAL PRE-PASS ALIASES the body's FIRST node's input[0] onto its LAST
///      node's output — the loop-carried hidden — so the rope must be FIRST (its K
///      input is the residual `xℓ`) and the matmul LAST (its output is `xℓ+1`),
///      matching a real layer's node order.
///
/// Returns the IR plus the per-layer `W_router` and `W_dense` tensor ids, so the pin
/// can name every layer's manifest entry (the defect was per-layer: every layer's
/// router projection was mis-staged, not one).
fn router_head_ir(m: u32, e: u32, hidden: u32) -> (SubtileIR<NeoX>, Vec<usize>, Vec<usize>) {
    const LAYERS: u32 = 4;
    let hd = 64; // hidden/heads: 256/4 — a whole number of NeoX heads
    let cap = 32; // the KV caches' capacity rows
    let d = hidden; // the dense companion's n — SAME width as the residual, so the
    // pre-pass's in-place aliasing (first input[0] ↔ last output) is shape-honest.
    let mut tensors = vec![
        TensorShape {
            rows: m,
            cols: hidden,
        }, // t0 = x0, the residual seed (source)
        TensorShape {
            rows: m,
            cols: hidden,
        }, // t1 = cos (shared source)
        TensorShape {
            rows: m,
            cols: hidden,
        }, // t2 = sin (shared source)
    ];
    let mut w_router = Vec::new();
    let mut w_dense = Vec::new();
    for _ in 0..LAYERS {
        tensors.push(TensorShape {
            rows: m,
            cols: hidden,
        }); // V
        tensors.push(TensorShape {
            rows: cap,
            cols: hidden,
        }); // K_cache
        tensors.push(TensorShape {
            rows: cap,
            cols: hidden,
        }); // V_cache
        tensors.push(TensorShape {
            rows: hidden,
            cols: e,
        }); // W_router [k, n] (source)
        tensors.push(TensorShape {
            rows: hidden,
            cols: d,
        }); // W_dense [k, n] (source)
        w_router.push(tensors.len() - 2);
        w_dense.push(tensors.len() - 1);
    }
    let num_sources = tensors.len() as u32;
    let nz = |n: u32| std::num::NonZeroU32::new(n).unwrap();
    let experts = NumExperts::new(nz(e));
    let topk = TopK::new(nz(2));
    let mut nodes = Vec::new();
    let mut next = tensors.len();
    let mut push = |op: SubOp, inputs: Vec<usize>, tensors: &mut Vec<TensorShape>| -> usize {
        let (rows, cols) = match &op {
            SubOp::RopeAppend { .. } => (m, hidden),
            SubOp::RouterLogits { .. } => (m, e),
            SubOp::RouteArgsort => (m, e),
            SubOp::RouteTopK { .. } => (m, 2),
            SubOp::RouteGatherScores => (m, 2),
            SubOp::MatmulTile { n, .. } => (m, *n),
            _ => unreachable!("router head op"),
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
    let mut result = 0;
    for l in 0..LAYERS {
        let (v, kc, vc, wr, wd) = (
            3 + 5 * l as usize,
            4 + 5 * l as usize,
            5 + 5 * l as usize,
            6 + 5 * l as usize,
            7 + 5 * l as usize,
        );
        let x = if l == 0 { 0 } else { result }; // the chain: xℓ = mm(ℓ-1)'s output
        // The KV writer FIRST — it names the layer and its K input is the residual.
        let _rot_k = push(
            SubOp::RopeAppend {
                head_dim: scratchy_subtile::model_geometry::HeadDim::new(hd),
                layer: l,
                attn: AttnMask::Causal,
                pairing: RopeFormTag::NeoX,
                layout: KvCacheLayout::for_cache_tensors(
                    TensorId::from_index(kc),
                    TensorId::from_index(vc),
                ),
                _form: std::marker::PhantomData,
            },
            vec![x, 1, 2, v, kc, vc],
            &mut tensors,
        );
        let lg = push(
            SubOp::RouterLogits {
                experts,
                router: RouterBundle::Gemma,
            },
            vec![x, wr],
            &mut tensors,
        );
        let ranks = push(SubOp::RouteArgsort, vec![lg], &mut tensors);
        let idx = push(SubOp::RouteTopK { k: topk }, vec![ranks], &mut tensors);
        // The top-k's ONE RouteGatherScores consumer — the fused pair's contract.
        let _scores = push(SubOp::RouteGatherScores, vec![lg, idx], &mut tensors);
        // The dense companion LAST — its output is the next layer's residual.
        result = push(
            SubOp::MatmulTile {
                n: d,
                weight: scratchy_subtile::lower::GemmWeight::Dense,
            },
            vec![x, wd],
            &mut tensors,
        );
    }
    (
        SubtileIR {
            result: TensorId::from_index(result),
            tensors,
            num_sources,
            nodes,
            op_output: Vec::new(),
        },
        w_router,
        w_dense,
    )
}

/// The router's projection IS a retiled kernel weight, at the POINTWISE floor — for
/// EVERY layer of a rolled model, not just one.
///
/// `m=3` keeps the matmul below the util floor (3·128·256 = 98K macs < 2^20), so the
/// m-conditional `for_output(3, E, k)` would say 2 out-sticks — the pin's 8 proves the
/// law is the unconditional bump, and each entry's EXISTENCE proves the collection no
/// longer skips `RouterLogits` (the exclusion itself was the 26b card defect: no entry
/// → verbatim staging → `A @ perm(W)`, on every layer). The per-layer copies (the
/// loop's iterations 2 and 3) must carry the SAME descriptor: the manifest inserts
/// them from the roll's own per-layer tables, and a missing copy would stage that
/// layer's router weight verbatim — the same defect, one layer later.
#[test]
fn the_router_projection_w_is_a_retiled_kernel_weight() {
    let (m, e, hidden) = (3u32, 128u32, 256u32);
    let (ir, w_router, w_dense) = router_head_ir(m, e, hidden);
    let weight_ids = std::collections::HashSet::new();
    // ⭐ THE BAKE'S OWN SEQUENCE, verbatim from `codegen.rs`: validate →
    // `lower_dag_to_tape` → the class roll (layer classes first, the
    // single-class roll as the fallback — this single-class chain takes the
    // fallback, which rolls layers 1..3 into one 3-iteration loop with layer 0
    // as the prefix) → `lower_subtile_tape_to_superdsc`. That re-rolled walk is
    // the ONLY place `kernel_weights` is written, so it is the only entry this
    // law can be pinned at (reason 4 above).
    let valid =
        scratchy_subtile::subtile_ir::ValidatedGraph::new(&ir).expect("the router head validates");
    let tape_unrolled = scratchy_subtile::subtile_tape::lower_dag_to_tape(&valid);
    let tape_rolled = scratchy_subtile::subtile_tape::reroll_layer_classes(&tape_unrolled, &ir)
        .unwrap_or_else(|| {
            scratchy_subtile::subtile_tape::reroll_subtile_tape(&tape_unrolled, &ir)
        });
    let rolled = superdsc::lower_subtile_tape_to_superdsc(
        &tape_rolled,
        &ir,
        &weight_ids,
        superdsc::ActiveCap::FULL,
        false,
        // The router head walks at m=3 — three rows, not a one-request decode, no
        // split-KV chunk, and no composite argmax (neither tied nor one-request).
        superdsc::RungFacts {
            one_request: false,
            chunk_pages: None,
            argmax: None,
        },
    )
    .unwrap_or_else(|e| panic!("lower the router head: {e:?}"));
    // The POINTWISE floor, not the m-conditional matmul width: at m=3 the
    // matmul is below the util floor (2 sticks), the floor bumps to 8.
    let n_router = superdsc::DeviceWidth::for_pointwise(e).get();
    assert_eq!(n_router, 512, "the 26b router geometry bumps E=128 to 512");
    // The dense companion's n (== hidden) lands on the same floor here —
    // for_pointwise(256) is 512 too, so both families pin the SAME law.
    let n_dense = superdsc::DeviceWidth::for_pointwise(hidden).get();
    assert_eq!(n_dense, 512);
    for (tids, n_out, what) in [
        (&w_router, n_router, "router"),
        (&w_dense, n_dense, "dense companion"),
    ] {
        for &t in tids {
            let w = rolled
                .layout
                .kernel_weights
                .get(&(t as u32))
                .unwrap_or_else(|| {
                    panic!(
                        "layer weight t{t} ({what}) MUST be a kernel weight — a missing entry \
                         stages it verbatim into a stick-tiled slot (the 26b router-defect class)"
                    )
                });
            assert_eq!(
                w.device_size,
                vec![(n_out / 64) as u64, hidden as u64, 64],
                "device tile = [out/64, in, 64] at the pointwise floor — the same width the \
                 slot reservation sizes and the worker zero-pads to ({what})"
            );
            assert_eq!(w.stick_size, 64, "fp16 stick ({what})");
            assert_eq!(w.word_length, 2, "fp16 word ({what})");
            // The byte contract the shim's staging guard checks:
            // prod(device_size)·wl == the worker's padded buffer (n·k·2).
            let staged: u64 = w.device_size.iter().product::<u64>() * w.word_length as u64;
            assert_eq!(staged, (n_out * hidden * 2) as u64, "staged bytes ({what})");
        }
    }
}
