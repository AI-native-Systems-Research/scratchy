// SPDX-License-Identifier: Apache-2.0
// ⛔ RUN THIS FILE WITH `SCRATCHY_PLAN_ONLY_BAKE=1` — the door side of these gates
// runs the KTIR→SuperDSC lowering, whose fixtures must not demand a device compiler.
//
//     SCRATCHY_PLAN_ONLY_BAKE=1 cargo test -p scratchy-target-spyre \
//         --features spyre-emu --test moe_router_chain
//
// (See `superdsc_time_tile.rs`'s header for why the variable is not set from
// inside the test.)
#![cfg(feature = "spyre-emu")]
//! THE MoE ROUTER'S INDEX CHAIN — the four ops this landing wired into the KTIR
//! lowering (`RouteArgsort`, `RouteTopK`, `RouteGatherScores`,
//! `RouteExpertScale`), each driven through the REAL launch path
//! (`lower_graph_to_ktir` → `ktir_groups` → `SpyreSession::new_multi` /
//! `run_step`) and compared against a HAND-COMPUTED golden that states metal's
//! semantics exactly:
//!
//!   * Argsort — MLX `block_sort` ascending, NaN-as-greater, ties by index:
//!     `rank[i,j] = |{h : x[i,h] < x[i,j]}| + |{h : x[i,h]==x[i,j] ∧ h<j}|`
//!     after NaN→+inf sanitization. Pins the NaN ordering AND the tie-break.
//!   * TopK — the FUSED VENDOR PAIR over the raw logits (the argsort and
//!     the score gather fold into it): a descending insertion ripple,
//!     `topk.ddl`'s k_loop verbatim — slot 0 the maximum, slot j the
//!     (j+1)-th largest, strict `>` so a tie's NEWCOMER takes the slot,
//!     ids are scan positions UNIQUE BY CONSTRUCTION (wall-11's cure).
//!   * GatherScores — `take_along_axis` at axis -1: `out[n,j] = src[n, idx[n,j]]`
//!     (the bare corner only — the chain's own gather is the pair's values
//!     landing).
//!   * ExpertScale — `out[m,j] = scores[m,j] · per_expert_scale[idx[m,j]]`,
//!     one f32 multiply then the f16 narrow (metal's `moe_per_expert_scale`).
//!
//! The golden is hand-computed here rather than `eval_dag` because these ops
//! are EXPANSION ops — the host reference refuses them (the router bundle's
//! sources are opaque to `gather`), so the metal-shader semantics are the only
//! reference, stated once in each test's golden closure.

use scratchy_subtile::subtile_ir::{
    NeoX, RouterBundle, SubOp, SubtileIR, SubtileId, SubtileNode, TensorId, TensorRegion,
    TensorShape,
};
use scratchy_target_spyre::lower_subtile_tape_to_superdsc as superdsc;

/// One `[rows, cols]` source-plus-op-output graph: a single node of `op` over
/// `whole(0..sources)`, writing `whole(out)`. The tensors are laid out source
/// tensors first, then the op's output.
fn one_node_ir(
    tensors: Vec<TensorShape>,
    num_sources: u32,
    op: SubOp,
    input_ids: &[usize],
) -> SubtileIR<NeoX> {
    let tr = |t: usize| TensorRegion {
        tensor: TensorId::from_index(t),
        region: tensors[t].whole(),
    };
    let node = SubtileNode {
        id: SubtileId::from_index(0),
        op,
        inputs: input_ids.iter().map(|&t| tr(t)).collect(),
        output: tr(tensors.len() - 1),
    };
    SubtileIR {
        result: TensorId::from_index(tensors.len() - 1),
        tensors,
        num_sources,
        nodes: vec![node],
        op_output: Vec::new(),
    }
}

/// Lower a one-node graph and run it on the emulator, returning the output
/// buffer. `sources` are the (id, data) pairs in tensor-index order.
fn run_one_node(ir: &SubtileIR<NeoX>, sources: Vec<(u64, Vec<f32>)>) -> Vec<f32> {
    let weight_ids = std::collections::HashSet::new();
    let (ops, _layout) =
        superdsc::lower_graph_to_ktir(ir, &weight_ids, superdsc::ActiveCap::FULL, false)
            .unwrap_or_else(|e| panic!("lower the router node: {e:?}"));
    let out_tid = ir.result.index() as u64;
    let groups = superdsc::ktir_groups(&ops, superdsc::FoldGrouping::Split)
        .expect("bind the router node's program");
    let mut session = scratchy_target_spyre::runner::SpyreSession::new_multi(
        &[(&groups, &[out_tid])],
        Vec::new(),
    )
    .expect("build the router-node session");
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
        .expect("run the router node");
    out[&out_tid].clone()
}

/// The stable ascending rank with NaN-as-greater — the golden argsort.
fn argsort_rank(row: &[f32]) -> Vec<f32> {
    let san: Vec<f32> = row
        .iter()
        .map(|&v| if v.is_nan() { f32::INFINITY } else { v })
        .collect();
    (0..row.len())
        .map(|j| {
            let mut r = 0f32;
            for (h, &v) in san.iter().enumerate() {
                if v < san[j] || (v == san[j] && h < j) {
                    r += 1.0;
                }
            }
            r
        })
        .collect()
}

/// A hand-picked `[m, E]` score fixture: distinct values, DUPLICATES (the
/// tie-break), and a NaN — every branch of the rank form.
const ARGSORT_M: u32 = 3;
const ARGSORT_E: u32 = 5;
fn argsort_scores() -> Vec<f32> {
    // Row 0: distinct. Row 1: duplicates (two 2.0s, two 5.0s). Row 2: a NaN.
    vec![
        3.0,
        1.0,
        4.0,
        1.5,
        2.0, //   ranks: 2,0,4,1,3
        5.0,
        2.0,
        5.0,
        2.0,
        7.0, //   duplicates
        1.0,
        f32::NAN,
        3.0,
        2.0,
        4.0, // NaN row
    ]
}

/// The argsort's rank output equals the stable ascending rank with
/// NaN-as-greater: metal `argpartition.metal`'s exact ordering.
#[test]
fn route_argsort_matches_the_stable_ascending_rank() {
    let scores = argsort_scores();
    let ir = one_node_ir(
        vec![
            TensorShape {
                rows: ARGSORT_M,
                cols: ARGSORT_E,
            },
            TensorShape {
                rows: ARGSORT_M,
                cols: ARGSORT_E,
            },
        ],
        1,
        SubOp::RouteArgsort,
        &[0],
    );
    let got = run_one_node(&ir, vec![(0, scores.clone())]);
    let golden: Vec<f32> = (0..ARGSORT_M as usize)
        .flat_map(|r| argsort_rank(&scores[r * ARGSORT_E as usize..(r + 1) * ARGSORT_E as usize]))
        .collect();
    assert_eq!(got.len(), golden.len(), "argsort width");
    for (i, (g, w)) in golden.iter().zip(&got).enumerate() {
        assert!((g - w).abs() < 0.5, "argsort[{i}] golden(rank)={g} got={w}");
    }
}

/// The fused top-k pair: the argsort and the score gather FOLD INTO the
/// top-k's program, which reads the RAW LOGITS and lands BOTH outputs —
/// the ids (`desc_o`, the node's own) and the values (`desc_v`, the folded
/// gather's output). The chain is the binding walk's own minimum: the
/// ranks' producer must be the argsort, and the top-k must have EXACTLY
/// ONE RouteGatherScores consumer reading the same logits.
///
/// The golden is the vendor pair's own law (`topk.ddl`'s k_loop, mirrored
/// in the kernel's header): a DESCENDING insertion ripple — slot 0 the
/// maximum, slot j the (j+1)-th largest; `run > cand` is STRICT, so on a
/// tie the NEWCOMER takes the slot and the displaced pair ripples down
/// (among equals, the LATER scan position ranks higher); a NaN compares
/// false, enters slot 0, and sinks one slot per later candidate. The ids
/// are scan positions, UNIQUE BY CONSTRUCTION — the property whose absence
/// (the argsort's dup ranks summing expert ids past the bank) faulted the
/// 26b card's FETCH. This pins that uniqueness too: a duplicated id in the
/// golden's ripple is a bug in the model, caught by construction.
#[test]
fn route_topk_lands_the_descending_pair() {
    let (m, e, k) = (ARGSORT_M, ARGSORT_E, 2u32);
    let scores = argsort_scores();
    // t0 logits (source) → t1 ranks → t2 ids → t3 values (the gather's
    // output, where the pair lands the scores).
    let tensors = vec![
        TensorShape { rows: m, cols: e },
        TensorShape { rows: m, cols: e },
        TensorShape { rows: m, cols: k },
        TensorShape { rows: m, cols: k },
    ];
    let tr = |t: usize| TensorRegion {
        tensor: TensorId::from_index(t),
        region: tensors[t].whole(),
    };
    // Typed: none of the router ops carries an `F`-typed field, so the vec
    // alone cannot infer the form — pin it the way `run_one_node`'s callers do.
    let nodes: Vec<SubtileNode<NeoX>> = vec![
        SubtileNode {
            id: SubtileId::from_index(0),
            op: SubOp::RouteArgsort,
            inputs: vec![tr(0)],
            output: tr(1),
        },
        SubtileNode {
            id: SubtileId::from_index(1),
            op: SubOp::RouteTopK {
                k: scratchy_subtile::subtile_ir::TopK::new(std::num::NonZeroU32::new(k).unwrap()),
            },
            inputs: vec![tr(1)],
            output: tr(2),
        },
        SubtileNode {
            id: SubtileId::from_index(2),
            op: SubOp::RouteGatherScores,
            inputs: vec![tr(0), tr(2)],
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
    // Run through the real path, requesting BOTH landings back.
    let weight_ids = std::collections::HashSet::new();
    let (ops, _layout) =
        superdsc::lower_graph_to_ktir(&ir, &weight_ids, superdsc::ActiveCap::FULL, false)
            .unwrap_or_else(|e| panic!("lower the fused pair: {e:?}"));
    let groups = superdsc::ktir_groups(&ops, superdsc::FoldGrouping::Split)
        .expect("bind the fused pair's programs");
    let mut session =
        scratchy_target_spyre::runner::SpyreSession::new_multi(&[(&groups, &[2, 3])], Vec::new())
            .expect("build the fused pair's session");
    let out = session
        .run_step(
            0,
            vec![(0u64, scores.clone(), vec![m as usize, e as usize])],
            &[(2, 0), (3, 0)],
        )
        .expect("run the fused pair");
    let ids = out[&2].clone();
    let vals = out[&3].clone();
    // The golden ripple, per row, verbatim: K running (value, index)
    // registers init (−inf, −1); per candidate h in scan order, per slot j:
    // strict `run > cand` keeps the register, else the newcomer takes the
    // slot and the old pair becomes the candidate.
    let mut golden_ids: Vec<f32> = Vec::with_capacity((m * k) as usize);
    let mut golden_vals: Vec<f32> = Vec::with_capacity((m * k) as usize);
    for r in 0..m as usize {
        let row = &scores[r * e as usize..(r + 1) * e as usize];
        let mut run: Vec<(f32, usize)> = vec![(f32::NEG_INFINITY, usize::MAX); k as usize];
        for (h, &cand) in row.iter().enumerate() {
            let mut cand = cand;
            let mut cand_h = h;
            for slot in run.iter_mut() {
                let old = *slot;
                if old.0 > cand {
                    continue; // strict `run > cand` keeps the register
                }
                // else the NEWCOMER takes the slot and the old pair becomes
                // the candidate for the next slot down
                *slot = (cand, cand_h);
                cand = old.0;
                cand_h = old.1;
            }
        }
        for &(v, i) in &run {
            golden_vals.push(v);
            golden_ids.push(if i == usize::MAX { -1.0 } else { i as f32 });
        }
    }
    assert_eq!(ids.len(), golden_ids.len(), "ids width");
    assert_eq!(vals.len(), golden_vals.len(), "values width");
    for (i, (g, w)) in golden_ids.iter().zip(&ids).enumerate() {
        assert!((g - w).abs() < 0.5, "ids[{i}] golden={g} got={w}");
    }
    for (i, (g, w)) in golden_vals.iter().zip(&vals).enumerate() {
        assert!(
            (g - w).abs() < 2e-2,
            "values[{i}] golden={g} got={w} (the pair lands the RAW logits, NaN row included)"
        );
    }
    // ⛔ THE UNIQUENESS PIN — wall-11's own property: the ids are scan
    // positions, each candidate entering the ripple exactly once, so a row's
    // landed ids are DISTINCT and in range. A duplicated id is the exact
    // defect class that summed expert ids past the bank.
    for r in 0..m as usize {
        let row: Vec<f32> = ids[r * k as usize..(r + 1) * k as usize].to_vec();
        let real: Vec<f32> = row.iter().filter(|&&v| v >= 0.0).copied().collect();
        let mut uniq = real.clone();
        uniq.sort_by(|a, b| a.total_cmp(b));
        uniq.dedup();
        assert_eq!(
            real.len(),
            uniq.len(),
            "row {r}'s landed ids {row:?} must be DISTINCT scan positions — a duplicate is \
             the wall-11 defect class (dup ranks summing expert ids past the bank)"
        );
    }
}

/// The gather is `take_along_axis` at axis -1: `out[n,j] = src[n, idx[n,j]]`.
#[test]
fn route_gather_scores_takes_along_the_last_axis() {
    let (m, e, k) = (3u32, 5u32, 2u32);
    let scores: Vec<f32> = (0..m * e).map(|i| (i as f32) * 0.25 - 1.0).collect();
    let idx: Vec<f32> = vec![4.0, 0.0, 2.0, 3.0, 1.0, 4.0];
    let ir = one_node_ir(
        vec![
            TensorShape { rows: m, cols: e }, // scores (source)
            TensorShape { rows: m, cols: k }, // indices (source)
            TensorShape { rows: m, cols: k }, // out
        ],
        2,
        SubOp::RouteGatherScores,
        &[0, 1],
    );
    let got = run_one_node(&ir, vec![(0, scores.clone()), (1, idx.clone())]);
    let golden: Vec<f32> = (0..(m * k) as usize)
        .map(|flat| {
            let n = flat / k as usize;
            scores[n * e as usize + idx[flat] as usize]
        })
        .collect();
    assert_eq!(got.len(), golden.len(), "gather width");
    for (i, (g, w)) in golden.iter().zip(&got).enumerate() {
        assert!((g - w).abs() < 2e-2, "gather[{i}] golden={g} got={w}");
    }
}

/// The expert scale: `out[m,j] = scores[m,j] · per_expert_scale[idx[m,j]]`,
/// ONE f32 multiply then the f16 narrow.
#[test]
fn route_expert_scale_multiplies_the_per_expert_scale() {
    let (m, k, e) = (2u32, 2u32, 4u32);
    let scores: Vec<f32> = vec![0.5, 0.25, 0.75, 0.125];
    let idx: Vec<f32> = vec![3.0, 1.0, 0.0, 2.0];
    let per_expert: Vec<f32> = vec![1.5, -0.5, 2.0, 0.25];
    let ir = one_node_ir(
        vec![
            TensorShape { rows: m, cols: k }, // scores (source)
            TensorShape { rows: m, cols: k }, // indices (source)
            TensorShape { rows: 1, cols: e }, // per-expert scale row (source)
            TensorShape { rows: m, cols: k }, // out
        ],
        3,
        SubOp::RouteExpertScale {
            router: RouterBundle::Gemma,
        },
        &[0, 1, 2],
    );
    let got = run_one_node(
        &ir,
        vec![
            (0, scores.clone()),
            (1, idx.clone()),
            (2, per_expert.clone()),
        ],
    );
    let golden: Vec<f32> = (0..(m * k) as usize)
        .map(|flat| scores[flat] * per_expert[idx[flat] as usize])
        .collect();
    assert_eq!(got.len(), golden.len(), "expert-scale width");
    for (i, (g, w)) in golden.iter().zip(&got).enumerate() {
        assert!((g - w).abs() < 2e-2, "expert-scale[{i}] golden={g} got={w}");
    }
}

// ── THE EXPERT LAYOUT CHAIN (gathered/decode semantics) ─────────────────────────

/// The sort materializes the pair rows: output column block `j` is the input
/// row, i.e. the `[m, k, w]` (token, slot, width) layout metal's gathered
/// kernels read. Every element moves once, no arithmetic.
#[test]
fn expert_sort_copies_the_row_into_each_slot_block() {
    let (m, w, k) = (3u32, 4u32, 2u32);
    let x: Vec<f32> = (0..m * w).map(|i| (i as f32) * 0.5 - 3.0).collect();
    let idx: Vec<f32> = vec![1.0, 0.0, 3.0, 2.0, 0.0, 1.0];
    let ir = one_node_ir(
        vec![
            TensorShape { rows: m, cols: w }, // x (source)
            TensorShape { rows: m, cols: k }, // indices (source, rides along)
            TensorShape {
                rows: m,
                cols: w * k,
            }, // pair rows out
        ],
        2,
        SubOp::ExpertSort {
            experts: scratchy_subtile::subtile_ir::NumExperts::new(
                std::num::NonZeroU32::new(4).unwrap(),
            ),
            k: scratchy_subtile::subtile_ir::TopK::new(std::num::NonZeroU32::new(k).unwrap()),
            bundle: scratchy_subtile::subtile_ir::ExpertBundle::SwitchGlu,
        },
        &[0, 1],
    );
    let got = run_one_node(&ir, vec![(0, x.clone()), (1, idx)]);
    // Golden: each output column block j is the input row verbatim.
    let golden: Vec<f32> = (0..m as usize)
        .flat_map(|r| {
            let row = &x[r * w as usize..(r + 1) * w as usize];
            row.iter()
                .copied()
                .chain(row.iter().copied())
                .collect::<Vec<_>>()
        })
        .collect();
    assert_eq!(got.len(), golden.len(), "sort width");
    for (i, (g, v)) in golden.iter().zip(&got).enumerate() {
        assert!((g - v).abs() < 1e-3, "sort[{i}] golden={g} got={v}");
    }
}

/// The unsort is the identity copy in the (token, slot) layout.
#[test]
fn expert_unsort_is_the_identity_copy() {
    let (m, w, k) = (2u32, 5u32, 3u32);
    let rows: Vec<f32> = (0..m * w * k).map(|i| (i as f32) * 0.25 - 2.0).collect();
    let routing: Vec<f32> = vec![0.0; (m * k) as usize];
    let ir = one_node_ir(
        vec![
            TensorShape {
                rows: m,
                cols: w * k,
            }, // pair rows (source)
            TensorShape { rows: m, cols: k }, // routing (source, rides along)
            TensorShape {
                rows: m,
                cols: w * k,
            }, // out
        ],
        2,
        SubOp::ExpertUnsort,
        &[0, 1],
    );
    let got = run_one_node(&ir, vec![(0, rows.clone()), (1, routing)]);
    assert_eq!(got.len(), rows.len(), "unsort width");
    for (i, (g, v)) in rows.iter().zip(&got).enumerate() {
        assert!((g - v).abs() < 1e-3, "unsort[{i}] golden={g} got={v}");
    }
}

/// The combine: `out[n, d] = Σ_k rows[n, k·w + d] · scores[n, k]`, the f32
/// fma chain with ONE f16 narrowing — `moe_weighted_sum`'s arithmetic.
#[test]
fn expert_combine_weighted_sums_the_pair_rows() {
    let (m, w, k) = (2u32, 6u32, 3u32);
    let rows: Vec<f32> = (0..m * w * k)
        .map(|i| (((i * 37) % 23) as f32 / 23.0 - 0.5) * 4.0)
        .collect();
    let scores: Vec<f32> = vec![0.5, 0.3, 0.2, 0.1, 0.6, 0.3];
    let ir = one_node_ir(
        vec![
            TensorShape {
                rows: m,
                cols: w * k,
            }, // pair rows (source)
            TensorShape { rows: m, cols: k }, // scores (source)
            TensorShape { rows: m, cols: w }, // out
        ],
        2,
        SubOp::ExpertCombine {
            hidden: w,
            shared: scratchy_subtile::subtile_ir::SharedExpertBound(None),
        },
        &[0, 1],
    );
    let got = run_one_node(&ir, vec![(0, rows.clone()), (1, scores.clone())]);
    // Golden: the f32 fma chain, one f16 rounding at the end (metal's kernel).
    let golden: Vec<f32> = (0..m as usize)
        .flat_map(|n| {
            (0..w as usize)
                .map(|d| {
                    let mut acc = 0f32;
                    for kk in 0..k as usize {
                        acc += rows[(n * k as usize + kk) * w as usize + d]
                            * scores[n * k as usize + kk];
                    }
                    acc
                })
                .collect::<Vec<_>>()
        })
        .collect();
    assert_eq!(got.len(), golden.len(), "combine width");
    for (i, (g, v)) in golden.iter().zip(&got).enumerate() {
        assert!(
            (g - v).abs() <= 2e-2 * g.abs().max(1.0),
            "combine[{i}] golden={g} got={v}"
        );
    }
}

// ── THE EXPERT PROJECTION (gathered fp8 ExpertMatmul) ──────────────────────────

/// Run the sort → matmul pair with the weight bank staged as TYPED BYTES —
/// the fp8 codes 1-byte verbatim (`Arg::TensorBytes` at `Fp8E4m3`, the
/// `weight_arg` contract) and the scales as f16 bytes — through the same
/// launch path as [`run_one_node`].
fn run_expert_matmul(
    ir: &SubtileIR<NeoX>,
    sources: Vec<(u64, Vec<f32>)>,
    weights: Vec<(usize, Vec<u8>, scratchy_tensors::DType, Vec<usize>)>,
) -> Vec<f32> {
    let weight_ids = std::collections::HashSet::new();
    let (ops, _layout) =
        superdsc::lower_graph_to_ktir(ir, &weight_ids, superdsc::ActiveCap::FULL, false)
            .unwrap_or_else(|e| panic!("lower the expert-matmul graph: {e:?}"));
    let out_tid = ir.result.index() as u64;
    let groups = superdsc::ktir_groups(&ops, superdsc::FoldGrouping::Split)
        .expect("bind the expert-matmul programs");
    let mut session =
        scratchy_target_spyre::runner::SpyreSession::new_multi(&[(&groups, &[out_tid])], weights)
            .expect("build the expert-matmul session");
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
        .expect("run the expert-matmul graph");
    out[&out_tid].clone()
}

/// f16 round-trip through the emulator's storage: the value an f16 element
/// holds after `Tile::compute`'s single rounding.
fn f16(v: f32) -> f32 {
    let bits = ktir_emulator::codec::encode(&[v], ktir_emulator::dtypes::DType::F16);
    ktir_emulator::codec::decode(&bits, 1, ktir_emulator::dtypes::DType::F16)[0]
}

/// The rank vector whose DESCENDING top-k IS `idx` in slot order — the fused
/// pair's own selection law (the ripple keeps the k highest, ties by later
/// scan position; P1's provenance law: the expert tapes' ids must come from
/// a real `RouteTopK`, the only mint of in-range ids). Slot `j`'s expert
/// carries rank `E−1−j` — the k highest ranks, descending in slot order —
/// and the unchosen lanes take the remaining ranks `0..E−k−1` ascending; a
/// permutation, tie-free, so the ripple lands the desired ids bit-exact in
/// slot order (every value an integer below E, f16-exact).
fn ranks_for(idx: &[f32], m: u32, k: u32, e: u32) -> Vec<f32> {
    (0..m as usize)
        .flat_map(|r| {
            let row = &idx[r * k as usize..(r + 1) * k as usize];
            let mut rank = vec![0f32; e as usize];
            let mut next_low = 0f32;
            for (h, slot) in rank.iter_mut().enumerate() {
                match row.iter().position(|&id| id as usize == h) {
                    Some(j) => *slot = (e as usize - 1 - j) as f32,
                    None => {
                        *slot = next_low;
                        next_low += 1.0;
                    }
                }
            }
            rank
        })
        .collect()
}

/// The gathered expert projection, against the metal `affine_gather_qmv`
/// semantics: each (token, slot) pair reads ITS expert's slab out of the
/// stacked `[E·out, in]` fp8 bank, contracts in f32 with ONE f16 narrowing,
/// and scales by the expert's per-channel row with ONE more f16 rounding —
/// the same arithmetic chain the dense fp8 matmul (`KtirFunc::matmul_fp8`)
/// established for the W8A8 gate.
///
/// ⛔ THE WEIGHT BLOCK IS DELIBERATELY SMALL ENOUGH TO GO UNBLOCKED (`nb =
/// n`), so the test also exercises the recognize-escape: the emulator's
/// `matmul_tile` pass must NOT re-tile this contraction (it would drop the
/// dynamic expert row corner), and the golden proves it did not — a
/// re-tiled run reads slab 0 for every pair and disagrees on every pair
/// whose expert is not 0.
#[test]
fn expert_matmul_projects_each_pair_through_its_experts_slab() {
    let (m, e, k, w_in, n_out) = (2u32, 3u32, 2u32, 4u32, 5u32);
    // The sort's inputs: token activations `[m, w_in]` and — through the
    // router's honest chain (P1's provenance law) — expert indices `[m, k]`.
    // `ranks` is the constructed rank vector whose descending top-k IS `idx`.
    let x: Vec<f32> = (0..m * w_in).map(|i| (i as f32) * 0.5 - 1.5).collect();
    let idx: Vec<f32> = vec![2.0, 0.0, 1.0, 2.0];
    let ranks = ranks_for(&idx, m, k, e);
    // The stacked fp8 bank `[E·n_out, w_in]` and its scales `[E·n_out]`:
    // distinct per (expert, channel) so a wrong slab is visible everywhere.
    let bank_rows = (e * n_out) as usize;
    let w_f32: Vec<f32> = (0..bank_rows * w_in as usize)
        .map(|i| (((i * 29) % 17) as f32 / 17.0 - 0.5) * 2.0)
        .collect();
    let s_f32: Vec<f32> = (0..bank_rows)
        .map(|i| (((i * 7) % 13) as f32 / 13.0 - 0.5) * 3.0 + 1.0)
        .collect();
    // Encode the codes fp8 and the scales f16 — what the loader stages.
    let w_bytes: Vec<u8> = w_f32
        .iter()
        .map(|&v| ktir_core::codec::f32_to_e4m3(v))
        .collect();
    let s_bytes = ktir_emulator::codec::encode(&s_f32, ktir_emulator::dtypes::DType::F16);
    // The sort's output (the pair rows) is the matmul's rows AND routing
    // input; the indices ride through the sort node. Tensors: x(0) logits(1)
    // topk-idx(6) pairs(2) w(3) s(4) out(5) argsort-ranks(7) gather-vals(8)
    // — the topk's output appended AFTER the result so every downstream
    // `_o{tid}` pin keeps its id.
    let tensors = vec![
        TensorShape {
            rows: m,
            cols: w_in,
        }, // 0: x
        TensorShape { rows: m, cols: e }, // 1: the router logits (source)
        TensorShape {
            rows: m,
            cols: w_in * k,
        }, // 2: pair rows (sort out)
        TensorShape {
            rows: e * n_out,
            cols: w_in,
        }, // 3: stacked fp8 bank
        TensorShape {
            rows: e * n_out,
            cols: 1,
        }, // 4: stacked scales
        TensorShape {
            rows: m,
            cols: n_out * k,
        }, // 5: out
        TensorShape { rows: m, cols: k }, // 6: top-k indices (the sort's routing)
        TensorShape { rows: m, cols: e }, // 7: the argsort's rank vector (folds)
        TensorShape { rows: m, cols: k }, // 8: the gather's scores (folds)
    ];
    let tr = |t: usize| TensorRegion {
        tensor: TensorId::from_index(t),
        region: tensors[t].whole(),
    };
    // The honest router chain's two folding legs: the argsort (its only
    // consumer is the topk) and the gather (its ids come from the topk over
    // the same logits). Both fold into the fused pair, so the pair's binding
    // walk binds and only the topk program mints.
    let argsort = SubtileNode {
        id: SubtileId::from_index(3),
        op: SubOp::RouteArgsort,
        inputs: vec![tr(1)],
        output: tr(7),
    };
    let topk = SubtileNode {
        id: SubtileId::from_index(0),
        op: SubOp::RouteTopK {
            k: scratchy_subtile::subtile_ir::TopK::new(std::num::NonZeroU32::new(k).unwrap()),
        },
        inputs: vec![tr(7)],
        output: tr(6),
    };
    let gather = SubtileNode {
        id: SubtileId::from_index(4),
        op: SubOp::RouteGatherScores,
        inputs: vec![tr(1), tr(6)],
        output: tr(8),
    };
    let sort = SubtileNode {
        id: SubtileId::from_index(1),
        op: SubOp::ExpertSort {
            experts: scratchy_subtile::subtile_ir::NumExperts::new(
                std::num::NonZeroU32::new(e).unwrap(),
            ),
            k: scratchy_subtile::subtile_ir::TopK::new(std::num::NonZeroU32::new(k).unwrap()),
            bundle: scratchy_subtile::subtile_ir::ExpertBundle::SwitchGlu,
        },
        inputs: vec![tr(0), tr(6)],
        output: tr(2),
    };
    let matmul = SubtileNode {
        id: SubtileId::from_index(2),
        op: SubOp::ExpertMatmul {
            proj: scratchy_subtile::subtile_ir::ExpertProj::Gate,
            n: n_out,
            k: scratchy_subtile::subtile_ir::TopK::new(std::num::NonZeroU32::new(k).unwrap()),
            quant: scratchy_subtile::lower::ExpertQuant::declared(64, 8),
            bundle: scratchy_subtile::subtile_ir::ExpertBundle::SwitchGlu,
        },
        inputs: vec![tr(2), tr(2), tr(3), tr(4)],
        output: tr(5),
    };
    let ir = SubtileIR {
        result: TensorId::from_index(5),
        tensors,
        num_sources: 5,
        nodes: vec![argsort, topk, gather, sort, matmul],
        op_output: Vec::new(),
    };
    let got = run_expert_matmul(
        &ir,
        vec![(0, x.clone()), (1, ranks.clone())],
        vec![
            (
                3,
                w_bytes.clone(),
                scratchy_tensors::DType::Fp8E4m3,
                vec![(e * n_out) as usize, w_in as usize],
            ),
            (
                4,
                s_bytes.clone(),
                scratchy_tensors::DType::F16,
                vec![bank_rows],
            ),
        ],
    );
    // ── golden: metal's gathered qmv arithmetic, stated element by element ──
    // Pair (n, slot)'s expert id, its activation row (the sort's copy makes
    // slot `slot`'s block the token row again), and its slab base.
    let decode = |b: u8| ktir_core::codec::e4m3_to_f32(b);
    let mut golden = vec![0f32; (m * n_out * k) as usize];
    for n in 0..m as usize {
        for slot in 0..k as usize {
            let e_id = idx[n * k as usize + slot] as usize;
            let x_row = &x[n * w_in as usize..(n + 1) * w_in as usize];
            for c in 0..n_out as usize {
                // f32 dot over the fp8-decoded slab row, ONE f16 rounding
                // (Tile::compute after the gemv), then the per-channel
                // scale multiply in f32 on that narrowed value, ONE more
                // f16 rounding (binary_float's single Tile::compute).
                let slab_row = (e_id * n_out as usize) + c;
                let mut acc = 0f32;
                for (j, &xv) in x_row.iter().enumerate() {
                    let wv = decode(w_bytes[slab_row * w_in as usize + j]);
                    acc += f16(xv) * wv;
                }
                let part = f16(acc);
                let scale = f16(s_f32[slab_row]);
                golden[(n * k as usize + slot) * n_out as usize + c] = f16(part * scale);
            }
        }
    }
    assert_eq!(got.len(), golden.len(), "expert-matmul width");
    for (i, (g, v)) in golden.iter().zip(&got).enumerate() {
        assert!(
            (g - v).abs() <= 2e-2 * g.abs().max(1.0),
            "expert-matmul[{i}] golden={g} got={v}"
        );
    }
}

/// The BLOCKED bank: a projection wide enough that the whole expert slab
/// does not fit the W-block budget (`n·in·2` past 512 KB) walks `nb`-wide
/// column blocks — the 26b gate/up regime (`[704, 2816]`). This pins the
/// block loop's scale slicing (block `c_off > 0` reads ITS OWN scale rows)
/// and its output placement, against the same element-wise golden.
#[test]
fn expert_matmul_blocks_wide_banks_by_output_columns() {
    let (m, e, k, w_in, n_out) = (2u32, 3u32, 2u32, 2048u32, 256u32);
    // nb must be < n for the blocking to fire: 256·2048·2 = 1 MB > 512 KB
    // budget ⇒ nb = 128, two blocks per pair.
    assert!(
        u64::from(n_out) * u64::from(w_in) * 2 > 512 * 1024,
        "the fixture must exceed the W-block budget"
    );
    let x: Vec<f32> = (0..m * w_in)
        .map(|i| (((i * 31) % 97) as f32 / 97.0 - 0.5) * 2.0)
        .collect();
    let idx: Vec<f32> = vec![0.0, 2.0, 2.0, 1.0];
    let ranks = ranks_for(&idx, m, k, e);
    let bank_rows = (e * n_out) as usize;
    let w_f32: Vec<f32> = (0..bank_rows * w_in as usize)
        .map(|i| (((i * 53) % 29) as f32 / 29.0 - 0.5) * 0.5)
        .collect();
    let s_f32: Vec<f32> = (0..bank_rows)
        .map(|i| (((i * 11) % 19) as f32 / 19.0 - 0.5) + 1.0)
        .collect();
    let w_bytes: Vec<u8> = w_f32
        .iter()
        .map(|&v| ktir_core::codec::f32_to_e4m3(v))
        .collect();
    let s_bytes = ktir_emulator::codec::encode(&s_f32, ktir_emulator::dtypes::DType::F16);
    let tensors = vec![
        TensorShape {
            rows: m,
            cols: w_in,
        }, // 0: x
        TensorShape { rows: m, cols: e }, // 1: the router logits (source)
        TensorShape {
            rows: m,
            cols: w_in * k,
        }, // 2: pair rows (sort out)
        TensorShape {
            rows: e * n_out,
            cols: w_in,
        }, // 3: stacked fp8 bank
        TensorShape {
            rows: e * n_out,
            cols: 1,
        }, // 4: stacked scales
        TensorShape {
            rows: m,
            cols: n_out * k,
        }, // 5: out
        TensorShape { rows: m, cols: k }, // 6: top-k indices (the sort's routing)
        TensorShape { rows: m, cols: e }, // 7: the argsort's rank vector (folds)
        TensorShape { rows: m, cols: k }, // 8: the gather's scores (folds)
    ];
    let tr = |t: usize| TensorRegion {
        tensor: TensorId::from_index(t),
        region: tensors[t].whole(),
    };
    // The honest router chain's two folding legs: the argsort (its only
    // consumer is the topk) and the gather (its ids come from the topk over
    // the same logits). Both fold into the fused pair, so the pair's binding
    // walk binds and only the topk program mints.
    let argsort = SubtileNode {
        id: SubtileId::from_index(3),
        op: SubOp::RouteArgsort,
        inputs: vec![tr(1)],
        output: tr(7),
    };
    let topk = SubtileNode {
        id: SubtileId::from_index(0),
        op: SubOp::RouteTopK {
            k: scratchy_subtile::subtile_ir::TopK::new(std::num::NonZeroU32::new(k).unwrap()),
        },
        inputs: vec![tr(7)],
        output: tr(6),
    };
    let gather = SubtileNode {
        id: SubtileId::from_index(4),
        op: SubOp::RouteGatherScores,
        inputs: vec![tr(1), tr(6)],
        output: tr(8),
    };
    let sort = SubtileNode {
        id: SubtileId::from_index(1),
        op: SubOp::ExpertSort {
            experts: scratchy_subtile::subtile_ir::NumExperts::new(
                std::num::NonZeroU32::new(e).unwrap(),
            ),
            k: scratchy_subtile::subtile_ir::TopK::new(std::num::NonZeroU32::new(k).unwrap()),
            bundle: scratchy_subtile::subtile_ir::ExpertBundle::SwitchGlu,
        },
        inputs: vec![tr(0), tr(6)],
        output: tr(2),
    };
    let matmul = SubtileNode {
        id: SubtileId::from_index(2),
        op: SubOp::ExpertMatmul {
            proj: scratchy_subtile::subtile_ir::ExpertProj::Down,
            n: n_out,
            k: scratchy_subtile::subtile_ir::TopK::new(std::num::NonZeroU32::new(k).unwrap()),
            quant: scratchy_subtile::lower::ExpertQuant::declared(64, 8),
            bundle: scratchy_subtile::subtile_ir::ExpertBundle::SwitchGlu,
        },
        inputs: vec![tr(2), tr(2), tr(3), tr(4)],
        output: tr(5),
    };
    let ir = SubtileIR {
        result: TensorId::from_index(5),
        tensors,
        num_sources: 5,
        nodes: vec![argsort, topk, gather, sort, matmul],
        op_output: Vec::new(),
    };
    let got = run_expert_matmul(
        &ir,
        vec![(0, x.clone()), (1, ranks.clone())],
        vec![
            (
                3,
                w_bytes.clone(),
                scratchy_tensors::DType::Fp8E4m3,
                vec![(e * n_out) as usize, w_in as usize],
            ),
            (
                4,
                s_bytes.clone(),
                scratchy_tensors::DType::F16,
                vec![bank_rows],
            ),
        ],
    );
    let decode = |b: u8| ktir_core::codec::e4m3_to_f32(b);
    let mut golden = vec![0f32; (m * n_out * k) as usize];
    for n in 0..m as usize {
        for slot in 0..k as usize {
            let e_id = idx[n * k as usize + slot] as usize;
            let x_row = &x[n * w_in as usize..(n + 1) * w_in as usize];
            for c in 0..n_out as usize {
                let slab_row = (e_id * n_out as usize) + c;
                let mut acc = 0f32;
                for (j, &xv) in x_row.iter().enumerate() {
                    acc += f16(xv) * decode(w_bytes[slab_row * w_in as usize + j]);
                }
                let part = f16(acc);
                let scale = f16(s_f32[slab_row]);
                golden[(n * k as usize + slot) * n_out as usize + c] = f16(part * scale);
            }
        }
    }
    assert_eq!(got.len(), golden.len(), "blocked expert-matmul width");
    let mut worst = 0f32;
    for (i, (g, v)) in golden.iter().zip(&got).enumerate() {
        let err = (g - v).abs();
        worst = worst.max(err);
        assert!(
            err <= 2e-2 * g.abs().max(1.0),
            "blocked expert-matmul[{i}] golden={g} got={v}"
        );
    }
    // A run whose second block (c_off > 0) read the FIRST block's scale rows
    // or wrote to the wrong output column would show here as a systematic
    // error, not a rounding tail — assert the worst is well inside tolerance.
    assert!(
        worst < 5e-2,
        "worst blocked error {worst} is not a rounding tail"
    );
}
