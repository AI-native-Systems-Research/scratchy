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
//!   * TopK — the TRAILING k slice of the sorted row (metal's
//!     `slice_trailing_cols`: `src_col = axis_size - top_k + j`).
//!   * GatherScores — `take_along_axis` at axis -1: `out[n,j] = src[n, idx[n,j]]`.
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
fn run_one_node(
    ir: &SubtileIR<NeoX>,
    sources: Vec<(u64, Vec<f32>)>,
) -> Vec<f32> {
    let weight_ids = std::collections::HashSet::new();
    let (ops, _layout) = superdsc::lower_graph_to_ktir(
        ir,
        &weight_ids,
        superdsc::ActiveCap::FULL,
        false,
    )
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
        3.0, 1.0, 4.0, 1.5, 2.0, //   ranks: 2,0,4,1,3
        5.0, 2.0, 5.0, 2.0, 7.0, //   duplicates
        1.0, f32::NAN, 3.0, 2.0, 4.0, // NaN row
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
        assert!(
            (g - w).abs() < 0.5,
            "argsort[{i}] golden(rank)={g} got={w}"
        );
    }
}

/// The top-k is the TRAILING k columns of the sorted row — the highest ranks,
/// i.e. the LARGEST scores — matching metal's `slice_trailing_cols`.
#[test]
fn route_topk_takes_the_trailing_k_sorted_columns() {
    let (m, e, k) = (ARGSORT_M, ARGSORT_E, 2u32);
    let scores = argsort_scores();
    // The sorted rows of the fixture, from the golden rank above.
    let sorted: Vec<f32> = (0..m as usize)
        .flat_map(|r| {
            let row = &scores[r * e as usize..(r + 1) * e as usize];
            let mut idx: Vec<usize> = (0..row.len()).collect();
            idx.sort_by(|&a, &b| {
                row[a]
                    .partial_cmp(&row[b])
                    .unwrap_or(std::cmp::Ordering::Greater)
            });
            idx.iter().map(|&i| i as f32).collect::<Vec<_>>()
        })
        .collect();
    let ir = one_node_ir(
        vec![
            TensorShape { rows: m, cols: e },
            TensorShape { rows: m, cols: k },
        ],
        1,
        SubOp::RouteTopK {
            k: scratchy_subtile::subtile_ir::TopK::new(
                std::num::NonZeroU32::new(k).unwrap(),
            ),
        },
        &[0],
    );
    let got = run_one_node(&ir, vec![(0, sorted.clone())]);
    // Golden: the last k columns of each sorted row.
    let golden: Vec<f32> = (0..m as usize)
        .flat_map(|r| sorted[r * e as usize + (e - k) as usize..(r + 1) * e as usize].to_vec())
        .collect();
    assert_eq!(got.len(), golden.len(), "topk width");
    for (i, (g, w)) in golden.iter().zip(&got).enumerate() {
        assert!((g - w).abs() < 0.5, "topk[{i}] golden={g} got={w}");
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
        vec![(0, scores.clone()), (1, idx.clone()), (2, per_expert.clone())],
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
            row.iter().copied().chain(row.iter().copied()).collect::<Vec<_>>()
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
