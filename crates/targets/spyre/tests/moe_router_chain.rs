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

/// The top-k selects, from the argsort's RANK vector, the indices of the k
/// highest-scoring experts — exactly metal's `sorted[..., -k:]` selection
/// (`slice_trailing_cols`'s experts), in the same ascending-score order.
///
/// ⛔ THE INPUT IS THE RANK VECTOR, NOT THE SORTED-INDEX PERMUTATION. This
/// lowering's argsort emits ranks (`rank[n,h]` = expert h's sorted position),
/// and top-k INVERTS that permutation (a one-hot selector sum over the rank
/// row); metal's kernel emits the permutation and slices it. The two compose
/// to the same experts — this pins that equivalence, on the same fixture rows
/// (distinct values, duplicates, a NaN) the argsort test uses.
#[test]
fn route_topk_inverts_the_rank_vector_to_the_topk_experts() {
    let (m, e, k) = (ARGSORT_M, ARGSORT_E, 2u32);
    let scores = argsort_scores();
    // The rank rows, exactly what RouteArgsort's program writes.
    let ranks: Vec<f32> = (0..m as usize)
        .flat_map(|r| argsort_rank(&scores[r * e as usize..(r + 1) * e as usize]))
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
    let got = run_one_node(&ir, vec![(0, ranks.clone())]);
    // Golden: the k highest-scoring experts' indices, ascending by score —
    // the trailing slice of metal's ascending sorted-index permutation. NaN
    // counts as greatest (the argsort's own sanitization; a bare
    // `partial_cmp().unwrap_or(Greater)` is NOT a total order on NaNs).
    let golden: Vec<f32> = (0..m as usize)
        .flat_map(|r| {
            let row = &scores[r * e as usize..(r + 1) * e as usize];
            let san: Vec<f32> = row
                .iter()
                .map(|&v| if v.is_nan() { f32::INFINITY } else { v })
                .collect();
            let mut idx: Vec<usize> = (0..row.len()).collect();
            idx.sort_by(|&a, &b| san[a].total_cmp(&san[b]));
            idx[e as usize - k as usize..]
                .iter()
                .map(|&i| i as f32)
                .collect::<Vec<_>>()
        })
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
    let (ops, _layout) = superdsc::lower_graph_to_ktir(
        ir,
        &weight_ids,
        superdsc::ActiveCap::FULL,
        false,
    )
    .unwrap_or_else(|e| panic!("lower the expert-matmul graph: {e:?}"));
    let out_tid = ir.result.index() as u64;
    let groups = superdsc::ktir_groups(&ops, superdsc::FoldGrouping::Split)
        .expect("bind the expert-matmul programs");
    let mut session = scratchy_target_spyre::runner::SpyreSession::new_multi(
        &[(&groups, &[out_tid])],
        weights,
    )
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
    // The sort's inputs: token activations `[m, w_in]` and expert indices
    // `[m, k]` (staged f16, as RouteTopK emits them).
    let x: Vec<f32> = (0..m * w_in).map(|i| (i as f32) * 0.5 - 1.5).collect();
    let idx: Vec<f32> = vec![2.0, 0.0, 1.0, 2.0];
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
    let w_bytes: Vec<u8> = w_f32.iter().map(|&v| ktir_core::codec::f32_to_e4m3(v)).collect();
    let s_bytes = ktir_emulator::codec::encode(&s_f32, ktir_emulator::dtypes::DType::F16);
    // The sort's output (the pair rows) is the matmul's rows AND routing
    // input; the indices ride through the sort node. Tensors: x(0) idx(1)
    // pairs(2) w(3) s(4) out(5).
    let tensors = vec![
        TensorShape {
            rows: m,
            cols: w_in,
        }, // 0: x
        TensorShape { rows: m, cols: k }, // 1: indices
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
    ];
    let tr = |t: usize| TensorRegion {
        tensor: TensorId::from_index(t),
        region: tensors[t].whole(),
    };
    let sort = SubtileNode {
        id: SubtileId::from_index(0),
        op: SubOp::ExpertSort {
            experts: scratchy_subtile::subtile_ir::NumExperts::new(
                std::num::NonZeroU32::new(e).unwrap(),
            ),
            k: scratchy_subtile::subtile_ir::TopK::new(std::num::NonZeroU32::new(k).unwrap()),
            bundle: scratchy_subtile::subtile_ir::ExpertBundle::SwitchGlu,
        },
        inputs: vec![tr(0), tr(1)],
        output: tr(2),
    };
    let matmul = SubtileNode {
        id: SubtileId::from_index(1),
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
        nodes: vec![sort, matmul],
        op_output: Vec::new(),
    };
    let got = run_expert_matmul(
        &ir,
        vec![(0, x.clone()), (1, idx.clone())],
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
    let bank_rows = (e * n_out) as usize;
    let w_f32: Vec<f32> = (0..bank_rows * w_in as usize)
        .map(|i| (((i * 53) % 29) as f32 / 29.0 - 0.5) * 0.5)
        .collect();
    let s_f32: Vec<f32> = (0..bank_rows)
        .map(|i| (((i * 11) % 19) as f32 / 19.0 - 0.5) + 1.0)
        .collect();
    let w_bytes: Vec<u8> = w_f32.iter().map(|&v| ktir_core::codec::f32_to_e4m3(v)).collect();
    let s_bytes = ktir_emulator::codec::encode(&s_f32, ktir_emulator::dtypes::DType::F16);
    let tensors = vec![
        TensorShape {
            rows: m,
            cols: w_in,
        }, // 0: x
        TensorShape { rows: m, cols: k }, // 1: indices
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
    ];
    let tr = |t: usize| TensorRegion {
        tensor: TensorId::from_index(t),
        region: tensors[t].whole(),
    };
    let sort = SubtileNode {
        id: SubtileId::from_index(0),
        op: SubOp::ExpertSort {
            experts: scratchy_subtile::subtile_ir::NumExperts::new(
                std::num::NonZeroU32::new(e).unwrap(),
            ),
            k: scratchy_subtile::subtile_ir::TopK::new(std::num::NonZeroU32::new(k).unwrap()),
            bundle: scratchy_subtile::subtile_ir::ExpertBundle::SwitchGlu,
        },
        inputs: vec![tr(0), tr(1)],
        output: tr(2),
    };
    let matmul = SubtileNode {
        id: SubtileId::from_index(1),
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
        nodes: vec![sort, matmul],
        op_output: Vec::new(),
    };
    let got = run_expert_matmul(
        &ir,
        vec![(0, x.clone()), (1, idx.clone())],
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
    assert!(worst < 5e-2, "worst blocked error {worst} is not a rounding tail");
}
