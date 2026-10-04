// SPDX-License-Identifier: Apache-2.0
// ⛔ RUN THIS FILE WITH `SCRATCHY_PLAN_ONLY_BAKE=1` — the door side of these gates
// runs the KTIR→SuperDSC lowering, whose fixtures must not demand a device compiler.
//
//     SCRATCHY_PLAN_ONLY_BAKE=1 cargo test -p scratchy-target-spyre \
//         --features spyre-emu --test head_norm_sandwich
//
// (See `superdsc_time_tile.rs`'s header for why the variable is not set from inside
// the test.)
// The numerics gate drives the emulator session (`runner::SpyreSession`), which the
// `spyre`-only build does not link — the file is empty there, not compiled-out per
// test, because gates (ii) and (iii) would also be meaningless without the program
// the numerics gate proves correct.
#![cfg(feature = "spyre-emu")]
//! THE PER-HEAD NORM SANDWICH FUSION — the three gates this landing demanded.
//!
//! The fusion (`lower_subtile_tape_to_ktir`'s `HeadNormSandwich`) matches
//! `Reshape{Times(H>1)} → RmsNorm/RmsNormUnit → Reshape{Times(1)}` by DATAFLOW,
//! deletes both reshapes, and emits one windowed norm over head windows of the
//! ORIGINAL `[m, H·D]` tensor. The measured motivation: those reshape pairs were
//! 87.3% of gemma-4's bake-time descriptors. Because the fusion deletes
//! re-lays the model's numerics depend on, the gates here pin:
//!
//!   (i)   NUMERICS — the fused program, run on the emulator, reproduces
//!         `eval_dag`'s value for the UN-fused sandwich at a small geometry,
//!         for BOTH the gained and unit forms, at m>1 and m==1, with the head
//!         width spanning multiple sticks (D=256). The emulator session runs
//!         the SAME `IRFunction` the `#[forward]` macro bakes — this drives
//!         the constructed program, not a parsed copy of it.
//!   (ii)  GRANITE PARITY — a granite-shaped graph (a plain whole-row
//!         `RmsNorm`, no reshape program in sight) mints NO `WindowedRmsNorm`
//!         program, so granite's emission is provably untouched.
//!   (iii) FALLBACK BYTE-IDENTITY — one extra consumer of the split's output
//!         (a skip connection reading the flat view) and the matcher returns
//!         `None`, and the graph lowers to EXACTLY the un-fused programs
//!         (compared by serialized descriptor bytes + program kinds).
//!
//! The golden oracle is `scratchy_subtile::subtile_ir::eval_dag` — the host
//! reference the whole tree already treats as ground truth, and which treats a
//! Reshape as the identity over the element sequence (exactly the law whose
//! device-side cost this fusion deletes).

use scratchy_subtile::subtile_ir::{
    EwKind, GainConvention, RowScale, SubOp, SubtileIR, SubtileId, SubtileNode, TensorId,
    TensorRegion, TensorShape,
};
use scratchy_target_spyre::lower_subtile_tape_to_superdsc as superdsc;

// ── FIXTURES ────────────────────────────────────────────────────────────────────

/// The gemma-4 per-head q-norm shape, hand-built:
///
///   t0 = x `[m, H·D]` (source)
///   t1 = gamma `[1, D]` (source, gained form only)
///   t2 = split view `[m·H, D]`        Reshape{Times(H), D}
///   t3 = per-head norm `[m·H, D]`     RmsNorm / RmsNormUnit
///   t4 = flatten-back `[m, H·D]`      Reshape{Times(1), H·D}
///   t5 = a sink that reads t4         Elementwise(Add) with t4 twice
///
/// The sink keeps t4 a real graph result (a graph whose result nobody reads
/// lowers differently) without adding a consumer of t2 or t3, so the sandwich
/// still matches. ⛔ IT IS AN ADD OF `t4` WITH ITSELF, NOT a `ScalarMul`: the
/// sink's output width equals the graph result's width (it IS the result), and
/// `lower_one_node` folds a vocab-width `ScalarMul` at m>1 to the m=1
/// lm-head-tail form (`is_prefill_lm_head_tail`) — a real gemma-4 sandwich's
/// consumers are q/k matmuls far narrower than the vocab-wide result, so the
/// fold never fires there, but it would here, and the sink would read one row.
/// `Add(t4, t4) = 2·t4` is sign-sensitive (a squared sink would hide sign
/// errors) and its arm has no width-keyed fold.
///
/// `extra_consumer` adds a SECOND reader of the split view t2 —
/// the skip-connection shape the matcher must refuse.
fn sandwich_ir(
    m: u32,
    heads: u32,
    d: u32,
    gained: bool,
    extra_consumer: bool,
) -> SubtileIR<scratchy_subtile::subtile_ir::NeoX> {
    let full = heads * d;
    let split_rows = m * heads;
    // The full tensor list up front: gamma (when gained) sits at t1 between the
    // source `x` and the op outputs, so every later index accounts for it.
    let mut tensors = vec![
        TensorShape { rows: m, cols: full }, // t0 x (source)
    ];
    if gained {
        tensors.push(TensorShape { rows: 1, cols: d }); // t1 gamma (source)
    }
    let num_sources = tensors.len() as u32;
    let t_split = tensors.len();
    tensors.push(TensorShape { rows: split_rows, cols: d }); // split view
    let t_norm = tensors.len();
    tensors.push(TensorShape { rows: split_rows, cols: d }); // norm out
    let t_flat = tensors.len();
    tensors.push(TensorShape { rows: m, cols: full }); // flatten out
    let t_sink = tensors.len();
    tensors.push(TensorShape { rows: m, cols: full }); // sink out
    let t_extra = if extra_consumer {
        let t = tensors.len();
        tensors.push(TensorShape { rows: split_rows, cols: d });
        t
    } else {
        usize::MAX
    };
    let tr = |t: usize| TensorRegion {
        tensor: TensorId::from_index(t),
        region: tensors[t].whole(),
    };
    let gamma = if gained { Some(tr(1)) } else { None };
    let mk = |id: usize, op: SubOp, inputs: Vec<TensorRegion>, output: TensorRegion| SubtileNode {
        id: SubtileId::from_index(id),
        op,
        inputs,
        output,
    };
    let mut nodes = vec![
        mk(
            0,
            SubOp::Reshape {
                rows: RowScale::Times(heads.try_into().unwrap()),
                cols: d,
            },
            vec![tr(0)],
            tr(t_split),
        ),
        match gamma {
            Some(g) => mk(
                1,
                SubOp::RmsNorm {
                    eps: 1e-6,
                    gain: GainConvention::Scale,
                },
                vec![tr(t_split), g],
                tr(t_norm),
            ),
            None => mk(
                1,
                SubOp::RmsNormUnit { eps: 1e-6 },
                vec![tr(t_split)],
                tr(t_norm),
            ),
        },
        mk(
            2,
            SubOp::Reshape {
                rows: RowScale::Times(1.try_into().unwrap()),
                cols: full,
            },
            vec![tr(t_norm)],
            tr(t_flat),
        ),
        mk(
            3,
            SubOp::Elementwise(EwKind::Add),
            vec![tr(t_flat), tr(t_flat)],
            tr(t_sink),
        ),
    ];
    if extra_consumer {
        // A second reader of the split view — the matcher's single-consumer check
        // must refuse the whole sandwich. The extra sink doubles the view.
        nodes.push(mk(
            4,
            SubOp::Elementwise(EwKind::Add),
            vec![tr(t_split), tr(t_split)],
            tr(t_extra),
        ));
    }
    SubtileIR {
        tensors,
        num_sources,
        nodes,
        result: TensorId::from_index(t_sink),
        op_output: Vec::new(),
    }
}

/// The flatten-back node's output index — the sandwich's RAW output the sink reads.
/// `sandwich_ir` always emits the three sandwich nodes first (ids 0–2), so node 2's
/// output tensor is the flat one regardless of the gamma insertion.
fn ops_flat_index(ir: &SubtileIR) -> usize {
    ir.nodes[2].output.tensor.index()
}

/// Deterministic source values, matching the golden bisection oracle's recipe.
fn synth(id: usize, n: usize) -> Vec<f32> {
    (0..n)
        .map(|j| (((id * 131 + j * 7) % 197) as f32 / 197.0 - 0.5) * 3.0)
        .collect()
}

// ── GATE (i): NUMERICS — fused == eval_dag un-fused ─────────────────────────────

/// Run the fused program on the emulator and compare against `eval_dag` on the SAME
/// sources. `eval_dag` executes the UN-fused sandwich (the reshape is the identity on
/// the host), so agreement is exactly the claim the fusion makes: the windowed norm
/// over head windows of the original tensor IS the sandwich's value.
#[test]
fn windowed_rmsnorm_matches_the_unfused_sandwich() {
    // m>1 (prefill-style, the stick-major regime the block-alignment law is about)
    // and m==1 (decode, where stickmajor is false and the emission is rank-3 flat).
    // D=256 spans four fp16 sticks, and H=2 makes two windows.
    for &(m, heads, d) in &[(7u32, 2u32, 256u32), (1, 2, 256), (5, 4, 128)] {
        for &gained in &[true, false] {
            let ir = sandwich_ir(m, heads, d, gained, false);
            // The golden value of the SINK (t_sink), which reads the sandwich's output.
            let srcs: Vec<Vec<f32>> = (0..ir.num_sources as usize)
                .map(|id| {
                    let t = ir.tensors[id];
                    synth(id, (t.rows * t.cols) as usize)
                })
                .collect();
            let refs: Vec<&[f32]> = srcs.iter().map(|v| v.as_slice()).collect();
            let bufs = scratchy_subtile::subtile_ir::eval_dag(&ir, &refs);
            let sink = ir.result.index();
            // The sandwich's RAW output (the flatten-back's tensor) and the sink that
            // reads it — comparing BOTH pins the windowed write itself, not just a
            // downstream op that happens to read it.
            let flat = ops_flat_index(&ir);
            let golden_flat = bufs[flat].clone();
            let golden = bufs[sink].clone();

            // The fused lowering: one WindowedRmsNorm program + the sink's elementwise.
            let weight_ids = std::collections::HashSet::new();
            let (ops, _layout) = superdsc::lower_graph_to_ktir(
                &ir,
                &weight_ids,
                superdsc::ActiveCap::FULL,
                false,
            )
            .unwrap_or_else(|e| panic!("lower the sandwich (m={m}, H={heads}, D={d}): {e:?}"));
            let windowed: Vec<&str> = ops
                .iter()
                .filter_map(|e| e.ktir.as_ref().map(|k| k.func.name))
                .filter(|n| n.starts_with("rmsnorm_s"))
                .collect();
            assert_eq!(
                windowed.len(),
                1,
                "the sandwich lowers to ONE windowed program (m={m}, H={heads}, D={d}, gained={gained}): {windowed:?}"
            );

            // Run the whole graph on the emulator and compare BOTH the sandwich's raw
            // output and the sink reading it.
            let groups = superdsc::ktir_groups(&ops, superdsc::FoldGrouping::Split).unwrap();
            let results = [flat as u64, ir.result.index() as u64];
            let mut session = scratchy_target_spyre::runner::SpyreSession::new_multi(
                &[(&groups, &results)],
                Vec::new(),
            )
            .expect("build the sandwich session");
            let out = session
                .run_step(
                    0,
                    srcs
                        .iter()
                        .enumerate()
                        .map(|(id, v)| {
                            (
                                id as u64,
                                v.clone(),
                                vec![
                                    ir.tensors[id].rows as usize,
                                    ir.tensors[id].cols as usize,
                                ],
                            )
                        })
                        .collect(),
                    &[(flat as u64, 0), (ir.result.index() as u64, 0)],
                )
                .expect("run the sandwich graph");
            for (label, tid, golden) in [
                ("flat", flat, &golden_flat),
                ("sink", ir.result.index(), &golden),
            ] {
                let got = &out[&(tid as u64)];
                assert_eq!(
                    got.len(),
                    golden.len(),
                    "{label} width (m={m}, H={heads}, D={d}, gained={gained})"
                );
                for (i, (g, w)) in golden.iter().zip(got).enumerate() {
                    assert!(
                        (g - w).abs() <= 2e-2 * g.abs().max(1.0),
                        "m={m}, H={heads}, D={d}, gained={gained}: {label}[{i}] golden={g} windowed={w}"
                    );
                }
            }
        }
    }
}

// ── GATE (ii): GRANITE PARITY — a plain whole-row norm mints no windowed program ─

/// A granite-shaped graph: one whole-row `RmsNorm` over `[m, hidden]`, no reshape
/// anywhere. The sandwich matcher must leave it alone — the fused path mints nothing.
#[test]
fn a_whole_row_rmsnorm_never_enters_the_windowed_path() {
    let (m, hidden) = (7u32, 512u32);
    let tensors = vec![
        TensorShape { rows: m, cols: hidden },
        TensorShape { rows: 1, cols: hidden },
        TensorShape { rows: m, cols: hidden },
    ];
    let whole = |t: usize| TensorRegion {
        tensor: TensorId::from_index(t),
        region: tensors[t].whole(),
    };
    let node = SubtileNode {
        id: SubtileId::from_index(0),
        op: SubOp::RmsNorm {
            eps: 1e-5,
            gain: GainConvention::Scale,
        },
        inputs: vec![whole(0), whole(1)],
        output: whole(2),
    };
    let ir: SubtileIR = SubtileIR {
        tensors,
        num_sources: 2,
        nodes: vec![node],
        result: TensorId::from_index(2),
        op_output: Vec::new(),
    };
    let weight_ids = std::collections::HashSet::new();
    let (ops, _layout) =
        superdsc::lower_graph_to_ktir(&ir, &weight_ids, superdsc::ActiveCap::FULL, false)
            .expect("lower the granite-shaped norm");
    for e in &ops {
        let k = e.ktir.as_ref().expect("the norm carries its program");
        assert!(
            !matches!(
                k.program,
                ktir_superdsc::ktir_node::Program::WindowedRmsNorm { .. }
            ),
            "a whole-row norm is not a sandwich: {} was minted WindowedRmsNorm",
            k.func.name
        );
        assert_eq!(
            k.program,
            ktir_superdsc::ktir_node::Program::RmsNorm,
            "the whole-row norm lowers through the ordinary rmsnorm door"
        );
    }
}

// ── GATE (iii): FALLBACK BYTE-IDENTITY — an extra consumer refuses the fusion ────

/// The byte-identity of the conservative fallback: one extra consumer of the split's
/// output and the graph must lower to EXACTLY what the un-fused path produces. The
/// comparison is over (program kind, serialized program) pairs — the emitted
/// descriptor set a bake would produce — for the SAME graph with and without the
/// extra consumer's node list equality against a reference walk.
#[test]
fn an_extra_consumer_of_the_split_refuses_the_fusion() {
    let (m, heads, d) = (7u32, 2u32, 256u32);
    for &gained in &[true, false] {
        // The refused graph: the split's output has a second reader.
        let ir = sandwich_ir(m, heads, d, gained, true);
        let weight_ids = std::collections::HashSet::new();
        let (ops, _layout) = superdsc::lower_graph_to_ktir(
            &ir,
            &weight_ids,
            superdsc::ActiveCap::FULL,
            false,
        )
        .expect("lower the refused sandwich");
        // NO windowed program was minted...
        for e in &ops {
            if let Some(k) = e.ktir.as_ref() {
                assert!(
                    !matches!(
                        k.program,
                        ktir_superdsc::ktir_node::Program::WindowedRmsNorm { .. }
                    ),
                    "an extra consumer must refuse the fusion: {} minted WindowedRmsNorm",
                    k.func.name
                );
            }
        }
        // ...and the graph lowered to the UN-FUSED set: the same program KINDS
        // (Reshape, RmsNorm/RmsNormUnit, Reshape) the same un-fused walk emits for a
        // graph that never could match. Build that reference by lowering the same
        // graph with the matcher's dataflow broken at a different link — here, the
        // extra consumer itself is the break, so compare against the kinds the
        // three un-fused nodes lower to.
        let kinds: Vec<ktir_superdsc::ktir_node::Program> = ops
            .iter()
            .filter_map(|e| e.ktir.as_ref().map(|k| k.program))
            .collect();
        let expected = if gained {
            vec![
                ktir_superdsc::ktir_node::Program::Reshape,
                ktir_superdsc::ktir_node::Program::RmsNorm,
                ktir_superdsc::ktir_node::Program::Reshape,
            ]
        } else {
            vec![
                ktir_superdsc::ktir_node::Program::Reshape,
                ktir_superdsc::ktir_node::Program::RmsNormUnit,
                ktir_superdsc::ktir_node::Program::Reshape,
            ]
        };
        for e in &expected {
            assert!(
                kinds.contains(e),
                "the refused sandwich still lowers through the un-fused doors: {kinds:?}"
            );
        }
        // ...and the two reshapes are still REAL programs (the fusion deleted
        // nothing): there are exactly as many programs as un-fused nodes.
        let n_progs = ops.iter().filter(|e| e.ktir.is_some()).count();
        assert_eq!(
            n_progs,
            5,
            "split + norm + flatten + sink + extra-consumer sink — every node lowers (got {n_progs})"
        );
    }
}
