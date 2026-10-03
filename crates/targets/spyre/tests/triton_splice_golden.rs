// SPDX-License-Identifier: Apache-2.0
//! THE SPLICE'S BYTE-IDENTITY GATE — the first Triton-sourced kernel vs the builder.
//!
//! One rmsnorm node, lowered twice through the SAME door (`ktir_superdsc_door::lower`,
//! under the SAME `BundleLayout` the tape walk computes):
//!
//! 1. the BUILDER path — `lower_graph_to_ktir`'s `KtirFunc::rmsnorm`, the card-proven
//!    status quo;
//! 2. the SPLICE — `scratchy_triton_splice::lower`, which compiles
//!    `crates/targets/spyre/kernels/rmsnorm.py` at expansion time through the re-hosted
//!    Triton ladder and hands back the same `EmittedOp`.
//!
//! The descriptors must be byte-identical. That is the landing gate for every registry
//! row: the builder arm for an op is deleted only when this passes for the shapes in
//! scope, and a row that passes here cannot emit a different descriptor than the one
//! the card has already run.
//!
//! ⛔ DESCRIPTOR LEVEL, NOT KTIR LEVEL. The two producers legitimately spell the program
//! differently (`math.sqrt(mean + eps)` with an f32 divisor vs `rsqrt` with a folded
//! reciprocal) — the consumer assembles descriptors from `regions()` + the program's
//! stated constants, so the op soup never reaches them, and the emitted `SdscOp` JSON is
//! the strongest gate that is not also a false one.
//!
//! Lives in `tests/` so it can name the builder as the control (a private-free public
//! API walk) and the splice (a dev-dependency), without the crate's own feature gates.

use std::collections::HashSet;

use ktir_superdsc::ktir_node::ActiveCap;
use scratchy_subtile::subtile_ir::{
    EwKind, GainConvention, SubOp, SubtileIR, SubtileNode, TensorId, TensorRegion, TensorShape,
};
use scratchy_target_spyre::ktir_superdsc_door::lower as door_lower;
use scratchy_target_spyre::lower_subtile_tape_to_ktir::lower_graph_to_ktir;

/// The rmsnorm shapes that matter for the delivery scope: granite 3.2/3.3 at 2b and 8b
/// both normalize at hidden 2048 (2b) and 4096 (8b), decode rows 1 and a prefill rung's
/// width. (M, D_MODEL).
const SHAPES: &[(u32, u32)] = &[(1, 2048), (1, 4096), (31, 2048), (64, 4096)];

#[test]
fn spliced_rmsnorm_is_byte_identical_to_the_builder() {
    for &(m, c) in SHAPES {
        let ir = rmsnorm_ir(m, c);
        let weight_ids: HashSet<u32> = [0u32, 1u32].into_iter().collect();

        // 1. The builder path — the control.
        let (builder_ops, layout) =
            lower_graph_to_ktir(&ir, &weight_ids, ActiveCap::FULL, false)
                .unwrap_or_else(|e| panic!("builder lowered m={m} c={c}: {e}"));
        let [builder] = &builder_ops[..] else {
            panic!("one rmsnorm node lowers to one op, got {}", builder_ops.len())
        };
        let builder_ktir = builder.ktir.as_ref().expect("builder op carries its program");

        // 2. The splice — the row compiles the kernel for this node.
        let node = &ir.nodes[0];
        let spliced = scratchy_triton_splice::lower(node, &ir, false)
            .unwrap_or_else(|e| panic!("splice compiled m={m} c={c}: {e}"))
            .expect("registry has a row for Scale-gain RmsNorm");

        // ⛔ THE NAME LAW IS PART OF THE GATE. The builder names its program
        // `rmsnorm_s{id}`; the splice reuses the law so the op_name and the emulator's
        // function key are identical.
        assert_eq!(spliced.op_name, builder.op_name, "op_name (m={m} c={c})");

        // 3. Both programs go through the SAME door under the SAME layout — the consumer
        //    is the thing being pinned.
        let mut sym = 0i64;
        let mut quantized = HashSet::new();
        let builder_emitted = door_lower(
            builder_ktir,
            &mut sym,
            Some(&layout),
            &mut quantized,
            None,
        )
        .unwrap_or_else(|e| panic!("builder program lowered (m={m} c={c}): {}", e.message));
        let mut sym = 0i64;
        let spliced_emitted = door_lower(
            spliced.ktir.as_ref().expect("spliced op carries its program"),
            &mut sym,
            Some(&layout),
            &mut quantized,
            None,
        )
        .unwrap_or_else(|e| panic!("spliced program lowered (m={m} c={c}): {}", e.message));

        assert_eq!(
            builder_emitted.len(),
            spliced_emitted.len(),
            "op count (m={m} c={c})"
        );
        for (b, s) in builder_emitted.iter().zip(spliced_emitted.iter()) {
            let bj = serde_json::to_string(b.dsc()).unwrap();
            let sj = serde_json::to_string(s.dsc()).unwrap();
            assert_eq!(
                bj, sj,
                "descriptor bytes (m={m} c={c}): builder vs splice diverged"
            );
            assert_eq!(b.op_name, s.op_name, "emitted op_name (m={m} c={c})");
        }
    }
}

/// `rmsnorm(x, gamma) -> out` as a one-node [`SubtileIR`] — the same fixture shape
/// `superdsc_time_tile.rs` mints for its matmul control. t0 = x source, t1 = gamma
/// source, t2 = result. Eps is granite's `1e-5`.
fn rmsnorm_ir(m: u32, c: u32) -> SubtileIR {
    let tensors = vec![
        TensorShape { rows: m, cols: c },
        TensorShape { rows: 1, cols: c },
        TensorShape { rows: m, cols: c },
    ];
    let whole = |t: usize| TensorRegion {
        tensor: TensorId::from_index(t),
        region: tensors[t].whole(),
    };
    let node = SubtileNode {
        id: scratchy_subtile::subtile_ir::SubtileId::from_index(0),
        op: SubOp::RmsNorm {
            eps: 1e-5,
            gain: GainConvention::Scale,
        },
        inputs: vec![whole(0), whole(1)],
        output: whole(2),
    };
    SubtileIR {
        tensors,
        num_sources: 2,
        nodes: vec![node],
        result: TensorId::from_index(2),
        op_output: Vec::new(),
    }
}

/// The silu-mul shapes that matter for the delivery scope: granite's d_ff (2b: 0, 8b:
/// 12800) at decode rows and a prefill rung's width. (M, N).
const SILUMUL_SHAPES: &[(u32, u32)] = &[(1, 4096), (1, 12800), (31, 4096), (64, 12800)];

#[test]
fn spliced_silumul_is_byte_identical_to_the_builder() {
    for &(m, c) in SILUMUL_SHAPES {
        let ir = silumul_ir(m, c);
        let weight_ids: HashSet<u32> = [0u32, 1u32].into_iter().collect();

        // 1. The builder path — the control.
        let (builder_ops, layout) =
            lower_graph_to_ktir(&ir, &weight_ids, ActiveCap::FULL, false)
                .unwrap_or_else(|e| panic!("builder lowered m={m} c={c}: {e}"));
        let [builder] = &builder_ops[..] else {
            panic!("one silumul node lowers to one op, got {}", builder_ops.len())
        };
        let builder_ktir = builder.ktir.as_ref().expect("builder op carries its program");

        // 2. The splice — the row compiles the kernel for this node.
        let node = &ir.nodes[0];
        let spliced = scratchy_triton_splice::lower(node, &ir, false)
            .unwrap_or_else(|e| panic!("splice compiled m={m} c={c}: {e}"))
            .expect("registry has a row for SiluMul");

        // ⛔ THE NAME LAW IS PART OF THE GATE — `silumul_s{id}` on both paths.
        assert_eq!(spliced.op_name, builder.op_name, "op_name (m={m} c={c})");

        // 3. Both programs go through the SAME door under the SAME layout.
        let mut sym = 0i64;
        let mut quantized = HashSet::new();
        let builder_emitted = door_lower(
            builder_ktir,
            &mut sym,
            Some(&layout),
            &mut quantized,
            None,
        )
        .unwrap_or_else(|e| panic!("builder program lowered (m={m} c={c}): {}", e.message));
        let mut sym = 0i64;
        let spliced_emitted = door_lower(
            spliced.ktir.as_ref().expect("spliced op carries its program"),
            &mut sym,
            Some(&layout),
            &mut quantized,
            None,
        )
        .unwrap_or_else(|e| panic!("spliced program lowered (m={m} c={c}): {}", e.message));

        assert_eq!(
            builder_emitted.len(),
            spliced_emitted.len(),
            "op count (m={m} c={c})"
        );
        for (b, s) in builder_emitted.iter().zip(spliced_emitted.iter()) {
            let bj = serde_json::to_string(b.dsc()).unwrap();
            let sj = serde_json::to_string(s.dsc()).unwrap();
            assert_eq!(
                bj, sj,
                "descriptor bytes (m={m} c={c}): builder vs splice diverged"
            );
            assert_eq!(b.op_name, s.op_name, "emitted op_name (m={m} c={c})");
        }
    }
}

/// `silu(gate) * up -> out` as a one-node [`SubtileIR`]. t0 = gate source, t1 = up
/// source, t2 = result. Both operands are activations, but the builder path is
/// shape-driven and does not read the distinction, so the fixture pins both as sources.
fn silumul_ir(m: u32, c: u32) -> SubtileIR {
    let tensors = vec![
        TensorShape { rows: m, cols: c },
        TensorShape { rows: m, cols: c },
        TensorShape { rows: m, cols: c },
    ];
    let whole = |t: usize| TensorRegion {
        tensor: TensorId::from_index(t),
        region: tensors[t].whole(),
    };
    let node = SubtileNode {
        id: scratchy_subtile::subtile_ir::SubtileId::from_index(0),
        op: SubOp::SiluMul,
        inputs: vec![whole(0), whole(1)],
        output: whole(2),
    };
    SubtileIR {
        tensors,
        num_sources: 2,
        nodes: vec![node],
        result: TensorId::from_index(2),
        op_output: Vec::new(),
    }
}

/// The elementwise shapes that matter for the delivery scope: granite's hidden 2048
/// (the residual adds' width) at decode rows and a prefill rung's width, all inside
/// the splice's LX budget (a blocked region is a builder-only node by the splice's own
/// guard, so it has no splice side to compare). (M, N).
const EW_SHAPES: &[(u32, u32)] = &[(1, 2048), (1, 4096), (31, 2048), (64, 4096)];

/// Every `EwKind` the splice has a row for. The kinds the builder REFUSES
/// (Gelu/QuickGelu/GeluErf) are absent: they have no builder side, so a byte-identity
/// comparison would pin nothing.
const EW_KINDS: &[EwKind] = &[
    EwKind::Add,
    EwKind::BiasAdd,
    EwKind::Mul,
    EwKind::Sub,
    EwKind::Silu,
];

#[test]
fn spliced_elementwise_is_byte_identical_to_the_builder() {
    for &kind in EW_KINDS {
        for &(m, c) in EW_SHAPES {
            let ir = elementwise_ir(m, c, kind);
            let weight_ids: HashSet<u32> = [0u32, 1u32].into_iter().collect();

            // 1. The builder path — the control.
            let (builder_ops, layout) =
                lower_graph_to_ktir(&ir, &weight_ids, ActiveCap::FULL, false)
                    .unwrap_or_else(|e| panic!("builder lowered {kind:?} m={m} c={c}: {e}"));
            let [builder] = &builder_ops[..] else {
                panic!(
                    "one {kind:?} node lowers to one op, got {}",
                    builder_ops.len()
                )
            };
            let builder_ktir = builder.ktir.as_ref().expect("builder op carries its program");

            // 2. The splice — the row compiles the kernel for this node. A region whose
            // live set exceeds the builder's LX budget is a BUILDER-ONLY node (the
            // builder row-blocks it inside one program; a one-tile kernel cannot spell
            // that), and the splice's own guard falls through — pinned here, because a
            // splice that took such a node would emit a program the descriptor-level
            // golden cannot compare and the emulator could not run.
            let node = &ir.nodes[0];
            let spliced = scratchy_triton_splice::lower(node, &ir, false)
                .unwrap_or_else(|e| panic!("splice compiled {kind:?} m={m} c={c}: {e}"));
            let Some(spliced) = spliced else {
                let live: u32 = if matches!(kind, EwKind::Silu) { 6 } else { 3 };
                assert!(
                    m > 1 && u64::from(m) * u64::from(c) * u64::from(live) > 1024 * 1024,
                    "{kind:?} m={m} c={c}: the splice fell through but the region FITS the \
                     builder's LX budget — the registry row is missing or the guard is wrong"
                );
                continue;
            };

            // ⛔ THE NAME LAW IS PART OF THE GATE — the BUILDER's `ew_kind_stem`
            // (`add_s{id}`, `mul_s{id}`, `sub_s{id}`, `silu_s{id}`; BiasAdd is `add`),
            // read off the producer's kind on both paths.
            assert_eq!(spliced.op_name, builder.op_name, "op_name ({kind:?} m={m} c={c})");

            // 3. Both programs go through the SAME door under the SAME layout.
            let mut sym = 0i64;
            let mut quantized = HashSet::new();
            let builder_emitted = door_lower(
                builder_ktir,
                &mut sym,
                Some(&layout),
                &mut quantized,
                None,
            )
            .unwrap_or_else(|e| panic!("builder program lowered ({kind:?} m={m} c={c}): {}", e.message));
            let mut sym = 0i64;
            let spliced_emitted = door_lower(
                spliced.ktir.as_ref().expect("spliced op carries its program"),
                &mut sym,
                Some(&layout),
                &mut quantized,
                None,
            )
            .unwrap_or_else(|e| panic!("spliced program lowered ({kind:?} m={m} c={c}): {}", e.message));

            assert_eq!(
                builder_emitted.len(),
                spliced_emitted.len(),
                "op count ({kind:?} m={m} c={c})"
            );
            for (b, s) in builder_emitted.iter().zip(spliced_emitted.iter()) {
                let bj = serde_json::to_string(b.dsc()).unwrap();
                let sj = serde_json::to_string(s.dsc()).unwrap();
                assert_eq!(
                    bj, sj,
                    "descriptor bytes ({kind:?} m={m} c={c}): builder vs splice diverged"
                );
                assert_eq!(b.op_name, s.op_name, "emitted op_name ({kind:?} m={m} c={c})");
            }
        }
    }
}

/// One elementwise node as a one-node [`SubtileIR`]: unary kinds read t0 and write t2;
/// binary kinds read t0 and t1 and write t2. All operands the output's shape (a
/// broadcast operand is a builder `EwOperand` path this splice deliberately does not
/// state, and the door's own extent guard would refuse it).
fn elementwise_ir(m: u32, c: u32, kind: EwKind) -> SubtileIR {
    let unary = matches!(kind, EwKind::Silu);
    let tensors = vec![
        TensorShape { rows: m, cols: c },
        TensorShape { rows: m, cols: c },
        TensorShape { rows: m, cols: c },
    ];
    let whole = |t: usize| TensorRegion {
        tensor: TensorId::from_index(t),
        region: tensors[t].whole(),
    };
    let inputs = if unary {
        vec![whole(0)]
    } else {
        vec![whole(0), whole(1)]
    };
    let node = SubtileNode {
        id: scratchy_subtile::subtile_ir::SubtileId::from_index(0),
        op: SubOp::Elementwise(kind),
        inputs,
        output: whole(2),
    };
    SubtileIR {
        tensors,
        num_sources: 2,
        nodes: vec![node],
        result: TensorId::from_index(2),
        op_output: Vec::new(),
    }
}

/// The dense fp16 matmul shapes that matter for the delivery scope: granite 2b's hidden
/// 2048 projections at decode and a prefill rung, plus the wide qkv form. (M, K, N).
const MATMUL_SHAPES: &[(u32, u32, u32)] = &[
    (1, 2048, 2048),   // DECODE — the shape every chat token runs
    (1, 2048, 512),    // decode, non-square
    (31, 2048, 2048),  // prefill rung, square (both orientations pass extents)
    (31, 2048, 512),   // NON-SQUARE — the orientation discriminator (a tile of a wide N)
];

#[test]
fn spliced_dense_matmul_is_byte_identical_to_the_builder() {
    for &(m, k, n) in MATMUL_SHAPES {
        let ir = matmul_ir(m, k, n);
        let weight_ids: HashSet<u32> = [0u32, 1u32].into_iter().collect();

        // 1. The builder path — the control. `rows_are_requests: true` disables the
        // prefill lm-head tail fold (the builder's own `!rows_are_requests` guard): this
        // fixture's matmul IS the graph result, so its cols equal `result_cols` and the
        // fold would otherwise rewrite the node instead of lowering it — the same
        // condition the splice's own fallthrough conservatively honors.
        let (builder_ops, layout) =
            lower_graph_to_ktir(&ir, &weight_ids, ActiveCap::FULL, true)
                .unwrap_or_else(|e| panic!("builder lowered m={m} k={k} n={n}: {e}"));
        let [builder] = &builder_ops[..] else {
            panic!("one matmul node lowers to one op, got {}", builder_ops.len())
        };
        let builder_ktir = builder.ktir.as_ref().expect("builder op carries its program");

        // 2. The splice — the row compiles the kernel for this node.
        let node = &ir.nodes[0];
        let spliced = scratchy_triton_splice::lower(node, &ir, true)
            .unwrap_or_else(|e| panic!("splice compiled m={m} k={k} n={n}: {e}"))
            .expect("registry has a row for Dense MatmulTile");

        // ⛔ THE NAME LAW IS PART OF THE GATE — `matmul_s{id}` on both paths.
        assert_eq!(spliced.op_name, builder.op_name, "op_name (m={m} k={k} n={n})");

        // 3. Both programs go through the SAME door under the SAME layout.
        let mut sym = 0i64;
        let mut quantized = HashSet::new();
        let builder_emitted = door_lower(
            builder_ktir,
            &mut sym,
            Some(&layout),
            &mut quantized,
            None,
        )
        .unwrap_or_else(|e| panic!("builder program lowered (m={m} k={k} n={n}): {}", e.message));
        let mut sym = 0i64;
        let spliced_emitted = door_lower(
            spliced.ktir.as_ref().expect("spliced op carries its program"),
            &mut sym,
            Some(&layout),
            &mut quantized,
            None,
        )
        .unwrap_or_else(|e| panic!("spliced program lowered (m={m} k={k} n={n}): {}", e.message));

        assert_eq!(
            builder_emitted.len(),
            spliced_emitted.len(),
            "op count (m={m} k={k} n={n})"
        );
        for (b, s) in builder_emitted.iter().zip(spliced_emitted.iter()) {
            let bj = serde_json::to_string(b.dsc()).unwrap();
            let sj = serde_json::to_string(s.dsc()).unwrap();
            assert_eq!(
                bj, sj,
                "descriptor bytes (m={m} k={k} n={n}): builder vs splice diverged"
            );
            assert_eq!(b.op_name, s.op_name, "emitted op_name (m={m} k={k} n={n})");
        }
    }
}

/// ⛔ THE GOLDEN IS NOT EXECUTION. The byte-identity gate pins the DESCRIPTOR emission
/// through the door, but the E2E serving path (`spyre-emu`) executes the spliced KTIR
/// programs themselves through the emulator's fused/resident path — a different consumer
/// with its own rewrite set (the `matmul_tile` LX re-tiling among them). This test drives
/// the REAL spliced program — `scratchy_triton_splice::lower`'s own output, never a
/// hand-built copy — through the production session entry (`SpyreSession::new_multi` /
/// `run_step`, the same `build_spec` walk the worker's session takes) and checks the
/// numbers against a host reference. The rmsnorm and silumul rows pass both gates; a
/// matmul row that passes the golden but fails here is exactly the divergence this
/// catches.
#[test]
#[cfg(feature = "spyre-emu")]
fn spliced_dense_matmul_executes_the_real_program() {
    // DECODE (m=1, the shape every chat token runs) and a PREFILL rung (m=31), both
    // NON-SQUARE on purpose: a square `[k, k]` W satisfies both orientation readings,
    // so it cannot catch a transposed read.
    for (m, k, n) in [(1u32, 2048u32, 512u32), (31u32, 2048u32, 512u32)] {
        execute_one_spliced_matmul(m, k, n);
    }
}

/// One spliced dense matmul through the production session entry, checked against a
/// host reference over the SAME on-disk `[n, k]` weight bytes the worker binds.
fn execute_one_spliced_matmul(m: u32, k: u32, n: u32) {
    use std::borrow::Cow;

    let ir = matmul_ir(m, k, n);
    let node = &ir.nodes[0];
    let spliced = scratchy_triton_splice::lower(node, &ir, true)
        .unwrap_or_else(|e| panic!("splice compiled m={m} k={k} n={n}: {e}"))
        .expect("registry has a row for Dense MatmulTile");
    let k_node = spliced.ktir.as_ref().expect("spliced op carries its program");

    // The launch binding: parameter `i` -> `bindings[i]`, exactly the pairing
    // `build_spec` reads off `LaunchProgram::args` — the tape's own numbering.
    let args: Vec<(ktir_core::ir::Ssa, scratchy_target_spyre::bundle_code::PlaceId)> = k_node
        .func
        .arguments
        .iter()
        .map(|(ssa, _ty)| (*ssa, scratchy_target_spyre::bundle_code::PlaceId::Act(k_node.bindings[ssa.slot()].get())))
        .collect();
    let group = scratchy_target_spyre::bundle_code::LaunchGroup {
        kv: Default::default(),
        programs: Cow::Owned(vec![scratchy_target_spyre::bundle_code::LaunchProgram {
            func: k_node.func,
            args: Cow::Owned(args),
        }]),
        init_binary: Cow::Borrowed(&[]),
        job_bin_ptr: 0,
        correction: Cow::Borrowed(&[]),
    };

    // The host data, bound EXACTLY as the worker binds it (`spyre_load.rs`'s
    // non-hw arm): A `[m, k]` and the GEMM weight VERBATIM in its on-disk `[n, k]`
    // orientation — the same bytes the builder's transpose-B maps read. The f32
    // reference is over that same buffer: `W[ni, ki]` at `ni * k + ki`.
    let a: Vec<f32> = (0..m * k).map(|i| ((i % 13) as f32) * 0.01 - 0.06).collect();
    let w: Vec<f32> = (0..k * n).map(|i| ((i % 17) as f32) * 0.02 - 0.16).collect();
    let mut want = vec![0.0f32; (m * n) as usize];
    for mi in 0..m {
        for ni in 0..n {
            let mut acc = 0.0f32;
            for ki in 0..k {
                acc += a[(mi * k + ki) as usize] * w[(ni * k + ki) as usize];
            }
            want[(mi * n + ni) as usize] = acc;
        }
    }

    let mut session = scratchy_target_spyre::runner::SpyreSession::new_multi(
        &[(&[group], &[2u64])],
        Vec::new(),
    )
    .expect("build the one-program session");
    let out = session
        .run_step(
            0,
            vec![
                (0, a, vec![m as usize, k as usize]),
                (1, w, vec![n as usize, k as usize]),
            ],
            &[(2, 0)],
        )
        .expect("run the spliced matmul program");
    let got = &out[&2];
    assert_eq!(got.len(), (m * n) as usize);
    let mut max_abs = 0.0f32;
    for (g, wnt) in got.iter().zip(&want) {
        max_abs = max_abs.max((g - wnt).abs());
    }
    assert!(
        max_abs < 0.05,
        "the REAL spliced matmul program diverged from the host reference: max abs err {max_abs}"
    );
}

/// The elementwise rows' EXECUTION gate — same calibration as the matmul one above: the
/// REAL spliced program through the production session entry, against a host reference.
/// Every kind the splice has a row for, at a decode shape and a prefill rung.
#[test]
#[cfg(feature = "spyre-emu")]
fn spliced_elementwise_executes_the_real_program() {
    for &kind in EW_KINDS {
        for (m, c) in [(1u32, 2048u32), (31u32, 2048u32)] {
            // Skip the builder-only combinations (the LX-budget fallthrough, pinned by
            // the golden above).
            let live: u32 = if matches!(kind, EwKind::Silu) { 6 } else { 3 };
            if m > 1 && u64::from(m) * u64::from(c) * u64::from(live) > 1024 * 1024 {
                continue;
            }
            execute_one_spliced_elementwise(m, c, kind);
        }
    }
}

/// One spliced elementwise program through the production session entry, checked
/// against a host reference.
fn execute_one_spliced_elementwise(m: u32, c: u32, kind: EwKind) {
    use std::borrow::Cow;

    let ir = elementwise_ir(m, c, kind);
    let node = &ir.nodes[0];
    let spliced = scratchy_triton_splice::lower(node, &ir, false)
        .unwrap_or_else(|e| panic!("splice compiled {kind:?} m={m} c={c}: {e}"))
        .unwrap_or_else(|| panic!("registry has a row for {kind:?}"));
    let k_node = spliced.ktir.as_ref().expect("spliced op carries its program");

    let args: Vec<(ktir_core::ir::Ssa, scratchy_target_spyre::bundle_code::PlaceId)> = k_node
        .func
        .arguments
        .iter()
        .map(|(ssa, _ty)| (*ssa, scratchy_target_spyre::bundle_code::PlaceId::Act(k_node.bindings[ssa.slot()].get())))
        .collect();
    let group = scratchy_target_spyre::bundle_code::LaunchGroup {
        kv: Default::default(),
        programs: Cow::Owned(vec![scratchy_target_spyre::bundle_code::LaunchProgram {
            func: k_node.func,
            args: Cow::Owned(args),
        }]),
        init_binary: Cow::Borrowed(&[]),
        job_bin_ptr: 0,
        correction: Cow::Borrowed(&[]),
    };

    // Two operands and the host reference over them, per kind — the same f16 values
    // the program reads (run_step narrows to f16).
    let n = (m * c) as usize;
    let a: Vec<f32> = (0..n).map(|i| ((i % 13) as f32) * 0.01 - 0.06).collect();
    let b: Vec<f32> = (0..n).map(|i| ((i % 17) as f32) * 0.02 - 0.16).collect();
    let want: Vec<f32> = match kind {
        EwKind::Add | EwKind::BiasAdd => a.iter().zip(&b).map(|(x, y)| x + y).collect(),
        EwKind::Mul => a.iter().zip(&b).map(|(x, y)| x * y).collect(),
        EwKind::Sub => a.iter().zip(&b).map(|(x, y)| x - y).collect(),
        EwKind::Silu => a.iter().map(|&x| x / (1.0 + (-x).exp())).collect(),
        other => unreachable!("exec fixture for {other:?}"),
    };

    let mut session = scratchy_target_spyre::runner::SpyreSession::new_multi(
        &[(&[group], &[2u64])],
        Vec::new(),
    )
    .expect("build the one-program session");
    let sources = if matches!(kind, EwKind::Silu) {
        vec![(0u64, a, vec![m as usize, c as usize])]
    } else {
        vec![
            (0u64, a, vec![m as usize, c as usize]),
            (1u64, b, vec![m as usize, c as usize]),
        ]
    };
    let out = session
        .run_step(0, sources, &[(2, 0)])
        .unwrap_or_else(|_| panic!("run the spliced {kind:?} program"));
    let got = &out[&2];
    assert_eq!(got.len(), n, "{kind:?} m={m} c={c}");
    let mut max_abs = 0.0f32;
    for (g, w) in got.iter().zip(&want) {
        max_abs = max_abs.max((g - w).abs());
    }
    assert!(
        max_abs < 0.05,
        "the REAL spliced {kind:?} program diverged from the host reference: max abs err {max_abs}"
    );
}

/// `hidden[m, k] @ W[k, n] -> out[m, n]` as a one-node [`SubtileIR`], dense weights —
/// the same fixture shape `superdsc_time_tile.rs`'s `single_matmul_ir` mints.
fn matmul_ir(m: u32, k: u32, n: u32) -> SubtileIR {
    let tensors = vec![
        TensorShape { rows: m, cols: k },
        TensorShape { rows: k, cols: n },
        TensorShape { rows: m, cols: n },
    ];
    let whole = |t: usize| TensorRegion {
        tensor: TensorId::from_index(t),
        region: tensors[t].whole(),
    };
    let node = SubtileNode {
        id: scratchy_subtile::subtile_ir::SubtileId::from_index(0),
        op: SubOp::MatmulTile {
            n,
            weight: scratchy_subtile::lower::GemmWeight::Dense,
        },
        inputs: vec![whole(0), whole(1)],
        output: whole(2),
    };
    SubtileIR {
        tensors,
        num_sources: 2,
        nodes: vec![node],
        result: TensorId::from_index(2),
        op_output: Vec::new(),
    }
}
