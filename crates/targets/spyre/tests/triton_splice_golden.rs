// SPDX-License-Identifier: Apache-2.0
//! THE SPLICE'S BYTE-IDENTITY GATE — the Triton-sourced kernel vs the builder.
//!
//! One node, lowered twice through the SAME door (`ktir_superdsc_door::lower`,
//! under the SAME `BundleLayout` the tape walk computes):
//!
//! 1. the BUILDER path — the card-proven `lower_*_node` body (`KtirFunc`'s own
//!    program), called DIRECTLY as the control;
//! 2. the SPLICE — `scratchy_triton_splice::lower`, which compiles
//!    `crates/targets/spyre/kernels/<row>.py` at expansion time through the
//!    re-hosted Triton ladder and hands back the same `EmittedOp`.
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
//! ⛔ THE CONTROL IS THE BUILDER BODY, NOT THE WALK. `lower_one_node` dispatches every
//! splicable kind through the splice, so lowering a one-node graph through
//! `lower_graph_to_ktir` would compare the splice with itself. The control therefore
//! calls the card-proven builder functions directly (`lower_rmsnorm_node` and kin —
//! the bodies the splice replaced, kept as the controls this gate needs) under a layout
//! minted by the same `compute_bundle_layout` the walk uses.
//!
//! Lives in `tests/` so it can name the builder as the control (a public API walk)
//! and the splice (a dev-dependency), without the crate's own feature gates.

use std::collections::HashSet;

use scratchy_subtile::subtile_ir::{
    EwKind, GainConvention, Range, Region, SubOp, SubtileIR, SubtileNode, TensorId, TensorRegion,
    TensorShape,
};
use scratchy_target_spyre::ktir_superdsc_door::lower as door_lower;
use scratchy_target_spyre::lower_subtile_tape_to_ktir::{
    lower_elementwise_node, lower_matmul_node, lower_rmsnorm_node, lower_scalarmul_node,
    lower_silumul_node, LowerRope,
};
use scratchy_target_spyre::lower_subtile_tape_to_superdsc::compute_bundle_layout;

/// The rmsnorm shapes that matter for the delivery scope: granite 3.2/3.3 at 2b and 8b
/// both normalize at hidden 2048 (2b) and 4096 (8b), decode rows 1 and a prefill rung's
/// width. (M, D_MODEL). The kernel widens the squares to f32 before its reduce (the
/// accumulator's own precision, matching the builder's program), so every width is
/// splicable and byte-compared.
const SHAPES: &[(u32, u32)] = &[(1, 2048), (1, 4096), (31, 2048), (64, 4096)];

#[test]
fn spliced_rmsnorm_is_byte_identical_to_the_builder() {
    for &(m, c) in SHAPES {
        let ir = rmsnorm_ir(m, c);
        let weight_ids: HashSet<u32> = [0u32, 1u32].into_iter().collect();

        // 1. The builder path — the control, called directly.
        let layout = compute_bundle_layout(&ir, &weight_ids, false, &Default::default())
            .unwrap_or_else(|e| panic!("layout minted m={m} c={c}: {e}"));
        let mut sym = 0i64;
        let builder_ops = lower_rmsnorm_node(&ir.nodes[0], &ir, 1e-5, &mut sym)
            .unwrap_or_else(|e| panic!("builder lowered m={m} c={c}: {e}"));
        let [builder] = &builder_ops[..] else {
            panic!(
                "one rmsnorm node lowers to one op, got {}",
                builder_ops.len()
            )
        };
        let builder_ktir = builder
            .ktir
            .as_ref()
            .expect("builder op carries its program");

        // 2. The splice — the row compiles the kernel for this node. Every width is
        // splicable (the kernel's f32 accumulator), so an Err here is a broken row.
        let node = &ir.nodes[0];
        let spliced = scratchy_triton_splice::lower(node, &ir, false)
            .unwrap_or_else(|e| panic!("splice compiled m={m} c={c}: {e}"));

        // ⛔ THE NAME LAW IS PART OF THE GATE. The builder names its program
        // `rmsnorm_s{id}`; the splice reuses the law so the op_name and the emulator's
        // function key are identical.
        assert_eq!(spliced.op_name, builder.op_name, "op_name (m={m} c={c})");

        // 3. Both programs go through the SAME door under the SAME layout — the consumer
        //    is the thing being pinned.
        let mut sym = 0i64;
        let mut quantized = HashSet::new();
        let builder_emitted =
            door_lower(builder_ktir, &mut sym, Some(&layout), &mut quantized, None)
                .unwrap_or_else(|e| panic!("builder program lowered (m={m} c={c}): {}", e.message));
        let mut sym = 0i64;
        let spliced_emitted = door_lower(
            spliced
                .ktir
                .as_ref()
                .expect("spliced op carries its program"),
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
/// 12800) at decode rows and a prefill rung's width. (M, N). ⛔ THE `[64, 12800]` AND
/// `[96, 4096]` RUNGS ARE BUILDER-ONLY (the eight-live-tile LX budget the splice's
/// guard mirrors), so they pin the GUARD's loud Err, not a comparison.
const SILUMUL_SHAPES: &[(u32, u32)] = &[(1, 4096), (1, 12800), (31, 4096), (64, 12800), (96, 4096)];

#[test]
fn spliced_silumul_is_byte_identical_to_the_builder() {
    for &(m, c) in SILUMUL_SHAPES {
        let ir = silumul_ir(m, c);
        let weight_ids: HashSet<u32> = [0u32, 1u32].into_iter().collect();

        // 1. The builder path — the control, called directly.
        let layout = compute_bundle_layout(&ir, &weight_ids, false, &Default::default())
            .unwrap_or_else(|e| panic!("layout minted m={m} c={c}: {e}"));
        let mut sym = 0i64;
        let builder_ops = lower_silumul_node(&ir.nodes[0], &ir, &mut sym, Some(&layout))
            .unwrap_or_else(|e| panic!("builder lowered m={m} c={c}: {e}"));
        let [builder] = &builder_ops[..] else {
            panic!(
                "one silumul node lowers to one op, got {}",
                builder_ops.len()
            )
        };
        let builder_ktir = builder
            .ktir
            .as_ref()
            .expect("builder op carries its program");

        // 2. The splice — the row compiles the kernel for this node. A region whose
        // eight-tile live set exceeds the LX budget is a BUILDER-ONLY node (the guard
        // the granite-8b `[31, 12800]` overflow measured): the splice's loud Err names
        // the missing row-blocked kernel, pinned here so the Err cannot silently
        // become a wrong whole-region program.
        let node = &ir.nodes[0];
        let spliced = scratchy_triton_splice::lower(node, &ir, false);
        let spliced = match spliced {
            Ok(s) => s,
            Err(reason) => {
                assert!(
                    m > 1 && u64::from(m) * u64::from(c) * 8 > 1024 * 1024,
                    "m={m} c={c}: the splice refused but the region FITS the eight-tile \
                     LX budget — the registry row is missing or the guard is wrong ({reason})"
                );
                continue;
            }
        };

        // ⛔ THE NAME LAW IS PART OF THE GATE — `silumul_s{id}` on both paths.
        assert_eq!(spliced.op_name, builder.op_name, "op_name (m={m} c={c})");

        // 3. Both programs go through the SAME door under the SAME layout.
        let mut sym = 0i64;
        let mut quantized = HashSet::new();
        let builder_emitted =
            door_lower(builder_ktir, &mut sym, Some(&layout), &mut quantized, None)
                .unwrap_or_else(|e| panic!("builder program lowered (m={m} c={c}): {}", e.message));
        let mut sym = 0i64;
        let spliced_emitted = door_lower(
            spliced
                .ktir
                .as_ref()
                .expect("spliced op carries its program"),
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

/// ⛔ THE COLUMN-CHUNKED NODE IS A LOUD ERR — the 8b defect's own gate. The front end
/// tiles a wide pointwise op into column chunks (production `nb = 8192`; granite 8b's
/// 12800-wide MLP silumul is chunks `0..8192` and `8192..12800`), and the one-tile
/// kernels state ONE whole-tensor tile at corner 0 — a windowed load is not expressible
/// in them. The splice must REFUSE BY NAME (never fall through and never mis-address),
/// and this pins exactly that: a chunk-shaped node returns the whole-region Err, while
/// the whole-tensor node of the same width still splices (the two paths are
/// discriminated by the REGION, never by the width).
///
/// The windowed-kernel family (access-tile corners stated from the region) replaces
/// this Err; when it lands, this test becomes the chunked identity test.
#[test]
fn a_column_chunked_silumul_is_a_loud_refusal() {
    // The CHUNK-1 shape, measured on the card: 12800-wide intermediate, second block.
    let ir = silumul_ir(1, 12800);
    let chunk = Range::new(8192, 12800 - 8192);
    let window = |t: usize| TensorRegion {
        tensor: TensorId::from_index(t),
        region: Region {
            rows: Range::new(0, 1),
            cols: chunk,
        },
    };
    let mut chunked = ir.clone();
    chunked.nodes[0].inputs = vec![window(0), window(1)];
    chunked.nodes[0].output = window(2);

    let err = match scratchy_triton_splice::lower(&chunked.nodes[0], &chunked, false) {
        Err(reason) => reason,
        Ok(_) => panic!("a column-chunked silumul must refuse loudly, not compile"),
    };
    assert!(
        err.contains("windowed region"),
        "the chunked refusal must name the windowed-region cause: {err}"
    );

    // THE CONTROL: the whole-tensor node at the same total width still splices — the
    // refusal is the REGION's, not the width's.
    let whole_node = &ir.nodes[0];
    let spliced = scratchy_triton_splice::lower(whole_node, &ir, false)
        .expect("the whole-tensor node compiles");
    assert_eq!(
        spliced.op_name,
        format!("silumul_s{}", whole_node.id.index()),
        "a whole-tensor 12800-wide silumul is inside the splice's reach — the registry row \
         must take it (only its prefill LX-budget rows refuse)"
    );
}

/// The elementwise shapes that matter for the delivery scope: granite's hidden 2048
/// (the residual adds' width) at decode rows and a prefill rung's width, all inside
/// the splice's LX budget (a blocked region is a builder-only node by the splice's own
/// guard, so it has no splice side to compare). (M, N).
const EW_SHAPES: &[(u32, u32)] = &[(1, 2048), (1, 4096), (31, 2048), (64, 4096), (96, 4096)];

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

            // 1. The builder path — the control, called directly.
            let layout = compute_bundle_layout(&ir, &weight_ids, false, &Default::default())
                .unwrap_or_else(|e| panic!("layout minted {kind:?} m={m} c={c}: {e}"));
            let mut sym = 0i64;
            let builder = lower_elementwise_node(&ir.nodes[0], &ir, kind, &mut sym, Some(&layout))
                .unwrap_or_else(|e| panic!("builder lowered {kind:?} m={m} c={c}: {e}"));
            let builder_ktir = builder
                .ktir
                .as_ref()
                .expect("builder op carries its program");

            // 2. The splice — the row compiles the kernel for this node. A region whose
            // live set exceeds the LX budget is a BUILDER-ONLY node (the builder
            // row-blocks it inside one program; a one-tile kernel cannot spell that),
            // and the splice's own loud Err names the missing row-blocked kernel —
            // pinned here, because a splice that took such a node would emit a program
            // the descriptor-level golden cannot compare and the emulator could not run.
            let node = &ir.nodes[0];
            let spliced = scratchy_triton_splice::lower(node, &ir, false);
            let spliced = match spliced {
                Ok(s) => s,
                Err(reason) => {
                    let live: u32 = if matches!(kind, EwKind::Silu) { 6 } else { 3 };
                    assert!(
                        m > 1 && u64::from(m) * u64::from(c) * u64::from(live) > 1024 * 1024,
                        "{kind:?} m={m} c={c}: the splice refused but the region FITS the \
                         builder's LX budget — the registry row is missing or the guard is \
                         wrong ({reason})"
                    );
                    continue;
                }
            };

            // ⛔ THE NAME LAW IS PART OF THE GATE — the BUILDER's `ew_kind_stem`
            // (`add_s{id}`, `mul_s{id}`, `sub_s{id}`, `silu_s{id}`; BiasAdd is `add`),
            // read off the producer's kind on both paths.
            assert_eq!(
                spliced.op_name, builder.op_name,
                "op_name ({kind:?} m={m} c={c})"
            );

            // 3. Both programs go through the SAME door under the SAME layout.
            let mut sym = 0i64;
            let mut quantized = HashSet::new();
            let builder_emitted =
                door_lower(builder_ktir, &mut sym, Some(&layout), &mut quantized, None)
                    .unwrap_or_else(|e| {
                        panic!(
                            "builder program lowered ({kind:?} m={m} c={c}): {}",
                            e.message
                        )
                    });
            let mut sym = 0i64;
            let spliced_emitted = door_lower(
                spliced
                    .ktir
                    .as_ref()
                    .expect("spliced op carries its program"),
                &mut sym,
                Some(&layout),
                &mut quantized,
                None,
            )
            .unwrap_or_else(|e| {
                panic!(
                    "spliced program lowered ({kind:?} m={m} c={c}): {}",
                    e.message
                )
            });

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
                assert_eq!(
                    b.op_name, s.op_name,
                    "emitted op_name ({kind:?} m={m} c={c})"
                );
            }
        }
    }
}

/// One elementwise node as a one-node [`SubtileIR`]: unary kinds read t0 and write t2;
/// binary kinds read t0 and t1 and write t2. All operands the output's shape (a
/// broadcast operand is a builder `EwOperand` path this splice deliberately does not
/// state, and the door's own extent guard would refuse it).
fn elementwise_ir(m: u32, c: u32, kind: EwKind) -> SubtileIR {
    let unary = matches!(kind, EwKind::Silu | EwKind::Gelu);
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
    (1, 2048, 2048),  // DECODE — the shape every chat token runs
    (1, 2048, 512),   // decode, non-square
    (31, 2048, 2048), // prefill rung, square (both orientations pass extents)
    (31, 2048, 512),  // NON-SQUARE — the orientation discriminator (a tile of a wide N)
];

#[test]
fn spliced_dense_matmul_is_byte_identical_to_the_builder() {
    for &(m, k, n) in MATMUL_SHAPES {
        let ir = matmul_ir(m, k, n);
        let weight_ids: HashSet<u32> = [0u32, 1u32].into_iter().collect();

        // 1. The builder path — the control, called directly. `rows_are_requests: true`
        // is the decode/batched shape (the prefill lm-head tail fold is a walk-level
        // rewrite this direct control deliberately bypasses).
        let layout = compute_bundle_layout(&ir, &weight_ids, true, &Default::default())
            .unwrap_or_else(|e| panic!("layout minted m={m} k={k} n={n}: {e}"));
        let mut sym = 0i64;
        let mut quantized = HashSet::new();
        let builder_ops = lower_matmul_node(
            &ir.nodes[0],
            &ir,
            None,
            &mut sym,
            Some(&layout),
            &mut quantized,
        )
        .unwrap_or_else(|e| panic!("builder lowered m={m} k={k} n={n}: {e}"));
        let [builder] = &builder_ops[..] else {
            panic!(
                "one matmul node lowers to one op, got {}",
                builder_ops.len()
            )
        };
        let builder_ktir = builder
            .ktir
            .as_ref()
            .expect("builder op carries its program");

        // 2. The splice — the row compiles the kernel for this node.
        let node = &ir.nodes[0];
        let spliced = scratchy_triton_splice::lower(node, &ir, true)
            .unwrap_or_else(|e| panic!("splice compiled m={m} k={k} n={n}: {e}"));

        // ⛔ THE NAME LAW IS PART OF THE GATE — `matmul_s{id}` on both paths.
        assert_eq!(
            spliced.op_name, builder.op_name,
            "op_name (m={m} k={k} n={n})"
        );

        // 3. Both programs go through the SAME door under the SAME layout.
        let mut sym = 0i64;
        let mut quantized = HashSet::new();
        let builder_emitted =
            door_lower(builder_ktir, &mut sym, Some(&layout), &mut quantized, None).unwrap_or_else(
                |e| panic!("builder program lowered (m={m} k={k} n={n}): {}", e.message),
            );
        let mut sym = 0i64;
        let spliced_emitted = door_lower(
            spliced
                .ktir
                .as_ref()
                .expect("spliced op carries its program"),
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
        .unwrap_or_else(|e| panic!("splice compiled m={m} k={k} n={n}: {e}"));
    let k_node = spliced
        .ktir
        .as_ref()
        .expect("spliced op carries its program");

    // The launch binding: parameter `i` -> `bindings[i]`, exactly the pairing
    // `build_spec` reads off `LaunchProgram::args` — the tape's own numbering.
    let args: Vec<(
        ktir_core::ir::Ssa,
        scratchy_target_spyre::bundle_code::PlaceId,
    )> = k_node
        .func
        .arguments
        .iter()
        .map(|(ssa, _ty)| {
            (
                *ssa,
                scratchy_target_spyre::bundle_code::PlaceId::Act(k_node.bindings[ssa.slot()].get()),
            )
        })
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
    let a: Vec<f32> = (0..m * k)
        .map(|i| ((i % 13) as f32) * 0.01 - 0.06)
        .collect();
    let w: Vec<f32> = (0..k * n)
        .map(|i| ((i % 17) as f32) * 0.02 - 0.16)
        .collect();
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

    let mut session =
        scratchy_target_spyre::runner::SpyreSession::new_multi(&[(&[group], &[2u64])], Vec::new())
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
            // Skip the builder-only combinations (the LX-budget refusal, pinned by
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
        .unwrap_or_else(|e| panic!("splice compiled {kind:?} m={m} c={c}: {e}"));
    let k_node = spliced
        .ktir
        .as_ref()
        .expect("spliced op carries its program");

    let args: Vec<(
        ktir_core::ir::Ssa,
        scratchy_target_spyre::bundle_code::PlaceId,
    )> = k_node
        .func
        .arguments
        .iter()
        .map(|(ssa, _ty)| {
            (
                *ssa,
                scratchy_target_spyre::bundle_code::PlaceId::Act(k_node.bindings[ssa.slot()].get()),
            )
        })
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

    let mut session =
        scratchy_target_spyre::runner::SpyreSession::new_multi(&[(&[group], &[2u64])], Vec::new())
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

/// The gelu row's EXECUTION gate. ⛔ NO BUILDER SIDE, SO NO BYTE-IDENTITY GATE: the
/// builder's elementwise arm refuses Gelu (main 8362-8432 has no `gelu` op), so the
/// only control is arithmetic — the tanh-form gelu over the same f16 values the
/// program reads. The kernel spells the sigmoid identity
/// `gelu(x) = x/(1+exp(−2√(2/π)(x+0.044715x³)))` (the frontend has no `tanh`), which
/// is the tanh form to f16 rounding, so the tolerance is one f16 ulp-scale.
#[test]
#[cfg(feature = "spyre-emu")]
fn spliced_gelu_executes_the_real_program() {
    for (m, c) in [(1u32, 2048u32), (31u32, 2048u32)] {
        execute_one_spliced_gelu(m, c);
    }
}

/// One spliced gelu program through the production session entry, checked against the
/// tanh-form host reference.
fn execute_one_spliced_gelu(m: u32, c: u32) {
    use std::borrow::Cow;

    let ir = elementwise_ir(m, c, EwKind::Gelu);
    let node = &ir.nodes[0];
    let spliced = scratchy_triton_splice::lower(node, &ir, false)
        .unwrap_or_else(|e| panic!("splice compiled gelu m={m} c={c}: {e}"));
    let k_node = spliced
        .ktir
        .as_ref()
        .expect("spliced op carries its program");

    let args: Vec<(
        ktir_core::ir::Ssa,
        scratchy_target_spyre::bundle_code::PlaceId,
    )> = k_node
        .func
        .arguments
        .iter()
        .map(|(ssa, _ty)| {
            (
                *ssa,
                scratchy_target_spyre::bundle_code::PlaceId::Act(k_node.bindings[ssa.slot()].get()),
            )
        })
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

    let n = (m * c) as usize;
    let a: Vec<f32> = (0..n)
        .map(|i| ((i % 23) as f32) * 0.02 - 0.22)
        .collect();
    // The tanh form over the same values, in f32 — the reference the kernel's sigmoid
    // identity must reproduce to f16 scale.
    let want: Vec<f32> = a
        .iter()
        .map(|&x| {
            let v = 0.797_884_6 * (x + 0.044_715 * x * x * x);
            0.5 * x * (1.0 + v.tanh())
        })
        .collect();

    let mut session =
        scratchy_target_spyre::runner::SpyreSession::new_multi(&[(&[group], &[2u64])], Vec::new())
            .expect("build the one-program session");
    let out = session
        .run_step(0, vec![(0u64, a, vec![m as usize, c as usize])], &[(2, 0)])
        .unwrap_or_else(|_| panic!("run the spliced gelu program (m={m} c={c})"));
    let got = &out[&2];
    assert_eq!(got.len(), n, "gelu m={m} c={c}");
    let mut max_abs = 0.0f32;
    for (g, w) in got.iter().zip(&want) {
        max_abs = max_abs.max((g - w).abs());
    }
    assert!(
        max_abs < 0.05,
        "the REAL spliced gelu program diverged from the tanh-form reference (m={m} \
         c={c}): max abs err {max_abs}"
    );
}

/// The scalarmul row's byte-identity gate — the same law as every other row: the
/// builder's `lower_scalarmul_node` program vs the spliced `scalarmul.py` through the
/// SAME door under the SAME layout, at granite's own multipliers (the logits scale's
/// `1/√hidden` at 2b/8b widths) and a decode + prefill-rung shape. ⛔ THE CHUNKED
/// FORM IS NOT HERE: a windowed region is a loud splice Err (pinned by the silumul
/// chunk test above), and the whole-tensor shapes at these widths fit the
/// three-live-tile budget.
#[test]
fn spliced_scalarmul_is_byte_identical_to_the_builder() {
    // (M, C, SCALE) — 2b's and 8b's logits scale at the hidden each normalizes to.
    for (m, c, scale) in [
        (1u32, 2048u32, 0.022_097_087f32),
        (1u32, 4096u32, 0.015_625f32),
        (31u32, 2048u32, 0.022_097_087f32),
    ] {
        let ir = scalarmul_ir(m, c, scale);
        let weight_ids: HashSet<u32> = [0u32].into_iter().collect();

        // 1. The builder path — the control, called directly.
        let layout = compute_bundle_layout(&ir, &weight_ids, false, &Default::default())
            .unwrap_or_else(|e| panic!("layout minted m={m} c={c}: {e}"));
        let mut sym = 0i64;
        let builder = lower_scalarmul_node(&ir.nodes[0], &ir, scale, &mut sym)
            .unwrap_or_else(|e| panic!("builder lowered m={m} c={c}: {e}"));
        let builder_ktir = builder
            .ktir
            .as_ref()
            .expect("builder op carries its program");

        // 2. The splice — the row compiles `scalarmul.py` for this node.
        let node = &ir.nodes[0];
        let spliced = scratchy_triton_splice::lower(node, &ir, false)
            .unwrap_or_else(|e| panic!("splice compiled m={m} c={c}: {e}"));

        // ⛔ THE NAME LAW — `scalarmul_s{id}` on both paths.
        assert_eq!(
            spliced.op_name, builder.op_name,
            "op_name (m={m} c={c} scale={scale})"
        );

        // 3. Both programs go through the SAME door under the SAME layout. ⛔ THE
        // SCALE MUST HAVE A REGISTRY SLOT on both paths — the door reads it off the
        // program (`program_scalarmul_scale`) and looks it up in
        // `BundleLayout::scalarmul_scales`, which `compute_bundle_layout` fills from
        // the node payload, so a splice that states a different scale than the node's
        // fails HERE by registry desync rather than by a wrong number on the card.
        let mut sym = 0i64;
        let mut quantized = HashSet::new();
        let builder_emitted =
            door_lower(builder_ktir, &mut sym, Some(&layout), &mut quantized, None)
                .unwrap_or_else(|e| panic!("builder program lowered (m={m} c={c}): {}", e.message));
        let mut sym = 0i64;
        let spliced_emitted = door_lower(
            spliced
                .ktir
                .as_ref()
                .expect("spliced op carries its program"),
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

/// `x * scale -> out` as a one-node [`SubtileIR`]. t0 = x source, t1 = result.
fn scalarmul_ir(m: u32, c: u32, scale: f32) -> SubtileIR {
    let tensors = vec![
        TensorShape { rows: m, cols: c },
        TensorShape { rows: m, cols: c },
    ];
    let whole = |t: usize| TensorRegion {
        tensor: TensorId::from_index(t),
        region: tensors[t].whole(),
    };
    let node = SubtileNode {
        id: scratchy_subtile::subtile_ir::SubtileId::from_index(0),
        op: SubOp::ScalarMul { scale },
        inputs: vec![whole(0)],
        output: whole(1),
    };
    SubtileIR {
        tensors,
        num_sources: 1,
        nodes: vec![node],
        result: TensorId::from_index(1),
        op_output: Vec::new(),
    }
}

/// The rope row's EXECUTION gate — same calibration as the matmul one above: the REAL
/// spliced program through the production session entry, against a host reference over
/// the SAME worker-staged bytes. ⛔ THE ORIENTATION LAW, AND ROPE HAS TWO OF THEM: the
/// golden cannot catch a wrong-but-legal binding, so this test stages cos/sin the way
/// `spyre_forward.rs`'s `tile` does (per-position `rope_cos_sin` rows replicated across
/// heads, token-major `[mq*heads, hd]`), and a kernel that read the head-major nest or
/// a position-indexed table would come out with wrong numbers here, not at the golden.
#[test]
#[cfg(feature = "spyre-emu")]
fn spliced_rope_executes_the_real_program() {
    // DECODE (mq=1, the shape every chat token runs, hd 64 = 2b's q plane) and a
    // PREFILL rung (mq=31, hd 128 = 8b's slab form) — both with heads > 1, because a
    // head-count of 1 cannot discriminate the token-major nest from the head-major one.
    for (mq, heads, hd) in [(1u32, 32u32, 64u32), (31u32, 8u32, 128u32)] {
        execute_one_spliced_rope(mq, heads, hd);
    }
}

/// One spliced rope program through the production session entry, checked against a
/// host NeoX reference over the worker-staged bytes.
fn execute_one_spliced_rope(mq: u32, heads: u32, hd: u32) {
    use std::borrow::Cow;

    let ir = rope_ir(mq, heads, hd);
    let node = &ir.nodes[0];
    let spliced = scratchy_triton_splice::lower(node, &ir, false)
        .unwrap_or_else(|e| panic!("splice compiled mq={mq} heads={heads} hd={hd}: {e}"));
    let k_node = spliced
        .ktir
        .as_ref()
        .expect("spliced op carries its program");

    let args: Vec<(
        ktir_core::ir::Ssa,
        scratchy_target_spyre::bundle_code::PlaceId,
    )> = k_node
        .func
        .arguments
        .iter()
        .map(|(ssa, _ty)| {
            (
                *ssa,
                scratchy_target_spyre::bundle_code::PlaceId::Act(k_node.bindings[ssa.slot()].get()),
            )
        })
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

    // The host data, staged EXACTLY as the worker stages it: x token-major
    // `[mq*heads, hd]` (row = position*heads + head), and cos/sin as `spyre_forward`'s
    // `tile` builds them — `rope_cos_sin(pos, hd, theta)` rows replicated across the
    // heads of each position (granite's theta 1e7). The f32 reference is NeoX
    // `x*cos + rotate_half(x)*sin` over the same bytes.
    let tall = (mq * heads) as usize;
    let half = (hd / 2) as usize;
    let theta = 1e7f32;
    let x: Vec<f32> = (0..tall * hd as usize)
        .map(|i| ((i % 13) as f32) * 0.01 - 0.06)
        .collect();
    let table = |which: usize| -> Vec<f32> {
        let mut buf = vec![0.0f32; tall * hd as usize];
        for p in 0..mq as usize {
            // `rope_cos_sin`'s own row: full-width, both halves equal.
            let mut row = vec![0.0f32; hd as usize];
            for i in 0..half {
                let inv_freq = theta.powf(-(2.0 * i as f32) / hd as f32);
                let ang = p as f32 * inv_freq;
                let (s, c) = ang.sin_cos();
                row[i] = if which == 0 { c } else { s };
                row[i + half] = if which == 0 { c } else { s };
            }
            for h in 0..heads as usize {
                let off = (p * heads as usize + h) * hd as usize;
                buf[off..off + hd as usize].copy_from_slice(&row);
            }
        }
        buf
    };
    let cos = table(0);
    let sin = table(1);
    let mut want = vec![0.0f32; tall * hd as usize];
    for r in 0..tall {
        for i in 0..half {
            let (x1, x2) = (x[r * hd as usize + i], x[r * hd as usize + i + half]);
            let (c, s) = (cos[r * hd as usize + i], sin[r * hd as usize + i]);
            want[r * hd as usize + i] = x1 * c - x2 * s;
            want[r * hd as usize + i + half] = x2 * c + x1 * s;
        }
    }

    let mut session =
        scratchy_target_spyre::runner::SpyreSession::new_multi(&[(&[group], &[2u64])], Vec::new())
            .expect("build the one-program session");
    let out = session
        .run_step(
            0,
            vec![
                (0, x, vec![tall, hd as usize]),
                (1, cos, vec![tall, hd as usize]),
                (2, sin, vec![tall, hd as usize]),
            ],
            &[(3, 0)],
        )
        .unwrap_or_else(|_| panic!("run the spliced rope program (mq={mq} heads={heads} hd={hd})"));
    let got = &out[&3];
    assert_eq!(
        got.len(),
        tall * hd as usize,
        "mq={mq} heads={heads} hd={hd}"
    );
    let mut max_abs = 0.0f32;
    for (g, w) in got.iter().zip(&want) {
        max_abs = max_abs.max((g - w).abs());
    }
    assert!(
        max_abs < 0.05,
        "the REAL spliced rope program diverged from the host reference (mq={mq} \
         heads={heads} hd={hd}): max abs err {max_abs}"
    );
}

/// ⛔ THE BATCHED-DECODE LM-HEAD GATE. A result-width matmul with an ODD vocab at
/// `rows > 1` under `rows_are_requests` — granite's batched-decode lm_head
/// (vocab 49155) — must SPLICE, byte-identically to the builder: the odd-N
/// SingleCorelet re-patterning in `PlanCorelets` (the C++'s own one-stick fix) made
/// odd N compilable, and the splice covers the whole batched-decode lm_head shape.
/// This test pins the compile at the exact conjunction that once escaped the splice's
/// guards (odd vocab + rows>1 + rows_are_requests), then runs the byte-identity
/// comparison against the builder's own `lower_matmul_node` at that shape.
#[test]
fn batched_decode_odd_vocab_lm_head_splices_byte_identically() {
    // rows > 1 AND rows_are_requests (so the prefill-fold guard does not take it)
    // AND odd result-width cols — the exact conjunction that used to escape.
    let (m, k, n) = (8u32, 2048u32, 49155u32);
    let ir = matmul_ir(m, k, n);
    let weight_ids: HashSet<u32> = [0u32, 1u32].into_iter().collect();

    // The builder control, called directly.
    let layout = compute_bundle_layout(&ir, &weight_ids, true, &Default::default())
        .unwrap_or_else(|e| panic!("layout minted m={m} k={k} n={n}: {e}"));
    let mut sym = 0i64;
    let mut quantized = HashSet::new();
    let builder_ops = lower_matmul_node(
        &ir.nodes[0],
        &ir,
        None,
        &mut sym,
        Some(&layout),
        &mut quantized,
    )
    .unwrap_or_else(|e| panic!("builder lowered m={m} k={k} n={n}: {e}"));
    let [builder] = &builder_ops[..] else {
        panic!(
            "one matmul node lowers to one op, got {}",
            builder_ops.len()
        )
    };
    let builder_ktir = builder
        .ktir
        .as_ref()
        .expect("builder op carries its program");

    // The splice — the odd vocab must compile at ANY row count (PlanCorelets'
    // SingleCorelet re-patterning), never refuse.
    let node = &ir.nodes[0];
    let spliced = scratchy_triton_splice::lower(node, &ir, true)
        .unwrap_or_else(|e| panic!("odd vocab must splice at any row count: {e}"));
    assert_eq!(
        spliced.op_name, builder.op_name,
        "op_name (m={m} k={k} n={n})"
    );

    // Byte-identity through the same door under the same layout.
    let mut sym = 0i64;
    let mut quantized = HashSet::new();
    let builder_emitted =
        door_lower(builder_ktir, &mut sym, Some(&layout), &mut quantized, None).unwrap_or_else(
            |e| panic!("builder program lowered (m={m} k={k} n={n}): {}", e.message),
        );
    let mut sym = 0i64;
    let spliced_emitted = door_lower(
        spliced
            .ktir
            .as_ref()
            .expect("spliced op carries its program"),
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

/// The fp8 shapes that matter for the delivery scope: granite 3.2's projections
/// at decode rows and a prefill rung, non-square on purpose (the orientation
/// discriminator). (M, K, N) — N is the weight's own out-features; the wscale row
/// is `[1, N]`.
const MATMUL_FP8_SHAPES: &[(u32, u32, u32)] = &[
    (1, 2048, 512),   // DECODE — the shape every chat token runs
    (1, 2048, 2048),  // decode, square
    (31, 2048, 512),  // prefill rung, non-square
    (31, 2048, 2048), // prefill rung, square
    // The 8b's own shapes (hidden 4096): q/k/v at n=4096 square, o_proj
    // n=4096, the MLP gate/up n=12800, down n=4096 — at the real PREFILL rung's
    // mq=96 (the rung the card bakes) as well as decode. The byte-identity law
    // is shape-independent but the LADDER's compile is not — these pin that the
    // wide-kernel monomorphisation still yields the builder's descriptors.
    (1, 4096, 4096),
    (31, 4096, 12800),
    (96, 4096, 4096),
    (96, 4096, 12800),
];

/// The fp8 W8A8 row's byte-identity gate: the spliced `matmul_fp8.py` program vs
/// the builder's `KtirFunc::matmul_fp8` through the SAME door. Both programs are
/// `Program::Matmul` with an fp8 weight view and arity-3 bindings, so the door
/// emits its 12-op W8A8 chain (abs→amax→amaxfl→ascale→invs→sc→chi→cl→qfp8ch→
/// batchmatmulfp8→dqa→dqw) for BOTH — at one-node grain the `quantized` set is
/// fresh, so neither path can dedup and the comparison is per-node exact.
#[test]
fn spliced_fp8_matmul_is_byte_identical_to_the_builder() {
    for &(m, k, n) in MATMUL_FP8_SHAPES {
        let ir = matmul_fp8_ir(m, k, n);
        let weight_ids: HashSet<u32> = [0u32, 1u32, 2u32].into_iter().collect();

        // 1. The builder path — the control, called directly.
        let layout = compute_bundle_layout(&ir, &weight_ids, true, &Default::default())
            .unwrap_or_else(|e| panic!("layout minted m={m} k={k} n={n}: {e}"));
        let mut sym = 0i64;
        let mut quantized = HashSet::new();
        let builder_ops = lower_matmul_node(
            &ir.nodes[0],
            &ir,
            None,
            &mut sym,
            Some(&layout),
            &mut quantized,
        )
        .unwrap_or_else(|e| panic!("builder lowered m={m} k={k} n={n}: {e}"));
        let [builder] = &builder_ops[..] else {
            panic!(
                "one fp8 matmul node lowers to one op, got {}",
                builder_ops.len()
            )
        };
        let builder_ktir = builder
            .ktir
            .as_ref()
            .expect("builder op carries its program");

        // 2. The splice — the Fp8Dynamic row compiles the kernel for this node.
        let node = &ir.nodes[0];
        let spliced = scratchy_triton_splice::lower(node, &ir, true)
            .unwrap_or_else(|e| panic!("splice compiled m={m} k={k} n={n}: {e}"));

        // ⛔ THE NAME LAW — `matmul_s{id}` on both paths (the stem is the PROGRAM's,
        // and fp8 shares dense's `Program::Matmul`).
        assert_eq!(
            spliced.op_name, builder.op_name,
            "op_name (m={m} k={k} n={n})"
        );

        // 3. Both programs go through the SAME door under the SAME layout. A FRESH
        // `quantized` set per program: at one-node grain neither path dedups, so
        // both emit the full quant chain and the chain itself is compared.
        let mut sym = 0i64;
        let mut quantized = HashSet::new();
        let builder_emitted =
            door_lower(builder_ktir, &mut sym, Some(&layout), &mut quantized, None).unwrap_or_else(
                |e| panic!("builder program lowered (m={m} k={k} n={n}): {}", e.message),
            );
        let mut sym = 0i64;
        let mut quantized = HashSet::new();
        let spliced_emitted = door_lower(
            spliced
                .ktir
                .as_ref()
                .expect("spliced op carries its program"),
            &mut sym,
            Some(&layout),
            &mut quantized,
            None,
        )
        .unwrap_or_else(|e| panic!("spliced program lowered (m={m} k={k} n={n}): {}", e.message));

        // The 12-op W8A8 chain is the door's own signature — pin the count so a
        // divergence in WHICH chain the door picked is caught by name, not just by
        // byte comparison.
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

/// The fp8 row's EXECUTION gate. ⛔ THE ORIENTATION LAW, AND fp8 HAS TWO BINDING
/// LAWS THE GOLDEN CANNOT SEE: (1) the weight orientation (non-square W, same as
/// the dense gate); (2) the BYTES-VERBATIM law — the fp8 weight must cross as
/// PACKED 1-byte e4m3 through `weight_arg`'s `Fp8E4m3` arm (`new_multi`'s weights
/// parameter), not as f32-narrowed data, because the program's weight VIEW is
/// what widens each byte on read. The emulator executes the program's own compute
/// — `o[m,n] = Σ_k a[m,k]·e4m3(W[n,k])·ws[n]` — NOT the device's
/// activation-quantized W8A8 (that chain is door-emitted and is a CARD fact,
/// exercised by E2E). The host reference is that same function over the same
/// packed bytes.
#[test]
#[cfg(feature = "spyre-emu")]
fn spliced_fp8_matmul_executes_the_real_program() {
    // DECODE (m=1) and a PREFILL rung (m=31), both NON-SQUARE: a square W
    // satisfies both orientation readings.
    for (m, k, n) in [
        (1u32, 2048u32, 512u32),
        (31u32, 2048u32, 512u32),
        (1u32, 2048u32, 2048u32),
        (31u32, 2048u32, 2048u32),
    ] {
        execute_one_spliced_fp8_matmul(m, k, n);
    }
}

/// One spliced fp8 matmul through the production session entry, checked against
/// a host reference over the SAME packed e4m3 bytes and per-channel scale the
/// worker binds (`spyre_load.rs`'s fp8 arm stages the weight verbatim 1-byte;
/// `weight_arg`'s `Fp8E4m3` arm mirrors it here).
fn execute_one_spliced_fp8_matmul(m: u32, k: u32, n: u32) {
    let ir = matmul_fp8_ir(m, k, n);
    let node = &ir.nodes[0];
    let spliced = scratchy_triton_splice::lower(node, &ir, true)
        .unwrap_or_else(|e| panic!("splice compiled m={m} k={k} n={n}: {e}"));
    let k_node = spliced
        .ktir
        .as_ref()
        .expect("spliced op carries its program");

    // The host data, bound EXACTLY as the worker binds it: x `[m, k]` as f16
    // (run_step narrows), W as PACKED e4m3 bytes in its on-disk `[n, k]`
    // orientation (each byte = `f32_to_e4m3` of the reference value — the
    // checkpoint's own codec), ws `[1, n]` as f16. The reference decodes the
    // same bytes back through `e4m3_to_f32`, so the quantization error is
    // inside both sides and only the COMPUTE is compared.
    let a: Vec<f32> = (0..m * k)
        .map(|i| ((i % 13) as f32) * 0.01 - 0.06)
        .collect();
    let w_val: Vec<f32> = (0..k * n)
        .map(|i| ((i % 17) as f32) * 0.02 - 0.16)
        .collect();
    let w: Vec<u8> = w_val
        .iter()
        .map(|&v| ktir_emulator::codec::f32_to_e4m3(v))
        .collect();
    let ws: Vec<f32> = (0..n).map(|i| ((i % 7) as f32) * 0.05 + 0.1).collect();
    let mut want = vec![0.0f32; (m * n) as usize];
    for mi in 0..m {
        for ni in 0..n {
            let mut acc = 0.0f32;
            for ki in 0..k {
                let wq = ktir_emulator::codec::e4m3_to_f32(w[(ni * k + ki) as usize]);
                acc += a[(mi * k + ki) as usize] * wq;
            }
            want[(mi * n + ni) as usize] = acc * ws[ni as usize];
        }
    }

    // CONTROL: the BUILDER's own program for the same node, through the identical
    // session/binding/data. Its result must match the spliced one — the exec-level
    // twin of the byte-identity gate (and the control that LOCATED the resident
    // executor's fp8 binding bug: both paths failed identically before it).
    let weight_ids: HashSet<u32> = [0u32, 1u32, 2u32].into_iter().collect();
    let builder_ktir = {
        let layout = compute_bundle_layout(&ir, &weight_ids, true, &Default::default())
            .unwrap_or_else(|e| panic!("layout minted m={m} k={k} n={n}: {e}"));
        let mut sym = 0i64;
        let mut quantized = HashSet::new();
        let builder_ops = lower_matmul_node(
            &ir.nodes[0],
            &ir,
            None,
            &mut sym,
            Some(&layout),
            &mut quantized,
        )
        .unwrap_or_else(|e| panic!("builder lowered m={m} k={k} n={n}: {e}"));
        builder_ops[0]
            .ktir
            .clone()
            .expect("builder op carries its program")
    };
    let builder_got = execute_fp8_program(
        m,
        k,
        n,
        a.clone(),
        w.clone(),
        ws.clone(),
        builder_ktir.clone(),
    );

    let got = execute_fp8_program(m, k, n, a, w, ws, k_node.clone());
    assert_eq!(got.len(), (m * n) as usize, "m={m} k={k} n={n}");
    for (g, b) in got.iter().zip(&builder_got) {
        assert!(
            (g - b).abs() < 1e-6,
            "spliced vs builder program diverged at execution (m={m} k={k} n={n}): {g} vs {b}"
        );
    }
    let mut max_abs = 0.0f32;
    for (g, wnt) in got.iter().zip(&want) {
        max_abs = max_abs.max((g - wnt).abs());
    }
    assert!(
        max_abs < 0.05,
        "the REAL spliced fp8 matmul program diverged from the host reference (m={m} \
         k={k} n={n}): max abs err {max_abs}"
    );
}

/// Run ONE fp8 matmul KtirNode through the production session entry with the
/// given host bytes, returning the `[m, n]` output — the shared body of the
/// splice's exec gate and its builder control.
fn execute_fp8_program(
    m: u32,
    k: u32,
    n: u32,
    a: Vec<f32>,
    w: Vec<u8>,
    ws: Vec<f32>,
    k_node: ktir_superdsc::ktir_node::KtirNode,
) -> Vec<f32> {
    use std::borrow::Cow;

    let args: Vec<(
        ktir_core::ir::Ssa,
        scratchy_target_spyre::bundle_code::PlaceId,
    )> = k_node
        .func
        .arguments
        .iter()
        .map(|(ssa, _ty)| {
            (
                *ssa,
                scratchy_target_spyre::bundle_code::PlaceId::Act(k_node.bindings[ssa.slot()].get()),
            )
        })
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

    // W and ws are WEIGHTS (resident, typed bytes); x is the per-step SOURCE.
    // Both cross through `new_multi`/`weight_arg`'s own dtype mapping — the
    // production entry, not a test-side bypass.
    let mut session = scratchy_target_spyre::runner::SpyreSession::new_multi(
        &[(&[group], &[3u64])],
        vec![
            (
                1,
                w,
                scratchy_tensors::DType::Fp8E4m3,
                vec![n as usize, k as usize],
            ),
            (
                2,
                {
                    // ws as f16 bytes: encode the f32 row through the codec the
                    // `Arg::TensorBytes { F16 }` arm expects (typed, zero f32 hop).
                    use ktir_emulator::codec;
                    let f16s: Vec<u16> = ws.iter().map(|&v| codec::f32_to_f16_bits(v)).collect();
                    f16s.iter().flat_map(|&b| b.to_le_bytes()).collect()
                },
                scratchy_tensors::DType::F16,
                vec![1, n as usize],
            ),
        ],
    )
    .unwrap_or_else(|e| panic!("build the one-program session (m={m} k={k} n={n}): {e}"));
    let out = session
        .run_step(0, vec![(0, a, vec![m as usize, k as usize])], &[(3, 0)])
        .unwrap_or_else(|e| panic!("run the fp8 matmul program (m={m} k={k} n={n}): {e}"));
    out.into_iter()
        .next()
        .map(|(_, v)| v)
        .expect("session produced the output tensor")
}

/// `hidden[m, k] @ W_fp8[k, n] * ws[n] -> out[m, n]` as a one-node
/// [`SubtileIR`], fp8-dynamic weights — the arity-3 twin of [`matmul_ir`]:
/// t0 = x, t1 = W (on-disk `[n, k]`), t2 = wscale (`[1, n]`), t3 = result.
fn matmul_fp8_ir(m: u32, k: u32, n: u32) -> SubtileIR {
    let tensors = vec![
        TensorShape { rows: m, cols: k },
        TensorShape { rows: n, cols: k },
        TensorShape { rows: 1, cols: n },
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
            weight: scratchy_subtile::lower::GemmWeight::Fp8Dynamic,
        },
        inputs: vec![whole(0), whole(1), whole(2)],
        output: whole(3),
    };
    SubtileIR {
        tensors,
        num_sources: 3,
        nodes: vec![node],
        result: TensorId::from_index(3),
        op_output: Vec::new(),
    }
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

/// The rope shapes that matter for the delivery scope: granite 2b (hd 64, the
/// collapsed head-major form — 32 q-heads decode, 8 kv-heads decode) and 8b (hd 128,
/// the slab form), at decode mq=1 and a prefill rung mq=31. (MQ, HEADS, HD).
const ROPE_SHAPES: &[(u32, u32, u32)] = &[
    (1, 32, 64),   // 2b decode, the q plane
    (1, 8, 64),    // 2b decode, the kv plane
    (31, 32, 64),  // 2b prefill rung
    (1, 32, 128),  // 8b decode, the q plane
    (1, 8, 128),   // 8b decode, the kv plane
    (31, 32, 128), // 8b prefill rung (the slab form)
    // The 8b's REAL prefill rung — mq=96 (the rung the card bakes) and the q
    // plane's own head count at hd=128 (32 heads = the slab form at its widest
    // head-major extent the 2b never reaches).
    (96, 8, 128),
    (96, 32, 128),
];

#[test]
fn spliced_rope_is_byte_identical_to_the_builder() {
    for &(mq, heads, hd) in ROPE_SHAPES {
        for rows_are_requests in [false, true] {
            let ir = rope_ir(mq, heads, hd);
            let weight_ids: HashSet<u32> = [0u32, 1u32, 2u32].into_iter().collect();

            // 1. The builder path — the control, called directly through the
            // head-dim value→const door (`LowerRope`), exactly as the walk does.
            let layout = compute_bundle_layout(&ir, &weight_ids, rows_are_requests, &Default::default())
                .unwrap_or_else(|e| {
                    panic!(
                        "layout minted mq={mq} heads={heads} hd={hd} r_ar={rows_are_requests}: {e}"
                    )
                });
            let mut sym = 0i64;
            let builder_ops = scratchy_subtile::model_geometry::with_config_head_dim(
                ktir_superdsc::head_counts::HeadDim::new(hd),
                LowerRope::new(
                    &ir.nodes[0],
                    &ir,
                    &mut sym,
                    Some(&layout),
                    rows_are_requests,
                ),
            )
            .unwrap_or_else(|| {
                panic!(
                    "no head-dim arm for hd={hd} — the configs in scope must declare it \
                     (mq={mq} heads={heads})"
                )
            })
            .unwrap_or_else(|e| {
                panic!(
                    "builder lowered mq={mq} heads={heads} hd={hd} r_ar={rows_are_requests}: {e}"
                )
            });
            let [builder] = &builder_ops[..] else {
                panic!("one rope node lowers to one op, got {}", builder_ops.len())
            };
            let builder_ktir = builder
                .ktir
                .as_ref()
                .expect("builder op carries its program");

            // 2. The splice — the row compiles the kernel for this node. The splice has
            // no rope-specific refusal (the builder's own `total % hd` refusal is
            // mirrored as an Err, not a fallthrough), so an Err here is a broken row.
            let node = &ir.nodes[0];
            let spliced = scratchy_triton_splice::lower(node, &ir, rows_are_requests)
                .unwrap_or_else(|e| {
                    panic!("splice compiled mq={mq} heads={heads} hd={hd}: {e}")
                });

            // ⛔ THE NAME LAW IS PART OF THE GATE — `rope_s{id}` on both paths.
            assert_eq!(
                spliced.op_name, builder.op_name,
                "op_name (mq={mq} heads={heads} hd={hd})"
            );

            // 3. Both programs go through the SAME door under the SAME layout. Rope's
            // door arm reads `rows_are_requests` off `BundleAttnParams`, so the bundle
            // fact is stated here exactly as the tape walk states it (a geometry from
            // the same config the model declares — 32/8 at the node's head dim).
            let geom = ktir_superdsc::head_counts::ModelAttnGeometry::mint(
                ktir_superdsc::head_counts::QueryHeads::new(32),
                ktir_superdsc::head_counts::KvHeads::new(8),
                ktir_superdsc::head_counts::HeadDim::new(hd),
            )
            .expect("granite's 32/8 geometry mints");
            let attn_params = scratchy_target_spyre::ktir_superdsc_door::BundleAttnParams {
                geom,
                rows_are_requests,
            };
            let mut sym = 0i64;
            let mut quantized = HashSet::new();
            let builder_emitted = door_lower(
                builder_ktir,
                &mut sym,
                Some(&layout),
                &mut quantized,
                Some(attn_params),
            )
            .unwrap_or_else(|e| {
                panic!(
                    "builder program lowered (mq={mq} heads={heads} hd={hd} r_ar={rows_are_requests}): {}",
                    e.message
                )
            });
            let mut sym = 0i64;
            let spliced_emitted = door_lower(
                spliced.ktir.as_ref().expect("spliced op carries its program"),
                &mut sym,
                Some(&layout),
                &mut quantized,
                Some(attn_params),
            )
            .unwrap_or_else(|e| {
                panic!(
                    "spliced program lowered (mq={mq} heads={heads} hd={hd} r_ar={rows_are_requests}): {}",
                    e.message
                )
            });

            assert_eq!(
                builder_emitted.len(),
                spliced_emitted.len(),
                "op count (mq={mq} heads={heads} hd={hd} r_ar={rows_are_requests})"
            );
            for (b, s) in builder_emitted.iter().zip(spliced_emitted.iter()) {
                let bj = serde_json::to_string(b.dsc()).unwrap();
                let sj = serde_json::to_string(s.dsc()).unwrap();
                assert_eq!(
                    bj, sj,
                    "descriptor bytes (mq={mq} heads={heads} hd={hd} r_ar={rows_are_requests}): \
                     builder vs splice diverged"
                );
                assert_eq!(
                    b.op_name, s.op_name,
                    "emitted op_name (mq={mq} heads={heads} hd={hd})"
                );
            }
        }
    }
}

/// One rope node as a one-node [`SubtileIR`]: `rotate(x, cos, sin) -> out` over
/// `[mq, heads*hd]`. The x/cos/sin/out tensors carry the WORKER's staging — x/out as
/// `[mq*heads, hd]` tall views of the `[mq, heads*hd]` plane (same bytes, the
/// arrangement `addr_eq` admits), cos/sin as the worker's head-tiled
/// `[mq*heads, hd]` tables (`spyre_forward.rs`'s `tile` over `rope_cos_sin` rows) —
/// because that is the binding the spliced kernel and the builder's program both
/// address. The node kind is `RopeRotate` (the pure rotation); `RopeAppend`'s extra
/// inputs are cache destinations that flow through graph edges, and the splice covers
/// both kinds with the same row.
fn rope_ir(mq: u32, heads: u32, hd: u32) -> SubtileIR {
    let tall = mq * heads;
    // t0 = x source (staged [mq*heads, hd], i.e. the [mq, heads*hd] plane reshaped),
    // t1 = cos source (head-tiled [mq*heads, hd]), t2 = sin, t3 = result.
    let tensors = vec![
        TensorShape {
            rows: tall,
            cols: hd,
        },
        TensorShape {
            rows: tall,
            cols: hd,
        },
        TensorShape {
            rows: tall,
            cols: hd,
        },
        TensorShape {
            rows: tall,
            cols: hd,
        },
    ];
    let whole = |t: usize| TensorRegion {
        tensor: TensorId::from_index(t),
        region: tensors[t].whole(),
    };
    let node = SubtileNode {
        id: scratchy_subtile::subtile_ir::SubtileId::from_index(0),
        op: SubOp::RopeRotate {
            head_dim: ktir_superdsc::head_counts::HeadDim::new(hd),
            _form: std::marker::PhantomData,
        },
        inputs: vec![whole(0), whole(1), whole(2)],
        output: whole(3),
    };
    SubtileIR {
        tensors,
        num_sources: 3,
        nodes: vec![node],
        result: TensorId::from_index(3),
        op_output: Vec::new(),
    }
}
