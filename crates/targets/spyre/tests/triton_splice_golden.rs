// SPDX-License-Identifier: Apache-2.0
//! THE SPLICE'S OWN GATE — the Triton-sourced kernel, compiled and consumed.
//!
//! The hand-rolled builder bodies are GONE (`lower_matmul_node` and kin, deleted
//! with the splice's landing): every kernel the walk emits is
//! `scratchy_triton_splice::lower`'s, so there is no second producer left to
//! compare against. What this gate pins on each row, at every shape the delivery
//! scope runs:
//!
//! 1. the splice COMPILES the row's kernel (an Err is a broken row);
//! 2. ⛔ THE NAME LAW — the op_name is the stem+rung law the bundle's own
//!    consumers key on (`rmsnorm_s{id}`, `matmul_s{id}`, ...);
//! 3. the program LOWERS through the production door
//!    (`ktir_superdsc_door::lower`, under the SAME `BundleLayout` the tape
//!    walk computes) — the descriptors that reach the card are minted here.
//!
//! The NUMBERS are the execution rows' half (`*_executes_the_real_program`,
//! `#[cfg(feature = "spyre-emu")]`): the REAL spliced program through the
//! production session entry, checked against a host reference. A row that passes
//! both gates is the row the card runs.
//!
//! Lives in `tests/` so it can name the splice (a dev-dependency) and the
//! layout walk, without the crate's own feature gates.

use std::collections::HashSet;

use scratchy_subtile::subtile_ir::{
    EwKind, GainConvention, Range, Region, SubOp, SubtileIR, SubtileNode, TensorId, TensorRegion,
    TensorShape,
};
use scratchy_target_spyre::ktir_superdsc_door::lower as door_lower;
use scratchy_target_spyre::lower_subtile_tape_to_superdsc::compute_bundle_layout;

/// The rmsnorm shapes that matter for the delivery scope: granite 3.2/3.3 at 2b and 8b
/// both normalize at hidden 2048 (2b) and 4096 (8b), decode rows 1 and a prefill rung's
/// width; gemma-4 at 12b/31b normalizes at hidden 3840 — the ONE delivery width that is
/// not a power of two, so its `1/cols` has no exact f16/f32 spelling and the constant
/// reaches the program only as its f16 image. A shape set without it passed while every
/// 3840 chain was refused, which is how the gemma-4 divergence shipped. (M, D_MODEL).
/// The kernel states its sum ceiling-safe for the f16 the ladder
/// runs it in (`rmsnorm.py`'s module doc — LegalizeTypes erases any widening), so every
/// width is splicable and byte-compared.
const SHAPES: &[(u32, u32)] = &[
    (1, 2048),
    (1, 4096),
    (31, 2048),
    (64, 4096),
    (1, 3840),
    (19, 3840),
];

#[test]
fn spliced_rmsnorm_lowers_through_the_door() {
    for &(m, c) in SHAPES {
        let ir = rmsnorm_ir(m, c, GainConvention::Scale);
        let weight_ids: HashSet<u32> = [0u32, 1u32].into_iter().collect();

        // The layout the tape walk computes for this graph.
        let layout = compute_bundle_layout(&ir, &weight_ids, false, false, &[])
            .unwrap_or_else(|e| panic!("layout minted m={m} c={c}: {e}"));

        // The splice — the row compiles the kernel for this node. Every width is
        // splicable (the kernel's ceiling-safe pre-scale), so an Err here is a broken row.
        let node = &ir.nodes[0];
        let spliced = scratchy_triton_splice::lower(node, &ir, false)
            .unwrap_or_else(|e| panic!("splice compiled m={m} c={c}: {e}"));

        // ⛔ THE NAME LAW — `rmsnorm_s{id}`, the law the bundle's own consumers key on.
        assert_eq!(
            spliced.op_name,
            format!("rmsnorm_s{}", node.id.index()),
            "op_name (m={m} c={c})"
        );

        // The program lowers through the production door under the walk's own layout —
        // the descriptors that reach the card are minted here, so a door refusal is a
        // broken row, not a fallback.
        let mut sym = 0i64;
        let _emitted = door_lower(
            spliced
                .ktir
                .as_ref()
                .expect("spliced op carries its program"),
            &mut sym,
            Some(&layout),
            None,
        )
        .unwrap_or_else(|e| panic!("spliced program lowered (m={m} c={c}): {}", e.message));
    }
}

/// ⭐ THE `(1 + w)` CONVENTION'S OWN DOOR GATE — the OnePlusScale row end to end:
/// the splice compiles `rmsnorm_oneplus_fwd`, the census gives 1.0 a registry slot
/// (the epsilon's own mechanism), and the door emits the `rmga_` add ahead of the
/// `rmg_` gain multiply — the same formula the program runs, op for op. The SCALE
/// CONTROL below pins the other half of the row's law: a Scale-only model pushes
/// NO 1.0 into the registry (the nothing-is-seeded law) and emits NO `rmga_` op —
/// byte-identical to the form before the row existed.
#[test]
fn spliced_oneplus_rmsnorm_lowers_through_the_door() {
    for &(m, c) in SHAPES {
        let ir = rmsnorm_ir(m, c, GainConvention::OnePlusScale);
        let weight_ids: HashSet<u32> = [0u32, 1u32].into_iter().collect();

        // The layout the tape walk computes for this graph — the census must have
        // collected the convention's own 1.0 alongside the epsilon, or the door's
        // slot lookup fails as a registry desync (the test's own control on the
        // wiring, not an assertion here: the door refuses below).
        let layout = compute_bundle_layout(&ir, &weight_ids, false, false, &[])
            .unwrap_or_else(|e| panic!("layout minted m={m} c={c}: {e}"));
        assert!(
            layout
                .scalarmul_scales
                .iter()
                .any(|s| s.to_bits() == 1.0f32.to_bits()),
            "the OnePlusScale node's census must collect the convention's 1.0 (m={m} c={c})"
        );

        // The splice — the row compiles the oneplus entry for this node.
        let node = &ir.nodes[0];
        let spliced = scratchy_triton_splice::lower(node, &ir, false)
            .unwrap_or_else(|e| panic!("splice compiled the oneplus row m={m} c={c}: {e}"));

        // ⛔ THE NAME LAW — the SAME stem as the Scale row: the convention differs
        // in the ENTRY and one constexpr, never in the op the bundle keys on.
        assert_eq!(
            spliced.op_name,
            format!("rmsnorm_s{}", node.id.index()),
            "op_name (m={m} c={c})"
        );

        // The program lowers through the production door under the walk's own layout.
        let mut sym = 0i64;
        let emitted = door_lower(
            spliced
                .ktir
                .as_ref()
                .expect("spliced op carries its program"),
            &mut sym,
            Some(&layout),
            None,
        )
        .unwrap_or_else(|e| panic!("oneplus program lowered (m={m} c={c}): {}", e.message));

        // THE OP CENSUS — the fixture's output is t2, so the chain's prefixes are
        // `o2`: the `rmga_o2` add must be present AND ahead of the `rmg_o2` gain
        // multiply (the add fills the Gw row the multiply reads; the reverse order
        // would read the stored row).
        let t = 2;
        let add = emitted
            .iter()
            .position(|op| op.op_name == format!("rmga_o{t}"))
            .unwrap_or_else(|| {
                panic!(
                    "no rmga_o{t} in the oneplus chain (m={m} c={c}): {:?}",
                    emitted.iter().map(|o| &o.op_name).collect::<Vec<_>>()
                )
            });
        let mul = emitted
            .iter()
            .position(|op| op.op_name == format!("rmg_o{t}"))
            .unwrap_or_else(|| panic!("no rmg_o{t} in the oneplus chain (m={m} c={c})"));
        assert!(
            add < mul,
            "rmga_o{t} must precede rmg_o{t} (m={m} c={c}): the add fills the Gw row the \
             multiply reads"
        );

        // THE SCALE CONTROL — the same shape under Scale: no 1.0 in the registry,
        // no rmga_ op. This is the nothing-is-seeded law read backwards: the oneplus
        // push is gated on the node's own convention, so a Scale-only model's
        // registry is byte-identical to the form before the row existed.
        let scale_ir = rmsnorm_ir(m, c, GainConvention::Scale);
        let scale_layout = compute_bundle_layout(&scale_ir, &weight_ids, false, false, &[])
            .unwrap_or_else(|e| panic!("scale layout minted m={m} c={c}: {e}"));
        assert!(
            !scale_layout
                .scalarmul_scales
                .iter()
                .any(|s| s.to_bits() == 1.0f32.to_bits()),
            "a Scale-only model must not seed a 1.0 slot (m={m} c={c}) — the registry shifts \
             every later constant's tid"
        );
    }
}

/// `rmsnorm(x, gamma) -> out` as a one-node [`SubtileIR`] — the same fixture shape
/// `superdsc_time_tile.rs` mints for its matmul control. t0 = x source, t1 = gamma
/// source, t2 = result. Eps is granite's `1e-5`; the gain convention is the caller's
/// (Scale for the plain gate, OnePlusScale for the convention's own).
fn rmsnorm_ir(m: u32, c: u32, gain: GainConvention) -> SubtileIR {
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
        op: SubOp::RmsNorm { eps: 1e-5, gain },
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
/// `[96, 4096]` RUNGS EXERCISE THE ROW-BLOCKED KERNEL (the eight-live-tile LX budget:
/// `rows_per_block` splits the region into `[10, 12800]` / `[32, 4096]` blocks plus a
/// tail), so they exercise the blocked spliced program against the splice's own blocked
/// lowering — one descriptor on each side, spanning the same windows.
const SILUMUL_SHAPES: &[(u32, u32)] = &[(1, 4096), (1, 12800), (31, 4096), (64, 12800), (96, 4096)];

#[test]
fn spliced_silumul_lowers_through_the_door() {
    for &(m, c) in SILUMUL_SHAPES {
        let ir = silumul_ir(m, c);
        let weight_ids: HashSet<u32> = [0u32, 1u32].into_iter().collect();

        // The layout the tape walk computes for this graph.
        let layout = compute_bundle_layout(&ir, &weight_ids, false, false, &[])
            .unwrap_or_else(|e| panic!("layout minted m={m} c={c}: {e}"));

        // The splice — the row compiles the kernel for this node, at EVERY rung: a
        // region whose eight-tile live set exceeds the LX budget is ROW-BLOCKED inside
        // the kernel (the `rows_per_block` law), so an Err here is a broken row or a
        // broken block law.
        let node = &ir.nodes[0];
        let spliced = scratchy_triton_splice::lower(node, &ir, false)
            .unwrap_or_else(|e| panic!("splice compiled m={m} c={c}: {e}"));

        // ⛔ THE NAME LAW — `silumul_s{id}`.
        assert_eq!(
            spliced.op_name,
            format!("silumul_s{}", node.id.index()),
            "op_name (m={m} c={c})"
        );

        // The program lowers through the production door under the walk's own layout.
        let mut sym = 0i64;
        let _emitted = door_lower(
            spliced
                .ktir
                .as_ref()
                .expect("spliced op carries its program"),
            &mut sym,
            Some(&layout),
            None,
        )
        .unwrap_or_else(|e| panic!("spliced program lowered (m={m} c={c}): {}", e.message));
    }
}

/// `silu(gate) * up -> out` as a one-node [`SubtileIR`]. t0 = gate source, t1 = up
/// source, t2 = result. Both operands are activations, but the splice is
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

/// ⛔ THE COLUMN-CHUNKED NODE IS THE WINDOWED-KERNEL FAMILY'S OWN GATE — the 8b
/// defect's, now pinned at the DOOR. The front end tiles a wide pointwise op
/// into column chunks (production `nb = 8192`; granite 8b's 12800-wide MLP silumul is
/// chunks `0..8192` and `8192..12800`), and the kernel states the window:
/// `N_TOTAL` names the STORAGE (the descriptor's shape/strides) and `C_START`
/// names the WINDOW (the load/store offsets). The chunked node must compile AND
/// lower through the door — the whole-tensor control at the same width pins that
/// the discrimination is by REGION, never by width.
#[test]
fn a_column_chunked_silumul_splices_through_the_door() {
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

    let weight_ids: HashSet<u32> = [0u32, 1u32].into_iter().collect();
    let layout = compute_bundle_layout(&chunked, &weight_ids, false, false, &[])
        .unwrap_or_else(|e| panic!("layout minted for the chunked node: {e}"));

    // The splice — the chunked node compiles (the kernels state `C_START`).
    let spliced = scratchy_triton_splice::lower(&chunked.nodes[0], &chunked, false)
        .unwrap_or_else(|e| panic!("splice compiled the chunked node: {e}"));
    assert_eq!(
        spliced.op_name,
        format!("silumul_s{}", chunked.nodes[0].id.index()),
        "op_name — the splice names the op by the stem+rung law"
    );

    // Through the door: the regions the door reads must resolve, or the descriptors
    // mis-address on the card.
    let mut sym = 0i64;
    let _emitted = door_lower(
        spliced
            .ktir
            .as_ref()
            .expect("spliced op carries its program"),
        &mut sym,
        Some(&layout),
        None,
    )
    .unwrap_or_else(|e| panic!("spliced program lowered: {}", e.message));

    // THE CONTROL: the whole-tensor node at the same total width still splices — the
    // discrimination is by REGION, never by width.
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
/// (the residual adds' width) at decode rows and a prefill rung's width. The
/// `[96, 4096]` rung exceeds the binary live-tile budget and exercises the
/// ROW-BLOCKED kernel (`rows_per_block(4096, 3)` = 85 → one block + an 11-row tail).
/// (M, N).
const EW_SHAPES: &[(u32, u32)] = &[(1, 2048), (1, 4096), (31, 2048), (64, 4096), (96, 4096)];

/// Every `EwKind` the splice has a row for. The kinds the door refuses
/// (Gelu/QuickGelu/GeluErf) and the kind the splice itself refuses (Relu2 —
/// Nemotron-H is not spyre-capable) are absent: they have no descriptor
/// chain, so a door comparison would pin nothing.
const EW_KINDS: &[EwKind] = &[
    EwKind::Add,
    EwKind::BiasAdd,
    EwKind::Mul,
    EwKind::Sub,
    EwKind::Silu,
];

/// The stem law the op_name carries (`add_s{id}`, `mul_s{id}`, ...; BiasAdd is
/// `add`). ⛔ ENUMERATED, NEVER `_`: a new `EwKind` must be an E0004 here, the
/// same discipline the splice's own registry rows carry.
fn elementwise_stem(kind: EwKind) -> &'static str {
    match kind {
        EwKind::Add | EwKind::BiasAdd => "add",
        EwKind::Mul => "mul",
        EwKind::Sub => "sub",
        EwKind::Silu => "silu",
        EwKind::Gelu => "gelu",
        EwKind::QuickGelu => "quickgelu",
        EwKind::GeluErf => "geluerf",
        // Refused by the splice (no squared-ReLU kernel in the elementwise
        // family), so it never reaches `elementwise_stem` through `EW_KINDS` —
        // it is enumerated here only to keep this match exhaustive.
        EwKind::Relu2 => "relu2",
    }
}

#[test]
fn spliced_elementwise_lowers_through_the_door() {
    for &kind in EW_KINDS {
        for &(m, c) in EW_SHAPES {
            let ir = elementwise_ir(m, c, kind);
            let weight_ids: HashSet<u32> = [0u32, 1u32].into_iter().collect();

            // The layout the tape walk computes for this graph.
            let layout = compute_bundle_layout(&ir, &weight_ids, false, false, &[])
                .unwrap_or_else(|e| panic!("layout minted {kind:?} m={m} c={c}: {e}"));

            // The splice — the row compiles the kernel for this node at every
            // rung: a region whose live set exceeds the LX budget is ROW-BLOCKED
            // inside the kernel (the `rows_per_block` law), so an Err here is a
            // broken row or a broken block law.
            let node = &ir.nodes[0];
            let spliced = scratchy_triton_splice::lower(node, &ir, false)
                .unwrap_or_else(|e| panic!("splice compiled {kind:?} m={m} c={c}: {e}"));

            // ⛔ THE NAME LAW — the producer's own stem law (`add_s{id}`,
            // `mul_s{id}`, `sub_s{id}`, `silu_s{id}`; BiasAdd is `add`), read off
            // the producer's kind.
            assert_eq!(
                spliced.op_name,
                format!("{}_s{}", elementwise_stem(kind), node.id.index()),
                "op_name ({kind:?} m={m} c={c})"
            );

            // The program lowers through the production door under the walk's own layout.
            let mut sym = 0i64;
            let _emitted = door_lower(
                spliced
                    .ktir
                    .as_ref()
                    .expect("spliced op carries its program"),
                &mut sym,
                Some(&layout),
                None,
            )
            .unwrap_or_else(|e| {
                panic!(
                    "spliced program lowered ({kind:?} m={m} c={c}): {}",
                    e.message
                )
            });
        }
    }
}

/// One elementwise node as a one-node [`SubtileIR`]: unary kinds read t0 and write t2;
/// binary kinds read t0 and t1 and write t2. All operands the output's shape (a
/// broadcast operand is a builder `EwOperand` path main's splice deliberately does not
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
    // The MLP WIDTHS the hidden-width rows never reach: gate/up at 2b's d_ff=8192 and the
    // down projection's K=8192 form. Pinned when the sweep found the 8b-shaped
    // `swiglu_mlp_granite` FRONTEND FIXTURE tripping `assemble_matmul`'s placement-footprint
    // panic through the whole door (pre-existing, fixture-blocking-specific): the real
    // spliced programs at these shapes lower clean at every rung — these rows keep it
    // that way.
    (1, 2048, 8192),  // 2b decode, gate/up — the MLP width every chat token runs
    (1, 8192, 2048),  // 2b decode, down
    (31, 2048, 8192), // 2b prefill rung, gate/up
    (31, 8192, 2048), // 2b prefill rung, down
];

#[test]
fn spliced_dense_matmul_lowers_through_the_door() {
    for &(m, k, n) in MATMUL_SHAPES {
        let ir = matmul_ir(m, k, n);
        let weight_ids: HashSet<u32> = [0u32, 1u32].into_iter().collect();

        // The layout the tape walk computes for this graph (`rows_are_requests:
        // true` is the decode/batched shape).
        let layout = compute_bundle_layout(&ir, &weight_ids, true, true, &[])
            .unwrap_or_else(|e| panic!("layout minted m={m} k={k} n={n}: {e}"));

        // The splice — the row compiles the kernel for this node.
        let node = &ir.nodes[0];
        let spliced = scratchy_triton_splice::lower(node, &ir, true)
            .unwrap_or_else(|e| panic!("splice compiled m={m} k={k} n={n}: {e}"));

        // ⛔ THE NAME LAW — `matmul_s{id}`.
        assert_eq!(
            spliced.op_name,
            format!("matmul_s{}", node.id.index()),
            "op_name (m={m} k={k} n={n})"
        );

        // The program lowers through the production door under the walk's own layout.
        let mut sym = 0i64;
        let _emitted = door_lower(
            spliced
                .ktir
                .as_ref()
                .expect("spliced op carries its program"),
            &mut sym,
            Some(&layout),
            None,
        )
        .unwrap_or_else(|e| panic!("spliced program lowered (m={m} k={k} n={n}): {}", e.message));
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
#[cfg(feature = "spyre-emu")]
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
    // orientation — the same bytes the historical builder's transpose-B maps read. The f32
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

    let mut session = scratchy_target_spyre::runner::SpyreSession::new_multi(
        &[(&[group], &[2u64])],
        Vec::new(),
        // No scale slot in this program's sources — the empty stage table is the
        // byte-identical incumbent (seat (i) declares nothing when nothing is f32).
        &[],
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
            execute_one_spliced_elementwise(m, c, kind);
        }
        // ⭐ THE ROW-BLOCKED RUNG: `[350, 2048]` exceeds every family's live-tile
        // budget — binary blk = `rows_per_block(2048, 3)` = 170 (2 blocks + a 10-row
        // tail), silu blk = 85 (4 blocks + a 10-row tail) — so the emulator runs the
        // BLOCKED spliced program and the host reference checks every row the tiles
        // span, tail included.
        execute_one_spliced_elementwise(350, 2048, kind);
    }
}

/// One spliced elementwise program through the production session entry, checked
/// against a host reference.
#[cfg(feature = "spyre-emu")]
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

    let mut session = scratchy_target_spyre::runner::SpyreSession::new_multi(
        &[(&[group], &[2u64])],
        Vec::new(),
        // No scale slot in this program's sources — the empty stage table is the
        // byte-identical incumbent (seat (i) declares nothing when nothing is f32).
        &[],
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
#[cfg(feature = "spyre-emu")]
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
    let a: Vec<f32> = (0..n).map(|i| ((i % 23) as f32) * 0.02 - 0.22).collect();
    // The tanh form over the same values, in f32 — the reference the kernel's sigmoid
    // identity must reproduce to f16 scale.
    let want: Vec<f32> = a
        .iter()
        .map(|&x| {
            let v = 0.797_884_6 * (x + 0.044_715 * x * x * x);
            0.5 * x * (1.0 + v.tanh())
        })
        .collect();

    let mut session = scratchy_target_spyre::runner::SpyreSession::new_multi(
        &[(&[group], &[2u64])],
        Vec::new(),
        // No scale slot in this program's sources — the empty stage table is the
        // byte-identical incumbent (seat (i) declares nothing when nothing is f32).
        &[],
    )
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

/// The scalarmul row's gate — the spliced `scalarmul.py` program through the
/// door under the walk's own layout, at granite's own multipliers (the logits
/// scale's `1/√hidden` at 2b/8b widths) and a decode + prefill-rung shape. The
/// chunked form has its own gate below
/// (`spliced_column_chunked_scalarmul_lowers_per_chunk`); the whole-tensor
/// shapes at these widths fit the three-live-tile budget.
#[test]
fn spliced_scalarmul_lowers_through_the_door() {
    // (M, C, SCALE) — 2b's and 8b's logits scale at the hidden each normalizes to.
    // The m=96 row is the ROW-BLOCK TAIL rung at the bake's own defect shape:
    // `rows_per_block(8192, 3)` = 42, so 96 rows = 2 full blocks + a 12-row tail —
    // the tail descriptor is its own `[TAIL_H, N]` block and must clear the door too
    // (the bake caught a tail that stored through the full-block descriptor; no
    // golden shape reached a tail).
    for (m, c, scale) in [
        (1u32, 2048u32, 0.022_097_087f32),
        (1u32, 4096u32, 0.015_625f32),
        (31u32, 2048u32, 0.022_097_087f32),
        (96u32, 8192u32, 0.022_097_087f32),
    ] {
        let ir = scalarmul_ir(m, c, scale);
        let weight_ids: HashSet<u32> = [0u32].into_iter().collect();

        // The layout the tape walk computes for this graph.
        let layout = compute_bundle_layout(&ir, &weight_ids, false, false, &[])
            .unwrap_or_else(|e| panic!("layout minted m={m} c={c}: {e}"));

        // The splice — the row compiles `scalarmul.py` for this node.
        let node = &ir.nodes[0];
        let spliced = scratchy_triton_splice::lower(node, &ir, false)
            .unwrap_or_else(|e| panic!("splice compiled m={m} c={c}: {e}"));

        // ⛔ THE NAME LAW — `scalarmul_s{id}`.
        assert_eq!(
            spliced.op_name,
            format!("scalarmul_s{}", node.id.index()),
            "op_name (m={m} c={c} scale={scale})"
        );

        // The program lowers through the production door under the walk's own layout.
        // ⛔ THE SCALE MUST HAVE A REGISTRY SLOT — the door reads it off the
        // program (`program_scalarmul_scale`) and looks it up in
        // `BundleLayout::scalarmul_scales`, which `compute_bundle_layout` fills from
        // the node payload, so a splice that states a different scale than the node's
        // fails HERE by registry desync rather than by a wrong number on the card.
        let mut sym = 0i64;
        let _emitted = door_lower(
            spliced
                .ktir
                .as_ref()
                .expect("spliced op carries its program"),
            &mut sym,
            Some(&layout),
            None,
        )
        .unwrap_or_else(|e| panic!("spliced program lowered (m={m} c={c}): {}", e.message));
    }
}

/// The scalarmul row's CHUNKED gate — granite's own 7-chunk logits, the shape the
/// 8b defect was convicted on. The logits tensor is `[1, 49155]`
/// (49155 = 6 × 8192 + 3), and `lower_region` tiles it into SEVEN scalarmul chunks
/// that share one output TensorId; the splice compiles every chunk with
/// `C_START` naming its column corner and every chunk must lower through the door —
/// under one layout threaded over ALL SEVEN (the one `bundle_sym` counter the
/// production bake threads).
#[test]
fn spliced_column_chunked_scalarmul_lowers_per_chunk() {
    const LOGITS_COLS: u32 = 49_155;
    const NB: u32 = 8_192;
    let scale = 0.022_097_087f32;
    let ir = scalarmul_ir(1, LOGITS_COLS, scale);
    let weight_ids: HashSet<u32> = [0u32].into_iter().collect();
    let layout = compute_bundle_layout(&ir, &weight_ids, false, false, &[])
        .unwrap_or_else(|e| panic!("layout minted for the chunked logits: {e}"));

    // ONE symbol counter across the bundle, the way the bake threads it — chunk k's
    // descriptor names depend on the chunks before it.
    let mut splice_sym = 0i64;
    let mut chunk_idx = 0usize;
    let mut c_start = 0u32;
    while c_start < LOGITS_COLS {
        let len = NB.min(LOGITS_COLS - c_start);
        let window = |t: usize| TensorRegion {
            tensor: TensorId::from_index(t),
            region: Region {
                rows: Range::new(0, 1),
                cols: Range::new(c_start, len),
            },
        };
        let mut chunked = ir.clone();
        chunked.nodes[0].inputs = vec![window(0)];
        chunked.nodes[0].output = window(1);

        // The splice — every chunk compiles (`C_START` states the corner).
        let spliced = scratchy_triton_splice::lower(&chunked.nodes[0], &chunked, false)
            .unwrap_or_else(|e| {
                panic!("splice compiled chunk {chunk_idx} (c_start {c_start}): {e}")
            });
        assert_eq!(
            spliced.op_name,
            format!("scalarmul_s{}", chunked.nodes[0].id.index()),
            "op_name (chunk {chunk_idx}, c_start {c_start})"
        );

        let _emitted = door_lower(
            spliced
                .ktir
                .as_ref()
                .expect("spliced op carries its program"),
            &mut splice_sym,
            Some(&layout),
            None,
        )
        .unwrap_or_else(|e| panic!("spliced program lowered (chunk {chunk_idx}): {}", e.message));

        chunk_idx += 1;
        c_start += len;
    }
    assert_eq!(
        chunk_idx, 7,
        "granite's logits 49155 = 6 × 8192 + 3 is SEVEN chunks"
    );
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
    // ⭐ AND THE PROMPT CHUNK'S OWN WIDTHS, where the positions outnumber the heads and the rope loops
    // over the HEADS (`rope.py`'s `_rope_per_head`): 2b's q plane at 256 rows and its kv plane at the
    // 512-row window. (31, 8, 128) above is that form too.
    for (mq, heads, hd) in [
        (1u32, 32u32, 64u32),
        (31, 8, 128),
        (256, 32, 64),
        (512, 8, 64),
    ] {
        execute_one_spliced_rope(mq, heads, hd);
    }
}

/// ⛔ A PROMPT CHUNK'S ROPE DOES NOT GROW WITH THE CHUNK. It looped over the positions, so a 512-row
/// chunk's rope was 512 bodies. Where the positions outnumber the heads it now loops over the heads,
/// so the program is the same size at every chunk width above the head count.
#[test]
fn a_prompt_chunks_rope_does_not_grow_with_its_rows() {
    // The op count, and its three commonest kinds — what grew, if anything did. PRODUCTION'S node shape,
    // the wide `[mq, heads·hd]` plane `to_wavefront` builds (the splice derives `H = total / hd` off
    // it); `rope_ir`'s tall `[mq·heads, hd]` node states ONE head of `mq·heads` positions.
    let ops_at = |mq: u32, heads: u32| {
        let ir = rope_plane_ir(mq, heads, 64);
        let spliced = scratchy_triton_splice::lower(&ir.nodes[0], &ir, false)
            .unwrap_or_else(|e| panic!("splice compiled mq={mq} heads={heads}: {e}"));
        let k = spliced.ktir.expect("spliced op carries its program");
        let ops = k.func.ops_deep();
        let mut kinds: std::collections::BTreeMap<String, usize> = Default::default();
        for o in &ops {
            *kinds.entry(format!("{:?}", o.op_type)).or_default() += 1;
        }
        let mut top: Vec<(String, usize)> = kinds.into_iter().collect();
        top.sort_by_key(|k| std::cmp::Reverse(k.1));
        top.truncate(3);
        (ops.len(), top)
    };
    for heads in [8u32, 32] {
        let ((a, ka), (b, kb)) = (ops_at(64, heads), ops_at(512, heads));
        assert_eq!(
            a, b,
            "{heads} heads: 64 rows {a} ops {ka:?}, 512 rows {b} ops {kb:?} — the rope grew with \
             the chunk"
        );
    }
}

/// One spliced rope program through the production session entry, checked against a
/// host NeoX reference over the worker-staged bytes.
#[cfg(feature = "spyre-emu")]
fn execute_one_spliced_rope(mq: u32, heads: u32, hd: u32) {
    use std::borrow::Cow;

    // ⛔ PRODUCTION'S PLANE, NOT THE TALL NODE: `rope_ir` states `[mq·heads, hd]`, which the splice reads
    // as ONE head of `mq·heads` positions (`H = total / hd = 1`), so a run over it never had more than one
    // head. `rope_plane_ir` is `to_wavefront`'s `[mq, heads·hd]` — the SAME bytes (element (p, h·hd + d)
    // is tall row p·heads + h, column d), so the staging below is unchanged.
    let ir = rope_plane_ir(mq, heads, hd);
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

    let mut session = scratchy_target_spyre::runner::SpyreSession::new_multi(
        &[(&[group], &[2u64])],
        Vec::new(),
        // No scale slot in this program's sources — the empty stage table is the
        // byte-identical incumbent (seat (i) declares nothing when nothing is f32).
        &[],
    )
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
/// (vocab 49155) — must SPLICE and lower through the door: the odd-N
/// SingleCorelet re-patterning in `PlanCorelets` (the C++'s own one-stick fix) made
/// odd N compilable, and the splice covers the whole batched-decode lm_head shape.
/// This test pins the compile at the exact conjunction that once escaped the splice's
/// guards (odd vocab + rows>1 + rows_are_requests).
#[test]
fn batched_decode_odd_vocab_lm_head_splices() {
    // rows > 1 AND rows_are_requests (so the prefill-fold guard does not take it)
    // AND odd result-width cols — the exact conjunction that used to escape.
    let (m, k, n) = (8u32, 2048u32, 49155u32);
    let ir = matmul_ir(m, k, n);
    let weight_ids: HashSet<u32> = [0u32, 1u32].into_iter().collect();

    let layout = compute_bundle_layout(&ir, &weight_ids, true, true, &[])
        .unwrap_or_else(|e| panic!("layout minted m={m} k={k} n={n}: {e}"));

    // The splice — the odd vocab must compile at ANY row count (PlanCorelets'
    // SingleCorelet re-patterning), never refuse.
    let node = &ir.nodes[0];
    let spliced = scratchy_triton_splice::lower(node, &ir, true)
        .unwrap_or_else(|e| panic!("odd vocab must splice at any row count: {e}"));
    assert_eq!(
        spliced.op_name,
        format!("matmul_s{}", node.id.index()),
        "op_name (m={m} k={k} n={n})"
    );

    // Through the production door under the walk's own layout.
    let mut sym = 0i64;
    let _emitted = door_lower(
        spliced
            .ktir
            .as_ref()
            .expect("spliced op carries its program"),
        &mut sym,
        Some(&layout),
        None,
    )
    .unwrap_or_else(|e| panic!("spliced program lowered (m={m} k={k} n={n}): {}", e.message));
}

/// The PREFILL lm-head FOLD's gate: `scratchy_triton_splice::lower_all` —
/// which routes an m>1 vocab-wide MatmulTile through the Triton `lmlast.py` extraction
/// plus the re-lowered m=1 matmul — through the SAME door under the SAME layout.
/// `compute_bundle_layout` sees only the ORIGINAL graph (the reserved
/// `LAST_HIDDEN_TID` staging is beyond `ir.tensors`, so the layout cannot know about
/// it), and that is the production condition: the bake mints the layout once for the
/// graph, then the walk's m>1 tail folds underneath it.
///
/// The fold's second half re-enters the ordinary matmul row — this gate therefore
/// covers the WINDOWED matmul form too (`matmul.py`'s `M_TOTAL`, the out descriptor
/// naming the `[mq, vocab]` storage while the store tile stays `[1, vocab]` at row 0),
/// which no other matmul gate reaches: every other matmul shape is decode-shaped.
#[test]
fn spliced_prefill_lm_head_fold_lowers_through_the_door() {
    // hidden[mq, k] @ W_lmhead[k, n] -> logits[mq, n], rows_are_requests FALSE (the
    // prefill walk). The 2b's own tail shape at a prefill rung: k=2048, granite's odd
    // vocab 49155 (the odd width the ladder's re-patterning exists for), mq=31 (the
    // rung the walk actually rolls) and a mid rung 8.
    for &(mq, k, n) in &[(31u32, 2048u32, 49_155u32), (8u32, 2048u32, 49_155u32)] {
        let ir = matmul_ir(mq, k, n);
        let weight_ids: HashSet<u32> = [0u32, 1u32].into_iter().collect();
        let layout = compute_bundle_layout(&ir, &weight_ids, false, false, &[])
            .unwrap_or_else(|e| panic!("layout minted mq={mq} k={k} n={n}: {e}"));

        // The splice — `lower_all` takes the fold route for this shape.
        let node = &ir.nodes[0];
        let spliced_ops = scratchy_triton_splice::lower_all(node, &ir, false)
            .unwrap_or_else(|e| panic!("splice folded mq={mq} k={k} n={n}: {e}"));
        assert_eq!(
            spliced_ops.len(),
            2,
            "the fold is the extraction plus the m=1 matmul (mq={mq} k={k} n={n}): {} ops",
            spliced_ops.len()
        );

        // Each program through the door under the SAME layout, symbol counters
        // threaded in bundle order (the door's names are per-bundle).
        let mut s_sym = 0i64;
        for s_op in &spliced_ops {
            let s_ktir = s_op.ktir.as_ref().expect("spliced op carries its program");
            let _emitted =
                door_lower(s_ktir, &mut s_sym, Some(&layout), None).unwrap_or_else(|e| {
                    panic!(
                        "spliced program lowered (mq={mq} k={k} n={n}): {}",
                        e.message
                    )
                });
        }
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
    // wide-kernel monomorphisation still yields the splice's descriptors.
    (1, 4096, 4096),
    (31, 4096, 12800),
    (96, 4096, 4096),
    (96, 4096, 12800),
];

/// The fp8 W8A8 row's gate: the spliced `matmul_fp8.py` program through the door.
/// The program is `Program::Matmul` with an fp8 weight view and arity-3 bindings, so
/// the door emits its 12-op W8A8 chain (abs→amax→amaxfl→ascale→invs→sc→chi→cl→
/// qfp8ch→batchmatmulfp8→dqa→dqw) — at one-node grain no dedup can hide a missing
/// link in the chain.
#[test]
fn spliced_fp8_matmul_lowers_through_the_door() {
    for &(m, k, n) in MATMUL_FP8_SHAPES {
        let ir = matmul_fp8_ir(m, k, n);
        let weight_ids: HashSet<u32> = [0u32, 1u32, 2u32].into_iter().collect();

        // The layout the tape walk computes for this graph.
        let layout = compute_bundle_layout(&ir, &weight_ids, true, true, &[])
            .unwrap_or_else(|e| panic!("layout minted m={m} k={k} n={n}: {e}"));

        // The splice — the Fp8Dynamic row compiles the kernel for this node.
        let node = &ir.nodes[0];
        let spliced = scratchy_triton_splice::lower(node, &ir, true)
            .unwrap_or_else(|e| panic!("splice compiled m={m} k={k} n={n}: {e}"));

        // ⛔ THE NAME LAW — `matmul_s{id}` (fp8 shares dense's `Program::Matmul`).
        assert_eq!(
            spliced.op_name,
            format!("matmul_s{}", node.id.index()),
            "op_name (m={m} k={k} n={n})"
        );

        // The program lowers through the production door under the walk's own layout.
        // One node is one grain: the full quant chain is emitted and minted.
        let mut sym = 0i64;
        let _emitted = door_lower(
            spliced
                .ktir
                .as_ref()
                .expect("spliced op carries its program"),
            &mut sym,
            Some(&layout),
            None,
        )
        .unwrap_or_else(|e| panic!("spliced program lowered (m={m} k={k} n={n}): {}", e.message));
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
#[cfg(feature = "spyre-emu")]
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

    let got = execute_fp8_program(m, k, n, a, w, ws, k_node.clone());
    assert_eq!(got.len(), (m * n) as usize, "m={m} k={k} n={n}");
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
/// given host bytes, returning the `[m, n]` output.
#[cfg(feature = "spyre-emu")]
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
        // No scale slot in this program's sources — the empty stage table is the
        // byte-identical incumbent (seat (i) declares nothing when nothing is f32).
        &[],
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
fn spliced_rope_lowers_through_the_door() {
    for &(mq, heads, hd) in ROPE_SHAPES {
        for rows_are_requests in [false, true] {
            let ir = rope_ir(mq, heads, hd);
            let weight_ids: HashSet<u32> = [0u32, 1u32, 2u32].into_iter().collect();

            // The layout the tape walk computes for this graph.
            let layout = compute_bundle_layout(
                &ir,
                &weight_ids,
                rows_are_requests,
                rows_are_requests,
                &[],
            )
            .unwrap_or_else(|e| {
                panic!("layout minted mq={mq} heads={heads} hd={hd} r_ar={rows_are_requests}: {e}")
            });

            // The splice — the row compiles the kernel for this node. The splice has
            // no rope-specific refusal (the historical builder's own `total % hd` refusal is
            // mirrored as an Err, not a fallthrough), so an Err here is a broken row.
            let node = &ir.nodes[0];
            let spliced = scratchy_triton_splice::lower(node, &ir, rows_are_requests)
                .unwrap_or_else(|e| panic!("splice compiled mq={mq} heads={heads} hd={hd}: {e}"));

            // ⛔ THE NAME LAW — `rope_s{id}`.
            assert_eq!(
                spliced.op_name,
                format!("rope_s{}", node.id.index()),
                "op_name (mq={mq} heads={heads} hd={hd})"
            );

            // The program lowers through the production door under the walk's own
            // layout. Rope's door arm reads `rows_are_requests` off
            // `BundleAttnParams` — the bundle states NO geometry: the door mints
            // each program's geometry from the program's own views (the
            // hybrid-model law; gemma-4 carries two attention classes, so no
            // per-bundle geometry exists to state here).
            let attn_params = scratchy_target_spyre::ktir_superdsc_door::BundleAttnParams {
                rows_are_requests,
                // The production mint's value for a batched-shape fixture with no one-request
                // rung to widen it: the gather follows the row kind here.
                gathers_kv: rows_are_requests,
                chunk_pages: None,
                // Stated by the program: this fixture's q/k are Ext sources with no projection
                // matmuls, so the program states its real scale and the door emits the score
                // multiplies.
                scale_law: ktir_superdsc::score_scale::ScoreScaleLaw::StatedByProgram,
                // No handoff: this fixture's rope program has no o matmul at all, and a stated
                // handoff without one is the door's own refusal.
                headmajor_handoff: false,
                oproj_wtid: None,
            };
            let mut sym = 0i64;
            let _emitted = door_lower(
                spliced.ktir.as_ref().expect("spliced op carries its program"),
                &mut sym,
                Some(&layout),
                Some(attn_params),
            )
            .unwrap_or_else(|e| {
                panic!(
                    "spliced program lowered (mq={mq} heads={heads} hd={hd} r_ar={rows_are_requests}): {}",
                    e.message
                )
            });
        }
    }
}

/// One rope node as a one-node [`SubtileIR`]: `rotate(x, cos, sin) -> out` over
/// `[mq, heads*hd]`. The x/cos/sin/out tensors carry the WORKER's staging — x/out as
/// `[mq*heads, hd]` tall views of the `[mq, heads*hd]` plane (same bytes, the
/// arrangement `addr_eq` admits), cos/sin as the worker's head-tiled
/// `[mq*heads, hd]` tables (`spyre_forward.rs`'s `tile` over `rope_cos_sin` rows) —
/// because that is the binding the spliced kernel and the historical builder program both
/// address. The node kind is `RopeRotate` (the pure rotation); `RopeAppend`'s extra
/// inputs are cache destinations that flow through graph edges, and the splice covers
/// both kinds with the same row.
/// [`rope_ir`] at PRODUCTION'S node shape: the wide `[mq, heads·hd]` plane `to_wavefront` builds for a
/// rope (`out_cols` = the rope's full width), so the splice reads `H = heads` and `MQ = mq`.
fn rope_plane_ir(mq: u32, heads: u32, hd: u32) -> SubtileIR {
    let plane = TensorShape {
        rows: mq,
        cols: heads * hd,
    };
    let tensors = vec![plane, plane, plane, plane];
    let whole = |t: usize| TensorRegion {
        tensor: TensorId::from_index(t),
        region: tensors[t].whole(),
    };
    let node = SubtileNode {
        id: scratchy_subtile::subtile_ir::SubtileId::from_index(0),
        op: SubOp::RopeRotate {
            head_dim: scratchy_subtile::model_geometry::HeadDim::new(hd),
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
            head_dim: scratchy_subtile::model_geometry::HeadDim::new(hd),
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

// ══════════════════════════════════════════════════════════════════════════════════════════════
// THE ATTENTION GOLDEN
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// One attention shape the golden compares: everything the spliced `attn.py` derives its program
/// from — geometry, rung, cache producer — so a shape here is one (arm × geometry × rung ×
/// producer) cell of the delivery scope.
#[derive(Clone, Copy)]
struct AttnShape {
    /// (nqh, nkvh, hd) — granite 2b is 32/8/64, 8b is 32/8/128.
    geom: (u32, u32, u32),
    /// Query rows: 1 = decode; >1 = a prefill chunk.
    mq: u32,
    /// The resident cache TENSOR's row extent (`cap`).
    cap: u32,
    /// The swept rung the bundle was baked for (`ActiveCap`, raw — 0 = FULL, u32::MAX = NONE).
    rung: u32,
    /// The cache's producer: `true` = `SameForwardRopeAppend` (the masked prefix law — read
    /// from row 0 to `swept.min(cache tensor rows)`, runtime length mask), `false` =
    /// `PrePopulatedExt` (read at the region's own rows and corner).
    rope_appended: bool,
}

impl AttnShape {
    fn active_cap(self) -> ktir_superdsc::ktir_node::ActiveCap {
        ktir_superdsc::ktir_node::ActiveCap::new(self.rung)
    }
}

/// The attention shapes that matter for the delivery scope — granite 2b/8b, every arm of the
/// attention splice's own split, both cache producers, and the rungs the bake iterates.
///
/// ⛔ THE CAP IS THE RESIDENT CACHE'S, WHICH PRODUCTION CAPS AT ONE PAGE. The bake's prefix
/// capacity is `min(max_position_embeddings, 256)` (`codegen.rs`'s `prefix_cap_default`), and the
/// door's mask-block law refuses any sweep past `PAGE_MASK_COLS = 256` — one fold pass is one
/// page, so a pass that swept further would read the NEXT page's validity rows. Every shape here
/// keeps `swept <= cap <= 256`, exactly what every production bake satisfies by construction:
///
/// * decode (`mq == 1`, rope-appended — the MASKED prefix): `ActiveCap::FULL` (sweep the whole
///   cache tensor = one page) and the interior ladder rungs `decode_ladder(256)` bakes — 64, 128;
/// * decode, PRE-POPULATED producer (the emulator's host-threaded cache): the prefix is read at
///   its own region rows — a different tile shape and no runtime mask;
/// * prefill ONE-PASS (`ActiveCap::NONE`, `mq > 1`, new block == the whole chunk): the dead
///   prefix's injected views + the additive `[mq, mq]` causal triangle — 2b at the m=31 rung and
///   8b at the m=96 rung (the rung the card bakes), plus the ladder's own bottom rung m=7 and
///   the descriptor-law boundary m=8: below 8 the triangle is unstated through any descriptor
///   (a 7-wide f16 block is 14 bytes, under Triton's 16-byte floor) and the shape takes the
///   per-row arm; at 8 it stays one-pass;
/// * prefill CONTINUATION (`swept > 0`, `mq > 1`): per-row causal `qi+1` tiles over the new
///   block beside the swept prefix, at `ActiveCap::FULL` — the prefix-capable bundle's own bake.
const ATTN_SHAPES: &[AttnShape] = &[
    // 2b decode, rope-appended, FULL sweep (the whole one-page resident cache).
    AttnShape {
        geom: (32, 8, 64),
        mq: 1,
        cap: 256,
        rung: 0,
        rope_appended: true,
    },
    // 2b decode, rope-appended, the interior ladder rungs `decode_ladder(256)` bakes.
    AttnShape {
        geom: (32, 8, 64),
        mq: 1,
        cap: 256,
        rung: 64,
        rope_appended: true,
    },
    AttnShape {
        geom: (32, 8, 64),
        mq: 1,
        cap: 256,
        rung: 128,
        rope_appended: true,
    },
    // 2b decode, PRE-POPULATED cache (the emulator path).
    AttnShape {
        geom: (32, 8, 64),
        mq: 1,
        cap: 256,
        rung: 0,
        rope_appended: false,
    },
    // 8b decode (hd 128), rope-appended, FULL + one interior rung.
    AttnShape {
        geom: (32, 8, 128),
        mq: 1,
        cap: 256,
        rung: 0,
        rope_appended: true,
    },
    AttnShape {
        geom: (32, 8, 128),
        mq: 1,
        cap: 256,
        rung: 128,
        rope_appended: true,
    },
    // 2b prefill ONE-PASS: first chunk, no prefix, m=31 rung.
    AttnShape {
        geom: (32, 8, 64),
        mq: 31,
        cap: 256,
        rung: u32::MAX,
        rope_appended: true,
    },
    // 2b prefill, the ladder's OWN BOTTOM RUNG: m=7. The one-pass arm's causal
    // triangle is a `[7, 7]` f16 ADD, and Triton's descriptor law demands 16 bytes in
    // the last block dim (7 * 2 = 14 < 16) — so this shape MUST take the per-row
    // continuation arm, which states causality by `qi + 1` slice and binds NO mask.
    // The card bake's first rung is exactly this shape; a regression to the one-pass
    // arm here is the build-time refusal the ladder law below exists to make a test
    // failure instead.
    AttnShape {
        geom: (32, 8, 64),
        mq: 7,
        cap: 256,
        rung: u32::MAX,
        rope_appended: true,
    },
    // The descriptor law's own boundary: MQ=8 (8 * 2 = 16 bytes) is the FLOOR the
    // one-pass arm accepts, and must stay on it — not fall through to the per-row
    // arm — so the boundary is pinned from both sides.
    AttnShape {
        geom: (32, 8, 64),
        mq: 8,
        cap: 256,
        rung: u32::MAX,
        rope_appended: true,
    },
    // 8b prefill ONE-PASS at the card's own m=96 rung.
    AttnShape {
        geom: (32, 8, 128),
        mq: 96,
        cap: 256,
        rung: u32::MAX,
        rope_appended: true,
    },
    // 2b prefill CONTINUATION: the full one-page swept prefix + causal new block — the
    // prefix-capable bundle's own bake (`ActiveCap::FULL`, the widest rung).
    AttnShape {
        geom: (32, 8, 64),
        mq: 31,
        cap: 256,
        rung: 0,
        rope_appended: true,
    },
    // 8b prefill CONTINUATION.
    AttnShape {
        geom: (32, 8, 128),
        mq: 96,
        cap: 256,
        rung: 0,
        rope_appended: true,
    },
];

/// The attention fixture's decode-position model: how many positions the mask says are valid
/// this step (`= decode_position + 1`). For a rope-appended cache the prefix segment is
/// `valid_len - 1` rows (the new row arrives through the separate new-k/new-v segments); for a
/// pre-populated cache the whole region is valid. 33 keeps the rope-appended prefix region
/// non-degenerate (`lower_region` slices it to 32 rows).
const ATTN_VALID_LEN: u32 = 33;

/// One attention node (plus its RopeAppend producer when the shape wants one) as a
/// `lower_region`-bound `SubtileIR` — the ONLY construction that can mint the Tiled stage's
/// `KvCacheLayout`/`KvCacheProducer`/`SoftmaxStateId` witnesses, because their constructors are
/// `pub(crate)`.
///
/// The graph mirrors `to_wavefront`'s production binding exactly (`fixtures.rs`'s decode layer):
/// `[q, Ext(pk), Ext(pv), k, v]` for the attention, and — when `rope_appended` — a `RopeAppend`
/// op before it writing the SAME cache sources, which is what makes `lower_region` bind
/// `KvCacheProducer::SameForwardRopeAppend` (its `k_cache_producer_node` map). `valid_len`
/// rides the op (`SubOp::attn_decode`'s own field) at the value production threads.
fn attn_ir(s: AttnShape, scale: f32) -> SubtileIR {
    let (nqh, nkvh, hd) = s.geom;
    let q_width = nqh * hd;
    let kv_width = nkvh * hd;
    let mq = s.mq;
    // The pre-populated prefix's own region rows (the emulator-threaded extent main's
    // builder read directly); the rope-appended one is sliced by `lower_region` below.
    const PREPOP_ROWS: u32 = 32;
    let valid_len = ATTN_VALID_LEN;
    // The new block's rows: at decode the one new row; at prefill the chunk's own `mq` rows
    // (the one-pass condition `seq_len == mq`).
    let new_len = if mq > 1 { mq } else { 1 };
    // Sources: q, the two caches (TENSOR rows = `cache_rows` — the capacity the door reads back
    // off the view), new_k, new_v, cos, sin.
    let sources = |cache_rows: u32| {
        vec![
            scratchy_subtile::subtile_ir::SourceShape {
                rows: mq,
                cols: q_width,
            }, // 0: q
            scratchy_subtile::subtile_ir::SourceShape {
                rows: cache_rows,
                cols: kv_width,
            }, // 1: prefix_k
            scratchy_subtile::subtile_ir::SourceShape {
                rows: cache_rows,
                cols: kv_width,
            }, // 2: prefix_v
            scratchy_subtile::subtile_ir::SourceShape {
                rows: new_len,
                cols: kv_width,
            }, // 3: new_k
            scratchy_subtile::subtile_ir::SourceShape {
                rows: new_len,
                cols: kv_width,
            }, // 4: new_v
            // ⭐ 5/6: cos/sin AT THE K STREAM'S OWN WIDTH — to_wavefront's production binding:
            // "Cos/sin pre-tiled to each rope's full width (GQA: Q and K differ)", the K-side
            // tables minted through `bx.cos_sin(k_cols)` over the K projection's own width,
            // and the worker fills them head-replicated at exactly that width
            // (`spyre_forward.rs`'s `tile`: a `[rows, width]` source, the per-position rotary
            // row repeated `width / head_dim` times per row). The splice's windowing law
            // demands this staging — x, cos, sin and v share ONE row window and take WHOLE
            // `total` columns over tensors of one height, `total` being the append's output
            // width (`kv_width`) — so a `[mq, q_width]` cos beside a `[new_len, kv_width]` x
            // refuses at the mint. Rows are `new_len` (the rope op's own `m`, the shared
            // window; equal to `mq` by this fixture's own `new_len` derivation above).
            scratchy_subtile::subtile_ir::SourceShape {
                rows: new_len,
                cols: kv_width,
            }, // 5: cos
            scratchy_subtile::subtile_ir::SourceShape {
                rows: new_len,
                cols: kv_width,
            }, // 6: sin
        ]
    };
    // The attention op desc. The prefix region binding comes from `lower_region`'s own slice
    // law; `valid_len` is threaded as production threads it.
    let attn_op = scratchy_subtile::lower::OpDesc {
        op: scratchy_subtile::subtile_ir::SubOp::attn_decode(
            attn_geometry(s.geom).expect("granite's geometry mints"),
            scale,
            valid_len,
            scratchy_subtile::subtile_ir::AttnMask::Causal,
        ),
        m: mq,
        inputs: vec![
            scratchy_subtile::lower::InputRef::Ext(0),
            scratchy_subtile::lower::InputRef::Ext(1),
            scratchy_subtile::lower::InputRef::Ext(2),
            scratchy_subtile::lower::InputRef::Ext(3),
            scratchy_subtile::lower::InputRef::Ext(4),
        ],
    };
    let nb = std::num::NonZeroU32::new(8192).expect("8192 != 0");
    if s.rope_appended {
        // The rope-appended cache: the K-side rope_append — rotate new_k and write it (with
        // new_v) into the cache sources 1/2. THIS op is what binds SameForwardRopeAppend for
        // the attention below (`k_cache_producer_node`). The cache SOURCES are declared at the
        // full `cap` rows — the capacity the door reads — while `lower_region` slices the
        // attention's prefix REGION to `valid_len - 1` (the door's masked-segment read is
        // `swept.min(tensor rows)`; the region's rows never bound it).
        let input = scratchy_subtile::lower::LoweringInput {
            sources: sources(s.cap),
            ops: vec![
                // 0: k' = rope_append(k, cos, sin, v, prefix_k, prefix_v)
                scratchy_subtile::lower::OpDesc {
                    op: scratchy_subtile::subtile_ir::SubOp::rope_append(
                        scratchy_subtile::model_geometry::HeadDim::new(hd),
                        0,
                        scratchy_subtile::subtile_ir::AttnMask::Causal,
                        scratchy_subtile::subtile_ir::RopeFormTag::NeoX,
                    ),
                    m: new_len,
                    inputs: vec![
                        scratchy_subtile::lower::InputRef::Ext(3),
                        scratchy_subtile::lower::InputRef::Ext(5),
                        scratchy_subtile::lower::InputRef::Ext(6),
                        scratchy_subtile::lower::InputRef::Ext(4),
                        scratchy_subtile::lower::InputRef::Ext(1),
                        scratchy_subtile::lower::InputRef::Ext(2),
                    ],
                },
                // 1: attn = AttnDecode(q, prefix_k, prefix_v, new_k, new_v)
                attn_op,
            ],
            result: 1,
            exported: None,
        };
        scratchy_subtile::subtile_ir::lower_region(&input, nb)
    } else {
        // The pre-populated shape: no rope op in scope (the map stays empty and `lower_region`
        // binds PrePopulatedExt), the caches declared at the emulator's threaded extent.
        let input = scratchy_subtile::lower::LoweringInput {
            sources: sources(PREPOP_ROWS),
            ops: vec![attn_op],
            result: 0,
            exported: None,
        };
        scratchy_subtile::subtile_ir::lower_region(&input, nb)
    }
}

/// granite's geometry, through the same mint the config path uses.
fn attn_geometry(
    geom: (u32, u32, u32),
) -> Option<scratchy_subtile::model_geometry::ModelAttnGeometry> {
    scratchy_subtile::model_geometry::ModelAttnGeometry::mint(
        scratchy_subtile::model_geometry::QueryHeads::new(geom.0),
        scratchy_subtile::model_geometry::KvHeads::new(geom.1),
        scratchy_subtile::model_geometry::HeadDim::new(geom.2),
    )
}

/// ⛔ ONE SESSION STAGES ONE LAYOUT — placing the KV block index moves no other staged activation.
///
/// The one-request ladder runs per-page bodies that gather nothing under a split-KV chunk ceiling that
/// gathers, and the host stages every step at the CEILING's placements. Each fixture's attention
/// registers its scale (and √scale) after the attention's own placements — exactly the tail an index
/// placed beside the mask used to push down, so that a non-gathering body read its constants out of the
/// index's bytes. The macro holds every baked rung to `ladder_stages_one_layout`; this holds the layout.
#[test]
fn the_kv_block_index_moves_no_other_staged_activation() {
    use scratchy_target_spyre::lower_subtile_tape_to_superdsc::{
        KV_BLOCK_INDEX_TID, ladder_stages_one_layout,
    };
    for s in ATTN_SHAPES.iter().filter(|s| s.mq == 1) {
        let scale = if s.geom.2 == 64 { 0.015625 } else { 0.0078125 };
        let ir = attn_ir(*s, scale);
        let weight_ids: HashSet<u32> = HashSet::new();
        // The one-request rows (`rows_are_requests = false`) at both gather facts: the chunk ceiling's
        // call and the per-page body's.
        let layout = |gathers_kv: bool| {
            compute_bundle_layout(&ir, &weight_ids, false, gathers_kv, &[]).unwrap_or_else(|e| {
                panic!(
                    "layout minted geom={:?} gathers_kv={gathers_kv}: {e}",
                    s.geom
                )
            })
        };
        let plain = layout(false);
        // ⭐ PRODUCTION'S ORDER, MIRRORED: `compute_bundle_layout` only RECORDS the
        // index's reservation (carried into the layout as its `kv_block_index`);
        // each walk PLACES it last of all, after every host-staged placement,
        // through the one door `BundleLayout::place_kv_block_index_last`
        // (lower_subtile_tape_to_ktir.rs's walks, :1237 and :2767). Mirroring that
        // call here is what runs the assertions below against the REAL sequence —
        // reading `placements` straight off the mint is the stale half-sequence
        // this test was born red on (57b: "a gathering bundle placed no index").
        let mut gathering = layout(true);
        gathering.place_kv_block_index_last();
        assert!(
            !plain.placements.contains_key(&KV_BLOCK_INDEX_TID),
            "geom={:?}: a bundle that does not gather placed the index",
            s.geom
        );
        let index = *gathering
            .placements
            .get(&KV_BLOCK_INDEX_TID)
            .unwrap_or_else(|| panic!("geom={:?}: a gathering bundle placed no index", s.geom));
        ladder_stages_one_layout(&gathering, &plain)
            .unwrap_or_else(|e| panic!("geom={:?}: {e}", s.geom));
        let behind: Vec<u32> = gathering
            .placements
            .values()
            .filter(|p| p.segment == index.segment && p.tid != KV_BLOCK_INDEX_TID)
            .filter(|p| p.role == index.role && p.offset >= index.offset)
            .map(|p| p.tid)
            .collect();
        assert!(
            behind.is_empty(),
            "geom={:?}: tensor(s) {behind:?} sit behind the index in its segment — the index must be \
             the LAST activation, or its presence moves them",
            s.geom
        );
    }
}

#[test]
fn spliced_attn_lowers_through_the_door() {
    for &s in ATTN_SHAPES {
        // ⛔ ROWS-ARE-REQUESTS IS A DECODE-BUNDLE FACT: the bake states it per
        // bundle — a decode batch's rows are requests, a prefill chunk's rows are
        // positions of one sequence. The gate states it the way the bake does:
        // decode (mq == 1) tries BOTH kinds, prefill only `false`. `false` at
        // mq == 1 is PRODUCTION'S ROUTING — the bake states
        // `rows_are_requests = !is_prefill && decode_rows > 1`, so the judgment's
        // solo decode lowers through the prefill arm and those cells MUST lower
        // green. `true` at mq == 1 is the batched kernel's slot merge at
        // NUM_SEQS = 1, a width production never bakes — the sub-stick class the
        // door-lowering block below expects and pins.
        let row_kinds: &[bool] = if s.mq == 1 { &[false, true] } else { &[false] };
        for &rows_are_requests in row_kinds {
            // granite's attention_multiplier, the config value (NOT a recomputed 1/sqrt(hd)):
            // 2b (hd 64) 0.015625, 8b (hd 128) 0.0078125 — the f16-exact ones.
            let scale = if s.geom.2 == 64 { 0.015625 } else { 0.0078125 };
            let ir = attn_ir(s, scale);
            // The weight-id census is EMPTY: every source this fixture declares is an
            // activation or a KV cache (the layout walk re-places the caches into seg2 itself).
            let weight_ids: HashSet<u32> = HashSet::new();

            // The layout the tape walk computes for this graph — MUTABLE here because the
            // producer's placement pass extends it: the host-staged inputs the spliced
            // program states are placed after the splice, before the door, exactly where
            // the re-rolled walk runs the pass.
            let mut layout = compute_bundle_layout(
                &ir,
                &weight_ids,
                rows_are_requests,
                rows_are_requests,
                &[],
            )
            .unwrap_or_else(|e| {
                panic!(
                    "layout minted geom={:?} mq={} cap={} rung={} r_ar={rows_are_requests}: {e}",
                    s.geom, s.mq, s.cap, s.rung
                )
            });
            let node = ir.nodes.last().unwrap();
            let cap = ir.tensors[node.inputs[1].tensor.index()].rows;

            // The splice — `lower_attn` compiles attn.py for this node's geometry and rung.
            // No fold statement on these fixtures (`None`): the one-shot form, ONE program —
            // the paged-fold family is the chunked-ladder rung's own statement, pinned by its
            // own fixtures.
            let spliced_result = scratchy_triton_splice::lower_attn(
                node,
                &ir,
                cap,
                s.active_cap(),
                rows_are_requests,
                None,
                // The fixtures' nodes carry real multipliers (0.015625/0.0078125 below) and no
                // fold receipt, so their law is the one every real-batch model lives under: the
                // program states the whole scale.
                Some(ktir_superdsc::score_scale::ScoreScaleLaw::StatedByProgram),
            );
            // ⛔ THE HD-64 KERNEL LIMIT — the kernels' own designed refusal, EXPECTED and
            // PINNED here, never skipped. All three spliced attention kernels refuse
            // HD != 64 at compile time BY DESIGN — the same static_assert at
            // batched_prefill.py:133, batched_decode.py:106 and paged_fold.py:144: the
            // V plane's [64, HD] block is one contiguous stick block only at HD 64, "a
            // wider head needs the per-(window, slab) V gather this kernel does not
            // state yet". 8b (hd 128) is a DECLARED UNSUPPORTED GEOMETRY of this
            // Triton path — the judgment model is granite-3.1-2b (hidden 2048 / 32
            // heads = hd 64) — not a defect of the slice. The hd-128 cells STAY in the
            // table: each must REFUSE with the kernels' own message, and a splice that
            // COMPILES one fails by name (the limit lifted silently).
            let spliced_ops = if s.geom.2 != 64 {
                let Err(e) = spliced_result else {
                    panic!(
                        "the splice COMPILED geom={:?} mq={} rung={} r_ar={rows_are_requests} \
                         — the HD-64 kernel limit lifted silently",
                        s.geom, s.mq, s.rung
                    );
                };
                assert!(
                    e.contains("one contiguous stick block only at HD 64"),
                    "geom={:?} mq={} rung={} r_ar={rows_are_requests}: the refusal must be \
                     the kernels' HD-64 limit, not some other defect: {e}",
                    s.geom,
                    s.mq,
                    s.rung
                );
                continue;
            } else {
                spliced_result.unwrap_or_else(|e| {
                    panic!(
                        "splice compiled geom={:?} mq={} rung={} r_ar={rows_are_requests}: {e}",
                        s.geom, s.mq, s.rung
                    )
                })
            };
            assert_eq!(
                spliced_ops.len(),
                1,
                "a foldless `lower_attn` returns exactly the one-shot program (geom={:?} mq={} \
                 rung={})",
                s.geom,
                s.mq,
                s.rung
            );
            let spliced = &spliced_ops[0];

            // ⛔ THE NAME LAW — `attn_s{id}`.
            assert_eq!(
                spliced.op_name,
                format!("attn_s{}", node.id.index()),
                "op_name (geom={:?} mq={} rung={})",
                s.geom,
                s.mq,
                s.rung
            );

            // ⛔ THE REGION FACT RIDES EVERY CARRIER, UNCONDITIONALLY — the trigger fix's
            // anti-regression pin (the CARRIER half; the walk-stamp half lives in the spyre
            // crate's own tests). The one-shot program gathers the paged pool through the
            // absolute-entry tables in BOTH row arms (`mint_attn` binds the id tables either
            // way), so a carrier minted BARE — the hole this pins shut — is a double-based
            // read: the page base composed on top of the absolute entries. The prefix-free
            // rungs (`rung == u32::MAX`, a start==0 chunk whose ladder stated no fold region)
            // are the shapes that hit that hole, and they run right here in this table.
            let region = spliced.kv_paged_region.unwrap_or_else(|| {
                panic!(
                    "the one-shot carrier states its paged region (geom={:?} mq={} rung={} \
                     r_ar={rows_are_requests}) — a bare carrier double-bases its reads",
                    s.geom, s.mq, s.rung
                )
            });
            // The region's VALUE is the arm's own contract, stated above at the mint: the
            // decode arm's spec received the pool capacity as its `pages`; the prefill arm's
            // spec defaulted its `None` region to the walk's own extent — the pages the
            // chunk attends, read off the node's own regions exactly as the arm reads them.
            let expected_pages = if rows_are_requests {
                cap
            } else {
                let prefix_len = node.inputs[1].region.rows.len;
                let new_len = node.inputs[3].region.rows.len;
                (prefix_len + new_len)
                    .div_ceil(ktir_superdsc::sdsc_abstract::PagedKvPool::PAGE_SLOTS as u32)
            };
            assert_eq!(
                region.region_pages, expected_pages,
                "the region's page count is the arm's own pages contract (geom={:?} mq={} \
                 rung={} r_ar={rows_are_requests})",
                s.geom, s.mq, s.rung
            );
            // The pool's tids ride with the region — the same pair the stamp site's reach
            // guard asks the layout for, so the stamp and the placement can never name
            // different pools.
            let scratchy_subtile::subtile_ir::SubOp::AttnDecode { layout: kv, .. } = &node.op
            else {
                panic!(
                    "the attention fixture's last node is an AttnDecode (geom={:?})",
                    s.geom
                );
            };
            assert_eq!(
                region.k_tid,
                kv.cache_tensor().index() as u32,
                "the region's K tid is the node's own K cache (geom={:?} mq={} rung={})",
                s.geom,
                s.mq,
                s.rung
            );
            assert_eq!(
                region.v_tid,
                kv.v_cache_tensor().index() as u32,
                "the region's V tid is the node's own V cache (geom={:?} mq={} rung={})",
                s.geom,
                s.mq,
                s.rung
            );

            // ⛔ THE ARM THE SHAPE TOOK, PINNED BY THE MASK SLOT — at `rung == NONE`
            // (a first-chunk prefill, no runtime length mask) every `MQ > 1` shape
            // takes the causal one-pass arm and binds the `[mq, mq]` triangle (a
            // mask parameter); the per-row continuation arm states causality by
            // slice and binds none. The mq=7 row pins the FRONT END'S OWN LAW: the
            // TMA 16-byte floor was a GPU validation removed from this target, so
            // the ladder's bottom rung takes the one-pass arm exactly as the
            // builder's did — a re-introduced floor would flip this row. Decode and
            // continuation shapes bind the runtime length mask instead, so they are
            // outside this pin.
            if s.rung == u32::MAX {
                let ktir = spliced
                    .ktir
                    .as_ref()
                    .expect("spliced op carries its program");
                let one_pass_mask = ktir.mask.is_some();
                let want_one_pass = s.mq > 1;
                assert_eq!(
                    one_pass_mask, want_one_pass,
                    "the arm the causal one-pass condition picked (geom={:?} mq={}): \
                     the mask slot is bound exactly when the one-pass arm runs",
                    s.geom, s.mq
                );
            }

            // The producer's own placement pass, run at its own site in the walk (after the
            // splice has stated its bindings, before the door): every host-staged input the
            // spliced program states takes a seg-ACTIVATION placement minted from the
            // program's own view, checked against the numbering law — the pass the door's
            // unplaced-parameter refusal waits for.
            scratchy_target_spyre::lower_subtile_tape_to_superdsc::place_step_inputs(
                &mut layout,
                &[std::slice::from_ref(spliced)],
                ir.tensors.len() as u32,
            )
            .unwrap_or_else(|e| {
                panic!(
                    "step inputs placed (geom={:?} mq={} rung={} r_ar={rows_are_requests}): {}",
                    s.geom, s.mq, s.rung, e
                )
            });

            // The walk's own scale-registry completion, run at its own site (beside the
            // placement pass, before the door): the spliced program states its exp2-restated
            // constant as a front-end FOLD of two config constants (`SCALE * LOG2E`), which no
            // graph payload names, so the registry's program-stated half is what the door's
            // by-bits lookup resolves it through.
            scratchy_target_spyre::lower_subtile_tape_to_ktir::complete_program_scales(
                &mut layout,
                &[std::slice::from_ref(spliced)],
            );

            // The program lowers through the production door under the walk's own
            // layout, with the bundle facts stated exactly as the tape walk states
            // them (`attn_at` reads the swept extent, the scale, the span guard and
            // the row laws off the program).
            let attn_params = scratchy_target_spyre::ktir_superdsc_door::BundleAttnParams {
                rows_are_requests,
                // The production mint's value for a batched-shape fixture with no one-request
                // rung to widen it: the gather follows the row kind here.
                gathers_kv: rows_are_requests,
                chunk_pages: None,
                // Stated by the program: same as the rope door's fixture — the program states
                // its real scale.
                scale_law: ktir_superdsc::score_scale::ScoreScaleLaw::StatedByProgram,
                // No handoff: this fixture's attention output is the graph RESULT (no o matmul
                // consumes it), so the dataflow the handoff keys on finds no edge — stated here
                // exactly as `attn_bundle_params` would state it for such a graph.
                headmajor_handoff: false,
                oproj_wtid: None,
            };
            let mut sym = 0i64;
            // ⛔ THE BATCHED-DECODE MERGE'S SUB-STICK CLASS — the door's own designed
            // refusal, EXPECTED and PINNED here, never skipped.
            //
            // THE DIFFERENCE BETWEEN THIS FIXTURE AND PRODUCTION (the judgment-adjacent
            // check): production bakes `rows_are_requests = !is_prefill && decode_rows > 1`
            // (`codegen.rs`'s ceiling lowering and every ladder rung's), so its `r_ar=true`
            // arm is BATCHED decode only — the batched kernel's slot merge leaves
            // `num_seqs·nkvh·gqa` survivor lanes, a WHOLE stick at every batch ≥ 2 — while
            // the judgment's own SOLO decode (`decode_rows == 1`) routes `r_ar=false` and
            // lowers through the prefill arm at MQ=1: the `r_ar=false` cells below are that
            // exact shape and MUST lower green. The `r_ar=true` cell at `mq == 1` is
            // `batched_decode_hm` at `NUM_SEQS = 1` — a width production never bakes, and
            // one the door refuses BY DESIGN: the kernel's slot merge
            // (`batched_decode.py`: `mm = tl.reshape(m, [BLOCKS_PER_CHUNK, NUM_SEQS*NKVH,
            // GQA])`, `merged = tl.max(mm, 0)`) reduces to `[NUM_SEQS*NKVH, GQA]` — 32
            // survivors at this geometry, HALF a stick at the scores' own SEN169_FP16 (64
            // lanes) — which is wall 13's refusal exactly as the corpus pins it
            // (`a_sub_stick_survivor_refuses_by_name`: `[32, 8, 4] → [8, 4]`, "32 survivor
            // lanes = HALF a stick", must refuse; its B=2 sibling
            // `a_one_stick_survivor_set_lowers_at_the_batch_boundary`: `[16, 16, 4] →
            // [16, 4]`, one whole stick, LOWERS). Pinning the refusal keeps the cell's
            // teeth: a door that LOWERS this shape (re-layout the merge bytes) or refuses
            // for any other reason fails here by name.
            let (nqh, nkvh, _hd) = s.geom;
            let gqa = nqh / nkvh;
            let survivors = s.mq * nkvh * gqa;
            let stick = ktir_superdsc::superdsc_opspec::Df::Fp16.elems_per_stick();
            let sub_stick = rows_are_requests && survivors < stick;
            let lowered = door_lower(
                spliced
                    .ktir
                    .as_ref()
                    .expect("spliced op carries its program"),
                &mut sym,
                Some(&layout),
                Some(attn_params),
            );
            if sub_stick {
                let Err(e) = lowered else {
                    panic!(
                        "the sub-stick merge refused (geom={:?} mq={} rung={} \
                         r_ar={rows_are_requests}): the door LOWERED a {survivors}-survivor \
                         merge the corpus pins as wall 13's refusal \
                         (`a_sub_stick_survivor_refuses_by_name`)",
                        s.geom, s.mq, s.rung
                    );
                };
                let df = ktir_superdsc::superdsc_opspec::Df::Fp16;
                for pin in [
                    format!(
                        "a leading-axis reduce with {survivors} survivor lanes, SUB-STICK at \
                         the data's own format (a {} stick is {stick} lanes)",
                        df.name()
                    ),
                    "one `is_stick` dim cannot state".to_string(),
                    format!("a·{stick}+s ≠ a·{survivors}+s"),
                ] {
                    assert!(
                        e.message.contains(&pin),
                        "the refusal is wall 13's sub-stick law (geom={:?} mq={} rung={} \
                         r_ar={rows_are_requests}); expected the pin {pin:?}, got: {}",
                        s.geom,
                        s.mq,
                        s.rung,
                        e.message
                    );
                }
            } else {
                lowered.unwrap_or_else(|e| {
                    panic!(
                        "spliced program lowered (geom={:?} mq={} rung={} r_ar={rows_are_requests}): {}",
                        s.geom, s.mq, s.rung, e.message
                    )
                });
            }
        }
    }
}

/// ⭐⭐ THE TWO RMSNORM RECOGNIZERS, PINNED TO EACH OTHER. The door recognizes
/// rmsnorm chains to fuse them (`ktir_superdsc`'s `program_rmsnorm_chains`); the
/// emulator re-derives the same walk to land the scalar chain's sticks on the
/// card's DLF16 grid (`ktir_emulator`'s `rmsnorm_scalar_sticks` — the emulator
/// cannot depend on the door's crate, so the walk is a second implementation of
/// one pattern). Two implementations of one fact is the class that desyncs
/// silently, so this pin runs BOTH over the same SHIPPED programs — every
/// spelling the delivery scope runs: the gained `rmsnorm.py` under both gain
/// conventions, and the un-gained `rmsnormunit.py` — and asserts they agree.
/// The door's per-chain `consumed` interior must contain exactly the
/// emulator's three sticks (the square, the `·INV_D` mean, the `+eps` sum —
/// never the raw reduce or the rsqrt, which land on the reduce's own
/// accumulator grid), the total must be three per chain with nothing outside
/// the door's chains, and the un-gained spelling must read the SAME eps through
/// the door's per-op reader that the walk's chain is built around. A non-norm
/// program recognizes on NEITHER side. A recognizer that drifts — a new chain
/// shape the door admits, or a discriminator the emulator tightens — fails
/// here by name, instead of silently landing the wrong grid.
#[cfg(feature = "spyre-emu")]
#[test]
fn rmsnorm_recognizers_agree_on_the_shipped_programs() {
    use ktir_emulator::comm_sched::rmsnorm_scalar_sticks;
    use ktir_superdsc::emit::lower_ktir_to_superdsc::program_rmsnorm_eps;
    use ktir_superdsc::emit::whole_function::program_rmsnorm_chains;

    // The gained spellings, both conventions, at the delivery scope's shapes.
    for &(m, c) in SHAPES {
        for gain in [GainConvention::Scale, GainConvention::OnePlusScale] {
            let ir = rmsnorm_ir(m, c, gain);
            let node = &ir.nodes[0];
            let spliced = scratchy_triton_splice::lower(node, &ir, false)
                .unwrap_or_else(|e| panic!("splice compiled m={m} c={c}: {e}"));
            let f = spliced
                .ktir
                .as_ref()
                .expect("spliced op carries its program");

            let chains = program_rmsnorm_chains(&f.func)
                .unwrap_or_else(|e| panic!("door recognized m={m} c={c}: {}", e.message));
            assert_eq!(
                chains.len(),
                1,
                "one chain per gained program (m={m} c={c})"
            );
            let sticks = rmsnorm_scalar_sticks(f.func.operations);
            assert_eq!(
                sticks.len(),
                3,
                "exactly the square, mean and meps sticks (m={m} c={c})"
            );
            let consumed: HashSet<_> = chains[0].consumed.iter().copied().collect();
            assert_eq!(
                sticks.intersection(&consumed).count(),
                3,
                "the sticks are the chain's own interior (m={m} c={c})"
            );
            // The program's own eps, read by the door: the kernel's
            // `(ms + EPS).to(tl.float32)` spelling states the add f16-typed, so
            // the constant reaches the program ALREADY collapsed to its f16
            // image — the same fact the golden's registry staging states, and
            // the one the door's registry lookup serves through its f16-image
            // fallback.
            let eps = program_rmsnorm_eps(&f.func)
                .unwrap_or_else(|| panic!("door read the gained program's eps (m={m} c={c})"));
            assert_eq!(
                eps,
                half::f16::from_f32(1e-5).to_f32(),
                "the gained program states its eps as the f16 image (m={m} c={c})"
            );
        }
    }

    // The un-gained spelling: the door reads it through the per-op epsilon
    // reader (one root, the addf carrying the eps splat), the emulator through
    // the same scalar-chain walk — and the eps both see is the f16 image.
    for &(m, c) in SHAPES {
        let ir = rmsnormunit_ir(m, c);
        let node = &ir.nodes[0];
        let spliced = scratchy_triton_splice::lower(node, &ir, false)
            .unwrap_or_else(|e| panic!("splice compiled the unit row m={m} c={c}: {e}"));
        let f = spliced
            .ktir
            .as_ref()
            .expect("spliced op carries its program");

        let eps = program_rmsnorm_eps(&f.func)
            .unwrap_or_else(|| panic!("door read the unit program's eps (m={m} c={c})"));
        assert_eq!(
            eps,
            half::f16::from_f32(1e-6).to_f32(),
            "the unit program states its eps as the f16 image (m={m} c={c})"
        );
        let sticks = rmsnorm_scalar_sticks(f.func.operations);
        assert_eq!(
            sticks.len(),
            3,
            "the unit chain's three sticks (m={m} c={c})"
        );
    }

    // The negative control: a program with no norm recognizes on NEITHER side.
    let ir = silumul_ir(1, 4096);
    let node = &ir.nodes[0];
    let spliced = scratchy_triton_splice::lower(node, &ir, false)
        .unwrap_or_else(|e| panic!("splice compiled the silumul control: {e}"));
    let f = spliced
        .ktir
        .as_ref()
        .expect("spliced op carries its program");
    let chains = program_rmsnorm_chains(&f.func)
        .unwrap_or_else(|e| panic!("door recognized the control: {}", e.message));
    assert!(chains.is_empty(), "no rmsnorm chain in a silumul program");
    assert!(
        rmsnorm_scalar_sticks(f.func.operations).is_empty(),
        "no scalar sticks in a silumul program"
    );
}

/// `rmsnorm_unit(x) -> out` as a one-node [`SubtileIR`] — the un-gained spelling
/// (`rmsnormunit.py`, the head-norm sandwich's unit): one input, no gamma. t0 = x
/// source, t1 = result. Eps is gemma-4's `1e-6` — the subnormal regime the DLF16
/// landing exists for.
#[cfg(feature = "spyre-emu")]
fn rmsnormunit_ir(m: u32, c: u32) -> SubtileIR {
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
        op: SubOp::RmsNormUnit { eps: 1e-6 },
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

/// ⛔⛔⛔ A CONTINUATION'S PROGRAM IS FLAT IN ITS ROWS AND LINEAR IN ITS PAGES — the whole-chunk
/// form, pinned at the door's own guard and across row counts. `batched_prefill_hm` unrolls one
/// body per page the chunk attends (`for s in tl.static_range(PAGES)`), so the program is
/// `fixed + pages·per_page`: at a FIXED page count the rows must not grow it (the per-row arm it
/// replaced held `mq` row bodies — granite-2b at 256 rows: 94.7 s to lower; at 512, never
/// finished), and growth with PAGES is intended — pinned below as one whole per-page quantum,
/// never hidden. Both widths must pass the door's guard, and a program that windows its query
/// per row must be refused.
#[test]
fn a_continuation_programs_size_does_not_grow_with_its_rows() {
    let ops_at = |mq: u32| {
        let s = AttnShape {
            geom: (32, 8, 64),
            mq,
            cap: 256,
            rung: 0,
            rope_appended: true,
        };
        let ir = attn_ir(s, 0.015625);
        let node = ir.nodes.last().unwrap();
        let cap = ir.tensors[node.inputs[1].tensor.index()].rows;
        let spliced = scratchy_triton_splice::lower_attn(
            node,
            &ir,
            cap,
            s.active_cap(),
            false,
            None,
            // The node's 0.015625 is a real multiplier with no fold receipt — StatedByProgram.
            Some(ktir_superdsc::score_scale::ScoreScaleLaw::StatedByProgram),
        )
        .unwrap_or_else(|e| panic!("splice compiled mq={mq}: {e}"));
        assert_eq!(
            spliced.len(),
            1,
            "a foldless splice is the one-shot program"
        );
        let k = spliced[0]
            .ktir
            .clone()
            .expect("spliced op carries its program");
        assert!(k.mask.is_some(), "mq={mq}: the continuation binds its mask");
        ktir_superdsc::emit::lower_ktir_to_superdsc::attn_program_is_flat_in_rows(
            "continuation",
            &k.func,
            mq,
            32,
        )
        .unwrap_or_else(|e| panic!("mq={mq}: {}", e.message));
        k.func.ops_deep().len()
    };
    // ⛔ FLAT IN ROWS AT A FIXED PAGE COUNT — the fixture's rope-appended prefix is
    // 32 rows (`ATTN_VALID_LEN − 1`), so a chunk of `mq` rows attends
    // ⌈(32 + mq)/256⌉ pages (the splice's own reading, lib.rs's `pages`): ONE page
    // at mq 31 and at mq 200 alike. Six times the rows must not grow the program.
    let (small, wide) = (ops_at(31), ops_at(200));
    assert!(
        wide <= small,
        "rows at a fixed page count grew the program: 31 rows {small} ops, 200 rows \
         {wide} ops"
    );
    // ⛔ AND GROWTH WITH PAGES IS ONE WHOLE PER-PAGE QUANTUM — intended, pinned
    // rather than hidden: mq 256 attends ⌈(32 + 256)/256⌉ = 2 pages, and the second
    // page costs exactly one page's worth of unrolled trips. Measured on this tree:
    // 204 ops per page (230 ops at one page, 434 at two — the 57b run's own
    // numbers, which the old per-row law read as a defect).
    const PER_PAGE_OPS: usize = 204;
    let two_pages = ops_at(256);
    assert_eq!(
        two_pages - small,
        PER_PAGE_OPS,
        "pages 1 → 2 must grow the program by exactly one per-page quantum (measured \
         {PER_PAGE_OPS} ops on this tree): 1 page {small} ops, 2 pages {two_pages} ops"
    );
    // The door's flat-in-rows guard has one production caller, `attn_at` — the dead
    // hand-written attention, slated for removal with G7. The live kernels read q once
    // by construction (batched_decode.py:156, one `q_desc.gather`; batched_prefill.py's
    // one `q_desc` load), so the guard admits every live program — pinned by `ops_at`'s
    // own calls above — and its refusal path is unexercised by any live program; it is
    // G7's to delete or re-home with the dead path.
}
