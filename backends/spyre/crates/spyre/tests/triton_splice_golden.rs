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

use ktir_superdsc::emit::ktir_matmul_fp8::QuantizedActivations;
use scratchy_subtile::subtile_ir::{
    EwKind, GainConvention, Range, Region, SubOp, SubtileIR, SubtileNode, TensorId, TensorRegion,
    TensorShape,
};
use scratchy_target_spyre::ktir_superdsc_door::{BundleAttnParams, lower as bundle_lower};

/// One program through the production door as a bundle of its own — the fp8 quantize set starts
/// empty, which is what a one-program bundle's is.
fn door_lower(
    k: &ktir_superdsc::ktir_node::KtirNode,
    sym: &mut i64,
    layout: Option<&ktir_superdsc::placement::BundleLayout>,
    attn: Option<BundleAttnParams>,
) -> Result<
    Vec<ktir_superdsc::emit::EmittedOp<ktir_superdsc::collective::TpWorld<1>>>,
    ktir_superdsc::emit::lower_ktir_to_superdsc::Error,
> {
    bundle_lower(k, sym, layout, attn, &mut QuantizedActivations::default())
}
use scratchy_target_spyre::lower_subtile_tape_to_superdsc::compute_bundle_layout;
use scratchy_triton_splice::PagedRows;

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
        let layout = compute_bundle_layout::<ktir_superdsc::collective::TpWorld<1>, _>(
            &ir,
            &weight_ids,
            PagedRows::Sequence,
            &[],
            pair_of_one_request(m),
        )
        .unwrap_or_else(|e| panic!("layout minted m={m} c={c}: {e}"));

        // The splice — the row compiles the kernel for this node. Every width is
        // splicable (the kernel's ceiling-safe pre-scale), so an Err here is a broken row.
        let node = &ir.nodes[0];
        let spliced = scratchy_triton_splice::lower::<ktir_superdsc::collective::TpWorld<1>, _>(
            node, &ir, false,
        )
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
        let layout = compute_bundle_layout::<ktir_superdsc::collective::TpWorld<1>, _>(
            &ir,
            &weight_ids,
            PagedRows::Sequence,
            &[],
            pair_of_one_request(m),
        )
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
        let spliced = scratchy_triton_splice::lower::<ktir_superdsc::collective::TpWorld<1>, _>(
            node, &ir, false,
        )
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
        let scale_layout = compute_bundle_layout::<ktir_superdsc::collective::TpWorld<1>, _>(
            &scale_ir,
            &weight_ids,
            PagedRows::Sequence,
            &[],
            pair_of_one_request(m),
        )
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
        let layout = compute_bundle_layout::<ktir_superdsc::collective::TpWorld<1>, _>(
            &ir,
            &weight_ids,
            PagedRows::Sequence,
            &[],
            pair_of_one_request(m),
        )
        .unwrap_or_else(|e| panic!("layout minted m={m} c={c}: {e}"));

        // The splice — the row compiles the kernel for this node, at EVERY rung: a
        // region whose eight-tile live set exceeds the LX budget is ROW-BLOCKED inside
        // the kernel (the `rows_per_block` law), so an Err here is a broken row or a
        // broken block law.
        let node = &ir.nodes[0];
        let spliced = scratchy_triton_splice::lower::<ktir_superdsc::collective::TpWorld<1>, _>(
            node, &ir, false,
        )
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
    let layout = compute_bundle_layout::<ktir_superdsc::collective::TpWorld<1>, _>(
        &chunked,
        &weight_ids,
        PagedRows::Sequence,
        &[],
        pair_of_one_request(1),
    )
    .unwrap_or_else(|e| panic!("layout minted for the chunked node: {e}"));

    // The splice — the chunked node compiles (the kernels state `C_START`).
    let spliced = scratchy_triton_splice::lower::<ktir_superdsc::collective::TpWorld<1>, _>(
        &chunked.nodes[0],
        &chunked,
        false,
    )
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
    let spliced = scratchy_triton_splice::lower::<ktir_superdsc::collective::TpWorld<1>, _>(
        whole_node, &ir, false,
    )
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

/// Every `EwKind` the splice has a row for. The kinds the DDL has no primitive for
/// (QuickGelu/GeluErf — the consumer's own `elementwise_op_func` refuses them by name) and
/// the kind the splice itself refuses (Relu2 — an arch using it fails the build at that row,
/// by name) are absent: they have no descriptor chain, so a door comparison would pin
/// nothing. `Gelu` IS here: its kernel spells the tanh-form longhand, which the walk's gelu
/// recogniser folds into the one `"gelu"` primitive — before that fold existed, the longhand
/// reached the SILU recogniser's shared exp-island head and was refused there, which is why
/// this list once excluded it.
const EW_KINDS: &[EwKind] = &[
    EwKind::Add,
    EwKind::BiasAdd,
    EwKind::Mul,
    EwKind::Sub,
    EwKind::Silu,
    EwKind::Gelu,
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
            let layout = compute_bundle_layout::<ktir_superdsc::collective::TpWorld<1>, _>(
                &ir,
                &weight_ids,
                PagedRows::Sequence,
                &[],
                pair_of_one_request(m),
            )
            .unwrap_or_else(|e| panic!("layout minted {kind:?} m={m} c={c}: {e}"));

            // The splice — the row compiles the kernel for this node at every
            // rung: a region whose live set exceeds the LX budget is ROW-BLOCKED
            // inside the kernel (the `rows_per_block` law), so an Err here is a
            // broken row or a broken block law.
            let node = &ir.nodes[0];
            let spliced =
                scratchy_triton_splice::lower::<ktir_superdsc::collective::TpWorld<1>, _>(
                    node, &ir, false,
                )
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
        let layout = compute_bundle_layout::<ktir_superdsc::collective::TpWorld<1>, _>(
            &ir,
            &weight_ids,
            PagedRows::Sequence,
            &[],
            pair_of_one_request(m),
        )
        .unwrap_or_else(|e| panic!("layout minted m={m} k={k} n={n}: {e}"));

        // The splice — the row compiles the kernel for this node.
        let node = &ir.nodes[0];
        let spliced = scratchy_triton_splice::lower::<ktir_superdsc::collective::TpWorld<1>, _>(
            node, &ir, true,
        )
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
    let group = spliced_matmul_group(m, k, n);

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
    let want = host_matmul(&a, &w, m, k, n);

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
    let max_abs = max_abs_err(&out[&2], &want);
    assert!(
        max_abs < 0.05,
        "the REAL spliced matmul program diverged from the host reference: max abs err {max_abs}"
    );
}

/// ⭐ TWO SESSIONS ON ONE THREAD READ THEIR OWN RESIDENT WEIGHTS. The emulator's GPU weight cache
/// is per THREAD and keyed by a weight's HBM address, and a resident pass trusts the key without
/// sampling content; two sessions over the same program lay the same weight tid out at the same
/// address. The two ranks of the two-rank gate are exactly that — and before the cache was
/// claimed per executor, the second session's prefill GEMM (m > 1 always offloads) ran on the
/// FIRST session's weight. Alternating A, B, A, each against its own reference, pins it.
#[test]
#[cfg(feature = "spyre-emu")]
fn two_sessions_on_one_thread_read_their_own_resident_weights() {
    let (m, k, n) = (31u32, 2048u32, 512u32);
    let a: Vec<f32> = (0..m * k)
        .map(|i| ((i % 13) as f32) * 0.01 - 0.06)
        .collect();
    // B's weight is A's negated, so a session served the other's weight is off by twice its product.
    let weights = |sign: f32| -> Vec<f32> {
        (0..(k * n) as usize)
            .map(|i| sign * (((i % 17) as f32) * 0.02 - 0.16))
            .collect()
    };
    let session = |w: &[f32]| {
        let bytes: Vec<u8> = w
            .iter()
            .flat_map(|v| half::f16::from_f32(*v).to_le_bytes())
            .collect();
        scratchy_target_spyre::runner::SpyreSession::new_multi(
            &[(&[spliced_matmul_group(m, k, n)], &[2u64])],
            vec![(
                1,
                scratchy_target_spyre::HostBytes::Owned(bytes),
                scratchy_tensors::DType::F16,
                vec![n as usize, k as usize],
            )],
        )
        .expect("build a one-program session over its own resident weight")
    };
    let (wa, wb) = (weights(1.0), weights(-1.0));
    let (want_a, want_b) = (host_matmul(&a, &wa, m, k, n), host_matmul(&a, &wb, m, k, n));
    // L1 error relative to the reference: ~1e-3 for the right weight (f16 operands), 2 for the other.
    let rel = |got: &[f32], want: &[f32]| -> f32 {
        let d: f32 = got.iter().zip(want).map(|(g, w)| (g - w).abs()).sum();
        d / want.iter().map(|w| w.abs()).sum::<f32>()
    };
    assert!(
        rel(&want_a, &want_b) > 1.0,
        "the two weights must give far-apart products, or the test cannot tell them apart"
    );
    let mut sessions = [session(&wa), session(&wb)];
    for (who, s, want) in [
        ("A", 0, &want_a),
        ("B", 1, &want_b),
        ("A again", 0, &want_a),
    ] {
        let out = sessions[s]
            .run_step(
                0,
                vec![(0, a.clone(), vec![m as usize, k as usize])],
                &[(2, 0)],
            )
            .expect("run the spliced matmul program");
        let err = rel(&out[&2], want);
        assert!(
            err < 0.02,
            "session {who} read another session's weight: rel err {err} against its own"
        );
    }
}

/// ⭐ A NARROW PREFILL MATMUL WRITES THE WIDTH ITS READERS READ. `[31, 2048] · [2048, 256]` crosses
/// the util floor's MAC threshold, and `DeviceWidth::for_matmul` pads its 4 output sticks to 8; a
/// program that STATES that padded width writes every row at a 512 stride, while an elementwise
/// reader of the same tensor states the logical 256 — row 0 right, every later row read from the
/// wrong place (measured: smollm2's one-rank and granite's two-rank prefill k/v). The splice states
/// the logical width when every reader does, so the output reads back row-major at 256.
#[test]
#[cfg(feature = "spyre-emu")]
fn a_narrow_prefill_matmul_writes_the_width_its_readers_read() {
    let (m, k, n) = (31u32, 2048u32, 256u32);
    assert!(
        ktir_superdsc::work::DeviceWidth::for_matmul(m, n, k, false).get() > n,
        "the shape must be one the door pads, or the test proves nothing"
    );
    // The matmul (t0 · t1 → t2) and an elementwise reader of its output (t2 + t3 → t4).
    let mut ir = matmul_ir(m, k, n);
    ir.tensors.push(TensorShape { rows: m, cols: n });
    ir.tensors.push(TensorShape { rows: m, cols: n });
    let whole = |t: usize, s: &[TensorShape]| TensorRegion {
        tensor: TensorId::from_index(t),
        region: s[t].whole(),
    };
    let reader = SubtileNode {
        id: scratchy_subtile::subtile_ir::SubtileId::from_index(1),
        op: SubOp::Elementwise(EwKind::Add),
        inputs: vec![whole(2, &ir.tensors), whole(3, &ir.tensors)],
        output: whole(4, &ir.tensors),
    };
    ir.nodes.push(reader);
    let a: Vec<f32> = (0..m * k)
        .map(|i| ((i % 13) as f32) * 0.01 - 0.06)
        .collect();
    let w: Vec<f32> = (0..k * n)
        .map(|i| ((i % 17) as f32) * 0.02 - 0.16)
        .collect();
    let want = host_matmul(&a, &w, m, k, n);
    let mut session = scratchy_target_spyre::runner::SpyreSession::new_multi(
        &[(&[spliced_node_group(&ir, 0)], &[2u64])],
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
    let row_err = |r: usize| {
        let at = r * n as usize..(r + 1) * n as usize;
        max_abs_err(&got[at.clone()], &want[at])
    };
    for r in [0, 1, (m - 1) as usize] {
        assert!(
            row_err(r) < 0.05,
            "row {r} of the narrow prefill matmul reads back wrong: max abs err {}",
            row_err(r)
        );
    }
}

/// `A[m, k] · W[n, k]ᵀ` on the host, over the on-disk `[n, k]` weight.
#[cfg(feature = "spyre-emu")]
fn host_matmul(a: &[f32], w: &[f32], m: u32, k: u32, n: u32) -> Vec<f32> {
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
    want
}

#[cfg(feature = "spyre-emu")]
fn max_abs_err(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len());
    got.iter()
        .zip(want)
        .map(|(g, w)| (g - w).abs())
        .fold(0.0f32, f32::max)
}

/// The one spliced dense matmul's launch group — the tape's own numbering (A t0, W t1, O t2).
#[cfg(feature = "spyre-emu")]
fn spliced_matmul_group(
    m: u32,
    k: u32,
    n: u32,
) -> scratchy_target_spyre::bundle_code::LaunchGroup<'static, ktir_superdsc::collective::TpWorld<1>>
{
    spliced_node_group(&matmul_ir(m, k, n), 0)
}

/// Node `at` of `ir`, spliced, as a one-program launch group bound by the tape's own numbering.
#[cfg(feature = "spyre-emu")]
fn spliced_node_group(
    ir: &SubtileIR,
    at: usize,
) -> scratchy_target_spyre::bundle_code::LaunchGroup<'static, ktir_superdsc::collective::TpWorld<1>>
{
    use std::borrow::Cow;

    let node = &ir.nodes[at];
    let spliced =
        scratchy_triton_splice::lower::<ktir_superdsc::collective::TpWorld<1>, _>(node, ir, true)
            .unwrap_or_else(|e| panic!("splice compiled node {at}: {e}"));
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
    scratchy_target_spyre::bundle_code::LaunchGroup::<ktir_superdsc::collective::TpWorld<1>> {
        kv: Default::default(),
        programs: Cow::Owned(vec![scratchy_target_spyre::bundle_code::LaunchProgram {
            func: k_node.func,
            program: k_node.program,
            args: Cow::Owned(args),
            // Stated EXPLICITLY (auditor ruling 2026-10-27): this hand-built program
            // minted no LX pins, so the window its fused segments run under is zero —
            // a program without a stated window is refused, never silently charged.
            lx_pin_window_bytes: Some(0),
        }]),
        init_binary: scratchy_target_spyre::bundle_code::BakedImage::Raw(Cow::Borrowed(&[])),
        job_bin_ptr: 0,
        correction: Cow::Borrowed(&[]),
    }
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
    let spliced =
        scratchy_triton_splice::lower::<ktir_superdsc::collective::TpWorld<1>, _>(node, &ir, false)
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
    let group =
        scratchy_target_spyre::bundle_code::LaunchGroup::<ktir_superdsc::collective::TpWorld<1>> {
            kv: Default::default(),
            programs: Cow::Owned(vec![scratchy_target_spyre::bundle_code::LaunchProgram {
                func: k_node.func,
                program: k_node.program,
                args: Cow::Owned(args),
                // Stated EXPLICITLY (auditor ruling 2026-10-27): this hand-built program
                // minted no LX pins, so the window its fused segments run under is zero —
                // a program without a stated window is refused, never silently charged.
                lx_pin_window_bytes: Some(0),
            }]),
            init_binary: scratchy_target_spyre::bundle_code::BakedImage::Raw(Cow::Borrowed(&[])),
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
        // The TANH form — the function `OpFunc::Gelu`'s polynomial and the longhand's sigmoid
        // identity both state — over the same f16-scale values (the fixture's ±0.06 window keeps
        // the whole run inside one f16 ulp of the reference either way).
        EwKind::Gelu => a
            .iter()
            .map(|&x| {
                // √(2/π) = (2/√π)·(1/√2), the std constants' own derivation, cast once to the
                // reference's float type — not a 16-digit literal clippy must audit against
                // f32's precision.
                let k = (std::f64::consts::FRAC_2_SQRT_PI * std::f64::consts::FRAC_1_SQRT_2) as f32;
                let u = k * (x + 0.044_715 * x * x * x);
                0.5 * x * (1.0 + u.tanh())
            })
            .collect(),
        other => unreachable!("exec fixture for {other:?}"),
    };

    let mut session =
        scratchy_target_spyre::runner::SpyreSession::new_multi(&[(&[group], &[2u64])], Vec::new())
            .expect("build the one-program session");
    let sources = if matches!(kind, EwKind::Silu | EwKind::Gelu) {
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
    let spliced =
        scratchy_triton_splice::lower::<ktir_superdsc::collective::TpWorld<1>, _>(node, &ir, false)
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
    let group =
        scratchy_target_spyre::bundle_code::LaunchGroup::<ktir_superdsc::collective::TpWorld<1>> {
            kv: Default::default(),
            programs: Cow::Owned(vec![scratchy_target_spyre::bundle_code::LaunchProgram {
                func: k_node.func,
                program: k_node.program,
                args: Cow::Owned(args),
                // Stated EXPLICITLY (auditor ruling 2026-10-27): this hand-built program
                // minted no LX pins, so the window its fused segments run under is zero —
                // a program without a stated window is refused, never silently charged.
                lx_pin_window_bytes: Some(0),
            }]),
            init_binary: scratchy_target_spyre::bundle_code::BakedImage::Raw(Cow::Borrowed(&[])),
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
        let layout = compute_bundle_layout::<ktir_superdsc::collective::TpWorld<1>, _>(
            &ir,
            &weight_ids,
            PagedRows::Sequence,
            &[],
            pair_of_one_request(m),
        )
        .unwrap_or_else(|e| panic!("layout minted m={m} c={c}: {e}"));

        // The splice — the row compiles `scalarmul.py` for this node.
        let node = &ir.nodes[0];
        let spliced = scratchy_triton_splice::lower::<ktir_superdsc::collective::TpWorld<1>, _>(
            node, &ir, false,
        )
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
    let layout = compute_bundle_layout::<ktir_superdsc::collective::TpWorld<1>, _>(
        &ir,
        &weight_ids,
        PagedRows::Sequence,
        &[],
        pair_of_one_request(1),
    )
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
        let spliced = scratchy_triton_splice::lower::<ktir_superdsc::collective::TpWorld<1>, _>(
            &chunked.nodes[0],
            &chunked,
            false,
        )
        .unwrap_or_else(|e| panic!("splice compiled chunk {chunk_idx} (c_start {c_start}): {e}"));
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
        let spliced = scratchy_triton_splice::lower::<ktir_superdsc::collective::TpWorld<1>, _>(
            &ir.nodes[0],
            &ir,
            false,
        )
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
    let spliced =
        scratchy_triton_splice::lower::<ktir_superdsc::collective::TpWorld<1>, _>(node, &ir, false)
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
    let group =
        scratchy_target_spyre::bundle_code::LaunchGroup::<ktir_superdsc::collective::TpWorld<1>> {
            kv: Default::default(),
            programs: Cow::Owned(vec![scratchy_target_spyre::bundle_code::LaunchProgram {
                func: k_node.func,
                program: k_node.program,
                args: Cow::Owned(args),
                // Stated EXPLICITLY (auditor ruling 2026-10-27): this hand-built program
                // minted no LX pins, so the window its fused segments run under is zero —
                // a program without a stated window is refused, never silently charged.
                lx_pin_window_bytes: Some(0),
            }]),
            init_binary: scratchy_target_spyre::bundle_code::BakedImage::Raw(Cow::Borrowed(&[])),
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

    let layout = compute_bundle_layout::<ktir_superdsc::collective::TpWorld<1>, _>(
        &ir,
        &weight_ids,
        PagedRows::Sequence,
        &[],
        pair_of_one_request(m),
    )
    .unwrap_or_else(|e| panic!("layout minted m={m} k={k} n={n}: {e}"));

    // The splice — the odd vocab must compile at ANY row count (PlanCorelets'
    // SingleCorelet re-patterning), never refuse.
    let node = &ir.nodes[0];
    let spliced =
        scratchy_triton_splice::lower::<ktir_superdsc::collective::TpWorld<1>, _>(node, &ir, true)
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
        let layout = compute_bundle_layout::<ktir_superdsc::collective::TpWorld<1>, _>(
            &ir,
            &weight_ids,
            PagedRows::Sequence,
            &[],
            pair_of_one_request(mq),
        )
        .unwrap_or_else(|e| panic!("layout minted mq={mq} k={k} n={n}: {e}"));

        // The splice — `lower_all` takes the fold route for this shape.
        let node = &ir.nodes[0];
        let spliced_ops = scratchy_triton_splice::lower_all::<
            ktir_superdsc::collective::TpWorld<1>,
            _,
        >(node, &ir, false, pair_of_one_request(mq))
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
        let mut s_fp8 = QuantizedActivations::default();
        for s_op in &spliced_ops {
            let s_ktir = s_op.ktir.as_ref().expect("spliced op carries its program");
            let _emitted = bundle_lower::<ktir_superdsc::collective::TpWorld<1>>(
                s_ktir,
                &mut s_sym,
                Some(&layout),
                None,
                &mut s_fp8,
            )
            .unwrap_or_else(|e| {
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
        let layout = compute_bundle_layout::<ktir_superdsc::collective::TpWorld<1>, _>(
            &ir,
            &weight_ids,
            PagedRows::Sequence,
            &[],
            pair_of_one_request(m),
        )
        .unwrap_or_else(|e| panic!("layout minted m={m} k={k} n={n}: {e}"));

        // The splice — the Fp8Dynamic row compiles the kernel for this node.
        let node = &ir.nodes[0];
        let spliced = scratchy_triton_splice::lower::<ktir_superdsc::collective::TpWorld<1>, _>(
            node, &ir, true,
        )
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
    let spliced =
        scratchy_triton_splice::lower::<ktir_superdsc::collective::TpWorld<1>, _>(node, &ir, true)
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
    let group =
        scratchy_target_spyre::bundle_code::LaunchGroup::<ktir_superdsc::collective::TpWorld<1>> {
            kv: Default::default(),
            programs: Cow::Owned(vec![scratchy_target_spyre::bundle_code::LaunchProgram {
                func: k_node.func,
                program: k_node.program,
                args: Cow::Owned(args),
                // Stated EXPLICITLY (auditor ruling 2026-10-27): this hand-built program
                // minted no LX pins, so the window its fused segments run under is zero —
                // a program without a stated window is refused, never silently charged.
                lx_pin_window_bytes: Some(0),
            }]),
            init_binary: scratchy_target_spyre::bundle_code::BakedImage::Raw(Cow::Borrowed(&[])),
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
                scratchy_target_spyre::HostBytes::Owned(w),
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
                    scratchy_target_spyre::HostBytes::Owned(
                        f16s.iter().flat_map(|&b| b.to_le_bytes()).collect(),
                    )
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
fn spliced_rope_lowers_through_the_door() {
    for &(mq, heads, hd) in ROPE_SHAPES {
        for rows_are_requests in [false, true] {
            let ir = rope_ir(mq, heads, hd);
            let weight_ids: HashSet<u32> = [0u32, 1u32, 2u32].into_iter().collect();

            // The layout the tape walk computes for this graph.
            let layout = compute_bundle_layout::<ktir_superdsc::collective::TpWorld<1>, _>(
                &ir,
                &weight_ids,
                PagedRows::Sequence,
                &[],
                pair_of_one_request(mq),
            )
            .unwrap_or_else(|e| {
                panic!("layout minted mq={mq} heads={heads} hd={hd} r_ar={rows_are_requests}: {e}")
            });

            // The splice — the row compiles the kernel for this node. The splice has
            // no rope-specific refusal (the historical builder's own `total % hd` refusal is
            // mirrored as an Err, not a fallthrough), so an Err here is a broken row.
            let node = &ir.nodes[0];
            let spliced =
                scratchy_triton_splice::lower::<ktir_superdsc::collective::TpWorld<1>, _>(
                    node,
                    &ir,
                    rows_are_requests,
                )
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
            let attn_params =
                scratchy_target_spyre::ktir_superdsc_door::BundleAttnParams { rows_are_requests };
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
            head_dim: scratchy_ir::model_facts::HeadDim::new(hd),
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
            head_dim: scratchy_ir::model_facts::HeadDim::new(hd),
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
    // ⭐ THE PER-RANK GEOMETRIES (TP phase 1): granite's 32/8 head split sharded over the
    // KV heads — tp=2 holds (16, 4), tp=4 (8, 2), tp=8 (4, 1), at both head dims. The
    // PREFILL rows only, mirroring the four prefill classes above at each geometry: the
    // one-pass rung (m=31 at hd 64, m=96 at hd 128), the ladder's bottom rung m=7 (the
    // per-row causal arm under Triton's 16-byte descriptor floor), the descriptor law's
    // boundary m=8 (staying ON the one-pass arm), and the continuation at rung 0 (the
    // swept prefix beside the causal new block). Decode `mq == 1` rows are NOT added at
    // `nkvh < 8`: production decode always takes the BATCHED kernel, and the per-sequence
    // decode kernel's `< 8` branch is a known door refusal out of this phase's scope.
    AttnShape {
        geom: (16, 4, 64),
        mq: 31,
        cap: 256,
        rung: u32::MAX,
        rope_appended: true,
    },
    AttnShape {
        geom: (16, 4, 64),
        mq: 7,
        cap: 256,
        rung: u32::MAX,
        rope_appended: true,
    },
    AttnShape {
        geom: (16, 4, 64),
        mq: 8,
        cap: 256,
        rung: u32::MAX,
        rope_appended: true,
    },
    AttnShape {
        geom: (16, 4, 64),
        mq: 31,
        cap: 256,
        rung: 0,
        rope_appended: true,
    },
    AttnShape {
        geom: (8, 2, 64),
        mq: 31,
        cap: 256,
        rung: u32::MAX,
        rope_appended: true,
    },
    AttnShape {
        geom: (8, 2, 64),
        mq: 7,
        cap: 256,
        rung: u32::MAX,
        rope_appended: true,
    },
    AttnShape {
        geom: (8, 2, 64),
        mq: 8,
        cap: 256,
        rung: u32::MAX,
        rope_appended: true,
    },
    AttnShape {
        geom: (8, 2, 64),
        mq: 31,
        cap: 256,
        rung: 0,
        rope_appended: true,
    },
    // The (4, 1) rows — granite's tp=8 per-rank shape, ONE KV head per card — were a named
    // FINDING (the GQA group spray read as a middle axis); the unit-axis fold in the door's
    // broadcast classification landed, and they run with the rest.
    AttnShape {
        geom: (4, 1, 64),
        mq: 31,
        cap: 256,
        rung: u32::MAX,
        rope_appended: true,
    },
    AttnShape {
        geom: (4, 1, 64),
        mq: 7,
        cap: 256,
        rung: u32::MAX,
        rope_appended: true,
    },
    AttnShape {
        geom: (4, 1, 64),
        mq: 8,
        cap: 256,
        rung: u32::MAX,
        rope_appended: true,
    },
    AttnShape {
        geom: (4, 1, 64),
        mq: 31,
        cap: 256,
        rung: 0,
        rope_appended: true,
    },
    // The same four at head dim 128 (granite-3.1-8b's per-rank shapes).
    AttnShape {
        geom: (16, 4, 128),
        mq: 96,
        cap: 256,
        rung: u32::MAX,
        rope_appended: true,
    },
    AttnShape {
        geom: (16, 4, 128),
        mq: 7,
        cap: 256,
        rung: u32::MAX,
        rope_appended: true,
    },
    AttnShape {
        geom: (16, 4, 128),
        mq: 8,
        cap: 256,
        rung: u32::MAX,
        rope_appended: true,
    },
    AttnShape {
        geom: (16, 4, 128),
        mq: 96,
        cap: 256,
        rung: 0,
        rope_appended: true,
    },
    AttnShape {
        geom: (8, 2, 128),
        mq: 96,
        cap: 256,
        rung: u32::MAX,
        rope_appended: true,
    },
    AttnShape {
        geom: (8, 2, 128),
        mq: 7,
        cap: 256,
        rung: u32::MAX,
        rope_appended: true,
    },
    AttnShape {
        geom: (8, 2, 128),
        mq: 8,
        cap: 256,
        rung: u32::MAX,
        rope_appended: true,
    },
    AttnShape {
        geom: (8, 2, 128),
        mq: 96,
        cap: 256,
        rung: 0,
        rope_appended: true,
    },
    AttnShape {
        geom: (4, 1, 128),
        mq: 96,
        cap: 256,
        rung: u32::MAX,
        rope_appended: true,
    },
    AttnShape {
        geom: (4, 1, 128),
        mq: 7,
        cap: 256,
        rung: u32::MAX,
        rope_appended: true,
    },
    AttnShape {
        geom: (4, 1, 128),
        mq: 8,
        cap: 256,
        rung: u32::MAX,
        rope_appended: true,
    },
    AttnShape {
        geom: (4, 1, 128),
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
            // The K rotation's tables, at the rotated K's own rows and width.
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
    // granite's geometry as the facts' `Attention` (uniform classes): the op carries only
    // its class, and every consumer reads the class's numbers off this one value.
    let attention = scratchy_subtile::fixtures::attention_at(nqh, nkvh, hd, scale);
    let attn_op = scratchy_subtile::lower::OpDesc {
        op: scratchy_subtile::subtile_ir::SubOp::attn_decode(
            attention,
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
                        attention,
                        // A uniform fixture: the base class states the same head dim.
                        scratchy_subtile::subtile_ir::RopeClass::Base,
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

/// spyre-inference's paged kernels attend causally, so a sliding-window attention (gemma's local
/// layers) is refused by name rather than run as causal.
#[test]
fn a_sliding_window_attention_is_refused() {
    let s = ATTN_SHAPES[0];
    let mut ir = attn_ir(s, 0.015625);
    let node = ir.nodes.last_mut().expect("the attention");
    let scratchy_subtile::subtile_ir::SubOp::AttnDecode { mask, .. } = &mut node.op else {
        panic!("the last node is the attention");
    };
    *mask = scratchy_subtile::subtile_ir::AttnMask::SlidingWindow {
        window: std::num::NonZeroU32::new(4096).expect("a window"),
    };
    let node = ir.nodes.last().expect("the attention");
    let Err(e) = scratchy_triton_splice::lower_attn::<ktir_superdsc::collective::TpWorld<1>, _>(
        node,
        &ir,
        scratchy_subtile::subtile_ir::AttnScaleSite::InKernel,
        s.active_cap(),
        // One request of the graph's `mq` rows — the walk this refusal gates.
        pair_of_one_request(s.mq),
        PagedRows::Sequence,
    ) else {
        panic!("a sliding window must be refused");
    };
    assert!(e.contains("sliding window"), "{e}");
}

/// ⛔ A POOL NO BIGGER THAN A CHUNK IS REFUSED BY NAME — spyre-inference's own eligibility
/// (`_batched_decode_preconditions_met`): a gather of every page the pool holds faults the card, and
/// their per-sequence LOOP serves that step, which this port does not have. One request's decode is
/// a batch of one, so the bound is its too: a chunk takes `min(32, blocks)` pages, and the ceiling
/// rung walks the whole pool, so it fires exactly where the pool is at most 32 pages — a context of
/// at most 4096 positions. A pool past that (granite's own) splices.
#[test]
fn a_pool_no_bigger_than_a_chunk_is_refused() {
    const P: u32 = scratchy_triton_splice::SchedulerBlockTokens::REFERENCE.get();
    let at_pages = |pages: u32| AttnShape {
        geom: (32, 8, 64),
        mq: 1,
        cap: pages * P,
        rung: 0,
        rope_appended: true,
    };
    // 16 pages (a 2048-position context): the ceiling rung walks all 16 and a chunk gathers 16.
    let small = at_pages(16);
    let ir = attn_ir(small, 0.015625);
    let node = ir.nodes.last().expect("the attention");
    let Err(e) = scratchy_triton_splice::lower_attn::<ktir_superdsc::collective::TpWorld<1>, _>(
        node,
        &ir,
        scratchy_subtile::subtile_ir::AttnScaleSite::InKernel,
        small.active_cap(),
        // The one-request decode's own pair: one row that is one request.
        batch_pair_of_one_row_each(small.mq),
        PagedRows::Batch,
    ) else {
        panic!("a chunk gathering the whole pool must be refused");
    };
    assert!(
        e.contains("gathers 16 page(s) a chunk, and the pool holds 16")
            && e.contains("per-sequence loop"),
        "the refusal must name the pool and their loop: {e}"
    );
    // 64 pages: a chunk takes 32 of them, so the same one-request decode splices.
    let big = at_pages(64);
    let ir = attn_ir(big, 0.015625);
    let node = ir.nodes.last().expect("the attention");
    scratchy_triton_splice::lower_attn::<ktir_superdsc::collective::TpWorld<1>, _>(
        node,
        &ir,
        scratchy_subtile::subtile_ir::AttnScaleSite::InKernel,
        big.active_cap(),
        // The one-request decode's own pair: one row that is one request.
        batch_pair_of_one_row_each(big.mq),
        PagedRows::Batch,
    )
    .expect("a pool wider than one chunk splices");
}

#[test]
fn the_paged_attention_splices_with_its_contract() {
    for &s in ATTN_SHAPES {
        let scale = if s.geom.2 == 64 { 0.015625 } else { 0.0078125 };
        let ir = attn_ir(s, scale);
        let node = ir.nodes.last().unwrap();
        let at = format!("geom={:?} mq={} rung={}", s.geom, s.mq, s.rung);
        let spliced = scratchy_triton_splice::lower_attn::<ktir_superdsc::collective::TpWorld<1>, _>(
            node,
            &ir,
            scratchy_subtile::subtile_ir::AttnScaleSite::InKernel,
            s.active_cap(),
            // One request of the graph's `mq` rows — the S=1 extent pair.
            pair_of_one_request(s.mq),
            PagedRows::Sequence,
        );
        // ⛔ A PRE-POPULATED CACHE IS REFUSED: the walk reads the step's keys from the pools, and
        // only this forward's RopeAppend scatters them there.
        if !s.rope_appended {
            let Err(e) = spliced else {
                panic!("{at}: a cache no RopeAppend writes must be refused");
            };
            assert!(
                e.contains("not written by this forward's RopeAppend"),
                "{at}: {e}"
            );
            continue;
        }
        let spliced = spliced.unwrap_or_else(|e| panic!("splice compiled {at}: {e}"));
        // ⛔ THE NAME LAW — `attn_s{id}`.
        assert_eq!(
            spliced.op_name,
            format!("attn_s{}", node.id.index()),
            "{at}"
        );
        let k = spliced
            .ktir
            .as_ref()
            .expect("spliced op carries its program");
        assert_eq!(
            k.program,
            ktir_superdsc::ktir_node::Program::PagedAttn,
            "{at}"
        );
        // spyre-inference's operands, in their order: query, query_row_index, the K and V pools, the
        // page index table, `kv_row_pool` (the decode kernel's, at one query row), the mask stack
        // and out — the step's sources at the ids the one law (`PagedStepTids`) states.
        let tids = scratchy_triton_splice::PagedStepTids::RESERVED;
        let bound: Vec<u32> = k.bindings.iter().map(|b| b.get()).collect();
        let mut want = vec![
            node.inputs[0].tensor.index() as u32,
            tids.query_rows,
            node.inputs[1].tensor.index() as u32,
            node.inputs[2].tensor.index() as u32,
            tids.page_table,
        ];
        if s.mq == 1 {
            want.push(tids.kv_row_pool);
        }
        want.extend([tids.mask, node.output.tensor.index() as u32]);
        assert_eq!(bound, want, "{at}");
        assert_eq!(k.mask.map(|m| m.get()), Some(tids.mask), "{at}");
        // ⭐ THE CARD'S HALF, through the production door under the bundle's own layout: the walk
        // lowers at every geometry — its body at trip 0, marked for the card's per-trip relaunch
        // exactly when the walk loops (a one-page pool's walk is one trip, which the front end
        // promotes to straight-line code). At a head two sticks wide the decode's pages are permuted
        // per kv head and its output row back, and the prefill's query and output are relaid
        // head-major and back, its page gathered in pieces.
        let layout = triton_ktir_superdsc::layout::for_node(k)
            .unwrap_or_else(|e| panic!("{at}: layout: {e:?}"));
        scratchy_triton_splice::allocate_paged_cache(k, &layout)
            .unwrap_or_else(|e| panic!("{at}: {e}"));
        let mut sym = 0i64;
        let ops = door_lower(k, &mut sym, Some(&layout), None)
            .unwrap_or_else(|e| panic!("{at}: the walk lowers: {}", e.message));
        assert!(!ops.is_empty(), "{at}: the walk emitted nothing");
        let walking = ktir_superdsc::emit::loop_fold::flatten(k)
            .expect("the program flattens")
            .is_some();
        assert_eq!(
            ops.iter().any(|e| e.walk.is_some()),
            walking,
            "{at}: the body is marked for the relaunch exactly when the walk loops"
        );
        // And a walk's body never fuses away, so the attention has no fold-fused twin: the bake
        // would compile its split program twice, under two names.
        assert!(
            !scratchy_target_spyre::lower_subtile_tape_to_superdsc::has_fused_twin(&ops),
            "{at}: the paged attention has a fold-fused twin"
        );
    }
}

/// ⭐ THE PAGED STEP, EXECUTED — the `RopeAppend`'s two programs (the rotation, then the KV write's
/// scatter into the pools) and the `AttnDecode`'s page walk, the three REAL spliced programs run in
/// order through the production session entry, staged by the worker's own law
/// (`scratchy_target_spyre::paged_step`), against an f64 softmax over the same keys: the cached
/// prefix the pools already hold, and this step's own rows, causally.
///
/// The rotation tables are cos 1 / sin 0, so the rotated K is the K and the reference reads the new
/// block straight from its sources; the rotation itself has its own goldens. Every step's blocks
/// are the pool's pages REVERSED, and the multi-page shapes run a TWO-PAGE pool, so the walk's pages
/// come through the table and not at their block's own index. Pools and contexts are stated in
/// pages of `SchedulerBlockTokens::REFERENCE`, so each shape keeps its place against the page boundaries.
///
/// ⛔ AND IT IS THE KERNEL'S OWN OPS THAT RUN: each shape asserts none of the emulator's native
/// attention recognizers claims the walk, so a pass here is the kernel's arithmetic.
#[test]
#[cfg(feature = "spyre-emu")]
fn spliced_paged_step_executes_the_real_programs() {
    // One page, and a position in pages.
    const P: u32 = scratchy_triton_splice::SchedulerBlockTokens::REFERENCE.get();
    let at = |pages: u32, offset: u32| (pages * P + offset) as usize;
    // (shape, cached positions before the step)
    for (s, cached) in [
        // Decode at a one-page pool, half a page cached.
        (
            AttnShape {
                geom: (32, 8, 64),
                mq: 1,
                cap: P,
                rung: 0,
                rope_appended: true,
            },
            at(0, P / 2),
        ),
        // Decode, head dim 128, past the first page.
        (
            AttnShape {
                geom: (32, 8, 128),
                mq: 1,
                cap: 2 * P,
                rung: 0,
                rope_appended: true,
            },
            at(1, P / 2),
        ),
        // Decode over a TWO-PAGE pool, the context spanning both pages.
        (
            AttnShape {
                geom: (32, 8, 64),
                mq: 1,
                cap: 2 * P,
                rung: 0,
                rope_appended: true,
            },
            at(1, P / 2 + 3),
        ),
        // Decode at TWO kv heads (qwen2.5-0.5b's attention, 14 query heads): the step's write is
        // two rows, under Triton's scatter floor of 8, so the KV write runs row by row.
        (
            AttnShape {
                geom: (14, 2, 64),
                mq: 1,
                cap: P,
                rung: 0,
                rope_appended: true,
            },
            at(0, P / 2 + 8),
        ),
        // The first prompt chunk: nothing cached, 31 rows.
        (
            AttnShape {
                geom: (32, 8, 64),
                mq: 31,
                cap: P,
                rung: u32::MAX,
                rope_appended: true,
            },
            0,
        ),
        // A prompt chunk over a cached prefix, its rows crossing into the second page.
        (
            AttnShape {
                geom: (32, 8, 64),
                mq: 31,
                cap: 2 * P,
                rung: 0,
                rope_appended: true,
            },
            at(0, P / 2 + 8),
        ),
        // The PREFIX-FREE rung running a chunk over a cached prefix: its walk is the one page its
        // rows fill, so a chunk whose prefix sits in that page is the rung's to run — the reach law
        // the worker routes a continuation chunk by (`PagedStep::reach`).
        (
            AttnShape {
                geom: (32, 8, 64),
                mq: 31,
                cap: P,
                rung: u32::MAX,
                rope_appended: true,
            },
            at(0, P - 31 - 5),
        ),
        // The same over a TWO-PAGE pool: the one-block walk reads the request's first block, which
        // the reversed table puts on the pool's LAST page.
        (
            AttnShape {
                geom: (32, 8, 64),
                mq: 31,
                cap: 2 * P,
                rung: u32::MAX,
                rope_appended: true,
            },
            at(0, P - 31 - 5),
        ),
    ] {
        execute_one_paged_step(s, cached);
    }
    // ⭐ THE PER-RANK GEOMETRIES (TP phase 1): the SAME prefill chunk shapes — the first
    // chunk (nothing cached), a chunk over a cached prefix crossing into the second page,
    // the prefix-free rung over a cached prefix, and the prefix-free rung over a two-page
    // pool — at granite's per-rank shapes (16, 4) / (8, 2) / (4, 1) at BOTH head dims.
    // The chunk rung is the geometry's own head dim's (m=31 at hd 64, m=96 at hd 128),
    // the rung the card bakes for that model.
    let per_rank_step = |nqh: u32, nkvh: u32, hd: u32| AttnShape {
        geom: (nqh, nkvh, hd),
        mq: if hd == 64 { 31 } else { 96 },
        cap: P,
        rung: u32::MAX,
        rope_appended: true,
    };
    for (nqh, nkvh) in [(16u32, 4u32), (8, 2), (4, 1)] {
        for hd in [64u32, 128] {
            let base = per_rank_step(nqh, nkvh, hd);
            for (s, cached) in [
                // The first prompt chunk: nothing cached.
                (base, 0),
                // A chunk over a cached prefix, its rows crossing into the second page.
                (
                    AttnShape {
                        cap: 2 * P,
                        rung: 0,
                        ..base
                    },
                    at(0, P / 2 + 8),
                ),
                // The prefix-free rung running a chunk over a cached prefix.
                (base, at(0, P - base.mq - 5)),
                // The same over a TWO-PAGE pool.
                (AttnShape { cap: 2 * P, ..base }, at(0, P - base.mq - 5)),
            ] {
                execute_one_paged_step(s, cached);
            }
        }
    }
}

/// ⭐⭐ THE EXTENT PAIR MEETS THE GRAPH — the S=2 shape M2's packed prefill rung will bake, at the
/// splice's own boundary, WITHOUT running it (the numeric cell is the later step's): a 128-row
/// graph minted `(64, 2)` SPLICES — `NUM_TOKENS` stays the graph's 128 rows while
/// `PADDED_QUERY_LEN` and the walk become one request's 64 (the compile fixture in the splice's
/// own roster states that exact constexpr pair) — and a mint whose held product is NOT the
/// graph's rows is REFUSED BY NAME, the one place the two are proven to agree. The S=1 spelling
/// (`(128, ONE)`) splices the same graph too: at one request the two extents agree, which is why
/// one `mq` served until now.
#[test]
#[cfg(feature = "spyre-emu")]
fn the_extent_pair_meets_the_graph_at_the_splice() {
    let s = AttnShape {
        geom: (32, 8, 64),
        mq: 128,
        cap: 2 * scratchy_triton_splice::SchedulerBlockTokens::REFERENCE.get(),
        rung: u32::MAX, // ActiveCap::NONE — a start==0 chunk, the packed rung's own walk
        rope_appended: true,
    };
    let ir = attn_ir(s, 0.125f32);
    let attn = &ir.nodes[1];
    let splice = |pair: ktir_superdsc::sdsc_abstract::QueryExtent| {
        scratchy_triton_splice::lower_attn::<ktir_superdsc::collective::TpWorld<1>, _>(
            attn,
            &ir,
            scratchy_subtile::subtile_ir::AttnScaleSite::InKernel,
            s.active_cap(),
            pair,
            PagedRows::Sequence,
        )
    };
    // The S=2 mint: two requests of 64 rows over the 128-row graph.
    let packed = ktir_superdsc::sdsc_abstract::QueryExtent::of_baked_rung(
        ktir_superdsc::sdsc_abstract::RungWidth::of_baked_rows(64)
            .expect("the rung's width is nonzero"),
        ktir_superdsc::sdsc_abstract::RungCap::of_baked_seqs(2)
            .expect("the rung's capacity is nonzero"),
    )
    .expect("128 rows fit a u32");
    let e = splice(packed)
        .unwrap_or_else(|e| panic!("the packed (64, 2) mint splices the 128-row graph: {e}"));
    assert!(
        e.ktir
            .as_ref()
            .is_some_and(|k| matches!(k.program, ktir_superdsc::ktir_node::Program::PagedAttn)),
        "the packed mint is the prefill walk's own program"
    );
    // The S=1 mint of the SAME graph — one request of all 128 rows — still splices.
    splice(pair_of_one_request(128))
        .unwrap_or_else(|e| panic!("the S=1 mint splices the same graph: {e}"));
    // ⛔ AND A MINT THAT DOES NOT DESCRIBE THE GRAPH IS REFUSED, BY NAME, with both extents
    // stated — never a silently-wrong `PADDED_QUERY_LEN`.
    let mismatch = ktir_superdsc::sdsc_abstract::QueryExtent::of_baked_rung(
        ktir_superdsc::sdsc_abstract::RungWidth::of_baked_rows(64)
            .expect("the rung's width is nonzero"),
        ktir_superdsc::sdsc_abstract::RungCap::of_baked_seqs(3)
            .expect("the rung's capacity is nonzero"),
    )
    .expect("192 rows fit a u32");
    let err = splice(mismatch)
        .err()
        .expect("a 192-row mint does not describe a 128-row graph");
    assert!(
        err.contains("the graph's 128 row(s) are not the extent pair's 192"),
        "the refusal names both extents: {err}"
    );
    // ⛔ THE PAGED STEP'S OWN AGREEMENT — `paged_step_of_ir` mints the wiring's step (the
    // layout sizes the tables by the pair), so the SAME mismatched pair is refused at the step
    // mint too, naming the graph's rows and the pair's — never a silently-wrong per-request
    // width the card would stage by.
    let err = scratchy_target_spyre::lower_subtile_tape_to_ktir::paged_step_of_ir(
        &ir,
        s.active_cap(),
        PagedRows::Sequence,
        mismatch,
    )
    .expect_err("the step mint refuses a pair the graph disagrees with");
    assert!(
        err.0
            .contains("the graph runs 128 query row(s) but the rung's minted extent states 192"),
        "the refusal names the graph's rows and the pair's: {}",
        err.0
    );
}

/// The S=1 extent pair a SEQUENCE walk states: one request of the graph's `mq` rows. Pure arithmetic,
/// read by the door tests every build runs, so it is not behind the emulator.
fn pair_of_one_request(mq: u32) -> ktir_superdsc::sdsc_abstract::QueryExtent {
    ktir_superdsc::sdsc_abstract::QueryExtent::of_baked_rung(
        ktir_superdsc::sdsc_abstract::RungWidth::of_baked_rows(mq)
            .expect("the shape's row count is nonzero"),
        ktir_superdsc::sdsc_abstract::RungCap::ONE,
    )
    .expect("one request's rows fit a u32")
}

/// The extent pair a BATCH walk states: `rows` rows that are `rows` requests, one row
/// each — the pair's held product is the graph's row count, which the batched kernel
/// arm reads.
fn batch_pair_of_one_row_each(rows: u32) -> ktir_superdsc::sdsc_abstract::QueryExtent {
    ktir_superdsc::sdsc_abstract::QueryExtent::of_baked_rung(
        ktir_superdsc::sdsc_abstract::RungWidth::of_baked_rows(1).expect("one row is nonzero"),
        ktir_superdsc::sdsc_abstract::RungCap::of_baked_seqs(rows)
            .expect("the batch's sequence count is nonzero"),
    )
    .expect("the batch's rows fit a u32")
}

/// `n` values uniform in `[-amp, amp)` from a fixed LCG — a stimulus both sides read as the same
/// f16 bytes (the session narrows each to f16; the reference reads the narrowed value back).
#[cfg(feature = "spyre-emu")]
fn attn_stimulus(n: usize, seed: u64, amp: f32) -> Vec<f32> {
    let mut x = seed;
    (0..n)
        .map(|_| {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = (x >> 40) as f32 / (1u64 << 24) as f32;
            half::f16::from_f32((2.0 * u - 1.0) * amp).to_f32()
        })
        .collect()
}

#[cfg(feature = "spyre-emu")]
fn execute_one_paged_step(s: AttnShape, cached: usize) {
    use scratchy_target_spyre::paged_step::{StepRows, pool_of_rows, step_sources};
    use std::borrow::Cow;

    let (nqh, nkvh, hd) = s.geom;
    let scale = if hd == 64 { 0.015625f32 } else { 0.0078125 };
    let ir = attn_ir(s, scale);
    let (rope, attn) = (&ir.nodes[0], &ir.nodes[1]);
    let at = format!(
        "geom={:?} mq={} cap={} cached={cached}",
        s.geom, s.mq, s.cap
    );
    let mut programs =
        scratchy_triton_splice::lower_all::<ktir_superdsc::collective::TpWorld<1>, _>(
            rope,
            &ir,
            false,
            pair_of_one_request(s.mq),
        )
        .unwrap_or_else(|e| panic!("the RopeAppend spliced {at}: {e}"));
    programs.push(
        scratchy_triton_splice::lower_attn(
            attn,
            &ir,
            scratchy_subtile::subtile_ir::AttnScaleSite::InKernel,
            s.active_cap(),
            // One request of the graph's `mq` rows — the S=1 extent pair this walk's
            // constexprs state (`NUM_TOKENS` = mq, `PADDED_QUERY_LEN` = mq).
            pair_of_one_request(s.mq),
            PagedRows::Sequence,
        )
        .unwrap_or_else(|e| panic!("the walk spliced {at}: {e}")),
    );
    assert_eq!(programs.len(), 3, "{at}: rope, KV write, walk");
    let walk = programs[2]
        .ktir
        .as_ref()
        .expect("the walk carries its program");
    let t_out = u64::from(
        walk.bindings
            .last()
            .expect("out is the walk's last operand")
            .get(),
    );
    let groups: Vec<
        scratchy_target_spyre::bundle_code::LaunchGroup<
            'static,
            ktir_superdsc::collective::TpWorld<1>,
        >,
    > =
        programs
            .into_iter()
            .map(|e| {
                let k = e.ktir.expect("every spliced op carries its program");
                let args = k
                    .func
                    .arguments
                    .iter()
                    .map(|(ssa, _)| {
                        (
                            *ssa,
                            scratchy_target_spyre::bundle_code::PlaceId::Act(
                                k.bindings[ssa.slot()].get(),
                            ),
                        )
                    })
                    .collect::<Vec<_>>();
                scratchy_target_spyre::bundle_code::LaunchGroup::<
                    ktir_superdsc::collective::TpWorld<1>,
                > {
                    kv: Default::default(),
                    programs: Cow::Owned(vec![scratchy_target_spyre::bundle_code::LaunchProgram {
                        func: k.func,
                        program: k.program,
                        args: Cow::Owned(args),
                        lx_pin_window_bytes: Some(0),
                    }]),
                    init_binary: scratchy_target_spyre::bundle_code::BakedImage::Raw(
                        Cow::Borrowed(&[]),
                    ),
                    job_bin_ptr: 0,
                    correction: Cow::Borrowed(&[]),
                }
            })
            .collect();

    // The step: `cached` keys already in the pools, this step's `mq` rows after them; the request's
    // blocks are the pool's pages REVERSED.
    let step = scratchy_target_spyre::lower_subtile_tape_to_ktir::paged_step_of_ir(
        &ir,
        s.active_cap(),
        PagedRows::Sequence,
        pair_of_one_request(s.mq),
    )
    .expect("the paged step derives")
    .expect("the graph has attention");
    let mq = s.mq as usize;
    let rows = StepRows::new(cached, mq).expect("the golden step holds its query rows");
    let blocks: Vec<u32> = (0..step.pages).rev().collect();
    let (q_width, kv_width, hd) = ((nqh * hd) as usize, (nkvh * hd) as usize, hd as usize);
    let kc = attn_stimulus(cached * kv_width, 2, 6.0);
    let vc = attn_stimulus(cached * kv_width, 3, 1.0);
    let q = attn_stimulus(mq * q_width, 1, 6.0);
    let kd = attn_stimulus(mq * kv_width, 4, 6.0);
    let vd = attn_stimulus(mq * kv_width, 5, 1.0);
    let src = |i: usize| u64::from(rope.inputs[i].tensor.index() as u32);
    let srcs = step_sources(&step, &blocks, &rows).unwrap_or_else(|e| panic!("{at}: {e}"));
    let typed: Vec<(
        usize,
        scratchy_target_spyre::HostBytes,
        scratchy_tensors::DType,
        Vec<usize>,
    )> = srcs
        .int32
        .into_iter()
        .map(|(tid, bytes, shape)| {
            (
                tid as usize,
                scratchy_target_spyre::HostBytes::Owned(bytes),
                scratchy_tensors::DType::I32,
                shape,
            )
        })
        .collect();
    let (mask_tid, mask, mask_shape) = srcs.mask;
    let mut sources = vec![
        (
            u64::from(attn.inputs[0].tensor.index() as u32),
            q.clone(),
            vec![mq, q_width],
        ),
        (src(0), kd.clone(), vec![mq, kv_width]),
        (src(1), vec![1.0; mq * kv_width], vec![mq, kv_width]),
        (src(2), vec![0.0; mq * kv_width], vec![mq, kv_width]),
        (src(3), vd.clone(), vec![mq, kv_width]),
        (u64::from(mask_tid), mask, mask_shape),
    ];
    // The pools' cached rows, as the prefix of each pool they land in (`pool_of_rows`); nothing
    // cached stages nothing, and the session's zeroed allocation is the pool.
    for (i, cache) in [(4usize, &kc), (5, &vc)] {
        let pool = pool_of_rows(&step, hd, &blocks, cache, cached).unwrap();
        if !pool.is_empty() {
            let rows = pool.len() / kv_width;
            sources.push((src(i), pool, vec![rows, kv_width]));
        }
    }
    let mut session =
        scratchy_target_spyre::runner::SpyreSession::new_multi(&[(&groups, &[t_out])], typed)
            .unwrap_or_else(|e| panic!("{at}: the session builds: {e}"));
    let out = session
        .run_step(0, sources, &[(t_out, 0)])
        .unwrap_or_else(|e| panic!("{at}: the step runs: {e}"));
    let got = &out[&t_out];
    assert_eq!(got.len(), mq * q_width, "{at}");

    // The f64 reference: row r (position cached + r) attends every cached key and the step's own
    // rows up to itself.
    let group_size = (nqh / nkvh) as usize;
    let dot = |a: &[f32], b: &[f32]| {
        a.iter()
            .zip(b)
            .map(|(x, y)| f64::from(*x) * f64::from(*y))
            .sum::<f64>()
    };
    let mut max_err = 0.0f64;
    for r in 0..mq {
        for h in 0..nqh as usize {
            let kvh = h / group_size;
            let qrow = &q[r * q_width + h * hd..][..hd];
            let mut keys: Vec<(f64, &[f32])> = Vec::new();
            for t in 0..cached {
                let k = &kc[t * kv_width + kvh * hd..][..hd];
                let v = &vc[t * kv_width + kvh * hd..][..hd];
                keys.push((f64::from(scale) * dot(qrow, k), v));
            }
            for t in 0..=r {
                let k = &kd[t * kv_width + kvh * hd..][..hd];
                let v = &vd[t * kv_width + kvh * hd..][..hd];
                keys.push((f64::from(scale) * dot(qrow, k), v));
            }
            let m = keys
                .iter()
                .map(|(s, _)| *s)
                .fold(f64::NEG_INFINITY, f64::max);
            let l: f64 = keys.iter().map(|(s, _)| (s - m).exp()).sum();
            for d in 0..hd {
                let want: f64 = keys
                    .iter()
                    .map(|(s, v)| (s - m).exp() / l * f64::from(v[d]))
                    .sum();
                let g = f64::from(got[r * q_width + h * hd + d]);
                max_err = max_err.max((g - want).abs());
            }
        }
    }
    println!("  paged step {at}: max abs err {max_err:.3e} (gate 0.05)");
    assert!(
        max_err < 0.05,
        "the REAL paged step diverged from the f64 softmax ({at}): max abs err {max_err}"
    );
}

/// ⭐ A DECODE BATCH, EXECUTED — one token of each of several sequences through the `RopeAppend`'s
/// two programs and `batched_decode_head_major`, run through the production session entry at
/// granite's geometry, staged by `batch_sources` (the worker's law), against an f64 softmax over
/// each sequence's OWN keys. Every sequence holds its own pages (the pool's pages reversed, dealt
/// out in turn), so a kernel that reads one sequence's pages for another, or one query row for
/// every entry, misses. Four chunkings: two chunks (each block slot its own softmax, merged after
/// the walk), one chunk (one maximum per sequence), and two chunks under Triton's gather floor in
/// pages, gathered by kv head (two sequences on the 2-page rung, and on the 1-page rung: the shape
/// two concurrent short requests decode at), and one request's decode as a batch of one, deep and
/// short. One bundle row past the real sequences is padded, as a bucketed batch is.
#[test]
#[cfg(feature = "spyre-emu")]
fn spliced_decode_batch_executes_the_real_programs() {
    const P: u32 = scratchy_triton_splice::SchedulerBlockTokens::REFERENCE.get();
    let at_hd = |seqs: u32, walk: u32, hd: u32| AttnShape {
        geom: (32, 8, hd),
        mq: seqs,
        cap: 64 * P,
        rung: walk * P,
        rope_appended: true,
    };
    // granite-3.1-2b's head dim; the 8b's 128 is `at_hd`.
    let shape = |seqs: u32, walk: u32| at_hd(seqs, walk, 64);
    for (s, contexts) in [
        // 4-sequence bucket over the 64-page walk: J = 8, C = 8.
        (
            AttnShape {
                rung: 0,
                ..shape(4, 64)
            },
            vec![150usize, 40, 100],
        ),
        // The 4-page rung: J = 4, C = 1.
        (shape(4, 4), vec![150, 40, 100]),
        // Two sequences on the 2-page rung: J = 2 is 4 pages, under the floor, so its 32 kv rows.
        (shape(2, 2), vec![100, 30]),
        // ...and on the 1-page rung: J = 1, 2 pages, 16 kv rows.
        (shape(2, 1), vec![100, 30]),
        // ONE request's decode, a batch of one (spyre-inference's sequence buckets start at 1): over
        // the 64-page walk, J = 32, C = 2, whole pages; on the 1-page rung, its page's 8 kv rows.
        (
            AttnShape {
                rung: 0,
                ..shape(1, 64)
            },
            vec![4000],
        ),
        (shape(1, 1), vec![100]),
        // ...and both of those at head dim 128 (granite-3.1-8b), whose two-stick head rows take the
        // relayout the one-stick forms never build: the 8b is where the per-sequence route's single
        // request decoded garbage, so a batch of one is executed there too.
        (
            AttnShape {
                rung: 0,
                ..at_hd(1, 64, 128)
            },
            vec![4000],
        ),
        (at_hd(1, 1, 128), vec![100]),
    ] {
        execute_one_decode_batch(s, &contexts);
    }
    // ⭐ THE PER-RANK GEOMETRIES (TP phase 1): the SAME chunkings — 4 sequences over the
    // 64-page walk, the 4-page rung, 2 sequences on the 2-page and 1-page rungs, and a
    // batch of one deep and short — at granite's per-rank shapes (16, 4) / (8, 2) / (4, 1)
    // at BOTH head dims. Production decode always takes the BATCHED kernel, so these are
    // the rows TP's per-rank decode actually runs; a refusal or divergence here is a
    // finding against `min_kv_heads_per_rank`, never a loosened row.
    let per_rank = |nqh: u32, nkvh: u32, hd: u32, seqs: u32, walk: u32| AttnShape {
        geom: (nqh, nkvh, hd),
        mq: seqs,
        cap: 64 * P,
        rung: walk * P,
        rope_appended: true,
    };
    for (nqh, nkvh) in [(16u32, 4u32), (8, 2), (4, 1)] {
        for hd in [64u32, 128] {
            for (s, contexts) in [
                (per_rank(nqh, nkvh, hd, 4, 64), vec![150usize, 40, 100]),
                (per_rank(nqh, nkvh, hd, 4, 4), vec![150, 40, 100]),
                (per_rank(nqh, nkvh, hd, 2, 2), vec![100, 30]),
                (per_rank(nqh, nkvh, hd, 2, 1), vec![100, 30]),
                (
                    AttnShape {
                        rung: 0,
                        ..per_rank(nqh, nkvh, hd, 1, 64)
                    },
                    vec![4000],
                ),
                (per_rank(nqh, nkvh, hd, 1, 1), vec![100]),
            ] {
                execute_one_decode_batch(s, &contexts);
            }
        }
    }
}

#[cfg(feature = "spyre-emu")]
fn execute_one_decode_batch(s: AttnShape, contexts: &[usize]) {
    use scratchy_target_spyre::paged_step::{BatchSeq, batch_sources, pool_of_rows};
    use std::borrow::Cow;

    let (nqh, nkvh, hd) = s.geom;
    let scale = if hd == 64 { 0.015625f32 } else { 0.0078125 };
    let ir = attn_ir(s, scale);
    let (rope, attn) = (&ir.nodes[0], &ir.nodes[1]);
    let at = format!(
        "geom={:?} B={} cap={} rung={} contexts={contexts:?}",
        s.geom, s.mq, s.cap, s.rung
    );
    let mut programs =
        scratchy_triton_splice::lower_all::<ktir_superdsc::collective::TpWorld<1>, _>(
            rope,
            &ir,
            true,
            batch_pair_of_one_row_each(s.mq),
        )
        .unwrap_or_else(|e| panic!("the RopeAppend spliced {at}: {e}"));
    programs.push(
        scratchy_triton_splice::lower_attn(
            attn,
            &ir,
            scratchy_subtile::subtile_ir::AttnScaleSite::InKernel,
            s.active_cap(),
            // The batch's own pair: `mq` rows that are `mq` requests — one row each
            // (the batched kernel arm reads the graph's rows, which the pair's held
            // product states).
            batch_pair_of_one_row_each(s.mq),
            PagedRows::Batch,
        )
        .unwrap_or_else(|e| panic!("the batched walk spliced {at}: {e}")),
    );
    assert_eq!(programs.len(), 3, "{at}: rope, KV write, batched walk");
    let walk = programs[2]
        .ktir
        .as_ref()
        .expect("the walk carries its program");
    let tids = scratchy_triton_splice::PagedStepTids::RESERVED;
    let bound: Vec<u32> = walk.bindings.iter().map(|b| b.get()).collect();
    assert!(
        [tids.rep_rows, tids.chunk_pages, tids.chunk_masks]
            .iter()
            .all(|t| bound.contains(t))
            && !bound.contains(&tids.page_table),
        "{at}: a decode batch runs the batched kernel, over the batch's operands: {bound:?}"
    );
    let t_out = u64::from(
        walk.bindings
            .last()
            .expect("out is the walk's last operand")
            .get(),
    );
    let groups: Vec<
        scratchy_target_spyre::bundle_code::LaunchGroup<
            'static,
            ktir_superdsc::collective::TpWorld<1>,
        >,
    > =
        programs
            .into_iter()
            .map(|e| {
                let k = e.ktir.expect("every spliced op carries its program");
                let args = k
                    .func
                    .arguments
                    .iter()
                    .map(|(ssa, _)| {
                        (
                            *ssa,
                            scratchy_target_spyre::bundle_code::PlaceId::Act(
                                k.bindings[ssa.slot()].get(),
                            ),
                        )
                    })
                    .collect::<Vec<_>>();
                scratchy_target_spyre::bundle_code::LaunchGroup::<
                    ktir_superdsc::collective::TpWorld<1>,
                > {
                    kv: Default::default(),
                    programs: Cow::Owned(vec![scratchy_target_spyre::bundle_code::LaunchProgram {
                        func: k.func,
                        program: k.program,
                        args: Cow::Owned(args),
                        lx_pin_window_bytes: Some(0),
                    }]),
                    init_binary: scratchy_target_spyre::bundle_code::BakedImage::Raw(
                        Cow::Borrowed(&[]),
                    ),
                    job_bin_ptr: 0,
                    correction: Cow::Borrowed(&[]),
                }
            })
            .collect();

    let step = scratchy_target_spyre::lower_subtile_tape_to_ktir::paged_step_of_ir(
        &ir,
        s.active_cap(),
        PagedRows::Batch,
        batch_pair_of_one_row_each(s.mq),
    )
    .expect("the paged step derives")
    .expect("the graph has attention");
    let (b, n) = (s.mq as usize, contexts.len());
    let slots = step.page_slots as usize;
    let (q_width, kv_width, hd) = ((nqh * hd) as usize, (nkvh * hd) as usize, hd as usize);
    // Each sequence's blocks: the pool's pages reversed, dealt out in turn.
    let mut free = (0..step.pages).rev();
    let blocks: Vec<Vec<u32>> = contexts
        .iter()
        .map(|&c| (&mut free).take((c + 1).div_ceil(slots)).collect())
        .collect();
    // Each sequence's cached keys and values, and its one new token; a pad row repeats the last
    // real sequence's token (the worker's pad law).
    let kc: Vec<Vec<f32>> = contexts
        .iter()
        .enumerate()
        .map(|(i, &c)| attn_stimulus(c * kv_width, 20 + i as u64, 6.0))
        .collect();
    let vc: Vec<Vec<f32>> = contexts
        .iter()
        .enumerate()
        .map(|(i, &c)| attn_stimulus(c * kv_width, 40 + i as u64, 1.0))
        .collect();
    let real = |w: usize, seed: u64, amp: f32| -> Vec<f32> {
        let rows = attn_stimulus(n * w, seed, amp);
        (0..b)
            .flat_map(|r| rows[r.min(n - 1) * w..][..w].to_vec())
            .collect()
    };
    let q = real(q_width, 1, 6.0);
    let kd = real(kv_width, 4, 6.0);
    let vd = real(kv_width, 5, 1.0);
    let seqs: Vec<BatchSeq<'_>> = contexts
        .iter()
        .zip(&blocks)
        .map(|(&c, bl)| BatchSeq { blocks: bl, pos: c })
        .collect();
    let srcs = batch_sources(&step, &seqs).unwrap_or_else(|e| panic!("{at}: {e}"));
    let typed: Vec<(
        usize,
        scratchy_target_spyre::HostBytes,
        scratchy_tensors::DType,
        Vec<usize>,
    )> = srcs
        .int32
        .into_iter()
        .map(|(tid, bytes, shape)| {
            (
                tid as usize,
                scratchy_target_spyre::HostBytes::Owned(bytes),
                scratchy_tensors::DType::I32,
                shape,
            )
        })
        .collect();
    let (mask_tid, mask, mask_shape) = srcs.mask;
    let src = |i: usize| u64::from(rope.inputs[i].tensor.index() as u32);
    let mut sources = vec![
        (
            u64::from(attn.inputs[0].tensor.index() as u32),
            q.clone(),
            vec![b, q_width],
        ),
        (src(0), kd.clone(), vec![b, kv_width]),
        (src(1), vec![1.0; b * kv_width], vec![b, kv_width]),
        (src(2), vec![0.0; b * kv_width], vec![b, kv_width]),
        (src(3), vd.clone(), vec![b, kv_width]),
        (u64::from(mask_tid), mask, mask_shape),
    ];
    // The pools: every sequence's cached rows laid into ITS pages (`pool_of_rows`), page by page.
    let page_elems = step.nkvh as usize * slots * hd;
    for (i, cache) in [(4usize, &kc), (5, &vc)] {
        let mut pool = vec![0.0f32; step.pages as usize * page_elems];
        for (sq, (&c, bl)) in contexts.iter().zip(&blocks).enumerate() {
            let rows = pool_of_rows(&step, hd, bl, &cache[sq], c).unwrap();
            for &p in &bl[..c.div_ceil(slots)] {
                let r = p as usize * page_elems..(p as usize + 1) * page_elems;
                pool[r.clone()].copy_from_slice(&rows[r]);
            }
        }
        let rows = pool.len() / kv_width;
        sources.push((src(i), pool, vec![rows, kv_width]));
    }
    let mut session =
        scratchy_target_spyre::runner::SpyreSession::new_multi(&[(&groups, &[t_out])], typed)
            .unwrap_or_else(|e| panic!("{at}: the session builds: {e}"));
    let out = session
        .run_step(0, sources, &[(t_out, 0)])
        .unwrap_or_else(|e| panic!("{at}: the step runs: {e}"));
    let got = &out[&t_out];
    assert_eq!(got.len(), b * q_width, "{at}");

    // The f64 reference: sequence r's query attends its own cached keys and its own new token.
    let group_size = (nqh / nkvh) as usize;
    let dot = |a: &[f32], b: &[f32]| {
        a.iter()
            .zip(b)
            .map(|(x, y)| f64::from(*x) * f64::from(*y))
            .sum::<f64>()
    };
    let mut max_err = 0.0f64;
    for (r, &c) in contexts.iter().enumerate() {
        for h in 0..nqh as usize {
            let kvh = h / group_size;
            let qrow = &q[r * q_width + h * hd..][..hd];
            let mut keys: Vec<(f64, &[f32])> = (0..c)
                .map(|t| {
                    let k = &kc[r][t * kv_width + kvh * hd..][..hd];
                    let v = &vc[r][t * kv_width + kvh * hd..][..hd];
                    (f64::from(scale) * dot(qrow, k), v)
                })
                .collect();
            let k = &kd[r * kv_width + kvh * hd..][..hd];
            keys.push((
                f64::from(scale) * dot(qrow, k),
                &vd[r * kv_width + kvh * hd..][..hd],
            ));
            let m = keys
                .iter()
                .map(|(s, _)| *s)
                .fold(f64::NEG_INFINITY, f64::max);
            let l: f64 = keys.iter().map(|(s, _)| (s - m).exp()).sum();
            for d in 0..hd {
                let want: f64 = keys
                    .iter()
                    .map(|(s, v)| (s - m).exp() / l * f64::from(v[d]))
                    .sum();
                let g = f64::from(got[r * q_width + h * hd + d]);
                max_err = max_err.max((g - want).abs());
            }
        }
    }
    println!("  decode batch {at}: max abs err {max_err:.3e} (gate 0.05)");
    assert!(
        max_err < 0.05,
        "the REAL decode batch diverged from the f64 softmax ({at}): max abs err {max_err}"
    );
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
            let spliced =
                scratchy_triton_splice::lower::<ktir_superdsc::collective::TpWorld<1>, _>(
                    node, &ir, false,
                )
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
            // ⛔ THE GAIN LEG IS CLAIMED ON THE REAL SPLICED PROGRAM, at every row count — the
            // one-row tile included, where the row's spray is extent-1, the producer folds the
            // broadcast away, and only the leg's own `[1, cols]` shape states the row. A close
            // as the identity here would flip every decode-bucket norm to the unit form and
            // move the gained bundles (the g2bfp8 gate's measured diff).
            assert!(
                match gain {
                    GainConvention::Scale => matches!(
                        chains[0].gain,
                        ktir_superdsc::emit::whole_function::GainLeg::Scale(_)
                    ),
                    GainConvention::OnePlusScale => matches!(
                        chains[0].gain,
                        ktir_superdsc::emit::whole_function::GainLeg::OnePlusScale(_)
                    ),
                },
                "the real spliced {gain:?} program claims its gain row (m={m} c={c})"
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

    // The un-gained spelling: the door recognizes it as the SAME chain with the gain leg
    // closed as the identity — one chain, `GainLeg::Identity`, its terminal the normalising
    // multiply — and reads it through the per-op epsilon reader (one root, the addf carrying
    // the eps splat), the emulator through the same scalar-chain walk — and the eps both see
    // is the f16 image.
    for &(m, c) in SHAPES {
        let ir = rmsnormunit_ir(m, c);
        let node = &ir.nodes[0];
        let spliced = scratchy_triton_splice::lower::<ktir_superdsc::collective::TpWorld<1>, _>(
            node, &ir, false,
        )
        .unwrap_or_else(|e| panic!("splice compiled the unit row m={m} c={c}: {e}"));
        let f = spliced
            .ktir
            .as_ref()
            .expect("spliced op carries its program");

        let chains = program_rmsnorm_chains(&f.func)
            .unwrap_or_else(|e| panic!("door recognized the unit row m={m} c={c}: {}", e.message));
        assert_eq!(chains.len(), 1, "one chain per unit program (m={m} c={c})");
        assert!(
            matches!(
                chains[0].gain,
                ktir_superdsc::emit::whole_function::GainLeg::Identity
            ),
            "the unit program's gain leg closes as the identity (m={m} c={c})"
        );
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
        let consumed: HashSet<_> = chains[0].consumed.iter().copied().collect();
        assert_eq!(
            sticks.intersection(&consumed).count(),
            3,
            "the unit chain's sticks are its own interior (m={m} c={c})"
        );
    }

    // The negative control: a program with no norm recognizes on NEITHER side.
    let ir = silumul_ir(1, 4096);
    let node = &ir.nodes[0];
    let spliced =
        scratchy_triton_splice::lower::<ktir_superdsc::collective::TpWorld<1>, _>(node, &ir, false)
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

/// ⛔⛔⛔ A CONTINUATION'S PROGRAM IS FLAT IN ITS ROWS — the whole-chunk form, pinned at the door's own
/// guard and across row counts. The per-row arm it replaced held `mq` row bodies (granite-2b at 256
/// rows: 94.7 s to lower; at 512, never finished). Eight times the rows must not grow the program, both
/// widths must pass the door's guard, and a program that windows its query per row must be refused.
#[test]
fn a_paged_walks_program_does_not_grow_with_its_rows() {
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
        let spliced =
            scratchy_triton_splice::lower_attn::<ktir_superdsc::collective::TpWorld<1>, _>(
                node,
                &ir,
                scratchy_subtile::subtile_ir::AttnScaleSite::InKernel,
                s.active_cap(),
                // One request of the graph's `mq` rows — the flat walk's own pair.
                pair_of_one_request(mq),
                PagedRows::Sequence,
            )
            .unwrap_or_else(|e| panic!("splice compiled mq={mq}: {e}"));
        spliced
            .ktir
            .expect("spliced op carries its program")
            .func
            .ops_deep()
            .len()
    };
    // The prompt walk is one head's rows against each page, so its program is a function of the
    // heads and the pages, never of the row count: 8x the rows is the same program.
    let (small, wide) = (ops_at(31), ops_at(256));
    assert!(
        wide <= small,
        "8x the rows grew the program: 31 rows {small} ops, 256 rows {wide} ops"
    );
}

/// ⭐ THE fp8-DYNAMIC EPILOGUE'S LIVE-TILE COUNT, COUNTED FROM THE MINTED
/// PROGRAM — the pin for `scratchy_triton_splice::FP8_EPILOGUE_LIVE_TILES`,
/// which the emu prefill-width derivation reads as the epilogue's LX cost (the
/// count is not restated here; it is measured off the KTIR the splice emits).
/// Mints the fp8 matmul's program, walks it in program order, and counts the
/// f16 `[m, N]` tensor values live at the peak — a value is live from its def
/// to its last use. At the peak (right after the scale-row broadcast) that is
/// the dot result `p`, the broadcast's shape-defining `tensor.empty`, and the
/// broadcast replica; the `arith.mulf` then consumes `p` and the replica, so
/// nothing else reaches the count. The collapsed `[N]` scale row is an `[N]`
/// tensor, not `[m, N]` — it is the peak-bytes formula's `+ N·2` term.
#[test]
fn the_fp8_epilogue_holds_three_live_mn_tiles_at_its_peak() {
    // a granite-scale fp8 gemm width and a small shape: the count is
    // structural in the emitted program, not a function of the extents.
    for (m, k, n) in [(8u32, 128u32, 256u32), (35u32, 2048u32, 8192u32)] {
        let ir = matmul_fp8_ir(m, k, n);
        let node = &ir.nodes[0];
        let spliced = scratchy_triton_splice::lower::<ktir_superdsc::collective::TpWorld<1>, _>(
            node, &ir, true,
        )
        .unwrap_or_else(|e| panic!("splice compiled m={m} k={k} n={n}: {e}"));
        let prog = spliced
            .ktir
            .as_ref()
            .expect("the spliced op carries its program");
        let ops = prog.func.ops_deep();
        let is_mn_f16 = |v: ktir_core::ir::Ssa| {
            ops.iter().any(|op| {
                op.result == Some(v)
                    && matches!(
                        op.result_type,
                        Some(ktir_core::irtype::IrType::Tensor {
                            dims,
                            elem: ktir_core::dtypes::DType::F16,
                        }) if dims == [i64::from(m), i64::from(n)]
                    )
            })
        };
        // Def point and last use of every value, in program order.
        use std::collections::HashMap;
        let mut def: HashMap<ktir_core::ir::Ssa, usize> = HashMap::new();
        let mut last_use: HashMap<ktir_core::ir::Ssa, usize> = HashMap::new();
        for (i, op) in ops.iter().enumerate() {
            if let Some(v) = op.result {
                def.insert(v, i);
            }
            for &u in op.operands {
                last_use.insert(u, i);
            }
        }
        // Live `[m, N]` f16 tensors at each program point; the peak is the max.
        let mut peak = 0usize;
        for t in 0..ops.len() {
            let live = def
                .keys()
                .filter(|&&v| {
                    is_mn_f16(v) && def[&v] <= t && last_use.get(&v).is_some_and(|&u| u >= t)
                })
                .count();
            peak = peak.max(live);
        }
        assert_eq!(
            peak,
            usize::try_from(scratchy_triton_splice::FP8_EPILOGUE_LIVE_TILES)
                .expect("the live-tile const fits a usize"),
            "m={m} k={k} n={n}: the fp8 epilogue's live [m, N] f16 tile count at its \
             peak must equal FP8_EPILOGUE_LIVE_TILES — the emu prefill-width derivation \
             charges this count against the LX budget"
        );
    }
}

/// ⭐⭐ THE TALL WIDE-HIDDEN RUNG — a prefill-width rmsnorm at hidden 16384 whose
/// mean(x²) bills past the per-core LX budget (2048 rows × 32 KiB/row, spread over the
/// cores, is past 1,677,721 B per core), so the chain's own reduce goes through the
/// LONG-REDUCE SPLIT inside `assemble_rmsnorm`: `time` single-shot slice reduces plus
/// `time − 1` running combines, in place on the chain's own mean. The door lowers the
/// whole program — nothing refuses — and the split's slice marker (`_p1_op`, an op name
/// only the split body mints) is present in the emission. The SHAPES sweep above is the
/// byte-identity control: every fitting rung still lowers one-for-one.
#[test]
fn the_tall_wide_hidden_rmsnorm_mean_splits_through_the_door() {
    let (m, c) = (2048u32, 16384u32);
    let ir = rmsnorm_ir(m, c, GainConvention::Scale);
    let weight_ids: HashSet<u32> = [0u32, 1u32].into_iter().collect();
    let layout = compute_bundle_layout::<ktir_superdsc::collective::TpWorld<1>, _>(
        &ir,
        &weight_ids,
        PagedRows::Sequence,
        &[],
        pair_of_one_request(m),
    )
    .unwrap_or_else(|e| panic!("layout minted m={m} c={c}: {e}"));
    let node = &ir.nodes[0];
    let spliced =
        scratchy_triton_splice::lower::<ktir_superdsc::collective::TpWorld<1>, _>(node, &ir, false)
            .unwrap_or_else(|e| panic!("splice compiled m={m} c={c}: {e}"));
    assert_eq!(
        spliced.op_name,
        format!("rmsnorm_s{}", node.id.index()),
        "op_name (m={m} c={c})"
    );
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
    .unwrap_or_else(|e| panic!("spliced program lowered (m={m} c={c}): {}", e.message));
    assert!(
        emitted.iter().any(|e| e.op_name.contains("_p1_op")),
        "the tall rung's mean(x²) splits: no `_p1_op` slice among {} emitted ops",
        emitted.len()
    );
    // ⛔ #1113's ADDRESS-RESOLVING CHECK (auditor ruling 2026-11-06): every time-tiled op
    // the door emitted — the split's own slices and combines included — must keep every
    // STRIDED operand's whole trip span inside that operand's own footprint, resolved
    // against the concrete per-trip addresses `concrete_trips` baked. This is the rung
    // whose rmg gain (an mb-broadcast read by a time-tiled op) advanced 64× past its
    // one-row frame on the card (run-4, RAS::PCI::BusFence 0xa35e); the own-frame stride
    // law plus the emitter's bake-time segment law must hold here op by op, and this
    // asserts it on the REAL door emission, not a hand-minted fixture.
    let mut tiled_ops = 0usize;
    for e in &emitted {
        if e.time == 1 {
            continue;
        }
        tiled_ops += 1;
        for (ti, strides) in e.affine_strides.iter().enumerate() {
            let Some(&stride) = strides.values().next() else {
                continue;
            };
            assert!(
                stride > 0,
                "m={m} c={c} op '{}': operand {ti} carries a stride map with a non-advance",
                e.op_name
            );
            let trips = e.concrete_trips();
            let start = |t: usize| -> u64 {
                trips
                    .get(t)
                    .unwrap_or_else(|| panic!("a trip per time ({})", e.time))
                    .dscs_[0]
                    .values()
                    .next()
                    .expect("one dsc")
                    .scheduleTree_
                    .iter()
                    .find(|n| n.nodeType_ == "allocate" && n.ldsIdx_ as usize == ti)
                    .unwrap_or_else(|| {
                        panic!("op '{}': an allocate node per operand {ti}", e.op_name)
                    })
                    .startAddressCoreCorelet_
                    .data_
                    .values()
                    .next()
                    .expect("a per-core address")
                    .parse()
                    .expect("a decimal address")
            };
            let span = start(e.time as usize - 1) - start(0) + stride as u64;
            assert!(
                span <= e.arg_bindings[ti].footprint_bytes,
                "m={m} c={c} op '{}': operand {ti}'s trips span {span} B (stride {stride} B × \
                 {} trips) but its own frame is {} B — the op's iteration outruns the operand",
                e.op_name,
                e.time,
                e.arg_bindings[ti].footprint_bytes
            );
        }
    }
    assert!(
        tiled_ops > 0,
        "m={m} c={c}: the tall rung's emission carries at least one time-tiled op — the \
         split's slices are single-trip by construction, but the program's wide pointwise \
         applies tile"
    );
}
