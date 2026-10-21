// SPDX-License-Identifier: Apache-2.0
//! The hand-written tapes, with no CUDA toolchain: each `static` is exactly what CUDA's compiler
//! lowers its canonical to, the numbers are the ones the layout rules give, every static passes
//! `CudaTape::validate`, and every op a kernel cannot run is refused at expansion, naming the
//! value and the rule.

mod common {
    pub mod attn_gate;
    pub mod chains;
    pub mod mlp;
}

use std::num::NonZeroU32;

use common::attn_gate::{ATTN_GATE_M128, ATTN_GATE_M256, attn_gate, attn_gate_input};
use common::chains::{
    QWEN3_5_FACTS, THREE_GEMM_M256, TWO_GEMM_M128, TWO_GEMM_M256, assert_tape_is_lowered,
    chain_input, three_gemm, two_gemm, with_hidden,
};
use common::mlp::{EPS, H, I, MLP_M128, MLP_M256, mlp};
use scratchy_subtile::fixtures::{
    front_end_lowered, gemm_m1_only_input, one_layer_input, silu_mul_only_input,
};
use scratchy_subtile::handoff::{LoweredDecode, SourceBinding};
use scratchy_subtile::lower::{
    AffineInt4, GemmWeight, GemmWeightKind, InputRef, LoweringInput, OpDesc,
};
use scratchy_subtile::model_geometry::HeadDim;
use scratchy_subtile::subtile_ir::{EwKind, GainConvention, RowScale, SourceShape, SubOp};
use scratchy_target_cuda::step::{CudaStep, GemmBf16};
use scratchy_target_cuda::tape::{ArenaOffset, CUDA_ARENA_ALIGN, CudaTape};
use scratchy_target_cuda_compiler::canonical::{
    ArenaRefusal, CanonicalAt, CanonicalRefusal, CudaModelFacts, GateShapeRule, GemmShapeRule,
    OperandRefusal, RopeShapeRule, RowShapeRule, ViewRefusal, lower_canonical,
};

fn lower_with(
    l: &LoweredDecode,
    facts: CudaModelFacts,
) -> Result<scratchy_target_cuda_compiler::canonical::CudaCanonical, CanonicalRefusal> {
    let at = CanonicalAt {
        stem: "fixture",
        m: u64::from(l.input.ops[0].m),
    };
    lower_canonical(l, facts, at)
}

/// The refusal and its message, which must name every value in `names`.
fn refused(l: &LoweredDecode, names: &[&str]) -> CanonicalRefusal {
    refused_with(l, QWEN3_5_FACTS, names)
}

/// [`refused`] under `facts`.
fn refused_with(l: &LoweredDecode, facts: CudaModelFacts, names: &[&str]) -> CanonicalRefusal {
    let e = match lower_with(l, facts) {
        Err(e) => e,
        Ok(c) => panic!("lowered a canonical the kernel cannot run: {c:?}"),
    };
    let msg = e.to_string();
    for name in names {
        assert!(msg.contains(name), "{msg:?} does not name {name:?}");
    }
    eprintln!("refused: {msg}");
    e
}

#[test]
fn each_static_is_what_its_canonical_lowers_to() {
    assert_tape_is_lowered("two_gemm(128)", &TWO_GEMM_M128, &two_gemm(128));
    assert_tape_is_lowered("two_gemm(256)", &TWO_GEMM_M256, &two_gemm(256));
    assert_tape_is_lowered("three_gemm", &THREE_GEMM_M256, &three_gemm());
}

fn gemm_steps(tape: &CudaTape) -> Vec<GemmBf16> {
    tape.steps
        .iter()
        .map(|s| match *s {
            CudaStep::Gemm(g) => g,
            other => panic!("a GEMM chain holds only GEMMs, got {other:?}"),
        })
        .collect()
}

#[test]
fn the_two_gemm_numbers() {
    for (tape, m) in [(&TWO_GEMM_M128, 128u64), (&TWO_GEMM_M256, 256)] {
        // x [m, 512], y1 [m, 1024], y2 [m, 512], bf16.
        let bytes = [m * 512 * 2, m * 1024 * 2, m * 512 * 2];
        assert_eq!(tape.arena.colour_bytes, bytes);
        assert_eq!(
            tape.arena.colour_offsets,
            [0, bytes[0], bytes[0] + bytes[1]]
        );
        assert_eq!(tape.arena.total_bytes, bytes.iter().sum::<u64>());
        assert_eq!((tape.input.offset.0, tape.input.bytes), (0, bytes[0]));
        assert_eq!(
            (tape.result.offset.0, tape.result.bytes),
            (bytes[0] + bytes[1], bytes[2])
        );
        assert_eq!(tape.weight_bytes, [0, 1048576, 1048576]);
        for &o in tape.arena.colour_offsets {
            assert_eq!(o % CUDA_ARENA_ALIGN, 0);
        }
    }
    assert_eq!(TWO_GEMM_M256.arena.total_bytes, 1048576);
    assert_eq!(TWO_GEMM_M128.arena.total_bytes, 524288);
    let s = gemm_steps(&TWO_GEMM_M256);
    assert_eq!(
        [
            (s[0].a.0, s[0].b.0, s[0].c.0),
            (s[1].a.0, s[1].b.0, s[1].c.0)
        ],
        [(0, 1, 262144), (262144, 2, 786432)]
    );
    assert_eq!(
        [(s[0].m, s[0].n, s[0].k), (s[1].m, s[1].n, s[1].k)],
        [(256, 1024, 512), (256, 512, 1024)]
    );
}

#[test]
fn the_three_gemm_chain_reuses_y1s_colour_for_y3() {
    let t = &THREE_GEMM_M256;
    let s = gemm_steps(t);
    assert_eq!(s[2].c, s[0].c, "y3 is written over y1's colour");
    assert_eq!(
        (s[2].a.0, s[2].b.0, s[2].c.0, s[2].m, s[2].n, s[2].k),
        (786432, 3, 262144, 256, 1024, 512)
    );
    assert_eq!(t.arena.colour_bytes, [262144, 524288, 262144]);
    assert_eq!(t.arena.colour_offsets, [0, 262144, 786432]);
    assert_eq!(t.arena.total_bytes, 1048576);
    // x, y1, y2 and y3 each in a buffer of its own.
    let per_value: u64 = 262144 + 524288 + 262144 + 524288;
    assert_eq!(per_value, 1572864);
    assert!(t.arena.total_bytes < per_value);
}

#[test]
fn a_decode_gemm_is_refused_naming_m_1() {
    let e = refused(
        &front_end_lowered(gemm_m1_only_input()),
        &["m=1", "decode GEMV"],
    );
    assert!(matches!(
        e,
        CanonicalRefusal::GemmShape {
            step: 0,
            m: 1,
            rule: GemmShapeRule::Decode,
            ..
        }
    ));
}

#[test]
fn a_gemm_off_the_tile_grid_is_refused_naming_the_dimension_and_its_tile() {
    let e = refused(
        &with_hidden(chain_input(256, 512, &[200, 512])),
        &["n = 200", "multiple of 256"],
    );
    assert!(matches!(
        e,
        CanonicalRefusal::GemmShape {
            step: 0,
            n: 200,
            rule: GemmShapeRule::OffTile { dim: "n", .. },
            ..
        }
    ));
    let e = refused(
        &with_hidden(chain_input(256, 100, &[1024, 512])),
        &["k = 100", "multiple of 64"],
    );
    assert!(matches!(
        e,
        CanonicalRefusal::GemmShape {
            step: 0,
            k: 100,
            rule: GemmShapeRule::OffTile { dim: "k", .. },
            ..
        }
    ));
}

#[test]
fn a_gemm_dimension_above_int_max_is_refused() {
    let n = 1u32 << 31;
    let e = refused(
        &with_hidden(chain_input(256, 512, &[n])),
        &["n = 2147483648", "2147483647"],
    );
    assert!(matches!(
        e,
        CanonicalRefusal::GemmShape {
            step: 0,
            rule: GemmShapeRule::AboveIntMax { dim: "n", .. },
            ..
        }
    ));
}

#[test]
fn an_affine_weight_is_refused() {
    let mut l = two_gemm(256);
    let affine = AffineInt4::mint(4, 64, 512).expect("4-bit, group 64, over k = 512");
    l.input.ops[0].op = SubOp::MatmulTile {
        n: 1024,
        weight: GemmWeight::Affine { affine },
    };
    let e = refused(&l, &["Affine", "Dense only"]);
    assert!(matches!(
        e,
        CanonicalRefusal::GemmWeight {
            step: 0,
            weight: GemmWeightKind::Affine
        }
    ));
}

#[test]
fn a_canonical_with_no_embedded_hidden_is_refused() {
    let e = refused(
        &front_end_lowered(chain_input(256, 512, &[1024, 512])),
        &["embedded hidden"],
    );
    assert!(matches!(
        e,
        CanonicalRefusal::Arena(ArenaRefusal::NoEmbeddedHidden)
    ));
}

/// A llama decode layer lowers its first op, the norm, and is refused at the first op after it:
/// a one-row (decode) GEMM.
#[test]
fn a_layer_lowers_its_norm_and_is_refused_at_its_first_gemm() {
    let e = refused(&with_hidden(one_layer_input()), &["m=1", "decode GEMV"]);
    assert!(matches!(
        e,
        CanonicalRefusal::GemmShape {
            step: 1,
            m: 1,
            rule: GemmShapeRule::Decode,
            ..
        }
    ));
}

/// An op with no lowering at all is refused by its name.
#[test]
fn an_op_with_no_lowering_is_refused_naming_it() {
    let e = refused(
        &ew_only(EwKind::Gelu, 256, H, None),
        &["Gelu", "does not lower"],
    );
    assert!(matches!(
        e,
        CanonicalRefusal::NotLowered {
            step: 0,
            op: "Gelu"
        }
    ));
}

// ── The qwen3.5-0.8b MLP block ─────────────────────────────────────

#[test]
fn the_mlp_statics_are_what_their_canonicals_lower_to() {
    assert_tape_is_lowered("mlp(128)", &MLP_M128, &mlp(128));
    assert_tape_is_lowered("mlp(256)", &MLP_M256, &mlp(256));
}

#[test]
fn every_static_validates() {
    for (name, tape) in [
        ("TWO_GEMM_M128", &TWO_GEMM_M128),
        ("TWO_GEMM_M256", &TWO_GEMM_M256),
        ("THREE_GEMM_M256", &THREE_GEMM_M256),
        ("MLP_M128", &MLP_M128),
        ("MLP_M256", &MLP_M256),
        ("ATTN_GATE_M128", &ATTN_GATE_M128),
        ("ATTN_GATE_M256", &ATTN_GATE_M256),
    ] {
        assert_eq!(tape.validate(), Ok(()), "{name}");
    }
}

/// The tapes before this one have no rope, so no rotary table.
#[test]
fn the_earlier_statics_have_no_rotary_table() {
    for tape in [
        &TWO_GEMM_M128,
        &TWO_GEMM_M256,
        &THREE_GEMM_M256,
        &MLP_M128,
        &MLP_M256,
    ] {
        assert_eq!(tape.rotary, None);
    }
}

#[test]
fn the_mlp_numbers() {
    for (tape, m) in [(&MLP_M128, 128u64), (&MLP_M256, 256)] {
        let (h, i) = (u64::from(H) * m * 2, u64::from(I) * m * 2);
        // x, n (then d), gate, up, s, a, y.
        assert_eq!(tape.arena.colour_bytes, [h, h, i, i, i, i, h]);
        assert_eq!(tape.arena.total_bytes, 3 * h + 4 * i);
        assert_eq!((tape.input.offset.0, tape.input.bytes), (0, h));
        assert_eq!(
            (tape.result.offset.0, tape.result.bytes),
            (tape.arena.colour_offsets[6], h)
        );
        for &o in tape.arena.colour_offsets {
            assert_eq!(o % CUDA_ARENA_ALIGN, 0);
        }
    }
    assert_eq!(MLP_M256.arena.total_bytes, 8912896);
    assert_eq!(MLP_M128.arena.total_bytes, 4456448);
    let [
        CudaStep::RmsNorm(n),
        CudaStep::Gemm(gate),
        CudaStep::Gemm(up),
        CudaStep::SiluMul(sm),
        CudaStep::Gemm(down),
        CudaStep::Add(y),
    ] = *MLP_M256.steps
    else {
        panic!("the MLP's steps are norm, gemm, gemm, silu_mul, gemm, add");
    };
    assert_eq!((n.eps(), n.convention), (EPS, GainConvention::OnePlusScale));
    assert_eq!((gate.a, up.a), (n.y, n.y), "both projections read n");
    assert_eq!((sm.gate, sm.up), (gate.c, up.c), "silu(gate) · up");
    assert_eq!((sm.rows, sm.cols), (256, I));
    assert_eq!(down.a, sm.out);
    assert_eq!(down.c, n.y, "d is written over n's colour");
    assert_eq!((y.a, y.b), (down.c, ArenaOffset(0)), "y = d + x");
}

/// The absorbed `Silu`'s colour (4) is minted where its driver runs, and no launch writes or reads
/// it: the `SiluMul` kernel reads the gate and up GEMMs' colours and writes the `Mul`'s.
#[test]
fn no_mlp_step_touches_the_absorbed_silus_colour() {
    for tape in [&MLP_M128, &MLP_M256] {
        let s = tape.arena.colour_offsets[4];
        for (k, step) in tape.steps.iter().enumerate() {
            let e = step.extents().expect("an MLP step's bytes fit u64");
            for span in e.reads.iter().flatten().chain([&e.writes]) {
                assert_ne!(span.offset.0, s, "step {k} touches colour 4: {step:?}");
            }
        }
    }
}

// ── The row kernels' refusals ──────────────────────────────────────

const fn shape(rows: u32, cols: u32) -> SourceShape {
    SourceShape { rows, cols }
}

/// `RmsNorm(x [m, cols], g)`, `x` the embedded hidden and `g` a weight of shape `gain`.
fn norm_only(m: u32, cols: u32, eps: f32, gain: SourceShape) -> LoweredDecode {
    with_hidden(LoweringInput {
        exported: None,
        sources: vec![shape(m, cols), gain],
        ops: vec![OpDesc {
            op: SubOp::RmsNorm {
                eps,
                gain: GainConvention::OnePlusScale,
            },
            m,
            inputs: vec![InputRef::Ext(0), InputRef::Ext(1)],
        }],
        result: 0,
    })
}

/// One elementwise op of `kind` over `x [m, cols]` (the embedded hidden) and, for a binary op, a
/// second source of shape `b`, bound as the front end binds it (to a weight).
fn ew_only(kind: EwKind, m: u32, cols: u32, b: Option<SourceShape>) -> LoweredDecode {
    let sources = [Some(shape(m, cols)), b].into_iter().flatten().collect();
    let inputs = (0..1 + usize::from(b.is_some()))
        .map(InputRef::Ext)
        .collect();
    with_hidden(LoweringInput {
        exported: None,
        sources,
        ops: vec![OpDesc {
            op: SubOp::Elementwise(kind),
            m,
            inputs,
        }],
        result: 0,
    })
}

#[test]
fn a_row_op_off_the_vector_width_is_refused() {
    let e = refused(
        &norm_only(256, 1004, EPS, shape(1, 1004)),
        &["cols = 1004", "multiple of 8"],
    );
    assert!(matches!(
        e,
        CanonicalRefusal::RowShape {
            step: 0,
            op: "RmsNorm",
            rows: 256,
            cols: 1004,
            rule: RowShapeRule::OffVector {
                cols: 1004,
                multiple: 8
            },
        }
    ));
}

#[test]
fn a_row_op_above_int_max_elements_is_refused() {
    let (m, cols) = (32768, 65536);
    let e = refused(
        &ew_only(EwKind::Add, m, cols, Some(shape(m, cols))),
        &["rows·cols = 2147483648", "2147483647"],
    );
    assert!(matches!(
        e,
        CanonicalRefusal::RowShape {
            step: 0,
            op: "Add",
            rows: 32768,
            cols: 65536,
            rule: RowShapeRule::TooManyElements {
                elements: 2147483648
            },
        }
    ));
}

#[test]
fn a_norm_eps_that_is_zero_or_nan_is_refused() {
    let e = refused(&norm_only(256, H, 0.0, shape(1, H)), &["eps = 0"]);
    assert!(matches!(e, CanonicalRefusal::NormEps { step: 0, eps } if eps == 0.0));
    let e = refused(&norm_only(256, H, f32::NAN, shape(1, H)), &["eps = NaN"]);
    assert!(matches!(e, CanonicalRefusal::NormEps { step: 0, eps } if eps.is_nan()));
}

#[test]
fn a_norm_gain_not_one_row_of_cols_is_refused() {
    let e = refused(
        &norm_only(256, H, EPS, shape(1, 512)),
        &["[1, 512]", "[1, 1024]"],
    );
    assert!(matches!(
        e,
        CanonicalRefusal::GainShape {
            step: 0,
            rows: 1,
            cols: 512,
            want: 1024
        }
    ));
}

/// An add operand of another shape than the output. (One with fewer rows, a broadcast, never
/// reaches CUDA: the shared graph validator refuses a read past its tensor first.)
#[test]
fn an_add_operand_of_another_shape_is_refused() {
    let e = refused(
        &ew_only(EwKind::Add, 256, H, Some(shape(512, H))),
        &[
            "operand 1",
            "[512, 1024]",
            "[256, 1024]",
            "broadcasts nothing",
        ],
    );
    assert!(matches!(
        e,
        CanonicalRefusal::OperandShape {
            step: 0,
            op: "Add",
            operand: 1,
            rows: 512,
            cols: 1024,
            want_rows: 256,
            want_cols: 1024
        }
    ));
}

/// A `Silu` no `Mul` reads, a `Mul` whose operand 0 is not a `Silu` (neither is folded), and the
/// pre-fused `SiluMul`: CUDA runs SwiGLU only as the fold.
#[test]
fn swiglu_outside_the_fold_is_refused() {
    for (l, op) in [
        (ew_only(EwKind::Silu, 256, I, None), "Silu"),
        (ew_only(EwKind::Mul, 256, I, Some(shape(256, I))), "Mul"),
        (front_end_lowered(silu_mul_only_input()), "SiluMul"),
    ] {
        let e = refused(&l, &[op, "no standalone kernel", "SiluMul fold"]);
        assert!(
            matches!(e, CanonicalRefusal::Unfused { step: 0, op: o } if o == op),
            "{op}: {e:?}"
        );
    }
}

// ── The operand bindings, through the new arms ─────────────────────

#[test]
fn a_row_op_activation_that_is_a_weight_is_refused() {
    let e = refused(
        &ew_only(EwKind::Add, 256, H, Some(shape(256, H))),
        &[
            "operand 1",
            "source 1",
            "computed slot or the embedded hidden",
        ],
    );
    assert!(matches!(
        e,
        CanonicalRefusal::Operand {
            step: 0,
            op: "Add",
            operand: 1,
            why: OperandRefusal::NotHiddenOrSlot { source: 1, .. },
        }
    ));
}

#[test]
fn a_norm_gain_that_is_not_a_weight_is_refused() {
    let mut l = norm_only(256, H, EPS, shape(1, H));
    l.bindings[1] = SourceBinding::Cos { local: false };
    let e = refused(&l, &["operand 1", "source 1", "must be a model weight"]);
    assert!(matches!(
        e,
        CanonicalRefusal::Operand {
            step: 0,
            op: "RmsNorm",
            operand: 1,
            why: OperandRefusal::NotAWeight {
                source: Some((1, SourceBinding::Cos { local: false }))
            },
        }
    ));

    // A computed gain: the second norm's gain is the first norm's `[1, H]` output.
    let mut input = norm_only(1, H, EPS, shape(1, H)).input;
    input.ops.push(OpDesc {
        op: SubOp::RmsNorm {
            eps: EPS,
            gain: GainConvention::OnePlusScale,
        },
        m: 1,
        inputs: vec![InputRef::Ext(0), InputRef::Op(0)],
    });
    input.result = 1;
    let e = refused(&with_hidden(input), &["operand 1", "computed value"]);
    assert!(matches!(
        e,
        CanonicalRefusal::Operand {
            step: 1,
            op: "RmsNorm",
            operand: 1,
            why: OperandRefusal::NotAWeight { source: None },
        }
    ));
}

// ── qwen3.5-0.8b's attention prelude and gate ──────────────────────

#[test]
fn the_attention_gate_statics_are_what_their_canonicals_lower_to() {
    assert_tape_is_lowered("attn_gate(128)", &ATTN_GATE_M128, &attn_gate(128));
    assert_tape_is_lowered("attn_gate(256)", &ATTN_GATE_M256, &attn_gate(256));
}

/// Ten ops, eight launches: the two views emit none. The gate (colour 8) is the split's second
/// buffer and the gate apply's second read; the rope reads the table after the colours; `o` is
/// written over `n`.
#[test]
fn the_attention_gate_numbers() {
    for (tape, m) in [(&ATTN_GATE_M128, 128u64), (&ATTN_GATE_M256, 256)] {
        let (h, q) = (1024 * m * 2, 2048 * m * 2);
        assert_eq!(tape.arena.colour_bytes, [h, h, 2 * q, q, q, q, q, h, q]);
        let colours_end = 3 * h + 7 * q;
        let rotary = tape.rotary.expect("the tape has a rope");
        assert_eq!(
            (rotary.span.offset.0, rotary.span.bytes),
            (colours_end, m * 64 * 2)
        );
        assert_eq!((rotary.rot_dim, rotary.theta()), (64, 1e7));
        assert_eq!(tape.arena.total_bytes, colours_end + m * 128);
        for &o in tape.arena.colour_offsets {
            assert_eq!(o % CUDA_ARENA_ALIGN, 0);
        }
    }
    assert_eq!(ATTN_GATE_M256.arena.total_bytes, 8945664);
    assert_eq!(ATTN_GATE_M128.arena.total_bytes, 4472832);
    let [
        CudaStep::RmsNorm(n),
        CudaStep::Gemm(qg),
        CudaStep::GateSplit(split),
        CudaStep::RmsNorm(qn),
        CudaStep::Rope(rope),
        CudaStep::GateApply(ga),
        CudaStep::Gemm(o),
        CudaStep::Add(y),
    ] = *ATTN_GATE_M256.steps
    else {
        panic!("the steps are norm, gemm, gate split, norm, rope, gate apply, gemm, add");
    };
    let colour = |c: usize| ArenaOffset(ATTN_GATE_M256.arena.colour_offsets[c]);
    assert_eq!((qg.a, split.qg), (n.y, qg.c));
    assert_eq!((split.q, split.gate), (colour(3), colour(8)));
    assert_eq!(
        (qn.x, qn.rows, qn.cols),
        (split.q, 2048, 256),
        "the per-head view of q"
    );
    assert_eq!(rope.x, qn.y, "the flatten-back view of qn");
    assert_eq!(rope.table, ATTN_GATE_M256.rotary.unwrap().span.offset);
    assert_eq!(
        (ga.attn, ga.gate),
        (rope.out, split.gate),
        "gated by the second output"
    );
    assert_eq!((o.a, o.c), (ga.out, n.y), "o is written over n's colour");
    assert_eq!((y.a, y.b), (o.c, ArenaOffset(0)), "y = o + x");
}

/// A view whose element count changes is refused (a `Reshape` to `[2m, 1024]` of `[m, 2048]`
/// keeps the count, so the view rows are `Times(8)` with 128 columns: half the elements).
#[test]
fn a_view_that_changes_the_element_count_is_refused() {
    let mut l = attn_gate(256);
    l.input.ops[3].op = SubOp::Reshape {
        rows: RowScale::Times(NonZeroU32::new(8).unwrap()),
        cols: 128,
    };
    l.input.sources[3] = SourceShape { rows: 1, cols: 128 };
    let e = refused(&l, &["Reshape", "cannot be a view", "524288", "262144"]);
    assert!(matches!(
        e,
        CanonicalRefusal::View {
            step: 3,
            why: ViewRefusal::ElementCount {
                input: 524288,
                output: 262144
            }
        }
    ));
}

/// A view of a source would get a fresh colour nothing writes.
#[test]
fn a_view_of_a_source_is_refused() {
    let l = with_hidden(LoweringInput {
        exported: None,
        sources: vec![shape(256, 2048)],
        ops: vec![OpDesc {
            op: SubOp::Reshape {
                rows: RowScale::Times(NonZeroU32::new(8).unwrap()),
                cols: 256,
            },
            m: 256,
            inputs: vec![InputRef::Ext(0)],
        }],
        result: 0,
    });
    let e = refused(&l, &["Reshape", "source"]);
    assert!(matches!(
        e,
        CanonicalRefusal::View {
            step: 0,
            why: ViewRefusal::OfASource
        }
    ));
}

/// The rope's `cos` bound to a weight, or to the sliding-window table (`local: true`).
#[test]
fn a_rope_operand_that_is_not_the_global_table_is_refused() {
    let mut l = attn_gate(256);
    l.bindings[4] = SourceBinding::Weight { id: 4, index: None };
    let e = refused(&l, &["RopeRotate", "operand 1", "Weight"]);
    assert!(matches!(
        e,
        CanonicalRefusal::RotaryOperand {
            step: 6,
            operand: 1,
            binding: Some(SourceBinding::Weight { .. })
        }
    ));
    let mut l = attn_gate(256);
    l.bindings[5] = SourceBinding::Sin { local: true };
    let e = refused(&l, &["RopeRotate", "operand 2", "local: true"]);
    assert!(matches!(
        e,
        CanonicalRefusal::RotaryOperand {
            step: 6,
            operand: 2,
            binding: Some(SourceBinding::Sin { local: true })
        }
    ));
}

/// `RopeRotate(x, cos, sin)` alone over `x [tokens, cols]`, the embedded hidden, and the global
/// table.
fn rope_only(tokens: u32, cols: u32, head_dim: u32) -> LoweredDecode {
    let mut l = with_hidden(LoweringInput {
        exported: None,
        sources: vec![
            shape(tokens, cols),
            shape(tokens, cols),
            shape(tokens, cols),
        ],
        ops: vec![OpDesc {
            op: SubOp::rope_rotate(HeadDim::new(head_dim)),
            m: tokens,
            inputs: vec![InputRef::Ext(0), InputRef::Ext(1), InputRef::Ext(2)],
        }],
        result: 0,
    });
    l.bindings[1] = SourceBinding::Cos { local: false };
    l.bindings[2] = SourceBinding::Sin { local: false };
    l
}

#[test]
fn a_rope_whose_rot_dim_is_off_the_vector_is_refused() {
    let facts = CudaModelFacts {
        rot_dim: 24,
        ..QWEN3_5_FACTS
    };
    let e = refused_with(
        &rope_only(256, 2048, 256),
        facts,
        &["rot_dim=24", "multiple of 16"],
    );
    assert!(matches!(
        e,
        CanonicalRefusal::RopeShape {
            step: 0,
            rot_dim: 24,
            rule: RopeShapeRule::RotOffVector { multiple: 16 },
            ..
        }
    ));
}

/// The rope's rule, check by check: a `rot_dim` above the head, a `head_dim` off the vector, a
/// `cols` that is not a whole number of heads, and a zero `rot_dim`.
#[test]
fn every_rope_shape_rule_is_refused() {
    let with = |rot_dim| CudaModelFacts {
        rot_dim,
        ..QWEN3_5_FACTS
    };
    let cases = [
        (
            rope_only(256, 2048, 256),
            with(512),
            RopeShapeRule::RotAboveHead,
        ),
        (
            rope_only(256, 2016, 252),
            with(64),
            RopeShapeRule::HeadOffVector { multiple: 8 },
        ),
        (
            rope_only(256, 1000, 256),
            with(64),
            RopeShapeRule::ColsOffHead,
        ),
        (
            rope_only(256, 2048, 256),
            with(0),
            RopeShapeRule::Empty { dim: "rot_dim" },
        ),
    ];
    for (l, facts, rule) in cases {
        let e = refused_with(&l, facts, &["RopeRotate"]);
        assert!(
            matches!(e, CanonicalRefusal::RopeShape { step: 0, rule: r, .. } if r == rule),
            "{rule:?}: {e:?}"
        );
    }
}

/// A rope over more rows than the bucket has tokens: the rotary table has one row per token. Its
/// `x` is the `[256, 4096]` hidden viewed as `[512, 2048]`.
#[test]
fn a_rope_whose_rows_are_not_the_tokens_is_refused() {
    let mut l = with_hidden(LoweringInput {
        exported: None,
        sources: vec![shape(256, 4096), shape(512, 2048), shape(512, 2048)],
        ops: vec![
            OpDesc {
                op: SubOp::Elementwise(EwKind::Add),
                m: 256,
                inputs: vec![InputRef::Ext(0), InputRef::Ext(0)],
            },
            OpDesc {
                op: SubOp::Reshape {
                    rows: RowScale::Times(NonZeroU32::new(2).unwrap()),
                    cols: 2048,
                },
                m: 256,
                inputs: vec![InputRef::Op(0)],
            },
            OpDesc {
                op: SubOp::rope_rotate(HeadDim::new(256)),
                m: 256,
                inputs: vec![InputRef::Op(1), InputRef::Ext(1), InputRef::Ext(2)],
            },
        ],
        result: 2,
    });
    l.bindings[1] = SourceBinding::Cos { local: false };
    l.bindings[2] = SourceBinding::Sin { local: false };
    let e = refused(&l, &["rows=512", "256 tokens"]);
    assert!(matches!(
        e,
        CanonicalRefusal::RopeShape {
            step: 2,
            rows: 512,
            rule: RopeShapeRule::RowsNotTokens { tokens: 256 },
            ..
        }
    ));
}

/// A gate split whose `half_cols` is not a whole number of the model's heads.
#[test]
fn a_gate_split_off_the_head_is_refused() {
    let facts = CudaModelFacts {
        head_dim: 384,
        ..QWEN3_5_FACTS
    };
    let e = refused_with(&attn_gate(256), facts, &["GateSplit", "half_cols=2048"]);
    assert!(matches!(
        e,
        CanonicalRefusal::GateShape {
            step: 2,
            half_cols: 2048,
            head_dim: 384,
            rule: GateShapeRule::HalfOffHead,
            ..
        }
    ));
    let facts = CudaModelFacts {
        head_dim: 0,
        ..QWEN3_5_FACTS
    };
    let e = refused_with(&attn_gate(256), facts, &["GateSplit", "head_dim = 0"]);
    assert!(matches!(
        e,
        CanonicalRefusal::GateShape {
            step: 2,
            rule: GateShapeRule::Empty { dim: "head_dim" },
            ..
        }
    ));
}

/// A gate apply whose gate is not a gate split's: here, the rope's output.
#[test]
fn a_gate_apply_whose_gate_is_not_a_split_is_refused() {
    let mut input = attn_gate_input(256);
    input.ops[7].inputs[1] = InputRef::Op(6);
    let mut l = with_hidden(input);
    l.bindings[4] = SourceBinding::Cos { local: false };
    l.bindings[5] = SourceBinding::Sin { local: false };
    let e = refused(&l, &["GateApply", "operand 1", "GateSplit"]);
    assert!(matches!(e, CanonicalRefusal::GateOperand { step: 7 }));
}

/// A gate split whose op the colour rule gives one output (as a rule change could) has no
/// buffer for its gate.
///
/// CUDA's rule gives `GateSplit` two outputs, so no canonical reaches this through
/// `lower_canonical`: it is unreachable under CUDA's table, and the refusal's message is checked
/// here instead.
#[test]
fn a_gate_split_with_no_second_colour_is_named() {
    let e = CanonicalRefusal::SecondOutput { step: 2 };
    let msg = e.to_string();
    assert!(
        msg.contains("step 2") && msg.contains("no second colour"),
        "{msg}"
    );
}

/// A row op with zero rows: the shared graph keeps a zero-row source, so `lower_canonical` reaches
/// the row rule's `Empty`.
#[test]
fn a_row_op_with_no_rows_is_refused() {
    let e = refused(
        &ew_only(EwKind::Add, 0, H, Some(shape(0, H))),
        &["rows = 0"],
    );
    assert!(matches!(
        e,
        CanonicalRefusal::RowShape {
            step: 0,
            op: "Add",
            rule: RowShapeRule::Empty { dim: "rows" },
            ..
        }
    ));
}
