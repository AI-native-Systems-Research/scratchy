// SPDX-License-Identifier: Apache-2.0
//! The hand-written tapes, with no CUDA toolchain: each `static` is exactly what CUDA's compiler
//! lowers its canonical to, the numbers are the ones the layout rules give, every static passes
//! `CudaTape::validate`, and every op a kernel cannot run is refused at expansion, naming the
//! value and the rule.

mod common {
    pub mod attn_gate;
    pub mod attn_layer;
    pub mod chains;
    pub mod gdn_layer;
    pub mod mlp;
}

use std::num::NonZeroU32;

use common::attn_gate::{ATTN_GATE_M128, ATTN_GATE_M256, attn_gate, attn_gate_input};
use common::attn_layer::{
    ATTN_LAYER_M128, ATTN_LAYER_M256, LAYER, PREFIX_CAPACITY, attn_layer, attn_layer_input,
};
use common::chains::GEMMA4_FACTS;
use common::chains::{
    QWEN3_5_FACTS, QWEN3_5_GDN_FACTS, THREE_GEMM_M256, TWO_GEMM_M128, TWO_GEMM_M256,
    assert_tape_is_lowered, assert_tape_is_lowered_with, chain_input, three_gemm, two_gemm,
    with_hidden,
};
use common::gdn_layer::{BUNDLE_BYTES, GDN_LAYER_M128, GDN_LAYER_M256, gdn_layer};
use common::mlp::{EPS, H, I, MLP_M128, MLP_M256, mlp};
use scratchy_subtile::fixtures::{
    front_end_lowered, gemm_m1_only_input, one_layer_input, silu_mul_only_input,
};
use scratchy_subtile::handoff::{LoweredDecode, SourceBinding, UnrollIndex};
use scratchy_subtile::lower::{
    AffineInt4, ArchOp, GemmWeight, GemmWeightKind, InputRef, LoweringInput, OpDesc,
};
use scratchy_subtile::model_geometry::{HeadDim, KvHeads, ModelAttnGeometry, QueryHeads};
use scratchy_subtile::subtile_ir::{
    AttnMask, EwKind, GainConvention, RopeFormTag, RowScale, SourceShape, SubOp,
};
use scratchy_subtile::subtile_tape::Instr;
use scratchy_target_cuda::step::{CudaStep, GemmBf16};
use scratchy_target_cuda::tape::{
    ArenaOffset, ArenaSpan, CUDA_ARENA_ALIGN, CudaTape, TapeInput, WeightIx,
};
use scratchy_target_cuda_compiler::canonical::{
    ArenaRefusal, AttnShapeRule, CanonicalAt, CanonicalRefusal, CudaGdnFacts, CudaModelFacts,
    GateShapeRule, GdnRegions, GdnShapeRule, GdnWidths, GemmShapeRule, KvCacheRefusal,
    OperandRefusal, ProducerRefusal, RopeShapeRule, RowShapeRule, ViewRefusal, lower_canonical,
    plan_canonical,
};
use scratchy_target_cuda_compiler::emit::static_tape;
use scratchy_target_cuda_compiler::steps_from_tape::{StepCx, TapeStep, lower_step};

const fn span(offset: u64, bytes: u64) -> ArenaSpan {
    ArenaSpan {
        offset: ArenaOffset(offset),
        bytes,
    }
}

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
        assert_eq!(tape.input, TapeInput::Hidden(span(0, bytes[0])));
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

/// A decode GEMM (`m = 1`) lowers to the GEMV, the tiled GEMM's operands at one row; a `k` off
/// the GEMV's 8-wide vector is refused by its rule, naming it.
#[test]
fn a_decode_gemm_lowers_to_the_gemv() {
    let c = lower_with(&with_hidden(gemm_m1_only_input()), QWEN3_5_FACTS)
        .expect("a one-row GEMM lowers");
    assert!(matches!(
        c.steps.as_slice(),
        [CudaStep::Gemv(GemmBf16 {
            a: ArenaOffset(0),
            b: WeightIx(1),
            m: 1,
            n: 2048,
            k: 2048,
            ..
        })]
    ));
    let e = refused(
        &with_hidden(chain_input(1, 100, &[1024])),
        &["m=1", "k = 100", "multiple of 8", "GEMV"],
    );
    assert!(matches!(
        e,
        CanonicalRefusal::GemmShape {
            step: 0,
            m: 1,
            rule: GemmShapeRule::GemvOffVector {
                value: 100,
                multiple: 8
            },
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

/// A llama decode layer lowers its norm and its three one-row GEMMs (GEMVs), and is refused at
/// the first op after them: its rope, whose cos the shared fixture binds as a weight.
#[test]
fn a_layer_lowers_its_norm_and_gemvs_and_is_refused_at_its_rope() {
    let e = refused(
        &with_hidden(one_layer_input()),
        &["step 4", "RopeRotate", "rotary table"],
    );
    assert!(matches!(
        e,
        CanonicalRefusal::RotaryOperand {
            step: 4,
            operand: 1,
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
        ("ATTN_LAYER_M128", &ATTN_LAYER_M128),
        ("ATTN_LAYER_M256", &ATTN_LAYER_M256),
        ("GDN_LAYER_M128", &GDN_LAYER_M128),
        ("GDN_LAYER_M256", &GDN_LAYER_M256),
    ] {
        assert_eq!(tape.validate(), Ok(()), "{name}");
    }
}

/// Only the attention layer binds a prefix cache, so only its tapes have a KV cache.
#[test]
fn only_the_attention_layer_has_a_kv_cache() {
    for tape in [
        &TWO_GEMM_M128,
        &TWO_GEMM_M256,
        &THREE_GEMM_M256,
        &MLP_M128,
        &MLP_M256,
        &ATTN_GATE_M128,
        &ATTN_GATE_M256,
    ] {
        assert_eq!(tape.kv, None);
    }
    assert!(ATTN_LAYER_M128.kv.is_some() && ATTN_LAYER_M256.kv.is_some());
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
        assert_eq!(tape.input, TapeInput::Hidden(span(0, h)));
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
            op: "RopeRotate",
            operand: 1,
            binding: Some(SourceBinding::Weight { .. }),
            ..
        }
    ));
    let mut l = attn_gate(256);
    l.bindings[5] = SourceBinding::Sin { local: true };
    let e = refused(&l, &["RopeRotate", "operand 2", "local: true"]);
    assert!(matches!(
        e,
        CanonicalRefusal::RotaryOperand {
            step: 6,
            op: "RopeRotate",
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

// ── qwen3.5-0.8b's full-attention layer ────────────────────────────

#[test]
fn the_attention_layer_statics_are_what_their_canonicals_lower_to() {
    assert_tape_is_lowered("attn_layer(128)", &ATTN_LAYER_M128, &attn_layer(128));
    assert_tape_is_lowered("attn_layer(256)", &ATTN_LAYER_M256, &attn_layer(256));
}

/// Seventeen ops, thirteen launches: the four views emit none. The rope-append reads the rotated
/// table and writes the rotated K (colour 8); the attention reads the rotated query, the tape's
/// block table and sequence, and writes colour 9; the gate keeps `q`'s split live to the gate
/// apply. After the colours: the rotary table, the block table, the slot mapping and the sequence.
#[test]
fn the_attention_layer_numbers() {
    for (tape, m) in [(&ATTN_LAYER_M128, 128u64), (&ATTN_LAYER_M256, 256)] {
        let (h, q, kv) = (1024 * m * 2, 2048 * m * 2, 512 * m * 2);
        assert_eq!(
            tape.arena.colour_bytes,
            [h, h, 2 * q, q, q, kv, kv, q, kv, q, h, q]
        );
        let colours_end = 3 * h + 7 * q + 3 * kv;
        let rotary = tape.rotary.expect("the tape has a rope");
        assert_eq!(
            (rotary.span.offset.0, rotary.span.bytes),
            (colours_end, m * 64 * 2)
        );
        let kv_tape = tape.kv.expect("the tape has a KV cache");
        let table_at = colours_end + m * 128;
        assert_eq!(
            (kv_tape.block_table.offset.0, kv_tape.block_table.bytes),
            (table_at, 512)
        );
        assert_eq!(
            (kv_tape.slot_mapping.offset.0, kv_tape.slot_mapping.bytes),
            (table_at + 512, m * 4)
        );
        let seq_at = table_at + 512 + (m * 4).next_multiple_of(256);
        assert_eq!(
            (kv_tape.sequence.offset.0, kv_tape.sequence.bytes),
            (seq_at, 8)
        );
        assert_eq!(tape.arena.total_bytes, seq_at + 256);
        assert_eq!(kv_tape.layers, [LAYER]);
        assert_eq!(
            (kv_tape.kv_heads, kv_tape.head_dim, kv_tape.block_size),
            (2, 256, 64)
        );
        assert_eq!(kv_tape.max_blocks_per_seq, 128);
        assert_eq!(tape.steps.len(), 13);
    }
    assert_eq!(ATTN_LAYER_M256.arena.total_bytes, 9733888);
    assert_eq!(ATTN_LAYER_M128.arena.total_bytes, 4867328);
    let t = &ATTN_LAYER_M256;
    let [
        CudaStep::RmsNorm(n),
        CudaStep::Gemm(qg),
        CudaStep::GateSplit(split),
        CudaStep::RmsNorm(qn),
        CudaStep::Gemm(k),
        CudaStep::RmsNorm(kn),
        CudaStep::Gemm(v),
        CudaStep::Rope(rope),
        CudaStep::RopeAppend(append),
        CudaStep::AttnPrefill(attn),
        CudaStep::GateApply(ga),
        CudaStep::Gemm(o),
        CudaStep::Add(y),
    ] = *t.steps
    else {
        panic!("the steps are the layer's thirteen launches, in order");
    };
    let colour = |c: usize| ArenaOffset(t.arena.colour_offsets[c]);
    let kv_tape = t.kv.expect("the tape has a KV cache");
    let table = t.rotary.expect("a rotary table").span.offset;
    assert_eq!((qg.a, k.a, v.a), (n.y, n.y, n.y), "q, k and v project n");
    assert_eq!(split.qg, qg.c);
    assert_eq!((qn.x, qn.rows), (split.q, 2048), "the per-head view of q");
    assert_eq!((kn.x, kn.rows), (k.c, 512), "the per-head view of k");
    assert_eq!(
        v.c, k.c,
        "v is written over k's colour: k is dead after its norm"
    );
    assert_eq!((rope.x, rope.table), (qn.y, table));
    assert_eq!(
        (append.k, append.v, append.table, append.k_out),
        (kn.y, v.c, table, colour(8))
    );
    assert_eq!(append.slot_mapping, kv_tape.slot_mapping.offset);
    assert_eq!(
        (
            append.layer,
            append.rows,
            append.cols,
            append.head_dim,
            append.rot_dim
        ),
        (0, 256, 512, 256, 64)
    );
    assert_eq!(
        (attn.q, attn.block_table, attn.sequence, attn.out),
        (
            rope.out,
            kv_tape.block_table.offset,
            kv_tape.sequence.offset,
            colour(9)
        )
    );
    assert_eq!(
        (
            attn.layer,
            attn.rows,
            attn.q_heads,
            attn.kv_heads,
            attn.head_dim
        ),
        (0, 256, 8, 2, 256)
    );
    assert_eq!(attn.scale_bits, 0.0625f32.to_bits());
    assert_eq!(attn.scale(), 0.0625);
    assert_eq!((ga.attn, ga.gate), (attn.out, split.gate));
    assert_eq!(ga.out, rope.out, "ga is written over qr's colour");
    assert_eq!((o.a, o.c), (ga.out, n.y), "o is written over n's colour");
    assert_eq!((y.a, y.b), (o.c, ArenaOffset(0)), "y = o + x");
}

/// [`attn_layer`]`(256)` with op `ix` replaced by `op`.
fn attn_layer_with(ix: usize, op: ArchOp) -> LoweredDecode {
    let mut l = attn_layer(256);
    l.input.ops[ix].op = op;
    l
}

/// qwen3.5-0.8b's attention geometry, or another one.
fn geom(q: u32, kv: u32, hd: u32) -> ModelAttnGeometry {
    ModelAttnGeometry::mint(QueryHeads::new(q), KvHeads::new(kv), HeadDim::new(hd))
        .expect("a valid geometry")
}

const fn rope_append_op(head_dim: u32, layer: u32, attn: AttnMask, pairing: RopeFormTag) -> ArchOp {
    SubOp::rope_append(HeadDim::new(head_dim), layer, attn, pairing)
}

const WINDOW: AttnMask = AttnMask::SlidingWindow {
    window: NonZeroU32::new(512).unwrap(),
};

#[test]
fn an_interleaved_rope_append_is_refused() {
    let l = attn_layer_with(
        12,
        rope_append_op(256, LAYER, AttnMask::Causal, RopeFormTag::Interleaved),
    );
    let e = refused(&l, &["RopeAppend", "interleaved"]);
    assert!(matches!(e, CanonicalRefusal::RopePairing { step: 12 }));
}

#[test]
fn a_sliding_window_rope_append_is_refused() {
    let l = attn_layer_with(12, rope_append_op(256, LAYER, WINDOW, RopeFormTag::NeoX));
    let e = refused(&l, &["RopeAppend", "sliding-window"]);
    assert!(matches!(
        e,
        CanonicalRefusal::SlidingWindow {
            step: 12,
            op: "RopeAppend"
        }
    ));
}

#[test]
fn a_rope_append_of_another_head_dim_is_refused() {
    let l = attn_layer_with(
        12,
        rope_append_op(128, LAYER, AttnMask::Causal, RopeFormTag::NeoX),
    );
    let e = refused(&l, &["RopeAppend", "head_dim = 128", "256"]);
    assert!(matches!(
        e,
        CanonicalRefusal::KvHeadDim {
            step: 12,
            op: "RopeAppend",
            head_dim: 128,
            want: 256
        }
    ));
}

/// `RopeAppend(k, cos, sin, v, pk, pv)` alone, every operand a source: `k` the embedded hidden
/// `[tokens, cols]`, the global table, `v` a weight, layer 3's prefix cache.
fn rope_append_only(tokens: u32, cols: u32) -> LoweredDecode {
    let mut l = with_hidden(LoweringInput {
        exported: None,
        sources: vec![
            shape(tokens, cols),
            shape(tokens, cols),
            shape(tokens, cols),
            shape(tokens, cols),
            shape(PREFIX_CAPACITY, cols),
            shape(PREFIX_CAPACITY, cols),
        ],
        ops: vec![OpDesc {
            op: rope_append_op(256, LAYER, AttnMask::Causal, RopeFormTag::NeoX),
            m: tokens,
            inputs: (0..6).map(InputRef::Ext).collect(),
        }],
        result: 0,
    });
    l.bindings[1] = SourceBinding::Cos { local: false };
    l.bindings[2] = SourceBinding::Sin { local: false };
    l.bindings[4] = SourceBinding::PrefixK {
        layer: u64::from(LAYER),
    };
    l.bindings[5] = SourceBinding::PrefixV {
        layer: u64::from(LAYER),
    };
    l
}

/// The rope's shape rule applied to K: a `rot_dim` off the vector, and K's `cols` not a whole
/// number of heads. The RopeAppend is alone, so no `RopeRotate` refuses first.
#[test]
fn a_rope_append_off_the_rope_shape_is_refused() {
    let facts = CudaModelFacts {
        rot_dim: 24,
        ..QWEN3_5_FACTS
    };
    let e = refused_with(
        &rope_append_only(256, 512),
        facts,
        &["RopeAppend", "rot_dim=24"],
    );
    assert!(matches!(
        e,
        CanonicalRefusal::RopeShape {
            step: 0,
            op: "RopeAppend",
            rule: RopeShapeRule::RotOffVector { multiple: 16 },
            ..
        }
    ));
    let e = refused(
        &rope_append_only(256, 640),
        &["RopeAppend", "whole number of heads"],
    );
    assert!(matches!(
        e,
        CanonicalRefusal::RopeShape {
            step: 0,
            op: "RopeAppend",
            rule: RopeShapeRule::ColsOffHead,
            ..
        }
    ));
}

/// A K that is not a computed slot: the kernel writes the rotated K into the step's own colour,
/// reading K from one.
#[test]
fn a_rope_append_whose_k_is_a_source_is_refused() {
    let e = refused(
        &rope_append_only(256, 512),
        &["RopeAppend", "operand 0", "computed slot"],
    );
    assert!(matches!(
        e,
        CanonicalRefusal::Operand {
            step: 0,
            op: "RopeAppend",
            operand: 0,
            why: OperandRefusal::NotASlot { source: 0, .. },
        }
    ));
}

#[test]
fn a_rope_append_whose_rotary_operand_is_not_the_global_table_is_refused() {
    let mut l = attn_layer(256);
    l.bindings[9] = SourceBinding::Cos { local: true };
    let e = refused(&l, &["RopeAppend", "operand 1", "local: true"]);
    assert!(matches!(
        e,
        CanonicalRefusal::RotaryOperand {
            step: 12,
            op: "RopeAppend",
            operand: 1,
            binding: Some(SourceBinding::Cos { local: true }),
        }
    ));
}

/// The RopeAppend's operands 4 and 5 must be its own layer's prefix cache: here it names layer 3's
/// and says it appends layer 5's.
#[test]
fn a_rope_append_whose_prefix_operand_is_another_layers_is_refused() {
    let l = attn_layer_with(
        12,
        rope_append_op(256, 5, AttnMask::Causal, RopeFormTag::NeoX),
    );
    let e = refused(
        &l,
        &[
            "RopeAppend",
            "operand 4",
            "PrefixK { layer: 3 }",
            "layer 5's PrefixK",
        ],
    );
    assert!(matches!(
        e,
        CanonicalRefusal::KvOperand {
            step: 12,
            op: "RopeAppend",
            operand: 4,
            layer: 5,
            want: "PrefixK",
            binding: Some(SourceBinding::PrefixK { layer: 3 }),
        }
    ));
    let mut l = attn_layer(256);
    l.bindings[12] = SourceBinding::Weight {
        id: 12,
        index: None,
    };
    l.bindings.push(SourceBinding::PrefixV {
        layer: u64::from(LAYER),
    });
    l.input.sources.push(shape(PREFIX_CAPACITY, 512));
    let e = refused(&l, &["RopeAppend", "operand 5", "PrefixV"]);
    assert!(matches!(
        e,
        CanonicalRefusal::KvOperand {
            step: 12,
            operand: 5,
            binding: Some(SourceBinding::Weight { .. }),
            ..
        }
    ));
}

/// A layer that binds a `PrefixK` and no `PrefixV` is not a cache: refused at the first KV step,
/// though no step names that layer. A step naming one layer's `PrefixK` beside another's
/// `PrefixV` is refused at its operand first.
#[test]
fn a_prefix_k_with_no_prefix_v_is_refused() {
    let mut l = attn_layer(256);
    l.input.sources.push(shape(PREFIX_CAPACITY, 512));
    l.bindings.push(SourceBinding::PrefixK { layer: 7 });
    let e = refused(
        &l,
        &["KV cache", "layer 7", "only one of PrefixK and PrefixV"],
    );
    assert!(matches!(
        e,
        CanonicalRefusal::KvCache(KvCacheRefusal::Unpaired { layer: 7 })
    ));
    let mut l = attn_layer(256);
    l.bindings[12] = SourceBinding::PrefixV { layer: 4 };
    let e = refused(&l, &["RopeAppend", "operand 5", "layer 3's PrefixV"]);
    assert!(matches!(
        e,
        CanonicalRefusal::KvOperand {
            step: 12,
            operand: 5,
            want: "PrefixV",
            binding: Some(SourceBinding::PrefixV { layer: 4 }),
            ..
        }
    ));
}

/// Two layers' caches of different widths cannot be one pool; a cache that is not a whole number
/// of heads cannot be addressed. The second layer here is unread by any step.
#[test]
fn a_kv_pool_of_two_widths_or_off_the_head_is_refused() {
    let mut l = attn_layer(256);
    l.input.sources.push(shape(PREFIX_CAPACITY, 768));
    l.input.sources.push(shape(PREFIX_CAPACITY, 768));
    l.bindings.push(SourceBinding::PrefixK { layer: 7 });
    l.bindings.push(SourceBinding::PrefixV { layer: 7 });
    let e = refused(&l, &["layer 7", "768", "512"]);
    assert!(matches!(
        e,
        CanonicalRefusal::KvCache(KvCacheRefusal::Width {
            layer: 7,
            cols: 768,
            want: 512
        })
    ));
    let facts = CudaModelFacts {
        head_dim: 384,
        ..QWEN3_5_FACTS
    };
    let e = refused_with(&rope_append_only(256, 512), facts, &["RopeAppend"]);
    assert!(
        matches!(
            e,
            CanonicalRefusal::KvHeadDim { .. }
                | CanonicalRefusal::KvCache(KvCacheRefusal::OffHead { .. })
        ),
        "{e:?}"
    );
}

/// A layer id above `u32::MAX` cannot be a tape's KV layer.
#[test]
fn a_kv_layer_above_u32_is_refused() {
    let mut l = attn_layer(256);
    l.input.sources.push(shape(PREFIX_CAPACITY, 512));
    l.input.sources.push(shape(PREFIX_CAPACITY, 512));
    let big = u64::from(u32::MAX) + 1;
    l.bindings.push(SourceBinding::PrefixK { layer: big });
    l.bindings.push(SourceBinding::PrefixV { layer: big });
    let e = refused(&l, &["4294967296", "u32"]);
    assert!(matches!(
        e,
        CanonicalRefusal::KvCache(KvCacheRefusal::LayerTooLarge { layer }) if layer == big
    ));
}

/// `AttnDecode(q, pk, pv, k, v)` alone over sources at `m` rows of `geom`: refused by its shape
/// before any operand is read.
fn attn_only(m: u32, g: ModelAttnGeometry) -> LoweredDecode {
    let mut l = with_hidden(LoweringInput {
        exported: None,
        sources: vec![
            shape(m, g.q_width()),
            shape(PREFIX_CAPACITY, g.kv_width()),
            shape(PREFIX_CAPACITY, g.kv_width()),
            shape(m, g.kv_width()),
            shape(m, g.kv_width()),
        ],
        ops: vec![OpDesc {
            op: SubOp::attn_decode(g, 0.0625, PREFIX_CAPACITY, AttnMask::Causal),
            m,
            inputs: (0..5).map(InputRef::Ext).collect(),
        }],
        result: 0,
    });
    l.bindings[1] = SourceBinding::PrefixK {
        layer: u64::from(LAYER),
    };
    l.bindings[2] = SourceBinding::PrefixV {
        layer: u64::from(LAYER),
    };
    l
}

/// One row is decode attention: the layer at `m = 1` lowers its attention to the decode kernel,
/// with the prefill kernel's operands at one row and the tape's KV regions.
#[test]
fn a_one_row_attention_lowers_to_the_decode_attention() {
    let c = lower_with(&attn_layer(1), QWEN3_5_FACTS).expect("the attention layer lowers at m = 1");
    let attn: Vec<_> = (c.steps.iter())
        .filter_map(|s| match s {
            CudaStep::AttnDecode(a) => Some(*a),
            CudaStep::AttnPrefill(_) => panic!("a one-row attention on the prefill kernel"),
            _ => None,
        })
        .collect();
    let kv = c.kv.as_ref().expect("the layer has a KV cache");
    let [a] = attn.as_slice() else {
        panic!("one decode attention, got {attn:?}");
    };
    assert_eq!(
        (
            a.rows,
            a.q_heads,
            a.kv_heads,
            a.head_dim,
            a.layer,
            a.scale()
        ),
        (1, 8, 2, 256, 0, 0.0625)
    );
    assert_eq!(a.block_table, kv.block_table.offset);
    assert_eq!(a.sequence, kv.sequence.offset);
    // The steps `tape_player`'s decode test reads: the query's rope, the append, the attention;
    // the projections GEMVs.
    assert!(matches!(
        c.steps[7..10],
        [
            CudaStep::Rope(_),
            CudaStep::RopeAppend(_),
            CudaStep::AttnDecode(_)
        ]
    ));
    for i in [1, 4, 6, 11] {
        assert!(
            matches!(c.steps[i], CudaStep::Gemv(GemmBf16 { m: 1, .. })),
            "step {i}: {:?}",
            c.steps[i]
        );
    }
    // That test reads `qn` (the rope's input), `kr` and `v` (the append's) and `attn` back from
    // the arena after the forward: no step after the one writing each overwrites it.
    let (CudaStep::Rope(rope), CudaStep::RopeAppend(append)) = (c.steps[7], c.steps[8]) else {
        unreachable!("matched above");
    };
    let row = |at: ArenaOffset, cols: u32| span(at.0, 2 * u64::from(cols));
    for (what, read, from) in [
        ("qn", row(rope.x, rope.cols), 7),
        ("kr", row(append.k_out, append.cols), 8),
        ("v", row(append.v, append.cols), 8),
        ("attn", row(a.out, a.q_heads * a.head_dim), 9),
    ] {
        for (i, s) in c.steps.iter().enumerate().skip(from + 1) {
            let e = s.extents().expect("extents fit");
            for w in [Some(e.writes), e.second_write].into_iter().flatten() {
                let (r0, w0) = (read.offset.0, w.offset.0);
                assert!(
                    w0 + w.bytes <= r0 || r0 + read.bytes <= w0,
                    "step {i} overwrites {what}, which the decode player test reads"
                );
            }
        }
    }
    let tape = static_tape(&c);
    tape.validate().expect("the one-row layer validates");
}

/// Rows that are neither one (the decode kernel's) nor a multiple of the prefill CTA's 128.
#[test]
fn an_attention_off_the_cta_rows_is_refused() {
    for rows in [2, 192] {
        let e = refused(
            &attn_only(rows, geom(8, 2, 256)),
            &["AttnDecode", &format!("rows={rows}"), "1", "128"],
        );
        assert!(
            matches!(
                e,
                CanonicalRefusal::AttnShape {
                    step: 0,
                    rows: r,
                    rule: AttnShapeRule::RowsOffTile { multiple: 128 },
                    ..
                } if r == rows
            ),
            "rows = {rows}"
        );
    }
}

#[test]
fn an_attention_of_another_head_dim_is_refused() {
    let e = refused(
        &attn_only(256, geom(16, 4, 128)),
        &["AttnDecode", "head_dim=128", "256"],
    );
    assert!(matches!(
        e,
        CanonicalRefusal::AttnShape {
            step: 0,
            head_dim: 128,
            rule: AttnShapeRule::HeadDim { want: 256 },
            ..
        }
    ));
}

#[test]
fn a_sliding_window_attention_is_refused() {
    let l = attn_layer_with(
        13,
        SubOp::attn_decode(geom(8, 2, 256), 0.0625, PREFIX_CAPACITY, WINDOW),
    );
    let e = refused(&l, &["AttnDecode", "sliding-window"]);
    assert!(matches!(
        e,
        CanonicalRefusal::SlidingWindow {
            step: 13,
            op: "AttnDecode"
        }
    ));
}

/// gpt-oss's sinks ride as a 6th input; CUDA's attention has none.
#[test]
fn an_attention_with_sinks_is_refused() {
    let mut l = attn_layer(256);
    l.input.sources.push(shape(1, 8));
    l.bindings.push(SourceBinding::Weight {
        id: 14,
        index: None,
    });
    l.input.ops[13].inputs.push(InputRef::Ext(14));
    let e = refused(&l, &["AttnDecode", "sinks"]);
    assert!(matches!(e, CanonicalRefusal::AttnSinks { step: 13 }));
}

/// Operand 3 must be a `RopeAppend`, of the attention's own layer, and operand 4 the V that append
/// read.
#[test]
fn an_attention_whose_new_k_or_v_is_not_its_rope_appends_is_refused() {
    let mut input = attn_layer_input(256);
    input.ops[13].inputs[3] = InputRef::Op(9);
    let mut l = attn_layer(256);
    l.input = input;
    let e = refused(&l, &["AttnDecode", "operand 3", "not a RopeAppend"]);
    assert!(matches!(
        e,
        CanonicalRefusal::AttnProducer {
            step: 13,
            operand: 3,
            why: ProducerRefusal::NotARopeAppend
        }
    ));

    let mut l = attn_layer(256);
    l.input.ops[13].inputs[4] = InputRef::Op(9);
    let e = refused(&l, &["AttnDecode", "operand 4", "RopeAppend wrote"]);
    assert!(matches!(
        e,
        CanonicalRefusal::AttnProducer {
            step: 13,
            operand: 4,
            why: ProducerRefusal::NotItsV
        }
    ));

    // The append writes layer 5's cache (sources 14, 15); the attention reads layer 3's.
    let mut l = attn_layer(256);
    l.input.sources.push(shape(PREFIX_CAPACITY, 512));
    l.input.sources.push(shape(PREFIX_CAPACITY, 512));
    l.bindings.push(SourceBinding::PrefixK { layer: 5 });
    l.bindings.push(SourceBinding::PrefixV { layer: 5 });
    l.input.ops[12].op = rope_append_op(256, 5, AttnMask::Causal, RopeFormTag::NeoX);
    l.input.ops[12].inputs[4] = InputRef::Ext(14);
    l.input.ops[12].inputs[5] = InputRef::Ext(15);
    let e = refused(&l, &["AttnDecode", "layer 5"]);
    assert!(matches!(
        e,
        CanonicalRefusal::AttnProducer {
            step: 13,
            operand: 3,
            why: ProducerRefusal::OtherLayer { layer: 5 }
        }
    ));
}

/// The attention's operand 2 must be its layer's `PrefixV`.
#[test]
fn an_attention_whose_prefix_operand_is_not_its_layers_is_refused() {
    let mut l = attn_layer(256);
    l.input.sources.push(shape(PREFIX_CAPACITY, 512));
    l.bindings.push(SourceBinding::Weight {
        id: 14,
        index: None,
    });
    l.input.ops[13].inputs[2] = InputRef::Ext(14);
    let e = refused(&l, &["AttnDecode", "operand 2", "PrefixV"]);
    assert!(matches!(
        e,
        CanonicalRefusal::KvOperand {
            step: 13,
            op: "AttnDecode",
            operand: 2,
            layer: 3,
            want: "PrefixV",
            binding: Some(SourceBinding::Weight { .. }),
        }
    ));
}

/// An attention whose kv heads are not the cache's: 8 query heads over 1 kv head of 256 lanes
/// read a cache the appends wrote as 2 heads.
#[test]
fn an_attention_of_another_kv_geometry_is_refused() {
    let l = attn_layer_with(
        13,
        SubOp::attn_decode(geom(8, 1, 256), 0.0625, PREFIX_CAPACITY, AttnMask::Causal),
    );
    let e = refused(&l, &["AttnDecode", "1 kv heads"]);
    assert!(matches!(
        e,
        CanonicalRefusal::KvGeometry {
            step: 13,
            op: "AttnDecode",
            kv_heads: 1,
            ..
        }
    ));
}

// ── The qwen3.5-0.8b linear-attention (GDN) layer ──────────────────

/// The refusal of `l` under the GDN facts, its message naming every value in `names`.
fn gdn_refused(l: &LoweredDecode, names: &[&str]) -> CanonicalRefusal {
    refused_with(l, QWEN3_5_GDN_FACTS, names)
}

/// [`QWEN3_5_GDN_FACTS`] with the GDN facts `f` changes.
fn gdn_facts(f: impl FnOnce(&mut CudaGdnFacts)) -> CudaModelFacts {
    let mut facts = QWEN3_5_GDN_FACTS;
    if let Some(g) = facts.gdn.as_mut() {
        f(g);
    }
    facts
}

#[test]
fn the_gdn_layer_statics_are_what_their_canonicals_lower_to() {
    for (name, tape, m) in [
        ("gdn_layer(128)", &GDN_LAYER_M128, 128),
        ("gdn_layer(256)", &GDN_LAYER_M256, 256),
    ] {
        assert_tape_is_lowered_with(name, tape, &gdn_layer(m), QWEN3_5_GDN_FACTS);
    }
}

/// The issue's predicted layout, read back from the lowering: 8 steps (the `a` and `b`
/// projections on the narrow GEMM), the arena totals, the bundle's 99,328 bytes, the GDN
/// sequence and scratch after the colours, and the scratch's four regions.
#[test]
fn the_gdn_layer_numbers() {
    use scratchy_target_cuda::step::gdn_scratch;
    for (tape, m, total, seq, scratch) in [
        (&GDN_LAYER_M128, 128u32, 7627008, 3416064, 3416320),
        (&GDN_LAYER_M256, 256, 15253760, 6832128, 6832384),
    ] {
        let c = lower_with(&gdn_layer(m), QWEN3_5_GDN_FACTS).expect("the GDN layer lowers");
        assert_eq!(c.total_bytes, total, "m = {m}");
        assert_eq!(c.weight_bytes[6], BUNDLE_BYTES, "m = {m}: the bundle");
        let names: Vec<_> = (c.steps.iter())
            .map(|s| match s {
                CudaStep::RmsNorm(_) => "RmsNorm",
                CudaStep::Gemm(_) => "Gemm",
                CudaStep::GemmNarrow(_) => "GemmNarrow",
                CudaStep::GatedDeltaNet(_) => "GatedDeltaNet",
                CudaStep::Add(_) => "Add",
                _ => "other",
            })
            .collect();
        assert_eq!(
            names,
            [
                "RmsNorm",
                "Gemm",
                "Gemm",
                "GemmNarrow",
                "GemmNarrow",
                "GatedDeltaNet",
                "Gemm",
                "Add"
            ],
            "m = {m}"
        );
        let gdn = tape.gdn.expect("the GDN layer has GDN state");
        assert_eq!((gdn.sequence.offset.0, gdn.sequence.bytes), (seq, 16));
        assert_eq!(gdn.scratch.offset.0, scratch);
        let m64 = u64::from(m);
        let s = gdn_scratch(m, 6144, 16, 2048).expect("the scratch fits u64");
        assert_eq!(
            (s.conv_out, s.g, s.beta, s.o, s.bytes),
            (
                0,
                m64 * 24576,
                m64 * 24576 + m64 * 64,
                m64 * 24576 + m64 * 128,
                m64 * 24576 + m64 * 128 + m64 * 8192
            ),
            "m = {m}"
        );
        assert_eq!(gdn.scratch.bytes, s.bytes);
        // The colours end where the sequence starts; the scratch is in the arena's last bytes.
        let colours_end = tape.arena.colour_offsets[7] + tape.arena.colour_bytes[7];
        assert_eq!(colours_end, seq);
        assert_eq!(scratch + s.bytes, total);
    }
}

/// Only the GDN layer's tapes have GDN state.
#[test]
fn only_the_gdn_layer_has_gdn_state() {
    for tape in [
        &TWO_GEMM_M128,
        &TWO_GEMM_M256,
        &THREE_GEMM_M256,
        &MLP_M128,
        &MLP_M256,
        &ATTN_GATE_M128,
        &ATTN_GATE_M256,
        &ATTN_LAYER_M128,
        &ATTN_LAYER_M256,
    ] {
        assert_eq!(tape.gdn, None);
    }
}

/// A narrow GEMM's `n` or `k` off its 8-wide vector, and an `n` between the narrow GEMM's 64 and
/// the tile's 256, which neither kernel takes.
#[test]
fn a_narrow_gemm_off_the_vector_is_refused() {
    let e = refused(
        &with_hidden(chain_input(256, 1024, &[12])),
        &["n = 12", "multiple of 8", "narrow"],
    );
    assert!(matches!(
        e,
        CanonicalRefusal::GemmShape {
            step: 0,
            n: 12,
            rule: GemmShapeRule::NarrowOffVector {
                dim: "n",
                value: 12,
                multiple: 8
            },
            ..
        }
    ));
    let e = refused(
        &with_hidden(chain_input(256, 100, &[16])),
        &["k = 100", "multiple of 8"],
    );
    assert!(matches!(
        e,
        CanonicalRefusal::GemmShape {
            step: 0,
            k: 100,
            rule: GemmShapeRule::NarrowOffVector { dim: "k", .. },
            ..
        }
    ));
    let e = refused(
        &with_hidden(chain_input(256, 1024, &[128])),
        &["n = 128", "multiple of 256"],
    );
    assert!(matches!(
        e,
        CanonicalRefusal::GemmShape {
            n: 128,
            rule: GemmShapeRule::OffTile { dim: "n", .. },
            ..
        }
    ));
}

/// The narrow GEMM's rows are not tiled: any `m` lowers to it, one row included (its bits for a
/// row do not depend on the row's place, so a decode's `a` and `b` projections keep the
/// prefill's).
#[test]
fn a_narrow_gemm_takes_any_rows() {
    let c = lower_with(&with_hidden(chain_input(37, 8, &[64])), QWEN3_5_FACTS)
        .expect("a 37-row narrow GEMM lowers");
    assert!(matches!(
        c.steps.as_slice(),
        [CudaStep::GemmNarrow(GemmBf16 {
            m: 37,
            n: 64,
            k: 8,
            ..
        })]
    ));
    let c = lower_with(&with_hidden(chain_input(1, 1024, &[16])), QWEN3_5_FACTS)
        .expect("a one-row narrow GEMM lowers");
    assert!(matches!(
        c.steps.as_slice(),
        [CudaStep::GemmNarrow(GemmBf16 {
            m: 1,
            n: 16,
            k: 1024,
            ..
        })]
    ));
}

/// A GEMM with an operand after its weight (the old `[act, wt, ..]` match took it silently). The
/// shared graph keeps a third `MatmulTile` input only for an fp8 weight's per-channel scale, so
/// that is the canonical: the operand count is refused before the weight's kind.
#[test]
fn a_gemm_with_an_extra_operand_is_refused() {
    let mut l = two_gemm(256);
    l.input.sources.push(shape(1, 1024));
    l.bindings
        .push(SourceBinding::WeightScale { id: 1, index: None });
    l.input.ops[0].op = SubOp::MatmulTile {
        n: 1024,
        weight: GemmWeight::Fp8Dynamic,
    };
    l.input.ops[0].inputs.push(InputRef::Ext(3));
    let e = refused(&l, &["MatmulTile", "operand 2", "past the operands"]);
    assert!(matches!(
        e,
        CanonicalRefusal::Operand {
            step: 0,
            op: "MatmulTile",
            operand: 2,
            why: OperandRefusal::Extra,
        }
    ));
}

/// The GDN op on facts with no GDN geometry.
#[test]
fn a_gdn_op_with_no_gdn_facts_is_refused() {
    let e = refused(&gdn_layer(256), &["GatedDeltaNet", "GDN geometry"]);
    assert!(matches!(e, CanonicalRefusal::GdnFacts { step: 5 }));
}

/// A bundle bound to a weight of no layer: the state pool is per layer.
#[test]
fn a_gdn_bundle_of_no_layer_is_refused() {
    let mut l = gdn_layer(256);
    l.bindings[6] = SourceBinding::Weight { id: 6, index: None };
    let e = gdn_refused(&l, &["GatedDeltaNet", "no model layer"]);
    assert!(matches!(e, CanonicalRefusal::GdnLayer { step: 5 }));
}

/// A bundle that is not a weight at all.
#[test]
fn a_gdn_bundle_that_is_not_a_weight_is_refused() {
    let mut l = gdn_layer(256);
    l.bindings[6] = SourceBinding::Cos { local: false };
    let e = gdn_refused(&l, &["GatedDeltaNet", "operand 4"]);
    assert!(matches!(
        e,
        CanonicalRefusal::Operand {
            step: 5,
            operand: 4,
            why: OperandRefusal::NotAWeight { .. },
            ..
        }
    ));
}

/// Every geometry rule of the GDN kernels, one at a time.
#[test]
fn every_gdn_shape_rule_is_refused() {
    let l = gdn_layer(256);
    let case = |facts: CudaModelFacts, names: &[&str]| match refused_with(&l, facts, names) {
        CanonicalRefusal::GdnShape { step: 5, rule } => rule,
        e => panic!("expected a GdnShape refusal at step 5, got: {e}"),
    };
    assert_eq!(
        case(gdn_facts(|g| g.head_k = 64), &["head_k = 64", "128"]),
        GdnShapeRule::HeadK { head_k: 64 }
    );
    assert_eq!(
        case(gdn_facts(|g| g.num_k_heads = 3), &["16 value heads over 3"]),
        GdnShapeRule::HeadsNotGrouped { nk: 3, nv: 16 }
    );
    assert_eq!(
        case(
            gdn_facts(|g| g.conv_kernel = 9),
            &["conv kernel = 9", "2..=8"]
        ),
        GdnShapeRule::ConvKernel { k: 9 }
    );
    assert_eq!(
        case(gdn_facts(|g| g.conv_kernel = 1), &["conv kernel = 1"]),
        GdnShapeRule::ConvKernel { k: 1 }
    );
    // 8 key heads of 128 with 2048 value lanes: q | k | v would be 4096, and qkv is 6144.
    assert_eq!(
        case(gdn_facts(|g| g.num_k_heads = 8), &["qkv is 6144", "4096"]),
        GdnShapeRule::Widths {
            conv_dim: 6144,
            want: 4096
        }
    );
}

/// A GDN layer of `nv` value heads over `value_dim` lanes (`a`, `b` `[m, nv]`, `z` and `core`
/// `[m, value_dim]`), its `qkv` `[m, conv_dim]`.
fn gdn_layer_of(m: u32, conv_dim: u32, value_dim: u32, nv: u32) -> LoweredDecode {
    let mut l = gdn_layer(m);
    l.input.sources[2] = shape(H, conv_dim);
    l.input.sources[3] = shape(H, value_dim);
    l.input.sources[4] = shape(H, nv);
    l.input.sources[5] = shape(H, nv);
    l.input.sources[7] = shape(value_dim, H);
    let matmul = |n| SubOp::MatmulTile {
        n,
        weight: GemmWeight::Dense,
    };
    l.input.ops[1].op = matmul(conv_dim);
    l.input.ops[2].op = matmul(value_dim);
    l.input.ops[3].op = matmul(nv);
    l.input.ops[4].op = matmul(nv);
    l
}

/// A value head dim off the scan's 32-row slice, or above the norm's 256 lanes; a value width
/// that is not a whole number of value heads. Every projection's width is one a GEMM takes, so
/// the GDN op is the first refused.
#[test]
fn a_gdn_value_head_off_the_rules_is_refused() {
    // 32 heads of 40 lanes: value 1280, qkv 2·2048 + 1280 = 5376 (both multiples of 256).
    let e = gdn_refused(&gdn_layer_of(256, 5376, 1280, 32), &["head_v = 40"]);
    assert!(matches!(
        e,
        CanonicalRefusal::GdnShape {
            step: 5,
            rule: GdnShapeRule::HeadV { head_v: 40 }
        }
    ));
    // 16 heads of 512 lanes.
    let e = gdn_refused(&gdn_layer_of(256, 12288, 8192, 16), &["head_v = 512"]);
    assert!(matches!(
        e,
        CanonicalRefusal::GdnShape {
            rule: GdnShapeRule::HeadV { head_v: 512 },
            ..
        }
    ));
    // 2048 value lanes over 48 heads (three per key head): not a whole number per head.
    let e = gdn_refused(
        &gdn_layer_of(256, 6144, 2048, 48),
        &["48 value heads over 16 key heads"],
    );
    assert!(matches!(
        e,
        CanonicalRefusal::GdnShape {
            rule: GdnShapeRule::HeadsNotGrouped { nk: 16, nv: 48 },
            ..
        }
    ));
}

/// A GDN op whose `z` has other rows than the output: `z` is a per-head view `[2m, 2048]` of a
/// `[m, 4096]` projection, so every step before the GDN op lowers.
#[test]
fn a_gdn_op_whose_rows_disagree_is_refused() {
    use InputRef::{Ext, Op};
    let m = 256;
    let matmul = |n| SubOp::MatmulTile {
        n,
        weight: GemmWeight::Dense,
    };
    let op = |op, inputs| OpDesc { op, m, inputs };
    let mut l = with_hidden(LoweringInput {
        exported: None,
        sources: vec![
            shape(m, H),
            shape(H, 6144),
            shape(H, 4096),
            shape(H, 16),
            shape(1, 1),
        ],
        ops: vec![
            op(matmul(6144), vec![Ext(0), Ext(1)]),
            op(matmul(4096), vec![Ext(0), Ext(2)]),
            op(
                SubOp::Reshape {
                    rows: RowScale::Times(NonZeroU32::new(2).expect("2")),
                    cols: 2048,
                },
                vec![Op(1)],
            ),
            op(matmul(16), vec![Ext(0), Ext(3)]),
            op(
                SubOp::GatedDeltaNet,
                vec![Op(0), Op(2), Op(3), Op(3), Ext(4)],
            ),
        ],
        result: 4,
    });
    l.bindings[4] = SourceBinding::Weight {
        id: 4,
        index: Some(UnrollIndex(0)),
    };
    let e = gdn_refused(&l, &["GatedDeltaNet", "rows"]);
    assert!(matches!(
        e,
        CanonicalRefusal::GdnShape {
            step: 4,
            rule: GdnShapeRule::RowsDisagree,
        }
    ));
}

/// A GDN op over more than `i32::MAX` elements of `qkv`: 349568 rows (a multiple of the GEMM's
/// 128, so the projections lower) of 6144 lanes.
#[test]
fn a_gdn_op_above_int_max_elements_is_refused() {
    let e = gdn_refused(&gdn_layer(349568), &["2147745792"]);
    assert!(matches!(
        e,
        CanonicalRefusal::GdnShape {
            step: 5,
            rule: GdnShapeRule::TooManyElements {
                elements: 2147745792
            }
        }
    ));
}

/// Two GDN layers of different value widths in one canonical: one scratch and one state pool
/// cannot hold both. The second layer has 9b's 32 value heads over 16 key heads, and reads the
/// first's output.
#[test]
fn two_gdn_layers_of_two_geometries_are_refused() {
    use InputRef::{Ext, Op};
    let mut input = common::gdn_layer::gdn_layer_input(256);
    let base = input.sources.len();
    input.sources.extend([
        shape(1, H),
        shape(H, 8192),
        shape(H, 4096),
        shape(H, 32),
        shape(H, 32),
        shape(1, 1),
    ]);
    let matmul = |n| SubOp::MatmulTile {
        n,
        weight: GemmWeight::Dense,
    };
    let op = |op, inputs| OpDesc { op, m: 256, inputs };
    input.ops.push(op(
        SubOp::RmsNorm {
            eps: 1e-6,
            gain: GainConvention::OnePlusScale,
        },
        vec![Op(7), Ext(base)],
    ));
    for (n, w) in [(8192, 1), (4096, 2), (32, 3), (32, 4)] {
        input.ops.push(op(matmul(n), vec![Op(8), Ext(base + w)]));
    }
    input.ops.push(op(
        SubOp::GatedDeltaNet,
        vec![Op(9), Op(10), Op(11), Op(12), Ext(base + 5)],
    ));
    input.result = input.ops.len() - 1;
    let mut l = with_hidden(input);
    for (ix, layer) in [(6, 0), (base + 5, 1)] {
        l.bindings[ix] = SourceBinding::Weight {
            id: ix as u32,
            index: Some(UnrollIndex(layer)),
        };
    }
    let e = gdn_refused(&l, &["another geometry"]);
    assert!(matches!(e, CanonicalRefusal::GdnPool { step: 13 }), "{e}");
}

/// `GdnPool` compares the GDN op's geometry with the one the scratch was sized by, not the
/// scratch's bytes: `lower_step` on the GDN layer's step 5, given regions of the right bytes but a
/// recorded geometry of 32 value heads, refuses it; given its own geometry, lowers it. No real
/// canonical reaches the first case (no other geometry at 256 rows has the same scratch bytes), so
/// the lowering function is called directly.
#[test]
fn a_gdn_op_is_checked_against_the_scratchs_geometry_not_its_bytes() {
    let l = gdn_layer(256);
    let at = CanonicalAt {
        stem: "gdn_layer",
        m: 256,
    };
    let c = lower_canonical(&l, QWEN3_5_GDN_FACTS, at).expect("the GDN layer lowers");
    let gdn = c.gdn.expect("the GDN layer has GDN state");
    let p = plan_canonical(&l, QWEN3_5_GDN_FACTS, at).expect("the GDN layer plans");
    let computes: Vec<TapeStep<'_>> = (p.tape.instrs().iter())
        .filter_map(|i| match i {
            Instr::Compute {
                node,
                writes,
                inputs,
                ..
            } => Some((&p.graph.nodes[node.index()], *writes, inputs)),
            _ => None,
        })
        .enumerate()
        .map(|(step, (node, writes, inputs))| TapeStep {
            step,
            node,
            writes,
            inputs,
        })
        .collect();
    let lower = |geometry| {
        let cx = StepCx {
            graph: &p.graph,
            bindings: &p.lowered.bindings,
            colouring: &p.colouring,
            facts: p.facts,
            tokens: Some(256),
            offsets: Ok(&c.colour_offsets),
            rotary: None,
            rotary_local: None,
            kv: Ok(None),
            kv_regions: None,
            gdn_layers: &[0],
            gdn_regions: Some(GdnRegions {
                sequence: gdn.sequence,
                scratch: gdn.scratch,
                geometry,
            }),
            moe: None,
        };
        lower_step(&cx, &computes[5], &computes)
    };
    let own = GdnWidths {
        conv_dim: 6144,
        value_dim: 2048,
        nv: 16,
    };
    let lowered = lower(own);
    assert!(
        matches!(lowered, Ok(Some(CudaStep::GatedDeltaNet(_)))),
        "{lowered:?}"
    );
    let other = lower(GdnWidths { nv: 32, ..own });
    eprintln!("refused: {other:?}");
    assert!(
        matches!(other, Err(CanonicalRefusal::GdnPool { step: 5 })),
        "{other:?}"
    );
}

// ── The 64-column GEMM (Gemma 4) ───────────────────────────────────

/// The one GEMM step `chain_input(m, k, &[n])` lowers to under `facts`.
fn one_gemm(m: u32, k: u32, n: u32, facts: CudaModelFacts) -> CudaStep {
    let c = lower_with(&with_hidden(chain_input(m, k, &[n])), facts)
        .unwrap_or_else(|e| panic!("m={m} k={k} n={n}: {e}"));
    let [step] = c.steps.as_slice() else {
        panic!("expected one step, got {:?}", c.steps);
    };
    *step
}

/// Under Gemma 4's facts, Gemma's router (128), expert (704) and dense MLP (2112) widths, each a
/// multiple of 64 and not of 256, lower to the 64-column GEMM at m = 256 (#885).
#[test]
fn a_gemma_gemm_off_the_256_tile_lowers_to_the_64_column_gemm() {
    for n in [128, 704, 2112] {
        let step = one_gemm(256, 2816, n, GEMMA4_FACTS);
        let CudaStep::GemmCol64(g) = step else {
            panic!("n = {n}: expected GemmCol64, got {step:?}");
        };
        assert_eq!((g.m, g.n, g.k), (256, n, 2816), "n = {n}");
    }
}

/// Under Gemma 4's facts, an `n` that is a multiple of 64 and not of 256 but whose `k` is off the
/// 64-column GEMM's 64 is refused under its own rule, naming the 64-column GEMM; an `n` off 64
/// (200) or a multiple of 32 only (96) keeps today's tiled refusal, since it is not the 64-column
/// GEMM's.
#[test]
fn a_gemma_gemm_off_the_64_column_tile_is_refused_by_its_rule() {
    let e = refused_with(
        &with_hidden(chain_input(256, 100, &[128])),
        GEMMA4_FACTS,
        &["k = 100", "multiple of 64 (the 64-column GEMM)"],
    );
    assert!(matches!(
        e,
        CanonicalRefusal::GemmShape {
            step: 0,
            k: 100,
            rule: GemmShapeRule::Col64OffTile {
                dim: "k",
                value: 100,
                tile: 64
            },
            ..
        }
    ));
    let e = refused_with(
        &with_hidden(chain_input(192, 1024, &[128])),
        GEMMA4_FACTS,
        &["m = 192", "multiple of 128 (the 64-column GEMM)"],
    );
    assert!(matches!(
        e,
        CanonicalRefusal::GemmShape {
            rule: GemmShapeRule::Col64OffTile {
                dim: "m",
                value: 192,
                tile: 128
            },
            ..
        }
    ));
    for n in [200, 96] {
        let e = refused_with(
            &with_hidden(chain_input(256, 1024, &[n])),
            GEMMA4_FACTS,
            &[&format!("n = {n}"), "multiple of 256"],
        );
        assert!(
            matches!(
                e,
                CanonicalRefusal::GemmShape {
                    rule: GemmShapeRule::OffTile { dim: "n", .. },
                    ..
                }
            ),
            "n = {n}: {e:?}"
        );
    }
}

/// Under Gemma 4's facts the 64-column GEMM takes only m ≥ 2: one row is still the GEMV, and an
/// `n` that is a multiple of 256 is still the tiled GEMM.
#[test]
fn a_gemma_gemm_of_one_row_or_on_the_256_tile_keeps_its_kernel() {
    let step = one_gemm(1, 2816, 128, GEMMA4_FACTS);
    assert!(matches!(step, CudaStep::Gemv(_)), "{step:?}");
    let step = one_gemm(256, 2816, 1024, GEMMA4_FACTS);
    assert!(matches!(step, CudaStep::Gemm(_)), "{step:?}");
}

/// The 64-column GEMM is Gemma 4's alone: under Qwen's facts, n = 128 is still refused as off the
/// 256-column tile, beside the Gemma fixture that lowers it.
#[test]
fn the_64_column_gemm_is_off_under_qwen_facts() {
    let l = with_hidden(chain_input(256, 1024, &[128]));
    let e = refused_with(&l, QWEN3_5_FACTS, &["n = 128", "multiple of 256"]);
    assert!(matches!(
        e,
        CanonicalRefusal::GemmShape {
            n: 128,
            rule: GemmShapeRule::OffTile { dim: "n", .. },
            ..
        }
    ));
    let step = one_gemm(256, 1024, 128, GEMMA4_FACTS);
    assert!(matches!(step, CudaStep::GemmCol64(_)), "{step:?}");
}
