// SPDX-License-Identifier: Apache-2.0
//! CUDA's canonical lowering on the shared fixtures: the shared passes run under CUDA's tables,
//! the one fold fires where it should and nowhere else, the conservative colour rule holds, and
//! opcode lowering refuses the first op it cannot lower.

use scratchy_subtile::fixtures::{
    add_only_input, front_end_lowered, gemm_m1_only_input, n_layer_input, one_layer_input,
    rmsnorm_only_input, rope_rotate_only_input, silu_mul_only_input,
};
use std::num::NonZeroU32;

use scratchy_subtile::handoff::{
    LoweredDecode, SourceBinding, TargetBucket, TargetModelFacts, TargetModelInput, TargetPoint,
};
use scratchy_subtile::lower::{GemmWeight, InputRef, LoweringInput, OpDesc};
use scratchy_subtile::model_geometry::HeadDim;
use scratchy_subtile::subtile_ir::{
    EwKind, GainConvention, RowScale, SourceShape, SubOp, SubtileIR,
};
use scratchy_subtile::subtile_tape::{ComputeInput, ComputeInputs, Instr};
use scratchy_subtile::tape_colouring::{ColourFacts, ColourRule, colour_tape};
use scratchy_subtile::tape_folding::FusedShape;
use scratchy_target_cuda::op_abi::{CUDA_COLOUR_FACTS, CudaFusion, cuda_colour_rule};
use scratchy_target_cuda::step::{CudaStep, EmbedBf16, GemmBf16};
use scratchy_target_cuda::tape::{ArenaOffset, ArenaSpan, TapeInput, WeightIx};
use scratchy_target_cuda::weights::CudaWeight;
use scratchy_target_cuda_compiler::canonical::{
    ArenaRefusal, CanonicalAt, CanonicalRefusal, CudaEmbedFacts, CudaModelFacts, CudaPlan,
    EmbedShapeRule, GateShapeRule, GemmShapeRule, OperandRefusal, RopeShapeRule, SharedExpertOwner,
    lower_canonical, lower_plan, plan_canonical,
};
use scratchy_target_cuda_compiler::emit::{ModelRefusal, lower_model, static_tape};
use scratchy_target_cuda_compiler::steps_from_tape::{StepCx, TapeStep, lower_fused, lower_step};

const AT: CanonicalAt<'static> = CanonicalAt {
    stem: "fixture",
    m: 1,
};

const AT_128: CanonicalAt<'static> = CanonicalAt {
    stem: "fixture",
    m: 128,
};

/// qwen3.5's model facts: `head_dim` 256, a quarter of it rotated, `rope_theta` 1e7. A fixture
/// with no rope or gate reads none of them.
const FACTS: CudaModelFacts = CudaModelFacts {
    head_dim: 256,
    rot_dim: 64,
    rope_theta: 1e7,
    gdn: None,
    embed: None,
    shared_expert: SharedExpertOwner::Undeclared,
};

fn fixtures() -> Vec<(&'static str, LoweringInput)> {
    vec![
        ("one_layer", one_layer_input()),
        ("n_layer(3)", n_layer_input(3)),
        ("rmsnorm_only", rmsnorm_only_input()),
        ("add_only", add_only_input()),
        ("silu_mul_only", silu_mul_only_input()),
        ("rope_rotate_only", rope_rotate_only_input()),
        ("gemm_m1_only", gemm_m1_only_input()),
    ]
}

fn plan(name: &str, input: LoweringInput) -> CudaPlan {
    plan_canonical(&front_end_lowered(input), FACTS, AT)
        .unwrap_or_else(|e| panic!("{name}: plan_canonical refused: {e}"))
}

/// None of the shared fixtures has a `Silu` read by a `Mul` (the layers use the pre-fused
/// `SiluMul`), so CUDA's one fold absorbs nothing in any of them.
#[test]
fn plan_canonical_folds_nothing_in_a_fixture_with_no_silu_into_mul() {
    for (name, input) in fixtures() {
        let p = plan(name, input);
        assert!(
            p.folds.absorbed_ops().is_empty(),
            "{name}: no Silu feeds a Mul, yet the fold pass absorbed {:?}",
            p.folds.absorbed_ops()
        );
        assert!(
            p.colouring.count().get() >= 1,
            "{name}: the colouring uses no colour"
        );
    }
}

/// The conservative rule's promise: every op but a view writes a fresh buffer and reads its inputs
/// across the step, so no output shares a colour with one of its own computed inputs.
#[test]
fn no_output_shares_a_colour_with_its_input() {
    for (name, input) in fixtures() {
        let p = plan(name, input);
        let mut checked = 0;
        for instr in p.tape.instrs() {
            let Instr::Compute {
                node,
                writes,
                inputs,
                ..
            } = instr
            else {
                continue;
            };
            let op = &p.graph.nodes[node.index()].op;
            if matches!(op, SubOp::Reshape { .. }) {
                continue;
            }
            let out = p
                .colouring
                .colour_of(*writes)
                .unwrap_or_else(|| panic!("{name}: {} has no colour", op.name()));
            for ci in inputs.iter() {
                let ComputeInput::Computed(slots) = ci else {
                    continue;
                };
                for s in slots {
                    let c = p.colouring.colour_of(*s);
                    assert_ne!(
                        c,
                        Some(out),
                        "{name}: {} writes colour {} over its own input",
                        op.name(),
                        out.index()
                    );
                    checked += 1;
                }
            }
        }
        // Every multi-op fixture has a step reading an earlier step; the single-op ones read only
        // sources, and check nothing here.
        if matches!(name, "one_layer" | "n_layer(3)") {
            assert!(checked > 0, "{name}: no computed input was checked");
        }
    }
}

/// `one_layer_input`'s first op, a norm, has a lowering now. As the front end binds it (every
/// source a weight) the norm reaches the arena, which has no embedded hidden. With `x` rebound
/// to the embedded hidden, the norm and the three one-row GEMMs (GEMVs) lower, and the layer is
/// refused at its rope, whose cos the front end binds as a weight. `rope_rotate_only_input`'s
/// rope has a lowering too, and is refused at its `x`, a weight; an op with no lowering at all
/// (`Gelu`) is refused as `NotLowered`.
#[test]
fn lower_canonical_refuses_the_first_computed_op() {
    let l = front_end_lowered(one_layer_input());
    let p = plan_canonical(&l, FACTS, AT).expect("plan_canonical on one_layer_input");
    let first = p
        .tape
        .instrs()
        .iter()
        .find_map(|i| match i {
            Instr::Compute { node, .. } => Some(p.graph.nodes[node.index()].op.name()),
            _ => None,
        })
        .expect("one_layer_input's tape computes something");
    assert_eq!(first, "RmsNorm");
    match lower_canonical(&l, FACTS, AT) {
        Err(CanonicalRefusal::Arena(ArenaRefusal::NoEmbeddedHidden)) => {}
        Err(e) => panic!("expected the arena's NoEmbeddedHidden at step 0, got: {e}"),
        Ok(_) => panic!("lower_canonical lowered a layer with no embedded hidden"),
    }
    match lower_canonical(&with_hidden(one_layer_input()), FACTS, AT) {
        Err(CanonicalRefusal::RotaryOperand {
            step: 4,
            op: "RopeRotate",
            operand: 1,
            ..
        }) => {}
        Err(e) => panic!("expected the rope's cos refused at step 4 (a weight), got: {e}"),
        Ok(_) => panic!("lower_canonical lowered a rope whose cos is a weight"),
    }
    match lower_canonical(&front_end_lowered(rope_rotate_only_input()), FACTS, AT) {
        Err(CanonicalRefusal::Operand {
            step: 0,
            op: "RopeRotate",
            operand: 0,
            why: OperandRefusal::NotHiddenOrSlot { source: 0, .. },
        }) => {}
        Err(e) => panic!("expected the rope's x refused at step 0 (a weight), got: {e}"),
        Ok(_) => panic!("lower_canonical lowered a rope whose x is a weight"),
    }
    let gelu = LoweringInput {
        exported: None,
        sources: vec![SourceShape { rows: 1, cols: 64 }],
        ops: vec![OpDesc {
            op: SubOp::Elementwise(EwKind::Gelu),
            m: 1,
            inputs: vec![InputRef::Ext(0)],
        }],
        result: 0,
    };
    match lower_canonical(&with_hidden(gelu), FACTS, AT) {
        Err(CanonicalRefusal::NotLowered {
            step: 0,
            op: "Gelu",
        }) => {}
        Err(e) => panic!("expected NotLowered at step 0 (Gelu), got: {e}"),
        Ok(_) => panic!("lower_canonical lowered a Gelu CUDA has no kernel for"),
    }
}

/// `rope_rotate_only_input` with `x` rebound to the embedded hidden and its `cos` and `sin` to
/// the global rotary table lowers to one `Rope` launch, reading the table the arena placed after
/// its colours, at `head_dim` 64 (the op's) and the model's `rot_dim`.
#[test]
fn a_rope_on_the_hidden_with_the_global_table_lowers_to_one_rope() {
    let mut l = with_hidden(rope_rotate_only_input());
    l.bindings[1] = SourceBinding::Cos { local: false };
    l.bindings[2] = SourceBinding::Sin { local: false };
    let facts = CudaModelFacts {
        head_dim: 64,
        rot_dim: 32,
        rope_theta: 1e4,
        gdn: None,
        embed: None,
        shared_expert: SharedExpertOwner::Undeclared,
    };
    let c = lower_canonical(&l, facts, AT).expect("the rope lowers");
    let [CudaStep::Rope(rope)] = c.steps.as_slice() else {
        panic!("expected one Rope, got {:?}", c.steps);
    };
    let table = c
        .rotary
        .expect("a canonical binding the global cos has a table");
    assert_eq!(
        (rope.rows, rope.cols, rope.head_dim, rope.rot_dim),
        (1, 2048, 64, 32)
    );
    assert_eq!(rope.x.0, 0, "x is the embedded hidden, colour 0");
    assert_eq!(rope.table, table.span.offset);
    assert_eq!(table.span.bytes, 64, "one row of 32 bf16");
    assert_eq!((table.rot_dim, table.theta()), (32, 1e4));
    assert_eq!(c.total_bytes, table.span.offset.0 + 256);
    assert_eq!(
        c.weight_bytes,
        [0, 0, 0],
        "cos and sin are not uploaded as weights"
    );
}

/// qwen3.5-0.8b's attention prelude and gate at `m` rows, as `tests/common/attn_gate.rs` builds
/// it, its `GateApply` gating the rotated query.
fn attn_gate(m: u32) -> LoweredDecode {
    let norm = SubOp::RmsNorm {
        eps: 1e-6,
        gain: GainConvention::OnePlusScale,
    };
    let mm = |n| SubOp::MatmulTile {
        n,
        weight: GemmWeight::Dense,
    };
    let op = |op, inputs| OpDesc { op, m, inputs };
    use InputRef::{Ext, Op};
    let shape = |rows, cols| SourceShape { rows, cols };
    let mut l = with_hidden(LoweringInput {
        exported: None,
        sources: vec![
            shape(m, 1024),
            shape(1, 1024),
            shape(1024, 4096),
            shape(1, 256),
            shape(m, 2048),
            shape(m, 2048),
            shape(2048, 1024),
        ],
        ops: vec![
            op(norm, vec![Ext(0), Ext(1)]),
            op(mm(4096), vec![Op(0), Ext(2)]),
            op(SubOp::GateSplit { half_cols: 2048 }, vec![Op(1)]),
            op(
                SubOp::Reshape {
                    rows: RowScale::Times(NonZeroU32::new(8).expect("8 heads")),
                    cols: 256,
                },
                vec![Op(2)],
            ),
            op(norm, vec![Op(3), Ext(3)]),
            op(
                SubOp::Reshape {
                    rows: RowScale::Times(NonZeroU32::MIN),
                    cols: 2048,
                },
                vec![Op(4)],
            ),
            op(
                SubOp::rope_rotate(HeadDim::new(256)),
                vec![Op(5), Ext(4), Ext(5)],
            ),
            op(SubOp::GateApply, vec![Op(6), Op(2)]),
            op(mm(1024), vec![Op(7), Ext(6)]),
            op(SubOp::Elementwise(EwKind::Add), vec![Op(8), Ext(0)]),
        ],
        result: 9,
    });
    l.bindings[4] = SourceBinding::Cos { local: false };
    l.bindings[5] = SourceBinding::Sin { local: false };
    l
}

/// The gate, the `GateSplit`'s second output, has a colour of its own with the query's shape,
/// `[M, 2048]`: the shared colourer records it, and the arena sizes it with no CUDA-side case.
#[test]
fn the_attention_gates_second_colour_is_sized() {
    for m in [128, 256] {
        let p = plan_canonical(&attn_gate(m), FACTS, AT).expect("plan_canonical on attn_gate");
        let split = (p.tape.instrs().iter())
            .find_map(|i| match i {
                Instr::Compute { node, writes, .. }
                    if matches!(p.graph.nodes[node.index()].op, SubOp::GateSplit { .. }) =>
                {
                    Some(*writes)
                }
                _ => None,
            })
            .expect("the fixture splits the gate");
        let second = p
            .colouring
            .second_colour_of(split)
            .expect("GateSplit writes two buffers");
        assert_eq!(p.colouring.shape_of(second), Some((m, 2048)), "m = {m}");
    }
}

/// `input` as the shared front end binds it, with source 0 rebound to the embedded hidden.
fn with_hidden(input: LoweringInput) -> LoweredDecode {
    let mut l = front_end_lowered(input);
    l.bindings[0] = SourceBinding::EmbeddedHidden;
    l
}

/// qwen3.5-0.8b's MLP block at 256 rows, its SwiGLU as the macro emits it: `Silu` (op 3) and
/// `Mul` (op 4) apart.
fn mlp_block() -> LoweredDecode {
    let (m, h, i) = (256, 1024, 3584);
    let matmul = |n, act, w| OpDesc {
        op: SubOp::MatmulTile {
            n,
            weight: GemmWeight::Dense,
        },
        m,
        inputs: vec![act, InputRef::Ext(w)],
    };
    let ew = |kind, inputs| OpDesc {
        op: SubOp::Elementwise(kind),
        m,
        inputs,
    };
    with_hidden(LoweringInput {
        exported: None,
        sources: vec![
            SourceShape { rows: m, cols: h },
            SourceShape { rows: 1, cols: h },
            SourceShape { rows: h, cols: i },
            SourceShape { rows: h, cols: i },
            SourceShape { rows: i, cols: h },
        ],
        ops: vec![
            OpDesc {
                op: SubOp::RmsNorm {
                    eps: 1e-6,
                    gain: GainConvention::OnePlusScale,
                },
                m,
                inputs: vec![InputRef::Ext(0), InputRef::Ext(1)],
            },
            matmul(i, InputRef::Op(0), 2),
            matmul(i, InputRef::Op(0), 3),
            ew(EwKind::Silu, vec![InputRef::Op(1)]),
            ew(EwKind::Mul, vec![InputRef::Op(3), InputRef::Op(2)]),
            matmul(h, InputRef::Op(4), 4),
            ew(EwKind::Add, vec![InputRef::Op(5), InputRef::Ext(0)]),
        ],
        result: 6,
    })
}

/// The shared fold pass, under CUDA's table, folds the MLP's `Silu` (op 3) into its `Mul` (op 4)
/// and nothing else, as the `SiluMul` kernel with the projections apart; the lowered tape has one
/// launch fewer than the block has ops.
#[test]
fn the_silu_folds_into_the_mul_it_gates() {
    let l = mlp_block();
    let p = plan_canonical(&l, FACTS, AT).expect("plan_canonical on the MLP block");
    assert_eq!(p.folds.absorbed_ops(), [(3, 4)]);
    assert!(p.folds.epilogue_ops().is_empty());
    let slots: Vec<_> = (p.tape.instrs().iter())
        .filter_map(|i| match i {
            Instr::Compute { writes, .. } => Some(*writes),
            _ => None,
        })
        .collect();
    assert_eq!(
        slots.len(),
        7,
        "the MLP's tape computes each of its seven ops once"
    );
    let fusions: Vec<_> = p.folds.fusions().collect();
    let [(driver, fusion)] = fusions.as_slice() else {
        panic!("expected one fold, got {fusions:?}");
    };
    assert_eq!(*driver, slots[4]);
    assert_eq!(fusion.kernel, CudaFusion::SiluMul);
    assert_eq!(
        fusion.shape,
        FusedShape::Gated {
            activation: slots[3],
            gate: None,
            up: None
        }
    );
    let c = lower_canonical(&l, FACTS, AT).expect("the MLP block lowers");
    assert_eq!(c.steps.len(), 6, "seven ops, one absorbed: {:?}", c.steps);
}

/// A one-row GEMM over the embedded hidden: `x [1, k] · W` (`W` the IR's `[k, n]`).
fn gemm_m1(n: u32, k: u32) -> LoweredDecode {
    with_hidden(LoweringInput {
        exported: None,
        sources: vec![
            SourceShape { rows: 1, cols: k },
            SourceShape { rows: k, cols: n },
        ],
        ops: vec![OpDesc {
            op: SubOp::MatmulTile {
                n,
                weight: GemmWeight::Dense,
            },
            m: 1,
            inputs: vec![InputRef::Ext(0), InputRef::Ext(1)],
        }],
        result: 0,
    })
}

/// A decode GEMM (`m = 1`) of an `n` the narrow GEMM does not take lowers to the GEMV, with the
/// tiled GEMM's operands at `m = 1`: `x` colour 0, the weight source 1, the output colour 1.
#[test]
fn a_decode_gemm_lowers_to_the_gemv() {
    let c = lower_canonical(&gemm_m1(1024, 1024), FACTS, AT).expect("the one-row GEMM lowers");
    assert_eq!(
        c.steps,
        [CudaStep::Gemv(GemmBf16 {
            a: ArenaOffset(0),
            b: WeightIx(1),
            c: ArenaOffset(c.colour_offsets[1]),
            m: 1,
            n: 1024,
            k: 1024,
        })]
    );
    assert_eq!(
        c.weight_bytes,
        [0, 1024 * 1024 * 2],
        "the weight's [n, k] bf16"
    );
    // The shared fixture's Llama-1B q_proj shape lowers the same way.
    let c = lower_canonical(&front_end_lowered(gemm_m1_only_input()), FACTS, AT);
    match c {
        Err(CanonicalRefusal::Arena(ArenaRefusal::NoEmbeddedHidden)) => {}
        other => panic!("expected the fixture's x refused as no embedded hidden, got {other:?}"),
    }
    let c = lower_canonical(&with_hidden(gemm_m1_only_input()), FACTS, AT)
        .expect("gemm_m1_only lowers");
    assert!(matches!(
        c.steps.as_slice(),
        [CudaStep::Gemv(GemmBf16 {
            m: 1,
            n: 2048,
            k: 2048,
            ..
        })]
    ));
}

/// A one-row GEMM whose `n` the narrow GEMM takes stays on the narrow GEMM, as at every other
/// `m`: the GDN `a` and `b` projections keep the prefill row's bits.
#[test]
fn a_decode_gemm_of_a_narrow_n_stays_on_the_narrow_gemm() {
    let c = lower_canonical(&gemm_m1(16, 1024), FACTS, AT).expect("the narrow one-row GEMM lowers");
    assert!(matches!(
        c.steps.as_slice(),
        [CudaStep::GemmNarrow(GemmBf16 {
            a: ArenaOffset(0),
            b: WeightIx(1),
            m: 1,
            n: 16,
            k: 1024,
            ..
        })]
    ));
}

/// A GEMV's `k` off its 8-wide vector is refused by the GEMV's rule.
#[test]
fn a_gemv_off_the_vector_is_refused() {
    match lower_canonical(&gemm_m1(1024, 100), FACTS, AT) {
        Err(
            e @ CanonicalRefusal::GemmShape {
                step: 0,
                m: 1,
                n: 1024,
                k: 100,
                rule:
                    GemmShapeRule::GemvOffVector {
                        value: 100,
                        multiple: 8,
                    },
            },
        ) => {
            let text = e.to_string();
            assert!(text.contains("k = 100") && text.contains("GEMV"), "{text}");
        }
        Err(e) => panic!("expected the GEMV's k refusal at step 0, got: {e}"),
        Ok(c) => panic!("lowered a GEMV off its vector: {:?}", c.steps),
    }
}

// ── Refusals no earlier test reached ───────────────────────────────

/// `RopeRotate(x, cos, sin)` alone over `[tokens, cols]`: `x` the embedded hidden, `cos` and
/// `sin` the global table.
fn rope_only(tokens: u32, cols: u32, head_dim: u32) -> LoweredDecode {
    let shape = SourceShape { rows: tokens, cols };
    let mut l = with_hidden(LoweringInput {
        exported: None,
        sources: vec![shape, shape, shape],
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

fn refusal(l: &LoweredDecode, facts: CudaModelFacts) -> CanonicalRefusal {
    match lower_canonical(l, facts, AT) {
        Err(e) => {
            eprintln!("refused: {e}");
            e
        }
        Ok(c) => panic!("lowered a canonical the kernel cannot run: {c:?}"),
    }
}

/// A rope whose `cos` is a computed value, not a source: the kernel reads the tape's table.
#[test]
fn a_rope_whose_cos_is_computed_is_refused() {
    let shape = SourceShape {
        rows: 256,
        cols: 2048,
    };
    let mut l = with_hidden(LoweringInput {
        exported: None,
        sources: vec![shape, shape],
        ops: vec![
            OpDesc {
                op: SubOp::Elementwise(EwKind::Add),
                m: 256,
                inputs: vec![InputRef::Ext(0), InputRef::Ext(0)],
            },
            OpDesc {
                op: SubOp::rope_rotate(HeadDim::new(256)),
                m: 256,
                inputs: vec![InputRef::Ext(0), InputRef::Op(0), InputRef::Ext(1)],
            },
        ],
        result: 1,
    });
    l.bindings[1] = SourceBinding::Sin { local: false };
    let e = refusal(&l, FACTS);
    assert!(e.to_string().contains("computed value"), "{e}");
    assert!(matches!(
        e,
        CanonicalRefusal::RotaryOperand {
            step: 1,
            op: "RopeRotate",
            operand: 1,
            binding: None,
        }
    ));
}

/// A rope over more than `i32::MAX` elements: 65536 rows of 32768 lanes.
#[test]
fn a_rope_above_int_max_elements_is_refused() {
    let e = refusal(&rope_only(65536, 32768, 256), FACTS);
    assert!(e.to_string().contains("2147483648"), "{e}");
    assert!(matches!(
        e,
        CanonicalRefusal::RopeShape {
            step: 0,
            rule: RopeShapeRule::TooManyElements {
                elements: 2147483648
            },
            ..
        }
    ));
}

/// A gate split whose model `head_dim` is not a whole number of 16-byte vectors.
#[test]
fn a_gate_split_off_the_vector_is_refused() {
    let facts = CudaModelFacts {
        head_dim: 12,
        ..FACTS
    };
    let e = refusal(&attn_gate(256), facts);
    assert!(e.to_string().contains("multiple of 8"), "{e}");
    assert!(matches!(
        e,
        CanonicalRefusal::GateShape {
            step: 2,
            head_dim: 12,
            rule: GateShapeRule::HeadOffVector { multiple: 8 },
            ..
        }
    ));
}

/// A gate split over more than `i32::MAX` elements: `qg` is 65536 rows of 2 · 16384 lanes.
#[test]
fn a_gate_split_above_int_max_elements_is_refused() {
    let l = with_hidden(LoweringInput {
        exported: None,
        sources: vec![SourceShape {
            rows: 65536,
            cols: 32768,
        }],
        ops: vec![OpDesc {
            op: SubOp::GateSplit { half_cols: 16384 },
            m: 65536,
            inputs: vec![InputRef::Ext(0)],
        }],
        result: 0,
    });
    let e = refusal(&l, FACTS);
    assert!(e.to_string().contains("2147483648"), "{e}");
    assert!(matches!(
        e,
        CanonicalRefusal::GateShape {
            step: 0,
            rule: GateShapeRule::TooManyElements {
                elements: 2147483648
            },
            ..
        }
    ));
}

/// `ProducerMissing` cannot be reached through `lower_canonical`: `computes` there is every
/// `Compute` of the tape, and every computed operand is a slot one of them writes. So the lowering
/// function is called directly, with the `Silu` the `SiluMul` fold absorbed left out of
/// `computes`.
#[test]
fn a_fold_whose_absorbed_step_is_not_on_the_tape_is_refused() {
    let l = mlp_block();
    let p = plan_canonical(&l, FACTS, AT).expect("plan_canonical on the MLP block");
    let cx = StepCx {
        graph: &p.graph,
        bindings: &p.lowered.bindings,
        colouring: &p.colouring,
        facts: p.facts,
        tokens: Some(256),
        offsets: Err(ArenaRefusal::TooLarge),
        rotary: None,
        kv: Ok(None),
        kv_regions: None,
        gdn_layers: &[],
        gdn_regions: None,
        moe: None,
    };
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
    let (driver, fusion) = p.folds.fusions().next().expect("the MLP folds its SiluMul");
    let mul = (computes.iter())
        .find(|c| c.writes == driver)
        .expect("the driver is on the tape");
    let without_silu: Vec<TapeStep<'_>> = (computes.iter())
        .filter(|c| !matches!(c.node.op, SubOp::Elementwise(EwKind::Silu)))
        .map(|c| TapeStep {
            step: c.step,
            node: c.node,
            writes: c.writes,
            inputs: c.inputs,
        })
        .collect();
    let e = lower_fused(&cx, mul, fusion, &without_silu).expect_err("the Silu is missing");
    assert!(e.to_string().contains("producer slot"), "{e}");
    assert!(matches!(
        e,
        CanonicalRefusal::ProducerMissing { step: 4, op: "Mul" }
    ));
}

// ── The carried refusals of #762's review ──────────────────────────
//
// `PerLayer`, `SplitWriters`, `PartialRead`, `PartialWrite` and `UnsizedColour` cannot come out of
// a canonical under CUDA's tables: CUDA does not re-roll (so no operand has a `per_layer` table),
// its column block is the whole width (one writer per value), `lower_region` binds whole regions,
// and its colour rule gives a second output only to an op that states a second shape. Each is
// reached by changing one fact of a real plan and lowering that plan (`lower_plan`), or by calling
// `lower_step` on a step with one input replaced. `SeveralEmbeddedHidden` comes out of
// `lower_canonical` itself.

/// `y = x + x`, then `y + x`, over `[256, 2048]`: step 1's operand 0 is a computed slot, operand 1
/// the embedded hidden.
fn two_adds() -> LoweredDecode {
    let shape = SourceShape {
        rows: 256,
        cols: 2048,
    };
    let add = |inputs| OpDesc {
        op: SubOp::Elementwise(EwKind::Add),
        m: 256,
        inputs,
    };
    with_hidden(LoweringInput {
        exported: None,
        sources: vec![shape, shape],
        ops: vec![
            add(vec![InputRef::Ext(0), InputRef::Ext(0)]),
            add(vec![InputRef::Op(0), InputRef::Ext(0)]),
        ],
        result: 1,
    })
}

/// Two sources bound to the embedded hidden: colour 0 is one buffer.
#[test]
fn two_embedded_hidden_sources_are_refused() {
    let mut l = two_adds();
    l.bindings[1] = SourceBinding::EmbeddedHidden;
    let e = refusal(&l, FACTS);
    assert!(e.to_string().contains("several sources"), "{e}");
    assert!(matches!(
        e,
        CanonicalRefusal::Arena(ArenaRefusal::SeveralEmbeddedHidden)
    ));
}

/// Step 1's second operand reads only part of its tensor.
#[test]
fn an_operand_reading_part_of_its_tensor_is_refused() {
    let mut p = plan_canonical(&two_adds(), FACTS, AT).expect("the adds plan");
    p.graph.nodes[1].inputs[1].region.cols.len -= 8;
    let e = lower_plan(p).expect_err("a partial read");
    assert!(e.to_string().contains("part of its tensor"), "{e}");
    assert!(matches!(
        e,
        CanonicalRefusal::Operand {
            step: 1,
            op: "Add",
            operand: 1,
            why: OperandRefusal::PartialRead,
        }
    ));
}

/// Step 1 writes only part of its output tensor.
#[test]
fn a_step_writing_part_of_its_output_is_refused() {
    let mut p = plan_canonical(&two_adds(), FACTS, AT).expect("the adds plan");
    p.graph.nodes[1].output.region.rows.len -= 1;
    let e = lower_plan(p).expect_err("a partial write");
    assert!(e.to_string().contains("part of its output"), "{e}");
    assert!(matches!(
        e,
        CanonicalRefusal::PartialWrite { step: 1, op: "Add" }
    ));
}

/// The plan's `Compute`s, numbered as `lower_canonical` numbers them.
fn computes(p: &CudaPlan) -> Vec<TapeStep<'_>> {
    (p.tape.instrs().iter())
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
        .collect()
}

/// `lower_step` of [`two_adds`]'s step 1 with its inputs replaced by `inputs(original)`.
fn lower_second_add(
    inputs: impl FnOnce(&SubtileIR, &[TapeStep<'_>]) -> ComputeInputs,
) -> CanonicalRefusal {
    let p = plan_canonical(&two_adds(), FACTS, AT).expect("the adds plan");
    let cx = StepCx {
        graph: &p.graph,
        bindings: &p.lowered.bindings,
        colouring: &p.colouring,
        facts: p.facts,
        tokens: Some(256),
        offsets: Ok(&[0, 1048576, 2097152]),
        rotary: None,
        kv: Ok(None),
        kv_regions: None,
        gdn_layers: &[],
        gdn_regions: None,
        moe: None,
    };
    let all = computes(&p);
    let replaced = inputs(&p.graph, &all);
    let s = TapeStep {
        step: 1,
        node: all[1].node,
        writes: all[1].writes,
        inputs: &replaced,
    };
    match lower_step(&cx, &s, &all) {
        Err(e) => {
            eprintln!("refused: {e}");
            e
        }
        Ok(step) => panic!("lowered a step CUDA cannot bind: {step:?}"),
    }
}

/// An operand that is a re-rolled per-layer source: CUDA does not re-roll, so its tapes never
/// carry a `per_layer` table.
#[test]
fn a_per_layer_operand_is_refused() {
    let e = lower_second_add(|graph, all| {
        let first = all[1].inputs.iter().next().cloned().expect("operand 0");
        let x = graph.nodes[0].inputs[0];
        ComputeInputs::A2([
            first,
            ComputeInput::External {
                tensor: x.tensor,
                region: x.region,
                per_layer: vec![x.tensor, x.tensor],
            },
        ])
    });
    assert!(e.to_string().contains("per-layer"), "{e}");
    assert!(matches!(
        e,
        CanonicalRefusal::Operand {
            step: 1,
            operand: 1,
            why: OperandRefusal::PerLayer,
            ..
        }
    ));
}

/// An operand written by two steps: CUDA's column block is the whole width, so one step writes
/// each value.
#[test]
fn an_operand_of_two_writers_is_refused() {
    let e = lower_second_add(|_, all| {
        let second = all[1].inputs.iter().nth(1).cloned().expect("operand 1");
        ComputeInputs::A2([
            ComputeInput::Computed(vec![all[0].writes, all[0].writes]),
            second,
        ])
    });
    assert!(e.to_string().contains("2 writers"), "{e}");
    assert!(matches!(
        e,
        CanonicalRefusal::Operand {
            step: 1,
            operand: 0,
            why: OperandRefusal::SplitWriters(2),
            ..
        }
    ));
}

/// CUDA's colour rule, but with an add writing two buffers. An add states no second shape, so
/// its second colour has none.
fn add_with_two_outputs(op: &SubOp) -> ColourRule {
    match op {
        SubOp::Elementwise(EwKind::Add) => ColourRule::FRESH.reads_across().two_outputs(),
        op => cuda_colour_rule(op),
    }
}

/// A colour the colourer reports no shape for cannot be sized.
#[test]
fn a_colour_with_no_shape_is_refused() {
    let mut p = plan_canonical(&two_adds(), FACTS, AT).expect("the adds plan");
    let facts = ColourFacts {
        rule: add_with_two_outputs,
        ..CUDA_COLOUR_FACTS
    };
    p.colouring = colour_tape(
        &p.graph,
        &p.tape,
        &p.lowered.bindings,
        &p.folds.fold_facts(),
        &facts,
    )
    .expect("the colourer colours both adds' two outputs");
    let e = lower_plan(p).expect_err("an unsized colour");
    assert!(e.to_string().contains("has no shape"), "{e}");
    assert!(
        matches!(e, CanonicalRefusal::Arena(ArenaRefusal::UnsizedColour(_))),
        "{e}"
    );
}

// THE HEAD AND THE WEIGHT TABLE: a model whose input is token ids, embedded from a table, with a
// tied lm_head that reads the same table.

/// `x [m, 512]` (the embedded hidden, source 0) → `RmsNorm(x, g)` (`g` source 1) → `lm_head`
/// (`[512, 256]`, source 2): a tied head's shape, the table's `[256, 512]`.
fn tied_head(m: u32) -> LoweredDecode {
    let h = 512;
    with_hidden(LoweringInput {
        exported: None,
        sources: vec![
            SourceShape { rows: m, cols: h },
            SourceShape { rows: 1, cols: h },
            SourceShape { rows: h, cols: 256 },
        ],
        ops: vec![
            OpDesc {
                op: SubOp::RmsNorm {
                    eps: 1e-6,
                    gain: GainConvention::OnePlusScale,
                },
                m,
                inputs: vec![InputRef::Ext(0), InputRef::Ext(1)],
            },
            OpDesc {
                op: SubOp::MatmulTile {
                    n: 256,
                    weight: GemmWeight::Dense,
                },
                m,
                inputs: vec![InputRef::Op(0), InputRef::Ext(2)],
            },
        ],
        result: 1,
    })
}

/// [`tied_head`] at `ms`, as the macro hands it over: the norm under `model.norm`, the lm_head
/// under `head` (the embedding table's prefix when tied), a `vocab`-row table.
fn tied_model(ms: &[u32], head: &str, vocab: u32) -> TargetModelInput {
    TargetModelInput {
        stem: "fixture".to_string(),
        buckets: (ms.iter())
            .map(|&m| TargetBucket {
                point: TargetPoint { m, sk: 1 },
                lowered: tied_head(m),
            })
            .collect(),
        unlowered: Vec::new(),
        facts: TargetModelFacts {
            head_dim: 0,
            rot_dim: 0,
            rope_theta: None,
            rms_norm_eps: 1e-6,
            vocab_size: vocab,
            gdn: None,
            dsl_shared_expert_bases: Vec::new(),
        },
        weight_prefixes: vec![
            (1, None, "model.norm".to_string()),
            (2, None, head.to_string()),
        ],
        embed_prefix: Some("model.embed_tokens".to_string()),
    }
}

/// With a table, the tape's first step gathers the token ids into colour 0, the ids sit after
/// every other region, and the embedded hidden's source declares the table's bytes.
#[test]
fn a_model_with_a_table_embeds_its_token_ids_first() {
    let l = tied_head(128);
    let hidden = lower_canonical(&l, FACTS, AT_128).expect("the hidden-input tape lowers");
    let facts = CudaModelFacts {
        embed: Some(CudaEmbedFacts { vocab: 256 }),
        ..FACTS
    };
    let c = lower_canonical(&l, facts, AT_128).expect("the token-id tape lowers");
    assert_eq!(c.steps.len(), hidden.steps.len() + 1, "one more step");
    assert_eq!(
        &c.steps[1..],
        hidden.steps.as_slice(),
        "then the same steps"
    );
    let tokens = ArenaSpan {
        offset: ArenaOffset(hidden.total_bytes),
        bytes: 128 * 4,
    };
    assert_eq!(
        c.steps[0],
        CudaStep::Embed(EmbedBf16 {
            tokens: tokens.offset,
            table: WeightIx(0),
            out: ArenaOffset(0),
            rows: 128,
            hidden: 512,
            vocab: 256,
        })
    );
    assert_eq!(
        c.input,
        TapeInput::TokenIds {
            span: tokens,
            vocab: 256
        }
    );
    let colour_0 = ArenaSpan {
        offset: ArenaOffset(0),
        bytes: 128 * 512 * 2,
    };
    assert_eq!(hidden.input, TapeInput::Hidden(colour_0));
    assert_eq!(
        c.total_bytes,
        hidden.total_bytes + 512,
        "the arena grows by 4·bucket_m"
    );
    assert_eq!(c.weight_bytes[0], 256 * 512 * 2, "the table's bytes");
    assert_eq!(hidden.weight_bytes[0], 0, "no table without one");
    assert_eq!(&c.weight_bytes[1..], &hidden.weight_bytes[1..]);
}

/// An embed of a vocabulary above `i32::MAX` rows.
#[test]
fn an_embed_above_int_max_rows_is_refused() {
    let facts = CudaModelFacts {
        embed: Some(CudaEmbedFacts { vocab: 1 << 31 }),
        ..FACTS
    };
    let e = match lower_canonical(&tied_head(128), facts, AT_128) {
        Err(e) => e,
        Ok(c) => panic!("lowered an embed the kernel cannot run: {c:?}"),
    };
    assert!(e.to_string().contains("vocab = 2147483648"), "{e}");
    assert!(matches!(
        e,
        CanonicalRefusal::EmbedShape {
            rule: EmbedShapeRule::AboveIntMax { dim: "vocab", .. },
            ..
        }
    ));
}

/// The tied lm_head names the table's tensor: it reads the table's buffer, and its own index is
/// emptied. Both buckets are rewritten, and each tape still validates.
#[test]
fn a_tied_head_reads_the_tables_buffer() {
    let model = lower_model(&tied_model(&[128, 256], "model.embed_tokens", 256)).expect("lowers");
    let table = |tensor, shape| Some(CudaWeight::Bf16 { tensor, shape });
    assert_eq!(
        model.weights,
        [
            table("model.embed_tokens.weight", &[256, 512][..]),
            table("model.norm.weight", &[512][..]),
            None,
        ]
    );
    for c in &model.tapes {
        let m = c.bucket_m;
        assert_eq!(c.weight_bytes, [256 * 512 * 2, 1024, 0], "m={m}");
        let Some(CudaStep::Gemm(head)) = c.steps.last() else {
            panic!("m={m}: the tape ends with the lm_head");
        };
        assert_eq!(head.b, WeightIx(0), "m={m}: the lm_head reads the table");
        static_tape(c)
            .validate()
            .unwrap_or_else(|e| panic!("m={m}: {e}"));
    }
}

/// An untied head is its own tensor and keeps its index.
#[test]
fn an_untied_head_keeps_its_own_buffer() {
    let model = lower_model(&tied_model(&[128], "lm_head", 256)).expect("lowers");
    assert_eq!(
        model.weights[2],
        Some(CudaWeight::Bf16 {
            tensor: "lm_head.weight",
            shape: &[256, 512],
        })
    );
    assert_eq!(
        model.tapes[0].weight_bytes,
        [256 * 512 * 2, 1024, 256 * 512 * 2]
    );
}

/// The head names the table's tensor at another shape (a 512-row table, a 256-column head).
#[test]
fn one_tensor_at_two_shapes_is_refused() {
    let e = lower_model(&tied_model(&[128], "model.embed_tokens", 512)).expect_err("refused");
    assert!(e.to_string().contains("model.embed_tokens.weight"), "{e}");
    assert!(matches!(
        e,
        ModelRefusal::SharedTensor {
            ix: 2,
            first: 0,
            ..
        }
    ));
}

/// A config with no vocabulary has no table to embed its token ids from.
#[test]
fn a_model_with_no_vocab_is_refused() {
    let e = lower_model(&tied_model(&[128], "lm_head", 0)).expect_err("refused");
    assert!(e.to_string().contains("vocab_size"), "{e}");
    assert!(matches!(e, ModelRefusal::NoVocab));
}

/// A weight the handoff names no checkpoint prefix for, and a table with no prefix.
#[test]
fn a_weight_with_no_prefix_is_refused() {
    let mut input = tied_model(&[128], "lm_head", 256);
    input.weight_prefixes.remove(0);
    let e = lower_model(&input).expect_err("refused");
    assert!(matches!(e, ModelRefusal::NoPrefix { ix: 1, .. }), "{e}");
    let mut input = tied_model(&[128], "lm_head", 256);
    input.embed_prefix = None;
    let e = lower_model(&input).expect_err("refused");
    assert!(
        matches!(
            e,
            ModelRefusal::NoPrefix {
                ix: 0,
                binding: SourceBinding::EmbeddedHidden
            }
        ),
        "{e}"
    );
}

/// Two buckets binding another source at one index cannot share a weight table.
#[test]
fn buckets_binding_other_sources_are_refused() {
    let mut input = tied_model(&[128, 256], "lm_head", 256);
    input.buckets[1].lowered.bindings[1] = SourceBinding::Weight { id: 7, index: None };
    let e = lower_model(&input).expect_err("refused");
    assert!(matches!(e, ModelRefusal::Bindings { m: 256, ix: 1 }), "{e}");
}

/// A weight source that declares bytes and that no step reads.
#[test]
fn a_weight_no_step_reads_is_refused() {
    let mut input = tied_model(&[128], "lm_head", 256);
    let l = &mut input.buckets[0].lowered;
    l.input.sources.push(SourceShape { rows: 1, cols: 512 });
    l.bindings
        .push(SourceBinding::Weight { id: 3, index: None });
    let e = lower_model(&input).expect_err("refused");
    assert!(
        matches!(e, ModelRefusal::UnreadWeight { ix: 3, bytes: 1024 }),
        "{e}"
    );
}
