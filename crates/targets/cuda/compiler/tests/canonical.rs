// SPDX-License-Identifier: Apache-2.0
//! CUDA's canonical lowering on the shared fixtures: the shared passes run under CUDA's tables,
//! the one fold fires where it should and nowhere else, the conservative colour rule holds, and
//! opcode lowering refuses the first op it cannot lower.

use scratchy_subtile::fixtures::{
    add_only_input, front_end_lowered, gemm_m1_only_input, n_layer_input, one_layer_input,
    rmsnorm_only_input, rope_rotate_only_input, silu_mul_only_input,
};
use std::num::NonZeroU32;

use scratchy_subtile::handoff::{LoweredDecode, SourceBinding};
use scratchy_subtile::lower::{GemmWeight, InputRef, LoweringInput, OpDesc};
use scratchy_subtile::model_geometry::HeadDim;
use scratchy_subtile::subtile_ir::{EwKind, GainConvention, RowScale, SourceShape, SubOp};
use scratchy_subtile::subtile_tape::{ComputeInput, Instr};
use scratchy_subtile::tape_folding::FusedShape;
use scratchy_target_cuda::op_abi::CudaFusion;
use scratchy_target_cuda::step::CudaStep;
use scratchy_target_cuda_compiler::canonical::{
    ArenaRefusal, CanonicalAt, CanonicalRefusal, CudaModelFacts, CudaPlan, GateShapeRule,
    GemmShapeRule, OperandRefusal, RopeShapeRule, lower_canonical, plan_canonical,
};
use scratchy_target_cuda_compiler::steps_from_tape::{StepCx, TapeStep, lower_fused};

const AT: CanonicalAt<'static> = CanonicalAt {
    stem: "fixture",
    m: 1,
};

/// qwen3.5's model facts: `head_dim` 256, a quarter of it rotated, `rope_theta` 1e7. A fixture
/// with no rope or gate reads none of them.
const FACTS: CudaModelFacts = CudaModelFacts {
    head_dim: 256,
    rot_dim: 64,
    rope_theta: 1e7,
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
/// to the embedded hidden, the norm lowers and the layer is refused at its first GEMM, a decode
/// one. `rope_rotate_only_input`'s rope has a lowering too, and is refused at its `x`, a weight;
/// an op with no lowering at all (`Gelu`) is refused as `NotLowered`.
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
        Err(CanonicalRefusal::GemmShape {
            step: 1,
            m: 1,
            rule: GemmShapeRule::Decode,
            ..
        }) => {}
        Err(e) => panic!("expected the decode GEMM's refusal at step 1, got: {e}"),
        Ok(_) => panic!("lower_canonical lowered a decode GEMM CUDA has no kernel for"),
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

/// A decode GEMM (`m = 1`) lowers to the bf16 GEMM's shape rule, which refuses it: no decode GEMV
/// kernel exists yet.
#[test]
fn lower_canonical_refuses_a_decode_gemm_by_its_shape() {
    match lower_canonical(&front_end_lowered(gemm_m1_only_input()), FACTS, AT) {
        Err(CanonicalRefusal::GemmShape {
            step: 0,
            m: 1,
            n: 2048,
            k: 2048,
            rule: GemmShapeRule::Decode,
        }) => {}
        Err(e) => panic!("expected the m = 1 shape refusal at step 0, got: {e}"),
        Ok(_) => panic!("lower_canonical lowered a decode GEMM CUDA has no kernel for"),
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
