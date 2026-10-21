// SPDX-License-Identifier: Apache-2.0
//! CUDA's canonical lowering on the shared fixtures: the shared passes run under CUDA's tables,
//! the one fold fires where it should and nowhere else, the conservative colour rule holds, and
//! opcode lowering refuses the first op it cannot lower.

use scratchy_subtile::fixtures::{
    add_only_input, front_end_lowered, gemm_m1_only_input, n_layer_input, one_layer_input,
    rmsnorm_only_input, rope_rotate_only_input, silu_mul_only_input,
};
use scratchy_subtile::handoff::{LoweredDecode, SourceBinding};
use scratchy_subtile::lower::{GemmWeight, InputRef, LoweringInput, OpDesc};
use scratchy_subtile::subtile_ir::{EwKind, GainConvention, SourceShape, SubOp};
use scratchy_subtile::subtile_tape::{ComputeInput, Instr};
use scratchy_subtile::tape_folding::FusedShape;
use scratchy_target_cuda::op_abi::CudaFusion;
use scratchy_target_cuda_compiler::canonical::{
    ArenaRefusal, CanonicalAt, CanonicalRefusal, CudaPlan, GemmShapeRule, lower_canonical,
    plan_canonical,
};

const AT: CanonicalAt<'static> = CanonicalAt {
    stem: "fixture",
    m: 1,
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
    plan_canonical(&front_end_lowered(input), AT)
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
/// one. An op with no lowering at all (`RopeRotate`) is refused as `NotLowered`.
#[test]
fn lower_canonical_refuses_the_first_computed_op() {
    let l = front_end_lowered(one_layer_input());
    let p = plan_canonical(&l, AT).expect("plan_canonical on one_layer_input");
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
    match lower_canonical(&l, AT) {
        Err(CanonicalRefusal::Arena(ArenaRefusal::NoEmbeddedHidden)) => {}
        Err(e) => panic!("expected the arena's NoEmbeddedHidden at step 0, got: {e}"),
        Ok(_) => panic!("lower_canonical lowered a layer with no embedded hidden"),
    }
    match lower_canonical(&with_hidden(one_layer_input()), AT) {
        Err(CanonicalRefusal::GemmShape {
            step: 1,
            m: 1,
            rule: GemmShapeRule::Decode,
            ..
        }) => {}
        Err(e) => panic!("expected the decode GEMM's refusal at step 1, got: {e}"),
        Ok(_) => panic!("lower_canonical lowered a decode GEMM CUDA has no kernel for"),
    }
    match lower_canonical(&front_end_lowered(rope_rotate_only_input()), AT) {
        Err(CanonicalRefusal::NotLowered {
            step: 0,
            op: "RopeRotate",
        }) => {}
        Err(e) => panic!("expected NotLowered at step 0 (RopeRotate), got: {e}"),
        Ok(_) => panic!("lower_canonical lowered a RopeRotate CUDA has no kernel for"),
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
    let p = plan_canonical(&l, AT).expect("plan_canonical on the MLP block");
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
    let c = lower_canonical(&l, AT).expect("the MLP block lowers");
    assert_eq!(c.steps.len(), 6, "seven ops, one absorbed: {:?}", c.steps);
}

/// A decode GEMM (`m = 1`) lowers to the bf16 GEMM's shape rule, which refuses it: no decode GEMV
/// kernel exists yet.
#[test]
fn lower_canonical_refuses_a_decode_gemm_by_its_shape() {
    match lower_canonical(&front_end_lowered(gemm_m1_only_input()), AT) {
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
