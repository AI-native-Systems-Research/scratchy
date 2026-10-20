// SPDX-License-Identifier: Apache-2.0
//! CUDA's canonical lowering on the shared fixtures: the shared passes run under CUDA's tables,
//! the conservative colour rule holds, and opcode lowering refuses the first op it meets.

use scratchy_subtile::fixtures::{
    add_only_input, front_end_lowered, gemm_m1_only_input, n_layer_input, one_layer_input,
    rmsnorm_only_input, rope_rotate_only_input, silu_mul_only_input,
};
use scratchy_subtile::lower::LoweringInput;
use scratchy_subtile::subtile_ir::SubOp;
use scratchy_subtile::subtile_tape::{ComputeInput, Instr};
use scratchy_target_cuda_compiler::canonical::{
    CanonicalAt, CanonicalRefusal, CudaPlan, lower_canonical, plan_canonical,
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

#[test]
fn plan_canonical_runs_every_fixture_with_no_fold() {
    for (name, input) in fixtures() {
        let p = plan(name, input);
        assert!(
            p.folds.absorbed_ops().is_empty(),
            "{name}: CUDA declares no fusion, yet the fold pass absorbed {:?}",
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
    match lower_canonical(&l, AT) {
        Err(CanonicalRefusal::NotLowered { step: 0, op }) => assert_eq!(op, first),
        Err(e) => panic!("expected NotLowered at step 0 ({first}), got: {e}"),
        Ok(_) => panic!("lower_canonical lowered a tape CUDA has no opcode for"),
    }
}
