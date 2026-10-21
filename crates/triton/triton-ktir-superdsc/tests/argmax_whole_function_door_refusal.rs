// SPDX-License-Identifier: Apache-2.0

//! ⛔ THE WHOLE-FUNCTION BODY CANNOT CROSS THE PER-`Program` DOOR — AND THE TEST THAT KEEPS
//! IT HONEST.
//!
//! `argmax_first_max`'s case row states `Program::ArgMax` — the SPLICE's own classification
//! for the production `kernels/argmax.py` (`SubOp::ArgMax`'s row), no longer a placeholder:
//! the splice produces this body, so the kind is stated like every spliced family's. The
//! kernel crosses through the WHOLE-FUNCTION door (`emit_whole`), which walks the function's
//! ops and never dispatches on the `Program`; the per-`Program` door REFUSES the kind by
//! name — an eight-parameter whole-function body (five whole-row reduces, the equality
//! chain, the fp32 island) is not a single-descriptor dispatch, and building one would be a
//! second, hand-rolled path. This file pins that refusal.
//!
//! The chain is driven from the configuration table (the same statement `bake_py` and
//! the banded twin read — a test that restated the fixture, the constexprs, or the grid
//! would be a second statement of a configuration, and a configuration restated in two
//! places drifts in one of them): source → ttir → KTIR → `to_ktir`'s passes → their
//! node. Then, over ONE node:
//!
//! * CONTROL — `emit_whole` with the node's own layout EMITS. The module is good, so
//!   the per-`Program` refusal below is the door's verdict about the body's shape,
//!   never a broken module masquerading as one.
//! * THE PIN — `emit_node` over the SAME node REFUSES, naming the whole-function door
//!   as the body's. If a future arm mints a per-`Program` dispatch for this kind, this
//!   is the test that goes red.

use ktir_superdsc::ktir_node::Program;

/// The node behind the row, through the same value-level legs `bake_py` runs — minus
/// the bake directory. `Err` carries the first refusing stage's message.
fn node_of(name: &str) -> ktir_superdsc::ktir_node::KtirNode {
    let (kspec, grid, door) = triton_ktir_superdsc::cases::case(name)
        .unwrap_or_else(|| panic!("`{name}` is not a configuration"));
    // This test drives BOTH doors, so a doorless row cannot run here by construction.
    let program = door.unwrap_or_else(|| panic!("`{name}` has no door kind to drive"));
    let src = std::fs::read_to_string(&kspec.file).map_err(|e| format!("read: {e}")).unwrap();
    let mut tt =
        triton_frontend::codegen::compile(&src, &kspec, triton_frontend::target::Target::spyre())
            .map_err(|e| format!("codegen: {e}"))
            .unwrap();
    triton_frontend::opt::make_ttir(&mut tt).map_err(|e| format!("make_ttir: {e}")).unwrap();
    let mut m = triton_ktir::from_ttir::convert(&tt)
        .map_err(|e| format!("from_ttir: {e}"))
        .unwrap();
    triton_ktir::make_ktir(&mut m, &grid).map_err(|e| format!("make_ktir: {e}")).unwrap();
    triton_ktir::passes::to_ktir::run(&mut m, &grid)
        .map_err(|e| format!("to_ktir: {e}"))
        .unwrap();
    triton_ktir_superdsc::node_for(&m, program)
        .map_err(|e| format!("{}: {}", e.stage, e.message))
        .unwrap()
}

#[test]
fn the_per_program_door_refuses_while_the_whole_door_emits() {
    let node = node_of("argmax_first_max");
    // CONTROL — the whole-function door emits over this node; a failure here means the
    // module broke, and the refusal below would prove nothing about the body's shape.
    let layout = triton_ktir_superdsc::layout::for_node(&node)
        .map_err(|e| format!("{}: {}", e.stage, e.message))
        .unwrap();
    let ops = triton_ktir_superdsc::emit_whole(&node, Some(&layout))
        .map_err(|e| format!("{}: {}", e.stage, e.message))
        .unwrap();
    assert!(
        !ops.is_empty(),
        "the whole-function door emitted no ops — an emission over nothing"
    );
    // THE PIN — the per-`Program` door refuses the whole-function body BY NAME, naming
    // `emit_whole` as its door. Whatever classification the row states, an eight-parameter
    // multi-op body is not a single-descriptor dispatch.
    let err = match triton_ktir_superdsc::emit_node(&node) {
        Err(e) => e,
        Ok(ops) => panic!(
            "the per-`Program` door must refuse the whole-kernel node, but it emitted {} \
             op(s) — a per-`Program` dispatch over this body is a second, hand-rolled path",
            ops.len()
        ),
    };
    assert!(
        err.message.contains("eight-parameter whole-function body"),
        "the per-`Program` refusal must name the whole-function door as the body's, got: \
         [{}] {} (door kind: {:?})",
        err.stage,
        err.message,
        Program::ArgMax
    );
}
