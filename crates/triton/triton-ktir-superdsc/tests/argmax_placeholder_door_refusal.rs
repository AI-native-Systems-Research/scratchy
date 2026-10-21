// SPDX-License-Identifier: Apache-2.0

//! ⛔ THE PLACEHOLDER DOOR KIND CANNOT LOWER — AND THE TEST THAT KEEPS IT HONEST.
//!
//! `argmax_first_max`'s case row states a door kind only because the signature demands
//! one: the kernel crosses through the WHOLE-FUNCTION door (`emit_whole`), which walks
//! the function's ops and never reads the `Program`. The stated value is a placeholder,
//! and a placeholder is honest only while the per-`Program` door REFUSES it loudly
//! rather than lowering some wrong body over the kernel's parameters. With the DOORLESS
//! set now empty, no row pins that refusal by merely existing — this file does.
//!
//! The chain is driven from the configuration table (the same statement `bake_py` and
//! the banded twin read — a test that restated the fixture, the constexprs, or the grid
//! would be a second statement of a configuration, and a configuration restated in two
//! places drifts in one of them): source → ttir → KTIR → `to_ktir`'s passes → their
//! node. Then, over ONE node:
//!
//! * CONTROL — `emit_whole` with the node's own layout EMITS. The module is good, so
//!   the per-`Program` refusal below is the door's verdict about the placeholder,
//!   never a broken module masquerading as one.
//! * THE PIN — `emit_node` over the SAME node REFUSES, stating the mismatch by number:
//!   the placeholder's elementwise body reads 2 tensor inputs, the kernel stages 7, and
//!   `split_regions` refuses exactly that. If a future row mints a placeholder whose
//!   per-`Program` run silently lowers, this is the test that goes red.

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
fn the_placeholder_door_kind_refuses_while_the_whole_door_emits() {
    let node = node_of("argmax_first_max");
    // CONTROL — the whole-function door emits over this node; a failure here means the
    // module broke, and the refusal below would prove nothing about the placeholder.
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
    // THE PIN — the per-`Program` door refuses the placeholder, naming the arity
    // mismatch by number. Whatever door kind the row states, it cannot lower this
    // kernel's 8-parameter function as one elementwise op.
    let err = match triton_ktir_superdsc::emit_node(&node) {
        Err(e) => e,
        Ok(ops) => panic!(
            "the placeholder door kind must refuse the whole-kernel node, but the \
             per-`Program` door emitted {} op(s) — a placeholder that lowers is a lying \
             door kind, not a placeholder",
            ops.len()
        ),
    };
    assert!(
        err.message.contains("tensor input(s) for an op that reads"),
        "the per-`Program` refusal must state the arity mismatch by number, got: \
         [{}] {} (door kind: {:?})",
        err.stage,
        err.message,
        Program::Elementwise(ktir_superdsc::ktir_node::Elementwise::Mul)
    );
}
