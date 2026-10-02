// SPDX-License-Identifier: Apache-2.0
//! THE SHARED TAPE METAL LOWERS FROM, RE-ROLLED — in the target's own crate.
//!
//! ⛔ THIS USED TO LIVE IN `compiler/macros/codegen.rs`. It is metal codegen: it names
//! `scratchy_target_metal::from_tape` in every line. A `#[cfg(feature = "metal")]` island in the
//! shared compiler crate compiles only when that feature is on, which is how target code rots
//! there unnoticed.

/// THE SHARED TAPE METAL LOWERS FROM, RE-ROLLED.
///
/// ⭐ ONE RE-ROLL, ON THE SHARED TAPE, BEFORE ANY TARGET LOWERS. `reroll_subtile_tape` finds the
/// repeating layer body once, on the `SubtileTape` — the same call the SuperDSC emitter makes,
/// on the same object. Both targets then READ the loop off it. Metal used to re-discover the
/// same fact afterwards from its own `Instruction` stream (`apply_loop_compression` +
/// `detect_repeating_run`): a second search over a second representation, which could disagree
/// with the first and which no test compared.
///
/// What metal needs to emit from the shared tape.
pub struct TapeProgram {
    /// The ROLLED tape's items: what metal emits from.
    pub rolled: Vec<scratchy_target_metal::from_tape::TapeItem>,
    /// The UN-rolled tape's items: the reference the roll is proven against.
    pub unrolled: Vec<scratchy_target_metal::from_tape::TapeItem>,
    /// Every layer rolled by its class ([`scratchy_target_metal::from_tape::roll_layer_classes`]),
    /// when the tape's layers can be read that way. A cut like any other: kept only if it proves.
    pub layer_rolled: Option<Vec<scratchy_target_metal::from_tape::TapeItem>>,
    /// The graph the tape computes, and the UN-rolled tape: what the shared colourer reads.
    pub graph: scratchy_subtile::subtile_ir::SubtileIR,
    pub tape: scratchy_subtile::subtile_tape::SubtileTape,
}

/// Builds the tape through `scratchy_target_metal::from_tape` — `lower_region` →
/// `ValidatedGraph` → `lower_dag_to_tape`, the same three calls the SuperDSC emitter makes.
///
/// ⛔ A REFUSAL HERE IS A BUILD DEFECT, NOT A FALLBACK. Every arch is tape-scheduled
/// (`is_tape_scheduled` is unconditionally true), so there is no second schedule to drop back
/// to; an arch the shared lowering cannot express has to be *expressed*, and the panic names it.
pub fn tape_program(
    lowered: &scratchy_subtile::handoff::LoweredDecode,
    stem: &str,
    m: u64,
) -> TapeProgram {
    use scratchy_target_metal::from_tape;
    let graph = from_tape::graph_for_metal(&lowered.input);
    let refuse = |what: &str, e: from_tape::TapeLoweringError| -> ! {
        panic!("[m2-flip] {} m={m}: {what}: {e}", stem)
    };
    let tape = from_tape::tape_for_metal(&graph)
        .unwrap_or_else(|e| refuse("the shared lowering could not build a tape", e));

    // ⛔ THE COLOURING READS THE UNROLLED TAPE, AND IT HAS TO. A rolled tape holds ONE layer
    // body, so the source ops of layers 1..N appear in no step and would get no colour.
    let items = from_tape::items_of(&graph, &tape)
        .unwrap_or_else(|e| refuse("the tape holds a step metal cannot play", e));

    // ⭐ THE ROLLED TAPE IS WHAT METAL EMITS FROM. One re-roll, on the shared tape, by the same
    // call the SuperDSC emitter makes. The emitted instruction stream comes out rolled, so
    // nothing downstream has to search it for a repeating run.
    let rolled = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        scratchy_subtile::subtile_tape::reroll_subtile_tape(&tape, &graph)
    })) {
        Ok(r) => r,
        Err(_) => panic!(
            "[m2-flip] {} m={m}: the SHARED reroll panicked on this tape",
            stem
        ),
    };
    let rolled_items = from_tape::items_of(&graph, &rolled)
        .unwrap_or_else(|e| refuse("the rolled tape holds a step metal cannot play", e));
    let layer_rolled = from_tape::roll_layer_classes(&graph, &tape, &rolled_items, &items);
    TapeProgram {
        rolled: rolled_items,
        unrolled: items,
        layer_rolled,
        graph,
        tape,
    }
}
