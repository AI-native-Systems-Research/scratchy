// SPDX-License-Identifier: Apache-2.0
//! 🛑🛑 THE DATAFLOWIR BRIDGE MAY NEVER REFUSE AT RUNTIME.
//!
//! ⛔⛔ A REFUSAL MOVES THE STOP EARLIER THAN dbo-opt, AND dbo-opt IS THE ONLY ORACLE. The pipeline
//! is `SubtileTape -> DataflowIR -> dbo-opt -> init_binary`. Every defect this bridge has ever fixed
//! was named by a dbo-opt refusal on our emitted MLIR — `vector<64xi1>` vs `i1`, "found no program
//! to compile", the SSA-scoping pair, "Unable to generate loops", the `ProgramUnitsReduction`
//! assertion, "Dangling non-compute op has no use". A lowering that returns `Err`, or panics, stops
//! BEFORE the tape is emitted. dbo-opt is never invoked, the sentence we needed is never produced,
//! and the build prints OUR message instead of the backend's.
//!
//! ⛔⛔ AND IT HAS HAPPENED FIVE TIMES. Each time it cost hours and a revert. The last one added one
//! variant to an error type that already existed — a two-line change that looked exactly like the
//! variants beside it — and the loop went blind until the raw `-vv` stream was read line by line.
//! Prose in `CLAUDE.md` did not prevent any of the five. This test is the thing that does.
//!
//! ⭐⭐ TWO CATEGORIES, AND ONLY ONE OF THEM IS AT ZERO.
//!
//! A `Result` is a VALUE. The caller can log it and carry on — and that is precisely what happened:
//! `codegen.rs` matched on the lowering's `Err`, printed it, and continued to the next stage, so the
//! build failed somewhere else entirely and the refusal read like a note. **`Err(`, `.ok_or` and
//! `Result<` are frozen at ZERO.** There is no error type in the bridge any more; adding one back
//! means adding a whole `enum` to a diff, which is visible.
//!
//! A panic is LOUD AND UNSWALLOWABLE. It still stops before dbo-opt, so it is still a stop worth
//! removing — but nobody can mistake it for a lowering that ran. `panic!` and `todo!` are capped at
//! what the bridge carries today and ratcheted DOWN. They are tolerated to make progress, not
//! because they are right; the goal is zero.
//!
//! ⭐ IT IS A RATCHET. Every count may go DOWN freely — each one removed is a stop that no longer
//! pre-empts the oracle. None may go UP: a new one fails with the file, the construct, the old count
//! and the new one.

use std::path::Path;

/// One construct that stops the lowering, and how many of it the bridge carries today.
struct Ratchet {
    /// The token as it appears in source.
    construct: &'static str,
    /// How many occurrences exist right now.
    today: usize,
}

/// ⭐⭐ THE FROZEN COUNTS, measured after `DfirError` was deleted.
///
/// ⛔ EACH IS A STOP THAT PRE-EMPTS dbo-opt, and removing it is the work of making the lowering
/// total. Enumerated rather than summed so a failure names WHICH construct grew.
const FROZEN: &[Ratchet] = &[
    // ⛔⛔ ZERO, AND STAYING THERE. A refusal returned as a value is one a caller can log and carry
    // on from, which is how the bridge went blind five times.
    Ratchet {
        construct: "Err(",
        today: 0,
    },
    Ratchet {
        construct: ".ok_or",
        today: 0,
    },
    Ratchet {
        construct: "Result<",
        today: 0,
    },
    // ⭐ CAPPED, NOT PERMITTED. Loud and unswallowable, so they can be lived with while the lowering
    // is made total — but every one is still a stop before the oracle, and the target is zero.
    Ratchet {
        construct: "panic!",
        today: 7,
    },
    // ⭐ TWELVE `SubOp`s WITH NO DECOMPOSITION YET, one `todo!` each so the message names the op.
    // Not a stand-in op-func: emitting a Moe as a copy is a program dbo-opt compiles and a model
    // that produces garbage. This number goes down as decompositions get written.
    Ratchet {
        construct: "todo!",
        today: 12,
    },
    Ratchet {
        construct: "unimplemented!",
        today: 0,
    },
    Ratchet {
        construct: "unreachable!",
        today: 0,
    },
    Ratchet {
        construct: ".expect(",
        today: 1,
    },
    Ratchet {
        construct: ".unwrap(",
        today: 0,
    },
];

/// The scratchy half of bridge 1 — the file that turns the tape into `deeptools`' vocabulary.
const BRIDGE: &str = "src/lower_subtile_tape_to_dataflow_ir.rs";

/// Count non-comment occurrences of `needle`.
///
/// ⛔ COMMENTS ARE SKIPPED SO THE RULE CAN BE WRITTEN DOWN. This file's own doc comments name every
/// construct it forbids; counting them would make the rule unstatable.
fn occurrences(source: &str, needle: &str) -> usize {
    source
        .lines()
        .filter(|line| {
            let t = line.trim_start();
            !t.starts_with("//") && !t.starts_with("/*") && !t.starts_with('*')
        })
        .map(|line| line.matches(needle).count())
        .sum()
}

/// ⭐⭐ NO NEW WAY FOR THE BRIDGE TO STOP BEFORE dbo-opt.
#[test]
fn the_dataflow_ir_bridge_gains_no_new_runtime_refusal() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(BRIDGE);
    let source = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("the bridge must be readable at {}: {e}", path.display()));

    let mut grew = Vec::new();
    for r in FROZEN {
        let now = occurrences(&source, r.construct);
        if now > r.today {
            grew.push(format!(
                "  `{}`: {} -> {} (+{})",
                r.construct,
                r.today,
                now,
                now - r.today
            ));
        }
    }

    assert!(
        grew.is_empty(),
        "🛑 A NEW RUNTIME REFUSAL ENTERED THE DATAFLOWIR BRIDGE ({BRIDGE}):\n{}\n\n\
         A refusal stops the lowering BEFORE the tape is emitted, so dbo-opt is never invoked and \
         the loop that drives this bridge goes blind — it has cost hours and a revert five times. \
         The fact you are trying to state belongs in a TYPE: an arity is an array length, a pairing \
         is a witness, an op set is a const the door proved. If the lowering genuinely cannot \
         produce a program yet, that is a decomposition to WRITE, not a value to return.\n\n\
         These counts may only go DOWN.",
        grew.join("\n")
    );
}

/// ⭐ AND THE RATCHET IS REAL — every construct it freezes is one the file could contain.
///
/// ⛔ A RATCHET WHOSE TOKENS NEVER APPEAR IS A TEST THAT MEASURES NOTHING. This asserts the counting
/// works by checking the constructs that ARE present are found, carrying the value: if `Err(`
/// silently stopped matching, the ratchet would read zero for everything and pass forever.
#[test]
fn the_ratchet_counts_what_is_actually_there() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(BRIDGE);
    let source = std::fs::read_to_string(&path).expect("the bridge must be readable");

    let present: Vec<&str> = FROZEN
        .iter()
        .filter(|r| r.today > 0)
        .map(|r| r.construct)
        .collect();
    assert!(
        !present.is_empty(),
        "the frozen table records no refusal at all — either the bridge became total (delete this \
         assertion and celebrate) or the table stopped describing the file"
    );
    for construct in present {
        assert!(
            occurrences(&source, construct) > 0,
            "`{construct}` is frozen at a non-zero count but is not found in {BRIDGE}: the counter \
             has stopped matching, so the ratchet is measuring nothing"
        );
    }
}
