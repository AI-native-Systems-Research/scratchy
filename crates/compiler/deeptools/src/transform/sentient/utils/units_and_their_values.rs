// SPDX-License-Identifier: Apache-2.0
//
// ╔══════════════════════════════════════════════════════════════════════════════════════════════╗
// ║ CRUSTIFY SENTIENT-PASSES CAMPAIGN — READ THIS BEFORE YOU FILL AN ANCHOR IN THIS FILE.        ║
// ║ Campaign statement: crustify-senpass/TASK.md   ·   worklist: crustify-senpass/UNITS.tsv      ║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════╝
//
// 1. THE AUTHORITY IS THE C++ TREE, NOT THE EXTRACT.
//       /Users/nickm/git/deeptools-src/<file>:<line>     (deeptools|master|a0d29abbed — repo_info.txt)
//    Every citation below resolves against that revision. `crustify-senpass/cpp/sentient.cpp` says
//    WHICH functions are in scope and IN WHAT ORDER; its bodies were verified byte-identical to the
//    authority (656/656, 962,619 bytes, two negative controls), so either may be read — but the
//    authority file is the one that carries the surrounding declarations you will need.
//    ⛔ /Users/nickm/git/deeptools is a DIFFERENT revision. ⛔ The pod is not reachable from here.
//
// 2. THESE PASSES REWRITE SentientIR IN PLACE. They are NOT a conversion between rungs like bridges
//    1–4: input and output are both `src/islands/sentient/`. Expect to EXTEND that island — ops
//    gaining an assigned register, a pinned address, a rolled loop — not to emit into a new one.
//    WHY IT MATTERS: bridge 2's emission assigns NO registers (`index: None` in 39 of 41 sites),
//    faithfully, because the reference's SentientIR carries `regIndex = -1 : i32` in all 54
//    occurrences of the committed golden corpus. THESE passes turn -1 into a real register file and
//    index. Without them ProgIR gets -1 where an instruction needs a register and the backend
//    refuses with `Register initialization out of boundary` (observed on lxsu0:LRF0, l3lu:LBR2).
//
// 3. PORTED MEANS THE WHOLE FUNCTION INCLUDING ITS EFFECT. For an in-place pass the effect IS the
//    port: WHICH ops are rewritten, WHICH attributes are set to WHAT, and IN WHAT ORDER. A hand
//    attempt on bridge 2 extracted each function's decision rule into a documented predicate,
//    omitted the part that changed the IR, and reported it done — nothing called any of it.
//    Droppable: only the mechanism for REACHING operands (walking uses, memoising, positioning a
//    builder). ⛔ If the island cannot express a result, EXTEND THE ISLAND. Deciding a function is
//    unnecessary is NOT the porter's call, and a predicate is not a port.
//
// 4. HARD CAPS — MEASURED, AND THEY BIND. Bridge 2's first 144 functions cost 43 h wall clock and
//    586 M cache-read tokens; of 30,991 lines produced only 5,306 were implementation (12,333 doc
//    comments, 12,575 tests). Per ported function:
//      • 8 DOC LINES: the `/// Replaces: eNNN_name` anchor, one line of what it does, any TRAP.
//        No tutorials, no restating the C++, no design essays.
//      • ONE TEST. Two only where the vendor's own case AND a negative both apply.
//      • `cargo check -p deeptools` + `cargo test -p deeptools` ONCE PER BATCH, not per function.
//      • Do NOT re-verify citations — the review pass owns that.
//      • Do NOT grep the crate to discover types; the anchor names what you need.
//    NOT capped: correctness, and the emission.
//
// 5. THIS IS A PURE-LOGIC PORT WITH NO C ANYWHERE. Whatever generic C-to-Rust conventions say:
//    ❌ no bindgen/allowlist/-sys, ❌ no `ffi::`/`extern "C"`, ❌ no `CRUSTIFY_<FILE>` switch,
//    ❌ no `Foo`/`FooRef`/`FooMut` triple, ❌ no `unsafe`, ❌ no C-vs-Rust equivalence harness.
//
// 6. CRATE RULES BIND YOU — `crates/compiler/deeptools/CLAUDE.md`, read it in full.
//    🛑 NEVER RUNTIME REFUSE: no `Result`, no `Err(`, no `.ok_or`, no `assert!`, no `debug_assert!`.
//    A closed set is an `enum`; an invariant is a TYPE; newtypes, never raw scalars. No strings for
//    closed sets. `Arch`/`Model`/`Workload` flow through. `todo!` NAMING UNPORTED WORK IS ALLOWED
//    here — and ⛔ never substitute a stand-in op to dodge one.
//
// 7. ANCHORS: each `// crustify:todo: eNNN_name` below is one scheduled unit. Replace it with the
//    ported item carrying the doc anchor `/// Replaces: eNNN_name`. ⛔ NEVER DELETE AN ANCHOR YOU
//    DID NOT PORT — on bridge 2 a deleted anchor was indistinguishable from a finished unit and the
//    campaign reported DONE having silently lost 149 of 384 functions, the biggest in its span.
//
// 8. GATE: `cargo check -p deeptools` and `cargo test -p deeptools`. ⛔ NEVER run the workspace or
//    the acceptance build in an agent worktree — ~6 GB of target/ each, and this host is tight.
//
// 9. 43 OF THE 52 PASSES CONSUME AN ANALYSIS THAT IS OUT OF SCOPE (122 of the 656 units name one):
//    `Analyses/` (Liveness, PropagationAnalysis, GraphColoring, ExpressionEvaluatorUtils,
//    InstructionEstimation, TimeStamps, RegisterPressureAnalysis, AddressPinningScheme,
//    XRFRegisterAnalyzer, CorrelationAnalysis, RedundantDefinitionEliminationTree, …) and
//    `RegisterInitialization/` (Collector, Evaluator, Selector, Transformer, UniformGrouper) —
//    ~11,700 lines NOT in this campaign. Per pass: `crustify-senpass/OUTSIDE-DEPS.tsv`; per unit:
//    `OUTSIDE-UNITS.tsv`. ⭐ When a unit needs one, port the part that is present and
//    `todo!("<Analysis>::<method> — out of campaign scope")` for the part that is not, leaving the
//    anchor FILLED so the unit is not lost. ⛔ DO NOT INVENT THE ANALYSIS and do not substitute a
//    constant for its result. Extending `src/islands/sentient/` is a different case and IS expected.

//! `Utils.cpp`/`Utils.hpp` — 6 of the campaign's 656 units (dependency level(s) [0, 1]).
//!
//! | unit | entry | level | lines | authority path:line |
//! |---|---|---|---|---|
//! | `e249_normalizeNullValues` | 249 | 0 | 14 | `dcc/src/Transform/Sentient/Utils.cpp:600` |
//! | `e394_dump` | 394 | 1 | 6 | `dcc/src/Transform/Sentient/Utils.cpp:615` |
//!
//! Promoted here from `utils/mod.rs`, whose scheduler TODOs they were — they are methods of the type
//! this file already owns, and a second Rust type with the same fields would be the split made real:
//!
//! | unit | entry | level | lines | authority path:line |
//! |---|---|---|---|---|
//! | `e251_add` | 251 | 0 | 4 | `dcc/src/Transform/Sentient/Utils.hpp:165` |
//! | `e252_size` | 252 | 0 | 4 | `dcc/src/Transform/Sentient/Utils.hpp:171` |
//! | `e396_replaceValue` | 396 | 1 | 4 | `dcc/src/Transform/Sentient/Utils.hpp:175` |
//! | `e253_areAllValuesEqual` | 253 | 0 | 6 | `dcc/src/Transform/Sentient/Utils.hpp:180` |


// ⛔ NOTHING CALLS THIS TYPE YET — its one consumer is `RegisterInitCandidatePromoter` (e606, e517,
// e570), all still open. `replaceValue` (e396) is now ported here beside the type it belongs to.
// ⭐ REMOVE THIS WITH THE FIRST OF THEM: an unused item here is a real defect from then on.
#![allow(dead_code)]

use std::fmt::Write as _;

use crate::islands::sentient::dialects::{self, Op, Val};
use crate::islands::sentient::print;

/// EVERY UNIT OF A UNIFORMIZED PROGRAM PAIRED WITH ITS VALUE — `dcc::utils::UnitsAndTheirValues`
/// (`dcc/src/Transform/Sentient/Utils.hpp:161-190`).
///
/// ⛔⛔ ONE LIST OF PAIRS, NOT THE REFERENCE'S TWO PARALLEL `ListTy`s. `size()`'s
/// `DT_CHECK(values_.size() == units_.size())` (`Utils.hpp:172`) is the ONE invariant the class exists
/// to hold, and it is the only thing `add` — the sole way to grow either list — enforces. A pair
/// cannot be half-pushed, so the check has nothing left to guard and the `units()[index]` /
/// `values()[index]` reads its callers do (`OldRegisterInitialization.cpp:996-1009`) stay one index.
///
/// ⛔ `Option<Val>` IS THE `mlir::Value` NULL A VALUE MAY BE. A unit is never null — `add` is always
/// called with a real one — but a value is deliberately left absent until
/// [`Self::normalize_null_values`] fills it, which is why that method exists at all.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct UnitsAndTheirValues {
    /// `units_` zipped with `values_`, in `add` order.
    pub(crate) pairs: Vec<(Val, Option<Val>)>,
}

impl UnitsAndTheirValues {
    /// Replaces: e249_normalizeNullValues
    ///
    /// Fills every null value with the FIRST non-null one, so the list ends up wholly filled or
    /// wholly null.
    ///
    /// ⛔ AN ALL-NULL (OR EMPTY) LIST IS A FIXED POINT, NOT A REFUSAL: `non_null_val` stays null and
    /// the reference's second loop writes null over null. The early return is that, without the write.
    pub(crate) fn normalize_null_values(&mut self) {
        let Some(non_null) = self.pairs.iter().find_map(|(_unit, value)| *value) else {
            return;
        };
        for (_unit, value) in &mut self.pairs {
            if value.is_none() {
                *value = Some(non_null);
            }
        }
    }

    /// Replaces: e251_add
    ///
    /// `add(unit, value)` (`Utils.hpp:165`) — one more mapping, with the two lists the same length by
    /// construction rather than by a check.
    pub(crate) fn add(&mut self, unit: Val, value: Option<Val>) {
        self.pairs.push((unit, value));
    }

    /// Replaces: e252_size
    ///
    /// `size()` (`Utils.hpp:171`) — how many units are mapped.
    ///
    /// ⭐ THE `DT_CHECK` IS GONE, NOT SKIPPED: [`Self::pairs`] makes the two lengths one number, so
    /// this cannot refuse.
    #[must_use]
    pub(crate) fn size(&self) -> usize {
        self.pairs.len()
    }

    /// Replaces: e396_replaceValue
    ///
    /// `replaceValue(index, new_val)` (`Utils.hpp:175`) — one mapped value overwritten, its unit left
    /// alone.
    ///
    /// ⛔ `DT_CHECK(index < size())` (`Utils.hpp:176`) STAYS A NAMED STOP, not a `Result`: the one
    /// consumer indexes with a position it took from the same list (`OldRegisterInitialization.cpp:996`),
    /// so an out-of-range index is a broken caller and not an input.
    /// ⭐ THE VALUE ARRIVES NON-NULL — the parameter is a plain `mlir::Value`, so this can only ever
    /// fill a hole [`Self::normalize_null_values`] would have.
    pub(crate) fn replace_value(&mut self, index: usize, new_val: Val) {
        let Some((_unit, value)) = self.pairs.get_mut(index) else {
            panic!(
                "DT_CHECK(index < size()) (`Utils.hpp:176`): {index} of {}",
                self.pairs.len()
            )
        };
        *value = Some(new_val);
    }

    /// Replaces: e253_areAllValuesEqual
    ///
    /// `areAllValuesEqual()` (`Utils.hpp:180`) — whether every mapped value is the same one.
    ///
    /// ⛔ `None` IS NOT `false`. It is `DT_CHECK(!values_.empty())` (`:181`), a crash on an empty
    /// mapping, and a caller that reads it as "not all equal" inverts the branch it guards
    /// (`OldRegisterInitialization.cpp:925`).
    #[must_use]
    pub(crate) fn are_all_values_equal(&self) -> Option<bool> {
        let (_, last) = *self.pairs.last()?;
        Some(self.pairs.iter().all(|(_, value)| *value == last))
    }

    /// Replaces: e394_dump
    ///
    /// `size: N`, then one `<unit>\t --> <value>` line per mapping, in `add` order.
    ///
    /// ⭐ A RETURNED `String` FOR `llvm::dbgs()`, as e014_dump, e065_dump and e102_dumpWeights already
    /// do: the reference's own caller is the only thing that decides where the text goes.
    #[must_use]
    pub(crate) fn dump(&self, scope: &[Op]) -> String {
        let mut out = format!("size: {}\n", self.size());
        for (unit, value) in &self.pairs {
            print_value(&mut out, Some(*unit), scope);
            out.push_str("\t --> ");
            print_value(&mut out, *value, scope);
            out.push('\n');
        }
        out
    }
}

/// `llvm::dbgs() << v` FOR AN `mlir::Value` — `Value::print`, which prints an op result's whole
/// defining operation and MLIR's own `<<NULL VALUE>>` where the value is null.
///
/// ⚠ A VALUE WHOSE DEFINING OP IS NOT IN `scope` PRINTS AS A BLOCK ARGUMENT WITHOUT ITS TYPE, and
/// [`print::emit`] ends its line — both are e102_dumpWeights' recorded divergences, unchanged here.
fn print_value(out: &mut String, val: Option<Val>, scope: &[Op]) {
    match val {
        None => out.push_str("<<NULL VALUE>>"),
        Some(val) => match dialects::defining_op(val, scope) {
            Some(op) => print::emit(out, op, 0),
            None => {
                let _ = write!(out, "<block argument> {val:?}");
            }
        },
    }
}


#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::islands::dataflow_ir::ty::ScalarTy;
    use crate::islands::sentient::dialects::sentient;

    /// e249 — the holes take the first non-null value, and an all-null list is left alone.
    #[test]
    fn normalize_null_values_fills_the_holes_and_leaves_an_all_null_list() {
        let mut uvs = UnitsAndTheirValues {
            pairs: vec![
                (Val(10), None),
                (Val(11), Some(Val(20))),
                (Val(12), None),
                (Val(13), Some(Val(21))),
            ],
        };
        uvs.normalize_null_values();
        assert_eq!(
            uvs.pairs,
            vec![
                (Val(10), Some(Val(20))),
                (Val(11), Some(Val(20))),
                (Val(12), Some(Val(20))),
                (Val(13), Some(Val(21))),
            ]
        );

        let mut all_null = UnitsAndTheirValues {
            pairs: vec![(Val(10), None), (Val(11), None)],
        };
        all_null.normalize_null_values();
        assert_eq!(all_null.pairs, vec![(Val(10), None), (Val(11), None)]);
    }

    /// `add` KEEPS THE PAIRS IN PUSH ORDER and `size` counts them — including a null value.
    #[test]
    fn e251_add_and_e252_size() {
        let mut uvs = UnitsAndTheirValues::default();
        uvs.add(Val(1), None);
        uvs.add(Val(2), Some(Val(3)));

        assert_eq!(uvs.size(), 2);
        assert_eq!(uvs.pairs, vec![(Val(1), None), (Val(2), Some(Val(3)))]);
    }

    /// ALL EQUAL, NOT ALL EQUAL, AND THE EMPTY MAPPING'S `None` — which is the reference's `DT_CHECK`
    /// and NOT an answer of `false`.
    #[test]
    fn e253_are_all_values_equal() {
        let mut uvs = UnitsAndTheirValues::default();
        assert_eq!(uvs.are_all_values_equal(), None);

        uvs.add(Val(1), Some(Val(9)));
        uvs.add(Val(2), Some(Val(9)));
        assert_eq!(uvs.are_all_values_equal(), Some(true));

        uvs.add(Val(3), None);
        assert_eq!(uvs.are_all_values_equal(), Some(false));
    }

    /// e396 — the value at one index is overwritten and its unit, and every other pair, is untouched.
    #[test]
    fn e396_replace_value() {
        let mut uvs = UnitsAndTheirValues::default();
        uvs.add(Val(1), Some(Val(9)));
        uvs.add(Val(2), None);

        uvs.replace_value(1, Val(7));

        assert_eq!(
            uvs.pairs,
            vec![(Val(1), Some(Val(9))), (Val(2), Some(Val(7)))]
        );
    }

    /// e396 — an index past the end is the reference's `DT_CHECK`, not an answer.
    #[test]
    #[should_panic(expected = "DT_CHECK(index < size())")]
    fn e396_replace_value_past_the_end_is_the_reference_check() {
        let mut uvs = UnitsAndTheirValues::default();
        uvs.add(Val(1), None);
        uvs.replace_value(1, Val(7));
    }

    /// e394 — the size line, then one line per pair: a defined value prints its whole defining op, a
    /// null value prints `<<NULL VALUE>>`, and a value with no defining op prints as a block argument.
    #[test]
    fn e394_dump() {
        let scope = vec![Op::Sentient(sentient::Op::ScalarConstant {
            value: 5,
            result: Val(20),
            reg_locale: sentient::RegType::Imm,
            ty: ScalarTy::Index,
            is_symbol: false,
        })];
        let uvs = UnitsAndTheirValues {
            pairs: vec![
                (Val(10), Some(Val(20))),
                (Val(11), None),
                (Val(12), Some(Val(30))),
            ],
        };

        let dumped = uvs.dump(&scope);

        assert!(dumped.starts_with("size: 3\n"), "{dumped}");
        assert_eq!(dumped.matches("\t --> ").count(), 3, "{dumped}");
        assert_eq!(dumped.matches("<<NULL VALUE>>").count(), 1, "{dumped}");
        // The three units and `Val(30)` have no defining op in `scope`; `Val(20)` does.
        assert_eq!(dumped.matches("<block argument>").count(), 4, "{dumped}");
        assert_eq!(dumped.matches("scalar_constant").count(), 1, "{dumped}");
    }
}
