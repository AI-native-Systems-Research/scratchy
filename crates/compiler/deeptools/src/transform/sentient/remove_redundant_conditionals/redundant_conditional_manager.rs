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

//! `RemoveRedundantConditionals.cpp` — 4 of the campaign's 656 units (dependency level(s) [1, 2, 3]).
//!
//! | unit | entry | level | lines | authority path:line |
//! |---|---|---|---|---|
//! | `e354_getReturnValsIfSimpleConditional` | 354 | 1 | 27 | `dcc/src/Transform/Sentient/RemoveRedundantConditionals.cpp:121` |
//! | `e465_updateIfOpBasedOnParentIfOp` | 465 | 2 | 65 | `dcc/src/Transform/Sentient/RemoveRedundantConditionals.cpp:151` |
//! | `e466_updateIfOpFeedingDynLoopBound` | 466 | 2 | 77 | `dcc/src/Transform/Sentient/RemoveRedundantConditionals.cpp:217` |
//! | `e526_processIfOp` | 526 | 3 | 22 | `dcc/src/Transform/Sentient/RemoveRedundantConditionals.cpp:98` |

// The other three units of this file are unported, so this file's first leaf has no caller yet and the
// crate is built with `-D warnings`.
// ⭐ REMOVE THIS WITH `e526_processIfOp`: at that point an unused item here is a real defect again.
#![allow(dead_code)]

use crate::islands::sentient::dialects::{Definitions, Op, sentient};

/// Replaces: e354_getReturnValsIfSimpleConditional
///
/// The `(then, else)` constants an `sentient.if` hands back at result `idx`, when BOTH its branches
/// hold nothing but their `sentient.yield` and both yield a `sentient.scalar_constant` there.
///
/// ⛔ TRAP: `getOperations().size() != 1` COUNTS THE TERMINATOR (`:125-127`), so "one op" means the
/// yield ALONE — a branch that computes anything at all is not simple.
/// ⛔ TRAP: AN `sentient.if` WITH NO `else` REGION ANSWERS `None` HERE, not "then only". An empty
/// `else_body` has no terminator to read, which is the reference's `front()` on an empty region.
#[must_use]
pub fn return_vals_if_simple_conditional(
    if_op: &Op,
    idx: usize,
    defs: Definitions<'_>,
) -> Option<(i64, i64)> {
    let Op::Sentient(sentient::Op::If {
        then_body,
        else_body,
        ..
    }) = if_op
    else {
        return None;
    };
    let yielded_const = |body: &[Op]| match body {
        [Op::Sentient(sentient::Op::Yield { results })] => {
            match defs.of(*results.get(idx)?)? {
                Op::Sentient(sentient::Op::ScalarConstant { value, .. }) => Some(*value),
                // `dyn_cast_or_null<ConstantOp>` failing — including the null the reference tolerates
                // here for a yielded iter arg (`:130`, `:139`).
                _ => None,
            }
        }
        _ => None,
    };
    Some((yielded_const(then_body)?, yielded_const(else_body)?))
}

// crustify:todo: e465_updateIfOpBasedOnParentIfOp
//   authority : dcc/src/Transform/Sentient/RemoveRedundantConditionals.cpp:151  (65 body lines, level 2)
//   original  : void RedundantConditionalManager::updateIfOpBasedOnParentIfOp()
//   calls     : e244_negatePredicate, e354_getReturnValsIfSimpleConditional

// crustify:todo: e466_updateIfOpFeedingDynLoopBound
//   authority : dcc/src/Transform/Sentient/RemoveRedundantConditionals.cpp:217  (77 body lines, level 2)
//   original  : void RedundantConditionalManager::updateIfOpFeedingDynLoopBound()
//   calls     : e244_negatePredicate, e354_getReturnValsIfSimpleConditional

// crustify:todo: e526_processIfOp
//   authority : dcc/src/Transform/Sentient/RemoveRedundantConditionals.cpp:98  (22 body lines, level 3)
//   original  : void RedundantConditionalManager::processIfOp()
//   calls     : e465_updateIfOpBasedOnParentIfOp, e466_updateIfOpFeedingDynLoopBound

#[cfg(test)]
mod unit_tests {
    use super::return_vals_if_simple_conditional;
    use crate::islands::dataflow_ir::ty::ScalarTy;
    use crate::islands::sentient::dialects::sentient::{CmpPredicate, RegType, Yielded};
    use crate::islands::sentient::dialects::{Definitions, Op, Val, sentient};

    /// `%r = sentient.scalar_constant {value = <value>}`.
    fn constant(result: Val, value: i64) -> Op {
        Op::Sentient(sentient::Op::ScalarConstant {
            value,
            result,
            reg_locale: RegType::Imm,
            ty: ScalarTy::Index,
            is_symbol: false,
        })
    }

    /// `sentient.if eq, %0, %1 -> (index)` with the two branch bodies given.
    fn if_op(then_body: Vec<Op>, else_body: Vec<Op>) -> Op {
        Op::Sentient(sentient::Op::If {
            predicate: CmpPredicate::Eq,
            lhs: Val(0),
            rhs: Val(1),
            yielded: vec![Yielded {
                result: Val(20),
                reg: sentient::Reg {
                    locale: RegType::Unknown,
                    index: None,
                },
            }],
            dbg_name: None,
            then_body,
            else_body,
        })
    }

    /// `sentient.yield %results`.
    fn yield_op(results: Vec<Val>) -> Op {
        Op::Sentient(sentient::Op::Yield { results })
    }

    /// Both branches yielding nothing but a constant answer the pair; an extra op in a branch, a
    /// non-constant yielded value and a missing `else` region each answer `None`.
    #[test]
    fn a_simple_conditional_is_a_yield_of_a_constant_in_both_branches() {
        let (c3, c4) = (Val(10), Val(11));
        let outer = vec![
            constant(c3, 3),
            constant(c4, 4),
            if_op(vec![yield_op(vec![c3])], vec![yield_op(vec![c4])]),
        ];
        let scopes: [&[Op]; 1] = [&outer];
        let defs = Definitions::from_innermost(&scopes);
        assert_eq!(
            return_vals_if_simple_conditional(&outer[2], 0, defs),
            Some((3, 4))
        );
        // ⛔ THE TERMINATOR IS COUNTED: a branch that computes anything is not simple.
        let busy = if_op(
            vec![constant(Val(12), 5), yield_op(vec![c3])],
            vec![yield_op(vec![c4])],
        );
        assert_eq!(return_vals_if_simple_conditional(&busy, 0, defs), None);
        // A value no `sentient.scalar_constant` binds, and an index no branch yields.
        let opaque = if_op(vec![yield_op(vec![Val(99)])], vec![yield_op(vec![c4])]);
        assert_eq!(return_vals_if_simple_conditional(&opaque, 0, defs), None);
        assert_eq!(return_vals_if_simple_conditional(&outer[2], 1, defs), None);
        // No `else` region at all.
        let then_only = if_op(vec![yield_op(vec![c3])], Vec::new());
        assert_eq!(return_vals_if_simple_conditional(&then_only, 0, defs), None);
        // Not a conditional.
        assert_eq!(return_vals_if_simple_conditional(&outer[0], 0, defs), None);
    }
}
