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

// ⛔ THE PASS IS NOT WIRED INTO THE PIPELINE YET, so everything below is reachable only from this
// file's own tests until `e575_runOnOperation` lands and something calls it. CI runs clippy with
// `-D warnings`, so without this the first ported leaf of the module fails the gate.
// ⭐ REMOVE THIS WITH `e575_runOnOperation`: at that point an unused item here is a real defect again.
#![allow(dead_code)]

use crate::islands::sentient::dialects::{Definitions, Op, sentient};

/// `typedef int64_t WidestIntType` (`:52`) — the width the pass compares yielded constants at.
///
/// ⭐ AN ALIAS AND NOT A NEWTYPE: [`sentient::Op::ScalarConstant::value`] is already the `i64` this
/// names, and wrapping it here would only be unwrapped again at the one comparison it exists for.
pub type WidestInt = i64;

/// A `sentient.if` — `sentient::IfOp if_op` as a witness over the union type, so that the two regions
/// this reads are reachable without a second `dyn_cast`.
#[derive(Debug, Clone, Copy)]
pub struct IfOp<'a> {
    /// `if_op.getThenRegion().front()`.
    then_body: &'a [Op],
    /// `if_op.getElseRegion().front()`.
    else_body: &'a [Op],
}

impl<'a> IfOp<'a> {
    /// The witness, or `None` for any other op.
    #[must_use]
    pub fn of(op: &'a Op) -> Option<IfOp<'a>> {
        match op {
            Op::Sentient(sentient::Op::If {
                then_body,
                else_body,
                ..
            }) => Some(IfOp {
                then_body,
                else_body,
            }),
            _ => None,
        }
    }
}

/// Replaces: e354_getReturnValsIfSimpleConditional
///
/// The `(then-val, else-val)` constants yielded at `idx` when BOTH regions hold nothing but their
/// yield, and `None` otherwise (`:121-148`).
///
/// ⛔ `None` COVERS ALL THREE REFUSALS: a region with more than the yield in it, and either side
/// yielding something that is not a `sentient.scalar_constant`.
/// ⛔ TRAP: THE CONSTANTS ARE DEFINED OUTSIDE THE REGIONS, which is why `defs` is a parameter — the
/// yield's operand is looked up in the enclosing scopes, innermost first.
/// ⛔ `dyn_cast_or_null` (`:130`, `:139`) IS ALREADY TOTAL HERE: a yielded region argument has no
/// defining op, and [`Definitions::of`] answers `None` where the reference relies on the null-tolerant
/// spelling to avoid an assert.
#[must_use]
pub fn get_return_vals_if_simple_conditional(
    if_op: IfOp<'_>,
    idx: usize,
    defs: Definitions<'_>,
) -> Option<(WidestInt, WidestInt)> {
    let then_val = yielded_constant(if_op.then_body, idx, defs)?;
    let else_val = yielded_constant(if_op.else_body, idx, defs)?;
    Some((then_val, else_val))
}

/// One region's `front().getOperations().size() != 1` check and its terminator's `idx`th operand.
///
/// ⭐ THE TERMINATOR IS THE ONE OP, so the size check and the `getTerminator()` are the same match.
fn yielded_constant(body: &[Op], idx: usize, defs: Definitions<'_>) -> Option<WidestInt> {
    let [Op::Sentient(sentient::Op::Yield { results })] = body else {
        return None;
    };
    match defs.of(*results.get(idx)?) {
        Some(Op::Sentient(sentient::Op::ScalarConstant { value, .. })) => Some(*value),
        _ => None,
    }
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
    use super::{IfOp, get_return_vals_if_simple_conditional};
    use crate::islands::dataflow_ir::ty::ScalarTy;
    use crate::islands::sentient::dialects::{Definitions, Op, Val, sentient};

    /// `%r = sentient.scalar_constant {value = <value>} : index`.
    fn constant(result: Val, value: i64) -> Op {
        Op::Sentient(sentient::Op::ScalarConstant {
            value,
            result,
            reg_locale: sentient::RegType::Imm,
            ty: ScalarTy::Index,
            is_symbol: false,
        })
    }

    /// `sentient.yield %results`.
    fn yield_op(results: Vec<Val>) -> Op {
        Op::Sentient(sentient::Op::Yield { results })
    }

    /// `%r = sentient.if eq(%lhs, %rhs) { <then> } else { <else> }`.
    fn if_op(result: Val, then_body: Vec<Op>, else_body: Vec<Op>) -> Op {
        Op::Sentient(sentient::Op::If {
            predicate: sentient::CmpPredicate::Eq,
            lhs: Val(0),
            rhs: Val(1),
            yielded: vec![sentient::Yielded {
                result,
                reg: sentient::Reg {
                    locale: sentient::RegType::Unknown,
                    index: None,
                },
            }],
            dbg_name: None,
            then_body,
            else_body,
        })
    }

    /// `e354` — the simple conditional's pair, and each of the three refusals.
    #[test]
    fn a_pair_of_constants_needs_both_regions_to_hold_only_the_yield() {
        let (c3, c7, other) = (Val(2), Val(3), Val(4));
        let simple = if_op(Val(5), vec![yield_op(vec![c3])], vec![yield_op(vec![c7])]);
        let body = vec![constant(c3, 3), constant(c7, 7), simple];
        let scope: [&[Op]; 1] = [body.as_slice()];
        let defs = Definitions::from_innermost(&scope);
        let witness = IfOp::of(&body[2]).expect("a sentient.if");
        assert_eq!(
            get_return_vals_if_simple_conditional(witness, 0, defs),
            Some((3, 7))
        );
        // ⛔ NO SUCH INDEX.
        assert_eq!(
            get_return_vals_if_simple_conditional(witness, 1, defs),
            None
        );
        // ⛔ A REGION WITH MORE THAN ITS YIELD IN IT.
        let busy = if_op(
            Val(6),
            vec![constant(other, 9), yield_op(vec![other])],
            vec![yield_op(vec![c7])],
        );
        assert_eq!(
            get_return_vals_if_simple_conditional(IfOp::of(&busy).unwrap(), 0, defs),
            None
        );
        // ⛔ A YIELDED VALUE THAT IS NOT A CONSTANT — here nothing in scope defines it.
        let unknown = if_op(
            Val(7),
            vec![yield_op(vec![Val(99)])],
            vec![yield_op(vec![c7])],
        );
        assert_eq!(
            get_return_vals_if_simple_conditional(IfOp::of(&unknown).unwrap(), 0, defs),
            None
        );
        // ⛔ AND ONLY A `sentient.if` IS A WITNESS.
        assert!(IfOp::of(&body[0]).is_none());
    }
}
