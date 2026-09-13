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

//! `AddressPinningAndToggle.cpp` — 2 of the campaign's 656 units (dependency level(s) [0, 2]).
//!
//! | unit | entry | level | lines | authority path:line |
//! |---|---|---|---|---|
//! | `e016_ConditionalConstantDescriptor` | 016 | 0 | 32 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:2744` |
//! | `e425_dump` | 425 | 2 | 15 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:2780` |

use super::write_evaluated_value;
use crate::islands::sentient::dialects::{Definitions, Op, Val, sentient};
use crate::transform::sentient::analyses::{EvaluatedValue, ExpressionEvaluator};
use crate::transform::sentient::utils::{ConstKind, is_constant};

/// A BASE ADDRESS CHOSEN BY NESTED `sentient.if`s FROM TWO OR MORE CONSTANTS —
/// `class ConditionalConstantDescriptor` (`AddressPinningAndToggle.cpp:303-340`).
///
/// ⛔ EMPTY IS THE INVALID STATE: `isValid()` is `!yielded_constants_.empty()` (`:318`), so this
/// pattern has no `invalidate()` of its own — the ctor either fills the list or leaves it empty.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ConditionalConstantDescriptor {
    /// `yielded_constants_` — the possible constants yielded from the conditional, in the order the
    /// match walked them.
    pub yielded_constants: Vec<EvaluatedValue>,
    /// `can_be_simplified_`, the base class's own field (`AddressPinningAndToggle.cpp:159`) — true
    /// when every path yields the SAME constant, so the conditional is not really a choice.
    pub can_be_simplified: bool,
}

impl ConditionalConstantDescriptor {
    /// Replaces: e016_ConditionalConstantDescriptor
    ///
    /// Matches `base_addr` as one `sentient.if` result whose every reachable yield is a constant,
    /// collecting those constants in region order (`AddressPinningAndToggle.cpp:2744-2778`).
    ///
    /// ⛔ ONE NON-CONSTANT YIELD CLEARS THE WHOLE LIST (`:2771`) — all-or-nothing, and an empty list
    /// IS the invalid state (`isValid()` is `!yielded_constants_.empty()`, `:318`).
    /// ⛔ A `base_addr` NO `sentient.if` BINDS STAYS DEFAULT: the reference's failing `dyn_cast`
    /// (`:2755`) leaves both base-class fields untouched.
    #[must_use]
    pub fn new(
        base_addr: Val,
        defs: Definitions<'_>,
        evaluator: &mut impl ExpressionEvaluator,
    ) -> ConditionalConstantDescriptor {
        // `int result_index = getIndexOfOperationResults(base_addr)` then
        // `dyn_cast<sentient::IfOp>(base_addr.getDefiningOp())` (`:2753-2755`). ⭐ THE INDEX IS ONLY
        // EVER USED TO INDEX THAT `if`'s YIELDS, so finding `base_addr` among them is both steps.
        let Some(Op::Sentient(if_op @ sentient::Op::If { yielded, .. })) = defs.of(base_addr)
        else {
            return ConditionalConstantDescriptor::default();
        };
        let Some(result_index) = yielded.iter().position(|entry| entry.result == base_addr) else {
            return ConditionalConstantDescriptor::default();
        };

        let mut operands = Vec::new();
        yielded_operands(if_op, result_index, defs, &mut operands);

        // `bool all_yield_operands_constant = true` and the lambda's two arms (`:2757-2768`).
        let mut yielded_constants = Vec::new();
        let mut all_yield_operands_constant = true;
        for operand in operands {
            if is_constant(operand, ConstKind::ScalarConstant, defs) {
                yielded_constants.push(evaluator.evaluate_value_handle(operand));
            } else {
                all_yield_operands_constant = false;
            }
        }
        if !all_yield_operands_constant {
            yielded_constants.clear();
        }

        // `if (yielded_constants_.empty()) return;` — `setCanBeSimplified` is never reached, so it
        // keeps the base class's `false` rather than the vacuous truth of `all_of` on nothing.
        let can_be_simplified = match yielded_constants.first() {
            None => false,
            Some(first) => yielded_constants.iter().all(|ev| ev == first),
        };
        ConditionalConstantDescriptor {
            yielded_constants,
            can_be_simplified,
        }
    }
}

impl ConditionalConstantDescriptor {
    /// Replaces: e425_dump
    ///
    /// This conditional's constants as debug text (`:2780-2794`) — one `\t`-indented constant per
    /// line, `;\n` BETWEEN them, wrapped in a counted `[` … `]`.
    ///
    /// ⛔ TRAP: THE VALID BRANCH ENDS IN [`write_evaluated_value`], and `isValid()` here IS
    /// `!yielded_constants.is_empty()` (`:318`) — so the first `interleave` element always reaches the
    /// out-of-scope `operator<<` and only the invalid branch is complete.
    /// ⛔ `llvm::interleave`'s SEPARATOR GOES BETWEEN ONLY: the last constant is followed by no `;\n`,
    /// so `"\t]\n"` lands directly against it.
    #[must_use]
    pub fn dump(&self) -> String {
        let mut out = String::from("N-Way Conditional Constant Descriptor:\n");
        // `const char indent = '\t'` (`:2783`), streamed before each line the reference indents.
        if !self.is_valid() {
            out.push_str("\tInvalid\n");
            return out;
        }
        if self.can_be_simplified {
            out.push_str("\t(Simplified)\n");
        }
        out.push_str("\tYielded Constants (");
        out.push_str(&self.yielded_constants.len().to_string());
        out.push_str(") [\n");
        for (index, ev) in self.yielded_constants.iter().enumerate() {
            if index > 0 {
                out.push_str(";\n");
            }
            out.push('\t');
            write_evaluated_value(Some(*ev), &mut out);
        }
        out.push_str("\t]\n");
        out
    }
}

/// `dcc::utils::applyToAllYields<sentient::IfOp>` (`dcc/src/Utils/Utils.cpp:164-179`) reduced to the
/// operands its callback reads: per region of `if_op`, the `sentient.yield` operand at
/// `result_index`, descending through a nested `sentient.if` bound at that position.
///
/// ⭐ AN EMPTY `else_body` IS NO ELSE REGION, which is the reference's `getNumRegions()` answering 1;
/// an empty region has no terminator to read and the reference would deref a null one.
fn yielded_operands(
    if_op: &sentient::Op,
    result_index: usize,
    defs: Definitions<'_>,
    out: &mut Vec<Val>,
) {
    let sentient::Op::If {
        then_body,
        else_body,
        ..
    } = if_op
    else {
        return;
    };
    for region in [then_body.as_slice(), else_body.as_slice()] {
        if region.is_empty() {
            continue;
        }
        let Some(Op::Sentient(sentient::Op::Yield { results })) = region.last() else {
            continue;
        };
        let Some(&operand) = results.get(result_index) else {
            continue;
        };
        // `!isa<BlockArgument>(yield_operand) && isa<IfOpTy>(yield_operand.getDefiningOp())` — the
        // nested case recurses at `getIndexOfOperationResults(yield_operand)`.
        match defs.of(operand) {
            Some(Op::Sentient(nested @ sentient::Op::If { yielded, .. })) => {
                match yielded.iter().position(|entry| entry.result == operand) {
                    Some(index) => yielded_operands(nested, index, defs, out),
                    None => out.push(operand),
                }
            }
            _ => out.push(operand),
        }
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::islands::dataflow_ir::ty::ScalarTy;
    use crate::islands::sentient::dialects::sentient::{CmpPredicate, Reg, RegType, Yielded};
    use crate::transform::sentient::analyses::OutOfScopeEvaluator;

    /// `sentient.scalar_constant %value -> result`.
    fn constant(value: i64, result: Val) -> Op {
        Op::Sentient(sentient::Op::ScalarConstant {
            value,
            result,
            reg_locale: RegType::Imm,
            ty: ScalarTy::Index,
            is_symbol: false,
        })
    }

    /// A `sentient.if` whose `then` yields `Val(1)` and whose `else` yields a nested `sentient.if`,
    /// so a faithful walk reaches THREE constants and a shallow one would reach two.
    fn nested_conditional() -> Vec<Op> {
        let inner = Op::Sentient(sentient::Op::If {
            predicate: CmpPredicate::Eq,
            lhs: Val(20),
            rhs: Val(21),
            yielded: vec![Yielded {
                result: Val(11),
                reg: Reg {
                    locale: RegType::Lbr,
                    index: None,
                },
                element_size: None,
            }],
            dbg_name: None,
            then_body: vec![Op::Sentient(sentient::Op::Yield {
                results: vec![Val(2)],
            })],
            else_body: vec![Op::Sentient(sentient::Op::Yield {
                results: vec![Val(3)],
            })],
        });
        vec![
            constant(4096, Val(1)),
            constant(8192, Val(2)),
            constant(8192, Val(3)),
            Op::Sentient(sentient::Op::If {
                predicate: CmpPredicate::Eq,
                lhs: Val(20),
                rhs: Val(21),
                yielded: vec![Yielded {
                    result: Val(10),
                    reg: Reg {
                        locale: RegType::Lbr,
                        index: None,
                    },
                    element_size: None,
                }],
                dbg_name: None,
                then_body: vec![Op::Sentient(sentient::Op::Yield {
                    results: vec![Val(1)],
                })],
                else_body: vec![
                    inner,
                    Op::Sentient(sentient::Op::Yield {
                        results: vec![Val(11)],
                    }),
                ],
            }),
        ]
    }

    /// The walk IS the ported half, so the seam it stops at proves it reached a constant to evaluate.
    #[test]
    #[should_panic(expected = "evaluateValue")]
    fn e016_descends_into_the_nested_conditional_before_evaluating() {
        let body = nested_conditional();
        let regions: [&[Op]; 1] = [&body];
        let defs = Definitions::from_innermost(&regions);
        let _desc =
            ConditionalConstantDescriptor::new(Val(10), defs, &mut OutOfScopeEvaluator::default());
    }

    /// The negative the reference reaches by `dyn_cast` failing: a base address no `sentient.if`
    /// binds never asks the evaluator anything, and an empty list is the invalid state.
    #[test]
    fn e016_a_base_addr_that_is_not_an_if_result_stays_empty() {
        let body = nested_conditional();
        let regions: [&[Op]; 1] = [&body];
        let defs = Definitions::from_innermost(&regions);
        let desc =
            ConditionalConstantDescriptor::new(Val(1), defs, &mut OutOfScopeEvaluator::default());
        assert_eq!(desc, ConditionalConstantDescriptor::default());
    }

    /// 425/656 — the complete branch: an unmatched conditional is the empty list, which `isValid()`
    /// reads, and its whole trace is the header and the word.
    #[test]
    fn e425_an_invalid_descriptor_dumps_the_header_and_invalid() {
        let desc = ConditionalConstantDescriptor::default();
        assert_eq!(
            desc.dump(),
            "N-Way Conditional Constant Descriptor:\n\tInvalid\n"
        );
    }

    /// The vendor's own case: a matched conditional reaches the constants, so the seam it stops at is
    /// the analysis's own `operator<<` and not some earlier gap. ⛔ Nothing written before the stop is
    /// observable — the `String` dies with the panic.
    #[test]
    #[should_panic(expected = "EvaluatedValue::operator<<")]
    fn e425_a_matched_descriptor_stops_at_the_out_of_scope_rendering() {
        let desc = ConditionalConstantDescriptor {
            yielded_constants: vec![EvaluatedValue(4), EvaluatedValue(4)],
            can_be_simplified: true,
        };
        let _ = desc.dump();
    }
}
