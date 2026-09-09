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

//! `ToggleReordering.cpp` — 4 of the campaign's 656 units (dependency level(s) [0, 1, 2, 3]).
//!
//! | unit | entry | level | lines | authority path:line |
//! |---|---|---|---|---|
//! | `e233_computeToggleInfoIfIsToggle` | 233 | 0 | 39 | `dcc/src/Transform/Sentient/ToggleReordering.cpp:222` |
//! | `e390_runOn` | 390 | 1 | 21 | `dcc/src/Transform/Sentient/ToggleReordering.cpp:200` |
//! | `e481_runOn` | 481 | 2 | 18 | `dcc/src/Transform/Sentient/ToggleReordering.cpp:181` |
//! | `e542_runOnOperation` | 542 | 3 | 5 | `dcc/src/Transform/Sentient/ToggleReordering.cpp:149` |

// ⛔ THE PASS DRIVERS ARE `e390_runOn`/`e481_runOn`/`e542_runOnOperation`, still open anchors below,
// so nothing outside this module calls any of it yet and CI's `-D warnings` would fail on the first
// ported leaf. ⭐ REMOVE THIS WITH `e390_runOn`.
#![allow(dead_code)]

pub(crate) mod toggle_info;

use crate::islands::dataflow_ir::ty::ScalarTy;
use crate::islands::sentient::dialects::{
    Op, Val, defining_op, parent_for_arg, sentient, uniform, use_count,
};
use crate::transform::sentient::analyses::ExpressionEvaluator;
use toggle_info::{ToggleInfo, yielded_results};

/// `sentient::SubOp` — the four things e233 reads off a candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SubOp {
    /// `getInp1()` — the minuend, which must be constant for this to be a toggle.
    pub minuend: Val,
    /// `getInp2()` — the subtrahend, which must be a loop iter arg.
    pub subtrahend: Val,
    /// `getResult()`.
    pub result: Val,
    /// `getType()`.
    pub ty: ScalarTy,
}

impl SubOp {
    /// `dyn_cast<sentient::SubOp>(op)` — the walk's own parameter type, as a match.
    #[must_use]
    pub fn of(op: &Op) -> Option<SubOp> {
        match op {
            Op::Sentient(sentient::Op::ScalarSub {
                lhs,
                rhs,
                result,
                ty,
                ..
            }) => Some(SubOp {
                minuend: *lhs,
                subtrahend: *rhs,
                result: *result,
                ty: *ty,
            }),
            _ => None,
        }
    }
}

/// Replaces: e233_computeToggleInfoIfIsToggle
///
/// A [`ToggleInfo`] when `sub_op` is `constant - iter_arg`, the iter arg is read by nothing else, its
/// slot of the loop's yield holds the sub's result, and the chain's outer initializer is constant.
///
/// ⛔ TRAP: `dcc::utils::getOutermostConstInitialization` IS THIS CAMPAIGN'S e246
/// (`dcc/src/Transform/Sentient/Utils.cpp:469`, homed in `transform/sentient/utils`), and the
/// scheduler did not record the edge — the call is namespace-qualified rather than a member call. Every
/// guard before it is ported; see [`outermost_const_initialization`] for the rest.
/// ⭐ `unit_op` IS DROPPED: it only positioned `const_builder`, which is
/// [`crate::transform::sentient::analyses::OffsetSites::consts`] at [`ToggleInfo::reorder_toggle`].
pub fn compute_toggle_info_if_is_toggle<E: ExpressionEvaluator>(
    evaluator: &mut E,
    unit_body: &[Op],
    sub_op: SubOp,
) -> Option<ToggleInfo> {
    if !is_sentient_constant(sub_op.minuend, unit_body) {
        return None;
    }
    // `getIndexOfLoopRegionIterArgs(sub_inp2)` and the `DT_CHECK_MSG(iter_arg, ..)` after it are one
    // question: only a `sentient.for` region argument has both a parent loop and an index.
    let (loop_op, iter_arg_idx) = index_of_loop_region_iter_args(sub_op.subtrahend, unit_body)?;
    if use_count(sub_op.subtrahend, unit_body) != 1 {
        return None;
    }
    // `if (yield_op.getOperand(iter_arg_idx) != sub_result) return nullptr;`
    let Op::Sentient(sentient::Op::For { body, .. }) = loop_op else {
        return None;
    };
    if yielded_results(body)?.get(iter_arg_idx) != Some(&sub_op.result) {
        return None;
    }

    let (outer_loop_iv, outer_loop_carried) =
        outermost_const_initialization(sub_op.subtrahend, unit_body)?;
    // `if (!isa<sentient::ConstantOp>(outer_loop.getIterOperands()[idx].getDefiningOp()))` — ⛔ THE
    // BARE `isa`, NOT `isConstant`: a `uniform.query_map` initializer is accepted by the FIRST guard
    // and refused by this one.
    let init = carried_init(unit_body, outer_loop_iv, outer_loop_carried)?;
    if !matches!(
        defining_op(init, unit_body),
        Some(Op::Sentient(sentient::Op::ScalarConstant { .. }))
    ) {
        return None;
    }

    Some(ToggleInfo {
        sub_result: sub_op.result,
        sub_ty: sub_op.ty,
        iter_arg: sub_op.subtrahend,
        minuend_ev: evaluator.evaluate_value(sub_op.minuend),
        outer_loop_iv,
        outer_loop_carried,
    })
}

/// `dcc::utils::getOutermostConstInitialization` (`dcc/src/Transform/Sentient/Utils.cpp:469`) — the
/// outer loop of `iter_arg`'s iter-arg chain and the carried index whose initializer is constant.
///
/// ⛔ IT IS e246 AND IT IS NOT PORTED YET, so the only honest body is the one below. `None` collapses
/// the caller's `!outer_loop || outer_loop_iter_arg_idx == -1` refusal (`:251`); the tuple's third
/// element, the chain's trip-count product, has no reader here.
fn outermost_const_initialization(iter_arg: Val, scope: &[Op]) -> Option<(Val, usize)> {
    let _ = (iter_arg, scope);
    todo!(
        "getOutermostConstInitialization (senpass e246, transform/sentient/utils) is not ported yet \
         — the inner-to-outer iter-arg chain walk, its constant-bound trip-count product and its \
         one-use/ForOp-user refusals (Transform/Sentient/Utils.cpp:469-524) are what this delegates \
         to (ToggleReordering.cpp:248)"
    )
}

/// `dcc::utils::getIndexOfLoopRegionIterArgs` (`Analyses/Utils.cpp:257`) — the owning `sentient.for`
/// and the position of `val` in its `getRegionIterArgs()`.
///
/// ⛔ THE INDUCTION VARIABLE ANSWERS `-1`: `getRegionIterArgs()` is the body arguments with the
/// induction variable dropped, so position 0 of [`parent_for_arg`] is in no iter-arg list.
fn index_of_loop_region_iter_args(val: Val, scope: &[Op]) -> Option<(&Op, usize)> {
    let (op, index) = parent_for_arg(val, scope)?;
    Some((op, index.checked_sub(1)?))
}

/// One `sentient.for`'s `getIterOperands()[carried]`, the loop named by its induction variable.
fn carried_init(scope: &[Op], iv: Val, carried: usize) -> Option<Val> {
    fn walk(scope: &[Op], iv: Val) -> Option<&[sentient::Carried]> {
        for op in scope {
            if let Op::Sentient(sentient::Op::For {
                iv: at,
                carried,
                body,
                ..
            }) = op
            {
                if *at == iv {
                    return Some(carried);
                }
                if let Some(found) = walk(body, iv) {
                    return Some(found);
                }
            } else if let Op::Sentient(inner) = op {
                for region in sentient::regions(inner) {
                    if let Some(found) = walk(region, iv) {
                        return Some(found);
                    }
                }
            } else if let Op::AffineFor(loop_op) = op
                && let Some(found) = walk(&loop_op.body, iv)
            {
                return Some(found);
            }
        }
        None
    }
    walk(scope, iv)
        .and_then(|list| list.get(carried))
        .map(|carried| carried.init)
}

/// `dcc::utils::isConstant<sentient::ConstantOp>` (`dcc/src/Utils/Utils.cpp:424`) — a
/// `sentient.scalar_constant`, or a `uniform.query_map` whose mapping answers with nothing else.
///
/// ⛔ A REGION ARGUMENT IS NOT CONSTANT — the `isa<BlockArgument>` refusal at `:426`, which here is
/// simply having no defining op.
fn is_sentient_constant(val: Val, scope: &[Op]) -> bool {
    let is_scalar_constant =
        |op: Option<&Op>| matches!(op, Some(Op::Sentient(sentient::Op::ScalarConstant { .. })));
    match defining_op(val, scope) {
        Some(Op::Uniform(uniform::Op::QueryMap { map, .. })) => match defining_op(*map, scope) {
            Some(Op::Uniform(uniform::Op::DefImmutableMapping { pairs, .. })) => {
                !pairs.is_empty()
                    && pairs
                        .iter()
                        .all(|(_, value)| is_scalar_constant(defining_op(*value, scope)))
            }
            _ => false,
        },
        other => is_scalar_constant(other),
    }
}

// crustify:todo: e390_runOn
//   authority : dcc/src/Transform/Sentient/ToggleReordering.cpp:200  (21 body lines, level 1)
//   original  : void ToggleReorderingPass::runOn(dataflow::ProgramUnitOp unit_op)
//   calls     : e232_reorderToggle, e233_computeToggleInfoIfIsToggle

// crustify:todo: e481_runOn
//   authority : dcc/src/Transform/Sentient/ToggleReordering.cpp:181  (18 body lines, level 2)
//   original  : void ToggleReorderingPass::runOn(ModuleOp module_op)
//   calls     : e390_runOn

// crustify:todo: e542_runOnOperation
//   authority : dcc/src/Transform/Sentient/ToggleReordering.cpp:149  (5 body lines, level 3)
//   original  : void runOnOperation()
//   calls     : e390_runOn, e481_runOn

#[cfg(test)]
mod unit_tests {
    use super::{SubOp, compute_toggle_info_if_is_toggle};
    use crate::islands::dataflow_ir::ty::ScalarTy;
    use crate::islands::sentient::dialects::{Op, Val, sentient};
    use crate::transform::sentient::analyses::OutOfScopeEvaluator;

    /// The candidate: `%31 = sentient.scalar_sub %30, %20`.
    const SUB: SubOp = SubOp {
        minuend: Val(30),
        subtrahend: Val(20),
        result: Val(31),
        ty: ScalarTy::Index,
    };

    /// One carried value, at the register defaults a fixture needs none of.
    fn carried(init: Val, arg: Val, result: Val) -> sentient::Carried {
        sentient::Carried {
            init,
            arg,
            result,
            reg: sentient::Reg {
                locale: sentient::RegType::Imm,
                index: None,
            },
            program_header: false,
            element_size: None,
        }
    }

    /// `%30 = sentient.scalar_constant`, then a loop carrying `%20` whose body holds the toggle and
    /// `yields` the ops of `body` after it.
    fn unit(minuend: Op, body: Vec<Op>) -> Vec<Op> {
        vec![
            minuend,
            Op::Sentient(sentient::Op::For {
                iv: Val(11),
                bound: Val(12),
                bound_reg: None,
                carried: vec![carried(Val(2), Val(20), Val(21))],
                dbg_name: None,
                body,
            }),
        ]
    }

    /// `%31 = sentient.scalar_sub %30, %20 : index`.
    fn toggle() -> Op {
        Op::Sentient(sentient::Op::ScalarSub {
            lhs: Val(30),
            rhs: Val(20),
            result: Val(31),
            reg: None,
            element_size: None,
            ty: ScalarTy::Index,
        })
    }

    /// `%30 = sentient.scalar_constant 7 : index`.
    fn constant() -> Op {
        Op::Sentient(sentient::Op::ScalarConstant {
            value: 7,
            result: Val(30),
            reg_locale: sentient::RegType::Imm,
            ty: ScalarTy::Index,
            is_symbol: false,
        })
    }

    /// 🎯 e233 — each of the three criteria the doc comment lists (`:159-165`) refuses on its own.
    #[test]
    fn a_sub_that_misses_any_criterion_is_not_a_toggle() {
        let yields_the_toggle = || {
            Op::Sentient(sentient::Op::Yield {
                results: vec![Val(31)],
            })
        };
        // A minuend that is not constant — a `scalar_add` defines `%30` instead.
        let not_constant = Op::Sentient(sentient::Op::ScalarAdd {
            lhs: Val(0),
            rhs: Val(0),
            result: Val(30),
            reg: None,
            element_size: None,
            ty: ScalarTy::Index,
        });
        // A second reader of the iter arg — `!iter_arg.hasOneUse()`.
        let second_reader = Op::Sentient(sentient::Op::ScalarAdd {
            lhs: Val(20),
            rhs: Val(20),
            result: Val(32),
            reg: None,
            element_size: None,
            ty: ScalarTy::Index,
        });
        // A yield carrying something else at the iter arg's index.
        let yields_other = Op::Sentient(sentient::Op::Yield {
            results: vec![Val(2)],
        });
        for unit_body in [
            unit(not_constant, vec![toggle(), yields_the_toggle()]),
            unit(
                constant(),
                vec![toggle(), second_reader, yields_the_toggle()],
            ),
            unit(constant(), vec![toggle(), yields_other]),
            // And a subtrahend that is no loop iter arg at all: `%12` is the bound, not an argument.
            unit(constant(), vec![yields_the_toggle()]),
        ] {
            assert_eq!(
                compute_toggle_info_if_is_toggle(&mut OutOfScopeEvaluator, &unit_body, SUB),
                None
            );
        }
    }

    /// 🎯 e233 — a candidate that passes every ported criterion reaches the one guard that is not
    /// ported, which is `e246_getOutermostConstInitialization`.
    #[test]
    #[should_panic(expected = "getOutermostConstInitialization (senpass e246")]
    fn a_real_toggle_reaches_the_unported_outer_initializer_walk() {
        let unit_body = unit(
            constant(),
            vec![
                toggle(),
                Op::Sentient(sentient::Op::Yield {
                    results: vec![Val(31)],
                }),
            ],
        );
        let _ = compute_toggle_info_if_is_toggle(&mut OutOfScopeEvaluator, &unit_body, SUB);
    }
}
