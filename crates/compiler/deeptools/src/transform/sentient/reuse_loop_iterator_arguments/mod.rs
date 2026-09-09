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

//! `ReuseLoopIteratorArguments.cpp` — 7 of the campaign's 656 units (dependency level(s) [0, 1, 2, 6]).
//!
//! | unit | entry | level | lines | authority path:line |
//! |---|---|---|---|---|
//! | `e147_runOn` | 147 | 0 | 7 | `dcc/src/Transform/Sentient/ReuseLoopIteratorArguments.cpp:94` |
//! | `e148_collectResultsOfNonYieldFeedingUsers` | 148 | 0 | 19 | `dcc/src/Transform/Sentient/ReuseLoopIteratorArguments.cpp:411` |
//! | `e355_runOnOperation` | 355 | 1 | 5 | `dcc/src/Transform/Sentient/ReuseLoopIteratorArguments.cpp:102` |
//! | `e356_reuseIdenticalIterArgs` | 356 | 1 | 64 | `dcc/src/Transform/Sentient/ReuseLoopIteratorArguments.cpp:345` |
//! | `e357_areUsersLiverangesOverlapping` | 357 | 1 | 23 | `dcc/src/Transform/Sentient/ReuseLoopIteratorArguments.cpp:458` |
//! | `e467_replaceCorrelatedIterArgsInEquivClass` | 467 | 2 | 81 | `dcc/src/Transform/Sentient/ReuseLoopIteratorArguments.cpp:259` |
//! | `e634_runOn` | 634 | 6 | 118 | `dcc/src/Transform/Sentient/ReuseLoopIteratorArguments.cpp:140` |

// ⛔ THE PASS IS NOT WIRED INTO THE PIPELINE YET, so everything below is reachable only from this
// file's own tests until `e355_runOnOperation` lands and something calls it. CI runs clippy with
// `-D warnings`, so without this the first ported leaf of the module fails the gate.
// ⭐ REMOVE THIS WITH `e355_runOnOperation`: at that point an unused item here is a real defect again.
#![allow(dead_code)]

use crate::arch::Arch;
use crate::islands::sentient::dialects::{self, Op, Val, sentient};
use crate::islands::sentient::{Program, ProgramUnit};
use crate::model::Model;
use crate::workload::Workload;

/// Replaces: e147_runOn
///
/// Runs the pass over every program unit of one module — the `walk<WalkOrder::PreOrder>` that visits
/// each `dataflow.program_unit` and then skips its subtree (`:94-100`).
///
/// ⛔ NAMED FOR ITS ARGUMENT: `runOn(ModuleOp)` and `runOn(dataflow::ProgramUnitOp)` (e634) are one
/// C++ overload set and cannot both be `run_on` here.
///
/// ⭐ `WalkResult::skip()` NEEDS NO EXPRESSION: a program's units are a flat list on this island, so
/// there is no nested `dataflow.program_unit` for the walk to have to decline.
#[expect(
    clippy::never_loop,
    reason = "the body diverges only because `run_on_unit` is e634's `todo!`; the loop is the port"
)]
pub(crate) fn run_on_program<A: Arch, M: Model, W: Workload>(program: &mut Program<A, M, W>) {
    for unit in program.units.iter_mut() {
        run_on_unit(unit);
    }
}

/// `runOn(dataflow::ProgramUnitOp)` — entry 634, level 6, not yet ported.
fn run_on_unit<A: Arch>(unit: &mut ProgramUnit<A>) -> ! {
    let _ = unit;
    todo!(
        "e634_runOn(dataflow::ProgramUnitOp) — the toggle-correlation round, the identical-iter-arg \
         round and the correlated-equivalence-class round (ReuseLoopIteratorArguments.cpp:140)"
    )
}

/// ONE LOOP ITERATOR ARGUMENT AND WHAT ITS SLOT YIELDS — `iter_arg` paired with
/// `yield_op.getOperand(iter_arg.getArgNumber() - 1)` (`:420-421`).
///
/// # ⛔⛔ THE OFF-BY-ONE IS THE INDUCTION VARIABLE AND NO CALLER SHOULD REPEAT IT
///
/// `getArgNumber()` counts the region's arguments and slot 0 is the `$iv`
/// (`SentientOps.td:96-99`), so carried value `i` arrives as argument `i + 1` and is yielded at
/// operand `i`. Pairing the two here is what stops a caller indexing the `sentient.yield` by the
/// ARGUMENT number: that reads the NEIGHBOURING slot's yielded value, which skips a user that should
/// have been collected and collects one that should have been skipped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IterArg {
    /// The region argument the body reads — [`sentient::Carried::arg`].
    arg: Val,
    /// What the loop's `sentient.yield` hands back at this position.
    yielded: Val,
}

impl IterArg {
    /// Carried value `at` of a `sentient.for`, read off the loop's own `carried` list and the
    /// `sentient.yield` that terminates its body.
    ///
    /// ⭐ `None` WHERE EITHER SIDE HAS NO SUCH SLOT, which stands for the two handles the reference
    /// has already checked before it gets here — `for_op` and the `dyn_cast<YieldOp>` of its
    /// terminator (e357, `:461-464`).
    #[must_use]
    pub fn at(carried: &[sentient::Carried], body: &[Op], at: usize) -> Option<IterArg> {
        let Some(Op::Sentient(sentient::Op::Yield { results })) = body.last() else {
            return None;
        };
        Some(IterArg {
            arg: carried.get(at)?.arg,
            yielded: *results.get(at)?,
        })
    }
}

/// Replaces: e148_collectResultsOfNonYieldFeedingUsers
///
/// For every use of the iterator argument in `body`, the user's result that reading it produces —
/// skipping the `scalar_add`/`scalar_sub` whose result IS what this slot yields (`:411-431`).
///
/// ⛔ TRAP: THE SKIP IS THE YIELD-FEEDING ARITHMETIC, NOT THE `sentient.yield` ITSELF. A loop that
/// yields its iter arg unchanged has the terminator among its users, and a terminator binds nothing —
/// so that shape reaches the reference's `DT_CHECK(user_result)` (`:427`) rather than being skipped by
/// it. See [`result_corresponding_to_operand_num`].
///
/// ⭐ ONE ENTRY PER **USE**, NOT PER USER: `%1 = scalar_add %arg1, %arg1` contributes `%1` twice,
/// because `iter_arg.getUses()` yields two uses that the same `getResults()[0]` answers.
/// ⚠️ IN PROGRAM ORDER, WHERE `getUses()` IS IN MLIR'S USE-LIST ORDER — inert, because the one
/// consumer runs the two lists as a full cross product with an early `return true` (`:472-481`).
#[must_use]
pub fn collect_results_of_non_yield_feeding_users(iter_arg: IterArg, body: &[Op]) -> Vec<Val> {
    let mut results: Vec<Val> = Vec::new();
    collect_into(iter_arg, body, &mut results);
    results
}

/// One region of the use walk. ⭐ [`dialects::regions`] AND NOT `regions_ref`, because it descends
/// into a lower-rung op's region too — a reader there is still a reader, and the reference's
/// `getUses()` finds it wherever it is.
fn collect_into(iter_arg: IterArg, scope: &[Op], out: &mut Vec<Val>) {
    for op in scope {
        for (operand_num, read) in dialects::operands(op).into_iter().enumerate() {
            if read != iter_arg.arg || feeds_the_yield(op, iter_arg.yielded) {
                continue;
            }
            match result_corresponding_to_operand_num(op, operand_num) {
                Some(result) => out.push(result),
                // `DT_CHECK(user_result);` (`:427`) — the reference aborts, and it is not this
                // function's decision to turn that into a skip: a dropped user is a liverange
                // overlap e357 would then fail to see.
                None => todo!(
                    "collectResultsOfNonYieldFeedingUsers: DT_CHECK(user_result) — no result of \
                     {op:?} corresponds to operand {operand_num} \
                     (ReuseLoopIteratorArguments.cpp:427, Analyses/Utils.cpp:674)"
                ),
            }
        }
        for region in dialects::regions(op) {
            collect_into(iter_arg, &region, out);
        }
    }
}

/// `isa<sentient::AddOp, sentient::SubOp>(user) && yield_op.getOperand(..) == user->getResults()[0]`
/// (`:418-421`) — the arithmetic that advances this slot, *"ignored as they would be removed if the
/// iterator argument is replaced by another plus an offset"* (`:455-457`).
fn feeds_the_yield(op: &Op, yielded: Val) -> bool {
    match op {
        Op::Sentient(
            sentient::Op::ScalarAdd { result, .. } | sentient::Op::ScalarSub { result, .. },
        ) => *result == yielded,
        _ => false,
    }
}

/// `dcc::utils::getResultCorrespondingToOperandNum` (`Analyses/Utils.cpp:674-694`) — the result an op
/// produces from the value it reads at `operand_num`.
///
/// ⛔ OUT OF THE CAMPAIGN'S FILE SCOPE AND STILL PORTED, for the reason `RemoveStaticCondition` was:
/// it is not one of the out-of-scope `Analyses/` ANALYSES, it is a nine-line table, and a `todo!`
/// here would panic on every iterator argument e357 examines.
///
/// ⛔ `None` IS THE REFERENCE'S `nullptr` AND ITS CALLER ABORTS ON IT: `getNumResults() == 0` is the
/// first line (`:676`), and the table names neither `sentient.if` nor any op of another dialect.
///
/// ⛔⛔ DELIBERATE DIVERGENCE ON `sentient.vector_mac`. The reference indexes its results by the
/// ABSOLUTE operand number, `op->getResults()[operand_num]` (`:683`), and `$mask` is operand 0 when
/// present (`SentientOps.td:237-239`) — so on a MASKED mac the write pointer at operand 1 reads back
/// the READ pointer's result and the read pointer at operand 2 indexes past a two-result op. This port
/// maps by POINTER POSITION instead, which is the unmasked case's answer and the documented intent,
/// *"the result value corresponding to the operand number"* (`Analyses/Utils.hpp:226-232`).
fn result_corresponding_to_operand_num(op: &Op, operand_num: usize) -> Option<Val> {
    let Op::Sentient(inner) = op else {
        return None;
    };
    match inner {
        // `op->getResults()[0]` for all five, whichever operand was read (`:678-680`).
        sentient::Op::ScalarAdd { result, .. }
        | sentient::Op::ScalarSub { result, .. }
        | sentient::Op::LoadAndSend { result, .. }
        | sentient::Op::ReceiveAndStore { result, .. }
        | sentient::Op::LoadComputeAndSend { result, .. } => Some(*result),
        // `op->getResults()[operand_num - 1]` — *"ForOp's operand 0 is the loop bound"* (`:681-682`),
        // and one result per carried value is what this island already binds.
        sentient::Op::For { carried, .. } => Some(carried.get(operand_num.checked_sub(1)?)?.result),
        // The write pointer's result then the read pointer's — see this function's divergence note.
        sentient::Op::VectorMac {
            mask,
            xrf_write_ptr,
            xrf_read_ptr,
            results,
            ..
        } => {
            let write_at = usize::from(mask.is_some());
            if xrf_write_ptr.is_some() && operand_num == write_at {
                return results.first().copied();
            }
            if xrf_read_ptr.is_some()
                && operand_num == write_at + usize::from(xrf_write_ptr.is_some())
            {
                return results.get(1).copied();
            }
            None
        }
        // The source end's three addresses answer `$src_res` and the destination's answer `$dst_res`
        // (`:685-692`); `$src`, `$dst` and `$multicast_info` answer neither. ⭐ THE POSITIONS ARE
        // [`dialects::operands`]'S, which is the `.td`'s own order (`SentientOps.td:722-730`).
        sentient::Op::LoadAndStore { results, .. } => match operand_num {
            2..=4 => Some(results.0),
            5..=7 => Some(results.1),
            _ => None,
        },
        // `return nullptr;` (`:694`) — every op the table does not name, `sentient.yield` included.
        _ => None,
    }
}

// crustify:todo: e355_runOnOperation
//   authority : dcc/src/Transform/Sentient/ReuseLoopIteratorArguments.cpp:102  (5 body lines, level 1)
//   original  : void runOnOperation()
//   calls     : e147_runOn

// crustify:todo: e356_reuseIdenticalIterArgs
//   authority : dcc/src/Transform/Sentient/ReuseLoopIteratorArguments.cpp:345  (64 body lines, level 1)
//   original  : void ReuseLoopIteratorArgumentsPass::reuseIdenticalIterArgs( sentient::ForOp for_op, PropagationAnalysis &expr_prop_analysis)
//   calls     : e252_size

// crustify:todo: e357_areUsersLiverangesOverlapping
//   authority : dcc/src/Transform/Sentient/ReuseLoopIteratorArguments.cpp:458  (23 body lines, level 1)
//   original  : bool ReuseLoopIteratorArgumentsPass::areUsersLiverangesOverlapping( BlockArgument iter_arg1, BlockArgument iter_arg2, const Liveness &liveness) const
//   calls     : e148_collectResultsOfNonYieldFeedingUsers

// crustify:todo: e467_replaceCorrelatedIterArgsInEquivClass
//   authority : dcc/src/Transform/Sentient/ReuseLoopIteratorArguments.cpp:259  (81 body lines, level 2)
//   original  : void ReuseLoopIteratorArgumentsPass::replaceCorrelatedIterArgsInEquivClass( OpBuilder &const_builder, sentient::ForOp for_op, CorrelationAnalysisBase::CorrelatedEquivClassContainer *correlated_equiv_classes, const Liveness &liveness, bool check_users_liverange_conflicts)
//   calls     : e357_areUsersLiverangesOverlapping

// crustify:todo: e634_runOn
//   authority : dcc/src/Transform/Sentient/ReuseLoopIteratorArguments.cpp:140  (118 body lines, level 6)
//   original  : void ReuseLoopIteratorArgumentsPass::runOn(dataflow::ProgramUnitOp unit_op)
//   calls     : e356_reuseIdenticalIterArgs, e467_replaceCorrelatedIterArgsInEquivClass, e597_runLightWeightSimplifications

#[cfg(test)]
mod unit_tests {
    use super::{IterArg, collect_results_of_non_yield_feeding_users};
    use crate::arch::Elements;
    use crate::formats::Bits;
    use crate::islands::dataflow_ir::ty::ScalarTy;
    use crate::islands::sentient::dialects::sentient::StoreSource;
    use crate::islands::sentient::dialects::{Op, Val, sentient};

    /// `%out = sentient.scalar_add %lhs, %rhs : index`.
    fn add(lhs: Val, rhs: Val, result: Val) -> Op {
        Op::Sentient(sentient::Op::ScalarAdd {
            lhs,
            rhs,
            result,
            reg: None,
            element_size: None,
            ty: ScalarTy::Index,
        })
    }

    /// A `sentient.receive_and_store` whose mutable address is `mutable_addr`.
    fn store(mutable_addr: Val, result: Val) -> Op {
        Op::Sentient(sentient::Op::ReceiveAndStore {
            mutable_addr,
            immutable_addr: Val(91),
            increment: Val(92),
            producer: StoreSource::Constant(Val(93)),
            result,
            dst: None,
            drop_first: None,
            multicast_info: None,
            extent: sentient::Extent::of(Elements(64), Bits(16)),
            interleaved_group: Elements(0),
            coalesce: false,
            subword_length: 0,
            stride: 0,
            permute: false,
            shuffle_mode: None,
            reg: sentient::Reg {
                locale: sentient::RegType::Imm,
                index: None,
            },
            dbg_name: None,
        })
    }

    /// One carried value, with nothing assigned to it yet.
    fn carried(init: Val, arg: Val, result: Val) -> sentient::Carried {
        sentient::Carried {
            init,
            arg,
            result,
            reg: sentient::Reg {
                locale: sentient::RegType::Unknown,
                index: None,
            },
            program_header: false,
            element_size: None,
        }
    }

    /// The vendor's own shape: the `scalar_add` that advances the iterator argument is skipped, and
    /// the transfer that reads it as an address contributes the value it binds.
    #[test]
    fn the_add_that_feeds_this_slots_yield_is_the_only_user_skipped() {
        let (arg, step, advanced, stored) = (Val(10), Val(11), Val(12), Val(13));
        let body = vec![
            add(arg, step, advanced),
            store(arg, stored),
            Op::Sentient(sentient::Op::Yield {
                results: vec![advanced],
            }),
        ];
        let iter_arg = IterArg::at(&[carried(Val(1), arg, Val(2))], &body, 0)
            .expect("the loop carries one value and its body ends in a yield");
        assert_eq!(
            collect_results_of_non_yield_feeding_users(iter_arg, &body),
            vec![stored]
        );
    }

    /// An `scalar_add` reading the same iterator argument but NOT producing what this slot yields is
    /// collected — one entry per USE, so reading it twice contributes its result twice.
    #[test]
    fn an_add_that_feeds_something_else_is_collected_once_per_use() {
        let (arg, other, advanced, aside) = (Val(10), Val(11), Val(12), Val(14));
        let body = vec![
            add(arg, arg, aside),
            add(arg, other, advanced),
            Op::Sentient(sentient::Op::Yield {
                results: vec![advanced],
            }),
        ];
        let iter_arg = IterArg::at(&[carried(Val(1), arg, Val(2))], &body, 0)
            .expect("the loop carries one value and its body ends in a yield");
        assert_eq!(
            collect_results_of_non_yield_feeding_users(iter_arg, &body),
            vec![aside, aside]
        );
    }
}
