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
// ⭐ REMOVE THIS WITH `e467_replaceCorrelatedIterArgsInEquivClass`, WHICH IS THE ONE CALLER MISSING.
// `e355_runOnOperation` is now ported and reaches e147 and e634, but e148 and e357 are read only from
// e467, so an unused item here stays indistinguishable from a real defect until that lands.
#![allow(dead_code)]

use crate::arch::Arch;
use crate::islands::sentient::dialects::{self, Definitions, Op, Val, sentient};
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

/// Replaces: e355_runOnOperation
///
/// The pass entry — `ReuseLoopIteratorArguments` over one whole program (`:102-106`).
///
/// ⛔ `dcc-reuse-loop-iterator-arguments-disable` IS DROPPED, as every `cl::opt` of this campaign is:
/// it is a flag of `dcc-opt`, not a property of a program, and this crate has no flags. `cl::init(false)`
/// (`:58-60`) makes running the pass the answer it always gives here.
pub fn run_on_operation<A: Arch, M: Model, W: Workload>(program: &mut Program<A, M, W>) {
    run_on_program(program);
}

/// Replaces: e356_reuseIdenticalIterArgs
///
/// Collapses a loop's iterator arguments onto one representative per group of arguments that carry the
/// same affine expression in the same register file at the same element size (`:345-409`).
///
/// ⛔ THE GROUPING PREDICATE IS `PropagationAnalysis::areExpressionsSame`, WHICH IS OUT OF CAMPAIGN
/// SCOPE — a loop with two or more still-ungrouped candidates reaches the `todo!` below. Everything
/// around it is ported: the two early exits, the locale/element-size filter and the rewrite.
/// ⛔ TRAP: THE GROUP HEAD IS THE LOWEST INDEX AND `curr_iter` SKIPS ANY ARGUMENT ALREADY ASSIGNED ONE
/// (`:373`, `:380`), so grouping is not transitive through a middle argument that was claimed first.
pub fn reuse_identical_iter_args(for_op: &mut Op) {
    let Op::Sentient(sentient::Op::For { carried, body, .. }) = for_op else {
        return;
    };
    // `if (n_iter_args == 0) return;` (`:348`).
    if carried.is_empty() {
        return;
    }
    // ⭐ `if (!locales || !element_sizes) return;` (`:352`). `getRegLocalesAttr()` CANNOT BE ABSENT ON
    // THIS ISLAND — [`sentient::Carried::reg`] is a field of the loop, not an optional attribute — so
    // the surviving half is the `element_sizes` array, whose absence is every slot being `None`.
    // ⭐ AND `DT_CHECK_MSG(getRegLocales().size() == 2 * n_iter_args + 1, ...)` (`:353-354`) IS THAT
    // SAME FIELD: one `Reg` per carried value is the shape, so the check has nothing left to refuse.
    if carried.iter().all(|value| value.element_size.is_none()) {
        return;
    }

    // `iter_arg_idx_to_head_idx`, initialised to the identity (`:366-368`).
    let mut head_of: Vec<usize> = (0..carried.len()).collect();
    for curr in 0..carried.len() {
        if head_of[curr] != curr {
            continue;
        }
        for next in curr + 1..carried.len() {
            if head_of[next] != next {
                continue;
            }
            // `locales[curr+1] != locales[next+1] || element_sizes[curr+1] != element_sizes[next+1]`
            // (`:384-387`) — the `+ 1` is the induction variable's slot, which `carried` does not have.
            if carried[curr].reg.locale != carried[next].reg.locale
                || carried[curr].element_size != carried[next].element_size
            {
                continue;
            }
            if expressions_are_same(carried[curr].arg, carried[next].arg) {
                head_of[next] = curr;
            }
        }
    }

    // `curr_iter_arg.replaceAllUsesWith(replace_with_arg)` (`:407`) — ⭐ WITHIN THE BODY, because a
    // region argument has no use outside the region that binds it.
    for curr in 0..carried.len() {
        if head_of[curr] == curr {
            continue;
        }
        dialects::replace_all_uses_with(body, carried[curr].arg, carried[head_of[curr]].arg);
    }
}

/// `PropagationAnalysis::areExpressionsSame` (`:389-390`) — whether two iterator arguments advance by
/// the same affine expression. `Analyses/PropagationAnalysis` is not in this campaign.
fn expressions_are_same(a: Val, b: Val) -> bool {
    todo!("PropagationAnalysis::areExpressionsSame({a:?}, {b:?}) — out of campaign scope")
}

/// Replaces: e357_areUsersLiverangesOverlapping
///
/// Whether any non-yield-feeding user of `first` and any of `second` bind results that are live at the
/// same time — the reason a correlated pair must NOT be collapsed (`:458-482`).
///
/// ⛔ `Liveness::isLiveRangeOverlaps` IS OUT OF CAMPAIGN SCOPE. The pair that shares a defining op is
/// answered here, since *"if both results belong to the same operation, their live ranges automatically
/// overlap"* (`:475-476`); anything else reaches the `todo!`.
/// ⛔ TRAP: EMPTY EITHER SIDE ANSWERS `false` — the cross product runs zero times (`:472`), so an
/// iterator argument whose every user feeds the yield never blocks the collapse.
#[must_use]
pub fn are_users_liveranges_overlapping(
    first: IterArg,
    second: IterArg,
    body: &[Op],
    defs: Definitions<'_>,
) -> bool {
    let firsts = collect_results_of_non_yield_feeding_users(first, body);
    let seconds = collect_results_of_non_yield_feeding_users(second, body);
    firsts.iter().any(|result1| {
        seconds.iter().any(|result2| {
            share_a_defining_op(*result1, *result2, defs) || live_range_overlaps(*result1, *result2)
        })
    })
}

/// `result1.getDefiningOp() == result2.getDefiningOp()` (`:475`) — ⛔ NOT `result1 == result2`: one
/// `sentient.load_and_store` binds a source result AND a destination result, and
/// [`result_corresponding_to_operand_num`] hands back either, so two distinct values can share an op.
fn share_a_defining_op(a: Val, b: Val, defs: Definitions<'_>) -> bool {
    a == b
        || defs
            .of(a)
            .is_some_and(|op| dialects::results(op).contains(&b))
}

/// `Liveness::isLiveRangeOverlaps` (`:478`) — `Analyses/Liveness` is not in this campaign.
fn live_range_overlaps(a: Val, b: Val) -> bool {
    todo!("Liveness::isLiveRangeOverlaps({a:?}, {b:?}) — out of campaign scope")
}

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
    use super::{
        IterArg, are_users_liveranges_overlapping, collect_results_of_non_yield_feeding_users,
        reuse_identical_iter_args,
    };
    use crate::arch::Elements;
    use crate::formats::Bits;
    use crate::islands::dataflow_ir::ty::ScalarTy;
    use crate::islands::sentient::dialects::Definitions;
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
            result_reg: sentient::Reg::UNALLOCATED,
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

    /// A `sentient.for` carrying `carried` and running `body`.
    fn for_op(carried: Vec<sentient::Carried>, body: Vec<Op>) -> Op {
        Op::Sentient(sentient::Op::For {
            iv: Val(1),
            bound: Val(2),
            carried,
            iv_reg: sentient::Reg::UNALLOCATED,
            dbg_name: None,
            body,
        })
    }

    /// Both early exits, and a loop with a single carried value that has nothing to group with — the
    /// three shapes that never reach `PropagationAnalysis::areExpressionsSame`.
    #[test]
    fn a_loop_with_nothing_to_group_is_left_exactly_as_it_was() {
        let sized = |mut value: sentient::Carried| {
            value.element_size = Some(Bits(16));
            value
        };
        let body = vec![
            add(Val(10), Val(11), Val(12)),
            Op::Sentient(sentient::Op::Yield {
                results: vec![Val(12)],
            }),
        ];
        // `n_iter_args == 0`.
        let mut none = for_op(Vec::new(), body.clone());
        let before = none.clone();
        reuse_identical_iter_args(&mut none);
        assert_eq!(none, before);
        // `!element_sizes` — every slot unset is the array being absent.
        let mut unsized_loop = for_op(vec![carried(Val(1), Val(10), Val(2))], body.clone());
        let before = unsized_loop.clone();
        reuse_identical_iter_args(&mut unsized_loop);
        assert_eq!(unsized_loop, before);
        // One sized carried value: the inner loop has no `next_iter`, so it is its own head.
        let mut single = for_op(vec![sized(carried(Val(1), Val(10), Val(2)))], body);
        let before = single.clone();
        reuse_identical_iter_args(&mut single);
        assert_eq!(single, before);
    }

    /// Two iterator arguments read by one op overlap by construction; two whose every user feeds the
    /// yield have nothing to compare and do not overlap.
    #[test]
    fn users_sharing_a_defining_op_overlap_and_yield_feeding_users_are_not_compared() {
        let (arg1, arg2, step1, step2) = (Val(10), Val(11), Val(20), Val(21));
        let (advanced1, advanced2, shared) = (Val(12), Val(13), Val(14));
        let carried_values = [carried(Val(1), arg1, Val(2)), carried(Val(3), arg2, Val(4))];
        // Every user is the arithmetic that advances its own slot.
        let feeding = vec![
            add(arg1, step1, advanced1),
            add(arg2, step2, advanced2),
            Op::Sentient(sentient::Op::Yield {
                results: vec![advanced1, advanced2],
            }),
        ];
        let scopes: [&[Op]; 1] = [&feeding];
        let defs = Definitions::from_innermost(&scopes);
        let first = IterArg::at(&carried_values, &feeding, 0).expect("slot 0 exists");
        let second = IterArg::at(&carried_values, &feeding, 1).expect("slot 1 exists");
        assert!(!are_users_liveranges_overlapping(
            first, second, &feeding, defs
        ));
        // ⭐ ONE `scalar_add` READING BOTH: `result1.getDefiningOp() == result2.getDefiningOp()`.
        let together = vec![
            add(arg1, arg2, shared),
            add(arg1, step1, advanced1),
            add(arg2, step2, advanced2),
            Op::Sentient(sentient::Op::Yield {
                results: vec![advanced1, advanced2],
            }),
        ];
        let scopes: [&[Op]; 1] = [&together];
        let defs = Definitions::from_innermost(&scopes);
        let first = IterArg::at(&carried_values, &together, 0).expect("slot 0 exists");
        let second = IterArg::at(&carried_values, &together, 1).expect("slot 1 exists");
        assert!(are_users_liveranges_overlapping(
            first, second, &together, defs
        ));
    }
}
