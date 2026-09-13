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
// ⭐ AND `e634_runOn` DID NOT REMOVE IT: e634 is now ported and calls e467, e356 and e597, but e634's
// own caller is e147, whose caller is the pass ENTRY e355 — and nothing calls a pass entry until the
// pipeline is wired. ⭐ REMOVE THIS WHEN SOMETHING CALLS `run_on_operation`.
#![allow(dead_code)]

use crate::arch::Arch;
use crate::islands::dataflow_ir::Values;
use crate::islands::dataflow_ir::ty::ScalarTy;
use crate::islands::sentient::dialects::{self, Op, Val, sentient};
use crate::islands::sentient::{Program, ProgramUnit};
use crate::model::Model;
use crate::transform::sentient::ForRef;
use crate::transform::sentient::analyses::{
    CorrelatedEquivClasses, Correlation, CorrelationAnalysis, ExpressionEvaluator, Liveness,
    PropagationAnalysis, UnitIndexMap,
};
use crate::transform::sentient::lightweight_simplification::sentient::run_light_weight_simplifications;
use crate::transform::sentient::scalar_simplifications::sentient::Propagation;
use crate::units::DfirUnit;
use crate::workload::Workload;

/// `-dcc-reuse-loop-iterator-arguments-disable`, `cl::init(false)` (`:57-60`).
const DISABLE_THIS_PASS: bool = false;

/// `-l3-lxlu-toggle-correlation-only`, `cl::init(true)` (`:61-65`) — ⛔ NOTE THE `true`: the pass's
/// default is to run on the L3 halves and the LX load unit ONLY.
const UNIT_SPECIFIC_TOGGLE_CORRELATION: bool = true;

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
///
/// ⭐ THE RETURNED [`Refused`]S ARE EVERY UNIT'S, IN UNIT ORDER — `signalPassFailure()` is a flag on
/// the pass and the walk does not stop for it, so this is a list and not a first failure.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_on_program<
    A: Arch,
    M: Model,
    W: Workload,
    L: Liveness,
    T: CorrelationAnalysis,
    F: CorrelationAnalysis,
    P: PropagationAnalysis,
    E: ExpressionEvaluator,
    U: UnitIndexMap,
>(
    program: &mut Program<A, M, W>,
    liveness: &L,
    toggle_correlation: &mut T,
    affine_correlation: &mut F,
    propagation: &mut P,
    evaluator: &mut E,
    unit_index_map: &U,
    values: &mut Values,
) -> Vec<Refused> {
    let Program {
        preamble, units, ..
    } = program;
    let mut refused = Vec::new();
    for unit in units.iter_mut() {
        refused.extend(run_on_unit(
            preamble,
            unit,
            liveness,
            toggle_correlation,
            affine_correlation,
            propagation,
            evaluator,
            unit_index_map,
            values,
        ));
    }
    refused
}

/// `unit_op->emitError(..); signalPassFailure();` AS DATA — which of the round's four analyses
/// refused (`:152-153`, `:200-201`, `:213-214`, `:239-241`).
///
/// ⭐ NOT A `Result`: the reference has no `return` after any of the four, so a refusal names one
/// analysis and every later round of the same unit still runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Refused {
    /// "Unable to perform toggle correlation analysis".
    ToggleCorrelation,
    /// "failed in simplifying code".
    Simplification,
    /// "Unable to perform expression propagation".
    ExpressionPropagation,
    /// "Unable to perform affine expression correlation analysis".
    AffineExpressionCorrelation,
}

/// `unit_op->walk<WalkOrder::PostOrder>([&](sentient::ForOp op) { .. })` — every `sentient.for` under
/// the unit, innermost first. ⭐ NO REWRITE HERE GROWS THE BLOCK BEING WALKED: both round-1 and
/// round-3 insert into the module preamble and into the loop's OWN body.
fn walk_for_ops_post_order(block: &mut [Op], visit: &mut impl FnMut(&mut Op)) {
    for op in block.iter_mut() {
        for region in dialects::regions_mut(op).iter_mut() {
            walk_for_ops_post_order(region, visit);
        }
        if matches!(op, Op::Sentient(sentient::Op::For { .. })) {
            visit(op);
        }
    }
}

/// Replaces: e634_runOn
///
/// One program unit in three rounds: correlated toggles become the head toggle plus an offset, the
/// arithmetic that leaves behind is simplified, and then identical and correlated AFFINE iterator
/// arguments are reused the same way (`:140-257`).
///
/// ⛔ THE WHOLE BODY IS UNDER ONE UNIT GATE (`:148-149`): with
/// [`UNIT_SPECIFIC_TOGGLE_CORRELATION`] on, a unit that is not an L3 half or the LX load unit gets
/// NOTHING from this pass — not even rounds 2 and 3.
/// ⛔ SIMPLIFICATION RUNS BETWEEN ROUND 1 AND ROUND 2 AND THAT ORDER IS LOAD-BEARING (`:195-199`,
/// the reference's own note): round 3 breaks the one-use assumption the simplifications rely on.
/// ⛔ THE CONST BUILDER IS IN THE **MODULE'S** BLOCK (`:161`, `unit_op->getBlock()`), which is where
/// both rounds' correlation constants and e597's constants go.
/// ⚠️ THE TWO CORRELATION ANALYSES ARE THE CALLER'S: the reference constructs one of each PER UNIT
/// (`:150`, `:236-237`), and their constructors are the walks `Analyses/` owns and this campaign does
/// not carry. ⭐ `DT_CHECK_MSG(get_unit_op, "Cannot determine GetUnitOp!")` is discharged by
/// [`crate::islands::dataflow_ir::Units`], which cannot name a unit without one.
#[allow(clippy::too_many_arguments)]
fn run_on_unit<
    A: Arch,
    L: Liveness,
    T: CorrelationAnalysis,
    F: CorrelationAnalysis,
    P: PropagationAnalysis,
    E: ExpressionEvaluator,
    U: UnitIndexMap,
>(
    preamble: &mut Vec<Op>,
    unit: &mut ProgramUnit<A>,
    liveness: &L,
    toggle_correlation: &mut T,
    affine_correlation: &mut F,
    propagation: &mut P,
    evaluator: &mut E,
    unit_index_map: &U,
    values: &mut Values,
) -> Vec<Refused> {
    let mut refused = Vec::new();
    if UNIT_SPECIFIC_TOGGLE_CORRELATION
        && !matches!(
            unit.on.kind(),
            DfirUnit::L3lu | DfirUnit::L3su | DfirUnit::Lxlu
        )
    {
        return refused;
    }
    if toggle_correlation.is_correlation_successful() == Correlation::Failed {
        refused.push(Refused::ToggleCorrelation);
    }
    // Optimization 1: every non-head iterator argument of a correlated-toggle class becomes its head
    // plus an offset. ⭐ AND WITHOUT THE LIVE-RANGE CHECK, for the reason the reference gives at
    // `:167-186`: an inner loop's arguments cannot correlate until these ones have.
    walk_for_ops_post_order(&mut unit.body, &mut |for_op| {
        let Op::Sentient(sentient::Op::For { iv, .. }) = for_op else {
            return;
        };
        let classes = toggle_correlation.correlated_equiv_classes(ForRef(*iv));
        replace_correlated_iter_args_in_equiv_class(
            preamble, for_op, classes, liveness, false, values,
        );
    });
    // ⭐ E597 SHARES THE SAME CONST BUILDER (`:200`), so its constants land at the front of the module
    // block beside round 1's rather than in the unit's own body.
    let mut consts = Vec::new();
    let simplified = run_light_weight_simplifications(
        &mut consts,
        &mut unit.body,
        evaluator,
        propagation,
        unit_index_map,
        values,
    );
    preamble.splice(0..0, consts);
    if simplified == Propagation::Failed {
        refused.push(Refused::Simplification);
    }
    if !propagation.is_propagation_successful() {
        refused.push(Refused::ExpressionPropagation);
    }
    // Optimization 2: iterator arguments with identical affine expressions share one slot's argument.
    walk_for_ops_post_order(&mut unit.body, &mut |for_op| {
        let Op::Sentient(sentient::Op::For { carried, body, .. }) = for_op else {
            return;
        };
        reuse_identical_iter_args(body, carried, propagation);
    });
    if affine_correlation.is_correlation_successful() == Correlation::Failed {
        refused.push(Refused::AffineExpressionCorrelation);
    }
    // Optimization 3: round 1 again over the AFFINE classes — and this one does check the users' live
    // ranges (`:249`).
    walk_for_ops_post_order(&mut unit.body, &mut |for_op| {
        let Op::Sentient(sentient::Op::For { iv, .. }) = for_op else {
            return;
        };
        let classes = affine_correlation.correlated_equiv_classes(ForRef(*iv));
        replace_correlated_iter_args_in_equiv_class(
            preamble, for_op, classes, liveness, true, values,
        );
    });
    refused
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
/// The pass entry: unless [`DISABLE_THIS_PASS`] is set, run [`run_on_program`] over the module.
///
/// ⛔ A THIRD NAME FOR A THIRD ENTRY: `runOnOperation`, `runOn(ModuleOp)` (e147) and
/// `runOn(dataflow::ProgramUnitOp)` (e634) are three members of one class, and only C++ overloading
/// lets two of them share a spelling.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_on_operation<
    A: Arch,
    M: Model,
    W: Workload,
    L: Liveness,
    T: CorrelationAnalysis,
    F: CorrelationAnalysis,
    P: PropagationAnalysis,
    E: ExpressionEvaluator,
    U: UnitIndexMap,
>(
    program: &mut Program<A, M, W>,
    liveness: &L,
    toggle_correlation: &mut T,
    affine_correlation: &mut F,
    propagation: &mut P,
    evaluator: &mut E,
    unit_index_map: &U,
    values: &mut Values,
) -> Vec<Refused> {
    if DISABLE_THIS_PASS {
        return Vec::new();
    }
    run_on_program(
        program,
        liveness,
        toggle_correlation,
        affine_correlation,
        propagation,
        evaluator,
        unit_index_map,
        values,
    )
}

/// Replaces: e356_reuseIdenticalIterArgs
///
/// Groups the loop's iterator arguments by identical affine expression — same register locale, same
/// element size, and `areExpressionsSame` — then points every use of a non-head argument at its
/// group's head (`:345-409`).
///
/// ⛔ THE EFFECT IS THE `replaceAllUsesWith`, and only `body` is rewritten: the `carried` list keeps
/// every slot, which is what e634's later rounds and the loop's own signature still read.
/// ⛔ TRAP: THE SCAN IS OVER HEADS ONLY, in index order, so the LOWEST index of a group is its head
/// and the grouping is deterministic where the reference's `unordered_map` is not.
/// ⛔ `locales[i + 1]` AND `element_sizes[i + 1]` ARE `carried[i]` HERE: the `+1` skips the induction
/// variable's slot of the reference's `1 + 2n` arrays, which [`sentient::Carried`] does not carry —
/// and that also discharges its `DT_CHECK(getRegLocales().size() == 2 * n_iter_args + 1)` (`:352`).
/// ⚠️ `if (!locales || !element_sizes) return;` (`:351`) IS DROPPED AS INEXPRESSIBLE: a
/// [`sentient::Carried`] always has both, and in the shipped pipeline both attributes are always
/// present by the time this pass runs (`dcc-standalone-main.cpp:318` assigns the locales).
pub fn reuse_identical_iter_args(
    body: &mut [Op],
    carried: &[sentient::Carried],
    expr_prop_analysis: &mut impl PropagationAnalysis,
) {
    let n_iter_args = carried.len();
    if n_iter_args == 0 {
        return;
    }
    let mut head: Vec<usize> = (0..n_iter_args).collect();
    for curr_iter in 0..n_iter_args {
        if head[curr_iter] != curr_iter {
            continue;
        }
        for next_iter in (curr_iter + 1)..n_iter_args {
            if head[next_iter] != next_iter
                || carried[curr_iter].reg.locale != carried[next_iter].reg.locale
                || carried[curr_iter].element_size != carried[next_iter].element_size
            {
                continue;
            }
            if expr_prop_analysis
                .are_expressions_same(carried[curr_iter].arg, carried[next_iter].arg)
            {
                head[next_iter] = curr_iter;
            }
        }
    }
    for curr_iter in 0..n_iter_args {
        if head[curr_iter] == curr_iter {
            continue;
        }
        dialects::replace_all_uses_with(body, carried[curr_iter].arg, carried[head[curr_iter]].arg);
    }
}

/// Replaces: e357_areUsersLiverangesOverlapping
///
/// Whether any non-yield-feeding user of the first iterator argument and any of the second hold values
/// that are live at once — in which case replacing one argument by the other plus an offset would add a
/// `scalar_add` without saving a register (`:458-482`).
///
/// ⛔ TWO RESULTS OF ONE OP OVERLAP BY CONSTRUCTION, and that is checked before the analysis
/// (`:477-478`): [`dialects::defining_op`] is recursive, so the op that binds one result is found
/// wherever it sits and asked whether it binds the other too.
/// ⛔ `Liveness::isLiveRangeOverlaps` IS OUT OF CAMPAIGN SCOPE — see
/// [`crate::transform::sentient::analyses::Liveness`]; it is a parameter so that the cross product,
/// the early exit and the same-op shortcut are all observable without it.
#[must_use]
pub fn are_users_liveranges_overlapping(
    iter_arg1: IterArg,
    iter_arg2: IterArg,
    body: &[Op],
    liveness: &impl Liveness,
) -> bool {
    let results1 = collect_results_of_non_yield_feeding_users(iter_arg1, body);
    let results2 = collect_results_of_non_yield_feeding_users(iter_arg2, body);
    for result1 in &results1 {
        for result2 in &results2 {
            if dialects::defining_op(*result1, body)
                .is_some_and(|op| dialects::results(op).contains(result2))
            {
                return true;
            }
            if liveness.is_live_range_overlaps(*result1, *result2) {
                return true;
            }
        }
    }
    false
}

/// Replaces: e467_replaceCorrelatedIterArgsInEquivClass
///
/// Points every correlated iterator argument at its class's head plus an offset: one
/// `sentient.scalar_constant` at the start of `preamble` and one `sentient.scalar_add` at the start of
/// the loop body per replacement, then a RAUW of the argument onto that add.
///
/// ⛔ EVERY OVERLAP CHECK RUNS BEFORE THE FIRST REWRITE (`:280-283`, the reference's own NOTE): each
/// `scalar_add` this creates is a new user of an iterator argument, so a replacement made mid-scan
/// would be checked against ops the reference had not created yet.
/// ⛔⛔ DELIBERATE DIVERGENCE, AND IT IS AN OFF-BY-ONE IN THE REFERENCE. `locales[correlated_idx + 1]`
/// with `correlated_idx = getArgNumber()` (`:333-338`) reads slot `i + 2` of `[bound, initArgs…,
/// results…]` for carried `i`, because `getArgNumber()` already counts the `$iv`
/// (`SentientOps.td:96-102`) — the NEIGHBOURING slot's locale and size. `carried[i]` is the intended
/// one, and e356 indexes the same arrays `locales[i + 1]` (`:378-380`).
/// ⭐ `if (locales && element_sizes)` IS DROPPED for e356's reason, and `iter_arg.getType()` is
/// [`ScalarTy::Index`] for [`hoist_candidate_out_of_loop`]'s.
///
/// [`hoist_candidate_out_of_loop`]: super::scalar_op_merging_and_hoisting::scalar_op_hoisting::hoist_candidate_out_of_loop
pub fn replace_correlated_iter_args_in_equiv_class(
    preamble: &mut Vec<Op>,
    for_op: &mut Op,
    correlated_equiv_classes: Option<&CorrelatedEquivClasses>,
    liveness: &impl Liveness,
    check_users_liverange_conflicts: bool,
    values: &mut Values,
) {
    let Op::Sentient(sentient::Op::For { carried, body, .. }) = for_op else {
        return;
    };
    if carried.is_empty() {
        return;
    }
    let Some(equiv_classes) = correlated_equiv_classes.filter(|classes| !classes.is_empty()) else {
        return;
    };
    // `(iter_arg_to_replace, head_iter_arg, correlation)` — the carried slot rather than the argument,
    // because the slot is what carries the locale and the element size the replacement copies.
    let mut replacements: Vec<(usize, Val, i64)> = Vec::new();
    for equiv_class in &equiv_classes.classes {
        let head_iter_arg = equiv_class.head_iter_arg;
        // ⭐ A HEAD THIS LOOP DOES NOT CARRY IS SKIPPED, which the reference cannot be handed: the
        // container is keyed by the `sentient.for` whose arguments the class holds.
        let Some(head_at) = carried.iter().position(|slot| slot.arg == head_iter_arg) else {
            continue;
        };
        for &(correlated_iter_arg, correlation) in &equiv_class.correlations {
            let Some(correlated_at) = carried.iter().position(|s| s.arg == correlated_iter_arg)
            else {
                continue;
            };
            if check_users_liverange_conflicts
                && let Some(head) = IterArg::at(carried, body, head_at)
                && let Some(correlated) = IterArg::at(carried, body, correlated_at)
                && are_users_liveranges_overlapping(head, correlated, body, liveness)
            {
                continue;
            }
            replacements.push((correlated_at, head_iter_arg, correlation));
        }
    }
    // ⭐ `inserted` IS BOTH BUILDERS' INSERTION POINT ADVANCING: `create` leaves the point after the
    // op it inserted, so successive constants and adds land in creation order at the start of the
    // block rather than in reverse.
    for (inserted, (correlated_at, head_iter_arg, correlation_value)) in
        replacements.into_iter().enumerate()
    {
        let correlation = values.mint();
        preamble.insert(
            inserted,
            Op::Sentient(sentient::Op::ScalarConstant {
                value: correlation_value,
                result: correlation,
                reg_locale: sentient::RegType::Imm,
                ty: ScalarTy::Index,
                is_symbol: false,
            }),
        );
        let replacement = values.mint();
        body.insert(
            inserted,
            Op::Sentient(sentient::Op::ScalarAdd {
                lhs: head_iter_arg,
                rhs: correlation,
                result: replacement,
                // `setAttr("regLocale", ..)` — the index is NOT set, and unsaid is not `Unknown`.
                reg: Some(sentient::Reg {
                    locale: carried[correlated_at].reg.locale,
                    index: None,
                }),
                element_size: carried[correlated_at].element_size,
                ty: ScalarTy::Index,
            }),
        );
        dialects::replace_all_uses_with(body, carried[correlated_at].arg, replacement);
    }
}

#[cfg(test)]
mod unit_tests {
    use super::{
        IterArg, Program, ProgramUnit, are_users_liveranges_overlapping,
        collect_results_of_non_yield_feeding_users, replace_correlated_iter_args_in_equiv_class,
        reuse_identical_iter_args, run_on_program,
    };
    use crate::arch::Dd2;
    use crate::arch::Elements;
    use crate::formats::Bits;
    use crate::generated::OpFunc;
    use crate::islands::dataflow_ir::Values;
    use crate::islands::dataflow_ir::ty::ScalarTy;
    use crate::islands::dataflow_ir::{GroupId, OpIndex, ProgramName, Units};
    use crate::islands::sentient::ProgramUnits;
    use crate::islands::sentient::dialects::sentient::StoreSource;
    use crate::islands::sentient::dialects::{Op, Val, sentient};
    use crate::model::Model;
    use crate::transform::sentient::analyses::{
        CorrelatedEquivClass, CorrelatedEquivClasses, Liveness,
        OutOfScopeAffineExpressionCorrelation, OutOfScopeEvaluator, OutOfScopePropagationAnalysis,
        OutOfScopeToggleCorrelation, OutOfScopeUnitIndexMap, PropagationAnalysis, VirtualAssigns,
    };
    use crate::units::DfirUnit;
    use crate::workload::Workload;

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

    /// `PropagationAnalysis&` that calls a fixed set of pairs equal, which is the only way to observe
    /// [`reuse_identical_iter_args`]'s grouping.
    struct SameFor(Vec<(Val, Val)>);
    impl PropagationAnalysis for SameFor {
        fn are_expressions_same(&mut self, val1: Val, val2: Val) -> bool {
            self.0.contains(&(val1, val2))
        }
    }

    /// `Liveness&` that never reports an overlap, so only the same-op shortcut can answer `true`.
    struct NeverOverlaps;
    impl Liveness for NeverOverlaps {
        fn update_live_ranges_for_program_header_promotion(&mut self, _candidate: Val) {}
        fn is_live_range_overlaps(&self, _v1: Val, _v2: Val) -> bool {
            false
        }

        fn clear(&mut self, _virtual_assigns: VirtualAssigns) {
            todo!("no unit here clears this fake")
        }

        fn compute_register_live_range(&mut self, _unit: &[Op]) {
            todo!("no unit here recomputes this fake")
        }

        fn add_virtual_assign_optional(&mut self, _set_of_subsets: &[Vec<Val>]) {
            todo!("no unit here links optional assignments through this fake")
        }

        fn add_virtual_assign_enforced(&mut self, _set_of_pairs: &[(Val, Val)]) {
            todo!("no unit here links enforced assignments through this fake")
        }
    }

    /// `e356` — the second argument's uses move to the first, and the third is left alone because its
    /// register locale differs however the analysis answers.
    #[test]
    fn identical_iter_args_are_reused_and_a_different_locale_is_not() {
        let (a0, a1, a2) = (Val(10), Val(11), Val(12));
        let mut slots = vec![
            carried(Val(1), a0, Val(20)),
            carried(Val(2), a1, Val(21)),
            carried(Val(3), a2, Val(22)),
        ];
        slots[2].reg.locale = sentient::RegType::Lbr;
        // ⭐ THE YIELD HANDS BACK THE SLOTS' ADVANCED VALUES, not the arguments themselves — a loop
        // that yields an argument unchanged is the reference's own `DT_CHECK` shape (see e148).
        let mut body = vec![
            store(a0, Val(30)),
            store(a1, Val(31)),
            store(a2, Val(32)),
            Op::Sentient(sentient::Op::Yield {
                results: vec![Val(50), Val(51), Val(52)],
            }),
        ];
        // The analysis calls all three pairs equal; only the first two share a locale.
        let mut prop = SameFor(vec![(a0, a1), (a0, a2), (a1, a2)]);

        reuse_identical_iter_args(&mut body, &slots, &mut prop);

        assert_eq!(
            collect_results_of_non_yield_feeding_users(
                IterArg::at(&slots, &body, 0).expect("slot 0"),
                &body
            ),
            vec![Val(30), Val(31)]
        );
        // ⛔ THE `carried` LIST STILL HOLDS EVERY SLOT.
        assert_eq!(slots.len(), 3);
        // ⛔ AND THE THIRD ARGUMENT'S USE IS UNTOUCHED.
        assert_eq!(
            collect_results_of_non_yield_feeding_users(
                IterArg::at(&slots, &body, 2).expect("slot 2"),
                &body
            ),
            vec![Val(32)]
        );
    }

    /// `e357` — two results of ONE op overlap before the analysis is consulted, and two separate
    /// transfers do not once it declines.
    #[test]
    fn two_results_of_one_op_overlap_without_asking_the_analysis() {
        let (a0, a1) = (Val(10), Val(11));
        let slots = [carried(Val(1), a0, Val(20)), carried(Val(2), a1, Val(21))];
        let shared = vec![
            add(a0, a1, Val(30)),
            Op::Sentient(sentient::Op::Yield {
                results: vec![Val(50), Val(51)],
            }),
        ];
        let (first, second) = (
            IterArg::at(&slots, &shared, 0).expect("slot 0"),
            IterArg::at(&slots, &shared, 1).expect("slot 1"),
        );
        assert!(are_users_liveranges_overlapping(
            first,
            second,
            &shared,
            &NeverOverlaps
        ));

        let apart = vec![
            store(a0, Val(30)),
            store(a1, Val(31)),
            Op::Sentient(sentient::Op::Yield {
                results: vec![Val(50), Val(51)],
            }),
        ];
        assert!(!are_users_liveranges_overlapping(
            IterArg::at(&slots, &apart, 0).expect("slot 0"),
            IterArg::at(&slots, &apart, 1).expect("slot 1"),
            &apart,
            &NeverOverlaps
        ));
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

    /// `e467` — the vendor's own shape: the second argument's one use moves onto `head + 4`, the
    /// constant lands in the enclosing block and the add copies the SLOT's locale and element size, not
    /// its neighbour's.
    #[test]
    fn a_correlated_iter_arg_is_replaced_by_the_head_plus_its_offset() {
        let (a0, a1) = (Val(10), Val(11));
        let mut slots = vec![carried(Val(1), a0, Val(20)), carried(Val(2), a1, Val(21))];
        slots[0].reg.locale = sentient::RegType::Lbr;
        slots[1].reg.locale = sentient::RegType::Lar;
        slots[1].element_size = Some(Bits(16));
        let mut for_op = Op::Sentient(sentient::Op::For {
            iv: Val(9),
            bound: Val(8),
            bound_reg: None,
            carried: slots,
            dbg_name: None,
            body: vec![
                store(a1, Val(31)),
                Op::Sentient(sentient::Op::Yield {
                    results: vec![Val(50), Val(51)],
                }),
            ],
        });
        let mut preamble: Vec<Op> = Vec::new();
        let classes = CorrelatedEquivClasses {
            classes: vec![CorrelatedEquivClass {
                head_iter_arg: a0,
                correlations: vec![(a1, 4)],
            }],
        };
        let mut values = Values::default();
        for _ in 0..100 {
            values.mint();
        }

        replace_correlated_iter_args_in_equiv_class(
            &mut preamble,
            &mut for_op,
            Some(&classes),
            &NeverOverlaps,
            true,
            &mut values,
        );

        let Op::Sentient(sentient::Op::For { carried, body, .. }) = &for_op else {
            unreachable!("the loop is still a loop")
        };
        let Op::Sentient(sentient::Op::ScalarConstant {
            value,
            result: correlation,
            ..
        }) = preamble[0]
        else {
            unreachable!("the offset constant is built in the enclosing block")
        };
        assert_eq!(value, 4);
        assert_eq!(
            body[0],
            Op::Sentient(sentient::Op::ScalarAdd {
                lhs: a0,
                rhs: correlation,
                result: Val(101),
                // ⛔ THE SLOT'S OWN LOCALE AND SIZE — `Lar`/16, not slot 0's `Lbr`/`None`, which is
                // what the reference's `locales[getArgNumber() + 1]` would have copied.
                reg: Some(sentient::Reg {
                    locale: sentient::RegType::Lar,
                    index: None,
                }),
                element_size: Some(Bits(16)),
                ty: ScalarTy::Index,
            })
        );
        // The one reader of the replaced argument now reads the add.
        assert_eq!(
            collect_results_of_non_yield_feeding_users(
                IterArg::at(carried, body, 1).expect("slot 1"),
                body
            ),
            Vec::<Val>::new()
        );
        assert_eq!(
            body[1],
            store(Val(101), Val(31)),
            "the transfer reads the add"
        );
        // ⛔ THE `carried` LIST IS UNTOUCHED — only uses moved.
        assert_eq!(carried.len(), 2);
    }

    /// ANY MODEL AND ANY RUNG — this pass reads neither.
    #[derive(Debug, Clone, PartialEq, Eq)]
    struct AnyModel;
    impl Model for AnyModel {
        const QUERY_HEADS: u32 = 32;
        const KV_HEADS: u32 = 8;
        const HEAD_DIM: u32 = 64;
        const HIDDEN: u32 = 2048;
        const LAYERS: u32 = 40;
        const FFN: u32 = 8192;
        const VOCAB: u32 = 49152;
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct AnyRung;
    impl Workload for AnyRung {
        const ROWS: u32 = 1;
        const ACTIVE_CAP: u32 = 64;
    }

    /// One program of one unit running on `on`, carrying the e467 fixture's loop.
    fn program_on(on: DfirUnit) -> Program<Dd2, AnyModel, AnyRung> {
        Program {
            name: ProgramName {
                group: GroupId(0),
                index: OpIndex(0),
                func: OpFunc::Add,
            },
            preamble: Vec::new(),
            units: ProgramUnits::of(
                ProgramUnit {
                    on: Units::one(on, Val(1)),
                    precision: None,
                    body: vec![Op::Sentient(sentient::Op::For {
                        iv: Val(9),
                        bound: Val(8),
                        bound_reg: None,
                        carried: vec![carried(Val(1), Val(10), Val(20))],
                        dbg_name: None,
                        body: vec![
                            store(Val(10), Val(31)),
                            Op::Sentient(sentient::Op::Yield {
                                results: vec![Val(50)],
                            }),
                        ],
                    })],
                    arch: core::marker::PhantomData,
                },
                Vec::new(),
            ),
            bound: core::marker::PhantomData,
        }
    }

    /// e634 — an LXLU unit is inside the gate, so the first thing the pass asks for is the toggle
    /// correlation this campaign does not carry.
    #[test]
    #[should_panic(expected = "ToggleCorrelationAnalysis::isCorrelationSuccessful")]
    fn e634_a_gated_in_unit_asks_for_the_toggle_correlation_first() {
        let mut program = program_on(DfirUnit::Lxlu);
        let _ = run_on_program(
            &mut program,
            &NeverOverlaps,
            &mut OutOfScopeToggleCorrelation,
            &mut OutOfScopeAffineExpressionCorrelation,
            &mut OutOfScopePropagationAnalysis,
            &mut OutOfScopeEvaluator,
            &OutOfScopeUnitIndexMap,
            &mut Values::default(),
        );
    }

    /// e634's negative — ⛔ AN SFP UNIT GETS NOTHING, not even rounds 2 and 3: every analysis here
    /// `todo!`s, so reaching any round at all would panic instead of returning.
    #[test]
    fn e634_a_unit_outside_the_gate_is_skipped_without_asking_anything() {
        let mut program = program_on(DfirUnit::Sfp);
        let before = program.units.iter().next().expect("one unit").body.clone();
        let refused = run_on_program(
            &mut program,
            &NeverOverlaps,
            &mut OutOfScopeToggleCorrelation,
            &mut OutOfScopeAffineExpressionCorrelation,
            &mut OutOfScopePropagationAnalysis,
            &mut OutOfScopeEvaluator,
            &OutOfScopeUnitIndexMap,
            &mut Values::default(),
        );
        assert_eq!(refused, Vec::new());
        assert_eq!(program.units.iter().next().expect("one unit").body, before);
        assert!(program.preamble.is_empty());
    }
}
