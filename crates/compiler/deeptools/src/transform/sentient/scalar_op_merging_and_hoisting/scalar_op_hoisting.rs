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

//! `ScalarOpMergingAndHoisting.cpp` — 13 of the campaign's 656 units (dependency level(s) [0, 1, 2, 3, 4, 5, 6]).
//!
//! | unit | entry | level | lines | authority path:line |
//! |---|---|---|---|---|
//! | `e170_addForOpResultAdjustment` | 170 | 0 | 28 | `dcc/src/Transform/Sentient/ScalarOpMergingAndHoisting.cpp:1543` |
//! | `e171_isMergeableOpOrChain` | 171 | 0 | 33 | `dcc/src/Transform/Sentient/ScalarOpMergingAndHoisting.cpp:1592` |
//! | `e172_addToOrReplaceOp` | 172 | 0 | 14 | `dcc/src/Transform/Sentient/ScalarOpMergingAndHoisting.cpp:1716` |
//! | `e173_hoistCandidateOutOfLoop` | 173 | 0 | 22 | `dcc/src/Transform/Sentient/ScalarOpMergingAndHoisting.cpp:1805` |
//! | `e365_applyOperationData` | 365 | 1 | 54 | `dcc/src/Transform/Sentient/ScalarOpMergingAndHoisting.cpp:1740` |
//! | `e468_processMergeableChain` | 468 | 2 | 58 | `dcc/src/Transform/Sentient/ScalarOpMergingAndHoisting.cpp:1645` |
//! | `e469_hoistForLinearChain` | 469 | 2 | 29 | `dcc/src/Transform/Sentient/ScalarOpMergingAndHoisting.cpp:2034` |
//! | `e529_processForOpResult` | 529 | 3 | 13 | `dcc/src/Transform/Sentient/ScalarOpMergingAndHoisting.cpp:1484` |
//! | `e530_processForDerivedIVElimination` | 530 | 3 | 38 | `dcc/src/Transform/Sentient/ScalarOpMergingAndHoisting.cpp:1843` |
//! | `e578_adjustCandidateForOpResult` | 578 | 4 | 19 | `dcc/src/Transform/Sentient/ScalarOpMergingAndHoisting.cpp:1514` |
//! | `e612_processForLinearChain` | 612 | 5 | 129 | `dcc/src/Transform/Sentient/ScalarOpMergingAndHoisting.cpp:1893` |
//! | `e613_processForGenericHoisting` | 613 | 5 | 81 | `dcc/src/Transform/Sentient/ScalarOpMergingAndHoisting.cpp:2075` |
//! | `e635_runScalarOpHoisting` | 635 | 6 | 61 | `dcc/src/Transform/Sentient/ScalarOpMergingAndHoisting.cpp:2184` |

#![allow(dead_code)]
// ⛔ THE WHOLE CHAIN IS STILL UNREACHABLE — `e635_runScalarOpHoisting` has its caller now
// (`e366_runOn`), but the pass is not wired into the pipeline until `e646_runOnOperation`
// (`ScalarOpMergingAndHoisting.cpp:2404`, level 7), which is a different batch's unit. CI runs clippy
// with `-D warnings`, so without this the batch fails its own gate.
// ⭐ REMOVE THIS WITH e646: an unused item here is a real defect at that point.

use super::{
    AddressScale, ImmRange, MemoryOpInfo, OperationData, ScalarOpComp, does_value_exceed_lrf_range,
    is_immutable_value_in_range,
};
use crate::arch::{Arch, Elements};
use crate::formats::Bits;
use crate::islands::dataflow_ir::Values;
use crate::islands::dataflow_ir::ty::{GenericComp, ScalarTy};
use crate::islands::sentient::dialects::sentient as ops;
use crate::islands::sentient::dialects::{
    Definitions, Op, Val, defining_op, erase_defining_op, operands, regions_mut, regions_ref,
    replace_all_uses_with, results, set_operand, uniform, use_count,
};
use crate::transform::sentient::IterArgIndex;
use crate::transform::sentient::analyses::{
    EvaluatedValue, Evaluation, ExpressionEvaluator, OffsetSites,
};
use crate::transform::sentient::old_register_initialization::register_init_info::is_target_constant;

/// `ibuff_space_` — how many instruction-buffer entries are left for the ops this pass creates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct IbuffSpace(pub i32);

/// `num_hoists_executed_per_unit` (`:95`) — the hoists this unit has already executed, which
/// `MaxHoists` caps and e635 resets between units.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct HoistCount(pub(crate) u32);

/// WHAT SPENDING AN IBUFF ENTRY LEFT BEHIND — `DT_CHECK_MSG(ibuff_space_ >= 0, "No IBUFF space!")`
/// (`:1571`) as data, because this crate does not spell a check as an abort.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub(crate) enum IbuffSpent {
    /// The unit still fits its instruction buffer.
    WithinBudget,
    /// It no longer does — the reference's abort, reported rather than performed.
    Overrun,
}

impl IbuffSpace {
    /// `--ibuff_space_` and the check that follows it (`:1570-1571`).
    ///
    /// ⭐ `Overrun` IS UNREACHABLE THROUGH THE REFERENCE'S OWN CALLER:
    /// `adjustCandidateForOpResult` declines at `ibuff_space_ <= 0` before it ever gets here
    /// (`:1524`).
    pub(crate) fn spend_one(&mut self) -> IbuffSpent {
        self.0 -= 1;
        if self.0 >= 0 {
            IbuffSpent::WithinBudget
        } else {
            IbuffSpent::Overrun
        }
    }
}

/// `OperationData::getReplaceWithMod()` — whether the modifier REPLACES the constant a chain op
/// reads, or is added to what it already holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Modification {
    /// `replace_with_mod == true`.
    Replace,
    /// `replace_with_mod == false`.
    AddTo,
}

/// AN `EvaluatedValue` PROVED ABSOLUTE — `DT_CHECK(adjustment_increment.isKnownAbsolute())` (`:1546`)
/// as a type, so [`add_for_op_result_adjustment`] has nothing left to assert.
#[derive(Debug, Clone, Copy)]
pub(crate) struct KnownAbsolute<'a>(&'a Evaluation);

impl<'a> KnownAbsolute<'a> {
    /// `None` for an evaluation that is not known absolute, which is the reference's abort.
    pub(crate) fn of(evaluation: &'a Evaluation) -> Option<KnownAbsolute<'a>> {
        evaluation
            .known_absolute
            .then_some(KnownAbsolute(evaluation))
    }
}

/// A VALUE PROVED CONSTANT — `DT_CHECK_MSG(isConstant<ConstantOp>(op->getResult(0)), "Expect op to
/// be constant")` (`:1719`) as a type, for the same reason [`KnownAbsolute`] is one.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ConstantValue(Val);

impl ConstantValue {
    /// `None` when the value is neither a `sentient.scalar_constant` nor an all-constant query map.
    pub(crate) fn of(val: Val, defs: Definitions<'_>) -> Option<ConstantValue> {
        is_sentient_constant(val, defs).then_some(ConstantValue(val))
    }
}

/// `dcc::utils::isConstant<sentient::ConstantOp>` (`Utils/Utils.cpp:423`) — a
/// `sentient.scalar_constant`, or a `uniform.query_map` every one of whose per-core values is one.
pub(super) fn is_sentient_constant(val: Val, defs: Definitions<'_>) -> bool {
    let is_constant_op =
        |op: Option<&Op>| matches!(op, Some(Op::Sentient(ops::Op::ScalarConstant { .. })));
    match defs.of(val) {
        Some(Op::Uniform(uniform::Op::QueryMap { map, .. })) => match defs.of(*map) {
            Some(Op::Uniform(uniform::Op::DefImmutableMapping { pairs, .. })) => pairs
                .iter()
                .all(|(_, value)| is_constant_op(defs.of(*value))),
            _ => false,
        },
        op => is_constant_op(op),
    }
}

/// `dcc::utils::getFirstConstOperandIndex` (`Analyses/Utils.cpp:30`) — ⭐ `None` IS ITS `-1`, and the
/// operand order is [`operands`]'s.
pub(super) fn first_const_operand_index(op: &Op, defs: Definitions<'_>) -> Option<usize> {
    operands(op)
        .into_iter()
        .position(|operand| is_sentient_constant(operand, defs))
}

/// `isa<sentient::AddOp, sentient::SubOp>(op)` — the arm of the chain walk that ends it.
fn is_scalar_add_or_sub(op: &Op) -> bool {
    matches!(
        op,
        Op::Sentient(ops::Op::ScalarAdd { .. } | ops::Op::ScalarSub { .. })
    )
}

/// `dcc::utils::isUpdateMode` (`Analyses/Utils.cpp:65`) at the three arms this pass reaches — the
/// increment is a non-zero constant, or a query map with a non-zero value among its constants.
///
/// ⭐ IT TAKES THE DESCRIPTOR RATHER THAN THE OP, which removes ONE of the two routes to the
/// reference's `llvm_unreachable("unhandled operation")` (`:125`): an op outside its five arms.
/// ⛔ THE SECOND ROUTE SURVIVES AND IS WHAT `_ => false` ANSWERS — every arm falls through to `:125`
/// when the increment is defined by neither a `ConstantOp` nor a `QueryMapOp`, and
/// [`is_mergeable_op_or_chain`] proves only the IMMUTABLE address constant (`:1611-1612`).
/// ⛔ THE `load_and_store` ARM IS THE ONE WITH AN OUT-OF-SCOPE ANALYSIS IN IT
/// (`L3GatherScatterChecker::isIBRWrite`, `:86`) and the chain walk never reaches it:
/// `sentient.load_and_store` is not in its `isa<>` list.
fn is_update_mode(mem_info: &MemoryOpInfo, defs: Definitions<'_>) -> bool {
    match defs.of(mem_info.increment) {
        Some(Op::Sentient(ops::Op::ScalarConstant { value, .. })) => *value != 0,
        Some(Op::Uniform(uniform::Op::QueryMap { map, .. })) => has_non_zero_constants(*map, defs),
        _ => false,
    }
}

/// `dcc::utils::hasNonZeroConstants` (`Analyses/Utils.cpp:50`) — a query map whose values are ALL
/// constants (its own first gate, `:51`) and at least one of which is not zero.
fn has_non_zero_constants(map: Val, defs: Definitions<'_>) -> bool {
    let Some(Op::Uniform(uniform::Op::DefImmutableMapping { pairs, .. })) = defs.of(map) else {
        return false;
    };
    let mut any_non_zero = false;
    for (_, value) in pairs {
        match defs.of(*value) {
            Some(Op::Sentient(ops::Op::ScalarConstant { value, .. })) => {
                any_non_zero |= *value != 0;
            }
            _ => return false,
        }
    }
    any_non_zero
}

/// `*op->getUsers().begin()` — the first op in `scope` that reads any of `of`, regions included.
///
/// ⭐ BLOCK ORDER RATHER THAN MLIR'S USE LIST, AND THE TWO CANNOT DISAGREE HERE:
/// [`is_mergeable_op_or_chain`] only steps to a user having already proved the op has exactly one.
fn first_user<'a>(of: &[Val], scope: &'a [Op]) -> Option<&'a Op> {
    for op in scope {
        if operands(op).iter().any(|read| of.contains(read)) {
            return Some(op);
        }
        for region in regions_ref(op) {
            if let Some(found) = first_user(of, region) {
                return Some(found);
            }
        }
    }
    None
}

/// `op->getUsers()` — every op in `scope` reading any of `of`, regions included, in block order.
///
/// ⭐ [`first_user`]'S SIBLING: the chain walk needs the one, the two units that adjust every reader
/// of a value need them all.
/// ⛔ AN OP READING THE VALUE TWICE APPEARS ONCE HERE and twice in `getUsers()`, which yields one
/// entry per USE — the reference would walk such a user's chain twice and record it twice.
fn users<'a>(of: &[Val], scope: &'a [Op]) -> Vec<&'a Op> {
    let mut found: Vec<&Op> = Vec::new();
    for op in scope {
        if operands(op).iter().any(|read| of.contains(read)) {
            found.push(op);
        }
        for region in regions_ref(op) {
            found.extend(users(of, region));
        }
    }
    found
}

/// `derived_iv->getAttr("element_size")` — the discardable attribute a `sentient.scalar_add` or
/// `scalar_sub` carries, which is what a derived induction variable is.
///
/// ⭐ NOT [`crate::islands::sentient::dialects::element_size`], which ports
/// `dcc::utils::getElementSize` — its arms are the composite transfers and it answers `None` for a
/// scalar add. ⛔ `None` here is the reference's own `hasAttr("element_size") == false` (`:1820`).
fn scalar_element_size(val: Val, block: &[Op]) -> Option<Bits> {
    match defining_op(val, block) {
        Some(Op::Sentient(
            ops::Op::ScalarAdd { element_size, .. } | ops::Op::ScalarSub { element_size, .. },
        )) => *element_size,
        _ => None,
    }
}

/// Replaces: e170_addForOpResultAdjustment
///
/// Inserts `sentient.scalar_add %result, %adjustment` immediately AFTER the loop and moves every
/// other reader of the loop's result onto it, spending one IBUFF entry.
///
/// ⛔ TRAP: `replaceAllUsesExcept` (`:1567-1568`) EXCEPTS THE NEW ADD, which therefore keeps reading
/// the loop result while everything else reads the add — a plain RAUW makes it read itself.
/// ⛔ TRAP: the size is `element_sizes[result_idx + 1]`, slot 0 being the loop iterator's
/// (`:1562-1564`), and this island's `Carried::element_size` already IS that slot.
pub(crate) fn add_for_op_result_adjustment<E: ExpressionEvaluator>(
    scope: &mut Vec<Op>,
    at: usize,
    result_idx: IterArgIndex,
    adjustment_increment: KnownAbsolute<'_>,
    comp: GenericComp,
    evaluator: &mut E,
    sites: &mut OffsetSites<'_>,
    ibuff_space: &mut IbuffSpace,
) -> IbuffSpent {
    // `for_op->getResult(result_idx)` beside `element_sizes[result_idx + 1]` — ⭐ ONE carried entry
    // holds both, and its `None` is the reference's `hasAttr("element_sizes") == false` (`:1559`).
    let Some(Op::Sentient(ops::Op::For { carried, .. })) = scope.get(at) else {
        return IbuffSpent::WithinBudget;
    };
    let Some(entry) = carried.get(result_idx.0 as usize).copied() else {
        return IbuffSpent::WithinBudget;
    };
    // ⭐ THE CONST IS BUILT BEFORE THE INSERTION POINT IS TAKEN (`:1547-1549`, then `:1551-1552`),
    // and building it can insert at the START of this very block (see [`OffsetSites::query_maps`]) —
    // the reference's point is anchored to the op, ours is an index, so the loop is re-found here.
    let new_const =
        evaluator.build_offset_value(adjustment_increment.0, sites, scope, ScalarTy::Index);
    let Some(at) = scope
        .iter()
        .position(|op| results(op).contains(&entry.result))
    else {
        // ⭐ UNREACHABLE, AND A NO-OP RATHER THAN A REFUSAL: the loop was just read at `at` above,
        // and building an offset value only ever inserts.
        return IbuffSpent::WithinBudget;
    };
    let new_add = sites.values.mint();
    scope.insert(
        at + 1,
        Op::Sentient(ops::Op::ScalarAdd {
            lhs: entry.result,
            rhs: new_const,
            result: new_add,
            reg: None,
            // Ensure element_size is set as it is essential for range checks.
            element_size: if ScalarOpComp::of(comp).is_some() {
                entry.element_size
            } else {
                None
            },
            ty: ScalarTy::Index,
        }),
    );
    replace_all_uses_with(scope, entry.result, new_add);
    // The exception set: the new add is the one reader that keeps reading the loop result.
    if let Some(Op::Sentient(ops::Op::ScalarAdd { lhs, .. })) = scope.get_mut(at + 1) {
        *lhs = entry.result;
    }
    ibuff_space.spend_one()
}

/// Replaces: e171_isMergeableOpOrChain
///
/// Whether the chain out of `first_op_in_chain` is a linear run of single-use ops ending in an
/// add/sub with a constant operand, or in an update-mode composite transfer.
///
/// ⛔ TRAP: each transfer's `mutable_addr` must be defined by the PREVIOUS op in the chain
/// (`:1609-1613`), so the address has to keep flowing forward or the chain is not mergeable.
/// ⛔ TRAP: an op with no uses at all passes the first gate and fails the last one (`:1620`) — only
/// the update-mode transfer may end the chain, and it must do so before the walk asks for a user.
/// ⛔ TRAP: a `load_compute_and_send` can never trip the burst/IL gate — [`MemoryOpInfo::of`] records
/// why: that arm sets neither field, so they stay at 1 and 0.
pub(crate) fn is_mergeable_op_or_chain<'a>(
    input_to_chain: &Op,
    first_op_in_chain: &'a Op,
    comp: GenericComp,
    scope: &'a [Op],
) -> bool {
    let regions: [&[Op]; 1] = [scope];
    let defs = Definitions::from_innermost(&regions);
    let mut prev_results = results(input_to_chain);
    let mut current = Some(first_op_in_chain);
    while let Some(op) = current {
        let uses: usize = results(op).iter().map(|val| use_count(*val, scope)).sum();
        // `if (!op->use_empty() && !op->hasOneUse()) break;`
        if uses > 1 {
            break;
        }
        if is_scalar_add_or_sub(op) {
            return first_const_operand_index(op, defs).is_some();
        }
        let Some(mem_info) = MemoryOpInfo::of(op) else {
            return false;
        };
        if ScalarOpComp::of(comp).is_none() {
            return false;
        }
        // TODO: Add support for burst/IL
        if mem_info.burst > Elements(1) || mem_info.il > Elements(0) {
            return false;
        }
        // Verify the immutable_addr is a constant value and the previously analyzed Operation is the
        // mutable_addr of this memory operation.
        if !is_sentient_constant(mem_info.immutable_addr, defs)
            || !prev_results.contains(&mem_info.mutable_addr)
        {
            return false;
        }
        if is_update_mode(&mem_info, defs) {
            return true;
        }
        if uses == 0 {
            return false;
        }
        prev_results = results(op);
        current = first_user(&prev_results, scope);
    }
    false
}

/// Replaces: e172_addToOrReplaceOp
///
/// The new value for a constant a chain op reads: `modifier` alone when it replaces, and `modifier`
/// plus what the constant already holds when it adds to.
///
/// ⛔ THE MODIFIER IS THE ARENA HANDLE, NOT A DECODED [`Evaluation`]: the reference's
/// `const EvaluatedValue &modifier` is the entry [`OperationData::mod_by`] stored, and e365 hands it
/// straight back — decoding it here would be inventing the out-of-scope analysis.
/// ⛔ TRAP: both arms need methods that are OUT OF CAMPAIGN SCOPE and therefore `todo!`s at
/// [`crate::transform::sentient::analyses::OutOfScopeEvaluator`] — the arms are present, and it is
/// the analysis behind them that is not.
pub(crate) fn add_to_or_replace_op<E: ExpressionEvaluator>(
    op: ConstantValue,
    modifier: EvaluatedValue,
    replace_with_mod: Modification,
    evaluator: &mut E,
    sites: &mut OffsetSites<'_>,
    walked: &mut Vec<Op>,
) -> Val {
    match replace_with_mod {
        Modification::Replace => {
            evaluator.build_offset_value_of(modifier, sites, walked, ScalarTy::Index)
        }
        Modification::AddTo => {
            let held = evaluator.evaluate_value_handle(op.0);
            let sum = evaluator.evaluate_sum_handle(modifier, held);
            evaluator.build_offset_value_of(sum, sites, walked, ScalarTy::Index)
        }
    }
}

/// Replaces: e173_hoistCandidateOutOfLoop
///
/// Hoists a derived induction variable out of the loop: its readers move onto the main iter arg, an
/// add of the offset is built BEFORE the loop, and that add becomes the iter arg's initialiser.
///
/// ⛔ TRAP: the ORDER is load-bearing — the element size is read off `derived_iv` before it is erased
/// (`:1820-1824`), and the erase comes before the loop's operand is re-pointed at the new add.
/// ⭐ `main_iv.getArgNumber()` IS BOTH INDICES: as a region argument it is `1 + i`, and as a
/// `for_op_->setOperand` slot it is `1 + i` too, operand 0 being `$bound` — so it is carried `i`.
pub(crate) fn hoist_candidate_out_of_loop(
    scope: &mut Vec<Op>,
    at: usize,
    main_iv: IterArgIndex,
    derived_iv: Val,
    new_add_offset: Val,
    comp: GenericComp,
    values: &mut Values,
) {
    let index = main_iv.0 as usize;
    let Some(Op::Sentient(ops::Op::For { carried, body, .. })) = scope.get_mut(at) else {
        return;
    };
    let Some(entry) = carried.get(index).copied() else {
        return;
    };
    // `derived_iv->getResult(0).replaceAllUsesWith(main_iv)` — the body is the only region the iter
    // argument is visible in, so it is the only region those uses can be in.
    replace_all_uses_with(body, derived_iv, entry.arg);
    // Copy element size as well as it is needed for range checks.
    let element_size = scalar_element_size(derived_iv, body);
    erase_defining_op(body, derived_iv);
    let new_add = values.mint();
    scope.insert(
        at,
        Op::Sentient(ops::Op::ScalarAdd {
            lhs: entry.init,
            rhs: new_add_offset,
            result: new_add,
            reg: None,
            element_size: if ScalarOpComp::of(comp).is_some() {
                element_size
            } else {
                None
            },
            ty: ScalarTy::Index,
        }),
    );
    // `for_op_->setOperand(main_iv_operand_idx, new_add)`, which is carried `index`'s initialiser.
    if let Some(Op::Sentient(ops::Op::For { carried, .. })) = scope.get_mut(at + 1)
        && let Some(entry) = carried.get_mut(index)
    {
        entry.init = new_add;
    }
}

/// THE MERGING INCREMENT AS BOTH THINGS THE CHAIN WALK NEEDS IT AS — the [`EvaluatedValue`] handle
/// [`OperationData`] stores, and the decoded [`Evaluation`] `doesValueExceedLRFRange` reads.
///
/// ⛔ THE PORT NEVER TURNS ONE INTO THE OTHER ITSELF: a handle names an entry in the evaluator's
/// arena, so decoding it is `Analyses/ExpressionEvaluatorUtils`' work — reachable only by ASKING the
/// seam ([`ExpressionEvaluator::evaluation_of`]), which is an out-of-scope `todo!` where stating both
/// flavours costs nothing. The reference has one `const EvaluatedValue &` and both readings of it for
/// free; here the caller states both, which is why they are one parameter rather than two.
#[derive(Debug, Clone, Copy)]
pub(crate) struct MergingIncrement<'a> {
    /// What [`OperationData::mod_by`] keeps.
    pub(crate) handle: EvaluatedValue,
    /// What the LRF range test measures.
    pub(crate) evaluation: &'a Evaluation,
}

/// Replaces: e468_processMergeableChain
///
/// Walks the chain out of `first_op_in_chain` recording how each op absorbs `merging_increment`, and
/// declines the whole chain the moment one of them cannot hold the result.
///
/// ⛔ THE CONSTRUCTOR ARGUMENT ORDER IS THE REFERENCE'S AND IT READS BACKWARDS:
/// `{op, &merging_increment, &getConstant(0)}` fills `mod_` with the INCREMENT and
/// `merging_increment_` with ZERO (`:342-344`, `:1663`, `:1687`).
/// ⛔ THE LRF TEST IS SKIPPED FOR EVERY NON-LX/L0 UNIT, and the add/sub is still recorded — only the
/// memory-op arm refuses outright on `!is_any_of(getComp(), ..)` (`:1691`).
/// ⭐ `Operation *input_to_chain` IS UNREAD IN THE BODY, droppable like e179's `OpBuilder&`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn process_mergeable_chain<A: Arch, E: ExpressionEvaluator>(
    first_op_in_chain: &Op,
    merging_increment: MergingIncrement<'_>,
    ldsti_imm_range: ImmRange,
    comp: GenericComp,
    scale: AddressScale,
    scope: &[Op],
    evaluator: &mut E,
    ops_to_update: &mut Vec<OperationData>,
) -> bool {
    let mut walked = Some(first_op_in_chain);
    while let Some(op) = walked {
        if is_scalar_add_or_sub(op) {
            let Op::Sentient(
                ops::Op::ScalarAdd {
                    result,
                    element_size,
                    ..
                }
                | ops::Op::ScalarSub {
                    result,
                    element_size,
                    ..
                },
            ) = op
            else {
                // [`is_scalar_add_or_sub`] names exactly these two.
                return true;
            };
            if let Some(scalar_comp) = ScalarOpComp::of(comp) {
                let Some(element_size) = *element_size else {
                    todo!(
                        "processMergeableChain: DT_CHECK(op->hasAttr(\"element_size\")) \
                         (ScalarOpMergingAndHoisting.cpp:1655) — {op:?} carries no element size"
                    )
                };
                if does_value_exceed_lrf_range::<A>(
                    merging_increment.evaluation,
                    element_size,
                    scalar_comp,
                    scale,
                ) {
                    return false;
                }
            }
            ops_to_update.push(OperationData {
                op: *result,
                mod_by: merging_increment.handle,
                merging_increment: evaluator.constant(0),
                replace_with_mod: false,
            });
            return true;
        }
        let Some(mem_info) = MemoryOpInfo::of(op) else {
            // The `else` arm — anything that is not one of the three composites ends the walk, and
            // `sentient.load_and_store` is deliberately among them (`:1689`).
            return true;
        };
        let Some(scalar_comp) = ScalarOpComp::of(comp) else {
            return false;
        };
        // `checkImmutableRange` — the lambda whose recorded [`OperationData`] is the one effect it has
        // besides its answer (`:1650-1665`).
        let immutable_addr_ev = evaluator.evaluate_value(mem_info.immutable_addr);
        let new_immutable_ev =
            evaluator.evaluate_sum(&immutable_addr_ev, merging_increment.evaluation);
        if !is_immutable_value_in_range::<A>(
            &new_immutable_ev,
            &immutable_addr_ev,
            mem_info.element_size,
            ldsti_imm_range,
            scalar_comp,
            scale,
        ) {
            return false;
        }
        let Op::Sentient(
            ops::Op::LoadAndSend { result, .. }
            | ops::Op::ReceiveAndStore { result, .. }
            | ops::Op::LoadComputeAndSend { result, .. },
        ) = op
        else {
            // [`MemoryOpInfo::of`] names exactly these three.
            return true;
        };
        ops_to_update.push(OperationData {
            op: *result,
            mod_by: merging_increment.handle,
            merging_increment: evaluator.constant(0),
            replace_with_mod: false,
        });
        if is_update_mode(&mem_info, Definitions::from_innermost(&[scope])) {
            return true;
        }
        walked = first_user(&[*result], scope);
    }
    true
}

/// WHERE ONE RECORDED MODIFICATION LANDS — the operand slot of an add or sub, or a transfer's
/// address pair, decided while the block is still borrowed for reading.
#[derive(Debug, Clone, Copy)]
enum Applied {
    /// `op->setOperand(const_idx, new_const)`.
    Operand(usize),
    /// `getImmutableAddrMutable().assign(new_const)`, and the increment as well when it is not zero.
    Address {
        /// `!isTargetConstant(getIncrement(), 0)`.
        also_increment: bool,
    },
}

/// Replaces: e365_applyOperationData
///
/// Applies every recorded modification: an add or sub gets a new constant operand, and a composite
/// transfer a new immutable address — and the same value as its increment, unless that increment is
/// already the constant zero (`:1740-1793`).
///
/// ⛔ THE SUB NEGATES THE MODIFIER FIRST, AND ONLY WHEN IT IS ADDING TO (`:1757-1760`): its constant
/// is subtracted, so absorbing `+m` means storing `-m` — and a REPLACING sub keeps the modifier as is.
/// ⛔ `llvm_unreachable("Unexpected operation encountered!")` (`:1791`) AND BOTH `DT_CHECK`s ARE NAMED
/// `todo!`s: an op no arm claims is a caller defect, not an input this may answer for.
fn apply_operation_data<E: ExpressionEvaluator>(
    ops_to_update: &[OperationData],
    scope: &mut Vec<Op>,
    comp: GenericComp,
    evaluator: &mut E,
    sites: &mut OffsetSites<'_>,
) {
    for op_data in ops_to_update {
        let (constant, negate_modifier, applied) = {
            let regions: [&[Op]; 1] = [scope.as_slice()];
            let defs = Definitions::from_innermost(&regions);
            let Some(op) = defining_op(op_data.op, scope) else {
                // `op_data.getOp()` is a live pointer in the reference; an op a later entry of the
                // same list already erased has nothing left to rewrite.
                continue;
            };
            let modification = match op {
                Op::Sentient(ops::Op::ScalarAdd { .. } | ops::Op::ScalarSub { .. }) => {
                    let Some(const_idx) = first_const_operand_index(op, defs) else {
                        // `getFirstConstOperandIndex`'s `-1`, which the reference then uses as an
                        // operand index.
                        continue;
                    };
                    let Some(constant) = operands(op).get(const_idx).copied() else {
                        continue;
                    };
                    let negate = matches!(op, Op::Sentient(ops::Op::ScalarSub { .. }))
                        && !op_data.replace_with_mod;
                    (constant, negate, Applied::Operand(const_idx))
                }
                Op::Sentient(
                    ops::Op::LoadAndSend {
                        immutable_addr,
                        increment,
                        ..
                    }
                    | ops::Op::ReceiveAndStore {
                        immutable_addr,
                        increment,
                        ..
                    }
                    | ops::Op::LoadComputeAndSend {
                        immutable_addr,
                        increment,
                        ..
                    },
                ) => {
                    if ScalarOpComp::of(comp).is_none() {
                        todo!(
                            "applyOperationData: DT_CHECK(is_any_of(getComp(), LXLU, LXSU, L0LU, \
                             L0SU)) (ScalarOpMergingAndHoisting.cpp:1766) — {comp:?} recorded a \
                             composite transfer"
                        )
                    }
                    (
                        *immutable_addr,
                        false,
                        Applied::Address {
                            also_increment: !is_target_constant(*increment, 0, defs),
                        },
                    )
                }
                other => todo!(
                    "applyOperationData: llvm_unreachable(\"Unexpected operation encountered!\") \
                     (ScalarOpMergingAndHoisting.cpp:1791) — {other:?} was recorded for update"
                ),
            };
            let (constant, negate, applied) = modification;
            let Some(constant) = ConstantValue::of(constant, defs) else {
                todo!(
                    "addToOrReplaceOp: DT_CHECK_MSG(isConstant<sentient::ConstantOp>(..), \"Expect \
                     op to be constant\") (ScalarOpMergingAndHoisting.cpp:1719) — {constant:?} is \
                     not a constant"
                )
            };
            (constant, negate, applied)
        };
        let modifier = if negate_modifier {
            evaluator.evaluate_multiply_by_const(op_data.mod_by, -1)
        } else {
            op_data.mod_by
        };
        let replace_with_mod = if op_data.replace_with_mod {
            Modification::Replace
        } else {
            Modification::AddTo
        };
        let new_const = add_to_or_replace_op(
            constant,
            modifier,
            replace_with_mod,
            evaluator,
            sites,
            scope,
        );
        let Some(op) = defining_op_mut(scope, op_data.op) else {
            continue;
        };
        match applied {
            Applied::Operand(const_idx) => set_operand(op, const_idx, new_const),
            Applied::Address { also_increment } => {
                if let Op::Sentient(
                    ops::Op::LoadAndSend {
                        immutable_addr,
                        increment,
                        ..
                    }
                    | ops::Op::ReceiveAndStore {
                        immutable_addr,
                        increment,
                        ..
                    }
                    | ops::Op::LoadComputeAndSend {
                        immutable_addr,
                        increment,
                        ..
                    },
                ) = op
                {
                    *immutable_addr = new_const;
                    if also_increment {
                        *increment = new_const;
                    }
                }
            }
        }
    }
}

/// Replaces: e469_hoistForLinearChain
///
/// Hoists a derived induction variable whose chain is a linear run of composite transfers: the FIRST
/// transfer swaps its increment for the derived variable's constant operand, the chain's recorded
/// modifications are applied, and the derived variable then leaves the loop with its old immutable
/// address as the new add's offset.
///
/// ⛔ THE ORDER IS THE PORT: the swap happens BEFORE `applyOperationData`, which is what lets that pass
/// rewrite the immutable address the swap just read (`:2046-2060`).
/// ⛔ `llvm_unreachable("Unexpected operation found!")` (`:2059`) IS A NAMED `todo!`: a first chain op
/// that is not one of the three composites is a caller defect, not an input this may answer for.
#[allow(clippy::too_many_arguments)]
pub(crate) fn hoist_for_linear_chain<E: ExpressionEvaluator>(
    scope: &mut Vec<Op>,
    at: usize,
    main_iv: IterArgIndex,
    derived_iv: Val,
    comp_ops: &[OperationData],
    comp: GenericComp,
    evaluator: &mut E,
    sites: &mut OffsetSites<'_>,
) {
    let Some(Op::Sentient(ops::Op::For { body, .. })) = scope.get_mut(at) else {
        return;
    };
    let Some(derived_iv_op) = defining_op(derived_iv, body) else {
        return;
    };
    let Some(const_operand_idx) = first_const_operand_index(
        derived_iv_op,
        Definitions::from_innermost(&[body.as_slice()]),
    ) else {
        // `getFirstConstOperandIndex`'s `-1`, which the reference then uses as an operand index.
        return;
    };
    let Some(derived_iv_operand) = operands(derived_iv_op).get(const_operand_idx).copied() else {
        return;
    };
    let Some(first_op) = comp_ops.first() else {
        // `comp_ops.front()` on an empty list — the caller (e612) only ever hands over a chain it
        // built, so there is nothing here for an empty one to mean.
        return;
    };
    let first_op_immutable_addr = match defining_op_mut(body, first_op.op) {
        Some(Op::Sentient(
            ops::Op::LoadAndSend {
                immutable_addr,
                increment,
                ..
            }
            | ops::Op::ReceiveAndStore {
                immutable_addr,
                increment,
                ..
            }
            | ops::Op::LoadComputeAndSend {
                immutable_addr,
                increment,
                ..
            },
        )) => {
            let immutable_addr = *immutable_addr;
            *increment = derived_iv_operand;
            immutable_addr
        }
        first => todo!(
            "hoistForLinearChain: llvm_unreachable(\"Unexpected operation found!\") \
             (ScalarOpMergingAndHoisting.cpp:2059) — {first:?} heads the chain"
        ),
    };
    apply_operation_data(comp_ops, body, comp, evaluator, sites);
    hoist_candidate_out_of_loop(
        scope,
        at,
        main_iv,
        derived_iv,
        first_op_immutable_addr,
        comp,
        sites.values,
    );
}

/// `Value::getDefiningOp()` in a block being rewritten — [`defining_op`], mutably.
///
/// ⭐ REGIONS INCLUDED, as [`defining_op`] and [`users`] both are: an op a `sentient.if` holds is one
/// `getUsers()` reports and one e365 then has to rewrite.
fn defining_op_mut(scope: &mut [Op], val: Val) -> Option<&mut Op> {
    for op in scope.iter_mut() {
        if results(op).contains(&val) {
            return Some(op);
        }
        for region in regions_mut(op) {
            if let Some(found) = defining_op_mut(region, val) {
                return Some(found);
            }
        }
    }
    None
}

/// Replaces: e529_processForOpResult
///
/// Whether every reader of one loop result can absorb the hoist's adjustment, recording each reader's
/// update in `ops_to_update` as it goes.
///
/// ⛔ TRAP: `processMergeableChain`'S ANSWER IS DISCARDED HERE (`:1497`) and checked by e530 (`:1871`)
/// — a chain e171 accepted but e468 then declined still leaves its partial records in the list and
/// this still answers true. Not a transcription slip: the two call sites differ in the reference.
/// ⭐ `result.use_empty()` (`:1489`) IS THE EMPTY WALK: no reader, nothing to absorb, true.
#[allow(clippy::too_many_arguments)]
pub(crate) fn process_for_op_result<A: Arch, E: ExpressionEvaluator>(
    scope: &[Op],
    at: usize,
    result_idx: IterArgIndex,
    adjustment_increment: MergingIncrement<'_>,
    ldsti_imm_range: ImmRange,
    comp: GenericComp,
    scale: AddressScale,
    evaluator: &mut E,
    ops_to_update: &mut Vec<OperationData>,
) -> bool {
    let Some(for_op @ Op::Sentient(ops::Op::For { carried, .. })) = scope.get(at) else {
        return true;
    };
    let Some(result) = carried.get(result_idx.0 as usize).map(|entry| entry.result) else {
        return true;
    };
    for user in users(&[result], scope) {
        if !is_mergeable_op_or_chain(for_op, user, comp, scope) {
            return false;
        }
        process_mergeable_chain::<A, E>(
            user,
            adjustment_increment,
            ldsti_imm_range,
            comp,
            scale,
            scope,
            evaluator,
            ops_to_update,
        );
    }
    true
}

/// Replaces: e530_processForDerivedIVElimination
///
/// Eliminates a derived induction variable outright: every reader absorbs its constant operand as a
/// merging increment, and the derived variable's readers move onto the main iter arg before it is
/// erased.
///
/// ⛔ THE SUB FORM IS `B - c` AND ONLY WITH `c` SECOND (`:1861`): as operand 0 the constant is the
/// MINUEND, so `derived_iv = c - B` is no offset of `B` and is declined. Otherwise the modifier is
/// negated, which is what makes the two forms one add.
/// ⛔ THE EVALUATION HAPPENS BEFORE THAT GATE (`:1858-1862`) and the analysis memoises, so asking is
/// not free of effect. ⛔ AND BEFORE ANY REWRITE: one declining reader leaves the loop untouched.
#[allow(clippy::too_many_arguments)]
pub(crate) fn process_for_derived_iv_elimination<A: Arch, E: ExpressionEvaluator>(
    scope: &mut Vec<Op>,
    at: usize,
    main_iv: IterArgIndex,
    derived_iv: Val,
    ldsti_imm_range: ImmRange,
    comp: GenericComp,
    scale: AddressScale,
    evaluator: &mut E,
    sites: &mut OffsetSites<'_>,
    hoists: &mut HoistCount,
) -> bool {
    // `is_any_of(getComp(), LXLU, LXSU, L0LU, L0SU)` — DerivedIV elimination is LX/L0 only (`:1845`).
    if ScalarOpComp::of(comp).is_none() {
        return false;
    }
    let Some(Op::Sentient(ops::Op::For { carried, body, .. })) = scope.get_mut(at) else {
        return false;
    };
    let Some(main_iv_arg) = carried.get(main_iv.0 as usize).map(|entry| entry.arg) else {
        return false;
    };
    let regions: [&[Op]; 1] = [body.as_slice()];
    let Some(derived_iv_op) = defining_op(derived_iv, body) else {
        return false;
    };
    let Some(const_operand_idx) =
        first_const_operand_index(derived_iv_op, Definitions::from_innermost(&regions))
    else {
        // `getFirstConstOperandIndex`'s `-1`, which the reference then uses as an operand index.
        return false;
    };
    let Some(constant) = operands(derived_iv_op).get(const_operand_idx).copied() else {
        return false;
    };
    let is_sub = matches!(derived_iv_op, Op::Sentient(ops::Op::ScalarSub { .. }));
    let modifier = evaluator.evaluate_value(constant);
    let modifier_handle = evaluator.evaluate_value_handle(constant);
    let (modifier, modifier_handle) = if is_sub {
        if const_operand_idx == 0 {
            return false;
        }
        (
            evaluator.multiply_by_const(&modifier, -1),
            evaluator.evaluate_multiply_by_const(modifier_handle, -1),
        )
    } else {
        (modifier, modifier_handle)
    };
    let increment = MergingIncrement {
        handle: modifier_handle,
        evaluation: &modifier,
    };
    let mut ops_to_update: Vec<OperationData> = Vec::new();
    for user in users(&[derived_iv], body) {
        if !is_mergeable_op_or_chain(derived_iv_op, user, comp, body) {
            return false;
        }
        if !process_mergeable_chain::<A, E>(
            user,
            increment,
            ldsti_imm_range,
            comp,
            scale,
            body,
            evaluator,
            &mut ops_to_update,
        ) {
            return false;
        }
    }
    apply_operation_data(&ops_to_update, body, comp, evaluator, sites);
    replace_all_uses_with(body, derived_iv, main_iv_arg);
    erase_defining_op(body, derived_iv);
    hoists.0 += 1;
    true
}

/// `-dcc-hoist-without-absorbing-ops` (`:114-119`) — *"Enable scalar op hoisting even if there are no
/// absorbing ops for the adjustment add op (innermost loop will always attempt to hoist)"*, whose
/// `llvm::cl::init(true)` is this value.
const HOIST_WITHOUT_ABSORBING_OPS: bool = true;

/// `is_inner_most_` (`:516`, `:564`) AT THE ONE ARM THAT READS IT (`:1524`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Innermost {
    /// The candidate loop encloses no other loop.
    Yes,
    /// It does.
    No,
}

/// Replaces: e578_adjustCandidateForOpResult
///
/// Absorbs the post-hoist adjustment into the users of the candidate loop's result, or — where they
/// cannot take it and an IBuff entry is free — adds the adjustment behind the loop (`:1514-1532`).
///
/// ⭐ `main_iv.getArgNumber() - 1` IS DISCHARGED BY THE TYPE: [`IterArgIndex`] already numbers the
/// carried entries, whose region argument is one past the induction variable.
/// ⚠️ TRAP: A NON-ABSOLUTE INCREMENT IS REPORTED, NOT ASSERTED — the reference's `DT_CHECK` at
/// `:1546` becomes `false`, the same answer its two callers give a candidate they cannot adjust.
/// ⭐ e170's `Overrun` IS UNREACHABLE FROM HERE: the `ibuff_space_ <= 0` guard above it is this
/// function's own.
#[allow(clippy::too_many_arguments)]
pub(crate) fn adjust_candidate_for_op_result<A: Arch, E: ExpressionEvaluator>(
    scope: &mut Vec<Op>,
    at: usize,
    main_iv: IterArgIndex,
    adjustment_increment: MergingIncrement<'_>,
    innermost: Innermost,
    ldsti_imm_range: ImmRange,
    comp: GenericComp,
    scale: AddressScale,
    evaluator: &mut E,
    sites: &mut OffsetSites<'_>,
    ibuff_space: &mut IbuffSpace,
) -> bool {
    let mut ops_to_update: Vec<OperationData> = Vec::new();
    if process_for_op_result::<A, E>(
        scope,
        at,
        main_iv,
        adjustment_increment,
        ldsti_imm_range,
        comp,
        scale,
        evaluator,
        &mut ops_to_update,
    ) {
        if !ops_to_update.is_empty() {
            apply_operation_data(&ops_to_update, scope, comp, evaluator, sites);
        }
        return true;
    }
    if (matches!(innermost, Innermost::No) && !HOIST_WITHOUT_ABSORBING_OPS) || ibuff_space.0 <= 0 {
        return false;
    }
    let Some(known) = KnownAbsolute::of(adjustment_increment.evaluation) else {
        return false;
    };
    let _ = add_for_op_result_adjustment(
        scope,
        at,
        main_iv,
        known,
        comp,
        evaluator,
        sites,
        ibuff_space,
    );
    true
}

/// `*op->getUsers().begin()` AND THE REFUSAL AT `:1945` AS ONE STEP — the reference takes the first
/// user wherever it is and then declines any chain whose next op is not a direct child of the
/// candidate loop's body.
///
/// ⭐ `None` IS THAT REFUSAL: a step with no user at all is its other reading, and `hasOneUse` has
/// already ruled that out before every step the walk makes.
fn first_user_in_body<'a>(of: &[Val], body: &'a [Op]) -> Option<&'a Op> {
    let found = first_user(of, body)?;
    body.iter().find(|op| core::ptr::eq(*op, found))
}

/// Replaces: e612_processForLinearChain
///
/// Hoists a derived induction variable whose users are a linear run of composite transfers from it to
/// the loop's own yield, each absorbing the offset the one before it passed on (`:1893-2021`).
///
/// ⛔ THE FIRST TRANSFER IS THE ONE THAT ABSORBS THE ADD: it alone refuses a burst or interleaved
/// group and it alone is recorded with `replace_with_mod`, and from there the running offset is the
/// NEGATED immutable address it just approved, not the derived variable's constant.
/// ⛔ THE INTERLEAVED-GROUP GATE IS `> 1` HERE AND `> 0` IN e171 (`:1969` against `:1607`) —
/// transcribed, not harmonised.
/// ⛔ THE CONSTANT IS EVALUATED BEFORE THE SUB GATE (`:1913-1919`) and the analysis memoises.
#[allow(clippy::too_many_arguments)]
pub(crate) fn process_for_linear_chain<A: Arch, E: ExpressionEvaluator>(
    scope: &mut Vec<Op>,
    at: usize,
    main_iv: IterArgIndex,
    derived_iv: Val,
    innermost: Innermost,
    ldsti_imm_range: ImmRange,
    comp: GenericComp,
    scale: AddressScale,
    evaluator: &mut E,
    sites: &mut OffsetSites<'_>,
    ibuff_space: &mut IbuffSpace,
    hoists: &mut HoistCount,
) -> bool {
    // `is_any_of(getComp(), LXLU, LXSU, L0LU, L0SU)` — linear-chain hoisting is LX/L0 only (`:1894`).
    let Some(scalar_comp) = ScalarOpComp::of(comp) else {
        return false;
    };
    let Some(Op::Sentient(ops::Op::For { carried, body, .. })) = scope.get(at) else {
        return false;
    };
    let Some(main_iv_arg) = carried.get(main_iv.0 as usize).map(|entry| entry.arg) else {
        return false;
    };
    // `!derived_iv->hasOneUse() || !main_iv.hasOneUse()` (`:1903`).
    if use_count(derived_iv, body) != 1 || use_count(main_iv_arg, body) != 1 {
        return false;
    }
    let regions: [&[Op]; 1] = [body.as_slice()];
    let defs = Definitions::from_innermost(&regions);
    let Some(derived_iv_op) = defining_op(derived_iv, body) else {
        return false;
    };
    let Some(const_operand_idx) = first_const_operand_index(derived_iv_op, defs) else {
        // `getFirstConstOperandIndex`'s `-1`, which the reference then uses as an operand index.
        return false;
    };
    let Some(constant) = operands(derived_iv_op).get(const_operand_idx).copied() else {
        return false;
    };
    let mut offset = evaluator.evaluate_value(constant);
    let mut offset_handle = evaluator.evaluate_value_handle(constant);
    if matches!(derived_iv_op, Op::Sentient(ops::Op::ScalarSub { .. })) {
        // `derived_iv = B - c` only, and then as `B + (c * -1)` (`:1916-1919`).
        if const_operand_idx == 0 {
            return false;
        }
        offset = evaluator.multiply_by_const(&offset, -1);
        offset_handle = evaluator.evaluate_multiply_by_const(offset_handle, -1);
    }
    let mut first_comp_op_found = false;
    let mut comp_ops: Vec<OperationData> = Vec::new();
    let mut prev_results = results(derived_iv_op);
    let mut current = first_user_in_body(&prev_results, body);
    loop {
        let Some(op) = current else {
            return false;
        };
        if let Op::Sentient(ops::Op::Yield { results: yielded }) = op {
            // `op == main_yield` — the chain ends here only if the yield hands the op before it back
            // at the main IV's own position (`:1934-1937`).
            if yielded.get(main_iv.0 as usize) == prev_results.first() {
                break;
            }
            return false;
        }
        // `!op->hasOneUse()` (`:1938`).
        if results(op)
            .iter()
            .map(|val| use_count(*val, body))
            .sum::<usize>()
            != 1
        {
            return false;
        }
        let Some(mem_info) = MemoryOpInfo::of(op) else {
            // The `else` arm: anything but the three composites ends the chain unhoisted (`:2004`).
            return false;
        };
        let Op::Sentient(
            ops::Op::LoadAndSend { result, .. }
            | ops::Op::ReceiveAndStore { result, .. }
            | ops::Op::LoadComputeAndSend { result, .. },
        ) = op
        else {
            // [`MemoryOpInfo::of`] names exactly these three.
            return false;
        };
        if !is_sentient_constant(mem_info.immutable_addr, defs) {
            return false;
        }
        let immutable_addr_ev = evaluator.evaluate_value(mem_info.immutable_addr);
        // `mem_info.mutable_addr_.getDefiningOp() != prev_op` — the address has to keep flowing.
        if !prev_results.contains(&mem_info.mutable_addr) {
            return false;
        }
        // "Memory operations with non-zero increment attributes would have been handled in
        // processForDerivedIVElimination()."
        if is_update_mode(&mem_info, defs) {
            return false;
        }
        if first_comp_op_found {
            let new_immutable_ev = evaluator.evaluate_sum(&immutable_addr_ev, &offset);
            if !is_immutable_value_in_range::<A>(
                &new_immutable_ev,
                &immutable_addr_ev,
                mem_info.element_size,
                ldsti_imm_range,
                scalar_comp,
                scale,
            ) {
                return false;
            }
            comp_ops.push(OperationData {
                op: *result,
                mod_by: offset_handle,
                merging_increment: evaluator.constant(0),
                replace_with_mod: false,
            });
        } else {
            // TODO: Add support for burst/IL — this transfer's increment is what the hoisted add
            // swaps into, which an unrolled transfer's own increments would contradict (`:1969`).
            if mem_info.burst > Elements(1) || mem_info.il > Elements(1) {
                return false;
            }
            if !is_immutable_value_in_range::<A>(
                &offset,
                &immutable_addr_ev,
                mem_info.element_size,
                ldsti_imm_range,
                scalar_comp,
                scale,
            ) {
                return false;
            }
            comp_ops.push(OperationData {
                op: *result,
                mod_by: offset_handle,
                merging_increment: evaluator.constant(0),
                replace_with_mod: true,
            });
            // This transfer's immutable address, negated, is what the rest of the chain absorbs.
            let immutable_handle = evaluator.evaluate_value_handle(mem_info.immutable_addr);
            offset_handle = evaluator.evaluate_multiply_by_const(immutable_handle, -1);
            offset = evaluator.multiply_by_const(&immutable_addr_ev, -1);
            first_comp_op_found = true;
        }
        prev_results = results(op);
        current = first_user_in_body(&prev_results, body);
    }
    if comp_ops.is_empty() {
        return false;
    }
    if !adjust_candidate_for_op_result::<A, E>(
        scope,
        at,
        main_iv,
        MergingIncrement {
            handle: offset_handle,
            evaluation: &offset,
        },
        innermost,
        ldsti_imm_range,
        comp,
        scale,
        evaluator,
        sites,
        ibuff_space,
    ) {
        return false;
    }
    hoist_for_linear_chain(
        scope, at, main_iv, derived_iv, &comp_ops, comp, evaluator, sites,
    );
    hoists.0 += 1;
    true
}

/// Replaces: e613_processForGenericHoisting
///
/// Hoists a derived induction variable whose loop yield is fed by an add or sub that can absorb the
/// adjustment, adjusting every other reader of that op and of the main iter arg by its opposite
/// (`:2075-2155`).
///
/// ⛔ THE SIGNS ARE NOT SYMMETRIC: the op feeding the yield takes `+adjustment` — or `-adjustment`
/// when it is a sub reading the constant FIRST — and every other reader takes `-adjustment`.
/// ⛔ A SUB FEEDING THE YIELD WITH THE CONSTANT SECOND FALLS THROUGH TO THE `isa<AddOp>` TEST AND IS
/// DECLINED (`:2109-2117`): the two arms are not exhaustive over add/sub, and that is the reference.
/// ⛔ THE CONSTANT IS EVALUATED BEFORE THE SUB GATE (`:2085-2092`) and the analysis memoises.
#[allow(clippy::too_many_arguments)]
pub(crate) fn process_for_generic_hoisting<A: Arch, E: ExpressionEvaluator>(
    scope: &mut Vec<Op>,
    at: usize,
    main_iv: IterArgIndex,
    derived_iv: Val,
    innermost: Innermost,
    ldsti_imm_range: ImmRange,
    comp: GenericComp,
    scale: AddressScale,
    evaluator: &mut E,
    sites: &mut OffsetSites<'_>,
    ibuff_space: &mut IbuffSpace,
    hoists: &mut HoistCount,
) -> bool {
    let Some(Op::Sentient(ops::Op::For {
        iv, carried, body, ..
    })) = scope.get(at)
    else {
        return false;
    };
    let for_iv = *iv;
    let Some(main_iv_arg) = carried.get(main_iv.0 as usize).map(|entry| entry.arg) else {
        return false;
    };
    let regions: [&[Op]; 1] = [body.as_slice()];
    let defs = Definitions::from_innermost(&regions);
    let Some(derived_iv_op) = defining_op(derived_iv, body) else {
        return false;
    };
    let Some(const_operand_idx) = first_const_operand_index(derived_iv_op, defs) else {
        // `getFirstConstOperandIndex`'s `-1`, which the reference then uses as an operand index.
        return false;
    };
    let Some(derived_iv_operand) = operands(derived_iv_op).get(const_operand_idx).copied() else {
        return false;
    };
    let mut adjustment = evaluator.evaluate_value(derived_iv_operand);
    let mut adjustment_handle = evaluator.evaluate_value_handle(derived_iv_operand);
    if matches!(derived_iv_op, Op::Sentient(ops::Op::ScalarSub { .. })) {
        // `derived_iv = B - c` only, and then as `B + (c * -1)` (`:2088-2092`).
        if const_operand_idx == 0 {
            return false;
        }
        adjustment = evaluator.multiply_by_const(&adjustment, -1);
        adjustment_handle = evaluator.evaluate_multiply_by_const(adjustment_handle, -1);
    }
    // `main_yield->getOperand(main_iv_idx).getDefiningOp()` — the op the loop hands back at the main
    // IV's position is the one that has to absorb the adjustment (`:2096-2102`).
    let Some((main_yield, yielded)) = body.iter().find_map(|op| match op {
        Op::Sentient(ops::Op::Yield { results }) => Some((op, results)),
        _ => None,
    }) else {
        return false;
    };
    let Some(yielded_val) = yielded.get(main_iv.0 as usize).copied() else {
        return false;
    };
    let Some(op_feeding_yield) = defining_op(yielded_val, body) else {
        return false;
    };
    let Some(const_idx) = first_const_operand_index(op_feeding_yield, defs) else {
        return false;
    };
    let mut ops_to_update: Vec<OperationData> = Vec::new();
    if matches!(op_feeding_yield, Op::Sentient(ops::Op::ScalarSub { .. })) && const_idx == 0 {
        let mod_by = evaluator.evaluate_multiply_by_const(adjustment_handle, -1);
        ops_to_update.push(OperationData {
            op: yielded_val,
            mod_by,
            merging_increment: evaluator.constant(0),
            replace_with_mod: false,
        });
    } else if matches!(op_feeding_yield, Op::Sentient(ops::Op::ScalarAdd { .. })) {
        ops_to_update.push(OperationData {
            op: yielded_val,
            mod_by: adjustment_handle,
            merging_increment: evaluator.constant(0),
            replace_with_mod: false,
        });
    } else {
        return false;
    }
    // Every OTHER reader of the op feeding the yield has to be mergeable, and takes the opposite
    // adjustment (`:2118-2128`).
    for user in users(&[yielded_val], body) {
        if core::ptr::eq(user, main_yield) {
            continue;
        }
        if !is_mergeable_op_or_chain(op_feeding_yield, user, comp, body) {
            return false;
        }
        let Some(user_val) = results(user).first().copied() else {
            // Unreachable: every arm e171 accepts binds one result.
            continue;
        };
        let mod_by = evaluator.evaluate_multiply_by_const(adjustment_handle, -1);
        ops_to_update.push(OperationData {
            op: user_val,
            mod_by,
            merging_increment: evaluator.constant(0),
            replace_with_mod: false,
        });
    }
    // And every other use of the main IV, once the derived one is gone (`:2130-2140`).
    for user in users(&[main_iv_arg], body) {
        if core::ptr::eq(user, derived_iv_op) {
            continue;
        }
        let Some(user_val) = results(user).first().copied() else {
            todo!(
                "processForGenericHoisting: applyOperationData's \
                 llvm_unreachable (ScalarOpMergingAndHoisting.cpp:1793) — {user:?} reads the main IV \
                 and binds no result to modify"
            )
        };
        let mod_by = evaluator.evaluate_multiply_by_const(adjustment_handle, -1);
        ops_to_update.push(OperationData {
            op: user_val,
            mod_by,
            merging_increment: evaluator.constant(0),
            replace_with_mod: false,
        });
    }
    let negated = evaluator.multiply_by_const(&adjustment, -1);
    let negated_handle = evaluator.evaluate_multiply_by_const(adjustment_handle, -1);
    if !adjust_candidate_for_op_result::<A, E>(
        scope,
        at,
        main_iv,
        MergingIncrement {
            handle: negated_handle,
            evaluation: &negated,
        },
        innermost,
        ldsti_imm_range,
        comp,
        scale,
        evaluator,
        sites,
        ibuff_space,
    ) {
        return false;
    }
    // ⭐ THE LOOP IS RE-FOUND BY ITS INDUCTION VARIABLE, for the reason e170 states: the adjustment
    // above builds a constant, and building one can insert at the START of this block.
    let Some(at) = scope
        .iter()
        .position(|op| matches!(op, Op::Sentient(ops::Op::For { iv, .. }) if *iv == for_iv))
    else {
        return false;
    };
    let Some(Op::Sentient(ops::Op::For { body, .. })) = scope.get_mut(at) else {
        return false;
    };
    apply_operation_data(&ops_to_update, body, comp, evaluator, sites);
    hoist_candidate_out_of_loop(
        scope,
        at,
        main_iv,
        derived_iv,
        derived_iv_operand,
        comp,
        sites.values,
    );
    hoists.0 += 1;
    true
}

/// `MaxHoists`, `cl::init(-1)` — "-1 indicates no maximum" (`:121-125`).
///
/// ⭐ THE CAP IS INERT AS SHIPPED, AND `None` IS WHY RATHER THAN A BIG NUMBER: the option is
/// `cl::opt<unsigned>`, so its `-1` is `UINT_MAX` and the guard `MaxHoists != -1` is FALSE (`:2189`).
pub(super) const MAX_HOISTS: Option<HoistCount> = None;

/// ONE `iter_arg.getUsers()` ENTRY, READ BEFORE ANY REWRITE — [`users`] cannot report a user's PARENT,
/// and the candidate gate declines a user a `sentient.if` holds (`:2213`).
struct MainIvUser {
    /// The add or sub's result, which is the candidate's identity for the reason
    /// [`crate::transform::sentient::ForRef`] gives. `None` is
    /// `!isa<AddOp, SubOp>(user) || getFirstConstOperandIndex(user) == -1` (`:2202-2203`).
    candidate: Option<Val>,
    /// `isa<sentient::IfOp>(user->getParentOp())`.
    parent_is_if: bool,
}

/// `iter_arg.getUsers()` in block order, each entry carrying whether a `sentient.if` is its parent.
fn main_iv_users(
    iter_arg: Val,
    block: &[Op],
    parent_is_if: bool,
    defs: Definitions<'_>,
    found: &mut Vec<MainIvUser>,
) {
    for op in block {
        if operands(op).contains(&iter_arg) {
            let eligible =
                is_scalar_add_or_sub(op) && first_const_operand_index(op, defs).is_some();
            found.push(MainIvUser {
                candidate: eligible.then(|| results(op).first().copied()).flatten(),
                parent_is_if,
            });
        }
        let is_if = matches!(op, Op::Sentient(ops::Op::If { .. }));
        for region in regions_ref(op) {
            main_iv_users(iter_arg, region, is_if, defs, found);
        }
    }
}

/// Replaces: e635_runScalarOpHoisting
///
/// One candidate loop, iter arg by iter arg: collect the adds and subs of a constant that read it and
/// offer each in turn to derived-IV elimination, then to linear-chain hoisting, then to generic
/// hoisting, re-examining the same iter arg after every hoist that took (`:2184-2245`).
///
/// ⛔ ONE INELIGIBLE USER DISQUALIFIES THE WHOLE ITER ARG (`hoisting_candidates.clear(); break;`,
/// `:2205-2215`) — the candidates are not filtered, they are dropped.
/// ⛔ `idx` ONLY ADVANCES WHEN NOTHING HOISTED (`:2244`): a hoist creates new opportunities on the
/// same iter arg, and the number of iter args is re-read because the hoist rewrote the loop.
/// ⛔ AND THE LOOP MOVES. `hoistCandidateOutOfLoop` inserts the new add AHEAD of it, so `at` is
/// re-found by the induction variable — MLIR's `for_op_` is a pointer, this island's is a position.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_scalar_op_hoisting<A: Arch, E: ExpressionEvaluator>(
    scope: &mut Vec<Op>,
    at: usize,
    innermost: Innermost,
    ldsti_imm_range: ImmRange,
    comp: GenericComp,
    scale: AddressScale,
    evaluator: &mut E,
    sites: &mut OffsetSites<'_>,
    ibuff_space: &mut IbuffSpace,
    hoists: &mut HoistCount,
) {
    let Some(Op::Sentient(ops::Op::For { iv, .. })) = scope.get(at) else {
        return;
    };
    let for_iv = *iv;
    let mut at = at;
    let mut idx = 0;
    loop {
        // `num_hoists_executed_per_unit >= MaxHoists` (`:2189`).
        if MAX_HOISTS.is_some_and(|max| *hoists >= max) {
            return;
        }
        let Some(Op::Sentient(ops::Op::For { carried, body, .. })) = scope.get(at) else {
            return;
        };
        // `idx < for_op_->getNumRegionIterArgs()`, re-read at every pass.
        let Some(entry) = carried.get(idx as usize) else {
            return;
        };
        let main_iv = IterArgIndex(idx);
        let iter_arg = entry.arg;
        let regions: [&[Op]; 2] = [body.as_slice(), scope.as_slice()];
        let defs = Definitions::from_innermost(&regions);
        let mut found = Vec::new();
        main_iv_users(iter_arg, body, false, defs, &mut found);
        let mut candidates = Vec::new();
        for user in found {
            match user.candidate {
                Some(candidate) if !user.parent_is_if => candidates.push(candidate),
                _ => {
                    candidates.clear();
                    break;
                }
            }
        }
        // `isa<sentient::YieldOp>(candidate_user)` — a candidate the loop yields would need its own
        // adjustment put back, so it is passed over rather than dropped (`:2223-2231`).
        candidates.retain(|candidate| {
            !users(&[*candidate], body)
                .iter()
                .any(|user| matches!(user, Op::Sentient(ops::Op::Yield { .. })))
        });

        let mut reanalyze_candidates = false;
        for candidate in candidates {
            if process_for_derived_iv_elimination::<A, E>(
                scope,
                at,
                main_iv,
                candidate,
                ldsti_imm_range,
                comp,
                scale,
                evaluator,
                sites,
                hoists,
            ) || process_for_linear_chain::<A, E>(
                scope,
                at,
                main_iv,
                candidate,
                innermost,
                ldsti_imm_range,
                comp,
                scale,
                evaluator,
                sites,
                ibuff_space,
                hoists,
            ) || process_for_generic_hoisting::<A, E>(
                scope,
                at,
                main_iv,
                candidate,
                innermost,
                ldsti_imm_range,
                comp,
                scale,
                evaluator,
                sites,
                ibuff_space,
                hoists,
            ) {
                reanalyze_candidates = true;
                break;
            }
        }

        if reanalyze_candidates {
            let Some(moved) = scope.iter().position(
                |op| matches!(op, Op::Sentient(ops::Op::For { iv, .. }) if *iv == for_iv),
            ) else {
                return;
            };
            at = moved;
        } else {
            idx += 1;
        }
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::arch::Dd2;
    use crate::islands::dataflow_ir::link::SendEnd;
    use crate::islands::sentient::dialects::sentient::{Carried, Reg, RegType, ShuffleMode};
    use crate::transform::sentient::analyses::{Offsets, ScalarOffset};

    /// AN EVALUATOR THAT ANSWERS WHAT THE TEST SAYS. ⛔ The analysis behind
    /// [`ExpressionEvaluator`] is out of campaign scope, so a test STATES its answers rather than
    /// deriving them — what is under test is the EFFECT a port has given an answer.
    struct StatedEvaluator {
        /// What `build_offset_value` materialises the offset as.
        offset: Val,
        /// Every `evaluate_sum` this evaluator was asked for, as `(lhs, rhs)`.
        sums: Vec<(Evaluation, Evaluation)>,
        /// Every `evaluate_sum_handle`, as `(lhs, rhs)` — where a NEGATED modifier shows.
        handle_sums: Vec<(EvaluatedValue, EvaluatedValue)>,
        /// Every handle `build_offset_value_of` was asked to materialise, in order.
        built: Vec<EvaluatedValue>,
    }

    impl ExpressionEvaluator for StatedEvaluator {
        fn evaluate_value(&mut self, _value: Val) -> Evaluation {
            absolute(ScalarOffset(4))
        }

        fn evaluate_sum(&mut self, lhs: &Evaluation, rhs: &Evaluation) -> Evaluation {
            self.sums.push((lhs.clone(), rhs.clone()));
            absolute(ScalarOffset(20))
        }

        fn build_offset_value(
            &mut self,
            _evaluation: &Evaluation,
            _sites: &mut OffsetSites<'_>,
            _walked: &mut Vec<Op>,
            _ty: ScalarTy,
        ) -> Val {
            self.offset
        }

        fn constant(&mut self, value: i64) -> EvaluatedValue {
            // A stated arena entry: `getConstant(0)` is the only literal e468 asks for.
            EvaluatedValue(value.unsigned_abs() as u32)
        }

        fn evaluate_value_handle(&mut self, _value: Val) -> EvaluatedValue {
            // The handle flavour of the same stated answer — what e530 hands the chain walk.
            EvaluatedValue(7)
        }

        fn build_offset_value_of(
            &mut self,
            immutable: EvaluatedValue,
            _sites: &mut OffsetSites<'_>,
            _walked: &mut Vec<Op>,
            _ty: ScalarTy,
        ) -> Val {
            self.built.push(immutable);
            self.offset
        }

        fn evaluate_sum_handle(
            &mut self,
            lhs: EvaluatedValue,
            rhs: EvaluatedValue,
        ) -> EvaluatedValue {
            self.handle_sums.push((lhs, rhs));
            // A stated arena entry, distinct from every other this double hands out.
            EvaluatedValue(200)
        }

        fn multiply_by_const(&mut self, ev: &Evaluation, by: i64) -> Evaluation {
            absolute(ScalarOffset(
                ev.all_unit_offset().map_or(0, |offset| offset.0) * by,
            ))
        }

        fn evaluate_multiply_by_const(&mut self, _ev: EvaluatedValue, _by: i64) -> EvaluatedValue {
            // A stated arena entry, distinct from the one `evaluate_value_handle` names.
            EvaluatedValue(70)
        }
    }

    /// The stated evaluator, with nothing recorded yet.
    fn stated_evaluator() -> StatedEvaluator {
        StatedEvaluator {
            offset: Val(9),
            sums: Vec::new(),
            handle_sums: Vec::new(),
            built: Vec::new(),
        }
    }

    /// THE LDSTI IMM WINDOW A TEST STATES in place of e366's own table lookup: every stated offset
    /// below scales to 16 or less, so no fixture is refused for exceeding it.
    fn imm_window() -> ImmRange {
        ImmRange {
            min: -128,
            max: 127,
        }
    }

    fn absolute(offset: ScalarOffset) -> Evaluation {
        Evaluation {
            known_absolute: true,
            base: None,
            offsets: Offsets::AllUnit(offset),
        }
    }

    /// A minter that has already issued `issued` values, so a fixture's hand-written [`Val`]s and a
    /// created op's cannot collide.
    fn values_after(issued: u32) -> Values {
        let mut values = Values::default();
        for _ in 0..issued {
            values.mint();
        }
        values
    }

    fn add(lhs: Val, rhs: Val, result: Val, element_size: Option<Bits>) -> Op {
        Op::Sentient(ops::Op::ScalarAdd {
            lhs,
            rhs,
            result,
            reg: None,
            element_size,
            ty: ScalarTy::Index,
        })
    }

    fn constant(value: i64, result: Val) -> Op {
        Op::Sentient(ops::Op::ScalarConstant {
            value,
            result,
            reg_locale: RegType::Imm,
            ty: ScalarTy::Index,
            is_symbol: false,
        })
    }

    fn carried(init: Val, arg: Val, result: Val, element_size: Option<Bits>) -> Carried {
        Carried {
            init,
            arg,
            result,
            reg: Reg {
                locale: RegType::Lar,
                index: None,
            },
            program_header: false,
            element_size,
        }
    }

    fn for_op(entries: Vec<Carried>, body: Vec<Op>) -> Op {
        Op::Sentient(ops::Op::For {
            iv: Val(0),
            bound: Val(1),
            bound_reg: None,
            carried: entries,
            dbg_name: None,
            body,
        })
    }

    fn load_and_send(mutable_addr: Val, immutable_addr: Val, increment: Val) -> Op {
        Op::Sentient(ops::Op::LoadAndSend {
            mutable_addr,
            immutable_addr,
            increment,
            consumer: SendEnd::to_self(Val(30)),
            result: Val(31),
            extent: ops::Extent {
                total_elements: Elements(64),
                element_size: Bits(32),
                chunk_size: Elements(1),
                chunk_stride: Elements(1),
                burst_size: Elements(0),
            },
            interleaved_group: Elements(0),
            rotate_val: None,
            dir: None,
            shuffle_mode: ShuffleMode::NoShuffle,
            reg: Reg {
                locale: RegType::Lar,
                index: None,
            },
            dbg_name: None,
        })
    }

    /// e170 — the add lands AFTER the loop, every other reader of the result moves onto it, the
    /// carried slot's element size is copied, and the add itself keeps reading the loop result.
    #[test]
    fn a_for_op_result_adjustment_inserts_the_add_after_the_loop_and_keeps_its_own_operand() {
        let mut scope = vec![
            for_op(
                vec![carried(Val(2), Val(3), Val(4), Some(Bits(8)))],
                Vec::new(),
            ),
            add(Val(4), Val(5), Val(6), None),
        ];
        let mut consts = Vec::new();
        let mut values = values_after(10);
        let mut sites = OffsetSites {
            consts: &mut consts,
            query_maps: None,
            values: &mut values,
        };
        let mut evaluator = stated_evaluator();
        let increment = absolute(ScalarOffset(16));
        let mut ibuff = IbuffSpace(2);
        assert_eq!(
            add_for_op_result_adjustment(
                &mut scope,
                0,
                IterArgIndex(0),
                KnownAbsolute::of(&increment).expect("the fixture states it absolute"),
                GenericComp::Lxlu,
                &mut evaluator,
                &mut sites,
                &mut ibuff,
            ),
            IbuffSpent::WithinBudget
        );
        assert_eq!(scope[1], add(Val(4), Val(9), Val(10), Some(Bits(8))));
        assert_eq!(scope[2], add(Val(10), Val(5), Val(6), None));
        assert_eq!(ibuff, IbuffSpace(1));
    }

    /// e170 — a `buildOffsetValue` that inserts ahead of the loop does not carry the add off it: the
    /// reference's insertion point is anchored to the op (`:1551-1552`) and this one is re-found.
    #[test]
    fn a_for_op_result_adjustment_re_finds_the_loop_after_the_const_is_built() {
        /// An evaluator that inserts at the START of the walked block, which is what
        /// [`OffsetSites::query_maps`] being `None` means the reference's const builder does.
        struct PrependingEvaluator;

        impl ExpressionEvaluator for PrependingEvaluator {
            fn evaluate_value(&mut self, _value: Val) -> Evaluation {
                absolute(ScalarOffset(4))
            }

            fn evaluate_sum(&mut self, _lhs: &Evaluation, _rhs: &Evaluation) -> Evaluation {
                absolute(ScalarOffset(20))
            }

            fn build_offset_value(
                &mut self,
                _evaluation: &Evaluation,
                _sites: &mut OffsetSites<'_>,
                walked: &mut Vec<Op>,
                _ty: ScalarTy,
            ) -> Val {
                walked.insert(0, constant(0, Val(9)));
                Val(9)
            }
        }

        let mut scope = vec![
            for_op(
                vec![carried(Val(2), Val(3), Val(4), Some(Bits(8)))],
                Vec::new(),
            ),
            add(Val(4), Val(5), Val(6), None),
        ];
        let mut consts = Vec::new();
        let mut values = values_after(10);
        let mut sites = OffsetSites {
            consts: &mut consts,
            query_maps: None,
            values: &mut values,
        };
        let increment = absolute(ScalarOffset(16));
        let mut ibuff = IbuffSpace(2);
        assert_eq!(
            add_for_op_result_adjustment(
                &mut scope,
                0,
                IterArgIndex(0),
                KnownAbsolute::of(&increment).expect("the fixture states it absolute"),
                GenericComp::Lxlu,
                &mut PrependingEvaluator,
                &mut sites,
                &mut ibuff,
            ),
            IbuffSpent::WithinBudget
        );
        assert_eq!(scope[0], constant(0, Val(9)));
        // The loop moved to 1, so the add belongs at 2 — index 1 is where a position taken before the
        // const was built would have put it, which is BEFORE the loop.
        assert!(matches!(scope[1], Op::Sentient(ops::Op::For { .. })));
        assert_eq!(scope[2], add(Val(4), Val(9), Val(10), Some(Bits(8))));
        assert_eq!(scope[3], add(Val(10), Val(5), Val(6), None));
    }

    /// e171 — the vendor's own shape: a chain whose one composite transfer takes its mutable address
    /// from the previous op, reads a constant immutable address and updates by a non-zero constant.
    #[test]
    fn a_chain_ending_in_an_update_mode_transfer_is_mergeable() {
        let scope = vec![
            add(Val(20), Val(21), Val(1), None),
            constant(64, Val(2)),
            constant(64, Val(3)),
            load_and_send(Val(1), Val(2), Val(3)),
        ];
        assert!(is_mergeable_op_or_chain(
            &scope[0],
            &scope[3],
            GenericComp::Lxlu,
            &scope
        ));
    }

    /// ⛔ THE NEGATIVE THE `isConstant` GATE AT `:1738-1741` IS THERE FOR: a transfer whose immutable
    /// address is computed rather than constant ends no mergeable chain.
    #[test]
    fn a_transfer_with_a_computed_immutable_address_is_not_mergeable() {
        let scope = vec![
            add(Val(20), Val(21), Val(1), None),
            add(Val(22), Val(23), Val(2), None),
            constant(64, Val(3)),
            load_and_send(Val(1), Val(2), Val(3)),
        ];
        assert!(!is_mergeable_op_or_chain(
            &scope[0],
            &scope[3],
            GenericComp::Lxlu,
            &scope
        ));
    }

    /// e172 — `AddTo` folds the modifier into the handle the constant ITSELF evaluates to, and
    /// `Replace` materialises the modifier alone.
    #[test]
    fn add_to_sums_with_the_constants_own_evaluation_and_replace_does_not() {
        let block = vec![constant(64, Val(2))];
        let regions: [&[Op]; 1] = [&block];
        let defs = Definitions::from_innermost(&regions);
        let op = ConstantValue::of(Val(2), defs).expect("the fixture states it constant");
        let mut consts = Vec::new();
        let mut values = values_after(10);
        let mut sites = OffsetSites {
            consts: &mut consts,
            query_maps: None,
            values: &mut values,
        };
        let mut walked = Vec::new();
        let mut evaluator = stated_evaluator();
        let modifier = EvaluatedValue(16);
        assert_eq!(
            add_to_or_replace_op(
                op,
                modifier,
                Modification::Replace,
                &mut evaluator,
                &mut sites,
                &mut walked
            ),
            Val(9)
        );
        assert!(evaluator.handle_sums.is_empty());
        assert_eq!(evaluator.built, vec![modifier]);
        assert_eq!(
            add_to_or_replace_op(
                op,
                modifier,
                Modification::AddTo,
                &mut evaluator,
                &mut sites,
                &mut walked
            ),
            Val(9)
        );
        // `EvaluatedValue(7)` is the constant's own handle, and the sum is what gets materialised.
        assert_eq!(evaluator.handle_sums, vec![(modifier, EvaluatedValue(7))]);
        assert_eq!(evaluator.built, vec![modifier, EvaluatedValue(200)]);
    }

    /// e365 — THE NEGATION TRAP: an add absorbs the modifier as it stands, and a sub absorbing one
    /// stores its OPPOSITE, because a sub's constant is subtracted rather than added.
    #[test]
    fn a_sub_absorbing_a_modifier_stores_its_opposite_and_an_add_does_not() {
        let mut scope = vec![
            constant(4, Val(1)),
            add(Val(20), Val(1), Val(2), Some(Bits(8))),
            sub(Val(21), Val(1), Val(3), Some(Bits(8))),
        ];
        let recorded = |op: Val| OperationData {
            op,
            mod_by: EvaluatedValue(5),
            merging_increment: EvaluatedValue(0),
            replace_with_mod: false,
        };
        let mut consts = Vec::new();
        let mut values = values_after(10);
        let mut evaluator = stated_evaluator();
        apply_operation_data(
            &[recorded(Val(2)), recorded(Val(3))],
            &mut scope,
            GenericComp::Lxlu,
            &mut evaluator,
            &mut sites_of(&mut consts, &mut values),
        );
        assert_eq!(scope[1], add(Val(20), Val(9), Val(2), Some(Bits(8))));
        assert_eq!(scope[2], sub(Val(21), Val(9), Val(3), Some(Bits(8))));
        // `5` as it stands for the add, and the stated `5 * -1` for the sub.
        assert_eq!(
            evaluator.handle_sums,
            vec![
                (EvaluatedValue(5), EvaluatedValue(7)),
                (EvaluatedValue(70), EvaluatedValue(7)),
            ]
        );
    }

    /// e468 — the TRAP: on a unit that is neither LX nor L0 the add still records the increment and
    /// the chain stops there, while the very same non-LX/L0 comp makes a composite transfer refuse.
    #[test]
    fn a_non_lx_unit_records_the_add_and_still_refuses_the_transfer() {
        let scope = vec![
            add(Val(20), Val(21), Val(1), Some(Bits(32))),
            load_and_send(Val(1), Val(2), Val(3)),
        ];
        let increment = absolute(ScalarOffset(16));
        let stated = MergingIncrement {
            handle: EvaluatedValue(7),
            evaluation: &increment,
        };
        let mut evaluator = stated_evaluator();
        let mut ops_to_update = Vec::new();
        assert!(process_mergeable_chain::<Dd2, _>(
            &scope[0],
            stated,
            imm_window(),
            GenericComp::Pt,
            AddressScale::ONE,
            &scope,
            &mut evaluator,
            &mut ops_to_update,
        ));
        assert_eq!(
            ops_to_update,
            vec![OperationData {
                op: Val(1),
                mod_by: EvaluatedValue(7),
                merging_increment: EvaluatedValue(0),
                replace_with_mod: false,
            }]
        );
        let mut refused = Vec::new();
        assert!(!process_mergeable_chain::<Dd2, _>(
            &scope[1],
            stated,
            imm_window(),
            GenericComp::Pt,
            AddressScale::ONE,
            &scope,
            &mut evaluator,
            &mut refused,
        ));
        assert!(refused.is_empty());
    }

    /// e469 — ⛔ THE ORDER IS THE PORT: the first transfer's increment takes the derived IV's constant
    /// operand, and only THEN does e365 rewrite the immutable address that swap just read — so the
    /// increment is non-zero by the time e365 decides to write it too.
    #[test]
    fn hoisting_a_linear_chain_swaps_the_increment_before_the_address_is_rewritten() {
        let mut scope = vec![for_op(
            vec![carried(Val(2), Val(3), Val(4), None)],
            vec![
                constant(4, Val(5)),
                add(Val(3), Val(5), Val(6), Some(Bits(16))),
                constant(64, Val(7)),
                constant(0, Val(8)),
                load_and_send(Val(6), Val(7), Val(8)),
            ],
        )];
        let comp_ops = vec![OperationData {
            op: Val(31),
            mod_by: EvaluatedValue(1),
            merging_increment: EvaluatedValue(0),
            replace_with_mod: false,
        }];
        let mut consts = Vec::new();
        let mut values = values_after(40);
        hoist_for_linear_chain(
            &mut scope,
            0,
            IterArgIndex(0),
            Val(6),
            &comp_ops,
            GenericComp::Lxlu,
            &mut stated_evaluator(),
            &mut sites_of(&mut consts, &mut values),
        );
        // The new add reads the transfer's OLD immutable address and initialises the iter arg.
        assert_eq!(scope[0], add(Val(2), Val(7), Val(40), Some(Bits(16))));
        assert_eq!(
            scope[1],
            for_op(
                vec![carried(Val(40), Val(3), Val(4), None)],
                vec![
                    constant(4, Val(5)),
                    constant(64, Val(7)),
                    constant(0, Val(8)),
                    load_and_send(Val(3), Val(9), Val(9)),
                ]
            )
        );
    }

    /// e173 — the derived IV's readers move onto the main iter arg, the derived add is erased, its
    /// element size travels to a new add BEFORE the loop, and that add becomes the arg's initialiser.
    #[test]
    fn hoisting_a_derived_iv_re_points_the_iter_arg_at_an_add_before_the_loop() {
        let mut scope = vec![for_op(
            vec![carried(Val(2), Val(3), Val(4), None)],
            vec![
                add(Val(3), Val(5), Val(6), Some(Bits(16))),
                add(Val(6), Val(7), Val(8), None),
            ],
        )];
        let mut values = values_after(10);
        hoist_candidate_out_of_loop(
            &mut scope,
            0,
            IterArgIndex(0),
            Val(6),
            Val(9),
            GenericComp::Lxlu,
            &mut values,
        );
        assert_eq!(scope[0], add(Val(2), Val(9), Val(10), Some(Bits(16))));
        assert_eq!(
            scope[1],
            for_op(
                vec![carried(Val(10), Val(3), Val(4), None)],
                vec![add(Val(3), Val(7), Val(8), None)]
            )
        );
    }
    /// e529 — the TRAP, both halves: a reader whose chain absorbs the adjustment is recorded, and a
    /// reader whose chain e468 DECLINES still leaves this answering true with nothing recorded.
    #[test]
    fn every_reader_of_a_loop_result_is_recorded_and_a_declined_chain_is_still_accepted() {
        let scope = vec![
            constant(4, Val(5)),
            for_op(vec![carried(Val(2), Val(3), Val(4), None)], Vec::new()),
            add(Val(4), Val(5), Val(6), Some(Bits(32))),
        ];
        let within = absolute(ScalarOffset(16));
        let mut evaluator = stated_evaluator();
        let mut ops_to_update = Vec::new();
        assert!(process_for_op_result::<Dd2, _>(
            &scope,
            1,
            IterArgIndex(0),
            MergingIncrement {
                handle: EvaluatedValue(7),
                evaluation: &within,
            },
            imm_window(),
            GenericComp::Pt,
            AddressScale::ONE,
            &mut evaluator,
            &mut ops_to_update,
        ));
        assert_eq!(
            ops_to_update,
            vec![OperationData {
                op: Val(6),
                mod_by: EvaluatedValue(7),
                merging_increment: EvaluatedValue(0),
                replace_with_mod: false,
            }]
        );

        // The same reader on an LX unit, with an increment no LRF can hold: e468 refuses it.
        let beyond = absolute(ScalarOffset(i64::from(i32::MAX)));
        let mut discarded = Vec::new();
        assert!(process_for_op_result::<Dd2, _>(
            &scope,
            1,
            IterArgIndex(0),
            MergingIncrement {
                handle: EvaluatedValue(7),
                evaluation: &beyond,
            },
            imm_window(),
            GenericComp::Lxlu,
            AddressScale::ONE,
            &mut evaluator,
            &mut discarded,
        ));
        assert!(discarded.is_empty(), "the refusal is what went unrecorded");
    }

    /// One loop whose body derives `%6 = %main_iv + 4` and one reader of it that can absorb a
    /// constant — the shape e530 eliminates.
    fn loop_with_a_derived_iv() -> Vec<Op> {
        vec![for_op(
            vec![carried(Val(2), Val(3), Val(4), None)],
            vec![
                constant(4, Val(5)),
                add(Val(3), Val(5), Val(6), Some(Bits(16))),
                add(Val(6), Val(5), Val(8), Some(Bits(32))),
            ],
        )]
    }

    /// e530 — DerivedIV elimination is LX/L0 only, so on any other unit the loop is left exactly as
    /// it was and no hoist is banked.
    #[test]
    fn a_non_lx_unit_eliminates_no_derived_iv() {
        let mut scope = loop_with_a_derived_iv();
        let untouched = scope.clone();
        let mut consts = Vec::new();
        let mut values = values_after(40);
        let mut hoists = HoistCount(0);
        assert!(!process_for_derived_iv_elimination::<Dd2, _>(
            &mut scope,
            0,
            IterArgIndex(0),
            Val(6),
            imm_window(),
            GenericComp::Pt,
            AddressScale::ONE,
            &mut stated_evaluator(),
            &mut sites_of(&mut consts, &mut values),
            &mut hoists,
        ));
        assert_eq!(scope, untouched);
        assert_eq!(hoists, HoistCount(0));
    }

    /// e530 — with its one reader absorbing the modifier the derived IV goes outright: the reader's
    /// constant operand becomes the sum e365 built, the reader moves onto the main iter arg, and the
    /// derived add is erased.
    #[test]
    fn eliminating_a_derived_iv_absorbs_the_modifier_into_its_reader_and_erases_the_add() {
        let mut scope = loop_with_a_derived_iv();
        let mut consts = Vec::new();
        let mut values = values_after(40);
        let mut hoists = HoistCount(0);
        assert!(process_for_derived_iv_elimination::<Dd2, _>(
            &mut scope,
            0,
            IterArgIndex(0),
            Val(6),
            imm_window(),
            GenericComp::Lxlu,
            AddressScale::ONE,
            &mut stated_evaluator(),
            &mut sites_of(&mut consts, &mut values),
            &mut hoists,
        ));
        assert_eq!(
            scope,
            vec![for_op(
                vec![carried(Val(2), Val(3), Val(4), None)],
                vec![
                    constant(4, Val(5)),
                    add(Val(3), Val(9), Val(8), Some(Bits(32))),
                ]
            )]
        );
        assert_eq!(hoists, HoistCount(1));
    }
    /// 578/656 — the one reader adds a value that is not a constant, so nothing can absorb the
    /// adjustment: with an IBuff entry free the add goes in behind the loop, and with none the
    /// candidate is declined and the scope is left alone.
    #[test]
    fn e578_adds_the_adjustment_after_the_loop_or_declines_when_there_is_no_ibuff() {
        let candidate = || {
            vec![
                for_op(
                    vec![carried(Val(2), Val(3), Val(4), Some(Bits(8)))],
                    Vec::new(),
                ),
                add(Val(4), Val(5), Val(6), Some(Bits(32))),
            ]
        };
        let increment = absolute(ScalarOffset(16));
        let adjustment = || MergingIncrement {
            handle: EvaluatedValue(7),
            evaluation: &increment,
        };
        let mut consts = Vec::new();
        let mut values = values_after(10);
        let mut sites = OffsetSites {
            consts: &mut consts,
            query_maps: None,
            values: &mut values,
        };
        let mut evaluator = stated_evaluator();

        let mut scope = candidate();
        let mut ibuff = IbuffSpace(2);
        assert!(adjust_candidate_for_op_result::<Dd2, _>(
            &mut scope,
            0,
            IterArgIndex(0),
            adjustment(),
            Innermost::Yes,
            imm_window(),
            GenericComp::Lxlu,
            AddressScale::ONE,
            &mut evaluator,
            &mut sites,
            &mut ibuff,
        ));
        assert_eq!(scope[1], add(Val(4), Val(9), Val(10), Some(Bits(8))));
        assert_eq!(scope[2], add(Val(10), Val(5), Val(6), Some(Bits(32))));
        assert_eq!(ibuff, IbuffSpace(1));

        let mut declined = candidate();
        let mut none_left = IbuffSpace(0);
        assert!(!adjust_candidate_for_op_result::<Dd2, _>(
            &mut declined,
            0,
            IterArgIndex(0),
            adjustment(),
            Innermost::Yes,
            imm_window(),
            GenericComp::Lxlu,
            AddressScale::ONE,
            &mut evaluator,
            &mut sites,
            &mut none_left,
        ));
        assert_eq!(declined, candidate());
    }
    fn sub(lhs: Val, rhs: Val, result: Val, element_size: Option<Bits>) -> Op {
        Op::Sentient(ops::Op::ScalarSub {
            lhs,
            rhs,
            result,
            reg: None,
            element_size,
            ty: ScalarTy::Index,
        })
    }

    /// The one loop both level-5 drivers are asked about: `%5` is the derived IV, `%6` is what the
    /// yield hands back, and `chain` is what sits between them.
    fn loop_with(chain: Vec<Op>, yielded: Vec<Val>) -> Vec<Op> {
        let mut body = vec![
            constant(4, Val(10)),
            add(Val(3), Val(10), Val(5), Some(Bits(8))),
            constant(6, Val(11)),
            constant(0, Val(12)),
        ];
        body.extend(chain);
        body.push(Op::Sentient(ops::Op::Yield { results: yielded }));
        vec![for_op(
            vec![carried(Val(2), Val(3), Val(4), Some(Bits(8)))],
            body,
        )]
    }

    fn sites_of<'a>(consts: &'a mut Vec<Op>, values: &'a mut Values) -> OffsetSites<'a> {
        OffsetSites {
            consts,
            query_maps: None,
            values,
        }
    }

    /// e612 — a one-transfer chain clears every gate: the derived IV feeds the transfer's mutable
    /// address and the transfer feeds the yield at the main IV's own slot, so the FIRST transfer has
    /// its address REPLACED by the offset and the IV leaves the loop as an add of the old address.
    #[test]
    fn a_linear_chain_of_one_transfer_replaces_its_address_and_leaves_the_loop() {
        let mut scope = loop_with(vec![load_and_send(Val(5), Val(11), Val(12))], vec![Val(31)]);
        let mut consts = Vec::new();
        let mut values = values_after(40);
        let mut hoists = HoistCount(0);
        assert!(process_for_linear_chain::<Dd2, _>(
            &mut scope,
            0,
            IterArgIndex(0),
            Val(5),
            Innermost::Yes,
            imm_window(),
            GenericComp::Lxlu,
            AddressScale::ONE,
            &mut stated_evaluator(),
            &mut sites_of(&mut consts, &mut values),
            &mut IbuffSpace(4),
            &mut hoists,
        ));
        assert_eq!(scope[0], add(Val(2), Val(11), Val(40), Some(Bits(8))));
        assert_eq!(
            scope[1],
            for_op(
                vec![carried(Val(40), Val(3), Val(4), Some(Bits(8)))],
                vec![
                    constant(4, Val(10)),
                    constant(6, Val(11)),
                    constant(0, Val(12)),
                    load_and_send(Val(3), Val(9), Val(9)),
                    Op::Sentient(ops::Op::Yield {
                        results: vec![Val(31)]
                    }),
                ]
            )
        );
        assert_eq!(hoists, HoistCount(1));
    }

    /// ⛔ `:1945`'S REFUSAL: the derived IV's one user is inside a NESTED loop, so the walk declines
    /// the candidate rather than following the chain out of the body it may rewrite.
    #[test]
    fn a_derived_iv_used_inside_a_nested_loop_is_declined() {
        let mut scope = loop_with(
            vec![for_op(
                Vec::new(),
                vec![load_and_send(Val(5), Val(11), Val(12))],
            )],
            vec![Val(11)],
        );
        let untouched = scope.clone();
        let mut consts = Vec::new();
        let mut values = values_after(40);
        let mut hoists = HoistCount(0);
        assert!(!process_for_linear_chain::<Dd2, _>(
            &mut scope,
            0,
            IterArgIndex(0),
            Val(5),
            Innermost::Yes,
            imm_window(),
            GenericComp::Lxlu,
            AddressScale::ONE,
            &mut stated_evaluator(),
            &mut sites_of(&mut consts, &mut values),
            &mut IbuffSpace(4),
            &mut hoists,
        ));
        assert_eq!(scope, untouched);
        assert_eq!(hoists, HoistCount(0));
    }

    /// e613 — the add feeding the yield absorbs the adjustment, the loop result needs none, and the
    /// derived IV then leaves the loop as an add of its OWN constant operand rather than an address.
    #[test]
    fn generic_hoisting_absorbs_the_adjustment_into_the_op_feeding_the_yield() {
        let mut scope = loop_with(
            vec![add(Val(5), Val(11), Val(6), Some(Bits(8)))],
            vec![Val(6)],
        );
        let mut consts = Vec::new();
        let mut values = values_after(40);
        let mut hoists = HoistCount(0);
        assert!(process_for_generic_hoisting::<Dd2, _>(
            &mut scope,
            0,
            IterArgIndex(0),
            Val(5),
            Innermost::Yes,
            imm_window(),
            GenericComp::Lxlu,
            AddressScale::ONE,
            &mut stated_evaluator(),
            &mut sites_of(&mut consts, &mut values),
            &mut IbuffSpace(4),
            &mut hoists,
        ));
        assert_eq!(scope[0], add(Val(2), Val(10), Val(40), Some(Bits(8))));
        assert_eq!(
            scope[1],
            for_op(
                vec![carried(Val(40), Val(3), Val(4), Some(Bits(8)))],
                vec![
                    constant(4, Val(10)),
                    constant(6, Val(11)),
                    constant(0, Val(12)),
                    add(Val(3), Val(9), Val(6), Some(Bits(8))),
                    Op::Sentient(ops::Op::Yield {
                        results: vec![Val(6)]
                    }),
                ]
            )
        );
        assert_eq!(hoists, HoistCount(1));
    }

    /// ⛔ THE ASYMMETRY e613's TWO ARMS LEAVE: a sub feeding the yield is absorbable only with the
    /// constant FIRST, and with it second the reference falls through to `isa<AddOp>` and declines.
    #[test]
    fn a_sub_feeding_the_yield_with_the_constant_second_is_declined() {
        let mut scope = loop_with(
            vec![sub(Val(5), Val(11), Val(6), Some(Bits(8)))],
            vec![Val(6)],
        );
        let untouched = scope.clone();
        let mut consts = Vec::new();
        let mut values = values_after(40);
        let mut hoists = HoistCount(0);
        assert!(!process_for_generic_hoisting::<Dd2, _>(
            &mut scope,
            0,
            IterArgIndex(0),
            Val(5),
            Innermost::Yes,
            imm_window(),
            GenericComp::Lxlu,
            AddressScale::ONE,
            &mut stated_evaluator(),
            &mut sites_of(&mut consts, &mut values),
            &mut IbuffSpace(4),
            &mut hoists,
        ));
        assert_eq!(scope, untouched);
        assert_eq!(hoists, HoistCount(0));
    }
    /// e635 — the driver takes the loop's one iter arg, finds the add of a constant that reads it and
    /// offers that add to derived-IV elimination FIRST, which takes it: the add is gone and its reader
    /// reads the iter arg. The second pass over the same iter arg then finds no candidate and stops.
    ///
    /// ⭐ THE SHAPE IS THE DISPATCH: only elimination erases the candidate outright, so this result
    /// proves the candidate found was `%6` on iter arg 0 and that the first of the three ran.
    #[test]
    fn e635_offers_each_iter_arg_candidate_to_derived_iv_elimination_first() {
        let mut scope = loop_with_a_derived_iv();
        let mut consts = Vec::new();
        let mut values = values_after(40);
        let mut hoists = HoistCount(0);
        run_scalar_op_hoisting::<Dd2, _>(
            &mut scope,
            0,
            Innermost::Yes,
            imm_window(),
            GenericComp::Lxlu,
            AddressScale::ONE,
            &mut stated_evaluator(),
            &mut sites_of(&mut consts, &mut values),
            &mut IbuffSpace(4),
            &mut hoists,
        );
        assert_eq!(
            scope,
            vec![for_op(
                vec![carried(Val(2), Val(3), Val(4), None)],
                vec![
                    constant(4, Val(5)),
                    add(Val(3), Val(9), Val(8), Some(Bits(32))),
                ]
            )]
        );
        assert_eq!(hoists, HoistCount(1));
    }
}
