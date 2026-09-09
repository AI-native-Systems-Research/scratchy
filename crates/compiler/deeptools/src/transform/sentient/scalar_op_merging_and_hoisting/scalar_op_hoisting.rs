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
// ⛔ NOTHING CALLS THESE FOUR LEAVES YET — every caller is a later level in a different batch
// (`e529`, `e530`, `e578`, `e612`, `e613`) and so is the pass entry `e635_runScalarOpHoisting`. CI
// runs clippy with `-D warnings`, so without this the batch fails its own gate.
// ⭐ REMOVE THIS WITH `e635_runScalarOpHoisting`: an unused item here is a real defect at that point.

use super::{MemoryOpInfo, ScalarOpComp};
use crate::arch::Elements;
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
    Evaluation, ExpressionEvaluator, OffsetSites, ScalarOffset,
};

/// `ibuff_space_` — how many instruction-buffer entries are left for the ops this pass creates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct IbuffSpace(pub i32);

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
fn is_sentient_constant(val: Val, defs: Definitions<'_>) -> bool {
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
fn first_const_operand_index(op: &Op, defs: Definitions<'_>) -> Option<usize> {
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

/// `dcc::utils::isTargetConstant` (`Analyses/Utils.cpp:156`) at the arms this pass reaches — a
/// `sentient.scalar_constant` holding `target`, or a query map whose constants all do.
///
/// ⛔ A NON-CONSTANT VALUE IN THE MAP IS SKIPPED, NOT REJECTED (`:169-178`), so a map of entirely
/// non-constant values answers TRUE. ⭐ A value with no definition here is the reference's
/// `isa<BlockArgument>` arm (`:158`) and answers false, as does a mapping this island cannot enumerate
/// — the same treatment [`has_non_zero_constants`] gives it.
fn is_target_constant(val: Val, target: ScalarOffset, defs: Definitions<'_>) -> bool {
    let holds_target = |op: Option<&Op>| match op {
        Some(Op::Sentient(ops::Op::ScalarConstant { value, .. })) => Some(*value == target.0),
        _ => None,
    };
    match defs.of(val) {
        Some(Op::Uniform(uniform::Op::QueryMap { map, .. })) => {
            let Some(Op::Uniform(uniform::Op::DefImmutableMapping { pairs, .. })) = defs.of(*map)
            else {
                return false;
            };
            pairs
                .iter()
                .all(|(_, value)| holds_target(defs.of(*value)).unwrap_or(true))
        }
        op => holds_target(op).unwrap_or(false),
    }
}

/// [`defining_op`]'s mutable twin — the op binding `val`, regions included, so a rewritten operand
/// lands on the op wherever in the walked block it sits.
fn defining_op_mut<'a>(scope: &'a mut [Op], val: Val) -> Option<&'a mut Op> {
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

/// WHERE A REWRITTEN CONSTANT GOES — the two shapes [`apply_operation_data`]'s five arms collapse to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Rewritten {
    /// An add's or a sub's constant operand slot, in [`operands`]'s order.
    Operand(usize),
    /// A composite transfer's `immutable_addr`, and whether its `increment` takes the value too.
    Address { update_mode: bool },
}

/// `OperationData` (`:340-368`) — one op a hoist has decided to modify, and by how much.
///
/// ⭐ THE OP IS NAMED BY THE VALUE IT BINDS rather than by a pointer: each of the five arms
/// [`apply_operation_data`] dispatches on binds exactly one result, and a [`Val`] survives the
/// insertions that building a new constant performs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct OperationData {
    /// `op_` — the op to modify, by the value it binds.
    pub(crate) op: Val,
    /// `mod_` — the amount to modify by.
    pub(crate) modifier: Evaluation,
    /// `merging_increment_` — the merging increment AT this op. ⛔ [`apply_operation_data`] never
    /// reads it; `processMergeableChain` (`:1645`) is what records and re-reads it.
    pub(crate) merging_increment: Evaluation,
    /// `replace_with_mod_` — ⛔ default `AddTo` (`:367`), the three-argument constructor's value.
    pub(crate) replace_with_mod: Modification,
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
/// ⛔ TRAP: the `AddTo` arm needs `ExpressionEvaluator::evaluateSum`, which is OUT OF CAMPAIGN SCOPE
/// and therefore a `todo!` at [`crate::transform::sentient::analyses::OutOfScopeEvaluator`] — the arm is present, and it
/// is the analysis behind it that is not.
pub(crate) fn add_to_or_replace_op<E: ExpressionEvaluator>(
    op: ConstantValue,
    modifier: &Evaluation,
    replace_with_mod: Modification,
    evaluator: &mut E,
    sites: &mut OffsetSites<'_>,
    walked: &mut Vec<Op>,
) -> Val {
    match replace_with_mod {
        Modification::Replace => {
            evaluator.build_offset_value(modifier, sites, walked, ScalarTy::Index)
        }
        Modification::AddTo => {
            let held = evaluator.evaluate_value(op.0);
            let sum = evaluator.evaluate_sum(modifier, &held);
            evaluator.build_offset_value(&sum, sites, walked, ScalarTy::Index)
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

/// Replaces: e365_applyOperationData
///
/// Rewrites each listed op's constant — an add/sub's first constant operand, a composite transfer's
/// immutable address — to the modifier, or to the modifier plus what that constant already held.
///
/// ⛔ TRAP: a `scalar_sub`'s modifier is NEGATED before it is added (`:1757`) — subtracting more is
/// adding less — but NOT when it replaces.
/// ⛔ TRAP: a transfer whose increment is not already the constant 0 has that increment assigned THE
/// SAME new value as its immutable address (`:1772-1773`), which is one value read by two operands.
pub(crate) fn apply_operation_data<E: ExpressionEvaluator>(
    ops_to_update: &[OperationData],
    comp: GenericComp,
    evaluator: &mut E,
    sites: &mut OffsetSites<'_>,
    walked: &mut Vec<Op>,
) {
    for op_data in ops_to_update {
        // What is being rewritten, decided before anything is built: building a constant inserts into
        // this very block, and only the values named here survive that.
        let Some((constant, negate, rewritten)) = ({
            let regions: [&[Op]; 1] = [walked];
            let defs = Definitions::from_innermost(&regions);
            // ⭐ THE MISSING OP IS THE REFERENCE'S `llvm_unreachable("Unexpected operation
            // encountered!")` (`:1795`), reported by leaving the op alone.
            let Some(op) = defining_op(op_data.op, walked) else {
                continue;
            };
            if is_scalar_add_or_sub(op) {
                // ⛔ `None` IS `getFirstConstOperandIndex() == -1`, which the reference feeds straight
                // to `getOperand(-1)`; here it leaves the op alone.
                let Some(index) = first_const_operand_index(op, defs) else {
                    continue;
                };
                let negate = matches!(op, Op::Sentient(ops::Op::ScalarSub { .. }));
                operands(op)
                    .get(index)
                    .copied()
                    .and_then(|constant| ConstantValue::of(constant, defs))
                    .map(|constant| (constant, negate, Rewritten::Operand(index)))
            } else {
                // `DT_CHECK(is_any_of(getComp(), LXLU, LXSU, L0LU, L0SU))` (`:1765`) — every caller
                // has already declined on this component, so the check has nothing left to abort on.
                let Some(mem_info) = MemoryOpInfo::of(op) else {
                    continue;
                };
                if ScalarOpComp::of(comp).is_none() {
                    continue;
                }
                let update_mode = !is_target_constant(mem_info.increment, ScalarOffset(0), defs);
                ConstantValue::of(mem_info.immutable_addr, defs)
                    .map(|constant| (constant, false, Rewritten::Address { update_mode }))
            }
        }) else {
            continue;
        };
        // `modifier = &getEvaluator().evaluateMultiplyByConst(*modifier, -1)` (`:1757`).
        let modifier = if negate && op_data.replace_with_mod == Modification::AddTo {
            evaluator.evaluate_multiply_by_const(&op_data.modifier, -1)
        } else {
            op_data.modifier.clone()
        };
        let new_const = add_to_or_replace_op(
            constant,
            &modifier,
            op_data.replace_with_mod,
            evaluator,
            sites,
            walked,
        );
        let Some(op) = defining_op_mut(walked, op_data.op) else {
            continue;
        };
        match op {
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
                *immutable_addr = new_const;
                if rewritten == (Rewritten::Address { update_mode: true }) {
                    *increment = new_const;
                }
            }
            // `op->setOperand(const_idx, new_const)` for the add and the sub.
            op => {
                if let Rewritten::Operand(index) = rewritten {
                    set_operand(op, index, new_const);
                }
            }
        }
    }
}

// crustify:todo: e468_processMergeableChain
//   authority : dcc/src/Transform/Sentient/ScalarOpMergingAndHoisting.cpp:1645  (58 body lines, level 2)
//   original  : bool ScalarOpHoisting::processMergeableChain( Operation *input_to_chain, Operation *first_op_in_chain, const EvaluatedValue &merging_increment, SmallVector<OperationData> &ops_to_update)
//   calls     : e158_doesValueExceedLRFRange, e361_isImmutableValueInRange

// crustify:todo: e469_hoistForLinearChain
//   authority : dcc/src/Transform/Sentient/ScalarOpMergingAndHoisting.cpp:2034  (29 body lines, level 2)
//   original  : void ScalarOpHoisting::hoistForLinearChain( BlockArgument &main_iv, Operation *derived_iv, SmallVector<OperationData> &comp_ops)
//   calls     : e173_hoistCandidateOutOfLoop, e365_applyOperationData

// crustify:todo: e529_processForOpResult
//   authority : dcc/src/Transform/Sentient/ScalarOpMergingAndHoisting.cpp:1484  (13 body lines, level 3)
//   original  : bool ScalarOpHoisting::processForOpResult( sentient::ForOp *for_op, int result_idx, const EvaluatedValue &adjustment_increment, SmallVector<OperationData> &ops_to_update)
//   calls     : e171_isMergeableOpOrChain, e468_processMergeableChain

// crustify:todo: e530_processForDerivedIVElimination
//   authority : dcc/src/Transform/Sentient/ScalarOpMergingAndHoisting.cpp:1843  (38 body lines, level 3)
//   original  : bool ScalarOpHoisting::processForDerivedIVElimination(BlockArgument &main_iv, Operation *derived_iv)
//   calls     : e171_isMergeableOpOrChain, e365_applyOperationData, e468_processMergeableChain

// crustify:todo: e578_adjustCandidateForOpResult
//   authority : dcc/src/Transform/Sentient/ScalarOpMergingAndHoisting.cpp:1514  (19 body lines, level 4)
//   original  : bool ScalarOpHoisting::adjustCandidateForOpResult( BlockArgument &main_iv, const EvaluatedValue &adjustment_increment)
//   calls     : e170_addForOpResultAdjustment, e365_applyOperationData, e529_processForOpResult

// crustify:todo: e612_processForLinearChain
//   authority : dcc/src/Transform/Sentient/ScalarOpMergingAndHoisting.cpp:1893  (129 body lines, level 5)
//   original  : bool ScalarOpHoisting::processForLinearChain(BlockArgument &main_iv, Operation *derived_iv)
//   calls     : e361_isImmutableValueInRange, e469_hoistForLinearChain, e578_adjustCandidateForOpResult

// crustify:todo: e613_processForGenericHoisting
//   authority : dcc/src/Transform/Sentient/ScalarOpMergingAndHoisting.cpp:2075  (81 body lines, level 5)
//   original  : bool ScalarOpHoisting::processForGenericHoisting(BlockArgument &main_iv, Operation *derived_iv)
//   calls     : e171_isMergeableOpOrChain, e173_hoistCandidateOutOfLoop, e365_applyOperationData, e578_adjustCandidateForOpResult

// crustify:todo: e635_runScalarOpHoisting
//   authority : dcc/src/Transform/Sentient/ScalarOpMergingAndHoisting.cpp:2184  (61 body lines, level 6)
//   original  : void ScalarOpHoisting::runScalarOpHoisting()
//   calls     : e530_processForDerivedIVElimination, e612_processForLinearChain, e613_processForGenericHoisting

#[cfg(test)]
mod unit_tests {
    use super::*;
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
            result_reg: Reg::UNALLOCATED,
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
            iv_reg: ops::Reg::UNALLOCATED,
            iv: Val(0),
            bound: Val(1),
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
        let mut evaluator = StatedEvaluator {
            offset: Val(9),
            sums: Vec::new(),
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

    /// e172 — `AddTo` folds the modifier into what the constant already evaluates to, and `Replace`
    /// asks for nothing but the modifier.
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
        let mut evaluator = StatedEvaluator {
            offset: Val(9),
            sums: Vec::new(),
        };
        let modifier = absolute(ScalarOffset(16));
        assert_eq!(
            add_to_or_replace_op(
                op,
                &modifier,
                Modification::Replace,
                &mut evaluator,
                &mut sites,
                &mut walked
            ),
            Val(9)
        );
        assert!(evaluator.sums.is_empty());
        assert_eq!(
            add_to_or_replace_op(
                op,
                &modifier,
                Modification::AddTo,
                &mut evaluator,
                &mut sites,
                &mut walked
            ),
            Val(9)
        );
        assert_eq!(evaluator.sums, vec![(modifier, absolute(ScalarOffset(4)))]);
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
    /// e365 — ⛔ BOTH TRAPS AT ONCE. A `scalar_sub` asked to ADD its modifier has it negated first, and
    /// a composite transfer whose increment is not already the constant 0 has that increment assigned
    /// THE SAME new value as its immutable address.
    #[test]
    fn applying_the_data_negates_a_subs_modifier_and_assigns_the_increment_too() {
        /// The evaluator of the two tests above with the `-1` multiply recorded.
        struct NegatingEvaluator {
            factors: Vec<i64>,
        }

        impl ExpressionEvaluator for NegatingEvaluator {
            fn evaluate_value(&mut self, _value: Val) -> Evaluation {
                absolute(ScalarOffset(4))
            }

            fn evaluate_sum(&mut self, _lhs: &Evaluation, _rhs: &Evaluation) -> Evaluation {
                absolute(ScalarOffset(20))
            }

            fn evaluate_multiply_by_const(
                &mut self,
                ev: &Evaluation,
                factor: i64,
            ) -> Evaluation {
                self.factors.push(factor);
                ev.clone()
            }

            fn build_offset_value(
                &mut self,
                _evaluation: &Evaluation,
                _sites: &mut OffsetSites<'_>,
                _walked: &mut Vec<Op>,
                _ty: ScalarTy,
            ) -> Val {
                Val(9)
            }
        }

        let sub = |lhs, rhs, result| {
            Op::Sentient(ops::Op::ScalarSub {
                lhs,
                rhs,
                result,
                reg: None,
                element_size: None,
                ty: ScalarTy::Index,
            })
        };
        let mut walked = vec![
            constant(64, Val(2)),
            sub(Val(20), Val(2), Val(3)),
            constant(64, Val(5)),
            // ⭐ NOT THE CONSTANT 0, which is what puts the transfer in update mode.
            constant(8, Val(6)),
            load_and_send(Val(1), Val(5), Val(6)),
        ];
        let modifier = absolute(ScalarOffset(16));
        let ops_to_update = [
            OperationData {
                op: Val(3),
                modifier: modifier.clone(),
                merging_increment: modifier.clone(),
                replace_with_mod: Modification::AddTo,
            },
            OperationData {
                op: Val(31),
                modifier: modifier.clone(),
                merging_increment: modifier.clone(),
                replace_with_mod: Modification::Replace,
            },
        ];
        let mut consts = Vec::new();
        let mut values = values_after(40);
        let mut sites = OffsetSites {
            consts: &mut consts,
            query_maps: None,
            values: &mut values,
        };
        let mut evaluator = NegatingEvaluator {
            factors: Vec::new(),
        };

        apply_operation_data(
            &ops_to_update,
            GenericComp::Lxlu,
            &mut evaluator,
            &mut sites,
            &mut walked,
        );

        // The sub's ONE constant operand is its rhs, and the negation was asked for exactly once.
        assert_eq!(walked[1], sub(Val(20), Val(9), Val(3)));
        assert_eq!(evaluator.factors, vec![-1]);
        assert_eq!(walked[4], load_and_send(Val(1), Val(9), Val(9)));
    }
}
