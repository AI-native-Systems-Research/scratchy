// SPDX-License-Identifier: Apache-2.0
//! THE OUT-OF-SCOPE `Analyses/` SEAM — the handles these passes hold, and never look inside.
//!
//! ⛔⛔ `dcc/src/Transform/Sentient/Analyses/` (39 files) AND `RegisterInitialization/` (12) ARE NOT
//! IN THIS CAMPAIGN — ~11,700 lines, and 122 of the 656 units name one (`crustify-senpass/
//! OUTSIDE-DEPS.tsv` per pass, `OUTSIDE-UNITS.tsv` per unit). What lives here is ONLY the identity
//! of a value such an analysis owns, because a ported field has to have a type.
//!
//! ⛔ EVERY *OPERATION* ON ONE IS A `todo!` NAMING THE ANALYSIS, at the unit that needs it. Do not
//! invent the analysis, do not inline a guess at what it would have returned, and do not substitute
//! a constant for its result.

use crate::bridges::dataflow_ir_to_sentient::vc_vector_operands::OpId;
use crate::islands::dataflow_ir::Values;
use crate::islands::dataflow_ir::ty::ScalarTy;
use crate::islands::sentient::dialects::{Op, Val};

/// AN `EvaluatedValue` THE EXPRESSION EVALUATOR OWNS — an identity, not a value.
///
/// ⛔ 50 OF THE 656 UNITS NAME THIS TYPE and none of them may look inside it:
/// `Analyses/ExpressionEvaluatorUtils.{h,cpp}` is out of campaign scope, so `evaluateSum`,
/// `evaluateMinMax`, `evaluateMultiplyByConst` and `getConstant` are `todo!`s at their call sites.
///
/// ⭐ A HANDLE BECAUSE THE REFERENCE STORES A BORROWED POINTER. Every field that holds one is
/// `const EvaluatedValue *` into the evaluator's own memoisation arena (`AddressPinningAndToggle.cpp`
/// `:174`, `:284`, `:449`, `:527`, `:626`), owned by the one global `ExpressionEvaluator &` the pass
/// threads through — so what a descriptor keeps is which entry, not the entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EvaluatedValue(pub u32);

/// A SIGNED SCALAR OFFSET — `ScalarValue`, what an evaluated expression carries beside its base.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ScalarOffset(pub i64);

/// `BaseValue` — the value an expression is an offset FROM, and whether it is subtracted
/// (`Analyses/ExpressionEvaluatorUtils.h:33-46`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BaseValue {
    /// `value_`.
    pub value: Val,
    /// `is_negated_` — the base is `-value`, so the expression is `offset - value`.
    pub negated: bool,
}

/// WHAT ONE `evaluateValue` ANSWER TELLS A PORTED PASS — the four queries, and nothing else.
///
/// ⛔⛔ THE FIELDS ARE THE CALLS, NOT A MODEL OF `EvaluatedValue`. The arithmetic and predicates on
/// the class hierarchy (`isAnyValLessThan`, `isDivisibleBy`, `retrieveAllValues`, `:63-205`) are out
/// of campaign scope, so what is spelled here is exactly `isKnownAbsolute()`, `baseValue()`,
/// `AllUnitEvaluatedValue::offsetValue()` and `PerUnitEvaluatedValue::offsetValuesPerUnit()` — every
/// query `LightweightSimplification.cpp` and `ScalarOpMergingAndHoisting.cpp` ask.
///
/// ⭐ A PER-UNIT KIND IS THE FAILED `dyn_cast`, and the distinction is load-bearing:
/// `addOrSubWithZeroSimplification` declines a per-unit value even when it is known absolute
/// (`:58-59`). See [`EvaluatedValue`] for the identity the memoising passes store instead.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Evaluation {
    /// `isKnownAbsolute()` (`:103`).
    pub known_absolute: bool,
    /// `baseValue()` (`:104`) — `None` when the expression has no base.
    pub base: Option<BaseValue>,
    /// Which of the two `EvaluatedValueKind`s this is, and the offset(s) it carries.
    pub offsets: Offsets,
}

/// WHICH OF THE TWO `EvaluatedValue` KINDS THIS IS, WITH ITS OFFSETS — `enum EvaluatedValueKind
/// { EVK_AllUnit, EVK_PerUnit }` (`Analyses/ExpressionEvaluatorUtils.h:59`).
///
/// ⛔ THE `dyn_cast` PAIR IS THIS MATCH, AND IT IS CLOSED AT TWO. `doesImmutableImmExceedRange`
/// casts to each kind in turn and falls through to `false` for neither
/// (`ScalarOpMergingAndHoisting.cpp:149-164`); with the kind an enum that fall-through is an arm
/// nothing can reach, so the port has none to get wrong.
///
/// ⛔ THE PER-UNIT KEY IS A `Value`, NOT A UNIT INDEX: `typedef llvm::DenseMap<Value, ScalarValue>
/// PerUnitValuesMap` (`:176`), keyed by the `unitKey()` the evaluator built the map against — so
/// the key is stated here from the header's own declaration and not guessed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Offsets {
    /// `AllUnitEvaluatedValue::offsetValue()` (`:163`) — one offset every unit shares.
    AllUnit(ScalarOffset),
    /// `PerUnitEvaluatedValue::offsetValuesPerUnit()` (`:194`) — one offset per unit.
    PerUnit(Vec<(Val, ScalarOffset)>),
}

impl Evaluation {
    /// `dyn_cast<AllUnitEvaluatedValue>(&v)->offsetValue()`, and `None` for the per-unit kind —
    /// the failed cast `addOrSubWithZeroSimplification` turns on
    /// (`LightweightSimplification.cpp:58-59`).
    #[must_use]
    pub fn all_unit_offset(&self) -> Option<ScalarOffset> {
        match &self.offsets {
            Offsets::AllUnit(offset) => Some(*offset),
            Offsets::PerUnit(_) => None,
        }
    }

    /// EVERY OFFSET THIS VALUE CARRIES — the single one of the all-unit kind, or one per unit.
    ///
    /// ⭐ THE TWO `dyn_cast` ARMS OF `doesImmutableImmExceedRange` DIFFER ONLY IN HOW THEY REACH THE
    /// OFFSETS (`ScalarOpMergingAndHoisting.cpp:149-163`): both apply the same range test to every
    /// offset they can see and return on the first failure, so one iterator serves both.
    pub fn offset_values(&self) -> impl Iterator<Item = ScalarOffset> + '_ {
        let (all_unit, per_unit): (Option<ScalarOffset>, &[(Val, ScalarOffset)]) =
            match &self.offsets {
                Offsets::AllUnit(offset) => (Some(*offset), &[]),
                Offsets::PerUnit(offsets) => (None, offsets.as_slice()),
            };
        all_unit
            .into_iter()
            .chain(per_unit.iter().map(|&(_, offset)| offset))
    }
}

/// WHERE `buildOffsetValue` PUTS THE OPS IT CREATES — the two `OpBuilder`s, as blocks.
///
/// ⛔⛔ THEY ARE TWO DIFFERENT BLOCKS AND ONE OF THEM CAN BE THE BLOCK BEING WALKED.
/// `const_builder` is set to the start of the block the `dataflow.program_unit` *sits in*
/// (`LightweightSimplification.cpp:338-339`), which is never the walked block; `query_map_builder`
/// is set to the start of `getLocalOrGlobalRegion(ops_list.back())`'s front block (`:278-281`),
/// which CAN be it. One `&mut` cannot be handed out twice, so [`OffsetSites::query_maps`] is
/// `None` for exactly that case and the walked block is passed separately.
pub struct OffsetSites<'a> {
    /// `const_builder`'s block — the constants an offset materialises into.
    pub consts: &'a mut Vec<Op>,
    /// `query_map_builder`'s block; `None` says it IS the block being walked.
    pub query_maps: Option<&'a mut Vec<Op>>,
    /// The minter, because every created op binds a fresh value.
    pub values: &'a mut Values,
}

/// THE `ExpressionEvaluator&` A PASS IS HANDED — a trait, because the analysis behind it is not in
/// this campaign and a test must still be able to state its answers.
///
/// ⛔ `Analyses/ExpressionEvaluatorUtils.{h,cpp}` IS OUT OF CAMPAIGN SCOPE, so the crate's only
/// implementation is [`OutOfScopeEvaluator`] and every method of it is a `todo!` naming the analysis.
/// ⭐ `build_offset_value` sits HERE rather than on [`Evaluation`] (where the reference has it, as
/// `EvaluatedValue::buildOffsetValue`, `:126`) so that one out-of-scope seam is one trait.
pub trait ExpressionEvaluator {
    /// `ExpressionEvaluator::evaluateValue` (`Analyses/ExpressionEvaluatorUtils.h:219`).
    fn evaluate_value(&mut self, value: Val) -> Evaluation;

    /// `ExpressionEvaluator::evaluateSum` (`Analyses/ExpressionEvaluatorUtils.h:242`) — the
    /// evaluation of `lhs + rhs`, which is how a hoist folds an increment into a constant it already
    /// evaluated.
    fn evaluate_sum(&mut self, lhs: &Evaluation, rhs: &Evaluation) -> Evaluation;

    /// `ExpressionEvaluator::evaluateSub` (`Analyses/ExpressionEvaluatorUtils.h:260`) — the
    /// evaluation of `lhs - rhs`, which is how a reordered toggle turns `minuend - init` into the
    /// loop's new initializer (`ToggleReordering.cpp:128-129`).
    ///
    /// ⭐ DEFAULTED, like [`ExpressionEvaluator::build_offset_value_of`]: the out-of-scope refusal is
    /// stated once, and a test double for a pass that only ever sums need not repeat it.
    fn evaluate_sub(&mut self, lhs: &Evaluation, rhs: &Evaluation) -> Evaluation {
        let _ = (lhs, rhs);
        todo!(
            "ExpressionEvaluator::evaluateSub (Analyses/ExpressionEvaluatorUtils.h:260) — out of campaign scope"
        )
    }

    /// `EvaluatedValue::buildOffsetValue` (`Analyses/ExpressionEvaluatorUtils.h:126`) — materialises
    /// the offset as a value, creating ops in `sites` (`walked` when `sites.query_maps` is `None`).
    fn build_offset_value(
        &mut self,
        evaluation: &Evaluation,
        sites: &mut OffsetSites<'_>,
        walked: &mut Vec<Op>,
        ty: ScalarTy,
    ) -> Val;

    /// The SAME `EvaluatedValue::buildOffsetValue` (`:126`) reached from a STORED handle rather than a
    /// fresh [`Evaluation`] — what a memoising pass has when it comes back to build the value.
    ///
    /// ⛔ THE HANDLE IS NOT AN [`Evaluation`] AND CANNOT BE TURNED INTO ONE HERE: the arena entry it
    /// names is the analysis's, so `ScalarOpMerging::unrollBurstAndIL` (`:1042`) calls the method on
    /// `const EvaluatedValue *` directly. ⭐ DEFAULTED so the out-of-scope refusal is stated once and
    /// a test double that never unrolls need not repeat it.
    fn build_offset_value_of(
        &mut self,
        immutable: EvaluatedValue,
        sites: &mut OffsetSites<'_>,
        walked: &mut Vec<Op>,
        ty: ScalarTy,
    ) -> Val {
        let _ = (immutable, sites, walked, ty);
        todo!(
            "EvaluatedValue::buildOffsetValue (Analyses/ExpressionEvaluatorUtils.h:126) — out of campaign scope"
        )
    }
}

/// THE ONE CRATE IMPLEMENTATION: the analysis is not ported, so asking it anything is a `todo!`.
#[derive(Debug, Clone, Copy, Default)]
pub struct OutOfScopeEvaluator;

impl ExpressionEvaluator for OutOfScopeEvaluator {
    fn evaluate_value(&mut self, _value: Val) -> Evaluation {
        todo!(
            "ExpressionEvaluator::evaluateValue (Analyses/ExpressionEvaluatorUtils.h:219) — out of campaign scope"
        )
    }

    fn evaluate_sum(&mut self, _lhs: &Evaluation, _rhs: &Evaluation) -> Evaluation {
        todo!(
            "ExpressionEvaluator::evaluateSum (Analyses/ExpressionEvaluatorUtils.h:242) — out of campaign scope"
        )
    }

    fn build_offset_value(
        &mut self,
        _evaluation: &Evaluation,
        _sites: &mut OffsetSites<'_>,
        _walked: &mut Vec<Op>,
        _ty: ScalarTy,
    ) -> Val {
        todo!(
            "EvaluatedValue::buildOffsetValue (Analyses/ExpressionEvaluatorUtils.h:126) — out of campaign scope"
        )
    }
}

/// AN INSTRUCTION COUNT FROM `InstructionEstimatorImpl` — the reference's `int`.
///
/// ⛔ SIGNED BECAUSE THE REFERENCE'S IS: `getRemainingIbuffSpace` goes negative once a unit overruns
/// the buffer, and the split/unroll cost model compares against it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct InstructionCount(pub i32);

/// THE `InstructionEstimatorImpl&` A PASS IS HANDED — a trait for the same reason
/// [`ExpressionEvaluator`] is one: the estimator is not in this campaign, and a test must still be
/// able to state its answers.
///
/// ⛔ `Analyses/InstructionEstimation.{h,cpp}` IS OUT OF CAMPAIGN SCOPE, so the crate's only
/// implementation is [`OutOfScopeInstructionEstimator`] and every method of it is a `todo!`.
/// ⭐ THE REFERENCE OVERLOADS ON THE ARGUMENT KIND (op, block, region, unit); the two overloads the
/// ported units call are named apart here because Rust has no overloading.
pub trait InstructionEstimator {
    /// `recalculate(ctx, unit)` (`Analyses/InstructionEstimation.h:74`) — recounts the whole unit.
    fn recalculate(&mut self, unit: &[Op]);

    /// `getEstimatedInstructionCount(ctx, Operation *)` (`Analyses/InstructionEstimation.h:56`) —
    /// the op and everything nested in it.
    fn estimated_instruction_count_of_op(&mut self, op: &Op) -> InstructionCount;

    /// `getEstimatedInstructionCount(ctx, Region *)` (`Analyses/InstructionEstimation.h:62`).
    fn estimated_instruction_count_of_region(&mut self, region: &[Op]) -> InstructionCount;
}

/// THE ONE CRATE IMPLEMENTATION: the estimator is not ported, so asking it anything is a `todo!`.
#[derive(Debug, Clone, Copy, Default)]
pub struct OutOfScopeInstructionEstimator;

impl InstructionEstimator for OutOfScopeInstructionEstimator {
    fn recalculate(&mut self, _unit: &[Op]) {
        todo!(
            "InstructionEstimatorImpl::recalculate (Analyses/InstructionEstimation.h:74) — out of campaign scope"
        )
    }

    fn estimated_instruction_count_of_op(&mut self, _op: &Op) -> InstructionCount {
        todo!(
            "InstructionEstimatorImpl::getEstimatedInstructionCount (Analyses/InstructionEstimation.h:56) — out of campaign scope"
        )
    }

    fn estimated_instruction_count_of_region(&mut self, _region: &[Op]) -> InstructionCount {
        todo!(
            "InstructionEstimatorImpl::getEstimatedInstructionCount (Analyses/InstructionEstimation.h:62) — out of campaign scope"
        )
    }
}

/// A `Candidate` THE REGISTER-INITIALISATION PIPELINE OWNS — an identity, not a candidate.
///
/// ⛔ `RegisterInitialization/Candidate.{h,cpp}` IS OUT OF CAMPAIGN SCOPE. The reference's lists are
/// `SmallVector<Candidate *>` (`Candidate.h:65`) whose entries the collector `delete`s in its own
/// destructor (`RegisterInitialization/Collector.h:59-62`), so what a ported list holds is WHICH
/// candidate, not the candidate — and nothing here may look inside one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Candidate(pub u32);

/// THE `CollectorInterface &` THE `Driver` IS HANDED (`RegisterInitialization/Collector.h:29`).
///
/// ⛔ OUT OF CAMPAIGN SCOPE, so the crate's only implementation is [`OutOfScopeCandidateCollector`].
/// ⭐ `&mut dyn` AT THE CALL SITES BECAUSE THE REFERENCE PICKS THE IMPLEMENTATION AT RUN TIME, from
/// `llvm::cl::opt<CollectorKind>` (`RegisterInitialization.cpp:40-52`).
pub trait CandidateCollector {
    /// `collectGlobalCandidates(results)` — the reference expects `results` empty on entry.
    fn collect_global_candidates(&mut self, results: &mut Vec<Candidate>);

    /// `collectLocalCandidates(core, results)` — `core` is a group leader's unit value.
    fn collect_local_candidates(&mut self, core: Val, results: &mut Vec<Candidate>);
}

/// THE `EvaluatorInterface &` THE `Driver` IS HANDED (`RegisterInitialization/Evaluator.h:36`).
///
/// ⛔ OUT OF CAMPAIGN SCOPE — see [`OutOfScopeCandidateEvaluator`]. Only the two methods
/// `runLocalAnalysis` calls are declared; `evaluateGlobally` lands with e346.
pub trait CandidateEvaluator {
    /// `evaluateLocally(candidates)` — weights, sorts and re-prioritises IN PLACE.
    fn evaluate_locally(&mut self, candidates: &mut Vec<Candidate>);

    /// `mergeInto(result, sublist)` (`Evaluator.h:66`) — inserts `sublist` keeping `result` sorted.
    fn merge_into(&mut self, result: &mut Vec<Candidate>, sublist: &[Candidate]);
}

/// THE `SelectorInterface &` THE `Driver` IS HANDED (`RegisterInitialization/Selector.h:38`).
///
/// ⛔ OUT OF CAMPAIGN SCOPE — see [`OutOfScopeCandidateSelector`]. `selectGlobally` lands with e346.
pub trait CandidateSelector {
    /// `selectLocally(local, global, core)` (`Selector.h:68`) — PURGES both lists in place; neither
    /// can grow.
    fn select_locally(
        &mut self,
        local: &mut Vec<Candidate>,
        global: &mut Vec<Candidate>,
        core: Val,
    );
}

/// THE `const UniformGroupAnalyzer &` THE `Driver` IS HANDED
/// (`Analyses/UniformGroupAnalysis.h:57`).
///
/// ⛔ OUT OF CAMPAIGN SCOPE — see [`OutOfScopeUniformGroups`]. Only `getGroupLeaders` is declared.
pub trait UniformGroups {
    /// `getGroupLeaders()` (`Analyses/UniformGroupAnalysis.h:67`) — one unit value per exclusive
    /// group.
    fn group_leaders(&self) -> Vec<Val>;
}

/// THE ONE CRATE IMPLEMENTATION of [`CandidateCollector`]: asking it anything is a `todo!`.
#[derive(Debug, Clone, Copy, Default)]
pub struct OutOfScopeCandidateCollector;

impl CandidateCollector for OutOfScopeCandidateCollector {
    fn collect_global_candidates(&mut self, _results: &mut Vec<Candidate>) {
        todo!(
            "CollectorInterface::collectGlobalCandidates (RegisterInitialization/Collector.h:37) — out of campaign scope"
        )
    }

    fn collect_local_candidates(&mut self, _core: Val, _results: &mut Vec<Candidate>) {
        todo!(
            "CollectorInterface::collectLocalCandidates (RegisterInitialization/Collector.h:47) — out of campaign scope"
        )
    }
}

/// THE ONE CRATE IMPLEMENTATION of [`CandidateEvaluator`]: asking it anything is a `todo!`.
#[derive(Debug, Clone, Copy, Default)]
pub struct OutOfScopeCandidateEvaluator;

impl CandidateEvaluator for OutOfScopeCandidateEvaluator {
    fn evaluate_locally(&mut self, _candidates: &mut Vec<Candidate>) {
        todo!(
            "EvaluatorInterface::evaluateLocally (RegisterInitialization/Evaluator.h:48) — out of campaign scope"
        )
    }

    fn merge_into(&mut self, _result: &mut Vec<Candidate>, _sublist: &[Candidate]) {
        todo!(
            "EvaluatorInterface::mergeInto (RegisterInitialization/Evaluator.h:66) — out of campaign scope"
        )
    }
}

/// THE ONE CRATE IMPLEMENTATION of [`CandidateSelector`]: asking it anything is a `todo!`.
#[derive(Debug, Clone, Copy, Default)]
pub struct OutOfScopeCandidateSelector;

impl CandidateSelector for OutOfScopeCandidateSelector {
    fn select_locally(
        &mut self,
        _local: &mut Vec<Candidate>,
        _global: &mut Vec<Candidate>,
        _core: Val,
    ) {
        todo!(
            "SelectorInterface::selectLocally (RegisterInitialization/Selector.h:68) — out of campaign scope"
        )
    }
}

/// THE ONE CRATE IMPLEMENTATION of [`UniformGroups`]: asking it anything is a `todo!`.
#[derive(Debug, Clone, Copy, Default)]
pub struct OutOfScopeUniformGroups;

impl UniformGroups for OutOfScopeUniformGroups {
    fn group_leaders(&self) -> Vec<Val> {
        todo!(
            "UniformGroupAnalyzer::getGroupLeaders (Analyses/UniformGroupAnalysis.h:67) — out of campaign scope"
        )
    }
}

/// A `LiveRange` THE PORT-ASSIGNMENT GRAPH KEEPS PER OPERAND — an identity, not the intervals.
///
/// ⛔ `Analyses/LiveRange.{hpp,cpp}` IS OUT OF CAMPAIGN SCOPE, so the `std::vector<LabeledRange>`
/// behind it is deliberately absent and `overlaps`/`unionWith` are `todo!`s at the units that need
/// them — `e572_computePortLiveRange` builds these and `e608_buildGraphEdges` reads them.
/// `e123_clean`, which only empties the map holding them, needs the type and nothing else.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct LiveRange;

/// THE `GraphColoring` A PASS OWNS — a trait, for the same reason [`ExpressionEvaluator`] is one: the
/// analysis is not in this campaign and a test must still be able to observe what a pass asks it.
///
/// ⛔ `Analyses/GraphColoring.{hpp,cpp}` IS OUT OF CAMPAIGN SCOPE. Only the methods a ported unit
/// actually calls are declared; `getOrAddNode`, `addBidirectionalEdge`, `addSameColorEdge` and
/// `doGraphColoring` belong to `e339`, `e608` and `e382` and are added by those units.
pub trait ColoringGraph {
    /// `GraphColoring::clear()` (`Analyses/GraphColoring.hpp:52-60`).
    ///
    /// ⛔⛔ NOT A RE-DEFAULT-CONSTRUCTION, AND THE DIFFERENCE OUTLIVES A PROGRAM UNIT. It deletes and
    /// clears `nodes_`, then clears `node_ids_` and `edges_` — and leaves `same_color_edges_` and
    /// `max_node_id_` STANDING, so both survive into the next unit the pass visits even though the
    /// constructor's own `clear()` ran on an empty object. Whoever gives this an interior must never
    /// write `*self = Self::default()`.
    fn clear(&mut self);
}

/// THE ONE CRATE IMPLEMENTATION: the analysis is not ported, so asking it anything is a `todo!`.
#[derive(Debug, Clone, Copy, Default)]
pub struct OutOfScopeColoringGraph;

impl ColoringGraph for OutOfScopeColoringGraph {
    fn clear(&mut self) {
        todo!("GraphColoring::clear (Analyses/GraphColoring.hpp:52) — out of campaign scope")
    }
}

/// `RegisterGraphs` (`Analyses/GraphColoring.hpp:152`) — one [`ColoringGraph`] per register locale,
/// plus their hyper-graphs and the value-to-locale cache, as the allocator holds it.
pub trait RegisterGraphs {
    /// `RegisterGraphs::clean()` (`Analyses/GraphColoring.hpp:179-183`).
    ///
    /// ⛔⛔ IT LEAVES `ec_map_` STANDING. Three of the four members are cleared and the same-colour
    /// equivalence classes are not, so they survive into the next program unit the allocator visits —
    /// the same shape of trap as [`ColoringGraph::clear`], and for the same reason nobody giving this
    /// an interior may write `*self = Self::default()`.
    fn clean(&mut self);
}

/// THE ONE CRATE IMPLEMENTATION: the analysis is not ported, so asking it anything is a `todo!`.
#[derive(Debug, Clone, Copy, Default)]
pub struct OutOfScopeRegisterGraphs;

impl RegisterGraphs for OutOfScopeRegisterGraphs {
    fn clean(&mut self) {
        todo!("RegisterGraphs::clean (Analyses/GraphColoring.hpp:179) — out of campaign scope")
    }
}

/// `RDENode` (`Analyses/RedundantDefinitionEliminationTree.hpp:34`) AS THE PORTED PASSES READ IT —
/// the tree itself is out of campaign scope, so only the two facts an `initializeDataflowInfo` or an
/// `isSimplifiable` asks of a node are represented.
///
/// ⭐ `Root` IS THE IDENTITY TEST, NOT A FLAG: `root_ = root_ ? root_ : new RDENode(nullptr)`
/// (`Analyses/RedundantDefinitionEliminationTree.cpp:294`) makes the root the ONLY node without an
/// operation, so `getRoot() == &node` is a CASE of this enum rather than a pointer comparison.
///
/// ⭐ ONE DEFINITION FOR THE WHOLE CAMPAIGN. Four RDE passes (`ImplicitSyncRE`,
/// `SetActiveMaskValueRE`, `SetMaskRE`, `SetSendDestinationRE`) override the same two hooks and ask a
/// node the same two questions, so this lives at the out-of-scope seam and not in one pass's module.
#[derive(Debug, Clone, Copy)]
pub enum RdeNode<'a> {
    /// The tree's root — `getOperation()` is null.
    Root,
    /// A node over one op.
    At {
        /// `getOperation()`.
        op: &'a Op,
        /// `isLeaf()` (`src/Analysis/OperationTree.hpp:70`).
        leaf: bool,
    },
}

/// THE `Liveness&` A PASS IS HANDED — a trait for the same reason [`ExpressionEvaluator`] is one:
/// the analysis is not in this campaign, and a test must still be able to observe WHICH values a
/// ported pass promotes.
///
/// ⛔ `Analyses/Liveness.{h,cpp}` IS OUT OF CAMPAIGN SCOPE, so the crate's only implementation is
/// [`OutOfScopeLiveness`] and every method of it is a `todo!`.
pub trait Liveness {
    /// `updateLiveRangesForProgramHeaderPromotion(candidate)` (`Analyses/Liveness.h:130`) — widens
    /// `candidate`'s live range to the whole program because it is about to live in the header.
    fn update_live_ranges_for_program_header_promotion(&mut self, candidate: Val);
}

/// THE ONE CRATE IMPLEMENTATION: liveness is not ported, so telling it anything is a `todo!`.
#[derive(Debug, Clone, Copy, Default)]
pub struct OutOfScopeLiveness;

impl Liveness for OutOfScopeLiveness {
    fn update_live_ranges_for_program_header_promotion(&mut self, _candidate: Val) {
        todo!(
            "Liveness::updateLiveRangesForProgramHeaderPromotion (Analyses/Liveness.h:130) — out of campaign scope"
        )
    }
}

/// A COUNT OF CYCLES — what `TimeStamp::getCyclesGap` answers and what an exposed-pipeline budget is
/// spent in (`Analyses/TimeStamps.h:20`, `TransformForExposedPipeline.cpp:308`).
///
/// ⛔ SIGNED, AND NEGATIVE IS A CASE THE CALLERS READ: `computeDependenciesSameBlock` drops a pair on
/// `gap < 0` (`:164`), and `insertNOPOperations` lets its own budget go below zero and still banks it
/// (`:333-337`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Cycles(pub i32);

/// ONE COLUMN OF A `TimeStamp` — `TimeStampColumnVal` (`Analyses/TimeStamps.h:40`) AS A PORTED PASS
/// HOLDS IT: an identity, like [`LiveRange`], because the loop or condition it names, and the
/// iteration count beside it, belong to the analysis.
///
/// ⛔ `Analyses/TimeStamps.{h,cpp}` IS OUT OF CAMPAIGN SCOPE. `isLoop`, `isCond`, `getLoop` and
/// `getCond` are `todo!`s at the units that need them — `e482_computeDependenciesSameBlock` walks a
/// timestamp to find a parent op, and `e234_printDependencies` only carries the vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TimeStampColumnVal;

/// `Dependency` (`Analyses/TimeStamps.h:30-34`) — one RAW hazard between two MACs, and how many
/// cycles apart they are.
///
/// ⛔ THE ENDS ARE POSITIONS, NOT `Operation *`: [`OpId`] is this crate's stand-in, so every entry in
/// a list of these GOES STALE the moment an op is inserted before it — which is exactly what
/// `e237_insertNOPOperations` does, and why it computes every count before it moves anything.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Dependency {
    /// `src` — the MAC whose result is written.
    pub src: OpId,
    /// `dst` — the MAC that reads it.
    pub dst: OpId,
    /// `gap`.
    pub gap: Cycles,
}
