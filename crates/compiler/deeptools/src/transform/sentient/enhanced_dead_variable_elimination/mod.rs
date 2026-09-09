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

//! `EnhancedDeadVariableElimination.cpp` — 15 of the campaign's 656 units (dependency level(s) [0, 1, 2, 3, 4, 5, 6, 7]).
//!
//! | unit | entry | level | lines | authority path:line |
//! |---|---|---|---|---|
//! | `e040_addToWorkListAndUpdateAssignment` | 040 | 0 | 31 | `dcc/src/Transform/Sentient/EnhancedDeadVariableElimination.cpp:66` |
//! | `e041_getInfluenceType` | 041 | 0 | 8 | `dcc/src/Transform/Sentient/EnhancedDeadVariableElimination.cpp:102` |
//! | `e042_printAssignments` | 042 | 0 | 12 | `dcc/src/Transform/Sentient/EnhancedDeadVariableElimination.cpp:112` |
//! | `e043_removeConstantIterArgs` | 043 | 0 | 19 | `dcc/src/Transform/Sentient/EnhancedDeadVariableElimination.cpp:690` |
//! | `e300_processWorkList` | 300 | 1 | 53 | `dcc/src/Transform/Sentient/EnhancedDeadVariableElimination.cpp:460` |
//! | `e301_initializeAssignmentForAnOperation` | 301 | 1 | 147 | `dcc/src/Transform/Sentient/EnhancedDeadVariableElimination.cpp:525` |
//! | `e437_initializeWorkList` | 437 | 2 | 4 | `dcc/src/Transform/Sentient/EnhancedDeadVariableElimination.cpp:685` |
//! | `e498_updateForOperation` | 498 | 3 | 141 | `dcc/src/Transform/Sentient/EnhancedDeadVariableElimination.cpp:127` |
//! | `e499_updateIfOperation` | 499 | 3 | 49 | `dcc/src/Transform/Sentient/EnhancedDeadVariableElimination.cpp:272` |
//! | `e500_updateUniformizeRegionsOperation` | 500 | 3 | 38 | `dcc/src/Transform/Sentient/EnhancedDeadVariableElimination.cpp:325` |
//! | `e557_exploreOperation` | 557 | 4 | 57 | `dcc/src/Transform/Sentient/EnhancedDeadVariableElimination.cpp:367` |
//! | `e558_exploreBlock` | 558 | 4 | 23 | `dcc/src/Transform/Sentient/EnhancedDeadVariableElimination.cpp:426` |
//! | `e596_updateProgramUnit` | 596 | 5 | 3 | `dcc/src/Transform/Sentient/EnhancedDeadVariableElimination.cpp:451` |
//! | `e622_runOn` | 622 | 6 | 36 | `dcc/src/Transform/Sentient/EnhancedDeadVariableElimination.cpp:710` |
//! | `e639_runOnOperation` | 639 | 7 | 6 | `dcc/src/Transform/Sentient/EnhancedDeadVariableElimination.cpp:58` |

// ⛔ THE PASS IS NOT WIRED INTO THE PIPELINE YET, so every item below is reachable only from this
// file's own tests until `e639_runOnOperation` (level 7) lands and something calls it. CI runs clippy
// with `-D warnings`, so without this the first ported leaf of a 15-unit module fails the gate.
// ⭐ REMOVE THIS WITH e639: at that point an unused item here is a real defect again.
#![allow(dead_code)]

use core::fmt::Write as _;
use std::collections::{BTreeMap, VecDeque};

use crate::islands::sentient::dialects::{
    self as dialects, Definitions, Op, Val, sentient, symbol, uniform,
};
use crate::islands::sentient::print;
use crate::units::DfirUnit;

/// `InfluenceType` (`EnhancedDeadVariableElimination.hpp:42`) — what an SSA value's value decides.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Influence {
    /// `kControlFlow`.
    ControlFlow,
    /// `kMemory`.
    Memory,
    /// `kNone`.
    None,
}

impl Influence {
    /// `enum_to_strings_influences_` (`EnhancedDeadVariableElimination.hpp:44`) — ⭐ A TABLE ON THE
    /// ENUM, not a `std::map` the pass carries, because the mapping is closed and total.
    #[must_use]
    pub(crate) const fn spelling(self) -> &'static str {
        match self {
            Influence::ControlFlow => "controlflow",
            Influence::Memory => "memory",
            Influence::None => "none",
        }
    }
}

/// WHERE A [`Val`] COMES FROM — the `!isa<BlockArgument>` / `getOwner()->getParentOp()` split.
///
/// ⭐ SUPPLIED BY THE CALLER, not stored: this island's ops carry no parent pointers, which the
/// campaign brief names as exactly the *mechanism* a port may drop and hand back (the precedent is
/// [`dialects::Definitions`]).
#[derive(Debug, Clone)]
pub(crate) enum Origin {
    /// An op result — `val.getDefiningOp()`.
    Defined(Op),
    /// A `dataflow.program_unit` iter_arg — the one origin e040 returns early on.
    ProgramUnitArg,
    /// A region argument of any other op — `cast<BlockArgument>(val).getOwner()->getParentOp()`.
    RegionArg(Op),
}

impl Origin {
    /// The op an assignment is blamed on, absent for a `dataflow.program_unit` iter_arg.
    #[must_use]
    fn owner(&self) -> Option<Owner> {
        match self {
            Origin::Defined(op) | Origin::RegionArg(op) => Some(Owner(op.clone())),
            Origin::ProgramUnitArg => None,
        }
    }
}

/// THE OP AN ASSIGNMENT IS BLAMED ON — and a witness that its value is not a `dataflow.program_unit`
/// iter_arg, which is why [`Assignment`] can hold one unconditionally.
#[derive(Debug, Clone)]
pub(crate) struct Owner(Op);

impl Owner {
    /// `isa<sentient::ConstantOp, symbol::CreateSymbolOp>` — the two owners allowed to disagree.
    ///
    /// ⛔ TRAP: `sentient::ConstantOp` IS `sentient.scalar_constant` (`SentientOps.td:848`), NOT
    /// `sentient.vector_constant` (`Sentient_VectorConstantOp`, `:866`).
    #[must_use]
    fn tolerates_disagreement(&self) -> bool {
        matches!(
            self.0,
            Op::Sentient(sentient::Op::ScalarConstant { .. })
                | Op::Symbol(symbol::Op::CreateSymbol { .. })
        )
    }

    /// `Operation::dump()` — ⭐ INTO A BUFFER, which is what makes the dump testable.
    #[must_use]
    fn dump(&self) -> String {
        let mut out = String::new();
        print::emit(&mut out, &self.0, 0);
        out
    }
}

/// One `assignments_` entry: the influence recorded, and the op that recorded it.
#[derive(Debug, Clone)]
pub(crate) struct Assignment {
    /// The influence this value was first seen to have.
    influence: Influence,
    /// `record->first`'s owner, kept because e040 and e042 both dump it.
    owner: Owner,
}

/// The message the disagreement carries (`EnhancedDeadVariableElimination.cpp:94`).
pub(crate) const INFLUENCE_CONFLICT: &str = "A SSA variable has influence over control and memory";

/// `op->emitError(...)` + `signalPassFailure()` AS DATA — one recorded disagreement.
///
/// ⭐ NOT A `Result` AND NOT A PANIC: `signalPassFailure` is a flag on the pass object, and the
/// port's flag is the round itself — the same shape `tf_transform_paged_mem_view` settled on. Keeping
/// it as data is what lets a caller say WHICH value disagreed.
#[derive(Debug, Clone)]
pub(crate) struct InfluenceConflict {
    /// The value with two influences.
    pub(crate) val: Val,
    /// What `op->dump()` wrote.
    pub(crate) dumped: String,
    /// The influence already in `assignments_`.
    pub(crate) recorded: Influence,
    /// The influence the caller proposed.
    pub(crate) proposed: Influence,
}

/// `op->emitError("Not expecting <op> in other units")` + `signalPassFailure()` AS DATA — the same
/// treatment [`InfluenceConflict`] gives e040's disagreement, for the four unit-placement refusals
/// e301 makes.
#[derive(Debug, Clone)]
pub(crate) struct MisplacedOp {
    /// What `op->dump()` would have written.
    pub(crate) dumped: String,
    /// The unit whose program the op appeared in.
    pub(crate) unit: DfirUnit,
}

impl MisplacedOp {
    /// One refusal, recorded.
    #[must_use]
    fn of(op: &Op, unit: DfirUnit) -> MisplacedOp {
        let mut dumped = String::new();
        print::emit(&mut dumped, op, 0);
        MisplacedOp { dumped, unit }
    }
}

/// `bool add` — whether the value also joins `worklist_`. The header's default is `true`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum WorklistAdd {
    /// `add = true`.
    Push,
    /// `add = false`.
    RecordOnly,
}

/// `EnhancedDeadVariableEliminationPass`'s dataflow state (`EnhancedDeadVariableElimination.hpp`).
#[derive(Debug, Clone, Default)]
pub(crate) struct EnhancedDeadVariableElimination {
    /// `assignments_` — ⭐ ORDERED, because e042 iterates it and `DenseMap` order is unspecified.
    assignments: BTreeMap<Val, Assignment>,
    /// `worklist_`.
    worklist: VecDeque<Val>,
    /// `signalPassFailure()`, as data.
    conflicts: Vec<InfluenceConflict>,
    /// The `emitError("Not expecting <op> in other units")` arms of e301, as data.
    misplaced: Vec<MisplacedOp>,
}

impl EnhancedDeadVariableElimination {
    /// Replaces: e040_addToWorkListAndUpdateAssignment
    ///
    /// Records `val`'s influence and queues it; a differing second influence is a conflict unless the
    /// owner is a `sentient.scalar_constant` or a `symbol.create_symbol`.
    ///
    /// ⛔ TRAP: `Operation *user` IS UNUSED IN THE BODY (`:66-99`) — dropped here, though e300 and
    /// e301 still pass it.
    /// ⛔ TRAP: the early return on a `dataflow.program_unit` iter_arg is why [`Assignment`] holds an
    /// [`Owner`] unconditionally.
    pub(crate) fn add_to_work_list_and_update_assignment(
        &mut self,
        val: Val,
        origin: &Origin,
        influence: Influence,
        add: WorklistAdd,
    ) {
        let Some(owner) = origin.owner() else {
            return;
        };
        if let Some(record) = self.assignments.get(&val) {
            if record.influence != influence && !record.owner.tolerates_disagreement() {
                let conflict = InfluenceConflict {
                    val,
                    dumped: record.owner.dump(),
                    recorded: record.influence,
                    proposed: influence,
                };
                self.conflicts.push(conflict);
            }
            return;
        }
        self.assignments
            .insert(val, Assignment { influence, owner });
        if matches!(add, WorklistAdd::Push) {
            self.worklist.push_back(val);
        }
    }

    /// Replaces: e041_getInfluenceType
    ///
    /// An unrecorded value has no influence.
    #[must_use]
    pub(crate) fn influence_type(&self, val: Val) -> Influence {
        self.assignments
            .get(&val)
            .map_or(Influence::None, |record| record.influence)
    }

    /// Replaces: e042_printAssignments
    ///
    /// Per record, in key order: the owning op as text, then `influence: <spelling>`.
    ///
    /// ⛔ TRAP: `llvm::outs()` is a PARAMETER here, so the pass writes into a caller's buffer rather
    /// than the process's stdout and a test can read what it wrote. ONE BUFFER FOR BOTH HALVES IS
    /// THIS PORT'S CHOICE: `Operation::dump()` (`:115`, `:117`) does not write to `llvm::outs()`, so
    /// the reference's two halves land on different streams and their interleaving was never
    /// observable. No MLIR header ships in `/Users/nickm/git/deeptools-src`, so which stream `dump()`
    /// takes could not be read here; debug trace only, and no golden compares these bytes.
    pub(crate) fn print_assignments(&self, out: &mut String) {
        for record in self.assignments.values() {
            print::emit(out, &record.owner.0, 0);
            let _ = writeln!(out, "influence: {}", record.influence.spelling());
        }
    }

    /// Every recorded disagreement, each one a `signalPassFailure()` the reference made.
    #[must_use]
    pub(crate) fn conflicts(&self) -> &[InfluenceConflict] {
        &self.conflicts
    }

    /// Every op e301 refused to see on the unit it was on.
    #[must_use]
    pub(crate) fn misplaced(&self) -> &[MisplacedOp] {
        &self.misplaced
    }
}

/// Replaces: e043_removeConstantIterArgs
///
/// Every `sentient.for` whose yield hands an iter_arg straight back has that arg AND the matching
/// loop result rewired to the arg's init.
///
/// ⛔ TRAP: PRE-ORDER MEANS INNER LOOPS FIRST — the enclosing `sentient.yield` is the LAST op in its
/// loop's body, so a nested loop's yield is reached before it. The walk is therefore collected as
/// paths in that order and each rewrite then addresses the WHOLE `scope`, because a loop result is
/// used OUTSIDE the loop and `replaceAllUsesWith` is not region-local.
pub(crate) fn remove_constant_iter_args(scope: &mut [Op]) {
    let mut paths = Vec::new();
    collect_for_paths(scope, &mut Vec::new(), &mut paths);
    for path in &paths {
        let count = at_path(scope, path).map_or(0, yielded_count);
        for i in (0..count).rev() {
            // ⭐ A FRESH READ EACH ROUND, because the reference re-reads `yield_op->getOperand(i)`
            // after the previous round's rewrites.
            let found = at_path(scope, path).and_then(|for_op| constant_iter_arg(for_op, i));
            if let Some((arg, init, result)) = found {
                dialects::replace_all_uses_with(scope, arg, init);
                dialects::replace_all_uses_with(scope, result, init);
            }
        }
    }
}

/// One step of the walk: which op of a region, and which of its regions to enter.
#[derive(Debug, Clone, Copy)]
struct Descent {
    /// The op's index in the region being walked.
    op: usize,
    /// Which of that op's regions to enter.
    region: usize,
}

/// A `sentient.for`'s place in the tree — the regions to enter, then its index in the innermost.
#[derive(Debug, Clone)]
struct ForPath {
    /// The regions entered on the way down.
    descents: Vec<Descent>,
    /// The loop's index in the innermost region.
    at: usize,
}

/// The regions of one op that can hold a `sentient.for`.
///
/// ⛔ NO SHARED-DIALECT ARM HOLDS ONE: the lower rung's `Op` has no `sentient` variant at all, so a
/// `sentient.for` cannot appear in one of its regions. Exhaustive, as [`dialects::defining_op`] is.
#[must_use]
fn regions_of(op: &Op) -> Vec<&[Op]> {
    match op {
        Op::Sentient(inner) => sentient::regions(inner),
        Op::AffineFor(loop_op) => vec![&loop_op.body],
        // ⭐ THIS ONE'S REGIONS ARE THIS RUNG'S, so a `sentient.for` CAN sit in one.
        Op::UniformRegions(regions) => regions
            .regions()
            .iter()
            .map(|region| region.body.as_slice())
            .collect(),
        Op::Arith(_)
        | Op::Scf(_)
        | Op::Affine(_)
        | Op::Vector(_)
        | Op::Dataflow(_)
        | Op::Agen(_)
        | Op::VectorChain(_)
        | Op::Symbol(_)
        | Op::Uniform(_) => Vec::new(),
    }
}

/// Every `sentient.for` under `scope`, in the order its `sentient.yield` is walked.
fn collect_for_paths(scope: &[Op], descents: &mut Vec<Descent>, out: &mut Vec<ForPath>) {
    for (at, op) in scope.iter().enumerate() {
        for (region, body) in regions_of(op).into_iter().enumerate() {
            descents.push(Descent { op: at, region });
            collect_for_paths(body, descents, out);
            descents.pop();
        }
        if matches!(op, Op::Sentient(sentient::Op::For { .. })) {
            out.push(ForPath {
                descents: descents.clone(),
                at,
            });
        }
    }
}

/// The op a [`ForPath`] names, absent only if the tree changed shape under it.
#[must_use]
fn at_path<'a>(scope: &'a [Op], path: &ForPath) -> Option<&'a Op> {
    let mut here = scope;
    for step in &path.descents {
        here = regions_of(here.get(step.op)?)
            .into_iter()
            .nth(step.region)?;
    }
    here.get(path.at)
}

/// The loop's own terminator, which is the last `sentient.yield` in its body.
#[must_use]
fn terminator(for_op: &Op) -> Option<&Vec<Val>> {
    let Op::Sentient(sentient::Op::For { body, .. }) = for_op else {
        return None;
    };
    body.iter().rev().find_map(|op| {
        if let Op::Sentient(sentient::Op::Yield { results }) = op {
            Some(results)
        } else {
            None
        }
    })
}

/// `yield_op->getNumOperands()`, and 0 where the loop has no terminator to walk.
#[must_use]
fn yielded_count(for_op: &Op) -> usize {
    terminator(for_op).map_or(0, Vec::len)
}

/// `(iter_arg, init, result)` at position `i` when the yield hands the arg straight back.
#[must_use]
fn constant_iter_arg(for_op: &Op, i: usize) -> Option<(Val, Val, Val)> {
    let Op::Sentient(sentient::Op::For { carried, .. }) = for_op else {
        return None;
    };
    let yielded = terminator(for_op)?;
    // ⛔ `.get(i)`: a yield with more operands than the loop carries is an out-of-bounds read in the
    // reference (`getRegionIterArgs()[i]`), and there is nothing here to rewrite.
    let carried = carried.get(i)?;
    (*yielded.get(i)? == carried.arg).then_some((carried.arg, carried.init, carried.result))
}

/// WHERE A VALUE COMES FROM, RESOLVED AGAINST A SCOPE — `val.getDefiningOp()` when it has one, and
/// `cast<BlockArgument>(val).getOwner()->getParentOp()` when it does not.
///
/// ⭐ THE `Operation *user` E300 AND E301 PASS IS NOT THIS. That argument is dead in e040's body
/// (`:66-99`); what e040 actually reads is the owner of `val` ITSELF (`:87-91`), which is what this
/// answers.
fn origin_of(val: Val, defs: Definitions<'_>) -> Origin {
    if let Some(def) = defs.of(val) {
        return Origin::Defined(def.clone());
    }
    if defs.program_unit_units_of(val).is_some() {
        return Origin::ProgramUnitArg;
    }
    match defs.block_arg_owner_of(val) {
        Some(op) => Origin::RegionArg(op.clone()),
        None => todo!(
            "addToWorkListAndUpdateAssignment: {val:?} is neither an op result nor a region argument \
             of anything in scope, so `cast<BlockArgument>(val).getOwner()` \
             (EnhancedDeadVariableElimination.cpp:73) has nothing to cast"
        ),
    }
}

impl EnhancedDeadVariableElimination {
    /// Replaces: e300_processWorkList
    ///
    /// Drains the worklist, propagating each value's influence to what feeds it: a loop-carried arg to
    /// its init and its yield operand, an add/sub/copy to its inputs, and a loop/if/uniformize result
    /// to every region's yield operand at the same position (plus that loop's own iter_arg).
    ///
    /// ⛔ TRAP: island block-arg position 0 is the induction variable, so
    /// `getIndexOfLoopRegionIterArgs` (`Analyses/Utils.cpp:257-271`) is that position MINUS ONE.
    /// ⛔ TRAP: the `else` arm's `isa<>(def)` is bare on a possibly-null op and therefore asserts.
    pub(crate) fn process_work_list(&mut self, defs: Definitions<'_>) {
        while let Some(val) = self.worklist.pop_front() {
            let influence = self.influence_type(val);
            let carried_at = defs
                .for_arg_of(val)
                .and_then(|(for_op, pos)| pos.checked_sub(1).map(|index| (for_op, index)));
            // ⭐ ONE FEED LIST PER ROUND, IN THE REFERENCE'S OWN ORDER, because e040 takes `&mut self`
            // and the ops the positions are read from are borrowed from `defs`.
            let mut feeds: Vec<Val> = Vec::new();
            match carried_at {
                Some((for_op, index)) => {
                    if let Op::Sentient(sentient::Op::For { carried, .. }) = for_op {
                        // ⛔ `.get`: `getIterOperands()[index]` past the end is an out-of-bounds read
                        // there and there is nothing to propagate to here.
                        feeds.extend(carried.get(index).map(|carried| carried.init));
                    }
                    feeds.extend(terminator(for_op).and_then(|yielded| yielded.get(index).copied()));
                }
                None => {
                    let Some(def) = defs.of(val) else {
                        todo!(
                            "processWorkList: {val:?} has no defining op and is no loop iter_arg, so \
                             `isa<sentient::AddOp, sentient::SubOp>(def)` \
                             (EnhancedDeadVariableElimination.cpp:481) is a bare isa<> on a null op"
                        );
                    };
                    match def {
                        Op::Sentient(
                            sentient::Op::ScalarAdd { lhs, rhs, .. }
                            | sentient::Op::ScalarSub { lhs, rhs, .. },
                        ) => feeds.extend([*lhs, *rhs]),
                        Op::Sentient(sentient::Op::ScalarCopy { input, .. }) => feeds.push(*input),
                        // ⭐ THE LOWER RUNG'S SPELLING OF A LOCAL REGION IS HERE TOO, and reachable:
                        // [`dialects::regions`] is total and raises a shared dialect's body, unlike
                        // the borrowing [`dialects::regions_ref`].
                        Op::Sentient(sentient::Op::For { .. } | sentient::Op::If { .. })
                        | Op::UniformRegions(_)
                        | Op::Uniform(uniform::Op::UniformizeRegions { .. }) => {
                            let results = dialects::results(def);
                            let Some(at) = results.iter().position(|result| *result == val) else {
                                todo!(
                                    "processWorkList: DT_CHECK(val_index != -1) \
                                     (EnhancedDeadVariableElimination.cpp:497) — {val:?} is not among \
                                     the results of the op that defines it"
                                );
                            };
                            for region in dialects::regions(def) {
                                // The block's terminator is its last op, and its operand list is what
                                // `getOperand(val_index)` indexes.
                                let yielded = region.last().map(dialects::operands).unwrap_or_default();
                                feeds.extend(yielded.get(at).copied());
                            }
                            if let Op::Sentient(sentient::Op::For { carried, .. }) = def {
                                feeds.extend(carried.get(at).map(|carried| carried.arg));
                            }
                        }
                        // ⭐ EVERY OTHER DEFINING OP ENDS THE CHAIN — the reference's `if`/`else if`
                        // has no final `else` (`:479-514`).
                        _ => {}
                    }
                }
            }
            for feed in feeds {
                let origin = origin_of(feed, defs);
                self.add_to_work_list_and_update_assignment(
                    feed,
                    &origin,
                    influence,
                    WorklistAdd::Push,
                );
            }
        }
    }

    /// Replaces: e301_initializeAssignmentForAnOperation
    ///
    /// Seeds the assignments from one op: every mask, address, increment and endpoint operand is a
    /// memory influencer, while a loop's induction variable and non-constant bound and a
    /// `sentient.if`'s two comparands are control-flow ones.
    ///
    /// ⛔ TRAP: a maskless `sentient.vector_ternary` reaches e040 with a NULL `Value` — `:557-559`
    /// guards nothing, unlike the `mac` arm's `if (mask)` — and `isa<BlockArgument>(null)` asserts.
    pub(crate) fn initialize_assignment_for_an_operation(
        &mut self,
        op: &Op,
        unit: DfirUnit,
        defs: Definitions<'_>,
    ) {
        let Op::Sentient(inner) = op else {
            return;
        };
        let mut seeds: Vec<(Val, Influence, WorklistAdd)> = Vec::new();
        let memory = |vals: &[Val]| {
            vals.iter()
                .map(|val| (*val, Influence::Memory, WorklistAdd::Push))
                .collect::<Vec<_>>()
        };
        match inner {
            sentient::Op::VectorMac {
                mask,
                xrf_write_ptr,
                xrf_read_ptr,
                ..
            } => {
                if !matches!(unit, DfirUnit::PtRow(_) | DfirUnit::Sfp | DfirUnit::Pe) {
                    todo!(
                        "initializeAssignmentForAnOperation: \
                         DT_CHECK(is_any_of(current_unit_type_, PT, SFP, PE)) \
                         (EnhancedDeadVariableElimination.cpp:529) — a sentient.vector_mac on \
                         {unit:?}"
                    );
                }
                // ⭐ WRITE POINTER THEN READ POINTER, which is the `pointers(..)` operand order
                // (`SentientOps.td:234`) `getPointers()` hands back.
                let pointers: Vec<Val> = [*xrf_write_ptr, *xrf_read_ptr]
                    .into_iter()
                    .flatten()
                    .collect();
                seeds.extend(memory(mask.as_slice()));
                seeds.extend(memory(&pointers));
            }
            sentient::Op::VectorBinary { mask, .. } | sentient::Op::VectorUnary { mask, .. } => {
                seeds.extend(memory(&[*mask]));
            }
            sentient::Op::VectorTernary { mask, .. } => {
                let Some(mask) = *mask else {
                    todo!(
                        "initializeAssignmentForAnOperation: a maskless sentient.vector_ternary \
                         reaches addToWorkListAndUpdateAssignment with a null Value \
                         (EnhancedDeadVariableElimination.cpp:557-559), and `isa<BlockArgument>` on \
                         one asserts"
                    );
                };
                seeds.extend(memory(&[mask]));
            }
            sentient::Op::SetMask { mask_value, .. } => seeds.extend(memory(&[*mask_value])),
            sentient::Op::For { iv, bound, .. } => {
                // ⛔ `RecordOnly`: the induction variable is `add = false`, the pass's ONE such call.
                seeds.push((*iv, Influence::ControlFlow, WorklistAdd::RecordOnly));
                match defs.of(*bound) {
                    Some(Op::Sentient(sentient::Op::ScalarConstant { .. })) => {}
                    Some(_) => seeds.push((*bound, Influence::ControlFlow, WorklistAdd::Push)),
                    None => todo!(
                        "initializeAssignmentForAnOperation: \
                         `isa<sentient::ConstantOp>(bound.getDefiningOp())` \
                         (EnhancedDeadVariableElimination.cpp:571) is a bare isa<> and {bound:?} is a \
                         region argument, so `getDefiningOp()` is null and it asserts"
                    ),
                }
            }
            sentient::Op::If { lhs, rhs, .. } => seeds.extend([
                (*lhs, Influence::ControlFlow, WorklistAdd::Push),
                (*rhs, Influence::ControlFlow, WorklistAdd::Push),
            ]),
            sentient::Op::LoadAndSend {
                mutable_addr,
                immutable_addr,
                increment,
                consumer,
                ..
            } => match unit {
                DfirUnit::L0lu | DfirUnit::Lxlu | DfirUnit::L3su => seeds.extend(memory(&[
                    *mutable_addr,
                    *immutable_addr,
                    *increment,
                    consumer.val(),
                ])),
                _ => self.misplaced.push(MisplacedOp::of(op, unit)),
            },
            sentient::Op::ReceiveAndStore {
                mutable_addr,
                immutable_addr,
                increment,
                producer,
                multicast_info,
                ..
            } => match unit {
                DfirUnit::L0su | DfirUnit::Lxsu | DfirUnit::L3lu => {
                    seeds.extend(memory(&[
                        *mutable_addr,
                        *immutable_addr,
                        *increment,
                        producer.val(),
                    ]));
                    seeds.extend(memory(multicast_info.as_slice()));
                }
                _ => self.misplaced.push(MisplacedOp::of(op, unit)),
            },
            sentient::Op::LoadAndStore {
                src_mutable_addr,
                src_immutable_addr,
                src_inc,
                dst_mutable_addr,
                dst_immutable_addr,
                dst_inc,
                multicast_info,
                ..
            } => match unit {
                // ⭐ SOURCE AND DESTINATION INTERLEAVED BY FIELD, not source-then-destination
                // (`:612-624`).
                DfirUnit::L3su | DfirUnit::L3lu => {
                    seeds.extend(memory(&[
                        *src_mutable_addr,
                        *dst_mutable_addr,
                        *src_immutable_addr,
                        *dst_immutable_addr,
                        *src_inc,
                        *dst_inc,
                    ]));
                    seeds.extend(memory(multicast_info.as_slice()));
                }
                _ => self.misplaced.push(MisplacedOp::of(op, unit)),
            },
            sentient::Op::LoadAndExtractScalar {
                mutable_addr,
                immutable_addr,
                increment,
                consumer,
                ..
            } => match unit {
                DfirUnit::Lxlu => seeds.extend(memory(&[
                    *mutable_addr,
                    *immutable_addr,
                    *increment,
                    consumer.val(),
                ])),
                _ => self.misplaced.push(MisplacedOp::of(op, unit)),
            },
            sentient::Op::LoadComputeAndSend {
                mutable_addr,
                immutable_addr,
                increment,
                consumer,
                ..
            } => {
                // ⛔ A `DT_CHECK` AND NOT AN `emitError` (`:665`): this one aborts rather than failing
                // the pass, so it is a `todo!` and not a [`MisplacedOp`].
                if !matches!(unit, DfirUnit::Lxlu) {
                    todo!(
                        "initializeAssignmentForAnOperation: \
                         DT_CHECK(current_unit_type_ == LXLU) \
                         (EnhancedDeadVariableElimination.cpp:665) — a \
                         sentient.load_compute_and_send on {unit:?}"
                    );
                }
                seeds.extend(memory(&[
                    *mutable_addr,
                    *immutable_addr,
                    *increment,
                    consumer.val(),
                ]));
            }
            sentient::Op::Splat { input, .. } => seeds.extend(memory(&[*input])),
            // ⭐ EVERY OTHER OP SEEDS NOTHING — the reference's chain has no final `else` (`:527-682`).
            _ => {}
        }
        for (val, influence, add) in seeds {
            let origin = origin_of(val, defs);
            self.add_to_work_list_and_update_assignment(val, &origin, influence, add);
        }
    }
}

// crustify:todo: e437_initializeWorkList
//   authority : dcc/src/Transform/Sentient/EnhancedDeadVariableElimination.cpp:685  (4 body lines, level 2)
//   original  : void EnhancedDeadVariableEliminationPass::initializeWorkList()
//   calls     : e301_initializeAssignmentForAnOperation

// crustify:todo: e498_updateForOperation
//   authority : dcc/src/Transform/Sentient/EnhancedDeadVariableElimination.cpp:127  (141 body lines, level 3)
//   original  : void EnhancedDeadVariableEliminationPass::updateForOperation( ForOp &for_op, std::set<int> deleted_pos)
//   calls     : e041_getInfluenceType, e252_size, e422_insert

// crustify:todo: e499_updateIfOperation
//   authority : dcc/src/Transform/Sentient/EnhancedDeadVariableElimination.cpp:272  (49 body lines, level 3)
//   original  : void EnhancedDeadVariableEliminationPass::updateIfOperation( mlir::sentient::IfOp &if_op)
//   calls     : e041_getInfluenceType, e422_insert

// crustify:todo: e500_updateUniformizeRegionsOperation
//   authority : dcc/src/Transform/Sentient/EnhancedDeadVariableElimination.cpp:325  (38 body lines, level 3)
//   original  : void EnhancedDeadVariableEliminationPass::updateUniformizeRegionsOperation( mlir::uniform::UniformizeRegionsOp &uniform_op)
//   calls     : e041_getInfluenceType, e422_insert

// crustify:todo: e557_exploreOperation
//   authority : dcc/src/Transform/Sentient/EnhancedDeadVariableElimination.cpp:367  (57 body lines, level 4)
//   original  : void EnhancedDeadVariableEliminationPass::exploreOperation(Operation *op)
//   calls     : e041_getInfluenceType, e498_updateForOperation, e499_updateIfOperation, e500_updateUniformizeRegionsOperation, e558_exploreBlock

// crustify:todo: e558_exploreBlock
//   authority : dcc/src/Transform/Sentient/EnhancedDeadVariableElimination.cpp:426  (23 body lines, level 4)
//   original  : void EnhancedDeadVariableEliminationPass::exploreBlock(Block *block)
//   calls     : e557_exploreOperation

// crustify:todo: e596_updateProgramUnit
//   authority : dcc/src/Transform/Sentient/EnhancedDeadVariableElimination.cpp:451  (3 body lines, level 5)
//   original  : void EnhancedDeadVariableEliminationPass::updateProgramUnit()
//   calls     : e558_exploreBlock

// crustify:todo: e622_runOn
//   authority : dcc/src/Transform/Sentient/EnhancedDeadVariableElimination.cpp:710  (36 body lines, level 6)
//   original  : void EnhancedDeadVariableEliminationPass::runOn(dataflow::ProgramUnitOp unit)
//   calls     : e043_removeConstantIterArgs, e300_processWorkList, e437_initializeWorkList, e596_updateProgramUnit

// crustify:todo: e639_runOnOperation
//   authority : dcc/src/Transform/Sentient/EnhancedDeadVariableElimination.cpp:58  (6 body lines, level 7)
//   original  : void EnhancedDeadVariableEliminationPass::runOnOperation()
//   calls     : e622_runOn

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::islands::dataflow_ir::ty::ScalarTy;
    use crate::islands::sentient::dialects::sentient::{Carried, Reg, RegType};

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

    /// `sentient.nop` — an owner that tolerates no disagreement.
    fn nop() -> Op {
        Op::Sentient(sentient::Op::Nop { dbg_name: None })
    }

    /// One `iter_args` position.
    fn carried(init: u32, arg: u32, result: u32) -> Carried {
        Carried {
            init: Val(init),
            arg: Val(arg),
            result: Val(result),
            reg: Reg {
                locale: RegType::Unknown,
                index: None,
            },
            program_header: false,
            element_size: None,
        }
    }

    /// `sentient.for %iv = %bound iter_args(..) { body }`.
    fn for_op(iv: u32, bound: u32, iter_args: Vec<Carried>, body: Vec<Op>) -> Op {
        Op::Sentient(sentient::Op::For {
            iv: Val(iv),
            bound: Val(bound),
            carried: iter_args,
            dbg_name: None,
            body,
        })
    }

    /// `sentient.yield %a, %b`.
    fn yield_op(results: Vec<Val>) -> Op {
        Op::Sentient(sentient::Op::Yield { results })
    }

    /// `%out = sentient.scalar_add %lhs, %rhs`.
    fn add(lhs: u32, rhs: u32, result: u32) -> Op {
        Op::Sentient(sentient::Op::ScalarAdd {
            lhs: Val(lhs),
            rhs: Val(rhs),
            result: Val(result),
            reg: None,
            element_size: None,
            ty: ScalarTy::Index,
        })
    }

    /// Every `sentient.yield`'s operands, in walk order.
    fn yields(scope: &[Op]) -> Vec<Vec<Val>> {
        let mut out = Vec::new();
        for op in scope {
            for region in regions_of(op) {
                out.extend(yields(region));
            }
            if let Op::Sentient(sentient::Op::Yield { results }) = op {
                out.push(results.clone());
            }
        }
        out
    }

    #[test]
    fn a_second_disagreeing_influence_conflicts_unless_the_owner_is_a_constant() {
        let mut pass = EnhancedDeadVariableElimination::default();
        let plain = Origin::Defined(nop());
        pass.add_to_work_list_and_update_assignment(
            Val(7),
            &plain,
            Influence::Memory,
            WorklistAdd::Push,
        );
        assert_eq!(pass.worklist.len(), 1);
        assert!(pass.conflicts().is_empty());
        pass.add_to_work_list_and_update_assignment(
            Val(7),
            &plain,
            Influence::ControlFlow,
            WorklistAdd::Push,
        );
        assert_eq!(pass.conflicts().len(), 1);
        assert_eq!(pass.conflicts()[0].recorded, Influence::Memory);
        // A `sentient.scalar_constant` owner tolerates the same disagreement, and `add = false` keeps
        // the value out of the worklist.
        let konst = Origin::Defined(constant(Val(8), 3));
        pass.add_to_work_list_and_update_assignment(
            Val(8),
            &konst,
            Influence::Memory,
            WorklistAdd::RecordOnly,
        );
        pass.add_to_work_list_and_update_assignment(
            Val(8),
            &konst,
            Influence::ControlFlow,
            WorklistAdd::Push,
        );
        assert_eq!(pass.conflicts().len(), 1);
        assert_eq!(pass.worklist.len(), 1);
        // A `dataflow.program_unit` iter_arg is never recorded at all.
        pass.add_to_work_list_and_update_assignment(
            Val(9),
            &Origin::ProgramUnitArg,
            Influence::Memory,
            WorklistAdd::Push,
        );
        assert_eq!(pass.influence_type(Val(9)), Influence::None);
    }

    #[test]
    fn an_unrecorded_value_has_no_influence() {
        let mut pass = EnhancedDeadVariableElimination::default();
        pass.add_to_work_list_and_update_assignment(
            Val(1),
            &Origin::RegionArg(nop()),
            Influence::ControlFlow,
            WorklistAdd::Push,
        );
        assert_eq!(pass.influence_type(Val(1)), Influence::ControlFlow);
        assert_eq!(pass.influence_type(Val(2)), Influence::None);
    }

    #[test]
    fn print_assignments_writes_the_owner_then_its_influence() {
        let mut pass = EnhancedDeadVariableElimination::default();
        pass.add_to_work_list_and_update_assignment(
            Val(1),
            &Origin::Defined(constant(Val(1), 4)),
            Influence::Memory,
            WorklistAdd::Push,
        );
        pass.add_to_work_list_and_update_assignment(
            Val(2),
            &Origin::RegionArg(nop()),
            Influence::ControlFlow,
            WorklistAdd::RecordOnly,
        );
        let mut out = String::new();
        pass.print_assignments(&mut out);
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(lines.len(), 4);
        assert_eq!(lines[1], "influence: memory");
        assert_eq!(lines[3], "influence: controlflow");
    }

    #[test]
    fn a_yielded_back_iter_arg_is_replaced_by_its_init_inside_and_outside_the_loop() {
        let inner = for_op(
            30,
            31,
            vec![carried(20, 21, 22)],
            vec![yield_op(vec![Val(21)])],
        );
        let outer = for_op(
            40,
            41,
            vec![carried(10, 11, 12)],
            vec![inner, yield_op(vec![Val(11)])],
        );
        // The trailing loop reads the outer loop's result, which is a use OUTSIDE that loop's region.
        let mut scope = vec![outer, for_op(50, 12, Vec::new(), Vec::new())];
        remove_constant_iter_args(&mut scope);
        assert_eq!(yields(&scope), vec![vec![Val(20)], vec![Val(10)]]);
        let bounds: Vec<Val> = scope
            .iter()
            .filter_map(|op| match op {
                Op::Sentient(sentient::Op::For { bound, .. }) => Some(*bound),
                _ => None,
            })
            .collect();
        assert_eq!(bounds, vec![Val(41), Val(10)]);
    }

    /// e300 propagates off a loop-carried argument to BOTH its init and its yield operand, and then
    /// off the `sentient.scalar_add` that yield names to both of that add's inputs.
    #[test]
    fn process_work_list_follows_the_carried_arg_then_its_adds_inputs() {
        let scope = vec![
            constant(Val(0), 4),
            constant(Val(1), 0),
            constant(Val(9), 1),
            for_op(
                2,
                0,
                vec![carried(1, 3, 4)],
                vec![add(3, 9, 8), yield_op(vec![Val(8)])],
            ),
        ];
        let regions: [&[Op]; 1] = [&scope];
        let defs = Definitions::from_innermost(&regions);

        let mut pass = EnhancedDeadVariableElimination::default();
        let origin = origin_of(Val(3), defs);
        pass.add_to_work_list_and_update_assignment(
            Val(3),
            &origin,
            Influence::Memory,
            WorklistAdd::Push,
        );
        pass.process_work_list(defs);

        for val in [Val(1), Val(3), Val(8), Val(9)] {
            assert_eq!(pass.influence_type(val), Influence::Memory, "{val:?}");
        }
        // ⛔ THE BOUND AND THE INDUCTION VARIABLE ARE NOT REACHED — e300 walks what feeds a value,
        // and neither feeds the carried argument.
        assert_eq!(pass.influence_type(Val(0)), Influence::None);
        assert_eq!(pass.influence_type(Val(2)), Influence::None);
        assert!(pass.conflicts().is_empty());
    }

    /// e301 seeds a `sentient.set_mask`'s value as memory and a `sentient.for`'s induction variable as
    /// control flow — the latter recorded WITHOUT being queued, which is the pass's one `add = false`.
    #[test]
    fn initialize_assignment_for_an_operation_seeds_masks_and_loop_control() {
        let set_mask = Op::Sentient(sentient::Op::SetMask {
            mask_value: Val(6),
            dbg_name: None,
        });
        let loop_op = for_op(2, 5, vec![carried(1, 3, 4)], vec![yield_op(vec![Val(3)])]);
        let scope = vec![
            constant(Val(0), 1),
            constant(Val(1), 0),
            constant(Val(6), 7),
            add(0, 1, 5),
            set_mask.clone(),
            loop_op.clone(),
        ];
        let regions: [&[Op]; 1] = [&scope];
        let defs = Definitions::from_innermost(&regions);

        let mut pass = EnhancedDeadVariableElimination::default();
        pass.initialize_assignment_for_an_operation(&set_mask, DfirUnit::Sfp, defs);
        pass.initialize_assignment_for_an_operation(&loop_op, DfirUnit::Sfp, defs);

        assert_eq!(pass.influence_type(Val(6)), Influence::Memory);
        assert_eq!(pass.influence_type(Val(2)), Influence::ControlFlow);
        assert_eq!(pass.influence_type(Val(5)), Influence::ControlFlow);
        assert_eq!(pass.worklist, VecDeque::from([Val(6), Val(5)]));
        assert!(pass.misplaced().is_empty());
    }
}
