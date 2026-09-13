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
use std::collections::{BTreeMap, BTreeSet, VecDeque};

use crate::islands::dataflow_ir::ty::GenericComp;
use crate::islands::sentient::dialects::{
    self as dialects, Definitions, Op, UniformRegions, Val, sentient, symbol, uniform,
};
use crate::islands::sentient::print;

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

/// WHICH TRANSFER OP TURNED UP IN A UNIT THAT CANNOT HOLD ONE — e301's four `emitError` arms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Misplaced {
    /// `sentient.load_and_send` outside `l0lu`/`lxlu`/`l3su`.
    LoadAndSend,
    /// `sentient.receive_and_store` outside `l0su`/`lxsu`/`l3lu`.
    ReceiveAndStore,
    /// `sentient.load_and_store` outside `l3su`/`l3lu`.
    LoadAndStore,
    /// `sentient.load_and_extract_scalar` outside `lxlu`.
    LoadAndExtractScalar,
}

impl Misplaced {
    /// The message the reference emitted (`EnhancedDeadVariableElimination.cpp:650`, `:679`, `:698`,
    /// `:717` of the extract) — ⭐ A TABLE ON THE ENUM, as [`Influence::spelling`] is.
    #[must_use]
    pub(crate) const fn message(self) -> &'static str {
        match self {
            Misplaced::LoadAndSend => "Not expecting load_and_send in other units",
            Misplaced::ReceiveAndStore => "Not expecting receive_and_store in other units",
            Misplaced::LoadAndStore => "Not expecting load_and_store in other units",
            Misplaced::LoadAndExtractScalar => {
                "Not expecting load_and_extract_scalar in other units"
            }
        }
    }
}

/// `op->emitError(..)` + `signalPassFailure()` AS DATA — the same shape as [`InfluenceConflict`], and
/// for the same reason: the flag is the round, and keeping it as data says WHICH op and WHERE.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MisplacedTransfer {
    /// Which of the four ops, and so which message.
    pub(crate) op: Misplaced,
    /// `current_unit_type_` — the unit that cannot hold it.
    pub(crate) unit: GenericComp,
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
    /// The other `signalPassFailure()` this pass makes — e301's misplaced transfers.
    misplaced: Vec<MisplacedTransfer>,
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

    /// Every transfer op found in a unit that cannot hold one — e301's other `signalPassFailure()`.
    #[must_use]
    pub(crate) fn misplaced(&self) -> &[MisplacedTransfer] {
        &self.misplaced
    }

    /// [`Self::add_to_work_list_and_update_assignment`] with the value's own [`Origin`], which is
    /// what e300 and e301 both have to supply — see [`origin_of`].
    fn propagate(
        &mut self,
        val: Val,
        influence: Influence,
        add: WorklistAdd,
        defs: Definitions<'_>,
    ) {
        if let Some(origin) = origin_of(val, defs) {
            self.add_to_work_list_and_update_assignment(val, &origin, influence, add);
        }
    }

    /// Replaces: e300_processWorkList
    ///
    /// Drains the worklist, handing each value's influence to whatever decides it: a loop iter arg to
    /// the loop's init operand and its yield, an add/sub/copy result to its inputs, and a
    /// region-carrying op's result to that result's position in EVERY region's yield.
    ///
    /// ⛔ TRAP: `DT_CHECK(val_index != -1)` is unwritable — `val` reached this arm BY having `def` as
    /// its defining op, so [`dialects::results`] cannot fail to find it.
    /// ⛔ TRAP: `val.getDefiningOp()` is NULL for a region argument of anything but a `sentient.for`,
    /// and the reference's `isa<>` on it ASSERTS rather than answering false; this port records
    /// nothing there.
    pub(crate) fn process_work_list(&mut self, defs: Definitions<'_>) {
        while let Some(val) = self.worklist.pop_front() {
            let influence = self.influence_type(val);
            // `getIndexOfLoopRegionIterArgs` (`Analyses/Utils.cpp:257-271`): position 0 is the
            // induction variable, which is NOT an iter arg — hence the `-1` the reference returns.
            let iter_arg = defs
                .for_arg_of(val)
                .and_then(|(for_op, at)| at.checked_sub(1).map(|index| (for_op, index)));
            if let Some((Op::Sentient(sentient::Op::For { carried, body, .. }), index)) = iter_arg {
                if let Some(init) = carried.get(index).map(|entry| entry.init) {
                    self.propagate(init, influence, WorklistAdd::Push, defs);
                }
                if let Some(yielded) = terminator_operands(body).get(index).copied() {
                    self.propagate(yielded, influence, WorklistAdd::Push, defs);
                }
                continue;
            }
            let Some(def) = defs.of(val) else { continue };
            match def {
                Op::Sentient(
                    sentient::Op::ScalarAdd { lhs, rhs, .. }
                    | sentient::Op::ScalarSub { lhs, rhs, .. },
                ) => {
                    let (lhs, rhs) = (*lhs, *rhs);
                    self.propagate(lhs, influence, WorklistAdd::Push, defs);
                    self.propagate(rhs, influence, WorklistAdd::Push, defs);
                }
                Op::Sentient(sentient::Op::ScalarCopy { input, .. }) => {
                    let input = *input;
                    self.propagate(input, influence, WorklistAdd::Push, defs);
                }
                // ⛔ THE RAISED SPELLING ONLY: `uniform.uniformize_regions` at THIS rung is
                // [`Op::UniformRegions`], and the lower-rung [`Op::Uniform`] one carries a body of
                // the rung below whose terminator is not one of this rung's values.
                Op::Sentient(sentient::Op::For { .. } | sentient::Op::If { .. })
                | Op::UniformRegions(UniformRegions::UniformizeRegions { .. }) => {
                    let Some(at) = dialects::results(def)
                        .iter()
                        .position(|result| *result == val)
                    else {
                        continue;
                    };
                    for region in dialects::regions_ref(def) {
                        if let Some(yielded) = terminator_operands(region).get(at).copied() {
                            self.propagate(yielded, influence, WorklistAdd::Push, defs);
                        }
                    }
                    if let Op::Sentient(sentient::Op::For { carried, .. }) = def {
                        if let Some(arg) = carried.get(at).map(|entry| entry.arg) {
                            self.propagate(arg, influence, WorklistAdd::Push, defs);
                        }
                    }
                }
                _ => {}
            }
        }
    }

    /// Replaces: e301_initializeAssignmentForAnOperation
    ///
    /// Seeds the worklist from one op: a compute op's mask and XRF pointers and every transfer's
    /// address, increment and peer operands influence MEMORY, while a loop's induction variable and
    /// non-constant bound and a `sentient.if`'s two comparands influence CONTROL FLOW.
    ///
    /// ⛔ TRAP: the four `emitError` + `signalPassFailure()` arms become [`MisplacedTransfer`]
    /// records, so the port says which op sat in the wrong unit instead of refusing.
    /// ⛔ TRAP: both `DT_CHECK`s on `current_unit_type_` — a MAC outside PT/SFP/PE, a
    /// `load_compute_and_send` outside LXLU — record nothing and gate nothing, so this port is total
    /// and seeds the operands regardless.
    /// ⛔ TRAP: `sentient.vector_ternary`'s `$mask` is OPTIONAL and the reference hands it over
    /// unguarded, where the MAC's is guarded by `if (mask)`; an absent one seeds nothing here.
    pub(crate) fn initialize_assignment_for_an_operation(
        &mut self,
        op: &Op,
        unit: GenericComp,
        defs: Definitions<'_>,
    ) {
        let memory = |this: &mut Self, val: Val| {
            this.propagate(val, Influence::Memory, WorklistAdd::Push, defs);
        };
        match op {
            Op::Sentient(sentient::Op::VectorMac {
                mask,
                xrf_write_ptr,
                xrf_read_ptr,
                ..
            }) => {
                // ⭐ PT masking is not connected to individual MACs at SentientIR, which is why the
                // mask is the one operand guarded here.
                for val in mask.iter().chain(xrf_write_ptr).chain(xrf_read_ptr) {
                    memory(self, *val);
                }
            }
            Op::Sentient(
                sentient::Op::VectorBinary { mask, .. } | sentient::Op::VectorUnary { mask, .. },
            ) => memory(self, *mask),
            Op::Sentient(sentient::Op::VectorTernary { mask, .. }) => {
                if let Some(mask) = mask {
                    memory(self, *mask);
                }
            }
            Op::Sentient(sentient::Op::SetMask { mask_value, .. }) => memory(self, *mask_value),
            Op::Sentient(sentient::Op::For { iv, bound, .. }) => {
                // ⭐ `add = false`: the induction variable is recorded but never explored.
                self.propagate(*iv, Influence::ControlFlow, WorklistAdd::RecordOnly, defs);
                if !matches!(
                    defs.of(*bound),
                    Some(Op::Sentient(sentient::Op::ScalarConstant { .. }))
                ) {
                    self.propagate(*bound, Influence::ControlFlow, WorklistAdd::Push, defs);
                }
            }
            Op::Sentient(sentient::Op::If { lhs, rhs, .. }) => {
                self.propagate(*lhs, Influence::ControlFlow, WorklistAdd::Push, defs);
                self.propagate(*rhs, Influence::ControlFlow, WorklistAdd::Push, defs);
            }
            Op::Sentient(sentient::Op::LoadAndSend {
                mutable_addr,
                immutable_addr,
                increment,
                consumer,
                ..
            }) => {
                if matches!(
                    unit,
                    GenericComp::L0lu | GenericComp::Lxlu | GenericComp::L3su
                ) {
                    for val in [*mutable_addr, *immutable_addr, *increment, consumer.val()] {
                        memory(self, val);
                    }
                } else {
                    self.misplaced.push(MisplacedTransfer {
                        op: Misplaced::LoadAndSend,
                        unit,
                    });
                }
            }
            Op::Sentient(sentient::Op::ReceiveAndStore {
                mutable_addr,
                immutable_addr,
                increment,
                producer,
                multicast_info,
                ..
            }) => {
                if matches!(
                    unit,
                    GenericComp::L0su | GenericComp::Lxsu | GenericComp::L3lu
                ) {
                    for val in [*mutable_addr, *immutable_addr, *increment, producer.val()] {
                        memory(self, val);
                    }
                    if let Some(multicast) = multicast_info {
                        memory(self, *multicast);
                    }
                } else {
                    self.misplaced.push(MisplacedTransfer {
                        op: Misplaced::ReceiveAndStore,
                        unit,
                    });
                }
            }
            Op::Sentient(sentient::Op::LoadAndStore {
                src_mutable_addr,
                src_immutable_addr,
                src_inc,
                dst_mutable_addr,
                dst_immutable_addr,
                dst_inc,
                multicast_info,
                ..
            }) => {
                if matches!(unit, GenericComp::L3su | GenericComp::L3lu) {
                    for val in [
                        *src_mutable_addr,
                        *dst_mutable_addr,
                        *src_immutable_addr,
                        *dst_immutable_addr,
                        *src_inc,
                        *dst_inc,
                    ] {
                        memory(self, val);
                    }
                    if let Some(multicast) = multicast_info {
                        memory(self, *multicast);
                    }
                } else {
                    self.misplaced.push(MisplacedTransfer {
                        op: Misplaced::LoadAndStore,
                        unit,
                    });
                }
            }
            Op::Sentient(sentient::Op::LoadAndExtractScalar {
                mutable_addr,
                immutable_addr,
                increment,
                consumer,
                ..
            }) => {
                if matches!(unit, GenericComp::Lxlu) {
                    for val in [*mutable_addr, *immutable_addr, *increment, consumer.val()] {
                        memory(self, val);
                    }
                } else {
                    self.misplaced.push(MisplacedTransfer {
                        op: Misplaced::LoadAndExtractScalar,
                        unit,
                    });
                }
            }
            Op::Sentient(sentient::Op::LoadComputeAndSend {
                mutable_addr,
                immutable_addr,
                increment,
                consumer,
                ..
            }) => {
                for val in [*mutable_addr, *immutable_addr, *increment, consumer.val()] {
                    memory(self, val);
                }
            }
            Op::Sentient(sentient::Op::Splat { input, .. }) => memory(self, *input),
            _ => {}
        }
    }
}

/// WHERE A VALUE COMES FROM, READ OUT OF THE CALLER'S SCOPE — the `getDefiningOp()` /
/// `getOwner()->getParentOp()` split [`Origin`] records and e300/e301 must therefore supply.
///
/// ⛔ `None` WHERE THE SCOPE BINDS IT NOWHERE, including a `uniform.uniformize_regions` region
/// argument: this island names that parent only through its unit list, and there is no op to blame.
#[must_use]
fn origin_of(val: Val, defs: Definitions<'_>) -> Option<Origin> {
    if let Some(def) = defs.of(val) {
        return Some(Origin::Defined(def.clone()));
    }
    if defs.program_unit_units_of(val).is_some() {
        return Some(Origin::ProgramUnitArg);
    }
    defs.for_arg_of(val)
        .map(|(for_op, _)| Origin::RegionArg(for_op.clone()))
}

/// `region.front().getTerminator()`'s OPERANDS, at either rung's spelling of a yield.
///
/// ⛔ EMPTY, NOT A PANIC, FOR A REGION WITH NO TERMINATOR — a `sentient.if` always has two regions
/// and its else region is routinely EMPTY, where `front().getTerminator()` hands the reference a null
/// `Operation *`.
#[must_use]
fn terminator_operands(region: &[Op]) -> Vec<Val> {
    region
        .iter()
        .rev()
        .find_map(|op| match op {
            Op::Sentient(sentient::Op::Yield { results }) => Some(results.clone()),
            Op::Uniform(uniform::Op::Yield { operands }) => Some(operands.clone()),
            _ => None,
        })
        .unwrap_or_default()
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

impl EnhancedDeadVariableElimination {
    /// Replaces: e437_initializeWorkList
    ///
    /// Seeds the assignments and the worklist from every operation of the program unit, pre-order —
    /// `current_unit_->walk<WalkOrder::PreOrder>(initializeAssignmentForAnOperation)`.
    ///
    /// ⭐ THE REGION STACK IS THE WALK: [`Definitions`] resolves a value against the innermost
    /// enclosing regions first, so each level pushes its own block ahead of `enclosing` before it
    /// recurses. `current_unit_` is the body plus the unit it runs on, both handed in.
    pub(crate) fn initialize_work_list(
        &mut self,
        unit_body: &[Op],
        unit: GenericComp,
        enclosing: &[&[Op]],
    ) {
        let mut regions: Vec<&[Op]> = Vec::with_capacity(enclosing.len() + 1);
        regions.push(unit_body);
        regions.extend_from_slice(enclosing);
        for op in unit_body {
            self.initialize_assignment_for_an_operation(
                op,
                unit,
                Definitions::from_innermost(&regions),
            );
            for region in dialects::regions_ref(op) {
                self.initialize_work_list(region, unit, &regions);
            }
        }
    }
}

/// `deleted_pos.find(i) == deleted_pos.end()` APPLIED TO ONE PARALLEL LIST — the filter e498, e499 and
/// e500 rebuild every one of their arrays through.
fn retain_positions<T>(items: &mut Vec<T>, deleted: &BTreeSet<usize>) {
    let mut at = 0;
    items.retain(|_| {
        let keep = !deleted.contains(&at);
        at += 1;
        keep
    });
}

impl EnhancedDeadVariableElimination {
    /// Replaces: e498_updateForOperation
    ///
    /// Drops every `iter_args` position whose result AND body argument both have no influence — or the
    /// positions `deleted_pos` names, an EMPTY set being the reference's own "not provided" (`:129`).
    ///
    /// ⛔ TRAP: THE REFERENCE MISALIGNS ITS OWN REGISTER ARRAYS BY ONE unless they are in the
    /// `1 + 2 * num_results` layout (`:140-141`) — the else arm reads surviving position `i` at
    /// `regLocales[i]`, which is the BOUND's slot, and writes an array with no bound slot at all
    /// (`:187-192`). [`sentient::Carried::reg`] beside `bound_reg` makes both unwritable.
    /// ⛔ THE SURVIVORS KEEP THEIR OWN VALUES: `create` + `replaceAllUsesWith` + `erase` (`:225-267`)
    /// only renumbers what a collapsed [`sentient::Carried`] drops in place.
    pub(crate) fn update_for_operation(
        &self,
        scope: &mut [Op],
        at: usize,
        deleted_pos: &BTreeSet<usize>,
    ) {
        let Some(Op::Sentient(sentient::Op::For { carried, .. })) = scope.get(at) else {
            return;
        };
        let mut deleted = deleted_pos.clone();
        if deleted_pos.is_empty() {
            for (i, value) in carried.iter().enumerate() {
                if self.influence_type(value.result) == Influence::None
                    && self.influence_type(value.arg) == Influence::None
                {
                    deleted.insert(i);
                }
            }
        }
        if deleted.is_empty() {
            return;
        }
        let dead: Vec<Val> = deleted
            .iter()
            .filter_map(|i| carried.get(*i).map(|value| value.result))
            .collect();
        for result in dead {
            if dialects::use_count(result, scope) != 0 {
                panic!(
                    "DT_CHECK_MSG(The loop's results at the indices to be deleted should have no \
                     uses.) (`EnhancedDeadVariableElimination.cpp:250-253`): {result:?}"
                );
            }
        }
        let Some(Op::Sentient(sentient::Op::For { carried, .. })) = scope.get_mut(at) else {
            return;
        };
        retain_positions(carried, &deleted);
    }

    /// Replaces: e499_updateIfOperation
    ///
    /// Drops every `sentient.if` result with no influence, and its `regLocales`/`regIndices` slot with
    /// it.
    ///
    /// ⛔ THE REGION BODIES ARE UNTOUCHED, `sentient.yield`s INCLUDED: e557 erases the dead yield
    /// operands BEFORE it reaches the parent (`:377-396`), so each region already hands back exactly
    /// the surviving results by the time this runs.
    /// ⛔ ONE SHARED `IRMapping` SPANS BOTH REGIONS of `cloneIfOp`
    /// (`Dialect/Sentient/Utils.cpp:218-228`) — dropped with the clone, which a collapsed
    /// [`sentient::Yielded`] has no need of.
    pub(crate) fn update_if_operation(&self, scope: &mut [Op], at: usize) {
        let Some(Op::Sentient(sentient::Op::If { yielded, .. })) = scope.get_mut(at) else {
            return;
        };
        let deleted: BTreeSet<usize> = yielded
            .iter()
            .enumerate()
            .filter(|(_, value)| self.influence_type(value.result) == Influence::None)
            .map(|(i, _)| i)
            .collect();
        retain_positions(yielded, &deleted);
    }

    /// Replaces: e500_updateUniformizeRegionsOperation
    ///
    /// Drops every `uniform.uniformize_regions` result with no influence, and its `regLocales` slot
    /// with it.
    ///
    /// ⛔ THE RAISED SPELLING ONLY — see e300's note; the lower-rung [`Op::Uniform`] op's terminator
    /// hands back values of the rung below, which this pass has no influence for.
    /// ⛔ `cloneUniformOp` NEVER CARRIES `element_sizes` ONTO THE OP IT BUILDS
    /// (`Dialect/Sentient/Utils.cpp:266-271`): it filters `regIndices` and `regLocales` independently,
    /// each over its own length (`:254-265`), and DISCARDS the whole `element_sizes` array where this
    /// keeps the surviving slots of the fused [`dialects::YieldedReg`] — the same `create`-shaped loss
    /// e497 records and calls discardable.
    pub(crate) fn update_uniformize_regions_operation(&self, scope: &mut [Op], at: usize) {
        let Some(Op::UniformRegions(UniformRegions::UniformizeRegions {
            results, yielded, ..
        })) = scope.get_mut(at)
        else {
            return;
        };
        let deleted: BTreeSet<usize> = results
            .iter()
            .enumerate()
            .filter(|(_, result)| self.influence_type(**result) == Influence::None)
            .map(|(i, _)| i)
            .collect();
        if deleted.is_empty() {
            return;
        }
        retain_positions(results, &deleted);
        retain_positions(yielded, &deleted);
    }
}

/// `dcc::utils::mayHaveSideEffects` (`Analyses/Utils.cpp:666-671`) — the nine ops the reference lists,
/// and every other op has effects.
///
/// ⭐ IN SCOPE AND NOT A `todo!`: it is a closed `isa<>` list over ops this island already spells.
#[must_use]
fn may_have_side_effects(op: &Op) -> bool {
    !matches!(
        op,
        Op::Sentient(
            sentient::Op::ScalarAdd { .. }
                | sentient::Op::ScalarSub { .. }
                | sentient::Op::ScalarConstant { .. }
                | sentient::Op::ScalarCopy { .. }
                | sentient::Op::VectorConstant { .. }
                | sentient::Op::LogicalPort { .. }
        ) | Op::Uniform(uniform::Op::QueryMap { .. } | uniform::Op::DefImmutableMapping { .. })
            | Op::Symbol(symbol::Op::CreateSymbol { .. })
    )
}

/// `isa<sentient::YieldOp, uniform::YieldOp>(op)` (`EnhancedDeadVariableElimination.cpp:377`) — the
/// one op whose operands e557 erases, in either spelling.
#[must_use]
fn yield_operands_mut(op: &mut Op) -> Option<&mut Vec<Val>> {
    match op {
        Op::Sentient(sentient::Op::Yield { results }) => Some(results),
        Op::Uniform(uniform::Op::Yield { operands }) => Some(operands),
        _ => None,
    }
}

impl EnhancedDeadVariableElimination {
    /// WHICH OF A PARENT OP'S YIELDED POSITIONS MAY LOSE THEIR OPERAND — `op->getParentOp()` (`:378`)
    /// asked before the descent, because this island keeps no parent pointers.
    ///
    /// ⭐ THE `sentient.for` ARM NEEDS **BOTH** ITS RESULT AND ITS BODY ARGUMENT TO BE UNINFLUENTIAL
    /// (`:382-387`); every other parent asks about its result alone (`:388-392`).
    #[must_use]
    fn dead_yielded_positions(&self, parent: &Op) -> Vec<bool> {
        if let Op::Sentient(sentient::Op::For { carried, .. }) = parent {
            return carried
                .iter()
                .map(|value| {
                    self.influence_type(value.result) == Influence::None
                        && self.influence_type(value.arg) == Influence::None
                })
                .collect();
        }
        dialects::results(parent)
            .into_iter()
            .map(|result| self.influence_type(result) == Influence::None)
            .collect()
    }

    /// Replaces: e557_exploreOperation
    ///
    /// Explores the op's regions, then either strips the dead operands off a terminator, erases an
    /// unread effect-free op outright, or hands a loop, conditional or uniformized region to its own
    /// position-dropping rewriter.
    ///
    /// ⛔ `parent_dead` IS `op->getParentOp()` (`:378`) PRECOMPUTED — see
    /// [`Self::dead_yielded_positions`]; the top-level block of a program unit has no yield and an
    /// empty slice is its verdict.
    /// ⛔ THE SCOPE IS THE BLOCK, NOT THE MODULE, for `use_empty()` (`:411`) — the widest reach this
    /// walk has, and dominance puts every use of a result in it or under it.
    pub(crate) fn explore_operation(&self, scope: &mut Vec<Op>, at: usize, parent_dead: &[bool]) {
        let Some(op) = scope.get(at) else {
            return;
        };
        // `for (auto &region : op->getRegions())` (`:369-375`) — the empty region and the empty block
        // are one `Vec` here, and [`Self::explore_block`] returns on either.
        let dead = self.dead_yielded_positions(op);
        for region in dialects::regions_mut(&mut scope[at]) {
            self.explore_block(region, &dead);
        }

        if dialects::results(&scope[at]).is_empty() {
            // `for (int i = op->getNumOperands() - 1; i >= 0; i--)` (`:379-393`) — reverse, because
            // erasing renumbers what follows.
            if let Some(operands) = yield_operands_mut(&mut scope[at]) {
                for i in (0..operands.len()).rev() {
                    if parent_dead.get(i) == Some(&true) {
                        operands.remove(i);
                    }
                }
            }
            return;
        }

        if !may_have_side_effects(&scope[at]) {
            let results = dialects::results(&scope[at]);
            let may_be_deleted = results
                .iter()
                .all(|result| self.influence_type(*result) == Influence::None);
            let use_empty = results
                .iter()
                .all(|result| dialects::use_count(*result, scope) == 0);
            if use_empty && may_be_deleted {
                scope.remove(at);
            }
            return;
        }
        // The three `dyn_cast` arms (`:415-422`), asked before the block is borrowed to rewrite.
        match &scope[at] {
            Op::Sentient(sentient::Op::For { .. }) => {
                self.update_for_operation(scope, at, &BTreeSet::new());
            }
            Op::Sentient(sentient::Op::If { .. }) => self.update_if_operation(scope, at),
            Op::UniformRegions(UniformRegions::UniformizeRegions { .. }) => {
                self.update_uniformize_regions_operation(scope, at);
            }
            _ => {}
        }
    }

    /// Replaces: e558_exploreBlock
    ///
    /// Explores every op of the block, last to first.
    ///
    /// ⭐ THE REVERSE **INDEX** IS THE SEPARATE `SmallVector` (`:433-441`): the reference copies the
    /// ops out because deleting the one it is standing on invalidates an intrusive-list iterator, and
    /// [`Self::explore_operation`] only ever erases the op AT the index, leaving the ones below it.
    pub(crate) fn explore_block(&self, block: &mut Vec<Op>, parent_dead: &[bool]) {
        if block.is_empty() {
            return;
        }
        for at in (0..block.len()).rev() {
            self.explore_operation(block, at, parent_dead);
        }
    }
}

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
    use crate::arch::Elements;
    use crate::formats::Bits;
    use crate::islands::dataflow_ir::link::SendEnd;
    use crate::islands::dataflow_ir::ty::ScalarTy;
    use crate::islands::sentient::dialects::sentient::{
        Carried, CmpPredicate, Extent, Reg, RegType, ShuffleMode, Yielded,
    };

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
            bound_reg: None,
            carried: iter_args,
            dbg_name: None,
            body,
        })
    }

    /// `sentient.yield %a, %b`.
    fn yield_op(results: Vec<Val>) -> Op {
        Op::Sentient(sentient::Op::Yield { results })
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

    /// e300 — a loop result carries its influence to the loop's yield operand AND to the matching
    /// iter arg, and that arg then carries it on to the loop's init operand.
    #[test]
    fn the_work_list_walks_a_loop_result_back_to_its_init() {
        let scope = vec![
            constant(Val(10), 4),
            nop(),
            for_op(
                30,
                31,
                vec![carried(10, 11, 12)],
                vec![
                    Op::Sentient(sentient::Op::ScalarAdd {
                        lhs: Val(11),
                        rhs: Val(10),
                        result: Val(13),
                        reg: None,
                        element_size: None,
                        ty: ScalarTy::Index,
                    }),
                    yield_op(vec![Val(13)]),
                ],
            ),
        ];
        let regions: Vec<&[Op]> = vec![&scope];
        let defs = Definitions::from_innermost(&regions);
        let mut pass = EnhancedDeadVariableElimination::default();
        pass.add_to_work_list_and_update_assignment(
            Val(12),
            &Origin::Defined(scope[2].clone()),
            Influence::Memory,
            WorklistAdd::Push,
        );

        pass.process_work_list(defs);

        // %12 -> the yielded %13 and the iter arg %11; %13 -> its two operands %11 and %10; %11 -> the
        // init %10. Nothing else, and no disagreement anywhere.
        assert_eq!(
            pass.assignments.keys().copied().collect::<Vec<Val>>(),
            vec![Val(10), Val(11), Val(12), Val(13)]
        );
        assert_eq!(pass.influence_type(Val(10)), Influence::Memory);
        assert!(pass.conflicts().is_empty());
    }

    /// e301 — a `load_and_send`'s four operands seed MEMORY where the unit can hold one, and the same
    /// op in a store unit seeds nothing and records the misplacement instead.
    #[test]
    fn a_transfer_seeds_memory_only_in_a_unit_that_can_hold_it() {
        let send = Op::Sentient(sentient::Op::LoadAndSend {
            mutable_addr: Val(1),
            immutable_addr: Val(2),
            increment: Val(3),
            consumer: SendEnd::to_self(Val(4)),
            result: Val(5),
            extent: Extent::of(Elements(1), Bits(32)),
            interleaved_group: Elements(1),
            rotate_val: None,
            dir: None,
            shuffle_mode: ShuffleMode::NoShuffle,
            reg: Reg {
                locale: RegType::Unknown,
                index: None,
            },
            dbg_name: None,
        });
        // Every operand is defined by a `sentient.nop`, so each one has an origin to blame.
        let scope: Vec<Op> = vec![
            Op::Sentient(sentient::Op::ScalarCopy {
                input: Val(0),
                result: Val(1),
                reg: Reg {
                    locale: RegType::Unknown,
                    index: None,
                },
                element_size: None,
                program_header: false,
            }),
            constant(Val(2), 0),
            constant(Val(3), 1),
            constant(Val(4), 2),
            send.clone(),
        ];
        let regions: Vec<&[Op]> = vec![&scope];
        let defs = Definitions::from_innermost(&regions);

        let mut lxlu = EnhancedDeadVariableElimination::default();
        lxlu.initialize_assignment_for_an_operation(&send, GenericComp::Lxlu, defs);
        assert_eq!(
            lxlu.assignments.keys().copied().collect::<Vec<Val>>(),
            vec![Val(1), Val(2), Val(3), Val(4)]
        );
        assert_eq!(lxlu.influence_type(Val(4)), Influence::Memory);
        assert!(lxlu.misplaced().is_empty());

        let mut lxsu = EnhancedDeadVariableElimination::default();
        lxsu.initialize_assignment_for_an_operation(&send, GenericComp::Lxsu, defs);
        assert!(lxsu.assignments.is_empty());
        assert_eq!(
            lxsu.misplaced(),
            [MisplacedTransfer {
                op: Misplaced::LoadAndSend,
                unit: GenericComp::Lxsu,
            }]
        );
        assert_eq!(
            lxsu.misplaced()[0].op.message(),
            "Not expecting load_and_send in other units"
        );
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

    /// e437 — the pre-order walk seeds the loop's induction variable and its non-constant bound at
    /// the top level and then descends into its body, where the nested `set_mask`'s operand resolves
    /// against the enclosing block the region stack carries.
    #[test]
    fn initialize_work_list_walks_into_a_loop_body_with_the_enclosing_block_in_scope() {
        let unit_body = vec![
            Op::Sentient(sentient::Op::ScalarCopy {
                input: Val(0),
                result: Val(31),
                reg: Reg {
                    locale: RegType::Unknown,
                    index: None,
                },
                element_size: None,
                program_header: false,
            }),
            constant(Val(20), 7),
            for_op(
                30,
                31,
                Vec::new(),
                vec![
                    Op::Sentient(sentient::Op::SetMask {
                        mask_value: Val(20),
                        dbg_name: None,
                    }),
                    yield_op(Vec::new()),
                ],
            ),
        ];
        let mut pass = EnhancedDeadVariableElimination::default();

        pass.initialize_work_list(&unit_body, GenericComp::Lxlu, &[]);

        assert_eq!(
            pass.assignments.keys().copied().collect::<Vec<Val>>(),
            vec![Val(20), Val(30), Val(31)]
        );
        assert_eq!(pass.influence_type(Val(20)), Influence::Memory);
        assert_eq!(pass.influence_type(Val(30)), Influence::ControlFlow);
        assert_eq!(pass.influence_type(Val(31)), Influence::ControlFlow);
        assert!(pass.conflicts().is_empty());
    }

    /// e498 — the position whose result and body argument both lack influence leaves the loop, once
    /// because the influence test says so and once because a caller named it.
    #[test]
    fn updating_a_loop_drops_the_iter_arg_positions_with_no_influence() {
        let mut pass = EnhancedDeadVariableElimination::default();
        // ⭐ THE BODY ARGUMENT, NOT THE RESULT: either one carrying influence keeps the position.
        pass.add_to_work_list_and_update_assignment(
            Val(4),
            &Origin::RegionArg(nop()),
            Influence::Memory,
            WorklistAdd::Push,
        );
        let loop_op = |body| for_op(2, 0, vec![carried(0, 3, 5), carried(1, 4, 6)], body);
        // ⭐ THE YIELD ALREADY HANDS BACK ONE VALUE — e557 erased the dead operand before it reached
        // the parent.
        let mut scope = vec![
            constant(Val(0), 4),
            constant(Val(1), 7),
            loop_op(vec![yield_op(vec![Val(4)])]),
        ];

        pass.update_for_operation(&mut scope, 2, &BTreeSet::new());

        let Op::Sentient(sentient::Op::For { carried: left, .. }) = &scope[2] else {
            panic!("the loop survives");
        };
        assert_eq!(left, &vec![carried(1, 4, 6)]);

        // A provided set deletes exactly what it names, influence or not.
        let mut named = vec![
            constant(Val(0), 4),
            constant(Val(1), 7),
            loop_op(vec![yield_op(vec![Val(3)])]),
        ];

        pass.update_for_operation(&mut named, 2, &BTreeSet::from([1]));

        let Op::Sentient(sentient::Op::For { carried: left, .. }) = &named[2] else {
            panic!("the loop survives");
        };
        assert_eq!(left, &vec![carried(0, 3, 5)]);
    }

    /// e499 — the influenceless result leaves the `sentient.if` and takes its register slot with it.
    #[test]
    fn updating_a_conditional_drops_the_result_with_no_influence() {
        let mut pass = EnhancedDeadVariableElimination::default();
        pass.add_to_work_list_and_update_assignment(
            Val(6),
            &Origin::RegionArg(nop()),
            Influence::ControlFlow,
            WorklistAdd::Push,
        );
        let slot = |result: u32, locale| Yielded {
            result: Val(result),
            reg: Reg {
                locale,
                index: None,
            },
            element_size: None,
        };
        let mut scope = vec![Op::Sentient(sentient::Op::If {
            predicate: CmpPredicate::Eq,
            lhs: Val(0),
            rhs: Val(1),
            yielded: vec![slot(5, RegType::Jcr), slot(6, RegType::Lrf)],
            dbg_name: None,
            then_body: vec![yield_op(vec![Val(6)])],
            else_body: Vec::new(),
        })];

        pass.update_if_operation(&mut scope, 0);

        let Op::Sentient(sentient::Op::If { yielded, .. }) = &scope[0] else {
            panic!("the conditional survives");
        };
        assert_eq!(yielded, &vec![slot(6, RegType::Lrf)]);
    }

    /// e500 — the influenceless result leaves the `uniform.uniformize_regions` and both discardable
    /// arrays lose that slot with it.
    #[test]
    fn updating_a_uniformize_regions_op_drops_the_result_with_no_influence() {
        let mut pass = EnhancedDeadVariableElimination::default();
        pass.add_to_work_list_and_update_assignment(
            Val(8),
            &Origin::RegionArg(nop()),
            Influence::Memory,
            WorklistAdd::Push,
        );
        let slot = |locale| dialects::YieldedReg {
            element_size: Some(Bits(16)),
            locale,
        };
        let mut scope = vec![Op::UniformRegions(UniformRegions::UniformizeRegions {
            regions: vec![dialects::LocalRegion {
                arg: Val(1),
                units: vec![Val(0)],
                body: vec![Op::Uniform(uniform::Op::Yield {
                    operands: vec![Val(8)],
                })],
            }],
            results: vec![Val(7), Val(8)],
            yielded: vec![slot(RegType::Jcr), slot(RegType::Lrf)],
        })];

        pass.update_uniformize_regions_operation(&mut scope, 0);

        let Op::UniformRegions(UniformRegions::UniformizeRegions {
            results, yielded, ..
        }) = &scope[0]
        else {
            panic!("the uniformize_regions op survives");
        };
        assert_eq!(results, &vec![Val(8)]);
        assert_eq!(yielded, &vec![slot(RegType::Lrf)]);
    }

    /// `%out = sentient.scalar_add %lhs, %rhs` — effect-free, so a candidate for deletion.
    fn scalar_add(lhs: u32, rhs: u32, result: u32) -> Op {
        Op::Sentient(sentient::Op::ScalarAdd {
            lhs: Val(lhs),
            rhs: Val(rhs),
            result: Val(result),
            reg: None,
            element_size: None,
            ty: ScalarTy::Index,
        })
    }

    /// e557 — the loop's only carried position has no influence, so the body's `sentient.yield` loses
    /// its operand, the add that fed it becomes unread and effect-free and goes, and the loop itself
    /// loses the slot.
    #[test]
    fn exploring_a_loop_strips_the_dead_yield_operand_then_the_op_that_fed_it() {
        let pass = EnhancedDeadVariableElimination::default();
        let mut scope = vec![
            constant(Val(0), 4),
            for_op(
                2,
                0,
                vec![carried(0, 3, 5)],
                vec![scalar_add(3, 3, 4), yield_op(vec![Val(4)])],
            ),
        ];

        pass.explore_operation(&mut scope, 1, &[]);

        let Op::Sentient(sentient::Op::For { carried, body, .. }) = &scope[1] else {
            panic!("the loop survives");
        };
        assert!(carried.is_empty(), "the influenceless position leaves");
        assert_eq!(body, &vec![yield_op(Vec::new())], "the add went with it");
    }

    /// e558 — LAST TO FIRST IS THE PORT: the reader is erased before the op it read is asked whether
    /// anything reads it, so one pass clears the whole chain. A forward walk would keep the producer.
    #[test]
    fn exploring_a_block_backwards_clears_a_whole_unread_chain() {
        let pass = EnhancedDeadVariableElimination::default();
        let mut block = vec![constant(Val(0), 4), scalar_add(0, 0, 1), nop()];

        pass.explore_block(&mut block, &[]);

        assert_eq!(block, vec![nop()], "only the op with effects stands");
    }
}
