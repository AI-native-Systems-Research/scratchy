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

//! `RegisterTypeAssignment.cpp` — 17 of the campaign's 656 units (dependency level(s) [0, 1, 2, 3, 4, 5, 6]).
//!
//! | unit | entry | level | lines | authority path:line |
//! |---|---|---|---|---|
//! | `e137_RegisterTypeAssignmentPass` | 137 | 0 | 6 | `dcc/src/Transform/Sentient/RegisterTypeAssignment.cpp:77` |
//! | `e138_clear` | 138 | 0 | 6 | `dcc/src/Transform/Sentient/RegisterTypeAssignment.cpp:222` |
//! | `e139_getLocale` | 139 | 0 | 8 | `dcc/src/Transform/Sentient/RegisterTypeAssignment.cpp:231` |
//! | `e140_moveToCommonDominator` | 140 | 0 | 7 | `dcc/src/Transform/Sentient/RegisterTypeAssignment.cpp:246` |
//! | `e141_addToWorkList` | 141 | 0 | 3 | `dcc/src/Transform/Sentient/RegisterTypeAssignment.cpp:378` |
//! | `e142_isCopyNeeded` | 142 | 0 | 10 | `dcc/src/Transform/Sentient/RegisterTypeAssignment.cpp:382` |
//! | `e143_updateAssignment` | 143 | 0 | 3 | `dcc/src/Transform/Sentient/RegisterTypeAssignment.cpp:394` |
//! | `e144_printAssignments` | 144 | 0 | 14 | `dcc/src/Transform/Sentient/RegisterTypeAssignment.cpp:425` |
//! | `e350_createCopyOperationAndUpdateAssignment` | 350 | 1 | 120 | `dcc/src/Transform/Sentient/RegisterTypeAssignment.cpp:255` |
//! | `e351_addToWorkListAndUpdateAssignment` | 351 | 1 | 4 | `dcc/src/Transform/Sentient/RegisterTypeAssignment.cpp:399` |
//! | `e352_updateProgramUnit` | 352 | 1 | 45 | `dcc/src/Transform/Sentient/RegisterTypeAssignment.cpp:441` |
//! | `e463_addToWorkListCreateCopyAndUpdateAssignment` | 463 | 2 | 17 | `dcc/src/Transform/Sentient/RegisterTypeAssignment.cpp:405` |
//! | `e523_processWorkList` | 523 | 3 | 93 | `dcc/src/Transform/Sentient/RegisterTypeAssignment.cpp:492` |
//! | `e524_initializeAssignmentForAnOperation` | 524 | 3 | 383 | `dcc/src/Transform/Sentient/RegisterTypeAssignment.cpp:594` |
//! | `e574_initializeWorkList` | 574 | 4 | 8 | `dcc/src/Transform/Sentient/RegisterTypeAssignment.cpp:987` |
//! | `e609_runOn` | 609 | 5 | 55 | `dcc/src/Transform/Sentient/RegisterTypeAssignment.cpp:996` |
//! | `e633_runOnOperation` | 633 | 6 | 17 | `dcc/src/Transform/Sentient/RegisterTypeAssignment.cpp:113` |

// ⛔ THE PASS IS NOT WIRED INTO THE PIPELINE YET, so every item below is reachable only from this
// file's own tests until `e633_runOnOperation` (level 6) lands and something calls it. CI runs clippy
// with `-D warnings`, so without this the first ported leaf of a 17-unit module fails the gate.
// ⭐ REMOVE THIS WITH e633: at that point an unused item here is a real defect again.
#![allow(dead_code)]

use std::collections::{BTreeMap, VecDeque};
use std::fmt::Write as _;

use crate::arch::{Arch, IsaGen};
use crate::formats::Bits;
use crate::islands::dataflow_ir::Values;
use crate::islands::sentient::dialects::{
    self as dialects, Op, UniformRegions, Val, dataflow, sentient, symbol, uniform,
};
use crate::islands::sentient::{Program, print};
use crate::model::Model;
use crate::transform::sentient::utils::{self, Hoisted, InBlock, NewUse, OpAt};
use crate::units::{Core, DfirUnit};
use crate::workload::Workload;

/// `DisableThisPass`, `cl::init(false)` (`:52-56`).
const DISABLE_THIS_PASS: bool = false;

/// `RegisterLocales` (`:87-102`) IS THE ISLAND'S [`sentient::RegType`], not a second enum.
///
/// ⭐ THE FOURTEEN NAMES AND THEIR FOURTEEN SPELLINGS ARE THE SAME SET, and the pass itself says so:
/// every locale it records leaves through `getLocaleAsString` into `symbolizeSentientRegType(..)`
/// (`:446-472`), so a locale that did not spell a `SentientRegType` could not be written back.
/// ⛔ ONLY THE DECLARATION ORDER DIFFERS (the pass puts `LCCR` third, `SentientTypes.td:268-286` puts
/// `JCR` there) and nothing observes it: `enum_to_strings_reg_locales_` is read by `.at()` alone, and
/// the one iteration over an assignment map is over `assignments_`, not over the enum.
pub(crate) use sentient::RegType as RegisterLocale;

/// `std::pair<RegisterLocales, int> locale_elem_size` (`:299`) — what every alias cache is keyed by.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct AliasKey {
    /// The locale the user op demanded.
    pub(crate) locale: RegisterLocale,
    /// `element_size` — ⛔ `None` is the `-1` default of
    /// `addToWorkListCreateCopyAndUpdateAssignment` (`:208-211`), meaning the use states no width.
    pub(crate) element_size: Option<Bits>,
}

/// THE STATE ONE PROGRAM UNIT BUILDS — ⭐ EXACTLY WHAT `clear()` RESETS (`:222-227`), grouped so that
/// resetting the wrong half is not expressible. See [`PerModule`] for the other half.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct PerUnit {
    /// `assignments_` — ⭐ ORDERED, because e144 iterates it and `DenseMap` order is unspecified.
    pub(crate) assignments: BTreeMap<Val, RegisterLocale>,
    /// `cst_aliases_`, keyed by the constant's own `getValue()` (`:277`).
    pub(crate) cst_aliases: BTreeMap<i64, BTreeMap<AliasKey, Val>>,
    /// `query_map_aliases_`, keyed by the `uniform.query_map` result (`:280`).
    pub(crate) query_map_aliases: BTreeMap<Val, BTreeMap<AliasKey, Val>>,
    /// `unit_aliases_`, keyed by `dcc::getCoreId(get_unit_op)` then the unit's type (`:276`, `:317`).
    ///
    /// ⛔ `None` IS THE REFERENCE'S `-1`: a `dataflow.get_unit` the whole device shares carries no
    /// `core` attribute at all ([`crate::units::Residency::core`]).
    pub(crate) unit_aliases: BTreeMap<Option<Core>, BTreeMap<DfirUnit, Val>>,
}

/// THE STATE THAT OUTLIVES A UNIT — ⛔⛔ `clear()` TOUCHES NEITHER FIELD, AND THAT IS THE POINT.
///
/// `clear()` names four of the six maps the pass carries (`:222-227`); `sym_aliases_` and
/// `global_const_assignments_` are absent from it, and both are read across units — e524 writes
/// `global_const_assignments_[value] = IMM` for a module-level constant (`:607`) and e574 re-inserts
/// the lot into a fresh unit's `assignments_` (`:989`). A single flat struct with a hand-written
/// `clear()` makes clearing one of these a one-word edit; this split makes it a type error.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct PerModule {
    /// `global_const_assignments_` — the locales of constants declared outside any unit.
    pub(crate) global_const_assignments: BTreeMap<Val, RegisterLocale>,
    /// `sym_aliases_`, keyed by `symbol_op.getSymbolID()` (`:278`).
    pub(crate) sym_aliases: BTreeMap<i64, BTreeMap<AliasKey, Val>>,
}

/// WHAT `getLocale` FOUND — and whether it had to invent the answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Locale {
    /// `record->second`.
    Recorded(RegisterLocale),
    /// The `llvm::errs() << "Unable to find register locale information\n"` arm (`:234-237`), whose
    /// return value is `Unknown` — ⭐ A DIAGNOSTIC KEPT AS DATA rather than as a write to a stream,
    /// so `Unknown` recorded on purpose and `Unknown` because nothing was recorded stay apart.
    Unrecorded,
}

impl Locale {
    /// The `RegisterLocales` the reference returns either way.
    #[must_use]
    pub(crate) const fn reg_type(self) -> RegisterLocale {
        match self {
            Locale::Recorded(locale) => locale,
            Locale::Unrecorded => RegisterLocale::Unknown,
        }
    }
}

/// WHETHER A `sentient.scalar_copy` HAS TO BE CREATED FOR ONE USE — `isCopyNeeded`'s answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CopyNeeded {
    /// `return false`: copies are off, the demand is `unrelated`, the value is unrecorded, or the
    /// record already agrees with the demand.
    No,
    /// `return true`: the value is recorded as `imm` and this use demands a register.
    Yes,
    /// `DT_CHECK_MSG(record->second == IMM, "multiple assignments to the same variable")` (`:387-388`)
    /// — a value recorded as something OTHER than `imm` and now demanded as a third thing.
    ///
    /// ⭐ RETURNED AS DATA, NOT ASSERTED: this crate never refuses at run time, and the caller (e463)
    /// is the one that knows which operand asked.
    MultipleAssignments {
        /// What `assignments_` already held.
        recorded: RegisterLocale,
    },
}

/// A `sentient.scalar_copy` OP — ⛔ `DT_CHECK_MSG(isa<sentient::CopyOp>(op), "expected a copy op")`
/// (`:251`) AS A TYPE, and `DT_CHECK_MSG(op && new_use, "expected valid ops")` (`:250`) as a reference.
#[derive(Debug, Clone, Copy)]
pub(crate) struct CopyOp<'a>(&'a Op);

impl<'a> CopyOp<'a> {
    /// The witness, or `None` for any other op.
    #[must_use]
    pub(crate) fn of(op: &'a Op) -> Option<CopyOp<'a>> {
        matches!(op, Op::Sentient(sentient::Op::ScalarCopy { .. })).then_some(CopyOp(op))
    }

    /// The op itself.
    #[must_use]
    pub(crate) const fn op(self) -> &'a Op {
        self.0
    }
}

/// `RegisterTypeAssignmentPass`'s state (`:133-149`).
///
/// ⛔⛔ `addScalarCopies` IS A CONST GENERIC BECAUSE IT DECIDES WHICH OPS EXIST. The `.td` declares it
/// an `Option<"addScalarCopies", .., /*default=*/"true">` (`Passes.td:393-394`) and the pipeline
/// instantiates the pass BOTH WAYS in one build — see [`TypesOnly`] and [`TypesAndCopies`] — so it is
/// not a build-wide constant either. With it in the type, `RegisterTypeAssignment<false>` cannot
/// reach `createCopyOperationAndUpdateAssignment` at all: [`Self::is_copy_needed`] is statically
/// [`CopyNeeded::No`], which is the crate's rule that a flag must REMOVE ops rather than be consulted.
///
/// ⛔ `dcc_ext_ctx_` IS NOT A FIELD. `dccExtContext()` is read by e524 alone (`:692-954`): its
/// `getArch()` is the `A: Arch` parameter of
/// [`Self::initialize_assignment_for_an_operation`], its `getProgPatch()` is reached only under
/// [`DO_EAR_OPTIMIZATION`], and the rest goes to [`utils::memory_op_requires_immut_addr_scalar_copy`],
/// which takes `A` too. `opts_` is read by nothing in the file.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct RegisterTypeAssignment<const ADD_SCALAR_COPIES: bool> {
    /// The maps `clear()` resets at each unit.
    pub(crate) per_unit: PerUnit,
    /// The maps that survive it.
    pub(crate) per_module: PerModule,
    /// `worklist_` — ⛔ NOT IN [`PerUnit`]: `clear()` does not name it (`:222-227`), and e523 drains
    /// it to empty instead.
    pub(crate) worklist: VecDeque<Val>,
    /// Every `op->emitError(..); signalPassFailure();` e524 raised — see [`Refused`].
    pub(crate) failures: Vec<Refused>,
}

/// `op->emitError(..); signalPassFailure();` AS DATA — one transfer e524 found on a unit that cannot
/// run it (`:760-761`, `:855-856`).
///
/// ⭐ NOT A `Result` AND NOT A PANIC, for the reason
/// [`register_allocation::UnknownLocale`](super::register_allocation) gives: `signalPassFailure` is a
/// flag on the pass object, and the walk CONTINUES past the op it refused — so the assignments the
/// rest of the unit produced are still there to be read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Refused {
    /// The value the offending op binds — an identity, for the reason [`super::ForRef`] gives.
    pub(crate) at: Val,
    /// `current_unit_type_` — the unit it was found on.
    pub(crate) unit_type: DfirUnit,
    /// The message the reference emitted.
    pub(crate) message: &'static str,
}

/// `createRegisterTypeAssignmentPass(dcc_ext_ctx, common_opts, /*skip_copies=*/true)` — the early run
/// that only wants the register TYPES of the SSA variables (`dcc-standalone-main.cpp:318-319`).
pub(crate) type TypesOnly = RegisterTypeAssignment<false>;

/// `createRegisterTypeAssignmentPass(dcc_ext_ctx, common_opts)` — `skip_copies` defaults to `false`
/// (`Passes.h:157-160`), so this instantiation inserts the copies (`dcc-standalone-main.cpp:668-669`).
pub(crate) type TypesAndCopies = RegisterTypeAssignment<true>;

impl<const ADD_SCALAR_COPIES: bool> RegisterTypeAssignment<ADD_SCALAR_COPIES> {
    /// Replaces: e137_RegisterTypeAssignmentPass
    ///
    /// A pass with every map empty; `addScalarCopies = !skip_copies` is the parameter's own inversion,
    /// stated once by [`TypesOnly`] (`skip_copies = true`) and [`TypesAndCopies`] (`false`).
    #[must_use]
    pub(crate) fn new() -> RegisterTypeAssignment<ADD_SCALAR_COPIES> {
        RegisterTypeAssignment {
            per_unit: PerUnit::default(),
            per_module: PerModule::default(),
            worklist: VecDeque::new(),
            failures: Vec::new(),
        }
    }

    /// Replaces: e138_clear
    ///
    /// Drops one unit's assignments and its three alias caches.
    ///
    /// ⛔ TRAP: `sym_aliases_` AND `global_const_assignments_` SURVIVE — see [`PerModule`], which is
    /// why this is an assignment and not four `.clear()` calls.
    pub(crate) fn clear(&mut self) {
        self.per_unit = PerUnit::default();
    }

    /// Replaces: e139_getLocale
    ///
    /// The locale recorded for `val`, or [`Locale::Unrecorded`] — the reference's `errs()` line and its
    /// `Unknown` return.
    #[must_use]
    pub(crate) fn locale(&self, val: Val) -> Locale {
        match self.per_unit.assignments.get(&val) {
            Some(locale) => Locale::Recorded(*locale),
            None => Locale::Unrecorded,
        }
    }
}

/// WHERE A `sentient.scalar_copy` SITS IN ITS PROGRAM UNIT — [`CopyOp`]'s `isa` check carried on the
/// [`OpAt`] that [`utils::move_to_common_dominator`] moves by.
#[derive(Debug, Clone)]
pub(crate) struct CopyAt(OpAt);

impl CopyAt {
    /// The witness, or `None` when `at` names no op or names some other op.
    #[must_use]
    pub(crate) fn of(at: OpAt, unit_body: &[Op]) -> Option<CopyAt> {
        CopyOp::of(at.op(unit_body)?).map(|_| CopyAt(at))
    }
}

/// Replaces: e140_moveToCommonDominator
///
/// Hoists a `scalar_copy` until it dominates `new_use`, failing when the common dominator would leave
/// the program unit.
///
/// ⛔ BOTH `DT_CHECK_MSG`s ARE THE SIGNATURE: [`CopyAt`] is `isa<sentient::CopyOp>(op)` and two
/// references are `op && new_use`. What is left of the body is the delegation.
/// ⛔ TRAP: `dcc::utils::moveToCommonDominator` IS THIS CAMPAIGN'S e241
/// (`dcc/src/Transform/Sentient/Utils.cpp:84`, homed in `transform/sentient/utils`), and the scheduler
/// did not record the edge — the call is namespace-qualified rather than a member call. e241 also owns
/// the success/failure type.
pub(crate) fn move_to_common_dominator(
    unit_body: &mut Vec<Op>,
    copy: CopyAt,
    new_use: &NewUse,
) -> Hoisted {
    utils::move_to_common_dominator(unit_body, &copy.0, new_use)
}

impl<const ADD_SCALAR_COPIES: bool> RegisterTypeAssignment<ADD_SCALAR_COPIES> {
    /// Replaces: e141_addToWorkList
    ///
    /// Queues `val` unless it already has an assignment.
    pub(crate) fn add_to_work_list(&mut self, val: Val) {
        if !self.per_unit.assignments.contains_key(&val) {
            self.worklist.push_back(val);
        }
    }

    /// Replaces: e142_isCopyNeeded
    ///
    /// A copy is needed when `val` is already recorded as `imm` and this use demands something else.
    ///
    /// ⛔ TRAP: THE `unrelated` DEMAND IS ANSWERED BEFORE THE MAP IS CONSULTED, so a value recorded as
    /// `imm` and used by an op that wants no register at all is left alone.
    #[must_use]
    pub(crate) fn is_copy_needed(&self, val: Val, locale: RegisterLocale) -> CopyNeeded {
        if !ADD_SCALAR_COPIES || locale == RegisterLocale::Unrelated {
            return CopyNeeded::No;
        }
        match self.per_unit.assignments.get(&val) {
            Some(recorded) if *recorded != locale => {
                if *recorded == RegisterLocale::Imm {
                    CopyNeeded::Yes
                } else {
                    CopyNeeded::MultipleAssignments {
                        recorded: *recorded,
                    }
                }
            }
            Some(_) | None => CopyNeeded::No,
        }
    }

    /// Replaces: e143_updateAssignment
    ///
    /// Records `locale` for `val` — ⛔ THE FIRST WRITER WINS, which is what makes e142's `imm` record
    /// reachable at all.
    pub(crate) fn update_assignment(&mut self, val: Val, locale: RegisterLocale) {
        self.per_unit.assignments.entry(val).or_insert(locale);
    }

    /// Replaces: e144_printAssignments
    ///
    /// Per record, in key order: a blank line, the owning op as text, then `- locale: <spelling>`.
    ///
    /// ⛔ `llvm::dbgs()` IS A PARAMETER HERE, so a test can read what the pass wrote.
    /// ⛔ TRAP: THE OWNER IS THE DEFINING OP, OR A BLOCK ARGUMENT'S `getOwner()->getParentOp()`
    /// (`:428-434`). This island's ops carry no parent pointers, so `scope` supplies both lookups,
    /// innermost region first — the same mechanism [`dialects::Definitions`] is.
    pub(crate) fn print_assignments(&self, out: &mut String, scope: &[&[Op]]) {
        for (val, locale) in &self.per_unit.assignments {
            out.push('\n');
            if let Some(owner) = owner_of(*val, scope) {
                print::emit(out, owner, 0);
            }
            let _ = write!(out, "- locale: {}", locale.spelling());
        }
    }
}

/// `record->first.getDefiningOp()`, else `cast<BlockArgument>(record->first).getOwner()->getParentOp()`.
///
/// ⛔ `None` WHERE THE REFERENCE WOULD DEREFERENCE A NULL: a value bound by neither an op result nor a
/// `sentient.for` region argument has no owner to print ([`dialects::parent_for_arg`]).
fn owner_of<'a>(val: Val, scope: &[&'a [Op]]) -> Option<&'a Op> {
    scope
        .iter()
        .find_map(|region| dialects::defining_op(val, region))
        .or_else(|| {
            scope
                .iter()
                .find_map(|region| dialects::parent_for_arg(val, region).map(|(op, _)| op))
        })
}

/// `DoConstCommoning` — `cl::opt<bool>` `-dcc-register-type-assignment-const-commoning`, `cl::init(true)`
/// (`:57-60`). ⛔ NOT A PASS OPTION: nothing in `dcc/src` sets it, so `true` is what every build runs.
const DO_CONST_COMMONING: bool = true;

/// WHAT ONE USE ASKS A `sentient.scalar_copy` FOR — the four parameters of
/// `createCopyOperationAndUpdateAssignment` beside `this` (`:255-258`), grouped so that the pair every
/// alias cache is keyed by cannot be split.
#[derive(Debug, Clone)]
pub(crate) struct CopyDemand {
    /// `Value val` — the constant, symbol, multicast group, query map or unit handle to copy.
    pub(crate) val: Val,
    /// `locale` and `element_size` — `std::pair<RegisterLocales, int>` (`:299`).
    pub(crate) key: AliasKey,
    /// `Operation* user` — where the copy goes, and what a reused alias must come to dominate.
    pub(crate) user: OpAt,
    /// `is_mutable_addr_or_xrf_ptr` — ⛔ THE FLAG THAT TURNS COMMONING OFF for this use alone
    /// (`:294-297`): an updated register no longer holds the value the second use expected.
    pub(crate) is_mutable_addr_or_xrf_ptr: bool,
}

/// `DT_CHECK(op && isa<sentient::ConstantOp, dataflow::GetUnitOp, dataflow::CreateMulticastGroupOp,
/// uniform::QueryMapOp, symbol::CreateSymbolOp>(op))` (`:259-263`) AS A TYPE, each variant carrying
/// exactly the key its own `else if` arm looks a cache up by (`:301-354`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DefKind {
    /// `const_op.getValue()`.
    Constant(i64),
    /// `dcc::getCoreId(get_unit_op)` — `None` is its `-1` — and `dcc::getUnitType(get_unit_op)`.
    GetUnit {
        /// The core the unit belongs to.
        core: Option<Core>,
        /// Which unit.
        unit: DfirUnit,
    },
    /// ⭐ NO CACHE OF ITS OWN: the multicast arm decides only WHERE the new copy goes (`:359-360`).
    Multicast,
    /// `query_map_val = query_map_op.getResult()`.
    QueryMap {
        /// The value the query map binds.
        result: Val,
    },
    /// `symbol_op.getSymbolID()`.
    Symbol(i64),
}

impl DefKind {
    /// The witness, or `None` for the `DT_CHECK`'s other ops.
    fn of(op: &Op) -> Option<DefKind> {
        match op {
            Op::Sentient(sentient::Op::ScalarConstant { value, .. }) => {
                Some(DefKind::Constant(*value))
            }
            Op::Dataflow(dataflow::Op::GetUnit {
                residency, unit, ..
            }) => Some(DefKind::GetUnit {
                core: residency.core(),
                unit: *unit,
            }),
            Op::Dataflow(dataflow::Op::CreateMulticastGroup { .. }) => Some(DefKind::Multicast),
            Op::Uniform(uniform::Op::QueryMap { result, .. }) => {
                Some(DefKind::QueryMap { result: *result })
            }
            Op::Symbol(symbol::Op::CreateSymbol { symbol_id, .. }) => {
                Some(DefKind::Symbol(*symbol_id))
            }
            _ => None,
        }
    }
}

/// `dcc::utils::isSymbol` (`Analyses/Utils.cpp:141`) — a `symbol.create_symbol`, or a
/// `uniform.query_map` ANY of whose answerable values is one.
///
/// ⛔ TRAP: `isSymbol` ASKS FOR **ANY** AND ITS TWO SIBLINGS BELOW FOR **ALL** — the reference returns
/// on the first symbol found (`:152`) and on the first non-match (`:230`, `:249`).
/// ⛔ TRAP: THE KEY IS LOAD-BEARING, so [`dialects::uniform_mapping_values`] and not every pair of the
/// `uniform.def_immutable_mapping`.
pub(super) fn is_symbol(val: Val, defs: dialects::Definitions<'_>) -> bool {
    match defs.of(val) {
        Some(Op::Symbol(symbol::Op::CreateSymbol { .. })) => true,
        Some(Op::Uniform(uniform::Op::QueryMap { map, key, .. })) => {
            dialects::uniform_mapping_values(*map, *key, defs)
                .iter()
                .any(|value| {
                    matches!(
                        defs.of(*value),
                        Some(Op::Symbol(symbol::Op::CreateSymbol { .. }))
                    )
                })
        }
        Some(_) | None => false,
    }
}

/// `dcc::utils::isMulticast` (`Analyses/Utils.cpp:221`) — a `dataflow.create_multicast_group`, or a
/// query map ALL of whose answerable values are.
///
/// ⚠️ DIVERGENCE, AS IN [`utils::is_constant`]: AN EMPTY ANSWER LIST IS `true` HERE AND A
/// `DT_CHECK(values.size() > 0)` THERE (`:229`) — `.all()` over no values is vacuously true.
fn is_multicast(val: Val, defs: dialects::Definitions<'_>) -> bool {
    query_map_values_all(val, defs, |op| {
        matches!(op, Op::Dataflow(dataflow::Op::CreateMulticastGroup { .. }))
    })
}

/// `dcc::utils::isGetUnit` (`Analyses/Utils.cpp:240`) — a `dataflow.get_unit`, or a query map ALL of
/// whose answerable values are.
fn is_get_unit(val: Val, defs: dialects::Definitions<'_>) -> bool {
    query_map_values_all(val, defs, |op| {
        matches!(op, Op::Dataflow(dataflow::Op::GetUnit { .. }))
    })
}

/// The shape `isMulticast` and `isGetUnit` share: the op itself, else every value the query answers.
fn query_map_values_all(
    val: Val,
    defs: dialects::Definitions<'_>,
    is_wanted: impl Fn(&Op) -> bool + Copy,
) -> bool {
    match defs.of(val) {
        Some(Op::Uniform(uniform::Op::QueryMap { map, key, .. })) => {
            dialects::uniform_mapping_values(*map, *key, defs)
                .iter()
                .all(|value| defs.of(*value).is_some_and(is_wanted))
        }
        Some(op) => is_wanted(op),
        None => false,
    }
}

/// `dcc::uniform::utils::areConstantQueryMapsIdentical` (`Dialect/Uniform/Utils.cpp:234`) — one op is
/// identical to itself, and two others are when every unit A's key names answers with the same
/// `sentient.scalar_constant` value in both mappings.
///
/// ⛔ NOT AN ANCHORED UNIT: `getDeltasBetweenValuesOfUniformMappings` (`:204`) is inlined as "all
/// deltas zero", and B is looked up with **A's** key list there too (`:220`).
fn are_constant_query_maps_identical(a: Val, b: Val, defs: dialects::Definitions<'_>) -> bool {
    if a == b {
        return true;
    }
    let (
        Some(Op::Uniform(uniform::Op::QueryMap {
            map: map_a,
            key: key_a,
            ..
        })),
        Some(Op::Uniform(uniform::Op::QueryMap {
            map: map_b,
            key: key_b,
            ..
        })),
    ) = (defs.of(a), defs.of(b))
    else {
        return false;
    };
    let keys_a = dialects::uniform_mapping_keys(*key_a, defs);
    if keys_a.len() != dialects::uniform_mapping_keys(*key_b, defs).len() {
        return false;
    }
    let values_a = dialects::uniform_mapping_values(*map_a, *key_a, defs);
    let values_b = dialects::uniform_mapping_values(*map_b, *key_a, defs);
    if values_a.len() != keys_a.len() || values_b.len() != keys_a.len() {
        return false;
    }
    values_a
        .iter()
        .zip(&values_b)
        .all(|(a, b)| match (defs.of(*a), defs.of(*b)) {
            (
                Some(Op::Sentient(sentient::Op::ScalarConstant { value: x, .. })),
                Some(Op::Sentient(sentient::Op::ScalarConstant { value: y, .. })),
            ) => x == y,
            _ => false,
        })
}

impl<const ADD_SCALAR_COPIES: bool> RegisterTypeAssignment<ADD_SCALAR_COPIES> {
    /// Replaces: e350_createCopyOperationAndUpdateAssignment
    ///
    /// Reuses an already-created `sentient.scalar_copy` of the same constant, symbol, unit or query
    /// map when one can be hoisted to dominate this use, and otherwise builds a fresh one, caches it
    /// and records its locale.
    ///
    /// ⛔ `None` IS BOTH `DT_CHECK`s AS DATA: the definition is not one of the five ops (`:259-263`),
    /// or none of the four predicates holds (`:273`).
    /// ⛔ TRAP: A MULTICAST COPY GOES AFTER ITS DEFINITION AND EVERY OTHER ONE BEFORE THE USE — the
    /// `create_multicast_group` position is the PCFG translator's anchor point (`:358-360`).
    /// ⛔ TRAP: INSERTING BEFORE `user` SHIFTS `user` BY ONE, so e463 must re-derive its path; a
    /// FAILED hoist above mutates nothing (`utils::move_to_common_dominator` returns first), which is
    /// what makes trying several cached aliases safe.
    /// ⛔ THE NEW COPY CARRIES NO `element_size`: the demand's width keys the cache and is not written
    /// onto the op, exactly as `CopyOp::create(builder, loc, type, operand, regType)` (`:362-366`).
    pub(crate) fn create_copy_operation_and_update_assignment(
        &mut self,
        unit_body: &mut Vec<Op>,
        demand: &CopyDemand,
        values: &mut Values,
    ) -> Option<Val> {
        let (kind, is_multicast_val, aliases) = {
            let scope: [&[Op]; 1] = [unit_body.as_slice()];
            let defs = dialects::Definitions::from_innermost(&scope);
            let kind = DefKind::of(defs.of(demand.val)?)?;
            let is_constant =
                utils::is_constant(demand.val, utils::ConstKind::ScalarConstant, defs);
            let is_symbol_val = is_symbol(demand.val, defs);
            let is_multicast_val = is_multicast(demand.val, defs);
            let is_get_unit_val = is_get_unit(demand.val, defs);
            if !(is_constant || is_symbol_val || is_multicast_val || is_get_unit_val) {
                return None;
            }
            let do_const_commoning = (is_constant || is_symbol_val || is_get_unit_val)
                && !demand.is_mutable_addr_or_xrf_ptr
                && DO_CONST_COMMONING;
            let aliases = if do_const_commoning {
                self.cached_aliases(kind, demand.key, defs)
            } else {
                Vec::new()
            };
            (kind, is_multicast_val, aliases)
        };

        for alias in aliases {
            let Some(at) = utils::path_of(unit_body, alias) else {
                continue;
            };
            let Some(copy) = CopyAt::of(at, unit_body) else {
                continue;
            };
            if move_to_common_dominator(unit_body, copy, &NewUse::SameUnit(demand.user.clone()))
                == Hoisted::Done
            {
                return Some(alias);
            }
        }

        let at = if is_multicast_val {
            utils::path_of(unit_body, demand.val)?.next()
        } else {
            demand.user.clone()
        };
        let result = values.mint();
        utils::insert_at(
            unit_body,
            &at,
            Op::Sentient(sentient::Op::ScalarCopy {
                input: demand.val,
                result,
                reg: sentient::Reg {
                    locale: demand.key.locale,
                    index: None,
                },
                element_size: None,
                program_header: false,
            }),
        );
        match kind {
            DefKind::Constant(value) => {
                self.per_unit
                    .cst_aliases
                    .entry(value)
                    .or_default()
                    .insert(demand.key, result);
            }
            DefKind::Symbol(symbol_id) => {
                self.per_module
                    .sym_aliases
                    .entry(symbol_id)
                    .or_default()
                    .insert(demand.key, result);
            }
            DefKind::GetUnit { core, unit } => {
                self.per_unit
                    .unit_aliases
                    .entry(core)
                    .or_default()
                    .insert(unit, result);
            }
            DefKind::QueryMap { result: query_map } => {
                self.per_unit
                    .query_map_aliases
                    .entry(query_map)
                    .or_default()
                    .insert(demand.key, result);
            }
            DefKind::Multicast => {}
        }
        // ⛔ NOT `update_assignment`: `assignments_[new] = locale` OVERWRITES (`:374`), and the value
        // is fresh, so the first-writer-wins rule has nothing to protect here.
        self.per_unit.assignments.insert(result, demand.key.locale);
        Some(result)
    }

    /// THE ALIASES THIS DEMAND MAY REUSE, best first — the `else if` chain of `:301-354` as a lookup.
    ///
    /// ⭐ THE QUERY-MAP ARM YIELDS MORE THAN ONE, AND THAT IS THE REFERENCE'S SHAPE: its exact hit is
    /// tried, and a failed hoist falls through into the scan for a DIFFERENT `uniform.query_map` op
    /// holding the same constants (`:337-353`), which unrolling duplicates.
    fn cached_aliases(
        &self,
        kind: DefKind,
        key: AliasKey,
        defs: dialects::Definitions<'_>,
    ) -> Vec<Val> {
        match kind {
            DefKind::Constant(value) => self
                .per_unit
                .cst_aliases
                .get(&value)
                .and_then(|per_key| per_key.get(&key))
                .copied()
                .into_iter()
                .collect(),
            DefKind::Symbol(symbol_id) => self
                .per_module
                .sym_aliases
                .get(&symbol_id)
                .and_then(|per_key| per_key.get(&key))
                .copied()
                .into_iter()
                .collect(),
            DefKind::GetUnit { core, unit } => self
                .per_unit
                .unit_aliases
                .get(&core)
                .and_then(|per_unit| per_unit.get(&unit))
                .copied()
                .into_iter()
                .collect(),
            DefKind::QueryMap { result } => {
                self.per_unit
                    .query_map_aliases
                    .get(&result)
                    .and_then(|per_key| per_key.get(&key))
                    .copied()
                    .into_iter()
                    .chain(self.per_unit.query_map_aliases.iter().filter_map(
                        |(query_map, per_key)| {
                            let alias = *per_key.get(&key)?;
                            are_constant_query_maps_identical(*query_map, result, defs)
                                .then_some(alias)
                        },
                    ))
                    .collect()
            }
            DefKind::Multicast => Vec::new(),
        }
    }
}

/// `isa<..thirteen sentient ops.., dataflow::CreateMulticastGroupOp, uniform::UniformizeRegionsOp>(op)
/// || (isa<dataflow::GetUnitOp>(op) && is_any_of(current_unit_type_, L3LU, L3SU))` (`:457-469`) —
/// which ops record a locale per RESULT.
fn records_result_locales(op: &Op, unit_type: DfirUnit) -> bool {
    match op {
        Op::Sentient(
            sentient::Op::ScalarAdd { .. }
            | sentient::Op::ScalarSub { .. }
            | sentient::Op::ScalarCopy { .. }
            | sentient::Op::If { .. }
            | sentient::Op::For { .. }
            | sentient::Op::VectorMac { .. }
            | sentient::Op::ScalarConstant { .. }
            | sentient::Op::LoadAndSend { .. }
            | sentient::Op::ReceiveAndStore { .. }
            | sentient::Op::LoadAndStore { .. }
            | sentient::Op::ReceiveAndExtractScalar { .. }
            | sentient::Op::LoadAndExtractScalar { .. }
            | sentient::Op::LoadComputeAndSend { .. },
        )
        | Op::Dataflow(dataflow::Op::CreateMulticastGroup { .. })
        | Op::Uniform(uniform::Op::UniformizeRegions { .. }) => true,
        // ⭐ THE EAR OPTIMIZATION'S ONE CASE: a unit handle only lands in a register on an L3 half.
        Op::Dataflow(dataflow::Op::GetUnit { .. }) => {
            matches!(unit_type, DfirUnit::L3lu | DfirUnit::L3su)
        }
        _ => false,
    }
}

impl<const ADD_SCALAR_COPIES: bool> RegisterTypeAssignment<ADD_SCALAR_COPIES> {
    /// Replaces: e351_addToWorkListAndUpdateAssignment
    ///
    /// Queues `val`, then records `locale` for it.
    ///
    /// ⛔ THE ORDER IS THE POINT: [`Self::add_to_work_list`] skips an already-assigned value, so
    /// recording first would drop the enqueue.
    pub(crate) fn add_to_work_list_and_update_assignment(
        &mut self,
        val: Val,
        locale: RegisterLocale,
    ) {
        self.add_to_work_list(val);
        self.update_assignment(val, locale);
    }

    /// Replaces: e352_updateProgramUnit
    ///
    /// Writes every recorded locale back onto the op that binds the value, over the whole unit: a
    /// `sentient.for`'s induction variable and iterator arguments first, then the results of the ops
    /// that carry `regLocales` — `dataflow.get_unit` among them only on an L3 half.
    ///
    /// ⛔ `regLocales` VS `regLocale` IS NOT A DECISION HERE (`:475-482`): the reference picks the
    /// array or the singular attribute by count, and this island stores one [`sentient::Reg`] per
    /// position, so writing the position is both spellings at once.
    /// ⛔ TRAP: A CARRIED VALUE'S RESULT LOCALE WINS OVER ITS ARGUMENT'S — arguments are written
    /// first, and [`sentient::Carried`] collapses the reference's `1 + 2n` array to one entry each.
    /// ⭐ THE INDUCTION VARIABLE WRITES ENTRY 0, which is [`sentient::Op::For::bound_reg`] — the
    /// reference's `locales[argNumber]` with `argNumber == 0` and no special case.
    pub(crate) fn update_program_unit(&self, unit_body: &mut [Op], unit_type: DfirUnit) {
        for op in unit_body.iter_mut() {
            let mut positions: Vec<Val> = Vec::new();
            if let Op::Sentient(sentient::Op::For { iv, carried, .. }) = op {
                positions.push(*iv);
                positions.extend(carried.iter().map(|value| value.arg));
            }
            if records_result_locales(op, unit_type) {
                positions.extend(dialects::results(op));
            }
            for val in positions {
                dialects::set_reg_locale_on(op, val, self.locale(val).reg_type());
            }
            for region in dialects::regions_mut(op) {
                self.update_program_unit(region, unit_type);
            }
        }
    }
}

/// THE ONE OPERAND OF ONE OP THAT ASKED FOR A LOCALE — `MutableOperandRange operand` with
/// `DT_CHECK_MSG(operand.size() == 1, "expected a single operand")` (`:409`) discharged by the type.
///
/// ⭐ THE RANGE'S OWNER IS THE `user` EVERY CALL SITE PASSES BESIDE IT, so one path names both — the
/// op the copy is positioned against and the op whose operand is rewritten.
#[derive(Debug, Clone)]
pub(crate) struct OperandSlot {
    /// `user` — where that op sits in the unit body.
    user: OpAt,
    /// Which slot of it.
    at: Slot,
    /// `operand[0].get()` — read once, at construction.
    val: Val,
}

/// WHICH SLOT OF THE USER ONE `MutableOperandRange` NAMES.
///
/// ⛔ A WIRE END IS NO [`dialects::operands`] POSITION AND IS STILL A `MutableOperandRange`: e524
/// hands e463 `load_and_send.getConsumerMutable()` (`:906-908`), which this island models as a
/// [`sentient::SendEnd`] rather than a [`Val`] — see [`dialects::operands`] and
/// [`sentient::wire_end`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Slot {
    /// `op->getOpOperand(at)`, in [`dialects::operands`] order.
    Operand(usize),
    /// `getConsumerMutable()`, `getProducerMutable()` or the receive's `$unit`.
    WireEnd,
}

impl OperandSlot {
    /// The slot, or `None` when `user` names no op or that op has no operand `at`.
    #[must_use]
    pub(crate) fn of(user: OpAt, at: usize, unit_body: &[Op]) -> Option<OperandSlot> {
        let val = *dialects::operands(user.op(unit_body)?).get(at)?;
        Some(OperandSlot {
            user,
            at: Slot::Operand(at),
            val,
        })
    }

    /// The wire-end slot, or `None` when `user` names no op or an op with no wire end.
    #[must_use]
    pub(crate) fn wire_end_of(user: OpAt, unit_body: &[Op]) -> Option<OperandSlot> {
        let Op::Sentient(inner) = user.op(unit_body)? else {
            return None;
        };
        let val = sentient::wire_end(inner)?;
        Some(OperandSlot {
            user,
            at: Slot::WireEnd,
            val,
        })
    }

    /// The value in the slot.
    #[must_use]
    pub(crate) const fn val(&self) -> Val {
        self.val
    }

    /// `operand.assign(val)` — ⭐ AN OUT-OF-RANGE OR ABSENT SLOT WRITES NOTHING, as
    /// [`dialects::set_operand`] does.
    fn assign(&self, op: &mut Op, val: Val) {
        match self.at {
            Slot::Operand(at) => dialects::set_operand(op, at, val),
            Slot::WireEnd => {
                if let Op::Sentient(inner) = op
                    && let Some(end) = sentient::wire_end_mut(inner)
                {
                    *end = val;
                }
            }
        }
    }
}

/// WHAT e463 DID TO THE SLOT — the reference writes its effect into the IR and returns `void`, and this
/// names the outcomes so a caller (and a test) can tell them apart.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CopyInstalled {
    /// The `else`: no copy was needed, and the demanded locale was recorded for the value itself.
    Recorded,
    /// `operand.assign(new_copy)` — the new copy went into THIS slot alone.
    Assigned(Val),
    /// `val.replaceAllUsesExcept(new_copy, new_copy.getDefiningOp())` — the `gtr` path, where every
    /// other reader of the value was redirected instead.
    AllUsesRedirected(Val),
    /// e350 answered `None` (its own two `DT_CHECK`s as data), so no copy exists and no slot changed.
    NoCopyProduced,
    /// [`CopyNeeded::MultipleAssignments`] — the reference's
    /// `DT_CHECK_MSG(record->second == IMM, "multiple assignments to the same variable")`, carried out
    /// to the caller that knows which operand asked.
    MultipleAssignments {
        /// What `assignments_` already held for the value in the slot.
        recorded: RegisterLocale,
    },
}

/// WHERE THE USER SITS AFTER e350 HAS RUN — its own path, or ONE LATER when the op there is no longer
/// it.
///
/// ⛔ THE SHIFT IS AT MOST ONE, WHICH IS WHY TWO CANDIDATES ARE ENOUGH: e350 either inserts a single
/// `sentient.scalar_copy` or hoists ONE op out of a strictly deeper block
/// ([`utils::move_to_common_dominator`]), so nothing is ever removed from ahead of the user in its own
/// block. A `scalar_copy` binding a freshly minted result equals no existing op, so the comparison
/// cannot pick the newcomer.
fn user_after_copy(unit_body: &[Op], user: &OpAt, snapshot: &Op) -> OpAt {
    let shifted = user.next();
    if user.op(unit_body) != Some(snapshot) && shifted.op(unit_body) == Some(snapshot) {
        return shifted;
    }
    user.clone()
}

impl<const ADD_SCALAR_COPIES: bool> RegisterTypeAssignment<ADD_SCALAR_COPIES> {
    /// Replaces: e463_addToWorkListCreateCopyAndUpdateAssignment
    ///
    /// Queues the value in `slot`, and either records the demanded locale on it or copies it into a
    /// register of that locale and rewires the reader — every other use for `gtr`, this operand for
    /// everything else.
    ///
    /// ⛔ TRAP: `gtr` IS PARTLY HARDWARE MANAGED, so exactly ONE copy may hang off a
    /// `create_multicast_group`: every OTHER use moves to the copy and the copy keeps reading the
    /// original, which is what `replaceAllUsesExcept(.., new_copy.getDefiningOp())` says on a
    /// one-operand op.
    /// ⛔ TRAP: e350 CAN SHIFT THE READER BY ONE (it inserts in front of it), so the operand is written
    /// through a re-derived path — see [`user_after_copy`].
    pub(crate) fn add_to_work_list_create_copy_and_update_assignment(
        &mut self,
        unit_body: &mut Vec<Op>,
        slot: &OperandSlot,
        key: AliasKey,
        is_mutable_addr_or_xrf_ptr: bool,
        values: &mut Values,
    ) -> CopyInstalled {
        let val = slot.val;
        self.add_to_work_list(val);
        match self.is_copy_needed(val, key.locale) {
            CopyNeeded::No => {
                self.update_assignment(val, key.locale);
                CopyInstalled::Recorded
            }
            CopyNeeded::MultipleAssignments { recorded } => {
                CopyInstalled::MultipleAssignments { recorded }
            }
            CopyNeeded::Yes => {
                let snapshot = match slot.user.op(unit_body) {
                    Some(op) => op.clone(),
                    None => return CopyInstalled::NoCopyProduced,
                };
                let demand = CopyDemand {
                    val,
                    key,
                    user: slot.user.clone(),
                    is_mutable_addr_or_xrf_ptr,
                };
                let Some(new_copy) =
                    self.create_copy_operation_and_update_assignment(unit_body, &demand, values)
                else {
                    return CopyInstalled::NoCopyProduced;
                };
                if key.locale == RegisterLocale::Gtr {
                    dialects::replace_all_uses_with(unit_body, val, new_copy);
                    // The copy itself is the one reader that keeps the original — its only operand.
                    if let Some(at) = utils::path_of(unit_body, new_copy) {
                        if let Some(op) = at.op_mut(unit_body) {
                            dialects::set_operand(op, 0, val);
                        }
                    }
                    CopyInstalled::AllUsesRedirected(new_copy)
                } else {
                    let now = user_after_copy(unit_body, &slot.user, &snapshot);
                    if let Some(op) = now.op_mut(unit_body) {
                        slot.assign(op, new_copy);
                    }
                    CopyInstalled::Assigned(new_copy)
                }
            }
        }
    }
}

/// `-dcc-sentient-create-copy-ops-for-iter-args`, `cl::init(true)` (`:68-72`) — a `dcc-opt`
/// command-line flag, not a program property, and this crate has no flags.
const CREATE_COPY_OPS_FOR_ITER_ARGS: bool = true;

/// `dcc::utils::isInductionVariable` (`Analyses/Utils.cpp:128-139`) — region argument 0 of a
/// `sentient.for`, and `false` for an op result or any other op's region argument.
fn is_induction_variable(val: Val, defs: dialects::Definitions<'_>) -> bool {
    matches!(defs.for_arg_of(val), Some((_, 0)))
}

/// `region.front().getTerminator()->getOperand(index)`, at either dialect's spelling of a yield.
///
/// ⭐ `None` FOR A REGION WITH NO TERMINATOR — a `sentient.if`'s `else` region is routinely empty,
/// where `front().getTerminator()` hands the reference a null `Operation *`.
fn terminator_operand(region: &[Op], index: usize) -> Option<Val> {
    let operands = region.iter().rev().find_map(|op| match op {
        Op::Sentient(sentient::Op::Yield { results }) => Some(results.as_slice()),
        Op::Uniform(uniform::Op::Yield { operands }) => Some(operands.as_slice()),
        _ => None,
    })?;
    operands.get(index).copied()
}

/// `op->getParentOfType<sentient::ForOp>()` — whether any op enclosing `at` is a `sentient.for`.
fn inside_a_for(unit_body: &[Op], at: &OpAt) -> bool {
    let mut cursor = at.parent();
    while let Some(enclosing) = cursor {
        if matches!(
            enclosing.op(unit_body),
            Some(Op::Sentient(sentient::Op::For { .. }))
        ) {
            return true;
        }
        cursor = enclosing.parent();
    }
    false
}

/// WHAT ONE DRAINED VALUE'S DEFINITION ASKS FOR — decided while the unit body is only borrowed, acted
/// on once it is not.
enum Propagate {
    /// The add/sub/copy arms: these operand positions of the definition need a register of the
    /// demanded locale.
    CopyOperands(Vec<usize>),
    /// The `for`/`if`/`uniformize_regions` arm: what every region hands back at the result's own
    /// position, and — for a loop — the matching iterator argument.
    RegionInterface {
        /// One per region that has a terminator.
        yields: Vec<Val>,
        /// `for_op.getRegionIterArgs()[val_index]`.
        iter_arg: Option<Val>,
    },
}

impl<const ADD_SCALAR_COPIES: bool> RegisterTypeAssignment<ADD_SCALAR_COPIES> {
    /// Replaces: e523_processWorkList
    ///
    /// Drains the worklist, pushing each value's locale back onto what produced it — a loop's
    /// initialiser, result and yield; an add/sub/copy's non-constant inputs; the yields and iterator
    /// argument behind a region-carrying op's result — copying an input into that locale where due.
    ///
    /// ⛔ TRAP: `getIndexOfLoopRegionIterArgs` ANSWERS `-1` FOR THE INDUCTION VARIABLE TOO, so
    /// position 0 of [`dialects::Definitions::for_arg_of`] takes the `else` branch, where its null
    /// `getDefiningOp()` makes every `isa<>` false and nothing happens.
    /// ⭐ AND THAT NULL `def` IS UNREACHABLE FOR ANYTHING ELSE — e141 queues only a value with NO
    /// assignment, and e524 assigns `lccr` to every induction variable before the drain begins, which
    /// is what keeps `isa<AddOp, SubOp>(nullptr)` from asserting.
    pub(crate) fn process_work_list(&mut self, unit_body: &mut Vec<Op>, values: &mut Values) {
        while let Some(val) = self.worklist.pop_front() {
            let (key, position) = {
                let scope: [&[Op]; 1] = [unit_body.as_slice()];
                let defs = dialects::Definitions::from_innermost(&scope);
                let key = AliasKey {
                    locale: self.locale(val).reg_type(),
                    element_size: dialects::element_size(val, defs),
                };
                (key, defs.for_arg_of(val).map(|(_, position)| position))
            };
            match position {
                Some(position) if position >= 1 => {
                    self.process_iter_arg(unit_body, val, position - 1, key, values);
                }
                _ => self.process_definition(unit_body, val, key, values),
            }
        }
    }

    /// e523's `index != -1` half — `val` is a `sentient.for`'s iterator argument `index`.
    ///
    /// ⛔ TRAP: THE COPY IS ONLY WORTH IT FOR A NESTED LOOP'S QUERY-MAP INITIALISER — an outermost
    /// loop's iter arg gets promoted to the header anyway, and a copy there ADDS `REGCOPY`s
    /// (`:506-513`).
    fn process_iter_arg(
        &mut self,
        unit_body: &mut Vec<Op>,
        val: Val,
        index: usize,
        key: AliasKey,
        values: &mut Values,
    ) {
        let (for_at, init, result, yielded, copy_the_init) = {
            let scope: [&[Op]; 1] = [unit_body.as_slice()];
            let defs = dialects::Definitions::from_innermost(&scope);
            let Some((Op::Sentient(sentient::Op::For { carried, body, .. }), _)) =
                defs.for_arg_of(val)
            else {
                return;
            };
            let Some(entry) = carried.get(index) else {
                return;
            };
            let Some(for_at) = utils::path_of(unit_body, entry.result) else {
                return;
            };
            let copy_the_init = CREATE_COPY_OPS_FOR_ITER_ARGS
                && matches!(
                    defs.of(entry.init),
                    Some(Op::Uniform(uniform::Op::QueryMap { .. }))
                )
                && inside_a_for(unit_body, &for_at);
            (
                for_at,
                entry.init,
                entry.result,
                terminator_operand(body, index),
                copy_the_init,
            )
        };
        // Operand 0 of a `sentient.for` is `$bound`, so initialiser `index` is operand `index + 1`.
        match copy_the_init
            .then(|| OperandSlot::of(for_at, index + 1, unit_body))
            .flatten()
        {
            Some(slot) => {
                self.add_to_work_list_create_copy_and_update_assignment(
                    unit_body, &slot, key, false, values,
                );
            }
            None => self.add_to_work_list_and_update_assignment(init, key.locale),
        }
        self.update_assignment(result, key.locale);
        if let Some(yielded) = yielded {
            self.add_to_work_list_and_update_assignment(yielded, key.locale);
        }
    }

    /// e523's `else` half — `val` is an op result.
    fn process_definition(
        &mut self,
        unit_body: &mut Vec<Op>,
        val: Val,
        key: AliasKey,
        values: &mut Values,
    ) {
        let plan = {
            let scope: [&[Op]; 1] = [unit_body.as_slice()];
            let defs = dialects::Definitions::from_innermost(&scope);
            let Some(def) = defs.of(val) else {
                return;
            };
            match def {
                Op::Sentient(
                    sentient::Op::ScalarAdd { lhs, rhs, .. }
                    | sentient::Op::ScalarSub { lhs, rhs, .. },
                ) => {
                    // A constant, a symbol and an induction variable each already own their register.
                    let owns_its_register = |operand: Val| {
                        utils::is_constant(operand, utils::ConstKind::ScalarConstant, defs)
                            || is_symbol(operand, defs)
                            || is_induction_variable(operand, defs)
                    };
                    Propagate::CopyOperands(
                        [*lhs, *rhs]
                            .into_iter()
                            .enumerate()
                            .filter(|(_, operand)| !owns_its_register(*operand))
                            .map(|(at, _)| at)
                            .collect(),
                    )
                }
                Op::Sentient(sentient::Op::ScalarCopy { input, .. }) => Propagate::CopyOperands(
                    if utils::is_constant(*input, utils::ConstKind::ScalarConstant, defs) {
                        Vec::new()
                    } else {
                        vec![0]
                    },
                ),
                Op::Sentient(sentient::Op::For { .. } | sentient::Op::If { .. })
                | Op::Uniform(uniform::Op::UniformizeRegions { .. })
                | Op::UniformRegions(UniformRegions::UniformizeRegions { .. }) => {
                    let Some(val_index) = dialects::results(def).iter().position(|r| *r == val)
                    else {
                        panic!("DT_CHECK(val_index != -1) (`:567`)");
                    };
                    Propagate::RegionInterface {
                        yields: dialects::regions_ref(def)
                            .into_iter()
                            .filter_map(|region| terminator_operand(region, val_index))
                            .collect(),
                        iter_arg: match def {
                            Op::Sentient(sentient::Op::For { carried, .. }) => {
                                carried.get(val_index).map(|entry| entry.arg)
                            }
                            _ => None,
                        },
                    }
                }
                _ => return,
            }
        };
        match plan {
            // ⛔ THE PATH IS RE-DERIVED PER OPERAND: e463 can insert a copy in front of the user, and
            // the reference holds a stable `Operation *` where this holds a position.
            Propagate::CopyOperands(slots) => {
                for at in slots {
                    let Some(slot) = utils::path_of(unit_body, val)
                        .and_then(|def_at| OperandSlot::of(def_at, at, unit_body))
                    else {
                        continue;
                    };
                    self.add_to_work_list_create_copy_and_update_assignment(
                        unit_body, &slot, key, false, values,
                    );
                }
            }
            Propagate::RegionInterface { yields, iter_arg } => {
                for yielded in yields {
                    self.add_to_work_list_and_update_assignment(yielded, key.locale);
                }
                if let Some(arg) = iter_arg {
                    self.add_to_work_list_and_update_assignment(arg, key.locale);
                }
            }
        }
    }
}

/// `-dcc-sentient-do-ear-optimization`, `cl::init(false)` (`:62-66`) — a `dcc-opt` command-line flag.
///
/// ⛔ AND SO ITS TWO ARMS ARE NOT BELOW (`:748-758`, `:844-854`). Both read `DoEAROptimization &&
/// dccExtContext().getProgPatch() && isa<dataflow::GetUnitOp>(..)`, and this crate's rule is that a
/// flag REMOVES the ops it guards rather than being consulted at run time — which is also why the
/// pass carries no `getProgPatch()`.
const DO_EAR_OPTIMIZATION: bool = false;

/// `dcc::utils::isUpdateMode` (`Analyses/Utils.cpp:65-125`) at the three arms e524 reaches — the
/// increment is a non-zero constant, or a query map with a non-zero value among its constants.
///
/// ⛔ NOT AN ANCHORED UNIT, and taking the increment VALUE rather than the op removes BOTH routes to
/// `llvm_unreachable("unhandled operation")` (`:125`) and the out-of-scope `L3GatherScatterChecker`
/// its `load_and_store` arm consults (`:86`) — the same reading
/// [`scalar_op_hoisting`](super::scalar_op_merging_and_hoisting) makes of it, kept local because that
/// one is keyed by its own descriptor.
fn is_update_mode(increment: Val, defs: dialects::Definitions<'_>) -> bool {
    match defs.of(increment) {
        Some(Op::Sentient(sentient::Op::ScalarConstant { value, .. })) => *value != 0,
        Some(Op::Uniform(uniform::Op::QueryMap { map, .. })) => has_non_zero_constants(*map, defs),
        _ => false,
    }
}

/// `dcc::utils::hasNonZeroConstants` (`Analyses/Utils.cpp:50-63`) — a query map whose values are ALL
/// constants (its own first gate, `:51`) and at least one of which is not zero.
fn has_non_zero_constants(map: Val, defs: dialects::Definitions<'_>) -> bool {
    let Some(Op::Uniform(uniform::Op::DefImmutableMapping { pairs, .. })) = defs.of(map) else {
        return false;
    };
    let mut any_non_zero = false;
    for (_, value) in pairs {
        match defs.of(*value) {
            Some(Op::Sentient(sentient::Op::ScalarConstant { value, .. })) => {
                any_non_zero |= *value != 0;
            }
            _ => return false,
        }
    }
    any_non_zero
}

/// `dcc::getUnitType(val.getDefiningOp())` (`Utils/DccExtContext.cpp:126-130`, `:191-206`) at the two
/// ends of a transfer — the `dataflow.get_unit`'s own type, or the one type every value a query map
/// answers with shares.
///
/// ⚠️ DIVERGENCE ON A BLOCK ARGUMENT AND ON AN EMPTY MAPPING: `None` where the reference's
/// `DT_CHECK_MSG(op, "expected valid op")` aborts — the reading
/// [`address_pinning_and_toggle`](super::address_pinning_and_toggle) already makes of this function.
fn unit_type_of(val: Val, defs: dialects::Definitions<'_>) -> Option<DfirUnit> {
    let unit_of = |value: Val| match defs.of(value) {
        Some(Op::Dataflow(dataflow::Op::GetUnit { unit, .. })) => Some(*unit),
        _ => None,
    };
    match defs.of(val)? {
        Op::Dataflow(dataflow::Op::GetUnit { unit, .. }) => Some(*unit),
        Op::Uniform(uniform::Op::QueryMap { map, key, .. }) => {
            let values = dialects::uniform_mapping_values(*map, *key, defs);
            let mut units = values.iter().map(|value| unit_of(*value));
            let first = units.next()??;
            units.all(|unit| unit == Some(first)).then_some(first)
        }
        _ => None,
    }
}

/// e247's `unit_type` parameter — [`utils::MemoryUnit`] is exactly the six units the memory arms'
/// `DT_CHECK`s admit (`:677`, `:780`), so `None` is a unit no arm that calls e247 can be on.
fn memory_unit(unit_type: DfirUnit) -> Option<utils::MemoryUnit> {
    match unit_type {
        DfirUnit::L0lu => Some(utils::MemoryUnit::L0lu),
        DfirUnit::L0su => Some(utils::MemoryUnit::L0su),
        DfirUnit::Lxlu => Some(utils::MemoryUnit::Lxlu),
        DfirUnit::Lxsu => Some(utils::MemoryUnit::Lxsu),
        DfirUnit::L3lu => Some(utils::MemoryUnit::L3lu),
        DfirUnit::L3su => Some(utils::MemoryUnit::L3su),
        _ => None,
    }
}

/// `is_any_of(current_unit_type_, L0LU, L0SU, LXLU, LXSU, L3LU, L3SU)` — the `DT_CHECK` the send and
/// the store share (`:677`, `:780`), stated as [`memory_unit`] so the two cannot drift apart.
fn is_a_memory_unit(unit_type: DfirUnit) -> bool {
    memory_unit(unit_type).is_some()
}

/// `(dccExtContext().getArch() >= IsaCoreGen::SEN1P5_ISA) ? Unrelated : LBR` (`:718-720`, `:821-823`,
/// `:899-900`) — where an L3 transfer's immutable address lives.
fn immutable_locale<A: Arch>() -> RegisterLocale {
    if A::GEN >= IsaGen::Sen1p5 {
        RegisterLocale::Unrelated
    } else {
        RegisterLocale::Lbr
    }
}

/// `isa<dataflow::CreateMulticastGroupOp>(def)` — [`find_yields_resolving_to`]'s first target.
fn is_a_multicast_group(op: &Op) -> bool {
    matches!(op, Op::Dataflow(dataflow::Op::CreateMulticastGroup { .. }))
}

/// `isa<dataflow::GetUnitOp>(def)` — its second.
fn is_a_get_unit(op: &Op) -> bool {
    matches!(op, Op::Dataflow(dataflow::Op::GetUnit { .. }))
}

/// `dcc::utils::findYieldsResolvingTo<TargetTy, sentient::IfOp>` (`Utils/Utils.cpp:182-208`) — whether
/// ANY region of `if_op` hands back, at `result_index`, a value the target op defines, a query map all
/// of whose answers it defines, or a nested `sentient.if` that does either.
///
/// ⛔ NOT AN ANCHORED UNIT, and its query-map arm IS [`query_map_values_all`] — the reference's
/// `isa<TargetTy>(def)` and its `all_of` over the mapping are that function's two arms.
fn find_yields_resolving_to(
    if_op: &Op,
    result_index: usize,
    defs: dialects::Definitions<'_>,
    is_target: impl Fn(&Op) -> bool + Copy,
) -> bool {
    dialects::regions_ref(if_op).into_iter().any(|region| {
        // `None` is the empty `else` region, whose `front().getTerminator()` is null.
        let Some(yielded) = terminator_operand(region, result_index) else {
            return false;
        };
        // `if (isa<BlockArgument>(yield_operand)) continue;`
        if defs.for_arg_of(yielded).is_some() {
            return false;
        }
        if query_map_values_all(yielded, defs, is_target) {
            return true;
        }
        match defs.of(yielded) {
            Some(nested @ Op::Sentient(sentient::Op::If { .. })) => dialects::results(nested)
                .iter()
                .position(|result| *result == yielded)
                .is_some_and(|index| find_yields_resolving_to(nested, index, defs, is_target)),
            _ => false,
        }
    })
}

/// `dcc::utils::L3GatherScatterChecker` (`Analyses/Utils.cpp:608-633`) — WHICH OF FOUR SHAPES one
/// `sentient.load_and_store` has.
///
/// ⛔ ONE ANSWER AND NOT FOUR FLAGS: the constructor's own `DT_CHECK`s say the states exclude each
/// other (`:619-620`, `:626-627`, `:630-632`), and e524 asks the four `is*()` questions of them.
/// ⛔ NOT AN OUT-OF-SCOPE ANALYSIS: three unit-type comparisons and one attribute, ported inline as
/// [`is_symbol`] and its siblings are — the ANALYSIS CLASSES are the out-of-scope thing.
/// ⚠️ e524's FIRST `DT_CHECK` OVER IT IS UNPORTED: `(src == LX && dst == QGI)` (`:868-871`) names a
/// unit [`DfirUnit`] does not model, so the check without that disjunct would panic on a valid
/// `LX -> QGI` transfer. The second group (`:872-880`) is expressible, and IS ported.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum L3GatherScatter {
    /// `is_gather_` — `L3IBR -> LX`, the read half of a gather.
    Gather,
    /// `is_scatter_` — `LX -> L3IBR` without the `is_ibr_write` attribute.
    Scatter,
    /// `is_ibr_write_` — the attribute, or `HBM -> L3IBR`, which is also
    /// `is_ibr_write_for_gather_`.
    IbrWrite {
        /// `isIBRWriteForGather()`.
        for_gather: bool,
    },
    /// `!isValid()` — an ordinary transfer between the LX and the HBM.
    Plain,
}

impl L3GatherScatter {
    /// The constructor, whose last `DT_CHECK` is the only one a state machine cannot discharge: the
    /// attribute on a `L3IBR -> LX` transfer claims both at once.
    fn of(
        src: Option<DfirUnit>,
        dst: Option<DfirUnit>,
        has_ibr_write_attr: bool,
    ) -> L3GatherScatter {
        let gather = src == Some(DfirUnit::L3Ibr) && dst == Some(DfirUnit::Lx);
        let scatter =
            src == Some(DfirUnit::Lx) && dst == Some(DfirUnit::L3Ibr) && !has_ibr_write_attr;
        let for_gather = src == Some(DfirUnit::Hbm) && dst == Some(DfirUnit::L3Ibr);
        let ibr_write = for_gather || has_ibr_write_attr;
        if ibr_write && (gather || scatter) {
            panic!(
                "DT_CHECK((!is_gather_ && !is_scatter_) && \"an IBR write must be distinct from \
                 actual gather/scatter operation\") (`Analyses/Utils.cpp:630-632`)"
            );
        }
        if gather {
            L3GatherScatter::Gather
        } else if scatter {
            L3GatherScatter::Scatter
        } else if ibr_write {
            L3GatherScatter::IbrWrite { for_gather }
        } else {
            L3GatherScatter::Plain
        }
    }

    /// `isGather()`.
    const fn is_gather(self) -> bool {
        matches!(self, L3GatherScatter::Gather)
    }

    /// `isScatter()`.
    const fn is_scatter(self) -> bool {
        matches!(self, L3GatherScatter::Scatter)
    }

    /// `isIBRWrite()`.
    const fn is_ibr_write(self) -> bool {
        matches!(self, L3GatherScatter::IbrWrite { .. })
    }
}

/// WHICH WALK REACHED THE OP — e633's over a `func`'s own three op kinds (`:117-124`) or e574's over
/// one program unit's every op (`:993-994`).
///
/// ⛔ THE UNIT TYPE IS INSIDE IT RATHER THAN BESIDE IT. `isa<mlir::func::FuncOp>(op->getParentOp())`
/// (`:606`) and `current_unit_type_` are one fact asked twice: a module-level op has no unit, and the
/// reference's own `current_unit_type_` is then whatever the last unit walked left behind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OpScope {
    /// The op sits directly in the `mlir::func::FuncOp`.
    Module,
    /// The op sits in `current_unit_`, whose type is `getUnitType(unit.getUnits()[0])` (`:1006-1007`).
    ProgramUnit(DfirUnit),
}

/// `current_unit_type_` AT AN ARM THAT READS IT, with that arm's own `DT_CHECK` discharged.
///
/// ⛔ PANICS AT MODULE SCOPE, because no arm that reads the unit type can be reached there: e633 hands
/// e524 a `sentient.constant`, a `symbol.create_symbol` or a `dataflow.get_unit` and nothing else.
fn on_unit(scope: OpScope, is_expected: impl Fn(DfirUnit) -> bool, dt_check: &str) -> DfirUnit {
    match scope {
        OpScope::ProgramUnit(unit_type) if is_expected(unit_type) => unit_type,
        _ => panic!("DT_CHECK({dt_check})"),
    }
}

/// WHERE AN OP SITS AFTER e463 HAS COPIED ONE OF ITS OPERANDS — its own path, or ONE LATER when the
/// copy landed in front of it.
///
/// ⛔ [`user_after_copy`] CANNOT ANSWER THIS ONE: it recognises the user by comparing against a
/// snapshot, and e463 has just rewritten the very operand that would make them differ. The op at the
/// user's own index is a `sentient.scalar_copy` exactly when e350 inserted one there, and none of the
/// arms that make TWO demands of one op is itself a copy.
/// ⛔ THE SHIFT IS AT MOST ONE for the reason [`user_after_copy`] gives.
fn user_after_operand_copy(unit_body: &[Op], user: &OpAt) -> OpAt {
    match user.op(unit_body) {
        Some(op) if CopyOp::of(op).is_some() => user.next(),
        Some(_) | None => user.clone(),
    }
}

/// THE OPERAND POSITIONS e524 NAMES BY `get<Name>Mutable()`, in [`dialects::operands`] order.
///
/// ⛔ A `receive_and_store`'s `$multicast_info` HAS NO FIXED POSITION: `$dst` and `$drop_first` are
/// optional and precede it, so it is `3 + present(dst) + present(drop_first)` — see
/// [`sentient::operands`]. The transfer's own is last of nine.
mod slot_at {
    /// `getMutableAddrMutable()` on all four address-carrying memory ops, and `getBoundMutable()` on
    /// a `sentient.for`.
    pub(super) const MUTABLE_ADDR: usize = 0;
    /// `getImmutableAddrMutable()`.
    pub(super) const IMMUTABLE_ADDR: usize = 1;
    /// `getLhsMutable()`, and `getMaskValueMutable()` on a `sentient.samv`.
    pub(super) const LHS: usize = 0;
    /// `getRhsMutable()`.
    pub(super) const RHS: usize = 1;
    /// `getSrcMutableAddrMutable()` — a `sentient.load_and_store` reads `$src` and `$dst` first.
    pub(super) const SRC_MUTABLE_ADDR: usize = 2;
    /// `getSrcImmutableAddrMutable()`.
    pub(super) const SRC_IMMUTABLE_ADDR: usize = 3;
    /// `getDstMutableAddrMutable()` — `$src_inc` sits between the two halves.
    pub(super) const DST_MUTABLE_ADDR: usize = 5;
    /// `getDstImmutableAddrMutable()`.
    pub(super) const DST_IMMUTABLE_ADDR: usize = 6;
    /// `getMulticastInfoMutable()` on a `sentient.load_and_store`.
    pub(super) const TRANSFER_MULTICAST_INFO: usize = 8;
    /// `getMulticastInfoMutable()` on a `sentient.receive_and_store`, BEFORE its two optional
    /// operands are counted.
    pub(super) const RECEIVE_MULTICAST_INFO: usize = 3;
}

/// ONE MEMORY OP'S ADDRESS PAIR AND WHERE EACH HALF HAS TO LIVE — the five arms that assign both
/// (`:684-706`, `:713-731`, `:790-812`, `:815-830`, `:955-976`) differ only in these four values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TransferAddrs {
    /// Where `$mutable_addr` goes — `lrf` on an L0/LX half, `lar` on an L3 one.
    mutable: RegisterLocale,
    /// Where `$immutable_addr` goes — `lrf`, or [`immutable_locale`] on an L3 half.
    immutable: RegisterLocale,
    /// `getElementSize()`, or `getSrcElementSize()` for a `load_compute_and_send` — ⭐ THE CACHE KEY
    /// ONLY. e247 reads the op's own width, which for that op is the DST one.
    element_size: Bits,
    /// `$increment` — ⛔ `None` WHERE THE ARM DOES NOT COMPUTE `treat_immutable_as_mutable`: the two
    /// L3 halves pass the flag's default `false` (`:723-725`, `:826-828`), and only the L0/LX halves
    /// ask `isUpdateMode`.
    increment: Option<Val>,
}

impl<const ADD_SCALAR_COPIES: bool> RegisterTypeAssignment<ADD_SCALAR_COPIES> {
    /// The `$mutable_addr` / `$immutable_addr` pair of one memory op, `anchor` being the value the op
    /// binds — its identity, and how the path is re-derived after the first copy shifted it.
    ///
    /// ⛔ TRAP: `mutable_addr_copy` IS READ BEFORE THE FIRST COPY (`:695`, `:801`, `:960`), so the
    /// comparison that follows is against the ORIGINAL value and not against the copy that replaced
    /// it — which is the whole point of `treat_immutable_as_mutable`: two uses of one SSA variable
    /// must not be commoned into one register when the op updates it.
    fn assign_transfer_addresses<A: Arch>(
        &mut self,
        unit_body: &mut Vec<Op>,
        anchor: Val,
        unit_type: utils::MemoryUnit,
        addrs: &TransferAddrs,
        values: &mut Values,
    ) {
        let Some(mutable) = utils::path_of(unit_body, anchor)
            .and_then(|at| OperandSlot::of(at, slot_at::MUTABLE_ADDR, unit_body))
        else {
            return;
        };
        let mutable_addr = mutable.val();
        self.add_to_work_list_create_copy_and_update_assignment(
            unit_body,
            &mutable,
            AliasKey {
                locale: addrs.mutable,
                element_size: Some(addrs.element_size),
            },
            true,
            values,
        );
        let Some(at) = utils::path_of(unit_body, anchor) else {
            return;
        };
        let Some((needs_a_copy, immutable_addr, treat_immutable_as_mutable, is_constant)) = ({
            let outer: [&[Op]; 1] = [unit_body.as_slice()];
            let defs = dialects::Definitions::from_innermost(&outer);
            at.op(unit_body)
                .and_then(utils::ImmutAddrMemoryOpInfo::of)
                .map(|info| {
                    (
                        utils::memory_op_requires_immut_addr_scalar_copy::<A>(
                            unit_type,
                            &info,
                            utils::RangeCheck::Check,
                            defs,
                        ),
                        info.immutable_addr,
                        info.immutable_addr == mutable_addr
                            && addrs
                                .increment
                                .is_some_and(|increment| is_update_mode(increment, defs)),
                        utils::is_constant(
                            info.immutable_addr,
                            utils::ConstKind::ScalarConstant,
                            defs,
                        ),
                    )
                })
        }) else {
            return;
        };
        if needs_a_copy {
            if let Some(slot) = OperandSlot::of(at, slot_at::IMMUTABLE_ADDR, unit_body) {
                self.add_to_work_list_create_copy_and_update_assignment(
                    unit_body,
                    &slot,
                    AliasKey {
                        locale: addrs.immutable,
                        element_size: Some(addrs.element_size),
                    },
                    treat_immutable_as_mutable,
                    values,
                );
            }
        } else if !is_constant {
            self.add_to_work_list_and_update_assignment(immutable_addr, addrs.immutable);
        }
    }

    /// Replaces: e524_initializeAssignmentForAnOperation
    ///
    /// Records the locale one op already fixes — a constant's `imm`, an induction variable's `lccr`, a
    /// condition's `jcr`, a MAC's XRF pointers, a memory op's address registers, a transfer's
    /// `ear`/`lar` pair — copying the operand into that locale where the value itself cannot hold it.
    ///
    /// ⛔ TRAP: THE `rhs` GUARD ASKS `isSymbol(lhs)` (`:668`) AND NOT `isSymbol(rhs)` — a copy-paste
    /// bug in the reference, kept verbatim, so a symbolic `rhs` beside a non-symbolic `lhs` is copied.
    /// ⚠️ ONE `DT_CHECK` IS DELIBERATELY ABSENT — see [`L3GatherScatter`].
    pub(crate) fn initialize_assignment_for_an_operation<A: Arch>(
        &mut self,
        unit_body: &mut Vec<Op>,
        at: &OpAt,
        scope: OpScope,
        values: &mut Values,
    ) {
        // A SNAPSHOT: every arm reads the op's operands while rewriting them, and the two arms that
        // must see the rewrite re-derive the op ([`user_after_operand_copy`], [`utils::path_of`]).
        let Some(op) = at.op(unit_body).cloned() else {
            return;
        };
        match &op {
            Op::Sentient(sentient::Op::ScalarConstant { .. })
            | Op::Dataflow(
                dataflow::Op::CreateMulticastGroup { .. } | dataflow::Op::GetUnit { .. },
            )
            | Op::Symbol(symbol::Op::CreateSymbol { .. }) => {
                if let Op::Dataflow(dataflow::Op::CreateMulticastGroup {
                    init_packet_opt_en, ..
                }) = &op
                    && *init_packet_opt_en
                    && inside_a_for(unit_body, at)
                {
                    panic!(
                        "DT_CHECK_MSG(!parent_loop || !isOptimizable(mc_op), \"an optimizable \
                         create_multicast_group op should not appear inside a loop (per \
                         pcfg-translator agreement)\") (`:600-604`)"
                    );
                }
                let Some(value) = dialects::results(&op).first().copied() else {
                    return;
                };
                match scope {
                    OpScope::Module => {
                        self.per_module
                            .global_const_assignments
                            .insert(value, RegisterLocale::Imm);
                    }
                    // No worklist entry: a constant has no operand to propagate to (`:608-610`).
                    OpScope::ProgramUnit(_) => self.update_assignment(value, RegisterLocale::Imm),
                }
            }
            // `getLocale(map_op.getValues().front())` — one locale for a whole map (`:611-615`).
            Op::Uniform(uniform::Op::DefImmutableMapping { result, pairs }) => {
                let Some((_, first)) = pairs.first() else {
                    return;
                };
                let locale = self.locale(*first).reg_type();
                self.update_assignment(*result, locale);
            }
            Op::Uniform(uniform::Op::QueryMap { result, map, .. }) => {
                let locale = self.locale(*map).reg_type();
                self.update_assignment(*result, locale);
            }
            Op::Sentient(sentient::Op::VectorMac {
                mask,
                xrf_write_ptr,
                xrf_read_ptr,
                results,
                ..
            }) => {
                on_unit(
                    scope,
                    |unit_type| {
                        matches!(unit_type, DfirUnit::PtRow(_) | DfirUnit::Sfp | DfirUnit::Pe)
                    },
                    "is_any_of(current_unit_type_, PT, SFP, PE) (`:621`)",
                );
                if xrf_write_ptr.is_none() && xrf_read_ptr.is_none() {
                    return;
                }
                if xrf_write_ptr.is_none() || xrf_read_ptr.is_none() {
                    panic!(
                        "DT_CHECK_MSG(mac_op.getPointers().size() == 2, \"unhandled number of \
                         pointer operands\") (`:623-624`)"
                    );
                }
                // `all_opnds.slice(0, 1)` and `.slice(1, 1)` of `getPointersMutable()`, which follows
                // the optional mask in [`sentient::operands`].
                let write_ptr = usize::from(mask.is_some());
                let mut user = at.clone();
                for (index, locale) in [
                    (write_ptr, RegisterLocale::XrfWrPtr),
                    (write_ptr + 1, RegisterLocale::XrfRdPtr),
                ] {
                    if let Some(slot) = OperandSlot::of(user.clone(), index, unit_body) {
                        self.add_to_work_list_create_copy_and_update_assignment(
                            unit_body,
                            &slot,
                            AliasKey {
                                locale,
                                element_size: None,
                            },
                            true,
                            values,
                        );
                    }
                    user = user_after_operand_copy(unit_body, &user);
                }
                // `if (mac_op->getNumResults() == 2)` — the advanced pointers (`:635-641`).
                if let [xrf_wr_out, xrf_rd_out] = results.as_slice() {
                    self.update_assignment(*xrf_wr_out, RegisterLocale::XrfWrPtr);
                    self.update_assignment(*xrf_rd_out, RegisterLocale::XrfRdPtr);
                }
            }
            Op::Sentient(sentient::Op::For { iv, bound, .. }) => {
                self.update_assignment(*iv, RegisterLocale::Lccr);
                let bound_is_fixed = {
                    let outer: [&[Op]; 1] = [unit_body.as_slice()];
                    let defs = dialects::Definitions::from_innermost(&outer);
                    utils::is_constant(*bound, utils::ConstKind::ScalarConstant, defs)
                        || is_symbol(*bound, defs)
                };
                // ⭐ `element_sizes[0]` IS `None`: no `sentient.for` of this island carries that
                // attribute, which is the reference's own `hasAttr == false` / `-1` (`:650-656`), and
                // the width only keys the alias cache.
                if !bound_is_fixed
                    && let Some(slot) =
                        OperandSlot::of(at.clone(), slot_at::MUTABLE_ADDR, unit_body)
                {
                    self.add_to_work_list_create_copy_and_update_assignment(
                        unit_body,
                        &slot,
                        AliasKey {
                            locale: RegisterLocale::Jcr,
                            element_size: None,
                        },
                        false,
                        values,
                    );
                }
            }
            Op::Sentient(sentient::Op::If { lhs, rhs, .. }) => {
                let (copy_lhs, copy_rhs) = {
                    let outer: [&[Op]; 1] = [unit_body.as_slice()];
                    let defs = dialects::Definitions::from_innermost(&outer);
                    let owns_its_register = |val: Val| {
                        utils::is_constant(val, utils::ConstKind::ScalarConstant, defs)
                            || is_induction_variable(val, defs)
                    };
                    (
                        !owns_its_register(*lhs) && !is_symbol(*lhs, defs),
                        !owns_its_register(*rhs) && !is_symbol(*lhs, defs),
                    )
                };
                let mut user = at.clone();
                for (index, wanted) in [(slot_at::LHS, copy_lhs), (slot_at::RHS, copy_rhs)] {
                    if wanted && let Some(slot) = OperandSlot::of(user.clone(), index, unit_body) {
                        self.add_to_work_list_create_copy_and_update_assignment(
                            unit_body,
                            &slot,
                            AliasKey {
                                locale: RegisterLocale::Jcr,
                                element_size: None,
                            },
                            false,
                            values,
                        );
                    }
                    user = user_after_operand_copy(unit_body, &user);
                }
            }
            Op::Sentient(sentient::Op::LoadAndSend {
                result,
                extent,
                increment,
                consumer,
                ..
            }) => {
                let unit_type = on_unit(
                    scope,
                    is_a_memory_unit,
                    "is_any_of(current_unit_type_, L0LU, L0SU, LXLU, LXSU, L3LU, L3SU) (`:677`)",
                );
                let Some(unit) = memory_unit(unit_type) else {
                    return;
                };
                match unit_type {
                    DfirUnit::L0lu | DfirUnit::Lxlu => {
                        self.add_to_work_list_and_update_assignment(*result, RegisterLocale::Lrf);
                        self.assign_transfer_addresses::<A>(
                            unit_body,
                            *result,
                            unit,
                            &TransferAddrs {
                                mutable: RegisterLocale::Lrf,
                                immutable: RegisterLocale::Lrf,
                                element_size: extent.element_size,
                                increment: Some(*increment),
                            },
                            values,
                        );
                    }
                    DfirUnit::L3lu | DfirUnit::L3su => {
                        self.add_to_work_list_and_update_assignment(*result, RegisterLocale::Lar);
                        self.assign_transfer_addresses::<A>(
                            unit_body,
                            *result,
                            unit,
                            &TransferAddrs {
                                mutable: RegisterLocale::Lar,
                                immutable: immutable_locale::<A>(),
                                element_size: extent.element_size,
                                increment: None,
                            },
                            values,
                        );
                        // THE `$consumer` WIRE END (`:732-747`) — a `gtr` when it names a multicast
                        // group, and otherwise whatever a `sentient.if` in front of it resolves to.
                        let consumer = consumer.val();
                        let (is_a_group, yields_a_group, yields_a_unit) = {
                            let outer: [&[Op]; 1] = [unit_body.as_slice()];
                            let defs = dialects::Definitions::from_innermost(&outer);
                            let via_if = match defs.of(consumer) {
                                Some(if_op @ Op::Sentient(sentient::Op::If { .. })) => {
                                    dialects::results(if_op)
                                        .iter()
                                        .position(|result| *result == consumer)
                                        .map(|index| {
                                            (
                                                find_yields_resolving_to(
                                                    if_op,
                                                    index,
                                                    defs,
                                                    is_a_multicast_group,
                                                ),
                                                find_yields_resolving_to(
                                                    if_op,
                                                    index,
                                                    defs,
                                                    is_a_get_unit,
                                                ),
                                            )
                                        })
                                }
                                Some(_) | None => None,
                            };
                            let (yields_a_group, yields_a_unit) = via_if.unwrap_or((false, false));
                            (is_multicast(consumer, defs), yields_a_group, yields_a_unit)
                        };
                        if is_a_group {
                            if let Some(slot) = utils::path_of(unit_body, *result)
                                .and_then(|at| OperandSlot::wire_end_of(at, unit_body))
                            {
                                self.add_to_work_list_create_copy_and_update_assignment(
                                    unit_body,
                                    &slot,
                                    AliasKey {
                                        locale: RegisterLocale::Gtr,
                                        element_size: Some(extent.element_size),
                                    },
                                    false,
                                    values,
                                );
                            }
                        } else if yields_a_group {
                            self.add_to_work_list_and_update_assignment(
                                consumer,
                                RegisterLocale::Gtr,
                            );
                        } else if yields_a_unit {
                            self.add_to_work_list_and_update_assignment(
                                consumer,
                                RegisterLocale::Ear,
                            );
                        }
                    }
                    // `op->emitError(..); signalPassFailure();` on an L0SU or an LXSU (`:759-762`).
                    _ => self.failures.push(Refused {
                        at: *result,
                        unit_type,
                        message: "Not expecting load_and_send in other units",
                    }),
                }
            }
            Op::Sentient(sentient::Op::LoadAndExtractScalar {
                addr_result,
                data_result,
                element_size,
                ..
            }) => {
                on_unit(
                    scope,
                    |unit_type| unit_type == DfirUnit::Lxlu,
                    "current_unit_type_ == LXLU (`:765`)",
                );
                self.add_to_work_list_and_update_assignment(*addr_result, RegisterLocale::Lrf);
                self.add_to_work_list_and_update_assignment(*data_result, RegisterLocale::Lrf);
                // ⭐ THE IMMUTABLE ADDRESS IS NOT TOUCHED HERE, which is why this arm does not reach
                // [`Self::assign_transfer_addresses`] (`:776-778`).
                if let Some(slot) = OperandSlot::of(at.clone(), slot_at::MUTABLE_ADDR, unit_body) {
                    self.add_to_work_list_create_copy_and_update_assignment(
                        unit_body,
                        &slot,
                        AliasKey {
                            locale: RegisterLocale::Lrf,
                            element_size: Some(*element_size),
                        },
                        true,
                        values,
                    );
                }
            }
            Op::Sentient(sentient::Op::ReceiveAndStore {
                result,
                extent,
                increment,
                producer,
                dst,
                drop_first,
                multicast_info,
                ..
            }) => {
                let unit_type = on_unit(
                    scope,
                    is_a_memory_unit,
                    "is_any_of(current_unit_type_, L0LU, L0SU, LXLU, LXSU, L3LU, L3SU) (`:780`)",
                );
                let Some(unit) = memory_unit(unit_type) else {
                    return;
                };
                match unit_type {
                    DfirUnit::L0su | DfirUnit::Lxsu => {
                        self.add_to_work_list_and_update_assignment(*result, RegisterLocale::Lrf);
                        self.assign_transfer_addresses::<A>(
                            unit_body,
                            *result,
                            unit,
                            &TransferAddrs {
                                mutable: RegisterLocale::Lrf,
                                immutable: RegisterLocale::Lrf,
                                element_size: extent.element_size,
                                increment: Some(*increment),
                            },
                            values,
                        );
                    }
                    DfirUnit::L3lu | DfirUnit::L3su => {
                        self.add_to_work_list_and_update_assignment(*result, RegisterLocale::Lar);
                        self.assign_transfer_addresses::<A>(
                            unit_body,
                            *result,
                            unit,
                            &TransferAddrs {
                                mutable: RegisterLocale::Lar,
                                immutable: immutable_locale::<A>(),
                                element_size: extent.element_size,
                                increment: None,
                            },
                            values,
                        );
                        if multicast_info.is_some() {
                            let index = slot_at::RECEIVE_MULTICAST_INFO
                                + usize::from(dst.is_some())
                                + usize::from(drop_first.is_some());
                            if let Some(slot) = utils::path_of(unit_body, *result)
                                .and_then(|at| OperandSlot::of(at, index, unit_body))
                            {
                                self.add_to_work_list_create_copy_and_update_assignment(
                                    unit_body,
                                    &slot,
                                    AliasKey {
                                        locale: RegisterLocale::Gtr,
                                        element_size: Some(extent.element_size),
                                    },
                                    false,
                                    values,
                                );
                            }
                        }
                        // THE `$producer` WIRE END (`:840-854`) — an `ear` when a `sentient.if` in
                        // front of it resolves to a unit handle. ⛔ AND NO MULTICAST ARM HERE, unlike
                        // the send's consumer.
                        let producer = producer.val();
                        let yields_a_unit = {
                            let outer: [&[Op]; 1] = [unit_body.as_slice()];
                            let defs = dialects::Definitions::from_innermost(&outer);
                            match defs.of(producer) {
                                Some(if_op @ Op::Sentient(sentient::Op::If { .. })) => {
                                    dialects::results(if_op)
                                        .iter()
                                        .position(|result| *result == producer)
                                        .is_some_and(|index| {
                                            find_yields_resolving_to(
                                                if_op,
                                                index,
                                                defs,
                                                is_a_get_unit,
                                            )
                                        })
                                }
                                Some(_) | None => false,
                            }
                        };
                        if yields_a_unit {
                            self.add_to_work_list_and_update_assignment(
                                producer,
                                RegisterLocale::Ear,
                            );
                        }
                    }
                    // `op->emitError(..); signalPassFailure();` on an L0LU or an LXLU (`:855-857`).
                    _ => self.failures.push(Refused {
                        at: *result,
                        unit_type,
                        message: "Not expecting receive_and_store in other units",
                    }),
                }
            }
            Op::Sentient(sentient::Op::LoadAndStore {
                src,
                dst,
                results,
                extent,
                multicast_info,
                is_ibr_write,
                ..
            }) => {
                let unit_type = on_unit(
                    scope,
                    |unit_type| matches!(unit_type, DfirUnit::L3lu | DfirUnit::L3su),
                    "is_any_of(current_unit_type_, L3LU, L3SU) (`:861`)",
                );
                let (transfer, src_unit_type, dst_unit_type) = {
                    let outer: [&[Op]; 1] = [unit_body.as_slice()];
                    let defs = dialects::Definitions::from_innermost(&outer);
                    let src_unit_type = unit_type_of(*src, defs);
                    let dst_unit_type = unit_type_of(*dst, defs);
                    (
                        L3GatherScatter::of(src_unit_type, dst_unit_type, *is_ibr_write),
                        src_unit_type,
                        dst_unit_type,
                    )
                };
                match transfer {
                    L3GatherScatter::Gather if unit_type != DfirUnit::L3lu => {
                        panic!("DT_CHECK(current_unit_type_ == L3LU) (`:874`)")
                    }
                    L3GatherScatter::Scatter if unit_type != DfirUnit::L3su => {
                        panic!("DT_CHECK(current_unit_type_ == L3SU) (`:876`)")
                    }
                    L3GatherScatter::IbrWrite { .. } | L3GatherScatter::Plain
                        if !((src_unit_type == Some(DfirUnit::Lx)
                            && unit_type == DfirUnit::L3su)
                            || (src_unit_type == Some(DfirUnit::Hbm)
                                && unit_type == DfirUnit::L3lu)) =>
                    {
                        panic!(
                            "DT_CHECK(((src_unit_type == LX && current_unit_type_ == L3SU) || \
                             (src_unit_type == HBM && current_unit_type_ == L3LU)) && \"unexpected \
                             producer/consumer unit\") (`:878-880`)"
                        )
                    }
                    L3GatherScatter::Gather
                    | L3GatherScatter::Scatter
                    | L3GatherScatter::IbrWrite { .. }
                    | L3GatherScatter::Plain => {}
                }
                let (src_locale, dst_locale) = match transfer {
                    L3GatherScatter::Gather => (RegisterLocale::Ear, RegisterLocale::Lar),
                    L3GatherScatter::Scatter => (RegisterLocale::Lar, RegisterLocale::Ear),
                    // ⭐ AN IBR WRITE'S DESTINATION STILL TAKES A `lar`: every use of an IBR register
                    // is implicit, but the senulator increments the LAR of an ibr-write LDIMU, so the
                    // register has to count as used (`:888-895`).
                    L3GatherScatter::IbrWrite { .. } => {
                        if unit_type == DfirUnit::L3lu {
                            (RegisterLocale::Ear, RegisterLocale::Lar)
                        } else {
                            (RegisterLocale::Lar, RegisterLocale::Ear)
                        }
                    }
                    L3GatherScatter::Plain => (
                        if src_unit_type == Some(DfirUnit::Lx) {
                            RegisterLocale::Lar
                        } else {
                            RegisterLocale::Ear
                        },
                        if dst_unit_type == Some(DfirUnit::Lx) {
                            RegisterLocale::Lar
                        } else {
                            RegisterLocale::Ear
                        },
                    ),
                };
                let immutable = immutable_locale::<A>();
                let imm_locale = |locale: RegisterLocale, jcr: bool| {
                    if locale == RegisterLocale::Ear {
                        if jcr {
                            RegisterLocale::Jcr
                        } else {
                            RegisterLocale::Ebr
                        }
                    } else {
                        immutable
                    }
                };
                self.add_to_work_list_and_update_assignment(results.0, src_locale);
                self.add_to_work_list_and_update_assignment(results.1, dst_locale);
                let mut demands = vec![
                    (slot_at::SRC_MUTABLE_ADDR, src_locale, true),
                    (
                        slot_at::SRC_IMMUTABLE_ADDR,
                        imm_locale(src_locale, transfer.is_gather()),
                        false,
                    ),
                ];
                if !(transfer.is_ibr_write() && unit_type == DfirUnit::L3su) {
                    demands.push((slot_at::DST_MUTABLE_ADDR, dst_locale, true));
                }
                if !transfer.is_ibr_write() {
                    demands.push((
                        slot_at::DST_IMMUTABLE_ADDR,
                        imm_locale(dst_locale, transfer.is_scatter()),
                        false,
                    ));
                }
                if multicast_info.is_some() {
                    demands.push((slot_at::TRANSFER_MULTICAST_INFO, RegisterLocale::Gtr, false));
                }
                // ⛔ THE PATH IS RE-DERIVED PER SLOT, as e523's own operand loop does: each copy can
                // land in front of the transfer.
                for (index, locale, is_mutable_addr) in demands {
                    let Some(slot) = utils::path_of(unit_body, results.0)
                        .and_then(|at| OperandSlot::of(at, index, unit_body))
                    else {
                        continue;
                    };
                    self.add_to_work_list_create_copy_and_update_assignment(
                        unit_body,
                        &slot,
                        AliasKey {
                            locale,
                            element_size: Some(extent.element_size),
                        },
                        is_mutable_addr,
                        values,
                    );
                }
            }
            Op::Sentient(sentient::Op::ReceiveAndExtractScalar { result, .. }) => {
                on_unit(
                    scope,
                    |unit_type| unit_type == DfirUnit::Lxsu,
                    "current_unit_type_ == LXSU (`:936`)",
                );
                self.add_to_work_list_and_update_assignment(*result, RegisterLocale::Lrf);
            }
            Op::Sentient(sentient::Op::LoadComputeAndSend {
                result,
                src_element_size,
                increment,
                ..
            }) => {
                let unit_type = on_unit(
                    scope,
                    |unit_type| unit_type == DfirUnit::Lxlu,
                    "current_unit_type_ == LXLU (`:940`)",
                );
                let Some(unit) = memory_unit(unit_type) else {
                    return;
                };
                self.add_to_work_list_and_update_assignment(*result, RegisterLocale::Lrf);
                self.assign_transfer_addresses::<A>(
                    unit_body,
                    *result,
                    unit,
                    &TransferAddrs {
                        mutable: RegisterLocale::Lrf,
                        immutable: RegisterLocale::Lrf,
                        element_size: *src_element_size,
                        increment: Some(*increment),
                    },
                    values,
                );
            }
            Op::Sentient(sentient::Op::Samv { .. }) => {
                on_unit(
                    scope,
                    |unit_type| unit_type == DfirUnit::Lxlu,
                    "current_unit_type_ == LXLU (`:973`)",
                );
                if let Some(slot) = OperandSlot::of(at.clone(), slot_at::LHS, unit_body) {
                    self.add_to_work_list_create_copy_and_update_assignment(
                        unit_body,
                        &slot,
                        AliasKey {
                            locale: RegisterLocale::Mvr,
                            element_size: None,
                        },
                        false,
                        values,
                    );
                }
            }
            // The `else if` chain ends here: every other op's locale is decided by propagation.
            _ => {}
        }
    }
}

impl<const ADD_SCALAR_COPIES: bool> RegisterTypeAssignment<ADD_SCALAR_COPIES> {
    /// Replaces: e574_initializeWorkList
    ///
    /// Starts one unit from nothing but the module's constant locales, then records what every op in
    /// it already fixes (`:987-995`).
    ///
    /// ⭐ `std::map::insert` DOES NOT OVERWRITE, AND HERE IT CANNOT MATTER: `clear()` empties
    /// `assignments_` on the line above (`:988`, `:222-227`) and `global_const_assignments_` is a map,
    /// so every key the loop seeds is fresh. The `or_insert` is the reference's spelling, not a live
    /// tie-break — nothing this unit assigned survives to be kept.
    pub(crate) fn initialize_work_list<A: Arch>(
        &mut self,
        unit_type: DfirUnit,
        unit_body: &mut Vec<Op>,
        values: &mut Values,
    ) {
        self.clear();
        for (value, locale) in self.per_module.global_const_assignments.clone() {
            self.per_unit.assignments.entry(value).or_insert(locale);
        }
        self.initialize_assignments_in::<A>(unit_type, unit_body, &[], values);
    }

    /// The `walk<WalkOrder::PreOrder>` of `:992-993`, one block at a time.
    ///
    /// ⛔ TRAP: e524 CREATES COPIES IN FRONT OF THE OP IT IS GIVEN, and MLIR's walk holds an iterator
    /// that an insertion before the current op does not move. The cursor therefore steps over
    /// whatever appeared ahead of the walked op, which is not itself walked.
    fn initialize_assignments_in<A: Arch>(
        &mut self,
        unit_type: DfirUnit,
        unit_body: &mut Vec<Op>,
        enclosing: &[(InBlock, usize)],
        values: &mut Values,
    ) {
        let mut index = 0;
        loop {
            let at = OpAt::at(enclosing, InBlock(index));
            let Some(before) = at.block(unit_body).map(<[Op]>::len) else {
                return;
            };
            if index >= before {
                return;
            }
            self.initialize_assignment_for_an_operation::<A>(
                unit_body,
                &at,
                OpScope::ProgramUnit(unit_type),
                values,
            );
            index += at
                .block(unit_body)
                .map_or(0, |block| block.len().saturating_sub(before));
            let at = OpAt::at(enclosing, InBlock(index));
            let regions = at
                .op(unit_body)
                .map_or(0, |op| dialects::regions_ref(op).len());
            for region in 0..regions {
                let mut inner = enclosing.to_vec();
                inner.push((InBlock(index), region));
                self.initialize_assignments_in::<A>(unit_type, unit_body, &inner, values);
            }
            index += 1;
        }
    }
}

/// Whether some `sentient.yield`, `sentient.scalar_copy` or `uniform.def_immutable_mapping` OF THIS
/// UNIT reads `val` — the `llvm::none_of` over the value's uses (`:1035-1042`), asked of the unit
/// body instead of of the value.
fn yielded_copied_or_mapped_in(val: Val, block: &[Op]) -> bool {
    block.iter().any(|op| {
        (matches!(
            op,
            Op::Sentient(sentient::Op::Yield { .. } | sentient::Op::ScalarCopy { .. })
                | Op::Uniform(uniform::Op::DefImmutableMapping { .. })
        ) && dialects::operands(op).contains(&val))
            || dialects::regions_ref(op)
                .into_iter()
                .any(|region| yielded_copied_or_mapped_in(val, region))
    })
}

impl<const ADD_SCALAR_COPIES: bool> RegisterTypeAssignment<ADD_SCALAR_COPIES> {
    /// Replaces: e609_runOn
    ///
    /// One program unit end to end: seed what the hardware already fixes, propagate to a fixed point,
    /// write the locales back — and on an L3 half only, give every function-level `dataflow.get_unit`
    /// this unit yields, copies or maps a `regLocale` of its own so the right assign instruction can
    /// be built for it (`:996-1050`).
    ///
    /// ⛔ TRAP: THE DIRTY QUEUE IS RECORDED AND THEN DRAINED ANYWAY — `signalPassFailure` neither
    /// clears `worklist_` nor returns, so the propagation below starts from whatever was left.
    /// ⭐ `getUses().empty()` IS DROPPED AS A FAST PATH: no use at all makes the `none_of` below
    /// vacuously true, so both spellings `continue` on exactly the same ops.
    pub(crate) fn run_on<A: Arch>(
        &mut self,
        preamble: &mut [Op],
        unit: &mut crate::islands::sentient::ProgramUnit<A>,
        values: &mut Values,
    ) {
        // `current_unit_type_ = dcc::getUnitType(getUnits()[0].getDefiningOp<GetUnitOp>())`, which is
        // what [`crate::islands::dataflow_ir::Units`] carries rather than re-derives.
        let unit_type = unit.on.kind();
        if let Some(left) = self.worklist.front().copied() {
            self.failures.push(Refused {
                at: left,
                unit_type,
                message: "Register type assignment internal worklist is not empty before processing \
                          a unit",
            });
        }
        self.initialize_work_list::<A>(unit_type, &mut unit.body, values);
        self.process_work_list(&mut unit.body, values);
        self.update_program_unit(&mut unit.body, unit_type);
        if !matches!(unit_type, DfirUnit::L3lu | DfirUnit::L3su) {
            return;
        }
        for op in preamble.iter_mut() {
            let Op::Dataflow(dataflow::Op::GetUnit { result, .. }) = op else {
                continue;
            };
            let handle = *result;
            if self.locale(handle) == Locale::Unrecorded
                || !yielded_copied_or_mapped_in(handle, &unit.body)
            {
                continue;
            }
            dialects::set_reg_locale_on(op, handle, self.locale(handle).reg_type());
        }
    }
}

impl<const ADD_SCALAR_COPIES: bool> RegisterTypeAssignment<ADD_SCALAR_COPIES> {
    /// Replaces: e633_runOnOperation
    ///
    /// The pass entry: record the module-level `imm` locales — constants, then symbols, then unit
    /// handles — and then run [`Self::run_on`] over every program unit (`:113-129`).
    ///
    /// ⛔ THE THREE MODULE WALKS ARE THREE PASSES OVER THE SAME BLOCK, IN THAT ORDER, and the order
    /// is observable: `global_const_assignments` is a map, but e524's [`OpScope::Module`] arm
    /// `insert`s, so the FIRST of two ops binding one value wins.
    /// ⛔ NEITHER WALK DESCENDS: `func.getOps<T>()` is the func's own block, so a constant inside a
    /// unit is e574's to find, not this one's — and the [`OpScope::Module`] arm creates no copies, so
    /// no cursor can be stepped over here.
    pub(crate) fn run_on_operation<A: Arch, M: Model, W: Workload>(
        &mut self,
        program: &mut Program<A, M, W>,
        values: &mut Values,
    ) {
        if DISABLE_THIS_PASS {
            return;
        }
        let Program {
            preamble, units, ..
        } = program;
        for is_kind in [
            (|op| matches!(op, Op::Sentient(sentient::Op::ScalarConstant { .. })))
                as fn(&Op) -> bool,
            |op| matches!(op, Op::Symbol(symbol::Op::CreateSymbol { .. })),
            |op| matches!(op, Op::Dataflow(dataflow::Op::GetUnit { .. })),
        ] {
            for index in 0..preamble.len() {
                let at = OpAt::top(InBlock(index));
                if at.op(preamble).is_some_and(is_kind) {
                    self.initialize_assignment_for_an_operation::<A>(
                        preamble,
                        &at,
                        OpScope::Module,
                        values,
                    );
                }
            }
        }
        for unit in units.iter_mut() {
            self.run_on::<A>(preamble, unit, values);
        }
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::arch::{Dd2, Elements};
    use crate::generated::OpFunc;
    use crate::islands::dataflow_ir::ty::ScalarTy;
    use crate::islands::dataflow_ir::{GroupId, OpIndex, ProgramName, Units};
    use crate::islands::sentient::dialects::sentient::{Extent, Reg, RegType, ShuffleMode};
    use crate::islands::sentient::{ProgramUnit, ProgramUnits};
    use crate::units::Residency;

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

    /// `%out = sentient.scalar_copy %inp`.
    fn copy(input: Val, result: Val) -> Op {
        Op::Sentient(sentient::Op::ScalarCopy {
            input,
            result,
            reg: Reg {
                locale: RegType::Unknown,
                index: None,
            },
            element_size: None,
            program_header: false,
        })
    }

    #[test]
    fn an_imm_record_and_a_register_demand_is_the_only_copy() {
        let mut pass = TypesAndCopies::new();
        // Unrecorded: nothing to copy from.
        assert_eq!(pass.is_copy_needed(Val(1), RegType::Lrf), CopyNeeded::No);
        pass.update_assignment(Val(1), RegType::Imm);
        assert_eq!(pass.is_copy_needed(Val(1), RegType::Lrf), CopyNeeded::Yes);
        // An agreeing demand, and an `unrelated` one, both leave the value alone.
        assert_eq!(pass.is_copy_needed(Val(1), RegType::Imm), CopyNeeded::No);
        assert_eq!(
            pass.is_copy_needed(Val(1), RegType::Unrelated),
            CopyNeeded::No
        );
        // A record that is neither `imm` nor the demand is the reference's DT_CHECK.
        pass.update_assignment(Val(2), RegType::Lccr);
        assert_eq!(
            pass.is_copy_needed(Val(2), RegType::Lrf),
            CopyNeeded::MultipleAssignments {
                recorded: RegType::Lccr
            }
        );
        // ⭐ `skip_copies = true` REMOVES THE `Yes` ARM, on the very same state.
        let mut types_only = TypesOnly::new();
        types_only.update_assignment(Val(1), RegType::Imm);
        assert_eq!(
            types_only.is_copy_needed(Val(1), RegType::Lrf),
            CopyNeeded::No
        );
    }

    #[test]
    fn clear_drops_the_units_four_maps_and_keeps_the_modules_two() {
        let mut pass = TypesAndCopies::new();
        let key = AliasKey {
            locale: RegType::Lrf,
            element_size: Some(Bits(32)),
        };
        pass.update_assignment(Val(1), RegType::Imm);
        pass.per_unit
            .cst_aliases
            .insert(7, BTreeMap::from([(key, Val(2))]));
        pass.per_unit
            .query_map_aliases
            .insert(Val(3), BTreeMap::from([(key, Val(4))]));
        pass.per_unit
            .unit_aliases
            .insert(Core::checked(0), BTreeMap::from([(DfirUnit::Sfp, Val(5))]));
        pass.per_module
            .global_const_assignments
            .insert(Val(6), RegType::Imm);
        pass.per_module
            .sym_aliases
            .insert(-43, BTreeMap::from([(key, Val(8))]));
        pass.worklist.push_back(Val(9));

        pass.clear();

        assert_eq!(pass.per_unit, PerUnit::default());
        assert_eq!(pass.per_module.global_const_assignments.len(), 1);
        assert_eq!(pass.per_module.sym_aliases.len(), 1);
        // `clear()` does not name the worklist either.
        assert_eq!(pass.worklist.len(), 1);
    }

    #[test]
    fn an_unrecorded_value_has_no_locale_and_reads_as_unknown() {
        let mut pass = TypesAndCopies::new();
        assert_eq!(pass.locale(Val(1)), Locale::Unrecorded);
        assert_eq!(pass.locale(Val(1)).reg_type(), RegType::Unknown);
        // ⭐ AND A DELIBERATE `unknown` IS NOT THE SAME ANSWER.
        pass.update_assignment(Val(1), RegType::Unknown);
        assert_eq!(pass.locale(Val(1)), Locale::Recorded(RegType::Unknown));
    }

    #[test]
    fn only_a_scalar_copy_is_a_copy_op() {
        let copy_op = copy(Val(1), Val(2));
        assert!(CopyOp::of(&copy_op).is_some());
        assert!(CopyOp::of(&constant(Val(1), 3)).is_none());
    }

    #[test]
    fn an_assigned_value_is_not_queued() {
        let mut pass = TypesAndCopies::new();
        pass.add_to_work_list(Val(1));
        pass.update_assignment(Val(1), RegType::Imm);
        pass.add_to_work_list(Val(1));
        pass.add_to_work_list(Val(2));
        assert_eq!(pass.worklist, VecDeque::from([Val(1), Val(2)]));
    }

    #[test]
    fn the_first_assignment_wins() {
        let mut pass = TypesAndCopies::new();
        pass.update_assignment(Val(1), RegType::Imm);
        pass.update_assignment(Val(1), RegType::Lrf);
        assert_eq!(pass.locale(Val(1)), Locale::Recorded(RegType::Imm));
    }

    /// `%r = sentient.scalar_add %lhs, %rhs`, with no register assigned yet.
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

    /// `sentient.for %iv = 0 to %bound iter_args(%arg = %init) -> %result { <body> }`.
    fn for_op(iv: Val, bound: Val, carried: sentient::Carried, body: Vec<Op>) -> Op {
        Op::Sentient(sentient::Op::For {
            iv,
            bound,
            bound_reg: None,
            carried: vec![carried],
            dbg_name: None,
            body,
        })
    }

    /// One carried value, with nothing assigned to it yet.
    fn carried(init: Val, arg: Val, result: Val) -> sentient::Carried {
        sentient::Carried {
            init,
            arg,
            result,
            reg: Reg {
                locale: RegType::Unknown,
                index: None,
            },
            program_header: false,
            element_size: None,
        }
    }

    #[test]
    fn queueing_before_recording_is_what_makes_the_value_reachable() {
        let mut pass = TypesAndCopies::new();
        pass.add_to_work_list_and_update_assignment(Val(1), RegType::Imm);
        assert_eq!(pass.worklist, VecDeque::from([Val(1)]));
        assert_eq!(pass.locale(Val(1)), Locale::Recorded(RegType::Imm));
        // ⭐ A SECOND CALL RECORDS NOTHING NEW AND QUEUES NOTHING.
        pass.add_to_work_list_and_update_assignment(Val(1), RegType::Lrf);
        assert_eq!(pass.worklist, VecDeque::from([Val(1)]));
        assert_eq!(pass.locale(Val(1)), Locale::Recorded(RegType::Imm));
    }

    /// `e352` — the nested walk, the L3-only `get_unit` arm and the induction variable's entry 0.
    #[test]
    fn every_recorded_locale_lands_on_its_op_including_the_induction_variable() {
        let mut pass = TypesAndCopies::new();
        pass.update_assignment(Val(0), RegType::Imm); // the loop bound's constant
        pass.update_assignment(Val(1), RegType::Jcr); // the induction variable — entry 0
        pass.update_assignment(Val(3), RegType::Lrf); // the carried argument
        pass.update_assignment(Val(4), RegType::Lccr); // the carried result
        pass.update_assignment(Val(6), RegType::Lbr); // the body's scalar_add
        let mut body = vec![
            constant(Val(0), 8),
            for_op(
                Val(1),
                Val(0),
                carried(Val(2), Val(3), Val(4)),
                vec![add(Val(3), Val(5), Val(6))],
            ),
        ];

        pass.update_program_unit(&mut body, DfirUnit::Lxlu);

        let scope: [&[Op]; 1] = [body.as_slice()];
        let defs = dialects::Definitions::from_innermost(&scope);
        assert_eq!(dialects::value_reg_locale(Val(0), defs), RegType::Imm);
        // ⭐ THE RESULT'S LOCALE WON THE ONE `Carried::reg` THE ISLAND HOLDS.
        assert_eq!(dialects::value_reg_locale(Val(4), defs), RegType::Lccr);
        assert_eq!(dialects::value_reg_locale(Val(6), defs), RegType::Lbr);
        // ⭐ THE INDUCTION VARIABLE'S `jcr` LANDED IN `bound_reg`, entry 0 of the array.
        assert_eq!(dialects::value_reg_locale(Val(1), defs), RegType::Jcr);
    }

    /// `e350` — a fresh copy before its use, then the same demand served from the constant cache.
    #[test]
    fn a_second_demand_for_one_constant_reuses_the_first_copy() {
        let mut pass = TypesAndCopies::new();
        let mut values = Values::default();
        for _ in 0..3 {
            let _ = values.mint();
        }
        // `%0 = scalar_constant 8`, then two users of it.
        let mut body = vec![
            constant(Val(0), 8),
            add(Val(0), Val(1), Val(2)),
            add(Val(0), Val(1), Val(3)),
        ];
        let key = AliasKey {
            locale: RegType::Lrf,
            element_size: Some(Bits(32)),
        };
        let first = pass.create_copy_operation_and_update_assignment(
            &mut body,
            &CopyDemand {
                val: Val(0),
                key,
                user: OpAt::top(utils::InBlock(1)),
                is_mutable_addr_or_xrf_ptr: false,
            },
            &mut values,
        );
        let copy = first.expect("a constant is one of the five definitions");
        // ⛔ THE COPY WENT IN FRONT OF THE USER, WHICH SHIFTED IT.
        assert_eq!(body.len(), 4);
        assert!(CopyOp::of(&body[1]).is_some());
        assert_eq!(pass.locale(copy), Locale::Recorded(RegType::Lrf));
        assert_eq!(pass.per_unit.cst_aliases[&8][&key], copy);

        // The second demand is served from the cache: no new op, and the same value back.
        let again = pass.create_copy_operation_and_update_assignment(
            &mut body,
            &CopyDemand {
                val: Val(0),
                key,
                user: OpAt::top(utils::InBlock(3)),
                is_mutable_addr_or_xrf_ptr: false,
            },
            &mut values,
        );
        assert_eq!(again, Some(copy));
        assert_eq!(body.len(), 4);
        // ⛔ AND A DEFINITION THAT IS NONE OF THE FIVE OPS IS THE `DT_CHECK`, ANSWERED AS `None`.
        assert_eq!(
            pass.create_copy_operation_and_update_assignment(
                &mut body,
                &CopyDemand {
                    val: Val(2),
                    key,
                    user: OpAt::top(utils::InBlock(3)),
                    is_mutable_addr_or_xrf_ptr: false,
                },
                &mut values,
            ),
            None
        );
    }

    #[test]
    fn print_assignments_names_the_owning_op_then_the_locale() {
        let mut pass = TypesAndCopies::new();
        pass.update_assignment(Val(7), RegType::Imm);
        pass.update_assignment(Val(9), RegType::Lbr);
        let body = vec![constant(Val(7), 4)];
        let mut out = String::new();
        pass.print_assignments(&mut out, &[&body]);
        assert!(out.starts_with('\n'), "{out}");
        assert!(out.contains("sentient.scalar_constant"), "{out}");
        assert!(out.contains("- locale: imm"), "{out}");
        // An ownerless value still reports its locale, in key order after `%7`.
        assert!(out.ends_with("- locale: lbr"), "{out}");
    }
    /// e463 — the copy goes in front of the reader, and the reader's own operand is the only one
    /// rewritten, the insertion having shifted the reader by one.
    #[test]
    fn e463_the_copy_replaces_the_operand_that_asked_for_it() {
        let mut pass = TypesAndCopies::new();
        let mut values = Values::default();
        for _ in 0..4 {
            let _ = values.mint();
        }
        let mut body = vec![
            constant(Val(0), 8),
            add(Val(0), Val(1), Val(2)),
            add(Val(0), Val(1), Val(3)),
        ];
        pass.update_assignment(Val(0), RegType::Imm);
        let slot = OperandSlot::of(OpAt::top(utils::InBlock(1)), 0, &body)
            .expect("the add reads the constant as its first operand");
        assert_eq!(slot.val(), Val(0));

        let installed = pass.add_to_work_list_create_copy_and_update_assignment(
            &mut body,
            &slot,
            AliasKey {
                locale: RegType::Lrf,
                element_size: None,
            },
            false,
            &mut values,
        );
        let copy = match installed {
            CopyInstalled::Assigned(copy) => copy,
            other => panic!("expected the operand to be assigned: {other:?}"),
        };
        assert_eq!(body.len(), 4);
        assert!(CopyOp::of(&body[1]).is_some());
        // ⭐ THE SHIFTED READER, AND ONLY IT: the second user still reads the constant.
        assert_eq!(dialects::operands(&body[2]), vec![copy, Val(1)]);
        assert_eq!(dialects::operands(&body[3]), vec![Val(0), Val(1)]);
    }

    /// e463 — for `gtr` the copy takes over every OTHER reader and keeps reading the original itself.
    #[test]
    fn e463_a_gtr_copy_takes_over_every_other_use() {
        let mut pass = TypesAndCopies::new();
        let mut values = Values::default();
        for _ in 0..4 {
            let _ = values.mint();
        }
        let mut body = vec![
            constant(Val(0), 8),
            add(Val(0), Val(1), Val(2)),
            add(Val(0), Val(1), Val(3)),
        ];
        pass.update_assignment(Val(0), RegType::Imm);
        let slot = OperandSlot::of(OpAt::top(utils::InBlock(1)), 0, &body)
            .expect("the add reads the constant as its first operand");
        let installed = pass.add_to_work_list_create_copy_and_update_assignment(
            &mut body,
            &slot,
            AliasKey {
                locale: RegType::Gtr,
                element_size: None,
            },
            false,
            &mut values,
        );
        let copy = match installed {
            CopyInstalled::AllUsesRedirected(copy) => copy,
            other => panic!("expected every use to be redirected: {other:?}"),
        };
        assert_eq!(dialects::operands(&body[1]), vec![Val(0)]);
        assert_eq!(dialects::operands(&body[2]), vec![copy, Val(1)]);
        assert_eq!(dialects::operands(&body[3]), vec![copy, Val(1)]);
    }
    /// e523 — one drain, running to empty, carries a loop result's locale all the way round its
    /// interface: the yield, the iterator argument, and from there the initialiser.
    #[test]
    fn the_drain_carries_a_loop_results_locale_round_its_whole_interface() {
        let mut pass = TypesAndCopies::new();
        let mut values = Values::default();
        let mut body = vec![
            constant(Val(0), 8),
            for_op(
                Val(1),
                Val(0),
                carried(Val(2), Val(3), Val(4)),
                vec![Op::Sentient(sentient::Op::Yield {
                    results: vec![Val(5)],
                })],
            ),
        ];
        // e524's own two records, which are what makes the null `def` of the trap unreachable.
        pass.update_assignment(Val(1), RegType::Lccr);
        pass.add_to_work_list_and_update_assignment(Val(4), RegType::Lrf);

        pass.process_work_list(&mut body, &mut values);

        assert!(pass.worklist.is_empty());
        assert_eq!(pass.locale(Val(5)), Locale::Recorded(RegType::Lrf));
        assert_eq!(pass.locale(Val(3)), Locale::Recorded(RegType::Lrf));
        // ⭐ THE ARG'S OWN TURN IN THE DRAIN IS WHAT REACHED THE INITIALISER (position >= 1).
        assert_eq!(pass.locale(Val(2)), Locale::Recorded(RegType::Lrf));
        // ⛔ AND THE INDUCTION VARIABLE'S `lccr` SURVIVED: first writer wins.
        assert_eq!(pass.locale(Val(1)), Locale::Recorded(RegType::Lccr));
    }

    /// `%u = dataflow.get_unit {type = <unit>}`.
    fn get_unit(result: Val, unit: DfirUnit) -> Op {
        Op::Dataflow(dataflow::Op::GetUnit {
            result,
            residency: Residency::Global,
            unit,
            num_folds: None,
            reg_locale: None,
        })
    }

    /// e524 — the module/unit split for a constant, the loop's `lccr`, and the `isSymbol(lhs)` slip.
    #[test]
    fn e524_records_what_the_hardware_fixes_and_copies_nothing_it_need_not() {
        let mut pass = TypesAndCopies::new();
        let mut values = Values::default();
        let mut body = vec![
            constant(Val(0), 8),
            for_op(Val(1), Val(0), carried(Val(2), Val(3), Val(4)), Vec::new()),
        ];

        pass.initialize_assignment_for_an_operation::<Dd2>(
            &mut body,
            &OpAt::top(utils::InBlock(0)),
            OpScope::Module,
            &mut values,
        );
        // ⭐ A MODULE-LEVEL CONSTANT GOES TO THE MAP `clear()` KEEPS, AND IS NOT QUEUED.
        assert_eq!(
            pass.per_module.global_const_assignments[&Val(0)],
            RegType::Imm
        );
        assert_eq!(pass.locale(Val(0)), Locale::Unrecorded);
        assert!(pass.worklist.is_empty());

        pass.initialize_assignment_for_an_operation::<Dd2>(
            &mut body,
            &OpAt::top(utils::InBlock(1)),
            OpScope::ProgramUnit(DfirUnit::Lxlu),
            &mut values,
        );
        assert_eq!(pass.locale(Val(1)), Locale::Recorded(RegType::Lccr));
        // A constant bound already owns its register: no `jcr` copy went in.
        assert_eq!(body.len(), 2);

        // ⛔ THE TRAP: with a SYMBOLIC `lhs` the `rhs` guard reads `isSymbol(lhs)` and so leaves the
        // non-symbolic `rhs` alone as well.
        let mut with_if = vec![
            Op::Symbol(symbol::Op::CreateSymbol {
                result: Val(10),
                symbol_id: 0,
                max_value: None,
            }),
            Op::Sentient(sentient::Op::If {
                predicate: sentient::CmpPredicate::Slt,
                lhs: Val(10),
                rhs: Val(11),
                yielded: Vec::new(),
                dbg_name: None,
                then_body: Vec::new(),
                else_body: Vec::new(),
            }),
        ];
        pass.initialize_assignment_for_an_operation::<Dd2>(
            &mut with_if,
            &OpAt::top(utils::InBlock(1)),
            OpScope::ProgramUnit(DfirUnit::Lxlu),
            &mut values,
        );
        assert_eq!(pass.locale(Val(11)), Locale::Unrecorded);
        // ⭐ AND A NON-SYMBOLIC `lhs` IS WHAT LETS THE SAME `rhs` BE ASKED FOR A `jcr`.
        let mut without = vec![Op::Sentient(sentient::Op::If {
            predicate: sentient::CmpPredicate::Slt,
            lhs: Val(12),
            rhs: Val(11),
            yielded: Vec::new(),
            dbg_name: None,
            then_body: Vec::new(),
            else_body: Vec::new(),
        })];
        pass.initialize_assignment_for_an_operation::<Dd2>(
            &mut without,
            &OpAt::top(utils::InBlock(0)),
            OpScope::ProgramUnit(DfirUnit::Lxlu),
            &mut values,
        );
        assert_eq!(pass.locale(Val(11)), Locale::Recorded(RegType::Jcr));
        assert_eq!(pass.locale(Val(12)), Locale::Recorded(RegType::Jcr));
    }

    /// e524 — an ordinary `LX -> HBM` store on the L3SU: both ends and all four addresses land on the
    /// register file the transfer's direction fixes.
    #[test]
    fn e524_an_l3_transfer_pins_both_ends_and_all_four_addresses() {
        let mut pass = TypesAndCopies::new();
        let mut values = Values::default();
        let mut body = vec![
            get_unit(Val(0), DfirUnit::Lx),
            get_unit(Val(1), DfirUnit::Hbm),
            Op::Sentient(sentient::Op::LoadAndStore {
                src: Val(0),
                dst: Val(1),
                src_mutable_addr: Val(2),
                src_immutable_addr: Val(3),
                src_inc: Val(4),
                dst_mutable_addr: Val(5),
                dst_immutable_addr: Val(6),
                dst_inc: Val(7),
                multicast_info: None,
                results: (Val(8), Val(9)),
                extent: Extent::of(Elements(8), Bits(16)),
                stride: 1,
                rotate_val: None,
                shuffle_mode: ShuffleMode::NoShuffle,
                src_reg: Reg {
                    locale: RegType::Unknown,
                    index: None,
                },
                dst_reg: Reg {
                    locale: RegType::Unknown,
                    index: None,
                },
                dir: None,
                is_ibr_write: false,
                dbg_name: None,
            }),
        ];

        pass.initialize_assignment_for_an_operation::<Dd2>(
            &mut body,
            &OpAt::top(utils::InBlock(2)),
            OpScope::ProgramUnit(DfirUnit::L3su),
            &mut values,
        );

        // The LX end is local, the HBM end is external.
        assert_eq!(pass.locale(Val(8)), Locale::Recorded(RegType::Lar));
        assert_eq!(pass.locale(Val(9)), Locale::Recorded(RegType::Ear));
        assert_eq!(pass.locale(Val(2)), Locale::Recorded(RegType::Lar));
        assert_eq!(pass.locale(Val(5)), Locale::Recorded(RegType::Ear));
        // ⭐ THE IMMUTABLE HALVES DIFFER: `lbr` beside a `lar` before SEN1P5, `ebr` beside an `ear`.
        assert_eq!(pass.locale(Val(3)), Locale::Recorded(RegType::Lbr));
        assert_eq!(pass.locale(Val(6)), Locale::Recorded(RegType::Ebr));
        // Nothing was copied — an unassigned address needs no register of its own — and nothing was
        // refused.
        assert_eq!(body.len(), 3);
        assert!(pass.failures.is_empty());
    }
    /// 574/656 — the module's constant locale is seeded first, then the whole unit is walked: the
    /// top-level constant, the loop's induction variable, and the constant inside the loop's region.
    #[test]
    fn e574_seeds_from_the_module_then_walks_the_whole_unit() {
        let mut pass = TypesAndCopies::new();
        let mut values = Values::default();
        pass.per_module
            .global_const_assignments
            .insert(Val(20), RegType::Imm);
        let mut body = vec![
            constant(Val(0), 8),
            for_op(
                Val(1),
                Val(0),
                carried(Val(2), Val(3), Val(4)),
                vec![constant(Val(5), 9)],
            ),
        ];

        pass.initialize_work_list::<Dd2>(DfirUnit::Lxlu, &mut body, &mut values);

        assert_eq!(pass.locale(Val(20)), Locale::Recorded(RegType::Imm));
        assert_eq!(pass.locale(Val(0)), Locale::Recorded(RegType::Imm));
        assert_eq!(pass.locale(Val(1)), Locale::Recorded(RegType::Lccr));
        assert_eq!(pass.locale(Val(5)), Locale::Recorded(RegType::Imm));
    }

    /// The `regLocale` a preamble `dataflow.get_unit` carries after one L3 unit ran, or [`None`].
    fn get_unit_locale(op: &Op) -> Option<RegType> {
        match op {
            Op::Dataflow(dataflow::Op::GetUnit { reg_locale, .. }) => *reg_locale,
            _ => None,
        }
    }

    /// 609/656 — the L3 tail writes the yielded handle's locale onto its `get_unit`, and skips both a
    /// handle nothing in the unit yields and one the propagation assigned nothing.
    #[test]
    fn e609_writes_reg_locale_only_on_a_yielded_and_assigned_unit_handle() {
        let mut pass = TypesAndCopies::new();
        let mut values = Values::default();
        let mut preamble = vec![
            get_unit(Val(0), DfirUnit::L3lu),
            get_unit(Val(1), DfirUnit::L3su),
            get_unit(Val(2), DfirUnit::Lx),
        ];
        for (handle, locale) in [(Val(0), RegType::Lar), (Val(1), RegType::Lbr)] {
            pass.per_module
                .global_const_assignments
                .insert(handle, locale);
        }
        let mut unit = ProgramUnit::<Dd2> {
            on: Units::one(DfirUnit::L3lu, Val(0)),
            precision: None,
            body: vec![Op::Sentient(sentient::Op::Yield {
                results: vec![Val(0), Val(2)],
            })],
            arch: core::marker::PhantomData,
        };

        pass.run_on::<Dd2>(&mut preamble, &mut unit, &mut values);

        assert_eq!(get_unit_locale(&preamble[0]), Some(RegType::Lar));
        // ⭐ ASSIGNED BUT NOT YIELDED HERE, and unrecorded though yielded — both are `continue`s.
        assert_eq!(get_unit_locale(&preamble[1]), None);
        assert_eq!(get_unit_locale(&preamble[2]), None);
        assert!(pass.failures.is_empty());
    }

    /// A model and rung, for the same reason [`ProgramUnit`] needs an arch: the pass reads neither.
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

    /// e633 — the module's constant, symbol and unit handle each get `imm` and the `scalar_copy`
    /// between them gets nothing; the per-unit run then starts from all three.
    #[test]
    fn e633_records_the_three_module_op_kinds_and_seeds_each_unit_with_them() {
        let mut program: Program<Dd2, AnyModel, AnyRung> = Program {
            name: ProgramName {
                group: GroupId(0),
                index: OpIndex(0),
                func: OpFunc::Add,
            },
            preamble: vec![
                Op::Dataflow(dataflow::Op::GetUnit {
                    result: Val(1),
                    residency: Residency::Global,
                    unit: DfirUnit::Lxlu,
                    num_folds: None,
                    reg_locale: None,
                }),
                copy(Val(1), Val(2)),
                Op::Symbol(symbol::Op::CreateSymbol {
                    result: Val(3),
                    symbol_id: 0,
                    max_value: None,
                }),
                constant(Val(4), 7),
            ],
            units: ProgramUnits::of(
                ProgramUnit {
                    on: Units::one(DfirUnit::Lxlu, Val(1)),
                    precision: None,
                    body: Vec::new(),
                    arch: core::marker::PhantomData,
                },
                Vec::new(),
            ),
            bound: core::marker::PhantomData,
        };
        let mut pass = TypesAndCopies::new();
        let mut values = Values::default();

        pass.run_on_operation(&mut program, &mut values);

        let recorded = BTreeMap::from([
            (Val(1), RegType::Imm),
            (Val(3), RegType::Imm),
            (Val(4), RegType::Imm),
        ]);
        assert_eq!(pass.per_module.global_const_assignments, recorded);
        // `initializeWorkList` re-inserted the module's three into the unit it then ran.
        assert_eq!(pass.per_unit.assignments, recorded);
        assert!(pass.failures.is_empty());
    }
}
