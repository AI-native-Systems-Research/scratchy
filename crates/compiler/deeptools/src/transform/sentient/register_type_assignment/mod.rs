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

use crate::formats::Bits;
use crate::islands::dataflow_ir::Values;
use crate::islands::sentient::dialects::{
    self as dialects, Op, Val, dataflow, sentient, symbol, uniform,
};
use crate::islands::sentient::print;
use crate::transform::sentient::utils::{self, Hoisted, NewUse, OpAt};
use crate::units::{Core, DfirUnit};

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
/// ⚠️ `dcc_ext_ctx_` IS NOT CARRIED YET. `dccExtContext()` is read by e524 alone (`:692-954`, for
/// `getArch()`, `getProgPatch()` and the `memoryOpRequiresImmutAddrScalarCopy` it forwards to), and
/// `opts_` by nothing in the file; both enter with e524.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct RegisterTypeAssignment<const ADD_SCALAR_COPIES: bool> {
    /// The maps `clear()` resets at each unit.
    pub(crate) per_unit: PerUnit,
    /// The maps that survive it.
    pub(crate) per_module: PerModule,
    /// `worklist_` — ⛔ NOT IN [`PerUnit`]: `clear()` does not name it (`:222-227`), and e523 drains
    /// it to empty instead.
    pub(crate) worklist: VecDeque<Val>,
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
fn is_symbol(val: Val, defs: dialects::Definitions<'_>) -> bool {
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

// crustify:todo: e463_addToWorkListCreateCopyAndUpdateAssignment
//   authority : dcc/src/Transform/Sentient/RegisterTypeAssignment.cpp:405  (17 body lines, level 2)
//   original  : void RegisterTypeAssignmentPass::addToWorkListCreateCopyAndUpdateAssignment( MutableOperandRange operand, const RegisterLocales locale, Operation* user, int element_size, const bool is_mutable_addr_or_xrf_ptr)
//   calls     : e141_addToWorkList, e142_isCopyNeeded, e143_updateAssignment, e252_size, e350_createCopyOperationAndUpdateAssignment

// crustify:todo: e523_processWorkList
//   authority : dcc/src/Transform/Sentient/RegisterTypeAssignment.cpp:492  (93 body lines, level 3)
//   original  : void RegisterTypeAssignmentPass::processWorkList()
//   calls     : e139_getLocale, e143_updateAssignment, e351_addToWorkListAndUpdateAssignment, e463_addToWorkListCreateCopyAndUpdateAssignment

// crustify:todo: e524_initializeAssignmentForAnOperation
//   authority : dcc/src/Transform/Sentient/RegisterTypeAssignment.cpp:594  (383 body lines, level 3)
//   original  : void RegisterTypeAssignmentPass::initializeAssignmentForAnOperation( Operation* op)
//   calls     : e139_getLocale, e143_updateAssignment, e247_memoryOpRequiresImmutAddrScalarCopy, e252_size, e278_isValid, e351_addToWorkListAndUpdateAssignment, e463_addToWorkListCreateCopyAndUpdateAssignment

// crustify:todo: e574_initializeWorkList
//   authority : dcc/src/Transform/Sentient/RegisterTypeAssignment.cpp:987  (8 body lines, level 4)
//   original  : void RegisterTypeAssignmentPass::initializeWorkList()
//   calls     : e138_clear, e422_insert, e524_initializeAssignmentForAnOperation

// crustify:todo: e609_runOn
//   authority : dcc/src/Transform/Sentient/RegisterTypeAssignment.cpp:996  (55 body lines, level 5)
//   original  : void RegisterTypeAssignmentPass::runOn(dataflow::ProgramUnitOp unit)
//   calls     : e352_updateProgramUnit, e523_processWorkList, e574_initializeWorkList

// crustify:todo: e633_runOnOperation
//   authority : dcc/src/Transform/Sentient/RegisterTypeAssignment.cpp:113  (17 body lines, level 6)
//   original  : void runOnOperation() override
//   calls     : e524_initializeAssignmentForAnOperation, e609_runOn

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::islands::dataflow_ir::ty::ScalarTy;
    use crate::islands::sentient::dialects::sentient::{Reg, RegType};

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
}
