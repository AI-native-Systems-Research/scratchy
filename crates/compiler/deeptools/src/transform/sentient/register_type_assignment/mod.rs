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
use crate::islands::dataflow_ir::ty::GenericComp;
use crate::islands::sentient::dialects::{
    self as dialects, Definitions, Op, UniformRegions, Val, dataflow, sentient, symbol, uniform,
};
use crate::islands::sentient::print;
use crate::transform::sentient::utils::{self, ConstKind, Hoisted, NewUse, OpAt};
use crate::units::Core;

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
    /// `unit_aliases_`, keyed by `dcc::getCoreId(get_unit_op)` then `dcc::getUnitType(get_unit_op)`
    /// (`:276`, `:317`).
    ///
    /// ⛔ `None` IS THE REFERENCE'S `-1`: a `dataflow.get_unit` the whole device shares carries no
    /// `core` attribute at all ([`crate::units::Residency::core`]).
    ///
    /// ⛔ THE INNER KEY IS THE **GENERIC** COMPONENT, NOT THE UNIT. `getUnitType` is
    /// `senCompToGenericComp.at(op.getType())` (`Utils/DccExtContext.cpp:126-131`), which this crate
    /// already spells [`DfirUnit::generic`] — so two `ptrow*` handles on one core share ONE alias.
    pub(crate) unit_aliases: BTreeMap<Option<Core>, BTreeMap<GenericComp, Val>>,
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

/// WHICH OF THE FIVE OPS A COPY IS BEING MADE OF, and therefore which alias cache records it —
/// `DT_CHECK(op && isa<ConstantOp, GetUnitOp, CreateMulticastGroupOp, QueryMapOp, CreateSymbolOp>(op))`
/// (`:259-261`) as a type.
///
/// ⛔ THE FIVE `dyn_cast`s, NOT THE FOUR `is_*` PREDICATES. The predicates see THROUGH a
/// `uniform.query_map` to what its mapping holds (`Analyses/Utils.cpp:141-153, 221-255`) and decide
/// whether commoning happens at all; the cast decides which cache the copy lands in, and a query map
/// gets its own.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CopySource {
    /// `sentient::ConstantOp`, cached by `getValue()`.
    Constant(i64),
    /// `symbol::CreateSymbolOp`, cached by `getSymbolID()`.
    Symbol(i64),
    /// `dataflow::GetUnitOp`, cached by `dcc::getCoreId` then `dcc::getUnitType`.
    GetUnit(Option<Core>, GenericComp),
    /// `uniform::QueryMapOp`, cached by the query map's own result.
    QueryMap,
    /// `dataflow::CreateMulticastGroupOp` — ⛔ THE ONE SOURCE WITH NO CACHE AND NO COMMONING, and the
    /// one whose copy lands AFTER the definition instead of before the use.
    Multicast,
}

impl CopySource {
    /// The witness, or `None` for a value the reference's `DT_CHECK` would refuse.
    #[must_use]
    pub(crate) fn of(val: Val, defs: Definitions<'_>) -> Option<CopySource> {
        match defs.of(val)? {
            Op::Sentient(sentient::Op::ScalarConstant { value, .. }) => {
                Some(CopySource::Constant(*value))
            }
            Op::Symbol(symbol::Op::CreateSymbol { symbol_id, .. }) => {
                Some(CopySource::Symbol(*symbol_id))
            }
            Op::Dataflow(dataflow::Op::GetUnit {
                residency, unit, ..
            }) => Some(CopySource::GetUnit(residency.core(), unit.generic())),
            Op::Uniform(uniform::Op::QueryMap { .. }) => Some(CopySource::QueryMap),
            Op::Dataflow(dataflow::Op::CreateMulticastGroup { .. }) => Some(CopySource::Multicast),
            _ => None,
        }
    }
}

/// ONE `createCopyOperationAndUpdateAssignment` CALL — its five parameters, plus the [`CopySource`]
/// that stands for its `DT_CHECK`.
#[derive(Debug, Clone)]
pub(crate) struct CopyRequest {
    /// `val` — what is being copied.
    pub(crate) val: Val,
    /// Which of the five ops binds it.
    pub(crate) source: CopySource,
    /// `locale` — the register file the use demands.
    pub(crate) locale: RegisterLocale,
    /// `user` — the op that reads the copy, which is where it is inserted and what it must dominate.
    pub(crate) user: OpAt,
    /// `is_mutable_addr_or_xrf_ptr` — ⛔ THE REAL COMMONING SWITCH; see the anchor.
    pub(crate) is_mutable_addr_or_xrf_ptr: bool,
    /// `element_size`, half of the cache key.
    pub(crate) element_size: Option<Bits>,
}

impl<const ADD_SCALAR_COPIES: bool> RegisterTypeAssignment<ADD_SCALAR_COPIES> {
    /// Replaces: e350_createCopyOperationAndUpdateAssignment
    ///
    /// An existing `scalar_copy` of the same constant, symbol, unit handle or query map when one can be
    /// hoisted to dominate the user; otherwise a fresh one, cached and recorded as `locale`.
    ///
    /// ⛔ `dcc-register-type-assignment-const-commoning` IS DROPPED — a flag of `dcc-opt`, not a
    /// program property, and `cl::init(true)` (`:57-60`) leaves commoning always on.
    /// ⛔ TRAP: A FAILED HOIST FALLS THROUGH TO THE CREATE, and only the arm the source picks is
    /// tried — a cached alias under any other key never blocks a second copy.
    /// ⛔ TRAP: THE CACHE WRITE IS UNCONDITIONAL (`:371-375`) even where commoning was off, and the
    /// `assignments_` write is DIRECT (`:376`), not e143.
    pub(crate) fn create_copy_operation_and_update_assignment(
        &mut self,
        request: &CopyRequest,
        unit_body: &mut Vec<Op>,
        values: &mut Values,
    ) -> Val {
        let key = AliasKey {
            locale: request.locale,
            element_size: request.element_size,
        };
        // ⭐ EVERY READ OF THE IR HAPPENS HERE, before the body is borrowed mutably to insert.
        let (is_multicast, alias, source_at) = {
            let regions: [&[Op]; 1] = [unit_body.as_slice()];
            let defs = Definitions::from_innermost(&regions);
            let commoning = (utils::is_constant(request.val, ConstKind::ScalarConstant, defs)
                || is_symbol(request.val, defs)
                || is_get_unit(request.val, defs))
                && !request.is_mutable_addr_or_xrf_ptr;
            let alias = if commoning {
                match request.source {
                    CopySource::Constant(value) => self
                        .per_unit
                        .cst_aliases
                        .get(&value)
                        .and_then(|by_key| by_key.get(&key)),
                    CopySource::Symbol(id) => self
                        .per_module
                        .sym_aliases
                        .get(&id)
                        .and_then(|by_key| by_key.get(&key)),
                    CopySource::GetUnit(core, comp) => self
                        .per_unit
                        .unit_aliases
                        .get(&core)
                        .and_then(|by_comp| by_comp.get(&comp)),
                    // ⭐ THE EXACT KEY FIRST, THEN THE SCAN (`:344-368`): unrolling duplicates an
                    // immutable mapping, so two query maps holding the same constants are different
                    // values and both would otherwise get their own copy.
                    CopySource::QueryMap => {
                        self.per_unit
                            .query_map_aliases
                            .get(&request.val)
                            .and_then(|by_key| by_key.get(&key))
                            .or_else(|| {
                                self.per_unit.query_map_aliases.iter().find_map(
                                    |(other, by_key)| {
                                        let copy = by_key.get(&key)?;
                                        are_constant_query_maps_identical(*other, request.val, defs)
                                            .then_some(copy)
                                    },
                                )
                            })
                    }
                    CopySource::Multicast => None,
                }
                .copied()
            } else {
                None
            };
            (
                is_multicast(request.val, defs),
                alias,
                utils::path_of(unit_body, request.val),
            )
        };

        // `if (succeeded(moveToCommonDominator(alias.getDefiningOp(), user))) return alias;`
        if let Some(alias) = alias
            && let Some(at) = utils::path_of(unit_body, alias)
            && let Some(copy) = CopyAt::of(at, unit_body)
            && move_to_common_dominator(unit_body, copy, &NewUse::SameUnit(request.user.clone()))
                == Hoisted::Done
        {
            return alias;
        }

        // ⛔ `op->getResult(0)` IS `val` ITSELF for all five sources: each binds exactly one result in
        // this island, and a query map's is the value that was asked for.
        let result = values.mint();
        let copy = Op::Sentient(sentient::Op::ScalarCopy {
            input: request.val,
            result,
            reg: sentient::Reg {
                locale: request.locale,
                index: None,
            },
            // `CopyOp::create(builder, loc, type, input, regLocale)` sets neither.
            element_size: None,
            program_header: false,
        });
        let at = match source_at {
            Some(source_at) if is_multicast => source_at.after(),
            Some(_) | None => request.user.clone(),
        };
        utils::insert_at(unit_body, &at, copy);

        match request.source {
            CopySource::Constant(value) => {
                self.per_unit
                    .cst_aliases
                    .entry(value)
                    .or_default()
                    .insert(key, result);
            }
            CopySource::Symbol(id) => {
                self.per_module
                    .sym_aliases
                    .entry(id)
                    .or_default()
                    .insert(key, result);
            }
            CopySource::GetUnit(core, comp) => {
                self.per_unit
                    .unit_aliases
                    .entry(core)
                    .or_default()
                    .insert(comp, result);
            }
            CopySource::QueryMap => {
                self.per_unit
                    .query_map_aliases
                    .entry(request.val)
                    .or_default()
                    .insert(key, result);
            }
            CopySource::Multicast => {}
        }
        self.per_unit.assignments.insert(result, request.locale);
        result
    }

    /// Replaces: e351_addToWorkListAndUpdateAssignment
    ///
    /// Queues `val` and records `locale` for it.
    ///
    /// ⛔ TRAP: THE ORDER IS THE WHOLE FUNCTION. e141 skips a value that already has an assignment, so
    /// recording first would drop the value from the worklist.
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
    /// Writes every locale this pass assigned back into the unit's ops, pre-order: a loop's induction
    /// variable and iter args, then the results of the fifteen ops that carry a register file.
    ///
    /// ⛔ THE `get_unit` ARM IS GATED ON THE UNIT (`:472-475`) — only an L3LU or L3SU program unit
    /// assigns its own unit handles, and `element_sizes` is untouched.
    /// ⛔ TRAP: AN UNRECORDED VALUE IS WRITTEN AS `unknown`, NOT SKIPPED — `getLocaleAsString` answers
    /// for every value ([`Locale::Unrecorded`]), so this OVERWRITES a locale an earlier pass set.
    /// ⛔ TRAP: THE ARGUMENT AND THE RESULT OF ONE CARRIED VALUE SHARE ONE SLOT HERE where the
    /// reference has two of a `1 + 2n` array ([`sentient::Carried::reg`]), so the result's locale
    /// stands; they agree in a loop whose carried register is one register.
    pub(crate) fn update_program_unit(&self, unit_body: &mut Vec<Op>, unit_type: GenericComp) {
        // ⭐ DECIDE, THEN WRITE: the walk reads the ops that the writes then mutate.
        let mut assigned: Vec<(Val, RegisterLocale)> = Vec::new();
        self.collect_locales(unit_body, unit_type, &mut assigned);
        for (val, locale) in assigned {
            dialects::set_value_reg_locale(unit_body, val, locale);
        }
    }

    /// The pre-order half of [`Self::update_program_unit`] — `(value, locale)` in the order the
    /// reference fills its `reg_locales_vector`.
    fn collect_locales(
        &self,
        block: &[Op],
        unit_type: GenericComp,
        out: &mut Vec<(Val, RegisterLocale)>,
    ) {
        for op in block {
            if let Op::Sentient(sentient::Op::For { iv, carried, .. }) = op {
                out.push((*iv, self.locale(*iv).reg_type()));
                out.extend(
                    carried
                        .iter()
                        .map(|value| (value.arg, self.locale(value.arg).reg_type())),
                );
            }
            if assigns_result_locales(op, unit_type) {
                out.extend(
                    dialects::results(op)
                        .into_iter()
                        .map(|result| (result, self.locale(result).reg_type())),
                );
            }
            for region in dialects::regions_ref(op) {
                self.collect_locales(region, unit_type, out);
            }
        }
    }
}

/// The `isa<>` list whose results `e352_updateProgramUnit` assigns (`:462-475`).
///
/// ⛔ `sentient.mac` IS IN IT AND HAS NO REGISTER FIELD — see [`dialects::set_value_reg_locale`].
fn assigns_result_locales(op: &Op, unit_type: GenericComp) -> bool {
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
        | Op::UniformRegions(UniformRegions::UniformizeRegions { .. }) => true,
        Op::Dataflow(dataflow::Op::GetUnit { .. }) => {
            matches!(unit_type, GenericComp::L3lu | GenericComp::L3su)
        }
        _ => false,
    }
}

/// `dcc::utils::isSymbol` (`Transform/Sentient/Analyses/Utils.cpp:141-153`) — a `symbol.create_symbol`,
/// or a `uniform.query_map` ANY of whose mapped values is one.
///
/// ⛔ `any`, NOT `all`: [`utils::is_constant`] with [`ConstKind::CreateSymbol`] is the OTHER predicate
/// (`isConstant<CreateSymbolOp>`, `Utils/Utils.cpp:424`) and demands every value.
fn is_symbol(val: Val, defs: Definitions<'_>) -> bool {
    match defs.of(val) {
        Some(Op::Symbol(symbol::Op::CreateSymbol { .. })) => true,
        Some(Op::Uniform(uniform::Op::QueryMap { map, key, .. })) => {
            dialects::uniform_mapping_values(*map, *key, defs)
                .into_iter()
                .any(|value| {
                    matches!(
                        defs.of(value),
                        Some(Op::Symbol(symbol::Op::CreateSymbol { .. }))
                    )
                })
        }
        Some(_) | None => false,
    }
}

/// `dcc::utils::isMulticast` (`:221-238`) — a `dataflow.create_multicast_group`, or a
/// `uniform.query_map` all of whose mapped values are ones.
///
/// ⚠️ DIVERGENCE: AN EMPTY MAPPING ANSWERS `true` HERE AND THROWS THERE (`DT_CHECK(values.size() > 0)`),
/// the same one [`utils::is_constant`] records.
fn is_multicast(val: Val, defs: Definitions<'_>) -> bool {
    match defs.of(val) {
        Some(Op::Dataflow(dataflow::Op::CreateMulticastGroup { .. })) => true,
        Some(Op::Uniform(uniform::Op::QueryMap { map, key, .. })) => {
            dialects::uniform_mapping_values(*map, *key, defs)
                .into_iter()
                .all(|value| {
                    matches!(
                        defs.of(value),
                        Some(Op::Dataflow(dataflow::Op::CreateMulticastGroup { .. }))
                    )
                })
        }
        Some(_) | None => false,
    }
}

/// `dcc::utils::isGetUnit` (`:240-255`) — see [`is_multicast`], which it is the twin of.
fn is_get_unit(val: Val, defs: Definitions<'_>) -> bool {
    match defs.of(val) {
        Some(Op::Dataflow(dataflow::Op::GetUnit { .. })) => true,
        Some(Op::Uniform(uniform::Op::QueryMap { map, key, .. })) => {
            dialects::uniform_mapping_values(*map, *key, defs)
                .into_iter()
                .all(|value| {
                    matches!(
                        defs.of(value),
                        Some(Op::Dataflow(dataflow::Op::GetUnit { .. }))
                    )
                })
        }
        Some(_) | None => false,
    }
}

/// `dcc::uniform::utils::areConstantQueryMapsIdentical` (`Dialect/Uniform/Utils.cpp:234-256`) — two
/// query maps whose per-key constants differ by zero everywhere.
///
/// ⛔ NOT AN ANCHORED UNIT — the scan half of e350's query-map arm, and `getDeltasBetweenValuesOfUniform`
/// `Mappings` (`:204-231`) inlined: every delta must exist and be zero, so a non-constant value or a
/// key either map lacks answers `false` rather than producing a delta.
fn are_constant_query_maps_identical(a: Val, b: Val, defs: Definitions<'_>) -> bool {
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
    if a == b {
        return true;
    }
    let keys_a = dialects::uniform_mapping_keys(*key_a, defs);
    // `if (key_vals_a.size() != key_vals_b.size()) return nullopt;` — then A's list indexes BOTH.
    if keys_a.len() != dialects::uniform_mapping_keys(*key_b, defs).len() {
        return false;
    }
    let (
        Some(Op::Uniform(uniform::Op::DefImmutableMapping { pairs: pairs_a, .. })),
        Some(Op::Uniform(uniform::Op::DefImmutableMapping { pairs: pairs_b, .. })),
    ) = (defs.of(*map_a), defs.of(*map_b))
    else {
        return false;
    };
    keys_a.iter().all(|sought| {
        let value_of = |pairs: &[(Val, Val)]| {
            let (_, value) = pairs.iter().find(|(mapped, _)| mapped == sought)?;
            match defs.of(*value)? {
                Op::Sentient(sentient::Op::ScalarConstant { value, .. }) => Some(*value),
                _ => None,
            }
        };
        match (value_of(pairs_a), value_of(pairs_b)) {
            (Some(from), Some(to)) => to == from,
            _ => false,
        }
    })
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
    use crate::transform::sentient::utils::InBlock;

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
        pass.per_unit.unit_aliases.insert(
            Core::checked(0),
            BTreeMap::from([(GenericComp::Sfp, Val(5))]),
        );
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

    /// A `sentient.for` carrying one value.
    fn for_op(iv: Val, bound: Val, carried: Vec<sentient::Carried>, body: Vec<Op>) -> Op {
        Op::Sentient(sentient::Op::For {
            iv,
            bound,
            carried,
            iv_reg: Reg::UNALLOCATED,
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
            reg: Reg::UNALLOCATED,
            result_reg: Reg::UNALLOCATED,
            program_header: false,
            element_size: None,
        }
    }

    /// The first request for a constant in a locale builds the copy before its user and caches it; the
    /// second reuses that copy, hoisted to dominate the new user rather than duplicated.
    #[test]
    fn a_second_use_of_one_constant_in_one_locale_reuses_the_first_copy() {
        let mut pass = TypesAndCopies::new();
        let mut values = Values::default();
        for _ in 0..3 {
            let _ = values.mint();
        }
        let mut body = vec![constant(Val(1), 7), copy(Val(1), Val(2))];
        let request = CopyRequest {
            val: Val(1),
            source: CopySource::Constant(7),
            locale: RegType::Lrf,
            user: OpAt::top(InBlock(1)),
            is_mutable_addr_or_xrf_ptr: false,
            element_size: None,
        };
        let first =
            pass.create_copy_operation_and_update_assignment(&request, &mut body, &mut values);
        assert_eq!(body.len(), 3);
        assert!(
            matches!(&body[1], Op::Sentient(sentient::Op::ScalarCopy { input, result, reg, .. })
            if *input == Val(1) && *result == first && reg.locale == RegType::Lrf)
        );
        let key = AliasKey {
            locale: RegType::Lrf,
            element_size: None,
        };
        assert_eq!(pass.per_unit.cst_aliases[&7][&key], first);
        assert_eq!(pass.locale(first), Locale::Recorded(RegType::Lrf));

        // ⭐ THE SECOND REQUEST ADDS NO OP: the cached copy already dominates the later user.
        let later = CopyRequest {
            user: OpAt::top(InBlock(2)),
            ..request.clone()
        };
        assert_eq!(
            pass.create_copy_operation_and_update_assignment(&later, &mut body, &mut values),
            first
        );
        assert_eq!(body.len(), 3);
        // ⛔ A DIFFERENT LOCALE IS A DIFFERENT KEY, so it gets its own copy.
        let elsewhere = CopyRequest {
            locale: RegType::Lbr,
            ..request.clone()
        };
        let second =
            pass.create_copy_operation_and_update_assignment(&elsewhere, &mut body, &mut values);
        assert_ne!(second, first);
        assert_eq!(body.len(), 4);
    }

    /// Queueing happens before recording, so the value is on the worklist AND assigned; a repeat leaves
    /// both alone.
    #[test]
    fn the_value_is_queued_before_it_is_assigned() {
        let mut pass = TypesAndCopies::new();
        pass.add_to_work_list_and_update_assignment(Val(1), RegType::Lrf);
        assert_eq!(pass.worklist, VecDeque::from([Val(1)]));
        assert_eq!(pass.locale(Val(1)), Locale::Recorded(RegType::Lrf));
        // e141 declines an already-assigned value and e143 keeps the first locale.
        pass.add_to_work_list_and_update_assignment(Val(1), RegType::Lbr);
        assert_eq!(pass.worklist, VecDeque::from([Val(1)]));
        assert_eq!(pass.locale(Val(1)), Locale::Recorded(RegType::Lrf));
    }

    /// The loop's induction variable takes slot 0, its carried value takes slot 1, and an op in the
    /// `isa<>` list has its own result's locale written — an unrecorded value included, as `unknown`.
    #[test]
    fn the_walk_writes_slot_zero_the_iter_args_and_then_every_listed_results_locale() {
        let mut pass = TypesAndCopies::new();
        let (iv, arg, carried_result) = (Val(10), Val(11), Val(12));
        pass.update_assignment(iv, RegType::Lccr);
        pass.update_assignment(arg, RegType::Lrf);
        pass.update_assignment(Val(2), RegType::Lbr);
        let mut body = vec![
            constant(Val(1), 7),
            for_op(
                iv,
                Val(1),
                vec![carried(Val(1), arg, carried_result)],
                vec![copy(arg, Val(2))],
            ),
        ];
        pass.update_program_unit(&mut body, GenericComp::Sfp);
        let Op::Sentient(sentient::Op::For {
            iv_reg,
            carried,
            body: inner,
            ..
        }) = &body[1]
        else {
            unreachable!("the loop is still a loop")
        };
        assert_eq!(iv_reg.locale, RegType::Lccr);
        // ⛔⛔ TWO SLOTS, TWO WRITES, AND THE SECOND MUST NOT LAND ON THE FIRST. `[i + 1]` is the
        // iter arg's, written from this walk's own `getRegionIterArgs()` list, and `[i + n + 1]` is
        // the result's, written by its generic half (`:444-457`). `%11` is recorded `lrf` and `%12`
        // is not recorded at all, so a single collapsed slot would read `unknown` — the arg's answer
        // silently overwritten by its result's.
        assert_eq!(carried[0].reg.locale, RegType::Lrf);
        assert_eq!(carried[0].result_reg.locale, RegType::Unknown);
        assert!(
            matches!(&inner[0], Op::Sentient(sentient::Op::ScalarCopy { reg, .. })
            if reg.locale == RegType::Lbr)
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
