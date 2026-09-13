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

//! `ScalarCopyInsertionForSymbols.cpp` — 14 of the campaign's 656 units (dependency level(s) [0, 1, 3, 4, 5]).
//!
//! | unit | entry | level | lines | authority path:line |
//! |---|---|---|---|---|
//! | `e149_collectOpsOfInterest` | 149 | 0 | 18 | `dcc/src/Transform/Sentient/ScalarCopyInsertionForSymbols.cpp:217` |
//! | `e150_pessimizeLiveness` | 150 | 0 | 20 | `dcc/src/Transform/Sentient/ScalarCopyInsertionForSymbols.cpp:276` |
//! | `e151_insertCopyOpsForJCRCandidate` | 151 | 0 | 23 | `dcc/src/Transform/Sentient/ScalarCopyInsertionForSymbols.cpp:416` |
//! | `e152_clear` | 152 | 0 | 7 | `dcc/src/Transform/Sentient/ScalarCopyInsertionForSymbols.cpp:441` |
//! | `e153_getLocale` | 153 | 0 | 26 | `dcc/src/Transform/Sentient/ScalarCopyInsertionForSymbols.cpp:449` |
//! | `e154_propagateElementSizeToCopyOp` | 154 | 0 | 24 | `dcc/src/Transform/Sentient/ScalarCopyInsertionForSymbols.cpp:476` |
//! | `e155_dumpSymbolicLocales` | 155 | 0 | 10 | `dcc/src/Transform/Sentient/ScalarCopyInsertionForSymbols.cpp:502` |
//! | `e156_dumpCandidates` | 156 | 0 | 12 | `dcc/src/Transform/Sentient/ScalarCopyInsertionForSymbols.cpp:545` |
//! | `e358_collectCandidates` | 358 | 1 | 65 | `dcc/src/Transform/Sentient/ScalarCopyInsertionForSymbols.cpp:297` |
//! | `e359_insertCopyOpsForCandidates` | 359 | 1 | 49 | `dcc/src/Transform/Sentient/ScalarCopyInsertionForSymbols.cpp:365` |
//! | `e360_dumpSymbolUsageInfo` | 360 | 1 | 32 | `dcc/src/Transform/Sentient/ScalarCopyInsertionForSymbols.cpp:512` |
//! | `e527_collectSymbolUsage` | 527 | 3 | 38 | `dcc/src/Transform/Sentient/ScalarCopyInsertionForSymbols.cpp:236` |
//! | `e576_runOn` | 576 | 4 | 67 | `dcc/src/Transform/Sentient/ScalarCopyInsertionForSymbols.cpp:148` |
//! | `e610_runOnOperation` | 610 | 5 | 17 | `dcc/src/Transform/Sentient/ScalarCopyInsertionForSymbols.cpp:130` |

use std::fmt::Write as _;

use crate::arch::Arch;
use crate::formats::Bits;
use crate::islands::dataflow_ir::Values;
use crate::islands::sentient::dialects::sentient::{Reg, RegType};
use crate::islands::sentient::dialects::{self as dialects, Op, Val, sentient, symbol, uniform};
use crate::islands::sentient::print;
use crate::islands::sentient::Program;
use crate::model::Model;
use crate::workload::Workload;

use super::ForRef;
use super::analyses::{InstructionEstimator, Liveness, Metric, Pressure, RegisterPressure};
use super::utils::{Hoisted, InBlock, NewUse, OpAt, insert_at, move_to_common_dominator, path_of};

/// `localeToRegisterClass` (`dcc/src/Dialect/Sentient/Utils.cpp:203`) — the register class a locale
/// names, which is `stringifySentientRegType(locale).upper()` with both XRF pointers folded onto one.
///
/// ⭐ A STRING BECAUSE IT IS DUMP TEXT, not a set: the two dumps below are the only readers, and
/// [`super::local_region_splitting_for_value_commoning`] already carries the same function with the
/// strings dropped for the reader that wants a `RegFile`.
const fn locale_to_register_class(locale: RegType) -> &'static str {
    match locale {
        RegType::Unknown => "UNKNOWN",
        RegType::Imm => "IMM",
        RegType::Jcr => "JCR",
        RegType::Lccr => "LCCR",
        RegType::Lrf => "LRF",
        // ⭐ BOTH POINTERS NAME THE ONE CLASS — the function's only special case (`Utils.cpp:204-205`).
        RegType::XrfRdPtr | RegType::XrfWrPtr => "XRF",
        RegType::Lar => "LAR",
        RegType::Lbr => "LBR",
        RegType::Ear => "EAR",
        RegType::Ebr => "EBR",
        RegType::Gtr => "GTR",
        RegType::Mvr => "MVR",
        RegType::Unrelated => "UNRELATED",
    }
}

/// ONE `OpOperand *` — WHICH SLOT OF WHICH OP reads a symbol, which is what `symbol_to_usage_` holds.
///
/// ⭐ A POSITION, NOT A POINTER: this rung's ops are a tree with no identity of their own, so a use is
/// named by the op's path from the unit body and its index in [`dialects::operands`]. The path is the
/// one [`super::address_register_precision_assignment`]'s `insert_at` uses — ordinals outermost
/// first, each level indexing that op's regions concatenated in [`dialects::regions_mut`] order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UseSite {
    /// The ordinals, outermost first, the last naming the using op.
    pub path: Vec<u32>,
    /// The slot, numbered as [`dialects::operands`] answers — `OpOperand::getOperandNumber()`.
    pub operand: usize,
}

/// `symbol_to_usage_` — per symbol, per locale, the uses recorded in walk order.
///
/// ⛔ THE ORDER IS LOAD-BEARING: [`insert_copy_ops_for_jcr_candidate`] builds its one copy beside the
/// FIRST recorded use, so a set would change which op the copy is created next to.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SymbolUsage {
    /// One entry per symbol, in the order `collectSymbolUsage` first saw it.
    per_symbol: Vec<(Val, Vec<(RegType, Vec<UseSite>)>)>,
}

impl SymbolUsage {
    /// `symbol_to_usage_[sym][locale].push_back(&use)` — the one way an entry appears.
    pub fn record(&mut self, symbol: Val, locale: RegType, at: UseSite) {
        let per_locale = match self.per_symbol.iter_mut().find(|(sym, _)| *sym == symbol) {
            Some((_, per_locale)) => per_locale,
            None => {
                self.per_symbol.push((symbol, Vec::new()));
                &mut self.per_symbol.last_mut().expect("the entry just pushed").1
            }
        };
        match per_locale.iter_mut().find(|(loc, _)| *loc == locale) {
            Some((_, uses)) => uses.push(at),
            None => per_locale.push((locale, vec![at])),
        }
    }

    /// `symbol_to_usage_.at(symbol).at(locale)` — ⭐ EMPTY WHERE THE REFERENCE'S `at` WOULD THROW,
    /// which the pass keeps unreachable: only a pair the collector recorded becomes a candidate.
    #[must_use]
    pub fn uses(&self, symbol: Val, locale: RegType) -> &[UseSite] {
        self.per_symbol
            .iter()
            .find(|(sym, _)| *sym == symbol)
            .and_then(|(_, per_locale)| per_locale.iter().find(|(loc, _)| *loc == locale))
            .map_or(&[][..], |(_, uses)| uses.as_slice())
    }

    /// Whether nothing was recorded — `symbol_to_usage_.empty()`.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.per_symbol.is_empty()
    }
}

/// ONE `CandidateEntryTy` — `std::pair<Value, SentientRegType>`, a symbol and a locale it is used in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CandidateEntry {
    /// `candidate.first`.
    pub symbol: Val,
    /// `candidate.second`.
    pub locale: RegType,
}

/// THE TWO LOCALES A JCR CANDIDATE MAY CARRY — `DT_CHECK(locale == lccr || locale == jcr)` (`:420`)
/// as a type, so [`insert_copy_ops_for_jcr_candidate`] cannot be handed anything else.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JcrLocale {
    /// `SentientRegType::lccr`.
    Lccr,
    /// `SentientRegType::jcr`.
    Jcr,
}

impl JcrLocale {
    /// The `DT_CHECK` itself — `None` for every other locale.
    #[must_use]
    pub const fn of(locale: RegType) -> Option<JcrLocale> {
        match locale {
            RegType::Lccr => Some(JcrLocale::Lccr),
            RegType::Jcr => Some(JcrLocale::Jcr),
            _ => None,
        }
    }

    /// The locale it stands for.
    #[must_use]
    pub const fn locale(self) -> RegType {
        match self {
            JcrLocale::Lccr => RegType::Lccr,
            JcrLocale::Jcr => RegType::Jcr,
        }
    }
}

/// `scalar_copy_ops_`, `outermost_loops_` and `symbol_queries_` — what one walk of a unit collects.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct OpsOfInterest {
    /// `scalar_copy_ops_`, each named by the value its `sentient.scalar_copy` binds.
    pub scalar_copy_ops: Vec<Val>,
    /// `outermost_loops_`, each named by its induction variable.
    pub outermost_loops: Vec<ForRef>,
    /// `symbol_queries_` — `op->getResult(0)` of every `symbol.query_map`.
    pub symbol_queries: Vec<Val>,
    /// The subjects of `emitWarning("skipping misplaced create_symbol op ..")`. ⭐ A WARNING DOES NOT
    /// FAIL THE PASS and the op is simply not collected, so the list is what a caller can observe.
    pub misplaced_create_symbols: Vec<Val>,
}

/// THE PER-UNIT STATE `clear()` RESETS — ⛔ `symbols_` IS NOT ONE OF ITS FIVE FIELDS and is therefore
/// not here: the function-scope symbol list is collected once and reused for every unit.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PerUnitState {
    /// `scalar_copy_ops_`, `outermost_loops_` and `symbol_queries_`.
    pub ops: OpsOfInterest,
    /// `symbol_to_usage_`.
    pub symbol_to_usage: SymbolUsage,
    /// `symbolic_locales_` — ⭐ INSERTION-ORDERED, which is what a `SmallSet` of `kMaxNumLocales` is
    /// in its small mode, and `collectCandidates` (`e358`) iterates it to build the candidate list.
    pub symbolic_locales: Vec<RegType>,
}

/// Replaces: e149_collectOpsOfInterest
///
/// Walks one unit collecting its `sentient.scalar_copy`s, its outermost `sentient.for`s and its
/// `symbol.query_map` results, warning on every `symbol.create_symbol` found below function scope.
///
/// ⛔ TRAP: THE TWO `DT_CHECK(..empty())` ARE THE RETURN TYPE. A fresh [`OpsOfInterest`] per call
/// cannot append to the previous unit's, which is what those checks exist to catch.
#[must_use]
pub fn collect_ops_of_interest(unit_body: &[Op]) -> OpsOfInterest {
    let mut found = OpsOfInterest::default();
    walk_ops_of_interest(unit_body, false, &mut found);
    found
}

/// The `unit->walk` itself. `inside_loop` is `isOuterMostLoop`'s parent chain (`Analyses/Utils.cpp:523`)
/// carried down, because an op of this island has no `getParentOp()` to walk up.
fn walk_ops_of_interest(scope: &[Op], inside_loop: bool, found: &mut OpsOfInterest) {
    for op in scope {
        match op {
            Op::Sentient(sentient::Op::ScalarCopy { result, .. }) => {
                found.scalar_copy_ops.push(*result);
            }
            Op::Sentient(sentient::Op::For { iv, .. }) => {
                if !inside_loop {
                    found.outermost_loops.push(ForRef(*iv));
                }
            }
            Op::Symbol(symbol::Op::QueryMap { result, .. }) => found.symbol_queries.push(*result),
            Op::Symbol(symbol::Op::CreateSymbol { result, .. }) => {
                found.misplaced_create_symbols.push(*result);
            }
            _ => {}
        }
        let nested_in_loop = inside_loop || matches!(op, Op::Sentient(sentient::Op::For { .. }));
        for region in dialects::regions(op) {
            walk_ops_of_interest(&region, nested_in_loop, found);
        }
    }
}

/// Replaces: e150_pessimizeLiveness
///
/// Widens the live range of every symbolic `sentient.scalar_copy` result, and of every outermost
/// loop's iter arg whose initialiser is a `sentient.scalar_constant` or a symbol.
///
/// ⛔ TRAP: THE ITER ARG'S LOCALE DECIDES, NOT THE INITIALISER'S — the initialiser only has to satisfy
/// `isConstant<sentient::ConstantOp>` or `isSymbol` (`Analyses/Utils.cpp:141`), while the locale
/// tested against `symbolic_locales_` is read off the region argument.
pub fn pessimize_liveness<L: Liveness>(state: &PerUnitState, unit_body: &[Op], liveness: &mut L) {
    let enclosing = [unit_body];
    let defs = dialects::Definitions::from_innermost(&enclosing);
    for copy_result in &state.ops.scalar_copy_ops {
        if state
            .symbolic_locales
            .contains(&dialects::value_reg_locale(*copy_result, defs))
        {
            liveness.update_live_ranges_for_program_header_promotion(*copy_result);
        }
    }
    for loop_ref in &state.ops.outermost_loops {
        let Some(sentient::Op::For { carried, .. }) = for_op_with_iv(loop_ref.0, unit_body) else {
            continue;
        };
        for value in carried {
            if !is_constant(value.init, defs) && !is_symbol(value.init, defs) {
                continue;
            }
            if !state
                .symbolic_locales
                .contains(&dialects::value_reg_locale(value.arg, defs))
            {
                continue;
            }
            liveness.update_live_ranges_for_program_header_promotion(value.arg);
        }
    }
}

/// The `sentient.for` an induction variable names — the loop handle a [`ForRef`] stands in for.
fn for_op_with_iv(iv: Val, scope: &[Op]) -> Option<&sentient::Op> {
    for op in scope {
        if let Op::Sentient(inner @ sentient::Op::For { iv: loop_iv, .. }) = op
            && *loop_iv == iv
        {
            return Some(inner);
        }
        for region in dialects::regions_ref(op) {
            if let Some(found) = for_op_with_iv(iv, region) {
                return Some(found);
            }
        }
    }
    None
}

/// `dcc::utils::isConstant<sentient::ConstantOp>` — ⭐ THAT ONE OP, so an `arith.constant` is not one,
/// and a region argument has no defining op and is never one.
fn is_constant(val: Val, defs: dialects::Definitions<'_>) -> bool {
    matches!(
        defs.of(val),
        Some(Op::Sentient(sentient::Op::ScalarConstant { .. }))
    )
}

/// `dcc::utils::isSymbol` (`Analyses/Utils.cpp:141-154`) — a `symbol.create_symbol`, or a
/// `uniform.query_map` any of whose ANSWERABLE values is one.
///
/// ⭐ STRUCTURAL, SO NOT AN OUT-OF-SCOPE `todo!`: it reads defining ops and the values the query's own
/// key selects — `getListOfValueOpsFromUniformMapping` (`Dialect/Uniform/Utils.cpp:194-202`), which
/// the island already ports as [`dialects::uniform_mapping_values`].
/// ⛔ TRAP: THE KEY IS LOAD-BEARING, so scanning every pair of the `uniform.def_immutable_mapping`
/// over-reports. A concrete `dataflow.get_unit` key stands for ITSELF and selects one pair
/// (`getListOfKeyOpsFromUniformMapping`, `:187-190`); only a region-argument key stands for the
/// enclosing uniformize/equalize unit list (`:154-186`).
fn is_symbol(val: Val, defs: dialects::Definitions<'_>) -> bool {
    match defs.of(val) {
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
        Some(Op::Symbol(symbol::Op::CreateSymbol { .. })) => true,
        _ => false,
    }
}

/// Replaces: e151_insertCopyOpsForJCRCandidate
///
/// Builds ONE `jcr` `sentient.scalar_copy` of a symbol and points every recorded use of it at the
/// copy — at the top of the unit, or just after the `symbol.query_map` that defines the symbol.
///
/// ⛔ TRAP: ONE COPY FOR ALL THE USES (`if (!new_copy_op)`), created `jcr` whichever of the two
/// locales the candidate carries, because `RegisterInitialization` does not promote JCR inits.
/// ⛔ TRAP: THE USES ARE REWIRED BEFORE THE COPY IS INSERTED — inserting first would shift every
/// recorded path in that block. The observable result is the reference's either way.
#[must_use]
pub fn insert_copy_ops_for_jcr_candidate(
    unit_body: &mut Vec<Op>,
    symbol: Val,
    locale: JcrLocale,
    usage: &SymbolUsage,
    values: &mut Values,
) -> Option<Val> {
    let uses = usage.uses(symbol, locale.locale());
    if uses.is_empty() {
        return None;
    }
    let copy_result = values.mint();
    for at in uses {
        if let Some(user) = op_at_mut(unit_body, &at.path) {
            dialects::set_operand(user, at.operand, copy_result);
        }
    }
    let copy = Op::Sentient(sentient::Op::ScalarCopy {
        input: symbol,
        result: copy_result,
        // `CopyOp::create(builder, loc, type, symbol, jcr)` — the five-argument builder, so `regIndex`
        // and `programHeader` keep the `.td`'s defaults (`SentientOps.td:830-846`).
        reg: Reg {
            locale: RegType::Jcr,
            index: None,
        },
        program_header: false,
        element_size: None,
    });
    // `isa<symbol::SymbolQueryMapOp>(symbol.getDefiningOp())` — *"hopefully symbol_query ops are
    // positioned optimally"*; anything else goes to the top of the unit region.
    let after_query = matches!(
        dialects::defining_op(symbol, unit_body),
        Some(Op::Symbol(symbol::Op::QueryMap { .. }))
    );
    if !(after_query && insert_after_definition(unit_body, symbol, copy.clone())) {
        unit_body.insert(0, copy);
    }
    Some(copy_result)
}

/// The op one [`UseSite::path`] names, borrowed to be rewritten — the reader half of `insert_at`.
fn op_at_mut<'a>(block: &'a mut [Op], path: &[u32]) -> Option<&'a mut Op> {
    let (&ordinal, rest) = path.split_first()?;
    let op = block.get_mut(ordinal as usize)?;
    let Some((&next, tail)) = rest.split_first() else {
        return Some(op);
    };
    let next = next as usize;
    let mut base = 0usize;
    for region in dialects::regions_mut(op) {
        let len = region.len();
        if next < base + len {
            let mut sub: Vec<u32> = Vec::with_capacity(1 + tail.len());
            sub.push((next - base) as u32);
            sub.extend_from_slice(tail);
            return op_at_mut(region, &sub);
        }
        base += len;
    }
    None
}

/// `builder.setInsertionPointAfter(symbol.getDefiningOp())` — `false` when nothing in this subtree
/// binds the value, which is the caller's fall back to the top of the unit.
fn insert_after_definition(scope: &mut Vec<Op>, val: Val, op: Op) -> bool {
    if let Some(at) = scope
        .iter()
        .position(|candidate| dialects::results(candidate).contains(&val))
    {
        scope.insert(at + 1, op);
        return true;
    }
    for holder in scope.iter_mut() {
        for region in dialects::regions_mut(holder) {
            if insert_after_definition(region, val, op.clone()) {
                return true;
            }
        }
    }
    false
}

/// Replaces: e152_clear
///
/// Drops the five per-unit collections between units.
///
/// ⛔ TRAP: `symbols_` SURVIVES. It is declared beside the five and this function leaves it alone,
/// which is why it is not a field of [`PerUnitState`] — the survival is a type-level fact here.
pub fn clear(state: &mut PerUnitState) {
    *state = PerUnitState::default();
}

/// Replaces: e153_getLocale
///
/// Which register file one use of a symbol reads it out of — the loop argument's locale, the parent's
/// `regLocales` entry for a `sentient.yield`, or the `scalar_add`/`scalar_sub`'s own `regLocale`.
///
/// ⛔ TRAP: DELIBERATE DIVERGENCE FOR A YIELD. The reference indexes `regLocales[getOperandNumber()]`,
/// but that array is `[bound, initArgs.., results..]` (`SentientOps.td:58-61`) so yield operand 0
/// reads the BOUND's entry; sibling `e154` adds the `+ 1` on the very same layout (`:490-492`). This
/// answers the carried value's own locale.
/// ⭐ `Unknown` IS THE TAIL — `emitError` then `DT_ERROR` then `return unknown` (`:471-473`), and the
/// caller's `DT_CHECK(locale != unknown)` is what turns it into a failure.
#[must_use]
pub fn locale_of_use(user: &Op, operand: usize, parent: Option<&Op>) -> RegType {
    match user {
        // `getValueRegLocale(for_op.getBody()->getArgument(operand))`: argument `i + 1` is
        // `carried[i]`'s. ⚠️ OPERAND 0 IS `$bound` AND THAT CASE IS REACHABLE — a symbol used as a
        // loop bound. Its argument 0 is the induction variable, whose `regLocales[0]` entry this
        // island does not carry (see [`dialects::value_reg_locale`]), so this answers `Unknown`
        // where the reference answers that entry.
        Op::Sentient(sentient::Op::For { carried, .. }) => operand
            .checked_sub(1)
            .and_then(|position| carried.get(position))
            .map_or(RegType::Unknown, |value| value.reg.locale),
        Op::Sentient(sentient::Op::Yield { .. }) => match parent {
            Some(Op::Sentient(sentient::Op::For { carried, .. })) => carried
                .get(operand)
                .map_or(RegType::Unknown, |value| value.reg.locale),
            Some(Op::Sentient(sentient::Op::If { yielded, .. })) => yielded
                .get(operand)
                .map_or(RegType::Unknown, |value| value.reg.locale),
            _ => RegType::Unknown,
        },
        Op::Sentient(sentient::Op::ScalarAdd { reg, .. } | sentient::Op::ScalarSub { reg, .. }) => {
            reg.map_or(RegType::Unknown, |reg| reg.locale)
        }
        _ => RegType::Unknown,
    }
}

/// Replaces: e154_propagateElementSizeToCopyOp
///
/// Carries a width onto the inserted `sentient.scalar_copy` — the using op's own `element_size`, or
/// for a `sentient.yield` the enclosing loop's `element_sizes` slot for the value being yielded.
///
/// ⛔ TRAP: THE FIRST BRANCH'S `|| user->hasAttr("element_sizes")` READS THE WRONG NAME. It then
/// copies `getAttr("element_size")`, which a `sentient.for` never has, so the reference sets a NULL
/// attribute — and its `DT_CHECK_MSG(getNumResults() == 1)` refuses any loop carrying two values
/// first. No width can come out of that branch, so a `for` user answers "nothing propagated"; the
/// caller (`:404`) discards this bool, so nothing observes the difference.
/// ⛔ TRAP: `sentient.if` IS STILL AN ISLAND GAP — [`sentient::Yielded`] carries no width, so that
/// parent answers false, which is the reference's own `hasAttr("element_sizes") == false`.
pub fn propagate_element_size_to_copy_op(
    user: &Op,
    operand: usize,
    parent: Option<&Op>,
    copy: &mut Op,
) -> bool {
    let Some(width) =
        user_element_size(user).or_else(|| yielded_element_size(user, operand, parent))
    else {
        return false;
    };
    if let Op::Sentient(sentient::Op::ScalarCopy { element_size, .. }) = copy {
        *element_size = Some(width);
        return true;
    }
    false
}

/// `element_sizes[operandNumber + 1]` of the loop a `sentient.yield` closes (`:484-498`).
///
/// ⭐ THE `+ 1` SKIPS THE BOUND'S ENTRY, so the reference's index and this island's carried-indexed
/// [`sentient::Carried::element_size`] name the same slot.
fn yielded_element_size(user: &Op, operand: usize, parent: Option<&Op>) -> Option<Bits> {
    let Op::Sentient(sentient::Op::Yield { .. }) = user else {
        return None;
    };
    match parent {
        Some(Op::Sentient(sentient::Op::For { carried, .. })) => {
            carried.get(operand).and_then(|value| value.element_size)
        }
        _ => None,
    }
}

/// `user->getAttr("element_size")` — the three ops of this island that carry the attribute.
const fn user_element_size(user: &Op) -> Option<Bits> {
    match user {
        Op::Sentient(
            sentient::Op::ScalarAdd { element_size, .. }
            | sentient::Op::ScalarSub { element_size, .. }
            | sentient::Op::ScalarCopy { element_size, .. },
        ) => *element_size,
        _ => None,
    }
}

/// Replaces: e155_dumpSymbolicLocales
///
/// The `symbolic_locales:[LRF, JCR]` trace line — `llvm::interleave` with `", "`.
#[must_use]
pub fn dump_symbolic_locales(symbolic_locales: &[RegType]) -> String {
    let mut out = String::from("symbolic_locales:[");
    for (position, locale) in symbolic_locales.iter().enumerate() {
        if position > 0 {
            out.push_str(", ");
        }
        out.push_str(locale_to_register_class(*locale));
    }
    out.push_str("]\n");
    out
}

/// Replaces: e156_dumpCandidates
///
/// Two lines per candidate: where the symbol is together with the op that defines it, then its locale.
///
/// ⛔ TRAP: `getLocation(..).getLine()` HAS NOTHING TO READ — no op on any rung of this crate's
/// islands carries a source location, and a line number is provenance of the input rather than a
/// result of the pass, so the parenthesised number prints `?`. Every other byte is the reference's.
/// ⭐ `Value::dump()` PRINTS THE DEFINING OPERATION, and [`print::emit`] already ends that line.
#[must_use]
pub fn dump_candidates(candidates: &[CandidateEntry], scope: &[Op]) -> String {
    let enclosing = [scope];
    let defs = dialects::Definitions::from_innermost(&enclosing);
    let mut out = String::new();
    for entry in candidates {
        out.push_str("symbol on line (?): ");
        match defs.of(entry.symbol) {
            Some(op) => print::emit(&mut out, op, 0),
            None => out.push('\n'),
        }
        let _ = writeln!(out, "\tlocale [{}]", locale_to_register_class(entry.locale));
    }
    out
}

/// `getOrComputeRegisterPressure(locale == lccr ? jcr : locale, kNumFreeRegisters)` — *"our goal is
/// to replace lccrs with jcrs"* (`:302-305`, `:352-355`), asked once per locale in each of e358's loops.
fn free_registers<P: RegisterPressure>(rp: &mut P, locale: RegType) -> Pressure {
    let asked = match locale {
        RegType::Lccr => RegType::Jcr,
        other => other,
    };
    rp.get_or_compute_register_pressure(asked, Metric::NumFreeRegisters)
}

/// Replaces: e358_collectCandidates
///
/// Picks, per symbolic register file that still has a free register, the symbols used in it —
/// most-used first — and takes as many as both the free registers and the free ibuff space allow.
///
/// ⛔ TRAP: A LOCALE WITH ONE CANDIDATE IS NOT DROPPED. `size() < 2` (`:331`) skips only the SORT and
/// the second loop still takes that candidate; the count that decides insertion is e359's.
/// ⛔ TRAP: A NEGATIVE `getRemainingIbuffSpace` WRAPS — it is read into an `unsigned` (`:346`), so an
/// overrun unit stops constraining `N` rather than driving it to zero.
/// ⭐ `DT_CHECK(candidates.empty())` IS THE RETURN TYPE, as e149's two are.
#[must_use]
pub fn collect_candidates<P: RegisterPressure, I: InstructionEstimator>(
    unit_body: &[Op],
    symbols: &[Val],
    state: &PerUnitState,
    rp: &mut P,
    estimator: &mut I,
    trace: &mut String,
) -> Vec<CandidateEntry> {
    let mut per_locale: Vec<(RegType, Vec<Val>)> = Vec::new();
    for &locale in &state.symbolic_locales {
        if free_registers(rp, locale).0 == 0 {
            if DEBUG {
                let _ = writeln!(
                    trace,
                    "skip collecting candidates for locale [{}] because there are no free registers.",
                    locale_to_register_class(locale)
                );
            }
            continue;
        }
        // The `collectPerLocalCandidate` lambda over `symbols_` then `symbol_queries_` — a pair the
        // collector never recorded has no uses, which is the two `find` misses it returns on.
        for &sym in symbols.iter().chain(&state.ops.symbol_queries) {
            if state.symbol_to_usage.uses(sym, locale).is_empty() {
                continue;
            }
            match per_locale.iter_mut().find(|(at, _)| *at == locale) {
                Some((_, syms)) => syms.push(sym),
                None => per_locale.push((locale, vec![sym])),
            }
        }
        let Some((_, syms)) = per_locale.iter_mut().find(|(at, _)| *at == locale) else {
            continue;
        };
        if syms.len() < 2 {
            continue;
        }
        // `std::stable_sort` descending by number of uses, and `sort_by` is stable too — which the
        // reference's own *"todo: for candidates that have the same num uses"* relies on.
        syms.sort_by(|a, b| {
            state
                .symbol_to_usage
                .uses(*b, locale)
                .len()
                .cmp(&state.symbol_to_usage.uses(*a, locale).len())
        });
    }
    // *"Since JCR copy_ops cannot be packet initialized, they use up ibuff space"* — asked once, after
    // every pressure query of the first loop.
    let num_free_insts = estimator.remaining_ibuff_space(unit_body).0 as u32;
    let mut candidates = Vec::new();
    for &locale in &state.symbolic_locales {
        let Some((_, syms)) = per_locale.iter().find(|(at, _)| *at == locale) else {
            continue;
        };
        let taken = free_registers(rp, locale)
            .0
            .min(num_free_insts)
            .min(syms.len() as u32);
        for &symbol in &syms[..taken as usize] {
            candidates.push(CandidateEntry { symbol, locale });
        }
    }
    candidates
}

/// `-dcc-scalar-copy-insertion-for-symbols-force-one-candidate`, `cl::init(false)` (`:67-72`) —
/// *"only meant to be enabled for testing"*, and this crate has no flags.
const FORCE_INSERTION_EVEN_IF_ONE_CANDIDATE: bool = false;

/// Replaces: e359_insertCopyOpsForCandidates
///
/// Inserts one `sentient.scalar_copy` per candidate and points every recorded use of that symbol in
/// that register file at it, unless too few candidates share a file to reduce a patch flit.
///
/// ⛔ TRAP: THE TWO COUNTS ARE SEPARATE — `jcr_count < 2 && non_jcr_count < 2`, so one candidate in
/// each file returns, and `lccr` counts as `jcr` here as it does in e358.
/// ⛔ TRAP: THE THREE PHASES ARE ONE LOOP THERE. A use is a POSITION on this island, so every use is
/// repointed first, e151 then inserts while those positions are still pristine, and each remaining
/// copy is placed last against the ops its own freshly minted result identifies.
pub fn insert_copy_ops_for_candidates(
    unit_body: &mut Vec<Op>,
    candidates: &[CandidateEntry],
    usage: &SymbolUsage,
    values: &mut Values,
    trace: &mut String,
) {
    if !FORCE_INSERTION_EVEN_IF_ONE_CANDIDATE {
        let jcr_count = candidates
            .iter()
            .filter(|entry| JcrLocale::of(entry.locale).is_some())
            .count();
        if jcr_count < 2 && candidates.len() - jcr_count < 2 {
            if DEBUG {
                trace.push_str(
                    "Skip inserting copy op since there are not enough candidates to create patch \
                     flit reduction opportunity\n",
                );
            }
            return;
        }
    }
    let mut planned: Vec<(CandidateEntry, Val)> = Vec::new();
    for entry in candidates {
        if JcrLocale::of(entry.locale).is_some() {
            continue;
        }
        // `DT_CHECK(symbol_to_usage_.find(symbol) != end)` and the same for the locale: a candidate
        // came from a recorded pair, so an empty list is the check and not a case.
        if usage.uses(entry.symbol, entry.locale).is_empty() {
            continue;
        }
        let copy_result = values.mint();
        for at in usage.uses(entry.symbol, entry.locale) {
            if let Some(user) = op_at_mut(unit_body, &at.path) {
                dialects::set_operand(user, at.operand, copy_result);
            }
        }
        planned.push((*entry, copy_result));
    }
    for entry in candidates {
        if let Some(locale) = JcrLocale::of(entry.locale) {
            // e151 mints its own copy result and repoints the uses itself; nothing here reads it.
            let _ =
                insert_copy_ops_for_jcr_candidate(unit_body, entry.symbol, locale, usage, values);
        }
    }
    for (entry, copy_result) in planned {
        insert_one_copy(unit_body, entry, copy_result, usage);
    }
}

/// The `for (OpOperand *opnd : ..)` body of e359 once its uses read `copy_result`: the copy is built
/// at the first of them with e154's width, then hoisted to a common dominator over each of the rest.
fn insert_one_copy(
    unit_body: &mut Vec<Op>,
    entry: CandidateEntry,
    copy_result: Val,
    usage: &SymbolUsage,
) {
    let users = use_positions(unit_body, copy_result);
    let Some(first) = users.first() else {
        return;
    };
    let mut copy = Op::Sentient(sentient::Op::ScalarCopy {
        input: entry.symbol,
        result: copy_result,
        reg: Reg {
            locale: entry.locale,
            index: None,
        },
        program_header: false,
        element_size: None,
    });
    if let Some(user) = first.op(unit_body) {
        let parent = first.parent().and_then(|at| at.op(unit_body));
        let operand = usage.uses(entry.symbol, entry.locale)[0].operand;
        propagate_element_size_to_copy_op(user, operand, parent, &mut copy);
    }
    // `OpBuilder builder(user)` — the copy lands immediately ahead of the first use, in its block.
    insert_at(unit_body, first, copy);
    for rank in 1..users.len() {
        let Some(copy_at) = path_of(unit_body, copy_result) else {
            continue;
        };
        // Recomputed, because e241 rehoists the operand definitions it has to move with the copy.
        let refreshed = use_positions(unit_body, copy_result);
        let Some(user_at) = refreshed.get(rank) else {
            continue;
        };
        if move_to_common_dominator(unit_body, &copy_at, &NewUse::SameUnit(user_at.clone()))
            == Hoisted::OutsideProgramUnit
        {
            // `user->emitError("failed to move copy op to common dominator"); signalPassFailure();`
            // (`:407-409`) — unreachable, since every recorded use is in THIS unit and that is
            // [`Hoisted::OutsideProgramUnit`]'s one precondition.
            panic!("failed to move copy op to common dominator (`:407`) for {copy_result:?}");
        }
    }
}

/// `val.getUses()` AS POSITIONS — every op of the unit reading `val` in its OWN operands, in the
/// pre-order [`collect_usage`] records them in, and never an enclosing op whose region holds one.
fn use_positions(unit_body: &[Op], val: Val) -> Vec<OpAt> {
    fn walk(block: &[Op], val: Val, enclosing: &mut Vec<(InBlock, usize)>, found: &mut Vec<OpAt>) {
        for (index, op) in block.iter().enumerate() {
            if dialects::operands(op).contains(&val) {
                found.push(OpAt::at(enclosing, InBlock(index)));
            }
            for (region, inner) in dialects::regions_ref(op).into_iter().enumerate() {
                enclosing.push((InBlock(index), region));
                walk(inner, val, enclosing, found);
                enclosing.pop();
            }
        }
    }
    let mut found = Vec::new();
    walk(unit_body, val, &mut Vec::new(), &mut found);
    found
}

/// Replaces: e360_dumpSymbolUsageInfo
///
/// The `=== Usage Info: ===` block — one line per symbol and register file, naming the symbol's ID or
/// the line its `symbol.query_map` sits on together with how many uses were recorded there.
///
/// ⛔ TRAP: `getLocation(..).getLine()` HAS NOTHING TO READ, as in e156 — a line number is provenance
/// of the input rather than a result of the pass, so a query's parenthesised number prints `?`.
/// ⭐ THE LOCALES ITERATED ARE `symbolic_locales_`, not the symbol's own, which is what makes the
/// `find` miss inside the loop a `continue` rather than the outer `return`.
#[must_use]
pub fn dump_symbol_usage_info(state: &PerUnitState, symbols: &[Val], scope: &[Op]) -> String {
    let enclosing = [scope];
    let defs = dialects::Definitions::from_innermost(&enclosing);
    let mut out = String::from("=== Usage Info: ===\n");
    for &sym in symbols.iter().chain(&state.ops.symbol_queries) {
        for &locale in &state.symbolic_locales {
            let count = state.symbol_to_usage.uses(sym, locale).len();
            if count == 0 {
                continue;
            }
            let class = locale_to_register_class(locale);
            match defs.of(sym) {
                Some(Op::Symbol(symbol::Op::CreateSymbol { symbol_id, .. })) => {
                    let _ = writeln!(
                        out,
                        "Sym ID ({symbol_id}) for locale [{class}] has {count} use(s)."
                    );
                }
                Some(Op::Symbol(symbol::Op::QueryMap { .. })) => {
                    let _ = writeln!(
                        out,
                        "Symbol Query On Line (?) for locale [{class}] has {count} use(s)."
                    );
                }
                _ => panic!(
                    "DT_CHECK(static_cast<bool>(sym_op) ^ static_cast<bool>(sym_query_op)) (`:526`) \
                     for {sym:?}"
                ),
            }
        }
    }
    out.push_str("=== End of Usage Info ===\n");
    out
}

/// Replaces: e527_collectSymbolUsage
///
/// Records, per symbol and per register file, every use in this unit a `jcr` copy could stand in front
/// of, skipping the copies and the mapping ops.
///
/// ⛔ TRAP: THE TWO `DT_CHECK(..empty())` ARE THE RETURN TYPE, and `DT_CHECK(use_unit)` with
/// `use_unit != unit` IS THE WALK — only `unit_body` is visited, so neither is constructible.
/// ⚠️ TRAP: e153 ANSWERS `Unknown` FOR A SYMBOL USED AS A `sentient.for` `$bound`, so that one input
/// reaches the `DT_CHECK` here where the reference reads the induction variable's locale instead.
#[must_use]
pub fn collect_symbol_usage(
    unit_body: &[Op],
    symbols: &[Val],
    symbol_queries: &[Val],
) -> (SymbolUsage, Vec<RegType>) {
    let mut usage = SymbolUsage::default();
    let mut symbolic_locales = Vec::new();
    for &sym in symbols.iter().chain(symbol_queries) {
        collect_usage(
            unit_body,
            sym,
            &[],
            0,
            None,
            false,
            &mut usage,
            &mut symbolic_locales,
        );
    }
    (usage, symbolic_locales)
}

/// The `collectUsage` lambda's `sym.getUses()` loop, as a walk: an op of this island has no use list,
/// so a use is found by visiting the unit and testing every operand slot. `inside_uniform` is
/// `getParentOfType<UniformizeRegionsOp, EqualizePatternOp>` carried down, as in
/// [`walk_ops_of_interest`]; `offset` is where this region starts in its holder's concatenated regions,
/// which is what a [`UseSite::path`] ordinal counts.
#[expect(
    clippy::too_many_arguments,
    reason = "the walk carries the reference's `this`, its lambda capture and the parent chain an op               of this island cannot be asked for"
)]
fn collect_usage(
    scope: &[Op],
    sym: Val,
    prefix: &[u32],
    offset: u32,
    parent: Option<&Op>,
    inside_uniform: bool,
    usage: &mut SymbolUsage,
    symbolic_locales: &mut Vec<RegType>,
) {
    for (index, op) in scope.iter().enumerate() {
        let mut path: Vec<u32> = Vec::with_capacity(prefix.len() + 1);
        path.extend_from_slice(prefix);
        path.push(offset + index as u32);
        if !matches!(
            op,
            Op::Sentient(sentient::Op::ScalarCopy { .. })
                | Op::Symbol(symbol::Op::ImmutableMapping { .. })
                | Op::Uniform(uniform::Op::DefImmutableMapping { .. })
        ) {
            for (operand, val) in dialects::operands(op).into_iter().enumerate() {
                if val != sym {
                    continue;
                }
                let locale = locale_of_use(op, operand, parent);
                if locale == RegType::Unknown {
                    panic!(
                        "DT_CHECK(locale != SentientRegType::unknown) (`:256`) for operand {operand}                          of {op:?}"
                    );
                }
                if inside_uniform && matches!(locale, RegType::Jcr | RegType::Lccr) {
                    continue;
                }
                if !symbolic_locales.contains(&locale) {
                    symbolic_locales.push(locale);
                }
                usage.record(
                    sym,
                    locale,
                    UseSite {
                        path: path.clone(),
                        operand,
                    },
                );
            }
        }
        let nested_uniform = inside_uniform
            || matches!(
                op,
                Op::Uniform(
                    uniform::Op::UniformizeRegions { .. } | uniform::Op::EqualizePattern { .. }
                )
            );
        let mut base = 0u32;
        for region in dialects::regions_ref(op) {
            collect_usage(
                region,
                sym,
                &path,
                base,
                Some(op),
                nested_uniform,
                usage,
                symbolic_locales,
            );
            base += region.len() as u32;
        }
    }
}

/// `LLVM_DEBUG` — `-debug-only=scalar-copy-insertion-for-symbols`, off unless `dcc-opt` is asked for
/// it, and the reason the trace calls below are present but not taken.
const DEBUG: bool = false;

/// `DEBUG_WITH_TYPE(VerboseDebug, ..)` — `DEBUG_TYPE "-verbose"` (`:41`), a second and narrower flag.
const VERBOSE_DEBUG: bool = false;

/// Replaces: e576_runOn
///
/// One program unit's six steps: collect its copies and outermost loops, map every symbol's uses to
/// the register files reading them, pessimize liveness over those files, then choose candidates
/// against the resulting register pressure and insert a copy for each (`:148-214`).
///
/// ⛔ TRAP: THE STEPS ARE ORDERED BY DATA AND NOT BY THE COMMENT. `collectSymbolUsage` reads
/// `symbol_queries_` that `collectOpsOfInterest` filled, and `pessimizeLiveness` reads the
/// `symbolic_locales_` that `collectSymbolUsage` filled — swapping any pair silently empties the next.
/// ⭐ THE `Liveness` AND `RegisterPressure` ARE PARAMETERS, not built here: both are out of campaign
/// scope, and the reference's own note is that an `rp` built before `pessimizeLiveness` is invalid
/// afterwards — which is why its first dump, under [`VERBOSE_DEBUG`], reads a DIFFERENT instance.
#[expect(
    clippy::too_many_arguments,
    reason = "the reference reaches five of these through `this` and two more through the pass \
              manager, and neither is a thing this crate has"
)]
pub fn run_on<L: Liveness, P: RegisterPressure, I: InstructionEstimator>(
    unit_body: &mut Vec<Op>,
    symbols: &[Val],
    state: &mut PerUnitState,
    liveness: &mut L,
    rp: &mut P,
    estimator: &mut I,
    values: &mut Values,
    trace: &mut String,
) {
    if VERBOSE_DEBUG {
        trace.push_str("Register Pressure before pessimizing liveness:\n");
        rp.compute_register_pressure_for_all_locales();
        trace.push_str(&rp.dump());
    }
    if DEBUG {
        let _ = writeln!(trace, "num global symbols (seen so far):{}", symbols.len());
    }
    state.ops = collect_ops_of_interest(unit_body);
    if DEBUG {
        let _ = writeln!(
            trace,
            "num query_symbol ops:{}\nnum scalar_copy ops:{}\nnum outermost loops:{}",
            state.ops.symbol_queries.len(),
            state.ops.scalar_copy_ops.len(),
            state.ops.outermost_loops.len()
        );
    }
    let (usage, symbolic_locales) =
        collect_symbol_usage(unit_body, symbols, &state.ops.symbol_queries);
    state.symbol_to_usage = usage;
    state.symbolic_locales = symbolic_locales;
    if DEBUG {
        trace.push_str(&dump_symbolic_locales(&state.symbolic_locales));
        trace.push_str(&dump_symbol_usage_info(state, symbols, unit_body));
    }
    // ⭐ A COPY OF THE CHILD ANALYSIS, *"since this pass does not preserve liveness, the changes made
    // in the liveness object will not affect downstream transformations"* — the copy is the mechanism.
    pessimize_liveness(state, unit_body, liveness);
    if VERBOSE_DEBUG {
        trace.push_str("Register Pressure after pessimizing liveness:\n");
        rp.compute_register_pressure_for_all_locales();
        trace.push_str(&rp.dump());
    }
    let candidates = collect_candidates(unit_body, symbols, state, rp, estimator, trace);
    if DEBUG {
        trace.push_str("Selected the following candidates for copy-op insertion:\n");
        trace.push_str(&dump_candidates(&candidates, unit_body));
    }
    insert_copy_ops_for_candidates(
        unit_body,
        &candidates,
        &state.symbol_to_usage,
        values,
        trace,
    );
}

/// `-dcc-scalar-copy-insertion-for-symbols-disable`, `cl::init(false)` (`:62-65`) — a `dcc-opt`
/// command-line flag, not a program property, and this crate has no flags.
const DISABLE_THIS_PASS: bool = false;

/// `opts_.OptLevel == 0 || opts_.OptLevel == 1` (`:132`) — the pipeline's optimization level, which
/// this crate compiles at its default and not at `-O0`/`-O1`.
const OPT_LEVEL_BELOW_TWO: bool = false;

/// EVERY `symbol.create_symbol` OF A BLOCK AND ITS REGIONS, in the pre-order the walk sees them.
fn create_symbols_in(block: &[Op], out: &mut Vec<Val>) {
    for op in block {
        if let Op::Symbol(symbol::Op::CreateSymbol { result, .. }) = op {
            out.push(*result);
        }
        for region in dialects::regions_ref(op) {
            create_symbols_in(region, out);
        }
    }
}

/// Replaces: e610_runOnOperation
///
/// The pass entry: unless a flag or `-O0`/`-O1` turns it off, insert the symbol copies of every
/// program unit, each against the symbols declared ahead of it and its own fresh state (`:130-146`).
///
/// ⛔ TRAP: THE REFERENCE'S ONE WALK IS SPLIT IN TWO HERE, so a unit sees EVERY declared symbol and
/// not only those preceding it. Its own comment claims the pre-order gives it all of them anyway
/// (`:137-138`), and on this island the declarations live in [`Program::preamble`] — outside the units
/// the walk `skip()`s — so there is no order left to observe.
#[expect(
    clippy::too_many_arguments,
    reason = "the three analyses are the pass manager's `getChildAnalysis`, and neither it nor a \
              pass object is a thing this crate has"
)]
pub fn run_on_operation<
    A: Arch,
    M: Model,
    W: Workload,
    L: Liveness,
    P: RegisterPressure,
    I: InstructionEstimator,
>(
    program: &mut Program<A, M, W>,
    liveness: &mut L,
    rp: &mut P,
    estimator: &mut I,
    values: &mut Values,
    trace: &mut String,
) {
    if DISABLE_THIS_PASS || OPT_LEVEL_BELOW_TWO {
        return;
    }
    let mut symbols = Vec::new();
    create_symbols_in(&program.preamble, &mut symbols);
    let mut state = PerUnitState::default();
    for unit in program.units.iter_mut() {
        run_on(
            &mut unit.body,
            &symbols,
            &mut state,
            liveness,
            rp,
            estimator,
            values,
            trace,
        );
        clear(&mut state);
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::arch::Dd2;
    use crate::generated::OpFunc;
    use crate::islands::dataflow_ir::ty::ScalarTy;
    use crate::islands::dataflow_ir::{GroupId, OpIndex, ProgramName, Units};
    use crate::islands::sentient::{ProgramUnit, ProgramUnits};
    use crate::islands::sentient::dialects::dataflow;
    use crate::islands::sentient::dialects::sentient::Carried;
    use crate::transform::sentient::analyses::{InstructionCount, VirtualAssigns};
    use crate::units::{DfirUnit, Residency};

    /// A `sentient.scalar_copy` with no width, as `CopyOp::create`'s five-argument builder makes one.
    fn a_copy(input: Val, result: Val, locale: RegType) -> Op {
        Op::Sentient(sentient::Op::ScalarCopy {
            input,
            result,
            reg: Reg {
                locale,
                index: None,
            },
            program_header: false,
            element_size: None,
        })
    }

    /// A `sentient.scalar_add`.
    fn an_add(lhs: Val, rhs: Val, result: Val, locale: RegType, width: Option<Bits>) -> Op {
        Op::Sentient(sentient::Op::ScalarAdd {
            lhs,
            rhs,
            result,
            reg: Some(Reg {
                locale,
                index: None,
            }),
            element_size: width,
            ty: ScalarTy::Index,
        })
    }

    /// One carried value, all three of its handles distinct and no width.
    fn a_carried(init: Val, arg: Val, result: Val, locale: RegType) -> Carried {
        Carried {
            init,
            arg,
            result,
            reg: Reg {
                locale,
                index: None,
            },
            program_header: false,
            element_size: None,
        }
    }

    /// `Liveness&` recording what it is told, which is the only way to observe [`pessimize_liveness`]'s
    /// selection.
    #[derive(Default)]
    struct Recorder(Vec<Val>);
    impl Liveness for Recorder {
        fn update_live_ranges_for_program_header_promotion(&mut self, candidate: Val) {
            self.0.push(candidate);
        }

        fn is_live_range_overlaps(&self, _val1: Val, _val2: Val) -> bool {
            todo!("this fake records promotions only; no unit here asks it about an overlap")
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
        fn operand_to_index(
            &mut self,
            _value: Val,
        ) -> crate::transform::sentient::analyses::RegNode {
        todo!("no unit here reads a colouring node through this fake")
    }
}

    /// A `RegisterPressure&` that states its answers and records what it was asked — the estimate is
    /// out of campaign scope, so what e358 owns is WHICH locale it asks about and what it then takes.
    #[derive(Default)]
    struct StatedPressure {
        /// Free registers per locale, `0` for any locale not listed.
        free: Vec<(RegType, u32)>,
        /// Every ask, in order.
        asked: Vec<(RegType, Metric)>,
    }

    impl RegisterPressure for StatedPressure {
        fn get_or_compute_register_pressure(
            &mut self,
            locale: RegType,
            metric: Metric,
        ) -> Pressure {
            self.asked.push((locale, metric));
            Pressure(
                self.free
                    .iter()
                    .find(|(at, _)| *at == locale)
                    .map_or(0, |(_, free)| *free),
            )
        }

        fn compute_register_pressure_for_all_locales(&mut self) {
            todo!("no unit here asks this fake for every locale at once")
        }

        fn dump(&self) -> String {
            todo!("no unit here dumps this fake")
        }
    }

    /// An `InstructionEstimator&` that states the one count e358 reads, for the same reason.
    #[derive(Default)]
    struct StatedEstimator {
        /// What is left of the ibuff.
        ibuff_space: i32,
        /// How many times it was asked for.
        ibuff_asks: usize,
    }

    impl InstructionEstimator for StatedEstimator {
        fn recalculate(&mut self, _unit: &[Op]) {
            todo!("no unit here recounts this fake")
        }

        fn estimated_instruction_count_of_op(&mut self, _op: &Op) -> InstructionCount {
            todo!("no unit here costs an op through this fake")
        }

        fn estimated_instruction_count_of_region(&mut self, _region: &[Op]) -> InstructionCount {
            todo!("no unit here costs a region through this fake")
        }

        fn have_ibuff_space(&mut self, _unit: &[Op]) -> bool {
            todo!("no unit here asks this fake whether the ibuff has space")
        }

        fn remaining_ibuff_space(&mut self, _unit: &[Op]) -> InstructionCount {
            self.ibuff_asks += 1;
            InstructionCount(self.ibuff_space)
        }
    }

    /// `e149` — the four arms, with the inner loop rejected by `isOuterMostLoop` and the misplaced
    /// `create_symbol` recorded rather than collected.
    #[test]
    fn ops_of_interest_takes_the_outer_loop_and_warns_on_the_stray_symbol() {
        let body = vec![
            Op::Symbol(symbol::Op::CreateSymbol {
                result: Val(0),
                symbol_id: 7,
                max_value: None,
            }),
            Op::Symbol(symbol::Op::QueryMap {
                result: Val(1),
                map: Val(20),
                key: Val(21),
            }),
            Op::Sentient(sentient::Op::For {
                iv: Val(2),
                bound: Val(3),
                bound_reg: None,
                carried: Vec::new(),
                dbg_name: None,
                body: vec![
                    a_copy(Val(1), Val(4), RegType::Jcr),
                    Op::Sentient(sentient::Op::For {
                        iv: Val(5),
                        bound: Val(6),
                        bound_reg: None,
                        carried: Vec::new(),
                        dbg_name: None,
                        body: Vec::new(),
                    }),
                ],
            }),
        ];

        let found = collect_ops_of_interest(&body);

        assert_eq!(found.scalar_copy_ops, vec![Val(4)]);
        assert_eq!(found.outermost_loops, vec![ForRef(Val(2))]);
        assert_eq!(found.symbol_queries, vec![Val(1)]);
        assert_eq!(found.misplaced_create_symbols, vec![Val(0)]);
    }

    /// `e150` — the copy result and the symbol-initialised iter arg are promoted; the iter arg whose
    /// initialiser is neither a constant nor a symbol is not.
    #[test]
    fn pessimize_liveness_promotes_the_copy_and_the_symbolic_iter_arg() {
        let body = vec![
            Op::Symbol(symbol::Op::CreateSymbol {
                result: Val(0),
                symbol_id: 7,
                max_value: None,
            }),
            a_copy(Val(0), Val(1), RegType::Jcr),
            an_add(Val(0), Val(0), Val(2), RegType::Jcr, None),
            Op::Sentient(sentient::Op::For {
                iv: Val(3),
                bound: Val(4),
                bound_reg: None,
                carried: vec![
                    a_carried(Val(0), Val(5), Val(6), RegType::Jcr),
                    a_carried(Val(2), Val(7), Val(8), RegType::Jcr),
                ],
                dbg_name: None,
                body: Vec::new(),
            }),
        ];
        let state = PerUnitState {
            ops: collect_ops_of_interest(&body),
            symbol_to_usage: SymbolUsage::default(),
            symbolic_locales: vec![RegType::Jcr],
        };

        let mut liveness = Recorder::default();
        pessimize_liveness(&state, &body, &mut liveness);

        assert_eq!(liveness.0, vec![Val(1), Val(5)]);
    }

    /// `e150` NEGATIVE — ⛔ THE KEY DECIDES WHICH VALUE `isSymbol` SEES. One
    /// `uniform.def_immutable_mapping` holds a non-symbol under `%0` and a symbol under `%1`; only the
    /// iter arg initialised by the query that asks for `%1` is promoted. Scanning both pairs — the
    /// shape this file carried before — promotes them both.
    #[test]
    fn a_query_map_is_symbolic_only_for_the_key_that_selects_the_symbol() {
        let a_unit = |result| {
            Op::Dataflow(dataflow::Op::GetUnit {
                result,
                residency: Residency::Global,
                unit: DfirUnit::Sfp,
                num_folds: None,
                reg_locale: None,
            })
        };
        let body = vec![
            a_unit(Val(0)),
            a_unit(Val(1)),
            Op::Sentient(sentient::Op::ScalarConstant {
                value: 4,
                result: Val(2),
                reg_locale: RegType::Imm,
                ty: ScalarTy::Index,
                is_symbol: false,
            }),
            Op::Symbol(symbol::Op::CreateSymbol {
                result: Val(3),
                symbol_id: 7,
                max_value: None,
            }),
            Op::Uniform(uniform::Op::DefImmutableMapping {
                result: Val(4),
                pairs: vec![(Val(0), Val(2)), (Val(1), Val(3))],
            }),
            Op::Uniform(uniform::Op::QueryMap {
                result: Val(5),
                map: Val(4),
                key: Val(0),
            }),
            Op::Uniform(uniform::Op::QueryMap {
                result: Val(6),
                map: Val(4),
                key: Val(1),
            }),
            Op::Sentient(sentient::Op::For {
                iv: Val(7),
                bound: Val(8),
                bound_reg: None,
                carried: vec![
                    a_carried(Val(5), Val(9), Val(10), RegType::Jcr),
                    a_carried(Val(6), Val(11), Val(12), RegType::Jcr),
                ],
                dbg_name: None,
                body: Vec::new(),
            }),
        ];
        let state = PerUnitState {
            ops: collect_ops_of_interest(&body),
            symbol_to_usage: SymbolUsage::default(),
            symbolic_locales: vec![RegType::Jcr],
        };

        let mut liveness = Recorder::default();
        pessimize_liveness(&state, &body, &mut liveness);

        assert_eq!(liveness.0, vec![Val(11)]);
    }

    /// `e151` — one copy for two uses, placed after the `symbol.query_map` that defines the symbol.
    #[test]
    fn one_jcr_copy_lands_after_the_query_and_takes_over_both_uses() {
        let mut values = Values::default();
        let map = values.mint();
        let key = values.mint();
        let sym = values.mint();
        let other = values.mint();
        let sum = values.mint();
        let difference = values.mint();
        let mut body = vec![
            Op::Symbol(symbol::Op::QueryMap {
                result: sym,
                map,
                key,
            }),
            an_add(sym, other, sum, RegType::Jcr, None),
            Op::Sentient(sentient::Op::ScalarSub {
                lhs: other,
                rhs: sym,
                result: difference,
                reg: Some(Reg {
                    locale: RegType::Jcr,
                    index: None,
                }),
                element_size: None,
                ty: ScalarTy::Index,
            }),
        ];
        let mut usage = SymbolUsage::default();
        usage.record(
            sym,
            RegType::Jcr,
            UseSite {
                path: vec![1],
                operand: 0,
            },
        );
        usage.record(
            sym,
            RegType::Jcr,
            UseSite {
                path: vec![2],
                operand: 1,
            },
        );

        let copied =
            insert_copy_ops_for_jcr_candidate(&mut body, sym, JcrLocale::Jcr, &usage, &mut values);

        assert_eq!(copied, Some(Val(6)));
        assert_eq!(body[1], a_copy(sym, Val(6), RegType::Jcr));
        assert_eq!(dialects::operands(&body[2]), vec![Val(6), other]);
        assert_eq!(dialects::operands(&body[3]), vec![other, Val(6)]);
    }

    /// `e152` — the five collections go, and `symbols_` is not among them to go.
    #[test]
    fn clear_empties_the_five_per_unit_collections() {
        let mut usage = SymbolUsage::default();
        usage.record(
            Val(0),
            RegType::Jcr,
            UseSite {
                path: vec![1],
                operand: 0,
            },
        );
        let mut state = PerUnitState {
            ops: OpsOfInterest {
                scalar_copy_ops: vec![Val(1)],
                outermost_loops: vec![ForRef(Val(2))],
                symbol_queries: vec![Val(3)],
                misplaced_create_symbols: vec![Val(4)],
            },
            symbol_to_usage: usage,
            symbolic_locales: vec![RegType::Jcr],
        };

        clear(&mut state);

        assert_eq!(state, PerUnitState::default());
    }

    /// `e153` — a loop argument, a yield inside each of the two parents, an arithmetic user, and the
    /// `emitError` tail.
    #[test]
    fn the_locale_of_a_use_comes_from_the_user() {
        let loop_op = Op::Sentient(sentient::Op::For {
            iv: Val(0),
            bound: Val(1),
            bound_reg: None,
            carried: vec![a_carried(Val(2), Val(3), Val(4), RegType::Lccr)],
            dbg_name: None,
            body: Vec::new(),
        });
        let yield_op = Op::Sentient(sentient::Op::Yield {
            results: vec![Val(5)],
        });

        // Operand 1 is region argument 1, which is `carried[0]`.
        assert_eq!(locale_of_use(&loop_op, 1, None), RegType::Lccr);
        // The bound's own entry, which this island does not carry.
        assert_eq!(locale_of_use(&loop_op, 0, None), RegType::Unknown);
        assert_eq!(locale_of_use(&yield_op, 0, Some(&loop_op)), RegType::Lccr);
        assert_eq!(
            locale_of_use(&an_add(Val(6), Val(7), Val(8), RegType::Jcr, None), 0, None),
            RegType::Jcr
        );
        // `unimplemented support for this type of user`.
        assert_eq!(
            locale_of_use(&a_copy(Val(9), Val(10), RegType::Jcr), 0, None),
            RegType::Unknown
        );
    }

    /// `e154` — the width crosses onto the copy from a compute and from the loop a yield closes, and
    /// a user carrying none leaves it absent.
    #[test]
    fn the_element_size_crosses_onto_the_copy_only_when_the_user_has_one() {
        let mut copy = a_copy(Val(0), Val(1), RegType::Jcr);

        let carried = propagate_element_size_to_copy_op(
            &an_add(Val(2), Val(3), Val(4), RegType::Jcr, Some(Bits(16))),
            0,
            None,
            &mut copy,
        );

        assert!(carried);
        assert_eq!(
            copy,
            Op::Sentient(sentient::Op::ScalarCopy {
                input: Val(0),
                result: Val(1),
                reg: Reg {
                    locale: RegType::Jcr,
                    index: None,
                },
                program_header: false,
                element_size: Some(Bits(16)),
            })
        );

        // `element_sizes[operandNumber + 1]` of the enclosing loop, for the value the yield carries.
        let loop_op = Op::Sentient(sentient::Op::For {
            iv: Val(5),
            bound: Val(6),
            bound_reg: None,
            carried: vec![Carried {
                element_size: Some(Bits(8)),
                ..a_carried(Val(7), Val(8), Val(9), RegType::Lccr)
            }],
            dbg_name: None,
            body: Vec::new(),
        });
        let mut yielded = a_copy(Val(0), Val(1), RegType::Jcr);
        let through_the_loop = propagate_element_size_to_copy_op(
            &Op::Sentient(sentient::Op::Yield {
                results: vec![Val(9)],
            }),
            0,
            Some(&loop_op),
            &mut yielded,
        );

        assert!(through_the_loop);
        assert_eq!(
            yielded,
            Op::Sentient(sentient::Op::ScalarCopy {
                input: Val(0),
                result: Val(1),
                reg: Reg {
                    locale: RegType::Jcr,
                    index: None,
                },
                program_header: false,
                element_size: Some(Bits(8)),
            })
        );

        let mut untouched = a_copy(Val(0), Val(1), RegType::Jcr);
        let none = propagate_element_size_to_copy_op(
            &an_add(Val(2), Val(3), Val(4), RegType::Jcr, None),
            0,
            None,
            &mut untouched,
        );

        assert!(!none);
        assert_eq!(untouched, a_copy(Val(0), Val(1), RegType::Jcr));
    }

    /// `e155` — the interleaved line, with both XRF pointers naming the one class.
    #[test]
    fn the_symbolic_locales_line_names_register_classes() {
        assert_eq!(
            dump_symbolic_locales(&[RegType::Lrf, RegType::XrfRdPtr, RegType::Jcr]),
            "symbolic_locales:[LRF, XRF, JCR]\n"
        );
    }

    /// `e156` — two lines per candidate, the defining op dumped between them.
    #[test]
    fn the_candidate_dump_carries_the_defining_op_and_the_locale() {
        let scope = vec![Op::Symbol(symbol::Op::CreateSymbol {
            result: Val(0),
            symbol_id: 7,
            max_value: None,
        })];
        let candidates = [CandidateEntry {
            symbol: Val(0),
            locale: RegType::Jcr,
        }];

        assert_eq!(
            dump_candidates(&candidates, &scope),
            "symbol on line (?): %0 = symbol.create_symbol {SymbolId = 7 : i32} : index\n\
             \tlocale [JCR]\n"
        );
    }
    /// e576 — the six steps in the order the data needs, ending on the ONE candidate e359 refuses to
    /// insert for.
    #[test]
    fn e576_collects_maps_and_pessimizes_then_finds_one_candidate_too_few() {
        let sym = Val(0);
        let mut body = vec![
            a_copy(sym, Val(1), RegType::Jcr),
            an_add(sym, Val(6), Val(7), RegType::Jcr, None),
        ];
        let mut state = PerUnitState::default();
        let mut liveness = Recorder::default();
        let mut rp = StatedPressure {
            free: vec![(RegType::Jcr, 4)],
            asked: Vec::new(),
        };
        let mut estimator = StatedEstimator {
            ibuff_space: 9,
            ibuff_asks: 0,
        };
        let mut values = Values::default();
        let mut trace = String::new();

        run_on(
            &mut body,
            &[sym],
            &mut state,
            &mut liveness,
            &mut rp,
            &mut estimator,
            &mut values,
            &mut trace,
        );

        // ⭐ ONE `jcr` CANDIDATE AND NO OTHER FILE, so e359 returns before inserting anything.
        assert_eq!(body.len(), 2);
        assert_eq!(state.ops.scalar_copy_ops, vec![Val(1)]);
        // ⛔ THE COPY'S OWN USE OF THE SYMBOL IS NOT ONE, so `jcr` comes from the `scalar_add`.
        assert_eq!(state.symbolic_locales, vec![RegType::Jcr]);
        assert_eq!(
            state.symbol_to_usage.uses(sym, RegType::Jcr),
            &[UseSite {
                path: vec![1],
                operand: 0,
            }]
        );
        // The copy's result is promoted BECAUSE `jcr` became symbolic one step earlier.
        assert_eq!(liveness.0, vec![Val(1)]);
        // ⭐ NEITHER TRACE IS TAKEN, which is what keeps the e360 seam out of the way.
        assert_eq!(trace, String::new());
    }

    /// A model, so the program is typed; nothing here reads it.
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

    /// A decode rung, for the same reason.
    #[derive(Debug, Clone, PartialEq, Eq)]
    struct AnyRung;
    impl Workload for AnyRung {
        const ROWS: u32 = 1;
        const ACTIVE_CAP: u32 = 64;
    }

    /// 610/656 — the entry takes the module's symbols from the preamble and runs the unit against
    /// them, so the promotion e150 makes is observable before the walk reaches the unported e358.
    #[test]
    fn e610_collects_the_preamble_symbols_and_runs_every_unit() {
        let sym = Val(0);
        let mut program: Program<Dd2, AnyModel, AnyRung> = Program {
            name: ProgramName {
                group: GroupId(0),
                index: OpIndex(0),
                func: OpFunc::Add,
            },
            preamble: vec![Op::Symbol(symbol::Op::CreateSymbol {
                result: sym,
                symbol_id: 7,
                max_value: None,
            })],
            units: ProgramUnits::of(
                ProgramUnit {
                    on: Units::one(DfirUnit::L3lu, Val(30)),
                    precision: None,
                    body: vec![
                        a_copy(sym, Val(1), RegType::Jcr),
                        an_add(sym, Val(6), Val(7), RegType::Jcr, None),
                    ],
                    arch: core::marker::PhantomData,
                },
                Vec::new(),
            ),
            bound: core::marker::PhantomData,
        };
        let mut liveness = Recorder::default();
        let mut rp = StatedPressure {
            free: vec![(RegType::Jcr, 4)],
            asked: Vec::new(),
        };
        let mut estimator = StatedEstimator {
            ibuff_space: 9,
            ibuff_asks: 0,
        };
        let mut values = Values::default();
        let mut trace = String::new();

        run_on_operation(
            &mut program,
            &mut liveness,
            &mut rp,
            &mut estimator,
            &mut values,
            &mut trace,
        );

        // ⭐ THE PROMOTION IS THE EFFECT THAT PROVES THE PREAMBLE'S SYMBOL REACHED THE UNIT.
        assert_eq!(liveness.0, vec![Val(1)]);
        assert_eq!(program.units.iter().next().expect("one unit").body.len(), 2);
    }

    /// e358 — the sort puts the most-used symbol first, `N` caps the take at the free registers, the
    /// one-candidate locale still contributes, and `lccr` is asked about as `jcr`.
    #[test]
    fn e358_takes_the_most_used_symbol_and_asks_about_lccr_as_jcr() {
        let (a, b, c) = (Val(0), Val(1), Val(2));
        let body = vec![
            an_add(b, Val(10), Val(11), RegType::Lrf, None),
            an_add(a, Val(10), Val(12), RegType::Lrf, None),
            an_add(a, Val(10), Val(13), RegType::Lrf, None),
            an_add(c, Val(10), Val(14), RegType::Lccr, None),
        ];
        // ⛔ `b` IS DECLARED FIRST, so only the descending sort can put `a` ahead of it.
        let (usage, symbolic_locales) = collect_symbol_usage(&body, &[b, a, c], &[]);
        let state = PerUnitState {
            ops: collect_ops_of_interest(&body),
            symbol_to_usage: usage,
            symbolic_locales,
        };
        let mut rp = StatedPressure {
            free: vec![(RegType::Lrf, 1), (RegType::Jcr, 3)],
            asked: Vec::new(),
        };
        let mut estimator = StatedEstimator {
            ibuff_space: 5,
            ibuff_asks: 0,
        };
        let mut trace = String::new();

        let candidates = collect_candidates(
            &body,
            &[b, a, c],
            &state,
            &mut rp,
            &mut estimator,
            &mut trace,
        );

        assert_eq!(state.symbolic_locales, vec![RegType::Lrf, RegType::Lccr]);
        assert_eq!(
            candidates,
            vec![
                CandidateEntry {
                    symbol: a,
                    locale: RegType::Lrf,
                },
                CandidateEntry {
                    symbol: c,
                    locale: RegType::Lccr,
                },
            ]
        );
        // Twice per locale, and `lccr` never under its own name.
        assert_eq!(
            rp.asked,
            vec![
                (RegType::Lrf, Metric::NumFreeRegisters),
                (RegType::Jcr, Metric::NumFreeRegisters),
                (RegType::Lrf, Metric::NumFreeRegisters),
                (RegType::Jcr, Metric::NumFreeRegisters),
            ]
        );
        assert_eq!(estimator.ibuff_asks, 1);
    }

    /// e359 — two `lrf` candidates: the first copy is built inside the loop that holds the first use,
    /// carries that use's width, then hoists to the block dominating the second use.
    #[test]
    fn e359_builds_one_copy_per_candidate_and_hoists_it_over_the_second_use() {
        let (a, b) = (Val(0), Val(1));
        let mut body = vec![
            Op::Sentient(sentient::Op::For {
                iv: Val(20),
                bound: Val(21),
                bound_reg: None,
                carried: Vec::new(),
                dbg_name: None,
                body: vec![an_add(a, Val(10), Val(11), RegType::Lrf, Some(Bits(16)))],
            }),
            an_add(a, Val(10), Val(12), RegType::Lrf, None),
            an_add(b, Val(10), Val(13), RegType::Lrf, None),
        ];
        let (usage, _) = collect_symbol_usage(&body, &[a, b], &[]);
        let candidates = vec![
            CandidateEntry {
                symbol: a,
                locale: RegType::Lrf,
            },
            CandidateEntry {
                symbol: b,
                locale: RegType::Lrf,
            },
        ];
        // ⛔ MINT PAST THE SYMBOLS THEMSELVES: a copy result that collides with the symbol it copies
        // repoints nothing, and this entry's whole effect is the repointing.
        let mut values = Values::default();
        for _ in 0..40 {
            values.mint();
        }
        let mut trace = String::new();

        insert_copy_ops_for_candidates(&mut body, &candidates, &usage, &mut values, &mut trace);

        // `[copy_a, for { add(copy_a) }, add(copy_a), copy_b, add(copy_b)]`.
        assert_eq!(body.len(), 5);
        let Op::Sentient(sentient::Op::ScalarCopy {
            input,
            result: copy_a,
            reg,
            element_size,
            ..
        }) = body[0]
        else {
            panic!(
                "the hoisted copy of `a` is the first op of the unit: {:?}",
                body[0]
            );
        };
        assert_eq!(input, a);
        assert_eq!(reg.locale, RegType::Lrf);
        // e154 read the width off the FIRST use, the one inside the loop.
        assert_eq!(element_size, Some(Bits(16)));
        assert_eq!(dialects::operands(&body[2])[0], copy_a);
        let Op::Sentient(sentient::Op::For {
            body: ref inner, ..
        }) = body[1]
        else {
            panic!("the loop survives its own hoist: {:?}", body[1]);
        };
        assert_eq!(dialects::operands(&inner[0])[0], copy_a);
        let Op::Sentient(sentient::Op::ScalarCopy {
            input: input_b,
            result: copy_b,
            element_size: width_b,
            ..
        }) = body[3]
        else {
            panic!("the copy of `b` sits ahead of its one use: {:?}", body[3]);
        };
        assert_eq!(input_b, b);
        assert_eq!(width_b, None);
        assert_eq!(dialects::operands(&body[4])[0], copy_b);
    }

    /// e359 NEGATIVE — ⛔ THE TWO COUNTS ARE SEPARATE: one `jcr` candidate and one `lrf` candidate is
    /// no patch-flit opportunity in either file, so nothing is inserted at all.
    #[test]
    fn e359_refuses_one_candidate_in_each_register_file() {
        let (a, b) = (Val(0), Val(1));
        let mut body = vec![
            an_add(a, Val(10), Val(11), RegType::Jcr, None),
            an_add(b, Val(10), Val(12), RegType::Lrf, None),
        ];
        let (usage, _) = collect_symbol_usage(&body, &[a, b], &[]);
        let candidates = vec![
            CandidateEntry {
                symbol: a,
                locale: RegType::Jcr,
            },
            CandidateEntry {
                symbol: b,
                locale: RegType::Lrf,
            },
        ];
        let mut values = Values::default();
        let mut trace = String::new();

        insert_copy_ops_for_candidates(&mut body, &candidates, &usage, &mut values, &mut trace);

        assert_eq!(body.len(), 2);
        assert_eq!(dialects::operands(&body[0])[0], a);
        assert_eq!(dialects::operands(&body[1])[0], b);
    }

    /// e360 — a symbol prints its ID, a query prints its line, and a locale with nothing recorded
    /// prints no line at all.
    #[test]
    fn e360_dumps_one_line_per_symbol_and_locale_that_has_a_use() {
        let (sym, query) = (Val(0), Val(1));
        let scope = vec![
            Op::Symbol(symbol::Op::CreateSymbol {
                result: sym,
                symbol_id: 7,
                max_value: None,
            }),
            Op::Symbol(symbol::Op::QueryMap {
                result: query,
                map: Val(20),
                key: Val(21),
            }),
        ];
        let mut usage = SymbolUsage::default();
        usage.record(
            sym,
            RegType::Jcr,
            UseSite {
                path: vec![2],
                operand: 0,
            },
        );
        for path in [vec![3], vec![4]] {
            usage.record(query, RegType::Lrf, UseSite { path, operand: 0 });
        }
        let state = PerUnitState {
            ops: OpsOfInterest {
                symbol_queries: vec![query],
                ..OpsOfInterest::default()
            },
            symbol_to_usage: usage,
            symbolic_locales: vec![RegType::Jcr, RegType::Lrf],
        };

        assert_eq!(
            dump_symbol_usage_info(&state, &[sym], &scope),
            "=== Usage Info: ===\n\
             Sym ID (7) for locale [JCR] has 1 use(s).\n\
             Symbol Query On Line (?) for locale [LRF] has 2 use(s).\n\
             === End of Usage Info ===\n"
        );
    }

    /// e527 — a use is recorded under the locale of the op that reads it, and the ops a copy cannot
    /// stand in front of are skipped.
    #[test]
    fn collect_symbol_usage_records_the_users_locale_and_skips_the_copies() {
        let sym = Val(1);
        let body = vec![
            a_copy(sym, Val(5), RegType::Lrf),
            an_add(sym, Val(6), Val(7), RegType::Lbr, None),
        ];

        let (usage, locales) = collect_symbol_usage(&body, &[sym], &[]);

        // ⛔ THE COPY'S OWN USE IS NOT ONE: `lrf` never appears.
        assert_eq!(locales, vec![RegType::Lbr]);
        assert!(usage.uses(sym, RegType::Lrf).is_empty());
        assert_eq!(
            usage.uses(sym, RegType::Lbr),
            &[UseSite {
                path: vec![1],
                operand: 0,
            }]
        );
    }
}
