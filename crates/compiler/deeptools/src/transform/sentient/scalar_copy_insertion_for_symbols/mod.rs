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

// ⛔ NOTHING CALLS THIS MODULE YET — the pass entry is `e610_runOnOperation` (level 5), this file's
// last unfilled anchor, and until it lands the ported leaves below are reachable only from this
// file's own tests. ⭐ DELETE THIS LINE WHEN THAT ANCHOR IS FILLED: a warning that survives it is a
// unit nothing calls, which the campaign's own note names as the failure mode to catch.
#![allow(dead_code)]

use std::fmt::Write as _;

use crate::formats::Bits;
use crate::islands::dataflow_ir::Values;
use crate::islands::sentient::dialects::sentient::{Reg, RegType};
use crate::islands::sentient::dialects::{self as dialects, Op, Val, sentient, symbol, uniform};
use crate::islands::sentient::print;

use super::ForRef;
use super::analyses::Liveness;

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

/// `dcc::utils::isSymbol` (`Analyses/Utils.cpp:141`) — a `symbol.create_symbol`, or a
/// `uniform.query_map` any of whose mapped values is one.
///
/// ⭐ STRUCTURAL, SO NOT AN OUT-OF-SCOPE `todo!`: it reads defining ops and one mapping's values, and
/// the mapped values are all of them — `getListOfKeyOpsFromUniformMapping` answers the enclosing
/// uniformize/equalize/program-unit's whole unit list (`Dialect/Uniform/Utils.cpp:194-202`).
fn is_symbol(val: Val, defs: dialects::Definitions<'_>) -> bool {
    match defs.of(val) {
        Some(Op::Uniform(uniform::Op::QueryMap { map, .. })) => match defs.of(*map) {
            Some(Op::Uniform(uniform::Op::DefImmutableMapping { pairs, .. })) => {
                pairs.iter().any(|(_, value)| {
                    matches!(
                        defs.of(*value),
                        Some(Op::Symbol(symbol::Op::CreateSymbol { .. }))
                    )
                })
            }
            _ => false,
        },
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
/// but that array is `[bound, initArgs.., results..]` (`SentientOps.td:57-61`) so yield operand 0
/// reads the BOUND's entry; sibling `e154` adds the `+ 1` on the very same layout (`:490-492`). This
/// answers the carried value's own locale.
/// ⭐ `Unknown` IS THE TAIL — `emitError` then `DT_ERROR` then `return unknown` (`:471-473`), and the
/// caller's `DT_CHECK(locale != unknown)` is what turns it into a failure.
#[must_use]
pub fn locale_of_use(user: &Op, operand: usize, parent: Option<&Op>) -> RegType {
    match user {
        // `getValueRegLocale(for_op.getBody()->getArgument(operand))`: argument 0 is the induction
        // variable, whose `regLocales[0]` entry this island does not carry (see
        // [`dialects::value_reg_locale`]), and argument `i + 1` is `carried[i]`'s.
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

// crustify:todo: e358_collectCandidates
//   authority : dcc/src/Transform/Sentient/ScalarCopyInsertionForSymbols.cpp:297  (65 body lines, level 1)
//   original  : void ScalarCopyInsertionForSymbolsPass::collectCandidates( dataflow::ProgramUnitOp unit, CandidateListTy &candidates, RegisterPressure &rp) const
//   calls     : e252_size

// crustify:todo: e359_insertCopyOpsForCandidates
//   authority : dcc/src/Transform/Sentient/ScalarCopyInsertionForSymbols.cpp:365  (49 body lines, level 1)
//   original  : void ScalarCopyInsertionForSymbolsPass::insertCopyOpsForCandidates( dataflow::ProgramUnitOp unit, const CandidateListTy &candidates)
//   calls     : e151_insertCopyOpsForJCRCandidate, e154_propagateElementSizeToCopyOp

// crustify:todo: e360_dumpSymbolUsageInfo
//   authority : dcc/src/Transform/Sentient/ScalarCopyInsertionForSymbols.cpp:512  (32 body lines, level 1)
//   original  : void ScalarCopyInsertionForSymbolsPass::dumpSymbolUsageInfo() const
//   calls     : e252_size

// crustify:todo: e527_collectSymbolUsage
//   authority : dcc/src/Transform/Sentient/ScalarCopyInsertionForSymbols.cpp:236  (38 body lines, level 3)
//   original  : void ScalarCopyInsertionForSymbolsPass::collectSymbolUsage( dataflow::ProgramUnitOp unit)
//   calls     : e153_getLocale, e422_insert

// crustify:todo: e576_runOn
//   authority : dcc/src/Transform/Sentient/ScalarCopyInsertionForSymbols.cpp:148  (67 body lines, level 4)
//   original  : void ScalarCopyInsertionForSymbolsPass::runOn(dataflow::ProgramUnitOp unit)
//   calls     : e149_collectOpsOfInterest, e150_pessimizeLiveness, e155_dumpSymbolicLocales, e156_dumpCandidates, e252_size, e358_collectCandidates, e359_insertCopyOpsForCandidates, e360_dumpSymbolUsageInfo, e527_collectSymbolUsage

// crustify:todo: e610_runOnOperation
//   authority : dcc/src/Transform/Sentient/ScalarCopyInsertionForSymbols.cpp:130  (17 body lines, level 5)
//   original  : void ScalarCopyInsertionForSymbolsPass::runOnOperation()
//   calls     : e152_clear, e576_runOn

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::islands::dataflow_ir::ty::ScalarTy;
    use crate::islands::sentient::dialects::sentient::Carried;

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
                carried: Vec::new(),
                dbg_name: None,
                body: vec![
                    a_copy(Val(1), Val(4), RegType::Jcr),
                    Op::Sentient(sentient::Op::For {
                        iv: Val(5),
                        bound: Val(6),
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
        /// `Liveness&` recording what it is told, which is the only way to observe the selection.
        #[derive(Default)]
        struct Recorder(Vec<Val>);
        impl Liveness for Recorder {
            fn update_live_ranges_for_program_header_promotion(&mut self, candidate: Val) {
                self.0.push(candidate);
            }
        }

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
}
