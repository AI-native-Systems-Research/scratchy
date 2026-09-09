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

//! `LiveRangeReduction.cpp` — 18 of the campaign's 656 units (dependency level(s) [0, 1, 2, 3, 4, 5]).
//!
//! | unit | entry | level | lines | authority path:line |
//! |---|---|---|---|---|
//! | `e056_set` | 056 | 0 | 5 | `dcc/src/Transform/Sentient/LiveRangeReduction.cpp:118` |
//! | `e057_print` | 057 | 0 | 14 | `dcc/src/Transform/Sentient/LiveRangeReduction.cpp:141` |
//! | `e058_areElementSizeIdentical` | 058 | 0 | 9 | `dcc/src/Transform/Sentient/LiveRangeReduction.cpp:466` |
//! | `e059_getFirstGlobalOrConstantAncestor` | 059 | 0 | 56 | `dcc/src/Transform/Sentient/LiveRangeReduction.cpp:1107` |
//! | `e060_areReglocalesMatching` | 060 | 0 | 5 | `dcc/src/Transform/Sentient/LiveRangeReduction.cpp:1261` |
//! | `e304_printSsaMap` | 304 | 1 | 18 | `dcc/src/Transform/Sentient/LiveRangeReduction.cpp:218` |
//! | `e305_getBaseExpr` | 305 | 1 | 116 | `dcc/src/Transform/Sentient/LiveRangeReduction.cpp:271` |
//! | `e306_getRootIterArg` | 306 | 1 | 72 | `dcc/src/Transform/Sentient/LiveRangeReduction.cpp:393` |
//! | `e307_cloneValueToRegion` | 307 | 1 | 40 | `dcc/src/Transform/Sentient/LiveRangeReduction.cpp:917` |
//! | `e440_addToMap` | 440 | 2 | 25 | `dcc/src/Transform/Sentient/LiveRangeReduction.cpp:241` |
//! | `e441_findDominantValue` | 441 | 2 | 38 | `dcc/src/Transform/Sentient/LiveRangeReduction.cpp:480` |
//! | `e442_addResultToYield` | 442 | 2 | 140 | `dcc/src/Transform/Sentient/LiveRangeReduction.cpp:961` |
//! | `e443_createMapAndQuery` | 443 | 2 | 43 | `dcc/src/Transform/Sentient/LiveRangeReduction.cpp:1216` |
//! | `e502_reconstructOperation` | 502 | 3 | 259 | `dcc/src/Transform/Sentient/LiveRangeReduction.cpp:546` |
//! | `e503_mapAllValues` | 503 | 3 | 48 | `dcc/src/Transform/Sentient/LiveRangeReduction.cpp:827` |
//! | `e504_optimizeUniformRegionYieldedValues` | 504 | 3 | 47 | `dcc/src/Transform/Sentient/LiveRangeReduction.cpp:1167` |
//! | `e560_reduceLiveRange` | 560 | 4 | 28 | `dcc/src/Transform/Sentient/LiveRangeReduction.cpp:885` |
//! | `e598_runOnOperation` | 598 | 5 | 45 | `dcc/src/Transform/Sentient/LiveRangeReduction.cpp:1267` |


// ⛔ THE PASS IS NOT WIRED INTO THE PIPELINE YET, so every item below is reachable only from this
// file's own tests until `e598_runOnOperation` (level 5) lands and something calls it. CI runs clippy
// with `-D warnings`, so without this the first ported leaves of an 18-unit module fail the gate.
// ⭐ REMOVE THIS WITH e598: at that point an unused item here is a real defect again.
#![allow(dead_code)]

use core::fmt::Write as _;
use std::collections::BTreeMap;

use crate::islands::dataflow_ir::Values;
use crate::islands::dataflow_ir::dialects as lower;
use crate::islands::sentient::dialects::{
    Definitions, LocalRegion, Op, UniformRegions, Val, element_size, sentient, symbol, uniform,
    value_reg_locale,
};
use crate::islands::sentient::print;
// The analysis's OWN data types moved to the out-of-scope seam when e372/e373 became their second
// consumer; the pass still names them unqualified, and so do its existing citations.
pub use crate::transform::sentient::analyses::{ExprInfoMap, PropagatedMap};

/// `affine::FlatAffineValueConstraints` — THE LOCAL-VARIABLE CONSTRAINT SYSTEM OF ONE FLATTENED
/// AFFINE EXPRESSION, opaque here.
///
/// ⛔ MLIR UPSTREAM, AND NOTHING IN THIS CAMPAIGN BUILDS OR READS ONE. The only producer is
/// `mlir::getFlattenedAffineExpr` (`LiveRangeReduction.cpp:308-310`, out of campaign scope) and the only
/// readers are the system's own `isEqual` and `print`. So this is an IDENTITY, like
/// [`crate::transform::sentient::analyses::EvaluatedValue`]: which system, not the system.
///
/// ⛔ NO DERIVED `PartialEq`: `ExprInfo::operator==` compares two systems with `isEqual`, which is
/// CONTENT equality (`:135`). Comparing identities would answer a different question.
#[derive(Debug, Clone, Copy, Default)]
pub struct FlatAffineValueConstraints(pub u32);

impl FlatAffineValueConstraints {
    /// `FlatAffineValueConstraints::print` (`IntegerRelation::print`) — MLIR upstream's own dump of
    /// the constraint system, and out of campaign scope.
    ///
    /// ⛔ NOT INVENTABLE: it renders the system's equalities, inequalities and local-variable
    /// divisions, none of which this identity carries. Reached only from [`ExprInfo::print`], which is
    /// a `DEBUG_WITH_TYPE` trace.
    fn write(self, _out: &mut String) {
        todo!(
            "affine::FlatAffineValueConstraints::print (mlir/Analysis/Presburger/IntegerRelation.h) \
             — MLIR upstream, out of campaign scope"
        )
    }
}

/// `LiveRangeReductionPass::ExprInfo` (`LiveRangeReduction.cpp:116`) — ONE FLATTENED AFFINE
/// EXPRESSION: its coefficients, the SSA values they multiply, and the local-variable constraints
/// `getFlattenedAffineExpr` produced alongside them.
///
/// ⛔ NOT `PropagationAnalysis::ExprInfo` (`Analyses/PropagationAnalysis.h:38`) — a different class of
/// the same name, in the out-of-scope `Analyses/` directory, and the one
/// `crustify-senpass/OUTSIDE-DEPS.tsv` flags for this pass. This one is the pass's own nested class.
///
/// ⛔ NO DERIVED `PartialEq`: `operator==` (`:125`) compares the two constraint systems with
/// `isEqual`, which is CONTENT equality; [`FlatAffineValueConstraints`] is held by identity, so a
/// derived `==` would answer a different question.
#[derive(Debug, Clone, Default)]
pub struct ExprInfo {
    /// `expr_coeffs_` — one per dim then per local var, the constant already popped off by
    /// `getBaseExpr` (`:320-321`).
    pub expr_coeffs: Vec<i64>,
    /// `args_` — the SSA values those coefficients multiply.
    pub args: Vec<Val>,
    /// `local_vars_constraints`.
    pub local_vars_constraints: FlatAffineValueConstraints,
}

impl ExprInfo {
    /// Replaces: e056_set
    ///
    /// Overwrites all three fields — the one way `getBaseExpr` (`:342`, `:350`) fills an `ExprInfo`.
    ///
    /// ⛔ `clearAndCopyFrom` IS A REPLACEMENT, NOT A MERGE: it drops whatever system was there and
    /// takes a copy, which owned assignment performs exactly.
    pub fn set(&mut self, expr_coeffs: Vec<i64>, args: Vec<Val>, cst: FlatAffineValueConstraints) {
        self.expr_coeffs = expr_coeffs;
        self.args = args;
        self.local_vars_constraints = cst;
    }

    /// Replaces: e057_print
    ///
    /// The three-section trace `printSsaMap` (`e304`) writes per mapped value, each header indented
    /// two spaces.
    ///
    /// ⭐ EVERY COEFFICIENT IS FOLLOWED BY `", "`, INCLUDING THE LAST (`:144`), and "constrains" is
    /// the reference's own spelling (`:152`) — this is its output text, not prose.
    ///
    /// ⛔ `Value::print` PRINTS THE DEFINING OPERATION, not an SSA name, falling back to
    /// `<block argument> of type '..' at index: N`; [`print::emit`] already terminates the line that
    /// the reference's `os << "\n"` adds after `Operation::print`.
    pub fn print(&self, defs: Definitions<'_>, out: &mut String) {
        out.push_str("  Expression coeffs: ");
        for item in &self.expr_coeffs {
            let _ = write!(out, "{item}, ");
        }
        out.push_str("\n  SSA variables: \n");
        for arg in &self.args {
            match defs.of(*arg) {
                Some(op) => print::emit(out, op, 0),
                // Every `args_` element is an affine dim operand, so `index`-typed.
                None => match defs.for_arg_of(*arg) {
                    Some((_, index)) => {
                        let _ = writeln!(
                            out,
                            "<block argument> of type 'index' at index: {index}"
                        );
                    }
                    None => out.push('\n'),
                },
            }
        }
        out.push_str("  local variable constrains: \n");
        self.local_vars_constraints.write(out);
    }
}

/// Replaces: e058_areElementSizeIdentical
///
/// Whether two addresses step by the SAME, KNOWN element width — the guard `findDominantValue`
/// (`e441`) puts in front of commoning two values' live ranges.
///
/// ⛔ AN UNKNOWN WIDTH IS NOT A MATCH. The reference's `< 1` tests are what reject
/// [`element_size`]'s `-1`, so two values that BOTH have no `element_size` answer `false` rather
/// than "identical" — which is also why the four island gaps [`element_size`] documents are
/// conservative here rather than wrong.
#[must_use]
pub fn are_element_size_identical(lhs: Val, rhs: Val, defs: Definitions<'_>) -> bool {
    match (element_size(lhs, defs), element_size(rhs, defs)) {
        (Some(lhs_size), Some(rhs_size)) => {
            lhs_size.0 >= 1 && rhs_size.0 >= 1 && lhs_size == rhs_size
        }
        _ => false,
    }
}

/// Replaces: e059_getFirstGlobalOrConstantAncestor
///
/// Walks an address back through the ops that merely OFFSET it until it reaches the constant, the
/// per-core query or the symbol it is founded on, stopping at the first op outside `uniform_region`.
///
/// ⛔ `None` IS `signalPassFailure()`: an op the walk cannot attribute an address operand to makes
/// the reference `emitOpError("can't find global ancestor")` and fail the pass (`:1153-1154`).
///
/// ⛔ AND THE SRC/DST SPLIT IS THE REFERENCE'S INTENT, NOT ITS CODE. `val ==
/// load_store_op.getSrcMutableAddr()` (`:1127`) compares an op RESULT against that op's own operand
/// 2 and so is always false, leaving the `src` branch dead and every `load_and_store` address
/// walking through `dst`; the `.td` names the two results `src_res`/`dst_res` as "the final,
/// potentially updated, address" of each side (`SentientOps.td:744-746`), so this dispatches on WHICH
/// RESULT `val` is.
#[must_use]
pub fn get_first_global_or_constant_ancestor(
    val: Val,
    uniform_region: &[Op],
    defs: Definitions<'_>,
) -> Option<Val> {
    let mut val = val;
    loop {
        // ⭐ `DT_CHECK(isa<sentient::ForOp>(op))` (`:1114`) IS A FACT ABOUT THE ISLAND, not a check:
        // `for` is the only op of this dialect that binds a region argument.
        let (op, next) = match defs.for_arg_of(val) {
            Some((op @ Op::Sentient(sentient::Op::For { bound, carried, .. }), index)) => {
                // Operands are `[bound, initArgs..]` and arguments `[iv, iterArgs..]`, so
                // `getOperand(getArgNumber())` is the bound for the iv and `carried[i - 1].init`
                // for iter arg `i - 1`.
                let next = index
                    .checked_sub(1)
                    .map_or(Some(*bound), |position| {
                        carried.get(position).map(|value| value.init)
                    })?;
                (op, next)
            }
            _ => {
                let op = defs.of(val)?;
                let next = match op {
                    Op::Sentient(
                        sentient::Op::LoadAndSend { mutable_addr, .. }
                        | sentient::Op::ReceiveAndStore { mutable_addr, .. }
                        | sentient::Op::LoadAndExtractScalar { mutable_addr, .. }
                        | sentient::Op::LoadComputeAndSend { mutable_addr, .. },
                    ) => *mutable_addr,
                    Op::Sentient(sentient::Op::LoadAndStore {
                        src_mutable_addr,
                        dst_mutable_addr,
                        results,
                        ..
                    }) => {
                        if val == results.0 {
                            *src_mutable_addr
                        } else {
                            *dst_mutable_addr
                        }
                    }
                    // ⭐ FOLLOW THE OPERAND THAT IS NOT A CONSTANT, PREFERRING `lhs` (`:1141-1146`)
                    // — an address plus a constant offset is still that address.
                    Op::Sentient(
                        sentient::Op::ScalarAdd { lhs, rhs, .. }
                        | sentient::Op::ScalarSub { lhs, rhs, .. },
                    ) => {
                        if is_sentient_constant(*lhs, defs) {
                            *rhs
                        } else {
                            *lhs
                        }
                    }
                    // `getOperand(getResultNumber() + 1)` — result `i` came from `initArgs[i]`.
                    Op::Sentient(sentient::Op::For { carried, .. }) => {
                        carried
                            .iter()
                            .find(|value| value.result == val)
                            .map(|value| value.init)?
                    }
                    // ⭐ THE THREE ANCESTORS THE WALK IS LOOKING FOR — returned WITHOUT the region
                    // test, which is why a terminal op outside the region still answers (`:1151`).
                    Op::Sentient(sentient::Op::ScalarConstant { .. })
                    | Op::Uniform(uniform::Op::QueryMap { .. })
                    | Op::Symbol(symbol::Op::CreateSymbol { .. }) => return Some(val),
                    // `emitOpError("can't find global ancestor"); signalPassFailure();`
                    _ => return None,
                };
                (op, next)
            }
        };
        // `if (op->getParentRegion() != uniform_region) break;` — the IMMEDIATE parent, so
        // membership at the top level of the region's op list.
        if !uniform_region.contains(op) {
            return Some(val);
        }
        val = next;
    }
}

/// `dcc::utils::isConstant<sentient::ConstantOp>` (`Utils/Utils.cpp:423`) at the one instantiation
/// this pass uses — a `sentient.scalar_constant`, or a `uniform.query_map` every one of whose
/// per-core values is one.
fn is_sentient_constant(val: Val, defs: Definitions<'_>) -> bool {
    let is_constant_op = |op: Option<&Op>| {
        matches!(
            op,
            Some(Op::Sentient(sentient::Op::ScalarConstant { .. }))
        )
    };
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

/// Replaces: e060_areReglocalesMatching
///
/// Whether two values live in the same register file — the second half of `findDominantValue`'s
/// (`e441`) commoning guard.
///
/// ⛔ TWO UNASSIGNED VALUES MATCH. [`sentient::RegType::Unknown`] is a locale like any other to an
/// `==`, and the reference compares the enums directly; before `RegisterTypeAssignment` (D66) has
/// run that is every value in the module.
#[must_use]
pub fn are_reglocales_matching(lhs: Val, rhs: Val, defs: Definitions<'_>) -> bool {
    value_reg_locale(lhs, defs) == value_reg_locale(rhs, defs)
}

/// ONE EQUIVALENCE CLASS OF THE PASS'S SSA MAP — `expr_info_list_[i]` and `ssa_value_list_[i]`
/// (`LiveRangeReduction.cpp:104-106`) as one record.
///
/// ⭐ ONE STRUCT WHERE THE REFERENCE KEEPS TWO PARALLEL VECTORS, because `addToMap` (`e440`, `:241`)
/// only ever grows the two together — so a length disagreement between them is unwritable here.
#[derive(Debug, Clone, Default)]
pub struct EquivalenceClass {
    /// `expr_info_list_[i]` — the flattened affine expression every value of this class shares.
    pub expr_info: ExprInfo,
    /// `ssa_value_list_[i]` — the values that share it, in discovery order.
    pub values: Vec<Val>,
}

/// `LiveRangeReductionPass`'S SSA MAP — the four members `mapAllValues` (`e503`) fills and
/// `reduceLiveRange` (`e560`) consumes (`:104-108`).
#[derive(Debug, Clone, Default)]
pub struct SsaMap {
    /// `expr_info_list_` zipped with `ssa_value_list_` — see [`EquivalenceClass`].
    pub classes: Vec<EquivalenceClass>,
    /// `ssa_expr_const_map_` — one constant offset per unit, for each mapped value.
    pub const_offsets: BTreeMap<Val, Vec<i64>>,
    /// `ssa_negated_map_` — whether the value's coefficients were sign-flipped by `getBaseExpr`.
    pub negated: BTreeMap<Val, bool>,
}

impl SsaMap {
    /// Replaces: e304_printSsaMap
    ///
    /// The `DEBUG_WITH_TYPE` dump of the whole map — one section per equivalence class, one block per
    /// value in it.
    ///
    /// ⛔ EVERY CLASS'S SECTION RUNS INTO [`ExprInfo::print`], whose tail is MLIR upstream's
    /// `FlatAffineValueConstraints::print` and out of campaign scope.
    pub fn print_ssa_map(&self, defs: Definitions<'_>, out: &mut String) {
        for class in &self.classes {
            out.push_str("+++printing ExprInfo:\n");
            class.expr_info.print(defs, out);
            for value in &class.values {
                out.push_str("--\n");
                self.print_mapped_value(*value, defs, out);
            }
            out.push_str("\n+++++++++++++\n\n");
        }
    }

    /// One mapped value's block of the dump — the value itself, its per-unit constant offsets, and
    /// whether its coefficients were negated (`:225-232`).
    ///
    /// ⛔ A VALUE WITH NO ENTRY PRINTS AS EMPTY AND `false`, NOT AS MISSING: both members are read
    /// through `DenseMap::operator[]`, which DEFAULT-CONSTRUCTS on a miss.
    /// ⭐ `os.indent(2) << "\nconst offset: "` PUTS THE TWO SPACES *BEFORE* THE NEWLINE (`:227`) —
    /// the reference's own output, kept character for character.
    fn print_mapped_value(&self, value: Val, defs: Definitions<'_>, out: &mut String) {
        print_value(value, defs, out);
        out.push_str("  \nconst offset: ");
        for item in self.const_offsets.get(&value).into_iter().flatten() {
            let _ = write!(out, "{item}, ");
        }
        let negated = self.negated.get(&value).copied().unwrap_or(false);
        let _ = write!(
            out,
            "\nexpr negated: {}\n",
            if negated { "true" } else { "false" }
        );
    }
}

/// `Value::print` — the DEFINING OPERATION, or the block-argument fallback.
///
/// ⛔ [`ExprInfo::print`] INLINES THE SAME THREE LINES rather than calling this: that anchor is
/// already filled, and this port does not reach into completed work to share a helper.
fn print_value(val: Val, defs: Definitions<'_>, out: &mut String) {
    match defs.of(val) {
        Some(op) => print::emit(out, op, 0),
        None => match defs.for_arg_of(val) {
            Some((_, index)) => {
                let _ = writeln!(out, "<block argument> of type 'index' at index: {index}");
            }
            None => out.push('\n'),
        },
    }
}

/// `mlir::getFlattenedAffineExpr`'S THREE OUTPUTS for one expression (`:308-310`).
#[derive(Debug, Clone, Default)]
pub struct FlattenedExpr {
    /// `flat_expr` — one coefficient per dim, then per local var, then the constant LAST.
    pub coeffs: Vec<i64>,
    /// The local-variable system built alongside them.
    pub constraints: FlatAffineValueConstraints,
    /// `local_vars_constraints.getNumLocalVars()`.
    pub num_local_vars: usize,
}

/// `PropagationAnalysis` (`Analyses/PropagationAnalysis.h`) — THE SEAM ONTO AN ANALYSIS THIS CAMPAIGN
/// DOES NOT PORT, the same arrangement [`crate::transform::sentient::analyses`] uses for the others.
///
/// ⛔ THE TRAIT IS STILL SCOPED TO THIS FILE, BUT ITS DATA TYPES ARE NOT: [`ExprInfoMap`],
/// [`PropagatedMap`] and
/// [`PropagatedExpr`](crate::transform::sentient::analyses::PropagatedExpr) sit beside the other seams
/// now that e372/e373 read a result of this analysis without calling it. Hoist the trait when a second
/// CALLER appears;
/// `crustify-senpass/OUTSIDE-DEPS.tsv` names this analysis for several more passes.
pub trait PropagationAnalysis {
    /// `getAffineExpression(Value)` (`Analyses/PropagationAnalysis.h:308`).
    fn affine_expression(&mut self, val: Val) -> ExprInfoMap;

    /// `mlir::getFlattenedAffineExpr(map.getResult(0), map.getNumDims(), 0, ..)` (`:308-310`).
    fn flattened_affine_expr(&self, map: PropagatedMap) -> FlattenedExpr;
}

/// THE ANALYSIS THIS CAMPAIGN DOES NOT PORT — every method `todo!`s, naming it.
#[derive(Debug, Clone, Copy, Default)]
pub struct OutOfScopePropagationAnalysis;

impl PropagationAnalysis for OutOfScopePropagationAnalysis {
    fn affine_expression(&mut self, _val: Val) -> ExprInfoMap {
        todo!(
            "PropagationAnalysis::getAffineExpression \
             (dcc/src/Transform/Sentient/Analyses/PropagationAnalysis.h) — out of campaign scope"
        )
    }

    fn flattened_affine_expr(&self, _map: PropagatedMap) -> FlattenedExpr {
        todo!(
            "mlir::getFlattenedAffineExpr (mlir/Dialect/Affine/Analysis/AffineStructures.h) \
             — MLIR upstream, out of campaign scope"
        )
    }
}

/// WHAT `getBaseExpr` ANSWERS WITH — the three out-parameters it fills on `LogicalResult::success()`.
#[derive(Debug, Clone, Default)]
pub struct BaseExpr {
    /// `expr_info` — the one flattened expression every unit agreed on.
    pub expr_info: ExprInfo,
    /// `const_val` — ONE CONSTANT PER UNIT, `i64::MAX` where no expression applies.
    pub const_val: Vec<i64>,
    /// `negated` — whether the agreed coefficients were sign-flipped to make the first non-zero one
    /// positive.
    pub negated: bool,
}

/// Replaces: e305_getBaseExpr
///
/// One value's base expression: flatten every unit's propagated map, insist they all agree, and hand
/// back the per-unit constant offsets they differ by.
///
/// ⛔ `None` IS `LogicalResult::failure()` — no unit resolved, or two of them disagreed.
/// ⛔ `i64::MAX` IS THE "NO EXPRESSION FOR THIS UNIT" SENTINEL, written by three separate paths, then
/// smoothed away for a `uniform.uniformize_regions` result because those live in the global region.
/// ⭐ THE `args.size() > dim_num` TRIM IS THE REFERENCE'S OWN WORKAROUND for a bug in expression
/// propagation (`:333-336`) — kept, comment and all.
#[must_use]
pub fn get_base_expr(
    analysis: &mut impl PropagationAnalysis,
    value: Val,
    defs: Definitions<'_>,
) -> Option<BaseExpr> {
    let expr_info_map = analysis.affine_expression(value);
    let is_constant = is_sentient_constant(value, defs);
    // `isa<OpResult>(value)`, whose complement is `isa<BlockArgument>(value)`.
    let is_op_result = defs.of(value).is_some();
    let mut expr_info = ExprInfo::default();
    let mut negated = false;
    let mut is_expr_info_known = false;
    let mut num_irresolvable = 0usize;
    let mut const_val_tmp: Vec<i64> = Vec::new();
    // Each bucket here may correspond to one or multiple units.
    for expr in &expr_info_map.exprs {
        // if no expression, use INT64_MAX to indicate no applicable expression for this unit
        let Some(expr) = expr else {
            const_val_tmp.push(i64::MAX);
            continue;
        };
        // for constant values, const_val needs to be filled.
        if expr.cannot_be_resolved && !is_constant {
            num_irresolvable += 1;
            // if expressions for all cores are irresolvable, return failure
            // if value is blockArgument, use itself as baseExpr
            if num_irresolvable == expr_info_map.exprs.len() && is_op_result {
                return None;
            }
            const_val_tmp.push(i64::MAX);
            continue;
        }

        let flat = analysis.flattened_affine_expr(expr.propagated_map);
        let mut coeffs = flat.coeffs;
        let dim_num = expr.propagated_map.num_dims;
        let mut args = expr.propagated_args.clone();
        // TODO there is a bug in expression propagation pass which incorrectly adds
        // additional value to args_. a temporary fix to remove the first entry.
        if args.len() > dim_num {
            args.remove(0);
        }
        // ⛔ `DT_CHECK(flat_expr.size() == dim_num + getNumLocalVars() + 1)` (`:337-338`).
        if coeffs.len() != dim_num + flat.num_local_vars + 1 {
            panic!(
                "a flattened affine expression has {} coefficients, not one per dimension, one per \
                 local variable and one constant (LiveRangeReduction.cpp:337-338)",
                coeffs.len()
            );
        }

        // the const expr is at the back.
        const_val_tmp.push(coeffs.pop().unwrap_or(i64::MAX));
        // normalize flat_expr to make the first non-zero value positive.
        let temp_negated = coeffs
            .iter()
            .find(|coeff| **coeff != 0)
            .is_some_and(|coeff| *coeff < 0);
        if temp_negated {
            for coeff in &mut coeffs {
                // ⭐ `-x` ON `i64::MIN` IS THE REFERENCE'S OWN TWO'S-COMPLEMENT WRAP.
                *coeff = coeff.wrapping_neg();
            }
        }
        if is_expr_info_known {
            // compare the new expr with the existing one
            let mut new_expr_info = ExprInfo::default();
            new_expr_info.set(coeffs, args, flat.constraints);
            // If mismatching expr_info (e.g. due to different coefficients), return failure.
            if !expr_info_eq(&new_expr_info, &expr_info) || negated != temp_negated {
                return None;
            }
        } else {
            // set expr_info for the first time
            expr_info.set(coeffs, args, flat.constraints);
            negated = temp_negated;
            is_expr_info_known = true;
        }
    }

    // At this point const_val_tmp contains one constant per bucket. However, we need one per unit.
    // ⭐ A BUCKET INDEX PAST THE LIST TAKES THE SENTINEL — the same "no expression for this unit"
    // answer the null bucket gives, and only reachable if the analysis disagreed with itself.
    let mut const_val: Vec<i64> = expr_info_map
        .buckets
        .iter()
        .map(|bucket| const_val_tmp.get(*bucket).copied().unwrap_or(i64::MAX))
        .collect();

    // if iter_arg is not resolvable for all units, copy the last entry of const_val to all others
    if num_irresolvable == expr_info_map.exprs.len()
        && expr_info_map.exprs.len() > 1
        && !is_op_result
    {
        if let Some(last) = const_val.last().copied() {
            // ⭐ SATURATING WHERE `const_val.size() - 1` WOULD WRAP on an empty vector (`:388`).
            let count = const_val.len().saturating_sub(1);
            for item in const_val.iter_mut().take(count) {
                *item = last;
            }
        }
    }

    // uniformize_regionsOp's results are in global region, so it is necessary to replace INT64_MAX
    // entries with any valid value in the vector to improve the uniformity across cores.
    if is_op_result && defs.of(value).is_some_and(is_uniformize_regions) {
        let mut tmp = i64::MAX;
        for item in &const_val {
            if *item != i64::MAX {
                tmp = *item;
            }
        }
        for item in &mut const_val {
            if *item == i64::MAX {
                *item = tmp;
            }
        }
    }

    Some(BaseExpr {
        expr_info,
        const_val,
        negated,
    })
}

/// `ExprInfo::operator==` (`:125`) — coefficients, then args, then the two constraint SYSTEMS.
///
/// ⛔ NOT A DERIVED `PartialEq`, AND THE ORDER IS THE REFERENCE'S: the element-wise tests answer
/// first, so the Presburger `isEqual` that [`FlatAffineValueConstraints`] cannot express is reached
/// only when every coefficient and every arg already matched.
/// ⭐ TWO IDENTICAL SYSTEMS ANSWER `true` WITHOUT IT — and that is the only case `getBaseExpr`
/// produces, since one `getFlattenedAffineExpr` call feeds both sides of the comparison.
fn expr_info_eq(lhs: &ExprInfo, rhs: &ExprInfo) -> bool {
    if lhs.expr_coeffs != rhs.expr_coeffs || lhs.args != rhs.args {
        return false;
    }
    if lhs.local_vars_constraints.0 == rhs.local_vars_constraints.0 {
        return true;
    }
    todo!(
        "affine::FlatAffineValueConstraints::isEqual (mlir/Analysis/Presburger/IntegerRelation.h) \
         — MLIR upstream, out of campaign scope"
    )
}

/// Replaces: e306_getRootIterArg
///
/// Walks an address back through every op that merely FORWARDS it — adds, subs, loop carries,
/// transfers, local regions — to the value it is rooted in.
///
/// ⛔ ONLY THE ADDRESS RESULT OF A `load_and_extract_scalar` CHAINS: the data result represents what
/// is STORED at that address and maps to nothing in the IR, so it ends the walk (`:456-461`).
/// ⭐ IT ALWAYS ANSWERS WITH A VALUE, NEVER `None`: an op with no forwarding arm ends the walk at the
/// result it was asked about, which is the reference's `break`.
#[must_use]
pub fn get_root_iter_arg(val: Val, defs: Definitions<'_>) -> Val {
    let mut val = val;
    // `while (val && !isa<BlockArgument>(val))` — a block argument has no defining op.
    while let Some(op) = defs.of(val) {
        let next = match op {
            // Follow the operand that is not a constant, preferring `$inp1` (`:422-434`).
            Op::Sentient(
                sentient::Op::ScalarAdd { lhs, rhs, .. } | sentient::Op::ScalarSub { lhs, rhs, .. },
            ) => {
                if defs.of(*lhs).is_none() || !is_sentient_constant(*lhs, defs) {
                    *lhs
                } else {
                    *rhs
                }
            }
            // `getOperand(getResultNumber() + getNumControlOperands())` — the one control operand is
            // `$bound`, so result `i` came from `initArgs[i]`.
            Op::Sentient(sentient::Op::For { carried, .. }) => {
                match carried.iter().find(|value| value.result == val) {
                    Some(value) => value.init,
                    None => break,
                }
            }
            Op::Sentient(
                sentient::Op::LoadAndSend { mutable_addr, .. }
                | sentient::Op::ReceiveAndStore { mutable_addr, .. }
                | sentient::Op::LoadComputeAndSend { mutable_addr, .. },
            ) => *mutable_addr,
            // `result_idx > 0 ? getDstMutableAddr() : getSrcMutableAddr()` — `results.0` is `$src_res`.
            Op::Sentient(sentient::Op::LoadAndStore {
                src_mutable_addr,
                dst_mutable_addr,
                results,
                ..
            }) => {
                if val == results.0 {
                    *src_mutable_addr
                } else {
                    *dst_mutable_addr
                }
            }
            Op::Sentient(sentient::Op::LoadAndExtractScalar {
                mutable_addr,
                addr_result,
                ..
            }) => {
                if val == *addr_result {
                    *mutable_addr
                } else {
                    break;
                }
            }
            local if is_uniformize_regions(local) => match uniformize_result_source(local, val) {
                Some(source) => source,
                None => break,
            },
            _ => break,
        };
        val = next;
    }
    val
}

/// `dyn_cast<uniform::UniformizeRegionsOp>(op)` IN BOTH ISLAND SPELLINGS — the same op whether its
/// regions hold this rung's ops or the rung below's, which is how `SimplifyUniformRegions` reads it.
fn is_uniformize_regions(op: &Op) -> bool {
    matches!(
        op,
        Op::Uniform(uniform::Op::UniformizeRegions { .. })
            | Op::UniformRegions(UniformRegions::UniformizeRegions { .. })
    )
}

/// `uniformize_op->getRegion(0).front().getTerminator()->getOperand(result_idx)` — what region 0
/// yields for the result `val` (`:467-478`), in both island spellings.
///
/// ⛔ THE TWO `DT_CHECK`s ARE A SHAPE, NOT A CHOICE: more than one region means EXACTLY two, the
/// second holding nothing but its terminator (`:470-476`).
fn uniformize_result_source(op: &Op, val: Val) -> Option<Val> {
    let (results, yielded, region_count, second_len) = match op {
        Op::UniformRegions(UniformRegions::UniformizeRegions { regions, results }) => (
            results.iter().map(|result| result.val).collect::<Vec<Val>>(),
            regions.first().and_then(|region| match region.body.last() {
                Some(Op::Uniform(uniform::Op::Yield { operands })) => Some(operands.as_slice()),
                _ => None,
            }),
            regions.len(),
            regions.get(1).map(|region| region.body.len()),
        ),
        Op::Uniform(uniform::Op::UniformizeRegions { regions, results }) => (
            results.clone(),
            regions.first().and_then(|region| match region.body.last() {
                Some(lower::Op::Uniform(uniform::Op::Yield { operands })) => {
                    Some(operands.as_slice())
                }
                _ => None,
            }),
            regions.len(),
            regions.get(1).map(|region| region.body.len()),
        ),
        _ => return None,
    };
    if region_count > 1 {
        if region_count != 2 {
            panic!(
                "a uniform.uniformize_regions has {region_count} regions, not one or two \
                 (LiveRangeReduction.cpp:470)"
            );
        }
        if second_len != Some(1) {
            panic!(
                "a uniform.uniformize_regions' second region holds more than its terminator \
                 (LiveRangeReduction.cpp:471-476)"
            );
        }
    }
    let at = results.iter().position(|result| *result == val)?;
    yielded?.get(at).copied()
}

/// THE SECOND REGION OF A `uniform.uniformize_regions`, WITH THE REGIONS BEFORE IT — what
/// `cloneValueToRegion` is handed.
///
/// ⛔⛔ `DT_CHECK(region.getRegionNumber() == 1)` (`:920`) IS THIS TYPE, not a check: only
/// [`Self::of`] can build one, and it answers `None` for an op that has no second region.
/// ⭐ [`Self::preceding`] IS THE OTHER HALF OF THE REFERENCE'S `isProperAncestor` TEST: a value
/// defined in an ENCLOSING region needs no clone, and a value defined in a SIBLING region is exactly
/// one found in here.
pub struct SecondRegion<'a> {
    /// Region 1 — where the clone goes.
    pub region: &'a mut LocalRegion,
    /// Regions `0..1` — where a sibling region's definition is found.
    pub preceding: &'a [LocalRegion],
}

impl<'a> SecondRegion<'a> {
    /// Region 1 of `op`, or `None` when it has none.
    pub fn of(op: &'a mut UniformRegions) -> Option<SecondRegion<'a>> {
        let regions = op.regions_mut();
        if regions.len() < 2 {
            return None;
        }
        let (preceding, rest) = regions.split_at_mut(1);
        Some(SecondRegion {
            region: rest.first_mut()?,
            preceding,
        })
    }
}

/// Replaces: e307_cloneValueToRegion
///
/// Makes a value defined in region 0 usable from region 1 by re-minting, INSIDE region 1, the
/// `uniform.query_map` behind it over a mapping of region 1's own units.
///
/// ⛔ `None` MEANS THE MAPPING DOES NOT COVER REGION 1'S UNITS, and the caller (`e442`) then erases the
/// op it was building and keeps the original (`LiveRangeReduction.cpp:1085-1088`).
/// ⛔ THE `!query_op->isProperAncestor(region.getParentOp())` GUARD IS DEAD CODE (`:929`): a
/// `uniform.query_map` carries no regions, so it can never be an ancestor of anything and the mapping
/// is ALWAYS re-minted. Ported as the reference behaves, not as it reads.
pub fn clone_value_to_region(val: Val, target: SecondRegion<'_>, vals: &mut Values) -> Option<Val> {
    let SecondRegion { region, preceding } = target;
    // `isa<OpResult>(val) && !getParentRegion()->isProperAncestor(&region)` — a sibling region's op.
    let bodies: Vec<&[Op]> = preceding
        .iter()
        .map(|earlier| earlier.body.as_slice())
        .collect();
    let defs = Definitions::from_innermost(&bodies);
    let Some(op) = defs.of(val) else {
        return Some(val);
    };
    // ⛔ `DT_CHECK(isConstant<sentient::ConstantOp>(val))` (`:923`).
    if !is_sentient_constant(val, defs) {
        panic!(
            "a non-constant value defined in a sibling local region cannot be cloned into region 1 \
             (LiveRangeReduction.cpp:923)"
        );
    }
    let Op::Uniform(uniform::Op::QueryMap { map, .. }) = op else {
        return Some(val);
    };
    let Some(Op::Uniform(uniform::Op::DefImmutableMapping { pairs, .. })) = defs.of(*map) else {
        panic!(
            "a uniform.query_map's $map is not defined by a uniform.def_immutable_mapping \
             (LiveRangeReduction.cpp:925-926)"
        );
    };
    // check if immutable map contains values for the region's units — `collectUnitVals(
    // getRegionUnitList(1))`, then `getNonNullValuesFromKeys`, whose MISSES ARE DROPPED.
    let keys = region.units.clone();
    let values: Vec<Val> = keys
        .iter()
        .filter_map(|sought| {
            pairs
                .iter()
                .find(|(mapped, _)| mapped == sought)
                .map(|(_, value)| *value)
        })
        .collect();
    if values.len() != keys.len() {
        // if val is a queryOp which doesn't cover region's units, quit adding
        return None;
    }
    let mapping = vals.mint();
    let result = vals.mint();
    // ⭐ THE BUILDER'S INSERTION POINT IS INSIDE THE REGION, so before its `uniform.yield`.
    let at = region.body.len().saturating_sub(1);
    region.body.insert(
        at,
        Op::Uniform(uniform::Op::DefImmutableMapping {
            result: mapping,
            pairs: keys.into_iter().zip(values).collect(),
        }),
    );
    region.body.insert(
        at + 1,
        Op::Uniform(uniform::Op::QueryMap {
            result,
            map: mapping,
            key: region.arg,
        }),
    );
    Some(result)
}

// crustify:todo: e440_addToMap
//   authority : dcc/src/Transform/Sentient/LiveRangeReduction.cpp:241  (25 body lines, level 2)
//   original  : void LiveRangeReductionPass::addToMap(PropagationAnalysis& expr_prop_analysis, Value& value)
//   calls     : e252_size, e305_getBaseExpr

// crustify:todo: e441_findDominantValue
//   authority : dcc/src/Transform/Sentient/LiveRangeReduction.cpp:480  (38 body lines, level 2)
//   original  : LogicalResult LiveRangeReductionPass::findDominantValue( SmallVector<Value> ssa_list, const int current_value_index, int& dominant_value_index, const DominanceInfo& dominance_info)
//   calls     : e058_areElementSizeIdentical, e060_areReglocalesMatching, e306_getRootIterArg

// crustify:todo: e442_addResultToYield
//   authority : dcc/src/Transform/Sentient/LiveRangeReduction.cpp:961  (140 body lines, level 2)
//   original  : uniform::UniformizeRegionsOp LiveRangeReductionPass::addResultToYield( uniform::UniformizeRegionsOp uniformize_op, SmallVector<Operation*>& ops_to_be_delected, int region_num, int op_num, int result_idx)
//   calls     : e059_getFirstGlobalOrConstantAncestor, e239_getOperationOfBlock, e252_size, e307_cloneValueToRegion, e392_getQueryKeyAndUnitsFromParentRegion

// crustify:todo: e443_createMapAndQuery
//   authority : dcc/src/Transform/Sentient/LiveRangeReduction.cpp:1216  (43 body lines, level 2)
//   original  : Operation* LiveRangeReductionPass::createMapAndQuery( OpBuilder& builder, std::vector<int64_t>& new_const_val, Operation* op)
//   calls     : e252_size, e392_getQueryKeyAndUnitsFromParentRegion

// crustify:todo: e502_reconstructOperation
//   authority : dcc/src/Transform/Sentient/LiveRangeReduction.cpp:546  (259 body lines, level 3)
//   original  : LogicalResult LiveRangeReductionPass::reconstructOperation( SmallVector<Value>& ssa_list, int current_value_index, int dominant_value_index)
//   calls     : e057_print, e252_size, e443_createMapAndQuery

// crustify:todo: e503_mapAllValues
//   authority : dcc/src/Transform/Sentient/LiveRangeReduction.cpp:827  (48 body lines, level 3)
//   original  : void LiveRangeReductionPass::mapAllValues(PropagationAnalysis& expr_prop, Operation* unit_op)
//   calls     : e440_addToMap

// crustify:todo: e504_optimizeUniformRegionYieldedValues
//   authority : dcc/src/Transform/Sentient/LiveRangeReduction.cpp:1167  (47 body lines, level 3)
//   original  : void LiveRangeReductionPass::optimizeUniformRegionYieldedValues( ModuleOp module_op)
//   calls     : e239_getOperationOfBlock, e252_size, e442_addResultToYield

// crustify:todo: e560_reduceLiveRange
//   authority : dcc/src/Transform/Sentient/LiveRangeReduction.cpp:885  (28 body lines, level 4)
//   original  : void LiveRangeReductionPass::reduceLiveRange(PropagationAnalysis& expr_prop, Operation* unit_op)
//   calls     : e252_size, e441_findDominantValue, e502_reconstructOperation

// crustify:todo: e598_runOnOperation
//   authority : dcc/src/Transform/Sentient/LiveRangeReduction.cpp:1267  (45 body lines, level 5)
//   original  : void LiveRangeReductionPass::runOnOperation()
//   calls     : e304_printSsaMap, e503_mapAllValues, e504_optimizeUniformRegionYieldedValues, e560_reduceLiveRange


#[cfg(test)]
mod unit_tests {
    use crate::transform::sentient::analyses::PropagatedExpr;
    use super::*;
    use crate::arch::Elements;
    use crate::formats::Bits;
    use crate::islands::dataflow_ir::link::SendEnd;
    use crate::islands::dataflow_ir::ty::ScalarTy;
    use crate::islands::sentient::dialects::sentient::{Extent, Reg, RegType};

    /// A `sentient.load_and_extract_scalar` reading `addr`, `size` bits at a time, into `locale`.
    fn extract(addr: Val, results: (Val, Val), size: u32, locale: RegType) -> Op {
        Op::Sentient(sentient::Op::LoadAndExtractScalar {
            mutable_addr: addr,
            immutable_addr: Val(0),
            increment: Val(0),
            consumer: SendEnd::to_self(Val(0)),
            addr_result: results.0,
            data_result: results.1,
            total_elements: Elements(8),
            element_size: Bits(size),
            addr_reg: Reg {
                locale,
                index: None,
            },
            data_reg: Reg {
                locale: RegType::Unknown,
                index: None,
            },
            dbg_name: None,
        })
    }

    fn constant(result: Val) -> Op {
        Op::Sentient(sentient::Op::ScalarConstant {
            value: 64,
            result,
            reg_locale: RegType::Imm,
            ty: ScalarTy::Index,
            is_symbol: false,
        })
    }

    fn add(lhs: Val, rhs: Val, result: Val) -> Op {
        Op::Sentient(sentient::Op::ScalarAdd {
            lhs,
            rhs,
            result,
            reg: None,
            ty: ScalarTy::Index,
            element_size: None,
        })
    }

    /// e056 — all three fields are replaced, not merged.
    #[test]
    fn set_replaces_every_field_including_the_constraint_system() {
        let mut info = ExprInfo {
            expr_coeffs: vec![9, 9, 9],
            args: vec![Val(1), Val(2), Val(3)],
            local_vars_constraints: FlatAffineValueConstraints(1),
        };
        info.set(vec![1, 2], vec![Val(7)], FlatAffineValueConstraints(2));
        assert_eq!(info.expr_coeffs, vec![1, 2]);
        assert_eq!(info.args, vec![Val(7)]);
        assert_eq!(info.local_vars_constraints.0, 2);
    }

    /// e058 — equal known widths match; an op with no `element_size` (the reference's `-1`) does not.
    #[test]
    fn element_sizes_match_only_when_both_are_known_and_equal() {
        let ops = vec![
            extract(Val(1), (Val(2), Val(3)), 16, RegType::Lar),
            extract(Val(4), (Val(5), Val(6)), 16, RegType::Lar),
            extract(Val(7), (Val(8), Val(9)), 8, RegType::Lbr),
            constant(Val(10)),
        ];
        let regions: [&[Op]; 1] = [&ops];
        let defs = Definitions::from_innermost(&regions);

        assert!(are_element_size_identical(Val(2), Val(5), defs));
        assert!(!are_element_size_identical(Val(2), Val(8), defs));
        assert!(!are_element_size_identical(Val(2), Val(10), defs));
    }

    /// e060 — the same register file matches, two different ones do not.
    #[test]
    fn reglocales_match_per_register_file() {
        let ops = vec![
            extract(Val(1), (Val(2), Val(3)), 16, RegType::Lar),
            extract(Val(4), (Val(5), Val(6)), 16, RegType::Lar),
            extract(Val(7), (Val(8), Val(9)), 16, RegType::Lbr),
        ];
        let regions: [&[Op]; 1] = [&ops];
        let defs = Definitions::from_innermost(&regions);

        assert!(are_reglocales_matching(Val(2), Val(5), defs));
        assert!(!are_reglocales_matching(Val(2), Val(8), defs));
    }

    /// e059 — the walk crosses the offsetting adds and stops on the `uniform.query_map`, taking the
    /// operand that is NOT a constant.
    #[test]
    fn ancestor_walk_skips_constant_offsets_and_stops_at_the_query_map() {
        let ops = vec![
            Op::Uniform(uniform::Op::DefImmutableMapping {
                result: Val(1),
                pairs: vec![(Val(30), Val(31))],
            }),
            Op::Uniform(uniform::Op::QueryMap {
                result: Val(2),
                map: Val(1),
                key: Val(0),
            }),
            constant(Val(3)),
            // `%4 = %2 + %3` — the constant is the rhs, so the walk follows the lhs.
            add(Val(2), Val(3), Val(4)),
            // `%5 = %3 + %4` — the constant is the LHS, so the walk follows the rhs.
            add(Val(3), Val(4), Val(5)),
            extract(Val(5), (Val(6), Val(7)), 16, RegType::Lar),
        ];
        let regions: [&[Op]; 1] = [&ops];
        let defs = Definitions::from_innermost(&regions);

        assert_eq!(
            get_first_global_or_constant_ancestor(Val(6), &ops, defs),
            Some(Val(2))
        );
    }

    /// e059 — a value defined by an op with no address operand is the `emitOpError` that fails the
    /// pass, and a `load_and_store` result walks the side it came out of.
    #[test]
    fn ancestor_walk_splits_load_and_store_by_result_and_refuses_the_rest() {
        let ops = vec![
            constant(Val(1)),
            constant(Val(2)),
            Op::Sentient(sentient::Op::ScalarCopy {
                input: Val(1),
                result: Val(3),
                reg: Reg {
                    locale: RegType::Lar,
                    index: None,
                },
                program_header: false,
                element_size: None,
            }),
        ];
        let mut ops = ops;
        ops.push(Op::Sentient(sentient::Op::LoadAndStore {
            src: Val(20),
            dst: Val(21),
            src_mutable_addr: Val(1),
            src_immutable_addr: Val(0),
            src_inc: Val(0),
            dst_mutable_addr: Val(2),
            dst_immutable_addr: Val(0),
            dst_inc: Val(0),
            multicast_info: None,
            results: (Val(4), Val(5)),
            extent: Extent::of(Elements(8), Bits(16)),
            stride: 1,
            rotate_val: None,
            shuffle_mode: sentient::ShuffleMode::NoShuffle,
            src_reg: Reg {
                locale: RegType::Lar,
                index: None,
            },
            dst_reg: Reg {
                locale: RegType::Lbr,
                index: None,
            },
            dir: None,
            is_ibr_write: false,
            dbg_name: None,
        }));
        let regions: [&[Op]; 1] = [&ops];
        let defs = Definitions::from_innermost(&regions);

        // `sentient.scalar_copy` has no arm — `emitOpError("can't find global ancestor")`.
        assert_eq!(get_first_global_or_constant_ancestor(Val(3), &ops, defs), None);
        // A constant IS the ancestor, and answers before any region test.
        assert_eq!(
            get_first_global_or_constant_ancestor(Val(1), &ops, defs),
            Some(Val(1))
        );
        // ⭐ EACH `load_and_store` RESULT WALKS ITS OWN SIDE — the divergence this port's doc names.
        assert_eq!(
            get_first_global_or_constant_ancestor(Val(4), &ops, defs),
            Some(Val(1))
        );
        assert_eq!(
            get_first_global_or_constant_ancestor(Val(5), &ops, defs),
            Some(Val(2))
        );
    }
    /// A `PropagationAnalysis` that answers with exactly what the fixture put in it.
    struct Canned {
        map: ExprInfoMap,
        flats: BTreeMap<u32, FlattenedExpr>,
    }

    impl PropagationAnalysis for Canned {
        fn affine_expression(&mut self, _val: Val) -> ExprInfoMap {
            self.map.clone()
        }

        fn flattened_affine_expr(&self, map: PropagatedMap) -> FlattenedExpr {
            self.flats[&map.id].clone()
        }
    }

    /// One resolved bucket over `num_dims` dimensions.
    fn bucket(id: u32, num_dims: usize, args: Vec<Val>) -> Option<PropagatedExpr> {
        Some(PropagatedExpr {
            propagated_map: PropagatedMap { id, num_dims },
            propagated_args: args,
            cannot_be_resolved: false,
        })
    }

    /// e304 — a mapped value's block, and the two spaces the reference puts before the newline.
    #[test]
    fn print_ssa_map_dumps_the_const_offsets_and_the_negated_flag_per_value() {
        let ops = vec![constant(Val(1)), constant(Val(2))];
        let regions: [&[Op]; 1] = [&ops];
        let defs = Definitions::from_innermost(&regions);
        let map = SsaMap {
            classes: vec![EquivalenceClass {
                expr_info: ExprInfo::default(),
                values: vec![Val(1), Val(2)],
            }],
            const_offsets: BTreeMap::from([(Val(1), vec![4, 8])]),
            negated: BTreeMap::from([(Val(1), true)]),
        };

        let mut out = String::new();
        map.print_mapped_value(Val(1), defs, &mut out);
        assert!(
            out.ends_with("  \nconst offset: 4, 8, \nexpr negated: true\n"),
            "{out}"
        );
        // ⛔ A VALUE WITH NO ENTRY IS EMPTY AND `false`, not missing.
        let mut out = String::new();
        map.print_mapped_value(Val(2), defs, &mut out);
        assert!(
            out.ends_with("  \nconst offset: \nexpr negated: false\n"),
            "{out}"
        );
        // No classes, no output at all.
        let mut out = String::new();
        SsaMap::default().print_ssa_map(defs, &mut out);
        assert!(out.is_empty());
    }

    /// e305 — the sign normalisation, the popped constant, the null bucket's sentinel, and the
    /// failure two disagreeing buckets produce.
    #[test]
    fn get_base_expr_normalizes_the_sign_and_refuses_disagreeing_units() {
        let ops = vec![constant(Val(1))];
        let regions: [&[Op]; 1] = [&ops];
        let defs = Definitions::from_innermost(&regions);

        let mut agreeing = Canned {
            // Unit 0 reads bucket 0; unit 1 reads the null bucket 1.
            map: ExprInfoMap {
                exprs: vec![bucket(0, 1, vec![Val(7)]), None],
                buckets: vec![0, 1],
            },
            flats: BTreeMap::from([(
                0,
                FlattenedExpr {
                    coeffs: vec![-2, 5],
                    constraints: FlatAffineValueConstraints(3),
                    num_local_vars: 0,
                },
            )]),
        };
        let base = get_base_expr(&mut agreeing, Val(1), defs).unwrap();
        // ⭐ THE CONSTANT IS POPPED OFF THE BACK, and the remaining coefficient is sign-flipped.
        assert_eq!(base.expr_info.expr_coeffs, vec![2]);
        assert!(base.negated);
        assert_eq!(base.expr_info.args, vec![Val(7)]);
        assert_eq!(base.const_val, vec![5, i64::MAX]);

        // Two resolved buckets whose coefficients differ — `LogicalResult::failure()`.
        let mut disagreeing = Canned {
            map: ExprInfoMap {
                exprs: vec![bucket(0, 1, vec![Val(7)]), bucket(1, 1, vec![Val(7)])],
                buckets: vec![0, 1],
            },
            flats: BTreeMap::from([
                (
                    0,
                    FlattenedExpr {
                        coeffs: vec![2, 5],
                        constraints: FlatAffineValueConstraints(3),
                        num_local_vars: 0,
                    },
                ),
                (
                    1,
                    FlattenedExpr {
                        coeffs: vec![3, 5],
                        constraints: FlatAffineValueConstraints(3),
                        num_local_vars: 0,
                    },
                ),
            ]),
        };
        assert!(get_base_expr(&mut disagreeing, Val(1), defs).is_none());
    }

    /// e306 — the walk crosses the constant offset, the loop carry and the address result, and stops
    /// dead on the data result.
    #[test]
    fn root_iter_arg_chains_addresses_and_stops_on_a_data_result() {
        let ops = vec![
            constant(Val(1)),
            // `%3 = %1 + %2` — `$inp1` is the constant, so the walk follows `$inp2`, a block argument.
            add(Val(1), Val(2), Val(3)),
            extract(Val(3), (Val(4), Val(5)), 16, RegType::Lar),
            Op::Sentient(sentient::Op::For {
                iv_reg: sentient::Reg::UNALLOCATED,
                iv: Val(60),
                bound: Val(1),
                carried: vec![sentient::Carried {
                    result_reg: sentient::Reg::UNALLOCATED,
                    init: Val(4),
                    arg: Val(61),
                    result: Val(62),
                    reg: Reg {
                        locale: RegType::Unknown,
                        index: None,
                    },
                    program_header: false,
                    element_size: None,
                }],
                dbg_name: None,
                body: Vec::new(),
            }),
        ];
        let regions: [&[Op]; 1] = [&ops];
        let defs = Definitions::from_innermost(&regions);

        assert_eq!(get_root_iter_arg(Val(4), defs), Val(2));
        // ⛔ THE DATA RESULT CHAINS NOWHERE.
        assert_eq!(get_root_iter_arg(Val(5), defs), Val(5));
        // A loop result walks back to the `initArgs` entry it came from, then on down the chain.
        assert_eq!(get_root_iter_arg(Val(62), defs), Val(2));
    }

    /// e307 — the query is re-minted inside region 1 over region 1's own units, and a mapping that
    /// misses one of them refuses.
    #[test]
    fn clone_value_to_region_remints_the_query_over_the_regions_own_units() {
        let mut vals = Values::default();
        let unit_a = vals.mint();
        let unit_b = vals.mint();
        let val_a = vals.mint();
        let val_b = vals.mint();
        let mapping = vals.mint();
        let query = vals.mint();
        let r0_arg = vals.mint();
        let r1_arg = vals.mint();

        let covering = |pairs: Vec<(Val, Val)>| {
            UniformRegions::UniformizeRegions {
                regions: vec![
                    LocalRegion {
                        arg: r0_arg,
                        units: vec![unit_a, unit_b],
                        body: vec![
                            constant(val_a),
                            constant(val_b),
                            Op::Uniform(uniform::Op::DefImmutableMapping {
                                result: mapping,
                                pairs,
                            }),
                            Op::Uniform(uniform::Op::QueryMap {
                                result: query,
                                map: mapping,
                                key: r0_arg,
                            }),
                            Op::Uniform(uniform::Op::Yield {
                                operands: Vec::new(),
                            }),
                        ],
                    },
                    LocalRegion {
                        arg: r1_arg,
                        units: vec![unit_a, unit_b],
                        body: vec![Op::Uniform(uniform::Op::Yield {
                            operands: Vec::new(),
                        })],
                    },
                ],
                results: Vec::new(),
            }
        };

        let mut op = covering(vec![(unit_a, val_a), (unit_b, val_b)]);
        let cloned =
            clone_value_to_region(query, SecondRegion::of(&mut op).unwrap(), &mut vals).unwrap();
        let region1 = &op.regions()[1];
        assert_eq!(region1.body.len(), 3);
        assert!(matches!(
            &region1.body[0],
            Op::Uniform(uniform::Op::DefImmutableMapping { pairs, .. })
                if *pairs == vec![(unit_a, val_a), (unit_b, val_b)]
        ));
        // ⭐ THE NEW QUERY READS REGION 1'S OWN ARGUMENT, which is what makes it usable there.
        assert!(matches!(
            &region1.body[1],
            Op::Uniform(uniform::Op::QueryMap { result, key, .. })
                if *result == cloned && *key == r1_arg
        ));

        // ⛔ A MAPPING THAT COVERS ONLY ONE OF THE TWO UNITS REFUSES, and nothing is inserted.
        let mut short = covering(vec![(unit_a, val_a)]);
        assert_eq!(
            clone_value_to_region(query, SecondRegion::of(&mut short).unwrap(), &mut vals),
            None
        );
        assert_eq!(short.regions()[1].body.len(), 1);
    }
}
