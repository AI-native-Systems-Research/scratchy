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

//! `LoopMerging.cpp` — 7 of the campaign's 656 units (dependency level(s) [0, 1, 3, 4, 5, 6]).
//!
//! | unit | entry | level | lines | authority path:line |
//! |---|---|---|---|---|
//! | `e075_getNextEligibleOp` | 075 | 0 | 11 | `dcc/src/Transform/Sentient/LoopMerging.cpp:63` |
//! | `e076_getNewDbgName` | 076 | 0 | 6 | `dcc/src/Transform/Sentient/LoopMerging.cpp:75` |
//! | `e316_mergeLoops` | 316 | 1 | 100 | `dcc/src/Transform/Sentient/LoopMerging.cpp:179` |
//! | `e508_loopsAreMergeable` | 508 | 3 | 94 | `dcc/src/Transform/Sentient/LoopMerging.cpp:83` |
//! | `e564_runLoopMerging` | 564 | 4 | 39 | `dcc/src/Transform/Sentient/LoopMerging.cpp:281` |
//! | `e601_runOn` | 601 | 5 | 13 | `dcc/src/Transform/Sentient/LoopMerging.cpp:321` |
//! | `e627_runOnOperation` | 627 | 6 | 5 | `dcc/src/Transform/Sentient/LoopMerging.cpp:335` |

use crate::bridges::dataflow_ir_to_sentient::tf_cfgs_dataflow_conditional_tree::{
    DbgNamePrefix, new_dbg_name_from_list,
};
use crate::islands::dataflow_ir::Values;
use crate::islands::dataflow_ir::ty::ScalarTy;
use crate::islands::sentient::dialects::{
    Definitions, Op, Val, dataflow, erase_defining_op, replace_all_uses_with, sentient, symbol,
    uniform, use_count,
};
use crate::transform::sentient::utils::{ConstKind, is_constant};

/// WHERE AN OP SITS IN ITS BLOCK — the `Operation *` identity `loopsAreMergeable` compares against
/// the second loop (`:85`).
///
/// ⛔ A POSITION AND NOT A BORROW, because `getNextEligibleOp`'s answer is used for IDENTITY
/// (`getNextEligibleOp(loop_a) != loop_b`), and this island's ops are a tree with no addresses to
/// compare.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct InBlock(pub usize);

/// THE FIVE OPS MERGING STEPS OVER — `dataflow.get_unit`, `sentient.scalar_constant`,
/// `symbol.create_symbol`, `uniform.def_immutable_mapping` and `uniform.query_map` (`:66-69`).
///
/// ⛔ IT IS `uniform.query_map`, NOT `symbol.query_map`: the symbol dialect declares one of the same
/// name and the reference names the uniform one, so a symbol query between two loops keeps them
/// apart.
fn skipped_between_loops(op: &Op) -> bool {
    matches!(
        op,
        Op::Dataflow(dataflow::Op::GetUnit { .. })
            | Op::Sentient(sentient::Op::ScalarConstant { .. })
            | Op::Symbol(symbol::Op::CreateSymbol { .. })
            | Op::Uniform(uniform::Op::DefImmutableMapping { .. } | uniform::Op::QueryMap { .. })
    )
}

/// Replaces: e075_getNextEligibleOp
///
/// The next op after `op` in its own block, walking past the bookkeeping ops of
/// [`skipped_between_loops`].
///
/// ⛔ TRAP: `None` IS THE REFERENCE'S NULL AND ITS CALLER TREATS IT AS "NOT ADJACENT" — the last op
/// of a block is a merge candidate with nothing, so returning the block's end here would make two
/// loops in different blocks look adjacent.
#[must_use]
pub fn get_next_eligible_op(block: &[Op], op: InBlock) -> Option<InBlock> {
    block
        .iter()
        .enumerate()
        .skip(op.0 + 1)
        .find(|(_, next)| !skipped_between_loops(next))
        .map(|(at, _)| InBlock(at))
}

/// A LOOP'S `dbgName`, PRESENT — `for_op.getDbgName().value()` (`:77-78`) as a type.
///
/// ⛔ `.value()` ON AN ABSENT NAME IS `std::bad_optional_access`, so the reference merges two
/// unnamed loops by throwing. Here a nameless loop cannot reach [`get_new_dbg_name`] at all.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DbgName(String);

impl DbgName {
    /// The name a `sentient.for` carries, or nothing when it is not a loop or carries none.
    #[must_use]
    pub fn of(op: &Op) -> Option<DbgName> {
        let Op::Sentient(sentient::Op::For { dbg_name, .. }) = op else {
            return None;
        };
        dbg_name.clone().map(DbgName)
    }

    /// The text it carries.
    #[must_use]
    pub fn text(&self) -> &str {
        &self.0
    }
}

/// Replaces: e076_getNewDbgName
///
/// `LM(<first>, <second>)` — the merged loop's `dbgName`, built from both sources' own.
#[must_use]
pub fn get_new_dbg_name(for_op1: &DbgName, for_op2: &DbgName) -> DbgName {
    DbgName(format!("LM({}, {})", for_op1.text(), for_op2.text()))
}

/// TWO LOOPS THE PASS MAY MERGE — the pair's own two `DT_CHECK`s (`:181`, `:184-185`) as a type.
///
/// ⛔⛔ *"loop pointers are identical"* AND *"expected second loop to come right after the first one"*
/// ARE BOTH UNWRITABLE HERE, because the second position is not given: it is DERIVED by
/// [`get_next_eligible_op`], which never answers with its own argument.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MergeablePair {
    /// `loops.first`.
    first: InBlock,
    /// `loops.second`.
    second: InBlock,
}

impl MergeablePair {
    /// The pair `first` opens, or nothing when the next eligible op is not a second `sentient.for`.
    #[must_use]
    pub fn of(block: &[Op], first: InBlock) -> Option<MergeablePair> {
        let is_loop = |at: InBlock| {
            matches!(
                block.get(at.0),
                Some(Op::Sentient(sentient::Op::For { .. }))
            )
        };
        let second = get_next_eligible_op(block, first)?;
        (is_loop(first) && is_loop(second)).then_some(MergeablePair { first, second })
    }
}

/// WHERE A MERGE HAPPENS — the `dataflow.program_unit`'s block, and the nest of `sentient.for` bodies
/// from it down to the block the two loops share.
///
/// ⛔⛔ `DT_CHECK_MSG(unit, "expected loop to appear in dataflow.program_unit")` (`:203`) IS THIS TYPE,
/// and so are the reference's TWO builders: `OpBuilder const_builder(unit.getRegion())` (`:205`)
/// inserts at the START of `unit_body` — it is passed `.getRegion()` and so starts INSIDE, unlike
/// `LoopAbsorption.cpp:71` — while `OpBuilder builder(parent->getFirstChild()..)` (`:243`) inserts into
/// the block `nest` names. ⭐ NO ARM USES BOTH.
#[derive(Debug)]
pub struct MergeSite<'u> {
    /// The `dataflow.program_unit`'s own block.
    pub unit_body: &'u mut Vec<Op>,
    /// The `sentient.for` positions from `unit_body` down to the loops' shared block, outermost first.
    /// EMPTY means the two loops are top-level, which is `getParentLoop() == getRoot()`.
    pub nest: Vec<usize>,
}

/// WHETHER THE PAIR BECAME ONE LOOP — the `++loops_merged_count` the pass keeps (`:277`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Merged {
    /// The two loops are now one.
    Pair,
    /// The site names no such pair — `DT_CHECK_MSG(parent, "did not expect root here")` (`:239`) among
    /// them: two TOP-LEVEL loops whose bounds are neither constants nor query maps have nowhere for
    /// the `sentient.scalar_add` to go.
    NotMerged,
}

/// The block `nest` names — each position's `sentient.for` body in turn. `&[]` is `unit_body` itself.
fn block_at<'a>(unit_body: &'a [Op], nest: &[usize]) -> Option<&'a [Op]> {
    let mut block: &[Op] = unit_body;
    for at in nest {
        let Some(Op::Sentient(sentient::Op::For { body, .. })) = block.get(*at) else {
            return None;
        };
        block = body;
    }
    Some(block)
}

/// [`block_at`], for the rewrite.
fn block_at_mut<'a>(unit_body: &'a mut Vec<Op>, nest: &[usize]) -> Option<&'a mut Vec<Op>> {
    let mut block: &mut Vec<Op> = unit_body;
    for at in nest {
        let Some(Op::Sentient(sentient::Op::For { body, .. })) = block.get_mut(*at) else {
            return None;
        };
        block = body;
    }
    Some(block)
}

/// The `sentient.for` at `at`, as the four fields the merge reads.
fn loop_at(block: &[Op], at: InBlock) -> Option<(Val, &[sentient::Carried], Option<&str>)> {
    let Some(Op::Sentient(sentient::Op::For {
        bound,
        carried,
        dbg_name,
        ..
    })) = block.get(at.0)
    else {
        return None;
    };
    Some((*bound, carried, dbg_name.as_deref()))
}

/// Replaces: e316_mergeLoops
///
/// Merges the second loop into the first by summing their trip counts, rewiring the second's results
/// to the first's, renaming the survivor `LM(..)` and erasing the second loop and its dead bound.
///
/// ⛔ SIX OF THE EIGHT BOUND ARMS ARE `dcc::utils::updateBoundToValuePlusMap` /
/// `updateBoundToMapPlusMap` (`Analyses/Utils.cpp:421`, `:462`), which are OUT OF CAMPAIGN SCOPE.
///
/// ⛔ TRAP: `if (first_loop.getBound().getDefiningOp()->use_empty()) ..->erase()` (`:271-272`) READS THE
/// NEW BOUND, which the loop it was just built for reads — so that erase cannot fire. The one that
/// does is `saved_def_op`, captured from the SECOND bound BEFORE the loop goes (`:273-275`).
pub fn merge_loops(site: MergeSite<'_>, pair: MergeablePair, values: &mut Values) -> Merged {
    let MergeSite { unit_body, nest } = site;

    // ── everything the arm choice needs, before anything moves ──
    let (new_bound, new_bound_op, second_bound, results, dbg_name) = {
        let scope: &[Op] = unit_body;
        let Some(siblings) = block_at(scope, &nest) else {
            return Merged::NotMerged;
        };
        let (Some((first_bound, first_carried, first_name)), Some((second_bound, second_carried, second_name))) =
            (loop_at(siblings, pair.first), loop_at(siblings, pair.second))
        else {
            return Merged::NotMerged;
        };
        let defs = Definitions::from_innermost(core::slice::from_ref(&scope));
        let constant_of = |bound: Val| match defs.of(bound) {
            Some(Op::Sentient(sentient::Op::ScalarConstant { value, ty, .. })) => Some((*value, *ty)),
            _ => None,
        };
        let is_query_map = |bound: Val| {
            matches!(defs.of(bound), Some(Op::Uniform(uniform::Op::QueryMap { .. })))
        };
        // `first_query_map && isConstant<sentient::ConstantOp>(first_query_map.getResult())`
        // (`:208-213`).
        let is_constant_query_map =
            |bound: Val| is_query_map(bound) && is_constant(bound, ConstKind::ScalarConstant, defs);

        let new_bound = values.mint();
        let new_bound_op = match (constant_of(first_bound), constant_of(second_bound)) {
            // `first_const_bound && second_const_bound` (`:214-218`) — the sum, at the START of the
            // unit's own block.
            (Some((first, ty)), Some((second, _))) => BuiltBound::AtUnitStart(Op::Sentient(
                sentient::Op::ScalarConstant {
                    value: first + second,
                    result: new_bound,
                    reg_locale: sentient::RegType::Imm,
                    ty,
                    is_symbol: false,
                },
            )),
            (Some(_), None) if is_constant_query_map(second_bound) => {
                todo!("dcc::utils::updateBoundToValuePlusMap — out of campaign scope (`:219-223`)")
            }
            (None, Some(_)) if is_constant_query_map(first_bound) => {
                todo!("dcc::utils::updateBoundToValuePlusMap — out of campaign scope (`:224-229`)")
            }
            _ if is_constant_query_map(first_bound) && is_constant_query_map(second_bound) => {
                todo!("dcc::utils::updateBoundToMapPlusMap — out of campaign scope (`:230-233`)")
            }
            // `!first_query_map && !second_query_map` (`:235-247`) — a `sentient.scalar_add` BEFORE the
            // parent's first child, so no instruction lands between two otherwise mergeable loops.
            _ if !is_query_map(first_bound) && !is_query_map(second_bound) => {
                if nest.is_empty() {
                    return Merged::NotMerged;
                }
                BuiltBound::BeforeTheParentsFirstChild(Op::Sentient(sentient::Op::ScalarAdd {
                    lhs: first_bound,
                    rhs: second_bound,
                    result: new_bound,
                    // `AddOp::create` sets no `regLocale`, so the `.td`'s absent register stands, and
                    // `bound_type` (`:200`) is a trip count.
                    reg: None,
                    ty: ScalarTy::Index,
                    element_size: None,
                }))
            }
            _ if is_query_map(first_bound) != is_query_map(second_bound) => {
                todo!("dcc::utils::updateBoundToValuePlusMap — out of campaign scope (`:248-255`)")
            }
            _ => todo!("dcc::utils::updateBoundToMapPlusMap — out of campaign scope (`:256-260`)"),
        };
        // `DT_CHECK_MSG(.., "expected equal number of results")` (`:263-265`) is discharged by
        // `loopsAreMergeable`, which proved the two bodies equivalent; `llvm::zip` pairs what it has.
        let results: Vec<(Val, Val)> = second_carried
            .iter()
            .zip(first_carried)
            .map(|(second, first)| (second.result, first.result))
            .collect();
        (
            new_bound,
            new_bound_op,
            second_bound,
            results,
            new_dbg_name_from_list(DbgNamePrefix::Lm, first_name, &[second_name]),
        )
    };

    // ── the rewrite, innermost block first ──
    {
        let Some(siblings) = block_at_mut(unit_body, &nest) else {
            return Merged::NotMerged;
        };
        if let Some(Op::Sentient(sentient::Op::For {
            bound,
            dbg_name: name,
            ..
        })) = siblings.get_mut(pair.first.0)
        {
            *bound = new_bound;
            // ⭐ ONLY WHEN BOTH SOURCES HAD ONE (`:266-268`) — `setDbgNameAttr(op, nullptr)` REMOVES
            // the attribute, so the reference's `if` keeps the survivor's own name instead.
            if let Some(merged) = dbg_name {
                *name = Some(merged);
            }
        }
        // `std::get<1>(pair).replaceAllUsesWith(std::get<0>(pair))` (`:265`).
        for (of, with) in results {
            replace_all_uses_with(siblings, of, with);
        }
        // `if (first_loop.getBound().getDefiningOp()->use_empty()) ..->erase()` (`:271-272`).
        if use_count(new_bound, siblings) == 0 {
            erase_defining_op(siblings, new_bound);
        }
        // `second_loop.getOperation()->erase()` (`:274`).
        siblings.remove(pair.second.0);
    }
    // `if (saved_def_op->use_empty()) saved_def_op->erase();` (`:275`) — the second bound's definer
    // outlived the loop that read it.
    if use_count(second_bound, unit_body) == 0 {
        erase_defining_op(unit_body, second_bound);
    }
    match new_bound_op {
        BuiltBound::AtUnitStart(op) => unit_body.insert(0, op),
        BuiltBound::BeforeTheParentsFirstChild(op) => {
            let Some(siblings) = block_at_mut(unit_body, &nest) else {
                return Merged::NotMerged;
            };
            let at = siblings
                .iter()
                .position(|op| matches!(op, Op::Sentient(sentient::Op::For { .. })))
                .unwrap_or(0);
            siblings.insert(at, op);
        }
    }
    Merged::Pair
}

/// THE MERGED BOUND AND WHICH BUILDER MAKES IT — see [`MergeSite`].
#[derive(Debug)]
enum BuiltBound {
    /// `const_builder`, at the start of the unit's own block.
    AtUnitStart(Op),
    /// `builder`, before the first `sentient.for` of the loops' own parent.
    BeforeTheParentsFirstChild(Op),
}

// crustify:todo: e508_loopsAreMergeable
//   authority : dcc/src/Transform/Sentient/LoopMerging.cpp:83  (94 body lines, level 3)
//   original  : bool LoopMergingPass::loopsAreMergeable(sentient::ForOp loop_a, sentient::ForOp loop_b)
//   calls     : e075_getNextEligibleOp, e252_size, e422_insert

// crustify:todo: e564_runLoopMerging
//   authority : dcc/src/Transform/Sentient/LoopMerging.cpp:281  (39 body lines, level 4)
//   original  : void LoopMergingPass::runLoopMerging(Operation *op)
//   calls     : e316_mergeLoops, e508_loopsAreMergeable

// crustify:todo: e601_runOn
//   authority : dcc/src/Transform/Sentient/LoopMerging.cpp:321  (13 body lines, level 5)
//   original  : void LoopMergingPass::runOn(ModuleOp module_op)
//   calls     : e564_runLoopMerging

// crustify:todo: e627_runOnOperation
//   authority : dcc/src/Transform/Sentient/LoopMerging.cpp:335  (5 body lines, level 6)
//   original  : void LoopMergingPass::runOnOperation()
//   calls     : e601_runOn

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::islands::dataflow_ir::ty::ScalarTy;
    use crate::islands::sentient::dialects::Val;
    use crate::units::{DfirUnit, Residency};

    /// `sentient.for` carrying `dbg_name`.
    fn sentient_for(dbg_name: Option<&str>) -> Op {
        Op::Sentient(sentient::Op::For {
            iv: Val(0),
            bound: Val(1),
            carried: Vec::new(),
            dbg_name: dbg_name.map(str::to_owned),
            body: Vec::new(),
        })
    }

    /// A `sentient.scalar_constant` a loop bound reads.
    fn constant_of(result: Val, value: i64) -> Op {
        Op::Sentient(sentient::Op::ScalarConstant {
            value,
            result,
            reg_locale: sentient::RegType::Imm,
            ty: ScalarTy::Index,
            is_symbol: false,
        })
    }

    /// A `sentient.for` over `bound`, binding `iv`.
    fn loop_over(bound: Val, iv: Val, dbg_name: Option<&str>) -> Op {
        Op::Sentient(sentient::Op::For {
            iv,
            bound,
            carried: Vec::new(),
            dbg_name: dbg_name.map(str::to_owned),
            body: Vec::new(),
        })
    }

    /// The four non-`sentient` bookkeeping ops the walk steps over, plus the constant.
    fn skipped_ops() -> Vec<Op> {
        vec![
            Op::Dataflow(dataflow::Op::GetUnit {
                result: Val(2),
                residency: Residency::Global,
                unit: DfirUnit::L3lu,
                num_folds: None,
            }),
            Op::Sentient(sentient::Op::ScalarConstant {
                value: 0,
                result: Val(3),
                reg_locale: sentient::RegType::Imm,
                ty: ScalarTy::Index,
                is_symbol: false,
            }),
            Op::Symbol(symbol::Op::CreateSymbol {
                result: Val(4),
                symbol_id: 0,
                max_value: None,
            }),
            Op::Uniform(uniform::Op::DefImmutableMapping {
                result: Val(5),
                pairs: Vec::new(),
            }),
            Op::Uniform(uniform::Op::QueryMap {
                result: Val(6),
                map: Val(5),
                key: Val(3),
            }),
        ]
    }

    /// Two loops separated by all five skipped ops are adjacent; one separated by anything else is
    /// not, and the last op of a block is adjacent to nothing.
    #[test]
    fn e075_walks_past_the_five_bookkeeping_ops_and_stops_at_the_block_end() {
        let mut block = vec![sentient_for(None)];
        block.extend(skipped_ops());
        block.push(sentient_for(None));
        assert_eq!(get_next_eligible_op(&block, InBlock(0)), Some(InBlock(6)));
        assert_eq!(get_next_eligible_op(&block, InBlock(6)), None);

        let blocked = vec![
            sentient_for(None),
            Op::Sentient(sentient::Op::Nop { dbg_name: None }),
            sentient_for(None),
        ];
        assert_eq!(get_next_eligible_op(&blocked, InBlock(0)), Some(InBlock(1)));
    }

    /// The merged name, and the absent name the reference's `.value()` throws on.
    #[test]
    fn e076_wraps_both_source_names_in_lm() {
        let first = sentient_for(Some("L0"));
        let second = sentient_for(Some("LM(L1, L2)"));
        let first = DbgName::of(&first).expect("a named loop");
        let second = DbgName::of(&second).expect("a named loop");
        assert_eq!(
            get_new_dbg_name(&first, &second).text(),
            "LM(L0, LM(L1, L2))"
        );
        assert!(DbgName::of(&sentient_for(None)).is_none());
        assert!(DbgName::of(&Op::Sentient(sentient::Op::Nop { dbg_name: None })).is_none());
    }
    /// Two adjacent constant-bound loops become one over the SUM, named from both, with the second
    /// loop and the second bound gone — and ⭐ THE FIRST LOOP'S OLD BOUND STILL STANDING, which is
    /// the reference's own dead erase (`:271-272`).
    #[test]
    fn e316_merges_two_constant_bound_loops_into_one_over_the_sum() {
        let mut values = Values::default();
        for _ in 0..20 {
            let _ = values.mint();
        }
        let mut unit_body = vec![
            constant_of(Val(0), 4),
            constant_of(Val(1), 6),
            loop_over(Val(0), Val(10), Some("a")),
            loop_over(Val(1), Val(11), Some("b")),
        ];
        let pair = MergeablePair::of(&unit_body, InBlock(2)).expect("two adjacent loops");
        let merged = merge_loops(
            MergeSite {
                unit_body: &mut unit_body,
                nest: Vec::new(),
            },
            pair,
            &mut values,
        );
        assert_eq!(merged, Merged::Pair);
        assert_eq!(
            unit_body,
            vec![
                constant_of(Val(20), 10),
                constant_of(Val(0), 4),
                loop_over(Val(20), Val(10), Some("LM(a, b)")),
            ]
        );
    }
}
