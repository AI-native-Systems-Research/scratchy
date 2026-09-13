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

//! `MultiDimLoopPeeling.cpp` — 7 of the campaign's 656 units (dependency level(s) [0, 1, 2, 4, 5, 6]).
//!
//! | unit | entry | level | lines | authority path:line |
//! |---|---|---|---|---|
//! | `e095_decrementPredicatesOnIV` | 095 | 0 | 27 | `dcc/src/Transform/Sentient/MultiDimLoopPeeling.cpp:433` |
//! | `e096_simplifyConditionals` | 096 | 0 | 11 | `dcc/src/Transform/Sentient/MultiDimLoopPeeling.cpp:529` |
//! | `e322_copyOneIter` | 322 | 1 | 65 | `dcc/src/Transform/Sentient/MultiDimLoopPeeling.cpp:461` |
//! | `e448_performLoopPeeling` | 448 | 2 | 63 | `dcc/src/Transform/Sentient/MultiDimLoopPeeling.cpp:543` |
//! | `e568_computePeelingInfo` | 568 | 4 | 49 | `dcc/src/Transform/Sentient/MultiDimLoopPeeling.cpp:351` |
//! | `e604_findCandidates` | 604 | 5 | 30 | `dcc/src/Transform/Sentient/MultiDimLoopPeeling.cpp:402` |
//! | `e630_run` | 630 | 6 | 130 | `dcc/src/Transform/Sentient/MultiDimLoopPeeling.cpp:607` |

#![allow(dead_code)]
// ⛔ NOTHING CALLS THIS FILE'S MANAGER YET — `e630_run` below is ported, but the pass ENTRY that
// constructs a manager per top-level loop is `e643_runOnOperation` (`MultiDimLoopPeeling.cpp:738`),
// still an unfilled anchor in this module's `mod.rs`. CI runs clippy with `-D warnings`, so without
// this the batch fails its own gate.
// ⭐ REMOVE THIS WITH `e643_runOnOperation`: an unused item here is a real defect at that point.

use super::{IvLoopInfo, Peeling, PeelingCandidates};
use crate::arch::Arch;
use crate::islands::dataflow_ir::ty::ScalarTy;
use crate::islands::dataflow_ir::{ValueMapping, Values};
use crate::islands::sentient::ProgramUnit;
use crate::islands::sentient::dialects::{
    self, Definitions, Op, StaticBranch, Val, replace_if_with_region, sentient, static_if_branch,
    static_operand,
};
use crate::transform::sentient::ForRef;
use crate::transform::sentient::analyses::{
    ExpressionEvaluator, InstructionCount, InstructionEstimator, PropagationAnalysis, UnitIndexMap,
};
use crate::transform::sentient::lightweight_simplification::sentient::run_light_weight_simplifications;
use crate::transform::sentient::loop_absorption::run_loop_absorption;
use crate::transform::sentient::loop_merging::run_loop_merging;
use crate::transform::sentient::op_rerolling::{MergeXrfIntoMac, RerollScope, run_op_rerolling};
use crate::transform::sentient::scalar_simplifications::sentient::Propagation;
use crate::transform::sentient::utils::{InBlock, OpAt, reverse_predicate};

/// WHERE ONE OP SITS IN A REGION TREE — the block reached by taking region `r` of the op at index `i`
/// for each `(i, r)` of `into`, then index `at` in that block.
///
/// ⛔⛔ AN MLIR `Operation *` IS ITS OWN CURSOR AND THIS IS NOT. Both ports below decide over the
/// whole tree and then mutate it, so a decision has to name its subject; a `&mut Op` would hold the
/// tree borrowed while the next decision is read. `Ord` is derived and load-bearing: `into` orders
/// before `at`, so a descendant (whose `into` extends its ancestor's) always sorts AFTER its
/// ancestor, and the greatest remaining path is always the deepest-then-last one — post-order.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct RegionPath {
    /// `(op index, region index)` from the root block down.
    into: Vec<(usize, usize)>,
    /// The op's index in the block `into` reaches.
    at: usize,
}

/// The block `into` names, or `None` where an index no longer resolves.
fn block_at<'a>(root: &'a mut Vec<Op>, into: &[(usize, usize)]) -> Option<&'a mut Vec<Op>> {
    let mut block = root;
    for &(op_index, region_index) in into {
        let op = block.get_mut(op_index)?;
        block = dialects::regions_mut(op).into_iter().nth(region_index)?;
    }
    Some(block)
}

/// Every `sentient.if` under `block`, deepest-last-first, as `(path, index of the op)`.
fn if_paths(block: &[Op], into: &[(usize, usize)], out: &mut Vec<RegionPath>) {
    for (at, op) in block.iter().enumerate() {
        if matches!(op, Op::Sentient(sentient::Op::If { .. })) {
            out.push(RegionPath {
                into: into.to_vec(),
                at,
            });
        }
        for (region_index, region) in dialects::regions_ref(op).into_iter().enumerate() {
            let mut deeper = into.to_vec();
            deeper.push((at, region_index));
            if_paths(region, &deeper, out);
        }
    }
}

/// WHICH OPERAND OF A `sentient.if` IS THE INDUCTION VARIABLE — `if_op.getLhs() == iv` against
/// `if_op.getRhs() == iv` (`MultiDimLoopPeeling.cpp:437-440`), which decides which operand index the
/// decremented constant is written to (`:452-455`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IvSide {
    /// `$lhs` is the IV, so `$rhs` is the constant side — `setOperand(1, ..)`.
    Lhs,
    /// `$rhs` is the IV, so `$lhs` is the constant side — `setOperand(0, ..)`.
    Rhs,
}

/// ONE `sentient.if` COMPARING THE INDUCTION VARIABLE AGAINST A `sentient.scalar_constant`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PredicateOnIv {
    path: RegionPath,
    side: IvSide,
    /// `const_op.getValue()`.
    value: i64,
    /// `const_side.getType()`, which the replacement constant is built with.
    ty: ScalarTy,
}

/// EVERY USE OF ONE LOOP'S INDUCTION VARIABLE, ALL OF THEM COMPARISONS AGAINST A CONSTANT.
///
/// ⛔⛔ THE TYPE IS `decrementPredicatesOnIV`'S THREE ABORTS. `llvm_unreachable("expect iv users to
/// be IfOps")` (`:457`), `llvm_unreachable("expect a comparison on the IV")` (`:442`) and
/// `DT_CHECK_MSG(const_op, "Expect a comparison between the IV and a ConstantOp")` (`:446-447`) all
/// say the same thing: an IV whose uses are not exactly this. [`PredicatesOnIv::of`] answers `None`
/// there instead, so the caller cannot reach the rewrite with a tree that would have aborted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PredicatesOnIv {
    iv: Val,
    predicates: Vec<PredicateOnIv>,
}

impl PredicatesOnIv {
    /// The comparisons on `iv` anywhere under `scope`, or `None` where `iv` has a use that is not
    /// one — `iv.getUsers()` with its three aborts read as a precondition.
    #[must_use]
    pub fn of(iv: Val, scope: &[Op]) -> Option<PredicatesOnIv> {
        let mut paths = Vec::new();
        if_paths(scope, &[], &mut paths);
        let mut predicates = Vec::new();
        for path in paths {
            let block = walk_to(scope, &path.into)?;
            let Op::Sentient(sentient::Op::If { lhs, rhs, .. }) = block.get(path.at)? else {
                return None;
            };
            let (side, const_side) = match (*lhs == iv, *rhs == iv) {
                (true, false) => (IvSide::Lhs, *rhs),
                (false, true) => (IvSide::Rhs, *lhs),
                // Neither side is the IV: not a use of it at all, so not this walk's business.
                (false, false) => continue,
                // Both sides: `const_side` would be the IV itself and its `dyn_cast` would fail.
                (true, true) => return None,
            };
            let Some(Op::Sentient(sentient::Op::ScalarConstant { value, ty, .. })) =
                dialects::defining_op(const_side, scope)
            else {
                return None;
            };
            predicates.push(PredicateOnIv {
                path,
                side,
                value: *value,
                ty: *ty,
            });
        }
        // ⭐ THE EXACT WITNESS THAT `getUsers()` YIELDED NOTHING ELSE: every use counted, and every
        // one of them accounted for by a comparison collected above.
        (dialects::use_count(iv, scope) == predicates.len())
            .then_some(PredicatesOnIv { iv, predicates })
    }

    /// The induction variable these comparisons are on.
    #[must_use]
    pub const fn iv(&self) -> Val {
        self.iv
    }
}

/// The block `into` names, immutably — see [`block_at`].
fn walk_to<'a>(root: &'a [Op], into: &[(usize, usize)]) -> Option<&'a [Op]> {
    let mut block = root;
    for &(op_index, region_index) in into {
        let op = block.get(op_index)?;
        block = dialects::regions_ref(op).into_iter().nth(region_index)?;
    }
    Some(block)
}

/// Replaces: e095_decrementPredicatesOnIV
///
/// Rebuilds every comparison on the induction variable against a constant one lower, which is what
/// the loop's bound losing its last iteration means for the conditions inside it.
///
/// ⛔ TRAP: THE OLD CONSTANT IS LEFT WHERE IT IS. The reference creates a second `ConstantOp` and
/// re-points the operand (`:449-455`); it never edits the old one, which other ops may still read.
///
/// ⛔ TRAP: THE REPLACEMENT TAKES `const_side.getType()`, NOT the IV's — the two differ wherever a
/// comparison is on `i1`.
pub fn decrement_predicates_on_iv(
    predicates: &PredicatesOnIv,
    values: &mut Values,
    block: &mut Vec<Op>,
) {
    // ⭐ MINTED IN COLLECTION ORDER, which fixes only the `Val` numbering, and APPLIED
    // deepest-and-last first: inserting one shifts the indices at and after it in that
    // one block, and every path still to be applied there names a smaller index.
    let mut ordered: Vec<(Val, &PredicateOnIv)> = predicates
        .predicates
        .iter()
        .map(|predicate| (values.mint(), predicate))
        .collect();
    ordered.sort_by(|(_, a), (_, b)| a.path.cmp(&b.path));
    for (result, predicate) in ordered.into_iter().rev() {
        let constant = Op::Sentient(sentient::Op::ScalarConstant {
            value: predicate.value - 1,
            result,
            // `ConstantOp`'s own default, which this creation does not override.
            reg_locale: sentient::RegType::Imm,
            ty: predicate.ty,
            is_symbol: false,
        });
        let Some(inner) = block_at(block, &predicate.path.into) else {
            continue;
        };
        inner.insert(predicate.path.at, constant);
        if let Some(Op::Sentient(sentient::Op::If { lhs, rhs, .. })) =
            inner.get_mut(predicate.path.at + 1)
        {
            match predicate.side {
                IvSide::Lhs => *rhs = result,
                IvSide::Rhs => *lhs = result,
            }
        }
    }
}

/// WHICH `sentient.if`s A ROUND OF CANONICALISATION MAY TOUCH — `applyOpPatternsGreedily`'s `ops`
/// argument under `GreedyRewriteStrictness::ExistingOps`.
///
/// ⛔ THE TWO CALLERS BUILD IT DIFFERENTLY. `performLoopPeeling` walks the loop's parent op for every
/// `sentient.if` (`:601-603`); `copyOneIter` lists only the users of the cloned loop's induction
/// variable (`:474-479`), and since it has already replaced that variable with a fresh constant that
/// nothing else reads, naming that constant selects exactly those.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IfOpsToSimplify {
    /// Every `sentient.if` in the block.
    Every,
    /// Those comparing this value on either side.
    Reading(Val),
}

/// Replaces: e096_simplifyConditionals
///
/// Folds every candidate `sentient.if` whose condition is decidable from the constants and loop
/// bounds around it, splicing the surviving branch into its place — a run of `sentient.if`'s ONE
/// canonicalisation pattern, `RemoveStaticCondition` (`SentientOps.cpp:1509-1512`).
///
/// ⛔ TRAP: ONE PASS DEEPEST-FIRST IS NOT ENOUGH AND THE LOOP IS NOT DECORATION. A folded `if`
/// rewires its readers onto the yielded values, which can turn an enclosing `if`'s operand into a
/// constant; the reference gets that from the greedy driver's worklist. Each round erases one
/// `sentient.if`, so this terminates.
pub fn simplify_conditionals(candidates: IfOpsToSimplify, block: &mut Vec<Op>) {
    while let Some((path, branch)) = next_static_if(candidates, block) {
        let rewires = match block_at(block, &path.into) {
            Some(inner) => replace_if_with_region(inner, path.at, branch),
            None => break,
        };
        // `rewriter.replaceOp(op, results)` — the readers are in the enclosing scope.
        for (of, with) in rewires {
            dialects::replace_all_uses_with(block, of, with);
        }
    }
}

/// The deepest-then-last candidate `sentient.if` the pattern matches — the greedy worklist's next
/// item, in the post-order the callers collect their lists in.
fn next_static_if(candidates: IfOpsToSimplify, block: &[Op]) -> Option<(RegionPath, StaticBranch)> {
    let mut paths = Vec::new();
    if_paths(block, &[], &mut paths);
    paths.sort();
    for path in paths.into_iter().rev() {
        let inner = walk_to(block, &path.into)?;
        let Some(Op::Sentient(sentient::Op::If {
            predicate,
            lhs,
            rhs,
            yielded,
            else_body,
            ..
        })) = inner.get(path.at)
        else {
            continue;
        };
        if let IfOpsToSimplify::Reading(val) = candidates
            && *lhs != val
            && *rhs != val
        {
            continue;
        }
        if let Some(branch) = static_if_branch(
            *predicate,
            static_operand(*lhs, block),
            static_operand(*rhs, block),
            !else_body.is_empty(),
            !yielded.is_empty(),
        ) {
            return Some((path, branch));
        }
    }
    None
}

#[cfg(test)]
mod unit_tests {
    use super::{
        IfOpsToSimplify, IvLoopInfo, Peeling, PeelingCandidates, PredicatesOnIv, ProgramUnit,
        compute_peeling_info, copy_one_iter, decrement_predicates_on_iv, find_candidates,
        perform_loop_peeling, run, simplify_conditionals, top_level_at,
    };
    use crate::arch::Dd2;
    use crate::islands::dataflow_ir::Units;
    use crate::islands::dataflow_ir::Values;
    use crate::islands::dataflow_ir::ty::ScalarTy;
    use crate::islands::sentient::dialects::{Definitions, Op, Val, sentient};
    use crate::transform::sentient::ForRef;
    use crate::transform::sentient::analyses::{
        Evaluation, ExpressionEvaluator, InstructionCount, InstructionEstimator, OffsetSites,
        Offsets, OutOfScopePropagationAnalysis, OutOfScopeUnitIndexMap,
    };
    use crate::units::DfirUnit;

    /// e568 — on a loop of bound 8 (whose IV counts 8 down to 1) `%iv >= 8` names the FIRST
    /// iteration and reads the same way with the operands swapped, `%iv > 1` names the LAST, and a
    /// loop of bound 1 is refused before any of that.
    #[test]
    fn e568_reads_the_comparison_with_the_iv_on_the_left() {
        let block = vec![
            constant(8, Val(1)),
            constant(1, Val(2)),
            for_loop(
                Val(3),
                Val(1),
                Vec::new(),
                vec![
                    if_op(
                        sentient::CmpPredicate::Sge,
                        Val(3),
                        Val(1),
                        Vec::new(),
                        Vec::new(),
                    ),
                    if_op(
                        sentient::CmpPredicate::Sle,
                        Val(1),
                        Val(3),
                        Vec::new(),
                        Vec::new(),
                    ),
                    if_op(
                        sentient::CmpPredicate::Sgt,
                        Val(3),
                        Val(2),
                        Vec::new(),
                        Vec::new(),
                    ),
                ],
            ),
        ];
        let regions = [block.as_slice()];
        let defs = Definitions::from_innermost(&regions);
        let Op::Sentient(sentient::Op::For { body, .. }) = &block[2] else {
            panic!("the loop is a sentient.for")
        };
        let mut ivs = IvLoopInfo::default();
        assert_eq!(
            compute_peeling_info(&body[0], &mut ivs, defs),
            Some((ForRef(Val(3)), Peeling::FirstIterOnly))
        );
        // `8 <= %iv` is `%iv >= 8` reversed, and answers the same end.
        assert_eq!(
            compute_peeling_info(&body[1], &mut ivs, defs),
            Some((ForRef(Val(3)), Peeling::FirstIterOnly))
        );
        assert_eq!(
            compute_peeling_info(&body[2], &mut ivs, defs),
            Some((ForRef(Val(3)), Peeling::LastIterOnly))
        );

        let small = vec![
            constant(1, Val(11)),
            for_loop(
                Val(12),
                Val(11),
                Vec::new(),
                vec![if_op(
                    sentient::CmpPredicate::Sge,
                    Val(12),
                    Val(11),
                    Vec::new(),
                    Vec::new(),
                )],
            ),
        ];
        let regions = [small.as_slice()];
        let defs = Definitions::from_innermost(&regions);
        let Op::Sentient(sentient::Op::For { body, .. }) = &small[1] else {
            panic!("the loop is a sentient.for")
        };
        assert_eq!(compute_peeling_info(&body[0], &mut ivs, defs), None);
    }

    /// `%c = scalar_constant N : index`.
    fn constant(value: i64, result: Val) -> Op {
        Op::Sentient(sentient::Op::ScalarConstant {
            value,
            result,
            reg_locale: sentient::RegType::Imm,
            ty: ScalarTy::Index,
            is_symbol: false,
        })
    }

    /// `sentient.if <predicate> lhs, rhs { then } else { else }`, resultless.
    fn if_op(
        predicate: sentient::CmpPredicate,
        lhs: Val,
        rhs: Val,
        then_body: Vec<Op>,
        else_body: Vec<Op>,
    ) -> Op {
        Op::Sentient(sentient::Op::If {
            predicate,
            lhs,
            rhs,
            yielded: Vec::new(),
            dbg_name: None,
            then_body,
            else_body,
        })
    }

    /// e604 — the walk is POST-ORDER, so the inner loop's opportunity is recorded before the outer
    /// loop's, and a comparison naming neither end is not one at all.
    #[test]
    fn e604_records_the_inner_loops_opportunity_first() {
        let block = vec![
            constant(8, Val(1)),
            constant(1, Val(2)),
            for_loop(
                Val(3),
                Val(1),
                Vec::new(),
                vec![
                    for_loop(
                        Val(4),
                        Val(1),
                        Vec::new(),
                        vec![if_op(
                            sentient::CmpPredicate::Sge,
                            Val(4),
                            Val(1),
                            Vec::new(),
                            Vec::new(),
                        )],
                    ),
                    if_op(
                        sentient::CmpPredicate::Sgt,
                        Val(3),
                        Val(2),
                        Vec::new(),
                        Vec::new(),
                    ),
                    // `%iv >= 1` is neither end of a loop counting 8 down to 1.
                    if_op(
                        sentient::CmpPredicate::Sge,
                        Val(3),
                        Val(2),
                        Vec::new(),
                        Vec::new(),
                    ),
                ],
            ),
        ];
        let mut ivs = IvLoopInfo::default();
        let candidates = find_candidates(ForRef(Val(3)), &block, &mut ivs);
        assert_eq!(
            candidates.records(),
            [
                (ForRef(Val(4)), Peeling::FirstIterOnly),
                (ForRef(Val(3)), Peeling::LastIterOnly),
            ]
        );
        // A loop this unit does not hold is the reference's null `for_op`, and holds nothing to walk.
        assert!(
            find_candidates(ForRef(Val(99)), &block, &mut ivs)
                .records()
                .is_empty()
        );
    }

    /// `%r = sentient.scalar_add %lhs, %rhs : index`.
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

    /// One carried value: `init` in, `arg` inside, `result` out.
    fn carried(init: Val, arg: Val, result: Val) -> sentient::Carried {
        sentient::Carried {
            init,
            arg,
            result,
            reg: sentient::Reg {
                locale: sentient::RegType::Lrf,
                index: None,
            },
            program_header: false,
            element_size: None,
        }
    }

    /// `sentient.for %iv = 1 to %bound iter_args(..) { body }`.
    fn for_loop(iv: Val, bound: Val, carried: Vec<sentient::Carried>, body: Vec<Op>) -> Op {
        Op::Sentient(sentient::Op::For {
            iv,
            bound,
            bound_reg: None,
            carried,
            dbg_name: None,
            body,
        })
    }

    /// A loop whose body compares the IV against 4 on each side, after the bound has lost an
    /// iteration: both constants are rebuilt one lower and the comparisons re-pointed, with the
    /// originals left in place.
    #[test]
    fn decrement_predicates_on_iv_rebuilds_both_constant_sides() {
        let (iv, bound, four) = (Val(0), Val(1), Val(2));
        let mut block = vec![
            constant(8, bound),
            constant(4, four),
            Op::Sentient(sentient::Op::For {
                iv,
                bound,
                bound_reg: None,
                carried: Vec::new(),
                dbg_name: None,
                body: vec![
                    if_op(
                        sentient::CmpPredicate::Slt,
                        iv,
                        four,
                        Vec::new(),
                        Vec::new(),
                    ),
                    if_op(
                        sentient::CmpPredicate::Sgt,
                        four,
                        iv,
                        Vec::new(),
                        Vec::new(),
                    ),
                ],
            }),
        ];
        let predicates = PredicatesOnIv::of(iv, &block).expect("both uses are comparisons");
        let mut values = Values::default();
        for _ in 0..3 {
            let _ = values.mint();
        }
        decrement_predicates_on_iv(&predicates, &mut values, &mut block);
        let Op::Sentient(sentient::Op::For { body, .. }) = &block[2] else {
            panic!("the loop is still the third op")
        };
        assert_eq!(body.len(), 4);
        assert!(matches!(
            body[0],
            Op::Sentient(sentient::Op::ScalarConstant { value: 3, .. })
        ));
        assert!(matches!(
            body[1],
            Op::Sentient(sentient::Op::If { lhs, rhs, .. }) if lhs == iv && rhs == Val(3)
        ));
        assert!(matches!(
            body[2],
            Op::Sentient(sentient::Op::ScalarConstant { value: 3, .. })
        ));
        assert!(matches!(
            body[3],
            Op::Sentient(sentient::Op::If { lhs, rhs, .. }) if lhs == Val(4) && rhs == iv
        ));
        // An IV also read by something that is not a comparison is the `llvm_unreachable`.
        let mut with_other_use = block.clone();
        if let Op::Sentient(sentient::Op::For { body, .. }) = &mut with_other_use[2] {
            body.push(Op::Sentient(sentient::Op::Yield { results: vec![iv] }));
        }
        assert_eq!(PredicatesOnIv::of(iv, &with_other_use), None);
    }

    /// `%iv < 1` over a loop running `[1, 8]` is statically false, so the `else` branch survives and
    /// the `sentient.if` inside it — now out of the folded one — folds in the same run.
    #[test]
    fn simplify_conditionals_folds_a_nest_to_fixpoint() {
        let (iv, bound, one) = (Val(0), Val(1), Val(2));
        let nop = Op::Sentient(sentient::Op::Nop { dbg_name: None });
        let mut block = vec![
            constant(8, bound),
            constant(1, one),
            Op::Sentient(sentient::Op::For {
                iv,
                bound,
                bound_reg: None,
                carried: Vec::new(),
                dbg_name: None,
                body: vec![if_op(
                    sentient::CmpPredicate::Slt,
                    iv,
                    one,
                    Vec::new(),
                    vec![if_op(
                        sentient::CmpPredicate::Sge,
                        iv,
                        one,
                        vec![nop.clone()],
                        Vec::new(),
                    )],
                )],
            }),
        ];
        simplify_conditionals(IfOpsToSimplify::Every, &mut block);
        let Op::Sentient(sentient::Op::For { body, .. }) = &block[2] else {
            panic!("the loop survives")
        };
        assert_eq!(body.len(), 1);
        assert!(matches!(body[0], Op::Sentient(sentient::Op::Nop { .. })));
    }

    /// e322 — one iteration of `for %iv = 1 to 8 iter_args(%a = %c0) { if %iv < 4 {..}; %n = %a + 4;
    /// yield %n }` peeled BEFORE the loop, then one peeled AFTER it: the clone's body lands in the
    /// clone's place with the IV pinned and the iter arg reading the init, the `sentient.if` the
    /// pinning decided is gone, and the loop's own result reader ends on the yielded value.
    #[test]
    fn e322_copies_one_iteration_out_of_the_loop() {
        let nop = Op::Sentient(sentient::Op::Nop { dbg_name: None });
        let loop_op = |sibling: Vec<Op>| {
            let mut block = vec![
                constant(0, Val(0)),
                constant(8, Val(1)),
                constant(4, Val(2)),
                for_loop(
                    Val(3),
                    Val(1),
                    vec![carried(Val(0), Val(4), Val(5))],
                    vec![
                        if_op(
                            sentient::CmpPredicate::Slt,
                            Val(3),
                            Val(2),
                            vec![nop.clone()],
                            Vec::new(),
                        ),
                        add(Val(4), Val(2), Val(6)),
                        Op::Sentient(sentient::Op::Yield {
                            results: vec![Val(6)],
                        }),
                    ],
                ),
            ];
            block.extend(sibling);
            block
        };

        // Peeled before the loop: constant, hoisted body, then the untouched original.
        let mut values = Values::default();
        for _ in 0..7 {
            let _ = values.mint();
        }
        let mut block = loop_op(Vec::new());
        let original = block[3].clone();
        let mut new_iter_args = vec![Val(0)];
        copy_one_iter(&mut block, 3, &mut new_iter_args, 5, false, &mut values);
        assert_eq!(new_iter_args, vec![Val(10)]);
        assert!(matches!(
            block[3],
            Op::Sentient(sentient::Op::ScalarConstant { value: 5, .. })
        ));
        assert_eq!(block[4], add(Val(0), Val(2), Val(10)));
        assert_eq!(block[5], original);
        assert_eq!(block.len(), 6);

        // Peeled after it, with a sibling reading the loop's result: that reader ends on the value
        // this iteration yielded, through the clone's result.
        let mut values = Values::default();
        for _ in 0..13 {
            let _ = values.mint();
        }
        let mut block = loop_op(vec![add(Val(5), Val(2), Val(12))]);
        let original = block[3].clone();
        let mut new_iter_args = vec![Val(0)];
        copy_one_iter(&mut block, 3, &mut new_iter_args, 8, true, &mut values);
        assert_eq!(new_iter_args, vec![Val(16)]);
        assert_eq!(block[3], original);
        assert_eq!(block[5], add(Val(0), Val(2), Val(16)));
        assert_eq!(block[6], add(Val(16), Val(2), Val(12)));
        assert_eq!(block.len(), 7);
    }

    /// e448 — a loop running `[1, 8]` peeled at both ends: the first iteration's copy is pinned to 8
    /// and lands BEFORE the loop, the last iteration's is pinned to 1 and lands after, the bound
    /// loses both, the three run in the loop's carried value, and the condition inside the loop is
    /// rebuilt one lower because the bound lost an iteration.
    #[test]
    fn e448_peels_both_ends_and_chains_them_through_the_carried_value() {
        let mut values = Values::default();
        let (init, bound, four) = (values.mint(), values.mint(), values.mint());
        let (iv, arg, result) = (values.mint(), values.mint(), values.mint());
        let (body_val, reader) = (values.mint(), values.mint());
        let nop = Op::Sentient(sentient::Op::Nop { dbg_name: None });
        let mut loop_op = for_loop(
            iv,
            bound,
            vec![carried(init, arg, result)],
            vec![
                if_op(
                    sentient::CmpPredicate::Slt,
                    iv,
                    four,
                    vec![nop.clone()],
                    Vec::new(),
                ),
                add(arg, four, body_val),
                Op::Sentient(sentient::Op::Yield {
                    results: vec![body_val],
                }),
            ],
        );
        if let Op::Sentient(sentient::Op::For { dbg_name, .. }) = &mut loop_op {
            *dbg_name = Some("L".to_string());
        }
        let mut unit = vec![
            constant(0, init),
            constant(8, bound),
            constant(4, four),
            loop_op,
            add(result, four, reader),
        ];

        let mut candidates = PeelingCandidates::default();
        candidates.insert_or_update(ForRef(iv), Peeling::FirstAndLastIter);
        perform_loop_peeling(&candidates, &mut unit, &mut values);

        assert_eq!(unit.len(), 11);
        // THE FIRST ITERATION IS THE HIGHEST BOUND — this island counts iterations DOWN — and its
        // copy reads the loop's own init, with `%iv < 4` decided false at 8 and its `then` gone.
        assert!(matches!(
            unit[3],
            Op::Sentient(sentient::Op::ScalarConstant { value: 8, .. })
        ));
        let Op::Sentient(sentient::Op::ScalarAdd {
            lhs: first_lhs,
            result: first_yielded,
            ..
        }) = unit[4]
        else {
            panic!("the first iteration's body is hoisted before the loop")
        };
        assert_eq!(first_lhs, init);
        // `orig_bound_val - 1 - 1`, and the loop reads what the first iteration yielded.
        let Op::Sentient(sentient::Op::ScalarConstant {
            value: 6,
            result: new_bound_val,
            ..
        }) = unit[5]
        else {
            panic!("the loop's new bound is built immediately before it")
        };
        let Op::Sentient(sentient::Op::For {
            bound: new_bound,
            carried: peeled_carried,
            dbg_name,
            body,
            ..
        }) = &unit[6]
        else {
            panic!("the loop follows its new bound")
        };
        assert_eq!(*new_bound, new_bound_val);
        assert_eq!(peeled_carried[0].init, first_yielded);
        assert_eq!(dbg_name.as_deref(), Some("MDLP(LFirstAndLastIter)"));
        // `%iv < 4` became `%iv < 3` on a fresh constant, and the original 4 is untouched.
        assert!(matches!(
            body[0],
            Op::Sentient(sentient::Op::ScalarConstant { value: 3, .. })
        ));
        assert!(matches!(
            body[1],
            Op::Sentient(sentient::Op::If { lhs, rhs, .. })
                if lhs == iv && rhs != four
        ));
        assert!(matches!(
            unit[2],
            Op::Sentient(sentient::Op::ScalarConstant { value: 4, .. })
        ));
        // The last iteration is bound 1, so `%iv < 4` held and its `then` branch survives; its body
        // reads the loop's result and the loop's original reader ends on what it yielded.
        assert!(matches!(
            unit[7],
            Op::Sentient(sentient::Op::ScalarConstant { value: 1, .. })
        ));
        assert_eq!(unit[8], nop);
        let Op::Sentient(sentient::Op::ScalarAdd {
            lhs: last_lhs,
            result: last_yielded,
            ..
        }) = unit[9]
        else {
            panic!("the last iteration's body is hoisted after the loop")
        };
        assert_eq!(last_lhs, result);
        assert!(matches!(
            unit[10],
            Op::Sentient(sentient::Op::ScalarAdd { lhs, .. }) if lhs == last_yielded
        ));
    }

    /// AN ESTIMATOR THAT STATES ITS TWO ANSWERS — the reference's `UnderEstimateInstructionEstimator`
    /// is out of campaign scope, and only these two of its methods are asked.
    struct StatedEstimator {
        original: InstructionCount,
        per_op: i32,
        ibuff: InstructionCount,
    }

    impl InstructionEstimator for StatedEstimator {
        fn recalculate(&mut self, _unit: &[Op]) {
            todo!("no run of this fixture recalculates")
        }

        fn estimated_instruction_count_of_op(&mut self, op: &Op) -> InstructionCount {
            // The first question is about the loop being peeled; every later one is about one op of
            // the dummy loop's body.
            if matches!(op, Op::Sentient(sentient::Op::For { .. })) && self.original.0 > 0 {
                let answer = self.original;
                self.original = InstructionCount(0);
                return answer;
            }
            InstructionCount(self.per_op)
        }

        fn estimated_instruction_count_of_region(&mut self, _region: &[Op]) -> InstructionCount {
            todo!("no run of this fixture counts a region")
        }

        fn have_ibuff_space(&mut self, _unit: &[Op]) -> bool {
            todo!("no run of this fixture asks whether the unit fits")
        }

        fn remaining_ibuff_space(&mut self, _unit: &[Op]) -> InstructionCount {
            self.ibuff
        }
    }

    /// An evaluator that knows nothing, so no simplification below turns on an offset it stated.
    struct BlindEvaluator;

    impl ExpressionEvaluator for BlindEvaluator {
        fn evaluate_value(&mut self, _value: Val) -> Evaluation {
            Evaluation {
                known_absolute: false,
                base: None,
                offsets: Offsets::PerUnit(Vec::new()),
            }
        }

        fn evaluate_sum(&mut self, _lhs: &Evaluation, _rhs: &Evaluation) -> Evaluation {
            todo!("no op of this fixture evaluates a sum")
        }

        fn build_offset_value(
            &mut self,
            _evaluation: &Evaluation,
            _sites: &mut OffsetSites<'_>,
            _walked: &mut Vec<Op>,
            _ty: ScalarTy,
        ) -> Val {
            todo!("no op of this fixture is known absolute")
        }
    }

    /// An SFP unit holding `body` — one of the three components the pass runs on.
    fn sfp_unit(body: Vec<Op>) -> ProgramUnit<Dd2> {
        ProgramUnit {
            on: Units::one(DfirUnit::Sfp, Val(900)),
            precision: None,
            body,
            arch: core::marker::PhantomData,
        }
    }

    /// e630's vendor shape — a loop of bound 8 whose body tests its own first iteration goes into the
    /// throwaway loop and is peeled there.
    ///
    /// ⛔ THE SPECULATION CANNOT BE SCORED YET: the third compression pass `run` runs over the dummy
    /// is [`run_op_rerolling`], and `e519_processOneBlock` reaches unported `e335` at the dummy body's
    /// first op — so what is observable is the state at that seam, with the dummy loop built beside an
    /// original that has not been replaced. ⭐ REVISIT THIS ASSERTION WITH e335.
    #[test]
    fn e630_peels_into_the_dummy_loop_and_stops_at_the_unported_rerolling() {
        let mut unit = sfp_unit(vec![
            constant(8, Val(1)),
            for_loop(
                Val(3),
                Val(1),
                Vec::new(),
                vec![
                    if_op(
                        sentient::CmpPredicate::Sge,
                        Val(3),
                        Val(1),
                        vec![add(Val(1), Val(1), Val(5))],
                        Vec::new(),
                    ),
                    Op::Sentient(sentient::Op::Yield {
                        results: Vec::new(),
                    }),
                ],
            ),
        ]);
        let mut preamble = Vec::new();
        let mut values = Values::default();
        for _ in 0..100 {
            values.mint();
        }

        let reached = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            run(
                ForRef(Val(3)),
                &mut unit,
                &mut preamble,
                &mut IvLoopInfo::default(),
                &mut StatedEstimator {
                    original: InstructionCount(100),
                    per_op: 1,
                    ibuff: InstructionCount(0),
                },
                &mut BlindEvaluator,
                &mut OutOfScopePropagationAnalysis,
                &OutOfScopeUnitIndexMap,
                &mut values,
            );
        }));

        assert!(
            reached.is_err(),
            "rerolling the dummy loop reaches unported e335"
        );
        // The dummy loop was built beside the original, and the original is still standing because
        // the profitability comparison is downstream of the seam.
        assert!(top_level_at(&unit.body, ForRef(Val(3))).is_some());
        assert_eq!(
            unit.body
                .iter()
                .filter(|op| matches!(op, Op::Sentient(sentient::Op::For { .. })))
                .count(),
            2,
            "the throwaway loop holding the peeled clone, and the original"
        );
    }

    /// e630's negative — ⛔ A LOOP WITH NOTHING TO PEEL LEAVES THE UNIT EXACTLY AS IT WAS: the dummy
    /// loop and its constant are both erased, so a non-candidate costs the unit nothing.
    #[test]
    fn e630_a_loop_with_no_candidate_leaves_the_unit_untouched() {
        let body = vec![
            constant(8, Val(1)),
            for_loop(
                Val(3),
                Val(1),
                Vec::new(),
                vec![
                    add(Val(1), Val(1), Val(5)),
                    Op::Sentient(sentient::Op::Yield {
                        results: Vec::new(),
                    }),
                ],
            ),
        ];
        let mut unit = sfp_unit(body.clone());
        let mut preamble = Vec::new();
        let mut values = Values::default();
        for _ in 0..100 {
            values.mint();
        }

        run(
            ForRef(Val(3)),
            &mut unit,
            &mut preamble,
            &mut IvLoopInfo::default(),
            &mut StatedEstimator {
                original: InstructionCount(100),
                per_op: 1,
                ibuff: InstructionCount(0),
            },
            &mut BlindEvaluator,
            &mut OutOfScopePropagationAnalysis,
            &OutOfScopeUnitIndexMap,
            &mut values,
        );

        assert_eq!(unit.body, body);
        assert!(preamble.is_empty());
    }
}

/// Replaces: e322_copyOneIter
///
/// Peels one iteration: clones the loop beside itself, pins the clone's IV to `iv_val`, rewires its
/// iter args onto `new_iter_args`, folds what pinning decided, hoists the clone's body into the
/// clone's place and leaves `new_iter_args` holding what that iteration yielded.
///
/// ⛔ TRAP: THE RESULT REWIRING IS IN TWO PARTS ON PURPOSE (`:484-493`) — the original's results go to
/// the clone's, and only the hoist sends those on to the yielded values, so an iter arg init value
/// that reads the original loop is NOT rewired to this iteration's value.
/// ⭐ DIVERGENCE: the `iv_val` constant lands immediately before the clone, not at `const_builder_`'s
/// program-unit top — [`static_operand`] resolves a definition in the block it is handed, as e095.
pub fn copy_one_iter(
    block: &mut Vec<Op>,
    at: usize,
    new_iter_args: &mut Vec<Val>,
    iv_val: i64,
    move_after: bool,
    values: &mut Values,
) {
    let Some(original @ Op::Sentient(sentient::Op::For { .. })) = block.get(at) else {
        return;
    };
    let original_results = dialects::results(original);
    let mut mapping = ValueMapping::new();
    let Some(cloned) =
        dialects::clone_ops(core::slice::from_ref(original), values, &mut mapping).pop()
    else {
        return;
    };
    let folded = values.mint();
    let constant = Op::Sentient(sentient::Op::ScalarConstant {
        value: iv_val,
        result: folded,
        reg_locale: sentient::RegType::Imm,
        // `iv.getType()` — a `sentient.for`'s induction variable is an `index`.
        ty: ScalarTy::Index,
        is_symbol: false,
    });
    // `OpBuilder builder(for_op)`, or `setInsertionPointAfter(for_op)` — with the pinned IV's
    // constant immediately before the clone either way.
    let const_at = at + usize::from(move_after);
    block.insert(const_at, constant);
    let cloned_at = const_at + 1;
    block.insert(cloned_at, cloned);

    let Some(Op::Sentient(sentient::Op::For { iv, carried, .. })) = block.get(cloned_at) else {
        return;
    };
    let cloned_iv = *iv;
    let cloned_args: Vec<Val> = carried.iter().map(|entry| entry.arg).collect();
    let cloned_results: Vec<Val> = carried.iter().map(|entry| entry.result).collect();

    dialects::replace_all_uses_with(block, cloned_iv, folded);
    if move_after {
        for (of, with) in original_results.iter().zip(&cloned_results) {
            dialects::replace_all_uses_with(block, *of, *with);
        }
    }
    // `DT_CHECK_MSG(new_iter_args.size() == cloned_for_op.getNumResults(), "Invalid set of iter arg
    // init values")` (`:498-499`) — the zip stops at the shorter, so a caller holding fewer inits
    // rewires fewer args rather than aborting.
    for (arg, init) in cloned_args.iter().zip(new_iter_args.iter()) {
        dialects::replace_all_uses_with(block, *arg, *init);
    }
    simplify_conditionals(IfOpsToSimplify::Reading(folded), block);

    let Some(Op::Sentient(sentient::Op::For { body, .. })) = block.get_mut(cloned_at) else {
        return;
    };
    let mut hoisted = core::mem::take(body);
    // `while (!body->empty())` BREAKS AT THE `sentient.yield`: whatever follows it dies with the loop.
    let mut yielded = None;
    if let Some(cut) = hoisted
        .iter()
        .position(|op| matches!(op, Op::Sentient(sentient::Op::Yield { .. })))
    {
        if let Op::Sentient(sentient::Op::Yield { results }) = &hoisted[cut] {
            yielded = Some(results.clone());
        }
        hoisted.truncate(cut);
    }
    let tail = block.split_off(cloned_at + 1);
    block.truncate(cloned_at);
    block.extend(hoisted);
    block.extend(tail);
    if let Some(results) = yielded {
        new_iter_args.clear();
        new_iter_args.extend(results);
        if move_after {
            for (of, with) in cloned_results.iter().zip(new_iter_args.iter()) {
                dialects::replace_all_uses_with(block, *of, *with);
            }
        }
    }
}

/// The `sentient.for` whose induction variable is `loop_ref`, wherever under `block` it sits.
///
/// ⭐ RE-DERIVED AFTER EVERY MUTATION. `copyOneIter` inserts a constant and a hoisted body around the
/// loop, so a [`RegionPath`] taken before it names a different op after it — an MLIR `ForOp` handle
/// survives that and a path does not.
fn for_path(block: &[Op], into: &[(usize, usize)], loop_ref: ForRef) -> Option<RegionPath> {
    for (at, op) in block.iter().enumerate() {
        if let Op::Sentient(sentient::Op::For { iv, .. }) = op
            && *iv == loop_ref.0
        {
            return Some(RegionPath {
                into: into.to_vec(),
                at,
            });
        }
        for (region_index, region) in dialects::regions_ref(op).into_iter().enumerate() {
            let mut deeper = into.to_vec();
            deeper.push((at, region_index));
            if let Some(found) = for_path(region, &deeper, loop_ref) {
                return Some(found);
            }
        }
    }
    None
}

/// `%c = sentient.scalar_constant {value} : ty`, as `const_builder_` creates one.
fn scalar_constant(value: i64, result: Val, ty: ScalarTy) -> Op {
    Op::Sentient(sentient::Op::ScalarConstant {
        value,
        result,
        reg_locale: sentient::RegType::Imm,
        ty,
        is_symbol: false,
    })
}

/// Replaces: e448_performLoopPeeling
///
/// Peels each candidate loop, INNERMOST FIRST: a copy of the body before the loop for the first
/// iteration, a copy after it for the last, the loop's bound reduced by however many were taken, and
/// the iterations chained through the loop's carried values.
///
/// ⛔ TRAP: THE ORDER IS THE REVERSE OF THE CANDIDATE LIST *because* peeling clones a body — a
/// candidate inside another must already be peeled when its enclosing loop is copied.
/// ⛔ TRAP: THE FIRST-ITERATION COPY IS PINNED TO `orig_bound_val`, NOT TO 1: this island counts a
/// loop's iterations DOWN, so the first iteration is the highest bound and the last is 1.
/// ⚠️ DIVERGENCE: the closing simplification runs over the whole unit rather than the loop's parent
/// op, because a nested block cannot resolve the constants above it — folding a statically decidable
/// `sentient.if` is the dialect's own canonicalisation and is valid wherever it applies.
pub fn perform_loop_peeling(
    candidates: &PeelingCandidates,
    unit: &mut Vec<Op>,
    values: &mut Values,
) {
    for &(loop_ref, peeling) in candidates.records().iter().rev() {
        // `DT_CHECK(p_type != kNoPeeling)` is [`Peeling`] itself, so there is no case here.
        let peel_first_iter = matches!(peeling, Peeling::FirstIterOnly | Peeling::FirstAndLastIter);
        let peel_last_iter = matches!(peeling, Peeling::LastIterOnly | Peeling::FirstAndLastIter);
        let Some(path) = for_path(unit, &[], loop_ref) else {
            continue;
        };
        let Some(scope) = walk_to(unit, &path.into) else {
            continue;
        };
        let Some(Op::Sentient(sentient::Op::For { bound, carried, .. })) = scope.get(path.at)
        else {
            continue;
        };
        // `DT_CHECK_MSG(const_bound_op, "Expect ForOp with constant bound")` read as a precondition.
        let Some(Op::Sentient(sentient::Op::ScalarConstant {
            value: orig_bound_val,
            ty: bound_ty,
            ..
        })) = dialects::defining_op(*bound, unit)
        else {
            continue;
        };
        let (orig_bound_val, bound_ty) = (*orig_bound_val, *bound_ty);
        let mut new_iter_args: Vec<Val> = carried.iter().map(|entry| entry.init).collect();

        if peel_first_iter {
            let Some(block) = block_at(unit, &path.into) else {
                continue;
            };
            copy_one_iter(
                block,
                path.at,
                &mut new_iter_args,
                orig_bound_val,
                false,
                values,
            );
        }

        // `orig_bound_val - (peel_first_iter ? 1 : 0) - (peel_last_iter ? 1 : 0)`.
        let new_bound_val = orig_bound_val - i64::from(peel_first_iter) - i64::from(peel_last_iter);
        let new_bound = values.mint();
        let Some(path) = for_path(unit, &[], loop_ref) else {
            continue;
        };
        let Some(block) = block_at(unit, &path.into) else {
            continue;
        };
        block.insert(path.at, scalar_constant(new_bound_val, new_bound, bound_ty));
        let at = path.at + 1;
        let Some(Op::Sentient(sentient::Op::For {
            bound,
            carried,
            dbg_name,
            ..
        })) = block.get_mut(at)
        else {
            continue;
        };
        *bound = new_bound;
        // `updateDbgName("MDLP(", for_op, stringifyPeelingType(p_type) + ")")` — ⛔ WRITES ONLY WHERE
        // A NAME ALREADY EXISTS (`dcc/src/Utils/Utils.cpp:491-496`).
        if let Some(name) = dbg_name {
            *name = format!("MDLP({name}{})", peeling.ty().spelling());
        }
        // `getInitArgsMutable().assign(new_iter_args)`, then `new_iter_args = for_op.getResults()`.
        for (entry, init) in carried.iter_mut().zip(&new_iter_args) {
            entry.init = *init;
        }
        new_iter_args = carried.iter().map(|entry| entry.result).collect();

        if peel_last_iter {
            copy_one_iter(block, at, &mut new_iter_args, 1, true, values);
            // The loop's bound has just been decremented, so every constant the induction variable
            // is compared against must be too.
            //
            // ⛔ THE SCOPE IS THE WHOLE UNIT, NOT THE LOOP'S BODY: `iv.getUsers()` and
            // `getDefiningOp()` are both global, and `const_builder_` puts the compared constant at
            // the program unit's top — a body-only scope finds no definition for it and answers "not
            // a comparison against a constant", which decrements nothing at all.
            if let Some(predicates) = PredicatesOnIv::of(loop_ref.0, unit) {
                decrement_predicates_on_iv(&predicates, values, unit);
            }
        }

        simplify_conditionals(IfOpsToSimplify::Every, unit);
    }
}

/// Replaces: e568_computePeelingInfo
///
/// Which loop one `sentient.if` lets us peel, and at which end: the comparison is read with the
/// induction variable on the left, and a constant one off either end names that end's iteration.
///
/// ⛔ TRAP: A BARE `sentient.for` IV COUNTS DOWN, so `first_val` is the loop's BOUND and `last_val`
/// is 1 — see [`crate::transform::sentient::utils::for_loop_info_if_iv`]. The `DT_CHECK_MSG(last_val
/// == 1, ..)` (`:381`) is unwritable here because `NormalizedIv::Rejected` guarantees it.
/// ⛔ TRAP: A BOUND OF 0 OR 1 IS REFUSED (`:371-373`) — `LightweightSimplifications` removes it.
pub fn compute_peeling_info(
    if_op: &Op,
    ivs: &mut IvLoopInfo,
    defs: Definitions<'_>,
) -> Option<(ForRef, Peeling)> {
    let Op::Sentient(sentient::Op::If {
        predicate,
        lhs,
        rhs,
        ..
    }) = if_op
    else {
        return None;
    };
    // The two attempts of `:355-369`. ⭐ `isa<BlockArgument>` GUARDS A NULL `getDefiningOp()` ONLY,
    // and a value with no definition in scope answers `None` here without the guard.
    let (info, const_value, rhs_is_const) = match (ivs.get_or_create(*lhs, defs), constant(*rhs, defs))
    {
        (Some(info), Some(value)) => (info, value, true),
        _ => match (ivs.get_or_create(*rhs, defs), constant(*lhs, defs)) {
            (Some(info), Some(value)) => (info, value, false),
            _ => return None,
        },
    };
    if info.lower_bound <= 1 {
        return None;
    }

    let (first_val, last_val) = (info.lower_bound, info.upper_bound);
    let predicate = if rhs_is_const {
        *predicate
    } else {
        reverse_predicate(*predicate)
    };
    use sentient::CmpPredicate::{Eq, Ne, Sge, Sgt, Sle, Slt};
    let peeling = match (const_value, predicate) {
        (v, Sge | Eq | Ne | Slt) if v == first_val => Peeling::FirstIterOnly,
        (v, Sgt | Sle) if v == first_val - 1 => Peeling::FirstIterOnly,
        (v, Sgt | Eq | Ne | Sle) if v == last_val => Peeling::LastIterOnly,
        (v, Sge | Slt) if v == last_val + 1 => Peeling::LastIterOnly,
        _ => return None,
    };
    Some((info.loop_op, peeling))
}

/// `dyn_cast<sentient::ConstantOp>(value.getDefiningOp()).getValue()`.
fn constant(value: Val, defs: Definitions<'_>) -> Option<i64> {
    match defs.of(value) {
        Some(Op::Sentient(sentient::Op::ScalarConstant { value, .. })) => Some(*value),
        _ => None,
    }
}

/// Every `sentient.if` under `block`, INNERMOST FIRST — `walk<WalkOrder::PostOrder>`.
///
/// ⛔ NOT [`if_paths`], WHICH IS PRE-ORDER: it pushes the op before descending, because its own reader
/// takes the greatest path first and gets post-order out of the [`RegionPath`] ordering instead.
fn if_ops_post_order<'a>(block: &'a [Op], out: &mut Vec<&'a Op>) {
    for op in block {
        for region in dialects::regions_ref(op) {
            if_ops_post_order(region, out);
        }
        if matches!(op, Op::Sentient(sentient::Op::If { .. })) {
            out.push(op);
        }
    }
}

/// Replaces: e604_findCandidates
///
/// Every peelable loop under `for_op`, found by asking each `sentient.if` inside it — innermost first
/// — which loop and which end its comparison names (`:402-431`).
///
/// ⛔ TRAP: THE `DT_CHECK_MSG(loop_to_peeling_type_.empty())` (`:405`) IS THE RETURN TYPE HERE. The
/// reference accumulates into a member the caller must have drained; this answers a fresh set, so the
/// caller's `!empty()` is `records().is_empty()` and there is no state to have failed to clear.
pub fn find_candidates(for_op: ForRef, unit: &[Op], ivs: &mut IvLoopInfo) -> PeelingCandidates {
    let mut candidates = PeelingCandidates::default();
    let Some(Op::Sentient(sentient::Op::For { body, .. })) = super::for_op_at(unit, for_op) else {
        return candidates;
    };
    // `getDefiningOp()` is global, so the constants a nested comparison reads are resolved against
    // the whole unit and not against the loop body the walk is scoped to.
    let regions: [&[Op]; 1] = [unit];
    let defs = Definitions::from_innermost(&regions);
    let mut if_ops = Vec::new();
    if_ops_post_order(body, &mut if_ops);
    for if_op in if_ops {
        if let Some((inner_for_op, peeling)) = compute_peeling_info(if_op, ivs, defs) {
            candidates.insert_or_update(inner_for_op, peeling);
        }
    }
    candidates
}

/// `SkipPeelingIfLeadsToMoreInstrs`, `cl::init(false)` (`:41-44`) — a `dcc-opt` flag, and this crate
/// has no flags.
const SKIP_PEELING_IF_LEADS_TO_MORE_INSTRS: bool = false;

/// The top-level index of the `sentient.for` `loop_ref` names.
fn top_level_at(block: &[Op], loop_ref: ForRef) -> Option<usize> {
    block.iter().position(
        |op| matches!(op, Op::Sentient(sentient::Op::For { iv, .. }) if *iv == loop_ref.0),
    )
}

/// `dcc::utils::getLoopNestLevel<sentient::ForOp>` (`dcc/src/Utils/Utils.cpp:211-220`) — how many
/// `sentient.for`s enclose this one, the outermost answering 0 and a loop not here `i64::MAX`.
fn loop_nest_level(loop_ref: ForRef, scope: &[Op], enclosing: i64) -> i64 {
    for op in scope {
        if let Op::Sentient(inner) = op {
            if let sentient::Op::For { iv, .. } = inner
                && *iv == loop_ref.0
            {
                return enclosing;
            }
            let deeper = enclosing + i64::from(matches!(inner, sentient::Op::For { .. }));
            for region in sentient::regions(inner) {
                let level = loop_nest_level(loop_ref, region, deeper);
                if level < i64::MAX {
                    return level;
                }
            }
        }
    }
    i64::MAX
}

/// Drops the dummy loop and its constant, whatever became of the nest inside it — the reference's
/// `dummy_loop->erase(); dummy_constant->erase();` (`:732-733`).
fn erase_dummy(block: &mut Vec<Op>, dummy_loop: ForRef, dummy_value: Val) {
    if let Some(at) = top_level_at(block, dummy_loop) {
        block.remove(at);
    }
    block.retain(
        |op| !matches!(op, Op::Sentient(sentient::Op::ScalarConstant { result, .. }) if *result == dummy_value),
    );
}

/// Replaces: e630_run
///
/// Peels one top-level loop SPECULATIVELY: a clone of it goes inside a throwaway loop, every peelable
/// loop of the clone is peeled and the result compressed, and the clone replaces the original only if
/// it costs no more instructions — or overruns by less than the IBUFF has left (`:607-734`).
///
/// ⛔⛔ THE DUMMY LOOP IS WHAT MAKES THE COMPARISON POSSIBLE AND ITS BOUND OF 2 IS LOAD-BEARING
/// (`:625-627`): at 0 or 1 [`run_light_weight_simplifications`] would delete the loop it is holding,
/// and its `sentient.yield` is what keeps the last iteration's results used.
/// ⛔ `orig_instr_count` IS TAKEN UP FRONT, unlike the reference (`:687-688`), because the four
/// compression passes below reach the whole unit here and would have counted the peeled loop.
/// ⚠️ DIVERGENCE, AND IT IS THE SAME ONE e448 MAKES: absorption, merging and simplification are
/// scoped to the unit body rather than to the dummy loop, since a nested block cannot resolve the
/// constants above it. Only [`RerollScope`] can state the reference's narrower scope, so it does.
/// ⭐ `llvm_unreachable("failed in simplifying code")` (`:670`) IS AN ABORT, which is what the
/// `panic!` is; [`Propagation::Failed`] is unreachable on the shipped simplification path.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run<
    A: Arch,
    I: InstructionEstimator,
    E: ExpressionEvaluator,
    P: PropagationAnalysis,
    U: UnitIndexMap,
>(
    outer_loop: ForRef,
    unit: &mut ProgramUnit<A>,
    preamble: &mut Vec<Op>,
    ivs: &mut IvLoopInfo,
    estimator: &mut I,
    evaluator: &mut E,
    propagation: &mut P,
    unit_index_map: &U,
    values: &mut Values,
) {
    let avail_ibuff_space = estimator.remaining_ibuff_space(&unit.body);
    let Some(at) = top_level_at(&unit.body, outer_loop) else {
        return;
    };
    let orig_instr_count = estimator.estimated_instruction_count_of_op(&unit.body[at]);

    // The dummy loop: a bound of 2, one dummy init per result of the loop being peeled, and a clone of
    // that loop yielding its results (`:610-653`).
    let Op::Sentient(sentient::Op::For { carried, .. }) = &unit.body[at] else {
        return;
    };
    let original_results: Vec<Val> = carried.iter().map(|entry| entry.result).collect();
    let dummy_value = values.mint();
    let dummy_carried: Vec<sentient::Carried> = carried
        .iter()
        .map(|entry| sentient::Carried {
            init: dummy_value,
            arg: values.mint(),
            result: values.mint(),
            // `outer_loop_.getRegLocales()` — the loop being peeled carries the locales.
            reg: entry.reg,
            program_header: false,
            element_size: entry.element_size,
        })
        .collect();
    let mut mapping = ValueMapping::default();
    let Some(cloned) =
        dialects::clone_ops(core::slice::from_ref(&unit.body[at]), values, &mut mapping).pop()
    else {
        return;
    };
    let Op::Sentient(sentient::Op::For {
        iv: cloned_iv,
        carried: cloned_carried,
        ..
    }) = &cloned
    else {
        return;
    };
    let cloned_loop = ForRef(*cloned_iv);
    let yielded: Vec<Val> = cloned_carried.iter().map(|entry| entry.result).collect();
    let dummy_loop = ForRef(values.mint());
    let dummy = Op::Sentient(sentient::Op::For {
        iv: dummy_loop.0,
        bound: dummy_value,
        bound_reg: None,
        carried: dummy_carried,
        dbg_name: None,
        body: vec![
            cloned,
            Op::Sentient(sentient::Op::Yield { results: yielded }),
        ],
    });
    // ⭐ THE CONSTANT LANDS IN THE UNIT BODY, not at `const_builder_`'s module-block position, for
    // e322's reason: a bound is resolved by [`dialects::defining_op`] against the block it is in.
    unit.body
        .insert(at, scalar_constant(2, dummy_value, ScalarTy::Index));
    unit.body.insert(at + 1, dummy);

    let mut candidates = find_candidates(cloned_loop, &unit.body, ivs);
    if candidates.records().is_empty() {
        erase_dummy(&mut unit.body, dummy_loop, dummy_value);
        return;
    }
    // `std::sort(.., getLoopNestLevel(a) < getLoopNestLevel(b))` — outermost first, which is the order
    // [`perform_loop_peeling`] consumes in REVERSE.
    candidates
        .0
        .sort_by_key(|&(loop_ref, _)| loop_nest_level(loop_ref, &unit.body, 0));
    perform_loop_peeling(&candidates, &mut unit.body, values);

    let mut consts = Vec::new();
    let simplified = run_light_weight_simplifications(
        &mut consts,
        &mut unit.body,
        evaluator,
        propagation,
        unit_index_map,
        values,
    );
    preamble.splice(0..0, consts);
    if simplified == Propagation::Failed {
        panic!("failed in simplifying code");
    }
    if let Some(dummy_at) = top_level_at(&unit.body, dummy_loop) {
        run_op_rerolling(
            unit,
            &RerollScope::Op(OpAt::top(InBlock(dummy_at))),
            MergeXrfIntoMac::No,
        );
    }
    run_loop_absorption(preamble, &mut unit.body, values);
    run_loop_merging(&mut unit.body, values);

    // Cost analysis: the dummy loop's body up to but not including its `sentient.yield` (`:690-695`).
    let Some(dummy_at) = top_level_at(&unit.body, dummy_loop) else {
        return;
    };
    let Op::Sentient(sentient::Op::For { body, .. }) = &unit.body[dummy_at] else {
        return;
    };
    let mut new_instr_count = InstructionCount(0);
    for op in body
        .iter()
        .take_while(|op| !matches!(op, Op::Sentient(sentient::Op::Yield { .. })))
    {
        new_instr_count.0 += estimator.estimated_instruction_count_of_op(op).0;
    }
    let profitable = new_instr_count <= orig_instr_count
        || (!SKIP_PEELING_IF_LEADS_TO_MORE_INSTRS
            && new_instr_count.0 - orig_instr_count.0 < avail_ibuff_space.0);
    if !profitable {
        // "Revert to the original by deleting dummy_loop" (`:726-727`).
        erase_dummy(&mut unit.body, dummy_loop, dummy_value);
        return;
    }

    // The original loop's results become what the dummy's terminator hands back, then the dummy's body
    // takes the original loop's place and the original goes (`:710-722`).
    let Op::Sentient(sentient::Op::For { body, .. }) = &mut unit.body[dummy_at] else {
        return;
    };
    let mut moved = core::mem::take(body);
    let terminator = moved.pop();
    if let Some(Op::Sentient(sentient::Op::Yield { results })) = &terminator {
        for (of, with) in original_results.iter().zip(results) {
            dialects::replace_all_uses_with(&mut unit.body, *of, *with);
        }
    }
    let Some(original_at) = top_level_at(&unit.body, outer_loop) else {
        return;
    };
    unit.body.splice(original_at..original_at, moved);
    if let Some(original_at) = top_level_at(&unit.body, outer_loop) {
        unit.body.remove(original_at);
    }
    erase_dummy(&mut unit.body, dummy_loop, dummy_value);
}
