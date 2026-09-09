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

use crate::islands::dataflow_ir::ty::ScalarTy;
use crate::islands::dataflow_ir::{ValueMapping, Values};
use crate::islands::sentient::dialects::{
    self, Op, StaticBranch, Val, replace_if_with_region, sentient, static_if_branch, static_operand,
};

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
        IfOpsToSimplify, PredicatesOnIv, copy_one_iter, decrement_predicates_on_iv,
        simplify_conditionals,
    };
    use crate::islands::dataflow_ir::Values;
    use crate::islands::dataflow_ir::ty::ScalarTy;
    use crate::islands::sentient::dialects::{Op, Val, sentient};

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

// crustify:todo: e448_performLoopPeeling
//   authority : dcc/src/Transform/Sentient/MultiDimLoopPeeling.cpp:543  (63 body lines, level 2)
//   original  : void LoopPeelingManager::performLoopPeeling()
//   calls     : e093_stringifyPeelingType, e095_decrementPredicatesOnIV, e096_simplifyConditionals, e322_copyOneIter

// crustify:todo: e568_computePeelingInfo
//   authority : dcc/src/Transform/Sentient/MultiDimLoopPeeling.cpp:351  (49 body lines, level 4)
//   original  : std::pair<sentient::ForOp, LoopPeelingManager::PeelingType> LoopPeelingManager::computePeelingInfo(sentient::IfOp if_op)
//   calls     : e245_reversePredicate, e515_getOrCreateTupleForIV

// crustify:todo: e604_findCandidates
//   authority : dcc/src/Transform/Sentient/MultiDimLoopPeeling.cpp:402  (30 body lines, level 5)
//   original  : bool LoopPeelingManager::findCandidates(sentient::ForOp for_op)
//   calls     : e093_stringifyPeelingType, e094_insertOrUpdatePeelingType, e568_computePeelingInfo

// crustify:todo: e630_run
//   authority : dcc/src/Transform/Sentient/MultiDimLoopPeeling.cpp:607  (130 body lines, level 6)
//   original  : void LoopPeelingManager::run()
//   calls     : e321_printLoopToPeelingType, e448_performLoopPeeling, e564_runLoopMerging, e571_runOpRerolling, e597_runLightWeightSimplifications, e600_runLoopAbsorption, e604_findCandidates
