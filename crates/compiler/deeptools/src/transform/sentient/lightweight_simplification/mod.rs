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

//! `LightweightSimplification.cpp` — 5 of the campaign's 656 units (dependency level(s) [0, 6]).
//!
//! | unit | entry | level | lines | authority path:line |
//! |---|---|---|---|---|
//! | `e052_addOrSubWithZeroSimplification` | 052 | 0 | 22 | `dcc/src/Transform/Sentient/LightweightSimplification.cpp:51` |
//! | `e053_constantValueSimplification` | 053 | 0 | 11 | `dcc/src/Transform/Sentient/LightweightSimplification.cpp:75` |
//! | `e054_coalesceScalarArithSimplification` | 054 | 0 | 62 | `dcc/src/Transform/Sentient/LightweightSimplification.cpp:90` |
//! | `e055_simplifyTrivialLoop` | 055 | 0 | 19 | `dcc/src/Transform/Sentient/LightweightSimplification.cpp:158` |
//! | `e623_runOnOperation` | 623 | 6 | 12 | `dcc/src/Transform/Sentient/LightweightSimplification.cpp:335` |

#![allow(dead_code)]
// ⛔ THE PASS IS NOT WIRED INTO THE PIPELINE YET — `e623_runOnOperation` (level 6) and its callee
// `e597_runLightWeightSimplifications` (level 5) are the units that call everything below, and
// neither is in this batch. ⭐ REMOVE THIS WITH e623.

use super::ForRef;
use super::analyses::{ExpressionEvaluator, OffsetSites, ScalarOffset};
use crate::formats::Bits;
use crate::islands::dataflow_ir::ty::ScalarTy;
use crate::islands::sentient::dialects::sentient as ops;
use crate::islands::sentient::dialects::{Op, Val, defining_op, replace_all_uses_with, use_count};

pub(crate) mod sentient;

/// WHETHER A PATTERN FIRED — `LogicalResult`, which here reports APPLICABILITY, not an error.
///
/// ⛔⛔ DECLINING IS NOT FAILING, AND THE CRATE FORBIDS SPELLING IT AS AN ERROR ANYWAY.
/// `runLightWeightSimplifications` tries the three arithmetic patterns in turn and moves to the next
/// on `failure()` (`LightweightSimplification.cpp:283-291`), so a `Result` here would refuse a pass
/// the reference completes. Same shape as
/// [`YieldRewrite`](crate::bridges::dataflow_ir_to_sentient::std_affine_to_standard).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub(crate) enum Simplified {
    /// `LogicalResult::success()` — the IR was rewritten.
    Rewritten,
    /// `LogicalResult::failure()` — this pattern does not apply; try the next one.
    DoesNotApply,
}

/// THE `isa<sentient::AddOp, sentient::SubOp>(op)` SHAPE all three arithmetic patterns open with —
/// the two operands, the result, and the register placement `setAttrs` copies.
#[derive(Debug, Clone, Copy)]
struct ScalarArith {
    /// `$inp1` — `getOperand(0)`.
    inp1: Val,
    /// `$inp2` — `getOperand(1)`.
    inp2: Val,
    /// `getResult(0)`.
    result: Val,
    /// `regLocale`/`regIndex` — what `op.getAttrs()` hands to a replacement.
    reg: Option<ops::Reg>,
    /// `result.getType()`.
    ty: ScalarTy,
    /// `element_size`, carried because `setAttrs(op.getAttrs())` copies the WHOLE dictionary.
    element_size: Option<Bits>,
    /// A `scalar_sub` rather than a `scalar_add`, which the zero pattern treats differently.
    subtracting: bool,
}

/// `isa<sentient::AddOp, sentient::SubOp>` and its operands, or `None` for any other op.
fn scalar_arith(op: &Op) -> Option<ScalarArith> {
    match op {
        Op::Sentient(ops::Op::ScalarAdd {
            lhs,
            rhs,
            result,
            reg,
            element_size,
            ty,
        }) => Some(ScalarArith {
            inp1: *lhs,
            inp2: *rhs,
            result: *result,
            reg: *reg,
            element_size: *element_size,
            ty: *ty,
            subtracting: false,
        }),
        Op::Sentient(ops::Op::ScalarSub {
            lhs,
            rhs,
            result,
            reg,
            element_size,
            ty,
        }) => Some(ScalarArith {
            inp1: *lhs,
            inp2: *rhs,
            result: *result,
            reg: *reg,
            element_size: *element_size,
            ty: *ty,
            subtracting: true,
        }),
        _ => None,
    }
}

/// `isConstantOp` (`LightweightSimplification.cpp:120-123`) — `getDefiningOp()` and
/// `isa<sentient::ConstantOp>`.
fn is_constant_op(value: Val, block: &[Op]) -> bool {
    matches!(
        defining_op(value, block),
        Some(Op::Sentient(ops::Op::ScalarConstant { .. }))
    )
}

/// `dcc::utils::isTargetConstant` (`Analyses/Utils.cpp:156-183`) — ⛔ OUT OF CAMPAIGN SCOPE.
///
/// ⛔ A NAMED DIVERGING FUNCTION RATHER THAN AN INLINE `todo!` SO NEITHER BRANCH OF THE CALLER'S `if`
/// IS LOST: an `if todo!() { .. } else { .. }` makes both arms `unreachable_code` and stops the
/// reader seeing which effect each one has. It reads a `sentient.scalar_constant`, an
/// `arith.constant` AND a `uniform.query_map`'s whole value list — the third is why it is not a
/// two-line match here.
fn is_target_constant(_value: Val, _target: ScalarOffset, _block: &[Op]) -> bool {
    todo!("dcc::utils::isTargetConstant (Analyses/Utils.cpp:156) — out of campaign scope")
}

/// Replaces: e052_addOrSubWithZeroSimplification
///
/// `a = 0 + b`, `a = b + 0` and `a = b - 0`: rewires every reader of `a` onto `b`.
///
/// ⛔ TRAP: a PER-UNIT evaluation is declined even when it is known absolute — the failed
/// `dyn_cast<AllUnitEvaluatedValue>` at `:58-59` is the second of three gates, not a formality.
/// ⛔ TRAP: a `scalar_sub` may only lose its SECOND operand; `0 - b` is not `b`.
pub(crate) fn add_or_sub_with_zero_simplification<E: ExpressionEvaluator>(
    evaluator: &mut E,
    block: &mut Vec<Op>,
    at: usize,
) -> Simplified {
    let Some(arith) = block.get(at).and_then(scalar_arith) else {
        return Simplified::DoesNotApply;
    };
    let operands = [arith.inp1, arith.inp2];
    let candidates: &[usize] = if arith.subtracting { &[1] } else { &[0, 1] };
    for &i in candidates {
        let evaluated = evaluator.evaluate_value(operands[i]);
        if !evaluated.known_absolute {
            continue;
        }
        if evaluated.all_unit_offset() != Some(ScalarOffset(0)) {
            continue;
        }
        replace_all_uses_with(block, arith.result, operands[1 - i]);
        return Simplified::Rewritten;
    }
    Simplified::DoesNotApply
}

/// Replaces: e053_constantValueSimplification
///
/// An add/sub whose RESULT is known absolute: materialises that constant and moves the readers onto it.
///
/// ⛔ TRAP: it evaluates the RESULT, not an operand — that is the whole difference from
/// [`add_or_sub_with_zero_simplification`], which shares the same first two lines.
pub(crate) fn constant_value_simplification<E: ExpressionEvaluator>(
    evaluator: &mut E,
    sites: &mut OffsetSites<'_>,
    block: &mut Vec<Op>,
    at: usize,
) -> Simplified {
    let Some(arith) = block.get(at).and_then(scalar_arith) else {
        return Simplified::DoesNotApply;
    };
    let evaluated = evaluator.evaluate_value(arith.result);
    if !evaluated.known_absolute {
        return Simplified::DoesNotApply;
    }
    let constant = evaluator.build_offset_value(&evaluated, sites, block, arith.ty);
    replace_all_uses_with(block, arith.result, constant);
    Simplified::Rewritten
}

/// Replaces: e054_coalesceScalarArithSimplification
///
/// `b = a + 10; x = b + 20` becomes `x2 = a + 30`, built BEFORE the op it replaces; `a + 0`
/// collapses onto `a` and `a` negated becomes a `scalar_sub` of the offset.
///
/// ⛔ TRAP: it only builds when an existing op will DIE to pay for the new one, and a CONSTANT
/// operand does not count as payment (`:125-129`) — so both [`use_count`]s are load-bearing.
/// ⛔ TRAP: `new_ops` collects the created ops' RESULTS. The block's dead-op sweep (`:308-311`)
/// reaches an op through the value it binds, which is the only identity an op has in this island.
pub(crate) fn coalesce_scalar_arith_simplification<E: ExpressionEvaluator>(
    evaluator: &mut E,
    sites: &mut OffsetSites<'_>,
    new_ops: &mut Vec<Val>,
    block: &mut Vec<Op>,
    at: usize,
) -> Simplified {
    let Some(arith) = block.get(at).and_then(scalar_arith) else {
        return Simplified::DoesNotApply;
    };
    let evaluated = evaluator.evaluate_value(arith.result);
    let Some(base) = evaluated.base else {
        return Simplified::DoesNotApply;
    };
    // Only coalesce if the add op will have a new base.
    if base.value == arith.result || base.value == arith.inp1 || base.value == arith.inp2 {
        return Simplified::DoesNotApply;
    }
    // Only transform if an op will be removed to compensate for the new one.
    let pays = |operand: Val, block: &[Op]| {
        use_count(operand, block) == 1 && !is_constant_op(operand, block)
    };
    if !pays(arith.inp1, block) && !pays(arith.inp2, block) {
        return Simplified::DoesNotApply;
    }
    let offset = evaluator.build_offset_value(&evaluated, sites, block, arith.ty);
    if !base.negated && is_target_constant(offset, ScalarOffset(0), block) {
        // x2 = a + 0 => replace x2 with a
        replace_all_uses_with(block, arith.result, base.value);
        return Simplified::Rewritten;
    }
    let created = sites.values.mint();
    // `SubOp::create(builder, loc, ty, offset, bv.value_)` — the OFFSET is the minuend when the base
    // is negated; the addition takes the base first. `OpBuilder builder(&op)` inserts before the op,
    // and `setAttrs(op.getAttrs())` copies its register placement onto the replacement.
    let new_op = if base.negated {
        ops::Op::ScalarSub {
            lhs: offset,
            rhs: base.value,
            result: created,
            reg: arith.reg,
            element_size: arith.element_size,
            ty: arith.ty,
        }
    } else {
        ops::Op::ScalarAdd {
            lhs: base.value,
            rhs: offset,
            result: created,
            reg: arith.reg,
            element_size: arith.element_size,
            ty: arith.ty,
        }
    };
    block.insert(at, Op::Sentient(new_op));
    new_ops.push(created);
    replace_all_uses_with(block, arith.result, created);
    Simplified::Rewritten
}

/// `body->getTerminator()->getOperand(index)` — what a `sentient.for` hands back for one iter arg.
///
/// ⭐ ABSENT MEANS THE LOOP CARRIES NOTHING BACK, not a refusal: `ForOp`'s verifier requires one
/// yield operand per iter arg, so a body with no `sentient.yield` has no iter args to rewire.
fn yielded_value(op: &Op, index: usize) -> Option<Val> {
    let Op::Sentient(ops::Op::For { body, .. }) = op else {
        return None;
    };
    match body.last() {
        Some(Op::Sentient(ops::Op::Yield { results })) => results.get(index).copied(),
        _ => None,
    }
}

/// THE BLOCK A `sentient.for` SITS IN AND ITS POSITION IN IT — the parent pointer this island's
/// trees do not have, supplied by the caller's scope instead. [`ForRef`] names the loop by its
/// induction variable.
fn loop_site(scope: &mut Vec<Op>, loop_ref: ForRef) -> Option<(&mut Vec<Op>, usize)> {
    let here = scope
        .iter()
        .position(|op| matches!(op, Op::Sentient(ops::Op::For { iv, .. }) if *iv == loop_ref.0));
    if let Some(at) = here {
        return Some((scope, at));
    }
    for op in scope.iter_mut() {
        match op {
            Op::Sentient(inner) => {
                for region in ops::regions_mut(inner) {
                    if let Some(site) = loop_site(region, loop_ref) {
                        return Some(site);
                    }
                }
            }
            Op::AffineFor(loop_op) => {
                if let Some(site) = loop_site(&mut loop_op.body, loop_ref) {
                    return Some(site);
                }
            }
            // ⭐ A LOWER-RUNG REGION CANNOT HOLD A `sentient.for`: it is typed with the rung below's
            // `Op`, which has no `Sentient` arm.
            _ => {}
        }
    }
    None
}

/// Replaces: e055_simplifyTrivialLoop
///
/// A bound-1 `sentient.for`: iter args become their inits, the loop's results become the yielded
/// values, the induction variable becomes the bound, and the body is hoisted out ahead of the loop.
///
/// ⛔ TRAP: it does NOT erase the emptied loop — the caller does, right after (`:225-228`).
/// ⛔ TRAP: the yield operand is re-read PER ITER ARG because an earlier arg's rewrite can change
/// it; `sentient.yield %arg2, %arg1` is why the two replacements interleave rather than run as two
/// passes. ⭐ `DoesNotApply` means the scope does not hold that loop; the reference returns `void`
/// because its caller hands it a `ForOp` it collected moments earlier (`:188-223`).
pub(crate) fn simplify_trivial_loop(block: &mut Vec<Op>, loop_ref: ForRef) -> Simplified {
    let Some((scope, at)) = loop_site(block, loop_ref) else {
        return Simplified::DoesNotApply;
    };
    let Op::Sentient(ops::Op::For {
        iv, bound, carried, ..
    }) = &scope[at]
    else {
        return Simplified::DoesNotApply;
    };
    let (induction_var, bound) = (*iv, *bound);
    let carried = carried.clone();
    for (index, entry) in carried.iter().enumerate() {
        replace_all_uses_with(scope, entry.arg, entry.init);
        if let Some(yielded) = yielded_value(&scope[at], index) {
            replace_all_uses_with(scope, entry.result, yielded);
        }
    }
    replace_all_uses_with(scope, induction_var, bound);

    // Hoist the body's operations out of the loop, in order, stopping at the terminator.
    let mut hoisted = Vec::new();
    if let Op::Sentient(ops::Op::For { body, .. }) = &mut scope[at] {
        while let Some(front) = body.first() {
            if matches!(front, Op::Sentient(ops::Op::Yield { .. })) {
                break;
            }
            hoisted.push(body.remove(0));
        }
    }
    for (offset, op) in hoisted.into_iter().enumerate() {
        scope.insert(at + offset, op);
    }
    Simplified::Rewritten
}

// crustify:todo: e623_runOnOperation
//   authority : dcc/src/Transform/Sentient/LightweightSimplification.cpp:335  (12 body lines, level 6)
//   original  : void LightweightSimplificationPass::runOnOperation()
//   calls     : e597_runLightWeightSimplifications

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::islands::dataflow_ir::Values;
    use crate::islands::sentient::dialects::sentient::{Carried, Reg, RegIndex, RegType};
    use crate::transform::sentient::analyses::{BaseValue, Evaluation, Offsets};

    /// AN EVALUATOR THAT ANSWERS WHAT THE TEST SAYS. ⛔ The analysis behind
    /// [`ExpressionEvaluator`] is out of campaign scope, so a test STATES its answers rather than
    /// deriving them — the effect under test is what the pattern does with an answer, not the answer.
    struct StatedEvaluator {
        answers: Vec<(Val, Evaluation)>,
        offset: Val,
    }

    impl ExpressionEvaluator for StatedEvaluator {
        fn evaluate_value(&mut self, value: Val) -> Evaluation {
            self.answers.iter().find(|(of, _)| *of == value).map_or(
                Evaluation {
                    known_absolute: false,
                    base: None,
                    offsets: Offsets::PerUnit(Vec::new()),
                },
                |(_, evaluated)| evaluated.clone(),
            )
        }

        fn evaluate_sum(&mut self, _lhs: &Evaluation, _rhs: &Evaluation) -> Evaluation {
            todo!("no unit of this batch evaluates a sum")
        }

        fn build_offset_value(
            &mut self,
            _evaluation: &Evaluation,
            _sites: &mut OffsetSites<'_>,
            _walked: &mut Vec<Op>,
            _ty: ScalarTy,
        ) -> Val {
            self.offset
        }
    }

    fn add(lhs: Val, rhs: Val, result: Val) -> Op {
        Op::Sentient(ops::Op::ScalarAdd {
            lhs,
            rhs,
            result,
            reg: None,
            ty: ScalarTy::Index,
            element_size: None,
        })
    }

    fn constant(value: i64, result: Val) -> Op {
        Op::Sentient(ops::Op::ScalarConstant {
            value,
            result,
            reg_locale: RegType::Imm,
            ty: ScalarTy::Index,
            is_symbol: false,
        })
    }

    fn absolute(offset: ScalarOffset) -> Evaluation {
        Evaluation {
            known_absolute: true,
            base: None,
            offsets: Offsets::AllUnit(offset),
        }
    }

    /// A minter that has already issued `issued` values, so a fixture's hand-written [`Val`]s and a
    /// created op's cannot collide.
    fn values_after(issued: u32) -> Values {
        let mut values = Values::default();
        for _ in 0..issued {
            values.mint();
        }
        values
    }

    /// `a = 0 + b` => every reader of `a` reads `b` (`LightweightSimplification.cpp:65-68`).
    #[test]
    fn a_zero_addend_rewires_the_readers_onto_the_other_operand() {
        let mut block = vec![
            constant(0, Val(0)),
            add(Val(0), Val(1), Val(2)),
            add(Val(2), Val(1), Val(3)),
        ];
        let mut evaluator = StatedEvaluator {
            answers: vec![(Val(0), absolute(ScalarOffset(0)))],
            offset: Val(9),
        };
        assert_eq!(
            add_or_sub_with_zero_simplification(&mut evaluator, &mut block, 1),
            Simplified::Rewritten
        );
        assert_eq!(block[2], add(Val(1), Val(1), Val(3)));
    }

    /// ⛔ THE NEGATIVE THE `dyn_cast` AT `:58-59` IS THERE FOR: a known-absolute PER-UNIT value is
    /// not an `AllUnitEvaluatedValue`, so the zero pattern declines and the IR is untouched.
    #[test]
    fn a_per_unit_zero_is_declined_even_though_it_is_known_absolute() {
        let mut block = vec![
            constant(0, Val(0)),
            add(Val(0), Val(1), Val(2)),
            add(Val(2), Val(1), Val(3)),
        ];
        let before = block.clone();
        let mut evaluator = StatedEvaluator {
            answers: vec![(
                Val(0),
                Evaluation {
                    known_absolute: true,
                    base: None,
                    offsets: Offsets::PerUnit(vec![(Val(0), ScalarOffset(0))]),
                },
            )],
            offset: Val(9),
        };
        assert_eq!(
            add_or_sub_with_zero_simplification(&mut evaluator, &mut block, 1),
            Simplified::DoesNotApply
        );
        assert_eq!(block, before);
    }

    /// An add whose RESULT is known absolute: the readers move onto the materialised constant.
    #[test]
    fn a_known_absolute_result_becomes_its_constant() {
        let mut block = vec![add(Val(0), Val(1), Val(2)), add(Val(2), Val(1), Val(3))];
        let mut consts = Vec::new();
        let mut values = values_after(10);
        let mut sites = OffsetSites {
            consts: &mut consts,
            query_maps: None,
            values: &mut values,
        };
        let mut evaluator = StatedEvaluator {
            answers: vec![(Val(2), absolute(ScalarOffset(30)))],
            offset: Val(9),
        };
        assert_eq!(
            constant_value_simplification(&mut evaluator, &mut sites, &mut block, 0),
            Simplified::Rewritten
        );
        assert_eq!(block[1], add(Val(9), Val(1), Val(3)));
    }

    /// A NEGATED BASE BUILDS A `scalar_sub` OF THE OFFSET, before the op it replaces, carrying that
    /// op's register placement — `setAttrs(op.getAttrs())` (`:134-139`).
    #[test]
    fn a_negated_base_coalesces_into_a_scalar_sub_that_keeps_the_register() {
        let placed = Reg {
            locale: RegType::Lrf,
            index: Some(RegIndex::at::<1>()),
        };
        let mut block = vec![
            constant(10, Val(0)),
            add(Val(5), Val(0), Val(2)),
            Op::Sentient(ops::Op::ScalarAdd {
                lhs: Val(2),
                rhs: Val(4),
                result: Val(3),
                reg: Some(placed),
                ty: ScalarTy::Index,
                element_size: None,
            }),
            add(Val(3), Val(4), Val(8)),
        ];
        let mut consts = Vec::new();
        let mut values = values_after(10);
        let mut sites = OffsetSites {
            consts: &mut consts,
            query_maps: None,
            values: &mut values,
        };
        let mut new_ops = Vec::new();
        let mut evaluator = StatedEvaluator {
            answers: vec![(
                Val(3),
                Evaluation {
                    known_absolute: false,
                    base: Some(BaseValue {
                        value: Val(5),
                        negated: true,
                    }),
                    offsets: Offsets::PerUnit(Vec::new()),
                },
            )],
            offset: Val(9),
        };
        assert_eq!(
            coalesce_scalar_arith_simplification(
                &mut evaluator,
                &mut sites,
                &mut new_ops,
                &mut block,
                2
            ),
            Simplified::Rewritten
        );
        assert_eq!(new_ops, vec![Val(10)]);
        assert_eq!(
            block[2],
            Op::Sentient(ops::Op::ScalarSub {
                lhs: Val(9),
                rhs: Val(5),
                result: Val(10),
                reg: Some(placed),
                ty: ScalarTy::Index,
                element_size: None,
            })
        );
        // The reader of the coalesced result now reads the new op's.
        assert_eq!(block[4], add(Val(10), Val(4), Val(8)));
    }

    /// A bound-1 loop: the body is hoisted ahead of it with the iter arg and induction variable
    /// substituted, the loop's result becomes the yielded value, and ⛔ THE LOOP IS STILL THERE.
    #[test]
    fn a_trivial_loop_hoists_its_body_and_hands_back_what_it_yielded() {
        let carried = Carried {
            init: Val(1),
            arg: Val(3),
            result: Val(4),
            reg: Reg {
                locale: RegType::Lrf,
                index: None,
            },
            program_header: false,
            element_size: None,
        };
        let mut block = vec![
            constant(1, Val(0)),
            constant(5, Val(1)),
            Op::Sentient(ops::Op::For {
                iv: Val(2),
                bound: Val(0),
                bound_reg: None,
                carried: vec![carried],
                dbg_name: None,
                body: vec![
                    add(Val(3), Val(2), Val(5)),
                    Op::Sentient(ops::Op::Yield {
                        results: vec![Val(5)],
                    }),
                ],
            }),
            add(Val(4), Val(4), Val(6)),
        ];
        assert_eq!(
            simplify_trivial_loop(&mut block, ForRef(Val(2))),
            Simplified::Rewritten
        );
        // The body op, hoisted, reading the init and the bound instead of the arg and the iv.
        assert_eq!(block[2], add(Val(1), Val(0), Val(5)));
        assert_eq!(
            block[3],
            Op::Sentient(ops::Op::For {
                iv: Val(2),
                bound: Val(0),
                bound_reg: None,
                carried: vec![carried],
                dbg_name: None,
                body: vec![Op::Sentient(ops::Op::Yield {
                    results: vec![Val(5)],
                })],
            })
        );
        // The reader of the loop's result now reads what the yield handed back.
        assert_eq!(block[4], add(Val(5), Val(5), Val(6)));
    }
}
