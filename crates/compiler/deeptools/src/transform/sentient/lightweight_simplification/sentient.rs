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

//! `LightweightSimplification.cpp` — 1 of the campaign's 656 units (dependency level(s) [5]).
//!
//! | unit | entry | level | lines | authority path:line |
//! |---|---|---|---|---|
//! | `e597_runLightWeightSimplifications` | 597 | 5 | 144 | `dcc/src/Transform/Sentient/LightweightSimplification.cpp:180` |


// ⛔ THE PASS IS NOT WIRED INTO THE PIPELINE YET — `e623_runOnOperation` (level 6) is what calls
// `run_light_weight_simplifications` below, and it is not in this batch. ⭐ REMOVE THIS WITH e623.
#![allow(dead_code)]

use std::collections::BTreeSet;

use super::{
    Simplified, add_or_sub_with_zero_simplification, coalesce_scalar_arith_simplification,
    constant_value_simplification, loop_site, simplify_trivial_loop,
};
use crate::islands::dataflow_ir::Values;
use crate::islands::sentient::dialects::sentient as ops;
use crate::islands::sentient::dialects::{
    Op, UniformRegions, Val, defining_op, regions_mut, replace_all_uses_with, results, use_count,
};
use crate::transform::sentient::analyses::{
    ExpressionEvaluator, OffsetSites, PropagationAnalysis, UnitIndexMap,
};
use crate::transform::sentient::enhanced_dead_variable_elimination::{
    EnhancedDeadVariableElimination, may_have_side_effects,
};
use crate::transform::sentient::scalar_simplifications::sentient::{
    Propagation, run_old_light_weight_simplifications,
};
use crate::transform::sentient::utils::{
    RedundantIterArgs, find_and_replace_redundant_iter_args_used_in_conditions,
};
use crate::transform::sentient::{ForRef, IterArgIndex};

/// `-dcc-old-lightweight-simplification`, `cl::init(false)` (`:42-47`) — a `dcc-opt` flag, not a
/// program property, and this crate has no flags.
const OLD_LIGHTWEIGHT_SIMPLIFICATION: bool = false;

/// `Operation *` — WHAT AN OP IS STILL CALLED after a rewrite has moved it: the values it binds, plus
/// a `sentient.for`'s induction variable, which is the only name a loop carrying nothing has.
///
/// ⭐ AN OP THAT BINDS NOTHING AND IS NOT A LOOP HAS NO IDENTITY HERE, AND NEEDS NONE: all three
/// arithmetic patterns open on a result, and every op [`may_have_side_effects`] exempts binds exactly
/// one value — so a nameless op is neither rewritten nor swept.
#[derive(Debug, Clone, PartialEq, Eq)]
struct OpIdent {
    /// `getResults()`.
    binds: Vec<Val>,
    /// `getInductionVar()`, for a `sentient.for`.
    iv: Option<Val>,
}

/// `&*I` — one entry of `ops_list`, or `None` for an op the walk cannot name.
fn ident(op: &Op) -> Option<OpIdent> {
    let iv = match op {
        Op::Sentient(ops::Op::For { iv, .. }) => Some(*iv),
        _ => None,
    };
    let binds = results(op);
    if binds.is_empty() && iv.is_none() {
        return None;
    }
    Some(OpIdent { binds, iv })
}

/// WHERE THE NAMED OP SITS NOW — re-found per use, because e053, e054 and e546 all insert ops into the
/// block being walked and every position after the insertion moves.
fn position_of(block: &[Op], of: &OpIdent) -> Option<usize> {
    block.iter().position(|op| ident(op).as_ref() == Some(of))
}

/// `op->walk([&](sentient::ForOp for_op) { .. })` — every loop of the unit, innermost first.
///
/// ⚠️ `WalkResult::skip()` AT `:206` AND `:216` IS INERT: `Operation::walk` defaults to
/// `WalkOrder::PostOrder`, which has already visited the regions the skip asks it not to enter. The
/// outcome is unchanged — an inner loop of a zero-bound loop is simplified into a loop that is then
/// erased whole.
fn for_refs_post_order(scope: &[Op], out: &mut Vec<ForRef>) {
    for op in scope {
        for region in crate::islands::sentient::dialects::regions_ref(op) {
            for_refs_post_order(region, out);
        }
        if let Op::Sentient(ops::Op::For { iv, .. }) = op {
            out.push(ForRef(*iv));
        }
    }
}

/// `replaceLoopByIterArgInit` (`:191-200`) MINUS ITS MARK — every reader of a result reads the
/// matching init, and the caller records the loop in `to_delete` rather than writing `TO_DELETE` on it.
///
/// ⭐ THE MARK NEVER LEAVES THE FUNCTION: `:229-231` is the only reader and it erases what it finds,
/// so the attribute is a local worklist and not something the island has to be able to hold.
fn replace_loop_by_iter_arg_init(unit_body: &mut [Op], carried: &[ops::Carried]) {
    for entry in carried {
        replace_all_uses_with(unit_body, entry.result, entry.init);
    }
}

/// `for_op->erase()` FOR A LOOP NAMED BY ITS INDUCTION VARIABLE — [`loop_site`] is the parent pointer
/// this island's trees do not have.
fn erase_loop(unit_body: &mut Vec<Op>, loop_ref: ForRef) {
    if let Some((scope, at)) = loop_site(unit_body, loop_ref) {
        scope.remove(at);
    }
}

/// `op->walk<WalkOrder::PostOrder>([&](uniform::UniformizeRegionsOp regions_op) { .. })` (`:233-247`).
///
/// ⛔ A `uniform.uniformize_regions` WITH NO REGIONS AT ALL IS ERASED: `is_empty` starts `true` and
/// only a region holding more than its terminator clears it, so the empty `for` never runs.
/// ⛔ `uniform.equalize_pattern` IS NOT TOUCHED — the walk is typed on the other op.
fn erase_empty_uniformize_regions(scope: &mut Vec<Op>) {
    for op in scope.iter_mut() {
        for region in regions_mut(op) {
            erase_empty_uniformize_regions(region);
        }
    }
    scope.retain(|op| {
        let Op::UniformRegions(UniformRegions::UniformizeRegions { regions, .. }) = op else {
            return true;
        };
        !regions.iter().all(|region| region.body.len() == 1)
    });
}

/// `operation->use_empty() && !dcc::utils::mayHaveSideEffects(operation)` (`:308-311`), applied to the
/// op the identity still names.
fn erase_if_dead(block: &mut Vec<Op>, of: &OpIdent) {
    let Some(at) = position_of(block, of) else {
        return;
    };
    if may_have_side_effects(&block[at]) {
        return;
    }
    let binds = results(&block[at]);
    if binds.iter().any(|bound| use_count(*bound, block) > 0) {
        return;
    }
    block.remove(at);
}

/// ONE BLOCK OF `op->walk([&](Block* block) { .. })` (`:258-324`): the three arithmetic patterns over
/// its ops from the end, then the dead-op sweep, then the loops that lost an iter arg.
fn simplify_one_block<E: ExpressionEvaluator>(
    block: &mut Vec<Op>,
    consts: &mut Vec<Op>,
    evaluator: &mut E,
    values: &mut Values,
) {
    // `for (auto I = block->rbegin(), E = block->rend(); I != E; ++I)` — LAST OP FIRST, "to prefer to
    // simplify operations at the end of blocks first" (`:265-267`).
    let ops_list: Vec<OpIdent> = block.iter().rev().filter_map(ident).collect();
    if ops_list.is_empty() {
        return;
    }
    let mut new_ops: Vec<Val> = Vec::new();
    let mut for_ops_to_be_updated: Vec<(ForRef, Vec<IterArgIndex>)> = Vec::new();
    for named in &ops_list {
        let Some(at) = position_of(block, named) else {
            continue;
        };
        let rewritten = {
            let mut sites = OffsetSites {
                consts: &mut *consts,
                query_maps: None,
                values: &mut *values,
            };
            add_or_sub_with_zero_simplification(evaluator, block, at) == Simplified::Rewritten
                || constant_value_simplification(evaluator, &mut sites, block, at)
                    == Simplified::Rewritten
                || coalesce_scalar_arith_simplification(
                    evaluator,
                    &mut sites,
                    &mut new_ops,
                    block,
                    at,
                ) == Simplified::Rewritten
        };
        if rewritten {
            continue;
        }
        if let Some(iv) = named.iv {
            let mut indices = Vec::new();
            let found = find_and_replace_redundant_iter_args_used_in_conditions(
                block,
                ForRef(iv),
                values,
                &mut indices,
            );
            if found == RedundantIterArgs::ToDelete {
                for_ops_to_be_updated.push((ForRef(iv), indices));
            }
        }
    }

    // ⭐ THE SWEEP RUNS FROM THE BLOCK'S END, because `ops_list` is the reversed block and erasing a
    // reader is what makes the op ahead of it dead.
    for named in &ops_list {
        erase_if_dead(block, named);
    }
    for created in &new_ops {
        erase_if_dead(
            block,
            &OpIdent {
                binds: vec![*created],
                iv: None,
            },
        );
    }

    // `EnhancedDeadVariableEliminationPass eve(opts)` (`:316-322`) — a FRESH pass, so its dataflow
    // state is empty and e498 deletes the positions e546 named rather than its own influence answers.
    if for_ops_to_be_updated.is_empty() {
        return;
    }
    let eve = EnhancedDeadVariableElimination::default();
    for (loop_ref, indices) in &for_ops_to_be_updated {
        let is_it =
            |op: &Op| matches!(op, Op::Sentient(ops::Op::For { iv, .. }) if *iv == loop_ref.0);
        let Some(at) = block.iter().position(is_it) else {
            continue;
        };
        let deleted: BTreeSet<usize> = indices.iter().map(|index| index.0 as usize).collect();
        eve.update_for_operation(block, at, &deleted);
    }
}

/// `op->walk([&](Block* block) { .. })` — POST-ORDER over blocks, so a nested block is simplified
/// before the block that holds it.
fn simplify_blocks<E: ExpressionEvaluator>(
    walked: &mut Vec<Op>,
    consts: &mut Vec<Op>,
    evaluator: &mut E,
    values: &mut Values,
) {
    for at in 0..walked.len() {
        for region in regions_mut(&mut walked[at]).iter_mut() {
            simplify_blocks(region, consts, evaluator, values);
        }
    }
    simplify_one_block(walked, consts, evaluator, values);
}

/// Replaces: e597_runLightWeightSimplifications
///
/// One program unit, in five sweeps (`:180-325`): loops whose body is empty or whose bound is zero
/// hand their results to their inits and go, bound-one loops are hoisted out and go, emptied
/// `uniform.uniformize_regions` go, and then every block is simplified from its end and swept clean.
///
/// ⛔ THE FIRST SWEEP REWIRES AS IT WALKS AND ERASES ONLY AFTERWARDS (`:229-231`), which is
/// load-bearing between SIBLINGS: one loop's bound can be the loop ahead of it handing back an init.
/// ⛔ `op == for_op` (`:211`) IS ALWAYS FALSE — every caller in the tree hands this a
/// `dataflow.program_unit`, never a `sentient.for`.
/// ⚠️ DIVERGENCE: the query-map builder is positioned at `getLocalOrGlobalRegion(..)`'s front block
/// (`:277-281`); here it is the block being walked. Positioning a builder is the mechanism, the two
/// differ only for a block inside a loop or a conditional, and [`OffsetSites::query_maps`] cannot name
/// an ancestor of the walked block and the walked block at once.
pub(crate) fn run_light_weight_simplifications<
    E: ExpressionEvaluator,
    P: PropagationAnalysis,
    U: UnitIndexMap,
>(
    consts: &mut Vec<Op>,
    unit_body: &mut Vec<Op>,
    evaluator: &mut E,
    propagation: &mut P,
    unit_index_map: &U,
    values: &mut Values,
) -> Propagation {
    if OLD_LIGHTWEIGHT_SIMPLIFICATION {
        return run_old_light_weight_simplifications(
            unit_body,
            propagation,
            unit_index_map,
            values,
        );
    }

    // Size zero/one and empty loop simplifications.
    let mut loops = Vec::new();
    for_refs_post_order(unit_body, &mut loops);
    let mut size_one_loops = Vec::new();
    let mut to_delete = Vec::new();
    for loop_ref in loops {
        let Some((scope, at)) = loop_site(unit_body, loop_ref) else {
            continue;
        };
        let Op::Sentient(ops::Op::For {
            bound, carried, body, ..
        }) = &scope[at]
        else {
            continue;
        };
        let bound = *bound;
        let carried = carried.clone();
        // `for_op.getBody()->without_terminator().empty()`.
        let body_is_empty = body
            .iter()
            .all(|op| matches!(op, Op::Sentient(ops::Op::Yield { .. })));
        if body_is_empty {
            replace_loop_by_iter_arg_init(unit_body, &carried);
            to_delete.push(loop_ref);
            continue;
        }
        let Some(Op::Sentient(ops::Op::ScalarConstant { value, .. })) =
            defining_op(bound, unit_body)
        else {
            continue;
        };
        let value = *value;
        if value > 1 {
            continue;
        }
        if value == 0 {
            replace_loop_by_iter_arg_init(unit_body, &carried);
            to_delete.push(loop_ref);
        } else {
            // ⭐ A NEGATIVE BOUND LANDS HERE TOO, on the reference's own `else` (`:218-222`).
            size_one_loops.push(loop_ref);
        }
    }
    for loop_ref in size_one_loops {
        let _ = simplify_trivial_loop(unit_body, loop_ref);
        erase_loop(unit_body, loop_ref);
    }
    for loop_ref in to_delete {
        erase_loop(unit_body, loop_ref);
    }
    erase_empty_uniformize_regions(unit_body);

    // Arithmetic-based simplifications.
    simplify_blocks(unit_body, consts, evaluator, values);
    Propagation::Succeeded
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::formats::Bits;
    use crate::islands::dataflow_ir::ty::ScalarTy;
    use crate::islands::sentient::dialects::sentient::{Carried, Reg, RegType};
    use crate::islands::sentient::dialects::{LocalRegion, uniform};
    use crate::transform::sentient::analyses::{Evaluation, Offsets, OutOfScopeUnitIndexMap};

    /// AN EVALUATOR THAT KNOWS NOTHING — every value is non-absolute with no base, which is what makes
    /// all three arithmetic patterns decline so the sweeps under test are the only effect.
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

    /// A propagation analysis nothing asks anything — `OldLightweightSimplification` ships `false`.
    struct UnaskedPropagation;

    impl PropagationAnalysis for UnaskedPropagation {
        fn are_expressions_same(&mut self, _val1: Val, _val2: Val) -> bool {
            todo!("the old lightweight simplification is off")
        }
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

    /// ⭐ NOT ON [`may_have_side_effects`]'S EXEMPT LIST, so it anchors the fixture's live chain
    /// against the sweep.
    fn mul(lhs: Val, rhs: Val, result: Val) -> Op {
        Op::Sentient(ops::Op::ScalarMul {
            lhs,
            rhs,
            result,
            reg_locale: None,
            ty: ScalarTy::Index,
        })
    }

    fn carrying(init: Val, arg: Val, result: Val) -> Carried {
        Carried {
            init,
            arg,
            result,
            reg: Reg {
                locale: RegType::Lrf,
                index: None,
            },
            program_header: false,
            element_size: Some(Bits(16)),
        }
    }

    /// e597 — the four sweeps at once: a bound-one loop is hoisted out and erased, a bound-zero loop
    /// hands its result to its init and is erased, an emptied `uniform.uniformize_regions` goes, and
    /// the arithmetic sweep drops the now-unread zero constant from the block's END forward.
    #[test]
    fn e597_hoists_the_trivial_loop_folds_the_empty_one_and_sweeps_what_they_left() {
        let mut unit_body = vec![
            constant(1, Val(0)),
            constant(5, Val(1)),
            Op::Sentient(ops::Op::For {
                iv: Val(2),
                bound: Val(0),
                bound_reg: None,
                carried: vec![carrying(Val(1), Val(3), Val(4))],
                dbg_name: None,
                body: vec![
                    add(Val(3), Val(2), Val(5)),
                    Op::Sentient(ops::Op::Yield {
                        results: vec![Val(5)],
                    }),
                ],
            }),
            constant(0, Val(6)),
            Op::Sentient(ops::Op::For {
                iv: Val(7),
                bound: Val(6),
                bound_reg: None,
                carried: vec![carrying(Val(1), Val(8), Val(9))],
                dbg_name: None,
                body: vec![
                    add(Val(8), Val(8), Val(10)),
                    Op::Sentient(ops::Op::Yield {
                        results: vec![Val(10)],
                    }),
                ],
            }),
            Op::UniformRegions(UniformRegions::UniformizeRegions {
                regions: vec![LocalRegion {
                    arg: Val(11),
                    units: vec![Val(12)],
                    body: vec![Op::Uniform(uniform::Op::Yield {
                        operands: Vec::new(),
                    })],
                }],
                results: Vec::new(),
                yielded: Vec::new(),
            }),
            mul(Val(4), Val(9), Val(13)),
        ];
        let mut consts = Vec::new();
        let mut values = Values::default();
        for _ in 0..14 {
            let _ = values.mint();
        }

        let ran = run_light_weight_simplifications(
            &mut consts,
            &mut unit_body,
            &mut BlindEvaluator,
            &mut UnaskedPropagation,
            &OutOfScopeUnitIndexMap,
            &mut values,
        );

        assert_eq!(ran, Propagation::Succeeded);
        assert!(consts.is_empty());
        assert_eq!(
            unit_body,
            vec![
                constant(1, Val(0)),
                constant(5, Val(1)),
                // The trivial loop's body, hoisted, reading the init and the bound.
                add(Val(1), Val(0), Val(5)),
                // Val(4) became what the trivial loop yielded; Val(9) became the empty loop's init.
                mul(Val(5), Val(1), Val(13)),
            ]
        );
    }
}

