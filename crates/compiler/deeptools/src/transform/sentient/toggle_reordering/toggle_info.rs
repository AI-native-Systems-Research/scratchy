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

//! `ToggleReordering.cpp` — 1 of the campaign's 656 units (dependency level(s) [0]).
//!
//! | unit | entry | level | lines | authority path:line |
//! |---|---|---|---|---|
//! | `e232_reorderToggle` | 232 | 0 | 31 | `dcc/src/Transform/Sentient/ToggleReordering.cpp:104` |

use crate::islands::dataflow_ir::ty::ScalarTy;
use crate::islands::sentient::dialects::{Op, Val, replace_all_uses_with, sentient};
use crate::transform::sentient::analyses::{Evaluation, ExpressionEvaluator, OffsetSites};

/// ONE TOGGLE'S DATA — `class ToggleInfo` (`:52-102`), whose toggle is `minuend - iter_arg`.
///
/// ⛔ VALUE IDENTITIES, NOT POSITIONS. An op's only identity in this island is the [`Val`] it binds,
/// and [`ToggleInfo::reorder_toggle`] materialises ops into the very block the loops sit in — so an
/// index would be stale by the time it is used.
/// ⭐ `evaluator_` AND `unit_op_` ARE NOT FIELDS: the evaluator is a parameter because the same call
/// also needs `&mut` on the block it writes into, and `unit_op_` only positioned `const_builder`,
/// which is [`OffsetSites::consts`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToggleInfo {
    /// `sub_op_`, by the value it binds.
    pub sub_result: Val,
    /// `sub_op_.getType()` — the type `buildOffsetValue` materialises the new initializer at.
    pub sub_ty: ScalarTy,
    /// `iter_arg_` — the `sentient.for` region argument the toggle subtracts.
    pub iter_arg: Val,
    /// `minuend_ev_` — `evaluateValue(sub_op.getInp1())`, taken once at construction.
    pub minuend_ev: Evaluation,
    /// `outer_loop_`, by its induction variable.
    pub outer_loop_iv: Val,
    /// `outer_loop_iter_arg_idx_` — ⛔ A CARRIED INDEX: `getRegionIterArgs()` drops the induction
    /// variable, so it indexes `getIterOperands()` and the yield directly.
    pub outer_loop_carried: usize,
}

impl ToggleInfo {
    /// Replaces: e232_reorderToggle
    ///
    /// Moves the toggle to just before its inner loop's yield, rewires its non-yield readers onto the
    /// iter arg, and rebuilds the outer loop's initializer as `minuend - init`.
    ///
    /// ⛔ TRAP: THE YIELD KEEPS **EVERY** USE, not just the one at the iter arg's index. The
    /// `replaceUsesWithIf` predicate (`:113-119`) tests `inner_yield_op.getOperand(idx)` and not the
    /// operand it was handed, so once that slot holds the sub result — which e233 requires — it
    /// declines the whole op. Hence save-replace-restore rather than a per-slot filter.
    /// ⛔ TRAP: `sites.consts` MUST NOT BE `walked`. `const_builder(unit_op_)` (`:124`) builds in the
    /// block the `dataflow.program_unit` SITS IN; `query_map_builder(outer_loop_)` (`:125`) builds in
    /// the block the outer loop sits in, which is inside `walked`.
    pub fn reorder_toggle<E: ExpressionEvaluator>(
        &self,
        evaluator: &mut E,
        sites: &mut OffsetSites<'_>,
        walked: &mut Vec<Op>,
    ) {
        // `iter_arg_.getOwner()->getParentOp()` and its `getBody(0)->getTerminator()`.
        if let Some(inner_body) = for_body_carrying(walked, self.iter_arg) {
            let yielded = yielded_results(inner_body);
            replace_all_uses_with(inner_body, self.sub_result, self.iter_arg);
            if let Some(yielded) = yielded {
                set_yielded_results(inner_body, yielded);
            }
            // `sub_op_->moveBefore(inner_yield_op)`.
            if let Some(sub_op) = take_op_binding(inner_body, self.sub_result) {
                let before_yield = inner_body
                    .iter()
                    .position(|op| matches!(op, Op::Sentient(sentient::Op::Yield { .. })))
                    .unwrap_or(inner_body.len());
                inner_body.insert(before_yield, sub_op);
            }
        }

        // `outer_loop_.setIterOperand(idx, new_init)`, the initializer being `minuend - init`.
        let Some(outer_block) = block_containing_for(walked, self.outer_loop_iv) else {
            return;
        };
        let Some(init) = carried_of(outer_block, self.outer_loop_iv)
            .and_then(|carried| carried.get(self.outer_loop_carried))
            .map(|carried| carried.init)
        else {
            return;
        };
        let init_ev = evaluator.evaluate_value(init);
        let new_init_ev = evaluator.evaluate_sub(&self.minuend_ev, &init_ev);
        let new_init = evaluator.build_offset_value(&new_init_ev, sites, outer_block, self.sub_ty);
        if let Some(carried) = carried_of(outer_block, self.outer_loop_iv)
            .and_then(|carried| carried.get_mut(self.outer_loop_carried))
        {
            carried.init = new_init;
        }
    }
}

/// The body of the `sentient.for` that binds `arg` as a carried region argument —
/// `arg.getOwner()->getParentOp()` reduced to the one thing its readers want.
fn for_body_carrying(block: &mut Vec<Op>, arg: Val) -> Option<&mut Vec<Op>> {
    for op in block.iter_mut() {
        let regions: Vec<&mut Vec<Op>> = match op {
            Op::Sentient(sentient::Op::For { carried, body, .. }) => {
                if carried.iter().any(|carried| carried.arg == arg) {
                    return Some(body);
                }
                vec![body]
            }
            Op::Sentient(inner) => sentient::regions_mut(inner),
            Op::AffineFor(loop_op) => vec![&mut loop_op.body],
            _ => Vec::new(),
        };
        for region in regions {
            if let Some(found) = for_body_carrying(region, arg) {
                return Some(found);
            }
        }
    }
    None
}

/// The block that HOLDS the `sentient.for` whose induction variable is `iv` — `OpBuilder(outer_loop_)`
/// as a block rather than an insertion point.
pub(super) fn block_containing_for(block: &mut Vec<Op>, iv: Val) -> Option<&mut Vec<Op>> {
    if block
        .iter()
        .any(|op| matches!(op, Op::Sentient(sentient::Op::For { iv: at, .. }) if *at == iv))
    {
        return Some(block);
    }
    for op in block.iter_mut() {
        let regions: Vec<&mut Vec<Op>> = match op {
            Op::Sentient(inner) => sentient::regions_mut(inner),
            Op::AffineFor(loop_op) => vec![&mut loop_op.body],
            _ => Vec::new(),
        };
        for region in regions {
            if let Some(found) = block_containing_for(region, iv) {
                return Some(found);
            }
        }
    }
    None
}

/// The carried list of the `sentient.for` at the top level of `block` whose induction variable is
/// `iv` — `getIterOperands()` and the register arrays, in this island's one entry per value.
pub(super) fn carried_of(block: &mut [Op], iv: Val) -> Option<&mut Vec<sentient::Carried>> {
    block.iter_mut().find_map(|op| match op {
        Op::Sentient(sentient::Op::For {
            iv: at, carried, ..
        }) if *at == iv => Some(carried),
        _ => None,
    })
}

/// `getBody(0)->getTerminator()`'s operands — `None` when the block has no `sentient.yield`.
pub(super) fn yielded_results(block: &[Op]) -> Option<Vec<Val>> {
    block.iter().find_map(|op| match op {
        Op::Sentient(sentient::Op::Yield { results }) => Some(results.clone()),
        _ => None,
    })
}

/// Puts `results` back on the block's `sentient.yield`.
fn set_yielded_results(block: &mut [Op], results: Vec<Val>) {
    for op in block.iter_mut() {
        if let Op::Sentient(sentient::Op::Yield { results: slot }) = op {
            *slot = results;
            return;
        }
    }
}

/// Removes the op binding `val`, regions included — the half of `moveBefore` that unlinks.
fn take_op_binding(block: &mut Vec<Op>, val: Val) -> Option<Op> {
    if let Some(at) = block.iter().position(|op| match op {
        Op::Sentient(inner) => sentient::results(inner).contains(&val),
        _ => false,
    }) {
        return Some(block.remove(at));
    }
    for op in block.iter_mut() {
        let regions: Vec<&mut Vec<Op>> = match op {
            Op::Sentient(inner) => sentient::regions_mut(inner),
            Op::AffineFor(loop_op) => vec![&mut loop_op.body],
            _ => Vec::new(),
        };
        for region in regions {
            if let Some(found) = take_op_binding(region, val) {
                return Some(found);
            }
        }
    }
    None
}

#[cfg(test)]
mod unit_tests {
    use super::ToggleInfo;
    use crate::islands::dataflow_ir::Values;
    use crate::islands::dataflow_ir::ty::ScalarTy;
    use crate::islands::sentient::dialects::{Op, Val, sentient};
    use crate::transform::sentient::analyses::{
        Evaluation, ExpressionEvaluator, OffsetSites, Offsets, ScalarOffset,
    };

    /// The out-of-scope evaluator with its three answers stated, so the rewrite is observable.
    struct StatedEvaluator {
        built: Val,
    }

    impl ExpressionEvaluator for StatedEvaluator {
        fn evaluate_value(&mut self, _value: Val) -> Evaluation {
            Evaluation {
                known_absolute: true,
                base: None,
                offsets: Offsets::AllUnit(ScalarOffset(4)),
            }
        }

        fn evaluate_sum(&mut self, _lhs: &Evaluation, _rhs: &Evaluation) -> Evaluation {
            todo!("no unit of this batch evaluates a sum")
        }

        fn evaluate_sub(&mut self, _lhs: &Evaluation, _rhs: &Evaluation) -> Evaluation {
            Evaluation {
                known_absolute: true,
                base: None,
                offsets: Offsets::AllUnit(ScalarOffset(3)),
            }
        }

        fn build_offset_value(
            &mut self,
            _evaluation: &Evaluation,
            _sites: &mut OffsetSites<'_>,
            _walked: &mut Vec<Op>,
            _ty: ScalarTy,
        ) -> Val {
            self.built
        }
    }

    /// One carried value, at the register defaults a fixture needs none of.
    fn carried(init: Val, arg: Val, result: Val) -> sentient::Carried {
        sentient::Carried {
            init,
            arg,
            result,
            reg: sentient::Reg {
                locale: sentient::RegType::Imm,
                index: None,
            },
            program_header: false,
            element_size: None,
        }
    }

    /// 🎯 e232 — the vendor's own diagram (`:67-88`): the toggle moves below its readers and just
    /// above the yield, the readers move onto the iter arg, the yield keeps the toggle, and the outer
    /// loop's initializer becomes the rebuilt `minuend - init`.
    #[test]
    fn a_toggle_moves_to_the_yield_and_its_readers_move_onto_the_iter_arg() {
        // for %iv0 = %b0 iter_args(%arg0 = %x) {
        //   for %iv1 = %b1 iter_args(%argN = %arg0) {
        //     %sub = sentient.scalar_sub %y, %argN
        //     %use = sentient.scalar_add %sub, %sub
        //     sentient.yield %sub
        //   }
        //   sentient.yield %inner
        // }
        let inner = Op::Sentient(sentient::Op::For {
            iv: Val(11),
            bound: Val(12),
            carried: vec![carried(Val(2), Val(20), Val(21))],
            dbg_name: None,
            body: vec![
                Op::Sentient(sentient::Op::ScalarSub {
                    lhs: Val(30),
                    rhs: Val(20),
                    result: Val(31),
                    reg: None,
                    element_size: None,
                    ty: ScalarTy::Index,
                }),
                Op::Sentient(sentient::Op::ScalarAdd {
                    lhs: Val(31),
                    rhs: Val(31),
                    result: Val(32),
                    reg: None,
                    element_size: None,
                    ty: ScalarTy::Index,
                }),
                Op::Sentient(sentient::Op::Yield {
                    results: vec![Val(31)],
                }),
            ],
        });
        let mut walked = vec![Op::Sentient(sentient::Op::For {
            iv: Val(1),
            bound: Val(0),
            carried: vec![carried(Val(9), Val(2), Val(3))],
            dbg_name: None,
            body: vec![
                inner,
                Op::Sentient(sentient::Op::Yield {
                    results: vec![Val(21)],
                }),
            ],
        })];
        let toggle = ToggleInfo {
            sub_result: Val(31),
            sub_ty: ScalarTy::Index,
            iter_arg: Val(20),
            minuend_ev: Evaluation {
                known_absolute: true,
                base: None,
                offsets: Offsets::AllUnit(ScalarOffset(7)),
            },
            outer_loop_iv: Val(1),
            outer_loop_carried: 0,
        };

        let mut consts = Vec::new();
        let mut values = Values::default();
        let mut sites = OffsetSites {
            consts: &mut consts,
            query_maps: None,
            values: &mut values,
        };
        toggle.reorder_toggle(
            &mut StatedEvaluator { built: Val(40) },
            &mut sites,
            &mut walked,
        );

        let Op::Sentient(sentient::Op::For {
            carried: outer_carried,
            body: outer_body,
            ..
        }) = &walked[0]
        else {
            panic!("the outer loop is still a sentient.for");
        };
        // `setIterOperand(0, new_init)` — the rebuilt `minuend - init`.
        assert_eq!(outer_carried[0].init, Val(40));
        let Op::Sentient(sentient::Op::For {
            body: inner_body, ..
        }) = &outer_body[0]
        else {
            panic!("the inner loop is still a sentient.for");
        };
        // The reader now reads `%argN`, the toggle sits just above the yield, and the yield still
        // yields the toggle.
        assert_eq!(
            inner_body[0],
            Op::Sentient(sentient::Op::ScalarAdd {
                lhs: Val(20),
                rhs: Val(20),
                result: Val(32),
                reg: None,
                element_size: None,
                ty: ScalarTy::Index,
            })
        );
        assert!(matches!(
            inner_body[1],
            Op::Sentient(sentient::Op::ScalarSub {
                lhs: Val(30),
                rhs: Val(20),
                result: Val(31),
                ..
            })
        ));
        assert_eq!(
            inner_body[2],
            Op::Sentient(sentient::Op::Yield {
                results: vec![Val(31)],
            })
        );
    }
}
