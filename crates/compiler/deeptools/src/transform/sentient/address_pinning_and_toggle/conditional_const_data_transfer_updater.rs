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

//! `AddressPinningAndToggle.cpp` — 3 of the campaign's 656 units (dependency level(s) [1, 6]).
//!
//! | unit | entry | level | lines | authority path:line |
//! |---|---|---|---|---|
//! | `e277_updateVariableOffsetCalculation` | 277 | 1 | 21 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:2065` |
//! | `e615_updateImmutableAddr` | 615 | 6 | 20 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:2043` |
//! | `e616_updateConstantMutableAddr` | 616 | 6 | 21 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:2088` |


// crustify:todo: e615_updateImmutableAddr
//   authority : dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:2043  (20 body lines, level 6)
//   original  : const EvaluatedValue & ConditionalConstDataTransferUpdater::updateImmutableAddr()
//   calls     : e009_createOffsetValue, e265_getConditionalConstantDescriptor, e266_getConditionalConstantDescriptor, e278_isValid, e399_getMin, e400_getMax, e403_getMin, e404_getMax, e405_getMin, e406_getMax, e409_getMin, e410_getMax, e412_getMin, e413_getMax …

// crustify:todo: e616_updateConstantMutableAddr
//   authority : dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:2088  (21 body lines, level 6)
//   original  : void ConditionalConstDataTransferUpdater::updateConstantMutableAddr( const EvaluatedValue &new_immut_addr_ev)
//   calls     : e011_getOffset, e012_getOffset, e013_getOffset, e265_getConditionalConstantDescriptor, e266_getConditionalConstantDescriptor, e277_updateVariableOffsetCalculation, e400_getMax, e404_getMax, e406_getMax, e410_getMax, e413_getMax, e418_getOffset, e548_getMax, e588_getMax …

use crate::islands::dataflow_ir::ty::ScalarTy;
use crate::islands::sentient::dialects::{Definitions, Op, Val, regions_mut, sentient};
use crate::transform::sentient::address_pinning_and_toggle::ConditionalConstDataTransferUpdater;
use crate::transform::sentient::analyses::{EvaluatedValue, ExpressionEvaluator, OffsetSites};
use crate::transform::sentient::utils::{ConstKind, is_constant};

/// ONE `applyToAllYields` CALLBACK INVOCATION, RECORDED RATHER THAN APPLIED — `(terminator, index)`
/// named by VALUE and never by position.
///
/// ⛔⛔ AN INSERTION INDEX WOULD GO STALE: `createOffsetValue` mints ops between the decision and the
/// write, which is exactly how e170 lost its position. A `sentient.if`'s result at the walked index is
/// unique to one (op, index) pair, so reopening the yield after a build cannot land elsewhere.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct YieldSite {
    /// `if_op->getResult(index)` — the identity of the `sentient.if` whose yield this is.
    result: Val,
    /// Which region of that `if`: 0 is `then_body`, 1 is `else_body`.
    region: usize,
    /// `result_index` — where in the yield the operand sits.
    index: usize,
    /// The constant currently yielded there.
    operand: Val,
}

impl ConditionalConstDataTransferUpdater {
    /// Replaces: e277_updateVariableOffsetCalculation
    ///
    /// REBASES EVERY REACHABLE YIELDED CONSTANT of the conditional on the pinned immutable address:
    /// each yield operand becomes a fresh offset value holding `operand - new_immut_addr_ev`
    /// (`:2081-2085`).
    ///
    /// ⛔ IT REWRITES THE IR — decide over the whole `if` first, then build and set one site at a
    /// time, because every build appends ops to the blocks being rewritten.
    /// ⭐ `createOffsetValue` (e009) IS `EvaluatedValue::buildOffsetValue` PLUS TWO BUILDER
    /// POSITIONS, which the campaign names droppable and which `sites`/`walked` carry instead.
    pub fn update_variable_offset_calculation<E: ExpressionEvaluator>(
        self,
        new_immut_addr_ev: EvaluatedValue,
        ty: ScalarTy,
        evaluator: &mut E,
        sites: &mut OffsetSites<'_>,
        walked: &mut Vec<Op>,
    ) {
        // `DT_CHECK(if_op_)` and `DT_CHECK(res_index_ >= 0)` (`:2069-2070`) are
        // [`super::ConditionalConstResult`]'s own existence — there is no absent case to check.
        let index = self.if_result.index.0;
        let mut yield_sites = Vec::new();
        {
            let regions: [&[Op]; 1] = [walked.as_slice()];
            let defs = Definitions::from_innermost(&regions);
            constant_yield_sites(self.if_result.val, index, defs, &mut yield_sites);
        }

        for site in yield_sites {
            let operand_ev = evaluator.evaluate_value_handle(site.operand);
            // `new_const = const_op.getValue() - new_immut_addr_ev` (`:2079-2080`).
            let new_const = evaluator.evaluate_sub_handle(operand_ev, new_immut_addr_ev);
            let new_operand = evaluator.build_offset_value_of(new_const, sites, walked, ty);
            // `terminator->setOperand(index, ..)` (`:2081-2082`).
            let Some(operands) = yield_operands_of(walked, &site) else {
                todo!(
                    "updateVariableOffsetCalculation: the yield of region {} of the sentient.if \
                     binding {:?} is gone after building its offset value (:2081-2082)",
                    site.region,
                    site.result
                )
            };
            let Some(slot) = operands.get_mut(site.index) else {
                todo!(
                    "updateVariableOffsetCalculation: yield operand {} is out of range after \
                     building its offset value (:2081-2082)",
                    site.index
                )
            };
            *slot = new_operand;
        }
    }
}

/// `dcc::utils::applyToAllYields<sentient::IfOp>` (`dcc/src/Utils/Utils.cpp:164-179`) with the
/// callback's `(terminator, index)` RECORDED: per region of the `sentient.if` binding `result`, the
/// yield operand at `index`, descending through a nested `sentient.if` bound at that position.
///
/// ⭐ AN EMPTY `else_body` IS NO ELSE REGION, which is the reference's `getNumRegions()` answering 1.
fn constant_yield_sites(
    result: Val,
    index: usize,
    defs: Definitions<'_>,
    out: &mut Vec<YieldSite>,
) {
    let Some(Op::Sentient(sentient::Op::If {
        then_body,
        else_body,
        ..
    })) = defs.of(result)
    else {
        return;
    };
    for (region, body) in [then_body.as_slice(), else_body.as_slice()]
        .into_iter()
        .enumerate()
    {
        if body.is_empty() {
            continue;
        }
        let Some(Op::Sentient(sentient::Op::Yield { results })) = body.last() else {
            continue;
        };
        let Some(&operand) = results.get(index) else {
            continue;
        };
        // `!isa<BlockArgument>(yield_operand) && isa<IfOpTy>(yield_operand.getDefiningOp())` — the
        // nested case recurses at `getIndexOfOperationResults(yield_operand)`.
        let nested = match defs.of(operand) {
            Some(Op::Sentient(sentient::Op::If { yielded, .. })) => {
                yielded.iter().position(|entry| entry.result == operand)
            }
            _ => None,
        };
        match nested {
            Some(nested_index) => constant_yield_sites(operand, nested_index, defs, out),
            None => {
                // `DT_CHECK(isConstant<ConstantOp>(terminator->getOperand(index)) && "Expect
                // constant yield operands")` (`:2074-2077`).
                if !is_constant(operand, ConstKind::ScalarConstant, defs) {
                    todo!(
                        "updateVariableOffsetCalculation: DT_CHECK(isConstant(..) && \"Expect \
                         constant yield operands\") — {operand:?} is yielded by region {region} \
                         of the sentient.if binding {result:?} (:2074-2077)"
                    )
                }
                out.push(YieldSite {
                    result,
                    region,
                    index,
                    operand,
                });
            }
        }
    }
}

/// The `sentient.yield` operand list `site` names, for a rewrite — the `terminator` the reference's
/// callback calls `setOperand` on, reopened by value after the offset value was built.
fn yield_operands_of<'a>(block: &'a mut Vec<Op>, site: &YieldSite) -> Option<&'a mut Vec<Val>> {
    for op in block.iter_mut() {
        let regions: Vec<&mut Vec<Op>> = match op {
            Op::Sentient(sentient::Op::If {
                yielded,
                then_body,
                else_body,
                ..
            }) => {
                if yielded
                    .get(site.index)
                    .is_some_and(|entry| entry.result == site.result)
                {
                    let body = if site.region == 0 {
                        then_body
                    } else {
                        else_body
                    };
                    let Some(Op::Sentient(sentient::Op::Yield { results })) = body.last_mut()
                    else {
                        return None;
                    };
                    return Some(results);
                }
                vec![then_body, else_body]
            }
            _ => regions_mut(op),
        };
        for region in regions {
            if let Some(found) = yield_operands_of(region, site) {
                return Some(found);
            }
        }
    }
    None
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::islands::dataflow_ir::Values;
    use crate::islands::sentient::dialects::sentient::{CmpPredicate, Reg, RegType, Yielded};
    use crate::transform::sentient::address_pinning_and_toggle::{
        ConditionalConstResult, YieldedIndex,
    };
    use crate::transform::sentient::analyses::Evaluation;

    /// The handle flavour with its answers stated as INTEGERS, plus a `buildOffsetValue` that mints a
    /// `sentient.scalar_constant` into `sites.consts` — so the rebase is observable as both a value
    /// and an op.
    #[derive(Default)]
    struct StatedEvaluator {
        held: Vec<i64>,
        constants: Vec<(Val, i64)>,
        built: Vec<(Val, i64)>,
    }

    impl StatedEvaluator {
        fn intern(&mut self, value: i64) -> EvaluatedValue {
            let index = self
                .held
                .iter()
                .position(|held| *held == value)
                .unwrap_or_else(|| {
                    self.held.push(value);
                    self.held.len() - 1
                });
            EvaluatedValue(u32::try_from(index).unwrap_or_default())
        }

        fn value(&self, ev: EvaluatedValue) -> i64 {
            self.held
                .get(usize::try_from(ev.0).unwrap_or_default())
                .copied()
                .unwrap_or_default()
        }
    }

    impl ExpressionEvaluator for StatedEvaluator {
        fn evaluate_value(&mut self, _value: Val) -> Evaluation {
            todo!("e277 asks for handles, never for a decoded evaluation")
        }

        fn evaluate_sum(&mut self, _lhs: &Evaluation, _rhs: &Evaluation) -> Evaluation {
            todo!("e277 never sums")
        }

        fn build_offset_value(
            &mut self,
            _evaluation: &Evaluation,
            _sites: &mut OffsetSites<'_>,
            _walked: &mut Vec<Op>,
            _ty: ScalarTy,
        ) -> Val {
            todo!("e277 builds from a stored handle, not from an evaluation")
        }

        fn build_offset_value_of(
            &mut self,
            immutable: EvaluatedValue,
            sites: &mut OffsetSites<'_>,
            _walked: &mut Vec<Op>,
            ty: ScalarTy,
        ) -> Val {
            let value = self.value(immutable);
            let result = sites.values.mint();
            self.built.push((result, value));
            sites
                .consts
                .push(Op::Sentient(sentient::Op::ScalarConstant {
                    value,
                    result,
                    reg_locale: RegType::Imm,
                    ty,
                    is_symbol: false,
                }));
            result
        }

        fn evaluate_value_handle(&mut self, value: Val) -> EvaluatedValue {
            let held = self
                .constants
                .iter()
                .find(|(val, _)| *val == value)
                .map(|(_, held)| *held);
            match held {
                Some(held) => self.intern(held),
                None => todo!("the fixture states no constant for {value:?}"),
            }
        }

        fn constant(&mut self, value: i64) -> EvaluatedValue {
            self.intern(value)
        }

        fn evaluate_sub_handle(
            &mut self,
            lhs: EvaluatedValue,
            rhs: EvaluatedValue,
        ) -> EvaluatedValue {
            let difference = self.value(lhs) - self.value(rhs);
            self.intern(difference)
        }
    }

    fn constant(value: i64, result: Val) -> Op {
        Op::Sentient(sentient::Op::ScalarConstant {
            value,
            result,
            reg_locale: RegType::Imm,
            ty: ScalarTy::Index,
            is_symbol: false,
        })
    }

    fn if_op(result: Val, then_body: Vec<Op>, else_body: Vec<Op>) -> Op {
        Op::Sentient(sentient::Op::If {
            predicate: CmpPredicate::Eq,
            lhs: Val(120),
            rhs: Val(121),
            yielded: vec![Yielded {
                result,
                reg: Reg {
                    locale: RegType::Lbr,
                    index: None,
                },
            }],
            dbg_name: None,
            then_body,
            else_body,
        })
    }

    fn yield_of(result: Val) -> Op {
        Op::Sentient(sentient::Op::Yield {
            results: vec![result],
        })
    }

    /// The reference's own shape (`e016`'s fixture): the `else` of the outer `if` yields a NESTED
    /// `if`'s result, so a faithful walk rebases THREE yields and a shallow one would rebase two.
    fn nested_conditional() -> Vec<Op> {
        vec![
            constant(4096, Val(101)),
            constant(8192, Val(102)),
            constant(8192, Val(103)),
            if_op(
                Val(110),
                vec![yield_of(Val(101))],
                vec![
                    if_op(Val(111), vec![yield_of(Val(102))], vec![yield_of(Val(103))]),
                    yield_of(Val(111)),
                ],
            ),
        ]
    }

    /// Each of the three reachable yields now names a fresh constant holding `yielded - pinned`, and
    /// the outer `else` still yields the nested `if`'s result.
    #[test]
    fn e277_rebases_every_reachable_yielded_constant_on_the_pinned_address() {
        let mut walked = nested_conditional();
        let mut consts = Vec::new();
        let mut values = Values::default();
        let mut evaluator = StatedEvaluator {
            held: Vec::new(),
            constants: vec![(Val(101), 4096), (Val(102), 8192), (Val(103), 8192)],
            built: Vec::new(),
        };
        let pinned = evaluator.constant(4096);
        let updater = ConditionalConstDataTransferUpdater {
            if_result: ConditionalConstResult {
                index: YieldedIndex(0),
                val: Val(110),
            },
        };
        {
            let mut sites = OffsetSites {
                consts: &mut consts,
                query_maps: None,
                values: &mut values,
            };
            updater.update_variable_offset_calculation(
                pinned,
                ScalarTy::Index,
                &mut evaluator,
                &mut sites,
                &mut walked,
            );
        }

        // `4096 - 4096`, then `8192 - 4096` twice — one built constant per reachable yield.
        let built: Vec<i64> = evaluator.built.iter().map(|(_, value)| *value).collect();
        assert_eq!(built, vec![0, 4096, 4096]);
        assert_eq!(consts.len(), 3);
        let rebased: Vec<Val> = evaluator.built.iter().map(|(val, _)| *val).collect();

        let Op::Sentient(sentient::Op::If {
            then_body: outer_then,
            else_body: outer_else,
            ..
        }) = &walked[3]
        else {
            panic!("the fixture's fourth op is the outer sentient.if")
        };
        assert_eq!(outer_then.last(), Some(&yield_of(rebased[0])));
        // ⭐ THE OUTER `else` IS UNTOUCHED: its operand is the nested `if`'s result, not a constant.
        assert_eq!(outer_else.last(), Some(&yield_of(Val(111))));
        let Some(Op::Sentient(sentient::Op::If {
            then_body: inner_then,
            else_body: inner_else,
            ..
        })) = outer_else.first()
        else {
            panic!("the outer else region opens with the nested sentient.if")
        };
        assert_eq!(inner_then.last(), Some(&yield_of(rebased[1])));
        assert_eq!(inner_else.last(), Some(&yield_of(rebased[2])));
    }
}
