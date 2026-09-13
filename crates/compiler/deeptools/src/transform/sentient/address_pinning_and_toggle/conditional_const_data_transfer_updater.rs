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

use super::abstract_data_transfer_updater::{insert_before, op_at_mut, scalar_add};
use super::looping_chain_mutable_addr_descriptor::{TransferEnd, mutable_addr_of};
use crate::formats::Bits;
use crate::islands::dataflow_ir::ty::ScalarTy;
use crate::islands::sentient::dialects::{self, Definitions, Op, Val, regions_mut, sentient};
use crate::transform::sentient::address_pinning_and_toggle::{
    ConditionalConstDataTransferUpdater, ConditionalConstResult, DataTransferDescriptor,
    YieldedIndex, create_offset_value, immutable_addr_mut, mutable_addr_mut, op_at,
};
use crate::transform::sentient::analyses::{
    EvaluatedValue, ExpressionEvaluator, OffsetSites, PinningSchemeManager,
};
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
    /// Replaces: e615_updateImmutableAddr
    ///
    /// Pins the transfer's immutable address between the LOWEST and HIGHEST constant the conditional
    /// can yield — "pretend we have a toggle situation between those two values" (`:2044-2049`) — and
    /// records which of the `sentient.if`'s results that operand was before replacing it.
    ///
    /// ⛔ THE RECORDED RESULT IS THE PRE-PINNING OPERAND: `res_index_` is set from
    /// `immutable_addr_[0]` (`:2053`) and the assignment below then replaces it, so
    /// [`Self::update_variable_offset_calculation`] and `getOffset` still name the conditional.
    /// ⭐ `element_size` IS IN BITS and `ty` is `mutable_addr_[0].get().getType()`, as for e420.
    pub fn update_immutable_addr(
        &mut self,
        dtd: &DataTransferDescriptor,
        op: &mut Op,
        end: TransferEnd,
        ty: ScalarTy,
        defs: Definitions<'_>,
        evaluator: &mut impl ExpressionEvaluator,
        ps_manager: &impl PinningSchemeManager,
        element_size: Bits,
    ) -> EvaluatedValue {
        let cc = dtd.conditional_constant_descriptor();
        let (Some(min), Some(max)) = (cc.min(evaluator), cc.max(evaluator)) else {
            panic!(
                "DT_CHECK_MSG(cc.isValid(), \"descriptor may be corrupt\") (`:2051`) — the \
                 conditional yielded no constants"
            )
        };
        let immutable_addr = *immutable_addr_mut(op, end);
        let Some(index) = index_of_operation_results(immutable_addr, defs) else {
            todo!(
                "updateImmutableAddr: getIndexOfOperationResults({immutable_addr:?}) is the -1 of a \
                 block argument, or its llvm_unreachable (dcc/src/Utils/Utils.cpp:139-153)"
            )
        };
        self.if_result = ConditionalConstResult {
            index: YieldedIndex(index),
            val: immutable_addr,
        };
        let new_immut_addr_ev =
            ps_manager.find_closest_pinned_addr(min, max, dtd.region, element_size);
        *immutable_addr_mut(op, end) = create_offset_value(new_immut_addr_ev, ty);
        new_immut_addr_ev
    }

    /// Replaces: e616_updateConstantMutableAddr
    ///
    /// Re-bases the yielded constants first, then adds the conditional's own selected result to the
    /// transfer's constant mutable address with a fresh `sentient.scalar_add` ahead of the memory op
    /// (`:2089-2110`).
    ///
    /// ⛔ THE OVERFLOW TEST USES `cc.getMax()` AND NOT THE PINNED ADDRESS'S OWN RANGE: the widest
    /// mutable offset this transfer can reach is `max + const_mutable_addr - new_immut_addr_ev`, and
    /// an LAR/EAR that cannot hold it is the abort (`:2103-2104`).
    /// ⛔ THE OPERAND IS ASSIGNED BEFORE THE INSERT, for e592's reason: the insert moves `dtd.op`.
    pub fn update_constant_mutable_addr<E: ExpressionEvaluator>(
        self,
        dtd: &DataTransferDescriptor,
        end: TransferEnd,
        ty: ScalarTy,
        evaluator: &mut E,
        ps_manager: &impl PinningSchemeManager,
        element_size: Bits,
        new_immut_addr_ev: EvaluatedValue,
        sites: &mut OffsetSites<'_>,
        walked: &mut Vec<Op>,
    ) {
        self.update_variable_offset_calculation(new_immut_addr_ev, ty, evaluator, sites, walked);

        let Some(memory_op) = op_at(&dtd.op, walked) else {
            todo!(
                "updateConstantMutableAddr: `dtd_.getOperation()` is at {:?}, which this unit body \
                 does not reach (:2105-2106)",
                dtd.op
            )
        };
        let mutable_addr = mutable_addr_of(memory_op, end);
        let is_const = {
            let regions: [&[Op]; 1] = [walked.as_slice()];
            let defs = Definitions::from_innermost(&regions);
            is_constant(mutable_addr, ConstKind::ScalarConstant, defs)
        };
        if !is_const {
            panic!("DT_CHECK(\"Expect constant mutable addr\") (`:2092-2094`) for {mutable_addr:?}")
        }
        let Some(cc_max) = dtd.conditional_constant_descriptor().max(evaluator) else {
            panic!(
                "DT_CHECK_MSG(cc.isValid(), \"descriptor may be corrupt\") (`:2051`) — the \
                 conditional yielded no constants"
            )
        };
        let const_ma_ev = evaluator.evaluate_value_handle(mutable_addr);
        let summed = evaluator.evaluate_sum_handle(cc_max, const_ma_ev);
        let max_ev = evaluator.evaluate_sub_handle(summed, new_immut_addr_ev);
        if ps_manager.overflows_register(max_ev, element_size) {
            panic!("DT_CHECK(\"LAR/EAR overflow detected\") (`:2103-2104`)")
        }
        let offset = self.get_offset(new_immut_addr_ev);
        let result = sites.values.mint();
        if let Some(op) = op_at_mut(walked, dtd.op.path()) {
            *mutable_addr_mut(op, end) = result;
        }
        insert_before(
            walked,
            dtd.op.path(),
            scalar_add(mutable_addr, offset, result, ty),
        );
    }

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

/// `dcc::utils::getIndexOfOperationResults(val)` (`dcc/src/Utils/Utils.cpp:139-153`) — WHICH of its
/// defining op's results a value is.
///
/// ⛔ `None` COVERS BOTH NON-ANSWERS: the `-1` for a block argument (`:141`), which
/// `DT_CHECK(res_index_ >= 0)` then rejects, and the trailing `llvm_unreachable` (`:150-152`).
fn index_of_operation_results(val: Val, defs: Definitions<'_>) -> Option<usize> {
    let def = defs.of(val)?;
    dialects::results(def)
        .iter()
        .position(|result| *result == val)
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
    use crate::arch::Elements;
    use crate::bridges::dataflow_ir_to_sentient::vc_vector_operands::OpId;
    use crate::islands::dataflow_ir::Values;
    use crate::islands::dataflow_ir::link::SendEnd;
    use crate::islands::sentient::dialects::sentient::{
        CmpPredicate, Reg, RegType, ShuffleMode, Yielded,
    };
    use crate::transform::sentient::address_pinning_and_toggle::{
        ConditionalConstResult, ConditionalConstantDescriptor, DescriptorMemoryUnit,
        PatternDescriptor, YieldedIndex,
    };
    use crate::transform::sentient::analyses::{Evaluation, MinMax, RegionSite};
    use std::cell::Cell;
    use std::panic::AssertUnwindSafe;

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

        fn evaluate_sum_handle(
            &mut self,
            lhs: EvaluatedValue,
            rhs: EvaluatedValue,
        ) -> EvaluatedValue {
            let sum = self.value(lhs) + self.value(rhs);
            self.intern(sum)
        }

        fn evaluate_min_max(&mut self, values: &[EvaluatedValue], which: MinMax) -> EvaluatedValue {
            let held: Vec<i64> = values.iter().map(|ev| self.value(*ev)).collect();
            let folded = match which {
                MinMax::Min => held.iter().min().copied(),
                MinMax::Max => held.iter().max().copied(),
            };
            self.intern(folded.unwrap_or_default())
        }
    }

    /// The pinning scheme with its answer stated, recording the pair it was offered — `&self` is the
    /// reference's own `const` manager, so a [`Cell`] is what lets the test read that pair back.
    struct StatedScheme {
        pinned: EvaluatedValue,
        offered: Cell<Option<(EvaluatedValue, EvaluatedValue)>>,
        overflow_test: Cell<Option<EvaluatedValue>>,
    }

    impl PinningSchemeManager for StatedScheme {
        fn find_closest_pinned_addr(
            &self,
            ev_x: EvaluatedValue,
            ev_y: EvaluatedValue,
            region: RegionSite,
            element_size: Bits,
        ) -> EvaluatedValue {
            assert_eq!(region, RegionSite::ProgramUnitBody);
            assert_eq!(element_size, Bits(16));
            self.offered.set(Some((ev_x, ev_y)));
            self.pinned
        }

        fn overflows_register(&self, addr_ev: EvaluatedValue, element_size: Bits) -> bool {
            assert_eq!(element_size, Bits(16));
            self.overflow_test.set(Some(addr_ev));
            false
        }
    }

    fn stated_scheme(pinned: EvaluatedValue) -> StatedScheme {
        StatedScheme {
            pinned,
            offered: Cell::new(None),
            overflow_test: Cell::new(None),
        }
    }

    /// The transfer these two updaters rewrite — the `sentient.if` result is its immutable address.
    fn load_and_send(mutable_addr: Val, immutable_addr: Val) -> Op {
        Op::Sentient(sentient::Op::LoadAndSend {
            mutable_addr,
            immutable_addr,
            increment: Val(131),
            consumer: SendEnd::to_self(Val(198)),
            result: Val(140),
            extent: sentient::Extent {
                total_elements: Elements(64),
                element_size: Bits(16),
                chunk_size: Elements(1),
                chunk_stride: Elements(1),
                burst_size: Elements(1),
            },
            interleaved_group: Elements(0),
            rotate_val: None,
            dir: None,
            shuffle_mode: ShuffleMode::NoShuffle,
            reg: Reg {
                locale: RegType::Lar,
                index: None,
            },
            dbg_name: None,
        })
    }

    fn conditional_constant(yielded: Vec<EvaluatedValue>) -> Option<PatternDescriptor> {
        Some(PatternDescriptor::ConditionalConstant(
            ConditionalConstantDescriptor {
                yielded_constants: yielded,
                can_be_simplified: false,
            },
        ))
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
                element_size: None,
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

    /// 615/656 — the pair offered to the pinning scheme is the LOWEST and HIGHEST constant the
    /// conditional can yield, and the `sentient.if` result the immutable address WAS is recorded
    /// before the assignment replaces it. ⛔ `create_offset_value` is `EvaluatedValue::buildOffsetValue`,
    /// out of campaign scope, so the write stops there and the record is read past the stop.
    #[test]
    fn e615_pins_between_the_lowest_and_highest_yielded_constant_and_records_the_conditional() {
        let mut evaluator = StatedEvaluator::default();
        let low = evaluator.constant(4096);
        let high = evaluator.constant(8192);
        let pinned = evaluator.constant(2048);
        let scheme = stated_scheme(pinned);
        let dtd = DataTransferDescriptor {
            op: OpId::at(&[4]),
            pattern_desc: conditional_constant(vec![high, low]),
            base_addrs: vec![low, high],
            region: RegionSite::ProgramUnitBody,
            memory_unit: DescriptorMemoryUnit::Lx,
            base_addr: Some(Val(110)),
            is_base_addr_mutable: false,
        };
        // The immutable address is the outer `sentient.if`'s one result.
        let body = nested_conditional();
        let regions: [&[Op]; 1] = [body.as_slice()];
        let mut op = load_and_send(Val(130), Val(110));
        let mut updater = ConditionalConstDataTransferUpdater {
            if_result: ConditionalConstResult {
                index: YieldedIndex(9),
                val: Val(999),
            },
        };

        let reached = std::panic::catch_unwind(AssertUnwindSafe(|| {
            updater.update_immutable_addr(
                &dtd,
                &mut op,
                TransferEnd::Src,
                ScalarTy::Index,
                Definitions::from_innermost(&regions),
                &mut evaluator,
                &scheme,
                Bits(16),
            )
        }));

        assert!(
            reached.is_err(),
            "the assignment reaches `buildOffsetValue`"
        );
        assert_eq!(
            updater.if_result,
            ConditionalConstResult {
                index: YieldedIndex(0),
                val: Val(110),
            }
        );
        // ⭐ THE ORDER OF `yielded_constants_` IS NOT THE ORDER OF THE PAIR: `getMin`/`getMax` fold it.
        let (ev_x, ev_y) = scheme.offered.get().expect("the pinning scheme was asked");
        assert_eq!((evaluator.value(ev_x), evaluator.value(ev_y)), (4096, 8192));
    }

    /// 616/656 — the yielded constants are rebased first, the overflow test sees
    /// `cc.getMax() + const_mutable_addr - pinned`, and the transfer's mutable address becomes a fresh
    /// `sentient.scalar_add` of that constant and the conditional's own result, inserted AHEAD of the
    /// memory op.
    #[test]
    fn e616_adds_the_conditionals_result_to_the_rebased_constant_mutable_addr() {
        let mut evaluator = StatedEvaluator {
            held: Vec::new(),
            constants: vec![(Val(101), 4096), (Val(102), 8192), (Val(130), 512)],
            built: Vec::new(),
        };
        let low = evaluator.constant(4096);
        let high = evaluator.constant(8192);
        let pinned = evaluator.constant(4096);
        let scheme = stated_scheme(pinned);
        let dtd = DataTransferDescriptor {
            op: OpId::at(&[4]),
            pattern_desc: conditional_constant(vec![low, high]),
            base_addrs: vec![low, high],
            region: RegionSite::ProgramUnitBody,
            memory_unit: DescriptorMemoryUnit::Lx,
            base_addr: Some(Val(110)),
            is_base_addr_mutable: false,
        };
        let mut walked = vec![
            constant(4096, Val(101)),
            constant(8192, Val(102)),
            if_op(Val(110), vec![yield_of(Val(101))], vec![yield_of(Val(102))]),
            constant(512, Val(130)),
            load_and_send(Val(130), Val(120)),
        ];
        let mut consts = Vec::new();
        let mut values = Values::default();
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
            updater.update_constant_mutable_addr(
                &dtd,
                TransferEnd::Src,
                ScalarTy::Index,
                &mut evaluator,
                &scheme,
                Bits(16),
                pinned,
                &mut sites,
                &mut walked,
            );
        }

        // `8192 + 512 - 4096` — the widest offset this transfer can reach, and NOT the pinned range.
        let overflow_test = scheme
            .overflow_test
            .get()
            .expect("the register was measured");
        assert_eq!(evaluator.value(overflow_test), 4608);
        // e277 rebased both yields on the way in: `4096 - 4096` and `8192 - 4096`.
        let built: Vec<i64> = evaluator.built.iter().map(|(_, value)| *value).collect();
        assert_eq!(built, vec![0, 4096]);
        // The add took the memory op's slot, and the memory op moved one later reading its result.
        assert_eq!(walked.len(), 6);
        assert_eq!(
            walked[4],
            scalar_add(Val(130), Val(110), Val(2), ScalarTy::Index)
        );
        assert_eq!(mutable_addr_of(&walked[5], TransferEnd::Src), Val(2));
    }
}
