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

//! `AddressPinningAndToggle.cpp` — 3 of the campaign's 656 units (dependency level(s) [3, 6]).
//!
//! | unit | entry | level | lines | authority path:line |
//! |---|---|---|---|---|
//! | `e490_updateVariableOffsetCalculation` | 490 | 3 | 10 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:2131` |
//! | `e617_updateImmutableAddr` | 617 | 6 | 15 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:2114` |
//! | `e618_updateConstantMutableAddr` | 618 | 6 | 21 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:2143` |

use super::abstract_data_transfer_updater::{insert_before, op_at_mut, scalar_add};
use super::looping_chain_mutable_addr_descriptor::{TransferEnd, mutable_addr_of};
use super::{
    DataTransferDescriptor, IntegerSequenceDataTransferUpdater, create_offset_value,
    immutable_addr_mut, mutable_addr_mut, op_at, set_iter_operand,
};
use crate::formats::Bits;
use crate::islands::dataflow_ir::ty::ScalarTy;
use crate::islands::sentient::dialects::{Definitions, Op};
use crate::transform::sentient::analyses::{
    EvaluatedValue, ExpressionEvaluator, OffsetSites, PinningSchemeManager,
};
use crate::transform::sentient::utils::{ConstKind, is_constant};

impl IntegerSequenceDataTransferUpdater {
    /// Replaces: e617_updateImmutableAddr
    ///
    /// "The same strategy used for conditional constant case" (`:2115`): the transfer's immutable
    /// address becomes one pinned address covering the whole sequence's span (`:2119-2126`).
    ///
    /// ⛔ NO `res_index_` HERE, unlike e615: the sequence is reached through the loop's iter operand
    /// (e490), never through a `sentient.if` result, so nothing has to be recorded before the
    /// assignment replaces the operand.
    pub fn update_immutable_addr(
        self,
        dtd: &DataTransferDescriptor,
        op: &mut Op,
        end: TransferEnd,
        ty: ScalarTy,
        evaluator: &mut impl ExpressionEvaluator,
        ps_manager: &impl PinningSchemeManager,
        element_size: Bits,
    ) -> EvaluatedValue {
        let isq = *dtd.integer_sequence_descriptor();
        let (Some(min), Some(max)) = (isq.min(evaluator), isq.max(evaluator)) else {
            panic!(
                "DT_CHECK_MSG(is.isValid(), \"descriptor may be corrupt\") (`:2117`) for {isq:?}"
            )
        };
        let new_immut_addr_ev =
            ps_manager.find_closest_pinned_addr(min, max, dtd.region, element_size);
        *immutable_addr_mut(op, end) = create_offset_value(new_immut_addr_ev, ty);
        new_immut_addr_ev
    }

    /// Replaces: e618_updateConstantMutableAddr
    ///
    /// Rebases the sequence's initializer first (e490), then adds the loop's own carried offset to the
    /// transfer's constant mutable address with a fresh `sentient.scalar_add` ahead of the memory op
    /// (`:2143-2163`).
    ///
    /// ⛔ THE OVERFLOW TEST IS `is.getMax() + const_mutable_addr - new_immut_addr_ev` — the widest
    /// address this transfer's LAR/EAR must hold once the sequence is rebased (`:2151-2157`).
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
        self.update_variable_offset_calculation(
            dtd,
            new_immut_addr_ev,
            ty,
            evaluator,
            sites,
            walked,
        );

        let Some(memory_op) = op_at(&dtd.op, walked) else {
            todo!(
                "updateConstantMutableAddr: `dtd_.getOperation()` is at {:?}, which this unit body \
                 does not reach (:2159-2160)",
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
            panic!("DT_CHECK(\"Expect constant mutable addr\") (`:2147-2149`) for {mutable_addr:?}")
        }
        let isq = *dtd.integer_sequence_descriptor();
        let Some(is_max) = isq.max(evaluator) else {
            panic!(
                "DT_CHECK_MSG(is.isValid(), \"descriptor may be corrupt\") (`:2117`) for {isq:?}"
            )
        };
        let const_ma_ev = evaluator.evaluate_value_handle(mutable_addr);
        let summed = evaluator.evaluate_sum_handle(is_max, const_ma_ev);
        let max_ev = evaluator.evaluate_sub_handle(summed, new_immut_addr_ev);
        if ps_manager.overflows_register(max_ev, element_size) {
            panic!("DT_CHECK(\"LAR/EAR overflow detected\") (`:2156-2157`)")
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

    /// Replaces: e490_updateVariableOffsetCalculation
    ///
    /// REBASES THE SEQUENCE ON THE PINNED IMMUTABLE ADDRESS: the loop's iter operand becomes a fresh
    /// offset value holding `init - new_immut_addr_ev`, so the induction still lands where the
    /// unpinned sequence did (`:2131-2141`).
    ///
    /// ⭐ THROUGH `buildOffsetValue` AND `sites`/`walked`, AS e277 DOES: `createOffsetValue` (e009)
    /// is that call plus two builder positions, which the campaign names droppable — and going via
    /// e009's stop would make this unit's one IR write unreachable.
    /// ⛔ THE WRITE IS THE PORT: `ty` is `mutable_addr_[0].get().getType()` (`:1029`), and
    /// [`set_iter_operand`] is `getOuterLoop().setIterOperand(getIterArgIndex(), ..)`.
    pub fn update_variable_offset_calculation<E: ExpressionEvaluator>(
        self,
        dtd: &DataTransferDescriptor,
        new_immut_addr_ev: EvaluatedValue,
        ty: ScalarTy,
        evaluator: &mut E,
        sites: &mut OffsetSites<'_>,
        walked: &mut Vec<Op>,
    ) {
        // `dtd_.getIntegerSequenceDescriptor()` — `DT_CHECK(isIntegerSequence())` (`:728`) panics in
        // [`DataTransferDescriptor::integer_sequence_descriptor`] itself.
        let isq = *dtd.integer_sequence_descriptor();
        let (Some(init), Some(outer_loop), Some(iter_arg_index)) =
            (isq.init, isq.outer_loop, isq.iter_arg_index)
        else {
            todo!(
                "IntegerSequenceDataTransferUpdater::updateVariableOffsetCalculation: \
                 *isq.getInit() / isq.getOuterLoop() on {isq:?}, which the reference dereferences \
                 unchecked (:2131-2141)"
            )
        };
        // `evaluator_.evaluateSub(isq.getInit(), new_immut_addr_ev)` (`:2137-2138`).
        let new_init = evaluator.evaluate_sub_handle(init, new_immut_addr_ev);
        let new_operand = evaluator.build_offset_value_of(new_init, sites, walked, ty);
        set_iter_operand(walked, outer_loop, iter_arg_index, new_operand);
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::arch::Elements;
    use crate::bridges::dataflow_ir_to_sentient::vc_vector_operands::OpId;
    use crate::islands::dataflow_ir::Values;
    use crate::islands::dataflow_ir::link::SendEnd;
    use crate::islands::sentient::dialects::sentient::{Carried, Reg, RegType, ShuffleMode};
    use crate::islands::sentient::dialects::{Val, sentient};
    use crate::transform::sentient::address_pinning_and_toggle::{
        DescriptorMemoryUnit, IntegerSequenceDescriptor, PatternDescriptor, SequenceSize,
    };
    use crate::transform::sentient::analyses::{Evaluation, MinMax, RegionSite};
    use crate::transform::sentient::{ForRef, IterArgIndex};
    use std::cell::Cell;
    use std::panic::AssertUnwindSafe;

    /// The handle flavour with its answers stated as INTEGERS, so `==` on handles is `==` on values.
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
            todo!("e490 asks for handles, never for a decoded evaluation")
        }

        fn evaluate_sum(&mut self, _lhs: &Evaluation, _rhs: &Evaluation) -> Evaluation {
            todo!("e490 never sums")
        }

        fn build_offset_value(
            &mut self,
            _evaluation: &Evaluation,
            _sites: &mut OffsetSites<'_>,
            _walked: &mut Vec<Op>,
            _ty: ScalarTy,
        ) -> Val {
            todo!("e490 builds from a stored handle, not from an evaluation")
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

        fn evaluate_multiply_by_const(&mut self, ev: EvaluatedValue, by: i64) -> EvaluatedValue {
            let product = self.value(ev) * by;
            self.intern(product)
        }

        fn evaluate_min_max(&mut self, values: &[EvaluatedValue], which: MinMax) -> EvaluatedValue {
            let held: Vec<i64> = values.iter().map(|ev| self.value(*ev)).collect();
            let folded = match which {
                MinMax::Min => held.iter().min().copied(),
                MinMax::Max => held.iter().max().copied(),
            };
            self.intern(folded.unwrap_or_default())
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
    }

    /// The pinning scheme with its answer stated, recording what it was asked — `&self` is the
    /// reference's own `const` manager, so a [`Cell`] is what lets the test read those back.
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
            assert_eq!(element_size, Bits(16));
            assert_eq!(region, RegionSite::default());
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

    fn scalar_const(value: i64, result: Val) -> Op {
        Op::Sentient(sentient::Op::ScalarConstant {
            value,
            result,
            reg_locale: RegType::Imm,
            ty: ScalarTy::Index,
            is_symbol: false,
        })
    }

    /// The transfer these two updaters rewrite.
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

    /// `init + stride * i` for `i` in `[0, 4)` from 8192 by 64, as a descriptor.
    fn integer_sequence(init: EvaluatedValue, stride: EvaluatedValue) -> Option<PatternDescriptor> {
        Some(PatternDescriptor::IntegerSequence(
            IntegerSequenceDescriptor {
                outer_loop: Some(ForRef(Val(10))),
                iter_arg_index: Some(IterArgIndex(0)),
                init: Some(init),
                stride: Some(stride),
                size: SequenceSize::Terms(4),
                can_be_simplified: false,
            },
        ))
    }

    fn carried(init: Val) -> Carried {
        Carried {
            init,
            arg: Val(11),
            result: Val(12),
            reg: Reg {
                locale: RegType::Unknown,
                index: None,
            },
            program_header: false,
            element_size: None,
        }
    }

    /// The loop's iter operand becomes a fresh constant holding `init - pinned`, and the walk finds
    /// that loop by the induction variable [`ForRef`] names.
    #[test]
    fn e490_rebases_the_loops_iter_operand_on_the_pinned_address() {
        let mut walked = vec![Op::Sentient(sentient::Op::For {
            iv: Val(10),
            bound: Val(1),
            bound_reg: None,
            carried: vec![carried(Val(2))],
            dbg_name: None,
            body: Vec::new(),
        })];
        let mut consts = Vec::new();
        let mut values = Values::default();
        let mut evaluator = StatedEvaluator::default();
        let init = evaluator.constant(8192);
        let pinned = evaluator.constant(4096);
        let dtd = DataTransferDescriptor {
            op: OpId::at(&[0]),
            pattern_desc: Some(PatternDescriptor::IntegerSequence(
                IntegerSequenceDescriptor {
                    outer_loop: Some(ForRef(Val(10))),
                    iter_arg_index: Some(IterArgIndex(0)),
                    init: Some(init),
                    stride: Some(evaluator.constant(64)),
                    size: SequenceSize::Terms(4),
                    can_be_simplified: false,
                },
            )),
            base_addrs: vec![init],
            region: RegionSite::default(),
            memory_unit: DescriptorMemoryUnit::Lx,
            base_addr: Some(Val(0)),
            is_base_addr_mutable: false,
        };
        let updater = IntegerSequenceDataTransferUpdater { iter_arg: Val(11) };
        {
            let mut sites = OffsetSites {
                consts: &mut consts,
                query_maps: None,
                values: &mut values,
            };
            updater.update_variable_offset_calculation(
                &dtd,
                pinned,
                ScalarTy::Index,
                &mut evaluator,
                &mut sites,
                &mut walked,
            );
        }

        // `8192 - 4096`, built once, and written over the loop's `init`.
        assert_eq!(evaluator.built.len(), 1);
        let (rebased, value) = evaluator.built[0];
        assert_eq!(value, 4096);
        assert_eq!(consts.len(), 1);
        let Op::Sentient(sentient::Op::For { carried, .. }) = &walked[0] else {
            panic!("the fixture's only op is the sentient.for")
        };
        assert_eq!(carried[0].init, rebased);
    }

    /// 617/656 — the pair offered to the pinning scheme spans the WHOLE sequence, `getMin` and
    /// `getMax` over its two ends. ⛔ `create_offset_value` is `EvaluatedValue::buildOffsetValue`, out
    /// of campaign scope, so the write stops there and the offered pair is read past the stop.
    #[test]
    fn e617_pins_one_address_over_the_sequences_whole_span() {
        let mut evaluator = StatedEvaluator::default();
        let init = evaluator.constant(8192);
        let stride = evaluator.constant(64);
        let pinned = evaluator.constant(4096);
        let scheme = stated_scheme(pinned);
        let dtd = DataTransferDescriptor {
            op: OpId::at(&[0]),
            pattern_desc: integer_sequence(init, stride),
            base_addrs: vec![init],
            region: RegionSite::default(),
            memory_unit: DescriptorMemoryUnit::Lx,
            base_addr: Some(Val(120)),
            is_base_addr_mutable: false,
        };
        let mut op = load_and_send(Val(130), Val(120));
        let updater = IntegerSequenceDataTransferUpdater { iter_arg: Val(11) };

        let reached = std::panic::catch_unwind(AssertUnwindSafe(|| {
            updater.update_immutable_addr(
                &dtd,
                &mut op,
                TransferEnd::Src,
                ScalarTy::Index,
                &mut evaluator,
                &scheme,
                Bits(16),
            )
        }));

        assert!(
            reached.is_err(),
            "the assignment reaches `buildOffsetValue`"
        );
        // ⛔ 8448 IS ONE STRIDE PAST THE SEQUENCE (`8192 + 64 * 4`), which e406 ports verbatim.
        let (ev_x, ev_y) = scheme.offered.get().expect("the pinning scheme was asked");
        assert_eq!((evaluator.value(ev_x), evaluator.value(ev_y)), (8192, 8448));
    }

    /// 618/656 — the sequence's initializer is rebased first, the overflow test sees
    /// `is.getMax() + const_mutable_addr - pinned`, and the transfer's mutable address becomes a fresh
    /// `sentient.scalar_add` of that constant and the loop's carried iter arg, AHEAD of the memory op.
    #[test]
    fn e618_adds_the_carried_iter_arg_to_the_rebased_constant_mutable_addr() {
        let mut evaluator = StatedEvaluator {
            held: Vec::new(),
            constants: vec![(Val(130), 512)],
            built: Vec::new(),
        };
        let init = evaluator.constant(8192);
        let stride = evaluator.constant(64);
        let pinned = evaluator.constant(4096);
        let scheme = stated_scheme(pinned);
        let dtd = DataTransferDescriptor {
            op: OpId::at(&[2]),
            pattern_desc: integer_sequence(init, stride),
            base_addrs: vec![init],
            region: RegionSite::default(),
            memory_unit: DescriptorMemoryUnit::Lx,
            base_addr: Some(Val(120)),
            is_base_addr_mutable: false,
        };
        let mut walked = vec![
            Op::Sentient(sentient::Op::For {
                iv: Val(10),
                bound: Val(1),
                bound_reg: None,
                carried: vec![carried(Val(102))],
                dbg_name: None,
                body: Vec::new(),
            }),
            scalar_const(512, Val(130)),
            load_and_send(Val(130), Val(120)),
        ];
        let mut consts = Vec::new();
        let mut values = Values::default();
        let updater = IntegerSequenceDataTransferUpdater { iter_arg: Val(11) };
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

        // `8448 + 512 - 4096` — the widest address this LAR must hold once the sequence is rebased.
        let overflow_test = scheme
            .overflow_test
            .get()
            .expect("the register was measured");
        assert_eq!(evaluator.value(overflow_test), 4864);
        // e490 rebased the loop's iter operand on the way in: `8192 - 4096`.
        assert_eq!(evaluator.built.len(), 1);
        let (rebased, value) = evaluator.built[0];
        assert_eq!(value, 4096);
        let Op::Sentient(sentient::Op::For { carried, .. }) = &walked[0] else {
            panic!("the fixture's first op is the sentient.for")
        };
        assert_eq!(carried[0].init, rebased);
        // The add took the memory op's slot, and the memory op moved one later reading its result.
        assert_eq!(walked.len(), 4);
        assert_eq!(
            walked[2],
            scalar_add(Val(130), Val(11), Val(1), ScalarTy::Index)
        );
        assert_eq!(mutable_addr_of(&walked[3], TransferEnd::Src), Val(1));
    }
}
