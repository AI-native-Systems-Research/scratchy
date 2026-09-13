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


use super::{DataTransferDescriptor, IntegerSequenceDataTransferUpdater, set_iter_operand};
use crate::islands::dataflow_ir::ty::ScalarTy;
use crate::islands::sentient::dialects::Op;
use crate::transform::sentient::analyses::{EvaluatedValue, ExpressionEvaluator, OffsetSites};

impl IntegerSequenceDataTransferUpdater {
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
        // `evaluator_.evaluateSub(isq.getInit(), new_immut_addr_ev)` (`:2136-2137`).
        let new_init = evaluator.evaluate_sub_handle(init, new_immut_addr_ev);
        let new_operand = evaluator.build_offset_value_of(new_init, sites, walked, ty);
        set_iter_operand(walked, outer_loop, iter_arg_index, new_operand);
    }
}

// crustify:todo: e617_updateImmutableAddr
//   authority : dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:2114  (15 body lines, level 6)
//   original  : const EvaluatedValue & IntegerSequenceDataTransferUpdater::updateImmutableAddr()
//   calls     : e009_createOffsetValue, e267_getIntegerSequenceDescriptor, e268_getIntegerSequenceDescriptor, e278_isValid, e399_getMin, e400_getMax, e403_getMin, e404_getMax, e405_getMin, e406_getMax, e409_getMin, e410_getMax, e412_getMin, e413_getMax …

// crustify:todo: e618_updateConstantMutableAddr
//   authority : dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:2143  (21 body lines, level 6)
//   original  : void IntegerSequenceDataTransferUpdater::updateConstantMutableAddr( const EvaluatedValue &new_immut_addr_ev)
//   calls     : e011_getOffset, e012_getOffset, e013_getOffset, e267_getIntegerSequenceDescriptor, e268_getIntegerSequenceDescriptor, e400_getMax, e404_getMax, e406_getMax, e410_getMax, e413_getMax, e418_getOffset, e490_updateVariableOffsetCalculation, e548_getMax, e588_getMax …

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::bridges::dataflow_ir_to_sentient::vc_vector_operands::OpId;
    use crate::islands::dataflow_ir::Values;
    use crate::islands::sentient::dialects::sentient::{Carried, Reg, RegType};
    use crate::islands::sentient::dialects::{Val, sentient};
    use crate::transform::sentient::address_pinning_and_toggle::{
        DescriptorMemoryUnit, IntegerSequenceDescriptor, PatternDescriptor, SequenceSize,
    };
    use crate::transform::sentient::analyses::{Evaluation, RegionSite};
    use crate::transform::sentient::{ForRef, IterArgIndex};

    /// The handle flavour with its answers stated as INTEGERS, so `==` on handles is `==` on values.
    #[derive(Default)]
    struct StatedEvaluator {
        held: Vec<i64>,
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
}
