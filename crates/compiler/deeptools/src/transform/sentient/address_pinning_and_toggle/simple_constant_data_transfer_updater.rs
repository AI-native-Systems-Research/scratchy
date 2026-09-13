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

//! `AddressPinningAndToggle.cpp` — 2 of the campaign's 656 units (dependency level(s) [2]).
//!
//! | unit | entry | level | lines | authority path:line |
//! |---|---|---|---|---|
//! | `e420_updateImmutableAddr` | 420 | 2 | 11 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:1896` |
//! | `e421_updateConstantMutableAddr` | 421 | 2 | 23 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:1908` |


use super::looping_chain_mutable_addr_descriptor::TransferEnd;
use super::{
    DataTransferDescriptor, SimpleConstantDataTransferUpdater, create_offset_value,
    immutable_addr_mut, mutable_addr_mut,
};
use crate::formats::Bits;
use crate::islands::dataflow_ir::ty::ScalarTy;
use crate::islands::sentient::dialects::{Definitions, Op};
use crate::transform::sentient::analyses::{
    EvaluatedValue, ExpressionEvaluator, PinningSchemeManager,
};
use crate::transform::sentient::utils::{ConstKind, is_constant};

impl SimpleConstantDataTransferUpdater {
    /// Replaces: e420_updateImmutableAddr
    ///
    /// Pins the transfer's immutable address: the closest pinned address to its ONE constant base
    /// address (`:1897-1899`) becomes `op`'s immutable-addr operand, and that answer is handed back
    /// for the mutable half to re-base against.
    ///
    /// ⭐ THE ONE-ADDRESS `findClosestPinnedAddr` IS THE PAIR CALL WITH `X == Y` by its own body
    /// (`Analyses/AddressPinningScheme.h:208-219`), so the base address goes in twice — this is the
    /// degenerate case the toggle's [`super::ToggleDataTransferUpdater::update_immutable_addr`] warns
    /// against, and here it is the correct one. `element_size` is in BITS, `ty` is
    /// `mutable_addr_[0].get().getType()` (`:1029`).
    pub fn update_immutable_addr(
        self,
        dtd: &DataTransferDescriptor,
        op: &mut Op,
        end: TransferEnd,
        ty: ScalarTy,
        ps_manager: &impl PinningSchemeManager,
        element_size: Bits,
    ) -> EvaluatedValue {
        let Some(base_addr) = dtd.base_addr() else {
            panic!(
                "DT_CHECK(base_addrs_.size() == 1) (`:658`) for a simple constant holding {} base \
                 address(es)",
                dtd.base_addrs.len()
            )
        };
        let new_immut_addr_ev =
            ps_manager.find_closest_pinned_addr(base_addr, base_addr, dtd.region, element_size);
        *immutable_addr_mut(op, end) = create_offset_value(new_immut_addr_ev, ty);
        new_immut_addr_ev
    }

    /// Replaces: e421_updateConstantMutableAddr
    ///
    /// Re-bases the transfer's constant mutable address onto the address just pinned:
    /// `(const_mutable_addr + base_addr) - new_immut_addr_ev` (`:1918-1920`) becomes the new
    /// mutable-addr operand.
    ///
    /// ⭐ THE SUM RESTORES THE ABSOLUTE ADDRESS AND THE SUB RE-RELATIVISES IT: `mutable_addr_[0]` is
    /// an offset from the OLD base, so it is only meaningful once `dtd_.getBaseAddr()` is added back.
    /// ⛔ `overflowsRegister` TRUE IS THE ABORT (`:1924-1925`) — LAR/EAR cannot hold the new offset.
    pub fn update_constant_mutable_addr(
        self,
        dtd: &DataTransferDescriptor,
        op: &mut Op,
        end: TransferEnd,
        ty: ScalarTy,
        defs: Definitions<'_>,
        evaluator: &mut impl ExpressionEvaluator,
        ps_manager: &impl PinningSchemeManager,
        element_size: Bits,
        new_immut_addr_ev: EvaluatedValue,
    ) {
        let mutable_addr = *mutable_addr_mut(op, end);
        if !is_constant(mutable_addr, ConstKind::ScalarConstant, defs) {
            panic!("DT_CHECK(\"Expect constant mutable addr\") (`:1911-1912`) for {mutable_addr:?}")
        }
        let Some(base_addr) = dtd.base_addr() else {
            panic!(
                "DT_CHECK(base_addrs_.size() == 1) (`:658`) for a simple constant holding {} base \
                 address(es)",
                dtd.base_addrs.len()
            )
        };
        let const_ma_ev = evaluator.evaluate_value_handle(mutable_addr);
        let absolute = evaluator.evaluate_sum_handle(const_ma_ev, base_addr);
        let new_mut_addr_ev = evaluator.evaluate_sub_handle(absolute, new_immut_addr_ev);
        if ps_manager.overflows_register(new_mut_addr_ev, element_size) {
            panic!("DT_CHECK_MSG(\"LAR/EAR overflow detected\") (`:1924-1925`)")
        }
        *mutable_addr_mut(op, end) = create_offset_value(new_mut_addr_ev, ty);
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::arch::Elements;
    use crate::bridges::dataflow_ir_to_sentient::vc_vector_operands::OpId;
    use crate::islands::dataflow_ir::link::SendEnd;
    use crate::islands::sentient::dialects::sentient::{Reg, RegType, ShuffleMode};
    use crate::islands::sentient::dialects::{Val, sentient};
    use crate::transform::sentient::analyses::{Evaluation, OffsetSites, RegionSite};

    /// The transfer whose two addresses these updaters rewrite — `load_and_send` with the mutable
    /// address `Val(1)` and the immutable address `Val(2)`.
    fn load_and_send() -> Op {
        Op::Sentient(sentient::Op::LoadAndSend {
            mutable_addr: Val(1),
            immutable_addr: Val(2),
            increment: Val(3),
            consumer: SendEnd::to_self(Val(98)),
            result: Val(4),
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

    /// 420/656 — the ONE base address is offered to the pinning scheme as BOTH ends of the pair, and
    /// its answer is handed back. ⛔ `create_offset_value` is `EvaluatedValue::buildOffsetValue`, out
    /// of campaign scope, so the assignment stops there — the choice of address is what this checks.
    #[test]
    #[should_panic(expected = "EvaluatedValue::buildOffsetValue")]
    fn e420_pins_the_single_constant_base_address_as_a_degenerate_pair() {
        struct StatedScheme;

        impl PinningSchemeManager for StatedScheme {
            fn find_closest_pinned_addr(
                &self,
                ev_x: EvaluatedValue,
                ev_y: EvaluatedValue,
                region: RegionSite,
                element_size: Bits,
            ) -> EvaluatedValue {
                assert_eq!((ev_x, ev_y), (EvaluatedValue(7), EvaluatedValue(7)));
                assert_eq!(region, RegionSite::ProgramUnitBody);
                assert_eq!(element_size, Bits(16));
                EvaluatedValue(21)
            }
        }

        let dtd = DataTransferDescriptor {
            op: OpId::at(&[0]),
            pattern_desc: None,
            base_addrs: vec![EvaluatedValue(7)],
            region: RegionSite::ProgramUnitBody,
        };
        let mut op = load_and_send();
        SimpleConstantDataTransferUpdater.update_immutable_addr(
            &dtd,
            &mut op,
            TransferEnd::Src,
            ScalarTy::Index,
            &StatedScheme,
            Bits(16),
        );
    }

    /// 421/656 — the constant mutable address is summed with the old base and then reduced by the
    /// newly pinned immutable address, and the overflow check sees THAT handle. ⛔ stops at
    /// `buildOffsetValue`, as e420 does.
    #[test]
    #[should_panic(expected = "EvaluatedValue::buildOffsetValue")]
    fn e421_rebases_the_constant_mutable_addr_onto_the_newly_pinned_immutable_addr() {
        struct StatedEvaluator;

        impl ExpressionEvaluator for StatedEvaluator {
            fn evaluate_value(&mut self, _value: Val) -> Evaluation {
                unreachable!("the handle flavour is what a descriptor keeps")
            }

            fn evaluate_sum(&mut self, _lhs: &Evaluation, _rhs: &Evaluation) -> Evaluation {
                unreachable!("the handle flavour is what a descriptor keeps")
            }

            fn build_offset_value(
                &mut self,
                _evaluation: &Evaluation,
                _sites: &mut OffsetSites<'_>,
                _walked: &mut Vec<Op>,
                _ty: ScalarTy,
            ) -> Val {
                unreachable!("`create_offset_value` is the out-of-scope builder e421 reaches")
            }

            fn evaluate_value_handle(&mut self, value: Val) -> EvaluatedValue {
                assert_eq!(value, Val(1));
                EvaluatedValue(30)
            }

            fn evaluate_sum_handle(
                &mut self,
                lhs: EvaluatedValue,
                rhs: EvaluatedValue,
            ) -> EvaluatedValue {
                assert_eq!((lhs, rhs), (EvaluatedValue(30), EvaluatedValue(7)));
                EvaluatedValue(37)
            }

            fn evaluate_sub_handle(
                &mut self,
                lhs: EvaluatedValue,
                rhs: EvaluatedValue,
            ) -> EvaluatedValue {
                assert_eq!((lhs, rhs), (EvaluatedValue(37), EvaluatedValue(21)));
                EvaluatedValue(16)
            }
        }

        struct StatedScheme;

        impl PinningSchemeManager for StatedScheme {
            fn find_closest_pinned_addr(
                &self,
                _ev_x: EvaluatedValue,
                _ev_y: EvaluatedValue,
                _region: RegionSite,
                _element_size: Bits,
            ) -> EvaluatedValue {
                unreachable!("e421 does not pin")
            }

            fn overflows_register(&self, addr_ev: EvaluatedValue, element_size: Bits) -> bool {
                assert_eq!((addr_ev, element_size), (EvaluatedValue(16), Bits(16)));
                false
            }
        }

        let dtd = DataTransferDescriptor {
            op: OpId::at(&[0]),
            pattern_desc: None,
            base_addrs: vec![EvaluatedValue(7)],
            region: RegionSite::ProgramUnitBody,
        };
        // `mutable_addr_[0]` is the `sentient.scalar_constant` the DT_CHECK insists on.
        let body = [Op::Sentient(sentient::Op::ScalarConstant {
            value: 512,
            result: Val(1),
            reg_locale: RegType::Imm,
            ty: ScalarTy::Index,
            is_symbol: false,
        })];
        let mut op = load_and_send();
        SimpleConstantDataTransferUpdater.update_constant_mutable_addr(
            &dtd,
            &mut op,
            TransferEnd::Src,
            ScalarTy::Index,
            Definitions::from_innermost(&[&body]),
            &mut StatedEvaluator,
            &StatedScheme,
            Bits(16),
            EvaluatedValue(21),
        );
    }
}
