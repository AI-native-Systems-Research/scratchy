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

//! `AddressPinningAndToggle.cpp` — 3 of the campaign's 656 units (dependency level(s) [1, 4]).
//!
//! | unit | entry | level | lines | authority path:line |
//! |---|---|---|---|---|
//! | `e276_updateImmutableAddr` | 276 | 1 | 12 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:1936` |
//! | `e550_updateVariableOffsetCalculation` | 550 | 4 | 17 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:1990` |
//! | `e551_updateConstantMutableAddr` | 551 | 4 | 29 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:2009` |


use super::looping_chain_mutable_addr_descriptor::TransferEnd;
use super::toggle_descriptor::ToggleDescriptor;
use super::{
    DataTransferDescriptor, ToggleDataTransferUpdater, create_offset_value, immutable_addr_mut,
    mutable_addr_mut, set_iter_operand,
};
use crate::formats::Bits;
use crate::islands::dataflow_ir::ty::ScalarTy;
use crate::islands::sentient::dialects::{Definitions, Op};
use crate::transform::sentient::analyses::{
    EvaluatedValue, ExpressionEvaluator, PinningSchemeManager,
};
use crate::transform::sentient::utils::{ConstKind, is_constant};

impl ToggleDataTransferUpdater {
    /// Replaces: e276_updateImmutableAddr
    ///
    /// Pins the transfer's immutable address: the closest pinned address to the toggling PAIR
    /// (`:1939-1941`) becomes `op`'s immutable-addr operand, and the pair's own answer is handed back
    /// for the mutable half to re-base against.
    ///
    /// ⛔ BOTH ADDRESSES, NOT `getBaseAddr()`: the one-address `findClosestPinnedAddr` is a degenerate
    /// `X == Y` of this call (`Analyses/AddressPinningScheme.h:208-219`) and would minimise the wrong
    /// distance for a toggle. `element_size` is in BITS (`:229-232`), and `ty` is
    /// `mutable_addr_[0].get().getType()`, which is what `createOffsetValue` builds with (`:1029`).
    pub fn update_immutable_addr(
        &self,
        dtd: &DataTransferDescriptor,
        op: &mut Op,
        end: TransferEnd,
        ty: ScalarTy,
        ps_manager: &impl PinningSchemeManager,
        element_size: Bits,
    ) -> EvaluatedValue {
        // `*dtd_.getBaseAddrList()[0], *dtd_.getBaseAddrList()[1]` (`:1940`) — the toggle's X and Y.
        let [ev_x, ev_y] = match dtd.base_addrs.as_slice() {
            [ev_x, ev_y, ..] => [*ev_x, *ev_y],
            addrs => todo!(
                "updateImmutableAddr: `getBaseAddrList()[1]` on a toggle descriptor holding {} \
                 base address(es) (AddressPinningAndToggle.cpp:1940)",
                addrs.len()
            ),
        };
        let new_immut_addr_ev =
            ps_manager.find_closest_pinned_addr(ev_x, ev_y, dtd.region, element_size);
        // `immutable_addr_.assign(createOffsetValue(&dtd_.getOperation(), new_immut_addr_ev))`.
        *immutable_addr_mut(op, end) = create_offset_value(new_immut_addr_ev, ty);
        new_immut_addr_ev
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
    use crate::transform::sentient::address_pinning_and_toggle::{
        DescriptorMemoryUnit, PatternDescriptor,
    };
    use crate::transform::sentient::analyses::{
        Evaluation, OffsetSites, OutOfScopeEvaluator, RegionSite,
    };
    use crate::transform::sentient::{ForRef, IterArgIndex};

    /// 276/656 — the pinned address chosen for the PAIR is handed back, and the transfer's immutable
    /// address is the offset value built for it. ⛔ `create_offset_value` is
    /// `EvaluatedValue::buildOffsetValue`, out of campaign scope, so the assignment stops there — the
    /// choice of address is what this can check.
    #[test]
    #[should_panic(expected = "EvaluatedValue::buildOffsetValue")]
    fn e276_pins_the_toggling_pair_and_then_needs_the_out_of_scope_offset_builder() {
        struct StatedScheme;

        impl PinningSchemeManager for StatedScheme {
            fn find_closest_pinned_addr(
                &self,
                ev_x: EvaluatedValue,
                ev_y: EvaluatedValue,
                region: RegionSite,
                element_size: Bits,
            ) -> EvaluatedValue {
                assert_eq!((ev_x, ev_y), (EvaluatedValue(7), EvaluatedValue(9)));
                assert_eq!(region, RegionSite::ProgramUnitBody);
                assert_eq!(element_size, Bits(16));
                EvaluatedValue(21)
            }
        }

        let dtd = DataTransferDescriptor {
            op: OpId::at(&[0]),
            pattern_desc: None,
            base_addrs: vec![EvaluatedValue(7), EvaluatedValue(9)],
            region: RegionSite::ProgramUnitBody,
            memory_unit: DescriptorMemoryUnit::Lx,
            base_addr: Val(0),
            is_base_addr_mutable: false,
        };
        let mut op = Op::Sentient(sentient::Op::LoadAndSend {
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
        });
        let updater = ToggleDataTransferUpdater {
            toggle_sub: super::super::ToggleSub::of(Val(5)),
        };
        updater.update_immutable_addr(
            &dtd,
            &mut op,
            TransferEnd::Src,
            ScalarTy::Index,
            &StatedScheme,
            Bits(16),
        );
    }

    /// An evaluator that answers for a CONSTANT and nothing else — everything e551 asks for before it
    /// reaches the toggle, so that the seam it stops at is the toggle's and not the fold's.
    struct ConstantsOnly;

    impl ExpressionEvaluator for ConstantsOnly {
        fn evaluate_value(&mut self, value: Val) -> Evaluation {
            OutOfScopeEvaluator.evaluate_value(value)
        }

        fn evaluate_sum(&mut self, lhs: &Evaluation, rhs: &Evaluation) -> Evaluation {
            OutOfScopeEvaluator.evaluate_sum(lhs, rhs)
        }

        fn build_offset_value(
            &mut self,
            evaluation: &Evaluation,
            sites: &mut OffsetSites<'_>,
            walked: &mut Vec<Op>,
            ty: ScalarTy,
        ) -> Val {
            OutOfScopeEvaluator.build_offset_value(evaluation, sites, walked, ty)
        }

        fn evaluate_value_handle(&mut self, _value: Val) -> EvaluatedValue {
            EvaluatedValue(4096)
        }
    }

    /// A pinning scheme nothing here asks anything of: both updaters read it only AFTER the toggle.
    struct NoScheme;
    impl PinningSchemeManager for NoScheme {}

    /// The shape both updaters rewrite: `%1` the loop's constant initializer, `%5` the `scalar_sub`
    /// whose `$inp1` carries the constant term, and `%7` the loop `getIterArgIndex()` indexes.
    fn toggling_body() -> Vec<Op> {
        vec![
            Op::Sentient(sentient::Op::ScalarConstant {
                value: 4096,
                result: Val(1),
                reg_locale: RegType::Imm,
                ty: ScalarTy::Index,
                is_symbol: false,
            }),
            Op::Sentient(sentient::Op::For {
                iv: Val(7),
                bound: Val(0),
                bound_reg: None,
                carried: vec![sentient::Carried {
                    init: Val(1),
                    arg: Val(3),
                    result: Val(4),
                    reg: Reg {
                        locale: RegType::Lbr,
                        index: None,
                    },
                    program_header: false,
                    element_size: None,
                }],
                dbg_name: None,
                body: vec![
                    Op::Sentient(sentient::Op::ScalarSub {
                        lhs: Val(1),
                        rhs: Val(3),
                        result: Val(5),
                        reg: None,
                        element_size: None,
                        ty: ScalarTy::Index,
                    }),
                    Op::Sentient(sentient::Op::Yield {
                        results: vec![Val(5)],
                    }),
                ],
            }),
        ]
    }

    /// A transfer whose base address toggles, holding the descriptor for [`toggling_body`]'s loop.
    fn toggling_transfer() -> DataTransferDescriptor {
        DataTransferDescriptor {
            op: OpId::at(&[0]),
            pattern_desc: Some(PatternDescriptor::Toggle(ToggleDescriptor {
                outer_loop: Some(ForRef(Val(7))),
                iter_arg_index: Some(IterArgIndex(0)),
                c1: Some(EvaluatedValue(3)),
                can_be_simplified: false,
            })),
            base_addrs: vec![EvaluatedValue(7), EvaluatedValue(9)],
            region: RegionSite::ProgramUnitBody,
            memory_unit: DescriptorMemoryUnit::Lx,
            base_addr: Val(0),
            is_base_addr_mutable: false,
        }
    }

    /// `%t = sentient.load_and_send` whose MUTABLE address is `%1`, the constant e551 folds in.
    fn load_and_send_from(mutable_addr: Val) -> Op {
        Op::Sentient(sentient::Op::LoadAndSend {
            mutable_addr,
            immutable_addr: Val(2),
            increment: Val(3),
            consumer: SendEnd::to_self(Val(98)),
            result: Val(6),
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

    /// 550/656 — the shift reads the toggle's `X` before it writes anything, and `getX()` resolves the
    /// loop's constant initializer: the evaluation of that constant is where it stops.
    ///
    /// ⭐ `defs` IS BUILT FROM A SNAPSHOT, which is this crate's read-then-write idiom (`lexical_ordering
    /// ::sortings_of`): a live [`Definitions`] borrows the very scope the updater rewrites, and every
    /// read here happens before either write.
    #[test]
    #[should_panic(expected = "constant iter arg init Val(1)")]
    fn e550_rebases_the_toggle_on_the_pinned_address_and_reads_its_x_first() {
        let mut body = toggling_body();
        let snapshot = body.clone();
        let regions: [&[Op]; 1] = [&snapshot];
        let updater = ToggleDataTransferUpdater {
            toggle_sub: super::super::ToggleSub::of(Val(5)),
        };

        updater.update_variable_offset_calculation(
            &toggling_transfer(),
            &mut body,
            Definitions::from_innermost(&regions),
            ScalarTy::Index,
            &mut OutOfScopeEvaluator,
            EvaluatedValue(21),
        );
    }

    /// 551/656 — the constant mutable address is checked AND evaluated before the toggle is rebased, so
    /// a fixture that satisfies `DT_CHECK("Expect constant mutable addr")` gets as far as `getX()`.
    #[test]
    #[should_panic(expected = "constant iter arg init Val(1)")]
    fn e551_folds_the_constant_mutable_addr_in_before_rebasing_the_toggle() {
        let mut body = toggling_body();
        let snapshot = body.clone();
        let regions: [&[Op]; 1] = [&snapshot];
        let mut op = load_and_send_from(Val(1));
        let updater = ToggleDataTransferUpdater {
            toggle_sub: super::super::ToggleSub::of(Val(5)),
        };

        updater.update_constant_mutable_addr(
            &toggling_transfer(),
            &mut op,
            TransferEnd::Src,
            &mut body,
            Definitions::from_innermost(&regions),
            ScalarTy::Index,
            &mut ConstantsOnly,
            &NoScheme,
            Bits(16),
            EvaluatedValue(21),
        );
    }
}

impl ToggleDataTransferUpdater {
    /// Replaces: e550_updateVariableOffsetCalculation
    ///
    /// Shifts the whole toggle down onto the address just pinned: the sub's constant term becomes
    /// `(X - pinned) + (Y - pinned)` and the loop's iter operand becomes `Y - pinned` (`:1990-2007`).
    ///
    /// ⭐ THE TRANSFER'S OWN OPERANDS ARE NOT TOUCHED — the sub already feeds the immutable address, so
    /// re-basing `c1` and `Y` re-bases both toggled values at once (the reference's own derivation,
    /// `:1959-1985`).
    /// ⛔ EVERY READ HAPPENS BEFORE EITHER WRITE: `getX()` resolves the loop's CURRENT iter operand
    /// (e015) and `setIterOperand` then replaces exactly that.
    pub fn update_variable_offset_calculation(
        self,
        dtd: &DataTransferDescriptor,
        body: &mut [Op],
        defs: Definitions<'_>,
        ty: ScalarTy,
        evaluator: &mut impl ExpressionEvaluator,
        new_immut_addr_ev: EvaluatedValue,
    ) {
        let toggle = *self.toggle_of(dtd);
        let Some(rebased) = rebased_toggle(&toggle, body, defs, evaluator, None, new_immut_addr_ev)
        else {
            panic!(
                "DT_CHECK(isValid()) reached through `getX()` (`:236`, `:1994`) on {toggle:?}, which \
                 `processToggle` checks before asking (`:1526`)"
            )
        };
        self.write_back(&toggle, body, ty, rebased);
    }

    /// Replaces: e551_updateConstantMutableAddr
    ///
    /// The same shift with the constant mutable address FOLDED IN, so the toggle's own `scalar_sub`
    /// becomes the new mutable address: `X + old_mutable - pinned` and its `Y` twin, their sum the new
    /// constant term (`:2009-2044`).
    ///
    /// ⛔ `overflowsRegister` TRUE IS THE ABORT (`:2038-2039`), and it is checked on the CONSTANT TERM
    /// `new_c1` — not on either toggled value, which is where e421's sibling check sits.
    /// ⭐ THE LAST LINE IS WHAT MAKES THE FOLD OBSERVABLE: `mutable_addr_.assign(toggle_sub_.getResult())`
    /// (`:2044`) points the transfer at the sub instead of at the constant it used to name.
    #[allow(clippy::too_many_arguments)]
    pub fn update_constant_mutable_addr(
        self,
        dtd: &DataTransferDescriptor,
        op: &mut Op,
        end: TransferEnd,
        body: &mut [Op],
        defs: Definitions<'_>,
        ty: ScalarTy,
        evaluator: &mut impl ExpressionEvaluator,
        ps_manager: &impl PinningSchemeManager,
        element_size: Bits,
        new_immut_addr_ev: EvaluatedValue,
    ) {
        let toggle = *self.toggle_of(dtd);
        let mutable_addr = *mutable_addr_mut(op, end);
        if !is_constant(mutable_addr, ConstKind::ScalarConstant, defs) {
            panic!("DT_CHECK(\"Expect constant mutable addr\") (`:2012-2014`) for {mutable_addr:?}")
        }
        let const_ma_ev = evaluator.evaluate_value_handle(mutable_addr);
        let Some(rebased) =
            rebased_toggle(&toggle, body, defs, evaluator, Some(const_ma_ev), new_immut_addr_ev)
        else {
            panic!(
                "DT_CHECK(isValid()) reached through `getX()` (`:236`, `:2023`) on {toggle:?}, which \
                 `processToggle` checks before asking (`:1526`)"
            )
        };
        if ps_manager.overflows_register(rebased.new_c1, element_size) {
            panic!("DT_CHECK_MSG(\"LAR/EAR overflow detected\") (`:2038-2039`)")
        }
        self.write_back(&toggle, body, ty, rebased);
        // `mutable_addr_.assign(toggle_sub_.getResult())` (`:2044`).
        *mutable_addr_mut(op, end) = self.toggle_sub.result();
    }

    /// `ToggleDescriptor &toggle = dtd_.getToggleDescriptor()` (`:1991`, `:2011`) — e264, whose `None`
    /// is a descriptor this updater was built for and cannot be.
    fn toggle_of(self, dtd: &DataTransferDescriptor) -> &ToggleDescriptor {
        let Some(toggle) = dtd.toggle_descriptor() else {
            panic!(
                "`getToggleDescriptor()` on a {:?} pattern (`:668-672`) — the ToggleDataTransferUpdater \
                 is only built for a toggle (`:1528`)",
                dtd.pattern_desc
            )
        };
        toggle
    }

    /// The two writes both updaters share (`:2002-2007`, `:2040-2043`).
    fn write_back(
        self,
        toggle: &ToggleDescriptor,
        body: &mut [Op],
        ty: ScalarTy,
        rebased: RebasedToggle,
    ) {
        let (Some(outer_loop), Some(iter_arg_index)) = (toggle.outer_loop, toggle.iter_arg_index)
        else {
            panic!("DT_CHECK(isValid()) before `getOuterLoop().setIterOperand` (`:2006`) — {toggle:?}")
        };
        // `toggle_sub_.getInp1Mutable().assign(createOffsetValue(.., new_c1))` (`:2002-2004`).
        let new_c1_val = create_offset_value(rebased.new_c1, ty);
        if !self.toggle_sub.set_inp1(body, new_c1_val) {
            panic!(
                "DT_CHECK_MSG(toggle_sub_, \"expected valid toggle_sub\") (`:2002`) — no \
                 `sentient.scalar_sub` binding {:?} in this scope",
                self.toggle_sub.result()
            )
        }
        // `getOuterLoop().setIterOperand(getIterArgIndex(), createOffsetValue(.., new_y))` (`:2005-2007`).
        let new_y_val = create_offset_value(rebased.new_y, ty);
        set_iter_operand(body, outer_loop, iter_arg_index, new_y_val);
    }
}

/// THE RE-BASED TOGGLE — the constant term the sub gets and the initial value the loop gets. `new_x` is
/// deliberately absent: it is summed into `new_c1` and then never read again (`:1999`, `:2031`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RebasedToggle {
    /// `new_c1 = new_x + new_y`.
    new_c1: EvaluatedValue,
    /// `new_y`.
    new_y: EvaluatedValue,
}

/// `new_x`, `new_y` and `new_c1 = new_x + new_y` (`:1993-1999`, `:2022-2031`) — the two updaters differ
/// only in `fold_in`, the constant mutable address e551 adds to each toggled value before shifting.
///
/// ⛔ `X` IS RESOLVED, SHIFTED, AND ONLY THEN IS `Y` RESOLVED. `getX()` is `evaluateSub(*c1_, getInit())`
/// (`:229`), so `getInit()` is asked twice and in this order; an evaluator that interns in call order
/// numbers its arena by it, and hoisting `getY()` above the first shift would renumber it.
fn rebased_toggle(
    toggle: &ToggleDescriptor,
    body: &[Op],
    defs: Definitions<'_>,
    evaluator: &mut impl ExpressionEvaluator,
    fold_in: Option<EvaluatedValue>,
    new_immut_addr_ev: EvaluatedValue,
) -> Option<RebasedToggle> {
    let x = toggle.x(evaluator, body, defs)?;
    let new_x = shifted(evaluator, x, fold_in, new_immut_addr_ev);
    let y = toggle.init(body, defs);
    let new_y = shifted(evaluator, y, fold_in, new_immut_addr_ev);
    let new_c1 = evaluator.evaluate_sum_handle(new_x, new_y);
    Some(RebasedToggle { new_c1, new_y })
}

/// One toggled value moved onto the pinned address — `ev + fold_in - new_immut_addr_ev`, the sum elided
/// where there is nothing to fold in (`:1994` against `:2023-2024`).
fn shifted(
    evaluator: &mut impl ExpressionEvaluator,
    ev: EvaluatedValue,
    fold_in: Option<EvaluatedValue>,
    new_immut_addr_ev: EvaluatedValue,
) -> EvaluatedValue {
    let folded = match fold_in {
        Some(const_ma_ev) => evaluator.evaluate_sum_handle(ev, const_ma_ev),
        None => ev,
    };
    evaluator.evaluate_sub_handle(folded, new_immut_addr_ev)
}
