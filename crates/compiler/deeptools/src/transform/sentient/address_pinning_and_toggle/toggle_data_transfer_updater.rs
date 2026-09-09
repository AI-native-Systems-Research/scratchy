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
use super::{DataTransferDescriptor, ToggleDataTransferUpdater, create_offset_value};
use crate::formats::Bits;
use crate::islands::dataflow_ir::ty::ScalarTy;
use crate::islands::sentient::dialects::{Op, Val, sentient};
use crate::transform::sentient::analyses::{EvaluatedValue, PinningSchemeManager};

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

/// THE IMMUTABLE-ADDR OPERAND AS A PLACE — what `immutable_addr_.assign(v)` (`:1943`) writes through,
/// `mlir::MutableOperandRange` being a reference into the op the descriptor describes.
///
/// ⛔ THE THREE OPS `collectDataTransfers` ADMITS (`:1400-1402`) AND NO OTHER, which is why the stop
/// names that filter rather than `getMutableAndImmutableAddr`'s wider `DT_CHECK`.
fn immutable_addr_mut(op: &mut Op, end: TransferEnd) -> &mut Val {
    match op {
        Op::Sentient(
            sentient::Op::LoadAndSend { immutable_addr, .. }
            | sentient::Op::ReceiveAndStore { immutable_addr, .. },
        ) => immutable_addr,
        Op::Sentient(sentient::Op::LoadAndStore {
            src_immutable_addr,
            dst_immutable_addr,
            ..
        }) => match end {
            TransferEnd::Src => src_immutable_addr,
            TransferEnd::Dst => dst_immutable_addr,
        },
        _ => todo!(
            "updateImmutableAddr: `immutable_addr_` on {op:?}, which \
             `isa<LoadAndSendOp, ReceiveAndStoreOp, LoadAndStoreOp>` rejects \
             (AddressPinningAndToggle.cpp:1400-1402)"
        ),
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::arch::Elements;
    use crate::islands::dataflow_ir::link::SendEnd;
    use crate::islands::sentient::dialects::sentient::{Reg, RegType, ShuffleMode};
    use crate::transform::sentient::analyses::RegionSite;

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
            pattern_desc: None,
            base_addrs: vec![EvaluatedValue(7), EvaluatedValue(9)],
            region: RegionSite::ProgramUnitBody,
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
}

// crustify:todo: e550_updateVariableOffsetCalculation
//   authority : dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:1990  (17 body lines, level 4)
//   original  : void ToggleDataTransferUpdater::updateVariableOffsetCalculation( const EvaluatedValue &new_immut_addr_ev)
//   calls     : e009_createOffsetValue, e263_getToggleDescriptor, e264_getToggleDescriptor, e402_getIterArgIndex, e485_getX

// crustify:todo: e551_updateConstantMutableAddr
//   authority : dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:2009  (29 body lines, level 4)
//   original  : void ToggleDataTransferUpdater::updateConstantMutableAddr( const EvaluatedValue &new_immut_addr_ev)
//   calls     : e009_createOffsetValue, e263_getToggleDescriptor, e264_getToggleDescriptor, e402_getIterArgIndex, e485_getX

