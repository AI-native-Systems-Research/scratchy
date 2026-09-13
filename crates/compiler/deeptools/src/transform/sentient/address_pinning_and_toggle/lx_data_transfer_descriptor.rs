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

//! `AddressPinningAndToggle.cpp` — 1 of the campaign's 656 units (dependency level(s) [6]).
//!
//! | unit | entry | level | lines | authority path:line |
//! |---|---|---|---|---|
//! | `e619_LXDataTransferDescriptor` | 619 | 6 | 23 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:2553` |

use super::data_transfer_descriptor_container::mutable_addr_end;
use super::{
    BaseAddrList, DataTransferDescriptor, DescriptorMemoryUnit, immutable_addr_of, op_at,
    unit_type_of,
};
use crate::bridges::dataflow_ir_to_sentient::vc_vector_operands::OpId;
use crate::islands::sentient::dialects::{Definitions, Op, sentient};
use crate::transform::sentient::analyses::{ExpressionEvaluator, RegionSite};
use crate::units::DfirUnit;

/// Replaces: e619_LXDataTransferDescriptor
///
/// The LX descriptor of one transfer: `comp` is checked against which way the transfer goes, an IBR
/// write for gather is left invalid, and every other transfer recognises its LX IMMUTABLE address.
///
/// ⛔ THE EARLY RETURN IS A DESCRIPTOR WITH NO ADDRESS AND NO PATTERN, deliberately: for an IBR write
/// for gather neither end is LX, so `getMutableAndImmutableAddr(op, LX)` would answer
/// `{nullptr, nullptr}` and abort on the `DT_CHECK_MSG` below — [`DataTransferDescriptor::is_valid`]
/// answers `false` on what comes back instead.
/// ⭐ `comp` IS NOT A FIELD: the base constructor drops it (`:637-644`) and these two checks, which
/// only ever see the `L3SU`/`L3LU` halves, are its only readers.
pub fn new_lx(
    comp: DfirUnit,
    op: OpId,
    body: &[Op],
    defs: Definitions<'_>,
    evaluator: &mut impl ExpressionEvaluator,
    region: RegionSite,
) -> DataTransferDescriptor {
    let mut dtd = DataTransferDescriptor {
        op,
        pattern_desc: None,
        base_addrs: BaseAddrList::new(),
        region,
        memory_unit: DescriptorMemoryUnit::Lx,
        base_addr: None,
        is_base_addr_mutable: false,
    };
    let Some(transfer) = op_at(&dtd.op, body) else {
        todo!("LXDataTransferDescriptor: no op at {:?} (:2556)", dtd.op)
    };
    match transfer {
        Op::Sentient(sentient::Op::LoadAndSend { .. }) => {
            if comp != DfirUnit::L3su {
                panic!(
                    "DT_CHECK_MSG(comp == L3SU, \"load_and_send unexpected in this unit\") — \
                     {comp:?} (:2559-2560)"
                )
            }
        }
        Op::Sentient(sentient::Op::ReceiveAndStore { .. }) => {
            if comp != DfirUnit::L3lu {
                panic!(
                    "DT_CHECK_MSG(comp == L3LU, \"receive_and_store unexpected in this unit\") — \
                     {comp:?} (:2561-2564)"
                )
            }
        }
        Op::Sentient(sentient::Op::LoadAndStore { .. }) => {
            if is_ibr_write_for_gather(transfer, defs) {
                // `LLVM_DEBUG(.. "Invalid descriptor created for IBR-write for gather"); return;`
                return dtd;
            }
        }
        _ => todo!(
            "LXDataTransferDescriptor: cast<sentient::LoadAndStoreOp>({transfer:?}), which the \
             three ops collectDataTransfers admits are the only ones to survive (:2565)"
        ),
    }
    let end = mutable_addr_end(transfer, DfirUnit::Lx, defs);
    dtd.base_addr = Some(immutable_addr_of(transfer, end));
    dtd.initialize_descriptor(body, defs, evaluator);
    dtd
}

/// `dcc::utils::L3GatherScatterChecker(ls_op).isIBRWriteForGather()` (`Analyses/Utils.hpp:195-197`,
/// `Analyses/Utils.cpp:608-632`) — ⭐ `is_ibr_write_ && is_ibr_write_for_gather_` REDUCES TO THE
/// SECOND, because `is_ibr_write_` is that flag OR the attribute (`:621-624`).
fn is_ibr_write_for_gather(op: &Op, defs: Definitions<'_>) -> bool {
    let Op::Sentient(sentient::Op::LoadAndStore { src, dst, .. }) = op else {
        return false;
    };
    unit_type_of(*src, defs) == Some(DfirUnit::Hbm)
        && unit_type_of(*dst, defs) == Some(DfirUnit::L3Ibr)
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::arch::Elements;
    use crate::formats::Bits;
    use crate::islands::dataflow_ir::ty::ScalarTy;
    use crate::islands::sentient::dialects::sentient::{Extent, Reg, RegType, ShuffleMode};
    use crate::islands::sentient::dialects::{Val, dataflow};
    use crate::transform::sentient::address_pinning_and_toggle::PatternDescriptor;
    use crate::transform::sentient::analyses::{
        EvaluatedValue, Evaluation, OffsetSites, RegionSite,
    };
    use crate::units::Residency;

    /// The arena entry stated as the value's own number — `initializeDescriptor` only ever KEEPS the
    /// handle it is given, so nothing here needs to decode one.
    struct StatedEvaluator;

    impl ExpressionEvaluator for StatedEvaluator {
        fn evaluate_value(&mut self, _value: Val) -> Evaluation {
            todo!("e619 keeps handles, never a decoded evaluation")
        }

        fn evaluate_sum(&mut self, _lhs: &Evaluation, _rhs: &Evaluation) -> Evaluation {
            todo!("e619 never sums")
        }

        fn build_offset_value(
            &mut self,
            _evaluation: &Evaluation,
            _sites: &mut OffsetSites<'_>,
            _walked: &mut Vec<Op>,
            _ty: ScalarTy,
        ) -> Val {
            todo!("e619 builds nothing")
        }

        fn evaluate_value_handle(&mut self, value: Val) -> EvaluatedValue {
            EvaluatedValue(value.0)
        }
    }

    fn get_unit(result: Val, unit: DfirUnit) -> Op {
        Op::Dataflow(dataflow::Op::GetUnit {
            result,
            residency: Residency::Global,
            unit,
            num_folds: None,
            reg_locale: None,
        })
    }

    fn scalar_const(result: Val, value: i64) -> Op {
        Op::Sentient(sentient::Op::ScalarConstant {
            value,
            result,
            reg_locale: RegType::Imm,
            ty: ScalarTy::Index,
            is_symbol: false,
        })
    }

    fn load_and_store(src: Val, dst: Val, dst_immutable_addr: Val) -> Op {
        Op::Sentient(sentient::Op::LoadAndStore {
            src,
            dst,
            src_mutable_addr: Val(30),
            src_immutable_addr: Val(31),
            src_inc: Val(32),
            dst_mutable_addr: Val(33),
            dst_immutable_addr,
            dst_inc: Val(34),
            multicast_info: None,
            results: (Val(40), Val(41)),
            extent: Extent::of(Elements(8), Bits(16)),
            stride: 1,
            rotate_val: None,
            shuffle_mode: ShuffleMode::NoShuffle,
            src_reg: Reg {
                locale: RegType::Lar,
                index: None,
            },
            dst_reg: Reg {
                locale: RegType::Lbr,
                index: None,
            },
            dir: None,
            is_ibr_write: false,
            dbg_name: None,
        })
    }

    /// 619/656 — an HBM-to-LX store recognises the LX end's IMMUTABLE address, and the constant it
    /// finds there becomes a simple-constant pattern over that one base address.
    #[test]
    fn e619_takes_the_lx_ends_immutable_address_as_the_base() {
        let body = vec![
            get_unit(Val(1), DfirUnit::Hbm),
            get_unit(Val(2), DfirUnit::Lx),
            scalar_const(Val(3), 4096),
            load_and_store(Val(1), Val(2), Val(3)),
        ];
        let regions: [&[Op]; 1] = [body.as_slice()];
        let dtd = new_lx(
            DfirUnit::L3lu,
            OpId::at(&[3]),
            &body,
            Definitions::from_innermost(&regions),
            &mut StatedEvaluator,
            RegionSite::default(),
        );

        assert_eq!(dtd.base_addr, Some(Val(3)));
        assert_eq!(dtd.base_addrs, vec![EvaluatedValue(3)]);
        assert!(matches!(
            dtd.pattern_desc,
            Some(PatternDescriptor::SimpleConstant(_))
        ));
        assert!(dtd.is_valid());
    }

    /// 619/656, the negative — an HBM-to-L3IBR store has NEITHER end on the LX, so the descriptor
    /// comes back with no address and no pattern rather than reaching a `getMutableAndImmutableAddr`
    /// that would abort.
    #[test]
    fn e619_leaves_an_ibr_write_for_gather_invalid() {
        let body = vec![
            get_unit(Val(1), DfirUnit::Hbm),
            get_unit(Val(2), DfirUnit::L3Ibr),
            scalar_const(Val(3), 4096),
            load_and_store(Val(1), Val(2), Val(3)),
        ];
        let regions: [&[Op]; 1] = [body.as_slice()];
        let dtd = new_lx(
            DfirUnit::L3lu,
            OpId::at(&[3]),
            &body,
            Definitions::from_innermost(&regions),
            &mut StatedEvaluator,
            RegionSite::default(),
        );

        assert_eq!(dtd.base_addr, None);
        assert!(dtd.pattern_desc.is_none());
        assert!(!dtd.is_valid());
    }
}
