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
//! | `e620_HBMDataTransferDescriptor` | 620 | 6 | 34 | `dcc/src/Transform/Sentient/AddressPinningAndToggle.cpp:2583` |

use super::abstract_data_transfer_updater::{first_user, uses_of};
use super::data_transfer_descriptor_container::{is_transfer, mutable_addr_end};
use super::looping_chain_mutable_addr_descriptor::{
    increment_val, mutable_addr_of, mutable_result,
};
use super::{BaseAddrList, DataTransferDescriptor, DescriptorMemoryUnit, immutable_addr_of, op_at};
use crate::bridges::dataflow_ir_to_sentient::vc_vector_operands::OpId;
use crate::islands::sentient::dialects::{Definitions, Op};
use crate::transform::sentient::analyses::{ExpressionEvaluator, RegionSite};
use crate::units::DfirUnit;

/// Replaces: e620_HBMDataTransferDescriptor
///
/// The HBM descriptor of one transfer: its base address is whichever half of the HBM pair
/// `is_base_addr_mutable` names, and a mutable-addressed head then SUMS the increments of the
/// non-looping chain it starts into [`ChainIncrement`].
///
/// ⛔ THE RESULT INDEX IS COMPUTED ONCE, FROM `op_`, AND REUSED FOR EVERY OP IN THE CHAIN (`:2639`,
/// `:2643`), so a chain that changes which end carries the HBM address keeps reading the first op's.
/// ⛔ THE WALK IS SKIPPED ENTIRELY unless the address is mutable, the descriptor valid and its pattern
/// NOT already a looping chain (`:2637-2638`).
/// ⭐ `comp` IS DROPPED, not stored: the base constructor takes it and keeps nothing (`:637-644`).
pub fn new_hbm(
    op: OpId,
    body: &[Op],
    defs: Definitions<'_>,
    evaluator: &mut impl ExpressionEvaluator,
    is_base_addr_mutable: bool,
    increment_via_burst: BurstIncrement,
    region: RegionSite,
) -> DataTransferDescriptor {
    let mut dtd = DataTransferDescriptor {
        op,
        pattern_desc: None,
        base_addrs: BaseAddrList::new(),
        region,
        memory_unit: DescriptorMemoryUnit::Hbm {
            total_chain_increment: ChainIncrement(0),
            increment_via_burst,
        },
        base_addr: None,
        is_base_addr_mutable,
    };
    let Some(transfer) = op_at(&dtd.op, body) else {
        todo!("HBMDataTransferDescriptor: no op at {:?} (:2586)", dtd.op)
    };
    let end = mutable_addr_end(transfer, DfirUnit::Hbm, defs);
    dtd.base_addr = Some(if is_base_addr_mutable {
        mutable_addr_of(transfer, end)
    } else {
        immutable_addr_of(transfer, end)
    });
    dtd.initialize_descriptor(body, defs, evaluator);
    if dtd.is_looping_chain_mutable_addr() || !dtd.is_valid() || !is_base_addr_mutable {
        return dtd;
    }

    let mut total = 0i64;
    let mut cur_op = transfer;
    loop {
        let Some(cur_result) = mutable_result(cur_op, end) else {
            todo!(
                "HBMDataTransferDescriptor: cur_op->getResult(mutable_addr_result_idx_) is out of \
                 range on {cur_op:?}, which binds no {end:?} address result (:2643)"
            )
        };
        let uses = uses_of(body, &[cur_result]);
        let Some(user) = first_user(body, cur_result).filter(|user| is_transfer(user)) else {
            break;
        };
        if uses != 1 {
            panic!(
                "DT_CHECK_MSG(cur_result.hasOneUse(), \"Expect one use in the chain\") — {uses} \
                 uses of {cur_result:?} (:2650)"
            )
        }
        total += increment_val(cur_op, end, defs);
        cur_op = user;
    }
    if let DescriptorMemoryUnit::Hbm {
        total_chain_increment,
        ..
    } = &mut dtd.memory_unit
    {
        *total_chain_increment = ChainIncrement(total);
    }
    dtd
}

/// `total_chain_increment_` — the total mutable-address increment of a NON-LOOPING chain this
/// transfer heads (`:861`, `using AddrTy = int64_t` at `:98`), zero when it heads none.
///
/// ⛔ ITS SIGN IS THE ONLY THING READ: `getMin` widens downward unless it is `> 0` and `getMax`
/// widens upward unless it is `< 0` (`:829-855`), so a distinct type from [`BurstIncrement`] is what
/// keeps the two out of each other's argument slot in `getMax`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default, Hash)]
pub struct ChainIncrement(pub i64);

/// `increment_via_burst_` — how far the burst dimension carries this HBM transfer's address past its
/// pattern's own maximum (`:863`), zero for a transfer that does not burst.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default, Hash)]
pub struct BurstIncrement(pub i32);

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::arch::Elements;
    use crate::formats::Bits;
    use crate::islands::dataflow_ir::ty::ScalarTy;
    use crate::islands::sentient::dialects::sentient::{Extent, Reg, RegType, ShuffleMode};
    use crate::islands::sentient::dialects::{Val, dataflow, sentient};
    use crate::transform::sentient::address_pinning_and_toggle::PatternDescriptor;
    use crate::transform::sentient::analyses::{
        EvaluatedValue, Evaluation, OffsetSites, RegionSite,
    };
    use crate::units::Residency;

    /// The arena entry stated as the value's own number — `initializeDescriptor` only ever KEEPS the
    /// handle it is given, and the chain sum is read off the increment CONSTANTS instead.
    struct StatedEvaluator;

    impl ExpressionEvaluator for StatedEvaluator {
        fn evaluate_value(&mut self, _value: Val) -> Evaluation {
            todo!("e620 keeps handles, never a decoded evaluation")
        }

        fn evaluate_sum(&mut self, _lhs: &Evaluation, _rhs: &Evaluation) -> Evaluation {
            todo!("e620 never sums evaluations")
        }

        fn build_offset_value(
            &mut self,
            _evaluation: &Evaluation,
            _sites: &mut OffsetSites<'_>,
            _walked: &mut Vec<Op>,
            _ty: ScalarTy,
        ) -> Val {
            todo!("e620 builds nothing")
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

    /// One link of an HBM-headed chain: it reads `src_mutable_addr` and binds the next link's.
    fn link(src_mutable_addr: Val, src_inc: Val, results: (Val, Val)) -> Op {
        Op::Sentient(sentient::Op::LoadAndStore {
            src: Val(1),
            dst: Val(2),
            src_mutable_addr,
            src_immutable_addr: Val(20),
            src_inc,
            dst_mutable_addr: Val(21),
            dst_immutable_addr: Val(22),
            dst_inc: Val(23),
            multicast_info: None,
            results,
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

    /// 620/656 — the mutable-addressed head sums the increments of the chain it starts, and the LAST
    /// link's increment is NOT among them: the walk stops at the link whose address result nothing
    /// reads, having already counted every op that handed its address on.
    #[test]
    fn e620_sums_the_increments_of_every_link_that_hands_its_address_on() {
        let body = vec![
            get_unit(Val(1), DfirUnit::Hbm),
            get_unit(Val(2), DfirUnit::Lx),
            scalar_const(Val(3), 4096),
            scalar_const(Val(4), 64),
            scalar_const(Val(5), 128),
            scalar_const(Val(6), 256),
            link(Val(3), Val(4), (Val(10), Val(11))),
            link(Val(10), Val(5), (Val(12), Val(13))),
            link(Val(12), Val(6), (Val(14), Val(15))),
        ];
        let regions: [&[Op]; 1] = [body.as_slice()];
        let dtd = new_hbm(
            OpId::at(&[6]),
            &body,
            Definitions::from_innermost(&regions),
            &mut StatedEvaluator,
            true,
            BurstIncrement(8),
            RegionSite::default(),
        );

        assert_eq!(dtd.base_addr, Some(Val(3)));
        assert!(matches!(
            dtd.pattern_desc,
            Some(PatternDescriptor::SimpleConstant(_))
        ));
        // `64 + 128`, and not the trailing `256`.
        assert_eq!(
            dtd.memory_unit,
            DescriptorMemoryUnit::Hbm {
                total_chain_increment: ChainIncrement(192),
                increment_via_burst: BurstIncrement(8),
            }
        );
    }
}
