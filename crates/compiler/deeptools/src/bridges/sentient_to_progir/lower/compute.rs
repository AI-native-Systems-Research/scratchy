//! THE COMPUTE OPERATION HANDLERS — mac, binary, unary, ternary, add, sub, splat, opaque, samv
//! and the mask pair, plus `LowerCommonOperations` that dispatches them.
//!
//! 12 units. Every citation resolves against the authority tree
//! `/Users/nickm/git/deeptools-src` at revision `a0d29abbed`.
//!
//! | unit | level | LoC | authority |
//! |---|---|---|---|
//! | `e107_LowerCommonOperations` | 3 | 26 | `dcc/src/Conversion/SentientToProgIR/LowerSentientHelper.cpp:538` |
//! | `e108_LowerBinaryOperation` | 3 | 8 | `dcc/src/Conversion/SentientToProgIR/LowerSentientHelper.cpp:565` |
//! | `e109_LowerUnaryOperation` | 3 | 7 | `dcc/src/Conversion/SentientToProgIR/LowerSentientHelper.cpp:574` |
//! | `e110_LowerTernaryOperation` | 3 | 8 | `dcc/src/Conversion/SentientToProgIR/LowerSentientHelper.cpp:582` |
//! | `e112_LowerMACOperation` | 3 | 31 | `dcc/src/Conversion/SentientToProgIR/LowerSentientHelper.cpp:658` |
//! | `e113_LowerSubOperation` | 3 | 40 | `dcc/src/Conversion/SentientToProgIR/LowerSentientHelper.cpp:690` |
//! | `e114_LowerAddOperation` | 3 | 50 | `dcc/src/Conversion/SentientToProgIR/LowerSentientHelper.cpp:731` |
//! | `e117_LowerOpaqueOperation` | 3 | 12 | `dcc/src/Conversion/SentientToProgIR/LowerSentientHelper.cpp:939` |
//! | `e118_LowerSAMVOperation` | 3 | 14 | `dcc/src/Conversion/SentientToProgIR/LowerSentientHelper.cpp:952` |
//! | `e119_LowerSetMaskOperation` | 3 | 10 | `dcc/src/Conversion/SentientToProgIR/LowerSentientHelper.cpp:1118` |
//! | `e120_LowerIncrMaskOperation` | 3 | 10 | `dcc/src/Conversion/SentientToProgIR/LowerSentientHelper.cpp:1129` |
//! | `e124_LowerSplatOperation` | 4 | 11 | `dcc/src/Conversion/SentientToProgIR/LowerSentientHelper.cpp:927` |

use crate::arch::{Arch, IsaGen};
use crate::bridges::sentient_to_progir::construct::compute::{
    BinaryInstrOp, BinaryRefusal, ComputeComp, FmaRefusal, Mac, MacPrecision, TernaryInstrOp,
    TernaryRefusal, UnaryInstrOp, UnaryRefusal, construct_binary_instr, construct_fma_instr,
    construct_ternary_instr, construct_unary_instr,
};
use crate::bridges::sentient_to_progir::construct::scalar::{
    AddrSub, JcrOperands, LrfSub, XrfPtr, construct_jsub_instr, construct_lar_or_ear_sub_instr,
    construct_lrf_sub_instr, construct_nop_instr, construct_xrf_add_instr,
};
use crate::bridges::sentient_to_progir::lower::control::CodeGraph;
use crate::bridges::sentient_to_progir::lower::labels_and_regs::{
    LoweredOp, address_scale, update_label_and_add_to_code_graph,
};
use crate::bridges::sentient_to_progir::state::{CopyOps, OpSite, RegsToInit, UnitKey};
use crate::bridges::sentient_to_progir::uniform::block::InstrIndex;
use crate::bridges::sentient_to_progir::uniform::instr::OperandMapRefusal;
use crate::bridges::sentient_to_progir::utils::ComputeUnit;
use crate::formats::Bits;
use crate::islands::sentient::dialects::sentient::{Port, RegIndex, RegType};
use sys_arch_spec::regfile::Component;

// ⛔ ONE `crustify:todo:` PER SCHEDULED UNIT. Replace each with the ported function
// carrying `/// Replaces: eNNN_name`. A surviving TODO is open work.

/// WHAT FOLLOWS AN OP WHOSE LABEL HAS TO GO SOMEWHERE — the `getNextNode` walk's three answers.
///
/// ⛔ A `uniform.yield`'s SUCCESSOR IS ITS PARENT'S, NOT ITS OWN (`:544-547`) — the caller resolves
/// that, because the parent link is the mechanism this port supplies rather than stores.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Successor {
    /// An op that could carry the label instead.
    Labelable(OpSite),
    /// A `uniform.uniformize_regions`, which may not — as with a null successor.
    UniformizedRegions,
}

/// WHERE AN UNLOWERED OP'S LABEL ENDED UP.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LabelPlacement {
    /// The op carried no label, so there was nothing to place.
    Unlabelled,
    /// The successor took it.
    MovedToNext,
    /// A `NOP` was emitted to carry it — ⭐ AND THAT IS `++num_nops_`, one per call.
    Nop,
}

/// Replaces: e107_LowerCommonOperations
///
/// A labelled op that emits no instruction hands its label to the next op, or else a `NOP` is
/// emitted to carry it.
///
/// ⛔ THE SUCCESSOR MUST BE UNLABELLED TOO: IBM's own `uniform-nop-incorrect-label.mlir:41-43` shows
/// the move blocked by an already-labelled successor and the `NOP for label` taking the tag.
#[must_use]
pub fn lower_common_operations(
    graph: CodeGraph<'_>,
    next: Option<Successor>,
    nop_for_labels: &mut Vec<InstrIndex>,
) -> LabelPlacement {
    let Some(label) = graph.labels.get(graph.at).map(str::to_owned) else {
        return LabelPlacement::Unlabelled;
    };
    if let Some(Successor::Labelable(next)) = next
        && graph.labels.get(next).is_none()
    {
        graph.labels.claim(next, label);
        if let Some(at) = graph
            .labels
            .per_op
            .iter()
            .position(|(site, _)| *site == graph.at)
        {
            graph.labels.per_op.remove(at);
        }
        return LabelPlacement::MovedToNext;
    }
    nop_for_labels.extend(graph.region.next_instr_index());
    let nop_instr = construct_nop_instr(Some("NOP for label"));
    update_label_and_add_to_code_graph(
        graph.labels,
        graph.region,
        graph.at,
        LoweredOp::Other,
        nop_instr,
        false,
    );
    LabelPlacement::Nop
}
/// Replaces: e108_LowerBinaryOperation
///
/// The two-operand compute, appended under its op's label.
///
/// ⭐ THE REFERENCE'S `unit_op` IS UNUSED, and the instruction is optional here because
/// [`construct_binary_instr`] has an operator with no opcode on this component.
#[must_use]
pub fn lower_binary_operation<A: Arch>(
    comp: ComputeComp,
    op: &BinaryInstrOp<'_>,
    graph: CodeGraph<'_>,
) -> Vec<BinaryRefusal> {
    let super_instr = construct_binary_instr::<A>(comp, op);
    if let Some(instr) = super_instr.instr {
        update_label_and_add_to_code_graph(
            graph.labels,
            graph.region,
            graph.at,
            LoweredOp::Other,
            instr,
            false,
        );
    }
    super_instr.refused
}
/// Replaces: e109_LowerUnaryOperation
///
/// The one-operand compute, appended under its op's label.
#[must_use]
pub fn lower_unary_operation<A: Arch>(
    comp: ComputeComp,
    op: &UnaryInstrOp<'_>,
    graph: CodeGraph<'_>,
) -> Vec<UnaryRefusal> {
    let instr = construct_unary_instr::<A>(comp, op);
    update_label_and_add_to_code_graph(
        graph.labels,
        graph.region,
        graph.at,
        LoweredOp::Other,
        instr.instr,
        false,
    );
    instr.refused
}
/// Replaces: e110_LowerTernaryOperation
///
/// The three-operand compute — the select — appended under its op's label.
///
/// ⭐ ONLY THE PE AND THE SFP HAVE ONE, which is why this takes a [`ComputeUnit`] where the binary
/// and unary lowerings take a [`ComputeComp`].
#[must_use]
pub fn lower_ternary_operation<A: Arch>(
    unit: ComputeUnit,
    op: &TernaryInstrOp<'_>,
    graph: CodeGraph<'_>,
) -> Vec<TernaryRefusal> {
    let super_instr = construct_ternary_instr::<A>(unit, op);
    update_label_and_add_to_code_graph(
        graph.labels,
        graph.region,
        graph.at,
        LoweredOp::Other,
        super_instr.instr,
        false,
    );
    super_instr.refused
}
/// WHAT A `sentient.mac` SAYS ABOUT THE XRF POINTERS — `xrfReadIncr`, `xrfWriteIncr`, and whether
/// the op binds a result at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MacXrfState {
    /// `getXrfReadIncrSigned()`.
    pub read_incr: i64,
    /// `getXrfWriteIncrSigned()`.
    pub write_incr: i64,
    /// ⛔ `getResults().size() > 0` GATES BOTH PREDICATES (`SentientOps.cpp:1656-1675`) — a MAC that
    /// binds nothing touches neither pointer however its ports are spelled.
    pub binds_result: bool,
}

/// HOW FAR THE XRF READ POINTER TRAVELS BY ITSELF AFTER A MAC — `getXrfRdPtrIncrValAfterMAC`
/// (`Utils/DccExtContext.cpp:354`).
///
/// ⭐ AN INTEGER PRECISION READS TWO PER MAC WHERE A FLOAT READS ONE, and sen1p5 reads eight
/// whatever the precision.
#[must_use]
pub fn xrf_rd_ptr_incr_after_mac<A: Arch>(compute: MacPrecision) -> i64 {
    match (A::GEN, compute) {
        (IsaGen::Sen1p5, _) => 8,
        (_, MacPrecision::Int4 | MacPrecision::Int8 | MacPrecision::Mxint4) => 2,
        (_, _) => 1,
    }
}

/// Replaces: e112_LowerMACOperation
///
/// The fused multiply-add, followed by an `XRFACCESS` correcting whichever pointer the op moves by
/// other than the amount the hardware moves it anyway.
///
/// ⛔ READ WINS OVER WRITE: the two are an `else if`, so a MAC both reading and writing the xrf only
/// ever corrects the read pointer.
/// ⛔ THE CORRECTION NEVER TAKES THE LABEL — `force_not_add_label` is `true`, because the MAC took it.
#[must_use]
pub fn lower_mac_operation<A: Arch>(
    comp: ComputeComp,
    mac: &Mac,
    xrf: MacXrfState,
    graph: CodeGraph<'_>,
) -> Vec<FmaRefusal> {
    let super_instr = construct_fma_instr::<A>(mac, comp);
    update_label_and_add_to_code_graph(
        graph.labels,
        graph.region,
        graph.at,
        LoweredOp::Other,
        super_instr.instr,
        false,
    );
    let reads_xrf = xrf.binds_result
        && [&mac.op_a, &mac.op_b, &mac.op_c]
            .iter()
            .any(|operand| operand.operand.port == Port::Xrf);
    let writes_xrf = xrf.binds_result && mac.result.forwarding.contains(&Port::Xrf);
    let correction = if reads_xrf {
        let required = xrf_rd_ptr_incr_after_mac::<A>(mac.compute);
        (xrf.read_incr != required).then(|| (XrfPtr::Read, xrf.read_incr - required))
    } else if writes_xrf {
        let unroll_factor = i64::from(mac.unroll_factor.count());
        (unroll_factor != xrf.write_incr).then(|| (XrfPtr::Write, xrf.write_incr - unroll_factor))
    } else {
        None
    };
    if let Some((ptr, delta)) = correction {
        update_label_and_add_to_code_graph(
            graph.labels,
            graph.region,
            graph.at,
            LoweredOp::Other,
            construct_xrf_add_instr::<A>(ptr, delta),
            true,
        );
    }
    super_instr.refused
}
/// WHICH SUBTRACT A `sentient.sub`'s TARGET REGISTER MAKES IT — `getRegLocale()`'s three arms and
/// the locales that have no ProgIR at all.
#[derive(Debug, Clone, PartialEq)]
pub enum ScalarSubForm {
    /// `jcr` — the jump-condition subtract, whose target register is its own operand.
    Jcr {
        /// `ConstructJSUBInstr`'s operands.
        operands: JcrOperands,
        /// The register the difference lands in.
        target: RegIndex,
    },
    /// `lrf`.
    Lrf(LrfSub),
    /// `lar` or `ear` — one lowering, the file chosen inside it.
    Addr(AddrSub),
    /// Anything else — *"Unable to find Prog.IR for Sentient SUB"*, returned rather than signalled.
    Unsupported(RegType),
}

/// A `sentient.sub` AS THIS LOWERING READS IT.
#[derive(Debug, Clone, PartialEq)]
pub struct ScalarSub {
    /// Which subtract it is.
    pub form: ScalarSubForm,
    /// `element_size` — `None` when the attribute is absent, so the scaled default applies.
    pub element_size: Option<Bits>,
}

/// WHAT A SUBTRACT COULD NOT LOWER.
#[derive(Debug, Clone, PartialEq)]
pub enum SubRefusal {
    /// The locale has no ProgIR subtract.
    Locale(RegType),
    /// A per-unit operand map could not hold its entries.
    Map(OperandMapRefusal),
}

/// A LOWERED `sentient.sub` — what it cost and what it could not express.
#[derive(Debug, Clone, PartialEq)]
pub struct LoweredSub {
    /// `++num_copy_ops_`.
    pub copy_ops: CopyOps,
    /// The offenders, in the order the reference reaches them.
    pub refused: Vec<SubRefusal>,
}

/// Replaces: e113_LowerSubOperation
///
/// The scalar subtract, as a `JSUB` on the jump register or as a register-file subtract that may
/// need a following copy.
///
/// ⛔ ONLY `instr[0]` TAKES THE OP'S LABEL, AND `instr[0]` IS THE `LARREGCOPY` when the source is not
/// already in the target — `return {copy_super_instr, super_instr}` (`ConstructProgIRHelper.cpp:1174`).
#[must_use]
pub fn lower_sub_operation<A: Arch>(
    comp: Component,
    sub: &ScalarSub,
    units: &[UnitKey],
    full_reg_init: bool,
    regs_to_init: &mut RegsToInit,
    graph: CodeGraph<'_>,
) -> LoweredSub {
    let element_size = sub
        .element_size
        .unwrap_or(Bits(8 * address_scale::<A>(comp).get()));
    let update = match &sub.form {
        ScalarSubForm::Jcr { operands, target } => {
            update_label_and_add_to_code_graph(
                graph.labels,
                graph.region,
                graph.at,
                LoweredOp::Other,
                construct_jsub_instr(operands.clone(), *target),
                false,
            );
            return LoweredSub {
                copy_ops: CopyOps(0),
                refused: Vec::new(),
            };
        }
        ScalarSubForm::Unsupported(locale) => {
            return LoweredSub {
                copy_ops: CopyOps(0),
                refused: vec![SubRefusal::Locale(*locale)],
            };
        }
        ScalarSubForm::Lrf(lrf) => construct_lrf_sub_instr::<A>(
            lrf,
            comp,
            element_size,
            units,
            full_reg_init,
            regs_to_init,
        ),
        ScalarSubForm::Addr(addr) => construct_lar_or_ear_sub_instr::<A>(
            addr,
            comp,
            element_size,
            units,
            full_reg_init,
            regs_to_init,
        ),
    };
    for (i, instr) in update.instrs.into_iter().enumerate() {
        update_label_and_add_to_code_graph(
            graph.labels,
            graph.region,
            graph.at,
            LoweredOp::Other,
            instr,
            i > 0,
        );
    }
    LoweredSub {
        copy_ops: update.copy_ops,
        refused: update.refused.into_iter().map(SubRefusal::Map).collect(),
    }
}
// crustify:todo: e114_LowerAddOperation
// crustify:todo: e117_LowerOpaqueOperation
// crustify:todo: e118_LowerSAMVOperation
// crustify:todo: e119_LowerSetMaskOperation
// crustify:todo: e120_LowerIncrMaskOperation
// crustify:todo: e124_LowerSplatOperation

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::arch::Dd2;
    use crate::bridges::sentient_to_progir::construct::compute::{
        BinaryOperand, ComputeSlot, MacOperand,
    };
    use crate::bridges::sentient_to_progir::construct::reg_init::ImmSource;
    use crate::bridges::sentient_to_progir::construct::{descriptive, int};
    use crate::bridges::sentient_to_progir::lower::control::RegionIndex;
    use crate::bridges::sentient_to_progir::state::Labels;
    use crate::bridges::sentient_to_progir::uniform::block::UniformInstrBlocks;
    use crate::bridges::sentient_to_progir::uniform::instr::Comment;
    use crate::islands::progir::{OpCode, OperandField};
    use crate::islands::sentient::dialects::sentient::{
        Binary, BinaryOp, FmaMode, IStateIndex, LrfIndex, Operand as SenOperand, Precision,
        ResultPorts, UnaryOp, UnrollFactor,
    };

    /// The op every lowering here appends to, and what came out: `(opcode, tag, comment)`.
    fn emitted(region: &UniformInstrBlocks) -> Vec<(OpCode, Option<String>, Comment)> {
        region.blocks[0].instr_lists()[0]
            .iter()
            .map(|instr| (instr.opcode, instr.tag.clone(), instr.comment.clone()))
            .collect()
    }

    /// e107 — ⭐⭐ IBM'S OWN CASE: `uniform-nop-incorrect-label.mlir:41-43` labels the successor
    /// already, so the move is blocked and `LX_NOP ::  // NOP for label` carries the tag instead.
    #[test]
    fn a_blocked_label_becomes_a_nop_and_an_unblocked_one_moves_on() {
        let mut labels = Labels::default();
        labels.claim(OpSite(0), "if-label-1-end".to_owned());
        labels.claim(OpSite(1), "if-label-0-end".to_owned());
        let mut region = UniformInstrBlocks::default();
        // ⛔ `next_instr_index` HAS NO ANSWER FOR AN EMPTY REGION (`block:  -1` in the reference), so
        // the recorded index is only meaningful once a block is open.
        region.add_instruction_to_last_block(construct_nop_instr(None));
        let mut nop_for_labels = Vec::new();
        assert_eq!(
            lower_common_operations(
                CodeGraph {
                    labels: &mut labels,
                    region: &mut region,
                    at: OpSite(0),
                },
                Some(Successor::Labelable(OpSite(1))),
                &mut nop_for_labels,
            ),
            LabelPlacement::Nop
        );
        assert_eq!(
            nop_for_labels,
            vec![InstrIndex {
                block: 0,
                region: RegionIndex(0),
                instr: 1,
            }]
        );
        assert_eq!(
            emitted(&region),
            vec![
                (OpCode::NOP, None, Comment::Common("NOP".to_owned())),
                (
                    OpCode::NOP,
                    Some("if-label-1-end".to_owned()),
                    Comment::Common("NOP for label".to_owned()),
                ),
            ]
        );
        // An unlabelled successor takes it instead, and nothing is emitted.
        let mut moved = Labels::default();
        moved.claim(OpSite(0), "if-label-1-end".to_owned());
        let mut empty = UniformInstrBlocks::default();
        assert_eq!(
            lower_common_operations(
                CodeGraph {
                    labels: &mut moved,
                    region: &mut empty,
                    at: OpSite(0),
                },
                Some(Successor::Labelable(OpSite(1))),
                &mut nop_for_labels,
            ),
            LabelPlacement::MovedToNext
        );
        assert_eq!(moved.per_op, vec![(OpSite(1), "if-label-1-end".to_owned())]);
        assert!(empty.blocks.is_empty());
        // ⛔ A `uniform.uniformize_regions` SUCCESSOR MAY NOT TAKE IT, as with no successor at all.
        let mut blocked = Labels::default();
        blocked.claim(OpSite(0), "if-label-1-end".to_owned());
        let mut region = UniformInstrBlocks::default();
        assert_eq!(
            lower_common_operations(
                CodeGraph {
                    labels: &mut blocked,
                    region: &mut region,
                    at: OpSite(0),
                },
                Some(Successor::UniformizedRegions),
                &mut nop_for_labels,
            ),
            LabelPlacement::Nop
        );
        // And an unlabelled op places nothing.
        assert_eq!(
            lower_common_operations(
                CodeGraph {
                    labels: &mut Labels::default(),
                    region: &mut UniformInstrBlocks::default(),
                    at: OpSite(0),
                },
                None,
                &mut nop_for_labels,
            ),
            LabelPlacement::Unlabelled
        );
    }

    /// e108 — IBM's `sub.mlir:8` FNMS, appended under its op's label.
    #[test]
    fn a_binary_operation_is_appended_under_its_own_label() {
        let op_a = SenOperand::from(Port::Lx);
        let op_b = SenOperand::from(Port::Pt);
        let result = ResultPorts {
            forwarding: vec![Port::Lx],
            ..ResultPorts::default()
        };
        let mut labels = Labels::default();
        labels.claim(OpSite(0), "sub-0".to_owned());
        let mut region = UniformInstrBlocks::default();
        let refused = lower_binary_operation::<Dd2>(
            ComputeComp::Pe,
            &BinaryInstrOp {
                op_a: BinaryOperand {
                    slot: ComputeSlot::Src2,
                    operand: &op_a,
                },
                op_b: BinaryOperand {
                    slot: ComputeSlot::Src0,
                    operand: &op_b,
                },
                binary_op: Binary::Plain(BinaryOp::Sub),
                result: &result,
                compute_precision: Precision::Fp16,
                unroll_factor: UnrollFactor::X8,
                fold_mode: None,
                mask: 0,
                dbg_name: None,
            },
            CodeGraph {
                labels: &mut labels,
                region: &mut region,
                at: OpSite(0),
            },
        );
        assert_eq!(refused, vec![]);
        assert_eq!(
            emitted(&region),
            vec![(OpCode::FNMS, Some("sub-0".to_owned()), Comment::None)]
        );
    }

    /// e109 — IBM's `unary_op.mlir:8` FEST, appended under its op's label.
    #[test]
    fn a_unary_operation_is_appended_under_its_own_label() {
        let from_lx = SenOperand::from(Port::Lx);
        let to_lx = ResultPorts {
            forwarding: vec![Port::Lx],
            ..ResultPorts::default()
        };
        let mut labels = Labels::default();
        labels.claim(OpSite(0), "est-0".to_owned());
        let mut region = UniformInstrBlocks::default();
        let refused = lower_unary_operation::<Dd2>(
            ComputeComp::Sfp,
            &UnaryInstrOp {
                op_a: BinaryOperand {
                    slot: ComputeSlot::Src0,
                    operand: &from_lx,
                },
                unary_op: UnaryOp::ExpA,
                result: &to_lx,
                compute_precision: Precision::Fp16,
                unroll_factor: UnrollFactor::X1,
                fold_mode: None,
                mask: 0,
                dbg_name: None,
            },
            CodeGraph {
                labels: &mut labels,
                region: &mut region,
                at: OpSite(0),
            },
        );
        assert_eq!(refused, vec![]);
        assert_eq!(
            emitted(&region),
            vec![(OpCode::FEST, Some("est-0".to_owned()), Comment::None)]
        );
    }

    /// e110 — IBM's `ternary.mlir:13` SELECT, appended under its op's label.
    #[test]
    fn a_ternary_operation_is_appended_under_its_own_label() {
        let predicate = SenOperand {
            precision: Precision::Int1,
            ..SenOperand::from(Port::IState(IStateIndex::S0))
        };
        let op_b = SenOperand::from(Port::Lx);
        let op_c = SenOperand::from(Port::Lrf(LrfIndex::L2));
        let result = ResultPorts {
            forwarding: vec![Port::Lx],
            ..ResultPorts::default()
        };
        let mut labels = Labels::default();
        labels.claim(OpSite(0), "select-0".to_owned());
        let mut region = UniformInstrBlocks::default();
        let refused = lower_ternary_operation::<Dd2>(
            ComputeUnit::Sfp,
            &TernaryInstrOp {
                op_a: &predicate,
                op_b: BinaryOperand {
                    slot: ComputeSlot::Src2,
                    operand: &op_b,
                },
                op_c: BinaryOperand {
                    slot: ComputeSlot::Src0,
                    operand: &op_c,
                },
                result: &result,
                compute_precision: Precision::Fp16,
                unroll_factor: UnrollFactor::X1,
                unroll_incr_logical_result: false,
                mask: 0,
                dbg_name: None,
            },
            CodeGraph {
                labels: &mut labels,
                region: &mut region,
                at: OpSite(0),
            },
        );
        assert_eq!(refused, vec![]);
        assert_eq!(
            emitted(&region),
            vec![(OpCode::SELECT, Some("select-0".to_owned()), Comment::None)]
        );
    }

    /// e112 — ⭐⭐ IBM'S OWN TWO MACS, both in `if_else_label4.mlir`: the int8 one reading the xrf
    /// with `xrfReadIncr = 0` (`:142`) needs `PTOP_XRFACCESS :: rdptr_imm:62 rdptr_upd:incr
    /// wrptr_upd:no  // xrf add` (`:34`), and the one writing the xrf with `xrfWriteIncr = 4` at
    /// `unrollFactor = x4` (`:120`) needs nothing (`:12-13`).
    #[test]
    fn an_int8_mac_reading_the_xrf_corrects_the_pointer_the_hardware_moved_two() {
        let operand = |port, slot| MacOperand {
            operand: SenOperand {
                port,
                forwarding: Vec::new(),
                precision: Precision::Fp16,
                data_id: None,
                port_id: None,
                unroll_incr: false,
            },
            slot,
        };
        let mut reading = Mac {
            op_a: MacOperand {
                operand: SenOperand {
                    forwarding: vec![Port::East],
                    ..SenOperand::from(Port::West)
                },
                slot: ComputeSlot::Src2,
            },
            op_b: operand(Port::Xrf, ComputeSlot::Src0),
            op_c: operand(Port::Zero, ComputeSlot::Src1),
            result: ResultPorts {
                forwarding: vec![Port::Lrf(LrfIndex::L0)],
                precision: Precision::Int24,
                unroll_incr: false,
            },
            mode: FmaMode::FusedMulAdd,
            compute: MacPrecision::Int8,
            fold_mode: None,
            unroll_factor: UnrollFactor::X1,
            mask: 0,
            is_data_weight: false,
            dbg_name: None,
        };
        let mut labels = Labels::default();
        labels.claim(OpSite(0), "mac-0".to_owned());
        let mut region = UniformInstrBlocks::default();
        let refused = lower_mac_operation::<Dd2>(
            ComputeComp::Pt,
            &reading,
            MacXrfState {
                read_incr: 0,
                write_incr: 0,
                binds_result: true,
            },
            CodeGraph {
                labels: &mut labels,
                region: &mut region,
                at: OpSite(0),
            },
        );
        assert_eq!(refused, vec![]);
        let instrs = &region.blocks[0].instr_lists()[0];
        assert_eq!(
            emitted(&region),
            vec![
                (OpCode::IMA8, Some("mac-0".to_owned()), Comment::None),
                (
                    OpCode::XRFACCESS,
                    None,
                    Comment::Common("xrf add".to_owned()),
                ),
            ],
            "⛔ THE CORRECTION NEVER TAKES THE LABEL"
        );
        assert_eq!(
            instrs[1].common_fields,
            vec![
                (OperandField::RdptrImm, int(62)),
                (OperandField::RdptrUpd, descriptive("incr")),
                (OperandField::WrptrUpd, descriptive("no")),
            ]
        );
        // The write side, at the increment the unroll factor already implies.
        reading.op_b = operand(Port::Zero, ComputeSlot::Src0);
        reading.result.forwarding = vec![Port::Xrf];
        reading.unroll_factor = UnrollFactor::X4;
        let mut region = UniformInstrBlocks::default();
        let refused = lower_mac_operation::<Dd2>(
            ComputeComp::Pt,
            &reading,
            MacXrfState {
                read_incr: 0,
                write_incr: 4,
                binds_result: true,
            },
            CodeGraph {
                labels: &mut Labels::default(),
                region: &mut region,
                at: OpSite(0),
            },
        );
        assert_eq!(refused, vec![]);
        assert_eq!(region.blocks[0].instr_lists()[0].len(), 1);
    }

    /// e113 — IBM's `uniform_scalar_sub.mlir:9-10` pair, whose `LARREGCOPY` comes FIRST and so is the
    /// instruction the op's label names.
    #[test]
    fn only_the_first_instruction_of_a_lar_sub_takes_the_ops_label() {
        let mut labels = Labels::default();
        labels.claim(OpSite(0), "sub-0".to_owned());
        let mut region = UniformInstrBlocks::default();
        let lowered = lower_sub_operation::<Dd2>(
            Component::L3lu,
            &ScalarSub {
                form: ScalarSubForm::Addr(AddrSub {
                    tgt: Some(RegIndex::at::<1>()),
                    imm: ImmSource::Constant(233_472),
                    src1: Some(RegIndex::at::<0>()),
                }),
                element_size: Some(Bits(8)),
            },
            &[],
            false,
            &mut RegsToInit::new(),
            CodeGraph {
                labels: &mut labels,
                region: &mut region,
                at: OpSite(0),
            },
        );
        assert_eq!(
            lowered,
            LoweredSub {
                copy_ops: CopyOps(1),
                refused: vec![],
            }
        );
        assert_eq!(
            emitted(&region),
            vec![
                (
                    OpCode::LARREGCOPY,
                    Some("sub-0".to_owned()),
                    Comment::Common("LAR <- LAR".to_owned()),
                ),
                (
                    OpCode::SUBLARIMM,
                    None,
                    Comment::Common("lar sub".to_owned()),
                ),
            ]
        );
        // A locale with no ProgIR subtract comes back as the offender.
        assert_eq!(
            lower_sub_operation::<Dd2>(
                Component::L3lu,
                &ScalarSub {
                    form: ScalarSubForm::Unsupported(RegType::Gtr),
                    element_size: None,
                },
                &[],
                false,
                &mut RegsToInit::new(),
                CodeGraph {
                    labels: &mut Labels::default(),
                    region: &mut UniformInstrBlocks::default(),
                    at: OpSite(0),
                },
            ),
            LoweredSub {
                copy_ops: CopyOps(0),
                refused: vec![SubRefusal::Locale(RegType::Gtr)],
            }
        );
    }
}
