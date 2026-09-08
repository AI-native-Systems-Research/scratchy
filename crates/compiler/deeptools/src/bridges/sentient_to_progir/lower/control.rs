//! THE CONTROL FLOW AND THE UNIFORM REGIONS — for, yield, return, sync, nop, set-send-dest, and
//! the uniform-operation pair.
//!
//! ⛔ `LowerUniformOperations` AND `GenerateProgIR` ARE MUTUALLY RECURSIVE (level 5, one
//! component). Neither can be ported as if the other were a leaf.
//! ⛔ THE SYNC ORDER IS dxp's AND CITED: reordering syncs times the card out (CB state=TimedOut,
//! sync rc=-1).
//!
//! 9 units. Every citation resolves against the authority tree
//! `/Users/nickm/git/deeptools-src` at revision `a0d29abbed`.
//!
//! | unit | level | LoC | authority |
//! |---|---|---|---|
//! | `e011_fillUnitToIdMap` | 0 | 17 | `dcc/src/Conversion/SentientToProgIR/LowerSentientHelper.cpp:967` |
//! | `e105_LowerSyncOperation` | 3 | 37 | `dcc/src/Conversion/SentientToProgIR/LowerSentientHelper.cpp:353` |
//! | `e106_LowerNOPOperation` | 3 | 14 | `dcc/src/Conversion/SentientToProgIR/LowerSentientHelper.cpp:391` |
//! | `e111_LowerForOperation` | 3 | 66 | `dcc/src/Conversion/SentientToProgIR/LowerSentientHelper.cpp:591` |
//! | `e115_LowerReturnOperation` | 3 | 7 | `dcc/src/Conversion/SentientToProgIR/LowerSentientHelper.cpp:900` |
//! | `e116_LowerSetSendDestinationOperation` | 3 | 18 | `dcc/src/Conversion/SentientToProgIR/LowerSentientHelper.cpp:908` |
//! | `e123_LowerYieldOperation` | 4 | 117 | `dcc/src/Conversion/SentientToProgIR/LowerSentientHelper.cpp:782` |
//! | `e125_LowerUniformYieldOperation` | 4 | 61 | `dcc/src/Conversion/SentientToProgIR/LowerSentientHelper.cpp:1056` |
//! | `e126_LowerUniformOperations` | 5 | 70 | `dcc/src/Conversion/SentientToProgIR/LowerSentientHelper.cpp:985` |

// ⛔ ONE `crustify:todo:` PER SCHEDULED UNIT. Replace each with the ported function
// carrying `/// Replaces: eNNN_name`. A surviving TODO is open work.

use crate::arch::{Arch, IsaGen};
use crate::bridges::sentient_to_progir::construct::mask_and_splat::{
    LxHalf, SetDestTarget, SetDstMaskRefusal, SetDstMaskTarget, construct_set_dest_instr,
    construct_set_dst_mask_instr,
};
use crate::bridges::sentient_to_progir::construct::scalar::{
    Assign, LoopBound, Sync, construct_assign_instr, construct_mv_loop_instr, construct_nop_instr,
    construct_return_instr, construct_sync_instr,
};
use crate::bridges::sentient_to_progir::lower::labels_and_regs::{
    LabelToJumps, LoweredOp, RegImmSource, add_to_labels_map, add_to_reg_init, address_scale,
    get_reg_imm_vals, update_label_and_add_to_code_graph,
};
use crate::bridges::sentient_to_progir::state::{
    CopyOps, LabelCounter, Labels, OpSite, RegGraphs, UnitKey,
};
use crate::bridges::sentient_to_progir::uniform::block::UniformInstrBlocks;
use crate::bridges::sentient_to_progir::uniform::instr::{OperandMapRefusal, UniformInstrInfo};
use crate::formats::Bits;
use crate::islands::progir::OperandField;
use crate::islands::progir::ty::FoldId;
use crate::islands::sentient::dialects::sentient::{Reg, RegIndex};
use crate::islands::sentient::dialects::{Op, Val, dataflow, defining_op, sentient};
use sys_arch_spec::regfile::Component;

/// WHICH REGION OF A UNIFORMIZED OP A UNIT TAKES ITS INSTRUCTIONS FROM — the value of
/// `unit_to_region_idx_map_` (`UniformInstrAndBlock.hpp:223`).
///
/// ⛔ ABSENCE IS AN `Option<RegionIndex>`, NEVER `(size_t)-1`, which is what `getUnitRegionIndex`
/// answers for a unit the map does not hold. A REGULAR block always answers 0.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RegionIndex(pub u32);

/// Replaces: e011_fillUnitToIdMap
///
/// Which region each unit of a uniformized op draws its instructions from; a group contributes every
/// unit it holds, under the same region.
/// ⛔ THE REFERENCE DEREFERENCES A GROUP MEMBER'S `GetUnitOp` UNCHECKED — `getDefiningOp<GetUnitOp>()`
/// with no null test — so a group holding anything else crashes there and is skipped here.
/// ⭐ `getNumRegions`/`getRegionUnitList` ARE THE MECHANISM, not the decision: the unit lists arrive
/// already partitioned, one per region, because `list_sizes` is what partitions them.
/// ⛔ A UNIT WITH NO CORE HAS NO KEY (see [`UnitKey::of`]) and is dropped, as `getUnitName` would
/// file it under `…core-1corelet-1`.
#[must_use]
pub fn fill_unit_to_id_map(region_units: &[Vec<Val>], scope: &[Op]) -> Vec<(UnitKey, RegionIndex)> {
    let mut map = Vec::new();
    let mut index = RegionIndex(0);
    for units in region_units {
        for unit in units {
            match defining_op(*unit, scope) {
                Some(Op::Dataflow(dataflow::Op::GetUnit {
                    residency, unit, ..
                })) => set_unit_region_index(&mut map, UnitKey::of(*unit, *residency), index),
                Some(Op::Dataflow(dataflow::Op::CreateGroup { unit_ids, .. })) => {
                    for member in unit_ids {
                        if let Some(Op::Dataflow(dataflow::Op::GetUnit {
                            residency, unit, ..
                        })) = defining_op(*member, scope)
                        {
                            set_unit_region_index(&mut map, UnitKey::of(*unit, *residency), index);
                        }
                    }
                }
                _ => {}
            }
        }
        index = RegionIndex(index.0 + 1);
    }
    map
}

/// `setUnitRegionIndex` (`UniformInstrAndBlock.hpp:205-207`).
///
/// ⛔⛔ `unit_to_region_idx_map_[unit] = idx` ASSIGNS, so a unit listed in two regions keeps only the
/// LAST and this vector holds ONE entry per unit. Appending both left the readers of that map
/// disagreeing: `getUnitInstrList` would have answered the first region and `getInstr` the second.
fn set_unit_region_index(
    map: &mut Vec<(UnitKey, RegionIndex)>,
    key: Option<UnitKey>,
    index: RegionIndex,
) {
    let Some(key) = key else {
        return;
    };
    match map.iter_mut().find(|(at, _)| *at == key) {
        Some(entry) => entry.1 = index,
        None => map.push((key, index)),
    }
}

/// WHERE A LOWERED INSTRUCTION GOES — `updateLabelAndAddToCodeGraph`'s own three parameters, less
/// the instruction (`LowerSentientHelper.cpp:47-49`).
///
/// ⭐ ONE SINK RATHER THAN THREE ARGUMENTS ON EVERY `Lower*Operation`, which is the grouping this
/// crate makes instead of permitting `#[allow(clippy::too_many_arguments)]`.
pub struct CodeGraph<'a> {
    /// `labels` — which op carries which label.
    pub labels: &'a mut Labels,
    /// `uniform_instr_region` — the blocks being appended to.
    pub region: &'a mut UniformInstrBlocks,
    /// The op being lowered, whose label the first emitted instruction may take.
    pub at: OpSite,
}

/// Replaces: e106_LowerNOPOperation
///
/// The do-nothing instruction a `sentient.nop` lowers to, carrying its `dbgName` as the comment.
///
/// ⭐ `++num_nops_` IS ONE PER CALL, so a caller collecting statistics counts calls; and the
/// reference's `comp` is unused because a NOP is the same instruction on every unit.
/// ⛔ AN ABSENT `dbgName` AND AN EMPTY ONE ARE THE SAME CALL — see [`construct_nop_instr`].
pub fn lower_nop_operation(dbg_name: Option<&str>, graph: CodeGraph<'_>) {
    let no_op_instr = construct_nop_instr(dbg_name);
    update_label_and_add_to_code_graph(
        graph.labels,
        graph.region,
        graph.at,
        LoweredOp::Other,
        no_op_instr,
        false,
    );
}
/// WHAT A PROGRAM-HEADER-CARRIED VALUE INITIALISES ITS REGISTER WITH.
///
/// ⛔ `getQueryKeyAndUnitsFromParentRegion`'s FAILURE — which abandons the whole loop lowering, the
/// MVLOOP included (`:617-621`) — CANNOT HAPPEN HERE: the unit list arrives as data.
#[derive(Debug, Clone, PartialEq)]
pub struct CarriedHeader {
    /// `getRegImmVals`' source — the defining op of `getIterOperands()[i]`.
    pub imm: RegImmSource,
    /// The units `getQueryKeyAndUnitsFromParentRegion` answers.
    pub units: Vec<(UnitKey, Option<FoldId>)>,
    /// `getRegionIterArgs()[i]`'s register.
    pub reg: Reg,
}

/// ONE VALUE A `sentient.for` CARRIES, AS THIS LOWERING READS IT.
///
/// ⛔ THE REFERENCE INDEXES ITS ARRAYS TWO WAYS: `element_sizes[i + 1]` and `regLocales[i + 1]` skip
/// the loop's own `lccr`, while `programHeader[i]` does not.
/// ⛔ `element_sizes` IS WRITTEN BY A D29-D75 PASS OUTSIDE THIS CAMPAIGN, so it arrives as data.
#[derive(Debug, Clone, PartialEq)]
pub struct CarriedAssign {
    /// `ConstructAssignInstr`'s view of `iterOperands[i] -> regionIterArgs[i]`.
    pub assign: Assign,
    /// `element_sizes[i + 1]` — ⛔ `None` IS BOTH AN ABSENT ATTRIBUTE AND THE `-1` A `jcr`/`lccr`
    /// CARRIES, whose assign ignores it; the reference's `DT_CHECK_MSG` aborts on `-1` anywhere else.
    pub element_size: Option<Bits>,
    /// `programHeader[i]` and what the header needs — `None` when the flag is false or absent.
    pub header: Option<CarriedHeader>,
}

/// A `sentient.for` AS THIS LOWERING READS IT — its trip count and the values it carries.
#[derive(Debug, Clone, PartialEq)]
pub struct ForLoop {
    /// `$bound`, already resolved.
    pub bound: LoopBound,
    /// `regIndices[0]` — the loop counter's own register.
    pub lccr: RegIndex,
    /// `$dbgName`.
    pub dbg_name: Option<String>,
    /// `getRegionIterArgs()`, one entry each.
    pub carried: Vec<CarriedAssign>,
}

/// A LOWERED `sentient.for` — what it cost and what a per-unit map could not hold.
#[derive(Debug, Clone, PartialEq)]
pub struct LoweredFor {
    /// `++num_copy_ops_`, summed over the carried values.
    pub copy_ops: CopyOps,
    /// The offenders, in the order the reference reaches them.
    pub refused: Vec<OperandMapRefusal>,
}

/// Replaces: e111_LowerForOperation
///
/// One assignment per carried value, each program-header value's immediate recorded as a register
/// initialiser instead, then the `MVLOOPCNT` holding the trip count.
///
/// ⛔ ONLY THE FIRST EMITTED INSTRUCTION TAKES THE OP'S LABEL, AND THE MVLOOP IS LAST — so a loop
/// with any assignment does not label its own loop count.
/// ⛔ THE JUMP IS FILED BEFORE THE MVLOOP IS APPENDED: `AddToLabelsMap` reads the slot append fills.
#[must_use]
pub fn lower_for_operation<A: Arch>(
    comp: Component,
    for_op: &ForLoop,
    labels_ctr: &mut LabelCounter,
    reg_graph: &mut RegGraphs,
    label_to_jumps: &mut LabelToJumps,
    graph: CodeGraph<'_>,
) -> LoweredFor {
    let scale = address_scale::<A>(comp);
    let mut label_attached = false;
    let mut copy_ops = CopyOps(0);
    let mut refused = Vec::new();
    for value in &for_op.carried {
        let element_size = value.element_size.unwrap_or(Bits(8 * scale.get()));
        let assigned =
            construct_assign_instr::<A>(&value.assign, comp, element_size, value.header.is_some());
        if let Some(header) = &value.header {
            let imm_vals =
                get_reg_imm_vals::<A>(&header.imm, &header.units, comp, element_size, scale, false);
            add_to_reg_init(header.reg, &imm_vals, reg_graph, false);
        }
        copy_ops = CopyOps(copy_ops.0 + assigned.copy_ops.0);
        refused.extend(assigned.refused);
        if let Some(instr) = assigned.instr {
            update_label_and_add_to_code_graph(
                graph.labels,
                graph.region,
                graph.at,
                LoweredOp::Other,
                instr,
                label_attached,
            );
            label_attached = true;
        }
    }
    let super_instr = construct_mv_loop_instr(
        &for_op.bound,
        for_op.lccr,
        for_op.dbg_name.as_deref(),
        graph.labels,
        labels_ctr,
    );
    refused.extend(super_instr.refused);
    if super_instr.instr.has_common_field(OperandField::PcTarget) {
        add_to_labels_map(graph.region, label_to_jumps, &super_instr.instr);
    }
    update_label_and_add_to_code_graph(
        graph.labels,
        graph.region,
        graph.at,
        LoweredOp::Other,
        super_instr.instr,
        label_attached,
    );
    LoweredFor { copy_ops, refused }
}

/// Replaces: e115_LowerReturnOperation
///
/// The `RETURN` that ends a unit's program, under its op's label.
/// ⭐ THE REFERENCE'S `comp` IS UNUSED — a RETURN is the same instruction on every unit.
pub fn lower_return_operation(graph: CodeGraph<'_>) {
    update_label_and_add_to_code_graph(
        graph.labels,
        graph.region,
        graph.at,
        LoweredOp::Other,
        construct_return_instr(),
        false,
    );
}

/// WHERE A `sentient.set_send_dst` SENDS, BY THE UNIT IT SITS IN — the three arms of `:909-923`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SendDestination<'a> {
    /// `is_any_of(comp, LXLU, LXSU)` — a `SETDSTMASK` naming the consumer, via the SFP.
    LxMask {
        /// Which half the transfer leaves.
        from: LxHalf,
        /// `op.getUnits().getDefiningOp()`.
        dest: SetDstMaskTarget<'a>,
    },
    /// `is_any_of(comp, SFP)` — a `SETDEST` naming another core's SFP.
    Sfp(SetDestTarget<'a>),
    /// Any other unit — *"did not expect set_send_dst in this unit"*.
    Elsewhere(Component),
}

/// WHAT A SET-SEND-DESTINATION COULD NOT LOWER.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SendDestRefusal {
    /// The unit itself has no send-destination instruction.
    Unit(Component),
    /// The `SETDSTMASK`'s own offenders.
    Mask(SetDstMaskRefusal),
    /// The `SETDEST`'s per-unit map.
    Map(OperandMapRefusal),
}

/// Replaces: e116_LowerSetSendDestinationOperation
///
/// Where this unit's send goes: an LX half aims its transfer at the consumer, the SFP aims at another
/// core's SFP, and every other unit is the offender rather than a `signalPassFailure`.
#[must_use]
pub fn lower_set_send_destination_operation(
    dest: SendDestination<'_>,
    graph: CodeGraph<'_>,
) -> Vec<SendDestRefusal> {
    let (instr, refused): (UniformInstrInfo, Vec<SendDestRefusal>) = match dest {
        SendDestination::LxMask { from, dest } => {
            let (instr, refused) = construct_set_dst_mask_instr(from, dest);
            (
                instr,
                refused.into_iter().map(SendDestRefusal::Mask).collect(),
            )
        }
        SendDestination::Sfp(target) => {
            let (instr, refused) = construct_set_dest_instr(target);
            (
                instr,
                refused.into_iter().map(SendDestRefusal::Map).collect(),
            )
        }
        SendDestination::Elsewhere(comp) => return vec![SendDestRefusal::Unit(comp)],
    };
    update_label_and_add_to_code_graph(
        graph.labels,
        graph.region,
        graph.at,
        LoweredOp::Other,
        instr,
        false,
    );
    refused
}

/// HOW MANY NOPS A LOWERING HAD TO INSERT — `cq_stats.num_nops_` (`:376`).
///
/// ⛔ A NEWTYPE BESIDE `CopyOps`, which is the other statistic these lowerings return, and the two are
/// both counts of instructions nobody asked for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Nops(pub u32);

/// Replaces: e105_LowerSyncOperation
///
/// Lower a `sentient.sync`: the rendezvous, plus — on DD2 only — a dummy NOP when an implicit tile
/// boundary is immediately followed by a transfer.
///
/// ⛔ THE NOP IS A HARDWARE WORKAROUND, NOT A SCHEDULING CHOICE (`:370-380`), and no vendor test
/// carries one, so its three conditions are all that state it.
/// ⛔ THE REFERENCE'S COMPONENT GATE IS UNREPRESENTABLE HERE: [`Sync`]'s arms ARE the six components
/// it admits, so *"Unknown unit for sync operation"* has no input left to fire on.
/// ⚠️ AND `getNextNode()` IS DEREFERENCED UNCHECKED THERE — `None` is the last-op case.
pub fn lower_sync_operation<A: Arch>(
    sync: &Sync,
    l0_tethered: bool,
    dbg_name: Option<&str>,
    next: Option<&Op>,
    graph: CodeGraph<'_>,
) -> Nops {
    let instr = construct_sync_instr::<A>(sync, l0_tethered, dbg_name);
    update_label_and_add_to_code_graph(
        graph.labels,
        graph.region,
        graph.at,
        LoweredOp::Other,
        instr,
        false,
    );
    // `implicit_sync_boundary_tile_size` is -1 unless the op sets the boundary (`:365-368`).
    let tiled = match sync {
        Sync::L0Implicit { tile } => tile.0 > 0,
        Sync::Lx { .. } | Sync::L3lu { .. } | Sync::L3su { .. } | Sync::L0 { .. } => false,
    };
    let next_transfers = matches!(
        next,
        Some(Op::Sentient(
            sentient::Op::LoadAndSend { .. } | sentient::Op::ReceiveAndStore { .. }
        ))
    );
    if matches!(A::GEN, IsaGen::Rcudd1a) && tiled && next_transfers {
        graph
            .region
            .add_instruction_to_last_block(construct_nop_instr(Some(
                "dummy_nop_after_implicit_sync_set",
            )));
        return Nops(1);
    }
    Nops(0)
}

// crustify:todo: e123_LowerYieldOperation
// crustify:todo: e125_LowerUniformYieldOperation
// crustify:todo: e126_LowerUniformOperations

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::arch::Target;
    use crate::bridges::sentient_to_progir::construct::mask_and_splat::MaskDest;
    use crate::bridges::sentient_to_progir::construct::scalar::{AssignKind, LrfImm};
    use crate::bridges::sentient_to_progir::lower::labels_and_regs::RegImm;
    use crate::bridges::sentient_to_progir::uniform::block::UniformInstrBlock;
    use crate::bridges::sentient_to_progir::uniform::instr::Comment;
    use crate::islands::progir::OpCode;
    use crate::islands::progir::ty::{Operand, OperandValue, PerFold};
    use crate::islands::sentient::dialects::sentient::RegType;
    use crate::units::{Core, Corelet, DfirUnit, Residency};

    #[test]
    fn an_implicit_tile_boundary_before_a_transfer_gets_a_dummy_nop_on_dd2() {
        use crate::arch::{Bytes, Elements};
        use crate::arch::{Dd2, Sen1p5 as Sen1p5Arch};
        use crate::bridges::sentient_to_progir::construct::scalar::BoundaryTile;
        use crate::islands::dataflow_ir::link::{Link, Lxlu, Sfp};
        use crate::islands::progir::{OpCode, OperandField};
        use crate::islands::sentient::dialects::sentient::{Extent, Reg, RegType, ShuffleMode};

        let (consumer, _) = Link::<Lxlu, Sfp>::between(Val(0), Val(1)).ends();
        let send = Op::Sentient(sentient::Op::LoadAndSend {
            mutable_addr: Val(2),
            immutable_addr: Val(3),
            increment: Val(4),
            consumer,
            result: Val(5),
            extent: Extent::of(Elements(128), Bytes(1)),
            interleaved_group: 0,
            rotate_val: None,
            dir: None,
            shuffle_mode: ShuffleMode::NoShuffle,
            reg: Reg {
                locale: RegType::Lar,
                index: None,
            },
            dbg_name: None,
        });
        let sync = Sync::L0Implicit {
            tile: BoundaryTile(32),
        };
        let mut labels = Labels::default();
        let mut blocks = UniformInstrBlocks::default();
        let nops = lower_sync_operation::<Dd2>(
            &sync,
            false,
            Some("sync #1"),
            Some(&send),
            CodeGraph {
                labels: &mut labels,
                region: &mut blocks,
                at: OpSite(0),
            },
        );
        let instrs = |blocks: &UniformInstrBlocks| match &blocks.blocks[0] {
            UniformInstrBlock::Regular(instrs) => instrs.clone(),
            UniformInstrBlock::Uniform(_) => panic!("a regular block"),
        };
        let emitted = instrs(&blocks);
        // `implicit-sync.mlir:9` — `L0_SYNC :: implicit:yes tilesize:32  // sync #1`.
        assert_eq!(emitted[0].opcode, OpCode::SYNC);
        assert_eq!(
            emitted[0].common_field(OperandField::Tilesize),
            Some(&Operand::every(OperandValue::Int(32)))
        );
        assert!(
            !emitted[0].has_common_field(OperandField::Synctag),
            "an implicit sync writes none"
        );
        assert_eq!(emitted[1].opcode, OpCode::NOP);
        assert_eq!(nops, Nops(1));
        // ⛔ SEN1P5 DOES NOT NEED IT — same sync, same next op, one instruction.
        let mut later = UniformInstrBlocks::default();
        let nops = lower_sync_operation::<Sen1p5Arch>(
            &sync,
            false,
            Some("sync #1"),
            Some(&send),
            CodeGraph {
                labels: &mut labels,
                region: &mut later,
                at: OpSite(0),
            },
        );
        assert_eq!(instrs(&later).len(), 1);
        assert_eq!(nops, Nops(0));
    }

    #[test]
    fn a_group_files_every_unit_it_holds_under_one_region() {
        let core = Core::checked(0).expect("core 0");
        let l3su = dataflow::Op::GetUnit {
            result: Val(0),
            residency: Residency::CoreWide { core },
            unit: DfirUnit::L3su,
            num_folds: None,
        };
        let lxlu = dataflow::Op::GetUnit {
            result: Val(1),
            residency: Residency::CoreWide { core },
            unit: DfirUnit::Lxlu,
            num_folds: None,
        };
        let group = dataflow::Op::CreateGroup {
            result: Val(2),
            unit_ids: vec![Val(0), Val(1)],
        };
        let scope = vec![Op::Dataflow(l3su), Op::Dataflow(lxlu), Op::Dataflow(group)];
        // Region 0 lists the group; region 1 lists one unit directly.
        let map = fill_unit_to_id_map(&[vec![Val(2)], vec![Val(1)]], &scope);
        let key = |unit| UnitKey::of(unit, Residency::CoreWide { core }).expect("core 0 keys");
        // ⛔ `lxlu` IS IN BOTH REGIONS AND KEEPS THE LAST — `setUnitRegionIndex` assigns.
        assert_eq!(
            map,
            vec![
                (key(DfirUnit::L3su), RegionIndex(0)),
                (key(DfirUnit::Lxlu), RegionIndex(1)),
            ]
        );
    }

    /// e106 — ⭐ IBM'S OWN LINE: `SFP_NOP ::  // NOP #1` (`nop.mlir:10`).
    #[test]
    fn a_nop_carries_its_dbg_name_as_its_comment() {
        let mut labels = Labels::default();
        let mut region = UniformInstrBlocks::default();
        lower_nop_operation(
            Some("NOP #1"),
            CodeGraph {
                labels: &mut labels,
                region: &mut region,
                at: OpSite(0),
            },
        );
        let instrs = &region.blocks[0].instr_lists()[0];
        assert_eq!(instrs.len(), 1);
        assert_eq!(instrs[0].opcode, OpCode::NOP);
        assert_eq!(instrs[0].comment, Comment::Common("NOP #1".to_owned()));
        assert_eq!(instrs[0].tag, None);
    }

    /// e111 — ⭐⭐ IBM'S OWN LOOP (`empty-loop.mlir:39`): one `lrf` iter arg with
    /// `element_sizes = [-1, 16, 16]` and `programHeader = [true]` over `960`, which emits only
    /// `LX_MVLOOPCNT :: imm:9  // for-loop-imm-lccr-0` and files `LRF : 0 : 1920` (`:9`, `:21`).
    #[test]
    fn a_program_header_loop_files_its_immediate_and_emits_only_its_loop_count() {
        let unit = UnitKey {
            unit: DfirUnit::Lxlu,
            core: Core::checked(0).expect("core 0"),
            corelet: Corelet::checked(0),
        };
        let for_op = ForLoop {
            bound: LoopBound::Constant(9),
            lccr: RegIndex::at::<0>(),
            dbg_name: None,
            carried: vec![CarriedAssign {
                assign: Assign {
                    src_index: None,
                    tgt_index: Some(RegIndex::at::<0>()),
                    kind: AssignKind::LrfFromImm(LrfImm::Constant(960)),
                },
                element_size: Some(Bits(16)),
                header: Some(CarriedHeader {
                    imm: RegImmSource::Common(RegImm::Constant(960)),
                    units: vec![(unit, None)],
                    reg: Reg {
                        locale: RegType::Lrf,
                        index: Some(RegIndex::at::<0>()),
                    },
                }),
            }],
        };
        let mut labels = Labels::default();
        let mut region = UniformInstrBlocks::default();
        let mut reg_graph = RegGraphs::default();
        let mut label_to_jumps = LabelToJumps::new();
        let lowered = lower_for_operation::<Target>(
            Component::Lxlu,
            &for_op,
            &mut LabelCounter(0),
            &mut reg_graph,
            &mut label_to_jumps,
            CodeGraph {
                labels: &mut labels,
                region: &mut region,
                at: OpSite(0),
            },
        );
        assert_eq!(
            lowered,
            LoweredFor {
                copy_ops: CopyOps(0),
                refused: vec![],
            }
        );
        assert!(label_to_jumps.is_empty(), "a counted loop jumps nowhere");
        // ⛔ THE PROGRAM HEADER CARRIES THE IMMEDIATE, so no `LX_IMMCOPY` precedes the loop count.
        let instrs = &region.blocks[0].instr_lists()[0];
        assert_eq!(instrs.len(), 1);
        assert_eq!(instrs[0].opcode, OpCode::MVLOOPCNT);
        assert_eq!(
            instrs[0].common_fields,
            vec![(OperandField::Imm, Operand::every(OperandValue::Int(9)))]
        );
        assert_eq!(
            instrs[0].comment,
            Comment::Common("for-loop-imm-lccr-0".to_owned())
        );
        let inits = reg_graph.get(unit).expect("the lxlu was initialised");
        assert_eq!(inits[0].file, crate::islands::progir::ty::RegType::Lrf);
        assert_eq!(inits[0].index, RegIndex::at::<0>());
        assert_eq!(
            inits[0].value.value,
            PerFold::Every(OperandValue::Int(1920)),
            "960 at 16 bits an element is 1920 address units"
        );
    }

    /// e115 — IBM's `setmask_incrmask.mlir:22` `PTOP_RETURN ::  // end of the program`, under its label.
    #[test]
    fn a_return_ends_the_program_under_its_own_label() {
        let mut labels = Labels::default();
        labels.claim(OpSite(0), "return-0".to_owned());
        let mut region = UniformInstrBlocks::default();
        lower_return_operation(CodeGraph {
            labels: &mut labels,
            region: &mut region,
            at: OpSite(0),
        });
        let instr = &region.blocks[0].instr_lists()[0][0];
        assert_eq!(instr.opcode, OpCode::RETURN);
        assert_eq!(instr.tag.as_deref(), Some("return-0"));
        assert_eq!(
            instr.comment,
            Comment::Common("end of the program".to_owned())
        );
    }

    /// e116 — IBM's `setdstmask-lxlu.mlir:9` `LX_SETDSTMASK :: mode:pt  // set dest for transfer from
    /// lxlu to pt, via sfp`; ⛔ AND A UNIT THAT IS NEITHER AN LX HALF NOR THE SFP IS THE OFFENDER.
    #[test]
    fn an_lx_half_aims_its_transfer_and_any_other_unit_is_the_offender() {
        let mut labels = Labels::default();
        labels.claim(OpSite(0), "set-dst-0".to_owned());
        let mut region = UniformInstrBlocks::default();
        let refused = lower_set_send_destination_operation(
            SendDestination::LxMask {
                from: LxHalf::Lxlu,
                dest: SetDstMaskTarget::Unit(MaskDest::Pt),
            },
            CodeGraph {
                labels: &mut labels,
                region: &mut region,
                at: OpSite(0),
            },
        );
        assert_eq!(refused, vec![]);
        let instr = &region.blocks[0].instr_lists()[0][0];
        assert_eq!(instr.opcode, OpCode::SETDSTMASK);
        assert_eq!(instr.tag.as_deref(), Some("set-dst-0"));
        assert_eq!(
            instr.comment,
            Comment::Common("set dest for transfer from lxlu to pt, via sfp".to_owned())
        );
        assert_eq!(
            instr.common_field(OperandField::Mode),
            Some(&Operand::every(OperandValue::Descriptive("pt".to_owned())))
        );
        let mut empty = UniformInstrBlocks::default();
        assert_eq!(
            lower_set_send_destination_operation(
                SendDestination::Elsewhere(Component::Pt),
                CodeGraph {
                    labels: &mut Labels::default(),
                    region: &mut empty,
                    at: OpSite(0),
                },
            ),
            vec![SendDestRefusal::Unit(Component::Pt)]
        );
        assert!(empty.blocks.is_empty());
    }
}
