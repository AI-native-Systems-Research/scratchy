//! THE LABELS, THE CODE GRAPH AND THE REGISTER IMMEDIATES — AddToLabelsMap,
//! updateLabelAndAddToCodeGraph, addToRegInit, GetAddressScale, GetOpCodePrefix, getRegImmVals.
//!
//! 6 units. Every citation resolves against the authority tree
//! `/Users/nickm/git/deeptools-src` at revision `a0d29abbed`.
//!
//! | unit | level | LoC | authority |
//! |---|---|---|---|
//! | `e008_addToRegInit` | 0 | 41 | `dcc/src/Conversion/SentientToProgIR/LowerSentientHelper.cpp:96` |
//! | `e009_GetAddressScale` | 0 | 16 | `dcc/src/Conversion/SentientToProgIR/LowerSentientHelper.cpp:502` |
//! | `e010_GetOpCodePrefix` | 0 | 18 | `dcc/src/Conversion/SentientToProgIR/LowerSentientHelper.cpp:519` |
//! | `e068_getRegImmVals` | 1 | 102 | `dcc/src/Conversion/SentientToProgIR/LowerSentientHelper.cpp:1140` |
//! | `e093_AddToLabelsMap` | 2 | 8 | `dcc/src/Conversion/SentientToProgIR/LowerSentientHelper.cpp:32` |
//! | `e094_updateLabelAndAddToCodeGraph` | 2 | 40 | `dcc/src/Conversion/SentientToProgIR/LowerSentientHelper.cpp:47` |

// ⛔ ONE `crustify:todo:` PER SCHEDULED UNIT. Replace each with the ported function
// carrying `/// Replaces: eNNN_name`. A surviving TODO is open work.

use crate::arch::Arch;
use crate::bridges::sentient_to_progir::state::{Labels, OpSite, RegGraphs, UnitKey};
use crate::bridges::sentient_to_progir::uniform::block::{InstrIndex, UniformInstrBlocks};
use crate::bridges::sentient_to_progir::uniform::instr::UniformInstrInfo;
use crate::bridges::sentient_to_progir::utils::{AddrSpace, addr_wraparounded};
use crate::formats::Bits;
use crate::islands::progir::ty::{FoldId, Operand, OperandValue, reg_file_of};
use crate::islands::progir::{OpCode, OperandField, RegInit};
use crate::islands::sentient::dialects::sentient::Reg;
use crate::units::Core;
use std::collections::{BTreeMap, BTreeSet};
use sys_arch_spec::regfile::Component;
use sys_arch_spec::values::OpUnit;

/// HOW MUCH ONE ADDRESS UNIT IS WORTH — an entry of `addressGranularityScalePerUnit`
/// (`sysdef.cpp:531-554`), which `getAddressGranularityScale` looks up by `{component, storage}`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct AddressScale(u32);

impl AddressScale {
    /// The multiplier, for the one place an address is divided by it.
    #[must_use]
    pub const fn get(self) -> u32 {
        self.0
    }

    /// `is_multicast ? 1 : GetAddressScale(comp, op)` (`LowerSentientHelper.cpp:415`) — the one scale
    /// nothing looks up, because a multicast id is not an address.
    #[must_use]
    pub const fn unscaled() -> AddressScale {
        AddressScale(1)
    }
}

/// Replaces: e009_GetAddressScale
///
/// The granularity an address operand of `comp` counts in.
/// ⛔ THE L0 STORE UNIT SCALES BY `numPTRows` AND THE LOAD UNIT BY 1 (`sysdef.cpp:542-543`) — the one
/// asymmetry in the table, and an arch fact rather than a constant.
/// ⛔ EVERY OTHER COMPONENT ANSWERS 1 THROUGH HERE. The PT/PE/SFP LRF-family entries are 128, but the
/// reference's fallthrough `return 1` reaches them first, so this function never reads them.
/// ⛔⛔ AND THE `locale` PARAMETER IS DEAD, so it is gone: it only ever chooses between `{L3*, HBM}`
/// and `{L3*, LX}`, which are the same 128 — see the body.
#[must_use]
pub fn address_scale<A: Arch>(comp: Component) -> AddressScale {
    match comp {
        Component::L0lu => AddressScale(1),
        Component::L0su => AddressScale(A::PT_ROWS),
        Component::Lxlu | Component::Lxsu => AddressScale(1),
        // ⭐ BOTH DEFINED KEYS ANSWER 128 AND SO THE LOCALE CANNOT BE READ OFF THE ANSWER:
        // `{L3*, HBM}` (ear/ebr/jcr) and `{L3*, LX}` (lar/lbr) carry the same scale
        // (`sysdef.cpp:532-536`). ⛔ THAT IS ALSO WHY THE REFERENCE'S `llvm_unreachable` FOR ANY
        // OTHER LOCALE IS GONE RATHER THAN TRANSLATED: there is no third value it could have given.
        Component::L3lu | Component::L3su => AddressScale(128),
        Component::Pt | Component::Pe | Component::Sfp => AddressScale(1),
    }
}

/// Replaces: e010_GetOpCodePrefix
///
/// Which unit family names a component's opcodes — the `PTOP`/`SFP`/`PE`/`L0`/`LX`/`L3` prefix the
/// reference writes into an opcode string, as the vendored enum that prefix selected.
/// ⛔ THE UNRECOGNISED-UNIT BRANCH IS UNREPRESENTABLE, and it was worse than a refusal: it printed to
/// `llvm::errs` and left `op_code` UNTOUCHED, so the caller went on with whatever string it held.
/// ⭐ AND THE MAPPING IS THE ONE ALREADY VENDORED — a second copy is where the L0/LX split drifts.
#[must_use]
pub fn op_code_prefix(comp: Component) -> OpUnit {
    OpUnit::of_component(comp)
}

/// Replaces: e008_addToRegInit
///
/// Record one SSA value's immediates as a register's initial contents on every unit.
///
/// ⭐ ONE OPERAND WHEN EVERY FOLD AGREES, per-fold otherwise (`:111-131`) — `all_same` collapses the
/// fold map so an unfolded program carries no fold ids at all.
///
/// ⚠️ A LOCALE WITH NO REGISTER FILE, OR NO INDEX, WRITES NOTHING, where the reference has no guard
/// and would turn `-1` into a huge `unsigned`; its own sibling `addToRegsToInit` reads `-1` as
/// nothing to track.
pub fn add_to_reg_init(
    reg: Reg,
    imm_vals: &[(UnitKey, Vec<(Option<FoldId>, i64)>)],
    reg_graph: &mut RegGraphs,
    is_symbolic: bool,
) {
    let (Some(index), Some(file)) = (reg.index, reg_file_of(reg.locale)) else {
        return;
    };
    for (unit, folds) in imm_vals {
        let Some((_, front)) = folds.first() else {
            continue;
        };
        let held = |value: i64| {
            if is_symbolic {
                OperandValue::VariableSymbol(value)
            } else {
                OperandValue::Int(value)
            }
        };
        let mut value = Operand::default();
        if folds.iter().all(|(_, held)| held == front) {
            value.set(None, held(*front));
        } else {
            for (fold, imm) in folds {
                value.set(*fold, held(*imm));
            }
        }
        reg_graph.add_reg_init(*unit, RegInit { file, index, value });
    }
}

/// ONE IMMEDIATE A REGISTER CAN START WITH — the four defining ops `getRegImmVals` reads a value out
/// of (`:1144,1154,1162,1171`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegImm {
    /// `sentient.constant` — ⛔ THE ONLY ONE THAT IS SCALED OR WRAPPED.
    Constant(i64),
    /// `symbol.create_symbol`'s `symbolID`.
    Symbol(i64),
    /// `dataflow.create_multicast_group`, ALREADY ENCODED: `encodeMulticastGroupInfo` is outside this
    /// campaign's 130.
    Multicast(i64),
    /// `dataflow.get_unit` — the core it names.
    Unit(Core),
}

/// WHERE A REGISTER'S IMMEDIATES COME FROM — one value for every unit, or one per unit.
#[derive(Debug, Clone, PartialEq)]
pub enum RegImmSource {
    /// One defining op, whose value every unit takes (`:1144-1178`).
    Common(RegImm),
    /// `uniform.query_map` — `None` is a unit the mapping does not cover, which the reference gets
    /// from `getValuesFromKeys` positionally.
    /// ⚠️ THE KIND IS PER ENTRY HERE and read off the query op once there (`:1181-1186`), so a mixed
    /// mapping is representable rather than being a null cast.
    Mapped(Vec<(UnitKey, Option<FoldId>, Option<RegImm>)>),
}

/// `imm_vals[unit][fold] = value` — ⭐ A PLAIN ASSIGNMENT into the two-level map.
fn set_imm(
    imm_vals: &mut Vec<(UnitKey, Vec<(Option<FoldId>, i64)>)>,
    unit: UnitKey,
    fold: Option<FoldId>,
    value: i64,
) {
    let folds = match imm_vals.iter_mut().find(|(at, _)| *at == unit) {
        Some(entry) => &mut entry.1,
        None => {
            imm_vals.push((unit, Vec::new()));
            &mut imm_vals.last_mut().expect("just pushed").1
        }
    };
    match folds.iter_mut().find(|(at, _)| *at == fold) {
        Some(entry) => entry.1 = value,
        None => folds.push((fold, value)),
    }
}

/// Replaces: e068_getRegImmVals
///
/// Every unit's immediate for one register, per fold — exactly what [`add_to_reg_init`] then files.
/// ⛔ ONLY A CONSTANT IS SCALED, AND AN MVR ONE IS NOT (`:1146`): a mask value is not an address.
/// ⚠️ THE MAPPED ARM SCALES ANYWAY, `isMVRReg` and all (`:1205`), and in integers where
/// [`crate::bridges::sentient_to_progir::construct::reg_init::fill_imm_field`] uses floats.
/// ⛔ A UNIT THE MAPPING MISSES TAKES THE FIRST COVERED UNIT'S WHOLE FOLD MAP — the reference's own
/// *"use a random value for this unit"* (`:1230-1236`); with none covered it takes nothing, where its
/// `DT_CHECK(!imm_vals.empty())` aborts.
#[must_use]
pub fn get_reg_imm_vals<A: Arch>(
    source: &RegImmSource,
    units: &[(UnitKey, Option<FoldId>)],
    comp: Component,
    element_size: Bits,
    scale: AddressScale,
    is_mvr_reg: bool,
) -> Vec<(UnitKey, Vec<(Option<FoldId>, i64)>)> {
    let space = AddrSpace::Unit(comp);
    let scaled = |value: i64| value * i64::from(element_size.0) / 8 / i64::from(scale.get());
    let mut imm_vals: Vec<(UnitKey, Vec<(Option<FoldId>, i64)>)> = Vec::new();
    match source {
        RegImmSource::Common(imm) => {
            let value = match *imm {
                RegImm::Constant(value) if is_mvr_reg => addr_wraparounded::<A>(value, space),
                RegImm::Constant(value) => addr_wraparounded::<A>(scaled(value), space),
                RegImm::Symbol(id) | RegImm::Multicast(id) => id,
                RegImm::Unit(core) => i64::from(core.get()),
            };
            for (unit, fold) in units {
                set_imm(&mut imm_vals, *unit, *fold, value);
            }
        }
        RegImmSource::Mapped(entries) => {
            let mut empty_units: Vec<UnitKey> = Vec::new();
            for (unit, fold, held) in entries {
                let Some(imm) = *held else {
                    empty_units.push(*unit);
                    continue;
                };
                let value = match imm {
                    RegImm::Constant(value) => addr_wraparounded::<A>(scaled(value), space),
                    RegImm::Symbol(id) | RegImm::Multicast(id) => id,
                    RegImm::Unit(core) => i64::from(core.get()),
                };
                set_imm(&mut imm_vals, *unit, *fold, value);
            }
            if let Some((_, donor)) = imm_vals.first().cloned() {
                for unit in empty_units {
                    match imm_vals.iter_mut().find(|(at, _)| *at == unit) {
                        Some(entry) => entry.1 = donor.clone(),
                        None => imm_vals.push((unit, donor.clone())),
                    }
                }
            }
        }
    }
    imm_vals
}

/// WHICH INSTRUCTIONS JUMP TO EACH LABEL — the reference's
/// `std::map<std::string, std::set<InstrIndex>>` (`LowerSentientHelper.cpp:34`), read back once every
/// label's own instruction is known so each jump can be repointed at it.
pub type LabelToJumps = BTreeMap<String, BTreeSet<InstrIndex>>;

/// Replaces: e093_AddToLabelsMap
///
/// Records that the instruction about to be appended jumps to its `pc_target` label.
///
/// ⛔ NOTHING IS RECORDED WITHOUT A LABEL OR A BLOCK, where the reference's `asString` aborts on a
/// jump carrying no `pc_target` and `getNextInstrIndex` reads `blocks_.back()` with no block at all.
pub fn add_to_labels_map(
    uniform_instr_region: &UniformInstrBlocks,
    label_to_jumps: &mut LabelToJumps,
    super_instr: &UniformInstrInfo,
) {
    let Some(label) = super_instr
        .common_field(OperandField::PcTarget)
        .and_then(|target| target.as_string(None))
    else {
        return;
    };
    let Some(index) = uniform_instr_region.next_instr_index() else {
        return;
    };
    label_to_jumps
        .entry(label.to_owned())
        .or_default()
        .insert(index);
}

/// WHETHER THE OP BEING LOWERED IS A `sentient.yield` — `isa<sentient::YieldOp>(op)`
/// (`LowerSentientHelper.cpp:61`), half of the MVLOOP tail test.
///
/// ⛔ A TYPE RATHER THAN A SECOND `bool` beside `force_not_add_label`, which is one call away from
/// silently swapping the two.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoweredOp {
    /// `sentient.yield`.
    Yield,
    /// Any other op.
    Other,
}

/// Replaces: e094_updateLabelAndAddToCodeGraph
///
/// Appends one instruction carrying the label its op claimed — except that an UNTAGGED NOP lowered
/// from a `sentient.yield` merges into the instruction before it as that one's `be` (block end)
/// instead, which is where an MVLOOP's tail goes.
///
/// ⛔ A TAG BLOCKS THE MERGE, and so does a predecessor that already has `be` or is the MVLOOPCNT
/// holding the loop count. ⛔ AND THE MERGE READS THE LAST BLOCK'S **LAST** REGION, not its current.
pub fn update_label_and_add_to_code_graph(
    labels: &Labels,
    uniform_instr_region: &mut UniformInstrBlocks,
    at: OpSite,
    op: LoweredOp,
    mut super_instr: UniformInstrInfo,
    force_not_add_label: bool,
) {
    let tag = if force_not_add_label {
        None
    } else {
        labels.get(at).map(str::to_owned)
    };
    // Both conditions valid means it is coming from MVLOOP; a tag present means don't merge.
    let is_instr_nop = super_instr.opcode == OpCode::NOP && op == LoweredOp::Yield;
    if tag.is_none() && is_instr_nop {
        let last_instr = uniform_instr_region
            .blocks
            .last_mut()
            .and_then(|block| block.instr_lists_mut().last_mut())
            .and_then(|region| region.last_mut());
        if let Some(last_instr) = last_instr
            && !last_instr.has_common_field(OperandField::Be)
            && last_instr.opcode != OpCode::MVLOOPCNT
        {
            last_instr.set_common_field(
                OperandField::Be,
                Operand::every(OperandValue::Descriptive("be".to_owned())),
            );
            return;
        }
    }
    super_instr.tag = tag;
    uniform_instr_region.add_instruction_to_last_block(super_instr);
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::arch::Target;
    use crate::bridges::sentient_to_progir::lower::control::RegionIndex;
    use crate::bridges::sentient_to_progir::uniform::block::UniformInstrBlock;
    use crate::islands::progir::ty::PerFold;
    use crate::islands::sentient::dialects::sentient::RegIndex;
    use crate::islands::sentient::dialects::sentient::RegType;
    use crate::units::{Core, Corelet, DfirUnit};

    #[test]
    fn the_l0_store_unit_scales_by_the_pt_row_count() {
        assert_eq!(
            address_scale::<Target>(Component::L0su).get(),
            Target::PT_ROWS,
            "{{L0SU, L0}} is numPTRows (sysdef.cpp:543)"
        );
        assert_eq!(address_scale::<Target>(Component::L0lu).get(), 1);
        assert_eq!(address_scale::<Target>(Component::L3su).get(), 128);
        assert_eq!(address_scale::<Target>(Component::L3lu).get(), 128);
        assert_eq!(address_scale::<Target>(Component::Lxsu).get(), 1);
        // The fallthrough: a component the lookup is never reached for.
        assert_eq!(address_scale::<Target>(Component::Pt).get(), 1);
    }

    #[test]
    fn the_nine_components_name_six_opcode_families() {
        assert_eq!(op_code_prefix(Component::Pt), OpUnit::Ptop);
        assert_eq!(op_code_prefix(Component::Sfp), OpUnit::Sfp);
        assert_eq!(op_code_prefix(Component::Pe), OpUnit::Pe);
        assert_eq!(op_code_prefix(Component::L0lu), OpUnit::L0);
        assert_eq!(op_code_prefix(Component::L0su), OpUnit::L0);
        assert_eq!(op_code_prefix(Component::Lxlu), OpUnit::Lx);
        assert_eq!(op_code_prefix(Component::Lxsu), OpUnit::Lx);
        assert_eq!(op_code_prefix(Component::L3lu), OpUnit::L3);
        assert_eq!(op_code_prefix(Component::L3su), OpUnit::L3);
    }

    fn pe(core: u32) -> UnitKey {
        UnitKey {
            unit: DfirUnit::Pe,
            core: Core::checked(core).expect("every arch has core 0"),
            corelet: Corelet::checked(0),
        }
    }

    fn lxlu(core: u32) -> UnitKey {
        UnitKey {
            unit: DfirUnit::Lxlu,
            core: Core::checked(core).expect("every arch has 32 cores"),
            corelet: Corelet::checked(0),
        }
    }

    /// Agreeing folds collapse to one operand; disagreeing ones stay per fold.
    #[test]
    fn agreeing_folds_collapse_to_a_single_operand() {
        let reg = Reg {
            locale: RegType::Lar,
            index: Some(RegIndex::at::<2>()),
        };
        let mut reg_graph = RegGraphs::default();
        add_to_reg_init(
            reg,
            &[
                (pe(0), vec![(Some(FoldId(0)), 7), (Some(FoldId(1)), 7)]),
                (pe(1), vec![(Some(FoldId(0)), 7), (Some(FoldId(1)), 9)]),
            ],
            &mut reg_graph,
            false,
        );
        let zero = reg_graph.get(pe(0)).expect("core 0 was initialised");
        assert_eq!(zero[0].file, crate::islands::progir::ty::RegType::Lar);
        assert_eq!(zero[0].index, RegIndex::at::<2>());
        assert_eq!(zero[0].value.value, PerFold::Every(OperandValue::Int(7)));
        let one = reg_graph.get(pe(1)).expect("core 1 was initialised");
        assert_eq!(
            one[0].value.value,
            PerFold::ByFold(vec![
                (FoldId(0), OperandValue::Int(7)),
                (FoldId(1), OperandValue::Int(9)),
            ])
        );
    }

    /// IBM'S OWN `uniformization_small_elem_size.mlir` — `element_size = 4` on an LXLU, where the
    /// constants 215552 and 215296 print as `lrfimm:107776` and `lrfimm:107648`, one per mapped core.
    /// A unit the mapping misses takes the first covered unit's map.
    #[test]
    fn only_constants_are_scaled_and_an_uncovered_unit_borrows_the_first_map() {
        let three = lxlu(3);
        let nineteen = lxlu(19);
        let missed = lxlu(0);
        let scale = address_scale::<Target>(Component::Lxlu);
        let mapped = get_reg_imm_vals::<Target>(
            &RegImmSource::Mapped(vec![
                (three, None, Some(RegImm::Constant(215_552))),
                (nineteen, None, Some(RegImm::Constant(215_296))),
                (missed, None, None),
            ]),
            &[],
            Component::Lxlu,
            Bits(4),
            scale,
            false,
        );
        assert_eq!(
            mapped,
            vec![
                (three, vec![(None, 107_776)]),
                (nineteen, vec![(None, 107_648)]),
                (missed, vec![(None, 107_776)]),
            ]
        );
        // The common arm, over the unit list — and an MVR register keeps its value unscaled.
        let units = [(three, Some(FoldId(0))), (nineteen, Some(FoldId(0)))];
        assert_eq!(
            get_reg_imm_vals::<Target>(
                &RegImmSource::Common(RegImm::Constant(215_296)),
                &units,
                Component::Lxlu,
                Bits(4),
                scale,
                false,
            ),
            vec![
                (three, vec![(Some(FoldId(0)), 107_648)]),
                (nineteen, vec![(Some(FoldId(0)), 107_648)]),
            ]
        );
        assert_eq!(
            get_reg_imm_vals::<Target>(
                &RegImmSource::Common(RegImm::Constant(215_296)),
                &units[..1],
                Component::Lxlu,
                Bits(4),
                scale,
                true,
            ),
            vec![(three, vec![(Some(FoldId(0)), 215_296)])],
            "a mask value is not an address"
        );
        // A symbol id and a core id are neither scaled nor wrapped, on the L0 store unit where both
        // would otherwise change.
        for (imm, expected) in [
            (RegImm::Symbol(4096), 4096),
            (RegImm::Unit(Core::checked(9).expect("core 9")), 9),
        ] {
            assert_eq!(
                get_reg_imm_vals::<Target>(
                    &RegImmSource::Common(imm),
                    &units[..1],
                    Component::L0su,
                    Bits(8),
                    address_scale::<Target>(Component::L0su),
                    false,
                ),
                vec![(three, vec![(Some(FoldId(0)), expected)])]
            );
        }
    }
    /// e093: each jump is filed under the label it targets, at the index it will land on — and a jump
    /// with no `pc_target` files nothing.
    #[test]
    fn every_jump_is_filed_under_the_label_it_targets() {
        let mut blocks = UniformInstrBlocks::default();
        let mut jump = UniformInstrInfo::of(OpCode::JCMP);
        jump.set_common_field(
            OperandField::PcTarget,
            Operand::every(OperandValue::InstrTag("tgt_0".to_owned())),
        );
        let mut label_to_jumps = LabelToJumps::new();
        add_to_labels_map(&blocks, &mut label_to_jumps, &jump);
        assert!(label_to_jumps.is_empty(), "no block, no index to file");
        blocks.add_instruction_to_last_block(UniformInstrInfo::of(OpCode::NOP));
        add_to_labels_map(&blocks, &mut label_to_jumps, &jump);
        blocks.add_instruction_to_last_block(jump.clone());
        add_to_labels_map(&blocks, &mut label_to_jumps, &jump);
        let at = |instr| InstrIndex {
            block: 0,
            region: RegionIndex(0),
            instr,
        };
        assert_eq!(
            label_to_jumps
                .get("tgt_0")
                .map(|jumps| jumps.iter().copied().collect::<Vec<_>>()),
            Some(vec![at(1), at(2)])
        );
        add_to_labels_map(
            &blocks,
            &mut label_to_jumps,
            &UniformInstrInfo::of(OpCode::JCMP),
        );
        assert_eq!(
            label_to_jumps.len(),
            1,
            "a jump with no target files nothing"
        );
    }

    /// e094: an untagged NOP from a yield becomes the previous instruction's `be`; a tagged one, or
    /// one whose predecessor already ends its block, is appended instead.
    #[test]
    fn an_untagged_yield_nop_becomes_the_previous_instructions_block_end() {
        let mut labels = Labels::default();
        labels.claim(OpSite(1), "tgt_0".to_owned());
        let nop = UniformInstrInfo::of(OpCode::NOP);
        let mut blocks = UniformInstrBlocks::default();
        blocks.add_instruction_to_last_block(UniformInstrInfo::of(OpCode::MVLOOPCNT));
        blocks.add_instruction_to_last_block(UniformInstrInfo::of(OpCode::LOGICAL));

        // The MVLOOPCNT refuses the merge, so the NOP is appended after it.
        let mut only_loop_count = blocks.clone();
        only_loop_count.blocks[0] =
            UniformInstrBlock::Regular(vec![UniformInstrInfo::of(OpCode::MVLOOPCNT)]);
        update_label_and_add_to_code_graph(
            &labels,
            &mut only_loop_count,
            OpSite(0),
            LoweredOp::Yield,
            nop.clone(),
            false,
        );
        assert_eq!(only_loop_count.blocks[0].instr_lists()[0].len(), 2);

        // The ADD takes it as `be`, and nothing is appended.
        update_label_and_add_to_code_graph(
            &labels,
            &mut blocks,
            OpSite(0),
            LoweredOp::Yield,
            nop.clone(),
            false,
        );
        let merged = &blocks.blocks[0].instr_lists()[0];
        assert_eq!(merged.len(), 2);
        assert_eq!(
            merged[1].common_field(OperandField::Be),
            Some(&Operand::every(OperandValue::Descriptive("be".to_owned())))
        );

        // A second yield NOP cannot merge into an instruction that already has `be`.
        update_label_and_add_to_code_graph(
            &labels,
            &mut blocks,
            OpSite(0),
            LoweredOp::Yield,
            nop.clone(),
            false,
        );
        assert_eq!(blocks.blocks[0].instr_lists()[0].len(), 3);

        // A labelled op is appended with its tag, and `force_not_add_label` drops that tag.
        update_label_and_add_to_code_graph(
            &labels,
            &mut blocks,
            OpSite(1),
            LoweredOp::Other,
            nop.clone(),
            false,
        );
        update_label_and_add_to_code_graph(
            &labels,
            &mut blocks,
            OpSite(1),
            LoweredOp::Other,
            nop,
            true,
        );
        let tags: Vec<Option<&str>> = blocks.blocks[0].instr_lists()[0]
            .iter()
            .map(|instr| instr.tag.as_deref())
            .collect();
        assert_eq!(
            tags,
            vec![None, None, None, Some("tgt_0"), None],
            "only the labelled op that did not force the drop keeps a tag"
        );
    }
}
