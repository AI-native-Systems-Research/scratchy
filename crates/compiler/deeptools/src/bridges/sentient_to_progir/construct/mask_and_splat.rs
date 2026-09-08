//! THE MASK, SPLAT AND SAMV INSTRUCTIONS — set-dest-mask, set-dest, splat, splat-pad, and the
//! set/incr mask pair.
//!
//! 9 units. Every citation resolves against the authority tree
//! `/Users/nickm/git/deeptools-src` at revision `a0d29abbed`.
//!
//! | unit | level | LoC | authority |
//! |---|---|---|---|
//! | `e005_ConstructIncrMaskInstr` | 0 | 11 | `dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:4118` |
//! | `e060_ConstructSetDstMaskInstr` | 1 | 41 | `dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:3559` |
//! | `e061_ConstructSetDestInstr` | 1 | 54 | `dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:3601` |
//! | `e062_ConstructImmCopyInstrFromSplatOp` | 1 | 60 | `dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:3657` |
//! | `e064_ConstructSAMVInstr` | 1 | 33 | `dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:4034` |
//! | `e065_ConstructSAMVResetInstruction` | 1 | 26 | `dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:4068` |
//! | `e066_ConstructSetMaskInstr` | 1 | 22 | `dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:4095` |
//! | `e092_ConstructSplatInstrFromSplatOp` | 2 | 65 | `dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:3719` |
//! | `e098_ConstructSplatPadInstr` | 3 | 65 | `dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:3792` |

use super::{descriptive, int};
use crate::arch::{Arch, IsaGen};
use crate::bridges::sentient_to_progir::state::UnitKey;
use crate::bridges::sentient_to_progir::uniform::instr::{
    Comment, FoldConstant, MapMode, MappedEntry, MappedOp, OperandMapRefusal, UniformInstrInfo,
    add_entry_to_operand_map,
};
use crate::formats::DataFormat;
use crate::islands::progir::ty::Operand;
use crate::islands::progir::{OpCode, OperandField};
use crate::islands::sentient::dialects::sentient::{
    RawPrecision, RegIndex, SliceId, SplatPad, ValidEntries, WslLen,
};
use crate::units::DfirUnit;
use sys_arch_spec::regfile::Component;

// ⛔ ONE `crustify:todo:` PER SCHEDULED UNIT. Replace each with the ported function
// carrying `/// Replaces: eNNN_name`. A surviving TODO is open work.

/// Replaces: e005_ConstructIncrMaskInstr
///
/// Advance the PT's mask by one.
///
/// ⛔ PT ONLY — the reference aborts on any other unit (`:4121-4122`). There is no component
/// parameter here, so a call for the SFP or a load unit cannot be written rather than being refused.
///
/// ⭐ AN UNNAMED OP GETS NO COMMENT AT ALL, not an empty one: the reference only calls
/// `setCommonComment` when `dbgName` is present, unlike [`construct_nop_instr`]'s literal fallback.
#[must_use]
pub fn construct_incr_mask_instr(dbg_name: Option<&str>) -> UniformInstrInfo {
    let instr = UniformInstrInfo::of(OpCode::INCRMASK);
    match dbg_name {
        Some(name) => instr.with_common_comment(name),
        None => instr,
    }
}

/// WHICH LX HALF A SET-DEST-MASK IS SENT FROM — `is_any_of(comp, LXLU, LXSU)` (`:3563-3565`), so a
/// call for any other component has no spelling rather than being refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LxHalf {
    /// `lxlu`.
    Lxlu,
    /// `lxsu`.
    Lxsu,
}

impl LxHalf {
    /// `senComponentsToString.at(comp)`.
    #[must_use]
    pub const fn spelling(self) -> &'static str {
        match self {
            Self::Lxlu => "lxlu",
            Self::Lxsu => "lxsu",
        }
    }
}

/// WHERE A SET-DEST-MASK MAY POINT — the three spellings its `DT_CHECK` admits (`:3571-3574`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaskDest {
    /// `sfp`.
    Sfp,
    /// `l0su`.
    L0su,
    /// `pt`.
    Pt,
}

impl MaskDest {
    /// The `mode` field's value.
    #[must_use]
    pub const fn spelling(self) -> &'static str {
        match self {
            Self::Sfp => "sfp",
            Self::L0su => "l0su",
            Self::Pt => "pt",
        }
    }

    /// The destination a unit is, and `None` where the reference's `DT_CHECK` aborts.
    ///
    /// ⛔ `pt` IS A SUBSTRING TEST, NOT A UNIT (`:3566-3567`): every `ptrow<N>` answers `pt`, and so
    /// does `crossptnlink`, whose spelling contains it too.
    #[must_use]
    pub fn of_unit(unit: DfirUnit) -> Option<MaskDest> {
        if unit.spelling().contains("pt") {
            return Some(Self::Pt);
        }
        match unit {
            DfirUnit::Sfp => Some(Self::Sfp),
            DfirUnit::L0su => Some(Self::L0su),
            _ => None,
        }
    }
}

/// WHAT NAMES A SET-DEST-MASK'S DESTINATION — the reference's `get_unit` / `query_map` pair
/// (`:3565,3577`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SetDstMaskTarget<'a> {
    /// `dataflow.get_unit` — one destination for every unit.
    Unit(MaskDest),
    /// `uniform.query_map` — one destination per key unit.
    Mapped(&'a [MappedEntry]),
}

/// WHAT A SET-DEST-MASK COULD NOT BE GIVEN.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SetDstMaskRefusal {
    /// The mapping itself — [`add_entry_to_operand_map`]'s own offenders.
    Mapped(OperandMapRefusal),
    /// *"unexpected destination unit"* (`:3574`), keyed by the unit whose destination it is.
    UnexpectedDestination(UnitKey),
}

/// Replaces: e060_ConstructSetDstMaskInstr
///
/// Point an LX half's transfer at the unit that consumes it, via the SFP.
///
/// ⛔ THE MAPPED PATH REWRITES THE MAP IT JUST BUILT (`:3577-3588`): mode `none` writes each value
/// unit's own spelling, and the loop replaces it with the collapsed [`MaskDest`] — so a unit whose
/// destination is unusable keeps no value at all here, where the reference aborts.
#[must_use]
pub fn construct_set_dst_mask_instr(
    from: LxHalf,
    target: SetDstMaskTarget<'_>,
) -> (UniformInstrInfo, Vec<SetDstMaskRefusal>) {
    let mut instr = UniformInstrInfo::of(OpCode::SETDSTMASK);
    let comment = |dest: MaskDest| {
        format!(
            "set dest for transfer from {} to {}, via sfp",
            from.spelling(),
            dest.spelling()
        )
    };
    match target {
        SetDstMaskTarget::Unit(dest) => {
            instr.set_common_field(OperandField::Mode, descriptive(dest.spelling()));
            (instr.with_common_comment(&comment(dest)), Vec::new())
        }
        SetDstMaskTarget::Mapped(entries) => {
            let mapped = add_entry_to_operand_map(entries, MapMode::None, 1.0);
            let mut refused: Vec<SetDstMaskRefusal> = mapped
                .refused
                .iter()
                .copied()
                .map(SetDstMaskRefusal::Mapped)
                .collect();
            let mut comments = Vec::new();
            if let Some(mut per_unit) = mapped.value {
                for (unit, operand) in
                    core::iter::once(&mut per_unit.first).chain(per_unit.rest.iter_mut())
                {
                    let dest = entries
                        .iter()
                        .rev()
                        .find(|entry| entry.key == *unit)
                        .and_then(|entry| match entry.value {
                            MappedOp::Unit(value) => MaskDest::of_unit(value.unit),
                            _ => None,
                        });
                    match dest {
                        Some(dest) => {
                            *operand = descriptive(dest.spelling());
                            comments.push((*unit, comment(dest)));
                        }
                        None => refused.push(SetDstMaskRefusal::UnexpectedDestination(*unit)),
                    }
                }
                instr.operand_map.push((OperandField::Mode, per_unit));
            }
            if !comments.is_empty() {
                instr.comment = Comment::PerUnit(comments);
            }
            (instr, refused)
        }
    }
}

/// WHAT NAMES A SET-DEST'S TARGET — the reference's `get_unit` / `query_map` pair (`:3609,3685`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SetDestTarget<'a> {
    /// `dataflow.get_unit` — this unit, and the one it sends to.
    Unit {
        /// The program unit the `set_send_dst` sits in.
        from: UnitKey,
        /// The unit named by the `get_unit`.
        to: UnitKey,
    },
    /// `uniform.query_map` — one target per key unit, off the SFP ring.
    Mapped(&'a [MappedEntry]),
}

/// Replaces: e061_ConstructSetDestInstr
///
/// Aim the SFP's send at another core's SFP — a one-hot core mask.
///
/// ⛔ THE `get_unit` PATH IS ONE ENTRY OF THE MAPPED ONE (`:3612-3629` against `cpp:170-176`): the
/// same SFP-to-SFP and same-corelet checks, the same `1 << core`, so both go through
/// [`add_entry_to_operand_map`] and only the destination of the value differs.
/// ⛔ SFP ONLY (`:3606-3608`), and the trailing `llvm_unreachable` is unreachable by the type.
#[must_use]
pub fn construct_set_dest_instr(
    target: SetDestTarget<'_>,
) -> (UniformInstrInfo, Vec<OperandMapRefusal>) {
    let mut instr = UniformInstrInfo::of(OpCode::SETDEST);
    match target {
        SetDestTarget::Unit { from, to } => {
            let entry = MappedEntry {
                key: from,
                fold: None,
                value: MappedOp::Unit(to),
            };
            let mapped = add_entry_to_operand_map(&[entry], MapMode::SetDestTgt, 1.0);
            if let Some(per_unit) = &mapped.value {
                instr.set_common_field(OperandField::Imm, per_unit.first.1.clone());
            }
            let instr = instr.with_common_comment(&format!(
                "set dest for transfer from SFP of core {} to SFP of core {}",
                from.core.get(),
                to.core.get()
            ));
            (instr, mapped.refused)
        }
        SetDestTarget::Mapped(entries) => {
            let mapped = add_entry_to_operand_map(entries, MapMode::SetDestTgt, 1.0);
            let mut instr = instr.with_common_comment("set dest for transfer through sfpring");
            if let Some(per_unit) = mapped.value {
                instr.operand_map.push((OperandField::Imm, per_unit));
            }
            (instr, mapped.refused)
        }
    }
}

/// THE ONE VALUE AN IMMCOPY SPLATS — the reference's `constant` / `query_map` pair (`:3667-3669`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImmCopyValue<'a> {
    /// `sentient.constant` — one immediate for every unit.
    Constant(i64),
    /// `uniform.query_map` with constant inputs — one immediate per unit and fold.
    Mapped(&'a [FoldConstant]),
}

/// WHICH PRECISION AN IMMCOPY'S `mode` BIT SPELLS — *"expected fp16 or fp32 precision as per the
/// ISA"* (`:3684-3686`), so the other twelve have no spelling here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImmCopyPrecision {
    /// `fp16`.
    Fp16,
    /// `fp32`.
    Fp32,
}

impl ImmCopyPrecision {
    /// `stringifySentientPrecision`.
    #[must_use]
    pub const fn spelling(self) -> &'static str {
        match self {
            Self::Fp16 => "fp16",
            Self::Fp32 => "fp32",
        }
    }
}

/// THE `sentient.splat` AN IMMCOPY IS BUILT FROM — every attribute it reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImmCopySplat<'a> {
    /// `op.getPad()` — `replica` is 1 for `none` and 0 for `left`.
    pub pad: SplatPad,
    /// `op.getPrecision()`.
    pub precision: ImmCopyPrecision,
    /// `op.getMask()`'s constant — the `mask` field takes `255 -` it.
    pub mask: i64,
    /// `op.getDbgName()`.
    pub dbg_name: Option<&'a str>,
}

/// WHAT AN IMMCOPY COULD NOT BE GIVEN.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImmCopyRefusal {
    /// *"fp32 precision only valid in SFP in DD2"* (`:3688-3690`).
    Fp32OnPe,
    /// One unit's folds carry different constants — the disagreement
    /// [`UniformInstrInfo::add_const_entries_to_operand_map`] hands back.
    FoldingNeeded(UnitKey),
}

/// Replaces: e062_ConstructImmCopyInstrFromSplatOp
///
/// Turn a splat of a constant into an IMMCOPY: the immediate in the op's format, the mask inverted
/// out of 255, and the target register.
///
/// ⛔ `unrollIncrResult` IS OMITTED, NOT DEFAULTED (`:3697-3699`): the reference refuses an unrolled
/// target here, so no field carries it.
/// ⛔ AND A DD2 PE SETS NO `mode` AT ALL — that branch is the fp32 refusal, not a mode of its own.
#[must_use]
pub fn construct_imm_copy_instr_from_splat_op<A: Arch>(
    mut instr: UniformInstrInfo,
    input: ImmCopyValue<'_>,
    comp: Component,
    op_precision: DataFormat,
    reg_val: RegIndex,
    splat: ImmCopySplat<'_>,
    copy_ops: Option<&mut u32>,
) -> (UniformInstrInfo, Vec<ImmCopyRefusal>) {
    let mut refused = Vec::new();
    match input {
        ImmCopyValue::Constant(value) => {
            instr.set_common_field(OperandField::Imm, int(value).in_format(op_precision));
        }
        ImmCopyValue::Mapped(entries) => {
            // `setOperandMap` ASSIGNS the whole map (`hpp:133`), so nothing built before survives.
            instr.operand_map.clear();
            refused.extend(
                instr
                    .add_const_entries_to_operand_map(
                        OperandField::Imm,
                        entries,
                        1.0,
                        Some(op_precision),
                    )
                    .into_iter()
                    .map(ImmCopyRefusal::FoldingNeeded),
            );
        }
    }
    let dd2_pe = matches!(A::GEN, IsaGen::Rcudd1a) && matches!(comp, Component::Pe);
    if !dd2_pe {
        instr.set_common_field(OperandField::Mode, descriptive(splat.precision.spelling()));
    } else if matches!(splat.precision, ImmCopyPrecision::Fp32) {
        refused.push(ImmCopyRefusal::Fp32OnPe);
    }
    instr.opcode = OpCode::IMMCOPY;
    if let Some(copy_ops) = copy_ops {
        *copy_ops += 1;
    }
    instr.set_common_field(
        OperandField::Tgtrf,
        descriptive(&format!("R{}", reg_val.get())),
    );
    instr.set_common_field(OperandField::Mask, int(255 - splat.mask));
    let replica = i64::from(matches!(splat.pad, SplatPad::None));
    instr.set_common_field(OperandField::Replica, int(replica));
    let desc = "splat/pad to create vector";
    let instr = match splat.dbg_name {
        Some(name) => instr.with_common_comment(&format!("{name} {desc}")),
        None => instr.with_common_comment(desc),
    };
    (instr, refused)
}

/// THE `sentient.set_active_mask_value` A SAMV IS BUILT FROM — every attribute it reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActiveMaskValue {
    /// `op.getMaskall()`.
    pub maskall: bool,
    /// `op.getSliceidXsl()`.
    pub slice_id_xsl: SliceId,
    /// `op.getNumvalidentry()`.
    pub valid_entries: ValidEntries,
    /// `getValueRegIndex(op.getMaskValue())` — the `MVR<n>` the mask lives in.
    pub mask_value: RegIndex,
    /// `op.getPrecision()`, the raw ISA field.
    pub precision: RawPrecision,
    /// `op.getXslinner()`.
    pub xslinner: bool,
    /// `op.getWsllen()`.
    pub wsl_len: WslLen,
}

/// Replaces: e064_ConstructSAMVInstr
///
/// Set the LX load unit's active mask from the op's own seven fields.
///
/// ⛔ LXLU ONLY, SO THERE IS NO COMPONENT PARAMETER (`:4036-4037`) — the same shape as
/// [`construct_samv_reset_instruction`], which is this with all seven fields fixed.
#[must_use]
pub fn construct_samv_instr(samv: ActiveMaskValue, dbg_name: Option<&str>) -> UniformInstrInfo {
    let instr = UniformInstrInfo::of(OpCode::SAMV);
    let mut instr = match dbg_name {
        Some(name) => instr.with_common_comment(name),
        None => instr,
    };
    instr.set_common_field(OperandField::Maskall, yes_no(samv.maskall));
    instr.set_common_field(
        OperandField::Sliceidxsl,
        int(i64::from(samv.slice_id_xsl.0)),
    );
    instr.set_common_field(
        OperandField::Numvalidentry,
        int(i64::from(samv.valid_entries.0)),
    );
    instr.set_common_field(
        OperandField::Mvridx,
        descriptive(&format!("MVR{}", samv.mask_value.get())),
    );
    instr.set_common_field(
        OperandField::Precision,
        descriptive(&format!("{}b", samv.precision.0)),
    );
    instr.set_common_field(OperandField::Xslinner, yes_no(samv.xslinner));
    instr.set_common_field(
        OperandField::Wsllen,
        descriptive(&format!("{}b", samv.wsl_len.0)),
    );
    instr
}

/// A BOOLEAN ISA FIELD — the `? "yes" : "no"` behind every flag this file sets.
fn yes_no(flag: bool) -> Operand {
    descriptive(if flag { "yes" } else { "no" })
}

/// Replaces: e065_ConstructSAMVResetInstruction
///
/// Put the mask register back: one slice, one bit wide, no valid entries.
/// ⛔ LXLU ONLY, SO THERE IS NO COMPONENT PARAMETER (`:4071-4072`).
/// ⛔ AND THE PROGRAM'S OWN SAMV IS AN INPUT, NOT A FLAG: the mask register and the precision are the
/// two fields this reset does not hardcode, so they arrive as values and
/// `DT_CHECK_MSG(has_samv_, "No SAMV was detected in the program")` has nothing left to refuse.
#[must_use]
pub fn construct_samv_reset_instruction(
    mask_value: RegIndex,
    precision: RawPrecision,
) -> UniformInstrInfo {
    let mut instr = UniformInstrInfo::of(OpCode::SAMV);
    instr.set_common_field(OperandField::Maskall, descriptive("no"));
    instr.set_common_field(OperandField::Sliceidxsl, int(7));
    instr.set_common_field(OperandField::Numvalidentry, int(0));
    instr.set_common_field(
        OperandField::Mvridx,
        descriptive(&format!("MVR{}", mask_value.get())),
    );
    instr.set_common_field(
        OperandField::Precision,
        descriptive(&format!("{}b", precision.0)),
    );
    instr.set_common_field(OperandField::Xslinner, descriptive("no"));
    instr.set_common_field(OperandField::Wsllen, descriptive("1b"));
    instr
}

/// Replaces: e066_ConstructSetMaskInstr
///
/// Load the PT's mask register with a constant.
/// ⛔ PT ONLY, SO THERE IS NO COMPONENT PARAMETER (`:4097-4098`).
/// ⛔ AND THE MASK VALUE IS A CONSTANT BY THE TYPE: `llvm_unreachable("expecting ConstantOp from
/// mask_value operand")` (`:4111`) is unreachable when a non-constant cannot be written down.
#[must_use]
pub fn construct_set_mask_instr(mask_value: i64, dbg_name: Option<&str>) -> UniformInstrInfo {
    let instr = UniformInstrInfo::of(OpCode::SETMASK);
    let mut instr = match dbg_name {
        Some(name) => instr.with_common_comment(name),
        None => instr,
    };
    instr.set_common_field(OperandField::Imm, int(mask_value));
    instr
}

// crustify:todo: e092_ConstructSplatInstrFromSplatOp
// crustify:todo: e098_ConstructSplatPadInstr

#[cfg(test)]
mod unit_tests {
    use super::{
        ActiveMaskValue, Component, DfirUnit, ImmCopyPrecision, ImmCopySplat, ImmCopyValue, LxHalf,
        MaskDest, OperandField, RawPrecision, RegIndex, SetDestTarget, SetDstMaskTarget, SliceId,
        SplatPad, UnitKey, ValidEntries, WslLen, construct_imm_copy_instr_from_splat_op,
        construct_incr_mask_instr, construct_samv_instr, construct_samv_reset_instruction,
        construct_set_dest_instr, construct_set_dst_mask_instr, construct_set_mask_instr,
        descriptive, int,
    };
    use crate::arch::{Dd2, Sen1p5};
    use crate::bridges::sentient_to_progir::uniform::instr::{Comment, UniformInstrInfo};
    use crate::formats::DataFormat;
    use crate::islands::progir::OpCode;
    use crate::units::{Core, Corelet, Row};

    fn sfp(core: u32) -> UnitKey {
        UnitKey {
            unit: DfirUnit::Sfp,
            core: Core::checked(core).expect("every arch has 32 cores"),
            corelet: Corelet::checked(0),
        }
    }

    /// IBM'S OWN `setmask_incrmask.mlir:13` — `PTOP_INCRMASK ::  // incr_mask #1`: the opcode, the
    /// dbgName as the comment, and NO operand fields.
    #[test]
    fn incr_mask_carries_only_the_dbg_name() {
        let named = construct_incr_mask_instr(Some("incr_mask #1"));
        assert_eq!(named.opcode, OpCode::INCRMASK);
        assert_eq!(named.comment, Comment::Common("incr_mask #1".to_owned()));
        assert!(named.common_fields.is_empty());
        assert_eq!(construct_incr_mask_instr(None).comment, Comment::None);
    }

    /// IBM'S OWN `samv.mlir:11` — `LX_SAMV :: maskall:no mvridx:MVR0 numvalidentry:0 precision:8b
    /// sliceidxsl:7 wsllen:1b xslinner:no  // samv #4`, which is the reset's whole field set.
    #[test]
    fn the_samv_reset_hardcodes_every_field_but_the_mask_register_and_the_precision() {
        let reset = construct_samv_reset_instruction(RegIndex::at::<0>(), RawPrecision(8));
        assert_eq!(reset.opcode, OpCode::SAMV);
        assert_eq!(
            reset.common_fields,
            vec![
                (OperandField::Maskall, descriptive("no")),
                (OperandField::Mvridx, descriptive("MVR0")),
                (OperandField::Numvalidentry, int(0)),
                (OperandField::Precision, descriptive("8b")),
                (OperandField::Sliceidxsl, int(7)),
                (OperandField::Wsllen, descriptive("1b")),
                (OperandField::Xslinner, descriptive("no")),
            ]
        );
        assert_eq!(reset.comment, Comment::None);
    }

    /// IBM'S OWN `setmask_incrmask.mlir:11,17` — `PTOP_SETMASK :: imm:0  // set_mask #1` and
    /// `imm:2  // set_mask #2`.
    #[test]
    fn set_mask_carries_the_constant_as_its_immediate() {
        let first = construct_set_mask_instr(0, Some("set_mask #1"));
        assert_eq!(first.opcode, OpCode::SETMASK);
        assert_eq!(first.common_fields, vec![(OperandField::Imm, int(0))]);
        assert_eq!(first.comment, Comment::Common("set_mask #1".to_owned()));
        let second = construct_set_mask_instr(2, None);
        assert_eq!(second.common_fields, vec![(OperandField::Imm, int(2))]);
        assert_eq!(second.comment, Comment::None);
    }

    /// IBM'S OWN `setdstmask.mlir` — `LX_SETDSTMASK :: mode:pt  // set dest for transfer from lxlu
    /// to pt, via sfp`, and the `l0su` line beside it.
    #[test]
    fn set_dst_mask_names_the_destination_and_repeats_it_in_the_comment() {
        let (pt, refused) =
            construct_set_dst_mask_instr(LxHalf::Lxlu, SetDstMaskTarget::Unit(MaskDest::Pt));
        assert_eq!(pt.opcode, OpCode::SETDSTMASK);
        assert_eq!(
            pt.common_fields,
            vec![(OperandField::Mode, descriptive("pt"))]
        );
        assert_eq!(
            pt.comment,
            Comment::Common("set dest for transfer from lxlu to pt, via sfp".to_owned())
        );
        assert!(refused.is_empty());
        let (l0su, _) =
            construct_set_dst_mask_instr(LxHalf::Lxsu, SetDstMaskTarget::Unit(MaskDest::L0su));
        assert_eq!(
            l0su.comment,
            Comment::Common("set dest for transfer from lxsu to l0su, via sfp".to_owned())
        );
        let row = DfirUnit::PtRow(Row::checked(0).expect("every arch has row 0"));
        assert_eq!(
            MaskDest::of_unit(row),
            Some(MaskDest::Pt),
            "a substring test"
        );
        assert_eq!(MaskDest::of_unit(DfirUnit::Pe), None);
    }

    /// IBM'S OWN `setdest.mlir` — `SFP_SETDEST :: imm:2  // set dest for transfer from SFP of core 0
    /// to SFP of core 1`, and the core 30 → 31 line, whose mask is the top bit.
    #[test]
    fn set_dest_masks_the_target_core_one_hot() {
        let (near, refused) = construct_set_dest_instr(SetDestTarget::Unit {
            from: sfp(0),
            to: sfp(1),
        });
        assert_eq!(near.opcode, OpCode::SETDEST);
        assert_eq!(near.common_fields, vec![(OperandField::Imm, int(2))]);
        assert_eq!(
            near.comment,
            Comment::Common("set dest for transfer from SFP of core 0 to SFP of core 1".to_owned())
        );
        assert!(refused.is_empty() && near.operand_map.is_empty());
        let (far, _) = construct_set_dest_instr(SetDestTarget::Unit {
            from: sfp(30),
            to: sfp(31),
        });
        assert_eq!(
            far.common_fields,
            vec![(OperandField::Imm, int(2_147_483_648))]
        );
    }

    /// IBM'S OWN `splat.mlir` — `PE_IMMCOPY :: imm:12121 mask:255 replica:1 tgtrf:R0  // splat/pad
    /// to create vector` (a DD2 PE, so NO `mode`) and `SFP_IMMCOPY :: imm:25256 mask:255 mode:fp32
    /// replica:0 tgtrf:R0  // splat #2 splat/pad to create vector`.
    #[test]
    fn imm_copy_carries_the_constant_the_inverted_mask_and_the_target_register() {
        let mut copy_ops = 0;
        let (pe, refused) = construct_imm_copy_instr_from_splat_op::<Dd2>(
            UniformInstrInfo::of(OpCode::NOP),
            ImmCopyValue::Constant(12121),
            Component::Pe,
            DataFormat::Sen169Fp16,
            RegIndex::at::<0>(),
            ImmCopySplat {
                pad: SplatPad::None,
                precision: ImmCopyPrecision::Fp16,
                mask: 0,
                dbg_name: None,
            },
            Some(&mut copy_ops),
        );
        assert_eq!(pe.opcode, OpCode::IMMCOPY);
        assert_eq!(
            pe.common_fields,
            vec![
                (
                    OperandField::Imm,
                    int(12121).in_format(DataFormat::Sen169Fp16)
                ),
                (OperandField::Mask, int(255)),
                (OperandField::Replica, int(1)),
                (OperandField::Tgtrf, descriptive("R0")),
            ],
            "a DD2 PE sets no mode bit"
        );
        assert_eq!(
            pe.comment,
            Comment::Common("splat/pad to create vector".to_owned())
        );
        assert!(refused.is_empty() && copy_ops == 1);

        let (sfp_splat, refused) = construct_imm_copy_instr_from_splat_op::<Sen1p5>(
            UniformInstrInfo::of(OpCode::NOP),
            ImmCopyValue::Constant(25256),
            Component::Sfp,
            DataFormat::IeeeFp32,
            RegIndex::at::<0>(),
            ImmCopySplat {
                pad: SplatPad::Left,
                precision: ImmCopyPrecision::Fp32,
                mask: 0,
                dbg_name: Some("splat #2"),
            },
            None,
        );
        assert_eq!(
            sfp_splat.common_fields,
            vec![
                (
                    OperandField::Imm,
                    int(25256).in_format(DataFormat::IeeeFp32)
                ),
                (OperandField::Mask, int(255)),
                (OperandField::Mode, descriptive("fp32")),
                (OperandField::Replica, int(0)),
                (OperandField::Tgtrf, descriptive("R0")),
            ]
        );
        assert_eq!(
            sfp_splat.comment,
            Comment::Common("splat #2 splat/pad to create vector".to_owned())
        );
        assert!(refused.is_empty());
    }

    /// IBM'S OWN `samv.mlir` — `LX_SAMV :: maskall:no mvridx:MVR0 numvalidentry:12 precision:8b
    /// sliceidxsl:5 wsllen:8b xslinner:yes  // samv #1`.
    #[test]
    fn samv_carries_all_seven_of_the_ops_own_fields() {
        let samv = construct_samv_instr(
            ActiveMaskValue {
                maskall: false,
                slice_id_xsl: SliceId(5),
                valid_entries: ValidEntries(12),
                mask_value: RegIndex::at::<0>(),
                precision: RawPrecision(8),
                xslinner: true,
                wsl_len: WslLen(8),
            },
            Some("samv #1"),
        );
        assert_eq!(samv.opcode, OpCode::SAMV);
        assert_eq!(
            samv.common_fields,
            vec![
                (OperandField::Maskall, descriptive("no")),
                (OperandField::Mvridx, descriptive("MVR0")),
                (OperandField::Numvalidentry, int(12)),
                (OperandField::Precision, descriptive("8b")),
                (OperandField::Sliceidxsl, int(5)),
                (OperandField::Wsllen, descriptive("8b")),
                (OperandField::Xslinner, descriptive("yes")),
            ]
        );
        assert_eq!(samv.comment, Comment::Common("samv #1".to_owned()));
    }
}
