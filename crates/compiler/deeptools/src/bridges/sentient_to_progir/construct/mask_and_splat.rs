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

use super::{boolean, descriptive, int};
use crate::arch::{Arch, IsaGen};
use crate::bridges::sentient_to_progir::construct::compute::{
    ComputeComp, ComputePrecisions, ComputeSlot, UnsupportedComputeInput, set_compute_input_operand,
};
use crate::bridges::sentient_to_progir::construct::reg_init::{
    ConstInput, RegInitFormat, RegInitSplat, UniformConst, VectorImm,
    add_pe_sfp_lrf_immcopy_to_reg_init,
};
use crate::bridges::sentient_to_progir::state::{RegGraphs, UnitKey};
use crate::bridges::sentient_to_progir::uniform::instr::{
    Comment, FoldConstant, MapMode, MappedEntry, MappedOp, OperandMapRefusal, UniformInstrInfo,
    add_entry_to_operand_map,
};
use crate::bridges::sentient_to_progir::utils::ComputeUnit;
use crate::formats::DataFormat;
use crate::islands::progir::ty::Operand;
use crate::islands::progir::{OpCode, OperandField};
use crate::islands::sentient::dialects::sentient::{
    LrfIndex, Port, Precision, RawPrecision, RegIndex, SliceId, SplatPad, UnrollFactor,
    ValidEntries, WslLen,
};
use crate::units::DfirUnit;
use sys_arch_spec::regfile::Component;

// ⛔ ONE `crustify:todo:` PER SCHEDULED UNIT. Replace each with the ported function
// carrying `/// Replaces: eNNN_name`. A surviving TODO is open work.

/// Replaces: e005_ConstructIncrMaskInstr
///
/// Advance the PT's mask by one.
///
/// ⛔ PT ONLY — the reference aborts on any other unit (`:4120-4121`). There is no component
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

/// WHERE A SET-DEST-MASK MAY POINT — the three spellings its `DT_CHECK` admits (`:3570-3572`).
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
    /// ⛔ `pt` IS A SUBSTRING TEST, NOT A UNIT (`:3569`): every `ptrow<N>` answers `pt`, and so
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
/// (`:3567,3578`).
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
    /// *"unexpected destination unit"* (`:3572,3591`), keyed by the unit whose destination it is.
    UnexpectedDestination(UnitKey),
}

/// Replaces: e060_ConstructSetDstMaskInstr
///
/// Point an LX half's transfer at the unit that consumes it, via the SFP.
///
/// ⛔ THE MAPPED PATH REWRITES THE MAP IT JUST BUILT (`:3582-3596`): mode `none` writes each value
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

/// WHAT NAMES A SET-DEST'S TARGET — the reference's `get_unit` / `query_map` pair (`:3609,3643`).
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
/// ⛔ THE `get_unit` PATH IS ONE ENTRY OF THE MAPPED ONE (`:3612-3635` against `cpp:143-151`): the
/// same SFP-to-SFP and same-corelet checks, the same `1 << core`, so both go through
/// [`add_entry_to_operand_map`] and only the destination of the value differs.
/// ⛔ SFP ONLY (`:3605-3607`), and the trailing `llvm_unreachable` is unreachable by the type.
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

/// THE ONE VALUE AN IMMCOPY SPLATS — the reference's `constant` / `query_map` pair (`:3661-3662`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImmCopyValue<'a> {
    /// `sentient.constant` — one immediate for every unit.
    Constant(i64),
    /// `uniform.query_map` with constant inputs — one immediate per unit and fold.
    Mapped(&'a [FoldConstant]),
}

/// WHICH PRECISION AN IMMCOPY'S `mode` BIT SPELLS — *"expected fp16 or fp32 precision as per the
/// ISA"* (`:3680-3682`), so the other twelve have no spelling here.
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
    /// *"fp32 precision only valid in SFP in DD2"* (`:3684-3686`).
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
/// ⛔ `unrollIncrResult` IS OMITTED, NOT DEFAULTED (`:3693-3695`): the reference refuses an unrolled
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
/// ⛔ PT ONLY, SO THERE IS NO COMPONENT PARAMETER (`:4098-4099`).
/// ⛔ AND THE MASK VALUE IS A CONSTANT BY THE TYPE: `llvm_unreachable("expecting ConstantOp from
/// mask_value operand")` (`:4109`) is unreachable when a non-constant cannot be written down.
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

/// WHICH `tgt<port>` FIELD A NON-LRF SPLAT TARGET NAMES.
///
/// ⛔ THE REFERENCE CONCATENATES `"tgt" + <port>` (`:3742`), so only the ports whose own spelling
/// completes a field name are reachable: a `north` output would name `tgtnorth`, which nothing has.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SplatPort {
    /// `tgtl0`.
    L0,
    /// `tgtlx`.
    Lx,
    /// `tgtpe`.
    Pe,
    /// `tgtpt`.
    Pt,
    /// `tgtsfp`.
    Sfp,
}

impl SplatPort {
    /// The field this port's name completes.
    #[must_use]
    pub const fn field(self) -> OperandField {
        match self {
            Self::L0 => OperandField::Tgtl0,
            Self::Lx => OperandField::Tgtlx,
            Self::Pe => OperandField::Tgtpe,
            Self::Pt => OperandField::Tgtpt,
            Self::Sfp => OperandField::Tgtsfp,
        }
    }
}

/// WHERE A SPLAT'S RESULT GOES — `output.contains("lrf")` (`:3734`) as the two shapes it is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SplatTarget {
    /// An `lrf<n>` port: `tgtrf` holding `R<n>`. ⛔ THE BOUND IS THE TYPE'S — *"invalid register
    /// number"* (`:3739-3740`) cannot be written down.
    Lrf(LrfIndex),
    /// Any other port: `tgt<port>` holding `result`.
    Port(SplatPort),
}

/// THE `sentient.splat` ATTRIBUTES THIS INSTRUCTION READS.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Splat {
    /// `$pad`.
    pub pad: SplatPad,
    /// The value of the constant `$mask` is defined by.
    pub mask: i64,
    /// `$unrollFactor`.
    pub unroll: UnrollFactor,
    /// `$unrollIncrResult`.
    pub unroll_incr_result: bool,
}

/// WHAT A SPLAT CANNOT BE — the one refusal of `:3753-3754` plus its input operand's own.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SplatRefusal {
    /// *"Expecting fp16 or fp32 precision for splat"* — on a unit that carries a `mode` at all.
    Precision(Precision),
    /// What [`set_compute_input_operand`] refused for `src0`.
    Input(UnsupportedComputeInput),
}

/// Replaces: e092_ConstructSplatInstrFromSplatOp
///
/// Splat one input across the vector: the mask inverted out of 255, the pad as the immediate, the
/// unroll factor, and `result` in the field the output port names.
///
/// ⛔ `mode` ONLY WHERE THE UNIT HAS ONE — the SFP always and the PE from SEN1P5 (`:3752`) — and
/// only fp16 or fp32 there. ⛔ AND `src2` IS THE STRING `"0.0"`, DESCRIPTIVE, not a float immediate.
#[must_use]
pub fn construct_splat_instr_from_splat_op<A: Arch>(
    mut instr: UniformInstrInfo,
    input: Port,
    target: SplatTarget,
    comp: ComputeComp,
    splat: Splat,
    precision: Precision,
) -> (UniformInstrInfo, Vec<SplatRefusal>) {
    let mut refused = Vec::new();
    let (field_out, operand_out) = match target {
        SplatTarget::Lrf(index) => (OperandField::Tgtrf, format!("R{}", index.get())),
        SplatTarget::Port(port) => (port.field(), "result".to_owned()),
    };
    let padding = i64::from(matches!(splat.pad, SplatPad::None));
    instr.opcode = OpCode::SPLAT;
    if matches!(comp, ComputeComp::Sfp)
        || (matches!(comp, ComputeComp::Pe) && matches!(A::GEN, IsaGen::Sen1p5))
    {
        if matches!(precision, Precision::Fp16 | Precision::Fp32) {
            instr.set_common_field(OperandField::Mode, descriptive(precision.spelling()));
        } else {
            refused.push(SplatRefusal::Precision(precision));
        }
    }
    instr.set_common_field(OperandField::Mask, int(255 - splat.mask));
    let precisions = ComputePrecisions {
        op: precision,
        compute: precision,
        result: precision,
    };
    refused.extend(
        set_compute_input_operand::<A>(
            comp,
            &mut instr,
            ComputeSlot::Src0,
            input,
            precisions,
            false,
        )
        .map(SplatRefusal::Input),
    );
    instr.set_common_field(OperandField::Imm, int(padding));
    instr.set_common_field(OperandField::Src2, descriptive("0.0"));
    instr.set_common_field(field_out, descriptive(&operand_out));
    instr.set_common_field(OperandField::Unroll, descriptive(splat.unroll.spelling()));
    instr.set_common_field(OperandField::Unrlfldtgt, boolean(splat.unroll_incr_result));
    (instr, refused)
}
/// THE CONSTANT A SPLAT-PAD FILLS A REGISTER WITH — three of the reference's four probes on the
/// input (`:3606-3611`).
///
/// ⛔ THE QUERY-MAP ARM CARRIES BOTH READINGS OF ONE OP, and that is not redundancy: the register
/// initialiser walks the mapping's own value ops (`getValuesFromKeys`) while the IMMCOPY takes
/// `getConstantTargetKeyValues`, and the two arms below hand the same op to those two helpers.
/// Deriving either from the other would be an invention.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SplatPadConst<'a> {
    /// `sentient.constant`.
    Scalar {
        /// `getValue()`.
        value: i64,
        /// Whether the value is a symbol id rather than a number.
        is_symbol: bool,
    },
    /// `sentient.vector_constant` — ⛔ NO IMMCOPY EXISTS FOR IT (`:3849`).
    Vector(VectorImm),
    /// `uniform.query_map` over constants.
    QueryMap {
        /// What [`add_pe_sfp_lrf_immcopy_to_reg_init`] reads.
        init: &'a [UniformConst],
        /// What [`construct_imm_copy_instr_from_splat_op`] reads.
        folds: &'a [FoldConstant],
    },
}

impl<'a> SplatPadConst<'a> {
    /// The same constant as the register initialiser takes it.
    fn reg_init(self) -> ConstInput {
        match self {
            Self::Scalar { value, is_symbol } => ConstInput::Scalar { value, is_symbol },
            Self::Vector(imm) => ConstInput::Vector(imm),
            Self::QueryMap { init, .. } => ConstInput::Uniform(init.to_vec()),
        }
    }

    /// The same constant as an IMMCOPY takes it — ⛔ `None` IS *"PE/SFP IMMCOPY doesn't support
    /// vector of constants"* (`:3849`).
    const fn imm_copy(self) -> Option<ImmCopyValue<'a>> {
        match self {
            Self::Scalar { value, .. } => Some(ImmCopyValue::Constant(value)),
            Self::QueryMap { folds, .. } => Some(ImmCopyValue::Mapped(folds)),
            Self::Vector(_) => None,
        }
    }
}

/// WHAT A `sentient.splat`'s INPUT IS DEFINED BY — ⛔ TWO ARMS, WHICH IS THE
/// `llvm_unreachable("Input to Splat Op is either a Constant or a LogicalPort Op")` (`:3860-3861`)
/// made unrepresentable.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SplatPadInput<'a> {
    /// `sentient.logical_port` — the SPLAT route.
    Port(Port),
    /// A constant, a vector of constants or a query map — the IMMCOPY or register-initialiser route.
    Const(SplatPadConst<'a>),
}

/// THE `sentient.splat` ATTRIBUTES THE OUTER LOWERING READS, on top of what its two routes read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SplatPadOp<'a> {
    /// `$pad`, `$mask`, `$unrollFactor`, `$unrollIncrResult`.
    pub splat: Splat,
    /// `$precision`.
    pub precision: Precision,
    /// `$programHeader`.
    pub program_header: bool,
    /// `$dbgName`.
    pub dbg_name: Option<&'a str>,
}

/// WHAT A SPLAT-PAD COULD NOT BE.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SplatPadRefusal {
    /// *"only LRF is currently supported"* (`:3828-3830`) — a constant aimed at a port.
    ConstantIntoPort(SplatPort),
    /// *"PE/SFP IMMCOPY doesn't support vector of constants."* (`:3849`).
    VectorOfConstants,
    /// *"No masking is supported in register initialization"* — what [`RegInitSplat::of`] refuses.
    MaskedRegInit,
    /// A precision `getDataFormatFromSentientPrecisionAttr` has no row for — its own
    /// `llvm_unreachable` (`Dialect/Sentient/Utils.cpp:191-192`), int16 among them.
    Precision(Precision),
    /// What the IMMCOPY route refused.
    ImmCopy(ImmCopyRefusal),
    /// What the SPLAT route refused.
    Splat(SplatRefusal),
}

/// `getDataFormatFromSentientPrecisionAttr` (`Dialect/Sentient/Utils.cpp:171-193`) — ⛔ SEVEN ROWS,
/// AND `int16` IS COMMENTED OUT THERE (*"no hw support"*), so `None` is that function's own
/// `llvm_unreachable` rather than a case this port dropped.
const fn splat_pad_format(precision: Precision) -> Option<RegInitFormat> {
    match precision {
        Precision::Int2 => Some(RegInitFormat::Senint2),
        Precision::Int4 => Some(RegInitFormat::Senint4),
        Precision::Int8 => Some(RegInitFormat::Senint8),
        Precision::Int32 => Some(RegInitFormat::IeeeInt32),
        Precision::Fp8 => Some(RegInitFormat::Sen143Fp8),
        Precision::Fp16 => Some(RegInitFormat::Sen169Fp16),
        Precision::Fp32 => Some(RegInitFormat::IeeeFp32),
        _ => None,
    }
}

/// Replaces: e098_ConstructSplatPadInstr
///
/// Route one `sentient.splat`: a port input SPLATs, a constant IMMCOPYs into an LRF, and a program
/// header's unpadded constant becomes that register's initial content and NO instruction.
///
/// ⛔ THE PROGRAM-HEADER TEST COMES BEFORE THE CONST-VS-VECTOR SPLIT (`:3841-3850`), so a vector of
/// constants still reaches the register initialiser even though it has no IMMCOPY.
/// ⛔ `None` IS THE REFERENCE'S OWN `std::nullopt` on that path — nothing was emitted, nothing failed.
/// ⭐ THE LRF DEPTH IS THE TYPE'S: `reg_val < maxNum` (`:3837-3840`) is [`LrfIndex`]'s 32 cases.
#[must_use]
pub fn construct_splat_pad_instr<A: Arch>(
    comp: ComputeUnit,
    input: SplatPadInput<'_>,
    output: SplatTarget,
    op: &SplatPadOp<'_>,
    units: &[UnitKey],
    reg_graph: &mut RegGraphs,
    copy_ops: Option<&mut u32>,
) -> (Option<UniformInstrInfo>, Vec<SplatPadRefusal>) {
    // `:3818-3822` — the PE/SFP `DT_CHECK` is this parameter's two cases.
    let (component, compute) = match comp {
        ComputeUnit::Pe => (Component::Pe, ComputeComp::Pe),
        ComputeUnit::Sfp => (Component::Sfp, ComputeComp::Sfp),
    };
    let mut refused = Vec::new();
    // `:3824-3825` — the comment is set before either route; the IMMCOPY route overwrites it.
    let opened = |opcode| match op.dbg_name {
        Some(name) => UniformInstrInfo::of(opcode).with_common_comment(name),
        None => UniformInstrInfo::of(opcode),
    };
    // `:3798-3800` — ⛔ THE DATA FORMAT IS DERIVED BEFORE EITHER ROUTE IS CHOSEN, so an unsupported
    // precision is the table's own `llvm_unreachable` on the SPLAT route too, not only the constant's.
    let Some(format) = splat_pad_format(op.precision) else {
        refused.push(SplatPadRefusal::Precision(op.precision));
        return (None, refused);
    };
    let constant = match input {
        SplatPadInput::Port(port) => {
            let (instr, splat_refused) = construct_splat_instr_from_splat_op::<A>(
                opened(OpCode::SPLAT),
                port,
                output,
                compute,
                op.splat,
                op.precision,
            );
            refused.extend(splat_refused.into_iter().map(SplatPadRefusal::Splat));
            return (Some(instr), refused);
        }
        SplatPadInput::Const(constant) => constant,
    };
    let reg_val = match output {
        SplatTarget::Lrf(index) => index.reg_index(),
        SplatTarget::Port(port) => {
            refused.push(SplatPadRefusal::ConstantIntoPort(port));
            return (None, refused);
        }
    };
    if op.program_header && matches!(op.splat.pad, SplatPad::None) {
        let Some(witness) = RegInitSplat::of(Some(op.splat.mask), op.splat.unroll_incr_result)
        else {
            refused.push(SplatPadRefusal::MaskedRegInit);
            return (None, refused);
        };
        add_pe_sfp_lrf_immcopy_to_reg_init(
            units,
            &constant.reg_init(),
            format,
            reg_val,
            witness,
            reg_graph,
        );
        return (None, refused);
    }
    let Some(value) = constant.imm_copy() else {
        refused.push(SplatPadRefusal::VectorOfConstants);
        return (None, refused);
    };
    // `:3680-3682` — the IMMCOPY's `mode` has only these two spellings, so the other five formats
    // this table holds are reachable only through the register initialiser above.
    let mode = match op.precision {
        Precision::Fp16 => ImmCopyPrecision::Fp16,
        Precision::Fp32 => ImmCopyPrecision::Fp32,
        other => {
            refused.push(SplatPadRefusal::Precision(other));
            return (None, refused);
        }
    };
    let (instr, imm_refused) = construct_imm_copy_instr_from_splat_op::<A>(
        opened(OpCode::IMMCOPY),
        value,
        component,
        format.format(),
        reg_val,
        ImmCopySplat {
            pad: op.splat.pad,
            precision: mode,
            mask: op.splat.mask,
            dbg_name: op.dbg_name,
        },
        copy_ops,
    );
    refused.extend(imm_refused.into_iter().map(SplatPadRefusal::ImmCopy));
    (Some(instr), refused)
}

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
    use super::{
        ComputeComp, LrfIndex, Port, Precision, Splat, SplatRefusal, SplatTarget, UnrollFactor,
        boolean, construct_splat_instr_from_splat_op,
    };
    use super::{
        ComputeUnit, RegGraphs, SplatPadConst, SplatPadInput, SplatPadOp, SplatPadRefusal,
        construct_splat_pad_instr,
    };
    use crate::arch::{Dd2, Sen1p5};
    use crate::bridges::sentient_to_progir::uniform::instr::{Comment, UniformInstrInfo};
    use crate::formats::DataFormat;
    use crate::islands::progir::OpCode;
    use crate::islands::progir::RegInit;
    use crate::islands::progir::ty::{Operand, OperandValue, RegType};
    use crate::units::{Core, Corelet, Row};

    fn sfp(core: u32) -> UnitKey {
        UnitKey {
            unit: DfirUnit::Sfp,
            core: Core::checked(core).expect("every arch has 32 cores"),
            corelet: Corelet::checked(0),
        }
    }

    /// IBM'S OWN `setmask_incrmask.mlir:13` — `PTOP_INCRMASK :: be:be  // incr_mask #1`: the opcode,
    /// the dbgName as the comment, and NO operand field of its own — `be` is the block-end marker a
    /// later pass adds (`LowerSentientHelper.cpp:75-78`).
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

    /// IBM'S OWN `setmask_incrmask.mlir:9,17` — `PTOP_SETMASK :: imm:0  // set_mask #1` and
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

    /// IBM'S OWN `uniform-nop-incorrect-label.mlir:9,15` — `LX_SETDSTMASK :: mode:pt  // set dest
    /// for transfer from lxlu to pt, via sfp`, and the `l0su` line beside it.
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

    /// IBM'S OWN `sfp-ring.mlir:9,25` — `SFP_SETDEST :: imm:2  // set dest for transfer from SFP of core 0
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

    /// IBM'S OWN TWO IMMCOPY LINES — `PE_IMMCOPY :: imm:12121 mask:255 replica:1 tgtrf:R0  //
    /// splat/pad to create vector` (a DD2 PE, so NO `mode`, `splat-vector-reg-init.mlir:17`) and
    /// `SFP_IMMCOPY :: imm:25256 mask:255 mode:fp32 replica:0 tgtrf:R0  // splat #2 splat/pad to
    /// create vector` (`splat_constant.mlir:9`).
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

    /// IBM'S OWN `samv.mlir:8` — `LX_SAMV :: maskall:no mvridx:MVR0 numvalidentry:12 precision:8b
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
    /// e092: an SFP splat of an lx input into lrf3 — the field order the reference writes, the mask
    /// inverted, and the pad as the immediate. A precision the mode cannot spell is an offender.
    #[test]
    fn a_splat_inverts_its_mask_and_carries_the_pad_as_the_immediate() {
        let splat = Splat {
            pad: SplatPad::None,
            mask: 8,
            unroll: UnrollFactor::X2,
            unroll_incr_result: true,
        };
        let (instr, refused) = construct_splat_instr_from_splat_op::<Sen1p5>(
            UniformInstrInfo::of(OpCode::NOP),
            Port::Lx,
            SplatTarget::Lrf(LrfIndex::L3),
            ComputeComp::Sfp,
            splat,
            Precision::Fp16,
        );
        assert_eq!(instr.opcode, OpCode::SPLAT);
        assert!(refused.is_empty());
        assert_eq!(
            instr.common_fields,
            vec![
                (OperandField::Imm, int(1)),
                (OperandField::Mask, int(247)),
                (OperandField::Mode, descriptive("fp16")),
                (OperandField::Src0, descriptive("lxlu")),
                (OperandField::Src2, descriptive("0.0")),
                (OperandField::Tgtrf, descriptive("R3")),
                (OperandField::Unrlfldtgt, boolean(true)),
                (OperandField::Unroll, descriptive("x2")),
            ]
        );

        // `left` padding is 0, a port target takes `result`, and a DD2 PE has no `mode` to refuse.
        let (padded, refused) = construct_splat_instr_from_splat_op::<Dd2>(
            UniformInstrInfo::of(OpCode::NOP),
            Port::Sfp,
            SplatTarget::Port(super::SplatPort::Lx),
            ComputeComp::Pe,
            Splat {
                pad: SplatPad::Left,
                ..splat
            },
            Precision::Int8,
        );
        assert!(
            refused.is_empty(),
            "a DD2 PE never reaches the precision check"
        );
        assert_eq!(padded.common_field(OperandField::Imm), Some(&int(0)));
        assert_eq!(
            padded.common_field(OperandField::Tgtlx),
            Some(&descriptive("result"))
        );
        assert_eq!(padded.common_field(OperandField::Mode), None);

        // An SFP does have a mode, and int8 is not one it spells.
        let (int8, refused) = construct_splat_instr_from_splat_op::<Sen1p5>(
            UniformInstrInfo::of(OpCode::NOP),
            Port::Lx,
            SplatTarget::Lrf(LrfIndex::L3),
            ComputeComp::Sfp,
            splat,
            Precision::Int8,
        );
        assert_eq!(refused, vec![SplatRefusal::Precision(Precision::Int8)]);
        assert_eq!(int8.common_field(OperandField::Mode), None);
    }

    /// e098: IBM'S THREE ROUTES, ONE PER `CHECK` LINE.
    /// `SFP_SPLAT :: imm:0 mask:255 mode:fp16 src0:pe src2:0.0 tgtrf:R0 unrlfldtgt:false unroll:x1
    /// // splat #1` (`splat_none_constant.mlir:7`); a program header's unpadded constant emitting NO
    /// instruction and `LRF : 0 : #SEN169_FP16 : 0x5a59…` instead (`splat-vector-reg-init.mlir:8-11`,
    /// 23129 packed twice per word); and the same constant PADDED, which stays an
    /// `PE_IMMCOPY :: imm:23129 mask:255 replica:0 tgtrf:R0` (`splat-vector-reg-init.mlir:26`).
    #[test]
    fn a_program_headers_unpadded_constant_initialises_the_register_instead_of_copying_into_it() {
        let unpadded = Splat {
            pad: SplatPad::None,
            mask: 0,
            unroll: UnrollFactor::X1,
            unroll_incr_result: false,
        };
        let mut reg_graph = RegGraphs::default();
        let (splat, refused) = construct_splat_pad_instr::<Dd2>(
            ComputeUnit::Sfp,
            SplatPadInput::Port(Port::Pe),
            SplatTarget::Lrf(LrfIndex::L0),
            &SplatPadOp {
                splat: Splat {
                    pad: SplatPad::Left,
                    ..unpadded
                },
                precision: Precision::Fp16,
                program_header: false,
                dbg_name: Some("splat #1"),
            },
            &[],
            &mut reg_graph,
            None,
        );
        assert!(refused.is_empty());
        let splat = splat.expect("a port input splats");
        assert_eq!(splat.opcode, OpCode::SPLAT);
        assert_eq!(
            splat.common_fields,
            vec![
                (OperandField::Imm, int(0)),
                (OperandField::Mask, int(255)),
                (OperandField::Mode, descriptive("fp16")),
                (OperandField::Src0, descriptive("pe")),
                (OperandField::Src2, descriptive("0.0")),
                (OperandField::Tgtrf, descriptive("R0")),
                (OperandField::Unrlfldtgt, boolean(false)),
                (OperandField::Unroll, descriptive("x1")),
            ]
        );
        assert_eq!(splat.comment, Comment::Common("splat #1".to_owned()));
        assert!(
            reg_graph.per_unit.is_empty(),
            "the SPLAT route initialises nothing"
        );
        // ⛔ AND THE PORT ROUTE ANSWERS FOR THE DATA FORMAT TOO (`:3798-3800`).
        let (none, refused) = construct_splat_pad_instr::<Dd2>(
            ComputeUnit::Sfp,
            SplatPadInput::Port(Port::Pe),
            SplatTarget::Lrf(LrfIndex::L0),
            &SplatPadOp {
                splat: unpadded,
                precision: Precision::Int16,
                program_header: false,
                dbg_name: None,
            },
            &[],
            &mut reg_graph,
            None,
        );
        assert_eq!(refused, vec![SplatPadRefusal::Precision(Precision::Int16)]);
        assert!(none.is_none(), "an unsupported format splats nothing");

        let constant = SplatPadConst::Scalar {
            value: 23129,
            is_symbol: false,
        };
        let header = SplatPadOp {
            splat: unpadded,
            precision: Precision::Fp16,
            program_header: true,
            dbg_name: None,
        };
        let pe = UnitKey {
            unit: DfirUnit::Pe,
            core: Core::checked(1).expect("every arch has core 1"),
            corelet: Corelet::checked(0),
        };
        let (none, refused) = construct_splat_pad_instr::<Dd2>(
            ComputeUnit::Pe,
            SplatPadInput::Const(constant),
            SplatTarget::Lrf(LrfIndex::L0),
            &header,
            &[pe],
            &mut reg_graph,
            None,
        );
        assert!(refused.is_empty());
        assert!(none.is_none(), "a program header emits no instruction");
        assert_eq!(
            reg_graph.get(pe),
            Some(&vec![RegInit {
                file: RegType::Lrf,
                index: RegIndex::at::<0>(),
                value: Operand::every(OperandValue::Int128([0x5a59_5a59; 4]))
                    .in_format(DataFormat::Sen169Fp16),
            }])
        );

        // ⛔ THE PAD DECIDES, NOT THE HEADER: `left` keeps the IMMCOPY even in a program header.
        let mut copy_ops = 0;
        let (padded, refused) = construct_splat_pad_instr::<Dd2>(
            ComputeUnit::Pe,
            SplatPadInput::Const(constant),
            SplatTarget::Lrf(LrfIndex::L0),
            &SplatPadOp {
                splat: Splat {
                    pad: SplatPad::Left,
                    ..unpadded
                },
                ..header
            },
            &[pe],
            &mut reg_graph,
            Some(&mut copy_ops),
        );
        assert!(refused.is_empty());
        let padded = padded.expect("a padded constant copies");
        assert_eq!(padded.opcode, OpCode::IMMCOPY);
        assert_eq!(copy_ops, 1);
        assert_eq!(
            padded.common_fields,
            vec![
                (
                    OperandField::Imm,
                    int(23129).in_format(DataFormat::Sen169Fp16)
                ),
                (OperandField::Mask, int(255)),
                (OperandField::Replica, int(0)),
                (OperandField::Tgtrf, descriptive("R0")),
            ]
        );
    }
}
