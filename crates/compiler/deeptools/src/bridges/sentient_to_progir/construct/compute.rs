//! THE COMPUTE INSTRUCTIONS — FMA, binary, unary and ternary, and the operand plumbing that
//! feeds them.
//!
//! ⛔ THE FMA DESTINATION IS `ResultForwarding` ON THE OP, not a register operand.
//! ⛔ `ConstructBinaryInstr` IS 485 LINES — the single largest unit in the span. Its branch
//! order and early returns ARE the port; a summary of its decision rule is not.
//!
//! 6 units. Every citation resolves against the authority tree
//! `/Users/nickm/git/deeptools-src` at revision `a0d29abbed`.
//!
//! | unit | level | LoC | authority |
//! |---|---|---|---|
//! | `e052_setSentientComputeInputProgIROperands` | 1 | 125 | `dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:1184` |
//! | `e053_setSentientComputeOutputProgIROperands` | 1 | 112 | `dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:1310` |
//! | `e085_ConstructFMAInstr` | 2 | 291 | `dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:1423` |
//! | `e086_ConstructBinaryInstr` | 2 | 485 | `dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:1715` |
//! | `e087_ConstructUnaryInstr` | 2 | 237 | `dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:2201` |
//! | `e088_ConstructTernaryInstr` | 2 | 122 | `dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:2439` |

use crate::arch::{Arch, IsaGen};
use crate::bridges::sentient_to_progir::construct::{boolean, descriptive, int};
use crate::bridges::sentient_to_progir::uniform::instr::UniformInstrInfo;
use crate::bridges::sentient_to_progir::utils::{
    ComputeUnit, InputPrecisions, SrcOperand, UnsupportedConversion, set_fc_value_from_fold_mode,
    unsupported_on_the_fly_conversions,
};
use crate::islands::progir::{OpCode, OperandField};
use crate::islands::sentient::dialects::sentient::{
    Binary, BinaryFcvt, BinaryOp, FmaMode, FoldMode, MergeWidth, Operand as SenOperand, Port,
    Precision, ResultPorts, UnrollFactor,
};
use sys_arch_spec::regfile::Component;

// ⛔ ONE `crustify:todo:` PER SCHEDULED UNIT. Replace each with the ported function
// carrying `/// Replaces: eNNN_name`. A surviving TODO is open work.

/// WHICH COMPUTE PRECISION AN FMA IS IN — the five opcode arms of `:3840-3866`. ⭐ THE TRAILING
/// `emitError` IS RETIRED BY THIS CLOSED SET, and its `"fp80"` (`:3845`) names no precision the
/// attribute can carry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MacPrecision {
    /// `int4` — `IMA4`.
    Int4,
    /// `mxint4` — `IMA4`.
    Mxint4,
    /// `int8` — `IMA8`.
    Int8,
    /// `fp8` — `FMA8`.
    Fp8,
    /// `mxfp8` — `FMA8`.
    Mxfp8,
    /// `mxfp4` — `FMA4`.
    Mxfp4,
    /// `fp16` — `FMA`.
    Fp16,
    /// `fp32` — `FMA`.
    Fp32,
    /// `bf16` — `FMA`.
    Bf16,
}

impl MacPrecision {
    /// The precision itself.
    #[must_use]
    pub const fn precision(self) -> Precision {
        match self {
            MacPrecision::Int4 => Precision::Int4,
            MacPrecision::Mxint4 => Precision::Mxint4,
            MacPrecision::Int8 => Precision::Int8,
            MacPrecision::Fp8 => Precision::Fp8,
            MacPrecision::Mxfp8 => Precision::Mxfp8,
            MacPrecision::Mxfp4 => Precision::Mxfp4,
            MacPrecision::Fp16 => Precision::Fp16,
            MacPrecision::Fp32 => Precision::Fp32,
            MacPrecision::Bf16 => Precision::Bf16,
        }
    }

    /// `setInstn` under `fused_mul_add` (`:3841-3857`).
    #[must_use]
    pub const fn opcode(self) -> OpCode {
        match self {
            MacPrecision::Int4 | MacPrecision::Mxint4 => OpCode::IMA4,
            MacPrecision::Int8 => OpCode::IMA8,
            MacPrecision::Fp8 | MacPrecision::Mxfp8 => OpCode::FMA8,
            MacPrecision::Mxfp4 => OpCode::FMA4,
            MacPrecision::Fp16 | MacPrecision::Fp32 | MacPrecision::Bf16 => OpCode::FMA,
        }
    }

    /// ⭐ ONLY THE `FMA` ARM NAMES ITS OWN MODE (`:3857-3862`).
    const fn writes_mode(self) -> bool {
        matches!(
            self,
            MacPrecision::Fp16 | MacPrecision::Fp32 | MacPrecision::Bf16
        )
    }
}

/// ONE OF THE MAC'S THREE OPERANDS AND THE SLOT IT LANDS IN.
#[derive(Debug, Clone, PartialEq)]
pub struct MacOperand {
    /// `op<X>`, with its precision and its forwarding.
    pub operand: SenOperand,
    /// ⛔ A SLOT, NOT A PORT ID: `"src" + getOp<X>PortID()` (`:3889`) spells an unassigned `-1` as
    /// the field `src-1`.
    pub slot: ComputeSlot,
}

/// A `sentient.mac` AS THE FMA READS IT — `MacOp`'s own fields, less the mechanisms.
#[derive(Debug, Clone, PartialEq)]
pub struct Mac {
    /// `opA`.
    pub op_a: MacOperand,
    /// `opB`.
    pub op_b: MacOperand,
    /// `opC`.
    pub op_c: MacOperand,
    /// `ResultForwarding` and `ResultPrecision` — ⛔ A REQUIRED ATTRIBUTE, so the reference's
    /// `!getResultForwarding()` (`:3823`) is dead and an empty forwarding is spelled `[none]`.
    pub result: ResultPorts,
    /// `mode`.
    pub mode: FmaMode,
    /// `ComputePrecision`.
    pub compute: MacPrecision,
    /// `getFoldMode()`.
    pub fold_mode: Option<FoldMode>,
    /// `unrollFactor`.
    pub unroll_factor: UnrollFactor,
    /// The mask constant — ⛔ *"Mask constant value has to exist"* (`:3880`, `:4005`) is
    /// unrepresentable: the value is here, not an op to look through.
    pub mask: i64,
    /// `isDataWeight`, set by `AnnotateMacXRFWtRange` — ⛔ THE REFERENCE'S BACKWARD WALK TO THE
    /// PREVIOUS MAC WITH XRF POINTERS (`:4041-4058`) READS THAT SAME PASS'S ANSWER, so it is one
    /// field here rather than a search.
    pub is_data_weight: bool,
    /// `dbgName`.
    pub dbg_name: Option<String>,
}

/// WHAT AN FMA CANNOT DO — its own three refusals and the ones its operand plumbing hands back.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FmaRefusal {
    /// `verifyOnTheFlyConversions`' (`:3826`).
    Conversion(UnsupportedConversion),
    /// An input's (`:3944-3957`).
    Input(UnsupportedComputeInput),
    /// An output's (`:3959-3970`).
    Output(UnsupportedComputeOutput),
    /// *"FPUOP is not well defined for FMA32 in RCUDD1A"* (`:3832-3835`).
    ResultFp32OnRcudd1a,
    /// *"FNMS is supported only in PE and SFP"* (`:3868-3869`).
    FnmsOnThePt,
    /// *"PT FMA/IMA does not support setting unrlfldsrc2, but it was requested"* (`:3991-3995`).
    UnrollOnPtSrc2,
    /// *"N-link has to be in fp4/fp8 precision"* (`:4029-4031`).
    NLinkPrecision(Precision),
}

/// AN FMA AND WHAT IT COULD NOT DO.
#[derive(Debug, Clone, PartialEq)]
pub struct Fma {
    /// The instruction.
    pub instr: UniformInstrInfo,
    /// The offenders, in the order the reference reaches them.
    pub refused: Vec<FmaRefusal>,
}

/// Replaces: e085_ConstructFMAInstr
///
/// The multiply-accumulate itself: its opcode from the compute precision, its three inputs in the
/// slots their port ids named, every forwarding destination, the mask and the unroll flags.
///
/// ⭐ THE MASK IS WRITTEN TWICE (`:3878`, `:4003`) — `{PE, SFP}` and `!= PT` are the same set of the
/// three units, so the second write is the first's own value.
/// ⚠️ `unroll_illegal` IS ASSIGNED, NOT ACCUMULATED (`:3971`, `:3978`, `:3985`), so only the last
/// PT-`src2` operand's flag is ever reported. ⛔ AND `SET_IFIFO_CONVERT` TAKES ITS DOCUMENTED
/// DEFAULT (`:4014`): no env gate decides an instruction here.
#[must_use]
pub fn construct_fma_instr<A: Arch>(mac: &Mac, comp: ComputeComp) -> Fma {
    let compute = mac.compute.precision();
    let result = mac.result.precision;
    let sen1p5 = matches!(A::GEN, IsaGen::Sen1p5);
    let mut refused = Vec::new();
    let operands = [&mac.op_a, &mac.op_b, &mac.op_c];
    // `:3814-3821` — src0 carries constraints the other two slots do not.
    let src0 = [SrcOperand::OpA, SrcOperand::OpB, SrcOperand::OpC]
        .into_iter()
        .zip(operands)
        .find(|(_, held)| held.slot == ComputeSlot::Src0)
        .map(|(which, _)| which);
    if let (Some(unit), true) = (comp.compute_unit(), result != Precision::None) {
        refused.extend(
            unsupported_on_the_fly_conversions::<A>(
                unit,
                InputPrecisions {
                    op_a: mac.op_a.operand.precision,
                    op_b: mac.op_b.operand.precision,
                    op_c: mac.op_c.operand.precision,
                },
                src0,
                compute,
                result,
            )
            .into_iter()
            .map(FmaRefusal::Conversion),
        );
    }
    let mut instr = UniformInstrInfo::of(match mac.mode {
        FmaMode::FusedMulAdd => mac.compute.opcode(),
        FmaMode::FusedNegMulSub => OpCode::FNMS,
    });
    if let Some(name) = &mac.dbg_name {
        instr = instr.with_common_comment(name);
    }
    if comp.compute_unit().is_some() && result != compute && result != Precision::None {
        if result == Precision::Fp32 && !sen1p5 {
            refused.push(FmaRefusal::ResultFp32OnRcudd1a);
        }
        // `:3838-3839` — fp16 out of a compute unit really means dlfp16.
        let fpuop = if result == Precision::Fp16 {
            "dlfp16"
        } else {
            result.spelling()
        };
        instr.set_common_field(OperandField::Fpuop, descriptive(fpuop));
    }
    if mac.mode == FmaMode::FusedNegMulSub && comp == ComputeComp::Pt {
        refused.push(FmaRefusal::FnmsOnThePt);
    }
    // `:3857-3861`, `:3870-3875` — the SFP always names its mode, the PE only on Sentient 1.5.
    if (mac.compute.writes_mode() || mac.mode == FmaMode::FusedNegMulSub)
        && (comp == ComputeComp::Sfp || (sen1p5 && comp == ComputeComp::Pe))
    {
        instr.set_common_field(OperandField::Mode, descriptive(compute.spelling()));
    }
    instr.set_common_field(OperandField::Mask, int(255 - mac.mask));
    for held in operands {
        if let Some(refusal) = set_compute_input_operand::<A>(
            comp,
            &mut instr,
            held.slot,
            held.operand.port,
            ComputePrecisions {
                op: held.operand.precision,
                compute,
                result,
            },
            false,
        ) {
            refused.push(FmaRefusal::Input(refusal));
        }
    }
    for held in operands {
        refused.extend(
            set_compute_output_operands::<A>(
                comp,
                &mut instr,
                &held.operand.forwarding,
                ComputeSource::Slot(held.slot),
            )
            .into_iter()
            .map(FmaRefusal::Output),
        );
    }
    refused.extend(
        set_compute_output_operands::<A>(
            comp,
            &mut instr,
            &mac.result.forwarding,
            ComputeSource::Result,
        )
        .into_iter()
        .map(FmaRefusal::Output),
    );
    if !instr.has_common_field(OperandField::Tgtrf) {
        instr.set_common_field(OperandField::Tgtrf, descriptive("no"));
    }
    instr.set_common_field(
        OperandField::Unroll,
        descriptive(mac.unroll_factor.spelling()),
    );
    let mut unroll_illegal = false;
    for held in operands {
        if comp == ComputeComp::Pt && held.slot == ComputeSlot::Src2 {
            unroll_illegal = held.operand.unroll_incr;
        } else {
            instr.set_common_field(held.slot.unroll_field(), boolean(held.operand.unroll_incr));
        }
    }
    instr.set_common_field(OperandField::Unrlfldtgt, boolean(mac.result.unroll_incr));
    if unroll_illegal {
        refused.push(FmaRefusal::UnrollOnPtSrc2);
    }
    if sen1p5 {
        set_fc_value_from_fold_mode(&mut instr, comp.component(), mac.fold_mode);
        // `:4016-4022` — the PT's fp8 n-link input is converted to fp9 by an explicit bit, where DD2
        // does it in hardware.
        let mx = matches!(
            mac.compute,
            MacPrecision::Mxfp8 | MacPrecision::Mxfp4 | MacPrecision::Mxint4
        );
        if comp == ComputeComp::Pt
            && (mx || mac.compute == MacPrecision::Fp8)
            && (mac.op_a.operand.port == Port::Zero || mac.op_b.operand.port == Port::Zero)
            && mac.op_c.operand.port == Port::North
        {
            let n_link = mac.op_c.operand.precision;
            if !matches!(n_link, Precision::Fp4 | Precision::Fp8) {
                refused.push(FmaRefusal::NLinkPrecision(n_link));
            }
            let convert = if mx {
                (mac.is_data_weight && mac.compute == MacPrecision::Mxfp8)
                    || (!mac.is_data_weight && mac.compute == MacPrecision::Mxfp4)
            } else {
                true
            };
            if convert {
                instr.set_common_field(OperandField::IfifoConv, descriptive("yes"));
            }
        }
    }
    Fma { instr, refused }
}
// crustify:todo: e087_ConstructUnaryInstr
// crustify:todo: e088_ConstructTernaryInstr

/// WHICH COMPUTE UNIT — `SenComponents` narrowed to the three an FMA runs on (`:1213,1234,1248`, `:1329`).
///
/// ⛔ THE PT IS ONE OF THEM HERE, so this is NOT
/// [`crate::bridges::sentient_to_progir::utils::ComputeUnit`], which is the PE/SFP-only pair the
/// on-the-fly conversion question is asked of. ⭐ THREE VARIANTS ALSO RETIRE THE REFERENCE'S FINAL
/// *"Unrecognized operand for FMA"* — with no fourth component it cannot be reached.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeComp {
    /// `PT`.
    Pt,
    /// `PE`.
    Pe,
    /// `SFP`.
    Sfp,
}

/// WHICH SOURCE FIELD AN INPUT LANDS IN — the `port_name` every callsite passes (`:1572-1583`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeSlot {
    /// `src0`.
    Src0,
    /// `src1`.
    Src1,
    /// `src2`.
    Src2,
}

impl ComputeSlot {
    /// The field itself.
    #[must_use]
    pub const fn field(self) -> OperandField {
        match self {
            ComputeSlot::Src0 => OperandField::Src0,
            ComputeSlot::Src1 => OperandField::Src1,
            ComputeSlot::Src2 => OperandField::Src2,
        }
    }
}

/// THE THREE PRECISIONS THE CONVERSION TEST READS — named so two cannot be swapped by position
/// (`:1187-1189`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ComputePrecisions {
    /// This operand's own precision.
    pub op: Precision,
    /// What the unit computes in.
    pub compute: Precision,
    /// ⛔ [`Precision::None`] IS A DANGLING MAC and suppresses the conversion (`:1278`).
    pub result: Precision,
}

/// WHAT AN INPUT OPERAND CANNOT BE — the three refusals of `:1222,1245,1264` and `:1282-1288`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnsupportedComputeInput {
    /// *"Unsupported operand for `<unit>` FMA"* — this unit has no such input.
    Port {
        /// The unit.
        unit: ComputeComp,
        /// The port it cannot read.
        port: Port,
    },
    /// *"Only expect data received from PT in PE units"* — a 24-bit input outside the PE.
    Wide {
        /// The unit.
        unit: ComputeComp,
        /// The op precision.
        precision: Precision,
    },
    /// *"Unsupported input on the fly conversion"* — only fp16→fp32 converts implicitly.
    Conversion {
        /// The op precision.
        op: Precision,
        /// The compute precision.
        compute: Precision,
    },
}

/// The ISA's name for one input port on one unit — `:1192-1270`, in the reference's own order.
fn compute_input_value<A: Arch>(
    unit: ComputeComp,
    port: Port,
    precisions: ComputePrecisions,
) -> Option<String> {
    match port {
        // ⛔ THE PT PAIRS ITS LOW FOUR REGISTERS at sub-byte compute precision (`:1193-1201`).
        Port::Lrf(index) => {
            let index = u32::from(index.get());
            let paired = matches!(unit, ComputeComp::Pt)
                && matches!(
                    precisions.compute,
                    Precision::Int2 | Precision::Int4 | Precision::Int8
                )
                && index < 4;
            Some(format!("R{}", if paired { index / 2 } else { index }))
        }
        Port::Irf0 => Some("irf0".to_owned()),
        Port::Irf1 => Some("irf1".to_owned()),
        Port::Latch => Some("reuse".to_owned()),
        Port::Zero => Some("0.0".to_owned()),
        // ⛔⛔ `"none".contains("one")` IS TRUE (`:1211`), so an ABSENT operand reads as a literal 1.0
        // rather than falling through to this unit's own ports.
        Port::One | Port::None => Some("1.0".to_owned()),
        _ => match unit {
            ComputeComp::Pt => match port {
                Port::Xrf => Some("xrf".to_owned()),
                Port::North | Port::CrossPtNorthLink => Some("n-link".to_owned()),
                Port::West => Some("w-link".to_owned()),
                _ => None,
            },
            // Shared by the PE and the SFP (`:1226-1233`), then each unit's own.
            ComputeComp::Pe | ComputeComp::Sfp => match (port, unit) {
                (Port::Lx, _) => Some("lxlu".to_owned()),
                (Port::NbrSlice, _) => Some("nbrslice".to_owned()),
                (Port::Two, _) => Some("2.0".to_owned()),
                (Port::Three, _) => Some("3.0".to_owned()),
                // ⭐ BOTH NEIGHBOUR-FORWARD PORTS SPELL `nfwd`; the reference's third `"nfwd"` arm
                // (`:1228`) is dead, as no port is spelled that.
                (Port::Nfwd0 | Port::Nfwd2, _) => Some("nfwd".to_owned()),
                (Port::Pe, ComputeComp::Sfp) => Some("pe".to_owned()),
                (Port::SfpRing, ComputeComp::Sfp) => Some("datafifo".to_owned()),
                (Port::Sfp, ComputeComp::Pe) => Some("sfp".to_owned()),
                // `pt` with an int16 operand is a field of its own on DD2 (`:1252-1257`).
                (Port::Pt, ComputeComp::Pe) => Some(
                    match (A::GEN, precisions.op) {
                        (IsaGen::Rcudd1a, Precision::Int16) => "ptint16",
                        _ => "pt",
                    }
                    .to_owned(),
                ),
                _ => None,
            },
        },
    }
}

/// Replaces: e052_setSentientComputeInputProgIROperands
///
/// Name one input port in the ISA's own vocabulary and put it in `src0`/`src1`/`src2`.
///
/// ⛔ THE FIRST WRITER WINS (`:1303`) — a slot already filled is left alone.
/// ⛔ THE `fold` SUFFIX IS UNCONDITIONAL: every fold mode other than `fold_AB_Both`, INCLUDING NO
/// FOLD MODE AT ALL, is a `DT_ERROR` there (`:1294-1298`), so reaching a conversion at all proves it.
/// ⭐ `is_operand_forwarded` IS PASSED WITH OPPOSITE POLARITY BY THE FMA AND BINARY CALLSITES
/// (`:1575`, `:2114`) AND NEVER READ, so it is not a parameter here.
pub fn set_compute_input_operand<A: Arch>(
    unit: ComputeComp,
    instr: &mut UniformInstrInfo,
    slot: ComputeSlot,
    port: Port,
    precisions: ComputePrecisions,
    is_gcvt_fcvt: bool,
) -> Option<UnsupportedComputeInput> {
    let Some(mut value) = compute_input_value::<A>(unit, port, precisions) else {
        return Some(UnsupportedComputeInput::Port { unit, port });
    };
    let converting = matches!(A::GEN, IsaGen::Sen1p5)
        && precisions.op != precisions.compute
        && precisions.compute != Precision::None
        && precisions.op != Precision::Int1
        && precisions.result != Precision::None
        && matches!(unit, ComputeComp::Pe | ComputeComp::Sfp)
        && matches!(port, Port::Sfp | Port::Pe | Port::Lx | Port::L0 | Port::Pt);
    if converting {
        let compute = match precisions.op {
            Precision::Fp24 | Precision::Int24 => {
                if !matches!(unit, ComputeComp::Pe) {
                    return Some(UnsupportedComputeInput::Wide {
                        unit,
                        precision: precisions.op,
                    });
                }
                Precision::Fp32
            }
            op => {
                if !is_gcvt_fcvt && (op != Precision::Fp16 || precisions.compute != Precision::Fp32)
                {
                    return Some(UnsupportedComputeInput::Conversion {
                        op,
                        compute: precisions.compute,
                    });
                }
                precisions.compute
            }
        };
        value.push_str(&format!(
            "{}to{}fold",
            precisions.op.spelling(),
            compute.spelling()
        ));
    }
    if !instr.has_common_field(slot.field()) {
        instr.set_common_field(slot.field(), descriptive(&value));
    }
    None
}

/// WHICH VALUE A FORWARDING FIELD CARRIES — the `port_name` the output callsites pass, either one
/// source slot or the op's own result (`:1585-1596`, `:2121-2129`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeSource {
    /// One of the inputs.
    Slot(ComputeSlot),
    /// `result`.
    Result,
}

impl ComputeSource {
    /// The spelling the field takes.
    #[must_use]
    pub const fn spelling(self) -> &'static str {
        match self {
            ComputeSource::Slot(ComputeSlot::Src0) => "src0",
            ComputeSource::Slot(ComputeSlot::Src1) => "src1",
            ComputeSource::Slot(ComputeSlot::Src2) => "src2",
            ComputeSource::Result => "result",
        }
    }
}

/// WHAT AN OUTPUT CANNOT DO — the refusals of `:1329-1400`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnsupportedComputeOutput {
    /// *"Unsupported destination for `<unit>` FMA"* — this unit cannot reach that port.
    Destination {
        /// The unit.
        unit: ComputeComp,
        /// The port it cannot write.
        port: Port,
    },
    /// *"unsupported tgtencoding for PE"* — the PE does not forward to the L0 or the PT.
    Forward(Port),
    /// *"Multiple operands forwarded in the same direction"* — this field is already taken.
    Collision(OperandField),
    /// *"Only FMA result can be written to register"* / *"…forwarded to the SFP ring"*.
    NotTheResult(OperandField),
}

/// Replaces: e053_setSentientComputeOutputProgIROperands
///
/// Point every one of this op's outputs at the field that forwards it.
///
/// ⛔ SEN1P5 DROPS THE PT'S `tgte` ENTIRELY (`:1322-1326`) — *"either always west data or stale"*.
/// ⛔ THE SELF-FORWARD REFUSAL IS DEAD CODE: `(comp == SFP && …) && (comp == PE && …)` (`:1374-1375`)
/// is one component being two at once, so no arm is emitted for it.
/// ⭐ EVERY REFUSAL HERE FALLS THROUGH AND STILL WRITES ITS FIELD, so the offenders are COLLECTED
/// rather than returned at the first one. Its two precision parameters are never read.
pub fn set_compute_output_operands<A: Arch>(
    unit: ComputeComp,
    instr: &mut UniformInstrInfo,
    outputs: &[Port],
    source: ComputeSource,
) -> Vec<UnsupportedComputeOutput> {
    let mut offenders = Vec::new();
    let sen1p5 = matches!(A::GEN, IsaGen::Sen1p5);
    for output in outputs.iter().copied() {
        if output == Port::None || (sen1p5 && unit == ComputeComp::Pt && output == Port::East) {
            continue;
        }
        let field = match unit {
            ComputeComp::Pt => match output {
                Port::East => Some(OperandField::Tgte),
                Port::South => Some(OperandField::Tgts),
                Port::Lrf(_) | Port::Irf0 | Port::Irf1 | Port::Xrf => Some(OperandField::Tgtrf),
                _ => None,
            },
            ComputeComp::Pe | ComputeComp::Sfp if !sen1p5 => match (output, unit) {
                (Port::Lx, _) => Some(OperandField::Tgtlx),
                (Port::Lrf(_), _) => Some(OperandField::Tgtrf),
                (Port::Sfp, ComputeComp::Pe) => Some(OperandField::Tgtsfp),
                (Port::Pe, ComputeComp::Sfp) => Some(OperandField::Tgtpe),
                (Port::Pt, ComputeComp::Sfp) => Some(OperandField::Tgtpt),
                (Port::L0, ComputeComp::Sfp) => Some(OperandField::Tgtl0),
                (Port::SfpRing, ComputeComp::Sfp) => {
                    if source != ComputeSource::Result {
                        offenders.push(UnsupportedComputeOutput::NotTheResult(
                            OperandField::Tgtdatafifo,
                        ));
                    }
                    Some(OperandField::Tgtdatafifo)
                }
                _ => None,
            },
            // SEN1P5 forwards through one encoding pair instead of one field per direction.
            ComputeComp::Pe | ComputeComp::Sfp => match output {
                Port::Pe | Port::Pt | Port::L0 | Port::Sfp | Port::Lx => {
                    if unit == ComputeComp::Pe && matches!(output, Port::L0 | Port::Pt) {
                        offenders.push(UnsupportedComputeOutput::Forward(output));
                    }
                    instr.set_common_field(
                        OperandField::Tgtencoding,
                        descriptive(&output.spelling()),
                    );
                    Some(OperandField::Fwdencoding)
                }
                Port::Lrf(_) => Some(OperandField::Tgtrf),
                Port::SfpRing => Some(OperandField::Tgtdatafifo),
                _ => None,
            },
        };
        let Some(field) = field else {
            offenders.push(UnsupportedComputeOutput::Destination { unit, port: output });
            continue;
        };
        if instr.has_common_field(field) {
            offenders.push(UnsupportedComputeOutput::Collision(field));
        }
        // The register file takes the RESULT, under the register's own name (`:1404-1414`).
        let value = if field == OperandField::Tgtrf {
            if source != ComputeSource::Result {
                offenders.push(UnsupportedComputeOutput::NotTheResult(field));
            }
            match output {
                Port::Xrf => "xrf".to_owned(),
                Port::Irf0 => "irf0".to_owned(),
                Port::Irf1 => "irf1".to_owned(),
                Port::Lrf(index) => format!("R{}", index.get()),
                _ => source.spelling().to_owned(),
            }
        } else {
            source.spelling().to_owned()
        };
        instr.set_common_field(field, descriptive(&value));
    }
    offenders
}

// WHICH `SenComponents` A COMPUTE UNIT IS.
impl ComputeComp {
    /// The component the shared helpers still take.
    #[must_use]
    pub const fn component(self) -> Component {
        match self {
            ComputeComp::Pt => Component::Pt,
            ComputeComp::Pe => Component::Pe,
            ComputeComp::Sfp => Component::Sfp,
        }
    }

    /// The PE/SFP pair the on-the-fly conversion question is asked of — ⛔ `None` FOR THE PT, which
    /// is never asked (`:1813-1815`).
    #[must_use]
    pub const fn compute_unit(self) -> Option<ComputeUnit> {
        match self {
            ComputeComp::Pt => None,
            ComputeComp::Pe => Some(ComputeUnit::Pe),
            ComputeComp::Sfp => Some(ComputeUnit::Sfp),
        }
    }
}

impl ComputeSlot {
    /// The `unrlfldsrc<n>` field this slot's unroll increment lands in (`:2146`, `:2153`).
    #[must_use]
    pub const fn unroll_field(self) -> OperandField {
        match self {
            ComputeSlot::Src0 => OperandField::Unrlfldsrc0,
            ComputeSlot::Src1 => OperandField::Unrlfldsrc1,
            ComputeSlot::Src2 => OperandField::Unrlfldsrc2,
        }
    }
}

/// ONE OPERAND OF A `vector_binary` AS THIS INSTRUCTION READS IT — the op's own bundle, plus the
/// source slot its port id names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BinaryOperand<'a> {
    /// `opXPortID`, resolved — ⛔ A SLOT, NOT AN INTEGER: the reference spells
    /// `"src" + to_string(id)` (`:2061-2062`), which writes `src-1` for the `.td`'s unassigned `-1`.
    pub slot: ComputeSlot,
    /// `opX`, `opXPrecision`, `opXForwarding` and `unrollIncrOpX`.
    pub operand: &'a SenOperand,
}

/// THE `sentient.vector_binary` A BINARY INSTRUCTION IS BUILT FROM — every attribute it reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BinaryInstrOp<'a> {
    /// `opA`, and where it lands.
    pub op_a: BinaryOperand<'a>,
    /// `opB`.
    pub op_b: BinaryOperand<'a>,
    /// `$binaryOp`, with the logical result forward the seven compares may carry.
    pub binary_op: Binary,
    /// `ResultForwarding`, `ResultPrecision` and `unrollIncrResult`.
    pub result: &'a ResultPorts,
    /// `$ComputePrecision`.
    pub compute_precision: Precision,
    /// `$unrollFactor`.
    pub unroll_factor: UnrollFactor,
    /// `$fold_mode`.
    pub fold_mode: Option<FoldMode>,
    /// `getMask()`'s constant — ⛔ THE FIELD TAKES `255 -` IT (`:2187`), and *"Mask constant value
    /// has to exist."* (`:2190`) is retired by this being a value rather than a defining op.
    pub mask: i64,
    /// `$dbgName`.
    pub dbg_name: Option<&'a str>,
}

/// WHICH OF THE THREE MODAL FAMILIES REFUSED A PRECISION — the reference's message names it, and
/// LOGICAL, MERGE and PACK make the identical pair of checks (`:1844-1849`, `:1865-1872`,
/// `:1897-1902`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModalFamily {
    /// `LOGICAL`.
    Logical,
    /// `MERGE`.
    Merge,
    /// `PACK`.
    Pack,
}

/// WHAT A `vector_binary` COULD NOT BE — every refusal of `ConstructBinaryInstr`, in the order the
/// reference reaches them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BinaryRefusal {
    /// *"Fused multiply and divide by 2 is not allowed in Pt units."* (`:1745-1747`).
    MulDiv2OnPt,
    /// *"FMA instructions in PT are currently supported only for int4, int8, mxint4, mxfp4, mxfp8,
    /// fp8 and fp16 types"* (`:1761-1764`).
    PtComputePrecision {
        /// `$ComputePrecision`.
        compute: Precision,
    },
    /// *"Unsupported operation in PT"* (`:1814`) — every operator but `add` and `mul`.
    OperationOnPt {
        /// `$binaryOp`.
        operator: BinaryOp,
    },
    /// *"Input on the fly conversion not supported by  FMINMAX"* (`:1822-1825`) — ⭐ `int1` IS
    /// ADMITTED FOR `opA` ALONE.
    FminmaxInput {
        /// `opAPrecision`.
        op_a: Precision,
        /// `opBPrecision`.
        op_b: Precision,
        /// `$ComputePrecision`.
        compute: Precision,
    },
    /// *"Output on the fly conversion not supported by FMINMAX"* (`:1826-1827`).
    FminmaxOutput {
        /// `$ComputePrecision`.
        compute: Precision,
        /// `ResultPrecision`.
        result: Precision,
    },
    /// *"`<family>` instruction precision expected to be none"*.
    ModalComputePrecision {
        /// Which family's message it is.
        family: ModalFamily,
        /// `$ComputePrecision`.
        compute: Precision,
    },
    /// *"Input precisions expected to match result precision for `<family>`"*.
    ModalOperandPrecision {
        /// Which family's message it is.
        family: ModalFamily,
        /// `opAPrecision`.
        op_a: Precision,
        /// `opBPrecision`.
        op_b: Precision,
        /// `ResultPrecision`.
        result: Precision,
    },
    /// *"GCVT instruction precision expected to be fp16"* (`:1917-1918`).
    GcvtComputePrecision {
        /// `$ComputePrecision`.
        compute: Precision,
    },
    /// *"Unexpected input/result precisions for GCVT mode 0/4/24/28"* (`:1929-1932`).
    GcvtOperands {
        /// `opAPrecision`.
        op_a: Precision,
        /// `opBPrecision`.
        op_b: Precision,
    },
    /// *"FCVT instruction precision expected to be fp32"* (`:1940-1941`).
    FcvtComputePrecision {
        /// `$ComputePrecision`.
        compute: Precision,
    },
    /// *"Unsupported FCVT instruction"* (`:1965-1967`) — ⛔ `fcvt_imm4` IS A SPELLABLE MODE THE
    /// `switch` HAS NO ARM FOR, which is why this is reachable where GCVT's twin is not.
    FcvtMode {
        /// `$binaryOp`'s immediate.
        mode: BinaryFcvt,
    },
    /// *"FCVT mode 7 only supported in Sentient1p5"* (`:1959-1960`).
    FcvtModeOnRcudd1a {
        /// `$binaryOp`'s immediate.
        mode: BinaryFcvt,
    },
    /// *"Unexpected input/result precisions for FCVT mode `<n>`"* (`:1949-1964`).
    FcvtPrecisions {
        /// `$binaryOp`'s immediate.
        mode: BinaryFcvt,
        /// `opAPrecision`.
        op_a: Precision,
        /// `opBPrecision`.
        op_b: Precision,
        /// `ResultPrecision`.
        result: Precision,
    },
    /// *"Input on the fly conversion not supported by FCMP"* (`:1974-1976`).
    FcmpInput {
        /// `opAPrecision`.
        op_a: Precision,
        /// `opBPrecision`.
        op_b: Precision,
        /// `$ComputePrecision`.
        compute: Precision,
    },
    /// *"Output on the fly conversion not supported by FCMP"* (`:1977-1978`).
    FcmpOutput {
        /// `$ComputePrecision`.
        compute: Precision,
        /// `ResultPrecision`.
        result: Precision,
    },
    /// `DT_CHECK(result_precision != "none")` (`:2027`) — the reference's one unmessaged check here.
    FpuopResultNone,
    /// *"Fpuop result conversion unsupported"* (`:2038-2039`) — a form with no `fpuop` field cannot
    /// convert on the way out.
    ResultConversion {
        /// `$ComputePrecision`.
        compute: Precision,
        /// `ResultPrecision`.
        result: Precision,
    },
    /// *"fp32 precision only valid in SFP in DD2"* (`:2044-2045`).
    Fp32OnPe,
    /// *"Expecting fp16 or fp32 precision in SFP or Sen1p5 PE"* (`:2051-2052`).
    ModePrecision {
        /// What the `mode` bit was asked to spell.
        mode: Precision,
    },
    /// *"Cannot have LOGICAL on pt input operands in Sen1p5"* (`:2165-2166`) — ⭐ AND THE
    /// REFERENCE'S TEST IS `contains("pt")`, so `crossptnlink` trips it too.
    LogicalOnPtInput,
    /// *"PT FMA/IMA does not support setting unrlfldsrc2, but it was requested"* (`:2178-2180`).
    PtUnrollOnSrc2,
    /// One of `verifyOnTheFlyConversions`' four (`:1796`, `:1805`).
    Conversion(UnsupportedConversion),
    /// One of `setSentientComputeInputProgIROperands`' three (`:2112`, `:2116`).
    Input(UnsupportedComputeInput),
    /// One of `setSentientComputeOutputProgIROperands`' four (`:2121-2129`).
    Output(UnsupportedComputeOutput),
}

/// A BINARY INSTRUCTION AND WHAT ITS OP COULD NOT ASK FOR.
#[derive(Debug, Clone, PartialEq)]
pub struct BinaryInstr {
    /// ⛔ `None` WHERE THE REFERENCE HAS NO OPCODE TO SET: `UniformInstrInfo` leaves `instn_`
    /// uninitialised (`UniformInstrAndBlock.hpp:146`), so its three PT refusals below hand back an
    /// instruction whose opcode is whatever the stack held — and two of them go on filling its
    /// fields (`:1764`, `:1815`).
    pub instr: Option<UniformInstrInfo>,
    /// The offenders, in the order the reference reaches them.
    pub refused: Vec<BinaryRefusal>,
}

/// WHAT ONE OPERATOR BECOMES ON ONE UNIT — the opcode, the constant operand it pins, and the `imm`
/// its modal forms carry.
struct BinaryForm {
    /// `setInstn`.
    opcode: OpCode,
    /// The `src1`/`src2` constant an FMA form pins before any operand is placed (`:1767-1790`).
    seed: Option<(ComputeSlot, &'static str)>,
    /// `instr_has_fpuop_field`.
    has_fpuop: bool,
    /// `is_gcvt_or_fcvt`.
    is_gcvt_or_fcvt: bool,
    /// `second_opcode` — ⛔ ONLY THE MODAL FORMS HAVE ONE, and only they write `imm` (`:2018`).
    second_opcode: Option<i64>,
}

/// The PT's FMA/IMA opcode for one compute precision (`:1750-1759`). ⭐ `"fp80"` (`:1754`) IS NOT A
/// PRECISION SPELLING, so that arm only ever fires on `fp8`.
const fn pt_fma_opcode(compute: Precision) -> Option<OpCode> {
    match compute {
        Precision::Int4 | Precision::Mxint4 => Some(OpCode::IMA4),
        Precision::Int8 => Some(OpCode::IMA8),
        Precision::Fp8 | Precision::Mxfp8 => Some(OpCode::FMA8),
        Precision::Mxfp4 => Some(OpCode::FMA4),
        Precision::Fp16 => Some(OpCode::FMA),
        _ => None,
    }
}

/// `verifyOnTheFlyConversions` as the FMA forms call it (`:1796`, `:1805`) — ⛔ `opC_precision` IS
/// THE COMPUTE PRECISION AT BOTH CALLSITES, which is what makes its opC arm never fire.
fn verify_conversions<A: Arch>(
    comp: ComputeComp,
    op: &BinaryInstrOp<'_>,
    refused: &mut Vec<BinaryRefusal>,
) {
    // `:1731-1735` — which operand carries port 0, and `None` where the reference keeps its `-1`.
    let src0 = if op.op_a.slot == ComputeSlot::Src0 {
        Some(SrcOperand::OpA)
    } else if op.op_b.slot == ComputeSlot::Src0 {
        Some(SrcOperand::OpB)
    } else {
        None
    };
    refused.extend(
        comp.compute_unit()
            .into_iter()
            .flat_map(|unit| {
                unsupported_on_the_fly_conversions::<A>(
                    unit,
                    InputPrecisions {
                        op_a: op.op_a.operand.precision,
                        op_b: op.op_b.operand.precision,
                        op_c: op.compute_precision,
                    },
                    src0,
                    op.compute_precision,
                    op.result.precision,
                )
            })
            .map(BinaryRefusal::Conversion),
    );
}

/// The pair of checks LOGICAL, MERGE and PACK each make word for word.
fn modal_precisions(family: ModalFamily, op: &BinaryInstrOp<'_>, refused: &mut Vec<BinaryRefusal>) {
    let compute = op.compute_precision;
    let (op_a, op_b) = (op.op_a.operand.precision, op.op_b.operand.precision);
    let result = op.result.precision;
    if compute != Precision::None {
        refused.push(BinaryRefusal::ModalComputePrecision { family, compute });
    }
    if op_a != result || op_b != result {
        refused.push(BinaryRefusal::ModalOperandPrecision {
            family,
            op_a,
            op_b,
            result,
        });
    }
}

/// `second_opcode` with the state register the logical result forwards to folded in three bits up
/// (`:1998-2016`). ⛔ THE REFERENCE'S `DT_CHECK(is_fcmp_or_fminmax)` (`:2001`) IS RETIRED BY
/// [`Binary::Forwarding`], which admits only the seven compare and min/max operators.
const fn with_logical_forward(second_opcode: i64, binary_op: Binary) -> i64 {
    match binary_op {
        Binary::Plain(_) => second_opcode,
        Binary::Forwarding {
            to: Port::IState(index),
            ..
        } => second_opcode + ((index.get() as i64) << 3),
        // `:2012-2014` — any other forwarding port encodes as seven.
        Binary::Forwarding { .. } => second_opcode + (7 << 3),
    }
}

/// Which opcode, pinned operand and `imm` one operator takes on one unit — the reference's outer
/// three-way branch (`:1740-2016`).
fn binary_form<A: Arch>(
    comp: ComputeComp,
    op: &BinaryInstrOp<'_>,
    refused: &mut Vec<BinaryRefusal>,
) -> Option<BinaryForm> {
    let compute = op.compute_precision;
    let result = op.result.precision;
    let (op_a, op_b) = (op.op_a.operand.precision, op.op_b.operand.precision);
    let fma = |opcode, seed, has_fpuop| {
        Some(BinaryForm {
            opcode,
            seed: Some(seed),
            has_fpuop,
            is_gcvt_or_fcvt: false,
            second_opcode: None,
        })
    };
    let modal = |opcode, second_opcode| {
        Some(BinaryForm {
            opcode,
            seed: None,
            has_fpuop: false,
            is_gcvt_or_fcvt: false,
            second_opcode: Some(second_opcode),
        })
    };
    let convert = |opcode, second_opcode| {
        Some(BinaryForm {
            opcode,
            seed: None,
            has_fpuop: false,
            is_gcvt_or_fcvt: true,
            second_opcode: Some(second_opcode),
        })
    };
    match (op.binary_op.op(), comp) {
        // `:1744-1748` — the PT has no fused multiply-and-halve, and no instruction comes back.
        (BinaryOp::MulDiv2, ComputeComp::Pt) => {
            refused.push(BinaryRefusal::MulDiv2OnPt);
            None
        }
        // `:1750-1772` — the PT's FMA/IMA, keyed by compute precision, with the source it does not
        // read pinned to a constant.
        (operator @ (BinaryOp::Add | BinaryOp::Mul), ComputeComp::Pt) => {
            let opcode = pt_fma_opcode(compute);
            if opcode.is_none() {
                refused.push(BinaryRefusal::PtComputePrecision { compute });
            }
            let seed = if matches!(operator, BinaryOp::Mul) {
                (ComputeSlot::Src1, "0.0")
            } else {
                (ComputeSlot::Src2, "1.0")
            };
            fma(opcode?, seed, false)
        }
        // `:1776-1790` — the PE/SFP forms, where the pinned source is the other one.
        (operator @ (BinaryOp::Add | BinaryOp::Mul | BinaryOp::MulDiv2), _) => {
            let (opcode, seed) = match operator {
                BinaryOp::Mul => (OpCode::FMUL, (ComputeSlot::Src2, "0.0")),
                BinaryOp::MulDiv2 => (OpCode::FMUL, (ComputeSlot::Src2, "muldiv2")),
                // ⭐ `"Unknown binary operation."` (`:1792`) CANNOT FIRE — this arm is `add`.
                _ => (OpCode::FMA, (ComputeSlot::Src1, "1.0")),
            };
            verify_conversions::<A>(comp, op, refused);
            fma(opcode, seed, true)
        }
        // `:1801-1810` — `sub` is an FNMS with its multiplier pinned to one.
        (BinaryOp::Sub, ComputeComp::Pe | ComputeComp::Sfp) => {
            verify_conversions::<A>(comp, op, refused);
            fma(OpCode::FNMS, (ComputeSlot::Src1, "1.0"), true)
        }
        // `:1813-1815` — every other operator on the PT, which reaches no opcode and no `imm`.
        (operator, ComputeComp::Pt) => {
            refused.push(BinaryRefusal::OperationOnPt { operator });
            None
        }
        // `:1816-1838` — the min/max pair and their absolute forms.
        (operator @ (BinaryOp::Min | BinaryOp::Max | BinaryOp::AbsMin | BinaryOp::AbsMax), _) => {
            if !(op_a == compute || op_a == Precision::Int1) || op_b != compute {
                refused.push(BinaryRefusal::FminmaxInput {
                    op_a,
                    op_b,
                    compute,
                });
            }
            if compute != result {
                refused.push(BinaryRefusal::FminmaxOutput { compute, result });
            }
            modal(
                OpCode::FMINMAX,
                match operator {
                    BinaryOp::Max => 0,
                    BinaryOp::AbsMax => 2,
                    BinaryOp::Min => 4,
                    // ⭐ `"Unsupported MINMAX variant"` (`:1837`) is `abs_min` and nothing else.
                    _ => 6,
                },
            )
        }
        // `:1840-1861` — the bitwise pair-ups, whose mode is always fp16 and so has no `mode` field.
        (operator @ (BinaryOp::And | BinaryOp::Or | BinaryOp::Xnor | BinaryOp::AndNot), _) => {
            modal_precisions(ModalFamily::Logical, op, refused);
            modal(
                OpCode::LOGICAL,
                match operator {
                    BinaryOp::And => 0,
                    BinaryOp::Or => 1,
                    BinaryOp::Xnor => 2,
                    // ⭐ `"Unsupported LOGICAL variant"` (`:1860`) is `and_not` and nothing else.
                    _ => 3,
                },
            )
        }
        // `:1863-1893` — ⛔ THE WIDTHS ARE NOT IN WIDTH ORDER: 8 encodes ABOVE 64.
        (BinaryOp::Merge { width, high }, _) => {
            modal_precisions(ModalFamily::Merge, op, refused);
            modal(
                OpCode::MERGE,
                match (width, high) {
                    (MergeWidth::W16, true) => 0,
                    (MergeWidth::W16, false) => 1,
                    (MergeWidth::W32, true) => 2,
                    (MergeWidth::W32, false) => 3,
                    (MergeWidth::W64, true) => 4,
                    (MergeWidth::W64, false) => 5,
                    (MergeWidth::W8, true) => 6,
                    (MergeWidth::W8, false) => 7,
                },
            )
        }
        // `:1895-1912` — the pack number the operator's own spelling carries. ⭐ THE RANGE TEST
        // (`:1910`) CANNOT FIRE: [`PackIndex`] tops out at 27.
        (BinaryOp::Pack(index), _) => {
            modal_precisions(ModalFamily::Pack, op, refused);
            modal(OpCode::PACK, i64::from(index.get()))
        }
        // `:1914-1935` — ⭐ ALL FOUR GCVT MODES MAKE THE SAME PRECISION CHECK, so the `switch`'s
        // `default` (`:1933-1935`) is dead: [`BinaryGcvt`] is exactly 0/4/24/28.
        (BinaryOp::GcvtImm(mode), _) => {
            if compute != Precision::Fp16 {
                refused.push(BinaryRefusal::GcvtComputePrecision { compute });
            }
            if op_a != Precision::Fp16 || op_b != Precision::Fp16 {
                refused.push(BinaryRefusal::GcvtOperands { op_a, op_b });
            }
            convert(OpCode::GCVT, i64::from(mode.get()))
        }
        // `:1937-1967` — one result precision per FCVT mode, and mode 7 needs the newer ISA.
        (BinaryOp::FcvtImm(mode), _) => {
            if compute != Precision::Fp32 {
                refused.push(BinaryRefusal::FcvtComputePrecision { compute });
            }
            let expected = match mode {
                BinaryFcvt::Imm2 => Some(Precision::Fp16),
                BinaryFcvt::Imm3 => Some(Precision::Fp8),
                BinaryFcvt::Imm7 => {
                    if !matches!(A::GEN, IsaGen::Sen1p5) {
                        refused.push(BinaryRefusal::FcvtModeOnRcudd1a { mode });
                    }
                    Some(Precision::Bf16)
                }
                BinaryFcvt::Imm4 => {
                    refused.push(BinaryRefusal::FcvtMode { mode });
                    None
                }
            };
            if let Some(expected) = expected {
                if op_a != Precision::Fp32 || op_b != Precision::Fp32 || result != expected {
                    refused.push(BinaryRefusal::FcvtPrecisions {
                        mode,
                        op_a,
                        op_b,
                        result,
                    });
                }
            }
            convert(OpCode::FCVT, i64::from(mode.get()))
        }
        // `:1969-1991` — the float compares. ⭐ AND THE CHAIN'S FINAL *"Unsupported operation in
        // PE/SFP"* (`:1994`) IS DEAD: the twenty operators are all accounted for above.
        (
            operator @ (BinaryOp::CompareEq
            | BinaryOp::CompareNeq
            | BinaryOp::CompareLt
            | BinaryOp::CompareLe),
            _,
        ) => {
            if op_a != compute || op_b != compute {
                refused.push(BinaryRefusal::FcmpInput {
                    op_a,
                    op_b,
                    compute,
                });
            }
            if compute != result {
                refused.push(BinaryRefusal::FcmpOutput { compute, result });
            }
            modal(
                OpCode::FCMP,
                match operator {
                    BinaryOp::CompareNeq => 4,
                    BinaryOp::CompareEq => 5,
                    BinaryOp::CompareLt => 6,
                    // ⭐ `"Unsupported FCMP variant"` (`:1990`) is `fcmp_le` and nothing else.
                    _ => 7,
                },
            )
        }
    }
}

/// Replaces: e086_ConstructBinaryInstr
///
/// One `vector_binary` as its unit's instruction: the opcode its operator and compute precision
/// choose, the constant operand an FMA form pins, the `imm` a modal form carries, and then every
/// field the two operands, the result and the unroll factor reach.
///
/// ⛔ A REFUSED PRECISION WRITES NO FIELD: where the reference aborts the compilation
/// (`DT_CHECK_MSG`) the offender comes back instead, and `fpuop`/`mode` stay unset rather than
/// carrying a value the ISA field cannot encode.
/// ⛔ AND THE `mode` BIT IS ASKED OF THE PT TOO (`:2043-2052`) — anything but fp16/fp32 there is
/// refused by a check the PT's own opcode table (`:1750-1759`) invites.
#[must_use]
pub fn construct_binary_instr<A: Arch>(comp: ComputeComp, op: &BinaryInstrOp<'_>) -> BinaryInstr {
    let mut refused = Vec::new();
    let Some(form) = binary_form::<A>(comp, op, &mut refused) else {
        return BinaryInstr {
            instr: None,
            refused,
        };
    };
    let compute = op.compute_precision;
    let result = op.result.precision;
    let mut instr = UniformInstrInfo::of(form.opcode);
    // `:1726-1727`.
    if let Some(name) = op.dbg_name {
        instr = instr.with_common_comment(name);
    }
    if let Some((slot, value)) = form.seed {
        instr.set_common_field(slot.field(), descriptive(value));
    }
    if let Some(second_opcode) = form.second_opcode {
        instr.set_common_field(
            OperandField::Imm,
            int(with_logical_forward(second_opcode, op.binary_op)),
        );
    }
    // `:2022-2041` — ⭐ THE `is_any_of(comp, PE, SFP)` CHECK (`:2025`) IS DEAD: only the PE/SFP arms
    // set `instr_has_fpuop_field`.
    if compute != Precision::None {
        if form.has_fpuop {
            if compute != result {
                if result == Precision::None {
                    refused.push(BinaryRefusal::FpuopResultNone);
                } else {
                    // `:2029-2032` — fp16 really means dlfp16 in Sen1p5.
                    let fpuop = match (result, A::GEN) {
                        (Precision::Fp16, IsaGen::Sen1p5) => "dlfp16",
                        _ => result.spelling(),
                    };
                    instr.set_common_field(OperandField::Fpuop, descriptive(fpuop));
                }
            }
        } else if !matches!(form.opcode, OpCode::GCVT | OpCode::FCVT) && compute != result {
            refused.push(BinaryRefusal::ResultConversion { compute, result });
        }
    }
    // `:2043-2057` — MERGE/PACK/LOGICAL have no compute precision, and their mode is always fp16.
    if matches!(A::GEN, IsaGen::Rcudd1a) && comp == ComputeComp::Pe {
        if compute == Precision::Fp32 {
            refused.push(BinaryRefusal::Fp32OnPe);
        }
    } else {
        let mode = if compute == Precision::None {
            Precision::Fp16
        } else {
            compute
        };
        if !matches!(mode, Precision::Fp16 | Precision::Fp32) {
            refused.push(BinaryRefusal::ModePrecision { mode });
        } else if form.opcode != OpCode::LOGICAL {
            instr.set_common_field(OperandField::Mode, descriptive(mode.spelling()));
        }
    }
    // `:2110-2119` — the two inputs, each into the slot its port id names.
    for operand in [op.op_a, op.op_b] {
        if let Some(refusal) = set_compute_input_operand::<A>(
            comp,
            &mut instr,
            operand.slot,
            operand.operand.port,
            ComputePrecisions {
                op: operand.operand.precision,
                compute,
                result,
            },
            form.is_gcvt_or_fcvt,
        ) {
            refused.push(BinaryRefusal::Input(refusal));
        }
    }
    // `:2121-2129` — an operand may be forwarded on as well as read, and the result always is.
    for (outputs, source) in [
        (
            op.op_a.operand.forwarding.as_slice(),
            ComputeSource::Slot(op.op_a.slot),
        ),
        (
            op.op_b.operand.forwarding.as_slice(),
            ComputeSource::Slot(op.op_b.slot),
        ),
        (op.result.forwarding.as_slice(), ComputeSource::Result),
    ] {
        refused.extend(
            set_compute_output_operands::<A>(comp, &mut instr, outputs, source)
                .into_iter()
                .map(BinaryRefusal::Output),
        );
    }
    // `:2131-2134` — no register file written is a written `no`.
    if !instr.has_common_field(OperandField::Tgtrf) {
        instr.set_common_field(OperandField::Tgtrf, descriptive("no"));
    }
    instr.set_common_field(
        OperandField::Unroll,
        descriptive(op.unroll_factor.spelling()),
    );
    // `:2143-2158` — ⛔ THE PT'S `src2` IS THE PINNED CONSTANT, so it has no unroll field to set.
    // ⭐ EITHER OPERAND ARMS THE REFUSAL: the reference's second assignment (`:2151`) would clear
    // what the first requested, which is reachable only with both operands in that one slot.
    let mut unroll_illegal = false;
    for operand in [op.op_a, op.op_b] {
        if comp == ComputeComp::Pt && operand.slot == ComputeSlot::Src2 {
            unroll_illegal |= operand.operand.unroll_incr;
        } else {
            instr.set_common_field(
                operand.slot.unroll_field(),
                boolean(operand.operand.unroll_incr),
            );
        }
    }
    instr.set_common_field(OperandField::Unrlfldtgt, boolean(op.result.unroll_incr));
    // `:2160-2166` — data from the PT arrives 24-bit and LOGICAL cannot upcast it.
    if matches!(A::GEN, IsaGen::Sen1p5)
        && comp == ComputeComp::Pe
        && form.opcode == OpCode::LOGICAL
        && [op.op_a, op.op_b]
            .iter()
            .any(|operand| operand.operand.port.spelling().contains("pt"))
    {
        refused.push(BinaryRefusal::LogicalOnPtInput);
    }
    // `:2168-2175` — ⛔ IN SEN1P5 `unrlfldsrc1` MOVES THE INTERNAL STATE REGISTER, overwriting
    // whatever operand sits in that slot. ⭐ AND ONLY A FORWARDING OP CAN ASK FOR IT, which retires
    // the `.td`'s independent `$unrollIncrLogicalResult`.
    if matches!(comp, ComputeComp::Pe | ComputeComp::Sfp)
        && matches!(A::GEN, IsaGen::Sen1p5)
        && matches!(
            op.binary_op,
            Binary::Forwarding {
                unroll_incr: true,
                ..
            }
        )
    {
        instr.set_common_field(OperandField::Unrlfldsrc1, boolean(true));
    }
    if unroll_illegal {
        refused.push(BinaryRefusal::PtUnrollOnSrc2);
    }
    // `:2184-2193` — ⛔ THE PT HAS NO LANE MASK, and the field is the complement of the op's.
    if comp != ComputeComp::Pt {
        instr.set_common_field(OperandField::Mask, int(255 - op.mask));
    }
    if matches!(A::GEN, IsaGen::Sen1p5) {
        set_fc_value_from_fold_mode(&mut instr, comp.component(), op.fold_mode);
    }
    BinaryInstr {
        instr: Some(instr),
        refused,
    }
}

#[cfg(test)]
mod unit_tests {
    use super::{
        BinaryInstr, BinaryInstrOp, BinaryOperand, ComputeComp, ComputePrecisions, ComputeSlot,
        ComputeSource, Mac, MacOperand, MacPrecision, UnsupportedComputeInput,
        construct_binary_instr, construct_fma_instr, set_compute_input_operand,
        set_compute_output_operands,
    };
    use crate::arch::{Dd2, Sen1p5};
    use crate::bridges::sentient_to_progir::construct::{boolean, descriptive, int};
    use crate::bridges::sentient_to_progir::uniform::instr::UniformInstrInfo;
    use crate::islands::progir::{OpCode, OperandField};
    use crate::islands::sentient::dialects::sentient::{
        Binary, BinaryOp, FmaMode, ForwardingOp, IStateIndex, LrfIndex, Operand as SenOperand,
        Port, Precision, ResultPorts, UnrollFactor,
    };

    /// All three fp16 — IBM's `vector_binary` carries no other precision in this test.
    const FP16: ComputePrecisions = ComputePrecisions {
        op: Precision::Fp16,
        compute: Precision::Fp16,
        result: Precision::Fp16,
    };

    /// ⭐⭐ IBM'S OWN LINE: `PE_FNMS :: mask:255 src0:pt src1:1.0 src2:lxlu tgtlx:result …`
    /// (`sub.mlir:8`) — ⛔ `src1:1.0` IS AN **ABSENT** OPERAND, because `"none".contains("one")`.
    #[test]
    fn an_absent_operand_reads_as_one_point_zero() {
        let mut instr = UniformInstrInfo::of(OpCode::FNMS);
        for (slot, port) in [
            (ComputeSlot::Src0, Port::Pt),
            (ComputeSlot::Src1, Port::None),
            (ComputeSlot::Src2, Port::Lx),
        ] {
            assert_eq!(
                set_compute_input_operand::<Dd2>(
                    ComputeComp::Pe,
                    &mut instr,
                    slot,
                    port,
                    FP16,
                    false
                ),
                None
            );
        }
        assert_eq!(
            instr.common_fields,
            vec![
                (OperandField::Src0, descriptive("pt")),
                (OperandField::Src1, descriptive("1.0")),
                (OperandField::Src2, descriptive("lxlu")),
            ]
        );
        // ⛔ AND THE FIRST WRITER WINS — a filled slot is left alone (`:1303`).
        set_compute_input_operand::<Dd2>(
            ComputeComp::Pe,
            &mut instr,
            ComputeSlot::Src0,
            Port::Sfp,
            FP16,
            false,
        );
        assert_eq!(
            instr.common_field(OperandField::Src0),
            Some(&descriptive("pt"))
        );
        // The PT's own links are not the PE's inputs.
        assert_eq!(
            set_compute_input_operand::<Dd2>(
                ComputeComp::Pe,
                &mut instr,
                ComputeSlot::Src1,
                Port::West,
                FP16,
                false
            ),
            Some(UnsupportedComputeInput::Port {
                unit: ComputeComp::Pe,
                port: Port::West
            })
        );
    }

    /// ⭐ `tgtlx:result` (`sub.mlir:8`) — and ⛔ ONLY THE RESULT MAY REACH A REGISTER FILE.
    #[test]
    fn the_forwarding_field_carries_what_produced_it() {
        let mut instr = UniformInstrInfo::of(OpCode::FNMS);
        assert_eq!(
            set_compute_output_operands::<Dd2>(
                ComputeComp::Pe,
                &mut instr,
                &[Port::None, Port::Lx],
                ComputeSource::Result
            ),
            vec![]
        );
        assert_eq!(
            instr.common_fields,
            vec![(OperandField::Tgtlx, descriptive("result"))]
        );
        let mut from_a_source = UniformInstrInfo::of(OpCode::FNMS);
        assert_eq!(
            set_compute_output_operands::<Dd2>(
                ComputeComp::Pe,
                &mut from_a_source,
                &[Port::Lrf(
                    crate::islands::sentient::dialects::sentient::LrfIndex::L3
                )],
                ComputeSource::Slot(ComputeSlot::Src0)
            ),
            vec![super::UnsupportedComputeOutput::NotTheResult(
                OperandField::Tgtrf
            )]
        );
        // ⛔ IT STILL WRITES THE FIELD, under the register's own name.
        assert_eq!(
            from_a_source.common_fields,
            vec![(OperandField::Tgtrf, descriptive("R3"))]
        );
    }

    /// ⭐⭐ IBM'S OWN LINE: `PE_FNMS :: mask:255 src0:pt src1:1.0 src2:lxlu tgtlx:result tgtrf:no
    /// unrlfldsrc0:false unrlfldsrc2:false unrlfldtgt:false unroll:x8` (`sub.mlir:8`) — ⛔ AND NO
    /// `mode`, because a DD2 PE gets the fp32 check instead of the field.
    #[test]
    fn a_subtraction_is_an_fnms_against_a_multiplier_pinned_to_one() {
        let op_a = SenOperand::from(Port::Lx);
        let op_b = SenOperand::from(Port::Pt);
        let result = ResultPorts {
            forwarding: vec![Port::Lx],
            ..ResultPorts::default()
        };
        let built = construct_binary_instr::<Dd2>(
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
        );
        let mut expected = UniformInstrInfo::of(OpCode::FNMS);
        expected.common_fields = vec![
            (OperandField::Mask, int(255)),
            (OperandField::Src0, descriptive("pt")),
            (OperandField::Src1, descriptive("1.0")),
            (OperandField::Src2, descriptive("lxlu")),
            (OperandField::Tgtlx, descriptive("result")),
            (OperandField::Tgtrf, descriptive("no")),
            (OperandField::Unrlfldsrc0, boolean(false)),
            (OperandField::Unrlfldsrc2, boolean(false)),
            (OperandField::Unrlfldtgt, boolean(false)),
            (OperandField::Unroll, descriptive("x8")),
        ];
        assert_eq!(
            built,
            BinaryInstr {
                instr: Some(expected),
                refused: vec![],
            }
        );
    }

    /// ⭐⭐ IBM'S OWN LINE: `SFP_FMINMAX :: foldctrl:folda imm:0 mask:255 mode:fp16 src0:R0 src2:nfwd
    /// tgtrf:R0 unrlfldsrc0:false unrlfldsrc2:false unrlfldtgt:false unroll:x1`
    /// (`fminmax_sen1p5.mlir:9`) — ⛔ `imm` CARRIES THE VARIANT **AND** THE STATE REGISTER THE
    /// LOGICAL RESULT FORWARDS TO, three bits up.
    #[test]
    fn a_maximum_carries_its_variant_and_its_state_register_in_one_immediate() {
        let op_a = SenOperand::from(Port::Lrf(LrfIndex::L0));
        let op_b = SenOperand::from(Port::Nfwd2);
        let result = ResultPorts {
            forwarding: vec![Port::Lrf(LrfIndex::L0)],
            ..ResultPorts::default()
        };
        let built = construct_binary_instr::<Sen1p5>(
            ComputeComp::Sfp,
            &BinaryInstrOp {
                op_a: BinaryOperand {
                    slot: ComputeSlot::Src0,
                    operand: &op_a,
                },
                op_b: BinaryOperand {
                    slot: ComputeSlot::Src2,
                    operand: &op_b,
                },
                binary_op: Binary::Forwarding {
                    op: ForwardingOp::Max,
                    to: Port::IState(IStateIndex::S0),
                    unroll_incr: false,
                },
                result: &result,
                compute_precision: Precision::Fp16,
                unroll_factor: UnrollFactor::X1,
                fold_mode: None,
                mask: 0,
                dbg_name: None,
            },
        );
        let mut expected = UniformInstrInfo::of(OpCode::FMINMAX);
        expected.common_fields = vec![
            (OperandField::Foldctrl, descriptive("folda")),
            (OperandField::Imm, int(0)),
            (OperandField::Mask, int(255)),
            (OperandField::Mode, descriptive("fp16")),
            (OperandField::Src0, descriptive("R0")),
            (OperandField::Src2, descriptive("nfwd")),
            (OperandField::Tgtrf, descriptive("R0")),
            (OperandField::Unrlfldsrc0, boolean(false)),
            (OperandField::Unrlfldsrc2, boolean(false)),
            (OperandField::Unrlfldtgt, boolean(false)),
            (OperandField::Unroll, descriptive("x1")),
        ];
        assert_eq!(
            built,
            BinaryInstr {
                instr: Some(expected),
                refused: vec![],
            }
        );
    }

    /// ⭐⭐ IBM'S OWN LINE — `PE_FMA :: mask:255 src0:ptint16 src1:1.0 src2:lxlu tgtrf:R0
    /// unrlfldsrc0:false unrlfldsrc1:false unrlfldsrc2:false unrlfldtgt:false unroll:x1`
    /// (`if_else_label3.mlir:19`, its `vector_mac` at `:259`). ⛔ `mask:255` IS `255 - 0`, and the
    /// PE names no `mode` on DD2.
    #[test]
    fn the_fma_takes_its_slots_from_the_port_ids() {
        let operand = |port, precision, slot| MacOperand {
            operand: SenOperand {
                port,
                forwarding: Vec::new(),
                precision,
                data_id: None,
                port_id: None,
                unroll_incr: false,
            },
            slot,
        };
        let mac = Mac {
            op_a: operand(Port::Pt, Precision::Int16, ComputeSlot::Src0),
            op_b: operand(Port::One, Precision::Fp16, ComputeSlot::Src1),
            op_c: operand(Port::Lx, Precision::Fp16, ComputeSlot::Src2),
            result: ResultPorts {
                forwarding: vec![Port::Lrf(LrfIndex::L0)],
                precision: Precision::Fp16,
                unroll_incr: false,
            },
            mode: FmaMode::FusedMulAdd,
            compute: MacPrecision::Fp16,
            fold_mode: None,
            unroll_factor: UnrollFactor::X1,
            mask: 0,
            is_data_weight: false,
            dbg_name: None,
        };
        let out = construct_fma_instr::<Dd2>(&mac, ComputeComp::Pe);
        assert_eq!(out.refused, vec![]);
        assert_eq!(out.instr.opcode, OpCode::FMA);
        assert_eq!(
            out.instr.common_fields,
            vec![
                (OperandField::Mask, int(255)),
                (OperandField::Src0, descriptive("ptint16")),
                (OperandField::Src1, descriptive("1.0")),
                (OperandField::Src2, descriptive("lxlu")),
                (OperandField::Tgtrf, descriptive("R0")),
                (OperandField::Unrlfldsrc0, boolean(false)),
                (OperandField::Unrlfldsrc1, boolean(false)),
                (OperandField::Unrlfldsrc2, boolean(false)),
                (OperandField::Unrlfldtgt, boolean(false)),
                (OperandField::Unroll, descriptive("x1")),
            ]
        );
    }
}
