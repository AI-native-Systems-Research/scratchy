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
use crate::bridges::sentient_to_progir::construct::descriptive;
use crate::bridges::sentient_to_progir::uniform::instr::UniformInstrInfo;
use crate::islands::progir::OperandField;
use crate::islands::sentient::dialects::sentient::{Port, Precision};

// ⛔ ONE `crustify:todo:` PER SCHEDULED UNIT. Replace each with the ported function
// carrying `/// Replaces: eNNN_name`. A surviving TODO is open work.

// crustify:todo: e085_ConstructFMAInstr
// crustify:todo: e086_ConstructBinaryInstr
// crustify:todo: e087_ConstructUnaryInstr
// crustify:todo: e088_ConstructTernaryInstr

/// WHICH COMPUTE UNIT — `SenComponents` narrowed to the three an FMA runs on (`:1191`, `:1329`).
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

/// WHICH SOURCE FIELD AN INPUT LANDS IN — the `port_name` every callsite passes (`:1580-1596`).
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
    /// ⛔ [`Precision::None`] IS A DANGLING MAC and suppresses the conversion (`:1279`).
    pub result: Precision,
}

/// WHAT AN INPUT OPERAND CANNOT BE — the three refusals of `:1213-1268` and `:1287-1291`.
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

/// The ISA's name for one input port on one unit — `:1191-1268`, in the reference's own order.
fn compute_input_value<A: Arch>(
    unit: ComputeComp,
    port: Port,
    precisions: ComputePrecisions,
) -> Option<String> {
    match port {
        // ⛔ THE PT PAIRS ITS LOW FOUR REGISTERS at sub-byte compute precision (`:1191-1199`).
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
            // Shared by the PE and the SFP (`:1225-1233`), then each unit's own.
            ComputeComp::Pe | ComputeComp::Sfp => match (port, unit) {
                (Port::Lx, _) => Some("lxlu".to_owned()),
                (Port::NbrSlice, _) => Some("nbrslice".to_owned()),
                (Port::Two, _) => Some("2.0".to_owned()),
                (Port::Three, _) => Some("3.0".to_owned()),
                // ⭐ BOTH NEIGHBOUR-FORWARD PORTS SPELL `nfwd`; the reference's third `"nfwd"` arm
                // (`:1227`) is dead, as no port is spelled that.
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
/// ⛔ THE FIRST WRITER WINS (`:1305`) — a slot already filled is left alone.
/// ⛔ THE `fold` SUFFIX IS UNCONDITIONAL: every fold mode other than `fold_AB_Both`, INCLUDING NO
/// FOLD MODE AT ALL, is a `DT_ERROR` there (`:1294-1298`), so reaching a conversion at all proves it.
/// ⭐ `is_operand_forwarded` IS PASSED WITH OPPOSITE POLARITY BY THE FMA AND BINARY CALLSITES
/// (`:1590`, `:2121`) AND NEVER READ, so it is not a parameter here.
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
/// source slot or the op's own result (`:1600`, `:2126`).
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
/// ⛔ THE SELF-FORWARD REFUSAL IS DEAD CODE: `(comp == SFP && …) && (comp == PE && …)` (`:1377-1378`)
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

#[cfg(test)]
mod unit_tests {
    use super::{
        ComputeComp, ComputePrecisions, ComputeSlot, ComputeSource, UnsupportedComputeInput,
        set_compute_input_operand, set_compute_output_operands,
    };
    use crate::arch::Dd2;
    use crate::bridges::sentient_to_progir::construct::descriptive;
    use crate::bridges::sentient_to_progir::uniform::instr::UniformInstrInfo;
    use crate::islands::progir::{OpCode, OperandField};
    use crate::islands::sentient::dialects::sentient::{Port, Precision};

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
        // ⛔ AND THE FIRST WRITER WINS — a filled slot is left alone (`:1305`).
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
}
