//! THE OPAQUE-TEMPLATE INSTRUCTION — `ConstructOpaqueInstr` and the string trim it uses.
//!
//! ⭐ THE TEMPLATES ARE DATA, IN `dcc/src/Conversion/SentientToProgIR/opaqueTemplates/` — 38
//! `.smc` files (exp, gelu, sigmoid, rsqrt, layernormscale, idx32toaddr, …). They are a
//! declared-data table, not code; this crate's `build.rs` ALREADY reads `.smc` mnemonics and
//! emits `InstOpCode::…`, so a template naming a non-opcode fails to compile.
//!
//! 2 units. Every citation resolves against the authority tree
//! `/Users/nickm/git/deeptools-src` at revision `a0d29abbed`.
//!
//! | unit | level | LoC | authority |
//! |---|---|---|---|
//! | `e004_rtrim` | 0 | 6 | `dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:3859` |
//! | `e063_ConstructOpaqueInstr` | 1 | 167 | `dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:3866` |

use super::{descriptive, int};
use crate::bridges::sentient_to_progir::uniform::instr::UniformInstrInfo;
use crate::generated::{
    OpaqueSlot, OpaqueSlotValue, OpaqueTemplate, ParamKey, ParamValue, RegName,
};
use crate::islands::dataflow_ir::dialects::dataflow::RegAddr;
use crate::islands::progir::OperandField;
use crate::islands::progir::ty::{Operand, OperandValue};

// ⛔ ONE `crustify:todo:` PER SCHEDULED UNIT. Replace each with the ported function
// carrying `/// Replaces: eNNN_name`. A surviving TODO is open work.

/// Replaces: e004_rtrim
///
/// One template line with its trailing whitespace dropped.
///
/// ⛔⛔ C'S `isspace` SET, WHICH NEITHER RUST SPELLING MATCHES. `str::trim_end` uses Unicode
/// `White_Space` and so also eats U+00A0 and U+2028; `u8::is_ascii_whitespace` omits `\v` (0x0B),
/// which `std::isspace` includes. A template line ending in a vertical tab would keep it under one
/// and lose a non-breaking space under the other.
#[must_use]
pub fn rtrim(s: &str) -> &str {
    s.trim_end_matches(|ch: char| ch == ' ' || ('\u{9}'..='\u{d}').contains(&ch))
}

/// THE TRIP COUNT AN OPAQUE CARRIES BESIDE ITS PARAMS — the `l0` the two `pt_slice_mask_*` bodies
/// splice into a `MVLOOPCNT`.
///
/// ⛔ NOT A [`ParamValue`], AND THE REFERENCE SAYS SO BY CALLING `stoi` ON IT
/// (`ConstructProgIRHelper.cpp:3955`): `ParamValue` is a closed set of SPELLINGS and this is an
/// arbitrary count, so a `params` entry could not hold it without opening that set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct LoopCount(pub u32);

/// ONE `sentient.opaque` AS ITS SPLICER READS IT — the body it names and the three dictionaries
/// that fill the body's holes.
#[derive(Debug, Clone, Copy)]
pub struct OpaqueInvocation<'a> {
    /// WHICH BODY, already resolved: [`OpaqueTemplate::of`] is the reference's file-name chain.
    pub template: OpaqueTemplate,
    pub read_write: &'a [(RegName, RegAddr)],
    pub read_only: &'a [(RegName, RegAddr)],
    pub params: &'a [(ParamKey, ParamValue)],
    pub loop_count: Option<LoopCount>,
    pub dbg_name: Option<&'a str>,
}

/// A HOLE THE INVOCATION DID NOT FILL — every `signalPassFailure` on this path, as an offender.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpaqueRefusal {
    /// A register the body reads that neither dictionary allocates.
    UnboundRegister(RegName),
    /// A parameter the body reads that `params` does not state.
    UnfilledParam(ParamKey),
    /// A name no dictionary COULD supply — a finding about the vendored body itself.
    UnboundVariable(&'static str),
    /// An `imm=l0` with no [`LoopCount`] beside it, which is the reference's `DT_CHECK(...)`.
    NoLoopCount,
}

/// Replaces: e063_ConstructOpaqueInstr
///
/// One opaque op spliced out into the instructions its template states.
///
/// ⛔⛔ THE UNIT PREFIX IS NOT PART OF THE OPCODE. The reference builds `op_code + "_" + line` and
/// re-parses it, so `PE_LOGICAL` is one string; here the opcode is bare and the unit is
/// [`OpaqueTemplate::unit`], which is what made the per-unit field tables usable at build time.
#[must_use]
pub fn construct_opaque_instr(
    op: &OpaqueInvocation<'_>,
) -> (Vec<UniformInstrInfo>, Vec<OpaqueRefusal>) {
    let mut refused = Vec::new();
    // SETMASK wraps after every increment of 8, so a count that is a multiple of 8 leaves the
    // state-resetting one at the tail redundant (`:3945-3956`).
    let drop_reset = op.loop_count.is_some_and(|count| count.0 % 8 == 0);
    let mut instrs = Vec::new();
    for instruction in op.template.body() {
        if drop_reset && instruction.resets_mask {
            continue;
        }
        let mut instr = UniformInstrInfo::of(instruction.opcode);
        for slot in instruction.slots {
            instr.set_common_field(slot.field, fill(op, slot, &mut refused));
        }
        if let Some(name) = op.dbg_name {
            instr = instr.with_common_comment(name);
        }
        instrs.push(instr);
    }
    (instrs, refused)
}

/// ONE SLOT'S OPERAND, with whatever the dictionaries owe it substituted in.
fn fill(op: &OpaqueInvocation<'_>, slot: &OpaqueSlot, refused: &mut Vec<OpaqueRefusal>) -> Operand {
    match slot.value {
        OpaqueSlotValue::Int(value) => int(value),
        OpaqueSlotValue::Float(value) => Operand::every(OperandValue::Float(value)),
        OpaqueSlotValue::Literal(name) => descriptive(name),
        OpaqueSlotValue::LoopCount => match op.loop_count {
            Some(count) => int(i64::from(count.0)),
            None => {
                refused.push(OpaqueRefusal::NoLoopCount);
                Operand::every(OperandValue::Unknown)
            }
        },
        OpaqueSlotValue::Unbound(name) => {
            refused.push(OpaqueRefusal::UnboundVariable(name));
            Operand::every(OperandValue::Unknown)
        }
        OpaqueSlotValue::Bound { reg, param } => bind(op, slot.field, reg, param, refused),
    }
}

/// THE THREE DICTIONARIES IN THE REFERENCE'S ORDER — read-only, then read-write, then the params
/// (`:3999-4009`).
///
/// ⛔⛔ THE `R` IS THE VALUE. `insertReg` writes `"R" + startAddress` and the consumer strips it back
/// off BY POSITION (`dcc/src/Dialect/Sentient/Utils.cpp:157`), so a substituted register is the
/// descriptive `R8` — the same string `read_write_register_dictionary = {P1 = "R1"}` states. An
/// integer here would print `8` and become `lrf` downstream, which is no port at all.
fn bind(
    op: &OpaqueInvocation<'_>,
    field: OperandField,
    reg: Option<RegName>,
    param: Option<ParamKey>,
    refused: &mut Vec<OpaqueRefusal>,
) -> Operand {
    if let Some(name) = reg {
        let found = op
            .read_only
            .iter()
            .chain(op.read_write)
            .find(|(held, _)| *held == name);
        if let Some((_, addr)) = found {
            return descriptive(&format!("R{}", addr.0));
        }
        if param.is_none() {
            refused.push(OpaqueRefusal::UnboundRegister(name));
            return Operand::every(OperandValue::Unknown);
        }
    }
    let Some(key) = param else {
        return Operand::every(OperandValue::Unknown);
    };
    let Some((_, value)) = op.params.iter().find(|(held, _)| *held == key) else {
        refused.push(OpaqueRefusal::UnfilledParam(key));
        return Operand::every(OperandValue::Unknown);
    };
    let spelling = value.spelling();
    match field {
        // `unroll` TAKES ITS `x` BACK: `params={"unroll"="2"}` names the file `reciprocalx2.smc` and
        // the field wants `x2` (`:4016-4019`).
        OperandField::Unroll if !spelling.starts_with('x') => descriptive(&format!("x{spelling}")),
        // An `imm` is a NUMBER, which is the reference's `stoi` on the same string (`:4014-4015`).
        OperandField::Imm => match spelling.parse::<i64>() {
            Ok(value) => int(value),
            Err(_) => Operand::every(OperandValue::Unknown),
        },
        _ => descriptive(spelling),
    }
}

#[cfg(test)]
mod unit_tests {
    use super::{
        LoopCount, OpaqueInvocation, OpaqueRefusal, OpaqueTemplate, ParamKey, ParamValue, RegAddr,
        RegName, construct_opaque_instr, descriptive, int, rtrim,
    };
    use crate::islands::progir::OperandField;

    /// ⛔ THE TAIL ONLY, AND C'S SET EXACTLY — the vertical tab goes, the non-breaking space stays.
    #[test]
    fn trims_cs_whitespace_off_the_tail_only() {
        assert_eq!(
            rtrim("  SFP_IMMCOPY imm:0 \t\r\n\u{b}\u{c}"),
            "  SFP_IMMCOPY imm:0"
        );
        assert_eq!(rtrim("PE_NOP"), "PE_NOP");
        assert_eq!(rtrim("   "), "");
        assert_eq!(rtrim("PE_NOP\u{a0}"), "PE_NOP\u{a0}");
    }
    /// ⛔ THE THREE DICTIONARIES, ALL AT ONCE, on `reciprocalx1.smc:2`'s
    /// `LOGICAL src0=in0 imm=0x1 src2=c1 tgtrf=p0_0 unrlfldtgt=1 unroll=x1 mask=255` — a param fills
    /// `in0`, two allocated registers become the `R<addr>` the dictionaries state, `0x1` is a
    /// number, `x1` stays the ISA's own name, and the fields come out in field order.
    #[test]
    fn the_opaque_body_takes_its_names_from_the_dictionaries() {
        let (instrs, refused) = construct_opaque_instr(&OpaqueInvocation {
            template: OpaqueTemplate::Reciprocalx1,
            read_write: &[
                (RegName::P00, RegAddr(0)),
                (RegName::T00, RegAddr(1)),
                (RegName::T20, RegAddr(2)),
                (RegName::T40, RegAddr(3)),
                (RegName::T60, RegAddr(4)),
            ],
            read_only: &[
                (RegName::C1, RegAddr(8)),
                (RegName::C2, RegAddr(9)),
                (RegName::C3, RegAddr(10)),
                (RegName::C4, RegAddr(11)),
            ],
            params: &[
                (ParamKey::In0, ParamValue::Lxlu),
                (ParamKey::Prec, ParamValue::Fp32),
                (ParamKey::Out0, ParamValue::Result),
                (ParamKey::Unroll, ParamValue::N1),
            ],
            loop_count: None,
            dbg_name: Some("opaque_op #1"),
        });
        assert_eq!(instrs.len(), OpaqueTemplate::Reciprocalx1.body().len());
        assert_eq!(
            instrs[0].common_fields,
            vec![
                (OperandField::Imm, int(1)),
                (OperandField::Mask, int(255)),
                (OperandField::Src0, descriptive("lxlu")),
                (OperandField::Src2, descriptive("R8")),
                (OperandField::Tgtrf, descriptive("R0")),
                (OperandField::Unrlfldtgt, int(1)),
                (OperandField::Unroll, descriptive("x1")),
            ]
        );
        // ⭐ `fwdencoding=out0` REACHES `tgtlx`, which is `convertArchDependentFields`' rewrite.
        let tail = instrs.last().expect("reciprocalx1 has a body");
        assert!(
            tail.common_fields
                .contains(&(OperandField::Tgtlx, descriptive("result")))
        );
        assert_eq!(refused, vec![]);
    }

    /// ⛔ THE REDUNDANT `SETMASK` GOES ONLY ON A MULTIPLE OF 8, and an absent count is the
    /// reference's `DT_CHECK(loop_count.has_value())` — reported here rather than aborted.
    #[test]
    fn the_tail_setmask_survives_a_loop_count_that_is_not_a_multiple_of_eight() {
        let write = |loop_count| {
            construct_opaque_instr(&OpaqueInvocation {
                template: OpaqueTemplate::PtSliceMaskArfWrite,
                read_write: &[],
                read_only: &[],
                params: &[],
                loop_count,
                dbg_name: None,
            })
        };
        let (kept, _) = write(Some(LoopCount(7)));
        let (dropped, _) = write(Some(LoopCount(16)));
        assert_eq!(kept.len(), 7);
        assert_eq!(dropped.len(), 6);
        // The trip count reaches the `MVLOOPCNT`'s own immediate.
        assert!(kept[1].common_fields.contains(&(OperandField::Imm, int(7))));
        let (_, refused) = write(None);
        assert_eq!(
            refused,
            vec![
                OpaqueRefusal::NoLoopCount,
                // 📏 THE ONE NAME THE VENDORED BODIES ASK FOR THAT NO RCUDD1A TABLE HAS.
                OpaqueRefusal::UnboundVariable("u0"),
            ]
        );
    }
}
