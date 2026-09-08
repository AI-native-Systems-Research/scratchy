// SPDX-License-Identifier: Apache-2.0
//! Ported from `sys-arch-spec/dpc/dpc.cpp` — 3 unit(s) of bridge 4, `ProgIR -> SenProg`.
//!
//! ⛔ SENPROG IS A PRINT FORMAT, so what is being ported is the exact TEXT: field order,
//! spelling and separators. A predicate that decides what to write, without writing it, is not
//! a port. The authority is `/Users/nickm/git/deeptools-src/<file>:<line>` per unit below.

use sys_arch_spec::arch_enums::{RegType, SenComponent};
use sys_arch_spec::regfile::Component;
use sys_arch_spec::{CoreId, InstOpCode};

use super::isa_fields::{Isa, op_code_with_prefix};
use super::progir_print::{
    Pretty, ProgramCorelet, TagLccr, TagPc, UnitName, has_variables_in_instr, tag_to_lccr,
    tag_to_pc,
};
use crate::arch::Arch;
use crate::islands::progir::ty::OperandValue;
use crate::islands::progir::{
    Instruction, OperandField, Program, RegInit, UnitProgram, UnitRegState,
};
use crate::model::Model;
use crate::workload::Workload;

/// WHAT A FORMAT CAN EXPRESS — `ProgFormatFeatures` (`dpc.h:34-37`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProgFormatFeatures {
    /// `supportEnums` — whether a `DESCRIPTIVE` operand may stay a name instead of an encoding.
    pub support_enums: bool,
    /// `supportVariables` — whether an unresolved variable may survive into the output.
    pub support_variables: bool,
}

/// WHICH OUTPUT FORMAT — `ProgFormatNames` (`dpc.h:39`), narrowed to the rows
/// `progFormatFeaturesMap` actually has.
///
/// ⛔ NO `PASM`. Its row is commented out as `// PASM: TBD` (`dpc.h:89`), so
/// `progFormatFeaturesMap.at(PASM)` throws; leaving the variant out makes that an E0308.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProgFormat {
    /// `PROGIR`.
    ProgIr,
    /// `SMC`.
    Smc,
    /// `SENPROG`.
    Senprog,
    /// `SYSTEMCPROG`.
    SystemCProg,
    /// `BINARY`.
    Binary,
}

impl ProgFormat {
    /// `progFormatFeaturesMap` (`dpc.h:84-90`), row for row.
    #[must_use]
    pub const fn features(self) -> ProgFormatFeatures {
        match self {
            Self::ProgIr | Self::Smc => ProgFormatFeatures {
                support_enums: true,
                support_variables: true,
            },
            Self::Senprog | Self::SystemCProg => ProgFormatFeatures {
                support_enums: false,
                support_variables: true,
            },
            Self::Binary => ProgFormatFeatures {
                support_enums: false,
                support_variables: false,
            },
        }
    }
}

/// Replaces: e010_checkProgFormatCompatibility
///
/// May every core's program be written in `target`.
///
/// ⛔ ONLY `BINARY` CAN EVER REFUSE — it is the single row with `supportVariables = false`
/// (`dpc.h:90`), so `convertIr2Senprog`'s own guard on `SENPROG` (`dpc.cpp:619`) never fires.
/// ⛔ AND AN EMPTY SET PASSES: the reference's loop body never runs.
#[must_use]
pub fn check_prog_format_compatibility<A: Arch, M: Model, W: Workload>(
    programs: &[(CoreId, Program<A, M, W>)],
    target: ProgFormat,
) -> bool {
    target.features().support_variables
        || !programs
            .iter()
            .any(|(_, prog)| has_variables_in_instr(prog))
}

/// Replaces: e028_strToupper
///
/// `std::transform(.., ::toupper)` — what the reg-init header does to the unit name (`dpc.cpp:735`).
///
/// ⛔ ASCII, DELIBERATELY: `::toupper` is the C locale's, byte by byte, so a non-ASCII byte is left
/// alone. `to_uppercase` would case-fold it and is the wrong function here.
#[must_use]
pub fn str_toupper(input: &str) -> String {
    input.to_ascii_uppercase()
}

/// WHAT ONE UNIT CONTRIBUTES — one `senCompProgram_` entry beside its `regState_` entry.
///
/// ⛔ THE KEY IS CORELET-RESOLVED, WHICH `Program::per_unit` IS NOT. `senCompProgram_` is keyed by
/// `SenComponents` (`PE0`, `PTROW3_1`), and the whole header line is derived from that key
/// (`dpc.cpp:633-646`) — a generic [`sys_arch_spec::regfile::Component`] cannot state a corelet or a
/// row, so the writer takes its own entries rather than guessing one.
/// ⭐ `reg_state` IS THE REFERENCE'S `regState_.count(unit) > 0` (`:726`), as an `Option`.
#[derive(Debug, Clone, Copy)]
pub struct UnitEntry<'a> {
    /// The `senCompProgram_` key.
    pub unit: SenComponent,
    /// `unitProgPair.second`, already simplified — see [`UnitProgram::simple_instr_vect`].
    pub program: &'a UnitProgram,
    /// `regState_.at(unit)`, when the core has one for this unit.
    pub reg_state: Option<&'a UnitRegState>,
}

/// THE SENPROG TEXT, AND EVERY PLACE THE REFERENCE WOULD HAVE ABORTED INSTEAD.
///
/// ⛔ NOT A `Result`. `convertIr2Senprog` reaches nine distinct `DT_ERROR`s mid-stream, each of which
/// discards a partly written file; this crate never runtime-refuses, so the text is always produced
/// and what could not be written is listed. An empty [`Self::refused`] is the reference's success.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Senprog {
    /// `outputStream`'s contents.
    pub text: String,
    /// Empty when the reference would have completed.
    pub refused: Vec<Refusal>,
}

/// ONE THING THE TEXT COULD NOT STATE — the reference's abort sites, as values.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Refusal {
    /// *"Undefined type of unit"* — a `senCompProgram_` key with no `(name, corelet)` behind it, so
    /// `getSenComponent(unitName)` finds no ISA (`dpc.cpp:646-647`).
    UnnamedUnit(SenComponent),
    /// `isGraphSimple(true)`'s refusal — the unit's program is not one code block (`progir.h:464`).
    UnsimplifiedProgram(SenComponent),
    /// *"Illegal instruction/operand combination for this architecture"* — ⭐ ALSO WHERE AN OPCODE
    /// THIS UNIT DOES NOT HAVE LANDS, because then no operand can have a position (`dpc.cpp:678-685`).
    NoSuchField(InstOpCode, OperandField),
    /// `typeToFieldEncoding..at(name)`'s throw — a `DESCRIPTIVE` text the field does not encode.
    UnknownEncoding(InstOpCode, OperandField, String),
    /// *"Unexpected operand type, conversion not supported"* (`dpc.cpp:709`).
    UnprintableOperand(InstOpCode, OperandField),
    /// `tagToPC`'s two aborts, with the verdict that reached them.
    UnresolvedPc(TagPc),
    /// `tagToLCCR`'s three (`progir.cpp:731-806`).
    UnresolvedLccr(TagLccr),
    /// `regTypeToString.at(SCALE)`'s throw — ⛔ THE ONE FILE THE TABLE OMITS.
    UnnamedRegFile(RegType),
    /// *"Unsupported reg init data type for senprog generation"* (`dpc.cpp:768`).
    UnprintableRegInit(RegType),
}

/// `regTypeToString` (`progir.cpp:670-673`) — the suffix a non-LRF header carries.
///
/// ⛔ `SCALE` IS NOT IN IT. `arch_enums.h`'s `MAX_VALUE = STATE` names the second-to-last enumerator,
/// so the table stops one short of the enum and `at(SCALE)` throws — hence the `None`.
#[must_use]
pub const fn reg_type_to_string(file: RegType) -> Option<&'static str> {
    Some(match file {
        RegType::Lrf => "LRF",
        RegType::Lar => "LAR",
        RegType::Lbr => "LBR",
        RegType::Ear => "EAR",
        RegType::Ebr => "EBR",
        RegType::Gtr => "GTR",
        RegType::Jcr => "JCR",
        RegType::Erat => "ERAT",
        RegType::Mvr => "MVR",
        RegType::Xrf => "XRF",
        RegType::Spr => "SPR",
        RegType::Arf => "ARF",
        RegType::Irf => "IRF",
        RegType::State => "STATE",
        RegType::Scale => return None,
    })
}

/// Replaces: e032_convertIr2Senprog
///
/// The whole senprog text for `cores` — a `prog.txt` block per unit, each followed by that unit's
/// `reg_initial.txt` block when it has one.
///
/// ⛔ `useOldFormat` IS A DEAD `const false` (`dpc.cpp:648`), so `instrStart` and its ` );` closer are
/// computed and never printed; only the shift separator reaches the text.
/// ⛔ AND THE FORMAT GUARD CANNOT FIRE HERE: `SENPROG`'s row sets `supportVariables = true`
/// (`dpc.h:89`), which is what [`check_prog_format_compatibility`] tests, so it is not repeated.
/// ⭐ `targetCores` AND ITS *"Core ID specified not found"* ARE THE CALLER'S: `cores` is the selection,
/// already joined to its programs, in the ascending order that `std::set<int>` walks.
#[must_use]
pub fn convert_ir_to_senprog<A: Arch>(cores: &[(CoreId, &[UnitEntry<'_>])]) -> Senprog {
    let mut out = Senprog {
        text: String::new(),
        refused: Vec::new(),
    };
    for &(core, units) in cores {
        for entry in units {
            write_unit::<A>(&mut out, core, *entry);
        }
    }
    out
}

/// One `senCompProgram_` entry's two blocks (`dpc.cpp:631-775`).
fn write_unit<A: Arch>(out: &mut Senprog, core: CoreId, entry: UnitEntry<'_>) {
    // ⛔ THE L3 HALVES ARE STATED AS CORELET `0`, WHICH IS NOT A CORELET THEY HAVE (`dpc.cpp:635-636`)
    // — and `UnitName::of` lands them there anyway, since their `1` row is a hole in `senCompMap`.
    let Some((name, corelet)) = UnitName::of(entry.unit) else {
        out.refused.push(Refusal::UnnamedUnit(entry.unit));
        return;
    };
    let unit_name = name.spelling();
    let component = name.component();
    let isa = Isa::<A>::of(component);
    // ⭐ THE `UL` SUFFIX FOLLOWS THE UNIT, NOT A WIDTH: exactly `PE0|PE1|SFP0|SFP1` (`dpc.cpp:649-655`).
    let shift = match name {
        UnitName::Pe | UnitName::Sfp => "UL << ",
        _ => " << ",
    };
    let banner = |edge| {
        format!(
            "========== Core: {} Corelet: {} Unit: {unit_name} Program {edge} ============\n",
            core.0,
            corelet.spelling(),
        )
    };

    out.text.push_str("===== START file: prog.txt =====\n");
    out.text.push_str(&banner("START"));
    match entry.program.simple_instr_vect() {
        Some(instrs) => {
            for instr in instrs {
                write_instr(out, &isa, component, entry.program, instr, shift);
            }
        }
        None => out
            .refused
            .push(Refusal::UnsimplifiedProgram(entry.unit)),
    }
    out.text.push_str(&banner("END"));
    out.text.push_str("===== END file: prog.txt =====\n");

    let Some(reg_state) = entry.reg_state else {
        return;
    };
    out.text
        .push_str("===== START file: reg_initial.txt =====\n");
    // ⛔ ONE HEADER PER FILE, NOT PER REGISTER: `getSimpleRegInit` is a map OF maps, so the `#` line
    // opens each `RegType`'s block and every register of it follows (`dpc.cpp:733-745`).
    let mut open: Option<RegType> = None;
    for init in reg_init_order(reg_state) {
        if open != Some(init.file) {
            write_reg_file_header(out, core, name, corelet, init.file);
            open = Some(init.file);
        }
        write_reg_init(out, name, init);
    }
    out.text
        .push_str("===== END file: reg_initial.txt =====\n");
}

/// One instruction's line: its label, its opcode token, one ` | (value << bit)` term per operand, and
/// its comment (`dpc.cpp:657-722`).
fn write_instr<A: Arch>(
    out: &mut Senprog,
    isa: &Isa<A>,
    unit: Component,
    prog: &UnitProgram,
    instr: &Instruction,
    shift: &str,
) {
    let ty = isa.opcode_type(instr.opcode);
    if instr.has_tag() {
        out.text.push_str(&format!("///{}\n", instr.tag_str()));
    }
    // ⛔ `JCMPI` PRINTS AS `JCMP` — *"L3_JCMP with isimm=1 is called L3_JCMPI because field names
    // depend on isimm bit, but it's not a real separate instruction"* (`dpc.cpp:659-666`). The
    // SUBSTITUTION IS THE MNEMONIC'S ONLY: the field lookup below still uses `JCMPI`.
    let printed = match instr.opcode {
        InstOpCode::JCMPI => InstOpCode::JCMP,
        opcode => opcode,
    };
    out.text
        .push_str(&op_code_with_prefix(unit, printed));

    // ⭐ OPERAND ORDER IS `OperandT`'s, because `instFields_` is a `std::map` — see `Operand`'s note
    // on its load-bearing `Ord`. The island holds a `Vec`, so the order is restored here.
    let mut operands: Vec<&(OperandField, OperandValue)> = instr.fields.iter().collect();
    operands.sort_by_key(|(field, _)| *field);
    for (field, value) in operands {
        let slot = ty.and_then(|ty| isa.field_slot(ty, *field));
        let (Some(ty), Some((pos, bit))) = (ty, slot) else {
            out.refused.push(Refusal::NoSuchField(instr.opcode, *field));
            continue;
        };
        let text = match value {
            // ⭐ THE `$..$` IS THE WRITER'S, NOT `print`'s (`dpc.cpp:688`).
            OperandValue::Variable(_) => value.print(Pretty::Off).map(|held| format!("${held}$")),
            // `print(id, true)`, whose pretty is INERT for this kind (`dpc.cpp:690`).
            OperandValue::VariableSymbol(_) => value.print(Pretty::On),
            OperandValue::Descriptive(name) => {
                match isa.field_encoding(ty, pos, name) {
                    Some(encoded) => Some(encoded.get().to_string()),
                    None => {
                        out.refused.push(Refusal::UnknownEncoding(
                            instr.opcode,
                            *field,
                            name.clone(),
                        ));
                        None
                    }
                }
            }
            OperandValue::InstrTag(tag) => resolve_tag(out, prog, instr, *field, tag),
            // One arm for both, as `dpc.cpp:705-706` has it: `print(id)` bare, a `BOOLEAN` as `0`/`1`.
            OperandValue::Int(_) | OperandValue::Boolean(_) => value.print(Pretty::Off),
            OperandValue::Float(_) | OperandValue::Int128(_) | OperandValue::Unknown => {
                out.refused
                    .push(Refusal::UnprintableOperand(instr.opcode, *field));
                None
            }
        };
        if let Some(text) = text {
            out.text
                .push_str(&format!(" | ({text}{shift}{})", bit.get()));
        }
    }
    // ⛔ DEAD CODE PRINTS THE `  // ` EVEN WITH NO COMMENT — the two conditions share one separator
    // and only the marker is dead code's own (`dpc.cpp:716-719`).
    if instr.has_comment() || instr.dead {
        out.text.push_str(&format!("  // {}", instr.comment_str()));
    }
    if instr.dead {
        out.text.push_str(" ==Dead code==");
    }
    out.text.push('\n');
}

/// WHICH RESOLVER A TAGGED OPERAND USES — `src0` of a `JCMP`/`JCMPI` names a loop counter, every
/// other tagged operand names a line (`dpc.cpp:695-707`).
///
/// ⭐ `allowNotFound` IS THE INSTRUCTION'S OWN `deadCode_`, which is what makes each verdict's
/// fallback reachable: a live instruction refuses where a dead one takes the default.
fn resolve_tag(
    out: &mut Senprog,
    prog: &UnitProgram,
    instr: &Instruction,
    field: OperandField,
    tag: &str,
) -> Option<String> {
    let instrs = prog.simple_instr_vect()?;
    let is_loop = field == OperandField::Src0
        && matches!(instr.opcode, InstOpCode::JCMP | InstOpCode::JCMPI);
    if is_loop {
        match tag_to_lccr(instrs, tag) {
            TagLccr::At(index) => Some(index.get().to_string()),
            TagLccr::NotOnLoop(index) if instr.dead => Some(index.get().to_string()),
            TagLccr::AfterAlteringJump | TagLccr::NoLoop if instr.dead => Some("0".to_owned()),
            verdict => {
                out.refused.push(Refusal::UnresolvedLccr(verdict));
                None
            }
        }
    } else {
        match tag_to_pc(instrs, tag) {
            TagPc::At(pc) => Some(pc.get().to_string()),
            TagPc::NotFound if instr.dead => Some("0".to_owned()),
            verdict => {
                out.refused.push(Refusal::UnresolvedPc(verdict));
                None
            }
        }
    }
}

/// THE ORDER `getSimpleRegInit` WALKS — `map<RegType, map<unsigned, OperandAttr>>`, so by file in the
/// C++ enumerator order (which [`RegType`]'s derived `Ord` is) and then by index.
///
/// ⛔ A REPEATED `(file, index)` KEEPS THE LAST: `regInfo[regType][regNum] = regContent` overwrites
/// (`progir.h:494-496`), so the map the reference walks holds one entry where this `Vec` holds two.
fn reg_init_order(state: &UnitRegState) -> Vec<&RegInit> {
    let mut inits: Vec<&RegInit> = state.iter().collect();
    inits.sort_by_key(|init| (init.file, init.index.get()));
    inits.dedup_by(|later, earlier| {
        let same = (later.file, later.index) == (earlier.file, earlier.index);
        if same {
            *earlier = *later;
        }
        same
    });
    inits
}

/// One file's `#` line and its `UNIT-FILE:core:corelet` header (`dpc.cpp:733-745`).
fn write_reg_file_header(
    out: &mut Senprog,
    core: CoreId,
    name: UnitName,
    corelet: ProgramCorelet,
    file: RegType,
) {
    out.text.push_str("#\n");
    out.text.push_str(&str_toupper(name.spelling()));
    // ⛔ THE LRF NAMES NO FILE (`dpc.cpp:736-739`); its absence of a suffix is how a reader tells it.
    if file != RegType::Lrf {
        match reg_type_to_string(file) {
            Some(spelling) => out.text.push_str(&format!("-{spelling}")),
            None => out.refused.push(Refusal::UnnamedRegFile(file)),
        }
    }
    out.text.push_str(&format!(":{}", core.0));
    // ⛔ THE L3 HEADER OMITS THE CORELET FIELD ENTIRELY, though the program banner printed `0` for it
    // — one unit, two answers (`dpc.cpp:741-743`).
    if !matches!(name, UnitName::L3lu | UnitName::L3su) {
        out.text.push_str(&format!(":{}", corelet.spelling()));
    }
    out.text.push('\n');
}

/// One register's line — its index, its format prefix and its value (`dpc.cpp:746-771`).
fn write_reg_init(out: &mut Senprog, name: UnitName, init: &RegInit) {
    out.text.push_str(&format!("{:010x} ", init.index.get()));
    if let Some(format) = init.sen_data_type {
        out.text.push_str(&format!("datatype:{} ", format.spelling()));
    }
    // ⭐ THE VALUE TEXT IS e033's `print` — this arm only chooses the flag and the affixes, which is
    // exactly what `dpc.cpp:753-770` does around the same five calls.
    let text = match &init.value {
        // ⛔ EIGHT COPIES ON A PE/SFP, ONE `$..$.000000` ELSEWHERE, and both read `asString`
        // (`dpc.cpp:753-761`).
        OperandValue::Variable(held) => Some(match name {
            UnitName::Pe | UnitName::Sfp => format!("@{held}@").repeat(8),
            _ => format!("${held}$.000000"),
        }),
        OperandValue::VariableSymbol(_) => init.value.print(Pretty::On),
        // ⭐ AN INTEGER GETS A `.000000` TAIL AND A FLOAT DOES NOT GET A SECOND ONE.
        OperandValue::Int(_) => init
            .value
            .print(Pretty::Off)
            .map(|held| format!("{held}.000000")),
        // ⛔ AND THE INT128 IS FOUR WORDS BARE, HIGH WORD FIRST, WITH NO `0x`: the `0x` is
        // `prettyPrint`'s and this site does not pass it (`dpc.cpp:766`).
        OperandValue::Float(_) | OperandValue::Int128(_) => init.value.print(Pretty::Off),
        OperandValue::Boolean(_)
        | OperandValue::Descriptive(_)
        | OperandValue::InstrTag(_)
        | OperandValue::Unknown => {
            out.refused.push(Refusal::UnprintableRegInit(init.file));
            None
        }
    };
    if let Some(text) = text {
        out.text.push_str(&text);
    }
    out.text.push('\n');
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::arch::Dd2;
    use crate::bridges::progir_to_senprog::test_fixtures::{instr, program};
    use crate::generated::DataType;
    use crate::islands::progir::Block;
    use crate::islands::sentient::dialects::sentient::RegIndex;

    /// e010: only `BINARY` refuses, and only a program that still holds a variable.
    #[test]
    fn only_binary_refuses_a_program_with_variables() {
        let with_var = vec![(
            CoreId(0),
            program(vec![Block::Code(vec![instr(vec![(
                OperandField::Be,
                OperandValue::Variable("v".to_owned()),
            )])])]),
        )];
        assert!(!check_prog_format_compatibility(
            &with_var,
            ProgFormat::Binary
        ));
        assert!(check_prog_format_compatibility(
            &with_var,
            ProgFormat::Senprog
        ));

        let resolved = vec![(
            CoreId(0),
            program(vec![Block::Code(vec![instr(vec![(
                OperandField::Be,
                OperandValue::Int(3),
            )])])]),
        )];
        assert!(check_prog_format_compatibility(
            &resolved,
            ProgFormat::Binary
        ));
    }

    /// e028: `::toupper` is the C locale's, byte by byte — a non-ASCII byte is left exactly as it is.
    #[test]
    fn the_header_name_is_upper_cased_byte_by_byte() {
        assert_eq!(str_toupper("pt_row3"), "PT_ROW3");
        assert_eq!(str_toupper("pt_ré"), "PT_Ré");
    }

    /// e032: one PT-row unit's whole contribution — the banner derived from the corelet-resolved key,
    /// the `///` label, the operand terms in `OperandT` order rather than the island's, and the
    /// register block grouped per file with the LRF unsuffixed.
    ///
    /// `MVLOOPCNT` is the PT's type 10, whose `be` is `Enc::Lit("be", 1)` at bit 31, `dyn_loop` is
    /// `yes`/`no` at 26 and `imm` is a plain immediate at 6 (`isa.cpp:259-262`).
    #[test]
    fn one_unit_writes_its_banner_its_terms_in_operand_order_and_its_register_block() {
        let unit_program = UnitProgram {
            blocks: vec![Block::Code(vec![Instruction {
                opcode: InstOpCode::MVLOOPCNT,
                tag: Some("loop".to_owned()),
                // ⛔ DELIBERATELY OUT OF ORDER: the `std::map` order is restored by the writer.
                fields: vec![
                    (OperandField::Imm, OperandValue::Int(5)),
                    (OperandField::Be, OperandValue::Descriptive("be".to_owned())),
                    (
                        OperandField::DynLoop,
                        OperandValue::Descriptive("no".to_owned()),
                    ),
                ],
                ..instr(Vec::new())
            }])],
        };
        // ⛔ ALSO OUT OF ORDER: `RegType`'s own `Ord` puts the LRF block first.
        let reg_state = vec![
            RegInit {
                file: RegType::Jcr,
                index: RegIndex::at::<0>(),
                value: OperandValue::Float(1.5),
                sen_data_type: Some(DataType::Senint8),
            },
            RegInit {
                file: RegType::Lrf,
                index: RegIndex::at::<2>(),
                value: OperandValue::Int(7),
                sen_data_type: None,
            },
        ];
        let entry = UnitEntry {
            unit: SenComponent::Ptrow3_1,
            program: &unit_program,
            reg_state: Some(&reg_state),
        };
        let got = convert_ir_to_senprog::<Dd2>(&[(CoreId(1), &[entry])]);

        assert!(got.refused.is_empty());
        assert_eq!(
            got.text,
            concat!(
                "===== START file: prog.txt =====\n",
                "========== Core: 1 Corelet: 1 Unit: pt_row3 Program START ============\n",
                "///loop\n",
                "PTOP_MVLOOPCNT | (1 << 31) | (0 << 26) | (5 << 6)\n",
                "========== Core: 1 Corelet: 1 Unit: pt_row3 Program END ============\n",
                "===== END file: prog.txt =====\n",
                "===== START file: reg_initial.txt =====\n",
                "#\n",
                "PT_ROW3:1:1\n",
                "0000000002 7.000000\n",
                "#\n",
                "PT_ROW3-JCR:1:1\n",
                "0000000000 datatype:SENINT8 1.500000\n",
                "===== END file: reg_initial.txt =====\n",
            )
        );
    }
}
