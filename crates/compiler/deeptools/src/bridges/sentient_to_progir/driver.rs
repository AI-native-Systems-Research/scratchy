//! THE D76 PASS ITSELF — `runOnOperation`, `GenerateProgIR`, `GenerateProgIRForProgramUnit`,
//! the register initialisation, the program-length equalisation and the `init.smc` collapse.
//!
//! ⭐ ONE `ProgramUnitOp` SPANS THE SET: handles = cores x corelets x num_folds.
//! ⛔ THE PROGRAM BOUNDS ARE READ, NOT RESTATED — `max_ibuff_entries(unit)` per component and
//! per arch, never `kMaxCompIBuff` as if it were any unit's limit.
//!
//! 7 units. Every citation resolves against the authority tree
//! `/Users/nickm/git/deeptools-src` at revision `a0d29abbed`.
//!
//! | unit | level | LoC | authority |
//! |---|---|---|---|
//! | `e015_initializeUtilizedRegisters` | 0 | 39 | `dcc/src/Conversion/SentientToProgIR/SentientToProgIR.cpp:429` |
//! | `e016_replaceProgramBodyWithSmcOp` | 0 | 49 | `dcc/src/Conversion/SentientToProgIR/SentientToProgIR.cpp:560` |
//! | `e095_equalizeProgramLength` | 2 | 90 | `dcc/src/Conversion/SentientToProgIR/SentientToProgIR.cpp:469` |
//! | `e127_GenerateProgIR` | 5 | 65 | `dcc/src/Conversion/SentientToProgIR/SentientToProgIR.cpp:53` |
//! | `e128_GenerateProgIR` | 5 | 111 | `dcc/src/Conversion/SentientToProgIR/SentientToProgIR.cpp:119` |
//! | `e129_GenerateProgIRForProgramUnit` | 6 | 186 | `dcc/src/Conversion/SentientToProgIR/SentientToProgIR.cpp:234` |
//! | `e130_runOnOperation` | 7 | 135 | `dcc/src/Conversion/SentientToProgIR/SentientToProgIR.cpp:610` |

// ⛔ ONE `crustify:todo:` PER SCHEDULED UNIT. Replace each with the ported function
// carrying `/// Replaces: eNNN_name`. A surviving TODO is open work.

use crate::arch::Arch;
use crate::bridges::sentient_to_progir::construct::instr_tag;
use crate::bridges::sentient_to_progir::construct::scalar::{
    construct_jmp_instr, construct_nop_instr,
};
use crate::bridges::sentient_to_progir::state::RegsToInit;
use crate::islands::progir::dialects::{Op as ProgIrOp, init};
use crate::islands::progir::ty::{Operand, OperandValue};
use crate::islands::progir::{
    Block, Instruction, OperandField, Program, RegInit, UnitRegState, print,
};
use crate::islands::sentient;
use crate::model::Model;
use crate::units::Core;
use crate::workload::Workload;
use sys_arch_spec::regfile::Component;

/// Replaces: e015_initializeUtilizedRegisters
///
/// Give every register a unit uses a zero start, except those the header already initialises.
/// ⭐ THE TWO BRANCHES COLLAPSE: a component with no register state is an empty snapshot, which is
/// exactly what the reference's else-branch does with one.
/// ⛔ THE SNAPSHOT CANNOT GO STALE OBSERVABLY. `auto reg_init = ...getSimpleRegInit()` copies the
/// header (`progir.h:497` returns a reference; the `auto` drops it), and each `(file, index)` occurs
/// once in a `UtilizedRegisters`, so nothing this loop adds could have suppressed a later add.
/// The reference's `addRegInit` ASSIGNS (`progir.h:491-496`) where our island's state is a `Vec`.
pub fn initialize_utilized_registers<A: Arch, M: Model, W: Workload>(
    regs_to_init: &RegsToInit,
    progstateinfo: &mut Vec<(Core, Program<A, M, W>)>,
) {
    for (unit, reg_map) in regs_to_init {
        // ⛔ A UNIT WITH NO PROGRAM COMPONENT IS THE REFERENCE'S `DT_ERROR` (see
        // [`UnitKey::program_key`]), and `regs_to_init_` only ever holds the nine compute units.
        let Some((core, comp)) = unit.program_key() else {
            continue;
        };
        // ⭐ `progstateinfo_[core]` — `operator[]` on a `std::map`, so a core nothing has emitted for
        // gets an empty program rather than nothing at all.
        if !progstateinfo.iter().any(|(id, _)| *id == core) {
            progstateinfo.push((core, Program::default()));
        }
        for (_, program) in progstateinfo.iter_mut().filter(|(id, _)| *id == core) {
            let already: Vec<RegInit> = program
                .reg_state
                .iter()
                .find(|(unit, _)| *unit == comp)
                .map(|(_, state)| state.clone())
                .unwrap_or_default();
            if !program.reg_state.iter().any(|(unit, _)| *unit == comp) {
                program.reg_state.push((comp, UnitRegState::new()));
            }
            for (_, state) in program
                .reg_state
                .iter_mut()
                .filter(|(unit, _)| *unit == comp)
            {
                for (file, regs) in reg_map {
                    for index in regs {
                        if already
                            .iter()
                            .any(|init| init.file == *file && init.index == *index)
                        {
                            continue;
                        }
                        state.push(RegInit {
                            file: *file,
                            index: *index,
                            value: Operand::every(OperandValue::Int(0)),
                        });
                    }
                }
            }
        }
    }
}

/// Replaces: e016_replaceProgramBodyWithSmcOp
///
/// The module this pass leaves behind: the program's own function, its body dropped, holding
/// `init.smc` and a return.
/// ⛔ BOTH `signalPassFailure` PATHS ARE TYPE GUARDS HERE. One [`sentient::Program`] is one function,
/// so *"expects exactly one function in the module"* is unviolatable, and [`print::module`] emits
/// `func.func @name()` with no results, so there is never anything left to return.
/// ⛔ AND THE TWO NAMES ARE DIFFERENT FACTS: the function keeps the program's own name, while the op
/// carries `dccExtContext().prog_name_`.
#[must_use]
pub fn replace_program_body_with_smc_op<A: Arch, M: Model, W: Workload>(
    program: &sentient::Program<A, M, W>,
    prog_name: &str,
) -> String {
    print::module(
        &program.name.to_string(),
        &[ProgIrOp::Init(init::Op::Smc {
            name: prog_name.to_owned(),
        })],
    )
}
/// HOW MANY UNIFORMIZATION PADDING LABELS HAVE BEEN MINTED — `uniformization_padding_counter_`, a
/// member of the pass rather than of any one program.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PaddingCounter(pub u32);

impl PaddingCounter {
    /// `uniformization_padding_counter_++` — the number this padding takes, then the step.
    pub fn bump(&mut self) -> u32 {
        let now = self.0;
        self.0 += 1;
        now
    }
}

/// WHY A PROGRAM COULD NOT BE PADDED TO THE LONGEST ONE — the reference's two failure paths, returned
/// rather than asserted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProgramLengthOffender {
    /// *"The program contains no instruction."* (`:502`) — there is no RETURN to pad in front of.
    Empty {
        /// Which core.
        core: Core,
        /// Which unit of it.
        unit: Component,
    },
    /// *"curr_prog_length can't be larger than max_prog_length."* (`:557-558`) — this core's program
    /// is longer than the one declared longest.
    Longer {
        /// Which core.
        core: Core,
        /// Which unit of it.
        unit: Component,
        /// What it holds.
        instructions: usize,
        /// What the declared longest holds.
        longest: usize,
    },
}

/// `static_cast<ProgIrCodeBlock*>(senCompProgram_.at(unit).blocks.front())->instrVector` — one unit's
/// straight-line program. ⛔ THE REFERENCE'S CAST IS UNCHECKED: a first block that is not CODE would
/// be read as one, where this answers `None`.
fn first_code_block<A: Arch, M: Model, W: Workload>(
    program: &Program<A, M, W>,
    unit: Component,
) -> Option<&Vec<Instruction>> {
    match program
        .per_unit
        .iter()
        .find(|(at, _)| *at == unit)?
        .1
        .blocks
        .first()?
    {
        Block::Code(instrs) => Some(instrs),
        _ => None,
    }
}

/// The same, mutably.
fn first_code_block_mut<A: Arch, M: Model, W: Workload>(
    program: &mut Program<A, M, W>,
    unit: Component,
) -> Option<&mut Vec<Instruction>> {
    match program
        .per_unit
        .iter_mut()
        .find(|(at, _)| *at == unit)?
        .1
        .blocks
        .first_mut()?
    {
        Block::Code(instrs) => Some(instrs),
        _ => None,
    }
}

/// Replaces: e095_equalizeProgramLength
///
/// Pads every core's program for one unit out to the longest core's: one short, a NOP before the
/// RETURN; further short, a tagged jump over a dead-code copy of the longest program's tail.
///
/// ⛔ THE REFERENCE'S OWN `curr > max` PATH IS DEAD CODE — both lengths are `size_t`, so the
/// difference WRAPS and `> 1` takes it, padding a program already too long. Returned here instead.
/// ⛔ AND AN EMPTY PROGRAM IS NOT PADDED, where the reference's `back()` on one is undefined.
pub fn equalize_program_length<A: Arch, M: Model, W: Workload>(
    max_length_unit_core_map: &[(Component, Core)],
    progstateinfo: &mut [(Core, Program<A, M, W>)],
    padding_counter: &mut PaddingCounter,
) -> Vec<ProgramLengthOffender> {
    let mut offenders = Vec::new();
    // iterate through all unit types
    for (sen_comp, max_core_id) in max_length_unit_core_map.iter().copied() {
        let Some(max_instr_list) = progstateinfo
            .iter()
            .find(|(core, _)| *core == max_core_id)
            .and_then(|(_, program)| first_code_block(program, sen_comp))
            .cloned()
        else {
            continue;
        };
        let max_prog_length = max_instr_list.len();
        // iterate through psInfo for all cores
        for (core, program) in progstateinfo.iter_mut() {
            let core = *core;
            // skip padding if current core doesn't has this unit type
            let Some(curr_instr_vector) = first_code_block_mut(program, sen_comp) else {
                continue;
            };
            let curr_prog_length = curr_instr_vector.len();
            if curr_prog_length == 0 {
                offenders.push(ProgramLengthOffender::Empty {
                    core,
                    unit: sen_comp,
                });
                continue;
            }
            if curr_prog_length > max_prog_length {
                offenders.push(ProgramLengthOffender::Longer {
                    core,
                    unit: sen_comp,
                    instructions: curr_prog_length,
                    longest: max_prog_length,
                });
                continue;
            }
            // only pad instructions when the length of current program is shorter.
            let before_return = curr_prog_length.saturating_sub(1);
            match max_prog_length - curr_prog_length {
                0 => continue,
                // if diff is one, insert NOP before RETURN
                1 => {
                    if let Some(nop) = construct_nop_instr(Some("NOP for uniformization")).regular()
                    {
                        curr_instr_vector.insert(before_return, nop.instr());
                    }
                }
                _ => {
                    let padding_counter = padding_counter.bump();
                    // use RETURN's tag if existing.
                    let jcmp_tag = match &curr_instr_vector[before_return].tag {
                        Some(tag) => tag.clone(),
                        None => format!("uniformization_padding_{padding_counter}"),
                    };
                    curr_instr_vector[before_return].tag = Some(jcmp_tag.clone());
                    let jcmp = construct_jmp_instr(&jcmp_tag)
                        .with_common_comment("jump for uniformization");
                    if let Some(jcmp) = jcmp.regular() {
                        curr_instr_vector.insert(before_return, jcmp.instr());
                    }
                    let mut pc_targets_in_dead_code: Vec<String> = Vec::new();
                    for donor in max_instr_list
                        .iter()
                        .take(max_prog_length.saturating_sub(1))
                        .skip(curr_prog_length)
                    {
                        let mut instr = donor.clone();
                        instr.comment = Some("padding for uniformization".to_owned());
                        instr.dead = true;
                        // sanitize pc_targets: modify them to avoid collision with the valid code
                        if let Some((_, operand)) = instr
                            .fields
                            .iter_mut()
                            .find(|(field, _)| *field == OperandField::PcTarget)
                            && let Some(target) = operand.as_string(None).map(str::to_owned)
                        {
                            pc_targets_in_dead_code.push(target.clone());
                            *operand = instr_tag(&format!("{target}_uniformization"));
                        }
                        // sanitize tags: keep one only where the dead code still jumps to it.
                        if let Some(tag) = instr.tag.take() {
                            instr.tag = pc_targets_in_dead_code
                                .contains(&tag)
                                .then(|| format!("{tag}_uniformization"));
                        }
                        let end = curr_instr_vector.len().saturating_sub(1);
                        curr_instr_vector.insert(end, instr);
                    }
                }
            }
        }
    }
    offenders
}
// crustify:todo: e127_GenerateProgIR
// crustify:todo: e128_GenerateProgIR
// crustify:todo: e129_GenerateProgIRForProgramUnit
// crustify:todo: e130_runOnOperation

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::arch::Target;
    use crate::bridges::sentient_to_progir::construct::descriptive;
    use crate::bridges::sentient_to_progir::state::{UnitKey, UtilizedRegisters};
    use crate::generated::OpFunc;
    use crate::islands::dataflow_ir::dialects::Val;
    use crate::islands::dataflow_ir::{GroupId, OpIndex, ProgramName, Units};
    use crate::islands::progir::ty::RegType;
    use crate::islands::progir::{OpCode, UnitProgram};
    use crate::islands::sentient::dialects::sentient::RegIndex;
    use crate::islands::sentient::{ProgramUnit, ProgramUnits};
    use crate::units::{DfirUnit, Residency};
    use std::collections::BTreeSet;
    use sys_arch_spec::regfile::Component;

    struct M;
    impl Model for M {
        const QUERY_HEADS: u32 = 32;
        const KV_HEADS: u32 = 8;
        const HEAD_DIM: u32 = 64;
        const HIDDEN: u32 = 2048;
        const LAYERS: u32 = 40;
        const FFN: u32 = 8192;
        const VOCAB: u32 = 49152;
    }

    struct W;
    impl Workload for W {
        const ROWS: u32 = 1;
        const ACTIVE_CAP: u32 = 128;
    }

    #[test]
    fn only_a_register_the_header_does_not_already_initialise_is_zeroed() {
        let core = Core::checked(0).expect("core 0");
        let key = |unit| UnitKey::of(unit, Residency::CoreWide { core }).expect("core 0 keys");
        let mut progstateinfo: Vec<(Core, Program<Target, M, W>)> = Vec::new();
        // The header already starts `lrf1`; `lrf1` and `lar2` are both used.
        progstateinfo.push((core, Program::default()));
        progstateinfo[0].1.reg_state.push((
            Component::Pe,
            vec![RegInit {
                file: RegType::Lrf,
                index: RegIndex::at::<1>(),
                value: Operand::every(OperandValue::Int(7)),
            }],
        ));
        let used = |file, index| UtilizedRegisters::from([(file, BTreeSet::from([index]))]);
        let mut regs_to_init = RegsToInit::new();
        regs_to_init.insert(
            key(DfirUnit::Pe),
            UtilizedRegisters::from([
                (RegType::Lrf, BTreeSet::from([RegIndex::at::<1>()])),
                (RegType::Lar, BTreeSet::from([RegIndex::at::<2>()])),
            ]),
        );
        initialize_utilized_registers(&regs_to_init, &mut progstateinfo);
        assert_eq!(
            progstateinfo[0].1.reg_state,
            vec![(
                Component::Pe,
                vec![
                    RegInit {
                        file: RegType::Lrf,
                        index: RegIndex::at::<1>(),
                        value: Operand::every(OperandValue::Int(7)),
                    },
                    RegInit {
                        file: RegType::Lar,
                        index: RegIndex::at::<2>(),
                        value: Operand::every(OperandValue::Int(0)),
                    },
                ],
            )]
        );
        // ⛔ A core with no program at all still gets one, and everything it uses is zeroed.
        let other = Core::checked(1).expect("core 1");
        let elsewhere =
            UnitKey::of(DfirUnit::Sfp, Residency::CoreWide { core: other }).expect("core 1 keys");
        initialize_utilized_registers(
            &RegsToInit::from([(elsewhere, used(RegType::Lrf, RegIndex::at::<0>()))]),
            &mut progstateinfo,
        );
        assert_eq!(
            progstateinfo[1].1.reg_state,
            vec![(
                Component::Sfp,
                vec![RegInit {
                    file: RegType::Lrf,
                    index: RegIndex::at::<0>(),
                    value: Operand::every(OperandValue::Int(0)),
                }],
            )]
        );
    }

    #[test]
    fn the_emptied_function_keeps_its_own_name_and_holds_only_the_smc_reference() {
        let program: sentient::Program<Target, M, W> = sentient::Program {
            name: ProgramName {
                group: GroupId(0),
                index: OpIndex(0),
                func: OpFunc::Add,
            },
            preamble: Vec::new(),
            units: ProgramUnits::of(
                ProgramUnit {
                    on: Units::one(DfirUnit::Pe, Val(0)),
                    precision: None,
                    body: Vec::new(),
                    arch: core::marker::PhantomData,
                },
                Vec::new(),
            ),
            bound: core::marker::PhantomData,
        };
        assert_eq!(
            replace_program_body_with_smc_op(&program, "default_prog_name"),
            concat!(
                "module {\n",
                "  func.func @g0_0_add() attributes {grid = [1]} {\n",
                "    init.smc {name = \"default_prog_name\"}\n",
                "    return\n",
                "  }\n",
                "}\n"
            )
        );
    }
    /// e095: three cores of one unit padded to the longest — a NOP where one short, and a tagged jump
    /// over the donor's sanitized tail where further short. A longer program is an offender.
    #[test]
    fn a_shorter_program_is_padded_to_the_longest_core() {
        let at = |core: u32| Core::checked(core).expect("every arch has 32 cores");
        let ret = Instruction {
            opcode: OpCode::RETURN,
            symbolic_opcode: None,
            fields: Vec::new(),
            dead: false,
            tag: None,
            comment: None,
        };
        let mut jump = ret.clone();
        jump.opcode = OpCode::JCMP;
        jump.fields = vec![(OperandField::PcTarget, instr_tag("loop"))];
        let mut labelled = ret.clone();
        labelled.opcode = OpCode::NOP;
        labelled.tag = Some("loop".to_owned());
        let core = |core: u32, instrs: Vec<Instruction>| {
            (
                at(core),
                Program::<Target, M, W> {
                    per_unit: vec![(
                        Component::Pt,
                        UnitProgram {
                            blocks: vec![Block::Code(instrs)],
                        },
                    )],
                    ..Program::default()
                },
            )
        };
        let donor = vec![labelled.clone(), jump, labelled, ret.clone()];
        let mut progstateinfo = vec![
            core(0, donor.clone()),
            core(1, vec![ret.clone()]),
            core(2, vec![ret.clone(), ret.clone(), ret.clone()]),
            core(
                3,
                vec![ret.clone(), ret.clone(), ret.clone(), ret.clone(), ret],
            ),
        ];
        let mut counter = PaddingCounter::default();
        let offenders =
            equalize_program_length(&[(Component::Pt, at(0))], &mut progstateinfo, &mut counter);
        assert_eq!(
            offenders,
            vec![ProgramLengthOffender::Longer {
                core: at(3),
                unit: Component::Pt,
                instructions: 5,
                longest: 4,
            }]
        );
        assert_eq!(counter, PaddingCounter(1), "only core 1 minted a label");

        let padded =
            first_code_block(&progstateinfo[1].1, Component::Pt).expect("core 1's program");
        assert_eq!(padded.len(), 4);
        assert_eq!(
            padded.iter().map(|instr| instr.opcode).collect::<Vec<_>>(),
            vec![OpCode::JCMP, OpCode::JCMP, OpCode::NOP, OpCode::RETURN]
        );
        assert_eq!(
            padded[0].comment.as_deref(),
            Some("jump for uniformization")
        );
        assert_eq!(
            padded[0].fields,
            vec![
                (OperandField::Mode, descriptive("always")),
                (
                    OperandField::PcTarget,
                    instr_tag("uniformization_padding_0")
                ),
            ]
        );
        assert_eq!(
            padded[3].tag.as_deref(),
            Some("uniformization_padding_0"),
            "the RETURN is what the jump lands on"
        );
        // The donor's own tail: a pc_target and a tag both get the suffix, and both are dead.
        assert!(padded[1].dead && padded[2].dead);
        assert_eq!(
            padded[1].fields,
            vec![(OperandField::PcTarget, instr_tag("loop_uniformization"))]
        );
        assert_eq!(padded[2].tag.as_deref(), Some("loop_uniformization"));

        // One short: a NOP before the RETURN, and nothing else touched.
        let nopped =
            first_code_block(&progstateinfo[2].1, Component::Pt).expect("core 2's program");
        assert_eq!(
            nopped.iter().map(|instr| instr.opcode).collect::<Vec<_>>(),
            vec![OpCode::RETURN, OpCode::RETURN, OpCode::NOP, OpCode::RETURN]
        );
        assert_eq!(nopped[2].comment.as_deref(), Some("NOP for uniformization"));
        assert_eq!(
            first_code_block(&progstateinfo[0].1, Component::Pt),
            Some(&donor),
            "the longest core is left alone"
        );
    }
}
