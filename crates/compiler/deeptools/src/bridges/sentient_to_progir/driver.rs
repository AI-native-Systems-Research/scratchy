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
use crate::islands::progir::dialects::{Op as ProgIrOp, init};
use crate::islands::progir::ty::{OperandValue, RegType};
use crate::islands::progir::{Program, RegBits, RegInit, UnitRegState, print};
use crate::islands::sentient;
use crate::model::Model;
use crate::units::Core;
use crate::workload::Workload;
use sys_arch_spec::regfile::Component;

/// WHICH REGISTERS ONE UNIT WAS SEEN TO USE — `regs_to_init_`'s value, a
/// `std::map<RegType, std::set<int>>` (`SentientToProgIR.h:118`).
pub type UtilizedRegisters = Vec<(RegType, RegBits)>;

/// Replaces: e015_initializeUtilizedRegisters
///
/// Give every register a unit uses a zero start, except those the header already initialises.
/// ⛔ THE SNAPSHOT IS TAKEN BEFORE ANY ADD — `getSimpleRegInit()` returns BY VALUE, so a register
/// this loop adds does not suppress a later add of the same one, and a duplicate init is the
/// reference's own output.
/// ⭐ THE TWO BRANCHES COLLAPSE: a component with no register state is an empty snapshot, which is
/// exactly what the reference's else-branch does with one.
pub fn initialize_utilized_registers<A: Arch, M: Model, W: Workload>(
    regs_to_init: &[((Core, Component), UtilizedRegisters)],
    progstateinfo: &mut Vec<(Core, Program<A, M, W>)>,
) {
    for ((core, comp), reg_map) in regs_to_init {
        // ⭐ `progstateinfo_[core]` — `operator[]` on a `std::map`, so a core nothing has emitted for
        // gets an empty program rather than nothing at all.
        if !progstateinfo.iter().any(|(id, _)| id == core) {
            progstateinfo.push((*core, Program::default()));
        }
        for (_, program) in progstateinfo.iter_mut().filter(|(id, _)| id == core) {
            let already: Vec<RegInit> = program
                .reg_state
                .iter()
                .find(|(unit, _)| unit == comp)
                .map(|(_, state)| state.clone())
                .unwrap_or_default();
            if !program.reg_state.iter().any(|(unit, _)| unit == comp) {
                program.reg_state.push((*comp, UnitRegState::new()));
            }
            for (_, state) in program
                .reg_state
                .iter_mut()
                .filter(|(unit, _)| unit == comp)
            {
                for (file, regs) in reg_map {
                    for index in regs.iter() {
                        if already
                            .iter()
                            .any(|init| init.file == *file && init.index == index)
                        {
                            continue;
                        }
                        state.push(RegInit {
                            file: *file,
                            index,
                            value: OperandValue::Int(0),
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
// crustify:todo: e095_equalizeProgramLength
// crustify:todo: e127_GenerateProgIR
// crustify:todo: e128_GenerateProgIR
// crustify:todo: e129_GenerateProgIRForProgramUnit
// crustify:todo: e130_runOnOperation

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::arch::Target;
    use crate::generated::OpFunc;
    use crate::islands::dataflow_ir::dialects::Val;
    use crate::islands::dataflow_ir::{GroupId, OpIndex, ProgramName, Units};
    use crate::islands::sentient::dialects::sentient::RegIndex;
    use crate::islands::sentient::{ProgramUnit, ProgramUnits};
    use crate::units::DfirUnit;

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
        let mut progstateinfo: Vec<(Core, Program<Target, M, W>)> = Vec::new();
        // The header already starts `lrf1`; `lrf1` and `lar2` are both used.
        progstateinfo.push((core, Program::default()));
        progstateinfo[0].1.reg_state.push((
            Component::Pe,
            vec![RegInit {
                file: RegType::Lrf,
                index: RegIndex::at::<1>(),
                value: OperandValue::Int(7),
            }],
        ));
        let regs_to_init = vec![(
            (core, Component::Pe),
            vec![
                (RegType::Lrf, RegBits::empty().with(RegIndex::at::<1>())),
                (RegType::Lar, RegBits::empty().with(RegIndex::at::<2>())),
            ],
        )];
        initialize_utilized_registers(&regs_to_init, &mut progstateinfo);
        assert_eq!(
            progstateinfo[0].1.reg_state,
            vec![(
                Component::Pe,
                vec![
                    RegInit {
                        file: RegType::Lrf,
                        index: RegIndex::at::<1>(),
                        value: OperandValue::Int(7),
                    },
                    RegInit {
                        file: RegType::Lar,
                        index: RegIndex::at::<2>(),
                        value: OperandValue::Int(0),
                    },
                ],
            )]
        );
        // ⛔ A core with no program at all still gets one, and everything it uses is zeroed.
        let other = Core::checked(1).expect("core 1");
        initialize_utilized_registers(
            &[(
                (other, Component::Sfp),
                vec![(RegType::Lrf, RegBits::empty().with(RegIndex::at::<0>()))],
            )],
            &mut progstateinfo,
        );
        assert_eq!(
            progstateinfo[1].1.reg_state,
            vec![(
                Component::Sfp,
                vec![RegInit {
                    file: RegType::Lrf,
                    index: RegIndex::at::<0>(),
                    value: OperandValue::Int(0),
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
}
