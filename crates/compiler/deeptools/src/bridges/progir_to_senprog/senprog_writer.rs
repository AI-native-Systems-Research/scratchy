// SPDX-License-Identifier: Apache-2.0
//! Ported from `sys-arch-spec/dpc/dpc.cpp` — 3 unit(s) of bridge 4, `ProgIR -> SenProg`.
//!
//! ⛔ SENPROG IS A PRINT FORMAT, so what is being ported is the exact TEXT: field order,
//! spelling and separators. A predicate that decides what to write, without writing it, is not
//! a port. The authority is `/Users/nickm/git/deeptools-src/<file>:<line>` per unit below.

use sys_arch_spec::CoreId;

use super::progir_print::has_variables_in_instr;
use crate::arch::Arch;
use crate::islands::progir::Program;
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

// crustify:todo: e028_strToupper
//   authority: sys-arch-spec/dpc/dpc.cpp:50  (4 lines)  `strToupper`

// crustify:todo: e032_convertIr2Senprog
//   authority: sys-arch-spec/dpc/dpc.cpp:615  (163 lines)  `Dpc::convertIr2Senprog`

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::bridges::progir_to_senprog::test_fixtures::{instr, program};
    use crate::islands::progir::ty::OperandValue;
    use crate::islands::progir::{Block, OperandField};

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
}
