// SPDX-License-Identifier: Apache-2.0
//! Ported from `sys-arch-spec/progir/progir.cpp` — 6 unit(s) of bridge 4, `ProgIR -> SenProg`.
//!
//! ⛔ SENPROG IS A PRINT FORMAT, so what is being ported is the exact TEXT: field order,
//! spelling and separators. A predicate that decides what to write, without writing it, is not
//! a port. The authority is `/Users/nickm/git/deeptools-src/<file>:<line>` per unit below.

use sys_arch_spec::arch_enums::SenComponent;

use crate::arch::Arch;
use crate::islands::progir::ty::OperandValue;
use crate::islands::progir::{Block, Program};
use crate::model::Model;
use crate::workload::Workload;

/// Replaces: e009_hasVariablesInInstr
///
/// Any variable operand anywhere, or any `CONDITION` block at all — whose expression is never read.
///
/// ⛔ RECURSES BECAUSE THE REFERENCE'S `blocks` IS FLAT (`progir.h:419`): a loop body is further
/// entries of that same vector, which one pass already sees, while this island nests it.
/// ⛔ A LOOP'S OWN `start`/`end` DO NOT COUNT — `FORLOOP` matches neither arm of the if/else-if.
#[must_use]
pub fn has_variables_in_instr<A: Arch, M: Model, W: Workload>(program: &Program<A, M, W>) -> bool {
    program
        .per_unit
        .iter()
        .any(|(_, unit)| unit.blocks.iter().any(block_has_variables))
}

/// One entry of the reference's flat `blocks` vector, plus everything this island nests inside it.
fn block_has_variables(block: &Block) -> bool {
    match block {
        Block::Condition { .. } => true,
        Block::Code(instrs) => instrs
            .iter()
            .flat_map(|instr| &instr.fields)
            .any(|(_, value)| matches!(value, OperandValue::Variable(_))),
        Block::ForLoop { body, .. } => body.iter().any(block_has_variables),
        // Neither is `CODE` or `CONDITION`, so neither arm of the reference's if/else-if takes them.
        Block::RegInit(_) | Block::VarDef(_) => false,
    }
}

/// WHICH COMPUTE HALF — the corelet `senCompMap` is keyed by, and the reference's `-1`.
///
/// ⛔ THE ARGUMENT IS A SINGLE CHARACTER AT THE CALLSITE: `corelet = unitName.back()`
/// (`dpc.cpp:638-639`), so a third corelet would be mislabelled rather than refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Corelet {
    /// `0`.
    C0,
    /// `1`.
    C1,
    /// `-1` — `getSenComponent`'s default (`progir.h:547`): the unit as a whole.
    Generic,
}

/// WHICH ROW OF THE PT — the eight `senCompMap` keys `pt_row0`..`pt_row7`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PtRow {
    /// `pt_row0`.
    Row0,
    /// `pt_row1`.
    Row1,
    /// `pt_row2`.
    Row2,
    /// `pt_row3`.
    Row3,
    /// `pt_row4`.
    Row4,
    /// `pt_row5`.
    Row5,
    /// `pt_row6`.
    Row6,
    /// `pt_row7`.
    Row7,
}

/// A UNIT AS `senCompMap` SPELLS IT — the first half of that table's key.
///
/// ⛔ A ROW IS ITS OWN NAME, not a PT with an index: `pt_row3` and `pt` are separate keys and the
/// table treats them differently.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnitName {
    /// `sfp`.
    Sfp,
    /// `pt`.
    Pt,
    /// `pt_row<N>`.
    PtRow(PtRow),
    /// `pe`.
    Pe,
    /// `lxlu`.
    Lxlu,
    /// `lxsu`.
    Lxsu,
    /// `l0lu`.
    L0lu,
    /// `l0su`.
    L0su,
    /// `l3lu`.
    L3lu,
    /// `l3su`.
    L3su,
}

impl UnitName {
    /// The key as `senCompMap` writes it.
    #[must_use]
    pub const fn spelling(self) -> &'static str {
        match self {
            Self::Sfp => "sfp",
            Self::Pt => "pt",
            Self::PtRow(PtRow::Row0) => "pt_row0",
            Self::PtRow(PtRow::Row1) => "pt_row1",
            Self::PtRow(PtRow::Row2) => "pt_row2",
            Self::PtRow(PtRow::Row3) => "pt_row3",
            Self::PtRow(PtRow::Row4) => "pt_row4",
            Self::PtRow(PtRow::Row5) => "pt_row5",
            Self::PtRow(PtRow::Row6) => "pt_row6",
            Self::PtRow(PtRow::Row7) => "pt_row7",
            Self::Pe => "pe",
            Self::Lxlu => "lxlu",
            Self::Lxsu => "lxsu",
            Self::L0lu => "l0lu",
            Self::L0su => "l0su",
            Self::L3lu => "l3lu",
            Self::L3su => "l3su",
        }
    }
}

/// Replaces: e014_getSenComponent
///
/// The `senCompMap` lookup, whose `DT_ERROR("Undefined type of unit: ")` is `None` here.
///
/// ⛔ THE TABLE HAS THREE HOLES (`progir.cpp:619-668`): `("pt", 0)`, `("pt", 1)` and
/// `("l3lu"/"l3su", 1)`. A PT with a corelet is named per ROW, and the L3 halves have only one
/// corelet — for them `0` and `-1` are the same entry.
#[must_use]
pub const fn sen_component(unit: UnitName, corelet: Corelet) -> Option<SenComponent> {
    Some(match (unit, corelet) {
        (UnitName::Sfp, Corelet::C0) => SenComponent::Sfp0,
        (UnitName::Sfp, Corelet::C1) => SenComponent::Sfp1,
        (UnitName::Sfp, Corelet::Generic) => SenComponent::Sfp,
        // ⛔ A ROW WITH NO CORELET COLLAPSES TO THE WHOLE PT, exactly as bare `("pt", -1)` does.
        (UnitName::Pt | UnitName::PtRow(_), Corelet::Generic) => SenComponent::Pt,
        (UnitName::PtRow(row), Corelet::C0) => match row {
            PtRow::Row0 => SenComponent::Ptrow0_0,
            PtRow::Row1 => SenComponent::Ptrow1_0,
            PtRow::Row2 => SenComponent::Ptrow2_0,
            PtRow::Row3 => SenComponent::Ptrow3_0,
            PtRow::Row4 => SenComponent::Ptrow4_0,
            PtRow::Row5 => SenComponent::Ptrow5_0,
            PtRow::Row6 => SenComponent::Ptrow6_0,
            PtRow::Row7 => SenComponent::Ptrow7_0,
        },
        (UnitName::PtRow(row), Corelet::C1) => match row {
            PtRow::Row0 => SenComponent::Ptrow0_1,
            PtRow::Row1 => SenComponent::Ptrow1_1,
            PtRow::Row2 => SenComponent::Ptrow2_1,
            PtRow::Row3 => SenComponent::Ptrow3_1,
            PtRow::Row4 => SenComponent::Ptrow4_1,
            PtRow::Row5 => SenComponent::Ptrow5_1,
            PtRow::Row6 => SenComponent::Ptrow6_1,
            PtRow::Row7 => SenComponent::Ptrow7_1,
        },
        (UnitName::Pe, Corelet::C0) => SenComponent::Pe0,
        (UnitName::Pe, Corelet::C1) => SenComponent::Pe1,
        (UnitName::Pe, Corelet::Generic) => SenComponent::Pe,
        (UnitName::Lxlu, Corelet::C0) => SenComponent::Lxlu0,
        (UnitName::Lxlu, Corelet::C1) => SenComponent::Lxlu1,
        (UnitName::Lxlu, Corelet::Generic) => SenComponent::Lxlu,
        (UnitName::Lxsu, Corelet::C0) => SenComponent::Lxsu0,
        (UnitName::Lxsu, Corelet::C1) => SenComponent::Lxsu1,
        (UnitName::Lxsu, Corelet::Generic) => SenComponent::Lxsu,
        (UnitName::L0lu, Corelet::C0) => SenComponent::L0lu0,
        (UnitName::L0lu, Corelet::C1) => SenComponent::L0lu1,
        (UnitName::L0lu, Corelet::Generic) => SenComponent::L0lu,
        (UnitName::L0su, Corelet::C0) => SenComponent::L0su0,
        (UnitName::L0su, Corelet::C1) => SenComponent::L0su1,
        (UnitName::L0su, Corelet::Generic) => SenComponent::L0su,
        // ⛔ FOR THE L3 HALVES `0` AND `-1` ARE THE SAME ENTRY — there is no second corelet.
        (UnitName::L3lu, Corelet::C0 | Corelet::Generic) => SenComponent::L3lu,
        (UnitName::L3su, Corelet::C0 | Corelet::Generic) => SenComponent::L3su,
        // ⛔ THE HOLES. See the note above.
        (UnitName::Pt, Corelet::C0 | Corelet::C1)
        | (UnitName::L3lu | UnitName::L3su, Corelet::C1) => return None,
    })
}

// crustify:todo: e017_isGraphSimple
//   authority: sys-arch-spec/progir/progir.cpp:532  (14 lines)  `ProgIrGraph::isGraphSimple`

// crustify:todo: e030_tagToLCCR
//   authority: sys-arch-spec/progir/progir.cpp:720  (88 lines)  `ProgramAndStateInfo::tagToLCCR`

// crustify:todo: e031_tagToPC
//   authority: sys-arch-spec/progir/progir.cpp:696  (23 lines)  `ProgramAndStateInfo::tagToPC`

// crustify:todo: e033_print
//   authority: sys-arch-spec/progir/progir.cpp:25  (38 lines)  `OperandAttr::print`

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::bridges::progir_to_senprog::test_fixtures::{instr, program};
    use crate::islands::progir::OperandField;

    /// e009: a variable operand counts however deeply this island nests it, and a loop's own bounds
    /// do not count at all.
    #[test]
    fn variables_are_found_through_a_loop_body_and_not_in_its_bounds() {
        let plain = program(vec![Block::Code(vec![instr(vec![(
            OperandField::Be,
            OperandValue::Int(3),
        )])])]);
        assert!(!has_variables_in_instr(&plain));

        let bounds_only = program(vec![Block::ForLoop {
            iterator: "i".to_owned(),
            // ⛔ THE BOUNDS ARE VARIABLES AND STILL DO NOT COUNT.
            start: OperandValue::Variable("lo".to_owned()),
            end: OperandValue::Variable("hi".to_owned()),
            body: vec![Block::Code(vec![instr(vec![(
                OperandField::Be,
                OperandValue::Int(3),
            )])])],
        }]);
        assert!(!has_variables_in_instr(&bounds_only));

        let nested = program(vec![Block::ForLoop {
            iterator: "i".to_owned(),
            start: OperandValue::Int(0),
            end: OperandValue::Int(4),
            body: vec![Block::Code(vec![instr(vec![(
                OperandField::Be,
                OperandValue::Variable("v".to_owned()),
            )])])],
        }]);
        assert!(has_variables_in_instr(&nested));
    }

    /// e014: the three holes in `senCompMap`, and the two collapses around them.
    #[test]
    fn the_component_table_has_three_holes() {
        assert_eq!(
            sen_component(UnitName::PtRow(PtRow::Row3), Corelet::C1),
            Some(SenComponent::Ptrow3_1)
        );
        // A row with no corelet is the whole PT.
        assert_eq!(
            sen_component(UnitName::PtRow(PtRow::Row3), Corelet::Generic),
            Some(SenComponent::Pt)
        );
        // The L3 halves have one corelet, so `0` and `-1` are the same entry.
        assert_eq!(
            sen_component(UnitName::L3su, Corelet::C0),
            sen_component(UnitName::L3su, Corelet::Generic)
        );
        // The holes.
        assert_eq!(sen_component(UnitName::Pt, Corelet::C0), None);
        assert_eq!(sen_component(UnitName::Pt, Corelet::C1), None);
        assert_eq!(sen_component(UnitName::L3lu, Corelet::C1), None);
    }
}
