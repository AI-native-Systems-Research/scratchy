// SPDX-License-Identifier: Apache-2.0
//! Ported from `sys-arch-spec/progir/progir.cpp` — 6 unit(s) of bridge 4, `ProgIR -> SenProg`.
//!
//! ⛔ SENPROG IS A PRINT FORMAT, so what is being ported is the exact TEXT: field order,
//! spelling and separators. A predicate that decides what to write, without writing it, is not
//! a port. The authority is `/Users/nickm/git/deeptools-src/<file>:<line>` per unit below.

use sys_arch_spec::arch_enums::SenComponent;

use crate::arch::Arch;
use crate::islands::progir::ty::OperandValue;
use crate::islands::progir::{Block, BlockKind, Program, UnitProgram};
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

/// WHETHER A GRAPH IS THE ONE BLOCK ITS `getSimple*` ACCESSORS CAST — `isGraphSimple`'s answer as a
/// value, because its `blocking` arm is a `DT_ERROR` and this crate never refuses at runtime.
///
/// ⛔ TRAP: THE REFERENCE CALLS A WRONG-KIND HEAD *"Graph has multiple blocks"* — one message for
/// both causes of `blocks.size() > 1 || head->type != graphType`. They are separate here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Simplicity {
    /// `blocks.empty()` — *"Graph is empty, cannot proceed"*.
    Empty,
    /// More than one block: a `simplifyGraph` is owed before anything reads it.
    ManyBlocks,
    /// The single block is not the kind this graph is read as, carrying what it is instead.
    WrongKind(BlockKind),
    /// One block, of the graph's own kind.
    Simple,
}

impl UnitProgram {
    /// Replaces: e017_isGraphSimple
    ///
    /// ⭐ `graph_type` IS THE SUBCLASS. `ProgIrCodeGraph`, `ProgIrRegGraph` and `ProgIrVarGraph`
    /// differ only in the `graphType` their constructor fixes (`progir.h:455-501`), so one Rust
    /// graph plus this argument is all three of them.
    #[must_use]
    pub fn simplicity(&self, graph_type: BlockKind) -> Simplicity {
        match self.blocks.as_slice() {
            [] => Simplicity::Empty,
            // ⛔ THE HEAD'S KIND IS CHECKED ONLY AT ONE BLOCK: `size() > 1` short-circuits it in the
            // reference, so a two-block graph is `ManyBlocks` whatever its head is.
            [head] if head.kind() == graph_type => Simplicity::Simple,
            [head] => Simplicity::WrongKind(head.kind()),
            _ => Simplicity::ManyBlocks,
        }
    }
}

// crustify:todo: e030_tagToLCCR
//   authority: sys-arch-spec/progir/progir.cpp:720  (88 lines)  `ProgramAndStateInfo::tagToLCCR`

// crustify:todo: e031_tagToPC
//   authority: sys-arch-spec/progir/progir.cpp:696  (23 lines)  `ProgramAndStateInfo::tagToPC`

/// WHETHER A VALUE IS DELIMITED — `print`'s `prettyPrint` (`progir.cpp:25`), as the closed set it is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pretty {
    /// `false` — bare text, and a `BOOLEAN` as `0`/`1`. What senprog's operand list takes.
    Off,
    /// `true` — a `VARIABLE` quoted, an `INSTR_TAG` parenthesised, a `BOOLEAN` as `true`/`false`
    /// and an `INT128` prefixed `0x`. ⛔ INERT FOR EVERY OTHER KIND, including `VARIABLE_SYMBOL`,
    /// which `dpc.cpp:690` and `:762` nonetheless pass `true`.
    On,
}

impl OperandValue {
    /// Replaces: e033_print
    ///
    /// ONE OPERAND AS TEXT; `UNKNOWN`'s `DT_ERROR("Uexpected OperandAttr type to print")` is `None`.
    ///
    /// ⛔ THE `ostream` OVERLOAD DROPS ITS OWN FLAG: `progir.h:187-190` defaults `prettyPrint` to
    /// `false` and then calls `print(id, true)`, so every stream caller is pretty regardless.
    /// ⛔ AND `char buff[40]` TRUNCATES `%.6f` — measured, `-3.4e38` loses a real digit and prints
    /// as a tenth of itself. ⭐ WE WRITE THE FULL TEXT; see `a_big_float_is_not_truncated`.
    #[must_use]
    pub fn print(&self, pretty: Pretty) -> Option<String> {
        Some(match self {
            Self::Variable(text) => match pretty {
                Pretty::On => format!("\"{text}\""),
                Pretty::Off => text.clone(),
            },
            Self::InstrTag(text) => match pretty {
                Pretty::On => format!("({text})"),
                Pretty::Off => text.clone(),
            },
            Self::Descriptive(text) => text.clone(),
            // ⛔ NOT `as_int` (e029): that accessor only re-checks the tag this arm already matched,
            // and `getValueImpl` then reads the one value this island carries inline.
            Self::Int(value) => value.to_string(),
            Self::VariableSymbol(id) => format!("%{id}%"),
            // `%.6f` of the varargs promotion to `double`, which `f64::from` is exactly. ⛔ NaN
            // prints `NaN` here against the reference's `nan`; no finite operand differs.
            Self::Float(value) => format!("{:.6}", f64::from(*value)),
            Self::Boolean(flag) => match pretty {
                Pretty::On => (if *flag { "true" } else { "false" }).to_owned(),
                // `to_string(asBool(id))` — there is no `bool` overload, so it promotes to `int`.
                Pretty::Off => u8::from(*flag).to_string(),
            },
            Self::Int128(words) => {
                // ⛔ WORD 0 IS LEFTMOST: `at(0)` is the first `%08x` of the four.
                let [w0, w1, w2, w3] = words;
                let prefix = match pretty {
                    Pretty::On => "0x",
                    Pretty::Off => "",
                };
                format!("{prefix}{w0:08x}{w1:08x}{w2:08x}{w3:08x}")
            }
            Self::Unknown => return None,
        })
    }
}

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

    /// The reference's three failing shapes and its one passing one: empty, more than one block, a
    /// head of the wrong kind, and a single block of the graph's own kind (`progir.cpp:533-544`).
    #[test]
    fn a_graph_is_simple_only_as_one_block_of_its_own_kind() {
        assert_eq!(
            UnitProgram::default().simplicity(BlockKind::Code),
            Simplicity::Empty
        );

        let code = UnitProgram {
            blocks: vec![Block::Code(Vec::new())],
        };
        assert_eq!(code.simplicity(BlockKind::Code), Simplicity::Simple);
        assert_eq!(
            code.simplicity(BlockKind::RegInit),
            Simplicity::WrongKind(BlockKind::Code)
        );

        let two = UnitProgram {
            blocks: vec![Block::Code(Vec::new()), Block::Code(Vec::new())],
        };
        assert_eq!(two.simplicity(BlockKind::Code), Simplicity::ManyBlocks);
    }

    /// e033: every kind's text, and the five `prettyPrint` moves — `progir.cpp:29-58`. There is no
    /// vendor case to port: `dsc/test/operandattr_unit_test.cpp` never calls `print`, and the
    /// `senulator/progs/*/senprog.txt` goldens are unfetched git-lfs pointers on this host.
    #[test]
    fn every_kind_prints_the_reference_text_and_pretty_moves_five_of_them() {
        let both = |value: &OperandValue| (value.print(Pretty::Off), value.print(Pretty::On));
        let text = |plain: &str, pretty: &str| (Some(plain.to_owned()), Some(pretty.to_owned()));
        let same = |plain: &str| text(plain, plain);

        assert_eq!(
            both(&OperandValue::Variable("act_addr".to_owned())),
            text("act_addr", "\"act_addr\"")
        );
        assert_eq!(
            both(&OperandValue::InstrTag("L3_loop_end".to_owned())),
            text("L3_loop_end", "(L3_loop_end)")
        );
        assert_eq!(
            both(&OperandValue::Descriptive("uselccr".to_owned())),
            same("uselccr")
        );
        assert_eq!(both(&OperandValue::Int(-7)), same("-7"));
        // ⛔ THE ONE KIND `dpc.cpp:690` ASKS PRETTY OF AND THAT IGNORES IT.
        assert_eq!(both(&OperandValue::VariableSymbol(3)), same("%3%"));
        assert_eq!(both(&OperandValue::Float(1.5)), same("1.500000"));
        assert_eq!(both(&OperandValue::Boolean(true)), text("1", "true"));
        assert_eq!(both(&OperandValue::Boolean(false)), text("0", "false"));
        assert_eq!(
            both(&OperandValue::Int128([0xdead_beef, 0, 1, 0xffff_ffff])),
            text(
                "deadbeef0000000000000001ffffffff",
                "0xdeadbeef0000000000000001ffffffff"
            )
        );
        // The `DT_ERROR` arm, which is an absent string rather than a stop.
        assert_eq!(both(&OperandValue::Unknown), (None, None));
    }

    /// ⛔ DELIBERATE DIVERGENCE, MEASURED: `snprintf(buff, 40, "%.6f", -3.4e38f)` wants 47 bytes and
    /// keeps 39, so the reference emits `-33999999521443642490773241379936429670` — a real digit
    /// short, one tenth of the value. Truncation starts at `1e32`, where the fraction goes first.
    #[test]
    fn a_big_float_is_not_truncated() {
        assert_eq!(
            OperandValue::Float(-3.4e38).print(Pretty::Off),
            Some("-339999995214436424907732413799364296704.000000".to_owned())
        );
        assert_eq!(
            OperandValue::Float(1e32).print(Pretty::Off),
            Some("100000003318135351409612647563264.000000".to_owned())
        );
    }
}
