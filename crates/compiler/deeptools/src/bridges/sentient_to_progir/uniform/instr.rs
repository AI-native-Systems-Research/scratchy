//! ONE INSTRUCTION AND ITS OPERAND MAP — `UniformInstrInfo` and `OperandMap`.
//!
//! ⭐ `addEntryToOperandMap` HAS TWO OVERLOADS (121L and 43L) and they are two units.
//!
//! 9 units. Every citation resolves against the authority tree
//! `/Users/nickm/git/deeptools-src` at revision `a0d29abbed`.
//!
//! | unit | level | LoC | authority |
//! |---|---|---|---|
//! | `e017_createNOPInstr` | 0 | 8 | `dcc/src/Conversion/SentientToProgIR/UniformInstrAndBlock.cpp:43` |
//! | `e018_getUniformizedInstr` | 0 | 21 | `dcc/src/Conversion/SentientToProgIR/UniformInstrAndBlock.cpp:218` |
//! | `e019_getRegularInstr` | 0 | 11 | `dcc/src/Conversion/SentientToProgIR/UniformInstrAndBlock.cpp:240` |
//! | `e020_getCommonField` | 0 | 5 | `dcc/src/Conversion/SentientToProgIR/UniformInstrAndBlock.cpp:252` |
//! | `e021_setCommonField` | 0 | 5 | `dcc/src/Conversion/SentientToProgIR/UniformInstrAndBlock.cpp:257` |
//! | `e022_hasCommonField` | 0 | 3 | `dcc/src/Conversion/SentientToProgIR/UniformInstrAndBlock.cpp:262` |
//! | `e071_createJmpInstr` | 1 | 12 | `dcc/src/Conversion/SentientToProgIR/UniformInstrAndBlock.cpp:30` |
//! | `e072_addEntryToOperandMap` | 1 | 121 | `dcc/src/Conversion/SentientToProgIR/UniformInstrAndBlock.cpp:52` |
//! | `e073_addEntryToOperandMap` | 1 | 43 | `dcc/src/Conversion/SentientToProgIR/UniformInstrAndBlock.cpp:174` |

use crate::bridges::sentient_to_progir::state::UnitKey;
use crate::islands::progir::ty::Operand;
use crate::islands::progir::{Instruction, OpCode, OperandField};

/// ONE INSTRUCTION BEFORE ITS UNITS ARE SPLIT APART — `UniformInstrInfo`
/// (`dcc/src/Conversion/SentientToProgIR/UniformInstrAndBlock.hpp:88-166`).
///
/// ⛔⛔ `UniformInstrInfo()` LEAVES `instn_` UNINITIALISED (`:146`) — unlike `InstrInfo`, which
/// defaults it to `NumOpCodes`. Every constructor in this span sets it immediately, so the opcode is
/// required here and that state has no spelling.
#[derive(Debug, Clone, PartialEq)]
pub struct UniformInstrInfo {
    /// `instn_`.
    pub opcode: OpCode,
    /// `instFields_` — the fields common to all units, in field order.
    pub common_fields: Vec<(OperandField, Operand)>,
    /// `operand_map_` — the fields whose value differs by unit, in field order
    /// (`UniformInstrAndBlock.hpp:84-86`). ⛔ ITS WRITERS ARE `e072`/`e073`; `getUniformizedInstr` is
    /// its reader.
    ///
    /// ⛔⛔ DISJOINT FROM [`Self::common_fields`] BY THE ACCESSORS, WHICH IS WHAT THREE `DT_CHECK`s
    /// TEST — `getCommonField`, `setCommonField` and `getRegularInstr` each abort on a field held in
    /// both (`cpp:254,258,248`).
    pub operand_map: Vec<(OperandField, PerUnitOperand)>,
    /// `tag_` — ⛔ THE EMPTY STRING IS THE ABSENCE in the reference (`hasTag`), so it is `None` here.
    pub tag: Option<String>,
    /// `unit_common_comment_` and `unit_to_comment_` together.
    pub comment: Comment,
    /// `deadCode_`.
    pub dead: bool,
}

/// AN INSTRUCTION'S COMMENT — ⛔ COMMON OR PER-UNIT, NEVER BOTH.
///
/// `setCommonComment` clears `unit_to_comment_` and `setUniformizedComment` clears
/// `unit_common_comment_` (`UniformInstrAndBlock.hpp:124-131`), so the pair is one choice and two
/// fields could hold a state neither setter can produce.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum Comment {
    /// Neither field set.
    #[default]
    None,
    /// `unit_common_comment_` — one comment for every unit.
    Common(String),
    /// `unit_to_comment_` — one per unit, for the units that have one.
    PerUnit(Vec<(UnitKey, String)>),
}

impl UniformInstrInfo {
    /// `UniformInstrInfo` with its opcode set and nothing else — what every `Construct*Instr` starts
    /// from.
    #[must_use]
    pub const fn of(opcode: OpCode) -> UniformInstrInfo {
        UniformInstrInfo {
            opcode,
            common_fields: Vec::new(),
            operand_map: Vec::new(),
            tag: None,
            comment: Comment::None,
            dead: false,
        }
    }

    /// `setCommonComment` (`UniformInstrAndBlock.hpp:128-131`).
    #[must_use]
    pub fn with_common_comment(self, comment: &str) -> UniformInstrInfo {
        UniformInstrInfo {
            comment: Comment::Common(comment.to_owned()),
            ..self
        }
    }
}

// ⛔ ONE `crustify:todo:` PER SCHEDULED UNIT. Replace each with the ported function
// carrying `/// Replaces: eNNN_name`. A surviving TODO is open work.

/// ONE FIELD'S VALUE ON EVERY UNIT — an `operand_map_` entry, an
/// `unordered_map<unit, OperandAttr>` (`UniformInstrAndBlock.hpp:85-86`).
///
/// ⛔ NEVER EMPTY: `getUniformizedInstr` answers `begin()->second` for a unit it does not find, which
/// has nothing to read when the entry holds no unit at all.
#[derive(Debug, Clone, PartialEq)]
pub struct PerUnitOperand {
    /// `begin()` — what a unit with no value of its own takes.
    pub first: (UnitKey, Operand),
    /// The rest of the units, in insertion order.
    pub rest: Vec<(UnitKey, Operand)>,
}

impl PerUnitOperand {
    /// This field's value on one unit — the first entry when that unit has none, and the LAST entry
    /// for a unit written twice, as `map[unit] = value` leaves.
    #[must_use]
    pub fn get(&self, unit: UnitKey) -> &Operand {
        self.rest
            .iter()
            .rev()
            .chain(core::iter::once(&self.first))
            .find(|(at, _)| *at == unit)
            .map_or(&self.first.1, |(_, value)| value)
    }
}

/// WHAT A FIELD HELD BEFORE IT WAS SET — [`UniformInstrInfo::set_common_field`]'s answer.
#[derive(Debug, Clone, PartialEq)]
pub enum Displaced {
    /// It held one value for every unit.
    Common(Operand),
    /// It held one per unit — ⛔ THE STATE `setCommonField`'s `DT_CHECK` ABORTS ON (`cpp:258`);
    /// handing it back loses nothing instead.
    PerUnit(PerUnitOperand),
}

/// Where `field` sits in a field-ordered map, whether or not it is already there.
fn slot<V>(fields: &[(OperandField, V)], field: OperandField) -> usize {
    fields.partition_point(|(held, _)| *held < field)
}

/// The value `fields` holds for `field`, in a field-ordered map.
fn held<V>(fields: &[(OperandField, V)], field: OperandField) -> Option<&V> {
    match fields.get(slot(fields, field)) {
        Some((at, value)) if *at == field => Some(value),
        _ => None,
    }
}

impl Comment {
    /// The comment one unit sees — its own if it has one, else the common one, which is empty
    /// whenever a per-unit comment was set (`hpp:124-131`).
    #[must_use]
    pub fn for_unit(&self, unit: UnitKey) -> Option<String> {
        match self {
            Comment::None => None,
            Comment::Common(comment) => Some(comment.clone()),
            Comment::PerUnit(comments) => comments
                .iter()
                .rev()
                .find(|(at, _)| *at == unit)
                .map(|(_, comment)| comment.clone()),
        }
    }
}

impl UniformInstrInfo {
    /// Replaces: e018_getUniformizedInstr
    ///
    /// This instruction as one unit runs it — the common fields, with every per-unit field carrying
    /// that unit's own value.
    ///
    /// ⛔ A UNIT WITH NO ENTRY TAKES THE FIRST ONE — the reference's own *"use a random entry for
    /// uniform purpose"* `begin()` into an `unordered_map`.
    /// ⭐ A UNIT'S OWN COMMENT WINS OVER THE COMMON ONE (`cpp:225-228`).
    #[must_use]
    pub fn uniformized_instr(&self, unit: UnitKey) -> Instruction {
        let mut fields = self.common_fields.clone();
        for (field, per_unit) in &self.operand_map {
            let value = per_unit.get(unit).clone();
            let at = slot(&fields, *field);
            match fields.get_mut(at) {
                Some((held, was)) if *held == *field => *was = value,
                _ => fields.insert(at, (*field, value)),
            }
        }
        Instruction {
            opcode: self.opcode,
            symbolic_opcode: None,
            fields,
            dead: self.dead,
            tag: self.tag.clone(),
            comment: self.comment.for_unit(unit),
        }
    }

    /// THE PROOF THAT NOTHING HERE IS PER-UNIT — `getRegularInstr`'s two `DT_CHECK`s
    /// (`cpp:245,248`), as the only way to reach [`Regular::instr`].
    #[must_use]
    pub fn regular(&self) -> Option<Regular<'_>> {
        let per_unit_comment = match &self.comment {
            Comment::PerUnit(comments) => !comments.is_empty(),
            Comment::None | Comment::Common(_) => false,
        };
        (!per_unit_comment && self.operand_map.is_empty()).then_some(Regular(self))
    }

    /// Replaces: e020_getCommonField
    ///
    /// The value this field carries on every unit.
    ///
    /// ⛔ `None` ALSO WHEN THE FIELD IS PER-UNIT, which the reference `DT_CHECK`s instead — either way
    /// there is no one value the field holds, and answering the common one would be reading a value
    /// this instruction has already superseded.
    #[must_use]
    pub fn common_field(&self, field: OperandField) -> Option<&Operand> {
        match held(&self.operand_map, field) {
            Some(_) => None,
            None => held(&self.common_fields, field),
        }
    }

    /// Replaces: e021_setCommonField
    ///
    /// Gives one field the same value on every unit, keeping the map in field order.
    ///
    /// ⛔ IT ANSWERS WHAT IT DISPLACED, including a per-unit entry the reference's `DT_CHECK` would
    /// abort on — dropping that map silently is the one outcome a setter must not have.
    pub fn set_common_field(&mut self, field: OperandField, value: Operand) -> Option<Displaced> {
        let at = slot(&self.operand_map, field);
        let evicted = match self.operand_map.get(at) {
            Some((held, _)) if *held == field => {
                Some(Displaced::PerUnit(self.operand_map.remove(at).1))
            }
            _ => None,
        };
        let at = slot(&self.common_fields, field);
        match self.common_fields.get_mut(at) {
            Some((held, was)) if *held == field => {
                let was = core::mem::replace(was, value);
                evicted.or(Some(Displaced::Common(was)))
            }
            _ => {
                self.common_fields.insert(at, (field, value));
                evicted
            }
        }
    }

    /// Replaces: e022_hasCommonField
    ///
    /// Whether `instFields_` holds this field at all — ⛔ THE ONE OF THE THREE THAT DOES NOT CONSULT
    /// `operand_map_` (`cpp:263`), so it answers the map it names.
    #[must_use]
    pub fn has_common_field(&self, field: OperandField) -> bool {
        held(&self.common_fields, field).is_some()
    }
}

/// AN INSTRUCTION WITH NOTHING PER-UNIT — what [`UniformInstrInfo::regular`] witnesses.
#[derive(Debug, Clone, Copy)]
pub struct Regular<'a>(&'a UniformInstrInfo);

impl Regular<'_> {
    /// Replaces: e019_getRegularInstr
    ///
    /// The single instruction a REGULAR block's units all run.
    #[must_use]
    pub fn instr(&self) -> Instruction {
        Instruction {
            opcode: self.0.opcode,
            symbolic_opcode: None,
            fields: self.0.common_fields.clone(),
            dead: self.0.dead,
            tag: self.0.tag.clone(),
            comment: match &self.0.comment {
                Comment::Common(comment) => Some(comment.clone()),
                // Unreachable through `regular`, which is the only maker of this view.
                Comment::None | Comment::PerUnit(_) => None,
            },
        }
    }
}

/// THE NAME OF A UNIFORM BRANCH TARGET — the `rand()` the reference draws for one (`cpp:46-48`).
///
/// ⛔ A CALLER'S NUMBER, NOT A DRAW: two labels only have to differ, and a compiler whose output
/// depends on `rand()` cannot be compared against itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UniformLabel(pub u32);

impl UniformLabel {
    /// `"uniform_tgt_" + std::to_string(random)`.
    #[must_use]
    pub fn name(self) -> String {
        format!("uniform_tgt_{}", self.0)
    }
}

/// Replaces: e017_createNOPInstr
///
/// The NOP that pads a short uniform region up to the longest one.
///
/// ⛔ THE LABEL IS ONLY A COMMENT, NOT A TAG: nothing jumps to this NOP — `createJmpInstr` draws its
/// own target into `pc_target` — and `setComment` is dropped outside debug (`progir.h:322-330`), so
/// the padding NOP carries the name of a target it does not define.
#[must_use]
pub fn create_nop_instr(label: UniformLabel) -> UniformInstrInfo {
    UniformInstrInfo::of(OpCode::NOP).with_common_comment(&label.name())
}
// crustify:todo: e071_createJmpInstr
// crustify:todo: e072_addEntryToOperandMap
// crustify:todo: e073_addEntryToOperandMap
#[cfg(test)]
mod unit_tests {
    use super::{
        Comment, Displaced, PerUnitOperand, UniformInstrInfo, UniformLabel, create_nop_instr,
    };
    use crate::bridges::sentient_to_progir::state::UnitKey;
    use crate::islands::progir::ty::{Operand, OperandValue};
    use crate::islands::progir::{OpCode, OperandField};
    use crate::units::{Core, Corelet, DfirUnit};

    fn unit(unit: DfirUnit, core: u32) -> UnitKey {
        UnitKey {
            unit,
            core: Core::checked(core).expect("every arch has core 0"),
            corelet: Corelet::checked(0),
        }
    }

    fn int(value: i64) -> Operand {
        Operand::every(OperandValue::Int(value))
    }

    /// e017: the padding NOP names a uniform target in its comment and carries nothing else.
    #[test]
    fn the_padding_nop_carries_its_label_as_a_comment_and_nothing_else() {
        let nop = create_nop_instr(UniformLabel(7));
        assert_eq!(nop.opcode, OpCode::NOP);
        assert_eq!(nop.comment, Comment::Common("uniform_tgt_7".to_owned()));
        assert_eq!(nop.tag, None, "the label is a comment, never a branch tag");
        assert!(nop.common_fields.is_empty() && nop.operand_map.is_empty());
    }

    /// e018: a unit takes its own field value and its own comment; a unit with neither falls back to
    /// the first entry and to no comment at all.
    #[test]
    fn a_unit_without_its_own_entry_takes_the_first_one() {
        let pe = unit(DfirUnit::Pe, 0);
        let sfp = unit(DfirUnit::Sfp, 0);
        let mut instr = UniformInstrInfo::of(OpCode::FMA);
        instr.set_common_field(OperandField::Tgtrf, int(3));
        instr.operand_map.push((
            OperandField::Src0,
            PerUnitOperand {
                first: (pe, int(10)),
                rest: vec![(sfp, int(20))],
            },
        ));
        instr.comment = Comment::PerUnit(vec![(sfp, "the sfp's".to_owned())]);

        let their_own = instr.uniformized_instr(sfp);
        assert_eq!(
            their_own.fields,
            vec![(OperandField::Src0, int(20)), (OperandField::Tgtrf, int(3))],
            "the unit's own field and the common one, in field order"
        );
        assert_eq!(their_own.comment.as_deref(), Some("the sfp's"));
        let unmapped = instr.uniformized_instr(unit(DfirUnit::Lxlu, 0));
        assert_eq!(
            unmapped.fields,
            vec![(OperandField::Src0, int(10)), (OperandField::Tgtrf, int(3))],
            "a unit with no entry reads begin()->second"
        );
        assert_eq!(unmapped.comment, None, "the common comment is empty here");
        assert_eq!(instr.uniformized_instr(pe).fields[0].1, int(10));
    }

    /// e019: only an instruction with nothing per-unit yields the regular view, and that view is the
    /// common fields with the common comment.
    #[test]
    fn only_an_instruction_with_nothing_per_unit_is_regular() {
        let mut instr = UniformInstrInfo::of(OpCode::RETURN).with_common_comment("end");
        instr.set_common_field(OperandField::Tgtrf, int(1));
        let regular = instr.regular().expect("nothing is per-unit").instr();
        assert_eq!(regular.opcode, OpCode::RETURN);
        assert_eq!(regular.fields, vec![(OperandField::Tgtrf, int(1))]);
        assert_eq!(regular.comment.as_deref(), Some("end"));

        let mut per_unit = instr.clone();
        per_unit.operand_map.push((
            OperandField::Src0,
            PerUnitOperand {
                first: (unit(DfirUnit::Pe, 0), int(4)),
                rest: Vec::new(),
            },
        ));
        assert!(
            per_unit.regular().is_none(),
            "a per-unit field is not regular"
        );
        let mut per_unit_comment = instr;
        per_unit_comment.comment = Comment::PerUnit(vec![(unit(DfirUnit::Pe, 0), "pe".to_owned())]);
        assert!(per_unit_comment.regular().is_none());
    }

    /// e020 + e022: a per-unit field has no common value, though `instFields_` may still name it.
    #[test]
    fn a_per_unit_field_has_no_common_value() {
        let mut instr = UniformInstrInfo::of(OpCode::FMA);
        assert_eq!(instr.common_field(OperandField::Src0), None);
        assert!(!instr.has_common_field(OperandField::Src0));
        instr.set_common_field(OperandField::Src0, int(5));
        assert_eq!(instr.common_field(OperandField::Src0), Some(&int(5)));
        assert!(instr.has_common_field(OperandField::Src0));
        instr.operand_map.push((
            OperandField::Src0,
            PerUnitOperand {
                first: (unit(DfirUnit::Pe, 0), int(9)),
                rest: Vec::new(),
            },
        ));
        assert_eq!(
            instr.common_field(OperandField::Src0),
            None,
            "the per-unit value supersedes the common one"
        );
        assert!(
            instr.has_common_field(OperandField::Src0),
            "hasCommonField reads instFields_ alone (cpp:263)"
        );
    }

    /// e021: setting keeps field order, answers what it displaced, and hands back a per-unit entry
    /// rather than dropping it.
    #[test]
    fn setting_a_field_keeps_field_order_and_answers_what_it_displaced() {
        let mut instr = UniformInstrInfo::of(OpCode::FMA);
        assert_eq!(instr.set_common_field(OperandField::Tgtrf, int(1)), None);
        assert_eq!(instr.set_common_field(OperandField::Src0, int(2)), None);
        assert!(
            instr
                .common_fields
                .iter()
                .map(|(field, _)| *field)
                .is_sorted(),
            "instFields_ is a std::map: field order is wire order"
        );
        assert_eq!(
            instr.set_common_field(OperandField::Tgtrf, int(3)),
            Some(Displaced::Common(int(1)))
        );
        let per_unit = PerUnitOperand {
            first: (unit(DfirUnit::Pe, 0), int(9)),
            rest: Vec::new(),
        };
        instr
            .operand_map
            .push((OperandField::Src1, per_unit.clone()));
        assert_eq!(
            instr.set_common_field(OperandField::Src1, int(4)),
            Some(Displaced::PerUnit(per_unit))
        );
        assert!(instr.operand_map.is_empty());
    }
}
