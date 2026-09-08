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
use crate::bridges::sentient_to_progir::utils::{ConsumerUnit, update_proper_consumer};
use crate::formats::DataFormat;
use crate::islands::dataflow_ir::ty::GenericComp;
use crate::islands::progir::ty::{FoldId, Operand, OperandValue};
use crate::islands::progir::{Instruction, OpCode, OperandField};
use crate::units::DfirUnit;

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
    /// both (`cpp:254,259,248`).
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
    /// It held one per unit — ⛔ THE STATE `setCommonField`'s `DT_CHECK` ABORTS ON (`cpp:259`);
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
    /// ⭐ A UNIT'S OWN COMMENT WINS OVER THE COMMON ONE (`cpp:224-227`).
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
    /// (`cpp:246,248`), as the only way to reach [`Regular::instr`].
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

    /// Replaces: e073_addEntryToOperandMap
    ///
    /// One field's value per unit, from a map of per-fold constants — scaled, truncated to an integer,
    /// in the format the field is in.
    /// ⛔⛔ NO FOLD ID IS EVER WRITTEN, so a unit's LAST fold wins, and the units whose folds disagree
    /// come back instead of `DT_CHECK_MSG(!folding_needed, "Folding in instruction fields are not
    /// supported")` (`cpp:212`) — whose `folding_needed` is dead either way: this overload sets
    /// `values_same` to `true` on a difference where its 121-line twin sets `false` (`:100`).
    pub fn add_const_entries_to_operand_map(
        &mut self,
        field: OperandField,
        entries: &[FoldConstant],
        scale: f64,
        format: Option<DataFormat>,
    ) -> Vec<UnitKey> {
        // `:180-197` — the per-unit agreement pass, and the only thing it decides.
        let mut seen: Vec<(UnitKey, i64)> = Vec::new();
        let mut disagreeing: Vec<UnitKey> = Vec::new();
        for entry in entries {
            match seen.iter().find(|(at, _)| *at == entry.unit) {
                Some((_, first)) if *first != entry.value && !disagreeing.contains(&entry.unit) => {
                    disagreeing.push(entry.unit);
                }
                Some(_) => {}
                None => seen.push((entry.unit, entry.value)),
            }
        }
        for entry in entries {
            // `:216` — `int64_t(pair.second * scale)`, truncated toward zero as that cast is.
            let scaled = (entry.value as f64 * scale) as i64;
            if let Some(operand) = self.per_unit_operand_mut(field, entry.unit) {
                operand.set(None, OperandValue::Int(scaled));
                // `:219-221` — `INVALID` is the absence, so only a real format is written.
                if let Some(format) = format {
                    operand.format = Some(format);
                }
            }
        }
        disagreeing
    }

    /// The place `operand_map_[field][unit]` names, default-constructed the way the reference's two
    /// subscripts are (`progir.h:257-271`) — `None` is unreachable, the entry having just been made.
    fn per_unit_operand_mut(&mut self, field: OperandField, unit: UnitKey) -> Option<&mut Operand> {
        let at = slot(&self.operand_map, field);
        if !matches!(self.operand_map.get(at), Some((held, _)) if *held == field) {
            let fresh = PerUnitOperand {
                first: (unit, Operand::default()),
                rest: Vec::new(),
            };
            self.operand_map.insert(at, (field, fresh));
        }
        let per_unit = &mut self.operand_map.get_mut(at)?.1;
        // The order [`PerUnitOperand::get`] reads in: the last entry for a unit is the one it sees.
        if let Some(at) = per_unit.rest.iter().rposition(|(held, _)| *held == unit) {
            return Some(&mut per_unit.rest.get_mut(at)?.1);
        }
        if per_unit.first.0 == unit {
            return Some(&mut per_unit.first.1);
        }
        per_unit.rest.push((unit, Operand::default()));
        per_unit.rest.last_mut().map(|(_, operand)| operand)
    }
}

/// ONE UNIT'S CONSTANT ON ONE FOLD — an entry of `getConstantTargetKeyValues`' map
/// (`Dialect/Uniform/Utils.cpp:421`), keyed by a fold's own `GetUnitOp`, so one unit appears once per
/// fold it takes part in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FoldConstant {
    /// `getUnitName(key)`.
    pub unit: UnitKey,
    /// `unit_foldid_map_.at(key)` — ⛔ READ ONLY TO COMPARE, never written into the operand; see
    /// [`UniformInstrInfo::add_const_entries_to_operand_map`].
    pub fold: Option<FoldId>,
    /// The `sentient.constant` this key maps to.
    pub value: i64,
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

/// Replaces: e071_createJmpInstr
///
/// The unconditional jump that ends one uniform region and enters the next.
/// ⛔ THE TARGET IS THE CALLER'S — see [`UniformLabel`] for why `rand()` is not portable, and note
/// that nothing sets this instruction's `tag`: the label it jumps to is defined by the NOP that
/// carries it as a comment, so a target's definition and its use are spelled differently.
#[must_use]
pub fn create_jmp_instr(target: UniformLabel) -> UniformInstrInfo {
    let mut instr = UniformInstrInfo::of(OpCode::JCMP).with_common_comment("jump");
    instr.set_common_field(
        OperandField::Mode,
        Operand::every(OperandValue::Descriptive("always".to_owned())),
    );
    instr.set_common_field(
        OperandField::PcTarget,
        Operand::every(OperandValue::InstrTag(target.name())),
    );
    instr
}

/// WHAT A MAPPED FIELD'S VALUE IS READ OUT OF — `OperandMap::Mode` (`UniformInstrAndBlock.hpp:38-44`),
/// which decides what a `get_unit` VALUE means and nothing else.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MapMode {
    /// `unit_name` — the value unit is a load's consumer; the field takes its `consumertag`.
    UnitName,
    /// `l0_wrap_around` — ⛔ A CONSTANT MODE, NOT A UNIT ONE: it is the only mode that touches the
    /// constant arm, and the only one [`crate::bridges::sentient_to_progir::construct::reg_init`]
    /// selects.
    L0WrapAround,
    /// `set_dest_tgt` — the field takes a one-hot core mask.
    SetDestTgt,
    /// `unit_id` — the field takes the value unit's core id.
    UnitId,
    /// `none` — the field takes the value unit's own `type=` spelling.
    None,
}

/// ONE DEFINING OP A UNIFORM MAPPING'S VALUE CAN BE — the three `dyn_cast`s of the value arm
/// (`cpp:150,153,196`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MappedOp {
    /// `sentient.constant`, scaled by the caller's factor.
    Constant(i64),
    /// `dataflow.get_unit` — what [`MapMode`] then interprets.
    Unit(UnitKey),
    /// `dataflow.create_multicast_group`, ALREADY ENCODED: `encodeMulticastGroupInfo` is outside this
    /// campaign's 130.
    Multicast(i64),
    /// ⛔ ANY OTHER OP, AND IT IS LOAD-BEARING: `operand_map_[name][unit]` default-constructs the
    /// entry before the dispatch (`cpp:147`), so a fourth kind leaves an UNKNOWN value behind rather
    /// than no entry at all.
    Other,
}

/// ONE `(key -> value)` OF A UNIFORM MAPPING, with the key unit's fold id.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MappedEntry {
    /// `key_ops[idx]` — ⭐ `getUnitName` IS `type` + `core` + `corelet` (`Utils/Utils.cpp:264-269`),
    /// which is exactly this key, so "the same unit name" is "the same [`UnitKey`]".
    pub key: UnitKey,
    /// `unit_foldid_map.at(key_unit)`.
    pub fold: Option<FoldId>,
    /// `value_ops[idx]`.
    pub value: MappedOp,
}

/// WHAT A MAPPED FIELD COULD NOT BE GIVEN — the `DT_CHECK`s of the value arm, as offenders.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperandMapRefusal {
    /// One unit's folds disagree — *"Folding in instruction fields are not supported"* (`cpp:145`).
    FoldingNeeded(UnitKey),
    /// `set_dest_tgt` between anything but two SFPs (`cpp:170-173`).
    SetDestNotSfp {
        /// The key unit.
        key: UnitKey,
        /// The value unit.
        value: UnitKey,
    },
    /// `set_dest_tgt` across corelets, or from a unit with none — the reference's `corelet_id >= 0`
    /// and its *"Currently SFP can only send data to the same corelet of another core"* (`cpp:167,174`).
    SetDestCorelet {
        /// The key unit.
        key: UnitKey,
        /// The value unit.
        value: UnitKey,
    },
    /// `unit_name` for a unit that is no load consumer — `updateProperConsumer`'s own `DT_CHECK`
    /// (`Utils.cpp:104`), reached through this mode's generic-component lookup.
    NotAConsumer(UnitKey),
}

/// ONE FIELD'S PER-UNIT VALUE AND WHAT COULD NOT BE PUT IN IT — [`add_entry_to_operand_map`]'s answer.
#[derive(Debug, Clone, PartialEq)]
pub struct MappedField {
    /// ⛔ `None` WHEN NOTHING WAS WRITTEN AT ALL, which is not an empty [`PerUnitOperand`]: a field
    /// whose map has no unit has no value for `getUniformizedInstr`'s `begin()` to answer.
    pub value: Option<PerUnitOperand>,
    /// The offenders, in entry order.
    pub refused: Vec<OperandMapRefusal>,
}

/// WHICH LOAD CONSUMER A UNIT IS — `senCompToGenericComp` then the four-set `updateProperConsumer`
/// admits (`cpp:157-163`, `Utils.cpp:104`).
fn consumer_unit(unit: DfirUnit) -> Option<ConsumerUnit> {
    match unit.generic() {
        GenericComp::Pe => Some(ConsumerUnit::Pe),
        GenericComp::Pt => Some(ConsumerUnit::Pt),
        GenericComp::Sfp => Some(ConsumerUnit::Sfp),
        GenericComp::L0su => Some(ConsumerUnit::L0su),
        _ => None,
    }
}

/// `operand_map_[operand_name][unit_name]` — the entry, default-constructed on first sight.
fn entry_of<'map>(per_unit: &'map mut Vec<(UnitKey, Operand)>, unit: UnitKey) -> &'map mut Operand {
    match per_unit.iter().position(|(at, _)| *at == unit) {
        Some(at) => &mut per_unit[at].1,
        None => {
            per_unit.push((unit, Operand::default()));
            &mut per_unit.last_mut().expect("just pushed").1
        }
    }
}

/// Replaces: e072_addEntryToOperandMap
///
/// One field's value on every unit of a uniform mapping, read out of each key's mapped op.
/// ⛔ THE FOLD ID IS DEAD BY THE CHECK ABOVE IT: every `folding_needed ? … , id : …` writes the common
/// operand, because the branch is only taken where the `DT_CHECK` has already aborted (`cpp:145`).
/// ⛔ AND AGREEMENT IS TESTED PER UNIT NAME, NOT PER ENTRY (`cpp:69-104`) — value equality here, where
/// the reference compares SSA identity and then structural equivalence, allowing two multicast ops
/// with the same producer to differ.
#[must_use]
pub fn add_entry_to_operand_map(entries: &[MappedEntry], mode: MapMode, scale: f64) -> MappedField {
    // `values_same_map_per_unit_name` — the first value each key was seen with, and whether every
    // other entry for that key agrees.
    let mut agreement: Vec<(UnitKey, MappedOp, bool)> = Vec::new();
    for entry in entries {
        match agreement.iter_mut().find(|(at, _, _)| *at == entry.key) {
            Some((_, base, same)) => *same = *same && *base == entry.value,
            None => agreement.push((entry.key, entry.value, true)),
        }
    }
    let mut per_unit: Vec<(UnitKey, Operand)> = Vec::new();
    let mut refused = Vec::new();
    for entry in entries {
        let agrees = agreement
            .iter()
            .find(|(at, _, _)| *at == entry.key)
            .is_some_and(|(_, _, same)| *same);
        if !agrees {
            refused.push(OperandMapRefusal::FoldingNeeded(entry.key));
            continue;
        }
        match entry.value {
            MappedOp::Constant(value) => {
                let mut imm = (value as f64 * scale) as i64;
                // ⛔ `> 1024`, NOT `>=`: the one immediate that is exactly the capacity is left alone.
                if matches!(mode, MapMode::L0WrapAround) && imm > 1024 {
                    imm %= 1024;
                }
                entry_of(&mut per_unit, entry.key).set(None, OperandValue::Int(imm));
            }
            MappedOp::Unit(value) => match mode {
                MapMode::UnitName => match consumer_unit(value.unit) {
                    Some(consumer) => {
                        update_proper_consumer(entry_of(&mut per_unit, entry.key), consumer, None)
                    }
                    None => refused.push(OperandMapRefusal::NotAConsumer(value)),
                },
                MapMode::SetDestTgt => {
                    if !matches!(entry.key.unit, DfirUnit::Sfp)
                        || !matches!(value.unit, DfirUnit::Sfp)
                    {
                        refused.push(OperandMapRefusal::SetDestNotSfp {
                            key: entry.key,
                            value,
                        });
                    } else if value.corelet.is_none() || value.corelet != entry.key.corelet {
                        refused.push(OperandMapRefusal::SetDestCorelet {
                            key: entry.key,
                            value,
                        });
                    } else {
                        // ⭐ THE MASK IS THE VALUE UNIT'S CORE, one-hot — and `core_id < 32` is the
                        // width of [`crate::units::Core`], so that third `DT_CHECK` has no spelling.
                        let mask = 1_i64 << value.core.get();
                        entry_of(&mut per_unit, entry.key).set(None, OperandValue::Int(mask));
                    }
                }
                MapMode::UnitId => {
                    let core = i64::from(value.core.get());
                    entry_of(&mut per_unit, entry.key).set(None, OperandValue::Int(core));
                }
                MapMode::L0WrapAround | MapMode::None => {
                    let spelling = value.unit.spelling().to_owned();
                    entry_of(&mut per_unit, entry.key)
                        .set(None, OperandValue::Descriptive(spelling));
                }
            },
            MappedOp::Multicast(encoded) => {
                // ⚠️ NARROWED TO `int` THERE (`cpp:199`) and kept whole here.
                entry_of(&mut per_unit, entry.key).set(None, OperandValue::Int(encoded));
            }
            // The default-constructed entry, and nothing written into it.
            MappedOp::Other => {
                entry_of(&mut per_unit, entry.key);
            }
        }
    }
    let value = per_unit.split_first().map(|(first, rest)| PerUnitOperand {
        first: first.clone(),
        rest: rest.to_vec(),
    });
    MappedField { value, refused }
}

#[cfg(test)]
mod unit_tests {
    use super::{
        Comment, Displaced, FoldConstant, MapMode, MappedEntry, MappedField, MappedOp,
        OperandMapRefusal, PerUnitOperand, UniformInstrInfo, UniformLabel,
        add_entry_to_operand_map, create_jmp_instr, create_nop_instr,
    };
    use crate::bridges::sentient_to_progir::state::UnitKey;
    use crate::formats::DataFormat;
    use crate::islands::progir::ty::{FoldId, Operand, OperandValue, PerFold};
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

    /// IBM'S OWN `symbolic_ebr.mlir:8` — `L3_JCMP :: mode:always pc_target:(uniform_tgt_[[N]])
    /// // jump`, with the label the caller's rather than a draw.
    #[test]
    fn the_uniform_jump_is_unconditional_and_names_its_target() {
        let jmp = create_jmp_instr(UniformLabel(3));
        assert_eq!(jmp.opcode, OpCode::JCMP);
        assert_eq!(jmp.comment, Comment::Common("jump".to_owned()));
        assert_eq!(
            jmp.common_fields,
            vec![
                (
                    OperandField::Mode,
                    Operand::every(OperandValue::Descriptive("always".to_owned()))
                ),
                (
                    OperandField::PcTarget,
                    Operand::every(OperandValue::InstrTag("uniform_tgt_3".to_owned()))
                ),
            ]
        );
        assert_eq!(jmp.tag, None, "the target is defined elsewhere");
    }

    /// e072: the four value kinds and the five modes, and the two offenders a mapping can carry.
    /// IBM'S OWN `uniformization_small_elem_size.mlir` maps each LXLU to its core's PE and prints
    /// `LX_LDSTI :: consumertag:pe`, which is the `unit_name` mode over a `get_unit` value.
    #[test]
    fn a_mapped_field_reads_each_value_through_its_mode() {
        let lxlu = unit(DfirUnit::Lxlu, 3);
        let other = unit(DfirUnit::Lxlu, 19);
        let mapped = |value: MappedOp, mode| {
            add_entry_to_operand_map(
                &[MappedEntry {
                    key: lxlu,
                    fold: None,
                    value,
                }],
                mode,
                1.0,
            )
        };
        let only = |field: &MappedField| field.value.clone().expect("one entry").first.1.clone();
        // `unit_name`: the consumer tag, which is the vendor's `consumertag:pe`.
        let consumer = mapped(MappedOp::Unit(unit(DfirUnit::Pe, 3)), MapMode::UnitName);
        assert_eq!(
            only(&consumer),
            Operand::every(OperandValue::Descriptive("pe".to_owned()))
        );
        assert!(consumer.refused.is_empty());
        // `none`: the value unit's own spelling. `unit_id`: its core.
        assert_eq!(
            only(&mapped(MappedOp::Unit(other), MapMode::None)),
            Operand::every(OperandValue::Descriptive("lxlu".to_owned()))
        );
        assert_eq!(
            only(&mapped(MappedOp::Unit(other), MapMode::UnitId)),
            Operand::every(OperandValue::Int(19))
        );
        // `l0_wrap_around` folds a constant over 1024 and leaves 1024 itself alone.
        let wrapped = add_entry_to_operand_map(
            &[MappedEntry {
                key: lxlu,
                fold: None,
                value: MappedOp::Constant(2100),
            }],
            MapMode::L0WrapAround,
            1.0,
        );
        assert_eq!(only(&wrapped), Operand::every(OperandValue::Int(52)));
        assert_eq!(
            only(&mapped(MappedOp::Constant(1024), MapMode::L0WrapAround)),
            Operand::every(OperandValue::Int(1024)),
            "`> 1024`, not `>=`"
        );
        // A multicast id passes through; a fourth kind leaves the default-constructed entry.
        assert_eq!(
            only(&mapped(MappedOp::Multicast(7), MapMode::None)),
            Operand::every(OperandValue::Int(7))
        );
        assert_eq!(
            only(&mapped(MappedOp::Other, MapMode::None)),
            Operand::default()
        );
        // `set_dest_tgt` is a one-hot core mask between two SFPs of the same corelet, and refuses
        // anything else.
        let sfp = unit(DfirUnit::Sfp, 0);
        let peer = unit(DfirUnit::Sfp, 5);
        let dest = add_entry_to_operand_map(
            &[MappedEntry {
                key: sfp,
                fold: None,
                value: MappedOp::Unit(peer),
            }],
            MapMode::SetDestTgt,
            1.0,
        );
        assert_eq!(only(&dest), Operand::every(OperandValue::Int(1 << 5)));
        assert_eq!(
            mapped(MappedOp::Unit(peer), MapMode::SetDestTgt).refused,
            vec![OperandMapRefusal::SetDestNotSfp {
                key: lxlu,
                value: peer
            }],
            "the key unit is an LXLU"
        );
        // A unit whose folds disagree is the reference's abort, and no consumer is another.
        assert_eq!(
            add_entry_to_operand_map(
                &[
                    MappedEntry {
                        key: lxlu,
                        fold: Some(crate::islands::progir::ty::FoldId(0)),
                        value: MappedOp::Constant(1),
                    },
                    MappedEntry {
                        key: lxlu,
                        fold: Some(crate::islands::progir::ty::FoldId(1)),
                        value: MappedOp::Constant(2),
                    },
                ],
                MapMode::None,
                1.0,
            ),
            MappedField {
                value: None,
                refused: vec![
                    OperandMapRefusal::FoldingNeeded(lxlu),
                    OperandMapRefusal::FoldingNeeded(lxlu),
                ],
            }
        );
        assert_eq!(
            mapped(MappedOp::Unit(other), MapMode::UnitName).refused,
            vec![OperandMapRefusal::NotAConsumer(other)],
            "an LXLU is no load consumer"
        );
    }

    /// e073: one field's value per unit, scaled and in one format — ⛔ AND THE UNIT WHOSE TWO FOLDS
    /// DISAGREE COMES BACK, its last fold having won.
    #[test]
    fn the_folds_of_one_unit_collapse_to_its_last_value() {
        let pe = unit(DfirUnit::Pe, 0);
        let sfp = unit(DfirUnit::Sfp, 0);
        let mut instr = UniformInstrInfo::of(OpCode::FMA);
        let entries = [
            FoldConstant {
                unit: pe,
                fold: Some(FoldId(0)),
                value: 3,
            },
            FoldConstant {
                unit: pe,
                fold: Some(FoldId(1)),
                value: 5,
            },
            FoldConstant {
                unit: sfp,
                fold: Some(FoldId(0)),
                value: 4,
            },
        ];
        let disagreeing = instr.add_const_entries_to_operand_map(
            OperandField::Imm,
            &entries,
            2.0,
            Some(DataFormat::IeeeFp32),
        );
        assert_eq!(disagreeing, vec![pe], "3 and 5 are not the same constant");
        let (field, per_unit) = instr.operand_map.first().expect("one field was written");
        assert_eq!(*field, OperandField::Imm);
        assert_eq!(
            per_unit.get(pe),
            &Operand {
                value: PerFold::Every(OperandValue::Int(10)),
                format: Some(DataFormat::IeeeFp32),
            },
            "the last fold's value, scaled, with no fold id on it"
        );
        assert_eq!(
            per_unit.get(sfp).value,
            PerFold::Every(OperandValue::Int(8))
        );
    }
}
