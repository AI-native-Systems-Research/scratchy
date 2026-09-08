// SPDX-License-Identifier: Apache-2.0
//! Ported from `sys-arch-spec/progir/progir.cpp` — 6 unit(s) of bridge 4, `ProgIR -> SenProg`.
//!
//! ⛔ SENPROG IS A PRINT FORMAT, so what is being ported is the exact TEXT: field order,
//! spelling and separators. A predicate that decides what to write, without writing it, is not
//! a port. The authority is `/Users/nickm/git/deeptools-src/<file>:<line>` per unit below.

use std::collections::{BTreeMap, BTreeSet};

use sys_arch_spec::arch_enums::SenComponent;
use sys_arch_spec::operand::Operand as OperandField;
use sys_arch_spec::regfile::Component;
use sys_arch_spec::InstOpCode;

use crate::arch::Arch;
use crate::islands::progir::ty::OperandValue;
use crate::islands::progir::{Block, BlockKind, Instruction, Program, UnitProgram};
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

/// WHICH COMPUTE HALF A PROGRAM'S KEY NAMES — the `corelet` string `dpc.cpp:635-639` prints, which
/// is always a digit: the L3 halves are forced to `"0"` and every other key ends in one.
///
/// ⛔ SEPARATE FROM [`Corelet`] BECAUSE `-1` CANNOT OCCUR HERE — a `senCompProgram_` key is
/// corelet-resolved, so the generic arm of the lookup is not one of this decomposition's answers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProgramCorelet {
    /// `0`.
    C0,
    /// `1`.
    C1,
}

impl ProgramCorelet {
    /// The digit as the header line and the reg-init line both print it.
    #[must_use]
    pub const fn spelling(self) -> &'static str {
        match self {
            Self::C0 => "0",
            Self::C1 => "1",
        }
    }

    /// The same half as [`sen_component`]'s argument.
    #[must_use]
    pub const fn as_corelet(self) -> Corelet {
        match self {
            Self::C0 => Corelet::C0,
            Self::C1 => Corelet::C1,
        }
    }
}

impl UnitName {
    /// Every key half, in `senCompMap`'s own order.
    pub const ALL: [Self; 17] = [
        Self::Sfp,
        Self::Pt,
        Self::PtRow(PtRow::Row0),
        Self::PtRow(PtRow::Row1),
        Self::PtRow(PtRow::Row2),
        Self::PtRow(PtRow::Row3),
        Self::PtRow(PtRow::Row4),
        Self::PtRow(PtRow::Row5),
        Self::PtRow(PtRow::Row6),
        Self::PtRow(PtRow::Row7),
        Self::Pe,
        Self::Lxlu,
        Self::Lxsu,
        Self::L0lu,
        Self::L0su,
        Self::L3lu,
        Self::L3su,
    ];

    /// `senComponentsToString.at(unit)` AND THE SURGERY THAT FOLLOWS IT — `corelet = name.back()`,
    /// `name.pop_back()`, then `insert(2, '_')` for a `row` (`dpc.cpp:635-645`), read as the INVERSE
    /// of [`sen_component`] so the two tables cannot disagree.
    ///
    /// ⛔ A GENERIC KEY HAS NO DECOMPOSITION: `"pt"` loses its `t` to the corelet and
    /// `getSenComponent("p")` then aborts, so `Pt`, `Pe`, `Sfp`, `Lxlu`… and every non-unit
    /// component — `Hbm`, `Ring`, `Zero` — refuse here exactly as they do there.
    #[must_use]
    pub fn of(unit: SenComponent) -> Option<(Self, ProgramCorelet)> {
        // ⭐ THE L3 HALVES ARE THE SPECIAL CASE AND THEY LAND ON `0` EITHER WAY: `dpc.cpp:635-636`
        // assigns `"0"` outright, and their `C1` row is one of `senCompMap`'s three holes.
        [ProgramCorelet::C0, ProgramCorelet::C1]
            .into_iter()
            .flat_map(|corelet| Self::ALL.map(|name| (name, corelet)))
            .find(|&(name, corelet)| sen_component(name, corelet.as_corelet()) == Some(unit))
    }

    /// THE ISA TABLE THIS UNIT READS — `isaPerUnit.at(getSenComponent(unitName))`, whose corelet
    /// argument defaults to `-1` (`dpc.cpp:646-647`), so a PT row and the whole PT share one table.
    ///
    /// ⭐ TOTAL, because [`Component`] is exactly the nine units the reference's generic lookup can
    /// land on — the memories, links and register files that make it `DT_ERROR` are not in it.
    #[must_use]
    pub const fn component(self) -> Component {
        match self {
            Self::Sfp => Component::Sfp,
            Self::Pt | Self::PtRow(_) => Component::Pt,
            Self::Pe => Component::Pe,
            Self::Lxlu => Component::Lxlu,
            Self::Lxsu => Component::Lxsu,
            Self::L0lu => Component::L0lu,
            Self::L0su => Component::L0su,
            Self::L3lu => Component::L3lu,
            Self::L3su => Component::L3su,
        }
    }

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

/// AN INSTRUCTION'S POSITION IN ITS UNIT'S PROGRAM — the `pc` `tagToPC` returns, which senprog
/// writes as the value of every tagged operand that is not a `JCMP`'s `src0` (`dpc.cpp:702-705`).
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Pc(usize);

impl Pc {
    /// The line number.
    #[must_use]
    pub const fn get(self) -> usize {
        self.0
    }
}

/// WHICH LOOP COUNTER A TAG NAMES — the `int` `tagToLCCR` returns, which senprog writes as the
/// `src0` of a `JCMP`/`JCMPI` (`dpc.cpp:698-700`).
///
/// ⛔ SIGNED, AND NOT BOUNDED BY `numLCCRs`: it is `loopNesting` after an unbalanced `be` has taken
/// it below zero, which is why the reference's own closing test is `lccrIdx < 0` and not `== -1`.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LccrIndex(i32);

impl LccrIndex {
    /// The index as the text carries it.
    #[must_use]
    pub const fn get(self) -> i32 {
        self.0
    }
}

/// WHERE A TAG RESOLVED, OR WHY IT DID NOT — `tagToPC`'s `int` together with its two `DT_ERROR`s.
///
/// ⭐ `allowNotFound` IS NOT A PARAMETER: it only chooses between `pc = 0` and an abort AFTER the
/// scan, so the verdict carries both and `convertIr2Senprog` reads it against the instruction's own
/// `deadCode_`, which is what it passes (`dpc.cpp:702-705`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TagPc {
    /// The line that carries the tag.
    At(Pc),
    /// *"Tag: %s found in lines %d and %d"* — ⛔ NO FALLBACK: `allowNotFound` does not excuse this.
    Twice(Pc, Pc),
    /// *"No instruction found with tag"* — `allowNotFound` returns `0` instead.
    NotFound,
}

/// THE SAME FOR `tagToLCCR`, whose three abort sites do NOT share one fallback.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TagLccr {
    /// The `lccrIdx` the tag's `MVLOOPCNT` fixed.
    At(LccrIndex),
    /// *"found after JCMP that could alter LCCR index"* — `allowNotFound` returns `0`.
    AfterAlteringJump,
    /// *"found on op different from MVLOOPCNT"* — `allowNotFound` returns `shadowLccrIdx + 1`,
    /// which is carried here because it is not the other arms' zero.
    NotOnLoop(LccrIndex),
    /// *"found in multiple lines"* — ⛔ NO FALLBACK.
    Twice,
    /// *"No loop found with tag"* — `allowNotFound` returns `0`.
    NoLoop,
}

/// Replaces: e031_tagToPC
///
/// Which line of `prog` carries `tag` — the reference's `getSimpleInstrVect()` is the slice.
///
/// ⛔ TRAP: A DEAD FIRST MATCH IS REPLACED, NOT REPORTED AS A DUPLICATE. The test is
/// `pc == -1 || tagFoundOnDeadCode`, so live code silently wins over dead code that came first, and
/// only two LIVE carriers are *"found in lines %d and %d"*.
#[must_use]
pub fn tag_to_pc(prog: &[Instruction], tag: &str) -> TagPc {
    let mut tag_found_on_dead_code = false;
    let mut pc: Option<usize> = None;
    for (line, instr) in prog.iter().enumerate() {
        if instr.tag_str() == tag {
            match pc {
                Some(first) if !tag_found_on_dead_code => {
                    return TagPc::Twice(Pc(first), Pc(line));
                }
                _ => {
                    pc = Some(line);
                    tag_found_on_dead_code = instr.dead;
                }
            }
        }
    }
    match pc {
        Some(line) => TagPc::At(Pc(line)),
        None => TagPc::NotFound,
    }
}

/// Replaces: e030_tagToLCCR
///
/// Which loop counter `tag`'s `MVLOOPCNT` was given, by walking the nesting the program opens and
/// closes: `MVLOOPCNT` deepens it, a set `be` closes one, and a jump over either unbalances both.
///
/// ⛔ TRAP: DEAD CODE MOVES THE **SHADOW** INDEX ONLY, so a tag on a dead loop still reports the
/// slot that loop would have taken while `loopNesting` — the live nesting — never sees it.
/// ⛔ TRAP: A DEAD DUPLICATE IS SILENTLY IGNORED. The reference's inner `if (!deadCode_)` has no
/// `else`, so a second dead carrier neither aborts nor overwrites the index already found.
/// ⛔ AND A `pc_target` THAT IS NOT A STRING REGISTERS NO JUMP HERE, where `asString` aborts.
#[must_use]
pub fn tag_to_lccr(prog: &[Instruction], tag: &str) -> TagLccr {
    let mut lccr_idx: i32 = -1;
    let mut loop_nesting: i32 = -1;
    let mut shadow_lccr_idx: i32 = -1;
    let mut tags: BTreeSet<&str> = BTreeSet::new();
    let mut pending_jumps: BTreeMap<&str, i32> = BTreeMap::new();
    let mut disable_loop_tag = false;
    let mut tag_found_on_dead_code = false;

    for instr in prog {
        if instr.has_tag() {
            tags.insert(instr.tag_str());
            // ⭐ A JUMP LANDING HERE HAVING CROSSED AN UNBALANCED LOOP IS WHAT DISABLES THE TAG:
            // the pending depth is the nesting the jump straddled, and only a non-zero one counts.
            if let Some(straddled) = pending_jumps.remove(instr.tag_str()) {
                if straddled != 0 {
                    disable_loop_tag = true;
                }
            }
        }
        if instr.opcode == InstOpCode::MVLOOPCNT {
            if instr.dead {
                shadow_lccr_idx += 1;
            } else {
                loop_nesting += 1;
                shadow_lccr_idx = loop_nesting;
                for straddled in pending_jumps.values_mut() {
                    *straddled += 1;
                }
            }
            if instr.tag_str() == tag {
                if lccr_idx != -1 && !tag_found_on_dead_code {
                    if !instr.dead {
                        return TagLccr::Twice;
                    }
                } else if disable_loop_tag {
                    return TagLccr::AfterAlteringJump;
                } else {
                    lccr_idx = shadow_lccr_idx;
                    tag_found_on_dead_code = instr.dead;
                }
            }
        } else if instr.tag_str() == tag {
            // Most likely a loop that progpatch turned into a JCMP.
            return TagLccr::NotOnLoop(LccrIndex(shadow_lccr_idx + 1));
        } else if instr.opcode == InstOpCode::RETURN && !instr.dead {
            if let Some(field) = instr.field(OperandField::Subroutine) {
                if is_set(field, "yes") {
                    disable_loop_tag = true;
                }
            }
        }

        // A jump can leave `MVLOOPCNT`/`be` unbalanced, so its target is remembered at depth 0.
        // ⛔ `MVLOOPCNT`'s OWN `imm` IS A TARGET WHEN IT HOLDS A TAG — one position, two names.
        let target = instr
            .field(OperandField::PcTarget)
            .or_else(|| {
                (instr.opcode == InstOpCode::MVLOOPCNT)
                    .then(|| instr.field(OperandField::Imm).filter(|imm| imm.is_tag()))
                    .flatten()
            })
            .and_then(OperandValue::as_string);
        if let Some(target) = target {
            if tags.contains(target) {
                disable_loop_tag = true;
            }
            pending_jumps.insert(target, 0);
        }

        if let Some(be) = instr.field(OperandField::Be) {
            if is_set(be, "be") {
                if instr.dead {
                    shadow_lccr_idx -= 1;
                } else {
                    loop_nesting -= 1;
                    shadow_lccr_idx = loop_nesting;
                    for straddled in pending_jumps.values_mut() {
                        *straddled -= 1;
                    }
                }
            }
        }
    }
    if lccr_idx < 0 {
        TagLccr::NoLoop
    } else {
        TagLccr::At(LccrIndex(lccr_idx))
    }
}

/// A FLAG FIELD THAT IS ON — `((isBool() || isInt()) && asInt() == 1) || (isDescriptive() && asString() == name)`,
/// the shape both `be` and `subroutine` are tested with (`progir.cpp:773-777`, `:798-800`).
///
/// ⛔ THE ENUM SPELLING IS THE FIELD'S OWN NAME, not a shared `"yes"`: `be` reads `"be"`.
fn is_set(field: &OperandValue, spelling: &str) -> bool {
    ((field.is_bool() || field.is_int()) && field.as_int() == Some(1))
        || (field.is_descriptive() && field.as_string() == Some(spelling))
}

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

    /// A tagged instruction, `dead` or not, with no other field.
    fn tagged(opcode: InstOpCode, tag: Option<&str>, dead: bool) -> Instruction {
        Instruction {
            opcode,
            tag: tag.map(str::to_owned),
            dead,
            ..instr(Vec::new())
        }
    }

    /// e031: a DEAD first carrier is replaced by the live one rather than reported as a duplicate,
    /// and only two LIVE carriers are the *"found in lines"* refusal.
    #[test]
    fn a_dead_carrier_yields_to_a_live_one_and_two_live_ones_collide() {
        let dead_then_live = [
            tagged(InstOpCode::NOP, Some("t"), true),
            tagged(InstOpCode::NOP, None, false),
            tagged(InstOpCode::NOP, Some("t"), false),
        ];
        assert_eq!(tag_to_pc(&dead_then_live, "t"), TagPc::At(Pc(2)));
        assert_eq!(tag_to_pc(&dead_then_live, "u"), TagPc::NotFound);

        let twice = [
            tagged(InstOpCode::NOP, Some("t"), false),
            tagged(InstOpCode::NOP, Some("t"), false),
        ];
        assert_eq!(tag_to_pc(&twice, "t"), TagPc::Twice(Pc(0), Pc(1)));
    }

    /// e030: nesting is what fixes the index — and a DEAD loop moves only the shadow, so it does not
    /// consume the slot the live loop after it takes.
    #[test]
    fn a_dead_loop_does_not_consume_the_slot_the_live_one_takes() {
        let nested = [
            tagged(InstOpCode::MVLOOPCNT, None, false),
            tagged(InstOpCode::MVLOOPCNT, Some("inner"), false),
        ];
        assert_eq!(tag_to_lccr(&nested, "inner"), TagLccr::At(LccrIndex(1)));
        assert_eq!(tag_to_lccr(&nested, "absent"), TagLccr::NoLoop);

        let after_dead = [
            tagged(InstOpCode::MVLOOPCNT, None, true),
            tagged(InstOpCode::MVLOOPCNT, Some("t"), false),
        ];
        // ⛔ `At(0)`, NOT `At(1)`: the dead loop never touched `loopNesting`.
        assert_eq!(tag_to_lccr(&after_dead, "t"), TagLccr::At(LccrIndex(0)));

        // A tag on anything but an `MVLOOPCNT` reports the slot the NEXT loop would take.
        let not_a_loop = [tagged(InstOpCode::NOP, Some("t"), false)];
        assert_eq!(
            tag_to_lccr(&not_a_loop, "t"),
            TagLccr::NotOnLoop(LccrIndex(0))
        );
    }
}
