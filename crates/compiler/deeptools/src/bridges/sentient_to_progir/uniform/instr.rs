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
use crate::islands::progir::{OpCode, OperandField};

/// ONE INSTRUCTION BEFORE ITS UNITS ARE SPLIT APART — `UniformInstrInfo`
/// (`dcc/src/Conversion/SentientToProgIR/UniformInstrAndBlock.hpp:88-166`).
///
/// ⛔⛔ `UniformInstrInfo()` LEAVES `instn_` UNINITIALISED (`:146`) — unlike `InstrInfo`, which
/// defaults it to `NumOpCodes`. Every constructor in this span sets it immediately, so the opcode is
/// required here and that state has no spelling.
///
/// ⚠️ `operand_map_` IS NOT HERE YET — it is `e072`/`e073`'s (`addEntryToOperandMap`), and adding it
/// with no writer would be a field nothing fills.
#[derive(Debug, Clone, PartialEq)]
pub struct UniformInstrInfo {
    /// `instn_`.
    pub opcode: OpCode,
    /// `instFields_` — the fields common to all units, in field order.
    pub common_fields: Vec<(OperandField, Operand)>,
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

// crustify:todo: e017_createNOPInstr
// crustify:todo: e018_getUniformizedInstr
// crustify:todo: e019_getRegularInstr
// crustify:todo: e020_getCommonField
// crustify:todo: e021_setCommonField
// crustify:todo: e022_hasCommonField
// crustify:todo: e071_createJmpInstr
// crustify:todo: e072_addEntryToOperandMap
// crustify:todo: e073_addEntryToOperandMap
