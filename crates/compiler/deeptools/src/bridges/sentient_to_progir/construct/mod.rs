//! `ConstructProgIRHelper.cpp` — 46 units, one submodule per family.

use crate::islands::progir::ty::{Operand, OperandValue};

/// ONE FIELD'S VALUE AS AN ISA FIELD NAME — `OperandAttr(text, DESCRIPTIVE)`, which is what every
/// `setCommonField` in this file that takes a string builds.
pub(crate) fn descriptive(name: &str) -> Operand {
    Operand::every(OperandValue::Descriptive(name.to_owned()))
}

/// ONE FIELD'S VALUE AS A BRANCH TARGET'S NAME — `OperandAttr(text, INSTR_TAG)`, which `tagToPC`
/// resolves to a program counter once every label's address is known.
pub(crate) fn instr_tag(label: &str) -> Operand {
    Operand::every(OperandValue::InstrTag(label.to_owned()))
}

/// ONE FIELD'S VALUE AS A SYMBOL'S ID — `OperandAttr(id, VARIABLE_SYMBOL)`, an integer the correction
/// table later substitutes.
pub(crate) const fn variable_symbol(id: i64) -> Operand {
    Operand::every(OperandValue::VariableSymbol(id))
}

/// ONE FIELD'S VALUE AS A FLAG — the implicit `OperandAttr(bool)` behind `setCommonField("isimm", …)`.
pub(crate) const fn boolean(flag: bool) -> Operand {
    Operand::every(OperandValue::Boolean(flag))
}

/// ONE FIELD'S VALUE AS AN INTEGER — the implicit `OperandAttr(int)` behind the reference's
/// `setCommonField("imm", 3)`.
pub(crate) const fn int(value: i64) -> Operand {
    Operand::every(OperandValue::Int(value))
}

/// THE COMPUTE INSTRUCTIONS — FMA, binary, unary and ternary, and the operand plumbing that
pub mod compute;

/// THE MASK, SPLAT AND SAMV INSTRUCTIONS — set-dest-mask, set-dest, splat, splat-pad, and the
pub mod mask_and_splat;

/// THE OPAQUE-TEMPLATE INSTRUCTION — `ConstructOpaqueInstr` and the string trim it uses.
pub mod opaque;

/// WHAT MUST BE IN A REGISTER BEFORE ANYTHING READS IT — the reg-init accumulation and the
pub mod reg_init;

/// THE SCALAR INSTRUCTIONS — the jumps, the compares, the adds and subs, and the sync.
pub mod scalar;

/// THE TRANSFER INSTRUCTIONS — the L3 and non-L3 loads and stores, the burst normalisation and
pub mod transfer;
