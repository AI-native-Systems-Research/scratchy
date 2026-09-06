//! THE ISLAND AS TEXT — ⛔ TWO OUTPUTS, NOT ONE, AND THEY ARE NOT ALTERNATIVES.
//!
//! ProgIR is the one rung whose content is *not* in the MLIR module, so it prints twice:
//!
//! * [`module`] writes the MLIR side — `init.smc`, and after codegen `init.bin`. Twelve lines.
//! * the senprog rendering writes the instructions, which is what `Dpc::convertIr2Senprog`
//!   (`sys-arch-spec/dpc/dpc.cpp:615`) produces. ⚠️ **NOT WRITTEN YET** — it needs the per-unit ISA
//!   to turn an [`super::OpCode`] number into `PTOP_IMA8`, and that table is owed; see [`super::OpCode`].
//!
//! ⭐⭐ AND SENPROG IS A **FORMAT, NOT A RUNG**. `SentientToProgIR.cpp:703-737` selects it inside
//! `if (dumpProgIR.getValue())`, as one of three renderings of the same `progstateinfo_`:
//! `kSenProg` → `convertIr2Senprog`, `kSmc` → `convertIr2SMC`, else `psinfo.print`. What dip actually
//! consumes is the structure — `GenerateInitPacket.cpp:48` takes
//! `std::map<int, ProgramAndStateInfo> &progstateinfo` — never the senprog text.
//!
//! ⛔ WHICH MEANS SENPROG IS AN **ORACLE**, NOT AN INTERCHANGE. It is worth emitting precisely because
//! the reference emits it too, so ours can be diffed against theirs; nothing downstream needs it.

use std::fmt::Write as _;

use crate::islands::progir::dialects::{Op, init};

/// ⭐ THE VALUE SPELLINGS ARE THE LADDER'S, re-exported — see
/// [`crate::islands::sentient::print`] for why there is only one of these functions.
pub(crate) use crate::islands::dataflow_ir::print::{val, vals};

/// ONE OP AS TEXT, dispatched to its dialect.
///
/// ⛔ NO `_` ARM.
pub(crate) fn emit(out: &mut String, op: &Op) {
    match op {
        Op::Init(op) => init::emit(out, op),
    }
}

/// THE MLIR SIDE OF ONE PROGRAM — ⛔ TWELVE LINES, and that is correct rather than empty.
#[must_use]
pub fn module(name: &str, body: &[Op]) -> String {
    let mut out = String::new();
    out.push_str("module {\n");
    let _ = writeln!(out, "  func.func @{name}() attributes {{grid = [1]}} {{");
    for op in body {
        out.push_str("    ");
        emit(&mut out, op);
    }
    out.push_str("    return\n  }\n}\n");
    out
}
