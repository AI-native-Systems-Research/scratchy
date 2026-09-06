//! `InitOps.td` — WHAT IS LEFT OF THE MLIR MODULE ONCE THE PROGRAM IS REAL.
//!
//! Authority: `dialects/Init/InitOps.td` on the pod. Three operations.
//!
//! ⭐⭐ THIS DIALECT IS THE WHOLE MLIR SIDE OF THE PROGIR RUNG, AND THAT IS MEASURED. Running a real
//! granite program through all 128 passes and dumping after `SentientToProgIRPass` leaves exactly:
//!
//! ```text
//! module {
//!   func.func @dataflowProgram() attributes {grid = [1]} {
//!     init.smc {name = "default_prog_name"}
//!     return
//!   }
//! }
//! ```
//!
//! Twelve lines. The instructions are NOT in the module — they went into the `ProgramAndStateInfo`
//! the pass filled in ([`super::super::Program`]), and `init.smc` is the reference the module keeps
//! to them.
//!
//! ⛔ SO AN EMPTY-LOOKING MODULE HERE IS NOT A FAILURE, and reading it as one cost real time: a green
//! run whose ProgIR dump was `module { init.smc … }` was recorded as "the ProgIR is empty" when it was
//! the expected shape. What tells you whether a program exists is the instruction count per unit, not
//! the module's length.

use std::fmt::Write as _;

use crate::islands::progir::dialects::Val;

/// ONE `init` OPERATION.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Op {
    /// `init.smc` — names the program the compiled instructions belong to (`InitOps.td:32-45`).
    ///
    /// ⭐ `default_prog_name` IS WHAT THE REFERENCE EMITS when nothing set one, so it is a real
    /// value rather than a placeholder to be filled in.
    Smc {
        /// `$name`.
        name: String,
    },

    /// `init.bin` — the generated packet, as a value (`InitOps.td:50-81`).
    ///
    /// ⭐⭐ IT PRODUCES THE BYTES RATHER THAN NAMING A FILE, and that is what puts the host's plan in
    /// order: `dbo-place-programs` writes an `explan.host_to_device` that takes this value as its
    /// operand, so the transfer cannot be written before what produced it.
    ///
    /// ⛔ THE SYMBOL IDS ARE THE **CORRECTION TABLE'S**, NOT EVERYTHING THE MODULE DECLARES. An
    /// argument already settled into the packet needs nothing written, so it does not appear here.
    Bin {
        /// `$symbol_definitions` — the operands the packet is a function of.
        symbol_definitions: Vec<Val>,
        /// `$packet` — the value it binds.
        result: Val,
        /// `name` — which program's packet.
        name: String,
        /// `size` — ⭐ HOW MANY BYTES, and the result's tensor type states the same count, so the two
        /// cannot disagree.
        size: u64,
        /// `symbol_ids` — what still has to be written in.
        symbol_ids: Vec<i64>,
    },

    /// `init.symloc` — a named location within the packet (`InitOps.td:91-104`).
    SymLoc {
        /// `$name`.
        name: String,
    },
}

/// ONE `init` OP AS TEXT. The caller has already indented.
pub(crate) fn emit(out: &mut String, op: &Op) {
    match op {
        Op::Smc { name } => {
            let _ = writeln!(out, "init.smc {{name = \"{name}\"}}");
        }
        Op::SymLoc { name } => {
            let _ = writeln!(out, "init.symloc {{name = \"{name}\"}}");
        }
        Op::Bin {
            symbol_definitions,
            result,
            name,
            size,
            symbol_ids,
        } => {
            let ids: Vec<String> = symbol_ids.iter().map(i64::to_string).collect();
            let operands = if symbol_definitions.is_empty() {
                String::new()
            } else {
                format!("({})", crate::islands::progir::print::vals(symbol_definitions))
            };
            // ⭐ THE RESULT'S TYPE RESTATES `size`, as the reference writes it:
            // `%bin = init.bin(%0) {name = "sdsc_0", size = 2688, symbol_ids = [-5]} : tensor<2688xi8>`
            let _ = writeln!(
                out,
                "{} = init.bin{operands} {{name = \"{name}\", size = {size}, symbol_ids = [{}]}} : tensor<{size}xi8>",
                crate::islands::progir::print::val(*result),
                ids.join(", ")
            );
        }
    }
}
