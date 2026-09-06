//! THE ISLAND AS MLIR TEXT — the one place SentientIR becomes characters.
//!
//! ⭐ DETERMINISTIC BY CONSTRUCTION, for the same reason the rung below is: names come from a
//! counter and attributes are written in a fixed order, so two emissions of one program are
//! byte-identical.
//!
//! ⛔⛔ AND THE FIXED ORDER IS **ALPHABETICAL**, NOT DECLARATION ORDER. MLIR's
//! `printOptionalAttrDict` sorts, and all nineteen of the dialect's custom printers delegate to it
//! (`SentientOps.cpp:2219-2229` and the same three lines for ternary and unary). This rung's only
//! oracle is a byte comparison against a reference dump, so an attribute order that merely parses is
//! worth nothing.

use std::fmt::Write as _;

use crate::islands::sentient::dialects::{
    Op, affine, agen, arith, dataflow, scf, sentient, vectorchain,
};
use crate::islands::sentient::{Program, Run};

/// ⭐ THE VALUE SPELLINGS ARE THE LOWER RUNG'S, re-exported rather than restated.
///
/// A Sentient module is the DataflowIR module rewritten in place — one SSA numbering throughout — so
/// `%14` means the same thing on both rungs and there is exactly one function that writes it.
pub(crate) use crate::islands::dataflow_ir::print::{val, vals};

/// A WHOLE RUN AS ONE MLIR MODULE.
///
/// ⛔ THE MODULE FRAMING IS THE SAME AS THE RUNG BELOW'S, and that is a fact about the *consumer*,
/// not a convenience: `dbo-run-program-pipelines` runs the dcc pipeline over the inner module of each
/// named program module (`dbo/src/Pipeline/RunProgramPipelines.cpp:199-211`), and it looks for that
/// shape whichever rung the body has reached.
#[must_use]
pub fn run<A, M, W>(run: &Run<A, M, W>) -> String
where
    A: crate::arch::Arch,
    M: crate::model::Model,
    W: crate::workload::Workload,
{
    let mut out = String::new();
    out.push_str("module {\n");
    out.push_str("  module {\n");
    let _ = writeln!(out, "    func.func @{}() {{", run.kernel);
    for program in &run.programs {
        let _ = writeln!(out, "      call @{}() : () -> ()", program.name);
    }
    out.push_str("      return\n    }\n");
    for program in &run.programs {
        let _ = writeln!(out, "    func.func private @{}()", program.name);
    }
    out.push_str("  }\n");
    for program in &run.programs {
        program_module(&mut out, program);
    }
    out.push_str("}\n");
    out
}

/// One named module holding a program's SentientIR.
fn program_module<A, M, W>(out: &mut String, program: &Program<A, M, W>)
where
    A: crate::arch::Arch,
    M: crate::model::Model,
    W: crate::workload::Workload,
{
    let _ = writeln!(out, "  module @{} {{", program.name);
    let _ = writeln!(out, "    func.func private @{}() {{", program.name);
    for op in &program.body {
        emit(out, op, 3);
    }
    out.push_str("      return\n    }\n  }\n");
}

/// Two spaces per level.
fn indent(out: &mut String, depth: usize) {
    for _ in 0..depth {
        out.push_str("  ");
    }
}

/// ONE OP AS TEXT, dispatched to its dialect's own printer.
///
/// ⭐ THE CALLER INDENTS AND THE DIALECT WRITES BARE — the same contract the rung below uses
/// ([`crate::islands::dataflow_ir::print::emit`]), so a dialect printer shared between the two
/// cannot disagree about whitespace.
///
/// ⛔ NO `_` ARM. A new dialect reaching this rung is a build error.
pub(crate) fn emit(out: &mut String, op: &Op, depth: usize) {
    indent(out, depth);
    match op {
        Op::Sentient(op) => sentient::emit(out, op, depth),
        Op::Arith(op) => arith::emit(out, op),
        Op::Scf(op) => scf::emit(out, op, depth),
        Op::Affine(op) => affine::emit(out, op, depth),
        Op::Dataflow(op) => dataflow::emit(out, op, depth),
        Op::Agen(op) => agen::emit(out, op, depth),
        Op::VectorChain(op) => vectorchain::emit(out, op),
    }
}
