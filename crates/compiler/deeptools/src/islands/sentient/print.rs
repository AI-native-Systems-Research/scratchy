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
    LocalRegion, Op, UniformRegions, affine, agen, arith, dataflow, scf, sentient, symbol, uniform,
    vector, vectorchain,
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
    // The preamble binds the units and views; the program units then run on them.
    for op in &program.preamble {
        emit(out, op, 3);
    }
    // ⭐ ONE `dataflow.program_unit` PER UNIT, which is the shape 657 of IBM's 668 SentientIR
    // expectations carry — see [`crate::islands::sentient::ProgramUnit`].
    for unit in program.units.iter() {
        let precision = unit.precision.map_or_else(String::new, |p| {
            format!(" {{precision = \"{}\"}}", p.spelling())
        });
        let _ = writeln!(
            out,
            "      dataflow.program_unit {}{precision} : {{",
            vals(&unit.on.vals())
        );
        for op in &unit.body {
            emit(out, op, 4);
        }
        out.push_str("      }\n");
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
        Op::AffineFor(loop_op) => {
            out.push_str(&affine::for_header(
                loop_op.iv,
                loop_op.lo,
                loop_op.hi,
                &loop_op.carried,
            ));
            for inner in &loop_op.body {
                emit(out, inner, depth + 1);
            }
            indent(out, depth);
            out.push_str(&affine::for_footer(loop_op.dbg_name.as_deref()));
        }
        Op::Vector(op) => vector::emit(out, op),
        Op::Dataflow(op) => dataflow::emit(out, op, depth),
        Op::Agen(op) => agen::emit(out, op, depth),
        Op::VectorChain(op) => vectorchain::emit(out, op),
        Op::Symbol(op) => symbol::emit(out, op),
        Op::Uniform(op) => uniform::emit(out, op, depth),
        // ⭐ THROUGH [`uniform`]'S OWN HEADER FUNCTIONS, so a local region prints identically
        // whichever rung its body has reached — the same arrangement `Op::AffineFor` above has with
        // [`affine::for_header`]. Only the BODY loop differs, because only the body's type does.
        Op::UniformRegions(regions) => {
            out.push_str(&match regions {
                UniformRegions::UniformizeRegions { results, .. } => {
                    uniform::uniformize_header(results)
                }
                UniformRegions::EqualizePattern { .. } => uniform::equalize_header(),
            });
            for region in regions.regions() {
                local_region(out, region, depth + 1);
            }
            indent(out, depth);
            out.push('}');
            // ⭐ THE ONLY ATTRIBUTES THIS OP CARRIES, and only once `addResultToYield` has written
            // them — see [`sentient::uniformize_result_attrs`].
            if let UniformRegions::UniformizeRegions { yielded, .. } = regions {
                out.push_str(&sentient::uniformize_result_attrs(yielded));
            }
            out.push('\n');
        }
    }
}

/// ONE LOCAL REGION OF A `uniform.uniformize_regions` AT THIS RUNG — its header, its body and its
/// closing brace.
///
/// ⭐ THE TERMINATOR PRINTS. `printRegion(region, /*printEntryBlockArgs=*/false,
/// /*printBlockTerminators=*/true)` (`Uniform.cpp:112`) — the region argument is already in the
/// header, the `uniform.yield` is not.
fn local_region(out: &mut String, region: &LocalRegion, depth: usize) {
    indent(out, depth);
    out.push_str(&uniform::region_header(region.arg, &region.units));
    for inner in &region.body {
        emit(out, inner, depth + 1);
    }
    indent(out, depth);
    out.push_str("}\n");
}

#[cfg(test)]
mod unit_tests {
    use crate::formats::Bits;
    use crate::islands::dataflow_ir::dialects::{self as lower, Val, uniform as lower_uniform};
    use crate::islands::sentient::dialects::{LocalRegion, Op, UniformRegions, sentient, uniform};
    use crate::islands::sentient::print::emit;

    /// ⭐ THE TWO RUNGS CANNOT DISAGREE ABOUT A CHARACTER — the same op, once with a lower-rung body
    /// and once with this rung's, printed through the same three header functions.
    ///
    /// The shape is `dcc/test/Transform/FlatteningLocalRegions/flatten_local_region.mlir:86-88`, the
    /// one the rung below asserts byte for byte.
    #[test]
    fn a_local_region_prints_the_same_at_both_rungs() {
        let here = Op::UniformRegions(UniformRegions::UniformizeRegions {
            regions: vec![LocalRegion {
                arg: Val(1),
                units: vec![Val(0), Val(2)],
                body: vec![Op::Uniform(uniform::Op::Yield {
                    operands: Vec::new(),
                })],
            }],
            results: Vec::new(),
            yielded: Vec::new(),
        });
        let below = lower::Op::Uniform(lower_uniform::Op::UniformizeRegions {
            regions: vec![lower_uniform::LocalRegion {
                arg: Val(1),
                units: vec![Val(0), Val(2)],
                body: vec![lower::Op::Uniform(lower_uniform::Op::Yield {
                    operands: Vec::new(),
                })],
            }],
            results: Vec::new(),
        });

        let mut got = String::new();
        emit(&mut got, &here, 0);
        let mut want = String::new();
        crate::islands::dataflow_ir::print::emit(&mut want, &below, 0);
        assert_eq!(got, want);

        // AND A `sentient.*` OP IN THE BODY IS WHY THIS VARIANT EXISTS: it has no lower-rung twin.
        let sunk = Op::UniformRegions(UniformRegions::EqualizePattern {
            regions: vec![LocalRegion {
                arg: Val(1),
                units: vec![Val(0)],
                body: vec![Op::Sentient(sentient::Op::ScalarCopy {
                    input: Val(3),
                    result: Val(4),
                    reg: sentient::Reg {
                        locale: sentient::RegType::Ebr,
                        index: None,
                    },
                    element_size: None,
                    program_header: false,
                })],
            }],
        });
        let mut got = String::new();
        emit(&mut got, &sunk, 0);
        assert!(
            got.starts_with(
                "uniform.equalize_pattern {\n  (%1 -> %0){\n    %4 = sentient.scalar_copy"
            ),
            "{got}"
        );
    }

    /// ⭐ ONE `element_sizes` ENTRY PER RESULT ON AN `if`, WITH NO LEADING `bound` SLOT — the array
    /// its readers index by the plain result number (`LowerSentientHelper.cpp:806`), written by
    /// `AddressRegisterPrecisionAssignment` (e554).
    #[test]
    fn an_if_prints_one_element_size_per_result() {
        let reg = sentient::Reg {
            locale: sentient::RegType::Lar,
            index: None,
        };
        let op = Op::Sentient(sentient::Op::If {
            predicate: sentient::CmpPredicate::Eq,
            lhs: Val(1),
            rhs: Val(2),
            yielded: vec![
                sentient::Yielded {
                    result: Val(3),
                    reg,
                    element_size: Some(Bits(16)),
                },
                sentient::Yielded {
                    result: Val(4),
                    reg,
                    element_size: None,
                },
            ],
            dbg_name: None,
            then_body: Vec::new(),
            else_body: Vec::new(),
        });
        let mut got = String::new();
        emit(&mut got, &op, 0);
        // ⭐ THE ISLAND'S OWN SPELLING OF THE ARRAY, THE SAME ONE `sentient.for` PRINTS — see
        // `sentient::size_array`, whose `-1` is an absent slot.
        assert!(got.contains("element_sizes = [16, -1]"), "{got}");
    }
}
