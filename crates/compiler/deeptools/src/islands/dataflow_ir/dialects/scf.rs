//! UPSTREAM `scf` — the one structured-control-flow op an emitted program contains.
//!
//! Not one of the scheduler's own dialects. A region here holds [`super::Op`]s of any dialect, which
//! is why the field types name the outer enum rather than this module's.

use std::fmt::Write as _;

use crate::islands::dataflow_ir::dialects::Val;
use crate::islands::dataflow_ir::print;

/// ONE `scf` OPERATION.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Op {
    /// `scf.if %cond { .. } else { .. }` - an undecided branch, BOTH arms in one op.
    ///
    /// (E) ONE OP WITH TWO REGIONS, WHICH IS WHAT IBM EMITS. `dcc/test/PT/fp8-bmm.mlir:1072-1101`
    /// is `scf.if %923 { .. } else { .. }`, and the `else` region there holds a whole further
    /// condition chain — so nesting arms inside an `else` is the vendored shape, not an
    /// optimisation.
    ///
    /// ⛔⛔ THE NEGATED-SIBLING FORM CRASHED THE BACKEND. An earlier version emitted the two arms as
    /// separate `scf.if`s on a predicate and its negation, reasoning that mutual exclusion kept a
    /// value produced in one out of the other's scope — which MLIR's own region scoping already
    /// guarantees. The negation is an `arith.xori %cond, true`, and once the predicate was a
    /// COMPOUND one (`arith.andi` of two positions, which `ddl.condition_and` at `bmm.ddl:234`
    /// genuinely asks for) `dbo-opt` converted the shared `andi` into a `sentient.if`, destroyed it
    /// converting the guard, and died on the `xori` still holding it: "'sentient.if' op operation
    /// destroyed but still has uses". With one op and two regions there is no negation to dangle.
    If {
        /// The predicate.
        cond: Val,
        /// The `then` region.
        body: Vec<super::Op>,
        /// The `else` region. Empty prints no `else` at all, which is the one-armed branch.
        else_body: Vec<super::Op>,
    },

    /// `scf.yield %operands` — what an `affine.yield` becomes (`AffineToStandard.cpp:48`).
    Yield {
        /// The carried values, which are the `affine.yield`'s own.
        operands: Vec<Val>,
    },

    /// `scf.parallel` — ⛔ PRESENT BECAUSE ONE REWRITE ASKS ABOUT IT, not because this crate emits it.
    ///
    /// ⛔⛔ `AffineYieldOpLowering` DECLINES WHEN ITS PARENT IS THIS OP (`AffineToStandard.cpp:42-46`),
    /// under the comment *"Terminator is rewritten as part of the 'affine.parallel' lowering
    /// pattern."* So the parent's kind is an INPUT to that rewrite, and without this variant the
    /// question cannot be asked and the rewrite would fire where the reference stands back — producing
    /// two rewrites of one terminator.
    Parallel {
        /// The induction variables, one per parallel dimension.
        ivs: Vec<Val>,
        /// The body.
        body: Vec<super::Op>,
    },
}

/// ONE `scf` OP AS TEXT. The caller has already indented the opening line.
pub(crate) fn emit(out: &mut String, op: &Op, depth: usize) {
    match op {
        Op::If {
            cond,
            body,
            else_body,
        } => {
            let _ = writeln!(out, "scf.if {} {{", print::val(*cond));
            for inner in body {
                print::emit(out, inner, depth + 1);
            }
            print::indent(out, depth);
            if else_body.is_empty() {
                out.push_str("}\n");
            } else {
                out.push_str("} else {\n");
                for inner in else_body {
                    print::emit(out, inner, depth + 1);
                }
                print::indent(out, depth);
                out.push_str("}\n");
            }
        }
        Op::Yield { operands } => {
            if operands.is_empty() {
                out.push_str("scf.yield\n");
            } else {
                let _ = writeln!(out, "scf.yield {}", print::vals(operands));
            }
        }
        Op::Parallel { ivs, body } => {
            let _ = writeln!(out, "scf.parallel ({}) {{", print::vals(ivs));
            for inner in body {
                print::emit(out, inner, depth + 1);
            }
            print::indent(out, depth);
            out.push_str("}\n");
        }
    }
}
