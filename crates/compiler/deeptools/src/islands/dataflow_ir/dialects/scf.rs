//! UPSTREAM `scf` — the one structured-control-flow op an emitted program contains.
//!
//! Not one of the scheduler's own dialects. A region here holds [`super::Op`]s of any dialect, which
//! is why the field types name the outer enum rather than this module's.

use std::fmt::Write as _;

use crate::islands::dataflow_ir::dialects::Val;
use crate::islands::dataflow_ir::dialects::affine::Carried;
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
        /// The `else` region.
        ///
        /// ⛔⛔ EMPTY MEANS **NO BLOCK**, AND THAT IS A DIFFERENT OP FROM A BLOCK HOLDING ONLY A
        /// TERMINATOR. `getRegions()[1].empty()` is the test `createDummyYieldInElseReg` (entry 096)
        /// guards on, and pushing a bare `scf.yield` into the region is that function's ENTIRE effect
        /// (`Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:383-396`). MLIR prints the
        /// difference: no block prints no `else` at all, a block with an elided terminator prints
        /// `} else {` and an empty pair of braces —
        /// ```text
        /// scf.if %53 {
        ///   ..
        /// } else {
        /// }
        /// ```
        /// (`dcc/test/PT/issue-236.mlir:65-71`). A `Vec` that flattened the two states would make
        /// entry 096 a function with no observable result.
        else_body: Vec<super::Op>,
    },

    /// `%r = scf.for %i = %lo to %hi step %st iter_args(%a = %init) -> (index) { .. }` — THE
    /// COUNTED LOOP WHOSE BOUNDS ARE VALUES.
    ///
    /// # ⛔⛔ THE WHOLE DIFFERENCE FROM [`super::affine::Op::For`] IS WHERE THE BOUNDS LIVE
    ///
    /// An `affine.for` states its bounds as affine maps over the enclosing loops' induction
    /// variables — literals, in every loop this crate builds. An `scf.for` takes them as OPERANDS,
    /// so a bound may be any SSA value at all, including an `arith.select` between two constants.
    /// That is precisely the shape `TransformLoopToLegalizeForSentientLowering` exists to remove:
    /// its header says it *"converts `scf.for` with conditional upper bounds into `scf.if` + affine
    /// loops so agen iterators are affine"* (`:8-16`), because the address generator can only walk
    /// an affine iteration space.
    ///
    /// ⛔ SO THIS IS AN **INPUT** OP, NOT ONE THE BRIDGE EMITS. It is here because the pass's input
    /// contains it — `dcc/test/Transform/TransformLoopToLegalizeForSentientLowering/scf_loop_with_result.mlir:32`
    /// is `%11 = scf.for %arg4 = %c0 to %10 step %c1 iter_args(%arg5 = %arg3) -> (index)` with `%10`
    /// an `arith.select` — and without it `transformSCFToAffineLoop` has no argument to be given and
    /// the pass could not be ported at all.
    ///
    /// ⭐ THE STEP IS A FIELD BECAUSE THE PASS BRANCHES ON IT. `transformSCFToAffineLoop` transforms
    /// only `lbound == 0 && step == 1` (`:105`) and reports failure otherwise, so a representation
    /// that assumed a unit step could not express the case it declines.
    ///
    /// ⭐ AND A SECOND PASS CLASSIFIES ITS INPUT BY IT. `performFullUnroll` dispatches over exactly
    /// two loop kinds (`LoopUnrollForShuffleOp.cpp:141-149`) and a candidate it cannot hold is a
    /// candidate the dispatch cannot be written total over — see
    /// [`crate::bridges::dataflow_ir_to_sentient::tf_loop_unroll_for_shuffle_op::Loop`]. Where an
    /// `affine.for` lets affine's own analysis derive the trip count from its bound maps (`:162-165`),
    /// this loop's three bounds are SSA operands and the count has to be reconstructed by asking each
    /// of them for its defining constant (`:167-182`) — which is why one overload is one line and the
    /// other needs a helper. Holding the bounds as [`Val`] is what makes that reconstruction
    /// expressible.
    For {
        /// The induction variable it binds — the region's first argument.
        iv: Val,
        /// `$lowerBound`, an operand.
        lo: Val,
        /// `$upperBound`, an operand.
        hi: Val,
        /// `$step`, an operand.
        step: Val,
        /// THE VALUES IT CARRIES ACROSS ITERATIONS — empty for the plain counted loop.
        ///
        /// ⭐ THE SAME THREE-VALUE RECORD AN `affine.for` USES, and deliberately the same type:
        /// `transformSCFToAffineLoop` hands `scf_for.getInits()` straight to
        /// `affine::AffineForOp::create` (`:107-108`), so the two ops' carried lists are the same
        /// list and a second type for it would be two records of one fact.
        carried: Vec<Carried>,
        /// The body, ending in an [`Op::Yield`] once anything is carried.
        body: Vec<super::Op>,
        /// `{dbgName = ".."}` — see [`super::affine::Op::For`]'s field of the same name for why an
        /// emitter must carry it.
        dbg_name: Option<String>,
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

/// ONE REGION OF AN `scf.if`, WITH ITS OPERAND-LESS TERMINATOR ELIDED.
///
/// ⛔ MLIR ELIDES IT BECAUSE THIS OP BINDS NOTHING. `SCF.cpp`'s printer passes
/// `printBlockTerminators = !getResults().empty()`, and [`Op::If`] carries no result list at all —
/// deliberately, see its note — so the `scf.yield` a region ends with is never printed. That is what
/// makes `} else {` followed by a bare `}` the reference's own text
/// (`dcc/test/PT/issue-236.mlir:70-71`) rather than a region printed with a stray terminator.
///
/// ⭐ ELIDED, NOT DROPPED. The op stays in the region: [`super::regions`] and every walk still see
/// it, and entry 096's whole job is to put one there.
fn region(out: &mut String, ops: &[super::Op], depth: usize) {
    for (n, inner) in ops.iter().enumerate() {
        let last = n + 1 == ops.len();
        if last
            && matches!(inner, super::Op::Scf(Op::Yield { operands }) if operands.is_empty())
        {
            continue;
        }
        print::emit(out, inner, depth + 1);
    }
}

/// ONE `scf` OP AS TEXT. The caller has already indented the opening line.
pub(crate) fn emit(out: &mut String, op: &Op, depth: usize) {
    match op {
        Op::For {
            iv,
            lo,
            hi,
            step,
            carried,
            body,
            dbg_name,
        } => {
            // ⭐ THE THREE RENDERINGS OF ONE LIST, exactly as `affine.for` prints them — see
            // [`super::affine::Carried`]. A loop that carries nothing prints no results, no
            // `iter_args` and no `-> (..)`.
            let (results, iter_args, result_tys) = if carried.is_empty() {
                (String::new(), String::new(), String::new())
            } else {
                (
                    format!(
                        "{} = ",
                        carried
                            .iter()
                            .map(|c| print::val(c.result))
                            .collect::<Vec<_>>()
                            .join(", ")
                    ),
                    format!(
                        " iter_args({})",
                        carried
                            .iter()
                            .map(|c| format!("{} = {}", print::val(c.arg), print::val(c.init)))
                            .collect::<Vec<_>>()
                            .join(", ")
                    ),
                    format!(" -> ({})", vec!["index"; carried.len()].join(", ")),
                )
            };
            let _ = writeln!(
                out,
                "{results}scf.for {} = {} to {} step {}{iter_args}{result_tys} {{",
                print::val(*iv),
                print::val(*lo),
                print::val(*hi),
                print::val(*step)
            );
            for inner in body {
                print::emit(out, inner, depth + 1);
            }
            print::indent(out, depth);
            match dbg_name {
                None => out.push_str("}\n"),
                Some(name) => {
                    let _ = writeln!(out, "}} {{dbgName = \"{name}\"}}");
                }
            }
        }
        Op::If {
            cond,
            body,
            else_body,
        } => {
            let _ = writeln!(out, "scf.if {} {{", print::val(*cond));
            region(out, body, depth);
            print::indent(out, depth);
            if else_body.is_empty() {
                out.push_str("}\n");
            } else {
                out.push_str("} else {\n");
                region(out, else_body, depth);
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
