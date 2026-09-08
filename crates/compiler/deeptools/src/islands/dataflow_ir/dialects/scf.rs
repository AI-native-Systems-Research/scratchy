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
        /// THE VALUES IT BINDS — one per result, all `index`. Empty for a branch taken for effect.
        ///
        /// ⛔⛔ THE VENDOR'S OWN INPUT BINDS ONE, so a conditional without a result list cannot hold
        /// the pass's input at all: `%13 = scf.if %12 -> (index) { scf.yield %c1 : index } else {
        /// scf.yield %c2 : index }`
        /// (`dcc/test/Transform/CFGSimplificationDataflowLevel/merging.mlir:129-133`), and `:117` is
        /// the same op binding an `i1`.
        ///
        /// ⛔ AND THE SHALLOW MERGE BRANCHES ON WHETHER IT IS EMPTY. `mergeShallow` (entry 177) keeps
        /// the destination's terminator when `dst->getNumResults() != 0` and the source's otherwise
        /// (`CFGSDataflowConditionalTree.cpp:420-428`), `shallowlyMergeConditionals` picks WHICH of
        /// two candidates is the destination by `n_if_op->getNumResults() > 0` (`:237-247`), and
        /// `areShallowlyMergeable` declines outright when BOTH bind something (`:348`). A census that
        /// answered "none" for every `scf.if` would make all three of those decisions constant.
        ///
        /// ⚠️ EVERY RESULT PRINTS AS `index`, as everywhere else in this island — see [`Carried`]. The
        /// fixture's `-> (i1)` conditional is a condition-forwarding form this crate does not emit;
        /// its own merged output binds an `index` (`merging.mlir:40`).
        results: Vec<Val>,
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
        /// `{dbgName = ".."}` — WHICH SOURCE CONDITIONALS THIS ONE WAS MERGED OUT OF.
        ///
        /// ⛔ AN `scf.if` DOES NOT IMPLEMENT `DebugNameOpInterface`, so `setDbgNameAttr` takes its
        /// fallback path and writes a DISCARDABLE attribute named `"dbgName"`
        /// (`DataflowOpInterfaces.cpp:38-50`, `DataflowInterfaces.td:68`) — which is why it prints in
        /// the attribute dictionary after the regions rather than inside the operation's own syntax:
        /// `} {dbgName = "SCF-If #2"}`
        /// (`dcc/test/Transform/CFGSimplificationDataflowLevel/merging.mlir:128`).
        ///
        /// ⭐ AND IT IS AN OUTPUT OF THE PASS, NOT ONLY OF ITS INPUT. `mergeShallow` (entry 177) ends
        /// by naming the merged conditional after both halves, so the reference's own expectation
        /// after two merges is `} {dbgName = "CFGSM(SCF-If #4, CFGSM(SCF-If #2, SCF-If #3))"}`
        /// (`merging.mlir:59`) — see
        /// [`crate::bridges::dataflow_ir_to_sentient::tf_cfgs_dataflow_conditional_tree::new_dbg_name_from_list`].
        dbg_name: Option<String>,
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
    /// ⛔ SO THIS IS AN **INPUT** OP TO BRIDGE 2, NOT ONE THAT BRIDGE EMITS. It is here because the
    /// pass's input contains it — `dcc/test/Transform/TransformLoopToLegalizeForSentientLowering/scf_loop_with_result.mlir:32`
    /// is `%11 = scf.for %arg4 = %c0 to %10 step %c1 iter_args(%arg5 = %arg3) -> (index)` with `%10`
    /// an `arith.select` — and without it `transformSCFToAffineLoop` has no argument to be given and
    /// the pass could not be ported at all.
    ///
    /// ⭐ AND **BRIDGE 1 IS WHERE THAT INPUT COMES FROM**: `constructImplicitLoopsForContiguousTransfer`
    /// emits exactly this op, with exactly that `arith.select` for its upper bound, whenever a
    /// contiguous transfer's steady-state and epilogue stick counts differ — see
    /// [`crate::bridges::superdsc_to_dataflow_ir::transfer::emit_implicit_loops_for_contiguous_transfer`].
    /// The two bridges' claims are consistent: one produces the conditional bound, the next removes it.
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

/// ONE REGION OF AN `scf.if`, WITH ITS TERMINATOR PRINTED OR ELIDED.
///
/// ⛔ MLIR ELIDES IT WHEN THE OP BINDS NOTHING. `SCF.cpp`'s printer passes
/// `printBlockTerminators = !getResults().empty()`, so a result-less conditional's `scf.yield` is
/// never printed — which is what makes `} else {` followed by a bare `}` the reference's own text
/// (`dcc/test/PT/issue-236.mlir:70-71`) rather than a region printed with a stray terminator. One
/// that DOES bind a result prints both terminators, and the vendor writes them:
/// `scf.yield %c1 : index` inside `%13 = scf.if %12 -> (index)`
/// (`dcc/test/Transform/CFGSimplificationDataflowLevel/merging.mlir:129-133`).
///
/// ⭐ ELIDED, NOT DROPPED. The op stays in the region: [`super::regions`] and every walk still see
/// it, and entry 096's whole job is to put one there.
fn region(out: &mut String, ops: &[super::Op], depth: usize, print_terminator: bool) {
    for (n, inner) in ops.iter().enumerate() {
        let last = n + 1 == ops.len();
        if last
            && !print_terminator
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
            results,
            body,
            else_body,
            dbg_name,
        } => {
            // ⛔ SCF PARENTHESISES A SINGLE RESULT WHERE AFFINE DOES NOT: the vendor's one-result
            // form is `%13 = scf.if %12 -> (index) {` (`merging.mlir:129`), against
            // [`super::affine::Op::If`]'s bare ` -> index` (`issue-236.mlir:59`).
            let (bound, result_tys) = if results.is_empty() {
                (String::new(), String::new())
            } else {
                (
                    format!("{} = ", print::vals(results)),
                    format!(" -> ({})", vec!["index"; results.len()].join(", ")),
                )
            };
            let _ = writeln!(out, "{bound}scf.if {}{result_tys} {{", print::val(*cond));
            // ⭐ THE TERMINATORS PRINT ONLY WHERE MLIR PRINTS THEM — see [`Op::If::results`].
            region(out, body, depth, !results.is_empty());
            print::indent(out, depth);
            if !else_body.is_empty() {
                out.push_str("} else {\n");
                region(out, else_body, depth, !results.is_empty());
                print::indent(out, depth);
            }
            // ⛔ AFTER THE REGIONS, NOT BEFORE THEM — a discardable attribute prints in the trailing
            // dictionary: `} {dbgName = "SCF-If #2"}` (`merging.mlir:128`).
            match dbg_name {
                None => out.push_str("}\n"),
                Some(name) => {
                    let _ = writeln!(out, "}} {{dbgName = \"{name}\"}}");
                }
            }
        }
        Op::Yield { operands } => {
            if operands.is_empty() {
                out.push_str("scf.yield\n");
            } else {
                // ⛔ THE TYPE LIST IS NOT OPTIONAL ONCE THERE ARE OPERANDS, exactly as for
                // `affine.yield` — the op this one is converted FROM (`AffineToStandard.cpp:48`), so
                // the two print the same list. `ScfYieldOp`'s assembly format is
                // `attr-dict ($results^ ':' type($results))?`, and the vendor's own text is
                // `scf.yield %20 : index`
                // (`dcc/test/Transform/CFGSimplificationDataflowLevel/simplify-conditional.mlir:311`);
                // EVERY `scf.yield` with operands under `dcc/test` carries its types.
                // ⭐ ALWAYS `index`, ONE PER OPERAND — see [`super::affine::Carried`] for why every
                // value a loop in this island carries is an address, the same assumption the
                // `-> (..)` result list above already makes.
                let _ = writeln!(
                    out,
                    "scf.yield {} : {}",
                    print::vals(operands),
                    vec!["index"; operands.len()].join(", ")
                );
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

#[cfg(test)]
mod tests {
    use crate::islands::dataflow_ir::dialects::affine::Carried;
    use crate::islands::dataflow_ir::dialects::scf::Op;
    use crate::islands::dataflow_ir::dialects::{self, Val};
    use crate::islands::dataflow_ir::print::emit;

    /// ⭐ THE PLAIN COUNTED FORM, AS THE VENDOR'S OWN INPUT WRITES IT.
    ///
    /// `dcc/test/Transform/CFGSimplificationDataflowLevel/simplify-conditional.mlir:319` is
    /// `scf.for %arg5 = %c0 to %c4 step %c1 {` — three operands, no results, no `iter_args`. This is
    /// the input to case 3 of that test, *"Same as 2. but with scf.for instead of affine.for"*
    /// (`:241`), which is the case bridge-2 entry 142's `scf` arm exists for.
    #[test]
    fn prints_the_vendors_counted_scf_loop() {
        let op = dialects::Op::Scf(Op::For {
            iv: Val(105),
            lo: Val(100),
            hi: Val(104),
            step: Val(101),
            carried: Vec::new(),
            body: vec![dialects::Op::Scf(Op::Yield {
                operands: Vec::new(),
            })],
            dbg_name: None,
        });

        let mut got = String::new();
        emit(&mut got, &op, 0);

        assert_eq!(
            "scf.for %105 = %100 to %104 step %101 {\n  scf.yield\n}\n",
            got
        );
    }

    /// ⭐ AND THE CARRYING FORM, WHICH THE SAME TEST'S EXPECTATION CONTAINS.
    ///
    /// `simplify-conditional.mlir:64`:
    /// `%50 = scf.for %51 = %47 to %11 step %6 iter_args(%52 = %49) -> (index) {` — init `%49`,
    /// region argument `%52`, result `%50`, the three values [`Carried`] keeps apart. Its terminator
    /// is `scf.yield %20 : index` (`:311`), types and all.
    #[test]
    fn prints_the_vendors_carrying_scf_loop() {
        let op = dialects::Op::Scf(Op::For {
            iv: Val(51),
            lo: Val(47),
            hi: Val(11),
            step: Val(6),
            carried: vec![Carried {
                init: Val(49),
                arg: Val(52),
                result: Val(50),
            }],
            body: vec![dialects::Op::Scf(Op::Yield {
                operands: vec![Val(52)],
            })],
            dbg_name: None,
        });

        let mut got = String::new();
        emit(&mut got, &op, 0);

        assert_eq!(
            "%50 = scf.for %51 = %47 to %11 step %6 iter_args(%52 = %49) -> (index) {\n  \
             scf.yield %52 : index\n}\n",
            got
        );
    }
}
