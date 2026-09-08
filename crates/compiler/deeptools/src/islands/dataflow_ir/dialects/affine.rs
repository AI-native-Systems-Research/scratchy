//! UPSTREAM `affine` — the loop nest, the applied maps, and `dcc-opt`'s vector accesses.
//!
//! Not one of the scheduler's own dialects. The two accesses here are the `dcc-opt` forms; the
//! bridge emits [`super::agen::Op::VectorLoad`] and [`super::agen::Op::VectorStore`] instead, for
//! the reason recorded there.

use std::fmt::Write as _;

use crate::islands::dataflow_ir::dialects::{Index, Val};
use crate::islands::dataflow_ir::print;
use crate::islands::dataflow_ir::ty::{AffineMap, MemRef, Vector};

/// A LOOP BOUND — a literal, or a value the program computed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bound {
    /// `affine.for %i = 0 to 8`.
    Const(i64),
    /// `affine.for %i = 0 to %extent`, where the extent is an `arith.constant` the program declared.
    Val(Val),
}

/// ONE VALUE A LOOP CARRIES — `iter_args(%arg = %init)`, and the result the loop binds.
///
/// ⛔⛔ THREE VALUES, NOT ONE, AND CONFLATING THEM EMITS AN UNVERIFIABLE LOOP. The vendor's own
/// output is `%5 = affine.for %arg0 = 0 to 1 iter_args(%arg1 = %c0) -> (index) {`
/// (`dcc/test/L0LU/sync_send_recv_L0LUrow0_src_unit.mlir:53-57`): `%c0` is the INIT, read once
/// before the loop; `%arg1` is the REGION ARGUMENT, the only name the body may use; `%5` is the
/// RESULT, what the last iteration's `affine.yield` leaves behind. A single `Val` would have to
/// stand for all three, and a body naming the loop's own result is not a program.
///
/// ⭐ ALWAYS `index`. The only carried values this bridge creates are the MUTABLE ADDRESSES of a
/// transfer, minted as `arith.constant .. : index` and advanced by `arith.addi`
/// (`AgenToSentient.hpp:502-520` takes the addend's type from `iter_arg.getType()`), so the result
/// list a loop prints is `-> (index, ..)` with one entry per carried value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Carried {
    /// The value the loop starts from — evaluated OUTSIDE the loop.
    pub init: Val,
    /// The region argument the body reads this carried value through.
    pub arg: Val,
    /// The result the loop binds once it is done.
    pub result: Val,
}

/// ONE `affine` OPERATION.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Op {
    /// `affine.for %i = <lo> to <hi> { .. }`, or
    /// `%r = affine.for %i = <lo> to <hi> iter_args(%a = %init) -> (index) { .. }`.
    For {
        /// The induction variable it binds.
        iv: Val,
        /// The lower bound.
        lo: Bound,
        /// The upper bound.
        hi: Bound,
        /// THE VALUES IT CARRIES ACROSS ITERATIONS — empty for the plain counted loop.
        ///
        /// ⭐ ONE WALK OVER THIS PRODUCES ALL THREE RENDERINGS, so the result list, the
        /// `iter_args` list and the region arguments are the same length by construction — the
        /// mismatch `affine.for` verifies for cannot be built. The TERMINATOR's operands are the
        /// fourth, and they live in the body as an [`Op::Yield`] rather than here, because the
        /// terminator is the op the rung below rewrites (`dataflow_ir_to_sentient/mod.rs:361`).
        carried: Vec<Carried>,
        /// The body, ending in an [`Op::Yield`] once anything is carried.
        body: Vec<super::Op>,
        /// `{dbgName = ".."}` — the loop's name in the reference's own output.
        ///
        /// ⛔⛔ AN ATTRIBUTE THE REFERENCE READS BACK, NOT A COMMENT. `getDbgNameAttr` /
        /// `setDbgNameAttr` are how dcc carries a loop's identity across a rewrite:
        /// `transformSCFToAffineLoop` copies the `scf.for`'s name onto the `affine.for` it builds
        /// (`TransformLoopToLegalizeForSentientLowering.cpp:110-111`), and the vendor's own
        /// expectation for that pass checks the name survived —
        /// `{dbgName = "c0-l3lu-loop-ibr-chunk-y"}` on the transformed loop
        /// (`dcc/test/Transform/TransformLoopToLegalizeForSentientLowering/scf_loop_with_result.mlir:59`).
        /// Dropping it would make a ported rewrite pass its own test and fail the vendor's.
        ///
        /// ⭐ AFTER THE CLOSING BRACE, WHICH IS WHERE MLIR PUTS AN OP'S ATTRIBUTE DICTIONARY WHEN
        /// THE OP HAS A REGION: `affine.for %arg9 = 0 to 4 { .. } {dbgName = ".."}`
        /// (`dcc/test/PT/fp8-bmm.mlir:1019-1027`).
        ///
        /// ⭐ `None` PRINTS NOTHING AT ALL. Most loops this bridge builds are unnamed, and an empty
        /// dictionary is not the same text as no dictionary.
        dbg_name: Option<String>,
    },

    /// `affine.if #set(%dims)[%symbols] -> index { .. } else { .. }` — a branch on an AFFINE
    /// predicate, and the DataflowIR rung's own conditional.
    ///
    /// # ⛔⛔ NOT A SPELLING OF [`super::scf::Op::If`] — THE CONDITION IS A SET, NOT A VALUE
    ///
    /// `scf.if` branches on an `i1` that was already computed; this op branches on whether its
    /// operands SATISFY an integer set, and it yields a value the program then compares. The vendor
    /// writes the pair together, in that order:
    ///
    /// ```text
    /// %52 = affine.if #set0(%arg9) -> index {
    ///   affine.yield %c1 : index
    /// } else {
    ///   affine.yield %c0 : index
    /// }
    /// %53 = arith.cmpi eq, %52, %c1 : index
    /// scf.if %53 {
    /// ```
    /// (`dcc/test/PT/issue-236.mlir:59-65`)
    ///
    /// ⛔ AND THE PASSES ASK ABOUT IT BY NAME. `isOperationSelected` — the whole predicate deciding
    /// what a conditional tree's NODES are — is `isa<mlir::affine::AffineIfOp, scf::IfOp>`
    /// (`Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:34`, entry 095), and
    /// `createDummyYieldInElseReg` (entry 096) closes an empty else region with an
    /// `affine::AffineYieldOp` for this op and an `scf::YieldOp` for the other
    /// (`:389-393`). With only one of the two in the island, half of each function is unreachable
    /// and the tree can never hold a node the reference would have put in it.
    ///
    /// ⭐ 9 OCCURRENCES OVER 6 FILES in the authority tree's `dcc/test` — rarer than `scf.if`'s 788,
    /// and every one of them at the DataflowIR rung this island models.
    If {
        /// `$condition` — the affine set the operands are tested against.
        set: crate::islands::dataflow_ir::ty::IntegerSet,
        /// The set's DIMENSION operands, printed `(%a, %b)`.
        args: Vec<Val>,
        /// The set's SYMBOL operands, printed `[%s]`.
        ///
        /// ⛔ A SEPARATE LIST BECAUSE MLIR PRINTS THEM SEPARATELY — `printDimAndSymbolList` splits at
        /// `set.getNumDims()`, and the vendor writes both:
        /// `affine.if affine_set<(d0, d1)[s0, s1] : (..)> (%i4, %i7)[%Din_Cin, %Cin_Sin] -> (index)`
        /// (`dcc/test/PT/int8-genkg3-pt.mlir:147-148`).
        symbol_args: Vec<Val>,
        /// The values it binds — one per result, all `index`. Empty for a branch taken for effect.
        ///
        /// ⛔⛔ WHETHER THIS IS EMPTY DECIDES WHETHER THE TERMINATORS PRINT. MLIR passes
        /// `printBlockTerminators = getNumResults()`, so a result-less `affine.if` prints its regions
        /// with the `affine.yield` ELIDED — and that is exactly the state entry 096 changes, from an
        /// else region with no block at all to one holding a bare terminator.
        results: Vec<Val>,
        /// The `then` region.
        body: Vec<super::Op>,
        /// The `else` region. ⛔ EMPTY MEANS NO BLOCK, WHICH IS WHAT `getRegions()[1].empty()` TESTS
        /// (`CFGSDataflowConditionalTree.cpp:386`) — and printing then omits `else` entirely, as
        /// MLIR's own printer does.
        else_body: Vec<super::Op>,
        /// `{dbgName = ".."}` — WHICH SOURCE CONDITIONALS THIS ONE WAS MERGED OUT OF.
        ///
        /// ⛔ THE SHALLOW MERGE NAMES ITS DESTINATION WHATEVER KIND IT IS. `mergeShallow` (entry 177)
        /// ends in `dataflow::setDbgNameAttr(dst, ..)` (`CFGSDataflowConditionalTree.cpp:455-458`) and
        /// the candidates it is given are `isa<mlir::affine::AffineIfOp, scf::IfOp>` (`:34`), so an
        /// `affine.if` reaches that line on exactly the same path an `scf.if` does — see
        /// [`super::scf::Op::If::dbg_name`], which carries the vendor's printed example.
        ///
        /// ⚠️ NO FIXTURE UNDER `dcc/test` PRINTS ONE ON AN `affine.if` (all 9 of them are unnamed),
        /// so the form is MLIR's trailing attribute dictionary as for every other op that carries the
        /// discardable attribute, and not a shape the vendor's own text pins down here.
        dbg_name: Option<String>,
    },

    /// `affine.apply affine_map<..>(%args)` — an index computed from induction variables.
    Apply {
        /// The index it binds.
        result: Val,
        /// The map.
        map: AffineMap,
        /// Its arguments, in order.
        args: Vec<Val>,
    },

    /// `affine.yield %operands` — a loop's terminator, carrying its loop-carried values.
    ///
    /// ⛔ THE OPERANDS ARE THE CARRIED VALUES, which is why the terminator is an op with a list and
    /// not a bare keyword: `AffineYieldOpLowering` rewrites it to an `scf.yield` carrying THE SAME
    /// operands (`AffineToStandard.cpp:48`), so an empty list and a two-value list are different
    /// terminators, not the same one written differently.
    Yield {
        /// The values the loop carries out of this iteration.
        operands: Vec<Val>,
    },

    /// `affine.vector_load %view[..] : memref<..>, vector<..>`.
    ///
    /// ⛔ THE `dcc-opt` FORM. The BRIDGE emits [`super::agen::Op::VectorLoad`]; see there.
    VectorLoad {
        /// The vector it binds.
        result: Val,
        /// The view read.
        view: Val,
        /// The indices.
        indices: Vec<Index>,
        /// The view's type.
        view_ty: MemRef,
        /// The vector's type.
        ty: Vector,
    },

    /// `affine.vector_store %value, %view[..] : memref<..>, vector<..>`.
    VectorStore {
        /// The vector written.
        value: Val,
        /// The view written to.
        view: Val,
        /// The indices.
        indices: Vec<Index>,
        /// The view's type.
        view_ty: MemRef,
        /// The vector's type.
        ty: Vector,
    },
}

/// ONE `affine` OP AS TEXT. The caller has already indented the opening line.
pub(crate) fn emit(out: &mut String, op: &Op, depth: usize) {
    match op {
        Op::For {
            iv,
            lo,
            hi,
            carried,
            body,
            dbg_name,
        } => {
            out.push_str(&for_header(*iv, *lo, *hi, carried));
            for inner in body {
                print::emit(out, inner, depth + 1);
            }
            print::indent(out, depth);
            out.push_str(&for_footer(dbg_name.as_deref()));
        }
        Op::If {
            set,
            args,
            symbol_args,
            results,
            body,
            else_body,
            dbg_name,
        } => {
            // `printOptionalArrowTypeList`: nothing when there are no results, ` -> index` for one,
            // ` -> (index, index)` for more (`issue-236.mlir:59` writes the single-result form).
            let result_tys = match results.len() {
                0 => String::new(),
                1 => " -> index".to_owned(),
                n => format!(" -> ({})", vec!["index"; n].join(", ")),
            };
            let bound = if results.is_empty() {
                String::new()
            } else {
                format!("{} = ", print::vals(results))
            };
            let symbols = if symbol_args.is_empty() {
                String::new()
            } else {
                format!("[{}]", print::vals(symbol_args))
            };
            let _ = writeln!(
                out,
                "{bound}affine.if {}({}){symbols}{result_tys} {{",
                print::integer_set(set),
                print::vals(args)
            );
            // ⭐ THE TERMINATOR PRINTS ONLY WHERE MLIR PRINTS IT — see [`Op::If::results`].
            region(out, body, depth, !results.is_empty());
            print::indent(out, depth);
            if !else_body.is_empty() {
                out.push_str("} else {\n");
                region(out, else_body, depth, !results.is_empty());
                print::indent(out, depth);
            }
            // ⛔ AFTER THE REGIONS — see [`Op::If::dbg_name`].
            match dbg_name {
                None => out.push_str("}\n"),
                Some(name) => {
                    let _ = writeln!(out, "}} {{dbgName = \"{name}\"}}");
                }
            }
        }
        Op::Apply { result, map, args } => {
            let _ = writeln!(
                out,
                "{} = affine.apply {}({})",
                print::val(*result),
                print::affine_map(map),
                print::vals(args)
            );
        }
        Op::Yield { operands } => {
            if operands.is_empty() {
                out.push_str("affine.yield\n");
            } else {
                // ⛔ THE TYPE LIST IS NOT OPTIONAL ONCE THERE ARE OPERANDS. `AffineYieldOp`'s
                // assembly format is `attr-dict ($operands^ `:` type($operands))?`, so
                // `affine.yield %30, %37` alone does not parse; the vendor's own output is
                // `affine.yield %30, %37 : index, index`
                // (`dcc/test/Conversion/AgenToSentient/lx_indirect_loads_stores_composite.mlir:47`).
                // ⭐ ALWAYS `index`, ONE PER OPERAND — see [`Carried`] for why every value a loop
                // here carries is an address.
                let _ = writeln!(
                    out,
                    "affine.yield {} : {}",
                    print::vals(operands),
                    vec!["index"; operands.len()].join(", ")
                );
            }
        }
        Op::VectorLoad {
            result,
            view,
            indices,
            view_ty,
            ty,
        } => {
            let _ = writeln!(
                out,
                "{} = affine.vector_load {}[{}] : {}, {}",
                print::val(*result),
                print::val(*view),
                print::index_list(indices),
                print::memref(view_ty),
                print::vector(*ty)
            );
        }
        Op::VectorStore {
            value,
            view,
            indices,
            view_ty,
            ty,
        } => {
            let _ = writeln!(
                out,
                "affine.vector_store {}, {}[{}] : {}, {}",
                print::val(*value),
                print::val(*view),
                print::index_list(indices),
                print::memref(view_ty),
                print::vector(*ty)
            );
        }
    }
}

/// ONE REGION OF AN `affine.if`, WITH ITS TERMINATOR PRINTED OR ELIDED.
///
/// ⛔ ELIDING IS NOT DROPPING. The `affine.yield` stays in the region — [`super::regions`] and every
/// walk still see it, and entry 096's whole effect is to PUT one there — it is only unprinted, which
/// is what MLIR does for a region whose parent binds no results.
fn region(out: &mut String, ops: &[super::Op], depth: usize, print_terminator: bool) {
    for (n, inner) in ops.iter().enumerate() {
        let last = n + 1 == ops.len();
        if last
            && !print_terminator
            && matches!(inner, super::Op::Affine(Op::Yield { operands }) if operands.is_empty())
        {
            continue;
        }
        print::emit(out, inner, depth + 1);
    }
}

/// AN `affine.for`'S OPENING LINE, up to and including the `{` and its newline.
///
/// ⭐ SHARED WITH THE RUNG ABOVE. A time loop whose body has reached SentientIR is a
/// [`crate::islands::sentient::dialects::Op::AffineFor`] — the same loop with a different body type
/// — and one rendering is what keeps the two printers from drifting apart.
pub(crate) fn for_header(iv: Val, lo: Bound, hi: Bound, carried: &[Carried]) -> String {
    // ⭐ THE THREE RENDERINGS OF ONE LIST. A loop that carries nothing prints exactly what it
    // printed before this field existed — no results, no `iter_args`, no `-> (..)`.
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
    let mut out = String::new();
    let _ = writeln!(
        out,
        "{results}affine.for {} = {} to {}{iter_args}{result_tys} {{",
        print::val(iv),
        bound(lo),
        bound(hi)
    );
    out
}

/// AN `affine.for`'S CLOSING BRACE AND THE `dbgName` THAT FOLLOWS IT. The caller has indented.
pub(crate) fn for_footer(dbg_name: Option<&str>) -> String {
    match dbg_name {
        None => "}\n".to_owned(),
        Some(name) => format!("}} {{dbgName = \"{name}\"}}\n"),
    }
}

fn bound(b: Bound) -> String {
    match b {
        Bound::Const(n) => n.to_string(),
        Bound::Val(v) => print::val(v),
    }
}
