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
        } => {
            // ⭐ THE THREE RENDERINGS OF ONE LIST. A loop that carries nothing prints exactly what
            // it printed before this field existed — no results, no `iter_args`, no `-> (..)`.
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
                "{results}affine.for {} = {} to {}{iter_args}{result_tys} {{",
                print::val(*iv),
                bound(*lo),
                bound(*hi)
            );
            for inner in body {
                print::emit(out, inner, depth + 1);
            }
            print::indent(out, depth);
            out.push_str("}\n");
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

fn bound(b: Bound) -> String {
    match b {
        Bound::Const(n) => n.to_string(),
        Bound::Val(v) => print::val(v),
    }
}
