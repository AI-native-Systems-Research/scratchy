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

/// ONE `affine` OPERATION.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Op {
    /// `affine.for %i = <lo> to <hi> { .. }`.
    For {
        /// The induction variable it binds.
        iv: Val,
        /// The lower bound.
        lo: Bound,
        /// The upper bound.
        hi: Bound,
        /// The body.
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
        Op::For { iv, lo, hi, body } => {
            let _ = writeln!(
                out,
                "affine.for {} = {} to {} {{",
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
                let _ = writeln!(out, "affine.yield {}", print::vals(operands));
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
