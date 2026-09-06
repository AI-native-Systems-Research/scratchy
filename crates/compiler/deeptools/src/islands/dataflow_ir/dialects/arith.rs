//! UPSTREAM `arith` — the constants a program declares and the integer predicates it branches on.
//!
//! Not one of the scheduler's own dialects: these are MLIR's standard arithmetic ops, used here for
//! exactly what `dcc/test/PT/xrfbmm_int8_fwd.mlir` uses them for.

use std::fmt::Write as _;

use crate::islands::dataflow_ir::dialects::Val;
use crate::islands::dataflow_ir::print;
use crate::islands::dataflow_ir::ty::{ElemType, Vector};

/// WHICH CONNECTIVE - `ddl.condition_and` / `_or` / `_not`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogicKind {
    /// `arith.andi`.
    And,
    /// `arith.ori`.
    Or,
    /// `arith.xori %c, true` - MLIR has no `not`, so a negation is an xor with true.
    Not,
}

/// ONE `arith` OPERATION.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Op {
    /// `arith.constant N : index` — the extents a loop counts to.
    Constant {
        /// The value it binds.
        result: Val,
        /// The literal.
        value: i64,
    },

    /// `arith.constant true` — the `i1` a negation xors against.
    ///
    /// (E) A SEPARATE OP BECAUSE THE TYPE DIFFERS. `Op::Constant` prints `: index`, and feeding that
    /// to `arith.xori .. : i1` is "use of value expects different type than prior uses: 'i1' vs
    /// 'index'".
    True {
        /// The i1 it binds.
        result: Val,
    },

    /// `arith.cmpi eq, %iv, <bound> : index` — one loop-position predicate.
    ///
    /// (E) `first` IS `iv == lower bound` AND `last` IS `iv == upper bound - 1`
    /// (`SNControlFlowLowering.cpp:100-108`). Comparing against the trip count rather than one less
    /// than it makes `last` true on no trip at all.
    Compare {
        /// The i1 it binds.
        result: Val,
        /// The induction variable tested.
        iv: Val,
        /// What it is compared against.
        ///
        /// (E) AN SSA VALUE, NOT A LITERAL. `arith.cmpi` takes two operands of the same type;
        /// writing `arith.cmpi eq, %14, 0 : index` is "expected SSA operand". The bound is minted as
        /// an `arith.constant` first.
        against: Val,
    },

    /// `arith.andi` / `arith.ori` / `arith.xori %c, true` - the connectives of a predicate.
    Logic {
        /// The i1 it binds.
        result: Val,
        /// Which connective.
        kind: LogicKind,
        /// Its operands. `Not` takes exactly one.
        operands: Vec<Val>,
    },

    /// `arith.constant dense<V> : vector<NxT>` — an immediate operand of a compute.
    ///
    /// (E) NOT A SPLAT OP. `constructComputeInputOperandAndAddToList`
    /// (`SNComputeLowering.cpp:461-495`) builds a dense `arith.constant` of the RESULT vector type
    /// for the `ZERO` and `ONE` pseudo-units. `createSplatOperation` is one rung down, in
    /// `dcc/src/Conversion/VectorChainLowering/` — it turns a vectorchain op into sentient ops, which
    /// is not this stage's job. An earlier refusal here cited it, and cited the wrong layer.
    DenseConstant {
        /// The vector it binds.
        result: Val,
        /// Whether the value is one rather than zero — the only two the templates name.
        one: bool,
        /// Its type.
        ty: Vector,
    },
}

/// ONE `arith` OP AS TEXT. The caller has already indented.
pub(crate) fn emit(out: &mut String, op: &Op) {
    match op {
        Op::Constant { result, value } => {
            let _ = writeln!(
                out,
                "{} = arith.constant {value} : index",
                print::val(*result)
            );
        }
        Op::True { result } => {
            let _ = writeln!(out, "{} = arith.constant true", print::val(*result));
        }
        Op::Compare {
            result,
            iv,
            against,
        } => {
            let _ = writeln!(
                out,
                "{} = arith.cmpi eq, {}, {} : index",
                print::val(*result),
                print::val(*iv),
                print::val(*against)
            );
        }
        Op::Logic {
            result,
            kind,
            operands,
        } => {
            let mnemonic = match kind {
                LogicKind::And => "arith.andi",
                LogicKind::Or => "arith.ori",
                LogicKind::Not => "arith.xori",
            };
            let _ = writeln!(
                out,
                "{} = {mnemonic} {} : i1",
                print::val(*result),
                print::vals(operands)
            );
        }
        Op::DenseConstant { result, one, ty } => {
            // MLIR prints a float splat in scientific form, which is what the vendored IR shows:
            // `arith.constant dense<0.000000e+00> : vector<64xf16>`.
            let literal = match (one, ty.elem) {
                (false, ElemType::Int(_)) => "0".to_owned(),
                (true, ElemType::Int(_)) => "1".to_owned(),
                (false, _) => "0.000000e+00".to_owned(),
                (true, _) => "1.000000e+00".to_owned(),
            };
            let _ = writeln!(
                out,
                "{} = arith.constant dense<{literal}> : {}",
                print::val(*result),
                print::vector(*ty)
            );
        }
    }
}
