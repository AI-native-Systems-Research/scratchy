//! UPSTREAM `arith` — the constants a program declares and the integer predicates it branches on.
//!
//! Not one of the scheduler's own dialects: these are MLIR's standard arithmetic ops, used here for
//! exactly what `dcc/test/PT/xrfbmm_int8_fwd.mlir` uses them for.

use std::fmt::Write as _;

use crate::islands::dataflow_ir::dialects::Val;
use crate::islands::dataflow_ir::print;
use crate::islands::dataflow_ir::ty::{ElemType, ScalarTy, Vector};

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

/// WHICH INTEGER COMPARISON — `mlir::arith::CmpIPredicate`, as `arith.cmpi` spells it.
///
/// ⭐ THE SPELLING IS THE ENUMERATOR'S OWN NAME, printed as a bare keyword before the operands:
/// `%21 = arith.cmpi eq, %20, %7 : index`
/// (`dcc/test/Transform/CFGSimplificationDataflowLevel/merging.mlir:37`).
///
/// # ⛔⛔ SIX, NOT MLIR'S TEN — THE UNSIGNED FOUR ARE WHAT THE REFERENCE ABORTS ON
///
/// `mlir::arith::CmpIPredicate` has ten enumerators; `ult`, `ule`, `ugt` and `uge` are absent here
/// because `getSentientCmpIPredicate` (`StandardToSentient.cpp:36-53`, bridge-2 entry 048) is an
/// if-chain over the six signed forms whose `else` is `DT_CHECK(0)`. An unsigned comparison reaching
/// this pipeline is a stop, not a lowering — so it is a value this island must not be able to hold,
/// and the reference's abort is then unreachable BY CONSTRUCTION rather than by convention.
/// `SentientTypes.td`'s own `CmpPredicate` is the same six.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CmpIPredicate {
    /// `eq` — equal. The only one this pipeline emits today, and the one a loop-position predicate
    /// and a value-based conditional are both built from.
    Eq,
    /// `ne` — not equal.
    Ne,
    /// `slt` — signed less than.
    Slt,
    /// `sle` — signed less than or equal.
    Sle,
    /// `sgt` — signed greater than.
    Sgt,
    /// `sge` — signed greater than or equal.
    Sge,
}

impl CmpIPredicate {
    /// THE KEYWORD `arith.cmpi` PRINTS.
    #[must_use]
    pub const fn spelling(self) -> &'static str {
        match self {
            CmpIPredicate::Eq => "eq",
            CmpIPredicate::Ne => "ne",
            CmpIPredicate::Slt => "slt",
            CmpIPredicate::Sle => "sle",
            CmpIPredicate::Sgt => "sgt",
            CmpIPredicate::Sge => "sge",
        }
    }
}

/// AN INTEGER LITERAL AND THE WIDTH IT IS TYPED AT — what an `arith.constant` of integer type binds.
///
/// ⛔⛔ `i1` IS A CASE OF ITS OWN BECAUSE THE REFERENCE READS IT BACK DIFFERENTLY. An `i1` constant
/// is an `IntegerAttr` of one signless bit, so `ConstantIntOp::value()` sign-extends it and `true`
/// arrives as **-1**; `LowerConstantIntToSentient` therefore reads it through `BoolAttr::getValue()`
/// instead, which is a `bool` (`StandardToSentient.cpp:361-366`). Splitting the case in the type is
/// what lets that lowering pick the right reader without asking the value what it is at run time —
/// and the reference's own output confirms which answer is intended: an `arith.constant false`
/// lowers to `sentient.scalar_constant {value = 0 : si64} : i1`, a zero and not a minus one
/// (`dcc/test/Conversion/SentientToProgIR/simplify_or_op.mlir:8`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntConst {
    /// `arith.constant true` / `arith.constant false` — the `i1` a predicate is, and the one a
    /// [`LogicKind::Not`] xors against.
    ///
    /// ⭐ MLIR PRINTS IT AS A KEYWORD, with no type: `%false = arith.constant false`
    /// (`dcc/test/Conversion/SentientToProgIR/simplify_or_op.mlir`).
    Bool(bool),
    /// `arith.constant N : i<bits>` at any other width.
    Int {
        /// The literal.
        value: i64,
        /// How many bits it is typed at.
        bits: u32,
    },
}

impl IntConst {
    /// THE TYPE THE CONSTANT'S RESULT CARRIES.
    #[must_use]
    pub const fn ty(self) -> ScalarTy {
        match self {
            IntConst::Bool(_) => ScalarTy::Int(1),
            IntConst::Int { bits, .. } => ScalarTy::Int(bits),
        }
    }
}

/// ONE TWO-OPERAND INTEGER ARITHMETIC OP — `arith.addi`, `arith.subi` or `arith.muli`.
///
/// ⭐ ONE STRUCT FOR THE THREE BECAUSE THE `.td` DECLARES THEM ALIKE: two operands and a result, all
/// of one type. Which operation it is, is which variant of [`Op`] holds it, so the three lowerings
/// stay three functions ([`Op::AddI`], [`Op::SubI`], [`Op::MulI`]) exactly as the reference keeps
/// them three (`StandardToSentient.cpp:79`, `:90`, `:102`).
///
/// ⛔ NO ATTRIBUTE DICTIONARY, AND THAT IS A STATEMENT. `LowerMulIOpToSentient` alone copies the
/// arith op's attributes onto the op it emits (`:108`) — `LowerSubIOpToSentient` says in a comment
/// that it deliberately does not (`:96`: *"We do not use Arith Attributes"*). Nothing in `dcc/src`
/// ever builds an `arith.muli`, no `arith.*` op in `dcc/test` carries an attribute dictionary, and
/// the `AddIOp::create` calls in `AgenToSentient` set none, so the dictionary that copy transfers is
/// empty at every site this pipeline can reach — and it is empty **by construction** here, which is
/// the only form of that fact a type can hold.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IntBinary {
    /// The value it binds.
    pub result: Val,
    /// `$lhs`.
    pub lhs: Val,
    /// `$rhs`.
    pub rhs: Val,
    /// The type of both operands and of the result.
    pub ty: ScalarTy,
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

    /// `arith.constant N : i<bits>` — an integer literal that is NOT an index.
    ///
    /// (E) A SEPARATE OP BECAUSE THE TYPE DIFFERS. [`Op::Constant`] prints `: index`, and feeding
    /// that to `arith.xori .. : i1` is "use of value expects different type than prior uses: 'i1' vs
    /// 'index'".
    ///
    /// ⭐ AND THE REFERENCE SPLITS IT THE SAME WAY: `arith::ConstantIndexOp` and
    /// `arith::ConstantIntOp` are two op classes with two lowerings
    /// (`StandardToSentient.cpp:347` and `:358`), dispatched apart at `:443-451`.
    ConstantInt {
        /// The value it binds.
        result: Val,
        /// The literal and the width it is typed at.
        value: IntConst,
    },

    /// `arith.addi` — ⛔ THE OPERANDS, NOT AN ATTRIBUTE DICTIONARY. See [`IntBinary`].
    ///
    /// ⛔⛔ AND IT IS THE ONLY WAY A MUTABLE ADDRESS MOVES. `insertCopyAndAddStmtsHelper`
    /// (`AgenToSentient.hpp:502-520`) closes every address-carrying loop with
    /// `arith.addi %iter_arg, %c<coeff>` and yields the sum, once per loop level, so the address a
    /// transfer reads at iteration `(i, j, ..)` is the base plus each level's own coefficient. A
    /// loop carrying an address with no `addi` in it re-reads the same elements every iteration. The
    /// reference passes `iter_arg.getType()` as the result type there and the carried addresses are
    /// all `index`, so [`IntBinary::ty`] is [`ScalarTy::Index`] at that site — see
    /// [`super::affine::Carried`].
    AddI(IntBinary),

    /// `arith.subi`.
    SubI(IntBinary),

    /// `arith.muli`.
    MulI(IntBinary),

    /// `arith.cmpi <predicate>, %lhs, %rhs : index` — one integer comparison.
    ///
    /// (E) `first` IS `iv == lower bound` AND `last` IS `iv == upper bound - 1`
    /// (`SNControlFlowLowering.cpp:100-108`). Comparing against the trip count rather than one less
    /// than it makes `last` true on no trip at all.
    ///
    /// # ⛔⛔ THE PREDICATE IS A FIELD BECAUSE READERS OF THIS OP BRANCH ON IT
    ///
    /// It was welded into the printer while every emitter here happened to want `eq`, and that made
    /// two of the reference's functions half unreachable:
    /// `ConditionalSimplificationManager::getLhsRhsOfEQPredicate`
    /// (`CFGSDataflowConditionalTree.cpp:519-532`, bridge-2 entry 098) declines any conditional whose
    /// condition is not an `eq` — a test that cannot fail against an island with no other predicate —
    /// and `getSentientCmpIPredicate` (`StandardToSentient.cpp:36-53`, entry 048) maps all six.
    /// ⭐ EVERY OP THIS CRATE EMITS TODAY IS AN [`CmpIPredicate::Eq`] AND PRINTS EXACTLY AS BEFORE.
    Compare {
        /// The i1 it binds.
        result: Val,
        /// Which comparison — `arith.cmpi`'s first token.
        predicate: CmpIPredicate,
        /// `$lhs` — the induction variable, where this is a loop-position predicate.
        lhs: Val,
        /// `$rhs` — what it is compared against.
        ///
        /// (E) AN SSA VALUE, NOT A LITERAL. `arith.cmpi` takes two operands of the same type;
        /// writing `arith.cmpi eq, %14, 0 : index` is "expected SSA operand". The bound is minted as
        /// an `arith.constant` first.
        rhs: Val,
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
        Op::ConstantInt { result, value } => {
            let literal = match value {
                // ⭐ NO TYPE ON THE BOOLEAN FORM — `arith.constant true`, not `... : i1`.
                IntConst::Bool(flag) => (if *flag { "true" } else { "false" }).to_owned(),
                IntConst::Int { value, bits } => format!("{value} : i{bits}"),
            };
            let _ = writeln!(out, "{} = arith.constant {literal}", print::val(*result));
        }
        Op::AddI(op) => int_binary(out, "arith.addi", op),
        Op::SubI(op) => int_binary(out, "arith.subi", op),
        Op::MulI(op) => int_binary(out, "arith.muli", op),
        Op::Compare {
            result,
            predicate,
            lhs,
            rhs,
        } => {
            let _ = writeln!(
                out,
                "{} = arith.cmpi {}, {}, {} : index",
                print::val(*result),
                predicate.spelling(),
                print::val(*lhs),
                print::val(*rhs)
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

/// ONE `arith` INTEGER BINARY AS TEXT — `%r = <mnemonic> %lhs, %rhs : <ty>`.
fn int_binary(out: &mut String, mnemonic: &str, op: &IntBinary) {
    let _ = writeln!(
        out,
        "{} = {mnemonic} {}, {} : {}",
        print::val(op.result),
        print::val(op.lhs),
        print::val(op.rhs),
        op.ty.spelling()
    );
}
