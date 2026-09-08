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

/// WHICH WAY A NUMERIC CONVERSION GOES — `arith.sitofp` or `arith.fptosi`.
///
/// ⛔⛔ TWO OPS THE CAST IS NOT. `constructPrecisionConversionOperation` picks between THREE ops by
/// the two signednesses (`SNComputeLowering.cpp:411-427`): a signed-integer source with a float
/// result is `arith::SIToFPOp`, a float source with an integer result is `arith::FPToSIOp`, and
/// float to float is [`super::vectorchain::Op::Cast`]. A cast reinterprets a vector's elements; these
/// two compute new ones, so folding either into the cast changes every value on the wire.
///
/// ⭐ AND THE VENDOR WRITES BOTH: `%pt_result_fp = arith.sitofp %pt_result : vector<64xi16> to
/// vector<64xf16>` (`dcc/test/PE/int8-kg3-pe.mlir:98`) and `%d1r = arith.fptosi %d1c :
/// vector<64xf16> to vector<64xi8>` (`dcc/test/SFP/csqint8-sfp.mlir:112`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvertKind {
    /// `arith.sitofp` — a signed integer becoming a float.
    SiToFp,
    /// `arith.fptosi` — a float becoming a signed integer, truncated toward zero.
    FpToSi,
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

    /// `arith.divsi` — SIGNED integer division.
    ///
    /// ⛔⛔ THE OP EVERY `sentient.for` BOUND IS. The scheduler writes a trip count as
    /// `(upper - lower) / step`, and it writes it as two ops:
    ///
    /// ```text
    /// %1 = arith.subi %c2, %c0 : index
    /// %2 = arith.divsi %1, %c1 : index
    /// sentient.for %arg1 = %2 { .. }
    /// ```
    /// (`dcc/test/Conversion/VectorChainToSentientPT/dynamic_pt_masking.mlir:226-228`)
    ///
    /// `getForOpBound` (entry 091) walks exactly that chain backwards from the loop's bound operand —
    /// `arith::DivSIOp` → its `arith::SubIOp` lhs → the `arith.constant`s or `symbol.create_symbol`s
    /// that feed it (`Conversion/VectorChainLowering/LoweringXRF.cpp:262-294`) — so without a
    /// `divsi` in the island that walk has nothing to start from and the constant offset it computes
    /// (`:498`) is unreachable.
    ///
    /// ⭐ SIGNED, NOT `divui`. The reference names `arith::DivSIOp` and every quantity involved is an
    /// `index`, which MLIR treats as signed.
    DivSI(IntBinary),

    /// `arith.remsi` — the SIGNED remainder.
    ///
    /// ⛔⛔ IN THE ISLAND BECAUSE `affine.apply`'S OWN EXPANSION EMITS IT. MLIR's
    /// `AffineApplyExpander::visitModExpr` lowers `a mod b` as a `remsi` plus a sign correction —
    /// *"Mod(a, b) = a - b * floordiv(a, b)"*, implemented as
    /// `remsi`, `cmpi slt`, `addi` and a [`Op::Select`] — so a port of
    /// `expandAffineApplyOps` (bridge-2 entry 183,
    /// `Transform/Dataflow/LoopUnrollForShuffleOp.cpp:183`) that lacked this op could not expand an
    /// [`AffineExpr::Mod`](crate::islands::dataflow_ir::ty::AffineExpr::Mod) at all.
    ///
    /// ⭐ SIGNED, LIKE [`Op::DivSI`], for the same reason: every quantity these maps compute is an
    /// `index`, which MLIR treats as signed.
    RemSI(IntBinary),

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
        /// THE TYPE OF BOTH OPERANDS — ⛔⛔ NOT ALWAYS `index`, AND THE ONE EXCEPTION IS THE
        /// NEGATED CONDITIONAL. `arith.cmpi` prints its OPERAND type, and 785 of the 788 under the
        /// authority tree's `dcc/test` are `index`; the three that are not compare an `i1` against an
        /// `arith.constant false` — `%284 = arith.cmpi eq, %283, %false : i1`
        /// (`Transform/FlatteningLocalRegions/flatten_local_region4.mlir:746`), which is exactly how
        /// `constructConditionalOperation` negates a condition (`SNControlFlowLowering.cpp:179-183`).
        /// A welded `index` states a type the compared value was not defined with, which is "use of
        /// value expects different type than prior uses".
        ty: ScalarTy,
    },

    /// `arith.select %cond, %true_value, %false_value : index` — ONE VALUE CHOSEN BY A PREDICATE.
    ///
    /// # ⛔⛔ A LOOP BOUND CAN BE ONE, AND A PASS READS THROUGH IT
    ///
    /// `getLoopTripCount` (bridge-2 entry 184,
    /// `Transform/Dataflow/MutableAddrSplitting.cpp:743`) has three arms for what may bind an
    /// `scf.for`'s upper bound, and this op is the third:
    ///
    /// ```text
    /// } else if (auto select_ub_op = dyn_cast<arith::SelectOp>(ub_op)) {
    ///   auto true_op  = dyn_cast_or_null<arith::ConstantOp>(select_ub_op.getTrueValue().getDefiningOp());
    ///   auto false_op = dyn_cast_or_null<arith::ConstantOp>(select_ub_op.getFalseValue().getDefiningOp());
    ///   DT_CHECK_MSG(true_op && false_op, "Expecting SelectOp upper bound to contain constant values.");
    ///   ub = true_val > false_val ? true_val : false_val;
    /// }
    /// ```
    ///
    /// ⭐⭐ AND THE VENDOR HAS A TEST NAMED AFTER IT. `select_ub` in
    /// `dcc/test/Transform/MutableAddrSplitting/mutable_addr_splitting_one_dim.mlir:290` writes
    /// `%905 = arith.select %904, %c8, %c4 : index` as an `scf.for` bound and expects the pass to
    /// partition against **8**. Without this op in the island that arm is unreachable, the bound
    /// falls into the `llvm_unreachable("unsupported upper loop bound operation")` and the vendor's
    /// own case cannot be built at all.
    ///
    /// ⭐ IT IS ALSO WHAT `affine.apply`'S EXPANSION BRANCHES WITH. MLIR's
    /// `AffineApplyExpander::visitFloorDivExpr` picks between a dividend and its negated,
    /// decremented form with this op, twice — see [`Op::RemSI`].
    Select {
        /// The value it binds.
        result: Val,
        /// `$condition` — the `i1` chosen on.
        condition: Val,
        /// `$true_value`.
        true_value: Val,
        /// `$false_value`.
        false_value: Val,
        /// The type of the two arms and of the result.
        ///
        /// ⛔ ONE TYPE FOR ALL THREE, WHICH IS `arith.select`'S OWN VERIFIER (`SameOperandsAndResultType`
        /// over the arms). Every site this island can reach chooses between two `index` values — a
        /// loop bound and an expanded subscript — so the field records which rather than assuming it.
        ty: ScalarTy,
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

    /// `%r = arith.sitofp %v : vector<NxI> to vector<NxF>` — A PRECISION CONVERSION BETWEEN THE
    /// INTEGER AND FLOAT DOMAINS.
    ///
    /// ⛔⛔ WITHOUT IT `constructPrecisionConversionOperation` (entry 052) COULD EMIT ONLY ITS THIRD
    /// ARM. That function has four: integer to float, float to integer, float to float and a
    /// failure (`SNComputeLowering.cpp:411-429`) — and this island held the float-to-float one alone,
    /// so the two conversions the PE's own fixtures perform on every int8 matmul result had no op to
    /// be.
    ///
    /// ⭐ ONE VARIANT FOR THE TWO DIRECTIONS because MLIR declares them alike — one operand, one
    /// result, both stated in the printed type — and which it is, is [`ConvertKind`].
    Convert {
        /// The vector it binds.
        result: Val,
        /// Which direction.
        kind: ConvertKind,
        /// What is converted.
        input: Val,
        /// The input's type — ⛔ PRINTED, NOT INFERRED: MLIR spells both types, `: tin to tout`.
        input_ty: Vector,
        /// The result's type.
        ty: Vector,
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
        /// THE VALUE EVERY LANE CARRIES, in the element's own domain.
        ///
        /// ⛔⛔ THIS WAS `one: bool` AND THAT COULD NOT SPELL THE VENDOR'S OWN INPUT.
        /// `getOperandFromConstantOp` (entry 166) reads a splat through `constValToField`, whose
        /// accepted domain is 0, 1, **2 and 3** (`VectorOperands.cpp:250-262`) — and
        /// `dcc/test/Conversion/VectorChainToSentientPESFP/splat.mlir:68` writes
        /// `%cst = arith.constant dense<2> : vector<128xi8>`. With a boolean here two of the four
        /// compute ports that function can return were unreachable, so the port would have been a
        /// predicate over a domain the island had shrunk.
        ///
        /// ⭐ INTEGRAL EVEN FOR A FLOAT VECTOR, WHICH IS A CENSUS AND NOT AN ASSUMPTION. Every
        /// `arith.constant dense<…>` in the authority's 825 `dcc/test/**/*.mlir` files is integral:
        /// 108 × `1.000000e+00`, 82 × `0.000000e+00`, 22 × `0`, 6 × `2.000000e+00`,
        /// 4 × `3.000000e+00`, 2 × `4.000000e+00`, 1 × `2`. A float element prints this in MLIR's
        /// scientific form, an integer element as the bare literal — see [`emit`].
        ///
        /// ⭐ AND 4 IS STILL SPELLABLE, so entry 166's refusal of a splat outside 0..=3 stays
        /// reachable rather than becoming statically dead.
        splat: i64,
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
        Op::DivSI(op) => int_binary(out, "arith.divsi", op),
        Op::RemSI(op) => int_binary(out, "arith.remsi", op),
        Op::Compare {
            result,
            predicate,
            lhs,
            rhs,
            ty,
        } => {
            let _ = writeln!(
                out,
                "{} = arith.cmpi {}, {}, {} : {}",
                print::val(*result),
                predicate.spelling(),
                print::val(*lhs),
                print::val(*rhs),
                ty.spelling()
            );
        }
        Op::Select {
            result,
            condition,
            true_value,
            false_value,
            ty,
        } => {
            // ⭐ THE CONDITION IS NOT PART OF THE PRINTED TYPE. `arith.select` prints one type, the
            // one the arms and the result share; the condition's `i1` is implied.
            let _ = writeln!(
                out,
                "{} = arith.select {}, {}, {} : {}",
                print::val(*result),
                print::val(*condition),
                print::val(*true_value),
                print::val(*false_value),
                ty.spelling()
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
        Op::Convert {
            result,
            kind,
            input,
            input_ty,
            ty,
        } => {
            let mnemonic = match kind {
                ConvertKind::SiToFp => "arith.sitofp",
                ConvertKind::FpToSi => "arith.fptosi",
            };
            // ⛔ `to`, NOT A COMMA. `vectorchain.cast` prints `: tin, tout`; MLIR's own conversion
            // ops print `: tin to tout` (`dcc/test/PE/int8-kg3-pe.mlir:98`).
            let _ = writeln!(
                out,
                "{} = {mnemonic} {} : {} to {}",
                print::val(*result),
                print::val(*input),
                print::vector(*input_ty),
                print::vector(*ty)
            );
        }
        Op::DenseConstant { result, splat, ty } => {
            // MLIR prints a float splat in scientific form, which is what the vendored IR shows:
            // `arith.constant dense<0.000000e+00> : vector<64xf16>`.
            let literal = match ty.elem {
                ElemType::Int(_) => splat.to_string(),
                ElemType::F16
                | ElemType::F32
                | ElemType::Bf16
                | ElemType::F8E4M3Fn
                | ElemType::F8E8M0Fnu
                | ElemType::F8E5M2
                | ElemType::F4E2M1Fn
                | ElemType::MxFloat(_) => float_splat(*splat),
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

/// A FLOAT SPLAT IN MLIR'S OWN `%e` FORM — one digit, a point, six digits, then a SIGNED TWO-DIGIT
/// exponent: `0.000000e+00`, `2.000000e+00`.
///
/// ⛔ RUST'S `{:e}` IS NOT THAT FORM. `format!("{:.6e}", 2.0_f64)` is `2.000000e0` — no sign and no
/// padding — so printing it raw would emit an attribute MLIR reads back as a different literal only
/// by luck. The exponent is re-spelled here rather than parsed, so nothing in this path can fail.
fn float_splat(splat: i64) -> String {
    // Lossless for every splat the corpus writes; see `Op::DenseConstant::splat`.
    #[expect(
        clippy::cast_precision_loss,
        reason = "the corpus's splats are 0..=4 and an f64 holds every i64 below 2^53 exactly"
    )]
    let scientific = format!("{:.6e}", splat as f64);
    let (mantissa, exponent) = match scientific.split_once('e') {
        Some(split) => split,
        // `{:.6e}` always writes an `e`; this arm is the total answer rather than an unwrap.
        None => (scientific.as_str(), "0"),
    };
    let (sign, digits) = match exponent.strip_prefix('-') {
        Some(magnitude) => ('-', magnitude),
        None => ('+', exponent),
    };
    if digits.len() < 2 {
        format!("{mantissa}e{sign}0{digits}")
    } else {
        format!("{mantissa}e{sign}{digits}")
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
