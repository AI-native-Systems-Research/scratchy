//! `VectorChain.td` — EVERYTHING THE PE AND THE SFP COMPUTE.
//!
//! The dialect declares twenty-three operations; fourteen of the fifteen here are the ones an
//! emitted program contains, and the fifteenth — [`Op::Merge`] — is here because
//! `getComputePrecisionOfOp` must be able to be asked about one. The attribute enumerations they
//! carry are `VectorChainEnums.td`'s, and each records which of the two — the enum's own name, or
//! the operand's — is the mnemonic MLIR parses.

use std::fmt::Write as _;

use crate::islands::dataflow_ir::dialects::Val;
use crate::islands::dataflow_ir::print;
use crate::islands::dataflow_ir::ty::{
    AffineExpr, AffineMap, Constraint, IntegerSet, ScalarTy, Vector,
};

/// ONE SCALAR A SHUFFLE'S NEGATIVE INDEX REACHES, WITH THE TYPE THE OP PRINTS FOR IT.
///
/// ⭐ THE TYPE IS THE OPERAND'S, NOT THE RESULT ELEMENT'S. `Variadic<AnyTypeOf<[Index, AnyInteger,
/// AnyFloat]>>:$variable` (`VectorChain.td:459`) and `ShuffleOp::print` writes each one out beside
/// the input's (`VectorChain.cpp:389-390`), so a loop counter prints `index` inside an op whose
/// result is a `vector<64xf16>`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShuffleVariable {
    /// The scalar.
    pub val: Val,
    /// How it is typed.
    pub ty: ScalarTy,
}

/// WHICH COMPARISON — `VectorChainElementWiseCompareOperator` (`VectorChainEnums.td`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompareOp {
    /// `compare_eq`.
    Eq,
    /// `compare_neq`.
    Neq,
    /// `compare_lt`.
    Lt,
    /// `compare_le`.
    Le,
    /// `compare_gt`.
    Gt,
    /// `compare_ge`.
    Ge,
}

impl CompareOp {
    /// THE ENUMERATOR'S VALUE, which is what the attribute carries.
    ///
    /// ⛔⛔ THE MNEMONIC IS THE ENUM'S NAME, NOT THE ATTRIBUTE'S, and both wrong guesses were caught
    /// by dbo-opt rather than by reading. `#vectorchain<compare_op compare_gt>` gives "unknown
    /// attribute `compare_op` in dialect `vectorchain`"; a plain `4 : i32` gives "failed to satisfy
    /// constraint". IBM's own IR writes
    /// `#vectorchain<element_wise_compare_operator compare_gt>` — the mnemonic comes from
    /// `VectorChainElementWiseCompareOperator`, while `compare_op` is only the operand's name.
    #[must_use]
    pub const fn spelling(self) -> &'static str {
        match self {
            Self::Eq => "compare_eq",
            Self::Neq => "compare_neq",
            Self::Lt => "compare_lt",
            Self::Le => "compare_le",
            Self::Gt => "compare_gt",
            Self::Ge => "compare_ge",
        }
    }
}

/// WHICH BINARY OPERATION — `VectorChainBinaryOperator` (`VectorChainEnums.td`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    /// `add`.
    Add,
    /// `sub`.
    Sub,
    /// `mul`.
    Mul,
    /// `mul_div2`.
    MulDiv2,
    /// `min`.
    Min,
    /// `max`.
    Max,
    /// `abs_min`.
    AbsMin,
    /// `abs_max`.
    AbsMax,
    /// `and0`.
    And,
    /// `or0`.
    Or,
    /// `xnor`.
    Xnor,
    /// `and_not`.
    AndNot,
}

impl BinaryOp {
    /// The enumerator's spelling, written as `#vectorchain<binary_operator add>` — the mnemonic is
    /// the enum's name, as it is for [`CompareOp::spelling`].
    #[must_use]
    pub const fn spelling(self) -> &'static str {
        match self {
            Self::And => "and0",
            Self::Or => "or0",
            Self::Xnor => "xnor",
            Self::AndNot => "and_not",
            Self::Min => "min",
            Self::Max => "max",
            Self::AbsMin => "abs_min",
            Self::AbsMax => "abs_max",
            Self::Add => "add",
            Self::Mul => "mul",
            Self::Sub => "sub",
            Self::MulDiv2 => "mul_div2",
        }
    }
}

/// WHICH TRANSCENDENTAL ESTIMATE — one `vectorchain` op each.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EstimateKind {
    /// `exp_estimate`.
    Exp,
    /// `rec_estimate`.
    Rec,
    /// `ln_estimate`.
    Ln,
    /// `rsqrt_estimate`.
    Rsqrt,
    /// `sigmoid_estimate`.
    Sigmoid,
    /// `tanh_estimate`.
    Tanh,
}

impl EstimateKind {
    /// The op's mnemonic.
    #[must_use]
    pub const fn spelling(self) -> &'static str {
        match self {
            Self::Exp => "exp_estimate",
            Self::Rec => "rec_estimate",
            Self::Ln => "ln_estimate",
            Self::Rsqrt => "rsqrt_estimate",
            Self::Sigmoid => "sigmoid_estimate",
            Self::Tanh => "tanh_estimate",
        }
    }

    /// The attribute mnemonic its `version` is written under, where it takes one.
    ///
    /// ⛔ THE EXPONENTIAL HAS ITS OWN ENUM. IBM's IR writes `#vectorchain<exp_estimate a>` for exp
    /// and `#vectorchain<estimate_versions slope>` for the rest — `VectorChainExpEstimate` and
    /// `VectorChainEstimateVersions` are two enums, so one mnemonic would be wrong for one of them.
    #[must_use]
    pub const fn version_mnemonic(self) -> &'static str {
        match self {
            Self::Exp => "exp_estimate",
            Self::Rec | Self::Ln | Self::Rsqrt | Self::Sigmoid | Self::Tanh => "estimate_versions",
        }
    }
}

/// WHICH VERSION OF AN ESTIMATE.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EstimateVersion {
    /// `a` — the exponential's first form.
    A,
    /// `b` — its second.
    B,
    /// `slope`.
    Slope,
    /// `offset`.
    Offset,
}

impl EstimateVersion {
    /// The enumerator's spelling.
    #[must_use]
    pub const fn spelling(self) -> &'static str {
        match self {
            Self::A => "a",
            Self::B => "b",
            Self::Slope => "slope",
            Self::Offset => "offset",
        }
    }
}

/// A VALUE A COMPUTE PRODUCED — and it must go somewhere.
///
/// # 🛑 A COMPUTE WHOSE RESULT NOTHING CONSUMES IS NEVER LOWERED
///
/// ⛔⛔ `BinaryOpLowering::matchAndRewrite` resolves the result's DESTINATION in `fillOpInfo`
/// before doing anything else (`VectorChainToSentientPESFP.cpp:332-335`), and returns `failure()`
/// when there is none — which is BEFORE `setReuseInformation` at `:338`. So the binary is never
/// lowered and its operands are never entered in `data_origins_`.
///
/// ⛔ AND THE FAILURE SURFACES SOMEWHERE ELSE ENTIRELY. `runOnOperation` DISCARDS
/// `fuseComputeOps`'s result (`:1405`), so the walk continues to
/// `lowerDanglingNonComputeOpsPESFP`, which finds a receive with no `data_origins_` entry
/// (`OperandReuse.cpp:73-78`) and reports *"Dangling non-compute op has no use"* naming THE
/// RECEIVE. The emitter read that as "too many operands arrive" and it means "the compute's result
/// goes nowhere". A lockdown on operand arity would have fixed none of it.
///
/// ⭐ IBM'S `sfp` UNIT ENDS IN A SEND EVERY TIME — `dataflow.send %27, %25`
/// (`/tmp/ktir_ref/export/debug/dfir.mlir:151`).
///
/// ⛔ AND THE TAPE ALREADY SAID WHERE IT GOES. `Node::output` is a placement the compute never
/// read — a fact carried and not exploited.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Computed {
    val: Val,
    ty: Vector,
}

impl Computed {
    /// ⛔ MINTED ONLY BY A COMPUTE. The op that binds it is pushed here, so the handle and the
    /// operation cannot disagree about the value or its type.
    pub fn of(result: Val, ty: Vector) -> Computed {
        Computed { val: result, ty }
    }

    /// The value it binds.
    #[must_use]
    pub const fn val(self) -> Val {
        self.val
    }

    /// Its type.
    #[must_use]
    pub const fn ty(self) -> Vector {
        self.ty
    }
}

/// A LANE MASK'S DEFINITION — how many lanes are live, and the i1 vector it is stated over.
///
/// ⭐ THE TWO TRAVEL TOGETHER because a prefix means nothing without the width it is a prefix OF.
/// This is what `vectorchain.create_affine_mask` carries, and [`LaneMask::binds`] is the ONLY way
/// to obtain a [`Predicate`] — so a mask's type at its USE is the same value that was written at
/// its DEFINITION.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LaneMask {
    lanes: u64,
    ty: Vector,
}

impl LaneMask {
    /// A continuous live prefix of `lanes`, over an i1 vector of type `ty`.
    #[must_use]
    pub const fn prefix_of(lanes: u64, ty: Vector) -> Self {
        Self { lanes, ty }
    }

    /// How many lanes are live.
    #[must_use]
    pub const fn live(self) -> u64 {
        self.lanes
    }

    /// The i1 vector type this mask is stated over.
    #[must_use]
    pub const fn ty(self) -> Vector {
        self.ty
    }

    /// THE AFFINE SET THIS PREFIX ELIDES — the `mask_set` attribute, as the other form of the op
    /// writes it out.
    ///
    /// ⛔⛔ THE PREFIX FORM IS NOT A DIFFERENT OPERATION, AND BOTH MASK READERS ASK FOR THE SET.
    /// `vectorchain.create_affine_mask` carries a REQUIRED `mask_set` whichever way this island
    /// prints it, and `getMaskValueForPT` (bridge-2 entry 089) and `getMaskValueConstantForNonPT`
    /// (entry 163) each read that attribute and nothing else. `lanes` live lanes of `ty.len` is
    /// `(d0 - lanes >= 0, -d0 + (ty.len - 1) >= 0)` — one dimension, no symbol — and ⛔ THE SET
    /// DESCRIBES THE LANES THAT ARE **OFF**: `d0` ranges over the masked tail, which is why a
    /// fully live mask states the EMPTY span `Lb = lanes > Ub = lanes - 1`.
    ///
    /// ⭐ ONE STATEMENT OF IT, PINNED BY A GOLDEN ON EACH SIDE. 48 live of 64 is IBM's `#set2` and
    /// lowers to `sentient.scalar_constant {value = 2 : si64}` on the PT
    /// (`dcc/test/Conversion/VectorChainToSentientPT/dynamic_pt_masking.mlir:86`); 64 live of 64 is
    /// `(d0 - 64 >= 0, -d0 + 63 >= 0)` and lowers to `{value = 0 : si64}` on the PT (`:85`) and on
    /// the PE/SFP (`dcc/test/Conversion/VectorChainToSentientPESFP/fnms_with_cast.mlir:11`). Both
    /// entries reconstructed it separately until this method existed, and two derivations of one
    /// set is one too many.
    ///
    /// ⭐ `None` FOR A MASK NO `i64` CAN STATE. The constraints hold signed constants; a lane count
    /// past `i64::MAX` is not a vector this compiler addresses.
    #[must_use]
    pub fn as_set(self) -> Option<IntegerSet> {
        let live = i64::try_from(self.lanes).ok()?;
        let len = i64::try_from(self.ty.len).ok()?;
        Some(IntegerSet {
            dims: 1,
            symbols: 0,
            constraints: vec![
                Constraint {
                    expr: AffineExpr::dim(0).plus(AffineExpr::Const(-live)),
                    is_equality: false,
                },
                Constraint {
                    expr: AffineExpr::dim(0)
                        .times(-1)
                        .plus(AffineExpr::Const(len - 1)),
                    is_equality: false,
                },
            ],
        })
    }

    /// THE VALUE A `create_affine_mask` BOUND, CARRYING THIS MASK'S OWN TYPE.
    #[must_use]
    pub const fn binds(self, result: Val) -> Predicate {
        Predicate {
            val: result,
            ty: self.ty,
        }
    }
}

/// A MASK VALUE AND THE TYPE IT WAS DEFINED AT.
///
/// ⛔⛔ THE TYPE TRAVELS WITH THE VALUE, AND THAT IS THE WHOLE POINT. dbo-opt refused granite-2b's
/// `group_7`: *"use of value '%42' expects different type than prior uses: 'vector<64xi1>' vs
/// 'i1'"*. `%42` was minted by `arith.constant true` — an `i1` — while the use site printed its
/// type by RECOMPUTING `vector<{lanes}xi1>` from the operands' lane count. Two records of one
/// fact, and they disagreed.
///
/// ⭐ THE FIELDS ARE PRIVATE AND THERE IS NO PUBLIC CONSTRUCTOR. A `Predicate` comes only from
/// [`LaneMask::binds`], so it cannot be built around a value whose definition says something else.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Predicate {
    val: Val,
    ty: Vector,
}

impl Predicate {
    /// The masked value.
    #[must_use]
    pub const fn val(self) -> Val {
        self.val
    }

    /// THE MASK'S VALUE, MUTABLY — for RENUMBERING ONLY.
    ///
    /// ⛔ THE TYPE IS NOT REACHABLE THROUGH THIS. A clone renames the values of the op it copied
    /// (`dialects::parts_mut`) and a rewriter redirects a single use in place
    /// (`dialects::vals_mut`, `VectorChainHelper.cpp:600-602`); the type this predicate was DEFINED
    /// at is the same fact before and after, which is the whole reason the two travel together — so
    /// substituting one value for another of the SAME definition, which is exactly what
    /// `redefineConstantVectors` does, cannot change a mask's width.
    pub(crate) const fn val_mut(&mut self) -> &mut Val {
        &mut self.val
    }

    /// The type it was DEFINED at — never recomputed at the use.
    #[must_use]
    pub const fn ty(self) -> Vector {
        self.ty
    }

}

/// ONE `vectorchain` OPERATION.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Op {
    /// `vectorchain.<kind>_estimate %in {version} : tin, tout` — the SFP's transcendental estimates.
    ///
    /// ⛔ WHICH ONE IS THE COMPUTE'S `mode=`, NOT ITS COMPUTETYPE. A `FEST` dispatches on mode 0-9
    /// into exp(a), exp(b), rec, ln, rsqrt, sigmoid(slope|offset) and tanh(slope|offset)
    /// (`SNComputeLowering.cpp:1316-1372`).
    Estimate {
        /// The vector it binds.
        result: Val,
        /// What is estimated.
        input: Val,
        /// Which estimate.
        kind: EstimateKind,
        /// Which version, where the op takes one. `rec` and `ln` do not.
        version: Option<EstimateVersion>,
        /// The input's type.
        input_ty: Vector,
        /// The result's type.
        ty: Vector,
    },

    /// `vectorchain.fast_exp %in : tin, tout` — the exponential's auxiliary function.
    ///
    /// ⛔⛔ NOT `Estimate { kind: Exp }`. `VectorChain_FastExpOp` and `VectorChain_ExpEstimateOp` are
    /// two op definitions in one `.td` (`VectorChain.td:237` and `:251`), with two mnemonics and two
    /// lowering patterns (`FastExpOpLowering`, `ExpEstimateOpLowering`,
    /// `VectorChainToSentientPESFP.cpp:1250`); the estimate takes a REQUIRED
    /// `VectorChainExpEstimateAttr:$version` and this one takes no attribute at all. Folding it into
    /// the estimate family would make `version` optional for a case that has none, and would print
    /// the wrong mnemonic.
    ///
    /// ⭐ ADDED FOR `e076_fuseComputeOps`, which names `FastExpOp` in the set of ops its conversion
    /// target declares illegal — an op the island could not spell.
    FastExp {
        /// The vector it binds.
        result: Val,
        /// What is exponentiated.
        input: Val,
        /// The input's type.
        input_ty: Vector,
        /// The result's type.
        ty: Vector,
    },

    /// `vectorchain.floor %in : tin, tout` — the floor function.
    ///
    /// ⭐ ADDED FOR `e076_fuseComputeOps`, alongside [`Op::FastExp`]: `FloorOp` is one of the
    /// thirteen ops that pass's target declares illegal (`VectorChainToSentientPESFP.cpp:1261-1263`).
    /// It is `VectorChain_FloorOp` (`VectorChain.td:267`) — unary, no attribute, its own mnemonic.
    Floor {
        /// The vector it binds.
        result: Val,
        /// What is rounded down.
        input: Val,
        /// The input's type.
        input_ty: Vector,
        /// The result's type.
        ty: Vector,
    },

    /// `vectorchain.neg %in [%mask : tm] : tin, tout` — the negation, `VectorChain_NegOp`
    /// (`VectorChain.td:359-373`): *"This operation negates the input SSA variable."*
    ///
    /// ⛔ IT CARRIES A MASK AND [`Op::Floor`] DOES NOT, which is why this is its own variant and not a
    /// third mnemonic on that arm. `(ins AnyVectorOfAnyRank:$op, Optional<VectorOfRankAndType<[1],
    /// [I1]>>:$mask, OptionalAttr<StrAttr>:$dbgName)` (`:366-367`) — and the mask is narrower than the
    /// elementwise family's `Optional<AnyVectorOfAnyRank>`: rank exactly 1, element type exactly `i1`.
    ///
    /// ⭐ ADDED FOR `e168_getOperandFromNegOp`, whose whole body is `op.getOperand(0).getDefiningOp()`
    /// followed by a recursion — an op the island could not spell, so the operand that function
    /// forwards had nowhere to come from.
    Neg {
        /// The vector it binds — `(outs AnyVectorOfAnyRank:$data)`.
        result: Val,
        /// `$op` — what is negated. ⭐ THE OPERAND `e168` FORWARDS IS THIS ONE, at index 0.
        input: Val,
        /// `$mask`, IF THIS OP CARRIES ONE — `Optional`, so an unmasked negation is [`None`] and
        /// prints no bracket at all.
        mask: Option<Predicate>,
        /// The input's type.
        input_ty: Vector,
        /// The result's type.
        ty: Vector,
    },

    /// `vectorchain.scan_with_gap %in {reduction_op, gap, eval_order} : tin, tout` — a reduction.
    ///
    /// ⛔ WHICH REDUCTION IS THE COMPUTE'S `mode=`: 1 add, 8 max, 10 abs_max, 12 min, 14 abs_min
    /// (`SNComputeLowering.cpp:1453-1467`). The gap is 8 and the order left-to-right, both fixed
    /// there (`:1468-1470`).
    ScanWithGap {
        /// The vector it binds.
        result: Val,
        /// What is reduced.
        input: Val,
        /// Which reduction.
        reduction_op: BinaryOp,
        /// The input's type.
        input_ty: Vector,
        /// The result's type.
        ty: Vector,
    },

    /// `vectorchain.select %in {selection_map} : vector<..>, vector<..>` — a lane permutation.
    Select {
        /// The vector it binds.
        result: Val,
        /// The input.
        input: Val,
        /// Which lane each output lane reads.
        selection_map: AffineMap,
        /// The input's type.
        input_ty: Vector,
        /// The result's type.
        ty: Vector,
    },

    /// `vectorchain.multiply %a, %b {reduction_map} : ..` — a product with a reduction.
    Multiply {
        /// The vector it binds.
        result: Val,
        /// The left operand.
        a: Val,
        /// The right operand.
        b: Val,
        /// Which product lanes fold into which result lane. Arch- and precision-dependent:
        /// `IMA8` on RCUDD1A is `(d0 mod 128) floordiv 2`, on SEN1P5 `d0 floordiv 16`
        /// (`SNComputeLowering.cpp:157-172`).
        reduction_map: AffineMap,
        /// The operands' type.
        operand_ty: Vector,
        /// The result's type.
        ty: Vector,
    },

    /// `vectorchain.multiply_and_accumulate %a, %b, %acc {reduction_map} : ..`.
    MultiplyAccumulate {
        /// The vector it binds.
        result: Val,
        /// The left operand.
        a: Val,
        /// The right operand.
        b: Val,
        /// The accumulator read in.
        acc: Val,
        /// As [`Op::Multiply`].
        reduction_map: AffineMap,
        /// The operands' type.
        operand_ty: Vector,
        /// The result's type.
        ty: Vector,
    },

    /// `vectorchain.element_wise_compare %op1, %op2 [%mask : t] {compare_op} : t1, t2, tres`.
    ///
    /// (E) A FMAX IS NOT ONE OP. `constructFMINorFMAXOperation` (`SNComputeLowering.cpp:1189-1272`)
    /// emits a COMPARE (`compare_gt` for FMAX, `compare_le` for FMIN) and then a SELECTION over its
    /// i1 result. An earlier `vectorchain.fmax` was rejected outright by dbo-opt: "custom op
    /// 'vectorchain.fmax' is unknown". The 23 real ops are in `VectorChain.td`.
    ElementWiseCompare {
        /// The i1 vector it binds.
        result: Val,
        /// Left operand.
        op1: Val,
        /// Right operand.
        op2: Val,
        /// The lane mask, IF THIS OP CARRIES ONE — `Optional<AnyVectorOfAnyRank>:$mask`
        /// (`VectorChain.td`), so `None` prints no bracket at all.
        mask: Option<Predicate>,
        /// Which comparison.
        compare_op: CompareOp,
        /// The operands' type.
        operand_ty: Vector,
        /// The i1 result's type.
        ty: Vector,
    },

    /// `vectorchain.element_wise_selection %cond ? %lhs : %rhs [%mask : t] : tc, tl, tr, tres`.
    ElementWiseSelection {
        /// The vector it binds.
        result: Val,
        /// The i1 vector chosen by, CARRYING ITS OWN TYPE.
        ///
        /// ⛔ THIS USED TO BE A `Val` BESIDE A `cond_ty: Vector` THE EMITTER FILLED IN — the same
        /// two-records shape that made the mask disagree with itself. A `Predicate` states it once.
        cond: Predicate,
        /// Taken where the condition holds.
        lhs: Val,
        /// Taken otherwise.
        rhs: Val,
        /// The lane mask, IF THIS OP CARRIES ONE. `Optional` in the dialect
        /// (`VectorChain.td:402`), and the condition is a SEPARATE operand from it.
        mask: Option<Predicate>,
        /// The operands' and result's type.
        ty: Vector,
    },

    /// `vectorchain.binary %op1, %op2 [%mask : t] {binary_op} : t1, t2, tres` — the elementwise family.
    Binary {
        /// The vector it binds.
        result: Val,
        /// Left operand.
        op1: Val,
        /// Right operand.
        op2: Val,
        /// The lane mask, IF THIS OP CARRIES ONE.
        ///
        /// ⛔⛔ `None` MEANS NO BRACKET, NOT AN ALL-TRUE CONSTANT. Every binary used to be handed
        /// an `arith.constant true` as a stand-in for "unmasked", which is both an `i1` where a
        /// `vector<Nxi1>` was printed (the granite-2b `group_7` refusal) and a mask that masks
        /// nothing. `Optional<AnyVectorOfAnyRank>:$mask` (`VectorChain.td:139`) — an unmasked
        /// binary simply omits the operand.
        mask: Option<Predicate>,
        /// Which operation.
        binary_op: BinaryOp,
        /// `op_specific_map=` — REQUIRED, and dbo-opt says so: "'vectorchain.binary' op requires
        /// attribute 'op_specific_map'". IBM's own IR writes the identity, `affine_map<(d0) -> (d0)>`,
        /// which is lane `i` of each operand into lane `i` of the result.
        op_specific_map: AffineMap,
        /// The operands' type.
        operand_ty: Vector,
        /// The result's type.
        ty: Vector,
    },

    /// `vectorchain.pack %op1, %op2 [%mask : t] {indices, repetition, sign_extend} : t1, t2, tres` —
    /// TWO VECTORS INTERLEAVED BY AN INDEX LIST, and the whole merge/pack/gcvt/fcvt family is this
    /// one op.
    ///
    /// ⛔⛔ THERE IS NO `vectorchain.merge_8h` AND NO `vectorchain.gcvt`. Every one of the
    /// thirty-four `merge<w><half>` and `pack<n>` mnemonics — and every `gcvt_imm<n>`/`fcvt_imm<n>`
    /// — is a `vectorchain.pack` distinguished ONLY by its `indices` and `sign_extend`, which is
    /// what `getMergeTypeFromIndices` and `getGCVTorFCVTTypeFromIndicesAndCastInputs` decode
    /// (`VectorChainHelper.cpp:301-414`, `:188-300`). IBM's own text spells `%merge8h =
    /// vectorchain.pack %x128i8, %z128i8 {indices = [1,17,3,19,…]}`
    /// (`dcc/test/SFP/merge_and_pack.mlir:161`) — the `merge8h` there is an SSA NAME, not a
    /// mnemonic.
    ///
    /// ⛔ AND WHICH OF THE TWO DECODERS RUNS IS DECIDED BY THE OPERANDS' DEFINING OPS, NOT BY THIS
    /// OP: both operands being [`Op::Cast`]s makes it a convert, neither being one makes it a
    /// merge/pack, and one of each is an error the reference emits
    /// (`VectorChainToSentientPESFP.cpp:693-715`).
    Pack {
        /// The vector it binds.
        result: Val,
        /// Left operand.
        op1: Val,
        /// Right operand.
        op2: Val,
        /// The lane mask, IF THIS OP CARRIES ONE — `Optional<AnyVectorOfAnyRank>:$mask`
        /// (`VectorChain.td:208`), so `None` prints no bracket at all.
        mask: Option<Predicate>,
        /// Which element of the concatenated pair each result element reads.
        ///
        /// ⛔ NEGATIVE ENTRIES ARE REAL AND MEAN "PAD". `pack12` is
        /// `[0, -1, 1, -1, 2, -1, …]` (`dcc/test/SFP/merge_and_pack.mlir:176`) — an `i32` list, not
        /// a `u32` one, and a `usize` here would refuse half the family.
        indices: Vec<i32>,
        /// How many times the index pattern repeats to fill the result — `IndexAttr`, printed
        /// `: index` and not `: i32`.
        ///
        /// ⭐ ALWAYS EIGHT IN THE REFERENCE'S OWN TABLE: every one of the thirty-four entries of
        /// `getMergeTypeFromIndices` is matched at `repetition == 8` and none of them overrides it.
        repetition: u32,
        /// Whether the pad lanes carry the sign — `true` for `pack14` and `pack15` alone, of the
        /// whole thirty-four-entry table.
        sign_extend: bool,
        /// The operands' type.
        operand_ty: Vector,
        /// The result's type.
        ty: Vector,
    },

    /// `vectorchain.merge %op1, %op2 {iteration_space_CA, access_function_CA, access_function_A,
    /// iteration_space_CB, access_function_CB, access_function_B} : t1, t2, tres` — a merge stated
    /// as a polyhedral mini-scop, *"where the order doesn't matter"* (`VectorChain.td:164-185`).
    ///
    /// ⛔⛔ A DIFFERENT OP FROM [`Op::Pack`], DESPITE THE NAME THE `pack` FAMILY'S MNEMONICS CARRY.
    /// The `merge8h` in a lowered program came from a `vectorchain.pack`; THIS op states two
    /// (iteration space, access function) pairs and no index list at all. `getComputePrecisionOfOp`
    /// short-circuits on both (`isa<PackOp, MergeOp>`), which is the only place in bridge 2 they are
    /// treated alike.
    ///
    /// ⚠️ NO VENDOR TEXT EXERCISES IT. `vectorchain.merge` appears in none of the authority tree's
    /// 825 `.mlir` cases, so its printed form here is derived from the `assemblyFormat` in the `.td`
    /// and is NOT byte-checked against IBM output the way [`Op::Pack`]'s is. Nothing emits one yet;
    /// it exists because `getComputePrecisionOfOp` must be able to be asked about one.
    ///
    /// ⛔ AND IT IS ALSO WHY `getElementType` MUST NOT BE REACHED FOR IT: `MergeOp` is absent from
    /// both `getVectorType` and `getCustomVectorType` (`Utils.cpp:538-694`), so an element-type
    /// query on one hits `DT_CHECK_MSG(.., "Type is not a known vector type")`.
    Merge {
        /// The vector it binds.
        result: Val,
        /// Left operand.
        op1: Val,
        /// Right operand.
        op2: Val,
        /// `iteration_space_CA` — which indices of the combined space the A side is written over.
        iteration_space_ca: IntegerSet,
        /// `access_function_CA` — where in the result the A side lands.
        access_function_ca: AffineMap,
        /// `access_function_A` — where in `op1` the A side is read from.
        access_function_a: AffineMap,
        /// `iteration_space_CB` — as `iteration_space_CA`, for the B side.
        iteration_space_cb: IntegerSet,
        /// `access_function_CB` — as `access_function_CA`, for the B side.
        access_function_cb: AffineMap,
        /// `access_function_B` — as `access_function_A`, for `op2`.
        access_function_b: AffineMap,
        /// The operands' type.
        operand_ty: Vector,
        /// The result's type.
        ty: Vector,
    },

    /// `vectorchain.constant_bitstream {value = [0x0, 0x1]} : vector<Nxt>` — a literal, as wide as
    /// the constant the template declares.
    ///
    /// ⭐ THE VALUES ARE ELEMENT BIT PATTERNS, which is what `ddl.define_constant`'s `value=[0xFFFF]`
    /// already holds — the two forms line up exactly.
    ConstantBitstream {
        /// The vector it binds.
        result: Val,
        /// The element bit patterns.
        value: Vec<i64>,
        /// Its type — as many elements as `value` has.
        ty: Vector,
        /// `is_symbol` — the values are SYMBOL IDS a later pass resolves, not bit patterns.
        ///
        /// ⛔⛔ IT CHANGES THE WHOLE PRINTED FORM, not just one attribute.
        /// `ConstantBitstreamOp::print` reads `is_symbol` FIRST and prints the hand-rolled
        /// `{value = [0x..]}` only when it is absent or false; when it is true the op prints its
        /// generic attribute dictionary instead — decimal `N : i64` values, `is_symbol` before
        /// `value` (`VectorChain.cpp:78-105`). `constructUniformizedFoldedConstantBitStream` sets it
        /// on every bitstream it builds when its `is_symbolic` argument is set
        /// (`SNDSCLowering.cpp:479-481, 519-521`).
        is_symbol: bool,
    },

    /// `vectorchain.shuffle input(%c) {indices = [..], repetition = N} : tin, tout` — the splat that
    /// widens a constant bitstream to the width its consumer reads.
    ///
    /// ⛔ `repetition` IS A QUOTIENT, NOT A COUNT SOMEONE PICKS:
    /// `getNumElements(result_type) / getNumElements(bitstream_vector_type)`
    /// (`SNTransferLowering.cpp:2505-2506`).
    Shuffle {
        /// The vector it binds.
        result: Val,
        /// What is widened.
        input: Val,
        /// The SCALARS a negative index reaches — `variable(%iv)`, indexed from `-1`
        /// (`VectorChain.td:445`).
        ///
        /// ⛔⛔ A NEGATIVE INDEX IS NOT A REORDERING, IT IS A DIFFERENT OPERAND. `indices = [-1]`
        /// broadcasts `variable`'s first scalar over the whole result and reads NOTHING out of
        /// `input`, so an empty list here with a negative index is an op whose elements have no
        /// source. That is what makes this a field rather than an attribute: the value travels.
        variable: Vec<ShuffleVariable>,
        /// `dbgName`.
        dbg_name: Option<String>,
        /// One index per element of the input, or a negative one per [`Op::Shuffle::variable`].
        indices: Vec<i32>,
        /// How many times the pattern repeats to fill the result.
        repetition: u32,
        /// The input's type.
        input_ty: Vector,
        /// The result's type.
        ty: Vector,
    },

    /// `vectorchain.rotate %in, %pos {right_shift} : tin, index, tout` — the circular shift.
    ///
    /// ⛔ ADDED FOR THE CHAIN THAT SITS BETWEEN A LOAD AND ITS SEND. `getLoadConsumer` accepts a
    /// load whose single user is a `SelectOp`, a `ShuffleOp` **or a `RotateOp`** before the send
    /// (`Helper.cpp:1268-1270`), and `addLoadChainToDeleteList` deletes the same three
    /// (`Helper.cpp:2977-2980`). With two of the three in this enum the third was a shape the
    /// island could not state, so those two functions could not be ported over their own domain.
    ///
    /// ⛔ THE POSITION IS AN SSA VALUE, NOT AN ATTRIBUTE (`VectorChain.td:552-555`), and the op
    /// *"views the vector as a circular array"* — linearised first when it is multi-dimensional.
    Rotate {
        /// The vector it binds.
        result: Val,
        /// What is rotated.
        input: Val,
        /// `$position` — where the rotation starts, as an `index` value.
        position: Val,
        /// `$right_shift` — ⛔ the `.td` DEFAULTS IT TO TRUE (`VectorChain.td:555`), so a left
        /// rotate is the one that has to say so.
        right_shift: bool,
        /// The input's type.
        input_ty: Vector,
        /// The result's type.
        ty: Vector,
    },

    /// `vectorchain.cast %v : tin, tout` — a precision conversion.
    ///
    /// ⛔ THE C++ EMITS ONE WHERE THE SOURCE AND DESTINATION FORMATS DIFFER: "Convert data if src
    /// precision and dst precision don't match" (`SNTransferLowering.cpp:2305-2323`), applied to the
    /// data AFTER the load or shuffle — which is why a shuffle's element type is the SOURCE's, and
    /// why `vectorchain.shuffle` refuses to change it: "input element type does not match output
    /// element type".
    Cast {
        /// The vector it binds.
        result: Val,
        /// What is converted.
        input: Val,
        /// The input's type.
        input_ty: Vector,
        /// The result's type.
        ty: Vector,
    },

    /// `vectorchain.create_affine_mask {mask_set} : vector<Nxi1>` — the lane mask every elementwise
    /// op takes, from `getStaticContinuousMaskValue`.
    CreateAffineMask {
        /// The i1 vector it binds.
        result: Val,
        /// THE MASK ITSELF — its live prefix AND the type it is stated over, as one value.
        ///
        /// ⛔⛔ `lanes: u64` AND `ty: Vector` AS SEPARATE FIELDS WERE TWO RECORDS OF ONE FACT. The
        /// definition printed from `ty` while every USE recomputed `vector<{len}xi1>` from the
        /// operands — which is how `%42` came to be defined as `i1` and used as `vector<64xi1>`.
        /// [`LaneMask::binds`] hands the SAME `Vector` to the use site, so they cannot differ.
        mask: LaneMask,
    },

    /// `vectorchain.create_affine_mask [%parameter] {mask_set} : vector<Nxi1>` — the SAME operation,
    /// stated as the affine set it actually carries.
    ///
    /// ⛔⛔ NOT A DUPLICATE OF [`Op::CreateAffineMask`] — THE PREFIX FORM CANNOT STATE A PT MASK.
    /// [`LaneMask`] holds a *live lane count*, which is enough for `getStaticContinuousMaskValue` and
    /// exactly what defends the definition/use type agreement there. But `getMaskValueForPT`
    /// (entry 089) reads FOUR facts off this op that a lane count does not carry
    /// (`Conversion/VectorChainLowering/VectorChainToSentientPT/Helper.cpp:26-215`):
    ///
    /// * `mask_set.getNumDims()` and `mask_set.getNumSymbols()` — separate refusals, `:32` and `:82`;
    /// * the CONSTRAINTS, matched term for term against
    ///   `d0 + s0 * num_lanes_in_slice - op_num_elems` and `-d0 + (op_num_elems - 1)` (`:203-215`);
    /// * the set's lower bound, which becomes the mask value (`:88-101`);
    /// * `cam_op.getMaskParameter()` — present or absent, and if present, whether it is an
    ///   `arith.constant` or an `arith.subi` (`:64-79`, `:122-131`).
    ///
    /// The vendor writes both forms in one file:
    /// `vectorchain.create_affine_mask %14 {mask_set = #set} : vector<64xi1>` and
    /// `vectorchain.create_affine_mask {mask_set = #set2} : vector<64xi1>`
    /// (`dcc/test/Conversion/VectorChainToSentientPT/dynamic_pt_masking.mlir:252`, `:296`).
    ///
    /// ⭐ ADDITIVE ON PURPOSE. `tape.rs` is the only emitter of the prefix form and its type-agreement
    /// defence is worth keeping intact, so this variant sits beside it rather than replacing it, and
    /// both bind their predicate through [`Op::binds_predicate`].
    CreateAffineMaskSet {
        /// The i1 vector it binds.
        result: Val,
        /// `$mask_set` — the affine set, dims and symbols and all.
        mask_set: IntegerSet,
        /// `$mask_parameter` — the set's one symbol, when the set takes one.
        ///
        /// ⛔ `Optional` IN THE DIALECT, AND ITS ABSENCE IS A FACT ENTRY 089 READS:
        /// `static_mask = !cam_op.getMaskParameter()` (`Helper.cpp:64`).
        mask_parameter: Option<Val>,
        /// The i1 vector type the mask is stated over — printed HERE and read from here by every use.
        ty: Vector,
    },
}

impl Op {
    /// THE PREDICATE THIS OP BINDS, TAKING ITS TYPE FROM THE DEFINITION — `None` for an op that
    /// binds no mask.
    ///
    /// ⭐ THE ONE WAY EITHER MASK FORM REACHES A USE. [`Predicate`]'s fields are private and it has
    /// no public constructor precisely so that a use cannot restate the type
    /// (see [`Predicate`]'s own note on granite-2b's `group_7`); this method extends that guarantee
    /// to [`Op::CreateAffineMaskSet`] without opening one.
    #[must_use]
    pub fn binds_predicate(&self) -> Option<Predicate> {
        match self {
            Op::CreateAffineMask { result, mask } => Some(mask.binds(*result)),
            Op::CreateAffineMaskSet { result, ty, .. } => Some(Predicate {
                val: *result,
                ty: *ty,
            }),
            _ => None,
        }
    }
}

/// ONE `vectorchain` OP AS TEXT. The caller has already indented.
pub(crate) fn emit(out: &mut String, op: &Op) {
    match op {
        Op::Estimate {
            result,
            input,
            kind,
            version,
            input_ty,
            ty,
        } => {
            let version = match version {
                Some(v) => format!(
                    " {{version = #vectorchain<{} {}>}}",
                    kind.version_mnemonic(),
                    v.spelling()
                ),
                None => String::new(),
            };
            let _ = writeln!(
                out,
                "{} = vectorchain.{} {}{version} : {}, {}",
                print::val(*result),
                kind.spelling(),
                print::val(*input),
                print::vector(*input_ty),
                print::vector(*ty)
            );
        }
        // ⭐ ONE ARM FOR THE TWO UNARY, ATTRIBUTE-FREE OPS — the mnemonic is the only difference,
        // and it is named here rather than derived so that neither can print as the other.
        Op::FastExp {
            result,
            input,
            input_ty,
            ty,
        }
        | Op::Floor {
            result,
            input,
            input_ty,
            ty,
        } => {
            let mnemonic = match op {
                Op::FastExp { .. } => "fast_exp",
                _ => "floor",
            };
            let _ = writeln!(
                out,
                "{} = vectorchain.{mnemonic} {} : {}, {}",
                print::val(*result),
                print::val(*input),
                print::vector(*input_ty),
                print::vector(*ty)
            );
        }
        // ⛔ ITS OWN ARM BECAUSE OF THE BRACKET: `$op (`[` $mask^ `:` type($mask) `]`)? attr-dict `:`
        // type($op) `,` type(results)` (`VectorChain.td:370-372`) — the mask sits BEFORE the `:`, so
        // folding this into the attribute-free unary arm above would drop it.
        Op::Neg {
            result,
            input,
            mask,
            input_ty,
            ty,
        } => {
            let _ = writeln!(
                out,
                "{} = vectorchain.neg {}{} : {}, {}",
                print::val(*result),
                print::val(*input),
                mask_bracket(*mask),
                print::vector(*input_ty),
                print::vector(*ty)
            );
        }
        Op::ScanWithGap {
            result,
            input,
            reduction_op,
            input_ty,
            ty,
        } => {
            let _ = writeln!(
                out,
                "{} = vectorchain.scan_with_gap {} {{reduction_op = #vectorchain<binary_operator {}>, \
                 gap = 8 : index, eval_order = #vectorchain<eval_order left_to_right>}} : {}, {}",
                print::val(*result),
                print::val(*input),
                reduction_op.spelling(),
                print::vector(*input_ty),
                print::vector(*ty)
            );
        }
        Op::Select {
            result,
            input,
            selection_map,
            input_ty,
            ty,
        } => {
            let _ = writeln!(
                out,
                "{} = vectorchain.select {} {{selection_map = {}}} : {}, {}",
                print::val(*result),
                print::val(*input),
                print::affine_map(selection_map),
                print::vector(*input_ty),
                print::vector(*ty)
            );
        }
        Op::Multiply {
            result,
            a,
            b,
            reduction_map,
            operand_ty,
            ty,
        } => {
            let _ = writeln!(
                out,
                "{} = vectorchain.multiply {}, {} {{reduction_map = {}}} : {}, {}, {}",
                print::val(*result),
                print::val(*a),
                print::val(*b),
                print::affine_map(reduction_map),
                print::vector(*operand_ty),
                print::vector(*operand_ty),
                print::vector(*ty)
            );
        }
        Op::MultiplyAccumulate {
            result,
            a,
            b,
            acc,
            reduction_map,
            operand_ty,
            ty,
        } => {
            let _ = writeln!(
                out,
                "{} = vectorchain.multiply_and_accumulate {}, {}, {} {{reduction_map = {}}} : {}, {}, {}, {}",
                print::val(*result),
                print::val(*a),
                print::val(*b),
                print::val(*acc),
                print::affine_map(reduction_map),
                print::vector(*operand_ty),
                print::vector(*operand_ty),
                print::vector(*ty),
                print::vector(*ty)
            );
        }
        Op::ElementWiseCompare {
            result,
            op1,
            op2,
            mask,
            compare_op,
            operand_ty,
            ty,
        } => {
            let _ = writeln!(
                out,
                "{} = vectorchain.element_wise_compare {}, {}{} {{compare_op = #vectorchain<element_wise_compare_operator {}>}} : {}, {}, {}",
                print::val(*result),
                print::val(*op1),
                print::val(*op2),
                mask_bracket(*mask),
                compare_op.spelling(),
                print::vector(*operand_ty),
                print::vector(*operand_ty),
                print::vector(*ty)
            );
        }
        Op::ElementWiseSelection {
            result,
            cond,
            lhs,
            rhs,
            mask,
            ty,
        } => {
            // ⭐ THE CONDITION'S TYPE IS THE CONDITION'S OWN — it was `cond_ty`, a field the
            // emitter filled in beside the value, which is the same two-records shape.
            let _ = writeln!(
                out,
                "{} = vectorchain.element_wise_selection {} ? {} : {} {} : {}, {}, {}, {}",
                print::val(*result),
                print::val(cond.val()),
                print::val(*lhs),
                print::val(*rhs),
                mask_bracket(*mask),
                print::vector(cond.ty()),
                print::vector(*ty),
                print::vector(*ty),
                print::vector(*ty)
            );
        }
        Op::Binary {
            result,
            op1,
            op2,
            mask,
            binary_op,
            op_specific_map,
            operand_ty,
            ty,
        } => {
            let _ = writeln!(
                out,
                "{} = vectorchain.binary {}, {}{} {{binary_op = #vectorchain<binary_operator {}>, op_specific_map = {}}} : {}, {}, {}",
                print::val(*result),
                print::val(*op1),
                print::val(*op2),
                mask_bracket(*mask),
                binary_op.spelling(),
                print::affine_map(op_specific_map),
                print::vector(*operand_ty),
                print::vector(*operand_ty),
                print::vector(*ty)
            );
        }
        Op::Pack {
            result,
            op1,
            op2,
            mask,
            indices,
            repetition,
            sign_extend,
            operand_ty,
            ty,
        } => {
            // ⭐ THE ATTRIBUTES IN BYTE-WISE NAME ORDER, which is the order `attr-dict` prints:
            // `indices`, `repetition`, `sign_extend`.
            //
            // ⛔ `: index` ON THE REPETITION, NOT `: i32`. `IndexAttr:$repetition`
            // (`VectorChain.td:207`) — where [`Op::Shuffle`]'s is an `i32`, so the two ops spell the
            // same word differently and swapping them is a parse failure.
            //
            // ⭐ AND THE INDICES CARRY `: i32` EACH. `ArrayAttr:$indices` holds `IntegerAttr`s of
            // whatever width built them, and `checkValidityOfPackAndShuffleLowering` accepts any
            // (`VectorChainHelper.cpp:156-167`) — so the width is a choice, and this is IBM's:
            // `indices = [0 : i32, 2 : i32, …]`
            // (`dcc/test/Conversion/VectorChainToSentientPESFP/fpuop.mlir:217`), matching what
            // [`Op::Shuffle`] already prints. `sign_extend` is spelled out even when false, as that
            // line spells it.
            let indices = indices
                .iter()
                .map(|index| format!("{index} : i32"))
                .collect::<Vec<_>>()
                .join(", ");
            let _ = writeln!(
                out,
                "{} = vectorchain.pack {}, {}{} {{indices = [{indices}], repetition = {repetition} : index, sign_extend = {sign_extend}}} : {}, {}, {}",
                print::val(*result),
                print::val(*op1),
                print::val(*op2),
                mask_bracket(*mask),
                print::vector(*operand_ty),
                print::vector(*operand_ty),
                print::vector(*ty)
            );
        }
        Op::Merge {
            result,
            op1,
            op2,
            iteration_space_ca,
            access_function_ca,
            access_function_a,
            iteration_space_cb,
            access_function_cb,
            access_function_b,
            operand_ty,
            ty,
        } => {
            // ⭐ BYTE-WISE NAME ORDER AGAIN, and here it interleaves the two sides rather than
            // grouping them: `access_function_A`, `access_function_B`, `access_function_CA`,
            // `access_function_CB`, `iteration_space_CA`, `iteration_space_CB` — `'A' < 'B' < 'C'`,
            // so the A and B access functions come BEFORE either CA or CB.
            //
            // ⚠️ DERIVED FROM THE `assemblyFormat`, NOT FROM VENDOR OUTPUT: `vectorchain.merge`
            // occurs in none of the 825 authority tests. See [`Op::Merge`].
            let _ = writeln!(
                out,
                "{} = vectorchain.merge {}, {} {{access_function_A = {}, access_function_B = {}, access_function_CA = {}, access_function_CB = {}, iteration_space_CA = {}, iteration_space_CB = {}}} : {}, {}, {}",
                print::val(*result),
                print::val(*op1),
                print::val(*op2),
                print::affine_map(access_function_a),
                print::affine_map(access_function_b),
                print::affine_map(access_function_ca),
                print::affine_map(access_function_cb),
                print::integer_set(iteration_space_ca),
                print::integer_set(iteration_space_cb),
                print::vector(*operand_ty),
                print::vector(*operand_ty),
                print::vector(*ty)
            );
        }
        Op::ConstantBitstream {
            result,
            value,
            ty,
            is_symbol,
        } => {
            // ⛔ TWO FORMS, PICKED BY `is_symbol` (`VectorChain.cpp:80-101`): the custom one prints
            // the values as hex with no type suffix, the generic attribute dictionary prints them
            // decimal with one — and puts `is_symbol` first, because a `DictionaryAttr` is sorted by
            // name. Not a cosmetic difference: `0x2a` and `42 : i64` are different attributes to the
            // parser on the far side.
            let attrs = if *is_symbol {
                let values = value
                    .iter()
                    .map(|v| format!("{v} : i64"))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("{{is_symbol = true, value = [{values}]}}")
            } else {
                let values = value
                    .iter()
                    .map(|bits| format!("{bits:#x}"))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("{{value = [{values}]}}")
            };
            let _ = writeln!(
                out,
                "{} = vectorchain.constant_bitstream {attrs} : {}",
                print::val(*result),
                print::vector(*ty)
            );
        }
        Op::Shuffle {
            result,
            input,
            variable,
            dbg_name,
            indices,
            repetition,
            input_ty,
            ty,
        } => {
            let indices = indices
                .iter()
                .map(|index| format!("{index} : i32"))
                .collect::<Vec<_>>()
                .join(", ");
            // `if (numVariable > 0) p << ", variable(" .. ")"` (`VectorChain.cpp:365-372`) — the
            // group is absent, not empty, when there is none.
            let variables = if variable.is_empty() {
                String::new()
            } else {
                let vals = variable
                    .iter()
                    .map(|scalar| print::val(scalar.val))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!(", variable({vals})")
            };
            // ⭐ ALPHABETICAL, WHICH IS `printOptionalAttrDict`'s ORDER (`:385`) — `dbgName` sorts
            // ahead of `indices` and `repetition`, and it is the only one that can be absent.
            let name = match dbg_name {
                Some(name) => format!("dbgName = \"{name}\", "),
                None => String::new(),
            };
            // `p << " : " << input; for (variable) p << ", " << type; p << ", " << result`
            // (`:387-395`) — one type per operand, in operand order, result last.
            let mut types = print::vector(*input_ty);
            for scalar in variable {
                let _ = write!(types, ", {}", scalar.ty.spelling());
            }
            let _ = writeln!(
                out,
                "{} = vectorchain.shuffle input({}){variables} {{{name}indices = [{indices}], repetition = {repetition} : i32}} : {types}, {}",
                print::val(*result),
                print::val(*input),
                print::vector(*ty)
            );
        }
        Op::Rotate {
            result,
            input,
            position,
            right_shift,
            input_ty,
            ty,
        } => {
            // `$input `,` $position attr-dict `:` type($input) `,` type($position) `,`
            // type(results)` (`VectorChain.td:558-560`) — and the position's type is always
            // `index`, which is what the operand is declared as.
            let _ = writeln!(
                out,
                "{} = vectorchain.rotate {}, {} {{right_shift = {right_shift}}} : {}, index, {}",
                print::val(*result),
                print::val(*input),
                print::val(*position),
                print::vector(*input_ty),
                print::vector(*ty)
            );
        }
        Op::Cast {
            result,
            input,
            input_ty,
            ty,
        } => {
            let _ = writeln!(
                out,
                "{} = vectorchain.cast {} : {}, {}",
                print::val(*result),
                print::val(*input),
                print::vector(*input_ty),
                print::vector(*ty)
            );
        }
        Op::CreateAffineMask { result, mask } => {
            // A continuous prefix of live lanes, as `getStaticContinuousMaskValue` builds.
            //
            // ⭐ THE TYPE PRINTED HERE IS THE ONE EVERY USE PRINTS, because both read it off the
            // same `LaneMask`. That is what makes the definition and the use unable to disagree.
            let _ = writeln!(
                out,
                "{} = vectorchain.create_affine_mask {{mask_set = affine_set<(d0) : (d0 >= 0, -d0 + {} >= 0)>}} : {}",
                print::val(*result),
                mask.live().saturating_sub(1),
                print::vector(mask.ty())
            );
        }
        Op::CreateAffineMaskSet {
            result,
            mask_set,
            mask_parameter,
            ty,
        } => {
            // `($mask_parameter^)? attr-dict `:` type(results)` — the operand is optional and the
            // vendor omits it entirely when the set has no symbol (`dynamic_pt_masking.mlir:296`).
            let parameter = match mask_parameter {
                Some(v) => format!(" {}", print::val(*v)),
                None => String::new(),
            };
            let _ = writeln!(
                out,
                "{} = vectorchain.create_affine_mask{parameter} {{mask_set = {}}} : {}",
                print::val(*result),
                print::integer_set(mask_set),
                print::vector(*ty)
            );
        }
    }
}

/// THE `[%mask : t]` BRACKET, OR NOTHING AT ALL.
///
/// ⛔⛔ THE TYPE COMES FROM THE PREDICATE, NEVER FROM THE OPERANDS. `fn mask_ty(ty: Vector)` used to
/// build `vector<{ty.len}xi1>` here at the USE — a second record of a type the DEFINITION had
/// already stated — and dbo-opt refused granite-2b's `group_7` with *"use of value '%42' expects
/// different type than prior uses: 'vector<64xi1>' vs 'i1'"*.
///
/// ⭐ AND `None` PRINTS NOTHING. Every masked op's `$mask` is `Optional` in the dialect, and the
/// assembly format wraps the bracket in `(...)?` — so an unmasked op omits the operand rather than
/// carrying an all-true constant that masks nothing.
fn mask_bracket(mask: Option<Predicate>) -> String {
    match mask {
        Some(p) => format!("[{} : {}]", print::val(p.val()), print::vector(p.ty())),
        None => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use crate::islands::dataflow_ir::dialects::Val;
    use crate::islands::dataflow_ir::dialects::vectorchain::{Op, emit};
    use crate::islands::dataflow_ir::ty::{ElemType, Vector};

    /// ⭐⭐ A BITSTREAM'S TWO PRINTED FORMS, AND `is_symbol` IS WHAT PICKS BETWEEN THEM.
    ///
    /// `ConstantBitstreamOp::print` reads the attribute FIRST and takes the hand-rolled hex form only
    /// when it is absent or false; when it is true the op prints its generic attribute dictionary
    /// instead (`VectorChain.cpp:80-101`). So the same values print as `0x2a` in one case and
    /// `42 : i64` in the other, with `is_symbol` ahead of `value` because a `DictionaryAttr` is sorted
    /// by name.
    ///
    /// ⛔ A SHAPE ASSERTION PASSES ON BOTH. "It names the op and carries the value" holds for
    /// `{value = [0x2a]}` and `{is_symbol = true, value = [42 : i64]}` alike, and the far side parses
    /// exactly one of them for a given op — which is why this is checked as bytes.
    #[test]
    fn a_symbolic_bitstream_prints_its_attribute_dictionary() {
        let f16x64 = Vector {
            len: 64,
            elem: ElemType::F16,
        };

        let mut literal = String::new();
        emit(
            &mut literal,
            &Op::ConstantBitstream {
                result: Val(3),
                value: vec![42, 1],
                ty: f16x64,
                is_symbol: false,
            },
        );
        assert_eq!(
            literal,
            "%3 = vectorchain.constant_bitstream {value = [0x2a, 0x1]} : vector<64xf16>\n"
        );

        let mut symbolic = String::new();
        emit(
            &mut symbolic,
            &Op::ConstantBitstream {
                result: Val(3),
                value: vec![42, 1],
                ty: f16x64,
                is_symbol: true,
            },
        );
        assert_eq!(
            symbolic,
            "%3 = vectorchain.constant_bitstream {is_symbol = true, value = [42 : i64, 1 : i64]} : \
             vector<64xf16>\n"
        );
    }

    /// ⭐⭐ IBM'S OWN PACK LINE, REPRODUCED BYTE FOR BYTE.
    ///
    /// `dcc/test/Conversion/VectorChainToSentientPESFP/fpuop.mlir:217` — the whole
    /// merge/pack/gcvt/fcvt family is this one op, so the shape of this single line is the shape of
    /// all thirty-four mnemonics, and its two consumers (`getMergeTypeFromIndices` and
    /// `getGCVTorFCVTTypeFromIndicesAndCastInputs`) read nothing but `indices`, `repetition` and
    /// `sign_extend`.
    ///
    /// ⛔ THE TWO SPELLINGS THAT ARE ONLY CHECKABLE AGAINST BYTES: `repetition = 8 : index` where
    /// [`Op::Shuffle`]'s same-named attribute is `: i32`, and `sign_extend = false` printed rather
    /// than elided. A shape assertion — "it names the op and lists sixteen indices" — passes on both
    /// wrong spellings.
    #[test]
    fn reproduces_ibms_pack_line() {
        let i8x128 = Vector {
            len: 128,
            elem: ElemType::Int(8),
        };
        let mut out = String::new();
        emit(
            &mut out,
            &Op::Pack {
                result: Val(29),
                op1: Val(24),
                op2: Val(27),
                mask: None,
                indices: (0..16).map(|i| i * 2).collect(),
                repetition: 8,
                sign_extend: false,
                operand_ty: i8x128,
                ty: i8x128,
            },
        );
        assert_eq!(
            out,
            "%29 = vectorchain.pack %24, %27 {indices = [0 : i32, 2 : i32, 4 : i32, 6 : i32, \
             8 : i32, 10 : i32, 12 : i32, 14 : i32, 16 : i32, 18 : i32, 20 : i32, 22 : i32, \
             24 : i32, 26 : i32, 28 : i32, 30 : i32], repetition = 8 : index, sign_extend = false} \
             : vector<128xi8>, vector<128xi8>, vector<128xi8>\n"
        );
    }
}
