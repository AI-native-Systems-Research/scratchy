//! THE VECTOR CHAINS — mac, binary and unary computes, and the precision they run at.
//! ⭐ ROPE IS TWO FMA STAGES, NOT A MULTIPLY: rope.ddl:26-27 declares rope64p1/rope64p2, two chained
//! FMA16 stages through an intermediate, and the m2 transfer's rotate_num_elements=32 IS the pair swap.
//!
//! 18 units. Every citation resolves against the authority tree
//! `/Users/nickm/git/deeptools-src` at revision `a0d29abbed`.
//!
//! | unit | level | LoC | authority |
//! |---|---|---|---|
//! | `e014_constructTypeFromFormat` | 0 | 91 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNComputeLowering.cpp:278` |
//! | `e015_constructSingleValCustomVector` | 0 | 15 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNComputeLowering.cpp:435` |
//! | `e016_mapToDicAttr` | 0 | 9 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNComputeLowering.cpp:1524` |
//! | `e047_getStaticContinuousMaskValue` | 1 | 23 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNComputeLowering.cpp:33` |
//! | `e048_constructDynamicMasking` | 1 | 43 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNComputeLowering.cpp:60` |
//! | `e049_getTypeBasedOnComputeType` | 1 | 12 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNComputeLowering.cpp:108` |
//! | `e050_getReductionMapForMACOperation` | 1 | 70 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNComputeLowering.cpp:126` |
//! | `e051_getSelectionMapForMACOperandFromL0` | 1 | 74 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNComputeLowering.cpp:200` |
//! | `e052_constructPrecisionConversionOperation` | 1 | 59 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNComputeLowering.cpp:375` |
//! | `e053_constructOpaqueOperation` | 1 | 25 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNComputeLowering.cpp:1534` |
//! | `e086_constructComputeInputOperandAndAddToList` | 3 | 320 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNComputeLowering.cpp:456` |
//! | `e087_constructComputeOutputOperand` | 3 | 139 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNComputeLowering.cpp:777` |
//! | `e094_constructComputeOutputOperands` | 4 | 14 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNComputeLowering.cpp:921` |
//! | `e095_constructFMINorFMAXOperation` | 4 | 86 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNComputeLowering.cpp:1189` |
//! | `e099_constructMACOperation` | 5 | 87 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNComputeLowering.cpp:942` |
//! | `e100_constructBinaryOrTernaryOperation` | 5 | 152 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNComputeLowering.cpp:1036` |
//! | `e101_constructUnaryOperation` | 5 | 247 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNComputeLowering.cpp:1276` |
//! | `e103_constructComputeOperation` | 6 | 83 | `dsc-based-utils/DSC2ToDataflowIR/V3/SNComputeLowering.cpp:1567` |

use super::construction::{MaskValue, static_continuous_mask};
use super::control_flow::{PrimaryDim, StartAddresses};
use super::dsc_lowering::{
    Component, DataLocation, Handlers, Handles, address_granularity_multiply_factor,
    constant_index, create_get_unit_op_in_different_core, emit_error, mlir_loop_from_loop_node,
    uniformized_folded_address, uniformized_folded_destination_core,
};
use super::transfer::{
    AddressedAs, AgenStorage, Latch, LoopStride, TransferElements, TransferSide, ViewSize,
    elements_of_affine_data_transfer_via_agen_transfer,
};
use crate::arch::{Arch, Elements, IsaGen};
use crate::generated::{DataType, OpaqueFunc, ParamKey, ParamValue, RegName};
use crate::islands::dataflow_ir::Values;
use crate::islands::dataflow_ir::dialects::dataflow::{Opaque, RegAddr, Received};
use crate::islands::dataflow_ir::dialects::vectorchain::{Computed, Predicate};
use crate::islands::dataflow_ir::dialects::{Op, Val, agen, arith, dataflow, vectorchain};
use crate::islands::dataflow_ir::link::{self as link, Link, RecvEnd, SendEnd};
use crate::islands::dataflow_ir::ty::{
    AffineExpr, AffineMap, Constraint, ElemType, GenericComp, IntegerSet, ScalarTy, TensorCategory,
    Vector,
};
use crate::units::{Core, Corelet, DfirUnit, NumFolds};

/// ⛔ A STICK IN **BITS** — 128 bytes, and the reference divides by it in bits (`(128 * 8) / width`,
/// `SNComputeLowering.cpp:363-370`).
const STICK_BITS: u32 = 128 * 8;

// ⛔ ONE `crustify:todo:` PER SCHEDULED UNIT. Replace each with the ported function
// carrying `/// Replaces: eNNN_name`. A surviving TODO is open work.

/// AN ELEMENT WIDTH THAT FILLS A WHOLE NUMBER OF STICKS — the reference's `DT_CHECK_MSG` as a type.
///
/// ⛔⛔ IT IS THE **MLIR** WIDTH, NOT THE PACKING WIDTH. `constructTypeFromFormat` divides by
/// `element_type.getIntOrFloatBitWidth()`, so `SENINT24` on the PT is 24 here where
/// [`DataType::bits`] says 16 (`sendefs.cpp:135`) — and 1024 is not divisible by 24, which is
/// exactly the case the reference aborts on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StickWidth(u32);

impl StickWidth {
    /// THE WIDTH ONE ELEMENT OF THIS FORMAT OCCUPIES, when a stick is divisible by it.
    ///
    /// ⛔ `BOOL` IS SIXTEEN, NOT ONE. `format != BOOL ? getIntOrFloatBitWidth() : 16`
    /// (`:365-366`) — the element type is `i1` and the width used to size the vector is not, so a
    /// boolean vector is 64 wide and not 1024.
    ///
    /// ⛔ `None` IS THE ABORT, and it is REACHABLE: `SENINT24` on the PT is 24 bits and
    /// `DT_CHECK_MSG((128 * 8) % width == 0, ..)` refuses it (`:367-370`). It is the TYPE that says
    /// so, because this crate cannot refuse at run time.
    #[must_use]
    pub const fn of(
        format: DataType,
        on: GenericComp,
        category: TensorCategory,
    ) -> Option<StickWidth> {
        let bits = match format {
            DataType::Bool => 16,
            _ => ElemType::of(format, on, category).bits(),
        };
        if bits != 0 && STICK_BITS % bits == 0 {
            Some(StickWidth(bits))
        } else {
            None
        }
    }

    /// HOW MANY ELEMENTS OF THAT WIDTH FILL ONE STICK — `(128 * 8) / width` (`:371`).
    #[must_use]
    pub const fn per_stick(self) -> Elements {
        Elements((STICK_BITS / self.0) as u64)
    }
}

/// HOW WIDE THE VECTOR IS — the reference's `int num_elements` and its `-1`.
///
/// ⛔ `-1` IS A SENTINEL, NOT A COUNT: it means *"one stick's worth of whatever this format is"*
/// (`:362-372`), and every other value is the caller's own element count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VectorWidth {
    /// The caller states the count.
    Given(Elements),
    /// The caller passed `-1`: fill one stick.
    OneStick(StickWidth),
}

impl VectorWidth {
    /// The element count either way.
    #[must_use]
    pub const fn elements(self) -> Elements {
        match self {
            VectorWidth::Given(count) => count,
            VectorWidth::OneStick(width) => width.per_stick(),
        }
    }
}

/// Replaces: e014_constructTypeFromFormat
///
/// THE VECTOR TYPE A DATA FORMAT COMPUTES IN — `SNComputeLowering.cpp:278`.
///
/// ⭐ THE ELEMENT HALF IS [`ElemType::of`], which is this function's `else if` chain already
/// transcribed; this is the `num_elements` half and the `constructVectorType` at `:374`.
///
/// ⛔ A CUSTOM VECTOR IS NOT A DIFFERENT SHAPE HERE. `constructVectorType` returns a
/// `CustomVectorType` exactly when the element is a `CustomMXFloatType` (`Dataflow/Utils.cpp:205`),
/// which [`ElemType::MxFloat`] already distinguishes — so one [`Vector`] answers for both.
#[must_use]
pub const fn type_from_format(
    format: DataType,
    on: GenericComp,
    category: TensorCategory,
    width: VectorWidth,
) -> Vector {
    Vector {
        len: width.elements().0,
        elem: ElemType::of(format, on, category),
    }
}

/// Replaces: e014_constructTypeFromFormat
///
/// ⭐ ITS THIRD OUT-PARAMETER, AND IT IS EXACTLY "THE ELEMENT IS AN INTEGER". Checked arm by arm
/// against all fifteen of the reference's `is_integer =` assignments: every `true` sets an
/// `IntegerType` and every `false` a float or MX type, so the flag carries nothing the type does not.
#[must_use]
pub const fn is_integer(elem: ElemType) -> bool {
    matches!(elem, ElemType::Int(_))
}

/// Replaces: e015_constructSingleValCustomVector
///
/// ONE VALUE SPLATTED ACROSS A CUSTOM VECTOR — `SNComputeLowering.cpp:435`.
///
/// ⛔⛔ `repetition` IS THE FULL ELEMENT COUNT HERE, NOT THE QUOTIENT. The bitstream is created
/// with `custom_vtype` itself while holding a SINGLE value, and `getI32IntegerAttr(getNumElements())`
/// is passed straight in (`:441-447`) — where `GenerateConstantBitStreamAndShuffle` sizes its
/// bitstream to the data and divides (`SNTransferLowering.cpp:2505-2506`).
///
/// ⛔ `indices` IS `{0}` — one index, so every lane reads element 0.
///
/// ⛔ AND THE BITSTREAM IS NOT SYMBOLIC. `is_symbol` is a discardable attribute the reference only
/// ever `setAttr`s (`SNDSCLowering.cpp:479-481`, `Splat.cpp:50-51`); this `create` does not, so the
/// op prints `{value = [..]}` with hex values rather than the `i64`-suffixed dictionary.
pub fn single_val_custom_vector(
    vals: &mut Values,
    into: &mut Vec<Op>,
    name: &str,
    value: i64,
    ty: Vector,
) -> Computed {
    let bitstream = vals.mint();
    into.push(Op::VectorChain(vectorchain::Op::ConstantBitstream {
        result: bitstream,
        value: vec![value],
        ty,
        is_symbol: false,
    }));
    let result = vals.mint();
    into.push(Op::VectorChain(vectorchain::Op::Shuffle {
        pad: Vec::new(),
        result,
        input: bitstream,
        variable: Vec::new(),
        // `getStringAttr(compute_op.name_)` (`:446`).
        dbg_name: Some(name.to_owned()),
        indices: vec![0],
        repetition: u32::try_from(ty.len).expect("a vector's element count fits a u32"),
        input_ty: ty,
        ty,
    }));
    Computed::of(result, ty)
}

/// Replaces: e016_mapToDicAttr
///
/// THE ORDER A DICTIONARY ATTRIBUTE'S ENTRIES ARE IN — `SNComputeLowering.cpp:1524`.
///
/// ⛔⛔ IT IS THE KEY'S SPELLING, NOT THE ORDER THEY WERE ADDED. The input is a
/// `std::map<std::string, std::string>`, so the range-`for` walks it in LEXICOGRAPHIC key order, and
/// `getDictionaryAttr` sorts by key again. A derived `Ord` on a key enum would give DECLARATION
/// order, which is a different sequence for any key set whose variants are not alphabetical.
#[must_use]
pub fn dictionary_order<K: Copy, V: Clone>(
    entries: &[(K, V)],
    spelling: impl Fn(K) -> &'static str,
) -> Vec<(K, V)> {
    let mut sorted = entries.to_vec();
    sorted.sort_by_key(|(key, _)| spelling(*key));
    sorted
}

/// Replaces: e047_getStaticContinuousMaskValue
///
/// THE SAME STATIC MASK, TAKEN OFF THE MASKED RESULT'S TYPE — `V3/SNComputeLowering.cpp:33`.
///
/// ⛔⛔ THIS IS ENTRY 046 WITH THE DEAD LINE REMOVED. The eight-row table (`:36-37`), the two
/// expressions (`:47-49`), the `IntegerSet::get(1, 0, ..)` (`:50`) and the `vector<{n}xi1>` result
/// (`:51`) are character for character `DataflowIRConstructionUtils.hpp:108-126`; the only
/// differences are that this one does not build the unused `arith.constant` — which is why entry 046
/// drops it too — that it reads `vector_dim` from `result_type` with `getDimSize(result_type, 0)`
/// (`:45`) instead of taking an `int`, and that its refusal is a diagnostic rather than a throw. So
/// this delegates; two transcriptions of one affine set is one too many.
///
/// ⛔⛔ AND THE REFUSAL IS WORSE THAN A THROW: `emitError` then `return nullptr` (`:39-40`), and the
/// caller does not test the result — `mask_op` goes straight into the compute's operands
/// (`:997-1008`, `:1014`). A [`MaskValue`] the caller already holds makes the null unrepresentable
/// rather than deferring it to the operand list.
///
/// ⭐ `vector_dim` IS THE MASKED VALUE'S OWN LANE COUNT, and the result carries it: a [`Predicate`]
/// states the type its definition printed, so the mask cannot reach a use under a different width.
pub fn static_mask_for_result(
    vals: &mut Values,
    into: &mut Vec<Op>,
    masked: Vector,
    mask: MaskValue,
) -> Predicate {
    static_continuous_mask(vals, into, masked.len, mask)
}

/// THE ONE-DIMENSIONAL DYNAMIC MASK A COMPUTE CARRIES — the two `DT_CHECK_MSG`s of
/// `V3/SNComputeLowering.cpp:66-75`, as a type.
///
/// ⛔⛔ BOTH ABORTS SAY THE SAME THING TWICE: *"Translator currently supports translating only 1-D
/// dynamic masking"* is asserted about the outer map's size (`:66-68`) and about the inner map's
/// (`:72-74`). One loop node, one dimension, one offset — a struct with one dim and one offset is
/// that pair of checks, and the caller's own `computeMaskLoopOffsets_.at(corelet_id)` (`:1007`) is
/// what selects it.
///
/// ⛔ THE MASK IS PT-ONLY. The caller aborts with *"Dynamic masking allowed only in PT units"*
/// before calling this (`:1002-1003`); the refusal belongs to the caller's component
/// classification, not to the set this builds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DynamicMask {
    /// `record->first` — which primary dimension the mask walks.
    pub dim: PrimaryDim,
    /// `record->second` — the offset the symbol's coefficient is scaled by.
    ///
    /// ⛔ EVERY WRITER WRITES 1, AND ONLY 1 SURVIVES THE NEXT BRIDGE. All three assignments to
    /// `computeMaskLoopOffsets_` store the literal 1 (`ddc/ddc_transformation.cpp:2220-2400`), and
    /// bridge 2's `getMaskValueForPT` accepts a dynamic set only when the symbol's coefficient is
    /// exactly `num_lanes_in_slice` — which `(vector_dim / 8) * offset` is only for `offset == 1`.
    /// It stays a number rather than becoming a unit type because it is DSC data, and a DSC that
    /// carries 2 must reach the reader that rejects it instead of being silently read as 1.
    pub offset: i64,
}

/// Replaces: e048_constructDynamicMasking
///
/// A `vectorchain.create_affine_mask` WHOSE FIRST MASKED LANE IS A LOOP'S INDUCTION VARIABLE —
/// `V3/SNComputeLowering.cpp:60`.
///
/// ⭐ THE REFERENCE PRINTS THE ANSWER IN ITS OWN COMMENT (`:88-90`):
/// `#set = affine_set<(d0)[s0] : (d0 + s0 * 8 - 64 >= 0, -d0 + 63 >= 0)>` for
/// `"loop_ds2_ds3_mb -> s0" : {"mb" : 1}` — 64 lanes, `64 / 8 = 8` as the coefficient, offset 1.
///
/// ⛔⛔ IT IS THE **SET** FORM OF THE OP, NOT THE PREFIX FORM. A live lane count cannot state this
/// mask: the first masked lane is `s0 * 8` and is not known until the loop runs, so the op carries
/// its `mask_set` and a `mask_parameter` ([`vectorchain::Op::CreateAffineMaskSet`]) where the static
/// entries carry a [`vectorchain::LaneMask`].
///
/// ⛔⛔ AND THE PARENTHESISATION IS THE OTHER WAY ROUND FROM THE STATIC FORM.
/// `symbol * (vector_dim / 8) * mask_offset` (`:95`) divides the lane count FIRST, where
/// `k * vector_dim / 8` (`:48`) divides the product; the two disagree for any lane count 8 does not
/// divide, so the static and dynamic bounds are not one expression. ⭐ The two multiplies fold into
/// one coefficient, `(vector_dim / 8) * mask_offset`, which is what the island's [`AffineExpr`]
/// carries and what bridge 2 reads back.
///
/// ⛔ `None` IS "NO EMITTED LOOP WALKS THAT DIMENSION" — entry 028's answer, and the reference's own
/// `nullptr` from `getMLIRLoopFromLoopNode` (`:79`). ⭐ IT ALSO ABSORBS THE UNINITIALISED `iv`: the
/// reference declares `mlir::Value iv;` and leaves it null when the loop is neither an
/// `affine.for` nor an `scf.for` (`:80-86`), then passes it as the op's operand. Entry 028 hands
/// back the induction variable itself, so there is no third case to leave empty.
pub fn dynamic_masking(
    vals: &mut Values,
    into: &mut Vec<Op>,
    masked: Vector,
    dims: &[PrimaryDim],
    mlir_loops: &[Val],
    mask: DynamicMask,
) -> Option<Predicate> {
    let iv = mlir_loop_from_loop_node(dims, mlir_loops, mask.dim)?;
    let lanes = i64::try_from(masked.len).ok()?;
    let mask_set = IntegerSet {
        dims: 1,
        symbols: 1,
        constraints: vec![
            // `id - vector_dim + symbol * (vector_dim / 8) * mask_offset`
            Constraint {
                expr: AffineExpr::dim(0)
                    .plus(AffineExpr::sym(0).times((lanes / 8) * mask.offset))
                    .plus(AffineExpr::Const(-lanes)),
                is_equality: false,
            },
            // `-id + vector_dim - 1`
            Constraint {
                expr: AffineExpr::dim(0)
                    .times(-1)
                    .plus(AffineExpr::Const(lanes - 1)),
                is_equality: false,
            },
        ],
    };
    let op = vectorchain::Op::CreateAffineMaskSet {
        result: vals.mint(),
        mask_set,
        mask_parameter: Some(iv),
        ty: Vector {
            len: masked.len,
            elem: ElemType::Int(1),
        },
    };
    // ⭐ THE PREDICATE COMES OFF THE OP so the width at the use is the width the definition printed.
    let predicate = op.binds_predicate();
    into.push(Op::VectorChain(op));
    predicate
}

/// Replaces: e049_getTypeBasedOnComputeType
///
/// THE VECTOR A COMPUTE NODE'S OWN FORMAT RUNS AT — `V3/SNComputeLowering.cpp:108`.
///
/// ⭐ ONE STICK OF A REGULAR TENSOR ON THIS LOWERING'S COMPONENT: the whole function is
/// `constructTypeFromFormat(builder, original_format, REGULAR_TENSOR, comp_, -1, ..)` (`:113-115`),
/// and `-1` is [`VectorWidth::OneStick`] — *"however many of this format fill one stick"*.
///
/// ⛔⛔ IT THROWS AWAY THE THIRD OUT-PARAMETER. `bool is_result_integer;` is declared uninitialised
/// (`:111`), filled by the callee, and never read — so a caller of this wrapper cannot learn what
/// [`is_integer`] answers, even though its own caller two frames up does
/// (`:380`, where the same call keeps it). Returning the [`Vector`] carries both `result_type` and
/// `element_type`, and [`is_integer`] recovers the discarded flag from the element.
///
/// ⛔ `None` IS THE ABORT INSIDE, NOT A NEW REFUSAL: `constructTypeFromFormat` aborts when the
/// element width does not divide a stick ([`StickWidth::of`]), and this function's
/// `return LogicalResult::failure()` (`:116`) is the path that propagates it.
#[must_use]
pub const fn type_from_compute_type(format: DataType, on: GenericComp) -> Option<Vector> {
    let Some(width) = StickWidth::of(format, on, TensorCategory::Regular) else {
        return None;
    };
    Some(type_from_format(
        format,
        on,
        TensorCategory::Regular,
        VectorWidth::OneStick(width),
    ))
}
/// A MAC THE COMPUTE PATH ACCEPTS — the seven `ComputeOpType`s `constructComputeOperation` routes to
/// `constructMACOperation` (`SNComputeLowering.cpp:1570-1573`).
///
/// ⛔ NOT THE GENERATED [`ComputeType`](crate::generated::ComputeType), whose `.ddl` census spells
/// `MACC` and never `FMA8`/`FMA4`/`IMA8`/`IMA4`: the two map-selecting chains below cannot state
/// their own input from it, and the `else` both end in is an `emitError` this crate may not be.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MacOp {
    /// `FMA16`.
    Fma16,
    /// `FNMS` — the negated-multiply form, which shares FMA16's reduction (`:131`).
    Fnms,
    /// `FMA32`.
    Fma32,
    /// `FMA8`.
    Fma8,
    /// `FMA4` — ⛔ SEN1P5 ONLY (`DT_CHECK(arch == SEN1P5_ISA)`, `:152`).
    Fma4,
    /// `IMA8`.
    Ima8,
    /// `IMA4`.
    Ima4,
}

impl MacOp {
    /// Replaces: e050_getReductionMapForMACOperation
    ///
    /// WHICH LANES ONE MAC ACCUMULATES TOGETHER — the `reduction_map` a
    /// `vectorchain.multiply_and_accumulate` carries (`SNComputeLowering.cpp:126`).
    ///
    /// ⛔⛔ IMA8 AND IMA4 OFF SEN1P5 ARE NOT A PLAIN FLOORDIV: they wrap at 128 lanes,
    /// `((d0 mod 128) floordiv k)`, and the reference's own examples say why — `0, 1, 128, 129 -> 0`
    /// (`:165`, `:182`). A bare floordiv would accumulate lane 128 into group 64.
    ///
    /// ⭐ THE GENERATION IS `A::GEN`, so which factor this is, is a COMPILE-TIME fact.
    #[must_use]
    pub fn reduction_map<A: Arch>(self) -> AffineMap {
        let sen1p5 = matches!(A::GEN, IsaGen::Sen1p5);
        let i = || AffineExpr::dim(0);
        AffineMap::unary(match self {
            // `is_any_of(type, FMA16, FNMS)` — one arm for the two (`:131`).
            MacOp::Fma16 | MacOp::Fnms => i().floordiv(if sen1p5 { 4 } else { 1 }),
            // ⭐ `floordiv 1` ON BOTH GENERATIONS, which the reference writes out rather than
            // returning the identity (`:139-142`) — and the printed map says `d0 floordiv 1`.
            MacOp::Fma32 => i().floordiv(1),
            MacOp::Fma8 => i().floordiv(if sen1p5 { 16 } else { 2 }),
            // ⛔ ONE FACTOR, NOT TWO: the `DT_CHECK` says the older generation has no FMA4 at all,
            // so there is no second value for an arm to choose between.
            MacOp::Fma4 => i().floordiv(32),
            MacOp::Ima8 if sen1p5 => i().floordiv(16),
            MacOp::Ima8 => i().modulo(128).floordiv(2),
            MacOp::Ima4 if sen1p5 => i().floordiv(32),
            MacOp::Ima4 => i().modulo(128).floordiv(4),
        })
    }

    /// `stringifyComputePrecision(type_)` (`DSC2ToDataflowIR.hpp:54-71`), prefixed `mx` where any
    /// operand's labelled data structure is a scaled tensor (`SNComputeLowering.cpp:1579-1591`).
    ///
    /// ⛔ THREE OF THE SEVEN `mx` PRODUCTS HAVE NO SPELLING ON THE MACHINE. The reference builds them
    /// by string concatenation; `SentientTypes.td:56-58` carries `mxfp4`, `mxfp8` and `mxint4` and no
    /// `mxint8`/`mxfp16`/`mxfp32`, and `ConstructProgIRHelper.cpp:1469-1488` refuses what it is handed.
    #[must_use]
    pub fn precision(self, category: TensorCategory) -> dataflow::Precision {
        match (self, category) {
            (MacOp::Ima8, TensorCategory::Regular) => dataflow::Precision::Int8,
            (MacOp::Ima4, TensorCategory::Regular) => dataflow::Precision::Int4,
            (MacOp::Fma4, TensorCategory::Regular) => dataflow::Precision::Fp4,
            (MacOp::Fma8, TensorCategory::Regular) => dataflow::Precision::Fp8,
            (MacOp::Fma16, TensorCategory::Regular) => dataflow::Precision::Fp16,
            // ⭐ `FNMS` IS `fp32` TOO (`:67-68`), which is the one arm that is not a name match.
            (MacOp::Fma32 | MacOp::Fnms, TensorCategory::Regular) => dataflow::Precision::Fp32,

            (MacOp::Ima4, TensorCategory::Scaled) => dataflow::Precision::Mxint4,
            (MacOp::Fma4, TensorCategory::Scaled) => dataflow::Precision::Mxfp4,
            (MacOp::Fma8, TensorCategory::Scaled) => dataflow::Precision::Mxfp8,
            (MacOp::Ima8, TensorCategory::Scaled) => todo!("precision \"mxint8\" is not a spelling"),
            (MacOp::Fma16, TensorCategory::Scaled) => todo!("precision \"mxfp16\" is not a spelling"),
            (MacOp::Fma32 | MacOp::Fnms, TensorCategory::Scaled) => {
                todo!("precision \"mxfp32\" is not a spelling")
            }
        }
    }

    /// Replaces: e051_getSelectionMapForMACOperandFromL0
    ///
    /// HOW L0LU DATA IS SPLATTED ACROSS THE MAC'S LANES — the `(d0) -> (d0 mod factor)` a
    /// `vectorchain.select` carries, and the vector the splat is stated over
    /// (`SNComputeLowering.cpp:200`).
    ///
    /// ⛔⛔ THE SELECTION FACTOR IS NOT THE REDUCTION FACTOR off SEN1P5: IMA8 selects `mod 4` while
    /// reducing `floordiv 2`, and IMA4 selects `mod 8` against `floordiv 4` (`:246`, `:255` versus
    /// `:165`, `:182`) — the two chains agree on SEN1P5 and on nothing else.
    ///
    /// ⛔ [`None`] IS REACHABLE: `constructComputeInputOperandAndAddToList` passes `compute_op.type_`
    /// straight in for every `PTWEST`/`L0LU`/`L0LUROW0` operand (`:504`), so FMA32 and FNMS — which
    /// have no arm here — reach the `emitError` at `:270`.
    #[must_use]
    pub fn selection_map_from_l0<A: Arch>(
        self,
        format: DataType,
        original: ElemType,
    ) -> Option<(AffineMap, Vector)> {
        let sen1p5 = matches!(A::GEN, IsaGen::Sen1p5);
        // `mlir::isa<VectorType>(original_data_type)` is false exactly for the MX custom vector,
        // whose element is a `CustomMXFloatType` — see [`type_from_format`].
        let mx = matches!(original, ElemType::MxFloat(_));
        let (factor, elem): (u32, ElemType) = match self {
            MacOp::Fma16 => (
                if sen1p5 { 4 } else { 1 },
                if matches!(format, DataType::Bfloat16) {
                    ElemType::Bf16
                } else {
                    ElemType::F16
                },
            ),
            MacOp::Fma8 => (
                if sen1p5 { 16 } else { 2 },
                if mx {
                    ElemType::MxFloat(8)
                } else {
                    ElemType::F8E4M3Fn
                },
            ),
            MacOp::Fma4 => (
                32,
                if mx {
                    ElemType::MxFloat(4)
                } else {
                    ElemType::F4E2M1Fn
                },
            ),
            // ⛔ AN INTEGER MAC IGNORES `original`: `builder.getIntegerType(8)` unconditionally
            // (`:249`, `:258`), so there is no MX spelling of an IMA operand.
            MacOp::Ima8 => (if sen1p5 { 16 } else { 4 }, ElemType::Int(8)),
            MacOp::Ima4 => (if sen1p5 { 32 } else { 8 }, ElemType::Int(4)),
            MacOp::Fma32 | MacOp::Fnms => return None,
        };
        Some((
            AffineMap::unary(AffineExpr::dim(0).modulo(i64::from(factor))),
            Vector {
                len: 64 * u64::from(factor),
                elem,
            },
        ))
    }
}

/// Replaces: e052_constructPrecisionConversionOperation
///
/// THE SOURCE VALUE AT THE RESULT FORMAT'S PRECISION — `SNComputeLowering.cpp:375`.
///
/// ⛔⛔ AN MX SOURCE IS RETURNED UNCONVERTED. That arm reports success with `result` never assigned
/// (`:389-391`), and all four call sites initialise `result` to the SOURCE — so it means the input
/// value, not an empty one: *"MX format is a logical format .. no precision conversion on those
/// units."*
///
/// ⛔ [`None`] IS THE INTEGER-TO-INTEGER ARM, the one `failure()` the chain falls through to
/// (`:429`): `vectorchain.cast` is float-to-float and there is no integer-width conversion to emit.
pub fn precision_conversion(
    vals: &mut Values,
    into: &mut Vec<Op>,
    src: Computed,
    result_format: DataType,
    on: GenericComp,
) -> Option<Computed> {
    if matches!(src.ty().elem, ElemType::MxFloat(_)) {
        return Some(src);
    }
    // ⛔ THE SOURCE'S OWN ELEMENT COUNT, not a stick's: `num_elements` is read off `src.getType()`
    // (`:395`, `:400`), so the `-1` sentinel cannot arise here and neither can its abort.
    let ty = type_from_format(
        result_format,
        on,
        TensorCategory::Regular,
        VectorWidth::Given(Elements(src.ty().len)),
    );
    if src.ty() == ty {
        return Some(src);
    }
    let result = vals.mint();
    let input_ty = src.ty();
    into.push(match (is_integer(input_ty.elem), is_integer(ty.elem)) {
        (true, false) => Op::Arith(arith::Op::Convert {
            result,
            kind: arith::ConvertKind::SiToFp,
            input: src.val(),
            input_ty,
            ty,
        }),
        (false, true) => Op::Arith(arith::Op::Convert {
            result,
            kind: arith::ConvertKind::FpToSi,
            input: src.val(),
            input_ty,
            ty,
        }),
        (false, false) => Op::VectorChain(vectorchain::Op::Cast {
            result,
            input: src.val(),
            input_ty,
            ty,
        }),
        (true, true) => return None,
    });
    Some(Computed::of(result, ty))
}

/// Replaces: e053_constructOpaqueOperation
///
/// A WHOLE `.smc` BODY AS ONE OP — `SNComputeLowering.cpp:1534`.
///
/// ⛔ THE FIRST `StringAttr` IS THE `dbgName`, NOT THE FUNCTION: `compute_op.name_` goes into
/// `OpaqueOp::create`'s `dbgName` slot and `computeTypeToString(type_)` into `func_name` (`:1552`),
/// which is [`OpaqueFunc`] here.
///
/// ⭐ NO SORT OF THE THREE DICTIONARIES, AND THAT IS DELIBERATE: `mapToDicAttr` walks a
/// `std::map<std::string, std::string>` in key order (entry 016), and this island's printer already
/// emits every dictionary in key-spelling order — sorting again would be one fact stated twice.
#[must_use]
pub fn opaque_operation(
    name: &str,
    func: OpaqueFunc,
    read_write: &[(RegName, RegAddr)],
    read_only: &[(RegName, RegAddr)],
    params: &[(ParamKey, ParamValue)],
) -> Op {
    Op::Dataflow(dataflow::Op::Opaque(Opaque {
        func,
        read_write: read_write.to_vec(),
        read_only: read_only.to_vec(),
        params: params.to_vec(),
        dbg_name: Some(name.to_owned()),
    }))
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 086 + 087/110 — ONE COMPUTE OPERAND, IN AND OUT
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// WHICH REGISTER FILE AN OPERAND IS READ FROM OR WRITTEN TO, AND WHAT THAT ACCESS THEN COSTS.
///
/// ⭐ THE TEN REGISTER-FILE `SenComponents` COLLAPSE TO FOUR SHAPES. `LRFREG`, `PTARF`, `PTXRF`,
/// `PELRF` and `SFPLRF` share one body (`SNComputeLowering.cpp:519`, `:804`); the two `STATE` files
/// differ only by suppressing the precision conversion (`:757`, `:782`); and the two forwards and
/// the scale register each add a shuffle. [`Component`] still says WHICH file it is — this says what
/// reading it costs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryFile {
    /// `LRFREG` | `PTARF` | `PTXRF` | `PELRF` | `SFPLRF` — the access IS the operand.
    Register,
    /// `PESTATE` | `SFPSTATE` — the same access, and the one flag that skips the conversion.
    State,
    /// `NFWD0` | `NFWD2` — a neighbour forward, read through a fixed index table.
    ///
    /// ⛔⛔ AND IT DOES NOT ADDRESS ITS OWN OPERAND'S STORAGE. `storage` and `data_info` are
    /// REASSIGNED to `outputs_.front()` and `outputsLdsAndLoopOffsets_.front()` (`:530-533`), and the
    /// unit view likewise to `outputsLoopsAndSizes_.front()` (`:547`, `:565`) — a forward reads where
    /// the compute's FIRST OUTPUT is written. That substitution picks a DSC record and so belongs to
    /// the caller; what is left here is the shuffle.
    Forward(Forward),
    /// `LXLUSCALEREG` — one scale register broadcast over half a vector, indexed by a loop counter.
    ScaleReg,
}

/// WHICH NEIGHBOUR FORWARD, and so which index table the load is read through.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Forward {
    /// `NFWD0`.
    Nfwd0,
    /// `NFWD2`.
    Nfwd2,
}

impl Forward {
    /// THE TABLE, PER ELEMENT WIDTH (`:591-628`).
    ///
    /// ⛔ [`None`] IS A WIDTH WITH NO ARM, which in the reference is a NULL `input_data` carried
    /// into `constructPrecisionConversionOperation`: neither `if` has an `else`. The refusal moves to
    /// the one line that can still name the width.
    ///
    /// ⭐ `repetition` IS 8 ON ALL FOUR ARMS, so `8 x 8` fills an f16 stick and `4 x 8` an f32 one.
    #[must_use]
    pub fn indices(self, elem: ElemType) -> Option<Vec<i32>> {
        match (self, elem) {
            (Forward::Nfwd0, ElemType::F16) => Some(vec![2, 3, 0, 1, 6, 7, 6, 7]),
            (Forward::Nfwd0, ElemType::F32) => Some(vec![1, 0, 3, 3]),
            (Forward::Nfwd2, ElemType::F16) => Some(vec![4, 5, 3, 3, 5, 5, 6, 7]),
            (Forward::Nfwd2, ElemType::F32) => Some(vec![2, 0, 1, 3]),
            _ => None,
        }
    }
}

/// ONE ENTRY OF `loopEleOffsets_.at(corelet)` — a loop, and how far this operand walks along it.
///
/// ⛔ `DT_CHECK(dim_offset.size() == 1)` (`:641`, `:727`) IS THIS SHAPE: one dim and one offset per
/// loop, so the reference's `*dim_offset.begin()` reads the only entry there is, and the induction
/// variable is [`mlir_loop_from_loop_node`]'s answer for that dim.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoopOffset {
    /// The loop's induction variable.
    pub iv: Val,
    /// `dim_offset.begin()->second`, in elements.
    pub offset: i64,
}

/// A REGISTER-FILE OPERAND'S ACCESS, AS ONE RECORD — the same fields on both sides of a compute.
pub struct MemoryOperand<'m> {
    /// Which file, and what the access then costs.
    pub file: MemoryFile,
    /// `storage` — what the view is taken over.
    pub storage: Component,
    /// `{comp_, storage}`, as the granularity table keys it.
    pub location: DataLocation,
    /// The format the granularity factor is read at — `getElementType(result_type)`.
    pub precision: DataType,
    /// `needsUniform()` — the handle set the address maps over, or [`None`] for the constant one.
    pub uniform: Option<&'m Handles>,
    /// `data_info->startAddr_`, read the two ways entry 030 reads it.
    pub addresses: StartAddresses<'m>,
    /// `*unit_view_sizes` — `sizesNoGaps_` under uniformization and `getSizesForCoreId(core_id_)`
    /// otherwise, which is a choice of DSC record and so the caller's.
    pub view_sizes: &'m [ViewSize],
    /// `unit_view->outerLoops_`.
    pub outer_loops: &'m [LoopStride],
    /// `myLdsIdx_ == -1 && constantId_ >= 0`.
    pub addressed_as: AddressedAs,
    /// `core_id_`.
    pub core: Core,
    /// `corelet_id_` — [`None`] is the reference's `-1`.
    pub corelet: Option<Corelet>,
    /// `loopEleOffsets_.at(clId)`, which only [`MemoryFile::ScaleReg`] reads.
    pub offsets: &'m [LoopOffset],
}

/// THE RING PEER — another core's copy of THIS unit, and the wire to or from it.
///
/// ⛔ `DT_CHECK(comp_ == SFP)` (`:702`, `:880`) IS THE TWO STATIC KINDS OF THAT WIRE. Only an SFP
/// reaches this arm and the peer is the same `comp_` in a different core, so [`link::Sfp`] at both
/// ends of the [`Link`] IS the check — and the ring is the one arm of either function whose unit
/// kind is known here rather than at the caller.
pub struct Ring<'r> {
    /// This unit's own handle — the end of the wire that stays here.
    pub own: Val,
    /// `needsUniform()`.
    pub uniform: Option<&'r Handles>,
    /// `startAddr_` READ AS A CORE ID, which is what this one manager holds (`:707`, `:884`).
    pub peer_core: &'r dyn Fn(Core, Corelet) -> Core,
    /// `core_id_`, which the non-uniform arm reads the peer at.
    pub core: Core,
    /// `corelet_id_` — the ring never crosses corelets.
    pub corelet: Corelet,
    /// `num_folds_`.
    pub num_folds: NumFolds,
    /// `uniform_region_iterator_`, for the query entry 059 makes.
    pub region_iterator: Option<Val>,
}

/// WHERE ONE COMPUTE INPUT OPERAND COMES FROM — `compute_op.inputs_[index]`, with each arm's facts.
///
/// ⛔⛔ THE FIVE PLAIN RECEIVES ARE ONE ARM. `PTNORTH`, `PT`, `LXLU`, `PE` and `SFP` differ in the
/// reference ONLY in the component `retrieveGetUnitOpInSameCore` is asked for (`:511-4973`), and `PT`
/// asks for `PTROW7` rather than for itself. The unit that answers is a NEIGHBOUR of the one being
/// lowered — `PTNORTH` of row 0 is the SFP and of row 3 is row 2, per [`PtDirection`] — so the kind
/// is not a property of the arm at all and cannot be spelled here. That is why the wire ARRIVES as a
/// [`RecvEnd`]: [`Link::ends`] is the only thing that mints one, and the send it pairs with belongs
/// to the producing unit's own lowering, exactly as entries 080, 081 and 090 take theirs.
pub enum ComputeInput<'i> {
    /// `ZERO` — a dense zero of the result's own type.
    Zero,
    /// `ONE`.
    One,
    /// `PTWEST` | `L0LU` | `L0LUROW0` — a receive off the west neighbour, splatted for the MAC.
    FromL0 {
        /// The wire from the west unit.
        west: RecvEnd,
        /// `compute_op.type_`.
        mac: MacOp,
        /// `compute_op.dataFormat_`.
        format: DataType,
    },
    /// `PTNORTH` | `PT` | `LXLU` | `PE` | `SFP` — one receive, and the operand is what arrives.
    Wire(RecvEnd),
    /// The ten-component register-file group.
    Memory(MemoryOperand<'i>),
    /// `SFPRING`.
    Ring(Ring<'i>),
    /// `LATCH` — `getFromLatchMap(latch_id)`, which is the lowering's own state and so the caller's.
    Latch {
        /// The value the map holds.
        data: Val,
        /// `loopEleOffsets_.at(clId)`, read only when the execution unit is the LXLU.
        offsets: &'i [LoopOffset],
    },
}

/// WHAT BOTH SIDES OF A COMPUTE OPERAND ARE LOWERED AGAINST.
#[derive(Clone, Copy)]
pub struct OperandContext<'c> {
    /// `compute_op.name_` — the `dbgName` of every op here that carries one.
    pub name: &'c str,
    /// `comp_` — the unit being lowered, which the conversion is chosen for.
    pub comp: GenericComp,
    /// `compute_op.exUnit_`, whose being the LXLU suppresses every precision conversion.
    pub ex_unit: GenericComp,
    /// `component_to_handler_`, for the view a register-file access is taken over.
    pub handlers: &'c Handlers,
    /// `result_type`.
    ///
    /// ⛔ AND `element_type` IS ITS ELEMENT, NOT A SECOND ARGUMENT. The reference passes both
    /// (`:456-457`); a `CustomVectorType` of `CustomMXFloatType` is [`ElemType::MxFloat`] here, so
    /// the `isa` test those two arguments exist for is a test on this one field.
    pub result_ty: Vector,
}

/// Replaces: e086_constructComputeInputOperandAndAddToList
///
/// ONE COMPUTE INPUT OPERAND, AT THE COMPUTE'S OWN PRECISION — `SNComputeLowering.cpp:456`.
///
/// ⛔⛔ THE TAIL IS NOT A FORMALITY: every arm above it produces a value at the SOURCE's width, and
/// `constructPrecisionConversionOperation` is what makes it an operand of THIS compute (`:757-771`).
/// Its two exemptions are the state files and an LXLU execution unit.
///
/// ⛔ [`None`] IS EVERY `failure()` THE REFERENCE HAS: the selection map's (`:502`), the affine
/// transfer's (`:585`), the shuffle table's, the `DT_CHECK` on the innermost loop, the conversion's
/// (`:766`) and the unmatched component (`:754`).
/// ⛔ AND ITS `DT_ERROR("Result type is expected to be a vector type")` IS UNREACHABLE — a
/// [`Vector`] is one, and the reference's third case is a type this island cannot spell.
pub fn compute_input_operand<A: Arch>(
    vals: &mut Values,
    into: &mut Vec<Op>,
    ctx: &OperandContext<'_>,
    source: &ComputeInput<'_>,
    format: DataType,
) -> Option<Computed> {
    let ty = ctx.result_ty;
    let input_data = match source {
        // `getZeroAttr(result_vtype)` and `getIntegerAttr/getFloatAttr(element_type, 1)`
        // (`:461-494`) — one body, and the literal is which of the two pseudo-units it is.
        ComputeInput::Zero | ComputeInput::One => {
            let splat = i64::from(matches!(source, ComputeInput::One));
            if matches!(ty.elem, ElemType::MxFloat(_)) {
                single_val_custom_vector(vals, into, ctx.name, splat, ty)
            } else {
                let result = vals.mint();
                into.push(Op::Arith(arith::Op::DenseConstant { result, splat, ty }));
                Computed::of(result, ty)
            }
        }
        // `ReceiveOp::create(..)` then `SelectOp::create(.., splat_data_type, data, selection_map)`.
        ComputeInput::FromL0 { west, mac, format } => {
            let data = Received::receive(into, vals.mint(), *west, ty);
            let (selection_map, splat_ty) = mac.selection_map_from_l0::<A>(*format, ty.elem)?;
            let result = vals.mint();
            into.push(Op::VectorChain(vectorchain::Op::Select {
                result,
                input: data.operand(),
                selection_map,
                input_ty: ty,
                ty: splat_ty,
            }));
            Computed::of(result, splat_ty)
        }
        ComputeInput::Wire(from) => {
            Computed::of(Received::receive(into, vals.mint(), *from, ty).operand(), ty)
        }
        ComputeInput::Memory(memory) => {
            let loaded = memory_load(vals, into, ctx, memory)?;
            match memory.file {
                // `input_data = load_op.getResult()`.
                MemoryFile::Register | MemoryFile::State => loaded,
                // `ShuffleOp::create(.., result_type, load_op.getResult(), nullptr, index_array, 8,
                //  name)` — the result type is the LOAD's, so the table only reorders (`:591-628`).
                MemoryFile::Forward(forward) => {
                    let result = vals.mint();
                    into.push(Op::VectorChain(vectorchain::Op::Shuffle {
                        pad: Vec::new(),
                        result,
                        input: loaded.val(),
                        variable: Vec::new(),
                        dbg_name: Some(ctx.name.to_owned()),
                        indices: forward.indices(ty.elem)?,
                        repetition: 8,
                        input_ty: ty,
                        ty,
                    }));
                    Computed::of(result, ty)
                }
                // ⛔ THE INNERMOST LOOP IS THE FIRST WITH A NON-ZERO OFFSET (`:638-648`), and
                // `DT_CHECK(loop_offset_it != loop_offset.end())` is this [`None`].
                MemoryFile::ScaleReg => {
                    let offset = memory.offsets.iter().find(|entry| entry.offset != 0)?;
                    broadcast_over_lanes(vals, into, ctx, loaded.val(), offset.iv, 2)
                }
            }
        }
        ComputeInput::Ring(ring) => {
            let peer = ring_peer(vals, into, ring)?;
            let (_, from) = Link::<link::Sfp, link::Sfp>::between(peer, ring.own).ends();
            Computed::of(Received::receive(into, vals.mint(), from, ty).operand(), ty)
        }
        // `if (compute_op.exUnit_ == LXLU)` — the latched vector is broadcast at QUARTER width, and
        // the ONE loop offset is taken with no search at all (`:723-745`).
        ComputeInput::Latch { data, offsets } => {
            if matches!(ctx.ex_unit, GenericComp::Lxlu) {
                broadcast_over_lanes(vals, into, ctx, *data, offsets.first()?.iv, 4)
            } else {
                Computed::of(*data, ty)
            }
        }
    };

    // `if (!is_any_of(inputs_[index], PESTATE, SFPSTATE) && compute_op.exUnit_ != LXLU)`.
    if skips_conversion(ctx, input_file(source)) {
        return Some(input_data);
    }
    precision_conversion(vals, into, input_data, format, ctx.comp)
}

/// THE `agen.vector_load` A REGISTER-FILE INPUT IS READ WITH, address and all (`:537-590`).
fn memory_load(
    vals: &mut Values,
    into: &mut Vec<Op>,
    ctx: &OperandContext<'_>,
    memory: &MemoryOperand<'_>,
) -> Option<Computed> {
    let elements = memory_view(vals, into, ctx, memory, TransferSide::Load)?;
    let result = vals.mint();
    into.push(Op::Agen(agen::Op::VectorLoad {
        result,
        view: elements.view.result,
        indices: elements.base_address.indices.clone(),
        dbg_name: Some(ctx.name.to_owned()),
        access: agen::Access::Stated(elements.transfer_set),
        view_ty: elements.view.ty.clone(),
        ty: ctx.result_ty,
    }));
    Some(Computed::of(result, ctx.result_ty))
}

/// THE VIEW, THE ADDRESS AND THE ELEMENT SET BOTH SIDES OF A REGISTER-FILE ACCESS SHARE.
///
/// ⭐ UNIFORMIZATION CHANGES ONLY THE ADDRESS — a mapping per handle against one `arith.constant`,
/// exactly the two arms entry 030 offers (`:541-568`, `:821-844`). Everything below it is identical
/// on the load and the store, which is why one function serves both.
fn memory_view(
    vals: &mut Values,
    into: &mut Vec<Op>,
    ctx: &OperandContext<'_>,
    memory: &MemoryOperand<'_>,
    side: TransferSide,
) -> Option<TransferElements> {
    let factor = address_granularity_multiply_factor(memory.location, memory.precision);
    let start_address = match memory.uniform {
        Some(handles) => uniformized_folded_address(
            vals,
            into,
            handles,
            memory.addresses.all,
            memory.addresses.at,
            factor,
        ),
        None => constant_index(vals, into, factor.scale(memory.addresses.single)),
    };
    elements_of_affine_data_transfer_via_agen_transfer(
        vals,
        into,
        ctx.handlers,
        &AgenStorage {
            storage: memory.storage,
            core: memory.core,
            corelet: memory.corelet,
            side,
            start_address,
            view_sizes: memory.view_sizes,
            elem: ctx.result_ty.elem,
            outer_loops: memory.outer_loops,
            addressed_as: memory.addressed_as,
        },
    )
}

/// ONE REGISTER'S VALUE OVER A NARROWED VECTOR — `indices = {-1}` against a loop counter, at
/// `num_elements / divisor` lanes (`:649-668`, `:735-745`).
///
/// ⛔⛔ `{-1}` READS THE `variable` OPERAND AND NOT THE INPUT: every element of the result comes from
/// the counter, and the input only says how wide the thing being indexed is — see
/// [`vectorchain::Op::Shuffle::variable`], which entry 086 is why the field exists.
fn broadcast_over_lanes(
    vals: &mut Values,
    into: &mut Vec<Op>,
    ctx: &OperandContext<'_>,
    input: Val,
    iv: Val,
    divisor: u64,
) -> Computed {
    // ⭐ THE REDUCED WIDTH IS TAKEN OFF `result_type` ON BOTH ARMS, including the latch one, whose
    // input is a map entry whose own type the reference never names either.
    let reduced = Vector {
        len: ctx.result_ty.len / divisor,
        elem: ctx.result_ty.elem,
    };
    let result = vals.mint();
    into.push(Op::VectorChain(vectorchain::Op::Shuffle {
        pad: Vec::new(),
        result,
        input,
        variable: vec![vectorchain::ShuffleVariable {
            val: iv,
            ty: ScalarTy::Index,
        }],
        dbg_name: Some(ctx.name.to_owned()),
        indices: vec![-1],
        repetition: u32::try_from(reduced.len).expect("a vector's element count fits a u32"),
        input_ty: ctx.result_ty,
        ty: reduced,
    }));
    Computed::of(result, reduced)
}

/// THE RING PEER'S HANDLE — a mapping per handle under uniformization, one `get_unit` otherwise.
///
/// ⛔ THE UNIT IS `comp_`, WHICH THIS ARM HAS ALREADY PROVED IS THE SFP — see [`Ring`].
fn ring_peer(vals: &mut Values, into: &mut Vec<Op>, ring: &Ring<'_>) -> Option<Val> {
    match ring.uniform {
        Some(handles) => Some(uniformized_folded_destination_core(
            vals,
            into,
            handles,
            DfirUnit::Sfp,
            ring.num_folds,
            ring.region_iterator,
            ring.peer_core,
        )),
        None => {
            let op = create_get_unit_op_in_different_core(
                vals,
                DfirUnit::Sfp,
                (ring.peer_core)(ring.core, ring.corelet),
                ring.corelet,
                ring.num_folds,
            );
            let dataflow::Op::GetUnit { result: peer, .. } = &op else {
                // Entry 023 builds nothing else, so this ring names no peer.
                return None;
            };
            let peer = *peer;
            into.push(Op::Dataflow(op));
            Some(peer)
        }
    }
}

/// `is_any_of(component, PESTATE, SFPSTATE) || exUnit_ == LXLU` — the two exemptions BOTH sides of a
/// compute operand share (`:757`, `:782`).
const fn skips_conversion(ctx: &OperandContext<'_>, file: Option<MemoryFile>) -> bool {
    matches!(file, Some(MemoryFile::State)) || matches!(ctx.ex_unit, GenericComp::Lxlu)
}

/// Which register file an input is, for that exemption.
const fn input_file(source: &ComputeInput<'_>) -> Option<MemoryFile> {
    match source {
        ComputeInput::Memory(memory) => Some(memory.file),
        _ => None,
    }
}

/// WHERE ONE COMPUTE OUTPUT OPERAND GOES — `compute_op.outputs_[i]`.
///
/// ⛔ THE FIVE SENDS ARE ONE ARM for the reason [`ComputeInput::Wire`] records: `PTSOUTH`, `LXSU`,
/// `SFP`, `PE` and `PT` differ only in the component asked for (`:5071-5180`).
pub enum ComputeOutput<'o> {
    /// `PTSOUTH` | `LXSU` | `SFP` | `PE` | `PT`.
    Wire(SendEnd),
    /// `LRFREG` | `PTXRF` | `PTARF` | `PELRF` | `SFPLRF` | `SFPSTATE` | `PESTATE`.
    ///
    /// ⭐ AND NO FORWARD OR SCALE REGISTER: nothing is ever WRITTEN to those, which is why
    /// [`MemoryFile`]'s other two arms cannot arise here.
    Memory(MemoryOperand<'o>),
    /// `SFPRING`.
    Ring(Ring<'o>),
    /// `LATCH` — `addToLatchMap(latch_id, compute_result)` emits NOTHING.
    Latch(Latch),
}

/// WHAT ONE OUTPUT OPERAND LEFT BEHIND.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stored {
    /// The ops are in the caller's list.
    Emitted,
    /// ⛔ THE LATCH MAP IS THE LOWERING'S STATE, so the entry is handed back rather than written
    /// here — and what it records is the value AFTER the conversion.
    Latched {
        /// `latchDataId_`.
        latch: Latch,
        /// `compute_result`.
        data: Val,
    },
}

/// THE FORMAT AN OUTPUT IS CONVERTED TO — the labelled data structure's, or the compute's own.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OutputFormat {
    /// `labeledDs_[myLdsIdx_].dataFormat_`, and [`None`] is the reference's `lds_idx == -1`.
    pub lds: Option<DataType>,
    /// `compute_op.getComputeOperandFormats(*dsc_)[i]`.
    pub operand: DataType,
}

impl OutputFormat {
    /// `if (lds_idx != -1) format = labeledDs_[..].dataFormat_ else getComputeOperandFormats(..)[i]`.
    #[must_use]
    pub const fn get(self) -> DataType {
        match self.lds {
            Some(format) => format,
            None => self.operand,
        }
    }
}

/// Replaces: e087_constructComputeOutputOperand
///
/// ONE COMPUTE OUTPUT OPERAND, AT ITS DESTINATION'S PRECISION — `SNComputeLowering.cpp:777`.
///
/// ⛔⛔ THE CONVERSION IS AT THE **HEAD** HERE, NOT THE TAIL. A result is converted once and every
/// arm below spends `compute_result` (`:780-796`), where entry 086 converts last — so the two
/// functions are not mirror images and a shared helper would have to pick one. The exemptions are
/// the same pair.
///
/// ⛔ AND THE FORMAT IS THE LABELLED DATA STRUCTURE'S WHERE THERE IS ONE (`:785-790`) — see
/// [`OutputFormat`].
/// ⛔ [`None`] IS THE THREE `failure()`s: the conversion's, the affine transfer's and the unmatched
/// component's (`:794`, `:857`, `:908`).
pub fn compute_output_operand(
    vals: &mut Values,
    into: &mut Vec<Op>,
    ctx: &OperandContext<'_>,
    output: &ComputeOutput<'_>,
    result: Computed,
    format: OutputFormat,
) -> Option<Stored> {
    let file = match output {
        ComputeOutput::Memory(memory) => Some(memory.file),
        _ => None,
    };
    let compute_result = if skips_conversion(ctx, file) {
        result
    } else {
        precision_conversion(vals, into, result, format.get(), ctx.comp)?
    };

    match output {
        // `SendOp::create(builder, loc, unit, compute_result, nullptr, name)`.
        ComputeOutput::Wire(to) => into.push(Op::Dataflow(dataflow::Op::Send {
            to: *to,
            data: compute_result.val(),
            ty: compute_result.ty(),
        })),
        // `agen::VectorStoreOp::create(.., compute_result, view.getResult(), name, ..)` — the value's
        // type is the CONVERTED one, while the view is still stated over `result_type`'s element.
        ComputeOutput::Memory(memory) => {
            let elements = memory_view(vals, into, ctx, memory, TransferSide::Store)?;
            into.push(Op::Agen(agen::Op::VectorStore {
                value: compute_result.val(),
                view: elements.view.result,
                indices: elements.base_address.indices.clone(),
                dbg_name: Some(ctx.name.to_owned()),
                access: agen::Access::Stated(elements.transfer_set),
                view_ty: elements.view.ty.clone(),
                ty: compute_result.ty(),
            }));
        }
        ComputeOutput::Ring(ring) => {
            let peer = ring_peer(vals, into, ring)?;
            let (to, _) = Link::<link::Sfp, link::Sfp>::between(ring.own, peer).ends();
            into.push(Op::Dataflow(dataflow::Op::Send {
                to,
                data: compute_result.val(),
                ty: compute_result.ty(),
            }));
        }
        ComputeOutput::Latch(latch) => {
            return Some(Stored::Latched {
                latch: *latch,
                data: compute_result.val(),
            });
        }
    }
    Some(Stored::Emitted)
}

/// Replaces: e094_constructComputeOutputOperands
///
/// **094/110** `SNComputeLowering.cpp:921` — [`compute_output_operand`] once per `outputs_[i]`,
/// stopping at the first refusal.
///
/// ⛔ EVERY OUTPUT STARTS FROM THE SAME UNCONVERTED RESULT: `Value compute_result = result` is
/// declared once and never reassigned (`:925`), and entry 087 takes its `result` BY VALUE — so a
/// second output does not inherit the first's precision conversion.
pub fn compute_output_operands(
    vals: &mut Values,
    into: &mut Vec<Op>,
    ctx: &OperandContext<'_>,
    outputs: &[(ComputeOutput<'_>, OutputFormat)],
    result: Computed,
) -> Option<Vec<Stored>> {
    outputs
        .iter()
        .map(|(output, format)| compute_output_operand(vals, into, ctx, output, result, *format))
        .collect()
}

/// WHICH OF THE PAIR IS BEING LOWERED — `compute_op.type_`, and the only two values that reach
/// `constructFMINorFMAXOperation`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MinOrMax {
    /// `FMAX`.
    Fmax,
    /// `FMIN` — the `else` of that one ternary.
    Fmin,
}

impl MinOrMax {
    /// `compute_op.type_ == FMAX ? compare_gt : compare_le` (`SNComputeLowering.cpp:1252-1255`).
    #[must_use]
    pub const fn comparison(self) -> vectorchain::CompareOp {
        match self {
            MinOrMax::Fmax => vectorchain::CompareOp::Gt,
            MinOrMax::Fmin => vectorchain::CompareOp::Le,
        }
    }
}

/// Replaces: e095_constructFMINorFMAXOperation
///
/// **095/110** `SNComputeLowering.cpp:1189` — two input operands, one `element_wise_compare` and one
/// `element_wise_selection` over them under the same lane mask, then every output operand.
///
/// ⛔⛔ THE STATE FILES TAKE THE **COMPARISON**, NOT THE SELECTION: a `PESTATE`/`SFPSTATE` output is
/// handed the `i1` vector at `bool_type` and every other output the selection at `result_type`
/// (`:1268-1281`), and `result_type` is entry 087's own conversion target — so the two arms differ in
/// the value AND in the type it is written at.
/// ⛔ EXACTLY TWO INPUTS IS THE TYPE: the reference builds every `inputs_[i]` but only `[0]` and
/// `[1]` reach either op. ⭐ An input's format rule is [`OutputFormat`]'s verbatim (`:1201-1206`),
/// where `get()` states the operand's type and entry 086 converts to `.operand`.
pub fn fmin_or_fmax_operation<A: Arch>(
    vals: &mut Values,
    into: &mut Vec<Op>,
    ctx: &OperandContext<'_>,
    which: MinOrMax,
    inputs: &[(ComputeInput<'_>, OutputFormat); 2],
    result_format: DataType,
    mask: MaskValue,
    outputs: &[(ComputeOutput<'_>, OutputFormat)],
) -> Option<Vec<Stored>> {
    let mut input_operand = |source: &ComputeInput<'_>, format: OutputFormat| -> Option<Computed> {
        let ty = type_from_compute_type(format.get(), ctx.comp)?;
        Some(
            compute_input_operand::<A>(
                vals,
                into,
                &OperandContext {
                    result_ty: ty,
                    ..*ctx
                },
                source,
                format.operand,
            )
            .unwrap_or_else(|| {
                emit_error(
                    ctx.name,
                    "Unable to construct Binary operation input operand",
                )
            }),
        )
    };
    let [(lhs_source, lhs_format), (rhs_source, rhs_format)] = inputs;
    let lhs = input_operand(lhs_source, *lhs_format)?;
    let rhs = input_operand(rhs_source, *rhs_format)?;

    // `getComputeOperandFormats(*dsc_).back()`, and `bool_type` is its lane count in `i1`.
    let result_ty = type_from_compute_type(result_format, ctx.comp)?;
    let bool_ty = Vector {
        len: result_ty.len,
        elem: ElemType::Int(1),
    };
    // ⛔ `DT_CHECK_MSG(mask_op, "Could not create valid mask")` CANNOT FIRE HERE: a [`Predicate`] has
    // no null, which is the guard entry 047 moved into the type.
    let lanes = static_mask_for_result(vals, into, result_ty, mask);

    let compared = vals.mint();
    let compare = vectorchain::Op::ElementWiseCompare {
        result: compared,
        op1: lhs.val(),
        op2: rhs.val(),
        mask: Some(lanes),
        compare_op: which.comparison(),
        dbg_name: Some(ctx.name.to_owned()),
        operand_ty: lhs.ty(),
        ty: bool_ty,
    };
    let cond = compare.binds_predicate()?;
    into.push(Op::VectorChain(compare));

    let selected = vals.mint();
    into.push(Op::VectorChain(vectorchain::Op::ElementWiseSelection {
        result: selected,
        cond,
        lhs: lhs.val(),
        rhs: rhs.val(),
        dbg_name: Some(ctx.name.to_owned()),
        mask: Some(lanes),
        ty: result_ty,
    }));

    Some(
        outputs
            .iter()
            .map(|(output, format)| {
                // `is_any_of(compute_op.outputs_[i], PESTATE, SFPSTATE)`.
                let state = matches!(
                    output,
                    ComputeOutput::Memory(memory) if matches!(memory.file, MemoryFile::State)
                );
                let (ty, data) = if state {
                    (bool_ty, compared)
                } else {
                    (result_ty, selected)
                };
                compute_output_operand(
                    vals,
                    into,
                    &OperandContext {
                        result_ty: ty,
                        ..*ctx
                    },
                    output,
                    Computed::of(data, ty),
                    *format,
                )
                .unwrap_or_else(|| {
                    emit_error(
                        ctx.name,
                        "Unable to construct FMIN/FMAX operation output operand",
                    )
                })
            })
            .collect(),
    )
}
/// WHICH LANE MASK A COMPUTE CARRIES — `computeMaskLoopOffsets_.empty()` picks
/// (`SNComputeLowering.cpp:997-1009`), and only the MAC reads the second arm.
pub enum ComputeMask<'m> {
    /// The map is empty: `getStaticContinuousMaskValue(.., compute_mask_)`.
    Static(MaskValue),
    /// It is not: `constructDynamicMasking(.., computeMaskLoopOffsets_.at(corelet_id))`.
    Dynamic(PtDynamicMask<'m>),
}

/// A DYNAMIC COMPUTE MASK, WHICH ONLY A PT MAY HAVE.
///
/// ⛔ `DT_CHECK_MSG(gen_comp == PT, "Dynamic masking allowed only in PT units")` (`:1002-1003`) IS
/// THIS TYPE: [`PtDynamicMask::on`] is the only way to build one and no other component yields it.
pub struct PtDynamicMask<'m> {
    /// `computeMaskLoopOffsets_.at(corelet_id_ == -1 ? 0 : corelet_id_)` — which corelet is the
    /// lowering's own state, so the entry arrives already chosen.
    pub mask: DynamicMask,
    /// Entry 028's inputs, for the induction variable the mask's symbol is.
    pub dims: &'m [PrimaryDim],
    /// ditto.
    pub mlir_loops: &'m [Val],
}

impl<'m> PtDynamicMask<'m> {
    /// ⛔ [`None`] IS THE `DT_CHECK`, and it is the whole check.
    #[must_use]
    pub fn on(
        comp: GenericComp,
        mask: DynamicMask,
        dims: &'m [PrimaryDim],
        mlir_loops: &'m [Val],
    ) -> Option<PtDynamicMask<'m>> {
        matches!(comp, GenericComp::Pt).then_some(PtDynamicMask {
            mask,
            dims,
            mlir_loops,
        })
    }
}

/// ONE MAC INPUT'S TYPE INPUTS — `formats[i]`, `elements[i]`, and the labelled data structure's
/// override of BOTH the format AND the category (`:964-979`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MacInputFormat {
    /// `{dataFormat_, scaledLdsCategory_}` of `labeledDs_[myLdsIdx_]`; [`None`] is `myLdsIdx_ == -1`.
    ///
    /// ⛔ THE CATEGORY COMES FROM THE LDS TOO, which no other compute in this file reads — a
    /// scaled operand's element is an MX type and the plain `REGULAR_TENSOR` default would print
    /// the wrong element for it.
    pub lds: Option<(DataType, TensorCategory)>,
    /// `getComputeOperandFormats(*dsc_)[i]` — also entry 086's conversion target.
    pub operand: DataType,
    /// `getComputeOperandSizes(dsc_global_.sysDef)[i]`.
    pub elements: Elements,
}

/// Replaces: e099_constructMACOperation
///
/// **099/110** `SNComputeLowering.cpp:942` — three input operands at their own widths, one
/// `multiply_and_accumulate` under the compute's lane mask, then every output operand.
///
/// ⛔⛔ THE RESULT TYPE IS THE **ACCUMULATOR'S**, `inputs[2].getType()` (`:996`): not a format's and
/// not the first operand's, because a MAC reduces and `op1`/`op2` are the wide pair.
/// ⛔ `FNMS` NEGATES `inputs[0]` AT THAT RESULT TYPE, not at its own (`:1013-1016`), which is why
/// the emitted MAC prints two different operand types — see [`vectorchain::Op::MultiplyAccumulate`].
/// ⭐ `constructTypeFromFormat`'s own failure cannot arise here: `elements[i]` is a stated count, so
/// [`VectorWidth::Given`] never consults [`StickWidth::of`].
pub fn mac_operation<A: Arch>(
    vals: &mut Values,
    into: &mut Vec<Op>,
    ctx: &OperandContext<'_>,
    mac: MacOp,
    inputs: &[(ComputeInput<'_>, MacInputFormat); 3],
    mask: ComputeMask<'_>,
    outputs: &[(ComputeOutput<'_>, OutputFormat)],
) -> Option<Vec<Stored>> {
    let mut input_operand = |source: &ComputeInput<'_>, format: MacInputFormat| -> Computed {
        let (format_of, category) = match format.lds {
            Some(pair) => pair,
            None => (format.operand, TensorCategory::Regular),
        };
        let result_ty = type_from_format(
            format_of,
            ctx.comp,
            category,
            VectorWidth::Given(format.elements),
        );
        compute_input_operand::<A>(
            vals,
            into,
            &OperandContext {
                result_ty,
                ..*ctx
            },
            source,
            format.operand,
        )
        .unwrap_or_else(|| emit_error(ctx.name, "Unable to construct MAC operation input operand"))
    };
    let [(a_source, a_format), (b_source, b_format), (acc_source, acc_format)] = inputs;
    let a = input_operand(a_source, *a_format);
    let b = input_operand(b_source, *b_format);
    let acc = input_operand(acc_source, *acc_format);

    let reduction_map = mac.reduction_map::<A>();
    let result_ty = acc.ty();
    let lanes = match mask {
        ComputeMask::Static(mask) => static_mask_for_result(vals, into, result_ty, mask),
        ComputeMask::Dynamic(dynamic) => dynamic_masking(
            vals,
            into,
            result_ty,
            dynamic.dims,
            dynamic.mlir_loops,
            dynamic.mask,
        )?,
    };

    // `if (type_ == FNMS) input0 = NegOp::create(builder, loc, result_type, input0)`.
    let a = if matches!(mac, MacOp::Fnms) {
        let negated = vals.mint();
        into.push(Op::VectorChain(vectorchain::Op::Neg {
            result: negated,
            input: a.val(),
            mask: None,
            input_ty: a.ty(),
            ty: result_ty,
        }));
        Computed::of(negated, result_ty)
    } else {
        a
    };

    let result = vals.mint();
    into.push(Op::VectorChain(vectorchain::Op::MultiplyAccumulate {
        result,
        a: a.val(),
        b: b.val(),
        acc: acc.val(),
        mask: Some(lanes),
        dbg_name: Some(ctx.name.to_owned()),
        reduction_map,
        a_ty: a.ty(),
        b_ty: b.ty(),
        ty: result_ty,
    }));

    compute_output_operands(
        vals,
        into,
        &OperandContext {
            result_ty,
            ..*ctx
        },
        outputs,
        Computed::of(result, result_ty),
    )
    .or_else(|| emit_error(ctx.name, "Unable to construct MAC operation output operand"))
}

/// `FMUL`'s two products — `instrAttribute_.mode_` (`SNComputeLowering.cpp:1099-1106`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MulMode {
    /// Any mode but 11 — `X*Y`, `mul`.
    Mul,
    /// `mode_ == 11` — `X*Y/2`, `mul_div2`.
    MulDiv2,
}

/// THE SIX `ComputeOpType`s THAT BECOME AN `element_wise_compare` (`:1136-1167`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Comparison {
    /// `GREATERTHAN`.
    GreaterThan,
    /// `GREATEREQUAL`.
    GreaterEqual,
    /// `LESSERTHAN`.
    LesserThan,
    /// `LESSEREQUAL`.
    LesserEqual,
    /// `EQUALTO`.
    EqualTo,
    /// `NOTEQUAL`.
    NotEqual,
}

impl Comparison {
    /// ⛔ THE REFERENCE'S OWN TWO LOCAL NAMES ARE SWAPPED — `LESSERTHAN` binds `le_operation` at
    /// `compare_lt` and `LESSEREQUAL` binds `lt_operation` at `compare_le` (`:1148-1159`). The
    /// ATTRIBUTE is what matters and this follows it, not the variable.
    #[must_use]
    pub const fn operator(self) -> vectorchain::CompareOp {
        match self {
            Comparison::GreaterThan => vectorchain::CompareOp::Gt,
            Comparison::GreaterEqual => vectorchain::CompareOp::Ge,
            Comparison::LesserThan => vectorchain::CompareOp::Lt,
            Comparison::LesserEqual => vectorchain::CompareOp::Le,
            Comparison::EqualTo => vectorchain::CompareOp::Eq,
            Comparison::NotEqual => vectorchain::CompareOp::Neq,
        }
    }
}

/// WHICH BINARY OR TERNARY COMPUTE — `compute_op.type_`, with `FMUL`'s `mode_` folded in.
///
/// ⛔ THE UNMATCHED `else` IS ABSENT RATHER THAN AN ARM: it is a `failure()` (`:1176`), so no
/// variant is the one thing this crate may not represent.
pub enum BinaryOrTernary<'b> {
    /// `FMAX` → `max`.
    Fmax,
    /// `FMIN` → `min`.
    Fmin,
    /// `FABSMAX` → `abs_max`.
    Fabsmax,
    /// `FMUL` → `mul` or `mul_div2`.
    Fmul(MulMode),
    /// `FSUB` → `sub`.
    Fsub,
    /// `OR` → `or0`.
    Or,
    /// `AND` → `and0`.
    And,
    /// `PACKMERGE` → a `pack`.
    PackMerge {
        /// `instrAttribute_.indices_`.
        indices: &'b [i32],
        /// `instrAttribute_.repetition_`, an `IndexAttr` on this op alone.
        repetition: u32,
        /// `instrAttribute_.sign_extend_`, as a `BoolAttr` here.
        sign_extend: bool,
    },
    /// One of the six comparisons.
    Compare(Comparison),
    /// `SELECT` → an `element_wise_selection`. ⛔ THE ONLY TERNARY.
    Select,
}

impl BinaryOrTernary<'_> {
    /// `is_any_of(type_, FMAX, FABSMAX, FMIN, FMUL, FSUB, OR, AND)` and the chain inside it
    /// (`:1090-1119`) — [`None`] for the arms that emit some other op.
    #[must_use]
    pub const fn binary_operator(&self) -> Option<vectorchain::BinaryOp> {
        match self {
            BinaryOrTernary::Fmax => Some(vectorchain::BinaryOp::Max),
            BinaryOrTernary::Fmin => Some(vectorchain::BinaryOp::Min),
            BinaryOrTernary::Fabsmax => Some(vectorchain::BinaryOp::AbsMax),
            BinaryOrTernary::Fmul(MulMode::MulDiv2) => Some(vectorchain::BinaryOp::MulDiv2),
            BinaryOrTernary::Fmul(MulMode::Mul) => Some(vectorchain::BinaryOp::Mul),
            BinaryOrTernary::Fsub => Some(vectorchain::BinaryOp::Sub),
            BinaryOrTernary::Or => Some(vectorchain::BinaryOp::Or),
            BinaryOrTernary::And => Some(vectorchain::BinaryOp::And),
            _ => None,
        }
    }
}

/// Replaces: e100_constructBinaryOrTernaryOperation
///
/// **100/110** `SNComputeLowering.cpp:1036` — two or three input operands, one of four `vectorchain`
/// ops under a static lane mask, then every output operand.
///
/// ⛔⛔ EVERY OP HERE IS BUILT AT `result_type`, WHICH IS `getComputeOperandFormats(..).back()` —
/// NOT `inputs[0].getType()`, as the reference's own commented-out line at `:1077` once had it. The
/// comparison arms therefore print a result of that format and not a `vector<Nxi1>`.
/// ⛔ THE `SELECT`'s CONDITION IS `inputs[0]` (`:1168-1171`) — an arriving operand, so the third
/// input is the one that needs three inputs to exist.
/// ⭐ `DT_CHECK_MSG(mask_op, "Could not create valid mask")` cannot fire: a [`Predicate`] has no null.
pub fn binary_or_ternary_operation<A: Arch>(
    vals: &mut Values,
    into: &mut Vec<Op>,
    ctx: &OperandContext<'_>,
    which: &BinaryOrTernary<'_>,
    inputs: &[(ComputeInput<'_>, OutputFormat)],
    result_format: DataType,
    mask: MaskValue,
    outputs: &[(ComputeOutput<'_>, OutputFormat)],
) -> Option<Vec<Stored>> {
    // `if (!is_any_of(compute_op.inputs_.size(), 2, 3)) return failure()`.
    let ([_, _] | [_, _, _]) = inputs else {
        return None;
    };
    let mut operands = Vec::with_capacity(inputs.len());
    for (source, format) in inputs {
        let result_ty = type_from_compute_type(format.get(), ctx.comp)?;
        operands.push(
            compute_input_operand::<A>(
                vals,
                into,
                &OperandContext {
                    result_ty,
                    ..*ctx
                },
                source,
                format.operand,
            )
            .unwrap_or_else(|| {
                emit_error(
                    ctx.name,
                    "Unable to construct Binary operation input operand",
                )
            }),
        );
    }
    let [op1, op2, rest @ ..] = operands.as_slice() else {
        return None;
    };

    let result_ty = type_from_compute_type(result_format, ctx.comp)?;
    let lanes = static_mask_for_result(vals, into, result_ty, mask);
    let result = vals.mint();
    let dbg_name = Some(ctx.name.to_owned());
    let op = match (which.binary_operator(), which) {
        (Some(binary_op), _) => vectorchain::Op::Binary {
            result,
            op1: op1.val(),
            op2: op2.val(),
            mask: Some(lanes),
            binary_op,
            dbg_name,
            // `builder.getDimIdentityMap()` — `(d0) -> (d0)`.
            op_specific_map: AffineMap::unary(AffineExpr::dim(0)),
            operand_ty: op1.ty(),
            ty: result_ty,
        },
        (
            None,
            BinaryOrTernary::PackMerge {
                indices,
                repetition,
                sign_extend,
            },
        ) => vectorchain::Op::Pack {
            result,
            op1: op1.val(),
            op2: op2.val(),
            mask: Some(lanes),
            indices: indices.to_vec(),
            dbg_name,
            repetition: *repetition,
            sign_extend: *sign_extend,
            operand_ty: op1.ty(),
            ty: result_ty,
        },
        (None, BinaryOrTernary::Compare(comparison)) => vectorchain::Op::ElementWiseCompare {
            result,
            op1: op1.val(),
            op2: op2.val(),
            mask: Some(lanes),
            compare_op: comparison.operator(),
            dbg_name,
            operand_ty: op1.ty(),
            ty: result_ty,
        },
        (None, BinaryOrTernary::Select) => {
            // ⛔ A TWO-INPUT `SELECT` READS `inputs[2]` IN THE REFERENCE AND IS OUT OF BOUNDS
            // (`:1170`); the arity that op needs is stated here instead.
            let [rhs] = rest else { return None };
            vectorchain::Op::ElementWiseSelection {
                result,
                cond: Predicate::of_operand(*op1),
                lhs: op2.val(),
                rhs: rhs.val(),
                dbg_name,
                mask: Some(lanes),
                ty: result_ty,
            }
        }
        (None, _) => return None,
    };
    into.push(Op::VectorChain(op));

    compute_output_operands(
        vals,
        into,
        &OperandContext {
            result_ty,
            ..*ctx
        },
        outputs,
        Computed::of(result, result_ty),
    )
    .or_else(|| {
        emit_error(
            ctx.name,
            "Unable to construct Binary operation output operand",
        )
    })
}

/// `FEST`'s TEN MODES — `instrAttribute_.mode_` (`SNComputeLowering.cpp:1316-1372`).
///
/// ⛔ 4 AND EVERYTHING ABOVE 9 ARE ABSENT: they are the *"Unknown estimate instruction."*
/// `emitError`, so no variant names them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FestMode {
    /// `0` — `exp_estimate {version = a}`.
    ExpA,
    /// `1` — `exp_estimate {version = b}`.
    ExpB,
    /// `2` — `rec_estimate`, which takes no version.
    Rec,
    /// `3` — `ln_estimate`, likewise.
    Ln,
    /// `5` — `rsqrt_estimate`, likewise. ⛔ AND 4 IS NOT A MODE.
    Rsqrt,
    /// `6` — `sigmoid_estimate {version = slope}`.
    SigmoidSlope,
    /// `7` — `sigmoid_estimate {version = offset}`.
    SigmoidOffset,
    /// `8` — `tanh_estimate {version = slope}`.
    TanhSlope,
    /// `9` — `tanh_estimate {version = offset}`.
    TanhOffset,
}

impl FestMode {
    /// WHICH ESTIMATE AND WHICH VERSION — ⛔ `rec`, `ln` AND `rsqrt` TAKE NO `version` ATTRIBUTE AT
    /// ALL, where the other six pass one explicitly.
    #[must_use]
    pub const fn estimate(
        self,
    ) -> (
        vectorchain::EstimateKind,
        Option<vectorchain::EstimateVersion>,
    ) {
        use vectorchain::EstimateKind as Kind;
        use vectorchain::EstimateVersion as Version;
        match self {
            FestMode::ExpA => (Kind::Exp, Some(Version::A)),
            FestMode::ExpB => (Kind::Exp, Some(Version::B)),
            FestMode::Rec => (Kind::Rec, None),
            FestMode::Ln => (Kind::Ln, None),
            FestMode::Rsqrt => (Kind::Rsqrt, None),
            FestMode::SigmoidSlope => (Kind::Sigmoid, Some(Version::Slope)),
            FestMode::SigmoidOffset => (Kind::Sigmoid, Some(Version::Offset)),
            FestMode::TanhSlope => (Kind::Tanh, Some(Version::Slope)),
            FestMode::TanhOffset => (Kind::Tanh, Some(Version::Offset)),
        }
    }
}

/// `REDUCE`'s FIVE MODES — `mode_` 1, 8, 10, 12, 14 (`:1444-1459`); every other value is the
/// *"Unknown binary operation for reduction."* `emitError`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceMode {
    /// `1` → `add`.
    Add,
    /// `8` → `max`.
    Max,
    /// `10` → `abs_max`.
    AbsMax,
    /// `12` → `min`.
    Min,
    /// `14` → `abs_min`.
    AbsMin,
}

impl ReduceMode {
    /// The `reduction_op` a `scan_with_gap` carries.
    #[must_use]
    pub const fn reduction(self) -> vectorchain::BinaryOp {
        match self {
            ReduceMode::Add => vectorchain::BinaryOp::Add,
            ReduceMode::Max => vectorchain::BinaryOp::Max,
            ReduceMode::AbsMax => vectorchain::BinaryOp::AbsMax,
            ReduceMode::Min => vectorchain::BinaryOp::Min,
            ReduceMode::AbsMin => vectorchain::BinaryOp::AbsMin,
        }
    }
}

/// `SPLAT`'s `instrAttribute_.sign_extend_` — ⛔ TWO VALUES, AND A THIRD IS THE `emitError`
/// *"sign_extend has to be either 0 or 1."* (`:1435-1437`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignExtend {
    /// `0` — every lane reads element 0 and the op has no `pad` operand.
    No,
    /// `1` — lane 0 reads element 0 and every other lane reads a `pad` zero.
    Yes,
}

impl SignExtend {
    /// THE `indices` TABLE, WHICH IS THE ELEMENT'S AS WELL AS THIS FLAG'S — eight entries for an
    /// `f16` and four for an `f32` (`:1409-1434`).
    ///
    /// ⛔ [`None`] IS THE REFERENCE'S NULL OP: neither `if` has an `else`, so any other element
    /// leaves `unary_op` unset and `getResult()` reads it anyway. The refusal moves to the one line
    /// that can still name the element.
    #[must_use]
    pub fn splat_indices(self, elem: ElemType) -> Option<Vec<i32>> {
        let lanes = match elem {
            ElemType::F16 => 8,
            ElemType::F32 => 4,
            _ => return None,
        };
        Some(match self {
            SignExtend::No => vec![0; lanes],
            SignExtend::Yes => std::iter::once(0)
                .chain(std::iter::repeat_n(-1, lanes - 1))
                .collect(),
        })
    }
}

/// WHICH UNARY COMPUTE — `compute_op.type_`, with the `mode_` each arm dispatches on folded in.
pub enum UnaryOp<'u> {
    /// `FEST`.
    Fest(FestMode),
    /// `ICVT` `mode_ == 7` → a `fast_exp`. ⛔ NO OTHER ICVT MODE IS AN OP (`:1394-1402`).
    IcvtFastExp,
    /// `FLOOR` → a `floor`.
    Floor,
    /// `SPLAT` → a `shuffle` over one element.
    Splat {
        /// `instrAttribute_.sign_extend_`.
        sign_extend: SignExtend,
        /// `instrAttribute_.repetition_`.
        repetition: u32,
    },
    /// `REDUCE` → a `scan_with_gap`.
    Reduce(ReduceMode),
    /// `SHUFFLE` → a `shuffle`, and ⛔ THE ONLY ARM THAT CHANGES THE RESULT TYPE.
    Shuffle {
        /// `outputsLdsAndLoopOffsets_[0].myLdsIdx_`, else `getComputeOperandFormats(*dsc_)[1]`.
        output: OutputFormat,
        /// `instrAttribute_.indices_`.
        indices: &'u [i32],
        /// `instrAttribute_.repetition_`.
        repetition: u32,
    },
}

/// Replaces: e101_constructUnaryOperation
///
/// **101/110** `SNComputeLowering.cpp:1276` — one input operand, one of six `vectorchain` ops under
/// a static lane mask, then every output operand.
///
/// ⛔⛔ THE RESULT TYPE IS THE **INPUT FORMAT'S**, `unary_op_result_type = input_type` (`:1310`),
/// and `SHUFFLE` is the one arm that reassigns it — to its own output format (`:1490`). The mask is
/// built BEFORE that reassignment, so a shuffle's mask is stated over the INPUT's width.
/// ⛔ AND THE ELEMENT THE SPLAT TABLE IS CHOSEN BY IS THAT FORMAT'S TOO, not the operand's: entry
/// 086 may have converted the value to a different precision on the way in.
/// ⭐ `scan_with_gap` AND BOTH SHUFFLES TAKE NO MASK, yet the `create_affine_mask` is still emitted.
pub fn unary_operation<A: Arch>(
    vals: &mut Values,
    into: &mut Vec<Op>,
    ctx: &OperandContext<'_>,
    which: &UnaryOp<'_>,
    input: (&ComputeInput<'_>, OutputFormat),
    mask: MaskValue,
    outputs: &[(ComputeOutput<'_>, OutputFormat)],
) -> Option<Vec<Stored>> {
    let (source, format) = input;
    let input_ty = type_from_compute_type(format.get(), ctx.comp)?;
    let operand = compute_input_operand::<A>(
        vals,
        into,
        &OperandContext {
            result_ty: input_ty,
            ..*ctx
        },
        source,
        format.operand,
    )
    .unwrap_or_else(|| emit_error(ctx.name, "Unable to construct unary operation input operand"));

    let lanes = static_mask_for_result(vals, into, input_ty, mask);
    let dbg_name = Some(ctx.name.to_owned());
    // ⛔ EACH ARM MINTS ITS RESULT AFTER ITS OWN OPERANDS — a `pad` zero is an `arith.constant` the
    // reference creates BEFORE the shuffle that reads it (`:1414-1416`).
    let (result, result_ty, op) = match which {
        UnaryOp::Fest(mode) => {
            let (kind, version) = mode.estimate();
            let result = vals.mint();
            (
                result,
                input_ty,
                vectorchain::Op::Estimate {
                    result,
                    input: operand.val(),
                    kind,
                    mask: Some(lanes),
                    dbg_name,
                    version,
                    input_ty: operand.ty(),
                    ty: input_ty,
                },
            )
        }
        UnaryOp::IcvtFastExp => {
            let result = vals.mint();
            (
                result,
                input_ty,
                vectorchain::Op::FastExp {
                    result,
                    input: operand.val(),
                    mask: Some(lanes),
                    dbg_name,
                    input_ty: operand.ty(),
                    ty: input_ty,
                },
            )
        }
        UnaryOp::Floor => {
            let result = vals.mint();
            (
                result,
                input_ty,
                vectorchain::Op::Floor {
                    result,
                    input: operand.val(),
                    mask: Some(lanes),
                    dbg_name,
                    input_ty: operand.ty(),
                    ty: input_ty,
                },
            )
        }
        UnaryOp::Splat {
            sign_extend,
            repetition,
        } => {
            let indices = sign_extend.splat_indices(input_ty.elem)?;
            // `ValueRange{zero.getResult()}` into the `pad` group, and only when sign-extending.
            let pad = match sign_extend {
                SignExtend::No => Vec::new(),
                SignExtend::Yes => vec![vectorchain::ShuffleVariable {
                    val: constant_index(vals, into, 0),
                    ty: ScalarTy::Index,
                }],
            };
            let result = vals.mint();
            (
                result,
                input_ty,
                vectorchain::Op::Shuffle {
                    result,
                    input: operand.val(),
                    variable: Vec::new(),
                    pad,
                    dbg_name,
                    indices,
                    repetition: *repetition,
                    input_ty: operand.ty(),
                    ty: input_ty,
                },
            )
        }
        UnaryOp::Reduce(mode) => {
            let result = vals.mint();
            (
                result,
                input_ty,
                vectorchain::Op::ScanWithGap {
                    result,
                    input: operand.val(),
                    reduction_op: mode.reduction(),
                    dbg_name,
                    input_ty: operand.ty(),
                    ty: input_ty,
                },
            )
        }
        UnaryOp::Shuffle {
            output,
            indices,
            repetition,
        } => {
            let output_ty = type_from_compute_type(output.get(), ctx.comp)?;
            // "In case there is zero padding in the pattern, pass in zero constant to pad operand"
            // (`:1495-1499`) — unconditionally, whatever the pattern.
            let pad = vec![vectorchain::ShuffleVariable {
                val: constant_index(vals, into, 0),
                ty: ScalarTy::Index,
            }];
            let result = vals.mint();
            (
                result,
                output_ty,
                vectorchain::Op::Shuffle {
                    result,
                    input: operand.val(),
                    variable: Vec::new(),
                    pad,
                    dbg_name,
                    indices: indices.to_vec(),
                    repetition: *repetition,
                    input_ty: operand.ty(),
                    ty: output_ty,
                },
            )
        }
    };
    into.push(Op::VectorChain(op));

    compute_output_operands(
        vals,
        into,
        &OperandContext {
            result_ty,
            ..*ctx
        },
        outputs,
        Computed::of(result, result_ty),
    )
    .or_else(|| {
        emit_error(
            ctx.name,
            "Unable to construct Unary operation output operand",
        )
    })
}
// ══════════════════════════════════════════════════════════════════════════════════════════════
// 103/110 — ONE COMPUTE STATEMENT, ROUTED TO ITS FAMILY
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// WHICH OF THE FIVE FAMILIES ONE COMPUTE'S `type_` ROUTES TO — the five `is_any_of` chains of
/// `constructComputeOperation` (`:1570-1635`), each carrying exactly its constructor's arguments.
///
/// ⛔ A SIXTH CHAIN IS ABSENT BY CONSTRUCTION: the `else` is
/// `emitError("Unknown compute operation in constructComputeOperation")` (`:1643`).
///
/// ⚠️ [`BinaryOrTernary::Fmax`] AND `Fmin` ARE UNREACHABLE THROUGH IT — `FMAX` and `FMIN` are claimed
/// by the third chain (`:1605-1606`), so entry 100's own arms for them are dead here.
pub enum ComputeFamily<'f> {
    /// `IMA8`|`IMA4`|`FMA4`|`FMA8`|`FMA16`|`FMA32`|`FNMS` → entry 099, and the only arm that writes
    /// `precision`.
    Mac {
        /// `compute_op.type_`.
        mac: MacOp,
        /// The three operands, whose `lds` categories decide the `mx` prefix.
        inputs: &'f [(ComputeInput<'f>, MacInputFormat); 3],
        /// `compute_mask_`, or the dynamic map a PT may carry.
        mask: ComputeMask<'f>,
        /// `compute_op.outputs_`.
        outputs: &'f [(ComputeOutput<'f>, OutputFormat)],
    },
    /// `FMUL`|`FSUB`|`PACKMERGE`|`FABSMAX`|the six comparisons|`SELECT`|`OR`|`AND` → entry 100.
    BinaryOrTernary {
        /// `compute_op.type_`, with `FMUL`'s `mode_` folded in.
        which: &'f BinaryOrTernary<'f>,
        /// Two operands, or three where the `SELECT`'s condition is one of them.
        inputs: &'f [(ComputeInput<'f>, OutputFormat)],
        /// `getComputeOperandFormats(*dsc_).back()`.
        result_format: DataType,
        /// `compute_mask_`.
        mask: MaskValue,
        /// `compute_op.outputs_`.
        outputs: &'f [(ComputeOutput<'f>, OutputFormat)],
    },
    /// `FMAX`|`FMIN` → entry 095.
    MinOrMax {
        /// Which of the pair.
        which: MinOrMax,
        /// The two operands it compares and then selects between.
        inputs: &'f [(ComputeInput<'f>, OutputFormat); 2],
        /// `getComputeOperandFormats(*dsc_).back()`.
        result_format: DataType,
        /// `compute_mask_`.
        mask: MaskValue,
        /// `compute_op.outputs_`.
        outputs: &'f [(ComputeOutput<'f>, OutputFormat)],
    },
    /// `FEST`|`ICVT`|`SPLAT`|`REDUCE`|`SHUFFLE`|`FLOOR` → entry 101.
    Unary {
        /// `compute_op.type_`, with the `mode_` each arm dispatches on folded in.
        which: &'f UnaryOp<'f>,
        /// The one operand and the format it is converted to.
        input: (&'f ComputeInput<'f>, OutputFormat),
        /// `compute_mask_`.
        mask: MaskValue,
        /// `compute_op.outputs_`.
        outputs: &'f [(ComputeOutput<'f>, OutputFormat)],
    },
    /// The 29 opaque types → entry 053, which stores nothing and cannot refuse.
    Opaque {
        /// `getOpaqueFunctionName(type_)`.
        func: OpaqueFunc,
        /// `read_write_registers`.
        read_write: &'f [(RegName, RegAddr)],
        /// `read_only_registers`.
        read_only: &'f [(RegName, RegAddr)],
        /// `parameters`.
        params: &'f [(ParamKey, ParamValue)],
    },
}

/// WHAT ONE COMPUTE STATEMENT LEAVES BEHIND.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComputeOperation {
    /// The output operands the family's constructor stored, and empty for the opaque call.
    pub stored: Vec<Stored>,
    /// `precision`, which only the MAC chain writes (`:1579-1580`) — [`None`] is the reference's `""`,
    /// which entry 008 refuses.
    pub precision: Option<dataflow::Precision>,
}

/// Replaces: e103_constructComputeOperation
///
/// **103/110** `SNComputeLowering.cpp:1567` — one compute statement, routed to its family's
/// constructor, plus the `precision` a MAC leaves on the unit for entry 008.
///
/// ⛔ [`None`] IS NOT A REFUSAL: `comp_ != compute->exUnit_` emits nothing and succeeds (`:1568`) —
/// every unit but the execution unit skips the whole body.
/// ⛔ AND NO `failure()` HERE IS REACHABLE: all six are preceded by an `emitError` that raises.
#[must_use]
pub fn compute_operation<A: Arch>(
    vals: &mut Values,
    into: &mut Vec<Op>,
    ctx: &OperandContext<'_>,
    family: ComputeFamily<'_>,
) -> Option<ComputeOperation> {
    // `if (comp_ == compute->exUnit_)` — the whole body is inside it.
    if ctx.comp != ctx.ex_unit {
        return None;
    }
    Some(match family {
        ComputeFamily::Mac {
            mac,
            inputs,
            mask,
            outputs,
        } => {
            let stored = mac_operation::<A>(vals, into, ctx, mac, inputs, mask, outputs)
                .unwrap_or_else(|| emit_error(ctx.name, "Unable to construct MAC operation"));
            // `for (i) { category = REGULAR_TENSOR; if (myLdsIdx_ != -1) { category = ..; if
            //  (category != REGULAR_TENSOR) { precision = "mx" + precision; break; } } }` — an
            // operand with no labelled data structure leaves the default and does not break.
            let category = inputs
                .iter()
                .find_map(|(_, format)| match format.lds {
                    Some((_, TensorCategory::Scaled)) => Some(TensorCategory::Scaled),
                    _ => None,
                })
                .unwrap_or(TensorCategory::Regular);
            ComputeOperation {
                stored,
                precision: Some(mac.precision(category)),
            }
        }
        ComputeFamily::BinaryOrTernary {
            which,
            inputs,
            result_format,
            mask,
            outputs,
        } => ComputeOperation {
            stored: binary_or_ternary_operation::<A>(
                vals,
                into,
                ctx,
                which,
                inputs,
                result_format,
                mask,
                outputs,
            )
            .unwrap_or_else(|| emit_error(ctx.name, "Unable to construct binary operation")),
            precision: None,
        },
        ComputeFamily::MinOrMax {
            which,
            inputs,
            result_format,
            mask,
            outputs,
        } => ComputeOperation {
            // ⚠️ THE SAME SENTENCE AS THE CHAIN ABOVE — "binary operation" (`:1608`).
            stored: fmin_or_fmax_operation::<A>(
                vals,
                into,
                ctx,
                which,
                inputs,
                result_format,
                mask,
                outputs,
            )
            .unwrap_or_else(|| emit_error(ctx.name, "Unable to construct binary operation")),
            precision: None,
        },
        ComputeFamily::Unary {
            which,
            input,
            mask,
            outputs,
        } => ComputeOperation {
            stored: unary_operation::<A>(vals, into, ctx, which, input, mask, outputs)
                .unwrap_or_else(|| emit_error(ctx.name, "Unable to construct unary operation")),
            precision: None,
        },
        // ⚠️ `emitError("Unable to construct operation for Opaque function")` (`:1637-1639`) IS
        // UNREACHABLE: entry 053 builds the op with no way to fail.
        ComputeFamily::Opaque {
            func,
            read_write,
            read_only,
            params,
        } => {
            into.push(opaque_operation(
                ctx.name,
                func,
                read_write,
                read_only,
                params,
            ));
            ComputeOperation {
                stored: Vec::new(),
                precision: None,
            }
        }
    })
}

#[cfg(test)]
mod unit_tests {
    use super::{
        BinaryOrTernary, ComputeFamily, ComputeInput, ComputeMask, ComputeOutput, DynamicMask,
        Handlers, Latch, MacInputFormat, MacOp, MaskValue, MinOrMax, MulMode, OperandContext,
        OutputFormat, SignExtend, StickWidth, Stored, UnaryOp, VectorWidth,
        binary_or_ternary_operation, compute_input_operand, compute_operation,
        compute_output_operand, compute_output_operands, dictionary_order, dynamic_masking,
        fmin_or_fmax_operation, is_integer, mac_operation, opaque_operation, precision_conversion,
        single_val_custom_vector, static_mask_for_result, type_from_compute_type, type_from_format,
        unary_operation,
    };
    use crate::arch::{Dd2, Elements, Sen1p5};
    use crate::bridges::dataflow_ir_to_sentient::vc_helper::{
        EnclosingLoop, MaskValue as PtMaskValue, PtUnit, get_mask_value_for_pt,
    };
    use crate::bridges::superdsc_to_dataflow_ir::control_flow::PrimaryDim;
    use crate::generated::{DataType, OpaqueFunc, ParamKey, ParamValue, RegName};
    use crate::islands::dataflow_ir::Values;
    use crate::islands::dataflow_ir::dialects::dataflow::{Opaque, RegAddr};
    use crate::islands::dataflow_ir::dialects::vectorchain::{Computed, LaneMask, Predicate};
    use crate::islands::dataflow_ir::dialects::{Op, Val, arith, dataflow, vectorchain};
    use crate::islands::dataflow_ir::print::emit;
    use crate::islands::dataflow_ir::ty::{
        AffineExpr, AffineMap, Constraint, ElemType, GenericComp, IntegerSet, ScalarTy,
        TensorCategory, Vector,
    };
    use crate::islands::sentient::dialects::{self as sen, Definitions, sentient};

    /// ⛔ THE `-1` ARM, FORMAT BY FORMAT — including `BOOL`'s forced 16 and the `SENINT24`-on-PT
    /// width the reference's own `DT_CHECK_MSG` refuses.
    #[test]
    fn one_stick_of_each_format_and_the_width_that_does_not_divide() {
        for (format, on, category, len, elem) in [
            (
                DataType::Sen169Fp16,
                GenericComp::Sfp,
                TensorCategory::Regular,
                64,
                ElemType::F16,
            ),
            (
                DataType::Sen143Fp8,
                GenericComp::Sfp,
                TensorCategory::Regular,
                128,
                ElemType::F8E4M3Fn,
            ),
            (
                DataType::Sen121Fp4,
                GenericComp::Sfp,
                TensorCategory::Scaled,
                256,
                ElemType::MxFloat(4),
            ),
            (
                DataType::Senint4,
                GenericComp::Pt,
                TensorCategory::Regular,
                256,
                ElemType::Int(4),
            ),
            // ⛔ `i1` ELEMENTS, SIXTY-FOUR OF THEM: the width is overridden to 16.
            (
                DataType::Bool,
                GenericComp::Sfp,
                TensorCategory::Regular,
                64,
                ElemType::Int(1),
            ),
            // ⛔ SENINT24 OFF THE PT IS `i16`, so it does divide.
            (
                DataType::Senint24,
                GenericComp::Sfp,
                TensorCategory::Regular,
                64,
                ElemType::Int(16),
            ),
        ] {
            let width =
                StickWidth::of(format, on, category).expect("these widths divide a 1024-bit stick");
            let ty = type_from_format(format, on, category, VectorWidth::OneStick(width));
            assert_eq!(ty, Vector { len, elem }, "{format:?} on {on:?}");
            assert_eq!(is_integer(ty.elem), matches!(elem, ElemType::Int(_)));
        }
        // ⛔ THE ABORT: 1024 % 24 != 0.
        assert_eq!(
            StickWidth::of(DataType::Senint24, GenericComp::Pt, TensorCategory::Regular),
            None
        );
        // ⭐ AND A GIVEN COUNT IS TAKEN AS WRITTEN.
        assert_eq!(
            type_from_format(
                DataType::Bfloat16,
                GenericComp::Pe,
                TensorCategory::Regular,
                VectorWidth::Given(Elements(32)),
            ),
            Vector {
                len: 32,
                elem: ElemType::Bf16,
            }
        );
    }

    /// ⛔ `repetition` IS THE ELEMENT COUNT AND THE BITSTREAM HOLDS ONE VALUE — `:441-447`.
    #[test]
    fn the_splat_repeats_element_zero_across_the_whole_vector() {
        let ty = Vector {
            len: 128,
            elem: ElemType::MxFloat(8),
        };
        let mut vals = Values::default();
        let mut body = Vec::new();
        let out = single_val_custom_vector(&mut vals, &mut body, "v", 7, ty);
        assert_eq!(
            body,
            vec![
                Op::VectorChain(vectorchain::Op::ConstantBitstream {
                    result: Val(0),
                    value: vec![7],
                    ty,
                    is_symbol: false,
                }),
                Op::VectorChain(vectorchain::Op::Shuffle {
                    pad: Vec::new(),
                    variable: Vec::new(),
                    dbg_name: Some("v".to_owned()),
                    result: Val(1),
                    input: Val(0),
                    indices: vec![0],
                    repetition: 128,
                    input_ty: ty,
                    ty,
                }),
            ]
        );
        assert_eq!(out, Computed::of(Val(1), ty));
    }

    /// ⛔ THE TWO GENERATIONS, ARM BY ARM — including the `mod 128` the older one needs and the
    /// `floordiv 1` FMA32 writes on both.
    #[test]
    fn the_reduction_factor_is_the_generations_and_ima_wraps_at_a_hundred_and_twenty_eight() {
        let i = || AffineExpr::dim(0);
        for (op, dd2, sen) in [
            (MacOp::Fma16, i().floordiv(1), i().floordiv(4)),
            (MacOp::Fnms, i().floordiv(1), i().floordiv(4)),
            (MacOp::Fma32, i().floordiv(1), i().floordiv(1)),
            (MacOp::Fma8, i().floordiv(2), i().floordiv(16)),
            (MacOp::Ima8, i().modulo(128).floordiv(2), i().floordiv(16)),
            (MacOp::Ima4, i().modulo(128).floordiv(4), i().floordiv(32)),
        ] {
            assert_eq!(
                op.reduction_map::<Dd2>(),
                AffineMap::unary(dd2),
                "{op:?} on dd2"
            );
            assert_eq!(
                op.reduction_map::<Sen1p5>(),
                AffineMap::unary(sen),
                "{op:?} on sen1p5"
            );
        }
        // ⛔ FMA4 HAS ONE FACTOR BECAUSE IT HAS ONE GENERATION.
        assert_eq!(
            MacOp::Fma4.reduction_map::<Sen1p5>(),
            AffineMap::unary(i().floordiv(32))
        );
    }

    /// ⛔ THE SELECTION FACTOR DIVERGES FROM THE REDUCTION FACTOR off SEN1P5, and FMA32/FNMS have no
    /// selection at all.
    #[test]
    fn the_l0_splat_is_a_mod_and_the_two_float_arms_turn_on_the_original_type() {
        assert_eq!(
            MacOp::Ima8.selection_map_from_l0::<Dd2>(DataType::Senint8, ElemType::Int(8)),
            Some((
                AffineMap::unary(AffineExpr::dim(0).modulo(4)),
                Vector {
                    len: 256,
                    elem: ElemType::Int(8),
                },
            ))
        );
        // ⛔ AND ITS REDUCTION IS `floordiv 2` OVER THE SAME OP ON THE SAME ARCH.
        assert_eq!(
            MacOp::Ima8.reduction_map::<Dd2>(),
            AffineMap::unary(AffineExpr::dim(0).modulo(128).floordiv(2))
        );
        assert_eq!(
            MacOp::Ima4.selection_map_from_l0::<Sen1p5>(DataType::Senint4, ElemType::Int(4)),
            Some((
                AffineMap::unary(AffineExpr::dim(0).modulo(32)),
                Vector {
                    len: 2048,
                    elem: ElemType::Int(4),
                },
            ))
        );
        // ⭐ `BFLOAT16` IS THE ONLY FORMAT THAT PICKS `bf16`.
        for (format, elem) in [
            (DataType::Bfloat16, ElemType::Bf16),
            (DataType::Sen169Fp16, ElemType::F16),
        ] {
            assert_eq!(
                MacOp::Fma16.selection_map_from_l0::<Sen1p5>(format, ElemType::F16),
                Some((
                    AffineMap::unary(AffineExpr::dim(0).modulo(4)),
                    Vector { len: 256, elem },
                ))
            );
        }
        // ⛔ AN MX ORIGINAL MAKES THE SPLAT A CUSTOM VECTOR.
        assert_eq!(
            MacOp::Fma8
                .selection_map_from_l0::<Dd2>(DataType::Sen143Fp8, ElemType::MxFloat(8))
                .map(|(_, ty)| ty),
            Some(Vector {
                len: 128,
                elem: ElemType::MxFloat(8),
            })
        );
        assert_eq!(
            MacOp::Fma8
                .selection_map_from_l0::<Dd2>(DataType::Sen143Fp8, ElemType::F8E4M3Fn)
                .map(|(_, ty)| ty),
            Some(Vector {
                len: 128,
                elem: ElemType::F8E4M3Fn,
            })
        );
        // ⛔ THE TWO THE REFERENCE REFUSES.
        for op in [MacOp::Fma32, MacOp::Fnms] {
            assert_eq!(
                op.selection_map_from_l0::<Sen1p5>(DataType::Sen169Fp16, ElemType::F16),
                None
            );
        }
    }

    /// ⛔ ALL FOUR ARMS PLUS THE TWO PASS-THROUGHS — and nothing is emitted for either of those.
    #[test]
    fn a_conversion_is_emitted_only_where_the_two_types_differ() {
        let int16 = Vector {
            len: 64,
            elem: ElemType::Int(16),
        };
        let mut vals = Values::default();
        let mut body = Vec::new();
        // ⛔ MX IN, THE SAME VALUE OUT.
        let mx = Computed::of(
            Val(9),
            Vector {
                len: 128,
                elem: ElemType::MxFloat(8),
            },
        );
        assert_eq!(
            precision_conversion(
                &mut vals,
                &mut body,
                mx,
                DataType::Sen169Fp16,
                GenericComp::Sfp
            ),
            Some(mx)
        );
        // ⛔ AND THE EQUAL-TYPE ARM TOO — `vector<64xi16>` is what `SENINT24` is off the PT.
        let same = Computed::of(Val(9), int16);
        assert_eq!(
            precision_conversion(
                &mut vals,
                &mut body,
                same,
                DataType::Senint24,
                GenericComp::Sfp
            ),
            Some(same)
        );
        assert!(body.is_empty(), "neither pass-through emits an op");
        // ⭐ THE PE'S OWN int8 RESULT, PROMOTED: `vector<64xi16>` to `vector<64xf16>`
        // (`dcc/test/PE/int8-kg3-pe.mlir:98`).
        let fp16 = Vector {
            len: 64,
            elem: ElemType::F16,
        };
        assert_eq!(
            precision_conversion(
                &mut vals,
                &mut body,
                same,
                DataType::Sen169Fp16,
                GenericComp::Sfp
            ),
            Some(Computed::of(Val(0), fp16))
        );
        assert_eq!(
            body,
            vec![Op::Arith(arith::Op::Convert {
                result: Val(0),
                kind: arith::ConvertKind::SiToFp,
                input: Val(9),
                input_ty: int16,
                ty: fp16,
            })]
        );
        // ⭐ AND BACK DOWN, WHICH IS THE SFP FIXTURE'S OWN `fptosi` (`dcc/test/SFP/csqint8-sfp.mlir:112`).
        body.clear();
        let from_fp16 = Computed::of(Val(9), fp16);
        let int8 = Vector {
            len: 64,
            elem: ElemType::Int(8),
        };
        assert_eq!(
            precision_conversion(
                &mut vals,
                &mut body,
                from_fp16,
                DataType::Senint8,
                GenericComp::Pt
            ),
            Some(Computed::of(Val(1), int8))
        );
        assert_eq!(
            body,
            vec![Op::Arith(arith::Op::Convert {
                result: Val(1),
                kind: arith::ConvertKind::FpToSi,
                input: Val(9),
                input_ty: fp16,
                ty: int8,
            })]
        );
        // ⭐ FLOAT TO FLOAT IS THE CAST.
        body.clear();
        let bf16 = Vector {
            len: 64,
            elem: ElemType::Bf16,
        };
        assert_eq!(
            precision_conversion(
                &mut vals,
                &mut body,
                from_fp16,
                DataType::Bfloat16,
                GenericComp::Sfp
            ),
            Some(Computed::of(Val(2), bf16))
        );
        assert_eq!(
            body,
            vec![Op::VectorChain(vectorchain::Op::Cast {
                result: Val(2),
                input: Val(9),
                input_ty: fp16,
                ty: bf16,
            })]
        );
        // ⛔ INTEGER TO INTEGER IS THE REFERENCE'S `failure()`.
        body.clear();
        assert_eq!(
            precision_conversion(
                &mut vals,
                &mut body,
                same,
                DataType::Senint8,
                GenericComp::Pt
            ),
            None
        );
    }

    /// ⛔ THE NAME IS THE `dbgName` AND THE DICTIONARIES ARE HANDED OVER UNSORTED — the printer's
    /// key order is what `std::map` was.
    #[test]
    fn the_opaque_carries_the_compute_name_as_its_debug_name() {
        let params = [
            (ParamKey::Prec, ParamValue::Fp16),
            (ParamKey::In0, ParamValue::Fp16),
        ];
        let read_only = [(RegName::A00, RegAddr(0))];
        let op = opaque_operation(
            "opaque_op #1",
            OpaqueFunc::Reciprocal,
            &[],
            &read_only,
            &params,
        );
        assert_eq!(
            op,
            Op::Dataflow(dataflow::Op::Opaque(Opaque {
                func: OpaqueFunc::Reciprocal,
                read_write: vec![],
                read_only: read_only.to_vec(),
                params: params.to_vec(),
                dbg_name: Some("opaque_op #1".to_owned()),
            }))
        );
        // ⭐ AND IT PRINTS IN KEY ORDER WITHOUT THIS FUNCTION SORTING ANYTHING.
        let mut text = String::new();
        emit(&mut text, &op, 0);
        assert!(text.contains("dbgName = \"opaque_op #1\""), "{text}");
        let dic = text
            .split("parameter_dictionary = ")
            .nth(1)
            .expect("the op prints a parameter dictionary");
        assert!(dic.starts_with("{in0 = "), "{dic}");
    }

    /// ⛔ KEY ORDER, NOT INSERTION ORDER.
    #[test]
    fn the_dictionary_comes_out_in_key_spelling_order() {
        let entries = [
            (ParamKey::Prec, ParamValue::Fp16),
            (ParamKey::In1, ParamValue::Fp16),
            (ParamKey::In0, ParamValue::Fp16),
        ];
        assert_eq!(
            dictionary_order(&entries, ParamKey::spelling)
                .iter()
                .map(|(key, _)| key.spelling())
                .collect::<Vec<_>>(),
            vec!["in0", "in1", "prec"]
        );
    }
    /// 🎯 047/110 — ⛔ IT IS ENTRY 046 OVER THE MASKED RESULT'S OWN WIDTH, and `getDimSize` is the
    /// only difference: the emitted op is the same `create_affine_mask` with the same prefix.
    #[test]
    fn the_static_mask_reads_its_width_off_the_result() {
        for (row, live) in [
            (
                Vector {
                    len: 64,
                    elem: ElemType::F16,
                },
                48,
            ),
            (
                Vector {
                    len: 128,
                    elem: ElemType::Int(8),
                },
                96,
            ),
        ] {
            let mut vals = Values::default();
            let mut body = Vec::new();
            let predicate = static_mask_for_result(&mut vals, &mut body, row, MaskValue::Live6);
            assert_eq!(
                body,
                vec![Op::VectorChain(vectorchain::Op::CreateAffineMask {
                    result: Val(0),
                    mask: LaneMask::prefix_of(
                        live,
                        Vector {
                            len: row.len,
                            elem: ElemType::Int(1),
                        }
                    ),
                })]
            );
            // ⛔ THE MASK IS `vector<{n}xi1>`, NOT THE MASKED VALUE'S ELEMENT TYPE.
            assert_eq!(predicate.ty().elem, ElemType::Int(1));
            assert_eq!(predicate.ty().len, row.len);
        }
    }

    /// 🎯 048/110 — ⭐ THE REFERENCE'S OWN COMMENTED SET, REBUILT: `#set = affine_set<(d0)[s0] :
    /// (d0 + s0 * 8 - 64 >= 0, -d0 + 63 >= 0)>` for `{"mb": 1}` over `vector<64xf16>`.
    #[test]
    fn the_dynamic_set_is_the_one_the_comment_prints() {
        let mut vals = Values::default();
        let iv = vals.mint();
        let mut body = Vec::new();
        let row = Vector {
            len: 64,
            elem: ElemType::F16,
        };
        let predicate = dynamic_masking(
            &mut vals,
            &mut body,
            row,
            &[PrimaryDim::Mb],
            &[iv],
            DynamicMask {
                dim: PrimaryDim::Mb,
                offset: 1,
            },
        );

        let expected = IntegerSet {
            dims: 1,
            symbols: 1,
            constraints: vec![
                Constraint {
                    expr: AffineExpr::dim(0)
                        .plus(AffineExpr::sym(0).times(8))
                        .plus(AffineExpr::Const(-64)),
                    is_equality: false,
                },
                Constraint {
                    expr: AffineExpr::dim(0).times(-1).plus(AffineExpr::Const(63)),
                    is_equality: false,
                },
            ],
        };
        assert_eq!(
            body,
            vec![Op::VectorChain(vectorchain::Op::CreateAffineMaskSet {
                result: Val(1),
                mask_set: expected,
                // ⛔ THE PARAMETER IS THE LOOP'S INDUCTION VARIABLE, not a fresh value.
                mask_parameter: Some(iv),
                ty: Vector {
                    len: 64,
                    elem: ElemType::Int(1),
                },
            })]
        );
        assert_eq!(predicate.map(|p| p.val()), Some(Val(1)));
    }

    /// 🎯 048/110 — ⛔⛔ THE DIVIDE BINDS FIRST HERE, WHERE THE STATIC FORM DIVIDES THE PRODUCT.
    ///
    /// Over 60 lanes with offset 2: `(60 / 8) * 2 = 14`, not `(2 * 60) / 8 = 15`.
    #[test]
    fn the_dynamic_coefficient_divides_before_it_scales() {
        let mut vals = Values::default();
        let iv = vals.mint();
        let mut body = Vec::new();
        dynamic_masking(
            &mut vals,
            &mut body,
            Vector {
                len: 60,
                elem: ElemType::F16,
            },
            &[PrimaryDim::Mb],
            &[iv],
            DynamicMask {
                dim: PrimaryDim::Mb,
                offset: 2,
            },
        )
        .expect("the loop for `mb` is present");
        let [Op::VectorChain(vectorchain::Op::CreateAffineMaskSet { mask_set, .. })] = &body[..]
        else {
            unreachable!("one `create_affine_mask` with a set")
        };
        assert_eq!(
            mask_set.constraints[0].expr,
            AffineExpr::dim(0)
                .plus(AffineExpr::sym(0).times(14))
                .plus(AffineExpr::Const(-60))
        );
        assert_ne!(
            mask_set.constraints[0].expr,
            AffineExpr::dim(0)
                .plus(AffineExpr::sym(0).times(15))
                .plus(AffineExpr::Const(-60))
        );
    }

    /// 🎯 048/110 — ⛔ A DIMENSION NO EMITTED LOOP WALKS IS `None`, which is entry 028's answer and
    /// the reference's null `Operation *`.
    #[test]
    fn a_mask_dim_with_no_loop_emits_nothing() {
        let mut vals = Values::default();
        let iv = vals.mint();
        let mut body = Vec::new();
        assert_eq!(
            dynamic_masking(
                &mut vals,
                &mut body,
                Vector {
                    len: 64,
                    elem: ElemType::F16,
                },
                &[PrimaryDim::In],
                &[iv],
                DynamicMask {
                    dim: PrimaryDim::Mb,
                    offset: 1,
                },
            ),
            None
        );
        assert!(body.is_empty(), "the refusal emits no op");
    }

    /// 🎯 048/110 — ⭐⭐ THE CROSS-BRIDGE ROUND TRIP: the set this emits is the ONE dynamic set
    /// bridge 2 accepts, and the value it hands back is the loop iterator.
    ///
    /// [`get_mask_value_for_pt`] (entry 089/384) checks the symbol's coefficient against
    /// `num_lanes_in_slice` and the parameter's definition against the enclosing loop's bound and
    /// induction variable. ⛔ AND IT PINS `offset`: 2 fails the same reader that 1 passes.
    #[test]
    fn the_dynamic_set_survives_the_trip_through_bridge_two() {
        let row = Vector {
            len: 64,
            elem: ElemType::F16,
        };
        // `sentient.for %iv = .. to %bound` with `%sub = %bound - %iv` as the mask parameter —
        // bridge 2's shape for a dynamic PT mask.
        let for_op = sen::Op::Sentient(sentient::Op::For {
            iv: Val(0),
            bound: Val(1),
            carried: Vec::new(),
            dbg_name: None,
            body: Vec::new(),
        });
        let scope = vec![sen::Op::Arith(
            crate::islands::dataflow_ir::dialects::arith::Op::SubI(
                crate::islands::dataflow_ir::dialects::arith::IntBinary {
                    result: Val(2),
                    lhs: Val(1),
                    rhs: Val(0),
                    ty: crate::islands::dataflow_ir::ty::ScalarTy::Index,
                },
            ),
        )];
        let loops = [EnclosingLoop::of(&for_op).expect("a sentient.for")];

        for (offset, accepted) in [(1, true), (2, false)] {
            let mut vals = Values::default();
            // Mint `%0`, `%1` and `%2` so the parameter this bridge passes is the subtraction.
            let (_iv, _bound, parameter) = (vals.mint(), vals.mint(), vals.mint());
            let mut body = Vec::new();
            dynamic_masking(
                &mut vals,
                &mut body,
                row,
                &[PrimaryDim::Mb],
                &[parameter],
                DynamicMask {
                    dim: PrimaryDim::Mb,
                    offset,
                },
            )
            .expect("the loop for `mb` is present");
            let [Op::VectorChain(emitted)] = &body[..] else {
                unreachable!("one `create_affine_mask`")
            };

            let mut sen_values = Values::default();
            let recovered = get_mask_value_for_pt::<Dd2>(
                PtUnit,
                row,
                emitted,
                Definitions::from_innermost(&[&scope]),
                &loops,
                &mut sen_values,
            );
            if accepted {
                // ⭐ THE LOOP'S INDUCTION VARIABLE, which is what a dynamic mask lowers to.
                assert_eq!(recovered, Some(PtMaskValue::LoopIterator(Val(0))));
            } else {
                assert_eq!(
                    recovered, None,
                    "an offset of {offset} scales the coefficient off `num_lanes_in_slice`"
                );
            }
        }
    }

    /// 🎯 049/110 — ⭐ ONE STICK OF A REGULAR TENSOR ON THE LOWERING'S OWN COMPONENT, and ⛔ the
    /// discarded `is_result_integer` recovered from the element.
    #[test]
    fn the_compute_type_is_one_stick_and_carries_the_dropped_flag() {
        for (format, on, len, elem) in [
            (DataType::Sen169Fp16, GenericComp::Sfp, 64, ElemType::F16),
            (
                DataType::Sen143Fp8,
                GenericComp::Sfp,
                128,
                ElemType::F8E4M3Fn,
            ),
            (DataType::Bool, GenericComp::Pt, 64, ElemType::Int(1)),
        ] {
            let ty = type_from_compute_type(format, on).expect("a width that divides a stick");
            assert_eq!(ty, Vector { len, elem });
            // ⭐ IT IS EXACTLY THE `-1` ARM OF ENTRY 014.
            assert_eq!(
                ty,
                type_from_format(
                    format,
                    on,
                    TensorCategory::Regular,
                    VectorWidth::OneStick(
                        StickWidth::of(format, on, TensorCategory::Regular).expect("a stick width")
                    ),
                )
            );
            // ⛔ THE FLAG THE REFERENCE THREW AWAY.
            assert_eq!(is_integer(ty.elem), matches!(elem, ElemType::Int(_)));
        }
        // ⛔ AND THE ABORT INSIDE PROPAGATES: `SENINT24` on the PT is 24 bits, which does not divide
        // a 1024-bit stick.
        assert_eq!(
            type_from_compute_type(DataType::Senint24, GenericComp::Pt),
            None
        );
    }

    /// 🎯 094/110 — ⛔ BOTH OUTPUTS CONVERT THE SAME UNCONVERTED RESULT: `compute_result` is declared
    /// once and never reassigned, so the second conversion reads `%9` and not the first's answer.
    #[test]
    fn every_output_operand_converts_the_result_the_compute_bound() {
        let ty = Vector {
            len: 64,
            elem: ElemType::F16,
        };
        let handlers = operand_handlers();
        let ctx = OperandContext {
            name: "c0",
            comp: GenericComp::Sfp,
            ex_unit: GenericComp::Sfp,
            handlers: &handlers,
            result_ty: ty,
        };
        let (first, second) = (
            Latch::new(3).expect("latch 3"),
            Latch::new(4).expect("latch 4"),
        );
        let outputs = [
            (
                ComputeOutput::Latch(first),
                OutputFormat {
                    lds: Some(DataType::Senint8),
                    operand: DataType::Sen169Fp16,
                },
            ),
            (
                ComputeOutput::Latch(second),
                OutputFormat {
                    lds: None,
                    operand: DataType::Senint4,
                },
            ),
        ];

        let mut vals = Values::default();
        let mut ops = Vec::new();
        let stored = compute_output_operands(
            &mut vals,
            &mut ops,
            &ctx,
            &outputs,
            Computed::of(Val(9), ty),
        )
        .expect("two latches take two conversions");
        assert_eq!(
            ops.iter()
                .filter_map(|op| match op {
                    Op::Arith(arith::Op::Convert { input, ty, .. }) => Some((*input, *ty)),
                    _ => None,
                })
                .collect::<Vec<_>>(),
            vec![
                (
                    Val(9),
                    Vector {
                        len: 64,
                        elem: ElemType::Int(8)
                    }
                ),
                (
                    Val(9),
                    Vector {
                        len: 64,
                        elem: ElemType::Int(4)
                    }
                ),
            ]
        );
        assert_eq!(
            stored,
            vec![
                Stored::Latched {
                    latch: first,
                    data: Val(0),
                },
                Stored::Latched {
                    latch: second,
                    data: Val(1),
                },
            ]
        );
    }

    /// 🎯 095/110 — ⛔ ONE COMPARISON AND ONE SELECTION OVER THE SAME MASK, both carrying the node's
    /// name, and ⭐ `compare_le` is the FMIN arm of that ternary.
    #[test]
    fn the_pair_compares_under_the_mask_and_selects_under_it_again() {
        let ty = Vector {
            len: 64,
            elem: ElemType::F16,
        };
        let bool_ty = Vector {
            len: 64,
            elem: ElemType::Int(1),
        };
        let handlers = operand_handlers();
        let ctx = OperandContext {
            name: "fmin_0",
            comp: GenericComp::Sfp,
            ex_unit: GenericComp::Sfp,
            handlers: &handlers,
            result_ty: ty,
        };
        let format = OutputFormat {
            lds: None,
            operand: DataType::Sen169Fp16,
        };
        let latch = Latch::new(3).expect("latch 3");

        let mut vals = Values::default();
        let mut ops = Vec::new();
        let stored = fmin_or_fmax_operation::<Dd2>(
            &mut vals,
            &mut ops,
            &ctx,
            MinOrMax::Fmin,
            &[(ComputeInput::Zero, format), (ComputeInput::One, format)],
            DataType::Sen169Fp16,
            MaskValue::Live6,
            &[(ComputeOutput::Latch(latch), format)],
        )
        .expect("both operands are splats at the compute's own format");

        let mask = LaneMask::prefix_of(48, bool_ty);
        assert_eq!(
            ops[2..],
            [
                Op::VectorChain(vectorchain::Op::CreateAffineMask {
                    result: Val(2),
                    mask,
                }),
                Op::VectorChain(vectorchain::Op::ElementWiseCompare {
                    result: Val(3),
                    op1: Val(0),
                    op2: Val(1),
                    mask: Some(mask.binds(Val(2))),
                    compare_op: vectorchain::CompareOp::Le,
                    dbg_name: Some("fmin_0".to_owned()),
                    operand_ty: ty,
                    ty: bool_ty,
                }),
                Op::VectorChain(vectorchain::Op::ElementWiseSelection {
                    result: Val(4),
                    cond: mask.binds(Val(3)),
                    lhs: Val(0),
                    rhs: Val(1),
                    dbg_name: Some("fmin_0".to_owned()),
                    mask: Some(mask.binds(Val(2))),
                    ty,
                }),
            ]
        );
        // ⛔ AND A NON-STATE OUTPUT TAKES THE SELECTION, at `result_type` and so unconverted here.
        assert_eq!(
            stored,
            vec![Stored::Latched {
                latch,
                data: Val(4),
            }]
        );
    }

    // ─────────────────────────────── 086-087/110 ───────────────────────────────

    /// The two register-file handles every operand context carries, and no bound unit.
    fn operand_handlers() -> Handlers {
        Handlers {
            units: Vec::new(),
            own_lrf: Val(1),
            pt_xrf: Val(2),
        }
    }

    /// ⛔⛔ THE TAIL IS THE OPERAND, NOT A FORMALITY: `ONE` is a dense splat at the RESULT's type,
    /// and what the compute is handed is that splat CONVERTED to the operand's format — while an
    /// LXLU execution unit is the exemption that leaves the splat standing alone.
    #[test]
    fn one_is_a_dense_splat_and_the_conversion_after_it_is_the_operand() {
        let ty = Vector {
            len: 64,
            elem: ElemType::F16,
        };
        let handlers = operand_handlers();
        let context = |ex_unit| OperandContext {
            name: "c0",
            comp: GenericComp::Sfp,
            ex_unit,
            handlers: &handlers,
            result_ty: ty,
        };

        let mut vals = Values::default();
        let mut ops = Vec::new();
        let operand = compute_input_operand::<Dd2>(
            &mut vals,
            &mut ops,
            &context(GenericComp::Sfp),
            &ComputeInput::One,
            DataType::Senint8,
        )
        .expect("a float splat converts to an integer operand");
        let converted = Vector {
            len: 64,
            elem: ElemType::Int(8),
        };
        assert_eq!(
            ops,
            vec![
                Op::Arith(arith::Op::DenseConstant {
                    result: Val(0),
                    splat: 1,
                    ty,
                }),
                Op::Arith(arith::Op::Convert {
                    result: Val(1),
                    kind: arith::ConvertKind::FpToSi,
                    input: Val(0),
                    input_ty: ty,
                    ty: converted,
                }),
            ]
        );
        assert_eq!((operand.val(), operand.ty()), (Val(1), converted));

        // ⛔ THE EXEMPTION: the splat IS the operand, at the result's own width.
        let mut vals = Values::default();
        let mut ops = Vec::new();
        let operand = compute_input_operand::<Dd2>(
            &mut vals,
            &mut ops,
            &context(GenericComp::Lxlu),
            &ComputeInput::One,
            DataType::Senint8,
        )
        .expect("the LXLU converts nothing");
        assert_eq!(ops.len(), 1);
        assert_eq!((operand.val(), operand.ty()), (Val(0), ty));
    }

    /// ⛔⛔ THE OUTPUT'S CONVERSION IS AT THE **HEAD**: the latch records the CONVERTED value, and
    /// the format converted to is the labelled data structure's where it has one.
    #[test]
    fn the_latch_records_the_value_the_labelled_data_structure_s_format_produced() {
        let ty = Vector {
            len: 64,
            elem: ElemType::F16,
        };
        let handlers = operand_handlers();
        let context = |ex_unit| OperandContext {
            name: "c0",
            comp: GenericComp::Sfp,
            ex_unit,
            handlers: &handlers,
            result_ty: ty,
        };
        let latch = Latch::new(3).expect("latch 3");
        let format = OutputFormat {
            lds: Some(DataType::Senint8),
            operand: DataType::Sen169Fp16,
        };
        // ⭐ AND THE LABELLED FORMAT WINS OVER THE COMPUTE'S OWN.
        assert_eq!(format.get(), DataType::Senint8);

        let mut vals = Values::default();
        let mut ops = Vec::new();
        let stored = compute_output_operand(
            &mut vals,
            &mut ops,
            &context(GenericComp::Sfp),
            &ComputeOutput::Latch(latch),
            Computed::of(Val(9), ty),
            format,
        )
        .expect("a float result converts to the labelled integer format");
        assert_eq!(
            ops,
            vec![Op::Arith(arith::Op::Convert {
                result: Val(0),
                kind: arith::ConvertKind::FpToSi,
                input: Val(9),
                input_ty: ty,
                ty: Vector {
                    len: 64,
                    elem: ElemType::Int(8),
                },
            })]
        );
        assert_eq!(
            stored,
            Stored::Latched {
                latch,
                data: Val(0),
            }
        );

        // ⛔ THE SAME EXEMPTION, AND THE LATCH THEN HOLDS THE UNCONVERTED RESULT.
        let mut vals = Values::default();
        let mut ops = Vec::new();
        let stored = compute_output_operand(
            &mut vals,
            &mut ops,
            &context(GenericComp::Lxlu),
            &ComputeOutput::Latch(latch),
            Computed::of(Val(9), ty),
            format,
        )
        .expect("the LXLU converts nothing");
        assert!(ops.is_empty());
        assert_eq!(
            stored,
            Stored::Latched {
                latch,
                data: Val(9),
            }
        );
    }

    // ─────────────────────────────── 099-101/110 ───────────────────────────────

    /// 🎯 099/110 — ⛔ THE MAC RUNS AT THE ACCUMULATOR'S TYPE and `FNMS` NEGATES `inputs[0]` AT IT, so
    /// the emitted op states three widths: the negated 32, the wide 128 and its own 32.
    #[test]
    fn the_mac_negates_its_first_factor_at_the_accumulators_own_type() {
        let f16 = |len| Vector {
            len,
            elem: ElemType::F16,
        };
        let handlers = operand_handlers();
        let ctx = OperandContext {
            name: "fnms_0",
            comp: GenericComp::Sfp,
            ex_unit: GenericComp::Sfp,
            handlers: &handlers,
            result_ty: f16(64),
        };
        let format = |elements| MacInputFormat {
            lds: None,
            operand: DataType::Sen169Fp16,
            elements: Elements(elements),
        };
        let latch = Latch::new(3).expect("latch 3");

        let mut vals = Values::default();
        let mut ops = Vec::new();
        let stored = mac_operation::<Dd2>(
            &mut vals,
            &mut ops,
            &ctx,
            MacOp::Fnms,
            &[
                (ComputeInput::One, format(64)),
                (ComputeInput::One, format(128)),
                (ComputeInput::Zero, format(32)),
            ],
            ComputeMask::Static(MaskValue::Live8),
            &[(
                ComputeOutput::Latch(latch),
                OutputFormat {
                    lds: None,
                    operand: DataType::Sen169Fp16,
                },
            )],
        )
        .expect("three splats at the compute's own format");

        let mask = LaneMask::prefix_of(
            32,
            Vector {
                len: 32,
                elem: ElemType::Int(1),
            },
        );
        assert_eq!(
            ops[3..],
            [
                Op::VectorChain(vectorchain::Op::CreateAffineMask {
                    result: Val(3),
                    mask,
                }),
                Op::VectorChain(vectorchain::Op::Neg {
                    result: Val(4),
                    input: Val(0),
                    mask: None,
                    input_ty: f16(64),
                    ty: f16(32),
                }),
                Op::VectorChain(vectorchain::Op::MultiplyAccumulate {
                    result: Val(5),
                    a: Val(4),
                    b: Val(1),
                    acc: Val(2),
                    mask: Some(mask.binds(Val(3))),
                    dbg_name: Some("fnms_0".to_owned()),
                    reduction_map: AffineMap::unary(AffineExpr::dim(0).floordiv(1)),
                    a_ty: f16(32),
                    b_ty: f16(128),
                    ty: f16(32),
                }),
            ]
        );
        assert_eq!(
            stored,
            vec![Stored::Latched {
                latch,
                data: Val(5),
            }]
        );
    }

    /// 🎯 100/110 — ⛔ THE SELECTION'S CONDITION IS `inputs[0]`, an arriving operand and not a
    /// comparison's result, and every op is stated at the LAST format's type. ⚠️ A ONE-INPUT COMPUTE
    /// is the reference's `failure()`.
    #[test]
    fn the_selection_takes_its_condition_from_the_first_operand() {
        let ty = Vector {
            len: 64,
            elem: ElemType::F16,
        };
        let handlers = operand_handlers();
        let ctx = OperandContext {
            name: "select_0",
            comp: GenericComp::Sfp,
            ex_unit: GenericComp::Sfp,
            handlers: &handlers,
            result_ty: ty,
        };
        let format = OutputFormat {
            lds: None,
            operand: DataType::Sen169Fp16,
        };
        let latch = Latch::new(2).expect("latch 2");
        let inputs = [
            (ComputeInput::One, format),
            (ComputeInput::Zero, format),
            (ComputeInput::One, format),
        ];

        let mut vals = Values::default();
        let mut ops = Vec::new();
        let stored = binary_or_ternary_operation::<Dd2>(
            &mut vals,
            &mut ops,
            &ctx,
            &BinaryOrTernary::Select,
            &inputs,
            DataType::Sen169Fp16,
            MaskValue::Live8,
            &[(ComputeOutput::Latch(latch), format)],
        )
        .expect("three splats at the compute's own format");

        let mask = LaneMask::prefix_of(
            64,
            Vector {
                len: 64,
                elem: ElemType::Int(1),
            },
        );
        assert_eq!(
            ops[3..],
            [
                Op::VectorChain(vectorchain::Op::CreateAffineMask {
                    result: Val(3),
                    mask,
                }),
                Op::VectorChain(vectorchain::Op::ElementWiseSelection {
                    result: Val(4),
                    cond: Predicate::of_operand(Computed::of(Val(0), ty)),
                    lhs: Val(1),
                    rhs: Val(2),
                    dbg_name: Some("select_0".to_owned()),
                    mask: Some(mask.binds(Val(3))),
                    ty,
                }),
            ]
        );
        assert_eq!(
            stored,
            vec![Stored::Latched {
                latch,
                data: Val(4),
            }]
        );
        // ⛔ `!is_any_of(compute_op.inputs_.size(), 2, 3)`.
        assert!(
            binary_or_ternary_operation::<Dd2>(
                &mut vals,
                &mut ops,
                &ctx,
                &BinaryOrTernary::Fmul(MulMode::MulDiv2),
                &inputs[..1],
                DataType::Sen169Fp16,
                MaskValue::Live8,
                &[],
            )
            .is_none()
        );
    }

    /// 🎯 101/110 — ⛔ A SIGN-EXTENDING SPLAT READS ELEMENT 0 ONCE AND A `pad` ZERO IN EVERY OTHER
    /// LANE, and that constant is emitted BEFORE the shuffle. ⚠️ THE TABLE HAS NO OTHER ELEMENT.
    #[test]
    fn the_sign_extending_splat_pads_every_lane_but_the_first() {
        let ty = Vector {
            len: 64,
            elem: ElemType::F16,
        };
        let handlers = operand_handlers();
        let ctx = OperandContext {
            name: "splat_0",
            comp: GenericComp::Sfp,
            ex_unit: GenericComp::Sfp,
            handlers: &handlers,
            result_ty: ty,
        };
        let format = OutputFormat {
            lds: None,
            operand: DataType::Sen169Fp16,
        };
        let latch = Latch::new(1).expect("latch 1");

        let mut vals = Values::default();
        let mut ops = Vec::new();
        let stored = unary_operation::<Dd2>(
            &mut vals,
            &mut ops,
            &ctx,
            &UnaryOp::Splat {
                sign_extend: SignExtend::Yes,
                repetition: 8,
            },
            (&ComputeInput::One, format),
            MaskValue::Live8,
            &[(ComputeOutput::Latch(latch), format)],
        )
        .expect("a splat at the compute's own format");

        assert_eq!(
            ops[2..],
            [
                Op::Arith(arith::Op::Constant {
                    result: Val(2),
                    value: 0,
                }),
                Op::VectorChain(vectorchain::Op::Shuffle {
                    result: Val(3),
                    input: Val(0),
                    variable: Vec::new(),
                    pad: vec![vectorchain::ShuffleVariable {
                        val: Val(2),
                        ty: ScalarTy::Index,
                    }],
                    dbg_name: Some("splat_0".to_owned()),
                    indices: vec![0, -1, -1, -1, -1, -1, -1, -1],
                    repetition: 8,
                    input_ty: ty,
                    ty,
                }),
            ]
        );
        assert_eq!(
            stored,
            vec![Stored::Latched {
                latch,
                data: Val(3),
            }]
        );
        // ⛔ NEITHER `if` HAS AN `else` — a `bf16` splat leaves the reference's op null.
        assert_eq!(SignExtend::Yes.splat_indices(ElemType::Bf16), None);
    }

    // ─────────────────────────────── 103/110 ───────────────────────────────

    /// 🎯 103/110 — ⛔ ONE SCALED OPERAND PREFIXES THE WHOLE UNIT'S PRECISION, and a unit that is not
    /// the compute's execution unit lowers the statement to nothing at all.
    ///
    /// A bystander unit that emitted the MAC anyway would run the same multiply on every unit of the
    /// core; a scaled MAC whose precision stayed `fp8` names the wrong MAC opcode in the backend.
    #[test]
    fn a_scaled_mac_operand_prefixes_the_precision_and_a_bystander_emits_nothing() {
        let handlers = operand_handlers();
        let ctx = |comp| OperandContext {
            name: "fma8_0",
            comp,
            ex_unit: GenericComp::Sfp,
            handlers: &handlers,
            result_ty: Vector {
                len: 64,
                elem: ElemType::F16,
            },
        };
        let format = |elements, lds| MacInputFormat {
            lds,
            operand: DataType::Sen169Fp16,
            elements: Elements(elements),
        };
        let scaled = Some((DataType::Sen143Fp8, TensorCategory::Scaled));
        let inputs = [
            (ComputeInput::One, format(64, scaled)),
            (ComputeInput::One, format(128, None)),
            (ComputeInput::Zero, format(32, None)),
        ];
        let outputs = [(
            ComputeOutput::Latch(Latch::new(3).expect("latch 3")),
            OutputFormat {
                lds: None,
                operand: DataType::Sen169Fp16,
            },
        )];
        let mac = |mask| ComputeFamily::Mac {
            mac: MacOp::Fma8,
            inputs: &inputs,
            mask,
            outputs: &outputs,
        };

        let mut vals = Values::default();
        let mut ops = Vec::new();
        let computed = compute_operation::<Dd2>(
            &mut vals,
            &mut ops,
            &ctx(GenericComp::Sfp),
            mac(ComputeMask::Static(MaskValue::Live8)),
        )
        .expect("the SFP is the execution unit");
        assert_eq!(computed.precision, Some(dataflow::Precision::Mxfp8));
        assert!(!ops.is_empty());

        let mut bystander = Vec::new();
        assert_eq!(
            compute_operation::<Dd2>(
                &mut vals,
                &mut bystander,
                &ctx(GenericComp::Lxlu),
                mac(ComputeMask::Static(MaskValue::Live8)),
            ),
            None
        );
        assert!(bystander.is_empty());

        // ⛔ THE ONE `mx` SPELLING THE ISLAND GAINED FOR THIS ENTRY.
        assert_eq!(
            MacOp::Ima4.precision(TensorCategory::Scaled),
            dataflow::Precision::Mxint4
        );
    }
}
