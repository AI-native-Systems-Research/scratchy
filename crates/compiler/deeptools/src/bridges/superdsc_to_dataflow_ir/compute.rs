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
use super::control_flow::PrimaryDim;
use super::dsc_lowering::mlir_loop_from_loop_node;
use crate::arch::Elements;
use crate::generated::DataType;
use crate::islands::dataflow_ir::Values;
use crate::islands::dataflow_ir::dialects::vectorchain::{Computed, Predicate};
use crate::islands::dataflow_ir::dialects::{Op, Val, vectorchain};
use crate::islands::dataflow_ir::ty::{
    AffineExpr, Constraint, ElemType, GenericComp, IntegerSet, TensorCategory, Vector,
};

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
        result,
        input: bitstream,
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

// crustify:todo: e050_getReductionMapForMACOperation
// crustify:todo: e051_getSelectionMapForMACOperandFromL0
// crustify:todo: e052_constructPrecisionConversionOperation
// crustify:todo: e053_constructOpaqueOperation
// crustify:todo: e086_constructComputeInputOperandAndAddToList
// crustify:todo: e087_constructComputeOutputOperand
// crustify:todo: e094_constructComputeOutputOperands
// crustify:todo: e095_constructFMINorFMAXOperation
// crustify:todo: e099_constructMACOperation
// crustify:todo: e100_constructBinaryOrTernaryOperation
// crustify:todo: e101_constructUnaryOperation
// crustify:todo: e103_constructComputeOperation

#[cfg(test)]
mod unit_tests {
    use super::{
        DynamicMask, MaskValue, StickWidth, VectorWidth, dictionary_order, dynamic_masking,
        is_integer, single_val_custom_vector, static_mask_for_result, type_from_compute_type,
        type_from_format,
    };
    use crate::arch::{Dd2, Elements};
    use crate::bridges::dataflow_ir_to_sentient::vc_helper::{
        EnclosingLoop, MaskValue as PtMaskValue, PtUnit, get_mask_value_for_pt,
    };
    use crate::bridges::superdsc_to_dataflow_ir::control_flow::PrimaryDim;
    use crate::generated::{DataType, ParamKey, ParamValue};
    use crate::islands::dataflow_ir::Values;
    use crate::islands::dataflow_ir::dialects::vectorchain::{Computed, LaneMask};
    use crate::islands::dataflow_ir::dialects::{Op, Val, vectorchain};
    use crate::islands::dataflow_ir::ty::{
        AffineExpr, Constraint, ElemType, GenericComp, IntegerSet, TensorCategory, Vector,
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
        let out = single_val_custom_vector(&mut vals, &mut body, 7, ty);
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
}
