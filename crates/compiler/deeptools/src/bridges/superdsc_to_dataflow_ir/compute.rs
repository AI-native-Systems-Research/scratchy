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

use crate::arch::Elements;
use crate::generated::DataType;
use crate::islands::dataflow_ir::Values;
use crate::islands::dataflow_ir::dialects::vectorchain::Computed;
use crate::islands::dataflow_ir::dialects::{Op, vectorchain};
use crate::islands::dataflow_ir::ty::{ElemType, GenericComp, TensorCategory, Vector};

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

// crustify:todo: e047_getStaticContinuousMaskValue
// crustify:todo: e048_constructDynamicMasking
// crustify:todo: e049_getTypeBasedOnComputeType
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
        StickWidth, VectorWidth, dictionary_order, is_integer, single_val_custom_vector,
        type_from_format,
    };
    use crate::arch::Elements;
    use crate::generated::{DataType, ParamKey, ParamValue};
    use crate::islands::dataflow_ir::Values;
    use crate::islands::dataflow_ir::dialects::vectorchain::Computed;
    use crate::islands::dataflow_ir::dialects::{Op, Val, vectorchain};
    use crate::islands::dataflow_ir::ty::{ElemType, GenericComp, TensorCategory, Vector};

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
}
