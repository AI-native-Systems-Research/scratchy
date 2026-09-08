//! DATAFLOWIR CONSTRUCTION — building the island's types and attributes.
//!
//! 6 units. Every citation resolves against the authority tree
//! `/Users/nickm/git/deeptools-src` at revision `a0d29abbed`.
//!
//! | unit | level | LoC | authority |
//! |---|---|---|---|
//! | `e011_constructLogicalMemoryViewOp` | 0 | 22 | `dsc-based-utils/DSC2ToDataflowIR/DataflowIRConstructionUtils.hpp:33` |
//! | `e012_convertPrecisionToType` | 0 | 20 | `dsc-based-utils/DSC2ToDataflowIR/DataflowIRConstructionUtils.hpp:130` |
//! | `e013_convertTypeToString` | 0 | 19 | `dsc-based-utils/DSC2ToDataflowIR/DataflowIRConstructionUtils.hpp:151` |
//! | `e045_getReductionMapForMACOperation` | 1 | 40 | `dsc-based-utils/DSC2ToDataflowIR/DataflowIRConstructionUtils.hpp:61` |
//! | `e046_getStaticContinuousMaskValue` | 1 | 24 | `dsc-based-utils/DSC2ToDataflowIR/DataflowIRConstructionUtils.hpp:105` |
//! | `e085_initializeUnit` | 3 | 22 | `dsc-based-utils/DSC2ToDataflowIR/DataflowIRConstructionUtils.hpp:171` |

use crate::islands::dataflow_ir::Values;
use crate::islands::dataflow_ir::dialects::{Op, Val, dataflow};
use crate::islands::dataflow_ir::ty::{AffineMap, ElemType, MemRef};

// ⛔ ONE `crustify:todo:` PER SCHEDULED UNIT. Replace each with the ported function
// carrying `/// Replaces: eNNN_name`. A surviving TODO is open work.

/// Replaces: e011_constructLogicalMemoryViewOp
///
/// A ONE-ELEMENT VIEW over a storage unit at `start` — `DataflowIRConstructionUtils.hpp:33`.
///
/// ⛔ THE `layout_expr` IT BUILDS IS DEAD. Six lines assemble `1 * d0 + 0` and then the op is
/// created with `AffineMap::getMultiDimIdentityMap(1, ..)` instead, so the emitted map is the plain
/// identity — reading those lines as the layout would put a stride on a view that has none.
///
/// ⛔ AND THE EXTENT IS LITERALLY `{1}`: the scalar this view exists to address is one element, not
/// the region behind it.
pub fn logical_memory_view(
    vals: &mut Values,
    into: &mut Vec<Op>,
    from: Val,
    start: Val,
    elem: ElemType,
) -> Val {
    let result = vals.mint();
    into.push(Op::Dataflow(dataflow::Op::GetLogicalMemoryView {
        result,
        from,
        start,
        layout: AffineMap::identity(1),
        ty: MemRef {
            shape: vec![1],
            elem,
        },
    }));
    result
}

/// THE EIGHT PRECISION NAMES `convertPrecisionToType` ANSWERS FOR.
///
/// ⛔ NOT [`dataflow::Precision`], which is the `program_unit` attribute's vocabulary: that one has
/// `fp8`, `fp4`, `mxfp8` and `fp80` and no `int16` at all. These are the spellings this one
/// function's `llvm_unreachable` bounds, so they are their own closed set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrecisionName {
    /// `fp16`.
    Fp16,
    /// `bf16`.
    Bf16,
    /// `fp32`.
    Fp32,
    /// `f8E4M3FN`.
    F8E4M3Fn,
    /// `f8E5M2`.
    F8E5M2,
    /// `int16`.
    Int16,
    /// `int8`.
    Int8,
    /// `int4`.
    Int4,
}

/// Replaces: e012_convertPrecisionToType
///
/// THE ELEMENT TYPE A PRECISION NAME MEANS — `DataflowIRConstructionUtils.hpp:130`.
///
/// ⛔ TOTAL, BECAUSE THE INPUT IS A TYPE. The reference falls off its `else if` chain into
/// `llvm_unreachable("unknown type string")`; [`PrecisionName`] admits exactly the eight it answers,
/// so the unreachable arm has no input to reach it.
#[must_use]
pub const fn precision_to_type(name: PrecisionName) -> ElemType {
    match name {
        PrecisionName::Fp16 => ElemType::F16,
        PrecisionName::Bf16 => ElemType::Bf16,
        PrecisionName::Fp32 => ElemType::F32,
        PrecisionName::F8E4M3Fn => ElemType::F8E4M3Fn,
        PrecisionName::F8E5M2 => ElemType::F8E5M2,
        PrecisionName::Int16 => ElemType::Int(16),
        PrecisionName::Int8 => ElemType::Int(8),
        PrecisionName::Int4 => ElemType::Int(4),
    }
}

/// WHAT `convertTypeToString` WRITES — and it is not [`PrecisionName`].
///
/// ⛔⛔ THE TWO FUNCTIONS ARE NOT INVERSES. The integers come back as `i16`/`i8`/`i4` where
/// `convertPrecisionToType` reads `int16`/`int8`/`int4`, so feeding this output back into that input
/// hits its `llvm_unreachable`. Separate types is that fact made unwritable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TypeName {
    /// `fp16`.
    Fp16,
    /// `bf16`.
    Bf16,
    /// `fp32`.
    Fp32,
    /// `f8E4M3FN`.
    F8E4M3Fn,
    /// `f8E5M2`.
    F8E5M2,
    /// `i16`.
    I16,
    /// `i8`.
    I8,
    /// `i4`.
    I4,
}

impl TypeName {
    /// THE STRING THE REFERENCE RETURNS.
    #[must_use]
    pub const fn spelling(self) -> &'static str {
        match self {
            Self::Fp16 => "fp16",
            Self::Bf16 => "bf16",
            Self::Fp32 => "fp32",
            Self::F8E4M3Fn => "f8E4M3FN",
            Self::F8E5M2 => "f8E5M2",
            Self::I16 => "i16",
            Self::I8 => "i8",
            Self::I4 => "i4",
        }
    }
}

/// Replaces: e013_convertTypeToString
///
/// THE NAME AN ELEMENT TYPE IS WRITTEN AS — `DataflowIRConstructionUtils.hpp:151`.
///
/// ⛔ `None` IS THE `llvm_unreachable("unknown type")`, and it is reachable: the reference compares
/// against eight types only, so `f8E8M0FNU`, `f4E2M1FN`, an MX element and any other integer width
/// abort there. It is the TYPE that has to say so — this crate cannot refuse at run time.
#[must_use]
pub const fn type_to_name(elem: ElemType) -> Option<TypeName> {
    match elem {
        ElemType::F16 => Some(TypeName::Fp16),
        ElemType::Bf16 => Some(TypeName::Bf16),
        ElemType::F32 => Some(TypeName::Fp32),
        ElemType::F8E4M3Fn => Some(TypeName::F8E4M3Fn),
        ElemType::F8E5M2 => Some(TypeName::F8E5M2),
        ElemType::Int(16) => Some(TypeName::I16),
        ElemType::Int(8) => Some(TypeName::I8),
        ElemType::Int(4) => Some(TypeName::I4),
        ElemType::Int(_) | ElemType::F8E8M0Fnu | ElemType::F4E2M1Fn | ElemType::MxFloat(_) => None,
    }
}

// crustify:todo: e045_getReductionMapForMACOperation
// crustify:todo: e046_getStaticContinuousMaskValue
// crustify:todo: e085_initializeUnit

#[cfg(test)]
mod unit_tests {
    use super::{PrecisionName, TypeName, logical_memory_view, precision_to_type, type_to_name};
    use crate::islands::dataflow_ir::Values;
    use crate::islands::dataflow_ir::dialects::{Op, Val, dataflow};
    use crate::islands::dataflow_ir::ty::{AffineMap, ElemType, MemRef};

    /// ⛔ THE IDENTITY MAP AND A SINGLE-ELEMENT MEMREF — the built `1 * d0 + 0` never reaches the op.
    #[test]
    fn the_view_is_one_element_under_the_identity() {
        let mut vals = Values::default();
        let (from, start) = (vals.mint(), vals.mint());
        let mut body = Vec::new();
        let view = logical_memory_view(&mut vals, &mut body, from, start, ElemType::F16);
        assert_eq!(
            body,
            vec![Op::Dataflow(dataflow::Op::GetLogicalMemoryView {
                result: view,
                from,
                start,
                layout: AffineMap::identity(1),
                ty: MemRef {
                    shape: vec![1],
                    elem: ElemType::F16,
                },
            })]
        );
        assert_eq!(view, Val(2));
    }

    /// ⛔ AND IT IS NOT A ROUND TRIP: `int16` in, `i16` out.
    #[test]
    fn the_eight_names_and_the_asymmetry_between_them() {
        for (name, elem, back) in [
            (PrecisionName::Fp16, ElemType::F16, TypeName::Fp16),
            (PrecisionName::Bf16, ElemType::Bf16, TypeName::Bf16),
            (PrecisionName::Fp32, ElemType::F32, TypeName::Fp32),
            (
                PrecisionName::F8E4M3Fn,
                ElemType::F8E4M3Fn,
                TypeName::F8E4M3Fn,
            ),
            (PrecisionName::F8E5M2, ElemType::F8E5M2, TypeName::F8E5M2),
            (PrecisionName::Int16, ElemType::Int(16), TypeName::I16),
            (PrecisionName::Int8, ElemType::Int(8), TypeName::I8),
            (PrecisionName::Int4, ElemType::Int(4), TypeName::I4),
        ] {
            assert_eq!(precision_to_type(name), elem);
            assert_eq!(type_to_name(elem), Some(back));
        }
        assert_eq!(
            type_to_name(ElemType::Int(16)).map(TypeName::spelling),
            Some("i16")
        );
        // ⛔ THE ARMS THE REFERENCE ABORTS ON.
        assert_eq!(type_to_name(ElemType::F8E8M0Fnu), None);
        assert_eq!(type_to_name(ElemType::MxFloat(8)), None);
    }
}
