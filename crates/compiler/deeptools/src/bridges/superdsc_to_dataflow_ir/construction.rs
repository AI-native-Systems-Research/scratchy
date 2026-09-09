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
use crate::islands::dataflow_ir::dialects::vectorchain::{LaneMask, Predicate};
use crate::islands::dataflow_ir::dialects::{Op, Val, dataflow, vectorchain};
use crate::islands::dataflow_ir::ty::{AffineExpr, AffineMap, ElemType, MemRef, Vector};
use crate::units::{Core, Corelet, DfirUnit, NumFolds, Residency};

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

/// THE SIX MAC OPERATIONS THIS REDUCTION MAP ANSWERS FOR — the `is_any_of` and the three `else if`s
/// of `DataflowIRConstructionUtils.hpp:66-94`.
///
/// ⛔ NOT `ComputeOpType`, WHICH HAS ~50 MEMBERS (`dsc/dscdefn.h:134-180`). The reference falls off
/// its chain into `DT_ERROR("Unknown MAC operation encountered")` (`:95`), which THROWS
/// (`util/dt_exception.hpp:110-121`) — so the `return LogicalResult::failure()` on the next line is
/// DEAD CODE. Six variants make the throw unreachable and the dead return unwritable.
///
/// ⛔⛔ AND THIS IS NOT ENTRY 050'S VOCABULARY EVEN THOUGH THE C++ NAMES MATCH.
/// `SNComputeLowering::getReductionMapForMACOperation` (`V3/SNComputeLowering.cpp:126`) is
/// ARCH-DEPENDENT and admits a seventh operation this one aborts on; a shared type would offer that
/// arm here, where the reference throws.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MacOp {
    /// `FMA16`.
    Fma16,
    /// `FNMS` — the negated multiply-subtract, which reduces exactly as `FMA16` does.
    Fnms,
    /// `FMA32`.
    Fma32,
    /// `FMA8`.
    Fma8,
    /// `IMA8`.
    Ima8,
    /// `IMA4`.
    Ima4,
}

/// Replaces: e045_getReductionMapForMACOperation
///
/// WHICH ACCUMULATOR LANE EACH MULTIPLIER LANE REDUCES INTO — `DataflowIRConstructionUtils.hpp:61`.
///
/// ⭐ THE MAP IS THE PACKING FACTOR. At 16 and 32 bits one lane feeds one accumulator; at 8 bits two
/// share one (`floordiv 2`); the integer forms wrap at 128 FIRST because a 128-lane group is one
/// XRF pass. The reference's own comments give the examples — `0, 1 -> 0` for `FMA8` and
/// `0, 1, 128, 129 -> 0` for `IMA8` (`:71`, `:77`, `:86`).
///
/// ⭐ THE `IMA8` ARM IS IBM'S OWN `#map5`: `affine_map<(d0) -> ((d0 mod 128) floordiv 2)>`, already
/// cited by this island's module doc — so the shape is pinned by a printed golden, not inferred.
///
/// ⛔⛔ `FMA32` IS THE **IDENTITY** HERE AND `d0 floordiv 1` IN ENTRY 050
/// (`V3/SNComputeLowering.cpp:138-144`). The two compute the same function and are DIFFERENT
/// [`AffineMap`]s: [`AffineMap::is_identity`] is true for one and false for the other, because the
/// other's single result is a `FloorDiv` node rather than a `Dim`. A consumer that tests for the
/// identity map sees them apart, which is why they are two entries and not one.
#[must_use]
pub fn reduction_map_for_mac(op: MacOp) -> AffineMap {
    match op {
        // `builder.getDimIdentityMap()` — `(d0) -> (d0)` (`:69`).
        MacOp::Fma16 | MacOp::Fnms | MacOp::Fma32 => AffineMap::identity(1),
        MacOp::Fma8 => AffineMap::unary(AffineExpr::dim(0).floordiv(2)),
        MacOp::Ima8 => AffineMap::unary(AffineExpr::dim(0).modulo(128).floordiv(2)),
        MacOp::Ima4 => AffineMap::unary(AffineExpr::dim(0).modulo(128).floordiv(4)),
    }
}

/// A MASK VALUE THE EIGHT-ENTRY TABLE ADMITS, named for how many of the stick's eight slices stay
/// LIVE — `{{255, 0}, {127, 1}, {63, 2}, {31, 3}, {15, 4}, {7, 5}, {3, 6}, {1, 7}}`
/// (`DataflowIRConstructionUtils.hpp:108-109`, and the same eight rows again at
/// `V3/SNComputeLowering.cpp:36-37`).
///
/// ⛔⛔ THE KEY IS A LIVE-SLICE BITMASK AND THE VALUE IS THE MASKED-SLICE COUNT. 255 is
/// `0b1111_1111` and maps to 0; 63 is `0b0011_1111`, six live slices, and maps to 2. So the table is
/// `8 - live` — and only the eight CONTINUOUS prefixes are keys. `0b1011_1111` is not in it, and the
/// reference refuses it (`:110-111`) rather than masking a hole, which is why this is an eight-armed
/// enum and not a `u8`.
///
/// ⛔ THE DEFAULT MASKS NOTHING. `int mask_value = 255` (`:107`) is [`MaskValue::LIVE_ALL`], and it
/// gives an EMPTY constraint region — which is exactly how the vendor spells "no mask": see
/// [`LaneMask::as_set`], whose golden for it is `{value = 0 : si64}`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaskValue {
    /// `255` — all eight slices live, nothing masked.
    Live8,
    /// `127` — the top slice masked.
    Live7,
    /// `63`.
    Live6,
    /// `31`.
    Live5,
    /// `15`.
    Live4,
    /// `7`.
    Live3,
    /// `3`.
    Live2,
    /// `1` — one slice live, seven masked.
    Live1,
}

impl MaskValue {
    /// The reference's default argument, `mask_value = 255` (`:107`).
    pub const LIVE_ALL: MaskValue = MaskValue::Live8;

    /// THE ROW WITH THIS KEY, or `None` where `static_mask_value.count(..) < 1` throws.
    #[must_use]
    pub const fn of(bits: i64) -> Option<MaskValue> {
        match bits {
            255 => Some(MaskValue::Live8),
            127 => Some(MaskValue::Live7),
            63 => Some(MaskValue::Live6),
            31 => Some(MaskValue::Live5),
            15 => Some(MaskValue::Live4),
            7 => Some(MaskValue::Live3),
            3 => Some(MaskValue::Live2),
            1 => Some(MaskValue::Live1),
            _ => None,
        }
    }

    /// The key it was found under — the bitmask the DSC carries.
    #[must_use]
    pub const fn bits(self) -> i64 {
        match self {
            MaskValue::Live8 => 255,
            MaskValue::Live7 => 127,
            MaskValue::Live6 => 63,
            MaskValue::Live5 => 31,
            MaskValue::Live4 => 15,
            MaskValue::Live3 => 7,
            MaskValue::Live2 => 3,
            MaskValue::Live1 => 1,
        }
    }

    /// The table's VALUE — `static_mask_value[mask_value]`, how many slices are masked off.
    #[must_use]
    pub const fn slices_masked(self) -> u64 {
        match self {
            MaskValue::Live8 => 0,
            MaskValue::Live7 => 1,
            MaskValue::Live6 => 2,
            MaskValue::Live5 => 3,
            MaskValue::Live4 => 4,
            MaskValue::Live3 => 5,
            MaskValue::Live2 => 6,
            MaskValue::Live1 => 7,
        }
    }
}

/// THE LIVE PREFIX A MASK VALUE LEAVES OVER `lanes` LANES — the arithmetic entries 046 and 047 share.
///
/// ⛔⛔ `vector_dim - k * vector_dim / 8` IS A LIVE LANE COUNT, and reading it as anything else
/// INVERTS THE MASK. The reference builds `d0 - vector_dim + k * vector_dim / 8 >= 0` paired with
/// `-d0 + vector_dim - 1 >= 0` (`:117-119`), and the bridge-2 reader states the rule that pair
/// obeys: the affine set describes which lanes are **MASKED**. So the lower bound is the FIRST
/// MASKED lane, and the live lanes are the ones below it.
///
/// ⭐⭐ ONE DERIVATION, PINNED FROM BOTH ENDS. [`LaneMask::as_set`] writes that very set from a live
/// prefix, so this function's whole content is the count — and the two goldens it cites are this
/// table's own rows: `k = 2` ([`MaskValue::Live6`]) over 64 lanes gives 48 live, which is IBM's
/// `#set2` lowering to `{value = 2 : si64}`, and `k = 0` (the default 255) gives 64 of 64, the empty
/// span that lowers to `{value = 0 : si64}`.
///
/// ⛔ THE MULTIPLY BINDS BEFORE THE DIVIDE. `k * vector_dim / 8` is `(k * vector_dim) / 8` in C++,
/// which differs from `k * (vector_dim / 8)` whenever 8 does not divide the lane count — and the
/// DYNAMIC form parenthesises it the OTHER way, `symbol * (vector_dim / 8) * mask_offset`
/// (`V3/SNComputeLowering.cpp:95`). The two are not one expression, so they are not one function.
///
/// ⭐ IT CANNOT UNDERFLOW: `k <= 7`, so `(k * lanes) / 8 < lanes` for every lane count. The
/// saturation states that in a form that cannot panic instead of asserting it.
#[must_use]
pub fn continuous_mask_prefix(lanes: u64, mask: MaskValue) -> LaneMask {
    let masked = (mask.slices_masked() * lanes) / 8;
    LaneMask::prefix_of(
        lanes.saturating_sub(masked),
        Vector {
            len: lanes,
            elem: ElemType::Int(1),
        },
    )
}

/// Replaces: e046_getStaticContinuousMaskValue
///
/// A `vectorchain.create_affine_mask` FOR A STATIC MASK VALUE — `DataflowIRConstructionUtils.hpp:105`.
///
/// ⭐ THE WHOLE OPERATION IS [`continuous_mask_prefix`] PLUS ONE OP, and it returns a [`Predicate`]
/// rather than a [`Val`] so the mask's type at its use is the value written at its definition.
///
/// ⛔⛔ DELIBERATE DIVERGENCE: THE `arith.constant 0 : index` AT `:121-123` IS DROPPED. `zero` is
/// built and never referenced — the op below it is created with `nullptr` for the mask parameter
/// (`:126`), so nothing consumes the constant. Emitting it would put a dangling non-compute op in
/// the region, and *"Dangling non-compute op has no use"* is a real dbo-opt refusal this repo has
/// already collected. The reference's own sibling omits it (`V3/SNComputeLowering.cpp:50-53`),
/// which is the same function without the dead line.
///
/// ⛔ AND THE MASK PARAMETER IS ABSENT, NOT ZERO. `nullptr` at `:126` is *no operand*; the printed
/// op has an empty `($mask_parameter^)?` — a `%c0` operand would be a different op.
pub fn static_continuous_mask(
    vals: &mut Values,
    into: &mut Vec<Op>,
    lanes: u64,
    mask: MaskValue,
) -> Predicate {
    let prefix = continuous_mask_prefix(lanes, mask);
    let result = vals.mint();
    into.push(Op::VectorChain(vectorchain::Op::CreateAffineMask {
        result,
        mask: prefix,
    }));
    prefix.binds(result)
}

/// Replaces: e085_initializeUnit
///
/// **085/110** `DataflowIRConstructionUtils.hpp:171` — the `get_unit`/`program_unit` pair for one
/// `(core, corelet, unit)`, and NOTHING ELSE: no `component_to_handler_`, no neighbour preamble.
/// The unit's own handle is handed back as the `program_unit`'s single-entry unit list.
///
/// ⛔ [`NumFolds::ONE`] IS THE TYPE, NOT A VALUE READ: this one writes the literal
/// `getI32IntegerAttr(1)` (`:184`) — *"this initalization is for only non-uniform and non-fold
/// mode"* — where its `DSC2ToDataflowIRUtils.hpp:624` namesake asserts the field equals it.
#[must_use]
pub fn initialize_unit(
    vals: &mut Values,
    into: &mut Vec<Op>,
    core: Core,
    corelet: Corelet,
    comp: DfirUnit,
) -> Val {
    let handle = vals.mint();
    into.push(Op::Dataflow(dataflow::Op::GetUnit {
        result: handle,
        residency: Residency::Corelet { core, corelet },
        unit: comp,
        num_folds: Some(NumFolds::ONE),
    }));
    into.push(Op::Dataflow(dataflow::Op::ProgramUnit {
        units: vec![handle],
        iter_arg: None,
        precision: None,
        body: Vec::new(),
    }));
    handle
}

#[cfg(test)]
mod unit_tests {
    use super::{
        MacOp, MaskValue, PrecisionName, TypeName, continuous_mask_prefix, initialize_unit,
        logical_memory_view, precision_to_type, reduction_map_for_mac, static_continuous_mask,
        type_to_name,
    };
    use crate::arch::Dd2;
    use crate::units::{Core, Corelet, DfirUnit, NumFolds, Residency};
    use crate::bridges::dataflow_ir_to_sentient::vc_helper::{
        MaskValue as PtMaskValue, PtUnit, get_mask_value_for_pt,
    };
    use crate::bridges::dataflow_ir_to_sentient::vc_loop_mask_tree::MaskedColumns;
    use crate::islands::dataflow_ir::Values;
    use crate::islands::dataflow_ir::dialects::vectorchain::LaneMask;
    use crate::islands::dataflow_ir::dialects::{Op, Val, dataflow, vectorchain};
    use crate::islands::dataflow_ir::ty::{
        AffineExpr, AffineMap, Constraint, ElemType, IntegerSet, MemRef, Vector,
    };
    use crate::islands::sentient::dialects::Definitions;

    /// 🎯 085/110 — ⛔ AN **EMPTY** REGION, WHERE ITS NAMESAKE FILLS ONE. This pair is the whole
    /// operation: no `component_to_handler_`, no neighbour preamble, and the handle it returns is the
    /// `program_unit`'s single unit.
    #[test]
    fn the_construction_pair_is_a_get_unit_and_an_empty_program_unit() {
        let mut vals = Values::default();
        let mut ops = Vec::new();
        let core = Core::checked(1).expect("core 1");
        let corelet = Corelet::checked(0).expect("corelet 0");
        let handle = initialize_unit(&mut vals, &mut ops, core, corelet, DfirUnit::Pe);
        assert_eq!(
            ops,
            vec![
                Op::Dataflow(dataflow::Op::GetUnit {
                    result: handle,
                    residency: Residency::Corelet { core, corelet },
                    unit: DfirUnit::Pe,
                    num_folds: Some(NumFolds::ONE),
                }),
                Op::Dataflow(dataflow::Op::ProgramUnit {
                    units: vec![handle],
                    iter_arg: None,
                    precision: None,
                    body: Vec::new(),
                }),
            ]
        );
    }

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
    /// 🎯 045/110 — THE PACKING FACTOR, ARM BY ARM, and `0, 1, 128, 129 -> 0` evaluated rather than
    /// asserted about.
    #[test]
    fn the_reduction_map_is_the_packing_factor() {
        // `(d0) -> (d0)` for all three 16/32-bit forms.
        for op in [MacOp::Fma16, MacOp::Fnms, MacOp::Fma32] {
            assert_eq!(reduction_map_for_mac(op), AffineMap::identity(1));
            assert!(reduction_map_for_mac(op).is_identity());
        }
        assert_eq!(
            reduction_map_for_mac(MacOp::Fma8),
            AffineMap::unary(AffineExpr::dim(0).floordiv(2))
        );
        assert_eq!(
            reduction_map_for_mac(MacOp::Ima8),
            AffineMap::unary(AffineExpr::dim(0).modulo(128).floordiv(2))
        );
        assert_eq!(
            reduction_map_for_mac(MacOp::Ima4),
            AffineMap::unary(AffineExpr::dim(0).modulo(128).floordiv(4))
        );
        // ⛔ THE COMMENTS' OWN EXAMPLES: `0, 1 -> 0` and `0, 1, 128, 129 -> 0`.
        let at = |op, i: i64| {
            let map = reduction_map_for_mac(op);
            assert_eq!(map.dims, 1);
            assert_eq!(map.syms, 0);
            match &map.results[..] {
                [AffineExpr::FloorDiv(lhs, rhs)] => {
                    let div = match **rhs {
                        AffineExpr::Const(k) => k,
                        _ => unreachable!("the divisor is a constant"),
                    };
                    let inner = match &**lhs {
                        AffineExpr::Mod(m, n) => match (&**m, &**n) {
                            (AffineExpr::Dim(0), AffineExpr::Const(k)) => i.rem_euclid(*k),
                            _ => unreachable!("the wrap is `d0 mod k`"),
                        },
                        AffineExpr::Dim(0) => i,
                        _ => unreachable!("the numerator is `d0` or `d0 mod k`"),
                    };
                    inner.div_euclid(div)
                }
                _ => unreachable!("every non-identity arm is one floordiv"),
            }
        };
        assert_eq!([at(MacOp::Fma8, 0), at(MacOp::Fma8, 1)], [0, 0]);
        assert_eq!(at(MacOp::Fma8, 2), 1);
        for i in [0, 1, 128, 129] {
            assert_eq!(at(MacOp::Ima8, i), 0);
        }
        for i in [0, 1, 2, 3, 128, 129, 130, 131] {
            assert_eq!(at(MacOp::Ima4, i), 0);
        }
    }

    /// 🎯 045/110 — ⛔ AND FMA32 HERE IS NOT ENTRY 050'S `d0 floordiv 1`, even though both name
    /// `FMA32` and both compute `i -> i`.
    #[test]
    fn the_identity_and_floordiv_one_are_different_maps() {
        let here = reduction_map_for_mac(MacOp::Fma32);
        let entry_050 = AffineMap::unary(AffineExpr::dim(0).floordiv(1));
        assert_ne!(here, entry_050);
        assert!(here.is_identity());
        assert!(!entry_050.is_identity());
    }

    /// 🎯 046/110 — ⛔ THE TABLE IS `8 - live`, AND ONLY CONTINUOUS PREFIXES ARE KEYS.
    #[test]
    fn the_eight_rows_and_the_hole_that_is_not_one() {
        for (bits, masked) in [
            (255, 0),
            (127, 1),
            (63, 2),
            (31, 3),
            (15, 4),
            (7, 5),
            (3, 6),
            (1, 7),
        ] {
            let value = MaskValue::of(bits).expect("a table key");
            assert_eq!(value.slices_masked(), masked);
            assert_eq!(value.bits(), bits);
            // The key really is a live-slice bitmask: `8 - popcount`.
            assert_eq!(
                u64::from(8 - u32::try_from(bits).unwrap().count_ones()),
                masked
            );
        }
        // ⛔ `0b1011_1111` MASKS A HOLE and is not in the table — the reference throws on it.
        assert_eq!(MaskValue::of(0b1011_1111), None);
        assert_eq!(MaskValue::of(0), None);
        assert_eq!(MaskValue::of(-1), None);
        assert_eq!(MaskValue::LIVE_ALL.bits(), 255);
        assert_eq!(MaskValue::LIVE_ALL.slices_masked(), 0);
    }

    /// 🎯 046/110 — ⭐⭐ THE SET THE PREFIX ELIDES IS THE REFERENCE'S, DERIVED FROM BOTH ENDS: this
    /// rebuilds `d0 - vector_dim + k * vector_dim / 8 >= 0` from the C++ text and compares it to what
    /// [`LaneMask::as_set`] writes out of the live count.
    #[test]
    fn the_prefix_and_the_reference_set_agree() {
        let reference = |lanes: i64, k: i64| IntegerSet {
            dims: 1,
            symbols: 0,
            constraints: vec![
                Constraint {
                    // `id - vector_dim + k * vector_dim / 8`, folded to one constant.
                    expr: AffineExpr::dim(0).plus(AffineExpr::Const(-lanes + (k * lanes) / 8)),
                    is_equality: false,
                },
                Constraint {
                    // `-id + vector_dim - 1`.
                    expr: AffineExpr::dim(0)
                        .times(-1)
                        .plus(AffineExpr::Const(lanes - 1)),
                    is_equality: false,
                },
            ],
        };
        for lanes in [64_u64, 128, 1024, 60] {
            for bits in [255, 127, 63, 31, 15, 7, 3, 1] {
                let mask = MaskValue::of(bits).expect("a table key");
                let prefix = continuous_mask_prefix(lanes, mask);
                assert_eq!(prefix.ty().len, lanes);
                assert_eq!(prefix.ty().elem, ElemType::Int(1));
                // ⛔ THE SET'S CONSTANT IS THE TABLE **VALUE**, NOT THE KEY: `k` is
                // `static_mask_value[mask_value]`, and passing 255 where 0 belongs puts
                // `(255 * 64) / 8` into the bound.
                let k = i64::try_from(mask.slices_masked()).unwrap();
                assert_eq!(
                    prefix.as_set(),
                    Some(reference(i64::try_from(lanes).unwrap(), k))
                );
            }
        }
        // ⭐ IBM'S `#set2`: 48 live of 64 is `k = 2`, and it lowers to `{value = 2 : si64}`.
        assert_eq!(continuous_mask_prefix(64, MaskValue::Live6).live(), 48);
        // ⭐ AND THE DEFAULT IS THE EMPTY SPAN: 64 of 64, `{value = 0 : si64}`.
        assert_eq!(continuous_mask_prefix(64, MaskValue::LIVE_ALL).live(), 64);
    }

    /// 🎯 046/110 — ⛔ THE MULTIPLY BINDS BEFORE THE DIVIDE, and 60 lanes is where that shows.
    #[test]
    fn the_product_is_divided_not_the_lane_count() {
        // `(2 * 60) / 8 = 15` masked lanes, so 45 live — NOT `2 * (60 / 8) = 14`, which would be 46.
        assert_eq!(continuous_mask_prefix(60, MaskValue::Live6).live(), 45);
        assert_ne!(continuous_mask_prefix(60, MaskValue::Live6).live(), 46);
        // ⭐ AND IT NEVER UNDERFLOWS: seven eighths of any lane count is less than all of it.
        for lanes in [1_u64, 2, 7, 8, 60, 64, 1024] {
            assert!(continuous_mask_prefix(lanes, MaskValue::Live1).live() <= lanes);
        }
    }

    /// 🎯 046/110 — ⛔⛔ ONE OP AND NO `arith.constant`: the deliberate divergence, stated as a test.
    #[test]
    fn the_static_mask_emits_one_op_and_no_dead_constant() {
        let mut vals = Values::default();
        let mut body = Vec::new();
        let predicate = static_continuous_mask(&mut vals, &mut body, 64, MaskValue::Live6);
        assert_eq!(
            body,
            vec![Op::VectorChain(vectorchain::Op::CreateAffineMask {
                result: Val(0),
                mask: LaneMask::prefix_of(
                    48,
                    Vector {
                        len: 64,
                        elem: ElemType::Int(1),
                    }
                ),
            })]
        );
        // ⛔ THE DEAD `arith.constant 0 : index` IS NOT HERE, and no mask parameter is either.
        assert_eq!(body.len(), 1);
        // ⭐ THE PREDICATE CARRIES THE DEFINITION'S OWN TYPE.
        assert_eq!(predicate.val(), Val(0));
        assert_eq!(
            predicate.ty(),
            Vector {
                len: 64,
                elem: ElemType::Int(1),
            }
        );
    }
    /// 🎯 046/110 — ⭐⭐ THE CROSS-BRIDGE ROUND TRIP: `k` GOES INTO BRIDGE 1 AND COMES BACK OUT OF
    /// BRIDGE 2.
    ///
    /// This is the check that is not a tautology: it does not re-derive the set with the same
    /// arithmetic, it hands the op this function emits to the OTHER bridge's reader
    /// ([`get_mask_value_for_pt`], entry 089/384, a separate transcription of
    /// `VectorChainToSentientPT/Helper.cpp`) and asserts that the masked-column count it recovers is
    /// the table value that went in. Over a 64-lane row, `num_masked_elems = (k * 64) / 8 = 8k` and
    /// `masked_columns = 8k / 8 = k` for every one of the eight rows — so the identity holds for the
    /// whole table, not just for IBM's `k = 2` golden.
    #[test]
    fn every_table_row_survives_the_trip_through_bridge_two() {
        let row = Vector {
            len: 64,
            elem: ElemType::F16,
        };
        for bits in [255, 127, 63, 31, 15, 7, 3, 1] {
            let mask = MaskValue::of(bits).expect("a table key");
            let mut vals = Values::default();
            let mut body = Vec::new();
            static_continuous_mask(&mut vals, &mut body, row.len, mask);
            let [Op::VectorChain(emitted)] = &body[..] else {
                unreachable!("one `vectorchain.create_affine_mask` and nothing else")
            };

            let mut sen_values = Values::default();
            let recovered = get_mask_value_for_pt::<Dd2>(
                PtUnit,
                row,
                emitted,
                Definitions::from_innermost(&[]),
                &[],
                &mut sen_values,
            );
            let Some(PtMaskValue::Constant { columns, .. }) = recovered else {
                unreachable!("a static mask with no parameter reaches the constant arm")
            };
            assert_eq!(
                columns,
                MaskedColumns(u32::try_from(mask.slices_masked()).unwrap()),
                "mask value {bits} masks {} of the eight slices",
                mask.slices_masked()
            );
        }
    }
}
