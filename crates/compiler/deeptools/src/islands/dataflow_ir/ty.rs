//! THE TYPES DATAFLOWIR IS WRITTEN IN — element formats, shaped types, and affine maps.

use crate::generated::{DataType, Unit};

/// WHICH GENERIC COMPONENT a unit belongs to — `senCompToGenericComp`.
///
/// ⭐ IT EXISTS BECAUSE ONE ELEMENT FORMAT DEPENDS ON IT. `SENINT24` is `i24` on the PT and `i16`
/// everywhere else (`SNComputeLowering.cpp:291-297`), so the format alone does not determine the
/// type — a fact that is invisible until an accumulator is silently eight bits narrow.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GenericComp {
    /// The matrix unit, including every row and row span of it.
    Pt,
    /// The processing element.
    Pe,
    /// The special function processor.
    Sfp,
    /// An LX load or store unit.
    Lx,
    /// An L0 load or store unit.
    L0,
    /// A constant source — not a hardware unit, but a `unit="constant"` a transfer sources from.
    Constant,
}

impl Unit {
    /// WHICH GENERIC COMPONENT THIS UNIT IS. A total match: the set is generated from the templates,
    /// so a new spelling stops this compiling rather than falling into a default.
    #[must_use]
    pub const fn generic(self) -> GenericComp {
        match self {
            // ⛔ EVERY PT ROW AND ROW SPAN IS THE PT. `ptrow1-7` is a span of seven instruction
            // streams (`bmm.ddl:260`), not a unit of its own — but for the purpose of an element
            // format they are all the matrix unit, which is what `senCompToGenericComp` says.
            Self::Pt
            | Self::Ptnorth
            | Self::Ptsouth
            | Self::Ptrow0
            | Self::Ptrow3
            | Self::Ptrow7
            | Self::Ptrow1To3
            | Self::Ptrow1To7 => GenericComp::Pt,
            Self::Pe => GenericComp::Pe,
            Self::Sfp | Self::Sfpring => GenericComp::Sfp,
            Self::Lxlu | Self::Lxsu => GenericComp::Lx,
            Self::L0lu | Self::L0su => GenericComp::L0,
            Self::Constant => GenericComp::Constant,
        }
    }
}

/// WHETHER A TENSOR CARRIES ITS OWN SCALE — `LabeledDsInfo::ScaledLdsCategory`.
///
/// ⛔ IT CHANGES THE ELEMENT TYPE, not just its interpretation: a regular fp8 tensor is
/// `f8E4M3FN`, and a scaled one is a `CustomMXFloatType` of the same width
/// (`SNComputeLowering.cpp:298-304`). Emitting the first where the second belongs writes a type MLIR
/// accepts and the microcode reads differently.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorCategory {
    /// A plain tensor.
    Regular,
    /// One of a scaled pair, whose elements are MX-format.
    Scaled,
}

/// AN ELEMENT TYPE, as MLIR spells it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElemType {
    /// A signless integer of this many bits.
    Int(u32),
    /// `f16`.
    F16,
    /// `f32`.
    F32,
    /// `bf16`.
    Bf16,
    /// `f8E4M3FN`.
    F8E4M3Fn,
    /// `f8E8M0FNU`.
    F8E8M0Fnu,
    /// `f4E2M1FN`.
    F4E2M1Fn,
    /// `dataflow.mxfloat<N>` — the MX element of a scaled tensor.
    MxFloat(u32),
}

impl ElemType {
    /// THE ELEMENT TYPE ONE FORMAT HAS, on this component, in this category.
    ///
    /// A transcription of `SNComputeLowering::constructTypeFromFormat`
    /// (`SNComputeLowering.cpp:278-345`). Total over the generated [`DataType`], so a template
    /// introducing a format stops this compiling.
    #[must_use]
    pub const fn of(format: DataType, on: GenericComp, category: TensorCategory) -> ElemType {
        match format {
            DataType::Senint8 => ElemType::Int(8),
            DataType::Senint4 => ElemType::Int(4),
            // ⛔ 24 BITS ON THE PT, 16 EVERYWHERE ELSE (`:291-297`). This is the partial-sum
            // accumulator's format, so getting it wrong on the PT silently truncates every matmul.
            DataType::Senint24 => match on {
                GenericComp::Pt => ElemType::Int(24),
                GenericComp::Pe
                | GenericComp::Sfp
                | GenericComp::Lx
                | GenericComp::L0
                | GenericComp::Constant => ElemType::Int(16),
            },
            DataType::Sen169Fp16 => ElemType::F16,
            DataType::Bfloat16 => ElemType::Bf16,
            DataType::IeeeFp32 => ElemType::F32,
            DataType::Senuint32 => ElemType::Int(32),
            DataType::Bool => ElemType::Int(1),
            // ⭐ `SEN053_FP8` USES E4M3 TOO, and that is the C++'s own note rather than an oversight:
            // "currently, using E4M3 instead of E5M3 because of lack of that availability in MLIR"
            // (`:322-324`).
            DataType::Sen143Fp8 | DataType::Sen053Fp8 => match category {
                TensorCategory::Regular => ElemType::F8E4M3Fn,
                TensorCategory::Scaled => ElemType::MxFloat(8),
            },
            DataType::Sen080Fp8 => match category {
                TensorCategory::Regular => ElemType::F8E8M0Fnu,
                TensorCategory::Scaled => ElemType::MxFloat(8),
            },
            DataType::Sen121Fp4 => match category {
                TensorCategory::Regular => ElemType::F4E2M1Fn,
                TensorCategory::Scaled => ElemType::MxFloat(4),
            },
        }
    }
}

/// A `memref<AxBx...xT>` — a multi-dimensional view over a linear region.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemRef {
    /// The extents, outermost first.
    pub shape: Vec<u64>,
    /// The element type.
    pub elem: ElemType,
}

/// A `vector<NxT>` — what a send carries and a compute reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Vector {
    /// How many elements.
    pub len: u64,
    /// The element type.
    pub elem: ElemType,
}

/// ONE AFFINE EXPRESSION — the arithmetic a layout map or a reduction map is written in.
///
/// ⭐ THE SET IS WHAT THE VENDORED IR USES AND NOTHING MORE: `d0`, a constant, `+`, `*`, `mod` and
/// `floordiv`. `#map5 = affine_map<(d0) -> ((d0 mod 128) floordiv 2)>`
/// (`dcc/test/PT/xrfbmm_int8_fwd.mlir`) is the deepest one an int8 MACC needs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AffineExpr {
    /// `d<N>` — the Nth dimension of the map.
    Dim(u32),
    /// A literal.
    Const(i64),
    /// `a + b`.
    Add(Box<AffineExpr>, Box<AffineExpr>),
    /// `a * b`.
    Mul(Box<AffineExpr>, Box<AffineExpr>),
    /// `a mod b`.
    Mod(Box<AffineExpr>, Box<AffineExpr>),
    /// `a floordiv b`.
    FloorDiv(Box<AffineExpr>, Box<AffineExpr>),
}

impl AffineExpr {
    /// `d<n>`.
    #[must_use]
    pub fn dim(n: u32) -> AffineExpr {
        AffineExpr::Dim(n)
    }

    /// `self + other`.
    #[must_use]
    pub fn plus(self, other: AffineExpr) -> AffineExpr {
        AffineExpr::Add(Box::new(self), Box::new(other))
    }

    /// `self * k`.
    #[must_use]
    pub fn times(self, k: i64) -> AffineExpr {
        AffineExpr::Mul(Box::new(self), Box::new(AffineExpr::Const(k)))
    }

    /// `self mod k`.
    #[must_use]
    pub fn modulo(self, k: i64) -> AffineExpr {
        AffineExpr::Mod(Box::new(self), Box::new(AffineExpr::Const(k)))
    }

    /// `self floordiv k`.
    #[must_use]
    pub fn floordiv(self, k: i64) -> AffineExpr {
        AffineExpr::FloorDiv(Box::new(self), Box::new(AffineExpr::Const(k)))
    }
}

/// An `affine_map<(d0, ..) -> (..)>`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AffineMap {
    /// How many dimensions it takes.
    pub dims: u32,
    /// What it produces — one expression per result.
    pub results: Vec<AffineExpr>,
}

impl AffineMap {
    /// A one-dimensional, one-result map.
    #[must_use]
    pub fn unary(expr: AffineExpr) -> AffineMap {
        AffineMap {
            dims: 1,
            results: vec![expr],
        }
    }

    /// THE IDENTITY OVER `n` DIMENSIONS, linearised by the given strides — the layout map a
    /// `get_logical_memory_view` carries.
    ///
    /// ⛔ IN ELEMENTS, NOT BYTES (`Dataflow.td:250`). `affine_map<(i, j) -> (i + j * 8)>` means index
    /// `(0, 1)` is ELEMENT 8 of the region.
    #[must_use]
    pub fn linear(strides: &[i64]) -> AffineMap {
        let expr = strides
            .iter()
            .enumerate()
            .map(|(i, stride)| {
                let dim = AffineExpr::dim(u32::try_from(i).expect("a shape rank fits a u32"));
                if *stride == 1 {
                    dim
                } else {
                    dim.times(*stride)
                }
            })
            .reduce(AffineExpr::plus)
            .unwrap_or(AffineExpr::Const(0));
        AffineMap {
            dims: u32::try_from(strides.len()).expect("a shape rank fits a u32"),
            results: vec![expr],
        }
    }
}
