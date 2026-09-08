//! THE TYPES DATAFLOWIR IS WRITTEN IN — element formats, shaped types, and affine maps.

use crate::generated::{DataType, Unit};

/// WHICH GENERIC COMPONENT a unit belongs to — the IMAGE of `senCompToGenericComp`
/// (`sys-arch-spec/arch_enums.cpp:124-211`).
///
/// ⭐ IT EXISTS BECAUSE ONE ELEMENT FORMAT DEPENDS ON IT. `SENINT24` is `i24` on the PT and `i16`
/// everywhere else (`SNComputeLowering.cpp:291-297`), so the format alone does not determine the
/// type — a fact that is invisible until an accumulator is silently eight bits narrow.
///
/// # 🛑 A LOAD UNIT AND A STORE UNIT ARE DIFFERENT GENERIC COMPONENTS
///
/// ⛔⛔ THIS ENUM HAD ONE `Lx` FOR BOTH LX HALVES AND ONE `L0` FOR BOTH L0 HALVES, while citing the
/// very map that keeps them apart. `senCompToGenericComp` sends `LXLU0`, `LXLU1` and `LXLU` to
/// **`LXLU`** and `LXSU0`, `LXSU1`, `LXSU` to **`LXSU`** (`arch_enums.cpp:167-175`), and the
/// twenty-odd `L0LUROW*` spellings to **`L0LU`** against `L0SU0`/`L0SU1`/`L0SU` to **`L0SU`**
/// (`:177-204`). `LX` and the two L3 halves are their own images as well.
///
/// ⛔ AND THE COLLAPSE MADE TWO REFERENCE FUNCTIONS INEXPRESSIBLE. `isSenComponentL0LU` and
/// `isSenComponentL0SU` (`DataflowToSentient.cpp:96,100`) are *"is this unit's generic component
/// `L0LU`"* and *"… `L0SU`"* — two functions that would have had the same answer under one `L0`, and
/// their callers use them to tell a producer from a consumer.
///
/// # 🛑 THREE OF OUR UNITS ARE NOT IN THE MAP AT ALL
///
/// ⛔ `L0`, `CONSTANT` AND `SFPRING` ARE NOT KEYS (checked against the whole table,
/// `arch_enums.cpp:124-211`), so the reference's `senCompToGenericComp.at(comp)` **throws** for
/// them. This crate never runtime-refuses, so they are their own images here — a total function
/// where the reference has a partial one. That is safe for every rule built on this: each asks
/// `generic() == <a specific component>`, and these three answer no.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GenericComp {
    /// `PT` — the matrix unit, including every row, row span and per-fold copy of it.
    Pt,
    /// `PE` — the processing element (`PE0`/`PE1` fold into it).
    Pe,
    /// `SFP` — the special function processor (`SFP0`/`SFP1` fold into it).
    Sfp,
    /// `LXLU` — the LX **load** unit.
    Lxlu,
    /// `LXSU` — the LX **store** unit.
    Lxsu,
    /// `LX` — the LX memory itself, which is its own image (`arch_enums.cpp:176`).
    Lx,
    /// `L0LU` — the L0 **load** unit, which every `L0LUROW*` spelling folds into.
    L0lu,
    /// `L0SU` — the L0 **store** unit.
    L0su,
    /// `L3LU` — the L3 load half.
    L3lu,
    /// `L3SU` — the L3 store half.
    L3su,
    /// `HBM` — the device's global memory.
    Hbm,
    /// `LXVIRTUALIBR` — the LX virtual indirection base register, its own image
    /// (`arch_enums.cpp:209`): the map sends `LXVIRTUALIBR` to `LXVIRTUALIBR`.
    LxVirtualIbr,
    /// `L3IBR` — the L3 indirection base register, its own image (`arch_enums.cpp:208`).
    L3Ibr,
    /// `CROSSPTNLINK` — the link out of this partition.
    CrossPtnLink,
    /// `SFPSTATE`.
    SfpState,
    /// `PESTATE`.
    PeState,
    /// The L0 memory itself. ⛔ NOT A KEY OF THE REFERENCE MAP — see the type's note.
    L0,
    /// A constant source — a `unit="constant"` a transfer reads from. ⛔ NOT A KEY.
    Constant,
    /// The SFP ring. ⛔ NOT A KEY.
    SfpRing,
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
            Self::Sfp => GenericComp::Sfp,
            // ⛔ THE RING IS NOT THE SFP. `SFPRING` is not a key of `senCompToGenericComp` at all
            // (`arch_enums.cpp:124-211`), so folding it into `SFP` here was this crate's invention.
            Self::Sfpring => GenericComp::SfpRing,
            // ⛔ THE LOAD HALF AND THE STORE HALF ARE DIFFERENT IMAGES (`arch_enums.cpp:167-175`).
            Self::Lxlu => GenericComp::Lxlu,
            Self::Lxsu => GenericComp::Lxsu,
            Self::L0lu => GenericComp::L0lu,
            Self::L0su => GenericComp::L0su,
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
    /// `f8E5M2`.
    F8E5M2,
    /// `dataflow.mxint<N>` — the MX element of a scaled INTEGER tensor, `mxint4` being the only
    /// width the reference's own string table names (`VectorChainHelper.hpp:196-221`).
    MxInt(u32),
}

impl ElemType {
    /// HOW MANY **BITS** ONE ELEMENT OCCUPIES.
    ///
    /// ⛔⛔ BITS, BECAUSE THAT IS WHAT CONSUMES IT. Every `element_size` attribute in
    /// `SentientOps.td` is a bit width — its own example walks *"64 16 bit elements"* at
    /// `element_size = 16` and reinterprets them as *"four 4 bit elements"* at `element_size = 4`
    /// (`SentientOps.td:443-456`) — so a byte width here would be eight times too small at every
    /// use while still printing plausibly for f16.
    ///
    /// ⛔ TOTAL OVER THE ENUM, NO WILDCARD: a new element type must state its width.
    #[must_use]
    pub const fn bits(self) -> u32 {
        match self {
            ElemType::Int(bits) | ElemType::MxFloat(bits) | ElemType::MxInt(bits) => bits,
            ElemType::F16 | ElemType::Bf16 => 16,
            ElemType::F32 => 32,
            ElemType::F8E4M3Fn | ElemType::F8E8M0Fnu | ElemType::F8E5M2 => 8,
            ElemType::F4E2M1Fn => 4,
        }
    }

    /// HOW MLIR SPELLS IT — the printer's form, and the one the reference's own string tables use
    /// (`EnumsConversion` and `VectorChainHelper.hpp:196-244` both spell `fp16`/`bf16`/`fp32`
    /// differently; those are DCC attribute spellings, not type spellings, and live at entry 231).
    ///
    /// ⛔ TOTAL OVER THE ENUM, NO WILDCARD.
    #[must_use]
    pub fn spelling(self) -> String {
        match self {
            ElemType::Int(bits) => format!("i{bits}"),
            ElemType::F16 => "f16".to_owned(),
            ElemType::F32 => "f32".to_owned(),
            ElemType::Bf16 => "bf16".to_owned(),
            ElemType::F8E4M3Fn => "f8E4M3FN".to_owned(),
            ElemType::F8E8M0Fnu => "f8E8M0FNU".to_owned(),
            ElemType::F4E2M1Fn => "f4E2M1FN".to_owned(),
            ElemType::F8E5M2 => "f8E5M2".to_owned(),
            ElemType::MxFloat(bits) => format!("!dataflow.mxfloat<{bits}>"),
            ElemType::MxInt(bits) => format!("!dataflow.mxint<{bits}>"),
        }
    }

    /// WHETHER `isa<FloatType>` HOLDS OF IT, which decides whether a `sentient.scalar_constant`
    /// prints its value as a hex bit pattern (`SentientOps.cpp:1704-1706`).
    ///
    /// ⛔ FALSE FOR THE TWO `Mx` ARMS. They are `!dataflow.mxfloat`/`!dataflow.mxint`, DCC's own
    /// dialect types — `isa<FloatType>` is false of a custom type however float-like its name.
    #[must_use]
    pub const fn is_float(self) -> bool {
        match self {
            ElemType::F16
            | ElemType::F32
            | ElemType::Bf16
            | ElemType::F8E4M3Fn
            | ElemType::F8E8M0Fnu
            | ElemType::F4E2M1Fn
            | ElemType::F8E5M2 => true,
            ElemType::Int(_) | ElemType::MxFloat(_) | ElemType::MxInt(_) => false,
        }
    }

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
                | GenericComp::Lxlu
                | GenericComp::Lxsu
                | GenericComp::Lx
                | GenericComp::L0lu
                | GenericComp::L0su
                | GenericComp::L0
                | GenericComp::L3lu
                | GenericComp::L3su
                | GenericComp::Hbm
                | GenericComp::LxVirtualIbr
                | GenericComp::L3Ibr
                | GenericComp::CrossPtnLink
                | GenericComp::SfpState
                | GenericComp::PeState
                | GenericComp::Constant
                | GenericComp::SfpRing => ElemType::Int(16),
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

/// THE TYPE OF ONE SCALAR THE PROGRAM COMPUTES WITH — `AnyTypeOf<[AnyInteger, Index]>`.
///
/// ⭐⭐ IT IS THE OPERAND **AND** RESULT TYPE OF EVERY SCALAR OP AT BOTH RUNGS. `arith.addi`,
/// `arith.subi`, `arith.muli` and `arith.constant` carry it below;
/// `sentient.scalar_add`/`_sub`/`_mul` declare it three times over with
/// `SameOperandsAndResultType`, and `sentient.scalar_constant` prints it as its result type
/// (`SentientOps.td:700`, `:801`, `:816`, `:848`). One type serves both because the lowering's
/// whole job is to carry it across unchanged — `AddOp::create(builder, loc,
/// addi_op.getLhs().getType(), ...)` (`StandardToSentient.cpp:81`) passes the operand's type in as
/// the result type.
///
/// ⛔ NOT [`ElemType`]. That is the element format of a tensor or a vector — it has `f16`, `bf16`
/// and the fp8 formats in it, and it has no `index` at all. A loop counter is not an element of
/// anything, and the two sets meet only at `Int`.
///
/// ⭐ AND IT ADMITS AN [`ElemType`] THROUGH [`ScalarTy::of_elem`], because `scalar_constant`'s
/// result type is `AnyTypeOf<[AnyInteger, Index, AnyFloat]>` (`SentientOps.td:850`) and entry 235
/// `createSentientConstants` passes a splat's ELEMENT type in as it: `vector<1xf16>` in,
/// `sentient.scalar_constant {value = 0xff : si64} : f16` out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarTy {
    /// `index` — what every loop bound, address and induction variable is typed.
    Index,
    /// `i<bits>` — a signless integer. `i1` is the type of a predicate.
    Int(u32),
    /// A tensor or vector ELEMENT type used as a scalar type — a float, in practice.
    ///
    /// ⛔ CONSTRUCT IT THROUGH [`ScalarTy::of_elem`], never directly: that is what keeps
    /// `Int(8)` and `Elem(Int(8))` from both arising for one type and comparing unequal.
    Elem(ElemType),
}

impl ScalarTy {
    /// HOW MLIR SPELLS IT.
    #[must_use]
    pub fn spelling(self) -> String {
        match self {
            ScalarTy::Index => "index".to_owned(),
            ScalarTy::Int(bits) => format!("i{bits}"),
            ScalarTy::Elem(elem) => elem.spelling(),
        }
    }

    /// THE SCALAR TYPE ONE ELEMENT TYPE IS — the only way to build [`ScalarTy::Elem`], and the
    /// reason an integer element cannot arrive spelled two ways.
    #[must_use]
    pub const fn of_elem(elem: ElemType) -> ScalarTy {
        match elem {
            ElemType::Int(bits) => ScalarTy::Int(bits),
            other => ScalarTy::Elem(other),
        }
    }

    /// WHETHER THE VALUE OF A CONSTANT OF THIS TYPE PRINTS AS A HEX BIT PATTERN
    /// (`SentientOps.cpp:1704-1706` takes that branch for `isa<FloatType>`).
    #[must_use]
    pub const fn is_float(self) -> bool {
        match self {
            ScalarTy::Index | ScalarTy::Int(_) => false,
            ScalarTy::Elem(elem) => elem.is_float(),
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
    /// `s<N>` — the Nth SYMBOL of the map.
    ///
    /// ⛔⛔ A SYMBOL IS NOT A DIMENSION, AND THE DIFFERENCE IS WHAT MAKES A CONSTRAINT SOLVABLE.
    /// MLIR's own rule: a dimension is a value the map is *indexed by*, a symbol is a value that is
    /// **loop-invariant with respect to the map's own iteration space** — so a symbol may be
    /// multiplied by a dimension and still be affine, while a product of two dimensions is not.
    /// `TPMVBase::replaceDimsInMapWithSyms` (`TransformPagedMemViewImpl.cpp:47`) exists for exactly
    /// that reason: it rewrites a subscripts map's loop iterators as symbols so the map can be fed
    /// to a `FlatLinearValueConstraints` system that solves for which PAGE a subscript lands in,
    /// with the iterators as parameters rather than as unknowns.
    ///
    /// ⭐ NUMBERED IN ITS OWN SPACE. `s0` and `d0` are two different variables; the map states how
    /// many of each it takes ([`AffineMap::dims`], [`AffineMap::syms`]) and prints them in two
    /// groups, `(d0, d1)[s0, s1]`.
    ///
    /// ⭐ AND A SECOND LOWERING TURNS ON THE SAME DIFFERENCE. `getMaskValueForPT` (entry 089) refuses
    /// a constant mask whose set still has a symbol (*"Mask affine set should not have any symbols"*)
    /// and refuses a dynamic one that does not have exactly one (*"Mask affine set should have 1
    /// symbol"*) — two counts, two messages, one set (`Helper.cpp:82-85`, `:117-120`). The vendor
    /// writes both in one line: `#set = affine_set<(d0)[s0] : (d0 + s0 * 8 - 64 >= 0, -d0 + 63 >= 0)>`
    /// (`dcc/test/Conversion/VectorChainToSentientPT/dynamic_pt_masking.mlir:5`), where `d0` is the
    /// lane and `s0` the loop iterator the mask advances with. The separate position spaces are what
    /// let [`IntegerSet::replace_symbols`] substitute `s0` without touching `d0`.
    Sym(u32),
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

    /// `s<n>` — `bindSymbols(context, s0)` (`Helper.cpp:204`).
    #[must_use]
    pub fn sym(n: u32) -> AffineExpr {
        AffineExpr::Sym(n)
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

    /// `AffineExpr::walk(callback)` — this node and everything under it, CHILDREN FIRST
    /// (`llvm-project/mlir/lib/IR/AffineExpr.cpp`, `AffineExprVisitor::walkPostOrder`).
    pub fn walk(&self, callback: &mut impl FnMut(&AffineExpr)) {
        match self {
            AffineExpr::Dim(_) | AffineExpr::Sym(_) | AffineExpr::Const(_) => {}
            AffineExpr::Add(lhs, rhs)
            | AffineExpr::Mul(lhs, rhs)
            | AffineExpr::Mod(lhs, rhs)
            | AffineExpr::FloorDiv(lhs, rhs) => {
                lhs.walk(callback);
                rhs.walk(callback);
            }
        }
        callback(self);
    }

    /// `AffineExpr::isFunctionOfDim(unsigned position)` — DOES THIS EXPRESSION MENTION `d<dim>`?
    ///
    /// ```cpp
    /// bool AffineExpr::isFunctionOfDim(unsigned position) const {
    ///   if (getKind() == AffineExprKind::DimId) {
    ///     return *this == mlir::getAffineDimExpr(position, getContext());
    ///   }
    ///   if (auto expr = llvm::dyn_cast<AffineBinaryOpExpr>(*this)) {
    ///     return expr.getLHS().isFunctionOfDim(position) ||
    ///            expr.getRHS().isFunctionOfDim(position);
    ///   }
    ///   return false;
    /// }
    /// ```
    /// (`llvm-project/mlir/lib/IR/AffineExpr.cpp:314-323`)
    ///
    /// ⛔ A SYMBOL IS NEVER A DIMENSION, whatever value it carries — see [`AffineExpr::Sym`]. The
    /// reference's own test is an identity comparison against `getAffineDimExpr(position)`, so
    /// `d1` answers `false` for position 0 and nothing else in the grammar answers at all.
    ///
    /// ⭐⭐ TWO PORTS TURN ON IT, BOTH OVER A SUBSCRIPTS MAP.
    /// `extractConstantOffsetsFromMapForDim` (`Dialect/Agen/Utils.cpp:520`) reads a **0** coefficient
    /// for every result the split dimension does not appear in, and
    /// `updateSubscriptsAndIndicesForExplicitTimeLoops` (`:466`) binds a **constant 0** for every time
    /// dimension the concatenated map turns out not to use. Both are "this result does not move with
    /// that iterator", and both are wrong in the same direction if a symbol were counted: an address
    /// that shifts with an iterator it does not depend on.
    #[must_use]
    pub fn is_function_of_dim(&self, dim: u32) -> bool {
        match self {
            AffineExpr::Dim(n) => *n == dim,
            AffineExpr::Sym(_) | AffineExpr::Const(_) => false,
            AffineExpr::Add(a, b)
            | AffineExpr::Mul(a, b)
            | AffineExpr::Mod(a, b)
            | AffineExpr::FloorDiv(a, b) => a.is_function_of_dim(dim) || b.is_function_of_dim(dim),
        }
    }

    /// `self + other`, **SIMPLIFIED** — MLIR's `AffineExpr::operator+`, which is `simplifyAdd`.
    ///
    /// ⛔⛔ THIS IS NOT A DUPLICATE OF [`Self::plus`] AND THE PAIR IS DELIBERATE. [`Self::plus`]
    /// builds the node verbatim, because a map this island *writes into a program* is printed exactly
    /// as it was transcribed — see [`fold_add`]'s note. This one is for a map the island *recovers*:
    /// `getLayoutMapAndIndices` reads `agen.vector_load`'s own `AffineMapAttr`
    /// (`VectorOperands.cpp:904`), and an `affine_map` attribute in MLIR is **always** canonical,
    /// because the only way to get one is through the parser or through these operators. Rebuilding it
    /// unsimplified would hand `compose` a `d0 * 1 + 0` where the vendor's attribute holds `d0`.
    #[must_use]
    pub fn added(self, other: AffineExpr) -> AffineExpr {
        fold_add(self, other)
    }

    /// `self * k`, **SIMPLIFIED** — MLIR's `AffineExpr::operator*`, which is `simplifyMul`. See
    /// [`Self::added`] for why this exists beside [`Self::times`].
    #[must_use]
    pub fn scaled(self, k: i64) -> AffineExpr {
        fold_mul(self, AffineExpr::Const(k))
    }

    /// DOES THIS EXPRESSION NAME NO DIMENSION? — `AffineExpr::isSymbolicOrConstant()`.
    ///
    /// ⛔ IT IS THE TEST THAT DECIDES WHICH SIDE OF A PRODUCT IS THE MULTIPLIER, and MLIR's
    /// `simplifyMul` gives up outright when NEITHER side passes it: a product of two dimensions is
    /// not affine, so `d0 * d1` is built and left alone rather than rewritten (`AffineExpr.cpp`,
    /// `simplifyMul`'s second guard). A symbol passes because a symbol is loop-invariant with respect
    /// to the map's own iteration space — see [`AffineExpr::Sym`].
    #[must_use]
    pub fn is_symbolic_or_constant(&self) -> bool {
        match self {
            AffineExpr::Const(_) | AffineExpr::Sym(_) => true,
            AffineExpr::Dim(_) => false,
            AffineExpr::Add(a, b)
            | AffineExpr::Mul(a, b)
            | AffineExpr::Mod(a, b)
            | AffineExpr::FloorDiv(a, b) => {
                a.is_symbolic_or_constant() && b.is_symbolic_or_constant()
            }
        }
    }

    /// SUBSTITUTE `d<i>` BY `dims[i]` AND `s<j>` BY `syms[j]`, LEAF BY LEAF —
    /// `AffineExpr::replaceDimsAndSymbols`.
    ///
    /// ⛔ A POSITION BEYOND ITS REPLACEMENT LIST KEEPS ITSELF, which is not a courtesy but the
    /// mechanism [`AffineMap::compose`] runs on: it hands the inner map an EMPTY dimension list in
    /// effect (an identity one) and a symbol list covering only the inner map's own symbols, and the
    /// OUTER map's symbols are then untouched because `compose` passes no symbol replacements for
    /// them at all.
    ///
    /// ⛔ AND THE UNCHANGED-CHILDREN GUARD IS PART OF THE CONTRACT, not an optimisation. MLIR returns
    /// `*this` when neither child moved, so a node nothing was substituted into is NOT re-simplified.
    /// That is why composing with an order map that is the identity gives back the other map exactly:
    /// every `d<i>` is replaced by `d<i>`, so no binary node is ever rebuilt and no fold runs.
    #[must_use]
    fn replace_dims_and_symbols(&self, dims: &[AffineExpr], syms: &[AffineExpr]) -> AffineExpr {
        match self {
            AffineExpr::Const(_) => self.clone(),
            AffineExpr::Dim(n) => match dims.get(*n as usize) {
                Some(with) => with.clone(),
                None => self.clone(),
            },
            AffineExpr::Sym(n) => match syms.get(*n as usize) {
                Some(with) => with.clone(),
                None => self.clone(),
            },
            AffineExpr::Add(a, b)
            | AffineExpr::Mul(a, b)
            | AffineExpr::Mod(a, b)
            | AffineExpr::FloorDiv(a, b) => {
                let new_a = a.replace_dims_and_symbols(dims, syms);
                let new_b = b.replace_dims_and_symbols(dims, syms);
                if new_a == **a && new_b == **b {
                    return self.clone();
                }
                match self {
                    AffineExpr::Add(..) => fold_add(new_a, new_b),
                    AffineExpr::Mul(..) => fold_mul(new_a, new_b),
                    AffineExpr::Mod(..) => fold_mod(new_a, new_b),
                    // The `FloorDiv` case; the four leaves are answered above.
                    _ => fold_floordiv(new_a, new_b),
                }
            }
        }
    }

    /// `AffineExpr::shiftDims(numDims, shift)` — RENUMBER `d<i>` TO `d<i + shift>`.
    ///
    /// ```cpp
    /// for (unsigned idx = 0; idx < offset; ++idx)
    ///   dims.push_back(getAffineDimExpr(idx, getContext()));
    /// for (unsigned idx = offset; idx < numDims; ++idx)
    ///   dims.push_back(getAffineDimExpr(idx + shift, getContext()));
    /// return replaceDimsAndSymbols(dims, {});
    /// ```
    /// (`llvm-project/mlir/include/mlir/IR/AffineExpr.h:139-149`)
    ///
    /// ⭐ MLIR'S `offset` PARAMETER DEFAULTS TO 0 AND THE ONE CALL SITE THIS CAMPAIGN REACHES DOES
    /// NOT PASS IT — `concatenateMaps` shifts EVERY dimension of the second map past the first map's
    /// dimensions (`map_B.shiftDims(map_A.getNumDims())`, `dialect_utils/Agen/Utils.cpp:261`) — so
    /// the leading-dimensions loop is absent here rather than carried as a parameter no port sets.
    ///
    /// ⭐ `num_dims` IS THE MAP'S ARITY, NOT THE EXPRESSION'S. A dimension at or above it is left
    /// alone, which is `replaceDimsAndSymbols`' short-list rule doing the work.
    #[must_use]
    pub fn shifted_dims(&self, num_dims: u32, shift: u32) -> AffineExpr {
        let dims: Vec<AffineExpr> = (0..num_dims)
            .map(|idx| AffineExpr::Dim(idx.saturating_add(shift)))
            .collect();
        self.replace_dims_and_symbols(&dims, &[])
    }
}

/// `lhs + rhs`, FOLDED — the head of MLIR's `simplifyAdd`, and the reason a composed map prints the
/// way the reference's does.
///
/// ⭐⭐ THIS RUNS ONLY WHERE A SUBSTITUTION HAPPENED. The island's own constructors
/// ([`AffineExpr::plus`], [`AffineExpr::times`]) still do not fold — that is stated at
/// [`substitute_symbols`] and unchanged. This is `getAffineBinaryOpExpr`, which MLIR reaches ONLY
/// from `replaceDimsAndSymbols` rebuilding a node, so an `affine_map` this island writes directly is
/// printed exactly as it was built.
///
/// ⚠️ THE SUBSET IS BOUNDED AND THE BOUNDARY IS TEXTUAL, NOT SEMANTIC. Transcribed here: the
/// constant fold, the canonicalisation that moves a constant (or a dimension-free expression) to the
/// right, `x + 0`, and the successive-addition merge `(d0 + 2) + 3`. NOT transcribed: `c1*e + c2*e`
/// collapsing, and the `expr + expr floordiv q * -q` to `expr mod q` recognition. An expression
/// reaching one of those is BUILT rather than rewritten — which is also what MLIR does when
/// `simplifyAdd` returns null — so the map we print is equal as a function and may carry one more
/// node. ⭐ AND dbo-opt CLOSES EVEN THAT: MLIR's own affine parser builds every parsed expression
/// through these same operators (`AsmParser/AffineParser.cpp:160-165` returns `lhs + rhs`), so
/// whatever we print is re-simplified in full the moment the backend reads it.
fn fold_add(lhs: AffineExpr, rhs: AffineExpr) -> AffineExpr {
    // `if (lhsConst && rhsConst)` — ⛔ ON OVERFLOW MLIR RETURNS NULL AND BUILDS THE NODE, which
    // `checked_add` reproduces rather than wrapping into a wrong literal.
    if let (AffineExpr::Const(a), AffineExpr::Const(b)) = (&lhs, &rhs) {
        match a.checked_add(*b) {
            Some(sum) => return AffineExpr::Const(sum),
            None => return AffineExpr::Add(Box::new(lhs), Box::new(rhs)),
        }
    }
    // "Canonicalize so that only the RHS is a constant. (4 + d0 becomes d0 + 4)."
    if matches!(lhs, AffineExpr::Const(_))
        || (lhs.is_symbolic_or_constant() && !rhs.is_symbolic_or_constant())
    {
        return fold_add(rhs, lhs);
    }
    // "Addition with a zero is a noop."
    if rhs == AffineExpr::Const(0) {
        return lhs;
    }
    // "Fold successive additions like (d0 + 2) + 3 into d0 + 5."
    if let (AffineExpr::Add(l, l_rhs), AffineExpr::Const(k)) = (&lhs, &rhs)
        && let AffineExpr::Const(c) = **l_rhs
        && let Some(sum) = c.checked_add(*k)
    {
        return fold_add((**l).clone(), AffineExpr::Const(sum));
    }
    AffineExpr::Add(Box::new(lhs), Box::new(rhs))
}

/// `lhs * rhs`, FOLDED — the head of MLIR's `simplifyMul`. See [`fold_add`] for the subset rule.
///
/// ⛔ THE SECOND GUARD IS THE AFFINE-NESS ONE AND IT COMES BEFORE EVERY CANONICALISATION: with
/// neither side dimension-free the product is not affine, and MLIR returns null immediately rather
/// than trying to pick a multiplier. Reordering it after the swap would recurse forever on `d0 * d1`.
fn fold_mul(lhs: AffineExpr, rhs: AffineExpr) -> AffineExpr {
    if let (AffineExpr::Const(a), AffineExpr::Const(b)) = (&lhs, &rhs) {
        match a.checked_mul(*b) {
            Some(product) => return AffineExpr::Const(product),
            None => return AffineExpr::Mul(Box::new(lhs), Box::new(rhs)),
        }
    }
    if !lhs.is_symbolic_or_constant() && !rhs.is_symbolic_or_constant() {
        return AffineExpr::Mul(Box::new(lhs), Box::new(rhs));
    }
    // "Canonicalize the mul expression so that the constant/symbolic term is the RHS."
    if !rhs.is_symbolic_or_constant() || matches!(lhs, AffineExpr::Const(_)) {
        return fold_mul(rhs, lhs);
    }
    // "Multiplication with a one is a noop" / "Multiplication with zero."
    if rhs == AffineExpr::Const(1) {
        return lhs;
    }
    if rhs == AffineExpr::Const(0) {
        return AffineExpr::Const(0);
    }
    if let AffineExpr::Mul(l, l_rhs) = &lhs
        && let AffineExpr::Const(c) = **l_rhs
    {
        // "Fold successive multiplications: eg: (d0 * 2) * 3 into d0 * 6."
        if let AffineExpr::Const(k) = rhs
            && let Some(product) = c.checked_mul(k)
        {
            return fold_mul((**l).clone(), AffineExpr::Const(product));
        }
        // "turn (d0 * 2) * d1 into (d0 * d1) * 2."
        return fold_mul(fold_mul((**l).clone(), rhs), AffineExpr::Const(c));
    }
    AffineExpr::Mul(Box::new(lhs), Box::new(rhs))
}

/// `lhs mod rhs`, FOLDED — the head of MLIR's `simplifyMod`.
///
/// ⛔ A NON-POSITIVE MODULUS IS PRESERVED AS IS, verbatim from the reference: *"mod w.r.t zero or
/// negative numbers is undefined and preserved as is."* The remainder is the EUCLIDEAN one (MLIR's
/// own `mod` helper), matching [`terms_in`]'s `rem_euclid`, not Rust's `%`.
///
/// ⚠️ NOT TRANSCRIBED: the `getLargestKnownDivisor` folds — `(i * 128) mod 64` to `0`, the
/// summand-wise reduction, and `(e % a) % b`. Those need a divisor analysis this island has no
/// reader for; see [`fold_add`] for why the difference is textual.
fn fold_mod(lhs: AffineExpr, rhs: AffineExpr) -> AffineExpr {
    if let AffineExpr::Const(d) = rhs
        && d >= 1
        && let AffineExpr::Const(n) = lhs
    {
        return AffineExpr::Const(n.rem_euclid(d));
    }
    AffineExpr::Mod(Box::new(lhs), Box::new(rhs))
}

/// `lhs floordiv rhs`, FOLDED — the head of MLIR's `simplifyFloorDiv`.
///
/// ⛔ A NON-CONSTANT OR ZERO DIVISOR IS PRESERVED AS IS, and the quotient FLOORS
/// (`divideFloorSigned`), which `div_euclid` gives for a positive divisor and Rust's `/` does not.
///
/// ⚠️ NOT TRANSCRIBED: `(i * 128) floordiv 64` to `i * 2` and the summand-wise reduction, for the
/// same reason as [`fold_mod`].
fn fold_floordiv(lhs: AffineExpr, rhs: AffineExpr) -> AffineExpr {
    if let AffineExpr::Const(d) = rhs
        && d != 0
    {
        if let AffineExpr::Const(n) = lhs {
            return AffineExpr::Const(n.div_euclid(d));
        }
        if d == 1 {
            return lhs;
        }
    }
    AffineExpr::FloorDiv(Box::new(lhs), Box::new(rhs))
}

/// ONE CONSTRAINT OF AN `affine_set` — an expression that is either zero or non-negative.
///
/// ⛔ THE FLAG IS THE WHOLE DIFFERENCE BETWEEN A POINT AND A SPAN. MLIR's `IntegerSet` carries a
/// parallel `eqFlags` array rather than two expression kinds, and reading a `>= 0` as an `== 0`
/// turns "these sixty-four lanes" into "lane zero".
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Constraint {
    /// The expression constrained.
    pub expr: AffineExpr,
    /// `true` for `expr == 0`, `false` for `expr >= 0`.
    pub is_equality: bool,
}

/// An `affine_set<(d0, ..) : (..)>` — which indices of a walk are live.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IntegerSet {
    /// How many dimensions it constrains.
    pub dims: u32,
    /// How many SYMBOLS it takes — `getNumSymbols()`, the `[s0, ..]` list.
    ///
    /// ⛔ A SEPARATE COUNT FROM `dims`, BECAUSE THE REFERENCE ASKS THEM SEPARATELY AND ANSWERS
    /// DIFFERENTLY. `mask_set.getNumDims() != 1` and `mask_set.getNumSymbols() != 0` are two refusals
    /// with two messages in one function (`Helper.cpp:32-35`, `:82-85`), and
    /// `replaceDimsAndSymbols({}, {affine_const}, mask_set.getNumDims(), 0)` (`:75-76`) rebuilds a set
    /// with the SAME dims and ZERO symbols — an operation that cannot even be written if the two share
    /// one field.
    pub symbols: u32,
    /// The constraints, in order.
    pub constraints: Vec<Constraint>,
}

impl IntegerSet {
    /// THE SET A RECTANGLE OF `sizes` OCCUPIES — `buildIntegerSetFromSizes`
    /// (`DataTransferLowering.cpp:40-69`).
    ///
    /// A size of one pins its dimension to zero; any larger size spans `0 ..= n-1`, written as the
    /// PAIR `dk >= 0` and `n-1 - dk >= 0` because an `IntegerSet` has no two-sided constraint.
    ///
    /// ⭐ VERIFIED AGAINST IBM'S OWN SETS. Sizes `[1, 1, 64]` give
    /// `affine_set<(d0, d1, d2) : (d0 == 0, d1 == 0, d2 >= 0, -d2 + 63 >= 0)>`, which is `#set` of
    /// `/tmp/ktir_ref/export/debug/dfir.mlir`; `[1]` gives `#set2`, `affine_set<(d0) : (d0 == 0)>`,
    /// the single pinned time step of a transfer that fits in one vector.
    ///
    /// ⛔ AN EMPTY `sizes` IS THE EMPTY SET, NOT AN UNCONSTRAINED ONE. The C++ returns
    /// `IntegerSet::getEmptySet(0, 0, ..)` (`:42-44`); a set with no dimensions and no constraints
    /// would instead admit everything.
    #[must_use]
    pub fn from_sizes(sizes: &[u64]) -> IntegerSet {
        let mut constraints = Vec::new();
        for (i, size) in sizes.iter().enumerate() {
            let dim = AffineExpr::dim(u32::try_from(i).expect("a rank fits a u32"));
            if *size == 1 {
                constraints.push(Constraint {
                    expr: dim,
                    is_equality: true,
                });
            } else {
                constraints.push(Constraint {
                    expr: dim.clone(),
                    is_equality: false,
                });
                // `n - 1 - dk >= 0`, which prints as `-dk + (n-1) >= 0`.
                let bound = i64::try_from(*size).expect("an extent fits an i64") - 1;
                constraints.push(Constraint {
                    expr: dim.times(-1).plus(AffineExpr::Const(bound)),
                    is_equality: false,
                });
            }
        }
        IntegerSet {
            dims: u32::try_from(sizes.len()).expect("a rank fits a u32"),
            // `buildIntegerSetFromSizes` passes `/*numSymbols=*/0` (`DataTransferLowering.cpp:68`):
            // a rectangle's bounds are the sizes themselves, with nothing left to substitute.
            symbols: 0,
            constraints,
        }
    }
}

/// WHICH SIDE OF A CONSTRAINT SYSTEM A BOUND COMES FROM — `mlir::presburger::BoundType`.
///
/// ⭐ ITS THIRD ENUMERATOR IS DELIBERATELY ABSENT. MLIR's own `getConstantBound` opens with
/// `assert(type != BoundType::EQ && "EQ not implemented")`, so `EQ` exists there only to be
/// rejected at run time; here it cannot be written down.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundType {
    /// The greatest constant `l` with `l <= d<pos>` over the whole system.
    Lb,
    /// The least constant `u` with `d<pos> <= u`. **Inclusive**, so a 64-lane span is `0 ..= 63`.
    Ub,
}

impl IntegerSet {
    /// THE CONSTANT BOUND THIS SET PUTS ON DIMENSION `pos`, or `None` when it puts none —
    /// `FlatAffineValueConstraints::getConstantBound`, MLIR's
    /// `IntegerRelation::computeConstantLowerOrUpperBound`.
    ///
    /// An equality pinning the dimension to a constant on its own answers immediately; otherwise the
    /// answer is the tightest of the one-sided inequality bounds — the MAX of the lower bounds for
    /// [`BoundType::Lb`], the MIN of the upper bounds for [`BoundType::Ub`] — and `None` when the
    /// requested side has none. A constraint on OTHER dimensions says nothing about this one.
    ///
    /// ⛔⛔ AN EMPTY SET STILL HAS BOUNDS, AND A LOWERING DEPENDS ON IT.
    /// `affine_set<(d0) : (d0 - 64 >= 0, -d0 + 63 >= 0)>` admits no integer at all, yet its `Lb` is
    /// 64 and its `Ub` is 63 — both present. IBM writes exactly that set as the mask of twenty
    /// `create_affine_mask` ops in
    /// `dcc/test/Conversion/VectorChainToSentientPESFP/mixed_precision.mlir:747-895`, whose
    /// `CHECK-SENT-IR` lowers each to `sentient.scalar_constant {value = 0 : si64}` (`:288`, `:296`)
    /// — the all-lanes-off mask. `getConstantBound` runs no emptiness test, so `Lb > Ub` is a
    /// reachable and meaningful pair, and an emptiness check added here would turn twenty legal masks
    /// into `emitOpError("Mask affine set has to have constant bounds.")`.
    ///
    /// ⛔ AN EQUALITY WHOSE COEFFICIENT IS NOT ±1 BOUNDS NOTHING, which is MLIR's answer rather than
    /// a simplification of it: `findEqualityToConstant` skips any row with `v * v != 1`, and the scan
    /// that follows reads INEQUALITIES only — so `2*d0 - 4 == 0` gives `None` on both sides even
    /// though `d0` is plainly 2. Every `affine_set` in the authority tree's tests writes `dk == 0` or
    /// a `>= 0` pair with unit coefficients, so nothing there reaches that corner.
    ///
    /// ⚠️ NOT VERIFIABLE FROM THE AUTHORITY TREE. `getConstantBound` is MLIR's, and no MLIR source
    /// or header is present on this host — the shape above is from MLIR's published implementation,
    /// and only the ANSWERS are pinned by IBM's sets and their `CHECK-SENT-IR` lines.
    #[must_use]
    pub fn constant_bound(&self, bound: BoundType, pos: u32) -> Option<i64> {
        self.bound_on(bound, Var::Dim(pos))
    }

    /// THE BOUND ON VARIABLE POSITION `pos` IN MLIR'S FLATTENED COLUMN ORDER, `[dims, symbols]` —
    /// which is what `FlatLinearValueConstraints::getConstantBound`'s `pos` actually indexes.
    ///
    /// ⛔⛔ NOT A DIMENSION NUMBER, AND ENTRY 198 IS THE CALLER THAT PROVES IT.
    /// `TPMVBase::getPageValidity` builds its page-selection system with ZERO dims and one SYMBOL
    /// per loop iterator (`TransformPagedMemViewImpl.cpp:120-125`, `:122-124`), and
    /// `createConditionsForHyperRectSubscripts` then reads `getConstantBound(LB, dim)` for `dim`
    /// over the SUBSCRIPT map's dimensions (`:299-302`) — so position 0 is symbol 0 there.
    /// [`Self::constant_bound`] asks about `d<pos>` and would answer `None` for every one of them,
    /// which is `"expected constant lower and upper bounds"` on a system that has both.
    #[must_use]
    pub fn constant_bound_on_var(&self, bound: BoundType, pos: u32) -> Option<i64> {
        self.bound_on(
            bound,
            if pos < self.dims {
                Var::Dim(pos)
            } else {
                Var::Sym(pos - self.dims)
            },
        )
    }

    /// Both doors' shared body, over the variable each resolved `pos` to.
    fn bound_on(&self, bound: BoundType, var: Var) -> Option<i64> {
        // `findEqualityToConstant(*this, 0, symbolic=false)`: a unit coefficient, and no other
        // dimension in the row. `-c / a` is exact because `a` is ±1.
        for constraint in &self.constraints {
            if let (true, Terms::OnPos { coeff, constant }) =
                (constraint.is_equality, terms_in(&constraint.expr, var))
                && (coeff == 1 || coeff == -1)
            {
                return Some(-constant / coeff);
            }
        }

        let mut tightest: Option<i64> = None;
        for constraint in &self.constraints {
            let Terms::OnPos { coeff, constant } = terms_in(&constraint.expr, var) else {
                continue;
            };
            if constraint.is_equality {
                continue;
            }
            // `a*d + c >= 0`, so `a > 0` reads `d >= -c/a` rounded UP, and `a < 0` reads
            // `d <= c/-a` rounded DOWN.
            let side = match (bound, coeff > 0) {
                (BoundType::Lb, true) => ceil_div(-constant, coeff),
                (BoundType::Ub, false) => constant.div_euclid(-coeff),
                // The constraint bounds this dimension's OTHER side.
                (BoundType::Lb, false) | (BoundType::Ub, true) => continue,
            };
            tightest = Some(match (tightest, bound) {
                (None, _) => side,
                (Some(so_far), BoundType::Lb) => so_far.max(side),
                (Some(so_far), BoundType::Ub) => so_far.min(side),
            });
        }
        tightest
    }

    /// SYMBOLS SUBSTITUTED AWAY — `IntegerSet::replaceDimsAndSymbols({}, symReplacements, dims, syms)`
    /// with the dimension list left empty, which is the only way entry 089 calls it:
    ///
    /// ```text
    /// mask_set = mask_set.replaceDimsAndSymbols({}, {affine_const}, mask_set.getNumDims(), 0);
    /// ```
    /// (`Conversion/VectorChainLowering/VectorChainToSentientPT/Helper.cpp:75-76`)
    ///
    /// ⛔⛔ THIS IS WHAT MAKES A DYNAMIC MASK STATIC, AND IT IS NOT A COSMETIC REWRITE. `getMaskValueForPT`
    /// reads its `d0` bounds with `getConstantBound`, which answers `None` for any row that still holds
    /// another variable. A mask whose parameter turned out to be an `arith.constant` therefore has to
    /// have `s0` folded into the row FIRST; without this the constant branch reads no bound at all and
    /// the reference's `static_mask = true` assignment (`:74`) would be a lie.
    ///
    /// ⭐ TOTAL, AND FAITHFUL ABOUT WHAT IT LEAVES ALONE. A symbol beyond `replacements` keeps its own
    /// position, exactly as MLIR's does — the caller states the resulting symbol count, so a partial
    /// substitution is expressible rather than silently completed.
    /// DIMENSIONS SUBSTITUTED AWAY — `IntegerSet::replaceDimsAndSymbols(dimReplacements, {}, dims,
    /// syms)` with the SYMBOL list left empty, which is how `getPageValidity` calls it:
    ///
    /// ```text
    /// subscripts_in_page = subscripts_in_page.replaceDimsAndSymbols(
    ///     subscripts_map_sym.getResults(), {}, 0, subscripts_map_sym.getNumSymbols());
    /// ```
    /// (`Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:122-125`)
    ///
    /// ⭐⭐ IT IS WHAT PUTS THE ACCESS AND THE PAGE IN ONE SPACE. The page's set is written over the
    /// view's dimensions; substituting each one by the SUBSCRIPT that indexes it leaves a set over the
    /// loop iterators alone, which is why [`super::super::super::bridges::dataflow_ir_to_sentient::tf_transform_paged_mem_view_impl::add_constraints_for_iv_ranges`]
    /// can then bound them as symbols.
    ///
    /// ⭐ TOTAL, AND FAITHFUL ABOUT WHAT IT LEAVES ALONE, exactly as [`Self::replace_symbols`] is: a
    /// dimension beyond `replacements` keeps its own position and the caller states both resulting
    /// counts.
    #[must_use]
    pub fn replace_dims(
        &self,
        replacements: &[AffineExpr],
        result_dims: u32,
        result_symbols: u32,
    ) -> IntegerSet {
        IntegerSet {
            dims: result_dims,
            symbols: result_symbols,
            constraints: self
                .constraints
                .iter()
                .map(|constraint| Constraint {
                    expr: constraint.expr.replace_dims_and_symbols(replacements, &[]),
                    is_equality: constraint.is_equality,
                })
                .collect(),
        }
    }

    #[must_use]
    pub fn replace_symbols(&self, replacements: &[AffineExpr], result_symbols: u32) -> IntegerSet {
        IntegerSet {
            dims: self.dims,
            symbols: result_symbols,
            constraints: self
                .constraints
                .iter()
                .map(|constraint| Constraint {
                    expr: substitute_symbols(&constraint.expr, replacements),
                    is_equality: constraint.is_equality,
                })
                .collect(),
        }
    }
}

/// ONE EXPRESSION WITH ITS SYMBOLS REPLACED, LEAF BY LEAF.
///
/// ⭐ NO FOLDING. `s0 * 8` with `s0 := 8` becomes `8 * 8`, not `64`: the constructors
/// ([`AffineExpr::times`], [`AffineExpr::plus`]) are the only place this island normalises, and
/// [`terms_in`] flattens a product of literals when it reads the row, so folding here would only
/// change the printed text.
fn substitute_symbols(expr: &AffineExpr, replacements: &[AffineExpr]) -> AffineExpr {
    match expr {
        AffineExpr::Sym(n) => match replacements.get(*n as usize) {
            Some(with) => with.clone(),
            None => AffineExpr::Sym(*n),
        },
        AffineExpr::Dim(_) | AffineExpr::Const(_) => expr.clone(),
        AffineExpr::Add(a, b) => AffineExpr::Add(
            Box::new(substitute_symbols(a, replacements)),
            Box::new(substitute_symbols(b, replacements)),
        ),
        AffineExpr::Mul(a, b) => AffineExpr::Mul(
            Box::new(substitute_symbols(a, replacements)),
            Box::new(substitute_symbols(b, replacements)),
        ),
        AffineExpr::Mod(a, b) => AffineExpr::Mod(
            Box::new(substitute_symbols(a, replacements)),
            Box::new(substitute_symbols(b, replacements)),
        ),
        AffineExpr::FloorDiv(a, b) => AffineExpr::FloorDiv(
            Box::new(substitute_symbols(a, replacements)),
            Box::new(substitute_symbols(b, replacements)),
        ),
    }
}

/// `ceil(n / d)` for a POSITIVE `d` — the rounding a lower bound needs.
///
/// ⛔ `/` TRUNCATES TOWARD ZERO, which is neither floor nor ceiling and rounds a negative bound the
/// wrong way. `2*d0 + 1 >= 0` means `d0 >= -1/2`, and the integers satisfying it start at 0, not at
/// `(-1)/2 == 0`… which agrees here and disagrees at `3*d0 + 1 >= 0`, where truncation gives 0 and
/// the ceiling of `-1/3` is also 0 — the divergence appears in the LOWER direction, at
/// `-3 >= 0`-style rows a caller can write. `div_euclid` floors unconditionally, so negate around it.
fn ceil_div(n: i64, d: i64) -> i64 {
    -((-n).div_euclid(d))
}

/// ONE VARIABLE OF A CONSTRAINT SYSTEM — a dimension or a symbol, told apart.
///
/// MLIR gives the two adjacent coefficient columns and indexes them by one position; this names
/// which column that position landed in, so [`IntegerSet::bound_on`] can read either.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Var {
    /// `d<n>`.
    Dim(u32),
    /// `s<n>`.
    Sym(u32),
}

/// WHAT ONE CONSTRAINT EXPRESSION SAYS ABOUT ONE DIMENSION.
///
/// ⭐ THIS IS THE `FlatAffineValueConstraints` MATRIX, INLINED. MLIR flattens every constraint into
/// a coefficient row over all variables and then eliminates every variable but `pos`; a row naming
/// only other variables survives that elimination with a zero coefficient here, which is what
/// [`Terms::OtherVars`] stands for. Every `affine_set` in the authority tree's tests constrains one
/// dimension per constraint, so the row and this enum agree on all of them.
///
/// ⛔ A SYMBOL IS ANOTHER VARIABLE OF THAT ROW, NOT A CONSTANT. MLIR's flattening gives dimensions
/// and symbols adjacent coefficient columns (`getNumDimAndSymbolVars()`), so an
/// [`AffineExpr::Sym`] is [`Terms::OtherVars`] for every [`Var::Dim`] asked about — never
/// [`Terms::Const`], which would let a parameterised bound be read as a literal one. It is
/// [`Terms::OnPos`] only when the variable asked about is that very symbol.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Terms {
    /// `c` — no dimension at all.
    Const(i64),
    /// `a * d<pos> + c` with a nonzero `a`: the constraint bounds the dimension asked about.
    OnPos {
        /// The coefficient of `d<pos>`. Its SIGN decides which side is bounded.
        coeff: i64,
        /// Everything else, moved to the constant column.
        constant: i64,
    },
    /// Some variable other than `d<pos>` — a dimension that is not `pos`, or a symbol. A row that
    /// says nothing about the dimension asked about.
    OtherVars,
}

impl Terms {
    /// `a * d<pos> + c`, degenerating to a constant when `a` is zero.
    fn on_pos(coeff: i64, constant: i64) -> Terms {
        if coeff == 0 {
            Terms::Const(constant)
        } else {
            Terms::OnPos { coeff, constant }
        }
    }
}

/// WHAT `expr` SAYS ABOUT `d<pos>`.
///
/// ⛔ MIXING `pos` WITH ANOTHER DIMENSION IS THE ONE SHAPE THIS CANNOT ANSWER, and answering
/// "unbounded" for it would be a wrong answer rather than a missing one — eliminating the other
/// variable combines the row with that variable's OPPOSING bounds, which needs the whole system.
fn terms_in(expr: &AffineExpr, var: Var) -> Terms {
    match expr {
        AffineExpr::Dim(n) if Var::Dim(*n) == var => Terms::OnPos {
            coeff: 1,
            constant: 0,
        },
        AffineExpr::Sym(n) if Var::Sym(*n) == var => Terms::OnPos {
            coeff: 1,
            constant: 0,
        },
        // ⛔ A SYMBOL IS ANOTHER VARIABLE, NOT A CONSTANT. `d0 + s0 * 8 - 64 >= 0` bounds `d0` only
        // once `s0` is known, and MLIR's own answer is the same: `getConstantBound` reads the
        // flattened matrix, where a symbol column is as unresolved as another dimension's. Entry 089
        // never asks: it SUBSTITUTES the symbol first ([`IntegerSet::replace_symbols`]) and only then
        // reads a bound, which is exactly why `replaceDimsAndSymbols` exists in that function.
        AffineExpr::Dim(_) | AffineExpr::Sym(_) => Terms::OtherVars,
        AffineExpr::Const(c) => Terms::Const(*c),
        AffineExpr::Add(a, b) => match (terms_in(a, var), terms_in(b, var)) {
            (Terms::Const(x), Terms::Const(y)) => Terms::Const(x + y),
            (Terms::Const(c), Terms::OnPos { coeff, constant })
            | (Terms::OnPos { coeff, constant }, Terms::Const(c)) => {
                Terms::on_pos(coeff, constant + c)
            }
            (
                Terms::OnPos {
                    coeff: a_coeff,
                    constant: a_const,
                },
                Terms::OnPos {
                    coeff: b_coeff,
                    constant: b_const,
                },
            ) => Terms::on_pos(a_coeff + b_coeff, a_const + b_const),
            (Terms::OtherVars, Terms::Const(_) | Terms::OtherVars)
            | (Terms::Const(_), Terms::OtherVars) => Terms::OtherVars,
            (Terms::OtherVars, Terms::OnPos { .. }) | (Terms::OnPos { .. }, Terms::OtherVars) => {
                todo!(
                    "a constant bound on {var:?} from a constraint that also mentions another variable needs \
                     the Fourier-Motzkin elimination FlatAffineValueConstraints::projectOut runs"
                )
            }
        },
        AffineExpr::Mul(a, b) => match (terms_in(a, var), terms_in(b, var)) {
            (Terms::Const(x), Terms::Const(y)) => Terms::Const(x * y),
            (Terms::Const(k), Terms::OnPos { coeff, constant })
            | (Terms::OnPos { coeff, constant }, Terms::Const(k)) => {
                Terms::on_pos(coeff * k, constant * k)
            }
            // ⭐ `0 * d1` IS ZERO, not "some other dimension" — the row loses the variable.
            (Terms::Const(0), Terms::OtherVars) | (Terms::OtherVars, Terms::Const(0)) => {
                Terms::Const(0)
            }
            (Terms::Const(_), Terms::OtherVars) | (Terms::OtherVars, Terms::Const(_)) => {
                Terms::OtherVars
            }
            (Terms::OnPos { .. } | Terms::OtherVars, Terms::OnPos { .. } | Terms::OtherVars) => {
                todo!("a constraint that multiplies two dimensions is not affine")
            }
        },
        AffineExpr::Mod(a, b) => match (terms_in(a, var), terms_in(b, var)) {
            (Terms::Const(n), Terms::Const(d)) => Terms::Const(n.rem_euclid(d)),
            _ => todo!(
                "a constant bound across a `mod` needs the local variable MLIR's affine flattening \
                 introduces for it"
            ),
        },
        AffineExpr::FloorDiv(a, b) => match (terms_in(a, var), terms_in(b, var)) {
            (Terms::Const(n), Terms::Const(d)) => Terms::Const(n.div_euclid(d)),
            _ => todo!(
                "a constant bound across a `floordiv` needs the local variable MLIR's affine \
                 flattening introduces for it"
            ),
        },
    }
}

/// An `affine_map<(d0, ..)[s0, ..] -> (..)>`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AffineMap {
    /// How many dimensions it takes.
    pub dims: u32,
    /// How many SYMBOLS it takes.
    ///
    /// ⛔⛔ A SEPARATE COUNT, NOT DERIVED FROM THE RESULTS. `AffineMap::get(numDims, numSymbols, ..)`
    /// carries both arities, and a symbol a map DECLARES but never mentions is a real map: the
    /// output of `TPMVBase::replaceDimsInMapWithSyms` (`TransformPagedMemViewImpl.cpp:47`) declares
    /// one symbol per dimension of the ORIGINAL map, and a subscript that ignored one of its
    /// iterators leaves the matching `s<N>` unused. Recomputing this from the largest
    /// [`AffineExpr::Sym`] present would silently narrow such a map, and every constraint row built
    /// from it would then be one column short.
    ///
    /// ⭐ ZERO FOR EVERY MAP THIS BRIDGE EMITS INTO A PROGRAM. Symbols exist here for the
    /// constraint systems `TransformPagedMemView` solves; a printed `affine_map` in an emitted
    /// DataflowIR program has none, which is why `syms: 0` prints exactly what it printed before
    /// this field existed.
    pub syms: u32,
    /// What it produces — one expression per result.
    pub results: Vec<AffineExpr>,
}

impl AffineMap {
    /// A one-dimensional, one-result map.
    #[must_use]
    pub fn unary(expr: AffineExpr) -> AffineMap {
        AffineMap {
            dims: 1,
            syms: 0,
            results: vec![expr],
        }
    }

    /// `(d0, .., dn) -> (d0, .., dn)` — the `load_order`/`store_order` of an access.
    ///
    /// ⭐ ORDER SAYS WHICH AXIS MOVES FASTEST, and the scheduler writes the identity for every
    /// access in its own output (`#map2` over three dims, `#map4` over five). Row-major order is
    /// what the view's `layout_map` already states, so ordering it again differently would be two
    /// answers to one question.
    #[must_use]
    pub fn identity(rank: u32) -> AffineMap {
        AffineMap {
            dims: rank,
            syms: 0,
            results: (0..rank).map(AffineExpr::dim).collect(),
        }
    }

    /// THE ONE LITERAL THIS MAP PRODUCES, if a literal is all it produces.
    ///
    /// ⭐ `isSingleConstant()` AND `getSingleConstantResult()` AS ONE CALL, because the reference
    /// never asks the first without immediately taking the second — `getOperandFromLoadOrStoreOp`
    /// (entry 232) canonicalizes a fully-composed access map and reads the offset out of it.
    #[must_use]
    pub fn single_constant(&self) -> Option<i64> {
        match self.results.as_slice() {
            [AffineExpr::Const(offset)] => Some(*offset),
            _ => None,
        }
    }

    /// IS THIS THE IDENTITY? `(d0, .., dn) -> (d0, .., dn)` and nothing else.
    ///
    /// ⛔ MLIR'S OWN PREDICATE, TRANSCRIBED. `AffineMap::isIdentity()` requires as many results as
    /// dimensions and result `i` to be exactly `d<i>` — so `(d0, d1) -> (d0)` is NOT the identity
    /// even though it drops nothing but a dimension, and `(d0) -> (d0 * 1)` is not one either
    /// because the expression is a `Mul`. `checkIndirectMemViewForExtractOp`
    /// (`Helper.cpp:428-431`) pairs it with `getNumDims() != 1` to insist an indirect memory view is
    /// a FLAT, UNPERMUTED window: `expecting 1D identity map for layout_map`.
    #[must_use]
    pub fn is_identity(&self) -> bool {
        usize::try_from(self.dims).is_ok_and(|dims| dims == self.results.len())
            && self
                .results
                .iter()
                .enumerate()
                .all(|(i, r)| u32::try_from(i).is_ok_and(|i| *r == AffineExpr::Dim(i)))
    }

    /// `(d0, .., dm) -> (r0, .., rn)` where every result is a literal — the `*_time_addr_map` of a
    /// transfer that walks nothing.
    ///
    /// ⭐ THE ARITIES DIFFER, WHICH IS THE POINT: it takes one dimension per TIME step and produces
    /// one offset per MEMREF dimension. `#map3 = affine_map<(d0) -> (0, 0, 0)>` is a one-step walk
    /// over a rank-three source; `#map5` is the same step over the rank-five destination.
    #[must_use]
    pub fn constants(dims: u32, results: &[i64]) -> AffineMap {
        AffineMap {
            dims,
            syms: 0,
            results: results.iter().copied().map(AffineExpr::Const).collect(),
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
            syms: 0,
            results: vec![expr],
        }
    }

    /// `self ∘ inner` — `AffineMap::compose(AffineMap map)`, this map applied to the other's results.
    ///
    /// ⛔⛔ THE ARGUMENT IS THE INNER MAP AND THE RESULT TAKES *ITS* DIMENSIONS. `self`'s `d<i>`
    /// becomes `inner.results[i]`, so the composite is indexed by whatever `inner` is indexed by:
    /// `(d0,d1) -> (d0*64 + d1)` composed with `(d0,d1,d2) -> (d1, d2)` is
    /// `(d0,d1,d2) -> (d1*64 + d2)`. Reading the operand as the outer map would compose the pair
    /// backwards and address the wrong element of every view.
    ///
    /// ⛔ THE ARITY PRECONDITION IS MLIR'S `assert(getNumDims() == map.getNumResults())`, and here it
    /// is a `debug_assert`-free TOTAL function instead: a mismatched position simply keeps itself
    /// (see [`AffineExpr::replace_dims_and_symbols`]), because this crate never runtime-refuses.
    /// ⭐ THE CALLER IS WHAT GUARANTEES IT — `getLayoutMapAndIndices` composes a view's `layout_map`
    /// with the access's own indices map, and the view is the memref the access indexes.
    ///
    /// ⛔ THE SYMBOLS CONCATENATE, OUTER FIRST. The result declares `self.syms + inner.syms`; the
    /// outer map's symbols keep positions `0..self.syms` because `compose` passes no replacement for
    /// them, and the inner map's `s<j>` is SHIFTED to `s<self.syms + j>`. Both maps numbering from
    /// zero would make two different values one variable.
    ///
    /// ⭐ THE TWO COMPOSES ELSEWHERE IN THE CAMPAIGN ARE `mem_view_layout_map.compose(time_addr_map)`
    /// (`dialect_utils/Agen/Utils.cpp:103`) and `transfer_order.compose(subscripts_map)`
    /// (`dataflow-scheduler-dialects/lib/Dialect/Agen/Utils.cpp:57-60`). Both of those maps are
    /// symbol-free, so the shift is unobservable there — and dropping it would make this function
    /// wrong for the paged-view maps, which are not.
    #[must_use]
    pub fn compose(&self, inner: &AffineMap) -> AffineMap {
        let shifted_syms: Vec<AffineExpr> = (0..inner.syms)
            .map(|j| AffineExpr::Sym(self.syms + j))
            .collect();
        // `newDims[idx] = getAffineDimExpr(idx)` — the identity, so the inner map's dimensions are
        // rewritten to themselves and only its symbols move.
        let inner_dims: Vec<AffineExpr> = (0..inner.dims).map(AffineExpr::Dim).collect();
        let rewritten: Vec<AffineExpr> = inner
            .results
            .iter()
            .map(|r| r.replace_dims_and_symbols(&inner_dims, &shifted_syms))
            .collect();
        AffineMap {
            dims: inner.dims,
            syms: self.syms + inner.syms,
            // `expr.compose(newMap)` is `replaceDims(newMap.getResults())` — dimensions only, so the
            // outer map's own symbols are the ones left in place.
            results: self
                .results
                .iter()
                .map(|r| r.replace_dims_and_symbols(&rewritten, &[]))
                .collect(),
        }
    }

    /// THIS MAP APPLIED TO A VECTOR OF LITERALS — `AffineMap::compose(ArrayRef<int64_t> values)`,
    /// which is how `calculateTimeBounds` and `calculateTimeOffsets` REORDER their vectors through
    /// `time_order`: `time_bounds = time_order.compose(time_bounds)`
    /// (`dialect_utils/Agen/Utils.cpp:255`), `time_offsets = time_order.compose(tmp_non_const)`
    /// (`:115`).
    ///
    /// ⭐ FOR A PERMUTATION IT IS A GATHER: result `i` is `values[j]` where this map's result `i` is
    /// `d<j>`. Nothing here assumes one, though — MLIR substitutes the literals and folds, so a map
    /// with arithmetic in it evaluates.
    ///
    /// ⛔ MLIR'S `assert(getNumInputs() == values.size())` IS A MISSING INPUT READ AS ZERO HERE, and
    /// a surplus one dropped, because this crate never runtime-refuses. A result that still mentions
    /// a SYMBOL after the substitution has no literal value and comes back as 0 for the same reason;
    /// [`Self::syms`] is zero for every map this is called on.
    #[must_use]
    pub fn compose_constants(&self, values: &[i64]) -> Vec<i64> {
        let literals = AffineMap {
            dims: 0,
            syms: 0,
            results: (0..self.dims as usize)
                .map(|i| AffineExpr::Const(values.get(i).copied().unwrap_or(0)))
                .collect(),
        };
        let composed = self.compose(&literals);
        composed
            .results
            .iter()
            .map(|r| {
                r.flatten(composed.dims, composed.syms)
                    .as_constant()
                    .unwrap_or(0)
            })
            .collect()
    }

    /// DROP THE SYMBOLS NOBODY MENTIONS AND RENUMBER THE REST DENSELY — `compressUnusedSymbols`.
    ///
    /// ⛔ IT IS `projectSymbols` WITH `compressSymbolsFlag=true` (`AffineMap.cpp:724-731`): each
    /// symbol position is either replaced by `s<newPos>` with `newPos` counting only the KEPT ones,
    /// or — for a projected one — by the constant `0`. Only unused symbols are projected here, so no
    /// zero can appear in the result; the renumbering is the whole observable effect.
    ///
    /// ⭐ A NO-OP FOR EVERY MAP THIS BRIDGE COMPOSES, AND THAT IS WORTH SAYING RATHER THAN OMITTING
    /// THE CALL. `getLayoutMapAndIndices` ends with it, and both of its inputs are printed maps with
    /// [`syms`](AffineMap::syms)` == 0` — see that field's own note — so the composite has none to
    /// compress. The reference calls it because a `get_logical_memory_view` MAY carry a parameterised
    /// layout; leaving it out would make this port right only for the maps we happen to emit today.
    #[must_use]
    pub fn compress_unused_symbols(&self) -> AffineMap {
        let used: Vec<bool> = (0..self.syms)
            .map(|j| self.results.iter().any(|r| mentions_symbol(r, j)))
            .collect();
        let mut next = 0;
        let replacements: Vec<AffineExpr> = used
            .iter()
            .map(|keep| {
                if *keep {
                    let at = next;
                    next += 1;
                    AffineExpr::Sym(at)
                } else {
                    AffineExpr::Const(0)
                }
            })
            .collect();
        AffineMap {
            dims: self.dims,
            syms: next,
            results: self
                .results
                .iter()
                // ⛔ SYMBOLS ONLY. `projectCommonImpl` calls `e.replaceSymbols(replacements)` for the
                // symbol instantiation, so the dimensions are untouched and keep their positions.
                .map(|r| r.replace_dims_and_symbols(&[], &replacements))
                .collect(),
        }
    }
}

/// DOES ANY LEAF OF `expr` NAME `s<j>`? — the bit `getUnusedSymbolsBitVector` clears.
fn mentions_symbol(expr: &AffineExpr, j: u32) -> bool {
    match expr {
        AffineExpr::Sym(n) => *n == j,
        AffineExpr::Dim(_) | AffineExpr::Const(_) => false,
        AffineExpr::Add(a, b)
        | AffineExpr::Mul(a, b)
        | AffineExpr::Mod(a, b)
        | AffineExpr::FloorDiv(a, b) => mentions_symbol(a, j) || mentions_symbol(b, j),
    }
}

// ═══════════════════════════════════════════════════════════════════════════════════════════════
// THE AFFINE ALGEBRA A TRANSFER'S GEOMETRY IS DERIVED WITH.
//
// ⭐ THE ISLAND WIDENED HERE BECAUSE `constructExtentAndTotalElements` CANNOT BE WRITTEN WITHOUT IT.
// That function is four MLIR calls before it looks at anything —
// `FlatAffineValueConstraints(set)`, `composeMatchingMap(order)`, `projectOut(..)`,
// `removeRedundantConstraints()` (`dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:33-40`) —
// and its answer is read back out of the resulting MATRIX by two predicates that scan coefficient
// ROWS (`dialect_utils/Agen/Utils.cpp:121-170`). An `IntegerSet` holds expressions, not rows, so the
// matrix is a type of its own rather than a reading of the set.
// ═══════════════════════════════════════════════════════════════════════════════════════════════

/// ONE AFFINE EXPRESSION AS A COEFFICIENT ROW — `getFlattenedAffineExpr(expr, numDims, numSymbols,
/// &coeffs, &csts)` (`mlir/Dialect/Affine/Analysis/AffineStructures.h`), which
/// `agen::utils::getMapCoefficients` (`dialect_utils/Agen/Utils.cpp:65-71`) and
/// `constructDimCoefficients` (`:78-95`) are both thin wrappers around.
///
/// ⛔⛔ THE LAST ENTRY IS THE CONSTANT TERM, NOT A VARIABLE, and every consumer in this campaign
/// depends on knowing that: `LoweringXRF.cpp:50` asserts `layout_coeffs.size() == operands.size() +
/// 1`, `calculateTimeOffsets` splits the row at `size() - 1` and re-appends `back()`
/// (`Utils.cpp:112-116`), and `constructIteratorCoeffDict` reads `coeffs.back()` as the offset
/// (`dataflow-scheduler-dialects/lib/Dialect/Agen/Utils.cpp:73-76`). So the row is
/// `[d0 .. dn, s0 .. sm, constant]`: `d0 * 4 + d1 + 7` over two dimensions and no symbols is
/// `[4, 1, 7]`.
///
/// ⛔ NO LOCAL VARIABLES. MLIR's flattener introduces one extra column per `mod`, `floordiv` or
/// `ceildiv` it meets, together with the inequalities that define it; nothing this bridge builds has
/// one, so those two arms are [`todo!`]s that name the missing machinery rather than a silent
/// mis-flattening. [`terms_in`] already declines the same two shapes for the same reason.
fn flattened(expr: &AffineExpr, dims: u32, syms: u32) -> Vec<i64> {
    let vars = dims as usize + syms as usize;
    let mut row = vec![0_i64; vars + 1];
    match expr {
        AffineExpr::Dim(n) => match usize::try_from(*n) {
            Ok(pos) if pos < dims as usize => row[pos] = 1,
            _ => todo!(
                "flattening d{n} against a map that declares {dims} dimensions — MLIR's own \
                 getFlattenedAffineExpr asserts the position is inside the space it is given"
            ),
        },
        AffineExpr::Sym(n) => match usize::try_from(*n) {
            Ok(pos) if pos < syms as usize => row[dims as usize + pos] = 1,
            _ => todo!(
                "flattening s{n} against a map that declares {syms} symbols — MLIR's own \
                 getFlattenedAffineExpr asserts the position is inside the space it is given"
            ),
        },
        AffineExpr::Const(c) => row[vars] = *c,
        AffineExpr::Add(a, b) => {
            let (lhs, rhs) = (flattened(a, dims, syms), flattened(b, dims, syms));
            for (slot, (x, y)) in row.iter_mut().zip(lhs.iter().zip(rhs.iter())) {
                *slot = x + y;
            }
        }
        // ⛔ ONE SIDE MUST BE A LITERAL. `SimpleAffineExprFlattener::visitMulExpr` requires the RHS
        // to flatten to a constant and reports failure otherwise — a product of two variables is
        // SEMI-affine, which no `AffineMap` in the reference's IR is.
        AffineExpr::Mul(a, b) => {
            let (lhs, rhs) = (flattened(a, dims, syms), flattened(b, dims, syms));
            let literal = |flat: &[i64]| {
                flat[..vars]
                    .iter()
                    .all(|coeff| *coeff == 0)
                    .then(|| flat[vars])
            };
            match (literal(&lhs), literal(&rhs)) {
                (_, Some(k)) => {
                    for (slot, x) in row.iter_mut().zip(lhs.iter()) {
                        *slot = x * k;
                    }
                }
                (Some(k), None) => {
                    for (slot, y) in row.iter_mut().zip(rhs.iter()) {
                        *slot = y * k;
                    }
                }
                (None, None) => todo!("a product of two affine variables is not an affine map"),
            }
        }
        AffineExpr::Mod(..) => todo!(
            "flattening a `mod` needs the local variable and the two inequalities MLIR's \
             SimpleAffineExprFlattener introduces for it"
        ),
        AffineExpr::FloorDiv(..) => todo!(
            "flattening a `floordiv` needs the local variable and the two inequalities MLIR's \
             SimpleAffineExprFlattener introduces for it"
        ),
    }
    row
}

impl AffineMap {
    /// THIS MAP'S RESULT `result` AS A COEFFICIENT ROW — `agen::utils::getMapCoefficients(coeffs,
    /// map, result)` (`dialect_utils/Agen/Utils.cpp:65-71`).
    ///
    /// ⛔ ONE COEFFICIENT PER DIMENSION AND SYMBOL, PLUS THE CONSTANT TERM LAST — see [`flattened`].
    /// The layout map of a rank-`n` view therefore gives `n + 1` coefficients, which is the
    /// relation `LoweringXRF.cpp:50` asserts and the reason
    /// `constructExtentAndTotalElements` compares `layout_coeffs.size()` against the constraint
    /// system's DIMENSION count rather than assuming they are equal
    /// (`AccessDetails.cpp:47-50`).
    ///
    /// ⭐ AN ABSENT RESULT GIVES AN EMPTY ROW, where MLIR's `map.getResult(result)` would read past
    /// the end. Nothing in the reference asks for one: every caller passes 0 on a map whose single
    /// result is the linearised address.
    #[must_use]
    pub fn coefficients(&self, result: usize) -> Vec<i64> {
        match self.results.get(result) {
            Some(expr) => flattened(expr, self.dims, self.syms),
            None => Vec::new(),
        }
    }

    /// DIMENSIONS AND SYMBOLS SUBSTITUTED, WITH THE RESULT'S ARITIES STATED —
    /// `AffineMap::replaceDimsAndSymbols(dimReplacements, symReplacements, numResultDims,
    /// numResultSyms)`.
    ///
    /// ⛔⛔ THE NEW ARITIES ARE THE CALLER'S, NOT THE SUBSTITUTION'S, and that is what
    /// `constructIndices` uses it for: it rewrites the subscripts map's operand list so that the
    /// dimensions it FOLDED become constants and the ones it KEPT are renumbered densely, then
    /// declares the surviving count — `subscripts_map.replaceDimsAndSymbols(operand_exprs,
    /// symbol_exprs, ndims, 0)` with `ndims` counted as it went
    /// (`AccessDetails.cpp:385-401`). A map that inferred its own arity from the expressions left in
    /// it would keep the folded dimensions' columns and every constraint row built from it would be
    /// too wide.
    #[must_use]
    pub fn replace_dims_and_symbols(
        &self,
        dim_repl: &[AffineExpr],
        sym_repl: &[AffineExpr],
        result_dims: u32,
        result_syms: u32,
    ) -> AffineMap {
        AffineMap {
            dims: result_dims,
            syms: result_syms,
            // ⛔⛔ AND IT SIMPLIFIES, WHICH IS [`AffineExpr::replace_dims_and_symbols`]'S JOB AND NOT
            // A COURTESY. MLIR's map-level form is `llvm::map_range(getResults(), [](AffineExpr e) {
            // return e.replaceDimsAndSymbols(..); })` handed to `AffineMap::get`, so every rebuilt
            // node goes through `getAffineBinaryOpExpr` — and the vendor's own answer key is the
            // proof: `constructIndices` folds two of five dimensions to `0` in
            // `mutable_addr_splitting_time_dims.mlir` and the attribute it writes is
            // `(d0, d1, d2) -> (d2 * 64, d0 * 16, d1 * 8)`, not `d0 * 16 + 0` and `d1 * 8 + 0 * 8`.
            results: self
                .results
                .iter()
                .map(|expr| expr.replace_dims_and_symbols(dim_repl, sym_repl))
                .collect(),
        }
    }

    /// `AffineMap::shiftDims(shift)` — RENUMBER EVERY DIMENSION UP BY `shift` AND WIDEN THE MAP TO
    /// MATCH.
    ///
    /// ```cpp
    /// AffineMap shiftDims(unsigned shift, unsigned offset = 0) const {
    ///   assert(offset <= getNumDims());
    ///   return AffineMap::get(getNumDims() + shift, getNumSymbols(),
    ///       llvm::map_range(getResults(), [&](AffineExpr e) {
    ///         return e.shiftDims(getNumDims(), shift, offset);
    ///       }), getContext());
    /// }
    /// ```
    /// (`llvm-project/mlir/include/mlir/IR/AffineMap.h:311-318`)
    ///
    /// ⭐⭐ THE ARITY GROWS, WHICH IS THE POINT: the shifted map takes the ORIGINAL map's dimensions
    /// too, it just ignores them. `concatenateMaps` shifts the second map past the first so the sum
    /// of the two can be indexed by both sets of iterators at once
    /// (`dialect_utils/Agen/Utils.cpp:258-266`) — `(d0, d1, d2) -> (d0 * 64, d1, d2 * 8)` over a
    /// two-dimensional first map becomes `(d0, .., d4) -> (d2 * 64, d3, d4 * 8)`.
    ///
    /// ⭐ SYMBOLS ARE UNTOUCHED, because they are numbered in their own space
    /// (see [`AffineExpr::Sym`]).
    #[must_use]
    pub fn shift_dims(&self, shift: u32) -> AffineMap {
        AffineMap {
            dims: self.dims.saturating_add(shift),
            syms: self.syms,
            results: self
                .results
                .iter()
                .map(|expr| expr.shifted_dims(self.dims, shift))
                .collect(),
        }
    }

    /// `AffineMap::isFunctionOfDim(position)` — DOES ANY RESULT OF THIS MAP MENTION `d<position>`?
    ///
    /// ```cpp
    /// bool isFunctionOfDim(unsigned position) const {
    ///   return llvm::any_of(getResults(),
    ///                       [&](AffineExpr e) { return e.isFunctionOfDim(position); });
    /// }
    /// ```
    /// (`llvm-project/mlir/include/mlir/IR/AffineMap.h:344-347`)
    ///
    /// ⭐⭐ THE QUESTION THAT DECIDES WHETHER A TIME DIMENSION BECOMES A LOOP OR A ZERO.
    /// `updateSubscriptsAndIndicesForExplicitTimeLoops` binds a constant 0 for every dimension of the
    /// concatenated map the map does not actually use (`Dialect/Agen/Utils.cpp:466`), and
    /// `updateTimeSetForExplicitDims` keeps only those constraints that are a function of no
    /// now-explicit time dimension (`:503`). Both are "this map does not move with that iterator";
    /// see [`AffineExpr::is_function_of_dim`] for why a symbol never counts.
    #[must_use]
    pub fn is_function_of_dim(&self, position: u32) -> bool {
        self.results
            .iter()
            .any(|expr| expr.is_function_of_dim(position))
    }

    /// `AffineMap::getDimPosition(idx)` — WHICH DIMENSION RESULT `idx` IS, when the result is a bare
    /// dimension and nothing else.
    ///
    /// ```cpp
    /// unsigned AffineMap::getDimPosition(unsigned idx) const {
    ///   return llvm::cast<AffineDimExpr>(getResult(idx)).getPosition();
    /// }
    /// ```
    /// (`llvm-project/mlir/lib/IR/AffineMap.cpp:319-321`)
    ///
    /// ⛔ `None` IS WHERE THE REFERENCE ABORTS. `cast` is not `dyn_cast`: a result that is not a bare
    /// `d<n>` kills the compiler, so every caller is asserting the map is a PERMUTATION. This
    /// campaign's one caller, `updateTimeSetForExplicitDims`, asks it of a `time_order`
    /// (`Dialect/Agen/Utils.cpp:503`), and every `*_time_order` in the authority tree's tests is a
    /// permutation of its dimensions — `#map = affine_map<(d0, d1, d2) -> (d2, d1, d0)>`
    /// (`dcc/test/Transform/MutableAddrSplitting/mutable_addr_splitting_time_dims.mlir:5`).
    /// Reporting the shape instead of aborting is what lets the port stay total.
    #[must_use]
    pub fn dim_position(&self, idx: usize) -> Option<u32> {
        match self.results.get(idx) {
            Some(AffineExpr::Dim(position)) => Some(*position),
            _ => None,
        }
    }

    /// `AffineMap::walkExprs(callback)` — EVERY SUBEXPRESSION OF EVERY RESULT, POST-ORDER
    /// (`llvm-project/mlir/lib/IR/AffineMap.cpp:475-477`, `AffineExpr::walk`).
    ///
    /// ⭐ `constructChunkAndShuffleInfo` READS A SELECT MAP THROUGH IT (`AccessDetails.cpp:227-236`):
    /// any `Mod` anywhere makes the load a splat, and any node that is neither `Mod`, a constant nor a
    /// bare `d<n>` refuses the map — which is why the walk has to reach INSIDE each result and not
    /// just visit its root.
    pub fn walk_exprs(&self, callback: &mut impl FnMut(&AffineExpr)) {
        for result in &self.results {
            result.walk(callback);
        }
    }

    /// THE INVERSE OF A PERMUTATION MAP, or [`None`] when the map is not one —
    /// `mlir::inversePermutation` (`llvm-project/mlir/lib/IR/AffineMap.cpp:784-806`).
    ///
    /// ⭐⭐ ONLY ITS EXISTENCE IS EVER READ. `checkBasicConditions` writes
    /// `if (!inversePermutation(data_order))` and refuses the access (`Helper.cpp:143-152`); the
    /// inverse itself is dropped.
    ///
    /// ⛔ A NON-`Dim` RESULT IS **SKIPPED**, NOT REFUSED, and the count at the end is what catches it:
    /// every input must have been named exactly once, so `(d0, d1) -> (d0, d0)` and
    /// `(d0, d1) -> (d0 + 1, d1)` both leave an input unnamed and answer [`None`].
    /// ⛔ THE EMPTY MAP IS ITS OWN INVERSE (`:785-786`), so `affine_map<() -> ()>` passes the test.
    /// ⛔ SYMBOLS ARE AN `assert` THERE (`:787`); a declared symbol is an input no result can name, so
    /// the count refuses the map here instead of aborting.
    #[must_use]
    pub fn inverse_permutation(&self) -> Option<AffineMap> {
        // `map.isEmpty()` — `() -> ()` (`:785`).
        if self.dims == 0 && self.syms == 0 && self.results.is_empty() {
            return Some(self.clone());
        }
        let mut exprs: Vec<Option<AffineExpr>> = vec![None; self.dims as usize];
        for (at, result) in self.results.iter().enumerate() {
            // `:791-796` — the first result naming a position wins; a later one is skipped.
            if let AffineExpr::Dim(position) = result
                && let Some(slot) = exprs.get_mut(*position as usize)
                && slot.is_none()
            {
                *slot = Some(AffineExpr::dim(
                    u32::try_from(at).expect("a rank fits a u32"),
                ));
            }
        }
        let seen: Vec<AffineExpr> = exprs.into_iter().flatten().collect();
        // `:803-804` — `seenExprs.size() != map.getNumInputs()`, and inputs are dims PLUS symbols.
        if seen.len() != self.dims as usize + self.syms as usize {
            return None;
        }
        Some(AffineMap {
            dims: u32::try_from(self.results.len()).expect("a rank fits a u32"),
            syms: 0,
            results: seen,
        })
    }
}

/// AN INTEGER SET AS A COEFFICIENT MATRIX — `mlir::affine::FlatAffineValueConstraints`.
///
/// # ⭐⭐ THE ROWS ARE WHAT THE REFERENCE READS, NOT THE EXPRESSIONS
///
/// `isDimValueZero` and `isDimAConstantRange` (`dialect_utils/Agen/Utils.cpp:121-170`) walk
/// `atEq(r, c)` / `atIneq(r, c)` counting the NONZERO COLUMNS of a row, and
/// `constructExtentAndTotalElements` reaches them only after `composeMatchingMap` and `projectOut`
/// have rewritten the system into a space the original set has no dimensions in
/// (`AccessDetails.cpp:33-40`). Neither step is expressible on [`IntegerSet`], whose constraints are
/// expression trees.
///
/// ⛔ COLUMN ORDER IS MLIR'S: every DIMENSION, then every SYMBOL, then the CONSTANT term — one row
/// per constraint, `getNumCols() == getNumDimVars() + getNumSymbolVars() + 1`. Both predicates
/// scan `c < getNumCols() - 1`, i.e. the variables only, and then read the constant at
/// `getNumCols() - 1`, so a layout that put the constant anywhere else would change both answers.
///
/// ⛔ NO LOCAL-VARIABLE COLUMNS, for the reason [`flattened`] gives: nothing here builds a `mod` or
/// a `floordiv`, and MLIR's local columns sit between the symbols and the constant — so admitting
/// them later is a widening of this type rather than a reinterpretation of it.
///
/// ⛔ AND A ROW IS `expr >= 0` OR `expr == 0`, NEVER `<= 0`. `IntegerSet` states its inequalities
/// that way ([`Constraint`]), which is why an upper bound arrives as `-dk + n >= 0` and
/// `isDimAConstantRange` recognises an upper bound by a coefficient of **-1**.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FlatConstraints {
    /// How many DIMENSION columns — `getNumDimVars()`.
    pub dims: u32,
    /// How many SYMBOL columns — `getNumSymbolVars()`.
    pub syms: u32,
    /// The `== 0` rows, in order — `getNumEqualities()` of them.
    pub equalities: Vec<Vec<i64>>,
    /// The `>= 0` rows, in order — `getNumInequalities()` of them.
    pub inequalities: Vec<Vec<i64>>,
}

impl FlatConstraints {
    /// THE SYSTEM AN INTEGER SET FLATTENS INTO — `FlatAffineValueConstraints(IntegerSet)`, the
    /// constructor `constructExtentAndTotalElements` opens with (`AccessDetails.cpp:33-34`) and
    /// `calculateTimeBounds` too (`dialect_utils/Agen/Utils.cpp:220`).
    #[must_use]
    pub fn from_integer_set(set: &IntegerSet) -> FlatConstraints {
        let mut equalities = Vec::new();
        let mut inequalities = Vec::new();
        for constraint in &set.constraints {
            let row = flattened(&constraint.expr, set.dims, set.symbols);
            if constraint.is_equality {
                equalities.push(row);
            } else {
                inequalities.push(row);
            }
        }
        FlatConstraints {
            dims: set.dims,
            syms: set.symbols,
            equalities,
            inequalities,
        }
    }

    /// `getNumCols()` — one per variable, plus the constant.
    #[must_use]
    pub fn num_cols(&self) -> usize {
        self.dims as usize + self.syms as usize + 1
    }

    /// `getNumDimVars()`, which `constructExtentAndTotalElements` reads TWICE: to decide whether the
    /// layout map has one coefficient more than the system has dimensions (`AccessDetails.cpp:47-50`)
    /// and as the loop bound over the extents (`:56`).
    #[must_use]
    pub fn num_dim_vars(&self) -> u32 {
        self.dims
    }

    /// A MAP COMPOSED ONTO THIS SYSTEM'S DIMENSIONS — `composeMatchingMap(AffineMap other)`.
    ///
    /// ⭐⭐ WHAT IT DOES IS RE-EXPRESS THE SET IN THE MAP'S **INPUT** SPACE. `other`'s inputs are
    /// inserted as new dimensions at position 0 (`insertDimVar(0, other.getNumDims())`), and one
    /// equality per result ties each ORIGINAL dimension to the expression that produces it —
    /// `flat_expr - d_i == 0`. The original dimensions are then dead weight that
    /// [`Self::project_out`] removes, which is exactly the pair of calls
    /// `constructExtentAndTotalElements` makes (`AccessDetails.cpp:35-40`).
    ///
    /// ⛔ THE NEW DIMENSIONS COME FIRST, so after this call column `i` is the map's `i`-th input and
    /// column `other.dims + i` is the set's own `i`-th dimension. `projectOut(order.getNumDims(),
    /// order.getNumDims())` reads exactly that layout: it starts at the first original dimension and
    /// removes as many as the order has results.
    ///
    /// ⛔ MLIR ASSERTS `other.getNumResults() == getNumDimVars()` AND THAT ASSERTION IS NOT A
    /// FORMALITY — it is what makes the composed system square. Here a shorter result list simply
    /// leaves the surplus original dimensions untied, and [`Self::project_out`] then eliminates them
    /// by Fourier–Motzkin instead of by substitution; the extents that come out are a
    /// non-hyper-rectangular system, which is the one shape
    /// `constructExtentAndTotalElements` already has a refusal for. A longer one is impossible: a
    /// result past the last original dimension has nothing to be tied to and is dropped.
    ///
    /// ⛔ THE SYMBOL SPACES ARE ASSUMED ALIGNED, which is MLIR's own precondition —
    /// `flattenAlignedMapAndMergeLocals` requires the map's symbols to be the system's symbols. Both
    /// are zero everywhere this is used (`buildIntegerSetFromSizes` passes `numSymbols=0`, and a
    /// transfer order is a permutation), and the wider of the two counts is kept so a symbol column
    /// is never silently dropped.
    #[must_use]
    pub fn compose_matching_map(&self, other: &AffineMap) -> FlatConstraints {
        let inserted = other.dims as usize;
        let old_dims = self.dims as usize;
        let dims = self.dims + other.dims;
        let syms = self.syms.max(other.syms);
        let width = dims as usize + syms as usize + 1;
        let shift = |row: &Vec<i64>| -> Vec<i64> {
            let mut out = vec![0_i64; width];
            out[inserted..inserted + old_dims].copy_from_slice(&row[..old_dims]);
            for j in 0..self.syms as usize {
                out[dims as usize + j] = row[old_dims + j];
            }
            out[width - 1] = row[self.num_cols() - 1];
            out
        };
        let mut equalities: Vec<Vec<i64>> = self.equalities.iter().map(shift).collect();
        let inequalities: Vec<Vec<i64>> = self.inequalities.iter().map(shift).collect();

        for (i, expr) in other.results.iter().enumerate().take(old_dims) {
            let flat = flattened(expr, other.dims, other.syms);
            let mut row = vec![0_i64; width];
            row[..inserted].copy_from_slice(&flat[..inserted]);
            for j in 0..other.syms as usize {
                row[dims as usize + j] = flat[inserted + j];
            }
            row[width - 1] = flat[flat.len() - 1];
            // `eq[other.getNumDims() + i] = -1` — the system's own dimension `i`, now shifted right
            // by the inserted inputs.
            row[inserted + i] = -1;
            equalities.push(row);
        }

        FlatConstraints {
            dims,
            syms,
            equalities,
            inequalities,
        }
    }

    /// `num` VARIABLES ELIMINATED, STARTING AT COLUMN `pos` — `projectOut(pos, num)`, MLIR's
    /// `IntegerRelation::eliminateVars`.
    ///
    /// ⭐⭐ TWO ELIMINATIONS, AND WHICH ONE RUNS IS DECIDED PER VARIABLE, exactly as MLIR decides
    /// it: an EQUALITY naming the variable lets it be substituted away (Gaussian), and otherwise its
    /// lower bounds are combined with its upper bounds pairwise (Fourier–Motzkin) and every row that
    /// named it is dropped.
    ///
    /// ⛔ GAUSSIAN IS THE PATH THIS CAMPAIGN TAKES, AND IT IS EXACT.
    /// `constructExtentAndTotalElements` projects out precisely the dimensions
    /// [`Self::compose_matching_map`] has just tied down with one `flat_expr - d_i == 0` row each, so
    /// every eliminated variable has a ±1 coefficient in an equality and the substitution is integer
    /// arithmetic with no rounding. Fourier–Motzkin is here because MLIR's `projectOut` is total and
    /// a caller may hand this system anything.
    ///
    /// ⚠️ FOURIER–MOTZKIN COMPUTES THE **RATIONAL** SHADOW, which is what MLIR's own
    /// `fourierMotzkinEliminate(.., darkShadow=false)` computes: the result may admit a rational
    /// point where the original admitted no integer one. That is an over-approximation of the
    /// projection, not a wrong bound on a dimension that has one.
    ///
    /// ⛔ AN OUT-OF-RANGE RANGE ELIMINATES WHAT IT CAN AND NOTHING ELSE. MLIR asserts
    /// `pos + num <= getNumVars()`; the loop below simply stops at the last real column, because a
    /// column that does not exist cannot be holding a variable.
    #[must_use]
    pub fn project_out(&self, pos: u32, num: u32) -> FlatConstraints {
        let mut system = self.clone();
        let last = (pos as usize + num as usize).min(system.dims as usize + system.syms as usize);
        // ⛔ HIGHEST COLUMN FIRST, so that removing one does not renumber the ones still to go.
        for column in (pos as usize..last).rev() {
            system.eliminate(column);
        }
        system
    }

    /// ONE VARIABLE COLUMN ELIMINATED AND REMOVED — see [`Self::project_out`].
    fn eliminate(&mut self, column: usize) {
        // `gaussianEliminateVar`: an equality with a ±1 coefficient substitutes exactly.
        let pivot = self
            .equalities
            .iter()
            .position(|row| row[column] == 1 || row[column] == -1);
        if let Some(index) = pivot {
            let pivot = self.equalities.remove(index);
            let coeff = pivot[column];
            for row in self
                .equalities
                .iter_mut()
                .chain(self.inequalities.iter_mut())
            {
                // `row - (row[column] / coeff) * pivot`, and `coeff` is ±1 so the division is exact.
                let factor = row[column] / coeff;
                if factor != 0 {
                    for (slot, term) in row.iter_mut().zip(pivot.iter()) {
                        *slot -= factor * term;
                    }
                }
            }
        } else {
            // An equality with a coefficient other than ±1 becomes the two inequalities it is worth,
            // so Fourier–Motzkin sees the whole system.
            let mut equalities = Vec::new();
            for row in std::mem::take(&mut self.equalities) {
                if row[column] == 0 {
                    equalities.push(row);
                } else {
                    self.inequalities
                        .push(row.iter().map(|term| -term).collect());
                    self.inequalities.push(row);
                }
            }
            self.equalities = equalities;

            let (mut kept, mut lower, mut upper) = (Vec::new(), Vec::new(), Vec::new());
            for row in std::mem::take(&mut self.inequalities) {
                match row[column].cmp(&0) {
                    std::cmp::Ordering::Equal => kept.push(row),
                    // `a * x + rest >= 0` with `a > 0` bounds `x` from BELOW.
                    std::cmp::Ordering::Greater => lower.push(row),
                    std::cmp::Ordering::Less => upper.push(row),
                }
            }
            for low in &lower {
                for high in &upper {
                    let (a, b) = (low[column], -high[column]);
                    let mut combined: Vec<i64> = low
                        .iter()
                        .zip(high.iter())
                        .map(|(x, y)| b * x + a * y)
                        .collect();
                    normalise(&mut combined);
                    kept.push(combined);
                }
            }
            self.inequalities = kept;
        }

        for row in self
            .equalities
            .iter_mut()
            .chain(self.inequalities.iter_mut())
        {
            row.remove(column);
        }
        if column < self.dims as usize {
            self.dims -= 1;
        } else {
            self.syms -= 1;
        }
    }

    /// THE ROWS THAT SAY SOMETHING, TIGHTENED — `removeRedundantConstraints()`.
    ///
    /// ⭐⭐ IT IS WHAT MAKES [`Self::is_dim_value_zero`] ANSWERABLE AT ALL. That predicate reports a
    /// dimension pinned as soon as it finds an equality row with no OTHER variable in it — a row of
    /// all zeros qualifies, so a single `0 == 0` left behind would report EVERY dimension pinned and
    /// give a transfer of one element per dimension. Gaussian elimination leaves such rows behind
    /// routinely, which is why the reference's call order is `projectOut` then this and not the
    /// reverse (`AccessDetails.cpp:39-40`).
    ///
    /// What it does, in MLIR's order:
    ///
    /// 1. `gcdTightenInequalities` — divide a row's variable coefficients by their GCD and FLOOR the
    ///    constant, so `2*d0 - 3 >= 0` becomes `d0 - 2 >= 0`. That is an integer tightening, not a
    ///    rescaling: it is why the predicates may insist on a coefficient of exactly ±1.
    /// 2. Drop the trivially true rows — `removeTrivialRedundancy`'s first half.
    /// 3. Keep the TIGHTEST of the rows that name the same variables with the same coefficients. For
    ///    `sum + c >= 0` a SMALLER `c` is the stronger row whichever side it bounds, so the minimum
    ///    wins; identical equalities collapse to one.
    ///
    /// ⚠️ ONE DELIBERATE DIVERGENCE: MLIR follows the above with a Simplex over the whole system
    /// (`Simplex::detectRedundant`), which finds rows made redundant by COMBINATIONS of others. It
    /// has nothing to find here — the systems this campaign builds are one bound pair per dimension
    /// out of `buildIntegerSetFromSizes` (`DataTransferLowering.cpp:40-69`) composed with a
    /// permutation — and a row it would have removed cannot change either predicate's answer anyway:
    /// both scan for rows naming a single variable, and step 3 has already left at most one of those
    /// per variable and coefficient vector.
    ///
    /// ⛔ A TRIVIALLY **FALSE** ROW IS KEPT. `0 >= 1` means the system admits nothing, and
    /// `IntegerSet::constant_bound` documents why an empty system still has to be readable: IBM
    /// writes twenty masks whose sets are empty and expects each to lower to the all-lanes-off
    /// constant. Dropping the row would make the emptiness unobservable.
    #[must_use]
    pub fn remove_redundant_constraints(&self) -> FlatConstraints {
        let vars = self.dims as usize + self.syms as usize;
        let names_no_variable = |row: &[i64]| row[..vars].iter().all(|term| *term == 0);

        let mut equalities: Vec<Vec<i64>> = Vec::new();
        for row in &self.equalities {
            // `0 == 0` says nothing; `0 == 5` says the system is empty and is kept.
            if names_no_variable(row) && row[vars] == 0 {
                continue;
            }
            if !equalities.contains(row) {
                equalities.push(row.clone());
            }
        }

        let mut inequalities: Vec<Vec<i64>> = Vec::new();
        for row in &self.inequalities {
            let mut row = row.clone();
            gcd_tighten(&mut row);
            if names_no_variable(&row) && row[vars] >= 0 {
                continue;
            }
            match inequalities
                .iter_mut()
                .find(|kept| kept[..vars] == row[..vars])
            {
                Some(kept) => kept[vars] = kept[vars].min(row[vars]),
                None => inequalities.push(row),
            }
        }

        FlatConstraints {
            dims: self.dims,
            syms: self.syms,
            equalities,
            inequalities,
        }
    }

    /// IS DIMENSION `dim_pos` PINNED BY AN EQUALITY THAT NAMES NOTHING ELSE? —
    /// `agen::utils::isDimValueZero(csts, dim_pos)` (`dialect_utils/Agen/Utils.cpp:121-137`).
    ///
    /// ```cpp
    /// int num = csts.getNumCols() - 1;
    /// for (unsigned r = 0, e = csts.getNumEqualities(); r < e; r++) {
    ///   unsigned sum = 1;
    ///   for (unsigned c = 0; c < num; c++)
    ///     if (c != dim_pos) { if (csts.atEq(r, c) != 0) sum++; }
    ///   if (sum > 1) continue; else return true;
    /// }
    /// return false;
    /// ```
    ///
    /// ⛔⛔ ITS NAME OVERSTATES WHAT IT CHECKS, AND THE PORT KEEPS THE REFERENCE'S ANSWER. The scan
    /// stops at `getNumCols() - 1`, so the CONSTANT column is never looked at: `d2 - 5 == 0` answers
    /// **true** as loudly as `d2 == 0`. That is correct for the caller's purpose either way — a
    /// dimension pinned to any single value has an extent of 1, which is what
    /// `constructExtentAndTotalElements` (`:58-60`) and `calculateTimeBounds` (`Utils.cpp:232-233`)
    /// push — so this is a misleading NAME rather than a defect, and narrowing it to "pinned to
    /// zero" would change the extents of every set that pins a dimension to a nonzero offset.
    ///
    /// ⛔ NOR DOES IT REQUIRE A COEFFICIENT ON `dim_pos` AT ALL: a row of all zeros passes. See
    /// [`Self::remove_redundant_constraints`], which is what keeps one from reaching here.
    #[must_use]
    pub fn is_dim_value_zero(&self, dim_pos: u32) -> bool {
        let vars = self.num_cols() - 1;
        self.equalities
            .iter()
            .any(|row| (0..vars).all(|column| column == dim_pos as usize || row[column] == 0))
    }

    /// THE WIDTH OF DIMENSION `dim_pos`'S CONSTANT RANGE, or [`None`] when it has none —
    /// `agen::utils::isDimAConstantRange(csts, dim_pos, width)`
    /// (`dialect_utils/Agen/Utils.cpp:139-170`).
    ///
    /// ⭐ THE OUT-PARAMETER AND THE `bool` ARE ONE VALUE. The C++ writes `width = max - min + 1` and
    /// returns true, or writes `width = -1` and returns false; both callers test the `bool` and then
    /// read `width` (`AccessDetails.cpp:61`, `Utils.cpp:234-235`), and `constructExtentAndTotalElements`
    /// separately refuses a non-positive width as *"Extent along a dimension is negative"* (`:81-83`).
    /// [`Option`] carries the same pair with the `-1` unspellable.
    ///
    /// ⛔ INCLUSIVE, AND ONLY FROM ROWS THAT NAME THIS DIMENSION ALONE. A lower bound is a
    /// coefficient of exactly **+1** and gives `min = -constant`; an upper bound is exactly **-1**
    /// and gives `max = constant`; both must be found. So `d0 >= 0` with `-d0 + 63 >= 0` is a width
    /// of 64, and a row whose coefficient is 2 is ignored rather than halved — which is what
    /// [`Self::remove_redundant_constraints`]' GCD tightening exists to prevent from mattering.
    ///
    /// ⛔ THE **LAST** MATCHING ROW WINS, because the C++ overwrites `min_value`/`max_value` without
    /// testing whether it already found one. After the tightening and de-duplication above there is
    /// at most one row per side, so the two agree; transcribing the reference's assignment keeps them
    /// agreeing on a system that was not simplified.
    #[must_use]
    pub fn is_dim_a_constant_range(&self, dim_pos: u32) -> Option<i64> {
        let vars = self.num_cols() - 1;
        let (mut min_value, mut max_value) = (None, None);
        for row in &self.inequalities {
            let names_only_this =
                (0..vars).all(|column| column == dim_pos as usize || row[column] == 0);
            if !names_only_this {
                continue;
            }
            match row[dim_pos as usize] {
                1 => min_value = Some(-row[vars]),
                -1 => max_value = Some(row[vars]),
                _ => {}
            }
        }
        match (min_value, max_value) {
            (Some(min_value), Some(max_value)) => Some(max_value - min_value + 1),
            _ => None,
        }
    }

    /// WHICH SYMBOL BOUNDS DIMENSION `dim_pos` FROM ABOVE, when a symbol is what bounds it —
    /// `agen::utils::isDimASymbolicRange(csts, dim_pos, symbol_pos)`
    /// (`dialect_utils/Agen/Utils.cpp:171-208`). The out-parameter and the `bool` are one value, as
    /// in [`Self::is_dim_a_constant_range`].
    ///
    /// ⛔ THE SHAPE IT RECOGNISES IS EXACTLY `d >= 0` AND `-d + s - 1 >= 0`: a lower bound is a
    /// coefficient of +1 with NO symbol and a zero constant, an upper bound is -1 with exactly ONE
    /// symbol and a constant of **-1**. So the extent is the symbol's own value, which is why
    /// `calculateTimeBounds` pushes `time_symbols[symbol_pos]`'s constant unchanged (`:238-241`).
    ///
    /// ⛔ THE LAST SYMBOL-MENTIONING ROW WINS, whether or not that row is the one that matched — the
    /// C++ assigns `symbol_pos` before it tests the coefficient. (Its `c != dim_pos` guard inside the
    /// symbol loop is vacuous: `dim_pos` is a DIMENSION column and the loop runs over symbol ones.)
    #[must_use]
    pub fn is_dim_a_symbolic_range(&self, dim_pos: u32) -> Option<u32> {
        let dims = self.dims as usize;
        let constant = self.num_cols() - 1;
        let (mut found_min, mut found_max) = (false, false);
        let mut symbol_pos = None;
        for row in &self.inequalities {
            let names_only_this =
                (0..dims).all(|column| column == dim_pos as usize || row[column] == 0);
            if !names_only_this {
                continue;
            }
            let mut symbols = 0;
            for (offset, coeff) in row[dims..dims + self.syms as usize].iter().enumerate() {
                if *coeff != 0 {
                    symbols += 1;
                    symbol_pos = u32::try_from(offset).ok();
                }
            }
            match row[dim_pos as usize] {
                1 if symbols == 0 && row[constant] == 0 => found_min = true,
                -1 if symbols == 1 && row[constant] == -1 => found_max = true,
                _ => {}
            }
        }
        if found_min && found_max {
            symbol_pos
        } else {
            None
        }
    }

    /// IS EVERY CONSTRAINT ONE-DIMENSIONAL OVER COLUMNS `pos .. pos + num`? —
    /// `IntegerRelation::isHyperRectangular`
    /// (`llvm-project/mlir/lib/Analysis/Presburger/IntegerRelation.cpp:1885-1907`), the reference's
    /// own *"simple (naive and conservative) check"*.
    ///
    /// ⭐ `checkBasicConditions` ASKS IT OVER EVERY VARIABLE — `isHyperRectangular(0, getNumCols() - 1)`
    /// (`Helper.cpp:155-157`) — so one row naming two of them is what refuses a load set.
    /// ⛔ THE CONSTANT COLUMN IS OUTSIDE THE RANGE, which is why a rectangle at a nonzero offset
    /// passes; and `>= 0` rows are scanned exactly like `== 0` rows.
    #[must_use]
    pub fn is_hyper_rectangular(&self, pos: usize, num: usize) -> bool {
        self.inequalities
            .iter()
            .chain(self.equalities.iter())
            .all(|row| {
                (pos..pos.saturating_add(num))
                    .filter(|column| row.get(*column).is_some_and(|value| *value != 0))
                    .count()
                    <= 1
            })
    }

    /// WHETHER THE SYSTEM ADMITS NOTHING AT ALL — `isEmpty()`, which `getPageValidity` asks twice
    /// (`TransformPagedMemViewImpl.cpp:127`, `:132`) to decide whether a page is reachable.
    ///
    /// ⭐⭐ ELIMINATE EVERY VARIABLE, THEN READ THE ROWS THAT ARE LEFT. A system with no variables is
    /// a list of constant claims, and one that says `c >= 0` for a negative `c` or `c == 0` for a
    /// nonzero one is unsatisfiable. [`Self::project_out`] is already the elimination MLIR's own
    /// `IntegerRelation::isEmpty` runs (Gaussian where an equality pins a variable, Fourier–Motzkin
    /// otherwise), and it keeps a trivially false row rather than dropping it — which is what makes
    /// the contradiction observable here.
    ///
    /// ⭐ THE GCD TEST COMES FIRST, as `isEmptyByGCDTest` does: an equality whose variable
    /// coefficients share a divisor the constant does not have has no INTEGER solution, whatever the
    /// rationals say.
    ///
    /// ⚠️ AND WHAT REMAINS IS RATIONAL. Fourier–Motzkin computes the rational shadow (see
    /// [`Self::project_out`]), so a system with rational but no integer points is reported NON-empty
    /// where MLIR's Simplex would find it empty. For `getPageValidity` that direction is the safe one:
    /// it keeps a page conditional that never fires, rather than dropping a page that does.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        // `isEmptyByGCDTest()` — `gcd(coeffs) | constant` must hold for every equality.
        let gcd_refutes = self.equalities.iter().any(|row| {
            let (coeffs, constant) = row.split_at(row.len().saturating_sub(1));
            let divisor = coeffs.iter().fold(0i64, |acc, term| gcd(acc, term.abs()));
            divisor > 1 && constant.first().is_some_and(|term| term % divisor != 0)
        });
        if gcd_refutes {
            return true;
        }

        let constants = self.project_out(0, self.dims + self.syms);
        constants
            .equalities
            .iter()
            .any(|row| row.first().is_some_and(|term| *term != 0))
            || constants
                .inequalities
                .iter()
                .any(|row| row.first().is_some_and(|term| *term < 0))
    }
}

/// A ROW DIVIDED BY THE GCD OF EVERYTHING IN IT — the normalisation Fourier–Motzkin needs so that
/// combined rows do not grow coefficients without bound.
///
/// ⭐ THE CONSTANT IS INCLUDED IN THE GCD, so the division is exact and the row keeps its exact
/// meaning. [`gcd_tighten`] is the other, INTEGER-tightening rule.
fn normalise(row: &mut [i64]) {
    let divisor = row.iter().fold(0_i64, |acc, term| gcd(acc, term.abs()));
    if divisor > 1 {
        for term in row.iter_mut() {
            *term /= divisor;
        }
    }
}

/// `a * x + c >= 0` TIGHTENED OVER THE INTEGERS — `gcdTightenInequalities()`.
///
/// ⛔ THE CONSTANT IS **FLOORED**, NOT DIVIDED. `2*d0 - 3 >= 0` is `d0 >= 1.5`, and over the
/// integers that is `d0 - 2 >= 0`: dividing the constant would have given `d0 - 1 >= 0`, which
/// admits `d0 == 1` and is a bound the original row does not have.
fn gcd_tighten(row: &mut [i64]) {
    let Some((constant, variables)) = row.split_last_mut() else {
        return;
    };
    let divisor = variables
        .iter()
        .fold(0_i64, |acc, term| gcd(acc, term.abs()));
    if divisor > 1 {
        for term in variables.iter_mut() {
            *term /= divisor;
        }
        *constant = constant.div_euclid(divisor);
    }
}

/// The greatest common divisor of two NON-NEGATIVE numbers, with `gcd(0, n) == n`.
fn gcd(a: i64, b: i64) -> i64 {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// ONE LOCAL VARIABLE OF A FLATTENED ROW — the quotient MLIR introduces for a `floordiv` or a `mod`
/// whose divisor does not cancel.
///
/// ⭐ IT CARRIES WHAT IT STANDS FOR, WHICH IS WHAT MAKES TWO OF THEM ONE COLUMN.
/// `SimpleAffineExprFlattener` keeps a parallel `localExprs` list and `findLocalId` reuses the column
/// of a local it has already introduced, so `(x floordiv 2) + (x floordiv 2)` is ONE column with
/// coefficient 2 rather than two columns of 1. That difference is visible: an enclosing `mod 2` asks
/// whether every column divides by 2, and the two spellings answer differently.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Local {
    /// The dividend, itself flattened and already divided through by the common factor.
    pub dividend: Box<FlatAffineExpr>,
    /// The divisor, always positive.
    pub divisor: i64,
    /// This local's coefficient in the row that holds it.
    pub coeff: i64,
}

/// ONE AFFINE EXPRESSION AS A COEFFICIENT ROW — the `flattenedExpr` of MLIR's
/// `getFlattenedAffineExpr(expr, numDims, numSymbols, &coeffs, &constraints)`.
///
/// MLIR hands back ONE `SmallVector<int64_t>` laid out as
/// `[dims.., syms.., locals.., constant]`, which is why its callers read the constant term as
/// `coeffs.back()`:
///
/// ```cpp
/// SmallVector<int64_t> coeffs;
/// affine::FlatAffineValueConstraints constraints;
/// auto flat_result =
///     getFlattenedAffineExpr(expr, num_dims, 0, &coeffs, &constraints);
/// int constant_offset = coeffs.size() != num_dims ? coeffs.back() : 0;
/// ```
/// (`Transform/Dataflow/MutableStartAddrShifting.cpp:474-478`, entries 191 and 192)
///
/// ⛔ THE FOUR GROUPS ARE SEPARATE FIELDS HERE, BECAUSE THE ONE FLAT VECTOR IS WHAT MAKES THAT
/// `coeffs.size() != num_dims` GUARD LOOK LIKE A BOUNDS CHECK. It is not one: the row is always
/// `num_dims + num_syms + num_locals + 1` wide, so it is longer than `num_dims` for every expression
/// that flattens at all, and the `: 0` arm is dead. What it actually guards is FAILURE — a
/// non-affine expression leaves `coeffs` EMPTY, and `coeffs.back()` on an empty `SmallVector` is
/// undefined behaviour whenever `num_dims != 0`. Naming the constant column ([`Self::constant`])
/// makes both the dead arm and the undefined one unwritable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FlatAffineExpr {
    /// One coefficient per dimension of the space the expression was flattened over.
    pub dims: Vec<i64>,
    /// One coefficient per SYMBOL of that space.
    pub syms: Vec<i64>,
    /// The locals introduced along the way, in the order they were introduced.
    pub locals: Vec<Local>,
    /// The constant term — `coeffs.back()`.
    pub constant: i64,
}

impl FlatAffineExpr {
    /// THE ALL-ZERO ROW over a space of `dims` dimensions and `syms` symbols.
    #[must_use]
    pub fn zero(dims: u32, syms: u32) -> FlatAffineExpr {
        FlatAffineExpr {
            dims: vec![0; dims as usize],
            syms: vec![0; syms as usize],
            locals: Vec::new(),
            constant: 0,
        }
    }

    /// THE ROW AS A LITERAL, or `None` when it still holds a variable — `isa<AffineConstantExpr>` on
    /// the flattened side, which is the test MLIR's `visitMulExpr`, `visitModExpr` and `visitDivExpr`
    /// each make of their right-hand operand.
    #[must_use]
    pub fn as_constant(&self) -> Option<i64> {
        let variable = self.dims.iter().chain(self.syms.iter()).any(|c| *c != 0)
            || self.locals.iter().any(|local| local.coeff != 0);
        if variable { None } else { Some(self.constant) }
    }

    /// EVERY COLUMN OF THE ROW, THE CONSTANT ONE INCLUDED — MLIR's `for (i = 0, e = lhs.size(); ..)`,
    /// which runs over the whole `SmallVector` and therefore over the constant term as well.
    fn row(&self) -> impl Iterator<Item = i64> {
        self.dims
            .iter()
            .copied()
            .chain(self.syms.iter().copied())
            .chain(self.locals.iter().map(|local| local.coeff))
            .chain(core::iter::once(self.constant))
    }

    /// `for (int64_t &v : lhs) v *= rhsConst;` — `visitMulExpr`.
    fn scale(&mut self, k: i64) {
        for coeff in self.dims.iter_mut().chain(self.syms.iter_mut()) {
            *coeff *= k;
        }
        for local in &mut self.locals {
            local.coeff *= k;
        }
        self.constant *= k;
    }

    /// `for (auto &c : lhs) c /= gcd;` — the numerator simplification `visitDivExpr` and
    /// `visitModExpr` share. Every column divides exactly, `gcd` being their common factor.
    fn divide(&mut self, gcd: i64) {
        for coeff in self.dims.iter_mut().chain(self.syms.iter_mut()) {
            *coeff /= gcd;
        }
        for local in &mut self.locals {
            local.coeff /= gcd;
        }
        self.constant /= gcd;
    }

    /// TWO ROWS ADDED COLUMN BY COLUMN — `visitAddExpr`.
    ///
    /// ⭐ A LOCAL THE OTHER ROW ALREADY NAMES SHARES ITS COLUMN, which is `findLocalId`; one it does
    /// not gets a new column, which is `addLocalFloorDivId`. See [`Local`].
    fn add(&mut self, other: &FlatAffineExpr) {
        if self.dims.len() < other.dims.len() {
            self.dims.resize(other.dims.len(), 0);
        }
        if self.syms.len() < other.syms.len() {
            self.syms.resize(other.syms.len(), 0);
        }
        for (mine, theirs) in self.dims.iter_mut().zip(&other.dims) {
            *mine += *theirs;
        }
        for (mine, theirs) in self.syms.iter_mut().zip(&other.syms) {
            *mine += *theirs;
        }
        for local in &other.locals {
            self.add_local(local.clone());
        }
        self.constant += other.constant;
    }

    /// ONE LOCAL INTO THE ROW — `findLocalId` first, `addLocalFloorDivId` otherwise.
    fn add_local(&mut self, local: Local) {
        match self
            .locals
            .iter_mut()
            .find(|held| held.dividend == local.dividend && held.divisor == local.divisor)
        {
            Some(held) => held.coeff += local.coeff,
            None => self.locals.push(local),
        }
    }
}

impl AffineExpr {
    /// THIS EXPRESSION AS A COEFFICIENT ROW — `getFlattenedAffineExpr(expr, num_dims, num_syms, ..)`,
    /// MLIR's `SimpleAffineExprFlattener` transcribed.
    ///
    /// The row's constant column is what entries 191 and 192 read out of a subscript to learn how much
    /// of it is a CONSTANT OFFSET that can be shifted into an immutable start address; see
    /// [`FlatAffineExpr`].
    ///
    /// # ⭐ THE THREE INTERESTING CASES, AND WHY EACH IS WHAT IT IS
    ///
    /// **A product** must have a literal on one side (`visitMulExpr`); it scales the other side's row.
    /// MLIR insists the literal be the RIGHT one, its canonical form, and reports failure otherwise —
    /// this accepts either side, because `Mul(Const(8), Dim(0))` is writable in this island and
    /// answering `8 * d0` for it is not a different answer, only a reachable one.
    ///
    /// **A `mod k`** is zero when every column of the dividend divides by `k` — `x * 8 mod 4` is
    /// nothing — and otherwise becomes `dividend - k * q`, where `q` is a fresh local standing for
    /// `dividend floordiv k`. The constant column is left ALONE in that second case: the local absorbs
    /// the remainder, so a subscript like `(d0 + 5) mod 8` still reports a constant offset of 5.
    ///
    /// **A `floordiv k`** cancels when `k` divides every column, giving the divided row; otherwise the
    /// whole row becomes a single fresh local with coefficient 1 and the constant column goes to ZERO
    /// — `(d0 + 5) floordiv 8` offers no constant offset to shift, and reporting 5 there would shift
    /// eight times too much.
    ///
    /// # ⛔ A LITERAL DIVIDEND IS FOLDED, BECAUSE MLIR FOLDS IT BEFORE THE FLATTENER EVER SEES IT
    ///
    /// `getAffineConstantExpr(-3).floorDiv(4)` is `-1` the moment it is built (`simplifyFloorDiv`,
    /// `simplifyMod`), so the flattener's local-variable path is unreachable for an expression with no
    /// variables in it. This island does not fold in its constructors — [`substitute_symbols`] says so
    /// — so the fold happens HERE instead, and `Const(-3).floordiv(4)` reports `-1` rather than a
    /// local standing for a value that is already known.
    ///
    /// # ⛔ WHAT IS A `todo!` AND NOT AN ANSWER
    ///
    /// A product of two variables, and a `mod`/`floordiv` by anything but a positive literal, are not
    /// pure affine expressions. MLIR asserts (`"RHS constant has to be positive"`) or reports failure,
    /// and its caller in entry 191 then reads `coeffs.back()` off an empty vector — undefined
    /// behaviour. A `todo!` naming the shape is this crate's answer to that, exactly as
    /// [`terms_in`] answers it for a constraint row.
    #[must_use]
    pub fn flatten(&self, dims: u32, syms: u32) -> FlatAffineExpr {
        match self {
            AffineExpr::Dim(n) => {
                let mut row = FlatAffineExpr::zero(dims, syms);
                set_column(&mut row.dims, *n);
                row
            }
            AffineExpr::Sym(n) => {
                let mut row = FlatAffineExpr::zero(dims, syms);
                set_column(&mut row.syms, *n);
                row
            }
            AffineExpr::Const(c) => FlatAffineExpr {
                constant: *c,
                ..FlatAffineExpr::zero(dims, syms)
            },
            AffineExpr::Add(a, b) => {
                let mut row = a.flatten(dims, syms);
                row.add(&b.flatten(dims, syms));
                row
            }
            AffineExpr::Mul(a, b) => {
                let lhs = a.flatten(dims, syms);
                let rhs = b.flatten(dims, syms);
                match (lhs.as_constant(), rhs.as_constant()) {
                    // `int64_t rhsConst = rhs[getConstantIndex()]; for (v : lhs) v *= rhsConst;`
                    (_, Some(k)) => {
                        let mut row = lhs;
                        row.scale(k);
                        row
                    }
                    (Some(k), None) => {
                        let mut row = rhs;
                        row.scale(k);
                        row
                    }
                    (None, None) => todo!(
                        "a product of two variables is not an affine expression; MLIR's \
                         visitMulExpr reports failure for it and its caller then reads the \
                         constant column off an empty row"
                    ),
                }
            }
            AffineExpr::Mod(a, b) => {
                let lhs = a.flatten(dims, syms);
                let modulus = positive_literal(b, dims, syms, "mod");
                if let Some(dividend) = lhs.as_constant() {
                    // Folded at construction by `simplifyMod`, which is floored: `-3 mod 4` is 1.
                    return FlatAffineExpr {
                        constant: dividend.rem_euclid(modulus),
                        ..FlatAffineExpr::zero(dims, syms)
                    };
                }
                // "Check if the LHS expression is a multiple of modulo factor" — if it is, the
                // whole expression is zero.
                if lhs.row().all(|coeff| coeff % modulus == 0) {
                    return FlatAffineExpr::zero(dims, syms);
                }
                // `expr % c` becomes `expr - c * q` with `q = expr floordiv c`, the GCD of the
                // dividend and `c` cancelled out of the quotient first.
                let common = lhs.row().fold(modulus, |g, coeff| gcd(g, coeff.abs()));
                let mut dividend = lhs.clone();
                dividend.divide(common);
                let mut row = lhs;
                row.add_local(Local {
                    dividend: Box::new(dividend),
                    divisor: modulus / common,
                    coeff: -modulus,
                });
                row
            }
            AffineExpr::FloorDiv(a, b) => {
                let lhs = a.flatten(dims, syms);
                let divisor = positive_literal(b, dims, syms, "floordiv");
                if let Some(dividend) = lhs.as_constant() {
                    // Folded at construction by `simplifyFloorDiv`: `-3 floordiv 4` is -1.
                    return FlatAffineExpr {
                        constant: dividend.div_euclid(divisor),
                        ..FlatAffineExpr::zero(dims, syms)
                    };
                }
                // "Simplify the floordiv, ceildiv if possible by canceling out the greatest common
                // divisors of the numerator and denominator."
                let common = lhs.row().fold(divisor, |g, coeff| gcd(g, coeff.abs()));
                let mut row = lhs;
                row.divide(common);
                let denominator = divisor / common;
                // "If the denominator becomes 1, the updated LHS is the result."
                if denominator == 1 {
                    return row;
                }
                let mut divided = FlatAffineExpr::zero(dims, syms);
                divided.locals.push(Local {
                    dividend: Box::new(row),
                    divisor: denominator,
                    coeff: 1,
                });
                divided
            }
        }
    }
}

/// COLUMN `pos` OF A COEFFICIENT GROUP SET TO ONE.
///
/// ⚠️ A POSITION PAST THE STATED ARITY GETS ITS OWN COLUMN rather than being dropped. MLIR asserts
/// `position < numDims`; dropping the variable instead would make `d7` read as ZERO in a two-
/// dimensional space, which is a wrong constant column rather than a missing one.
fn set_column(coeffs: &mut Vec<i64>, pos: u32) {
    let pos = pos as usize;
    if coeffs.len() <= pos {
        coeffs.resize(pos + 1, 0);
    }
    coeffs[pos] = 1;
}

/// THE DIVISOR OF A `mod` OR A `floordiv`, WHICH HAS TO BE A POSITIVE LITERAL —
/// `assert(rhsConst > 0 && "RHS constant has to be positive")`.
fn positive_literal(expr: &AffineExpr, dims: u32, syms: u32, op: &str) -> i64 {
    match expr.flatten(dims, syms).as_constant() {
        Some(k) if k > 0 => k,
        _ => todo!(
            "a `{op}` by anything but a positive literal is not a pure affine expression; MLIR's \
             flattener asserts `rhsConst > 0` and its caller then reads the constant column off an \
             empty row"
        ),
    }
}

#[cfg(test)]
mod tests {
    use crate::islands::dataflow_ir::print;
    use crate::islands::dataflow_ir::ty::{
        AffineExpr, AffineMap, BoundType, Constraint, FlatAffineExpr, FlatConstraints, IntegerSet,
    };

    /// `expr >= 0`.
    fn ineq(expr: AffineExpr) -> Constraint {
        Constraint {
            expr,
            is_equality: false,
        }
    }

    /// A one-dimensional set — the only shape `getMaskValueConstantForNonPT` lets through.
    fn set(constraints: Vec<Constraint>) -> IntegerSet {
        IntegerSet {
            dims: 1,
            symbols: 0,
            constraints,
        }
    }

    /// `-d0 + c >= 0`.
    fn upper(c: i64) -> Constraint {
        ineq(AffineExpr::dim(0).times(-1).plus(AffineExpr::Const(c)))
    }

    /// `-d<k> + c >= 0`.
    fn upper_of(k: u32, c: i64) -> Constraint {
        ineq(AffineExpr::dim(k).times(-1).plus(AffineExpr::Const(c)))
    }

    /// `d0 + c >= 0`.
    fn lower(c: i64) -> Constraint {
        ineq(AffineExpr::dim(0).plus(AffineExpr::Const(c)))
    }

    /// ⭐⭐ IBM'S ALL-LANES-OFF MASK: AN EMPTY SET THAT STILL HAS BOTH BOUNDS.
    ///
    /// `affine_set<(d0) : (d0 - 64 >= 0, -d0 + 63 >= 0)>` holds no integer, and IBM writes it as the
    /// mask of twenty `create_affine_mask` ops over `vector<64xi1>`
    /// (`dcc/test/Conversion/VectorChainToSentientPESFP/mixed_precision.mlir:419`, `:747-895`) whose
    /// `CHECK-SENT-IR` lowers each to `sentient.scalar_constant {value = 0 : si64}` (`:288`, `:296`).
    /// The bounds CROSS, and both are present — which is the whole reason that mask lowers instead of
    /// erroring out.
    #[test]
    fn the_empty_mask_set_ibm_writes_still_has_both_bounds() {
        let empty = set(vec![lower(-64), upper(63)]);
        assert_eq!(empty.constant_bound(BoundType::Lb, 0), Some(64));
        assert_eq!(empty.constant_bound(BoundType::Ub, 0), Some(63));
    }

    /// ⭐ A FULL LANE SPAN — the commonest set in the authority tree's tests, 22 files' `#set`.
    ///
    /// `affine_set<(d0) : (d0 >= 0, -d0 + 63 >= 0)>` is lanes `0 ..= 63`, so the upper bound is
    /// INCLUSIVE: 63, not 64. A 64 here would set one slice too many in every mask value derived
    /// from it.
    #[test]
    fn a_full_lane_span_bounds_at_zero_and_at_the_last_lane() {
        let span = set(vec![ineq(AffineExpr::dim(0)), upper(63)]);
        assert_eq!(span.constant_bound(BoundType::Lb, 0), Some(0));
        assert_eq!(span.constant_bound(BoundType::Ub, 0), Some(63));
    }

    /// ⭐ AND AN EQUALITY PINS BOTH SIDES TO THE SAME CONSTANT.
    ///
    /// `affine_set<(d0) : (d0 == 0)>` is what [`IntegerSet::from_sizes`] writes for a size of one —
    /// `#set2` of the scheduler's own output.
    #[test]
    fn an_equality_pins_both_sides_to_one_constant() {
        let pinned = IntegerSet::from_sizes(&[1]);
        assert_eq!(pinned.constant_bound(BoundType::Lb, 0), Some(0));
        assert_eq!(pinned.constant_bound(BoundType::Ub, 0), Some(0));
    }

    /// ⛔ A SET BOUNDS ONLY THE SIDE IT STATES, and the missing side is `None` rather than a default.
    ///
    /// A zero for an absent upper bound would read as "lane 0 only" — a mask of one lane where the
    /// truth is "no constant bound", which is the case `hasConstantBounds` exists to reject.
    #[test]
    fn a_one_sided_set_has_only_the_bound_it_states() {
        let half = set(vec![ineq(AffineExpr::dim(0))]);
        assert_eq!(half.constant_bound(BoundType::Lb, 0), Some(0));
        assert_eq!(half.constant_bound(BoundType::Ub, 0), None);

        let other = set(vec![upper(63)]);
        assert_eq!(other.constant_bound(BoundType::Lb, 0), None);
        assert_eq!(other.constant_bound(BoundType::Ub, 0), Some(63));

        let nothing = set(vec![]);
        assert_eq!(nothing.constant_bound(BoundType::Lb, 0), None);
        assert_eq!(nothing.constant_bound(BoundType::Ub, 0), None);
    }

    /// ⛔ A CONSTRAINT ON ANOTHER DIMENSION SAYS NOTHING ABOUT THIS ONE.
    ///
    /// `affine_set<(d0, d1, d2) : (d0 == 0, d1 == 0, d2 >= 0, -d2 + 63 >= 0)>` — the shape
    /// [`IntegerSet::from_sizes`] gives `[1, 1, 64]`, and the shape 17 of the authority tree's tests
    /// write — must answer 0 and 63 about `d2` while its two equalities are read as rows that pin a
    /// DIFFERENT variable. Treating `d0 == 0` as a statement about `d2` would pin every walk to its
    /// first lane.
    #[test]
    fn a_constraint_on_another_dimension_is_not_a_bound_on_this_one() {
        let rect = IntegerSet::from_sizes(&[1, 1, 64]);
        assert_eq!(rect.constant_bound(BoundType::Lb, 2), Some(0));
        assert_eq!(rect.constant_bound(BoundType::Ub, 2), Some(63));
        assert_eq!(rect.constant_bound(BoundType::Lb, 0), Some(0));
        assert_eq!(rect.constant_bound(BoundType::Ub, 0), Some(0));

        let elsewhere = IntegerSet {
            dims: 2,
            symbols: 0,
            constraints: vec![ineq(AffineExpr::dim(1)), upper_of(1, 63)],
        };
        assert_eq!(elsewhere.constant_bound(BoundType::Lb, 0), None);
        assert_eq!(elsewhere.constant_bound(BoundType::Ub, 0), None);
    }

    /// ⛔ THE TIGHTEST BOUND WINS — max over the lower ones, min over the upper ones.
    ///
    /// Taking the FIRST one found instead would answer 0 and 63 below, admitting 58 lanes the set
    /// excludes.
    #[test]
    fn the_tightest_of_several_bounds_wins() {
        let narrowed = set(vec![
            ineq(AffineExpr::dim(0)),
            lower(-5),
            upper(63),
            upper(31),
        ]);
        assert_eq!(narrowed.constant_bound(BoundType::Lb, 0), Some(5));
        assert_eq!(narrowed.constant_bound(BoundType::Ub, 0), Some(31));
    }

    /// ⛔⛔ A LOWER BOUND ROUNDS **UP** AND AN UPPER BOUND ROUNDS **DOWN**, on both signs.
    ///
    /// `2*d0 + 5 >= 0` means `d0 >= -2.5`, whose least integer is -2; `-2*d0 - 5 >= 0` means
    /// `d0 <= -2.5`, whose greatest integer is -3. Rust's `/` truncates toward zero and would answer
    /// -2 for BOTH — widening the first bound is harmless and widening the second admits a lane the
    /// set excludes. This is the case that separates `div_euclid` from `/`.
    #[test]
    fn a_bound_rounds_toward_the_side_that_keeps_the_set() {
        let scaled = |coeff: i64, c: i64| {
            set(vec![ineq(
                AffineExpr::dim(0).times(coeff).plus(AffineExpr::Const(c)),
            )])
        };
        assert_eq!(scaled(2, -5).constant_bound(BoundType::Lb, 0), Some(3));
        assert_eq!(scaled(-2, 5).constant_bound(BoundType::Ub, 0), Some(2));
        assert_eq!(scaled(2, 5).constant_bound(BoundType::Lb, 0), Some(-2));
        assert_eq!(scaled(-2, -5).constant_bound(BoundType::Ub, 0), Some(-3));
    }

    /// ⛔ AN EQUALITY WHOSE COEFFICIENT IS NOT ±1 BOUNDS NOTHING — MLIR's answer, not a shortcut.
    ///
    /// `findEqualityToConstant` skips rows with `v * v != 1`, and the scan after it reads
    /// inequalities only, so `2*d0 - 4 == 0` gives `None` on both sides even though `d0` is 2.
    /// Nothing in the authority tree writes such an equality; reproducing the quirk costs nothing and
    /// keeps this from being a different function than the one it ports.
    #[test]
    fn a_non_unit_equality_bounds_nothing() {
        let scaled = IntegerSet {
            dims: 1,
            symbols: 0,
            constraints: vec![Constraint {
                expr: AffineExpr::dim(0).times(2).plus(AffineExpr::Const(-4)),
                is_equality: true,
            }],
        };
        assert_eq!(scaled.constant_bound(BoundType::Lb, 0), None);
        assert_eq!(scaled.constant_bound(BoundType::Ub, 0), None);
    }

    // ───────────── THE FLATTENED MATRIX, AND WHAT `constructExtentAndTotalElements` READS ─────────────

    /// A LAYOUT MAP'S COEFFICIENTS ARE ITS STRIDES, PLUS A TRAILING CONSTANT TERM.
    ///
    /// `affine_map<(d0, d1) -> (d0 * 64 + d1)>` is the layout of a `memref<8x64xf16>` view, and
    /// `getMapCoefficients` gives `[64, 1, 0]` — three entries for two dimensions, which is the
    /// relation `LoweringXRF.cpp:50` asserts.
    #[test]
    fn a_layout_maps_coefficients_are_its_strides_and_a_constant() {
        assert_eq!(AffineMap::linear(&[64, 1]).coefficients(0), vec![64, 1, 0]);
        assert_eq!(AffineMap::linear(&[1]).coefficients(0), vec![1, 0]);
        // A constant term survives into the last column and nowhere else.
        let shifted = AffineMap {
            dims: 2,
            syms: 0,
            results: vec![
                AffineExpr::dim(0)
                    .times(64)
                    .plus(AffineExpr::dim(1))
                    .plus(AffineExpr::Const(7)),
            ],
        };
        assert_eq!(shifted.coefficients(0), vec![64, 1, 7]);
    }

    /// A SYMBOL GETS ITS OWN COLUMN, AFTER EVERY DIMENSION — never the constant one.
    #[test]
    fn a_symbol_is_a_column_of_its_own() {
        let map = AffineMap {
            dims: 2,
            syms: 1,
            results: vec![
                AffineExpr::dim(1)
                    .plus(AffineExpr::Sym(0).times(8))
                    .plus(AffineExpr::Const(-1)),
            ],
        };
        assert_eq!(map.coefficients(0), vec![0, 1, 8, -1]);
    }

    /// `mem_view_layout_map.compose(time_addr_map)` — the composition `calculateTimeOffsets` takes
    /// (`dialect_utils/Agen/Utils.cpp:103`).
    ///
    /// The layout of a `memref<2x64>` view is `(d0, d1) -> (d0 * 64 + d1)`; a one-step time address
    /// map is `(d0) -> (d0, 0)`. Composed, one time step moves 64 elements.
    #[test]
    fn composing_a_layout_over_a_time_address_map_gives_the_step_in_elements() {
        let layout = AffineMap::linear(&[64, 1]);
        let time_addr = AffineMap {
            dims: 1,
            syms: 0,
            results: vec![AffineExpr::dim(0), AffineExpr::Const(0)],
        };
        let composed = layout.compose(&time_addr);
        assert_eq!(
            composed.dims, 1,
            "the composition takes the inner map's inputs"
        );
        assert_eq!(composed.coefficients(0), vec![64, 0]);
    }

    /// ⛔ THE INNER MAP'S SYMBOLS ARE RENUMBERED ABOVE THE OUTER MAP'S, so two `s0`s cannot collide.
    #[test]
    fn composition_keeps_the_two_symbol_spaces_apart() {
        let outer = AffineMap {
            dims: 1,
            syms: 1,
            results: vec![AffineExpr::dim(0).plus(AffineExpr::Sym(0))],
        };
        let inner = AffineMap {
            dims: 1,
            syms: 1,
            results: vec![AffineExpr::Sym(0).times(4)],
        };
        let composed = outer.compose(&inner);
        assert_eq!(composed.syms, 2);
        // `s0` is still the OUTER map's; the inner map's became `s1`.
        assert_eq!(composed.coefficients(0), vec![0, 1, 4, 0]);
    }

    /// `constructIndices`' REWRITE: a constant subscript is folded into the map and the surviving
    /// dimensions are renumbered densely (`AccessDetails.cpp:385-401`).
    ///
    /// Subscripts `%iv, %c0` over the layout above become a one-dimensional map, and the declared
    /// arity is the caller's `ndims` — 1 — not what the expressions happen to mention.
    #[test]
    fn folding_a_constant_subscript_renumbers_the_rest_and_states_the_new_arity() {
        let subscripts = AffineMap {
            dims: 2,
            syms: 0,
            results: vec![AffineExpr::dim(0), AffineExpr::dim(1)],
        };
        let folded = subscripts.replace_dims_and_symbols(
            &[AffineExpr::dim(0), AffineExpr::Const(3)],
            &[],
            1,
            0,
        );
        assert_eq!(folded.dims, 1);
        assert_eq!(folded.syms, 0);
        assert_eq!(
            folded.results,
            vec![AffineExpr::dim(0), AffineExpr::Const(3)]
        );
    }

    /// THE TRANSFER SET'S OWN GEOMETRY, READ OFF THE MATRIX — `IntegerSet::from_sizes(&[1, 64])`.
    ///
    /// Dimension 0 is pinned by an equality, so `isDimValueZero` answers true and it has no constant
    /// range; dimension 1 spans `0 ..= 63`, so it has a width of 64 and is not pinned.
    #[test]
    fn a_rectangle_reads_back_as_one_pinned_dimension_and_one_span() {
        let csts = FlatConstraints::from_integer_set(&IntegerSet::from_sizes(&[1, 64]));
        assert_eq!(csts.dims, 2);
        assert_eq!(csts.num_cols(), 3);
        assert_eq!(csts.equalities, vec![vec![1, 0, 0]]);
        assert_eq!(csts.inequalities, vec![vec![0, 1, 0], vec![0, -1, 63]]);

        assert!(csts.is_dim_value_zero(0));
        assert_eq!(csts.is_dim_a_constant_range(0), None);
        assert!(!csts.is_dim_value_zero(1));
        assert_eq!(csts.is_dim_a_constant_range(1), Some(64));
    }

    /// ⛔ A DIMENSION PINNED TO A **NONZERO** CONSTANT ALSO ANSWERS `isDimValueZero` — the scan never
    /// looks at the constant column. Its extent is 1 either way, which is what both callers push.
    #[test]
    fn a_dimension_pinned_to_a_nonzero_constant_still_reads_as_pinned() {
        let pinned = IntegerSet {
            dims: 1,
            symbols: 0,
            constraints: vec![Constraint {
                expr: AffineExpr::dim(0).plus(AffineExpr::Const(-5)),
                is_equality: true,
            }],
        };
        assert!(FlatConstraints::from_integer_set(&pinned).is_dim_value_zero(0));
    }

    /// 🎯 THE FOUR CALLS `constructExtentAndTotalElements` OPENS WITH, ON IBM'S OWN SHAPE.
    ///
    /// The transfer set of a `[1, 1, 64]` access is `affine_set<(d0, d1, d2) : (d0 == 0, d1 == 0,
    /// d2 >= 0, -d2 + 63 >= 0)>` and its order is the identity, so composing the order, projecting
    /// out the set's own dimensions and simplifying must leave the SAME three extents — `[1, 1, 64]`.
    #[test]
    fn composing_the_identity_order_and_projecting_leaves_the_extents_alone() {
        let csts = FlatConstraints::from_integer_set(&IntegerSet::from_sizes(&[1, 1, 64]))
            .compose_matching_map(&AffineMap::identity(3))
            .project_out(3, 3)
            .remove_redundant_constraints();

        assert_eq!(csts.num_dim_vars(), 3);
        let extents: Vec<Option<i64>> = (0..3)
            .map(|dim| {
                if csts.is_dim_value_zero(dim) {
                    Some(1)
                } else {
                    csts.is_dim_a_constant_range(dim)
                }
            })
            .collect();
        assert_eq!(extents, vec![Some(1), Some(1), Some(64)]);
    }

    /// 🎯 AND A **TRANSPOSING** ORDER MOVES THEM, WHICH IS THE WHOLE REASON THE ORDER IS COMPOSED.
    ///
    /// `store_order = affine_map<(d0, d1) -> (d1, d0)>` over a `[1, 64]` set: read in the order's own
    /// input space the extents are `[64, 1]`, not `[1, 64]`.
    #[test]
    fn a_transposing_order_transposes_the_extents() {
        let transposed = AffineMap {
            dims: 2,
            syms: 0,
            results: vec![AffineExpr::dim(1), AffineExpr::dim(0)],
        };
        let csts = FlatConstraints::from_integer_set(&IntegerSet::from_sizes(&[1, 64]))
            .compose_matching_map(&transposed)
            .project_out(2, 2)
            .remove_redundant_constraints();

        assert_eq!(csts.num_dim_vars(), 2);
        assert_eq!(csts.is_dim_a_constant_range(0), Some(64));
        assert!(!csts.is_dim_value_zero(0));
        assert!(csts.is_dim_value_zero(1));
    }

    /// ⛔⛔ THE ZERO ROW GAUSSIAN ELIMINATION LEAVES BEHIND WOULD PIN EVERY DIMENSION.
    ///
    /// `projectOut` substitutes each tied dimension away with the equality that ties it, and MLIR
    /// drops that row as it goes. If one survived as `0 == 0`, `isDimValueZero` would answer true for
    /// EVERY dimension — the extents would come out all ones and the transfer would move one element.
    /// This is why the reference calls `removeRedundantConstraints()` immediately after
    /// (`AccessDetails.cpp:39-40`).
    #[test]
    fn projection_leaves_no_equality_that_pins_everything() {
        let csts = FlatConstraints::from_integer_set(&IntegerSet::from_sizes(&[1, 64]))
            .compose_matching_map(&AffineMap::identity(2))
            .project_out(2, 2);
        assert!(
            !csts
                .equalities
                .iter()
                .any(|row| row.iter().all(|t| *t == 0)),
            "an all-zero equality row makes every dimension read as pinned"
        );
        assert!(csts.is_dim_value_zero(0));
        assert!(!csts.is_dim_value_zero(1));
    }

    /// FOURIER–MOTZKIN, WHEN NO EQUALITY TIES THE VARIABLE DOWN.
    ///
    /// `d0 >= 0`, `-d0 + 7 >= 0`, `-d0 + d1 >= 0` — eliminating `d0` pairs its one lower bound with
    /// each upper bound: `0 + 7 >= 0` is trivially true and drops out, and `-0 + d1 >= 0` survives as
    /// the bound `d1` inherits.
    #[test]
    fn eliminating_a_variable_with_no_equality_combines_its_bounds() {
        let system = FlatConstraints {
            dims: 2,
            syms: 0,
            equalities: Vec::new(),
            inequalities: vec![vec![1, 0, 0], vec![-1, 0, 7], vec![-1, 1, 0]],
        };
        let projected = system.project_out(0, 1).remove_redundant_constraints();
        assert_eq!(projected.dims, 1);
        assert_eq!(projected.inequalities, vec![vec![1, 0]]);
        // A lower bound alone is not a constant range.
        assert_eq!(projected.is_dim_a_constant_range(0), None);
    }

    /// ⛔ THE GCD TIGHTENING IS AN INTEGER TIGHTENING, AND IT IS WHAT LETS THE PREDICATES INSIST ON ±1.
    ///
    /// `2*d0 - 3 >= 0` is `d0 >= 1.5`, so over the integers it is `d0 - 2 >= 0`. Paired with
    /// `-d0 + 5 >= 0` the width is `5 - 2 + 1 == 4`; without the tightening the row's coefficient is
    /// 2, `isDimAConstantRange` ignores it, and the dimension reports no range at all.
    #[test]
    fn a_scaled_lower_bound_is_tightened_to_a_unit_one() {
        let system = FlatConstraints {
            dims: 1,
            syms: 0,
            equalities: Vec::new(),
            inequalities: vec![vec![2, -3], vec![-1, 5]],
        };
        assert_eq!(
            system.is_dim_a_constant_range(0),
            None,
            "a coefficient of 2 is not a bound this predicate reads"
        );
        let tightened = system.remove_redundant_constraints();
        assert_eq!(tightened.inequalities, vec![vec![1, -2], vec![-1, 5]]);
        assert_eq!(tightened.is_dim_a_constant_range(0), Some(4));
    }

    /// THE TIGHTEST OF TWO ROWS WITH THE SAME COEFFICIENTS SURVIVES — `sum + c >= 0` is stronger for
    /// a SMALLER `c`, whichever side it bounds.
    #[test]
    fn duplicate_rows_collapse_onto_the_tightest_one() {
        let system = FlatConstraints {
            dims: 1,
            syms: 0,
            equalities: vec![vec![1, 0], vec![1, 0]],
            inequalities: vec![vec![1, 0], vec![1, -4], vec![-1, 63], vec![-1, 31]],
        };
        let simplified = system.remove_redundant_constraints();
        assert_eq!(simplified.equalities, vec![vec![1, 0]]);
        assert_eq!(simplified.inequalities, vec![vec![1, -4], vec![-1, 31]]);
        // `d0 >= 4` and `d0 <= 31`.
        assert_eq!(simplified.is_dim_a_constant_range(0), Some(28));
    }

    /// ⛔ AN EMPTY SYSTEM STAYS READABLE. `0 >= 1` is kept, for the reason
    /// [`IntegerSet::constant_bound`] documents: IBM writes empty mask sets and expects them to lower.
    #[test]
    fn a_trivially_false_row_is_not_dropped() {
        let system = FlatConstraints {
            dims: 1,
            syms: 0,
            equalities: Vec::new(),
            inequalities: vec![vec![0, -1], vec![1, 0]],
        };
        assert_eq!(
            system.remove_redundant_constraints().inequalities,
            vec![vec![0, -1], vec![1, 0]]
        );
    }

    /// ⭐⭐ THE COMPOSE THE VENDOR'S OWN VIEW AND SUBSCRIPT PAIR PERFORMS.
    ///
    /// `#map = affine_map<(d0, d1) -> (d0 * 64 + d1)>` is the `layout_map` of a `4x64` view, and a
    /// store subscripted `[%c0, %arg]` over a three-deep nest has an `affine_map` picking two of its
    /// three iterators. The composite is indexed by the NEST, not by the view — which is the whole
    /// reason `getLayoutMapAndIndices` returns the operands of the ACCESS beside the composed map.
    #[test]
    fn a_layout_composed_with_a_subscript_is_indexed_by_the_nest() {
        let layout = AffineMap::linear(&[1, 64]);
        let subscripts = AffineMap {
            dims: 3,
            syms: 0,
            results: vec![AffineExpr::dim(2), AffineExpr::dim(1)],
        };
        let composed = layout.compose(&subscripts);
        assert_eq!(composed.dims, 3);
        assert_eq!(composed.syms, 0);
        assert_eq!(
            composed.results,
            vec![AffineExpr::dim(2).plus(AffineExpr::dim(1).times(64))]
        );
    }

    /// ⛔⛔ COMPOSING WITH THE IDENTITY ORDER MAP RETURNS THE OTHER MAP UNTOUCHED, node for node.
    ///
    /// `indices_map = order_map.compose(indices_map)` is the first of the two composes in
    /// `getLayoutMapAndIndices`, and every access this island can state carries the identity order
    /// (see [`AffineMap::identity`]). The unchanged-children guard is what makes this exact rather
    /// than merely equivalent: no binary node is rebuilt, so no fold runs and nothing is renormalised.
    #[test]
    fn the_identity_order_map_composes_to_the_subscripts_verbatim() {
        let subscripts = AffineMap {
            dims: 2,
            syms: 0,
            results: vec![
                AffineExpr::dim(1).times(3).plus(AffineExpr::Const(7)),
                AffineExpr::dim(0),
            ],
        };
        assert_eq!(AffineMap::identity(2).compose(&subscripts), subscripts);
    }

    /// ⛔ A CONSTANT SUBSCRIPT FOLDS THROUGH THE STRIDE, and this is the one shape where the compose
    /// MUST fold to print what the reference prints: `#map3 = affine_map<(d0) -> (0, 0, 0)>` is a real
    /// time-address map in the scheduler's output, and `0 * 64 + 0` composed into a layout has to
    /// arrive as `0`. `getConstantBound` reads a literal row; a `Mul` of two literals is not one.
    #[test]
    fn a_constant_subscript_folds_to_a_literal_offset() {
        let composed = AffineMap::linear(&[1, 64]).compose(&AffineMap::constants(1, &[0, 0]));
        assert_eq!(composed.results, vec![AffineExpr::Const(0)]);

        let offset = AffineMap::linear(&[1, 64]).compose(&AffineMap {
            dims: 1,
            syms: 0,
            results: vec![AffineExpr::Const(3), AffineExpr::Const(2)],
        });
        assert_eq!(offset.results, vec![AffineExpr::Const(131)]);
    }

    /// ⛔ THE INNER MAP'S SYMBOLS SHIFT PAST THE OUTER MAP'S, so two `s0`s do not become one variable.
    #[test]
    fn composing_concatenates_the_symbols_outer_first() {
        let outer = AffineMap {
            dims: 1,
            syms: 1,
            results: vec![AffineExpr::dim(0).plus(AffineExpr::sym(0))],
        };
        let inner = AffineMap {
            dims: 1,
            syms: 1,
            results: vec![AffineExpr::sym(0)],
        };
        let composed = outer.compose(&inner);
        assert_eq!(composed.syms, 2);
        assert_eq!(
            composed.results,
            // `s1` is the INNER map's symbol; `s0` is still the outer's.
            vec![AffineExpr::sym(1).plus(AffineExpr::sym(0))]
        );
    }

    /// ⭐ `compressUnusedSymbols` RENUMBERS DENSELY AND IS A NO-OP ON A SYMBOL-FREE MAP — the two
    /// cases `getLayoutMapAndIndices` can reach.
    #[test]
    fn compressing_symbols_renumbers_only_the_ones_that_are_used() {
        let sparse = AffineMap {
            dims: 1,
            syms: 3,
            results: vec![AffineExpr::dim(0).plus(AffineExpr::sym(2))],
        };
        let compressed = sparse.compress_unused_symbols();
        assert_eq!(compressed.syms, 1);
        assert_eq!(
            compressed.results,
            vec![AffineExpr::dim(0).plus(AffineExpr::sym(0))]
        );

        let plain = AffineMap::linear(&[1, 64]);
        assert_eq!(plain.compress_unused_symbols(), plain);
    }

    /// ⛔ A PRODUCT OF TWO DIMENSIONS IS BUILT, NOT REWRITTEN — MLIR's affine-ness guard. Without it
    /// `fold_mul`'s canonicalisation would swap the operands forever.
    #[test]
    fn a_product_of_two_dimensions_is_left_alone() {
        let outer = AffineMap {
            dims: 2,
            syms: 0,
            results: vec![AffineExpr::Mul(
                Box::new(AffineExpr::dim(0)),
                Box::new(AffineExpr::dim(1)),
            )],
        };
        let composed = outer.compose(&AffineMap {
            dims: 2,
            syms: 0,
            results: vec![AffineExpr::dim(1), AffineExpr::dim(0)],
        });
        assert_eq!(
            composed.results,
            vec![AffineExpr::Mul(
                Box::new(AffineExpr::dim(1)),
                Box::new(AffineExpr::dim(0))
            )]
        );
    }

    /// ⭐ COMPOSING A SUBSCRIPTS MAP WITH ITS TRANSFER ORDER REORDERS THE SUBSCRIPTS — the one thing
    /// `ad.getTransferOrder().compose(ad.getSubscriptsMap())` is for (entries 191 and 192).
    ///
    /// An order of `(d0, d1) -> (d1, d0)` over subscripts `(d0 + 3, d1 * 8 + 5)` puts the SECOND
    /// subscript first, so the constant offsets entry 191 reads come out in transfer order: 5 then 3.
    #[test]
    fn compose_applies_the_transfer_order_to_the_subscripts() {
        let order = AffineMap {
            dims: 2,
            syms: 0,
            results: vec![AffineExpr::dim(1), AffineExpr::dim(0)],
        };
        let subscripts = AffineMap {
            dims: 2,
            syms: 0,
            results: vec![
                AffineExpr::dim(0).plus(AffineExpr::Const(3)),
                AffineExpr::dim(1).times(8).plus(AffineExpr::Const(5)),
            ],
        };
        let ordered = order.compose(&subscripts);
        assert_eq!(ordered.dims, 2);
        assert_eq!(ordered.syms, 0);
        assert_eq!(
            ordered.results,
            vec![
                AffineExpr::dim(1).times(8).plus(AffineExpr::Const(5)),
                AffineExpr::dim(0).plus(AffineExpr::Const(3)),
            ]
        );
        // And the constant columns come out in that order.
        let offsets: Vec<i64> = ordered
            .results
            .iter()
            .map(|expr| expr.flatten(ordered.dims, 0).constant)
            .collect();
        assert_eq!(offsets, vec![5, 3]);
    }

    /// ⭐ AND THE IDENTITY ORDER LEAVES THEM ALONE — the common case, an unpermuted transfer.
    #[test]
    fn composing_with_the_identity_order_changes_nothing() {
        let subscripts = AffineMap {
            dims: 3,
            syms: 0,
            results: vec![
                AffineExpr::Const(0),
                AffineExpr::dim(1).plus(AffineExpr::Const(64)),
                AffineExpr::dim(2),
            ],
        };
        assert_eq!(AffineMap::identity(3).compose(&subscripts), subscripts);
    }

    /// ⛔ THE COMPOSED MAP'S SYMBOL SPACES ARE CONCATENATED, `self`'s FIRST.
    ///
    /// MLIR renumbers the inner map's symbols to start after the outer map's and states the SUM as the
    /// result's symbol count, so the inner `s0` becomes `s1` here. Aliasing them would make one
    /// constraint row solve for the other map's parameter.
    #[test]
    fn compose_renumbers_the_inner_maps_symbols() {
        let outer = AffineMap {
            dims: 1,
            syms: 1,
            results: vec![AffineExpr::dim(0).plus(AffineExpr::sym(0))],
        };
        let inner = AffineMap {
            dims: 1,
            syms: 1,
            results: vec![AffineExpr::sym(0).plus(AffineExpr::Const(2))],
        };
        let composed = outer.compose(&inner);
        assert_eq!(composed.dims, 1);
        assert_eq!(composed.syms, 2);
        assert_eq!(
            composed.results,
            vec![
                AffineExpr::sym(1)
                    .plus(AffineExpr::Const(2))
                    .plus(AffineExpr::sym(0)),
            ]
        );
    }

    /// ⭐ THE CONSTANT COLUMN OF A FLATTENED SUBSCRIPT IS THE OFFSET ENTRIES 191 AND 192 SHIFT.
    #[test]
    fn flatten_reads_a_subscripts_constant_offset() {
        assert_eq!(AffineExpr::dim(0).flatten(2, 0).constant, 0);
        assert_eq!(
            AffineExpr::dim(0).plus(AffineExpr::Const(3)).flatten(2, 0),
            FlatAffineExpr {
                dims: vec![1, 0],
                syms: vec![],
                locals: vec![],
                constant: 3,
            }
        );
        assert_eq!(
            AffineExpr::dim(1)
                .times(8)
                .plus(AffineExpr::Const(5))
                .flatten(2, 0),
            FlatAffineExpr {
                dims: vec![0, 8],
                syms: vec![],
                locals: vec![],
                constant: 5,
            }
        );
        // A symbol has its own column, and a constant multiple of it scales that column.
        assert_eq!(
            AffineExpr::sym(0).times(-4).flatten(1, 2),
            FlatAffineExpr {
                dims: vec![0],
                syms: vec![-4, 0],
                locals: vec![],
                constant: 0,
            }
        );
    }

    /// ⛔ THE `coeffs.size() != num_dims` GUARD IS DEAD, WHICH IS WHY THE PORT DOES NOT WRITE IT.
    ///
    /// The row is `num_dims + num_syms + num_locals + 1` wide, so it is longer than `num_dims` for
    /// every expression that flattens at all — the `: 0` arm of entry 191's ternary is unreachable and
    /// what the guard really shields is a FAILED flattening, where MLIR leaves `coeffs` empty and
    /// `coeffs.back()` is undefined behaviour.
    #[test]
    fn the_row_is_always_wider_than_the_dimension_count() {
        for expr in [
            AffineExpr::Const(0),
            AffineExpr::dim(0),
            AffineExpr::dim(1).times(8).plus(AffineExpr::Const(5)),
            AffineExpr::dim(0).plus(AffineExpr::Const(5)).modulo(8),
        ] {
            let row = expr.flatten(2, 0);
            assert!(row.dims.len() + row.syms.len() + row.locals.len() + 1 > 2);
        }
    }

    /// ⭐ A `mod` WHOSE DIVIDEND IS A MULTIPLE OF THE MODULUS IS NOTHING AT ALL, and one that is not
    /// keeps its constant column while a local absorbs the remainder.
    ///
    /// `(d0 * 8) mod 4` is zero — every column divides by 4. `(d0 + 5) mod 8` still reports the
    /// offset 5, because `expr % c` flattens to `expr - c * q` and only `q`'s column carries the `-8`.
    #[test]
    fn flatten_of_a_mod_is_zero_only_when_the_dividend_is_a_multiple() {
        assert_eq!(
            AffineExpr::dim(0).times(8).modulo(4).flatten(1, 0),
            FlatAffineExpr::zero(1, 0)
        );
        let remainder = AffineExpr::dim(0)
            .plus(AffineExpr::Const(5))
            .modulo(8)
            .flatten(1, 0);
        assert_eq!(remainder.constant, 5);
        assert_eq!(remainder.dims, vec![1]);
        assert_eq!(remainder.locals.len(), 1);
        assert_eq!(remainder.locals[0].coeff, -8);
        assert_eq!(remainder.locals[0].divisor, 8);
    }

    /// ⭐ A `floordiv` THAT CANCELS DIVIDES THE ROW; ONE THAT DOES NOT ZEROES THE CONSTANT COLUMN.
    ///
    /// ⛔ AND THAT SECOND HALF IS LOAD-BEARING FOR ENTRY 191. `(d0 + 5) floordiv 8` offers NO constant
    /// offset to shift out — the 5 is inside the quotient — and reporting 5 there would move the
    /// immutable start address by eight times too much.
    #[test]
    fn flatten_of_a_floordiv_cancels_or_becomes_a_local() {
        assert_eq!(
            AffineExpr::dim(0)
                .times(8)
                .plus(AffineExpr::Const(16))
                .floordiv(8)
                .flatten(1, 0),
            FlatAffineExpr {
                dims: vec![1],
                syms: vec![],
                locals: vec![],
                constant: 2,
            }
        );
        let uncancelled = AffineExpr::dim(0)
            .plus(AffineExpr::Const(5))
            .floordiv(8)
            .flatten(1, 0);
        assert_eq!(uncancelled.constant, 0);
        assert_eq!(uncancelled.dims, vec![0]);
        assert_eq!(uncancelled.locals.len(), 1);
        assert_eq!(uncancelled.locals[0].coeff, 1);
    }

    /// ⛔ A LITERAL DIVIDEND IS FOLDED, AND FLOORED — what MLIR's own constructors do before the
    /// flattener is ever called (`simplifyFloorDiv`, `simplifyMod`).
    ///
    /// `-3 floordiv 4` is `-1` and `-3 mod 4` is `1`; truncating division would answer `0` and `-3`.
    #[test]
    fn a_literal_dividend_folds_the_way_mlirs_constructors_fold_it() {
        assert_eq!(AffineExpr::Const(-3).floordiv(4).flatten(1, 0).constant, -1);
        assert_eq!(AffineExpr::Const(-3).modulo(4).flatten(1, 0).constant, 1);
        assert_eq!(AffineExpr::Const(9).floordiv(4).flatten(1, 0).constant, 2);
        assert_eq!(AffineExpr::Const(9).modulo(4).flatten(1, 0).constant, 1);
    }

    /// ⛔ ONE LOCAL, ONE COLUMN — `findLocalId`, and the divisibility test that depends on it.
    ///
    /// `(d0 floordiv 2) + (d0 floordiv 2)` is one column with coefficient 2, so an enclosing `mod 2`
    /// cancels the whole expression. Two columns of 1 each — a flattener that appended blindly —
    /// would answer that nothing divides by 2 and report a constant offset of 4 for the `+ 4` below.
    #[test]
    fn a_repeated_local_shares_its_column() {
        let half = AffineExpr::dim(0).floordiv(2);
        let doubled = half.clone().plus(half);
        let row = doubled.flatten(1, 0);
        assert_eq!(row.locals.len(), 1);
        assert_eq!(row.locals[0].coeff, 2);
        assert_eq!(
            doubled
                .plus(AffineExpr::Const(4))
                .modulo(2)
                .flatten(1, 0)
                .constant,
            0
        );
    }

    /// ⭐ `isFunctionOfDim` IS AN IDENTITY TEST ON THE POSITION, NOT "IS THERE A DIMENSION".
    ///
    /// `d2 * 256 + d1 * 64 + d0` — the layout map all five `MutableAddrSplitting` answer keys carry
    /// (`mutable_addr_splitting_one_dim.mlir:5`) — is a function of each of its three dimensions and of
    /// nothing else.
    #[test]
    fn a_dimension_is_found_by_its_position_alone() {
        let expr = AffineExpr::dim(2)
            .times(256)
            .plus(AffineExpr::dim(1).times(64))
            .plus(AffineExpr::dim(0));
        assert!(expr.is_function_of_dim(0));
        assert!(expr.is_function_of_dim(1));
        assert!(expr.is_function_of_dim(2));
        assert!(!expr.is_function_of_dim(3));
    }

    /// ⛔⛔ AND A SYMBOL IS NOT A DIMENSION, whatever it holds.
    ///
    /// `TPMVBase::replaceDimsInMapWithSyms` rewrites a subscripts map's loop iterators AS SYMBOLS
    /// (`TransformPagedMemViewImpl.cpp:47`), so `s0` here is exactly the value `d0` was — and the
    /// answer is still `false`, for either position.
    #[test]
    fn a_symbol_is_not_a_dimension_at_any_position() {
        let expr = AffineExpr::sym(0).times(8).plus(AffineExpr::Const(64));
        assert!(!expr.is_function_of_dim(0));
        assert!(!expr.is_function_of_dim(1));
    }

    /// ⭐ AND IT REACHES THROUGH EVERY BINARY FORM, since the reference recurses on both operands of
    /// any `AffineBinaryOpExpr`.
    #[test]
    fn every_binary_form_is_searched_on_both_sides() {
        assert!(
            AffineExpr::dim(0)
                .modulo(128)
                .floordiv(2)
                .is_function_of_dim(0)
        );
        assert!(
            AffineExpr::Const(0)
                .plus(AffineExpr::dim(4).times(3))
                .is_function_of_dim(4)
        );
        assert!(!AffineExpr::Const(7).is_function_of_dim(0));
    }

    /// ⭐⭐ THE WHOLE OF `concatenateMaps` FOLLOWED BY `createExplicitTimeLoops`' SUBSTITUTION, OVER
    /// THE VENDOR'S OWN TWO MAPS.
    ///
    /// ```cpp
    /// auto shifted_map_B = map_B.shiftDims(map_A.getNumDims());
    /// for (int i = 0; i < map_A.getNumResults(); ++i)
    ///   exprs[i] = map_A.getResult(i) + shifted_map_B.getResult(i);
    /// return AffineMap::get(shifted_map_B.getNumDims(), 0, exprs, ...);
    /// ```
    /// (`dialect_utils/Agen/Utils.cpp:258-266`)
    ///
    /// The store of `mutable_addr_splitting_time_dims.mlir` carries
    /// `dst_map = affine_map<(d0, d1) -> (0, d0 * 16, d1 * 8)>` (`:8`) and
    /// `store_time_addr_map = affine_map<(d0, d1, d2) -> (d0 * 64, d1, d2 * 8)>` (`:6`), so the
    /// concatenation is five-dimensional: `(d0, .., d4) -> (d2 * 64, d0 * 16 + d3, d1 * 8 + d4 * 8)`.
    ///
    /// ⛔ AND THEN ONLY ONE TIME DIMENSION BECOMES A LOOP. `time_dim_idx` is 0 for this case, so
    /// `updateSubscriptsAndIndicesForExplicitTimeLoops` binds `d3` and `d4` to a constant 0 and keeps
    /// three dimensions (`Dialect/Agen/Utils.cpp:466-478`) — `d0 * 16 + 0` folds back to `d0 * 16`
    /// and `d1 * 8 + 0 * 8` to `d1 * 8`. ⭐ THAT the addresses come out UNCHANGED is the point: it is
    /// why the answer key's `dst_map` prints exactly as it went in.
    #[test]
    fn two_maps_concatenate_and_the_unused_time_dimensions_fold_away() {
        let dst_map = AffineMap {
            dims: 2,
            syms: 0,
            results: vec![
                AffineExpr::Const(0),
                AffineExpr::dim(0).times(16),
                AffineExpr::dim(1).times(8),
            ],
        };
        let time_addr_map = AffineMap {
            dims: 3,
            syms: 0,
            results: vec![
                AffineExpr::dim(0).times(64),
                AffineExpr::dim(1),
                AffineExpr::dim(2).times(8),
            ],
        };

        let shifted = time_addr_map.shift_dims(dst_map.dims);
        assert_eq!(
            print::affine_map(&shifted),
            "affine_map<(d0, d1, d2, d3, d4) -> (d2 * 64, d3, d4 * 8)>"
        );

        let concatenated = AffineMap {
            dims: shifted.dims,
            syms: 0,
            results: dst_map
                .results
                .iter()
                .zip(&shifted.results)
                .map(|(a, b)| a.clone().added(b.clone()))
                .collect(),
        };
        assert_eq!(
            print::affine_map(&concatenated),
            "affine_map<(d0, d1, d2, d3, d4) -> (d2 * 64, d0 * 16 + d3, d1 * 8 + d4 * 8)>"
        );

        let folded = concatenated.replace_dims_and_symbols(
            &[
                AffineExpr::dim(0),
                AffineExpr::dim(1),
                AffineExpr::dim(2),
                AffineExpr::Const(0),
                AffineExpr::Const(0),
            ],
            &[],
            3,
            0,
        );
        assert_eq!(
            print::affine_map(&folded),
            "affine_map<(d0, d1, d2) -> (d2 * 64, d0 * 16, d1 * 8)>"
        );
    }

    /// ⭐ A MAP'S ARITY GROWS WITH ITS SHIFT, and its symbols do not move
    /// (`llvm-project/mlir/include/mlir/IR/AffineMap.h:311-318`).
    #[test]
    fn shifting_a_map_widens_it_and_leaves_its_symbols_alone() {
        let map = AffineMap {
            dims: 2,
            syms: 1,
            results: vec![AffineExpr::dim(1).plus(AffineExpr::sym(0))],
        };
        let shifted = map.shift_dims(3);
        assert_eq!(shifted.dims, 5);
        assert_eq!(shifted.syms, 1);
        assert_eq!(
            print::affine_map(&shifted),
            "affine_map<(d0, d1, d2, d3, d4)[s0] -> (d4 + s0)>"
        );
    }

    /// ⭐ AND A MAP ANSWERS FOR EVERY RESULT AT ONCE — `isFunctionOfDim` over the whole map is an
    /// `any_of` (`llvm-project/mlir/include/mlir/IR/AffineMap.h:344-347`).
    #[test]
    fn a_map_is_a_function_of_a_dimension_any_result_mentions() {
        let map = AffineMap {
            dims: 3,
            syms: 0,
            results: vec![AffineExpr::Const(0), AffineExpr::dim(2).times(8)],
        };
        assert!(map.is_function_of_dim(2));
        assert!(!map.is_function_of_dim(0));
        assert!(!map.is_function_of_dim(1));
    }

    /// ⛔ `getDimPosition` ON A RESULT THAT IS NOT A BARE DIMENSION IS AN ABORT IN C++ AND A `None`
    /// HERE (`llvm-project/mlir/lib/IR/AffineMap.cpp:319-321`).
    ///
    /// The `time_order` of `mutable_addr_splitting_time_dims.mlir:5` is the reversal
    /// `affine_map<(d0, d1, d2) -> (d2, d1, d0)>`, so `updateTimeSetForExplicitDims` reads 2, 1, 0
    /// from it.
    #[test]
    fn a_dimension_position_is_reported_only_for_a_bare_dimension() {
        let reversal = AffineMap {
            dims: 3,
            syms: 0,
            results: vec![AffineExpr::dim(2), AffineExpr::dim(1), AffineExpr::dim(0)],
        };
        assert_eq!(reversal.dim_position(0), Some(2));
        assert_eq!(reversal.dim_position(2), Some(0));
        assert_eq!(reversal.dim_position(3), None);
        assert_eq!(AffineMap::unary(AffineExpr::Const(0)).dim_position(0), None);
    }
}
