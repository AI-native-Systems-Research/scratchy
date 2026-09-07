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
            ElemType::Int(bits) | ElemType::MxFloat(bits) => bits,
            ElemType::F16 | ElemType::Bf16 => 16,
            ElemType::F32 => 32,
            ElemType::F8E4M3Fn | ElemType::F8E8M0Fnu => 8,
            ElemType::F4E2M1Fn => 4,
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
/// ⛔ NO `AnyFloat`. `scalar_constant`'s result admits one (`SentientOps.td:850`) and prints its
/// value as a hex bit pattern when it is a float (`SentientOps.cpp:1704-1706`); nothing at this rung
/// emits one, so the case is absent rather than guessed at.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarTy {
    /// `index` — what every loop bound, address and induction variable is typed.
    Index,
    /// `i<bits>` — a signless integer. `i1` is the type of a predicate.
    Int(u32),
}

impl ScalarTy {
    /// HOW MLIR SPELLS IT.
    #[must_use]
    pub fn spelling(self) -> String {
        match self {
            ScalarTy::Index => "index".to_owned(),
            ScalarTy::Int(bits) => format!("i{bits}"),
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
            constraints,
        }
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
            results: (0..rank).map(AffineExpr::dim).collect(),
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
            results: vec![expr],
        }
    }
}
