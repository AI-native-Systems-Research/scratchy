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
        // `findEqualityToConstant(*this, 0, symbolic=false)`: a unit coefficient, and no other
        // dimension in the row. `-c / a` is exact because `a` is ±1.
        for constraint in &self.constraints {
            if let (true, Terms::OnPos { coeff, constant }) =
                (constraint.is_equality, terms_in(&constraint.expr, pos))
                && (coeff == 1 || coeff == -1)
            {
                return Some(-constant / coeff);
            }
        }

        let mut tightest: Option<i64> = None;
        for constraint in &self.constraints {
            let Terms::OnPos { coeff, constant } = terms_in(&constraint.expr, pos) else {
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
/// [`AffineExpr::Sym`] is [`Terms::OtherVars`] for every `pos` — never [`Terms::Const`], which
/// would let a parameterised bound be read as a literal one.
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
fn terms_in(expr: &AffineExpr, pos: u32) -> Terms {
    match expr {
        AffineExpr::Dim(n) if *n == pos => Terms::OnPos {
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
        AffineExpr::Add(a, b) => match (terms_in(a, pos), terms_in(b, pos)) {
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
                    "a constant bound on d{pos} from a constraint that also mentions another \
                     dimension needs the Fourier-Motzkin elimination \
                     FlatAffineValueConstraints::projectOut runs"
                )
            }
        },
        AffineExpr::Mul(a, b) => match (terms_in(a, pos), terms_in(b, pos)) {
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
        AffineExpr::Mod(a, b) => match (terms_in(a, pos), terms_in(b, pos)) {
            (Terms::Const(n), Terms::Const(d)) => Terms::Const(n.rem_euclid(d)),
            _ => todo!(
                "a constant bound across a `mod` needs the local variable MLIR's affine flattening \
                 introduces for it"
            ),
        },
        AffineExpr::FloorDiv(a, b) => match (terms_in(a, pos), terms_in(b, pos)) {
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
}

#[cfg(test)]
mod tests {
    use crate::islands::dataflow_ir::ty::{AffineExpr, BoundType, Constraint, IntegerSet};

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
        let scaled = |coeff: i64, c: i64| set(vec![ineq(AffineExpr::dim(0).times(coeff).plus(AffineExpr::Const(c)))]);
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
}
