// Copyright (c) 2026 IBM Corporation. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files
// (the "Software"), to deal in the Software without restriction,
// including without limitation the rights to use, copy, modify, merge,
// publish, distribute, sublicense, and/or sell copies of the Software,
// and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
// IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
// CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
// TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
// SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

//! THE KTIR VALUE TYPE -- and the ONE module boundary the vendoring decision
//! touches. See the crate docs for the table of which shapes are scratchy's.
//!
//! One IR model serves BOTH ttir and KTIR, which is not a shortcut: MLIR itself
//! has exactly one `Operation` class and distinguishes dialects by the op name.
//! A "ttir value" here is a [`Module`] whose [`OpKind`]s are `Tt*`/`Arith*`; a
//! "KTIR value" is the same [`Module`] after the passes have replaced those with
//! `Ktdp*`/`Ktdf*`/`Linalg*`. Having two structurally identical types would mean
//! a conversion pass that could only ever be the identity.

use std::collections::HashMap;

//===----------------------------------------------------------------------===//
// Ssa
//===----------------------------------------------------------------------===//

/// AN SSA VALUE'S NAME -- an identity, not a spelling.
///
/// Identical to scratchy's `ktir::ir::Ssa`. The spelling a value had in text is
/// kept out of band in [`Module::hints`], because it is a DIAGNOSTIC, and letting
/// passes read it invites a pass that behaves differently for `%acc` than for
/// `%17`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Ssa(pub u32);

impl Ssa {
    pub fn slot(self) -> usize {
        self.0 as usize
    }
}

//===----------------------------------------------------------------------===//
// OpKind
//===----------------------------------------------------------------------===//

/// THE OP VOCABULARY, AS A TYPE. Same shape as scratchy's `OpKind`: one variant
/// per op spelling, so a typo is a compile error rather than a run-time miss.
///
/// [`OpKind::Other`] is the escape hatch and it is NOT a silent one: an op that
/// lands there is refused by name at the first pass that has to understand it
/// (see `passes::census`), which is the fail-closed behaviour. It exists so the
/// parser can READ an unfamiliar op and then refuse it with its real spelling in
/// the message, instead of failing with "unparseable line".
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum OpKind {
    // --- builtin / structural
    Module,
    TtFunc,
    TtReturn,
    /// `func.func` -- the Triton-free container the scheduler parses.
    FuncFunc,
    /// `func.return`.
    FuncReturn,
    ScfFor,
    ScfYield,
    UnrealizedConversionCast,

    // --- arith
    ArithConstant,
    ArithAddf,
    ArithAddi,
    ArithSubf,
    ArithMulf,
    ArithMuli,
    ArithDivf,
    ArithDivsi,
    ArithDivui,
    ArithRemsi,
    ArithRemui,
    ArithMaxnumf,
    ArithMinnumf,
    ArithCmpi,
    ArithAndi,
    ArithExtf,
    ArithTruncf,
    ArithIndexCast,
    ArithSelect,

    // --- math
    MathExp2,
    MathExp,
    MathLog2,
    MathSqrt,

    // --- tensor
    TensorSplat,
    TensorEmpty,
    TensorCollapseShape,
    TensorExpandShape,

    // --- triton
    TtGetProgramId,
    TtMakeTensorDescriptor,
    TtDescriptorLoad,
    TtDescriptorStore,
    TtDescriptorGather,
    TtDescriptorScatter,
    TtLoad,
    TtStore,
    TtDot,
    TtTrans,
    TtReduce,
    TtReduceReturn,
    TtExpandDims,
    TtBroadcast,
    TtSplat,
    TtMakeRange,
    TtAddptr,

    // --- linalg
    LinalgMatmul,
    LinalgGeneric,
    LinalgReduce,
    LinalgYield,

    // --- ktdp
    KtdpGetComputeTileId,
    KtdpConstructMemoryView,
    KtdpConstructAccessTile,
    KtdpConstructIndirectAccessTile,
    KtdpLoad,
    KtdpStore,

    // --- ktdf
    KtdfCoreletPlan,
    KtdfCorelet,

    /// An op this crate has no variant for, kept with its spelling so a refusal
    /// can name it.
    Other(String),
}

impl OpKind {
    /// The MLIR spelling. The printer and the parser share this table, so a
    /// round trip cannot drift.
    pub fn spelling(&self) -> &str {
        use OpKind::*;
        match self {
            Module => "builtin.module",
            TtFunc => "tt.func",
            TtReturn => "tt.return",
            FuncFunc => "func.func",
            FuncReturn => "func.return",
            ScfFor => "scf.for",
            ScfYield => "scf.yield",
            UnrealizedConversionCast => "builtin.unrealized_conversion_cast",
            ArithConstant => "arith.constant",
            ArithAddf => "arith.addf",
            ArithAddi => "arith.addi",
            ArithSubf => "arith.subf",
            ArithMulf => "arith.mulf",
            ArithMuli => "arith.muli",
            ArithDivf => "arith.divf",
            ArithDivsi => "arith.divsi",
            ArithDivui => "arith.divui",
            ArithRemsi => "arith.remsi",
            ArithRemui => "arith.remui",
            ArithMaxnumf => "arith.maxnumf",
            ArithMinnumf => "arith.minnumf",
            ArithCmpi => "arith.cmpi",
            ArithAndi => "arith.andi",
            ArithExtf => "arith.extf",
            ArithTruncf => "arith.truncf",
            ArithIndexCast => "arith.index_cast",
            ArithSelect => "arith.select",
            MathExp2 => "math.exp2",
            MathExp => "math.exp",
            MathLog2 => "math.log2",
            MathSqrt => "math.sqrt",
            TensorSplat => "tensor.splat",
            TensorEmpty => "tensor.empty",
            TensorCollapseShape => "tensor.collapse_shape",
            TensorExpandShape => "tensor.expand_shape",
            TtGetProgramId => "tt.get_program_id",
            TtMakeTensorDescriptor => "tt.make_tensor_descriptor",
            TtDescriptorLoad => "tt.descriptor_load",
            TtDescriptorStore => "tt.descriptor_store",
            TtDescriptorGather => "tt.descriptor_gather",
            TtDescriptorScatter => "tt.descriptor_scatter",
            TtLoad => "tt.load",
            TtStore => "tt.store",
            TtDot => "tt.dot",
            TtTrans => "tt.trans",
            TtReduce => "tt.reduce",
            TtReduceReturn => "tt.reduce.return",
            TtExpandDims => "tt.expand_dims",
            TtBroadcast => "tt.broadcast",
            TtSplat => "tt.splat",
            TtMakeRange => "tt.make_range",
            TtAddptr => "tt.addptr",
            LinalgMatmul => "linalg.matmul",
            LinalgGeneric => "linalg.generic",
            LinalgReduce => "linalg.reduce",
            LinalgYield => "linalg.yield",
            KtdpGetComputeTileId => "ktdp.get_compute_tile_id",
            KtdpConstructMemoryView => "ktdp.construct_memory_view",
            KtdpConstructAccessTile => "ktdp.construct_access_tile",
            KtdpConstructIndirectAccessTile => "ktdp.construct_indirect_access_tile",
            KtdpLoad => "ktdp.load",
            KtdpStore => "ktdp.store",
            KtdfCoreletPlan => "ktdf.corelet_plan",
            KtdfCorelet => "ktdf.corelet",
            Other(s) => s,
        }
    }

    /// Parse a spelling. Unknown spellings become [`OpKind::Other`] so the
    /// refusal that follows can quote the real name.
    pub fn from_spelling(s: &str) -> OpKind {
        use OpKind::*;
        match s {
            "builtin.module" | "module" => Module,
            "tt.func" => TtFunc,
            "func.func" => FuncFunc,
            "tt.return" => TtReturn,
            "func.return" | "return" => FuncReturn,
            "scf.for" => ScfFor,
            "scf.yield" => ScfYield,
            "builtin.unrealized_conversion_cast" | "unrealized_conversion_cast" => {
                UnrealizedConversionCast
            }
            "arith.constant" => ArithConstant,
            "arith.addf" => ArithAddf,
            "arith.addi" => ArithAddi,
            "arith.subf" => ArithSubf,
            "arith.mulf" => ArithMulf,
            "arith.muli" => ArithMuli,
            "arith.divf" => ArithDivf,
            "arith.divsi" => ArithDivsi,
            "arith.divui" => ArithDivui,
            "arith.remsi" => ArithRemsi,
            "arith.remui" => ArithRemui,
            "arith.maxnumf" => ArithMaxnumf,
            "arith.minnumf" => ArithMinnumf,
            "arith.cmpi" => ArithCmpi,
            "arith.andi" => ArithAndi,
            "arith.extf" => ArithExtf,
            "arith.truncf" => ArithTruncf,
            "arith.index_cast" => ArithIndexCast,
            "arith.select" => ArithSelect,
            "math.exp2" => MathExp2,
            "math.exp" => MathExp,
            "math.log2" => MathLog2,
            "math.sqrt" => MathSqrt,
            "tensor.splat" => TensorSplat,
            "tensor.empty" => TensorEmpty,
            "tensor.collapse_shape" => TensorCollapseShape,
            "tensor.expand_shape" => TensorExpandShape,
            "tt.get_program_id" => TtGetProgramId,
            "tt.make_tensor_descriptor" => TtMakeTensorDescriptor,
            "tt.descriptor_load" => TtDescriptorLoad,
            "tt.descriptor_store" => TtDescriptorStore,
            "tt.descriptor_gather" => TtDescriptorGather,
            "tt.descriptor_scatter" => TtDescriptorScatter,
            "tt.load" => TtLoad,
            "tt.store" => TtStore,
            "tt.dot" => TtDot,
            "tt.trans" => TtTrans,
            "tt.reduce" => TtReduce,
            "tt.reduce.return" => TtReduceReturn,
            "tt.expand_dims" => TtExpandDims,
            "tt.broadcast" => TtBroadcast,
            "tt.splat" => TtSplat,
            "tt.make_range" => TtMakeRange,
            "tt.addptr" => TtAddptr,
            "linalg.matmul" => LinalgMatmul,
            "linalg.generic" => LinalgGeneric,
            "linalg.reduce" => LinalgReduce,
            "linalg.yield" => LinalgYield,
            "ktdp.get_compute_tile_id" => KtdpGetComputeTileId,
            "ktdp.construct_memory_view" => KtdpConstructMemoryView,
            "ktdp.construct_access_tile" => KtdpConstructAccessTile,
            "ktdp.construct_indirect_access_tile" => KtdpConstructIndirectAccessTile,
            "ktdp.load" => KtdpLoad,
            "ktdp.store" => KtdpStore,
            "ktdf.corelet_plan" => KtdfCoreletPlan,
            "ktdf.corelet" => KtdfCorelet,
            other => Other(other.to_string()),
        }
    }

    /// Does this op carry regions? Used by the parser to decide whether a
    /// trailing `{` opens a body or an attribute dictionary.
    pub fn is_region_carrying(&self) -> bool {
        matches!(
            self,
            OpKind::Module
                | OpKind::TtFunc
                | OpKind::FuncFunc
                | OpKind::ScfFor
                | OpKind::TtReduce
                | OpKind::LinalgGeneric
                | OpKind::LinalgReduce
                | OpKind::KtdfCoreletPlan
                | OpKind::KtdfCorelet
        )
    }

    /// Is this op free of side effects, so DCE may erase it when its results are
    /// unused?
    ///
    /// `ktdp.load` is DELIBERATELY NOT in this set even though it only reads. The
    /// C++ says why, at `FoldTransIntoAccessTileOrder`: "ktdp.load declares no
    /// memory effects, so the greedy driver treats it as potentially
    /// side-effecting and will NOT DCE it". Reproducing the C++'s DCE means
    /// reproducing that, or a dead untransposed read disappears here and survives
    /// there, and the op counts diverge for a reason no diff would explain.
    pub fn is_pure(&self) -> bool {
        use OpKind::*;
        matches!(
            self,
            ArithConstant
                | ArithAddf
                | ArithAddi
                | ArithSubf
                | ArithMulf
                | ArithMuli
                | ArithDivf
                | ArithDivsi
                | ArithDivui
                | ArithRemsi
                | ArithRemui
                | ArithMaxnumf
                | ArithMinnumf
                | ArithCmpi
                | ArithAndi
                | ArithExtf
                | ArithTruncf
                | ArithIndexCast
                | ArithSelect
                | MathExp2
                | MathExp
                | MathLog2
                | MathSqrt
                | TensorSplat
                | TensorEmpty
                | TensorCollapseShape
                | TensorExpandShape
                | TtExpandDims
                | TtBroadcast
                | TtSplat
                | TtMakeRange
                | TtTrans
                | UnrealizedConversionCast
                | KtdpConstructMemoryView
                | KtdpConstructAccessTile
                | LinalgMatmul
        )
    }
}

//===----------------------------------------------------------------------===//
// AttrKey / Attr
//===----------------------------------------------------------------------===//

/// AN ATTRIBUTE'S KEY, AS A TYPE. Same shape as scratchy's `AttrKey`.
///
/// The keys scratchy also declares keep scratchy's variant name, so a swap is a
/// rename table and not a semantic review.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum AttrKey {
    /// `value` -- an `arith.constant`'s payload.
    Value,
    /// `axis` -- `tt.reduce` / `tt.expand_dims`.
    Axis,
    /// `order` -- `tt.trans`.
    Order,
    /// `dimensions` -- `linalg.reduce`.
    Dimensions,
    /// `indexing_maps` -- `linalg.generic`.
    IndexingMaps,
    /// `iterator_types` -- `linalg.generic`.
    IteratorTypes,
    /// `sizes` -- `ktdp.construct_memory_view` static sizes.
    Shape,
    /// `strides` -- `ktdp.construct_memory_view` static strides.
    Strides,
    /// `coordinate_set` -- the memory view's range set.
    CoordinateSet,
    /// `memory_space`.
    MemorySpace,
    /// `access_tile_set`.
    AccessTileSet,
    /// `access_tile_order`.
    AccessTileOrder,
    /// `base_map`.
    BaseMap,
    /// `pattern` -- `ktdf.corelet_plan`.
    Pattern,
    /// `work_division` -- `ktdf.corelet_plan`.
    WorkDivision,
    /// `index` -- `ktdf.corelet`.
    Index,
    /// `data_bounds` -- `ktdf.corelet`.
    DataBounds,
    /// `output_partition` -- `ktdf.corelet`.
    OutputPartition,
    /// `pt_rows` -- `ktdf.corelet`.
    PtRows,
    /// `xrf_capacity` -- `ktdf.corelet`.
    XrfCapacity,
    /// `role` -- `ktdf.corelet`.
    Role,
    /// `ring_send` -- `ktdf.corelet`.
    RingSend,
    /// `ring_recv` -- `ktdf.corelet`.
    RingRecv,
    /// `sym_name` -- the function's `@name`.
    SymName,
    /// `grid` -- the 1-D launch extent the scheduler reads off `func.func`.
    Grid,
    /// `reassociation` -- `tensor.collapse_shape`/`expand_shape`.
    Reassociation,
    /// `spyre.folded_grid_loop` -- THE LAUNCH CONTRACT: launch exactly `work_items`
    /// compute tiles, because the loop's zero-trip guard was folded away.
    FoldedGridLoop,
    /// `spyre.canonical_verified` -- DotToLinalg's P1.0 trust tag.
    CanonicalVerified,
    /// `noinline`.
    Noinline,
    /// `predicate` -- `arith.cmpi` / `arith.cmpf`'s comparison keyword (`slt`, `oge`).
    ///
    /// # THIS IS A BARE KEYWORD IN MLIR, NOT AN ATTRIBUTE, AND MISSING IT IS SILENT
    ///
    /// `arith.cmpi slt, %a, %b : i32` prints the predicate as part of the assembly form
    /// with no `= ` and no dictionary, exactly like `tt.get_program_id`'s axis. So a
    /// generic attribute reader finds nothing and the op arrives with NO predicate --
    /// which for a consumer that maps `slt` onto a `lesserthan` opFuncName means every
    /// comparison silently becomes the same one. It is modelled here so the consumer can
    /// read a field, and refused by name when absent.
    Predicate,
    /// Any attribute this crate has no variant for, kept verbatim.
    Other(String),
}

impl AttrKey {
    pub fn spelling(&self) -> &str {
        use AttrKey::*;
        match self {
            Value => "value",
            Axis => "axis",
            Order => "order",
            Dimensions => "dimensions",
            IndexingMaps => "indexing_maps",
            IteratorTypes => "iterator_types",
            Shape => "sizes",
            Strides => "strides",
            CoordinateSet => "coordinate_set",
            MemorySpace => "memory_space",
            AccessTileSet => "access_tile_set",
            AccessTileOrder => "access_tile_order",
            BaseMap => "base_map",
            Pattern => "pattern",
            WorkDivision => "work_division",
            Index => "index",
            DataBounds => "data_bounds",
            OutputPartition => "output_partition",
            PtRows => "pt_rows",
            XrfCapacity => "xrf_capacity",
            Role => "role",
            RingSend => "ring_send",
            RingRecv => "ring_recv",
            SymName => "sym_name",
            Grid => "grid",
            Reassociation => "reassociation",
            FoldedGridLoop => "spyre.folded_grid_loop",
            CanonicalVerified => "spyre.canonical_verified",
            Noinline => "noinline",
            Predicate => "predicate",
            Other(s) => s,
        }
    }

    pub fn from_spelling(s: &str) -> AttrKey {
        use AttrKey::*;
        match s {
            "value" => Value,
            "axis" => Axis,
            "order" => Order,
            "dimensions" => Dimensions,
            "indexing_maps" => IndexingMaps,
            "iterator_types" => IteratorTypes,
            "sizes" => Shape,
            "strides" => Strides,
            "coordinate_set" => CoordinateSet,
            "memory_space" => MemorySpace,
            "access_tile_set" => AccessTileSet,
            "access_tile_order" => AccessTileOrder,
            "base_map" => BaseMap,
            "pattern" => Pattern,
            "work_division" => WorkDivision,
            "index" => Index,
            "data_bounds" => DataBounds,
            "output_partition" => OutputPartition,
            "pt_rows" => PtRows,
            "xrf_capacity" => XrfCapacity,
            "role" => Role,
            "ring_send" => RingSend,
            "ring_recv" => RingRecv,
            "sym_name" => SymName,
            "grid" => Grid,
            "reassociation" => Reassociation,
            "spyre.folded_grid_loop" => FoldedGridLoop,
            "spyre.canonical_verified" => CanonicalVerified,
            "noinline" => Noinline,
            "predicate" => Predicate,
            other => Other(other.to_string()),
        }
    }
}

/// A floating-point constant, kept as BITS plus its width.
///
/// NOT an `f64`. A splat constant's whole job here is to survive
/// `LegalizeTypes`'s re-round to IEEE half and then be PRINTED the way MLIR
/// prints it, and an `f64` round trip through decimal is exactly where
/// `1.275630e-01` becomes `1.2756e-01`. Bits compare and hash exactly, which a
/// float does not.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FloatBits {
    pub bits: u64,
    pub width: u32,
}

impl FloatBits {
    pub fn f32(v: f32) -> FloatBits {
        FloatBits { bits: v.to_bits() as u64, width: 32 }
    }

    pub fn f16_from_f32(v: f32) -> FloatBits {
        FloatBits { bits: f32_to_f16_bits(v) as u64, width: 16 }
    }

    pub fn as_f64(self) -> f64 {
        match self.width {
            16 => f16_bits_to_f32(self.bits as u16) as f64,
            32 => f32::from_bits(self.bits as u32) as f64,
            _ => f64::from_bits(self.bits),
        }
    }

    /// Re-round to IEEE half, the way `LegalizeTypes::convertF32ValueAttr` does
    /// (`APFloat::convert(IEEEhalf, rmNearestTiesToEven)`).
    pub fn to_f16(self) -> FloatBits {
        if self.width == 16 {
            return self;
        }
        FloatBits::f16_from_f32(self.as_f64() as f32)
    }

    pub fn is_zero(self) -> bool {
        // Both signed zeros.
        let mask = match self.width {
            16 => 0x7fff,
            32 => 0x7fff_ffff,
            _ => u64::MAX >> 1,
        };
        self.bits & mask == 0
    }
}

/// A parsed attribute. Same variants as scratchy's `Attr`, owned rather than
/// arena-borrowed (see the crate docs for why).
#[derive(Clone, Debug, PartialEq)]
pub enum Attr {
    Unit,
    Int(i64),
    IntList(Vec<i64>),
    Float(FloatBits),
    Str(String),
    StrList(Vec<String>),
    Bool(bool),
    /// A dense constant that is a SPLAT: one value, fanned out to `ty`.
    SplatFloat(FloatBits),
    /// An affine map, as its printed body (`"(d0, d1) -> (d1, d0)"`).
    ///
    /// Kept as text ON PURPOSE, and it is the one place this module does. The
    /// passes ported here only ever build identity maps, permutations of them, and
    /// the three fixed `linalg.generic` maps; NONE of them inspects a map's
    /// interior. scratchy has a real `AffineMap` (1378 lines of `affine.rs`) and a
    /// swap adopts it -- which is exactly why this stays behind [`Attr`] instead
    /// of being spread through the passes.
    AffineMap(String),
    AffineMapList(Vec<String>),
    /// An affine set (`ktdp` coordinate/tile sets), same reasoning as
    /// [`Attr::AffineMap`].
    AffineSet(String),
    /// A verbatim attribute this crate does not model, kept so it round-trips.
    Verbatim(String),
}

impl Attr {
    pub fn as_int(&self) -> Option<i64> {
        match self {
            Attr::Int(v) => Some(*v),
            _ => None,
        }
    }
    pub fn as_int_list(&self) -> Option<&[i64]> {
        match self {
            Attr::IntList(v) => Some(v),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Attr::Str(s) => Some(s),
            _ => None,
        }
    }
    /// The single float of a scalar or splat float attribute.
    pub fn as_float(&self) -> Option<FloatBits> {
        match self {
            Attr::Float(f) | Attr::SplatFloat(f) => Some(*f),
            _ => None,
        }
    }
}

//===----------------------------------------------------------------------===//
// IrType
//===----------------------------------------------------------------------===//

/// An element type. Same role as scratchy's `DType`, narrowed to what this
/// backend's fixtures actually carry -- a wider set is a variant away.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum DType {
    F16,
    F32,
    I1,
    I32,
    /// `si32` -- MLIR's SIGNED integer spelling.
    ///
    /// # A SEPARATE VARIANT, NOT AN ALIAS FOR `I32`, AND THE REASON IS MEASURED
    ///
    /// MLIR drops Triton's signedness in most positions (`i32`), but NOT inside a
    /// `!tt.tensordesc<>` block type -- and `ConvertTTIRToKTDP` carries that spelling
    /// through onto the memory view it builds, so `embedding_granite`'s KTIR contains
    /// `ktdp.construct_memory_view ... : memref<256xsi32>`. Collapsing it to `I32` would
    /// print `memref<256xi32>` on a round trip, which is a different type spelling from
    /// the one the C++ emitted; keeping the variant means the reader and the printer agree
    /// with the toolchain instead of with each other.
    SI32,
    I64,
    Index,
    /// `f8E4M3FN` -- OCP E4M3, the ONE fp8 variant this backend carries, and the one
    /// scratchy's W8A8 chain is card-proven on (`DType::Fp8E4m3`, `ktir_matmul_fp8.rs`).
    /// It exists in this enum for the WEIGHT VIEW only: a Triton kernel loads an fp8
    /// weight through a descriptor whose memref elem is `f8E4M3FN`, and the view's
    /// `AttrKey::Dtype` -- the attribute `ktir-superdsc`'s `regions()` reads to set
    /// `is_fp8` -- must carry it or the arity-3 fp8 door refuses ("the weight view
    /// declares f16 but the program binds 3 input(s)"). A COMPUTED value is never fp8
    /// here: the frontend refuses a mixed `tl.dot(f16, fp8)` (`semantic.py`'s "Both
    /// operands must be same dtype"), so the kernel widens with `.to(tl.float16)` and
    /// the load's RESULT type is f16 already -- which is also scratchy's own contract
    /// (`KTIR_ELEM` types even the fp8 weight load's result F16; `ktdp.load` widens on
    /// read via `TileStorage::Fp8E4m3`).
    ///
    /// # 1 BYTE PER ELEMENT, AND THE SPELLING IS THE DEVICE'S
    ///
    /// MLIR spells it `f8E4M3FN` (triton-frontend parses and mangles it; Triton's
    /// Python name is `fp8e4nv`). One byte per element, packed -- that IS the residency
    /// the device reads, so a host stage for an fp8 weight is byte-per-element at
    /// 128-element sticks, half an f16 stage's size. E5M2 (`f8E5M2`) is deliberately NOT
    /// a variant: nothing in scratchy's chain is proven on it and a second fp8 spelling
    /// would be an invitation to spell a view the device cannot read.
    Fp8E4m3,
}

impl DType {
    pub fn spelling(self) -> &'static str {
        match self {
            DType::F16 => "f16",
            DType::F32 => "f32",
            DType::I1 => "i1",
            DType::I32 => "i32",
            DType::SI32 => "si32",
            DType::I64 => "i64",
            DType::Index => "index",
            DType::Fp8E4m3 => "f8E4M3FN",
        }
    }
    pub fn from_spelling(s: &str) -> Option<DType> {
        Some(match s {
            "f16" => DType::F16,
            "f32" => DType::F32,
            "i1" => DType::I1,
            "i32" => DType::I32,
            "si32" => DType::SI32,
            "i64" => DType::I64,
            "index" => DType::Index,
            "f8E4M3FN" => DType::Fp8E4m3,
            _ => return None,
        })
    }
    /// An fp8 view IS a float view -- it names what the bytes MEAN, not a different
    /// kind of arithmetic. `is_float` gates where a float is legal, and the fp8 weight
    /// view must not be excluded from those positions by accident while its RESULT
    /// values stay f16.
    pub fn is_float(self) -> bool {
        matches!(self, DType::F16 | DType::F32 | DType::Fp8E4m3)
    }
}

/// `ShapedType::kDynamic`, spelled `?`. Same sentinel MLIR uses, so a dynamic
/// extent stays representable rather than being guessed at.
pub const DYNAMIC: i64 = i64::MIN;

/// A VALUE'S TYPE, AS A TYPE -- same shape as scratchy's `IrType`, extended with
/// the two Triton wrapper types the descriptor path needs.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum IrType {
    /// `tensor<D0xD1x...xE>`
    Tensor { dims: Vec<i64>, elem: DType },
    /// `memref<D0xD1x...xE>`
    MemRef { dims: Vec<i64>, elem: DType },
    /// `!ktdp.access_tile<D0x...xindex>`
    AccessTile { dims: Vec<i64> },
    /// `!tt.ptr<E>`
    Ptr { elem: DType },
    /// `!tt.tensordesc<D0x...xE>` -- the BLOCK shape, which is what the
    /// descriptor patterns read (never the load's result type; a rank-reduced
    /// load would otherwise build a tile that mismatches the view).
    TensorDesc { dims: Vec<i64>, elem: DType },
    Index,
    Scalar(DType),
    /// A type this crate does not model, kept verbatim so it round-trips.
    Verbatim(String),
}

impl IrType {
    pub fn dims(&self) -> Option<&[i64]> {
        match self {
            IrType::Tensor { dims, .. }
            | IrType::MemRef { dims, .. }
            | IrType::AccessTile { dims }
            | IrType::TensorDesc { dims, .. } => Some(dims),
            _ => None,
        }
    }

    pub fn elem(&self) -> Option<DType> {
        match self {
            IrType::Tensor { elem, .. }
            | IrType::MemRef { elem, .. }
            | IrType::TensorDesc { elem, .. }
            | IrType::Ptr { elem } => Some(*elem),
            IrType::Scalar(e) => Some(*e),
            IrType::Index => Some(DType::Index),
            IrType::AccessTile { .. } | IrType::Verbatim(_) => None,
        }
    }

    pub fn rank(&self) -> usize {
        self.dims().map(|d| d.len()).unwrap_or(0)
    }

    /// The same type over a different element type. This is how `LegalizeTypes`
    /// retypes an f32 result to f16 without touching the shape.
    pub fn with_elem(&self, e: DType) -> IrType {
        match self {
            IrType::Tensor { dims, .. } => IrType::Tensor { dims: dims.clone(), elem: e },
            IrType::MemRef { dims, .. } => IrType::MemRef { dims: dims.clone(), elem: e },
            IrType::TensorDesc { dims, .. } => {
                IrType::TensorDesc { dims: dims.clone(), elem: e }
            }
            IrType::Scalar(_) => IrType::Scalar(e),
            other => other.clone(),
        }
    }

    pub fn with_dims(&self, d: Vec<i64>) -> IrType {
        match self {
            IrType::Tensor { elem, .. } => IrType::Tensor { dims: d, elem: *elem },
            IrType::MemRef { elem, .. } => IrType::MemRef { dims: d, elem: *elem },
            IrType::AccessTile { .. } => IrType::AccessTile { dims: d },
            IrType::TensorDesc { elem, .. } => IrType::TensorDesc { dims: d, elem: *elem },
            other => other.clone(),
        }
    }

    pub fn num_elements(&self) -> Option<i64> {
        let d = self.dims()?;
        if d.contains(&DYNAMIC) {
            return None;
        }
        Some(d.iter().product())
    }

    pub fn has_static_shape(&self) -> bool {
        self.dims().map(|d| !d.contains(&DYNAMIC)).unwrap_or(true)
    }

    /// Is this a COMPUTE-DOMAIN f32 -- an f32 scalar or an f32-elemented tensor?
    /// `LegalizeTypes::isComputeF32`.
    pub fn is_compute_f32(&self) -> bool {
        matches!(self, IrType::Scalar(DType::F32) | IrType::Tensor { elem: DType::F32, .. })
    }

    /// Does this carry f32 but sit OUT OF SCOPE for `LegalizeTypes` -- a ptr,
    /// tensordesc or memref? `LegalizeTypes::isOutOfScopeF32`.
    pub fn is_out_of_scope_f32(&self) -> bool {
        matches!(
            self,
            IrType::Ptr { .. } | IrType::TensorDesc { .. } | IrType::MemRef { .. }
        )
    }

    /// `getElementTypeOrSelf`.
    pub fn element_or_self(&self) -> Option<DType> {
        self.elem()
    }
}

//===----------------------------------------------------------------------===//
// Op / Region / Func / Module
//===----------------------------------------------------------------------===//

/// A block argument list: an entry block's `(%a: T, %b: T)`.
///
/// An argument IS an SSA value, exactly as in scratchy's `IRFunction::arguments`.
pub type BlockArgs = Vec<(Ssa, IrType)>;

/// A region: block arguments plus a flat op list.
///
/// One block per region. MLIR allows more; NOTHING in the nine ported passes
/// produces a multi-block region, and a form no pass emits is a form no test
/// covers -- so it is refused at the parser rather than half-modelled.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Region {
    pub args: BlockArgs,
    pub ops: Vec<Op>,
}

/// A single operation.
///
/// [`Op::results`] is a LIST, where scratchy's `Operation::result` is one
/// `Option<Ssa>` plus a `result_names` attribute. `scf.for` with five `iter_args`
/// defines five values and every pass here rewires them individually, so the list
/// is the form that makes "rewired result 3 but not result 4" a normal edit
/// instead of an attribute update. A swap to scratchy's shape moves the tail into
/// `result_names`.
#[derive(Clone, Debug, PartialEq)]
pub struct Op {
    pub kind: OpKind,
    pub results: Vec<Ssa>,
    pub result_types: Vec<IrType>,
    pub operands: Vec<Ssa>,
    pub attrs: Vec<(AttrKey, Attr)>,
    pub regions: Vec<Region>,
}

impl Op {
    pub fn new(kind: OpKind) -> Op {
        Op {
            kind,
            results: Vec::new(),
            result_types: Vec::new(),
            operands: Vec::new(),
            attrs: Vec::new(),
            regions: Vec::new(),
        }
    }

    pub fn with_result(mut self, v: Ssa, ty: IrType) -> Op {
        self.results.push(v);
        self.result_types.push(ty);
        self
    }

    pub fn with_operands(mut self, ops: impl IntoIterator<Item = Ssa>) -> Op {
        self.operands.extend(ops);
        self
    }

    pub fn with_attr(mut self, k: AttrKey, v: Attr) -> Op {
        self.set_attr(k, v);
        self
    }

    pub fn with_region(mut self, r: Region) -> Op {
        self.regions.push(r);
        self
    }

    pub fn attr(&self, k: &AttrKey) -> Option<&Attr> {
        self.attrs.iter().find(|(key, _)| key == k).map(|(_, v)| v)
    }

    pub fn set_attr(&mut self, k: AttrKey, v: Attr) {
        match self.attrs.iter_mut().find(|(key, _)| *key == k) {
            Some(slot) => slot.1 = v,
            None => self.attrs.push((k, v)),
        }
    }

    pub fn remove_attr(&mut self, k: &AttrKey) {
        self.attrs.retain(|(key, _)| key != k);
    }

    pub fn result(&self) -> Option<Ssa> {
        self.results.first().copied()
    }

    pub fn result_type(&self) -> Option<&IrType> {
        self.result_types.first()
    }

    /// EVERY op inside this one, regions included, in program order.
    pub fn ops_deep(&self) -> Vec<&Op> {
        let mut out = Vec::new();
        fn walk<'o>(ops: &'o [Op], out: &mut Vec<&'o Op>) {
            for o in ops {
                out.push(o);
                for r in &o.regions {
                    walk(&r.ops, out);
                }
            }
        }
        for r in &self.regions {
            walk(&r.ops, &mut out);
        }
        out
    }
}

/// A top-level module: one function per entry, plus the SSA-name hints the text
/// instruments use.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Module {
    /// The module's own ops. Normally one [`OpKind::TtFunc`].
    pub ops: Vec<Op>,
    /// Module-level attributes, e.g. DotToLinalg's `spyre.canonical_verified`.
    pub attrs: Vec<(AttrKey, Attr)>,
    /// The next unused [`Ssa`] id. A pass that creates a value takes one from
    /// here, so two passes cannot mint the same name.
    pub next_ssa: u32,
    /// `Ssa -> the spelling it had in text`, PURELY a diagnostic.
    ///
    /// Out of band, and out of [`Op`], on purpose: a pass that could read a
    /// value's name could behave differently for `%acc` than for `%17`, and the
    /// C++ it is ported from cannot -- MLIR's own names are regenerated on every
    /// print. The printer uses these to make a golden diff legible; no pass may.
    pub hints: HashMap<Ssa, String>,
}

impl Module {
    pub fn new() -> Module {
        Module::default()
    }

    /// Mint a fresh SSA name.
    pub fn fresh(&mut self) -> Ssa {
        let v = Ssa(self.next_ssa);
        self.next_ssa += 1;
        v
    }

    /// A fresh name that carries `hint` as its diagnostic spelling. MLIR derives
    /// a rewritten value's printed name from the value it replaced, and matching
    /// that is what makes the golden diff readable.
    pub fn fresh_named(&mut self, hint: &str) -> Ssa {
        let v = self.fresh();
        self.hints.insert(v, hint.to_string());
        v
    }

    pub fn hint(&self, v: Ssa) -> String {
        match self.hints.get(&v) {
            Some(s) => s.clone(),
            None => format!("{}", v.0),
        }
    }

    /// The single kernel function, or an error saying what was found -- the same
    /// contract `triton-superdsc-lower`'s `Module::kernel` had when it read text.
    pub fn kernel(&self) -> crate::Result<&Op> {
        let funcs: Vec<&Op> = self
            .ops
            .iter()
            .filter(|o| o.kind == OpKind::TtFunc || o.kind == OpKind::FuncFunc)
            .collect();
        match funcs.len() {
            1 => Ok(funcs[0]),
            0 => Err(crate::Refusal::new("ktir", "no tt.func/func.func in this module")),
            n => Err(crate::Refusal::new(
                "ktir",
                format!(
                    "expected exactly one kernel function, found {n}: {}",
                    funcs
                        .iter()
                        .map(|o| o
                            .attr(&AttrKey::SymName)
                            .and_then(|a| a.as_str())
                            .unwrap_or("<unnamed>")
                            .to_string())
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
            )),
        }
    }

    pub fn kernel_mut(&mut self) -> crate::Result<&mut Op> {
        let n = self
            .ops
            .iter()
            .filter(|o| o.kind == OpKind::TtFunc || o.kind == OpKind::FuncFunc)
            .count();
        if n != 1 {
            // Re-use the immutable path's diagnostic so the two cannot drift.
            self.kernel()?;
        }
        Ok(self
            .ops
            .iter_mut()
            .find(|o| o.kind == OpKind::TtFunc || o.kind == OpKind::FuncFunc)
            .unwrap())
    }

    /// EVERY op in the module, regions included, in program order.
    pub fn ops_deep(&self) -> Vec<&Op> {
        let mut out = Vec::new();
        fn walk<'o>(ops: &'o [Op], out: &mut Vec<&'o Op>) {
            for o in ops {
                out.push(o);
                for r in &o.regions {
                    walk(&r.ops, out);
                }
            }
        }
        walk(&self.ops, &mut out);
        out
    }

    /// How many of each [`OpKind`] the module contains, regions included.
    ///
    /// THE CENSUS. A wrong count IS the bug, so the golden tests assert on this
    /// rather than on a hand-picked op.
    pub fn census(&self) -> Vec<(String, usize)> {
        let mut counts: HashMap<String, usize> = HashMap::new();
        for op in self.ops_deep() {
            *counts.entry(op.kind.spelling().to_string()).or_insert(0) += 1;
        }
        let mut v: Vec<(String, usize)> = counts.into_iter().collect();
        v.sort();
        v
    }

    /// The op that defines `v`, searched depth-first over the whole module.
    pub fn def_of(&self, v: Ssa) -> Option<&Op> {
        self.ops_deep().into_iter().find(|o| o.results.contains(&v))
    }

    /// `v`'s type, from its defining op or from a block argument list.
    pub fn type_of(&self, v: Ssa) -> Option<IrType> {
        for op in self.ops_deep() {
            if let Some(i) = op.results.iter().position(|r| *r == v) {
                return op.result_types.get(i).cloned();
            }
            for r in &op.regions {
                if let Some((_, t)) = r.args.iter().find(|(a, _)| *a == v) {
                    return Some(t.clone());
                }
            }
        }
        None
    }
}

//===----------------------------------------------------------------------===//
// f16 <-> f32, by bits
//===----------------------------------------------------------------------===//

/// f32 -> IEEE half, round-nearest-ties-to-even. This is `APFloat::convert(
/// APFloat::IEEEhalf(), APFloat::rmNearestTiesToEven)`, which is what
/// `LegalizeTypes::convertF32ValueAttr` calls, so the re-rounded constant a
/// ported `LegalizeTypes` produces is bit-identical to the C++'s.
pub fn f32_to_f16_bits(v: f32) -> u16 {
    let x = v.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let exp = ((x >> 23) & 0xff) as i32;
    let mant = x & 0x007f_ffff;

    if exp == 0xff {
        // Inf or NaN. A NaN keeps a non-zero payload so it stays a NaN.
        let m = (mant >> 13) as u16;
        return sign | 0x7c00 | if mant != 0 { m.max(1) } else { 0 };
    }

    // Unbiased exponent, re-biased for half (15).
    let he = exp - 127 + 15;
    if he >= 0x1f {
        return sign | 0x7c00; // overflow -> inf
    }
    if he <= 0 {
        // Subnormal half, or zero. Shift the implicit 1 back in and round.
        if he < -10 {
            return sign; // rounds to zero
        }
        let m = mant | 0x0080_0000;
        let shift = (14 - he) as u32;
        let mut half = (m >> shift) as u16;
        // Round to nearest, ties to even, from the bits shifted out.
        let rem = m & ((1u32 << shift) - 1);
        let halfway = 1u32 << (shift - 1);
        if rem > halfway || (rem == halfway && (half & 1) == 1) {
            half += 1;
        }
        return sign | half;
    }

    let mut h = ((he as u16) << 10) | ((mant >> 13) as u16);
    let rem = mant & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && (h & 1) == 1) {
        h += 1; // may carry into the exponent, which is correct
    }
    sign | h
}

/// IEEE half bits -> f32, exact (every half is representable).
pub fn f16_bits_to_f32(h: u16) -> f32 {
    let sign = ((h as u32) & 0x8000) << 16;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = ((h as u32) & 0x03ff) << 13;

    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign);
        }
        // Subnormal half -> normal f32: renormalize.
        let mut e = 0u32;
        let mut m = mant;
        while m & 0x0080_0000 == 0 {
            m <<= 1;
            e += 1;
        }
        m &= !0x0080_0000;
        return f32::from_bits(sign | ((127 - 15 - e + 1) << 23) | m);
    }
    if exp == 0x1f {
        return f32::from_bits(sign | 0x7f80_0000 | mant);
    }
    f32::from_bits(sign | ((exp + 127 - 15) << 23) | mant)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_round_trip_matches_mlir_printing_of_the_flash_scale() {
        // THE constant the attention fixture carries: sm_scale * 1.44269504 with
        // sm_scale = 1/sqrt(128). ttir has it as f32 `0.127517432`; the C++ KTIR
        // prints the f16 as `1.275630e-01`. If our re-round differs by one ULP the
        // golden diff fails on a constant, which is the bug this pins.
        let f32v = 0.127_517_43_f32;
        let h = FloatBits::f32(f32v).to_f16();
        assert_eq!(h.width, 16);
        // The nearest f16 is 0.1275634765625: exponent -3 (biased 12) with mantissa
        // 21, i.e. 0x3015. MLIR prints it `1.275630e-01`.
        assert_eq!(h.bits, 0x3015, "f16 bits of the flash qk_scale");
        let back = h.as_f64();
        assert!((back - 0.1275634765625).abs() < 1e-12, "got {back}");
    }

    #[test]
    fn f16_specials_survive() {
        // -inf is the flash kernel's m_i init, printed `0xFC00`.
        let ninf = FloatBits { bits: 0xFC00, width: 16 };
        assert!(ninf.as_f64().is_infinite() && ninf.as_f64() < 0.0);
        assert_eq!(FloatBits::f16_from_f32(1.0).bits, 0x3c00);
        assert_eq!(FloatBits::f16_from_f32(0.0).bits, 0x0000);
        assert!(FloatBits::f16_from_f32(0.0).is_zero());
        assert!(!FloatBits::f16_from_f32(1.0).is_zero());
    }

    #[test]
    fn ties_go_to_even() {
        // 1 + 2^-11 is exactly halfway between 1.0 (0x3c00) and the next half
        // (0x3c01); ties-to-even must pick 0x3c00.
        let halfway = 1.0_f32 + 2.0_f32.powi(-11);
        assert_eq!(f32_to_f16_bits(halfway), 0x3c00);
        // 1 + 3*2^-11 is halfway between 0x3c01 and 0x3c02 -> even is 0x3c02.
        let halfway2 = 1.0_f32 + 3.0 * 2.0_f32.powi(-11);
        assert_eq!(f32_to_f16_bits(halfway2), 0x3c02);
    }

    #[test]
    fn out_of_scope_f32_is_not_compute_f32() {
        // The distinction LegalizeTypes' island gate turns on: a memref<f32> must
        // never be retyped, or its bytes are reinterpreted as f16.
        let t = IrType::Tensor { dims: vec![64], elem: DType::F32 };
        assert!(t.is_compute_f32() && !t.is_out_of_scope_f32());
        let m = IrType::MemRef { dims: vec![64], elem: DType::F32 };
        assert!(!m.is_compute_f32() && m.is_out_of_scope_f32());
        let p = IrType::Ptr { elem: DType::F32 };
        assert!(!p.is_compute_f32() && p.is_out_of_scope_f32());
    }

    #[test]
    fn ktdp_load_is_not_pure_so_dce_leaves_it_where_the_cpp_does() {
        assert!(!OpKind::KtdpLoad.is_pure());
        assert!(OpKind::ArithConstant.is_pure());
    }
}
