// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License").
//
//! THE OP VOCABULARY, AS A TYPE.
//!
//! An operation names itself with an [`OpKind`], not a spelling. The dispatch tables, the
//! optimizer's pattern matches and the lowering all compare variants, so a typo is a compile
//! error and the set of ops the interpreter can execute is the set this enum declares.

/// Every operation this tree names.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum OpKind {
    /// `arith.absf`
    ArithAbsf,
    /// `arith.addf`
    ArithAddf,
    /// `arith.addi`
    ArithAddi,
    /// `arith.andi`
    ArithAndi,
    /// `arith.bitcast`
    ArithBitcast,
    /// `arith.ceildivsi`
    ArithCeildivsi,
    /// `arith.ceildivui`
    ArithCeildivui,
    /// `arith.cmpf`
    ArithCmpf,
    /// `arith.cmpi`
    ArithCmpi,
    /// `arith.constant`
    ArithConstant,
    /// `arith.convertf`
    ArithConvertf,
    /// `arith.divf`
    ArithDivf,
    /// `arith.divsi`
    ArithDivsi,
    /// `arith.divui`
    ArithDivui,
    /// `arith.extf`
    ArithExtf,
    /// `arith.extsi`
    ArithExtsi,
    /// `arith.extui`
    ArithExtui,
    /// `arith.floordivsi`
    ArithFloordivsi,
    /// `arith.fptosi`
    ArithFptosi,
    /// `arith.fptoui`
    ArithFptoui,
    /// `arith.index_cast`
    ArithIndexCast,
    /// `arith.index_castui`
    ArithIndexCastui,
    /// `arith.maxf`
    ArithMaxf,
    /// `arith.maximumf`
    ArithMaximumf,
    /// `arith.maxnumf`
    ArithMaxnumf,
    /// `arith.maxsi`
    ArithMaxsi,
    /// `arith.maxui`
    ArithMaxui,
    /// `arith.minf`
    ArithMinf,
    /// `arith.minimumf`
    ArithMinimumf,
    /// `arith.minnumf`
    ArithMinnumf,
    /// `arith.minsi`
    ArithMinsi,
    /// `arith.minui`
    ArithMinui,
    /// `arith.mulf`
    ArithMulf,
    /// `arith.muli`
    ArithMuli,
    /// `arith.negf`
    ArithNegf,
    /// `arith.ori`
    ArithOri,
    /// `arith.remf`
    ArithRemf,
    /// `arith.remsi`
    ArithRemsi,
    /// `arith.remui`
    ArithRemui,
    /// `arith.select`
    ArithSelect,
    /// `arith.shli`
    ArithShli,
    /// `arith.shrsi`
    ArithShrsi,
    /// `arith.shrui`
    ArithShrui,
    /// `arith.sitofp`
    ArithSitofp,
    /// `arith.subf`
    ArithSubf,
    /// `arith.subi`
    ArithSubi,
    /// `arith.truncf`
    ArithTruncf,
    /// `arith.trunci`
    ArithTrunci,
    /// `arith.uitofp`
    ArithUitofp,
    /// `arith.xori`
    ArithXori,
    /// `func.return`
    FuncReturn,
    /// `ktdp.allgather`
    KtdpAllgather,
    /// `ktdp.construct_access_tile`
    KtdpConstructAccessTile,
    /// `ktdp.construct_distributed_memory_view`
    KtdpConstructDistributedMemoryView,
    /// `ktdp.construct_indirect_access_tile`
    KtdpConstructIndirectAccessTile,
    /// `ktdp.construct_memory_view`
    KtdpConstructMemoryView,
    /// `ktdp.coreid`
    KtdpCoreid,
    /// `ktdp.get_compute_tile_id`
    KtdpGetComputeTileId,
    /// `ktdp.inter_tile_produce`
    KtdpInterTileProduce,
    /// `ktdp.inter_tile_reduce`
    KtdpInterTileReduce,
    /// `ktdp.load`
    KtdpLoad,
    /// `ktdp.not_yet`
    KtdpNotYet,
    /// `ktdp.reduce`
    KtdpReduce,
    /// `ktdp.store`
    KtdpStore,
    /// `ktdp.yield_partial`
    KtdpYieldPartial,
    /// `ktdp.yield_reduced`
    KtdpYieldReduced,
    /// `linalg.add`
    LinalgAdd,
    /// `linalg.batch_matmul`
    LinalgBatchMatmul,
    /// `linalg.broadcast`
    LinalgBroadcast,
    /// `linalg.div`
    LinalgDiv,
    /// `linalg.fill`
    LinalgFill,
    /// `linalg.generic`
    LinalgGeneric,
    /// `linalg.index`
    LinalgIndex,
    /// `linalg.matmul`
    LinalgMatmul,
    /// `linalg.max`
    LinalgMax,
    /// `linalg.min`
    LinalgMin,
    /// `linalg.mul`
    LinalgMul,
    /// `linalg.reduce`
    LinalgReduce,
    /// `linalg.sub`
    LinalgSub,
    /// `linalg.transpose`
    LinalgTranspose,
    /// `linalg.yield`
    LinalgYield,
    /// `math.absf`
    MathAbsf,
    /// `math.absi`
    MathAbsi,
    /// `math.ceil`
    MathCeil,
    /// `math.cos`
    MathCos,
    /// `math.erf`
    MathErf,
    /// `math.exp`
    MathExp,
    /// `math.floor`
    MathFloor,
    /// `math.fma`
    MathFma,
    /// `math.log`
    MathLog,
    /// `math.log1p`
    MathLog1p,
    /// `math.log2`
    MathLog2,
    /// `math.powf`
    MathPowf,
    /// `math.rsqrt`
    MathRsqrt,
    /// `math.sin`
    MathSin,
    /// `math.sqrt`
    MathSqrt,
    /// `math.tanh`
    MathTanh,
    /// `region.bb0_args`
    RegionBb0Args,
    /// `scf.for`
    ScfFor,
    /// `scf.forall`
    ScfForall,
    /// `scf.if`
    ScfIf,
    /// `scf.parallel`
    ScfParallel,
    /// `scf.while`
    ScfWhile,
    /// `scf.yield`
    ScfYield,
    /// `tensor.collapse_shape`
    TensorCollapseShape,
    /// `tensor.empty`
    TensorEmpty,
    /// `tensor.expand_shape`
    TensorExpandShape,
    /// `tensor.extract`
    TensorExtract,
    /// `tensor.extract_slice`
    TensorExtractSlice,
    /// `tensor.from_elements`
    TensorFromElements,
    /// `tensor.generate`
    TensorGenerate,
    /// `tensor.insert_slice`
    TensorInsertSlice,
    /// `tensor.reshape`
    TensorReshape,
    /// `tensor.splat`
    TensorSplat,
    /// `tensor.yield`
    TensorYield,
}

/// What an operation states about its result's frame — the law the frame solve
/// applies to the op, stated once here and never re-derived by a reader. The
/// solve (one pass over the program) turns each law into the equalities and
/// anchors it names; the classification is a property of the op, like
/// [`OpKind::dialect`], so a variant that forgets to state its law is a
/// compile error, never a silent default at a reader.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FrameLaw {
    /// Reads its operands lane for lane: the result holds each operand's own
    /// lanes at the result's count, so the solve joins it with every
    /// same-shaped operand. A scalar operand names no frame; an operand
    /// holding one value per result row is deferred to the solved frame.
    Elementwise,
    /// Moves no element: the result is its operand's value, restated at
    /// another shape — the solve joins it with its operand.
    MovesNoElement,
    /// Writes a dense tile of rows — the solve anchors it to the flat frame.
    DenseTile,
    /// The axis it folds decides the frame it writes: a fold of the last axis
    /// writes one value per row; a fold of any other axis writes a tile of
    /// its operand's rows.
    Reduction,
    /// A spray: addressing, not data — the solve anchors its source by the
    /// axis it sprays and holds no frame for the result.
    Spray,
    /// A load: the frame its own tile, shape, or store states.
    Load,
    /// States no equality and no absolute of its own — the solve joins it by
    /// its consumers or resolves it through the one default.
    Unstated,
}

/// The dialect an [`OpKind`] belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Dialect {
    /// `arith.*`
    Arith,
    /// `func.*`
    Func,
    /// `ktdp.*`
    Ktdp,
    /// `linalg.*`
    Linalg,
    /// `math.*`
    Math,
    /// `region.*`
    Region,
    /// `scf.*`
    Scf,
    /// `tensor.*`
    Tensor,
}

impl OpKind {
    /// Which dialect declares this operation.
    pub fn dialect(self) -> Dialect {
        match self {
            OpKind::ArithAbsf => Dialect::Arith,
            OpKind::ArithAddf => Dialect::Arith,
            OpKind::ArithAddi => Dialect::Arith,
            OpKind::ArithAndi => Dialect::Arith,
            OpKind::ArithBitcast => Dialect::Arith,
            OpKind::ArithCeildivsi => Dialect::Arith,
            OpKind::ArithCeildivui => Dialect::Arith,
            OpKind::ArithCmpf => Dialect::Arith,
            OpKind::ArithCmpi => Dialect::Arith,
            OpKind::ArithConstant => Dialect::Arith,
            OpKind::ArithConvertf => Dialect::Arith,
            OpKind::ArithDivf => Dialect::Arith,
            OpKind::ArithDivsi => Dialect::Arith,
            OpKind::ArithDivui => Dialect::Arith,
            OpKind::ArithExtf => Dialect::Arith,
            OpKind::ArithExtsi => Dialect::Arith,
            OpKind::ArithExtui => Dialect::Arith,
            OpKind::ArithFloordivsi => Dialect::Arith,
            OpKind::ArithFptosi => Dialect::Arith,
            OpKind::ArithFptoui => Dialect::Arith,
            OpKind::ArithIndexCast => Dialect::Arith,
            OpKind::ArithIndexCastui => Dialect::Arith,
            OpKind::ArithMaxf => Dialect::Arith,
            OpKind::ArithMaximumf => Dialect::Arith,
            OpKind::ArithMaxnumf => Dialect::Arith,
            OpKind::ArithMaxsi => Dialect::Arith,
            OpKind::ArithMaxui => Dialect::Arith,
            OpKind::ArithMinf => Dialect::Arith,
            OpKind::ArithMinimumf => Dialect::Arith,
            OpKind::ArithMinnumf => Dialect::Arith,
            OpKind::ArithMinsi => Dialect::Arith,
            OpKind::ArithMinui => Dialect::Arith,
            OpKind::ArithMulf => Dialect::Arith,
            OpKind::ArithMuli => Dialect::Arith,
            OpKind::ArithNegf => Dialect::Arith,
            OpKind::ArithOri => Dialect::Arith,
            OpKind::ArithRemf => Dialect::Arith,
            OpKind::ArithRemsi => Dialect::Arith,
            OpKind::ArithRemui => Dialect::Arith,
            OpKind::ArithSelect => Dialect::Arith,
            OpKind::ArithShli => Dialect::Arith,
            OpKind::ArithShrsi => Dialect::Arith,
            OpKind::ArithShrui => Dialect::Arith,
            OpKind::ArithSitofp => Dialect::Arith,
            OpKind::ArithSubf => Dialect::Arith,
            OpKind::ArithSubi => Dialect::Arith,
            OpKind::ArithTruncf => Dialect::Arith,
            OpKind::ArithTrunci => Dialect::Arith,
            OpKind::ArithUitofp => Dialect::Arith,
            OpKind::ArithXori => Dialect::Arith,
            OpKind::FuncReturn => Dialect::Func,
            OpKind::KtdpAllgather => Dialect::Ktdp,
            OpKind::KtdpConstructAccessTile => Dialect::Ktdp,
            OpKind::KtdpConstructDistributedMemoryView => Dialect::Ktdp,
            OpKind::KtdpConstructIndirectAccessTile => Dialect::Ktdp,
            OpKind::KtdpConstructMemoryView => Dialect::Ktdp,
            OpKind::KtdpCoreid => Dialect::Ktdp,
            OpKind::KtdpGetComputeTileId => Dialect::Ktdp,
            OpKind::KtdpInterTileProduce => Dialect::Ktdp,
            OpKind::KtdpInterTileReduce => Dialect::Ktdp,
            OpKind::KtdpLoad => Dialect::Ktdp,
            OpKind::KtdpNotYet => Dialect::Ktdp,
            OpKind::KtdpReduce => Dialect::Ktdp,
            OpKind::KtdpStore => Dialect::Ktdp,
            OpKind::KtdpYieldPartial => Dialect::Ktdp,
            OpKind::KtdpYieldReduced => Dialect::Ktdp,
            OpKind::LinalgAdd => Dialect::Linalg,
            OpKind::LinalgBatchMatmul => Dialect::Linalg,
            OpKind::LinalgBroadcast => Dialect::Linalg,
            OpKind::LinalgDiv => Dialect::Linalg,
            OpKind::LinalgFill => Dialect::Linalg,
            OpKind::LinalgGeneric => Dialect::Linalg,
            OpKind::LinalgIndex => Dialect::Linalg,
            OpKind::LinalgMatmul => Dialect::Linalg,
            OpKind::LinalgMax => Dialect::Linalg,
            OpKind::LinalgMin => Dialect::Linalg,
            OpKind::LinalgMul => Dialect::Linalg,
            OpKind::LinalgReduce => Dialect::Linalg,
            OpKind::LinalgSub => Dialect::Linalg,
            OpKind::LinalgTranspose => Dialect::Linalg,
            OpKind::LinalgYield => Dialect::Linalg,
            OpKind::MathAbsf => Dialect::Math,
            OpKind::MathAbsi => Dialect::Math,
            OpKind::MathCeil => Dialect::Math,
            OpKind::MathCos => Dialect::Math,
            OpKind::MathErf => Dialect::Math,
            OpKind::MathExp => Dialect::Math,
            OpKind::MathFloor => Dialect::Math,
            OpKind::MathFma => Dialect::Math,
            OpKind::MathLog => Dialect::Math,
            OpKind::MathLog1p => Dialect::Math,
            OpKind::MathLog2 => Dialect::Math,
            OpKind::MathPowf => Dialect::Math,
            OpKind::MathRsqrt => Dialect::Math,
            OpKind::MathSin => Dialect::Math,
            OpKind::MathSqrt => Dialect::Math,
            OpKind::MathTanh => Dialect::Math,
            OpKind::RegionBb0Args => Dialect::Region,
            OpKind::ScfFor => Dialect::Scf,
            OpKind::ScfForall => Dialect::Scf,
            OpKind::ScfIf => Dialect::Scf,
            OpKind::ScfParallel => Dialect::Scf,
            OpKind::ScfWhile => Dialect::Scf,
            OpKind::ScfYield => Dialect::Scf,
            OpKind::TensorCollapseShape => Dialect::Tensor,
            OpKind::TensorEmpty => Dialect::Tensor,
            OpKind::TensorExpandShape => Dialect::Tensor,
            OpKind::TensorExtract => Dialect::Tensor,
            OpKind::TensorExtractSlice => Dialect::Tensor,
            OpKind::TensorFromElements => Dialect::Tensor,
            OpKind::TensorGenerate => Dialect::Tensor,
            OpKind::TensorInsertSlice => Dialect::Tensor,
            OpKind::TensorReshape => Dialect::Tensor,
            OpKind::TensorSplat => Dialect::Tensor,
            OpKind::TensorYield => Dialect::Tensor,
        }
    }

    /// The frame law this operation states — exhaustive over the vocabulary,
    /// no default arm: a new variant that states no law is a compile error
    /// here, never a silent resolution at a reader.
    pub const fn frame_law(self) -> FrameLaw {
        match self {
            // ── THE ELEMENTWISE FAMILY: reads its operands lane for lane — every arith
            // and math pointwise op, float AND integer, the comparisons and the casts,
            // states the one law. ──
            OpKind::ArithAbsf
            | OpKind::ArithAddf
            | OpKind::ArithAddi
            | OpKind::ArithAndi
            | OpKind::ArithBitcast
            | OpKind::ArithCeildivsi
            | OpKind::ArithCeildivui
            | OpKind::ArithCmpf
            | OpKind::ArithCmpi
            | OpKind::ArithConvertf
            | OpKind::ArithDivf
            | OpKind::ArithDivsi
            | OpKind::ArithDivui
            | OpKind::ArithExtf
            | OpKind::ArithExtsi
            | OpKind::ArithExtui
            | OpKind::ArithFloordivsi
            | OpKind::ArithFptosi
            | OpKind::ArithFptoui
            | OpKind::ArithIndexCast
            | OpKind::ArithIndexCastui
            | OpKind::ArithMaxf
            | OpKind::ArithMaximumf
            | OpKind::ArithMaxnumf
            | OpKind::ArithMaxsi
            | OpKind::ArithMaxui
            | OpKind::ArithMinf
            | OpKind::ArithMinimumf
            | OpKind::ArithMinnumf
            | OpKind::ArithMinsi
            | OpKind::ArithMinui
            | OpKind::ArithMulf
            | OpKind::ArithMuli
            | OpKind::ArithNegf
            | OpKind::ArithOri
            | OpKind::ArithRemf
            | OpKind::ArithRemsi
            | OpKind::ArithRemui
            | OpKind::ArithSelect
            | OpKind::ArithShli
            | OpKind::ArithShrsi
            | OpKind::ArithShrui
            | OpKind::ArithSitofp
            | OpKind::ArithSubf
            | OpKind::ArithSubi
            | OpKind::ArithTruncf
            | OpKind::ArithTrunci
            | OpKind::ArithUitofp
            | OpKind::ArithXori
            | OpKind::MathAbsf
            | OpKind::MathAbsi
            | OpKind::MathCeil
            | OpKind::MathCos
            | OpKind::MathErf
            | OpKind::MathExp
            | OpKind::MathFloor
            | OpKind::MathFma
            | OpKind::MathLog
            | OpKind::MathLog1p
            | OpKind::MathLog2
            | OpKind::MathPowf
            | OpKind::MathRsqrt
            | OpKind::MathSin
            | OpKind::MathSqrt
            | OpKind::MathTanh
            | OpKind::LinalgAdd
            | OpKind::LinalgDiv
            | OpKind::LinalgMax
            | OpKind::LinalgMin
            | OpKind::LinalgMul
            | OpKind::LinalgSub => FrameLaw::Elementwise,
            // ── THE SHAPE OPS: no element moves. ──
            OpKind::TensorReshape | OpKind::TensorExpandShape | OpKind::TensorCollapseShape => {
                FrameLaw::MovesNoElement
            }
            // ── CONTRACTIONS AND THE TRANSPOSE: dense outputs. ──
            OpKind::LinalgMatmul | OpKind::LinalgBatchMatmul | OpKind::LinalgTranspose => {
                FrameLaw::DenseTile
            }
            // ── THE REDUCTION: the axis it folds decides the world it writes. ──
            OpKind::LinalgReduce => FrameLaw::Reduction,
            // ── THE SPRAY: a broadcast is addressing, not data. ──
            OpKind::LinalgBroadcast => FrameLaw::Spray,
            // ── THE LOAD: the world its tile, shape, or store states. ──
            OpKind::KtdpLoad => FrameLaw::Load,
            // ── EVERYTHING ELSE states no equality and no absolute of its own: a
            // constant, a splat, a slice, a region arg, a collective, a generated
            // tile, plumbing (tiles, views, stores, yields, the loop's own
            // machinery) — joined by consumers or resolved by the one default. ──
            OpKind::ArithConstant
            | OpKind::FuncReturn
            | OpKind::KtdpAllgather
            | OpKind::KtdpConstructAccessTile
            | OpKind::KtdpConstructDistributedMemoryView
            | OpKind::KtdpConstructIndirectAccessTile
            | OpKind::KtdpConstructMemoryView
            | OpKind::KtdpCoreid
            | OpKind::KtdpGetComputeTileId
            | OpKind::KtdpInterTileProduce
            | OpKind::KtdpInterTileReduce
            | OpKind::KtdpNotYet
            | OpKind::KtdpReduce
            | OpKind::KtdpStore
            | OpKind::KtdpYieldPartial
            | OpKind::KtdpYieldReduced
            | OpKind::LinalgFill
            | OpKind::LinalgGeneric
            | OpKind::LinalgIndex
            | OpKind::LinalgYield
            | OpKind::RegionBb0Args
            | OpKind::ScfFor
            | OpKind::ScfForall
            | OpKind::ScfIf
            | OpKind::ScfParallel
            | OpKind::ScfWhile
            | OpKind::ScfYield
            | OpKind::TensorEmpty
            | OpKind::TensorExtract
            | OpKind::TensorExtractSlice
            | OpKind::TensorFromElements
            | OpKind::TensorGenerate
            | OpKind::TensorInsertSlice
            | OpKind::TensorSplat
            | OpKind::TensorYield => FrameLaw::Unstated,
        }
    }
}
