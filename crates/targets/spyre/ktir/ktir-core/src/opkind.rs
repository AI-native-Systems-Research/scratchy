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
}
