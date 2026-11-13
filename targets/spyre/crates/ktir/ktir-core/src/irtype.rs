// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License").
//
//! A VALUE'S TYPE, AS A TYPE.
//!
//! An operation's result type used to be a string like `"tensor<32x64xf16>"`, so every pass that
//! needed a dim took the string apart and printed it back — a parse and a render in the middle of a
//! compiler that already knew the shape. [`IrType`] states it directly: the dims are a slice you
//! read, and reshaping is editing a field.

use crate::dtypes::DType;

/// The type of an SSA value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum IrType<'a> {
    /// `tensor<D0xD1x...xE>` — a value-semantics tile.
    Tensor { dims: &'a [i64], elem: DType },
    /// `memref<D0xD1x...xE>` — a memory view.
    MemRef { dims: &'a [i64], elem: DType },
    /// `!ktdp.access_tile<D0xD1x...xindex>` — a window over a view.
    AccessTile { dims: &'a [i64] },
    /// `index` — an address or a loop bound.
    Index,
    /// A bare scalar, e.g. `f16`.
    Scalar(DType),
}

impl<'a> IrType<'a> {
    /// The shaped dims, or `None` for an unshaped type.
    pub fn dims(self) -> Option<&'a [i64]> {
        match self {
            IrType::Tensor { dims, .. } | IrType::MemRef { dims, .. } => Some(dims),
            IrType::AccessTile { dims } => Some(dims),
            IrType::Index | IrType::Scalar(_) => None,
        }
    }

    /// The element type, where the type has one. An access tile INDEXES rather than holds, so it
    /// carries no element type of its own.
    pub fn elem(self) -> Option<DType> {
        match self {
            IrType::Tensor { elem, .. } | IrType::MemRef { elem, .. } => Some(elem),
            IrType::Scalar(e) => Some(e),
            IrType::AccessTile { .. } | IrType::Index => None,
        }
    }

    /// The same type over `dims` — how a pass that reshapes states its result.
    pub fn with_dims(self, dims: &'a [i64]) -> IrType<'a> {
        match self {
            IrType::Tensor { elem, .. } => IrType::Tensor { dims, elem },
            IrType::MemRef { elem, .. } => IrType::MemRef { dims, elem },
            IrType::AccessTile { .. } => IrType::AccessTile { dims },
            other => other,
        }
    }
}
