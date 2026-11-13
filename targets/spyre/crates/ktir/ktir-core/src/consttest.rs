// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License").
//
//! A PROGRAM IS CONST DATA — the property `#[forward]` depends on.
//!
//! The baked bundle reaches the binary through `inventory::submit!`, which stores its value in a
//! `static` and therefore needs it const-constructible. This module builds a whole [`IRFunction`]
//! as one, so the property is checked by the compiler on every build rather than assumed until a
//! macro tries it.

use crate::attrkey::AttrKey;
use crate::dtypes::DType;
use crate::ir::{Attr, IRFunction, Operation, Ssa};
use crate::irtype::IrType;
use crate::opkind::OpKind;

const DIMS: &[i64] = &[64, 64];
const OPERANDS: &[Ssa] = &[Ssa(0)];
const ATTRS: &[(AttrKey, Attr<'static>)] = &[
    (AttrKey::Shape, Attr::IntList(DIMS)),
    (AttrKey::Dtype, Attr::Dtype(DType::F16)),
];
const OPS: &[Operation<'static>] = &[Operation {
    result: Some(Ssa(1)),
    op_type: OpKind::KtdpConstructMemoryView,
    operands: OPERANDS,
    attributes: ATTRS,
    result_type: Some(IrType::MemRef {
        dims: DIMS,
        elem: DType::F16,
    }),
    regions: &[],
}];
const ARGS: &[(Ssa, IrType<'static>)] = &[(Ssa(0), IrType::Index)];

/// A whole function, as a `static`.
pub static PROGRAM: IRFunction<'static> = IRFunction {
    name: "mm",
    arguments: ARGS,
    operations: OPS,
    grid: (32, 1, 1),
    return_type: None,
};
