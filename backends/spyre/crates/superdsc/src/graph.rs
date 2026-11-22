// SPDX-License-Identifier: Apache-2.0
//! A KTIR function as the device runs it: a sequence of device ops over strided views of three
//! kinds of storage.

use crate::format::DataFormat;
use crate::opfunc::OpFunc;
use crate::view::View;

/// Where an operand's elements live.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Store {
    /// The buffer the function's `i`-th parameter addresses.
    Param(usize),
    /// An intermediate the lowering owns.
    Local(usize),
    /// A constant stick: every lane of one stick holds the value.
    Const(usize),
}

/// One operand of a device op: a view over a store, with the view's axes aligned to the op's
/// space (see [`Node`]).
#[derive(Clone, Debug, PartialEq)]
pub struct Operand {
    pub store: Store,
    pub view: View,
    pub format: DataFormat,
    /// `Some` for an operand read through an index tensor: the gather.
    pub gather: Option<Gather>,
}

/// An access through an index tensor: along `axis` of the operand's view, position `p` is the row
/// `index[p]` names rather than row `p` (a row is one step of the axis's stride).
#[derive(Clone, Debug, PartialEq)]
pub struct Gather {
    pub axis: usize,
    /// The index tensor: a 1-D view, one entry per position of `axis`.
    pub index: Box<Operand>,
    /// Rows the indexed buffer holds along `axis`.
    pub rows: u64,
}

/// One device op.
///
/// `space` is the op's iteration space, outermost axis first. For a [`Family::Pointwise`] or a
/// reduction op every operand's view has one axis per space axis, in space order (a broadcast
/// operand's axis has stride 0, and a reduction's result has stride 0 along the folded axis). A
/// contraction's space is `(batch…, m, n, k)` and its operands are `a: (batch…, m, k)`,
/// `b: (batch…, k, n)`, `out: (batch…, m, n)`.
///
/// [`Family::Pointwise`]: crate::opfunc::Family::Pointwise
#[derive(Clone, Debug, PartialEq)]
pub struct Node {
    pub func: OpFunc,
    pub space: Vec<u64>,
    pub inputs: Vec<Operand>,
    pub output: Operand,
}

/// A constant stick's value: one lane's bits, repeated across the stick.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ConstStick {
    pub format: DataFormat,
    /// The value's bits in `format` (an fp16 value is its 16-bit pattern).
    pub bits: u64,
}

/// An intermediate's storage: `elements` of `format`, contiguous.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Local {
    pub elements: u64,
    pub format: DataFormat,
}

/// A lowered function.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Graph {
    /// The format each parameter's buffer is read and written in, by parameter.
    pub params: Vec<Option<DataFormat>>,
    /// Every memory view the function constructs over each parameter, by parameter.
    pub param_views: Vec<Vec<View>>,
    pub locals: Vec<Local>,
    pub consts: Vec<ConstStick>,
    pub nodes: Vec<Node>,
}
