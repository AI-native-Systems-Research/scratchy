// SPDX-License-Identifier: Apache-2.0
//! The device ops a lowered program is made of, and each op's ABI: the name dxp reads, the unit
//! that runs it, and how its operands must be sticked. This is a TABLE the lowering reads — the
//! only place a fact about one op lives.

/// The execution unit an op runs on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExUnit {
    /// The special-function pipeline: every pointwise, reduction and data-movement op.
    Sfp,
    /// The processing-element array: the contractions.
    Pt,
}

impl ExUnit {
    pub const fn wire(self) -> &'static str {
        match self {
            ExUnit::Sfp => "sfp",
            ExUnit::Pt => "pt",
        }
    }
}

/// How an op relates its operands to its iteration space.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Family {
    /// Every operand and the result index the whole space (an operand may broadcast).
    Pointwise,
    /// The input indexes the whole space; the result drops the stick dimension (one value per
    /// stick, `scale -2`).
    StickReduce,
    /// The input indexes the whole space; the result drops one non-stick dimension (`scale -1`).
    NonStickReduce,
    /// `out[mb, out] = Σ_in a[mb, in] · b[in, out]`.
    Contraction,
}

/// A comparison predicate (`arith.cmpf`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Compare {
    Gt,
    Ge,
    Lt,
    Le,
    Eq,
    Ne,
}

/// A reduction's combiner.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Combine {
    Sum,
    Max,
    Min,
}

/// One device op.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OpFunc {
    // ── binary pointwise
    Add,
    Sub,
    Mul,
    Div,
    Maximum,
    Minimum,
    Cmp(Compare),
    // ── ternary pointwise
    Select,
    // ── unary pointwise
    Neg,
    Abs,
    Exp,
    Log,
    Sqrt,
    Rsqrt,
    Reciprocal,
    Tanh,
    Sigmoid,
    Erf,
    Floor,
    /// A copy: moves elements between two views of the same format and stick (the gather's
    /// carrier too).
    Identity,
    /// A copy into a layout whose stick is a different axis.
    Restick,
    /// fp16 → fp32.
    Widen,
    /// fp32 → fp16.
    Narrow,
    /// fp8 → fp16.
    Fp8ToFp16,
    /// fp16 → fp8: a format conversion, the value already in fp8's range.
    Fp16ToFp8,
    // ── reductions
    Reduce(Combine, Axis),
    // ── contractions
    Matmul,
    MatmulFp8,
}

/// Which axis a reduction folds, relative to the stick.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Axis {
    Stick,
    NonStick,
}

impl OpFunc {
    /// Whether this op converts its operand's element format — its two sides hold different
    /// sticks, and dxp's conversion templates bind them by role.
    pub const fn converts(self) -> bool {
        matches!(self, OpFunc::Widen | OpFunc::Narrow | OpFunc::Fp8ToFp16 | OpFunc::Fp16ToFp8)
    }

    /// The `opFuncName` dxp reads.
    pub const fn wire(self) -> &'static str {
        match self {
            OpFunc::Add => "add",
            OpFunc::Sub => "sub",
            OpFunc::Mul => "mul",
            OpFunc::Div => "realdiv",
            OpFunc::Maximum => "maximum",
            OpFunc::Minimum => "minimum",
            OpFunc::Cmp(Compare::Gt) => "greaterthan",
            OpFunc::Cmp(Compare::Ge) => "greaterequal",
            OpFunc::Cmp(Compare::Lt) => "lesserthan",
            OpFunc::Cmp(Compare::Le) => "lesserequal",
            OpFunc::Cmp(Compare::Eq) => "equal",
            OpFunc::Cmp(Compare::Ne) => "notequal",
            OpFunc::Select => "where3",
            OpFunc::Neg => "neg",
            OpFunc::Abs => "abs",
            OpFunc::Exp => "exp",
            OpFunc::Log => "log",
            OpFunc::Sqrt => "sqrt",
            OpFunc::Rsqrt => "rsqrt",
            OpFunc::Reciprocal => "reciprocal",
            OpFunc::Tanh => "tanh",
            OpFunc::Sigmoid => "sigmoid",
            OpFunc::Erf => "erf",
            OpFunc::Floor => "floor",
            OpFunc::Identity => "identity",
            OpFunc::Restick => "ReStickifyOpHBM",
            OpFunc::Widen => "dl16tofp32",
            OpFunc::Narrow => "fp32todl16",
            OpFunc::Fp8ToFp16 => "fp8todl16",
            OpFunc::Fp16ToFp8 => "qfp8ch",
            OpFunc::Reduce(Combine::Sum, Axis::Stick) => "sum",
            OpFunc::Reduce(Combine::Max, Axis::Stick) => "max",
            OpFunc::Reduce(Combine::Min, Axis::Stick) => "min",
            OpFunc::Reduce(Combine::Sum, Axis::NonStick) => "sumnonstick",
            OpFunc::Reduce(Combine::Max, Axis::NonStick) => "maxnonstick",
            OpFunc::Reduce(Combine::Min, Axis::NonStick) => "minnonstick",
            OpFunc::Matmul => "batchmatmul",
            OpFunc::MatmulFp8 => "batchmatmulfp8",
        }
    }

    pub const fn unit(self) -> ExUnit {
        match self {
            OpFunc::Matmul | OpFunc::MatmulFp8 => ExUnit::Pt,
            _ => ExUnit::Sfp,
        }
    }

    pub const fn family(self) -> Family {
        match self {
            OpFunc::Reduce(_, Axis::Stick) => Family::StickReduce,
            OpFunc::Reduce(_, Axis::NonStick) => Family::NonStickReduce,
            OpFunc::Matmul | OpFunc::MatmulFp8 => Family::Contraction,
            _ => Family::Pointwise,
        }
    }

    /// A reduction reads its result as its first operand's accumulator — `inputLabeledDs` names
    /// the result too (the vendor's `sum`/`max` fixtures).
    pub const fn accumulates(self) -> bool {
        matches!(self, OpFunc::Reduce(..))
    }

    /// The value a reduction's result starts from.
    pub const fn identity(self) -> Option<f64> {
        match self {
            OpFunc::Reduce(Combine::Sum, _) => Some(0.0),
            OpFunc::Reduce(Combine::Max, _) => Some(f64::NEG_INFINITY),
            OpFunc::Reduce(Combine::Min, _) => Some(f64::INFINITY),
            _ => None,
        }
    }
}
