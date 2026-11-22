// SPDX-License-Identifier: Apache-2.0
//! The DDL's dimension labels — the names an SDSC iteration space and every tensor layout use.

/// One DDL dimension label.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Dim {
    /// The batch / row dimension.
    Mb,
    /// Outer spatial dimensions, in the order a pointwise space takes them.
    X,
    Y,
    I,
    J,
    /// The output channel — the stick dimension of a pointwise space and of a matmul's output.
    Out,
    /// A matmul's contraction dimension.
    In,
}

impl Dim {
    /// The label dxp reads.
    pub const fn wire(self) -> &'static str {
        match self {
            Dim::Mb => "mb",
            Dim::X => "x",
            Dim::Y => "y",
            Dim::I => "i",
            Dim::J => "j",
            Dim::Out => "out",
            Dim::In => "in",
        }
    }

    /// The `N_` / `ss_` key of this dimension (`"mb_"`).
    pub const fn extent_key(self) -> &'static str {
        match self {
            Dim::Mb => "mb_",
            Dim::X => "x_",
            Dim::Y => "y_",
            Dim::I => "i_",
            Dim::J => "j_",
            Dim::Out => "out_",
            Dim::In => "in_",
        }
    }
}

/// The labels a space's non-stick axes take, outermost first: a space of rank `r` labels its
/// leading `r - 1` axes from this list and its last (stick) axis [`Dim::Out`] — torch-spyre's
/// `INPUT_DIM_LABELS[:r-1] + ["out"]`.
pub const OUTER: [Dim; 5] = [Dim::Mb, Dim::X, Dim::Y, Dim::I, Dim::J];

/// A matmul's batch labels, innermost batch first: `(…, x, mb, out, in)` is torch-spyre's
/// `MATMUL_DIM_LABELS[-ndim:]` for a `(batch…, m, n, k)` space.
pub const MATMUL_BATCH: [Dim; 3] = [Dim::X, Dim::Y, Dim::I];
