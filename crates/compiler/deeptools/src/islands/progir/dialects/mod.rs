//! THE PROGIR RUNG'S MLIR OPS — one dialect, three operations.
//!
//! ⭐⭐ THE SHORTEST `dialects/` IN THE LADDER, AND THAT IS THE POINT. Every rung above this one is a
//! module full of ops; here the module holds only [`init`]'s reference to a program that lives outside
//! MLIR entirely. See [`init`] for the measured twelve-line dump, and [`super::Program`] for where the
//! instructions actually are.

pub mod init;

/// AN SSA VALUE — ⭐ THE LADDER'S ONE NUMBERING, re-exported.
///
/// The `init.bin` operands are the symbol definitions the enclosing function computed, so they are
/// that function's values, not new ones.
pub use crate::islands::dataflow_ir::dialects::Val;

/// ONE PROGIR-RUNG MLIR OPERATION.
///
/// ⛔ NO `_` ARM WHERE THIS IS MATCHED, for the same reason as every other rung.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Op {
    /// `InitOps.td` — the packet and its references.
    Init(init::Op),
}
