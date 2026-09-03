// SPDX-License-Identifier: Apache-2.0
//! Spyre's per-op ABI as DECLARED DATA — the counterpart to metal's
//! `metal_op_abi`.
//!
//! Adding one throwaway `SubOp` and reading the compiler's own errors
//! measured what a new `SubOp` costs:
//! five compiler-forced sites, three of them in spyre, with NOTHING
//! absorbing any of them. Metal's equivalent number is three, one of
//! which is the ABI table that takes its class's bridge arm. This
//! module is that table for spyre's tile-op bridge.
//!
//! Most ops the bridge translates have one shape: a pointwise/reduce
//! tile over `[mb, out_active, y]`, differing only in how many
//! operands the DDL call takes (every declared input, plus the
//! output). That is data. Three ops are not:
//!
//! * `MatmulTile` — a different `TileOpKind` and its own dims, since
//!   the K-chunk's output extent comes from the A-slice.
//! * `ScalarMul` — DEVICE width rather than logical `cols`, so a
//!   scalar-mul on padded logits addresses the same layout its
//!   producer matmul emitted.
//! * `RmsNormReduce` — a reduction whose output extent is the accum,
//!   not the active row.
//!
//! Those keep their arms. The table does not pretend to describe
//! them, exactly as metal's declines the weight-binding ops.

use scratchy_subtile::subtile_ir::{EwKind, RopeForm, SubOp};

/// How many operands the DDL call takes: `n_operands` counts every
/// declared input PLUS the output (the `pointwise_lx_resident`
/// convention).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Operands {
    /// Derived from the node's own input count.
    FromInputs,
    /// Fixed by the op's shape regardless of how the tape wired it.
    Fixed(u32),
}

/// The ops whose tile-op is the standard `[mb, out_active, y]`
/// pointwise/reduce shape. Used in PATTERN position by the bridge so
/// its match stays exhaustive — an `if let` guard would compile but
/// leave every arm in place, absorbing nothing (learned on the metal
/// side, `714860e08`).
macro_rules! spyre_standard_tile_pat {
    () => {
        SubOp::SumReduce
            | SubOp::Elementwise(_)
            | SubOp::SiluMul
            | SubOp::RmsNormApply { .. }
            | SubOp::RmsNorm { .. }
            | SubOp::RopeRotate { .. }
            | SubOp::RopeAppend { .. }
            | SubOp::AttnDecode { .. }
    };
}
pub(crate) use spyre_standard_tile_pat;

/// The operand rule for a member of that class.
pub(crate) fn spyre_standard_operands<F: RopeForm>(op: &SubOp<F>) -> Option<Operands> {
    Some(match op {
        SubOp::SumReduce => Operands::FromInputs,
        // Silu/Gelu are 1 input + output; Mul/Add are 2 + output.
        //
        // ⛔ GELU IS LISTED EXPLICITLY BECAUSE THE ARM BELOW IS A WILDCARD.
        // `_ => return None` means "not a standard tile op", so a unary that
        // is merely FORGOTTEN here does not fail to compile — it silently
        // leaves the class and loses its operand rule. Adding an `EwKind`
        // must therefore touch this table, and the compiler cannot say so.
        SubOp::Elementwise(EwKind::Silu | EwKind::Gelu | EwKind::QuickGelu | EwKind::GeluErf) => {
            Operands::Fixed(2)
        }
        SubOp::Elementwise(EwKind::Mul | EwKind::Add | EwKind::Sub) => Operands::Fixed(3),
        SubOp::SiluMul => Operands::Fixed(3),
        // data + gain + accum + output.
        SubOp::RmsNormApply { .. } | SubOp::RmsNorm { .. } => Operands::Fixed(4),
        SubOp::RopeRotate { .. } | SubOp::RopeAppend { .. } => Operands::FromInputs,
        SubOp::AttnDecode { .. } => Operands::FromInputs,
        _ => return None,
    })
}
