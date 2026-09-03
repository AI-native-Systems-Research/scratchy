// SPDX-License-Identifier: Apache-2.0
//! Function-constant value/slot types for the metal tape — host-buildable data
//! (the objc `MTLDataType` mapping stays in `scratchy-target-metal`).

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize)]
pub enum ConstantType {
    UInt,
    /// Signed 32-bit. Required for MLX-port kernels whose function
    /// constants are declared `constant int` — Metal validates the
    /// type byte-for-byte against the constant declaration, so a
    /// `uint` payload bound to an `int` slot fails pipeline build
    /// with `MTLLibraryErrorDomain` 3 ("Constant X is of type
    /// MTLDataTypeInt but value found has type MTLDataTypeUInt").
    Int,
    Float,
    /// MTLDataType::Bool. The bit payload is a single byte (0/1) but
    /// Metal validates the declared type per-constant — passing a UInt
    /// to a `constant bool [[function_constant(N)]]` slot fails with
    /// `MTLLibraryErrorDomain` 3. Required for MLX-port kernels whose
    /// function constants are `align_Q` / `align_K` / `has_mask` /
    /// `do_causal` / `has_sinks` (the steel_attention family).
    Bool,
}

/// `[[function_constant(N)]]` slot index newtype.
///
/// Distinct from a raw `u16` so that a `ConstantValue::uint(slot,
/// value)` call can't have its two arguments swapped — the value
/// (`u32`) doesn't satisfy `Into<ConstSlot>`.
///
/// `From<u16>` is provided so existing literal call sites
/// `ConstantValue::uint(0u16, …)` keep working unchanged; Phase 2's
/// per-kernel constants struct migrates the literals to typed slot
/// references.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize)]
pub struct ConstSlot(pub u16);

impl ConstSlot {
    pub fn get(self) -> u16 {
        self.0
    }
}

impl From<u16> for ConstSlot {
    fn from(v: u16) -> Self {
        Self(v)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize)]
pub struct ConstantValue {
    pub index: u16,
    pub bits: u32,
    pub ty: ConstantType,
}

impl ConstantValue {
    pub fn uint(index: impl Into<ConstSlot>, value: u32) -> Self {
        Self {
            index: index.into().0,
            bits: value,
            ty: ConstantType::UInt,
        }
    }

    /// Signed 32-bit constant. Use for kernels whose function
    /// constants are declared `constant int` (the qmv / qmm_t MLX
    /// ports — see `quantized_qmv.metal::IN_VEC_SIZE` /
    /// `quantized_qmm.metal::QMM_K`).
    pub fn int(index: impl Into<ConstSlot>, value: i32) -> Self {
        Self {
            index: index.into().0,
            bits: value as u32,
            ty: ConstantType::Int,
        }
    }

    pub fn float(index: impl Into<ConstSlot>, value: f32) -> Self {
        Self {
            index: index.into().0,
            bits: value.to_bits(),
            ty: ConstantType::Float,
        }
    }

    pub fn boolean(index: impl Into<ConstSlot>, value: bool) -> Self {
        Self {
            index: index.into().0,
            bits: u32::from(value),
            ty: ConstantType::Bool,
        }
    }
}
