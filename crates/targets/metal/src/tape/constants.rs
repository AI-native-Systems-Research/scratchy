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
    /// A `uint` the tape variant supplies ([`TapeVariant::cap`]): its KV cap rung, the block
    /// table's row stride.
    KvCap,
    /// A `uint` the tape variant supplies ([`TapeVariant::tq_heads`]): the query heads one
    /// TurboQuant decode threadgroup serves.
    TqHeads,
}

/// The values a tape variant binds ([`ConstantType::KvCap`], [`ConstantType::TqHeads`]). One tape
/// body serves every variant of its bucket; each variant's kernels are baked with its own values,
/// and the device picks the variant.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TapeVariant {
    pub cap: super::ids::MaxBlocksPerSeq,
    /// `None`: the tape has no TurboQuant decode attention.
    pub tq_heads: Option<super::ids::TqDecodeHeads>,
}

/// A constant bound to a value its tape variant does not carry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UnboundConstant {
    pub slot: ConstSlot,
    pub ty: ConstantType,
}

impl std::fmt::Display for UnboundConstant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let Self { slot, ty } = self;
        write!(
            f,
            "constant slot {} is {ty:?}, which its tape variant does not bind",
            slot.0
        )
    }
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

    /// The tape variant's KV cap ([`ConstantType::KvCap`]).
    pub fn kv_cap(index: impl Into<ConstSlot>) -> Self {
        Self {
            index: index.into().0,
            bits: 0,
            ty: ConstantType::KvCap,
        }
    }

    /// The tape variant's TurboQuant decode heads ([`ConstantType::TqHeads`]).
    pub fn tq_heads(index: impl Into<ConstSlot>) -> Self {
        Self {
            index: index.into().0,
            bits: 0,
            ty: ConstantType::TqHeads,
        }
    }

    /// `self` with a variant-bound value replaced by `variant`'s; every other constant as it is.
    pub fn resolve(self, variant: TapeVariant) -> Result<Self, UnboundConstant> {
        let unbound = UnboundConstant {
            slot: ConstSlot(self.index),
            ty: self.ty,
        };
        match self.ty {
            ConstantType::KvCap => Ok(Self::uint(self.index, variant.cap.get())),
            ConstantType::TqHeads => (variant.tq_heads)
                .map(|h| Self::uint(self.index, h.get()))
                .ok_or(unbound),
            ConstantType::UInt | ConstantType::Int | ConstantType::Float | ConstantType::Bool => {
                Ok(self)
            }
        }
    }
}
