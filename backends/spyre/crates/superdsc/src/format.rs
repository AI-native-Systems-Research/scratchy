// SPDX-License-Identifier: Apache-2.0
//! What one lane of a stick holds.

use ktir_core::ddl_binds::DeviceGen;
use ktir_core::dtypes::DType;

use crate::qty::{Bytes, Lanes};

/// A device element format.
///
/// The device stores every format in 128-byte sticks; a format's lane count is the stick's bytes
/// over the element's ([`Self::lanes`]), the vendor's own stick law (`getNumElemsInStick`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize)]
pub enum DataFormat {
    /// The device's 16-bit float (DLF16, `SEN169_FP16`): what an `f16` KTIR value is on the card.
    Fp16,
    /// IEEE single precision.
    Fp32,
    /// 8-bit E4M3 float (`SEN143_FP8`).
    Fp8,
    /// A 32-bit unsigned word — every index tensor (page tables, row indices), and the integer
    /// ops' operands.
    UInt32,
    /// A 32-bit signed integer (`IEEE_INT32`): an integer cast's integer side.
    Int32,
    /// A 64-bit integer.
    Int64,
}

impl DataFormat {
    /// The format a KTIR element type is held in on the device. A predicate (`i1`) is a 0/1 value
    /// in the compute format, which is how every compare op writes it.
    pub const fn of(d: DType) -> DataFormat {
        match d {
            DType::F16 | DType::Bool => DataFormat::Fp16,
            DType::F32 => DataFormat::Fp32,
            DType::Fp8E4m3 => DataFormat::Fp8,
            DType::I32 => DataFormat::UInt32,
            DType::I64 => DataFormat::Int64,
        }
    }

    /// Whether two formats hold the same words — equal, or the two 32-bit integer readings of one
    /// word.
    pub const fn same_words(self, other: DataFormat) -> bool {
        matches!(
            (self, other),
            (DataFormat::Fp16, DataFormat::Fp16)
                | (DataFormat::Fp32, DataFormat::Fp32)
                | (DataFormat::Fp8, DataFormat::Fp8)
                | (DataFormat::Int64, DataFormat::Int64)
                | (
                    DataFormat::UInt32 | DataFormat::Int32,
                    DataFormat::UInt32 | DataFormat::Int32
                )
        )
    }

    /// Bytes per element.
    pub const fn word_bytes(self) -> Bytes {
        Bytes::new(match self {
            DataFormat::Fp16 => 2,
            DataFormat::Fp32 | DataFormat::UInt32 | DataFormat::Int32 => 4,
            DataFormat::Fp8 => 1,
            DataFormat::Int64 => 8,
        })
    }

    /// Elements per stick on generation `G`.
    pub const fn lanes<G: DeviceGen>(self) -> Lanes {
        Lanes::new(G::FLIT_BYTES / self.word_bytes().get())
    }

    /// The name dxp reads for this format.
    pub const fn wire(self) -> &'static str {
        match self {
            DataFormat::Fp16 => "SEN169_FP16",
            DataFormat::Fp32 => "IEEE_FP32",
            DataFormat::Fp8 => "SEN143_FP8",
            DataFormat::UInt32 => "SENUINT32",
            DataFormat::Int32 => "IEEE_INT32",
            DataFormat::Int64 => "IEEE_INT64",
        }
    }
}
