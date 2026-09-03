// SPDX-License-Identifier: Apache-2.0
//! Standalone GGUF file format parser.
//!
//! Standalone GGUF binary format reader. This module provides:
//! - `Content`: parsed GGUF header with metadata and tensor info
//! - `Value`/`ValueType`: metadata value types
//! - `TensorInfo`: per-tensor shape, dtype tag, and byte offset
//!
//! The dtype is stored as a raw `u32` tag matching the GGUF/GGML spec.
//! Consumers map to their own dtype enums via `GgufDType::from_u32()`.

use byteorder::{LittleEndian, ReadBytesExt};
use std::collections::HashMap;

pub const DEFAULT_ALIGNMENT: u64 = 32;

// ---------------------------------------------------------------------------
// GGUF dtype tag (raw u32 from the spec)
// ---------------------------------------------------------------------------

/// Raw GGML dtype tag as stored in GGUF files.
///
/// This is the u32 value from the GGUF spec, not an enum — consumers should
/// map to their own dtype enum via `from_u32()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GgufDType(pub u32);

impl GgufDType {
    pub const F32: Self = Self(0);
    pub const F16: Self = Self(1);
    pub const Q4_0: Self = Self(2);
    pub const Q4_1: Self = Self(3);
    pub const Q5_0: Self = Self(6);
    pub const Q5_1: Self = Self(7);
    pub const Q8_0: Self = Self(8);
    pub const Q8_1: Self = Self(9);
    pub const Q2K: Self = Self(10);
    pub const Q3K: Self = Self(11);
    pub const Q4K: Self = Self(12);
    pub const Q5K: Self = Self(13);
    pub const Q6K: Self = Self(14);
    pub const Q8K: Self = Self(15);
    pub const IQ2_XXS: Self = Self(16);
    pub const IQ2_XS: Self = Self(17);
    pub const IQ1_S: Self = Self(19);
    pub const IQ4_NL: Self = Self(20);
    pub const IQ3_S: Self = Self(21);
    pub const IQ2_S: Self = Self(22);
    pub const IQ4_XS: Self = Self(23);
    pub const IQ1_M: Self = Self(29);
    pub const BF16: Self = Self(30);

    /// Size in bytes of one quantization block (type_size in GGML).
    pub const fn type_size(self) -> usize {
        match self.0 {
            0 => 4,      // F32
            1 | 30 => 2, // F16, BF16
            2 => 18,     // Q4_0
            3 => 20,     // Q4_1
            6 => 22,     // Q5_0
            7 => 24,     // Q5_1
            8 => 34,     // Q8_0
            9 => 40,     // Q8_1
            10 => 84,    // Q2K
            11 => 110,   // Q3K
            12 => 144,   // Q4K
            13 => 176,   // Q5K
            14 => 210,   // Q6K
            15 => 292,   // Q8K
            16 => 66,    // IQ2_XXS
            17 => 74,    // IQ2_XS
            19 => 50,    // IQ1_S
            20 => 18,    // IQ4_NL
            21 => 110,   // IQ3_S
            22 => 82,    // IQ2_S
            23 => 136,   // IQ4_XS
            29 => 56,    // IQ1_M
            _ => 0,
        }
    }

    /// Number of elements per quantization block (block_size in GGML).
    pub const fn block_size(self) -> usize {
        match self.0 {
            0 | 1 | 30 => 1,               // F32, F16, BF16
            2..=3 | 6..=9 => 32,           // Q4_0..Q8_1
            10..=15 => 256,                // Q2K..Q8K
            16 | 17 | 19 | 21 | 22 => 256, // IQ2_XXS, IQ2_XS, IQ1_S, IQ3_S, IQ2_S
            20 => 32,                      // IQ4_NL
            23 => 256,                     // IQ4_XS
            29 => 256,                     // IQ1_M
            _ => 1,
        }
    }

    /// Whether this is an unquantized float type (F32, F16, BF16).
    pub const fn is_float(self) -> bool {
        matches!(self.0, 0 | 1 | 30)
    }
}

// ---------------------------------------------------------------------------
// Shape
// ---------------------------------------------------------------------------

/// Minimal tensor shape — just a Vec of dimensions.
#[derive(Debug, Clone)]
pub struct Shape(Vec<usize>);

impl Shape {
    pub fn dims(&self) -> &[usize] {
        &self.0
    }

    pub fn elem_count(&self) -> usize {
        self.0.iter().product()
    }
}

impl From<Vec<usize>> for Shape {
    fn from(dims: Vec<usize>) -> Self {
        Self(dims)
    }
}

// ---------------------------------------------------------------------------
// TensorInfo
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct TensorInfo {
    pub ggml_dtype: GgufDType,
    pub shape: Shape,
    pub offset: u64,
}

// ---------------------------------------------------------------------------
// Value / ValueType
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ValueType {
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    U64,
    I64,
    F32,
    F64,
    Bool,
    String,
    Array,
}

impl ValueType {
    fn from_u32(v: u32) -> Result<Self, String> {
        match v {
            0 => Ok(Self::U8),
            1 => Ok(Self::I8),
            2 => Ok(Self::U16),
            3 => Ok(Self::I16),
            4 => Ok(Self::U32),
            5 => Ok(Self::I32),
            6 => Ok(Self::F32),
            7 => Ok(Self::Bool),
            8 => Ok(Self::String),
            9 => Ok(Self::Array),
            10 => Ok(Self::U64),
            11 => Ok(Self::I64),
            12 => Ok(Self::F64),
            _ => Err(format!("unrecognized value-type {v:#08x}")),
        }
    }
}

#[derive(Debug, Clone)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    String(String),
    Array(Vec<Value>),
}

impl Value {
    pub fn to_u8(&self) -> Result<u8, String> {
        match self {
            Self::U8(v) => Ok(*v),
            v => Err(format!("not a u8 {v:?}")),
        }
    }

    pub fn to_i8(&self) -> Result<i8, String> {
        match self {
            Self::I8(v) => Ok(*v),
            v => Err(format!("not a i8 {v:?}")),
        }
    }

    pub fn to_u16(&self) -> Result<u16, String> {
        match self {
            Self::U16(v) => Ok(*v),
            v => Err(format!("not a u16 {v:?}")),
        }
    }

    pub fn to_i16(&self) -> Result<i16, String> {
        match self {
            Self::I16(v) => Ok(*v),
            v => Err(format!("not a i16 {v:?}")),
        }
    }

    pub fn to_u32(&self) -> Result<u32, String> {
        match self {
            Self::U32(v) => Ok(*v),
            v => Err(format!("not a u32 {v:?}")),
        }
    }

    pub fn to_i32(&self) -> Result<i32, String> {
        match self {
            Self::I32(v) => Ok(*v),
            v => Err(format!("not a i32 {v:?}")),
        }
    }

    pub fn to_u64(&self) -> Result<u64, String> {
        match self {
            Self::U64(v) => Ok(*v),
            Self::U8(v) => Ok(*v as u64),
            Self::U16(v) => Ok(*v as u64),
            Self::U32(v) => Ok(*v as u64),
            Self::Bool(v) => Ok(*v as u64),
            v => Err(format!("not a u64 or upcastable to u64 {v:?}")),
        }
    }

    pub fn to_i64(&self) -> Result<i64, String> {
        match self {
            Self::I64(v) => Ok(*v),
            v => Err(format!("not a i64 {v:?}")),
        }
    }

    pub fn to_f32(&self) -> Result<f32, String> {
        match self {
            Self::F32(v) => Ok(*v),
            v => Err(format!("not a f32 {v:?}")),
        }
    }

    pub fn to_f64(&self) -> Result<f64, String> {
        match self {
            Self::F64(v) => Ok(*v),
            v => Err(format!("not a f64 {v:?}")),
        }
    }

    pub fn to_bool(&self) -> Result<bool, String> {
        match self {
            Self::Bool(v) => Ok(*v),
            v => Err(format!("not a bool {v:?}")),
        }
    }

    pub fn to_vec(&self) -> Result<&Vec<Value>, String> {
        match self {
            Self::Array(v) => Ok(v),
            v => Err(format!("not a vec {v:?}")),
        }
    }

    pub fn to_string(&self) -> Result<&String, String> {
        match self {
            Self::String(v) => Ok(v),
            v => Err(format!("not a string {v:?}")),
        }
    }

    fn read<R: std::io::Read>(
        reader: &mut R,
        value_type: ValueType,
        magic: &VersionedMagic,
    ) -> Result<Self, String> {
        let v = match value_type {
            ValueType::U8 => Self::U8(reader.read_u8().map_err(|e| e.to_string())?),
            ValueType::I8 => Self::I8(reader.read_i8().map_err(|e| e.to_string())?),
            ValueType::U16 => Self::U16(
                reader
                    .read_u16::<LittleEndian>()
                    .map_err(|e| e.to_string())?,
            ),
            ValueType::I16 => Self::I16(
                reader
                    .read_i16::<LittleEndian>()
                    .map_err(|e| e.to_string())?,
            ),
            ValueType::U32 => Self::U32(
                reader
                    .read_u32::<LittleEndian>()
                    .map_err(|e| e.to_string())?,
            ),
            ValueType::I32 => Self::I32(
                reader
                    .read_i32::<LittleEndian>()
                    .map_err(|e| e.to_string())?,
            ),
            ValueType::U64 => Self::U64(
                reader
                    .read_u64::<LittleEndian>()
                    .map_err(|e| e.to_string())?,
            ),
            ValueType::I64 => Self::I64(
                reader
                    .read_i64::<LittleEndian>()
                    .map_err(|e| e.to_string())?,
            ),
            ValueType::F32 => Self::F32(
                reader
                    .read_f32::<LittleEndian>()
                    .map_err(|e| e.to_string())?,
            ),
            ValueType::F64 => Self::F64(
                reader
                    .read_f64::<LittleEndian>()
                    .map_err(|e| e.to_string())?,
            ),
            ValueType::Bool => match reader.read_u8().map_err(|e| e.to_string())? {
                0 => Self::Bool(false),
                1 => Self::Bool(true),
                b => return Err(format!("unexpected bool value {b}")),
            },
            ValueType::String => Self::String(read_string(reader, magic)?),
            ValueType::Array => {
                let value_type = reader
                    .read_u32::<LittleEndian>()
                    .map_err(|e| e.to_string())?;
                let value_type = ValueType::from_u32(value_type)?;
                let len = match magic {
                    VersionedMagic::GgufV1 => reader
                        .read_u32::<LittleEndian>()
                        .map_err(|e| e.to_string())?
                        as usize,
                    VersionedMagic::GgufV2 | VersionedMagic::GgufV3 => reader
                        .read_u64::<LittleEndian>()
                        .map_err(|e| e.to_string())?
                        as usize,
                };
                let mut vs = Vec::with_capacity(len);
                for _ in 0..len {
                    vs.push(Value::read(reader, value_type, magic)?);
                }
                Self::Array(vs)
            }
        };
        Ok(v)
    }
}

// ---------------------------------------------------------------------------
// Magic / Version
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VersionedMagic {
    GgufV1,
    GgufV2,
    GgufV3,
}

impl VersionedMagic {
    fn read<R: std::io::Read>(reader: &mut R) -> Result<Self, String> {
        let magic = reader
            .read_u32::<LittleEndian>()
            .map_err(|e| e.to_string())?;
        match magic {
            0x46554747 | 0x47475546 => {}
            _ => return Err(format!("unknown magic 0x{magic:08x}")),
        }
        let version = reader
            .read_u32::<LittleEndian>()
            .map_err(|e| e.to_string())?;
        match version {
            1 => Ok(Self::GgufV1),
            2 => Ok(Self::GgufV2),
            3 => Ok(Self::GgufV3),
            _ => Err(format!("gguf: unsupported version {version}")),
        }
    }
}

// ---------------------------------------------------------------------------
// Content
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct Content {
    pub magic: VersionedMagic,
    pub metadata: HashMap<String, Value>,
    pub tensor_infos: HashMap<String, TensorInfo>,
    pub tensor_data_offset: u64,
}

fn read_string<R: std::io::Read>(reader: &mut R, magic: &VersionedMagic) -> Result<String, String> {
    let len = match magic {
        VersionedMagic::GgufV1 => reader
            .read_u32::<LittleEndian>()
            .map_err(|e| e.to_string())? as usize,
        VersionedMagic::GgufV2 | VersionedMagic::GgufV3 => reader
            .read_u64::<LittleEndian>()
            .map_err(|e| e.to_string())?
            as usize,
    };
    let mut v = vec![0u8; len];
    reader.read_exact(&mut v).map_err(|e| e.to_string())?;
    // GGUF strings are supposed to be non-null terminated but in practice this happens.
    while let Some(0) = v.last() {
        v.pop();
    }
    Ok(String::from_utf8_lossy(&v).into_owned())
}

impl Content {
    pub fn read<R: std::io::Seek + std::io::Read>(reader: &mut R) -> Result<Self, String> {
        let magic = VersionedMagic::read(reader)?;

        let tensor_count = match magic {
            VersionedMagic::GgufV1 => reader
                .read_u32::<LittleEndian>()
                .map_err(|e| e.to_string())? as usize,
            VersionedMagic::GgufV2 | VersionedMagic::GgufV3 => reader
                .read_u64::<LittleEndian>()
                .map_err(|e| e.to_string())?
                as usize,
        };
        let metadata_kv_count = match magic {
            VersionedMagic::GgufV1 => reader
                .read_u32::<LittleEndian>()
                .map_err(|e| e.to_string())? as usize,
            VersionedMagic::GgufV2 | VersionedMagic::GgufV3 => reader
                .read_u64::<LittleEndian>()
                .map_err(|e| e.to_string())?
                as usize,
        };

        let mut metadata = HashMap::new();
        for _idx in 0..metadata_kv_count {
            let key = read_string(reader, &magic)?;
            let value_type = reader
                .read_u32::<LittleEndian>()
                .map_err(|e| e.to_string())?;
            let value_type = ValueType::from_u32(value_type)?;
            let value = Value::read(reader, value_type, &magic)?;
            metadata.insert(key, value);
        }

        let mut tensor_infos = HashMap::new();
        for _idx in 0..tensor_count {
            let tensor_name = read_string(reader, &magic)?;
            let n_dimensions = reader
                .read_u32::<LittleEndian>()
                .map_err(|e| e.to_string())?;

            let mut dimensions: Vec<usize> = match magic {
                VersionedMagic::GgufV1 => {
                    let mut dims = vec![0u32; n_dimensions as usize];
                    reader
                        .read_u32_into::<LittleEndian>(&mut dims)
                        .map_err(|e| e.to_string())?;
                    dims.into_iter().map(|c| c as usize).collect()
                }
                VersionedMagic::GgufV2 | VersionedMagic::GgufV3 => {
                    let mut dims = vec![0u64; n_dimensions as usize];
                    reader
                        .read_u64_into::<LittleEndian>(&mut dims)
                        .map_err(|e| e.to_string())?;
                    dims.into_iter().map(|c| c as usize).collect()
                }
            };

            dimensions.reverse();
            let ggml_dtype = reader
                .read_u32::<LittleEndian>()
                .map_err(|e| e.to_string())?;
            let offset = reader
                .read_u64::<LittleEndian>()
                .map_err(|e| e.to_string())?;
            tensor_infos.insert(
                tensor_name,
                TensorInfo {
                    shape: Shape::from(dimensions),
                    offset,
                    ggml_dtype: GgufDType(ggml_dtype),
                },
            );
        }

        let position = reader.stream_position().map_err(|e| e.to_string())?;
        let alignment = match metadata.get("general.alignment") {
            Some(Value::U8(v)) => *v as u64,
            Some(Value::U16(v)) => *v as u64,
            Some(Value::U32(v)) => *v as u64,
            Some(Value::I8(v)) if *v >= 0 => *v as u64,
            Some(Value::I16(v)) if *v >= 0 => *v as u64,
            Some(Value::I32(v)) if *v >= 0 => *v as u64,
            _ => DEFAULT_ALIGNMENT,
        };
        let tensor_data_offset = position.div_ceil(alignment) * alignment;
        Ok(Self {
            magic,
            metadata,
            tensor_infos,
            tensor_data_offset,
        })
    }
}
