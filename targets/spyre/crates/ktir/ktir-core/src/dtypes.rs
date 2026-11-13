// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License").
//
//! Canonical KTIR dtype mappings — Rust port of `ktir_cpu/dtypes.py`.
//!
//! The Python source is a string-keyed dict with several spelling aliases per
//! canonical type (`f16`/`fp16`/`float16`). Here the canonical form is a closed
//! enum and the alias soup lives only at the parse boundary (`DType::parse`).

use std::fmt;

/// A KTIR element type. Closed set mirroring `SUPPORTED_DTYPES`.
///
/// Note `index` lowers to `I32` and `i1` to `Bool`, exactly as the Python
/// `SUPPORTED_DTYPES` table maps them onto NumPy dtypes.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum DType {
    F16,
    F32,
    Bool, // i1
    I32,  // also: si32, index
    I64,  // also: si64
    /// `float8_e4m3fn` — 1 sign / 4 exponent (bias 7) / 3 mantissa.
    ///
    /// The AIU's W8A8 weight is this, kept PACKED one byte per element: it IS the device format,
    /// not a compressed form of something else, so a view over it states `Fp8E4m3` and the tile it
    /// loads widens on read. e4m3fn has no infinities — the only NaN is `S.1111.111`, and the other
    /// `1111` exponents are normal numbers up to 448.
    Fp8E4m3,
}

impl DType {
    /// Parse a KTIR dtype string, accepting all the aliases the Python table does.
    ///
    /// Mirrors `to_np_dtype`: placeholder dtypes (`fp8`, `mxfp8`) are a hard
    /// error so any example that uses them fails loudly until implemented.
    pub fn parse(s: &str) -> Result<Self, String> {
        Ok(match s {
            "f16" | "fp16" | "float16" => DType::F16,
            "f32" | "float32" => DType::F32,
            "i1" => DType::Bool,
            "i32" | "si32" | "index" => DType::I32,
            "i64" | "si64" => DType::I64,
            "f8E4M3FN" => DType::Fp8E4m3,
            // ⛔ `fp8` STAYS AN ERROR, and it is not the same string as the one above. In IBM's
            // map `fp8`/`mxfp8` are PLACEHOLDER names pending hardware — what
            // `port_dtypes::placeholder_dtype_raises` pins — while `f8E4M3FN` is the settled MLIR
            // type keyword. Accepting the keyword and rejecting the placeholder keeps both true.
            "fp8" | "mxfp8" => {
                return Err(format!(
                    "dtype {s:?} is a placeholder pending hardware confirmation; \
                     extend DType before adding examples that use it"
                ));
            }
            _ => return Err(format!("unsupported KTIR dtype: {s:?}")),
        })
    }

    /// Element size in bytes — mirrors `bytes_per_elem`.
    pub fn bytes_per_elem(self) -> usize {
        match self {
            DType::F16 => 2,
            DType::F32 => 4,
            DType::Bool => 1,
            DType::I32 => 4,
            DType::I64 => 8,
            DType::Fp8E4m3 => 1,
        }
    }

    /// Canonical spelling — mirrors `to_ktir_dtype`'s reverse map.
    pub fn as_str(self) -> &'static str {
        match self {
            DType::F16 => "f16",
            DType::F32 => "f32",
            DType::Bool => "i1",
            DType::I32 => "i32",
            DType::I64 => "i64",
            // ⭐ MLIR's OWN BUILTIN SPELLING, not the `fp8` placeholder name. Every other entry
            // here is the MLIR type keyword, and this one has to be too: it is printed straight
            // into a view's element type, and IBM's own MLIR writes `memref<2x64x4x1xf8E4M3FN>`
            // (`~/tmp/dt/dcc/test/PT/fp8-bmm.mlir:1022`). `tensor<..xfp8>` names no builtin type.
            DType::Fp8E4m3 => "f8E4M3FN",
        }
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aliases_collapse_to_canonical() {
        for s in ["f16", "fp16", "float16"] {
            assert_eq!(DType::parse(s).unwrap(), DType::F16);
        }
        assert_eq!(DType::parse("index").unwrap(), DType::I32);
        assert_eq!(DType::parse("i1").unwrap(), DType::Bool);
    }

    #[test]
    fn roundtrip_through_canonical_string() {
        for dt in [DType::F16, DType::F32, DType::Bool, DType::I32, DType::I64] {
            assert_eq!(DType::parse(dt.as_str()).unwrap(), dt);
        }
    }

    #[test]
    fn placeholders_and_garbage_error() {
        assert!(DType::parse("fp8").is_err());
        assert!(DType::parse("mxfp8").is_err());
        assert!(DType::parse("bfloat16").is_err());
    }

    #[test]
    fn sizes_match_spec() {
        assert_eq!(DType::F16.bytes_per_elem(), 2);
        assert_eq!(DType::I64.bytes_per_elem(), 8);
    }
}
