// SPDX-License-Identifier: Apache-2.0
//! Integration-test re-export of the crate's MTL4 single-op dispatch
//! helper ([`scratchy_target_metal::mtl4_dispatch`]). Kept as `mod common`
//! so the converted parity tests reach it as `common::dispatch_threadgroups`,
//! `common::shared_slice`, `common::Buffer`, etc.
//!
//! Each integration test compiles its own copy of this module and reaches a
//! different slice of the dtype surface (mamba2 rounds and decodes in f32 and
//! bf16, the bit-identity test packs bytes and bits in all three) — what one
//! target leaves unused is not dead.
#![allow(unused_imports, dead_code)]

pub use scratchy_target_metal::mtl4_dispatch::*;

use half::{bf16, f16};
pub use scratchy_target_metal::mtl4_dispatch::{Buffer, read_slice};

/// The dtype a kernel instantiation reads its activations and `T` weights in, as the
/// integration tests exercise it: f32 the reference tests' own (the fine-grained bits a
/// ULP of drift shows in), f16 and bf16 what the lowerings lower.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dtype {
    F32,
    F16,
    Bf16,
}

impl Dtype {
    pub fn tag(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::Bf16 => "bf16",
        }
    }

    /// `x` as this dtype holds it.
    pub fn round(self, x: f32) -> f32 {
        match self {
            Self::F32 => x,
            Self::F16 => f16::from_f32(x).to_f32(),
            Self::Bf16 => bf16::from_f32(x).to_f32(),
        }
    }

    /// `xs` packed as this dtype holds them.
    pub fn bytes(self, xs: &[f32]) -> Vec<u8> {
        match self {
            Self::F32 => xs.iter().flat_map(|x| x.to_le_bytes()).collect(),
            Self::F16 => xs
                .iter()
                .flat_map(|&x| f16::from_f32(x).to_bits().to_le_bytes())
                .collect(),
            Self::Bf16 => xs
                .iter()
                .flat_map(|&x| bf16::from_f32(x).to_bits().to_le_bytes())
                .collect(),
        }
    }

    /// `n` elements of `buf` as this dtype holds them, as f32.
    pub fn decode(self, buf: &Buffer, n: usize) -> Vec<f32> {
        match self {
            Self::F32 => read_slice::<f32>(buf, n),
            Self::F16 => read_slice::<u16>(buf, n)
                .into_iter()
                .map(|b| f16::from_bits(b).to_f32())
                .collect(),
            Self::Bf16 => read_slice::<u16>(buf, n)
                .into_iter()
                .map(|b| bf16::from_bits(b).to_f32())
                .collect(),
        }
    }

    /// The output's tolerance: its rounding to this dtype, plus f32 reassociation.
    pub fn tol(self, reference: f32) -> f32 {
        match self {
            Self::F32 => 2e-4 * (1.0 + reference.abs()),
            // f16 keeps eleven significand bits, bf16 eight.
            Self::F16 => 1.2e-3 * reference.abs() + 2.4e-4,
            Self::Bf16 => 1e-2 * reference.abs() + 2e-3,
        }
    }

    /// `<kernel>_<tag>`: the instantiation's symbol.
    pub fn symbol(self, kernel: &str) -> &'static str {
        Box::leak(format!("{kernel}_{}", self.tag()).into_boxed_str())
    }

    /// The output buffer's raw bit patterns, widened — equal iff the outputs are.
    pub fn bits(self, buf: &Buffer, n: usize) -> Vec<u32> {
        match self {
            Self::F32 => read_slice::<f32>(buf, n)
                .iter()
                .map(|v| v.to_bits())
                .collect(),
            _ => read_slice::<u16>(buf, n)
                .into_iter()
                .map(u32::from)
                .collect(),
        }
    }
}
