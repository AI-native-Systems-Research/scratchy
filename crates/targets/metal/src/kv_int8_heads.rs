// SPDX-License-Identifier: Apache-2.0
//! The head dims Metal's int8 KV kernels code, as the codec rule takes them
//! ([`crate::kv_int8::METAL_CODED_HEADS`]) and as the build writes them into the kernels
//! (`kv_int8_codec.h`). It depends on nothing, so the build script can include it.

/// The narrowest head the kernels code: the decode splits a head's dims over whole 32-lane
/// simdgroups, two at the least.
pub const MIN_CODED_HEAD_DIM: u32 = 64;

/// The widest head the kernels code: their threadgroup arrays hold one element per thread of a
/// head.
pub const MAX_CODED_HEAD_DIM: u32 = 512;
