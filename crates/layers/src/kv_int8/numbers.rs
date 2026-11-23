// SPDX-License-Identifier: Apache-2.0
//! The int8 KV codec's numbers, the one definition every target's kernels take theirs from: a
//! target's build reads this file and writes them into its kernels' source. It holds constants
//! alone and depends on nothing, so a build script can include it.

/// The sign hash (splitmix64's finalizer): `e` plus [`SIGN_SEED`], then per round `z ^= z >> shift;
/// z *= mul`, then a last `z ^= z >> SIGN_LAST_SHIFT`; element `e`'s sign is its low bit.
pub const SIGN_SEED: u64 = 0x9E37_79B9_7F4A_7C15;
/// The hash's rounds, `(shift, mul)` each.
pub const SIGN_ROUNDS: [(u32, u64); 2] = [(30, 0xBF58_476D_1CE4_E5B9), (27, 0x94D0_49BB_1331_11EB)];
/// The hash's last shift.
pub const SIGN_LAST_SHIFT: u32 = 31;

/// The largest positive code and the most negative one.
pub const MAX_CODE: f32 = 127.0;
pub const MIN_CODE: f32 = -128.0;
