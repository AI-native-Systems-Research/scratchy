// SPDX-License-Identifier: Apache-2.0
//! The int8 KV codec, oMLX's `affine8` (`omlx/affine4.py`, `Affine8Codec`): each head row of K and
//! V is rotated — a fixed ±1 sign per element, then an orthonormal Walsh–Hadamard transform — and
//! stored as signed 8-bit codes against one scale for the row:
//! `scale = max(max(r) / 127, −min(r) / 128)`, `code = round(r / scale)` in `−128..=127`.
//!
//! A row decodes, in the rotated domain, as `code · scale`; attention runs there — the query
//! rotated once, `q·k = rot(q)·rot(k)`, and the output, a sum of rotated values, rotated back
//! once. A layer's store holds a row per (KV slot, KV head), `slot · num_kv_heads + kv_head`: its
//! `head_dim` code bytes, and its scale as one `f32`. This module is the host reference a target's
//! kernels encode and decode against, and the rule deciding which models code their KV.

/// How a model's KV cache is stored. Fixed per model when it is built ([`codec_for`]); nothing
/// chooses between them at runtime.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum KvCodec {
    /// Uncompressed, in the model's own dtype.
    Dense,
    /// One int8 code per rotated element and one f32 scale per head row.
    Int8,
}

impl KvCodec {
    pub const fn is_int8(self) -> bool {
        matches!(self, Self::Int8)
    }
}

impl std::fmt::Display for KvCodec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Dense => f.write_str("dense"),
            Self::Int8 => f.write_str("int8"),
        }
    }
}

mod numbers;
pub use numbers::{MAX_CODE, MIN_CODE};
use numbers::{SIGN_LAST_SHIFT, SIGN_ROUNDS, SIGN_SEED};

/// The head dims a target's kernels code: the powers of two from `min` to `max`, its table's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CodedHeads {
    min: u32,
    max: u32,
}

impl CodedHeads {
    /// A target's table: the narrowest and the widest head its kernels code.
    pub const fn new(min: u32, max: u32) -> Self {
        Self { min, max }
    }

    pub const fn min(self) -> u32 {
        self.min
    }

    pub const fn max(self) -> u32 {
        self.max
    }

    /// Whether a head of `head_dim` is coded: the one law.
    pub const fn admits(self, head_dim: u32) -> bool {
        head_dim.is_power_of_two() && head_dim >= self.min && head_dim <= self.max
    }
}

/// Element `e`'s sign before the transform: a fixed ±1 the sign hash of `e` decides.
pub const fn sign(e: u32) -> f32 {
    let mut z = (e as u64).wrapping_add(SIGN_SEED);
    let mut r = 0;
    while r < SIGN_ROUNDS.len() {
        let (shift, mul) = SIGN_ROUNDS[r];
        z = (z ^ (z >> shift)).wrapping_mul(mul);
        r += 1;
    }
    z ^= z >> SIGN_LAST_SHIFT;
    match z & 1 {
        0 => 1.0,
        _ => -1.0,
    }
}

/// What a model's KV cache holds a row of: what its codec is decided from.
#[derive(Clone, Copy, Debug)]
pub enum KvRows {
    /// No KV cache (an encoder, a vision tower).
    None,
    /// A compressed latent (MLA), not per-head K and V.
    Latent,
    /// Per-head K and V: the base class's head width, and the full-context layers' (the same on
    /// a uniform model).
    Heads { head_dim: u32, global_head_dim: u32 },
}

/// Why a model keeps a dense KV cache.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DenseReason {
    NoKvCache,
    /// The codec rotates head vectors; an MLA latent is not one.
    LatentKv,
    /// A head the target's kernels do not code ([`CodedHeads`]).
    HeadDim {
        head_dim: u32,
        coded: CodedHeads,
    },
}

impl std::fmt::Display for DenseReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoKvCache => f.write_str("it has no KV cache"),
            Self::LatentKv => f.write_str("its KV cache is an MLA latent, not per-head K and V"),
            Self::HeadDim { head_dim, coded } => write!(
                f,
                "head_dim {head_dim} is not a power of two from {} to {}",
                coded.min(),
                coded.max()
            ),
        }
    }
}

/// Int8 for a model whose KV cache holds `rows`, on a target whose kernels code the heads `coded`,
/// or why it stays dense. The one rule: the lowering injects the codec, the factory provisions it
/// and the worker sizes its pool from the `KV_CODEC` this decides.
pub fn codec_for(rows: KvRows, coded: CodedHeads) -> Result<(), DenseReason> {
    let (head_dim, global_head_dim) = match rows {
        KvRows::None => return Err(DenseReason::NoKvCache),
        KvRows::Latent => return Err(DenseReason::LatentKv),
        KvRows::Heads {
            head_dim,
            global_head_dim,
        } => (head_dim, global_head_dim),
    };
    match [head_dim, global_head_dim]
        .into_iter()
        .find(|&hd| !coded.admits(hd))
    {
        Some(head_dim) => Err(DenseReason::HeadDim { head_dim, coded }),
        None => Ok(()),
    }
}

/// The orthonormal Walsh–Hadamard transform of `x` in place (its own inverse).
fn hadamard(x: &mut [f32]) {
    let n = x.len();
    let mut h = 1;
    while h < n {
        for i in (0..n).step_by(2 * h) {
            for j in i..i + h {
                let (a, b) = (x[j], x[j + h]);
                x[j] = a + b;
                x[j + h] = a - b;
            }
        }
        h *= 2;
    }
    let norm = (n as f32).sqrt().recip();
    x.iter_mut().for_each(|v| *v *= norm);
}

/// `x` rotated: signed, then transformed.
pub fn rotate(x: &[f32]) -> Vec<f32> {
    let mut r: Vec<f32> = (0u32..).zip(x).map(|(e, &v)| sign(e) * v).collect();
    hadamard(&mut r);
    r
}

/// `r` rotated back: transformed, then unsigned.
pub fn unrotate(r: &[f32]) -> Vec<f32> {
    let mut x = r.to_vec();
    hadamard(&mut x);
    (0u32..).zip(x).map(|(e, v)| sign(e) * v).collect()
}

/// A row coded: its codes and its scale.
pub fn encode_row(x: &[f32]) -> (Vec<i8>, f32) {
    let r = rotate(x);
    let hi = r.iter().copied().fold(0.0f32, f32::max);
    let lo = r.iter().copied().fold(0.0f32, f32::min);
    let scale = (hi / MAX_CODE).max(lo / MIN_CODE);
    let codes = r
        .iter()
        .map(|&v| match scale > 0.0 {
            true => (v / scale).round_ties_even().clamp(MIN_CODE, MAX_CODE) as i8,
            false => 0,
        })
        .collect();
    (codes, scale)
}

/// A row decoded in the rotated domain: `code · scale`.
pub fn decode_rotated(codes: &[i8], scale: f32) -> Vec<f32> {
    codes.iter().map(|&c| f32::from(c) * scale).collect()
}

/// A row decoded and rotated back.
pub fn decode_row(codes: &[i8], scale: f32) -> Vec<f32> {
    unrotate(&decode_rotated(codes, scale))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The rotation is orthonormal: it keeps a row's dot products, and undoes itself.
    #[test]
    fn the_rotation_keeps_dot_products() {
        let x: Vec<f32> = (0..128)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) / 7.0)
            .collect();
        let y: Vec<f32> = (0..128)
            .map(|i| ((i * 11 % 53) as f32 - 26.0) / 5.0)
            .collect();
        let dot = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(p, q)| p * q).sum::<f32>();
        assert!((dot(&x, &y) - dot(&rotate(&x), &rotate(&y))).abs() < 1e-2);
        for (a, b) in x.iter().zip(unrotate(&rotate(&x))) {
            assert!((a - b).abs() < 1e-4);
        }
    }

    /// A row round-trips within half a code step a rotated element.
    #[test]
    fn a_row_round_trips() {
        for dim in [64usize, 128, 256, 512] {
            let x: Vec<f32> = (0..dim)
                .map(|i| ((i * 37 % 101) as f32 - 50.0) / 7.0)
                .collect();
            let (codes, scale) = encode_row(&x);
            let back = decode_row(&codes, scale);
            let err = x
                .iter()
                .zip(&back)
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f32>()
                .sqrt();
            assert!(
                err <= 0.5 * scale * (dim as f32).sqrt(),
                "dim {dim}: error {err}, scale {scale}"
            );
        }
    }

    /// The codec takes a power-of-two head no wider than its kernels hold, and per-head K and V.
    #[test]
    fn the_codec_takes_power_of_two_heads() {
        let coded = CodedHeads::new(64, 512);
        let g = |head_dim| KvRows::Heads {
            head_dim,
            global_head_dim: head_dim,
        };
        let dense = |head_dim| Err(DenseReason::HeadDim { head_dim, coded });
        assert_eq!(codec_for(g(128), coded), Ok(()));
        assert_eq!(codec_for(g(512), coded), Ok(()));
        assert_eq!(codec_for(g(96), coded), dense(96));
        assert_eq!(codec_for(g(32), coded), dense(32));
        assert_eq!(codec_for(g(1024), coded), dense(1024));
        assert_eq!(codec_for(KvRows::Latent, coded), Err(DenseReason::LatentKv));
        assert_eq!(codec_for(KvRows::None, coded), Err(DenseReason::NoKvCache));
    }
}
