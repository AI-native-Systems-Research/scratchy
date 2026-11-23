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

/// The widest head the codec's kernels take: their threadgroup arrays hold one element per thread
/// of a head.
pub const MAX_HEAD_DIM: u32 = 512;

/// The narrowest head the codec's kernels take: the decode attention splits a head's dims over
/// whole 32-lane simdgroups, two at the least.
pub const MIN_HEAD_DIM: u32 = 64;

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
    /// The rotation is a Walsh-Hadamard transform, over a power-of-two length from
    /// [`MIN_HEAD_DIM`] to [`MAX_HEAD_DIM`].
    HeadDim(u32),
}

impl std::fmt::Display for DenseReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoKvCache => f.write_str("it has no KV cache"),
            Self::LatentKv => f.write_str("its KV cache is an MLA latent, not per-head K and V"),
            Self::HeadDim(hd) => write!(
                f,
                "head_dim {hd} is not a power of two from {MIN_HEAD_DIM} to {MAX_HEAD_DIM}"
            ),
        }
    }
}

/// Int8 for a model whose KV cache holds `rows`, or why it stays dense. The one rule: the lowering
/// injects the codec, the factory provisions it and the worker sizes its pool from the `KV_CODEC`
/// this decides.
pub fn codec_for(rows: KvRows) -> Result<(), DenseReason> {
    let (head_dim, global_head_dim) = match rows {
        KvRows::None => return Err(DenseReason::NoKvCache),
        KvRows::Latent => return Err(DenseReason::LatentKv),
        KvRows::Heads {
            head_dim,
            global_head_dim,
        } => (head_dim, global_head_dim),
    };
    let supported = |hd: u32| hd.is_power_of_two() && (MIN_HEAD_DIM..=MAX_HEAD_DIM).contains(&hd);
    match [head_dim, global_head_dim]
        .into_iter()
        .find(|&hd| !supported(hd))
    {
        Some(hd) => Err(DenseReason::HeadDim(hd)),
        None => Ok(()),
    }
}

/// The largest positive code and the most negative one.
pub const MAX_CODE: f32 = 127.0;
pub const MIN_CODE: f32 = -128.0;

/// Element `e`'s sign before the transform: a fixed ±1 a hash of `e` decides, every target's.
pub const fn sign(e: u32) -> f32 {
    let mut z = (e as u64).wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    match z & 1 {
        0 => 1.0,
        _ => -1.0,
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
        let g = |head_dim| KvRows::Heads {
            head_dim,
            global_head_dim: head_dim,
        };
        assert_eq!(codec_for(g(128)), Ok(()));
        assert_eq!(codec_for(g(512)), Ok(()));
        assert_eq!(codec_for(g(96)), Err(DenseReason::HeadDim(96)));
        assert_eq!(codec_for(g(32)), Err(DenseReason::HeadDim(32)));
        assert_eq!(codec_for(g(1024)), Err(DenseReason::HeadDim(1024)));
        assert_eq!(codec_for(KvRows::Latent), Err(DenseReason::LatentKv));
        assert_eq!(codec_for(KvRows::None), Err(DenseReason::NoKvCache));
    }
}
