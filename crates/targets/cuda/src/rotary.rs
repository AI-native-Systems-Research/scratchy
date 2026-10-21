// SPDX-License-Identifier: Apache-2.0
//! THE ROTARY TABLE'S ROWS, BUILT ON THE HOST from the forward's token positions.
//!
//! One row per token, `rot_dim` bf16 wide, laid out `[cos θ(0..R/2) | sin θ(0..R/2)]` with
//! `θ_i = p · inv_freq[i]` and `inv_freq[i] = 1 / theta^(2i/R)`: the rows metal's
//! `build_mrope_cos_sin_override` (`metal/src/interpreter/metal/forward.rs`) builds for a
//! one-dimensional position. The angle is f64; `cos` and `sin` go f64 → f32 → bf16, each rounding
//! to nearest even.
//!
//! Interleaved MRoPE (qwen3.5's `mrope_section`, `mrope_interleaved`) drives each band of pairs
//! from its own position; for text all three positions are the token's, so every band gets the
//! same angle and the table is this plain one. Image positions (`[3, M]`) are not built here.
//!
//! ⛔ NOT SHARED, AND SAID SO. This duplicates metal's table math, which cannot be called from here:
//! the metal crate links `objc2`. One builder in a shared crate is a follow-up.
//!
//! Pure Rust with no toolchain, so its tests run in CI with no GPU.

use half::bf16;

/// `rope_theta`'s inverse frequency for pair `i` of a rotated width `rot_dim`.
fn inv_freq(i: usize, rot_dim: u32, theta: f64) -> f64 {
    1.0 / theta.powf(2.0 * i as f64 / f64::from(rot_dim))
}

/// `v` rounded f64 → f32 → bf16 (both round to nearest even), as its bits.
fn to_bf16(v: f64) -> u16 {
    bf16::from_f32(v as f32).to_bits()
}

/// The rotary table's rows for `positions` (one token each), `positions.len() · rot_dim` bf16
/// bits, row-major. `rot_dim` must be even.
pub fn rotary_rows(positions: &[u32], rot_dim: u32, theta: f64) -> Vec<u16> {
    let mut out = vec![0; positions.len() * rot_dim as usize];
    fill_rotary_rows(positions, rot_dim, theta, &mut out);
    out
}

/// [`rotary_rows`] into `out`, which must hold exactly `positions.len() · rot_dim` elements: a
/// worker fills a buffer it allocated once, so a forward allocates nothing.
pub fn fill_rotary_rows(positions: &[u32], rot_dim: u32, theta: f64, out: &mut [u16]) {
    assert!(rot_dim.is_multiple_of(2), "rot_dim = {rot_dim} is not even");
    let width = rot_dim as usize;
    let half = width / 2;
    assert_eq!(
        out.len(),
        positions.len() * width,
        "the table holds one rot_dim-wide row per position"
    );
    for (row, &p) in out.chunks_exact_mut(width).zip(positions) {
        let (cos, sin) = row.split_at_mut(half);
        for (i, (c, s)) in cos.iter_mut().zip(sin).enumerate() {
            let angle = f64::from(p) * inv_freq(i, rot_dim, theta);
            *c = to_bf16(angle.cos());
            *s = to_bf16(angle.sin());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// qwen3.5's rotary width (`0.25 · 256`) and `rope_theta`.
    const R: u32 = 64;
    const THETA: f64 = 1e7;

    fn widen(bits: u16) -> f64 {
        f64::from(bf16::from_bits(bits).to_f32())
    }

    /// The f64 `cos` and `sin` of the f64 angles of `p`'s row.
    fn exact(p: u32, rot_dim: u32) -> Vec<f64> {
        let angles: Vec<f64> = (0..rot_dim as usize / 2)
            .map(|i| f64::from(p) * inv_freq(i, rot_dim, THETA))
            .collect();
        (angles.iter().map(|a| a.cos()))
            .chain(angles.iter().map(|a| a.sin()))
            .collect()
    }

    fn bound(v: f64) -> f64 {
        v.abs() / 256.0 + 1.0 / 16_777_216.0
    }

    /// How many of `row`'s entries miss the bound around the f64 reference of position `p`.
    fn misses(row: &[u16], p: u32) -> usize {
        (row.iter().zip(exact(p, R)))
            .filter(|&(&b, r)| (widen(b) - r).abs() > bound(r))
            .count()
    }

    #[test]
    fn position_zero_is_cos_one_and_sin_plus_zero_bit_for_bit() {
        let row = rotary_rows(&[0], R, THETA);
        let (cos, sin) = row.split_at(R as usize / 2);
        assert!(cos.iter().all(|&b| b == 0x3F80), "{cos:x?}");
        assert!(sin.iter().all(|&b| b == 0x0000), "{sin:x?}");
    }

    #[test]
    fn every_entry_is_within_the_bound_of_the_f64_angle() {
        let positions = [0, 1, 2, 7, 1000, 65535, 200000, 262143];
        let rows = rotary_rows(&positions, R, THETA);
        for (row, &p) in rows.chunks_exact(R as usize).zip(&positions) {
            assert_eq!(misses(row, p), 0, "position {p}");
        }
    }

    #[test]
    fn the_layout_is_cos_then_sin_one_row_per_position() {
        let rows = rotary_rows(&[1, 2], R, THETA);
        assert_eq!(rows.len(), 2 * R as usize);
        // Pair 0's frequency is 1: row 1's first cos is cos(1), its first sin sin(1).
        assert_eq!(rows[0], to_bf16(1f64.cos()));
        assert_eq!(rows[32], to_bf16(1f64.sin()));
        assert_eq!(rows[64], to_bf16(2f64.cos()));
        assert_eq!(rows[96], to_bf16(2f64.sin()));
        let mut filled = vec![0xFFFF; 2 * R as usize];
        fill_rotary_rows(&[1, 2], R, THETA, &mut filled);
        assert_eq!(filled, rows);
    }

    /// The same row with its angle computed in f32, as a careless port would.
    fn f32_angle_row(p: u32) -> Vec<u16> {
        let angles: Vec<f32> = (0..R as usize / 2)
            .map(|i| {
                let inv = 1.0f32 / (THETA as f32).powf(2.0 * i as f32 / R as f32);
                p as f32 * inv
            })
            .collect();
        (angles.iter().map(|a| bf16::from_f32(a.cos()).to_bits()))
            .chain(angles.iter().map(|a| bf16::from_f32(a.sin()).to_bits()))
            .collect()
    }

    /// At position 200000 the real table is within the bound, and an f32-angle one misses it in
    /// at least one entry (the f32 ulp of 200000 is 1/64 rad, which shows on the
    /// highest-frequency pairs), so the f64 angle cannot silently become f32.
    #[test]
    fn an_f32_angle_misses_the_bound_at_position_200000() {
        let p = 200000;
        assert_eq!(misses(&rotary_rows(&[p], R, THETA), p), 0);
        let missed = misses(&f32_angle_row(p), p);
        eprintln!("f32-angle table at position {p}: {missed} of {R} entries miss the bound");
        assert!(missed >= 1, "an f32-angle table passes at position {p}");
    }

    /// At position 1000, a table using the head dim (256) in the exponent instead of the rotated
    /// width (64) differs from the real one in more than 90% of the entries.
    #[test]
    fn the_exponent_is_the_rotated_width_not_the_head_dim() {
        let real = rotary_rows(&[1000], R, THETA);
        let wrong: Vec<u16> = {
            let angles: Vec<f64> = (0..R as usize / 2)
                .map(|i| 1000.0 * inv_freq(i, 256, THETA))
                .collect();
            (angles.iter().map(|a| to_bf16(a.cos())))
                .chain(angles.iter().map(|a| to_bf16(a.sin())))
                .collect()
        };
        let differ = real.iter().zip(&wrong).filter(|(a, b)| a != b).count();
        eprintln!("head_dim exponent at position 1000: {differ} of {R} entries differ");
        assert!(differ * 10 > real.len() * 9, "{differ} of {}", real.len());
    }

    /// bf16, not f16: 1.0 is `0x3F80` in bf16 and `0x3C00` in f16.
    #[test]
    fn the_table_is_bf16() {
        assert_eq!(rotary_rows(&[0], 2, THETA)[0], 0x3F80);
        assert_ne!(
            rotary_rows(&[0], 2, THETA)[0],
            half::f16::from_f32(1.0).to_bits()
        );
    }
}
