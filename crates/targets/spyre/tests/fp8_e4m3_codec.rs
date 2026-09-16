// SPDX-License-Identifier: Apache-2.0
//! `float8_e4m3fn` — the encoding a W8A8 weight is STORED in, checked exhaustively.
//!
//! ⛔ WHY THIS LIVES HERE AND NOT NEXT TO THE CODEC. `ktir-core`'s own test module does not compile
//! in this tree — pre-existing, and it predates this work: at `HEAD` its helpers already build
//! `AffineExpr::Add(Rc::new(a), ..)` (`affine.rs:935`) against a type whose doc at `:33` says
//! children are BORROWS, not `Rc`. `ktir-emulator`'s suite is mostly dark for two measured reasons:
//! of its 36 test targets only 4 compile, 7 of the rest `include_str!` `.mlir` fixtures from an
//! `examples/` tree that was never vendored, and the others still call the pre-arena string IR
//! (`Operation::new(Some("%r"), name, &["%a"])`). So the nearest home whose tests actually RUN is
//! this crate, which already depends on `ktir-core`.
//!
//! ⭐ EXHAUSTIVE, WHICH IS CHEAP HERE: e4m3 has 256 bit patterns, so "spot-check a few values" is
//! strictly worse than checking all of them — the same reasoning `ktir-core`'s own codec tests give
//! for sweeping every one of f16's 65536 patterns.

use ktir_core::codec::{e4m3_to_f32, f32_to_e4m3};

/// The only NaN in e4m3fn is `S.1111.111` — the format has no infinities, so the encodings that
/// would be inf in IEEE binary8 are ordinary finite numbers instead.
const NAN_MAG: u8 = 0x7F;

#[test]
fn every_bit_pattern_round_trips() {
    for b in 0u8..=255 {
        if b & 0x7F == NAN_MAG {
            assert!(e4m3_to_f32(b).is_nan(), "0x{b:02x} must decode to NaN");
            continue;
        }
        let v = e4m3_to_f32(b);
        let back = f32_to_e4m3(v);
        assert_eq!(
            back, b,
            "0x{b:02x} decoded to {v} and re-encoded to 0x{back:02x}"
        );
    }
}

/// The anchors of the format, each computed from the field layout rather than copied from the
/// implementation: 1 sign / 4 exponent (bias 7) / 3 mantissa.
#[test]
fn spec_anchors() {
    // exp=0111 (7), mant=000 ⇒ (1 + 0/8) · 2^(7-7) = 1.0
    assert_eq!(e4m3_to_f32(0x38), 1.0);
    // exp=1111 (15), mant=110 ⇒ (1 + 6/8) · 2^(15-7) = 1.75 · 256 = 448, the largest FINITE
    assert_eq!(e4m3_to_f32(0x7E), 448.0);
    assert_eq!(e4m3_to_f32(0xFE), -448.0);
    // exp=0001, mant=000 ⇒ 2^(1-7) = 2^-6, the smallest NORMAL
    assert_eq!(e4m3_to_f32(0x08), 2.0f32.powi(-6));
    // exp=0000, mant=001 ⇒ 1 · 2^-9, the smallest SUBNORMAL
    assert_eq!(e4m3_to_f32(0x01), 2.0f32.powi(-9));
    assert_eq!(e4m3_to_f32(0x00), 0.0);
}

/// ⛔ PAST THE MAXIMUM IS 448, NOT INF. The encoding that would be inf is the NaN, so an
/// out-of-range activation must CLAMP: a quantize step is lossy-but-finite, never a tile poisoned
/// with NaN that then propagates through every downstream matmul.
#[test]
fn out_of_range_saturates_and_does_not_become_nan() {
    for v in [449.0f32, 1.0e9, f32::MAX, f32::INFINITY] {
        assert_eq!(f32_to_e4m3(v), 0x7E, "{v} must clamp to +448");
        assert_eq!(f32_to_e4m3(-v), 0xFE, "-{v} must clamp to -448");
        assert!(!e4m3_to_f32(f32_to_e4m3(v)).is_nan());
    }
    assert_eq!(f32_to_e4m3(f32::NAN), NAN_MAG);
}

/// Rounding is to NEAREST, so a value between two grid points lands on the closer one and a value
/// exactly on a grid point is preserved. Checked against the grid the decoder itself defines, so
/// this cannot drift from the format.
#[test]
fn rounds_to_the_nearest_representable() {
    for b in 0u8..0x7F {
        let lo = e4m3_to_f32(b);
        let hi = e4m3_to_f32(b + 1);
        if !lo.is_finite() || !hi.is_finite() || hi <= lo {
            continue;
        }
        // Just above `lo` rounds back to `lo`; just below `hi` rounds to `hi`.
        let eps = (hi - lo) / 8.0;
        assert_eq!(
            f32_to_e4m3(lo + eps),
            b,
            "{} should round down to 0x{b:02x}",
            lo + eps
        );
        assert_eq!(
            f32_to_e4m3(hi - eps),
            b + 1,
            "{} should round up to 0x{:02x}",
            hi - eps,
            b + 1
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
//  THE SLICE PATH — what a real fp8 store/load actually runs.
//
//  `ops_memory::store_data` encodes a tile with `encode(&tile_data, dtype)` and the load side
//  decodes with `decode(&bytes, n, dtype)`. That pair, and `Tile`'s own fp8 storage, are separate
//  code from the scalar conversions above: they carry the ONE-BYTE-PER-ELEMENT contract, which is
//  where an f16-shaped assumption (`bytes / 2`) would silently halve every fp8 tile.
// ─────────────────────────────────────────────────────────────────────────────────────────────

use ktir_core::codec::{decode, encode};
use ktir_core::dtypes::DType;
use ktir_core::tile::Tile;

/// The values used below, chosen to land on both grids: exact powers of two, a subnormal, the
/// boundary the encoder got wrong, and the saturating extreme.
fn sample() -> Vec<f32> {
    vec![
        0.0,
        1.0,
        -1.0,
        2.0f32.powi(-6), // smallest normal
        2.0f32.powi(-9), // smallest subnormal
        0.015380859375,  // the subnormal→normal boundary that clamp(0, 7) rounded the wrong way
        448.0,
        -448.0,
        1.0e9, // saturates
    ]
}

#[test]
fn encode_is_one_byte_per_element() {
    let v = sample();
    let raw = encode(&v, DType::Fp8E4m3);
    assert_eq!(
        raw.len(),
        v.len(),
        "fp8 is ONE byte per element — a `bytes/2` length would halve every tile"
    );
}

#[test]
fn slice_round_trip_equals_the_scalar_grid() {
    let v = sample();
    let raw = encode(&v, DType::Fp8E4m3);
    let back = decode(&raw, v.len(), DType::Fp8E4m3);
    assert_eq!(back.len(), v.len());
    for (i, (&orig, &got)) in v.iter().zip(back.iter()).enumerate() {
        let want = e4m3_to_f32(f32_to_e4m3(orig));
        assert_eq!(
            got, want,
            "element {i}: {orig} decoded to {got}, want {want}"
        );
    }
}

/// A `Tile` built at `Fp8E4m3` stores one byte per element and widens on read — `len()` is the
/// ELEMENT count, not the byte count of some wider representation.
#[test]
fn fp8_tile_len_and_widening() {
    let v = sample();
    let t = Tile::compute(v.clone(), DType::Fp8E4m3, vec![v.len()]);
    assert_eq!(t.len(), v.len(), "fp8 tile len is its element count");
    let widened = t.as_f32();
    assert_eq!(widened.len(), v.len());
    for (i, (&orig, &got)) in v.iter().zip(widened.iter()).enumerate() {
        assert_eq!(got, e4m3_to_f32(f32_to_e4m3(orig)), "element {i} of {orig}");
    }
}
