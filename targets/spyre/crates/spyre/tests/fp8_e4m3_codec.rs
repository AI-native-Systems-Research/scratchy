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

/// ⭐⭐⭐ THE ENCODER IS THE CARD'S `qfp8ch`, BYTE FOR BYTE — the bytes ONE `qfp8ch` descriptor wrote on
/// the card (32 cores, a `[1, 16384]` fp16 row → fp8, 2026-11-08; the cell and its readback are
/// `/work/tp-fp8mm-runs/qfp8-cell` on the TP pod: `in.bin`, `out.bin`, and a lane-identifying twin
/// that fixed the convert's in-stick output order). Every row below is an (fp16 input, card byte)
/// pair from that run.
///
/// ⛔ PAST THE FINITE RANGE THE CARD WRITES NaN, NOT 448. e4m3fn's largest finite is 448 (`0x7E`);
/// the next step it would have is 480, and that is the NaN encoding. Rounding to nearest sends
/// `[448, 464)` to 448 and everything at or past the midpoint 464 — `±inf` included — to
/// `0x7F`/`0xFF`. MEASURED over every finite fp16 value in `[448, 65504]` and its negation (plus the
/// head rows below, which repeat a few of them): every positive lane in `[448, 464)` was `0x7E` and
/// every one at or past 464 `0x7F` (68 / 7,370 of 7,438 lanes), and likewise `0xFE`/`0xFF` for the
/// negations (66 / 7,364 of 7,430). A program must state its own clamp to ±448 before
/// the convert — the W8A8 chain's `clip`, the MoE kernel's `tl.minimum`/`tl.maximum` — and the
/// emulator now poisons what the card poisons when it does not.
#[test]
fn the_encoder_is_the_cards_qfp8ch() {
    const CARD: &[(f32, u8)] = &[
        (0.0, 0x00),
        (1.0, 0x38),
        (-1.0, 0xB8),
        (240.0, 0x77),
        (416.0, 0x7D),
        (432.0, 0x7E),
        (447.75, 0x7E),
        (448.0, 0x7E),
        (448.25, 0x7E),
        (448.5, 0x7E),
        (449.0, 0x7E),
        (456.0, 0x7E),
        (464.0, 0x7F),
        (480.0, 0x7F),
        (500.0, 0x7F),
        (512.0, 0x7F),
        (1024.0, 0x7F),
        (4096.0, 0x7F),
        (30000.0, 0x7F),
        (65504.0, 0x7F),
        (f32::INFINITY, 0x7F),
        (-448.0, 0xFE),
        (-448.25, 0xFE),
        (-448.5, 0xFE),
        (-464.0, 0xFF),
        (-65504.0, 0xFF),
        (f32::NEG_INFINITY, 0xFF),
    ];
    for &(v, byte) in CARD {
        assert_eq!(
            f32_to_e4m3(v),
            byte,
            "{v}: the card's qfp8ch wrote 0x{byte:02x}, the encoder 0x{:02x}",
            f32_to_e4m3(v)
        );
    }
    // The band edge, at every fp16 value the card swept: `[448, 464)` is ±448 and `[464, 65504]`
    // NaN, both signs — the split the card's lanes showed.
    let fp16 = |bits: u16| half::f16::from_bits(bits).to_f32();
    for bits in 0x5F00u16..=0x7BFF {
        let v = fp16(bits);
        if v < 448.0 {
            continue;
        }
        let (pos, neg) = if v < 464.0 {
            (0x7E, 0xFE)
        } else {
            (0x7F, 0xFF)
        };
        assert_eq!(f32_to_e4m3(v), pos, "{v}");
        assert_eq!(f32_to_e4m3(-v), neg, "-{v}");
    }
    assert_eq!(
        (0x5F00u16..=0x7BFF)
            .map(fp16)
            .filter(|&v| (448.0..464.0).contains(&v))
            .count(),
        64,
        "the fp16 grid holds 64 values in [448, 464) (448 .. 463.75 at a 0.25 step)"
    );
    assert_eq!(f32_to_e4m3(f32::NAN), NAN_MAG);
    // A finite input that overflows is NaN on decode, exactly as on the card.
    assert!(e4m3_to_f32(f32_to_e4m3(464.0)).is_nan());
    assert_eq!(e4m3_to_f32(f32_to_e4m3(463.75)), 448.0);
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
/// boundary the encoder got wrong, and a value past the finite range (NaN on the card).
fn sample() -> Vec<f32> {
    vec![
        0.0,
        1.0,
        -1.0,
        2.0f32.powi(-6), // smallest normal
        2.0f32.powi(-9), // smallest subnormal
        0.015_380_859,   // the subnormal→normal boundary that clamp(0, 7) rounded the wrong way
        448.0,
        -448.0,
        1.0e9, // past the finite range: NaN, as on the card
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
        // By BITS: a value past the finite range decodes to NaN on both paths, and NaN != NaN.
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
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
        assert_eq!(
            got.to_bits(),
            e4m3_to_f32(f32_to_e4m3(orig)).to_bits(),
            "element {i} of {orig}"
        );
    }
}
