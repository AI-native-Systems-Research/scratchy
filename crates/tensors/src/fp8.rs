// SPDX-License-Identifier: Apache-2.0
//! THE `float8_e4m3fn` LAW, for every crate this tree owns.
//!
//! The conversions between `f32` and the e4m3fn grid that THIS tree states —
//! the metal target's weight decode, the spyre numeric gate's golden, and the
//! subtile host reference of a float8 KV cache's quantize — live here, on the
//! dtype's own crate, once. Metal's `layers_quant` used to carry a second
//! decode of its own; it consumes this one now.
//!
//! `ktir-core` (vendored from torch-spyre/ktir-cpu, dependency-free by its
//! RFC 0682) keeps its own copy — upstream's, not ours to move — and the spyre
//! crate's `fp8_e4m3_codec` fixture pins the two statements byte-for-byte
//! equal, so the vendored copy cannot drift from this one silently.

/// The largest finite `float8_e4m3fn` value — `0x7E`, exponent `1111` mantissa `110`, `1.75·2⁸`.
/// e4m3fn has no infinity and `0x7F` is its NaN, so this is `torch.finfo(torch.float8_e4m3fn).max`.
pub const E4M3_FMAX: f32 = 448.0;

/// Decode one `float8_e4m3fn` byte to `f32`. 1 sign / 4 exponent (bias 7) / 3 mantissa.
///
/// e4m3fn has NO infinities: the only NaN is `S.1111.111`, and the other `1111` exponents are
/// ordinary numbers — so the largest finite magnitude is 448, not a saturating inf. Mirrors
/// `torch.float8_e4m3fn -> float32`.
pub fn e4m3_to_f32(byte: u8) -> f32 {
    // Table lookup, like `f16_bits_to_f32`: 256 inputs, filled once by the exact arithmetic in
    // `e4m3_to_f32_compute`. MEASURED: the per-element `powi` was a top bake-gate cost (every fp8
    // weight decodes through here).
    use std::sync::OnceLock;
    static TABLE: OnceLock<[f32; 256]> = OnceLock::new();
    TABLE.get_or_init(|| std::array::from_fn(|b| e4m3_to_f32_compute(b as u8)))[byte as usize]
}

/// Reference e4m3fn→f32 arithmetic (exact); used to build the table.
fn e4m3_to_f32_compute(byte: u8) -> f32 {
    let sign = if byte & 0x80 != 0 { -1.0f32 } else { 1.0f32 };
    let exp = ((byte >> 3) & 0x0F) as i32;
    let mant = (byte & 0x07) as i32;
    if exp == 0x0F && mant == 0x07 {
        return f32::NAN;
    }
    let mag = if exp == 0 {
        // Subnormal: mant · 2^(1-7) / 8 = mant · 2^-9.
        (mant as f32) * 2.0f32.powi(-9)
    } else {
        // Normal: (1 + mant/8) · 2^(exp-7).
        (1.0 + (mant as f32) / 8.0) * 2.0f32.powi(exp - 7)
    };
    sign * mag
}

/// Encode one `f32` to a `float8_e4m3fn` byte, round-to-nearest-even — THE CARD'S `qfp8ch`, byte
/// for byte, including past the finite range.
///
/// ⛔⛔⛔ NO SATURATION, BECAUSE THE CARD DOES NOT SATURATE. MEASURED on the card (granite tp/8
/// pod, one `qfp8ch` descriptor over every finite fp16 value in `[448, 65504]`, its negation and
/// `±inf`): a value that ROUNDS to `±448` — everything below the midpoint `464` to the next grid
/// step `480`, which e4m3fn does not have — is `0x7E`/`0xFE`; everything at or past `464`, `±inf`
/// included, is the NaN encoding `0x7F`/`0xFF`. This encoder used to clamp every magnitude past 448
/// to `0x7E`, which made the EMULATOR's numeric gate pass a program the card answers with NaN: the
/// convert is lossy-but-finite on one executor and poisons the tile on the other. A program must
/// state its own clamp to ±448 before an e4m3 convert (the W8A8 chain's `clip`, the MoE kernel's
/// `tl.minimum`/`tl.maximum`), and with this encoder the emulator sees what the card sees when it
/// does not. The fixture table is `fp8_e4m3_codec.rs::the_encoder_is_the_cards_qfp8ch`.
pub fn f32_to_e4m3(v: f32) -> u8 {
    if v.is_nan() {
        return 0x7F;
    }
    let sign: u8 = if v.is_sign_negative() { 0x80 } else { 0 };
    let a = v.abs();
    if a == 0.0 {
        return sign;
    }
    // The finite range ends at 448 (`0x7E`), and the next step e4m3fn would have is 480, which is
    // the NaN encoding. Round-to-nearest therefore sends `[448, 464)` to 448 and `[464, ∞]` to
    // NaN — the tie at exactly 464 rounds to the EVEN mantissa, `111`, which is NaN (measured).
    const NAN_FROM: f32 = 464.0;
    if a >= NAN_FROM {
        return sign | 0x7F;
    }
    if a >= E4M3_FMAX {
        return sign | 0x7E; // exp 1111, mant 110 — the largest finite
    }
    // Subnormal range: below 2^-6, the grid is a multiple of 2^-9.
    //
    // ⛔ THE TOP STEP CARRIES INTO THE EXPONENT, so the clamp is 8 and not 7. A value just under
    // 2^-6 rounds to `q == 8`, and 8 IS the byte for the smallest NORMAL (exp 0001, mant 000 =
    // 2^-6) — the subnormal and normal grids meet there, which is exactly what makes the encoding
    // continuous. Clamping to 7 pushed such a value back DOWN a whole grid step to the largest
    // subnormal, i.e. it rounded away from nearest: 0.01538086 sat 0.000244 from 2^-6 and
    // 0.001709 from 7·2^-9, and came back as the FARTHER one.
    // ⭐ NO TRANSCENDENTALS ON THE HOT PATH: the powers of two are constants and the exponent is
    // the f32's own (every `a` past the subnormal branch is an f32 normal), so encoding is a few
    // integer ops. MEASURED: `powi`/`log2` per element were the bake gate's synth-staging cost
    // (`__powisf2` + `log2f` atop the profile). Bit-identical to the `log2().floor()` spelling —
    // pinned by `the_fast_e4m3_encoder_matches_the_reference_spelling`.
    if a < 0.015_625 {
        // 2^-6; the subnormal grid is 2^-9 = 1/512.
        let q = (a * 512.0).round() as i32;
        return sign | (q.clamp(0, 8) as u8);
    }
    let exp = ((a.to_bits() >> 23) & 0xFF) as i32 - 127;
    let frac = a / f32::from_bits(((exp + 127) as u32) << 23) - 1.0;
    let mut mant = (frac * 8.0).round() as i32;
    let mut exp = exp;
    if mant == 8 {
        mant = 0;
        exp += 1;
    }
    if exp + 7 >= 0x0F && mant >= 7 {
        return sign | 0x7E;
    }
    sign | (((exp + 7) as u8) << 3) | (mant as u8 & 0x07)
}

/// The SATURATING e4m3 store every clamping kernel performs: `clamp(±448)` then the encode —
/// `quantize_kv.py`'s `tl.minimum(tl.maximum(x, -FMAX), FMAX).to(f8)`, the W8A8 chain's clip. The
/// clamp is the KERNEL's own statement (see [`f32_to_e4m3`]: the bare encode does not saturate —
/// past 464 it is NaN), so a value reaches this fn only from a store that states it.
pub fn e4m3_saturating_encode(v: f32) -> u8 {
    f32_to_e4m3(v.clamp(-E4M3_FMAX, E4M3_FMAX))
}

/// `v` as a saturating e4m3 store's reader sees it: the clamp, the grid, and the decode — the
/// value a float8 KV pool holds for `v` and every reader of that pool (the golden's pool rows,
/// the quantize's host reference). The clamp rides inside: a stimulus the staging encodes and a
/// writer's row the quantize computes take the SAME law, never a bare round trip that would turn
/// an out-of-range draw into NaN.
pub fn e4m3_saturating_round_trip(v: f32) -> f32 {
    e4m3_to_f32(e4m3_saturating_encode(v))
}

/// Round `data` onto the e4m3fn grid in place — the narrowing a tile of this dtype performs
/// (NO clamp: the tile's own convert, not a kernel's saturating store).
pub fn round_e4m3_in_place(data: &mut [f32]) {
    for v in data.iter_mut() {
        *v = e4m3_to_f32(f32_to_e4m3(*v));
    }
}

#[cfg(test)]
mod tests {
    /// The encoder as it was spelled before the fast path: `powi` constants and a
    /// `log2().floor()` exponent. The oracle the fast path must equal bit for bit.
    fn reference(v: f32) -> u8 {
        if v.is_nan() {
            return 0x7F;
        }
        let sign: u8 = if v.is_sign_negative() { 0x80 } else { 0 };
        let a = v.abs();
        if a == 0.0 {
            return sign;
        }
        // The card's top (`f32_to_e4m3`'s doc): `[448, 464)` rounds to 448, `[464, ∞]` is NaN.
        if a >= 464.0 {
            return sign | 0x7F;
        }
        if a >= 448.0 {
            return sign | 0x7E;
        }
        if a < 2.0f32.powi(-6) {
            let q = (a / 2.0f32.powi(-9)).round() as i32;
            return sign | (q.clamp(0, 8) as u8);
        }
        let exp = a.log2().floor() as i32;
        let frac = a / 2.0f32.powi(exp) - 1.0;
        let mut mant = (frac * 8.0).round() as i32;
        let mut exp = exp;
        if mant == 8 {
            mant = 0;
            exp += 1;
        }
        if exp + 7 >= 0x0F && mant >= 7 {
            return sign | 0x7E;
        }
        sign | (((exp + 7) as u8) << 3) | (mant as u8 & 0x07)
    }

    /// THE CARD'S OWN `qfp8ch`, byte for byte — the fixture table this law is (measured on the
    /// granite tp/8 pod; see `f32_to_e4m3`'s doc). Bit-identical on a strided sweep of every
    /// finite magnitude below 448 (both signs), plus a dense window of ulps around every power of
    /// two in range — where an exponent read from the bits and a `log2().floor()` could disagree —
    /// and the specials.
    #[test]
    fn the_fast_e4m3_encoder_matches_the_reference_spelling() {
        let check = |v: f32| {
            assert_eq!(
                super::f32_to_e4m3(v),
                reference(v),
                "v = {v:e} ({:#010x})",
                v.to_bits()
            );
            assert_eq!(super::f32_to_e4m3(-v), reference(-v), "v = -{v:e}");
        };
        let top = 448.0f32.to_bits();
        let mut b = 0u32;
        while b <= top {
            check(f32::from_bits(b));
            b += 97;
        }
        for e in -10i32..=9 {
            let p = 2.0f32.powi(e).to_bits();
            for d in 0..4096u32 {
                check(f32::from_bits(p - d));
                check(f32::from_bits(p + d));
            }
        }
        for v in [
            0.0,
            448.0,
            1e9,
            f32::INFINITY,
            f32::MIN_POSITIVE,
            1e-45,
            0.015_625,
            0.001_953_125,
        ] {
            check(v);
        }
        assert_eq!(super::f32_to_e4m3(f32::NAN), reference(f32::NAN));
    }

    /// The decode table equals the exact arithmetic on all 256 bytes (NaN by class).
    #[test]
    fn the_e4m3_decode_table_is_the_exact_arithmetic() {
        for b in 0..=255u8 {
            let (got, want) = (super::e4m3_to_f32(b), super::e4m3_to_f32_compute(b));
            assert!(
                got.to_bits() == want.to_bits() || (got.is_nan() && want.is_nan()),
                "byte {b:#04x}: {got} vs {want}"
            );
        }
    }

    /// The saturating store never yields NaN: every finite value lands inside the format, and the
    /// round trip is idempotent (a stored row re-encoded is the same byte).
    #[test]
    fn the_saturating_store_stays_finite_and_settles() {
        for v in [
            0.0,
            1.0,
            447.99,
            448.0,
            463.99,
            464.0,
            1e9,
            f32::INFINITY,
            -1e9,
        ] {
            let b = super::e4m3_saturating_encode(v);
            assert!(!super::e4m3_to_f32(b).is_nan(), "v = {v:e} -> {b:#04x}");
            assert_eq!(super::e4m3_saturating_encode(super::e4m3_to_f32(b)), b);
            let rt = super::e4m3_saturating_round_trip(v);
            assert!(
                rt.abs() <= super::E4M3_FMAX || rt.is_nan(),
                "round trip out of range: {v:e} -> {rt:e}"
            );
            assert!(!rt.is_nan(), "a finite v must round-trip finite: {v:e}");
        }
    }
}
