// SPDX-License-Identifier: Apache-2.0
//! IEEE-fp16 ⟷ SEN169 (DLF16) bit conversion — the device-native fp16 codec, lifted out of C++.
//!
//! The AIU's fp16 is SEN169/DLF16 (sign:1, exponent:6 bias-31, mantissa:9), NOT IEEE fp16
//! (sign:1, exponent:5 bias-15, mantissa:10). Every 2-byte element is converted IEEE→sen on H2D
//! and sen→IEEE on D2H. The scalar mappings here are a VERBATIM port of deeptools
//! `util/sen_data_convert.cpp` (`Ieeef16BinToFp16bin` / `Fp16BinToIeeef16Bin`); the bulk forms go
//! through two full-domain 64K tables **built at compile time** (16 bits in, 16 bits out — the
//! whole mapping is 128 KB of rodata, L2-resident).
//!
//! 🛑 THIS IS A PORT, SO IT REPRODUCES deeptools EXACTLY, quirks included (e.g. sen exp-field 63
//! with mantissa 0x1FF converts to INF, not NaN — that is what the C++ does). The executor
//! additionally sweeps both scalar maps against the linked SDK's own `deeptools::` functions over
//! the whole 65536-value domain once at session load, and refuses the session on any mismatch — so
//! a divergence between this port and the SDK is a loud startup error, never silent garble.

/// One IEEE-754 binary16 value, as its raw bits. The type is the unit: an [`IeeeF16`] can only
/// become device bytes through [`ieee_to_sen`], so a host buffer and a device buffer cannot be
/// mixed up without the compiler noticing.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct IeeeF16(pub u16);

/// One SEN169/DLF16 value (the device-native fp16), as its raw bits.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct SenF16(pub u16);

/// IEEE fp16 → SEN169. Verbatim `Ieeef16BinToFp16bin` (sen_data_convert.cpp):
/// round-half-up 10→9 mantissa, subnormals normalized (flushed to signed zero if the exponent
/// falls below sen minimum), INF/NaN → sen exp-field 63, and the rounding carry past IEEE max
/// finite clamped to sen exp 46 / mantissa 0x1FF so finite stays finite.
pub const fn ieee_to_sen(v: IeeeF16) -> SenF16 {
    let val = v.0;
    let sign = val & 0x8000;
    let exp_field = (val >> 10) & 0x1F;
    let fraction = val & 0x3FF;

    if exp_field == 0 && fraction == 0 {
        return SenF16(sign); // zero, preserve sign
    }

    if exp_field == 0 {
        // IEEE FP16 subnormal: actual exponent = -14, implicit leading 0.
        // Normalize: shift fraction until bit 10 is set.
        let mut exponent: i32 = -14 + 31; // DLF16 field for actual exp = -14
        let mut frac: u32 = fraction as u32;
        while frac & 0x400 == 0 {
            frac <<= 1;
            exponent -= 1;
        }
        // DLF16 has no denormals — flush to zero if exponent fell below minimum.
        if exponent < 1 {
            return SenF16(sign);
        }
        frac &= 0x3FF; // remove the now-implicit leading 1
        // 10 → 9 mantissa bits with rounding
        let rounding = frac & 1;
        frac = (frac >> 1) + rounding;
        if frac > 0x1FF {
            frac = 0x1FF;
        }
        return SenF16(sign | ((exponent as u16) << 9) | frac as u16);
    }

    // IEEE FP16 INF / NaN (exponent field all ones) → DLF16 exp field 63.
    if exp_field == 0x1F {
        if fraction == 0 {
            return SenF16(sign | 0x7E00); // INF with sign
        }
        // NaN — preserve sign and some payload bits (10 → 9 mantissa).
        return SenF16(sign | 0x7E00 | ((fraction >> 1) & 0x01FF));
    }

    // IEEE FP16 normal
    let mut exponent: i32 = exp_field as i32 - 15 + 31;
    // 10 → 9 mantissa bits with rounding
    let rounding = (fraction & 1) as u32;
    let mut frac: u32 = ((fraction >> 1) as u32) + rounding;
    if frac > 0x1FF {
        frac = 0;
        exponent += 1;
    }
    // Rounding can carry the IEEE FP16 max finite (±65504) up to DLF16 exp field 47
    // (magnitude 2^16) — legal finite DLF16, but converting back would saturate to INF.
    // Keep finite inputs finite: clamp to DLF16 max finite at exp 46.
    if exponent > 46 {
        exponent = 46;
        frac = 0x1FF;
    }
    SenF16(sign | ((exponent as u16) << 9) | (frac as u16 & 0x1FF))
}

/// SEN169 → IEEE fp16. Verbatim `Fp16BinToIeeef16Bin` (sen_data_convert.cpp): sen exp-field 63
/// maps to INF for mantissa 0 AND 0x1FF (quirk preserved) and to NaN otherwise; exponents past
/// IEEE range saturate to INF; small exponents produce IEEE denormals down to a signed-zero flush.
pub const fn sen_to_ieee(v: SenF16) -> IeeeF16 {
    let val = v.0;
    let s = (val >> 15) & 1;
    let x = (val & 0x7E00) >> 9;
    let y = val & 0x01FF;

    if x == 63 {
        if y == 0x1FF || y == 0 {
            return IeeeF16((s << 15) | 0x7C00);
        }
        let mut nan_mantissa = (y << 1) & 0x3FF;
        if nan_mantissa == 0 {
            nan_mantissa = 1;
        }
        return IeeeF16((s << 15) | 0x7C00 | nan_mantissa);
    }

    let sen_exp = x as i32 - 31;
    let ieee_exp = sen_exp + 15;

    if ieee_exp >= 31 {
        return IeeeF16((s << 15) | 0x7C00);
    }
    if ieee_exp <= 0 {
        if ieee_exp < -10 {
            return IeeeF16(s << 15);
        }
        let shift = (1 - ieee_exp) as u32;
        let denorm_frac = ((y as u32) | 0x200) >> shift;
        if denorm_frac == 0 {
            return IeeeF16(s << 15);
        }
        return IeeeF16((s << 15) | (denorm_frac as u16 & 0x3FF));
    }

    let ieee_frac = (y << 1) & 0x3FF;
    IeeeF16((s << 15) | (((ieee_exp as u16) & 0x1F) << 10) | ieee_frac)
}

/// The full-domain conversion tables, computed AT COMPILE TIME — the mapping is a constant of the
/// two formats, so it lives in rodata, not in a runtime `OnceLock` build.
const TABLE_LEN: usize = 1 << 16;

const fn build_sen_table() -> [u16; TABLE_LEN] {
    let mut t = [0u16; TABLE_LEN];
    let mut i = 0;
    while i < TABLE_LEN {
        t[i] = ieee_to_sen(IeeeF16(i as u16)).0;
        i += 1;
    }
    t
}

const fn build_ieee_table() -> [u16; TABLE_LEN] {
    let mut t = [0u16; TABLE_LEN];
    let mut i = 0;
    while i < TABLE_LEN {
        t[i] = sen_to_ieee(SenF16(i as u16)).0;
        i += 1;
    }
    t
}

static SEN_TABLE: [u16; TABLE_LEN] = build_sen_table();
static IEEE_TABLE: [u16; TABLE_LEN] = build_ieee_table();

/// Bulk IEEE→sen over a slice of raw 16-bit elements (table lookup, memory-bound).
pub fn ieee_to_sen_slice(src: &[u16], dst: &mut [u16]) {
    assert_eq!(src.len(), dst.len());
    for (d, &s) in dst.iter_mut().zip(src) {
        *d = SEN_TABLE[s as usize];
    }
}

/// Bulk sen→IEEE over a slice of raw 16-bit elements.
pub fn sen_to_ieee_slice(src: &[u16], dst: &mut [u16]) {
    assert_eq!(src.len(), dst.len());
    for (d, &s) in dst.iter_mut().zip(src) {
        *d = IEEE_TABLE[s as usize];
    }
}

/// Bulk IEEE→sen over raw BYTE buffers (little-endian 2-byte elements) — the form the executor's
/// H2D staging works in. `dst` must be at least as long as `src`; the trailing odd byte, if any,
/// is not an element and is left alone.
pub fn ieee_to_sen_bytes(src: &[u8], dst: &mut [u8]) {
    for (s, d) in src
        .as_chunks::<2>()
        .0
        .iter()
        .zip(dst.as_chunks_mut::<2>().0)
    {
        let v = SEN_TABLE[u16::from_le_bytes(*s) as usize];
        *d = v.to_le_bytes();
    }
}

/// Bulk sen→IEEE over raw BYTE buffers — the D2H read-back form.
pub fn sen_to_ieee_bytes(src: &[u8], dst: &mut [u8]) {
    for (s, d) in src
        .as_chunks::<2>()
        .0
        .iter()
        .zip(dst.as_chunks_mut::<2>().0)
    {
        let v = IEEE_TABLE[u16::from_le_bytes(*s) as usize];
        *d = v.to_le_bytes();
    }
}

/// sen bits → f32, for the executor's debug/optrace dumps (sen→IEEE table, then the standard
/// IEEE half→single widening).
pub fn sen_to_f32(v: SenF16) -> f32 {
    half::f16::from_bits(sen_to_ieee(v).0).to_f32()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zeros_infs_nans() {
        // signed zero both ways
        assert_eq!(ieee_to_sen(IeeeF16(0x0000)), SenF16(0x0000));
        assert_eq!(ieee_to_sen(IeeeF16(0x8000)), SenF16(0x8000));
        assert_eq!(sen_to_ieee(SenF16(0x0000)), IeeeF16(0x0000));
        assert_eq!(sen_to_ieee(SenF16(0x8000)), IeeeF16(0x8000));
        // IEEE INF → sen exp 63 / mantissa 0
        assert_eq!(ieee_to_sen(IeeeF16(0x7C00)), SenF16(0x7E00));
        assert_eq!(ieee_to_sen(IeeeF16(0xFC00)), SenF16(0xFE00));
        // sen INF → IEEE INF (and the preserved quirk: mantissa 0x1FF ALSO maps to INF)
        assert_eq!(sen_to_ieee(SenF16(0x7E00)), IeeeF16(0x7C00));
        assert_eq!(sen_to_ieee(SenF16(0x7E00 | 0x1FF)), IeeeF16(0x7C00));
        // IEEE NaN stays NaN
        let n = sen_to_ieee(ieee_to_sen(IeeeF16(0x7E01))).0;
        assert!(
            n & 0x7C00 == 0x7C00 && n & 0x03FF != 0,
            "NaN survived as {n:#06x}"
        );
    }

    #[test]
    fn one_and_max_finite() {
        // 1.0: IEEE 0x3C00 (exp 15, frac 0) → sen exp field 31, mantissa 0 = 0x3E00
        assert_eq!(ieee_to_sen(IeeeF16(0x3C00)), SenF16(31 << 9));
        assert_eq!(sen_to_ieee(SenF16(31 << 9)), IeeeF16(0x3C00));
        // IEEE max finite 65504 (0x7BFF) rounds up past sen exp 46 → clamped finite, not INF
        let m = ieee_to_sen(IeeeF16(0x7BFF));
        assert_eq!(m, SenF16((46 << 9) | 0x1FF));
        let back = sen_to_ieee(m).0;
        assert_ne!(back & 0x7C00, 0x7C00, "max finite came back INF");
    }

    /// Every finite sen value that IEEE fp16 can represent exactly (9→10 mantissa bits is a
    /// widening) must round-trip sen→IEEE→sen bit-identically — the D2H→H2D identity that keeps a
    /// read-back-and-re-upload from drifting.
    #[test]
    fn sen_roundtrip_is_identity_where_representable() {
        for v in 0u16..=u16::MAX {
            let x = (v & 0x7E00) >> 9;
            if x == 63 {
                continue; // INF/NaN
            }
            let ieee = sen_to_ieee(SenF16(v));
            // skip sen values IEEE flushed (underflow) or saturated (overflow) — not representable
            let ieee_exp = (x as i32 - 31) + 15;
            if !(1..31).contains(&ieee_exp) {
                continue;
            }
            assert_eq!(
                ieee_to_sen(ieee),
                SenF16(v),
                "sen {v:#06x} → ieee {:#06x} did not round-trip",
                ieee.0
            );
        }
    }

    /// The compile-time tables ARE the scalar functions (they were built from them; this pins the
    /// table build against a refactor that changes one but not the other).
    #[test]
    fn tables_match_scalars() {
        for v in 0u16..=u16::MAX {
            assert_eq!(SEN_TABLE[v as usize], ieee_to_sen(IeeeF16(v)).0);
            assert_eq!(IEEE_TABLE[v as usize], sen_to_ieee(SenF16(v)).0);
        }
    }
}
