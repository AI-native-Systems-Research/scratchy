// SPDX-License-Identifier: Apache-2.0
//! The host e4m3 and the host per-token quantizer (vLLM v0.31.0's), shared by the fp8 tests: the
//! GEMMs' inputs and the bit-for-bit reference of `QuantRowsFp8`.

use half::bf16;

/// e4m3's largest finite magnitude.
pub const E4M3_MAX: f32 = 448.0;
/// vLLM's `min_scaling_factor` for e4m3: `1 / (448 · 512)`.
pub const MIN_SCALE: f32 = 1.0 / (448.0 * 512.0);

/// `x` to e4m3 (`fn`, bias 7, no infinity, `0x7F`/`0xFF` NaN), round-to-nearest-even, saturating
/// to ±448: CUDA's `__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3)`, which TK's `warp::copy`
/// to an fp8 tile calls through `__nv_fp8x4_e4m3`. NaN is `0x7F`.
pub fn e4m3_satfinite(x: f32) -> u8 {
    if x.is_nan() {
        return 0x7F;
    }
    let sign = if x.is_sign_negative() { 0x80u8 } else { 0 };
    let a = f64::from(x.abs());
    if a >= f64::from(E4M3_MAX) {
        return sign | 0x7E;
    }
    // The spacing of e4m3 at `a`: 2^(e − 3) for a normal of exponent e ≥ −6, else 2^-9.
    let e = if a < 2f64.powi(-6) {
        -6
    } else {
        a.log2().floor() as i32
    };
    let ulp = 2f64.powi(e - 3);
    let q = a / ulp;
    let (fl, frac) = (q.floor(), q - q.floor());
    let n = if frac > 0.5 || (frac == 0.5 && fl % 2.0 == 1.0) {
        fl + 1.0
    } else {
        fl
    };
    let v = n * ulp;
    if v >= f64::from(E4M3_MAX) {
        return sign | 0x7E;
    }
    sign | encode_e4m3(v)
}

/// The e4m3 bits of a non-negative value `v` that e4m3 holds exactly, below 448.
fn encode_e4m3(v: f64) -> u8 {
    if v == 0.0 {
        return 0;
    }
    if v < 2f64.powi(-6) {
        return (v / 2f64.powi(-9)) as u8;
    }
    let e = v.log2().floor() as i32;
    let mant = (v / 2f64.powi(e) - 1.0) * 8.0;
    (((e + 7) as u8) << 3) | mant as u8
}

/// The value of e4m3 bits `b` (`0x7F`/`0xFF` NaN).
pub fn e4m3_value(b: u8) -> f64 {
    let (s, e, m) = ((b >> 7) & 1, (b >> 3) & 0xF, b & 7);
    if e == 0xF && m == 7 {
        return f64::NAN;
    }
    let mag = if e == 0 {
        f64::from(m) * 2f64.powi(-9)
    } else {
        (1.0 + f64::from(m) / 8.0) * 2f64.powi(i32::from(e) - 7)
    };
    if s == 1 { -mag } else { mag }
}

/// vLLM v0.31.0's per-token quantization of `x [rows, cols]` bf16, in f32 as its kernel computes:
/// `scale = max(amax / 448, 1 / (448·512))` and `q = e4m3_satfinite(clamp(x / scale, ±448))`, the
/// division IEEE (`scaled_fp8_conversion<false>`), its clamp NaN-dropping as `fminf`/`fmaxf`.
/// Returns `(q, scale)`.
pub fn quantize_rows(x: &[u16], rows: usize, cols: usize) -> (Vec<u8>, Vec<f32>) {
    let mut q = Vec::with_capacity(rows * cols);
    let mut scales = Vec::with_capacity(rows);
    for r in 0..rows {
        let row = &x[r * cols..][..cols];
        let amax = (row.iter()).fold(0.0f32, |m, &b| m.max(bf16::from_bits(b).to_f32().abs()));
        let scale = (amax / E4M3_MAX).max(MIN_SCALE);
        scales.push(scale);
        q.extend(row.iter().map(|&b| {
            // vLLM's `fmaxf(-max, fminf(x, max))`: a NaN quotient becomes +448.
            let v = bf16::from_bits(b).to_f32() / scale;
            e4m3_satfinite((-E4M3_MAX).max(v.min(E4M3_MAX)))
        }));
    }
    (q, scales)
}
