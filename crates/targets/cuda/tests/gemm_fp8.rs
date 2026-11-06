// SPDX-License-Identifier: Apache-2.0
//! FP8 S3 on the H100: the per-token quantization (`kernels/quant_rows_fp8.cu`) against the host
//! quantizer bit for bit, and the prefill fp8 GEMMs (`kernels/gemm_fp8.cu`,
//! `kernels/gemm_narrow_fp8.cu`) against an f64 reference on the same e4m3 inputs and scales.
//!
//! Built only under `nvcc` (`required-features`). No GPU fails rather than skips.
//!
//! The f64 bound is the GEMM's (`common::numerics`, the one the bf16 GEMV's f64 gate uses):
//! `|gpu − ref| ≤ 2⁻⁷·|ref| + 2⁻¹⁴·S`, `S = Σₖ |a·w|·a_scale·w_scale`: one bf16 rounding of the
//! output, and the f32 accumulation over `k`. Every worst ratio is printed. Hopper's fp8 WGMMA adds
//! at reduced precision inside the tensor core (FP8 plan Risk A); this file measures it at every
//! fp8 `k` of the three FP8 checkpoints, k = 17408 (the 27B's down projection) included, and does
//! not widen the bound for it.

// `pub`: each test binary uses a different part of `common`, and nothing in it is dead code.
pub mod common {
    pub mod numerics;
}

use std::ffi::c_void;

use common::numerics::{bf16_inputs, mismatched_bits};
use half::bf16;
use scratchy_target_cuda::kernels::{
    Fp8Operand, KernelError, expert_gemm_fp8, gate_up_gemm_fp8, gemm_fp8_abt, gemm_group_fp8_abt,
    gemm_narrow_fp8_abt, gemm_narrow_group_fp8_abt, quant_rows_fp8,
};
use scratchy_target_cuda::runtime::{CudaDevice, DeviceBuffer};

/// Every output element before a launch: a bf16 NaN, so an unwritten element fails.
const SENTINEL: u16 = 0xFFFF;
/// Extra output elements past `m·n`, which a kernel must leave untouched.
const TAIL: usize = 4096;
/// e4m3's largest finite magnitude.
const E4M3_MAX: f32 = 448.0;
/// vLLM's `min_scaling_factor` for e4m3: `1 / (448 · 512)`.
const MIN_SCALE: f32 = 1.0 / (448.0 * 512.0);

fn h100() -> CudaDevice {
    CudaDevice::open()
        .unwrap_or_else(|e| panic!("the `nvcc` feature's tests need an sm_90 GPU (H100): {e}"))
}

fn verdict(pass: bool) -> &'static str {
    if pass { "PASS" } else { "FAIL" }
}

// ── The host e4m3 and the host quantizer ───────────────────────────

/// `x` to e4m3 (`fn`, bias 7, no infinity, `0x7F`/`0xFF` NaN), round-to-nearest-even, saturating
/// to ±448: CUDA's `__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3)`, which TK's `warp::copy`
/// to an fp8 tile calls through `__nv_fp8x4_e4m3`. NaN is `0x7F`.
fn e4m3_satfinite(x: f32) -> u8 {
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
fn e4m3_value(b: u8) -> f64 {
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
fn quantize_rows(x: &[u16], rows: usize, cols: usize) -> (Vec<u8>, Vec<f32>) {
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

/// splitmix64 e4m3 weight bytes, every finite value but the largest magnitudes' tails: bits
/// `0x00..=0x7E` and their negatives, never NaN.
fn e4m3_weights(len: usize, seed: u64) -> Vec<u8> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            let b = (z % 0x7F) as u8;
            if z & (1 << 40) != 0 { b | 0x80 } else { b }
        })
        .collect()
}

/// A weight's per-channel scales, as a checkpoint's (`amax / 448` of a unit-range row): in
/// `[2⁻¹⁰, 2⁻⁸)`, bf16-exact (the checkpoint stores them bf16; the loader widens them, exactly).
fn weight_scales(n: usize, seed: u64) -> Vec<f32> {
    let u = bf16_inputs(n, seed);
    u.iter()
        .map(|&b| {
            let v = bf16::from_bits(b).to_f32().abs();
            bf16::from_f32((1.0 + v * 3.0) / 1024.0).to_f32()
        })
        .collect()
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn down_bytes(b: &DeviceBuffer) -> Vec<u8> {
    b.download()
        .expect("download")
        .iter()
        .flat_map(|h| h.to_le_bytes())
        .collect()
}

fn down_f32(b: &DeviceBuffer) -> Vec<f32> {
    down_bytes(b)
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&c| f32::from_le_bytes(c))
        .collect()
}

/// An e4m3 operand on the device: its bytes and its f32 scales, each in its own allocation.
struct Fp8Buf {
    q: DeviceBuffer,
    s: DeviceBuffer,
}

impl Fp8Buf {
    fn new(device: &CudaDevice, q: &[u8], s: &[f32]) -> Self {
        // A whole number of u16s, and at least 16 bytes beyond the data for the TMA boxes.
        let mut qb = q.to_vec();
        qb.resize(q.len().next_multiple_of(256), 0);
        let mut sb = f32_bytes(s);
        sb.resize(sb.len().next_multiple_of(256), 0);
        Self {
            q: DeviceBuffer::upload_bytes(device, &qb).expect("upload e4m3"),
            s: DeviceBuffer::upload_bytes(device, &sb).expect("upload scales"),
        }
    }

    fn op(&self) -> Fp8Operand {
        (self.q.ptr().cast_const(), self.s.ptr().cast_const())
    }
}

/// `(ref, S)` per element of `C = (A·Bᵀ)·w_scale[n]·a_scale[m]` in f64 from the e4m3 bytes and
/// scales, split across threads by row. The bytes are read through a 256-entry table of their
/// values: a whole weight widened to f64 would be 8 bytes an element.
fn fp8_reference(
    (a, a_s): (&[u8], &[f32]),
    (b, b_s): (&[u8], &[f32]),
    (m, n, k): (usize, usize, usize),
) -> Vec<(f64, f64)> {
    let lut: Vec<f64> = (0..=255u8).map(e4m3_value).collect();
    let mut out = vec![(0.0, 0.0); m * n];
    let threads = std::thread::available_parallelism().map_or(1, |p| p.get());
    let rows_per = m.div_ceil(threads);
    std::thread::scope(|s| {
        for (chunk, rows) in out.chunks_mut(rows_per * n).enumerate() {
            let lut = &lut;
            s.spawn(move || {
                for (r, row) in rows.chunks_mut(n).enumerate() {
                    let i = chunk * rows_per + r;
                    let ar: Vec<f64> = a[i * k..][..k].iter().map(|&x| lut[x as usize]).collect();
                    for (j, cell) in row.iter_mut().enumerate() {
                        let br = &b[j * k..][..k];
                        let (mut dot, mut abs) = (0.0, 0.0);
                        for (x, &y) in ar.iter().zip(br) {
                            let p = x * lut[y as usize];
                            dot += p;
                            abs += p.abs();
                        }
                        let sc = f64::from(a_s[i]) * f64::from(b_s[j]);
                        *cell = (dot * sc, abs * sc);
                    }
                }
            });
        }
    });
    out
}

/// `gpu` (`[m, n]` bf16 bits) against `refs` under the GEMM bound `2⁻⁷·|ref| + 2⁻¹⁴·S`: the
/// out-of-bound count and the worst ratio. NaN or Inf fails.
fn compare_fp8(gpu: &[u16], refs: &[(f64, f64)]) -> (usize, f64) {
    let (mut bad, mut worst) = (0usize, 0.0f64);
    for (&bits, &(r, s)) in gpu.iter().zip(refs) {
        let g = f64::from(bf16::from_bits(bits).to_f32());
        let bound = r.abs() / 128.0 + s / 16384.0;
        let ratio = if g.is_finite() {
            (g - r).abs() / bound.max(f64::MIN_POSITIVE)
        } else {
            f64::INFINITY
        };
        bad += usize::from(ratio.is_nan() || ratio > 1.0);
        worst = worst.max(ratio);
    }
    (bad, worst)
}

/// A `[rows, cols]` bf16 activation in [-1, 1] and its quantization.
fn activation(rows: usize, cols: usize, seed: u64) -> (Vec<u16>, Vec<u8>, Vec<f32>) {
    let x = bf16_inputs(rows * cols, seed);
    let (q, s) = quantize_rows(&x, rows, cols);
    (x, q, s)
}

/// A weight `[n, k]`: e4m3 bytes and per-channel scales.
fn weight(n: usize, k: usize, seed: u64) -> (Vec<u8>, Vec<f32>) {
    (e4m3_weights(n * k, seed), weight_scales(n, seed ^ 0x5CA1E))
}

/// Run `launch` on an output of `len` elements plus [`TAIL`], NaN-poisoned; return the output and
/// whether the tail was left untouched.
fn run_out(
    device: &CudaDevice,
    len: usize,
    launch: impl FnOnce(*mut c_void) -> Result<(), KernelError>,
) -> Result<(Vec<u16>, bool), KernelError> {
    let c = DeviceBuffer::upload(device, &vec![SENTINEL; len + TAIL]).expect("upload C");
    launch(c.ptr())?;
    let out = c.download().expect("download C");
    let tail = out[len..].iter().all(|&x| x == SENTINEL);
    Ok((out[..len].to_vec(), tail))
}

// ── QuantRowsFp8 ───────────────────────────────────────────────────

/// Quantize `x [rows, cols]` on the device; returns `(q, scale)`.
fn device_quant(device: &CudaDevice, x: &[u16], rows: usize, cols: usize) -> (Vec<u8>, Vec<f32>) {
    let xb = DeviceBuffer::upload(device, x).expect("upload x");
    let q = DeviceBuffer::upload_bytes(device, &vec![0xA5; rows * cols + 256]).expect("q");
    let s = DeviceBuffer::upload_bytes(device, &vec![0xA5; rows * 4 + 256]).expect("scale");
    // SAFETY: `x` holds rows·cols bf16, `q` rows·cols bytes and `scale` rows f32, all cudaMalloc'd
    // (256-byte aligned) and disjoint; the download waits for the default stream.
    unsafe {
        quant_rows_fp8(
            xb.ptr(),
            (q.ptr(), s.ptr()),
            (rows as u32, cols as u32),
            std::ptr::null_mut(),
        )
    }
    .expect("quant_rows_fp8 launch");
    let (qd, sd) = (down_bytes(&q), down_f32(&s));
    let q_tail = qd[rows * cols..].iter().all(|&b| b == 0xA5);
    let s_tail = down_bytes(&s)[rows * 4..].iter().all(|&b| b == 0xA5);
    assert!(
        q_tail && s_tail,
        "rows={rows} cols={cols}: wrote past q or scale"
    );
    (qd[..rows * cols].to_vec(), sd[..rows].to_vec())
}

/// Every `k` of the three FP8 checkpoints' fp8 GEMMs (FP8 plan §1a), and a partial last 256-column
/// tile (320): bytes and scales bit-identical to the host quantizer at 16 and 4096 rows, one row
/// at each scale extreme (all zeros: the floor; one at ±448 times a large value).
#[test]
fn quant_rows_fp8_is_the_host_quantizer_bit_for_bit() {
    let device = h100();
    let mut pass = true;
    for cols in [
        320usize, 512, 704, 2048, 2112, 2816, 4096, 5120, 6144, 8192, 17408,
    ] {
        for rows in [16usize, 4096] {
            let mut x = bf16_inputs(rows * cols, 0xF8_0000 ^ cols as u64 ^ ((rows as u64) << 32));
            // Row 0 all zeros (the scale floor), row 1 scaled up past e4m3's range.
            x[..cols].fill(0);
            for v in &mut x[cols..2 * cols] {
                *v = bf16::from_f32(bf16::from_bits(*v).to_f32() * 3.0e4).to_bits();
            }
            let (hq, hs) = quantize_rows(&x, rows, cols);
            let (dq, ds) = device_quant(&device, &x, rows, cols);
            let q_bad = dq.iter().zip(&hq).filter(|(a, b)| a != b).count();
            let s_bad = (ds.iter().zip(&hs))
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            let ok = q_bad == 0 && s_bad == 0;
            eprintln!(
                "quant_rows_fp8 rows={rows} cols={cols}: mismatched_bytes={q_bad} \
                 mismatched_scales={s_bad} {}",
                verdict(ok)
            );
            pass &= ok;
        }
    }
    assert!(
        pass,
        "a quantization differs from the host's: see the lines above"
    );
}

/// All 65,536 bf16 inputs through the device conversion: row `v` is `[v, 448, 0, …]` (64 columns),
/// so its scale is exactly 1 for every `|v| ≤ 448` and the device's byte for `v` is
/// `e4m3_satfinite(clamp(v / 1))`; above 448 (and ±inf, NaN) the scale is the row's own, and the
/// bytes still the host quantizer's. Bytes and scales bit-identical.
#[test]
fn quant_rows_fp8_converts_every_bf16_as_the_host_does() {
    let device = h100();
    let (rows, cols) = (65_536usize, 64usize);
    let mut x = vec![0u16; rows * cols];
    for v in 0..rows {
        x[v * cols] = v as u16;
        x[v * cols + 1] = bf16::from_f32(E4M3_MAX).to_bits();
    }
    let (hq, hs) = quantize_rows(&x, rows, cols);
    let (dq, ds) = device_quant(&device, &x, rows, cols);
    let mut first = None;
    let mut q_bad = 0usize;
    for v in 0..rows {
        if dq[v * cols..][..cols] != hq[v * cols..][..cols] {
            q_bad += 1;
            first.get_or_insert(v);
        }
    }
    let s_bad = (ds.iter().zip(&hs))
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    let ok = q_bad == 0 && s_bad == 0;
    eprintln!(
        "quant_rows_fp8 every bf16: mismatched_rows={q_bad} first={first:?} \
         mismatched_scales={s_bad} {}",
        verdict(ok)
    );
    assert!(ok, "the device conversion differs from the host's");
}

#[test]
fn quant_rows_fp8_refuses_shapes_it_does_not_take() {
    let device = h100();
    let x = DeviceBuffer::upload(&device, &[0u16; 4096]).expect("x");
    let q = DeviceBuffer::upload_bytes(&device, &[0xA5u8; 4096]).expect("q");
    let mut pass = true;
    for (rows, cols) in [
        (0u32, 64u32),
        (8, 64),
        (17, 64),
        (16, 0),
        (16, 32),
        (16, 100),
    ] {
        // SAFETY: the launcher refuses each shape before it touches a pointer.
        let got = unsafe {
            quant_rows_fp8(
                x.ptr(),
                (q.ptr(), q.ptr()),
                (rows, cols),
                std::ptr::null_mut(),
            )
        };
        let kept = down_bytes(&q).iter().all(|&b| b == 0xA5);
        let ok = got == Err(KernelError::RowShape { rows, cols }) && kept;
        eprintln!(
            "quant_rows_fp8 refusal rows={rows} cols={cols}: {got:?} {}",
            verdict(ok)
        );
        pass &= ok;
    }
    assert!(pass, "a shape was not refused");
}

// ── The fp8 GEMM, plain and 64-column ──────────────────────────────

/// `(m, n, k, col64, what)`: every fp8 `k` of the three checkpoints at a 128- or 256-row bucket,
/// the persistent loop's second pass, and the ring wrapping many times.
const GEMM_SHAPES: &[(u32, u32, u32, bool, &str)] = &[
    (128, 256, 64, false, "one tile, one K step"),
    (2048, 2560, 64, false, "160 tiles on 132 blocks"),
    (1280, 512, 256, false, "the tile order's remainder branch"),
    (256, 1024, 5120, false, "27B k 5120 (k/v)"),
    (128, 5120, 6144, false, "27B k 6144 (o_proj, out_proj)"),
    (128, 5120, 17408, false, "27B k 17408 (down): Risk A"),
    (128, 2048, 512, false, "35B k 512 (shared expert down)"),
    (128, 1024, 2048, false, "35B k 2048"),
    (128, 2048, 4096, false, "35B k 4096 (o_proj)"),
    (128, 4096, 2816, false, "Gemma k 2816"),
    (128, 2816, 4096, true, "Gemma k 4096, n 2816 (col64)"),
    (128, 2816, 8192, true, "Gemma k 8192 (col64)"),
    (128, 2816, 2112, true, "Gemma k 2112 (dense down, col64)"),
    (128, 2816, 704, true, "Gemma k 704 (col64)"),
    (128, 704, 2816, true, "Gemma n 704 (col64)"),
];

/// Run one fp8 GEMM; returns `(c, tail_intact)`.
fn run_gemm(
    device: &CudaDevice,
    (a, b): (&Fp8Buf, &Fp8Buf),
    (m, n, k): (u32, u32, u32),
    col64: bool,
) -> Result<(Vec<u16>, bool), KernelError> {
    run_out(device, m as usize * n as usize, |c| {
        // SAFETY: `a` holds m·k bytes and m f32, `b` n·k bytes and n f32, `c` m·n + TAIL bf16, all
        // cudaMalloc'd and disjoint; the download in `run_out` waits for the default stream.
        unsafe { gemm_fp8_abt(a.op(), b.op(), c, (m, n, k), col64, std::ptr::null_mut()) }
    })
}

/// Every shape against f64 on the same e4m3 inputs and scales, under the GEMM bound
/// (`out_of_bound=0`), its worst ratio printed; the tail untouched; and a repeat bit-identical.
#[test]
fn gemm_fp8_matches_the_f64_reference() {
    let device = h100();
    let mut pass = true;
    for &(m, n, k, col64, what) in GEMM_SHAPES {
        let (mu, nu, ku) = (m as usize, n as usize, k as usize);
        let (_, aq, a_s) = activation(mu, ku, 0xF8A0 ^ u64::from(k));
        let (bq, b_s) = weight(nu, ku, 0xF8B0 ^ (u64::from(n) << 20) ^ u64::from(k));
        let (a, b) = (
            Fp8Buf::new(&device, &aq, &a_s),
            Fp8Buf::new(&device, &bq, &b_s),
        );
        let (first, second) = match (
            run_gemm(&device, (&a, &b), (m, n, k), col64),
            run_gemm(&device, (&a, &b), (m, n, k), col64),
        ) {
            (Ok(x), Ok(y)) => (x, y),
            (Err(e), _) | (_, Err(e)) => {
                eprintln!("gemm_fp8 m={m} n={n} k={k}: launch failed: {e} FAIL");
                pass = false;
                continue;
            }
        };
        let refs = fp8_reference((&aq, &a_s), (&bq, &b_s), (mu, nu, ku));
        let (bad, worst) = compare_fp8(&first.0, &refs);
        let repeat = mismatched_bits(&first.0, &second.0);
        let ok = bad == 0 && first.1 && repeat == 0;
        eprintln!(
            "gemm_fp8 m={m} n={n} k={k} col64={col64}: out_of_bound={bad} worst_ratio={worst:.4} \
             tail_intact={} repeat_mismatched_bits={repeat} {} ({what})",
            first.1,
            verdict(ok)
        );
        pass &= ok;
    }
    assert!(pass, "an fp8 GEMM shape failed: see the lines above");
}

/// On an `n` both tile grids take, the 64-column GEMM is the 256-column one's bits.
#[test]
fn gemm_fp8_col64_is_the_wide_gemm_on_the_256_grid() {
    let device = h100();
    let mut pass = true;
    for (m, n, k) in [
        (128u32, 256u32, 2816u32),
        (256, 1024, 704),
        (128, 512, 17408),
    ] {
        let (_, aq, a_s) = activation(m as usize, k as usize, 0xC064 ^ u64::from(k));
        let (bq, b_s) = weight(n as usize, k as usize, 0xC065 ^ u64::from(n));
        let (a, b) = (
            Fp8Buf::new(&device, &aq, &a_s),
            Fp8Buf::new(&device, &bq, &b_s),
        );
        let wide = run_gemm(&device, (&a, &b), (m, n, k), false).expect("wide");
        let narrow = run_gemm(&device, (&a, &b), (m, n, k), true).expect("col64");
        let bad = mismatched_bits(&wide.0, &narrow.0);
        eprintln!(
            "gemm_fp8 col64 vs wide m={m} n={n} k={k}: mismatched_bits={bad} {}",
            verdict(bad == 0)
        );
        pass &= bad == 0;
    }
    assert!(pass, "the 64-column GEMM differs from the wide one");
}

#[test]
fn gemm_fp8_refuses_shapes_off_its_tile_grid() {
    let device = h100();
    let small = Fp8Buf::new(&device, &[0u8; 1 << 16], &[0.0f32; 1 << 12]);
    let mut pass = true;
    for (m, n, k, col64) in [
        (100u32, 256u32, 64u32, false),
        (128, 200, 64, false),
        (128, 64, 64, false),
        (128, 256, 32, false),
        (128, 256, 100, false),
        (128, 100, 64, true),
        (2_147_483_648, 256, 64, false),
    ] {
        let got = run_out(&device, 1024, |c| {
            // SAFETY: the launcher refuses each shape before it touches a pointer.
            unsafe {
                gemm_fp8_abt(
                    small.op(),
                    small.op(),
                    c,
                    (m, n, k),
                    col64,
                    std::ptr::null_mut(),
                )
            }
        });
        let ok = matches!(got, Err(KernelError::Shape { .. }));
        eprintln!(
            "gemm_fp8 refusal m={m} n={n} k={k} col64={col64}: {:?} {}",
            got.err(),
            verdict(ok)
        );
        pass &= ok;
    }
    assert!(pass, "a shape was not refused");
}

// ── Grouped members are their single GEMM ──────────────────────────

/// The 27B's q/k/v (12288, 1024, 1024 over k 5120) and GDN qkv/z (10240, 6144), and the 35B's
/// q/k/v (8192, 512, 512 over 2048): each member bit-identical to its own [`gemm_fp8_abt`].
#[test]
fn gemm_group_fp8_members_are_their_single_gemm() {
    let device = h100();
    let mut pass = true;
    for (m, k, ns) in [
        (128u32, 5120u32, vec![12288u32, 1024, 1024]),
        (256, 5120, vec![10240, 6144]),
        (128, 2048, vec![8192, 512, 512]),
    ] {
        let (_, aq, a_s) = activation(m as usize, k as usize, 0x6E0 ^ u64::from(k));
        let a = Fp8Buf::new(&device, &aq, &a_s);
        let ws: Vec<Fp8Buf> = (ns.iter().enumerate())
            .map(|(i, &n)| {
                let (q, s) = weight(n as usize, k as usize, 0x6E1 ^ i as u64);
                Fp8Buf::new(&device, &q, &s)
            })
            .collect();
        let outs: Vec<DeviceBuffer> = (ns.iter())
            .map(|&n| DeviceBuffer::upload(&device, &vec![SENTINEL; (m * n) as usize]).expect("c"))
            .collect();
        let members: Vec<(Fp8Operand, *mut c_void, u32)> = (ws.iter().zip(&outs).zip(&ns))
            .map(|((w, c), &n)| (w.op(), c.ptr(), n))
            .collect();
        // SAFETY: `a` holds m·k bytes and m f32, member i's weight n_i·k bytes and n_i f32 and its
        // output m·n_i bf16, all cudaMalloc'd and disjoint; the downloads wait for the stream.
        unsafe { gemm_group_fp8_abt(a.op(), &members, (m, k), std::ptr::null_mut()) }
            .expect("group launch");
        for (i, (&n, c)) in ns.iter().zip(&outs).enumerate() {
            let grouped = c.download().expect("download");
            let single = run_gemm(&device, (&a, &ws[i]), (m, n, k), false).expect("single");
            let bad = mismatched_bits(&grouped, &single.0);
            eprintln!(
                "gemm_group_fp8 m={m} k={k} member {i} n={n}: mismatched_bits={bad} {}",
                verdict(bad == 0)
            );
            pass &= bad == 0;
        }
    }
    assert!(pass, "a grouped member differs from its single GEMM");
}

// ── The gate/up GEMM ───────────────────────────────────────────────

/// SwiGLU's `g / (1 + e^-g) · u` in f64, and its bound: the output's rounding (`2⁻⁸·|ref|`), each
/// projection's f64 GEMM bound `δ = 2⁻⁷·|r| + 2⁻¹⁴·S` carried through the activation by the
/// magnitude of its partial derivatives (`|∂/∂g| ≤ 1.1·|u|`, `|∂/∂u| = |silu(g)|`), and the
/// activation's own f32 evaluation (`2⁻¹⁶·|ref|` and the row floor).
fn silu_ref(g: (f64, f64), u: (f64, f64)) -> (f64, f64) {
    let (gr, ur) = (g.0, u.0);
    let silu = gr / (1.0 + (-gr).exp());
    let (dg, du) = (
        gr.abs() / 128.0 + g.1 / 16384.0,
        ur.abs() / 128.0 + u.1 / 16384.0,
    );
    let r = silu * ur;
    let bound = r.abs() / 256.0
        + 1.1 * ur.abs() * dg
        + silu.abs() * du
        + dg * du * 1.1
        + r.abs() / 65536.0
        + 1.0 / 16_777_216.0;
    (r, bound)
}

/// GeGLU's `0.5·g·(1 + tanh(√(2/π)(g + 0.044715 g³)))·u` in f64, and its bound as `silu_ref`'s
/// (`|∂/∂g| ≤ 1.13·|u|`) plus `2⁻¹⁰·|g·u|` for the fast-math `tanhf` (`numerics::GELU_TANH`).
fn gelu_ref(g: (f64, f64), u: (f64, f64)) -> (f64, f64) {
    let (gr, ur) = (g.0, u.0);
    let c = (2.0 / std::f64::consts::PI).sqrt();
    let gelu = 0.5 * gr * (1.0 + (c * (gr + 0.044715 * gr * gr * gr)).tanh());
    let (dg, du) = (
        gr.abs() / 128.0 + g.1 / 16384.0,
        ur.abs() / 128.0 + u.1 / 16384.0,
    );
    let r = gelu * ur;
    let bound = r.abs() / 256.0
        + 1.13 * ur.abs() * dg
        + gelu.abs() * du
        + dg * du * 1.13
        + (gr * ur).abs() / 1024.0
        + 1.0 / 16_777_216.0;
    (r, bound)
}

/// The fp8 gate/up fold against f64 on the same inputs: the 27B's MLP (n 17408 over k 5120), the
/// 35B's shared expert (512 over 2048), Gemma's dense MLP (2112 over 2816, col64), each activation.
#[test]
fn gate_up_gemm_fp8_matches_the_f64_reference() {
    let device = h100();
    let mut pass = true;
    for (m, n, k, act, col64, what) in [
        (128u32, 17408u32, 5120u32, 0u32, false, "27B MLP, SwiGLU"),
        (128, 512, 2048, 0, false, "35B shared expert, SwiGLU"),
        (128, 2112, 2816, 1, true, "Gemma dense MLP, GeGLU, col64"),
        (128, 512, 2816, 1, false, "GeGLU on the 256 grid"),
    ] {
        let (mu, nu, ku) = (m as usize, n as usize, k as usize);
        let (_, aq, a_s) = activation(mu, ku, 0x6A7E ^ u64::from(k));
        let (gq, g_s) = weight(nu, ku, 0x6A7F ^ u64::from(n));
        let (uq, u_s) = weight(nu, ku, 0x6A80 ^ u64::from(n));
        let (a, gb, ub) = (
            Fp8Buf::new(&device, &aq, &a_s),
            Fp8Buf::new(&device, &gq, &g_s),
            Fp8Buf::new(&device, &uq, &u_s),
        );
        let got = run_out(&device, mu * nu, |c| {
            // SAFETY: `a` m·k bytes and m f32, `gate` and `up` n·k bytes and n f32 each, `c`
            // m·n + TAIL bf16, all cudaMalloc'd and disjoint.
            unsafe {
                gate_up_gemm_fp8(
                    (a.op(), gb.op(), ub.op()),
                    c,
                    (m, n, k),
                    (act, col64),
                    std::ptr::null_mut(),
                )
            }
        });
        let (out, tail) = match got {
            Ok(x) => x,
            Err(e) => {
                eprintln!("gate_up_gemm_fp8 m={m} n={n} k={k}: launch failed: {e} FAIL");
                pass = false;
                continue;
            }
        };
        let gr = fp8_reference((&aq, &a_s), (&gq, &g_s), (mu, nu, ku));
        let ur = fp8_reference((&aq, &a_s), (&uq, &u_s), (mu, nu, ku));
        let (mut bad, mut worst) = (0usize, 0.0f64);
        for ((&bits, &g), &u) in out.iter().zip(&gr).zip(&ur) {
            let (r, bound) = if act == 0 {
                silu_ref(g, u)
            } else {
                gelu_ref(g, u)
            };
            let v = f64::from(bf16::from_bits(bits).to_f32());
            let ratio = if v.is_finite() {
                (v - r).abs() / bound
            } else {
                f64::INFINITY
            };
            bad += usize::from(ratio.is_nan() || ratio > 1.0);
            worst = worst.max(ratio);
        }
        let ok = bad == 0 && tail;
        eprintln!(
            "gate_up_gemm_fp8 m={m} n={n} k={k} act={act} col64={col64}: out_of_bound={bad} \
             worst_ratio={worst:.4} tail_intact={tail} {} ({what})",
            verdict(ok)
        );
        pass &= ok;
    }
    assert!(pass, "an fp8 gate/up shape failed: see the lines above");
}

// ── The narrow fp8 GEMM ────────────────────────────────────────────

/// Run one narrow fp8 GEMM; returns `(c, tail_intact)`.
fn run_narrow(
    device: &CudaDevice,
    (a, b): (&Fp8Buf, &Fp8Buf),
    (m, n, k): (u32, u32, u32),
) -> Result<(Vec<u16>, bool), KernelError> {
    run_out(device, m as usize * n as usize, |c| {
        // SAFETY: as `run_gemm`'s.
        unsafe { gemm_narrow_fp8_abt(a.op(), b.op(), c, (m, n, k), std::ptr::null_mut()) }
    })
}

/// The 27B's `in_proj_a`/`b` (n 48 over k 5120) at the buckets and at a CTA's dead second
/// warpgroup (m 16 and 80), against f64, the tail untouched; and the grouped a + b launch, each
/// member bit-identical to its own narrow GEMM.
#[test]
fn gemm_narrow_fp8_matches_the_f64_reference_and_groups() {
    let device = h100();
    let mut pass = true;
    for (m, n, k) in [
        (16u32, 48u32, 5120u32),
        (80, 48, 5120),
        (512, 48, 5120),
        (128, 8, 64),
        (256, 64, 704),
    ] {
        let (mu, nu, ku) = (m as usize, n as usize, k as usize);
        let (_, aq, a_s) = activation(mu, ku, 0x7A0 ^ u64::from(m));
        let (bq, b_s) = weight(nu, ku, 0x7A1 ^ u64::from(n));
        let (a, b) = (
            Fp8Buf::new(&device, &aq, &a_s),
            Fp8Buf::new(&device, &bq, &b_s),
        );
        let got = run_narrow(&device, (&a, &b), (m, n, k));
        let (out, tail) = match got {
            Ok(x) => x,
            Err(e) => {
                eprintln!("gemm_narrow_fp8 m={m} n={n} k={k}: launch failed: {e} FAIL");
                pass = false;
                continue;
            }
        };
        let (bad, worst) =
            compare_fp8(&out, &fp8_reference((&aq, &a_s), (&bq, &b_s), (mu, nu, ku)));
        let ok = bad == 0 && tail;
        eprintln!(
            "gemm_narrow_fp8 m={m} n={n} k={k}: out_of_bound={bad} worst_ratio={worst:.4} \
             tail_intact={tail} {}",
            verdict(ok)
        );
        pass &= ok;
    }
    // The grouped `in_proj_a` + `in_proj_b`.
    let (m, k) = (512u32, 5120u32);
    let (_, aq, a_s) = activation(m as usize, k as usize, 0x7A2);
    let a = Fp8Buf::new(&device, &aq, &a_s);
    let ws: Vec<Fp8Buf> = (0..2)
        .map(|i| {
            let (q, s) = weight(48, k as usize, 0x7A3 ^ i);
            Fp8Buf::new(&device, &q, &s)
        })
        .collect();
    let outs: Vec<DeviceBuffer> = (0..2)
        .map(|_| DeviceBuffer::upload(&device, &vec![SENTINEL; (m * 48) as usize]).expect("c"))
        .collect();
    let members: Vec<(Fp8Operand, *mut c_void, u32)> = ws
        .iter()
        .zip(&outs)
        .map(|(w, c)| (w.op(), c.ptr(), 48))
        .collect();
    // SAFETY: as `gemm_group_fp8_members_are_their_single_gemm`'s.
    unsafe { gemm_narrow_group_fp8_abt(a.op(), &members, (m, k), std::ptr::null_mut()) }
        .expect("narrow group launch");
    for (i, c) in outs.iter().enumerate() {
        let single = run_narrow(&device, (&a, &ws[i]), (m, 48, k)).expect("single");
        let bad = mismatched_bits(&c.download().expect("download"), &single.0);
        eprintln!(
            "gemm_narrow_group_fp8 member {i}: mismatched_bits={bad} {}",
            verdict(bad == 0)
        );
        pass &= bad == 0;
    }
    assert!(pass, "a narrow fp8 case failed: see the lines above");
}

#[test]
fn gemm_narrow_fp8_refuses_shapes_it_does_not_take() {
    let device = h100();
    let small = Fp8Buf::new(&device, &[0u8; 1 << 16], &[0.0f32; 1 << 12]);
    let mut pass = true;
    for (m, n, k) in [
        (0u32, 48u32, 64u32),
        (8, 48, 64),
        (16, 0, 64),
        (16, 12, 64),
        (16, 72, 64),
        (16, 48, 32),
    ] {
        let got = run_out(&device, 1024, |c| {
            // SAFETY: the launcher refuses each shape before it touches a pointer.
            unsafe {
                gemm_narrow_fp8_abt(small.op(), small.op(), c, (m, n, k), std::ptr::null_mut())
            }
        });
        let ok = matches!(got, Err(KernelError::NarrowShape { .. }));
        eprintln!(
            "gemm_narrow_fp8 refusal m={m} n={n} k={k}: {:?} {}",
            got.err(),
            verdict(ok)
        );
        pass &= ok;
    }
    assert!(pass, "a shape was not refused");
}

// ── The expert GEMM is the dense fp8 GEMM per expert tile ──────────

fn up_u32(device: &CudaDevice, host: &[u32]) -> DeviceBuffer {
    let bytes: Vec<u8> = host.iter().flat_map(|w| w.to_le_bytes()).collect();
    DeviceBuffer::upload_bytes(device, &bytes).expect("upload u32")
}

/// A bank of `experts` e4m3 `[expert_rows, k]` blocks with `[experts, expert_rows]` scales, and a
/// sort's index regions over `tiles` live tiles of the `max_tiles`: tile `t` names expert
/// `owner(t)`. Each live tile's rows are bit-identical to [`gemm_fp8_abt`] (`n_tile` 256) or its
/// 64-column instance on the same 128 rows against that expert's first `n` rows and scales; the
/// dead tiles are never written. Qwen3.6's routed experts (256, gate+up 1024 over 2048, down 2048
/// over 512) and Gemma's (128, gate 704 over 2816 at 64 columns, down 2816 over 704).
#[test]
fn expert_gemm_fp8_is_the_dense_fp8_gemm_per_expert_tile() {
    let device = h100();
    let mut pass = true;
    for (experts, expert_rows, n, k, n_tile, what) in [
        (256u32, 1024u32, 512u32, 2048u32, 256u32, "qwen3.6 gate"),
        (256, 2048, 2048, 512, 256, "qwen3.6 down"),
        (128, 1408, 704, 2816, 64, "gemma gate"),
        (128, 2816, 2816, 704, 256, "gemma down"),
    ] {
        let (eu, ru, nu, ku) = (
            experts as usize,
            expert_rows as usize,
            n as usize,
            k as usize,
        );
        let (max_tiles, live) = (6usize, 4usize);
        let owner = |t: usize| [3usize, 3, 0, eu - 1][t];
        let bank_q = e4m3_weights(eu * ru * ku, 0xE8 ^ u64::from(k));
        let bank_s = weight_scales(eu * ru, 0xE9 ^ u64::from(n));
        let bank = Fp8Buf::new(&device, &bank_q, &bank_s);
        let mut offsets = vec![0u32; eu + 1];
        for t in 0..live {
            for o in &mut offsets[owner(t) + 1..] {
                *o += 128;
            }
        }
        let tile_expert: Vec<u32> = (0..max_tiles)
            .map(|t| if t < live { owner(t) as u32 } else { u32::MAX })
            .collect();
        let (ob, tb) = (up_u32(&device, &offsets), up_u32(&device, &tile_expert));
        let rows = max_tiles * 128;
        let (_, aq, a_s) = activation(rows, ku, 0xEA ^ u64::from(k));
        let a = Fp8Buf::new(&device, &aq, &a_s);
        let got = run_out(&device, rows * nu, |c| {
            // SAFETY: `a` holds rows·k bytes and rows f32, the bank E·rows_e·k bytes and E·rows_e
            // f32, the index regions E+1 and max_tiles u32 (the live tiles below E), `c`
            // rows·n + TAIL bf16, all cudaMalloc'd and disjoint.
            unsafe {
                expert_gemm_fp8(
                    (a.op(), bank.op()),
                    (ob.ptr().cast_const(), tb.ptr().cast_const()),
                    c,
                    (max_tiles as u32, n, k, experts, expert_rows),
                    n_tile,
                    std::ptr::null_mut(),
                )
            }
        });
        let (out, tail) = match got {
            Ok(x) => x,
            Err(e) => {
                eprintln!("expert_gemm_fp8 {what}: launch failed: {e} FAIL");
                pass = false;
                continue;
            }
        };
        // Tile t's rows against expert `tile_expert[t]`, as the kernel reads them.
        let mut bits_bad = 0usize;
        for t in 0..live {
            let e = owner(t);
            let a_t = Fp8Buf::new(
                &device,
                &aq[t * 128 * ku..][..128 * ku],
                &a_s[t * 128..][..128],
            );
            let w = Fp8Buf::new(
                &device,
                &bank_q[e * ru * ku..][..nu * ku],
                &bank_s[e * ru..][..nu],
            );
            let dense = run_gemm(&device, (&a_t, &w), (128, n, k), n_tile == 64).expect("dense");
            bits_bad += mismatched_bits(&out[t * 128 * nu..][..128 * nu], &dense.0);
        }
        let dead = out[live * 128 * nu..].iter().all(|&v| v == SENTINEL);
        let ok = bits_bad == 0 && dead && tail;
        eprintln!(
            "expert_gemm_fp8 {what} experts={experts} n={n} k={k} n_tile={n_tile}: \
             mismatched_bits={bits_bad} dead_tiles_untouched={dead} tail_intact={tail} {}",
            verdict(ok)
        );
        pass &= ok;
    }
    assert!(pass, "an expert tile differs from the dense fp8 GEMM");
}

// ── Timing, recorded and not asserted ──────────────────────────────

/// `cudaEvent_t`, declared here for the timing lines only (the runtime declares no event call).
type CudaEvent = *mut c_void;

unsafe extern "C" {
    fn cudaEventCreate(event: *mut CudaEvent) -> i32;
    fn cudaEventRecord(event: CudaEvent, stream: *mut c_void) -> i32;
    fn cudaEventSynchronize(event: CudaEvent) -> i32;
    fn cudaEventElapsedTime(ms: *mut f32, start: CudaEvent, end: CudaEvent) -> i32;
    fn cudaEventDestroy(event: CudaEvent) -> i32;
}

/// The median µs of 20 launches of `launch` after 3 warm-ups, each between two CUDA events on the
/// default stream.
fn event_time(mut launch: impl FnMut()) -> f64 {
    let (mut start, mut end): (CudaEvent, CudaEvent) = (std::ptr::null_mut(), std::ptr::null_mut());
    // SAFETY: plain runtime calls on the default stream with out-pointers to locals; each event is
    // destroyed below.
    unsafe {
        assert_eq!(cudaEventCreate(&mut start), 0, "cudaEventCreate");
        assert_eq!(cudaEventCreate(&mut end), 0, "cudaEventCreate");
    }
    let mut times = Vec::new();
    for i in 0..23 {
        let mut ms = 0.0f32;
        // SAFETY: as above.
        unsafe {
            assert_eq!(cudaEventRecord(start, std::ptr::null_mut()), 0, "record");
            launch();
            assert_eq!(cudaEventRecord(end, std::ptr::null_mut()), 0, "record");
            assert_eq!(cudaEventSynchronize(end), 0, "synchronize");
            assert_eq!(cudaEventElapsedTime(&mut ms, start, end), 0, "elapsed");
        }
        if i >= 3 {
            times.push(f64::from(ms) * 1e3);
        }
    }
    // SAFETY: both events were created above and are no longer used.
    unsafe {
        cudaEventDestroy(start);
        cudaEventDestroy(end);
    }
    times.sort_by(f64::total_cmp);
    times[times.len() / 2]
}

/// The big shapes' kernel time by CUDA events, as TFLOPS and as TB/s of the bytes each must move
/// (A, B and the scales once, C once): the 27B's gate/up and down at a 2048-token prefill, the
/// q/k/v group, Gemma's col64 down, and the quantization of a 2048 × 17408 activation. Recorded,
/// not asserted.
#[test]
fn the_fp8_kernel_times_are_recorded() {
    let device = h100();
    for (m, n, k, col64, what) in [
        (2048u32, 34816u32, 5120u32, false, "27B gate+up as one GEMM"),
        (2048, 5120, 17408, false, "27B down"),
        (2048, 12288, 5120, false, "27B q_proj"),
        (2048, 2816, 2112, true, "Gemma dense down (col64)"),
    ] {
        let (mu, nu, ku) = (m as usize, n as usize, k as usize);
        let (_, aq, a_s) = activation(mu, ku, 0x7130);
        let (bq, b_s) = weight(nu, ku, 0x7131);
        let (a, b) = (
            Fp8Buf::new(&device, &aq, &a_s),
            Fp8Buf::new(&device, &bq, &b_s),
        );
        let c = DeviceBuffer::upload(&device, &vec![0u16; mu * nu]).expect("c");
        let us = event_time(|| {
            // SAFETY: as `run_gemm`'s.
            unsafe {
                gemm_fp8_abt(
                    a.op(),
                    b.op(),
                    c.ptr(),
                    (m, n, k),
                    col64,
                    std::ptr::null_mut(),
                )
            }
            .expect("launch");
        });
        let flops = 2.0 * mu as f64 * nu as f64 * ku as f64;
        let bytes = (mu * ku + nu * ku + 4 * (mu + nu) + 2 * mu * nu) as f64;
        eprintln!(
            "gemm_fp8 time m={m} n={n} k={k} col64={col64}: median={us:.1}us \
             {:.1} TFLOPS {:.2} TB/s (recorded, not asserted) ({what})",
            flops / us / 1e6,
            bytes / us / 1e6
        );
    }
    let (rows, cols) = (2048usize, 17408usize);
    let x = DeviceBuffer::upload(&device, &bf16_inputs(rows * cols, 0x7132)).expect("x");
    let q = DeviceBuffer::upload_bytes(&device, &vec![0u8; rows * cols]).expect("q");
    let s = DeviceBuffer::upload_bytes(&device, &vec![0u8; rows * 4]).expect("s");
    let us = event_time(|| {
        // SAFETY: as `device_quant`'s.
        unsafe {
            quant_rows_fp8(
                x.ptr(),
                (q.ptr(), s.ptr()),
                (rows as u32, cols as u32),
                std::ptr::null_mut(),
            )
        }
        .expect("launch");
    });
    // Two reads of `x` (2 bytes an element each), one write of `q` (1) and of the scales.
    let bytes = (rows * cols * 5 + rows * 4) as f64;
    eprintln!(
        "quant_rows_fp8 time rows={rows} cols={cols}: median={us:.1}us {:.2} TB/s (two reads of \
         x, one write of q; recorded, not asserted)",
        bytes / us / 1e6
    );
}

// The host e4m3 itself, with no GPU: spot values of CUDA's satfinite conversion.
#[test]
fn the_host_e4m3_is_cudas_satfinite_conversion() {
    for (x, b) in [
        (0.0f32, 0x00u8),
        (-0.0, 0x80),
        (1.0, 0x38),
        (448.0, 0x7E),
        (500.0, 0x7E),
        (-1e9, 0xFE),
        (f32::INFINITY, 0x7E),
        (f32::NEG_INFINITY, 0xFE),
        (2f32.powi(-9), 0x01),
        (2f32.powi(-10), 0x00),
        (2f32.powi(-10) * 1.0001, 0x01),
        (1.0625, 0x38),
        (1.1875, 0x3A),
        (464.0, 0x7E),
    ] {
        assert_eq!(e4m3_satfinite(x), b, "{x}");
    }
    assert_eq!(e4m3_satfinite(f32::NAN), 0x7F);
    for b in 0u8..=0xFF {
        let v = e4m3_value(b);
        if v.is_nan() {
            continue;
        }
        assert_eq!(e4m3_satfinite(v as f32), b, "round trip of {b:#04x}");
    }
}
