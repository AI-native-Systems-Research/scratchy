// SPDX-License-Identifier: Apache-2.0
//! FP8 S3 on the H100: the prefill fp8 GEMMs (`kernels/gemm_fp8.cu`, `kernels/gemm_narrow_fp8.cu`)
//! against an f64 reference on the same e4m3 inputs and scales. The per-token quantization's tests
//! are `tests/quant_rows_fp8.rs`.
//!
//! Built only under `nvcc` (`required-features`). No GPU fails rather than skips.
//!
//! The f64 bound is `|gpu − ref| ≤ 2⁻⁷·|ref| + 2⁻¹¹·S`, `S = Σₖ |a·w|·a_scale·w_scale`: one bf16
//! rounding of the output, and the fp8 tensor core's accumulation over `k`. The bf16 GEMMs'
//! `2⁻¹⁴·S` (`common::numerics`) assumes f32 accumulation; Hopper's fp8 WGMMA does not do that
//! (FP8 plan Risk A). Per k32 add it aligns the addends to the largest, `M`, and keeps about 13
//! bits below its leading bit, so each addend loses under `2⁻¹³·M ≤ 2⁻¹³·S_step`, truncated, its
//! sign the term's; summed over the steps that is a few `2⁻¹³·S` at most, which `2⁻¹¹·S`
//! (4 × 2⁻¹³) covers. On the H100 (user decision 2026-11-06): the single-dot probe in
//! `the_fp8_wgmma_accumulation_is_recorded` keeps a 2⁻⁵ addend beside 2⁸ and drops it at 2⁻⁶ (13
//! bits), and the same ~20 elements of the k = 64 tile fail `2⁻¹⁴·S` with every scale 1, at
//! 0.47–0.48 of `2⁻¹¹·S`. A dropped or repeated 64-wide K step at k = 17408 still moves an
//! element by about `2⁻⁸·S`, 8× this term. Every worst ratio is printed, at every fp8 `k` of the
//! three FP8 checkpoints, k = 17408 (the 27B's down projection) included.

// `pub`: each test binary uses a different part of `common`, and nothing in it is dead code.
pub mod common {
    pub mod fp8;
    pub mod numerics;
    pub mod watchdog;
}

use std::ffi::c_void;

use common::fp8::{e4m3_value, quantize_rows};
use common::numerics::{bf16_inputs, mismatched_bits};
use half::bf16;
use scratchy_target_cuda::kernels::{
    Fp8Operand, KernelError, expert_gemm_fp8, expert_gemm_fp8_variant, gate_up_gemm_fp8,
    gemm_fp8_abt, gemm_group_fp8_abt, gemm_narrow_fp8_abt, gemm_narrow_group_fp8_abt,
    quant_rows_fp8,
};
use scratchy_target_cuda::runtime::{CudaDevice, DeviceBuffer};

/// Every output element before a launch: a bf16 NaN, so an unwritten element fails.
const SENTINEL: u16 = 0xFFFF;
/// Extra output elements past `m·n`, which a kernel must leave untouched.
const TAIL: usize = 4096;

fn h100() -> CudaDevice {
    common::watchdog::arm();
    CudaDevice::open()
        .unwrap_or_else(|e| panic!("the `nvcc` feature's tests need an sm_90 GPU (H100): {e}"))
}

fn verdict(pass: bool) -> &'static str {
    if pass { "PASS" } else { "FAIL" }
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

/// `gpu` (`[m, n]` bf16 bits) against `refs` under the fp8 GEMM bound `2⁻⁷·|ref| + 2⁻¹¹·S`: the
/// out-of-bound count and the worst ratio. NaN or Inf fails.
fn compare_fp8(gpu: &[u16], refs: &[(f64, f64)]) -> (usize, f64) {
    let (mut bad, mut worst) = (0usize, 0.0f64);
    for (&bits, &(r, s)) in gpu.iter().zip(refs) {
        let g = f64::from(bf16::from_bits(bits).to_f32());
        let bound = r.abs() / 128.0 + s / 2048.0;
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
/// projection's fp8 GEMM bound `δ = 2⁻⁷·|r| + 2⁻¹¹·S` carried through the activation by the
/// magnitude of its partial derivatives (`|∂/∂g| ≤ 1.1·|u|`, `|∂/∂u| = |silu(g)|`), and the
/// activation's own f32 evaluation (`2⁻¹⁶·|ref|` and the row floor).
fn silu_ref(g: (f64, f64), u: (f64, f64)) -> (f64, f64) {
    let (gr, ur) = (g.0, u.0);
    let silu = gr / (1.0 + (-gr).exp());
    let (dg, du) = (
        gr.abs() / 128.0 + g.1 / 2048.0,
        ur.abs() / 128.0 + u.1 / 2048.0,
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
        gr.abs() / 128.0 + g.1 / 2048.0,
        ur.abs() / 128.0 + u.1 / 2048.0,
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

// ── The fp8 WGMMA's accumulation, recorded and not asserted ────────

/// One tile, one K step (128 × 256 × 64), under four scale cases: all scales 1, `a_scale` 1 with
/// the weight's scales, `w_scale` 1 with the activation's, and both as the GEMM test's. A wrong
/// scale index fails only the cases whose scales vary; the tensor core's own accumulation fails
/// all four alike. Each line gives the out-of-bound count under the GEMM bound, the worst element
/// with its scales, and its ratio against `2⁻⁷·|ref| + 2⁻¹¹·S` too (a bound for the tensor core's
/// own accumulation, not this file's). Then a single-dot probe, all scales 1: row `[16, 16, 1 × 30, 0 …]` against column
/// `[16, −16, 2⁻ˢ × 30, 0 …]`, exact sum `30·2⁻ˢ` after the two ±256 products cancel, for
/// s = 1..9: the first s whose device value falls short is where the tensor core stops keeping
/// bits below 256. Recorded, not asserted.
#[test]
fn the_fp8_wgmma_accumulation_is_recorded() {
    let device = h100();
    let (m, n, k) = (128usize, 256usize, 64usize);
    let (_, aq, a_rand) = activation(m, k, 0xF8A0 ^ k as u64);
    let (bq, b_rand) = weight(n, k, 0xF8B0 ^ ((n as u64) << 20) ^ k as u64);
    let (a_one, b_one) = (vec![1.0f32; m], vec![1.0f32; n]);
    for (what, a_s, b_s) in [
        ("all scales 1", &a_one, &b_one),
        ("a_scale 1, w_scale random", &a_one, &b_rand),
        ("w_scale 1, a_scale random", &a_rand, &b_one),
        ("both random", &a_rand, &b_rand),
    ] {
        let (a, b) = (
            Fp8Buf::new(&device, &aq, a_s),
            Fp8Buf::new(&device, &bq, b_s),
        );
        let (out, _) =
            run_gemm(&device, (&a, &b), (m as u32, n as u32, k as u32), false).expect("launch");
        let refs = fp8_reference((&aq, a_s), (&bq, b_s), (m, n, k));
        let (bad, _) = compare_fp8(&out, &refs);
        let (mut worst, mut at, mut wide) = (0.0f64, 0usize, 0.0f64);
        for (e, (&bits, &(r, sum))) in out.iter().zip(&refs).enumerate() {
            let err = (f64::from(bf16::from_bits(bits).to_f32()) - r).abs();
            let ratio = err / (r.abs() / 128.0 + sum / 16384.0).max(f64::MIN_POSITIVE);
            if ratio > worst {
                (worst, at) = (ratio, e);
            }
            wide = wide.max(err / (r.abs() / 128.0 + sum / 2048.0).max(f64::MIN_POSITIVE));
        }
        let (i, j) = (at / n, at % n);
        eprintln!(
            "fp8 wgmma {what}: out_of_bound={bad} worst=({i},{j}) gpu={} ref={} a_scale={} \
             w_scale={} ratio={worst:.4} ratio_at_2^-11_S={wide:.4} (recorded, not asserted)",
            bf16::from_bits(out[at]).to_f32(),
            refs[at].0,
            a_s[i],
            b_s[j],
        );
    }
    // e4m3 16 is 0x58, 1 is 0x38, −16 is 0xD8, and 2⁻ˢ is 0x38 − 8s (normal) or the subnormal
    // 2⁻ˢ = 2⁻⁹ · 2^(9 − s), bits 1 << (9 − s), for s ≥ 7.
    let small = |s: usize| {
        if s < 7 {
            0x38 - 8 * s as u8
        } else {
            1u8 << (9 - s)
        }
    };
    let mut a_row = vec![0u8; k];
    a_row[..2].fill(0x58);
    a_row[2..32].fill(0x38);
    let aq: Vec<u8> = a_row.repeat(m);
    let bq: Vec<u8> = (0..n)
        .flat_map(|j| {
            let mut col = vec![0u8; k];
            (col[0], col[1]) = (0x58, 0xD8);
            col[2..32].fill(small(1 + j % 9));
            col
        })
        .collect();
    let (a, b) = (
        Fp8Buf::new(&device, &aq, &a_one),
        Fp8Buf::new(&device, &bq, &b_one),
    );
    let (out, _) =
        run_gemm(&device, (&a, &b), (m as u32, n as u32, k as u32), false).expect("launch");
    for s in 1..=9 {
        eprintln!(
            "fp8 wgmma probe s={s}: exact={} gpu={} (recorded, not asserted)",
            30.0 * 2f64.powi(-(s as i32)),
            bf16::from_bits(out[s - 1]).to_f32(),
        );
    }
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

// ── FP8 S8 P3(a): the expert GEMM at 64-row tiles and K step 128 ────────

/// The routed experts' four GEMMs (`expert_gemm_fp8_is_the_dense_fp8_gemm_per_expert_tile`'s):
/// `(experts, expert_rows, n, k, n_tile, what)`.
const P3_SHAPES: [(u32, u32, u32, u32, u32, &str); 4] = [
    (256, 1024, 512, 2048, 256, "qwen3.6 gate"),
    (256, 2048, 2048, 512, 256, "qwen3.6 down"),
    (128, 1408, 704, 2816, 64, "gemma gate"),
    (128, 2816, 2816, 704, 256, "gemma down"),
];

/// A sort's `offsets` (`[E + 1]`, its entry `E` the live rows) and `tile_expert` (`[max_tiles]`)
/// for live tiles of `tile` rows owned by `owners`, in order.
fn p3_index(
    owners: &[usize],
    max_tiles: usize,
    tile: usize,
    experts: usize,
) -> (Vec<u32>, Vec<u32>) {
    let mut offsets = vec![0u32; experts + 1];
    for &e in owners {
        for o in &mut offsets[e + 1..] {
            *o += tile as u32;
        }
    }
    let mut tile_expert = vec![u32::MAX; max_tiles];
    for (t, &e) in owners.iter().enumerate() {
        tile_expert[t] = e as u32;
    }
    (offsets, tile_expert)
}

/// One expert GEMM at `(m_tile, k_step)` over `max_tiles` tiles of `a`, into a fresh output.
fn p3_run(
    device: &CudaDevice,
    (a, bank): (&Fp8Buf, &Fp8Buf),
    (offsets, tile_expert): (&[u32], &[u32]),
    (n, k, experts, expert_rows, n_tile): (u32, u32, u32, u32, u32),
    (m_tile, k_step): (u32, u32),
) -> (Vec<u16>, bool) {
    let (ob, tb) = (up_u32(device, offsets), up_u32(device, tile_expert));
    let max_tiles = tile_expert.len() as u32;
    let rows = max_tiles as usize * m_tile as usize;
    run_out(device, rows * n as usize, |c| {
        // SAFETY: `a` holds at least rows·k bytes and rows f32, the bank E·rows_e·k bytes and
        // E·rows_e f32, the index regions E+1 and max_tiles u32 (the live tiles below E), `c`
        // rows·n + TAIL bf16, all cudaMalloc'd and disjoint; null is the default stream.
        unsafe {
            expert_gemm_fp8_variant(
                (a.op(), bank.op()),
                (ob.ptr().cast_const(), tb.ptr().cast_const()),
                c,
                (max_tiles, n, k, experts, expert_rows),
                (n_tile, m_tile, k_step),
                std::ptr::null_mut(),
            )
        }
    })
    .expect("an expert GEMM variant")
}

/// FP8 S8 P3(a): the expert GEMM at 64-row tiles, K step 64 and (where `k` is a multiple of 128)
/// 128, is today's (128-row tiles, K step 64), bit for bit on every live row: on today's 128-row
/// layout with each tile named twice as two 64-row tiles, and on a 64-row layout of 5 live tiles
/// against the dense fp8 GEMM of each tile's rows. Dead tiles are never written. At the four routed
/// GEMMs of the 35B and Gemma, as `P3_BITS` lines.
#[test]
fn the_expert_gemm_at_64_row_tiles_and_k_128_is_todays_bit_for_bit() {
    let device = h100();
    let mut pass = true;
    for (experts, expert_rows, n, k, n_tile, what) in P3_SHAPES {
        let (eu, ru, nu, ku) = (
            experts as usize,
            expert_rows as usize,
            n as usize,
            k as usize,
        );
        let bank_q = e4m3_weights(eu * ru * ku, 0x0F60 ^ u64::from(k));
        let bank_s = weight_scales(eu * ru, 0x0F61 ^ u64::from(n));
        let bank = Fp8Buf::new(&device, &bank_q, &bank_s);
        let rows = 13 * 64;
        let (_, aq, a_s) = activation(rows, ku, 0x0F62 ^ u64::from(k));
        let a = Fp8Buf::new(&device, &aq, &a_s);
        let dims = (n, k, experts, expert_rows, n_tile);
        // Today's: 4 live 128-row tiles of 6.
        let owners = [3usize, 3, 0, eu - 1];
        let (o128, t128) = p3_index(&owners, 6, 128, eu);
        let (today, _) = p3_run(&device, (&a, &bank), (&o128, &t128), dims, (128, 64));
        let live = 4 * 128 * nu;
        let k_steps: &[u32] = if k % 128 == 0 { &[64, 128] } else { &[64] };
        for &k_step in k_steps {
            // The same rows as 8 live 64-row tiles of 12.
            let owners64: Vec<usize> = owners.iter().flat_map(|&e| [e, e]).collect();
            let (o64, t64) = p3_index(&owners64, 12, 64, eu);
            let (got, tail) = p3_run(&device, (&a, &bank), (&o64, &t64), dims, (64, k_step));
            let bad = mismatched_bits(&got[..live], &today[..live]);
            let dead = got[live..].iter().all(|&v| v == SENTINEL);
            // A 64-row layout: 5 live tiles, each against the dense GEMM of its 64 rows (and the
            // next 64, the dense GEMM's 128-row tile).
            let owners5 = [3usize, 3, 3, 0, eu - 1];
            let (o5, t5) = p3_index(&owners5, 12, 64, eu);
            let (got5, tail5) = p3_run(&device, (&a, &bank), (&o5, &t5), dims, (64, k_step));
            let mut bad5 = 0usize;
            for (t, &e) in owners5.iter().enumerate() {
                let a_t = Fp8Buf::new(
                    &device,
                    &aq[t * 64 * ku..][..128 * ku],
                    &a_s[t * 64..][..128],
                );
                let w = Fp8Buf::new(
                    &device,
                    &bank_q[e * ru * ku..][..nu * ku],
                    &bank_s[e * ru..][..nu],
                );
                let dense =
                    run_gemm(&device, (&a_t, &w), (128, n, k), n_tile == 64).expect("dense");
                bad5 += mismatched_bits(&got5[t * 64 * nu..][..64 * nu], &dense.0[..64 * nu]);
            }
            let dead5 = got5[5 * 64 * nu..].iter().all(|&v| v == SENTINEL);
            let ok = bad == 0 && dead && tail && bad5 == 0 && dead5 && tail5;
            eprintln!(
                "P3_BITS {what} experts={experts} n={n} k={k} m_tile=64 k_step={k_step}: \
                 vs_today mismatched_bits={bad} 64_row_layout mismatched_bits={bad5} \
                 dead_tiles_untouched={} tails_intact={} {}",
                dead && dead5,
                tail && tail5,
                verdict(ok)
            );
            pass &= ok;
        }
    }
    assert!(pass, "an expert GEMM form is not today's bits");
}

/// Recorded, not asserted (P3(a)'s gate reads it): the routed GEMMs of a 2,048-token prefill
/// (16,384 pairs, uniform over the experts), today's (128-row tiles, K step 64) against 64-row
/// tiles at K step 64 and 128, each on its own sort padding, as `P3_TIME` lines, two rounds (A, B,
/// A, B), with the computed rows of each.
#[test]
fn the_p3_expert_gemm_times_are_recorded() {
    let device = h100();
    for (experts, expert_rows, n, k, n_tile, what) in P3_SHAPES {
        let (eu, ru, ku) = (experts as usize, expert_rows as usize, k as usize);
        // 16,384 picks over the experts (a splitmix64 stream), counted per expert.
        let mut state = 0x0F70u64 ^ u64::from(k);
        let mut counts = vec![0usize; eu];
        for _ in 0..2048 * 8 {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            counts[((z ^ (z >> 31)) % experts as u64) as usize] += 1;
        }
        let owners = |tile: usize| -> Vec<usize> {
            (0..eu)
                .flat_map(|e| std::iter::repeat_n(e, counts[e].div_ceil(tile)))
                .collect()
        };
        let (own128, own64) = (owners(128), owners(64));
        let rows = (own64.len() * 64).max(own128.len() * 128);
        let bank = Fp8Buf::new(
            &device,
            &e4m3_weights(eu * ru * ku, 0x0F71),
            &weight_scales(eu * ru, 0x0F72),
        );
        let a = Fp8Buf::new(
            &device,
            &e4m3_weights(rows * ku, 0x0F73),
            &weight_scales(rows, 0x0F74),
        );
        let k_steps: &[u32] = if k % 128 == 0 { &[64, 128] } else { &[64] };
        let mut forms: Vec<(u32, u32, &Vec<usize>)> = vec![(128, 64, &own128)];
        forms.extend(k_steps.iter().map(|&ks| (64, ks, &own64)));
        for round in 1..=2 {
            for &(m_tile, k_step, own) in &forms {
                let (offsets, tile_expert) = p3_index(own, own.len(), m_tile as usize, eu);
                let (ob, tb) = (up_u32(&device, &offsets), up_u32(&device, &tile_expert));
                let c = DeviceBuffer::upload(
                    &device,
                    &vec![0u16; own.len() * m_tile as usize * n as usize],
                )
                .expect("C");
                let us = event_time(|| {
                    // SAFETY: as `p3_run`'s.
                    unsafe {
                        expert_gemm_fp8_variant(
                            (a.op(), bank.op()),
                            (ob.ptr().cast_const(), tb.ptr().cast_const()),
                            c.ptr(),
                            (own.len() as u32, n, k, experts, expert_rows),
                            (n_tile, m_tile, k_step),
                            std::ptr::null_mut(),
                        )
                    }
                    .expect("an expert GEMM variant")
                });
                eprintln!(
                    "P3_TIME {what} round={round} m_tile={m_tile} k_step={k_step}: {us:.1} us \
                     ({} computed rows for 16384 pairs)",
                    own.len() * m_tile as usize
                );
            }
        }
    }
}
