// SPDX-License-Identifier: Apache-2.0
//! FP8 S4 on the H100: the decode fp8 GEMVs (`kernels/gemv_fp8.cu`) against an f64 reference on
//! the same e4m3 inputs and scales, their prologue quantization against S3's `quant_rows_fp8`,
//! and the bit identities between the instances.
//!
//! Built only under `nvcc` (`required-features`). No GPU fails rather than skips.
//!
//! The f64 bound is `tests/gemm_fp8.rs`'s, `|gpu − ref| ≤ 2⁻⁷·|ref| + 2⁻¹¹·S`, `S = Σₖ |q·w|·
//! a_scale·w_scale`. These kernels add their exact f32 products in f32, which `2⁻¹¹·S` covers
//! with room to spare; the bound is one for every fp8 kernel, so the GEMV and the GEMM are held to
//! the same one. Each worst ratio is printed.

// `pub`: each test binary uses a different part of `common`, and nothing in it is dead code.
pub mod common {
    pub mod fp8;
    pub mod numerics;
}

use std::ffi::c_void;

use common::fp8::{e4m3_value, quantize_rows};
use common::numerics::{bf16_inputs, mismatched_bits};
use half::bf16;
use scratchy_target_cuda::kernels::{
    Fp8Operand, KernelError, expert_gate_up_gemv_fp8, expert_gemv_fp8, gate_up_gemv_fp8,
    gemv_fp8_abt, gemv_fp8_quant_probe, gemv_group_fp8_abt, quant_rows_fp8,
};
use scratchy_target_cuda::runtime::{CudaDevice, DeviceBuffer};
use scratchy_target_cuda::step::{gemv_scratch_bytes, gemv_splits};

/// Every output element before a launch: a bf16 NaN, so an unwritten element fails.
const SENTINEL: u16 = 0xFFFF;
/// Extra output elements past the output, which a kernel must leave untouched.
const TAIL: usize = 4096;

fn h100() -> CudaDevice {
    CudaDevice::open()
        .unwrap_or_else(|e| panic!("the `nvcc` feature's tests need an sm_90 GPU (H100): {e}"))
}

fn verdict(pass: bool) -> &'static str {
    if pass { "PASS" } else { "FAIL" }
}

/// splitmix64 e4m3 weight bytes, every finite value: bits `0x00..=0x7E` and their negatives
/// (`tests/gemm_fp8.rs`'s).
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

/// A weight's per-channel scales in `[2⁻¹⁰, 2⁻⁸)`, bf16-exact (`tests/gemm_fp8.rs`'s).
fn weight_scales(n: usize, seed: u64) -> Vec<f32> {
    bf16_inputs(n, seed)
        .iter()
        .map(|&b| {
            let v = bf16::from_bits(b).to_f32().abs();
            bf16::from_f32((1.0 + v * 3.0) / 1024.0).to_f32()
        })
        .collect()
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn down_f32(b: &DeviceBuffer) -> Vec<f32> {
    let bytes: Vec<u8> = (b.download().expect("download").iter())
        .flat_map(|h| h.to_le_bytes())
        .collect();
    (bytes.as_chunks::<4>().0.iter())
        .map(|&c| f32::from_le_bytes(c))
        .collect()
}

/// An e4m3 weight on the device: its bytes and its f32 scales, each in its own allocation.
struct Fp8Buf {
    q: DeviceBuffer,
    s: DeviceBuffer,
}

impl Fp8Buf {
    fn new(device: &CudaDevice, q: &[u8], s: &[f32]) -> Self {
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

/// A weight `[n, k]`: e4m3 bytes and per-channel scales.
fn weight(n: usize, k: usize, seed: u64) -> (Vec<u8>, Vec<f32>) {
    (e4m3_weights(n * k, seed), weight_scales(n, seed ^ 0x5CA1E))
}

/// `(ref, S)` of `(q · Bᵀ) · w_scale[j] · a_scale` per output row `j`, in f64 from the host
/// quantizer's bytes of `x` and the weight's.
fn gemv_reference(x: &[u16], (b, b_s): (&[u8], &[f32]), n: usize, k: usize) -> Vec<(f64, f64)> {
    let (q, a_s) = quantize_rows(x, 1, k);
    let lut: Vec<f64> = (0..=255u8).map(e4m3_value).collect();
    let qa: Vec<f64> = q.iter().map(|&v| lut[v as usize]).collect();
    (0..n)
        .map(|j| {
            let (mut dot, mut abs) = (0.0, 0.0);
            for (x, &w) in qa.iter().zip(&b[j * k..][..k]) {
                let p = x * lut[w as usize];
                dot += p;
                abs += p.abs();
            }
            let sc = f64::from(a_s[0]) * f64::from(b_s[j]);
            (dot * sc, abs * sc)
        })
        .collect()
}

/// `gpu` (bf16 bits) against `refs` under the fp8 bound: the out-of-bound count and worst ratio.
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

/// A zeroed f32 scratch of `bytes` (at least 256), or a 256-byte one when `bytes` is 0.
fn scratch(device: &CudaDevice, bytes: u64) -> DeviceBuffer {
    DeviceBuffer::upload_bytes(device, &vec![0u8; (bytes as usize).max(256)]).expect("scratch")
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

/// One dense fp8 GEMV of `x` against `w` at `splits`; returns `(c, tail_intact)`.
fn run_gemv(
    device: &CudaDevice,
    x: &DeviceBuffer,
    w: &Fp8Buf,
    (n, k, splits): (u32, u32, u32),
) -> Result<(Vec<u16>, bool), KernelError> {
    let p = scratch(
        device,
        gemv_scratch_bytes(n, 1, splits).expect("scratch bytes"),
    );
    run_out(device, n as usize, |c| {
        // SAFETY: `x` holds k bf16, `w` n·k bytes and n f32, `c` n + TAIL bf16, `p` the split's
        // partials, all cudaMalloc'd and disjoint; the download in `run_out` waits for the
        // default stream.
        unsafe {
            gemv_fp8_abt(
                (x.ptr().cast_const(), w.op(), c),
                if splits > 1 {
                    p.ptr()
                } else {
                    std::ptr::null_mut()
                },
                (n, k, splits),
                std::ptr::null_mut(),
            )
        }
    })
}

/// The fp8 GEMV shapes of the three FP8 checkpoints at decode, each at its own split
/// (`step::gemv_splits`), with what the shape is.
const GEMV_SHAPES: &[(u32, u32, &str)] = &[
    (34816, 5120, "27B gate+up as one GEMV"),
    (5120, 17408, "27B down"),
    (10240, 5120, "27B in_proj_qkv"),
    (12288, 5120, "27B q_proj"),
    (1024, 5120, "27B k/v (split)"),
    (5120, 6144, "27B o_proj / out_proj"),
    (2048, 512, "35B shared down (split)"),
    (4096, 2048, "35B q_proj"),
    (512, 2048, "35B k/v (split)"),
    (2816, 2112, "Gemma dense down"),
    (2816, 704, "Gemma k 704"),
    (64, 64, "one chunk, one block, partial chunk"),
];

/// Every shape against f64 on the host quantizer's bytes under the fp8 bound (`out_of_bound=0`),
/// the tail untouched, and a repeat bit-identical.
#[test]
fn gemv_fp8_matches_the_f64_reference() {
    let device = h100();
    let mut pass = true;
    for &(n, k, what) in GEMV_SHAPES {
        let (nu, ku) = (n as usize, k as usize);
        let x = bf16_inputs(ku, 0x5A40 ^ u64::from(k));
        let (bq, b_s) = weight(nu, ku, 0x5A41 ^ (u64::from(n) << 20) ^ u64::from(k));
        let (xb, w) = (
            DeviceBuffer::upload(&device, &x).expect("x"),
            Fp8Buf::new(&device, &bq, &b_s),
        );
        let splits = gemv_splits(n, k, 1);
        let (first, second) = match (
            run_gemv(&device, &xb, &w, (n, k, splits)),
            run_gemv(&device, &xb, &w, (n, k, splits)),
        ) {
            (Ok(a), Ok(b)) => (a, b),
            (Err(e), _) | (_, Err(e)) => {
                eprintln!("gemv_fp8 n={n} k={k}: launch failed: {e} FAIL");
                pass = false;
                continue;
            }
        };
        let refs = gemv_reference(&x, (&bq, &b_s), nu, ku);
        let (bad, worst) = compare_fp8(&first.0, &refs);
        let repeat = mismatched_bits(&first.0, &second.0);
        let ok = bad == 0 && first.1 && repeat == 0;
        eprintln!(
            "gemv_fp8 n={n} k={k} splits={splits}: out_of_bound={bad} worst_ratio={worst:.4} \
             tail_intact={} repeat_mismatched_bits={repeat} {} ({what})",
            first.1,
            verdict(ok)
        );
        pass &= ok;
    }
    assert!(pass, "an fp8 GEMV is out of the f64 bound");
}

/// The prologue's quantized activation is `quant_rows_fp8`'s, bit for bit: at every fp8 `k` of the
/// three checkpoints and every split 1..=16 the rule takes, each split's values are the values of
/// `quant_rows_fp8`'s bytes of the same row (zeros past `k`) and its scale is that row's. The row
/// is `quant_rows_fp8`'s exhaustive pattern too: every bf16 value beside 448 in one row.
#[test]
fn the_prologue_quantizes_as_quant_rows_fp8() {
    let device = h100();
    let mut pass = true;
    let mut rows: Vec<(u32, Vec<u16>, &str)> = [704u32, 2048, 2112, 2816, 5120, 6144, 17408]
        .iter()
        .map(|&k| (k, bf16_inputs(k as usize, 0x9A0 ^ u64::from(k)), "random"))
        .collect();
    // Large magnitudes and the clamp: values beside 448 (0x43E0) and subnormal-range ones.
    let mut wide = bf16_inputs(5120, 0x9A1);
    for (i, v) in wide.iter_mut().enumerate() {
        *v = match i % 4 {
            0 => 0x43E0,
            1 => *v ^ 0x3000,
            2 => *v & 0x80FF,
            _ => *v,
        };
    }
    rows.push((5120, wide, "448, wide and tiny values"));
    for (k, x, what) in &rows {
        let k = *k;
        let ku = k as usize;
        let chunks = ku.div_ceil(256);
        // `quant_rows_fp8` takes 16 rows: the row, then 15 copies, all the same bytes.
        let x16 = x.repeat(16);
        let xq = DeviceBuffer::upload(&device, &x16).expect("x16");
        let q = DeviceBuffer::upload_bytes(&device, &vec![0u8; 16 * ku + 256]).expect("q");
        let s = DeviceBuffer::upload_bytes(&device, &[0u8; 16 * 4 + 256]).expect("s");
        // SAFETY: `xq` holds 16·k bf16, `q` 16·k bytes, `s` 16 f32, all disjoint.
        unsafe { quant_rows_fp8(xq.ptr(), (q.ptr(), s.ptr()), (16, k), std::ptr::null_mut()) }
            .expect("quant_rows_fp8");
        let qbytes: Vec<u8> = (q.download().expect("q").iter())
            .flat_map(|h| h.to_le_bytes())
            .take(ku)
            .collect();
        let want_scale = down_f32(&s)[0];
        let want: Vec<f32> = (0..chunks * 256)
            .map(|i| {
                if i < ku {
                    e4m3_value(qbytes[i]) as f32
                } else {
                    0.0
                }
            })
            .collect();
        let xb = DeviceBuffer::upload(&device, x).expect("x");
        let (mut bad, mut tried) = (0usize, 0u32);
        for splits in 1..=16u32 {
            let out = DeviceBuffer::upload_bytes(&device, &vec![0xFFu8; chunks * 1024 + 256])
                .expect("out");
            let sc = DeviceBuffer::upload_bytes(&device, &[0xFFu8; 16 * 4 + 256]).expect("sc");
            // SAFETY: `xb` holds k bf16, `out` chunks·256 f32, `sc` 16 f32, all disjoint.
            let r = unsafe {
                gemv_fp8_quant_probe(
                    xb.ptr(),
                    (out.ptr(), sc.ptr()),
                    (k, splits),
                    std::ptr::null_mut(),
                )
            };
            match r {
                Ok(()) => {}
                // A split the rule does not take at this `k`: nothing launched.
                Err(KernelError::GemvSplit { .. }) => continue,
                Err(e) => {
                    eprintln!("gemv_fp8 prologue k={k} splits={splits}: launch failed: {e} FAIL");
                    bad += 1;
                    continue;
                }
            }
            tried += 1;
            let got = down_f32(&out);
            let scales = down_f32(&sc);
            let mism = (got.iter().zip(&want))
                .filter(|(g, w)| g.to_bits() != w.to_bits())
                .count();
            let scale_bad = (scales[..splits as usize].iter())
                .filter(|v| v.to_bits() != want_scale.to_bits())
                .count();
            if mism + scale_bad > 0 {
                eprintln!(
                    "gemv_fp8 prologue k={k} splits={splits}: mismatched_values={mism} \
                     mismatched_scales={scale_bad} FAIL ({what})"
                );
            }
            bad += mism + scale_bad;
        }
        let ok = bad == 0 && tried > 0;
        eprintln!(
            "gemv_fp8 prologue k={k}: splits_tried={tried} mismatched={bad} scale={want_scale} \
             {} ({what})",
            verdict(ok)
        );
        pass &= ok;
    }
    assert!(pass, "the prologue's quantization is not quant_rows_fp8's");
}

/// A GEMV's bits do not depend on anything but its split: at every split the rule takes, the
/// output is within the f64 bound; the same split twice, or with the operands at other offsets, is
/// the same bits; and the step's own split (`gemv_splits`) is one of those tried.
#[test]
fn the_split_fp8_gemv_is_within_the_bound_and_repeatable() {
    let device = h100();
    let mut pass = true;
    for (n, k, what) in [
        (1024u32, 5120u32, "27B k/v"),
        (512, 2048, "35B k/v"),
        (2816, 704, "Gemma k 704"),
    ] {
        let (nu, ku) = (n as usize, k as usize);
        let x = bf16_inputs(ku, 0x5B0 ^ u64::from(k));
        let (bq, b_s) = weight(nu, ku, 0x5B1 ^ u64::from(n));
        let xb = DeviceBuffer::upload(&device, &x).expect("x");
        let w = Fp8Buf::new(&device, &bq, &b_s);
        // The same operands 512 bytes into larger allocations.
        let mut xo = vec![0u16; 256];
        xo.extend_from_slice(&x);
        let xb2 = DeviceBuffer::upload(&device, &xo).expect("x off");
        let refs = gemv_reference(&x, (&bq, &b_s), nu, ku);
        let own = gemv_splits(n, k, 1);
        for splits in 1..=16u32 {
            let first = match run_gemv(&device, &xb, &w, (n, k, splits)) {
                Ok(v) => v,
                // A split the rule does not take at this `k`: nothing launched.
                Err(KernelError::GemvSplit { .. }) => continue,
                Err(e) => {
                    eprintln!(
                        "gemv_fp8 split n={n} k={k} splits={splits}: launch failed: {e} FAIL"
                    );
                    pass = false;
                    continue;
                }
            };
            let second = run_gemv(&device, &xb, &w, (n, k, splits)).expect("repeat");
            let p = scratch(
                &device,
                gemv_scratch_bytes(n, 1, splits).expect("bytes") + 512,
            );
            let shifted = run_out(&device, nu, |c| {
                // SAFETY: as `run_gemv`'s, `xb2` and `p` 512 bytes in, still 256-byte aligned
                // with room for the operand.
                unsafe {
                    gemv_fp8_abt(
                        (xb2.ptr().wrapping_byte_add(512).cast_const(), w.op(), c),
                        p.ptr().wrapping_byte_add(512),
                        (n, k, splits),
                        std::ptr::null_mut(),
                    )
                }
            })
            .expect("shifted");
            let (bad, worst) = compare_fp8(&first.0, &refs);
            let (rep, off) = (
                mismatched_bits(&first.0, &second.0),
                mismatched_bits(&first.0, &shifted.0),
            );
            let ok = bad == 0 && rep == 0 && off == 0 && first.1;
            eprintln!(
                "gemv_fp8 split n={n} k={k} splits={splits}{}: out_of_bound={bad} \
                 worst_ratio={worst:.4} repeat={rep} offset={off} {} ({what})",
                if splits == own { " (the step's)" } else { "" },
                verdict(ok)
            );
            pass &= ok;
        }
    }
    assert!(pass, "a split fp8 GEMV is out of bound or not repeatable");
}

/// The grouped GEMV's members are their own GEMV's bits: the 27B's q/k/v (12288, 1024, 1024 over
/// 5120, the k and v split) and the 35B's (4096, 512, 512 over 2048), and a two-member group.
#[test]
fn gemv_group_fp8_is_each_member_s_own_gemv() {
    let device = h100();
    let mut pass = true;
    for (ns, k, what) in [
        (vec![12288u32, 1024, 1024], 5120u32, "27B q/k/v"),
        (vec![4096, 512, 512], 2048, "35B q/k/v"),
        (vec![10240, 5120], 5120, "27B qkv + z"),
    ] {
        let ku = k as usize;
        let x = bf16_inputs(ku, 0x6C0 ^ u64::from(k));
        let xb = DeviceBuffer::upload(&device, &x).expect("x");
        let ws: Vec<Fp8Buf> = (ns.iter().enumerate())
            .map(|(i, &n)| {
                let (q, s) = weight(n as usize, ku, 0x6C1 ^ i as u64);
                Fp8Buf::new(&device, &q, &s)
            })
            .collect();
        let splits: Vec<u32> = ns.iter().map(|&n| gemv_splits(n, k, 1)).collect();
        let outs: Vec<DeviceBuffer> = (ns.iter())
            .map(|&n| DeviceBuffer::upload(&device, &vec![SENTINEL; n as usize]).expect("c"))
            .collect();
        let parts: Vec<DeviceBuffer> = (ns.iter().zip(&splits))
            .map(|(&n, &s)| scratch(&device, gemv_scratch_bytes(n, 1, s).expect("bytes")))
            .collect();
        let members: Vec<(Fp8Operand, *mut c_void, u32)> = (ws.iter().zip(&outs).zip(&ns))
            .map(|((w, c), &n)| (w.op(), c.ptr(), n))
            .collect();
        let pp: Vec<*mut c_void> = (parts.iter().zip(&splits))
            .map(|(p, &s)| if s > 1 { p.ptr() } else { std::ptr::null_mut() })
            .collect();
        // SAFETY: every member's weight, output and partials are their own allocations; `xb`
        // holds k bf16.
        unsafe {
            gemv_group_fp8_abt(
                xb.ptr().cast_const(),
                &members,
                &pp,
                (k, &splits),
                std::ptr::null_mut(),
            )
        }
        .expect("group launch");
        let mut bad = 0;
        for (i, &n) in ns.iter().enumerate() {
            let single = run_gemv(&device, &xb, &ws[i], (n, k, splits[i])).expect("single");
            bad += mismatched_bits(&outs[i].download().expect("c"), &single.0);
        }
        let ok = bad == 0;
        eprintln!(
            "gemv_group_fp8 {what} n={ns:?} k={k} splits={splits:?}: mismatched_bits={bad} {}",
            verdict(ok)
        );
        pass &= ok;
    }
    assert!(pass, "a grouped fp8 GEMV member differs from its own GEMV");
}

/// SiLU·up / GeLU·up in f64 with `tests/gemm_fp8.rs`'s propagated bound (its `silu_ref` and
/// `gelu_ref`), from each projection's `(ref, S)`.
fn act_ref(act: u32, g: (f64, f64), u: (f64, f64)) -> (f64, f64) {
    let (gr, ur) = (g.0, u.0);
    let (dg, du) = (
        gr.abs() / 128.0 + g.1 / 2048.0,
        ur.abs() / 128.0 + u.1 / 2048.0,
    );
    if act == 0 {
        let silu = gr / (1.0 + (-gr).exp());
        let r = silu * ur;
        let bound = r.abs() / 256.0
            + 1.1 * ur.abs() * dg
            + silu.abs() * du
            + dg * du * 1.1
            + r.abs() / 65536.0
            + 1.0 / 16_777_216.0;
        (r, bound)
    } else {
        let c = (2.0 / std::f64::consts::PI).sqrt();
        let gelu = 0.5 * gr * (1.0 + (c * (gr + 0.044715 * gr * gr * gr)).tanh());
        let r = gelu * ur;
        let bound = r.abs() / 256.0
            + 1.13 * ur.abs() * dg
            + gelu.abs() * du
            + dg * du * 1.13
            + (gr * ur).abs() / 1024.0
            + 1.0 / 16_777_216.0;
        (r, bound)
    }
}

/// The fp8 gate/up fold against f64 at its split: the 27B's MLP (17408 over 5120), the 35B's
/// shared expert (512 over 2048, split), Gemma's dense MLP (2112 over 2816, GeGLU).
#[test]
fn gate_up_gemv_fp8_matches_the_f64_reference() {
    let device = h100();
    let mut pass = true;
    for (n, k, act, what) in [
        (17408u32, 5120u32, 0u32, "27B MLP, SwiGLU"),
        (512, 2048, 0, "35B shared expert, SwiGLU (split)"),
        (2112, 2816, 1, "Gemma dense MLP, GeGLU"),
    ] {
        let (nu, ku) = (n as usize, k as usize);
        let x = bf16_inputs(ku, 0x7D0 ^ u64::from(k));
        let (gq, g_s) = weight(nu, ku, 0x7D1);
        let (uq, u_s) = weight(nu, ku, 0x7D2);
        let xb = DeviceBuffer::upload(&device, &x).expect("x");
        let (gw, uw) = (
            Fp8Buf::new(&device, &gq, &g_s),
            Fp8Buf::new(&device, &uq, &u_s),
        );
        let splits = gemv_splits(n, k, 1);
        let p = scratch(
            &device,
            2 * gemv_scratch_bytes(n, 1, splits).expect("bytes"),
        );
        let got = run_out(&device, nu, |c| {
            // SAFETY: as `run_gemv`'s, for both weights and both projections' partials.
            unsafe {
                gate_up_gemv_fp8(
                    (xb.ptr().cast_const(), gw.op(), uw.op()),
                    (c, p.ptr()),
                    (n, k, splits),
                    act,
                    std::ptr::null_mut(),
                )
            }
        });
        let (out, tail) = match got {
            Ok(v) => v,
            Err(e) => {
                eprintln!("gate_up_gemv_fp8 {what}: launch failed: {e} FAIL");
                pass = false;
                continue;
            }
        };
        let gr = gemv_reference(&x, (&gq, &g_s), nu, ku);
        let ur = gemv_reference(&x, (&uq, &u_s), nu, ku);
        let (mut bad, mut worst) = (0usize, 0.0f64);
        for ((&bits, &g), &u) in out.iter().zip(&gr).zip(&ur) {
            let (r, bound) = act_ref(act, g, u);
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
            "gate_up_gemv_fp8 n={n} k={k} act={act} splits={splits}: out_of_bound={bad} \
             worst_ratio={worst:.4} tail_intact={tail} {} ({what})",
            verdict(ok)
        );
        pass &= ok;
    }
    assert!(pass, "an fp8 gate/up GEMV is out of the f64 bound");
}

fn up_u32(device: &CudaDevice, host: &[u32]) -> DeviceBuffer {
    let bytes: Vec<u8> = host.iter().flat_map(|w| w.to_le_bytes()).collect();
    DeviceBuffer::upload_bytes(device, &bytes).expect("upload u32")
}

/// The expert GEMV over an fp8 bank is the dense fp8 GEMV per pair: for the 8 picks of one token,
/// each pair's row is bit-identical to [`gemv_fp8_abt`] on its row and its expert's first `n`
/// rows and scales at the same split; with token rows (gate/up) and pair rows (down). Qwen3.6's
/// routed experts (256; gate 512 of a 1024-row block over 2048, down 2048 over 512) and Gemma's
/// (128; gate 704 of 1408 over 2816, down 2816 over 704). The expert gate/up fold is the dense
/// gate/up fold per pair too.
#[test]
fn expert_gemv_fp8_is_the_dense_fp8_gemv_per_pair() {
    let device = h100();
    let mut pass = true;
    let picks = [3u32, 0, 7, 3, 1, 5, 2, 6];
    for (experts, expert_rows, n, k, pair_rows, what) in [
        (256u32, 1024u32, 512u32, 2048u32, false, "qwen3.6 gate"),
        (256, 2048, 2048, 512, true, "qwen3.6 down"),
        (128, 1408, 704, 2816, false, "gemma gate"),
        (128, 2816, 2816, 704, true, "gemma down"),
    ] {
        let (eu, ru, nu, ku) = (
            experts as usize,
            expert_rows as usize,
            n as usize,
            k as usize,
        );
        let (m, top_k) = (1u32, picks.len() as u32);
        let pairs = (m * top_k) as usize;
        let topk: Vec<u32> = picks.iter().map(|&e| e * (experts / 8) + 1).collect();
        // Only the picked experts' blocks hold random bytes; the rest are zero, which the
        // kernel never reads.
        let mut bank_q = vec![0u8; eu * ru * ku];
        for &e in &topk {
            let e = e as usize;
            bank_q[e * ru * ku..][..ru * ku]
                .copy_from_slice(&e4m3_weights(ru * ku, 0xE80 ^ e as u64));
        }
        let bank_s = weight_scales(eu * ru, 0xE81 ^ u64::from(n));
        let bank = Fp8Buf::new(&device, &bank_q, &bank_s);
        let a_rows = if pair_rows { pairs } else { m as usize };
        let x = bf16_inputs(a_rows * ku, 0xE82 ^ u64::from(k));
        let xb = DeviceBuffer::upload(&device, &x).expect("x");
        let tb = up_u32(&device, &topk);
        let splits = gemv_splits(n, k, m * top_k);
        let p = scratch(
            &device,
            2 * gemv_scratch_bytes(n, m * top_k, splits).expect("bytes"),
        );
        let got = run_out(&device, pairs * nu, |c| {
            // SAFETY: `xb` holds a_rows·k bf16, the bank E·rows_e·k bytes and E·rows_e f32,
            // `tb` the pairs' ids (each below E), `c` pairs·n + TAIL bf16, `p` the partials.
            unsafe {
                expert_gemv_fp8(
                    (xb.ptr().cast_const(), bank.op(), tb.ptr().cast_const()),
                    (c, p.ptr()),
                    (m, top_k, n, k),
                    (experts, expert_rows),
                    (pair_rows, splits),
                    std::ptr::null_mut(),
                )
            }
        });
        let (out, tail) = match got {
            Ok(v) => v,
            Err(e) => {
                eprintln!("expert_gemv_fp8 {what}: launch failed: {e} FAIL");
                pass = false;
                continue;
            }
        };
        let mut bits_bad = 0usize;
        let mut fold_bad = 0usize;
        // The gate/up fold of the gate's slab and the up's (the next `n` rows of each block).
        let fold = (!pair_rows).then(|| {
            run_out(&device, pairs * nu, |c| {
                // SAFETY: as above; the up's slab is the gate's `n` rows on, inside each block
                // (`expert_rows ≥ 2n`), its scales likewise.
                unsafe {
                    expert_gate_up_gemv_fp8(
                        (
                            xb.ptr().cast_const(),
                            bank.op(),
                            (
                                bank.q.ptr().wrapping_byte_add(nu * ku).cast_const(),
                                bank.s.ptr().wrapping_byte_add(nu * 4).cast_const(),
                            ),
                            tb.ptr().cast_const(),
                        ),
                        (c, p.ptr()),
                        (m, top_k, n, k),
                        (experts, expert_rows, splits),
                        0,
                        std::ptr::null_mut(),
                    )
                }
            })
            .expect("expert gate/up")
        });
        // Each pair against the dense GEMV at the pair's split (the step's, of `m · top_k`).
        for (pi, &e) in topk.iter().enumerate() {
            let e = e as usize;
            let row = if pair_rows { pi } else { 0 };
            let xr = DeviceBuffer::upload(&device, &x[row * ku..][..ku]).expect("row");
            let w = Fp8Buf::new(
                &device,
                &bank_q[e * ru * ku..][..nu * ku],
                &bank_s[e * ru..][..nu],
            );
            let dense = run_gemv(&device, &xr, &w, (n, k, splits)).expect("dense");
            bits_bad += mismatched_bits(&out[pi * nu..][..nu], &dense.0);
            if let Some((ref f, _)) = fold {
                let uw = Fp8Buf::new(
                    &device,
                    &bank_q[e * ru * ku + nu * ku..][..nu * ku],
                    &bank_s[e * ru + nu..][..nu],
                );
                let pd = scratch(
                    &device,
                    2 * gemv_scratch_bytes(n, 1, splits).expect("bytes"),
                );
                let dense = run_out(&device, nu, |c| {
                    // SAFETY: as `run_gemv`'s, for both of the pair's slabs.
                    unsafe {
                        gate_up_gemv_fp8(
                            (xr.ptr().cast_const(), w.op(), uw.op()),
                            (c, pd.ptr()),
                            (n, k, splits),
                            0,
                            std::ptr::null_mut(),
                        )
                    }
                })
                .expect("dense gate/up");
                fold_bad += mismatched_bits(&f[pi * nu..][..nu], &dense.0);
            }
        }
        let ok = bits_bad == 0 && fold_bad == 0 && tail;
        eprintln!(
            "expert_gemv_fp8 {what} experts={experts} n={n} k={k} splits={splits} \
             : mismatched={bits_bad} gate_up_mismatched={fold_bad} \
             tail_intact={tail} {}",
            verdict(ok)
        );
        pass &= ok;
    }
    assert!(pass, "an fp8 expert pair differs from the dense fp8 GEMV");
}

/// Shapes the launchers refuse before touching a pointer: `k` not a multiple of 64, `k` past
/// 17,408, a zero `n`, a split the rule does not take, a null scratch above one split; and the
/// grouped launcher's member count.
#[test]
fn the_fp8_gemvs_refuse_shapes_they_do_not_take() {
    let null = std::ptr::null_mut::<c_void>();
    let w: Fp8Operand = (std::ptr::null(), std::ptr::null());
    let mut pass = true;
    for (n, k, splits, what) in [
        (64u32, 96u32, 1u32, "k not a multiple of 64"),
        (64, 17664, 1, "k past 17,408"),
        (0, 256, 1, "n zero"),
        (64, 256, 2, "more splits than chunks"),
        (64, 4096, 17, "17 splits"),
        (64, 4096, 2, "a null scratch above one split"),
    ] {
        // SAFETY: refused before any pointer is touched.
        let r = unsafe { gemv_fp8_abt((std::ptr::null(), w, null), null, (n, k, splits), null) };
        let ok = matches!(r, Err(KernelError::GemvSplit { .. }));
        eprintln!(
            "gemv_fp8 refuses n={n} k={k} splits={splits}: {r:?} {} ({what})",
            verdict(ok)
        );
        pass &= ok;
    }
    // SAFETY: refused before any pointer is touched.
    let r = unsafe {
        gemv_group_fp8_abt(
            std::ptr::null(),
            &[(w, null, 64)],
            &[null],
            (256, &[1]),
            null,
        )
    };
    let ok = matches!(r, Err(KernelError::GroupShape { .. }));
    eprintln!("gemv_group_fp8 refuses one member: {r:?} {}", verdict(ok));
    pass &= ok;
    assert!(pass, "an fp8 GEMV took a shape it must refuse");
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
    fn cudaMemsetAsync(dst: *mut c_void, value: i32, bytes: usize, stream: *mut c_void) -> i32;
}

/// `(median, min, max)` µs of 20 launches of `launch` after 3 warm-ups, each between two CUDA
/// events on the default stream; `scrub` (when given) is cleared before each, so the weight is
/// read from DRAM and not L2.
fn event_times(scrub: Option<&DeviceBuffer>, mut launch: impl FnMut()) -> (f64, f64, f64) {
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
        // SAFETY: as above; `scrub` is a live allocation of `SCRUB_BYTES`.
        unsafe {
            if let Some(s) = scrub {
                assert_eq!(
                    cudaMemsetAsync(s.ptr(), i, SCRUB_BYTES, std::ptr::null_mut()),
                    0
                );
            }
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
    (times[times.len() / 2], times[0], times[times.len() - 1])
}

/// The bytes cleared before a cold run: twice the H100's 50 MiB L2.
const SCRUB_BYTES: usize = 128 << 20;

/// The four S0 shapes' kernel time by CUDA events, cold (L2 cleared before each run) and hot, as
/// TB/s of the e4m3 weight bytes: the 27B's gate+up and down, the 35B's 8 gathered expert gate+up
/// picks and Gemma's 8 gathered expert down picks. Recorded, not asserted (the clock ramp moves a
/// single run by about 35%, so each line gives the median and the spread of 20).
#[test]
fn the_fp8_gemv_bandwidth_is_recorded() {
    let device = h100();
    let scrub = DeviceBuffer::upload_bytes(&device, &vec![0u8; SCRUB_BYTES]).expect("scrub");
    for (n, k, what) in [
        (34816u32, 5120u32, "27B gate+up"),
        (5120, 17408, "27B down"),
    ] {
        let (nu, ku) = (n as usize, k as usize);
        let x = DeviceBuffer::upload(&device, &bf16_inputs(ku, 0x8E0)).expect("x");
        let (bq, b_s) = weight(nu, ku, 0x8E1);
        let w = Fp8Buf::new(&device, &bq, &b_s);
        let c = DeviceBuffer::upload(&device, &vec![0u16; nu]).expect("c");
        let splits = gemv_splits(n, k, 1);
        let p = scratch(&device, gemv_scratch_bytes(n, 1, splits).expect("bytes"));
        for (temp, sc) in [("cold", Some(&scrub)), ("hot", None)] {
            let (med, lo, hi) = event_times(sc, || {
                // SAFETY: as `run_gemv`'s.
                unsafe {
                    gemv_fp8_abt(
                        (x.ptr().cast_const(), w.op(), c.ptr()),
                        p.ptr(),
                        (n, k, splits),
                        std::ptr::null_mut(),
                    )
                }
                .expect("launch");
            });
            eprintln!(
                "gemv_fp8 time {temp} n={n} k={k} splits={splits}: median={med:.1}us \
                 min={lo:.1}us max={hi:.1}us spread={:.0}% {:.2} TB/s (recorded, not asserted) \
                 ({what})",
                100.0 * (hi - lo) / med,
                (nu * ku) as f64 / med / 1e6
            );
        }
    }
    for (experts, expert_rows, n, k, pair_rows, what) in [
        (
            256u32,
            1024u32,
            1024u32,
            2048u32,
            false,
            "35B experts gate+up, 8 picks",
        ),
        (128, 2816, 2816, 704, true, "Gemma experts down, 8 picks"),
    ] {
        let (eu, ru, ku) = (experts as usize, expert_rows as usize, k as usize);
        let bank_q = e4m3_weights(eu * ru * ku, 0x8E2);
        let bank = Fp8Buf::new(&device, &bank_q, &weight_scales(eu * ru, 0x8E3));
        let top_k = 8u32;
        let topk: Vec<u32> = (0..top_k).map(|i| i * (experts / 8) + 3).collect();
        let tb = up_u32(&device, &topk);
        let a_rows = if pair_rows { top_k as usize } else { 1 };
        let x = DeviceBuffer::upload(&device, &bf16_inputs(a_rows * ku, 0x8E4)).expect("x");
        let c = DeviceBuffer::upload(&device, &vec![0u16; top_k as usize * n as usize]).expect("c");
        let splits = gemv_splits(n, k, top_k);
        let p = scratch(
            &device,
            gemv_scratch_bytes(n, top_k, splits).expect("bytes"),
        );
        for (temp, sc) in [("cold", Some(&scrub)), ("hot", None)] {
            let (med, lo, hi) = event_times(sc, || {
                // SAFETY: as the expert test's.
                unsafe {
                    expert_gemv_fp8(
                        (x.ptr().cast_const(), bank.op(), tb.ptr().cast_const()),
                        (c.ptr(), p.ptr()),
                        (1, top_k, n, k),
                        (experts, expert_rows),
                        (pair_rows, splits),
                        std::ptr::null_mut(),
                    )
                }
                .expect("launch");
            });
            eprintln!(
                "expert_gemv_fp8 time {temp} picks={top_k} n={n} k={k} splits={splits}: \
                 median={med:.1}us min={lo:.1}us max={hi:.1}us spread={:.0}% {:.2} TB/s \
                 (recorded, not asserted) ({what})",
                100.0 * (hi - lo) / med,
                (top_k as usize * n as usize * ku) as f64 / med / 1e6
            );
        }
    }
}
