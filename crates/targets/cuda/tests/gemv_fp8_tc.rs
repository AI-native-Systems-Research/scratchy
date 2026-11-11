// SPDX-License-Identifier: Apache-2.0
//! FP8 S8's tensor-core decode GEMV (`kernels/gemv_fp8_tc.cu`), step 1, on the H100: is it faster
//! than today's `gemv_fp8`, and is it inside S3's WGMMA bound? No tape launches it yet.
//!
//! The activation is quantized once by `quant_rows_fp8` (its 16-row tile: row 0 is the GEMV's,
//! rows 1..15 zero), then the GEMV reads row 0 and its scale. The bound is S3's,
//! `|gpu − ref| ≤ 2⁻⁷·|ref| + 2⁻¹¹·S` (`tests/gemm_fp8.rs`), against f64 from the host quantizer's
//! bytes; every member's worst ratio is printed. The timing is recorded, not asserted: today's
//! kernel then the tensor-core one, two rounds (A, B, A, B), read-flush cold and hot, as
//! `TC_TIME <temp> <shape> <impl> ... <TB/s>` of the weight bytes.
//!
//! Built only under `nvcc` (`required-features`). No GPU fails rather than skips.

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
    Fp8Operand, gate_up_gemv_fp8, gemv_fp8_abt, gemv_fp8_tc, gemv_fp8_tc_partial_bytes,
    gemv_group_fp8_abt, quant_rows_fp8,
};
use scratchy_target_cuda::runtime::{CudaDevice, DeviceBuffer};
use scratchy_target_cuda::step::{gemv_scratch_bytes, gemv_splits};

/// A NaN that no GEMV writes: the poison of every output and its tail.
const SENTINEL: u16 = 0xFFFF;
/// The elements past each output that must stay [`SENTINEL`].
const TAIL: usize = 4096;
/// `quant_rows_fp8`'s tile: the quantized activation's rows (row 0 the GEMV's).
const QROWS: usize = 16;

fn h100() -> CudaDevice {
    common::watchdog::arm();
    CudaDevice::open()
        .unwrap_or_else(|e| panic!("the `nvcc` feature's tests need an sm_90 GPU (H100): {e}"))
}

fn verdict(pass: bool) -> &'static str {
    if pass { "PASS" } else { "FAIL" }
}

/// splitmix64 e4m3 weight bytes, every finite value (`tests/gemv_fp8.rs`'s).
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

/// A weight's per-channel scales in `[2⁻¹⁰, 2⁻⁸)`, bf16-exact (`tests/gemv_fp8.rs`'s).
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

/// A buffer of `len` copies of `byte`, at least 256 bytes.
fn filled(device: &CudaDevice, len: usize, byte: u8) -> DeviceBuffer {
    DeviceBuffer::upload_bytes(device, &vec![byte; len.max(256)]).expect("upload")
}

/// An e4m3 weight on the device: its bytes and its f32 scales.
struct Fp8Buf {
    q: DeviceBuffer,
    s: DeviceBuffer,
}

impl Fp8Buf {
    fn new(device: &CudaDevice, q: &[u8], s: &[f32]) -> Self {
        let mut sb = f32_bytes(s);
        sb.resize(sb.len().next_multiple_of(256), 0);
        Self {
            q: DeviceBuffer::upload_bytes(device, q).expect("upload e4m3"),
            s: DeviceBuffer::upload_bytes(device, &sb).expect("upload scales"),
        }
    }

    /// Every weight byte 1.0 (e4m3 0x38) and every scale 0: the timing's.
    fn ones(device: &CudaDevice, n: usize, k: usize) -> Self {
        Self {
            q: filled(device, n * k, 0x38),
            s: filled(device, (n * 4).next_multiple_of(256), 0),
        }
    }

    fn op(&self) -> Fp8Operand {
        (self.q.ptr().cast_const(), self.s.ptr().cast_const())
    }
}

/// `(ref, S)` of `(q · Bᵀ) · w_scale[j] · a_scale` per output row `j`, in f64 from the host
/// quantizer's bytes of `x` (`tests/gemv_fp8.rs`'s).
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

/// `gpu` (bf16 bits) against `refs` under S3's bound: the out-of-bound count and worst ratio.
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

/// The activation `x [k]` bf16 as `quant_rows_fp8`'s 16-row input (rows 1..15 zero), and the
/// buffers it quantizes into: `(x16, q [16, k], scale [16])`.
struct Act {
    x: DeviceBuffer,
    q: DeviceBuffer,
    s: DeviceBuffer,
    k: u32,
}

impl Act {
    fn new(device: &CudaDevice, x: &[u16]) -> Self {
        let k = x.len();
        let mut x16 = x.to_vec();
        x16.resize(QROWS * k, 0);
        Self {
            x: DeviceBuffer::upload(device, &x16).expect("x"),
            q: filled(device, QROWS * k, 0),
            s: filled(device, QROWS * 4, 0),
            k: k as u32,
        }
    }

    /// The once-per-group quantization, asynchronous on the default stream.
    fn quantize(&self) {
        // SAFETY: `x` holds 16·k bf16, `q` 16·k bytes and `s` 16 f32, three allocations; null is
        // the default stream.
        unsafe {
            quant_rows_fp8(
                self.x.ptr().cast_const(),
                (self.q.ptr(), self.s.ptr()),
                (QROWS as u32, self.k),
                std::ptr::null_mut(),
            )
        }
        .expect("quant_rows_fp8 launch");
    }

    fn row(&self) -> Fp8Operand {
        (self.q.ptr().cast_const(), self.s.ptr().cast_const())
    }
}

/// The tensor-core GEMV of `act`'s quantized row against `ws`, into `outs`, asynchronous.
fn launch_tc(act: &Act, ws: &[&Fp8Buf], outs: &[*mut c_void], ns: &[u32], p: &DeviceBuffer) {
    let members: Vec<(Fp8Operand, *mut c_void, u32)> = (ws.iter().zip(outs).zip(ns))
        .map(|((w, &c), &n)| (w.op(), c, n))
        .collect();
    // SAFETY: every member's weight and output are their own allocations of their size, `p` holds
    // `gemv_fp8_tc_partial_bytes`, `act.q`/`act.s` the quantized row and its scale.
    unsafe { gemv_fp8_tc(act.row(), &members, p.ptr(), act.k, std::ptr::null_mut()) }
        .expect("gemv_fp8_tc launch");
}

/// Each member of the tensor-core GEMV against f64 under S3's bound, its tail untouched, a second
/// run the same bits; and `quant_rows_fp8`'s row 0 the host quantizer's bytes. Shapes: the 27B's
/// gate+up (as one GEMV and as the two-member group), down, GDN qkv+z group and q/k/v group, the
/// 35B's q, Gemma's down and its gate+up (2,112 rows: a partial 128-row tile), and one 64 × 64 (a
/// single unit, one CTA).
#[test]
fn gemv_fp8_tc_matches_the_f64_reference() {
    let device = h100();
    let mut pass = true;
    for (ns, k, what) in [
        (vec![34816u32], 5120u32, "27B gate+up as one"),
        (vec![17408, 17408], 5120, "27B gate+up group"),
        (vec![5120], 17408, "27B down"),
        (vec![10240, 6144], 5120, "27B GDN qkv+z group"),
        (vec![12288, 1024, 1024], 5120, "27B q/k/v group"),
        (vec![4096], 2048, "35B q"),
        (vec![2816], 2112, "Gemma down"),
        (vec![2112, 2112], 2816, "Gemma gate+up group"),
        (vec![64], 64, "one unit"),
    ] {
        let ku = k as usize;
        let x = bf16_inputs(ku, 0x7C0 ^ u64::from(k));
        let act = Act::new(&device, &x);
        act.quantize();
        let (hq, _) = quantize_rows(&x, 1, ku);
        let q_ok = act
            .q
            .download()
            .expect("q")
            .iter()
            .flat_map(|h| h.to_le_bytes())
            .take(ku)
            .eq(hq.iter().copied());
        let host: Vec<(Vec<u8>, Vec<f32>)> = (ns.iter().enumerate())
            .map(|(i, &n)| {
                let n = n as usize;
                (
                    e4m3_weights(n * ku, 0x7C1 ^ i as u64),
                    weight_scales(n, 0x7C2 ^ i as u64),
                )
            })
            .collect();
        let ws: Vec<Fp8Buf> = host
            .iter()
            .map(|(q, s)| Fp8Buf::new(&device, q, s))
            .collect();
        let wr: Vec<&Fp8Buf> = ws.iter().collect();
        let p = filled(
            &device,
            gemv_fp8_tc_partial_bytes(&ns, k).expect("partial bytes") as usize,
            0,
        );
        let run = || -> Vec<Vec<u16>> {
            let outs: Vec<DeviceBuffer> = (ns.iter())
                .map(|&n| {
                    DeviceBuffer::upload(&device, &vec![SENTINEL; n as usize + TAIL]).expect("c")
                })
                .collect();
            let op: Vec<*mut c_void> = outs.iter().map(|c| c.ptr()).collect();
            launch_tc(&act, &wr, &op, &ns, &p);
            outs.iter().map(|c| c.download().expect("c")).collect()
        };
        let (first, second) = (run(), run());
        let mut ok = q_ok && first == second;
        for (i, (&n, (b, s))) in ns.iter().zip(&host).enumerate() {
            let nu = n as usize;
            let refs = gemv_reference(&x, (b, s), nu, ku);
            let (bad, worst) = compare_fp8(&first[i][..nu], &refs);
            let tail = first[i][nu..].iter().all(|&v| v == SENTINEL);
            let m_ok = bad == 0 && tail;
            eprintln!(
                "TC_BOUND {what} member {i} n={n} k={k}: out_of_bound={bad} worst_ratio={worst:.4} \
                 tail_intact={tail} {}",
                verdict(m_ok)
            );
            ok &= m_ok;
        }
        let repeat = (first.iter().zip(&second))
            .map(|(a, b)| mismatched_bits(a, b))
            .sum::<usize>();
        eprintln!(
            "TC_BOUND {what}: q_row_is_the_host_quantizer's={q_ok} repeat_mismatched_bits={repeat} {}",
            verdict(ok)
        );
        pass &= ok;
    }
    assert!(
        pass,
        "the tensor-core fp8 GEMV is out of S3's bound, or not repeatable"
    );
}

/// `cudaEvent_t`, declared here for the timing only.
type CudaEvent = *mut c_void;

unsafe extern "C" {
    fn cudaEventCreate(event: *mut CudaEvent) -> i32;
    fn cudaEventRecord(event: CudaEvent, stream: *mut c_void) -> i32;
    fn cudaEventSynchronize(event: CudaEvent) -> i32;
    fn cudaEventElapsedTime(ms: *mut f32, start: CudaEvent, end: CudaEvent) -> i32;
    fn cudaEventDestroy(event: CudaEvent) -> i32;
}

/// `(median, min, max)` µs of 20 launches of `launch` after 3 warm-ups, each between two CUDA
/// events on the default stream, `before` run (untimed, on the same stream) before each
/// (`tests/gemv_fp8_ring_probe.rs`'s).
fn timed(mut before: impl FnMut(), mut launch: impl FnMut()) -> (f64, f64, f64) {
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
        before();
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
    (times[times.len() / 2], times[0], times[times.len() - 1])
}

/// The read flush: today's GEMV over a 100 MiB `[20480, 5120]` weight (twice the H100's L2), so L2
/// holds clean lines of other data, as in the decode tape.
struct Flush {
    x: DeviceBuffer,
    w: Fp8Buf,
    c: DeviceBuffer,
    p: DeviceBuffer,
}

const FLUSH_N: u32 = 20480;
const FLUSH_K: u32 = 5120;

impl Flush {
    fn new(device: &CudaDevice) -> Self {
        let splits = gemv_splits(FLUSH_N, FLUSH_K, 1);
        Self {
            x: DeviceBuffer::upload(device, &vec![0x3F80u16; FLUSH_K as usize]).expect("x"),
            w: Fp8Buf::ones(device, FLUSH_N as usize, FLUSH_K as usize),
            c: filled(device, FLUSH_N as usize * 2, 0),
            p: filled(
                device,
                gemv_scratch_bytes(FLUSH_N, 1, splits).expect("bytes") as usize,
                0,
            ),
        }
    }

    fn run(&self) {
        let splits = gemv_splits(FLUSH_N, FLUSH_K, 1);
        // SAFETY: `x` k bf16, `w` n·k bytes and n f32, `c` n bf16, `p` the split's partials, all
        // their own allocations; null is the default stream.
        unsafe {
            gemv_fp8_abt(
                (self.x.ptr().cast_const(), self.w.op(), self.c.ptr()),
                self.p.ptr(),
                (FLUSH_N, FLUSH_K, splits),
                std::ptr::null_mut(),
            )
        }
        .expect("read flush");
    }
}

/// One `TC_TIME` line per temperature: read-flush cold and hot.
fn record(
    flush: &Flush,
    shape: &str,
    imp: &str,
    round: u32,
    bytes: usize,
    mut launch: impl FnMut(),
) {
    let cold = timed(|| flush.run(), &mut launch);
    let hot = timed(|| {}, &mut launch);
    for (temp, (med, lo, hi)) in [("readcold", cold), ("hot", hot)] {
        eprintln!(
            "TC_TIME {temp} {shape} {imp}: median={med:.1}us min={lo:.1}us max={hi:.1}us \
             {:.2} TB/s (round {round})",
            bytes as f64 / med / 1e6
        );
    }
}

/// Today's fp8 GEMV against the tensor-core one at the 27B's gate+up (today's fused gate/up GEMV
/// and its one 34,816-row GEMV; the tensor-core GEMV as the two-member group and as one GEMV),
/// down and GDN qkv+z group, read-flush cold and hot, A, B, A, B. `tc` is the GEMV's two launches
/// on an already quantized row (`quant_rows_fp8` runs once per group); `tc+quant` adds that launch.
#[test]
fn the_tc_fp8_gemv_bandwidth_is_recorded() {
    let device = h100();
    let flush = Flush::new(&device);
    for (ns, k, shape) in [
        (vec![17408u32, 17408], 5120u32, "27B_gate+up"),
        (vec![5120], 17408, "27B_down"),
        (vec![10240, 6144], 5120, "27B_gdn_qkv+z"),
    ] {
        let ku = k as usize;
        let total: u32 = ns.iter().sum();
        let bytes = total as usize * ku;
        let act = Act::new(&device, &vec![0x3F80u16; ku]);
        act.quantize();
        let ws: Vec<Fp8Buf> = ns
            .iter()
            .map(|&n| Fp8Buf::ones(&device, n as usize, ku))
            .collect();
        let wr: Vec<&Fp8Buf> = ws.iter().collect();
        let outs: Vec<DeviceBuffer> = ns
            .iter()
            .map(|&n| filled(&device, n as usize * 2, 0))
            .collect();
        let op: Vec<*mut c_void> = outs.iter().map(|c| c.ptr()).collect();
        let p = filled(
            &device,
            gemv_fp8_tc_partial_bytes(&ns, k).expect("bytes") as usize,
            0,
        );
        // The 27B gate+up as one 34,816-row weight, for both kernels' single-GEMV form.
        let one = Fp8Buf::ones(&device, total as usize, ku);
        let one_c = filled(&device, total as usize * 2, 0);
        let one_p = filled(
            &device,
            gemv_fp8_tc_partial_bytes(&[total], k).expect("bytes") as usize,
            0,
        );
        let splits: Vec<u32> = ns.iter().map(|&n| gemv_splits(n, k, 1)).collect();
        let today_p: Vec<DeviceBuffer> = (ns.iter().zip(&splits))
            .map(|(&n, &s)| {
                filled(
                    &device,
                    2 * gemv_scratch_bytes(n, 1, s).expect("bytes") as usize,
                    0,
                )
            })
            .collect();
        let one_splits = gemv_splits(total, k, 1);
        let one_today_p = filled(
            &device,
            gemv_scratch_bytes(total, 1, one_splits).expect("bytes") as usize,
            0,
        );
        let x = &act.x;
        for round in 1..=2 {
            // A: today's kernel.
            if ns.len() == 2 && ns[0] == ns[1] {
                record(
                    &flush,
                    shape,
                    "gemv_fp8_fused_gate_up",
                    round,
                    bytes,
                    || {
                        // SAFETY: as `tests/gemv_fp8.rs`'s gate/up test: `x` k bf16 (row 0 of the
                        // 16-row input), both weights n·k bytes and n f32, `outs[0]` n bf16, the
                        // partials twice the split's.
                        unsafe {
                            gate_up_gemv_fp8(
                                (x.ptr().cast_const(), ws[0].op(), ws[1].op()),
                                (outs[0].ptr(), today_p[0].ptr()),
                                (ns[0], k, splits[0]),
                                0,
                                std::ptr::null_mut(),
                            )
                        }
                        .expect("gate_up_gemv_fp8 launch");
                    },
                );
                record(&flush, shape, "gemv_fp8_as_one", round, bytes, || {
                    // SAFETY: as `tests/gemv_fp8.rs`'s `run_gemv`, for the 34,816-row weight.
                    unsafe {
                        gemv_fp8_abt(
                            (x.ptr().cast_const(), one.op(), one_c.ptr()),
                            one_today_p.ptr(),
                            (total, k, one_splits),
                            std::ptr::null_mut(),
                        )
                    }
                    .expect("gemv_fp8_abt launch");
                });
            } else if ns.len() == 1 {
                record(&flush, shape, "gemv_fp8", round, bytes, || {
                    // SAFETY: as `tests/gemv_fp8.rs`'s `run_gemv`.
                    unsafe {
                        gemv_fp8_abt(
                            (x.ptr().cast_const(), ws[0].op(), outs[0].ptr()),
                            today_p[0].ptr(),
                            (ns[0], k, splits[0]),
                            std::ptr::null_mut(),
                        )
                    }
                    .expect("gemv_fp8_abt launch");
                });
            } else {
                let members: Vec<(Fp8Operand, *mut c_void, u32)> = (ws.iter().zip(&op).zip(&ns))
                    .map(|((w, &c), &n)| (w.op(), c, n))
                    .collect();
                let pp: Vec<*mut c_void> = today_p.iter().map(|p| p.ptr()).collect();
                record(&flush, shape, "gemv_group_fp8", round, bytes, || {
                    // SAFETY: as `tests/gemv_fp8.rs`'s group test.
                    unsafe {
                        gemv_group_fp8_abt(
                            x.ptr().cast_const(),
                            &members,
                            &pp,
                            (k, &splits),
                            std::ptr::null_mut(),
                        )
                    }
                    .expect("gemv_group_fp8_abt launch");
                });
            }
            // B: the tensor-core GEMV.
            record(&flush, shape, "tc", round, bytes, || {
                launch_tc(&act, &wr, &op, &ns, &p)
            });
            record(&flush, shape, "tc+quant", round, bytes, || {
                act.quantize();
                launch_tc(&act, &wr, &op, &ns, &p);
            });
            if ns.len() == 2 && ns[0] == ns[1] {
                record(&flush, shape, "tc_as_one", round, bytes, || {
                    launch_tc(&act, &[&one], &[one_c.ptr()], &[total], &one_p)
                });
            }
        }
    }
}

// ── The ceiling probe (/tmp/d-cap.md §3): P-B, the TC kernel's knobs, and P-D ──────────

/// The P-B instances, by the variant launcher's number and `main_only` (B5 is B1's main kernel).
const VARIANTS: &[(u32, bool, &str)] = &[
    (0, false, "B1"),
    (1, false, "B2_wait1"),
    (2, false, "B3_k128"),
    (3, false, "B4_2cta"),
    (0, true, "B5_noreduce"),
];

/// [`launch_tc`] at instance `variant`, the main kernel alone when `main_only`.
fn launch_variant(
    act: &Act,
    (ws, outs, ns): (&[&Fp8Buf], &[*mut c_void], &[u32]),
    p: &DeviceBuffer,
    (variant, main_only): (u32, bool),
) {
    use scratchy_target_cuda::kernels::gemv_fp8_tc_variant;
    let members: Vec<(Fp8Operand, *mut c_void, u32)> = (ws.iter().zip(outs).zip(ns))
        .map(|((w, &c), &n)| (w.op(), c, n))
        .collect();
    // SAFETY: as `launch_tc`'s; `p` holds `gemv_fp8_tc_partial_bytes`, which covers every instance.
    unsafe {
        gemv_fp8_tc_variant(
            act.row(),
            &members,
            p.ptr(),
            act.k,
            (variant, main_only),
            std::ptr::null_mut(),
        )
    }
    .expect("gemv_fp8_tc_variant launch");
}

/// The ceiling probe's P-B bound: B2 (`wait1`) is B1's bits, and B3 (`k128`) and B4 (`2cta`),
/// which move the K grouping and the stream-K split, are inside S3's f64 bound at the 27B's
/// gate+up (as one), down and GDN qkv+z group; every worst ratio printed as `CAP_PB_BOUND`.
#[test]
fn the_tc_variants_are_inside_s3_s_bound() {
    let device = h100();
    let mut pass = true;
    for (ns, k, what) in [
        (vec![34816u32], 5120u32, "gate+up"),
        (vec![5120], 17408, "down"),
        (vec![10240, 6144], 5120, "gdn_qkv+z"),
    ] {
        let ku = k as usize;
        let x = bf16_inputs(ku, 0x7D0 ^ u64::from(k));
        let act = Act::new(&device, &x);
        act.quantize();
        let host: Vec<(Vec<u8>, Vec<f32>)> = (ns.iter().enumerate())
            .map(|(i, &n)| {
                let n = n as usize;
                (
                    e4m3_weights(n * ku, 0x7D1 ^ i as u64),
                    weight_scales(n, 0x7D2 ^ i as u64),
                )
            })
            .collect();
        let ws: Vec<Fp8Buf> = host
            .iter()
            .map(|(q, s)| Fp8Buf::new(&device, q, s))
            .collect();
        let wr: Vec<&Fp8Buf> = ws.iter().collect();
        let refs: Vec<Vec<(f64, f64)>> = (ns.iter().zip(&host))
            .map(|(&n, (b, s))| gemv_reference(&x, (b, s), n as usize, ku))
            .collect();
        let p = filled(
            &device,
            gemv_fp8_tc_partial_bytes(&ns, k).expect("bytes") as usize,
            0,
        );
        let run = |variant: u32| -> Vec<Vec<u16>> {
            let outs: Vec<DeviceBuffer> = (ns.iter())
                .map(|&n| DeviceBuffer::upload(&device, &vec![SENTINEL; n as usize]).expect("c"))
                .collect();
            let op: Vec<*mut c_void> = outs.iter().map(|c| c.ptr()).collect();
            launch_variant(&act, (&wr, &op, &ns), &p, (variant, false));
            outs.iter().map(|c| c.download().expect("c")).collect()
        };
        let b1 = run(0);
        for (variant, name) in [(1u32, "B2_wait1"), (2, "B3_k128"), (3, "B4_2cta")] {
            let got = run(variant);
            for (i, (g, r)) in got.iter().zip(&refs).enumerate() {
                let (bad, worst) = compare_fp8(g, r);
                let same = mismatched_bits(g, &b1[i]);
                let ok = bad == 0 && (variant != 1 || same == 0);
                eprintln!(
                    "CAP_PB_BOUND {name} {what} member {i} n={} k={k}: out_of_bound={bad} \
                     worst_ratio={worst:.4} mismatched_bits_vs_B1={same} {}",
                    ns[i],
                    verdict(ok)
                );
                pass &= ok;
            }
        }
    }
    assert!(
        pass,
        "a ceiling-probe TC instance is out of S3's bound, or B2 is not B1's bits"
    );
}

/// The ceiling probe's P-B timing: B1 to B5 at the 27B's gate+up (as one), down and GDN qkv+z
/// group, read-flush cold and hot, two rounds (each instance in turn, so A, B, A, B), as `TC_TIME`
/// lines (`record`'s) with the instance as the implementation. B5 times the main kernel only.
#[test]
fn the_tc_variants_bandwidth_is_recorded() {
    let device = h100();
    let flush = Flush::new(&device);
    for (ns, k, shape) in [
        (vec![34816u32], 5120u32, "27B_gate+up"),
        (vec![5120], 17408, "27B_down"),
        (vec![10240, 6144], 5120, "27B_gdn_qkv+z"),
    ] {
        let ku = k as usize;
        let bytes = ns.iter().sum::<u32>() as usize * ku;
        let act = Act::new(&device, &vec![0x3F80u16; ku]);
        act.quantize();
        let ws: Vec<Fp8Buf> = ns
            .iter()
            .map(|&n| Fp8Buf::ones(&device, n as usize, ku))
            .collect();
        let wr: Vec<&Fp8Buf> = ws.iter().collect();
        let outs: Vec<DeviceBuffer> = ns
            .iter()
            .map(|&n| filled(&device, n as usize * 2, 0))
            .collect();
        let op: Vec<*mut c_void> = outs.iter().map(|c| c.ptr()).collect();
        let p = filled(
            &device,
            gemv_fp8_tc_partial_bytes(&ns, k).expect("bytes") as usize,
            0,
        );
        for round in 1..=2 {
            for &(variant, main_only, name) in VARIANTS {
                record(&flush, shape, name, round, bytes, || {
                    launch_variant(&act, (&wr, &op, &ns), &p, (variant, main_only))
                });
            }
        }
    }
}

/// The ceiling probe's P-D: `quant_rows_fp8` alone on its 16-row input (row 0 the activation) at
/// the 27B's 5,120 and 17,408 columns, CUDA events around its launch only, read-flush cold and
/// hot, two rounds, as `TC_TIME` lines (the TB/s is of the 16-row bf16 input).
#[test]
fn quant_rows_fp8_one_row_times_are_recorded() {
    let device = h100();
    let flush = Flush::new(&device);
    for k in [5120u32, 17408] {
        let ku = k as usize;
        let act = Act::new(&device, &bf16_inputs(ku, 0x7E0 ^ u64::from(k)));
        for round in 1..=2 {
            record(
                &flush,
                &format!("quant_16x{k}"),
                "quant_rows_fp8",
                round,
                QROWS * ku * 2,
                || act.quantize(),
            );
            // Ten launches back to back between the events: each launch's host work (its tensor
            // maps) overlaps the one before on the GPU, so a gap from the single line is the
            // harness's, not the kernel's.
            record(
                &flush,
                &format!("quant_16x{k}"),
                "quant_rows_fp8_x10",
                round,
                10 * QROWS * ku * 2,
                || (0..10).for_each(|_| act.quantize()),
            );
        }
    }
}
