// SPDX-License-Identifier: Apache-2.0
//! FP8 S3 on the H100: the per-token quantization (`kernels/quant_rows_fp8.cu`) against the host
//! quantizer bit for bit, and the host e4m3 encoder itself; FP8 S8's F3 (the wide form) and Q1
//! (the one-row form) against it.
//!
//! Built only under `nvcc` (`required-features`). No GPU fails rather than skips. A binary of its
//! own, so a fault in an fp8 GEMM's test (`tests/gemm_fp8.rs`) cannot leave these without a
//! verdict: a CUDA fault poisons the context for every later test in its process.

// `pub`: each test binary uses a different part of `common`, and nothing in it is dead code.
pub mod common {
    pub mod fp8;
    pub mod numerics;
    pub mod watchdog;
}

use common::fp8::{E4M3_MAX, e4m3_satfinite, e4m3_value, quantize_rows};
use common::numerics::bf16_inputs;
use half::bf16;
use scratchy_target_cuda::kernels::{KernelError, quant_rows_fp8};
use scratchy_target_cuda::runtime::{CudaDevice, DeviceBuffer};

fn h100() -> CudaDevice {
    common::watchdog::arm();
    CudaDevice::open()
        .unwrap_or_else(|e| panic!("the `nvcc` feature's tests need an sm_90 GPU (H100): {e}"))
}

fn verdict(pass: bool) -> &'static str {
    if pass { "PASS" } else { "FAIL" }
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

// ── FP8 S8 F3: eight warps a 16-row block ────────────────────────────────────────

/// `quant_rows_fp8_variant` on `x [rows, cols]`: `(q, scale)` as bytes.
fn variant_quant(
    device: &CudaDevice,
    x: &DeviceBuffer,
    (rows, cols): (usize, usize),
    variant: u32,
) -> (Vec<u8>, Vec<u8>) {
    use scratchy_target_cuda::kernels::quant_rows_fp8_variant;
    let q = DeviceBuffer::upload_bytes(device, &vec![0xA5; rows * cols + 256]).expect("q");
    let s = DeviceBuffer::upload_bytes(device, &vec![0xA5; rows * 4 + 256]).expect("scale");
    // SAFETY: `x` holds rows·cols bf16, `q` rows·cols bytes and `s` rows f32, all their own
    // allocations; the downloads wait for the default stream.
    unsafe {
        quant_rows_fp8_variant(
            x.ptr(),
            (q.ptr(), s.ptr()),
            (rows as u32, cols as u32),
            variant,
            std::ptr::null_mut(),
        )
    }
    .expect("quant_rows_fp8_variant launch");
    (down_bytes(&q), down_bytes(&s))
}

/// The prefill shapes: 512 and 2048 rows × the three models' fp8 `k`s (2,048, 2,816, 5,120,
/// 17,408), and the expert inputs' pair rows (512 or 2,048 tokens × 8 picks) × 512 and 704.
const F3_SHAPES: &[(usize, usize)] = &[
    (512, 2048),
    (512, 2816),
    (512, 5120),
    (512, 17408),
    (2048, 2048),
    (2048, 2816),
    (2048, 5120),
    (2048, 17408),
    (4096, 512),
    (4096, 704),
    (16384, 512),
    (16384, 704),
];

/// `rows · cols` bf16 of one kind: `random`; `padded` (the first quarter of the rows random, the
/// rest zero, as a 128-token prompt in a 512-row bucket); `edge` (random, with signed zeros, bf16
/// subnormals and one large value a row).
fn f3_input(kind: &str, rows: usize, cols: usize) -> Vec<u16> {
    let mut x = bf16_inputs(rows * cols, 0xF30 ^ (rows * cols) as u64);
    match kind {
        "padded" => x[(rows / 4) * cols..].fill(0),
        "edge" => {
            for (i, v) in x.iter_mut().enumerate() {
                *v = match i % 11 {
                    0 => 0x8000,
                    1 => 0x0000,
                    2 => 0x0001,
                    3 => 0x8001,
                    4 if i % cols == 4 => 0x7F00,
                    _ => *v,
                };
            }
        }
        _ => {}
    }
    x
}

/// F3's kernel (variant 1, the launcher's) is the FP8 S3 kernel's (variant 0) bit for bit, bytes
/// and scales, at every prefill shape, on random, padded and edge rows, as `F3_BITS` lines.
#[test]
fn the_wide_quantize_is_the_s3_quantize_bit_for_bit() {
    let device = h100();
    let mut pass = true;
    for &(rows, cols) in F3_SHAPES {
        for kind in ["random", "padded", "edge"] {
            let x = DeviceBuffer::upload(&device, &f3_input(kind, rows, cols)).expect("x");
            let (q0, s0) = variant_quant(&device, &x, (rows, cols), 0);
            let (q1, s1) = variant_quant(&device, &x, (rows, cols), 1);
            let q_bad = q0.iter().zip(&q1).filter(|(a, b)| a != b).count();
            let s_bad = s0
                .chunks(4)
                .zip(s1.chunks(4))
                .filter(|(a, b)| a != b)
                .count();
            let ok = q_bad == 0 && s_bad == 0;
            eprintln!(
                "F3_BITS rows={rows} cols={cols} {kind}: mismatched_bytes={q_bad} mismatched_scales={s_bad} {}",
                verdict(ok)
            );
            pass &= ok;
        }
    }
    assert!(pass, "the wide quantize is not the S3 quantize's bits");
}

/// The wide quantize at one tile a row (one warp a block, eight blocks a CTA), with a partial last
/// CTA, is the S3 quantize's bits.
#[test]
fn the_wide_quantize_at_one_tile_a_row_is_the_s3_quantize_bit_for_bit() {
    let device = h100();
    for &(rows, cols) in &[(16, 64), (48, 192), (4096, 256)] {
        for kind in ["random", "padded", "edge"] {
            let x = DeviceBuffer::upload(&device, &f3_input(kind, rows, cols)).expect("x");
            let s3 = variant_quant(&device, &x, (rows, cols), 0);
            let wide = variant_quant(&device, &x, (rows, cols), 1);
            assert!(
                s3 == wide,
                "rows={rows} cols={cols} {kind}: not the S3 quantize's bits"
            );
        }
    }
}

type CudaEvent = *mut std::ffi::c_void;

unsafe extern "C" {
    fn cudaEventCreate(event: *mut CudaEvent) -> i32;
    fn cudaEventRecord(event: CudaEvent, stream: *mut std::ffi::c_void) -> i32;
    fn cudaEventSynchronize(event: CudaEvent) -> i32;
    fn cudaEventElapsedTime(ms: *mut f32, start: CudaEvent, end: CudaEvent) -> i32;
    fn cudaEventDestroy(event: CudaEvent) -> i32;
}

/// `(median, min, max)` µs of 20 launches of `launch` after 3 warm-ups between two CUDA events on
/// the default stream, `before` run untimed on the same stream before each.
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

/// The two kernels' time at every prefill shape (random rows) and at the 512-row bucket's padded
/// rows, read-flush cold (a 100 MiB quantize of other data before each launch, so L2 holds clean
/// lines of it) and hot, two rounds (A, B, A, B), as `TC_TIME` lines; the gate is ≤ 25 µs at
/// 2048 × 5,120.
#[test]
fn the_wide_quantize_times_are_recorded() {
    use scratchy_target_cuda::kernels::quant_rows_fp8_variant;
    let device = h100();
    // The flush: 2,048 × 25,600 bf16 (100 MiB) through the S3 kernel.
    let (fr, fc) = (2048usize, 25600usize);
    let fx = DeviceBuffer::upload_bytes(&device, &vec![0u8; fr * fc * 2]).expect("flush x");
    let fq = DeviceBuffer::upload_bytes(&device, &vec![0u8; fr * fc]).expect("flush q");
    let fs = DeviceBuffer::upload_bytes(&device, &vec![0u8; fr * 4]).expect("flush s");
    let flush = || {
        // SAFETY: the flush buffers hold their shape; null is the default stream.
        unsafe {
            quant_rows_fp8_variant(
                fx.ptr(),
                (fq.ptr(), fs.ptr()),
                (fr as u32, fc as u32),
                1,
                std::ptr::null_mut(),
            )
        }
        .expect("flush");
    };
    let mut cases: Vec<(usize, usize, &str)> =
        F3_SHAPES.iter().map(|&(r, c)| (r, c, "random")).collect();
    cases.extend([
        (512, 2048, "padded"),
        (512, 2816, "padded"),
        (512, 5120, "padded"),
        (512, 17408, "padded"),
    ]);
    for (rows, cols, kind) in cases {
        let x = DeviceBuffer::upload(&device, &f3_input(kind, rows, cols)).expect("x");
        let q = DeviceBuffer::upload_bytes(&device, &vec![0u8; rows * cols]).expect("q");
        let s = DeviceBuffer::upload_bytes(&device, &vec![0u8; rows * 4]).expect("s");
        let shape = format!("qr_{rows}x{cols}_{kind}");
        for round in 1..=2 {
            for (variant, name) in [(0u32, "v0_s3_one_warp"), (1, "v1_f3_eight_warps")] {
                let launch = || {
                    // SAFETY: `x` holds rows·cols bf16, `q` rows·cols bytes and `s` rows f32; null
                    // is the default stream.
                    unsafe {
                        quant_rows_fp8_variant(
                            x.ptr(),
                            (q.ptr(), s.ptr()),
                            (rows as u32, cols as u32),
                            variant,
                            std::ptr::null_mut(),
                        )
                    }
                    .expect("quant_rows_fp8_variant launch");
                };
                let cold = timed(flush, launch);
                let hot = timed(|| {}, launch);
                for (temp, (med, lo, hi)) in [("readcold", cold), ("hot", hot)] {
                    eprintln!(
                        "TC_TIME {temp} {shape} {name}: median={med:.1}us min={lo:.1}us max={hi:.1}us \
                         {:.2} TB/s (round {round})",
                        (rows * cols * 3) as f64 / med / 1e6
                    );
                }
            }
        }
    }
}

// ── FP8 S8 Q1: the one-row quantize ──────────────────────────────────────────────

/// `quant_rows_fp8`'s tile: the quantized activation's rows (row 0 the GEMV's).
const QROWS: usize = 16;

/// A buffer of `len` copies of `byte`, at least 256 bytes.
fn filled(device: &CudaDevice, len: usize, byte: u8) -> DeviceBuffer {
    DeviceBuffer::upload_bytes(device, &vec![byte; len.max(256)]).expect("upload")
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
}

/// Q1 on `x` (`[1, k]` bf16): its e4m3 bytes and its scale's bits.
fn quant_one(device: &CudaDevice, x: &[u16]) -> (Vec<u8>, u32) {
    use scratchy_target_cuda::kernels::quant_row_fp8_one;
    let k = x.len();
    let xb = DeviceBuffer::upload(device, x).expect("x");
    let q = filled(device, k, 0xEE);
    let s = filled(device, 4, 0xEE);
    // SAFETY: `xb` holds k bf16, `q` k bytes and `s` one f32, three allocations; null is the
    // default stream, which the downloads wait for.
    unsafe {
        quant_row_fp8_one(
            xb.ptr().cast_const(),
            (q.ptr(), s.ptr()),
            k as u32,
            std::ptr::null_mut(),
        )
    }
    .expect("quant_row_fp8_one launch");
    let qb: Vec<u8> = q
        .download()
        .expect("q")
        .iter()
        .flat_map(|h| h.to_le_bytes())
        .take(k)
        .collect();
    let sb: Vec<u8> = s
        .download()
        .expect("s")
        .iter()
        .flat_map(|h| h.to_le_bytes())
        .collect();
    (qb, u32::from_le_bytes([sb[0], sb[1], sb[2], sb[3]]))
}

/// Q1 is `quant_rows_fp8`'s row 0 bit for bit (the bytes and the scale), at the 27B's 5,120 and
/// 17,408 columns, Gemma's 2,816 and 2,112 (a partial last 256-column tile), 64 and 1,088, on
/// random rows, a zero row (the minimum scale) and a row of one large value (the clamp).
#[test]
fn the_one_row_quantize_is_quant_rows_fp8_bit_for_bit() {
    let device = h100();
    let mut pass = true;
    for k in [5120usize, 17408, 2816, 2112, 64, 1088] {
        let mut rows = vec![
            ("random", bf16_inputs(k, 0x7A0 ^ k as u64)),
            ("zero", vec![0u16; k]),
        ];
        let mut big = bf16_inputs(k, 0x7A1);
        big[k / 2] = 0x7F00; // bf16 1.7e38
        rows.push(("one_large", big));
        for (what, x) in rows {
            let act = Act::new(&device, &x);
            act.quantize();
            let want_q: Vec<u8> = act
                .q
                .download()
                .expect("q")
                .iter()
                .flat_map(|h| h.to_le_bytes())
                .take(k)
                .collect();
            let sb: Vec<u8> = act
                .s
                .download()
                .expect("s")
                .iter()
                .flat_map(|h| h.to_le_bytes())
                .collect();
            let want_s = u32::from_le_bytes([sb[0], sb[1], sb[2], sb[3]]);
            let (got_q, got_s) = quant_one(&device, &x);
            let bad = got_q.iter().zip(&want_q).filter(|(a, b)| a != b).count();
            let ok = bad == 0 && got_s == want_s;
            eprintln!(
                "Q1_BITS k={k} {what}: mismatched_bytes={bad} scale={:#010x} want={want_s:#010x} {}",
                got_s,
                verdict(ok)
            );
            pass &= ok;
        }
    }
    assert!(pass, "Q1 is not quant_rows_fp8's row 0");
}

// ── FP8 S8 P2: the dense prefill quantize's probe forms ─────────────────────────

/// P2's probe shapes: 2,048 token rows × the dense fp8 `k`s of the three models (2,048, 2,816,
/// 4,096, 5,120, 6,144, 17,408).
const P2_ROWS: usize = 2048;
const P2_KS: [usize; 6] = [2048, 2816, 4096, 5120, 6144, 17408];

/// The forms `cols` is probed at, `(variant, name)`: F3 (the launcher's today), RESIDENT where it
/// takes the row (at most 5,120 columns), PER ROW.
fn p2_forms(cols: usize) -> Vec<(u32, &'static str)> {
    let mut forms = vec![(1, "f3")];
    if cols <= 5120 {
        forms.push((2, "resident"));
    }
    forms.push((3, "per_row"));
    forms
}

/// P2's probe forms (variant 2, a block's tiles resident in shared memory; 3, one CTA a row) are
/// F3's bits, bytes and scales, at 2,048 rows × every probed `k` on random, padded and edge rows,
/// and RESIDENT at a partial last tile and a single block, as `P2_BITS` lines; each refuses the
/// shapes it does not take, as `P2_REFUSED` lines.
#[test]
fn the_dense_quantize_probe_forms_are_f3s_bits() {
    use scratchy_target_cuda::kernels::quant_rows_fp8_variant;
    let device = h100();
    let mut pass = true;
    let shapes = P2_KS
        .iter()
        .map(|&k| (P2_ROWS, k))
        .chain([(16, 320), (48, 5120)]);
    for (rows, cols) in shapes {
        for kind in ["random", "padded", "edge"] {
            let x = DeviceBuffer::upload(&device, &f3_input(kind, rows, cols)).expect("x");
            let f3 = variant_quant(&device, &x, (rows, cols), 1);
            for (variant, name) in p2_forms(cols).into_iter().skip(1) {
                if variant == 3 && !cols.is_multiple_of(256) {
                    continue;
                }
                let (q, s) = variant_quant(&device, &x, (rows, cols), variant);
                let q_bad = f3.0.iter().zip(&q).filter(|(a, b)| a != b).count();
                let s_bad = (f3.1.chunks(4).zip(s.chunks(4)))
                    .filter(|(a, b)| a != b)
                    .count();
                let ok = q_bad == 0 && s_bad == 0;
                eprintln!(
                    "P2_BITS rows={rows} cols={cols} {kind} {name}: mismatched_bytes={q_bad} \
                     mismatched_scales={s_bad} {}",
                    verdict(ok)
                );
                pass &= ok;
            }
        }
    }
    let null = std::ptr::null_mut();
    for (what, cols, variant) in [
        ("resident above 20 tiles", 5376u32, 2u32),
        ("per row at a partial chunk", 320, 3),
        ("per row above 68 chunks", 17664, 3),
        ("variant 4", 2048, 4),
    ] {
        // SAFETY: every shape here is refused before any pointer is touched.
        let got = unsafe { quant_rows_fp8_variant(null, (null, null), (16, cols), variant, null) };
        let ok = got == Err(KernelError::RowShape { rows: 16, cols });
        eprintln!("P2_REFUSED {what}: {got:?} {}", verdict(ok));
        pass &= ok;
    }
    assert!(
        pass,
        "a P2 probe form is not F3's bits, or took a shape it refuses"
    );
}

/// P2's probe: F3, RESIDENT and PER ROW at 2,048 rows × every probed `k` (random rows), read-flush
/// cold and hot, two rounds (each form in turn, twice), as `TC_TIME` lines, then per `k` and
/// temperature the fastest form by its better round's median, as `P2_WINNER` lines.
#[test]
fn the_dense_quantize_probe_times_are_recorded() {
    use scratchy_target_cuda::kernels::quant_rows_fp8_variant;
    let device = h100();
    let (fr, fc) = (2048usize, 25600usize);
    let fx = DeviceBuffer::upload_bytes(&device, &vec![0u8; fr * fc * 2]).expect("flush x");
    let fq = DeviceBuffer::upload_bytes(&device, &vec![0u8; fr * fc]).expect("flush q");
    let fs = DeviceBuffer::upload_bytes(&device, &vec![0u8; fr * 4]).expect("flush s");
    let flush = || {
        // SAFETY: the flush buffers hold their shape; null is the default stream.
        unsafe {
            quant_rows_fp8_variant(
                fx.ptr(),
                (fq.ptr(), fs.ptr()),
                (fr as u32, fc as u32),
                1,
                std::ptr::null_mut(),
            )
        }
        .expect("flush");
    };
    for cols in P2_KS {
        let rows = P2_ROWS;
        let x = DeviceBuffer::upload(&device, &f3_input("random", rows, cols)).expect("x");
        let q = DeviceBuffer::upload_bytes(&device, &vec![0u8; rows * cols]).expect("q");
        let s = DeviceBuffer::upload_bytes(&device, &vec![0u8; rows * 4]).expect("s");
        let shape = format!("qr_{rows}x{cols}_random");
        // Per (temperature, form): the best round's median.
        let mut best: Vec<(&str, &str, f64)> = Vec::new();
        for round in 1..=2 {
            for (variant, name) in p2_forms(cols) {
                let launch = || {
                    // SAFETY: `x` holds rows·cols bf16, `q` rows·cols bytes and `s` rows f32; null
                    // is the default stream.
                    unsafe {
                        quant_rows_fp8_variant(
                            x.ptr(),
                            (q.ptr(), s.ptr()),
                            (rows as u32, cols as u32),
                            variant,
                            std::ptr::null_mut(),
                        )
                    }
                    .expect("quant_rows_fp8_variant launch");
                };
                let cold = timed(flush, launch);
                let hot = timed(|| {}, launch);
                for (temp, (med, lo, hi)) in [("readcold", cold), ("hot", hot)] {
                    eprintln!(
                        "TC_TIME {temp} {shape} p2_{name}: median={med:.1}us min={lo:.1}us \
                         max={hi:.1}us {:.2} TB/s (round {round})",
                        (rows * cols * 3) as f64 / med / 1e6
                    );
                    match best.iter_mut().find(|(t, n, _)| (*t, *n) == (temp, name)) {
                        Some(b) => b.2 = b.2.min(med),
                        None => best.push((temp, name, med)),
                    }
                }
            }
        }
        for temp in ["readcold", "hot"] {
            let mine: Vec<_> = best.iter().filter(|(t, ..)| *t == temp).collect();
            let f3 = mine
                .iter()
                .find(|(_, n, _)| *n == "f3")
                .map_or(f64::NAN, |b| b.2);
            if let Some((_, name, med)) = mine.iter().min_by(|a, b| a.2.total_cmp(&b.2)) {
                eprintln!(
                    "P2_WINNER k={cols} {temp}: {name} {med:.1}us (f3 {f3:.1}us, saves {:.1}us)",
                    f3 - med
                );
            }
        }
    }
}
