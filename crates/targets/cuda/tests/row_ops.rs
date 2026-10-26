// SPDX-License-Identifier: Apache-2.0
//! The row kernels (`kernels/rms_norm_bf16.cu`, `add_bf16.cu`, `silu_mul_bf16.cu`) against an f64
//! CPU reference, on the H100, and their launchers' refusals.
//!
//! Built only under `nvcc` (`required-features`). No GPU fails rather than skips: this test is the
//! kernels' only gate. The references and bounds are `common::numerics`'. Every shape runs and
//! prints its line before a test fails, so one run reports all the numbers.

// `pub`: each test binary uses a different part of `common`, and nothing in it is dead code.
pub mod common {
    pub mod numerics;
}

use common::numerics::{
    add_reference, bf16_inputs, bits, centred_gains, compare_rows, mismatched_bits,
    rms_norm_reference, row_bound, silu_mul_reference,
};
use common::numerics::{
    bf16_in, compare_rows_with, gelu_mul_bound, gelu_mul_reference, out_of_bound_fraction,
    soft_cap_bound, soft_cap_reference, widen,
};
use scratchy_target_cuda::kernels::{KernelError, add_bf16, rms_norm_bf16, silu_mul_bf16};
use scratchy_target_cuda::kernels::{
    gelu_mul_bf16, layer_scale_bf16, rms_norm_unit_bf16, scale_bf16, soft_cap_bf16,
};
use scratchy_target_cuda::runtime::{CudaDevice, DeviceBuffer};

/// qwen3.5's `rms_norm_eps`.
const EPS: f32 = 1e-6;
/// The zero-centred convention's offset (`GainConvention::OnePlusScale.offset()`).
const ONE_PLUS_SCALE: f32 = 1.0;
/// The stored-gain convention's offset (`GainConvention::Scale.offset()`).
const SCALE: f32 = 0.0;

/// Extra output elements past `rows·cols`, which a kernel must leave untouched.
const TAIL: usize = 4096;
/// Every output element before a launch: a bf16 NaN, so an element a kernel never writes fails.
const SENTINEL: u16 = 0xFFFF;

fn h100() -> CudaDevice {
    CudaDevice::open()
        .unwrap_or_else(|e| panic!("the `nvcc` feature's tests need an sm_90 GPU (H100): {e}"))
}

fn verdict(pass: bool) -> &'static str {
    if pass { "PASS" } else { "FAIL" }
}

/// An output buffer of `len` elements and the tail, every element [`SENTINEL`].
fn output(device: &CudaDevice, len: usize) -> DeviceBuffer {
    DeviceBuffer::upload(device, &vec![SENTINEL; len + TAIL]).expect("upload the output")
}

/// The output's `len` elements, and whether the tail past them is intact. Downloading waits for
/// the default stream, and with it the launch.
fn result(out: &DeviceBuffer, len: usize) -> (Vec<u16>, bool) {
    let mut bits = out.download().expect("download the output");
    let tail = bits[len..].iter().all(|&b| b == SENTINEL);
    bits.truncate(len);
    (bits, tail)
}

/// Run the norm on `x` and `g` over `(rows, cols)`; the GPU's output and whether its tail held.
fn run_norm(
    device: &CudaDevice,
    x: &[u16],
    g: &[u16],
    shape: (usize, usize),
    eps: f32,
    offset: f32,
) -> (Vec<u16>, bool) {
    let (xb, gb) = (
        DeviceBuffer::upload(device, x).expect("upload x"),
        DeviceBuffer::upload(device, g).expect("upload g"),
    );
    let out = output(device, x.len());
    let dims = (shape.0 as u32, shape.1 as u32);
    // SAFETY: cudaMalloc pointers are 256-byte aligned; `x` and `out` hold rows·cols elements and
    // `g` cols; `out` is its own allocation; null is the default stream.
    unsafe {
        rms_norm_bf16(
            xb.ptr(),
            gb.ptr(),
            out.ptr(),
            dims,
            eps,
            offset,
            std::ptr::null_mut(),
        )
    }
    .expect("rms_norm_bf16 launch");
    result(&out, x.len())
}

/// The norm over `(rows, cols)` with random `x` and gains: every element within the bound.
fn norm_shape(device: &CudaDevice, (rows, cols): (usize, usize), offset: f32) -> bool {
    let seed = (rows as u64) << 32 | cols as u64;
    let x = bf16_inputs(rows * cols, 0x0A0A_0000 ^ seed);
    let g = match offset {
        SCALE => bf16_inputs(cols, 0x0B0B_0000 ^ seed),
        _ => centred_gains(cols, 0x0B0B_0000 ^ seed),
    };
    let (gpu, tail) = run_norm(device, &x, &g, (rows, cols), EPS, offset);
    let refs = rms_norm_reference(&x, &g, (rows, cols), f64::from(EPS), f64::from(offset));
    let cmp = compare_rows(&gpu, &refs, cols);
    let pass = cmp.pass() && tail;
    eprintln!(
        "rms_norm_bf16 rows={rows} cols={cols} offset={offset}: {} tail_intact={tail} {}",
        cmp.line(),
        verdict(pass)
    );
    pass
}

#[test]
fn rms_norm_bf16_matches_the_cpu_reference() {
    let device = h100();
    let mut pass = true;
    for shape in [(1, 4096), (256, 4096), (256, 1024), (4096, 256)] {
        pass &= norm_shape(&device, shape, ONE_PLUS_SCALE);
    }
    pass &= norm_shape(&device, (128, 4096), SCALE);
    assert!(pass, "an RMSNorm shape is out of bound");
}

/// `(7, 4096)`: random rows, a zero row, an eps-sensitive row and a large row, each checked by
/// itself; and the convention self-check on the random rows.
#[test]
fn rms_norm_bf16_edge_rows() {
    const COLS: usize = 4096;
    const ROWS: usize = 7;
    let (zero, small, large) = (2, 4, 6);
    let device = h100();
    let mut x = bf16_inputs(ROWS * COLS, 0x0E0E_0007);
    let row = |r: usize| r * COLS..(r + 1) * COLS;
    x[row(zero)].fill(0);
    // Constant 2⁻¹⁰: mean(x²) = 2⁻²⁰ ≈ 9.5e-7, so eps = 1e-6 moves the result by about √2.
    x[row(small)].fill(bits(1.0 / 1024.0));
    // |x| near 2¹⁰: the f32 sum of squares (about 2³² here) must not overflow.
    for (i, v) in x[row(large)].iter_mut().enumerate() {
        *v = bits(if i % 2 == 0 { 1000.0 } else { -1020.0 });
    }
    let g = centred_gains(COLS, 0x0E0E_00FF);
    let (gpu, tail) = run_norm(&device, &x, &g, (ROWS, COLS), EPS, ONE_PLUS_SCALE);
    let refs = rms_norm_reference(&x, &g, (ROWS, COLS), f64::from(EPS), 1.0);
    let cmp = compare_rows(&gpu, &refs, COLS);
    let mut pass = cmp.pass() && tail;
    eprintln!(
        "rms_norm_bf16 rows={ROWS} cols={COLS} (zero, eps-sensitive, large rows): {} \
         tail_intact={tail} {}",
        cmp.line(),
        verdict(cmp.pass() && tail)
    );

    let zero_exact = gpu[row(zero)].iter().all(|&b| b == 0 || b == 0x8000);
    eprintln!(
        "zero row is exactly 0: {zero_exact} {}",
        verdict(zero_exact)
    );
    pass &= zero_exact;

    // The eps-sensitive row passes (above), and a reference that drops eps misses it.
    let no_eps = rms_norm_reference(&x[row(small)], &g, (1, COLS), 0.0, 1.0);
    let missed = gpu[row(small)]
        .iter()
        .zip(&no_eps)
        .filter(|&(&b, &r)| (f64::from(half::bf16::from_bits(b).to_f32()) - r).abs() > row_bound(r))
        .count();
    let eps_matters = missed == COLS;
    eprintln!(
        "eps-sensitive row: the eps = 0 reference misses {missed} of {COLS} elements {}",
        verdict(eps_matters)
    );
    pass &= eps_matters;

    let large_finite = gpu[row(large)]
        .iter()
        .all(|&b| half::bf16::from_bits(b).is_finite());
    eprintln!(
        "large row is finite: {large_finite} {}",
        verdict(large_finite)
    );
    pass &= large_finite;

    // The convention self-check: on the random rows, the reference without the `+1` differs from
    // the zero-centred one by more than the bound in more than 90% of elements.
    let random: Vec<u16> = [0, 1, 3, 5]
        .iter()
        .flat_map(|&r| x[row(r)].to_vec())
        .collect();
    let with = rms_norm_reference(&random, &g, (4, COLS), f64::from(EPS), 1.0);
    let without = rms_norm_reference(&random, &g, (4, COLS), f64::from(EPS), 0.0);
    let apart = with
        .iter()
        .zip(&without)
        .filter(|&(&a, &b)| (a - b).abs() > row_bound(a))
        .count();
    let distinguishes = apart * 10 > with.len() * 9;
    eprintln!(
        "convention self-check: Scale and OnePlusScale references differ past the bound in \
         {apart} of {} elements {}",
        with.len(),
        verdict(distinguishes)
    );
    pass &= distinguishes;
    assert!(pass, "an RMSNorm edge row failed");
}

/// Launch the add or SiluMul (`which`) on `a` and `b`; the GPU's output and its tail.
fn run_binary(
    device: &CudaDevice,
    which: Binary,
    a: &[u16],
    b: &[u16],
    dims: (u32, u32),
) -> (Vec<u16>, bool) {
    let (ab, bb) = (
        DeviceBuffer::upload(device, a).expect("upload a"),
        DeviceBuffer::upload(device, b).expect("upload b"),
    );
    let out = output(device, a.len());
    // SAFETY: cudaMalloc pointers are 256-byte aligned; each buffer holds rows·cols elements;
    // `out` is its own allocation; null is the default stream.
    unsafe { which.launch(ab.ptr(), bb.ptr(), out.ptr(), dims) }.expect("launch");
    result(&out, a.len())
}

#[derive(Clone, Copy, Debug)]
enum Binary {
    Add,
    SiluMul,
}

impl Binary {
    /// # Safety
    ///
    /// As [`add_bf16`]'s and [`silu_mul_bf16`]'s, on the default stream.
    unsafe fn launch(
        self,
        a: *const std::ffi::c_void,
        b: *const std::ffi::c_void,
        out: *mut std::ffi::c_void,
        dims: (u32, u32),
    ) -> Result<(), KernelError> {
        let stream = std::ptr::null_mut();
        // SAFETY: the caller's contract.
        unsafe {
            match self {
                Self::Add => add_bf16(a, b, out, dims, stream),
                Self::SiluMul => silu_mul_bf16(a, b, out, dims, stream),
            }
        }
    }
}

#[test]
fn add_bf16_is_bit_exact() {
    let device = h100();
    let mut pass = true;
    for (rows, cols) in [(1, 4096), (7, 4096), (256, 4096), (256, 1024)] {
        let seed = (rows as u64) << 32 | cols as u64;
        let a = bf16_inputs(rows * cols, 0x0ADD_0000 ^ seed);
        let b = bf16_inputs(rows * cols, 0x0ADD_FFFF ^ seed);
        let dims = (rows as u32, cols as u32);
        let (gpu, tail) = run_binary(&device, Binary::Add, &a, &b, dims);
        let mismatched = mismatched_bits(&gpu, &add_reference(&a, &b));
        let ok = mismatched == 0 && tail;
        eprintln!(
            "add_bf16 rows={rows} cols={cols}: mismatched_bits={mismatched} tail_intact={tail} {}",
            verdict(ok)
        );
        pass &= ok;
    }
    assert!(pass, "an add shape is not bit-exact");
}

#[test]
fn silu_mul_bf16_matches_the_cpu_reference() {
    let device = h100();
    let mut pass = true;
    for (rows, cols) in [(1, 12288), (256, 12288), (256, 3584)] {
        let seed = (rows as u64) << 32 | cols as u64;
        // Gates in [-8, 8]: the activation's whole curve, not only its linear middle.
        let gate: Vec<u16> = bf16_inputs(rows * cols, 0x5150_0000 ^ seed)
            .into_iter()
            .map(|b| bits(half::bf16::from_bits(b).to_f32() * 8.0))
            .collect();
        let up = bf16_inputs(rows * cols, 0x5150_FFFF ^ seed);
        let dims = (rows as u32, cols as u32);
        let (gpu, tail) = run_binary(&device, Binary::SiluMul, &gate, &up, dims);
        let cmp = compare_rows(&gpu, &silu_mul_reference(&gate, &up), cols);
        let ok = cmp.pass() && tail;
        eprintln!(
            "silu_mul_bf16 rows={rows} cols={cols}: {} tail_intact={tail} {}",
            cmp.line(),
            verdict(ok)
        );
        pass &= ok;
    }
    assert!(pass, "a SiluMul shape is out of bound");
}

/// `(7, 12288)` with row 3 holding the extreme gates `{−100, −88, −20, −1, 0, 1, 20, 88, 100}`
/// against `u ∈ {±1, ±0.5}`: no NaN or Inf, and `g ≤ −88` gives ±0 or a value within the bound.
#[test]
fn silu_mul_bf16_extreme_gates() {
    const COLS: usize = 12288;
    const ROWS: usize = 7;
    const EXTREME: usize = 3;
    const GATES: [f32; 9] = [-100.0, -88.0, -20.0, -1.0, 0.0, 1.0, 20.0, 88.0, 100.0];
    const UPS: [f32; 4] = [1.0, -1.0, 0.5, -0.5];
    let device = h100();
    let mut gate = bf16_inputs(ROWS * COLS, 0x5150_0007);
    let mut up = bf16_inputs(ROWS * COLS, 0x5150_0070);
    let at = EXTREME * COLS;
    for c in 0..COLS {
        gate[at + c] = bits(GATES[c % GATES.len()]);
        up[at + c] = bits(UPS[(c / GATES.len()) % UPS.len()]);
    }
    let (gpu, tail) = run_binary(&device, Binary::SiluMul, &gate, &up, (7, COLS as u32));
    let refs = silu_mul_reference(&gate, &up);
    let cmp = compare_rows(&gpu, &refs, COLS);
    let mut pass = cmp.pass() && tail;
    eprintln!(
        "silu_mul_bf16 rows={ROWS} cols={COLS} (extreme row {EXTREME}): {} tail_intact={tail} {}",
        cmp.line(),
        verdict(cmp.pass() && tail)
    );
    let row = &gpu[at..at + COLS];
    let finite = row.iter().all(|&b| half::bf16::from_bits(b).is_finite());
    let very_negative_ok = (0..COLS)
        .filter(|c| GATES[c % GATES.len()] <= -88.0)
        .all(|c| {
            let g = f64::from(half::bf16::from_bits(row[c]).to_f32());
            g == 0.0 || (g - refs[at + c]).abs() <= row_bound(refs[at + c])
        });
    eprintln!(
        "extreme gates: no NaN or Inf {finite}, g <= -88 gives ±0 or within the bound \
         {very_negative_ok} {}",
        verdict(finite && very_negative_ok)
    );
    pass &= finite && very_negative_ok;
    assert!(pass, "the SiluMul extreme row failed");
}

/// The shapes each launcher must refuse with `RowShape`, writing nothing. The last is 2³²
/// elements, above `INT_MAX`: refused before any pointer is touched, so its buffers are 16 bytes.
const REFUSED: [(u32, u32); 4] = [(256, 1004), (256, 0), (0, 4096), (65536, 65536)];

#[test]
fn every_row_launcher_refuses_a_shape_it_cannot_run() {
    let device = h100();
    let mut pass = true;
    for (rows, cols) in REFUSED {
        // 16-byte buffers when the shape is huge, else the shape's own size (at least one vector).
        let len = (rows as usize * cols as usize).clamp(8, 1 << 20);
        let len = if rows as u64 * cols as u64 > i32::MAX as u64 {
            8
        } else {
            len
        };
        let input = DeviceBuffer::upload(&device, &vec![0; len]).expect("upload an input");
        let gain = DeviceBuffer::upload(&device, &vec![0; len]).expect("upload a gain");
        for kernel in ["rms_norm_bf16", "add_bf16", "silu_mul_bf16"] {
            let out = DeviceBuffer::upload(&device, &vec![SENTINEL; len]).expect("upload out");
            let null = std::ptr::null_mut();
            // SAFETY: each launcher refuses these shapes before it touches any pointer.
            let got = unsafe {
                match kernel {
                    "rms_norm_bf16" => rms_norm_bf16(
                        input.ptr(),
                        gain.ptr(),
                        out.ptr(),
                        (rows, cols),
                        EPS,
                        ONE_PLUS_SCALE,
                        null,
                    ),
                    "add_bf16" => add_bf16(input.ptr(), gain.ptr(), out.ptr(), (rows, cols), null),
                    _ => silu_mul_bf16(input.ptr(), gain.ptr(), out.ptr(), (rows, cols), null),
                }
            };
            let unchanged = out
                .download()
                .expect("download out")
                .iter()
                .all(|&b| b == SENTINEL);
            let ok = got == Err(KernelError::RowShape { rows, cols }) && unchanged;
            eprintln!(
                "{kernel} rows={rows} cols={cols}: {got:?} out_unchanged={unchanged} {}",
                verdict(ok)
            );
            pass &= ok;
        }
    }
    assert!(pass, "a row launcher did not refuse a shape it cannot run");
}

/// `(7, 1000)`: 1000 is a multiple of 8 but not of 256 (3 · 256 + 232), so each kernel's last
/// column tile is partial (`kernels/*.cu`: a plain ThunderKittens tile reads zeros past `cols` and
/// stores nothing there), and 7 rows leave the 16-row block partial. Under each kernel's existing
/// bound (bit-exactness for the add), the output's tail intact.
#[test]
fn every_row_kernel_takes_a_partial_last_column_tile() {
    let device = h100();
    let (rows, cols) = (7usize, 1000usize);
    let mut pass = norm_shape(&device, (rows, cols), ONE_PLUS_SCALE);
    let dims = (rows as u32, cols as u32);
    let a = bf16_inputs(rows * cols, 0x0ADD_1000);
    let b = bf16_inputs(rows * cols, 0x0ADD_1001);
    let (gpu, tail) = run_binary(&device, Binary::Add, &a, &b, dims);
    let mismatched = mismatched_bits(&gpu, &add_reference(&a, &b));
    let ok = mismatched == 0 && tail;
    eprintln!(
        "add_bf16 rows={rows} cols={cols} (partial last column tile): \
         mismatched_bits={mismatched} tail_intact={tail} {}",
        verdict(ok)
    );
    pass &= ok;
    let gate: Vec<u16> = bf16_inputs(rows * cols, 0x5150_1000)
        .into_iter()
        .map(|b| bits(half::bf16::from_bits(b).to_f32() * 8.0))
        .collect();
    let up = bf16_inputs(rows * cols, 0x5150_1001);
    let (gpu, tail) = run_binary(&device, Binary::SiluMul, &gate, &up, dims);
    let cmp = compare_rows(&gpu, &silu_mul_reference(&gate, &up), cols);
    let ok = cmp.pass() && tail;
    eprintln!(
        "silu_mul_bf16 rows={rows} cols={cols} (partial last column tile): {} tail_intact={tail} {}",
        cmp.line(),
        verdict(ok)
    );
    pass &= ok;
    assert!(pass, "a row kernel's partial last column tile failed");
}

/// Each row kernel run twice on its largest test shape gives the same bits: every reduction and
/// every map runs in a fixed order.
#[test]
fn every_row_kernel_repeats_bit_identically() {
    let device = h100();
    let mut pass = true;
    let (rows, cols) = (4096usize, 256usize);
    let x = bf16_inputs(rows * cols, 0x0A0A_2000);
    let g = centred_gains(cols, 0x0B0B_2000);
    let run = || run_norm(&device, &x, &g, (rows, cols), EPS, ONE_PLUS_SCALE).0;
    let mismatched = mismatched_bits(&run(), &run());
    eprintln!(
        "rms_norm_bf16 rows={rows} cols={cols} repeat mismatched_bits={mismatched} {}",
        verdict(mismatched == 0)
    );
    pass &= mismatched == 0;
    for (which, (rows, cols)) in [
        (Binary::Add, (256usize, 4096usize)),
        (Binary::SiluMul, (256, 12288)),
    ] {
        let a = bf16_inputs(rows * cols, 0x0ADD_2000);
        let b = bf16_inputs(rows * cols, 0x0ADD_2001);
        let dims = (rows as u32, cols as u32);
        let run = || run_binary(&device, which, &a, &b, dims).0;
        let mismatched = mismatched_bits(&run(), &run());
        eprintln!(
            "{which:?} rows={rows} cols={cols} repeat mismatched_bits={mismatched} {}",
            verdict(mismatched == 0)
        );
        pass &= mismatched == 0;
    }
    assert!(pass, "a row kernel's repeat differs");
}

/// For the base-against-change comparison (#850): the SiluMul's output at each of its test shapes
/// (the seeds of `silu_mul_bf16_matches_the_cpu_reference`, then `(7, 1000)`), written as raw
/// little-endian bf16 to `<CARGO_TARGET_TMPDIR>/row_ops_dump/<case>.bin`, with `case= elems=
/// fnv1a=` printed. The same file run at two commits gives the `base_vs_pr mismatched_bits=` per
/// case; this test asserts only that each launch ran.
#[test]
fn silu_mul_bf16_outputs_are_dumped_for_a_base_comparison() {
    let device = h100();
    for (rows, cols) in [(1usize, 12288usize), (256, 12288), (256, 3584), (7, 1000)] {
        let seed = (rows as u64) << 32 | cols as u64;
        let gate: Vec<u16> = bf16_inputs(rows * cols, 0x5150_0000 ^ seed)
            .into_iter()
            .map(|b| bits(half::bf16::from_bits(b).to_f32() * 8.0))
            .collect();
        let up = bf16_inputs(rows * cols, 0x5150_FFFF ^ seed);
        let dims = (rows as u32, cols as u32);
        let (gpu, _) = run_binary(&device, Binary::SiluMul, &gate, &up, dims);
        dump_case(&format!("silu_mul_bf16_r{rows}_c{cols}"), &gpu);
    }
}

/// Write `out` as `<case>.bin` under `<CARGO_TARGET_TMPDIR>/row_ops_dump/`, and print its line.
pub fn dump_case(case: &str, out: &[u16]) {
    let dir = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("row_ops_dump");
    std::fs::create_dir_all(&dir).expect("create the dump directory");
    let bytes: Vec<u8> = out.iter().flat_map(|e| e.to_le_bytes()).collect();
    std::fs::write(dir.join(format!("{case}.bin")), &bytes).expect("write a dump");
    let fnv = (bytes.iter()).fold(0xCBF2_9CE4_8422_2325u64, |h, &b| {
        (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01B3)
    });
    eprintln!("case={case} elems={} fnv1a={fnv:016x}", out.len());
}

// ── Gemma 4's row kernels (#921): the one-operand maps, the unit norm, the GeGLU ──

/// Gemma 4 26B-A4B's hidden size, and `√hidden`, its embedding scale.
const GEMMA_HIDDEN: usize = 2816;
/// Gemma's final-logit soft-cap.
const GEMMA_CAP: f32 = 30.0;
/// Gemma's vocabulary: the soft-cap's columns.
const GEMMA_VOCAB: usize = 262_144;

fn gemma_scale() -> f32 {
    (GEMMA_HIDDEN as f32).sqrt()
}

/// One of `pointwise_bf16.cu`'s maps, or the unit norm.
#[derive(Clone, Copy, Debug)]
enum Pointwise {
    Scale(f32),
    LayerScale(u16),
    SoftCap(f32),
    RmsNormUnit,
}

/// Run `which` on `x` over `dims`; the output and whether the tail past it held. `in_place` runs it
/// with `out == x` (one buffer holding `x`, then the tail).
fn run_pointwise(
    device: &CudaDevice,
    which: Pointwise,
    x: &[u16],
    dims: (u32, u32),
    in_place: bool,
) -> (Vec<u16>, bool) {
    let mut host = x.to_vec();
    host.extend(std::iter::repeat_n(SENTINEL, TAIL));
    let xb = DeviceBuffer::upload(device, &host).expect("upload x");
    let out = if in_place {
        None
    } else {
        Some(output(device, x.len()))
    };
    let op = out.as_ref().map_or(xb.ptr(), |o| o.ptr());
    let w = match which {
        Pointwise::LayerScale(w) => Some(DeviceBuffer::upload(device, &[w; 8]).expect("upload w")),
        _ => None,
    };
    let null = std::ptr::null_mut();
    // SAFETY: cudaMalloc pointers are 256-byte aligned; `x` and `out` hold rows·cols elements and
    // the tail, `w` one bf16; `out` is `x` exactly or its own allocation; null is the default
    // stream, which the download waits for.
    unsafe {
        match which {
            Pointwise::Scale(s) => scale_bf16(xb.ptr(), op, dims, s, null),
            Pointwise::LayerScale(_) => {
                let w = w.as_ref().expect("the layer scale's weight").ptr();
                layer_scale_bf16(xb.ptr(), w, op, dims, null)
            }
            Pointwise::SoftCap(c) => soft_cap_bf16(xb.ptr(), op, dims, c, null),
            Pointwise::RmsNormUnit => rms_norm_unit_bf16(xb.ptr(), op, dims, EPS, null),
        }
    }
    .expect("launch");
    result(out.as_ref().unwrap_or(&xb), x.len())
}

/// `bf16(f32(x) · s)` per element, round-to-nearest-even: the scale kernels' own sequence.
fn scale_emulation(x: &[u16], s: f32) -> Vec<u16> {
    (x.iter())
        .map(|&b| half::bf16::from_f32(half::bf16::from_bits(b).to_f32() * s).to_bits())
        .collect()
}

#[test]
fn gemma_scale_bf16_is_bit_exact() {
    let device = h100();
    let mut pass = true;
    let s = gemma_scale();
    for (rows, cols) in [
        (1, GEMMA_HIDDEN),
        (512, GEMMA_HIDDEN),
        (4096, GEMMA_HIDDEN),
        (7, 8),
    ] {
        let x = bf16_inputs(rows * cols, 0x5CA1_0000 ^ (rows as u64) << 32 ^ cols as u64);
        let dims = (rows as u32, cols as u32);
        let (gpu, tail) = run_pointwise(&device, Pointwise::Scale(s), &x, dims, false);
        let mismatched = mismatched_bits(&gpu, &scale_emulation(&x, s));
        let ok = mismatched == 0 && tail;
        eprintln!(
            "gemma scale_bf16 rows={rows} cols={cols} scale={s}: mismatched_bits={mismatched} \
             tail_intact={tail} {}",
            verdict(ok)
        );
        pass &= ok;
    }
    assert!(pass, "a Gemma scale shape is not bit-exact");
}

#[test]
fn gemma_layer_scale_bf16_is_bit_exact() {
    let device = h100();
    let mut pass = true;
    let random = bf16_inputs(1, 0x5CA1_0001)[0];
    for w in [bits(0.5), bits(1.0), bits(-0.75), random] {
        for (rows, cols) in [(1, GEMMA_HIDDEN), (512, GEMMA_HIDDEN)] {
            let x = bf16_inputs(rows * cols, 0x5CA1_0002 ^ u64::from(w) << 32 ^ rows as u64);
            let dims = (rows as u32, cols as u32);
            let (gpu, tail) = run_pointwise(&device, Pointwise::LayerScale(w), &x, dims, false);
            let w0 = half::bf16::from_bits(w).to_f32();
            let mismatched = mismatched_bits(&gpu, &scale_emulation(&x, w0));
            let ok = mismatched == 0 && tail;
            eprintln!(
                "gemma layer_scale_bf16 rows={rows} cols={cols} w0={w0}: \
                 mismatched_bits={mismatched} tail_intact={tail} {}",
                verdict(ok)
            );
            pass &= ok;
        }
    }
    assert!(pass, "a Gemma layer scale is not bit-exact");
}

/// Logit-like inputs in [-64, 64]: the soft-cap's whole curve, linear middle and saturation.
fn logits(len: usize, seed: u64) -> Vec<u16> {
    bf16_in(len, seed, (-64.0, 64.0))
}

#[test]
fn gemma_soft_cap_bf16_matches_the_f64_reference() {
    let device = h100();
    let mut pass = true;
    let cap = f64::from(GEMMA_CAP);
    for (rows, cols) in [(1, GEMMA_VOCAB), (512, GEMMA_VOCAB), (16, GEMMA_HIDDEN)] {
        let x = logits(rows * cols, 0x50F7_0000 ^ (rows as u64) << 32 ^ cols as u64);
        let dims = (rows as u32, cols as u32);
        let (gpu, tail) = run_pointwise(&device, Pointwise::SoftCap(GEMMA_CAP), &x, dims, false);
        let refs = soft_cap_reference(&x, cap);
        let cmp = compare_rows_with(&gpu, &refs, cols, |_, r| soft_cap_bound(r));
        let ok = cmp.pass() && tail;
        eprintln!(
            "gemma soft_cap_bf16 rows={rows} cols={cols}: {} tail_intact={tail} {}",
            cmp.line(),
            verdict(ok)
        );
        pass &= ok;
        if rows == 16 {
            // Self-check: the cap-20 reference must be out of bound on at least 90% of the
            // elements whose rounding it changes.
            let wrong = soft_cap_reference(&x, 20.0);
            let (frac, changed) =
                out_of_bound_fraction(&gpu, (&refs, &wrong), |_, r| soft_cap_bound(r));
            let ok = frac >= 0.9;
            eprintln!(
                "gemma soft_cap self-check cap=20: out_of_bound_fraction={frac:.4} \
                 changed={changed} {}",
                verdict(ok)
            );
            pass &= ok;
        }
    }
    // The edge values, exact: ±0 → ±0, ±inf → ±30, ±300 → ±30, NaN → NaN.
    let edges = [
        0x0000,
        0x8000,
        0x7F80,
        0xFF80,
        bits(300.0),
        bits(-300.0),
        0x7FC0,
        0x0000,
    ];
    let (gpu, _) = run_pointwise(
        &device,
        Pointwise::SoftCap(GEMMA_CAP),
        &edges,
        (1, 8),
        false,
    );
    let want = [
        0x0000,
        0x8000,
        bits(30.0),
        bits(-30.0),
        bits(30.0),
        bits(-30.0),
    ];
    let nan = half::bf16::from_bits(gpu[6]).is_nan();
    let ok = gpu[..6] == want && nan;
    eprintln!(
        "gemma soft_cap edges: got={:04x?} want={want:04x?} nan_to_nan={nan} {}",
        &gpu[..7],
        verdict(ok)
    );
    pass &= ok;
    assert!(pass, "a Gemma soft-cap shape is out of bound");
}

#[test]
fn gemma_soft_cap_bf16_in_place_is_bit_identical() {
    let device = h100();
    let mut pass = true;
    for (rows, cols) in [(1, GEMMA_VOCAB), (512, GEMMA_VOCAB)] {
        let x = logits(rows * cols, 0x50F7_1000 ^ rows as u64);
        let dims = (rows as u32, cols as u32);
        let which = Pointwise::SoftCap(GEMMA_CAP);
        let (fresh, _) = run_pointwise(&device, which, &x, dims, false);
        let (same, tail) = run_pointwise(&device, which, &x, dims, true);
        let mismatched = mismatched_bits(&same, &fresh);
        let ok = mismatched == 0 && tail;
        eprintln!(
            "gemma soft_cap_bf16 rows={rows} cols={cols} out==x: mismatched_bits={mismatched} \
             tail_intact={tail} {}",
            verdict(ok)
        );
        pass &= ok;
    }
    assert!(
        pass,
        "the in-place soft-cap differs from the out-of-place one"
    );
}

#[test]
fn gemma_rms_norm_unit_bf16_is_the_unit_gain_norm() {
    let device = h100();
    let mut pass = true;
    for cols in [256usize, 512] {
        for rows in [1usize, 16, 4096 * 8] {
            let x = bf16_inputs(rows * cols, 0x0A11_0000 ^ (rows as u64) << 32 ^ cols as u64);
            let dims = (rows as u32, cols as u32);
            let (gpu, tail) = run_pointwise(&device, Pointwise::RmsNormUnit, &x, dims, false);
            let ones = vec![0x3F80u16; cols];
            let (gained, _) = run_norm(&device, &x, &ones, (rows, cols), EPS, SCALE);
            let mismatched = mismatched_bits(&gpu, &gained);
            let refs = rms_norm_reference(&x, &ones, (rows, cols), f64::from(EPS), 0.0);
            let cmp = compare_rows(&gpu, &refs, cols);
            let ok = mismatched == 0 && cmp.pass() && tail;
            eprintln!(
                "gemma rms_norm_unit_bf16 rows={rows} cols={cols}: vs_ones_gain \
                 mismatched_bits={mismatched} vs_f64 {} tail_intact={tail} {}",
                cmp.line(),
                verdict(ok)
            );
            pass &= ok;
            if rows == 16 {
                // Self-check: the norm with a random gain must be out of bound on at least 90%.
                let g = bf16_inputs(cols, 0x0A11_0001);
                let wrong = rms_norm_reference(&x, &g, (rows, cols), f64::from(EPS), 0.0);
                let (frac, changed) =
                    out_of_bound_fraction(&gpu, (&refs, &wrong), |_, r| row_bound(r));
                let ok = frac >= 0.9;
                eprintln!(
                    "gemma rms_norm_unit self-check random gain cols={cols}: \
                     out_of_bound_fraction={frac:.4} changed={changed} {}",
                    verdict(ok)
                );
                pass &= ok;
            }
        }
    }
    assert!(pass, "the unit norm is not the unit-gain norm");
}

/// Run the GeGLU on `gate` and `up` over `dims`; the output and whether the tail past it held.
fn run_gelu(device: &CudaDevice, gate: &[u16], up: &[u16], dims: (u32, u32)) -> (Vec<u16>, bool) {
    let (gb, ub) = (
        DeviceBuffer::upload(device, gate).expect("upload gate"),
        DeviceBuffer::upload(device, up).expect("upload up"),
    );
    let out = output(device, gate.len());
    // SAFETY: cudaMalloc pointers are 256-byte aligned; each buffer holds rows·cols elements;
    // `out` is its own allocation; null is the default stream.
    unsafe { gelu_mul_bf16(gb.ptr(), ub.ptr(), out.ptr(), dims, std::ptr::null_mut()) }
        .expect("gelu_mul_bf16 launch");
    result(&out, gate.len())
}

/// The GeGLU's bound at element `i` of `gate` and `up`.
fn gelu_bound<'a>(gate: &'a [f64], up: &'a [f64]) -> impl Fn(usize, f64) -> f64 + 'a {
    move |i, r| gelu_mul_bound(r, gate[i], up[i])
}

/// `erf(x)`, Abramowitz and Stegun 7.1.26 (absolute error under 1.5e-7, far below the 1e-3 the
/// two GELUs differ by): the self-check's exact GELU.
fn erf(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.327_591_1 * x.abs());
    let poly = t
        * (0.254_829_592
            + t * (-0.284_496_736
                + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    (1.0 - poly * (-x * x).exp()).copysign(x)
}

#[test]
fn gemma_gelu_mul_bf16_matches_the_f64_reference() {
    let device = h100();
    let mut pass = true;
    for (rows, cols) in [(1usize, 2112usize), (512, 2112), (49024, 704)] {
        let seed = (rows as u64) << 32 | cols as u64;
        let gate = bf16_in(rows * cols, 0x6E1D_0000 ^ seed, (-8.0, 8.0));
        let up = bf16_inputs(rows * cols, 0x6E1D_FFFF ^ seed);
        let (gw, uw) = (widen(&gate), widen(&up));
        let (gpu, tail) = run_gelu(&device, &gate, &up, (rows as u32, cols as u32));
        let refs = gelu_mul_reference(&gate, &up);
        let cmp = compare_rows_with(&gpu, &refs, cols, gelu_bound(&gw, &uw));
        let ok = cmp.pass() && tail;
        eprintln!(
            "gemma gelu_mul_bf16 rows={rows} cols={cols}: {} tail_intact={tail} {}",
            cmp.line(),
            verdict(ok)
        );
        pass &= ok;
    }
    // Self-check on gates in [-3, 3]: the erf GELU's fraction is reported; below 90% (the two
    // GELUs differ by about 1e-3 there, under the tanh term of the bound), SiLU replaces it.
    let (rows, cols) = (16usize, 2112usize);
    let gate = bf16_in(rows * cols, 0x6E1D_5E1F, (-3.0, 3.0));
    let up = bf16_inputs(rows * cols, 0x6E1D_5E20);
    let (gw, uw) = (widen(&gate), widen(&up));
    let (gpu, _) = run_gelu(&device, &gate, &up, (rows as u32, cols as u32));
    let refs = gelu_mul_reference(&gate, &up);
    let erf_gelu: Vec<f64> = (gw.iter().zip(&uw))
        .map(|(g, u)| 0.5 * g * (1.0 + erf(g / std::f64::consts::SQRT_2)) * u)
        .collect();
    let (erf_frac, erf_changed) =
        out_of_bound_fraction(&gpu, (&refs, &erf_gelu), gelu_bound(&gw, &uw));
    let silu = silu_mul_reference(&gate, &up);
    let (silu_frac, silu_changed) =
        out_of_bound_fraction(&gpu, (&refs, &silu), gelu_bound(&gw, &uw));
    let (which, frac) = if erf_frac >= 0.9 {
        ("erf GELU", erf_frac)
    } else {
        ("SiLU (the erf GELU is below 90%)", silu_frac)
    };
    let ok = frac >= 0.9;
    eprintln!(
        "gemma gelu_mul self-check: erf_gelu out_of_bound_fraction={erf_frac:.4} \
         changed={erf_changed}; silu out_of_bound_fraction={silu_frac:.4} changed={silu_changed}; \
         checked with {which} {}",
        verdict(ok)
    );
    pass &= ok;
    assert!(pass, "a Gemma GeGLU shape is out of bound");
}

/// `(7, 2112)` with row 3 holding the gates ±{0, 1e-3, 3, 10, 60, 1e4, 3e38} against
/// `u ∈ {±1, ±0.5}`: a finite gate never gives NaN or Inf, and `g ≤ −10` gives ±0 or a value
/// within the bound.
#[test]
fn gemma_gelu_mul_bf16_extreme_gates() {
    const COLS: usize = 2112;
    const ROWS: usize = 7;
    const EXTREME: usize = 3;
    const MAGS: [f32; 7] = [0.0, 1e-3, 3.0, 10.0, 60.0, 1e4, 3e38];
    const UPS: [f32; 4] = [1.0, -1.0, 0.5, -0.5];
    let device = h100();
    let gates: Vec<f32> = MAGS.iter().flat_map(|&m| [m, -m]).collect();
    let mut gate = bf16_inputs(ROWS * COLS, 0x6E1D_0007);
    let mut up = bf16_inputs(ROWS * COLS, 0x6E1D_0070);
    let at = EXTREME * COLS;
    for c in 0..COLS {
        gate[at + c] = bits(gates[c % gates.len()]);
        up[at + c] = bits(UPS[(c / gates.len()) % UPS.len()]);
    }
    let (gw, uw) = (widen(&gate), widen(&up));
    let (gpu, tail) = run_gelu(&device, &gate, &up, (ROWS as u32, COLS as u32));
    let refs = gelu_mul_reference(&gate, &up);
    let cmp = compare_rows_with(&gpu, &refs, COLS, gelu_bound(&gw, &uw));
    let mut pass = cmp.pass() && tail;
    eprintln!(
        "gemma gelu_mul_bf16 rows={ROWS} cols={COLS} (extreme row {EXTREME}): {} \
         tail_intact={tail} {}",
        cmp.line(),
        verdict(pass)
    );
    let row = &gpu[at..at + COLS];
    let finite = row.iter().all(|&b| half::bf16::from_bits(b).is_finite());
    let very_negative_ok = (0..COLS).filter(|&c| gw[at + c] <= -10.0).all(|c| {
        let g = f64::from(half::bf16::from_bits(row[c]).to_f32());
        g == 0.0 || (g - refs[at + c]).abs() <= gelu_mul_bound(refs[at + c], gw[at + c], uw[at + c])
    });
    eprintln!(
        "gemma extreme gates: no NaN or Inf {finite}, g <= -10 gives ±0 or within the bound \
         {very_negative_ok} {}",
        verdict(finite && very_negative_ok)
    );
    pass &= finite && very_negative_ok;
    assert!(pass, "the GeGLU extreme row failed");
}

#[test]
fn gemma_pointwise_launchers_refuse_bad_shapes() {
    let device = h100();
    let mut pass = true;
    for (rows, cols) in REFUSED {
        let len = if rows as u64 * cols as u64 > i32::MAX as u64 {
            8
        } else {
            (rows as usize * cols as usize).clamp(8, 1 << 20)
        };
        let input = DeviceBuffer::upload(&device, &vec![0; len]).expect("upload an input");
        for kernel in [
            "scale_bf16",
            "layer_scale_bf16",
            "soft_cap_bf16",
            "rms_norm_unit_bf16",
            "gelu_mul_bf16",
        ] {
            let out = DeviceBuffer::upload(&device, &vec![SENTINEL; len]).expect("upload out");
            let (x, o, null) = (input.ptr(), out.ptr(), std::ptr::null_mut());
            let dims = (rows, cols);
            // SAFETY: each launcher refuses these shapes before it touches any pointer.
            let got = unsafe {
                match kernel {
                    "scale_bf16" => scale_bf16(x, o, dims, 2.0, null),
                    "layer_scale_bf16" => layer_scale_bf16(x, x, o, dims, null),
                    "soft_cap_bf16" => soft_cap_bf16(x, o, dims, GEMMA_CAP, null),
                    "rms_norm_unit_bf16" => rms_norm_unit_bf16(x, o, dims, EPS, null),
                    _ => gelu_mul_bf16(x, x, o, dims, null),
                }
            };
            let unchanged = (out.download().expect("download out").iter()).all(|&b| b == SENTINEL);
            let ok = got == Err(KernelError::RowShape { rows, cols }) && unchanged;
            eprintln!(
                "gemma {kernel} rows={rows} cols={cols}: {got:?} out_unchanged={unchanged} {}",
                verdict(ok)
            );
            pass &= ok;
        }
    }
    assert!(
        pass,
        "a Gemma row launcher did not refuse a shape it cannot run"
    );
}

/// Each new kernel run twice gives the same bits, at `(7, 704)`, whose last column tile is
/// partial (704 = 2 · 256 + 192): the sentinel past the output stays intact.
#[test]
fn gemma_pointwise_kernels_repeat_bit_identically() {
    let device = h100();
    let mut pass = true;
    let (rows, cols) = (7usize, 704usize);
    let dims = (rows as u32, cols as u32);
    let x = logits(rows * cols, 0x6E1D_2000);
    let w = bf16_inputs(1, 0x6E1D_2001)[0];
    for which in [
        Pointwise::Scale(gemma_scale()),
        Pointwise::LayerScale(w),
        Pointwise::SoftCap(GEMMA_CAP),
        Pointwise::RmsNormUnit,
    ] {
        let (a, tail_a) = run_pointwise(&device, which, &x, dims, false);
        let (b, tail_b) = run_pointwise(&device, which, &x, dims, false);
        let mismatched = mismatched_bits(&a, &b);
        let ok = mismatched == 0 && tail_a && tail_b;
        eprintln!(
            "gemma {which:?} rows={rows} cols={cols} repeat mismatched_bits={mismatched} \
             tail_intact={} {}",
            tail_a && tail_b,
            verdict(ok)
        );
        pass &= ok;
    }
    let up = bf16_inputs(rows * cols, 0x6E1D_2002);
    let (a, tail_a) = run_gelu(&device, &x, &up, dims);
    let (b, tail_b) = run_gelu(&device, &x, &up, dims);
    let mismatched = mismatched_bits(&a, &b);
    let ok = mismatched == 0 && tail_a && tail_b;
    eprintln!(
        "gemma GeluMul rows={rows} cols={cols} repeat mismatched_bits={mismatched} tail_intact={} {}",
        tail_a && tail_b,
        verdict(ok)
    );
    pass &= ok;
    assert!(pass, "a Gemma row kernel's repeat differs");
}
