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
use scratchy_target_cuda::kernels::{KernelError, add_bf16, rms_norm_bf16, silu_mul_bf16};
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
