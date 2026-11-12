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
    pub mod watchdog;
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
use scratchy_target_cuda::kernels::{add_rms_norm_bf16, norm_add_bf16, rms_norm_one_row_variant};
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
    common::watchdog::arm();
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
    // One KV head per row, as the tape runs Gemma's v_norm: the sliding layers' `(8m, 256)` and
    // the global layers' `(2m, 512)`, up to m = 4096.
    for (cols, rows_set) in [
        (256usize, [1usize, 16, 8, 4096 * 8]),
        (512, [1, 16, 2, 4096 * 2]),
    ] {
        for rows in rows_set {
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

/// The two launches #1049's add-norm folds replace, run one after the other on the default stream:
/// [`add_bf16`] of `a` and `b`, then [`rms_norm_bf16`] of the sum (`ADD_NORM`), or
/// [`rms_norm_bf16`] of `a`, then [`add_bf16`] of its output and `b` (`NORM_ADD`). Returns the
/// chain's two outputs: the sum (or the norm) and the norm (or the sum).
fn run_chain(
    device: &CudaDevice,
    norm_first: bool,
    (a, b, g): (&[u16], &[u16], &[u16]),
    dims: (u32, u32),
) -> (Vec<u16>, Vec<u16>) {
    let up = |v: &[u16]| DeviceBuffer::upload(device, v).expect("upload");
    let (ab, bb, gb) = (up(a), up(b), up(g));
    let (first, second) = (output(device, a.len()), output(device, a.len()));
    let null = std::ptr::null_mut();
    // SAFETY: cudaMalloc pointers are 256-byte aligned; `a`, `b`, `first` and `second` hold
    // rows·cols elements and `g` cols; each output is its own allocation; null is the default
    // stream, so the second launch reads the first's output after it is written.
    unsafe {
        if norm_first {
            rms_norm_bf16(ab.ptr(), gb.ptr(), first.ptr(), dims, EPS, SCALE, null)
                .expect("rms_norm_bf16 launch");
            add_bf16(first.ptr(), bb.ptr(), second.ptr(), dims, null).expect("add_bf16 launch");
        } else {
            add_bf16(ab.ptr(), bb.ptr(), first.ptr(), dims, null).expect("add_bf16 launch");
            rms_norm_bf16(
                first.ptr(),
                gb.ptr(),
                second.ptr(),
                dims,
                EPS,
                ONE_PLUS_SCALE,
                null,
            )
            .expect("rms_norm_bf16 launch");
        }
    }
    (result(&first, a.len()).0, result(&second, a.len()).0)
}

/// `add_rms_norm_bf16` writes the bits of `add_bf16` then `rms_norm_bf16` (#1049), both its
/// outputs, with each output's tail intact; and `norm_add_bf16` the bits of `rms_norm_bf16` then
/// `add_bf16`. At one row (decode), a 16-row block, a partial row block and a partial last
/// column tile.
#[test]
fn the_add_norm_folds_are_their_chains_bit_for_bit() {
    let device = h100();
    let mut pass = true;
    for (rows, cols) in [(1usize, 1024usize), (16, 2816), (7, 1000), (300, 4096)] {
        let seed = (rows as u64) << 32 | cols as u64;
        let a = bf16_inputs(rows * cols, 0x0FAD_0000 ^ seed);
        let b = bf16_inputs(rows * cols, 0x0FAD_1000 ^ seed);
        let dims = (rows as u32, cols as u32);
        let up = |v: &[u16]| DeviceBuffer::upload(&device, v).expect("upload");
        let null = std::ptr::null_mut();

        let g = centred_gains(cols, 0x0FAD_2000 ^ seed);
        let (sum_chain, norm_chain) = run_chain(&device, false, (&a, &b, &g), dims);
        let (ab, bb, gb) = (up(&a), up(&b), up(&g));
        let (sum, y) = (output(&device, a.len()), output(&device, a.len()));
        // SAFETY: as `run_chain`'s; `sum` and `y` are two allocations.
        unsafe {
            add_rms_norm_bf16(
                (ab.ptr(), bb.ptr()),
                gb.ptr(),
                (sum.ptr(), y.ptr()),
                dims,
                (EPS, ONE_PLUS_SCALE),
                null,
            )
        }
        .expect("add_rms_norm_bf16 launch");
        let ((sum, sum_tail), (y, y_tail)) = (result(&sum, a.len()), result(&y, a.len()));
        let (ms, my) = (
            mismatched_bits(&sum, &sum_chain),
            mismatched_bits(&y, &norm_chain),
        );
        let ok = ms == 0 && my == 0 && sum_tail && y_tail;
        eprintln!(
            "add_rms_norm_bf16 rows={rows} cols={cols}: sum mismatched_bits={ms} norm \
             mismatched_bits={my} tails_intact={} {}",
            sum_tail && y_tail,
            verdict(ok)
        );
        pass &= ok;

        let g = bf16_inputs(cols, 0x0FAD_3000 ^ seed);
        let (_, sum_chain) = run_chain(&device, true, (&a, &b, &g), dims);
        let (ab, bb, gb) = (up(&a), up(&b), up(&g));
        let out = output(&device, a.len());
        // SAFETY: as `run_chain`'s; `out` is its own allocation.
        unsafe {
            norm_add_bf16(
                (ab.ptr(), bb.ptr()),
                gb.ptr(),
                out.ptr(),
                dims,
                (EPS, SCALE),
                null,
            )
        }
        .expect("norm_add_bf16 launch");
        let (out, tail) = result(&out, a.len());
        let m = mismatched_bits(&out, &sum_chain);
        let ok = m == 0 && tail;
        eprintln!(
            "norm_add_bf16 rows={rows} cols={cols}: mismatched_bits={m} tail_intact={tail} {}",
            verdict(ok)
        );
        pass &= ok;
    }
    assert!(pass, "an add-norm fold differs from its chain");
}

// ── FP8 S8 W1: the one-row norms ───────────────────────────────────

/// The four norm forms, one launch each over `(rows, cols)`: `x` (and `b`, the addend or the
/// residual), `g`, `offset`. Returns the output(s) of row 0 and whether every tail held:
/// `[y]`, or `[sum, y]` for `ADD_NORM`.
#[derive(Clone, Copy, Debug, PartialEq)]
enum NormForm {
    Norm,
    Unit,
    AddNorm,
    NormAdd,
}

fn run_form(
    device: &CudaDevice,
    form: NormForm,
    (x, b, g): (&[u16], &[u16], &[u16]),
    (rows, cols): (usize, usize),
    offset: f32,
    pad: usize,
) -> (Vec<Vec<u16>>, bool) {
    // `pad` elements of SENTINEL before each input and output: the same row at another address.
    let up = |v: &[u16]| {
        let mut h = vec![SENTINEL; pad];
        h.extend_from_slice(v);
        DeviceBuffer::upload(device, &h).expect("upload")
    };
    let (xb, bb, gb) = (up(x), up(b), up(g));
    let n = rows * cols;
    let (y, sum) = (output(device, pad + n), output(device, pad + n));
    let at = |d: &DeviceBuffer| d.ptr().wrapping_byte_add(2 * pad);
    let dims = (rows as u32, cols as u32);
    let null = std::ptr::null_mut();
    // SAFETY: cudaMalloc pointers are 256-byte aligned and `pad` is a multiple of 8 bf16 (16
    // bytes); every buffer holds `pad` + rows·cols elements (gains `pad` + cols) past its start;
    // each output is its own allocation; null is the default stream.
    let r = unsafe {
        match form {
            NormForm::Norm => rms_norm_bf16(at(&xb), at(&gb), at(&y), dims, EPS, offset, null),
            NormForm::Unit => rms_norm_unit_bf16(at(&xb), at(&y), dims, EPS, null),
            NormForm::AddNorm => add_rms_norm_bf16(
                (at(&xb), at(&bb)),
                at(&gb),
                (at(&sum), at(&y)),
                dims,
                (EPS, offset),
                null,
            ),
            NormForm::NormAdd => norm_add_bf16(
                (at(&xb), at(&bb)),
                at(&gb),
                at(&y),
                dims,
                (EPS, offset),
                null,
            ),
        }
    };
    r.expect("a norm launch");
    let take = |d: &DeviceBuffer| {
        let (all, tail) = result(d, pad + n);
        let head = all[..pad].iter().all(|&v| v == SENTINEL);
        (all[pad..pad + cols].to_vec(), tail && head)
    };
    let (yr, yt) = take(&y);
    if form == NormForm::AddNorm {
        let (sr, st) = take(&sum);
        (vec![sr, yr], yt && st)
    } else {
        (vec![yr], yt)
    }
}

/// `row` as row 0 of `rows` rows, the others zero.
fn as_row0(row: &[u16], rows: usize) -> Vec<u16> {
    let mut v = row.to_vec();
    v.resize(rows * row.len(), 0);
    v
}

/// The widths of the three FP8 models' norms (27B 5,120 and its 6,144 GDN out; 35B 2,048;
/// Gemma 2,816 and its 256/512 head norms; the 128-wide q/k norms), a power of two each side, and
/// two partial last tiles.
const ONE_ROW_COLS: [usize; 12] = [
    128, 256, 512, 704, 1024, 2048, 2112, 2816, 4096, 5120, 6144, 8192,
];

/// FP8 S8 W1 Step A: at `rows == 1` every norm form (the norm, the unit norm, `ADD_NORM` with its
/// sum, `NORM_ADD`) takes the one-row kernel (`col_tiles` CTAs), and its output is the 16-row
/// kernel's bits for the same row: the row as row 0 of a 16-row launch, and of a 2-row launch,
/// both today's kernel. At every width in [`ONE_ROW_COLS`], gain offsets 0 and 1, 3 seeds, plus an
/// all-zero row and a row of ±65,504 (whose squares overflow to inf, as today). Every output's
/// tail is intact.
#[test]
fn the_one_row_norms_are_the_16_row_norms_bit_for_bit() {
    let device = h100();
    let mut pass = true;
    let forms = [
        NormForm::Norm,
        NormForm::Unit,
        NormForm::AddNorm,
        NormForm::NormAdd,
    ];
    for cols in ONE_ROW_COLS {
        let mut rows_in: Vec<(String, Vec<u16>, Vec<u16>)> = (0..3u64)
            .map(|s| {
                let seed = 0x0E01_0000 ^ (cols as u64) << 8 ^ s;
                (
                    format!("seed{s}"),
                    bf16_inputs(cols, seed),
                    bf16_inputs(cols, seed ^ 0x55),
                )
            })
            .collect();
        rows_in.push(("zeros".into(), vec![0; cols], vec![0; cols]));
        let big: Vec<u16> = (0..cols)
            .map(|c| bits(if c % 2 == 0 { 65504.0 } else { -65504.0 }))
            .collect();
        rows_in.push(("max".into(), big.clone(), big));
        for (what, x, b) in &rows_in {
            for offset in [SCALE, ONE_PLUS_SCALE] {
                let g = match offset {
                    SCALE => bf16_inputs(cols, 0x0E02_0000 ^ cols as u64),
                    _ => centred_gains(cols, 0x0E03_0000 ^ cols as u64),
                };
                for form in forms {
                    if form == NormForm::Unit && offset != SCALE {
                        continue;
                    }
                    let (one, t1) = run_form(&device, form, (x, b, &g), (1, cols), offset, 0);
                    let mut ok = t1;
                    let mut line = String::new();
                    for rows in [16usize, 2] {
                        let (want, t) = run_form(
                            &device,
                            form,
                            (&as_row0(x, rows), &as_row0(b, rows), &g),
                            (rows, cols),
                            offset,
                            0,
                        );
                        let m: usize = one
                            .iter()
                            .zip(&want)
                            .map(|(o, w)| mismatched_bits(o, w))
                            .sum();
                        line += &format!(" vs rows={rows} mismatched_bits={m}");
                        ok &= t && m == 0;
                    }
                    eprintln!(
                        "one-row {form:?} cols={cols} offset={offset} {what}:{line} tails_intact={t1} {}",
                        verdict(ok)
                    );
                    pass &= ok;
                }
            }
        }
    }
    assert!(pass, "a one-row norm differs from the 16-row norm");
}

/// The one-row norms repeat bit for bit, and give the same bits at another buffer address (the
/// inputs and outputs shifted by 64 elements), at the three models' widths.
#[test]
fn the_one_row_norms_repeat_and_ignore_placement() {
    let device = h100();
    let mut pass = true;
    for cols in [2048usize, 2816, 5120] {
        let x = bf16_inputs(cols, 0x0E04_0000 ^ cols as u64);
        let b = bf16_inputs(cols, 0x0E05_0000 ^ cols as u64);
        let g = centred_gains(cols, 0x0E06_0000 ^ cols as u64);
        for form in [
            NormForm::Norm,
            NormForm::Unit,
            NormForm::AddNorm,
            NormForm::NormAdd,
        ] {
            let first = run_form(&device, form, (&x, &b, &g), (1, cols), ONE_PLUS_SCALE, 0);
            let again = run_form(&device, form, (&x, &b, &g), (1, cols), ONE_PLUS_SCALE, 0);
            let moved = run_form(&device, form, (&x, &b, &g), (1, cols), ONE_PLUS_SCALE, 64);
            let diff = |a: &[Vec<u16>], b: &[Vec<u16>]| -> usize {
                a.iter().zip(b).map(|(x, y)| mismatched_bits(x, y)).sum()
            };
            let (r, p) = (diff(&first.0, &again.0), diff(&first.0, &moved.0));
            let ok = r == 0 && p == 0 && first.1 && again.1 && moved.1;
            eprintln!(
                "one-row {form:?} cols={cols}: repeat mismatched_bits={r} moved mismatched_bits={p} {}",
                verdict(ok)
            );
            pass &= ok;
        }
    }
    assert!(
        pass,
        "a one-row norm did not repeat, or depends on its address"
    );
}

/// `cudaEvent_t`, declared here for the timing lines only (the runtime declares no event call).
type CudaEvent = *mut std::ffi::c_void;

unsafe extern "C" {
    fn cudaEventCreate(event: *mut CudaEvent) -> i32;
    fn cudaEventRecord(event: CudaEvent, stream: *mut std::ffi::c_void) -> i32;
    fn cudaEventSynchronize(event: CudaEvent) -> i32;
    fn cudaEventElapsedTime(ms: *mut f32, start: CudaEvent, end: CudaEvent) -> i32;
    fn cudaEventDestroy(event: CudaEvent) -> i32;
    fn cudaMemsetAsync(
        dst: *mut std::ffi::c_void,
        value: i32,
        bytes: usize,
        stream: *mut std::ffi::c_void,
    ) -> i32;
}

/// The bytes cleared before a cold run: twice the H100's 50 MiB L2.
const SCRUB_BYTES: usize = 128 << 20;

/// The median µs of 20 launches after 3 warm-ups, each between two CUDA events on the default
/// stream; `scrub` (when given) cleared before each, so the row is read from DRAM, not L2.
fn event_median(scrub: Option<&DeviceBuffer>, mut launch: impl FnMut()) -> f64 {
    let (mut start, mut end): (CudaEvent, CudaEvent) = (std::ptr::null_mut(), std::ptr::null_mut());
    // SAFETY: plain runtime calls on the default stream with out-pointers to locals; each event
    // is destroyed below.
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
    times[times.len() / 2]
}

/// Recorded, not asserted (W1's prototype gate reads it): µs per launch of each norm form at
/// `rows == 1` (the one-row kernel) and at `rows == 2` (today's 16-row kernel on one CTA, the
/// time W0 measured in the tape), hot and with L2 cleared, at the three models' widths.
#[test]
fn the_one_row_norm_times_are_recorded() {
    let device = h100();
    let scrub = DeviceBuffer::upload_bytes(&device, &vec![0u8; SCRUB_BYTES]).expect("scrub");
    for cols in [2048usize, 2816, 5120, 6144] {
        let x = bf16_inputs(2 * cols, 0x0E07_0000 ^ cols as u64);
        let b = bf16_inputs(2 * cols, 0x0E08_0000 ^ cols as u64);
        let g = centred_gains(cols, 0x0E09_0000 ^ cols as u64);
        let up = |v: &[u16]| DeviceBuffer::upload(&device, v).expect("upload");
        let (xb, bb, gb) = (up(&x), up(&b), up(&g));
        let (y, sum) = (output(&device, 2 * cols), output(&device, 2 * cols));
        for form in [NormForm::Norm, NormForm::AddNorm, NormForm::NormAdd] {
            let mut line = format!("norm time {form:?} cols={cols}:");
            for rows in [1u32, 2] {
                let dims = (rows, cols as u32);
                let launch = || {
                    let null = std::ptr::null_mut();
                    // SAFETY: as `run_form`'s with no padding; every buffer holds 2·cols.
                    let r = unsafe {
                        match form {
                            NormForm::Norm => rms_norm_bf16(
                                xb.ptr(),
                                gb.ptr(),
                                y.ptr(),
                                dims,
                                EPS,
                                ONE_PLUS_SCALE,
                                null,
                            ),
                            NormForm::AddNorm => add_rms_norm_bf16(
                                (xb.ptr(), bb.ptr()),
                                gb.ptr(),
                                (sum.ptr(), y.ptr()),
                                dims,
                                (EPS, ONE_PLUS_SCALE),
                                null,
                            ),
                            _ => norm_add_bf16(
                                (xb.ptr(), bb.ptr()),
                                gb.ptr(),
                                y.ptr(),
                                dims,
                                (EPS, SCALE),
                                null,
                            ),
                        }
                    };
                    r.expect("a timed norm launch");
                };
                let hot = event_median(None, launch);
                let cold = event_median(Some(&scrub), launch);
                let kernel = if rows == 1 { "one-row" } else { "16-row" };
                line += &format!(" rows={rows} ({kernel}) hot {hot:.2} us cold {cold:.2} us;");
            }
            eprintln!("{line}");
        }
    }
}

/// The rows of the large one-row bit test, per width: 400 rows of the five model widths is about
/// 8.1 M outputs per form, so a 2-per-million flip (S5's d512 signature) shows.
const ONE_ROW_MANY: usize = 400;

/// FP8 S8 W1: at the three FP8 models' widths (2,048, 2,816, 4,096, 5,120, 6,144), every form of
/// [`ONE_ROW_MANY`] distinct random rows, each launched alone at `rows == 1` (the one-row kernel)
/// against the same rows as one `rows == 400` launch (today's 16-row kernel), row for row:
/// `mismatched_bits=0` over about 8 M outputs a form, `ADD_NORM`'s sum included. One upload per
/// width and form; each one-row launch reads its row in place, at its offset in that upload.
#[test]
fn the_one_row_norms_match_the_16_row_norms_over_millions_of_outputs() {
    let device = h100();
    let mut pass = true;
    let null = std::ptr::null_mut();
    for form in [
        NormForm::Norm,
        NormForm::Unit,
        NormForm::AddNorm,
        NormForm::NormAdd,
    ] {
        let (mut outputs, mut mismatched) = (0usize, 0usize);
        for cols in [2048usize, 2816, 4096, 5120, 6144] {
            let n = ONE_ROW_MANY * cols;
            let seed = 0x0E0A_0000 ^ (cols as u64) << 4 ^ form as u64;
            let up = |v: &[u16]| DeviceBuffer::upload(&device, v).expect("upload");
            let (x, b) = (bf16_inputs(n, seed), bf16_inputs(n, seed ^ 0x5A5A));
            let g = centred_gains(cols, seed ^ 0xA5A5);
            let (xb, bb, gb) = (up(&x), up(&b), up(&g));
            // Today's kernel over all rows at once, and the one-row kernel once per row.
            let (y16, s16) = (output(&device, n), output(&device, n));
            let (y1, s1) = (output(&device, n), output(&device, n));
            let at = |d: &DeviceBuffer, r: usize| d.ptr().wrapping_byte_add(2 * r * cols);
            let launch = |rows: usize, r: usize, (y, s): (&DeviceBuffer, &DeviceBuffer)| {
                let dims = (rows as u32, cols as u32);
                // SAFETY: cudaMalloc pointers are 256-byte aligned and every row starts at a
                // multiple of 2·cols bytes (cols a multiple of 256), so 16-byte aligned; rows
                // `r .. r + rows` lie inside each `ONE_ROW_MANY · cols` buffer; each output is its
                // own allocation; null is the default stream.
                let res = unsafe {
                    match form {
                        NormForm::Norm => rms_norm_bf16(
                            at(&xb, r),
                            gb.ptr(),
                            at(y, r),
                            dims,
                            EPS,
                            ONE_PLUS_SCALE,
                            null,
                        ),
                        NormForm::Unit => rms_norm_unit_bf16(at(&xb, r), at(y, r), dims, EPS, null),
                        NormForm::AddNorm => add_rms_norm_bf16(
                            (at(&xb, r), at(&bb, r)),
                            gb.ptr(),
                            (at(s, r), at(y, r)),
                            dims,
                            (EPS, ONE_PLUS_SCALE),
                            null,
                        ),
                        NormForm::NormAdd => norm_add_bf16(
                            (at(&xb, r), at(&bb, r)),
                            gb.ptr(),
                            at(y, r),
                            dims,
                            (EPS, SCALE),
                            null,
                        ),
                    }
                };
                res.expect("a norm launch");
            };
            launch(ONE_ROW_MANY, 0, (&y16, &s16));
            for r in 0..ONE_ROW_MANY {
                launch(1, r, (&y1, &s1));
            }
            let (want, wt) = result(&y16, n);
            let (got, gt) = result(&y1, n);
            let mut m = mismatched_bits(&got, &want);
            let mut tails = wt && gt;
            outputs += n;
            if form == NormForm::AddNorm {
                let (sw, swt) = result(&s16, n);
                let (sg, sgt) = result(&s1, n);
                m += mismatched_bits(&sg, &sw);
                tails &= swt && sgt;
                outputs += n;
            }
            mismatched += m;
            let ok = m == 0 && tails;
            eprintln!(
                "one-row {form:?} cols={cols}: {ONE_ROW_MANY} rows, mismatched_bits={m} \
                 tails_intact={tails} {}",
                verdict(ok)
            );
            pass &= ok;
        }
        eprintln!(
            "one-row {form:?}: {outputs} outputs, mismatched_bits={mismatched} {}",
            verdict(mismatched == 0)
        );
    }
    assert!(
        pass,
        "a one-row norm differs from the 16-row norm on some row"
    );
}

// ── FP8 S8 W1: the one-row pointwise maps ──────────────────────────

/// FP8 S8 W1: at `rows == 1` the scale, the layer scale and the soft-cap take their one-row form
/// (one CTA per 4 column tiles), and the row is the 16-row kernel's bits, as row 0 of a 16-row and
/// a 2-row launch: the soft-cap at Gemma's 262,144 logits and at 2,816, in place and not, over
/// logits in ±64 and with ±inf, NaN and ±65,504; the layer scale and the scale at Gemma's 2,816
/// (and 704, a partial last group of tiles), 3 seeds.
#[test]
fn the_one_row_maps_are_the_16_row_maps_bit_for_bit() {
    let device = h100();
    let mut pass = true;
    let extremes = |cols: usize| -> Vec<u16> {
        let special = [
            bits(f32::INFINITY),
            bits(f32::NEG_INFINITY),
            0x7FC0,
            bits(65504.0),
            bits(-65504.0),
            0x8000,
            0,
        ];
        (0..cols).map(|c| special[c % special.len()]).collect()
    };
    let cases = [
        ("soft_cap", Pointwise::SoftCap(GEMMA_CAP), GEMMA_VOCAB),
        ("soft_cap", Pointwise::SoftCap(GEMMA_CAP), GEMMA_HIDDEN),
        (
            "layer_scale",
            Pointwise::LayerScale(bits(0.3125)),
            GEMMA_HIDDEN,
        ),
        ("layer_scale", Pointwise::LayerScale(bits(0.3125)), 704),
        ("scale", Pointwise::Scale(gemma_scale()), GEMMA_HIDDEN),
        ("scale", Pointwise::Scale(gemma_scale()), 704),
    ];
    for (what, which, cols) in cases {
        let mut rows_in: Vec<(String, Vec<u16>)> = (0..3u64)
            .map(|s| {
                (
                    format!("seed{s}"),
                    logits(cols, 0x0E31_0000 ^ (cols as u64) << 4 ^ s),
                )
            })
            .collect();
        rows_in.push(("extremes".into(), extremes(cols)));
        for (label, x) in &rows_in {
            for in_place in [false, true] {
                let (one, t1) = run_pointwise(&device, which, x, (1, cols as u32), in_place);
                let mut ok = t1;
                let mut line = String::new();
                for rows in [16usize, 2] {
                    let mut x16 = x.clone();
                    x16.resize(rows * cols, 0);
                    let (want, t) =
                        run_pointwise(&device, which, &x16, (rows as u32, cols as u32), in_place);
                    let m = mismatched_bits(&one, &want[..cols]);
                    line += &format!(" vs rows={rows} mismatched_bits={m}");
                    ok &= t && m == 0;
                }
                eprintln!(
                    "one-row {what} cols={cols} {label} in_place={in_place}:{line} tail_intact={t1} {}",
                    verdict(ok)
                );
                pass &= ok;
            }
        }
    }
    assert!(pass, "a one-row pointwise map differs from the 16-row map");
}

/// Recorded, not asserted: µs per launch of the soft-cap, the layer scale and the scale at
/// `rows == 1` (the one-row form) and `rows == 2` (today's kernel on one CTA, the time W0
/// measured), hot.
#[test]
fn the_one_row_map_times_are_recorded() {
    let device = h100();
    let null = std::ptr::null_mut();
    for (what, which, cols) in [
        ("soft_cap", Pointwise::SoftCap(GEMMA_CAP), GEMMA_VOCAB),
        (
            "layer_scale",
            Pointwise::LayerScale(bits(0.3125)),
            GEMMA_HIDDEN,
        ),
        ("scale", Pointwise::Scale(gemma_scale()), GEMMA_HIDDEN),
    ] {
        let xb = DeviceBuffer::upload(&device, &logits(2 * cols, 0x0E32_0000)).expect("upload x");
        let out = output(&device, 2 * cols);
        let wb = DeviceBuffer::upload(&device, &[bits(0.3125); 8]).expect("upload w");
        let mut line = format!("map time {what} cols={cols}:");
        for rows in [1u32, 2] {
            let dims = (rows, cols as u32);
            let us = event_median(None, || {
                // SAFETY: as `run_pointwise`'s; `x` and `out` hold 2 rows, `w` 8 bf16.
                unsafe {
                    match which {
                        Pointwise::Scale(s) => scale_bf16(xb.ptr(), out.ptr(), dims, s, null),
                        Pointwise::LayerScale(_) => {
                            layer_scale_bf16(xb.ptr(), wb.ptr(), out.ptr(), dims, null)
                        }
                        Pointwise::SoftCap(c) => soft_cap_bf16(xb.ptr(), out.ptr(), dims, c, null),
                        Pointwise::RmsNormUnit => unreachable!("not timed here"),
                    }
                }
                .expect("a timed map launch");
            });
            let kernel = if rows == 1 { "one-row" } else { "16-row" };
            line += &format!(" rows={rows} ({kernel}) {us:.2} us;");
        }
        eprintln!("{line}");
    }
}

// ── FP8 S8 N1: the row norm against today's one-row norm ─────────────────────

/// One row's buffers for [`rms_norm_one_row_variant`]: `x`, `b` (the addend or the residual), the
/// gain, and both variants' `y` and `sum`.
struct N1Row {
    x: DeviceBuffer,
    b: DeviceBuffer,
    g: DeviceBuffer,
    out: [(DeviceBuffer, DeviceBuffer); 2],
    cols: usize,
}

impl N1Row {
    fn new(device: &CudaDevice, (x, b, g): (&[u16], &[u16], &[u16])) -> Self {
        let up = |v: &[u16]| DeviceBuffer::upload(device, v).expect("upload");
        let cols = x.len();
        let pair = || (output(device, cols), output(device, cols));
        Self {
            x: up(x),
            b: up(b),
            g: up(g),
            out: [pair(), pair()],
            cols,
        }
    }

    /// Form `form` (the launcher's code) at `variant` into its own outputs, on `stream`.
    fn launch(&self, form: NormForm, offset: f32, variant: u32, stream: *mut std::ffi::c_void) {
        let (y, sum) = &self.out[variant as usize];
        // SAFETY: every buffer is its own cudaMalloc allocation (256-byte aligned) of at least
        // `cols` bf16 (`output`'s are longer), the outputs distinct; `stream` is null or a live
        // stream of this device.
        unsafe {
            rms_norm_one_row_variant(
                (self.x.ptr().cast_const(), self.g.ptr().cast_const()),
                (y.ptr(), self.b.ptr().cast_const(), sum.ptr()),
                self.cols as u32,
                (EPS, offset),
                (form as u32, variant),
                stream,
            )
        }
        .expect("a one-row norm variant launch");
    }

    /// `variant`'s `y` (and `sum` under `ADD_NORM`) and whether every tail held.
    fn outputs(&self, form: NormForm, variant: u32) -> (Vec<u16>, bool) {
        let (y, sum) = &self.out[variant as usize];
        let (mut v, mut tails) = result(y, self.cols);
        if form == NormForm::AddNorm {
            let (s, st) = result(sum, self.cols);
            v.extend(s);
            tails &= st;
        }
        (v, tails)
    }
}

/// FP8 S8 N1: the row kernel (variant 1, the launchers' at `rows == 1`) is today's one-row kernel
/// (variant 0) byte for byte, `y` and `ADD_NORM`'s sum, every form at the three models' widths
/// (2,048, 2,816, 5,120), gain offsets 0 and 1, on 3 random rows and 3 edge rows (all zeros, ±65,504
/// whose squares overflow, and subnormals beside the largest finite bf16), as `N1_BITS` lines.
#[test]
fn the_row_norm_is_the_one_row_norm_bit_for_bit() {
    let device = h100();
    let mut pass = true;
    for cols in [2048usize, 2816, 5120] {
        let mut rows_in: Vec<(String, Vec<u16>, Vec<u16>)> = (0..3u64)
            .map(|s| {
                let seed = 0x0E11_0000 ^ (cols as u64) << 8 ^ s;
                (
                    format!("random{s}"),
                    bf16_inputs(cols, seed),
                    bf16_inputs(cols, seed ^ 0x55),
                )
            })
            .collect();
        rows_in.push(("zeros".into(), vec![0; cols], vec![0; cols]));
        let big: Vec<u16> = (0..cols)
            .map(|c| bits(if c % 2 == 0 { 65504.0 } else { -65504.0 }))
            .collect();
        rows_in.push(("max".into(), big.clone(), big));
        let tiny: Vec<u16> = (0..cols)
            .map(|c| [0x0001, 0x8001, 0x7F7F, 0x0080][c % 4])
            .collect();
        rows_in.push(("subnormal".into(), tiny.clone(), tiny));
        for (what, x, b) in &rows_in {
            for offset in [SCALE, ONE_PLUS_SCALE] {
                let g = match offset {
                    SCALE => bf16_inputs(cols, 0x0E12_0000 ^ cols as u64),
                    _ => centred_gains(cols, 0x0E13_0000 ^ cols as u64),
                };
                let r = N1Row::new(&device, (x, b, &g));
                for form in [
                    NormForm::Norm,
                    NormForm::Unit,
                    NormForm::AddNorm,
                    NormForm::NormAdd,
                ] {
                    if form == NormForm::Unit && offset != SCALE {
                        continue;
                    }
                    let null = std::ptr::null_mut();
                    r.launch(form, offset, 0, null);
                    r.launch(form, offset, 1, null);
                    let (want, wt) = r.outputs(form, 0);
                    let (got, gt) = r.outputs(form, 1);
                    let m = mismatched_bits(&got, &want);
                    let ok = m == 0 && wt && gt;
                    eprintln!(
                        "N1_BITS {form:?} cols={cols} offset={offset} {what}: mismatched_bits={m} \
                         tails_intact={} {}",
                        wt && gt,
                        verdict(ok)
                    );
                    pass &= ok;
                }
            }
        }
    }
    assert!(pass, "the row norm is not today's one-row norm's bits");
}

const STREAM_BLOCKING: u32 = 0;
const CAPTURE_THREAD_LOCAL: i32 = 1;

unsafe extern "C" {
    fn cudaStreamCreateWithFlags(stream: *mut *mut std::ffi::c_void, flags: u32) -> i32;
    fn cudaStreamDestroy(stream: *mut std::ffi::c_void) -> i32;
    fn cudaStreamBeginCapture(stream: *mut std::ffi::c_void, mode: i32) -> i32;
    fn cudaStreamEndCapture(
        stream: *mut std::ffi::c_void,
        graph: *mut *mut std::ffi::c_void,
    ) -> i32;
    fn cudaGraphInstantiateWithFlags(
        exec: *mut *mut std::ffi::c_void,
        graph: *mut std::ffi::c_void,
        flags: u64,
    ) -> i32;
    fn cudaGraphLaunch(exec: *mut std::ffi::c_void, stream: *mut std::ffi::c_void) -> i32;
    fn cudaGraphExecDestroy(exec: *mut std::ffi::c_void) -> i32;
    fn cudaGraphDestroy(graph: *mut std::ffi::c_void) -> i32;
}

/// `launch(stream)` ten times captured into one graph on a blocking stream, replayed between
/// [`event_median`]'s events (the legacy default stream orders them), with L2 cleared before each
/// (`readcold`) and not (`hot`): one `TC_TIME` line each, the replay's median per node.
fn graph_record(
    scrub: &DeviceBuffer,
    (shape, imp, round): (&str, &str, u32),
    launch: impl Fn(*mut std::ffi::c_void),
) {
    let (mut stream, mut graph, mut exec) = (
        std::ptr::null_mut(),
        std::ptr::null_mut(),
        std::ptr::null_mut(),
    );
    // SAFETY: out-pointers to locals; nothing else is captured on this thread; every handle is
    // destroyed below.
    unsafe {
        assert_eq!(
            cudaStreamCreateWithFlags(&mut stream, STREAM_BLOCKING),
            0,
            "stream"
        );
        assert_eq!(
            cudaStreamBeginCapture(stream, CAPTURE_THREAD_LOCAL),
            0,
            "begin capture"
        );
        for _ in 0..10 {
            launch(stream);
        }
        assert_eq!(cudaStreamEndCapture(stream, &mut graph), 0, "end capture");
        assert_eq!(
            cudaGraphInstantiateWithFlags(&mut exec, graph, 0),
            0,
            "instantiate"
        );
    }
    let replay = || {
        // SAFETY: `exec` is the instantiated graph and `stream` live.
        assert_eq!(unsafe { cudaGraphLaunch(exec, stream) }, 0, "graph launch");
    };
    let cold = event_median(Some(scrub), replay);
    let hot = event_median(None, replay);
    for (temp, us) in [("readcold", cold), ("hot", hot)] {
        eprintln!(
            "TC_TIME {temp} {shape} {imp}_graph_per_node: median={:.2}us per node, 0.00 TB/s \
             (round {round})",
            us / 10.0
        );
    }
    // SAFETY: the handles were created above and are no longer used.
    unsafe {
        cudaGraphExecDestroy(exec);
        cudaGraphDestroy(graph);
        cudaStreamDestroy(stream);
    }
}

/// The decode norms of the three FP8 models: (model, width, form, gain offset).
const N1_SHAPES: &[(&str, usize, NormForm, f32)] = &[
    ("27B", 5120, NormForm::Norm, ONE_PLUS_SCALE),
    ("27B", 5120, NormForm::AddNorm, ONE_PLUS_SCALE),
    ("35B", 2048, NormForm::Norm, ONE_PLUS_SCALE),
    ("35B", 2048, NormForm::AddNorm, ONE_PLUS_SCALE),
    ("Gemma", 2816, NormForm::Norm, SCALE),
    ("Gemma", 2816, NormForm::NormAdd, SCALE),
];

/// Recorded, not asserted (N1's gate reads it): inside a captured graph of 10 back-to-back nodes,
/// today's one-row norm (`w1_norm`) against the row norm (`n1_norm`), read-cold and hot, two rounds
/// (A, B, A, B), as `TC_TIME … _graph_per_node` lines.
#[test]
fn the_row_norm_graph_times_are_recorded() {
    let device = h100();
    let scrub = DeviceBuffer::upload_bytes(&device, &vec![0u8; SCRUB_BYTES]).expect("scrub");
    for &(model, cols, form, offset) in N1_SHAPES {
        let x = bf16_inputs(cols, 0x0E14_0000 ^ cols as u64);
        let b = bf16_inputs(cols, 0x0E15_0000 ^ cols as u64);
        let g = centred_gains(cols, 0x0E16_0000 ^ cols as u64);
        let r = N1Row::new(&device, (&x, &b, &g));
        let shape = format!("{model}_{form:?}_{cols}");
        for round in 1..=2 {
            graph_record(&scrub, (&shape, "w1_norm", round), |s| {
                r.launch(form, offset, 0, s)
            });
            graph_record(&scrub, (&shape, "n1_norm", round), |s| {
                r.launch(form, offset, 1, s)
            });
        }
    }
}

// ── FP8 S8 N3: the decode norm in one CTA a row, with the next GEMM's quant ─────

/// N3's bound's relative term: the output's one RNE rounding to bf16 (an 8-bit significand,
/// `2⁻⁸·|y|`) plus `2⁻¹⁶` for the f32 arithmetic before it. That arithmetic is about `2⁻²⁰`: the sum
/// of squares is at most 18 dependent f32 roundings deep (8 a lane, 5 shuffles, 1 chunk add, 4 warp
/// levels: `≤ 18·2⁻²⁴`), `rsqrt` halves it and adds its own `~2⁻²³`, the two products `3·2⁻²⁴`;
/// `2⁻¹⁶` is 16× that.
const N3_REL: f64 = 1.0 / 256.0 + 1.0 / 65_536.0;
/// N3's absolute floor: the smallest normal f32, below which fast math flushes a result to zero.
const N3_ABS: f64 = 1.0 / (1u128 << 126) as f64;
/// N3's widths: one chunk, the three models' (35B, Gemma, 27B) and the widest.
const N3_COLS: [usize; 5] = [256, 2048, 2816, 5120, 8192];

/// N3's forms, the launcher's codes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum N3Form {
    Norm = 0,
    AddNorm = 1,
    NormAdd = 2,
}

/// One N3 case's device buffers: the inputs, the outputs (sentinels), the quant row and scales.
struct N3 {
    x: DeviceBuffer,
    other: DeviceBuffer,
    g: DeviceBuffer,
    y: DeviceBuffer,
    sum: DeviceBuffer,
    q: DeviceBuffer,
    scale: DeviceBuffer,
    rows: usize,
    cols: usize,
}

impl N3 {
    fn new(device: &CudaDevice, (x, other, g): (&[u16], &[u16], &[u16]), rows: usize) -> Self {
        let up = |v: &[u16]| DeviceBuffer::upload(device, v).expect("upload");
        let cols = g.len();
        Self {
            x: up(x),
            other: up(other),
            g: up(g),
            y: up(&vec![SENTINEL; rows * cols]),
            sum: up(&vec![SENTINEL; rows * cols]),
            q: DeviceBuffer::upload_bytes(device, &vec![0xA5u8; rows * cols]).expect("q"),
            scale: DeviceBuffer::upload_bytes(device, &vec![0xA5u8; 4 * rows]).expect("scale"),
            rows,
            cols,
        }
    }

    /// N3 at `form` (with its quant when `quant`), on `stream`.
    fn launch(&self, form: N3Form, quant: bool, offset: f32, stream: *mut std::ffi::c_void) {
        use scratchy_target_cuda::kernels::rms_norm_quant_row;
        // SAFETY: every buffer is its own cudaMalloc allocation holding the elements the launcher
        // names for `rows · cols`; the outputs are distinct; `stream` null or live.
        unsafe {
            rms_norm_quant_row(
                (
                    self.x.ptr().cast_const(),
                    self.other.ptr().cast_const(),
                    self.g.ptr().cast_const(),
                ),
                (self.y.ptr(), self.sum.ptr()),
                quant.then(|| (self.q.ptr(), self.scale.ptr())),
                (self.rows as u32, self.cols as u32),
                (form as u32, EPS, offset),
                stream,
            )
        }
        .expect("an N3 launch");
    }
}

/// The N3 input rows of `cols`: two random rows, all zeros, ±65,504 alternating, and tiny normals
/// (`±2⁻¹⁰⁰`, whose squares flush to zero in f32 while `eps` dominates the f64 mean too).
fn n3_rows(cols: usize, seed: u64) -> Vec<(&'static str, Vec<u16>)> {
    let alternate = |v: f32| -> Vec<u16> {
        (0..cols)
            .map(|c| bits(if c % 2 == 0 { v } else { -v }))
            .collect()
    };
    vec![
        ("random0", bf16_inputs(cols, seed)),
        ("random1", bf16_inputs(cols, seed ^ 0x77)),
        ("zeros", vec![0; cols]),
        ("max", alternate(65504.0)),
        ("tiny", alternate(2f32.powi(-100))),
    ]
}

/// FP8 S8 N3, the bound and the quant: every form at every width of [`N3_COLS`], gain offsets 0
/// and 1, one row and three rows (one CTA a row), on [`n3_rows`]' kinds:
/// - the normed row within `N3_REL · |ref| + N3_ABS` of the f64 norm (`rms_norm_reference`); under
///   `NormAdd`, `y` within `N3_REL · |n| + 2⁻⁸ · |n + r| + N3_ABS` of `n + r` (the second rounding);
/// - `AddNorm`'s sum the add's bits (`add_reference`), and the norm's reference taken on that sum;
/// - with the quant (`Norm`, `AddNorm`): the e4m3 bytes and the scale of each row
///   `quant_row_fp8_one` (`quant_rows_fp8`'s row, bit for bit) of N3's own bf16 row, bit for bit.
///
/// `N3_BOUND` and `N3_QUANT` lines.
#[test]
fn n3_is_inside_its_f64_bound_and_quantizes_as_quant_rows_fp8() {
    use scratchy_target_cuda::kernels::quant_row_fp8_one;
    let device = h100();
    let mut pass = true;
    for cols in N3_COLS {
        for (kind, row) in n3_rows(cols, 0x0E30_0000 ^ cols as u64) {
            for offset in [SCALE, ONE_PLUS_SCALE] {
                let g = match offset {
                    SCALE => bf16_inputs(cols, 0x0E31_0000 ^ cols as u64),
                    _ => centred_gains(cols, 0x0E32_0000 ^ cols as u64),
                };
                for rows in [1usize, 3] {
                    // Row 0 the kind; the others random.
                    let mut x = row.clone();
                    for r in 1..rows {
                        x.extend(bf16_inputs(cols, 0x0E33_0000 ^ (r * cols) as u64));
                    }
                    let other = bf16_inputs(rows * cols, 0x0E34_0000 ^ cols as u64);
                    for form in [N3Form::Norm, N3Form::AddNorm, N3Form::NormAdd] {
                        let quant = form != N3Form::NormAdd;
                        let c = N3::new(&device, (&x, &other, &g), rows);
                        c.launch(form, quant, offset, std::ptr::null_mut());
                        let y = c.y.download().expect("y");
                        let sum = c.sum.download().expect("sum");
                        let normed_in = match form {
                            N3Form::AddNorm => add_reference(&x, &other),
                            _ => x.clone(),
                        };
                        let sum_bad = match form {
                            N3Form::AddNorm => mismatched_bits(&sum, &normed_in),
                            _ => 0,
                        };
                        let n = rms_norm_reference(
                            &normed_in,
                            &g,
                            (rows, cols),
                            f64::from(EPS),
                            f64::from(offset),
                        );
                        let r = widen(&other);
                        let cmp = match form {
                            N3Form::NormAdd => {
                                let want: Vec<f64> = n.iter().zip(&r).map(|(a, b)| a + b).collect();
                                compare_rows_with(&y, &want, cols, |i, w| {
                                    N3_REL * n[i].abs() + w.abs() / 256.0 + N3_ABS
                                })
                            }
                            _ => compare_rows_with(&y, &n, cols, |_, w| N3_REL * w.abs() + N3_ABS),
                        };
                        let ok = cmp.out_of_bound == 0 && sum_bad == 0;
                        eprintln!(
                            "N3_BOUND {form:?} cols={cols} offset={offset} rows={rows} {kind}: \
                             out_of_bound={} worst={:?} sum_mismatched={sum_bad} {}",
                            cmp.out_of_bound,
                            cmp.worst,
                            verdict(ok)
                        );
                        pass &= ok;
                        if !quant {
                            continue;
                        }
                        // Q1 on each of N3's own rows.
                        let (q1, s1) = (
                            DeviceBuffer::upload_bytes(&device, &vec![0x5Au8; rows * cols])
                                .expect("q1"),
                            DeviceBuffer::upload_bytes(&device, &vec![0x5Au8; 4 * rows])
                                .expect("s1"),
                        );
                        for r in 0..rows {
                            // SAFETY: row `r` of each buffer is inside its allocation; Q1 reads
                            // `cols` bf16 and writes `cols` bytes and one f32.
                            unsafe {
                                quant_row_fp8_one(
                                    c.y.ptr().wrapping_byte_add(r * cols * 2).cast_const(),
                                    (
                                        q1.ptr().wrapping_byte_add(r * cols),
                                        s1.ptr().wrapping_byte_add(r * 4),
                                    ),
                                    cols as u32,
                                    std::ptr::null_mut(),
                                )
                            }
                            .expect("Q1");
                        }
                        let bytes = |b: &DeviceBuffer| -> Vec<u8> {
                            (b.download().expect("download").iter())
                                .flat_map(|h| h.to_le_bytes())
                                .collect()
                        };
                        let differ =
                            |a: &[u8], b: &[u8]| (a.iter().zip(b)).filter(|(p, q)| p != q).count();
                        let (qb, sb) = (
                            differ(&bytes(&c.q), &bytes(&q1)),
                            differ(&bytes(&c.scale), &bytes(&s1)),
                        );
                        let ok = qb == 0 && sb == 0;
                        eprintln!(
                            "N3_QUANT {form:?} cols={cols} offset={offset} rows={rows} {kind}: \
                             q_differing_bytes={qb} scale_differing_bytes={sb} {}",
                            verdict(ok)
                        );
                        pass &= ok;
                    }
                }
            }
        }
    }
    assert!(
        pass,
        "N3 left its bound, or its quant is not quant_rows_fp8 of its row"
    );
}

/// N3 twice on the same inputs, every form with its quant where it has one, at the three models'
/// widths and three rows: 0 differing bytes in `y`, `AddNorm`'s sum, the e4m3 rows and the scales.
#[test]
fn n3_repeats_bit_identically() {
    let device = h100();
    let mut pass = true;
    for cols in [2048usize, 2816, 5120] {
        let x = bf16_inputs(3 * cols, 0x0E35_0000 ^ cols as u64);
        let other = bf16_inputs(3 * cols, 0x0E36_0000 ^ cols as u64);
        let g = centred_gains(cols, 0x0E37_0000 ^ cols as u64);
        for form in [N3Form::Norm, N3Form::AddNorm, N3Form::NormAdd] {
            let quant = form != N3Form::NormAdd;
            let run = || {
                let c = N3::new(&device, (&x, &other, &g), 3);
                c.launch(form, quant, ONE_PLUS_SCALE, std::ptr::null_mut());
                [&c.y, &c.sum, &c.q, &c.scale].map(|b| b.download().expect("download"))
            };
            let (a, b) = (run(), run());
            let differ: usize = (a.iter().zip(&b))
                .map(|(p, q)| p.iter().zip(q).filter(|(x, y)| x != y).count())
                .sum();
            eprintln!(
                "N3_REPEAT {form:?} cols={cols}: differing_halves={differ} {}",
                verdict(differ == 0)
            );
            pass &= differ == 0;
        }
    }
    assert!(pass, "two N3 runs differ");
}

/// N3 refuses, before it touches a pointer and with its outputs unchanged: rows 0 and 65,536; cols
/// 0, 255, 320 and 8,448; form 3; eps 0, NaN; gain offset Inf; a null `x`, `gain` and `y`; a null
/// `other` at forms 1 and 2; a null `sum` at form 1; `q` without `scale` and `scale` without `q`;
/// the quant at form 2.
#[test]
fn n3_refuses_what_it_does_not_take() {
    use scratchy_target_cuda::kernels::rms_norm_quant_row;
    let device = h100();
    let c = N3::new(
        &device,
        (
            &bf16_inputs(4 * 8192, 1),
            &bf16_inputs(4 * 8192, 2),
            &bf16_inputs(8192, 3),
        ),
        4,
    );
    let null = std::ptr::null_mut::<std::ffi::c_void>();
    let (x, o, g) = (
        c.x.ptr().cast_const(),
        c.other.ptr().cast_const(),
        c.g.ptr().cast_const(),
    );
    let (y, s, q, sc) = (c.y.ptr(), c.sum.ptr(), c.q.ptr(), c.scale.ptr());
    type Case = (
        &'static str,
        (
            *const std::ffi::c_void,
            *const std::ffi::c_void,
            *const std::ffi::c_void,
        ),
        (*mut std::ffi::c_void, *mut std::ffi::c_void),
        Option<(*mut std::ffi::c_void, *mut std::ffi::c_void)>,
        (u32, u32),
        (u32, f32, f32),
    );
    let ok_in = (x, o, g);
    let cases: [Case; 18] = [
        ("rows 0", ok_in, (y, s), None, (0, 2048), (0, EPS, 1.0)),
        (
            "rows 65536",
            ok_in,
            (y, s),
            None,
            (65536, 2048),
            (0, EPS, 1.0),
        ),
        ("cols 0", ok_in, (y, s), None, (1, 0), (0, EPS, 1.0)),
        ("cols 255", ok_in, (y, s), None, (1, 255), (0, EPS, 1.0)),
        ("cols 320", ok_in, (y, s), None, (1, 320), (0, EPS, 1.0)),
        ("cols 8448", ok_in, (y, s), None, (1, 8448), (0, EPS, 1.0)),
        ("form 3", ok_in, (y, s), None, (1, 2048), (3, EPS, 1.0)),
        ("eps 0", ok_in, (y, s), None, (1, 2048), (0, 0.0, 1.0)),
        (
            "eps NaN",
            ok_in,
            (y, s),
            None,
            (1, 2048),
            (0, f32::NAN, 1.0),
        ),
        (
            "offset Inf",
            ok_in,
            (y, s),
            None,
            (1, 2048),
            (0, EPS, f32::INFINITY),
        ),
        (
            "null x",
            (std::ptr::null(), o, g),
            (y, s),
            None,
            (1, 2048),
            (0, EPS, 1.0),
        ),
        (
            "null gain",
            (x, o, std::ptr::null()),
            (y, s),
            None,
            (1, 2048),
            (0, EPS, 1.0),
        ),
        ("null y", ok_in, (null, s), None, (1, 2048), (0, EPS, 1.0)),
        (
            "null other, form 1",
            (x, std::ptr::null(), g),
            (y, s),
            None,
            (1, 2048),
            (1, EPS, 1.0),
        ),
        (
            "null other, form 2",
            (x, std::ptr::null(), g),
            (y, s),
            None,
            (1, 2048),
            (2, EPS, 1.0),
        ),
        (
            "null sum, form 1",
            ok_in,
            (y, null),
            None,
            (1, 2048),
            (1, EPS, 1.0),
        ),
        (
            "q without scale",
            ok_in,
            (y, s),
            Some((q, null)),
            (1, 2048),
            (0, EPS, 1.0),
        ),
        (
            "quant at form 2",
            ok_in,
            (y, s),
            Some((q, sc)),
            (1, 2048),
            (2, EPS, 1.0),
        ),
    ];
    let before = [&c.y, &c.sum, &c.q, &c.scale].map(|b| b.download().expect("download"));
    let mut pass = true;
    for (what, ins, outs, quant, dims, (form, eps, offset)) in cases {
        // SAFETY: refused before any pointer is touched; every non-null pointer is a live buffer.
        let got = unsafe { rms_norm_quant_row(ins, outs, quant, dims, (form, eps, offset), null) };
        let want = KernelError::RowShape {
            rows: dims.0,
            cols: dims.1,
        };
        let ok = got == Err(want);
        eprintln!("N3 refused {what}: {got:?} {}", verdict(ok));
        pass &= ok;
    }
    // `scale` without `q`, through the launcher's own null `q`: refused the same way.
    // SAFETY: as above.
    let got = unsafe {
        rms_norm_quant_row(
            ok_in,
            (y, s),
            Some((null, sc)),
            (1, 2048),
            (0, EPS, 1.0),
            null,
        )
    };
    let ok = got
        == Err(KernelError::RowShape {
            rows: 1,
            cols: 2048,
        });
    eprintln!("N3 refused scale without q: {got:?} {}", verdict(ok));
    pass &= ok;
    let after = [&c.y, &c.sum, &c.q, &c.scale].map(|b| b.download().expect("download"));
    let unchanged = before == after;
    eprintln!(
        "N3 refusals left every output unchanged: {unchanged} {}",
        verdict(unchanged)
    );
    assert!(
        pass && unchanged,
        "N3 took what it must refuse, or wrote an output"
    );
}

/// Recorded, not asserted (N3's gate: bring it to the tape only if it saves ≥ ~0.5 ms a Gemma
/// token): inside a captured graph of 10 back-to-back nodes, a node one pair, today's pair (N1's
/// row norm, `rms_norm_bf16` / `add_rms_norm_bf16` / `norm_add_bf16` at one row, then Q1 on its
/// output where the GEMM after it is a T2 one) against N3 (one launch, with its quant), at the
/// three models' widths and forms, read-cold and hot, two rounds (A, B, A, B): `TC_TIME …
/// n1_q1_graph_per_node` and `… n3_graph_per_node` lines.
#[test]
fn n3_graph_times_are_recorded() {
    use scratchy_target_cuda::kernels::quant_row_fp8_one;
    let device = h100();
    let scrub = DeviceBuffer::upload_bytes(&device, &vec![0u8; SCRUB_BYTES]).expect("scrub");
    let shapes = [
        ("27B", 5120usize, N3Form::Norm, ONE_PLUS_SCALE),
        ("27B", 5120, N3Form::AddNorm, ONE_PLUS_SCALE),
        ("35B", 2048, N3Form::Norm, ONE_PLUS_SCALE),
        ("35B", 2048, N3Form::AddNorm, ONE_PLUS_SCALE),
        ("Gemma", 2816, N3Form::Norm, SCALE),
        ("Gemma", 2816, N3Form::NormAdd, SCALE),
    ];
    for (model, cols, form, offset) in shapes {
        let quant = form != N3Form::NormAdd;
        let x = bf16_inputs(cols, 0x0E38_0000 ^ cols as u64);
        let other = bf16_inputs(cols, 0x0E39_0000 ^ cols as u64);
        let g = centred_gains(cols, 0x0E3A_0000 ^ cols as u64);
        let c = N3::new(&device, (&x, &other, &g), 1);
        let shape = format!(
            "{model}_{form:?}_{cols}{}",
            if quant { "_quant" } else { "" }
        );
        let today = |s: *mut std::ffi::c_void| {
            let dims = (1u32, cols as u32);
            // SAFETY: as `N3::launch`'s, one row.
            unsafe {
                match form {
                    N3Form::Norm => {
                        rms_norm_bf16(c.x.ptr(), c.g.ptr(), c.y.ptr(), dims, EPS, offset, s)
                    }
                    N3Form::AddNorm => add_rms_norm_bf16(
                        (c.x.ptr(), c.other.ptr()),
                        c.g.ptr(),
                        (c.sum.ptr(), c.y.ptr()),
                        dims,
                        (EPS, offset),
                        s,
                    ),
                    N3Form::NormAdd => norm_add_bf16(
                        (c.x.ptr(), c.other.ptr()),
                        c.g.ptr(),
                        c.y.ptr(),
                        dims,
                        (EPS, offset),
                        s,
                    ),
                }
                .expect("N1");
                if quant {
                    quant_row_fp8_one(
                        c.y.ptr().cast_const(),
                        (c.q.ptr(), c.scale.ptr()),
                        cols as u32,
                        s,
                    )
                    .expect("Q1");
                }
            }
        };
        for round in 1..=2 {
            graph_record(&scrub, (&shape, "n1_q1", round), today);
            graph_record(&scrub, (&shape, "n3", round), |s| {
                c.launch(form, quant, offset, s)
            });
        }
    }
}
