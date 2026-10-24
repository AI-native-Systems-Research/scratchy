// SPDX-License-Identifier: Apache-2.0
//! The attention prelude's kernels (`kernels/rope_bf16.cu`, `gate_split_bf16.cu`,
//! `gate_apply_bf16.cu`) against an f64 CPU reference, on the H100, and their launchers' refusals.
//!
//! Built only under `nvcc` (`required-features`). No GPU fails rather than skips: this test is the
//! kernels' only gate. The references and bounds are `common::numerics`'. Every shape runs and
//! prints its line before a test fails, so one run reports all the numbers.

// `pub`: each test binary uses a different part of `common`, and nothing in it is dead code.
pub mod common {
    pub mod numerics;
}

use common::numerics::{
    bf16_inputs, bits, compare_rope, compare_rows, gate_apply_reference, gate_split_reference,
    mismatched_bits, rope_misses, rope_reference, row_bound, whole_row_split,
};
use scratchy_target_cuda::kernels::{KernelError, gate_apply_bf16, gate_split_bf16, rope_bf16};
use scratchy_target_cuda::rotary::rotary_rows;
use scratchy_target_cuda::runtime::{CudaDevice, DeviceBuffer};

/// qwen3.5's head dim, rotated width and `rope_theta`.
const HD: u32 = 256;
const R: u32 = 64;
const THETA: f64 = 1e7;

/// Extra output elements past the shape, which a kernel must leave untouched.
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

// ── The rope ────────────────────────────────────────────────────────

/// Run the rope on `x` with the table of `positions` over `(rows, cols)`; the GPU's output, its
/// tail, and the table it read.
fn run_rope(
    device: &CudaDevice,
    x: &[u16],
    positions: &[u32],
    (rows, cols): (usize, usize),
) -> (Vec<u16>, bool, Vec<u16>) {
    let table = rotary_rows(positions, R, THETA);
    let (xb, tb) = (
        DeviceBuffer::upload(device, x).expect("upload x"),
        DeviceBuffer::upload(device, &table).expect("upload the table"),
    );
    let out = output(device, x.len());
    // SAFETY: cudaMalloc pointers are 256-byte aligned; `x` and `out` hold rows·cols elements and
    // the table rows·R; `out` is its own allocation; null is the default stream.
    unsafe {
        rope_bf16(
            xb.ptr(),
            tb.ptr(),
            out.ptr(),
            (rows as u32, cols as u32),
            (HD, R),
            std::ptr::null_mut(),
        )
    }
    .expect("rope_bf16 launch");
    let (gpu, tail) = result(&out, x.len());
    (gpu, tail, table)
}

/// The rope over `(rows, cols)` at `positions`: every rotated element within its bound, the
/// pass-through lanes and every position-0 row bit-exact, the tail intact. Returns whether it
/// passed, and the GPU output, the input and the table, for the self-check.
fn rope_shape(
    device: &CudaDevice,
    (rows, cols): (usize, usize),
    positions: &[u32],
) -> (bool, Vec<u16>, Vec<u16>, Vec<u16>) {
    let seed = (rows as u64) << 32 | cols as u64;
    let x = bf16_inputs(rows * cols, 0x0E0E_0000 ^ seed);
    let (gpu, tail, table) = run_rope(device, &x, positions, (rows, cols));
    let shape = (HD as usize, R as usize);
    let refs = rope_reference(&x, &table, (rows, cols), shape, R as usize / 2);
    let cmp = compare_rope(&gpu, &x, &refs, cols);
    let zero_rows = (0..rows).filter(|&r| positions[r] == 0);
    let zero_mismatched: usize = zero_rows
        .map(|r| mismatched_bits(&gpu[r * cols..][..cols], &x[r * cols..][..cols]))
        .sum();
    let pass =
        cmp.rotated.pass() && cmp.pass_through_mismatched == 0 && zero_mismatched == 0 && tail;
    eprintln!(
        "rope_bf16 rows={rows} cols={cols} head_dim={HD} rot_dim={R}: {} \
         pass_through_mismatched_bits={} position0_mismatched_bits={zero_mismatched} \
         tail_intact={tail} {}",
        cmp.rotated.line(),
        cmp.pass_through_mismatched,
        verdict(pass)
    );
    (pass, gpu, x, table)
}

/// Positions `7 + 3t`, the larger fixtures'.
fn positions(rows: usize) -> Vec<u32> {
    (0..rows as u32).map(|t| 7 + 3 * t).collect()
}

#[test]
fn rope_bf16_matches_the_cpu_reference() {
    let device = h100();
    let mut pass = true;
    for shape in [(1, 4096), (256, 2048), (256, 512), (128, 4096)] {
        pass &= rope_shape(&device, shape, &positions(shape.0)).0;
    }
    assert!(pass, "a rope shape is out of bound");
}

/// `(7, 1024)`, rows at positions 0, 1, 2, 1000, 65535, 200000 and 262143: within the bound, the
/// position-0 row bit-exact; and the full-head-pairing self-check on the rows at 65535, 200000
/// and 262143, where the angles are large enough that a kernel pairing `d` with `d + 128` must
/// miss more than 90% of the rotated elements.
#[test]
fn rope_bf16_positions_and_the_full_head_pairing_self_check() {
    const POSITIONS: [u32; 7] = [0, 1, 2, 1000, 65535, 200000, 262143];
    const COLS: usize = 1024;
    let device = h100();
    let (mut pass, gpu, x, table) = rope_shape(&device, (7, COLS), &POSITIONS);
    let (mut missed, mut rotated) = (0, 0);
    for r in 4..7 {
        let at = r * COLS..(r + 1) * COLS;
        let row_table = &table[r * R as usize..][..R as usize];
        let wrong = rope_reference(
            &x[at.clone()],
            row_table,
            (1, COLS),
            (HD as usize, R as usize),
            HD as usize / 2,
        );
        let (m, n) = rope_misses(&gpu[at], &wrong);
        missed += m;
        rotated += n;
    }
    let distinguishes = missed * 10 > rotated * 9;
    eprintln!(
        "rope full-head pairing self-check (positions 65535, 200000, 262143): misses {missed} of \
         {rotated} rotated elements {}",
        verdict(distinguishes)
    );
    pass &= distinguishes;
    assert!(pass, "the rope's position fixture or its self-check failed");
}

// ── The gate split ──────────────────────────────────────────────────

/// Run the split of `qg` over `(rows, half_cols)`; `q` and `gate` and whether both tails held.
fn run_split(
    device: &CudaDevice,
    qg: &[u16],
    (rows, half_cols): (usize, usize),
) -> (Vec<u16>, Vec<u16>, bool) {
    let qgb = DeviceBuffer::upload(device, qg).expect("upload qg");
    let len = rows * half_cols;
    let (q, gate) = (output(device, len), output(device, len));
    // SAFETY: cudaMalloc pointers are 256-byte aligned; `qg` holds rows·2·half_cols elements and
    // `q`, `gate` rows·half_cols; each is its own allocation; null is the default stream.
    unsafe {
        gate_split_bf16(
            qgb.ptr(),
            q.ptr(),
            gate.ptr(),
            (rows as u32, half_cols as u32),
            HD,
            std::ptr::null_mut(),
        )
    }
    .expect("gate_split_bf16 launch");
    let ((q, qt), (g, gt)) = (result(&q, len), result(&gate, len));
    (q, g, qt && gt)
}

#[test]
fn gate_split_bf16_is_bit_exact() {
    let device = h100();
    let mut pass = true;
    for (rows, half_cols) in [(1, 4096), (7, 2048), (256, 2048), (256, 4096)] {
        let seed = (rows as u64) << 32 | half_cols as u64;
        let qg = bf16_inputs(rows * 2 * half_cols, 0x6A7E_0000 ^ seed);
        let (q, gate, tail) = run_split(&device, &qg, (rows, half_cols));
        let (want_q, want_gate) = gate_split_reference(&qg, (rows, half_cols), HD as usize);
        let mismatched = mismatched_bits(&q, &want_q) + mismatched_bits(&gate, &want_gate);
        let ok = mismatched == 0 && tail;
        eprintln!(
            "gate_split_bf16 rows={rows} half_cols={half_cols} head_dim={HD}: \
             mismatched_bits={mismatched} tail_intact={tail} {}",
            verdict(ok)
        );
        pass &= ok;
        // The self-check at 8 heads (half_cols 2048): a split by halves of the whole row differs
        // from the GPU's for more than 85% of the elements (87.5% is the most it can: it agrees on
        // head 0's query and head 7's gate).
        if half_cols == 2048 {
            let (wq, wg) = whole_row_split(&qg, half_cols);
            let differ = mismatched_bits(&q, &wq) + mismatched_bits(&gate, &wg);
            let total = 2 * rows * half_cols;
            let distinguishes = differ * 100 > total * 85;
            eprintln!(
                "gate split self-check rows={rows} (8 heads): the whole-row split differs in \
                 {differ} of {total} elements {}",
                verdict(distinguishes)
            );
            pass &= distinguishes;
        }
    }
    assert!(
        pass,
        "a gate split shape is not bit-exact, or its self-check failed"
    );
}

// ── The gate apply ──────────────────────────────────────────────────

fn run_apply(
    device: &CudaDevice,
    attn: &[u16],
    gate: &[u16],
    (rows, cols): (usize, usize),
) -> (Vec<u16>, bool) {
    let (ab, gb) = (
        DeviceBuffer::upload(device, attn).expect("upload attn"),
        DeviceBuffer::upload(device, gate).expect("upload gate"),
    );
    let out = output(device, attn.len());
    // SAFETY: cudaMalloc pointers are 256-byte aligned; each buffer holds rows·cols elements;
    // `out` is its own allocation; null is the default stream.
    unsafe {
        gate_apply_bf16(
            ab.ptr(),
            gb.ptr(),
            out.ptr(),
            (rows as u32, cols as u32),
            std::ptr::null_mut(),
        )
    }
    .expect("gate_apply_bf16 launch");
    result(&out, attn.len())
}

/// `(1, 4096)`, `(256, 2048)` and `(7, 4096)`, whose row 3 holds the extreme gates
/// `{−100, −88, −20, −1, 0, 1, 20, 88, 100}` against `a ∈ {±1, ±0.5}`: every element within the
/// bound, the extreme row with no NaN or Inf, and `g ≤ −88` giving ±0 or a value within the bound.
#[test]
fn gate_apply_bf16_matches_the_cpu_reference() {
    const EXTREME: usize = 3;
    const GATES: [f32; 9] = [-100.0, -88.0, -20.0, -1.0, 0.0, 1.0, 20.0, 88.0, 100.0];
    const ATTNS: [f32; 4] = [1.0, -1.0, 0.5, -0.5];
    let device = h100();
    let mut pass = true;
    for (rows, cols) in [(1, 4096), (7, 4096), (256, 2048)] {
        let seed = (rows as u64) << 32 | cols as u64;
        let attn_in = bf16_inputs(rows * cols, 0x6A77_0000 ^ seed);
        // Gates in [-8, 8]: the sigmoid's whole curve, not only its linear middle.
        let mut gate: Vec<u16> = bf16_inputs(rows * cols, 0x6A77_FFFF ^ seed)
            .into_iter()
            .map(|b| bits(half::bf16::from_bits(b).to_f32() * 8.0))
            .collect();
        let mut attn = attn_in;
        let extreme = rows == 7;
        if extreme {
            let at = EXTREME * cols;
            for c in 0..cols {
                gate[at + c] = bits(GATES[c % GATES.len()]);
                attn[at + c] = bits(ATTNS[(c / GATES.len()) % ATTNS.len()]);
            }
        }
        let (gpu, tail) = run_apply(&device, &attn, &gate, (rows, cols));
        let refs = gate_apply_reference(&attn, &gate);
        let cmp = compare_rows(&gpu, &refs, cols);
        let ok = cmp.pass() && tail;
        eprintln!(
            "gate_apply_bf16 rows={rows} cols={cols}: {} tail_intact={tail} {}",
            cmp.line(),
            verdict(ok)
        );
        pass &= ok;
        if extreme {
            let at = EXTREME * cols;
            let row = &gpu[at..at + cols];
            let finite = row.iter().all(|&b| half::bf16::from_bits(b).is_finite());
            let very_negative_ok = (0..cols)
                .filter(|c| GATES[c % GATES.len()] <= -88.0)
                .all(|c| {
                    let g = f64::from(half::bf16::from_bits(row[c]).to_f32());
                    g == 0.0 || (g - refs[at + c]).abs() <= row_bound(refs[at + c])
                });
            eprintln!(
                "gate apply extreme row {EXTREME}: no NaN or Inf {finite}, g <= -88 gives ±0 or \
                 within the bound {very_negative_ok} {}",
                verdict(finite && very_negative_ok)
            );
            pass &= finite && very_negative_ok;
        }
    }
    assert!(pass, "a gate apply shape is out of bound");
}

// ── The launchers' refusals ─────────────────────────────────────────

/// A refused launch's output buffer: `len` sentinel elements, at least one 16-byte vector.
fn refused_out(device: &CudaDevice, len: usize) -> DeviceBuffer {
    DeviceBuffer::upload(device, &vec![SENTINEL; len.max(8)]).expect("upload out")
}

fn unchanged(out: &DeviceBuffer) -> bool {
    out.download()
        .expect("download out")
        .iter()
        .all(|&b| b == SENTINEL)
}

/// 16-byte buffers when the shape is above `INT_MAX` elements (refused before any pointer is
/// touched), else the shape's own size, capped.
fn refused_len(elements: u64) -> usize {
    if elements > i32::MAX as u64 {
        8
    } else {
        (elements as usize).clamp(8, 1 << 20)
    }
}

#[test]
fn every_launcher_refuses_a_shape_it_cannot_run() {
    let device = h100();
    let null = std::ptr::null_mut();
    let mut pass = true;

    // The rope: (rows, cols, head_dim, rot_dim).
    for (rows, cols, head_dim, rot_dim) in [
        (7, 1024, 256, 0),
        (7, 1024, 256, 24),
        (7, 1024, 256, 512),
        (7, 1024, 0, 64),
        (7, 1008, 252, 64),
        (7, 1000, 256, 64),
        (0, 1024, 256, 64),
        (65536, 65536, 256, 64),
    ] {
        let len = refused_len(rows as u64 * cols as u64);
        let input = DeviceBuffer::upload(&device, &vec![0; len]).expect("upload x");
        let table = DeviceBuffer::upload(&device, &vec![0; len]).expect("upload the table");
        let out = refused_out(&device, len);
        // SAFETY: the launcher refuses these shapes before it touches any pointer.
        let got = unsafe {
            rope_bf16(
                input.ptr(),
                table.ptr(),
                out.ptr(),
                (rows, cols),
                (head_dim, rot_dim),
                null,
            )
        };
        let want = Err(KernelError::RopeShape {
            rows,
            cols,
            head_dim,
            rot_dim,
        });
        let kept = unchanged(&out);
        let ok = got == want && kept;
        eprintln!(
            "rope_bf16 rows={rows} cols={cols} head_dim={head_dim} rot_dim={rot_dim}: {got:?} \
             out_unchanged={kept} {}",
            verdict(ok)
        );
        pass &= ok;
    }

    // The gate split: (rows, half_cols, head_dim).
    for (rows, half_cols, head_dim) in [
        (7, 2048, 0),
        (7, 1000, 256),
        (0, 2048, 256),
        (65536, 65536, 256),
    ] {
        let len = refused_len(rows as u64 * 2 * half_cols as u64);
        let qg = DeviceBuffer::upload(&device, &vec![0; len]).expect("upload qg");
        let (q, gate) = (refused_out(&device, len), refused_out(&device, len));
        // SAFETY: the launcher refuses these shapes before it touches any pointer.
        let got = unsafe {
            gate_split_bf16(
                qg.ptr(),
                q.ptr(),
                gate.ptr(),
                (rows, half_cols),
                head_dim,
                null,
            )
        };
        let want = Err(KernelError::GateShape {
            rows,
            half_cols,
            head_dim,
        });
        let kept = unchanged(&q) && unchanged(&gate);
        let ok = got == want && kept;
        eprintln!(
            "gate_split_bf16 rows={rows} half_cols={half_cols} head_dim={head_dim}: {got:?} \
             out_unchanged={kept} {}",
            verdict(ok)
        );
        pass &= ok;
    }

    // The gate apply: (rows, cols).
    for (rows, cols) in [(7, 1004), (7, 0), (0, 4096), (65536, 65536)] {
        let len = refused_len(rows as u64 * cols as u64);
        let a = DeviceBuffer::upload(&device, &vec![0; len]).expect("upload attn");
        let g = DeviceBuffer::upload(&device, &vec![0; len]).expect("upload gate");
        let out = refused_out(&device, len);
        // SAFETY: the launcher refuses these shapes before it touches any pointer.
        let got = unsafe { gate_apply_bf16(a.ptr(), g.ptr(), out.ptr(), (rows, cols), null) };
        let kept = unchanged(&out);
        let ok = got == Err(KernelError::RowShape { rows, cols }) && kept;
        eprintln!(
            "gate_apply_bf16 rows={rows} cols={cols}: {got:?} out_unchanged={kept} {}",
            verdict(ok)
        );
        pass &= ok;
    }
    assert!(pass, "a launcher did not refuse a shape it cannot run");
}

/// `(7, 1000)`: 1000 is a multiple of 8 but not of 256, so the gate apply's last column tile is
/// partial (`kernels/gate_apply_bf16.cu`: a plain ThunderKittens tile reads zeros past `cols` and
/// stores nothing there), and 7 rows leave its 16-row block partial. Under the existing bound, the
/// output's tail intact; then a repeat on `(256, 2048)` gives the same bits.
#[test]
fn gate_apply_bf16_takes_a_partial_last_column_tile_and_repeats() {
    let device = h100();
    let (rows, cols) = (7usize, 1000usize);
    let attn = bf16_inputs(rows * cols, 0x6A77_1000);
    let gate: Vec<u16> = bf16_inputs(rows * cols, 0x6A77_1001)
        .into_iter()
        .map(|b| bits(half::bf16::from_bits(b).to_f32() * 8.0))
        .collect();
    let (gpu, tail) = run_apply(&device, &attn, &gate, (rows, cols));
    let cmp = compare_rows(&gpu, &gate_apply_reference(&attn, &gate), cols);
    let mut pass = cmp.pass() && tail;
    eprintln!(
        "gate_apply_bf16 rows={rows} cols={cols} (partial last column tile): {} tail_intact={tail} {}",
        cmp.line(),
        verdict(pass)
    );
    let (rows, cols) = (256usize, 2048usize);
    let attn = bf16_inputs(rows * cols, 0x6A77_2000);
    let gate = bf16_inputs(rows * cols, 0x6A77_2001);
    let run = || run_apply(&device, &attn, &gate, (rows, cols)).0;
    let mismatched = mismatched_bits(&run(), &run());
    eprintln!(
        "gate_apply_bf16 rows={rows} cols={cols} repeat mismatched_bits={mismatched} {}",
        verdict(mismatched == 0)
    );
    pass &= mismatched == 0;
    assert!(pass, "the gate apply's partial tile or repeat failed");
}

/// For the base-against-change comparison (#850): the gate apply's output at each of its test
/// shapes (the seeds of `gate_apply_bf16_matches_the_cpu_reference`, without its extreme row,
/// then `(7, 1000)`), written as raw little-endian bf16 to
/// `<CARGO_TARGET_TMPDIR>/row_ops_dump/<case>.bin`, with `case= elems= fnv1a=` printed. The same
/// file run at two commits gives the `base_vs_pr mismatched_bits=` per case; this test asserts
/// only that each launch ran.
#[test]
fn gate_apply_bf16_outputs_are_dumped_for_a_base_comparison() {
    let device = h100();
    for (rows, cols) in [(1usize, 4096usize), (7, 4096), (256, 2048), (7, 1000)] {
        let seed = (rows as u64) << 32 | cols as u64;
        let attn = bf16_inputs(rows * cols, 0x6A77_0000 ^ seed);
        let gate: Vec<u16> = bf16_inputs(rows * cols, 0x6A77_FFFF ^ seed)
            .into_iter()
            .map(|b| bits(half::bf16::from_bits(b).to_f32() * 8.0))
            .collect();
        let (gpu, _) = run_apply(&device, &attn, &gate, (rows, cols));
        let dir = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("row_ops_dump");
        std::fs::create_dir_all(&dir).expect("create the dump directory");
        let bytes: Vec<u8> = gpu.iter().flat_map(|e| e.to_le_bytes()).collect();
        let case = format!("gate_apply_bf16_r{rows}_c{cols}");
        std::fs::write(dir.join(format!("{case}.bin")), &bytes).expect("write a dump");
        let fnv = (bytes.iter()).fold(0xCBF2_9CE4_8422_2325u64, |h, &b| {
            (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01B3)
        });
        eprintln!("case={case} elems={} fnv1a={fnv:016x}", gpu.len());
    }
}
