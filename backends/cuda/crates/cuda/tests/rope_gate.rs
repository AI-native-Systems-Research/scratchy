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
    pub mod watchdog;
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

/// The kernel-side tightenings (#853): the gate split is instantiated for head_dim 256 only, so
/// head_dim 128 (a multiple of 8, whole heads of `half_cols` 2048) is refused; the rope for
/// rot_dim 64 only, so rot_dim 48 (a multiple of 16 but not of 32) is refused. Each refusal leaves
/// its output unchanged.
#[test]
fn the_tightened_launchers_refuse_a_shape_off_their_instantiation() {
    let device = h100();
    let null = std::ptr::null_mut();
    let mut pass = true;

    let (rows, half_cols, head_dim) = (7u32, 2048u32, 128u32);
    let len = refused_len(u64::from(rows) * 2 * u64::from(half_cols));
    let qg = DeviceBuffer::upload(&device, &vec![0; len]).expect("upload qg");
    let (q, gate) = (refused_out(&device, len), refused_out(&device, len));
    // SAFETY: the launcher refuses this shape before it touches any pointer.
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

    let (rows, cols, head_dim, rot_dim) = (7u32, 1024u32, 256u32, 48u32);
    let len = refused_len(u64::from(rows) * u64::from(cols));
    let input = DeviceBuffer::upload(&device, &vec![0; len]).expect("upload x");
    let table = DeviceBuffer::upload(&device, &vec![0; len]).expect("upload the table");
    let out = refused_out(&device, len);
    // SAFETY: the launcher refuses this shape before it touches any pointer.
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
    assert!(
        pass,
        "a tightened launcher did not refuse a shape off its instantiation"
    );
}

/// The same input twice gives the same bits: the rope at `(256, 2048)` and the gate split at
/// `(256, 4096)`, both with their tails intact.
#[test]
fn rope_bf16_and_gate_split_bf16_repeat() {
    let device = h100();
    let (rows, cols) = (256usize, 2048usize);
    let x = bf16_inputs(rows * cols, 0x0E0E_2000);
    let pos = positions(rows);
    let (a, a_tail, _) = run_rope(&device, &x, &pos, (rows, cols));
    let (b, b_tail, _) = run_rope(&device, &x, &pos, (rows, cols));
    let mismatched = mismatched_bits(&a, &b);
    let ok = mismatched == 0 && a_tail && b_tail;
    eprintln!(
        "rope_bf16 rows={rows} cols={cols} repeat mismatched_bits={mismatched} \
         tail_intact={} {}",
        a_tail && b_tail,
        verdict(ok)
    );
    let mut pass = ok;
    let (rows, half_cols) = (256usize, 4096usize);
    let qg = bf16_inputs(rows * 2 * half_cols, 0x6A7E_2000);
    let (q0, g0, t0) = run_split(&device, &qg, (rows, half_cols));
    let (q1, g1, t1) = run_split(&device, &qg, (rows, half_cols));
    let mismatched = mismatched_bits(&q0, &q1) + mismatched_bits(&g0, &g1);
    let ok = mismatched == 0 && t0 && t1;
    eprintln!(
        "gate_split_bf16 rows={rows} half_cols={half_cols} repeat mismatched_bits={mismatched} \
         tail_intact={} {}",
        t0 && t1,
        verdict(ok)
    );
    pass &= ok;
    assert!(
        pass,
        "the rope or the gate split did not repeat bit for bit"
    );
}

/// For the base-against-change comparison (#853): the rope's output at each of its test shapes
/// (the inputs of `rope_bf16_matches_the_cpu_reference`, then the positions fixture `(7, 1024)`),
/// written as raw little-endian bf16 to `<CARGO_TARGET_TMPDIR>/row_ops_dump/<case>.bin`, with
/// `case= elems= head_dim= rot_dim= fnv1a=` printed. Within each `head_dim` lanes of a row, the
/// first `rot_dim` are rotated and the rest pass through, so the same file run at two commits gives
/// the `base_vs_pr mismatched_bits=` per case for each kind of lane; this test asserts only that
/// each launch ran.
#[test]
fn rope_bf16_outputs_are_dumped_for_a_base_comparison() {
    const POSITIONS: [u32; 7] = [0, 1, 2, 1000, 65535, 200000, 262143];
    let device = h100();
    let cases: [((usize, usize), Vec<u32>); 5] = [
        ((1, 4096), positions(1)),
        ((256, 2048), positions(256)),
        ((256, 512), positions(256)),
        ((128, 4096), positions(128)),
        ((7, 1024), POSITIONS.to_vec()),
    ];
    for ((rows, cols), pos) in cases {
        let seed = (rows as u64) << 32 | cols as u64;
        let x = bf16_inputs(rows * cols, 0x0E0E_0000 ^ seed);
        let (gpu, _, _) = run_rope(&device, &x, &pos, (rows, cols));
        let dir = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("row_ops_dump");
        std::fs::create_dir_all(&dir).expect("create the dump directory");
        let bytes: Vec<u8> = gpu.iter().flat_map(|e| e.to_le_bytes()).collect();
        let case = format!("rope_bf16_r{rows}_c{cols}");
        std::fs::write(dir.join(format!("{case}.bin")), &bytes).expect("write a dump");
        let fnv = (bytes.iter()).fold(0xCBF2_9CE4_8422_2325u64, |h, &b| {
            (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01B3)
        });
        eprintln!(
            "case={case} elems={} head_dim={HD} rot_dim={R} fnv1a={fnv:016x}",
            gpu.len()
        );
    }
}

// ── The full rope at rot_dim 256 (Gemma 4's sliding layers, #922) ──

/// Gemma 4's sliding heads: a full rope on all 256 lanes, at the sliding layers' `rope_theta`.
const FULL: u32 = 256;
const THETA_LOCAL: f64 = 1e4;

/// The full 256-lane rope over `(rows, cols)` at head dim 256: every lane rotated (lane `d < 128`
/// with `d + 128`) within its bound, every position-0 row bit-exact (c = 1, s = 0), the tail
/// intact, over positions `0 ..` spread to 8191 (the codegen capacity). The self-check: a
/// pairing at `d + 64` instead of `d + 128` misses the bound on at least 90% of the rotated lanes
/// of the rows past position 0. `rot_dim` 128 and 192 are refused with the output untouched.
#[test]
fn rope_bf16_full_256_matches_the_cpu_reference() {
    let device = h100();
    let mut pass = true;
    for (rows, cols) in [(1usize, 4096usize), (512, 4096), (7, 2048)] {
        // Row 0 at position 0, the last row at 8191, the rest spread between.
        let positions: Vec<u32> = (0..rows as u32)
            .map(|t| {
                if rows == 1 {
                    0
                } else {
                    t * 8191 / (rows as u32 - 1)
                }
            })
            .collect();
        let seed = 0x0922_0E0E ^ ((rows as u64) << 32 | cols as u64);
        let x = bf16_inputs(rows * cols, seed);
        let table = rotary_rows(&positions, FULL, THETA_LOCAL);
        let (xb, tb) = (
            DeviceBuffer::upload(&device, &x).expect("upload x"),
            DeviceBuffer::upload(&device, &table).expect("upload the table"),
        );
        let out = output(&device, x.len());
        // SAFETY: as `run_rope`'s, with the table rows·FULL.
        unsafe {
            rope_bf16(
                xb.ptr(),
                tb.ptr(),
                out.ptr(),
                (rows as u32, cols as u32),
                (HD, FULL),
                std::ptr::null_mut(),
            )
        }
        .expect("rope_bf16 launch at rot_dim 256");
        let (gpu, tail) = result(&out, x.len());
        let shape = (HD as usize, FULL as usize);
        let refs = rope_reference(&x, &table, (rows, cols), shape, FULL as usize / 2);
        let cmp = compare_rope(&gpu, &x, &refs, cols);
        let zero_mismatched: usize = (0..rows)
            .filter(|&r| positions[r] == 0)
            .map(|r| mismatched_bits(&gpu[r * cols..][..cols], &x[r * cols..][..cols]))
            .sum();
        // The self-check over the rows past position 0: `d` paired with `d + 64`.
        let (mut missed, mut rotated) = (0, 0);
        for r in (0..rows).filter(|&r| positions[r] != 0) {
            let at = r * cols..(r + 1) * cols;
            let row_table = &table[r * FULL as usize..][..FULL as usize];
            let wrong = rope_reference(&x[at.clone()], row_table, (1, cols), shape, 64);
            let (m, n) = rope_misses(&gpu[at], &wrong);
            missed += m;
            rotated += n;
        }
        let distinguishes = rotated == 0 || missed * 10 >= rotated * 9;
        let ok = cmp.rotated.pass()
            && cmp.pass_through_mismatched == 0
            && zero_mismatched == 0
            && tail
            && distinguishes;
        eprintln!(
            "rope_bf16 full rows={rows} cols={cols} head_dim={HD} rot_dim={FULL} \
             theta={THETA_LOCAL}: {} pass_through_mismatched_bits={} \
             position0_mismatched_bits={zero_mismatched} tail_intact={tail} \
             pairing_d_plus_64_self_check_misses={missed}/{rotated} (≥ 90%) {}",
            cmp.rotated.line(),
            cmp.pass_through_mismatched,
            verdict(ok)
        );
        pass &= ok;
    }
    let (rows, cols) = (7u32, 2048u32);
    let len = (rows * cols) as usize;
    let input = DeviceBuffer::upload(&device, &vec![0; len]).expect("upload x");
    let table = DeviceBuffer::upload(&device, &vec![0; len]).expect("upload the table");
    for rot_dim in [128u32, 192] {
        let out = refused_out(&device, len);
        // SAFETY: the launcher refuses these widths before it touches any pointer.
        let got = unsafe {
            rope_bf16(
                input.ptr(),
                table.ptr(),
                out.ptr(),
                (rows, cols),
                (HD, rot_dim),
                std::ptr::null_mut(),
            )
        };
        let want = Err(KernelError::RopeShape {
            rows,
            cols,
            head_dim: HD,
            rot_dim,
        });
        let kept = unchanged(&out);
        let ok = got == want && kept;
        eprintln!(
            "rope_bf16 full refusal rot_dim={rot_dim}: {got:?} out_unchanged={kept} {}",
            verdict(ok)
        );
        pass &= ok;
    }
    assert!(
        pass,
        "the full 256-lane rope is out of bound, or a refusal wrote"
    );
}

// ── The proportional rope (Gemma 4's global layers, #923) ──────────

/// Gemma 4's global heads: 128 of each 512-lane head rotated, lane `d < 64` paired with lane
/// `d + 256` (`RopePairing::HalfHead`), the frequencies over the head dim, at `rope_theta` 1e6.
const G_HD: u32 = 512;
const G_ROT: u32 = 128;
const THETA_GLOBAL: f64 = 1e6;

/// The proportional rope over `(rows, cols)` at head dim 512: the rotated lanes within the RoPE
/// bound against `rope_reference` with `pair = 256`, the pass-through lanes (boxes 1..3 and
/// 5..7 of each head) bit-exact, every position-0 row bit-exact, the tail intact, over positions
/// `0 ..` spread to 8191. Self-checks on the rows past position 0: pairing at `d + 64` (the
/// half-rotated pairing) misses the bound on at least 90% of the rotated lanes; and a table built
/// with `freq_dim = rot_dim` misses on at least 90% at position 1000. A half-head launch at head
/// dim 256, at rot_dim 64, and a half-rotated launch at rot_dim 128 are refused with the output
/// untouched.
#[test]
fn rope_bf16_proportional_matches_the_cpu_reference() {
    use scratchy_target_cuda::kernels::rope_paired_bf16;
    use scratchy_target_cuda::rotary::rotary_rows_with_freq_dim;
    use scratchy_target_cuda::step::RopePairing;
    let device = h100();
    let mut pass = true;
    let run = |x: &[u16], table: &[u16], (rows, cols): (usize, usize)| {
        let (xb, tb) = (
            DeviceBuffer::upload(&device, x).expect("upload x"),
            DeviceBuffer::upload(&device, table).expect("upload the table"),
        );
        let out = output(&device, x.len());
        // SAFETY: x and out are rows·cols bf16, the table rows·G_ROT, none overlapping, all live
        // until `result` downloads `out`, which waits for the launch.
        unsafe {
            rope_paired_bf16(
                xb.ptr(),
                tb.ptr(),
                out.ptr(),
                (rows as u32, cols as u32),
                (G_HD, G_ROT, RopePairing::HalfHead),
                std::ptr::null_mut(),
            )
        }
        .expect("rope_paired_bf16 launch, half-head");
        result(&out, x.len())
    };
    let shape = (G_HD as usize, G_ROT as usize);
    for (rows, cols) in [(1usize, 8192usize), (512, 8192), (7, 1024)] {
        let positions: Vec<u32> = (0..rows as u32)
            .map(|t| {
                if rows == 1 {
                    0
                } else {
                    t * 8191 / (rows as u32 - 1)
                }
            })
            .collect();
        let seed = 0x0923_0E0E ^ ((rows as u64) << 32 | cols as u64);
        let x = bf16_inputs(rows * cols, seed);
        let table = rotary_rows_with_freq_dim(&positions, G_ROT, G_HD, THETA_GLOBAL);
        let (gpu, tail) = run(&x, &table, (rows, cols));
        let refs = rope_reference(&x, &table, (rows, cols), shape, G_HD as usize / 2);
        let cmp = compare_rope(&gpu, &x, &refs, cols);
        let zero_mismatched: usize = (0..rows)
            .filter(|&r| positions[r] == 0)
            .map(|r| mismatched_bits(&gpu[r * cols..][..cols], &x[r * cols..][..cols]))
            .sum();
        let (mut missed, mut rotated) = (0, 0);
        for r in (0..rows).filter(|&r| positions[r] != 0) {
            let at = r * cols..(r + 1) * cols;
            let row_table = &table[r * G_ROT as usize..][..G_ROT as usize];
            let wrong = rope_reference(&x[at.clone()], row_table, (1, cols), shape, 64);
            let (m, n) = rope_misses(&gpu[at], &wrong);
            missed += m;
            rotated += n;
        }
        let distinguishes = rotated == 0 || missed * 10 >= rotated * 9;
        let ok = cmp.rotated.pass()
            && cmp.pass_through_mismatched == 0
            && zero_mismatched == 0
            && tail
            && distinguishes;
        eprintln!(
            "rope_bf16 proportional rows={rows} cols={cols} head_dim={G_HD} rot_dim={G_ROT} \
             theta={THETA_GLOBAL}: {} pass_through_mismatched={} \
             position0_mismatched_bits={zero_mismatched} tail_intact={tail} \
             pairing_d_plus_64_self_check_misses={missed}/{rotated} (≥ 90%) {}",
            cmp.rotated.line(),
            cmp.pass_through_mismatched,
            verdict(ok)
        );
        pass &= ok;
    }
    // The frequency self-check: the table with `freq_dim = rot_dim` at position 1000, against the
    // kernel run on the right table.
    {
        let (rows, cols) = (1usize, 8192usize);
        let x = bf16_inputs(rows * cols, 0x0923_F0F0);
        let table = rotary_rows_with_freq_dim(&[1000], G_ROT, G_HD, THETA_GLOBAL);
        let wrong_table = rotary_rows_with_freq_dim(&[1000], G_ROT, G_ROT, THETA_GLOBAL);
        let (gpu, _) = run(&x, &table, (rows, cols));
        let wrong = rope_reference(&x, &wrong_table, (rows, cols), shape, G_HD as usize / 2);
        let (missed, rotated) = rope_misses(&gpu, &wrong);
        let ok = missed * 10 >= rotated * 9;
        eprintln!(
            "rope_bf16 proportional freq_dim_rot_dim_self_check position=1000: \
             misses={missed}/{rotated} (≥ 90%) {}",
            verdict(ok)
        );
        pass &= ok;
    }
    let (rows, cols) = (7u32, 1024u32);
    let len = (rows * cols) as usize;
    let input = DeviceBuffer::upload(&device, &vec![0; len]).expect("upload x");
    let table = DeviceBuffer::upload(&device, &vec![0; len]).expect("upload the table");
    for (head_dim, rot_dim, pairing) in [
        (256u32, 128u32, RopePairing::HalfHead),
        (G_HD, 64, RopePairing::HalfHead),
        (G_HD, G_ROT, RopePairing::HalfRotated),
    ] {
        let out = refused_out(&device, len);
        // SAFETY: the launcher refuses these shapes before it touches any pointer.
        let got = unsafe {
            rope_paired_bf16(
                input.ptr(),
                table.ptr(),
                out.ptr(),
                (rows, cols),
                (head_dim, rot_dim, pairing),
                std::ptr::null_mut(),
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
            "rope_bf16 proportional refusal head_dim={head_dim} rot_dim={rot_dim} \
             pairing={pairing:?}: {got:?} out_unchanged={kept} {}",
            verdict(ok)
        );
        pass &= ok;
    }
    assert!(
        pass,
        "the proportional rope is out of bound, a pass-through lane moved, or a refusal wrote"
    );
}

// ── FP8 S8 W1: the one-row rope and gates ──────────────────────────

/// `row` as row 0 of `rows` rows, the others zero.
fn as_row0(row: &[u16], rows: usize) -> Vec<u16> {
    let mut v = row.to_vec();
    v.resize(rows * row.len(), 0);
    v
}

/// The rope launch, any shape and pairing; the output's first `cols` and whether the tail held.
fn run_rope_paired(
    device: &CudaDevice,
    (x, table): (&[u16], &[u16]),
    (rows, cols): (usize, usize),
    (head_dim, rot_dim, pairing): (u32, u32, scratchy_target_cuda::step::RopePairing),
) -> (Vec<u16>, bool) {
    use scratchy_target_cuda::kernels::rope_paired_bf16;
    let up = |v: &[u16]| DeviceBuffer::upload(device, v).expect("upload");
    let (xb, tb) = (up(x), up(table));
    let out = output(device, x.len());
    // SAFETY: cudaMalloc pointers are 256-byte aligned; `x` and `out` hold rows·cols bf16, the
    // table rows·rot_dim; `out` is its own allocation; null is the default stream.
    unsafe {
        rope_paired_bf16(
            xb.ptr(),
            tb.ptr(),
            out.ptr(),
            (rows as u32, cols as u32),
            (head_dim, rot_dim, pairing),
            std::ptr::null_mut(),
        )
    }
    .expect("rope_paired_bf16 launch");
    let (all, tail) = result(&out, x.len());
    (all[..cols].to_vec(), tail)
}

/// FP8 S8 W1: at `rows == 1` the rope takes its one-row form (one CTA per head), and its output is
/// the 16-row kernel's bits for the same row, as row 0 of a 16-row and of a 2-row launch. The
/// three instances at the FP8 models' shapes: `<64, false>` at the 27B's 24 + 4 heads of 256 and
/// the 35B's 16 + 2; `<256, false>` at Gemma's 16 heads of 256 (its sliding q) and 8 (its k);
/// `<128, true>` at Gemma's 16 and 2 global heads of 512. Positions 0, 1, 191 and 8,191, 3 seeds.
#[test]
fn the_one_row_rope_is_the_16_row_rope_bit_for_bit() {
    use scratchy_target_cuda::rotary::rotary_rows_with_freq_dim;
    use scratchy_target_cuda::step::RopePairing;
    let device = h100();
    let mut pass = true;
    // (what, heads, head_dim, rot_dim, pairing, theta)
    let shapes = [
        (
            "27b q+k <64>",
            28u32,
            HD,
            R,
            RopePairing::HalfRotated,
            THETA,
        ),
        ("35b q+k <64>", 18, HD, R, RopePairing::HalfRotated, THETA),
        (
            "gemma q <256>",
            16,
            FULL,
            FULL,
            RopePairing::HalfRotated,
            THETA_LOCAL,
        ),
        (
            "gemma k <256>",
            8,
            FULL,
            FULL,
            RopePairing::HalfRotated,
            THETA_LOCAL,
        ),
        (
            "gemma global q <128,half>",
            16,
            G_HD,
            G_ROT,
            RopePairing::HalfHead,
            THETA_GLOBAL,
        ),
        (
            "gemma global k <128,half>",
            2,
            G_HD,
            G_ROT,
            RopePairing::HalfHead,
            THETA_GLOBAL,
        ),
    ];
    for (what, heads, hd, rot, pairing, theta) in shapes {
        let cols = (heads * hd) as usize;
        for pos in [0u32, 1, 191, 8191] {
            let table = match pairing {
                RopePairing::HalfHead => rotary_rows_with_freq_dim(&[pos], rot, hd, theta),
                _ => rotary_rows(&[pos], rot, theta),
            };
            for seed in 0..3u64 {
                let x = bf16_inputs(
                    cols,
                    0x0E11_0000 ^ (cols as u64) << 20 ^ (pos as u64) << 4 ^ seed,
                );
                let dims = (hd, rot, pairing);
                let (one, t1) = run_rope_paired(&device, (&x, &table), (1, cols), dims);
                let mut ok = t1;
                let mut line = String::new();
                for rows in [16usize, 2] {
                    let (want, t) = run_rope_paired(
                        &device,
                        (&as_row0(&x, rows), &as_row0(&table, rows)),
                        (rows, cols),
                        dims,
                    );
                    let m = mismatched_bits(&one, &want);
                    line += &format!(" vs rows={rows} mismatched_bits={m}");
                    ok &= t && m == 0;
                }
                eprintln!(
                    "one-row rope {what} pos={pos} seed={seed}:{line} tail_intact={t1} {}",
                    verdict(ok)
                );
                pass &= ok;
            }
        }
    }
    assert!(pass, "a one-row rope differs from the 16-row rope");
}

/// `cudaEvent_t`, declared here for the timing lines only (the runtime declares no event call).
type CudaEvent = *mut std::ffi::c_void;

unsafe extern "C" {
    fn cudaEventCreate(event: *mut CudaEvent) -> i32;
    fn cudaEventRecord(event: CudaEvent, stream: *mut std::ffi::c_void) -> i32;
    fn cudaEventSynchronize(event: CudaEvent) -> i32;
    fn cudaEventElapsedTime(ms: *mut f32, start: CudaEvent, end: CudaEvent) -> i32;
    fn cudaEventDestroy(event: CudaEvent) -> i32;
}

/// The median µs of 20 launches after 3 warm-ups, each between two CUDA events on the default
/// stream (hot: the inputs stay in L2, as a decode step's activations do).
fn event_median(mut launch: impl FnMut()) -> f64 {
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

/// Recorded, not asserted: µs per launch of the rope at `rows == 1` (the one-row form) and at
/// `rows == 2` (today's 16-row kernel on one CTA, the time W0 measured), at the shapes above.
#[test]
fn the_one_row_rope_times_are_recorded() {
    use scratchy_target_cuda::kernels::rope_paired_bf16;
    use scratchy_target_cuda::rotary::rotary_rows_with_freq_dim;
    use scratchy_target_cuda::step::RopePairing;
    let device = h100();
    let shapes = [
        (
            "27b q+k <64>",
            28u32,
            HD,
            R,
            RopePairing::HalfRotated,
            THETA,
        ),
        ("35b q+k <64>", 18, HD, R, RopePairing::HalfRotated, THETA),
        (
            "gemma q <256>",
            16,
            FULL,
            FULL,
            RopePairing::HalfRotated,
            THETA_LOCAL,
        ),
        (
            "gemma global q <128,half>",
            16,
            G_HD,
            G_ROT,
            RopePairing::HalfHead,
            THETA_GLOBAL,
        ),
    ];
    for (what, heads, hd, rot, pairing, theta) in shapes {
        let cols = (heads * hd) as usize;
        let table = match pairing {
            RopePairing::HalfHead => rotary_rows_with_freq_dim(&[191, 192], rot, hd, theta),
            _ => rotary_rows(&[191, 192], rot, theta),
        };
        let up = |v: &[u16]| DeviceBuffer::upload(&device, v).expect("upload");
        let (xb, tb) = (up(&bf16_inputs(2 * cols, 0x0E12_0000)), up(&table));
        let out = output(&device, 2 * cols);
        let mut line = format!("rope time {what} cols={cols}:");
        for rows in [1u32, 2] {
            let us = event_median(|| {
                // SAFETY: as `run_rope_paired`'s; every buffer holds 2 rows.
                unsafe {
                    rope_paired_bf16(
                        xb.ptr(),
                        tb.ptr(),
                        out.ptr(),
                        (rows, cols as u32),
                        (hd, rot, pairing),
                        std::ptr::null_mut(),
                    )
                }
                .expect("a timed rope launch");
            });
            let kernel = if rows == 1 { "one-row" } else { "16-row" };
            line += &format!(" rows={rows} ({kernel}) {us:.2} us;");
        }
        eprintln!("{line}");
    }
}

/// GateApply over `(rows, cols)`; the output's first `cols` and whether the tail held.
fn run_apply_at(
    device: &CudaDevice,
    (a, g): (&[u16], &[u16]),
    (rows, cols): (usize, usize),
) -> (Vec<u16>, bool) {
    let up = |v: &[u16]| DeviceBuffer::upload(device, v).expect("upload");
    let (ab, gb) = (up(a), up(g));
    let out = output(device, a.len());
    // SAFETY: cudaMalloc pointers are 256-byte aligned; `a`, `g` and `out` hold rows·cols bf16;
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
    let (all, tail) = result(&out, a.len());
    (all[..cols].to_vec(), tail)
}

/// GateSplit of `qg` (`[rows, 2 · half_cols]`); its two outputs' first `half_cols`, and whether
/// both tails held.
fn run_split_at(
    device: &CudaDevice,
    qg: &[u16],
    (rows, half_cols): (usize, usize),
) -> (Vec<u16>, Vec<u16>, bool) {
    let qgb = DeviceBuffer::upload(device, qg).expect("upload qg");
    let (q, gate) = (
        output(device, rows * half_cols),
        output(device, rows * half_cols),
    );
    // SAFETY: cudaMalloc pointers are 256-byte aligned; `qg` holds rows·2·half_cols bf16, `q` and
    // `gate` rows·half_cols each, their own allocations; null is the default stream.
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
    let ((qa, qt), (ga, gt)) = (
        result(&q, rows * half_cols),
        result(&gate, rows * half_cols),
    );
    (qa[..half_cols].to_vec(), ga[..half_cols].to_vec(), qt && gt)
}

/// FP8 S8 W1: at `rows == 1` GateApply (one CTA per 4 column tiles) and GateSplit (one CTA per
/// head) take their one-row form, and the row is the 16-row kernel's bits, as row 0 of a 16-row
/// and a 2-row launch. GateApply at the 27B's 6,144 and 4,096 wide and 2,112 (a partial last
/// tile group), gates random and at ±88 and ±inf; GateSplit at the 27B's 24 heads and the 35B's
/// 16. 3 seeds each.
#[test]
fn the_one_row_gates_are_the_16_row_gates_bit_for_bit() {
    let device = h100();
    let mut pass = true;
    for cols in [6144usize, 4096, 2112] {
        for seed in 0..3u64 {
            let a = bf16_inputs(cols, 0x0E41_0000 ^ (cols as u64) << 4 ^ seed);
            let mut g = bf16_inputs(cols, 0x0E42_0000 ^ (cols as u64) << 4 ^ seed);
            for (i, v) in [88.0f32, -88.0, f32::INFINITY, f32::NEG_INFINITY]
                .into_iter()
                .enumerate()
            {
                g[i] = bits(v);
            }
            let (one, t1) = run_apply_at(&device, (&a, &g), (1, cols));
            let mut ok = t1;
            let mut line = String::new();
            for rows in [16usize, 2] {
                let (want, t) = run_apply_at(
                    &device,
                    (&as_row0(&a, rows), &as_row0(&g, rows)),
                    (rows, cols),
                );
                let m = mismatched_bits(&one, &want);
                line += &format!(" vs rows={rows} mismatched_bits={m}");
                ok &= t && m == 0;
            }
            eprintln!(
                "one-row gate_apply cols={cols} seed={seed}:{line} tail_intact={t1} {}",
                verdict(ok)
            );
            pass &= ok;
        }
    }
    for heads in [24usize, 16] {
        let half = heads * HD as usize;
        for seed in 0..3u64 {
            let qg = bf16_inputs(2 * half, 0x0E43_0000 ^ (heads as u64) << 4 ^ seed);
            let (q1, g1, t1) = run_split_at(&device, &qg, (1, half));
            let mut ok = t1;
            let mut line = String::new();
            for rows in [16usize, 2] {
                let (q, g, t) = run_split_at(&device, &as_row0(&qg, rows), (rows, half));
                let m = mismatched_bits(&q1, &q) + mismatched_bits(&g1, &g);
                line += &format!(" vs rows={rows} mismatched_bits={m}");
                ok &= t && m == 0;
            }
            eprintln!(
                "one-row gate_split heads={heads} seed={seed}:{line} tails_intact={t1} {}",
                verdict(ok)
            );
            pass &= ok;
        }
    }
    assert!(pass, "a one-row gate differs from the 16-row gate");
}

/// Recorded, not asserted: µs per launch of GateApply (the 27B's 6,144) and GateSplit (its 24
/// heads) at `rows == 1` (the one-row form) and `rows == 2` (today's kernel on one CTA), hot.
#[test]
fn the_one_row_gate_times_are_recorded() {
    let device = h100();
    let null = std::ptr::null_mut();
    let up = |v: &[u16]| DeviceBuffer::upload(&device, v).expect("upload");
    let cols = 6144usize;
    let (ab, gb) = (
        up(&bf16_inputs(2 * cols, 0x0E44_0000)),
        up(&bf16_inputs(2 * cols, 0x0E45_0000)),
    );
    let out = output(&device, 2 * cols);
    let mut line = format!("gate time gate_apply cols={cols}:");
    for rows in [1u32, 2] {
        let us = event_median(|| {
            // SAFETY: as `run_apply_at`'s; every buffer holds 2 rows.
            unsafe { gate_apply_bf16(ab.ptr(), gb.ptr(), out.ptr(), (rows, cols as u32), null) }
                .expect("a timed gate_apply launch");
        });
        line += &format!(
            " rows={rows} ({}) {us:.2} us;",
            if rows == 1 { "one-row" } else { "16-row" }
        );
    }
    eprintln!("{line}");
    let half = 24 * HD as usize;
    let qgb = up(&bf16_inputs(2 * 2 * half, 0x0E46_0000));
    let (q, gate) = (output(&device, 2 * half), output(&device, 2 * half));
    let mut line = "gate time gate_split heads=24:".to_string();
    for rows in [1u32, 2] {
        let us = event_median(|| {
            // SAFETY: as `run_split_at`'s; every buffer holds 2 rows.
            unsafe {
                gate_split_bf16(
                    qgb.ptr(),
                    q.ptr(),
                    gate.ptr(),
                    (rows, half as u32),
                    HD,
                    null,
                )
            }
            .expect("a timed gate_split launch");
        });
        line += &format!(
            " rows={rows} ({}) {us:.2} us;",
            if rows == 1 { "one-row" } else { "16-row" }
        );
    }
    eprintln!("{line}");
}

// ── FP8 S8 W4c: the one-row rope append ────────────────────────────

/// The geometries the fused rope append runs at: q heads, kv heads, head dim, rotated width,
/// pairing, `rope_theta` (and so the table), and whether its head norms run (`NormedRope`: Gemma's
/// sliding layers, and its global ones where the fold takes them).
#[derive(Clone, Copy)]
struct W4cShape {
    what: &'static str,
    q_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    rot: u32,
    half: bool,
    theta: f64,
    norms: bool,
}

impl W4cShape {
    const fn geom(self) -> (u32, u32, u32, u32, bool) {
        (
            self.q_heads,
            self.kv_heads,
            self.head_dim,
            self.rot,
            self.half,
        )
    }
}

const fn w4c_shape(
    what: &'static str,
    (q_heads, kv_heads, head_dim, rot, half): (u32, u32, u32, u32, bool),
    theta: f64,
    norms: bool,
) -> W4cShape {
    W4cShape {
        what,
        q_heads,
        kv_heads,
        head_dim,
        rot,
        half,
        theta,
        norms,
    }
}

const W4C_SHAPES: [W4cShape; 5] = [
    w4c_shape(
        "Gemma sliding, normed",
        (16, 8, 256, 256, false),
        THETA_LOCAL,
        true,
    ),
    w4c_shape("Gemma global", (16, 2, 512, 128, true), THETA_GLOBAL, false),
    w4c_shape(
        "Gemma global, normed",
        (16, 2, 512, 128, true),
        THETA_GLOBAL,
        true,
    ),
    w4c_shape("27B", (24, 4, 256, 64, false), THETA, false),
    w4c_shape("35B", (16, 2, 256, 64, false), THETA, false),
];
/// The KV pool the rope append tests write into, and the slot they use.
const W4C_BLOCKS: u32 = 4;
const W4C_EPS: f32 = 1e-6;

/// One case's device buffers: the raw projections, the table, the slot, the caches and the gains.
struct W4c {
    q: DeviceBuffer,
    k: DeviceBuffer,
    v: DeviceBuffer,
    table: DeviceBuffer,
    slot: DeviceBuffer,
    q_gain: DeviceBuffer,
    k_gain: DeviceBuffer,
    qc: usize,
    kc: usize,
}

fn w4c_case(device: &CudaDevice, sh: W4cShape, position: u32, seed: u64) -> W4c {
    let (qh, kvh, hd, rot, half) = sh.geom();
    let theta = sh.theta;
    use scratchy_target_cuda::rotary::rotary_rows_with_freq_dim;
    let (qc, kc) = ((qh * hd) as usize, (kvh * hd) as usize);
    let up = |v: &[u16]| DeviceBuffer::upload(device, v).expect("upload");
    let table = match half {
        true => rotary_rows_with_freq_dim(&[position], rot, hd, theta),
        false => rotary_rows(&[position], rot, theta),
    };
    let slot = [position % (W4C_BLOCKS * 64)];
    W4c {
        q: up(&bf16_inputs(qc, seed)),
        k: up(&bf16_inputs(kc, seed ^ 0x11)),
        v: up(&bf16_inputs(kc, seed ^ 0x22)),
        table: up(&table),
        slot: DeviceBuffer::upload_bytes(device, &slot[0].to_le_bytes()).expect("slot"),
        q_gain: up(&bf16_inputs(hd as usize, seed ^ 0x33)),
        k_gain: up(&bf16_inputs(hd as usize, seed ^ 0x44)),
        qc,
        kc,
    }
}

fn w4c_caches(device: &CudaDevice, kvh: u32, hd: u32) -> (DeviceBuffer, DeviceBuffer) {
    let bytes = (W4C_BLOCKS * kvh * 64 * hd) as usize;
    let z = || DeviceBuffer::upload_bytes(device, &vec![0x5Au8; bytes]).expect("cache");
    (z(), z())
}

/// The chain the fused launch replaces, as today's launches: (normed) RmsNorm(q heads), RmsNorm(k
/// heads), RmsNormUnit(v heads), then Rope(q), and RopeAppend = Rope(k) + kv_append_fp8. Returns
/// q's rope and k's rope (the attention's operands), and leaves the caches written.
fn w4c_chain(
    device: &CudaDevice,
    c: &W4c,
    (qh, kvh, hd, rot, half): (u32, u32, u32, u32, bool),
    norms: bool,
    caches: (&DeviceBuffer, &DeviceBuffer),
) -> (Vec<u16>, Vec<u16>) {
    use scratchy_target_cuda::kernels::{
        kv_append_fp8, rms_norm_bf16, rms_norm_unit_bf16, rope_paired_bf16,
    };
    use scratchy_target_cuda::step::RopePairing;
    let pairing = if half {
        RopePairing::HalfHead
    } else {
        RopePairing::HalfRotated
    };
    let null = std::ptr::null_mut();
    let buf = |n: usize| DeviceBuffer::upload(device, &vec![0u16; n]).expect("buffer");
    let (qn, kn, vn) = (buf(c.qc), buf(c.kc), buf(c.kc));
    let (qr, kr) = (buf(c.qc), buf(c.kc));
    // SAFETY: every buffer holds what the launch reads or writes, each its own allocation; null is
    // the default stream, which the downloads wait for.
    unsafe {
        let (q, k, v) = if norms {
            rms_norm_bf16(
                c.q.ptr(),
                c.q_gain.ptr(),
                qn.ptr(),
                (qh, hd),
                W4C_EPS,
                0.0,
                null,
            )
            .expect("q norm");
            rms_norm_bf16(
                c.k.ptr(),
                c.k_gain.ptr(),
                kn.ptr(),
                (kvh, hd),
                W4C_EPS,
                0.0,
                null,
            )
            .expect("k norm");
            rms_norm_unit_bf16(c.v.ptr(), vn.ptr(), (kvh, hd), W4C_EPS, null).expect("v norm");
            (qn.ptr(), kn.ptr(), vn.ptr())
        } else {
            (c.q.ptr(), c.k.ptr(), c.v.ptr())
        };
        rope_paired_bf16(
            q,
            c.table.ptr(),
            qr.ptr(),
            (1, c.qc as u32),
            (hd, rot, pairing),
            null,
        )
        .expect("q rope");
        rope_paired_bf16(
            k,
            c.table.ptr(),
            kr.ptr(),
            (1, c.kc as u32),
            (hd, rot, pairing),
            null,
        )
        .expect("k rope");
        kv_append_fp8(
            (kr.ptr(), v),
            c.slot.ptr(),
            (caches.0.ptr(), caches.1.ptr()),
            (1, kvh, hd),
            (64, W4C_BLOCKS),
            null,
        )
        .expect("append");
    }
    (qr.download().expect("qr"), kr.download().expect("kr"))
}

/// The fused launch on the same inputs; q's and k's ropes, the caches written.
fn w4c_fused(
    device: &CudaDevice,
    c: &W4c,
    (qh, kvh, hd, rot, half): (u32, u32, u32, u32, bool),
    norms: bool,
    caches: (&DeviceBuffer, &DeviceBuffer),
) -> (Vec<u16>, Vec<u16>) {
    let (qr, kr) = (
        DeviceBuffer::upload(device, &vec![SENTINEL; c.qc]).expect("qr"),
        DeviceBuffer::upload(device, &vec![SENTINEL; c.kc]).expect("kr"),
    );
    w4c_launch(c, (qh, kvh, hd, rot, half), norms, caches, (&qr, &kr)).expect("fused");
    (qr.download().expect("qr"), kr.download().expect("kr"))
}

fn w4c_launch(
    c: &W4c,
    (qh, kvh, hd, rot, half): (u32, u32, u32, u32, bool),
    norms: bool,
    caches: (&DeviceBuffer, &DeviceBuffer),
    (qr, kr): (&DeviceBuffer, &DeviceBuffer),
) -> Result<(), KernelError> {
    use scratchy_target_cuda::kernels::rope_append_fp8;
    use scratchy_target_cuda::step::RopePairing;
    let pairing = if half {
        RopePairing::HalfHead
    } else {
        RopePairing::HalfRotated
    };
    let gains = norms.then(|| {
        (
            c.q_gain.ptr().cast_const(),
            c.k_gain.ptr().cast_const(),
            W4C_EPS,
        )
    });
    // SAFETY: as `w4c_chain`'s; the outputs are their own allocations.
    unsafe {
        rope_append_fp8(
            (c.q.ptr(), qr.ptr(), c.k.ptr(), kr.ptr(), c.v.ptr()),
            (c.table.ptr(), c.slot.ptr()),
            (caches.0.ptr(), caches.1.ptr()),
            (qh, kvh, hd, rot, pairing),
            (64, W4C_BLOCKS),
            gains,
            std::ptr::null_mut(),
        )
    }
}

/// FP8 S8 W4c's gate, measured before anything else (the user's rule: stop unless the fused launch
/// is faster than the stages it replaces): the fused rope append's µs per launch beside the sum of
/// the launches it replaces, by CUDA events, hot, at each geometry. `FASTER` or `SLOWER` per line;
/// recorded, not asserted (the clock ramp moves a run by about 35%).
#[test]
fn the_fused_rope_append_time_is_recorded() {
    use scratchy_target_cuda::kernels::{
        kv_append_fp8, rms_norm_bf16, rms_norm_unit_bf16, rope_paired_bf16,
    };
    use scratchy_target_cuda::step::RopePairing;
    let device = h100();
    for sh in W4C_SHAPES {
        let (what, norms) = (sh.what, sh.norms);
        let (qh, kvh, hd, rot, half) = sh.geom();
        let c = w4c_case(&device, sh, 191, 0x4C00);
        let caches = w4c_caches(&device, kvh, hd);
        let buf = |n: usize| DeviceBuffer::upload(&device, &vec![0u16; n]).expect("buffer");
        let (qn, kn, vn, qr, kr) = (buf(c.qc), buf(c.kc), buf(c.kc), buf(c.qc), buf(c.kc));
        let pairing = if half {
            RopePairing::HalfHead
        } else {
            RopePairing::HalfRotated
        };
        let null = std::ptr::null_mut();
        let shape = (qh, kvh, hd, rot, half);
        let fused = event_median(|| {
            w4c_launch(&c, shape, norms, (&caches.0, &caches.1), (&qr, &kr)).expect("fused");
        });
        // SAFETY (every stage below): as `w4c_chain`'s.
        let stage = |f: &mut dyn FnMut()| event_median(f);
        let mut sum = 0.0;
        let mut line = format!("rope append time {what}: fused {fused:.2} us; stages");
        if norms {
            let a = stage(&mut || unsafe {
                rms_norm_bf16(
                    c.q.ptr(),
                    c.q_gain.ptr(),
                    qn.ptr(),
                    (qh, hd),
                    W4C_EPS,
                    0.0,
                    null,
                )
                .expect("q norm");
            });
            let b = stage(&mut || unsafe {
                rms_norm_bf16(
                    c.k.ptr(),
                    c.k_gain.ptr(),
                    kn.ptr(),
                    (kvh, hd),
                    W4C_EPS,
                    0.0,
                    null,
                )
                .expect("k norm");
            });
            let u = stage(&mut || unsafe {
                rms_norm_unit_bf16(c.v.ptr(), vn.ptr(), (kvh, hd), W4C_EPS, null).expect("v norm");
            });
            sum += a + b + u;
            line += &format!(" q norm {a:.2} + k norm {b:.2} + v unit norm {u:.2}");
        }
        let qrope = stage(&mut || unsafe {
            rope_paired_bf16(
                c.q.ptr(),
                c.table.ptr(),
                qr.ptr(),
                (1, c.qc as u32),
                (hd, rot, pairing),
                null,
            )
            .expect("q rope");
        });
        let krope = stage(&mut || unsafe {
            rope_paired_bf16(
                c.k.ptr(),
                c.table.ptr(),
                kr.ptr(),
                (1, c.kc as u32),
                (hd, rot, pairing),
                null,
            )
            .expect("k rope");
        });
        let append = stage(&mut || unsafe {
            kv_append_fp8(
                (kr.ptr(), c.v.ptr()),
                c.slot.ptr(),
                (caches.0.ptr(), caches.1.ptr()),
                (1, kvh, hd),
                (64, W4C_BLOCKS),
                null,
            )
            .expect("append");
        });
        sum += qrope + krope + append;
        let verdict = if fused < sum { "FASTER" } else { "SLOWER" };
        eprintln!(
            "{line} + q rope {qrope:.2} + k rope {krope:.2} + append {append:.2} = {sum:.2} us; \
             saved {:.2} us {verdict} (recorded, not asserted)",
            sum - fused
        );
    }
}

/// FP8 S8 W4c: the fused rope append is its chain bit for bit, at Gemma's sliding geometry (16 q
/// heads × 256, 8 kv heads, the full-width rope, normed: RmsNorm(q), RmsNorm(k), RmsNormUnit(v),
/// Rope, RopeAppend), its global one (16 × 512, 2 kv heads, `<128, true>`, unnormed and normed:
/// two 256-column tiles a head view) and the 27B's
/// and 35B's (`<64, false>`), at positions 0, 1, 63, 64, 191, 255, 4,095, 8,191 and 100,003 (the
/// slot `position % 256`: block edges and the pool's last slot), and at the two slots that append
/// nothing (`u32::MAX`, and block 4 of a 4-block pool), 3 seeds: `mismatched_bits=0` on q's rope
/// and k's rope, and 0 differing bytes over both whole caches (the written slot and every other
/// byte).
#[test]
fn the_normed_rope_is_its_chain_bit_for_bit() {
    let device = h100();
    let mut pass = true;
    for sh in W4C_SHAPES {
        let (kvh, hd) = (sh.kv_heads, sh.head_dim);
        let cases = [0u32, 1, 63, 64, 191, 255, 4095, 8191, 100_003]
            .map(|p| (p, None))
            .into_iter()
            .chain([(191, Some(u32::MAX)), (191, Some(W4C_BLOCKS * 64))]);
        for (position, slot) in cases {
            for seed in 0..3u64 {
                let mut c = w4c_case(
                    &device,
                    sh,
                    position,
                    0x4C10 ^ seed << 8 ^ u64::from(position),
                );
                if let Some(s) = slot {
                    c.slot = DeviceBuffer::upload_bytes(&device, &s.to_le_bytes()).expect("slot");
                }
                let (a, b) = (w4c_caches(&device, kvh, hd), w4c_caches(&device, kvh, hd));
                let (cq, ck) = w4c_chain(&device, &c, sh.geom(), sh.norms, (&a.0, &a.1));
                let (fq, fk) = w4c_fused(&device, &c, sh.geom(), sh.norms, (&b.0, &b.1));
                let bytes = |x: &DeviceBuffer, y: &DeviceBuffer| {
                    let (x, y) = (x.download().expect("cache"), y.download().expect("cache"));
                    x.iter().zip(&y).filter(|(p, q)| p != q).count()
                };
                let bad_q = mismatched_bits(&fq, &cq);
                let bad_k = mismatched_bits(&fk, &ck);
                let bad_cache = bytes(&a.0, &b.0) + bytes(&a.1, &b.1);
                let ok = bad_q == 0 && bad_k == 0 && bad_cache == 0;
                let slot = slot.map_or("position % 256".to_string(), |s| s.to_string());
                eprintln!(
                    "W4 rope append {} pos={position} slot={slot} seed={seed}: q mismatched_bits={bad_q} k \
                     mismatched_bits={bad_k} cache mismatched_bytes={bad_cache} {}",
                    sh.what,
                    verdict(ok)
                );
                pass &= ok;
            }
        }
    }
    assert!(pass, "the fused rope append differs from its chain");
}

// ── FP8 S8 R1: the four-warp rope append (`rope_append_wide_ker`) ─────────────

/// The rope append (the four-warp kernel the launcher runs) on `stream`.
fn r1_launch(
    c: &W4c,
    (qh, kvh, hd, rot, half): (u32, u32, u32, u32, bool),
    norms: bool,
    caches: (&DeviceBuffer, &DeviceBuffer),
    (qr, kr): (&DeviceBuffer, &DeviceBuffer),
    stream: *mut std::ffi::c_void,
) -> Result<(), KernelError> {
    use scratchy_target_cuda::kernels::rope_append_fp8;
    use scratchy_target_cuda::step::RopePairing;
    let pairing = if half {
        RopePairing::HalfHead
    } else {
        RopePairing::HalfRotated
    };
    let gains = norms.then(|| {
        (
            c.q_gain.ptr().cast_const(),
            c.k_gain.ptr().cast_const(),
            W4C_EPS,
        )
    });
    // SAFETY: as `w4c_chain`'s; the outputs are their own allocations, `stream` live or null.
    unsafe {
        rope_append_fp8(
            (c.q.ptr(), qr.ptr(), c.k.ptr(), kr.ptr(), c.v.ptr()),
            (c.table.ptr(), c.slot.ptr()),
            (caches.0.ptr(), caches.1.ptr()),
            (qh, kvh, hd, rot, pairing),
            (64, W4C_BLOCKS),
            gains,
            stream,
        )
    }
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
/// [`event_median`]'s events (the legacy default stream orders them): the replay's median µs per
/// node.
fn graph_per_node(launch: impl Fn(*mut std::ffi::c_void)) -> f64 {
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
    let us = event_median(|| {
        // SAFETY: `exec` is the instantiated graph and `stream` live.
        assert_eq!(unsafe { cudaGraphLaunch(exec, stream) }, 0, "graph launch");
    });
    // SAFETY: the handles were created above and are no longer used.
    unsafe {
        cudaGraphExecDestroy(exec);
        cudaGraphDestroy(graph);
        cudaStreamDestroy(stream);
    }
    us / 10.0
}

/// Recorded, not asserted (R1's gate reads it): inside a captured graph of 10 back-to-back nodes,
/// the rope append (`four_warp`, the launcher's kernel), at every W4c shape, two rounds, as
/// `R1_TIME` lines.
#[test]
fn the_rope_append_graph_times_are_recorded() {
    let device = h100();
    for sh in W4C_SHAPES {
        let c = w4c_case(&device, sh, 191, 0x0E1F_7000);
        let caches = w4c_caches(&device, sh.kv_heads, sh.head_dim);
        let buf = |n: usize| DeviceBuffer::upload(&device, &vec![0u16; n]).expect("buffer");
        let (qr, kr) = (buf(c.qc), buf(c.kc));
        let run = || {
            graph_per_node(|s| {
                r1_launch(
                    &c,
                    sh.geom(),
                    sh.norms,
                    (&caches.0, &caches.1),
                    (&qr, &kr),
                    s,
                )
                .expect("the rope append launches");
            })
        };
        for round in 1..=2 {
            let b = run();
            eprintln!(
                "R1_TIME {} round {round}: four_warp={b:.2} us per node",
                sh.what
            );
        }
    }
}
