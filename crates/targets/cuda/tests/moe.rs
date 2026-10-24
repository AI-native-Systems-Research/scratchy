// SPDX-License-Identifier: Apache-2.0
//! The qwen3-5-moe block's kernels (AI-native-Systems-Research/scratchy#857) on the H100, against
//! f64 CPU references and exact host results, and their launchers' refusals.
//!
//! Built only under `nvcc` (`required-features`). No GPU fails rather than skips: this test is the
//! kernels' only gate. Every case prints its line before a test fails, so one run reports all the
//! numbers. Each bound is #841's, stated with the case that uses it.

// `pub`: each test binary uses a different part of `common`, and nothing in it is dead code.
pub mod common {
    pub mod numerics;
}

use common::numerics::{bf16_inputs, bits, mismatched_bits};
use half::bf16;
use scratchy_target_cuda::kernels::{KernelError, gate_scale_bf16};
use scratchy_target_cuda::runtime::{CudaDevice, DeviceBuffer};

/// Extra output elements past the shape, which a kernel must leave untouched.
const TAIL: usize = 4096;
/// Every output element before a launch: a bf16 NaN, so an element a kernel never writes fails.
const SENTINEL: u16 = 0xFFFF;
/// qwen3-5-moe's hidden size.
const HIDDEN: usize = 2048;

fn h100() -> CudaDevice {
    CudaDevice::open()
        .unwrap_or_else(|e| panic!("the `nvcc` feature's tests need an sm_90 GPU (H100): {e}"))
}

fn verdict(pass: bool) -> &'static str {
    if pass { "PASS" } else { "FAIL" }
}

fn up(device: &CudaDevice, host: &[u16]) -> DeviceBuffer {
    DeviceBuffer::upload(device, host).expect("upload")
}

fn f(b: u16) -> f64 {
    f64::from(bf16::from_bits(b).to_f32())
}

// ── M6 GateScale ───────────────────────────────────────────────────

/// Run the gate scale over `(rows, cols)`; the output and whether the tail past it held.
fn run_gate_scale(
    device: &CudaDevice,
    (routed, shared, g): (&[u16], &[u16], &[u16]),
    (rows, cols): (usize, usize),
) -> (Vec<u16>, bool) {
    let (rb, sb, gb) = (up(device, routed), up(device, shared), up(device, g));
    let out = up(device, &vec![SENTINEL; rows * cols + TAIL]);
    // SAFETY: cudaMalloc pointers are 256-byte aligned; `routed`, `shared` hold rows·cols bf16,
    // `g` rows, `out` rows·cols and the tail; `out` is its own allocation; null is the default
    // stream, which the download below waits for.
    unsafe {
        gate_scale_bf16(
            (rb.ptr(), sb.ptr(), gb.ptr()),
            out.ptr(),
            (rows as u32, cols as u32),
            std::ptr::null_mut(),
        )
    }
    .expect("gate_scale_bf16 launch");
    let mut o = out.download().expect("download the output");
    let tail = o[rows * cols..].iter().all(|&b| b == SENTINEL);
    o.truncate(rows * cols);
    (o, tail)
}

/// `routed + shared · σ(g)` at qwen3-5-moe's hidden size and 1, 7 and 512 rows, gates in [-8, 8]
/// plus the extremes ±100 on row 0: every element within #841's bound `2⁻⁸·|ref| +
/// 2⁻¹⁸·|shared·σ(g)| + 2⁻²⁴`, no output non-finite, the tail intact; a repeat bit-identical; and
/// the launcher refusing `cols` off its 8-wide rule and zero dimensions with `out` unchanged.
#[test]
fn gate_scale_bf16_matches_the_f64_reference() {
    let device = h100();
    let mut pass = true;
    for rows in [1usize, 7, 512] {
        let cols = HIDDEN;
        let seed = 0x6A5C_0000 ^ rows as u64;
        let routed = bf16_inputs(rows * cols, seed);
        let shared = bf16_inputs(rows * cols, seed ^ 1);
        let mut g: Vec<u16> = (bf16_inputs(rows, seed ^ 2).into_iter())
            .map(|b| bits(bf16::from_bits(b).to_f32() * 8.0))
            .collect();
        g[0] = bits(if rows == 7 { -100.0 } else { 100.0 });
        let (gpu, tail) = run_gate_scale(&device, (&routed, &shared, &g), (rows, cols));
        let (mut bad, mut worst, mut nonfinite) = (0usize, 0.0f64, 0usize);
        for (r, &gr) in g.iter().enumerate() {
            let sig = 1.0 / (1.0 + (-f(gr)).exp());
            for c in 0..cols {
                let i = r * cols + c;
                let scaled = f(shared[i]) * sig;
                let want = f(routed[i]) + scaled;
                let bound = want.abs() / 256.0 + scaled.abs() / 262_144.0 + 2f64.powi(-24);
                let got = f(gpu[i]);
                nonfinite += usize::from(!got.is_finite());
                let err = (got - want).abs();
                bad += usize::from(err.is_nan() || err > bound);
                worst = worst.max(err / bound);
            }
        }
        let repeat = run_gate_scale(&device, (&routed, &shared, &g), (rows, cols)).0;
        let rep = mismatched_bits(&gpu, &repeat);
        let ok = bad == 0 && nonfinite == 0 && tail && rep == 0;
        eprintln!(
            "gate_scale_bf16 rows={rows} cols={cols}: out_of_bound={bad} worst_ratio={worst:.4} \
             nonfinite={nonfinite} tail_intact={tail} repeat_mismatched_bits={rep} {}",
            verdict(ok)
        );
        pass &= ok;
    }
    for (rows, cols) in [(7u32, 2044u32), (7, 0), (0, 2048), (65_536, 65_536)] {
        let len = (rows as usize * cols as usize).clamp(8, 1 << 16);
        let x = up(&device, &vec![0; len]);
        let out = up(&device, &vec![SENTINEL; len]);
        // SAFETY: the launcher refuses these shapes before it touches any pointer.
        let got = unsafe {
            gate_scale_bf16(
                (x.ptr(), x.ptr(), x.ptr()),
                out.ptr(),
                (rows, cols),
                std::ptr::null_mut(),
            )
        };
        let kept = (out.download().expect("out").iter()).all(|&b| b == SENTINEL);
        let ok = got == Err(KernelError::RowShape { rows, cols }) && kept;
        eprintln!(
            "gate_scale_bf16 refusal rows={rows} cols={cols}: {got:?} out_unchanged={kept} {}",
            verdict(ok)
        );
        pass &= ok;
    }
    assert!(pass, "the gate scale failed: see the lines above");
}
