// SPDX-License-Identifier: Apache-2.0
//! The bf16 GEMM (`kernels/gemm_bf16.cu`, `C = A · Bᵀ`) against a CPU reference, on the H100.
//!
//! Built only under `nvcc` (`required-features`). No GPU fails rather than skips: this test is the
//! kernel's only gate, and a skip that passes would let it pass silently.
//!
//! The reference widens the same bf16 inputs to f64 and accumulates in f64. Each element must
//! satisfy `|gpu − ref| ≤ 2⁻⁷·|ref| + 2⁻¹⁴·S`, with `S = Σₖ |A[i,k]·B[j,k]|`: the relative term is
//! 2× the half-ulp the single bf16 rounding on store costs, and the absolute term covers f32
//! accumulation-order error (about √K·2⁻²⁴·S) with a wide margin. A dropped or repeated K step, or
//! a wrong-layout B operand, moves an element far past the bound. NaN or Inf fails.

mod common {
    pub mod cudart;
}

use common::cudart::{self, DeviceBuffer};
use half::bf16;
use scratchy_target_cuda::kernels::{KernelError, gemm_bf16_abt};

/// `(M, N, K)`, each a multiple of the kernel's (128, 256, 64).
const SHAPES: &[(u32, u32, u32, &str)] = &[
    (128, 256, 64, "one output tile, one K step"),
    (
        128,
        256,
        4096,
        "one tile, 64 K steps: the 4-stage input ring wraps many times",
    ),
    (
        1280,
        512,
        256,
        "10 row blocks with SUPER_M = 8: the remainder branch of the tile order",
    ),
    (
        2048,
        2560,
        64,
        "160 tiles on 132 blocks: the persistent loop's second pass",
    ),
    (
        256,
        1024,
        4096,
        "qwen3.5-9b K/V projection at a 256-token prefill",
    ),
    (128, 4096, 12288, "qwen3.5-9b down-projection"),
];

/// Shapes the launcher must refuse, launching nothing.
const REFUSED: &[(u32, u32, u32)] = &[(100, 256, 64), (128, 200, 64), (128, 256, 100)];

/// Extra `C` elements past `m*n`, which the kernel must leave untouched.
const TAIL: usize = 4096;
/// Every byte of `C` before the launch. As bf16, `0xFFFF` is a NaN, so an output element the
/// kernel never writes also fails the tolerance check.
const SENTINEL_BYTE: u8 = 0xFF;
const SENTINEL: u16 = 0xFFFF;

/// Panic unless device 0 exists and is compute capability 9.0 (the kernel is `sm_90a` only).
fn require_h100() {
    let cc = cudart::compute_capability();
    assert_eq!(
        cc,
        (9, 0),
        "the `nvcc` feature's tests need an sm_90 GPU (H100); device 0 is compute capability {}.{}",
        cc.0,
        cc.1
    );
}

/// splitmix64, mapped to bf16 in [-1, 1].
fn bf16_inputs(len: usize, seed: u64) -> Vec<u16> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            let unit = (z >> 11) as f64 / (1u64 << 53) as f64; // [0, 1)
            bf16::from_f64(unit * 2.0 - 1.0).to_bits()
        })
        .collect()
}

fn widen(bits: &[u16]) -> Vec<f64> {
    bits.iter()
        .map(|&b| f64::from(bf16::from_bits(b).to_f32()))
        .collect()
}

/// `(ref, S)` for every element of `C = A · Bᵀ`, row-major, split across threads by row.
fn reference(a: &[f64], b: &[f64], m: usize, n: usize, k: usize) -> Vec<(f64, f64)> {
    let mut out = vec![(0.0, 0.0); m * n];
    let threads = std::thread::available_parallelism().map_or(1, |p| p.get());
    let rows_per = m.div_ceil(threads);
    std::thread::scope(|s| {
        for (chunk, rows) in out.chunks_mut(rows_per * n).enumerate() {
            s.spawn(move || {
                for (r, row) in rows.chunks_mut(n).enumerate() {
                    let ar = &a[(chunk * rows_per + r) * k..][..k];
                    for (j, cell) in row.iter_mut().enumerate() {
                        let br = &b[j * k..][..k];
                        let (mut dot, mut abs) = (0.0, 0.0);
                        for (x, y) in ar.iter().zip(br) {
                            dot += x * y;
                            abs += (x * y).abs();
                        }
                        *cell = (dot, abs);
                    }
                }
            });
        }
    });
    out
}

/// Run one shape; print its line; return whether it passed.
fn run_shape(m: u32, n: u32, k: u32, what: &str) -> bool {
    let (mu, nu, ku) = (m as usize, n as usize, k as usize);
    let a_bits = bf16_inputs(mu * ku, 0x5EED_0000 ^ (u64::from(m) << 32) ^ u64::from(k));
    let b_bits = bf16_inputs(nu * ku, 0xB0B0_0000 ^ (u64::from(n) << 32) ^ u64::from(k));
    let a = DeviceBuffer::from_host(&a_bits);
    let b = DeviceBuffer::from_host(&b_bits);
    let c = DeviceBuffer::new(mu * nu + TAIL);
    c.fill_bytes(SENTINEL_BYTE);
    cudart::synchronize();

    // SAFETY: the buffers hold m*k, n*k and m*n + TAIL bf16 elements, cudaMalloc pointers are
    // 256-byte aligned, `c` overlaps neither input, and null is the default stream.
    let launched =
        unsafe { gemm_bf16_abt(a.ptr(), b.ptr(), c.ptr(), m, n, k, std::ptr::null_mut()) };
    if let Err(e) = launched {
        eprintln!("gemm_bf16_abt m={m} n={n} k={k}: launch failed: {e} FAIL");
        return false;
    }
    cudart::synchronize();
    let c_bits = c.to_host();

    let tail_intact = c_bits[mu * nu..].iter().all(|&x| x == SENTINEL);
    let refs = reference(&widen(&a_bits), &widen(&b_bits), mu, nu, ku);
    let (mut max_err, mut sum_err, mut bad) = (0.0f64, 0.0f64, 0usize);
    // The element with the largest error relative to its bound.
    let (mut worst, mut worst_ratio) = ((0, 0, 0.0f64, 0.0f64, 0.0f64), -1.0f64);
    for (idx, (&bits, &(r, s))) in c_bits[..mu * nu].iter().zip(&refs).enumerate() {
        let gpu = f64::from(bf16::from_bits(bits).to_f32());
        let bound = r.abs() / 128.0 + s / 16384.0;
        let err = (gpu - r).abs();
        let ok = gpu.is_finite() && err <= bound;
        if !ok {
            bad += 1;
        }
        let ratio = if gpu.is_finite() {
            err / bound.max(f64::MIN_POSITIVE)
        } else {
            f64::INFINITY
        };
        if ratio > worst_ratio {
            worst_ratio = ratio;
            worst = (idx / nu, idx % nu, gpu, r, bound);
        }
        if gpu.is_finite() {
            max_err = max_err.max(err);
            sum_err += err;
        }
    }
    let pass = bad == 0 && tail_intact;
    let (wi, wj, wgpu, wref, wbound) = worst;
    eprintln!(
        "gemm_bf16_abt m={m} n={n} k={k}: max_abs_err={max_err:.6} mean_abs_err={:.6} \
         worst=({wi},{wj}) gpu={wgpu} ref={wref:.6} bound={wbound:.6} out_of_bound={bad} \
         tail_intact={tail_intact} {} ({what})",
        sum_err / (mu * nu) as f64,
        if pass { "PASS" } else { "FAIL" }
    );
    pass
}

/// Every shape runs and prints before the test fails, so one run reports all the numbers.
#[test]
fn gemm_bf16_abt_matches_the_cpu_reference() {
    require_h100();
    let failed: Vec<_> = SHAPES
        .iter()
        .filter(|&&(m, n, k, what)| !run_shape(m, n, k, what))
        .map(|&(m, n, k, _)| (m, n, k))
        .collect();
    assert!(
        failed.is_empty(),
        "gemm_bf16_abt failed for (m, n, k) = {failed:?}"
    );
}

/// A shape off the (128, 256, 64) grid returns `Shape`, and `C` is left as it was.
#[test]
fn gemm_bf16_abt_refuses_shapes_off_the_tile_grid() {
    require_h100();
    for &(m, n, k) in REFUSED {
        let (mu, nu, ku) = (m as usize, n as usize, k as usize);
        let a = DeviceBuffer::from_host(&vec![0; mu * ku]);
        let b = DeviceBuffer::from_host(&vec![0; nu * ku]);
        let c = DeviceBuffer::new(mu * nu + TAIL);
        c.fill_bytes(SENTINEL_BYTE);
        cudart::synchronize();
        // SAFETY: as in `run_shape`; the launcher refuses before touching any pointer.
        let got =
            unsafe { gemm_bf16_abt(a.ptr(), b.ptr(), c.ptr(), m, n, k, std::ptr::null_mut()) };
        cudart::synchronize();
        let unchanged = c.to_host().iter().all(|&x| x == SENTINEL);
        eprintln!(
            "gemm_bf16_abt m={m} n={n} k={k}: {got:?} c_unchanged={unchanged} {}",
            if got == Err(KernelError::Shape { m, n, k }) && unchanged {
                "PASS"
            } else {
                "FAIL"
            }
        );
        assert_eq!(
            got,
            Err(KernelError::Shape { m, n, k }),
            "m={m} n={n} k={k}"
        );
        assert!(unchanged, "m={m} n={n} k={k}: a refused launch wrote C");
    }
}
