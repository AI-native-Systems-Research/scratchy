// SPDX-License-Identifier: Apache-2.0
//! The bf16 GEMMs (`kernels/gemm_bf16.cu`, the narrow `kernels/gemm_narrow_bf16.cu` and the
//! one-row GEMV `kernels/gemv_bf16.cu`, all `C = A · Bᵀ`) against a CPU reference, on the H100.
//!
//! Built only under `nvcc` (`required-features`). No GPU fails rather than skips: this test is the
//! kernel's only gate, and a skip that passes would let it pass silently. The reference and the
//! bound are `common::numerics`'.

// `pub`: each test binary uses a different part of `common`, and nothing in it is dead code.
pub mod common {
    pub mod numerics;
    pub mod watchdog;
}

use common::numerics::{bf16_inputs, compare, mismatched_bits, reference, widen};
use scratchy_target_cuda::kernels::gemm_col64_bf16_abt;
use scratchy_target_cuda::kernels::{
    KernelError, gemm_bf16_abt, gemm_narrow_bf16_abt, gemv_bf16_abt,
};
use scratchy_target_cuda::kernels::{
    gate_up_gemm_bf16, gate_up_gemv_bf16, gelu_mul_bf16, silu_mul_bf16,
};
use scratchy_target_cuda::runtime::{CudaDevice, DeviceBuffer};
use scratchy_target_cuda::step::GatedOp;

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

/// Shapes the launcher must refuse, launching nothing. The last is above `INT_MAX` in `m`: the
/// launcher refuses it before it touches a pointer, so its buffers are the small ones of
/// [`REFUSED_BUFFER`] elements (an `m·k` allocation would be 256 GiB).
const REFUSED: &[(u32, u32, u32)] = &[
    (100, 256, 64),
    (128, 200, 64),
    (128, 256, 100),
    (2147483648, 256, 64),
];

/// The elements of each buffer a refused launch is given.
const REFUSED_BUFFER: usize = 1 << 16;

/// Extra `C` elements past `m*n`, which the kernel must leave untouched.
const TAIL: usize = 4096;
/// Every element of `C` before the launch. As bf16, `0xFFFF` is a NaN, so an output element the
/// kernel never writes also fails the tolerance check.
const SENTINEL: u16 = 0xFFFF;

/// Device 0, which must be an H100; panics with the reason otherwise.
fn h100() -> CudaDevice {
    common::watchdog::arm();
    CudaDevice::open()
        .unwrap_or_else(|e| panic!("the `nvcc` feature's tests need an sm_90 GPU (H100): {e}"))
}

/// Launch on the default stream, then wait for it.
///
/// # Safety
///
/// As [`gemm_bf16_abt`]'s, for the default stream.
unsafe fn launch_and_wait(
    a: &DeviceBuffer,
    b: &DeviceBuffer,
    c: &DeviceBuffer,
    (m, n, k): (u32, u32, u32),
) -> Result<(), KernelError> {
    // SAFETY: the caller's contract.
    let launched =
        unsafe { gemm_bf16_abt(a.ptr(), b.ptr(), c.ptr(), m, n, k, std::ptr::null_mut()) };
    // Downloading `c` waits for the default stream, and with it this launch; a fault in the
    // kernel surfaces there.
    c.download().expect("waiting for the launch");
    launched
}

/// Run one shape; print its line; return whether it passed.
fn run_shape(device: &CudaDevice, m: u32, n: u32, k: u32, what: &str) -> bool {
    let (mu, nu, ku) = (m as usize, n as usize, k as usize);
    let a_bits = bf16_inputs(mu * ku, 0x5EED_0000 ^ (u64::from(m) << 32) ^ u64::from(k));
    let b_bits = bf16_inputs(nu * ku, 0xB0B0_0000 ^ (u64::from(n) << 32) ^ u64::from(k));
    let a = DeviceBuffer::upload(device, &a_bits).expect("upload A");
    let b = DeviceBuffer::upload(device, &b_bits).expect("upload B");
    let c = DeviceBuffer::upload(device, &vec![SENTINEL; mu * nu + TAIL]).expect("upload C");

    // SAFETY: the buffers hold m*k, n*k and m*n + TAIL bf16 elements, cudaMalloc pointers are
    // 256-byte aligned, `c` overlaps neither input, and `CudaDevice::open` ran `prepare_all`.
    if let Err(e) = unsafe { launch_and_wait(&a, &b, &c, (m, n, k)) } {
        eprintln!("gemm_bf16_abt m={m} n={n} k={k}: launch failed: {e} FAIL");
        return false;
    }
    let c_bits = c.download().expect("download C");
    let tail_intact = c_bits[mu * nu..].iter().all(|&x| x == SENTINEL);
    let cmp = compare(&c_bits[..mu * nu], &a_bits, &b_bits, mu, nu, ku);
    let pass = cmp.pass() && tail_intact;
    eprintln!(
        "gemm_bf16_abt m={m} n={n} k={k}: {} tail_intact={tail_intact} {} ({what})",
        cmp.line(),
        if pass { "PASS" } else { "FAIL" }
    );
    pass
}

/// Every shape runs and prints before the test fails, so one run reports all the numbers.
#[test]
fn gemm_bf16_abt_matches_the_cpu_reference() {
    let device = h100();
    let failed: Vec<_> = SHAPES
        .iter()
        .filter(|&&(m, n, k, what)| !run_shape(&device, m, n, k, what))
        .map(|&(m, n, k, _)| (m, n, k))
        .collect();
    assert!(
        failed.is_empty(),
        "gemm_bf16_abt failed for (m, n, k) = {failed:?}"
    );
}

/// A shape off the (128, 256, 64) grid, or above `INT_MAX`, returns `Shape`, and `C` is left as
/// it was.
#[test]
fn gemm_bf16_abt_refuses_shapes_off_the_tile_grid() {
    let device = h100();
    for &(m, n, k) in REFUSED {
        let len = |rows: u32, cols: u32| (rows as usize * cols as usize).min(REFUSED_BUFFER);
        let a = DeviceBuffer::upload(&device, &vec![0; len(m, k)]).expect("upload A");
        let b = DeviceBuffer::upload(&device, &vec![0; len(n, k)]).expect("upload B");
        let c = DeviceBuffer::upload(&device, &vec![SENTINEL; len(m, n) + TAIL]).expect("upload C");
        // SAFETY: the launcher refuses each of these shapes before it touches any pointer.
        let got = unsafe { launch_and_wait(&a, &b, &c, (m, n, k)) };
        let unchanged = c
            .download()
            .expect("download C")
            .iter()
            .all(|&x| x == SENTINEL);
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

unsafe extern "C" {
    /// Declared here only to provoke an earlier, unrelated, non-sticky error.
    fn cudaDeviceGetAttribute(value: *mut i32, attr: i32, device: i32) -> i32;
}

/// `cudaErrorInvalidDevice`.
const CUDA_ERROR_INVALID_DEVICE: i32 = 101;

/// A launch reports its own result: an error an unrelated call left in the runtime's last-error
/// state is not this launch's. Before the launcher returned `cudaLaunchKernel`'s own status, it
/// read (and cleared) that last-error state after launching, and returned `Cuda(101)` here.
#[test]
fn gemm_bf16_abt_ignores_an_earlier_unrelated_error() {
    let device = h100();
    let mut value = 0;
    // SAFETY: `value` is a valid out-pointer; device 9999 does not exist, which is the point.
    let code = unsafe { cudaDeviceGetAttribute(&mut value, 75, 9999) };
    assert_eq!(
        code, CUDA_ERROR_INVALID_DEVICE,
        "the provoking call must fail with cudaErrorInvalidDevice"
    );
    let pass = run_shape(&device, 128, 256, 64, "right after an unrelated error");
    eprintln!(
        "earlier unrelated error {code}, then a launch: {}",
        if pass { "PASS" } else { "FAIL" }
    );
    assert!(pass, "a launch after an unrelated error failed");
}

// ── The narrow GEMM (n ≤ 64) ───────────────────────────────────────

/// `(M, N, K)` for the narrow GEMM: qwen3.5-0.8b's `a`/`b` projection at a 128-token bucket, 9b's
/// at 256, rows that are no multiple of a warp's CTA with the smallest `k`, and 8 columns.
const NARROW_SHAPES: &[(u32, u32, u32, &str)] = &[
    (128, 16, 1024, "qwen3.5-0.8b a/b projection, 128 tokens"),
    (256, 32, 4096, "qwen3.5-9b a/b projection, 256 tokens"),
    (
        37,
        64,
        8,
        "37 rows (a partial CTA), the widest n, one k vector",
    ),
    (256, 8, 2048, "the narrowest n"),
];

/// Shapes the narrow launcher refuses: `n` not a multiple of 8, above 64, `k` off the vector, zero
/// rows.
const NARROW_REFUSED: &[(u32, u32, u32)] =
    &[(128, 12, 64), (128, 72, 64), (128, 16, 100), (0, 16, 64)];

/// Launch the narrow GEMM on the default stream, then wait for it.
///
/// # Safety
///
/// As [`gemm_narrow_bf16_abt`]'s, for the default stream.
unsafe fn narrow_and_wait(
    a: &DeviceBuffer,
    b: &DeviceBuffer,
    c: &DeviceBuffer,
    (m, n, k): (u32, u32, u32),
) -> Result<(), KernelError> {
    // SAFETY: the caller's contract.
    let launched =
        unsafe { gemm_narrow_bf16_abt(a.ptr(), b.ptr(), c.ptr(), (m, n, k), std::ptr::null_mut()) };
    c.download().expect("waiting for the launch");
    launched
}

/// `B` read as `[k, n]` instead of `[n, k]`: the self-check's wrong layout, `C'[i, j] = Σ_l
/// A[i, l] · B[l·n + j]`, under the GEMM bound of its own products.
fn transposed_misses(gpu: &[u16], a: &[u16], b: &[u16], (m, n, k): (usize, usize, usize)) -> usize {
    let (a, b) = (widen(a), widen(b));
    let mut missed = 0;
    for i in 0..m {
        for j in 0..n {
            let (mut dot, mut abs) = (0.0f64, 0.0f64);
            for l in 0..k {
                let p = a[i * k + l] * b[l * n + j];
                dot += p;
                abs += p.abs();
            }
            let g = f64::from(half::bf16::from_bits(gpu[i * n + j]).to_f32());
            missed += usize::from((g - dot).abs() > dot.abs() / 128.0 + abs / 16384.0);
        }
    }
    missed
}

/// Run one narrow shape; print its line and the transposed-B self-check's; return whether both
/// passed.
fn run_narrow(device: &CudaDevice, m: u32, n: u32, k: u32, what: &str) -> bool {
    let (mu, nu, ku) = (m as usize, n as usize, k as usize);
    let a_bits = bf16_inputs(mu * ku, 0xA11A_0000 ^ (u64::from(m) << 32) ^ u64::from(k));
    let b_bits = bf16_inputs(nu * ku, 0xB11B_0000 ^ (u64::from(n) << 32) ^ u64::from(k));
    let a = DeviceBuffer::upload(device, &a_bits).expect("upload A");
    let b = DeviceBuffer::upload(device, &b_bits).expect("upload B");
    let c = DeviceBuffer::upload(device, &vec![SENTINEL; mu * nu + TAIL]).expect("upload C");
    // SAFETY: the buffers hold m*k, n*k and m*n + TAIL bf16, cudaMalloc pointers are 256-byte
    // aligned and `c` overlaps neither input.
    if let Err(e) = unsafe { narrow_and_wait(&a, &b, &c, (m, n, k)) } {
        eprintln!("gemm_narrow_bf16_abt m={m} n={n} k={k}: launch failed: {e} FAIL");
        return false;
    }
    let c_bits = c.download().expect("download C");
    let tail_intact = c_bits[mu * nu..].iter().all(|&x| x == SENTINEL);
    let cmp = compare(&c_bits[..mu * nu], &a_bits, &b_bits, mu, nu, ku);
    // The wrong layout must miss: for k = 8 and n = 64 the two readings of B share no element in
    // most positions, so at least half of C is out of bound.
    let missed = transposed_misses(&c_bits[..mu * nu], &a_bits, &b_bits, (mu, nu, ku));
    let fraction = missed as f64 / (mu * nu) as f64;
    let self_check = fraction >= 0.5;
    let pass = cmp.pass() && tail_intact && self_check;
    eprintln!(
        "gemm_narrow_bf16_abt m={m} n={n} k={k}: {} tail_intact={tail_intact} \
         transposed_b_out_of_bound={missed}/{} ({fraction:.4}, need >= 0.5) {} ({what})",
        cmp.line(),
        mu * nu,
        if pass { "PASS" } else { "FAIL" }
    );
    pass
}

#[test]
fn gemm_narrow_bf16_abt_matches_the_cpu_reference() {
    let device = h100();
    let failed: Vec<_> = NARROW_SHAPES
        .iter()
        .filter(|&&(m, n, k, what)| !run_narrow(&device, m, n, k, what))
        .map(|&(m, n, k, _)| (m, n, k))
        .collect();
    assert!(
        failed.is_empty(),
        "gemm_narrow_bf16_abt failed for (m, n, k) = {failed:?}"
    );
}

/// Two runs of the same input give the same bits: the reduction order is fixed.
#[test]
fn gemm_narrow_bf16_abt_is_deterministic() {
    let device = h100();
    let (m, n, k) = (256u32, 32u32, 4096u32);
    let (mu, nu, ku) = (m as usize, n as usize, k as usize);
    let a = DeviceBuffer::upload(&device, &bf16_inputs(mu * ku, 0xDE7)).expect("upload A");
    let b = DeviceBuffer::upload(&device, &bf16_inputs(nu * ku, 0xDE8)).expect("upload B");
    let run = || {
        let c = DeviceBuffer::upload(&device, &vec![SENTINEL; mu * nu]).expect("upload C");
        // SAFETY: as `run_narrow`'s.
        unsafe { narrow_and_wait(&a, &b, &c, (m, n, k)) }.expect("launch");
        c.download().expect("download C")
    };
    let (first, second) = (run(), run());
    let differ = first.iter().zip(&second).filter(|(x, y)| x != y).count();
    eprintln!(
        "gemm_narrow_bf16_abt repeat m={m} n={n} k={k}: bit_mismatches={differ} {}",
        if differ == 0 { "PASS" } else { "FAIL" }
    );
    assert_eq!(differ, 0);
}

#[test]
fn gemm_narrow_bf16_abt_refuses_shapes_it_does_not_take() {
    let device = h100();
    for &(m, n, k) in NARROW_REFUSED {
        let len = |rows: u32, cols: u32| (rows as usize * cols as usize).clamp(1, REFUSED_BUFFER);
        let a = DeviceBuffer::upload(&device, &vec![0; len(m, k)]).expect("upload A");
        let b = DeviceBuffer::upload(&device, &vec![0; len(n, k)]).expect("upload B");
        let c = DeviceBuffer::upload(&device, &vec![SENTINEL; len(m, n) + TAIL]).expect("upload C");
        // SAFETY: the launcher refuses each of these shapes before it touches any pointer.
        let got = unsafe { narrow_and_wait(&a, &b, &c, (m, n, k)) };
        let unchanged = (c.download().expect("download C").iter()).all(|&x| x == SENTINEL);
        let want = Err(KernelError::NarrowShape { m, n, k });
        eprintln!(
            "gemm_narrow_bf16_abt m={m} n={n} k={k}: {got:?} c_unchanged={unchanged} {}",
            if got == want && unchanged {
                "PASS"
            } else {
                "FAIL"
            }
        );
        assert_eq!(got, want, "m={m} n={n} k={k}");
        assert!(unchanged, "m={m} n={n} k={k}: a refused launch wrote C");
    }
}

/// A GEMM launch that waits for itself: [`launch_and_wait`] or [`narrow_and_wait`].
type Launch = unsafe fn(
    &DeviceBuffer,
    &DeviceBuffer,
    &DeviceBuffer,
    (u32, u32, u32),
) -> Result<(), KernelError>;

/// The same 212 rows of `A` placed at row offsets 0, 37, 88 and 300 of an `M`-row operand (the
/// rest of `A` other values): each placement's output rows bit-identical to offset 0's. An
/// element's K-accumulation must not depend on which row of the 128-row tile, which consumer
/// warpgroup, or which tile of the grid its row lands in. `cuda_prefill`'s (e) moves no row; its
/// (f) moves sequence rows 300..512 to bucket rows 0..212.
fn placement_misses(
    device: &CudaDevice,
    (n, k): (u32, u32),
    m: u32,
    run: Launch,
    what: &str,
) -> bool {
    let (mu, nu, ku) = (m as usize, n as usize, k as usize);
    let rows = 212usize;
    let block = bf16_inputs(rows * ku, 0x0F5E_0001 ^ u64::from(n));
    let b = DeviceBuffer::upload(device, &bf16_inputs(nu * ku, 0x0F5E_0002 ^ u64::from(k)))
        .expect("upload B");
    let mut placed: Vec<Vec<u16>> = Vec::new();
    let mut pass = true;
    for (i, off) in [0usize, 37, 88, 300].into_iter().enumerate() {
        let mut a = bf16_inputs(mu * ku, 0x0F5E_0100 ^ i as u64);
        a[off * ku..][..rows * ku].copy_from_slice(&block);
        let a = DeviceBuffer::upload(device, &a).expect("upload A");
        let c = DeviceBuffer::upload(device, &vec![SENTINEL; mu * nu]).expect("upload C");
        // SAFETY: the buffers hold m*k, n*k and m*n bf16, cudaMalloc pointers are 256-byte
        // aligned, and `c` overlaps neither input.
        unsafe { run(&a, &b, &c, (m, n, k)) }.expect("launch");
        let out = c.download().expect("download C");
        placed.push(out[off * nu..][..rows * nu].to_vec());
        let mismatched = mismatched_bits(&placed[i], &placed[0]);
        let worst_row = (0..rows)
            .max_by_key(|&r| {
                mismatched_bits(&placed[i][r * nu..][..nu], &placed[0][r * nu..][..nu])
            })
            .filter(|_| mismatched > 0);
        let ok = mismatched == 0;
        eprintln!(
            "{what} m={m} n={n} k={k}: rows at offset {off} vs offset 0: \
             mismatched_bits={mismatched}/{} worst_row={worst_row:?} {}",
            rows * nu,
            if ok { "PASS" } else { "FAIL" }
        );
        pass &= ok;
    }
    pass
}

#[test]
fn an_output_row_does_not_depend_on_where_its_input_row_sits() {
    let device = h100();
    let mut pass = true;
    // The 0.8b's shapes: the QKV/gate projection, the down projection, the lm_head's K, at the
    // m=512 bucket.
    for (n, k) in [(6144u32, 1024u32), (1024, 3584), (2048, 1024)] {
        pass &= placement_misses(&device, (n, k), 512, launch_and_wait, "gemm_bf16_abt");
    }
    // The narrow GEMM: the GDN a/b projections (n = 16).
    for (n, k) in [(16u32, 1024u32), (64, 1024)] {
        pass &= placement_misses(
            &device,
            (n, k),
            512,
            narrow_and_wait,
            "gemm_narrow_bf16_abt",
        );
    }
    assert!(
        pass,
        "an output row depends on where its input row sits: see the lines above"
    );
}

// ── The GEMV (m = 1) ───────────────────────────────────────────────

/// `(n, k)` for the GEMV: every qwen3.5-0.8b decode shape (the q projection with its gate, k and
/// v, the attention's o and the GDN out projection, the MLP's gate/up and down, the GDN QKV and z
/// projections, the lm_head), every qwen3.5-9b one (the same projections at hidden 4096 and
/// intermediate 12288), then grid tails: one CTA of 8 columns, three warps of one CTA, a partial
/// last CTA with a `k` of 9 vectors (a lane takes 0 or 1), and one column.
const GEMV_SHAPES: &[(u32, u32, &str)] = &[
    (4096, 1024, "q_proj with its gate"),
    (512, 1024, "k_proj / v_proj"),
    (1024, 2048, "o_proj, the GDN out_proj"),
    (3584, 1024, "gate_proj / up_proj"),
    (1024, 3584, "down_proj"),
    (6144, 1024, "the GDN in_proj_qkv"),
    (2048, 1024, "the GDN in_proj_z"),
    (248_320, 1024, "the lm_head"),
    (1024, 4096, "9b: k_proj / v_proj"),
    (4096, 4096, "9b: o_proj, the GDN in_proj_z and out_proj"),
    (8192, 4096, "9b: q_proj with its gate, the GDN in_proj_qkv"),
    (12288, 4096, "9b: gate_proj / up_proj"),
    (4096, 12288, "9b: down_proj"),
    (248_320, 4096, "9b: the lm_head"),
    (8, 8, "one CTA, one k vector"),
    (24, 64, "three warps of a CTA"),
    (1000, 72, "a partial last CTA, 9 k vectors"),
    (1, 8, "one column"),
];

/// Shapes the GEMV launcher refuses: `k` zero or off the 8-wide vector, `n` zero, `n` or `k`
/// above `INT_MAX` (refused before any pointer is touched, so its buffers are small).
const GEMV_REFUSED: &[(u32, u32)] = &[
    (1024, 0),
    (1024, 100),
    (0, 1024),
    (2_147_483_648, 1024),
    (8, 2_147_483_648),
];

/// Launch the GEMV on the default stream, then wait for it.
///
/// # Safety
///
/// As [`gemv_bf16_abt`]'s, for the default stream.
unsafe fn gemv_and_wait(
    a: &DeviceBuffer,
    b: &DeviceBuffer,
    c: &DeviceBuffer,
    (n, k): (u32, u32),
) -> Result<(), KernelError> {
    // SAFETY: the caller's contract.
    let launched =
        unsafe { gemv_bf16_abt(a.ptr(), b.ptr(), c.ptr(), (n, k), std::ptr::null_mut()) };
    c.download().expect("waiting for the launch");
    launched
}

/// One GEMV of `(n, k)`: the output under the GEMM bound, its tail untouched, and a repeat
/// bit-identical. Prints its line; returns whether it passed.
fn run_gemv(device: &CudaDevice, n: u32, k: u32, what: &str) -> bool {
    let (nu, ku) = (n as usize, k as usize);
    let a_bits = bf16_inputs(ku, 0x6E70_0000 ^ u64::from(k));
    let b_bits = bf16_inputs(nu * ku, 0x6E71_0000 ^ (u64::from(n) << 32) ^ u64::from(k));
    let a = DeviceBuffer::upload(device, &a_bits).expect("upload a");
    let b = DeviceBuffer::upload(device, &b_bits).expect("upload B");
    let run = || {
        let c = DeviceBuffer::upload(device, &vec![SENTINEL; nu + TAIL]).expect("upload c");
        // SAFETY: the buffers hold k, n*k and n + TAIL bf16, cudaMalloc pointers are 256-byte
        // aligned and `c` overlaps neither input.
        unsafe { gemv_and_wait(&a, &b, &c, (n, k)) }.map(|()| c.download().expect("download c"))
    };
    let (first, second) = match (run(), run()) {
        (Ok(x), Ok(y)) => (x, y),
        (Err(e), _) | (_, Err(e)) => {
            eprintln!("gemv_bf16_abt n={n} k={k}: launch failed: {e} FAIL");
            return false;
        }
    };
    let tail_intact = first[nu..].iter().all(|&x| x == SENTINEL);
    let cmp = compare(&first[..nu], &a_bits, &b_bits, 1, nu, ku);
    let repeat = mismatched_bits(&first, &second);
    let pass = cmp.pass() && tail_intact && repeat == 0;
    eprintln!(
        "gemv_bf16_abt n={n} k={k}: {} tail_intact={tail_intact} repeat_mismatched_bits={repeat} \
         {} ({what})",
        cmp.line(),
        if pass { "PASS" } else { "FAIL" }
    );
    pass
}

/// Every decode shape and every grid tail within #761's bound of f64, bit-identical on a repeat.
#[test]
fn gemv_bf16_abt_matches_the_cpu_reference() {
    let device = h100();
    let failed: Vec<_> = GEMV_SHAPES
        .iter()
        .filter(|&&(n, k, what)| !run_gemv(&device, n, k, what))
        .map(|&(n, k, _)| (n, k))
        .collect();
    assert!(
        failed.is_empty(),
        "gemv_bf16_abt failed for (n, k) = {failed:?}"
    );
}

/// Each refused shape returns `GemvShape`, and the poisoned output is left as it was.
#[test]
fn gemv_bf16_abt_refuses_shapes_it_does_not_take() {
    let device = h100();
    let mut pass = true;
    for &(n, k) in GEMV_REFUSED {
        let len = |count: usize| count.clamp(1, REFUSED_BUFFER);
        let a = DeviceBuffer::upload(&device, &vec![0; len(k as usize)]).expect("upload a");
        let b = DeviceBuffer::upload(&device, &vec![0; len(n as usize * k as usize)])
            .expect("upload B");
        let c = DeviceBuffer::upload(&device, &vec![SENTINEL; len(n as usize) + TAIL])
            .expect("upload c");
        // SAFETY: the launcher refuses each of these shapes before it touches any pointer.
        let got = unsafe { gemv_and_wait(&a, &b, &c, (n, k)) };
        let unchanged = (c.download().expect("download c").iter()).all(|&x| x == SENTINEL);
        let want = Err(KernelError::GemvShape { n, k });
        let ok = got == want && unchanged;
        eprintln!(
            "gemv_bf16_abt n={n} k={k}: {got:?} c_unchanged={unchanged} {}",
            if ok { "PASS" } else { "FAIL" }
        );
        pass &= ok;
    }
    assert!(
        pass,
        "a GEMV refusal launched or wrote: see the lines above"
    );
}

/// The GEMV against row `r` of the tiled GEMM at `m = 128` over the same vector (the other rows
/// of `A` other values): two different kernels, so no bit requirement, but their difference is
/// within the sum of their two bounds, `2·(2⁻⁷·|ref| + 2⁻¹⁴·S)`. The worst ratio and the largest
/// difference in bf16 ulps are printed.
#[test]
fn the_gemv_is_within_two_bounds_of_the_tiled_gemms_row() {
    let device = h100();
    let mut pass = true;
    for (n, k) in [(4096u32, 1024u32), (1024, 3584), (6144, 1024), (2048, 1024)] {
        let (nu, ku, m) = (n as usize, k as usize, 128usize);
        let b_bits = bf16_inputs(nu * ku, 0x6E72_0000 ^ u64::from(n));
        let b = DeviceBuffer::upload(&device, &b_bits).expect("upload B");
        for r in [0usize, 37, 127] {
            let row = bf16_inputs(ku, 0x6E73_0000 ^ r as u64 ^ u64::from(k));
            let mut a_bits = bf16_inputs(m * ku, 0x6E74_0000 ^ r as u64);
            a_bits[r * ku..][..ku].copy_from_slice(&row);
            let a = DeviceBuffer::upload(&device, &a_bits).expect("upload A");
            let x = DeviceBuffer::upload(&device, &row).expect("upload a");
            let c = DeviceBuffer::upload(&device, &vec![SENTINEL; m * nu]).expect("upload C");
            let g = DeviceBuffer::upload(&device, &vec![SENTINEL; nu]).expect("upload c");
            // SAFETY: the buffers hold m*k, n*k, m*n, k and n bf16; cudaMalloc pointers are
            // 256-byte aligned; neither output overlaps an input; `h100` ran `prepare_all`.
            unsafe { launch_and_wait(&a, &b, &c, (m as u32, n, k)) }.expect("the tiled GEMM");
            // SAFETY: as above.
            unsafe { gemv_and_wait(&x, &b, &g, (n, k)) }.expect("the GEMV");
            let tiled = c.download().expect("download C")[r * nu..][..nu].to_vec();
            let gemv = g.download().expect("download c");
            let refs = reference(&widen(&row), &widen(&b_bits), 1, nu, ku);
            let (mut worst, mut ulps, mut bad) = (0.0f64, 0u16, 0usize);
            for ((&t, &v), &(re, s)) in tiled.iter().zip(&gemv).zip(&refs) {
                let (tf, vf) = (
                    f64::from(half::bf16::from_bits(t).to_f32()),
                    f64::from(half::bf16::from_bits(v).to_f32()),
                );
                let bound = 2.0 * (re.abs() / 128.0 + s / 16384.0);
                let diff = (tf - vf).abs();
                bad += usize::from(!(tf.is_finite() && vf.is_finite() && diff <= bound));
                worst = worst.max(diff / bound.max(f64::MIN_POSITIVE));
                ulps = ulps.max(t.abs_diff(v));
            }
            let ok = bad == 0;
            eprintln!(
                "gemv vs gemm_bf16_abt row {r} of m=128, n={n} k={k}: out_of_bound={bad} \
                 worst_ratio={worst:.4} max_ulps={ulps} {}",
                if ok { "PASS" } else { "FAIL" }
            );
            pass &= ok;
        }
    }
    assert!(
        pass,
        "the GEMV is outside two bounds of the tiled GEMM's row"
    );
}

/// The narrow GEMM at `m = 1` is bit-identical to the same row at `m = 128`: the decode's GDN
/// `a` and `b` projections stay on the narrow GEMM, and its bits for a row do not depend on the
/// rows around it.
#[test]
fn the_narrow_gemm_at_one_row_is_its_row_at_128() {
    let device = h100();
    let mut pass = true;
    for (n, k) in [(16u32, 1024u32), (32, 4096), (64, 1024)] {
        let (nu, ku, m) = (n as usize, k as usize, 128usize);
        let b = DeviceBuffer::upload(&device, &bf16_inputs(nu * ku, 0x6E75_0000 ^ u64::from(n)))
            .expect("upload B");
        let a_bits = bf16_inputs(m * ku, 0x6E76_0000 ^ u64::from(k));
        let a = DeviceBuffer::upload(&device, &a_bits).expect("upload A");
        let c = DeviceBuffer::upload(&device, &vec![SENTINEL; m * nu]).expect("upload C");
        // SAFETY: the buffers hold m*k, n*k and m*n bf16; cudaMalloc pointers are 256-byte
        // aligned and `c` overlaps neither input.
        unsafe { narrow_and_wait(&a, &b, &c, (m as u32, n, k)) }.expect("m = 128");
        let all = c.download().expect("download C");
        for r in [0usize, 37, 127] {
            let x = DeviceBuffer::upload(&device, &a_bits[r * ku..][..ku]).expect("upload a");
            let one = DeviceBuffer::upload(&device, &vec![SENTINEL; nu]).expect("upload c");
            // SAFETY: as above, at one row.
            unsafe { narrow_and_wait(&x, &b, &one, (1, n, k)) }.expect("m = 1");
            let mismatched =
                mismatched_bits(&one.download().expect("download c"), &all[r * nu..][..nu]);
            let ok = mismatched == 0;
            eprintln!(
                "gemm_narrow_bf16_abt m=1 vs row {r} of m=128, n={n} k={k}: \
                 mismatched_bits={mismatched}/{nu} {}",
                if ok { "PASS" } else { "FAIL" }
            );
            pass &= ok;
        }
    }
    assert!(
        pass,
        "the narrow GEMM's one-row output differs from its row at m = 128"
    );
}

/// Recorded, not required (PLAN-tk100 §8, R6): whether the GEMV's row is bit-identical to row 0
/// of the tiled GEMM on the same inputs. Both run WGMMA over 64-wide K steps in the same order,
/// with N tiles of 64 and 256; the printed count says whether that gives the same bits. No
/// assertion: `the_gemv_is_within_two_bounds_of_the_tiled_gemms_row` gates.
#[test]
fn the_gemv_row_against_the_tiled_gemms_row_0_is_recorded() {
    let device = h100();
    for (n, k) in [(4096u32, 1024u32), (1024, 3584), (6144, 1024), (2048, 1024)] {
        let (nu, ku, m) = (n as usize, k as usize, 128usize);
        let b_bits = bf16_inputs(nu * ku, 0x6E77_0000 ^ u64::from(n));
        let row = bf16_inputs(ku, 0x6E78_0000 ^ u64::from(k));
        let mut a_bits = bf16_inputs(m * ku, 0x6E79_0000 ^ u64::from(k));
        a_bits[..ku].copy_from_slice(&row);
        let b = DeviceBuffer::upload(&device, &b_bits).expect("upload B");
        let a = DeviceBuffer::upload(&device, &a_bits).expect("upload A");
        let x = DeviceBuffer::upload(&device, &row).expect("upload a");
        let c = DeviceBuffer::upload(&device, &vec![SENTINEL; m * nu]).expect("upload C");
        let g = DeviceBuffer::upload(&device, &vec![SENTINEL; nu]).expect("upload c");
        // SAFETY: the buffers hold m*k, n*k, m*n, k and n bf16; cudaMalloc pointers are 256-byte
        // aligned; neither output overlaps an input; `h100` ran `prepare_all`.
        unsafe { launch_and_wait(&a, &b, &c, (m as u32, n, k)) }.expect("the tiled GEMM");
        // SAFETY: as above.
        unsafe { gemv_and_wait(&x, &b, &g, (n, k)) }.expect("the GEMV");
        let tiled = c.download().expect("download C")[..nu].to_vec();
        let gemv = g.download().expect("download c");
        eprintln!(
            "gemv vs gemm_bf16_abt row 0 of m=128, n={n} k={k}: mismatched_bits={}/{nu} \
             (recorded, not asserted)",
            mismatched_bits(&gemv, &tiled)
        );
    }
}

/// The narrow GEMM's time per launch at the GDN `a`/`b` projections' shapes, printed and not
/// asserted: it records the choice of two consumer warpgroups (`kernels/gemm_narrow_bf16.cu`). The
/// median of 20 launches after 3 warm-ups, each timed from the host to its completion.
#[test]
fn the_narrow_gemm_time_is_recorded() {
    let device = h100();
    for (m, n, k) in [
        (1u32, 16u32, 1024u32),
        (512, 16, 1024),
        (4096, 16, 1024),
        (1, 32, 4096),
        (512, 32, 4096),
        (4096, 32, 4096),
    ] {
        let (mu, nu, ku) = (m as usize, n as usize, k as usize);
        let a = DeviceBuffer::upload(&device, &bf16_inputs(mu * ku, 0x7151)).expect("upload A");
        let b = DeviceBuffer::upload(&device, &bf16_inputs(nu * ku, 0x7152)).expect("upload B");
        let c = DeviceBuffer::upload(&device, &vec![SENTINEL; mu * nu]).expect("upload C");
        let mut times = Vec::new();
        for i in 0..23 {
            let start = std::time::Instant::now();
            // SAFETY: as `run_narrow`'s.
            unsafe { narrow_and_wait(&a, &b, &c, (m, n, k)) }.expect("launch");
            if i >= 3 {
                times.push(start.elapsed().as_secs_f64() * 1e6);
            }
        }
        times.sort_by(f64::total_cmp);
        eprintln!(
            "gemm_narrow_bf16_abt time m={m} n={n} k={k}: median={:.1}us min={:.1}us over {} \
             launches, host-timed including the download (recorded, not asserted)",
            times[times.len() / 2],
            times[0],
            times.len()
        );
    }
}

// ── #857 M7: the narrow GEMM at n = 1 (a shared expert's gate) ─────

/// The narrow GEMM's one-column path (`n = 1`, `kernels/gemm_narrow_bf16.cu`'s warp-vector kernel)
/// at qwen3-5-moe's `k` 2048 and `m` ∈ {1, 7, 512}: every row within the GEMM bound, the tail past
/// `c[m]` intact, and a repeat bit-identical; then its one tightening, `k` a multiple of 256,
/// refused at `k` 2040 (a multiple of 8) with `c` unchanged.
#[test]
fn gemm_narrow_bf16_abt_at_one_column_matches_the_cpu_reference() {
    let device = h100();
    let (n, k) = (1u32, 2048u32);
    let mut pass = true;
    for m in [1u32, 7, 512] {
        let (mu, ku) = (m as usize, k as usize);
        let a_bits = bf16_inputs(mu * ku, 0xA1A1_0000 ^ u64::from(m));
        let b_bits = bf16_inputs(ku, 0xB1B1_0000);
        let a = DeviceBuffer::upload(&device, &a_bits).expect("upload A");
        let b = DeviceBuffer::upload(&device, &b_bits).expect("upload B");
        let run = || {
            let c = DeviceBuffer::upload(&device, &vec![SENTINEL; mu + TAIL]).expect("upload C");
            // SAFETY: the buffers hold m*k, k and m + TAIL bf16, cudaMalloc pointers are 256-byte
            // aligned and `c` overlaps neither input.
            unsafe { narrow_and_wait(&a, &b, &c, (m, n, k)) }.expect("launch");
            c.download().expect("download C")
        };
        let (first, second) = (run(), run());
        let tail_intact = first[mu..].iter().all(|&x| x == SENTINEL);
        let cmp = compare(&first[..mu], &a_bits, &b_bits, mu, 1, ku);
        let repeat = mismatched_bits(&first, &second);
        let ok = cmp.pass() && tail_intact && repeat == 0;
        eprintln!(
            "gemm_narrow_bf16_abt m={m} n=1 k={k}: {} tail_intact={tail_intact} \
             repeat_mismatched_bits={repeat} {}",
            cmp.line(),
            if ok { "PASS" } else { "FAIL" }
        );
        pass &= ok;
    }
    let (m, k) = (7u32, 2040u32);
    let a = DeviceBuffer::upload(&device, &vec![0; m as usize * k as usize]).expect("upload A");
    let b = DeviceBuffer::upload(&device, &vec![0; k as usize]).expect("upload B");
    let c = DeviceBuffer::upload(&device, &vec![SENTINEL; m as usize + TAIL]).expect("upload C");
    // SAFETY: the launcher refuses this shape before it touches any pointer.
    let got = unsafe { narrow_and_wait(&a, &b, &c, (m, n, k)) };
    let unchanged = (c.download().expect("download C").iter()).all(|&x| x == SENTINEL);
    let want = Err(KernelError::NarrowShape { m, n, k });
    let ok = got == want && unchanged;
    eprintln!(
        "gemm_narrow_bf16_abt m={m} n=1 k={k}: {got:?} c_unchanged={unchanged} {}",
        if ok { "PASS" } else { "FAIL" }
    );
    pass &= ok;
    assert!(
        pass,
        "the narrow GEMM's one-column path failed: see the lines above"
    );
}

// ── The 64-column GEMM (Gemma 4, #924) ─────────────────────────────

/// `(M, N, K)` of Gemma 4 26B-A4B's 64-column GEMMs: the router's `[m, 128]` logits and the dense
/// gate and up's 2112 columns, at m = 512 and 4096 (k = 2816, its hidden), and the smallest tile.
const COL64_SHAPES: &[(u32, u32, u32, &str)] = &[
    (512, 128, 2816, "Gemma's router logits at m=512"),
    (512, 2112, 2816, "Gemma's dense gate/up at m=512"),
    (4096, 2112, 2816, "Gemma's dense gate/up at m=4096"),
    (4096, 128, 2816, "Gemma's router logits at m=4096"),
    (128, 64, 64, "one output tile, one K step"),
];

/// Shapes the 64-column launcher must refuse, launching nothing: `n` 96 and 200 (off 64), `k` 100,
/// `m` 100.
const COL64_REFUSED: &[(u32, u32, u32)] =
    &[(128, 96, 64), (128, 200, 64), (128, 64, 100), (100, 64, 64)];

/// Launch the 64-column GEMM on the default stream, then wait for it.
///
/// # Safety
///
/// As [`gemm_col64_bf16_abt`]'s, for the default stream.
unsafe fn col64_and_wait(
    a: &DeviceBuffer,
    b: &DeviceBuffer,
    c: &DeviceBuffer,
    (m, n, k): (u32, u32, u32),
) -> Result<(), KernelError> {
    // SAFETY: the caller's contract.
    let launched =
        unsafe { gemm_col64_bf16_abt(a.ptr(), b.ptr(), c.ptr(), (m, n, k), std::ptr::null_mut()) };
    c.download().expect("waiting for the launch");
    launched
}

/// A GEMM launch on the default stream that waits for it: [`launch_and_wait`] or
/// [`col64_and_wait`].
type LaunchAndWait = unsafe fn(
    &DeviceBuffer,
    &DeviceBuffer,
    &DeviceBuffer,
    (u32, u32, u32),
) -> Result<(), KernelError>;

/// `C` of one launch of `gemm` on `a` and `b` (`(m, n, k)`), from a sentinel, with [`TAIL`] past
/// its end.
fn run_c(
    device: &CudaDevice,
    (a, b): (&DeviceBuffer, &DeviceBuffer),
    (m, n, k): (u32, u32, u32),
    gemm: LaunchAndWait,
) -> Vec<u16> {
    let len = m as usize * n as usize;
    let c = DeviceBuffer::upload(device, &vec![SENTINEL; len + TAIL]).expect("upload C");
    // SAFETY: the buffers hold m*k, n*k and m*n + TAIL bf16, cudaMalloc pointers are 256-byte
    // aligned, `c` overlaps neither input, and `CudaDevice::open` ran `prepare_all`.
    unsafe { gemm(a, b, &c, (m, n, k)) }.expect("launch");
    c.download().expect("download C")
}

/// Every Gemma shape against the CPU reference within the #761 GEMM bound (`out_of_bound=0`), the
/// sentinel past `c`'s end intact.
#[test]
fn gemm_col64_bf16_matches_the_cpu_reference() {
    let device = h100();
    let mut pass = true;
    for &(m, n, k, what) in COL64_SHAPES {
        let (mu, nu, ku) = (m as usize, n as usize, k as usize);
        let a_bits = bf16_inputs(mu * ku, 0xC064_0000 ^ (u64::from(m) << 32) ^ u64::from(k));
        let b_bits = bf16_inputs(nu * ku, 0xC0B0_0000 ^ (u64::from(n) << 32) ^ u64::from(k));
        let a = DeviceBuffer::upload(&device, &a_bits).expect("upload A");
        let b = DeviceBuffer::upload(&device, &b_bits).expect("upload B");
        let c_bits = run_c(&device, (&a, &b), (m, n, k), col64_and_wait);
        let tail_intact = c_bits[mu * nu..].iter().all(|&x| x == SENTINEL);
        let cmp = compare(&c_bits[..mu * nu], &a_bits, &b_bits, mu, nu, ku);
        let ok = cmp.pass() && tail_intact;
        eprintln!(
            "gemm_col64_bf16_abt m={m} n={n} k={k}: {} tail_intact={tail_intact} {} ({what})",
            cmp.line(),
            if ok { "PASS" } else { "FAIL" }
        );
        pass &= ok;
    }
    assert!(pass, "the 64-column GEMM failed: see the lines above");
}

/// On an `n` that is also a multiple of 256, the 64-column GEMM is `gemm_bf16_abt` bit for bit: the
/// same WGMMA K order, each output element one 64-wide WGMMA column chain (#924's G1 STOP rule).
#[test]
fn gemm_col64_bf16_is_gemm_bf16_on_the_256_grid() {
    let device = h100();
    let mut pass = true;
    for (m, n, k) in [(512u32, 256u32, 2816u32), (4096, 1024, 2816)] {
        let (mu, nu, ku) = (m as usize, n as usize, k as usize);
        let a_bits = bf16_inputs(mu * ku, 0xC1D0_0000 ^ u64::from(m));
        let b_bits = bf16_inputs(nu * ku, 0xC1D1_0000 ^ u64::from(n));
        let a = DeviceBuffer::upload(&device, &a_bits).expect("upload A");
        let b = DeviceBuffer::upload(&device, &b_bits).expect("upload B");
        let col64 = run_c(&device, (&a, &b), (m, n, k), col64_and_wait);
        let tiled = run_c(&device, (&a, &b), (m, n, k), launch_and_wait);
        let bad = mismatched_bits(&col64[..mu * nu], &tiled[..mu * nu]);
        let rows: Vec<usize> = (0..mu)
            .filter(|&r| col64[r * nu..(r + 1) * nu] != tiled[r * nu..(r + 1) * nu])
            .take(8)
            .collect();
        let ok = bad == 0;
        eprintln!(
            "gemm_col64_bf16_abt vs gemm_bf16_abt m={m} n={n} k={k}: mismatched_bits={bad} \
             first_differing_rows={rows:?} {}",
            if ok { "PASS" } else { "FAIL" }
        );
        pass &= ok;
    }
    assert!(
        pass,
        "the 64-column GEMM is not gemm_bf16_abt on the 256 grid: see the lines above"
    );
}

/// A shape off the (128, 64, 64) grid returns `Shape`, and `C` is left as it was.
#[test]
fn gemm_col64_bf16_refuses_off_tile() {
    let device = h100();
    let mut pass = true;
    for &(m, n, k) in COL64_REFUSED {
        let len = |rows: u32, cols: u32| (rows as usize * cols as usize).min(REFUSED_BUFFER);
        let a = DeviceBuffer::upload(&device, &vec![0; len(m, k)]).expect("upload A");
        let b = DeviceBuffer::upload(&device, &vec![0; len(n, k)]).expect("upload B");
        let c = DeviceBuffer::upload(&device, &vec![SENTINEL; len(m, n) + TAIL]).expect("upload C");
        // SAFETY: the launcher refuses each of these shapes before it touches any pointer.
        let got = unsafe { col64_and_wait(&a, &b, &c, (m, n, k)) };
        let unchanged = (c.download().expect("download C").iter()).all(|&x| x == SENTINEL);
        let ok = got == Err(KernelError::Shape { m, n, k }) && unchanged;
        eprintln!(
            "gemm_col64_bf16_abt m={m} n={n} k={k}: {got:?} c_unchanged={unchanged} {}",
            if ok { "PASS" } else { "FAIL" }
        );
        pass &= ok;
    }
    assert!(
        pass,
        "the 64-column GEMM took a shape off its tile: see the lines above"
    );
}

/// Two launches on the same inputs give the same bits, at the router's and the dense gate's
/// shapes.
#[test]
fn gemm_col64_bf16_repeats_bit_identically() {
    let device = h100();
    let mut pass = true;
    for (m, n, k) in [(512u32, 128u32, 2816u32), (512, 2112, 2816)] {
        let (mu, nu, ku) = (m as usize, n as usize, k as usize);
        let a_bits = bf16_inputs(mu * ku, 0xC2E0_0000 ^ u64::from(n));
        let b_bits = bf16_inputs(nu * ku, 0xC2E1_0000 ^ u64::from(n));
        let a = DeviceBuffer::upload(&device, &a_bits).expect("upload A");
        let b = DeviceBuffer::upload(&device, &b_bits).expect("upload B");
        let first = run_c(&device, (&a, &b), (m, n, k), col64_and_wait);
        let second = run_c(&device, (&a, &b), (m, n, k), col64_and_wait);
        let bad = mismatched_bits(&first, &second);
        let ok = bad == 0;
        eprintln!(
            "gemm_col64_bf16_abt repeat m={m} n={n} k={k}: mismatched_bits={bad} {}",
            if ok { "PASS" } else { "FAIL" }
        );
        pass &= ok;
    }
    assert!(
        pass,
        "the 64-column GEMM is not deterministic: see the lines above"
    );
}

// ── #1031: the split-K decode GEMV ─────────────────────────────────

use scratchy_target_cuda::kernels::gemv_split_bf16_abt;
use scratchy_target_cuda::step::{gemv_scratch_bytes, gemv_splits};

/// Every real decode shape (#1031's speed table, the qwen and Gemma projections and lm_heads),
/// Gemma's off-chunk `k` 704 and 2112 (its shared expert, 2112 = 8¼ chunks) and the grid and `k`
/// tails of `GEMV_SHAPES`.
const SPLIT_SHAPES: &[(u32, u32, &str)] = &[
    (4096, 4096, "9b: o_proj"),
    (8192, 4096, "9b: q_proj with its gate"),
    (12288, 4096, "9b: gate_proj / up_proj"),
    (4096, 12288, "9b: down_proj"),
    (1024, 4096, "9b: k_proj / v_proj"),
    (248_320, 4096, "9b: the lm_head"),
    (248_320, 1024, "0.8b: the lm_head"),
    (4096, 1024, "0.8b: q_proj"),
    (3584, 1024, "0.8b: gate_proj"),
    (6144, 1024, "0.8b: in_proj_qkv"),
    (1024, 3584, "0.8b: down_proj"),
    (262_144, 2816, "gemma: the lm_head"),
    (2816, 704, "gemma: k 704"),
    (2816, 2112, "gemma: the shared expert's down, k 2112"),
    (8, 8, "one CTA, one k vector"),
    (1000, 72, "a partial last block, 9 k vectors"),
    (1, 8, "one column"),
];

/// One split GEMV of `(n, k)` at `splits` over `a` and `b` in fresh buffers, the scratch `partials`
/// (or null), `c` NaN-poisoned with a tail; returns `c`'s `n + TAIL` elements.
fn split_run(
    device: &CudaDevice,
    (a, b): (&DeviceBuffer, &DeviceBuffer),
    partials: Option<&DeviceBuffer>,
    (n, k, splits): (u32, u32, u32),
) -> Result<Vec<u16>, KernelError> {
    let c = DeviceBuffer::upload(device, &vec![SENTINEL; n as usize + TAIL]).expect("upload c");
    let p = partials.map_or(std::ptr::null_mut(), DeviceBuffer::ptr);
    // SAFETY: `a`, `b` and `c` hold k, n·k and n + TAIL bf16, `partials` at least the split's
    // scratch bytes, all cudaMalloc'd and disjoint; the download waits for the default stream.
    let launched = unsafe {
        gemv_split_bf16_abt(
            (a.ptr(), b.ptr(), c.ptr()),
            p,
            (n, k, splits),
            std::ptr::null_mut(),
        )
    };
    launched.map(|()| c.download().expect("download c"))
}

/// #1031's correctness rule at every split shape, at the split `gemv_splits` picks and at one
/// split: within the GEMV bound of f64, the tail untouched, bit-identical on a repeat, and the
/// same bits with the operands at another offset (a 256-byte-shifted `a` and `B`).
#[test]
fn the_split_gemv_is_within_the_bound_repeatable_and_offset_independent() {
    let device = h100();
    let mut pass = true;
    for &(n, k, what) in SPLIT_SHAPES {
        let (nu, ku) = (n as usize, k as usize);
        let a_bits = bf16_inputs(ku, 0x5917_0000 ^ u64::from(k));
        let b_bits = bf16_inputs(nu * ku, 0x5918_0000 ^ (u64::from(n) << 32) ^ u64::from(k));
        let a = DeviceBuffer::upload(&device, &a_bits).expect("upload a");
        let b = DeviceBuffer::upload(&device, &b_bits).expect("upload B");
        // The same data 128 elements (256 bytes) into a larger buffer.
        let shift = |x: &[u16]| [vec![0u16; 128], x.to_vec()].concat();
        let a_off = DeviceBuffer::upload(&device, &shift(&a_bits)).expect("upload a");
        let b_off = DeviceBuffer::upload(&device, &shift(&b_bits)).expect("upload B");
        for splits in [gemv_splits(n, k, 1), 1] {
            let bytes = gemv_scratch_bytes(n, 1, splits).expect("scratch bytes") as usize;
            let scratch = (bytes > 0)
                .then(|| DeviceBuffer::upload(&device, &vec![SENTINEL; bytes / 2 + 128]))
                .map(|r| r.expect("upload scratch"));
            let run = |a: &DeviceBuffer, b: &DeviceBuffer| {
                split_run(&device, (a, b), scratch.as_ref(), (n, k, splits))
            };
            let (first, second) = match (run(&a, &b), run(&a, &b)) {
                (Ok(x), Ok(y)) => (x, y),
                (Err(e), _) | (_, Err(e)) => {
                    eprintln!("gemv_split n={n} k={k} splits={splits}: launch failed: {e} FAIL");
                    pass = false;
                    continue;
                }
            };
            // The shifted copies: the launcher's pointers 256 bytes in.
            let moved = {
                let c = DeviceBuffer::upload(&device, &vec![SENTINEL; nu + TAIL]).expect("upload");
                let p = scratch
                    .as_ref()
                    .map_or(std::ptr::null_mut(), DeviceBuffer::ptr);
                let (ap, bp) = (
                    a_off.ptr().wrapping_byte_add(256),
                    b_off.ptr().wrapping_byte_add(256),
                );
                // SAFETY: as `split_run`'s, the operands 256 bytes into their larger buffers.
                unsafe {
                    gemv_split_bf16_abt((ap, bp, c.ptr()), p, (n, k, splits), std::ptr::null_mut())
                }
                .expect("the moved launch");
                c.download().expect("download c")
            };
            let tail = first[nu..].iter().all(|&x| x == SENTINEL);
            let cmp = compare(&first[..nu], &a_bits, &b_bits, 1, nu, ku);
            let (repeat, offset) = (
                mismatched_bits(&first, &second),
                mismatched_bits(&first, &moved),
            );
            let (_, _, g, r, bd) = cmp.worst;
            let ratio = (g - r).abs() / bd.max(f64::MIN_POSITIVE);
            let ok = cmp.pass() && tail && repeat == 0 && offset == 0;
            eprintln!(
                "gemv_split n={n} k={k} splits={splits}: {} tail_intact={tail} \
                 repeat_mismatched_bits={repeat} offset_mismatched_bits={offset}{} {} ({what})",
                cmp.line(),
                if ratio > 0.99 {
                    " RATIO ABOVE 0.99"
                } else {
                    ""
                },
                if ok { "PASS" } else { "FAIL" }
            );
            pass &= ok;
        }
    }
    assert!(pass, "a split GEMV shape failed: see the lines above");
}

/// The split launcher refuses, before touching any pointer: every `GEMV_REFUSED` shape, a split
/// of 0, above 16, above the row's chunks, one leaving a run empty, and a null scratch above one
/// split; `c` stays poisoned.
#[test]
fn the_split_gemv_refuses_bad_splits_and_a_null_scratch() {
    let device = h100();
    let a = DeviceBuffer::upload(&device, &vec![0; REFUSED_BUFFER]).expect("upload a");
    let b = DeviceBuffer::upload(&device, &vec![0; REFUSED_BUFFER]).expect("upload B");
    let s = DeviceBuffer::upload(&device, &vec![0; REFUSED_BUFFER]).expect("upload scratch");
    let mut cases: Vec<(u32, u32, u32, bool)> =
        GEMV_REFUSED.iter().map(|&(n, k)| (n, k, 1, true)).collect();
    cases.extend([
        (64, 1024, 0, true),
        (64, 8192, 17, true),
        (64, 1024, 5, true),
        (64, 2304, 7, true),
        (64, 1024, 2, false),
    ]);
    let mut pass = true;
    for (n, k, splits, with_scratch) in cases {
        let scratch = with_scratch.then_some(&s);
        let got = split_run(&device, (&a, &b), scratch, (n, k, splits));
        let ok = got.as_ref().err() == Some(&KernelError::GemvSplit { n, k, splits });
        eprintln!(
            "gemv_split n={n} k={k} splits={splits} scratch={with_scratch}: {:?} {}",
            got.err(),
            if ok { "PASS" } else { "FAIL" }
        );
        pass &= ok;
    }
    assert!(pass, "the split launcher took a shape it must refuse");
}

/// #1031's bandwidth table, recorded: per real shape, the weight bytes (`n · k · 2` plus `a`) over
/// the median of 50 launches after 10 warm-ups, each timed from the host to its completion (an
/// upper bound on the kernel time), at `gemv_splits`' split, beside the one-split time. The gates
/// (2.4 TB/s on the 9B's projections, 2.6 on the lm_heads, 1.8 and 1.5 on the small shapes) are
/// checked on the GKE run's CUDA-event numbers, not here.
#[test]
fn the_split_gemv_bandwidth_is_recorded() {
    let device = h100();
    for &(n, k, what) in &SPLIT_SHAPES[..14] {
        let (nu, ku) = (n as usize, k as usize);
        let a = DeviceBuffer::upload(&device, &bf16_inputs(ku, 0x7A)).expect("upload a");
        let b = DeviceBuffer::upload(&device, &bf16_inputs(nu * ku, 0x7B)).expect("upload B");
        let c = DeviceBuffer::upload(&device, &vec![0; nu]).expect("upload c");
        let bytes = (nu * ku + ku) as f64 * 2.0;
        let mut line = Vec::new();
        for splits in [gemv_splits(n, k, 1), 1] {
            let sb = gemv_scratch_bytes(n, 1, splits).expect("scratch") as usize;
            let s = DeviceBuffer::upload(&device, &vec![0; sb / 2 + 128]).expect("upload scratch");
            let mut times = Vec::new();
            for i in 0..60 {
                let start = std::time::Instant::now();
                // SAFETY: as `split_run`'s.
                unsafe {
                    gemv_split_bf16_abt(
                        (a.ptr(), b.ptr(), c.ptr()),
                        s.ptr(),
                        (n, k, splits),
                        std::ptr::null_mut(),
                    )
                }
                .expect("launch");
                c.download().expect("wait");
                if i >= 10 {
                    times.push(start.elapsed().as_secs_f64());
                }
            }
            times.sort_by(f64::total_cmp);
            let med = times[times.len() / 2];
            line.push(format!(
                "splits={splits} median={:.1}us {:.2}TB/s",
                med * 1e6,
                bytes / med / 1e12
            ));
        }
        eprintln!(
            "gemv_split bandwidth n={n} k={k}: {} (host-timed with the download; recorded, not \
             asserted) ({what})",
            line.join(", ")
        );
    }
}

/// Which chain a gate/up case folds: the plain GEMM's, the 64-column GEMM's, or the GEMV's.
#[derive(Clone, Copy, Debug, PartialEq)]
enum GateUpChain {
    Gemm,
    Col64,
    Gemv,
}

/// The chain #1049's gate/up fold replaces, on the default stream: the gate's and the up's GEMMs
/// (or GEMVs, at the split each takes on its own), then the activation `act` over their outputs.
///
/// # Safety
///
/// As the launchers'; `a`, `g`, `u` hold `m·k`, `n·k` and `n·k` bf16, `gc`, `uc` and `out` `m·n`.
unsafe fn gate_up_chain(
    chain: GateUpChain,
    (a, g, u): (&DeviceBuffer, &DeviceBuffer, &DeviceBuffer),
    (gc, uc, out, partials): (&DeviceBuffer, &DeviceBuffer, &DeviceBuffer, &DeviceBuffer),
    (m, n, k): (u32, u32, u32),
    act: GatedOp,
) {
    let null = std::ptr::null_mut();
    // SAFETY: the caller's contract; the default stream orders the launches.
    unsafe {
        for (b, c) in [(g, gc), (u, uc)] {
            match chain {
                GateUpChain::Gemm => gemm_bf16_abt(a.ptr(), b.ptr(), c.ptr(), m, n, k, null),
                GateUpChain::Col64 => {
                    gemm_col64_bf16_abt(a.ptr(), b.ptr(), c.ptr(), (m, n, k), null)
                }
                GateUpChain::Gemv => match gemv_splits(n, k, 1) {
                    1 => gemv_bf16_abt(a.ptr(), b.ptr(), c.ptr(), (n, k), null),
                    s => gemv_split_bf16_abt(
                        (a.ptr(), b.ptr(), c.ptr()),
                        partials.ptr(),
                        (n, k, s),
                        null,
                    ),
                },
            }
            .expect("the chain's GEMM");
        }
        match act {
            GatedOp::Silu => silu_mul_bf16(gc.ptr(), uc.ptr(), out.ptr(), (m, n), null),
            GatedOp::Gelu => gelu_mul_bf16(gc.ptr(), uc.ptr(), out.ptr(), (m, n), null),
        }
        .expect("the chain's activation");
    }
}

/// #1049's gate/up GEMV and GEMM are their chains bit for bit: the gate's and the up's GEMMs (or
/// GEMVs) then the activation, for SwiGLU and GeGLU, at the decode shapes (the 0.8B's and 9B's
/// MLP, which split, and Gemma's 2112-column one), a prefill shape of each GEMM, and the output's
/// tail intact.
#[test]
fn the_gate_up_fold_is_its_chain_bit_for_bit() {
    let device = h100();
    let mut pass = true;
    for (chain, (m, n, k)) in [
        (GateUpChain::Gemv, (1, 3584, 1024)),
        (GateUpChain::Gemv, (1, 12288, 4096)),
        (GateUpChain::Gemv, (1, 2112, 2816)),
        (GateUpChain::Gemm, (512, 3584, 1024)),
        (GateUpChain::Gemm, (256, 1024, 512)),
        (GateUpChain::Col64, (512, 2112, 2816)),
    ] {
        for act in [GatedOp::Silu, GatedOp::Gelu] {
            let (mu, nu, ku) = (m as usize, n as usize, k as usize);
            let seed = (u64::from(n) << 32) ^ u64::from(k) ^ u64::from(m) << 48;
            let up_buf = |v: &[u16]| DeviceBuffer::upload(&device, v).expect("upload");
            let a = up_buf(&bf16_inputs(mu * ku, 0x6A7E_0000 ^ seed));
            let g = up_buf(&bf16_inputs(nu * ku, 0x6A7E_1000 ^ seed));
            let u = up_buf(&bf16_inputs(nu * ku, 0x6A7E_2000 ^ seed));
            let out_buf = || up_buf(&vec![SENTINEL; mu * nu + TAIL]);
            let (gc, uc, chain_out, fused) = (out_buf(), out_buf(), out_buf(), out_buf());
            // Room for both projections' partials at the GEMVs' largest split (16 × n_pad).
            let partials = up_buf(&vec![0u16; 2 * 2 * 16 * (nu + 32)]);
            // SAFETY: cudaMalloc pointers are 256-byte aligned; every buffer holds what its
            // launcher reads or writes; the outputs are their own allocations; null is the
            // default stream, which the downloads below wait for.
            unsafe {
                gate_up_chain(
                    chain,
                    (&a, &g, &u),
                    (&gc, &uc, &chain_out, &partials),
                    (m, n, k),
                    act,
                );
                match chain {
                    GateUpChain::Gemv => gate_up_gemv_bf16(
                        (a.ptr(), g.ptr(), u.ptr()),
                        (fused.ptr(), partials.ptr()),
                        (n, k, gemv_splits(n, k, 1)),
                        act.code(),
                        std::ptr::null_mut(),
                    ),
                    GateUpChain::Gemm | GateUpChain::Col64 => gate_up_gemm_bf16(
                        (a.ptr(), g.ptr(), u.ptr()),
                        fused.ptr(),
                        (m, n, k),
                        (act.code(), chain == GateUpChain::Col64),
                        std::ptr::null_mut(),
                    ),
                }
            }
            .expect("the fused launch");
            let (want, got) = (
                chain_out.download().expect("download"),
                fused.download().expect("download"),
            );
            let tail = got[mu * nu..].iter().all(|&x| x == SENTINEL);
            let mismatched = mismatched_bits(&got[..mu * nu], &want[..mu * nu]);
            let ok = mismatched == 0 && tail;
            eprintln!(
                "gate_up {chain:?} {act:?} m={m} n={n} k={k}: mismatched_bits={mismatched} \
                 tail_intact={tail} {}",
                if ok { "PASS" } else { "FAIL" }
            );
            pass &= ok;
        }
    }
    assert!(pass, "a gate/up fold differs from its chain");
}

// ── FP8 S8 W4a: the one-row bf16 GEMV's fused ends ─────────────────

use scratchy_target_cuda::kernels::{
    Ends, add_bf16, gate_up_gemv_bf16_ends, gemv_bf16_ends, gemv_group_bf16_abt,
    gemv_group_bf16_ends, rms_norm_bf16,
};

/// The norms' eps.
const W4_EPS: f32 = 1e-6;

fn w4_scratch(device: &CudaDevice, bytes: u64) -> DeviceBuffer {
    DeviceBuffer::upload_bytes(device, &vec![0u8; (bytes as usize).max(256)]).expect("scratch")
}

/// `n` rows of `rows · n + TAIL` NaN, `launch` on it, then the rows and whether the tail held.
fn w4_out(
    device: &CudaDevice,
    len: usize,
    launch: impl FnOnce(*mut std::ffi::c_void) -> Result<(), KernelError>,
) -> (Vec<u16>, bool) {
    let c = DeviceBuffer::upload(device, &vec![SENTINEL; len + TAIL]).expect("c");
    launch(c.ptr()).expect("launch");
    let out = c.download().expect("download");
    (
        out[..len].to_vec(),
        out[len..].iter().all(|&x| x == SENTINEL),
    )
}

/// Random rows, the all-zero row and the ±65,504 row.
fn w4_rows(k: usize, seed: u64) -> Vec<(&'static str, Vec<u16>)> {
    let big: Vec<u16> = (0..k)
        .map(|c| half::bf16::from_f32(if c % 2 == 0 { 65504.0 } else { -65504.0 }).to_bits())
        .collect();
    vec![
        ("rand0", bf16_inputs(k, seed)),
        ("rand1", bf16_inputs(k, seed ^ 0x11)),
        ("zeros", vec![0; k]),
        ("max", big),
    ]
}

/// `rms_norm_bf16` of the one row `x` at offset `offset`.
fn w4_norm(
    device: &CudaDevice,
    x: &DeviceBuffer,
    g: &DeviceBuffer,
    k: u32,
    offset: f32,
) -> DeviceBuffer {
    let y = DeviceBuffer::upload(device, &vec![SENTINEL; k as usize]).expect("y");
    // SAFETY: `x`, `g`, `y` hold k bf16, their own allocations.
    unsafe {
        rms_norm_bf16(
            x.ptr().cast_const(),
            g.ptr().cast_const(),
            y.ptr(),
            (1, k),
            W4_EPS,
            offset,
            std::ptr::null_mut(),
        )
    }
    .expect("rms_norm_bf16");
    y
}

/// The bf16 GEMV at `n × k` from `a`, at its own split, with or without `ends`.
fn w4_gemv(
    device: &CudaDevice,
    a: &DeviceBuffer,
    b: &DeviceBuffer,
    (n, k): (u32, u32),
    ends: Option<Ends>,
) -> (Vec<u16>, bool) {
    let splits = gemv_splits(n, k, 1);
    let p = w4_scratch(device, gemv_scratch_bytes(n, 1, splits).expect("bytes"));
    w4_out(device, n as usize, |c| {
        // SAFETY: `a` k bf16, `b` n·k, `c` n + TAIL, `p` the partials, each its own allocation.
        unsafe {
            match ends {
                Some(e) => gemv_bf16_ends(
                    (a.ptr().cast_const(), b.ptr().cast_const(), c),
                    p.ptr(),
                    (n, k, splits),
                    e,
                    std::ptr::null_mut(),
                ),
                None => gemv_split_bf16_abt(
                    (a.ptr().cast_const(), b.ptr().cast_const(), c),
                    p.ptr(),
                    (n, k, splits),
                    std::ptr::null_mut(),
                ),
            }
        }
    })
}

/// The normed bf16 GEMV, its group and the gate/up GEMV are the one-row norm then today's kernel,
/// bit for bit: the 35B's GDN group (2,048 → 8,192 + 4,096), its router (2,048 → 256), lm_head
/// (2,048 → 248,320 at its split), the Gemma router with the router scale (2,816 → 128, offset 0),
/// at gain offset 0 and 1, on random, all-zero and ±65,504 rows. `mismatched_bits=0`.
#[test]
fn the_normed_bf16_gemv_and_router_are_their_chains_bit_for_bit() {
    let device = h100();
    let mut pass = true;
    let dense: [(u32, u32, &str); 4] = [
        (256, 2048, "35B router"),
        (248320, 2048, "35B lm_head"),
        (128, 2816, "Gemma router"),
        (512, 2048, "35B shared gate/up"),
    ];
    for offset in [0.0f32, 1.0] {
        for (n, k, what) in dense {
            let ku = k as usize;
            let g = bf16_inputs(ku, 0x4D00 ^ u64::from(k));
            let gb = DeviceBuffer::upload(&device, &g).expect("gain");
            let b = DeviceBuffer::upload(
                &device,
                &bf16_inputs(n as usize * ku, 0x4D10 ^ u64::from(n)),
            )
            .expect("b");
            let ends = Ends {
                gain: gb.ptr().cast_const(),
                eps: W4_EPS,
                gain_offset: offset,
                residual: std::ptr::null(),
            };
            for (row, x) in w4_rows(ku, 0x4D20 ^ u64::from(k)) {
                let xb = DeviceBuffer::upload(&device, &x).expect("x");
                let y = w4_norm(&device, &xb, &gb, k, offset);
                let fused = w4_gemv(&device, &xb, &b, (n, k), Some(ends));
                let chain = w4_gemv(&device, &y, &b, (n, k), None);
                let bad = mismatched_bits(&fused.0, &chain.0);
                let ok = bad == 0 && fused.1;
                eprintln!(
                    "W4 normed gemv_bf16 {what} n={n} k={k} offset={offset} row={row}: \
                     mismatched_bits={bad} tail={} {}",
                    fused.1,
                    if ok { "PASS" } else { "FAIL" }
                );
                pass &= ok;
                if n == 512 {
                    // The gate/up GEMV over the same row (the shared expert's), SwiGLU.
                    let up = DeviceBuffer::upload(&device, &bf16_inputs(n as usize * ku, 0x4D30))
                        .expect("up");
                    let splits = gemv_splits(n, k, 1);
                    let gu = |a: &DeviceBuffer, e: Option<Ends>| {
                        let p =
                            w4_scratch(&device, 2 * gemv_scratch_bytes(n, 1, splits).expect("b"));
                        w4_out(&device, n as usize, |c| {
                            // SAFETY: as `w4_gemv`'s, for both weights.
                            unsafe {
                                match e {
                                    Some(e) => gate_up_gemv_bf16_ends(
                                        (
                                            a.ptr().cast_const(),
                                            b.ptr().cast_const(),
                                            up.ptr().cast_const(),
                                        ),
                                        (c, p.ptr()),
                                        (n, k, splits),
                                        0,
                                        e,
                                        std::ptr::null_mut(),
                                    ),
                                    None => gate_up_gemv_bf16(
                                        (
                                            a.ptr().cast_const(),
                                            b.ptr().cast_const(),
                                            up.ptr().cast_const(),
                                        ),
                                        (c, p.ptr()),
                                        (n, k, splits),
                                        0,
                                        std::ptr::null_mut(),
                                    ),
                                }
                            }
                        })
                    };
                    let (f, c) = (gu(&xb, Some(ends)), gu(&y, None));
                    let bad = mismatched_bits(&f.0, &c.0);
                    eprintln!(
                        "W4 normed gate_up_gemv_bf16 {what} offset={offset} row={row}: mismatched_bits={bad} {}",
                        if bad == 0 && f.1 { "PASS" } else { "FAIL" }
                    );
                    pass &= bad == 0 && f.1;
                }
            }
        }
        // The 35B's GDN group: qkv-z 8,192 and z 4,096 off one normed row.
        let (k, ns) = (2048u32, [8192u32, 4096]);
        let ku = k as usize;
        let gb = DeviceBuffer::upload(&device, &bf16_inputs(ku, 0x4D40)).expect("gain");
        let ws: Vec<DeviceBuffer> = (ns.iter().enumerate())
            .map(|(i, &n)| {
                DeviceBuffer::upload(&device, &bf16_inputs(n as usize * ku, 0x4D41 ^ i as u64))
                    .expect("w")
            })
            .collect();
        let splits: Vec<u32> = ns.iter().map(|&n| gemv_splits(n, k, 1)).collect();
        let ends = Ends {
            gain: gb.ptr().cast_const(),
            eps: W4_EPS,
            gain_offset: offset,
            residual: std::ptr::null(),
        };
        for (row, x) in w4_rows(ku, 0x4D50) {
            let xb = DeviceBuffer::upload(&device, &x).expect("x");
            let y = w4_norm(&device, &xb, &gb, k, offset);
            let run = |a: &DeviceBuffer, e: Option<Ends>| {
                let outs: Vec<DeviceBuffer> = (ns.iter())
                    .map(|&n| {
                        DeviceBuffer::upload(&device, &vec![SENTINEL; n as usize]).expect("c")
                    })
                    .collect();
                let parts: Vec<DeviceBuffer> = (ns.iter().zip(&splits))
                    .map(|(&n, &s)| w4_scratch(&device, gemv_scratch_bytes(n, 1, s).expect("b")))
                    .collect();
                let members: Vec<(*const std::ffi::c_void, *mut std::ffi::c_void, u32)> =
                    (ws.iter().zip(&outs).zip(&ns))
                        .map(|((w, c), &n)| (w.ptr().cast_const(), c.ptr(), n))
                        .collect();
                let pp: Vec<*mut std::ffi::c_void> = (parts.iter().zip(&splits))
                    .map(|(p, &s)| if s > 1 { p.ptr() } else { std::ptr::null_mut() })
                    .collect();
                // SAFETY: every member's weight, output and partials are their own allocations.
                unsafe {
                    match e {
                        Some(e) => gemv_group_bf16_ends(
                            a.ptr().cast_const(),
                            &members,
                            &pp,
                            (k, &splits),
                            e,
                            std::ptr::null_mut(),
                        ),
                        None => gemv_group_bf16_abt(
                            a.ptr().cast_const(),
                            &members,
                            &pp,
                            (k, &splits),
                            std::ptr::null_mut(),
                        ),
                    }
                }
                .expect("group");
                outs.iter()
                    .map(|c| c.download().expect("c"))
                    .collect::<Vec<_>>()
            };
            let (f, c) = (run(&xb, Some(ends)), run(&y, None));
            let bad: usize = f.iter().zip(&c).map(|(a, b)| mismatched_bits(a, b)).sum();
            eprintln!(
                "W4 normed gemv_group_bf16 35B GDN n={ns:?} offset={offset} row={row}: mismatched_bits={bad} {}",
                if bad == 0 { "PASS" } else { "FAIL" }
            );
            pass &= bad == 0;
        }
    }
    assert!(
        pass,
        "a normed bf16 GEMV differs from the norm then the GEMV"
    );
}

/// The bf16 GEMV's residual epilogue is today's bf16 GEMV then `add_bf16`, bit for bit: the 35B's
/// GDN out_proj (2,048 × 4,096) at one split, its own and the largest, residual rows of random
/// values, ±0, ±inf and NaN.
#[test]
fn the_bf16_gemv_residual_epilogue_is_the_gemv_then_the_add_bit_for_bit() {
    let device = h100();
    let mut pass = true;
    let (n, k) = (2048u32, 4096u32);
    let (nu, ku) = (n as usize, k as usize);
    let xb = DeviceBuffer::upload(&device, &bf16_inputs(ku, 0x4E00)).expect("x");
    let b = DeviceBuffer::upload(&device, &bf16_inputs(nu * ku, 0x4E01)).expect("b");
    let specials = [0x0000u16, 0x8000, 0x7F80, 0xFF80, 0x7FC0];
    let residual: Vec<u16> = (bf16_inputs(nu, 0x4E02).into_iter().enumerate())
        .map(|(i, r)| {
            if i % 7 == 0 {
                specials[(i / 7) % specials.len()]
            } else {
                r
            }
        })
        .collect();
    let rb = DeviceBuffer::upload(&device, &residual).expect("residual");
    let mut splits_set = vec![1u32, gemv_splits(n, k, 1), 16];
    splits_set.dedup();
    for splits in splits_set {
        let p = w4_scratch(&device, gemv_scratch_bytes(n, 1, splits).expect("bytes"));
        let ends = Ends {
            residual: rb.ptr().cast_const(),
            ..Ends::NONE
        };
        let fused = w4_out(&device, nu, |c| {
            // SAFETY: as `w4_gemv`'s, and the residual n bf16 apart from `c`.
            unsafe {
                gemv_bf16_ends(
                    (xb.ptr().cast_const(), b.ptr().cast_const(), c),
                    p.ptr(),
                    (n, k, splits),
                    ends,
                    std::ptr::null_mut(),
                )
            }
        });
        let v = w4_out(&device, nu, |c| {
            // SAFETY: as above.
            unsafe {
                gemv_split_bf16_abt(
                    (xb.ptr().cast_const(), b.ptr().cast_const(), c),
                    p.ptr(),
                    (n, k, splits),
                    std::ptr::null_mut(),
                )
            }
        });
        let vb = DeviceBuffer::upload(&device, &v.0).expect("v");
        let sum = DeviceBuffer::upload(&device, &vec![SENTINEL; nu]).expect("sum");
        // SAFETY: `vb`, `rb`, `sum` hold n bf16 each.
        unsafe {
            add_bf16(
                vb.ptr().cast_const(),
                rb.ptr().cast_const(),
                sum.ptr(),
                (1, n),
                std::ptr::null_mut(),
            )
        }
        .expect("add_bf16");
        let bad = mismatched_bits(&fused.0, &sum.download().expect("sum"));
        eprintln!(
            "W4 gemv_bf16 residual 35B GDN out_proj splits={splits}: mismatched_bits={bad} tail={} {}",
            fused.1,
            if bad == 0 && fused.1 { "PASS" } else { "FAIL" }
        );
        pass &= bad == 0 && fused.1;
    }
    assert!(
        pass,
        "a bf16 GEMV's residual epilogue differs from the GEMV then the add"
    );
}
