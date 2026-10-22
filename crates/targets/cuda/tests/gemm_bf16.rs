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
}

use common::numerics::{bf16_inputs, compare, mismatched_bits, reference, widen};
use scratchy_target_cuda::kernels::{
    KernelError, gemm_bf16_abt, gemm_narrow_bf16_abt, gemv_bf16_abt,
};
use scratchy_target_cuda::runtime::{CudaDevice, DeviceBuffer};

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
/// projections, the lm_head), then grid tails: one CTA of 8 columns, three warps of one CTA, a
/// partial last CTA with a `k` of 9 vectors (a lane takes 0 or 1), and one column.
const GEMV_SHAPES: &[(u32, u32, &str)] = &[
    (4096, 1024, "q_proj with its gate"),
    (512, 1024, "k_proj / v_proj"),
    (1024, 2048, "o_proj, the GDN out_proj"),
    (3584, 1024, "gate_proj / up_proj"),
    (1024, 3584, "down_proj"),
    (6144, 1024, "the GDN in_proj_qkv"),
    (2048, 1024, "the GDN in_proj_z"),
    (248_320, 1024, "the lm_head"),
    (8, 8, "one CTA, one k vector"),
    (24, 64, "three warps of a CTA"),
    (1000, 72, "a partial last CTA, 9 k vectors"),
    (1, 8, "one column"),
];

/// Shapes the GEMV launcher refuses: `k` zero or off the 8-wide vector, `n` zero, `n` above
/// `INT_MAX` (refused before any pointer is touched, so its buffers are small).
const GEMV_REFUSED: &[(u32, u32)] = &[(1024, 0), (1024, 100), (0, 1024), (2_147_483_648, 1024)];

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
