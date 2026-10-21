// SPDX-License-Identifier: Apache-2.0
//! The bf16 GEMM (`kernels/gemm_bf16.cu`, `C = A · Bᵀ`) against a CPU reference, on the H100.
//!
//! Built only under `nvcc` (`required-features`). No GPU fails rather than skips: this test is the
//! kernel's only gate, and a skip that passes would let it pass silently. The reference and the
//! bound are `common::numerics`'.

mod common {
    pub mod numerics;
}

use common::numerics::{bf16_inputs, compare};
use scratchy_target_cuda::kernels::{KernelError, gemm_bf16_abt};
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
