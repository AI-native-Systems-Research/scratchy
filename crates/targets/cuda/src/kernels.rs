// SPDX-License-Identifier: Apache-2.0
//! The CUDA kernels' launchers, linked from `libscratchy_cuda_kernels.a` (`build.rs`).
//!
//! Each kernel in `kernels/*.cu` exposes one `extern "C"` launcher taking device pointers,
//! dimensions and a stream. This module declares them and wraps each in a typed function. It does
//! not choose how memory or streams are managed: the pointers and the stream are the caller's.

use std::ffi::c_void;
use std::fmt;

/// `SCRATCHY_CUDA_ERR_SHAPE` in the `.cu` launchers: the shape was refused, nothing launched.
const ERR_SHAPE: i32 = -1;
/// `SCRATCHY_CUDA_ERR_HOST` in the `.cu` launchers: a C++ exception was caught on the host side
/// (e.g. a failed tensor-map encode).
const ERR_HOST: i32 = -2;

unsafe extern "C" {
    fn scratchy_cuda_gemm_bf16_abt(
        a: *const c_void,
        b: *const c_void,
        c: *mut c_void,
        m: u32,
        n: u32,
        k: u32,
        stream: *mut c_void,
    ) -> i32;
}

/// Why a launcher did not launch, or the launch failed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KernelError {
    /// The kernel does not take this shape; nothing was launched.
    Shape { m: u32, n: u32, k: u32 },
    /// The launcher's C++ host code threw (caught before it could reach Rust).
    HostException,
    /// A `cudaError_t` from setting the kernel's attributes or from the launch.
    Cuda(i32),
}

impl fmt::Display for KernelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KernelError::Shape { m, n, k } => write!(f, "kernel refused shape m={m} n={n} k={k}"),
            KernelError::HostException => write!(f, "kernel launcher's host code threw"),
            KernelError::Cuda(code) => write!(f, "CUDA error {code}"),
        }
    }
}

impl std::error::Error for KernelError {}

/// Shared tail of every launcher wrapper: the launcher's return code as a `Result`.
fn check(code: i32, m: u32, n: u32, k: u32) -> Result<(), KernelError> {
    match code {
        0 => Ok(()),
        ERR_SHAPE => Err(KernelError::Shape { m, n, k }),
        ERR_HOST => Err(KernelError::HostException),
        code => Err(KernelError::Cuda(code)),
    }
}

/// Launch the bf16 GEMM `C = A · Bᵀ` on the H100: `A` is `[m, k]`, `B` is `[n, k]` and `C` is
/// `[m, n]`, all row-major bf16, accumulated in f32. The launch is asynchronous; nothing is
/// synchronized.
///
/// `m`, `n` and `k` must be non-zero multiples of 128, 256 and 64; any other shape returns
/// [`KernelError::Shape`] and launches nothing.
///
/// # Safety
///
/// `a`, `b` and `c` must be 16-byte-aligned device pointers to at least `m*k`, `n*k` and `m*n`
/// bf16 elements, valid for the whole kernel, with `c` not overlapping `a` or `b`. `stream` must
/// be a live `cudaStream_t` of the current device, or null for the default stream.
pub unsafe fn gemm_bf16_abt(
    a: *const c_void,
    b: *const c_void,
    c: *mut c_void,
    m: u32,
    n: u32,
    k: u32,
    stream: *mut c_void,
) -> Result<(), KernelError> {
    // SAFETY: the caller upholds the pointer and stream contract above; the launcher checks the
    // shape and catches every C++ exception before returning.
    let code = unsafe { scratchy_cuda_gemm_bf16_abt(a, b, c, m, n, k, stream) };
    check(code, m, n, k)
}
