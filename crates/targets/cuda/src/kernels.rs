// SPDX-License-Identifier: Apache-2.0
//! The CUDA kernels' launchers, linked from `libscratchy_cuda_kernels.a` (`build.rs`).
//!
//! Each kernel in `kernels/*.cu` exposes one `extern "C"` launcher taking device pointers,
//! dimensions and a stream. This module declares them and wraps each in a typed function. It does
//! not choose how memory or streams are managed: the pointers and the stream are the caller's.
//!
//! A kernel whose launch needs a one-time setup (a function attribute such as its dynamic shared
//! memory size) also exposes a `_prepare` function, listed in [`PREPARE`]. [`prepare_all`] runs
//! them once, when the device is opened (`runtime::CudaDevice::open`); a launcher never repeats
//! that setup.

use std::ffi::c_void;
use std::fmt;

/// `SCRATCHY_CUDA_ERR_SHAPE` in the `.cu` launchers: the shape was refused, nothing launched.
const ERR_SHAPE: i32 = -1;
/// `SCRATCHY_CUDA_ERR_HOST` in the `.cu` launchers: a C++ exception was caught on the host side
/// (e.g. a failed tensor-map encode).
const ERR_HOST: i32 = -2;

unsafe extern "C" {
    fn scratchy_cuda_gemm_bf16_abt_prepare() -> i32;
    fn scratchy_cuda_gemm_bf16_abt(
        a: *const c_void,
        b: *const c_void,
        c: *mut c_void,
        m: u32,
        n: u32,
        k: u32,
        stream: *mut c_void,
    ) -> i32;
    fn scratchy_cuda_rms_norm_bf16(
        x: *const c_void,
        gain: *const c_void,
        y: *mut c_void,
        rows: u32,
        cols: u32,
        eps: f32,
        gain_offset: f32,
        stream: *mut c_void,
    ) -> i32;
    fn scratchy_cuda_add_bf16(
        a: *const c_void,
        b: *const c_void,
        out: *mut c_void,
        rows: u32,
        cols: u32,
        stream: *mut c_void,
    ) -> i32;
    fn scratchy_cuda_silu_mul_bf16(
        gate: *const c_void,
        up: *const c_void,
        out: *mut c_void,
        rows: u32,
        cols: u32,
        stream: *mut c_void,
    ) -> i32;
    fn scratchy_cuda_rope_bf16(
        x: *const c_void,
        table: *const c_void,
        out: *mut c_void,
        rows: u32,
        cols: u32,
        head_dim: u32,
        rot_dim: u32,
        stream: *mut c_void,
    ) -> i32;
    fn scratchy_cuda_gate_split_bf16(
        qg: *const c_void,
        q: *mut c_void,
        gate: *mut c_void,
        rows: u32,
        half_cols: u32,
        head_dim: u32,
        stream: *mut c_void,
    ) -> i32;
    fn scratchy_cuda_gate_apply_bf16(
        attn: *const c_void,
        gate: *const c_void,
        out: *mut c_void,
        rows: u32,
        cols: u32,
        stream: *mut c_void,
    ) -> i32;
}

/// Why a launcher did not launch, or the launch failed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KernelError {
    /// The GEMM does not take this shape; nothing was launched.
    Shape { m: u32, n: u32, k: u32 },
    /// A row kernel (RMSNorm, add, SiluMul, GateApply) does not take this shape; nothing was
    /// launched.
    RowShape { rows: u32, cols: u32 },
    /// The rope does not take this shape; nothing was launched.
    RopeShape {
        rows: u32,
        cols: u32,
        head_dim: u32,
        rot_dim: u32,
    },
    /// The gate split does not take this shape; nothing was launched.
    GateShape {
        rows: u32,
        half_cols: u32,
        head_dim: u32,
    },
    /// The launcher's C++ host code threw (caught before it could reach Rust).
    HostException,
    /// A `cudaError_t` from the kernel's one-time setup or from its launch.
    Cuda(i32),
}

impl fmt::Display for KernelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KernelError::Shape { m, n, k } => write!(f, "kernel refused shape m={m} n={n} k={k}"),
            KernelError::RowShape { rows, cols } => {
                write!(f, "kernel refused shape rows={rows} cols={cols}")
            }
            KernelError::RopeShape {
                rows,
                cols,
                head_dim,
                rot_dim,
            } => write!(
                f,
                "rope refused shape rows={rows} cols={cols} head_dim={head_dim} rot_dim={rot_dim}"
            ),
            KernelError::GateShape {
                rows,
                half_cols,
                head_dim,
            } => write!(
                f,
                "gate split refused shape rows={rows} half_cols={half_cols} head_dim={head_dim}"
            ),
            KernelError::HostException => write!(f, "kernel launcher's host code threw"),
            KernelError::Cuda(code) => write!(f, "CUDA error {code}"),
        }
    }
}

impl std::error::Error for KernelError {}

/// Every kernel's one-time setup, by kernel name: a later kernel with a setup of its own adds one
/// row.
const PREPARE: &[(&str, unsafe extern "C" fn() -> i32)] =
    &[("gemm_bf16_abt", scratchy_cuda_gemm_bf16_abt_prepare)];

/// Run every kernel's one-time setup on the current device, stopping at the first failure, which
/// names its kernel. Every launch on that device needs it to have succeeded first.
pub fn prepare_all() -> Result<(), (&'static str, KernelError)> {
    for &(kernel, prepare) in PREPARE {
        // SAFETY: a prepare function takes no argument, only sets attributes of its own kernel on
        // the current device, and returns a `cudaError_t`.
        match unsafe { prepare() } {
            0 => {}
            code => return Err((kernel, KernelError::Cuda(code))),
        }
    }
    Ok(())
}

/// Shared tail of every launcher wrapper: the launcher's return code as a `Result`, `shape` the
/// refusal its `SCRATCHY_CUDA_ERR_SHAPE` means.
fn check(code: i32, shape: KernelError) -> Result<(), KernelError> {
    match code {
        0 => Ok(()),
        ERR_SHAPE => Err(shape),
        ERR_HOST => Err(KernelError::HostException),
        code => Err(KernelError::Cuda(code)),
    }
}

/// Launch the bf16 GEMM `C = A · Bᵀ` on the H100: `A` is `[m, k]`, `B` is `[n, k]` and `C` is
/// `[m, n]`, all row-major bf16, accumulated in f32. The launch is asynchronous; nothing is
/// synchronized.
///
/// `m`, `n` and `k` must be non-zero multiples of 128, 256 and 64 (`step::GEMM_BF16_TILE`), and
/// none above `i32::MAX`, since ThunderKittens' `gl` stores each dimension as an `int`; any other
/// shape returns [`KernelError::Shape`] before any pointer is touched, and launches nothing. A
/// failed launch returns the launch's own `cudaError_t`, never an error an earlier, unrelated
/// runtime call left behind.
///
/// # Safety
///
/// [`prepare_all`] must have succeeded on the current device. `a`, `b` and `c` must be
/// 16-byte-aligned device pointers to at least `m*k`, `n*k` and `m*n` bf16 elements, valid for
/// the whole kernel, with `c` not overlapping `a` or `b`. `stream` must be a live `cudaStream_t`
/// of the current device, or null for the default stream.
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
    check(code, KernelError::Shape { m, n, k })
}

// THE ROW KERNELS (RMSNorm, add, SiluMul). Each launcher takes `(rows, cols)` and applies one
// shape rule: `rows` and `cols` non-zero, `cols` a multiple of 8 (one 16-byte vector of bf16:
// `step::RMS_NORM_BF16_COLS`, `ADD_BF16_COLS`, `SILU_MUL_BF16_COLS`), and `rows · cols` at most
// `i32::MAX`. Any other shape returns `KernelError::RowShape` before any pointer is touched, and
// launches nothing. A failed launch returns the launch's own `cudaError_t`. None needs a one-time
// setup (each uses under 48 KiB of shared memory), so none is in `PREPARE`.

/// Launch the bf16 RMSNorm: `y[r, c] = x[r, c] / sqrt(mean(x[r, :]²) + eps) · (gain_offset +
/// gain[c])`, `x` and `y` `[rows, cols]`, `gain` `[cols]`, in f32 with one bf16 rounding per
/// output. `gain_offset` is the convention's (`GainConvention::offset`). Asynchronous; the shape
/// rule is the row kernels'.
///
/// # Safety
///
/// `x`, `gain` and `y` must be 16-byte-aligned device pointers to at least `rows·cols`, `cols`
/// and `rows·cols` bf16 elements, valid for the whole kernel, with `y` overlapping neither input.
/// `stream` must be a live `cudaStream_t` of the current device, or null for the default stream.
pub unsafe fn rms_norm_bf16(
    x: *const c_void,
    gain: *const c_void,
    y: *mut c_void,
    (rows, cols): (u32, u32),
    eps: f32,
    gain_offset: f32,
    stream: *mut c_void,
) -> Result<(), KernelError> {
    // SAFETY: the caller upholds the pointer and stream contract above; the launcher checks the
    // shape before it touches a pointer.
    let code =
        unsafe { scratchy_cuda_rms_norm_bf16(x, gain, y, rows, cols, eps, gain_offset, stream) };
    check(code, KernelError::RowShape { rows, cols })
}

/// Launch the bf16 add `out = a + b` over `[rows, cols]`: one f32 add and one bf16 rounding per
/// element. Asynchronous; the shape rule is the row kernels'.
///
/// # Safety
///
/// `a`, `b` and `out` must be 16-byte-aligned device pointers to at least `rows·cols` bf16
/// elements each, valid for the whole kernel, with `out` overlapping neither input. `stream` must
/// be a live `cudaStream_t` of the current device, or null for the default stream.
pub unsafe fn add_bf16(
    a: *const c_void,
    b: *const c_void,
    out: *mut c_void,
    (rows, cols): (u32, u32),
    stream: *mut c_void,
) -> Result<(), KernelError> {
    // SAFETY: as `rms_norm_bf16`'s.
    let code = unsafe { scratchy_cuda_add_bf16(a, b, out, rows, cols, stream) };
    check(code, KernelError::RowShape { rows, cols })
}

/// Launch the bf16 `out = gate / (1 + e^(−gate)) · up` over `[rows, cols]`, in f32 with one bf16
/// rounding per element. Asynchronous; the shape rule is the row kernels'.
///
/// # Safety
///
/// `gate`, `up` and `out` must be 16-byte-aligned device pointers to at least `rows·cols` bf16
/// elements each, valid for the whole kernel, with `out` overlapping neither input. `stream` must
/// be a live `cudaStream_t` of the current device, or null for the default stream.
pub unsafe fn silu_mul_bf16(
    gate: *const c_void,
    up: *const c_void,
    out: *mut c_void,
    (rows, cols): (u32, u32),
    stream: *mut c_void,
) -> Result<(), KernelError> {
    // SAFETY: as `rms_norm_bf16`'s.
    let code = unsafe { scratchy_cuda_silu_mul_bf16(gate, up, out, rows, cols, stream) };
    check(code, KernelError::RowShape { rows, cols })
}

/// Launch the bf16 partial rotary embedding over `x` and `out` `[rows, cols]`, `cols = heads ·
/// head_dim`: per head, the first `rot_dim` lanes rotated in NeoX pairing within that width
/// (`(d, d + rot_dim/2)`) by `table`'s row for the token, `[cos(0..R/2) | sin(0..R/2)]`, in f32
/// with one bf16 rounding per output; the lanes `rot_dim..head_dim` copied bit for bit.
/// Asynchronous.
///
/// Refused with [`KernelError::RopeShape`], before any pointer is touched, unless `rows`, `cols`,
/// `head_dim` and `rot_dim` are non-zero, `rot_dim` is a multiple of 16
/// (`step::ROPE_BF16_ROT_MULTIPLE`) and at most `head_dim`, `head_dim` a multiple of 8, `cols` a
/// multiple of `head_dim`, and `rows · cols` at most `i32::MAX`. A failed launch returns the
/// launch's own `cudaError_t`.
///
/// # Safety
///
/// `x`, `table` and `out` must be 16-byte-aligned device pointers to at least `rows·cols`,
/// `rows·rot_dim` and `rows·cols` bf16 elements, valid for the whole kernel, with `out`
/// overlapping neither input. `stream` must be a live `cudaStream_t` of the current device, or
/// null for the default stream.
pub unsafe fn rope_bf16(
    x: *const c_void,
    table: *const c_void,
    out: *mut c_void,
    (rows, cols): (u32, u32),
    (head_dim, rot_dim): (u32, u32),
    stream: *mut c_void,
) -> Result<(), KernelError> {
    // SAFETY: as `rms_norm_bf16`'s.
    let code =
        unsafe { scratchy_cuda_rope_bf16(x, table, out, rows, cols, head_dim, rot_dim, stream) };
    check(
        code,
        KernelError::RopeShape {
            rows,
            cols,
            head_dim,
            rot_dim,
        },
    )
}

/// Launch the bf16 per-head gate split: for head `h` and lane `d`, `q[r, h·hd + d] =
/// qg[r, 2h·hd + d]` and `gate[r, h·hd + d] = qg[r, 2h·hd + hd + d]`, `qg` `[rows, 2·half_cols]`
/// and `q`, `gate` `[rows, half_cols]`. Copies, so bit-exact. Asynchronous.
///
/// Refused with [`KernelError::GateShape`], before any pointer is touched, unless `rows`,
/// `half_cols` and `head_dim` are non-zero, `head_dim` a multiple of 8
/// (`step::GATE_SPLIT_BF16_HEAD_MULTIPLE`), `half_cols` a multiple of `head_dim`, and
/// `rows · 2·half_cols` at most `i32::MAX`. A failed launch returns the launch's own
/// `cudaError_t`.
///
/// # Safety
///
/// `qg`, `q` and `gate` must be 16-byte-aligned device pointers to at least `rows·2·half_cols`,
/// `rows·half_cols` and `rows·half_cols` bf16 elements, valid for the whole kernel, no two
/// overlapping. `stream` must be a live `cudaStream_t` of the current device, or null for the
/// default stream.
pub unsafe fn gate_split_bf16(
    qg: *const c_void,
    q: *mut c_void,
    gate: *mut c_void,
    (rows, half_cols): (u32, u32),
    head_dim: u32,
    stream: *mut c_void,
) -> Result<(), KernelError> {
    // SAFETY: as `rms_norm_bf16`'s.
    let code =
        unsafe { scratchy_cuda_gate_split_bf16(qg, q, gate, rows, half_cols, head_dim, stream) };
    check(
        code,
        KernelError::GateShape {
            rows,
            half_cols,
            head_dim,
        },
    )
}

/// Launch the bf16 `out = attn / (1 + e^(−gate))` over `[rows, cols]`, in f32 with one bf16
/// rounding per element. Asynchronous; the shape rule is the row kernels'
/// (`step::GATE_APPLY_BF16_COLS`).
///
/// # Safety
///
/// `attn`, `gate` and `out` must be 16-byte-aligned device pointers to at least `rows·cols` bf16
/// elements each, valid for the whole kernel, with `out` overlapping neither input. `stream` must
/// be a live `cudaStream_t` of the current device, or null for the default stream.
pub unsafe fn gate_apply_bf16(
    attn: *const c_void,
    gate: *const c_void,
    out: *mut c_void,
    (rows, cols): (u32, u32),
    stream: *mut c_void,
) -> Result<(), KernelError> {
    // SAFETY: as `rms_norm_bf16`'s.
    let code = unsafe { scratchy_cuda_gate_apply_bf16(attn, gate, out, rows, cols, stream) };
    check(code, KernelError::RowShape { rows, cols })
}
