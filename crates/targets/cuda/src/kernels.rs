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
    fn scratchy_cuda_embed_bf16(
        tokens: *const c_void,
        table: *const c_void,
        out: *mut c_void,
        rows: u32,
        hidden: u32,
        vocab: u32,
        stream: *mut c_void,
    ) -> i32;
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
    fn scratchy_cuda_rms_norm_bf16_prepare() -> i32;
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
    fn scratchy_cuda_add_bf16_prepare() -> i32;
    fn scratchy_cuda_add_bf16(
        a: *const c_void,
        b: *const c_void,
        out: *mut c_void,
        rows: u32,
        cols: u32,
        stream: *mut c_void,
    ) -> i32;
    fn scratchy_cuda_silu_mul_bf16_prepare() -> i32;
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
    fn scratchy_cuda_gate_apply_bf16_prepare() -> i32;
    fn scratchy_cuda_gate_apply_bf16(
        attn: *const c_void,
        gate: *const c_void,
        out: *mut c_void,
        rows: u32,
        cols: u32,
        stream: *mut c_void,
    ) -> i32;
    fn scratchy_cuda_kv_append_bf16(
        k: *const c_void,
        v: *const c_void,
        slot_mapping: *const c_void,
        k_cache: *mut c_void,
        v_cache: *mut c_void,
        rows: u32,
        kv_heads: u32,
        head_dim: u32,
        block_size: u32,
        num_blocks: u32,
        stream: *mut c_void,
    ) -> i32;
    fn scratchy_cuda_attn_prefill_bf16_prepare() -> i32;
    fn scratchy_cuda_attn_prefill_bf16(
        q: *const c_void,
        k_cache: *const c_void,
        v_cache: *const c_void,
        block_table: *const c_void,
        sequence: *const c_void,
        out: *mut c_void,
        rows: u32,
        q_heads: u32,
        kv_heads: u32,
        head_dim: u32,
        block_size: u32,
        num_blocks: u32,
        scale: f32,
        stream: *mut c_void,
    ) -> i32;
    fn scratchy_cuda_gemm_narrow_bf16_abt_prepare() -> i32;
    fn scratchy_cuda_gemm_narrow_bf16_abt(
        a: *const c_void,
        b: *const c_void,
        c: *mut c_void,
        m: u32,
        n: u32,
        k: u32,
        stream: *mut c_void,
    ) -> i32;
    fn scratchy_cuda_gemv_bf16_abt_prepare() -> i32;
    fn scratchy_cuda_gemv_bf16_abt(
        a: *const c_void,
        b: *const c_void,
        c: *mut c_void,
        n: u32,
        k: u32,
        stream: *mut c_void,
    ) -> i32;
    fn scratchy_cuda_tma_box_probe_load(
        src: *const c_void,
        poison: *const c_void,
        dump: *mut c_void,
        rows: u32,
        cols: u32,
        tile_row: u32,
        tile_col: u32,
        swizzled: u32,
        stream: *mut c_void,
    ) -> i32;
    fn scratchy_cuda_tma_box_probe_store(
        pattern: *const c_void,
        dst: *mut c_void,
        rows: u32,
        cols: u32,
        tile_row: u32,
        tile_col: u32,
        swizzled: u32,
        stream: *mut c_void,
    ) -> i32;
    fn scratchy_cuda_attn_decode_bf16_prepare() -> i32;
    fn scratchy_cuda_attn_decode_bf16(
        q: *const c_void,
        k_cache: *const c_void,
        v_cache: *const c_void,
        block_table: *const c_void,
        sequence: *const c_void,
        out: *mut c_void,
        q_heads: u32,
        kv_heads: u32,
        head_dim: u32,
        block_size: u32,
        num_blocks: u32,
        scale: f32,
        stream: *mut c_void,
    ) -> i32;
    fn scratchy_cuda_gdn_conv1d(
        x: *const c_void,
        w: *const c_void,
        conv_state: *mut c_void,
        slot_elems: u64,
        sequence: *const c_void,
        conv_out: *mut c_void,
        rows: u32,
        conv_dim: u32,
        kernel: u32,
        stream: *mut c_void,
    ) -> i32;
    fn scratchy_cuda_gdn_gating(
        a: *const c_void,
        b: *const c_void,
        a_log: *const c_void,
        dt_bias: *const c_void,
        sequence: *const c_void,
        g: *mut c_void,
        beta: *mut c_void,
        rows: u32,
        nv: u32,
        stream: *mut c_void,
    ) -> i32;
    fn scratchy_cuda_gdn_scan(
        conv_out: *const c_void,
        g: *const c_void,
        beta: *const c_void,
        ssm_state: *mut c_void,
        slot_elems: u64,
        sequence: *const c_void,
        o: *mut c_void,
        rows: u32,
        nk: u32,
        nv: u32,
        head_k: u32,
        head_v: u32,
        scale: f32,
        stream: *mut c_void,
    ) -> i32;
    fn scratchy_cuda_gdn_rank_one_probe(
        s: *const c_void,
        t: *const c_void,
        u: *const c_void,
        out: *mut c_void,
        stream: *mut c_void,
    ) -> i32;
    fn scratchy_cuda_gdn_rms_norm_gated(
        o: *const c_void,
        z: *const c_void,
        w: *const c_void,
        sequence: *const c_void,
        out: *mut c_void,
        rows: u32,
        nv: u32,
        head_v: u32,
        eps: f32,
        stream: *mut c_void,
    ) -> i32;
}

/// Why a launcher did not launch, or the launch failed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KernelError {
    /// The embed does not take this shape; nothing was launched.
    EmbedShape { rows: u32, hidden: u32, vocab: u32 },
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
    /// The KV append does not take this shape; nothing was launched.
    KvAppendShape {
        rows: u32,
        kv_heads: u32,
        head_dim: u32,
        block_size: u32,
    },
    /// The prefill attention does not take this shape or scale; nothing was launched.
    AttnShape {
        rows: u32,
        q_heads: u32,
        kv_heads: u32,
        head_dim: u32,
        block_size: u32,
        num_blocks: u32,
    },
    /// The narrow GEMM does not take this shape; nothing was launched.
    NarrowShape { m: u32, n: u32, k: u32 },
    /// The GEMV does not take this shape; nothing was launched.
    GemvShape { n: u32, k: u32 },
    /// The decode attention does not take this shape or scale; nothing was launched.
    AttnDecodeShape {
        q_heads: u32,
        kv_heads: u32,
        head_dim: u32,
        block_size: u32,
        num_blocks: u32,
    },
    /// The GDN conv1d does not take this shape or state stride; nothing was launched.
    GdnConvShape {
        rows: u32,
        conv_dim: u32,
        kernel: u32,
    },
    /// The GDN gating does not take this shape; nothing was launched.
    GdnGatingShape { rows: u32, nv: u32 },
    /// The GDN scan does not take this geometry, state stride or scale; nothing was launched.
    GdnScanShape {
        rows: u32,
        nk: u32,
        nv: u32,
        head_k: u32,
        head_v: u32,
    },
    /// The GDN gated norm does not take this shape or eps; nothing was launched.
    GdnNormShape { rows: u32, nv: u32, head_v: u32 },
    /// The launcher's C++ host code threw (caught before it could reach Rust).
    HostException,
    /// A `cudaError_t` from the kernel's one-time setup or from its launch.
    Cuda(i32),
}

impl fmt::Display for KernelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KernelError::EmbedShape {
                rows,
                hidden,
                vocab,
            } => write!(
                f,
                "embed refused shape rows={rows} hidden={hidden} vocab={vocab}"
            ),
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
            KernelError::KvAppendShape {
                rows,
                kv_heads,
                head_dim,
                block_size,
            } => write!(
                f,
                "kv append refused shape rows={rows} kv_heads={kv_heads} head_dim={head_dim} \
                 block_size={block_size}"
            ),
            KernelError::AttnShape {
                rows,
                q_heads,
                kv_heads,
                head_dim,
                block_size,
                num_blocks,
            } => write!(
                f,
                "prefill attention refused shape rows={rows} q_heads={q_heads} \
                 kv_heads={kv_heads} head_dim={head_dim} block_size={block_size} \
                 num_blocks={num_blocks}, or its scale"
            ),
            KernelError::NarrowShape { m, n, k } => {
                write!(f, "narrow GEMM refused shape m={m} n={n} k={k}")
            }
            KernelError::GemvShape { n, k } => write!(f, "GEMV refused shape n={n} k={k}"),
            KernelError::AttnDecodeShape {
                q_heads,
                kv_heads,
                head_dim,
                block_size,
                num_blocks,
            } => write!(
                f,
                "decode attention refused shape q_heads={q_heads} kv_heads={kv_heads} \
                 head_dim={head_dim} block_size={block_size} num_blocks={num_blocks}, or its scale"
            ),
            KernelError::GdnConvShape {
                rows,
                conv_dim,
                kernel,
            } => write!(
                f,
                "GDN conv1d refused shape rows={rows} conv_dim={conv_dim} kernel={kernel}, or its \
                 state stride"
            ),
            KernelError::GdnGatingShape { rows, nv } => {
                write!(f, "GDN gating refused shape rows={rows} nv={nv}")
            }
            KernelError::GdnScanShape {
                rows,
                nk,
                nv,
                head_k,
                head_v,
            } => write!(
                f,
                "GDN scan refused shape rows={rows} nk={nk} nv={nv} head_k={head_k} \
                 head_v={head_v}, or its state stride or scale"
            ),
            KernelError::GdnNormShape { rows, nv, head_v } => write!(
                f,
                "GDN gated norm refused shape rows={rows} nv={nv} head_v={head_v}, or its eps"
            ),
            KernelError::HostException => write!(f, "kernel launcher's host code threw"),
            KernelError::Cuda(code) => write!(f, "CUDA error {code}"),
        }
    }
}

impl std::error::Error for KernelError {}

/// Every kernel's one-time setup, by kernel name: a later kernel with a setup of its own adds one
/// row.
const PREPARE: &[(&str, unsafe extern "C" fn() -> i32)] = &[
    ("gemm_bf16_abt", scratchy_cuda_gemm_bf16_abt_prepare),
    ("attn_prefill_bf16", scratchy_cuda_attn_prefill_bf16_prepare),
    (
        "gemm_narrow_bf16_abt",
        scratchy_cuda_gemm_narrow_bf16_abt_prepare,
    ),
    ("gemv_bf16_abt", scratchy_cuda_gemv_bf16_abt_prepare),
    ("rms_norm_bf16", scratchy_cuda_rms_norm_bf16_prepare),
    ("add_bf16", scratchy_cuda_add_bf16_prepare),
    ("silu_mul_bf16", scratchy_cuda_silu_mul_bf16_prepare),
    ("gate_apply_bf16", scratchy_cuda_gate_apply_bf16_prepare),
    ("attn_decode_bf16", scratchy_cuda_attn_decode_bf16_prepare),
];

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

/// Launch the bf16 token-embedding gather: `out[r, :] = table[tokens[r], :]` for every one of the
/// `rows` rows, padding rows included, the table's bits copied as they are; an id at or above
/// `vocab` writes a `+0.0` row and reads nothing. Asynchronous.
///
/// Refused with [`KernelError::EmbedShape`], before any pointer is touched, unless `rows`,
/// `hidden` and `vocab` are non-zero and at most `i32::MAX`, `hidden` a multiple of 8
/// (`step::EMBED_BF16_COLS`), and `rows · hidden` at most `i32::MAX`. A failed launch returns the
/// launch's own `cudaError_t`. No one-time setup, so it is not in [`PREPARE`].
///
/// # Safety
///
/// `tokens` must be a 4-byte-aligned device pointer to `rows` u32; `table` and `out`
/// 16-byte-aligned device pointers to at least `vocab·hidden` and `rows·hidden` bf16, valid for
/// the whole kernel, with `out` overlapping neither input. `stream` must be a live
/// `cudaStream_t` of the current device, or null for the default stream.
pub unsafe fn embed_bf16(
    tokens: *const c_void,
    table: *const c_void,
    out: *mut c_void,
    (rows, hidden, vocab): (u32, u32, u32),
    stream: *mut c_void,
) -> Result<(), KernelError> {
    // SAFETY: the caller upholds the pointer and stream contract above; the launcher checks the
    // shape before it touches a pointer.
    let code = unsafe { scratchy_cuda_embed_bf16(tokens, table, out, rows, hidden, vocab, stream) };
    check(
        code,
        KernelError::EmbedShape {
            rows,
            hidden,
            vocab,
        },
    )
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
// launches nothing. A failed launch returns the launch's own `cudaError_t`; a C++ exception from
// ThunderKittens' host code (a tensor-map encode), `KernelError::HostException`. Each is a
// ThunderKittens 16-row tile kernel whose dynamic shared memory is over the default 48 KiB, so
// each is in `PREPARE`, which must have run on the current device first.

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
/// Refused with [`KernelError::RopeShape`], before any pointer is touched, unless `rows`, `cols`
/// and `head_dim` are non-zero, `rot_dim` is 64 (`step::ROPE_BF16_KERNEL_ROT_DIM`, the one width
/// the kernel is instantiated for), `head_dim` a multiple of 64
/// (`step::ROPE_BF16_KERNEL_HEAD_MULTIPLE`), `cols` a multiple of `head_dim`, and `rows · cols`
/// at most `i32::MAX`. A failed launch returns the launch's own `cudaError_t`.
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
/// Refused with [`KernelError::GateShape`], before any pointer is touched, unless `rows` and
/// `half_cols` are non-zero, `head_dim` is 256 (`step::GATE_SPLIT_BF16_HEAD_DIM`, the one head dim
/// the kernel is instantiated for), `half_cols` a multiple of `head_dim`, and
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

/// Launch the paged KV cache's write: for each token row `t` whose `slot_mapping[t]` is not
/// `u32::MAX`, row `t` of `k` and of `v` (`[rows, kv_heads · head_dim]`) copied bit for bit into
/// `k_cache` and `v_cache` (`[num_blocks, kv_heads, 64, head_dim]`) at that slot. A slot whose
/// block is at or past `num_blocks` writes nothing. Asynchronous.
///
/// Refused with [`KernelError::KvAppendShape`], before any pointer is touched, unless `rows` and
/// `kv_heads` are non-zero, `head_dim` is 256 (`step::KV_APPEND_BF16_HEAD_DIM`, the one head dim
/// the kernel is instantiated for), `block_size` 64 (`step::CUDA_KV_BLOCK_SIZE`), and
/// `rows · kv_heads · head_dim` at most `i32::MAX`. A `num_blocks` of 0 writes nothing and
/// launches nothing. A failed
/// launch returns the launch's own `cudaError_t`.
///
/// # Safety
///
/// `k` and `v` must be 16-byte-aligned device pointers to at least `rows · kv_heads · head_dim`
/// bf16, `slot_mapping` to `rows` u32, `k_cache` and `v_cache` to `num_blocks · kv_heads · 64 ·
/// head_dim` bf16 each, valid for the whole kernel, the caches overlapping no input. `stream`
/// must be a live `cudaStream_t` of the current device, or null for the default stream.
pub unsafe fn kv_append_bf16(
    (k, v): (*const c_void, *const c_void),
    slot_mapping: *const c_void,
    (k_cache, v_cache): (*mut c_void, *mut c_void),
    (rows, kv_heads, head_dim): (u32, u32, u32),
    (block_size, num_blocks): (u32, u32),
    stream: *mut c_void,
) -> Result<(), KernelError> {
    // SAFETY: as `rms_norm_bf16`'s.
    let code = unsafe {
        scratchy_cuda_kv_append_bf16(
            k,
            v,
            slot_mapping,
            k_cache,
            v_cache,
            rows,
            kv_heads,
            head_dim,
            block_size,
            num_blocks,
            stream,
        )
    };
    check(
        code,
        KernelError::KvAppendShape {
            rows,
            kv_heads,
            head_dim,
            block_size,
        },
    )
}

/// Launch the causal GQA prefill attention of `q` (`[rows, q_heads · 256]`) over one layer's
/// paged `k_cache` and `v_cache` (`[num_blocks, kv_heads, 64, 256]`) into `out`, the keys read
/// through `block_table`, `P` and `L` read from `sequence`: query row `i` attends to keys `0 ..=
/// P + i` below `P + L`, query head `g` to kv head `g / (q_heads / kv_heads)`, the softmax scaled
/// by `scale`. Asynchronous.
///
/// Refused with [`KernelError::AttnShape`], before any pointer is touched, unless `head_dim` is
/// 256 (`step::ATTN_PREFILL_BF16_HEAD_DIM`), `rows` a non-zero multiple of 128
/// (`step::ATTN_PREFILL_BF16_ROWS`), `q_heads` and `kv_heads` non-zero with `q_heads` a multiple
/// of `kv_heads`, `block_size` 64, `num_blocks` non-zero, and `scale` finite and above 0. Returns
/// [`KernelError::HostException`] if building a tensor map throws, and the launch's own
/// `cudaError_t` if it fails.
///
/// # Safety
///
/// [`prepare_all`] must have succeeded on the current device. `q` and `out` must be
/// 16-byte-aligned device pointers to `rows · q_heads · 256` bf16, `out` overlapping no input;
/// `k_cache`, `v_cache` to `num_blocks · kv_heads · 64 · 256` bf16 each; `sequence` to `[P, L]`
/// u32 with `1 ≤ L ≤ rows`; `block_table` to at least `ceil((P + L) / 64)` u32 block ids, each
/// below `num_blocks`. All valid for the whole kernel. `stream` must be a live `cudaStream_t` of
/// the current device, or null for the default stream.
pub unsafe fn attn_prefill_bf16(
    (q, out): (*const c_void, *mut c_void),
    (k_cache, v_cache): (*const c_void, *const c_void),
    (block_table, sequence): (*const c_void, *const c_void),
    (rows, q_heads, kv_heads, head_dim): (u32, u32, u32, u32),
    (block_size, num_blocks): (u32, u32),
    scale: f32,
    stream: *mut c_void,
) -> Result<(), KernelError> {
    // SAFETY: the caller upholds the pointer and stream contract above; the launcher checks the
    // shape and scale before it touches a pointer and catches every C++ exception.
    let code = unsafe {
        scratchy_cuda_attn_prefill_bf16(
            q,
            k_cache,
            v_cache,
            block_table,
            sequence,
            out,
            rows,
            q_heads,
            kv_heads,
            head_dim,
            block_size,
            num_blocks,
            scale,
            stream,
        )
    };
    check(
        code,
        KernelError::AttnShape {
            rows,
            q_heads,
            kv_heads,
            head_dim,
            block_size,
            num_blocks,
        },
    )
}

/// Launch the narrow bf16 GEMM `C = A · Bᵀ` for at most 64 output columns: `A` is `[m, k]`, `B`
/// `[n, k]` and `C` `[m, n]`, row-major bf16, accumulated in f32 and rounded once. ThunderKittens'
/// WGMMA over 64-row tiles of `A` against all of `B`'s rows, in a fixed K order, so an output
/// row's bits depend only on its own row of `A` and on `B`, not on the launch or on `m`.
/// Asynchronous.
///
/// Refused with [`KernelError::NarrowShape`], before any pointer is touched, unless `m` is
/// non-zero, `n` a non-zero multiple of 8 and at most 64 (`step::GEMM_NARROW_BF16_MULTIPLE`,
/// `GEMM_NARROW_BF16_MAX_N`), `k` a non-zero multiple of 8, and `m` and `k` at most `i32::MAX`.
///
/// # Safety
///
/// [`prepare_all`] must have succeeded on the current device. `a`, `b` and `c` must be
/// 16-byte-aligned device pointers to at least `m*k`, `n*k` and `m*n` bf16, valid for the whole
/// kernel, with `c` overlapping neither input. `stream` must be a live `cudaStream_t` of the
/// current device, or null for the default stream.
pub unsafe fn gemm_narrow_bf16_abt(
    a: *const c_void,
    b: *const c_void,
    c: *mut c_void,
    (m, n, k): (u32, u32, u32),
    stream: *mut c_void,
) -> Result<(), KernelError> {
    // SAFETY: the caller upholds the pointer and stream contract above; the launcher checks the
    // shape before it touches a pointer and catches every C++ exception.
    let code = unsafe { scratchy_cuda_gemm_narrow_bf16_abt(a, b, c, m, n, k, stream) };
    check(code, KernelError::NarrowShape { m, n, k })
}

/// Launch the bf16 GEMV `c = a · Bᵀ`, the one-row GEMM: `a` is `[1, k]`, `B` `[n, k]` and `c`
/// `[1, n]`, row-major bf16, accumulated in f32 and rounded once. ThunderKittens' WGMMA over 64
/// columns of `c` per CTA, in a fixed K order, so the bits do not depend on the launch; only the
/// `n` elements of `c` are written. Asynchronous.
///
/// Refused with [`KernelError::GemvShape`], before any pointer is touched, unless `n` is
/// non-zero, `k` a non-zero multiple of 8 (`step::GEMV_BF16_MULTIPLE`), and both at most
/// `i32::MAX`.
///
/// # Safety
///
/// [`prepare_all`] must have succeeded on the current device. `a`, `b` and `c` must be
/// 16-byte-aligned device pointers to at least `k`, `n*k` and `n` bf16, valid for the whole
/// kernel, with `c` overlapping neither input. `stream` must be a live `cudaStream_t` of the
/// current device, or null for the default stream.
pub unsafe fn gemv_bf16_abt(
    a: *const c_void,
    b: *const c_void,
    c: *mut c_void,
    (n, k): (u32, u32),
    stream: *mut c_void,
) -> Result<(), KernelError> {
    // SAFETY: the caller upholds the pointer and stream contract above; the launcher checks the
    // shape before it touches a pointer and catches every C++ exception.
    let code = unsafe { scratchy_cuda_gemv_bf16_abt(a, b, c, n, k, stream) };
    check(code, KernelError::GemvShape { n, k })
}

// THE TMA BOX PROBE (`kernels/tma_box_probe.cu`): test support, for `tests/tma_bounds.rs` only.
// No `CudaStep` and no lowering use it. It records what a ThunderKittens TMA tile load and store
// do with a 64×64 bf16 box that runs past its array's edge, swizzled or plain: the facts the GEMV's
// and the narrow GEMM's choice of tiles rests on. Its shared memory is under 48 KiB, so it is not
// in `PREPARE`.

/// Test support for `tests/tma_bounds.rs` only. TMA-load the 64×64 box at tile `(tile_row,
/// tile_col)` of `src` (`[rows, cols]` bf16) into a swizzled (`swizzled`) or plain shared tile,
/// over the tile first loaded from `poison` (`[64, 64]` bf16), then TMA-store the tile into `dump`
/// (`[64, 64]` bf16). Refused with [`KernelError::RowShape`] when `rows` or `cols` is zero, `cols`
/// is not a multiple of 8, or a dimension or tile coordinate is at or above `i32::MAX / 64`.
/// Asynchronous.
///
/// # Safety
///
/// `src`, `poison` and `dump` must be 16-byte-aligned device pointers to `rows · cols`, 4096 and
/// 4096 bf16, valid for the whole kernel; `dump` overlaps neither input. A swizzled load may read
/// past `src`'s `rows · cols` elements, up to the next 64-element boundary of its last row (that
/// is what it records), so `src` must be readable that far. `stream` must be a live
/// `cudaStream_t` of the current device, or null for the default stream.
#[doc(hidden)]
pub unsafe fn tma_box_probe_load(
    (src, poison, dump): (*const c_void, *const c_void, *mut c_void),
    (rows, cols): (u32, u32),
    (tile_row, tile_col): (u32, u32),
    swizzled: bool,
    stream: *mut c_void,
) -> Result<(), KernelError> {
    // SAFETY: the caller upholds the pointer and stream contract above; the launcher checks the
    // shape before it touches a pointer and catches every C++ exception.
    let code = unsafe {
        scratchy_cuda_tma_box_probe_load(
            src,
            poison,
            dump,
            rows,
            cols,
            tile_row,
            tile_col,
            u32::from(swizzled),
            stream,
        )
    };
    check(code, KernelError::RowShape { rows, cols })
}

/// Test support for `tests/tma_bounds.rs` only. TMA-load `pattern` (`[64, 64]` bf16) into a
/// swizzled (`swizzled`) or plain shared tile, then TMA-store it at tile `(tile_row, tile_col)` of
/// `dst` (`[rows, cols]` bf16). Refused as [`tma_box_probe_load`]. Asynchronous.
///
/// # Safety
///
/// `pattern` and `dst` must be 16-byte-aligned device pointers to 4096 and `rows · cols` bf16,
/// valid for the whole kernel, not overlapping. A swizzled store may write past `dst`'s
/// `rows · cols` elements, up to the next 64-element boundary of its last row (that is what it
/// records), so `dst` must be writable that far. `stream` must be a live `cudaStream_t` of the
/// current device, or null for the default stream.
#[doc(hidden)]
pub unsafe fn tma_box_probe_store(
    (pattern, dst): (*const c_void, *mut c_void),
    (rows, cols): (u32, u32),
    (tile_row, tile_col): (u32, u32),
    swizzled: bool,
    stream: *mut c_void,
) -> Result<(), KernelError> {
    // SAFETY: the caller upholds the pointer and stream contract above; the launcher checks the
    // shape before it touches a pointer and catches every C++ exception.
    let code = unsafe {
        scratchy_cuda_tma_box_probe_store(
            pattern,
            dst,
            rows,
            cols,
            tile_row,
            tile_col,
            u32::from(swizzled),
            stream,
        )
    };
    check(code, KernelError::RowShape { rows, cols })
}

/// Launch the GQA decode attention of one query row `q` (`[1, q_heads · 256]`) over one layer's
/// paged `k_cache` and `v_cache` (`[num_blocks, kv_heads, 64, 256]`) into `out`, the keys
/// `0 .. P + L` read through `block_table`, `P` and `L` read from `sequence`; query head `g` reads
/// kv head `g / (q_heads / kv_heads)`, the softmax scaled by `scale`. Whole 64-slot blocks are
/// loaded; the slots at or past `P + L` are neutralised before use (their scores set to −inf,
/// their V rows zeroed), so no unwritten slot reaches the output. Asynchronous.
///
/// Refused with [`KernelError::AttnDecodeShape`], before any pointer is touched, unless
/// `head_dim` is 256 (`step::ATTN_DECODE_BF16_HEAD_DIM`), `q_heads` and `kv_heads` non-zero with
/// `q_heads` a multiple of `kv_heads` by at most 16 (`step::ATTN_DECODE_BF16_MAX_GROUP`),
/// `block_size` 64, `num_blocks` non-zero, and `scale` finite and above 0. Returns the launch's
/// own `cudaError_t` if it fails.
///
/// # Safety
///
/// [`prepare_all`] must have succeeded on the current device. `q` and `out` must be 16-byte-aligned device pointers to `q_heads · 256` bf16, `out`
/// overlapping no input; `k_cache`, `v_cache` to `num_blocks · kv_heads · 64 · 256` bf16 each;
/// `sequence` to `[P, L]` u32 with `L ≥ 1`; `block_table` to at least `ceil((P + L) / 64)` u32
/// block ids, each below `num_blocks`. All valid for the whole kernel. `stream` must be a live
/// `cudaStream_t` of the current device, or null for the default stream.
pub unsafe fn attn_decode_bf16(
    (q, out): (*const c_void, *mut c_void),
    (k_cache, v_cache): (*const c_void, *const c_void),
    (block_table, sequence): (*const c_void, *const c_void),
    (q_heads, kv_heads, head_dim): (u32, u32, u32),
    (block_size, num_blocks): (u32, u32),
    scale: f32,
    stream: *mut c_void,
) -> Result<(), KernelError> {
    // SAFETY: the caller upholds the pointer and stream contract above; the launcher checks the
    // shape and scale before it touches a pointer.
    let code = unsafe {
        scratchy_cuda_attn_decode_bf16(
            q,
            k_cache,
            v_cache,
            block_table,
            sequence,
            out,
            q_heads,
            kv_heads,
            head_dim,
            block_size,
            num_blocks,
            scale,
            stream,
        )
    };
    check(
        code,
        KernelError::AttnDecodeShape {
            q_heads,
            kv_heads,
            head_dim,
            block_size,
            num_blocks,
        },
    )
}

// THE GDN KERNELS. Four launches make one `GatedDeltaNet` step, in this order on one stream, each
// reading `[slot, q_len, start, 0]` u32 from `sequence` (`start` 1 on a sequence's first chunk, 0
// to continue from its slot) and touching only the `q_len` real rows. Each is plain CUDA with
// under 48 KiB of static shared memory, so none is in `PREPARE`.

/// Launch the GDN causal depthwise conv1d, then SiLU: `conv_out[t, c]` (f32) for the real rows of
/// `x [rows, conv_dim]` (bf16) under the taps `w [conv_dim, kernel]` (f32), the rows before 0 the
/// slot's conv ring (zero, never read, on a fresh start); then the ring updated to the last
/// `kernel − 1` inputs. Asynchronous.
///
/// Refused with [`KernelError::GdnConvShape`], before any pointer is touched, unless `rows` and
/// `conv_dim` are non-zero, `kernel` in `step::GDN_CONV_KERNEL_RANGE`, `rows · conv_dim` at most
/// `i32::MAX`, and `slot_elems` exactly `conv_dim · (kernel − 1)`.
///
/// # Safety
///
/// `x`, `w`, `conv_out` must be 16-byte-aligned device pointers to `rows·conv_dim` bf16,
/// `conv_dim·kernel` f32 and `rows·conv_dim` f32; `sequence` to 4 u32 with `1 ≤ q_len ≤ rows` and
/// `slot` a slot of `conv_state`, which must hold `(slot + 1) · slot_elems` f32. All valid for
/// the whole kernel; `conv_out` and `conv_state` overlap nothing else. `stream` must be a live
/// `cudaStream_t` of the current device, or null for the default stream.
pub unsafe fn gdn_conv1d(
    (x, w): (*const c_void, *const c_void),
    (conv_state, slot_elems): (*mut c_void, u64),
    sequence: *const c_void,
    conv_out: *mut c_void,
    (rows, conv_dim, kernel): (u32, u32, u32),
    stream: *mut c_void,
) -> Result<(), KernelError> {
    // SAFETY: as `rms_norm_bf16`'s.
    let code = unsafe {
        scratchy_cuda_gdn_conv1d(
            x, w, conv_state, slot_elems, sequence, conv_out, rows, conv_dim, kernel, stream,
        )
    };
    check(
        code,
        KernelError::GdnConvShape {
            rows,
            conv_dim,
            kernel,
        },
    )
}

/// Launch the GDN gating: for the real rows of `a` and `b` (`[rows, nv]` bf16), `g = −exp(a_log)
/// · softplus(a + dt_bias)` and `beta = sigmoid(b)`, f32 `[rows, nv]` each. Asynchronous.
///
/// Refused with [`KernelError::GdnGatingShape`], before any pointer is touched, unless `rows` and
/// `nv` are non-zero and `rows · nv` at most `i32::MAX`.
///
/// # Safety
///
/// `a`, `b` must be device pointers to `rows·nv` bf16, `a_log`, `dt_bias` to `nv` f32, `g`,
/// `beta` to `rows·nv` f32, `sequence` to 4 u32 with `1 ≤ q_len ≤ rows`; all 4-byte aligned and
/// valid for the whole kernel, `g` and `beta` overlapping nothing else. `stream` must be a live
/// `cudaStream_t` of the current device, or null for the default stream.
pub unsafe fn gdn_gating(
    (a, b): (*const c_void, *const c_void),
    (a_log, dt_bias): (*const c_void, *const c_void),
    sequence: *const c_void,
    (g, beta): (*mut c_void, *mut c_void),
    (rows, nv): (u32, u32),
    stream: *mut c_void,
) -> Result<(), KernelError> {
    // SAFETY: as `rms_norm_bf16`'s.
    let code = unsafe {
        scratchy_cuda_gdn_gating(a, b, a_log, dt_bias, sequence, g, beta, rows, nv, stream)
    };
    check(code, KernelError::GdnGatingShape { rows, nv })
}

/// Launch the GDN recurrent gated delta-rule scan: per value head, the slot's f32 state (zero,
/// never read, on a fresh start) through the real tokens of `conv_out` (`q | k | v`), `g` and
/// `beta`, each token's output into `o [rows, nv·head_v]` (f32); the state written back once.
/// Value head `h` reads key head `h / (nv / nk)`. Each warp holds 16 state rows as one
/// ThunderKittens `rt_fl<16, 128>`: the matrix-vector products are `mul_col` then `row_sum`, the
/// rank-1 update `row_map<fma_AxCtB>` (`kernels/gdn_scan.cu`). Asynchronous.
///
/// Refused with [`KernelError::GdnScanShape`], before any pointer is touched, unless `head_k` is
/// 128 (`step::GDN_HEAD_K`), `head_v` a non-zero multiple of 32 and at most 256, `nk` and `nv`
/// non-zero with `nv` a multiple of `nk`, `rows` non-zero, `rows · conv_dim` at most `i32::MAX`
/// (`conv_dim = 2·nk·128 + nv·head_v`), `scale` finite and above 0, and `slot_elems` exactly
/// `nv · head_v · 128`.
///
/// # Safety
///
/// `conv_out` must be a device pointer to `rows·conv_dim` f32, `g` and `beta` to `rows·nv` f32,
/// `o` to `rows·nv·head_v` f32, `sequence` to 4 u32 with `1 ≤ q_len ≤ rows` and `slot` a slot of
/// `ssm_state`, which must hold `(slot + 1) · slot_elems` f32. All 16-byte aligned (`conv_out`'s
/// q, k and v rows are staged by bulk copy) and valid for the whole kernel; `o` and `ssm_state` overlap nothing else. `stream` must be a live
/// `cudaStream_t` of the current device, or null for the default stream.
pub unsafe fn gdn_scan(
    (conv_out, g, beta): (*const c_void, *const c_void, *const c_void),
    (ssm_state, slot_elems): (*mut c_void, u64),
    sequence: *const c_void,
    o: *mut c_void,
    (rows, nk, nv): (u32, u32, u32),
    (head_k, head_v, scale): (u32, u32, f32),
    stream: *mut c_void,
) -> Result<(), KernelError> {
    // SAFETY: as `rms_norm_bf16`'s.
    let code = unsafe {
        scratchy_cuda_gdn_scan(
            conv_out, g, beta, ssm_state, slot_elems, sequence, o, rows, nk, nv, head_k, head_v,
            scale, stream,
        )
    };
    check(
        code,
        KernelError::GdnScanShape {
            rows,
            nk,
            nv,
            head_k,
            head_v,
        },
    )
}

/// Test support for `tests/gdn.rs` only (R8): `out = T · u + S` by the GDN scan's own rank-1
/// update, `warp::row_map<base_ops::fma_AxCtB>(S, T, S, u)` on its `rt_fl<16, 128>` state tile
/// (`kernels/gdn_scan.cu`), `u` one value per row. No `CudaStep` uses it. Never refused: a failed
/// launch returns its `cudaError_t`. Asynchronous.
///
/// # Safety
///
/// `s`, `t` and `out` must be 16-byte-aligned device pointers to 16 · 128 f32 and `u` to 16 f32,
/// valid for the whole kernel, `out` overlapping no input. `stream` must be a live `cudaStream_t`
/// of the current device, or null for the default stream.
#[doc(hidden)]
pub unsafe fn gdn_rank_one_probe(
    (s, t, u): (*const c_void, *const c_void, *const c_void),
    out: *mut c_void,
    stream: *mut c_void,
) -> Result<(), KernelError> {
    // SAFETY: the caller upholds the pointer and stream contract above; the launcher catches
    // every C++ exception.
    let code = unsafe { scratchy_cuda_gdn_rank_one_probe(s, t, u, out, stream) };
    check(
        code,
        KernelError::RowShape {
            rows: 16,
            cols: 128,
        },
    )
}

/// Launch the GDN gated RMSNorm: per real row and value head, `o / sqrt(mean(o²) + eps) · w ·
/// SiLU(z)` over the head's `head_v` lanes, rounded once to bf16 into `out`; every padding row of
/// `out` written `+0.0` without reading `o` or `z`. Asynchronous.
///
/// Refused with [`KernelError::GdnNormShape`], before any pointer is touched, unless `rows` and
/// `nv` are non-zero, `head_v` a non-zero multiple of 32 and at most 256, `rows · nv · head_v` at
/// most `i32::MAX`, and `eps` finite and above 0.
///
/// # Safety
///
/// `o` must be a device pointer to `rows·nv·head_v` f32, `z` and `out` to that many bf16, `w` to
/// `head_v` f32, `sequence` to 4 u32 with `q_len ≤ rows`; all 4-byte aligned and valid for the
/// whole kernel, `out` overlapping no input. `stream` must be a live `cudaStream_t` of the
/// current device, or null for the default stream.
pub unsafe fn gdn_rms_norm_gated(
    (o, z, w): (*const c_void, *const c_void, *const c_void),
    sequence: *const c_void,
    out: *mut c_void,
    (rows, nv, head_v): (u32, u32, u32),
    eps: f32,
    stream: *mut c_void,
) -> Result<(), KernelError> {
    // SAFETY: as `rms_norm_bf16`'s.
    let code = unsafe {
        scratchy_cuda_gdn_rms_norm_gated(o, z, w, sequence, out, rows, nv, head_v, eps, stream)
    };
    check(code, KernelError::GdnNormShape { rows, nv, head_v })
}
