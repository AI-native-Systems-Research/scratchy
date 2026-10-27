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

use crate::step::RopePairing;

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
    fn scratchy_cuda_gemm_col64_bf16_abt_prepare() -> i32;
    fn scratchy_cuda_gemm_col64_bf16_abt(
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
    fn scratchy_cuda_rms_norm_unit_bf16(
        x: *const c_void,
        y: *mut c_void,
        rows: u32,
        cols: u32,
        eps: f32,
        stream: *mut c_void,
    ) -> i32;
    fn scratchy_cuda_pointwise_bf16_prepare() -> i32;
    fn scratchy_cuda_scale_bf16(
        x: *const c_void,
        out: *mut c_void,
        rows: u32,
        cols: u32,
        scale: f32,
        stream: *mut c_void,
    ) -> i32;
    fn scratchy_cuda_layer_scale_bf16(
        x: *const c_void,
        w: *const c_void,
        out: *mut c_void,
        rows: u32,
        cols: u32,
        stream: *mut c_void,
    ) -> i32;
    fn scratchy_cuda_soft_cap_bf16(
        x: *const c_void,
        out: *mut c_void,
        rows: u32,
        cols: u32,
        cap: f32,
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
    fn scratchy_cuda_gelu_mul_bf16(
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
    fn scratchy_cuda_rope_paired_bf16(
        x: *const c_void,
        table: *const c_void,
        out: *mut c_void,
        rows: u32,
        cols: u32,
        head_dim: u32,
        rot_dim: u32,
        pairing: u32,
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
    fn scratchy_cuda_gate_scale_bf16_prepare() -> i32;
    fn scratchy_cuda_gate_scale_bf16(
        routed: *const c_void,
        shared: *const c_void,
        g: *const c_void,
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
        window: u32,
        stream: *mut c_void,
    ) -> i32;
    fn scratchy_cuda_attn_prefill_d512_bf16_prepare() -> i32;
    fn scratchy_cuda_attn_prefill_d512_bf16(
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
        window: u32,
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
        window: u32,
        stream: *mut c_void,
    ) -> i32;
    fn scratchy_cuda_attn_decode_d512_bf16_prepare() -> i32;
    fn scratchy_cuda_attn_decode_d512_bf16(
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
        window: u32,
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
    fn scratchy_cuda_moe_route_bf16(
        logits: *const c_void,
        expert_scale: *const c_void,
        topk_idx: *mut c_void,
        scores: *mut c_void,
        m: u32,
        experts: u32,
        top_k: u32,
        renorm: u32,
        program: u32,
        scale: f32,
        stream: *mut c_void,
    ) -> i32;
    fn scratchy_cuda_moe_combine_bf16(
        down: *const c_void,
        pos: *const c_void,
        scores: *const c_void,
        out: *mut c_void,
        m: u32,
        top_k: u32,
        hidden: u32,
        down_rows: u32,
        sorted: u32,
        stream: *mut c_void,
    ) -> i32;
    fn scratchy_cuda_moe_sort(
        x: *const c_void,
        topk_idx: *const c_void,
        chunk_hist: *mut c_void,
        counts: *mut c_void,
        offsets: *mut c_void,
        tile_expert: *mut c_void,
        pos: *mut c_void,
        rows: *mut c_void,
        m: u32,
        experts: u32,
        top_k: u32,
        hidden: u32,
        max_tiles: u32,
        stream: *mut c_void,
    ) -> i32;
    fn scratchy_cuda_expert_gemm_bf16_prepare() -> i32;
    fn scratchy_cuda_expert_gemm_e128_bf16_prepare() -> i32;
    fn scratchy_cuda_expert_gemm_e128_col64_bf16_prepare() -> i32;
    fn scratchy_cuda_expert_gemm_bf16(
        a: *const c_void,
        b: *const c_void,
        offsets: *const c_void,
        tile_expert: *const c_void,
        c: *mut c_void,
        max_tiles: u32,
        n: u32,
        k: u32,
        experts: u32,
        expert_rows: u32,
        n_tile: u32,
        stream: *mut c_void,
    ) -> i32;
    fn scratchy_cuda_expert_gemv_bf16_prepare() -> i32;
    fn scratchy_cuda_expert_gemv_bf16(
        a: *const c_void,
        b: *const c_void,
        topk_idx: *const c_void,
        c: *mut c_void,
        m: u32,
        top_k: u32,
        n: u32,
        k: u32,
        experts: u32,
        expert_rows: u32,
        pair_rows: u32,
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
    /// A row kernel (RMSNorm, add, SiluMul, GateApply, and Gemma's unit RMSNorm, GeluMul, scale,
    /// layer scale and soft-cap) does not take this shape; nothing was launched.
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
    /// The prefill attention does not take this shape, scale or window; nothing was launched.
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
    /// The decode attention does not take this shape, scale or window; nothing was launched.
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
    /// A MoE kernel (`kernel`, its file's stem) does not take this shape; `dims` are its launch
    /// dimensions in its launcher's order. Nothing was launched.
    MoeShape {
        kernel: &'static str,
        dims: [u32; 4],
    },
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
                 num_blocks={num_blocks}, or its scale or window"
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
                 head_dim={head_dim} block_size={block_size} num_blocks={num_blocks}, or its scale \
                 or window"
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
            KernelError::MoeShape { kernel, dims } => {
                write!(f, "{kernel} refused shape {dims:?}")
            }
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
    (
        "gemm_col64_bf16_abt",
        scratchy_cuda_gemm_col64_bf16_abt_prepare,
    ),
    ("attn_prefill_bf16", scratchy_cuda_attn_prefill_bf16_prepare),
    (
        "gemm_narrow_bf16_abt",
        scratchy_cuda_gemm_narrow_bf16_abt_prepare,
    ),
    ("gemv_bf16_abt", scratchy_cuda_gemv_bf16_abt_prepare),
    ("rms_norm_bf16", scratchy_cuda_rms_norm_bf16_prepare),
    ("add_bf16", scratchy_cuda_add_bf16_prepare),
    ("silu_mul_bf16", scratchy_cuda_silu_mul_bf16_prepare),
    ("pointwise_bf16", scratchy_cuda_pointwise_bf16_prepare),
    ("gate_apply_bf16", scratchy_cuda_gate_apply_bf16_prepare),
    ("gate_scale_bf16", scratchy_cuda_gate_scale_bf16_prepare),
    ("attn_decode_bf16", scratchy_cuda_attn_decode_bf16_prepare),
    (
        "attn_prefill_d512_bf16",
        scratchy_cuda_attn_prefill_d512_bf16_prepare,
    ),
    (
        "attn_decode_d512_bf16",
        scratchy_cuda_attn_decode_d512_bf16_prepare,
    ),
    ("expert_gemm_bf16", scratchy_cuda_expert_gemm_bf16_prepare),
    (
        "expert_gemm_e128_bf16",
        scratchy_cuda_expert_gemm_e128_bf16_prepare,
    ),
    (
        "expert_gemm_e128_col64_bf16",
        scratchy_cuda_expert_gemm_e128_col64_bf16_prepare,
    ),
    ("expert_gemv_bf16", scratchy_cuda_expert_gemv_bf16_prepare),
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

/// Launch the bf16 GEMM `C = A · Bᵀ` on 64-column tiles (`kernels/gemm_bf16.cu`'s template at one
/// 64-column B block): the operands and the contract of [`gemm_bf16_abt`], for an `n` that is a
/// multiple of 64 (Gemma 4's 128, 704 and 2112). On an `n` both take, each output element is the
/// bits [`gemm_bf16_abt`] gives: the same WGMMA K order. Asynchronous.
///
/// `m`, `n` and `k` must be non-zero multiples of 128, 64 and 64 (`step::GEMM_COL64_TILE`), none
/// above `i32::MAX`; any other shape returns [`KernelError::Shape`] before any pointer is touched,
/// and launches nothing.
///
/// # Safety
///
/// As [`gemm_bf16_abt`]'s.
pub unsafe fn gemm_col64_bf16_abt(
    a: *const c_void,
    b: *const c_void,
    c: *mut c_void,
    (m, n, k): (u32, u32, u32),
    stream: *mut c_void,
) -> Result<(), KernelError> {
    // SAFETY: as `gemm_bf16_abt`'s.
    let code = unsafe { scratchy_cuda_gemm_col64_bf16_abt(a, b, c, m, n, k, stream) };
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

/// Launch the bf16 RMSNorm with no gain (Gemma's `v_norm`, `kernels/rms_norm_bf16.cu`'s unit
/// instantiation): `y[r, c] = x[r, c] / sqrt(mean(x[r, :]²) + eps)`, `x` and `y` `[rows, cols]`,
/// the bits of [`rms_norm_bf16`] with a gain of all ones and offset 0. Asynchronous; the shape rule
/// is the row kernels'.
///
/// # Safety
///
/// `x` and `y` must be 16-byte-aligned device pointers to at least `rows·cols` bf16 elements,
/// valid for the whole kernel, with `y` not overlapping `x`. `stream` must be a live
/// `cudaStream_t` of the current device, or null for the default stream.
pub unsafe fn rms_norm_unit_bf16(
    x: *const c_void,
    y: *mut c_void,
    (rows, cols): (u32, u32),
    eps: f32,
    stream: *mut c_void,
) -> Result<(), KernelError> {
    // SAFETY: as `rms_norm_bf16`'s.
    let code = unsafe { scratchy_cuda_rms_norm_unit_bf16(x, y, rows, cols, eps, stream) };
    check(code, KernelError::RowShape { rows, cols })
}

/// Launch the bf16 `out = 0.5 · gate · (1 + tanh(√(2/π) · (gate + 0.044715 · gate³))) · up` (GeGLU,
/// `kernels/silu_mul_bf16.cu`'s GELU instantiation) over `[rows, cols]`, in f32 with one bf16
/// rounding per element. Asynchronous; the shape rule is the row kernels'.
///
/// # Safety
///
/// As [`silu_mul_bf16`]'s.
pub unsafe fn gelu_mul_bf16(
    gate: *const c_void,
    up: *const c_void,
    out: *mut c_void,
    (rows, cols): (u32, u32),
    stream: *mut c_void,
) -> Result<(), KernelError> {
    // SAFETY: as `rms_norm_bf16`'s.
    let code = unsafe { scratchy_cuda_gelu_mul_bf16(gate, up, out, rows, cols, stream) };
    check(code, KernelError::RowShape { rows, cols })
}

// GEMMA'S ONE-OPERAND ROW MAPS (`kernels/pointwise_bf16.cu`): the row kernels' shape rule
// (`step::SCALE_BF16_COLS`, `SOFT_CAP_BF16_COLS`), one setup in `PREPARE`. `out` may equal `x`
// exactly (the same bits as out of place), and may overlap it in no other way.

/// Launch the bf16 `out = x · scale` over `[rows, cols]` (Gemma's embedding scale): one f32
/// multiply of the exact input, one bf16 rounding. Asynchronous.
///
/// # Safety
///
/// `x` and `out` must be 16-byte-aligned device pointers to at least `rows·cols` bf16 elements,
/// valid for the whole kernel, `out` equal to `x` or overlapping it nowhere. `stream` must be a
/// live `cudaStream_t` of the current device, or null for the default stream.
pub unsafe fn scale_bf16(
    x: *const c_void,
    out: *mut c_void,
    (rows, cols): (u32, u32),
    scale: f32,
    stream: *mut c_void,
) -> Result<(), KernelError> {
    // SAFETY: as `rms_norm_bf16`'s.
    let code = unsafe { scratchy_cuda_scale_bf16(x, out, rows, cols, scale, stream) };
    check(code, KernelError::RowShape { rows, cols })
}

/// Launch the bf16 `out = x · w[0]` over `[rows, cols]`, `w` a `[1]` bf16 weight (Gemma's
/// `layer_scalar`) widened exactly: one f32 multiply, one bf16 rounding. Asynchronous.
///
/// # Safety
///
/// As [`scale_bf16`]'s, and `w` a device pointer to one bf16, overlapping neither `x` nor `out`.
pub unsafe fn layer_scale_bf16(
    x: *const c_void,
    w: *const c_void,
    out: *mut c_void,
    (rows, cols): (u32, u32),
    stream: *mut c_void,
) -> Result<(), KernelError> {
    // SAFETY: as `rms_norm_bf16`'s.
    let code = unsafe { scratchy_cuda_layer_scale_bf16(x, w, out, rows, cols, stream) };
    check(code, KernelError::RowShape { rows, cols })
}

/// Launch the bf16 `out = cap · tanh(x / cap)` over `[rows, cols]` (Gemma's final-logit
/// soft-cap), in f32 with one bf16 rounding. Asynchronous.
///
/// # Safety
///
/// As [`scale_bf16`]'s.
pub unsafe fn soft_cap_bf16(
    x: *const c_void,
    out: *mut c_void,
    (rows, cols): (u32, u32),
    cap: f32,
    stream: *mut c_void,
) -> Result<(), KernelError> {
    // SAFETY: as `rms_norm_bf16`'s.
    let code = unsafe { scratchy_cuda_soft_cap_bf16(x, out, rows, cols, cap, stream) };
    check(code, KernelError::RowShape { rows, cols })
}

/// Launch the bf16 partial rotary embedding over `x` and `out` `[rows, cols]`, `cols = heads ·
/// head_dim`: per head, the first `rot_dim` lanes rotated in NeoX pairing within that width
/// (`(d, d + rot_dim/2)`) by `table`'s row for the token, `[cos(0..R/2) | sin(0..R/2)]`, in f32
/// with one bf16 rounding per output; the lanes `rot_dim..head_dim` copied bit for bit.
/// Asynchronous.
///
/// Refused with [`KernelError::RopeShape`], before any pointer is touched, unless `rows`, `cols`
/// and `head_dim` are non-zero, `rot_dim` is 64 or 256 (the half-rotated entries of
/// `step::ROPE_BF16_KERNEL_SHAPES`, the widths the kernel is instantiated for), `head_dim` a multiple of 64
/// (`step::ROPE_BF16_KERNEL_HEAD_MULTIPLE`) and of `rot_dim`, `cols` a multiple of `head_dim`, and
/// `rows · cols` at most `i32::MAX`. A failed launch returns the launch's own `cudaError_t`.
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

/// [`rope_bf16`] in either pairing: `HalfRotated` is [`rope_bf16`] itself, the same launch; in
/// `HalfHead` (Gemma 4's proportional rope) lane `d < rot_dim/2` pairs with lane `d + head_dim/2`,
/// and every other lane is copied bit for bit. Asynchronous.
///
/// Refused with [`KernelError::RopeShape`] as [`rope_bf16`], except that `(rot_dim, pairing)` must
/// be an entry of `step::ROPE_BF16_KERNEL_SHAPES`, and a `HalfHead` `head_dim` must be 512
/// (`step::ROPE_BF16_HALF_HEAD_DIM`).
///
/// # Safety
///
/// As [`rope_bf16`]'s.
pub unsafe fn rope_paired_bf16(
    x: *const c_void,
    table: *const c_void,
    out: *mut c_void,
    (rows, cols): (u32, u32),
    (head_dim, rot_dim, pairing): (u32, u32, RopePairing),
    stream: *mut c_void,
) -> Result<(), KernelError> {
    // The launcher's pairing codes, `kernels/rope_bf16.cu`'s `PAIRING_*`.
    let code = match pairing {
        RopePairing::HalfRotated => 0,
        RopePairing::HalfHead => 1,
    };
    // SAFETY: as `rms_norm_bf16`'s.
    let code = unsafe {
        scratchy_cuda_rope_paired_bf16(x, table, out, rows, cols, head_dim, rot_dim, code, stream)
    };
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

/// Launch the bf16 shared-expert gate scale `out = routed + shared · (1 / (1 + e^(−g)))` over
/// `[rows, cols]`, `g` `[rows, 1]` broadcast across its row, in f32 with one bf16 rounding per
/// element (`kernels/gate_scale_bf16.cu`). Asynchronous; the shape rule is the row kernels'
/// (`step::GATE_SCALE_BF16_COLS`).
///
/// # Safety
///
/// [`prepare_all`] must have succeeded on the current device. `routed`, `shared` and `out` must
/// be 16-byte-aligned device pointers to at least `rows·cols` bf16 elements each and `g` to `rows`,
/// valid for the whole kernel, with `out` overlapping no input. `stream` must be a live
/// `cudaStream_t` of the current device, or null for the default stream.
pub unsafe fn gate_scale_bf16(
    (routed, shared, g): (*const c_void, *const c_void, *const c_void),
    out: *mut c_void,
    (rows, cols): (u32, u32),
    stream: *mut c_void,
) -> Result<(), KernelError> {
    // SAFETY: as `rms_norm_bf16`'s.
    let code = unsafe { scratchy_cuda_gate_scale_bf16(routed, shared, g, out, rows, cols, stream) };
    check(code, KernelError::RowShape { rows, cols })
}

/// Launch the paged KV cache's write: for each token row `t` whose `slot_mapping[t]` is not
/// `u32::MAX`, row `t` of `k` and of `v` (`[rows, kv_heads · head_dim]`) copied bit for bit into
/// `k_cache` and `v_cache` (`[num_blocks, kv_heads, 64, head_dim]`) at that slot. A slot whose
/// block is at or past `num_blocks` writes nothing. Asynchronous.
///
/// Refused with [`KernelError::KvAppendShape`], before any pointer is touched, unless `rows` and
/// `kv_heads` are non-zero, `head_dim` is 256 or 512 (`step::KV_APPEND_BF16_HEAD_DIMS`, the head
/// dims the kernel is instantiated for), `block_size` 64 (`step::CUDA_KV_BLOCK_SIZE`), and
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
/// by `scale`. Asynchronous. The full causal attention: [`attn_prefill_windowed_bf16`] at a
/// window of 0, with its contract.
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
    // SAFETY: the caller upholds the contract above, which is the windowed launch's.
    unsafe {
        attn_prefill_windowed_bf16(
            (q, out),
            (k_cache, v_cache),
            (block_table, sequence),
            (rows, q_heads, kv_heads, head_dim),
            (block_size, num_blocks),
            (scale, 0),
            stream,
        )
    }
}

/// [`attn_prefill_bf16`] under a sliding window: query row `i` also drops the keys `j ≤ P + i −
/// window`, so it sees its own key and the `window − 1` before it (Gemma 4's sliding layers,
/// `window` 1024); `window` 0 is the full causal attention, the same launch. The KV tiles wholly
/// below every row's window in a CTA are never read, so the blocks there may hold anything.
/// Asynchronous.
///
/// Refused with [`KernelError::AttnShape`] as [`attn_prefill_bf16`], and also unless `window` is
/// a multiple of 64 (`step::ATTN_WINDOW_MULTIPLE`).
///
/// # Safety
///
/// As [`attn_prefill_bf16`]'s, but `block_table` need only name valid blocks from the window's
/// first tile on: an entry below it is never read.
pub unsafe fn attn_prefill_windowed_bf16(
    (q, out): (*const c_void, *mut c_void),
    (k_cache, v_cache): (*const c_void, *const c_void),
    (block_table, sequence): (*const c_void, *const c_void),
    (rows, q_heads, kv_heads, head_dim): (u32, u32, u32, u32),
    (block_size, num_blocks): (u32, u32),
    (scale, window): (f32, u32),
    stream: *mut c_void,
) -> Result<(), KernelError> {
    // SAFETY: the caller upholds the pointer and stream contract above; the launcher checks the
    // shape, scale and window before it touches a pointer and catches every C++ exception.
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
            window,
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

/// [`attn_prefill_windowed_bf16`] at head dim 512 (Gemma 4's global layers,
/// `kernels/attn_prefill_d512_bf16.cu`): `q` and `out` are `[rows, q_heads · 512]`, the caches
/// `[num_blocks, kv_heads, 64, 512]`. Each of a CTA's two consumer warpgroups keeps one
/// 256-column half of the output over the same 64 query rows; the masks and the online softmax are
/// the head-dim-256 kernel's. Asynchronous.
///
/// Refused with [`KernelError::AttnShape`] as [`attn_prefill_windowed_bf16`], except that
/// `head_dim` must be 512 (`step::ATTN_D512_BF16_HEAD_DIM`).
///
/// # Safety
///
/// As [`attn_prefill_windowed_bf16`]'s, with 512 for 256 in every size.
pub unsafe fn attn_prefill_d512_bf16(
    (q, out): (*const c_void, *mut c_void),
    (k_cache, v_cache): (*const c_void, *const c_void),
    (block_table, sequence): (*const c_void, *const c_void),
    (rows, q_heads, kv_heads, head_dim): (u32, u32, u32, u32),
    (block_size, num_blocks): (u32, u32),
    (scale, window): (f32, u32),
    stream: *mut c_void,
) -> Result<(), KernelError> {
    // SAFETY: as `attn_prefill_windowed_bf16`'s.
    let code = unsafe {
        scratchy_cuda_attn_prefill_d512_bf16(
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
            window,
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
/// `n = 1` (`step::GEMM_NARROW_BF16_ONE`, a shared expert's gate) runs a ThunderKittens
/// warp-vector kernel instead, one warp per row in a fixed order, its one bf16 per row rounded
/// once.
///
/// Refused with [`KernelError::NarrowShape`], before any pointer is touched, unless `m` is
/// non-zero, `n` 1 or a non-zero multiple of 8 and at most 64 (`step::GEMM_NARROW_BF16_MULTIPLE`,
/// `GEMM_NARROW_BF16_MAX_N`), `k` a non-zero multiple of 8 (of 256 when `n` is 1,
/// `step::GEMM_NARROW_BF16_ONE_K_MULTIPLE`), and `m` and `k` at most `i32::MAX`.
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
/// their V rows zeroed), so no unwritten slot reaches the output. Asynchronous. Every key:
/// [`attn_decode_windowed_bf16`] at a window of 0, with its contract.
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
    // SAFETY: the caller upholds the contract above, which is the windowed launch's.
    unsafe {
        attn_decode_windowed_bf16(
            (q, out),
            (k_cache, v_cache),
            (block_table, sequence),
            (q_heads, kv_heads, head_dim),
            (block_size, num_blocks),
            (scale, 0),
            stream,
        )
    }
}

/// [`attn_decode_bf16`] under a sliding window: the one query, at `P + L − 1`, sees only the keys
/// from `max(0, P + L − window)` on (Gemma 4's sliding layers, `window` 1024); `window` 0 is every
/// key, the same launch. The KV tiles wholly below the window are never read, so the blocks there
/// may hold anything. Asynchronous.
///
/// Refused with [`KernelError::AttnDecodeShape`] as [`attn_decode_bf16`], and also unless
/// `window` is a multiple of 64 (`step::ATTN_WINDOW_MULTIPLE`).
///
/// # Safety
///
/// As [`attn_decode_bf16`]'s, but `block_table` need only name valid blocks from the window's
/// first tile on: an entry below it is never read.
pub unsafe fn attn_decode_windowed_bf16(
    (q, out): (*const c_void, *mut c_void),
    (k_cache, v_cache): (*const c_void, *const c_void),
    (block_table, sequence): (*const c_void, *const c_void),
    (q_heads, kv_heads, head_dim): (u32, u32, u32),
    (block_size, num_blocks): (u32, u32),
    (scale, window): (f32, u32),
    stream: *mut c_void,
) -> Result<(), KernelError> {
    // SAFETY: the caller upholds the pointer and stream contract above; the launcher checks the
    // shape, scale and window before it touches a pointer.
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
            window,
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

/// [`attn_decode_windowed_bf16`] at head dim 512 (Gemma 4's global layers,
/// `kernels/attn_decode_d512_bf16.cu`): `q` and `out` are `[1, q_heads · 512]`, the caches
/// `[num_blocks, kv_heads, 64, 512]`. Both consumer warpgroups walk every KV tile in ascending
/// order, each keeping one 256-column half of the output, so there is no merge. Asynchronous.
///
/// Refused with [`KernelError::AttnDecodeShape`] as [`attn_decode_windowed_bf16`], except that
/// `head_dim` must be 512 (`step::ATTN_D512_BF16_HEAD_DIM`).
///
/// # Safety
///
/// As [`attn_decode_windowed_bf16`]'s, with 512 for 256 in every size.
pub unsafe fn attn_decode_d512_bf16(
    (q, out): (*const c_void, *mut c_void),
    (k_cache, v_cache): (*const c_void, *const c_void),
    (block_table, sequence): (*const c_void, *const c_void),
    (q_heads, kv_heads, head_dim): (u32, u32, u32),
    (block_size, num_blocks): (u32, u32),
    (scale, window): (f32, u32),
    stream: *mut c_void,
) -> Result<(), KernelError> {
    // SAFETY: as `attn_decode_windowed_bf16`'s.
    let code = unsafe {
        scratchy_cuda_attn_decode_d512_bf16(
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
            window,
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
// to continue from its slot) and touching only the `q_len` real rows. Each is a ThunderKittens
// kernel with under 48 KiB of static shared memory, so none is in `PREPARE`.

/// Launch the GDN causal depthwise conv1d, then SiLU: `conv_out[t, c]` (f32) for the real rows of
/// `x [rows, conv_dim]` (bf16) under the taps `w [conv_dim, kernel]` (f32), the rows before 0 the
/// slot's conv ring (zero, never read, on a fresh start); then the ring updated to the last
/// `kernel − 1` inputs. One warp per 128 channels and 64 tokens, the channels a ThunderKittens
/// warp vector, each tap one `fmaf` in ascending order (`kernels/gdn_conv1d.cu`). Asynchronous.
///
/// Refused with [`KernelError::GdnConvShape`], before any pointer is touched, unless `rows` and
/// `conv_dim` are non-zero, `conv_dim` a multiple of 128 (`step::GDN_CONV_CHANNEL_GROUP`),
/// `kernel` in `step::GDN_CONV_KERNEL_RANGE`, `rows · conv_dim` at most `i32::MAX`, and
/// `slot_elems` exactly `conv_dim · (kernel − 1)`.
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
/// · softplus(a + dt_bias)` and `beta = sigmoid(b)`, f32 `[rows, nv]` each. One warp per real
/// row, the row a ThunderKittens warp vector (`kernels/gdn_gating.cu`). Asynchronous.
///
/// Refused with [`KernelError::GdnGatingShape`], before any pointer is touched, unless `rows` and
/// `nv` are non-zero, `nv` 16 or 32 (a multiple of `step::GDN_GATING_NV_MULTIPLE`, the widths the
/// kernel is instantiated for), and `rows · nv` at most `i32::MAX`.
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
/// `out` written `+0.0` without reading `o` or `z`. One warp per row, the row a ThunderKittens
/// warp vector, its sum of squares by `warp::mul` then `warp::sum` (`kernels/gdn_rms_norm_gated.cu`).
/// Asynchronous.
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

// THE MOE KERNELS (#857): the qwen3-5-moe block's steps. Each is a ThunderKittens kernel; a shape
// it does not take is refused with [`KernelError::MoeShape`] naming it.

/// Launch the MoE routing (`kernels/moe_route_bf16.cu`): per token row of `logits` (`[m, 256]`
/// bf16), `softmax` in f32, its `top_k` largest picked (equal probabilities by the higher expert
/// index, a NaN lowest; always `top_k` distinct indices in `0..256`), stored ascending (slot
/// `top_k − 1` the best) as i32 into `topk_idx` and their probabilities, divided by their sum when
/// `renorm`, as f32 into `scores` (`[m, top_k]` each). Asynchronous.
///
/// Refused with [`KernelError::MoeShape`] `{ kernel: "moe_route_bf16", dims: [m, experts, top_k,
/// 0] }`, before any pointer is touched, unless `m` is non-zero and at most `i32::MAX`, `experts`
/// is 256 (`step::MOE_ROUTE_EXPERTS`) and `top_k` in `1..=32` (`step::MOE_ROUTE_MAX_TOP_K`).
///
/// # Safety
///
/// `logits` must be a device pointer to `m · 256` bf16, `topk_idx` to `m · top_k` i32 and `scores`
/// to `m · top_k` f32, 16-byte aligned and valid for the whole kernel, the outputs overlapping
/// nothing. `stream` must be a live `cudaStream_t` of the current device, or null for the default
/// stream.
pub unsafe fn moe_route_bf16(
    logits: *const c_void,
    (topk_idx, scores): (*mut c_void, *mut c_void),
    (m, experts, top_k): (u32, u32, u32),
    renorm: bool,
    stream: *mut c_void,
) -> Result<(), KernelError> {
    // SAFETY: as `rms_norm_bf16`'s; the Qwen program reads no per-expert scale.
    let code = unsafe {
        scratchy_cuda_moe_route_bf16(
            logits,
            std::ptr::null(),
            topk_idx,
            scores,
            m,
            experts,
            top_k,
            u32::from(renorm),
            ROUTE_QWEN,
            0.0,
            stream,
        )
    };
    check(
        code,
        KernelError::MoeShape {
            kernel: "moe_route_bf16",
            dims: [m, experts, top_k, ROUTE_QWEN],
        },
    )
}

/// `kernels/moe_route_bf16.cu`'s `program`: the Qwen routing.
const ROUTE_QWEN: u32 = 0;
/// `kernels/moe_route_bf16.cu`'s `program`: the Gemma routing.
const ROUTE_GEMMA: u32 = 1;

/// Launch Gemma's MoE routing (`kernels/moe_route_bf16.cu`'s Gemma program): per token row of
/// `logits` (`[m, experts]` bf16), the `top_k` largest RAW logits picked (equal logits by the
/// higher expert index, a NaN lowest; always `top_k` distinct indices in `0..experts`), each
/// scaled by `scale` (`hidden^-0.5`), a softmax over the picks in f32, then each times its
/// expert's `expert_scale` entry; stored ascending (slot `top_k − 1` the best) as i32 into
/// `topk_idx` and f32 into `scores` (`[m, top_k]` each). Asynchronous.
///
/// Refused with [`KernelError::MoeShape`] `{ kernel: "moe_route_bf16", dims: [m, experts, top_k,
/// 1] }`, before any pointer is touched, unless `m` is non-zero and at most `i32::MAX`, `experts`
/// one of `op_abi::MOE_EXPERT_COUNTS` (128, 256), `top_k` in `1..=32`
/// (`step::MOE_ROUTE_MAX_TOP_K`), and `expert_scale` non-null.
///
/// # Safety
///
/// `logits` must be a device pointer to `m · experts` bf16, `expert_scale` to `experts` bf16,
/// `topk_idx` to `m · top_k` i32 and `scores` to `m · top_k` f32, 16-byte aligned and valid for
/// the whole kernel, the outputs overlapping nothing. `stream` must be a live `cudaStream_t` of the
/// current device, or null for the default stream.
pub unsafe fn gemma_route_bf16(
    (logits, expert_scale): (*const c_void, *const c_void),
    (topk_idx, scores): (*mut c_void, *mut c_void),
    (m, experts, top_k): (u32, u32, u32),
    scale: f32,
    stream: *mut c_void,
) -> Result<(), KernelError> {
    // SAFETY: as `rms_norm_bf16`'s.
    let code = unsafe {
        scratchy_cuda_moe_route_bf16(
            logits,
            expert_scale,
            topk_idx,
            scores,
            m,
            experts,
            top_k,
            0,
            ROUTE_GEMMA,
            scale,
            stream,
        )
    };
    check(
        code,
        KernelError::MoeShape {
            kernel: "moe_route_bf16",
            dims: [m, experts, top_k, ROUTE_GEMMA],
        },
    )
}

/// Launch the MoE combine (`kernels/moe_combine_bf16.cu`): `out[t, :] = Σ_j scores[t, j] ·
/// down[row(t, j), :]` over the `top_k` slots in order, in f32, rounded once to bf16, every token
/// row written; `row(t, j)` is `pos[t · top_k + j]` when `pos` is given (sorted pairs), else
/// `t · top_k + j` (gathered). Asynchronous.
///
/// Refused with [`KernelError::MoeShape`] `{ kernel: "moe_combine_bf16", dims: [m, top_k, hidden,
/// down_rows] }`, before any pointer is touched, unless `m` is non-zero, `top_k` in `1..=32`,
/// `hidden` a non-zero multiple of 256 (`step::MOE_COMBINE_BF16_CHUNK`), `down_rows` at least
/// `m · top_k`, and `m · top_k` and `down_rows · hidden` at most `i32::MAX`.
///
/// # Safety
///
/// `down` must be a device pointer to `down_rows · hidden` bf16, `pos` (when given) to
/// `m · top_k` u32 each below `down_rows`, `scores` to `m · top_k` f32 and `out` to `m · hidden`
/// bf16, 16-byte aligned and valid for the whole kernel, `out` overlapping no input. `stream`
/// must be a live `cudaStream_t` of the current device, or null for the default stream.
pub unsafe fn moe_combine_bf16(
    (down, pos, scores): (*const c_void, Option<*const c_void>, *const c_void),
    out: *mut c_void,
    (m, top_k, hidden, down_rows): (u32, u32, u32, u32),
    stream: *mut c_void,
) -> Result<(), KernelError> {
    let (pos, sorted) = match pos {
        Some(p) => (p, 1),
        None => (std::ptr::null(), 0),
    };
    // SAFETY: as `rms_norm_bf16`'s; a null `pos` is never read (`sorted` 0).
    let code = unsafe {
        scratchy_cuda_moe_combine_bf16(
            down, pos, scores, out, m, top_k, hidden, down_rows, sorted, stream,
        )
    };
    check(
        code,
        KernelError::MoeShape {
            kernel: "moe_combine_bf16",
            dims: [m, top_k, hidden, down_rows],
        },
    )
}

/// The sort's index regions, each a device pointer into the MoE scratch ([`crate::step::SortLayout`]):
/// `chunk_hist`, `counts`, `offsets`, `tile_expert` and `pos`.
#[derive(Clone, Copy, Debug)]
pub struct SortRegions {
    pub chunk_hist: *mut c_void,
    pub counts: *mut c_void,
    pub offsets: *mut c_void,
    pub tile_expert: *mut c_void,
    pub pos: *mut c_void,
}

/// Launch the MoE sort (`kernels/moe_sort.cu`, three stream-ordered launches): the `m · top_k`
/// pairs, pair `p` naming expert `topk_idx[p]`, sorted by expert stably, each expert's run padded
/// to whole 128-row tiles. Writes `rows[pos[p], :] = x[p / top_k, :]` (bit-exact) and `+0.0` on
/// every padding row of every live tile, and the index regions: `counts`, `offsets` (the padded
/// exclusive scan, `offsets[experts]` the total), `tile_expert` for each live tile, `pos`, and
/// `chunk_hist` as scratch. No row outside a live tile is written. Asynchronous.
///
/// Refused with [`KernelError::MoeShape`] `{ kernel: "moe_sort", dims: [m, top_k, hidden,
/// max_tiles] }`, before any pointer is touched, unless `m` is non-zero, `experts` one of
/// `op_abi::MOE_EXPERT_COUNTS` (128, 256), `top_k` in `1..=32`, `hidden` a non-zero multiple of 8, `m ·
/// top_k` and `m · hidden` at most `i32::MAX`, and `max_tiles` at least `tape::moe_tiles`' bound
/// with `max_tiles · 128 · hidden` at most `i32::MAX`. A failed launch returns the first failing
/// launch's own `cudaError_t`, the ones after it not launched. No one-time setup, so it is not in
/// [`PREPARE`].
///
/// # Safety
///
/// `x` must be a device pointer to `m · hidden` bf16, `topk_idx` to `m · top_k` i32 each in
/// `0..experts`, `rows` to `max_tiles · 128 · hidden` bf16, and `regions` to the sort's regions of
/// [`crate::step::SortLayout`] for this geometry, all 16-byte aligned and valid for the whole
/// kernel, no output overlapping another or an input. `stream` must be a live `cudaStream_t` of
/// the current device, or null for the default stream.
pub unsafe fn moe_sort(
    (x, topk_idx): (*const c_void, *const c_void),
    regions: SortRegions,
    rows: *mut c_void,
    (m, experts, top_k, hidden, max_tiles): (u32, u32, u32, u32, u32),
    stream: *mut c_void,
) -> Result<(), KernelError> {
    // SAFETY: as `rms_norm_bf16`'s.
    let code = unsafe {
        scratchy_cuda_moe_sort(
            x,
            topk_idx,
            regions.chunk_hist,
            regions.counts,
            regions.offsets,
            regions.tile_expert,
            regions.pos,
            rows,
            m,
            experts,
            top_k,
            hidden,
            max_tiles,
            stream,
        )
    };
    check(
        code,
        KernelError::MoeShape {
            kernel: "moe_sort",
            dims: [m, top_k, hidden, max_tiles],
        },
    )
}

/// Launch the grouped expert GEMM (`kernels/expert_gemm_bf16.cu`) on 256-column blocks:
/// [`expert_gemm_tiled_bf16`] at `n_tile` 256, the qwen3-5-moe tapes' one tile.
///
/// # Safety
///
/// As [`expert_gemm_tiled_bf16`]'s.
pub unsafe fn expert_gemm_bf16(
    (a, b): (*const c_void, *const c_void),
    (offsets, tile_expert): (*const c_void, *const c_void),
    c: *mut c_void,
    (max_tiles, n, k, experts, expert_rows): (u32, u32, u32, u32, u32),
    stream: *mut c_void,
) -> Result<(), KernelError> {
    // SAFETY: the caller upholds `expert_gemm_tiled_bf16`'s contract.
    unsafe {
        expert_gemm_tiled_bf16(
            (a, b),
            (offsets, tile_expert),
            c,
            (max_tiles, n, k, experts, expert_rows),
            EXPERT_GEMM_WIDE_TILE,
            stream,
        )
    }
}

/// [`crate::step::EXPERT_GEMM_N_TILES`]' wide tile, every Qwen tape's.
const EXPERT_GEMM_WIDE_TILE: u32 = crate::step::EXPERT_GEMM_N_TILES[1];

/// Launch the grouped expert GEMM (`kernels/expert_gemm_bf16.cu`): `c[128t .. 128t + 128, :] =
/// a[128t .. 128t + 128, :] · B_eᵀ` for every live tile `t < offsets[experts] / 128` of the
/// sorted rows, `e = tile_expert[t]`, `B_e` the first `n` rows of expert `e`'s `[expert_rows, k]`
/// block at `b`, in column blocks of `n_tile`. A tile past the live ones is never read or written.
/// Every live tile's rows are bit-identical to [`gemm_bf16_abt`] (`n_tile` 256) or
/// [`gemm_col64_bf16_abt`] (`n_tile` 64) on the same rows and expert slice. Asynchronous.
///
/// Refused with [`KernelError::MoeShape`] `{ kernel: "expert_gemm_bf16", dims: [max_tiles, n, k,
/// expert_rows] }`, before any pointer is touched, unless `max_tiles` is non-zero, `(experts,
/// n_tile)` one of (256, 256), (128, 256) and (128, 64) (`op_abi::MOE_EXPERT_COUNTS`,
/// `step::EXPERT_GEMM_N_TILES`), `n` a non-zero multiple of `n_tile` and `k` of 64, `expert_rows`
/// at least `n`, and `max_tiles · 128 · max(n, k)` and `experts · expert_rows · k` at most
/// `i32::MAX`. A failed launch returns the launch's own `cudaError_t`.
///
/// # Safety
///
/// `a` must be a device pointer to `max_tiles · 128 · k` bf16, `b` to `experts · expert_rows · k`
/// bf16, `offsets` and `tile_expert` to the sort's regions (`experts + 1` and `max_tiles` u32,
/// the live tiles each naming an expert below `experts`), and `c` to `max_tiles · 128 · n` bf16,
/// all 16-byte aligned and valid for the whole kernel, `c` overlapping no input. The kernel's
/// setup in [`PREPARE`] must have run on the current device. `stream` must be a live
/// `cudaStream_t` of the current device, or null for the default stream.
pub unsafe fn expert_gemm_tiled_bf16(
    (a, b): (*const c_void, *const c_void),
    (offsets, tile_expert): (*const c_void, *const c_void),
    c: *mut c_void,
    (max_tiles, n, k, experts, expert_rows): (u32, u32, u32, u32, u32),
    n_tile: u32,
    stream: *mut c_void,
) -> Result<(), KernelError> {
    // SAFETY: as `gemm_bf16_abt`'s.
    let code = unsafe {
        scratchy_cuda_expert_gemm_bf16(
            a,
            b,
            offsets,
            tile_expert,
            c,
            max_tiles,
            n,
            k,
            experts,
            expert_rows,
            n_tile,
            stream,
        )
    };
    check(
        code,
        KernelError::MoeShape {
            kernel: "expert_gemm_bf16",
            dims: [max_tiles, n, k, expert_rows],
        },
    )
}

/// Launch the gathered expert GEMV (`kernels/expert_gemv_bf16.cu`): `c[p, :] = a[row(p), :] ·
/// B_eᵀ` for each of the `m · top_k` pairs, `e = topk_idx[p]`, `B_e` the first `n` rows of expert
/// `e`'s `[expert_rows, k]` block at `b`; `row(p)` is `p` when `pair_rows` (the down), else the
/// pair's token row `p / top_k` (the gate and up). Each pair's row is bit-identical to
/// [`gemv_bf16_abt`] on the same row and expert slice. Asynchronous.
///
/// Refused with [`KernelError::MoeShape`] `{ kernel: "expert_gemv_bf16", dims: [m, top_k, n, k]
/// }`, before any pointer is touched, unless `m` is non-zero, `top_k` in `1..=32`, `m · top_k` at
/// most 65,535, `n` and `k` non-zero multiples of 64, `experts` one of
/// `op_abi::MOE_EXPERT_COUNTS` (128, 256), `expert_rows` at least `n`,
/// and the operands' element counts at most `i32::MAX`. A failed launch returns the launch's own
/// `cudaError_t`.
///
/// # Safety
///
/// `a` must be a device pointer to `a_rows · k` bf16 (`a_rows` `m · top_k` when `pair_rows`, else
/// `m`), `b` to `experts · expert_rows · k` bf16, `topk_idx` to `m · top_k` i32 each in
/// `0..experts`
/// and `c` to `m · top_k · n` bf16, all 16-byte aligned and valid for the whole kernel, `c`
/// overlapping no input. The kernel's setup in [`PREPARE`] must have run on the current device.
/// `stream` must be a live `cudaStream_t` of the current device, or null for the default stream.
pub unsafe fn expert_gemv_bf16(
    (a, b, topk_idx): (*const c_void, *const c_void, *const c_void),
    c: *mut c_void,
    (m, top_k, n, k): (u32, u32, u32, u32),
    (experts, expert_rows): (u32, u32),
    pair_rows: bool,
    stream: *mut c_void,
) -> Result<(), KernelError> {
    // SAFETY: as `gemv_bf16_abt`'s.
    let code = unsafe {
        scratchy_cuda_expert_gemv_bf16(
            a,
            b,
            topk_idx,
            c,
            m,
            top_k,
            n,
            k,
            experts,
            expert_rows,
            u32::from(pair_rows),
            stream,
        )
    };
    check(
        code,
        KernelError::MoeShape {
            kernel: "expert_gemv_bf16",
            dims: [m, top_k, n, k],
        },
    )
}
