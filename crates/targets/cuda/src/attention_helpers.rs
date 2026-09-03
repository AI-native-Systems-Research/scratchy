// SPDX-License-Identifier: Apache-2.0
//! Shared attention helpers for FP8 KV cache support.
//!
//! All model attention modules call these instead of directly invoking
//! reshape_and_cache/flash_attn to get FP8 quantize-on-write and
//! dequant-on-read behavior transparently.
//!
//! During CUDA graph capture/replay, the [`Fp8GraphCtx`] thread-local provides
//! pre-allocated dequant buffers and cached scales, eliminating the D2H syncs
//! and variable-size allocations that would otherwise break graph capture.

use std::cell::Cell;

use crate::alloc::{CachingAllocator, OwnedTensor};
use crate::dtype::DType;
use crate::flashinfer::{self, FlashInferConfig, FlashInferPlanCache};
use crate::kernels;
use crate::kv_cache::KvCachePool;
use crate::tensor::{GpuTensor, TensorView};

type CUstream = cudarc::driver::sys::CUstream;

/// Gather K or V for ONE layer from the paged cache into a contiguous
/// `[total_tokens, kv_heads, head_dim]` tensor, for the naive-SDPA decode path
/// (Gemma-4 GLOBAL attention, head_dim 512 > the flash cap). Block layout is
/// `[num_blocks, block_size, kv_heads, head_dim]`; the per-layer pool gives the
/// global layer its own `kv_heads*head_dim` stride, so passing the global dims
/// here reads the right bytes. bs=1 (single request) — `block_table` is `[1, nb]`.
///
/// # Safety
/// Valid CUDA context + stream; `block_table` holds the request's block ids.
#[allow(clippy::too_many_arguments)]
pub unsafe fn gather_global_kv_contiguous(
    kv_cache: &KvCachePool,
    block_table: GpuTensor,
    layer: usize,
    is_key: bool,
    total_tokens: usize,
    kv_heads: usize,
    head_dim: usize,
    alloc: &mut CachingAllocator,
    stream: CUstream,
) -> OwnedTensor {
    let cache = if is_key {
        kv_cache.k_cache_raw(layer)
    } else {
        kv_cache.v_cache_raw(layer)
    };
    let out = alloc.alloc_tensor(&[total_tokens, kv_heads, head_dim], cache.dtype());
    let bs = kv_cache.block_size_for(layer);

    if cache.dtype() == DType::BF16 {
        // On-device gather (single-sequence): reads block ids from GPU, no host
        // readback → CUDA-graph-capture-safe. This is the default gemma4 path
        // (bf16 KV). Over-sized grid is bounded inside the kernel.
        unsafe {
            kernels::gather_pages_bf16_paged(
                cache,
                block_table,
                total_tokens,
                kv_heads,
                head_dim,
                bs,
                out.as_gpu_tensor(),
                stream,
            );
        }
        return out;
    }

    // FP8 / F16 fallback: host-driven D2H(block ids) + per-block D2D copy. This
    // does a blocking stream_synchronize, ILLEGAL inside CUDA graph capture — but
    // fp8 KV skips graph capture entirely (variable scratch sizes), so this path
    // is never reached under capture. Guard it so any future capture path that
    // DID reach it fails loudly at the site instead of poisoning the context.
    assert!(
        !unsafe { is_stream_capturing(stream) },
        "gather_global_kv_contiguous (fp8/f16 fallback): blocking D2H of block_table \
         is incompatible with CUDA graph capture. The bf16 path uses an on-device \
         gather; add an on-device fp8 gather for a capturable fp8 global-attn path."
    );
    let nbt = block_table.dim(block_table.ndim() - 1);
    let mut block_ids = vec![0i32; nbt];
    crate::driver::memcpy_dtoh_async(
        block_ids.as_mut_ptr() as *mut u8,
        block_table.raw_ptr() as *const u8,
        nbt * 4,
        stream,
    )
    .expect("D2H block_table (global gather)");
    crate::driver::stream_synchronize(stream).expect("sync block_table D2H (global gather)");

    let elem = cache.dtype().size_bytes();
    let row_bytes = kv_heads * head_dim * elem;
    let block_stride = bs * row_bytes;
    let mut rem = total_tokens;
    let mut dst_off = 0usize;
    for &bid in &block_ids {
        if rem == 0 {
            break;
        }
        let n = rem.min(bs);
        let src = (cache.raw_ptr() as *const u8).add(bid as usize * block_stride);
        let dst = (out.raw_ptr() as *mut u8).add(dst_off);
        crate::driver::memcpy_dtod_async(dst, src, n * row_bytes, stream)
            .expect("D2D gather block (global)");
        dst_off += n * row_bytes;
        rem -= n;
    }
    out
}

// ---------------------------------------------------------------------------
// FP8 CUDA graph context (thread-local)
// ---------------------------------------------------------------------------

/// Pre-allocated buffers and cached scales for FP8 decode during graph
/// capture/replay. Set via [`set_fp8_graph_ctx`] before capture, cleared after.
///
/// All pointer fields are persistent GPU allocations owned by [`CudaGraphRunner`].
#[derive(Clone, Copy)]
pub struct Fp8GraphCtx {
    /// `[max_total_kv, num_kv_heads, head_dim]` in model dtype (BF16/F16).
    pub k_buf: *mut u8,
    /// Same shape as `k_buf`.
    pub v_buf: *mut u8,
    /// `[max_batch + 1]` i32 — prefix-sum of per-sequence KV lengths.
    pub cu_seqlens_k: *mut u8,
    /// Grid launch size for dequant kernels (= buffer capacity in tokens).
    pub max_total_kv: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub output_dtype: DType,
    /// Host-side per-layer K scales (len = num_layers). Pointer into a Vec
    /// owned by CudaGraphRunner — valid for the duration of the graph ctx.
    pub k_scales: *const f32,
    /// Host-side per-layer V scales.
    pub v_scales: *const f32,
}

// Raw pointers are Send — the buffers are GPU-side and only accessed on the
// thread that owns the CUDA context.
unsafe impl Send for Fp8GraphCtx {}

thread_local! {
    static FP8_GRAPH_CTX: Cell<Option<Fp8GraphCtx>> = const { Cell::new(None) };
}

/// Set the FP8 graph context for the current thread. Call before graph capture.
pub fn set_fp8_graph_ctx(ctx: Fp8GraphCtx) {
    FP8_GRAPH_CTX.set(Some(ctx));
}

/// Clear the FP8 graph context. Call after graph capture completes.
pub fn clear_fp8_graph_ctx() {
    FP8_GRAPH_CTX.set(None);
}

/// Returns `true` if the FP8 graph context is active on this thread.
pub fn has_fp8_graph_ctx() -> bool {
    FP8_GRAPH_CTX.get().is_some()
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Write K/V into the paged cache, handling both BF16 and FP8 paths.
///
/// When the cache is FP8, quantizes BF16/F16 K/V using per-layer scales.
/// When BF16/F16, performs a direct copy (existing path).
#[allow(clippy::too_many_arguments)]
pub unsafe fn write_kv_cache(
    k: TensorView<'_>,
    v: TensorView<'_>,
    slot_mapping: TensorView<'_>,
    kv_cache: &KvCachePool,
    layer_idx: usize,
    stream: CUstream,
) {
    if kv_cache.is_fp8() {
        kernels::reshape_and_cache_fp8(
            *k,
            *v,
            *kv_cache.k_cache(layer_idx),
            *kv_cache.v_cache(layer_idx),
            *slot_mapping,
            kv_cache.k_scale_ptr(layer_idx),
            kv_cache.v_scale_ptr(layer_idx),
            kv_cache.block_size_for(layer_idx),
            stream,
        );
    } else {
        kernels::reshape_and_cache(
            *k,
            *v,
            *kv_cache.k_cache(layer_idx),
            *kv_cache.v_cache(layer_idx),
            *slot_mapping,
            kv_cache.block_size_for(layer_idx),
            stream,
        );
    }
}

/// Run attention: fresh prefill (contiguous) or decode (paged/dequant).
///
/// Standard path (no softcap, no sliding window). Used by LLaMA, CommandR,
/// DeepSeek, Qwen3Next.
///
/// When FP8 decode: dequantizes KV from cache pages into contiguous BF16
/// buffers, then runs contiguous FA2. Fresh prefill always uses BF16 K/V
/// from the QKV projection directly.
#[allow(clippy::too_many_arguments)]
/// `cos_sin_cache_ptr`: optional pointer to `[max_pos, rotary_dim]` cos/sin cache
///   for fused RoPE on cached K (spans). Pass `std::ptr::null()` when not using spans.
pub unsafe fn attention_standard(
    q: TensorView<'_>,
    k: TensorView<'_>,
    v: TensorView<'_>,
    cu_seqlens_q: TensorView<'_>,
    seqused_k: TensorView<'_>,
    block_table: TensorView<'_>,
    max_seqlen_q: usize,
    max_seqlen_k: usize,
    scale: f32,
    kv_cache: &KvCachePool,
    layer_idx: usize,
    num_sm: i32,
    alloc: &mut CachingAllocator,
    stream: CUstream,
    cos_sin_cache_ptr: *const u8,
    rotary_dim: usize,
    is_rotary_interleaved: bool,
) -> OwnedTensor {
    let fresh_prefill = max_seqlen_q > 1 && max_seqlen_q == max_seqlen_k;

    if fresh_prefill {
        // Fresh prefill: use BF16 K/V from QKV projection (no cache read).
        // Pass cos_sin_cache so FA2 can apply RoPE to contiguous K if needed.
        kernels::flash_attn_contiguous(
            *q,
            *k,
            *v,
            *cu_seqlens_q,
            *cu_seqlens_q,
            max_seqlen_q,
            max_seqlen_k,
            scale,
            true,
            0.0,
            -1,
            alloc,
            stream,
            cos_sin_cache_ptr,
            rotary_dim,
            is_rotary_interleaved,
        )
    } else if kv_cache.is_fp8() {
        // FP8 decode: dequant pages → contiguous FA2 with fused RoPE.
        fp8_decode_attention(
            *q,
            *cu_seqlens_q,
            *seqused_k,
            *block_table,
            max_seqlen_q,
            max_seqlen_k,
            scale,
            0.0,
            -1,
            kv_cache,
            layer_idx,
            alloc,
            stream,
            cos_sin_cache_ptr,
            rotary_dim,
            is_rotary_interleaved,
        )
    } else {
        // BF16 paged FA2 — with optional fused RoPE for spans.
        kernels::flash_attn_paged_ext(
            *q,
            *kv_cache.k_cache(layer_idx),
            *kv_cache.v_cache(layer_idx),
            *cu_seqlens_q,
            *seqused_k,
            *block_table,
            max_seqlen_q,
            max_seqlen_k,
            scale,
            true,
            0.0,
            -1,
            kv_cache.block_size_for(layer_idx),
            num_sm,
            alloc,
            stream,
            cos_sin_cache_ptr,
            rotary_dim,
            is_rotary_interleaved,
            kv_cache.block_unrotated_gpu(),
        )
    }
}

/// Decode-only attention from cache (K/V already written by fused kernel).
///
/// Handles both BF16 paged and FP8 dequant paths. Does NOT support prefill
/// (caller must ensure max_seqlen_q == 1).
#[allow(clippy::too_many_arguments)]
pub unsafe fn attention_decode_from_cache(
    q: TensorView<'_>,
    cu_seqlens_q: TensorView<'_>,
    seqused_k: TensorView<'_>,
    block_table: TensorView<'_>,
    max_seqlen_q: usize,
    max_seqlen_k: usize,
    scale: f32,
    softcap: f32,
    window_size_left: i32,
    kv_cache: &KvCachePool,
    layer_idx: usize,
    num_sm: i32,
    alloc: &mut CachingAllocator,
    stream: CUstream,
    cos_sin_cache_ptr: *const u8,
    rotary_dim: usize,
    is_rotary_interleaved: bool,
) -> OwnedTensor {
    if kv_cache.is_fp8() {
        fp8_decode_attention(
            *q,
            *cu_seqlens_q,
            *seqused_k,
            *block_table,
            max_seqlen_q,
            max_seqlen_k,
            scale,
            softcap,
            window_size_left,
            kv_cache,
            layer_idx,
            alloc,
            stream,
            cos_sin_cache_ptr,
            rotary_dim,
            is_rotary_interleaved,
        )
    } else {
        kernels::flash_attn_paged_ext(
            *q,
            *kv_cache.k_cache(layer_idx),
            *kv_cache.v_cache(layer_idx),
            *cu_seqlens_q,
            *seqused_k,
            *block_table,
            max_seqlen_q,
            max_seqlen_k,
            scale,
            true,
            softcap,
            window_size_left,
            kv_cache.block_size_for(layer_idx),
            num_sm,
            alloc,
            stream,
            cos_sin_cache_ptr,
            rotary_dim,
            is_rotary_interleaved,
            kv_cache.block_unrotated_gpu(),
        )
    }
}

/// Run attention with softcap and/or sliding window (Gemma2/Gemma3).
///
/// Uses `flash_attn_paged_ext` for the BF16 decode path.
#[allow(clippy::too_many_arguments)]
pub unsafe fn attention_ext(
    q: TensorView<'_>,
    k: TensorView<'_>,
    v: TensorView<'_>,
    cu_seqlens_q: TensorView<'_>,
    seqused_k: TensorView<'_>,
    block_table: TensorView<'_>,
    max_seqlen_q: usize,
    max_seqlen_k: usize,
    scale: f32,
    softcap: f32,
    window_size_left: i32,
    kv_cache: &KvCachePool,
    layer_idx: usize,
    num_sm: i32,
    alloc: &mut CachingAllocator,
    stream: CUstream,
    cos_sin_cache_ptr: *const u8,
    rotary_dim: usize,
    is_rotary_interleaved: bool,
) -> OwnedTensor {
    let fresh_prefill = max_seqlen_q > 1 && max_seqlen_q == max_seqlen_k;

    if fresh_prefill {
        kernels::flash_attn_contiguous(
            *q,
            *k,
            *v,
            *cu_seqlens_q,
            *cu_seqlens_q,
            max_seqlen_q,
            max_seqlen_k,
            scale,
            true,
            softcap,
            window_size_left,
            alloc,
            stream,
            cos_sin_cache_ptr,
            rotary_dim,
            is_rotary_interleaved,
        )
    } else if kv_cache.is_fp8() {
        fp8_decode_attention(
            *q,
            *cu_seqlens_q,
            *seqused_k,
            *block_table,
            max_seqlen_q,
            max_seqlen_k,
            scale,
            softcap,
            window_size_left,
            kv_cache,
            layer_idx,
            alloc,
            stream,
            cos_sin_cache_ptr,
            rotary_dim,
            is_rotary_interleaved,
        )
    } else {
        kernels::flash_attn_paged_ext(
            *q,
            *kv_cache.k_cache(layer_idx),
            *kv_cache.v_cache(layer_idx),
            *cu_seqlens_q,
            *seqused_k,
            *block_table,
            max_seqlen_q,
            max_seqlen_k,
            scale,
            true,
            softcap,
            window_size_left,
            kv_cache.block_size_for(layer_idx),
            num_sm,
            alloc,
            stream,
            cos_sin_cache_ptr,
            rotary_dim,
            is_rotary_interleaved,
            kv_cache.block_unrotated_gpu(),
        )
    }
}

// ---------------------------------------------------------------------------
// FP8 decode internals
// ---------------------------------------------------------------------------

/// FP8 decode path: dequant KV pages → contiguous BF16 → FA2.
///
/// Two modes:
/// 1. **Normal** (no graph ctx): D2H seqused_k + scales, allocate dequant buffers,
///    run dequant+FA2. Used during eager decode.
/// 2. **Graphed** (thread-local [`Fp8GraphCtx`] set): uses pre-allocated buffers
///    and cached scales — no D2H syncs, no dynamic allocations. Used during CUDA
///    graph capture.
#[allow(clippy::too_many_arguments)]
unsafe fn fp8_decode_attention(
    q: GpuTensor,
    cu_seqlens_q: GpuTensor,
    seqused_k: GpuTensor,
    block_table: GpuTensor,
    max_seqlen_q: usize,
    max_seqlen_k: usize,
    scale: f32,
    softcap: f32,
    window_size_left: i32,
    kv_cache: &KvCachePool,
    layer_idx: usize,
    alloc: &mut CachingAllocator,
    stream: CUstream,
    cos_sin_cache_ptr: *const u8,
    rotary_dim: usize,
    is_rotary_interleaved: bool,
) -> OwnedTensor {
    // Check for pre-allocated graph context.
    if let Some(ctx) = FP8_GRAPH_CTX.get() {
        return fp8_decode_attention_graphed(
            q,
            cu_seqlens_q,
            seqused_k,
            block_table,
            max_seqlen_q,
            max_seqlen_k,
            scale,
            softcap,
            window_size_left,
            kv_cache,
            layer_idx,
            ctx,
            alloc,
            stream,
            cos_sin_cache_ptr,
            rotary_dim,
            is_rotary_interleaved,
        );
    }

    // --- Normal (eager) path ---
    let batch_size = seqused_k.dim(0);
    let num_kv_heads = kv_cache.num_kv_heads;
    let head_dim = kv_cache.head_dim;
    let output_dtype = q.dtype(); // BF16 or F16

    // D2H seqused_k to build cu_seqlens_k on CPU.
    let mut seq_lens = vec![0i32; batch_size];
    crate::driver::memcpy_dtoh_async(
        seq_lens.as_mut_ptr() as *mut u8,
        seqused_k.raw_ptr() as *const u8,
        batch_size * 4,
        stream,
    )
    .expect("D2H seqused_k");
    crate::driver::stream_synchronize(stream).expect("sync seqused_k");

    // Build cu_seqlens_k prefix sum: [0, s0, s0+s1, ...]
    let mut cu_seqlens_k_host = vec![0i32; batch_size + 1];
    let mut total_kv_tokens: usize = 0;
    for i in 0..batch_size {
        total_kv_tokens += seq_lens[i] as usize;
        cu_seqlens_k_host[i + 1] = total_kv_tokens as i32;
    }

    if total_kv_tokens == 0 {
        // Edge case: no KV tokens (shouldn't happen in practice).
        return kernels::flash_attn_contiguous(
            q,
            q,
            q, // dummy, won't actually run
            cu_seqlens_q,
            cu_seqlens_q,
            max_seqlen_q,
            0,
            scale,
            true,
            softcap,
            window_size_left,
            alloc,
            stream,
            std::ptr::null(),
            0,
            false,
        );
    }

    // Upload cu_seqlens_k to GPU.
    let cu_seqlens_k_gpu = alloc.alloc_tensor(&[batch_size + 1], DType::I32);
    crate::driver::memcpy_htod_async(
        cu_seqlens_k_gpu.as_gpu_tensor().raw_ptr() as *mut u8,
        cu_seqlens_k_host.as_ptr() as *const u8,
        (batch_size + 1) * 4,
        stream,
    )
    .expect("H2D cu_seqlens_k");

    // Read K scale and V scale from GPU (single f32 each).
    let mut k_scale_host: f32 = 1.0;
    let mut v_scale_host: f32 = 1.0;
    crate::driver::memcpy_dtoh_async(
        &mut k_scale_host as *mut f32 as *mut u8,
        kv_cache.k_scale_ptr(layer_idx) as *const u8,
        4,
        stream,
    )
    .expect("D2H k_scale");
    crate::driver::memcpy_dtoh_async(
        &mut v_scale_host as *mut f32 as *mut u8,
        kv_cache.v_scale_ptr(layer_idx) as *const u8,
        4,
        stream,
    )
    .expect("D2H v_scale");
    crate::driver::stream_synchronize(stream).expect("sync scales");

    // Dequant+gather K and V from FP8 cache pages to contiguous BF16.
    let k_contiguous = kernels::dequant_gather_pages(
        *kv_cache.k_cache(layer_idx),
        block_table,
        cu_seqlens_k_gpu.as_gpu_tensor(),
        k_scale_host,
        total_kv_tokens,
        num_kv_heads,
        head_dim,
        kv_cache.block_size_for(layer_idx),
        output_dtype,
        alloc,
        stream,
    );
    let v_contiguous = kernels::dequant_gather_pages(
        *kv_cache.v_cache(layer_idx),
        block_table,
        cu_seqlens_k_gpu.as_gpu_tensor(),
        v_scale_host,
        total_kv_tokens,
        num_kv_heads,
        head_dim,
        kv_cache.block_size_for(layer_idx),
        output_dtype,
        alloc,
        stream,
    );

    // Contiguous FA2 on the dequantized BF16 K/V.
    kernels::flash_attn_contiguous(
        q,
        k_contiguous.as_gpu_tensor(),
        v_contiguous.as_gpu_tensor(),
        cu_seqlens_q,
        cu_seqlens_k_gpu.as_gpu_tensor(),
        max_seqlen_q,
        max_seqlen_k,
        scale,
        true,
        softcap,
        window_size_left,
        alloc,
        stream,
        cos_sin_cache_ptr,
        rotary_dim,
        is_rotary_interleaved,
    )
}

/// FP8 decode path for CUDA graph capture: uses pre-allocated buffers and cached
/// scales. No D2H syncs, no dynamic allocations — all addresses are fixed.
///
/// The dequant kernel is launched with `ctx.max_total_kv` grid blocks. Blocks
/// beyond the actual total (determined by `cu_seqlens_k`) exit early via the
/// kernel's bounds check (`seq_idx >= batch_size`).
#[allow(clippy::too_many_arguments)]
unsafe fn fp8_decode_attention_graphed(
    q: GpuTensor,
    cu_seqlens_q: GpuTensor,
    _seqused_k: GpuTensor,
    block_table: GpuTensor,
    max_seqlen_q: usize,
    max_seqlen_k: usize,
    scale: f32,
    softcap: f32,
    window_size_left: i32,
    kv_cache: &KvCachePool,
    layer_idx: usize,
    ctx: Fp8GraphCtx,
    alloc: &mut CachingAllocator,
    stream: CUstream,
    cos_sin_cache_ptr: *const u8,
    rotary_dim: usize,
    is_rotary_interleaved: bool,
) -> OwnedTensor {
    let batch_size = cu_seqlens_q.dim(0) - 1;

    // Use cached host-side scales (no D2H sync needed).
    let k_scale = *ctx.k_scales.add(layer_idx);
    let v_scale = *ctx.v_scales.add(layer_idx);

    // cu_seqlens_k lives in the graph runner's persistent buffer.
    // During capture: filled with dummy prefix sums by fill_dummy_decode_fp8.
    // During replay: updated by cuda_worker.rs before graph launch.
    let cu_seqlens_k_gpu = GpuTensor::new(ctx.cu_seqlens_k, &[batch_size + 1], DType::I32);

    // Dequant into pre-allocated buffers with over-sized grid.
    // Blocks beyond actual total_kv_tokens exit early (kernel bounds check).
    kernels::dequant_gather_pages_into(
        *kv_cache.k_cache(layer_idx),
        block_table,
        cu_seqlens_k_gpu,
        k_scale,
        ctx.max_total_kv,
        ctx.num_kv_heads,
        ctx.head_dim,
        kv_cache.block_size_for(layer_idx),
        ctx.output_dtype,
        ctx.k_buf,
        stream,
    );
    kernels::dequant_gather_pages_into(
        *kv_cache.v_cache(layer_idx),
        block_table,
        cu_seqlens_k_gpu,
        v_scale,
        ctx.max_total_kv,
        ctx.num_kv_heads,
        ctx.head_dim,
        kv_cache.block_size_for(layer_idx),
        ctx.output_dtype,
        ctx.v_buf,
        stream,
    );

    // Create tensor views into the pre-allocated buffers.
    let k_contiguous = GpuTensor::new(
        ctx.k_buf,
        &[ctx.max_total_kv, ctx.num_kv_heads, ctx.head_dim],
        ctx.output_dtype,
    );
    let v_contiguous = GpuTensor::new(
        ctx.v_buf,
        &[ctx.max_total_kv, ctx.num_kv_heads, ctx.head_dim],
        ctx.output_dtype,
    );

    // Contiguous FA2 on the dequantized K/V.
    kernels::flash_attn_contiguous(
        q,
        k_contiguous,
        v_contiguous,
        cu_seqlens_q,
        cu_seqlens_k_gpu,
        max_seqlen_q,
        max_seqlen_k,
        scale,
        true,
        softcap,
        window_size_left,
        alloc,
        stream,
        cos_sin_cache_ptr,
        rotary_dim,
        is_rotary_interleaved,
    )
}

// ---------------------------------------------------------------------------
// Spans: pre/post rotation helpers
// ---------------------------------------------------------------------------

/// Rotate span blocks in the KV cache before attention, run the attention
/// function, then un-rotate span blocks after. This is a no-op when spans
/// are disabled (both GPU flag pointers are null).
///
/// All model architectures should use this instead of calling attention
/// functions directly when spans may be active.
///
/// `cos_sin_cache`: `[max_pos, rotary_dim]` — the model's rotary cache.
/// `attn_fn`: closure that runs the actual attention (any variant).
#[allow(clippy::too_many_arguments)]
pub unsafe fn with_span_rotation<F>(
    kv_cache: &KvCachePool,
    layer_idx: usize,
    cos_sin_cache: TensorView<'_>,
    cu_seqlens_q: TensorView<'_>,
    seqused_k: TensorView<'_>,
    block_table: TensorView<'_>,
    max_seqlen_k: usize,
    stream: CUstream,
    attn_fn: F,
) -> OwnedTensor
where
    F: FnOnce() -> OwnedTensor,
{
    let unrotated_flags = kv_cache.block_unrotated_gpu();
    let span_flags = kv_cache.block_span_gpu();
    let has_spans = !unrotated_flags.is_null() && !span_flags.is_null();

    if has_spans {
        let batch_size = cu_seqlens_q.dim(0) as usize - 1;
        let max_blocks = block_table.dim(1) as usize;

        // Pre-attention: rotate blocks that are currently unrotated.
        kernels::rotary_paged_k_cache(
            *kv_cache.k_cache(layer_idx),
            *cos_sin_cache,
            *block_table,
            *seqused_k,
            unrotated_flags,
            batch_size,
            max_seqlen_k,
            max_blocks,
            kv_cache.block_size_for(layer_idx),
            kv_cache.num_kv_heads,
            kv_cache.head_dim,
            false, // forward rotation
            stream,
        );
    }

    let result = attn_fn();

    if has_spans {
        let batch_size = cu_seqlens_q.dim(0) as usize - 1;
        let max_blocks = block_table.dim(1) as usize;

        // Post-attention: un-rotate all span blocks.
        kernels::rotary_paged_k_cache(
            *kv_cache.k_cache(layer_idx),
            *cos_sin_cache,
            *block_table,
            *seqused_k,
            span_flags,
            batch_size,
            max_seqlen_k,
            max_blocks,
            kv_cache.block_size_for(layer_idx),
            kv_cache.num_kv_heads,
            kv_cache.head_dim,
            true, // inverse rotation
            stream,
        );
    }

    result
}

/// True if the given stream is currently inside a `cuStreamBeginCapture`
/// region. Used at FA3/FA2 dispatch sites to skip backends whose workspace
/// sizing varies per call (and would therefore be unsafe to capture once
/// and replay across different shapes).
///
/// # Safety
/// `stream` must be a valid CUDA stream.
pub unsafe fn is_stream_capturing(stream: CUstream) -> bool {
    let mut status = cudarc::driver::sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE;
    let rc = cudarc::driver::sys::cuStreamIsCapturing(stream, &mut status);
    rc == cudarc::driver::sys::CUresult::CUDA_SUCCESS
        && status == cudarc::driver::sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_ACTIVE
}

// ---------------------------------------------------------------------------
// FlashAttention-3 (Hopper-native) decode
// ---------------------------------------------------------------------------

/// Hopper-native FA3 paged decode wrapper. Mirrors [`flashinfer_attention`]'s
/// argument shape so the call site in `FlashInferAttentionDecode` can swap
/// between the two without restructuring.
///
/// Caller must verify `cuda_arch >= 90` (Hopper) before invoking — the
/// underlying `libvllm_flash_attn_3.a` is built with `-arch=sm_90a` and won't
/// run on older devices. The build skips the lib on pre-sm_90 hosts so this
/// function exists but the linker will refuse to resolve it without FA3 —
/// see `crates/scratchy-serving-cuda/build.rs`.
///
/// `layer_idx`: 0-based layer index within the forward step. The FA3
/// wrapper uses `layer_idx == 0` as the AOT-build trigger — at layer 0
/// the scheduler prelude (`prepare_varlen_num_blocks`) runs once into a
/// process-persistent metadata workspace; subsequent layers (1..N-1) skip
/// the prelude and read the pre-built metadata. This matches Python vLLM's
/// AOT-scheduling behavior and saves N-1 prelude kernel launches per
/// captured forward step.
///
/// Currently only bf16 hdim=128 is supported (matches the single
/// instantiation we vendor). Other shapes panic in debug.
#[cfg(fa3_built)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn flash_attn_3_decode(
    q: TensorView<'_>,
    cu_seqlens_q: TensorView<'_>,
    seqused_k: TensorView<'_>,
    block_table: TensorView<'_>,
    max_seqlen_q: usize,
    max_seqlen_k: usize,
    scale: f32,
    kv_cache: &KvCachePool,
    // kv_layer: absolute layer index into kv_cache (includes layer_offset
    // for pipeline-parallel sub-models).
    // step_layer: 0-based layer index WITHIN this forward step (resets to 0
    // at the start of each forward, even on the second pp stage). Used to
    // detect the "first FA3 call this step" so the wrapper can run the AOT
    // scheduler-prelude exactly once per step.
    kv_layer: usize,
    step_layer: usize,
    num_sm: i32,
    alloc: &mut CachingAllocator,
    stream: CUstream,
) -> OwnedTensor {
    let k_layer = kv_cache.k_cache(kv_layer).as_raw();
    let v_layer = kv_cache.v_cache(kv_layer).as_raw();
    let block_size = kv_cache.block_size_for(kv_layer);
    crate::flash_attn_3::flash_attn_3_paged_decode_bf16_hdim128(
        q.as_raw(),
        k_layer,
        v_layer,
        cu_seqlens_q.as_raw(),
        seqused_k.as_raw(),
        block_table.as_raw(),
        max_seqlen_q,
        max_seqlen_k,
        scale,
        block_size,
        num_sm,
        step_layer,
        alloc,
        stream,
    )
}

// ---------------------------------------------------------------------------
// FlashInfer attention
// ---------------------------------------------------------------------------

/// Process-global FlashInfer plan cache.
///
/// Intentionally NOT `thread_local!`: scr's executor captures the forward
/// pass on one worker thread (where [`flashinfer_attention`] builds the
/// plan) but runs graph replay (and our pre-replay
/// [`replan_fi_for_decode`]) on a different thread. A per-thread cache
/// splits that state and the replan call finds a null handle at replay.
///
/// The plan handle is opaque CUDA state; the FI shim doesn't hold any
/// locks of its own, so a Mutex around the cache is sufficient. Lock
/// contention is minimal — each forward step briefly holds it during
/// (1) build/replan, (2) 16 per-layer set_io+run pairs, and (3) the
/// pre-replay replan.
static FI_PLAN_CACHE: std::sync::Mutex<FlashInferPlanCache> =
    std::sync::Mutex::new(FlashInferPlanCache::new());

/// Round `max_seqlen_k` up to the nearest bucket used by the solver-calibrated
/// FI cost table. Matches `SK_BUCKETS` in scratchy-forward-compiler-macro.
///
/// The bucket is used as a plan-cache key: changing sk_bucket tears down the
/// previous plan and rebuilds workspaces. Keeping the mapping monotonic
/// ensures that a decode sweep from sk=128 → 8192 rebuilds at bucket
/// boundaries, not on every token.
pub fn sk_bucket_for(max_seqlen_k: usize) -> u32 {
    const BUCKETS: &[u32] = &[128, 256, 512, 1024, 2048, 4096, 8192];
    for &b in BUCKETS {
        if (max_seqlen_k as u32) <= b {
            return b;
        }
    }
    // Beyond the largest bucket — round up to next power-of-two to keep the
    // plan-cache key stable across similar lengths.
    (max_seqlen_k as u32).next_power_of_two()
}

/// Run one layer of FlashInfer paged attention. Called from
/// `FlashInferAttentionDecodeImpl` / `…PrefillImpl` codegen.
///
/// Expects `cu_seqlens_q.dim(0) == 2` (batch_size=1) — the unified shim is
/// single-sequence-only. When this precondition doesn't hold the caller
/// must use the FA2 fallback instead.
///
/// `cfg` selects the compiled tuple; if the tuple is not in
/// `FLASHINFER_CONFIG_SET` (see `scratchy-builder-cuda`), this returns
/// `None` and the caller must fall back.
///
/// `sk_bucket` is the monotonic ceiling of `max_seqlen_k` — see
/// [`sk_bucket_for`]. Changing it invalidates the cached plan.
#[allow(clippy::too_many_arguments)]
pub unsafe fn flashinfer_attention(
    q: TensorView<'_>,
    cu_seqlens_q: TensorView<'_>,
    _seqused_k: TensorView<'_>,
    block_table: TensorView<'_>,
    max_seqlen_q: usize,
    max_seqlen_k: usize,
    scale: f32,
    softcap: f32,
    kv_cache: &KvCachePool,
    layer_idx: usize,
    num_sm: i32,
    cfg: FlashInferConfig,
    sk_bucket: u32,
    alloc: &mut CachingAllocator,
    stream: CUstream,
) -> Option<OwnedTensor> {
    // The shim is single-sequence-only (see doc precondition above). Without
    // this guard, multi-seq callers — chunked-prefill chunk 2+ and any mixed
    // batch where max_q > 1 — feed a flattened (seq_len, max_seqlen_k) into
    // the FI planner that misrepresents the shape and produces corrupted
    // scheduling. The planner can return rc=0 with garbage state and a later
    // unrelated kernel faults with ILLEGAL_ADDRESS (cf. the rc!=0 branch
    // below — same poison pattern).
    if cu_seqlens_q.dim(0) != 2 {
        return None;
    }
    debug_assert_eq!(
        q.ndim(),
        3,
        "FlashInfer expects Q shape [seq_len, num_qo_heads, head_dim]"
    );
    debug_assert_eq!(q.dim(2) as u32, cfg.head_dim, "Q head_dim != cfg.head_dim");

    let seq_len = q.dim(0);
    let num_qo_heads = q.dim(1);
    let num_kv_heads = kv_cache.num_kv_heads;
    let head_dim = kv_cache.head_dim;
    let page_size = kv_cache.block_size_for(layer_idx);
    let num_pages = max_seqlen_k.div_ceil(page_size);

    if std::env::var("FI_TRACE").is_ok() && layer_idx == 0 {
        eprintln!(
            "[FI] L0 seq_len={} max_k={} scale={} softcap={} num_qo={} num_kv={} hd={} ps={} np={}",
            seq_len,
            max_seqlen_k,
            scale,
            softcap,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            num_pages
        );
    }

    let o = alloc.alloc_tensor(&[seq_len, num_qo_heads, head_dim], q.dtype());
    let (float_ws, int_ws) = flashinfer::workspace_bytes(num_sm, head_dim, num_kv_heads);

    // block_table is [batch_size, max_pages]; batch=1 → row 0 is the kv_indices
    // array. Shim reads exactly `num_pages` entries.
    let kv_indices_ptr = block_table.raw_ptr() as *const i32;
    let k_layer = kv_cache.k_cache(layer_idx);
    let v_layer = kv_cache.v_cache(layer_idx);

    {
        let mut cache = FI_PLAN_CACHE.lock().expect("FI_PLAN_CACHE poisoned");
        let built = unsafe {
            cache
                .ensure(
                    cfg,
                    max_seqlen_q as u32,
                    sk_bucket,
                    q.raw_ptr() as *const core::ffi::c_void,
                    k_layer.raw_ptr() as *const core::ffi::c_void,
                    v_layer.raw_ptr() as *const core::ffi::c_void,
                    kv_indices_ptr,
                    o.as_gpu_tensor().raw_ptr() as *mut core::ffi::c_void,
                    max_seqlen_k as i32,
                    num_qo_heads as i32,
                    num_kv_heads as i32,
                    head_dim as i32,
                    page_size as i32,
                    num_pages as i32,
                    num_sm,
                    float_ws,
                    int_ws,
                    scale,
                    softcap,
                    stream,
                )
                .is_some()
        };
        if !built {
            tracing::error!(
                ?cfg,
                sk_bucket,
                "FlashInfer plan build failed — falling back"
            );
            return None;
        }
        // Re-run the scheduler with the current step's (seqlen_k, num_pages).
        // The captured memcpy inside TwoStageHolisticPlanWithNumSm reads from
        // pinned int_ws_h — at graph replay, the pre-launch `replan_fi_for_decode`
        // updates int_ws_h with that step's scheduling before the captured
        // memcpy re-executes, so int_ws_d ends up with fresh data.
        unsafe {
            let rc = cache.replan(
                seq_len as i32,
                max_seqlen_k as i32,
                num_pages as i32,
                stream,
            );
            if rc != 0 {
                tracing::error!(rc, "fi_replan returned nonzero");
            }
        }

        unsafe {
            cache.set_io(
                q.raw_ptr() as *const core::ffi::c_void,
                k_layer.raw_ptr() as *const core::ffi::c_void,
                v_layer.raw_ptr() as *const core::ffi::c_void,
                kv_indices_ptr,
                o.as_gpu_tensor().raw_ptr() as *mut core::ffi::c_void,
            );
            let rc = cache.run(stream);
            if rc != 0 {
                tracing::error!(rc, "fi_run returned nonzero — falling back to FA2");
                // The kernel launch failed; `o` contains garbage. Returning
                // Some(o) here would propagate that garbage downstream and
                // poison the CUDA context (subsequent kernels error with
                // ILLEGAL_ADDRESS). Return None so the caller takes the FA2
                // fallback path.
                return None;
            }
        }
    }
    Some(o)
}

/// Re-plan the thread-local FI plan cache for the current decode step.
/// Call this BEFORE `graph_launch` at each decode step so the replayed
/// FI kernels read fresh scheduling data from `int_ws_d`. The H2D
/// memcpy issued here lands on `stream` and completes before any
/// subsequent graph kernel launch on the same stream.
///
/// No-op when no FI plan has been built (models without `sk_buckets`).
///
/// # Safety
/// `stream` must be the compute stream used by the graph runner.
pub unsafe fn replan_fi_for_decode(
    max_seqlen_q: usize,
    max_seqlen_k: usize,
    page_size: usize,
    stream: CUstream,
) {
    let num_pages = max_seqlen_k.div_ceil(page_size) as i32;
    let mut cache = FI_PLAN_CACHE.lock().expect("FI_PLAN_CACHE poisoned");
    unsafe {
        cache.replan(max_seqlen_q as i32, max_seqlen_k as i32, num_pages, stream);
    }
}

/// Drop the global FlashInfer plan cache. Call on worker teardown to
/// free the planner's `cudaMalloc`ed workspaces deterministically.
pub fn reset_fi_plan_cache() {
    FI_PLAN_CACHE
        .lock()
        .expect("FI_PLAN_CACHE poisoned")
        .clear();
}
