// SPDX-License-Identifier: Apache-2.0
//! FlashAttention-3 (Hopper-native) FFI + Rust wrapper.
//!
//! Mirrors the AOT-scheduling pattern used by Python vLLM
//! (`vllm/v1/attention/backends/flash_attn.py`):
//!
//! 1. Once per forward step (at `layer_idx == 0`), call
//!    [`fa3_aot_build_metadata`] to run `prepare_varlen_num_blocks` into a
//!    process-persistent workspace.
//! 2. At every attention layer call, [`flash_attn_3_paged_decode_bf16_hdim128`]
//!    invokes the main kernel with `skip_scheduler_metadata=1`, so the
//!    kernel reads the pre-populated workspace instead of running its own
//!    prelude. This saves N-1 prelude kernel launches per forward
//!    (where N = num attention layers).
//!
//! Under CUDA graph capture, the layer-0 metadata build kernel and the
//! per-layer consumer kernels are all captured into the same graph.
//! At replay, the metadata build re-fires with the captured seqused_k
//! pointer (whose underlying buffer is updated host-side before each
//! replay), and the consumer kernels read the freshly-rebuilt workspace.
//! This matches Python vLLM's full-CUDA-graph behavior.
//!
//! Persistent workspace allocation is process-lifetime (`cuMemAlloc`,
//! never freed). We do NOT route through the caching allocator because:
//! - the workspace must outlive any single forward step,
//! - it is shared across all captured graphs (and graph replay reuses
//!   captured pointers, which the caching allocator cannot guarantee
//!   across capture sessions).

use crate::alloc::{CachingAllocator, OwnedTensor};
use crate::driver;
use crate::dtype::DType;
use crate::tensor::GpuTensor;
use core::ffi::c_void;

type CUstream = cudarc::driver::sys::CUstream;

unsafe extern "C" {
    /// FA3 paged decode: bf16, hdim128, sm_90a. See
    /// `third_party/flash-attn-3-shim/ffi_shim.cu`.
    ///
    /// `skip_scheduler_metadata`:
    ///   0 = the kernel runs its own prelude (`prepare_varlen_num_blocks`).
    ///   1 = the kernel skips the prelude and reads the pre-populated
    ///       workspace. Caller must have called `fa3_get_scheduler_metadata_*`
    ///       earlier this step with the same shape parameters.
    fn fa3_paged_decode_bf16_hdim128_sm90(
        q: *const c_void,
        k_cache: *const c_void,
        v_cache: *const c_void,
        out: *mut c_void,
        softmax_lse: *mut c_void,
        oaccum: *mut c_void,
        softmax_lseaccum: *mut c_void,
        scheduler_workspace: *mut i32,
        page_table: *const i32,
        cu_seqlens_q: *const i32,
        seqused_k: *const i32,
        batch_size: i32,
        total_q_tokens: i32,
        num_q_heads: i32,
        num_kv_heads: i32,
        max_pages_per_seq: i32,
        num_pages_total: i32,
        page_size: i32,
        max_seqlen_q: i32,
        max_seqlen_k: i32,
        num_splits: i32,
        skip_scheduler_metadata: i32,
        softmax_scale: f32,
        sm_count: i32,
        stream: CUstream,
    ) -> i32;

    /// AOT prelude: runs `prepare_varlen_num_blocks` once into the
    /// caller-provided workspace. Mirrors flash_api.cpp's
    /// `mha_fwd_get_scheduler_metadata`.
    ///
    /// `num_splits` must match the value the consumer call will use, so
    /// the workspace layout matches.
    fn fa3_get_scheduler_metadata_bf16_hdim128_sm90(
        workspace: *mut i32,
        page_table: *const i32,
        cu_seqlens_q: *const i32,
        seqused_k: *const i32,
        batch_size: i32,
        total_q_tokens: i32,
        num_q_heads: i32,
        num_kv_heads: i32,
        max_pages_per_seq: i32,
        num_pages_total: i32,
        page_size: i32,
        max_seqlen_q: i32,
        max_seqlen_k: i32,
        num_splits: i32,
        softmax_scale: f32,
        sm_count: i32,
        stream: CUstream,
    ) -> i32;
}

// ---------------------------------------------------------------------------
// FA3 num_splits heuristic — port of heuristics.h:32 (Hopper paged-KV decode).
// ---------------------------------------------------------------------------

/// Pick num_splits to saturate `num_sm` SMs given `total_mblocks` m-tiles.
/// `total_mblocks = batch * num_kv_heads * num_m_blocks` for pack_gqa.
/// kBlockN for hdim128 paged_kv_non_TMA on sm90 = 128.
pub fn fa3_num_splits(
    total_mblocks: usize,
    num_sm: usize,
    seqlen_k: usize,
    is_causal: bool,
) -> usize {
    const K_BLOCK_N: usize = 128;
    let num_n_blocks = seqlen_k.div_ceil(K_BLOCK_N);

    if total_mblocks as f32 >= 0.8 * num_sm as f32 {
        return 1;
    }
    if num_n_blocks <= 4 {
        return 1;
    }

    let max_splits = 128usize.min(num_sm).min(num_n_blocks);
    let mut max_eff = 0.0_f32;
    let mut effs = Vec::with_capacity(max_splits);
    for s in 1..=max_splits {
        let n_waves = (total_mblocks * s) as f32 / num_sm as f32;
        let eff = n_waves / n_waves.ceil();
        if eff > max_eff {
            max_eff = eff;
        }
        effs.push(eff);
    }
    for s in 1..=max_splits {
        if effs[s - 1] >= 0.85 * max_eff {
            return s;
        }
    }
    let _ = is_causal;
    1
}

// ---------------------------------------------------------------------------
// Process-persistent FA3 metadata buffer.
// ---------------------------------------------------------------------------
//
// Only the scheduler-metadata buffer is process-persistent here, matching
// Python vLLM's `self.scheduler_metadata` allocation in
// `flash_attn.py:362-371`. It is small (~1 KB at max_num_seqs=256) and
// must outlive any single forward step because layer 0's AOT build
// writes into it and layers 1..N-1 read from it.
//
// `oaccum` / `lseaccum` are allocated per-call from the caching allocator
// — same as torch in vllm-python. They live for the duration of one
// attention call (dropped after `run_mha_fwd_combine_`); under graph
// capture the caching allocator gives consistent pointers across replay.
// We don't pre-allocate them because (a) they don't need to be shared
// across layers, and (b) at max-batch decode the persistent allocation
// would steal hundreds of MB from the KV cache budget without a
// corresponding speedup.
//
// Pre-allocating the metadata buffer at worker init (before
// `determine_available_memory`'s profile run) keeps the budget honest:
// the profile + KV-cache-sizing flow then sees this allocation as
// non-torch memory and trims `num_gpu_blocks` accordingly. Lazy-init
// here is also safe — if the first FA3 call happens during the profile
// run we still get this 1 KB onto the persistent side of the ledger
// before KV cache sizing finalizes.

/// Max num_splits to support — matches Python's
/// `flash_attn_max_num_splits_for_cuda_graph` default. Used as the static
/// upper bound passed to the kernel; the runtime per-batch split count is
/// set by the prelude via `num_splits_dynamic_ptr` (see
/// `flash_prepare_scheduler.cu:161`).
pub const FA3_MAX_NUM_SPLITS: usize = 32;
/// Max batch size for the metadata buffer. Mirrors Python's
/// `max(scheduler_config.max_num_seqs, max_cudagraph_size)` at
/// flash_attn.py:362; in scratchy we don't have direct access to those
/// here so we hardcode a generous bound. Round up to mul-of-4 to match
/// Python's `round_up(max_batch_size, 4)`.
const FA3_MAX_BATCH: usize = 256;

// AtomicPtr is Send/Sync without unsafe; CAS for first-init.
use std::sync::atomic::{AtomicPtr, Ordering};
static FA3_METADATA_PTR: AtomicPtr<i32> = AtomicPtr::new(std::ptr::null_mut());

unsafe fn fa3_alloc_metadata_buffer() -> *mut i32 {
    let b_rounded_max = FA3_MAX_BATCH.div_ceil(4) * 4;
    // Always 4 prepare_batch_vectors slots — matches Python's static
    // sizing at `flash_attn.py:367`. Actual num_prepare_batch_vectors
    // varies (2 or 3), but allocating for the max is safe and trivially
    // small.
    let metadata_ints = 1 + b_rounded_max * 4;
    let bytes = metadata_ints * std::mem::size_of::<i32>();
    driver::mem_alloc(bytes).expect("FA3 metadata cuMemAlloc failed") as *mut i32
}

/// Pre-allocate the FA3 scheduler-metadata buffer. Idempotent — calling
/// this multiple times after the first is a no-op. Call before
/// `determine_available_memory` so the worker's KV-cache-sizing budget
/// includes this allocation as part of `weights_held` baseline.
///
/// # Safety
/// Must be called with a current CUDA context.
pub unsafe fn fa3_init_metadata() {
    if !FA3_METADATA_PTR.load(Ordering::Acquire).is_null() {
        return;
    }
    let ptr = fa3_alloc_metadata_buffer();
    // Race: if another thread won the CAS, we leak our local alloc.
    // In practice this is called once at worker init, no contention.
    if FA3_METADATA_PTR
        .compare_exchange(
            std::ptr::null_mut(),
            ptr,
            Ordering::AcqRel,
            Ordering::Acquire,
        )
        .is_err()
    {
        // Lost the race — free our local allocation.
        let _ = driver::mem_free(ptr as *mut u8);
    }
}

fn fa3_metadata_ptr() -> *mut i32 {
    let p = FA3_METADATA_PTR.load(Ordering::Acquire);
    if !p.is_null() {
        return p;
    }
    // Fallback lazy init — caller forgot fa3_init_metadata at worker
    // init. Still correct (subsequent forwards use the same persistent
    // allocation) but means this 4 KB lands in `peak_activations`
    // rather than `weights_held` and shaves the corresponding amount
    // off `num_gpu_blocks`. Not a correctness issue.
    let ptr = unsafe { fa3_alloc_metadata_buffer() };
    if FA3_METADATA_PTR
        .compare_exchange(
            std::ptr::null_mut(),
            ptr,
            Ordering::AcqRel,
            Ordering::Acquire,
        )
        .is_err()
    {
        unsafe {
            let _ = driver::mem_free(ptr as *mut u8);
        }
    }
    FA3_METADATA_PTR.load(Ordering::Acquire)
}

// ---------------------------------------------------------------------------
// AOT build entry point — call ONCE per forward step before the first
// FA3 attention layer fires.
// ---------------------------------------------------------------------------

/// Run `prepare_varlen_num_blocks` into the persistent metadata workspace.
/// Captures into the active CUDA graph if one is recording.
///
/// Caller (`flash_attn_3_paged_decode_bf16_hdim128`) invokes this when
/// `layer_idx == 0`. `num_splits` is the value the per-layer consumer will
/// pass; the workspace layout depends on it (use_dynamic_split adds an
/// extra prepare_batch_vector).
#[allow(clippy::too_many_arguments)]
unsafe fn fa3_aot_build_metadata(
    cu_seqlens_q: *const i32,
    seqused_k: *const i32,
    block_table: *const i32,
    batch_size: usize,
    total_q: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    max_pages_per_seq: usize,
    num_pages_total: usize,
    page_size: usize,
    max_seqlen_q: usize,
    max_seqlen_k: usize,
    num_splits: usize,
    softmax_scale: f32,
    num_sm: i32,
    stream: CUstream,
) {
    let metadata_ptr = fa3_metadata_ptr();
    let rc = fa3_get_scheduler_metadata_bf16_hdim128_sm90(
        metadata_ptr,
        block_table,
        cu_seqlens_q,
        seqused_k,
        batch_size as i32,
        total_q as i32,
        num_q_heads as i32,
        num_kv_heads as i32,
        max_pages_per_seq as i32,
        num_pages_total as i32,
        page_size as i32,
        max_seqlen_q as i32,
        max_seqlen_k as i32,
        num_splits as i32,
        softmax_scale,
        num_sm,
        stream,
    );
    if rc != 0 {
        panic!("fa3_get_scheduler_metadata returned {}", rc);
    }
}

// ---------------------------------------------------------------------------
// Public Rust wrapper.
// ---------------------------------------------------------------------------

/// FA3 paged decode for bf16 hdim128 on Hopper.
///
/// `layer_idx`: 0-based layer index within the forward step. When `0`, this
/// function first runs the AOT scheduler build (populating the persistent
/// metadata workspace) before invoking the consumer kernel. When `> 0`, the
/// metadata is already built (by layer 0) and the consumer is called
/// directly with `skip_scheduler_metadata=1`. `num_splits` is read from
/// the per-step state set by layer 0, so all layers in a step run with the
/// same split count.
///
/// Caller must verify `cuda_arch >= 90` before invoking; the underlying
/// library is sm_90a-only.
#[allow(clippy::too_many_arguments)]
pub unsafe fn flash_attn_3_paged_decode_bf16_hdim128(
    q: GpuTensor,
    k_cache: GpuTensor,
    v_cache: GpuTensor,
    cu_seqlens_q: GpuTensor,
    seqused_k: GpuTensor,
    block_table: GpuTensor,
    max_seqlen_q: usize,
    max_seqlen_k: usize,
    softmax_scale: f32,
    block_size: usize,
    num_sm: i32,
    layer_idx: usize,
    alloc: &mut CachingAllocator,
    stream: CUstream,
) -> OwnedTensor {
    debug_assert_eq!(q.dim(2), 128, "flash_attn_3 hdim128 only");

    let total_q = q.dim(0);
    let num_q_heads = q.dim(1);
    let head_dim = q.dim(2);
    let num_kv_heads = k_cache.dim(2);
    let num_pages_total = k_cache.dim(0);
    let batch_size = cu_seqlens_q.dim(0) - 1;
    let max_pages_per_seq = if block_table.ndim() == 2 {
        block_table.dim(1)
    } else {
        0
    };

    // Output + softmax_lse — caching allocator (per-step lifetime).
    let out = alloc.alloc_tensor(&[total_q, num_q_heads, head_dim], q.dtype());
    let softmax_lse = alloc.alloc_tensor(&[num_q_heads * total_q], DType::F32);

    // num_splits is ALWAYS the upper bound (FA3_MAX_NUM_SPLITS), matching
    // Python vLLM's `num_splits=max_num_splits` pattern at
    // flash_attn.py:472. The prelude (`prepare_varlen_num_blocks`) writes
    // the per-batch optimal split count to `num_splits_dynamic_ptr`,
    // capped at this static upper bound. The kernel reads the dynamic
    // value at runtime (block.h:58) and skips CTAs above it. With
    // `num_splits=14` baked into the captured launch, replay at sk=8192
    // would be stuck at 14 splits even though 29 is optimal — passing
    // the max bound here is what lets the dynamic scheduler adapt.
    //
    // Layer 0 also runs the AOT prelude into the persistent metadata
    // workspace; layers 1..N-1 reuse it via `skip_scheduler_metadata=1`.
    let num_splits = FA3_MAX_NUM_SPLITS;
    if layer_idx == 0 {
        fa3_aot_build_metadata(
            cu_seqlens_q.as_ptr::<i32>(),
            seqused_k.as_ptr::<i32>(),
            block_table.as_ptr::<i32>(),
            batch_size,
            total_q,
            num_q_heads,
            num_kv_heads,
            max_pages_per_seq,
            num_pages_total,
            block_size,
            max_seqlen_q,
            max_seqlen_k,
            num_splits,
            softmax_scale,
            num_sm,
            stream,
        );
    }

    // Per-call oaccum/lseaccum (only when splits>1) — allocated from the
    // caching allocator, matching Python vLLM which lets torch's allocator
    // create these inside `flash_attn_varlen_func` per call. They get
    // freed at end-of-call when the OwnedTensors drop. Under graph
    // capture the caching allocator returns consistent pointers across
    // replay; under eager execution they're real fresh allocations.
    //
    // Sized at actual shape, not at the max num_splits envelope: only
    // `num_splits * num_q_heads * total_q * head_dim` floats. At BS=8
    // decode hd=128 with num_splits=32 that's ~3.7 MB per call; falls
    // into `peak_activations` when the profile run drives a decode
    // shape (or naturally fits within scratchy's KV-sizing safety
    // margin when the profile is prefill-only, matching python).
    let metadata_ptr = fa3_metadata_ptr();
    let (oaccum_ptr, lseaccum_ptr, _oa_keep, _la_keep) = if num_splits > 1 {
        let oa = alloc.alloc_tensor(&[num_splits, num_q_heads, total_q, head_dim], DType::F32);
        let la = alloc.alloc_tensor(&[num_splits, num_q_heads, total_q], DType::F32);
        let oa_ptr = oa.raw_ptr() as *mut c_void;
        let la_ptr = la.raw_ptr() as *mut c_void;
        (oa_ptr, la_ptr, Some(oa), Some(la))
    } else {
        (std::ptr::null_mut(), std::ptr::null_mut(), None, None)
    };

    let rc = fa3_paged_decode_bf16_hdim128_sm90(
        q.raw_ptr() as *const c_void,
        k_cache.raw_ptr() as *const c_void,
        v_cache.raw_ptr() as *const c_void,
        out.raw_ptr() as *mut c_void,
        softmax_lse.raw_ptr() as *mut c_void,
        oaccum_ptr,
        lseaccum_ptr,
        metadata_ptr,
        block_table.as_ptr::<i32>(),
        cu_seqlens_q.as_ptr::<i32>(),
        seqused_k.as_ptr::<i32>(),
        batch_size as i32,
        total_q as i32,
        num_q_heads as i32,
        num_kv_heads as i32,
        max_pages_per_seq as i32,
        num_pages_total as i32,
        block_size as i32,
        max_seqlen_q as i32,
        max_seqlen_k as i32,
        num_splits as i32,
        /*skip_scheduler_metadata=*/ 1,
        softmax_scale,
        num_sm,
        stream,
    );
    if rc != 0 {
        panic!("fa3_paged_decode returned {}", rc);
    }
    out
}
