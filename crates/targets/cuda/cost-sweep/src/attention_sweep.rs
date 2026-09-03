// SPDX-License-Identifier: Apache-2.0
//! Attention sweep — FA2 vs FlashInfer across `(num_tokens, sk) × per-tuple`.
//!
//! For every head_dim in [`ATTN_SHAPES`] and every `FlashInferConfig` in
//! `FLASHINFER_CONFIG_SET`, times the paged attention step at each
//! `(m, sk)` grid point and emits CSV rows in the solver's
//! `(kernel, M, N, K, cost_us)` schema with axis overloading:
//!
//! ```text
//!   M = num_tokens
//!   N = sk_bucket (KV-cache span in tokens)
//!   K = head_dim
//! ```
//!
//! Kernel names: `fa2_attn_bf16_h{h}` (FA2 is softcap-agnostic — the
//! capping is a runtime float arg, not a template) and
//! `flashinfer_attn_bf16_h{h}_{no}softcap` (FI specializes on softcap at
//! compile time). `(num_qo_heads, num_kv_heads)` is a runtime parameter
//! of both kernels — the GQA ratio doesn't appear in the kernel name,
//! so the same row serves every model at that head_dim.
//!
//! Task #6's FI Impls consume these rows via
//! `profile.cost_us_for(fi_csv_name(head_dim, softcap), ctx.num_tokens(),
//!                      ctx.sk_bucket(), head_dim)` and gate compatibility
//! on CSV-row presence AND [`dispatch_for`] availability.
//!
//! Grid cells with `m > sk` are skipped — physically the prefill invariant
//! is `m ≤ sk` and decode is always `m == 1`. An absent row causes the
//! solver to route through FA2's fallback path at that workload.

#![cfg(feature = "cuda")]

use core::ffi::c_void;

use cudarc::driver::sys;
use scratchy_builder_cuda::flashinfer_config::{DType as BuilderDType, FLASHINFER_CONFIG_SET};
use scratchy_target_cuda::flashinfer::{
    FiDType, FlashInferConfig as RtFlashInferConfig, dispatch_for, workspace_bytes,
};

use crate::util::{
    bench_kernel, fa3_supported_on_this_host, gpu_alloc_fill, gpu_alloc_zeros, h2d_copy,
    query_num_sm,
};

// FA2 paged attention — raw extern-C symbol from `libvllm_flash_attn.a`.
// Mirrors `scratchy_target_cuda::kernels::mha_varlen_fwd` exactly. Re-declared
// here so the sweep times the kernel at the extern boundary (no
// `OwnedTensor` / `CachingAllocator` noise in the measurement).
#[allow(clippy::too_many_arguments)]
unsafe extern "C" {
    fn mha_varlen_fwd(
        q_ptr: *mut c_void,
        k_ptr: *mut c_void,
        v_ptr: *mut c_void,
        out_ptr: *mut c_void,
        softmax_lse_ptr: *mut c_void,

        cu_seqlens_q: *const i32,
        cu_seqlens_k: *const i32,
        seqused_k: *const i32,

        block_table: *const i32,
        block_table_batch_stride: i32,

        batch_size: i32,
        max_seqlen_q: i32,
        max_seqlen_k: i32,
        num_heads: i32,
        num_heads_k: i32,
        head_size: i32,
        page_block_size: i32,

        q_row_stride: i64,
        q_head_stride: i64,
        k_batch_stride: i64,
        k_row_stride: i64,
        k_head_stride: i64,
        o_row_stride: i64,
        o_head_stride: i64,

        softmax_scale: f32,
        is_causal: i32,
        window_size_left: i32,
        window_size_right: i32,
        softcap: f32,
        is_bf16: i32,
        num_splits: i32,

        softmax_lse_accum_ptr: *mut c_void,
        out_accum_ptr: *mut c_void,
        seqlenq_ngroups_swapped: i32,
        total_q: i32,

        rotary_cos_ptr: *const c_void,
        rotary_sin_ptr: *const c_void,
        rotary_dim: i32,
        rotate_cached_k: i32,
        is_rotary_interleaved: i32,

        block_unrotated_flags: *const u8,

        stream: sys::CUstream,
    );
}

// FA3 paged decode + scheduler-metadata prelude. Available only when the
// FA3 lib was compiled into this build (Hopper builds; see
// `crates/scratchy-builder-cuda/build.rs::build_flash_attention_3`).
// The sweep gates timing on a runtime sm_version check, so calling these
// on non-Hopper devices is impossible.
#[allow(clippy::too_many_arguments)]
unsafe extern "C" {
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
        stream: sys::CUstream,
    ) -> i32;

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
        stream: sys::CUstream,
    ) -> i32;
}

/// Per-cell iteration counts. Attention is much heavier than a GEMM row,
/// so fewer iterations — but still enough to average out variance. Matches
/// the shape of `attn_bench.cu`'s `warmup=20 / iters=500` at the tighter
/// budget a CSV calibration tolerates.
const WARMUP: u32 = 10;
const ITERS: u32 = 50;

/// Static upper bound passed to FA3 — mirrors the wrapper's
/// `FA3_MAX_NUM_SPLITS` and Python's `flash_attn_max_num_splits_for_cuda_graph`.
/// The runtime per-batch split count adapts via `num_splits_dynamic_ptr`,
/// capped at this value.
const FA3_NUM_SPLITS: i32 = 32;

/// Calibration shapes — one per head_dim present in `FLASHINFER_CONFIG_SET`.
/// Cost rows are keyed only on `(head_dim, softcap)` (the compile-time FI
/// specialization dims); `(num_qo_heads, num_kv_heads)` is a runtime
/// parameter of the compiled kernel, so a single canonical `(q, k)` per
/// head_dim gives a cost table that serves every GQA ratio. Picked to
/// match common Llama/Qwen shapes (32, 8 for h64/h128; 16, 2 for h256)
/// so the first row measured is representative.
const ATTN_SHAPES: &[(u32, u32, u32)] = &[
    (64, 32, 8),  // Llama-3.2-1B / Qwen2.5-0.5B (any q/k works at runtime)
    (128, 32, 8), // Llama-3.1-8B / Qwen2.5-7B
    (256, 16, 2), // Gemma/DeepSeek-style large head_dim
];

/// Page size for paged-KV. Fixed at vLLM's default block size — matches
/// `attn_bench.cu` and every in-tree model's `KvCachePool::block_size`.
const PAGE_SIZE: u32 = 16;

/// `num_tokens` grid — same layout as `gemm_sweep::M_VALUES`.
const M_VALUES: &[u32] = &[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096];

/// `sk_bucket` grid — matches `attention_helpers::sk_bucket_for` (and the
/// optional `sk_buckets = [...]` arg on `#[forward]`).
const SK_VALUES: &[u32] = &[128, 256, 512, 1024, 2048, 4096, 8192];

/// Softmax scale. Value is arbitrary for timing purposes — FA2 and FI
/// both apply it as a single fused multiply inside the inner loop.
const SM_SCALE: f32 = 0.125;

/// bf16 byte pattern producing a small positive value (~0.0097). Mirrors
/// `attn_bench.cu:79-81` — avoids the degenerate all-zeros softmax where
/// every attention score is equal. Timing-only, so the exact value is
/// unimportant beyond "finite, non-denormal".
const BF16_FILL: u8 = 0x3c;

pub fn run(launch_overhead_us: f64) {
    let stream: sys::CUstream = std::ptr::null_mut();
    let num_sm = query_num_sm();

    // ATTENTION_SWEEP_ONLY — narrow the sweep grid for fast iteration:
    //   "fa3" : only the cells where the FA3 row fires
    //           (head_dim == 128, m == 1, sk ∈ SK_VALUES).
    //           Skips FA2 and FI bench in those cells so the sweep
    //           ONLY emits fa3_* rows. ~1 minute on H100.
    //   "decode" : every (h, m=1, sk) cell — calibrates all decode
    //              kernels but skips prefill m's. ~5 minutes.
    //   unset : full grid (default).
    let mode = std::env::var("ATTENTION_SWEEP_ONLY").unwrap_or_default();

    for &(head_dim, num_qo_heads, num_kv_heads) in ATTN_SHAPES {
        if mode == "fa3" && head_dim != 128 {
            continue;
        }
        assert_eq!(
            num_qo_heads % num_kv_heads,
            0,
            "GQA invariant: num_qo_heads must be divisible by num_kv_heads"
        );
        for &sk in SK_VALUES {
            for &m in M_VALUES {
                if m > sk {
                    // Skip physically invalid cells — prefill requires
                    // `m ≤ sk`, decode has `m == 1`. Solver handles
                    // missing rows via its existing fallback logic.
                    continue;
                }
                if (mode == "fa3" || mode == "decode") && m != 1 {
                    continue;
                }
                let fa3_only = mode == "fa3";
                bench_cell(
                    stream,
                    launch_overhead_us,
                    num_sm,
                    head_dim,
                    num_qo_heads,
                    num_kv_heads,
                    PAGE_SIZE,
                    m,
                    sk,
                    fa3_only,
                );
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn bench_cell(
    stream: sys::CUstream,
    launch_overhead_us: f64,
    num_sm: i32,
    head_dim: u32,
    num_qo_heads: u32,
    num_kv_heads: u32,
    page_size: u32,
    m: u32,
    sk: u32,
    fa3_only: bool,
) {
    let num_pages = sk.div_ceil(page_size);
    let head_dim_us = head_dim as usize;

    // Buffer sizes (all bf16 = 2 bytes/elem unless noted).
    let q_bytes = (m * num_qo_heads * head_dim) as usize * 2;
    let kv_bytes = (num_pages * page_size * num_kv_heads * head_dim) as usize * 2;
    let lse_bytes = (m * num_qo_heads) as usize * 4; // f32

    let q = gpu_alloc_fill(q_bytes, BF16_FILL);
    let k = gpu_alloc_fill(kv_bytes, BF16_FILL);
    let v = gpu_alloc_fill(kv_bytes, BF16_FILL);
    let o = gpu_alloc_zeros(q_bytes);
    let softmax_lse = gpu_alloc_zeros(lse_bytes);

    // Small index buffers — all batch=1.
    let cu_seqlens_q = gpu_alloc_zeros(2 * 4);
    let cu_seqlens_k = gpu_alloc_zeros(2 * 4);
    let seqused_k = gpu_alloc_zeros(4);
    let block_table = gpu_alloc_zeros(num_pages as usize * 4);
    let kv_indices = gpu_alloc_zeros(num_pages as usize * 4);

    // Upload small host-side setup once per cell.
    unsafe {
        let cu_q: [i32; 2] = [0, m as i32];
        let cu_k: [i32; 2] = [0, sk as i32];
        let sused: [i32; 1] = [sk as i32];
        h2d_copy(cu_seqlens_q, cu_q.as_ptr() as *const u8, 2 * 4);
        h2d_copy(cu_seqlens_k, cu_k.as_ptr() as *const u8, 2 * 4);
        h2d_copy(seqused_k, sused.as_ptr() as *const u8, 4);

        let indices: Vec<i32> = (0..num_pages as i32).collect();
        h2d_copy(
            block_table,
            indices.as_ptr() as *const u8,
            num_pages as usize * 4,
        );
        h2d_copy(
            kv_indices,
            indices.as_ptr() as *const u8,
            num_pages as usize * 4,
        );
    }

    // ── FA2 ──
    // Strides match kernels.rs::flash_attn_paged_ext's paged layout.
    let q_row_stride = (num_qo_heads * head_dim) as i64;
    let q_head_stride = head_dim as i64;
    let k_batch_stride = (page_size * num_kv_heads * head_dim) as i64;
    let k_row_stride = (num_kv_heads * head_dim) as i64;
    let k_head_stride = head_dim as i64;

    let fa2_launch_disabled = fa3_only;
    let fa2_launch = || unsafe {
        mha_varlen_fwd(
            q as *mut c_void,
            k as *mut c_void,
            v as *mut c_void,
            o as *mut c_void,
            softmax_lse as *mut c_void,
            cu_seqlens_q as *const i32,
            cu_seqlens_k as *const i32,
            seqused_k as *const i32,
            block_table as *const i32,
            num_pages as i32, // block_table_batch_stride
            1,                // batch_size
            m as i32,
            sk as i32,
            num_qo_heads as i32,
            num_kv_heads as i32,
            head_dim as i32,
            page_size as i32,
            q_row_stride,
            q_head_stride,
            k_batch_stride,
            k_row_stride,
            k_head_stride,
            q_row_stride,  // o_row_stride
            q_head_stride, // o_head_stride
            SM_SCALE,
            1,   // is_causal
            -1,  // window_size_left (disabled)
            0,   // window_size_right (causal)
            0.0, // softcap — runtime; kernel is not specialized on it
            1,   // is_bf16
            1,   // num_splits
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            0,        // seqlenq_ngroups_swapped
            m as i32, // total_q
            std::ptr::null(),
            std::ptr::null(),
            0,
            0,
            0,
            std::ptr::null(),
            stream,
        )
    };
    if !fa2_launch_disabled {
        let us = (bench_kernel(stream, WARMUP, ITERS, fa2_launch) - launch_overhead_us).max(0.0);
        println!("fa2_attn_bf16_h{head_dim},{m},{sk},{head_dim},{us:.2}");
    }

    // ── FlashInfer (one row per FLASHINFER_CONFIG_SET tuple matching this head_dim) ──
    let (float_ws, int_ws) = workspace_bytes(num_sm, head_dim_us, num_kv_heads as usize);

    for cfg in FLASHINFER_CONFIG_SET.iter().copied().filter(|_| !fa3_only) {
        if cfg.head_dim != head_dim {
            continue;
        }
        // Only bf16 tuples compile today — the Fp16 variant is a
        // placeholder (see FLASHINFER_CONFIG_SET doc comment).
        let rt_cfg = match cfg.dtype {
            BuilderDType::Bf16 => RtFlashInferConfig {
                dtype: FiDType::Bf16,
                head_dim,
                use_logits_soft_cap: cfg.use_logits_soft_cap,
            },
            BuilderDType::Fp16 => continue,
        };
        let Some(dispatch) = dispatch_for(rt_cfg) else {
            eprintln!("attention_sweep: no dispatch for {:?} (skipping)", rt_cfg);
            continue;
        };

        let mut rc: i32 = 0;
        // `logits_soft_cap` is a template-specialized constant in the FI
        // shim; the runtime float arg is only consulted when the template
        // has softcap enabled. Pass 0.0 otherwise — matches the bench's
        // convention and the prior-session benchmark.
        let soft_cap_arg = if cfg.use_logits_soft_cap { 30.0 } else { 0.0 };
        let handle = unsafe {
            (dispatch.plan_new)(
                q as *const c_void,
                k as *const c_void,
                v as *const c_void,
                kv_indices as *const i32,
                o as *mut c_void,
                m as i32,
                sk as i32,
                num_qo_heads as i32,
                num_kv_heads as i32,
                head_dim as i32,
                page_size as i32,
                num_pages as i32,
                num_sm, // target_num_clusters
                float_ws,
                int_ws,
                SM_SCALE,
                soft_cap_arg,
                stream,
                &mut rc as *mut i32,
            )
        };
        if handle.is_null() {
            eprintln!(
                "fi_plan_new failed for cfg={:?} m={m} sk={sk} rc={rc}",
                rt_cfg
            );
            continue;
        }

        // Trial launch — detects runtime errors the planner can't catch
        // (e.g. sm_89 vs head_dim=256 exceeds the 100 KB max dynamic smem
        // budget via cudaFuncSetAttribute). If the kernel fails here we
        // skip row emission so the solver's CostTable doesn't ingest
        // fast-fail timings as if they were legitimate costs. The loss
        // of one failed cell means the Impl falls back to FA2 at that
        // workload — a graceful degradation, not a correctness hole.
        let trial_rc = unsafe { (dispatch.run)(handle, stream) };
        if trial_rc != 0 {
            eprintln!(
                "fi_run trial rc={trial_rc} for cfg={:?} m={m} sk={sk} — skipping row",
                rt_cfg
            );
            unsafe { (dispatch.plan_delete)(handle) };
            continue;
        }

        let fi_launch = || {
            let _ = unsafe { (dispatch.run)(handle, stream) };
        };
        let us = (bench_kernel(stream, WARMUP, ITERS, fi_launch) - launch_overhead_us).max(0.0);
        let softcap_tok = if cfg.use_logits_soft_cap {
            "softcap"
        } else {
            "nosoftcap"
        };
        println!("flashinfer_attn_bf16_h{head_dim}_{softcap_tok},{m},{sk},{head_dim},{us:.2}");

        unsafe { (dispatch.plan_delete)(handle) };
    }

    // ── FA3 (Hopper-only, hdim128 only) ──
    //
    // Only emit a row when this build host is sm_90+ (otherwise the
    // `fa3_*` symbols may not be resolvable, and even if they are, the
    // sm_90a-compiled kernel would refuse to run). Also gated to
    // hdim128 (the only instantiation in `build_flash_attention_3`)
    // and to decode shape `m == 1` (the FA3 Decode Impl's
    // `WorkloadConstraint::NumTokensAndSkRange::num_tokens=(1,1)`).
    //
    // Bench mirrors `flash_attn_3_paged_decode_bf16_hdim128`:
    // (1) one-shot AOT prelude into a workspace, (2) timed loop of the
    // consumer with `skip_scheduler_metadata=1`. The prelude is hoisted
    // out of the timed loop so the row reflects the actual per-call
    // cost the dispatcher pays at layers 1..N-1 — layer 0's prelude
    // amortizes across the layer count.
    //
    // TODO(graph-mode-sweep): the rows emitted here measure FA3 in
    // EAGER mode but scratchy runs decode in CUDA graph replay where
    // launches collapse to ~0 µs. FA3 has 2 kernel launches per call
    // (main + combine) vs FI's 1, so eager-mode microbench undercounts
    // FA3's graph-mode advantage by `1 * launch_overhead_us` ≈ 3 µs.
    // We subtract `2 * launch_overhead_us` below to compensate, but
    // the residual gap to reality is still ~3-6 µs at sk=8192 — the
    // latency bench (`scr bench latency`) shows FA3 winning by
    // 1-2 % at every BS=1 shape, but eager-mode rows put FA3 above FI
    // at every cell. As of 2026-05-29 the H100 cost CSV has hand-
    // edited fa3_attn_* rows (see comment block in
    // `crates/scratchy-target-cuda/profiles/cost_h100_sm90.csv`) that
    // override the sweep output. Proper fix: capture the FA3 launches
    // in a CUDA graph and time replay; that becomes the row.
    if head_dim == 128 && m == 1 && fa3_supported_on_this_host() {
        // Persistent workspace for the prelude. Sized for max
        // num_prepare_batch_vectors=4 + 1 semaphore = `1 + b_rounded*4`.
        // batch_size=1 → b_rounded=4 → 17 ints.
        let metadata_bytes = (1 + 4 * 4) * std::mem::size_of::<i32>();
        let metadata = gpu_alloc_zeros(metadata_bytes);

        // oaccum / lseaccum sized for FA3_NUM_SPLITS x h x m x hdim
        // (matches the shim's per-call alloc shape).
        let oaccum_bytes =
            FA3_NUM_SPLITS as usize * num_qo_heads as usize * m as usize * head_dim as usize * 4;
        let lseaccum_bytes = FA3_NUM_SPLITS as usize * num_qo_heads as usize * m as usize * 4;
        let oaccum = gpu_alloc_zeros(oaccum_bytes);
        let lseaccum = gpu_alloc_zeros(lseaccum_bytes);

        let max_pages_per_seq = num_pages as i32;
        let num_pages_total = num_pages as i32;

        // One-shot AOT prelude — populates the persistent metadata
        // buffer. Mirrors layer-0 of the dispatcher.
        let prelude_rc = unsafe {
            fa3_get_scheduler_metadata_bf16_hdim128_sm90(
                metadata as *mut i32,
                block_table as *const i32,
                cu_seqlens_q as *const i32,
                seqused_k as *const i32,
                1, // batch_size
                m as i32,
                num_qo_heads as i32,
                num_kv_heads as i32,
                max_pages_per_seq,
                num_pages_total,
                page_size as i32,
                m as i32,
                sk as i32,
                FA3_NUM_SPLITS,
                SM_SCALE,
                num_sm,
                stream,
            )
        };
        if prelude_rc != 0 {
            eprintln!(
                "fa3 get_scheduler_metadata rc={prelude_rc} for m={m} sk={sk} — skipping row"
            );
        } else {
            // Trial launch — same protective check as FI's row emission.
            let trial_rc = unsafe {
                fa3_paged_decode_bf16_hdim128_sm90(
                    q as *const c_void,
                    k as *const c_void,
                    v as *const c_void,
                    o as *mut c_void,
                    softmax_lse as *mut c_void,
                    oaccum as *mut c_void,
                    lseaccum as *mut c_void,
                    metadata as *mut i32,
                    block_table as *const i32,
                    cu_seqlens_q as *const i32,
                    seqused_k as *const i32,
                    1, // batch_size
                    m as i32,
                    num_qo_heads as i32,
                    num_kv_heads as i32,
                    max_pages_per_seq,
                    num_pages_total,
                    page_size as i32,
                    m as i32,
                    sk as i32,
                    FA3_NUM_SPLITS,
                    1, // skip_scheduler_metadata
                    SM_SCALE,
                    num_sm,
                    stream,
                )
            };
            if trial_rc != 0 {
                eprintln!("fa3 trial rc={trial_rc} for m={m} sk={sk} — skipping row");
            } else {
                let fa3_launch = || unsafe {
                    let _ = fa3_paged_decode_bf16_hdim128_sm90(
                        q as *const c_void,
                        k as *const c_void,
                        v as *const c_void,
                        o as *mut c_void,
                        softmax_lse as *mut c_void,
                        oaccum as *mut c_void,
                        lseaccum as *mut c_void,
                        metadata as *mut i32,
                        block_table as *const i32,
                        cu_seqlens_q as *const i32,
                        seqused_k as *const i32,
                        1, // batch_size
                        m as i32,
                        num_qo_heads as i32,
                        num_kv_heads as i32,
                        max_pages_per_seq,
                        num_pages_total,
                        page_size as i32,
                        m as i32,
                        sk as i32,
                        FA3_NUM_SPLITS,
                        1, // skip_scheduler_metadata
                        SM_SCALE,
                        num_sm,
                        stream,
                    );
                };
                // FA3 with num_splits>1 launches TWO kernels per call:
                // the main split kernel + the combine kernel. FI's
                // persistent decode is a single kernel launch. Subtract
                // both launch overheads so the row reports kernel-only
                // cost — the solver's per-cell comparison is then
                // apples-to-apples with FI's row, AND under CUDA graph
                // replay (where scratchy actually runs decode) the
                // launches collapse to ~0 µs each anyway. Without this
                // adjustment FA3 looks 6+ µs slower than FI per cell
                // even though the latency bench shows FA3 winning at
                // every BS=1 shape — that's the launch-overhead
                // mismatch between eager-mode sweep and graph-mode
                // serving.
                let raw_us = bench_kernel(stream, WARMUP, ITERS, fa3_launch);
                let n_launches = if FA3_NUM_SPLITS > 1 { 2 } else { 1 } as f64;
                let us = (raw_us - launch_overhead_us * n_launches).max(0.0);
                println!("fa3_attn_bf16_h{head_dim}_nosoftcap_decode,{m},{sk},{head_dim},{us:.2}");
            }
        }

        unsafe {
            sys::cuMemFree_v2(metadata);
            sys::cuMemFree_v2(oaccum);
            sys::cuMemFree_v2(lseaccum);
        }
    }

    unsafe {
        sys::cuMemFree_v2(q);
        sys::cuMemFree_v2(k);
        sys::cuMemFree_v2(v);
        sys::cuMemFree_v2(o);
        sys::cuMemFree_v2(softmax_lse);
        sys::cuMemFree_v2(cu_seqlens_q);
        sys::cuMemFree_v2(cu_seqlens_k);
        sys::cuMemFree_v2(seqused_k);
        sys::cuMemFree_v2(block_table);
        sys::cuMemFree_v2(kv_indices);
    }
}
