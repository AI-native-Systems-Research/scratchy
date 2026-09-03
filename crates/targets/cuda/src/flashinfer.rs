// SPDX-License-Identifier: Apache-2.0
//! FlashInfer FFI + plan cache.
//!
//! Links against `libflashinfer_attn.a` produced by
//! `scratchy-builder-cuda/build.rs::build_flashinfer_attention`. That library
//! exports one set of three entry points per tuple in
//! `scratchy_builder_cuda::flashinfer_config::FLASHINFER_CONFIG_SET`:
//!
//! - `fi_plan_<suffix>_new`    — build + cache a FlashInfer plan handle.
//! - `fi_plan_<suffix>_delete` — destroy a plan handle.
//! - `fi_plan_<suffix>_set_io` — rebind I/O pointers on an existing plan.
//! - `fi_run_<suffix>`         — dispatch the persistent runner on a plan.
//!
//! The Rust-side [`FlashInferConfig`] mirrors the builder's type so that the
//! match in [`dispatch_for`] can select the correct extern-C symbol at
//! runtime. If you add a tuple to `FLASHINFER_CONFIG_SET`, mirror it here by
//! adding the extern declaration + dispatch arm — otherwise `dispatch_for`
//! returns `None` and the caller must fall back to FA2.

use core::ffi::c_void;

type CUstream = cudarc::driver::sys::CUstream;

// ---------------------------------------------------------------------------
// Config mirror
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FiDType {
    Bf16,
    Fp16,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FlashInferConfig {
    pub dtype: FiDType,
    pub head_dim: u32,
    pub use_logits_soft_cap: bool,
}

// ---------------------------------------------------------------------------
// Extern-C decls — one set per tuple in FLASHINFER_CONFIG_SET.
// Keep in sync with `scratchy-builder-cuda/src/flashinfer_config.rs`.
// ---------------------------------------------------------------------------

macro_rules! fi_decls {
    ($new:ident, $del:ident, $set_io:ident, $replan:ident, $run:ident) => {
        unsafe extern "C" {
            pub fn $new(
                q: *const c_void,
                k: *const c_void,
                v: *const c_void,
                kv_indices: *const i32,
                o: *mut c_void,
                seq_len: i32,
                seqlen_k: i32,
                num_qo_heads: i32,
                num_kv_heads: i32,
                head_dim: i32,
                page_size: i32,
                num_pages: i32,
                target_num_clusters: i32,
                float_ws_bytes: usize,
                int_ws_bytes: usize,
                sm_scale: f32,
                logits_soft_cap: f32,
                stream: CUstream,
                rc_out: *mut i32,
            ) -> *mut c_void;

            pub fn $del(handle: *mut c_void);

            pub fn $set_io(
                handle: *mut c_void,
                q: *const c_void,
                k: *const c_void,
                v: *const c_void,
                kv_indices: *const i32,
                o: *mut c_void,
            );

            pub fn $replan(
                handle: *mut c_void,
                seq_len: i32,
                seqlen_k: i32,
                num_pages: i32,
                stream: CUstream,
            ) -> i32;

            pub fn $run(handle: *mut c_void, stream: CUstream) -> i32;
        }
    };
}

fi_decls!(
    fi_plan_bf16_h64_nosoftcap_new,
    fi_plan_bf16_h64_nosoftcap_delete,
    fi_plan_bf16_h64_nosoftcap_set_io,
    fi_plan_bf16_h64_nosoftcap_replan,
    fi_run_bf16_h64_nosoftcap
);
fi_decls!(
    fi_plan_bf16_h64_softcap_new,
    fi_plan_bf16_h64_softcap_delete,
    fi_plan_bf16_h64_softcap_set_io,
    fi_plan_bf16_h64_softcap_replan,
    fi_run_bf16_h64_softcap
);
fi_decls!(
    fi_plan_bf16_h128_nosoftcap_new,
    fi_plan_bf16_h128_nosoftcap_delete,
    fi_plan_bf16_h128_nosoftcap_set_io,
    fi_plan_bf16_h128_nosoftcap_replan,
    fi_run_bf16_h128_nosoftcap
);
fi_decls!(
    fi_plan_bf16_h128_softcap_new,
    fi_plan_bf16_h128_softcap_delete,
    fi_plan_bf16_h128_softcap_set_io,
    fi_plan_bf16_h128_softcap_replan,
    fi_run_bf16_h128_softcap
);
fi_decls!(
    fi_plan_bf16_h256_nosoftcap_new,
    fi_plan_bf16_h256_nosoftcap_delete,
    fi_plan_bf16_h256_nosoftcap_set_io,
    fi_plan_bf16_h256_nosoftcap_replan,
    fi_run_bf16_h256_nosoftcap
);
fi_decls!(
    fi_plan_bf16_h256_softcap_new,
    fi_plan_bf16_h256_softcap_delete,
    fi_plan_bf16_h256_softcap_set_io,
    fi_plan_bf16_h256_softcap_replan,
    fi_run_bf16_h256_softcap
);

// ---------------------------------------------------------------------------
// Dispatch table
// ---------------------------------------------------------------------------

#[allow(clippy::type_complexity)]
pub type PlanNewFn = unsafe extern "C" fn(
    *const c_void,
    *const c_void,
    *const c_void,
    *const i32,
    *mut c_void,
    i32,
    i32,
    i32,
    i32,
    i32,
    i32,
    i32,
    i32,
    usize,
    usize,
    f32,
    f32,
    CUstream,
    *mut i32,
) -> *mut c_void;

pub type PlanDeleteFn = unsafe extern "C" fn(*mut c_void);
pub type PlanSetIoFn = unsafe extern "C" fn(
    handle: *mut c_void,
    q: *const c_void,
    k: *const c_void,
    v: *const c_void,
    kv_indices: *const i32,
    o: *mut c_void,
);
pub type ReplanFn = unsafe extern "C" fn(*mut c_void, i32, i32, i32, CUstream) -> i32;
pub type RunFn = unsafe extern "C" fn(*mut c_void, CUstream) -> i32;

#[derive(Clone, Copy)]
pub struct FiDispatch {
    pub plan_new: PlanNewFn,
    pub plan_delete: PlanDeleteFn,
    pub plan_set_io: PlanSetIoFn,
    pub replan: ReplanFn,
    pub run: RunFn,
}

/// Resolve extern-C symbols for `cfg`. Returns `None` when the tuple is not
/// compiled into `libflashinfer_attn.a` (out-of-set — caller must fall back).
pub fn dispatch_for(cfg: FlashInferConfig) -> Option<FiDispatch> {
    match (cfg.dtype, cfg.head_dim, cfg.use_logits_soft_cap) {
        (FiDType::Bf16, 64, false) => Some(FiDispatch {
            plan_new: fi_plan_bf16_h64_nosoftcap_new,
            plan_delete: fi_plan_bf16_h64_nosoftcap_delete,
            plan_set_io: fi_plan_bf16_h64_nosoftcap_set_io,
            replan: fi_plan_bf16_h64_nosoftcap_replan,
            run: fi_run_bf16_h64_nosoftcap,
        }),
        (FiDType::Bf16, 64, true) => Some(FiDispatch {
            plan_new: fi_plan_bf16_h64_softcap_new,
            plan_delete: fi_plan_bf16_h64_softcap_delete,
            plan_set_io: fi_plan_bf16_h64_softcap_set_io,
            replan: fi_plan_bf16_h64_softcap_replan,
            run: fi_run_bf16_h64_softcap,
        }),
        (FiDType::Bf16, 128, false) => Some(FiDispatch {
            plan_new: fi_plan_bf16_h128_nosoftcap_new,
            plan_delete: fi_plan_bf16_h128_nosoftcap_delete,
            plan_set_io: fi_plan_bf16_h128_nosoftcap_set_io,
            replan: fi_plan_bf16_h128_nosoftcap_replan,
            run: fi_run_bf16_h128_nosoftcap,
        }),
        (FiDType::Bf16, 128, true) => Some(FiDispatch {
            plan_new: fi_plan_bf16_h128_softcap_new,
            plan_delete: fi_plan_bf16_h128_softcap_delete,
            plan_set_io: fi_plan_bf16_h128_softcap_set_io,
            replan: fi_plan_bf16_h128_softcap_replan,
            run: fi_run_bf16_h128_softcap,
        }),
        (FiDType::Bf16, 256, false) => Some(FiDispatch {
            plan_new: fi_plan_bf16_h256_nosoftcap_new,
            plan_delete: fi_plan_bf16_h256_nosoftcap_delete,
            plan_set_io: fi_plan_bf16_h256_nosoftcap_set_io,
            replan: fi_plan_bf16_h256_nosoftcap_replan,
            run: fi_run_bf16_h256_nosoftcap,
        }),
        (FiDType::Bf16, 256, true) => Some(FiDispatch {
            plan_new: fi_plan_bf16_h256_softcap_new,
            plan_delete: fi_plan_bf16_h256_softcap_delete,
            plan_set_io: fi_plan_bf16_h256_softcap_set_io,
            replan: fi_plan_bf16_h256_softcap_replan,
            run: fi_run_bf16_h256_softcap,
        }),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Workspace sizing
// ---------------------------------------------------------------------------

/// FlashInfer plan workspace sizes in bytes: `(float_ws, int_ws)`.
///
/// Ported verbatim from the prior-session benchmark
/// `attn_bench.cu:50-57`. The planner reads these numbers from the caller
/// and sizes its `AlignedAllocator` inside the workspace — drift here
/// silently trashes memory. On L4 (num_sm=58, head_dim=64,
/// num_kv_heads=8): float ≈ 67 MB, int ≈ 12 MB.
pub fn workspace_bytes(num_sm: i32, head_dim: usize, num_kv_heads: usize) -> (usize, usize) {
    const CTA_TILE_SUM: u64 = 128 + 16;
    let ks: u64 = 4 * (num_sm as u64) * CTA_TILE_SUM;
    let float_ws =
        (ks * 2 * head_dim as u64 * num_kv_heads as u64 + ks * 4 * num_kv_heads as u64) * 2;
    let int_ws = (11u64 * 2 * 65536 * 4 + 2 * ks * 4 + 4096) * 2;
    (float_ws as usize, int_ws as usize)
}

// ---------------------------------------------------------------------------
// Plan cache — single-slot, keyed by (num_tokens, sk_bucket, cfg).
// ---------------------------------------------------------------------------

/// Workspace-level key. Only fields that affect workspace SIZING
/// (cudaMalloc'd buffers) go here. Scheduling-level params (seqlen_k,
/// num_pages) are handled by `replan()` which reuses existing
/// workspaces.
#[derive(Clone, Copy, PartialEq, Eq)]
struct PlanKey {
    cfg: FlashInferConfig,
}

/// One slot of the multi-slot plan cache: a single `(cfg)` plan with
/// its own device workspaces and per-forward-pass replan memo.
struct PlanSlot {
    handle: *mut c_void,
    dispatch: FiDispatch,
    /// `(seq_len, seqlen_k, num_pages)` the plan's scheduling was last
    /// computed for. Per-forward-pass memoization: `replan()` skips the
    /// planner call when the key matches, so layer 0 of a pass pays the
    /// planner cost once and layers 1..N reuse the fresh `int_ws_d`.
    /// Reset to `None` when the slot is first created.
    ///
    /// All three dims must be keyed: `seq_len` is FI's total Q-token
    /// count (≠ batch_size for prefill, = batch_size for decode) and
    /// drives the q_indptr / work_indptr scheduling. A memo keyed only
    /// on `(seqlen_k, num_pages)` reused a prefill plan's scheduling
    /// for 2048 Q tokens on a decode pass with 56 Q tokens — the
    /// captured kernel then read q_indptr entries beyond the actual Q
    /// tensor's bounds, corrupting memory and surfacing as a cublas
    /// `CUBLAS_STATUS_EXECUTION_FAILED` on the next GEMM of the
    /// captured forward. Per-layer dedup is still intact: every layer
    /// of a forward pass sees the same three values.
    last_replan: Option<(u32, u32, u32)>,
}

/// Multi-slot FlashInfer plan cache. Each `(head_dim, softcap)` `cfg`
/// gets its own persistent `PlanSlot` (one cudaMalloc'd
/// float_ws/int_ws pair, one shim handle). Slots are NEVER evicted
/// during a session — only on `clear()` (worker teardown) / `Drop`.
///
/// Why multi-slot: when a CUDA-graph capture pass picks
/// `cfg=nosoftcap` and a subsequent eager call picks `cfg=softcap`
/// (cost-solver alternation across decode shapes), evicting the
/// nosoftcap plan would `cudaFree` workspaces that the captured graph
/// still references via params_1/params_2 baked into kernel args at
/// capture time. Replay then reads freed memory →
/// `CUDA_ERROR_ILLEGAL_ADDRESS`. Holding one slot per cfg sidesteps
/// the whole class of bug. ~93 MB per slot × 6 cfgs (3 head_dim × 2
/// softcap) = ~558 MB worst case, easy on H100/L40S.
///
/// `current` is a cursor: the most recently `ensure()`d cfg, used by
/// `replan()/set_io()/run()` so the existing call sites don't need to
/// thread a cfg through every call.
pub struct FlashInferPlanCache {
    /// Linear vec of slots — at most 6 entries for the current
    /// FLASHINFER_CONFIG_SET (3 head_dim × 2 softcap), so linear
    /// search beats a HashMap and lets `new()` stay `const`
    /// (HashMap::new() is not const because RandomState needs
    /// runtime entropy, and the cache is held in a static Mutex).
    slots: Vec<(PlanKey, PlanSlot)>,
    /// Cursor: most recently `ensure()`d cfg. `replan/set_io/run`
    /// operate on the corresponding slot.
    current: Option<PlanKey>,
}

// Handle is a heap-allocated opaque pointer owned by this cache; the FI
// shim does not spawn threads or hold host locks, so Send is sound.
unsafe impl Send for FlashInferPlanCache {}

impl Default for FlashInferPlanCache {
    fn default() -> Self {
        Self::new()
    }
}

impl FlashInferPlanCache {
    pub const fn new() -> Self {
        Self {
            slots: Vec::new(),
            current: None,
        }
    }

    fn slot_idx(&self, key: &PlanKey) -> Option<usize> {
        self.slots.iter().position(|(k, _)| k == key)
    }

    fn current_slot(&self) -> Option<&PlanSlot> {
        let key = self.current.as_ref()?;
        self.slots.iter().find_map(|(k, s)| (k == key).then_some(s))
    }

    fn current_slot_mut(&mut self) -> Option<&mut PlanSlot> {
        let key = self.current?;
        self.slots
            .iter_mut()
            .find_map(|(k, s)| (*k == key).then_some(s))
    }

    /// Ensure workspaces are allocated for `cfg`. Only rebuilds (plan_delete
    ///     + plan_new) when the config changes or on first call. Does NOT run the
    ///     scheduler — callers MUST follow up with [`Self::replan`] (once per
    ///     forward step) then [`Self::set_io`] + [`Self::run`] per layer.
    ///
    /// # Safety
    /// All pointer args must be valid at the time of the call if a rebuild
    /// occurs (first call or cfg change). Subsequent `set_io` will override.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn ensure(
        &mut self,
        cfg: FlashInferConfig,
        num_tokens: u32,
        _sk_bucket: u32,
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        kv_indices: *const i32,
        o: *mut c_void,
        seqlen_k: i32,
        num_qo_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        page_size: i32,
        num_pages: i32,
        target_num_clusters: i32,
        float_ws_bytes: usize,
        int_ws_bytes: usize,
        sm_scale: f32,
        logits_soft_cap: f32,
        stream: CUstream,
    ) -> Option<FiDispatch> {
        let want = PlanKey { cfg };
        // Already have a slot for this cfg: just move the cursor.
        // Do NOT touch other slots — captured graphs may still hold
        // their handles' workspace pointers in baked kernel args.
        if self.slot_idx(&want).is_some() {
            self.current = Some(want);
            return self
                .slots
                .iter()
                .find_map(|(k, s)| (*k == want).then_some(s.dispatch));
        }

        // Need a new slot. Allocate via the FI shim's plan_new (which
        // also runs the planner once for the supplied num_tokens /
        // seqlen_k — the per-step replan() will overwrite that as
        // forward passes progress).
        let dispatch = dispatch_for(cfg)?;
        let mut rc: i32 = 0;
        let handle = unsafe {
            (dispatch.plan_new)(
                q,
                k,
                v,
                kv_indices,
                o,
                num_tokens as i32,
                seqlen_k,
                num_qo_heads,
                num_kv_heads,
                head_dim,
                page_size,
                num_pages,
                target_num_clusters,
                float_ws_bytes,
                int_ws_bytes,
                sm_scale,
                logits_soft_cap,
                stream,
                &mut rc as *mut i32,
            )
        };
        if handle.is_null() {
            tracing::error!(rc, "fi_plan_new returned null");
            return None;
        }
        self.slots.push((
            want,
            PlanSlot {
                handle,
                dispatch,
                last_replan: None,
            },
        ));
        self.current = Some(want);
        Some(dispatch)
    }

    /// Rebind the current plan's I/O pointers in-place. Use before each
    /// per-layer [`Self::run`] to supply that layer's `k`/`v` pages and
    /// its fresh `q`/`o` tensors.
    ///
    /// # Safety
    /// A plan must exist (call [`Self::ensure`] first). Pointers must be
    /// valid device memory for the duration of the subsequent `run`.
    pub unsafe fn set_io(
        &self,
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        kv_indices: *const i32,
        o: *mut c_void,
    ) {
        if let Some(slot) = self.current_slot()
            && !slot.handle.is_null()
        {
            unsafe { (slot.dispatch.plan_set_io)(slot.handle, q, k, v, kv_indices, o) };
        }
    }

    /// Re-run the FI scheduler on the existing plan's workspaces with
    /// updated `(seq_len, seqlen_k, num_pages)`. Issues a host-to-device
    /// memcpy of the fresh scheduling data on `stream`. This is the
    /// graph-compatible fast path: call it EAGERLY before `graph_launch`
    /// at each decode step so the replayed kernels read updated
    /// `kv_len`, `kv_indptr`, etc. from `int_ws_d`.
    ///
    /// Skips the planner call entirely when `(seqlen_k, num_pages)`
    /// matches `last_replan` — every attention layer in one forward
    /// pass sees the same values, so without this memo each layer
    /// would queue its own redundant ~12 MB H2D memcpy. When that
    /// memcpy is captured into a CUDA graph, 16 layers × 12 MB = 200
    /// MB of PCIe traffic replays per decode step.
    ///
    /// # Safety
    /// A plan must exist (call [`Self::ensure`] first).
    pub unsafe fn replan(
        &mut self,
        seq_len: i32,
        seqlen_k: i32,
        num_pages: i32,
        stream: CUstream,
    ) -> i32 {
        let want = (seq_len as u32, seqlen_k as u32, num_pages as u32);
        let Some(slot) = self.current_slot_mut() else {
            return -1;
        };
        if slot.last_replan == Some(want) {
            return 0;
        }
        if slot.handle.is_null() {
            return -1;
        }
        let rc =
            unsafe { (slot.dispatch.replan)(slot.handle, seq_len, seqlen_k, num_pages, stream) };
        if rc == 0 {
            slot.last_replan = Some(want);
        }
        rc
    }

    /// Run the currently-cached plan on `stream`. Returns the FI shim's
    /// status code (0 on success).
    ///
    /// # Safety
    /// A plan must have been built via [`Self::ensure`] since the last
    /// [`Self::clear`].
    pub unsafe fn run(&self, stream: CUstream) -> i32 {
        match self.current_slot() {
            Some(s) if !s.handle.is_null() => unsafe { (s.dispatch.run)(s.handle, stream) },
            _ => -1,
        }
    }

    /// Destroy ALL plan handles and reset. Call only on worker
    /// teardown — clearing mid-session would `cudaFree` workspaces
    /// that captured CUDA graphs may still reference (the very bug
    /// this multi-slot cache exists to avoid).
    pub fn clear(&mut self) {
        for (_, slot) in self.slots.drain(..) {
            if !slot.handle.is_null() {
                unsafe { (slot.dispatch.plan_delete)(slot.handle) };
            }
        }
        self.current = None;
    }
}

impl Drop for FlashInferPlanCache {
    fn drop(&mut self) {
        self.clear();
    }
}
