// SPDX-License-Identifier: Apache-2.0
//! `SpyreWorker` — PLUMBING. The struct, its construction, and the factory the
//! engine resolves a `--device spyre` worker through. Nothing else.
//!
//! ⛔ THE WORKER IS NOT WHERE A MODEL LIVES. It looks the model up in
//! `inventory` (the `#[forward]` macro's `ScratchyArchRegistration`) and invokes
//! it; every model fact reaches it as GENERATED data. The jobs that used to sit
//! in this one file are now the things they are:
//!
//! | module | what it is |
//! |---|---|
//! | [`crate::spyre_types`] | the nouns — `BundleMeta`, `Loaded`, `ReqState` |
//! | [`crate::spyre_load`] | resolve the generated wiring, stage the generated weights |
//! | [`crate::spyre_forward`] | play the forward tape — one chunk or one step |
//! | [`crate::spyre_pool`] | cut the paged KV pool, bind a request's pages |
//! | [`crate::spyre_exec`] | the `Worker` trait: scheduling and the step loop |
//!
//! It was 7,148 lines when they were all one file. The split is not cosmetic:
//! `spyre_forward` cannot reach into scheduling state and `spyre_exec` cannot
//! stage an activation, so the boundaries the architecture claims are boundaries
//! the compiler now enforces.

// Force the linker to keep every compiled arch crate (and thus its
// `ScratchyArchRegistration` `inventory::submit!`) — otherwise the model crates,
// referenced only via the inventory registry, get dead-stripped and `try_load`
// finds an empty registry. (The cuda/metal path does the same in `gpu_worker`,
// which isn't compiled under spyre.)
extern crate scratchy_models as _;

use std::collections::HashMap;
use std::path::Path;

use scratchy_core_model::weight::HfModelConfig;
use scratchy_subtile::sdsc_abstract::PagedKvPool;
// ⭐ THE CARD PATH NO LONGER PARSES A MANIFEST. `Manifest` survives only for
// the KTIR-emulator session, whose `new_multi` takes `&Manifest` to thread its
// HBM buffers. Under `sendnn` every fact it carried comes from the GENERATED
// `SUPERDSC_WIRINGS` static instead, so the type is not even in scope.
#[cfg(not(feature = "sendnn"))]
use scratchy_target_spyre::manifest::Manifest;
// `--target sendnn` swaps the KTIR emulator runner for the on-silicon sendnn
// runner; the bundle type + session type are cfg-selected, everything else
// (weight load, dynamic sources, KV loop, sampling) is shared.
use crate::spyre_types::*;
#[cfg(not(feature = "sendnn"))]
use scratchy_target_spyre::manifest::KtirBundle;

#[cfg(not(feature = "sendnn"))]
use scratchy_target_spyre::runner::SpyreSession;

use crate::error::ExecutorError;
use crate::worker::{Worker, WorkerConfig};

pub(crate) fn werr(msg: impl Into<String>) -> ExecutorError {
    ExecutorError::WorkerExecution(msg.into())
}

/// One program's inputs to [`SpyreSession::new_multi`]: its node `(func, mlir)`
/// pairs, its manifest, and its extra result ids. The shared session is built
/// over one of these per phase (prefill + one per decode cap bucket).
#[cfg(not(feature = "sendnn"))]
pub(crate) type ProgramInput<'a> = (&'a [(&'a str, &'a str)], &'a Manifest, &'a [usize]);

/// Host-only KTIR serving worker.
pub struct SpyreWorker {
    pub(crate) config: WorkerConfig,
    /// HF repo id or local dir (resolved to on-disk weights + `config.json`).
    pub(crate) model_path: String,
    pub(crate) is_shutdown: bool,
    pub(crate) model: Option<Loaded>,
    pub(crate) requests: HashMap<String, ReqState>,
    /// ⭐ THE HOST'S PER-REQUEST BLOCK TABLE, IN THE STRUCT THE ENGINE ALREADY DEFINES FOR IT.
    ///
    /// `InputBatch` (`crates/serving/engine/src/input_batch.rs`) is backend-neutral and is what cuda and
    /// metal store their block tables in; its `update_blocks` carries the rule a private copy in this
    /// worker got wrong — *"The scheduler ships the FULL table each step, so this REPLACES (not
    /// appends)"*. This path used to keep its own `Vec<usize>` per request (and before that, pages the
    /// worker ALLOCATED ITSELF from a free list, which is what made a prefix-cache hit meaningless).
    ///
    /// ⛔ ONLY THE BLOCK TABLE, FOR NOW. The rest of `ReqState` — tokens, `n_computed`, the KV row, and
    /// `KvHistory` — is not migrated: the first three duplicate `InputBatch` fields and SHOULD move, and
    /// `KvHistory` is genuinely this backend's (a pool whose batched write appends every row at one slot
    /// has no counterpart on cuda/metal). Migrating the rest is mechanical and belongs in its own change,
    /// not bundled with a correctness fix.
    pub(crate) input_batch: scratchy_serving_engine::input_batch::InputBatch,
    /// HOW MANY REQUESTS THIS RUN MAY ADMIT AT ONCE — `--max-num-seqs`, as the scheduler was configured.
    ///
    /// ⛔ IT IS HERE BECAUSE THE POOL'S HOLE RESERVE IS CHARGED AGAINST IT. `PoolPartition` holds back
    /// `rows * HOLE_PAGES_PER_ROW + 1` pages for the masked holes a LAUNCH's rows write into, and `rows`
    /// used to be the widest rung the ladder bakes (32) regardless of what the run would admit — 65 of a
    /// 136-page pool, against launches a `--max-num-seqs 4` run cannot perform. This is the decided value
    /// that replaces the guess; [`PoolRows::for_admission`] rounds it UP to a ladder rung.
    ///
    /// `None` when the cap is unusable (zero), in which case the pool falls back to the widest rung — a
    /// pool cut for FEWER rows than a launch binds would address slots it never seated, so the fallback
    /// direction is the safe one.
    pub(crate) admitted: Option<scratchy_subtile::sdsc_abstract::AdmittedRequests>,
    /// HOW DEEP ONE REQUEST MAY GO — `--max-model-len`, as the CLI gave it. `None` falls back to the
    /// model's own `max_position_embeddings`.
    ///
    /// ⛔ THE OTHER HALF OF WHAT SIZES THE POOL, and it did not exist here until the pool stopped being
    /// sized by a constant. See [`SpyreWorker::declare_workload`] and
    /// [`scratchy_subtile::sdsc_abstract::PoolDemand`].
    pub(crate) declared_context: Option<usize>,
    /// WHAT FRACTION OF THE CARD THIS RUN MAY SPEND — `--gpu-memory-utilization`, as the CLI gave it.
    ///
    /// ⛔ SPYRE IGNORED THIS FLAG ENTIRELY, because its KV budget was a constant and a constant has no
    /// fraction to take. cuda and metal both size from `total_memory * utilization`, so the flag meant
    /// something on two backends out of three.
    pub(crate) gpu_memory_utilization: f64,
}

impl SpyreWorker {
    pub fn new(config: WorkerConfig, model_path: String) -> Self {
        Self {
            config,
            model_path,
            is_shutdown: false,
            model: None,
            requests: HashMap::new(),
            input_batch: scratchy_serving_engine::input_batch::InputBatch::new(),
            admitted: None,
            declared_context: None,
            // Matches the CLI's own `--gpu-memory-utilization` default; `declare_workload` overwrites
            // it with what the run actually asked for.
            gpu_memory_utilization: 0.9,
        }
    }

    /// Tell this worker THE WHOLE DECLARED WORKLOAD — `--max-num-seqs` and `--max-model-len` — before
    /// `load_model` sizes the pool from it.
    ///
    /// Separate from `new` because `WorkerConfig` is the engine's shared struct and these are scheduler
    /// numbers that only this backend's pool arithmetic consumes — adding them there would put fields on
    /// cuda's and metal's config that neither reads. The factory calls this immediately after `new`.
    ///
    /// ⛔ ONE SETTER FOR BOTH, BECAUSE THE POOL NEEDS BOTH AND NEITHER IS RECOVERABLE FROM THE OTHER.
    /// `PoolDemand` is `depth × width`: with only the width (which is all this used to take) the depth
    /// fell back to a constant, and a constant depth is what made the pool's size unanswerable against
    /// what the caller asked for. Two setters would let a future caller set one and leave the other at a
    /// default, silently restoring exactly that.
    pub fn declare_workload(
        &mut self,
        max_num_seqs: usize,
        max_model_len: Option<usize>,
        gpu_memory_utilization: f64,
    ) {
        self.admitted = scratchy_subtile::sdsc_abstract::AdmittedRequests::new(max_num_seqs);
        self.declared_context = max_model_len;
        self.gpu_memory_utilization = gpu_memory_utilization;
    }

    /// Parsed HF config — available after [`Worker::load_model`]. `create_worker`
    /// needs it for engine-side scheduling/cache sizing.
    pub fn hf_config(&self) -> Option<&HfModelConfig> {
        self.model.as_ref().map(|m| &m.hf_config)
    }

    /// On-disk model dir (weights + `config.json`), available after load.
    pub fn model_dir(&self) -> Option<&Path> {
        self.model.as_ref().map(|m| m.model_dir.as_path())
    }

    /// KV-cache element size: the spyre host path is f32 end-to-end.
    pub fn resolved_dtype_elem_bytes(&self) -> usize {
        4
    }
}

/// DEFAULT-mode single-graph forward: process `toks` ONE token at a time (the
/// proven static-SDPA graph is m=1), growing the KV host-side and masking padding
/// prefix rows via `attn_lenmask`. Each step `p` (absolute position `start+i`):
///   • bind the embedding `t{embed}` `[1, hidden]` of `toks[i]`;
///   • bind cos/sin `t{id}` `[1, width]` (per-position RoPE tables tiled across
///     heads) for every cos/sin source;
///   • bind each layer's prefix-KV `t{id}` `[cap, kv_dim]` = the host KV cache
///     rows `0..p` (positions `0..p-1`), zero-padded to `cap`;
///   • bind `attn_lenmask` `[1,1,cap+1]`: 0 for valid columns `0..p` and the
///     new-token column `cap`, large-negative for the padding columns `p..cap-1`
///     (PROVEN on dd2 — without it the padding rows flood the softmax and the
///     argmax flips, e.g. 1217 vs the golden 7042);
///   • Predict → logits `[1,vocab]` + grown-KV (each `[cap+1, kv_dim]`, layer

/// Spyre / KTIR backend factory. Registered with `inventory::submit!` so the
/// backend-neutral `create_worker` (engine's `WorkerFactory` registry) finds it
/// without naming `SpyreWorker` — the same seam cuda/metal register through. In
/// a spyre-only build (mutually exclusive with cuda/metal) `"auto"` resolves
/// here, so `scr chat` without an explicit `--device` serves on spyre.
pub struct SpyreWorkerFactory;

impl scratchy_serving_engine::worker_factory::WorkerFactory for SpyreWorkerFactory {
    fn matches(&self, device: &str) -> bool {
        // `sendnn` = the on-silicon target (this worker, built with -Fsendnn,
        // routes through sdsc_runner). `spyre` = the KTIR emulator path. `auto`
        // resolves to whichever backend this binary was built with.
        device == "spyre" || device == "sendnn" || device == "auto"
    }

    /// ⭐ TRUE, AND THE THREE THINGS THAT MAKE IT TRUE.
    ///
    /// It was `false` for a year, with a comment saying the pool "cannot support it BY CONSTRUCTION"
    /// because `PageRun::physical(lp) = row * pages_per_row + lp` gave every row an exclusive page
    /// stripe. The striping was real; "by construction" was not — it described a layout WE chose so one
    /// launch could stride the batch, and the device's actual limit is that it has no indirect
    /// addressing. The refusal was a feature switched off so a batch test would pass.
    ///
    /// What a cache hit needs is that the reusing request's keys for the shared prefix ARE the earlier
    /// request's keys. That now holds because:
    ///
    /// 1. **A REQUEST'S PAGES ARE THE SCHEDULER'S BLOCKS.** `install_host_blocks` installs
    ///    `block_ids` / `new_block_ids` through [`BlockTable`], whose only constructor takes host ids —
    ///    so the free-list search that used to fill the map (a second allocator over the scheduler's own
    ///    pool) cannot be written again, and a hit hands both requests the same physical page.
    /// 2. **THE POOL AND THE ALLOCATOR AGREE ON THE PAGE SPACE.** `required_block_size` is the pool's
    ///    `PAGE_SLOTS` and `kv_cache_num_blocks_override` is the pool's page count, so `block_id` IS a
    ///    page and no id can name one the pool lacks.
    /// 3. **THE HOST CACHES ONLY WHAT IS TOKEN-ADDRESSED.** A batched step appends every row at ONE slot,
    ///    so a request shorter than its batch-mates spreads its keys over more slots than it has tokens.
    ///    `ReqState::kv_extent` reports both the span (blocks are allocated for it) and the leading
    ///    contiguous run (only that may be hashed) — see `KvHistory::cacheable_prefix`. Without the
    ///    second half a block covering a hole would promise a later request keys for the wrong tokens.
    ///
    /// ⛔ THE GATE IS A CACHING-ON RUN DIFFED AGAINST CACHING-OFF, not a run that fails to crash: the
    /// original symptom was fluent WRONG output from rows attending an incomplete prefix.
    fn supports_prefix_caching(&self) -> bool {
        true
    }

    /// The pool's page IS the KV block, so the scheduler's `slot = block_id * block_size + offset` and
    /// the worker's page map address the same cells. Sourced from the pool constant — the literal `64`
    /// this used to be in `effective_block_size` described a pool that no longer exists.
    fn required_block_size(&self) -> Option<scratchy_core_common::KvBlockTokens> {
        scratchy_core_common::KvBlockTokens::new(PagedKvPool::PAGE_SLOTS)
    }

    fn create(
        &self,
        cfg: scratchy_serving_engine::worker_factory::WorkerCreateConfig,
        _progress: Option<scratchy_serving_engine::worker_factory::ProgressCallback>,
    ) -> anyhow::Result<scratchy_serving_engine::worker_factory::WorkerCreationResult> {
        use anyhow::Context as _;
        let worker_config = WorkerConfig {
            local_rank: cfg.tp_rank,
            rank: cfg.tp_rank,
            is_driver_worker: cfg.tp_rank == 0,
            distributed_init_method: String::new(),
            hf_token: cfg.hf_token.clone(),
            gguf_file: cfg.gguf_file.clone(),
        };
        let mut worker = SpyreWorker::new(worker_config, cfg.model_path.clone());
        // BEFORE `load_model`, which is where the pool is sized and split: the declared depth × width IS
        // the pool's size, and the admission cap also decides how many pages are reserved for launch holes.
        worker.declare_workload(
            cfg.max_num_seqs,
            cfg.max_model_len,
            cfg.gpu_memory_utilization,
        );
        worker.init_device().context("spyre init_device")?;
        worker.load_model().context("spyre load_model")?;
        let hf_config = worker
            .hf_config()
            .context("spyre model config not available after load")?
            .clone();
        let model_dir = worker.model_dir().map(|p| p.to_path_buf());
        let dtype_elem_bytes = worker.resolved_dtype_elem_bytes();
        Ok((Box::new(worker), hf_config, model_dir, dtype_elem_bytes))
    }
}

inventory::submit!(
    &SpyreWorkerFactory as &dyn scratchy_serving_engine::worker_factory::WorkerFactory
);
