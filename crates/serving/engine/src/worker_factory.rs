// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Backend-neutral worker construction via an `inventory` registry.
//!
//! Each backend (cuda, metal, future spyre) registers a `WorkerFactory`
//! with `inventory::submit!` in [`crate::gpu_worker`]. `create_worker`
//! (in `scratchy-serving-api`) selects the first factory whose
//! [`WorkerFactory::matches`] accepts the device string, then calls
//! [`WorkerFactory::create`] — no `#[cfg]` arms in the dispatch.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use hf_hub_downloader as hfdl;
use scratchy_core_config::CudaGraphMode;
use scratchy_core_model::weight::HfModelConfig;

use tracing::info;

use crate::error::{ExecutorError, ExecutorResult};
use crate::worker::Worker;

/// Configuration for a `CudaWorker` / `MetalWorker`.
///
/// Backend-neutral: lives in this neutral module (not the cuda|metal-gated
/// `gpu_worker`) so the dispatch path — the `WorkerFactory` trait below and
/// `create_worker` — compiles for every target, including ones without a GPU.
#[derive(Debug, Clone)]
pub struct WorkerCreateConfig {
    /// Path to a local model directory, or a HuggingFace model ID.
    pub model_path: String,
    /// Data type for model weights: "auto", "f16", "bf16".
    pub dtype: String,
    /// Optional HuggingFace token for gated models.
    pub hf_token: Option<String>,
    /// KV cache block size in tokens (must match the scheduler's block size).
    pub block_size: usize,
    /// GPU device index.
    pub device_id: i32,
    /// Skip CUDA graph capture (--enforce-eager).
    pub enforce_eager: bool,
    /// CUDA graph mode: controls piecewise vs monolithic graph capture.
    pub cuda_graph_mode: CudaGraphMode,
    /// Maximum tokens per scheduler iteration (controls arena pre-sizing).
    pub max_num_batched_tokens: usize,
    /// Maximum concurrently-resident sequences (scheduler
    /// `max_num_seqs`). Sizes the Gated-DeltaNet state pool — one
    /// recurrent-state slot per resident sequence for hybrid arches
    /// (Qwen3.5 / Qwen3-Next); unused by non-hybrid arches.
    pub max_num_seqs: usize,
    /// Batch sizes to capture as CUDA graphs (sorted, deduplicated).
    pub cuda_graph_sizes: Vec<usize>,
    /// Run cublasLt algorithm benchmarking during warmup (--cublas-autotune).
    pub cublas_autotune: bool,
    /// Fraction of GPU memory to use (0.0-1.0). Used to compute KV cache budget.
    pub gpu_memory_utilization: f64,
    /// Pooling strategy: "auto", "last", "cls", "mean".
    pub pooling_strategy: String,
    /// Whether the worker runs in pooling mode (--runner pooling).
    pub is_pooling: bool,
    /// Tensor parallelism rank (0 = single GPU / rank 0).
    pub tp_rank: usize,
    /// Tensor parallelism world size (1 = no TP).
    pub tp_world_size: usize,
    /// Pipeline parallelism rank (0 = first stage).
    pub pp_rank: usize,
    /// Pipeline parallelism size (1 = no PP).
    pub pp_size: usize,
    /// Optional GGUF file name for HF Hub download (e.g. "model-Q4_K_M.gguf").
    pub gguf_file: Option<String>,
    /// Optional LoRA adapter path (local directory or HF repo ID).
    pub lora_adapter: Option<String>,
    /// KV cache data type: "auto" (use model dtype) or "fp8_e4m3".
    pub kv_cache_dtype: String,
    /// Compute KV scales dynamically from the first forward pass.
    pub calculate_kv_scales: bool,
    /// EOS token IDs for seal-pad processor (from model config).
    pub eos_token_ids: Vec<u32>,
    /// Runtime `max_model_len` (CLI `--max-model-len`). `None` falls
    /// back to `hf_config.max_position_embeddings` at load time.
    /// Threaded into scratchy's `try_load` so Phi-3 LongRoPE can
    /// decide `use_long_rope = max_model_len > original_max_pos`
    /// the same way Python vLLM does at init time.
    pub max_model_len: Option<usize>,
    /// Optional draft-model path / HF repo ID for speculative decoding.
    /// `Some(_)` triggers a second model + KV pool load inside the same
    /// worker after the target loads. `None` = no draft.
    pub draft_model_path: Option<String>,
    /// Optional dtype override for the draft model's weights ("auto",
    /// "bfloat16", "float16", …). `None` inherits the target's dtype.
    /// Currently passed through but the metal path always coerces to
    /// bf16 to match the target — accepted for CLI parity with Python.
    pub draft_model_dtype: Option<String>,
}

/// Result of worker creation: the worker plus metadata needed for init.
///
/// The trailing `usize` is the KV cache element size in bytes (e.g. 2 for
/// F16/BF16, 4 for F32). The extract methods (`hf_config`, `model_dir`,
/// `resolved_dtype_elem_bytes`) are inherent on the concrete workers (not
/// on the `Worker` trait), so the factory returns the full tuple rather
/// than a bare `Box<dyn Worker>`.
pub type WorkerCreationResult = (Box<dyn Worker>, HfModelConfig, Option<PathBuf>, usize);

/// A progress callback the factory may wire into the worker during load.
pub type ProgressCallback = std::sync::Arc<dyn Fn(&str) + Send + Sync>;

/// A backend's worker constructor, registered via `inventory::submit!`.
///
/// Exactly one backend factory compiles per build (cuda XOR metal — they
/// are `#[cfg]`-mutex), so `matches` overlap on `"auto"` carries no
/// runtime ambiguity. A future spyre factory would match `"spyre"`,
/// disjoint from both.
pub trait WorkerFactory: Sync {
    /// Whether this backend handles the given device string.
    fn matches(&self, device: &str) -> bool;

    /// Construct → init → load → extract for this backend, returning the
    /// worker plus the metadata `initialize_core` needs.
    fn create(
        &self,
        cfg: WorkerCreateConfig,
        progress: Option<ProgressCallback>,
    ) -> anyhow::Result<WorkerCreationResult>;

    /// Total device memory (bytes) and device name for the active backend, if
    /// it can be queried without a constructed worker (used to pick batch-size
    /// defaults before model load). `None` when the backend can't report it.
    fn device_total_bytes_and_name(&self) -> Option<(u64, String)> {
        None
    }

    /// Recommended max working-set size (bytes) for the active backend, if the
    /// backend exposes one (e.g. Metal's `recommendedMaxWorkingSetSize`).
    fn recommended_working_set_size(&self) -> Option<u64> {
        None
    }

    /// Current `(free, total)` device memory in bytes, if queryable (used by the
    /// `/gpu_memory` test endpoint). `None` when the backend can't report it.
    fn device_memory_free_total(&self) -> Option<(u64, u64)> {
        None
    }

    /// Whether this backend supports the engine's cross-request prefix cache.
    /// `true` for the paged-KV GPU backends (cuda/metal). A backend whose worker
    /// keeps a per-request KV cache (spyre — no shared/paged KV) cannot honor a
    /// scheduler-reported cached prefix it has no KV for, so it returns `false`
    /// and the engine disables prefix caching for it.
    fn supports_prefix_caching(&self) -> bool {
        true
    }

    /// THE BLOCK SIZE THIS BACKEND'S KV IS ADDRESSED BY, when the backend — not the CLI — is what
    /// decides it. `None` = the configured `--block-size` stands, which is every budget-sized GPU pool.
    ///
    /// ⛔ ASKED HERE BECAUSE THE ANSWER IS A BAKED CONSTANT OF THE POOL, and the engine cannot hold a
    /// copy of it. `effective_block_size` used to answer with a literal `64` under
    /// `#[cfg(feature = "spyre")]`, describing a pool (`[nblk, 64, nkv, hd]`) that has since been
    /// replaced by one whose page is `PagedKvPool::PAGE_SLOTS` — two spellings of one quantity, four
    /// crates apart, with a comment as the only thing claiming they agreed. The scheduler computes
    /// `slot = block_id * block_size + offset` and the worker resolves `block_id` to a pool page, so a
    /// disagreement is not a policy difference: it is the scheduler and the worker naming different
    /// cells for one token, silently.
    fn required_block_size(&self) -> Option<scratchy_core_common::KvBlockTokens> {
        None
    }
}

inventory::collect!(&'static dyn WorkerFactory);

/// Sink for byte-level model-download progress, so a caller that owns the
/// terminal (the ratatui TUI) can render its own progress instead of letting
/// the default `indicatif` bars draw to stderr over its alt-screen.
///
/// Methods are called from the parallel shard-download threads (up to 8
/// concurrent), keyed by shard filename — implementors must be internally
/// synchronized. The `Debug` supertrait lets an `Option<Arc<dyn
/// DownloadObserver>>` sit inside the `#[derive(Debug)]` `VllmConfig`.
pub trait DownloadObserver: Send + Sync + std::fmt::Debug {
    /// A shard's download started; `total_bytes` is its full size.
    fn on_start(&self, shard: &str, total_bytes: usize);
    /// `delta_bytes` more bytes of `shard` have landed on disk.
    fn on_advance(&self, shard: &str, delta_bytes: usize);
    /// `shard` finished downloading.
    fn on_finish(&self, shard: &str);
}

/// Per-shard progress sink.
///
/// One enum (rather than two generic instantiations) keeps a single type
/// across the `std::thread::scope` closures. `Bar` is the CLI/server path
/// (indicatif → stderr); `Observed` forwards to a caller-supplied
/// [`DownloadObserver`] (the TUI path).
enum ShardProgress {
    Bar(indicatif::ProgressBar),
    Observed {
        observer: Arc<dyn DownloadObserver>,
        shard: String,
    },
}

impl hfdl::Progress for ShardProgress {
    fn init(&mut self, total: u64, filename: &str) {
        match self {
            ShardProgress::Bar(bar) => {
                bar.set_length(total);
                bar.set_message(filename.to_string());
            }
            ShardProgress::Observed { observer, shard } => observer.on_start(shard, total as usize),
        }
    }

    fn update(&mut self, delta: u64) {
        match self {
            ShardProgress::Bar(bar) => bar.inc(delta),
            ShardProgress::Observed { observer, shard } => {
                observer.on_advance(shard, delta as usize)
            }
        }
    }

    fn finish(&mut self) {
        match self {
            ShardProgress::Bar(bar) => bar.finish(),
            ShardProgress::Observed { observer, shard } => observer.on_finish(shard),
        }
    }
}

/// Fetch a single repo file, cache-aware, reporting to `observer` when there
/// is one.
///
/// `download` is cache-first on its own, so this no longer has to check the
/// cache separately before asking for progress — hf-hub needed that dance
/// because its `download_with_progress` re-downloaded unconditionally.
fn fetch_file(
    repo: &hfdl::Repo<'_>,
    filename: &str,
    observer: &Option<Arc<dyn DownloadObserver>>,
) -> Result<PathBuf, hfdl::Error> {
    match observer {
        Some(obs) => repo.download(
            filename,
            &mut ShardProgress::Observed {
                observer: obs.clone(),
                shard: filename.to_string(),
            },
        ),
        None => repo.get(filename),
    }
}

/// Resolve model path: local dir, local GGUF file, or HF download.
///
/// Returns a `PathBuf` that is either:
/// - A directory containing safetensors + config.json (normal path)
/// - A `.gguf` file path (GGUF path — load_model detects this)
///
/// Resolve a model identifier to a local path. Single canonical
/// implementation shared by every Worker `load_model` body and by
/// scratchy-serving-api's parallel-tokenizer hoist.
///
/// Three cases:
/// 1. Local `.gguf` file path → returns as-is.
/// 2. Local directory → returns as-is.
/// 3. HuggingFace Hub model ID:
///    - First check the on-disk cache (network-free lookup).
///      Skips the ~100ms TLS handshake + ETag probe the Api path runs
///      even for fully cached models. MLX doesn't pay this cost.
///    - On cache miss, fall through to the Api path: download
///      config.json + tokenizer + safetensors shards (parallel, up
///      to 8 concurrent), or auto-detect a `.gguf` for GGUF repos.
///
/// Used to live duplicated in `gpu_worker_base.rs`; consolidated
/// here so the Hub plumbing (parallel shard download with progress
/// bars, GGUF auto-detect, etc.) has one home.
///
/// `observer`: when `Some`, safetensors-shard download progress is forwarded
/// to it (and the stderr `indicatif` bars are suppressed) so a terminal-owning
/// caller — the TUI — can render progress itself. `None` keeps the default
/// stderr bars (CLI / server).
pub fn resolve_model_path(
    model_path: &str,
    hf_token: Option<&str>,
    gguf_file: Option<&str>,
    observer: Option<Arc<dyn DownloadObserver>>,
) -> ExecutorResult<PathBuf> {
    let path = Path::new(model_path);

    // Local .gguf file.
    if path.is_file() && path.extension().is_some_and(|e| e == "gguf") {
        return Ok(path.to_path_buf());
    }

    // Local directory.
    if path.is_dir() {
        return Ok(path.to_path_buf());
    }

    // With an observer (the TUI), suppress stderr progress bars for every
    // fetch — config.json, tokenizer, and single-file weights alike — so
    // nothing scribbles over the alt-screen. The big downloads are re-routed
    // through the observer below.
    let mut builder = hfdl::Client::builder().retries(HF_MAX_RETRIES);
    if let Some(token) = hf_token {
        builder = builder.token(Some(token.to_string()));
    }
    let client = builder.build();
    let repo = client.model(model_path);

    // Network-free local-cache fast path: resolve refs/<revision> →
    // snapshots/<commit>/<file> on disk without any HTTP. If `config.json`
    // is already there, the model has been pulled before; return its dir
    // directly. The network path would otherwise burn ~100ms on a TLS
    // handshake + ETag probe for a model we already have locally.
    if gguf_file.is_none()
        && let Some(config_path) = repo.cached("config.json")
        && let Some(model_dir) = config_path.parent().map(|p| p.to_path_buf())
    {
        // `config.json` being present is NOT proof the model is fully
        // cached: an interrupted pull (network timeout mid-download)
        // commonly leaves config.json + the index on disk but only
        // some shards. Returning here would hand a hole-y dir to the
        // weight loader, which dies deep in `mmap` with a bare "No
        // such file or directory" and zero context. Validate that
        // every weight file is actually present before taking the
        // network-free path; otherwise fall through to the download
        // path below, which resumes the missing shards.
        if weights_present(&model_dir) {
            info!(
                "Using cached model: {} (HF cache, no network)",
                model_dir.display()
            );
            // Backfill the Gemma-4 standalone chat template (HF
            // transformers#45205) for caches pulled before it was
            // added to the download set. One-time best-effort network
            // fetch only when the sibling file is genuinely absent —
            // present caches stay fully network-free.
            if !model_dir.join("chat_template.jinja").exists() {
                let _ = repo.get("chat_template.jinja");
            }
            return Ok(model_dir);
        }
        info!(
            "Cached model {} is incomplete (missing weight shards); resuming download",
            model_dir.display()
        );
    }

    info!("Downloading model from HuggingFace Hub: {model_path}");

    // GGUF download: explicit filename or auto-detect from repo.
    let gguf_filename = gguf_file.map(String::from).or_else(|| {
        // Auto-detect: if model name looks like a GGUF repo, find smallest Q4_K_M file.
        if !model_path.to_ascii_uppercase().contains("GGUF") {
            return None;
        }
        let mut gguf_files: Vec<String> = repo
            .files()
            .ok()?
            .into_iter()
            .filter(|f| f.ends_with(".gguf"))
            .collect();
        if gguf_files.is_empty() {
            return None;
        }
        for pattern in &["Q4_K_M", "Q4_K_S", "Q4_K", "Q4_0", "Q8_0"] {
            if let Some(f) = gguf_files.iter().find(|s| s.contains(pattern)) {
                return Some(f.clone());
            }
        }
        gguf_files.sort();
        Some(gguf_files.swap_remove(0))
    });
    if let Some(ref gguf_file) = gguf_filename {
        info!("Downloading GGUF file: {gguf_file}");
        let gguf_path = fetch_file(&repo, gguf_file, &observer).map_err(|e| {
            ExecutorError::WorkerInit(format!("failed to download GGUF {gguf_file}: {e}"))
        })?;
        let _ = repo.get("tokenizer.json");
        let _ = repo.get("tokenizer_config.json");
        // Gemma-4 (HF transformers#45205) ships the chat template as a
        // standalone sibling file rather than embedded in
        // tokenizer_config.json — best-effort, most repos lack it.
        let _ = repo.get("chat_template.jinja");
        return Ok(gguf_path);
    }

    let config_path = repo
        .get("config.json")
        .map_err(|e| ExecutorError::WorkerInit(format!("failed to download config.json: {e}")))?;
    let model_dir = config_path.parent().unwrap().to_path_buf();

    let _ = repo.get("tokenizer.json");
    let _ = repo.get("tokenizer_config.json");
    // Gemma-4 (HF transformers#45205) ships the chat template as a
    // standalone sibling file rather than embedded in
    // tokenizer_config.json — best-effort, most repos lack it.
    let _ = repo.get("chat_template.jinja");

    // "This repo has no single-file weights" and "the single-file download
    // failed" are different facts, and only the first means we should go
    // looking for shards. Branching on the typed `NotFound` keeps a network
    // error, a full disk or a bad digest from being reported as
    // "no safetensors weights found" — the one cause they are not.
    match fetch_file(&repo, "model.safetensors", &observer) {
        Ok(_) => return Ok(model_dir),
        Err(e) if e.is_not_found() => {}
        Err(e) => {
            return Err(ExecutorError::WorkerInit(format!(
                "failed to download model.safetensors for {model_path}: {e}"
            )));
        }
    }
    if let Ok(index_path) = repo.get("model.safetensors.index.json") {
        let index = scratchy_core_model::weight::SafeTensorsIndex::from_file(&index_path)
            .map_err(|e| ExecutorError::WorkerInit(format!("failed to parse index: {e}")))?;
        let sorted_shards = index.shard_files();
        let total = sorted_shards.len();

        let needed: Vec<&String> = sorted_shards
            .iter()
            .filter(|s| !model_dir.join(s).exists())
            .collect();

        // Fail fast on a full disk rather than starting a doomed multi-GB
        // download (and burning the downloader's retry budget on ENOSPC, which
        // never recovers). A disk-full download is what leaves the partial
        // cache the fast path above now detects and resumes.
        index
            .ensure_disk_space(&model_dir)
            .map_err(|e| ExecutorError::WorkerInit(format!("cannot download {model_path}: {e}")))?;

        if needed.is_empty() {
            info!("All {total} shard files already cached");
        } else {
            info!(
                "Downloading {} of {total} shard files (up to 8 in parallel)",
                needed.len()
            );

            // No `MultiProgress` (and thus no stderr bars) when an observer
            // owns the display; the caller renders progress from its own sink.
            let multi = observer.is_none().then(indicatif::MultiProgress::new);
            // Shards in flight here MULTIPLY with the ranges each download
            // splits itself into. The client's own permit cap
            // (`max_concurrency`, 8) bounds the product, so this stays at 8
            // without opening 64 connections: sharded models keep the
            // request count they always had, and a single-file model — which
            // presents this loop with exactly one item — spends the whole
            // budget on ranges within that file. That is the case hf-hub
            // 0.4.3 downloaded as one sequential stream.
            const MAX_PARALLEL: usize = 8;
            let repo = &repo;
            let multi = &multi;
            let observer = &observer;

            for chunk in needed.chunks(MAX_PARALLEL) {
                let results: Vec<ExecutorResult<()>> = std::thread::scope(|s| {
                    let handles: Vec<_> = chunk
                        .iter()
                        .map(|shard| {
                            let progress = match (multi, observer) {
                                (_, Some(obs)) => ShardProgress::Observed {
                                    observer: obs.clone(),
                                    shard: (*shard).clone(),
                                },
                                (Some(multi), None) => {
                                    ShardProgress::Bar(multi.add(indicatif::ProgressBar::new(0)))
                                }
                                // `multi` is `Some` iff `observer` is `None`
                                // (built above), so this arm is unreachable —
                                // fall back to a hidden bar rather than panic.
                                (None, None) => {
                                    ShardProgress::Bar(indicatif::ProgressBar::hidden())
                                }
                            };
                            s.spawn(move || {
                                let mut progress = progress;
                                repo.download(shard, &mut progress)
                                    .map(|_| ())
                                    .map_err(|e| {
                                        ExecutorError::WorkerInit(format!(
                                            "failed to download {shard}: {e}"
                                        ))
                                    })
                            })
                        })
                        .collect();
                    handles.into_iter().map(|h| h.join().unwrap()).collect()
                });
                for result in results {
                    result?;
                }
            }
        }
        return Ok(model_dir);
    }

    // Reaching here means the Hub reported neither layout present, which is
    // now the only way to get this message.
    Err(ExecutorError::WorkerInit(format!(
        "no safetensors weights found for {model_path} \
         (neither model.safetensors nor model.safetensors.index.json)"
    )))
}

/// Attempts per file before giving up, including the first.
///
/// The budget is per FILE, shared across every range that file is split
/// into, so this is five for the whole download rather than five each.
/// Retries are cheap: a range restarts from the offset its resume log
/// recorded, so a reset costs the bytes since the last checkpoint and never
/// re-fetches completed ones.
const HF_MAX_RETRIES: usize = 5;

/// Returns true iff every safetensors weight file this model references
/// is present on disk in `model_dir`. Used to distinguish a fully-cached
/// model from a partially-downloaded one (interrupted pull): the latter
/// must re-enter the resuming download path rather than be handed to the
/// weight loader.
///
/// Safetensors-only by design — and that is the safe default for every
/// other format:
///   * The partial-cache crash is *structural* to multi-file safetensors
///     (config.json + index + N shards, where config can exist while some
///     shards are still missing). It is the only weights layout
///     `GpuWeights::from_dir` loads from a directory.
///   * GGUF is a single file that downloads to `.incomplete` and renames
///     atomically — it is wholly present or absent, never partial — and is
///     resolved as a file path, not via this dir fast path. Anything that
///     is not complete safetensors returns false here and falls through to
///     the cache-first, retrying download path. The only unsafe direction
///     would be a false *positive* (claiming complete when it is not), which
///     this never does for any format.
fn weights_present(model_dir: &Path) -> bool {
    // Single-file model.
    if model_dir.join("model.safetensors").exists() {
        return true;
    }
    // Sharded model: every shard named in the index must exist.
    let index_path = model_dir.join("model.safetensors.index.json");
    if !index_path.exists() {
        return false;
    }
    match scratchy_core_model::weight::SafeTensorsIndex::from_file(&index_path) {
        Ok(index) => index
            .shard_files()
            .iter()
            .all(|shard| model_dir.join(shard).exists()),
        Err(_) => false,
    }
}
