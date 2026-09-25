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

use scratchy_core_config::CudaGraphMode;
use scratchy_core_model::hub;
use scratchy_core_model::weight::HfModelConfig;

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

pub use scratchy_core_model::hub::DownloadObserver;

/// Resolve a model identifier to a local path. Single canonical
/// implementation shared by every Worker `load_model` body and by
/// scratchy-serving-api's parallel-tokenizer hoist.
///
/// Returns a `PathBuf` that is either:
/// - A directory containing safetensors + config.json (normal path)
/// - A `.gguf` file path (GGUF path — load_model detects this)
///
/// Three cases:
/// 1. Local `.gguf` file path → returns as-is.
/// 2. Local directory → returns as-is.
/// 3. HuggingFace Hub model ID → [`hub::download`], the one routine for it,
///    which `scr model pull` calls too.
///
/// `observer`: when `Some`, weight download progress is forwarded to it (and
/// the stderr bars are suppressed) so a terminal-owning caller — the TUI —
/// can render progress itself. `None` draws stderr bars (CLI / server).
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

    let gguf = gguf_file.map_or(hub::Gguf::Auto, hub::Gguf::File);
    hub::download(model_path, hf_token, gguf, observer)
        .map_err(|e| ExecutorError::WorkerInit(e.to_string()))
}
