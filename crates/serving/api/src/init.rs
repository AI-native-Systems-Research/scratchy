// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Stack initialization: wires together worker → executor → engine → server.
//!
//! This module provides the programmatic API for initializing the full vLLM
//! inference stack. No CLI dependency needed — just construct a [`VllmConfig`]
//! and call [`initialize_stack`].

// Anchor the worker crate into the link so its `WorkerFactory`
// `inventory::submit!` registrations (CudaWorkerFactory / MetalWorkerFactory
// in scratchy_serving_worker::gpu_worker) are NOT garbage-collected by the
// linker. After the dispatch rewrite, `create_worker` no longer NAMES
// CudaWorker/MetalWorker, so without this explicit reference the registry
// could come up empty at runtime. `create_worker` still bails loudly if it
// does (empty-registry assertion) — this anchor is the belt to that bail's
// suspenders.
extern crate scratchy_serving_worker as _;

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use scratchy_core_config::{CudaGraphConfig, CudaGraphMode, SchedulerConfig, SchedulerPolicy};
use scratchy_core_model::weight::HfModelConfig;
use scratchy_serving_engine::core_client::InprocClient;
use scratchy_serving_engine::engine_core::{EngineCoreConfig, HybridKvConfig};
use scratchy_serving_engine::spec_decode::{
    DraftModelProposerConfig, NgramProposerConfig, ProposerConfig,
};
use scratchy_serving_worker::uniproc::UniProcExecutor;
use scratchy_serving_worker::worker::Worker;
use tracing::info;

use crate::chat_template::ChatTemplate;
use crate::engine::AsyncEngine;
use crate::tokenizer::Tokenizer;

// vllm-mlx has been removed. The metal path lives entirely in
// scratchy-target-metal (`MetalWorker`).
// If you hit `ExecutorError::ArchNotSupported` at runtime, the fix is
// to add the missing arch / quantization overlay to
// `crates/models/arch/configs/<arch>/quantizations.json` (or land
// the matching backend Impl in `crates/scratchy-forward-compiler-macro/src/metal/`)
// — NOT to bring back MLX.

/// Re-export so callers that build a [`VllmConfig`] (e.g. the TUI) can name
/// the observer trait for `download_observer` without depending on
/// `scratchy-serving-engine` directly.
pub use scratchy_serving_engine::worker_factory::DownloadObserver;

/// Configuration for initializing the vLLM inference stack.
///
/// This is the programmatic API — no CLI dependency needed.
#[derive(Debug, Clone, serde::Serialize)]
pub struct VllmConfig {
    /// Model path or HuggingFace model ID (required).
    pub model: String,
    /// Device: "cpu", "cuda:N", "metal", or "auto" (auto-detect best GPU).
    pub device: String,
    /// Weight dtype: "auto", "float16", "bfloat16", "float32".
    pub dtype: String,
    /// Maximum model context length (None = use config.json).
    pub max_model_len: Option<usize>,
    /// Maximum number of concurrent sequences. `None` = unset, resolved by
    /// [`resolve_max_num_seqs`]: the backend's own batched-decode width where it has one, else a
    /// device-appropriate default. `Some(n)` is honoured, capped to what the backend can batch.
    pub max_num_seqs: Option<usize>,
    /// KV cache block size in tokens.
    pub block_size: usize,
    /// Fraction of GPU memory to use for KV cache (0.0–1.0).
    pub gpu_memory_utilization: f64,
    /// HuggingFace token for gated models (excluded from serialization).
    #[serde(skip_serializing)]
    pub hf_token: Option<String>,
    /// Specific GGUF filename to download from a HuggingFace repo.
    pub gguf_file: Option<String>,
    /// Speculative model: `"ngram"` selects the n-gram proposer; any other
    /// value is treated as a draft-model local path or HuggingFace repo ID
    /// (not yet wired). `None` disables speculative decoding.
    pub speculative_model: Option<String>,
    /// Number of speculative tokens to propose per step.
    pub num_speculative_tokens: usize,
    /// Maximum n-gram size for prompt lookup.
    pub ngram_prompt_lookup_max: usize,
    /// Minimum n-gram size for prompt lookup.
    pub ngram_prompt_lookup_min: usize,
    /// Optional dtype override for the draft model's weights ("auto",
    /// "float16", "bfloat16", etc). `None` inherits the target's dtype.
    /// Only meaningful when `speculative_model` points at a draft model.
    pub draft_model_dtype: Option<String>,
    /// LoRA adapter path (local directory or HF repo ID). None = disabled.
    pub lora_adapter: Option<String>,
    /// Pooling strategy for embeddings: "auto", "last", "cls", "mean".
    pub pooling_strategy: String,
    /// Number of GPUs for tensor parallelism (1 = single GPU, no TP).
    pub tensor_parallel_size: usize,
    /// Number of GPU stages for pipeline parallelism (1 = no PP).
    pub pipeline_parallel_size: usize,
    /// Number of nodes for multi-node TP (1 = single node).
    pub num_nodes: usize,
    /// This node's rank (0 = master).
    pub node_rank: usize,
    /// Master address for multi-node NCCL rendezvous.
    pub master_addr: String,
    /// Master port for multi-node NCCL rendezvous.
    pub master_port: u16,
    /// Whether to disable async scheduling (overlap GPU/CPU work).
    /// Default false — async scheduling is enabled by default.
    pub disable_async_scheduling: bool,
    /// Runner type: "generate" (default) or "pooling".
    pub runner: String,
    /// CUDA graph configuration. When `Some`, CUDA graphs may be captured
    /// for decode-step acceleration.
    pub cuda_graph_config: Option<CudaGraphConfig>,
    /// Whether prefix caching is enabled (KV cache reuse for shared prompts).
    /// Default: true.
    pub enable_prefix_caching: bool,
    /// Disable CUDA graph capture and run all steps eagerly.
    /// Default: false.
    pub enforce_eager: bool,
    /// CUDA graph mode: controls piecewise vs monolithic graph capture.
    /// Default: "full" (maintains current behavior).
    pub cuda_graph_mode: String,
    /// Maximum number of tokens processed in a single scheduler iteration.
    /// None = auto (min(max_model_len, 8192)).
    pub max_num_batched_tokens: Option<usize>,
    /// Benchmark cublasLt algorithms during warmup.
    pub cublas_autotune: bool,
    /// KV cache data type: "auto" (use model dtype) or "fp8_e4m3".
    pub kv_cache_dtype: String,
    /// Compute KV scales dynamically from the first forward pass.
    pub calculate_kv_scales: bool,
    /// Distributed executor backend: "auto" or "external_launcher".
    /// "external_launcher" reads RANK/LOCAL_RANK/WORLD_SIZE/MASTER_ADDR/MASTER_PORT
    /// from env and uses TCP-based NCCL init for inter-process TP.
    pub distributed_executor_backend: String,
    /// Optional chat template override.
    ///
    /// If `Some`, takes precedence over every auto-detected source
    /// (GGUF metadata `tokenizer.chat_template`, `tokenizer_config.json`,
    /// `chat_template.jinja`). Accepts either an inline Jinja template
    /// string or a filesystem path that gets read as the template body.
    /// Required for GGUFs (e.g. mmnga's Moonlight) whose converters
    /// dropped the template and whose source HF repo isn't co-located.
    pub chat_template: Option<String>,
    /// Optional sink for model-download progress. When `Some`, the HF shard
    /// download forwards byte-level progress here and suppresses the default
    /// `indicatif` stderr bars — used by the ratatui TUI so it can render
    /// progress itself instead of having bars scribble over its alt-screen.
    /// `None` (the default) keeps the stderr bars (CLI / server).
    #[serde(skip)]
    pub download_observer:
        Option<Arc<dyn scratchy_serving_engine::worker_factory::DownloadObserver>>,
}

impl Default for VllmConfig {
    fn default() -> Self {
        Self {
            model: String::new(),
            device: "auto".to_string(),
            dtype: "auto".to_string(),
            max_model_len: None,
            max_num_seqs: None,
            block_size: 16,
            gpu_memory_utilization: 0.9,
            hf_token: None,
            gguf_file: None,
            speculative_model: None,
            // K=2 is the empirical sweet spot for draft-model spec
            // decode on Apple Silicon when the target dominates
            // (e.g. 8B+1B). See `scr/src/args.rs` for the bench
            // distribution that led to this default.
            num_speculative_tokens: 2,
            ngram_prompt_lookup_max: 4,
            ngram_prompt_lookup_min: 1,
            draft_model_dtype: None,
            lora_adapter: None,
            pooling_strategy: "auto".to_string(),
            tensor_parallel_size: 1,
            pipeline_parallel_size: 1,
            num_nodes: 1,
            node_rank: 0,
            master_addr: "localhost".to_string(),
            master_port: 29500,
            cuda_graph_mode: "auto".to_string(),
            disable_async_scheduling: false,
            runner: "generate".to_string(),
            cuda_graph_config: None,
            enable_prefix_caching: true,
            enforce_eager: true, // TODO: debug multi-turn — disable CUDA graphs to isolate
            max_num_batched_tokens: None,
            cublas_autotune: false,
            kv_cache_dtype: "auto".to_string(),
            calculate_kv_scales: false,
            distributed_executor_backend: "auto".to_string(),
            chat_template: None,
            download_observer: None,
        }
    }
}

/// Fully initialized stack ready to serve requests.
#[allow(dead_code)]
pub struct InitializedStack {
    /// The async engine (wraps engine core + tokenizer).
    pub engine: Arc<AsyncEngine>,
    /// Model name for display.
    pub model_name: String,
    /// Maximum model length.
    pub max_model_len: usize,
}

/// Result of [`initialize_stack_sync`]: the raw engine components for
/// synchronous (offline) use — no async channels, no background step loop.
/// Used by [`LLM`](crate::llm::LLM) to match Python's direct engine path.
pub struct InitializedSyncStack {
    /// In-process engine core client (owns scheduler + executor).
    pub client: InprocClient,
    /// Optional tokenizer for encoding prompts / decoding outputs.
    pub tokenizer: Option<Arc<Tokenizer>>,
    /// Optional chat template for formatting chat messages.
    pub chat_template: Option<ChatTemplate>,
    /// Model name for display.
    pub model_name: String,
    /// Maximum model length.
    pub max_model_len: usize,
    /// KV cache block size in tokens.
    pub block_size: usize,
    /// Model-recommended sampling defaults (`generation_config.json`).
    pub generation_defaults: scratchy_core_common::sampling::GenerationDefaults,
}

/// `num_hidden_layers`, falling through to the nested `text_config`
/// (verbatim VL-wrapper configs — Qwen3.5 family) so the startup log
/// doesn't claim "num_layers=1".
fn resolve_num_layers(hf_config: &HfModelConfig) -> usize {
    hf_config
        .num_hidden_layers
        .or_else(|| {
            hf_config
                .extra
                .get("text_config")
                .and_then(|t| t.get("num_hidden_layers"))
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
        })
        .unwrap_or(1)
}

/// Effective KV-cache `block_size` for the active backend: the configured `--block-size` unless the
/// backend's own pool decides it ([`WorkerFactory::required_block_size`]).
///
/// The scheduler's block allocator computes `slot = block_id * block_size + offset` and the worker
/// resolves `block_id` to a pool page, so the two MUST agree or they name different cells for one
/// token — silent KV corruption, no fault raised.
///
/// 🛑 THIS FUNCTION USED TO BE THE DISAGREEMENT. It answered with a literal `64` under
/// `#[cfg(feature = "spyre")]` and a comment justifying it by the shape of a pool that had since been
/// rebuilt: "the PagedAttn ops HARD-CODE `block_size`=64 (the pool is `[nblk, 64, nkv, hd]`)". The
/// superdsc pool's page is `PagedKvPool::PAGE_SLOTS` = 256. The mismatch was inert only because the
/// worker IGNORED the scheduler's block ids and allocated pages from a free list of its own — which is
/// the same "two allocators for one resource" that made prefix caching impossible. Ask the backend.
fn effective_block_size(configured: usize, device: &str) -> usize {
    inventory::iter::<&dyn WorkerFactory>()
        .find(|f| f.matches(device))
        .and_then(|f| f.required_block_size())
        .map(|b| b.get())
        .unwrap_or(configured)
}

/// Prefill-chunk cap (tokens) for sliding-window models, enforced at the
/// scheduler via `max_num_batched_tokens.min(SWA_PREFILL_CHUNK_CAP)`. Bounds a
/// sliding group's in-flight KV per step, so its live block set stays ~window +
/// chunk (the hybrid pool is sized to `window + 2*cap` for eviction lag in
/// `compute_kv_blocks`). Keeps sliding KV a few hundred MiB at 32k vs ~1 GiB.
const SWA_PREFILL_CHUNK_CAP: usize = 2048;

/// Resolve the full set of stop-on-generate token IDs the engine
/// should honor, merging three HuggingFace conventions:
///
/// 1. `config.json::eos_token_id` — single int or array. Always read.
/// 2. `generation_config.json::eos_token_id` — same shape, but
///    authoritative when present (HF transformers reads this first
///    at `model.generate()` time; vLLM Python mirrors that).
/// 3. `tokenizer_config.json::eos_token` — a string (or
///    `{content: ...}` object) naming the actual end-of-turn token
///    for the chat template (e.g. Llama-3-Instruct stores
///    `"<|eot_id|>"` here, distinct from `config.json::eos_token_id`
///    = 128001 which is `<|end_of_text|>`). Resolved against the
///    loaded tokenizer's vocab; missing tokenizer → skipped.
///
/// Without #3 the chat loop never stops on `<|eot_id|>` (128009) for
/// Llama-3-Instruct: `config.json::eos_token_id` is 128001, the base
/// model's end-of-document token, which the Instruct fine-tune
/// emits only after a final `<|eot_id|>`. Replicates the merge
/// `transformers.generation.utils.GenerationMixin._prepare_generation_config`
/// performs on the Python side.
fn resolve_eos_token_ids(
    hf_config: &HfModelConfig,
    model_dir: Option<&std::path::Path>,
    preloaded_tokenizer: Option<&tokenizers::Tokenizer>,
) -> Vec<u32> {
    fn extract_ids(v: &serde_json::Value) -> Vec<u32> {
        if let Some(id) = v.as_u64() {
            return vec![id as u32];
        }
        if let Some(arr) = v.as_array() {
            return arr
                .iter()
                .filter_map(|e| e.as_u64().map(|id| id as u32))
                .collect();
        }
        Vec::new()
    }

    let mut out: Vec<u32> = Vec::new();
    let push = |id: u32, out: &mut Vec<u32>| {
        if !out.contains(&id) {
            out.push(id);
        }
    };

    // (1) config.json::eos_token_id
    if let Some(v) = hf_config.extra.get("eos_token_id") {
        for id in extract_ids(v) {
            push(id, &mut out);
        }
    }

    let Some(dir) = model_dir else {
        return out;
    };

    // (2) generation_config.json::eos_token_id
    let gen_cfg_path = dir.join("generation_config.json");
    if let Ok(bytes) = std::fs::read(&gen_cfg_path)
        && let Ok(json) = serde_json::from_slice::<serde_json::Value>(&bytes)
        && let Some(v) = json.get("eos_token_id")
    {
        for id in extract_ids(v) {
            push(id, &mut out);
        }
    }

    // (3) tokenizer_config.json::eos_token (string) → resolved via tokenizer.
    let tok_cfg_path = dir.join("tokenizer_config.json");
    if let (Ok(bytes), Some(tokenizer)) = (std::fs::read(&tok_cfg_path), preloaded_tokenizer)
        && let Ok(json) = serde_json::from_slice::<serde_json::Value>(&bytes)
        && let Some(eos) = json.get("eos_token")
    {
        let token_str: Option<&str> = if let Some(s) = eos.as_str() {
            Some(s)
        } else {
            eos.get("content").and_then(|v| v.as_str())
        };
        if let Some(s) = token_str
            && let Some(id) = tokenizer.token_to_id(s)
        {
            push(id, &mut out);
        }
    }

    // (4) The CHAT TEMPLATE's own turn terminator.
    //
    // A chat checkpoint ends each assistant turn with a token, and that
    // token is what the model actually emits when it is done — but it
    // is not always the declared `eos_token`. `gemma-3-1b-it` ships
    // `eos_token_id = 1` (`<eos>`) in config.json AND `<eos>` in
    // tokenizer_config, while its template closes turns with
    // `<end_of_turn>` (106). Sources (1)-(3) therefore all agree on the
    // WRONG id, generation never stops at the turn boundary, and the
    // model runs on into other-script garbage and then restarts its
    // answer. Its 4b sibling ships `eos_token_id = [1, 106]` and is
    // fine — the difference is checkpoint metadata, not the model.
    //
    // A stop STRING cannot cover this: 106 is a special token, so
    // detokenization strips it and the literal never appears in the
    // text to match against.
    //
    // Scanning the template for tokens the tokenizer knows keeps this
    // checkpoint-driven — no per-model table, and no effect on a
    // checkpoint whose template terminator is already the declared eos
    // (the `push` dedups).
    if let (Ok(bytes), Some(tokenizer)) = (std::fs::read(&tok_cfg_path), preloaded_tokenizer)
        && let Ok(json) = serde_json::from_slice::<serde_json::Value>(&bytes)
        && let Some(tpl) = json.get("chat_template").and_then(|v| v.as_str())
    {
        for cand in TEMPLATE_TURN_TERMINATORS {
            if tpl.contains(cand)
                && let Some(id) = tokenizer.token_to_id(cand)
            {
                push(id, &mut out);
            }
        }
    }

    out
}

/// Turn-terminating special tokens a chat template may close an
/// assistant turn with. Only added when the template ACTUALLY contains
/// the literal and the tokenizer knows it, so this is a recogniser for
/// what the checkpoint already declares — not a list of model
/// defaults.
const TEMPLATE_TURN_TERMINATORS: &[&str] = &[
    "<end_of_turn>",
    "<|im_end|>",
    "<|eot_id|>",
    "<|end|>",
    "<|endoftext|>",
];

// Worker creation result now lives in `scratchy-serving-worker` alongside
// the `WorkerFactory` trait and `Box<dyn Worker>`/`HfModelConfig` it names.
use scratchy_serving_worker::worker_factory::{
    ProgressCallback, WorkerCreationResult, WorkerFactory,
};

/// Create the appropriate worker via the `WorkerFactory` inventory registry.
///
/// Builds the shared [`WorkerCreateConfig`] once, then selects the first
/// registered backend factory whose `matches` accepts `config.device` and
/// calls its `create`. The per-backend construct→init→load→extract logic
/// lives in `scratchy_serving_worker::gpu_worker` (`CudaWorkerFactory` /
/// `MetalWorkerFactory`); there are no `#[cfg]` backend arms here.
///
/// Returns `(worker, hf_config, model_dir, dtype)`.
fn create_worker(
    config: &VllmConfig,
    model_path: String,
    progress: Option<&Arc<crate::progress::StartupProgress>>,
) -> Result<WorkerCreationResult> {
    use scratchy_serving_worker::worker_factory::WorkerCreateConfig;

    let is_pooling = config.runner == "pooling";

    // Parse device ID generically (e.g. "cuda:1" → 1, "cuda"/"auto"/"metal" → 0).
    // Harmless for metal, which is always device 0.
    let device_id = if config.device.starts_with("cuda:") {
        config.device[5..].parse::<i32>().unwrap_or(0)
    } else {
        0
    };

    let worker_config = WorkerCreateConfig {
        model_path: model_path.clone(),
        dtype: config.dtype.clone(),
        hf_token: config.hf_token.clone(),
        // The backend's own page size when it has one (see `effective_block_size`); a mismatch
        // between this and the scheduler's block size silently corrupts KV slots.
        block_size: effective_block_size(config.block_size, &config.device),
        device_id,
        max_num_seqs: resolve_max_num_seqs(config.max_num_seqs, None, false),
        enforce_eager: config.enforce_eager,
        cuda_graph_mode: config
            .cuda_graph_mode
            .parse()
            .unwrap_or(CudaGraphMode::Auto),
        // Default 1024 (not 8192 like Python). Our worker splits mixed
        // batches into a decode CUDA-graph pass + a prefill eager pass.
        // Smaller prefill chunks keep the eager pass fast (~25ms for 1024
        // tokens) while decode runs through the captured graph (~5ms).
        // Benchmarked: 1024 → 21.8 req/s vs 8192 → 12.1 req/s on Qwen2.5-3B.
        max_num_batched_tokens: config.max_num_batched_tokens.unwrap_or(1024),
        cuda_graph_sizes: config
            .cuda_graph_config
            .as_ref()
            .map(|c| c.capture_sizes.clone())
            .unwrap_or_default(),
        cublas_autotune: config.cublas_autotune,
        gpu_memory_utilization: config.gpu_memory_utilization,
        pooling_strategy: config.pooling_strategy.clone(),
        is_pooling,
        tp_rank: 0,
        tp_world_size: 1,
        pp_rank: 0,
        pp_size: 1,
        gguf_file: config.gguf_file.clone(),
        lora_adapter: config.lora_adapter.clone(),
        kv_cache_dtype: config.kv_cache_dtype.clone(),
        calculate_kv_scales: config.calculate_kv_scales,
        eos_token_ids: vec![],
        max_model_len: config.max_model_len,
        draft_model_path: spec_decode_draft_model_path(config),
        draft_model_dtype: config.draft_model_dtype.clone(),
    };

    // Progress closure (the former cuda-arm body). For now it just observes
    // the message; layer-by-layer progress would require deeper integration
    // into model loading. Backends that don't wire progress ignore it.
    let progress_cb: Option<ProgressCallback> = progress.map(|pb| {
        let _pb = pb.clone();
        std::sync::Arc::new(move |msg: &str| {
            // For now, just observe the message. Layer-by-layer progress
            // would require deeper integration into model loading.
            if msg.contains("layer") {
                // Could parse layer number and update sub-progress here
            }
        }) as ProgressCallback
    });

    // Empty registry — distinct from the per-device "no backend" error below,
    // which means a backend IS registered but doesn't claim this device string.
    // The common cause is simply a build with no GPU backend feature; the rarer
    // cause is the linker GC'ing the registration.
    if inventory::iter::<&dyn WorkerFactory>().next().is_none() {
        anyhow::bail!(
            "No inference backend is compiled into this binary. Rebuild with a GPU \
             backend feature:\n    --features metal   (Apple Silicon)\n    --features \
             cuda    (NVIDIA GPU)\ne.g. `cargo build --release \
             -p scratchy-cli --features metal`.\n(If you DID build with a \
             backend, an empty registry instead means the linker GC'd its \
             `inventory::submit!` — ensure scratchy-serving-worker stays linked via \
             an `extern crate ... as _` anchor.)"
        );
    }

    let factory = inventory::iter::<&dyn WorkerFactory>()
        .find(|f| f.matches(&config.device))
        .ok_or_else(|| {
            anyhow::anyhow!(
                "No backend available for device '{}'. \
                 Build with --features metal (Apple Silicon) or --features cuda (NVIDIA GPU).",
                config.device
            )
        })?;

    factory.create(worker_config, progress_cb)
}

/// Initialize cache on the worker and compute block counts.
///
/// On CPU, `gpu_memory_utilization` is ignored and a default of 50% is used
/// instead (matching Python vLLM's `DEFAULT_CPU_MEM_UTILIZATION`). This
/// prevents allocating most of system RAM for KV cache on CPU-only hosts.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn init_cache(
    mut worker: Box<dyn Worker>,
    block_size: usize,
    hf_config: &HfModelConfig,
    dtype_elem_bytes: usize,
    gpu_memory_utilization: f64,
    device: &str,
    kv_cache_dtype: &str,
    max_model_len: usize,
) -> Result<(
    Box<dyn Worker>,
    usize,
    usize,
    Option<HybridKvConfig>,
    f64,
    usize,
)> {
    let available_memory = worker
        .determine_available_memory()
        .context("failed to determine available memory")?;

    // For CUDA AND Metal workers, determine_available_memory already returns
    // the KV cache budget with gpu_memory_utilization baked in (matching
    // Python vLLM: total * util - non_kv_cache). On Metal `total` is further
    // capped at `maxBufferLength` (the real wireable ceiling) and util is
    // applied inside the worker, so the returned value is the FINAL KV budget
    // — do NOT re-apply utilization here (that double-shrinks KV). Only CPU
    // gets the returned value as raw free memory and applies util locally.
    let is_gpu_prebaked = (cfg!(feature = "cuda")
        && device != "cpu"
        && (device == "auto" || device.starts_with("cuda")))
        || (cfg!(feature = "metal") && device != "cpu");
    let is_cpu = device == "cpu"
        || (device == "auto" && !cfg!(feature = "cuda") && !cfg!(feature = "metal"));

    let (kv_cache_bytes, utilization) = if is_gpu_prebaked {
        // Worker already computed: total * util - non_kv_cache (capped at
        // maxBufferLength on Metal).
        (available_memory, gpu_memory_utilization)
    } else if is_cpu {
        const DEFAULT_CPU_MEM_UTILIZATION: f64 = 0.5;
        info!(
            "CPU device: using {:.0}% of system memory for KV cache (override with --gpu-memory-utilization)",
            DEFAULT_CPU_MEM_UTILIZATION * 100.0
        );
        let bytes = (available_memory as f64 * DEFAULT_CPU_MEM_UTILIZATION) as usize;
        (bytes, DEFAULT_CPU_MEM_UTILIZATION)
    } else {
        // Metal or other GPU — apply utilization to raw free memory.
        let bytes = (available_memory as f64 * gpu_memory_utilization) as usize;
        (bytes, gpu_memory_utilization)
    };

    // Cap `max_model_len` to the longest context the backend's paged-attention
    // kernels can address (hybrid SWA arches: the sliding class's block table
    // is baked at a fixed row stride). Beyond it the kernel would index past
    // the block table; capping here means an over-long prompt is rejected by
    // the scheduler instead. Also sizes the KV pool to this context, not the
    // whole memory budget.
    let effective_max_model_len = match worker.kv_max_addressable_tokens() {
        Some(cap) if cap < max_model_len => {
            info!(
                "Capping max_model_len {} → {} (paged-attention addressable limit)",
                max_model_len, cap
            );
            cap
        }
        _ => max_model_len,
    };
    let (mut num_gpu_blocks, swa_hybrid_kv) = compute_kv_blocks(
        kv_cache_bytes,
        block_size,
        hf_config,
        dtype_elem_bytes,
        utilization,
        kv_cache_dtype,
        effective_max_model_len,
        worker.supports_hybrid_swa_kv(),
    );
    // Fixed-size resident KV pool cap (spyre/sendnn PAGED CB): the on-card pool is
    // exactly `nblk` blocks, so the scheduler's block allocator must not exceed it
    // (else it hands out a block id past the pool → silent KV corruption). The
    // uniform `compute_num_blocks` sizes from the memory budget and can be larger;
    // clamp it. `None` for cuda/metal (budget-sized pools).
    if let Some(cap) = worker.kv_cache_num_blocks_override()
        && cap < num_gpu_blocks
    {
        info!(
            "Capping num_gpu_blocks {} → {} (fixed resident KV pool: {} blocks)",
            num_gpu_blocks, cap, cap
        );
        num_gpu_blocks = cap;
    }
    worker
        .initialize_cache(num_gpu_blocks, 0)
        .context("failed to initialize cache")?;
    Ok((
        worker,
        available_memory,
        num_gpu_blocks,
        swa_hybrid_kv,
        utilization,
        effective_max_model_len,
    ))
}

#[allow(dead_code)]
struct InitializedCore {
    client: InprocClient,
    tokenizer: Option<Arc<Tokenizer>>,
    model_name: String,
    max_model_len: usize,
    model_dir: Option<std::path::PathBuf>,
    hf_config: HfModelConfig,
}

/// `--speculative-model <repo>` as a draft-model path, or `None` for
/// the n-gram proposer / no spec-decode case. Threaded into each
/// `WorkerCreateConfig` construction so the worker can load the draft
/// alongside the target.
fn spec_decode_draft_model_path(config: &VllmConfig) -> Option<String> {
    match config.speculative_model.as_deref() {
        Some("ngram") | None => None,
        Some(path) => Some(path.to_string()),
    }
}

/// Build the speculative-decoding proposer config from a `VllmConfig`.
///
/// Three outcomes:
///   * `speculative_model == None` → `Ok(None)` (spec decode disabled).
///   * `speculative_model == Some("ngram")` → `Ok(Some(Ngram(...)))`.
///   * `speculative_model == Some(other)` → `Ok(Some(DraftModel(...)))` —
///     but [`validate_speculative_decoding`] (called earlier from each
///     `initialize_stack*` entry point) refuses to start an engine with
///     this variant until the draft-model proposer is wired.
///
/// Today this fn returns `Err` for the draft-model branch as a defense in
/// depth in case the entry-point validator is bypassed. Phase 4 replaces
/// that `Err` with the real draft-model wiring.
fn build_proposer_config(
    config: &VllmConfig,
    max_model_len: usize,
) -> Result<Option<ProposerConfig>> {
    let Some(spec) = config.speculative_model.as_deref() else {
        return Ok(None);
    };
    if spec == "ngram" {
        return Ok(Some(ProposerConfig::Ngram(NgramProposerConfig {
            num_speculative_tokens: config.num_speculative_tokens,
            max_ngram_size: config.ngram_prompt_lookup_max,
            min_ngram_size: config.ngram_prompt_lookup_min,
            max_model_len,
        })));
    }
    // Non-"ngram" value: treat as a draft-model path / HF repo. The
    // engine logs the registration and proceeds as baseline until phase 4
    // wires the real proposer (`EngineCore::new` handles this variant by
    // leaving the proposer as `None`).
    Ok(Some(ProposerConfig::DraftModel(DraftModelProposerConfig {
        model: spec.to_string(),
        num_speculative_tokens: config.num_speculative_tokens,
        dtype: config.draft_model_dtype.clone(),
        max_model_len,
    })))
}

/// Validate `--speculative-model` against the phase guards before any
/// model load. Called at the top of each `initialize_stack*` entry.
///
/// For draft-model speculative decoding (`--speculative-model <path>`):
///   1. Vocab/tokenizer alignment (`validate_target_draft_pair`).
///   2. Metal memory budget rough-check (metal feature only).
///
/// Once wired, the worker runs a K-step draft
/// chain after each target step and stashes the drafts on
/// `ModelRunnerOutput.draft_token_ids`; the engine forwards them to the
/// scheduler via `set_spec_token_ids`. The metal-side M>1 verify path
/// is a follow-up — until it lands, drafts are proposed but discarded
/// by the worker's next prepare_inputs, and the engine reports 0%
/// acceptance.
fn validate_speculative_decoding(config: &VllmConfig) -> Result<()> {
    let Some(spec) = config.speculative_model.as_deref() else {
        return Ok(());
    };
    if spec == "ngram" {
        return Ok(());
    }
    validate_target_draft_pair(config, spec)?;
    info!(
        "spec-decode: draft model {:?} accepted; worker-side proposer runs a K-step \
         decode chain per step.",
        spec
    );
    Ok(())
}

/// Mirror Python vLLM (vllm/config/vllm.py:709-721): force-disable async
/// scheduling for non-EAGLE spec-decode methods. Both ngram and draft-model
/// paths seed K-step drafts from step N's output for step N+1's verify;
/// async's 1-step lookahead skews that to N+2 and acceptance collapses to
/// ~0. EAGLE/MTP methods (not yet ported) re-use hidden states directly
/// and are compatible with async.
fn spec_decode_requires_sync(config: &VllmConfig) -> bool {
    config.speculative_model.is_some()
}

/// Phase-2 guards: vocab/tokenizer match + memory budget for the
/// target + draft pair.
///
/// Resolves both models to their HF-cache snapshot dirs and asserts:
///   * `target.vocab_size == draft.vocab_size` (mirrors Python vLLM's
///     SpeculativeConfig vocab-size check).
///   * `sha256(target/tokenizer.json) == sha256(draft/tokenizer.json)`
///     (the cheapest "same tokenizer" assertion that survives metadata
///     differences like padding-token whitespace).
///   * On metal builds: weight bytes + a coarse KV estimate fit under
///     `recommendedMaxWorkingSetSize`.
///
/// Files that don't exist locally trigger `resolve_model_path`'s normal
/// HF download path — so a misconfigured draft URL surfaces here, not
/// halfway through model load.
fn validate_target_draft_pair(config: &VllmConfig, draft_spec: &str) -> Result<()> {
    let target_dir = scratchy_serving_worker::worker_factory::resolve_model_path(
        &config.model,
        config.hf_token.as_deref(),
        config.gguf_file.as_deref(),
        None,
    )
    .with_context(|| format!("resolving target model path {:?}", config.model))?;
    let draft_dir = scratchy_serving_worker::worker_factory::resolve_model_path(
        draft_spec,
        config.hf_token.as_deref(),
        None, // draft never uses target's --gguf-file
        None,
    )
    .with_context(|| format!("resolving draft model path {draft_spec:?}"))?;

    let target_cfg = HfModelConfig::from_dir(&target_dir)
        .with_context(|| format!("reading target config.json at {:?}", target_dir))?;
    let draft_cfg = HfModelConfig::from_dir(&draft_dir)
        .with_context(|| format!("reading draft config.json at {:?}", draft_dir))?;

    if let (Some(tv), Some(dv)) = (target_cfg.vocab_size, draft_cfg.vocab_size)
        && tv != dv
    {
        anyhow::bail!(
            "target/draft vocab_size mismatch: target={} ({}), draft={} ({}). \
                 Spec-decode requires matching vocabularies — the draft model must \
                 emit token IDs the target can interpret. Pick a draft from the same \
                 model family (e.g. Llama-3.2-1B paired with Llama-3.2-3B).",
            tv,
            config.model,
            dv,
            draft_spec
        );
    }

    let target_tok = target_dir.join("tokenizer.json");
    let draft_tok = draft_dir.join("tokenizer.json");
    if target_tok.is_file() && draft_tok.is_file() {
        let tb = std::fs::read(&target_tok).with_context(|| format!("reading {:?}", target_tok))?;
        let db = std::fs::read(&draft_tok).with_context(|| format!("reading {:?}", draft_tok))?;
        if tb != db {
            // Byte-equal failed. The two HF families that pair well
            // for spec-decode (Llama-3.1 target + Llama-3.2 draft,
            // Qwen-2.5 target + Qwen-2.5-0.5B draft, etc.) ship
            // tokenizer.json files that are SEMANTICALLY identical
            // but serialize merges differently (older `"a b"` strings
            // vs newer `["a", "b"]` lists) or carry different
            // whitespace/version metadata. Reject only if the
            // SEMANTIC token table actually differs.
            tokenizer_semantic_mismatch_check(&tb, &db, &target_tok, &draft_tok)?;
        }
    }

    spec_decode_memory_budget_check(config, &target_cfg, &target_dir, &draft_cfg, &draft_dir)?;
    Ok(())
}

/// Compare two `tokenizer.json` files semantically (not byte-equal).
///
/// Pairs from the same family (Llama-3.1-8B target + Llama-3.2-1B
/// draft, Qwen-2.5-7B target + Qwen-2.5-0.5B draft, etc.) ship
/// tokenizer.json files that produce identical token IDs but
/// serialize the merges differently:
///   * Older HF tokenizers write `model.merges` as
///     `["a b", "c d", ...]` (space-separated strings).
///   * Newer tokenizers write `[["a","b"], ["c","d"], ...]` (lists).
///
/// Normalize before comparing so cosmetic format drift doesn't block
/// spec-decode. The actual token IDs are identical iff
/// (`model.vocab`, `added_tokens`, `normalized(model.merges)`) match.
fn tokenizer_semantic_mismatch_check(
    target_bytes: &[u8],
    draft_bytes: &[u8],
    target_path: &Path,
    draft_path: &Path,
) -> Result<()> {
    let parse = |bytes: &[u8], path: &Path| -> Result<serde_json::Value> {
        serde_json::from_slice::<serde_json::Value>(bytes)
            .with_context(|| format!("parsing {path:?} as JSON for spec-decode tokenizer check"))
    };
    let t = parse(target_bytes, target_path)?;
    let d = parse(draft_bytes, draft_path)?;

    let normalize_merges = |v: &serde_json::Value| -> Vec<(String, String)> {
        let arr = match v.as_array() {
            Some(a) => a,
            None => return Vec::new(),
        };
        let mut out = Vec::with_capacity(arr.len());
        for m in arr {
            if let Some(s) = m.as_str() {
                if let Some((a, b)) = s.split_once(' ') {
                    out.push((a.to_string(), b.to_string()));
                }
            } else if let Some(pair) = m.as_array()
                && pair.len() == 2
                && let (Some(a), Some(b)) = (pair[0].as_str(), pair[1].as_str())
            {
                out.push((a.to_string(), b.to_string()));
            }
        }
        out
    };

    let tv = t.pointer("/model/vocab");
    let dv = d.pointer("/model/vocab");
    if tv != dv {
        anyhow::bail!(
            "target/draft tokenizer.json mismatch: model.vocab differs ({} vs {} bytes on disk). \
             Spec-decode requires identical token IDs in both models. \
             Pick a draft from the same model family as the target.",
            target_bytes.len(),
            draft_bytes.len(),
        );
    }
    let ta = t.pointer("/added_tokens");
    let da = d.pointer("/added_tokens");
    if ta != da {
        anyhow::bail!(
            "target/draft tokenizer.json mismatch: added_tokens differs. \
             Spec-decode requires identical token IDs in both models. \
             Pick a draft from the same model family as the target.",
        );
    }
    let tm = normalize_merges(
        t.pointer("/model/merges")
            .unwrap_or(&serde_json::Value::Null),
    );
    let dm = normalize_merges(
        d.pointer("/model/merges")
            .unwrap_or(&serde_json::Value::Null),
    );
    if tm != dm {
        anyhow::bail!(
            "target/draft tokenizer.json mismatch: model.merges differs (normalized {} vs {} entries). \
             Spec-decode requires identical token IDs in both models. \
             Pick a draft from the same model family as the target.",
            tm.len(),
            dm.len(),
        );
    }
    // Vocab + added_tokens + merges all match semantically — the two
    // tokenizers produce identical token IDs even though the file
    // bytes differ. Safe for spec decode.
    info!(
        "spec-decode tokenizer.json byte-differs but is semantically identical \
         (vocab + added_tokens + normalized merges match) — accepted."
    );
    Ok(())
}

/// Coarse memory-budget check for the target+draft pair.
///
/// Checks **resident weight bytes only** — KV cache is paged and sized
/// dynamically at runtime against whatever the worker reports as
/// available, so it's the wrong shape to compare against a static
/// budget here. The goal is to catch obvious misconfigurations (e.g.
/// 7B + 3B target+draft on a 24 GiB machine) before model load.
///
/// Asserts that `target_weights + draft_weights` stays under
/// `working_set * gpu_memory_utilization * 0.6`, reserving ~40% for KV
/// cache, activations, and other working-set overhead. Backend-agnostic:
/// the working-set figure comes from whichever backend factory is
/// registered (it reports one) — a backend that doesn't report one
/// skips the check.
fn spec_decode_memory_budget_check(
    config: &VllmConfig,
    target_cfg: &HfModelConfig,
    target_dir: &std::path::Path,
    draft_cfg: &HfModelConfig,
    draft_dir: &std::path::Path,
) -> Result<()> {
    let _ = (target_cfg, draft_cfg);
    let Some(total) = backend_recommended_working_set_size() else {
        return Ok(()); // Backend doesn't report a working-set budget — skip.
    };
    // Reserve at least 40% of the gpu-memory-utilization budget for KV +
    // activations. Empirically the runtime sizes KV against
    // `(working_set - currentAllocatedSize) * gpu_memory_utilization`,
    // so weights consuming >60% leaves KV starved.
    let weight_budget = (total as f64 * config.gpu_memory_utilization * 0.6).round() as u64;

    let target_weights = weight_bytes_on_disk(target_dir);
    let draft_weights = weight_bytes_on_disk(draft_dir);
    let total_w = target_weights + draft_weights;

    if total_w > weight_budget {
        anyhow::bail!(
            "estimated combined weight footprint {} > {} (60% of \
             recommendedMaxWorkingSetSize × gpu_memory_utilization={:.2}). \
             Breakdown: target_weights={}, draft_weights={}. \
             Either pick a smaller draft, reduce --gpu-memory-utilization, \
             or run on a machine with more unified memory.",
            human_bytes(total_w),
            human_bytes(weight_budget),
            config.gpu_memory_utilization,
            human_bytes(target_weights),
            human_bytes(draft_weights),
        );
    }
    info!(
        "spec-decode weight budget OK: target_w={} + draft_w={} = {} / {} weight budget \
         (working set = {})",
        human_bytes(target_weights),
        human_bytes(draft_weights),
        human_bytes(total_w),
        human_bytes(weight_budget),
        human_bytes(total),
    );
    Ok(())
}

/// Recommended working-set size (bytes) from whichever backend factory is
/// registered in this build, without naming a specific backend crate.
fn backend_recommended_working_set_size() -> Option<u64> {
    inventory::iter::<&dyn WorkerFactory>().find_map(|f| f.recommended_working_set_size())
}

fn weight_bytes_on_disk(dir: &std::path::Path) -> u64 {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return 0;
    };
    entries
        .flatten()
        .filter(|e| {
            let n = e.file_name();
            let s = n.to_string_lossy();
            s.ends_with(".safetensors")
                || s.ends_with(".gguf")
                || (s.starts_with("model") && s.ends_with(".bin"))
        })
        // `entry.metadata()` reports the *symlink's* own size; HF cache
        // entries are symlinks into the blob store, so we have to call
        // `fs::metadata(path)` to follow the link to the actual file.
        .filter_map(|e| std::fs::metadata(e.path()).ok())
        .map(|m| m.len())
        .sum()
}

fn human_bytes(b: u64) -> String {
    const GIB: f64 = 1_073_741_824.0;
    const MIB: f64 = 1_048_576.0;
    let f = b as f64;
    if f >= GIB {
        format!("{:.2} GiB", f / GIB)
    } else {
        format!("{:.0} MiB", f / MIB)
    }
}

/// Whether an architecture is recurrent / hybrid — it keeps a sequential
/// conv/ssm (Mamba-style) or Gated-DeltaNet state that cannot be reconstructed
/// from a cached KV prefix. Prefix caching must be disabled for these models
/// (a prefix-cache hit would feed only the new tokens, leaving the recurrent
/// state computed over the wrong span). Mirrors vLLM's hybrid/SSM handling.
/// Detected by HF `architectures` string since the marker is per-layer.
fn is_recurrent_hybrid_arch(architectures: &[String]) -> bool {
    architectures.iter().any(|a| {
        let a = a.as_str();
        // Gated-DeltaNet hybrids (Qwen3.5 / Qwen3.6 / Qwen3-Next).
        a.contains("Qwen3_5")
            || a.contains("Qwen3_6")
            || a.contains("Qwen3Next")
            // Mamba / SSM families.
            || a.contains("Mamba")
            || a.contains("Jamba")
            || a.contains("FalconH")
            || a.contains("RecurrentGemma")
            || a.contains("Zamba")
            || a.contains("Lfm2")
            || a.contains("GraniteMoeHybrid")
    })
}

/// Common initialization: worker → cache → executor → InprocClient → tokenizer.
///
/// Handles both single-GPU (TP=1) and multi-GPU (TP>1) transparently.
/// Offline (LLM/throughput) vs online-server default for `max_num_batched_tokens`,
/// resolved per device (mirrors Python vLLM `EngineArgs.get_batch_defaults`):
/// an H100-class GPU on the offline path gets 16384 rather than the base 2048.
/// Falls back to the base `SchedulerConfig` constant when the device cannot be
/// queried or on non-CUDA builds. Only the batched-tokens value is consumed
/// here; `max_num_seqs` keeps its existing default for now.
fn resolve_default_max_num_batched_tokens(is_offline: bool) -> usize {
    if let Some((total_bytes, name)) = backend_device_total_bytes_and_name() {
        return SchedulerConfig::batch_defaults(total_bytes, &name, is_offline).0;
    }
    SchedulerConfig::DEFAULT_MAX_NUM_BATCHED_TOKENS
}

/// THE CONCURRENCY DEFAULT WHEN NOBODY ASKED — the `.1` of the same device-aware pair
/// [`resolve_default_max_num_batched_tokens`] takes its `.0` from. `.1` is equal for offline and
/// server, so a caller with no usage context in scope may pass either.
fn resolve_default_max_num_seqs(is_offline: bool) -> usize {
    if let Some((total_bytes, name)) = backend_device_total_bytes_and_name() {
        return SchedulerConfig::batch_defaults(total_bytes, &name, is_offline).1;
    }
    SchedulerConfig::DEFAULT_MAX_NUM_SEQS
}

/// ⭐⭐⭐ THE ONE PLACE `--max-num-seqs` IS DECIDED, and the order is WORKER, then DEVICE, then base.
///
/// Called TWICE with different knowledge, which is the whole reason it takes `worker_cap`:
///
/// * BEFORE the worker exists (`worker_cap = None`) to fill [`WorkerCreateConfig::max_num_seqs`] —
///   cuda sizes its Gated-DeltaNet state pool and its graph-capture ladder from it, and spyre's
///   `declare_workload` sizes the KV pool from it, all at construction time;
/// * AFTER `load_model` (`worker_cap = Some(..)`) for the SCHEDULER, where a backend whose decode
///   width is fixed at BAKE time reports the widest batch it can express
///   ([`Worker::max_num_seqs_override`]).
///
/// ⛔ AN EXPLICIT REQUEST IS CAPPED, NEVER REPLACED, and an unset one is ANSWERED, never guessed:
/// `None` + a worker cap is the backend's own default, so `scr serve` with no flags runs at the
/// width the bake can batch rather than at a generic 256 that merely happens to clamp to it.
fn resolve_max_num_seqs(
    requested: Option<usize>,
    worker_cap: Option<usize>,
    is_offline: bool,
) -> usize {
    match (requested, worker_cap) {
        // The caller named a width and the backend cannot express it: honour the smaller, and say so
        // — a silently narrower run is `capping-is-not-validating`.
        (Some(want), Some(cap)) if cap < want => {
            info!(
                "Capping max_num_seqs {} → {} (widest batched decode width this bake can express; \
                 beyond it decode falls back to one launch per request)",
                want, cap
            );
            cap
        }
        (Some(want), _) => want,
        // Nobody asked. The backend's own width IS the default when it has one.
        (None, Some(cap)) => cap,
        // No backend answer either (cuda/metal, whose ladders track `max_num_seqs`): the device's.
        (None, None) => resolve_default_max_num_seqs(is_offline),
    }
}

/// Total device memory + name from whichever backend factory is registered in
/// this build (cuda XOR metal XOR …), without naming a specific backend crate.
fn backend_device_total_bytes_and_name() -> Option<(u64, String)> {
    inventory::iter::<&dyn WorkerFactory>().find_map(|f| f.device_total_bytes_and_name())
}

/// Shared by [`initialize_stack`] (async server path) and
/// [`initialize_stack_sync`] (sync LLM path). `is_offline` selects the
/// device-aware batched-token default (offline LLM vs online server context).
fn initialize_core(
    config: &VllmConfig,
    progress: Option<&Arc<crate::progress::StartupProgress>>,
    is_offline: bool,
) -> Result<InitializedCore> {
    let tp_size = config.tensor_parallel_size;

    // TP > 1: multi-GPU path with NCCL.
    if tp_size > 1 {
        return initialize_core_tp(config);
    }

    let model_path = config.model.clone();
    let model_name = extract_model_name(&model_path).into_owned();

    if let Some(pb) = progress {
        pb.set_stage("Loading model");
    }

    // Resolve the model dir up-front so we can load the tokenizer in
    // parallel with the worker's heavy weight upload. `try_load_tokenizer`
    // is the canonical loader (init.rs::try_load_tokenizer) and lives
    // here in scratchy-serving-api — workers shouldn't be re-implementing it.
    // For cached models `resolve_model_path` is sub-ms; for HF download
    // it does the download once, then `worker.load_model` re-resolves
    // the same cached path (cheap). Returns `None` for `.gguf` sources
    // (no sibling tokenizer.json) — that path falls through to the
    // worker's `take_preloaded_tokenizer` (cuda/mlx GGUF builds the
    // tokenizer from file metadata via `scratchy_quantizations::gguf::gguf_tokenizer`,
    // the only reason any worker still touches tokenizers).
    let prefetched_model_dir = scratchy_serving_worker::worker_factory::resolve_model_path(
        &model_path,
        config.hf_token.as_deref(),
        config.gguf_file.as_deref(),
        config.download_observer.clone(),
    )
    .ok();
    let parallel_tokenizer_handle = prefetched_model_dir.as_ref().and_then(|dir| {
        let tok_path = dir.join("tokenizer.json");
        if !tok_path.exists() {
            return None;
        }
        Some(std::thread::spawn(move || {
            tokenizers::Tokenizer::from_file(&tok_path).ok()
        }))
    });

    let (mut worker, hf_config, model_dir, model_dtype) =
        create_worker(config, model_path, progress)?;

    if let Some(arch) = worker.architecture() {
        info!("Resolved model architecture: {}", arch);
    }

    let preloaded_tokenizer = worker
        .take_preloaded_tokenizer()
        .or_else(|| parallel_tokenizer_handle.and_then(|h| h.join().ok().flatten()));

    let max_model_len = config
        .max_model_len
        .or(hf_config.max_position_embeddings())
        .unwrap_or(4096);
    let num_layers = resolve_num_layers(&hf_config);

    info!(
        "Model: {}, max_model_len={}, num_layers={}",
        model_name, max_model_len, num_layers
    );

    if let Some(pb) = progress {
        pb.set_stage("Initializing KV cache");
    }
    // The backend's page size when it declares one, so the scheduler's block allocator
    // (slot = block_id * block_size + offset) addresses the same slots the pool uses. Must match
    // `create_worker`'s WorkerCreateConfig block_size — both go through `effective_block_size`.
    let block_size = effective_block_size(config.block_size, &config.device);

    let (
        mut worker,
        available_memory,
        num_gpu_blocks,
        swa_hybrid_kv,
        effective_utilization,
        max_model_len,
    ) = init_cache(
        worker,
        block_size,
        &hf_config,
        model_dtype,
        config.gpu_memory_utilization,
        &config.device,
        &config.kv_cache_dtype,
        max_model_len,
    )?;
    // Capture the target-reactive prefill-bucket cap before `worker` is moved
    // into the engine below — used to clamp the scheduler's batched-token bound.
    let worker_prefill_bucket_max_m = worker.prefill_bucket_max_m();
    // Same reason, same moment: the widest concurrency this backend can DECODE in one batched step, on a
    // backend whose decode-width ladder is fixed at bake time and so cannot grow to meet `max_num_seqs`.
    let worker_max_num_seqs = worker.max_num_seqs_override();

    info!(
        "Available memory: {:.1} GB, memory_utilization={}, num_gpu_blocks={}",
        available_memory as f64 / (1024.0 * 1024.0 * 1024.0),
        effective_utilization,
        num_gpu_blocks
    );

    let kv_cache_tokens = num_gpu_blocks * block_size;
    info!("KV cache size: {} tokens", kv_cache_tokens);
    info!(
        "Maximum concurrency for {} tokens per request: {:.2}x",
        max_model_len,
        kv_cache_tokens as f64 / max_model_len as f64
    );

    #[cfg(feature = "metrics")]
    {
        let m = crate::metrics::VllmMetrics::global();
        m.gpu_cache_blocks_total.set(num_gpu_blocks as i64);
    }

    if let Some(pb) = progress {
        pb.set_stage("Warming up model");
    }
    worker
        .compile_or_warm_up_model()
        .context("failed to compile or warm up model")?;

    let executor = UniProcExecutor::new_pre_initialized(worker);

    let eos_token_ids: Vec<u32> = resolve_eos_token_ids(
        &hf_config,
        model_dir.as_deref(),
        preloaded_tokenizer.as_ref(),
    );
    if !eos_token_ids.is_empty() {
        info!("EOS token IDs: {:?}", eos_token_ids);
    }

    let use_async_scheduling =
        !config.disable_async_scheduling && !spec_decode_requires_sync(config);
    // Recurrent / hybrid arches (Mamba, Gated-DeltaNet — Qwen3.5/3.6/Next, …)
    // carry a sequential conv/ssm state that CANNOT be reconstructed from a
    // cached KV prefix: a prefix-cache hit would feed the sequence only its
    // *new* tokens, leaving the recurrent state computed over the wrong span
    // (garbage output). vLLM disables prefix caching for these models; do the
    // same. (Per-layer linear-attention is the marker; detect by arch string.)
    let enable_prefix_caching = {
        let hybrid = is_recurrent_hybrid_arch(&hf_config.architectures);
        if hybrid && config.enable_prefix_caching {
            info!(
                "Prefix caching disabled: {:?} is a recurrent/hybrid (Mamba/Gated-DeltaNet) \
                 architecture — its conv/ssm state cannot be reconstructed from a cached KV prefix",
                hf_config.architectures
            );
        }
        // ⛔ AND THE BACKEND MUST BE ABLE TO HONOUR IT — asked HERE, where every entry point passes.
        //
        // A worker that cannot serve a cached prefix (`WorkerFactory::supports_prefix_caching`) gets a
        // scheduler that SKIPS the prefill for a prefix it believes is resident, while the worker holds
        // no KV for those slots. The rows then attend UNWRITTEN pool pages.
        //
        // ⭐ THIS CHECK ALREADY EXISTED — in `LlmEngine::from_config` ONLY. `scr batch` builds its stack
        // through `initialize_stack` -> `initialize_core` and never reached it, so batch runs had prefix
        // caching ON against a backend that declares it unsupported. Measured cost: rows whose prefill was
        // skipped read unwritten pages, and the resulting garbage was indistinguishable from a batched-decode
        // bug — it cost most of a session's debugging before `Prefix caching enabled` in the log gave it away.
        // ⛔⛔⛔ AND WHEN A BACKEND REFUSES, THE REFUSAL MUST NAME A PROPERTY OF THE BACKEND — NOT A
        // FEATURE SWITCHED OFF SO A TEST WOULD PASS.
        //
        // The comment that used to sit here said the spyre pool "cannot support it BY CONSTRUCTION"
        // because `PageRun::physical(lp) = row * pages_per_row + lp` gave every row an exclusive page
        // stripe. The striping was real; "by construction" was not — it described OUR construction, a
        // layout picked so one launch could stride the whole batch. The device's actual limit is that it
        // has no indirect addressing, which forbids nothing here. That `row *` term was also exactly the
        // "request concept below the host" the KV design forbids, so prefix caching was never a separate
        // feature to decline: it was blocked by a defect in the pool's addressing.
        //
        // ⭐ IT IS SUPPORTED NOW (`SpyreWorkerFactory::supports_prefix_caching`), by three changes that
        // between them make a shared prefix real rather than promised:
        //   1. a request's pages ARE the scheduler's `block_ids` (`install_host_blocks` + `BlockTable`,
        //      whose only constructor takes host ids), so the worker's own free-list allocator — a second
        //      allocator over the scheduler's pool, and the actual blocker — no longer exists;
        //   2. the block size and the block COUNT come from the pool that addresses them
        //      (`WorkerFactory::required_block_size`, `Worker::kv_cache_num_blocks_override`), so a block
        //      id is a page and never one the pool lacks;
        //   3. the worker reports, per request and per step, the SLOT SPAN its blocks must cover and the
        //      leading tokens a token-indexed cache may hash (`ModelRunnerOutput::kv_extent`) — because a
        //      batched step appends every row at one slot, so a short request's keys spread past its token
        //      count and only its first contiguous run is token-addressed.
        //
        // Asked at the ONE point that computes the flag, so no future entry point can skip it.
        let backend_refuses =
            inventory::iter::<&dyn scratchy_serving_engine::worker_factory::WorkerFactory>()
                .find(|f| f.matches(&config.device))
                .is_some_and(|f| !f.supports_prefix_caching());
        if backend_refuses && config.enable_prefix_caching && !hybrid {
            info!(
                "Prefix caching disabled: the {} backend's worker does not support it — the scheduler \
                 would skip the prefill for a prefix the worker holds no KV for",
                config.device
            );
        }
        config.enable_prefix_caching && !hybrid && !backend_refuses
    };

    // Device-specific `max_num_batched_tokens` default — no flat constant on any
    // backend. On backends that prune a compiled prefill-bucket ladder
    // target-reactively (Metal today), the largest resident bucket the device
    // just selected IS the right per-device default, and a single forward can
    // never exceed it (no NoBucketFits). On backends without ladder pruning
    // (CUDA), fall back to the existing device-aware default (queries GPU
    // mem/name; e.g. 16384 on an H100-class offline part). An explicit
    // `--max-num-batched-tokens` always wins, clamped to the resident bucket
    // where one exists so the user can't overflow it.
    let max_num_batched_tokens = match (config.max_num_batched_tokens, worker_prefill_bucket_max_m)
    {
        (Some(user), Some(cap)) => {
            let clamped = user.min(cap as usize);
            if clamped < user {
                info!(
                    "Clamping max_num_batched_tokens {} -> {} (largest resident prefill bucket)",
                    user, clamped
                );
            }
            clamped
        }
        (Some(user), None) => user,
        (None, Some(cap)) => {
            info!(
                "Device-selected max_num_batched_tokens = {} (largest resident prefill bucket)",
                cap
            );
            cap as usize
        }
        (None, None) => resolve_default_max_num_batched_tokens(is_offline),
    };
    // Sliding-window models: the sliding groups free out-of-window blocks once
    // per scheduler step (chunk), so a chunk holds up to `window + chunk` tokens
    // of sliding KV transiently before the next step frees the tail. On the
    // lazy-commit metal substrate that transient set is what commits, so the
    // prefill chunk bounds peak KV. Cap it well above the window (small chunks
    // are correct but force one forward per `cap` tokens → slow prefill);
    // measured ~234 MiB at 18k / chunk 512, scaling ~linearly in the chunk, so
    // 4096 keeps 32k well under 1 GiB while cutting the forward count ~8×.
    let swa_chunk_cap = std::env::var("SWA_CHUNK_CAP")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(SWA_PREFILL_CHUNK_CAP);
    let max_num_batched_tokens = if hf_config.is_swa_hybrid() {
        max_num_batched_tokens.min(swa_chunk_cap)
    } else {
        max_num_batched_tokens
    };
    // The concurrency the scheduler will actually admit under. A backend whose decode-width ladder is
    // BAKED (spyre's `BATCH_RUNGS`) cannot grow it to cover `max_num_seqs` the way cuda's captured-graph
    // ladder does, so it reports its widest batchable width and the scheduler is held to it — the same
    // "fast path covers the configured concurrency" invariant cuda gets for free, enforced from the other
    // end. Past the ladder a backend still runs CORRECTLY (spyre degrades to one launch per request), so
    // this is a throughput guard, never a correctness one — which is why it caps rather than refusing.
    let max_num_seqs = resolve_max_num_seqs(config.max_num_seqs, worker_max_num_seqs, is_offline);
    let engine_config = EngineCoreConfig {
        scheduler_config: SchedulerConfig {
            // Resolved above: the device-selected prefill bucket on pruning
            // backends (Metal), else the device-aware CUDA default.
            max_num_batched_tokens,
            max_num_seqs,
            policy: SchedulerPolicy::Fcfs,
            enable_chunked_prefill: true,
            async_scheduling: Some(use_async_scheduling),
            num_lookahead_tokens: if config.speculative_model.is_some() {
                config.num_speculative_tokens
            } else {
                0
            },
            ..Default::default()
        },
        max_model_len,
        num_gpu_blocks,
        block_size,
        engine_index: 0,
        async_scheduling: use_async_scheduling,
        use_spec_decode: config.speculative_model.is_some(),
        proposer_config: build_proposer_config(config, max_model_len)?,
        eos_token_ids,
        is_pooling: config.runner == "pooling",
        enable_prefix_caching,
        hybrid_kv: swa_hybrid_kv,
    };

    let client = InprocClient::new(engine_config, Box::new(executor));

    let loaded_tokenizer = preloaded_tokenizer
        .map(Tokenizer::from_hf_tokenizer)
        .or_else(|| {
            model_dir
                .as_ref()
                .and_then(|dir| match try_load_tokenizer(dir) {
                    Ok(tok) => Some(tok),
                    Err(e) => {
                        info!("No tokenizer found ({}), running without", e);
                        None
                    }
                })
        });

    let tokenizer = loaded_tokenizer.map(|mut tok| {
        if let Some(dir) = model_dir.as_ref() {
            patch_additional_special_tokens(&mut tok, dir);
        }
        info!("Tokenizer loaded");
        Arc::new(tok)
    });

    Ok(InitializedCore {
        client,
        tokenizer,
        model_name,
        max_model_len,
        model_dir,
        hf_config,
    })
}

/// TP variant of `initialize_core`: spawns one worker per GPU, sets up NCCL,
/// profiles memory, warms up, and returns an `InitializedCore` with a
/// `ThreadPoolExecutor`-backed `InprocClient`.
fn initialize_core_tp(config: &VllmConfig) -> Result<InitializedCore> {
    #[cfg(not(feature = "nccl"))]
    {
        let _ = config;
        anyhow::bail!(
            "Tensor parallelism requires the `nccl` feature; \
             rebuild with --features nccl"
        );
    }

    #[cfg(feature = "nccl")]
    {
        use scratchy_serving_worker::gpu_worker::CudaWorker;
        use scratchy_serving_worker::parallel::ResolvedParallelConfig;
        use scratchy_serving_worker::threadpool::ThreadPoolExecutor;
        use scratchy_serving_worker::worker_factory::WorkerCreateConfig;

        let tp_size = config.tensor_parallel_size;
        let model_name = extract_model_name(&config.model).into_owned();
        let is_pooling = config.runner == "pooling";

        info!("Tensor parallelism: {} GPUs", tp_size);

        let nccl_id =
            scratchy_target_cuda::NcclId::new().context("failed to generate NCCL unique ID")?;

        let worker_configs: Vec<WorkerCreateConfig> = (0..tp_size)
            .map(|rank| WorkerCreateConfig {
                model_path: config.model.clone(),
                dtype: config.dtype.clone(),
                hf_token: config.hf_token.clone(),
                block_size: config.block_size,
                device_id: rank as i32,
                max_num_seqs: resolve_max_num_seqs(config.max_num_seqs, None, false),
                enforce_eager: config.enforce_eager,
                max_num_batched_tokens: config.max_num_batched_tokens.unwrap_or(2048),
                cuda_graph_sizes: config
                    .cuda_graph_config
                    .as_ref()
                    .map(|c| c.capture_sizes.clone())
                    .unwrap_or_default(),
                cublas_autotune: config.cublas_autotune,
                gpu_memory_utilization: config.gpu_memory_utilization,
                pooling_strategy: config.pooling_strategy.clone(),
                is_pooling,
                tp_rank: rank,
                tp_world_size: tp_size,
                pp_rank: 0,
                pp_size: 1,
                gguf_file: config.gguf_file.clone(),
                lora_adapter: config.lora_adapter.clone(),
                kv_cache_dtype: config.kv_cache_dtype.clone(),
                calculate_kv_scales: config.calculate_kv_scales,
                cuda_graph_mode: config
                    .cuda_graph_mode
                    .parse()
                    .unwrap_or(CudaGraphMode::Auto),
                eos_token_ids: vec![],
                max_model_len: config.max_model_len,
                draft_model_path: spec_decode_draft_model_path(config),
                draft_model_dtype: config.draft_model_dtype.clone(),
            })
            .collect();

        let download_barrier = std::sync::Arc::new(std::sync::Barrier::new(tp_size));

        let handles: Vec<_> = worker_configs
            .into_iter()
            .enumerate()
            .map(|(local_rank, cfg)| {
                let barrier = download_barrier.clone();
                std::thread::spawn(move || -> Result<CudaWorker> {
                    let mut worker = CudaWorker::new(cfg);
                    worker.init_device().context("init_device failed")?;
                    if local_rank == 0 {
                        worker.load_model().context("load_model failed")?;
                        barrier.wait();
                    } else {
                        barrier.wait();
                        worker.load_model().context("load_model failed")?;
                    }
                    Ok(worker)
                })
            })
            .collect();

        let mut scratchy_workers: Vec<CudaWorker> = Vec::with_capacity(tp_size);
        let mut hf_config = None;
        let mut model_dir = None;
        let mut dtype_elem_bytes: usize = 2;
        // GGUF sources carry the tokenizer in the file's metadata, not as
        // a sibling `tokenizer.json` on disk. `MetalWorker::load_model`
        // reconstructs it and stashes it in `preloaded_tokenizer`. The
        // tp=1 path (`initialize_core`) harvests this via
        // `take_preloaded_tokenizer()` and falls back to
        // `try_load_tokenizer(dir)` only when there's no preloaded one.
        // The tp>1 path previously skipped this and went straight to
        // `try_load_tokenizer` — which returns `None` for GGUF (no
        // sibling tokenizer.json) and silently dropped the real
        // tokenizer, leaving the engine with a byte-level fallback that
        // tokenized `<|begin_of_text|>` as literal ASCII bytes (prompt
        // became 235 garbage tokens, all downstream inference followed).
        let mut preloaded_tokenizer: Option<tokenizers::Tokenizer> = None;

        for (rank, handle) in handles.into_iter().enumerate() {
            let mut worker = handle
                .join()
                .map_err(|_| anyhow::anyhow!("worker thread {rank} panicked"))?
                .with_context(|| format!("worker {rank} init failed"))?;
            if rank == 0 {
                hf_config = worker.hf_config().cloned();
                model_dir = worker.model_dir().map(|p| p.to_path_buf());
                dtype_elem_bytes = worker.resolved_dtype_elem_bytes();
                preloaded_tokenizer = worker.take_preloaded_tokenizer();
            }
            scratchy_workers.push(worker);
        }

        let hf_config = hf_config.context("model config not available after load")?;
        let max_model_len = config
            .max_model_len
            .or(hf_config.max_position_embeddings())
            .unwrap_or(4096);

        info!(
            "Model: {}, max_model_len={}, tp={}",
            model_name, max_model_len, tp_size
        );

        // NCCL init + memory profiling on persistent threads.
        let init_results: Vec<Result<(usize, CudaWorker)>> = std::thread::scope(|s| {
            let handles: Vec<_> = scratchy_workers
                .into_iter()
                .enumerate()
                .map(|(rank, mut worker)| {
                    s.spawn(move || -> Result<(usize, CudaWorker)> {
                        let device = worker.device_ref().expect("device not initialized");
                        unsafe {
                            scratchy_target_cuda::driver::ctx_set_current(device.ctx).unwrap();
                        }
                        let nccl_group = scratchy_target_cuda::NcclGroup::new(
                            rank,
                            tp_size,
                            nccl_id,
                            device.compute_stream,
                        )
                        .context("NCCL comm init failed")?;
                        worker.set_tp_group(std::sync::Arc::new(nccl_group));
                        let avail = worker.determine_available_memory().map_err(|e| {
                            anyhow::anyhow!("determine_available_memory rank {rank}: {e}")
                        })?;
                        Ok((avail, worker))
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });

        let mut workers: Vec<Box<dyn Worker>> = Vec::with_capacity(tp_size);
        let mut min_avail = usize::MAX;
        let mut supports_hybrid_swa_kv = true;
        for res in init_results {
            let (avail, worker) = res?;
            min_avail = min_avail.min(avail);
            supports_hybrid_swa_kv = worker.supports_hybrid_swa_kv();
            workers.push(Box::new(worker));
        }

        let (num_gpu_blocks, swa_hybrid_kv) = compute_kv_blocks(
            min_avail,
            config.block_size,
            &hf_config,
            dtype_elem_bytes,
            config.gpu_memory_utilization,
            &config.kv_cache_dtype,
            max_model_len,
            supports_hybrid_swa_kv,
        );

        for w in &mut workers {
            w.initialize_cache(num_gpu_blocks, 0)
                .context("initialize_cache")?;
        }

        info!(
            "TP: min available memory across ranks: {:.1} GB, num_gpu_blocks={}",
            min_avail as f64 / (1024.0 * 1024.0 * 1024.0),
            num_gpu_blocks,
        );

        // Warm up concurrently (NCCL collectives need all ranks).
        let warmup_results: Vec<Result<()>> = std::thread::scope(|s| {
            let handles: Vec<_> = workers
                .iter_mut()
                .enumerate()
                .map(|(rank, w)| {
                    s.spawn(move || {
                        w.compile_or_warm_up_model()
                            .with_context(|| format!("compile_or_warm_up_model rank {rank}"))
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        for res in warmup_results {
            res?;
        }

        let parallel_config = ResolvedParallelConfig::tensor_parallel(tp_size, 0);
        let executor = ThreadPoolExecutor::new(workers, parallel_config);

        let use_async_scheduling =
            !config.disable_async_scheduling && !spec_decode_requires_sync(config);
        let enable_prefix_caching = config.enable_prefix_caching;

        let eos_token_ids: Vec<u32> = hf_config
            .extra
            .get("eos_token_id")
            .map(|v| {
                if let Some(id) = v.as_u64() {
                    vec![id as u32]
                } else if let Some(arr) = v.as_array() {
                    arr.iter()
                        .filter_map(|v| v.as_u64().map(|id| id as u32))
                        .collect()
                } else {
                    vec![]
                }
            })
            .unwrap_or_default();

        let engine_config = EngineCoreConfig {
            scheduler_config: SchedulerConfig {
                max_num_batched_tokens: config.max_num_batched_tokens.unwrap_or(2048),
                max_num_seqs: resolve_max_num_seqs(config.max_num_seqs, None, false),
                policy: SchedulerPolicy::Fcfs,
                enable_chunked_prefill: true,
                async_scheduling: Some(use_async_scheduling),
                num_lookahead_tokens: if config.speculative_model.is_some() {
                    config.num_speculative_tokens
                } else {
                    0
                },
                ..Default::default()
            },
            max_model_len,
            num_gpu_blocks,
            block_size: config.block_size,
            engine_index: 0,
            async_scheduling: use_async_scheduling,
            use_spec_decode: config.speculative_model.is_some(),
            proposer_config: None,
            eos_token_ids,
            is_pooling: config.runner == "pooling",
            enable_prefix_caching,
            hybrid_kv: swa_hybrid_kv,
        };

        let client = InprocClient::new(engine_config, Box::new(executor));

        // Prefer the GGUF-preloaded tokenizer harvested from rank 0's
        // worker above. Falls back to `try_load_tokenizer(dir)` for
        // safetensors where a sibling `tokenizer.json` lives next to
        // the weights. Mirrors `initialize_core`'s tp=1 flow.
        let tokenizer = preloaded_tokenizer
            .map(Tokenizer::from_hf_tokenizer)
            .or_else(|| {
                model_dir
                    .as_ref()
                    .and_then(|dir| try_load_tokenizer(dir).ok())
            })
            .map(|tok| {
                info!("Tokenizer loaded");
                Arc::new(tok)
            });

        Ok(InitializedCore {
            client,
            tokenizer,
            model_name,
            max_model_len,
            model_dir,
            hf_config,
        })
    }
}

/// Initialize the sync stack for offline batch inference (no async overhead).
///
/// Used by [`LLM`](crate::llm::LLM). Returns an [`InprocClient`] that the
/// caller drives directly with `add_request()` + `get_output()`.
/// Load + log the model's `generation_config.json` sampling defaults
/// (Python vLLM `generation_config="auto"` parity).
fn load_generation_defaults(
    model_dir: Option<&std::path::Path>,
) -> scratchy_core_common::sampling::GenerationDefaults {
    let d = model_dir
        .map(scratchy_core_common::sampling::GenerationDefaults::from_model_dir)
        .unwrap_or_default();
    if !d.is_empty() {
        info!(
            "Sampling defaults from generation_config.json: {:?} \
             (request-level values override)",
            d
        );
    }
    d
}

pub fn initialize_stack_sync(config: &VllmConfig) -> Result<InitializedSyncStack> {
    validate_speculative_decoding(config)?;
    let init_start = Instant::now();

    // Create progress bar if logging is below INFO level
    let progress = {
        let show_progress = !tracing::enabled!(tracing::Level::INFO);
        Arc::new(crate::progress::StartupProgress::new(show_progress))
    };

    // Offline LLM/throughput context → device-aware offline batch defaults.
    let core = initialize_core(config, Some(&progress), true)?;
    progress.finish();
    info!(
        "init engine (load model, create kv cache) took {:.2} seconds",
        init_start.elapsed().as_secs_f64()
    );
    let chat_template = core
        .model_dir
        .as_deref()
        .and_then(|dir| resolve_chat_template(config.chat_template.as_deref(), dir));
    let generation_defaults = load_generation_defaults(core.model_dir.as_deref());
    Ok(InitializedSyncStack {
        client: core.client,
        tokenizer: core.tokenizer,
        chat_template,
        model_name: core.model_name,
        max_model_len: core.max_model_len,
        block_size: config.block_size,
        generation_defaults,
    })
}

/// The model's canonical architecture used to select parsers: `model_type` if
/// present (e.g. `qwen3_5_moe`, `gemma4`, `llama`), else the first
/// `architectures` entry. Never the model name/path.
fn model_arch(hf: &HfModelConfig) -> &str {
    hf.model_type
        .as_deref()
        .filter(|s| !s.is_empty())
        .or_else(|| hf.architectures.first().map(String::as_str))
        .unwrap_or("")
}

/// Install a tool-call parser on `engine`, chosen from the model **architecture**,
/// unless one is already set. Called at every engine-construction site so any
/// consumer of the engine — `scr serve` without a flag, `batch`, or a library
/// embedder — gets tool-call parsing for free. An explicit `--tool-call-parser`
/// applied by the caller after `initialize_stack` returns still overrides this.
fn auto_install_tool_parser(engine: &mut AsyncEngine, arch: &str) {
    if engine.tool_parser().is_some() {
        return;
    }
    match crate::tool_parser::detect_tool_parser(arch) {
        Some(name) => match crate::tool_parser::get_tool_parser(name) {
            Ok(parser) => {
                info!("Auto-selected tool-call parser '{name}' (override with --tool-call-parser)");
                engine.set_tool_parser(parser);
            }
            Err(e) => tracing::warn!("tool-call parser '{name}' unavailable: {e}"),
        },
        None => tracing::warn!(
            "no tool-call parser auto-detected for architecture '{arch}'; tool calls will not \
             be extracted (pass --tool-call-parser to force one)"
        ),
    }
}

/// Auto-select a reasoning parser from the model **architecture** so reasoning
/// models route their thinking to the `reasoning` field instead of leaking it
/// into content.
fn auto_install_reasoning_parser(engine: &mut AsyncEngine, arch: &str) {
    let Some(name) = crate::reasoning_parser::detect_reasoning_parser(arch) else {
        return;
    };
    // `<think>` parsers resolve token ids from the vocab; the harmony parser
    // needs none, so an empty vocab is fine when there's no tokenizer.
    let vocab = engine
        .tokenizer()
        .map(|t| t.get_vocab())
        .unwrap_or_default();
    match crate::reasoning_parser::get_reasoning_parser(name, &vocab) {
        Ok(parser) => {
            info!("Auto-selected reasoning parser '{name}'");
            engine.set_reasoning_parser(parser);
        }
        Err(e) => tracing::warn!("reasoning parser '{name}' unavailable: {e}"),
    }
}

pub fn initialize_stack(
    config: &VllmConfig,
    progress: Option<Arc<crate::progress::StartupProgress>>,
) -> Result<InitializedStack> {
    validate_speculative_decoding(config)?;
    let init_start = Instant::now();

    // Create progress bar if not provided and logging is below INFO level
    let progress = progress.unwrap_or_else(|| {
        // Check if tracing is enabled at INFO level or higher
        let show_progress = !tracing::enabled!(tracing::Level::INFO);
        Arc::new(crate::progress::StartupProgress::new(show_progress))
    });

    let tp_size = config.tensor_parallel_size;
    let model_name = extract_model_name(&config.model).into_owned();

    // External launcher: each process is a separate rank with its own GPU.
    if config.distributed_executor_backend == "external_launcher" {
        return initialize_stack_external(config, model_name, init_start);
    }

    // Multi-node TP ("mp" backend): leader runs engine+scheduler, followers
    // run headless. TCP for control plane, NCCL for data plane.
    // Followers (node_rank > 0) should use `initialize_and_run_follower` instead.
    if (config.num_nodes > 1 || config.distributed_executor_backend == "mp")
        && config.node_rank == 0
    {
        return initialize_stack_multinode(config, model_name, init_start);
    }

    let pp_size = config.pipeline_parallel_size;

    // TP+PP or PP-only: multi-GPU path with NCCL P2P for PP and all-reduce for TP.
    if pp_size > 1 {
        return initialize_stack_tp_pp(config, model_name, init_start);
    }

    // TP > 1: multi-GPU path with NCCL (in-process, thread-per-GPU).
    if tp_size > 1 {
        return initialize_stack_tp(config, model_name, init_start);
    }

    // TP=1: single-GPU path.
    progress.set_stage("Initializing backend");
    // Online-server context → keep the (smaller) server batch defaults.
    let core = initialize_core(config, Some(&progress), false)?;

    progress.set_stage("Creating engine");

    let client = Box::new(core.client);

    let model_name = core.model_name;
    let max_model_len = core.max_model_len;

    let engine = if let Some(tokenizer) = core.tokenizer {
        {
            let chat_template = core
                .model_dir
                .as_ref()
                .and_then(|dir| resolve_chat_template(config.chat_template.as_deref(), dir));
            if let Some(tpl) = chat_template {
                info!("Chat template loaded from tokenizer_config.json");
                AsyncEngine::with_tokenizer_and_template(
                    client,
                    model_name.clone(),
                    max_model_len,
                    tokenizer,
                    Arc::new(tpl),
                )
            } else {
                info!("No chat template found, using plain concatenation");
                AsyncEngine::with_tokenizer(client, model_name.clone(), max_model_len, tokenizer)
            }
        }
    } else {
        AsyncEngine::new(client, model_name.clone(), max_model_len)
    };

    let mut engine = engine;
    engine.set_generation_defaults(load_generation_defaults(core.model_dir.as_deref()));
    if !config.disable_async_scheduling && !spec_decode_requires_sync(config) {
        engine.set_async_scheduling(true);
    }
    if config.runner == "pooling" {
        engine.set_is_pooling(true);
        info!("Runner: pooling mode (embedding requests go through scheduler)");
    }

    // Configure multimodal support — the per-arch declaration lives in
    // scratchy-vision::MmMetadata (collected via scratchy-forward-compiler's
    // inventory). scratchy-serving-api names no arch; if no MM arch claims the
    // current HF architectures, the engine stays text-only.
    #[cfg(feature = "multimodal")]
    let hf_config = &core.hf_config;
    #[cfg(feature = "multimodal")]
    if let Some(processor) = crate::multimodal::resolve(
        &hf_config.architectures,
        &hf_config.extra,
        core.model_dir.as_deref(),
    ) {
        info!(
            "Multimodal config: arch={:?}, image_token_id={}, image_size={}, tokens_per_image={}",
            hf_config.architectures,
            processor.image_token_id,
            processor.image_size,
            processor.mm_tokens_per_image,
        );
        engine.set_mm_processor(Some(processor));
    }

    info!(
        "init engine (load model, create kv cache) took {:.2} seconds",
        init_start.elapsed().as_secs_f64()
    );

    // 11. Finish progress bar and return the stack.
    progress.finish();

    auto_install_tool_parser(&mut engine, model_arch(&core.hf_config));
    auto_install_reasoning_parser(&mut engine, model_arch(&core.hf_config));
    Ok(InitializedStack {
        engine: Arc::new(engine),
        model_name,
        max_model_len,
    })
}

/// Multi-node init flow ("mp" backend): leader creates engine+scheduler and
/// broadcasts scheduler output via TCP. Follower runs headless, receiving
/// commands via TCP and participating in NCCL collectives during forward.
///
/// - Leader (node_rank=0): Creates CudaWorker, NCCL comm, cache init,
///   warmup, wraps in `MultiNodeExecutor`, returns `InitializedStack`.
/// - Follower (node_rank>0): Handled by [`initialize_and_run_follower`].
fn initialize_stack_multinode(
    config: &VllmConfig,
    model_name: String,
    init_start: Instant,
) -> Result<InitializedStack> {
    #[cfg(not(feature = "nccl"))]
    {
        let _ = (config, model_name, init_start);
        anyhow::bail!(
            "Multi-node TP requires the `nccl` feature; \
             rebuild with --features nccl"
        );
    }

    #[cfg(feature = "nccl")]
    {
        use scratchy_serving_worker::gpu_worker::CudaWorker;
        use scratchy_serving_worker::multinode::MultiNodeExecutor;
        use scratchy_serving_worker::worker_factory::WorkerCreateConfig;

        let tp_size = config.tensor_parallel_size;
        let node_rank = config.node_rank;
        let num_nodes = config.num_nodes;
        let is_pooling = config.runner == "pooling";

        if node_rank != 0 {
            anyhow::bail!(
                "initialize_stack_multinode called on follower (node_rank={}). \
                 Use initialize_and_run_follower instead.",
                node_rank
            );
        }

        info!(
            "Multi-node TP: leader node, tp_size={}, num_nodes={}, master={}:{}",
            tp_size, num_nodes, config.master_addr, config.master_port
        );

        // Step 1: Exchange NCCL unique ID via TCP store.
        let nccl_id_bytes = scratchy_target_cuda::tcp_store::exchange_nccl_id(
            0,
            tp_size,
            &config.master_addr,
            config.master_port,
        )
        .context("failed to exchange NCCL ID via TCP store")?;
        let nccl_id = scratchy_target_cuda::NcclId::from_raw(nccl_id_bytes);
        info!("Leader: NCCL ID exchanged");

        // Step 2: Create CudaWorker for this node's GPU (rank 0).
        let cuda_config = WorkerCreateConfig {
            model_path: config.model.clone(),
            dtype: config.dtype.clone(),
            hf_token: config.hf_token.clone(),
            block_size: config.block_size,
            device_id: 0,
            max_num_seqs: resolve_max_num_seqs(config.max_num_seqs, None, false),
            enforce_eager: config.enforce_eager,
            max_num_batched_tokens: config.max_num_batched_tokens.unwrap_or(1024),
            cuda_graph_sizes: config
                .cuda_graph_config
                .as_ref()
                .map(|c| c.capture_sizes.clone())
                .unwrap_or_default(),
            cublas_autotune: config.cublas_autotune,
            gpu_memory_utilization: config.gpu_memory_utilization,
            pooling_strategy: config.pooling_strategy.clone(),
            is_pooling,
            tp_rank: 0,
            tp_world_size: tp_size,
            pp_rank: 0,
            pp_size: 1,
            gguf_file: config.gguf_file.clone(),
            lora_adapter: config.lora_adapter.clone(),
            kv_cache_dtype: config.kv_cache_dtype.clone(),
            calculate_kv_scales: config.calculate_kv_scales,
            cuda_graph_mode: config
                .cuda_graph_mode
                .parse()
                .unwrap_or(CudaGraphMode::Auto),
            eos_token_ids: vec![],
            max_model_len: config.max_model_len,
            draft_model_path: spec_decode_draft_model_path(config),
            draft_model_dtype: config.draft_model_dtype.clone(),
        };

        let mut worker = CudaWorker::new(cuda_config);
        worker
            .init_device()
            .context("failed to initialize CUDA device")?;
        worker.load_model().context("failed to load model")?;

        let hf_config = worker
            .hf_config()
            .context("model config not available after load")?
            .clone();
        let model_dir = worker.model_dir().map(|p| p.to_path_buf());
        let dtype_elem_bytes = worker.resolved_dtype_elem_bytes();

        // Step 3: Create NCCL communicator (collective — all ranks participate).
        let device = worker.device_ref().expect("device not initialized");
        unsafe {
            scratchy_target_cuda::driver::ctx_set_current(device.ctx).unwrap();
        }
        let nccl_group =
            scratchy_target_cuda::NcclGroup::new(0, tp_size, nccl_id, device.compute_stream)
                .context("NCCL comm init failed")?;
        worker.set_tp_group(std::sync::Arc::new(nccl_group));
        info!("Leader: NCCL communicator created");

        // Step 4: Profile available memory.
        let available_memory = worker
            .determine_available_memory()
            .context("failed to determine available memory")?;

        // Step 5: All-reduce MIN across ranks via TCP store.
        let min_memory = scratchy_target_cuda::tcp_store::allreduce_min(
            0,
            tp_size,
            available_memory,
            &config.master_addr,
            config.master_port,
        )
        .context("failed to allreduce memory")?;

        let max_model_len = config
            .max_model_len
            .or(hf_config.max_position_embeddings())
            .unwrap_or(4096);

        info!(
            "Leader: available_memory={:.1} GB, min_across_ranks={:.1} GB",
            available_memory as f64 / (1024.0 * 1024.0 * 1024.0),
            min_memory as f64 / (1024.0 * 1024.0 * 1024.0),
        );

        // Step 6: Compute block count.
        let (num_gpu_blocks, swa_hybrid_kv) = compute_kv_blocks(
            min_memory,
            config.block_size,
            &hf_config,
            dtype_elem_bytes,
            config.gpu_memory_utilization,
            &config.kv_cache_dtype,
            max_model_len,
            worker.supports_hybrid_swa_kv(),
        );

        info!(
            "Leader: num_gpu_blocks={}, kv_cache_tokens={}",
            num_gpu_blocks,
            num_gpu_blocks * config.block_size,
        );

        // Step 7: Establish TCP control channel (persistent connections).
        let mut channel = scratchy_target_cuda::TcpControlChannel::establish(
            0,
            tp_size,
            &config.master_addr,
            config.master_port,
        )
        .context("failed to establish TCP control channel")?;
        info!("Leader: TCP control channel established");

        // Step 8: Initialize cache — broadcast command to followers first,
        // then run locally. Both sides participate in any NCCL collectives.
        {
            use scratchy_serving_worker::multinode::ControlMessage;
            let msg = ControlMessage::InitCache {
                num_gpu_blocks,
                num_cpu_blocks: 0,
            };
            let data = bincode::serialize(&msg).context("serialize InitCache")?;
            channel.broadcast(&data).context("broadcast InitCache")?;
        }
        let mut worker: Box<dyn Worker> = Box::new(worker);
        worker
            .initialize_cache(num_gpu_blocks, 0)
            .context("failed to initialize cache")?;

        // Step 9: Warm up — broadcast to followers, then run locally.
        // Both sides' forward passes hit NCCL collectives simultaneously.
        {
            use scratchy_serving_worker::multinode::ControlMessage;
            let msg = ControlMessage::Warmup;
            let data = bincode::serialize(&msg).context("serialize Warmup")?;
            channel.broadcast(&data).context("broadcast Warmup")?;
        }
        worker
            .compile_or_warm_up_model()
            .context("failed to compile or warm up model")?;

        // Step 10: Wrap in UniProcExecutor → MultiNodeExecutor.
        let executor = UniProcExecutor::new_pre_initialized(worker);
        let multi_executor = MultiNodeExecutor::new(Box::new(executor), channel);

        // Step 11: Build engine.
        let eos_token_ids: Vec<u32> = hf_config
            .extra
            .get("eos_token_id")
            .map(|v| {
                if let Some(id) = v.as_u64() {
                    vec![id as u32]
                } else if let Some(arr) = v.as_array() {
                    arr.iter()
                        .filter_map(|v| v.as_u64().map(|id| id as u32))
                        .collect()
                } else {
                    vec![]
                }
            })
            .unwrap_or_default();

        let use_async_scheduling =
            !config.disable_async_scheduling && !spec_decode_requires_sync(config);
        let enable_prefix_caching = config.enable_prefix_caching;
        let engine_config = EngineCoreConfig {
            scheduler_config: SchedulerConfig {
                max_num_batched_tokens: config.max_num_batched_tokens.unwrap_or(1024),
                max_num_seqs: resolve_max_num_seqs(config.max_num_seqs, None, false),
                policy: SchedulerPolicy::Fcfs,
                enable_chunked_prefill: true,
                async_scheduling: Some(use_async_scheduling),
                num_lookahead_tokens: if config.speculative_model.is_some() {
                    config.num_speculative_tokens
                } else {
                    0
                },
                ..Default::default()
            },
            max_model_len,
            num_gpu_blocks,
            block_size: config.block_size,
            engine_index: 0,
            async_scheduling: use_async_scheduling,
            use_spec_decode: config.speculative_model.is_some(),
            proposer_config: None,
            eos_token_ids,
            is_pooling: config.runner == "pooling",
            enable_prefix_caching,
            hybrid_kv: swa_hybrid_kv,
        };

        let client: Box<dyn scratchy_serving_engine::core_client::EngineCoreClient + Send> =
            Box::new(InprocClient::new(engine_config, Box::new(multi_executor)));

        // Load tokenizer and build engine.
        let tokenizer = model_dir
            .as_ref()
            .and_then(|dir| try_load_tokenizer(dir).ok());

        let mut engine = if let Some(tok) = tokenizer {
            let tokenizer = Arc::new(tok);
            if let Some(ref dir) = model_dir {
                if let Some(ct) = resolve_chat_template(config.chat_template.as_deref(), dir) {
                    info!("Chat template loaded from tokenizer_config.json");
                    AsyncEngine::with_tokenizer_and_template(
                        client,
                        model_name.clone(),
                        max_model_len,
                        tokenizer,
                        Arc::new(ct),
                    )
                } else {
                    AsyncEngine::with_tokenizer(
                        client,
                        model_name.clone(),
                        max_model_len,
                        tokenizer,
                    )
                }
            } else {
                AsyncEngine::with_tokenizer(client, model_name.clone(), max_model_len, tokenizer)
            }
        } else {
            AsyncEngine::new(client, model_name.clone(), max_model_len)
        };

        if !config.disable_async_scheduling && !spec_decode_requires_sync(config) {
            engine.set_generation_defaults(load_generation_defaults(model_dir.as_deref()));
            engine.set_async_scheduling(true);
        }
        if config.runner == "pooling" {
            engine.set_is_pooling(true);
        }

        info!(
            "Leader: stack initialized in {:.1}s (multi-node, tp_size={})",
            init_start.elapsed().as_secs_f64(),
            tp_size,
        );

        auto_install_tool_parser(&mut engine, model_arch(&hf_config));
        auto_install_reasoning_parser(&mut engine, model_arch(&hf_config));
        Ok(InitializedStack {
            engine: Arc::new(engine),
            model_name,
            max_model_len,
        })
    }
}

/// Initialize a follower node and run the headless worker loop.
///
/// This function does NOT return (blocks forever in the headless loop)
/// until the leader sends a Shutdown command or the connection drops.
///
/// Called from the CLI when `node_rank > 0` and `num_nodes > 1`.
#[cfg(feature = "nccl")]
pub fn initialize_and_run_follower(config: &VllmConfig) -> Result<()> {
    use scratchy_serving_worker::gpu_worker::CudaWorker;
    use scratchy_serving_worker::worker_factory::WorkerCreateConfig;

    validate_speculative_decoding(config)?;

    let tp_size = config.tensor_parallel_size;
    let node_rank = config.node_rank;
    let is_pooling = config.runner == "pooling";

    info!(
        "Multi-node TP: follower node_rank={}, tp_size={}, master={}:{}",
        node_rank, tp_size, config.master_addr, config.master_port
    );

    // Step 1: Exchange NCCL unique ID via TCP store.
    let nccl_id_bytes = scratchy_target_cuda::tcp_store::exchange_nccl_id(
        node_rank,
        tp_size,
        &config.master_addr,
        config.master_port,
    )
    .context("failed to exchange NCCL ID via TCP store")?;
    let nccl_id = scratchy_target_cuda::NcclId::from_raw(nccl_id_bytes);
    info!("Follower {}: NCCL ID exchanged", node_rank);

    // Step 2: Create CudaWorker for this node's GPU.
    let cuda_config = WorkerCreateConfig {
        model_path: config.model.clone(),
        dtype: config.dtype.clone(),
        hf_token: config.hf_token.clone(),
        block_size: config.block_size,
        device_id: 0, // Each node has 1 GPU at device 0.
        max_num_seqs: resolve_max_num_seqs(config.max_num_seqs, None, false),
        enforce_eager: config.enforce_eager,
        max_num_batched_tokens: config.max_num_batched_tokens.unwrap_or(1024),
        cuda_graph_sizes: config
            .cuda_graph_config
            .as_ref()
            .map(|c| c.capture_sizes.clone())
            .unwrap_or_default(),
        cublas_autotune: config.cublas_autotune,
        gpu_memory_utilization: config.gpu_memory_utilization,
        pooling_strategy: config.pooling_strategy.clone(),
        is_pooling,
        tp_rank: node_rank,
        tp_world_size: tp_size,
        gguf_file: config.gguf_file.clone(),
        lora_adapter: config.lora_adapter.clone(),
        kv_cache_dtype: config.kv_cache_dtype.clone(),
        calculate_kv_scales: config.calculate_kv_scales,
        pp_rank: 0,
        pp_size: 1,
        cuda_graph_mode: config
            .cuda_graph_mode
            .parse()
            .unwrap_or(CudaGraphMode::Auto),
        eos_token_ids: vec![],
        max_model_len: config.max_model_len,
        draft_model_path: spec_decode_draft_model_path(config),
        draft_model_dtype: config.draft_model_dtype.clone(),
    };

    let mut worker = CudaWorker::new(cuda_config);
    worker
        .init_device()
        .context("failed to initialize CUDA device")?;
    worker.load_model().context("failed to load model")?;

    // Step 3: Create NCCL communicator (collective — all ranks participate).
    let device = worker.device_ref().expect("device not initialized");
    unsafe {
        scratchy_target_cuda::driver::ctx_set_current(device.ctx).unwrap();
    }
    let nccl_group =
        scratchy_target_cuda::NcclGroup::new(node_rank, tp_size, nccl_id, device.compute_stream)
            .context("NCCL comm init failed")?;
    worker.set_tp_group(std::sync::Arc::new(nccl_group));
    info!("Follower {}: NCCL communicator created", node_rank);

    // Step 4: Profile available memory.
    let available_memory = worker
        .determine_available_memory()
        .context("failed to determine available memory")?;

    // Step 5: All-reduce MIN across ranks via TCP store.
    let min_memory = scratchy_target_cuda::tcp_store::allreduce_min(
        node_rank,
        tp_size,
        available_memory,
        &config.master_addr,
        config.master_port,
    )
    .context("failed to allreduce memory")?;

    info!(
        "Follower {}: available_memory={:.1} GB, min_across_ranks={:.1} GB",
        node_rank,
        available_memory as f64 / (1024.0 * 1024.0 * 1024.0),
        min_memory as f64 / (1024.0 * 1024.0 * 1024.0),
    );

    // Step 6: Establish TCP control channel.
    let channel = scratchy_target_cuda::TcpControlChannel::establish(
        node_rank,
        tp_size,
        &config.master_addr,
        config.master_port,
    )
    .context("failed to establish TCP control channel")?;
    info!("Follower {}: TCP control channel established", node_rank);

    // Step 7: Wrap in UniProcExecutor and enter headless loop.
    // The headless loop receives InitCache, Warmup, ExecuteModel, and Shutdown
    // commands from the leader via TCP. NCCL collectives in the forward pass
    // synchronize with the leader automatically.
    let executor = UniProcExecutor::new_pre_initialized(Box::new(worker));

    info!("Follower {}: entering headless loop", node_rank,);
    crate::headless::run_headless(executor, channel)?;

    info!(
        "Follower {}: headless loop exited, shutting down",
        node_rank
    );
    Ok(())
}

/// Multi-GPU init flow: creates N workers (one per GPU rank), inits NCCL,
/// and wraps them in a ThreadPoolExecutor.
///
/// Each worker loads the full model weights and keeps only its shard
/// Initialize full stack for TP+PP or PP-only (pipeline parallelism).
///
/// Creates `tp_size * pp_size` workers, each with a TP NCCL comm and a PP
/// NCCL comm. Workers load only their PP stage's layers.
fn initialize_stack_tp_pp(
    config: &VllmConfig,
    model_name: String,
    init_start: Instant,
) -> Result<InitializedStack> {
    let _ = (config, model_name, init_start);
    anyhow::bail!(
        "Pipeline parallelism is not supported on the scratchy-forward-compiler forwards \
         (the previous PP path lived in `scratchy_target_cuda::model::*` and has been \
         removed). Run with `--pipeline-parallel-size 1`."
    );
    #[allow(unreachable_code)]
    {
        #[cfg(not(feature = "nccl"))]
        {
            let _ = (config, model_name, init_start);
            anyhow::bail!(
                "Pipeline parallelism requires the `nccl` feature; \
             rebuild with --features nccl"
            );
        }

        #[cfg(feature = "nccl")]
        {
            use scratchy_serving_worker::gpu_worker::CudaWorker;
            use scratchy_serving_worker::parallel::ResolvedParallelConfig;
            use scratchy_serving_worker::threadpool::ThreadPoolExecutor;
            use scratchy_serving_worker::worker_factory::WorkerCreateConfig;

            let tp_size = config.tensor_parallel_size;
            let pp_size = config.pipeline_parallel_size;
            let world_size = tp_size * pp_size;
            let is_pooling = config.runner == "pooling";

            info!(
                "Pipeline parallelism: {} PP stages × {} TP ranks = {} GPUs",
                pp_size, tp_size, world_size
            );

            // Generate NCCL unique IDs:
            // - One TP NcclId per PP stage (pp_size total — ranks in same stage share it).
            // - One PP NcclId per TP position (tp_size total — ranks with same tp_rank share it).
            let tp_nccl_ids: Vec<scratchy_target_cuda::NcclId> = (0..pp_size)
                .map(|_| {
                    scratchy_target_cuda::NcclId::new().context("failed to generate TP NCCL ID")
                })
                .collect::<Result<Vec<_>>>()?;
            let pp_nccl_ids: Vec<scratchy_target_cuda::NcclId> = (0..tp_size)
                .map(|_| {
                    scratchy_target_cuda::NcclId::new().context("failed to generate PP NCCL ID")
                })
                .collect::<Result<Vec<_>>>()?;

            // Build per-rank configs.
            // Rank layout: global_rank = pp_rank * tp_size + tp_rank.
            let worker_configs: Vec<WorkerCreateConfig> = (0..world_size)
                .map(|global_rank| {
                    let tp_rank = global_rank % tp_size;
                    let pp_rank = global_rank / tp_size;
                    WorkerCreateConfig {
                        model_path: config.model.clone(),
                        dtype: config.dtype.clone(),
                        hf_token: config.hf_token.clone(),
                        block_size: config.block_size,
                        device_id: global_rank as i32,
                        max_num_seqs: resolve_max_num_seqs(config.max_num_seqs, None, false),
                        enforce_eager: config.enforce_eager,
                        max_num_batched_tokens: config.max_num_batched_tokens.unwrap_or(1024),
                        cuda_graph_sizes: config
                            .cuda_graph_config
                            .as_ref()
                            .map(|c| c.capture_sizes.clone())
                            .unwrap_or_default(),
                        cublas_autotune: config.cublas_autotune,
                        gpu_memory_utilization: config.gpu_memory_utilization,
                        pooling_strategy: config.pooling_strategy.clone(),
                        is_pooling,
                        tp_rank,
                        tp_world_size: tp_size,
                        pp_rank,
                        pp_size,
                        gguf_file: config.gguf_file.clone(),
                        lora_adapter: config.lora_adapter.clone(),
                        kv_cache_dtype: config.kv_cache_dtype.clone(),
                        calculate_kv_scales: config.calculate_kv_scales,
                        cuda_graph_mode: config
                            .cuda_graph_mode
                            .parse()
                            .unwrap_or(CudaGraphMode::Auto),
                        eos_token_ids: vec![],
                        max_model_len: config.max_model_len,
                        draft_model_path: spec_decode_draft_model_path(config),
                        draft_model_dtype: config.draft_model_dtype.clone(),
                    }
                })
                .collect();

            // Download barrier: rank 0 downloads first, others wait.
            let download_barrier = std::sync::Arc::new(std::sync::Barrier::new(world_size));

            // Phase 1: Spawn one thread per GPU — init device + load model (PP-aware).
            let handles: Vec<_> = worker_configs
                .into_iter()
                .enumerate()
                .map(|(global_rank, cfg)| {
                    let barrier = download_barrier.clone();
                    std::thread::spawn(move || -> Result<CudaWorker> {
                        let mut worker = CudaWorker::new(cfg);
                        worker.init_device().context("init_device failed")?;

                        // Rank 0 loads first (downloads model files to cache).
                        if global_rank == 0 {
                            worker.load_model().context("load_model failed")?;
                            barrier.wait();
                        } else {
                            barrier.wait();
                            worker.load_model().context("load_model failed")?;
                        }

                        Ok(worker)
                    })
                })
                .collect();

            // Collect workers.
            let mut scratchy_workers: Vec<CudaWorker> = Vec::with_capacity(world_size);
            let mut hf_config = None;
            let mut model_dir = None;
            let mut dtype_elem_bytes: usize = 2; // BF16 default

            for (rank, handle) in handles.into_iter().enumerate() {
                let worker = handle
                    .join()
                    .map_err(|_| anyhow::anyhow!("worker thread {rank} panicked"))?
                    .with_context(|| format!("worker {rank} init failed"))?;

                if rank == 0 {
                    hf_config = worker.hf_config().cloned();
                    model_dir = worker.model_dir().map(|p| p.to_path_buf());
                    dtype_elem_bytes = worker.resolved_dtype_elem_bytes();
                }
                scratchy_workers.push(worker);
            }

            let hf_config = hf_config.context("model config not available after load")?;

            let max_model_len = config
                .max_model_len
                .or(hf_config.max_position_embeddings())
                .unwrap_or(4096);
            let num_layers = resolve_num_layers(&hf_config);

            info!(
                "Model: {}, max_model_len={}, num_layers={}, tp={}, pp={}",
                model_name, max_model_len, num_layers, tp_size, pp_size
            );

            // Phase 2: Create NCCL comms (TP + PP) + profile memory + warmup.
            // All NCCL init calls require ranks in the same group to participate
            // simultaneously, so we use scoped threads.
            //
            // We create TP comms first (all ranks in each TP group sync), then PP
            // comms (all ranks in each PP group sync). This ordering ensures no
            // deadlock since all ranks follow the same order.
            let init_results: Vec<Result<(usize, CudaWorker)>> = std::thread::scope(|s| {
                let handles: Vec<_> = scratchy_workers
                    .into_iter()
                    .enumerate()
                    .map(|(global_rank, mut worker)| {
                        let tp_nccl_ids = &tp_nccl_ids;
                        let pp_nccl_ids = &pp_nccl_ids;
                        s.spawn(move || -> Result<(usize, CudaWorker)> {
                            let tp_rank = global_rank % tp_size;
                            let pp_rank = global_rank / tp_size;

                            // Set CUDA context.
                            let device = worker.device_ref().expect("device not initialized");
                            unsafe {
                                scratchy_target_cuda::driver::ctx_set_current(device.ctx).unwrap();
                            }

                            // Create TP NCCL comm (ranks in the same PP stage).
                            if tp_size > 1 {
                                let tp_nccl_id = tp_nccl_ids[pp_rank];
                                let nccl_group = scratchy_target_cuda::NcclGroup::new(
                                    tp_rank,
                                    tp_size,
                                    tp_nccl_id,
                                    device.compute_stream,
                                )
                                .with_context(|| {
                                    format!("TP NCCL comm init failed for rank {global_rank}")
                                })?;
                                worker.set_tp_group(std::sync::Arc::new(nccl_group));
                            }

                            // Create PP NCCL comm (ranks with the same TP rank).
                            let pp_nccl_id = pp_nccl_ids[tp_rank];
                            let device = worker.device_ref().unwrap();
                            let pp_group = scratchy_target_cuda::NcclGroup::new(
                                pp_rank,
                                pp_size,
                                pp_nccl_id,
                                device.compute_stream,
                            )
                            .with_context(|| {
                                format!("PP NCCL comm init failed for rank {global_rank}")
                            })?;
                            worker.set_pp_group(std::sync::Arc::new(pp_group));

                            // Allocate PP recv buffers on non-first stages.
                            worker.allocate_pp_recv_buffers();

                            // Profile activation memory with dummy forward.
                            let avail = worker.determine_available_memory().map_err(|e| {
                                anyhow::anyhow!(
                                    "determine_available_memory rank {global_rank}: {e}"
                                )
                            })?;

                            Ok((avail, worker))
                        })
                    })
                    .collect();
                handles.into_iter().map(|h| h.join().unwrap()).collect()
            });

            let mut workers: Vec<Box<dyn Worker>> = Vec::with_capacity(world_size);
            let mut min_avail = usize::MAX;
            let mut supports_hybrid_swa_kv = true;
            for res in init_results {
                let (avail, worker) = res?;
                min_avail = min_avail.min(avail);
                supports_hybrid_swa_kv = worker.supports_hybrid_swa_kv();
                workers.push(Box::new(worker));
            }

            // num_gpu_blocks = min across ALL workers (matches Python).
            let (num_gpu_blocks, swa_hybrid_kv) = compute_kv_blocks(
                min_avail,
                config.block_size,
                &hf_config,
                dtype_elem_bytes,
                config.gpu_memory_utilization,
                &config.kv_cache_dtype,
                max_model_len,
                supports_hybrid_swa_kv,
            );

            // Initialize cache: each worker allocates KV for its layer count only.
            for w in &mut workers {
                w.initialize_cache(num_gpu_blocks, 0)
                    .context("initialize_cache")?;
            }

            info!(
                "TP+PP: min available memory across ranks: {:.1} GB, num_gpu_blocks={}",
                min_avail as f64 / (1024.0 * 1024.0 * 1024.0),
                num_gpu_blocks,
            );

            // Warm up (CUDA graph capture) concurrently — forwards use NCCL collectives.
            let warmup_results: Vec<Result<()>> = std::thread::scope(|s| {
                let handles: Vec<_> = workers
                    .iter_mut()
                    .enumerate()
                    .map(|(rank, w)| {
                        s.spawn(move || {
                            w.compile_or_warm_up_model()
                                .with_context(|| format!("compile_or_warm_up_model rank {rank}"))
                        })
                    })
                    .collect();
                handles.into_iter().map(|h| h.join().unwrap()).collect()
            });
            for res in warmup_results {
                res?;
            }

            // Wrap in ThreadPoolExecutor with TP+PP config.
            let parallel_config =
                ResolvedParallelConfig::tensor_pipeline_parallel(tp_size, pp_size, 0);
            let executor = ThreadPoolExecutor::new(workers, parallel_config);

            // Build engine (same as TP-only path).
            let eos_token_ids: Vec<u32> = hf_config
                .extra
                .get("eos_token_id")
                .map(|v| {
                    if let Some(id) = v.as_u64() {
                        vec![id as u32]
                    } else if let Some(arr) = v.as_array() {
                        arr.iter()
                            .filter_map(|v| v.as_u64().map(|id| id as u32))
                            .collect()
                    } else {
                        vec![]
                    }
                })
                .unwrap_or_default();

            // PP: force sync scheduling — async scheduling with PP requires token broadcast
            // from last stage to non-last stages, which is not yet implemented.
            let use_async_scheduling = false;
            let enable_prefix_caching = config.enable_prefix_caching;
            let engine_config = EngineCoreConfig {
                scheduler_config: SchedulerConfig {
                    max_num_batched_tokens: config.max_num_batched_tokens.unwrap_or(1024),
                    max_num_seqs: resolve_max_num_seqs(config.max_num_seqs, None, false),
                    policy: SchedulerPolicy::Fcfs,
                    enable_chunked_prefill: true,
                    async_scheduling: Some(use_async_scheduling),
                    num_lookahead_tokens: if config.speculative_model.is_some() {
                        config.num_speculative_tokens
                    } else {
                        0
                    },
                    use_pp: true,
                    ..Default::default()
                },
                max_model_len,
                num_gpu_blocks,
                block_size: config.block_size,
                engine_index: 0,
                async_scheduling: use_async_scheduling,
                use_spec_decode: config.speculative_model.is_some(),
                proposer_config: None,
                eos_token_ids,
                is_pooling: config.runner == "pooling",
                enable_prefix_caching,
                hybrid_kv: swa_hybrid_kv,
            };

            let client: Box<dyn scratchy_serving_engine::core_client::EngineCoreClient + Send> =
                Box::new(InprocClient::new(engine_config, Box::new(executor)));

            // Load tokenizer and build engine.
            let tokenizer = model_dir
                .as_ref()
                .and_then(|dir| try_load_tokenizer(dir).ok());

            let mut engine = if let Some(tok) = tokenizer {
                let tokenizer = Arc::new(tok);
                if let Some(ref dir) = model_dir {
                    if let Some(ct) = resolve_chat_template(config.chat_template.as_deref(), dir) {
                        info!("Chat template loaded from tokenizer_config.json");
                        AsyncEngine::with_tokenizer_and_template(
                            client,
                            model_name.clone(),
                            max_model_len,
                            tokenizer,
                            Arc::new(ct),
                        )
                    } else {
                        AsyncEngine::with_tokenizer(
                            client,
                            model_name.clone(),
                            max_model_len,
                            tokenizer,
                        )
                    }
                } else {
                    AsyncEngine::with_tokenizer(
                        client,
                        model_name.clone(),
                        max_model_len,
                        tokenizer,
                    )
                }
            } else {
                AsyncEngine::new(client, model_name.clone(), max_model_len)
            };

            if false {
                // PP: sync scheduling forced
                engine.set_generation_defaults(load_generation_defaults(model_dir.as_deref()));
                engine.set_async_scheduling(true);
            }
            if config.runner == "pooling" {
                engine.set_is_pooling(true);
            }

            auto_install_tool_parser(&mut engine, model_arch(&hf_config));
            auto_install_reasoning_parser(&mut engine, model_arch(&hf_config));
            let engine = Arc::new(engine);

            info!(
                "Stack initialized with TP={}, PP={} in {:.1}s",
                tp_size,
                pp_size,
                init_start.elapsed().as_secs_f64()
            );

            Ok(InitializedStack {
                engine,
                model_name,
                max_model_len,
            })
        }
    } // close `#[allow(unreachable_code)] {` wrapper
}

/// (via ColumnParallelLinear/RowParallelLinear sharding at load time).
///
/// TODO: Port to CudaWorker.
fn initialize_stack_tp(
    config: &VllmConfig,
    model_name: String,
    init_start: Instant,
) -> Result<InitializedStack> {
    #[cfg(not(feature = "nccl"))]
    {
        let _ = (config, model_name, init_start);
        anyhow::bail!(
            "Tensor parallelism requires the `nccl` feature; \
             rebuild with --features nccl"
        );
    }

    #[cfg(feature = "nccl")]
    {
        use scratchy_serving_worker::gpu_worker::CudaWorker;
        use scratchy_serving_worker::parallel::ResolvedParallelConfig;
        use scratchy_serving_worker::threadpool::ThreadPoolExecutor;
        use scratchy_serving_worker::worker_factory::WorkerCreateConfig;

        let tp_size = config.tensor_parallel_size;
        let is_pooling = config.runner == "pooling";

        info!("Tensor parallelism: {} GPUs", tp_size);

        // Generate NCCL unique ID on rank 0.
        let nccl_id =
            scratchy_target_cuda::NcclId::new().context("failed to generate NCCL unique ID")?;

        // Build per-rank configs.
        let worker_configs: Vec<WorkerCreateConfig> = (0..tp_size)
            .map(|rank| WorkerCreateConfig {
                model_path: config.model.clone(),
                dtype: config.dtype.clone(),
                hf_token: config.hf_token.clone(),
                block_size: config.block_size,
                device_id: rank as i32,
                max_num_seqs: resolve_max_num_seqs(config.max_num_seqs, None, false),
                enforce_eager: config.enforce_eager,
                max_num_batched_tokens: config.max_num_batched_tokens.unwrap_or(2048),
                cuda_graph_sizes: config
                    .cuda_graph_config
                    .as_ref()
                    .map(|c| c.capture_sizes.clone())
                    .unwrap_or_default(),
                cublas_autotune: config.cublas_autotune,
                gpu_memory_utilization: config.gpu_memory_utilization,
                pooling_strategy: config.pooling_strategy.clone(),
                is_pooling,
                tp_rank: rank,
                tp_world_size: tp_size,
                pp_rank: 0,
                pp_size: 1,
                gguf_file: config.gguf_file.clone(),
                lora_adapter: config.lora_adapter.clone(),
                kv_cache_dtype: config.kv_cache_dtype.clone(),
                calculate_kv_scales: config.calculate_kv_scales,
                cuda_graph_mode: config
                    .cuda_graph_mode
                    .parse()
                    .unwrap_or(CudaGraphMode::Auto),
                eos_token_ids: vec![],
                max_model_len: config.max_model_len,
                draft_model_path: spec_decode_draft_model_path(config),
                draft_model_dtype: config.draft_model_dtype.clone(),
            })
            .collect();

        // Download barrier: rank 0 downloads first, others wait.
        let download_barrier = std::sync::Arc::new(std::sync::Barrier::new(tp_size));

        // Spawn one thread per GPU. Each: init device → load model → create NCCL comm.
        let handles: Vec<_> = worker_configs
            .into_iter()
            .enumerate()
            .map(|(local_rank, cfg)| {
                let barrier = download_barrier.clone();

                std::thread::spawn(move || -> Result<CudaWorker> {
                    let mut worker = CudaWorker::new(cfg);
                    worker.init_device().context("init_device failed")?;

                    // Rank 0 loads first (downloads model files to cache).
                    if local_rank == 0 {
                        worker.load_model().context("load_model failed")?;
                        barrier.wait();
                    } else {
                        barrier.wait();
                        worker.load_model().context("load_model failed")?;
                    }

                    Ok(worker)
                })
            })
            .collect();

        // Collect concrete CudaWorkers (not yet boxed as dyn Worker).
        let mut scratchy_workers: Vec<CudaWorker> = Vec::with_capacity(tp_size);
        let mut hf_config = None;
        let mut model_dir = None;
        let mut dtype_elem_bytes: usize = 2; // BF16 default

        for (rank, handle) in handles.into_iter().enumerate() {
            let worker = handle
                .join()
                .map_err(|_| anyhow::anyhow!("worker thread {rank} panicked"))?
                .with_context(|| format!("worker {rank} init failed"))?;

            if rank == 0 {
                hf_config = worker.hf_config().cloned();
                model_dir = worker.model_dir().map(|p| p.to_path_buf());
                dtype_elem_bytes = worker.resolved_dtype_elem_bytes();
            }
            scratchy_workers.push(worker);
        }

        let hf_config = hf_config.context("model config not available after load")?;

        let max_model_len = config
            .max_model_len
            .or(hf_config.max_position_embeddings())
            .unwrap_or(4096);
        let num_layers = resolve_num_layers(&hf_config);

        info!(
            "Model: {}, max_model_len={}, num_layers={}, tp={}",
            model_name, max_model_len, num_layers, tp_size
        );

        // Create NCCL comms + profile memory + warmup all on persistent threads.
        // NCCL collectives require all ranks to participate simultaneously, so all
        // TP operations (NCCL init, profiling forward, CUDA graph capture) run on
        // dedicated per-rank threads that persist throughout init.
        let init_results: Vec<Result<(usize, CudaWorker)>> = std::thread::scope(|s| {
            let handles: Vec<_> = scratchy_workers
                .into_iter()
                .enumerate()
                .map(|(rank, mut worker)| {
                    s.spawn(move || -> Result<(usize, CudaWorker)> {
                        // Create NCCL communicator (collective — all ranks participate).
                        let device = worker.device_ref().expect("device not initialized");
                        unsafe {
                            scratchy_target_cuda::driver::ctx_set_current(device.ctx).unwrap();
                        }
                        let nccl_group = scratchy_target_cuda::NcclGroup::new(
                            rank,
                            tp_size,
                            nccl_id,
                            device.compute_stream,
                        )
                        .context("NCCL comm init failed")?;
                        worker.set_tp_group(std::sync::Arc::new(nccl_group));

                        // Profile activation memory with dummy forward.
                        let avail = worker.determine_available_memory().map_err(|e| {
                            anyhow::anyhow!("determine_available_memory rank {rank}: {e}")
                        })?;

                        Ok((avail, worker))
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });

        let mut workers: Vec<Box<dyn Worker>> = Vec::with_capacity(tp_size);
        let mut min_avail = usize::MAX;
        let mut supports_hybrid_swa_kv = true;
        for res in init_results {
            let (avail, worker) = res?;
            min_avail = min_avail.min(avail);
            supports_hybrid_swa_kv = worker.supports_hybrid_swa_kv();
            workers.push(Box::new(worker));
        }

        let (num_gpu_blocks, swa_hybrid_kv) = compute_kv_blocks(
            min_avail,
            config.block_size,
            &hf_config,
            dtype_elem_bytes,
            config.gpu_memory_utilization,
            &config.kv_cache_dtype,
            max_model_len,
            supports_hybrid_swa_kv,
        );

        // initialize_cache doesn't run forwards, safe to call sequentially.
        for w in &mut workers {
            w.initialize_cache(num_gpu_blocks, 0)
                .context("initialize_cache")?;
        }

        info!(
            "TP: min available memory across ranks: {:.1} GB, num_gpu_blocks={}",
            min_avail as f64 / (1024.0 * 1024.0 * 1024.0),
            num_gpu_blocks,
        );

        // Warm up (CUDA graph capture) concurrently — forwards use NCCL collectives.
        let warmup_results: Vec<Result<()>> = std::thread::scope(|s| {
            let handles: Vec<_> = workers
                .iter_mut()
                .enumerate()
                .map(|(rank, w)| {
                    s.spawn(move || {
                        w.compile_or_warm_up_model()
                            .with_context(|| format!("compile_or_warm_up_model rank {rank}"))
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        for res in warmup_results {
            res?;
        }

        // Wrap in ThreadPoolExecutor.
        let parallel_config = ResolvedParallelConfig::tensor_parallel(tp_size, 0);
        let executor = ThreadPoolExecutor::new(workers, parallel_config);

        // Build engine.
        let eos_token_ids: Vec<u32> = hf_config
            .extra
            .get("eos_token_id")
            .map(|v| {
                if let Some(id) = v.as_u64() {
                    vec![id as u32]
                } else if let Some(arr) = v.as_array() {
                    arr.iter()
                        .filter_map(|v| v.as_u64().map(|id| id as u32))
                        .collect()
                } else {
                    vec![]
                }
            })
            .unwrap_or_default();

        let use_async_scheduling =
            !config.disable_async_scheduling && !spec_decode_requires_sync(config);
        let enable_prefix_caching = config.enable_prefix_caching;
        let engine_config = EngineCoreConfig {
            scheduler_config: SchedulerConfig {
                max_num_batched_tokens: config.max_num_batched_tokens.unwrap_or(2048),
                max_num_seqs: resolve_max_num_seqs(config.max_num_seqs, None, false),
                policy: SchedulerPolicy::Fcfs,
                enable_chunked_prefill: true,
                async_scheduling: Some(use_async_scheduling),
                num_lookahead_tokens: if config.speculative_model.is_some() {
                    config.num_speculative_tokens
                } else {
                    0
                },
                ..Default::default()
            },
            max_model_len,
            num_gpu_blocks,
            block_size: config.block_size,
            engine_index: 0,
            async_scheduling: use_async_scheduling,
            use_spec_decode: config.speculative_model.is_some(),
            proposer_config: None,
            eos_token_ids,
            is_pooling: config.runner == "pooling",
            enable_prefix_caching,
            hybrid_kv: swa_hybrid_kv,
        };

        let client: Box<dyn scratchy_serving_engine::core_client::EngineCoreClient + Send> =
            Box::new(InprocClient::new(engine_config, Box::new(executor)));

        // Load tokenizer. Apply the same `additional_special_tokens`
        // patch as the single-rank path — Qwen2-VL chat templates
        // reference `<|image_pad|>` / `<|video_pad|>` which live on
        // `tokenizer_config.json::additional_special_tokens` but are
        // missing from `tokenizer.json` for some HF mirrors, so the
        // chat renderer errors out without this patch.
        let tokenizer = model_dir.as_ref().and_then(|dir| {
            try_load_tokenizer(dir).ok().map(|mut tok| {
                patch_additional_special_tokens(&mut tok, dir);
                tok
            })
        });

        let mut engine = if let Some(tok) = tokenizer {
            let tokenizer = Arc::new(tok);
            if let Some(ref dir) = model_dir {
                if let Some(ct) = resolve_chat_template(config.chat_template.as_deref(), dir) {
                    info!("Chat template loaded from tokenizer_config.json");
                    AsyncEngine::with_tokenizer_and_template(
                        client,
                        model_name.clone(),
                        max_model_len,
                        tokenizer,
                        Arc::new(ct),
                    )
                } else {
                    AsyncEngine::with_tokenizer(
                        client,
                        model_name.clone(),
                        max_model_len,
                        tokenizer,
                    )
                }
            } else {
                AsyncEngine::with_tokenizer(client, model_name.clone(), max_model_len, tokenizer)
            }
        } else {
            AsyncEngine::new(client, model_name.clone(), max_model_len)
        };

        if !config.disable_async_scheduling && !spec_decode_requires_sync(config) {
            engine.set_generation_defaults(load_generation_defaults(model_dir.as_deref()));
            engine.set_async_scheduling(true);
        }
        if config.runner == "pooling" {
            engine.set_is_pooling(true);
        }

        // Multimodal config — generic scratchy-vision inventory lookup.
        // See the single-rank `initialize_stack` path for the rationale;
        // this is the same call shape, threaded through the TP-init
        // engine instead.
        #[cfg(feature = "multimodal")]
        if let Some(processor) = crate::multimodal::resolve(
            &hf_config.architectures,
            &hf_config.extra,
            model_dir.as_deref(),
        ) {
            info!(
                "Multimodal config: arch={:?}, image_token_id={}, image_size={}, tokens_per_image={}",
                hf_config.architectures,
                processor.image_token_id,
                processor.image_size,
                processor.mm_tokens_per_image,
            );
            engine.set_mm_processor(Some(processor));
        }

        auto_install_tool_parser(&mut engine, model_arch(&hf_config));
        auto_install_reasoning_parser(&mut engine, model_arch(&hf_config));
        let engine = Arc::new(engine);

        info!(
            "Stack initialized with TP={} in {:.1}s",
            tp_size,
            init_start.elapsed().as_secs_f64()
        );

        Ok(InitializedStack {
            engine,
            model_name,
            max_model_len,
        })
    }
}

/// Parsed distributed environment variables for external launcher mode.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExternalLauncherEnv {
    pub rank: usize,
    pub local_rank: usize,
    pub world_size: usize,
    pub master_addr: String,
    pub master_port: u16,
}

impl ExternalLauncherEnv {
    /// Parse distributed env vars (RANK, LOCAL_RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT).
    ///
    /// Falls back to `config` values for MASTER_ADDR and MASTER_PORT if not set in env.
    pub fn from_env(config: &VllmConfig) -> Result<Self> {
        let rank: usize = std::env::var("RANK")
            .context("RANK env var not set (required for external_launcher)")?
            .parse()
            .context("RANK must be an integer")?;
        let local_rank: usize = std::env::var("LOCAL_RANK")
            .context("LOCAL_RANK env var not set (required for external_launcher)")?
            .parse()
            .context("LOCAL_RANK must be an integer")?;
        let world_size: usize = std::env::var("WORLD_SIZE")
            .context("WORLD_SIZE env var not set (required for external_launcher)")?
            .parse()
            .context("WORLD_SIZE must be an integer")?;
        let master_addr =
            std::env::var("MASTER_ADDR").unwrap_or_else(|_| config.master_addr.clone());
        let master_port: u16 = std::env::var("MASTER_PORT")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(config.master_port);

        // Validate consistency.
        if config.tensor_parallel_size > 1 && config.tensor_parallel_size != world_size {
            anyhow::bail!(
                "--tensor-parallel-size ({}) does not match WORLD_SIZE ({})",
                config.tensor_parallel_size,
                world_size
            );
        }

        Ok(Self {
            rank,
            local_rank,
            world_size,
            master_addr,
            master_port,
        })
    }
}

/// External launcher init path: one process per GPU, NCCL via TCP store.
///
/// Used when `--distributed-executor-backend external_launcher`. The job launcher
/// (torchrun, mpirun, SLURM) spawns N processes, each calling this function.
/// Each process:
/// 1. Reads RANK, LOCAL_RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT from env.
/// 2. Sets CUDA device to LOCAL_RANK.
/// 3. Exchanges NCCL unique ID via TCP store (rank 0 serves, others connect).
/// 4. Creates a single CudaWorker with TP sharding for this rank.
/// 5. Coordinates memory allocation via TCP all-reduce MIN.
/// 6. Wraps in UniProcExecutor (one worker per process).
/// 7. Builds AsyncEngine and returns the stack.
///
/// Each process runs its own HTTP server on a different port (set via --port).
fn initialize_stack_external(
    config: &VllmConfig,
    model_name: String,
    init_start: Instant,
) -> Result<InitializedStack> {
    #[cfg(not(feature = "nccl"))]
    {
        let _ = (config, model_name, init_start);
        anyhow::bail!(
            "External launcher requires the `nccl` feature; \
             rebuild with --features nccl"
        );
    }

    #[cfg(feature = "nccl")]
    {
        use scratchy_serving_worker::gpu_worker::CudaWorker;
        use scratchy_serving_worker::worker_factory::WorkerCreateConfig;

        let env = ExternalLauncherEnv::from_env(config)
            .context("failed to read external launcher env vars")?;
        let rank = env.rank;
        let local_rank = env.local_rank;
        let world_size = env.world_size;
        let master_addr = env.master_addr;
        let master_port = env.master_port;

        let is_pooling = config.runner == "pooling";

        info!(
            "External launcher: rank={}, local_rank={}, world_size={}, master={}:{}",
            rank, local_rank, world_size, master_addr, master_port
        );

        // Step 1: Exchange NCCL unique ID via TCP store.
        let nccl_id_bytes = scratchy_target_cuda::tcp_store::exchange_nccl_id(
            rank,
            world_size,
            &master_addr,
            master_port,
        )
        .context("failed to exchange NCCL ID via TCP store")?;
        let nccl_id = scratchy_target_cuda::NcclId::from_raw(nccl_id_bytes);
        info!("Rank {}: NCCL ID exchanged", rank);

        // Step 2: Create CudaWorker for this rank's GPU.
        let cuda_config = WorkerCreateConfig {
            model_path: config.model.clone(),
            dtype: config.dtype.clone(),
            hf_token: config.hf_token.clone(),
            block_size: config.block_size,
            device_id: local_rank as i32,
            max_num_seqs: resolve_max_num_seqs(config.max_num_seqs, None, false),
            enforce_eager: config.enforce_eager,
            max_num_batched_tokens: config.max_num_batched_tokens.unwrap_or(2048),
            cuda_graph_sizes: config
                .cuda_graph_config
                .as_ref()
                .map(|c| c.capture_sizes.clone())
                .unwrap_or_default(),
            cublas_autotune: config.cublas_autotune,
            gpu_memory_utilization: config.gpu_memory_utilization,
            pooling_strategy: config.pooling_strategy.clone(),
            is_pooling,
            tp_rank: rank,
            tp_world_size: world_size,
            pp_rank: 0,
            pp_size: 1,
            gguf_file: config.gguf_file.clone(),
            lora_adapter: config.lora_adapter.clone(),
            kv_cache_dtype: config.kv_cache_dtype.clone(),
            calculate_kv_scales: config.calculate_kv_scales,
            cuda_graph_mode: config
                .cuda_graph_mode
                .parse()
                .unwrap_or(CudaGraphMode::Auto),
            eos_token_ids: vec![],
            max_model_len: config.max_model_len,
            draft_model_path: spec_decode_draft_model_path(config),
            draft_model_dtype: config.draft_model_dtype.clone(),
        };

        let mut worker = CudaWorker::new(cuda_config);
        worker
            .init_device()
            .context("failed to initialize CUDA device")?;
        worker.load_model().context("failed to load model")?;

        let hf_config = worker
            .hf_config()
            .context("model config not available after load")?
            .clone();
        let model_dir = worker.model_dir().map(|p| p.to_path_buf());
        let dtype_elem_bytes = worker.resolved_dtype_elem_bytes();

        // Step 3: Create NCCL communicator.
        let device = worker.device_ref().expect("device not initialized");
        unsafe {
            scratchy_target_cuda::driver::ctx_set_current(device.ctx).unwrap();
        }
        let nccl_group =
            scratchy_target_cuda::NcclGroup::new(rank, world_size, nccl_id, device.compute_stream)
                .context("NCCL comm init failed")?;
        worker.set_tp_group(std::sync::Arc::new(nccl_group));
        info!("Rank {}: NCCL communicator created", rank);

        // Step 4: Profile available memory.
        let available_memory = worker
            .determine_available_memory()
            .context("failed to determine available memory")?;

        // Step 5: All-reduce MIN across ranks via TCP store.
        let min_memory = scratchy_target_cuda::tcp_store::allreduce_min(
            rank,
            world_size,
            available_memory,
            &master_addr,
            master_port,
        )
        .context("failed to allreduce memory")?;

        let max_model_len = config
            .max_model_len
            .or(hf_config.max_position_embeddings())
            .unwrap_or(4096);

        info!(
            "Rank {}: available_memory={:.1} GB, min_across_ranks={:.1} GB",
            rank,
            available_memory as f64 / (1024.0 * 1024.0 * 1024.0),
            min_memory as f64 / (1024.0 * 1024.0 * 1024.0),
        );

        // Step 6: Compute block count and initialize cache.
        let (num_gpu_blocks, swa_hybrid_kv) = compute_kv_blocks(
            min_memory,
            config.block_size,
            &hf_config,
            dtype_elem_bytes,
            config.gpu_memory_utilization,
            &config.kv_cache_dtype,
            max_model_len,
            worker.supports_hybrid_swa_kv(),
        );

        let mut worker: Box<dyn Worker> = Box::new(worker);
        worker
            .initialize_cache(num_gpu_blocks, 0)
            .context("failed to initialize cache")?;

        info!(
            "Rank {}: num_gpu_blocks={}, kv_cache_tokens={}",
            rank,
            num_gpu_blocks,
            num_gpu_blocks * config.block_size,
        );

        // Step 7: Warm up / CUDA graph capture.
        worker
            .compile_or_warm_up_model()
            .context("failed to compile or warm up model")?;

        // Step 8: Wrap in UniProcExecutor (single worker per process).
        let executor = UniProcExecutor::new_pre_initialized(worker);

        // Step 9: Build engine.
        let eos_token_ids: Vec<u32> = hf_config
            .extra
            .get("eos_token_id")
            .map(|v| {
                if let Some(id) = v.as_u64() {
                    vec![id as u32]
                } else if let Some(arr) = v.as_array() {
                    arr.iter()
                        .filter_map(|v| v.as_u64().map(|id| id as u32))
                        .collect()
                } else {
                    vec![]
                }
            })
            .unwrap_or_default();

        let use_async_scheduling =
            !config.disable_async_scheduling && !spec_decode_requires_sync(config);
        let enable_prefix_caching = config.enable_prefix_caching;
        let engine_config = EngineCoreConfig {
            scheduler_config: SchedulerConfig {
                max_num_batched_tokens: config.max_num_batched_tokens.unwrap_or(2048),
                max_num_seqs: resolve_max_num_seqs(config.max_num_seqs, None, false),
                policy: SchedulerPolicy::Fcfs,
                enable_chunked_prefill: true,
                async_scheduling: Some(use_async_scheduling),
                num_lookahead_tokens: if config.speculative_model.is_some() {
                    config.num_speculative_tokens
                } else {
                    0
                },
                ..Default::default()
            },
            max_model_len,
            num_gpu_blocks,
            block_size: config.block_size,
            engine_index: 0,
            async_scheduling: use_async_scheduling,
            use_spec_decode: config.speculative_model.is_some(),
            proposer_config: None,
            eos_token_ids,
            is_pooling: config.runner == "pooling",
            enable_prefix_caching,
            hybrid_kv: swa_hybrid_kv,
        };

        let client: Box<dyn scratchy_serving_engine::core_client::EngineCoreClient + Send> =
            Box::new(InprocClient::new(engine_config, Box::new(executor)));

        // Load tokenizer and build engine.
        let tokenizer = model_dir
            .as_ref()
            .and_then(|dir| try_load_tokenizer(dir).ok());

        let mut engine = if let Some(tok) = tokenizer {
            let tokenizer = Arc::new(tok);
            if let Some(ref dir) = model_dir {
                if let Some(ct) = resolve_chat_template(config.chat_template.as_deref(), dir) {
                    info!("Chat template loaded from tokenizer_config.json");
                    AsyncEngine::with_tokenizer_and_template(
                        client,
                        model_name.clone(),
                        max_model_len,
                        tokenizer,
                        Arc::new(ct),
                    )
                } else {
                    AsyncEngine::with_tokenizer(
                        client,
                        model_name.clone(),
                        max_model_len,
                        tokenizer,
                    )
                }
            } else {
                AsyncEngine::with_tokenizer(client, model_name.clone(), max_model_len, tokenizer)
            }
        } else {
            AsyncEngine::new(client, model_name.clone(), max_model_len)
        };

        if !config.disable_async_scheduling && !spec_decode_requires_sync(config) {
            engine.set_generation_defaults(load_generation_defaults(model_dir.as_deref()));
            engine.set_async_scheduling(true);
        }
        if config.runner == "pooling" {
            engine.set_is_pooling(true);
        }

        info!(
            "Rank {}: stack initialized in {:.1}s (external launcher, world_size={})",
            rank,
            init_start.elapsed().as_secs_f64(),
            world_size,
        );

        auto_install_tool_parser(&mut engine, model_arch(&hf_config));
        auto_install_reasoning_parser(&mut engine, model_arch(&hf_config));
        Ok(InitializedStack {
            engine: Arc::new(engine),
            model_name,
            max_model_len,
        })
    }
}

/// Try to load a HuggingFace tokenizer from a model directory.
pub fn try_load_tokenizer(model_dir: &Path) -> Result<Tokenizer> {
    let tokenizer_path = model_dir.join("tokenizer.json");
    if !tokenizer_path.exists() {
        anyhow::bail!("tokenizer.json not found in {}", model_dir.display());
    }
    Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {e}"))
}

/// Mirror `transformers`' `AutoTokenizer` behavior: after the
/// `tokenizer.json::added_tokens` list is loaded, scan
/// `tokenizer_config.json::additional_special_tokens` (a string list)
/// and register any entries missing from the former. Required for
/// arches like Qwen2-VL / Qwen2.5-VL that declare
/// `<|image_pad|>` / `<|video_pad|>` / `<|vision_pad|>` only in
/// `tokenizer_config.json` — without this, the chat template's
/// `<|image_pad|>` emission subword-splits at encode time and
/// `expand_image_placeholders` finds zero occurrences. Idempotent;
/// no-op when `tokenizer_config.json` is absent or has no
/// `additional_special_tokens` field. Applied to every tokenizer
/// path (preloaded via the worker's bare HfTokenizer load AND
/// `try_load_tokenizer`'s fallback path) so configurations converge.
fn patch_additional_special_tokens(tok: &mut Tokenizer, model_dir: &Path) {
    let config_path = model_dir.join("tokenizer_config.json");
    let Ok(bytes) = std::fs::read(&config_path) else {
        return;
    };
    let Ok(cfg) = serde_json::from_slice::<serde_json::Value>(&bytes) else {
        return;
    };
    let Some(arr) = cfg
        .get("additional_special_tokens")
        .and_then(|v| v.as_array())
    else {
        return;
    };
    let contents: Vec<String> = arr
        .iter()
        .filter_map(|v| v.as_str().map(|s| s.to_string()))
        .collect();
    if !contents.is_empty() {
        tok.register_additional_special_tokens(&contents);
    }
}

/// Resolve a chat template, honoring an explicit operator override
/// (`config.chat_template`) before falling back to auto-detection.
///
/// `override_str` is interpreted as a filesystem path if the value
/// names a readable file; otherwise treated as inline Jinja. This
/// covers `--chat-template /path/to/tokenizer_config.json` as well
/// as raw template strings.
pub(crate) fn resolve_chat_template(
    override_str: Option<&str>,
    model_dir: &Path,
) -> Option<ChatTemplate> {
    if let Some(s) = override_str {
        let trimmed = s.trim();
        let from_path = std::path::Path::new(trimmed);
        let raw: Option<String> = if from_path.is_file() {
            // Either a .jinja template or a tokenizer_config.json. If
            // the file parses as JSON with a `chat_template` field,
            // use that; otherwise treat the file body as raw Jinja.
            std::fs::read_to_string(from_path).ok().map(|body| {
                match serde_json::from_str::<serde_json::Value>(&body) {
                    Ok(v) => v
                        .get("chat_template")
                        .and_then(|x| x.as_str().map(String::from))
                        .unwrap_or(body),
                    Err(_) => body,
                }
            })
        } else {
            Some(trimmed.to_string())
        };
        if let Some(template_str) = raw {
            match ChatTemplate::new(template_str) {
                Ok(mut tpl) => {
                    // GGUFs almost always lack bos/eos at the
                    // template-string level — the original
                    // tokenizer_config.json renders them via
                    // `{{- bos_token }}` / `{{ eos_token }}`. When the
                    // override is paired with a GGUF, pull bos/eos
                    // from the file's metadata (same path the
                    // auto-detect branch uses) so an operator-supplied
                    // template still gets the right special tokens.
                    if let Some(gguf_path) = gguf_in_or_under(model_dir)
                        && let Ok(gguf) = scratchy_quantizations::gguf::GgufFile::open(&gguf_path)
                    {
                        tpl = apply_gguf_special_tokens(&gguf, tpl);
                    }
                    info!("Chat template loaded from --chat-template override");
                    return Some(tpl);
                }
                Err(e) => {
                    info!("Failed to parse --chat-template override: {e}");
                }
            }
        }
    }
    try_load_chat_template(model_dir)
}

/// Helper used by both `resolve_chat_template` and `try_load_chat_template`:
/// locate the .gguf file at, or directly inside, `model_dir`.
fn gguf_in_or_under(model_dir: &Path) -> Option<std::path::PathBuf> {
    if model_dir.is_file() && model_dir.extension().is_some_and(|e| e == "gguf") {
        Some(model_dir.to_path_buf())
    } else if model_dir.is_dir() {
        std::fs::read_dir(model_dir).ok().and_then(|mut it| {
            it.find_map(|entry| {
                let p = entry.ok()?.path();
                (p.extension().is_some_and(|e| e == "gguf")).then_some(p)
            })
        })
    } else {
        None
    }
}

/// Pull bos/eos token strings out of GGUF metadata and stamp them
/// onto a `ChatTemplate`. Does nothing if either id or the tokens
/// array is missing.
fn apply_gguf_special_tokens(
    gguf: &scratchy_quantizations::gguf::GgufFile,
    mut tpl: ChatTemplate,
) -> ChatTemplate {
    if let Some(tokens_val) = gguf.metadata().get("tokenizer.ggml.tokens")
        && let Ok(tokens) = tokens_val.to_vec()
    {
        let resolve =
            |id: u32| -> Option<String> { tokens.get(id as usize)?.to_string().ok().cloned() };
        if let Some(bos_id) = gguf.get_metadata_u32("tokenizer.ggml.bos_token_id")
            && let Some(s) = resolve(bos_id)
        {
            tpl = tpl.with_bos_token(s);
        }
        if let Some(eos_id) = gguf.get_metadata_u32("tokenizer.ggml.eos_token_id")
            && let Some(s) = resolve(eos_id)
        {
            tpl = tpl.with_eos_token(s);
        }
    }
    tpl
}

/// Try to load a chat template. Sources tried, in order:
/// 1. `model_dir/tokenizer_config.json` (HF convention).
/// 2. `model_dir/chat_template.jinja` (some quantized repos).
/// 3. `tokenizer.chat_template` metadata in a `.gguf` file —
///    handles both `model_dir` being a `.gguf` file directly and
///    a directory containing one.
fn try_load_chat_template(model_dir: &Path) -> Option<ChatTemplate> {
    // GGUF: read from metadata. Try direct file, then look for any
    // .gguf file in the directory.
    if let Some(gguf_path) = gguf_in_or_under(model_dir)
        && let Ok(gguf) = scratchy_quantizations::gguf::GgufFile::open(&gguf_path)
        && let Some(template_str) = scratchy_quantizations::gguf::gguf_chat_template(&gguf)
    {
        match ChatTemplate::new(template_str) {
            Ok(tpl) => {
                let tpl = apply_gguf_special_tokens(&gguf, tpl);
                info!("Chat template loaded from GGUF metadata");
                return Some(tpl);
            }
            Err(e) => {
                info!("Failed to parse GGUF chat_template: {e}");
            }
        }
    }

    // Bare .gguf file path with no template in metadata: fall through to
    // checking the parent directory for tokenizer_config.json /
    // chat_template.jinja (some HF GGUF repos ship those alongside).
    let probe_dir: std::borrow::Cow<'_, Path> = if model_dir.is_file() {
        std::borrow::Cow::Owned(model_dir.parent().unwrap_or(model_dir).to_path_buf())
    } else {
        std::borrow::Cow::Borrowed(model_dir)
    };
    let model_dir = probe_dir.as_ref();

    let config_path = model_dir.join("tokenizer_config.json");
    match ChatTemplate::from_tokenizer_config(&config_path) {
        Ok(Some(tpl)) => Some(tpl),
        Ok(None) => {
            // Fallback: some models (e.g. AWQ quantized) store the template in a
            // separate Jinja file instead of embedding it in tokenizer_config.json.
            let jinja_path = model_dir.join("chat_template.jinja");
            if jinja_path.exists() {
                match std::fs::read_to_string(&jinja_path) {
                    Ok(template_str) => match ChatTemplate::new(template_str) {
                        Ok(tpl) => {
                            info!("Chat template loaded from chat_template.jinja");
                            Some(tpl)
                        }
                        Err(e) => {
                            info!("Failed to parse chat_template.jinja: {e}");
                            None
                        }
                    },
                    Err(e) => {
                        info!("Failed to read chat_template.jinja: {e}");
                        None
                    }
                }
            } else {
                None
            }
        }
        Err(e) => {
            info!("Failed to parse chat template: {e}");
            None
        }
    }
}

/// Compute the number of KV cache blocks from available memory.
///
/// `available_bytes` is the KV cache budget in bytes, already accounting for
/// `gpu_memory_utilization`. This matches Python vLLM where
/// `determine_available_memory` returns `total * util - non_kv_cache` and
/// the block count is simply `available / bytes_per_block`.
fn compute_num_blocks(
    available_bytes: usize,
    block_size: usize,
    hf_config: &HfModelConfig,
    dtype_elem_bytes: usize,
    _gpu_memory_utilization: f64,
    kv_cache_dtype: &str,
) -> usize {
    let num_layers = resolve_num_layers(hf_config);
    let num_kv_heads = hf_config.num_kv_heads().unwrap_or(0);
    let head_dim = hf_config.head_dim().unwrap_or(0);

    // Each block holds block_size tokens of KV for all layers.
    // KV per token per layer = 2 * num_kv_heads * head_dim * sizeof(dtype)
    // FP8 KV cache stores 1 byte/element instead of 2 (BF16), doubling capacity.
    let elem_bytes = if kv_cache_dtype == "fp8_e4m3" || kv_cache_dtype == "fp8" {
        1
    } else {
        dtype_elem_bytes
    };
    let bytes_per_token_per_layer = 2 * num_kv_heads * head_dim * elem_bytes;
    let bytes_per_block = block_size * num_layers * bytes_per_token_per_layer;

    if bytes_per_block == 0 {
        return 1024; // Fallback.
    }

    let num_blocks = available_bytes / bytes_per_block;

    // NOTE: bumping num_blocks for SWA here was REVERTED — the uniform
    // `KvCachePool` sizes EVERY layer's buffer to num_blocks, and that VA is
    // residency-attached to the command queue (so it counts against the GPU
    // working set, not just committed pages). Doubling it OOMs the command
    // buffer. Fitting 32k needs the sliding LAYERS' buffers to be window-sized
    // (per-layer block counts), not a larger uniform pool.

    // At least 16 blocks.
    num_blocks.max(16)
}

/// Compute the KV-cache block count AND the hybrid SWA layout, if any.
///
/// For uniform (non-SWA) models this is exactly [`compute_num_blocks`] with no
/// hybrid config. For `is_swa_hybrid` models (gemma4) it computes vLLM's
/// group-shared layout: the shared-pool `num_blocks`
/// (`available / page_size / group_size`, the SAME total VA as the uniform pool
/// but holding `group_size`× more blocks since each physical tensor is shared
/// across groups) and the per-group `(is_sliding, window, block_size)` the
/// scheduler's hybrid allocator consumes. Returns `(num_blocks, hybrid_kv)`.
#[allow(clippy::too_many_arguments)]
fn compute_kv_blocks(
    available_bytes: usize,
    block_size: usize,
    hf_config: &HfModelConfig,
    dtype_elem_bytes: usize,
    utilization: f64,
    kv_cache_dtype: &str,
    max_model_len: usize,
    allow_hybrid: bool,
) -> (usize, Option<HybridKvConfig>) {
    let uniform = compute_num_blocks(
        available_bytes,
        block_size,
        hf_config,
        dtype_elem_bytes,
        utilization,
        kv_cache_dtype,
    );
    // The CUDA decode reads a single block table for every layer (no grouped
    // sliding-window KV); only metal wires the `sliding_groups` decode. A CUDA
    // worker passes `allow_hybrid=false` so gemma4 (and any hybrid-SWA arch)
    // stays on the uniform single pool — the worker's single-block-table decode
    // then reads each sliding layer's KV from the correct (single) block space.
    if !allow_hybrid {
        return (uniform, None);
    }
    let Some(geom_tuples) = hf_config.hybrid_layer_geometry() else {
        return (uniform, None);
    };
    let elem_bytes = if kv_cache_dtype == "fp8_e4m3" || kv_cache_dtype == "fp8" {
        1
    } else {
        dtype_elem_bytes
    };
    let geom: Vec<scratchy_core_config::LayerKvGeometry> = geom_tuples
        .into_iter()
        .map(|(is_sliding, num_kv_heads, head_size, sliding_window)| {
            scratchy_core_config::LayerKvGeometry {
                is_sliding,
                num_kv_heads,
                head_size,
                head_size_v: None,
                sliding_window,
            }
        })
        .collect();
    // Only the page-DIFFERENTIATED case (gemma4: sliding 8×256 vs full 2×512)
    // takes the group-shared layout — that matches the worker's trigger
    // (`per_layer_kv_token_elems`, emitted only when the classes' dims differ).
    // Same-dims SWA (gemma2/3) stays on the uniform pool in BOTH the scheduler
    // and the worker, so they never disagree on group count.
    let distinct_pages: std::collections::HashSet<usize> =
        geom.iter().map(|g| g.num_kv_heads * g.head_size).collect();
    if distinct_pages.len() <= 1 {
        return (uniform, None);
    }
    match scratchy_core_config::compute_hybrid_kv_layout(
        &geom,
        block_size,
        available_bytes,
        elem_bytes,
    ) {
        Some(layout) => {
            // Size the shared pool to what `max_model_len` actually needs, not
            // the whole KV budget. Unlike a uniform pool (which pre-commits all
            // num_blocks), the hybrid pool is lazy-committed AND the sliding
            // groups free out-of-window blocks, so the live set is ~the full
            // group's full-context blocks plus a window's worth per sliding
            // group. Reserving the full budget of residency-attached VA (gemma4:
            // 256k → ~5 GiB) just starves the activation arena → OOM, while the
            // actual KV stays well under 1 GiB. mlx-lm sizes its caches to the
            // sequence the same way (RotatingKVCache grows to min(ctx, window)).
            //
            // Upper bound on the live block-ID set: the full group's
            // full-context blocks (`mml/full_bs`) plus, generously, one
            // full-context sliding group's worth (`mml/sliding_bs`) — that
            // covers all N sliding groups since each is window-bounded well
            // below a full context. Still clamped to the budget-derived count.
            let full_bs = layout.full_block_size().max(1);
            // Each SLIDING group draws from this one shared pool (gemma4-12b: 5,
            // 26b: 5). The scheduler's SWA eviction (remove_skipped_blocks) frees
            // out-of-window blocks every step and the prefill chunk is capped at
            // SWA_PREFILL_CHUNK_CAP, so a sliding group's live block set is bounded
            // by (window + chunk_cap), NOT the full context. Sizing to mml here
            // over-reserves ~10x of residency-attached VA and OOMs big models
            // (gemma-4-26b weights ~24 GB on a 32 GB Mac); measured max_block tops
            // out at ~(window+cap)/bs * num_sliding + global, matching this.
            let num_sliding = layout.num_groups().saturating_sub(1).max(1);
            let sliding_window = geom
                .iter()
                .filter(|g| g.is_sliding)
                .filter_map(|g| g.sliding_window)
                .max()
                .unwrap_or(0);
            // Live sliding set = window + the in-flight prefill chunk, plus a
            // chunk of eviction lag: the scheduler frees WHOLE out-of-window
            // blocks at the START of a step, so the prior chunk's blocks can
            // still be resident when the next chunk's are added. Budget
            // 2*chunk_cap of headroom over the window. Measured sliding usage at
            // 24k (~1.1-1.2k blocks across 5 groups) sits inside this and far
            // under the full-context ~7.7k that sizing to mml would reserve.
            let sliding_blocks_per_group =
                ((sliding_window + 2 * SWA_PREFILL_CHUNK_CAP).div_ceil(block_size) + 2)
                    // A sequence of at most `max_model_len` tokens can never need more
                    // than its own block count, regardless of the window+lag headroom.
                    .min(max_model_len.div_ceil(block_size));
            let need = max_model_len.div_ceil(full_bs) + num_sliding * sliding_blocks_per_group;
            let num_blocks = layout.config.num_blocks.min(need).max(16);
            tracing::info!(
                "Hybrid SWA KV layout: {} shared blocks ({} groups, group_size {}, page {} B; \
                 sized for max_model_len {} → {} blocks, budget cap {})",
                num_blocks,
                layout.num_groups(),
                layout.group_size,
                layout.page_size_bytes,
                max_model_len,
                need,
                layout.config.num_blocks,
            );
            (num_blocks, Some((num_blocks, layout.engine_groups())))
        }
        None => (uniform, None),
    }
}

/// Extract a human-friendly model name from a path or HF ID.
fn extract_model_name(model_path: &str) -> std::borrow::Cow<'_, str> {
    // For HF IDs like "meta-llama/Llama-3.2-1B", return the full ID.
    // For local paths, return the last directory component.
    if model_path.contains('/') && !model_path.starts_with('/') {
        // Looks like an HF model ID — borrow directly, no allocation needed.
        std::borrow::Cow::Borrowed(model_path)
    } else {
        std::borrow::Cow::Owned(
            std::path::Path::new(model_path)
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| model_path.to_string()),
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_model_name_hf_id() {
        assert_eq!(
            extract_model_name("meta-llama/Llama-3.2-1B"),
            "meta-llama/Llama-3.2-1B"
        );
    }

    #[test]
    fn test_extract_model_name_local_path() {
        assert_eq!(extract_model_name("/home/user/models/my-model"), "my-model");
    }

    #[test]
    fn test_compute_num_blocks_reasonable() {
        let config = HfModelConfig {
            num_hidden_layers: Some(32),
            num_attention_heads: Some(32),
            num_key_value_heads: Some(8),
            hidden_size: Some(4096),
            ..Default::default()
        };
        let blocks = compute_num_blocks(4 * 1024 * 1024 * 1024, 16, &config, 4, 0.9, "auto");
        assert!(blocks >= 16);
    }

    #[test]
    fn test_compute_num_blocks_f16_more_blocks() {
        // F16 uses half the bytes per element → should yield ~2x as many blocks.
        let config = HfModelConfig {
            num_hidden_layers: Some(32),
            num_attention_heads: Some(32),
            num_key_value_heads: Some(8),
            hidden_size: Some(4096),
            ..Default::default()
        };
        let blocks_f32 = compute_num_blocks(4 * 1024 * 1024 * 1024, 16, &config, 4, 0.9, "auto");
        let blocks_f16 = compute_num_blocks(4 * 1024 * 1024 * 1024, 16, &config, 2, 0.9, "auto");
        assert!(blocks_f16 > blocks_f32);
        // F16 should give approximately 2x the blocks.
        assert!((blocks_f16 as f64 / blocks_f32 as f64 - 2.0).abs() < 0.1);
    }

    #[test]
    fn test_compute_num_blocks_zero_dim() {
        let config = HfModelConfig::default();
        let blocks = compute_num_blocks(1024, 16, &config, 4, 0.9, "auto");
        // Should fall back to 1024.
        assert_eq!(blocks, 1024);
    }

    #[test]
    fn test_compute_num_blocks_proportional_to_memory() {
        // compute_num_blocks no longer applies utilization internally —
        // the caller is responsible for providing the final KV cache budget.
        // Verify that halving the input memory halves the blocks.
        let config = HfModelConfig {
            num_hidden_layers: Some(32),
            num_attention_heads: Some(32),
            num_key_value_heads: Some(8),
            hidden_size: Some(4096),
            ..Default::default()
        };
        let blocks_full = compute_num_blocks(4 * 1024 * 1024 * 1024, 16, &config, 2, 0.9, "auto");
        let blocks_half = compute_num_blocks(2 * 1024 * 1024 * 1024, 16, &config, 2, 0.9, "auto");
        let ratio = blocks_half as f64 / blocks_full as f64;
        assert!((ratio - 0.5).abs() < 0.01, "ratio was {ratio}");
    }

    #[test]
    fn test_scratchy_core_config_default() {
        let config = VllmConfig::default();
        assert_eq!(config.device, "auto");
        assert_eq!(config.dtype, "auto");
        assert_eq!(config.max_num_seqs, None);
        assert_eq!(config.block_size, 16);
        assert!((config.gpu_memory_utilization - 0.9).abs() < f64::EPSILON);
        assert!(config.model.is_empty());
        assert!(config.max_num_batched_tokens.is_none());
    }

    #[test]
    fn test_max_num_batched_tokens_default_none() {
        let config = VllmConfig::default();
        assert!(
            config.max_num_batched_tokens.is_none(),
            "default should be None (auto)"
        );
    }

    #[test]
    fn test_max_num_batched_tokens_explicit_override() {
        let config = VllmConfig {
            max_num_batched_tokens: Some(4096),
            ..VllmConfig::default()
        };
        assert_eq!(config.max_num_batched_tokens, Some(4096));
    }

    #[test]
    fn test_scratchy_core_config_default_distributed_backend() {
        let config = VllmConfig::default();
        assert_eq!(config.distributed_executor_backend, "auto");
    }

    #[test]
    fn test_scratchy_core_config_external_launcher() {
        let config = VllmConfig {
            distributed_executor_backend: "external_launcher".to_string(),
            ..VllmConfig::default()
        };
        assert_eq!(config.distributed_executor_backend, "external_launcher");
    }

    // -- ExternalLauncherEnv tests --
    // Note: these tests manipulate env vars, which is inherently global state.
    // We use a mutex to serialize them and restore original values.

    use std::sync::Mutex;
    static ENV_MUTEX: Mutex<()> = Mutex::new(());

    /// Helper: set env vars, run closure, restore originals.
    fn with_env_vars<F, R>(vars: &[(&str, Option<&str>)], f: F) -> R
    where
        F: FnOnce() -> R,
    {
        let _lock = ENV_MUTEX.lock().unwrap();
        let mut originals = Vec::new();
        for &(key, val) in vars {
            originals.push((key, std::env::var(key).ok()));
            // SAFETY: we hold ENV_MUTEX, serializing all env var access in tests.
            unsafe {
                match val {
                    Some(v) => std::env::set_var(key, v),
                    None => std::env::remove_var(key),
                }
            }
        }
        let result = f();
        for (key, original) in originals {
            // SAFETY: same mutex guard still held.
            unsafe {
                match original {
                    Some(v) => std::env::set_var(key, v),
                    None => std::env::remove_var(key),
                }
            }
        }
        result
    }

    #[test]
    fn test_external_launcher_env_all_vars_set() {
        with_env_vars(
            &[
                ("RANK", Some("1")),
                ("LOCAL_RANK", Some("1")),
                ("WORLD_SIZE", Some("4")),
                ("MASTER_ADDR", Some("10.0.0.1")),
                ("MASTER_PORT", Some("29600")),
            ],
            || {
                let config = VllmConfig::default();
                let env = ExternalLauncherEnv::from_env(&config).unwrap();
                assert_eq!(env.rank, 1);
                assert_eq!(env.local_rank, 1);
                assert_eq!(env.world_size, 4);
                assert_eq!(env.master_addr, "10.0.0.1");
                assert_eq!(env.master_port, 29600);
            },
        );
    }

    #[test]
    fn test_external_launcher_env_fallback_master_addr() {
        with_env_vars(
            &[
                ("RANK", Some("0")),
                ("LOCAL_RANK", Some("0")),
                ("WORLD_SIZE", Some("2")),
                ("MASTER_ADDR", None),
                ("MASTER_PORT", None),
            ],
            || {
                let config = VllmConfig {
                    master_addr: "my-host".to_string(),
                    master_port: 12345,
                    ..VllmConfig::default()
                };
                let env = ExternalLauncherEnv::from_env(&config).unwrap();
                assert_eq!(env.master_addr, "my-host");
                assert_eq!(env.master_port, 12345);
            },
        );
    }

    #[test]
    fn test_external_launcher_env_missing_rank() {
        with_env_vars(
            &[
                ("RANK", None),
                ("LOCAL_RANK", Some("0")),
                ("WORLD_SIZE", Some("2")),
            ],
            || {
                let config = VllmConfig::default();
                let err = ExternalLauncherEnv::from_env(&config).unwrap_err();
                assert!(
                    err.to_string().contains("RANK"),
                    "error should mention RANK: {}",
                    err
                );
            },
        );
    }

    #[test]
    fn test_external_launcher_env_missing_local_rank() {
        with_env_vars(
            &[
                ("RANK", Some("0")),
                ("LOCAL_RANK", None),
                ("WORLD_SIZE", Some("2")),
            ],
            || {
                let config = VllmConfig::default();
                let err = ExternalLauncherEnv::from_env(&config).unwrap_err();
                assert!(
                    err.to_string().contains("LOCAL_RANK"),
                    "error should mention LOCAL_RANK: {}",
                    err
                );
            },
        );
    }

    #[test]
    fn test_external_launcher_env_missing_world_size() {
        with_env_vars(
            &[
                ("RANK", Some("0")),
                ("LOCAL_RANK", Some("0")),
                ("WORLD_SIZE", None),
            ],
            || {
                let config = VllmConfig::default();
                let err = ExternalLauncherEnv::from_env(&config).unwrap_err();
                assert!(
                    err.to_string().contains("WORLD_SIZE"),
                    "error should mention WORLD_SIZE: {}",
                    err
                );
            },
        );
    }

    #[test]
    fn test_external_launcher_env_invalid_rank() {
        with_env_vars(
            &[
                ("RANK", Some("not_a_number")),
                ("LOCAL_RANK", Some("0")),
                ("WORLD_SIZE", Some("2")),
            ],
            || {
                let config = VllmConfig::default();
                let err = ExternalLauncherEnv::from_env(&config).unwrap_err();
                assert!(
                    err.to_string().contains("integer"),
                    "error should mention integer: {}",
                    err
                );
            },
        );
    }

    #[test]
    fn test_external_launcher_env_tp_mismatch() {
        with_env_vars(
            &[
                ("RANK", Some("0")),
                ("LOCAL_RANK", Some("0")),
                ("WORLD_SIZE", Some("4")),
                ("MASTER_ADDR", Some("127.0.0.1")),
                ("MASTER_PORT", Some("29500")),
            ],
            || {
                let config = VllmConfig {
                    tensor_parallel_size: 2, // mismatch with WORLD_SIZE=4
                    ..VllmConfig::default()
                };
                let err = ExternalLauncherEnv::from_env(&config).unwrap_err();
                assert!(
                    err.to_string().contains("does not match"),
                    "error should mention mismatch: {}",
                    err
                );
            },
        );
    }

    #[test]
    fn test_external_launcher_env_tp_size_one_no_mismatch() {
        // tp_size=1 (default) should NOT conflict with any WORLD_SIZE.
        with_env_vars(
            &[
                ("RANK", Some("0")),
                ("LOCAL_RANK", Some("0")),
                ("WORLD_SIZE", Some("8")),
                ("MASTER_ADDR", Some("127.0.0.1")),
                ("MASTER_PORT", Some("29500")),
            ],
            || {
                let config = VllmConfig::default(); // tp_size=1
                let env = ExternalLauncherEnv::from_env(&config).unwrap();
                assert_eq!(env.world_size, 8);
            },
        );
    }

    #[test]
    fn test_external_launcher_env_tp_matches_world_size() {
        with_env_vars(
            &[
                ("RANK", Some("0")),
                ("LOCAL_RANK", Some("0")),
                ("WORLD_SIZE", Some("4")),
                ("MASTER_ADDR", Some("127.0.0.1")),
                ("MASTER_PORT", Some("29500")),
            ],
            || {
                let config = VllmConfig {
                    tensor_parallel_size: 4, // matches WORLD_SIZE=4
                    ..VllmConfig::default()
                };
                let env = ExternalLauncherEnv::from_env(&config).unwrap();
                assert_eq!(env.world_size, 4);
            },
        );
    }

    #[test]
    fn test_external_launcher_env_rank0() {
        with_env_vars(
            &[
                ("RANK", Some("0")),
                ("LOCAL_RANK", Some("0")),
                ("WORLD_SIZE", Some("1")),
            ],
            || {
                let config = VllmConfig::default();
                let env = ExternalLauncherEnv::from_env(&config).unwrap();
                assert_eq!(env.rank, 0);
                assert_eq!(env.local_rank, 0);
                assert_eq!(env.world_size, 1);
            },
        );
    }
}
