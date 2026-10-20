// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Stack initialization: wires together worker → executor → engine → server.
//!
//! This module provides the programmatic API for initializing the full vLLM
//! inference stack. No CLI dependency needed — just construct a [`VllmConfig`]
//! and call [`initialize_stack`].

// Anchor the worker crate into the link so its `WorkerFactory`
// `inventory::submit!` registrations (MetalWorkerFactory / SpyreWorkerFactory
// in scratchy_serving_worker) are NOT garbage-collected by the
// linker. `create_worker` never NAMES a concrete worker, so without this explicit reference the registry
// could come up empty at runtime. `create_worker` still bails loudly if it
// does (empty-registry assertion) — this anchor is the belt to that bail's
// suspenders.
extern crate scratchy_serving_worker as _;

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use scratchy_core_config::{SchedulerConfig, SchedulerPolicy};
use scratchy_core_model::weight::HfModelConfig;
use scratchy_serving_engine::core_client::InprocClient;
use scratchy_serving_engine::engine_core::{EngineCoreConfig, HybridKvConfig};
use scratchy_serving_engine::gpu_budget::{MaxNumSeqsFacts, resolve_default_max_num_seqs};
use scratchy_serving_engine::spec_decode::{
    DraftBudget, DraftModelProposerConfig, MtpProposerConfig, NgramProposerConfig, ProposerConfig,
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
    /// Device: "cpu", "metal", "spyre", or "auto" (the backend this binary was built with).
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
    /// Speculative model: `"ngram"` selects the n-gram proposer; any other value names a draft
    /// checkpoint (local path or HuggingFace repo ID) — a multi-token-prediction head published
    /// apart from its target, or a standalone draft model. `None` drafts with the target's
    /// compiled multi-token-prediction head when the build has one (metal), else disables
    /// speculative decoding.
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
    /// Tensor-parallel degree. Only 1 is supported (one device per model).
    pub tensor_parallel_size: usize,
    /// Pipeline-parallel degree. Only 1 is supported (one device per model).
    pub pipeline_parallel_size: usize,
    /// Whether to disable async scheduling (overlap GPU/CPU work).
    /// Default false — async scheduling is enabled by default.
    pub disable_async_scheduling: bool,
    /// Runner type: "generate" (default) or "pooling".
    pub runner: String,
    /// Whether prefix caching is enabled (KV cache reuse for shared prompts).
    /// Default: true.
    pub enable_prefix_caching: bool,
    /// Maximum number of tokens processed in a single scheduler iteration.
    /// None = auto (min(max_model_len, 8192)).
    pub max_num_batched_tokens: Option<usize>,
    /// KV cache data type: "auto" (use model dtype) or "fp8_e4m3".
    pub kv_cache_dtype: String,
    /// Compute KV scales dynamically from the first forward pass.
    pub calculate_kv_scales: bool,
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
            disable_async_scheduling: false,
            runner: "generate".to_string(),
            enable_prefix_caching: true,
            max_num_batched_tokens: None,
            kv_cache_dtype: "auto".to_string(),
            calculate_kv_scales: false,
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
/// chunk. Keeps sliding KV a few hundred MiB at 32k vs ~1 GiB.
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
    preloaded_tokenizer: Option<&Tokenizer>,
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
    Draft, ProgressCallback, VerifySeqs, WorkerCreationResult, WorkerFactory,
};

/// Create the appropriate worker via the `WorkerFactory` inventory registry.
///
/// Builds the shared [`WorkerCreateConfig`] once, then selects the first
/// registered backend factory whose `matches` accepts `config.device` and
/// calls its `create`. The per-backend construct→init→load→extract logic
/// lives in `scratchy_serving_worker` (`MetalWorkerFactory` /
/// `SpyreWorkerFactory`); there are no `#[cfg]` backend arms here.
///
/// Returns `(worker, hf_config, model_dir, dtype)`.
fn create_worker(
    config: &VllmConfig,
    speculation: &Speculation,
    model_path: String,
    progress: Option<&Arc<crate::progress::StartupProgress>>,
) -> Result<WorkerCreationResult> {
    use scratchy_serving_worker::worker_factory::WorkerCreateConfig;

    let is_pooling = config.runner == "pooling";

    let worker_config = WorkerCreateConfig {
        model_path: model_path.clone(),
        dtype: config.dtype.clone(),
        hf_token: config.hf_token.clone(),
        // The backend's own page size when it has one (see `effective_block_size`); a mismatch
        // between this and the scheduler's block size silently corrupts KV slots.
        block_size: effective_block_size(config.block_size, &config.device),
        // The REQUEST, not a resolved number: `None` lets the worker resolve
        // the width itself (device tier / memory-affordable GDN slots), while
        // `Some(n)` pins every worker-side consumer to the caller's ask.
        max_num_seqs: config.max_num_seqs,
        enable_prefix_caching: requested_prefix_caching(config),
        max_num_batched_tokens: config.max_num_batched_tokens.unwrap_or(1024),
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
        draft: speculation.worker_draft(),
        verify_seqs: speculation.verify_seqs(),
        draft_model_dtype: config.draft_model_dtype.clone(),
    };

    // Progress closure. For now it just observes
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
            "No inference backend is compiled into this binary. Rebuild with a \
             backend feature:\n    --features metal   (Apple Silicon)\n    --features \
             spyre   (IBM Spyre)\ne.g. `cargo build --release \
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
                 Build with --features metal (Apple Silicon) or --features spyre (IBM Spyre).",
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
    declared_max_model_len: Option<usize>,
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

    // For Metal workers, determine_available_memory already returns
    // the KV cache budget with gpu_memory_utilization baked in (matching
    // Python vLLM: total * util - non_kv_cache). On Metal `total` is further
    // capped at `maxBufferLength` (the real wireable ceiling) and util is
    // applied inside the worker, so the returned value is the FINAL KV budget
    // — do NOT re-apply utilization here (that double-shrinks KV). Spyre is the
    // same case on both its tiers (see [`kv_budget_is_prebaked`]). Only CPU
    // gets the returned value as raw free memory and applies util locally.
    let is_gpu_prebaked = kv_budget_is_prebaked(device, cfg!(feature = "spyre"));
    let is_cpu = device_is_cpu(device, cfg!(feature = "spyre"));

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

    let (mut num_gpu_blocks, swa_hybrid_kv) = compute_kv_blocks(
        kv_cache_bytes,
        block_size,
        hf_config,
        dtype_elem_bytes,
        utilization,
        kv_cache_dtype,
        worker.kv_block_bytes(block_size),
        worker.supports_hybrid_swa_kv(),
    );
    // Fixed-size resident KV pool cap (spyre/sendnn PAGED CB): the on-card pool is
    // exactly `nblk` blocks, so the scheduler's block allocator must not exceed it
    // (else it hands out a block id past the pool → silent KV corruption). The
    // uniform `compute_num_blocks` sizes from the memory budget and can be larger;
    // clamp it. `None` for metal (budget-sized pools).
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
    // ⭐ THE CONTEXT LAW, ENGINE SIDE — the same ONE predicate the worker's pool sizing
    // already ran ([`scratchy_serving_engine::gpu_budget::resolve_served_context`]): an
    // explicit `--max-model-len` the cache just built cannot serve is REFUSED here —
    // capping a stated ask is not validating it; a client sized to the ask (`scr launch
    // claude`'s Claude Code) would fail mid-session when it found out. An UNSET flag
    // resolves to `min(max_model_len, affordable)`, and that resolution is logged at
    // info: it is the designed default, not a shortfall. The scheduler admits a prompt
    // only if its blocks fit the whole pool (`num_gpu_blocks × block_size` tokens), and
    // the backend's kernels address no further than `kv_max_addressable_tokens` (metal
    // hybrid SWA: the sliding class's block-table row; spyre: what its pool funds and its
    // mask baked) — the affordable is the narrower of the two.
    let pool_tokens = num_gpu_blocks * block_size;
    let reach = worker.kv_max_addressable_tokens();
    let affordable = reach.unwrap_or(usize::MAX).min(pool_tokens);
    let effective_max_model_len = match scratchy_serving_engine::gpu_budget::resolve_served_context(
        declared_max_model_len,
        max_model_len,
        affordable,
    ) {
        scratchy_serving_engine::gpu_budget::ServedContext::Refused {
            declared,
            affordable,
        } => {
            anyhow::bail!(
                "--max-model-len {declared} exceeds what this run can serve: the KV cache just built \
                     holds {} block(s) × {} tokens = {} tokens{}, and the longest context it can serve is \
                     {affordable} tokens. A request would fail mid-generation once the pool ran out, so the \
                     run is refused here instead. Lower --max-model-len to about {affordable}, or raise \
                     --gpu-memory-utilization to grow the cache.",
                num_gpu_blocks,
                block_size,
                pool_tokens,
                reach
                    .map(|r| format!(", the backend's kernels address at most {r}"))
                    .unwrap_or_default(),
            )
        }
        scratchy_serving_engine::gpu_budget::ServedContext::Tokens(served) => {
            if served < max_model_len {
                info!(
                    "max_model_len unset: serving {served} of the model's {max_model_len} — what the KV \
                         cache just built funds ({} block(s) × {} tokens{})",
                    num_gpu_blocks,
                    block_size,
                    reach
                        .map(|r| format!(", backend addressable to {r}"))
                        .unwrap_or_default(),
                );
            }
            served
        }
    };
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

/// The speculative decoding a run does: `--speculative-model` and this build's compiled heads,
/// resolved once per stack ([`resolve_speculative_decoding`]) and read everywhere after.
#[derive(Debug, Clone)]
enum Speculation {
    Off,
    /// The n-gram proposer (`--speculative-model ngram`).
    Ngram,
    /// A standalone draft model (a local path or Hub repo), validated against the target.
    DraftModel(String),
    /// A multi-token-prediction head: the target's compiled one ([`Draft::TargetHead`]), or the
    /// one `--speculative-model` names ([`Draft::Checkpoint`]).
    Head {
        head: scratchy_forward_compiler::HeadRegistration,
        draft: Draft,
    },
}

impl Speculation {
    fn is_on(&self) -> bool {
        !matches!(self, Self::Off)
    }

    /// The tokens a step drafts a sequence at most — the scheduler's lookahead. A head drafts what
    /// it was compiled to (`HeadRegistration::drafts`, fewer in a step of more sequences:
    /// [`Self::verify_seqs`]'s budget): its target's verify steps carry at most those rows, which
    /// both models' kernels bake, so `--num-speculative-tokens` counts only an n-gram or draft
    /// model's.
    ///
    /// `--num-speculative-tokens 0` serves without the head ([`resolve_speculative_decoding`]).
    /// TODO: any other count given with a head is ignored, and the flag's default (2) cannot be
    /// told from a given value: make it optional, and refuse one that is not the head's.
    fn drafts(&self, config: &VllmConfig) -> usize {
        match self {
            Self::Off => 0,
            Self::Ngram | Self::DraftModel(_) => config.num_speculative_tokens,
            Self::Head { head, .. } => usize::from(head.drafts),
        }
    }

    /// What the worker loads to draft with.
    fn worker_draft(&self) -> Option<Draft> {
        match self {
            Self::Off | Self::Ngram => None,
            Self::DraftModel(model) => Some(Draft::Checkpoint(model.clone())),
            Self::Head { draft, .. } => Some(draft.clone()),
        }
    }

    /// The sequences a step may verify drafts for: a head drafts for as many as its budget does,
    /// an n-gram or draft model for every sequence the engine schedules.
    fn verify_seqs(&self) -> VerifySeqs {
        match self {
            Self::Off => VerifySeqs::Off,
            Self::Head { head, .. } => VerifySeqs::Head(Self::budget(head)),
            Self::Ngram | Self::DraftModel(_) => VerifySeqs::Every,
        }
    }

    /// The drafts `head` makes a step's sequences: its compiled counts.
    fn budget(head: &scratchy_forward_compiler::HeadRegistration) -> DraftBudget {
        DraftBudget {
            drafts: usize::from(head.drafts),
            verify_rows: usize::from(head.verify_rows),
        }
    }

    /// Mirror Python vLLM (vllm/config/vllm.py:709-721): force-disable async scheduling for
    /// non-EAGLE spec-decode methods. Both ngram and draft-model paths seed K-step drafts from
    /// step N's output for step N+1's verify; async's 1-step lookahead skews that to N+2 and
    /// acceptance collapses to ~0. A multi-token-prediction head drafts in the worker, at the end
    /// of the step it drafts from, and the async loop waits only on a step that drafts or
    /// verifies, so it is compatible with async.
    fn requires_sync(&self) -> bool {
        matches!(self, Self::Ngram | Self::DraftModel(_))
    }

    /// The engine's proposer.
    fn proposer_config(&self, config: &VllmConfig, max_model_len: usize) -> Option<ProposerConfig> {
        let num_speculative_tokens = self.drafts(config);
        match self {
            Self::Off => None,
            Self::Ngram => Some(ProposerConfig::Ngram(NgramProposerConfig {
                num_speculative_tokens,
                max_ngram_size: config.ngram_prompt_lookup_max,
                min_ngram_size: config.ngram_prompt_lookup_min,
                max_model_len,
            })),
            Self::DraftModel(model) => Some(ProposerConfig::DraftModel(DraftModelProposerConfig {
                model: model.clone(),
                num_speculative_tokens,
                dtype: config.draft_model_dtype.clone(),
                max_model_len,
            })),
            Self::Head { head, .. } => Some(ProposerConfig::Mtp(MtpProposerConfig {
                num_speculative_tokens,
                max_seqs: Self::budget(head).max_seqs(),
                max_model_len,
            })),
        }
    }
}

/// The registration of the `--speculative-model` checkpoint's multi-token-prediction head, when
/// its arch, as compiled into this build, is one (it reads the target's hidden states).
fn draft_head(
    config: &VllmConfig,
    spec: &str,
) -> Result<Option<scratchy_forward_compiler::HeadRegistration>> {
    let draft_dir = scratchy_serving_worker::worker_factory::resolve_model_path(
        spec,
        config.hf_token.as_deref(),
        None,
        None,
    )
    .with_context(|| format!("resolving draft model path {spec:?}"))?;
    let draft_cfg = HfModelConfig::from_dir(&draft_dir)
        .with_context(|| format!("reading draft config.json at {draft_dir:?}"))?;
    Ok(draft_cfg
        .arch_hint()
        .and_then(scratchy_forward_compiler::draft_head))
}

/// The speculative decoding a run does ([`Speculation`]), validated before any model load. Called
/// once at the top of each `initialize_stack*` entry.
///
/// - `--speculative-model ngram`: the n-gram proposer.
/// - `--speculative-model <checkpoint>`: a multi-token-prediction head when its config names a
///   head arch this build compiled — it embeds and decodes through the target's own embedding
///   and lm_head (the worker lends them), so its token IDs are the target's by construction —
///   else a standalone draft model, checked against the target (`validate_target_draft_pair`:
///   vocab/tokenizer alignment, and on metal a rough memory budget).
/// - none, on metal: the target's compiled multi-token-prediction head, when this build compiled
///   one for the target and its tensors are found ([`target_head`]); the head is a fact of the
///   compiled model. `--num-speculative-tokens 0` turns it off.
fn resolve_speculative_decoding(config: &VllmConfig) -> Result<Speculation> {
    let speculation = match config.speculative_model.as_deref() {
        None if !cfg!(feature = "metal") || config.num_speculative_tokens == 0 => Speculation::Off,
        None => match target_head(config)? {
            Some((head, draft)) => Speculation::Head { head, draft },
            None => Speculation::Off,
        },
        Some("ngram") => Speculation::Ngram,
        Some(spec) => match draft_head(config, spec)? {
            Some(head) => Speculation::Head {
                head,
                draft: Draft::Checkpoint(spec.to_string()),
            },
            None => {
                validate_target_draft_pair(config, spec)?;
                info!(
                    "spec-decode: draft model {spec:?} accepted; worker-side proposer runs a \
                     K-step decode chain per step."
                );
                Speculation::DraftModel(spec.to_string())
            }
        },
    };
    if let Speculation::Head { head, draft } = &speculation {
        let source = match draft {
            Draft::TargetHead => "the multi-token-prediction head in its checkpoint".to_string(),
            Draft::Checkpoint(head) => format!("the multi-token-prediction head {head}"),
        };
        info!(
            "spec-decode: {} drafts {} tokens a step with {source}.",
            config.model, head.drafts,
        );
    }
    Ok(speculation)
}

/// The target's multi-token-prediction head and where its tensors are, when this build compiled
/// one for the target's checkpoint architecture. The drafter a target drafts with by default when
/// no checkpoint of the target carries it (Gemma 4's assistant, which Google ships only as a repo
/// of its own) loads from the repo it is published as for the target's shape
/// (`HeadRegistration::published_for`). A head the target's config declares MTP layers for loads
/// from the target's own checkpoint when that holds tensors under the head's `checkpoint_prefix`
/// ([`Draft::TargetHead`]) or names the sidecar file an MLX conversion stripped the head into
/// (`mtp_file`), else from the repo it is published under, named from the target's
/// (`HeadRegistration::repo_of`). A published head drafts when its repo resolves
/// ([`Draft::Checkpoint`]). With none the target serves without drafting, and says why — a build
/// that compiles the head serves every checkpoint a build without it does.
fn target_head(
    config: &VllmConfig,
) -> Result<Option<(scratchy_forward_compiler::HeadRegistration, Draft)>> {
    let target_dir = scratchy_serving_worker::worker_factory::resolve_model_path(
        &config.model,
        config.hf_token.as_deref(),
        None,
        None,
    )
    .with_context(|| format!("resolving model path {:?}", config.model))?;
    let target = HfModelConfig::from_dir(&target_dir)
        .with_context(|| format!("reading config.json at {target_dir:?}"))?;
    let arch = target.arch_hint().unwrap_or_default();
    let head = scratchy_forward_compiler::head_of(arch);
    let shape = (target.hidden_size().zip(target.num_hidden_layers())).map(
        |(hidden_size, num_hidden_layers)| scratchy_forward_compiler::TargetShape {
            hidden_size,
            num_hidden_layers,
        },
    );
    if let Some(head) = head
        && let Some(repo) = shape.and_then(|shape| head.published_for(shape))
    {
        return Ok(published_head(config, head, repo.to_string()));
    }
    // A multimodal checkpoint nests its language model's fields under `text_config`.
    let text = target.extra.get("text_config");
    let mtp_layers = (target.extra.get("mtp_num_hidden_layers"))
        .or_else(|| text?.get("mtp_num_hidden_layers"))
        .and_then(|v| v.as_u64());
    let Some(layers) = mtp_layers.filter(|&n| n > 0) else {
        return Ok(None);
    };
    let Some(head) = head else {
        info!(
            "spec-decode: {}'s checkpoint declares {layers} multi-token-prediction layer(s), but \
             this build compiled no head for {arch}; serving without drafting.",
            config.model
        );
        return Ok(None);
    };
    let in_checkpoint = (head.checkpoint_prefix)
        .map(|prefix| scratchy_core_model::weight::checkpoint_holds(&target_dir, prefix))
        .transpose()
        .with_context(|| format!("reading {target_dir:?}'s tensor names"))?;
    // An MLX conversion that strips the head out of the main shards can still publish it as a
    // sidecar file the checkpoint's config names (`mtp_file`, e.g. Nemotron-3 Super OptiQ's
    // `optiq/mtp.safetensors`): the shard index does not list its tensors, but the worker loads
    // the head from that file under the same `checkpoint_prefix`.
    let sidecar = (target.extra.get("mtp_file"))
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .is_some_and(|file| target_dir.join(file).is_file());
    if in_checkpoint == Some(true) || sidecar {
        return Ok(Some((head, Draft::TargetHead)));
    }
    let Some(repo) = head.repo_of(&config.model) else {
        info!(
            "spec-decode: {}'s checkpoint carries no multi-token-prediction head and names no \
             repo one is published under; serving without drafting (`--speculative-model <head>` \
             names one).",
            config.model
        );
        return Ok(None);
    };
    Ok(published_head(config, head, repo))
}

/// `head` drafting from `repo`, the checkpoint it is published as apart from its target's, when
/// that resolves; else none, and why.
fn published_head(
    config: &VllmConfig,
    head: scratchy_forward_compiler::HeadRegistration,
    repo: String,
) -> Option<(scratchy_forward_compiler::HeadRegistration, Draft)> {
    let published = scratchy_serving_worker::worker_factory::resolve_model_path(
        &repo,
        config.hf_token.as_deref(),
        None,
        None,
    );
    match published {
        Ok(_) => Some((head, Draft::Checkpoint(repo))),
        Err(e) => {
            info!(
                "spec-decode: {}'s checkpoint carries no multi-token-prediction head, and {repo} \
                 does not resolve ({e:#}); serving without drafting (`--speculative-model <head>` \
                 names one).",
                config.model
            );
            None
        }
    }
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
/// from a cached KV prefix. A prefix-cache hit on one is only correct where a
/// snapshot of that state is restored with it (a hit would otherwise feed only
/// the new tokens, leaving the recurrent state computed over the wrong span), so
/// prefix caching is off for these models unless the worker keeps snapshots —
/// see [`effective_prefix_caching`]. Mirrors vLLM's hybrid/SSM handling.
/// Detected by HF `architectures` string since the marker is per-layer.
fn is_recurrent_hybrid_arch(architectures: &[String]) -> bool {
    architectures.iter().any(|a| {
        let a = a.as_str();
        // Gated-DeltaNet hybrids (Qwen3.5 / Qwen3.6 / Qwen3-Next).
        a.contains("Qwen3_5")
            || a.contains("Qwen3_6")
            || a.contains("Qwen3Next")
            // Mamba / SSM families (Nemotron-H: Mamba-2 + attention + MoE).
            || a.contains("Mamba")
            || a.contains("NemotronH")
            || a.contains("Jamba")
            || a.contains("FalconH")
            || a.contains("RecurrentGemma")
            || a.contains("Zamba")
            || a.contains("Lfm2")
            || a.contains("GraniteMoeHybrid")
    })
}

/// Prefix caching as requested AND as the backend can honour it at all
/// (`WorkerFactory::supports_prefix_caching`) — what the worker is created with,
/// so it reserves recurrent-state snapshots only for an engine that will use them.
fn requested_prefix_caching(config: &VllmConfig) -> bool {
    config.enable_prefix_caching
        && !inventory::iter::<&dyn scratchy_serving_engine::worker_factory::WorkerFactory>()
            .find(|f| f.matches(&config.device))
            .is_some_and(|f| !f.supports_prefix_caching())
}

/// ⭐ THE ONE RULE FOR WHETHER THE ENGINE SERVES CACHED PREFIXES, at every `EngineCoreConfig` site.
///
/// `requested` is [`requested_prefix_caching`]; a recurrent-hybrid model (see
/// [`is_recurrent_hybrid_arch`]) additionally needs a worker that keeps `snapshots` of its
/// recurrent state, because a hit that restores no state runs the model over the wrong span.
fn effective_prefix_caching(requested: bool, recurrent_hybrid: bool, snapshots: bool) -> bool {
    requested && (!recurrent_hybrid || snapshots)
}

/// Offline (LLM/throughput) vs online-server default for `max_num_batched_tokens`,
/// resolved per device (mirrors Python vLLM `EngineArgs.get_batch_defaults`):
/// an H100-class GPU on the offline path gets 16384 rather than the base 2048.
/// Falls back to the base `SchedulerConfig` constant when the device cannot be
/// queried. Only the batched-tokens value is consumed
/// here; `max_num_seqs` keeps its existing default for now.
fn resolve_default_max_num_batched_tokens(is_offline: bool) -> usize {
    if let Some((total_bytes, name)) = backend_device_total_bytes_and_name() {
        return SchedulerConfig::batch_defaults(total_bytes, &name, is_offline).0;
    }
    SchedulerConfig::DEFAULT_MAX_NUM_BATCHED_TOKENS
}

/// ⭐⭐⭐ THE ONE PLACE the SCHEDULER's `max_num_seqs` IS DECIDED, and the order is
/// WORKER, then DEVICE, then base.
///
/// The REQUEST reaches workers as `Option` through `WorkerCreateConfig::max_num_seqs`:
/// `None` tells the worker nobody asked, so it RESOLVES the width itself at the end of
/// `load_model` (the shared `gpu_budget::resolve_default_max_num_seqs` — device tier
/// where the backend reports facts, else the base constant, clamped to what the device's
/// memory affords where the model carries per-sequence GDN state). AFTER `load_model`,
/// the worker reports the width it actually built its pools, sampler arena, and capture
/// ladder at ([`Worker::max_num_seqs_override`]) — this function honours that answer for
/// an unset flag, so one process runs at one number. Multi-worker paths additionally
/// ALIGN every worker to the narrowest reported width
/// ([`Worker::align_max_num_seqs`]) before `initialize_cache`/warmup, so ranks whose
/// memory clamps landed differently still build identical ladders.
///
/// ⛔ AN EXPLICIT REQUEST IS CAPPED, NEVER REPLACED, and an unset one is ANSWERED, never
/// guessed: `None` + a worker answer is the backend's own default, so `scr serve` with no
/// flags runs at the width the backend sized its pools at rather than at a generic 256
/// that merely happens to clamp to it.
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
        // Nobody asked. The backend's answer IS the default when it has one — its baked width, or
        // the memory-affordable count it just resolved an unset flag into.
        (None, Some(cap)) => cap,
        // No backend answer either: the device tier, else the base constant.
        // Same shared resolver the workers use, with facts built from the
        // registered factory — one arithmetic everywhere.
        (None, None) => {
            let (total, name) = backend_device_total_bytes_and_name()
                .map_or((None, None), |(t, n)| (Some(t), Some(n)));
            resolve_default_max_num_seqs(
                &MaxNumSeqsFacts {
                    device_total_bytes: total,
                    device_name: name,
                    // The fallback only needs the TIER half: every backend
                    // with pools to size answers through
                    // `max_num_seqs_override` before the scheduler asks, so
                    // this arm carries no budget to clamp against.
                    device_budget_bytes: None,
                    allocated_bytes: 0,
                    peak_activation_bytes: 0,
                    gdn_per_slot_bytes: None,
                    sampler_bytes_per_row: 0,
                },
                is_offline,
            )
        }
    }
}

/// Total device memory + name from whichever backend factory is registered in
/// this build (metal XOR spyre), without naming a specific backend crate.
fn backend_device_total_bytes_and_name() -> Option<(u64, String)> {
    inventory::iter::<&dyn WorkerFactory>().find_map(|f| f.device_total_bytes_and_name())
}

/// Shared by [`initialize_stack`] (async server path) and
/// [`initialize_stack_sync`] (sync LLM path). `is_offline` selects the
/// device-aware batched-token default (offline LLM vs online server context).
fn initialize_core(
    config: &VllmConfig,
    speculation: &Speculation,
    progress: Option<&Arc<crate::progress::StartupProgress>>,
    is_offline: bool,
) -> Result<InitializedCore> {
    refuse_multi_device(config)?;

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
    // worker's `take_preloaded_tokenizer`.
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
            Tokenizer::from_file(&tok_path).ok()
        }))
    });

    let (mut worker, hf_config, model_dir, model_dtype) =
        create_worker(config, speculation, model_path, progress)?;

    if let Some(arch) = worker.architecture() {
        info!("Resolved model architecture: {}", arch);
    }

    let preloaded_tokenizer = worker
        .take_preloaded_tokenizer()
        .map(Tokenizer::from_hf_tokenizer)
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
        config.max_model_len,
        max_model_len,
    )?;
    // Capture the target-reactive prefill-bucket cap before `worker` is moved
    // into the engine below — used to clamp the scheduler's batched-token bound.
    let worker_prefill_bucket_max_m = worker.prefill_bucket_max_m();
    // Same reason, same moment: the widest concurrency this backend can DECODE in one batched step, on a
    // backend whose decode-width ladder is fixed at bake time and so cannot grow to meet `max_num_seqs`.
    let worker_max_num_seqs = worker.max_num_seqs_override();
    // And the snapshots of the model's recurrent state it can keep, which decide whether a
    // recurrent-hybrid model may be prefix cached at all.
    let worker_snapshots = worker.snapshot_budget();

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

    let use_async_scheduling = !config.disable_async_scheduling && !speculation.requires_sync();
    // Recurrent / hybrid arches (Mamba, Gated-DeltaNet — Qwen3.5/3.6/Next, …)
    // carry a sequential conv/ssm state that CANNOT be reconstructed from a
    // cached KV prefix: a prefix-cache hit that fed the sequence only its *new*
    // tokens would leave the recurrent state computed over the wrong span
    // (garbage output). A hit on one is correct only where a SNAPSHOT of that
    // state is restored with it, so caching stays off for these models unless
    // the worker keeps snapshots (`Worker::snapshot_budget`) — see
    // `effective_prefix_caching`. (Per-layer linear-attention is the marker;
    // detect by arch string.)
    let enable_prefix_caching = {
        let hybrid = is_recurrent_hybrid_arch(&hf_config.architectures);
        let requested = requested_prefix_caching(config);
        if hybrid && requested {
            if let Some(budget) = worker_snapshots {
                info!(
                    "Prefix caching on for {:?}, a recurrent/hybrid (Mamba/Gated-DeltaNet) \
                     architecture: the worker keeps up to {} snapshots of its recurrent state in \
                     the KV budget, and a hit resumes only where one exists",
                    hf_config.architectures, budget.max_slots
                );
            } else {
                info!(
                    "Prefix caching disabled: {:?} is a recurrent/hybrid (Mamba/Gated-DeltaNet) \
                     architecture and the worker keeps no snapshots of its conv/ssm state, which a \
                     cached KV prefix cannot reconstruct (a backend without them, a pooling or \
                     draft-model run, or no memory for one beside the KV floor)",
                    hf_config.architectures
                );
            }
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
        if config.enable_prefix_caching && !requested {
            info!(
                "Prefix caching disabled: the {} backend's worker does not support it — the scheduler \
                 would skip the prefill for a prefix the worker holds no KV for",
                config.device
            );
        }
        effective_prefix_caching(requested, hybrid, worker_snapshots.is_some())
    };

    // Device-specific `max_num_batched_tokens` default — no flat constant on any
    // backend. On backends that prune a compiled prefill-bucket ladder
    // target-reactively (Metal today), the largest resident bucket the device
    // just selected IS the right per-device default, and a single forward can
    // never exceed it (no NoBucketFits). On backends without ladder pruning,
    // fall back to the device-aware default (queries device mem/name). An explicit
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
    // BAKED (spyre's `BATCH_RUNGS`) cannot grow it to cover `max_num_seqs`, so it
    // reports its widest batchable width and the scheduler is held to it, so the fast path covers the
    // configured concurrency. Past the ladder a backend still runs CORRECTLY (spyre degrades to one launch per request), so
    // this is a throughput guard, never a correctness one — which is why it caps rather than refusing.
    let max_num_seqs = resolve_max_num_seqs(config.max_num_seqs, worker_max_num_seqs, is_offline);
    let engine_config = EngineCoreConfig {
        scheduler_config: SchedulerConfig {
            // Resolved above: the device-selected prefill bucket on pruning
            // backends (Metal), else the device-aware default.
            max_num_batched_tokens,
            max_num_seqs,
            policy: SchedulerPolicy::Fcfs,
            enable_chunked_prefill: true,
            async_scheduling: Some(use_async_scheduling),
            num_lookahead_tokens: speculation.drafts(config),
            ..Default::default()
        },
        max_model_len,
        num_gpu_blocks,
        block_size,
        engine_index: 0,
        async_scheduling: use_async_scheduling,
        use_spec_decode: speculation.is_on(),
        proposer_config: speculation.proposer_config(config, max_model_len),
        eos_token_ids,
        is_pooling: config.runner == "pooling",
        enable_prefix_caching,
        hybrid_kv: swa_hybrid_kv,
    };

    let client = InprocClient::new(engine_config, Box::new(executor));

    let loaded_tokenizer = preloaded_tokenizer.or_else(|| {
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

/// Metal and spyre each run a model on exactly one device: refuse tensor /
/// pipeline parallelism up front rather than silently serving on one device.
fn refuse_multi_device(config: &VllmConfig) -> Result<()> {
    if config.tensor_parallel_size > 1 || config.pipeline_parallel_size > 1 {
        anyhow::bail!(
            "multi-device serving is not supported: metal and spyre run on one device \
             (got --tensor-parallel-size {}, --pipeline-parallel-size {})",
            config.tensor_parallel_size,
            config.pipeline_parallel_size,
        );
    }
    Ok(())
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
    let speculation = resolve_speculative_decoding(config)?;
    let init_start = Instant::now();

    // Create progress bar if logging is below INFO level
    let progress = {
        let show_progress = !tracing::enabled!(tracing::Level::INFO);
        Arc::new(crate::progress::StartupProgress::new(show_progress))
    };

    // Offline LLM/throughput context → device-aware offline batch defaults.
    let core = initialize_core(config, &speculation, Some(&progress), true)?;
    progress.finish();
    info!(
        "init engine (load model, create kv cache) took {:.2} seconds",
        init_start.elapsed().as_secs_f64()
    );
    let chat_template = core
        .model_dir
        .as_deref()
        .map(|dir| resolve_chat_template(config.chat_template.as_deref(), dir))
        .transpose()?
        .flatten();
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
    let speculation = resolve_speculative_decoding(config)?;
    let init_start = Instant::now();

    // Create progress bar if not provided and logging is below INFO level
    let progress = progress.unwrap_or_else(|| {
        // Check if tracing is enabled at INFO level or higher
        let show_progress = !tracing::enabled!(tracing::Level::INFO);
        Arc::new(crate::progress::StartupProgress::new(show_progress))
    });

    progress.set_stage("Initializing backend");
    // Online-server context → keep the (smaller) server batch defaults.
    let core = initialize_core(config, &speculation, Some(&progress), false)?;

    progress.set_stage("Creating engine");

    let client = Box::new(core.client);

    let model_name = core.model_name;
    let max_model_len = core.max_model_len;

    let mut engine = build_engine(
        client,
        model_name.clone(),
        max_model_len,
        core.tokenizer,
        core.model_dir.as_deref(),
        config.chat_template.as_deref(),
    )?;
    engine.set_generation_defaults(load_generation_defaults(core.model_dir.as_deref()));
    if !config.disable_async_scheduling && !speculation.requires_sync() {
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
///
/// `Ok(None)` only when there is no template to use. A template that is
/// there but fails to load — the override or the model's own — is an
/// error: serving on without it would render every prompt by plain
/// concatenation, a format the model was never trained on.
pub(crate) fn resolve_chat_template(
    override_str: Option<&str>,
    model_dir: &Path,
) -> Result<Option<ChatTemplate>> {
    let Some(s) = override_str else {
        return try_load_chat_template(model_dir);
    };
    let trimmed = s.trim();
    let from_path = std::path::Path::new(trimmed);
    let template_str = if from_path.is_file() {
        // Either a .jinja template or a tokenizer_config.json. If
        // the file parses as JSON with a `chat_template` field,
        // use that; otherwise treat the file body as raw Jinja.
        let body = std::fs::read_to_string(from_path)
            .with_context(|| format!("reading --chat-template {}", from_path.display()))?;
        match serde_json::from_str::<serde_json::Value>(&body) {
            Ok(v) => v
                .get("chat_template")
                .and_then(|x| x.as_str().map(String::from))
                .unwrap_or(body),
            Err(_) => body,
        }
    } else {
        trimmed.to_string()
    };
    let mut tpl = ChatTemplate::new(template_str).context("the --chat-template override")?;
    // GGUFs almost always lack bos/eos at the template-string level —
    // the original tokenizer_config.json renders them via
    // `{{- bos_token }}` / `{{ eos_token }}`. When the override is paired
    // with a GGUF, pull bos/eos from the file's metadata (same path the
    // auto-detect branch uses) so an operator-supplied template still
    // gets the right special tokens.
    if let Some(gguf_path) = gguf_in_or_under(model_dir)
        && let Ok(gguf) = scratchy_quantizations::gguf::GgufFile::open(&gguf_path)
    {
        tpl = apply_gguf_special_tokens(&gguf, tpl);
    }
    info!("Chat template loaded from --chat-template override");
    Ok(Some(tpl))
}

/// The serving engine over `client`: with the model's tokenizer when it has
/// one, and then with its chat template ([`resolve_chat_template`]) when it
/// has one of those too.
fn build_engine(
    client: Box<dyn scratchy_serving_engine::core_client::EngineCoreClient + Send>,
    model_name: String,
    max_model_len: usize,
    tokenizer: Option<Arc<Tokenizer>>,
    model_dir: Option<&Path>,
    chat_template_override: Option<&str>,
) -> Result<AsyncEngine> {
    let Some(tokenizer) = tokenizer else {
        return Ok(AsyncEngine::new(client, model_name, max_model_len));
    };
    let chat_template = model_dir
        .map(|dir| resolve_chat_template(chat_template_override, dir))
        .transpose()?
        .flatten();
    Ok(match chat_template {
        Some(template) => AsyncEngine::with_tokenizer_and_template(
            client,
            model_name,
            max_model_len,
            tokenizer,
            Arc::new(template),
        ),
        None => {
            info!("No chat template found, using plain concatenation");
            AsyncEngine::with_tokenizer(client, model_name, max_model_len, tokenizer)
        }
    })
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

/// Load the chat template the model ships. Sources tried, in order:
/// 1. `tokenizer.chat_template` metadata in a `.gguf` file —
///    handles both `model_dir` being a `.gguf` file directly and
///    a directory containing one.
/// 2. `tokenizer_config.json` (HF convention), or the
///    `chat_template.jinja` beside it (Gemma 4, some quantized repos).
///
/// `Ok(None)` when no source carries a template; one that fails to load
/// is an error.
fn try_load_chat_template(model_dir: &Path) -> Result<Option<ChatTemplate>> {
    // GGUF: read from metadata. Try direct file, then look for any
    // .gguf file in the directory.
    if let Some(gguf_path) = gguf_in_or_under(model_dir)
        && let Ok(gguf) = scratchy_quantizations::gguf::GgufFile::open(&gguf_path)
        && let Some(template_str) = scratchy_quantizations::gguf::gguf_chat_template(&gguf)
    {
        let tpl = ChatTemplate::new(template_str)
            .with_context(|| format!("the chat template in {}", gguf_path.display()))?;
        info!("Chat template loaded from GGUF metadata");
        return Ok(Some(apply_gguf_special_tokens(&gguf, tpl)));
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

    let tpl = ChatTemplate::from_tokenizer_config(&model_dir.join("tokenizer_config.json"))
        .with_context(|| format!("the chat template in {}", model_dir.display()))?;
    if tpl.is_some() {
        info!("Chat template loaded from {}", model_dir.display());
    }
    Ok(tpl)
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
    blocks_within(
        available_bytes,
        block_size * num_layers * bytes_per_token_per_layer,
    )
}

/// How many blocks of `bytes_per_block` a uniform KV pool fits in
/// `available_bytes` — at least 16.
fn blocks_within(available_bytes: usize, bytes_per_block: usize) -> usize {
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

/// ⭐ DOES THIS RUN'S WORKER HAND `init_cache` A FINAL KV BUDGET — one with utilization
/// already baked in (metal: `total × util − non_kv_cache`), or one where utilization
/// is deliberately absent (spyre: the card tier's region-capped pool budget, the emulator
/// tier's flat host-RAM allowance — different shapes, but the triage question is the same
/// one: is the worker's number final? For both spyre tiers it is)? For those the returned
/// number IS the budget and must reach `compute_kv_blocks` whole — re-applying utilization
/// would double-shrink KV. Only a CPU worker returns raw free memory for the engine to
/// scale.
///
/// The spyre arm tripped on `spyre-hw` alone until the emulator's fall-through was caught
/// in review: a spyre tier landing in the CPU branch gets its budget halved by that
/// branch's 50%, which sized the 2b card run's scheduler at 204 blocks against a 544-page
/// pool. `spyre_backend` is a PARAMETER, not a cfg read, so one unit test can hold every
/// tier at once; the call site passes the api crate's plain `spyre` feature — the common
/// parent of `spyre-hw` and `spyre-emu`, so BOTH tiers trip it, never only the card.
fn kv_budget_is_prebaked(device: &str, spyre_backend: bool) -> bool {
    (cfg!(feature = "metal") && device != "cpu") || (spyre_backend && device != "cpu")
}

/// Is this run on the CPU worker, whose `determine_available_memory` is raw free memory
/// the engine scales by utilization? `device == "auto"` is CPU only when NO gpu backend
/// was compiled in.
fn device_is_cpu(device: &str, spyre_backend: bool) -> bool {
    device == "cpu" || (device == "auto" && !cfg!(feature = "metal") && !spyre_backend)
}

/// Compute the KV-cache block count AND the hybrid SWA layout, if any.
///
/// Both layouts spend the WHOLE KV budget, as vLLM does: the scheduler admits
/// requests until blocks run out, so a pool smaller than the budget is a
/// concurrency cap nobody asked for. `block_bytes` is the worker's own cost of
/// one block ([`Worker::kv_block_bytes`]) when it is not the dense row derived
/// here — TurboQuant codes, or a hybrid model's shared page plus its TurboQuant
/// global store — and the budget buys that many blocks.
///
/// For uniform (non-SWA) models this is [`compute_num_blocks`] with no hybrid
/// config. For `is_swa_hybrid` models (gemma4) it computes vLLM's
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
    block_bytes: Option<usize>,
    allow_hybrid: bool,
) -> (usize, Option<HybridKvConfig>) {
    let uniform = match block_bytes {
        Some(bytes) => blocks_within(available_bytes, bytes),
        None => compute_num_blocks(
            available_bytes,
            block_size,
            hf_config,
            dtype_elem_bytes,
            utilization,
            kv_cache_dtype,
        ),
    };
    // Only metal wires the `sliding_groups` decode. A worker that reads a single
    // block table for every layer passes `allow_hybrid=false` so gemma4 (and any
    // hybrid-SWA arch) stays on the uniform single pool.
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
            // The WHOLE budget, at what a block really costs. The scheduler admits
            // requests until blocks run out and recomputes a preempted one from
            // token 0 (prefix caching is off on this layout), so this pool's size
            // IS the concurrency the server sustains: a pool of one 32k sequence
            // preempts 12 requests of 640 tokens and never fits 8 of 4.6k. The
            // pool commits lazily and the lowest-free-first allocator keeps live
            // block ids near the working set, so a large pool costs reserved VA;
            // charging each block its TurboQuant global store (`block_bytes`)
            // keeps that inside the budget the activation arena was carved from.
            let num_blocks = block_bytes
                .map_or(layout.config.num_blocks, |bytes| {
                    blocks_within(available_bytes, bytes)
                })
                .max(16);
            tracing::info!(
                "Hybrid SWA KV layout: {} shared blocks ({} groups, group_size {}, page {} B, \
                 {} B per block)",
                num_blocks,
                layout.num_groups(),
                layout.group_size,
                layout.page_size_bytes,
                block_bytes.unwrap_or(layout.page_size_bytes * layout.group_size),
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
    fn every_spyre_tier_hands_init_cache_a_final_budget() {
        // BOTH spyre tiers — the card AND the emulator — must be prebaked and must never
        // take the CPU branch. The emulator's budget is a different SHAPE (a flat
        // host-RAM allowance; no paged pool, no stride, no override — the engine's
        // dense-row estimate is all it divides by), but the triage question is the same
        // one and a tier that falls through gets its budget halved by the CPU branch's
        // 50%. Keyed on the plain `spyre` parent at the call site so neither tier can be
        // added back without this holding.
        for device in ["auto", "spyre"] {
            assert!(
                kv_budget_is_prebaked(device, true),
                "the spyre tier must be prebaked (device {device})"
            );
            assert!(
                !device_is_cpu(device, true),
                "the spyre tier must not take the CPU branch (device {device})"
            );
        }
        // An explicit CPU ask stays CPU even on a spyre build — the worker is chosen by
        // the factory match on the device string, not by the feature list.
        assert!(device_is_cpu("cpu", true));
        assert!(!kv_budget_is_prebaked("cpu", true));
    }

    #[test]
    fn a_recurrent_hybrid_is_prefix_cached_only_with_snapshots() {
        assert!(
            effective_prefix_caching(true, false, false),
            "attention-only: as requested"
        );
        assert!(
            !effective_prefix_caching(true, true, false),
            "no snapshots: off"
        );
        assert!(effective_prefix_caching(true, true, true), "snapshots: on");
        assert!(
            !effective_prefix_caching(false, true, true),
            "never past the request"
        );
    }

    #[test]
    fn test_extract_model_name_hf_id() {
        assert_eq!(
            extract_model_name("meta-llama/Llama-3.2-1B"),
            "meta-llama/Llama-3.2-1B"
        );
    }

    /// Nemotron-H's Mamba-2 layers carry conv/ssm state no cached KV prefix
    /// restores, so its architecture string turns prefix caching off.
    #[test]
    fn test_nemotron_h_is_a_recurrent_hybrid() {
        for arch in ["NemotronHForCausalLM", "Qwen3_5MoeForConditionalGeneration"] {
            assert!(is_recurrent_hybrid_arch(&[arch.to_string()]), "{arch}");
        }
        assert!(!is_recurrent_hybrid_arch(&["LlamaForCausalLM".to_string()]));
    }

    /// The mlx-community Nemotron-3 checkpoints' stop ids: `config.json` says
    /// 11 (Nano) — the base model's 2 drifted — and `generation_config.json`
    /// `[2, 11]`; the merge keeps both.
    #[test]
    fn test_nemotron_eos_ids_merge_config_and_generation_config() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("generation_config.json"),
            r#"{"bos_token_id": 1, "eos_token_id": [2, 11], "pad_token_id": 0}"#,
        )
        .unwrap();
        let mut config = HfModelConfig::default();
        config
            .extra
            .insert("eos_token_id".to_string(), serde_json::json!(11));
        assert_eq!(
            resolve_eos_token_ids(&config, Some(dir.path()), None),
            vec![11, 2]
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

    /// The hybrid (gemma4) pool spends the WHOLE budget at the worker's cost of a
    /// block, like every other pool. Sized to one `max_model_len` sequence it was
    /// 2,634 blocks on gemma-4-26b whatever the budget — a concurrency cap that
    /// preempted 12 concurrent 640-token requests and never fit 8 of 4,608.
    #[test]
    fn test_hybrid_pool_spends_the_budget_at_the_workers_block_cost() {
        let config = HfModelConfig::from_file(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../models/arch/configs/gemma4-moe/gemma-4-26b-a4b-it.json"
        ))
        .expect("gemma-4-26b config");
        // gemma-4-26b's sliding page: 16 tokens × 8 KV heads × (256 K + 256 V) × 2 B.
        let page = 16 * 8 * 512 * 2;
        let group_size = 5;
        // ~ what the metal worker reports: the shared pages + the TurboQuant global store.
        let block_bytes = page * group_size + 297_472;
        for budget in [3usize << 30, 22 << 30] {
            let (num_blocks, hybrid) =
                compute_kv_blocks(budget, 16, &config, 2, 0.9, "auto", Some(block_bytes), true);
            let (pool, groups) = hybrid.expect("gemma4 takes the hybrid layout");
            assert_eq!(num_blocks, budget / block_bytes, "budget {budget}");
            assert_eq!(pool, num_blocks, "scheduler and worker size one pool");
            assert_eq!(groups.len(), 6, "1 full + 5 sliding groups");
            // No worker cost: the dense shared pages alone.
            let (dense, _) = compute_kv_blocks(budget, 16, &config, 2, 0.9, "auto", None, true);
            assert_eq!(dense, budget / (page * group_size), "budget {budget}");
        }
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

    /// A template that will not compile, in each place one comes from.
    const BROKEN_TEMPLATE: &str = "{% for m in messages %}{{ m.content }}";

    /// No template anywhere is the one case that serves by plain
    /// concatenation; a model that ships a template it cannot load does not
    /// start on it silently.
    #[test]
    fn test_resolve_chat_template_refuses_a_template_that_fails_to_load() {
        let none = tempfile::tempdir().unwrap();
        assert!(resolve_chat_template(None, none.path()).unwrap().is_none());

        let config = tempfile::tempdir().unwrap();
        std::fs::write(
            config.path().join("tokenizer_config.json"),
            serde_json::json!({ "chat_template": BROKEN_TEMPLATE }).to_string(),
        )
        .unwrap();
        assert!(resolve_chat_template(None, config.path()).is_err());

        let jinja = tempfile::tempdir().unwrap();
        std::fs::write(jinja.path().join("chat_template.jinja"), BROKEN_TEMPLATE).unwrap();
        assert!(resolve_chat_template(None, jinja.path()).is_err());

        assert!(resolve_chat_template(Some(BROKEN_TEMPLATE), none.path()).is_err());
    }

    /// An empty embedded `chat_template` falls through to the
    /// `chat_template.jinja` beside it, as a missing one does.
    #[test]
    fn test_resolve_chat_template_reads_the_jinja_beside_an_empty_one() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("tokenizer_config.json"),
            r#"{"chat_template": "", "bos_token": "<s>"}"#,
        )
        .unwrap();
        std::fs::write(
            dir.path().join("chat_template.jinja"),
            "{{ bos_token }}{% for m in messages %}{{ m.content }}{% endfor %}",
        )
        .unwrap();
        let template = resolve_chat_template(None, dir.path()).unwrap().unwrap();
        let messages = [serde_json::json!({"role": "user", "content": "hi"})];
        assert_eq!(template.apply(&messages, false, None).unwrap(), "<s>hi");
    }
}
