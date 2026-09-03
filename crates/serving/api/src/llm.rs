// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Offline batch inference API.
//!
//! Provides `LLM` — a synchronous, Python-like programmatic interface for
//! running inference without an HTTP server. Mirrors the Python
//! `vllm.LLM(model=...).generate()` / `.chat()` pattern.
//!
//! ```rust,no_run
//! use scratchy_serving_api::llm::LLM;
//!
//! let mut llm = LLM::new("HuggingFaceTB/SmolLM2-135M")?;
//! let outputs = llm.generate(&["Hello, world!"], None)?;
//! for output in &outputs {
//!     println!("{}", output.outputs[0].text);
//! }
//! # Ok::<(), anyhow::Error>(())
//! ```

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Result;

use scratchy_core_common::EngineCoreRequest;
pub use scratchy_core_common::SamplingParams;
use scratchy_core_config::CudaGraphConfig;
use scratchy_serving_engine::core_client::{EngineCoreClient, InprocClient};
use scratchy_serving_engine::engine_core::StepOutcome;

use crate::chat_template::ChatTemplate;
use crate::detokenizer::IncrementalDetokenizer;
use crate::init::VllmConfig;
use crate::tokenizer::Tokenizer;

// ---------------------------------------------------------------------------
// Output types
// ---------------------------------------------------------------------------

/// A single completion output (one of possibly `n` per prompt).
#[derive(Debug, Clone)]
pub struct CompletionOutput {
    /// Index within the request's `n` completions.
    pub index: u32,
    /// Generated text.
    pub text: String,
    /// Generated token IDs.
    pub token_ids: Vec<u32>,
    /// Why generation stopped (e.g. "stop", "length").
    pub finish_reason: Option<String>,
}

/// Output for a single prompt / request.
#[derive(Debug, Clone)]
pub struct RequestOutput {
    /// Unique request identifier.
    pub request_id: String,
    /// The original prompt text (if available).
    pub prompt: Option<String>,
    /// Prompt token IDs.
    pub prompt_token_ids: Vec<u32>,
    /// One or more completion outputs.
    pub outputs: Vec<CompletionOutput>,
    /// Whether generation is complete.
    pub finished: bool,
    /// Time to first token in seconds (measured in generate_impl output loop).
    pub ttft_s: Option<f64>,
    /// Average inter-token latency in seconds.
    pub avg_itl_s: Option<f64>,
}

/// One generate step returned by [`LLM::execute_query`].
#[derive(Debug, Clone)]
pub struct GenerateStep {
    /// Human-readable label: `"inner[0]"`, `"inner[1]"`, …, `"outer"`.
    pub label: String,
    /// The completion output for this step.
    pub output: RequestOutput,
    /// Wall-clock time for this step in milliseconds.
    pub elapsed_ms: f64,
}

/// Result of [`LLM::execute_query`] — per-step outputs and timing for all
/// generate calls in the query (inner generates + the final outer generate).
#[derive(Debug, Clone)]
pub struct QueryOutput {
    pub steps: Vec<GenerateStep>,
}

impl std::ops::Index<usize> for QueryOutput {
    type Output = RequestOutput;
    fn index(&self, i: usize) -> &RequestOutput {
        &self.steps[i].output
    }
}

impl QueryOutput {
    /// The final (outer) generate output.
    pub fn output(&self) -> &RequestOutput {
        &self
            .steps
            .last()
            .expect("QueryOutput must have at least one step")
            .output
    }
    /// Inner generate steps (all but the last).
    pub fn inner_steps(&self) -> &[GenerateStep] {
        let n = self.steps.len();
        if n > 1 { &self.steps[..n - 1] } else { &[] }
    }
    /// The outer (final) step.
    pub fn outer_step(&self) -> &GenerateStep {
        self.steps
            .last()
            .expect("QueryOutput must have at least one step")
    }
}

// ---------------------------------------------------------------------------
// Prompt — text or pre-tokenized input (mirrors Python's PromptType)
// ---------------------------------------------------------------------------

/// A prompt for [`LLM::generate()`].
///
/// Mirrors Python vLLM's `PromptType`: either a text string or pre-tokenized
/// token IDs. Use the `From` impls for ergonomic construction:
///
/// ```rust
/// use scratchy_serving_api::llm::Prompt;
///
/// let text: Prompt = "Hello, world!".into();
/// let token_ids: Prompt = vec![1u32, 2, 3].into();
/// ```
#[derive(Debug, Clone)]
pub enum Prompt {
    /// A text prompt (will be tokenized by the engine).
    Text(String),
    /// Pre-tokenized prompt token IDs (skips tokenization).
    TokenIds(Vec<u32>),
    /// Pre-tokenized with span block annotations for relocatable caching.
    TokenIdsWithAnnotations(Vec<u32>, scratchy_core_common::BlockAnnotations),
}

impl From<&str> for Prompt {
    fn from(s: &str) -> Self {
        Prompt::Text(s.to_string())
    }
}

impl From<String> for Prompt {
    fn from(s: String) -> Self {
        Prompt::Text(s)
    }
}

impl From<Vec<u32>> for Prompt {
    fn from(ids: Vec<u32>) -> Self {
        Prompt::TokenIds(ids)
    }
}

// ---------------------------------------------------------------------------
// ChatMessage — ergonomic wrapper
// ---------------------------------------------------------------------------

/// A chat message for `LLM::chat()`.
///
/// Thin convenience type so callers don't need to construct
/// `protocol::ChatCompletionMessageParam` directly.
#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl ChatMessage {
    /// Create a new chat message.
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
        }
    }

    /// Shorthand for a system message.
    pub fn system(content: impl Into<String>) -> Self {
        Self::new("system", content)
    }

    /// Shorthand for a user message.
    pub fn user(content: impl Into<String>) -> Self {
        Self::new("user", content)
    }

    /// Shorthand for an assistant message.
    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new("assistant", content)
    }
}

// ---------------------------------------------------------------------------
// LLMBuilder
// ---------------------------------------------------------------------------

/// Builder for configuring and constructing an [`LLM`] instance.
pub struct LLMBuilder {
    config: VllmConfig,
}

impl LLMBuilder {
    /// Create a builder for the given model.
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            config: VllmConfig {
                model: model.into(),
                ..VllmConfig::default()
            },
        }
    }

    /// Set the device ("cpu", "cuda:0", "metal", "auto").
    pub fn device(mut self, device: impl Into<String>) -> Self {
        self.config.device = device.into();
        self
    }

    /// Set the weight dtype ("auto", "float16", "bfloat16", "float32").
    pub fn dtype(mut self, dtype: impl Into<String>) -> Self {
        self.config.dtype = dtype.into();
        self
    }

    /// Set the KV cache dtype ("auto", "fp16", "turboquant", "fp8_e4m3").
    /// "auto" upgrades to TurboQuant on metal for supported arches; pass
    /// "fp16" to force a lossless KV cache (required for relocatable spans /
    /// rope-on-read transparency).
    pub fn kv_cache_dtype(mut self, dtype: impl Into<String>) -> Self {
        self.config.kv_cache_dtype = dtype.into();
        self
    }

    /// Set the maximum model context length.
    pub fn max_model_len(mut self, len: usize) -> Self {
        self.config.max_model_len = Some(len);
        self
    }

    /// Set the maximum number of concurrent sequences.
    pub fn max_num_seqs(mut self, n: usize) -> Self {
        self.config.max_num_seqs = Some(n);
        self
    }

    /// Set the maximum number of tokens per scheduler iteration.
    pub fn max_num_batched_tokens(mut self, n: usize) -> Self {
        self.config.max_num_batched_tokens = Some(n);
        self
    }

    /// Set the KV cache block size in tokens.
    pub fn block_size(mut self, size: usize) -> Self {
        self.config.block_size = size;
        self
    }

    /// Set the fraction of GPU memory for KV cache (0.0–1.0).
    pub fn gpu_memory_utilization(mut self, frac: f64) -> Self {
        self.config.gpu_memory_utilization = frac;
        self
    }

    /// Set the HuggingFace token for gated models.
    pub fn hf_token(mut self, token: impl Into<String>) -> Self {
        self.config.hf_token = Some(token.into());
        self
    }

    /// Set a specific GGUF filename to download.
    pub fn gguf_file(mut self, filename: impl Into<String>) -> Self {
        self.config.gguf_file = Some(filename.into());
        self
    }

    /// Set the number of GPUs for tensor parallelism.
    pub fn tensor_parallel_size(mut self, n: usize) -> Self {
        self.config.tensor_parallel_size = n;
        self
    }

    /// Set the number of GPU stages for pipeline parallelism.
    pub fn pipeline_parallel_size(mut self, n: usize) -> Self {
        self.config.pipeline_parallel_size = n;
        self
    }

    /// Set the number of nodes for multi-node TP.
    pub fn num_nodes(mut self, n: usize) -> Self {
        self.config.num_nodes = n;
        self
    }

    /// Set this node's rank (0 = master).
    pub fn node_rank(mut self, rank: usize) -> Self {
        self.config.node_rank = rank;
        self
    }

    /// Set the master address for multi-node NCCL rendezvous.
    pub fn master_addr(mut self, addr: &str) -> Self {
        self.config.master_addr = addr.to_string();
        self
    }

    /// Set the master port for multi-node NCCL rendezvous.
    pub fn master_port(mut self, port: u16) -> Self {
        self.config.master_port = port;
        self
    }

    /// Enable or disable prefix caching (KV cache reuse for shared prefixes).
    pub fn enable_prefix_caching(mut self, enabled: bool) -> Self {
        self.config.enable_prefix_caching = enabled;
        self
    }

    /// Disable CUDA graph capture and run all steps eagerly.
    pub fn enforce_eager(mut self, eager: bool) -> Self {
        self.config.enforce_eager = eager;
        self
    }

    /// Set an explicit chat template override. Accepts either inline
    /// Jinja or a path to a `tokenizer_config.json` / `.jinja` file.
    /// Used to inject a template for GGUFs (e.g. mmnga's Moonlight)
    /// whose metadata dropped the field.
    pub fn chat_template(mut self, tpl: impl Into<String>) -> Self {
        self.config.chat_template = Some(tpl.into());
        self
    }

    /// Set the CUDA graph configuration for decode acceleration.
    pub fn cuda_graph_config(mut self, config: CudaGraphConfig) -> Self {
        self.config.cuda_graph_config = Some(config);
        self
    }

    /// Build the [`LLM`] instance, loading the model.
    pub fn build(self) -> Result<LLM> {
        LLM::from_config(self.config)
    }
}

// ---------------------------------------------------------------------------
// LLM
// ---------------------------------------------------------------------------

/// ⛔⛔⛔ WHAT A BLOCKING STEP LOOP OWES A STALLED STEP.
///
/// Nothing else runs on this thread — no HTTP handler is about to submit a request, no background
/// loop is about to free a block — so a step that made no progress will make none on the next tick
/// either. The state is terminal; only the loop disagrees, and it disagrees forever at 100% CPU.
///
/// This is how a REFUSED launch reaches the user as a HANG instead of the message the executor
/// already wrote: `chat --bench` discarded its warmup's `Err`, that request stayed RUNNING, and with
/// `max_num_seqs(1)` the scheduler could then never admit the real one.
///
/// [`StepOutcome::Stalled`] makes forgetting this a COMPILE error rather than a hang, so this
/// function is the answer both blocking loops give, in one place.
fn stalled_step(context: &str) -> anyhow::Error {
    anyhow::anyhow!(
        "engine STALLED during {context}: the step scheduled no work and produced no output while \
         requests are still unfinished. A blocking step loop has no other source of progress, so \
         this cannot resolve itself. The usual cause is an EARLIER step whose error was discarded, \
         leaving a request RUNNING that can never advance."
    )
}

/// Offline batch inference engine.
///
/// Owns an [`InprocClient`] and drives it synchronously with a tight
/// `add_request()` + `while has_unfinished: get_output()` loop — matching
/// Python's `LLM._run_engine()`. No async channels, no background step loop.
pub struct LLM {
    client: InprocClient,
    tokenizer: Option<Arc<Tokenizer>>,
    chat_template: Option<ChatTemplate>,
    model_name: String,
    max_model_len: usize,
    block_size: usize,
    /// Model-recommended sampling defaults (`generation_config.json`),
    /// Python vLLM `generation_config="auto"` parity.
    generation_defaults: scratchy_core_common::sampling::GenerationDefaults,
}

impl LLM {
    /// Create an LLM with default settings for the given model.
    ///
    /// Equivalent to `LLM::builder(model).build()`.
    pub fn new(model: impl Into<String>) -> Result<Self> {
        Self::builder(model).build()
    }

    /// Return a builder for fine-grained configuration.
    pub fn builder(model: impl Into<String>) -> LLMBuilder {
        LLMBuilder::new(model)
    }

    /// Internal constructor from a fully-specified config.
    fn from_config(mut config: VllmConfig) -> Result<Self> {
        // Disable prefix caching for any backend whose worker can't honor it
        // (spyre keeps a per-request KV cache — see `WorkerFactory::
        // supports_prefix_caching`). Otherwise the scheduler reports a cached
        // prefix the worker has no KV for and the forward desyncs. Done here,
        // before the scheduler/cache are built from `config`.
        if config.enable_prefix_caching
            && inventory::iter::<&dyn scratchy_serving_engine::worker_factory::WorkerFactory>()
                .find(|f| f.matches(&config.device))
                .is_some_and(|f| !f.supports_prefix_caching())
        {
            config.enable_prefix_caching = false;
        }
        let mut stack = crate::init::initialize_stack_sync(&config)?;
        // Start the background executor pipeline for overlapping CPU scheduling with GPU execution
        // (the server path uses its own pipeline via spawn_step_loop_async instead).
        //
        // ONLY when more than one sequence can be in flight. The pipeline overlaps by keeping two
        // batches queued, but `schedule_next` cannot produce a second batch for a single sequence —
        // the next token is not known until the current forward returns — so `gpu_in_flight` never
        // reaches 2 and no overlap ever happens. What it does instead is hand every result back one
        // forward late: `get_output_pipelined` finalizes the PREVIOUS forward, then schedules the
        // next batch and BLOCKS on it before returning. Measured on granite via SCRATCHY_TTFT_PHASE
        // (max_num_seqs=1, 20-token prompt): the prefill completes in 57.13 ms and produces the first
        // token, but it is not returned until 88.32 ms — the caller waits 31 ms, a whole decode step
        // for the SECOND token, holding a token that is already finished. The sync path returns it as
        // soon as it exists, and drops a thread hop per step.
        //
        // Token N still lands at the same wall clock either way (the forwards are unchanged and
        // strictly ordered — decode k needs token k), so this moves latency out of TTFT rather than
        // removing work. Reported tok/s will FALL, because chat.rs measures it from the first token
        // onward and that window now starts earlier over the same run; that is the metric moving,
        // not the run getting slower.
        // `None` = the backend decides, and every backend's own default is wider than one.
        if config.max_num_seqs.unwrap_or(usize::MAX) > 1 {
            stack.client.start_pipeline();
        }
        Ok(Self {
            client: stack.client,
            tokenizer: stack.tokenizer,
            chat_template: stack.chat_template,
            model_name: stack.model_name,
            max_model_len: stack.max_model_len,
            block_size: stack.block_size,
            generation_defaults: stack.generation_defaults,
        })
    }

    /// The model name / HuggingFace ID.
    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    /// [`SamplingParams::default`] overlaid with the model's
    /// `generation_config.json` recommendations — the base callers
    /// should start from before applying explicit user choices
    /// (matches HF transformers / mlx-lm / Python vLLM "auto").
    pub fn default_sampling_params(&self) -> SamplingParams {
        self.generation_defaults.as_base()
    }

    /// `generation_config.json::max_new_tokens` if the model ships one
    /// (distinct from `SamplingParams::default().max_tokens`, which is
    /// the OpenAI-conventional 16 and NOT a model recommendation).
    pub fn generation_max_new_tokens(&self) -> Option<u32> {
        self.generation_defaults.max_new_tokens
    }

    /// The maximum context length.
    pub fn max_model_len(&self) -> usize {
        self.max_model_len
    }

    /// The tokenizer, if one was loaded.
    pub fn tokenizer(&self) -> Option<&Arc<crate::tokenizer::Tokenizer>> {
        self.tokenizer.as_ref()
    }

    /// Tokenize a text string, using the tokenizer if available.
    fn tokenize_text(&self, text: &str) -> Result<Vec<u32>> {
        if let Some(tok) = &self.tokenizer {
            if text.is_empty() {
                Ok(vec![])
            } else {
                Ok(tok.encode(text, false)?)
            }
        } else if text.is_empty() {
            Ok(vec![0])
        } else {
            Ok(text.as_bytes().iter().map(|&b| b as u32).collect())
        }
    }

    /// Resolve `max_tokens` to fit within `max_model_len - prompt_len`.
    fn resolve_max_tokens(&self, sp: &mut SamplingParams, num_prompt_tokens: usize) {
        let remaining = self.max_model_len.saturating_sub(num_prompt_tokens);
        match sp.max_tokens {
            None => sp.max_tokens = Some(remaining as u32),
            Some(val) => sp.max_tokens = Some(val.min(remaining as u32)),
        }
    }

    // -----------------------------------------------------------------------
    // generate()
    // -----------------------------------------------------------------------

    /// Generate completions for one or more prompts.
    ///
    /// Accepts both text and pre-tokenized prompts via [`Prompt`], mirroring
    /// Python vLLM's `PromptType`. Set `SamplingParams::detokenize` to
    /// `false` to skip detokenization (e.g. for benchmarking).
    ///
    /// Each prompt produces a [`RequestOutput`] with one or more
    /// [`CompletionOutput`]s (controlled by `SamplingParams::n`).
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use scratchy_serving_api::llm::LLM;
    /// # let mut llm = LLM::new("model")?;
    /// // Text prompts:
    /// llm.generate(&["Hello", "World"], None)?;
    ///
    /// // Token ID prompts:
    /// llm.generate(&[vec![1u32, 2, 3]], None)?;
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn generate<P: Into<Prompt> + Clone>(
        &mut self,
        prompts: &[P],
        params: Option<SamplingParams>,
    ) -> Result<Vec<RequestOutput>> {
        self.generate_impl(prompts, params, false, false, false)
    }

    /// Like [`generate`](Self::generate), but with seal/volatile lifecycle flags
    /// applied to all requests in the batch.
    pub fn generate_sealed<P: Into<Prompt> + Clone>(
        &mut self,
        prompts: &[P],
        params: Option<SamplingParams>,
        seal: bool,
        volatile: bool,
    ) -> Result<Vec<RequestOutput>> {
        self.generate_impl(prompts, params, false, seal, volatile)
    }

    // -----------------------------------------------------------------------
    // embed()
    // -----------------------------------------------------------------------

    /// Generate embedding vectors for one or more prompts.
    ///
    /// Accepts both text and pre-tokenized prompts via [`Prompt`], mirroring
    /// Python vLLM's `LLM.embed()`. The model must support embeddings
    /// (pooling mode).
    ///
    /// Returns one embedding vector per prompt.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use scratchy_serving_api::llm::LLM;
    /// # let mut llm = LLM::new("nomic-embed-text")?;
    /// let embeddings = llm.embed(&["Hello world", "Another sentence"])?;
    /// assert_eq!(embeddings.len(), 2);
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn embed<P: Into<Prompt> + Clone>(&mut self, prompts: &[P]) -> Result<Vec<Vec<f32>>> {
        let token_id_seqs: Vec<Vec<u32>> = prompts
            .iter()
            .map(|p| {
                let prompt: Prompt = p.clone().into();
                match prompt {
                    Prompt::Text(text) => self.tokenize_text(&text),
                    Prompt::TokenIds(ids) => Ok(ids),
                    Prompt::TokenIdsWithAnnotations(ids, _) => Ok(ids),
                }
            })
            .collect::<Result<Vec<_>>>()?;

        self.client
            .embed(token_id_seqs)
            .map_err(|e| anyhow::anyhow!("embed failed: {e}"))
    }

    /// Execute a SPNL span query (JSON string), using the same tokenization
    /// and annotation logic as the server's `/v1/query/execute` endpoint.
    ///
    /// `seal` and `volatile` control KV cache lifecycle:
    /// - `seal=true`: seal generated blocks so they persist for reuse
    /// - `volatile=true`: mark the request as volatile (output evictable)
    ///
    /// If `params` is provided, its `max_tokens` and `temperature` override
    /// the values in the SPNL query metadata.
    pub fn execute_query(
        &mut self,
        spnl_json: &str,
        params: Option<SamplingParams>,
        seal: bool,
        volatile: bool,
    ) -> Result<QueryOutput> {
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("execute_query requires a tokenizer"))?
            .clone();
        // Safety: template ref is valid for the duration of this call; we need
        // a raw pointer to avoid the borrow conflict with the `self` closure below.
        let template_ptr: *const crate::chat_template::ChatTemplate =
            self.chat_template
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("execute_query requires a chat template"))?;

        let cfg = crate::spans::SpanConfig::from_tokenizer(self.block_size, &tokenizer);
        let block_size = self.block_size;

        // SAFETY: template_ptr points into self.chat_template which is not mutated
        // during this call.
        let template = unsafe { &*template_ptr };

        #[cfg(feature = "rag")]
        let aug_options = {
            let embedder: Option<std::sync::Arc<dyn crate::augment::embed::TokenEmbedder>> = self
                .client
                .embed_sender()
                .map(|s| std::sync::Arc::new(s) as _);
            crate::augment::AugmentOptions {
                current_model: Some(self.model_name.clone()),
                embedder,
                tokenizer: self.tokenizer.clone(),
                sidecar_manager: Some(std::sync::Arc::new(crate::augment::SidecarManager::new())),
                ..Default::default()
            }
        };

        crate::spans::execute_spnl_query_sync(
            spnl_json,
            params,
            seal,
            volatile,
            #[cfg(feature = "rag")]
            &aug_options,
            &tokenizer,
            template,
            &cfg,
            block_size,
            |prompts, sp, s, v| self.generate_impl(prompts, sp, false, s, v),
        )
    }

    /// Reset the prefix cache, evicting all cached KV blocks.
    pub fn reset_prefix_cache(&mut self) -> Result<bool> {
        Ok(self.client.reset_prefix_cache()?)
    }

    /// Like [`generate`](Self::generate), but with a tqdm-style progress bar
    /// showing estimated input/output token throughput — mirrors Python's
    /// `LLM.generate(use_tqdm=True)`.
    pub fn generate_with_tqdm<P: Into<Prompt> + Clone>(
        &mut self,
        prompts: &[P],
        params: Option<SamplingParams>,
    ) -> Result<Vec<RequestOutput>> {
        self.generate_impl(prompts, params, true, false, false)
    }

    fn generate_impl<P: Into<Prompt> + Clone>(
        &mut self,
        prompts: &[P],
        params: Option<SamplingParams>,
        use_tqdm: bool,
        seal: bool,
        volatile: bool,
    ) -> Result<Vec<RequestOutput>> {
        let params = params.unwrap_or_else(|| self.generation_defaults.as_base());
        params
            .validate()
            .map_err(|e| anyhow::anyhow!("invalid sampling params: {e}"))?;

        let n = params.n.max(1) as usize;
        let detokenize = params.detokenize;

        // Convert all prompts to Prompt enum.
        let prompts: Vec<Prompt> = prompts.iter().map(|p| p.clone().into()).collect();

        // Tokenize text prompts; pass token ID prompts through directly.
        let prompt_data: Vec<(Vec<u32>, Option<scratchy_core_common::BlockAnnotations>)> = prompts
            .iter()
            .map(|p| match p {
                Prompt::Text(text) => Ok((self.tokenize_text(text)?, None)),
                Prompt::TokenIds(ids) => Ok((ids.clone(), None)),
                Prompt::TokenIdsWithAnnotations(ids, ann) => Ok((ids.clone(), Some(ann.clone()))),
            })
            .collect::<Result<Vec<_>>>()?;
        let prompt_token_ids: Vec<Vec<u32>> =
            prompt_data.iter().map(|(ids, _)| ids.clone()).collect();

        let total = prompt_token_ids.len() * n;
        let base_id = format!("llm-{}", uuid::Uuid::new_v4());

        // Submit all requests to the engine.
        let mut request_ids: Vec<String> = Vec::with_capacity(total);
        for (p_idx, prompt_ids) in prompt_token_ids.iter().enumerate() {
            for n_idx in 0..n {
                let request_id = if total == 1 {
                    base_id.clone()
                } else {
                    format!("{base_id}-{}", p_idx * n + n_idx)
                };

                let mut sp = params.clone();
                sp.seed = sp.seed.map(|s| s.wrapping_add(n_idx as u64));
                sp.seal = seal;
                sp.volatile = volatile;
                self.resolve_max_tokens(&mut sp, prompt_ids.len());

                self.client
                    .add_request(EngineCoreRequest {
                        request_id: request_id.clone(),
                        prompt_token_ids: Some(prompt_ids.clone()),
                        sampling_params: Some(sp),
                        arrival_time: SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_secs_f64(),
                        client_index: 0,
                        priority: 0,
                        cache_salt: None,
                        data_parallel_rank: None,
                        is_pooling: false,
                        mm_data: None,
                        block_annotations: prompt_data[p_idx].1.clone(),
                        seal,
                        volatile,
                    })
                    .map_err(|e| anyhow::anyhow!("add_request failed: {e}"))?;

                request_ids.push(request_id);
            }
        }

        // Sync step loop — mirrors Python's LLM._run_engine().
        // O(1) request-id -> slot lookup: avoids an O(num_requests) linear scan
        // per output per step (O(batch^2)/step) on the saturated hot path.
        let id_to_idx: std::collections::HashMap<&str, usize> = request_ids
            .iter()
            .enumerate()
            .map(|(i, id)| (id.as_str(), i))
            .collect();
        let mut generated_tokens: Vec<Vec<u32>> = vec![Vec::new(); total];
        let mut finish_reasons: Vec<Option<String>> = vec![None; total];

        // Progress bar — mirrors Python's tqdm in _run_engine().
        let pbar = if use_tqdm {
            let pb = indicatif::ProgressBar::new(total as u64);
            pb.set_style(
                indicatif::ProgressStyle::with_template(
                    "Processed prompts: {wide_bar:.cyan/blue} {pos}/{len} \
                     [{elapsed}<{eta}, {per_sec}, {msg}]",
                )
                .unwrap(),
            );
            pb.set_message("est. speed input: 0.00 toks/s, output: 0.00 toks/s");
            Some(pb)
        } else {
            None
        };
        let start = std::time::Instant::now();
        let mut total_in_toks: usize = 0;
        let mut total_out_toks: usize = 0;

        // Per-request timing: track first-token time and ITL.
        let mut first_token_time: Vec<Option<std::time::Instant>> = vec![None; total];
        let mut last_token_time: Vec<Option<std::time::Instant>> = vec![None; total];
        let mut itl_sum: Vec<f64> = vec![0.0; total];
        let mut itl_count: Vec<u32> = vec![0; total];

        while self.client.has_unfinished_requests() {
            let outcome = self
                .client
                .get_output()
                .map_err(|e| anyhow::anyhow!("engine step failed: {e}"))?;
            let outputs = match outcome {
                StepOutcome::Progressed { outputs, .. } => outputs,
                StepOutcome::Stalled => return Err(stalled_step("offline generate")),
                StepOutcome::Idle => break,
            };

            let mut newly_finished = 0usize;
            for output in &outputs.outputs {
                if let Some(&idx) = id_to_idx.get(output.request_id.as_str()) {
                    // Track TTFT / ITL timing.
                    if !output.new_token_ids.is_empty() {
                        let now = std::time::Instant::now();
                        if first_token_time[idx].is_none() {
                            first_token_time[idx] = Some(now);
                        } else if let Some(last) = last_token_time[idx] {
                            let itl = now.duration_since(last).as_secs_f64();
                            itl_sum[idx] += itl;
                            itl_count[idx] += 1;
                        }
                        last_token_time[idx] = Some(now);
                    }

                    generated_tokens[idx].extend_from_slice(&output.new_token_ids);
                    if let Some(ref reason) = output.finish_reason
                        && finish_reasons[idx].is_none()
                    {
                        finish_reasons[idx] = Some(reason.to_string());
                        newly_finished += 1;
                        if pbar.is_some() {
                            let p_idx = idx / n;
                            total_in_toks += prompt_token_ids[p_idx].len();
                            total_out_toks += generated_tokens[idx].len();
                        }
                    }
                }
            }

            if let Some(ref pb) = pbar
                && newly_finished > 0
            {
                let elapsed = start.elapsed().as_secs_f64().max(1e-9);
                let in_spd = total_in_toks as f64 / elapsed;
                let out_spd = total_out_toks as f64 / elapsed;
                pb.set_message(format!(
                    "est. speed input: {in_spd:.2} toks/s, output: {out_spd:.2} toks/s"
                ));
                pb.inc(newly_finished as u64);
            }
        }

        if let Some(pb) = pbar {
            pb.finish();
        }

        // Build RequestOutputs, grouping n completions per prompt.
        // Detokenize finished sequences (if requested).
        let mut results = Vec::with_capacity(prompts.len());
        for p_idx in 0..prompts.len() {
            let mut completion_outputs = Vec::with_capacity(n);
            for i in 0..n {
                let idx = p_idx * n + i;
                let text = if detokenize {
                    if let Some(tok) = &self.tokenizer {
                        let mut detok = IncrementalDetokenizer::new(
                            Arc::clone(tok),
                            &prompt_token_ids[p_idx],
                            params.stop.clone(),
                            params.min_tokens,
                            params.include_stop_str_in_output,
                            params.skip_special_tokens,
                        );
                        detok.update(&generated_tokens[idx], false);
                        detok.get_next_output_text(true, false)
                    } else {
                        String::new()
                    }
                } else {
                    String::new()
                };
                completion_outputs.push(CompletionOutput {
                    index: i as u32,
                    text,
                    token_ids: generated_tokens[idx].clone(),
                    finish_reason: finish_reasons[idx].clone(),
                });
            }

            let prompt_text = match &prompts[p_idx] {
                Prompt::Text(text) => Some(text.clone()),
                Prompt::TokenIds(_) | Prompt::TokenIdsWithAnnotations(_, _) => None,
            };

            // Compute TTFT and avg ITL for the first completion of this prompt.
            let first_idx = p_idx * n;
            let ttft_s =
                first_token_time[first_idx].map(|ft| ft.duration_since(start).as_secs_f64());
            let avg_itl_s = if itl_count[first_idx] > 0 {
                Some(itl_sum[first_idx] / itl_count[first_idx] as f64)
            } else {
                None
            };

            results.push(RequestOutput {
                request_id: request_ids[first_idx].clone(),
                prompt: prompt_text,
                prompt_token_ids: prompt_token_ids[p_idx].clone(),
                outputs: completion_outputs,
                finished: true,
                ttft_s,
                avg_itl_s,
            });
        }

        Ok(results)
    }

    // -----------------------------------------------------------------------
    // chat()
    // -----------------------------------------------------------------------

    /// Generate a chat completion from a list of messages.
    ///
    /// Applies the model's chat template to produce a prompt, then runs the
    /// same sync engine loop as [`generate()`](Self::generate).
    pub fn chat(
        &mut self,
        messages: &[ChatMessage],
        params: Option<SamplingParams>,
    ) -> Result<RequestOutput> {
        let tpl = self
            .chat_template
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("model does not have a chat template"))?;

        // Convert ChatMessages to serde_json::Value for the template engine.
        let msg_values: Vec<serde_json::Value> = messages
            .iter()
            .map(|m| {
                serde_json::json!({
                    "role": m.role,
                    "content": m.content,
                })
            })
            .collect();

        let prompt = tpl
            .apply(&msg_values, true, None)
            .map_err(|e| anyhow::anyhow!("chat template render failed: {e}"))?;

        let mut results = self.generate(&[prompt.as_str()], params)?;
        results
            .pop()
            .ok_or_else(|| anyhow::anyhow!("generate returned no results"))
    }

    /// Generate a single streaming chat turn: yields token strings as they
    /// are generated, then returns the full output.
    ///
    /// The callback is invoked with each new token text fragment. This enables
    /// print-as-you-go UX for the CLI chat command.
    pub fn chat_stream(
        &mut self,
        messages: &[ChatMessage],
        params: Option<SamplingParams>,
        mut on_token: impl FnMut(&str),
    ) -> Result<RequestOutput> {
        let tpl = self
            .chat_template
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("model does not have a chat template"))?;

        let msg_values: Vec<serde_json::Value> = messages
            .iter()
            .map(|m| {
                serde_json::json!({
                    "role": m.role,
                    "content": m.content,
                })
            })
            .collect();

        // TTFT ATTRIBUTION (SCRATCHY_TTFT_PHASE). `scr chat <model>` runs IN-PROCESS through here,
        // not through the HTTP server, so none of the server-side timers see it. The device side is
        // priced to the microsecond by SCRATCHY_SDSC_PHASE_TIME; this prices everything around it, so
        // render+tokenize+submit+to_first_token accounts for the whole of the CLI's reported TTFT.
        let ttft_phase = std::env::var_os("SCRATCHY_TTFT_PHASE").is_some();
        let _t_render = std::time::Instant::now();
        let prompt = tpl
            .apply(&msg_values, true, None)
            .map_err(|e| anyhow::anyhow!("chat template render failed: {e}"))?;
        let _d_render = _t_render.elapsed();

        let params = params.unwrap_or_else(|| self.generation_defaults.as_base());
        params
            .validate()
            .map_err(|e| anyhow::anyhow!("invalid sampling params: {e}"))?;

        // Tokenize the rendered prompt.
        let _t_tok = std::time::Instant::now();
        let prompt_token_ids = self.tokenize_text(&prompt)?;
        let _d_tok = _t_tok.elapsed();
        let mut sp = params.clone();
        self.resolve_max_tokens(&mut sp, prompt_token_ids.len());
        if std::env::var_os("VLLM_DEBUG_PARAMS").is_some() {
            eprintln!(
                "[debug-params] max_model_len={} prompt={} sp={sp:?}",
                self.max_model_len,
                prompt_token_ids.len()
            );
        }

        let request_id = format!("llm-chat-{}", uuid::Uuid::new_v4());
        let _t_submit = std::time::Instant::now();
        self.client
            .add_request(EngineCoreRequest {
                request_id: request_id.clone(),
                prompt_token_ids: Some(prompt_token_ids.clone()),
                sampling_params: Some(sp),
                arrival_time: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs_f64(),
                client_index: 0,
                priority: 0,
                cache_salt: None,
                data_parallel_rank: None,
                is_pooling: false,
                mm_data: None,
                block_annotations: None,
                seal: false,
                volatile: false,
            })
            .map_err(|e| anyhow::anyhow!("add_request failed: {e}"))?;
        let _d_submit = _t_submit.elapsed();
        let _t_step = std::time::Instant::now();

        // Step loop with incremental detokenization for streaming output.
        let mut all_token_ids: Vec<u32> = Vec::new();
        let mut finish_reason: Option<String> = None;
        let mut detok = self.tokenizer.as_ref().map(|tok| {
            IncrementalDetokenizer::new(
                Arc::clone(tok),
                &prompt_token_ids,
                params.stop.clone(),
                params.min_tokens,
                params.include_stop_str_in_output,
                params.skip_special_tokens,
            )
        });
        let mut full_text = String::new();

        // Step accounting for SCRATCHY_TTFT_PHASE: `steps` counts get_output() round trips before the
        // first token and `_d_step0` times the first one. Together they separate "one slow step" (the
        // forward plus whatever the scheduler does around it) from "many empty steps" (the loop
        // spinning while the request waits to be admitted) — two very different bugs, same symptom.
        let mut steps = 0usize;
        let mut _d_step0 = std::time::Duration::ZERO;
        while self.client.has_unfinished_requests() {
            let _t_step_i = std::time::Instant::now();
            let outcome = self
                .client
                .get_output()
                .map_err(|e| anyhow::anyhow!("engine step failed: {e}"))?;
            let outputs = match outcome {
                StepOutcome::Progressed { outputs, .. } => outputs,
                StepOutcome::Stalled => return Err(stalled_step("streaming chat/completion")),
                StepOutcome::Idle => break,
            };
            if steps == 0 {
                _d_step0 = _t_step_i.elapsed();
            }
            steps += 1;

            for output in &outputs.outputs {
                if output.request_id != request_id {
                    continue;
                }
                if ttft_phase && all_token_ids.is_empty() && !output.new_token_ids.is_empty() {
                    // `to_first_token` is queue + prefill forward(s) + sample; the SDSC phase lines
                    // sit inside it, so the difference is the engine/scheduler overhead around them.
                    eprintln!(
                        "[ttft-host] render={:.2}  tokenize={:.2}  submit={:.2}  to_first_token={:.2}  \
                         TOTAL={:.2} ms  ({} prompt tokens)",
                        _d_render.as_secs_f64() * 1e3,
                        _d_tok.as_secs_f64() * 1e3,
                        _d_submit.as_secs_f64() * 1e3,
                        _t_step.elapsed().as_secs_f64() * 1e3,
                        _t_render.elapsed().as_secs_f64() * 1e3,
                        prompt_token_ids.len(),
                    );
                    eprintln!(
                        "[ttft-host]   steps_to_first_token={steps}  first_step={:.2} ms",
                        _d_step0.as_secs_f64() * 1e3,
                    );
                }
                all_token_ids.extend_from_slice(&output.new_token_ids);
                if let Some(ref reason) = output.finish_reason {
                    finish_reason = Some(reason.to_string());
                }

                // Incremental detokenize and stream.
                // Match Python logic (output_processor.py line 628):
                //   stop_string = req_state.detokenizer.update(
                //       new_token_ids, finish_reason == FinishReason.STOP
                //   )
                if let Some(ref mut d) = detok {
                    use scratchy_core_common::engine_io::FinishReason;
                    let stop_terminated = output.finish_reason == Some(FinishReason::Stop);
                    d.update(&output.new_token_ids, stop_terminated);
                    let new_text = d.get_next_output_text(false, true);
                    if !new_text.is_empty() {
                        on_token(&new_text);
                        full_text.push_str(&new_text);
                    }
                }
            }
        }

        // Flush any remaining detokenizer state.
        if let Some(ref mut d) = detok {
            let remaining = d.get_next_output_text(true, true);
            if !remaining.is_empty() {
                on_token(&remaining);
                full_text.push_str(&remaining);
            }
        }

        Ok(RequestOutput {
            request_id,
            prompt: Some(prompt),
            prompt_token_ids,
            outputs: vec![CompletionOutput {
                index: 0,
                text: full_text,
                token_ids: all_token_ids,
                finish_reason,
            }],
            finished: true,
            ttft_s: None,
            avg_itl_s: None,
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol;

    fn build_batched_completion_request(
        prompts: &[Prompt],
        params: &SamplingParams,
        model: &str,
    ) -> protocol::CompletionRequest {
        let prompt = if prompts.len() == 1 {
            match &prompts[0] {
                Prompt::Text(text) => protocol::CompletionPrompt::Single(text.clone()),
                Prompt::TokenIds(ids) | Prompt::TokenIdsWithAnnotations(ids, _) => {
                    protocol::CompletionPrompt::TokenIds(ids.clone())
                }
            }
        } else {
            let all_token_ids = prompts.iter().all(|p| {
                matches!(
                    p,
                    Prompt::TokenIds(_) | Prompt::TokenIdsWithAnnotations(_, _)
                )
            });
            if all_token_ids {
                let seqs: Vec<Vec<u32>> = prompts
                    .iter()
                    .map(|p| match p {
                        Prompt::TokenIds(ids) | Prompt::TokenIdsWithAnnotations(ids, _) => {
                            ids.clone()
                        }
                        Prompt::Text(_) => unreachable!(),
                    })
                    .collect();
                protocol::CompletionPrompt::MultipleTokenIds(seqs)
            } else {
                let texts: Vec<String> = prompts
                    .iter()
                    .map(|p| match p {
                        Prompt::Text(text) => text.clone(),
                        Prompt::TokenIds(ids) | Prompt::TokenIdsWithAnnotations(ids, _) => {
                            format!("<token_ids:{}>", ids.len())
                        }
                    })
                    .collect();
                protocol::CompletionPrompt::Multiple(texts)
            }
        };

        let stop = if params.stop.is_empty() {
            None
        } else {
            Some(protocol::StopCondition::Multiple(params.stop.clone()))
        };

        protocol::CompletionRequest {
            model: Some(model.to_string()),
            prompt: Some(prompt),
            echo: false,
            temperature: Some(params.temperature),
            top_p: Some(params.top_p),
            n: params.n,
            max_tokens: params.max_tokens,
            stream: false,
            stream_options: None,
            stop,
            frequency_penalty: Some(params.frequency_penalty),
            presence_penalty: Some(params.presence_penalty),
            logit_bias: None,
            logprobs: params.logprobs.map(|v| v.max(0) as u32),
            prompt_logprobs: params.prompt_logprobs.map(|v| v.max(0) as u32),
            suffix: None,
            seed: params.seed.map(|s| s as i64),
            user: None,
            top_k: Some(params.top_k),
            min_p: Some(params.min_p),
            repetition_penalty: Some(params.repetition_penalty),
            min_tokens: params.min_tokens,
            stop_token_ids: params.stop_token_ids.clone(),
            include_stop_str_in_output: params.include_stop_str_in_output,
            ignore_eos: params.ignore_eos,
            skip_special_tokens: params.skip_special_tokens,
            add_special_tokens: true,
            priority: 0,
            cache_salt: None,
            request_id: None,
            guided_regex: None,
            guided_grammar: None,
            allowed_token_ids: params.allowed_token_ids.clone(),
            bad_words: None,
            truncate_prompt_tokens: None,
            block_annotations: None,
            seal: false,
            volatile: false,
        }
    }

    #[test]
    fn test_builder_defaults() {
        let builder = LLMBuilder::new("test-model");
        assert_eq!(builder.config.model, "test-model");
        assert_eq!(builder.config.device, "auto");
        assert_eq!(builder.config.dtype, "auto");
        // Unset by default: the BACKEND decides the width (resolve_max_num_seqs), not a CLI literal.
        assert_eq!(builder.config.max_num_seqs, None);
        assert_eq!(builder.config.block_size, 16);
        assert!(builder.config.max_model_len.is_none());
    }

    #[test]
    fn test_builder_chaining() {
        let builder = LLMBuilder::new("test-model")
            .device("cpu")
            .dtype("float16")
            .max_model_len(2048)
            .max_num_seqs(32)
            .block_size(8)
            .gpu_memory_utilization(0.5);

        assert_eq!(builder.config.device, "cpu");
        assert_eq!(builder.config.dtype, "float16");
        assert_eq!(builder.config.max_model_len, Some(2048));
        assert_eq!(builder.config.max_num_seqs, Some(32));
        assert_eq!(builder.config.block_size, 8);
        assert!((builder.config.gpu_memory_utilization - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_chat_message_constructors() {
        let sys = ChatMessage::system("You are helpful.");
        assert_eq!(sys.role, "system");
        assert_eq!(sys.content, "You are helpful.");

        let user = ChatMessage::user("Hello");
        assert_eq!(user.role, "user");

        let asst = ChatMessage::assistant("Hi there!");
        assert_eq!(asst.role, "assistant");
    }

    #[test]
    fn test_completion_output_debug() {
        let output = CompletionOutput {
            index: 0,
            text: "hello".to_string(),
            token_ids: vec![1, 2, 3],
            finish_reason: Some("stop".to_string()),
        };
        // Ensure Debug is derived and doesn't panic.
        let _ = format!("{output:?}");
    }

    #[test]
    fn test_request_output_debug() {
        let output = RequestOutput {
            request_id: "test-id".to_string(),
            prompt: Some("Hello".to_string()),
            prompt_token_ids: vec![1, 2],
            outputs: vec![CompletionOutput {
                index: 0,
                text: "world".to_string(),
                token_ids: vec![3],
                finish_reason: None,
            }],
            finished: true,
            ttft_s: None,
            avg_itl_s: None,
        };
        let _ = format!("{output:?}");
    }

    #[test]
    fn test_build_batched_completion_request_single_text() {
        let params = SamplingParams {
            temperature: 0.7,
            max_tokens: Some(100),
            stop: vec!["END".to_string()],
            ..SamplingParams::default()
        };
        let prompts = vec![Prompt::Text("Hello".to_string())];
        let req = build_batched_completion_request(&prompts, &params, "test-model");
        assert_eq!(req.model, Some("test-model".to_string()));
        assert_eq!(req.temperature, Some(0.7));
        assert_eq!(req.max_tokens, Some(100));
        assert!(!req.stream);
        assert!(req.stop.is_some());
        assert!(matches!(
            req.prompt,
            Some(protocol::CompletionPrompt::Single(ref s)) if s == "Hello"
        ));
    }

    #[test]
    fn test_build_batched_completion_request_multiple_token_ids() {
        let params = SamplingParams {
            temperature: 0.7,
            max_tokens: Some(100),
            ignore_eos: true,
            ..SamplingParams::default()
        };
        let prompts = vec![
            Prompt::TokenIds(vec![10, 20, 30]),
            Prompt::TokenIds(vec![40, 50]),
        ];
        let req = build_batched_completion_request(&prompts, &params, "test-model");
        assert_eq!(req.model, Some("test-model".to_string()));
        assert!(matches!(
            req.prompt,
            Some(protocol::CompletionPrompt::MultipleTokenIds(ref seqs))
                if seqs.len() == 2 && seqs[0] == [10, 20, 30] && seqs[1] == [40, 50]
        ));
    }

    #[test]
    fn test_build_batched_completion_request_single_token_ids() {
        let params = SamplingParams::default();
        let prompts = vec![Prompt::TokenIds(vec![1, 2, 3])];
        let req = build_batched_completion_request(&prompts, &params, "m");
        assert!(matches!(
            req.prompt,
            Some(protocol::CompletionPrompt::TokenIds(ref ids)) if ids == &[1, 2, 3]
        ));
    }

    #[test]
    fn test_build_batched_completion_request_multiple_text() {
        let params = SamplingParams::default();
        let prompts = vec![Prompt::Text("Hello".into()), Prompt::Text("World".into())];
        let req = build_batched_completion_request(&prompts, &params, "m");
        assert!(matches!(
            req.prompt,
            Some(protocol::CompletionPrompt::Multiple(ref texts))
                if texts.len() == 2 && texts[0] == "Hello" && texts[1] == "World"
        ));
    }

    #[test]
    fn test_prompt_from_str() {
        let p: Prompt = "hello".into();
        assert!(matches!(p, Prompt::Text(s) if s == "hello"));
    }

    #[test]
    fn test_prompt_from_string() {
        let p: Prompt = String::from("world").into();
        assert!(matches!(p, Prompt::Text(s) if s == "world"));
    }

    #[test]
    fn test_prompt_from_token_ids() {
        let p: Prompt = vec![1u32, 2, 3].into();
        assert!(matches!(p, Prompt::TokenIds(ids) if ids == [1, 2, 3]));
    }
}
