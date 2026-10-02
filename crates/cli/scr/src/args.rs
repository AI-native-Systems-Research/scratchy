// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! CLI argument definitions using clap derive.

use clap::{Parser, Subcommand};

#[cfg(feature = "bench")]
pub use scratchy_bench::BenchCommand;
#[cfg(feature = "bench")]
#[cfg(test)]
pub use scratchy_bench::BenchCommands;

/// vLLM — High-throughput LLM serving engine (Rust)
#[derive(Parser, Debug)]
#[command(name = "scr", version, about, long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Start the OpenAI-compatible API server.
    #[cfg(feature = "serve")]
    Serve(Box<ServeArgs>),
    /// Launch an external coding agent (e.g. Claude Code) against a served model.
    #[cfg(feature = "claude")]
    Launch(LaunchArgs),
    /// Run benchmarks (latency, serving, throughput).
    #[cfg(feature = "bench")]
    Bench(BenchCommand),
    /// Process a batch of OpenAI-compatible requests offline.
    #[cfg(any(feature = "chat", feature = "serve"))]
    Batch(BatchArgs),
    /// Process a batch of OpenAI-compatible requests offline (alias for `batch`).
    #[cfg(any(feature = "chat", feature = "serve"))]
    #[command(name = "run-batch")]
    RunBatch(BatchArgs),
    /// Chat with a model (locally in-process, or `--url` against a server).
    Chat(ChatArgs),
    /// Collect and print environment information for bug reports.
    CollectEnv(CollectEnvArgs),
    /// Generate text completions via the running API server.
    Complete(CompleteArgs),
    /// Manage models: list/pull/rm/convert/cache, inspect compiled-in
    /// backbones.
    Model(ModelCommand),
    /// Run a Spyre bundle this binary did not bake — a `dxp_standalone` output
    /// directory plus the producer's launch map. Every other launch path here needs
    /// the bundle baked into the binary at build time, which leaves an externally
    /// produced one unrunnable.
    #[cfg(feature = "spyre-hw")]
    Bundle(BundleCommand),
    /// Print (or install) a shell completion script for bash or zsh, enabling
    /// `scr chat <TAB>` / `scr serve <TAB>` to complete HuggingFace model ids
    /// this binary was compiled to run. Either pipe the output yourself (e.g.
    /// `scr completions zsh > ~/.zsh/completions/_scr`) or pass `--install` to
    /// have `scr` write the script and wire up the shell rc file itself.
    Completions(CompletionsArgs),
}

/// Arguments for `scr completions`.
#[derive(Parser, Debug)]
pub struct CompletionsArgs {
    /// Shell to generate a completion script for.
    #[arg(value_enum)]
    pub shell: CompletionShell,

    /// Write the script to the standard per-shell location and wire up the
    /// shell rc file (`~/.bashrc` / `~/.zshrc`) instead of printing the script
    /// to stdout. Idempotent — safe to run again after a rebuild.
    #[arg(long)]
    pub install: bool,
}

#[derive(Clone, Copy, Debug, clap::ValueEnum)]
pub enum CompletionShell {
    Bash,
    Zsh,
}

/// `scr launch <agent>` — serve a model locally and start an external coding
/// agent pointed at it (mirrors `ollama launch claude`).
#[derive(Parser, Debug)]
pub struct LaunchArgs {
    #[command(subcommand)]
    pub command: LaunchSubcommand,
}

#[derive(Subcommand, Debug)]
pub enum LaunchSubcommand {
    /// Launch Claude Code against a locally-served model via the Anthropic
    /// Messages API (`/v1/messages`).
    Claude(LaunchClaudeArgs),
}

/// Arguments for `scr launch claude`.
#[derive(Parser, Debug)]
#[command(override_usage = "scr launch claude [MODEL] [OPTIONS] [-- CLAUDE_ARGS...]")]
pub struct LaunchClaudeArgs {
    /// Model to serve: local path or HuggingFace model ID. Takes precedence
    /// over --model.
    pub model_tag: Option<String>,

    /// Model to serve (alternative to the positional argument).
    #[arg(short = 'm', long, env = "VLLM_MODEL")]
    pub model: Option<String>,

    /// Device: "cpu", "cuda:N", "metal", or "auto" (auto-detect best GPU).
    #[arg(long, default_value = "auto")]
    pub device: String,

    /// Weight dtype: "auto", "float16", "bfloat16", "float32".
    #[arg(long, default_value = "auto")]
    pub dtype: String,

    /// Tool-call parser ("hermes", "llama3_json", "kimi_k2", "mistral",
    /// "jamba", "granite", "gemma4"). Auto-detected from the model by default;
    /// pass this only to override the guess or to enable tool use for an
    /// unrecognized model family.
    #[arg(long)]
    pub tool_call_parser: Option<String>,

    /// Maximum model context length (overrides config.json).
    #[arg(long)]
    pub max_model_len: Option<usize>,

    /// Path to the `claude` executable.
    #[arg(long, default_value = "claude")]
    pub claude_bin: String,

    /// Auth token handed to Claude Code. Accepted but not validated by the
    /// server (mirrors `ollama launch`'s ignored token).
    #[arg(long, default_value = "scratchy")]
    pub auth_token: String,

    /// Reuse an already-running scratchy server at this base URL (e.g.
    /// "http://127.0.0.1:8000") instead of spawning one.
    #[arg(long)]
    pub server_url: Option<String>,

    /// Resume a previous Claude Code session by id. First-class equivalent of
    /// `-- --resume <id>`: forwards `--resume <id>` to claude AND keys this
    /// launch's per-session KV store by `<id>`, so the session's cached prefix is
    /// reused. Get the id from the resume hint a prior launch printed on exit.
    #[arg(long)]
    pub resume: Option<String>,

    /// Pin a specific Claude Code session id (first-class `--session-id <id>`):
    /// forwards it to claude AND keys the per-session KV store by it. Without
    /// this (or `--resume`), launch mints one and prints the resume command on
    /// exit.
    #[arg(long)]
    pub session_id: Option<String>,

    /// Extra arguments passed through to `claude` (after `--`).
    #[arg(last = true)]
    pub claude_args: Vec<String>,
}

/// `scr model <subcommand>`.
#[derive(Parser, Debug)]
pub struct ModelCommand {
    #[command(subcommand)]
    pub command: ModelSubcommand,
}

#[derive(Subcommand, Debug)]
pub enum ModelSubcommand {
    /// List cached models.
    Ls(ListArgs),
    /// List cached models.
    List(ListArgs),
    /// Download a model from HuggingFace Hub without starting the server.
    Pull(PullArgs),
    /// Remove a cached model.
    Rm(RmArgs),
    /// Convert model weights between formats (stub).
    Convert(ConvertArgs),
    /// Inspect and clean on-disk caches (HuggingFace Hub + scratchy weights).
    Cache(CacheCommand),
    /// Print the per-bucket backbone instruction list for compiled
    /// scratchy variants. Optional positional filters AND-substring
    /// match against `<arch>/<variant_stem>` — e.g.
    /// `scr model info llama 3.2 awq`.
    #[cfg(any(feature = "cuda", feature = "metal"))]
    Info(ModelInfoArgs),
    /// Print compiled-in HuggingFace model ids for shell tab completion
    /// (one per line), optionally filtered by prefix. Sourced from
    /// `model::compiled_hf_registry()` — resolved against huggingface.co
    /// at BUILD time (see `crates/models/arch/hf_registry_build.rs`), not
    /// a runtime network call or the local hf-hub cache. Hidden — invoked
    /// by the completion scripts from `scr completions`, not meant to be
    /// typed directly.
    #[cfg(feature = "model")]
    #[command(hide = true)]
    Names(NamesArgs),
}

/// Arguments for the hidden `scr model names` completion helper.
#[cfg(feature = "model")]
#[derive(Parser, Debug)]
pub struct NamesArgs {
    /// Prefix the shell is currently completing (empty = list everything).
    #[arg(default_value = "")]
    pub prefix: String,

    /// Which candidate set to draw from. Different arguments want different
    /// sources: a model argument wants what this build can actually run,
    /// `scr model rm` wants only what is actually cached, `scr model info`
    /// wants config stems.
    #[arg(long, value_enum, default_value_t = NamesSource::Compiled)]
    pub source: NamesSource,
}

/// Candidate sets for `scr model names`.
#[cfg(feature = "model")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum NamesSource {
    /// The build-time HF registry: repo ids whose arch, shape and quantization
    /// all match something this binary compiled. The default, and the only
    /// source valid for a model argument.
    ///
    /// Empty when the build resolved no registry — see the note on
    /// [`Cached`](Self::Cached) for why an empty list beats a plausible-looking
    /// wrong one.
    Compiled,
    /// Locally cached repos — every `org/name` in the hf-hub cache, which is
    /// whatever has ever been pulled on this machine.
    ///
    /// **Not scoped to this build**, which is why it is not part of
    /// [`Compiled`](Self::Compiled) and never offered for a model argument: a
    /// cache holds models of other architectures, other sizes and other
    /// quantizations, none of which this binary can necessarily load. Mixing it
    /// in made completion look broken — it offered `modernbert-embed-base` (not
    /// even a decoder) from a llama-only build, and produced the same list
    /// regardless of `-Fmodel` scope whenever the registry was empty. Used for
    /// `scr model rm`, whose argument "must match the model ID shown by
    /// `scr model ls`".
    Cached,
    /// Compiled config stems (`llama-3.2-1b`) — for `scr model info` filters.
    /// Never valid as a model argument.
    Stems,
}

#[cfg(any(feature = "cuda", feature = "metal"))]
#[derive(Parser, Debug)]
pub struct ModelInfoArgs {
    /// Color/style output. `auto` uses ANSI when stdout is a tty
    /// and `NO_COLOR` is unset; `always` forces it on (useful when
    /// piping into `less -R`); `never` disables.
    #[arg(long, value_enum, default_value_t = ColorWhen::Auto)]
    pub color: ColorWhen,
    /// Run cuBLAS-pick analysis instead of the per-bucket backbone
    /// dump: for every `Cublas` Gemm pick, classify by margin vs.
    /// the best non-cuBLAS standalone-GEMM kernel and identify
    /// fusion-gap reasons (Gemm→Add / Norm→Gemm / Gemm→ScalarMul /
    /// lm_head). Uses the bundled cost CSV for exact-row lookups.
    /// cuda-only — keyed off `scratchy_target_cuda::targets` profiles.
    #[cfg(feature = "cuda")]
    #[arg(short = 'c', long = "cublas-analysis")]
    pub cublas_analysis: bool,
    /// With `-c/--cublas-analysis`, also print a per-arch breakdown.
    #[cfg(feature = "cuda")]
    #[arg(long, requires = "cublas_analysis")]
    pub per_arch: bool,
    /// Substring filters. A variant is shown when its
    /// `<arch>/<variant_stem>` contains every filter (case-
    /// insensitive). Empty = show every compiled variant.
    pub filters: Vec<String>,
}

#[cfg(any(feature = "cuda", feature = "metal"))]
#[derive(clap::ValueEnum, Clone, Copy, Debug, Default)]
pub enum ColorWhen {
    #[default]
    Auto,
    Always,
    Never,
}

/// Arguments for the `serve` subcommand.
#[derive(Parser, Debug)]
#[command(override_usage = "scr serve [MODEL] [OPTIONS]")]
pub struct ServeArgs {
    /// Model to serve: local path or HuggingFace model ID
    /// (e.g. "meta-llama/Llama-3.2-1B"). Takes precedence over --model.
    pub model_tag: Option<String>,

    /// Path to a local model directory, or HuggingFace model ID.
    /// Can also be specified as a positional argument.
    #[arg(short = 'm', long, env = "VLLM_MODEL")]
    pub model: Option<String>,

    /// Host address to bind.
    #[arg(long, default_value = "0.0.0.0")]
    pub host: String,

    /// Port to listen on.
    #[arg(long, default_value_t = 8000)]
    pub port: u16,

    /// Device: "cpu", "cuda:N", "metal", or "auto" (auto-detect best GPU).
    #[arg(long, default_value = "auto")]
    pub device: String,

    /// Weight dtype: "auto", "float16", "bfloat16", "float32".
    /// "auto" reads torch_dtype from config.json (default).
    #[arg(long, default_value = "auto")]
    pub dtype: String,

    /// Maximum model context length (overrides config.json).
    #[arg(long)]
    pub max_model_len: Option<usize>,

    /// Maximum number of concurrent sequences. Unset lets the BACKEND choose:
    /// its own batched-decode width where that is fixed at build time (Spyre's
    /// baked rung ladder), else a device-appropriate default. An explicit value
    /// is honoured, capped to what the backend can actually batch.
    #[arg(long)]
    pub max_num_seqs: Option<usize>,

    /// Maximum number of tokens processed in a single scheduler iteration.
    #[arg(long)]
    pub max_num_batched_tokens: Option<usize>,

    /// HuggingFace token for gated models.
    #[arg(long, env = "HF_TOKEN")]
    pub hf_token: Option<String>,

    /// Log level: "trace", "debug", "info", "warn", "error".
    #[arg(long, default_value = "info")]
    pub log_level: String,

    /// Enable /metrics Prometheus endpoint.
    #[arg(long)]
    pub enable_metrics: bool,

    /// KV cache block size in tokens.
    #[arg(long, default_value_t = 16)]
    pub block_size: usize,

    /// Fraction of GPU memory to use for KV cache (0.0–1.0).
    #[arg(long, default_value_t = 0.9, env = "VLLM_GPU_MEMORY_UTILIZATION")]
    pub gpu_memory_utilization: f64,

    /// Specific GGUF filename to download from a HuggingFace repo.
    /// Example: --gguf-file llama-2-7b-chat.Q4_K_M.gguf
    #[arg(long)]
    pub gguf_file: Option<String>,

    /// Tool call parser to use (e.g. "hermes", "llama3_json", "gemma4").
    /// Enables structured tool call extraction from model output.
    #[arg(long)]
    pub tool_call_parser: Option<String>,

    /// Reasoning parser to use (e.g. "deepseek_r1", "qwen3").
    /// Extracts <think>...</think> blocks into a separate reasoning_content field.
    #[arg(long)]
    pub reasoning_parser: Option<String>,

    /// Default extra kwargs for the chat template, as JSON.
    /// Merged with per-request chat_template_kwargs (request overrides).
    /// Example: '{"enable_thinking": false}'
    #[arg(long, value_parser = parse_json_map)]
    pub default_chat_template_kwargs: Option<std::collections::HashMap<String, serde_json::Value>>,

    /// Chat template override. Accepts either an inline Jinja string
    /// or a path to a `tokenizer_config.json` / `.jinja` file. Used
    /// for GGUFs whose metadata lacks `tokenizer.chat_template`.
    #[arg(long)]
    pub chat_template: Option<String>,

    /// Enable automatic tool choice (model decides when to call tools).
    #[arg(long)]
    pub enable_auto_tool_choice: bool,

    /// Path to SSL/TLS private key file (PEM format).
    #[arg(long)]
    pub ssl_keyfile: Option<String>,

    /// Path to SSL/TLS certificate file (PEM format).
    #[arg(long)]
    pub ssl_certfile: Option<String>,

    /// Path to CA certificates file for client certificate verification (PEM).
    #[arg(long)]
    pub ssl_ca_certs: Option<String>,

    /// Pooling strategy for /v1/embeddings: "auto", "last", "cls", "mean".
    /// "auto" detects from 1_Pooling/config.json, defaults to "last".
    #[arg(long, default_value = "auto")]
    pub pooling_strategy: String,

    /// Speculative decoding model.
    ///
    /// `ngram` selects the n-gram proposer (matches in the request's own
    /// token history). Any other value is treated as a draft-model local
    /// path or HuggingFace repo ID; the draft-model proposer is not wired
    /// yet and the engine will refuse to start until it is.
    #[arg(long)]
    pub speculative_model: Option<String>,

    /// Number of speculative tokens to propose per step (default: 2).
    /// Only used when --speculative-model is set.
    ///
    /// K=2 is the empirical sweet spot for the realistic draft-model
    /// regime (target much larger than draft, e.g. Llama-3.1-8B
    /// target + Llama-3.2-1B draft on Apple Silicon). 5-run distribution
    /// at 8B+1B, M4: baseline 45 ms TPOT → K=1 34, K=2 32, K=4 39,
    /// K=6 53 (worse than baseline; chain cost overwhelms amortization
    /// and acceptance drops past K=2). Raise this only after measuring
    /// on the target+draft pair you care about; on smaller targets
    /// (e.g. 3B+1B) spec decode loses at any K and the right answer
    /// is no `--speculative-model`.
    #[arg(long, default_value_t = 2)]
    pub num_speculative_tokens: usize,

    /// Maximum n-gram size for prompt lookup (default: 4).
    /// Only used when --speculative-model ngram.
    #[arg(long, default_value_t = 4)]
    pub ngram_prompt_lookup_max: usize,

    /// Minimum n-gram size for prompt lookup (default: 1).
    /// Only used when --speculative-model ngram.
    #[arg(long, default_value_t = 1)]
    pub ngram_prompt_lookup_min: usize,

    /// Optional dtype override for the draft model's weights
    /// ("auto", "float16", "bfloat16", "float32"). Mirrors Python's
    /// `--speculative-config.draft_model_dtype`. Ignored unless
    /// --speculative-model points at a draft model.
    #[arg(long)]
    pub draft_model_dtype: Option<String>,

    /// LoRA adapter to load. Path to a local directory containing
    /// adapter_config.json and adapter_model.safetensors, or a
    /// HuggingFace repo ID.
    #[arg(long)]
    pub lora_adapter: Option<String>,

    /// Number of GPUs for tensor parallelism (default: 1).
    /// Splits model weights across N GPUs using NCCL all-reduce.
    #[arg(long, default_value_t = 1)]
    pub tensor_parallel_size: usize,

    /// Number of GPU stages for pipeline parallelism (default: 1).
    /// Splits model layers across N GPU stages using NCCL P2P send/recv.
    /// Total GPUs used = tensor_parallel_size * pipeline_parallel_size.
    #[arg(long, default_value_t = 1)]
    pub pipeline_parallel_size: usize,

    /// Number of nodes for multi-node tensor parallelism (default: 1).
    /// When > 1, GPUs are split across nodes using TCP rendezvous for NCCL.
    #[arg(long, default_value_t = 1)]
    pub num_nodes: usize,

    /// This node's rank in the multi-node setup (0 = master, default: 0).
    #[arg(long, default_value_t = 0)]
    pub node_rank: usize,

    /// Master node address for multi-node NCCL rendezvous (default: localhost).
    #[arg(long, default_value = "localhost")]
    pub master_addr: String,

    /// Master port for multi-node NCCL rendezvous (default: 29500).
    #[arg(long, default_value_t = 29500)]
    pub master_port: u16,

    /// Disable async scheduling (overlap of GPU execution and CPU scheduling).
    /// By default, async scheduling is enabled for better throughput.
    #[arg(long)]
    pub disable_async_scheduling: bool,

    /// Runner type: "generate" (default) or "pooling".
    /// In pooling mode, embedding requests go through the scheduler and
    /// generation endpoints (chat, completions) are rejected.
    #[arg(long, default_value = "generate")]
    pub runner: String,

    /// Comma-separated CUDA graph capture batch sizes.
    /// CUDA graphs accelerate decode steps by replaying a captured kernel
    /// sequence in a single driver call.
    #[arg(long, default_value = "auto")]
    pub cuda_graph_sizes: String,

    /// Disable CUDA graphs and run all steps eagerly.
    /// Equivalent to Python vLLM's --enforce-eager flag.
    /// Default: false (CUDA graphs enabled on CUDA devices).
    #[arg(long)]
    pub enforce_eager: bool,

    /// Distributed executor backend: "auto" (default) or "external_launcher".
    /// With "external_launcher", the job launcher (torchrun, mpirun, SLURM)
    /// spawns N processes. Each reads RANK, LOCAL_RANK, WORLD_SIZE,
    /// MASTER_ADDR, MASTER_PORT from env and runs its own engine instance.
    #[arg(long, default_value = "auto")]
    pub distributed_executor_backend: String,

    /// CUDA graph mode: controls piecewise vs monolithic graph capture.
    /// Options: "auto", "none", "full", "piecewise", "full-and-piecewise", "full-decode-only".
    /// - "auto": Full for SM < 90 (Ampere/Ada), FullAndPiecewise for SM >= 90 (Hopper+)
    /// - "none": No CUDA graphs (same as --enforce-eager)
    ///
    /// Default: "auto".
    #[arg(long, default_value = "auto")]
    pub cuda_graph_mode: String,

    /// Disable prefix caching (KV cache reuse for shared prompt prefixes).
    /// By default, prefix caching is enabled.
    #[arg(long)]
    pub no_prefix_caching: bool,

    /// Benchmark cublasLt algorithms during warmup to find faster GEMM kernels.
    /// Adds a few seconds to startup. Mainly benefits compute-bound prefill GEMMs.
    #[arg(long)]
    pub cublas_autotune: bool,

    /// KV cache data type: "auto" (use model dtype) or "fp8_e4m3" (FP8).
    /// FP8 halves KV cache memory, doubling capacity.
    #[arg(long, default_value = "auto")]
    pub kv_cache_dtype: String,

    /// Compute KV scales dynamically from the first forward pass.
    /// Only used with --kv-cache-dtype fp8_e4m3.
    #[arg(long)]
    pub calculate_kv_scales: bool,

    /// Target URL for OpenTelemetry traces (OTLP gRPC endpoint).
    /// Example: http://localhost:4317
    /// Requires building with --features otel.
    #[arg(long, env = "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")]
    pub otlp_traces_endpoint: Option<String>,
}

impl ServeArgs {
    /// Resolve the effective model path/ID.
    ///
    /// Priority: positional `model_tag` > `--model` flag > VLLM_MODEL env var.
    /// Returns an error if none is specified.
    #[cfg_attr(not(feature = "serve"), allow(dead_code))]
    pub fn resolved_model(&self) -> Result<String, String> {
        if let Some(ref tag) = self.model_tag {
            Ok(tag.clone())
        } else if let Some(ref m) = self.model {
            Ok(m.clone())
        } else {
            Err("model is required: provide as positional arg or --model flag".to_string())
        }
    }
}

/// Arguments for the `chat` subcommand.
///
/// Supports two modes:
/// - **In-process** (default when `--model` is given): loads the model locally
///   and runs inference directly — no server needed.
/// - **Remote** (when `--url` is given without `--model`): connects to a
///   running vLLM server's OpenAI-compatible API.
#[derive(Parser, Debug)]
#[command(override_usage = "scr chat [MODEL] [OPTIONS]")]
pub struct ChatArgs {
    /// Model to load in-process: local path or HuggingFace model ID.
    /// When set, runs inference locally without needing a running server.
    pub model_tag: Option<String>,

    /// Model: local path or HuggingFace model ID (alternative to positional arg).
    #[arg(short = 'm', long, env = "VLLM_MODEL")]
    pub model: Option<String>,

    /// URL of a running OpenAI-compatible API server (remote mode).
    /// Used only when no --model is specified.
    #[arg(long, default_value = "http://localhost:8000/v1")]
    pub url: String,

    /// Model name to request from the remote server.
    #[arg(long)]
    pub model_name: Option<String>,

    /// API key for remote server authentication.
    #[arg(long, env = "OPENAI_API_KEY")]
    pub api_key: Option<String>,

    /// System prompt to prepend to the conversation.
    #[arg(long)]
    pub system_prompt: Option<String>,

    /// Send a single message and exit (non-interactive mode).
    #[arg(short = 'q', long, value_name = "MESSAGE")]
    pub quick: Option<String>,

    /// Send prompt(s) and exit. Multiple values become separate turns in
    /// a multi-turn conversation (model responds to each in order).
    #[arg(short = 'p', long, value_name = "PROMPT", num_args = 1..)]
    pub prompt: Vec<String>,

    /// Print performance metrics after generation: startup time, TTFT,
    /// inter-token latency (ITL), and tokens/sec. Best used with --prompt.
    #[arg(long)]
    pub bench: bool,

    /// Device: "cpu", "cuda:N", "metal", or "auto" (auto-detect best GPU).
    #[arg(long, default_value = "auto")]
    pub device: String,

    /// Weight dtype: "auto", "float16", "bfloat16", "float32".
    #[arg(long, default_value = "auto")]
    pub dtype: String,

    /// HuggingFace token for gated models.
    #[arg(long, env = "HF_TOKEN")]
    pub hf_token: Option<String>,

    /// Specific GGUF filename to download from a HuggingFace repo.
    #[arg(long)]
    pub gguf_file: Option<String>,

    /// Maximum model context length (overrides config.json).
    #[arg(long)]
    pub max_model_len: Option<usize>,

    /// Maximum number of tokens to generate per response.
    #[arg(long)]
    pub max_tokens: Option<u32>,

    /// Sampling temperature (0.0 = greedy, 1.0 = default).
    #[arg(long)]
    pub temperature: Option<f64>,

    /// Number of GPUs for tensor parallelism (default: 1).
    #[arg(long, default_value_t = 1)]
    pub tensor_parallel_size: usize,

    /// Disable CUDA graphs (use eager mode).
    #[arg(long)]
    pub enforce_eager: bool,

    /// Chat template override. Accepts either an inline Jinja string
    /// or a path to a `tokenizer_config.json` / `.jinja` file. Useful
    /// for GGUFs whose metadata lacks `tokenizer.chat_template`.
    #[arg(long)]
    pub chat_template: Option<String>,
}

impl ChatArgs {
    /// Resolve the effective model path/ID (if any).
    #[cfg_attr(not(feature = "chat"), allow(dead_code))]
    pub fn resolved_model(&self) -> Option<String> {
        self.model_tag.clone().or_else(|| self.model.clone())
    }
}

/// Arguments for the `complete` subcommand.
#[derive(Parser, Debug)]
#[command(override_usage = "scr complete [OPTIONS]")]
pub struct CompleteArgs {
    /// URL of the running OpenAI-compatible API server.
    #[arg(long, default_value = "http://localhost:8000/v1")]
    pub url: String,

    /// Model name for completions (default: first model from server).
    #[arg(long)]
    pub model_name: Option<String>,

    /// API key for authentication.
    #[arg(long, env = "OPENAI_API_KEY")]
    pub api_key: Option<String>,

    /// Maximum number of tokens to generate.
    #[arg(long)]
    pub max_tokens: Option<usize>,

    /// Send a single prompt and exit (non-interactive mode).
    #[arg(short = 'q', long, value_name = "PROMPT")]
    pub quick: Option<String>,
}

/// Arguments for the `collect-env` subcommand (no options).
#[derive(Parser, Debug)]
#[command(override_usage = "scr collect-env")]
pub struct CollectEnvArgs {}

/// Arguments for the `batch` subcommand.
#[derive(Parser, Debug)]
#[command(override_usage = "scr batch [MODEL] [OPTIONS]")]
pub struct BatchArgs {
    /// Model: local path or HuggingFace model ID.
    pub model_tag: Option<String>,

    /// Path to a local model directory, or HuggingFace model ID.
    #[arg(short = 'm', long, env = "VLLM_MODEL")]
    pub model: Option<String>,

    /// Input JSONL file containing batch requests.
    #[arg(short = 'i', long)]
    pub input: String,

    /// Output JSONL file for batch results.
    #[arg(short = 'o', long)]
    pub output: String,

    /// Device: "cpu", "cuda:N", "metal", or "auto" (auto-detect best GPU).
    #[arg(long, default_value = "auto")]
    pub device: String,

    /// Weight dtype: "auto", "float16", "bfloat16", "float32".
    #[arg(long, default_value = "auto")]
    pub dtype: String,

    /// Maximum number of concurrent sequences. Unset lets the backend choose
    /// (see `ServeArgs::max_num_seqs`). Hybrid GDN arches (Qwen3.5 /
    /// Qwen3-Next) reserve a recurrent-state slot per sequence up-front, so
    /// large values cost real GPU memory.
    #[arg(long)]
    pub max_num_seqs: Option<usize>,

    /// Maximum tokens per scheduler iteration. THE knob that decides whether a step mixes a prefill with a
    /// decode, which is the configuration under investigation — `batch` could not set it until now.
    #[arg(long)]
    pub max_num_batched_tokens: Option<usize>,

    /// Disable async scheduling (overlap of device execution and CPU scheduling).
    ///
    /// ⛔ WIRED FOR A REASON: `SCRATCHY_SDSC_OPTRACE` segfaults in flex's response-completion thread at the
    /// first forward with async scheduling on, and its per-op `stream->synchronize()` is the suspect. The
    /// flag existed on `serve` but not here, so the hypothesis could not be tested through `scr batch` —
    /// the only sanctioned harness.
    #[arg(long)]
    pub disable_async_scheduling: bool,

    /// Disable prefix caching (KV reuse across shared prompt prefixes).
    ///
    /// ⛔ ALSO WIRED FOR A REASON: prefix caching silently confounded the first ragged-batch probe (every
    /// request but the first was reusing another's blocks), and turning it off through `batch` was not
    /// possible — the probe had to be redesigned with disjoint prefixes instead.
    #[arg(long)]
    pub no_prefix_caching: bool,

    /// HuggingFace token for gated models.
    #[arg(long, env = "HF_TOKEN")]
    pub hf_token: Option<String>,

    /// Log level: "trace", "debug", "info", "warn", "error".
    #[arg(long, default_value = "info")]
    pub log_level: String,

    /// Fraction of GPU memory to use for KV cache (0.0–1.0).
    #[arg(long, default_value_t = 0.9, env = "VLLM_GPU_MEMORY_UTILIZATION")]
    pub gpu_memory_utilization: f64,

    /// Tool call parser to use (e.g. "hermes", "llama3_json", "gemma4").
    #[arg(long)]
    pub tool_call_parser: Option<String>,

    /// Reasoning parser to use (e.g. "deepseek_r1", "qwen3").
    #[arg(long)]
    pub reasoning_parser: Option<String>,

    /// Default extra kwargs for the chat template, as JSON.
    /// Merged with per-request chat_template_kwargs (request overrides).
    /// Example: '{"enable_thinking": false}'
    #[arg(long, value_parser = parse_json_map)]
    pub default_chat_template_kwargs: Option<std::collections::HashMap<String, serde_json::Value>>,

    /// Specific GGUF filename to download from a HuggingFace repo.
    #[arg(long)]
    pub gguf_file: Option<String>,
}

impl BatchArgs {
    /// Resolve the effective model path/ID.
    #[cfg_attr(not(any(feature = "chat", feature = "serve")), allow(dead_code))]
    pub fn resolved_model(&self) -> Result<String, String> {
        if let Some(ref tag) = self.model_tag {
            Ok(tag.clone())
        } else if let Some(ref m) = self.model {
            Ok(m.clone())
        } else {
            Err("model is required: provide as positional arg or --model flag".to_string())
        }
    }
}

/// Arguments for the `convert` subcommand (stub).
#[derive(Parser, Debug)]
pub struct ConvertArgs {
    /// Input model directory.
    #[arg(long)]
    pub input: String,

    /// Output directory.
    #[arg(long)]
    pub output: String,

    /// Target dtype.
    #[arg(long, default_value = "f16")]
    pub dtype: String,
}

/// Arguments for `scr model rm`.
#[derive(Parser, Debug)]
#[command(override_usage = "scr model rm <MODEL>")]
pub struct RmArgs {
    /// Model to remove (e.g. "google/gemma-2-2b"). Must match the model ID shown by `scr model ls`.
    pub model: String,
}

/// Arguments for `scr model ls` / `scr model list`.
#[derive(Parser, Debug)]
pub struct ListArgs {
    /// Sort order: "name" (default) or "size".
    #[arg(short = 's', long, default_value = "name")]
    pub sort: ListSort,
}

#[derive(Clone, Debug, clap::ValueEnum)]
pub enum ListSort {
    Name,
    Size,
}

/// `scr model cache <subcommand>`.
#[derive(Parser, Debug)]
pub struct CacheCommand {
    #[command(subcommand)]
    pub command: CacheSubcommand,
}

#[derive(Subcommand, Debug)]
pub enum CacheSubcommand {
    /// Remove cached model weights to reclaim disk space. Scans the
    /// HuggingFace Hub cache and the scratchy aligned-weight cache.
    Clean(CacheCleanArgs),
    /// Show a disk-consumption breakdown of both caches (read-only).
    Inspect(CacheInspectArgs),
}

/// Arguments for `scr model cache clean`.
#[derive(Parser, Debug)]
#[command(override_usage = "scr model cache clean [OPTIONS]")]
pub struct CacheCleanArgs {
    /// Remove ALL cached items, not just those unused for `--days` days.
    #[arg(long)]
    pub nuke: bool,

    /// Skip the interactive y/N confirmation prompt.
    #[arg(short = 'f', long)]
    pub force: bool,

    /// Age threshold: remove items not used within this many days.
    /// Ignored with --nuke.
    #[arg(long, default_value_t = 30)]
    pub days: u64,
}

/// Arguments for `scr model cache inspect`.
#[derive(Parser, Debug)]
#[command(override_usage = "scr model cache inspect [OPTIONS]")]
pub struct CacheInspectArgs {
    /// Sort order: "size" (default, largest first) or "name".
    #[arg(short = 's', long, default_value = "size")]
    pub sort: ListSort,
}

/// Arguments for the `pull` subcommand.
#[derive(Parser, Debug)]
#[command(override_usage = "scr model pull <MODEL> [OPTIONS]")]
pub struct PullArgs {
    /// HuggingFace model ID or local path (e.g. "meta-llama/Llama-3.2-1B").
    pub model: String,

    /// HuggingFace API token for gated models.
    #[arg(long, env = "HF_TOKEN")]
    pub hf_token: Option<String>,

    /// Specific GGUF filename to download (for GGUF repos).
    #[arg(long)]
    pub gguf_file: Option<String>,

    /// GGUF quantization to prefer (e.g. Q4_K_M, Q8_0). Case-insensitive.
    /// When set, auto-selects the matching GGUF file from the repo.
    #[arg(short = 'q', long)]
    pub quantization: Option<String>,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Parse a JSON string into a `HashMap<String, serde_json::Value>`.
/// Used as a clap `value_parser` for `--default-chat-template-kwargs`.
fn parse_json_map(s: &str) -> Result<std::collections::HashMap<String, serde_json::Value>, String> {
    serde_json::from_str(s).map_err(|e| format!("invalid JSON: {e}"))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn test_cli_help_doesnt_panic() {
        // Verify the CLI definition is valid.
        Cli::command().debug_assert();
    }

    #[cfg(feature = "serve")]
    #[test]
    fn test_parse_serve_positional_model() {
        // Python-compatible: `vllm serve meta-llama/Llama-3.2-1B`
        let cli = Cli::parse_from(["scr", "serve", "meta-llama/Llama-3.2-1B", "--port", "9000"]);
        match cli.command {
            Commands::Serve(args) => {
                assert_eq!(args.resolved_model().unwrap(), "meta-llama/Llama-3.2-1B");
                assert_eq!(args.port, 9000);
            }
            _ => panic!("expected Serve command"),
        }
    }

    #[cfg(feature = "serve")]
    #[test]
    fn test_parse_serve_flag_model() {
        // Also supported: `scr serve --model meta-llama/Llama-3.2-1B`
        let cli = Cli::parse_from([
            "scr",
            "serve",
            "--model",
            "meta-llama/Llama-3.2-1B",
            "--device",
            "cpu",
        ]);
        match cli.command {
            Commands::Serve(args) => {
                assert_eq!(args.resolved_model().unwrap(), "meta-llama/Llama-3.2-1B");
                assert_eq!(args.device, "cpu");
            }
            _ => panic!("expected Serve command"),
        }
    }

    #[cfg(feature = "serve")]
    #[test]
    fn test_serve_positional_takes_precedence() {
        let cli = Cli::parse_from(["scr", "serve", "positional-model", "--model", "flag-model"]);
        match cli.command {
            Commands::Serve(args) => {
                // Positional wins.
                assert_eq!(args.resolved_model().unwrap(), "positional-model");
            }
            _ => panic!("expected Serve command"),
        }
    }

    #[cfg(feature = "serve")]
    #[test]
    fn test_serve_no_model_errors() {
        let cli = Cli::parse_from(["scr", "serve"]);
        match cli.command {
            Commands::Serve(args) => {
                assert!(args.resolved_model().is_err());
            }
            _ => panic!("expected Serve command"),
        }
    }

    #[test]
    #[cfg(feature = "bench")]
    fn test_parse_bench_latency_positional_model() {
        let cli = Cli::parse_from([
            "scr",
            "bench",
            "latency",
            "/path/to/model",
            "--num-iters",
            "5",
        ]);
        match cli.command {
            Commands::Bench(cmd) => match cmd.command {
                BenchCommands::Latency(args) => {
                    assert_eq!(
                        args.resolved_models().unwrap(),
                        vec!["/path/to/model".to_string()]
                    );
                    assert_eq!(args.num_iters, 5);
                }
                _ => panic!("expected Latency subcommand"),
            },
            _ => panic!("expected Bench command"),
        }
    }

    #[test]
    #[cfg(feature = "bench")]
    fn test_parse_bench_latency_flag_model() {
        let cli = Cli::parse_from(["scr", "bench", "latency", "--model", "/path/to/model"]);
        match cli.command {
            Commands::Bench(cmd) => match cmd.command {
                BenchCommands::Latency(args) => {
                    assert_eq!(
                        args.resolved_models().unwrap(),
                        vec!["/path/to/model".to_string()]
                    );
                }
                _ => panic!("expected Latency subcommand"),
            },
            _ => panic!("expected Bench command"),
        }
    }

    #[test]
    #[cfg(feature = "bench")]
    fn test_parse_bench_latency_defaults() {
        let cli = Cli::parse_from(["scr", "bench", "latency", "some-model"]);
        match cli.command {
            Commands::Bench(cmd) => match cmd.command {
                BenchCommands::Latency(args) => {
                    assert_eq!(args.num_iters, 30);
                    assert_eq!(args.input_len, 32);
                    assert_eq!(args.output_len, 128);
                    assert_eq!(args.num_iters_warmup, 10);
                    assert_eq!(args.batch_sizes, vec![8]);
                    assert!(args.output_json.is_none());
                }
                _ => panic!("expected Latency subcommand"),
            },
            _ => panic!("expected Bench command"),
        }
    }

    #[test]
    #[cfg(feature = "bench")]
    fn test_bench_requires_subcommand() {
        // `scr bench` alone (no subcommand) should fail to parse.
        let result = Cli::try_parse_from(["scr", "bench"]);
        assert!(result.is_err());
    }

    #[cfg(any(feature = "chat", feature = "serve"))]
    #[test]
    fn test_parse_batch_args() {
        let cli = Cli::parse_from([
            "scr",
            "batch",
            "meta-llama/Llama-3.2-1B",
            "-i",
            "input.jsonl",
            "-o",
            "output.jsonl",
        ]);
        match cli.command {
            Commands::Batch(args) => {
                assert_eq!(args.resolved_model().unwrap(), "meta-llama/Llama-3.2-1B");
                assert_eq!(args.input, "input.jsonl");
                assert_eq!(args.output, "output.jsonl");
                assert_eq!(args.device, "auto");
            }
            _ => panic!("expected Batch command"),
        }
    }

    #[cfg(any(feature = "chat", feature = "serve"))]
    #[test]
    fn test_batch_resolved_model() {
        // Flag model.
        let cli = Cli::parse_from([
            "scr",
            "batch",
            "--model",
            "flag-model",
            "-i",
            "in.jsonl",
            "-o",
            "out.jsonl",
        ]);
        match cli.command {
            Commands::Batch(args) => {
                assert_eq!(args.resolved_model().unwrap(), "flag-model");
            }
            _ => panic!("expected Batch command"),
        }

        // Positional takes precedence over --model.
        let cli = Cli::parse_from([
            "scr",
            "batch",
            "positional-model",
            "--model",
            "flag-model",
            "-i",
            "in.jsonl",
            "-o",
            "out.jsonl",
        ]);
        match cli.command {
            Commands::Batch(args) => {
                assert_eq!(args.resolved_model().unwrap(), "positional-model");
            }
            _ => panic!("expected Batch command"),
        }

        // No model → error.
        let cli = Cli::parse_from(["scr", "batch", "-i", "in.jsonl", "-o", "out.jsonl"]);
        match cli.command {
            Commands::Batch(args) => {
                assert!(args.resolved_model().is_err());
            }
            _ => panic!("expected Batch command"),
        }
    }

    #[test]
    fn test_parse_convert_args() {
        let cli = Cli::parse_from([
            "scr", "model", "convert", "--input", "/in", "--output", "/out", "--dtype", "bf16",
        ]);
        match cli.command {
            Commands::Model(cmd) => match cmd.command {
                ModelSubcommand::Convert(args) => {
                    assert_eq!(args.input, "/in");
                    assert_eq!(args.output, "/out");
                    assert_eq!(args.dtype, "bf16");
                }
                _ => panic!("expected Convert subcommand"),
            },
            _ => panic!("expected Model command"),
        }
    }

    // -- Runner flag tests --

    #[cfg(feature = "serve")]
    #[test]
    fn test_serve_runner_default_is_generate() {
        let cli = Cli::parse_from(["scr", "serve", "some-model"]);
        match cli.command {
            Commands::Serve(args) => {
                assert_eq!(args.runner, "generate");
            }
            _ => panic!("expected Serve command"),
        }
    }

    #[cfg(feature = "serve")]
    #[test]
    fn test_serve_runner_pooling() {
        let cli = Cli::parse_from(["scr", "serve", "some-model", "--runner", "pooling"]);
        match cli.command {
            Commands::Serve(args) => {
                assert_eq!(args.runner, "pooling");
            }
            _ => panic!("expected Serve command"),
        }
    }

    // -- Chat command tests --

    #[test]
    fn test_chat_defaults_remote_mode() {
        let cli = Cli::parse_from(["scr", "chat"]);
        match cli.command {
            Commands::Chat(args) => {
                assert!(args.resolved_model().is_none());
                assert_eq!(args.url, "http://localhost:8000/v1");
                assert!(args.system_prompt.is_none());
                assert!(args.quick.is_none());
                assert!(args.prompt.is_empty());
                assert!(!args.bench);
                assert_eq!(args.device, "auto");
                assert_eq!(args.dtype, "auto");
                assert_eq!(args.max_tokens, None);
            }
            _ => panic!("expected Chat command"),
        }
    }

    #[test]
    fn test_chat_inproc_positional_model() {
        let cli = Cli::parse_from(["scr", "chat", "Qwen/Qwen2.5-0.5B"]);
        match cli.command {
            Commands::Chat(args) => {
                assert_eq!(args.resolved_model().unwrap(), "Qwen/Qwen2.5-0.5B");
            }
            _ => panic!("expected Chat command"),
        }
    }

    #[test]
    fn test_chat_inproc_flag_model() {
        let cli = Cli::parse_from(["scr", "chat", "--model", "my-model", "--device", "cpu"]);
        match cli.command {
            Commands::Chat(args) => {
                assert_eq!(args.resolved_model().unwrap(), "my-model");
                assert_eq!(args.device, "cpu");
            }
            _ => panic!("expected Chat command"),
        }
    }

    #[test]
    fn test_chat_positional_takes_precedence() {
        let cli = Cli::parse_from(["scr", "chat", "positional", "--model", "flag"]);
        match cli.command {
            Commands::Chat(args) => {
                assert_eq!(args.resolved_model().unwrap(), "positional");
            }
            _ => panic!("expected Chat command"),
        }
    }

    #[test]
    fn test_chat_quick_short_flag() {
        let cli = Cli::parse_from(["scr", "chat", "-q", "hello"]);
        match cli.command {
            Commands::Chat(args) => {
                assert_eq!(args.quick.as_deref(), Some("hello"));
            }
            _ => panic!("expected Chat command"),
        }
    }

    #[test]
    fn test_chat_prompt_and_bench() {
        let cli = Cli::parse_from([
            "scr",
            "chat",
            "my-model",
            "--prompt",
            "Tell me a joke",
            "--bench",
        ]);
        match cli.command {
            Commands::Chat(args) => {
                assert_eq!(args.prompt, vec!["Tell me a joke"]);
                assert!(args.bench);
                assert_eq!(args.resolved_model().unwrap(), "my-model");
            }
            _ => panic!("expected Chat command"),
        }
    }

    #[test]
    fn test_chat_system_prompt() {
        let cli = Cli::parse_from([
            "scr",
            "chat",
            "--system-prompt",
            "You are a pirate.",
            "--url",
            "http://example.com/v1",
        ]);
        match cli.command {
            Commands::Chat(args) => {
                assert_eq!(args.system_prompt.as_deref(), Some("You are a pirate."));
                assert_eq!(args.url, "http://example.com/v1");
            }
            _ => panic!("expected Chat command"),
        }
    }

    #[test]
    fn test_chat_all_inproc_options() {
        let cli = Cli::parse_from([
            "scr",
            "chat",
            "my-model",
            "--device",
            "metal",
            "--dtype",
            "float16",
            "--max-model-len",
            "4096",
            "--gguf-file",
            "model.gguf",
            "--max-tokens",
            "256",
        ]);
        match cli.command {
            Commands::Chat(args) => {
                assert_eq!(args.device, "metal");
                assert_eq!(args.dtype, "float16");
                assert_eq!(args.max_model_len, Some(4096));
                assert_eq!(args.gguf_file.as_deref(), Some("model.gguf"));
                assert_eq!(args.max_tokens, Some(256));
            }
            _ => panic!("expected Chat command"),
        }
    }

    // -- Complete command tests --

    #[test]
    fn test_complete_defaults() {
        let cli = Cli::parse_from(["scr", "complete"]);
        match cli.command {
            Commands::Complete(args) => {
                assert_eq!(args.url, "http://localhost:8000/v1");
                assert!(args.model_name.is_none());
                assert!(args.max_tokens.is_none());
                assert!(args.quick.is_none());
            }
            _ => panic!("expected Complete command"),
        }
    }

    #[test]
    fn test_complete_all_options() {
        let cli = Cli::parse_from([
            "scr",
            "complete",
            "--url",
            "http://host:9000/v1",
            "--model-name",
            "gpt-4",
            "--max-tokens",
            "256",
            "-q",
            "Once upon a time",
        ]);
        match cli.command {
            Commands::Complete(args) => {
                assert_eq!(args.url, "http://host:9000/v1");
                assert_eq!(args.model_name.as_deref(), Some("gpt-4"));
                assert_eq!(args.max_tokens, Some(256));
                assert_eq!(args.quick.as_deref(), Some("Once upon a time"));
            }
            _ => panic!("expected Complete command"),
        }
    }

    // -- run-batch alias tests --

    #[cfg(any(feature = "chat", feature = "serve"))]
    #[test]
    fn test_run_batch_alias() {
        let cli = Cli::parse_from([
            "scr",
            "run-batch",
            "my-model",
            "-i",
            "in.jsonl",
            "-o",
            "out.jsonl",
        ]);
        match cli.command {
            Commands::RunBatch(args) => {
                assert_eq!(args.resolved_model().unwrap(), "my-model");
                assert_eq!(args.input, "in.jsonl");
                assert_eq!(args.output, "out.jsonl");
            }
            _ => panic!("expected RunBatch command"),
        }
    }

    #[cfg(any(feature = "chat", feature = "serve"))]
    #[test]
    fn test_run_batch_same_args_as_batch() {
        // Verify run-batch accepts all the same flags as batch.
        let cli = Cli::parse_from([
            "scr",
            "run-batch",
            "--model",
            "flag-model",
            "-i",
            "in.jsonl",
            "-o",
            "out.jsonl",
            "--device",
            "cuda:0",
            "--dtype",
            "bfloat16",
        ]);
        match cli.command {
            Commands::RunBatch(args) => {
                assert_eq!(args.resolved_model().unwrap(), "flag-model");
                assert_eq!(args.device, "cuda:0");
                assert_eq!(args.dtype, "bfloat16");
            }
            _ => panic!("expected RunBatch command"),
        }
    }

    // -- cache clean tests --

    #[test]
    fn test_cache_clean_defaults() {
        let cli = Cli::parse_from(["scr", "model", "cache", "clean"]);
        match cli.command {
            Commands::Model(cmd) => match cmd.command {
                ModelSubcommand::Cache(cache_cmd) => match cache_cmd.command {
                    CacheSubcommand::Clean(args) => {
                        assert!(!args.nuke);
                        assert!(!args.force);
                        assert_eq!(args.days, 30);
                    }
                    _ => panic!("expected Clean subcommand"),
                },
                _ => panic!("expected Cache subcommand"),
            },
            _ => panic!("expected Model command"),
        }
    }

    #[test]
    fn test_cache_clean_all_flags() {
        let cli = Cli::parse_from([
            "scr", "model", "cache", "clean", "--nuke", "--force", "--days", "7",
        ]);
        match cli.command {
            Commands::Model(cmd) => match cmd.command {
                ModelSubcommand::Cache(cache_cmd) => match cache_cmd.command {
                    CacheSubcommand::Clean(args) => {
                        assert!(args.nuke);
                        assert!(args.force);
                        assert_eq!(args.days, 7);
                    }
                    _ => panic!("expected Clean subcommand"),
                },
                _ => panic!("expected Cache subcommand"),
            },
            _ => panic!("expected Model command"),
        }
    }

    #[test]
    fn test_cache_clean_force_short_flag() {
        let cli = Cli::parse_from(["scr", "model", "cache", "clean", "-f"]);
        match cli.command {
            Commands::Model(cmd) => match cmd.command {
                ModelSubcommand::Cache(cache_cmd) => match cache_cmd.command {
                    CacheSubcommand::Clean(args) => assert!(args.force),
                    _ => panic!("expected Clean subcommand"),
                },
                _ => panic!("expected Cache subcommand"),
            },
            _ => panic!("expected Model command"),
        }
    }

    #[test]
    fn test_cache_requires_subcommand() {
        // `scr model cache` alone (no subcommand) should fail to parse.
        assert!(Cli::try_parse_from(["scr", "model", "cache"]).is_err());
    }

    #[test]
    fn test_cache_inspect_defaults_to_size_sort() {
        let cli = Cli::parse_from(["scr", "model", "cache", "inspect"]);
        match cli.command {
            Commands::Model(cmd) => match cmd.command {
                ModelSubcommand::Cache(cache_cmd) => match cache_cmd.command {
                    CacheSubcommand::Inspect(args) => assert!(matches!(args.sort, ListSort::Size)),
                    _ => panic!("expected Inspect subcommand"),
                },
                _ => panic!("expected Cache subcommand"),
            },
            _ => panic!("expected Model command"),
        }
    }

    #[test]
    fn test_cache_inspect_sort_name() {
        let cli = Cli::parse_from(["scr", "model", "cache", "inspect", "--sort", "name"]);
        match cli.command {
            Commands::Model(cmd) => match cmd.command {
                ModelSubcommand::Cache(cache_cmd) => match cache_cmd.command {
                    CacheSubcommand::Inspect(args) => assert!(matches!(args.sort, ListSort::Name)),
                    _ => panic!("expected Inspect subcommand"),
                },
                _ => panic!("expected Cache subcommand"),
            },
            _ => panic!("expected Model command"),
        }
    }

    // -- collect-env tests --

    #[test]
    fn test_collect_env_no_args() {
        let cli = Cli::parse_from(["scr", "collect-env"]);
        assert!(matches!(cli.command, Commands::CollectEnv(_)));
    }

    // -- distributed-executor-backend tests --

    #[cfg(feature = "serve")]
    #[test]
    fn test_serve_distributed_backend_default() {
        let cli = Cli::parse_from(["scr", "serve", "some-model"]);
        match cli.command {
            Commands::Serve(args) => {
                assert_eq!(args.distributed_executor_backend, "auto");
            }
            _ => panic!("expected Serve command"),
        }
    }

    #[cfg(feature = "serve")]
    #[test]
    fn test_serve_distributed_backend_external_launcher() {
        let cli = Cli::parse_from([
            "scr",
            "serve",
            "some-model",
            "--distributed-executor-backend",
            "external_launcher",
        ]);
        match cli.command {
            Commands::Serve(args) => {
                assert_eq!(args.distributed_executor_backend, "external_launcher");
            }
            _ => panic!("expected Serve command"),
        }
    }

    #[cfg(feature = "serve")]
    #[test]
    fn test_serve_distributed_backend_with_tp() {
        let cli = Cli::parse_from([
            "scr",
            "serve",
            "some-model",
            "--distributed-executor-backend",
            "external_launcher",
            "--tensor-parallel-size",
            "4",
        ]);
        match cli.command {
            Commands::Serve(args) => {
                assert_eq!(args.distributed_executor_backend, "external_launcher");
                assert_eq!(args.tensor_parallel_size, 4);
            }
            _ => panic!("expected Serve command"),
        }
    }
}

/// `scr bundle <subcommand>`.
#[cfg(feature = "spyre-hw")]
#[derive(Parser, Debug)]
pub struct BundleCommand {
    #[command(subcommand)]
    pub command: BundleSubcommand,
}

#[cfg(feature = "spyre-hw")]
#[derive(Subcommand, Debug)]
pub enum BundleSubcommand {
    /// Load, prepare and run one bundle directory on the card.
    Run(BundleRunArgs),
}

/// Arguments for `scr bundle run`.
#[cfg(feature = "spyre-hw")]
#[derive(Parser, Debug)]
pub struct BundleRunArgs {
    /// The bundle directory: `sdsc_*.json` + `placements.json` + `spyreCodeDir/`.
    pub dir: String,

    /// Fill a source tensor from a file: `--input t0=q.bin`. The file's length must equal that
    /// tensor's placement size — a short buffer would leave the tail whatever the device had there,
    /// which is a silently wrong answer rather than a failure.
    ///
    /// Three markers ride on this same list, so no argument plumbing changes:
    /// `stick:t<id>=<rows>` states the row count the descriptors address a tensor with (the device
    /// layout is stick-major and a row-major bind is a plausible wrong answer);
    /// `raw:t<id>` stages that tensor's bytes verbatim, with no IEEE-to-SEN fp16 re-encoding, which
    /// an int32 index buffer requires because the fp16 pass rewrites each 4-byte entry as two fp16
    /// mantissas; and `const:t<id>=<value>` binds a one-element fp16 scalar stated here rather than
    /// in a two-byte file, e.g. `const:t4294967275=12.0` for a ScalarMul scale.
    #[arg(long = "input")]
    pub input: Vec<String>,

    /// Read a tensor back after the run: `--output t2=o.bin`.
    #[arg(long = "output")]
    pub output: Vec<String>,
}
