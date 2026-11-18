// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Backend-neutral worker construction via an `inventory` registry.
//!
//! Each backend registers a `WorkerFactory` with
//! `inventory::submit!` in its target's worker crate, and a binary links exactly one target's
//! worker, so [`linked_factory`] is THE factory: `create_worker` (in `scratchy-serving-api`) calls
//! its [`WorkerFactory::create`] — no `#[cfg]` arms in the dispatch, and no device to select.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use scratchy_core_common::SamplingParams;
use scratchy_core_model::hub;
use scratchy_core_model::weight::HfModelConfig;

use crate::error::{ExecutorError, ExecutorResult};
use crate::worker::Worker;

/// The sampling a backend serves, one capability per field: what [`WorkerFactory::served`]
/// declares, in the form of `Worker::scores_prompts` (which answers for `prompt_logprobs`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Served {
    /// Sampling above temperature 0. Without it only greedy decoding is served: temperature 0,
    /// where top-k, top-p and min-p keep the argmax ([`SamplingParams::is_greedy`]).
    pub sampling: bool,
    /// The penalties that read a request's history: repetition, presence, frequency.
    pub penalties: bool,
    /// More than one completion per request (`n > 1`).
    pub completions: bool,
    /// Per-token `logprobs`.
    pub logprobs: bool,
    /// The transforms that can move the argmax: a logit bias, an allowed-token list, bad words,
    /// guided decoding.
    pub logit_transforms: bool,
}

impl Served {
    /// Every capability: a backend that samples, penalizes, completes `n` times, scores and
    /// transforms.
    pub const EVERYTHING: Self = Self {
        sampling: true,
        penalties: true,
        completions: true,
        logprobs: true,
        logit_transforms: true,
    };

    /// The first thing `p` asks for that the backend does not serve, in this order: a penalty,
    /// sampling, several completions, logprobs, a logit transform. `None` when it serves them
    /// all.
    pub fn refusal(&self, p: &SamplingParams) -> Option<Unserved> {
        let transform = [
            (p.logit_bias.is_some(), "logit_bias"),
            (p.allowed_token_ids.is_some(), "allowed_token_ids"),
            (p.bad_words_token_ids.is_some(), "bad_words"),
            (p.guided_grammar.is_some(), "guided decoding"),
        ]
        .into_iter()
        .find_map(|(set, field)| set.then_some(field));
        if !self.penalties && p.reads_history() {
            Some(Unserved::Penalties)
        } else if !self.sampling && !p.is_greedy() {
            Some(Unserved::Sampling {
                temperature: p.temperature,
            })
        } else if !self.completions && p.n != 1 {
            Some(Unserved::Completions { n: p.n })
        } else if !self.logprobs && p.logprobs.is_some() {
            Some(Unserved::Logprobs)
        } else if !self.logit_transforms {
            transform.map(|field| Unserved::LogitTransform { field })
        } else {
            None
        }
    }
}

/// What a request asks for that its backend does not serve ([`Served::refusal`]): the client's
/// validation error, naming what to change.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Unserved {
    Penalties,
    Sampling { temperature: f64 },
    Completions { n: u32 },
    Logprobs,
    LogitTransform { field: &'static str },
}

impl std::fmt::Display for Unserved {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Penalties => write!(
                f,
                "this backend serves no repetition, presence or frequency penalty: leave them at \
                 their defaults"
            ),
            Self::Sampling { temperature } => write!(
                f,
                "this backend decodes greedily only: temperature {temperature} is not served \
                 (pass \"temperature\": 0, or --temperature 0)"
            ),
            Self::Completions { n } => write!(
                f,
                "this backend serves one completion per request: n = {n} is not served (pass \
                 \"n\": 1)"
            ),
            Self::Logprobs => write!(f, "this backend does not serve logprobs: drop \"logprobs\""),
            Self::LogitTransform { field } => {
                write!(
                    f,
                    "this backend does not serve {field}: drop it from the request"
                )
            }
        }
    }
}

impl std::error::Error for Unserved {}

/// Configuration for a backend's worker.
///
/// Backend-neutral: lives in this neutral module (not a target's
/// worker crate) so the dispatch path — the `WorkerFactory` trait below and
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
    pub block_size: scratchy_core_common::KvBlockTokens,
    /// Maximum tokens per scheduler iteration (controls arena pre-sizing).
    pub max_num_batched_tokens: usize,
    /// Maximum concurrently-resident sequences (scheduler `max_num_seqs`).
    /// `None` = UNSET: the worker answers it (its own batched-decode width
    /// where it has one, else the device tier, clamped to what the device's
    /// memory affords where the model carries per-sequence state). `Some(n)`
    /// is the caller's explicit ask — honoured as-is; a backend that cannot
    /// express or afford it says so loudly rather than silently replacing it.
    /// Sizes the Gated-DeltaNet state pool — one recurrent-state slot per
    /// resident sequence for hybrid arches (Qwen3.5 / Qwen3-Next); unused by
    /// non-hybrid arches.
    pub max_num_seqs: Option<usize>,
    /// Whether the engine will serve cached prefixes (prefix caching requested and the backend able to
    /// honour it). A worker whose model carries recurrent state reserves the snapshots a prefix hit
    /// must restore only when this is set, so a run without caching does not pay their memory.
    pub enable_prefix_caching: bool,
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
    /// What the worker drafts with for speculative decoding: `Some(_)` loads a second model and
    /// its KV pool inside the same worker after the target loads. `None` = no draft.
    pub draft: Option<Draft>,
    /// The sequences a step may verify drafts for, whatever proposes them: what a hybrid
    /// target's draft state is sized for.
    pub verify_seqs: VerifySeqs,
    /// Optional dtype override for the draft model's weights ("auto",
    /// "bfloat16", "float16", …). `None` inherits the target's dtype.
    /// Currently passed through, but the draft path coerces to
    /// bf16 to match the target — accepted for CLI parity with Python.
    pub draft_model_dtype: Option<String>,
}

/// The sequences a step may verify drafts for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerifySeqs {
    /// None: the run drafts nothing.
    Off,
    /// As many a step as a multi-token-prediction head's budget drafts for
    /// ([`crate::spec_decode::DraftBudget::max_seqs`]).
    Head(crate::spec_decode::DraftBudget),
    /// Every sequence a step runs: an n-gram or standalone draft model drafts `drafts` tokens for
    /// each.
    Every { drafts: usize },
}

/// What a worker drafts with, besides its target.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Draft {
    /// A checkpoint of its own, a local path or Hub repo: a standalone draft model, or a
    /// multi-token-prediction head published apart from its target (its config names its arch).
    Checkpoint(String),
    /// The target's compiled multi-token-prediction head, which the target's own checkpoint
    /// carries: lent from the target's store. (A head published apart is a [`Self::Checkpoint`].)
    TargetHead,
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

/// A backend's worker constructor, registered via `inventory::submit!`. A binary links exactly
/// one: the worker crate of its target ([`linked_factory`]).
pub trait WorkerFactory: Sync {
    /// The target's name, for what the engine reports about its backend.
    fn name(&self) -> &'static str;

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
    /// backend exposes one (e.g. a device's recommended max working-set size).
    fn recommended_working_set_size(&self) -> Option<u64> {
        None
    }

    /// Current `(free, total)` device memory in bytes, if queryable (used by the
    /// `/gpu_memory` test endpoint). `None` when the backend can't report it.
    fn device_memory_free_total(&self) -> Option<(u64, u64)> {
        None
    }

    /// Whether this backend supports the engine's cross-request prefix cache.
    /// `true` for a paged-KV backend. A backend whose worker
    /// keeps a per-request KV cache (no shared/paged KV) cannot honor a
    /// scheduler-reported cached prefix it has no KV for, so it returns `false`
    /// and the engine disables prefix caching for it.
    fn supports_prefix_caching(&self) -> bool {
        true
    }

    /// THE BLOCK SIZE THIS BACKEND'S KV IS ADDRESSED BY, when the backend — not the CLI — is what
    /// decides it. `None` = the configured `--block-size` stands, which is every budget-sized GPU pool.
    ///
    /// ⛔ ASKED HERE BECAUSE THE ANSWER IS A BAKED CONSTANT OF THE POOL, and the engine cannot hold a
    /// copy of it: that would be two spellings of one quantity, crates apart, with a comment as the
    /// only thing claiming they agreed. The scheduler computes
    /// `slot = block_id * block_size + offset` and the worker resolves `block_id` to a pool page, so a
    /// disagreement is not a policy difference: it is the scheduler and the worker naming different
    /// cells for one token, silently.
    fn required_block_size(&self) -> Option<scratchy_core_common::KvBlockTokens> {
        None
    }

    /// THE SAMPLING THIS BACKEND SERVES ([`Served`]), a fact of the build the API reads at its
    /// request door ([`Served::refusal`]). The default serves everything.
    ///
    /// ⛔ READ BEFORE SCHEDULING, NOT IN THE WORKER. A worker that refuses a request it was
    /// handed fails a step the scheduler already formed: the client gets a server error, and the
    /// step queued behind it finds a request the worker never admitted. Read here, the request
    /// is refused alone, as a validation error, and the scheduler never sees it.
    fn served(&self) -> Served {
        Served::EVERYTHING
    }
}

inventory::collect!(&'static dyn WorkerFactory);

/// THE factory this binary serves through: the one its target's worker crate registered. A binary
/// links exactly one target's worker, so there is no device to choose; none (a binary that anchors
/// no worker crate, or a linker that dropped its registration) or several (two targets linked into
/// one binary) is a build defect, reported as such.
pub fn linked_factory() -> anyhow::Result<&'static dyn WorkerFactory> {
    let mut linked = inventory::iter::<&dyn WorkerFactory>();
    let Some(factory) = linked.next() else {
        anyhow::bail!(
            "No inference backend is linked into this binary: no `WorkerFactory` is registered. \
             A binary serves through the worker crate of its target, which it links and anchors \
             with an `extern crate ... as _` so the linker keeps the factory's \
             `inventory::submit!`."
        );
    };
    if let Some(other) = linked.next() {
        anyhow::bail!(
            "Two inference backends are linked into this binary (`{}` and `{}`): a binary links \
             exactly one target's worker.",
            factory.name(),
            other.name()
        );
    }
    Ok(*factory)
}

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

#[cfg(test)]
mod tests {
    use super::{Served, Unserved};
    use scratchy_core_common::SamplingParams;

    /// Greedy decoding of one completion, nothing else asked.
    fn greedy() -> SamplingParams {
        SamplingParams {
            temperature: 0.0,
            ..SamplingParams::default()
        }
    }

    /// A backend that serves everything refuses nothing; one that serves nothing refuses each
    /// request by the first capability it asks for, in `refusal`'s order.
    #[test]
    fn a_request_is_refused_by_the_first_capability_its_backend_lacks() {
        let none = Served {
            sampling: false,
            penalties: false,
            completions: false,
            logprobs: false,
            logit_transforms: false,
        };
        let hot = SamplingParams {
            temperature: 0.7,
            n: 2,
            ..greedy()
        };
        assert_eq!(Served::EVERYTHING.refusal(&hot), None);
        assert_eq!(none.refusal(&greedy()), None);
        let cases = [
            (
                SamplingParams {
                    presence_penalty: 0.5,
                    ..hot.clone()
                },
                Unserved::Penalties,
            ),
            (hot.clone(), Unserved::Sampling { temperature: 0.7 }),
            (
                SamplingParams { n: 2, ..greedy() },
                Unserved::Completions { n: 2 },
            ),
            (
                SamplingParams {
                    logprobs: Some(1),
                    ..greedy()
                },
                Unserved::Logprobs,
            ),
            (
                SamplingParams {
                    allowed_token_ids: Some(vec![1]),
                    ..greedy()
                },
                Unserved::LogitTransform {
                    field: "allowed_token_ids",
                },
            ),
        ];
        for (params, want) in cases {
            assert_eq!(none.refusal(&params), Some(want), "{want}");
        }
    }
}
