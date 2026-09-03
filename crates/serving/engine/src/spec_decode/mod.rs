// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Speculative-decoding proposer family.
//!
//! Two proposer shapes today:
//!
//!   * [`ProposerConfig::Ngram`] — pure-CPU n-gram lookup over the request's
//!     own token history. See [`ngram`].
//!   * [`ProposerConfig::DraftModel`] — a small "draft" model that runs K
//!     decode steps; the worker drives the GPU work and stashes drafts on
//!     `ModelRunnerOutput.draft_token_ids` for the engine to forward to the
//!     scheduler.
//!
//! Mirrors Python vLLM's `vllm/v1/spec_decode/` proposer family. Future
//! phases extract a `Proposer` trait + a `SpecDecodeBackend` trait so
//! draft-model spec decode works on any backend without per-backend wiring.

pub mod backend;
pub mod ngram;
pub mod proposer;
pub mod verify;

pub use backend::{
    BackendError, ForwardArgmaxRequest, ForwardHandle, KvPoolHandle, ModelHandle, SpecDecodeBackend,
};
pub use ngram::{NgramProposer, NgramProposerConfig};
pub use proposer::{DraftModelProposer, DraftSeedInputs, Proposer, ProposerStepCtx};
pub use verify::{RejectionResult, greedy_rejection_sample};

/// Configuration for the draft-model proposer.
///
/// Held by [`ProposerConfig::DraftModel`].
#[derive(Debug, Clone)]
pub struct DraftModelProposerConfig {
    /// Local path or HuggingFace repo ID for the draft model.
    pub model: String,
    /// Number of speculative tokens to propose per step.
    pub num_speculative_tokens: usize,
    /// Optional dtype override for the draft model's weights ("auto",
    /// "float16", "bfloat16", etc). `None` inherits the target's dtype.
    pub dtype: Option<String>,
    /// Maximum model context length, used to cap proposals.
    pub max_model_len: usize,
}

/// Speculative-decoding proposer selector.
///
/// `EngineCoreConfig::proposer_config` stores `Option<ProposerConfig>`:
/// `None` disables speculative decoding entirely, `Some(...)` selects one
/// of the proposer kinds.
#[derive(Debug, Clone)]
pub enum ProposerConfig {
    /// N-gram lookup proposer (no second model).
    Ngram(NgramProposerConfig),
    /// Draft-model proposer: a second model runs K autoregressive decode
    /// steps after each target step to produce drafts.
    DraftModel(DraftModelProposerConfig),
}

impl ProposerConfig {
    /// Number of speculative tokens this proposer drafts per step.
    /// Used by the scheduler to size lookahead block reservations.
    pub fn num_speculative_tokens(&self) -> usize {
        match self {
            Self::Ngram(c) => c.num_speculative_tokens,
            Self::DraftModel(c) => c.num_speculative_tokens,
        }
    }
}
