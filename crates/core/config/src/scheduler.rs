// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Scheduler configuration types, ported from `vllm/config/scheduler.py`.

use serde::{Deserialize, Serialize};

/// Scheduling policy for request ordering.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SchedulerPolicy {
    /// First come, first served -- requests are handled in arrival order.
    #[default]
    Fcfs,
    /// Priority-based -- requests are handled by priority value (lower = earlier),
    /// with ties broken by arrival time.
    Priority,
}

/// The type of model runner to launch.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RunnerType {
    #[default]
    Generate,
    Pooling,
    Draft,
}

/// Scheduler configuration.
///
/// Ported from `vllm.config.scheduler.SchedulerConfig`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerConfig {
    /// The runner type to launch for the model.
    #[serde(default)]
    pub runner_type: RunnerType,

    /// Maximum number of tokens that can be processed in a single iteration.
    #[serde(default = "SchedulerConfig::default_max_num_batched_tokens")]
    pub max_num_batched_tokens: usize,

    /// Maximum number of tokens that the scheduler may issue in a single
    /// iteration.  Usually equal to `max_num_batched_tokens` but can be
    /// smaller when the model might append tokens (e.g. speculative decoding).
    #[serde(default)]
    pub max_num_scheduled_tokens: Option<usize>,

    /// Maximum number of sequences to be processed in a single iteration.
    #[serde(default = "SchedulerConfig::default_max_num_seqs")]
    pub max_num_seqs: usize,

    /// For chunked prefill, the maximum number of sequences that can be
    /// partially prefilled concurrently.
    #[serde(default = "one")]
    pub max_num_partial_prefills: usize,

    /// For chunked prefill, the maximum number of prompts longer than
    /// `long_prefill_token_threshold` that will be prefilled concurrently.
    #[serde(default = "one")]
    pub max_long_partial_prefills: usize,

    /// For chunked prefill, a request is considered long if the prompt is
    /// longer than this number of tokens.
    #[serde(default)]
    pub long_prefill_token_threshold: usize,

    /// If true, prefill requests can be chunked based on the remaining
    /// `max_num_batched_tokens`.
    #[serde(default = "bool_true")]
    pub enable_chunked_prefill: bool,

    /// True if the model is multimodal.
    #[serde(default)]
    pub is_multimodal_model: bool,

    /// Multimodal encoder compute budget, only used in V1.
    #[serde(default)]
    pub max_num_encoder_input_tokens: usize,

    /// Multimodal encoder cache size, only used in V1.
    #[serde(default)]
    pub encoder_cache_size: usize,

    /// The scheduling policy.
    #[serde(default)]
    pub policy: SchedulerPolicy,

    /// If true and chunked prefill is enabled, do not partially schedule a
    /// multimodal item.
    #[serde(default)]
    pub disable_chunked_mm_input: bool,

    /// The scheduler class to use. `None` means the default scheduler.
    #[serde(default)]
    pub scheduler_cls: Option<String>,

    /// If true, KV cache manager allocates the same size for all attention
    /// layers even when layer types differ. `None` means auto-detect.
    #[serde(default)]
    pub disable_hybrid_kv_cache_manager: Option<bool>,

    /// If false, disable async scheduling. `None` means auto-detect.
    #[serde(default)]
    pub async_scheduling: Option<bool>,

    /// The interval (or buffer size) for streaming in terms of token length.
    #[serde(default = "one")]
    pub stream_interval: usize,

    /// Number of lookahead tokens for speculative decoding.
    /// Set to `num_speculative_tokens` when spec decode is enabled, 0 otherwise.
    /// Used by the KV cache allocator to reserve extra blocks for draft tokens.
    #[serde(default)]
    pub num_lookahead_tokens: usize,

    /// Whether pipeline parallelism is active. When true and async_scheduling
    /// is false, the scheduler populates `new_token_ids` in
    /// `CachedRequestData` so non-last PP stages can embed the correct tokens.
    #[serde(default)]
    pub use_pp: bool,
}

fn one() -> usize {
    1
}

fn bool_true() -> bool {
    true
}

impl SchedulerConfig {
    /// Default value for `max_num_batched_tokens` (mirrors Python
    /// `DEFAULT_MAX_NUM_BATCHED_TOKENS`).
    pub const DEFAULT_MAX_NUM_BATCHED_TOKENS: usize = 2048;

    /// Default value for `max_num_seqs` (mirrors Python
    /// `DEFAULT_MAX_NUM_SEQS`).
    pub const DEFAULT_MAX_NUM_SEQS: usize = 128;

    fn default_max_num_batched_tokens() -> usize {
        Self::DEFAULT_MAX_NUM_BATCHED_TOKENS
    }

    fn default_max_num_seqs() -> usize {
        Self::DEFAULT_MAX_NUM_SEQS
    }

    /// Return the effective maximum number of scheduled tokens per iteration.
    pub fn effective_max_num_scheduled_tokens(&self) -> usize {
        self.max_num_scheduled_tokens
            .unwrap_or(self.max_num_batched_tokens)
    }

    // ----- Device/usage-aware batch defaults --------------------------------
    // Mirror Python vLLM EngineArgs.get_batch_defaults
    // (vllm/engine/arg_utils.py:1969-1988): the base constants above are the
    // floor, but a ≥70 GiB non-A100 GPU (H100/MI300x-class) uses much larger
    // defaults, and the offline (LLM/throughput) context uses larger values
    // than the online server. The Rust port previously hard-wired the base
    // constants regardless of device — leaving an 8× smaller token budget than
    // Python on H100 offline.

    /// Total-memory threshold (bytes) at/above which a GPU uses the large-GPU
    /// batch defaults. Mirrors Python's `device_memory >= 70 * GiB`.
    pub const LARGE_GPU_MIN_TOTAL_BYTES: u64 = 70 * 1024 * 1024 * 1024;
    /// Large-GPU offline (LLM/throughput) defaults: `(max_num_batched_tokens, max_num_seqs)`.
    const LARGE_GPU_OFFLINE_DEFAULTS: (usize, usize) = (16384, 1024);
    /// Large-GPU online-server defaults.
    const LARGE_GPU_SERVER_DEFAULTS: (usize, usize) = (8192, 1024);
    /// Default-GPU offline (LLM/throughput) defaults.
    const SMALL_GPU_OFFLINE_DEFAULTS: (usize, usize) = (8192, 256);
    /// Default-GPU online-server defaults.
    const SMALL_GPU_SERVER_DEFAULTS: (usize, usize) = (2048, 256);

    /// Device- and usage-context-aware `(max_num_batched_tokens, max_num_seqs)`
    /// defaults, mirroring Python vLLM's `EngineArgs.get_batch_defaults`.
    ///
    /// `is_offline` selects the LLM/throughput context (vs the online server).
    /// A100 is excluded from the large-GPU tier: large batched-token budgets
    /// regress A100 throughput (Python vLLM PR #17885).
    pub fn batch_defaults(
        device_total_bytes: u64,
        device_name: &str,
        is_offline: bool,
    ) -> (usize, usize) {
        let is_large = device_total_bytes >= Self::LARGE_GPU_MIN_TOTAL_BYTES
            && !device_name.to_ascii_lowercase().contains("a100");
        match (is_large, is_offline) {
            (true, true) => Self::LARGE_GPU_OFFLINE_DEFAULTS,
            (true, false) => Self::LARGE_GPU_SERVER_DEFAULTS,
            (false, true) => Self::SMALL_GPU_OFFLINE_DEFAULTS,
            (false, false) => Self::SMALL_GPU_SERVER_DEFAULTS,
        }
    }
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            runner_type: RunnerType::default(),
            max_num_batched_tokens: Self::DEFAULT_MAX_NUM_BATCHED_TOKENS,
            max_num_scheduled_tokens: None,
            max_num_seqs: Self::DEFAULT_MAX_NUM_SEQS,
            max_num_partial_prefills: 1,
            max_long_partial_prefills: 1,
            long_prefill_token_threshold: 0,
            enable_chunked_prefill: true,
            is_multimodal_model: false,
            max_num_encoder_input_tokens: Self::DEFAULT_MAX_NUM_BATCHED_TOKENS,
            encoder_cache_size: Self::DEFAULT_MAX_NUM_BATCHED_TOKENS,
            policy: SchedulerPolicy::default(),
            disable_chunked_mm_input: false,
            scheduler_cls: None,
            disable_hybrid_kv_cache_manager: None,
            async_scheduling: None,
            stream_interval: 1,
            num_lookahead_tokens: 0,
            use_pp: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_scheduler_config() {
        let cfg = SchedulerConfig::default();
        assert_eq!(cfg.max_num_batched_tokens, 2048);
        assert_eq!(cfg.max_num_seqs, 128);
        assert!(cfg.enable_chunked_prefill);
        assert_eq!(cfg.policy, SchedulerPolicy::Fcfs);
        assert_eq!(cfg.effective_max_num_scheduled_tokens(), 2048);
    }

    #[test]
    fn test_batch_defaults_device_and_context_aware() {
        const GIB: u64 = 1024 * 1024 * 1024;
        // H100-class (>=70 GiB, non-A100): offline gets the large budget.
        assert_eq!(
            SchedulerConfig::batch_defaults(80 * GIB, "NVIDIA H100 80GB HBM3", true),
            (16384, 1024)
        );
        // ...and the online server gets the (smaller) server defaults.
        assert_eq!(
            SchedulerConfig::batch_defaults(80 * GIB, "NVIDIA H100 80GB HBM3", false),
            (8192, 1024)
        );
        // A100 is explicitly excluded from the large tier (PR #17885).
        assert_eq!(
            SchedulerConfig::batch_defaults(80 * GIB, "NVIDIA A100-SXM4-80GB", true),
            (8192, 256)
        );
        // Smaller GPU (e.g. L4 24 GiB): default tier.
        assert_eq!(
            SchedulerConfig::batch_defaults(24 * GIB, "NVIDIA L4", true),
            (8192, 256)
        );
        assert_eq!(
            SchedulerConfig::batch_defaults(24 * GIB, "NVIDIA L4", false),
            (2048, 256)
        );
    }

    #[test]
    fn test_scheduler_config_roundtrip() {
        let cfg = SchedulerConfig::default();
        let json = serde_json::to_string(&cfg).unwrap();
        let cfg2: SchedulerConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(cfg.max_num_batched_tokens, cfg2.max_num_batched_tokens);
        assert_eq!(cfg.max_num_seqs, cfg2.max_num_seqs);
    }
}
