// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Configuration for relocatable KV cache blocks ("spans").
//!
//! Spans enable position-independent KV cache block reuse, primarily for RAG
//! workloads where document blocks can be preloaded and shared across requests
//! regardless of their position in the sequence.
//!
//! Span behavior is driven by per-request [`BlockAnnotations`](scratchy_core_common::BlockAnnotations)
//! rather than global flags. Keys are always stored in the KV cache *without*
//! rotary position embedding (RoPE); position-specific rotation is applied at
//! attention time, making blocks relocatable to any sequence position.

use serde::{Deserialize, Serialize};

/// Configuration for relocatable KV cache blocks (spans).
///
/// Block hashing behavior is controlled by per-request
/// [`BlockKind`](scratchy_core_common::BlockKind) annotations produced by
/// `/v1/query/execute`. RoPE is always fused into attention (keys stored
/// without position encoding).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SpansConfig {
    /// Emit debug logging for span operations (hash resets, recomputation, etc.).
    /// Env: `VLLM_V1_SPANS_DEBUG` (default: `false`).
    pub debug: bool,
}

impl SpansConfig {
    /// Build a `SpansConfig` from environment variables.
    ///
    /// | Variable | Type | Default |
    /// |---|---|---|
    /// | `VLLM_V1_SPANS_DEBUG` | bool | `false` |
    pub fn from_env() -> Self {
        Self {
            debug: parse_bool_env("VLLM_V1_SPANS_DEBUG"),
        }
    }
}

/// Parse a boolean from an environment variable.
/// Accepts "true", "True", "1" as truthy; everything else (including unset) is false.
fn parse_bool_env(key: &str) -> bool {
    std::env::var(key)
        .map(|v| matches!(v.as_str(), "true" | "True" | "1"))
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_spans_config() {
        let cfg = SpansConfig::default();
        assert!(!cfg.debug);
    }

    #[test]
    fn test_spans_config_roundtrip() {
        let cfg = SpansConfig { debug: true };
        let json = serde_json::to_string(&cfg).unwrap();
        let cfg2: SpansConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(cfg.debug, cfg2.debug);
    }

    #[test]
    fn test_from_env() {
        let keys = ["VLLM_V1_SPANS_DEBUG"];
        let saved: Vec<_> = keys.iter().map(|k| std::env::var(k).ok()).collect();

        // SAFETY: test is single-threaded; we save and restore all modified vars.
        unsafe {
            std::env::set_var("VLLM_V1_SPANS_DEBUG", "1");
        }

        let cfg = SpansConfig::from_env();
        assert!(cfg.debug);

        // Restore env
        unsafe {
            for (key, val) in keys.iter().zip(saved.iter()) {
                match val {
                    Some(v) => std::env::set_var(key, v),
                    None => std::env::remove_var(key),
                }
            }
        }
    }
}
