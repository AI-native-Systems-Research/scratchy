// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! HTTP serving layer for vLLM Rust port.
//!
//! Provides an OpenAI-compatible HTTP API server using axum, with endpoints
//! for chat completions, text completions, model listing, health checks,
//! and version information.
//!
//! Port of: `vllm/entrypoints/openai/` (subset)

#[cfg(feature = "serve")]
pub mod anthropic;
#[cfg(feature = "rag")]
pub mod augment;
pub mod chat_template;
pub mod detokenizer;
pub mod engine;
pub mod error;
pub mod headless;
pub mod init;
pub use init::try_load_tokenizer;

pub mod tokenizer;
pub use tokenizer::Tokenizer;
pub mod llm;
#[cfg(feature = "metrics")]
pub mod metrics;
#[cfg(feature = "multimodal")]
pub mod multimodal;
#[cfg(feature = "metrics")]
pub mod orca;
pub mod progress;
pub mod protocol;
#[cfg(feature = "serve")]
pub mod query;
pub mod reasoning_parser;
#[cfg(feature = "serve")]
pub mod responses;
#[cfg(feature = "serve")]
pub mod server;
pub mod spans;
pub mod tool_parser;
