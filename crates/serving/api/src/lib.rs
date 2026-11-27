// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! scratchy's serving front end.
//!
//! An OpenAI- and Anthropic-compatible HTTP API server using axum (the `serve`
//! feature): chat and text completions, responses, messages, embeddings, model
//! listing, tokenize and detokenize, health checks and version information. It
//! also holds what every request path shares: chat templates, tokenization, and
//! tool-call and reasoning parsing.
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
pub mod init;
pub use init::try_load_tokenizer;
pub use scratchy_serving_worker::step_segs::{chunk_segment_table, step_segment_table};

pub mod tokenizer;
pub use tokenizer::Tokenizer;
#[cfg(test)]
mod alloc_count;
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
pub mod signal;
pub mod spans;
pub mod tool_parser;
