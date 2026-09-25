// SPDX-License-Identifier: Apache-2.0
//! Model configuration, weight index, and quantization config types.
//!
//! This crate provides:
//! - **HfModelConfig** for parsing HuggingFace `config.json`
//! - **SafeTensorsIndex** for sharded weight map lookups
//! - **LoRA adapter config** parsing
//! - **hub::download**, fetching a model from the HuggingFace Hub
//!
//! Quantization config parsing lives in the forward compiler
//! (`scratchy-forward-compiler-macro`, `QuantMethod`) with runtime support in
//! `scratchy-quantizations` and `scratchy-layers`; GGUF format support lives in
//! `scratchy-quantizations::gguf`.

pub mod attention_metadata;
pub mod embedding;
#[cfg(feature = "guided-decoding")]
pub mod grammar;
pub mod hub;
pub mod lora;
pub mod process_group;
pub mod tensor;
pub mod weight;

// Re-export for convenience.
pub use attention_metadata::AttentionMetadata;
pub use tensor::error;
pub use tensor::error::{ModelError, ModelResult};
