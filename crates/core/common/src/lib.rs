// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! `scratchy-core-common` -- shared types, errors, and utilities for the vLLM Rust port.
//!
//! This crate provides the foundational types that are used across all other
//! vLLM Rust crates:
//!
//! * [`sampling`] -- Sampling parameters and related enums.
//! * [`request`] -- The `Request` struct and `RequestStatus` enum.
//! * [`engine_io`] -- Engine-core I/O types (`EngineCoreRequest`,
//!   `EngineCoreOutput`, `EngineCoreOutputs`, events, `FinishReason`).

pub mod cache;
pub mod engine_io;
#[cfg(feature = "forward-telemetry")]
pub mod forward_telemetry;
pub mod kv;
pub mod multimodal;
pub mod request;
#[cfg(feature = "sampler-telemetry")]
pub mod sampler_telemetry;
pub mod sampling;
pub mod telemetry;

// ---- Convenience re-exports ------------------------------------------------
// These allow downstream crates to write `use scratchy_core_common::SamplingParams;`
// instead of `use scratchy_core_common::sampling::SamplingParams;`.

pub use engine_io::{
    EmbeddingData, EngineCoreEvent, EngineCoreEventType, EngineCoreOutput, EngineCoreOutputs,
    EngineCoreRequest, FinishReason, SchedulerStats, SpecDecodingStats, StopReason,
};
pub use kv::{CacheableTokens, InflightSlots, KvAddressing, KvBlockTokens, KvExtent, KvSlotSpan};
pub use multimodal::MultimodalData;
pub use request::{
    BlockAnnotations, BlockKind, Request, RequestStatus, compute_block_flags,
    compute_block_flags_rope_on_read,
};
pub use sampling::{
    GpuSampleParams, LogprobsOutput, RequestOutputKind, SamplingParams, SamplingType, TokenLogprob,
    fnv_seed, seed_to_uniform,
};
