// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! CLI subcommand implementations.

#[cfg(any(feature = "chat", feature = "serve"))]
pub mod batch;
#[cfg(feature = "bench")]
pub mod bench;
pub mod cache;
pub mod chat;
pub mod collect_env;
pub mod completions;
pub mod convert;
#[cfg(feature = "claude")]
pub mod launch;
pub mod model;
#[cfg(any(feature = "cuda", feature = "metal"))]
pub mod model_info;
pub mod pull;
#[cfg(feature = "serve")]
pub mod serve;
