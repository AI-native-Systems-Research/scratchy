// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

//! `MetalStreamError` — the metal backend's shared command/execution
//! error type.
//!
//! The `MetalStream` classic-command-buffer wrapper (and `wait_for_completion`)
//! were removed: production dispatches through the MTL4 path
//! (`interpreter::metal::run_bucket_mtl4`), and tests / benches / the
//! cost-sweep profiler use [`crate::mtl4_dispatch`]. Only the error enum
//! remains, since it is the common `Result` error across the backend.

/// Error types for Metal command/execution operations.
#[derive(Debug, Clone)]
pub enum MetalStreamError {
    /// Command buffer creation failed
    CommandBufferCreationFailed,
    /// Command buffer execution failed
    ExecutionFailed(String),
    /// Device lost or unavailable
    DeviceLost,
    /// Timeout waiting for completion
    Timeout,
    /// Shader compilation failed
    ShaderCompilationFailed(String),
}

impl std::fmt::Display for MetalStreamError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CommandBufferCreationFailed => write!(f, "Failed to create command buffer"),
            Self::ExecutionFailed(msg) => write!(f, "Command buffer execution failed: {}", msg),
            Self::DeviceLost => write!(f, "Metal device lost or unavailable"),
            Self::Timeout => write!(f, "Timeout waiting for command buffer completion"),
            Self::ShaderCompilationFailed(msg) => write!(f, "Shader compilation failed: {}", msg),
        }
    }
}

impl std::error::Error for MetalStreamError {}
