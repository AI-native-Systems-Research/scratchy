// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

//! `MetalStreamError` — the metal backend's shared command/execution
//! error type.
//!
//! The `MetalStream` classic-command-buffer wrapper (and `wait_for_completion`)
//! were removed: production dispatches through the MTL4 path
//! (`interpreter::metal::run_bucket_mtl4`), and tests and the
//! cost-sweep profiler use [`crate::mtl4_dispatch`]. Only the error enum
//! remains, since it is the common `Result` error across the backend.

/// Error types for Metal command/execution operations.
#[derive(Debug, Clone)]
pub enum MetalStreamError {
    /// Command buffer execution failed
    ExecutionFailed(String),
    /// Shader compilation failed
    ShaderCompilationFailed(String),
    /// The device's IO-registry entry has no GPU core count
    /// ([`crate::device::gpu_cores`]).
    UnknownGpuCores,
    /// `ScratchyWeights::metal_off_tape` handed out something other than
    /// [`crate::off_tape::OffTapeKernels`].
    NotOffTapeKernels,
}

impl std::fmt::Display for MetalStreamError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ExecutionFailed(msg) => write!(f, "Command buffer execution failed: {}", msg),
            Self::ShaderCompilationFailed(msg) => write!(f, "Shader compilation failed: {}", msg),
            Self::UnknownGpuCores => {
                write!(f, "the device's IO-registry entry has no gpu-core-count")
            }
            Self::NotOffTapeKernels => {
                write!(f, "the model's off-tape kernels are not OffTapeKernels")
            }
        }
    }
}

impl std::error::Error for MetalStreamError {}
