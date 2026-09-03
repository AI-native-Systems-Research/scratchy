// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Engine-specific error types.

/// Errors that can occur in the engine core.
#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    /// Scheduler error.
    #[error("scheduler error: {0}")]
    Scheduler(String),

    /// Executor error.
    #[error("executor error: {0}")]
    Executor(String),

    /// Request not found.
    #[error("request not found: {0}")]
    RequestNotFound(String),

    /// Engine is shut down.
    #[error("engine is shut down")]
    Shutdown,

    /// Protocol/transport error.
    #[cfg(feature = "multiproc")]
    #[error("transport error: {0}")]
    Transport(#[from] scratchy_serving_transport::transport::TransportError),

    /// Configuration error.
    #[error("configuration error: {0}")]
    Config(String),
}

pub type EngineResult<T> = Result<T, EngineError>;

/// Errors that can occur in worker / executor operations. Lives here (below
/// `compiler` and the target crates) so backend worker impls in `targets/*`
/// can name it without a dependency cycle.
#[derive(Debug, thiserror::Error)]
pub enum ExecutorError {
    /// Worker initialization failed.
    #[error("worker initialization failed: {0}")]
    WorkerInit(String),

    /// The worker backend has no implementation for this model architecture.
    /// Surfaced when `scratchy_target_cuda::try_load` returns `Ok(None)`.
    #[error("architecture `{0}` not supported by this backend")]
    ArchNotSupported(String),

    /// Worker execution failed.
    #[error("worker execution failed: {0}")]
    WorkerExecution(String),

    /// Worker is not healthy.
    #[error("worker health check failed: {0}")]
    WorkerUnhealthy(String),

    /// Worker process died unexpectedly.
    #[error("worker process died: rank {rank}")]
    WorkerDied { rank: usize },

    /// Communication error between executor and worker.
    #[error("communication error: {0}")]
    Communication(String),

    /// Executor is shut down.
    #[error("executor is shut down")]
    Shutdown,

    /// Invalid configuration.
    #[error("invalid configuration: {0}")]
    Config(String),

    /// Timeout waiting for worker response.
    #[error("timeout waiting for worker (rank {rank}): {message}")]
    Timeout { rank: usize, message: String },

    /// Engine error (forwarded).
    #[error("engine error: {0}")]
    Engine(#[from] EngineError),
}

pub type ExecutorResult<T> = Result<T, ExecutorError>;
