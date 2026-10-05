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

    /// No compiled model claims this architecture. Surfaced when `try_load`
    /// returns `ArchLoad::ArchNotCompiled` — no `#[forward]` registration owns
    /// the HF arch at this tp size.
    ///
    /// "Not supported" alone reads as a missing backend, but by far the commoner
    /// cause is BUILD SCOPE: `scratchy-models` has no default model scope, so an
    /// arch is absent simply because no `model/<stem>` feature named it. Say
    /// both, since the reader cannot tell them apart from the outside.
    #[error(
        "architecture `{0}` is not compiled into this build. Either no model of that \
         architecture was named at build time — `scratchy-models` has no default model scope, \
         so name one with `model/<stem>` (or `model/<arch>` / `model/all` to widen) — or this \
         backend has no implementation for it."
    )]
    ArchNotSupported(String),

    /// The architecture IS compiled, but no compiled model variant's
    /// fingerprint accepted this checkpoint — `ArchLoad::NoVariantMatched`.
    /// Distinct from [`ExecutorError::ArchNotSupported`] because the fix is the
    /// build's model/quant scope, not a missing backend: the checkpoint's shapes
    /// or quantization layout differ from every `(model stem, quant preset)`
    /// pair this binary compiled.
    #[error(
        "architecture `{0}` is compiled, but no compiled model variant matches this checkpoint: \
         its shapes or quantization layout differ from every (model stem, quant preset) pair in \
         this build. Check that the build names this checkpoint's stem (`model/<stem>`) and, for \
         a quantized repo, a quant preset describing its actual on-disk widths (`quant/<preset>`, \
         declared in crates/models/arch/configs/<arch>/quantizations.json)."
    )]
    NoVariantMatched(String),

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
