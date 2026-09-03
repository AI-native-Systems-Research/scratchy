// SPDX-License-Identifier: Apache-2.0
//! Error types for model loading and inference.

pub mod error {
    use thiserror::Error;

    pub type ModelResult<T> = Result<T, ModelError>;

    #[derive(Debug, Error)]
    pub enum ModelError {
        #[error("unsupported dtype: {0}")]
        UnsupportedDType(String),

        #[error("shape mismatch: expected {expected} bytes, got {got}")]
        ShapeMismatch { expected: usize, got: usize },

        #[error("byte cast error for type {0}: alignment or size mismatch")]
        ByteCastError(&'static str),

        #[error(
            "sharding error: dim {dim} of size {dim_size} not divisible by world_size {world_size}"
        )]
        ShardingError {
            dim: usize,
            dim_size: usize,
            world_size: usize,
        },

        #[error("weight not found: {0}")]
        WeightNotFound(String),

        #[error("IO error: {0}")]
        Io(#[from] std::io::Error),

        #[error("JSON parse error: {0}")]
        Json(#[from] serde_json::Error),

        #[error("safetensors error: {0}")]
        SafeTensors(String),

        #[error("{0}")]
        Other(String),
    }
}
