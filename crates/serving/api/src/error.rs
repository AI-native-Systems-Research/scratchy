// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Error types for the serving layer.

#[cfg(feature = "serve")]
use axum::Json;
#[cfg(feature = "serve")]
use axum::http::StatusCode;
#[cfg(feature = "serve")]
use axum::response::{IntoResponse, Response};

#[cfg(feature = "serve")]
use crate::protocol::ErrorResponse;

/// Errors that can occur in the serving layer.
#[derive(Debug, thiserror::Error)]
pub enum ServeError {
    #[error("validation error: {0}")]
    Validation(String),

    #[error("model not found: {0}")]
    ModelNotFound(String),

    #[error("engine error: {0}")]
    Engine(String),

    #[error("engine is shut down")]
    EngineShutdown,

    #[error("request not found: {0}")]
    RequestNotFound(String),

    #[error("internal error: {0}")]
    Internal(String),
}

#[cfg(feature = "serve")]
impl ServeError {
    fn status_code(&self) -> StatusCode {
        match self {
            ServeError::Validation(_) => StatusCode::BAD_REQUEST,
            ServeError::ModelNotFound(_) => StatusCode::NOT_FOUND,
            ServeError::Engine(_) => StatusCode::INTERNAL_SERVER_ERROR,
            ServeError::EngineShutdown => StatusCode::SERVICE_UNAVAILABLE,
            ServeError::RequestNotFound(_) => StatusCode::NOT_FOUND,
            ServeError::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }

    fn error_type(&self) -> &str {
        match self {
            ServeError::Validation(_) => "BadRequestError",
            ServeError::ModelNotFound(_) => "NotFoundError",
            ServeError::Engine(_) => "InternalServerError",
            ServeError::EngineShutdown => "ServiceUnavailableError",
            ServeError::RequestNotFound(_) => "NotFoundError",
            ServeError::Internal(_) => "InternalServerError",
        }
    }
}

#[cfg(feature = "serve")]
impl IntoResponse for ServeError {
    fn into_response(self) -> Response {
        let status = self.status_code();
        let error_type = self.error_type();
        let message = self.to_string();

        let body = ErrorResponse::new(message, error_type, status.as_u16());
        (status, Json(body)).into_response()
    }
}

impl From<scratchy_serving_engine::error::EngineError> for ServeError {
    fn from(e: scratchy_serving_engine::error::EngineError) -> Self {
        ServeError::Engine(e.to_string())
    }
}

pub type ServeResult<T> = Result<T, ServeError>;
