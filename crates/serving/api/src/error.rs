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

    /// The request body is past the size the server buffers (axum's body limit).
    #[error("request too large: {0}")]
    PayloadTooLarge(String),

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

/// A JSON request body, read as axum's [`Json`] reads it, but rejected as the server's own error,
/// in the endpoint's error JSON rather than axum's plain text (#787).
/// - A body that isn't JSON, or doesn't deserialize into `T` (a missing field, an unknown
///   `tool_choice` form), is a 400 [`ServeError::Validation`], as the upstream APIs answer it,
///   where axum gives a 422. The message keeps serde's, which names the field.
/// - One past the body limit keeps axum's 413, as [`ServeError::PayloadTooLarge`]: a client must
///   still tell "too big" from "malformed".
#[cfg(feature = "serve")]
pub struct ApiJson<T>(pub T);

#[cfg(feature = "serve")]
impl<T, S> axum::extract::FromRequest<S> for ApiJson<T>
where
    T: serde::de::DeserializeOwned,
    S: Send + Sync,
{
    type Rejection = ServeError;

    async fn from_request(req: axum::extract::Request, state: &S) -> Result<Self, ServeError> {
        match Json::<T>::from_request(req, state).await {
            Ok(Json(value)) => Ok(Self(value)),
            Err(rejection) if rejection.status() == StatusCode::PAYLOAD_TOO_LARGE => {
                Err(ServeError::PayloadTooLarge(rejection.body_text()))
            }
            Err(rejection) => Err(ServeError::Validation(rejection.body_text())),
        }
    }
}

#[cfg(feature = "serve")]
impl ServeError {
    pub(crate) fn status_code(&self) -> StatusCode {
        match self {
            ServeError::Validation(_) => StatusCode::BAD_REQUEST,
            ServeError::PayloadTooLarge(_) => StatusCode::PAYLOAD_TOO_LARGE,
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
            ServeError::PayloadTooLarge(_) => "RequestTooLargeError",
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
