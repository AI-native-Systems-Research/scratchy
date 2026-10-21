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

/// A JSON request body, read as axum's [`Json`] reads it, but rejected as the server's own
/// [`ServeError::Validation`]. So a body that isn't JSON, or doesn't deserialize into `T` (a missing
/// field, an unknown `tool_choice` form), gets a 400 in the endpoint's error JSON, as the upstream
/// APIs answer it, rather than axum's plain-text 422 (#787). The message keeps serde's, which
/// names the field.
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
            Err(rejection) => Err(ServeError::Validation(rejection.body_text())),
        }
    }
}

#[cfg(feature = "serve")]
impl ServeError {
    pub(crate) fn status_code(&self) -> StatusCode {
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
