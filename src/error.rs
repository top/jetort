use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde::Serialize;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum AppError {
    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Model '{0}' not found in registry")]
    ModelNotFound(String),

    #[error("Model '{name}' is not loaded ({state})")]
    ModelNotLoaded { name: String, state: String },

    #[error("Invalid input format or dimensions: {0}")]
    InvalidInput(String),

    #[error("Request validation failed: {0}")]
    Unprocessable(String),

    #[error("Inference execution error: {0}")]
    Inference(String),

    #[error("Tokenizer error: {0}")]
    Tokenizer(String),

    #[error("Context length exceeded: {0}")]
    ContextLengthExceeded(String),

    #[error("Internal runtime error: {0}")]
    Internal(String),

    #[error("{0}")]
    NotImplemented(String),
}

#[derive(Serialize)]
pub struct ErrorResponse {
    pub error: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub state: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retryable: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<String>,
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, message, code, state, retryable) = match &self {
            AppError::ModelNotFound(name) => (
                StatusCode::NOT_FOUND,
                format!("Model '{}' not found", name),
                Some("model_not_found".to_string()),
                None,
                Some(false),
            ),
            AppError::ModelNotLoaded { name, state } => (
                StatusCode::SERVICE_UNAVAILABLE,
                format!("Model '{}' is not loaded", name),
                Some("model_not_loaded".to_string()),
                Some(state.clone()),
                Some(true),
            ),
            AppError::InvalidInput(msg) => (
                StatusCode::BAD_REQUEST,
                format!("Invalid input tensor: {}", msg),
                None,
                None,
                None,
            ),
            AppError::Unprocessable(msg) => (
                StatusCode::UNPROCESSABLE_ENTITY,
                msg.clone(),
                None,
                None,
                None,
            ),
            AppError::Tokenizer(msg) => (
                StatusCode::BAD_REQUEST,
                format!("Tokenizer processing failure: {}", msg),
                None,
                None,
                None,
            ),
            AppError::ContextLengthExceeded(msg) => {
                (StatusCode::BAD_REQUEST, msg.clone(), None, None, None)
            }
            AppError::Config(msg) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Server configuration failure: {}", msg),
                None,
                None,
                None,
            ),
            AppError::Inference(msg) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Inference engine failure: {}", msg),
                None,
                None,
                None,
            ),
            AppError::Internal(msg) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Internal server error: {}", msg),
                None,
                None,
                None,
            ),
            AppError::NotImplemented(msg) => (
                StatusCode::NOT_IMPLEMENTED,
                msg.clone(),
                Some("not_implemented".to_string()),
                None,
                Some(false),
            ),
        };

        if status.is_server_error() {
            tracing::error!(
                status = %status,
                error = %self,
                "Request processing failed with 500 Internal Server Error"
            );
        } else if status.is_client_error() {
            tracing::warn!(
                status = %status,
                error = %self,
                "Request rejected with client error"
            );
        }

        let body = Json(ErrorResponse {
            error: message,
            code,
            state,
            retryable,
            details: Some(self.to_string()),
        });

        let mut response = (status, body).into_response();
        if matches!(self, AppError::ModelNotLoaded { .. }) {
            response.headers_mut().insert(
                axum::http::header::RETRY_AFTER,
                axum::http::HeaderValue::from_static("2"),
            );
        }
        response
    }
}
