use crate::engine::registry::ModelRegistry;
use crate::server::handlers::{
    audio_transcriptions, get_openai_model, list_openai_models, live_check, memory_check,
    model_infer, model_load, model_metadata, model_unload, openai_embeddings, ready_check, rerank,
    system_one,
};
use axum::{
    extract::Request,
    http::header::{HeaderValue, CONTENT_TYPE},
    http::Method,
    middleware::{from_fn, Next},
    response::Response,
    routing::{get, post},
    Router,
};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use tower_http::cors::CorsLayer;
use tracing::Instrument;

static REQUEST_SEQUENCE: AtomicU64 = AtomicU64::new(1);

/// Middleware that defaults or normalizes Content-Type to application/json
/// for POST/PUT/PATCH requests when omitted or sent as text/plain or application/x-www-form-urlencoded (e.g. raw curl -d).
async fn default_json_content_type_middleware(mut req: Request, next: Next) -> Response {
    let method = req.method();
    if method == Method::POST || method == Method::PUT || method == Method::PATCH {
        let should_force_json = match req.headers().get(CONTENT_TYPE) {
            None => true,
            Some(val) => {
                if let Ok(s) = val.to_str() {
                    s.is_empty()
                        || s.contains("application/x-www-form-urlencoded")
                        || s.contains("text/plain")
                        || s.contains("*/*")
                } else {
                    true
                }
            }
        };

        if should_force_json {
            req.headers_mut()
                .insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        }
    }

    next.run(req).await
}

async fn request_logging_middleware(req: Request, next: Next) -> Response {
    let request_id = req
        .headers()
        .get("x-request-id")
        .and_then(|value| value.to_str().ok())
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
        .unwrap_or_else(|| {
            let millis = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis();
            let sequence = REQUEST_SEQUENCE.fetch_add(1, Ordering::Relaxed);
            format!("req-{millis:x}-{sequence:x}")
        });
    let method = req.method().clone();
    let path = req.uri().path().to_string();
    let span = tracing::info_span!(
        "http_request",
        request_id = %request_id,
        method = %method,
        path = %path
    );
    let started = Instant::now();
    tracing::info!(
        parent: &span,
        content_length = req
            .headers()
            .get(axum::http::header::CONTENT_LENGTH)
            .and_then(|value| value.to_str().ok())
            .unwrap_or("-"),
        "request started"
    );

    let mut response = next.run(req).instrument(span.clone()).await;
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    let memory = crate::engine::memory::get_memory_snapshot();
    tracing::info!(
        parent: &span,
        status = response.status().as_u16(),
        latency_ms = elapsed_ms,
        rss_mb = memory.process_rss_mb,
        system_available_mb = memory.system_available_mb,
        "request completed"
    );
    if let Ok(value) = HeaderValue::from_str(&request_id) {
        response.headers_mut().insert("x-request-id", value);
    }
    response
}

pub fn create_router(registry: ModelRegistry) -> Router {
    Router::new()
        // OpenAI-compatible Model Discovery
        .route("/v1/models", get(list_openai_models))
        .route("/models", get(list_openai_models))
        .route("/v1/models/:model_name", get(get_openai_model))
        .route("/models/:model_name", get(get_openai_model))
        // Health & Diagnostics
        .route("/v1/health/memory", get(memory_check))
        .route("/health/memory", get(memory_check))
        .route("/v2/health/memory", get(memory_check))
        .route("/v2/health/live", get(live_check))
        .route("/health/live", get(live_check))
        .route("/v2/health/ready", get(ready_check))
        .route("/health/ready", get(ready_check))
        // OpenAI-compatible Embeddings
        .route("/v1/embeddings", post(openai_embeddings))
        .route("/embeddings", post(openai_embeddings))
        // Vector Search Reranking Extension
        .route("/v1/rerank", post(rerank))
        .route("/rerank", post(rerank))
        // TypeSafe Decision / System One compatible decisions
        .route("/v1/decision", post(system_one))
        .route("/decision", post(system_one))
        .route("/v1/decide", post(system_one))
        .route("/decide", post(system_one))
        .route("/v1/systemone", post(system_one))
        .route("/systemone", post(system_one))
        // OpenAI-compatible Audio Transcriptions
        .route("/v1/audio/transcriptions", post(audio_transcriptions))
        // KServe v2 Model Metadata & Raw Inference
        .route("/v2/models/:model_name", get(model_metadata))
        .route("/v2/models/:model_name/infer", post(model_infer))
        // KServe v2 Repository Lifecycle
        .route("/v2/repository/models/:model_name/load", post(model_load))
        .route(
            "/v2/repository/models/:model_name/unload",
            post(model_unload),
        )
        // Global Middlewares
        .layer(from_fn(default_json_content_type_middleware))
        .layer(from_fn(request_logging_middleware))
        .layer(CorsLayer::permissive())
        .with_state(registry)
}
