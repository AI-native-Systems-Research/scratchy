// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Axum HTTP server with OpenAI-compatible API routes.
//!
//! Provides endpoints:
//! - `POST /v1/chat/completions` — chat completion (streaming + non-streaming)
//! - `POST /v1/completions` — text completion (streaming + non-streaming)
//! - `POST /v1/embeddings` — text embedding
//! - `GET  /v1/models` — list available models
//! - `GET  /health` — health check
//! - `GET  /version` — version info
//!
//! Port of: `vllm/entrypoints/openai/api_server.py` (subset)

use std::convert::Infallible;
use std::sync::Arc;

use axum::extract::{Query, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use tokio_stream::StreamExt;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tower_http::classify::ServerErrorsFailureClass;
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;
use tracing::{Span, info};

use crate::engine::{AsyncEngine, StreamDelta};
use crate::protocol;

// ---------------------------------------------------------------------------
// Server configuration
// ---------------------------------------------------------------------------

/// Configuration for the HTTP server.
pub struct ServerConfig {
    /// Address to bind to (e.g., "0.0.0.0:8000").
    pub bind_address: String,

    /// vLLM version string.
    pub version: String,

    /// Whether CORS is enabled.
    pub cors_enabled: bool,

    /// Whether the `/metrics` endpoint is enabled.
    pub metrics_enabled: bool,

    /// Path to SSL/TLS private key file (PEM format).
    pub ssl_keyfile: Option<String>,

    /// Path to SSL/TLS certificate file (PEM format).
    pub ssl_certfile: Option<String>,

    /// Path to CA certificates file for client certificate verification (PEM).
    pub ssl_ca_certs: Option<String>,

    /// Instant when the process started, for total startup time reporting.
    pub startup_instant: Option<std::time::Instant>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            bind_address: "0.0.0.0:8000".to_string(),
            version: "0.1.0-rust".to_string(),
            cors_enabled: true,
            metrics_enabled: false,
            ssl_keyfile: None,
            ssl_certfile: None,
            ssl_ca_certs: None,
            startup_instant: None,
        }
    }
}

// ---------------------------------------------------------------------------
// App state
// ---------------------------------------------------------------------------

/// Shared application state.
pub struct AppState {
    pub engine: Arc<AsyncEngine>,
    pub config: ServerConfig,
    /// Whether the server is in pooling mode.
    pub is_pooling: bool,
    /// The VllmConfig used to initialize the stack (for `/server_info`).
    pub scratchy_core_config: Option<crate::init::VllmConfig>,
}

// ---------------------------------------------------------------------------
// Router
// ---------------------------------------------------------------------------

/// Build the axum router with all API routes.
pub fn build_router(state: Arc<AppState>) -> Router {
    #[allow(unused_mut)]
    let mut router = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/chat/completions/render", post(render_chat_completion))
        .route("/v1/messages", post(crate::anthropic::messages))
        .route("/v1/responses", post(crate::responses::responses))
        .route("/v1/completions", post(completions))
        .route("/v1/query/execute", post(crate::query::execute_query))
        .route("/v1/embeddings", post(embeddings))
        .route("/v1/models", get(list_models))
        .route("/tokenize", post(tokenize))
        .route("/detokenize", post(detokenize))
        .route("/health", get(health))
        .route("/reset_prefix_cache", post(reset_prefix_cache))
        .route("/sleep", post(sleep_endpoint))
        .route("/wake_up", post(wake_up_endpoint))
        .route("/is_sleeping", get(is_sleeping_endpoint))
        .route("/gpu_memory", get(gpu_memory_endpoint))
        .route("/version", get(version))
        .route("/server_info", get(server_info));

    #[cfg(feature = "metrics")]
    if state.config.metrics_enabled {
        router = router.route("/metrics", get(metrics));
    }

    let mut router = router.with_state(state.clone());

    // Log requests at INFO level: method, URI, status, latency.
    router = router.layer(
        TraceLayer::new_for_http()
            .make_span_with(|request: &http::Request<_>| {
                tracing::info_span!(
                    "request",
                    method = %request.method(),
                    uri = %request.uri(),
                )
            })
            .on_response(
                |response: &http::Response<_>, latency: std::time::Duration, _span: &Span| {
                    info!(status = %response.status(), latency = ?latency, "response");
                },
            )
            .on_failure(
                |error: ServerErrorsFailureClass, latency: std::time::Duration, _span: &Span| {
                    tracing::error!(%error, latency = ?latency, "request failed");
                },
            ),
    );

    if state.config.cors_enabled {
        router = router.layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any),
        );
    }

    router
}

/// Log the available API routes.
fn log_routes(state: &AppState) {
    info!("Available routes are:");
    info!("Route: /v1/chat/completions, Methods: POST");
    info!("Route: /v1/chat/completions/render, Methods: POST");
    info!("Route: /v1/messages, Methods: POST");
    info!("Route: /v1/completions, Methods: POST");
    info!("Route: /v1/query/execute, Methods: POST");
    info!("Route: /v1/embeddings, Methods: POST");
    info!("Route: /v1/models, Methods: GET");
    info!("Route: /health, Methods: GET");
    info!("Route: /reset_prefix_cache, Methods: POST");
    info!("Route: /version, Methods: GET");
    info!("Route: /server_info, Methods: GET");
    if state.config.metrics_enabled {
        info!("Route: /metrics, Methods: GET");
    }
}

/// Start the HTTP server (plain HTTP or HTTPS if SSL cert/key are configured).
#[allow(clippy::needless_return)]
pub async fn serve(state: Arc<AppState>) -> Result<(), Box<dyn std::error::Error>> {
    let router = build_router(state.clone());

    log_routes(&state);

    #[cfg(feature = "tls")]
    if let (Some(certfile), Some(keyfile)) = (&state.config.ssl_certfile, &state.config.ssl_keyfile)
    {
        let tls_config = build_tls_config(certfile, keyfile, state.config.ssl_ca_certs.as_deref())?;
        let addr: std::net::SocketAddr = state.config.bind_address.parse()?;
        info!(
            "vLLM Rust server listening on https://{}",
            state.config.bind_address
        );
        if let Some(start) = state.config.startup_instant {
            info!(
                "Application startup complete. ({:.2}s)",
                start.elapsed().as_secs_f64()
            );
        } else {
            info!("Application startup complete.");
        }
        axum_server::bind_rustls(addr, tls_config)
            .serve(router.into_make_service())
            .await?;
        return Ok(());
    }

    #[cfg(not(feature = "tls"))]
    if state.config.ssl_certfile.is_some() || state.config.ssl_keyfile.is_some() {
        return Err("TLS support is not enabled (compile with --features tls)".into());
    }

    let listener = tokio::net::TcpListener::bind(&state.config.bind_address).await?;
    info!(
        "vLLM Rust server listening on http://{}",
        state.config.bind_address
    );
    if let Some(start) = state.config.startup_instant {
        info!(
            "Application startup complete. ({:.2}s)",
            start.elapsed().as_secs_f64()
        );
    } else {
        info!("Application startup complete.");
    }
    axum::serve(listener, router)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    Ok(())
}

/// Resolve when the process receives a termination signal (Ctrl-C / SIGINT
/// or SIGTERM). Used by [`serve`] as the `axum::serve` graceful-shutdown
/// future so the server loop RETURNS on a catchable signal instead of being
/// terminated by the default disposition — which lets the engine/worker
/// teardown run (`InprocClient::Drop -> worker.shutdown()`), ending the Metal
/// residency set and releasing system-wired GPU memory deterministically.
///
/// Note: `kill -9` (SIGKILL) is uncatchable and bypasses this entirely — the
/// only mitigation there is to not `kill -9` a busy server.
async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };

    #[cfg(unix)]
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut sig) => {
                sig.recv().await;
            }
            Err(_) => std::future::pending::<()>().await,
        }
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {}
        _ = terminate => {}
    }

    info!("Termination signal received — shutting down gracefully (releasing GPU residency)");
}

/// Build a rustls [`axum_server::tls_rustls::RustlsConfig`] from PEM file paths.
#[cfg(feature = "tls")]
fn build_tls_config(
    certfile: &str,
    keyfile: &str,
    ca_certs: Option<&str>,
) -> Result<axum_server::tls_rustls::RustlsConfig, Box<dyn std::error::Error>> {
    use std::io::BufReader;

    let cert_pem = std::fs::read(certfile)
        .map_err(|e| format!("failed to read ssl_certfile '{}': {}", certfile, e))?;
    let key_pem = std::fs::read(keyfile)
        .map_err(|e| format!("failed to read ssl_keyfile '{}': {}", keyfile, e))?;

    // Parse server cert chain and private key.
    let server_certs: Vec<_> = rustls_pemfile::certs(&mut BufReader::new(cert_pem.as_slice()))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| format!("failed to parse server certificates: {}", e))?;
    let server_key = rustls_pemfile::private_key(&mut BufReader::new(key_pem.as_slice()))
        .map_err(|e| format!("failed to parse private key: {}", e))?
        .ok_or("no private key found in ssl_keyfile")?;

    let provider = rustls::crypto::ring::default_provider().into();

    let tls_config = if let Some(ca_path) = ca_certs {
        info!("Client certificate verification enabled (CA: {})", ca_path);
        let ca_pem = std::fs::read(ca_path)
            .map_err(|e| format!("failed to read ssl_ca_certs '{}': {}", ca_path, e))?;

        let ca_certs_parsed: Vec<_> = rustls_pemfile::certs(&mut BufReader::new(ca_pem.as_slice()))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("failed to parse CA certificates: {}", e))?;

        let mut root_store = rustls::RootCertStore::empty();
        for cert in ca_certs_parsed {
            root_store.add(cert)?;
        }
        let client_verifier = rustls::server::WebPkiClientVerifier::builder(root_store.into())
            .build()
            .map_err(|e| format!("failed to build client verifier: {}", e))?;

        rustls::ServerConfig::builder_with_provider(provider)
            .with_safe_default_protocol_versions()
            .map_err(|e| format!("failed to set protocol versions: {}", e))?
            .with_client_cert_verifier(client_verifier)
            .with_single_cert(server_certs, server_key)
            .map_err(|e| format!("failed to build TLS config with mTLS: {}", e))?
    } else {
        rustls::ServerConfig::builder_with_provider(provider)
            .with_safe_default_protocol_versions()
            .map_err(|e| format!("failed to set protocol versions: {}", e))?
            .with_no_client_auth()
            .with_single_cert(server_certs, server_key)
            .map_err(|e| format!("failed to build TLS config: {}", e))?
    };

    Ok(axum_server::tls_rustls::RustlsConfig::from_config(
        std::sync::Arc::new(tls_config),
    ))
}

// ---------------------------------------------------------------------------
// Route handlers
// ---------------------------------------------------------------------------

/// POST /v1/chat/completions
async fn chat_completions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(request): Json<protocol::ChatCompletionRequest>,
) -> Response {
    info!(
        "POST /v1/chat/completions: model={:?}, max_tokens={:?}, max_completion_tokens={:?}, stream={}",
        request.model, request.max_tokens, request.max_completion_tokens, request.stream
    );
    if request.stream {
        // Streaming responses skip ORCA headers (consistent with Python vLLM).
        match state.engine.chat_completion_stream(request).await {
            Ok((request_id, model, rx)) => {
                stream_chat_response(request_id, model, rx).into_response()
            }
            Err(e) => e.into_response(),
        }
    } else {
        match state.engine.chat_completion(request).await {
            Ok(response) => attach_orca_header(&headers, Json(response).into_response()),
            Err(e) => e.into_response(),
        }
    }
}

/// POST /v1/chat/completions/render
async fn render_chat_completion(
    State(state): State<Arc<AppState>>,
    Json(request): Json<protocol::ChatCompletionRequest>,
) -> Response {
    info!(
        "POST /v1/chat/completions/render: model={:?}",
        request.model
    );
    match state.engine.render_chat_completion(request) {
        Ok(response) => Json(response).into_response(),
        Err(e) => e.into_response(),
    }
}

/// POST /v1/completions
async fn completions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(request): Json<protocol::CompletionRequest>,
) -> Response {
    if request.stream {
        let include_usage = request
            .stream_options
            .as_ref()
            .and_then(|o| o.include_usage)
            .unwrap_or(false);
        match state.engine.completion_stream(request).await {
            Ok((request_id, model, rx)) => {
                stream_completion_response(request_id, model, rx, include_usage).into_response()
            }
            Err(e) => e.into_response(),
        }
    } else {
        match state.engine.completion(request).await {
            Ok(response) => attach_orca_header(&headers, Json(response).into_response()),
            Err(e) => e.into_response(),
        }
    }
}

/// POST /v1/embeddings
async fn embeddings(
    State(state): State<Arc<AppState>>,
    Json(request): Json<protocol::EmbeddingRequest>,
) -> Response {
    match state.engine.embeddings(request).await {
        Ok(response) => Json(response).into_response(),
        Err(e) => e.into_response(),
    }
}

/// GET /v1/models
async fn list_models(State(state): State<Arc<AppState>>) -> Json<protocol::ModelList> {
    let mut card = protocol::ModelCard::new(state.engine.model_name().to_string());
    card.max_model_len = Some(state.engine.max_model_len());
    Json(protocol::ModelList::new(vec![card]))
}

/// GET /health
async fn health() -> Json<protocol::HealthResponse> {
    Json(protocol::HealthResponse { status: "ok" })
}

/// POST /reset_prefix_cache
///
/// Reset the prefix cache. Returns 200 on success.
/// Query params: `reset_running_requests` (ignored, for Python compat).
async fn reset_prefix_cache(
    State(state): State<Arc<AppState>>,
) -> Result<StatusCode, (StatusCode, String)> {
    info!("Resetting prefix cache...");
    match state.engine.reset_prefix_cache().await {
        Ok(_success) => Ok(StatusCode::OK),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to reset prefix cache: {e}"),
        )),
    }
}

/// POST /sleep
///
/// Put the engine to sleep, freeing GPU memory.
/// Body: `{"level": 1}` (default: 1)
async fn sleep_endpoint(
    State(state): State<Arc<AppState>>,
    body: Option<Json<serde_json::Value>>,
) -> Result<StatusCode, (StatusCode, String)> {
    let level = body
        .and_then(|Json(v)| v.get("level").and_then(|l| l.as_u64()))
        .unwrap_or(1) as u32;
    info!("Sleep requested (level {level})");
    state.engine.sleep(level).await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Sleep failed: {e}"),
        )
    })?;
    Ok(StatusCode::OK)
}

/// POST /wake_up
///
/// Wake the engine from sleep.
/// Body: `{"tags": null}` or `{"tags": ["weights", "kv_cache"]}`
async fn wake_up_endpoint(
    State(state): State<Arc<AppState>>,
    body: Option<Json<serde_json::Value>>,
) -> Result<StatusCode, (StatusCode, String)> {
    let tags: Option<Vec<String>> = body.and_then(|Json(v)| {
        v.get("tags")
            .and_then(|t| serde_json::from_value(t.clone()).ok())
    });
    info!("Wake up requested (tags: {:?})", tags);
    state.engine.wake_up(tags).await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Wake up failed: {e}"),
        )
    })?;
    Ok(StatusCode::OK)
}

/// GET /is_sleeping
///
/// Returns whether the engine is currently sleeping.
async fn is_sleeping_endpoint(
    State(state): State<Arc<AppState>>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let sleeping = state.engine.is_sleeping().await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to query sleep state: {e}"),
        )
    })?;
    Ok(Json(serde_json::json!({ "is_sleeping": sleeping })))
}

/// GET /gpu_memory
///
/// Returns current GPU memory usage (for testing sleep/wake).
async fn gpu_memory_endpoint() -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    use scratchy_serving_worker::worker_factory::WorkerFactory;
    // Query whichever backend is registered for this build; backends that can't
    // report device memory (or a no-GPU build) yield zeros.
    let (free, total) = inventory::iter::<&dyn WorkerFactory>()
        .find_map(|f| f.device_memory_free_total())
        .unwrap_or((0, 0));
    let used = total.saturating_sub(free);
    Ok(Json(serde_json::json!({
        "used_bytes": used,
        "free_bytes": free,
        "total_bytes": total,
    })))
}

/// GET /version
async fn version(State(state): State<Arc<AppState>>) -> Json<protocol::VersionResponse> {
    Json(protocol::VersionResponse {
        version: state.config.version.clone(),
    })
}

/// GET /server_info — return server configuration, environment variables, and system info.
///
/// Mirrors Python vLLM's `/server_info` endpoint. Accepts an optional
/// `?config_format=text|json` query parameter (default: `"text"`).
async fn server_info(
    State(state): State<Arc<AppState>>,
    Query(params): Query<protocol::ServerInfoParams>,
) -> Json<protocol::ServerInfoResponse> {
    // scratchy_core_config: either a human-readable string or structured JSON.
    let scratchy_core_config = match state.scratchy_core_config {
        Some(ref cfg) => match params.config_format.as_deref().unwrap_or("text") {
            "json" => serde_json::to_value(cfg).unwrap_or(serde_json::Value::Null),
            _ => serde_json::Value::String(format!("{cfg:#?}")),
        },
        None => serde_json::Value::Null,
    };

    // vllm_env: collect VLLM_* env vars, excluding secrets.
    let vllm_env = {
        let secret_terms = ["secret", "token", "api", "access", "password", "key"];
        let mut map = serde_json::Map::new();
        for (k, v) in std::env::vars() {
            if !k.starts_with("VLLM_") {
                continue;
            }
            if secret_terms.iter().any(|t| k.to_lowercase().contains(t)) {
                continue;
            }
            map.insert(k, serde_json::Value::String(v));
        }
        serde_json::Value::Object(map)
    };

    // system_env: basic system info (matches collect-env output fields).
    let system_env = {
        let mut map = serde_json::Map::new();
        map.insert(
            "os".to_string(),
            serde_json::Value::String(format!(
                "{} ({})",
                std::env::consts::OS,
                std::env::consts::ARCH
            )),
        );
        map.insert(
            "vllm_version".to_string(),
            serde_json::Value::String(env!("CARGO_PKG_VERSION").to_string()),
        );
        serde_json::Value::Object(map)
    };

    Json(protocol::ServerInfoResponse {
        scratchy_core_config,
        vllm_env,
        system_env,
    })
}

/// POST /tokenize
async fn tokenize(
    State(state): State<Arc<AppState>>,
    Json(request): Json<protocol::TokenizeRequest>,
) -> Response {
    let tokenizer = match state.engine.tokenizer() {
        Some(t) => t,
        None => {
            return Json(protocol::ErrorResponse::new(
                "Tokenizer not available",
                "ServiceUnavailableError",
                503,
            ))
            .into_response();
        }
    };

    // Determine the text to tokenize: either from prompt or chat messages.
    let text = if let Some(prompt) = &request.prompt {
        prompt.clone()
    } else if let Some(messages) = &request.messages {
        {
            let template = match state.engine.chat_template() {
                Some(t) => t,
                None => {
                    return Json(protocol::ErrorResponse::new(
                        "Chat-style tokenize requires a chat template",
                        "BadRequestError",
                        400,
                    ))
                    .into_response();
                }
            };
            let message_values: Vec<serde_json::Value> = messages
                .iter()
                .map(|msg| serde_json::to_value(msg).unwrap_or_default())
                .collect();
            match template.apply(&message_values, request.add_generation_prompt, None) {
                Ok(text) => text,
                Err(e) => {
                    return Json(protocol::ErrorResponse::new(
                        format!("Chat template rendering failed: {e}"),
                        "BadRequestError",
                        400,
                    ))
                    .into_response();
                }
            }
        }
    } else {
        return Json(protocol::ErrorResponse::new(
            "Either 'prompt' or 'messages' must be provided",
            "BadRequestError",
            400,
        ))
        .into_response();
    };

    let add_special = request
        .add_special_tokens
        .unwrap_or(request.prompt.is_some());

    match tokenizer.encode(&text, add_special) {
        Ok(token_ids) => {
            let token_strs = if request.return_token_strs.unwrap_or(false) {
                Some(
                    token_ids
                        .iter()
                        .map(|&id| tokenizer.id_to_token(id).unwrap_or_default())
                        .collect(),
                )
            } else {
                None
            };
            Json(protocol::TokenizeResponse {
                count: token_ids.len(),
                max_model_len: state.engine.max_model_len(),
                tokens: token_ids,
                token_strs,
            })
            .into_response()
        }
        Err(e) => Json(protocol::ErrorResponse::new(
            format!("Tokenization failed: {e}"),
            "InternalError",
            500,
        ))
        .into_response(),
    }
}

/// POST /detokenize
async fn detokenize(
    State(state): State<Arc<AppState>>,
    Json(request): Json<protocol::DetokenizeRequest>,
) -> Response {
    let tokenizer = match state.engine.tokenizer() {
        Some(t) => t,
        None => {
            return Json(protocol::ErrorResponse::new(
                "Tokenizer not available",
                "ServiceUnavailableError",
                503,
            ))
            .into_response();
        }
    };

    match tokenizer.decode(&request.tokens, false) {
        Ok(text) => Json(protocol::DetokenizeResponse { prompt: text }).into_response(),
        Err(e) => Json(protocol::ErrorResponse::new(
            format!("Detokenization failed: {e}"),
            "InternalError",
            500,
        ))
        .into_response(),
    }
}

/// GET /metrics — Prometheus metrics endpoint.
#[cfg(feature = "metrics")]
async fn metrics() -> String {
    crate::metrics::VllmMetrics::global().encode()
}

// ---------------------------------------------------------------------------
// ORCA header helper
// ---------------------------------------------------------------------------

/// If the request includes the ORCA opt-in header, attach the load-metrics
/// response header. Otherwise return the response unchanged.
#[cfg(feature = "metrics")]
fn attach_orca_header(request_headers: &HeaderMap, mut response: Response) -> Response {
    if let Some(format_value) = request_headers.get(crate::orca::ORCA_REQUEST_HEADER)
        && let Ok(format_str) = format_value.to_str()
        && let Some((name, value)) = crate::orca::orca_header(format_str)
        && let Ok(hv) = axum::http::HeaderValue::from_str(&value)
    {
        response.headers_mut().insert(
            axum::http::HeaderName::from_bytes(name.as_bytes()).expect("valid header name"),
            hv,
        );
    }
    response
}

/// If the request includes the ORCA opt-in header, attach the load-metrics
/// response header. Otherwise return the response unchanged.
#[cfg(not(feature = "metrics"))]
fn attach_orca_header(_request_headers: &HeaderMap, response: Response) -> Response {
    response
}

// ---------------------------------------------------------------------------
// SSE streaming
// ---------------------------------------------------------------------------

/// Build an SSE stream for chat completions.
fn stream_chat_response(
    request_id: String,
    model: String,
    rx: tokio::sync::mpsc::UnboundedReceiver<StreamDelta>,
) -> Sse<impl tokio_stream::Stream<Item = Result<Event, Infallible>>> {
    let stream = UnboundedReceiverStream::new(rx).map(move |delta| {
        // Determine finish reason — override to "tool_calls" if tool call deltas present.
        let has_tool_calls = delta.tool_call_deltas.is_some();
        let finish_reason_str = delta.finish_reason.map(|r| {
            if has_tool_calls {
                "tool_calls".to_string()
            } else {
                r.to_string()
            }
        });

        // Convert tool call deltas to protocol JSON values.
        let tool_calls_json: Option<Vec<serde_json::Value>> =
            delta.tool_call_deltas.as_ref().map(|deltas| {
                deltas
                    .iter()
                    .map(|tc| {
                        let mut obj = serde_json::Map::new();
                        obj.insert(
                            "index".to_string(),
                            serde_json::Value::Number(tc.index.into()),
                        );
                        if let Some(ref id) = tc.id {
                            obj.insert("id".to_string(), serde_json::Value::String(id.clone()));
                        }
                        if let Some(ref ct) = tc.call_type {
                            obj.insert("type".to_string(), serde_json::Value::String(ct.clone()));
                        }
                        let mut func = serde_json::Map::new();
                        if let Some(ref name) = tc.function_name {
                            func.insert(
                                "name".to_string(),
                                serde_json::Value::String(name.clone()),
                            );
                        }
                        if let Some(ref args) = tc.function_arguments {
                            func.insert(
                                "arguments".to_string(),
                                serde_json::Value::String(args.clone()),
                            );
                        }
                        if !func.is_empty() {
                            obj.insert("function".to_string(), serde_json::Value::Object(func));
                        }
                        serde_json::Value::Object(obj)
                    })
                    .collect()
            });

        // When tool calls are present, suppress content.
        let (content, tool_calls) = if tool_calls_json.is_some() {
            (None, tool_calls_json)
        } else {
            // Use detokenized text if available, otherwise fall back to placeholders.
            let text = delta.text.unwrap_or_else(|| {
                use std::fmt::Write;
                let mut s = String::new();
                for id in &delta.new_token_ids {
                    let _ = write!(s, "<token_{id}>");
                }
                s
            });
            let content = if text.is_empty() { None } else { Some(text) };
            (content, None)
        };

        let chunk = protocol::ChatCompletionStreamResponse::new(
            format!("chatcmpl-{}", request_id),
            model.clone(),
            vec![protocol::ChatCompletionResponseStreamChoice {
                index: delta.index,
                delta: protocol::DeltaMessage {
                    role: None,
                    content,
                    reasoning: delta.reasoning,
                    tool_calls,
                },
                logprobs: delta.logprobs.as_ref().map(|lps| {
                    let content: Vec<protocol::ChatCompletionLogProbsContent> = lps
                        .iter()
                        .map(|lp| {
                            let top: Vec<protocol::ChatCompletionLogProb> = lp
                                .top_logprobs
                                .iter()
                                .map(|tlp| protocol::ChatCompletionLogProb {
                                    token: format!("<token_{}>", tlp.token_id),
                                    logprob: tlp.logprob as f64,
                                    bytes: None,
                                })
                                .collect();
                            protocol::ChatCompletionLogProbsContent {
                                token: format!("<token_{}>", lp.sampled.token_id),
                                logprob: lp.sampled.logprob as f64,
                                bytes: None,
                                top_logprobs: top,
                            }
                        })
                        .collect();
                    protocol::ChatCompletionLogProbs {
                        content: Some(content),
                    }
                }),
                finish_reason: finish_reason_str,
                stop_reason: delta.stop_reason.map(|sr| match sr {
                    scratchy_core_common::StopReason::Token(id) => {
                        serde_json::Value::Number(id.into())
                    }
                    scratchy_core_common::StopReason::String(s) => serde_json::Value::String(s),
                }),
            }],
        );

        let data = serde_json::to_string(&chunk).unwrap_or_default();
        Ok(Event::default().data(data))
    });

    // Append [DONE] sentinel after the stream ends.
    let done_stream = tokio_stream::once(Ok(Event::default().data("[DONE]")));
    let full_stream = stream.chain(done_stream);

    Sse::new(full_stream).keep_alive(KeepAlive::default())
}

/// Build an SSE stream for text completions.
fn stream_completion_response(
    request_id: String,
    model: String,
    rx: tokio::sync::mpsc::UnboundedReceiver<StreamDelta>,
    include_usage: bool,
) -> Sse<impl tokio_stream::Stream<Item = Result<Event, Infallible>>> {
    let total_completion_tokens = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));
    let tc = total_completion_tokens.clone();
    let req_id = request_id.clone();
    let model2 = model.clone();

    let stream = UnboundedReceiverStream::new(rx).map(move |delta| {
        let token_count = delta.new_token_ids.len() as u32;
        tc.fetch_add(token_count, std::sync::atomic::Ordering::Relaxed);

        let finish_reason_str = delta.finish_reason.map(|r| r.to_string());

        let text = delta.text.unwrap_or_else(|| {
            use std::fmt::Write;
            let mut s = String::new();
            for id in &delta.new_token_ids {
                let _ = write!(s, "<token_{id}>");
            }
            s
        });

        let chunk = protocol::CompletionStreamResponse::new(
            format!("cmpl-{}", req_id),
            model.clone(),
            vec![protocol::CompletionResponseStreamChoice {
                index: delta.index,
                text,
                logprobs: None,
                finish_reason: finish_reason_str,
                stop_reason: delta.stop_reason.map(|sr| match sr {
                    scratchy_core_common::StopReason::Token(id) => {
                        serde_json::Value::Number(id.into())
                    }
                    scratchy_core_common::StopReason::String(s) => serde_json::Value::String(s),
                }),
            }],
        );

        let data = serde_json::to_string(&chunk).unwrap_or_default();
        Ok(Event::default().data(data))
    });

    // After all deltas, optionally emit a usage chunk, then [DONE].
    let usage_stream: Box<
        dyn tokio_stream::Stream<Item = Result<Event, Infallible>> + Send + Unpin,
    > = if include_usage {
        let req_id2 = request_id.clone();
        Box::new(tokio_stream::once(()).map(move |()| {
            let completion_tokens =
                total_completion_tokens.load(std::sync::atomic::Ordering::Relaxed);
            let usage = protocol::UsageInfo {
                prompt_tokens: 0,
                completion_tokens: Some(completion_tokens),
                total_tokens: completion_tokens,
                prompt_tokens_details: None,
            };
            let mut chunk = protocol::CompletionStreamResponse::new(
                format!("cmpl-{}", req_id2),
                model2.clone(),
                vec![],
            );
            chunk.usage = Some(usage);
            let data = serde_json::to_string(&chunk).unwrap_or_default();
            Ok(Event::default().data(data))
        }))
    } else {
        Box::new(tokio_stream::empty())
    };

    let done_stream = tokio_stream::once(Ok(Event::default().data("[DONE]")));
    let full_stream = stream.chain(usage_stream).chain(done_stream);

    Sse::new(full_stream).keep_alive(KeepAlive::default())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::StatusCode;
    use http::Request;
    use http_body_util::BodyExt;
    use scratchy_core_config::{SchedulerConfig, SchedulerPolicy};
    use scratchy_serving_engine::core_client::InprocClient;
    use scratchy_serving_engine::engine_core::EngineCoreConfig;
    use scratchy_serving_engine::executor::NoopExecutor;
    use tower::ServiceExt;

    fn make_test_state() -> Arc<AppState> {
        let engine_config = EngineCoreConfig {
            hybrid_kv: None,
            scheduler_config: SchedulerConfig {
                max_num_batched_tokens: 8192,
                max_num_seqs: 256,
                max_num_scheduled_tokens: None,
                policy: SchedulerPolicy::Fcfs,
                enable_chunked_prefill: true,
                long_prefill_token_threshold: 0,
                ..Default::default()
            },
            max_model_len: 4096,
            num_gpu_blocks: 1024,
            block_size: 16,
            engine_index: 0,
            async_scheduling: false,
            use_spec_decode: false,
            proposer_config: None,
            eos_token_ids: vec![],
            is_pooling: false,
            enable_prefix_caching: false,
        };
        let executor = Box::new(NoopExecutor::new(1024));
        let client = Box::new(InprocClient::new(engine_config, executor));
        let engine = Arc::new(AsyncEngine::new(client, "test-model".to_string(), 4096));

        Arc::new(AppState {
            engine,
            config: ServerConfig::default(),
            is_pooling: false,
            scratchy_core_config: None,
        })
    }

    #[tokio::test]
    async fn test_health_endpoint() {
        let state = make_test_state();
        let app = build_router(state);

        let request = Request::builder()
            .uri("/health")
            .body(Body::empty())
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let body = response.into_body().collect().await.unwrap().to_bytes();
        let parsed: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(parsed["status"], "ok");
    }

    #[tokio::test]
    async fn test_version_endpoint() {
        let state = make_test_state();
        let app = build_router(state);

        let request = Request::builder()
            .uri("/version")
            .body(Body::empty())
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let body = response.into_body().collect().await.unwrap().to_bytes();
        let parsed: protocol::VersionResponse = serde_json::from_slice(&body).unwrap();
        assert_eq!(parsed.version, "0.1.0-rust");
    }

    #[tokio::test]
    async fn test_server_info_text_format() {
        let state = make_test_state();
        let app = build_router(state);

        let request = Request::builder()
            .uri("/server_info")
            .body(Body::empty())
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let body = response.into_body().collect().await.unwrap().to_bytes();
        let parsed: protocol::ServerInfoResponse = serde_json::from_slice(&body).unwrap();

        // scratchy_core_config is null when no config is provided (test state has None).
        assert!(parsed.scratchy_core_config.is_null());
        // vllm_env should be an object.
        assert!(parsed.vllm_env.is_object());
        // system_env should contain os and vllm_version.
        let sys = parsed.system_env.as_object().unwrap();
        assert!(sys.contains_key("os"));
        assert!(sys.contains_key("vllm_version"));
    }

    fn make_test_state_with_config() -> Arc<AppState> {
        use crate::init::VllmConfig;

        let engine_config = EngineCoreConfig {
            hybrid_kv: None,
            scheduler_config: SchedulerConfig {
                max_num_batched_tokens: 8192,
                max_num_seqs: 256,
                max_num_scheduled_tokens: None,
                policy: SchedulerPolicy::Fcfs,
                enable_chunked_prefill: true,
                long_prefill_token_threshold: 0,
                ..Default::default()
            },
            max_model_len: 4096,
            num_gpu_blocks: 1024,
            block_size: 16,
            engine_index: 0,
            async_scheduling: false,
            use_spec_decode: false,
            proposer_config: None,
            eos_token_ids: vec![],
            is_pooling: false,
            enable_prefix_caching: false,
        };
        let executor = Box::new(NoopExecutor::new(1024));
        let client = Box::new(InprocClient::new(engine_config, executor));
        let engine = Arc::new(AsyncEngine::new(client, "test-model".to_string(), 4096));

        Arc::new(AppState {
            engine,
            config: ServerConfig::default(),
            is_pooling: false,
            scratchy_core_config: Some(VllmConfig {
                model: "test-model".to_string(),
                ..VllmConfig::default()
            }),
        })
    }

    #[tokio::test]
    async fn test_server_info_with_config_text() {
        let state = make_test_state_with_config();
        let app = build_router(state);

        let request = Request::builder()
            .uri("/server_info")
            .body(Body::empty())
            .unwrap();
        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let body = response.into_body().collect().await.unwrap().to_bytes();
        let parsed: protocol::ServerInfoResponse = serde_json::from_slice(&body).unwrap();
        // Text format (default): scratchy_core_config should be a debug string.
        assert!(parsed.scratchy_core_config.is_string());
        assert!(
            parsed
                .scratchy_core_config
                .as_str()
                .unwrap()
                .contains("test-model")
        );
    }

    #[tokio::test]
    async fn test_server_info_with_config_json() {
        let state = make_test_state_with_config();
        let app = build_router(state);

        let request = Request::builder()
            .uri("/server_info?config_format=json")
            .body(Body::empty())
            .unwrap();
        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let body = response.into_body().collect().await.unwrap().to_bytes();
        let parsed: protocol::ServerInfoResponse = serde_json::from_slice(&body).unwrap();
        // JSON format: scratchy_core_config should be a structured object.
        assert!(parsed.scratchy_core_config.is_object());
        assert_eq!(parsed.scratchy_core_config["model"], "test-model");
        // hf_token should NOT appear (skip_serializing).
        assert!(parsed.scratchy_core_config.get("hf_token").is_none());
    }

    #[tokio::test]
    async fn test_server_info_env_filters_secrets() {
        // Set a secret and a visible VLLM_ var.
        unsafe {
            std::env::set_var("VLLM_TEST_API_KEY_X", "secret-value");
            std::env::set_var("VLLM_TEST_NORMAL_VAR", "visible");
        }

        let state = make_test_state();
        let app = build_router(state);

        let request = Request::builder()
            .uri("/server_info")
            .body(Body::empty())
            .unwrap();
        let response = app.oneshot(request).await.unwrap();
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let parsed: protocol::ServerInfoResponse = serde_json::from_slice(&body).unwrap();

        let env_obj = parsed.vllm_env.as_object().unwrap();
        assert!(
            !env_obj.contains_key("VLLM_TEST_API_KEY_X"),
            "secrets should be filtered"
        );
        assert_eq!(
            env_obj.get("VLLM_TEST_NORMAL_VAR").and_then(|v| v.as_str()),
            Some("visible"),
            "non-secret VLLM_ vars should be present"
        );

        unsafe {
            std::env::remove_var("VLLM_TEST_API_KEY_X");
            std::env::remove_var("VLLM_TEST_NORMAL_VAR");
        }
    }

    #[tokio::test]
    async fn test_models_endpoint() {
        let state = make_test_state();
        let app = build_router(state);

        let request = Request::builder()
            .uri("/v1/models")
            .body(Body::empty())
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let body = response.into_body().collect().await.unwrap().to_bytes();
        let parsed: protocol::ModelList = serde_json::from_slice(&body).unwrap();
        assert_eq!(parsed.object, "list");
        assert_eq!(parsed.data.len(), 1);
        assert_eq!(parsed.data[0].id, "test-model");
        assert_eq!(parsed.data[0].max_model_len, Some(4096));
    }

    #[tokio::test]
    async fn test_chat_completions_bad_json() {
        let state = make_test_state();
        let app = build_router(state);

        let request = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from("not json"))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        // Axum returns 400 for malformed JSON.
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_chat_completions_missing_messages() {
        let state = make_test_state();
        let app = build_router(state);

        let request = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"model": "test"}"#))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        // Missing required field "messages" → 422
        assert_eq!(
            response.status(),
            StatusCode::UNPROCESSABLE_ENTITY,
            "actual status: {}",
            response.status()
        );
    }

    #[tokio::test]
    async fn test_completions_bad_json() {
        let state = make_test_state();
        let app = build_router(state);

        let request = Request::builder()
            .method("POST")
            .uri("/v1/completions")
            .header("content-type", "application/json")
            .body(Body::from("{invalid"))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        // Axum returns 400 for malformed JSON.
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[cfg(feature = "metrics")]
    fn make_test_state_with_metrics() -> Arc<AppState> {
        let engine_config = EngineCoreConfig {
            hybrid_kv: None,
            scheduler_config: SchedulerConfig {
                max_num_batched_tokens: 8192,
                max_num_seqs: 256,
                max_num_scheduled_tokens: None,
                policy: SchedulerPolicy::Fcfs,
                enable_chunked_prefill: true,
                long_prefill_token_threshold: 0,
                ..Default::default()
            },
            max_model_len: 4096,
            num_gpu_blocks: 1024,
            block_size: 16,
            engine_index: 0,
            async_scheduling: false,
            use_spec_decode: false,
            proposer_config: None,
            eos_token_ids: vec![],
            is_pooling: false,
            enable_prefix_caching: false,
        };
        let executor = Box::new(NoopExecutor::new(1024));
        let client = Box::new(InprocClient::new(engine_config, executor));
        let engine = Arc::new(AsyncEngine::new(client, "test-model".to_string(), 4096));

        Arc::new(AppState {
            engine,
            config: ServerConfig {
                metrics_enabled: true,
                ..ServerConfig::default()
            },
            is_pooling: false,
            scratchy_core_config: None,
        })
    }

    #[cfg(feature = "metrics")]
    #[tokio::test]
    async fn test_metrics_endpoint_enabled() {
        let state = make_test_state_with_metrics();
        let app = build_router(state);

        let request = Request::builder()
            .uri("/metrics")
            .body(Body::empty())
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let body = response.into_body().collect().await.unwrap().to_bytes();
        let text = String::from_utf8_lossy(&body);
        assert!(text.contains("vllm_requests_total"));
    }

    #[tokio::test]
    async fn test_metrics_endpoint_disabled() {
        let state = make_test_state(); // default has metrics_enabled: false
        let app = build_router(state);

        let request = Request::builder()
            .uri("/metrics")
            .body(Body::empty())
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_not_found() {
        let state = make_test_state();
        let app = build_router(state);

        let request = Request::builder()
            .uri("/nonexistent")
            .body(Body::empty())
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    // -- ORCA header attachment tests --

    #[cfg(feature = "metrics")]
    #[test]
    fn test_attach_orca_header_with_text_format() {
        use axum::http::{HeaderMap, HeaderValue};
        use axum::response::IntoResponse;

        let mut request_headers = HeaderMap::new();
        request_headers.insert(
            "endpoint-load-metrics-format",
            HeaderValue::from_static("TEXT"),
        );

        let response = "test body".into_response();
        let response = attach_orca_header(&request_headers, response);

        assert!(
            response.headers().contains_key("endpoint-load-metrics"),
            "Response should contain ORCA header when TEXT format requested"
        );
        let val = response.headers()["endpoint-load-metrics"]
            .to_str()
            .unwrap();
        assert!(val.contains("kv_cache_usage_perc="));
        assert!(val.contains("num_requests_waiting="));
    }

    #[cfg(feature = "metrics")]
    #[test]
    fn test_attach_orca_header_with_json_format() {
        use axum::http::{HeaderMap, HeaderValue};
        use axum::response::IntoResponse;

        let mut request_headers = HeaderMap::new();
        request_headers.insert(
            "endpoint-load-metrics-format",
            HeaderValue::from_static("JSON"),
        );

        let response = "test body".into_response();
        let response = attach_orca_header(&request_headers, response);

        assert!(response.headers().contains_key("endpoint-load-metrics"));
        let val = response.headers()["endpoint-load-metrics"]
            .to_str()
            .unwrap();
        let parsed: serde_json::Value = serde_json::from_str(val).unwrap();
        assert!(parsed["named_metrics"]["kv_cache_usage_perc"].is_number());
    }

    #[test]
    fn test_attach_orca_header_absent_when_not_requested() {
        use axum::http::HeaderMap;
        use axum::response::IntoResponse;

        let request_headers = HeaderMap::new(); // No ORCA header.
        let response = "test body".into_response();
        let response = attach_orca_header(&request_headers, response);

        assert!(
            !response.headers().contains_key("endpoint-load-metrics"),
            "Response should NOT contain ORCA header when not requested"
        );
    }

    // -- TLS tests --

    #[cfg(feature = "tls")]
    /// Generate a self-signed CA certificate and a server certificate signed by it.
    /// Returns (ca_cert_pem, server_cert_pem, server_key_pem).
    fn generate_test_certs() -> (String, String, String) {
        use rcgen::{CertificateParams, Issuer, KeyPair};

        // Generate CA.
        let ca_key = KeyPair::generate().unwrap();
        let mut ca_params = CertificateParams::new(vec!["Test CA".to_string()]).unwrap();
        ca_params.is_ca = rcgen::IsCa::Ca(rcgen::BasicConstraints::Unconstrained);
        let ca_cert = ca_params.self_signed(&ca_key).unwrap();

        // Generate server cert signed by CA.
        let server_key = KeyPair::generate().unwrap();
        let server_params =
            CertificateParams::new(vec!["localhost".to_string(), "127.0.0.1".to_string()]).unwrap();
        let issuer = Issuer::from_params(&ca_params, &ca_key);
        let server_cert = server_params.signed_by(&server_key, &issuer).unwrap();

        (ca_cert.pem(), server_cert.pem(), server_key.serialize_pem())
    }

    #[cfg(feature = "tls")]
    /// Write PEM content to a temp file and return the path.
    fn write_pem_file(dir: &tempfile::TempDir, name: &str, pem: &str) -> String {
        let path = dir.path().join(name);
        std::fs::write(&path, pem).unwrap();
        path.to_str().unwrap().to_string()
    }

    #[cfg(feature = "tls")]
    fn make_tls_test_state(
        certfile: String,
        keyfile: String,
        ca_certs: Option<String>,
    ) -> Arc<AppState> {
        let engine_config = EngineCoreConfig {
            hybrid_kv: None,
            scheduler_config: SchedulerConfig {
                max_num_batched_tokens: 8192,
                max_num_seqs: 256,
                max_num_scheduled_tokens: None,
                policy: SchedulerPolicy::Fcfs,
                enable_chunked_prefill: true,
                long_prefill_token_threshold: 0,
                ..Default::default()
            },
            max_model_len: 4096,
            num_gpu_blocks: 1024,
            block_size: 16,
            engine_index: 0,
            async_scheduling: false,
            use_spec_decode: false,
            proposer_config: None,
            eos_token_ids: vec![],
            is_pooling: false,
            enable_prefix_caching: false,
        };
        let executor = Box::new(NoopExecutor::new(1024));
        let client = Box::new(InprocClient::new(engine_config, executor));
        let engine = Arc::new(AsyncEngine::new(client, "test-model".to_string(), 4096));

        // Use port 0 to get a random available port.
        Arc::new(AppState {
            engine,
            config: ServerConfig {
                bind_address: "127.0.0.1:0".to_string(),
                ssl_certfile: Some(certfile),
                ssl_keyfile: Some(keyfile),
                ssl_ca_certs: ca_certs,
                ..ServerConfig::default()
            },
            is_pooling: false,
            scratchy_core_config: None,
        })
    }

    #[cfg(feature = "tls")]
    #[tokio::test]
    async fn test_tls_health_endpoint() {
        let (ca_pem, cert_pem, key_pem) = generate_test_certs();
        let tmp = tempfile::tempdir().unwrap();
        let certfile = write_pem_file(&tmp, "server.crt", &cert_pem);
        let keyfile = write_pem_file(&tmp, "server.key", &key_pem);

        // Build the TLS config directly and start axum-server.
        let tls_config = build_tls_config(&certfile, &keyfile, None).unwrap();
        let state = make_tls_test_state(certfile, keyfile, None);
        let router = build_router(state);

        let addr: std::net::SocketAddr = "127.0.0.1:0".parse().unwrap();
        let handle = axum_server::Handle::new();
        let server_handle = handle.clone();

        let server_task = tokio::spawn(async move {
            axum_server::bind_rustls(addr, tls_config)
                .handle(server_handle)
                .serve(router.into_make_service())
                .await
                .unwrap();
        });

        // Wait for the server to start and get the actual port.
        let listening_addr = tokio::time::timeout(std::time::Duration::from_secs(5), async {
            loop {
                if let Some(addr) = handle.listening().await {
                    return addr;
                }
            }
        })
        .await
        .expect("server did not start within 5s");

        // Connect with reqwest trusting the test CA.
        let _ = rustls::crypto::ring::default_provider().install_default();
        let ca_cert = reqwest::Certificate::from_pem(ca_pem.as_bytes()).unwrap();
        let client = reqwest::Client::builder()
            .add_root_certificate(ca_cert)
            .build()
            .unwrap();

        let resp = client
            .get(format!(
                "https://127.0.0.1:{}/health",
                listening_addr.port()
            ))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);

        let body: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(body["status"], "ok");

        // Shutdown.
        handle.shutdown();
        let _ = server_task.await;
    }

    #[cfg(feature = "tls")]
    #[tokio::test]
    async fn test_tls_rejects_plain_http() {
        let (_ca_pem, cert_pem, key_pem) = generate_test_certs();
        let tmp = tempfile::tempdir().unwrap();
        let certfile = write_pem_file(&tmp, "server.crt", &cert_pem);
        let keyfile = write_pem_file(&tmp, "server.key", &key_pem);

        let tls_config = build_tls_config(&certfile, &keyfile, None).unwrap();
        let state = make_tls_test_state(certfile, keyfile, None);
        let router = build_router(state);

        let addr: std::net::SocketAddr = "127.0.0.1:0".parse().unwrap();
        let handle = axum_server::Handle::new();
        let server_handle = handle.clone();

        let server_task = tokio::spawn(async move {
            axum_server::bind_rustls(addr, tls_config)
                .handle(server_handle)
                .serve(router.into_make_service())
                .await
                .unwrap();
        });

        let listening_addr = tokio::time::timeout(std::time::Duration::from_secs(5), async {
            loop {
                if let Some(addr) = handle.listening().await {
                    return addr;
                }
            }
        })
        .await
        .expect("server did not start within 5s");

        // Plain HTTP to an HTTPS port should fail.
        let _ = rustls::crypto::ring::default_provider().install_default();
        let client = reqwest::Client::new();
        let result = client
            .get(format!("http://127.0.0.1:{}/health", listening_addr.port()))
            .send()
            .await;
        assert!(result.is_err(), "plain HTTP to TLS port should fail");

        handle.shutdown();
        let _ = server_task.await;
    }

    #[cfg(feature = "tls")]
    #[test]
    fn test_build_tls_config_missing_certfile() {
        let result = build_tls_config("/nonexistent/cert.pem", "/nonexistent/key.pem", None);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("ssl_certfile"), "error was: {err}");
    }

    #[cfg(feature = "tls")]
    #[test]
    fn test_build_tls_config_missing_keyfile() {
        let tmp = tempfile::tempdir().unwrap();
        let certfile = write_pem_file(&tmp, "server.crt", "not real but file exists");
        let result = build_tls_config(&certfile, "/nonexistent/key.pem", None);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("ssl_keyfile"), "error was: {err}");
    }

    #[test]
    fn test_server_config_default_no_ssl() {
        let config = ServerConfig::default();
        assert!(config.ssl_keyfile.is_none());
        assert!(config.ssl_certfile.is_none());
        assert!(config.ssl_ca_certs.is_none());
    }

    // -------------------------------------------------------------------
    // Pooling mode tests
    // -------------------------------------------------------------------

    fn make_pooling_test_state() -> Arc<AppState> {
        let engine_config = EngineCoreConfig {
            hybrid_kv: None,
            scheduler_config: SchedulerConfig {
                max_num_batched_tokens: 8192,
                max_num_seqs: 256,
                max_num_scheduled_tokens: None,
                policy: SchedulerPolicy::Fcfs,
                enable_chunked_prefill: true,
                long_prefill_token_threshold: 0,
                ..Default::default()
            },
            max_model_len: 4096,
            num_gpu_blocks: 1024,
            block_size: 16,
            engine_index: 0,
            async_scheduling: false,
            use_spec_decode: false,
            proposer_config: None,
            eos_token_ids: vec![],
            is_pooling: true,
            enable_prefix_caching: false,
        };
        let executor = Box::new(NoopExecutor::new(1024));
        let client = Box::new(InprocClient::new(engine_config, executor));
        let mut engine = AsyncEngine::new(client, "test-model".to_string(), 4096);
        engine.set_is_pooling(true);
        let engine = Arc::new(engine);
        engine.spawn_step_loop();

        Arc::new(AppState {
            engine,
            config: ServerConfig::default(),
            is_pooling: true,
            scratchy_core_config: None,
        })
    }

    #[tokio::test]
    async fn test_pooling_mode_rejects_chat_completions() {
        let state = make_pooling_test_state();
        let app = build_router(state);

        let body = serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}]
        });

        let request = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_string(&body).unwrap()))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(
            response.status(),
            StatusCode::BAD_REQUEST,
            "Chat completions should return 400 in pooling mode, got {}",
            response.status()
        );

        let body = response.into_body().collect().await.unwrap().to_bytes();
        let parsed: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let message = parsed["error"]["message"].as_str().unwrap_or("");
        assert!(
            message.contains("pooling mode"),
            "Error should mention pooling mode, got: {message}"
        );
    }

    #[tokio::test]
    async fn test_pooling_mode_rejects_completions() {
        let state = make_pooling_test_state();
        let app = build_router(state);

        let body = serde_json::json!({
            "model": "test-model",
            "prompt": "Hello"
        });

        let request = Request::builder()
            .method("POST")
            .uri("/v1/completions")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_string(&body).unwrap()))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(
            response.status(),
            StatusCode::BAD_REQUEST,
            "Completions should return 400 in pooling mode, got {}",
            response.status()
        );

        let body = response.into_body().collect().await.unwrap().to_bytes();
        let parsed: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let message = parsed["error"]["message"].as_str().unwrap_or("");
        assert!(
            message.contains("pooling mode"),
            "Error should mention pooling mode, got: {message}"
        );
    }

    #[tokio::test]
    async fn test_pooling_mode_allows_health_and_models() {
        let state = make_pooling_test_state();
        let app = build_router(Arc::clone(&state));

        // Health should still work.
        let request = Request::builder()
            .uri("/health")
            .body(Body::empty())
            .unwrap();
        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        // Models should still work.
        let app2 = build_router(state);
        let request = Request::builder()
            .uri("/v1/models")
            .body(Body::empty())
            .unwrap();
        let response = app2.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_render_chat_completion_bad_json() {
        let state = make_test_state();
        let app = build_router(state);

        let request = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions/render")
            .header("content-type", "application/json")
            .body(Body::from("not json"))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_render_chat_completion_missing_messages() {
        let state = make_test_state();
        let app = build_router(state);

        let request = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions/render")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"model": "test"}"#))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(
            response.status(),
            StatusCode::UNPROCESSABLE_ENTITY,
            "actual status: {}",
            response.status()
        );
    }

    /// Create a test state with the step loop spawned (needed for control messages).
    fn make_test_state_with_step_loop() -> Arc<AppState> {
        let state = make_test_state();
        state.engine.spawn_step_loop();
        state
    }

    #[tokio::test]
    async fn test_reset_prefix_cache_endpoint() {
        let state = make_test_state_with_step_loop();
        let app = build_router(state);

        let request = Request::builder()
            .method("POST")
            .uri("/reset_prefix_cache")
            .body(Body::empty())
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_reset_prefix_cache_get_returns_405() {
        let state = make_test_state_with_step_loop();
        let app = build_router(state);

        let request = Request::builder()
            .method("GET")
            .uri("/reset_prefix_cache")
            .body(Body::empty())
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::METHOD_NOT_ALLOWED);
    }
}
