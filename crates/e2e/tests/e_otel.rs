// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! OpenTelemetry E2E tests.
//!
//! These tests verify that the server starts and serves requests correctly
//! when OpenTelemetry tracing is enabled. We use a fake OTLP endpoint (the
//! exporter is lazy and won't fail on connection errors during request handling).
//!
//! Run with:
//!   cargo test -p vllm-e2e --features e2e,otel --test e_otel -- --ignored

#![cfg(all(feature = "e2e", feature = "otel"))]

use std::sync::Arc;
use std::time::Duration;

use scratchy_e2e::Client;
use scratchy_e2e::assertions::{assert_coherent_text, assert_valid_completion_response};
use scratchy_serving_api::protocol::{CompletionPrompt, CompletionRequest};

/// Start a server in-process with OTel tracing enabled (pointing at a
/// non-existent collector — the exporter is non-blocking so this is fine).
async fn start_otel_server() -> (u16, String, OtelTestServer) {
    let port = {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        listener.local_addr().unwrap().port()
    };
    let base_url = format!("http://127.0.0.1:{port}");
    let bind_address = format!("127.0.0.1:{port}");

    // Init tracing with OTel pointing to a fake endpoint.
    let otel_config = scratchy_core_common::telemetry::OtelConfig {
        endpoint: "http://127.0.0.1:4317".to_string(), // nothing listening — that's OK
    };
    let _otel_guard = scratchy_core_common::telemetry::init_tracing_with_otel("info", &otel_config);

    let model = scratchy_e2e::TestModels::SMOLLM;
    let config = scratchy_serving_api::init::VllmConfig {
        model: model.to_string(),
        device: "auto".to_string(),
        dtype: "auto".to_string(),
        max_num_seqs: 16,
        block_size: 16,
        gpu_memory_utilization: 0.9,
        ..Default::default()
    };

    let stack =
        tokio::task::spawn_blocking(move || scratchy_serving_api::init::initialize_stack(&config))
            .await
            .expect("panicked")
            .expect("failed to init stack");

    let step_handle = stack.engine.spawn_step_loop();

    let server_config = scratchy_serving_api::server::ServerConfig {
        bind_address,
        version: "test-otel".to_string(),
        cors_enabled: true,
        metrics_enabled: false,
        ssl_keyfile: None,
        ssl_certfile: None,
        ssl_ca_certs: None,
        startup_instant: None,
    };

    let app_state = Arc::new(scratchy_serving_api::server::AppState {
        engine: stack.engine,
        config: server_config,
        is_pooling: false,
        scratchy_core_config: None,
    });

    let server_handle = tokio::spawn(async move {
        if let Err(e) = scratchy_serving_api::server::serve(app_state).await {
            tracing::error!("otel test server error: {e}");
        }
    });

    // Wait for health.
    let _ = rustls::crypto::ring::default_provider().install_default();
    let client = reqwest::Client::new();
    let health_url = format!("{base_url}/health");
    let deadline = tokio::time::Instant::now() + Duration::from_secs(60);
    loop {
        if tokio::time::Instant::now() > deadline {
            panic!("otel test server did not become healthy");
        }
        if let Ok(resp) = client.get(&health_url).send().await {
            if resp.status().is_success() {
                break;
            }
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    (
        port,
        base_url,
        OtelTestServer {
            _step_handle: step_handle,
            _server_handle: server_handle,
        },
    )
}

struct OtelTestServer {
    _step_handle: tokio::task::JoinHandle<()>,
    _server_handle: tokio::task::JoinHandle<()>,
}

impl Drop for OtelTestServer {
    fn drop(&mut self) {
        self._step_handle.abort();
        self._server_handle.abort();
    }
}

fn default_completion_request() -> CompletionRequest {
    serde_json::from_str(r#"{}"#).unwrap()
}

/// Verify the server starts and serves a completion with OTel tracing enabled.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_otel_server_serves_completions() {
    let (_port, base_url, _server) = start_otel_server().await;
    let client = Client::new(&base_url);

    let req = CompletionRequest {
        prompt: Some(CompletionPrompt::Single(
            "The capital of France is".to_string(),
        )),
        max_tokens: Some(16),
        temperature: Some(0.0),
        ..default_completion_request()
    };

    let resp = client.completion(&req).await.expect("request failed");
    assert_valid_completion_response(&resp);
    let text = &resp.choices[0].text;
    assert_coherent_text(text, 2);
    eprintln!("[otel E2E] generated: {text}");
}

/// Verify /health works with OTel enabled.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_otel_health_endpoint() {
    let (_port, base_url, _server) = start_otel_server().await;

    let resp = reqwest::get(format!("{base_url}/health"))
        .await
        .expect("health request failed");
    assert_eq!(resp.status(), 200);
}
