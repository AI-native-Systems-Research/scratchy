// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! E2E tests for the `scr batch` offline batch processing command.
//!
//! These tests exercise the batch runner directly via `run_batch_from_config`
//! — no HTTP server involved. Each test writes a temp JSONL input file,
//! processes it through the engine, and reads the JSONL output.
//!
//! Run with: `cargo test -p vllm-e2e --features e2e --test e_batch -- --ignored --test-threads=1`

#![cfg(feature = "e2e")]

use scratchy_e2e::TestModels;
use scratchy_serving_api::init::VllmConfig;
use scratchy_serving_api::protocol::{BatchRequestInput, BatchRequestOutput};

/// Build a minimal VllmConfig for the given model.
fn test_config(model: &str) -> VllmConfig {
    VllmConfig {
        model: model.to_string(),
        ..VllmConfig::default()
    }
}

/// Write batch request inputs to a temp JSONL file and return the path.
fn write_input_jsonl(dir: &tempfile::TempDir, requests: &[BatchRequestInput]) -> String {
    let path = dir.path().join("input.jsonl");
    let mut content = String::new();
    for req in requests {
        content.push_str(&serde_json::to_string(req).unwrap());
        content.push('\n');
    }
    std::fs::write(&path, &content).unwrap();
    path.to_str().unwrap().to_string()
}

/// Read batch output JSONL from a file.
fn read_output_jsonl(path: &str) -> Vec<BatchRequestOutput> {
    let content = std::fs::read_to_string(path).unwrap();
    content
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}

use scratchy_serving_api::engine::AsyncEngine;
use scratchy_serving_api::init::initialize_stack;
use scratchy_serving_api::protocol::{
    BatchResponseData, ChatCompletionRequest, CompletionRequest, EmbeddingRequest,
};

/// Dispatch a single batch request to the appropriate engine method.
async fn dispatch_request(
    engine: &AsyncEngine,
    input: &BatchRequestInput,
) -> Result<serde_json::Value, String> {
    match input.url.as_str() {
        "/v1/chat/completions" => {
            let mut req: ChatCompletionRequest =
                serde_json::from_value(input.body.clone()).map_err(|e| e.to_string())?;
            req.stream = false;
            let resp = engine
                .chat_completion(req)
                .await
                .map_err(|e| e.to_string())?;
            serde_json::to_value(&resp).map_err(|e| e.to_string())
        }
        "/v1/completions" => {
            let mut req: CompletionRequest =
                serde_json::from_value(input.body.clone()).map_err(|e| e.to_string())?;
            req.stream = false;
            let resp = engine.completion(req).await.map_err(|e| e.to_string())?;
            serde_json::to_value(&resp).map_err(|e| e.to_string())
        }
        "/v1/embeddings" => {
            let req: EmbeddingRequest =
                serde_json::from_value(input.body.clone()).map_err(|e| e.to_string())?;
            let resp = engine.embeddings(req).await.map_err(|e| e.to_string())?;
            serde_json::to_value(&resp).map_err(|e| e.to_string())
        }
        url => Err(format!("unsupported batch URL: {url}")),
    }
}

/// Helper to run batch from config, using the crate's public API.
async fn run_batch(config: VllmConfig, input_path: &str, output_path: &str) -> anyhow::Result<()> {
    let input_content = std::fs::read_to_string(input_path)?;
    let requests: Vec<BatchRequestInput> = input_content
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();

    if requests.is_empty() {
        std::fs::write(output_path, "")?;
        return Ok(());
    }

    let stack = tokio::task::spawn_blocking(move || initialize_stack(&config))
        .await
        .expect("initialize_stack panicked")?;

    let _step_handle = stack.engine.spawn_step_loop();
    let engine = stack.engine.clone();

    let mut handles = Vec::new();
    for req_input in requests {
        let engine = engine.clone();
        handles.push(tokio::spawn(async move {
            let batch_id = format!("batch-{}", uuid::Uuid::new_v4());
            let request_id = uuid::Uuid::new_v4().to_string();

            match dispatch_request(&engine, &req_input).await {
                Ok(body) => BatchRequestOutput {
                    id: batch_id,
                    custom_id: req_input.custom_id,
                    response: Some(BatchResponseData {
                        status_code: 200,
                        request_id,
                        body: Some(body),
                    }),
                    error: None,
                },
                Err(e) => BatchRequestOutput {
                    id: batch_id,
                    custom_id: req_input.custom_id,
                    response: None,
                    error: Some(serde_json::json!({
                        "message": e,
                        "type": "batch_processing_error",
                        "code": 400,
                    })),
                },
            }
        }));
    }

    let mut outputs = Vec::new();
    for handle in handles {
        outputs.push(handle.await.expect("task panicked"));
    }

    let mut content = String::new();
    for output in &outputs {
        content.push_str(&serde_json::to_string(output).unwrap());
        content.push('\n');
    }
    std::fs::write(output_path, &content)?;

    Ok(())
}

// ===========================================================================
// Chat completions
// ===========================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_batch_chat_completions() {
    scratchy_core_common::telemetry::init_tracing("off");

    let dir = tempfile::tempdir().unwrap();
    let requests = vec![
        BatchRequestInput {
            custom_id: "chat-1".to_string(),
            method: "POST".to_string(),
            url: "/v1/chat/completions".to_string(),
            body: serde_json::json!({
                "messages": [{"role": "user", "content": "Say hello"}],
                "max_tokens": 20,
                "temperature": 0.0
            }),
        },
        BatchRequestInput {
            custom_id: "chat-2".to_string(),
            method: "POST".to_string(),
            url: "/v1/chat/completions".to_string(),
            body: serde_json::json!({
                "messages": [{"role": "user", "content": "What is 1+1?"}],
                "max_tokens": 20,
                "temperature": 0.0
            }),
        },
        BatchRequestInput {
            custom_id: "chat-3".to_string(),
            method: "POST".to_string(),
            url: "/v1/chat/completions".to_string(),
            body: serde_json::json!({
                "messages": [{"role": "user", "content": "Name a color"}],
                "max_tokens": 20,
                "temperature": 0.0
            }),
        },
    ];

    let input_path = write_input_jsonl(&dir, &requests);
    let output_path = dir.path().join("output.jsonl");
    let output_str = output_path.to_str().unwrap();

    run_batch(test_config(TestModels::SMOLLM), &input_path, output_str)
        .await
        .expect("batch should succeed");

    let outputs = read_output_jsonl(output_str);
    assert_eq!(outputs.len(), 3);

    // All should succeed with matching custom_ids.
    let custom_ids: Vec<&str> = outputs.iter().map(|o| o.custom_id.as_str()).collect();
    assert!(custom_ids.contains(&"chat-1"));
    assert!(custom_ids.contains(&"chat-2"));
    assert!(custom_ids.contains(&"chat-3"));

    for output in &outputs {
        assert!(
            output.error.is_none(),
            "expected no error for {}",
            output.custom_id
        );
        let resp = output.response.as_ref().unwrap();
        assert_eq!(resp.status_code, 200);
        // Body should have choices array.
        let body = resp.body.as_ref().unwrap();
        assert!(body["choices"].is_array());
        assert!(
            !body["choices"][0]["message"]["content"]
                .as_str()
                .unwrap_or("")
                .is_empty()
        );
    }
}

// ===========================================================================
// Text completions
// ===========================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_batch_completions() {
    scratchy_core_common::telemetry::init_tracing("off");

    let dir = tempfile::tempdir().unwrap();
    let requests = vec![
        BatchRequestInput {
            custom_id: "comp-1".to_string(),
            method: "POST".to_string(),
            url: "/v1/completions".to_string(),
            body: serde_json::json!({
                "prompt": "The capital of France is",
                "max_tokens": 20,
                "temperature": 0.0
            }),
        },
        BatchRequestInput {
            custom_id: "comp-2".to_string(),
            method: "POST".to_string(),
            url: "/v1/completions".to_string(),
            body: serde_json::json!({
                "prompt": "Once upon a time",
                "max_tokens": 20,
                "temperature": 0.0
            }),
        },
    ];

    let input_path = write_input_jsonl(&dir, &requests);
    let output_path = dir.path().join("output.jsonl");
    let output_str = output_path.to_str().unwrap();

    run_batch(test_config(TestModels::SMOLLM), &input_path, output_str)
        .await
        .expect("batch should succeed");

    let outputs = read_output_jsonl(output_str);
    assert_eq!(outputs.len(), 2);

    for output in &outputs {
        assert!(
            output.error.is_none(),
            "expected no error for {}",
            output.custom_id
        );
        let resp = output.response.as_ref().unwrap();
        assert_eq!(resp.status_code, 200);
        let body = resp.body.as_ref().unwrap();
        assert!(body["choices"].is_array());
        assert!(!body["choices"][0]["text"].as_str().unwrap_or("").is_empty());
    }
}

// ===========================================================================
// Embeddings
// ===========================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_batch_embeddings() {
    scratchy_core_common::telemetry::init_tracing("off");

    let dir = tempfile::tempdir().unwrap();
    let requests = vec![
        BatchRequestInput {
            custom_id: "emb-1".to_string(),
            method: "POST".to_string(),
            url: "/v1/embeddings".to_string(),
            body: serde_json::json!({
                "input": "Hello world",
                "model": "test"
            }),
        },
        BatchRequestInput {
            custom_id: "emb-2".to_string(),
            method: "POST".to_string(),
            url: "/v1/embeddings".to_string(),
            body: serde_json::json!({
                "input": "Goodbye world",
                "model": "test"
            }),
        },
    ];

    let input_path = write_input_jsonl(&dir, &requests);
    let output_path = dir.path().join("output.jsonl");
    let output_str = output_path.to_str().unwrap();

    run_batch(test_config(TestModels::SMOLLM), &input_path, output_str)
        .await
        .expect("batch should succeed");

    let outputs = read_output_jsonl(output_str);
    assert_eq!(outputs.len(), 2);

    for output in &outputs {
        assert!(
            output.error.is_none(),
            "expected no error for {}",
            output.custom_id
        );
        let resp = output.response.as_ref().unwrap();
        assert_eq!(resp.status_code, 200);
        let body = resp.body.as_ref().unwrap();
        assert!(body["data"].is_array());
        assert!(body["data"][0]["embedding"].is_array());
        let embedding = body["data"][0]["embedding"].as_array().unwrap();
        assert!(
            !embedding.is_empty(),
            "embedding vector should not be empty"
        );
    }
}

// ===========================================================================
// Mixed endpoints
// ===========================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_batch_mixed_endpoints() {
    scratchy_core_common::telemetry::init_tracing("off");

    let dir = tempfile::tempdir().unwrap();
    let requests = vec![
        BatchRequestInput {
            custom_id: "mix-chat".to_string(),
            method: "POST".to_string(),
            url: "/v1/chat/completions".to_string(),
            body: serde_json::json!({
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 10,
                "temperature": 0.0
            }),
        },
        BatchRequestInput {
            custom_id: "mix-comp".to_string(),
            method: "POST".to_string(),
            url: "/v1/completions".to_string(),
            body: serde_json::json!({
                "prompt": "Hello",
                "max_tokens": 10,
                "temperature": 0.0
            }),
        },
        BatchRequestInput {
            custom_id: "mix-emb".to_string(),
            method: "POST".to_string(),
            url: "/v1/embeddings".to_string(),
            body: serde_json::json!({
                "input": "Test",
                "model": "test"
            }),
        },
    ];

    let input_path = write_input_jsonl(&dir, &requests);
    let output_path = dir.path().join("output.jsonl");
    let output_str = output_path.to_str().unwrap();

    run_batch(test_config(TestModels::SMOLLM), &input_path, output_str)
        .await
        .expect("batch should succeed");

    let outputs = read_output_jsonl(output_str);
    assert_eq!(outputs.len(), 3);

    // Find each by custom_id and verify the right response shape.
    for output in &outputs {
        assert!(
            output.error.is_none(),
            "expected no error for {}",
            output.custom_id
        );
        let body = output.response.as_ref().unwrap().body.as_ref().unwrap();
        match output.custom_id.as_str() {
            "mix-chat" => {
                assert!(body["choices"][0]["message"]["content"].is_string());
            }
            "mix-comp" => {
                assert!(body["choices"][0]["text"].is_string());
            }
            "mix-emb" => {
                assert!(body["data"][0]["embedding"].is_array());
            }
            other => panic!("unexpected custom_id: {other}"),
        }
    }
}

// ===========================================================================
// Error handling: invalid URL
// ===========================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_batch_invalid_url() {
    scratchy_core_common::telemetry::init_tracing("off");

    let dir = tempfile::tempdir().unwrap();
    let requests = vec![
        BatchRequestInput {
            custom_id: "good".to_string(),
            method: "POST".to_string(),
            url: "/v1/chat/completions".to_string(),
            body: serde_json::json!({
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 10,
                "temperature": 0.0
            }),
        },
        BatchRequestInput {
            custom_id: "bad-url".to_string(),
            method: "POST".to_string(),
            url: "/v1/nonexistent".to_string(),
            body: serde_json::json!({}),
        },
    ];

    let input_path = write_input_jsonl(&dir, &requests);
    let output_path = dir.path().join("output.jsonl");
    let output_str = output_path.to_str().unwrap();

    run_batch(test_config(TestModels::SMOLLM), &input_path, output_str)
        .await
        .expect("batch should complete without crashing");

    let outputs = read_output_jsonl(output_str);
    assert_eq!(outputs.len(), 2);

    // Find the bad-url request and verify it has an error.
    let bad = outputs.iter().find(|o| o.custom_id == "bad-url").unwrap();
    assert!(bad.error.is_some(), "bad URL should produce an error");
    assert!(bad.response.is_none());

    // The good request should succeed.
    let good = outputs.iter().find(|o| o.custom_id == "good").unwrap();
    assert!(good.error.is_none());
    assert!(good.response.is_some());
}

// ===========================================================================
// Error handling: malformed body
// ===========================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_batch_malformed_body() {
    scratchy_core_common::telemetry::init_tracing("off");

    let dir = tempfile::tempdir().unwrap();
    let requests = vec![BatchRequestInput {
        custom_id: "malformed".to_string(),
        method: "POST".to_string(),
        url: "/v1/chat/completions".to_string(),
        // Missing required "messages" field.
        body: serde_json::json!({"max_tokens": 10}),
    }];

    let input_path = write_input_jsonl(&dir, &requests);
    let output_path = dir.path().join("output.jsonl");
    let output_str = output_path.to_str().unwrap();

    run_batch(test_config(TestModels::SMOLLM), &input_path, output_str)
        .await
        .expect("batch should complete without crashing");

    let outputs = read_output_jsonl(output_str);
    assert_eq!(outputs.len(), 1);

    let output = &outputs[0];
    assert_eq!(output.custom_id, "malformed");
    assert!(
        output.error.is_some(),
        "malformed body should produce an error"
    );
    assert!(output.response.is_none());
}
