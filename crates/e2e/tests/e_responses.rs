// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! E2E tests for the OpenAI `/v1/responses` endpoint.
//!
//! Run with: `cargo test -p vllm-e2e --features e2e --test e_responses -- --ignored`

#![cfg(feature = "e2e")]

use scratchy_e2e::{Client, TestModels, TestServer};
use serde_json::json;

async fn start_smollm() -> (TestServer, Client) {
    let server = TestServer::builder(TestModels::SMOLLM)
        .start()
        .await
        .expect("SmolLM server should start");
    let client = Client::new(server.base_url());
    (server, client)
}

// ===========================================================================
// Non-streaming
// ===========================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_responses_simple_text() {
    let (_server, client) = start_smollm().await;

    let resp = client
        .responses_raw(&json!({
            "input": "Say hello",
            "max_output_tokens": 20
        }))
        .await
        .unwrap();

    assert!(resp.status().is_success(), "status: {}", resp.status());

    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["object"], "response");
    assert_eq!(body["status"], "completed");
    assert!(body["output"].is_array());
    assert!(!body["output"].as_array().unwrap().is_empty());
    assert_eq!(body["output"][0]["type"], "message");
    assert_eq!(body["output"][0]["role"], "assistant");
    assert_eq!(body["output"][0]["content"][0]["type"], "output_text");
    assert!(
        body["output"][0]["content"][0]["text"]
            .as_str()
            .unwrap()
            .len()
            > 0
    );
    assert!(body["usage"]["input_tokens"].as_u64().unwrap() > 0);
    assert!(body["usage"]["output_tokens"].as_u64().unwrap() > 0);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_responses_with_instructions() {
    let (_server, client) = start_smollm().await;

    let resp = client
        .responses_raw(&json!({
            "input": "Hi",
            "instructions": "You are a helpful assistant.",
            "max_output_tokens": 20
        }))
        .await
        .unwrap();

    assert!(resp.status().is_success());
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["object"], "response");
    assert_eq!(body["status"], "completed");
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_responses_multi_turn() {
    let (_server, client) = start_smollm().await;

    let resp = client
        .responses_raw(&json!({
            "input": [
                {"type": "message", "role": "user", "content": "My name is Alice."},
                {"type": "message", "role": "assistant", "content": "Hello Alice!"},
                {"type": "message", "role": "user", "content": "What is my name?"}
            ],
            "max_output_tokens": 20
        }))
        .await
        .unwrap();

    assert!(resp.status().is_success());
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["object"], "response");
    assert!(!body["output"].as_array().unwrap().is_empty());
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_responses_max_tokens_respected() {
    let (_server, client) = start_smollm().await;

    let resp = client
        .responses_raw(&json!({
            "input": "Write a long essay about everything",
            "max_output_tokens": 5
        }))
        .await
        .unwrap();

    assert!(resp.status().is_success());
    let body: serde_json::Value = resp.json().await.unwrap();
    assert!(body["usage"]["output_tokens"].as_u64().unwrap() <= 6);
    // Should be incomplete due to max_output_tokens
    assert_eq!(body["status"], "incomplete");
}

// ===========================================================================
// Streaming
// ===========================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_responses_streaming() {
    let (_server, client) = start_smollm().await;

    let events = client
        .responses_stream(&json!({
            "input": "Say hello",
            "max_output_tokens": 20,
            "stream": true
        }))
        .await
        .unwrap();

    assert!(!events.is_empty(), "should have received SSE events");

    let types: Vec<&str> = events.iter().filter_map(|e| e["type"].as_str()).collect();

    assert_eq!(
        types[0], "response.created",
        "first event should be response.created"
    );
    assert!(types.contains(&"response.in_progress"));
    assert!(types.contains(&"response.output_item.added"));
    assert!(types.contains(&"response.content_part.added"));
    assert!(types.contains(&"response.output_text.delta"));
    assert!(types.contains(&"response.output_text.done"));
    assert!(types.contains(&"response.content_part.done"));
    assert!(types.contains(&"response.output_item.done"));
    assert!(types.contains(&"response.completed"));

    // Check response.created has expected structure
    assert_eq!(events[0]["response"]["object"], "response");
    assert_eq!(events[0]["response"]["status"], "queued");

    // Check text deltas have content
    let text_deltas: Vec<&serde_json::Value> = events
        .iter()
        .filter(|e| e["type"] == "response.output_text.delta")
        .collect();
    assert!(!text_deltas.is_empty());
    for td in &text_deltas {
        assert!(td["delta"].is_string());
    }

    // Check response.completed has usage
    let completed = events
        .iter()
        .find(|e| e["type"] == "response.completed")
        .unwrap();
    assert!(
        completed["response"]["usage"]["output_tokens"]
            .as_u64()
            .unwrap()
            > 0
    );
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_responses_streaming_with_instructions() {
    let (_server, client) = start_smollm().await;

    let events = client
        .responses_stream(&json!({
            "input": "Hi",
            "instructions": "Reply in one word.",
            "max_output_tokens": 10,
            "stream": true
        }))
        .await
        .unwrap();

    let types: Vec<&str> = events.iter().filter_map(|e| e["type"].as_str()).collect();
    assert!(types.contains(&"response.created"));
    assert!(types.contains(&"response.completed"));
}

// ===========================================================================
// CUDA E2E
// ===========================================================================

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_responses_simple() {
    let server = TestServer::builder(TestModels::SMOLLM_135M_CUDA)
        .start()
        .await
        .expect("SmolLM CUDA server should start");
    let client = Client::new(server.base_url());

    let resp = client
        .responses_raw(&json!({
            "input": "Say hello",
            "max_output_tokens": 20
        }))
        .await
        .unwrap();

    assert!(resp.status().is_success(), "status: {}", resp.status());
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["object"], "response");
    assert_eq!(body["status"], "completed");
    assert!(
        body["output"][0]["content"][0]["text"]
            .as_str()
            .unwrap()
            .len()
            > 0
    );
}

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_responses_streaming() {
    let server = TestServer::builder(TestModels::SMOLLM_135M_CUDA)
        .start()
        .await
        .expect("SmolLM CUDA server should start");
    let client = Client::new(server.base_url());

    let events = client
        .responses_stream(&json!({
            "input": "Say hello",
            "max_output_tokens": 20,
            "stream": true
        }))
        .await
        .unwrap();

    let types: Vec<&str> = events.iter().filter_map(|e| e["type"].as_str()).collect();
    assert!(types.contains(&"response.created"));
    assert!(types.contains(&"response.output_text.delta"));
    assert!(types.contains(&"response.completed"));
}
