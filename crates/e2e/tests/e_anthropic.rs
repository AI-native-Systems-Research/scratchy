// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! E2E tests for the Anthropic `/v1/messages` endpoint.
//!
//! Run with: `cargo test -p vllm-e2e --features e2e --test e_anthropic -- --ignored`

#![cfg(feature = "e2e")]

use scratchy_e2e::{Client, TestModels, TestServer};
use serde_json::json;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

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
async fn test_anthropic_simple_message() {
    let (_server, client) = start_smollm().await;

    let resp = client
        .anthropic_messages_raw(&json!({
            "max_tokens": 20,
            "messages": [{"role": "user", "content": "Say hello"}]
        }))
        .await
        .unwrap();

    assert!(resp.status().is_success(), "status: {}", resp.status());

    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["type"], "message");
    assert_eq!(body["role"], "assistant");
    assert!(body["content"].is_array());
    assert!(!body["content"].as_array().unwrap().is_empty());
    assert_eq!(body["content"][0]["type"], "text");
    assert!(body["content"][0]["text"].as_str().unwrap().len() > 0);
    assert!(body["usage"]["input_tokens"].as_u64().unwrap() > 0);
    assert!(body["usage"]["output_tokens"].as_u64().unwrap() > 0);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_anthropic_with_system() {
    let (_server, client) = start_smollm().await;

    let resp = client
        .anthropic_messages_raw(&json!({
            "max_tokens": 20,
            "system": "You are a helpful assistant.",
            "messages": [{"role": "user", "content": "Hi"}]
        }))
        .await
        .unwrap();

    assert!(resp.status().is_success());
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["type"], "message");
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_anthropic_system_blocks() {
    let (_server, client) = start_smollm().await;

    let resp = client
        .anthropic_messages_raw(&json!({
            "max_tokens": 20,
            "system": [{"text": "Be concise."}, {"text": "Be accurate."}],
            "messages": [{"role": "user", "content": "Hi"}]
        }))
        .await
        .unwrap();

    assert!(resp.status().is_success());
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_anthropic_multi_turn() {
    let (_server, client) = start_smollm().await;

    let resp = client
        .anthropic_messages_raw(&json!({
            "max_tokens": 20,
            "messages": [
                {"role": "user", "content": "My name is Alice."},
                {"role": "assistant", "content": "Hello Alice!"},
                {"role": "user", "content": "What is my name?"}
            ]
        }))
        .await
        .unwrap();

    assert!(resp.status().is_success());
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["type"], "message");
    assert!(!body["content"].as_array().unwrap().is_empty());
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_anthropic_content_blocks() {
    let (_server, client) = start_smollm().await;

    let resp = client
        .anthropic_messages_raw(&json!({
            "max_tokens": 20,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is 2+2?"},
                    {"type": "text", "text": "Tell me quickly."}
                ]
            }]
        }))
        .await
        .unwrap();

    assert!(resp.status().is_success());
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_anthropic_temperature() {
    let (_server, client) = start_smollm().await;

    let resp = client
        .anthropic_messages_raw(&json!({
            "max_tokens": 20,
            "temperature": 0.0,
            "messages": [{"role": "user", "content": "Count to 5"}]
        }))
        .await
        .unwrap();

    assert!(resp.status().is_success());
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_anthropic_stop_sequences() {
    let (_server, client) = start_smollm().await;

    let resp = client
        .anthropic_messages_raw(&json!({
            "max_tokens": 100,
            "stop_sequences": ["."],
            "messages": [{"role": "user", "content": "Tell me a story"}]
        }))
        .await
        .unwrap();

    assert!(resp.status().is_success());
    let body: serde_json::Value = resp.json().await.unwrap();
    // Should stop at first period or before max_tokens
    let text = body["content"][0]["text"].as_str().unwrap_or("");
    // The text should not contain a period (it stops before outputting it by default)
    assert!(text.len() < 500, "should have stopped early");
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_anthropic_max_tokens_respected() {
    let (_server, client) = start_smollm().await;

    let resp = client
        .anthropic_messages_raw(&json!({
            "max_tokens": 5,
            "messages": [{"role": "user", "content": "Write a long essay about everything"}]
        }))
        .await
        .unwrap();

    assert!(resp.status().is_success());
    let body: serde_json::Value = resp.json().await.unwrap();
    // With max_tokens=5, output should be short
    assert!(body["usage"]["output_tokens"].as_u64().unwrap() <= 6);
}

// ===========================================================================
// Streaming
// ===========================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_anthropic_streaming() {
    let (_server, client) = start_smollm().await;

    let events = client
        .anthropic_messages_stream(&json!({
            "max_tokens": 20,
            "stream": true,
            "messages": [{"role": "user", "content": "Say hello"}]
        }))
        .await
        .unwrap();

    assert!(!events.is_empty(), "should have received SSE events");

    // Check event sequence
    let types: Vec<&str> = events.iter().filter_map(|e| e["type"].as_str()).collect();

    assert_eq!(
        types[0], "message_start",
        "first event should be message_start"
    );
    assert!(types.contains(&"ping"), "should contain ping");
    assert!(
        types.contains(&"content_block_start"),
        "should contain content_block_start"
    );
    assert!(
        types.contains(&"content_block_delta"),
        "should contain content_block_delta"
    );
    assert!(
        types.contains(&"content_block_stop"),
        "should contain content_block_stop"
    );
    assert!(
        types.contains(&"message_delta"),
        "should contain message_delta"
    );
    assert!(
        types.contains(&"message_stop"),
        "should contain message_stop"
    );

    // Check message_start has expected structure
    let msg_start = &events[0];
    assert_eq!(msg_start["message"]["type"], "message");
    assert_eq!(msg_start["message"]["role"], "assistant");

    // Check we got text deltas
    let text_deltas: Vec<&serde_json::Value> = events
        .iter()
        .filter(|e| e["type"] == "content_block_delta")
        .collect();
    assert!(!text_deltas.is_empty());
    for td in &text_deltas {
        assert_eq!(td["delta"]["type"], "text_delta");
        assert!(td["delta"]["text"].is_string());
    }

    // Check message_delta has stop_reason
    let msg_delta = events
        .iter()
        .find(|e| e["type"] == "message_delta")
        .unwrap();
    assert!(msg_delta["delta"]["stop_reason"].is_string());
    assert!(msg_delta["usage"]["output_tokens"].as_u64().unwrap() > 0);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_anthropic_streaming_with_system() {
    let (_server, client) = start_smollm().await;

    let events = client
        .anthropic_messages_stream(&json!({
            "max_tokens": 10,
            "stream": true,
            "system": "Reply in one word.",
            "messages": [{"role": "user", "content": "Hi"}]
        }))
        .await
        .unwrap();

    let types: Vec<&str> = events.iter().filter_map(|e| e["type"].as_str()).collect();
    assert!(types.contains(&"message_start"));
    assert!(types.contains(&"message_stop"));
}

// ===========================================================================
// Error cases
// ===========================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_anthropic_missing_max_tokens() {
    let (_server, client) = start_smollm().await;

    // max_tokens is required in Anthropic API
    let resp = client
        .anthropic_messages_raw(&json!({
            "messages": [{"role": "user", "content": "Hi"}]
        }))
        .await
        .unwrap();

    // Should fail with 422 (Unprocessable Entity) since max_tokens is required
    assert!(
        !resp.status().is_success(),
        "should fail without max_tokens"
    );
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_anthropic_empty_messages() {
    let (_server, client) = start_smollm().await;

    let resp = client
        .anthropic_messages_raw(&json!({
            "max_tokens": 10,
            "messages": []
        }))
        .await
        .unwrap();

    // Empty messages should fail at the engine level
    assert!(!resp.status().is_success());
}
