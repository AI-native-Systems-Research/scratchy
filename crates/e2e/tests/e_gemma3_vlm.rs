// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Gemma 3 VLM (Gemma3ForConditionalGeneration) E2E tests via Candle backend.
//!
//! Validates that the multimodal config with nested `text_config` loads and
//! serves correctly through the non-MLX (SafeTensors) path. This catches
//! regressions in config parsing for fields that rely on transformers defaults
//! (vocab_size, head_dim, num_attention_heads, etc.).
//!
//! Model: google/gemma-3-4b-it (~8 GB BF16, Tier 4 — weekly/manual only)
//!
//! Run with:
//! ```bash
//! cargo test -p vllm-e2e --features e2e --test e_gemma3_vlm -- --ignored --test-threads=1
//! ```

#![cfg(feature = "e2e")]

use scratchy_e2e::assertions::{assert_coherent_text, assert_valid_chat_response};
use scratchy_e2e::{Client, TestModels, TestServer};
use scratchy_serving_api::protocol::{ChatCompletionMessageParam, ChatCompletionRequest};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn default_chat_request() -> ChatCompletionRequest {
    serde_json::from_str(r#"{"messages": []}"#).unwrap()
}

fn user_msg(content: &str) -> ChatCompletionMessageParam {
    ChatCompletionMessageParam {
        role: "user".to_string(),
        content: Some(serde_json::Value::String(content.to_string())),
        name: None,
        tool_calls: None,
        tool_call_id: None,
    }
}

async fn start_gemma3_vlm_candle() -> (TestServer, Client) {
    let server = TestServer::builder(TestModels::GEMMA3_4B_IT)
        .start()
        .await
        .expect("Gemma 3 4B VLM server should start");
    let client = Client::new(server.base_url());
    (server, client)
}

// ===========================================================================
// Tests
// ===========================================================================

/// Server starts with Gemma3ForConditionalGeneration (SafeTensors / Candle path).
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_gemma3_vlm_candle_server_starts() {
    let (server, client) = start_gemma3_vlm_candle().await;

    assert!(client.health().await.unwrap(), "server should be healthy");

    let models = client.list_models().await.unwrap();
    assert_eq!(models.data.len(), 1);
    assert!(
        models.data[0].id.contains("gemma-3"),
        "model name should contain 'gemma-3', got: {}",
        models.data[0].id
    );

    drop(server);
}

/// Text-only chat works through the multimodal model.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_gemma3_vlm_candle_text_only_chat() {
    let (_server, client) = start_gemma3_vlm_candle().await;

    let request = ChatCompletionRequest {
        messages: vec![user_msg("What is 2 + 2?")],
        max_tokens: Some(32),
        temperature: Some(0.0),
        ..default_chat_request()
    };

    let resp = client.chat_completion(&request).await.unwrap();
    assert_valid_chat_response(&resp);

    let text = resp.choices[0].message.content.as_deref().unwrap_or("");
    assert_coherent_text(text, 1);
}

/// max_tokens is respected.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_gemma3_vlm_candle_max_tokens() {
    let (_server, client) = start_gemma3_vlm_candle().await;

    let request = ChatCompletionRequest {
        messages: vec![user_msg("Write a long story about a cat.")],
        max_tokens: Some(5),
        temperature: Some(0.0),
        ..default_chat_request()
    };

    let resp = client.chat_completion(&request).await.unwrap();
    assert_valid_chat_response(&resp);
    let completion_tokens = resp.usage.completion_tokens.unwrap_or(0);
    assert!(
        completion_tokens <= 5,
        "completion_tokens ({completion_tokens}) should be <= 5",
    );
}
