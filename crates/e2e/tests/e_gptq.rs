// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! E2E tests for GPTQ quantized models.
//!
//! Uses `Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4` (~459 MB).
//! Runs on CPU, NOT MLX.
//!
//! Run with: `cargo test -p vllm-e2e --features e2e --test e_gptq -- --ignored --test-threads=1`

#![cfg(feature = "e2e")]

use scratchy_e2e::assertions::{
    assert_coherent_text, assert_valid_chat_response, assert_valid_completion_response,
};
use scratchy_e2e::{Client, TestModels, TestServer};
use scratchy_serving_api::protocol::{
    ChatCompletionMessageParam, ChatCompletionRequest, CompletionPrompt, CompletionRequest,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn user_msg(content: &str) -> ChatCompletionMessageParam {
    ChatCompletionMessageParam {
        role: "user".to_string(),
        content: Some(serde_json::Value::String(content.to_string())),
        name: None,
        tool_calls: None,
        tool_call_id: None,
    }
}

fn simple_chat_request(content: &str, max_tokens: Option<u32>) -> ChatCompletionRequest {
    ChatCompletionRequest {
        messages: vec![user_msg(content)],
        max_tokens,
        temperature: Some(0.0),
        ..default_chat_request()
    }
}

fn default_chat_request() -> ChatCompletionRequest {
    serde_json::from_str(r#"{"messages": []}"#).unwrap()
}

fn simple_completion_request(prompt: &str, max_tokens: u32) -> CompletionRequest {
    CompletionRequest {
        prompt: Some(CompletionPrompt::Single(prompt.to_string())),
        max_tokens: Some(max_tokens),
        temperature: Some(0.0),
        ..default_completion_request()
    }
}

fn default_completion_request() -> CompletionRequest {
    serde_json::from_str(r#"{}"#).unwrap()
}

// ===========================================================================
// Qwen2.5-0.5B-Instruct-GPTQ-Int4 (Qwen2ForCausalLM, GPTQ INT4)
// ===========================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_gptq_qwen2_server_starts() {
    let server = TestServer::builder(TestModels::QWEN2_0_5B_GPTQ_INT4)
        .start()
        .await
        .expect("GPTQ server should start");

    let client = Client::new(server.base_url());
    assert!(client.health().await.unwrap(), "server should be healthy");

    let models = client.list_models().await.unwrap();
    assert_eq!(models.data.len(), 1);
    assert!(
        models.data[0].id.contains("Qwen"),
        "model name should contain 'Qwen', got: {}",
        models.data[0].id
    );
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_gptq_qwen2_chat_basic() {
    let server = TestServer::builder(TestModels::QWEN2_0_5B_GPTQ_INT4)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = simple_chat_request("Say hello in one sentence.", Some(50));
    let resp = client.chat_completion(&request).await.unwrap();

    assert_valid_chat_response(&resp);
    let text = resp.choices[0].message.content.as_deref().unwrap_or("");
    assert_coherent_text(text, 2);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_gptq_qwen2_completion_basic() {
    let server = TestServer::builder(TestModels::QWEN2_0_5B_GPTQ_INT4)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = simple_completion_request("The capital of France is", 20);
    let resp = client.completion(&request).await.unwrap();

    assert_valid_completion_response(&resp);
    assert!(
        !resp.choices[0].text.is_empty(),
        "completion should not be empty"
    );
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_gptq_qwen2_max_tokens() {
    let server = TestServer::builder(TestModels::QWEN2_0_5B_GPTQ_INT4)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = simple_chat_request("Write a long story about a cat.", Some(5));
    let resp = client.chat_completion(&request).await.unwrap();

    assert_valid_chat_response(&resp);
    assert!(
        resp.usage.completion_tokens.unwrap_or(0) <= 5,
        "completion_tokens should be <= 5, got: {:?}",
        resp.usage.completion_tokens
    );
}
