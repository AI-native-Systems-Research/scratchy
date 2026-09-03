// SPDX-License-Identifier: Apache-2.0

//! E2E tests for FP8-quantized MoE (Mixture of Experts) models.
//!
//! Uses a 2-layer Mixtral 8x7B FP8 checkpoint (~3GB) that fits on a single L40S.
//! The model won't produce coherent text (only 2 layers), but these tests verify
//! the full FP8 MoE pipeline: FP8 weight detection → load_fp8_moe_experts →
//! Fp8FusedMoELayer.forward_owned → fused_moe_fp8_gemm kernel.
//!
//! Requires SM89+ GPU (L40S, H100).
//!
//! Run with:
//!   cargo test -p vllm-e2e --features e2e,cuda --release --test e_fp8_moe -- --ignored --test-threads=1

#![cfg(all(feature = "e2e", feature = "cuda"))]

use scratchy_e2e::assertions::{assert_valid_chat_response, assert_valid_completion_response};
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

fn default_chat_request() -> ChatCompletionRequest {
    serde_json::from_str(r#"{"messages": []}"#).unwrap()
}

fn simple_chat_request(content: &str, max_tokens: Option<u32>) -> ChatCompletionRequest {
    ChatCompletionRequest {
        messages: vec![user_msg(content)],
        max_tokens,
        temperature: Some(0.0),
        ..default_chat_request()
    }
}

fn default_completion_request() -> CompletionRequest {
    serde_json::from_str(r#"{}"#).unwrap()
}

fn simple_completion_request(prompt: &str, max_tokens: u32) -> CompletionRequest {
    CompletionRequest {
        prompt: Some(CompletionPrompt::Single(prompt.to_string())),
        max_tokens: Some(max_tokens),
        temperature: Some(0.0),
        ..default_completion_request()
    }
}

// ===========================================================================
// Mixtral 8x7B FP8 (2-layer, ~3GB — fits on single L40S)
// ===========================================================================

/// FP8 MoE server starts and passes health check.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_fp8_moe_server_starts() {
    let server = TestServer::builder(TestModels::MIXTRAL_8X7B_FP8_2L)
        .with_args(&["--enforce-eager"])
        .start()
        .await
        .expect("FP8 MoE server should start");

    let client = Client::new(server.base_url());
    assert!(client.health().await.unwrap(), "server should be healthy");

    let models = client.list_models().await.unwrap();
    assert_eq!(models.data.len(), 1);
    assert!(
        models.data[0].id.contains("Mixtral"),
        "model name should contain 'Mixtral', got: {}",
        models.data[0].id
    );
}

/// FP8 MoE completion produces non-empty output.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_fp8_moe_completion() {
    let server = TestServer::builder(TestModels::MIXTRAL_8X7B_FP8_2L)
        .with_args(&["--enforce-eager"])
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = simple_completion_request("The capital of France is", 20);
    let resp = client.completion(&request).await.unwrap();

    assert_valid_completion_response(&resp);
    // 2-layer model won't produce coherent text, just verify non-empty
    assert!(
        !resp.choices[0].text.is_empty(),
        "FP8 MoE completion should produce non-empty output"
    );
}

/// FP8 MoE chat completion produces non-empty response.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_fp8_moe_chat() {
    let server = TestServer::builder(TestModels::MIXTRAL_8X7B_FP8_2L)
        .with_args(&["--enforce-eager"])
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = simple_chat_request("Hello", Some(20));
    let resp = client.chat_completion(&request).await.unwrap();

    assert_valid_chat_response(&resp);
    let text = resp.choices[0].message.content.as_deref().unwrap_or("");
    assert!(
        !text.is_empty(),
        "FP8 MoE chat should produce non-empty output"
    );
}

/// FP8 MoE multi-turn — verifies KV cache works with FP8 MoE routing.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_fp8_moe_multi_turn() {
    let server = TestServer::builder(TestModels::MIXTRAL_8X7B_FP8_2L)
        .with_args(&["--enforce-eager"])
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());

    // Turn 1
    let req1 = simple_chat_request("Hello", Some(10));
    let resp1 = client.chat_completion(&req1).await.unwrap();
    assert_valid_chat_response(&resp1);
    let turn1_text = resp1.choices[0].message.content.clone().unwrap_or_default();

    // Turn 2 — exercises KV cache reuse with FP8 MoE
    let assistant_msg = ChatCompletionMessageParam {
        role: "assistant".to_string(),
        content: Some(serde_json::Value::String(turn1_text)),
        name: None,
        tool_calls: None,
        tool_call_id: None,
    };
    let req2 = ChatCompletionRequest {
        messages: vec![user_msg("Hello"), assistant_msg, user_msg("How are you?")],
        max_tokens: Some(20),
        temperature: Some(0.0),
        ..default_chat_request()
    };
    let resp2 = client.chat_completion(&req2).await.unwrap();
    assert_valid_chat_response(&resp2);
    let text2 = resp2.choices[0].message.content.as_deref().unwrap_or("");
    assert!(
        !text2.is_empty(),
        "FP8 MoE multi-turn should produce non-empty output"
    );
}
