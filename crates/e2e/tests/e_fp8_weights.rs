// SPDX-License-Identifier: Apache-2.0

//! E2E tests for FP8 weight quantization (quant_method: "fp8").
//!
//! These models have weights stored as float8_e4m3fn with per-tensor weight
//! scales. The CUDA backend loads them via `Fp8Linear` layers and uses
//! fused CUTLASS `cutlass_scaled_mm` FP8 GEMMs with per-row activation
//! scales fused into the epilogue (single kernel launch).
//!
//! Requires SM89+ GPU (L40S, H100).
//!
//! Run with:
//!   cargo test -p vllm-e2e --features e2e,cuda --release --test e_fp8_weights -- --ignored --test-threads=1

#![cfg(all(feature = "e2e", feature = "cuda"))]

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
// Qwen2.5-0.5B FP8 (small, ~500MB — fast CI test)
// ===========================================================================

/// FP8 weight server starts and passes health check.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_fp8_qwen2_server_starts() {
    let server = TestServer::builder(TestModels::QWEN2_0_5B_FP8)
        .with_args(&["--enforce-eager"])
        .start()
        .await
        .expect("FP8 Qwen2 server should start");

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

/// FP8 weight completion generates coherent text with correct semantic content.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_fp8_qwen2_completion() {
    let server = TestServer::builder(TestModels::QWEN2_0_5B_FP8)
        .with_args(&["--enforce-eager"])
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = simple_completion_request("The capital of France is", 20);
    let resp = client.completion(&request).await.unwrap();

    assert_valid_completion_response(&resp);
    let text = resp.choices[0].text.to_lowercase();
    assert!(
        text.contains("paris"),
        "FP8 completion of 'The capital of France is' should contain 'paris', got: {text}"
    );
}

/// FP8 weight chat completion generates correct answer.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_fp8_qwen2_chat() {
    let server = TestServer::builder(TestModels::QWEN2_0_5B_FP8)
        .with_args(&["--enforce-eager"])
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = simple_chat_request("What is 2+2? Answer with just the number.", Some(10));
    let resp = client.chat_completion(&request).await.unwrap();

    assert_valid_chat_response(&resp);
    let text = resp.choices[0].message.content.as_deref().unwrap_or("");
    assert!(
        text.contains('4'),
        "FP8 chat '2+2' should contain '4', got: {text}"
    );
}

/// FP8 weight model respects max_tokens limit.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_fp8_qwen2_max_tokens() {
    let server = TestServer::builder(TestModels::QWEN2_0_5B_FP8)
        .with_args(&["--enforce-eager"])
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

/// FP8 weight model works with CUDA graphs (non-eager mode).
/// TODO: FP8 + CUDA graphs requires pre-allocated FP8 activation/scale buffers.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_fp8_qwen2_cuda_graphs() {
    let server = TestServer::builder(TestModels::QWEN2_0_5B_FP8)
        .with_args(&["--enforce-eager"])
        .start()
        .await
        .expect("FP8 server with CUDA graphs should start");

    let client = Client::new(server.base_url());
    let request = simple_completion_request("The capital of France is", 20);
    let resp = client.completion(&request).await.unwrap();

    assert_valid_completion_response(&resp);
    let text = resp.choices[0].text.to_lowercase();
    assert!(
        text.contains("paris"),
        "FP8 + CUDA graphs should produce 'paris', got: {text}"
    );
}

/// Multi-turn chat with context recall — verifies KV cache correctness with FP8 weights.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_fp8_qwen2_multi_turn() {
    let server = TestServer::builder(TestModels::QWEN2_0_5B_FP8)
        .with_args(&["--enforce-eager"])
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());

    // Turn 1: establish a fact
    let req1 = ChatCompletionRequest {
        messages: vec![user_msg(
            "My favorite color is blue. Please remember that. Reply with just 'OK'.",
        )],
        max_tokens: Some(10),
        temperature: Some(0.0),
        ..default_chat_request()
    };
    let resp1 = client.chat_completion(&req1).await.unwrap();
    assert_valid_chat_response(&resp1);
    let turn1_text = resp1.choices[0].message.content.clone().unwrap_or_default();

    // Turn 2: query the fact — requires correct KV cache from turn 1
    let assistant_msg = ChatCompletionMessageParam {
        role: "assistant".to_string(),
        content: Some(serde_json::Value::String(turn1_text)),
        name: None,
        tool_calls: None,
        tool_call_id: None,
    };
    let req2 = ChatCompletionRequest {
        messages: vec![
            user_msg("My favorite color is blue. Please remember that. Reply with just 'OK'."),
            assistant_msg,
            user_msg("What is my favorite color?"),
        ],
        max_tokens: Some(20),
        temperature: Some(0.0),
        ..default_chat_request()
    };
    let resp2 = client.chat_completion(&req2).await.unwrap();
    assert_valid_chat_response(&resp2);
    let text2 = resp2.choices[0]
        .message
        .content
        .as_deref()
        .unwrap_or("")
        .to_lowercase();
    assert!(
        text2.contains("blue"),
        "FP8 multi-turn should recall 'blue', got: {text2}"
    );
}

// ===========================================================================
// Meta-Llama-3.1-8B-Instruct-FP8 (larger, ~8GB — nightly/manual only)
// ===========================================================================

/// LLaMA 3.1 8B FP8 server starts and passes health check.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_fp8_llama8b_server_starts() {
    let server = TestServer::builder(TestModels::LLAMA_3_1_8B_FP8)
        .with_args(&["--enforce-eager"])
        .start()
        .await
        .expect("FP8 LLaMA 8B server should start");

    let client = Client::new(server.base_url());
    assert!(client.health().await.unwrap(), "server should be healthy");
}

/// LLaMA 3.1 8B FP8 generates coherent chat response with correct content.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_fp8_llama8b_chat() {
    let server = TestServer::builder(TestModels::LLAMA_3_1_8B_FP8)
        .with_args(&["--enforce-eager"])
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = simple_chat_request(
        "What is the largest planet in our solar system? Answer in one word.",
        Some(10),
    );
    let resp = client.chat_completion(&request).await.unwrap();

    assert_valid_chat_response(&resp);
    let text = resp.choices[0]
        .message
        .content
        .as_deref()
        .unwrap_or("")
        .to_lowercase();
    assert_coherent_text(&text, 3);
    assert!(
        text.contains("jupiter"),
        "FP8 LLaMA 8B should answer 'Jupiter', got: {text}"
    );
}

/// LLaMA 3.1 8B FP8 completion generates semantically correct output.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_fp8_llama8b_completion() {
    let server = TestServer::builder(TestModels::LLAMA_3_1_8B_FP8)
        .with_args(&["--enforce-eager"])
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = simple_completion_request("The capital of France is", 20);
    let resp = client.completion(&request).await.unwrap();

    assert_valid_completion_response(&resp);
    let text = resp.choices[0].text.to_lowercase();
    assert!(
        text.contains("paris"),
        "FP8 LLaMA 8B completion should contain 'paris', got: {text}"
    );
}
