// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Phase E5 (partial): Gemma 4 tool parser E2E tests.
//!
//! Validates that the `gemma4` tool call parser integrates correctly into the
//! server stack. Uses SmolLM-135M as the backend model — the model won't
//! generate Gemma-4-format tool calls, but these tests verify:
//!
//! 1. The server starts with `gemma4` parser configured.
//! 2. Non-streaming requests with tools produce valid responses (passthrough).
//! 3. Streaming requests with tools produce valid SSE chunks (passthrough).
//! 4. Requests without tools (no parser activation) work normally.
//!
//! The parser format itself (`<|tool_call>call:NAME{…}<tool_call|>`) is unit
//! tested in `scratchy-serving-api`'s `tool_parser` module; these tests cover
//! server wiring only.
//!
//! Run with:
//! ```bash
//! cargo test -p vllm-e2e --features e2e,metal --test e5_gemma4_tool_parser -- --ignored --test-threads=1
//! ```

#![cfg(feature = "e2e")]

use scratchy_e2e::assertions::{
    assert_coherent_text, assert_valid_chat_response, assert_valid_stream, collect_stream_text,
};
use scratchy_e2e::{Client, TestModels, TestServer};
use scratchy_serving_api::protocol::{
    ChatCompletionMessageParam, ChatCompletionRequest, ChatCompletionToolsParam, FunctionDefinition,
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

fn weather_tool() -> ChatCompletionToolsParam {
    ChatCompletionToolsParam {
        tool_type: "function".to_string(),
        function: FunctionDefinition {
            name: "get_weather".to_string(),
            description: Some("Get the weather for a city".to_string()),
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "city": {"type": "string"}
                },
                "required": ["city"]
            })),
        },
    }
}

async fn start_smollm_with_gemma4_parser() -> (TestServer, Client) {
    let server = TestServer::builder(TestModels::SMOLLM)
        .with_tool_call_parser("gemma4")
        .start()
        .await
        .expect("SmolLM server with gemma4 parser should start");
    let client = Client::new(server.base_url());
    (server, client)
}

// ===========================================================================
// Server lifecycle
// ===========================================================================

/// Verify the server starts correctly with the gemma4 tool parser configured.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_gemma4_server_starts() {
    scratchy_e2e::skip_if_no_gpu!();
    let (_server, client) = start_smollm_with_gemma4_parser().await;
    assert!(client.health().await.unwrap(), "server should be healthy");

    let models = client.list_models().await.unwrap();
    assert_eq!(models.data.len(), 1);
}

// ===========================================================================
// Non-streaming with tools
// ===========================================================================

/// Non-streaming request with tools — the gemma4 parser processes the output
/// but SmolLM won't emit Gemma-format tool calls, so we should get regular text.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_gemma4_chat_with_tools() {
    scratchy_e2e::skip_if_no_gpu!();
    let (_server, client) = start_smollm_with_gemma4_parser().await;

    let request = ChatCompletionRequest {
        messages: vec![user_msg("What is the weather in San Francisco?")],
        tools: Some(vec![weather_tool()]),
        max_tokens: Some(50),
        temperature: Some(0.0),
        ..default_chat_request()
    };

    let resp = client.chat_completion(&request).await.unwrap();
    assert_valid_chat_response(&resp);

    // SmolLM won't produce Gemma-format tool calls, so we expect text content.
    let text = resp.choices[0].message.content.as_deref().unwrap_or("");
    assert!(
        !text.is_empty() || resp.choices[0].message.tool_calls.is_some(),
        "response should have content or tool_calls"
    );
}

// ===========================================================================
// Non-streaming without tools (passthrough)
// ===========================================================================

/// Without tools, the gemma4 parser should not interfere with normal output.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_gemma4_chat_without_tools() {
    scratchy_e2e::skip_if_no_gpu!();
    let (_server, client) = start_smollm_with_gemma4_parser().await;

    let request = ChatCompletionRequest {
        messages: vec![user_msg("Say hello in one sentence.")],
        max_tokens: Some(30),
        temperature: Some(0.0),
        ..default_chat_request()
    };

    let resp = client.chat_completion(&request).await.unwrap();
    assert_valid_chat_response(&resp);

    let text = resp.choices[0].message.content.as_deref().unwrap_or("");
    assert_coherent_text(text, 2);
}

// ===========================================================================
// Streaming with tools
// ===========================================================================

/// Streaming request with tools — parser processes each delta but SmolLM will
/// produce regular text, not Gemma-format tool calls.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_gemma4_stream_with_tools() {
    scratchy_e2e::skip_if_no_gpu!();
    let (_server, client) = start_smollm_with_gemma4_parser().await;

    let request = ChatCompletionRequest {
        messages: vec![user_msg("What is the weather in San Francisco?")],
        tools: Some(vec![weather_tool()]),
        stream: true,
        max_tokens: Some(50),
        temperature: Some(0.0),
        ..default_chat_request()
    };

    let chunks = client.chat_completion_stream(&request).await.unwrap();
    assert_valid_stream(&chunks);

    let _text = collect_stream_text(&chunks);
    // May be empty if the model tried to produce tool calls that didn't match
    // the format, but the stream itself should be well-formed.
    assert!(!chunks.is_empty(), "should have at least 1 stream chunk");
}

/// Streaming without tools — gemma4 parser should not interfere.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_gemma4_stream_without_tools() {
    scratchy_e2e::skip_if_no_gpu!();
    let (_server, client) = start_smollm_with_gemma4_parser().await;

    let request = ChatCompletionRequest {
        messages: vec![user_msg("Say hello.")],
        stream: true,
        max_tokens: Some(30),
        temperature: Some(0.0),
        ..default_chat_request()
    };

    let chunks = client.chat_completion_stream(&request).await.unwrap();
    assert_valid_stream(&chunks);

    let text = collect_stream_text(&chunks);
    assert_coherent_text(&text, 2);
}

// ===========================================================================
// CUDA variants
// ===========================================================================

// Run with: cargo test -p vllm-e2e --features e2e,cuda --release --test e5_gemma4_tool_parser test_cuda -- --ignored --test-threads=1

#[cfg(feature = "cuda")]
async fn start_smollm_cuda_with_gemma4_parser() -> (TestServer, Client) {
    let server = TestServer::builder(TestModels::SMOLLM_135M_CUDA)
        .with_tool_call_parser("gemma4")
        .start()
        .await
        .expect("SmolLM CUDA server with gemma4 parser should start");
    let client = Client::new(server.base_url());
    (server, client)
}

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_gemma4_chat_with_tools() {
    let (_server, client) = start_smollm_cuda_with_gemma4_parser().await;

    let request = ChatCompletionRequest {
        messages: vec![user_msg("What is the weather in San Francisco?")],
        tools: Some(vec![weather_tool()]),
        max_tokens: Some(50),
        temperature: Some(0.0),
        ..default_chat_request()
    };

    let resp = client.chat_completion(&request).await.unwrap();
    assert_valid_chat_response(&resp);

    let text = resp.choices[0].message.content.as_deref().unwrap_or("");
    assert!(
        !text.is_empty() || resp.choices[0].message.tool_calls.is_some(),
        "response should have content or tool_calls"
    );
}

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_gemma4_stream_with_tools() {
    let (_server, client) = start_smollm_cuda_with_gemma4_parser().await;

    let request = ChatCompletionRequest {
        messages: vec![user_msg("What is the weather in San Francisco?")],
        tools: Some(vec![weather_tool()]),
        stream: true,
        max_tokens: Some(50),
        temperature: Some(0.0),
        ..default_chat_request()
    };

    let chunks = client.chat_completion_stream(&request).await.unwrap();
    assert_valid_stream(&chunks);
    assert!(!chunks.is_empty(), "should have at least 1 stream chunk");
}

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_gemma4_chat_without_tools() {
    let (_server, client) = start_smollm_cuda_with_gemma4_parser().await;

    let request = ChatCompletionRequest {
        messages: vec![user_msg("Say hello in one sentence.")],
        max_tokens: Some(30),
        temperature: Some(0.0),
        ..default_chat_request()
    };

    let resp = client.chat_completion(&request).await.unwrap();
    assert_valid_chat_response(&resp);

    let text = resp.choices[0].message.content.as_deref().unwrap_or("");
    assert_coherent_text(text, 2);
}
