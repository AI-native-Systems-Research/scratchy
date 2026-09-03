// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Mistral tool parser E2E tests.
//!
//! Validates that the `mistral` tool call parser integrates correctly into the
//! server stack. Uses SmolLM-135M as the backend model — the model won't
//! generate Mistral-format tool calls, but these tests verify:
//!
//! 1. The server starts with `mistral` parser configured.
//! 2. Non-streaming requests with tools produce valid responses (passthrough).
//! 3. Streaming requests with tools produce valid SSE chunks (passthrough).
//! 4. Requests without tools (no parser activation) work normally.
//!
//! Run with:
//! ```bash
//! # Metal/CPU:
//! cargo test -p vllm-e2e --features e2e,metal --test e5_mistral_tool_parser -- --ignored --test-threads=1
//! # CUDA:
//! cargo test -p vllm-e2e --features e2e,cuda --test e5_mistral_tool_parser -- --ignored --test-threads=1
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

fn test_model() -> &'static str {
    if cfg!(feature = "cuda") {
        TestModels::SMOLLM_135M_CUDA
    } else {
        TestModels::SMOLLM
    }
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

async fn start_smollm_with_mistral_parser() -> (TestServer, Client) {
    let server = TestServer::builder(test_model())
        .with_tool_call_parser("mistral")
        .start()
        .await
        .expect("SmolLM server with mistral parser should start");
    let client = Client::new(server.base_url());
    (server, client)
}

// ===========================================================================
// Server lifecycle
// ===========================================================================

/// Verify the server starts correctly with the mistral tool parser configured.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_mistral_server_starts() {
    let (_server, client) = start_smollm_with_mistral_parser().await;
    assert!(client.health().await.unwrap(), "server should be healthy");

    let models = client.list_models().await.unwrap();
    assert_eq!(models.data.len(), 1);
}

// ===========================================================================
// Non-streaming with tools
// ===========================================================================

/// Non-streaming request with tools — the mistral parser processes the output
/// but SmolLM won't emit Mistral-format tool calls, so we should get regular text.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_mistral_chat_with_tools() {
    let (_server, client) = start_smollm_with_mistral_parser().await;

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

// ===========================================================================
// Non-streaming without tools (passthrough)
// ===========================================================================

/// Without tools, the mistral parser should not interfere with normal output.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_mistral_chat_without_tools() {
    let (_server, client) = start_smollm_with_mistral_parser().await;

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
/// produce regular text, not Mistral-format tool calls.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_mistral_stream_with_tools() {
    let (_server, client) = start_smollm_with_mistral_parser().await;

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
    assert!(chunks.len() >= 1, "should have at least 1 stream chunk");
}

/// Streaming without tools — mistral parser should not interfere.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_mistral_stream_without_tools() {
    let (_server, client) = start_smollm_with_mistral_parser().await;

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
