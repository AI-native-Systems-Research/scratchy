// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Phase E15: Multimodal / Vision-Language E2E tests.
//!
//! Validates that a VLM model (Gemma3ForConditionalGeneration) can load,
//! serve text-only requests, and process image+text requests end-to-end.
//!
//! Model: mlx-community/gemma-3-4b-it-qat-3bit (~2.8 GB, Tier 3 — nightly only)
//!
//! Run with:
//! ```bash
//! cargo test -p vllm-e2e --features e2e,metal --test e_multimodal -- --ignored --test-threads=1
//! ```

#![cfg(feature = "e2e")]

use scratchy_e2e::assertions::{
    assert_coherent_text, assert_valid_chat_response, assert_valid_stream,
};
use scratchy_e2e::{Client, TestModels, TestServer};
use scratchy_serving_api::protocol::{ChatCompletionMessageParam, ChatCompletionRequest};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Pre-computed 4x4 red PNG as base64 (73 bytes raw, 100 chars encoded).
/// Generated from a solid red (#FF0000) 4x4 pixel image.
const TINY_RED_PNG_BASE64: &str = "iVBORw0KGgoAAAANSUhEUgAAAAQAAAAECAIAAAAmkwkpAAAAEElEQVR4nGP4z8AARwzEcQCukw/x0F8jngAAAABJRU5ErkJggg==";

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

/// Build a user message with both an image (base64 data URI) and text.
///
/// Follows the OpenAI content-parts format:
/// ```json
/// { "role": "user", "content": [
///     { "type": "image_url", "image_url": { "url": "data:image/png;base64,..." } },
///     { "type": "text", "text": "What is in this image?" }
/// ] }
/// ```
fn user_msg_with_image(text: &str, image_base64: &str) -> ChatCompletionMessageParam {
    let content = serde_json::json!([
        {
            "type": "image_url",
            "image_url": {
                "url": format!("data:image/png;base64,{}", image_base64)
            }
        },
        {
            "type": "text",
            "text": text
        }
    ]);
    ChatCompletionMessageParam {
        role: "user".to_string(),
        content: Some(content),
        name: None,
        tool_calls: None,
        tool_call_id: None,
    }
}

async fn start_gemma3_vlm() -> (TestServer, Client) {
    let server = TestServer::builder(TestModels::GEMMA3_VLM)
        .start()
        .await
        .expect("Gemma 3 VLM server should start");
    let client = Client::new(server.base_url());
    (server, client)
}

// ===========================================================================
// Tests
// ===========================================================================

/// Server starts with Gemma3ForConditionalGeneration architecture.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_vlm_gemma3_server_starts() {
    let (server, client) = start_gemma3_vlm().await;

    // /health
    assert!(client.health().await.unwrap(), "server should be healthy");

    // /v1/models
    let models = client.list_models().await.unwrap();
    assert_eq!(models.data.len(), 1);
    assert!(
        models.data[0].id.contains("gemma-3"),
        "model name should contain 'gemma-3'"
    );

    drop(server);
}

/// Text-only chat still works (no images — delegates to text backbone).
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_vlm_gemma3_text_only_chat() {
    let (_server, client) = start_gemma3_vlm().await;

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

/// Chat completion with a base64-encoded image + text prompt.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_vlm_gemma3_image_chat() {
    let (_server, client) = start_gemma3_vlm().await;

    let request = ChatCompletionRequest {
        messages: vec![user_msg_with_image(
            "What color is this image?",
            TINY_RED_PNG_BASE64,
        )],
        max_tokens: Some(64),
        temperature: Some(0.0),
        ..default_chat_request()
    };

    let resp = client.chat_completion(&request).await.unwrap();
    assert_valid_chat_response(&resp);

    let text = resp.choices[0].message.content.as_deref().unwrap_or("");
    // The model should produce some output (even if it doesn't perfectly
    // identify the tiny red square, it should generate tokens).
    assert_coherent_text(text, 1);
}

/// Streaming chat with image — valid SSE chunks.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_vlm_gemma3_image_stream() {
    let (_server, client) = start_gemma3_vlm().await;

    let request = ChatCompletionRequest {
        messages: vec![user_msg_with_image(
            "Describe this image briefly.",
            TINY_RED_PNG_BASE64,
        )],
        max_tokens: Some(32),
        temperature: Some(0.0),
        stream: true,
        ..default_chat_request()
    };

    let chunks = client.chat_completion_stream(&request).await.unwrap();
    assert_valid_stream(&chunks);

    // Concatenate streamed content.
    let full_text: String = chunks
        .iter()
        .filter_map(|c| c.choices.first().and_then(|ch| ch.delta.content.as_deref()))
        .collect();
    assert_coherent_text(&full_text, 1);
}

/// max_tokens is respected with image input.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_vlm_gemma3_image_max_tokens() {
    let (_server, client) = start_gemma3_vlm().await;

    let request = ChatCompletionRequest {
        messages: vec![user_msg_with_image("What do you see?", TINY_RED_PNG_BASE64)],
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
