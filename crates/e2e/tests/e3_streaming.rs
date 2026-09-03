// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Phase E3: Streaming tests.
//!
//! Validates SSE streaming for chat completions.
//!
//! Run with: `cargo test -p vllm-e2e --features e2e --test e3_streaming -- --ignored`

#![cfg(feature = "e2e")]

use scratchy_e2e::assertions::{assert_coherent_text, assert_valid_stream, collect_stream_text};
use scratchy_e2e::{Client, TestModels, TestServer};
use scratchy_serving_api::protocol::{ChatCompletionMessageParam, ChatCompletionRequest};

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

async fn start_smollm() -> (TestServer, Client) {
    let server = TestServer::builder(TestModels::SMOLLM)
        .start()
        .await
        .expect("SmolLM server should start");
    let client = Client::new(server.base_url());
    (server, client)
}

// ===========================================================================
// E3a: Basic streaming
// ===========================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_stream_basic() {
    let (_server, client) = start_smollm().await;

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

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_stream_content_matches_nonstream() {
    let (_server, client) = start_smollm().await;

    let base = ChatCompletionRequest {
        messages: vec![user_msg("What is 2+2?")],
        max_tokens: Some(20),
        temperature: Some(0.0),
        seed: Some(42),
        ..default_chat_request()
    };

    // Non-streaming
    let non_stream_resp = client.chat_completion(&base).await.unwrap();
    let non_stream_text = non_stream_resp.choices[0]
        .message
        .content
        .as_deref()
        .unwrap_or("");

    // Streaming
    let stream_req = ChatCompletionRequest {
        stream: true,
        ..base
    };
    let chunks = client.chat_completion_stream(&stream_req).await.unwrap();
    let stream_text = collect_stream_text(&chunks);

    assert_eq!(
        non_stream_text, stream_text,
        "stream and non-stream should produce same text with same seed"
    );
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_stream_last_chunk_has_finish_reason() {
    let (_server, client) = start_smollm().await;

    let request = ChatCompletionRequest {
        messages: vec![user_msg("Hello!")],
        stream: true,
        max_tokens: Some(10),
        temperature: Some(0.0),
        ..default_chat_request()
    };

    let chunks = client.chat_completion_stream(&request).await.unwrap();
    assert!(!chunks.is_empty());

    let last = chunks.last().unwrap();
    assert!(
        !last.choices.is_empty() && last.choices[0].finish_reason.is_some(),
        "last chunk should have finish_reason"
    );
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_stream_intermediate_chunks() {
    let (_server, client) = start_smollm().await;

    let request = ChatCompletionRequest {
        messages: vec![user_msg("Write a sentence about dogs.")],
        stream: true,
        max_tokens: Some(30),
        temperature: Some(0.0),
        ..default_chat_request()
    };

    let chunks = client.chat_completion_stream(&request).await.unwrap();

    if chunks.len() > 2 {
        // Middle chunks should have content but no finish_reason
        for chunk in &chunks[1..chunks.len() - 1] {
            for choice in &chunk.choices {
                assert!(
                    choice.finish_reason.is_none(),
                    "intermediate chunks should not have finish_reason"
                );
            }
        }
    }
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_stream_max_tokens() {
    let (_server, client) = start_smollm().await;

    let request = ChatCompletionRequest {
        messages: vec![user_msg("Write a long essay about history.")],
        stream: true,
        max_tokens: Some(5),
        temperature: Some(0.0),
        ..default_chat_request()
    };

    let chunks = client.chat_completion_stream(&request).await.unwrap();
    assert_valid_stream(&chunks);

    // The last chunk's finish_reason should be "length"
    let last = chunks.last().unwrap();
    assert_eq!(
        last.choices[0].finish_reason.as_deref(),
        Some("length"),
        "finish_reason should be 'length' when max_tokens is reached"
    );
}

// ===========================================================================
// E3b: Streaming with n>1
// ===========================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_stream_n2() {
    let (_server, client) = start_smollm().await;

    let request = ChatCompletionRequest {
        messages: vec![user_msg("Hello!")],
        stream: true,
        n: 2,
        max_tokens: Some(15),
        temperature: Some(0.8),
        ..default_chat_request()
    };

    let chunks = client.chat_completion_stream(&request).await.unwrap();
    assert!(!chunks.is_empty());

    // Collect indices we've seen
    let mut seen_indices: std::collections::HashSet<u32> = std::collections::HashSet::new();
    let mut finished_indices: std::collections::HashSet<u32> = std::collections::HashSet::new();

    for chunk in &chunks {
        for choice in &chunk.choices {
            seen_indices.insert(choice.index);
            if choice.finish_reason.is_some() {
                finished_indices.insert(choice.index);
            }
        }
    }

    assert!(seen_indices.contains(&0), "should have chunks for index 0");
    assert!(seen_indices.contains(&1), "should have chunks for index 1");
    assert!(
        finished_indices.contains(&0) && finished_indices.contains(&1),
        "both choices should have finish_reason"
    );
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_stream_n3_all_finish() {
    let (_server, client) = start_smollm().await;

    let request = ChatCompletionRequest {
        messages: vec![user_msg("Hello!")],
        stream: true,
        n: 3,
        max_tokens: Some(10),
        temperature: Some(0.8),
        ..default_chat_request()
    };

    let chunks = client.chat_completion_stream(&request).await.unwrap();

    let mut finished: std::collections::HashSet<u32> = std::collections::HashSet::new();
    for chunk in &chunks {
        for choice in &chunk.choices {
            if choice.finish_reason.is_some() {
                finished.insert(choice.index);
            }
        }
    }

    assert_eq!(
        finished.len(),
        3,
        "all 3 choices should finish, got {:?}",
        finished
    );
}

// ===========================================================================
// E3c: Streaming edge cases
// ===========================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_stream_empty_response() {
    let (_server, client) = start_smollm().await;

    let request = ChatCompletionRequest {
        messages: vec![user_msg("Hi")],
        stream: true,
        max_tokens: Some(1),
        temperature: Some(0.0),
        ..default_chat_request()
    };

    let chunks = client.chat_completion_stream(&request).await.unwrap();
    // Should have at least one chunk even with max_tokens=1
    assert!(
        !chunks.is_empty(),
        "should have at least one chunk even with max_tokens=1"
    );
}
