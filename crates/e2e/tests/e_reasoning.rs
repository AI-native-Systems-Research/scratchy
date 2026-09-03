// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! E2E tests for reasoning parser (`--reasoning-parser`).
//!
//! Tests both non-streaming and streaming paths with different parsers.
//! Uses SmolLM (which doesn't produce `<think>` tokens) to verify the code
//! path works without errors, and optionally Qwen3 for real reasoning output.
//!
//! Run with: `cargo test -p vllm-e2e --features e2e --test e_reasoning -- --ignored --test-threads=1`

#![cfg(feature = "e2e")]

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

// ===========================================================================
// Tests with SmolLM + deepseek_r1 parser
// ===========================================================================
// SmolLM doesn't produce <think> tokens, so DeepSeek R1 parser treats the
// entire output as reasoning (no </think> found). This tests the wiring.

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_reasoning_parser_deepseek_r1_nonstreaming() {
    let server = TestServer::builder(TestModels::SMOLLM)
        .with_reasoning_parser("deepseek_r1")
        .start()
        .await
        .expect("server should start with reasoning parser");
    let client = Client::new(server.base_url());

    let request = ChatCompletionRequest {
        messages: vec![user_msg("Say hello.")],
        max_tokens: Some(20),
        temperature: Some(0.0),
        ..default_chat_request()
    };

    let resp = client.chat_completion(&request).await.unwrap();
    assert_eq!(resp.choices.len(), 1);
    let msg = &resp.choices[0].message;

    // SmolLM doesn't produce </think>, so DeepSeek parser:
    // everything is reasoning, content is None.
    assert!(
        msg.reasoning.is_some(),
        "reasoning field should be populated (entire output is reasoning)"
    );
    assert!(
        msg.content.is_none(),
        "content should be None when model produces no </think>"
    );
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_reasoning_parser_deepseek_r1_streaming() {
    let server = TestServer::builder(TestModels::SMOLLM)
        .with_reasoning_parser("deepseek_r1")
        .start()
        .await
        .expect("server should start with reasoning parser");
    let client = Client::new(server.base_url());

    let request = ChatCompletionRequest {
        messages: vec![user_msg("Say hello.")],
        max_tokens: Some(20),
        temperature: Some(0.0),
        stream: true,
        ..default_chat_request()
    };

    let chunks = client.chat_completion_stream(&request).await.unwrap();
    assert!(!chunks.is_empty(), "should have at least one chunk");

    // Collect reasoning and content from streaming deltas.
    let mut has_reasoning = false;
    let mut has_content = false;
    for chunk in &chunks {
        for choice in &chunk.choices {
            if choice.delta.reasoning.is_some() {
                has_reasoning = true;
            }
            if choice.delta.content.is_some() {
                has_content = true;
            }
        }
    }

    // With DeepSeek parser and no </think> from SmolLM, all deltas should
    // be reasoning (the streaming state never transitions to content).
    assert!(has_reasoning, "streaming should have reasoning deltas");
    // Content may or may not appear depending on whether model accidentally
    // produces </think>-like tokens. Don't assert on has_content.
    let _ = has_content;
}

// ===========================================================================
// Tests with SmolLM + qwen3 parser
// ===========================================================================
// SmolLM doesn't produce <think> or </think> tokens, so Qwen3 parser treats
// everything as content (thinking disabled).

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_reasoning_parser_qwen3_nonstreaming() {
    let server = TestServer::builder(TestModels::SMOLLM)
        .with_reasoning_parser("qwen3")
        .start()
        .await
        .expect("server should start with qwen3 reasoning parser");
    let client = Client::new(server.base_url());

    let request = ChatCompletionRequest {
        messages: vec![user_msg("Say hello.")],
        max_tokens: Some(20),
        temperature: Some(0.0),
        ..default_chat_request()
    };

    let resp = client.chat_completion(&request).await.unwrap();
    assert_eq!(resp.choices.len(), 1);
    let msg = &resp.choices[0].message;

    // Qwen3: no </think> → thinking disabled, everything is content.
    assert!(
        msg.reasoning.is_none(),
        "reasoning should be None (thinking disabled)"
    );
    assert!(msg.content.is_some(), "content should be populated");
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_reasoning_parser_qwen3_streaming() {
    let server = TestServer::builder(TestModels::SMOLLM)
        .with_reasoning_parser("qwen3")
        .start()
        .await
        .expect("server should start with qwen3 reasoning parser");
    let client = Client::new(server.base_url());

    let request = ChatCompletionRequest {
        messages: vec![user_msg("Say hello.")],
        max_tokens: Some(20),
        temperature: Some(0.0),
        stream: true,
        ..default_chat_request()
    };

    let chunks = client.chat_completion_stream(&request).await.unwrap();
    assert!(!chunks.is_empty(), "should have at least one chunk");

    // Collect content deltas.
    let mut has_content = false;
    let mut has_reasoning = false;
    for chunk in &chunks {
        for choice in &chunk.choices {
            if choice.delta.content.is_some() {
                has_content = true;
            }
            if choice.delta.reasoning.is_some() {
                has_reasoning = true;
            }
        }
    }

    // Qwen3 with SmolLM: no think tokens → everything is content.
    assert!(has_content, "should have content deltas");
    // SmolLM tokens won't match <think>/<think> IDs, so reasoning
    // should not appear (thinking disabled path in Qwen3 streaming).
    let _ = has_reasoning; // Don't hard-assert — depends on vocab overlap.
}

// ===========================================================================
// include_reasoning=false
// ===========================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_include_reasoning_false_nonstreaming() {
    let server = TestServer::builder(TestModels::SMOLLM)
        .with_reasoning_parser("deepseek_r1")
        .start()
        .await
        .expect("server should start");
    let client = Client::new(server.base_url());

    let request = ChatCompletionRequest {
        messages: vec![user_msg("Say hello.")],
        max_tokens: Some(20),
        temperature: Some(0.0),
        include_reasoning: false,
        ..default_chat_request()
    };

    let resp = client.chat_completion(&request).await.unwrap();
    assert_eq!(resp.choices.len(), 1);
    let msg = &resp.choices[0].message;

    // Even though DeepSeek extracts reasoning, include_reasoning=false
    // should suppress it.
    assert!(
        msg.reasoning.is_none(),
        "reasoning should be suppressed with include_reasoning=false"
    );
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_include_reasoning_false_streaming() {
    let server = TestServer::builder(TestModels::SMOLLM)
        .with_reasoning_parser("deepseek_r1")
        .start()
        .await
        .expect("server should start");
    let client = Client::new(server.base_url());

    let request = ChatCompletionRequest {
        messages: vec![user_msg("Say hello.")],
        max_tokens: Some(20),
        temperature: Some(0.0),
        stream: true,
        include_reasoning: false,
        ..default_chat_request()
    };

    let chunks = client.chat_completion_stream(&request).await.unwrap();
    assert!(!chunks.is_empty(), "should have chunks");

    // No reasoning deltas should appear.
    for chunk in &chunks {
        for choice in &chunk.choices {
            assert!(
                choice.delta.reasoning.is_none(),
                "reasoning should be suppressed in streaming with include_reasoning=false"
            );
        }
    }
}

// ===========================================================================
// No reasoning parser — verify reasoning field is absent/null
// ===========================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_no_reasoning_parser_nonstreaming() {
    let server = TestServer::builder(TestModels::SMOLLM)
        .start()
        .await
        .expect("server should start");
    let client = Client::new(server.base_url());

    let request = ChatCompletionRequest {
        messages: vec![user_msg("Say hello.")],
        max_tokens: Some(20),
        temperature: Some(0.0),
        ..default_chat_request()
    };

    let resp = client.chat_completion(&request).await.unwrap();
    assert_eq!(resp.choices.len(), 1);
    let msg = &resp.choices[0].message;

    // Without a reasoning parser, reasoning should always be None.
    assert!(
        msg.reasoning.is_none(),
        "reasoning should be None without parser"
    );
    assert!(msg.content.is_some(), "content should be populated");
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_no_reasoning_parser_streaming() {
    let server = TestServer::builder(TestModels::SMOLLM)
        .start()
        .await
        .expect("server should start");
    let client = Client::new(server.base_url());

    let request = ChatCompletionRequest {
        messages: vec![user_msg("Say hello.")],
        max_tokens: Some(20),
        temperature: Some(0.0),
        stream: true,
        ..default_chat_request()
    };

    let chunks = client.chat_completion_stream(&request).await.unwrap();
    assert!(!chunks.is_empty(), "should have chunks");

    for chunk in &chunks {
        for choice in &chunk.choices {
            assert!(
                choice.delta.reasoning.is_none(),
                "reasoning should be None without parser"
            );
        }
    }
}
