// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! E2E tests for reasoning parser (`--reasoning-parser`).
//!
//! Tests both non-streaming and streaming paths with different parsers, on Qwen3-0.6B
//! (`TestModels::QWEN3`), a model that thinks: its reply opens `<think>…</think>` and then
//! answers, so a parser has real reasoning to split off. A parser refuses a model whose
//! vocabulary has no think tokens (as vLLM's does), so these cannot run on SmolLM. The two
//! no-parser tests run on SmolLM, whose architecture installs no parser of its own.
//!
//! On metal the two models need two builds, as `TestModels::QWEN3` is MLX 4-bit there and a
//! quant preset replaces SmolLM2's dense emission. The parser tests (on cuda drop the quant
//! feature; QWEN3 is dense there):
//! `cargo test -p scratchy-e2e -p scratchy-models -p scratchy-quantizations --features
//! e2e,metal,scratchy-models/qwen3-0.6b,scratchy-quantizations/mlx-affine-b4-g64
//! --test e_reasoning -- --ignored --test-threads=1 --skip no_reasoning_parser`
//! and the no-parser tests:
//! `cargo test -p scratchy-e2e -p scratchy-models --features e2e,metal,scratchy-models/smollm2-135m
//! --test e_reasoning -- --ignored --test-threads=1 no_reasoning_parser`

#![cfg(feature = "e2e")]

use scratchy_e2e::{Client, TestModels, TestServer};
use scratchy_serving_api::protocol::{
    ChatCompletionMessageParam, ChatCompletionRequest, ChatCompletionStreamResponse,
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
        reasoning: None,
    }
}

fn default_chat_request() -> ChatCompletionRequest {
    serde_json::from_str(r#"{"messages": []}"#).unwrap()
}

/// A short question Qwen3-0.6B thinks about and then answers within the token budget.
fn thinking_request(stream: bool) -> ChatCompletionRequest {
    ChatCompletionRequest {
        messages: vec![user_msg("Say hello.")],
        max_tokens: Some(1024),
        temperature: Some(0.0),
        stream,
        ..default_chat_request()
    }
}

/// The same question with `enable_thinking: false`, which the template answers by closing the
/// thinking itself.
fn no_thinking(stream: bool) -> ChatCompletionRequest {
    ChatCompletionRequest {
        chat_template_kwargs: Some(
            [("enable_thinking".to_string(), serde_json::json!(false))].into(),
        ),
        ..thinking_request(stream)
    }
}

async fn start_qwen3(parser: &str) -> (TestServer, Client) {
    let server = TestServer::builder(TestModels::QWEN3)
        .with_reasoning_parser(parser)
        .start()
        .await
        .expect("server should start with the reasoning parser");
    let client = Client::new(server.base_url());
    (server, client)
}

/// What the deltas of a streamed reply carried: its reasoning and its content, concatenated.
fn streamed(chunks: &[ChatCompletionStreamResponse]) -> (String, String) {
    let (mut reasoning, mut content) = (String::new(), String::new());
    for choice in chunks.iter().flat_map(|c| &c.choices) {
        reasoning.push_str(choice.delta.reasoning.as_deref().unwrap_or(""));
        content.push_str(choice.delta.content.as_deref().unwrap_or(""));
    }
    (reasoning, content)
}

/// Neither part of a split reply carries the think markers themselves.
fn assert_no_markers(part: &str, what: &str) {
    assert!(
        !part.contains("<think>") && !part.contains("</think>"),
        "{what} carries a think marker: {part:?}"
    );
}

// ===========================================================================
// deepseek_r1 parser: the thinking is reasoning, the answer content
// ===========================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_reasoning_parser_deepseek_r1_nonstreaming() {
    let (_server, client) = start_qwen3("deepseek_r1").await;
    let resp = client
        .chat_completion(&thinking_request(false))
        .await
        .unwrap();
    let msg = &resp.choices[0].message;
    let reasoning = msg.reasoning.as_deref().unwrap_or("");
    let content = msg.content.as_deref().unwrap_or("");
    assert!(
        !reasoning.trim().is_empty(),
        "the thinking is reasoning: {msg:?}"
    );
    assert!(!content.trim().is_empty(), "the answer is content: {msg:?}");
    assert_no_markers(reasoning, "reasoning");
    assert_no_markers(content, "content");
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_reasoning_parser_deepseek_r1_streaming() {
    let (_server, client) = start_qwen3("deepseek_r1").await;
    let chunks = client
        .chat_completion_stream(&thinking_request(true))
        .await
        .unwrap();
    let (reasoning, content) = streamed(&chunks);
    assert!(!reasoning.trim().is_empty(), "reasoning deltas");
    assert!(!content.trim().is_empty(), "content deltas");
    assert_no_markers(&reasoning, "streamed reasoning");
    assert_no_markers(&content, "streamed content");
}

// ===========================================================================
// qwen3 parser: the same split, and none when the request turns thinking off
// ===========================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_reasoning_parser_qwen3_nonstreaming() {
    let (_server, client) = start_qwen3("qwen3").await;
    let resp = client
        .chat_completion(&thinking_request(false))
        .await
        .unwrap();
    let msg = &resp.choices[0].message;
    let reasoning = msg.reasoning.as_deref().unwrap_or("");
    let content = msg.content.as_deref().unwrap_or("");
    assert!(
        !reasoning.trim().is_empty(),
        "thinking on: reasoning populated: {msg:?}"
    );
    assert!(
        !content.trim().is_empty(),
        "thinking on: the answer is content: {msg:?}"
    );
    assert_no_markers(reasoning, "reasoning");
    assert_no_markers(content, "content");

    // `enable_thinking: false`: the template closes thinking, so the request has no parser and
    // the whole reply is content.
    let resp = client.chat_completion(&no_thinking(false)).await.unwrap();
    let msg = &resp.choices[0].message;
    assert!(
        msg.reasoning.is_none(),
        "thinking off: no reasoning: {msg:?}"
    );
    let content = msg.content.as_deref().unwrap_or("");
    assert!(!content.trim().is_empty(), "thinking off: content: {msg:?}");
    assert_no_markers(content, "content with thinking off");
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_reasoning_parser_qwen3_streaming() {
    let (_server, client) = start_qwen3("qwen3").await;
    let chunks = client
        .chat_completion_stream(&thinking_request(true))
        .await
        .unwrap();
    let (reasoning, content) = streamed(&chunks);
    assert!(!reasoning.trim().is_empty(), "reasoning deltas");
    assert!(!content.trim().is_empty(), "content deltas");
    assert_no_markers(&reasoning, "streamed reasoning");
    assert_no_markers(&content, "streamed content");

    let chunks = client
        .chat_completion_stream(&no_thinking(true))
        .await
        .unwrap();
    let (reasoning, content) = streamed(&chunks);
    assert!(
        reasoning.is_empty(),
        "thinking off: no reasoning deltas: {reasoning:?}"
    );
    assert!(!content.trim().is_empty(), "thinking off: content deltas");
    assert_no_markers(&content, "streamed content with thinking off");
}

// ===========================================================================
// include_reasoning=false: the reasoning is dropped, the answer kept
// ===========================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_include_reasoning_false_nonstreaming() {
    let (_server, client) = start_qwen3("deepseek_r1").await;
    let request = ChatCompletionRequest {
        include_reasoning: false,
        ..thinking_request(false)
    };
    let resp = client.chat_completion(&request).await.unwrap();
    let msg = &resp.choices[0].message;
    assert!(msg.reasoning.is_none(), "reasoning suppressed: {msg:?}");
    let content = msg.content.as_deref().unwrap_or("");
    assert!(!content.trim().is_empty(), "the answer is kept: {msg:?}");
    assert_no_markers(content, "content");
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_include_reasoning_false_streaming() {
    let (_server, client) = start_qwen3("deepseek_r1").await;
    let request = ChatCompletionRequest {
        include_reasoning: false,
        ..thinking_request(true)
    };
    let chunks = client.chat_completion_stream(&request).await.unwrap();
    let (reasoning, content) = streamed(&chunks);
    assert!(reasoning.is_empty(), "no reasoning deltas: {reasoning:?}");
    assert!(!content.trim().is_empty(), "the answer is kept");
    assert_no_markers(&content, "streamed content");
}

// ===========================================================================
// No reasoning parser — verify reasoning field is absent/null
// ===========================================================================
// On SmolLM: a server installs the reasoning parser its architecture names whatever the flags
// (`auto_install_reasoning_parser`), and Qwen3's is `qwen3`, so only an architecture with none
// serves without one.

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
