// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! E2E tests for the Anthropic `/v1/messages` endpoint.
//!
//! Run with: `cargo test -p vllm-e2e --features e2e --test e_anthropic -- --ignored`

#![cfg(feature = "e2e")]

use scratchy_e2e::{Client, TestModels, TestServer};
use serde_json::json;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

async fn start_smollm() -> (TestServer, Client) {
    let server = TestServer::builder(TestModels::SMOLLM)
        .start()
        .await
        .expect("SmolLM server should start");
    let client = Client::new(server.base_url());
    (server, client)
}

/// Tools-bearing requests shaped like the traffic Claude Code sends: a system
/// prompt, several tool schemas, a prior tool result, and client-declared
/// `cache_control` breakpoints. Every shape here has to be servable — Claude
/// Code sends all of them within one session.
fn tools_corpus() -> Vec<serde_json::Value> {
    let bash_tool = json!({
        "name": "Bash",
        "description": "Run a shell command and return its output.",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"]
        }
    });
    let read_tool = json!({
        "name": "Read",
        "description": "Read a file from the local filesystem.",
        "input_schema": {
            "type": "object",
            "properties": {"file_path": {"type": "string"}},
            "required": ["file_path"]
        }
    });

    vec![
        // One tool, no breakpoints — the conservative shape.
        json!({
            "max_tokens": 24,
            "temperature": 0.0,
            "messages": [{"role": "user", "content": "List the files here."}],
            "tools": [bash_tool]
        }),
        // Two tools plus a system prompt.
        json!({
            "max_tokens": 24,
            "temperature": 0.0,
            "system": "You are a terse coding assistant.",
            "messages": [{"role": "user", "content": "What is in README.md?"}],
            "tools": [bash_tool, read_tool]
        }),
        // A client breakpoint on the system block: everything up to and
        // including it becomes a relocatable span in a `tool-spans` build.
        json!({
            "max_tokens": 24,
            "temperature": 0.0,
            "system": [{"text": "You are a terse coding assistant.", "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": "Count the Rust files."}],
            "tools": [bash_tool, read_tool]
        }),
        // A prior `tool_use` + `tool_result` — i.e. every Claude Code turn
        // after the first. This maps to an OpenAI assistant message carrying
        // `tool_calls` and no `content`, which used to 500 for every template
        // that writes `message['content'] + …` (granite 3.3, SmolLM2): the
        // key was absent rather than empty. Keep this shape in the corpus; it
        // is the regression guard for that fix.
        json!({
            "max_tokens": 24,
            "temperature": 0.0,
            "messages": [
                {"role": "user", "content": "Run `ls`."},
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {"command": "ls"}}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": "Cargo.toml\nsrc"}
                ]}
            ],
            "tools": [bash_tool]
        }),
    ]
}

/// Collect the concatenated text of every `text` block in a response.
fn response_text(body: &serde_json::Value) -> String {
    body["content"]
        .as_array()
        .expect("content array")
        .iter()
        .filter(|b| b["type"] == "text")
        .filter_map(|b| b["text"].as_str())
        .collect::<Vec<_>>()
        .join("")
}

/// Run the whole corpus, returning each response's text.
async fn run_tools_corpus(client: &Client) -> Vec<String> {
    let mut out = Vec::new();
    for req in tools_corpus() {
        let resp = client.anthropic_messages_raw(&req).await.unwrap();
        assert!(
            resp.status().is_success(),
            "status {} for {req}",
            resp.status()
        );
        let body: serde_json::Value = resp.json().await.unwrap();
        out.push(response_text(&body));
    }
    out
}

// ===========================================================================
// Non-streaming
// ===========================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_anthropic_simple_message() {
    let (_server, client) = start_smollm().await;

    let resp = client
        .anthropic_messages_raw(&json!({
            "max_tokens": 20,
            "messages": [{"role": "user", "content": "Say hello"}]
        }))
        .await
        .unwrap();

    assert!(resp.status().is_success(), "status: {}", resp.status());

    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["type"], "message");
    assert_eq!(body["role"], "assistant");
    assert!(body["content"].is_array());
    assert!(!body["content"].as_array().unwrap().is_empty());
    assert_eq!(body["content"][0]["type"], "text");
    assert!(body["content"][0]["text"].as_str().unwrap().len() > 0);
    assert!(body["usage"]["input_tokens"].as_u64().unwrap() > 0);
    assert!(body["usage"]["output_tokens"].as_u64().unwrap() > 0);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_anthropic_with_system() {
    let (_server, client) = start_smollm().await;

    let resp = client
        .anthropic_messages_raw(&json!({
            "max_tokens": 20,
            "system": "You are a helpful assistant.",
            "messages": [{"role": "user", "content": "Hi"}]
        }))
        .await
        .unwrap();

    assert!(resp.status().is_success());
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["type"], "message");
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_anthropic_system_blocks() {
    let (_server, client) = start_smollm().await;

    let resp = client
        .anthropic_messages_raw(&json!({
            "max_tokens": 20,
            "system": [{"text": "Be concise."}, {"text": "Be accurate."}],
            "messages": [{"role": "user", "content": "Hi"}]
        }))
        .await
        .unwrap();

    assert!(resp.status().is_success());
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_anthropic_multi_turn() {
    let (_server, client) = start_smollm().await;

    let resp = client
        .anthropic_messages_raw(&json!({
            "max_tokens": 20,
            "messages": [
                {"role": "user", "content": "My name is Alice."},
                {"role": "assistant", "content": "Hello Alice!"},
                {"role": "user", "content": "What is my name?"}
            ]
        }))
        .await
        .unwrap();

    assert!(resp.status().is_success());
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["type"], "message");
    assert!(!body["content"].as_array().unwrap().is_empty());
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_anthropic_content_blocks() {
    let (_server, client) = start_smollm().await;

    let resp = client
        .anthropic_messages_raw(&json!({
            "max_tokens": 20,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is 2+2?"},
                    {"type": "text", "text": "Tell me quickly."}
                ]
            }]
        }))
        .await
        .unwrap();

    assert!(resp.status().is_success());
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_anthropic_temperature() {
    let (_server, client) = start_smollm().await;

    let resp = client
        .anthropic_messages_raw(&json!({
            "max_tokens": 20,
            "temperature": 0.0,
            "messages": [{"role": "user", "content": "Count to 5"}]
        }))
        .await
        .unwrap();

    assert!(resp.status().is_success());
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_anthropic_stop_sequences() {
    let (_server, client) = start_smollm().await;

    let resp = client
        .anthropic_messages_raw(&json!({
            "max_tokens": 100,
            "stop_sequences": ["."],
            "messages": [{"role": "user", "content": "Tell me a story"}]
        }))
        .await
        .unwrap();

    assert!(resp.status().is_success());
    let body: serde_json::Value = resp.json().await.unwrap();
    // Should stop at first period or before max_tokens
    let text = body["content"][0]["text"].as_str().unwrap_or("");
    // The text should not contain a period (it stops before outputting it by default)
    assert!(text.len() < 500, "should have stopped early");
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_anthropic_max_tokens_respected() {
    let (_server, client) = start_smollm().await;

    let resp = client
        .anthropic_messages_raw(&json!({
            "max_tokens": 5,
            "messages": [{"role": "user", "content": "Write a long essay about everything"}]
        }))
        .await
        .unwrap();

    assert!(resp.status().is_success());
    let body: serde_json::Value = resp.json().await.unwrap();
    // With max_tokens=5, output should be short
    assert!(body["usage"]["output_tokens"].as_u64().unwrap() <= 6);
}

// ===========================================================================
// Streaming
// ===========================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_anthropic_streaming() {
    let (_server, client) = start_smollm().await;

    let events = client
        .anthropic_messages_stream(&json!({
            "max_tokens": 20,
            "stream": true,
            "messages": [{"role": "user", "content": "Say hello"}]
        }))
        .await
        .unwrap();

    assert!(!events.is_empty(), "should have received SSE events");

    // Check event sequence
    let types: Vec<&str> = events.iter().filter_map(|e| e["type"].as_str()).collect();

    assert_eq!(
        types[0], "message_start",
        "first event should be message_start"
    );
    assert!(types.contains(&"ping"), "should contain ping");
    assert!(
        types.contains(&"content_block_start"),
        "should contain content_block_start"
    );
    assert!(
        types.contains(&"content_block_delta"),
        "should contain content_block_delta"
    );
    assert!(
        types.contains(&"content_block_stop"),
        "should contain content_block_stop"
    );
    assert!(
        types.contains(&"message_delta"),
        "should contain message_delta"
    );
    assert!(
        types.contains(&"message_stop"),
        "should contain message_stop"
    );

    // Check message_start has expected structure
    let msg_start = &events[0];
    assert_eq!(msg_start["message"]["type"], "message");
    assert_eq!(msg_start["message"]["role"], "assistant");

    // Check we got text deltas
    let text_deltas: Vec<&serde_json::Value> = events
        .iter()
        .filter(|e| e["type"] == "content_block_delta")
        .collect();
    assert!(!text_deltas.is_empty());
    for td in &text_deltas {
        assert_eq!(td["delta"]["type"], "text_delta");
        assert!(td["delta"]["text"].is_string());
    }

    // Check message_delta has stop_reason
    let msg_delta = events
        .iter()
        .find(|e| e["type"] == "message_delta")
        .unwrap();
    assert!(msg_delta["delta"]["stop_reason"].is_string());
    assert!(msg_delta["usage"]["output_tokens"].as_u64().unwrap() > 0);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_anthropic_streaming_with_system() {
    let (_server, client) = start_smollm().await;

    let events = client
        .anthropic_messages_stream(&json!({
            "max_tokens": 10,
            "stream": true,
            "system": "Reply in one word.",
            "messages": [{"role": "user", "content": "Hi"}]
        }))
        .await
        .unwrap();

    let types: Vec<&str> = events.iter().filter_map(|e| e["type"].as_str()).collect();
    assert!(types.contains(&"message_start"));
    assert!(types.contains(&"message_stop"));
}

// ===========================================================================
// Error cases
// ===========================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_anthropic_missing_max_tokens() {
    let (_server, client) = start_smollm().await;

    // max_tokens is required in Anthropic API
    let resp = client
        .anthropic_messages_raw(&json!({
            "messages": [{"role": "user", "content": "Hi"}]
        }))
        .await
        .unwrap();

    // Should fail with 422 (Unprocessable Entity) since max_tokens is required
    assert!(
        !resp.status().is_success(),
        "should fail without max_tokens"
    );
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_anthropic_empty_messages() {
    let (_server, client) = start_smollm().await;

    let resp = client
        .anthropic_messages_raw(&json!({
            "max_tokens": 10,
            "messages": []
        }))
        .await
        .unwrap();

    // Empty messages should fail at the engine level
    assert!(!resp.status().is_success());
}

// ===========================================================================
// Tools-bearing requests on the shipped path
// ===========================================================================
//
// There is no A/B here any more. Per-tool relocatable spans are gated behind
// the off-by-default `tool-spans` Cargo feature (issue #193), so a build either
// has that code or it does not, and these tests exercise whichever path the
// build compiled. A default build — what CI and users get — takes the flat chat
// path, the one that renders tools through the model's own template.
//
// Comparing the two arms is a two-build job now, not something one test binary
// can do. Recorded here so the next reader does not go looking for the switch:
// on granite-3.3-2b-instruct-4bit, one `Bash` tool, the same request is 105
// prompt tokens and a `text` block under `tool-spans`, against 218 tokens and a
// real `tool_use` block without it.

/// Every shape in the corpus must be served, whichever path is compiled.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_tools_corpus_is_served() {
    let (_server, client) = start_smollm().await;
    let texts = run_tools_corpus(&client).await;
    assert_eq!(texts.len(), tools_corpus().len());
}

/// A tools-bearing request must be answered with a well-formed message.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_tools_request_returns_a_message() {
    let (_server, client) = start_smollm().await;

    let resp = client
        .anthropic_messages_raw(&tools_corpus()[0])
        .await
        .unwrap();

    assert!(resp.status().is_success(), "status: {}", resp.status());
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["type"], "message");
    assert!(!body["content"].as_array().unwrap().is_empty());
}

/// A tools-bearing request that names no `model` must be served.
///
/// `model` is optional on `/v1/messages` and the flat path just forwards the
/// `None`. The spans path has to render SPNL's required `Generate.model`, and
/// rendering it as `null` made the SPNL parse fail and 400'd a request the flat
/// path serves — so this is the regression guard for a `tool-spans` build, and
/// a cheap sanity check for every other one.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_model_less_tools_request_is_served() {
    let (_server, client) = start_smollm().await;
    let body = json!({
        "max_tokens": 16,
        "temperature": 0.0,
        "messages": [{"role": "user", "content": "List the files here."}],
        "tools": [{
            "name": "Bash",
            "description": "Run a shell command and return its output.",
            "input_schema": {
                "type": "object",
                "properties": {"command": {"type": "string"}}
            }
        }]
    });

    let resp = client.anthropic_messages_raw(&body).await.unwrap();
    assert!(
        resp.status().is_success(),
        "a model-less request was rejected: {}",
        resp.status()
    );
    // The response names the model actually loaded, not an empty string.
    let got: serde_json::Value = resp.json().await.unwrap();
    assert!(
        !got["model"].as_str().unwrap_or("").is_empty(),
        "returned an empty model"
    );
}
