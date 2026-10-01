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

/// Start SmolLM with per-tool relocatable spans explicitly on or off — the two
/// arms of `scr serve --tool-spans` (off is the default).
async fn start_smollm_tool_spans(enabled: bool) -> (TestServer, Client) {
    let server = TestServer::builder(TestModels::SMOLLM)
        .with_tool_spans(enabled)
        .start()
        .await
        .expect("SmolLM server should start");
    let client = Client::new(server.base_url());
    (server, client)
}

/// Tools-bearing requests shaped like the traffic Claude Code sends: a system
/// prompt, several tool schemas, a prior tool result, and client-declared
/// `cache_control` breakpoints.
fn tool_spans_corpus() -> Vec<serde_json::Value> {
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
        // including it becomes a relocatable span on the spans arm.
        json!({
            "max_tokens": 24,
            "temperature": 0.0,
            "system": [{"text": "You are a terse coding assistant.", "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": "Count the Rust files."}],
            "tools": [bash_tool, read_tool]
        }),
    ]
    // NOT in the corpus: a turn carrying a prior `tool_use` + `tool_result`.
    // The flat arm maps a tool_use-only assistant block to an OpenAI message
    // with `tool_calls` and no `content`, and rendering that 500s on both
    // SmolLM2 and Granite 3.3 ("tried to use + operator on unsupported types
    // string and undefined"). It is not specific to this endpoint —
    // `/v1/chat/completions` 500s on the same message — so it is a separate
    // bug, not something the spans switch should paper over. Add the shape
    // here once that is fixed.
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

/// Run the whole corpus against one arm, returning each response's text.
async fn run_tool_spans_corpus(client: &Client) -> Vec<String> {
    let mut out = Vec::new();
    for req in tool_spans_corpus() {
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
// Tool spans A/B (`scr serve --tool-spans`)
// ===========================================================================

/// Both arms of the A/B must serve the same tools-bearing requests.
///
/// This is the half of the switch that has to hold for an ablation to mean
/// anything: a request either arm rejects is a request the comparison cannot
/// use. It caught one real asymmetry — a client that omits the optional `model`
/// field 400'd on the spans arm (SPNL's `Generate.model` is a required string)
/// while the flat arm served it.
///
/// It deliberately does NOT assert that the two arms produce the same text.
/// They do not, and the switch is what made that measurable: the spans arm
/// renders tool schemas as plain `Tool: …` text instead of through the model's
/// native tool template, so on Granite 3.3 the same request is 105 prompt
/// tokens and comes back as a `text` block, against 218 tokens and a real
/// `tool_use` block on the flat arm. That is why spans are opt-in rather than
/// the default. See the T0.3 notes on issue #162.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_tool_spans_both_arms_serve_the_corpus() {
    for enabled in [true, false] {
        let (_server, client) = start_smollm_tool_spans(enabled).await;
        let texts = run_tool_spans_corpus(&client).await;
        assert_eq!(
            texts.len(),
            tool_spans_corpus().len(),
            "tool_spans={enabled} did not serve the whole corpus"
        );
    }
}

/// A tools-bearing request that names no `model` must be served by both arms.
///
/// `model` is optional on `/v1/messages` and the flat path just forwards the
/// `None`; the spans path has to render SPNL's required `Generate.model`, and
/// rendering it as `null` made the SPNL parse fail and 400'd the request. An
/// arm that rejects requests the other serves cannot be A/B'd against it.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_tool_spans_arms_agree_on_model_less_requests() {
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

    for enabled in [true, false] {
        let (_server, client) = start_smollm_tool_spans(enabled).await;
        let resp = client.anthropic_messages_raw(&body).await.unwrap();
        assert!(
            resp.status().is_success(),
            "tool_spans={enabled} rejected a model-less request: {}",
            resp.status()
        );
        // Both arms report the model they actually loaded, not an empty string.
        let got: serde_json::Value = resp.json().await.unwrap();
        assert!(
            !got["model"].as_str().unwrap_or("").is_empty(),
            "tool_spans={enabled} returned an empty model"
        );
    }
}

/// The default arm must serve tools-bearing requests — it takes the flat chat
/// path, so a `tool_use` block can only come from the tool parser, never from
/// the spans renderer.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_default_arm_serves_tools_requests() {
    let (_server, client) = start_smollm_tool_spans(false).await;

    let resp = client
        .anthropic_messages_raw(&tool_spans_corpus()[0])
        .await
        .unwrap();

    assert!(resp.status().is_success(), "status: {}", resp.status());
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["type"], "message");
    assert!(!body["content"].as_array().unwrap().is_empty());
}
