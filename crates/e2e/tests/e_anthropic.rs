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
    assert!(!body["content"][0]["text"].as_str().unwrap().is_empty());
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
// Usage: prompt + cached token accounting (issue #298, epic #158)
//
// `/v1/messages` is the only endpoint Claude Code uses, and it is the endpoint
// the benchmark measures. Two things used to be unreportable from it:
//
//   - the prefix-cache hit, dropped in the Anthropic conversion; and
//   - on a STREAMING request, the prompt length at all — `message_start`
//     emitted a literal `"input_tokens": 0`.
//
// Anthropic's input fields are disjoint (`input_tokens` is the UNCACHED
// remainder, so `input_tokens + cache_read_input_tokens` is the whole prompt),
// unlike OpenAI's, where `cached_tokens` is a subset of `prompt_tokens`. These
// tests pin that invariant on both the buffered and streaming paths.
// ===========================================================================

/// A `/v1/messages` body whose prompt comfortably exceeds one KV block.
///
/// The reportable hit is floored to the block size (16), so a short body can
/// report 0 even on a real hit. Length comes from the text: SmolLM's chat
/// template never references `tools`, so tool schemas would add nothing.
#[cfg(target_os = "macos")]
fn long_messages_body(stream: bool) -> serde_json::Value {
    let mut prompt = String::new();
    for i in 0..40 {
        prompt.push_str(&format!(
            "Paragraph {i}: the scheduler hashes full blocks of prompt tokens and \
             reuses any contiguous run it has already computed. "
        ));
    }
    json!({
        "max_tokens": 4,
        "temperature": 0.0,
        "stream": stream,
        "messages": [{"role": "user", "content": prompt}]
    })
}

/// Dense SmolLM2 on metal. `TestModels::SMOLLM` is MLX 4-bit there and will not
/// load without a `quant/<preset>` feature; this stem matches
/// `model/smollm2-135m`, which the metal CI job already builds.
#[cfg(target_os = "macos")]
async fn start_metal_smollm_dense() -> (TestServer, Client) {
    let server = TestServer::builder("HuggingFaceTB/SmolLM2-135M-Instruct")
        .with_device("metal")
        .start()
        .await
        .expect("server should start");
    let client = Client::new(server.base_url());
    (server, client)
}

/// Non-streaming: the repeat must report the hit, and the input side must sum
/// to the same prompt length on both requests.
#[cfg(target_os = "macos")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_anthropic_usage_reports_cached_tokens_on_repeat() {
    scratchy_e2e::skip_if_no_gpu!();
    let (_server, client) = start_metal_smollm_dense().await;
    let body = long_messages_body(false);

    let read_usage = |v: &serde_json::Value| -> (u64, u64) {
        (
            v["usage"]["input_tokens"].as_u64().expect("input_tokens"),
            v["usage"]["cache_read_input_tokens"]
                .as_u64()
                .expect("cache_read_input_tokens must be present, even at 0"),
        )
    };

    // Fresh server => empty block pool, so request 1 is the cold arm.
    let cold: serde_json::Value = client
        .anthropic_messages_raw(&body)
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let (cold_in, cold_cached) = read_usage(&cold);
    assert_eq!(cold_cached, 0, "cold request must report no cache read");

    let warm: serde_json::Value = client
        .anthropic_messages_raw(&body)
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let (warm_in, warm_cached) = read_usage(&warm);

    assert!(
        warm_cached > 0,
        "repeat of an identical {cold_in}-token prompt must report a cache read, got 0"
    );
    assert!(
        warm_in < cold_in,
        "a cache hit must shrink input_tokens: cold={cold_in} warm={warm_in}"
    );
    // The invariant that makes the number comparable with ollama's.
    assert_eq!(
        cold_in + cold_cached,
        warm_in + warm_cached,
        "input_tokens + cache_read_input_tokens must equal the prompt on both"
    );

    eprintln!(
        "[/v1/messages] cold: input={cold_in} cache_read={cold_cached} | \
         warm: input={warm_in} cache_read={warm_cached} | reprefill={:.3}",
        warm_in as f64 / (warm_in + warm_cached) as f64
    );
}

/// Streaming: the path Claude Code actually uses. `message_start` used to carry
/// a hardcoded `"input_tokens": 0`, which also made #161's no-truncation proof
/// impossible from the endpoint under test.
#[cfg(target_os = "macos")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_anthropic_streaming_reports_prompt_and_cached_tokens() {
    scratchy_e2e::skip_if_no_gpu!();
    let (_server, client) = start_metal_smollm_dense().await;
    let body = long_messages_body(true);

    let cold = client.anthropic_messages_stream(&body).await.unwrap();
    let start = &cold[0];
    assert_eq!(start["type"], "message_start");
    let cold_in = start["message"]["usage"]["input_tokens"]
        .as_u64()
        .expect("input_tokens");
    assert!(
        cold_in > 0,
        "message_start must report the real prompt length, not the 0 it used to hardcode"
    );

    let warm = client.anthropic_messages_stream(&body).await.unwrap();
    let warm_start = &warm[0]["message"]["usage"];
    let warm_in = warm_start["input_tokens"].as_u64().expect("input_tokens");
    let warm_cached = warm_start["cache_read_input_tokens"]
        .as_u64()
        .expect("cache_read_input_tokens");

    assert!(
        warm_cached > 0,
        "streaming repeat must report a cache read in message_start, got 0"
    );
    assert_eq!(
        warm_in + warm_cached,
        cold_in,
        "the streaming input side must sum to the same prompt length"
    );

    // ollama reports the cache field on message_delta, so one client must be
    // able to read it in the same place on both engines.
    let delta = warm
        .iter()
        .find(|e| e["type"] == "message_delta")
        .expect("message_delta");
    assert_eq!(delta["usage"]["cache_read_input_tokens"], warm_cached);
    assert_eq!(delta["usage"]["input_tokens"], warm_in);
    assert!(delta["usage"]["output_tokens"].as_u64().unwrap() > 0);

    eprintln!(
        "[/v1/messages stream] cold_input={cold_in} | warm: input={warm_in} \
         cache_read={warm_cached}"
    );
}
