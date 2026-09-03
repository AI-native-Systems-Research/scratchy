// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! E2E tests for N-gram speculative decoding.
//!
//! Validates that spec decode:
//! 1. Produces coherent output
//! 2. Produces **identical** output to baseline greedy decoding (correctness invariant)
//! 3. Works with streaming
//! 4. Exposes spec decode metrics via /metrics
//!
//! Run with:
//!   cargo test -p vllm-e2e --features e2e,cuda --release --test e_spec_decode -- --ignored --test-threads=1

#![cfg(feature = "e2e")]

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

fn simple_completion_request(prompt: &str, max_tokens: u32) -> CompletionRequest {
    CompletionRequest {
        prompt: Some(CompletionPrompt::Single(prompt.to_string())),
        max_tokens: Some(max_tokens),
        temperature: Some(0.0),
        ..serde_json::from_str(r#"{}"#).unwrap()
    }
}

/// Start a server with n-gram speculative decoding enabled.
async fn start_spec_decode_server(model: &str) -> TestServer {
    TestServer::builder(model)
        .with_args(&[
            "--speculative-model",
            "ngram",
            "--num-speculative-tokens",
            "5",
            "--ngram-prompt-lookup-max",
            "4",
        ])
        .start()
        .await
        .expect("spec decode server should start")
}

/// Start a baseline server (no spec decode) for comparison.
async fn start_baseline_server(model: &str) -> TestServer {
    TestServer::builder(model)
        .start()
        .await
        .expect("baseline server should start")
}

// ===========================================================================
// CUDA spec decode tests
// ===========================================================================

/// Basic: server starts, generates coherent text with spec decode.
#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_spec_decode_basic_completion() {
    let server = start_spec_decode_server(TestModels::QWEN2_0_5B_CUDA).await;
    let client = Client::new(server.base_url());

    // Use a repetitive prompt to maximize n-gram matches.
    let prompt = "1 2 3 4 5 1 2 3 4 5 1 2 3 4 5 1 2 3";
    let req = simple_completion_request(prompt, 30);
    let resp = client
        .completion(&req)
        .await
        .expect("completion should succeed");

    assert!(!resp.choices.is_empty(), "should have at least one choice");
    let text = &resp.choices[0].text;
    assert!(!text.is_empty(), "generated text should not be empty");
    eprintln!("[spec_decode] basic completion output: {text:?}");
}

/// Basic: chat completion with spec decode.
#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_spec_decode_chat() {
    let server = start_spec_decode_server(TestModels::QWEN2_0_5B_CUDA).await;
    let client = Client::new(server.base_url());

    let req = simple_chat_request("Count from 1 to 20.", Some(50));
    let resp = client
        .chat_completion(&req)
        .await
        .expect("chat completion should succeed");

    assert!(!resp.choices.is_empty());
    let text = resp.choices[0].message.content.as_deref().unwrap_or("");
    assert!(!text.is_empty(), "chat response should not be empty");
    eprintln!("[spec_decode] chat output: {text:?}");
}

/// Correctness invariant: greedy spec decode MUST produce identical output
/// to greedy baseline decoding (temperature=0, no randomness).
///
/// This is the most important spec decode test. If this fails, the rejection
/// sampling logic is broken.
#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_spec_decode_matches_baseline() {
    // Use multiple prompts to increase coverage.
    let prompts = [
        "The capital of France is",
        "1 2 3 4 5 6 7 8 9 10",
        "Once upon a time, there was a",
        "The quick brown fox jumps over the",
    ];
    let max_tokens = 30;

    // Start baseline server (no spec decode).
    let baseline_server = start_baseline_server(TestModels::QWEN2_0_5B_CUDA).await;
    let baseline_client = Client::new(baseline_server.base_url());

    // Collect baseline outputs.
    let mut baseline_outputs = Vec::new();
    for prompt in &prompts {
        let req = simple_completion_request(prompt, max_tokens);
        let resp = baseline_client
            .completion(&req)
            .await
            .expect("baseline completion should succeed");
        assert!(!resp.choices.is_empty());
        baseline_outputs.push(resp.choices[0].text.clone());
    }

    // Stop baseline server to free GPU memory.
    drop(baseline_server);

    // Start spec decode server.
    let spec_server = start_spec_decode_server(TestModels::QWEN2_0_5B_CUDA).await;
    let spec_client = Client::new(spec_server.base_url());

    // Compare outputs.
    for (i, prompt) in prompts.iter().enumerate() {
        let req = simple_completion_request(prompt, max_tokens);
        let resp = spec_client
            .completion(&req)
            .await
            .expect("spec decode completion should succeed");
        assert!(!resp.choices.is_empty());
        let spec_output = &resp.choices[0].text;

        assert_eq!(
            spec_output, &baseline_outputs[i],
            "Spec decode output MUST match baseline for prompt {i}: {prompt:?}\n\
             Baseline: {:?}\n\
             Spec:     {:?}",
            baseline_outputs[i], spec_output
        );
    }
    eprintln!(
        "[spec_decode] All {} prompts match baseline!",
        prompts.len()
    );
}

/// Streaming: verify SSE events arrive and form coherent text.
#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_spec_decode_streaming() {
    let server = start_spec_decode_server(TestModels::QWEN2_0_5B_CUDA).await;
    let client = Client::new(server.base_url());

    let request = ChatCompletionRequest {
        messages: vec![user_msg("Count from 1 to 10.")],
        max_tokens: Some(40),
        temperature: Some(0.0),
        stream: true,
        ..default_chat_request()
    };

    let chunks = client
        .chat_completion_stream(&request)
        .await
        .expect("streaming should succeed");

    assert!(
        chunks.len() >= 2,
        "should receive at least 2 SSE chunks, got {}",
        chunks.len()
    );

    // Reconstruct the full text from deltas.
    let mut full_text = String::new();
    for chunk in &chunks {
        for choice in &chunk.choices {
            if let Some(ref content) = choice.delta.content {
                full_text.push_str(content);
            }
        }
    }

    assert!(
        !full_text.is_empty(),
        "reconstructed streaming text should not be empty"
    );
    eprintln!(
        "[spec_decode] streaming: {} chunks, text: {:?}",
        chunks.len(),
        full_text
    );
}

/// Metrics: verify spec decode Prometheus counters are present and populated.
#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_spec_decode_metrics() {
    let server = TestServer::builder(TestModels::QWEN2_0_5B_CUDA)
        .with_args(&[
            "--speculative-model",
            "ngram",
            "--num-speculative-tokens",
            "5",
            "--ngram-prompt-lookup-max",
            "4",
            "--enable-metrics",
        ])
        .start()
        .await
        .expect("spec decode server with metrics should start");

    let client = Client::new(server.base_url());

    // Send a few requests to generate spec decode activity.
    for _ in 0..3 {
        let req = simple_completion_request("1 2 3 4 5 1 2 3 4 5 1 2 3 4 5 1 2 3", 20);
        let _ = client.completion(&req).await;
    }

    // Check /metrics endpoint.
    let metrics_text = client
        .metrics()
        .await
        .expect("metrics endpoint should work");

    // Verify spec decode counters exist.
    assert!(
        metrics_text.contains("vllm_spec_decode_num_drafts"),
        "metrics should contain spec_decode_num_drafts counter"
    );
    assert!(
        metrics_text.contains("vllm_spec_decode_num_draft_tokens"),
        "metrics should contain spec_decode_num_draft_tokens counter"
    );
    assert!(
        metrics_text.contains("vllm_spec_decode_num_accepted_tokens"),
        "metrics should contain spec_decode_num_accepted_tokens counter"
    );

    eprintln!(
        "[spec_decode] metrics sample:\n{}",
        &metrics_text[..metrics_text.len().min(2000)]
    );
}

/// Multiple concurrent requests with spec decode.
#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_spec_decode_concurrent() {
    let server = start_spec_decode_server(TestModels::QWEN2_0_5B_CUDA).await;

    // Send 4 requests concurrently.
    let prompts = [
        "The meaning of life is",
        "1 1 1 1 1 1 1 1",
        "Hello world! Hello world!",
        "A B C D E F G H",
    ];

    let mut handles = Vec::new();
    for prompt in &prompts {
        let client = Client::new(server.base_url());
        let prompt = prompt.to_string();
        handles.push(tokio::spawn(async move {
            let req = simple_completion_request(&prompt, 20);
            client.completion(&req).await
        }));
    }

    let mut results = Vec::new();
    for handle in handles {
        let result = handle.await.expect("task should not panic");
        let resp = result.expect("completion should succeed");
        assert!(!resp.choices.is_empty());
        results.push(resp.choices[0].text.clone());
    }

    // All should produce non-empty output.
    for (i, text) in results.iter().enumerate() {
        assert!(
            !text.is_empty(),
            "concurrent request {i} should produce output"
        );
    }
    eprintln!(
        "[spec_decode] concurrent: all {} requests produced output",
        results.len()
    );
}
