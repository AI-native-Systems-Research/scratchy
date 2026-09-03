// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! E2E tests for prefix caching (KV cache reuse).
//!
//! Verifies that sending the same prompt twice produces correct, identical
//! output — exercising the full path from HTTP → scheduler (with cached
//! prefix lookup) → worker (token slicing) → model → response.
//!
//! MLX tests use `device=metal` with the worker-level prefix cache pool.
//! CUDA tests use `device=cuda` with paged KV cache.
//!
//! Run with: `cargo test -p vllm-e2e --features e2e --test e_prefix_caching -- --ignored`

#![cfg(feature = "e2e")]

use scratchy_e2e::assertions::assert_valid_completion_response;
use scratchy_e2e::{Client, TestServer};
use scratchy_serving_api::protocol::{CompletionPrompt, CompletionRequest};

fn greedy_completion(prompt: &str, max_tokens: u32) -> CompletionRequest {
    CompletionRequest {
        prompt: Some(CompletionPrompt::Single(prompt.to_string())),
        max_tokens: Some(max_tokens),
        temperature: Some(0.0),
        ..serde_json::from_str(r#"{}"#).unwrap()
    }
}

async fn start_cpu_server() -> (TestServer, Client) {
    let server = TestServer::builder("HuggingFaceTB/SmolLM2-135M-Instruct")
        .with_device("cpu")
        .with_dtype("f32")
        .start()
        .await
        .expect("server should start");
    let client = Client::new(server.base_url());
    (server, client)
}

/// Send the same prompt twice — both should produce identical, valid output.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_prefix_caching_same_prompt_twice() {
    let (_server, client) = start_cpu_server().await;

    // Prompt long enough that at least 1 full block (16 tokens) is cached.
    let prompt = "The history of artificial intelligence began in the 1950s when researchers first explored the concept of machines that could think and reason about";
    let request = greedy_completion(prompt, 5);

    let resp1 = client.completion(&request).await.unwrap();
    assert_valid_completion_response(&resp1);
    let text1 = &resp1.choices[0].text;
    assert!(!text1.is_empty(), "first response should have text");

    let resp2 = client.completion(&request).await.unwrap();
    assert_valid_completion_response(&resp2);
    let text2 = &resp2.choices[0].text;

    assert_eq!(
        text1, text2,
        "greedy completions with same prompt should match:\n  first:  {text1:?}\n  second: {text2:?}"
    );
}

/// Three repeats of the same prompt — no degradation over time.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_prefix_caching_many_repeats() {
    let (_server, client) = start_cpu_server().await;

    let prompt = "The capital of France is";
    let request = greedy_completion(prompt, 5);

    let mut first_text = String::new();
    for i in 0..3 {
        let resp = client.completion(&request).await.unwrap();
        assert_valid_completion_response(&resp);
        let text = &resp.choices[0].text;
        assert!(!text.is_empty(), "response {i} should have text");

        if i == 0 {
            first_text = text.clone();
        } else {
            assert_eq!(text, &first_text, "greedy response {i} should match first");
        }
    }
}

// ---------------------------------------------------------------------------
// MLX (Metal) prefix caching
// ---------------------------------------------------------------------------

#[cfg(target_os = "macos")]
async fn start_metal_server() -> (TestServer, Client) {
    let server = TestServer::builder(scratchy_e2e::TestModels::SMOLLM)
        .with_device("metal")
        .start()
        .await
        .expect("server should start");
    let client = Client::new(server.base_url());
    (server, client)
}

/// Same-prompt-twice on MLX — exercises the worker-level KV cache pool.
#[cfg(target_os = "macos")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_prefix_caching_mlx_same_prompt_twice() {
    let (_server, client) = start_metal_server().await;

    let prompt = "The history of artificial intelligence began in the 1950s when researchers first explored the concept of machines that could think and reason about";
    let request = greedy_completion(prompt, 5);

    let resp1 = client.completion(&request).await.unwrap();
    assert_valid_completion_response(&resp1);
    let text1 = &resp1.choices[0].text;
    assert!(!text1.is_empty(), "first response should have text");

    let resp2 = client.completion(&request).await.unwrap();
    assert_valid_completion_response(&resp2);
    let text2 = &resp2.choices[0].text;

    assert_eq!(
        text1, text2,
        "greedy MLX completions with same prompt should match:\n  first:  {text1:?}\n  second: {text2:?}"
    );
}
