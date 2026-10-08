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

// ---------------------------------------------------------------------------
// Cache-hit REPORTING (not just correctness)
//
// The tests above assert that a cached prefix produces the same text. They say
// nothing about whether the engine tells anyone how much it reused — and until
// the `engine_core.rs` fix it could not: `EngineCoreOutput.num_cached_tokens`
// was hardcoded `0` at every production site, so `prompt_tokens_details` was
// always `None` on every backend and the scheduler's correctly-computed hit was
// dropped on the floor.
//
// Epic #158 measures `reprefill_ratio` on an agent workload, so the reported
// number is the deliverable, not a diagnostic. This is the regression guard.
// ---------------------------------------------------------------------------

/// A chat request whose prompt is comfortably longer than one KV block.
///
/// The reportable hit is floored to the block size (16 by default), so a prompt
/// of <= 16 rendered tokens can never report a hit even when fully cached. This
/// prompt is a few hundred tokens, well clear of that edge. The text has to
/// carry the length itself: SmolLM's chat template does not reference `tools`,
/// so tool schemas on the request would render to nothing at all.
fn long_chat_request() -> scratchy_serving_api::protocol::ChatCompletionRequest {
    use scratchy_serving_api::protocol::{ChatCompletionMessageParam, ChatCompletionRequest};

    let mut prompt = String::new();
    for i in 0..40 {
        prompt.push_str(&format!(
            "Paragraph {i}: the scheduler hashes full blocks of prompt tokens and \
             reuses any contiguous run it has already computed. "
        ));
    }

    ChatCompletionRequest {
        messages: vec![ChatCompletionMessageParam {
            role: "user".to_string(),
            content: Some(serde_json::Value::String(prompt)),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }],
        max_tokens: Some(4),
        temperature: Some(0.0),
        // `ChatCompletionRequest.messages` has no serde default, so seed the
        // rest of the struct from a body that supplies it (same idiom as
        // `default_chat_request` in e1_basic_serving.rs).
        ..serde_json::from_str(r#"{"messages": []}"#).unwrap()
    }
}

/// Send the same long prompt twice and assert the SECOND request reports a
/// prefix-cache hit through `usage.prompt_tokens_details.cached_tokens`.
async fn assert_reports_cached_tokens(client: &Client, device: &str) {
    let request = long_chat_request();

    // A fresh TestServer means a fresh block pool, so the first request is the
    // cold arm by construction — no cache reset needed.
    let cold = client.chat_completion(&request).await.unwrap();
    let cold_cached = cold
        .usage
        .prompt_tokens_details
        .as_ref()
        .and_then(|d| d.cached_tokens)
        .unwrap_or(0);
    assert_eq!(
        cold_cached, 0,
        "first request to a fresh server must report no cache hit, got {cold_cached}"
    );

    let warm = client.chat_completion(&request).await.unwrap();
    let warm_cached = warm
        .usage
        .prompt_tokens_details
        .as_ref()
        .and_then(|d| d.cached_tokens)
        .unwrap_or(0);

    assert!(
        warm_cached > 0,
        "repeat of an identical {}-token prompt must report a cache hit, got 0 \
         (the exact failure mode the engine_core fix addresses: the scheduler \
         computes the hit and the output used to drop it)",
        cold.usage.prompt_tokens
    );
    assert!(
        warm_cached <= warm.usage.prompt_tokens,
        "cached {warm_cached} cannot exceed prompt {}",
        warm.usage.prompt_tokens
    );
    assert_eq!(
        cold.usage.prompt_tokens, warm.usage.prompt_tokens,
        "identical bodies must tokenize to the same prompt length"
    );
    eprintln!(
        "[prefix-cache/{device}] prompt={} cold_cached={} warm_cached={} reuse={:.1}%",
        warm.usage.prompt_tokens,
        cold_cached,
        warm_cached,
        100.0 * warm_cached as f64 / warm.usage.prompt_tokens as f64,
    );
}

/// Metal arm. There is no CPU arm: `device = "cpu"` has no backend in any
/// build ("No backend available for device 'cpu'"), which is also why this
/// file's older `start_cpu_server` tests cannot run.
///
/// On the hosted `macos-26` CI runner this SKIPS — paravirtual GPU, no Metal 4.
/// What CI gets from the matching workflow step is compile coverage of this
/// file; the assertions execute on a real GPU locally and via
/// `scripts/act-local.sh`.
///
/// Uses the DENSE checkpoint, not `TestModels::SMOLLM`: that constant is MLX
/// 4-bit on metal and will not load without a `quant/<preset>` feature. This
/// stem matches `model/smollm2-135m`, the feature set the metal CI job already
/// builds.
#[cfg(target_os = "macos")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_prefix_cache_reports_cached_tokens_metal() {
    scratchy_e2e::skip_if_no_gpu!();
    let server = TestServer::builder("HuggingFaceTB/SmolLM2-135M-Instruct")
        .with_device("metal")
        .start()
        .await
        .expect("server should start");
    let client = Client::new(server.base_url());
    assert_reports_cached_tokens(&client, "metal").await;
}
