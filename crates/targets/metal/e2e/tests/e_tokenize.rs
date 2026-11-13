// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! E2E tests for POST /tokenize and POST /detokenize endpoints.
//!
//! Run with: `cargo test -p vllm-e2e --features e2e --test e_tokenize -- --ignored`

#![cfg(feature = "e2e")]

use scratchy_e2e::{Client, TestModels, TestServer};
use scratchy_serving_api::protocol::{DetokenizeRequest, TokenizeRequest};

// ---------------------------------------------------------------------------
// POST /tokenize — prompt mode
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_tokenize_prompt() {
    let server = TestServer::builder(TestModels::SMOLLM)
        .start()
        .await
        .expect("server should start");

    let client = Client::new(server.base_url());
    let request = TokenizeRequest {
        prompt: Some("Hello, world!".to_string()),
        ..default_tokenize_request()
    };
    let resp = client.tokenize(&request).await.unwrap();

    assert!(resp.count > 0, "should produce at least one token");
    assert_eq!(resp.count, resp.tokens.len());
    assert!(resp.max_model_len > 0, "max_model_len should be positive");
    assert!(
        resp.token_strs.is_none(),
        "token_strs should be None by default"
    );
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_tokenize_with_token_strs() {
    let server = TestServer::builder(TestModels::SMOLLM)
        .start()
        .await
        .expect("server should start");

    let client = Client::new(server.base_url());
    let request = TokenizeRequest {
        prompt: Some("Hello".to_string()),
        return_token_strs: Some(true),
        ..default_tokenize_request()
    };
    let resp = client.tokenize(&request).await.unwrap();

    assert!(resp.count > 0);
    let strs = resp.token_strs.expect("token_strs should be present");
    assert_eq!(
        strs.len(),
        resp.count,
        "token_strs length should match count"
    );
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_tokenize_empty_prompt() {
    let server = TestServer::builder(TestModels::SMOLLM)
        .start()
        .await
        .expect("server should start");

    let client = Client::new(server.base_url());
    let request = TokenizeRequest {
        prompt: Some(String::new()),
        ..default_tokenize_request()
    };
    let resp = client.tokenize(&request).await.unwrap();

    // Empty prompt may still produce BOS token depending on add_special_tokens.
    assert_eq!(resp.count, resp.tokens.len());
}

// ---------------------------------------------------------------------------
// POST /detokenize
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_detokenize() {
    let server = TestServer::builder(TestModels::SMOLLM)
        .start()
        .await
        .expect("server should start");

    let client = Client::new(server.base_url());

    // First tokenize, then detokenize and verify roundtrip.
    let tok_req = TokenizeRequest {
        prompt: Some("The quick brown fox".to_string()),
        add_special_tokens: Some(false),
        ..default_tokenize_request()
    };
    let tok_resp = client.tokenize(&tok_req).await.unwrap();
    assert!(tok_resp.count > 0);

    let detok_req = DetokenizeRequest {
        model: None,
        tokens: tok_resp.tokens,
    };
    let detok_resp = client.detokenize(&detok_req).await.unwrap();

    assert_eq!(
        detok_resp.prompt, "The quick brown fox",
        "detokenize should roundtrip back to original text"
    );
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_detokenize_empty() {
    let server = TestServer::builder(TestModels::SMOLLM)
        .start()
        .await
        .expect("server should start");

    let client = Client::new(server.base_url());
    let request = DetokenizeRequest {
        model: None,
        tokens: vec![],
    };
    let resp = client.detokenize(&request).await.unwrap();
    assert!(
        resp.prompt.is_empty(),
        "empty tokens should produce empty text"
    );
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn default_tokenize_request() -> TokenizeRequest {
    serde_json::from_str(r#"{}"#).unwrap()
}
