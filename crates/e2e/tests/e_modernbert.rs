// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! ModernBERT encoder E2E tests.
//!
//! Validates that ModernBERT (encoder-only, bidirectional attention) works
//! end-to-end with the pooling runner for embedding generation.
//!
//! Run with: `cargo test -p vllm-e2e --features e2e,metal --test e_modernbert -- --ignored --test-threads=1`

#![cfg(feature = "e2e")]

use scratchy_e2e::{Client, TestModels, TestServer};
use scratchy_serving_api::protocol::{EmbeddingInput, EmbeddingRequest};

fn embed_request(input: EmbeddingInput) -> EmbeddingRequest {
    EmbeddingRequest {
        input,
        model: None,
        encoding_format: None,
        dimensions: None,
        user: None,
    }
}

/// Extract a flat `Vec<f32>` from the embedding JSON value (single-vector case).
fn embedding_as_vec(value: &serde_json::Value) -> Vec<f32> {
    value
        .as_array()
        .expect("embedding should be a JSON array")
        .iter()
        .map(|v| v.as_f64().expect("embedding element should be a number") as f32)
        .collect()
}

/// Get the length of the embedding JSON array.
fn embedding_len(value: &serde_json::Value) -> usize {
    value
        .as_array()
        .expect("embedding should be a JSON array")
        .len()
}

/// Check if the embedding JSON array is non-empty.
fn embedding_is_nonempty(value: &serde_json::Value) -> bool {
    value.as_array().map(|a| !a.is_empty()).unwrap_or(false)
}

// ===========================================================================
// ModernBERT-base — CLS pooling (default for ModernBERT)
// ===========================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_modernbert_cls_embedding() {
    let server = TestServer::builder(TestModels::MODERNBERT_BASE)
        .with_runner("pooling")
        .with_pooling_strategy("cls")
        .start()
        .await
        .expect("ModernBERT pooling server should start");

    let client = Client::new(server.base_url());

    let request = embed_request(EmbeddingInput::Single("Hello world".to_string()));
    let response = client.embedding(&request).await.unwrap();

    assert_eq!(response.object, "list");
    assert_eq!(response.data.len(), 1);
    assert_eq!(response.data[0].index, 0);
    assert!(embedding_is_nonempty(&response.data[0].embedding));
    // ModernBERT-base hidden_size = 768
    assert_eq!(
        embedding_len(&response.data[0].embedding),
        768,
        "ModernBERT-base should produce 768-dim embeddings"
    );

    // Should be L2-normalized.
    let emb = embedding_as_vec(&response.data[0].embedding);
    let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        (norm - 1.0).abs() < 0.01,
        "L2 norm should be ~1.0, got {norm}"
    );
}

// ===========================================================================
// ModernBERT-base — mean pooling
// ===========================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_modernbert_mean_embedding() {
    let server = TestServer::builder(TestModels::MODERNBERT_BASE)
        .with_runner("pooling")
        .with_pooling_strategy("mean")
        .start()
        .await
        .expect("ModernBERT pooling server should start");

    let client = Client::new(server.base_url());

    let request = embed_request(EmbeddingInput::Single(
        "The quick brown fox jumps over the lazy dog".to_string(),
    ));
    let response = client.embedding(&request).await.unwrap();

    assert_eq!(response.data.len(), 1);
    assert_eq!(embedding_len(&response.data[0].embedding), 768);

    let emb = embedding_as_vec(&response.data[0].embedding);
    let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        (norm - 1.0).abs() < 0.01,
        "L2 norm should be ~1.0, got {norm}"
    );
}

// ===========================================================================
// ModernBERT-base — multiple inputs
// ===========================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_modernbert_multiple_embeddings() {
    let server = TestServer::builder(TestModels::MODERNBERT_BASE)
        .with_runner("pooling")
        .with_pooling_strategy("cls")
        .start()
        .await
        .expect("ModernBERT pooling server should start");

    let client = Client::new(server.base_url());

    let request = embed_request(EmbeddingInput::Multiple(vec![
        scratchy_serving_api::protocol::EmbeddingInputItem::Text("Hello".to_string()),
        scratchy_serving_api::protocol::EmbeddingInputItem::Text("World".to_string()),
        scratchy_serving_api::protocol::EmbeddingInputItem::Text("Foo bar".to_string()),
    ]));
    let response = client.embedding(&request).await.unwrap();

    assert_eq!(response.data.len(), 3);
    for (i, d) in response.data.iter().enumerate() {
        assert_eq!(d.index, i);
        assert_eq!(embedding_len(&d.embedding), 768);
    }
}

// ===========================================================================
// ModernBERT-base — different inputs produce different embeddings
// ===========================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_modernbert_different_inputs_differ() {
    let server = TestServer::builder(TestModels::MODERNBERT_BASE)
        .with_runner("pooling")
        .with_pooling_strategy("cls")
        .start()
        .await
        .expect("ModernBERT pooling server should start");

    let client = Client::new(server.base_url());

    let resp1 = client
        .embedding(&embed_request(EmbeddingInput::Single(
            "The sky is blue".to_string(),
        )))
        .await
        .unwrap();
    let resp2 = client
        .embedding(&embed_request(EmbeddingInput::Single(
            "Quantum computing is complex".to_string(),
        )))
        .await
        .unwrap();

    let emb1 = embedding_as_vec(&resp1.data[0].embedding);
    let emb2 = embedding_as_vec(&resp2.data[0].embedding);
    let cosine: f32 = emb1.iter().zip(emb2.iter()).map(|(a, b)| a * b).sum();
    assert!(
        cosine < 0.999,
        "Different inputs should have cosine similarity < 1.0, got {cosine}"
    );
}

// ===========================================================================
// ModernBERT-base — AllTokens (ColBERT-style multi-vector)
// ===========================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_modernbert_all_tokens_embedding() {
    let server = TestServer::builder(TestModels::MODERNBERT_BASE)
        .with_runner("pooling")
        .with_pooling_strategy("all")
        .start()
        .await
        .expect("ModernBERT pooling server should start");

    let client = Client::new(server.base_url());

    let request = embed_request(EmbeddingInput::Single("Hello world".to_string()));
    let response = client.embedding(&request).await.unwrap();

    assert_eq!(response.data.len(), 1);
    // AllTokens returns 2D embedding — each token gets its own vector.
    // The embedding field is a JSON array of arrays.
    assert!(embedding_is_nonempty(&response.data[0].embedding));
}

// ===========================================================================
// ModernBERT-base — rejects generation endpoints
// ===========================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_modernbert_rejects_chat_completions() {
    let server = TestServer::builder(TestModels::MODERNBERT_BASE)
        .with_runner("pooling")
        .with_pooling_strategy("cls")
        .start()
        .await
        .expect("ModernBERT pooling server should start");

    let client = Client::new(server.base_url());

    let body = serde_json::json!({
        "messages": [{"role": "user", "content": "Hello"}]
    });
    let resp = client.chat_completion_raw(&body).await.unwrap();
    assert_eq!(
        resp.status().as_u16(),
        400,
        "Chat completions should return 400 in pooling mode, got {}",
        resp.status()
    );
}

// ===========================================================================
// CUDA backend — ModernBERT
// ===========================================================================

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_modernbert_cls_embedding() {
    let server = TestServer::builder(TestModels::MODERNBERT_BASE)
        .with_runner("pooling")
        .with_pooling_strategy("cls")
        .start()
        .await
        .expect("ModernBERT CUDA pooling server should start");

    let client = Client::new(server.base_url());

    let request = embed_request(EmbeddingInput::Single("Hello world".to_string()));
    let response = client.embedding(&request).await.unwrap();

    assert_eq!(response.data.len(), 1);
    assert_eq!(embedding_len(&response.data[0].embedding), 768);

    let emb = embedding_as_vec(&response.data[0].embedding);
    let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        (norm - 1.0).abs() < 0.01,
        "L2 norm should be ~1.0, got {norm}"
    );
}

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_modernbert_mean_embedding() {
    let server = TestServer::builder(TestModels::MODERNBERT_BASE)
        .with_runner("pooling")
        .with_pooling_strategy("mean")
        .start()
        .await
        .expect("ModernBERT CUDA pooling server should start");

    let client = Client::new(server.base_url());

    let request = embed_request(EmbeddingInput::Single(
        "The quick brown fox jumps over the lazy dog".to_string(),
    ));
    let response = client.embedding(&request).await.unwrap();

    assert_eq!(response.data.len(), 1);
    assert_eq!(embedding_len(&response.data[0].embedding), 768);
}

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_modernbert_multiple_embeddings() {
    let server = TestServer::builder(TestModels::MODERNBERT_BASE)
        .with_runner("pooling")
        .with_pooling_strategy("cls")
        .start()
        .await
        .expect("ModernBERT CUDA pooling server should start");

    let client = Client::new(server.base_url());

    let request = embed_request(EmbeddingInput::Multiple(vec![
        scratchy_serving_api::protocol::EmbeddingInputItem::Text("Hello".to_string()),
        scratchy_serving_api::protocol::EmbeddingInputItem::Text("World".to_string()),
    ]));
    let response = client.embedding(&request).await.unwrap();

    assert_eq!(response.data.len(), 2);
    assert_eq!(embedding_len(&response.data[0].embedding), 768);
    assert_eq!(embedding_len(&response.data[1].embedding), 768);
}

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_modernbert_all_tokens_embedding() {
    let server = TestServer::builder(TestModels::MODERNBERT_BASE)
        .with_runner("pooling")
        .with_pooling_strategy("all")
        .start()
        .await
        .expect("ModernBERT CUDA pooling server should start");

    let client = Client::new(server.base_url());

    let request = embed_request(EmbeddingInput::Single("Hello world".to_string()));
    let response = client.embedding(&request).await.unwrap();

    assert_eq!(response.data.len(), 1);
    assert!(embedding_is_nonempty(&response.data[0].embedding));
}
