// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Embedding endpoint E2E tests.
//!
//! Validates that the /v1/embeddings endpoint works with decoder models
//! using last-token, mean, and CLS pooling strategies with L2 normalization.
//!
//! Run with: `cargo test -p vllm-e2e --features e2e,metal --test e_embedding -- --ignored --test-threads=1`

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

// ===========================================================================
// SmolLM-135M-4bit — Tier 1 embedding tests
// ===========================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_embedding_single_string() {
    let server = TestServer::builder(TestModels::SMOLLM)
        .start()
        .await
        .expect("server should start");

    let client = Client::new(server.base_url());

    let request = embed_request(EmbeddingInput::Single("Hello world".to_string()));
    let response = client.embedding(&request).await.unwrap();

    assert_eq!(response.object, "list");
    assert_eq!(response.data.len(), 1);
    assert_eq!(response.data[0].index, 0);
    assert_eq!(response.data[0].object, "embedding");
    assert!(!response.data[0].embedding.is_empty());
    assert!(response.usage.prompt_tokens > 0);
    assert_eq!(response.usage.total_tokens, response.usage.prompt_tokens);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_embedding_multiple_strings() {
    let server = TestServer::builder(TestModels::SMOLLM)
        .start()
        .await
        .expect("server should start");

    let client = Client::new(server.base_url());

    let request = embed_request(EmbeddingInput::Multiple(vec![
        scratchy_serving_api::protocol::EmbeddingInputItem::Text("Hello".to_string()),
        scratchy_serving_api::protocol::EmbeddingInputItem::Text("World".to_string()),
        scratchy_serving_api::protocol::EmbeddingInputItem::Text("Foo bar".to_string()),
    ]));
    let response = client.embedding(&request).await.unwrap();

    assert_eq!(response.data.len(), 3);
    assert_eq!(response.data[0].index, 0);
    assert_eq!(response.data[1].index, 1);
    assert_eq!(response.data[2].index, 2);
    // All embeddings should have the same dimension.
    let dim = response.data[0].embedding.len();
    assert!(dim > 0);
    assert_eq!(response.data[1].embedding.len(), dim);
    assert_eq!(response.data[2].embedding.len(), dim);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_embedding_dimensions() {
    let server = TestServer::builder(TestModels::SMOLLM)
        .start()
        .await
        .expect("server should start");

    let client = Client::new(server.base_url());

    let mut request = embed_request(EmbeddingInput::Single("Test input".to_string()));
    request.dimensions = Some(32);
    let response = client.embedding(&request).await.unwrap();

    assert_eq!(response.data.len(), 1);
    assert_eq!(response.data[0].embedding.len(), 32);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_embedding_normalized() {
    let server = TestServer::builder(TestModels::SMOLLM)
        .start()
        .await
        .expect("server should start");

    let client = Client::new(server.base_url());

    let request = embed_request(EmbeddingInput::Single(
        "The quick brown fox jumps over the lazy dog".to_string(),
    ));
    let response = client.embedding(&request).await.unwrap();

    let emb = &response.data[0].embedding;
    let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        (norm - 1.0).abs() < 0.01,
        "L2 norm should be ~1.0, got {norm}"
    );
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_embedding_different_inputs() {
    let server = TestServer::builder(TestModels::SMOLLM)
        .start()
        .await
        .expect("server should start");

    let client = Client::new(server.base_url());

    let req1 = embed_request(EmbeddingInput::Single("The sky is blue".to_string()));
    let req2 = embed_request(EmbeddingInput::Single(
        "Quantum computing is complex".to_string(),
    ));

    let resp1 = client.embedding(&req1).await.unwrap();
    let resp2 = client.embedding(&req2).await.unwrap();

    // Embeddings should be different.
    let emb1 = &resp1.data[0].embedding;
    let emb2 = &resp2.data[0].embedding;
    let cosine: f32 = emb1.iter().zip(emb2.iter()).map(|(a, b)| a * b).sum();
    assert!(
        cosine < 0.999,
        "Different inputs should have cosine similarity < 1.0, got {cosine}"
    );
}

// ===========================================================================
// Pooling strategy tests
// ===========================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_embedding_mean_pooling() {
    let server = TestServer::builder(TestModels::SMOLLM)
        .with_pooling_strategy("mean")
        .start()
        .await
        .expect("server should start");

    let client = Client::new(server.base_url());

    let request = embed_request(EmbeddingInput::Single(
        "The quick brown fox jumps over the lazy dog".to_string(),
    ));
    let response = client.embedding(&request).await.unwrap();

    assert_eq!(response.data.len(), 1);
    let emb = &response.data[0].embedding;
    assert!(!emb.is_empty());

    // Should be L2-normalized regardless of pooling strategy.
    let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        (norm - 1.0).abs() < 0.01,
        "L2 norm should be ~1.0, got {norm}"
    );
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_embedding_cls_pooling() {
    let server = TestServer::builder(TestModels::SMOLLM)
        .with_pooling_strategy("cls")
        .start()
        .await
        .expect("server should start");

    let client = Client::new(server.base_url());

    let request = embed_request(EmbeddingInput::Single(
        "The quick brown fox jumps over the lazy dog".to_string(),
    ));
    let response = client.embedding(&request).await.unwrap();

    assert_eq!(response.data.len(), 1);
    let emb = &response.data[0].embedding;
    assert!(!emb.is_empty());

    let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        (norm - 1.0).abs() < 0.01,
        "L2 norm should be ~1.0, got {norm}"
    );
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_embedding_mean_vs_last_differ() {
    // Mean and last-token pooling should produce different embeddings.
    let server_mean = TestServer::builder(TestModels::SMOLLM)
        .with_pooling_strategy("mean")
        .start()
        .await
        .expect("server should start");

    let server_last = TestServer::builder(TestModels::SMOLLM)
        .with_pooling_strategy("last")
        .start()
        .await
        .expect("server should start");

    let input = "Hello world, this is a test sentence.".to_string();

    let resp_mean = Client::new(server_mean.base_url())
        .embedding(&embed_request(EmbeddingInput::Single(input.clone())))
        .await
        .unwrap();
    let resp_last = Client::new(server_last.base_url())
        .embedding(&embed_request(EmbeddingInput::Single(input)))
        .await
        .unwrap();

    let emb_mean = &resp_mean.data[0].embedding;
    let emb_last = &resp_last.data[0].embedding;

    // Same model, same input, different strategies → different embeddings.
    assert_eq!(emb_mean.len(), emb_last.len());
    let cosine: f32 = emb_mean
        .iter()
        .zip(emb_last.iter())
        .map(|(a, b)| a * b)
        .sum();
    assert!(
        cosine < 0.999,
        "Mean and last pooling should produce different embeddings, cosine={cosine}"
    );
}

// ===========================================================================
// CUDA backend embedding tests
// ===========================================================================

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_embedding_single_string() {
    let server = TestServer::builder(TestModels::SMOLLM_135M_CUDA)
        .start()
        .await
        .expect("server should start");

    let client = Client::new(server.base_url());

    let request = embed_request(EmbeddingInput::Single("Hello world".to_string()));
    let response = client.embedding(&request).await.unwrap();

    assert_eq!(response.object, "list");
    assert_eq!(response.data.len(), 1);
    assert_eq!(response.data[0].index, 0);
    assert!(!response.data[0].embedding.is_empty());
    assert!(response.usage.prompt_tokens > 0);
}

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_embedding_multiple_strings() {
    let server = TestServer::builder(TestModels::SMOLLM_135M_CUDA)
        .start()
        .await
        .expect("server should start");

    let client = Client::new(server.base_url());

    let request = embed_request(EmbeddingInput::Multiple(vec![
        scratchy_serving_api::protocol::EmbeddingInputItem::Text("Hello".to_string()),
        scratchy_serving_api::protocol::EmbeddingInputItem::Text("World".to_string()),
    ]));
    let response = client.embedding(&request).await.unwrap();

    assert_eq!(response.data.len(), 2);
    let dim = response.data[0].embedding.len();
    assert!(dim > 0);
    assert_eq!(response.data[1].embedding.len(), dim);
}

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_embedding_normalized() {
    let server = TestServer::builder(TestModels::SMOLLM_135M_CUDA)
        .start()
        .await
        .expect("server should start");

    let client = Client::new(server.base_url());

    let request = embed_request(EmbeddingInput::Single(
        "The quick brown fox jumps over the lazy dog".to_string(),
    ));
    let response = client.embedding(&request).await.unwrap();

    let emb = &response.data[0].embedding;
    let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        (norm - 1.0).abs() < 0.01,
        "L2 norm should be ~1.0, got {norm}"
    );
}

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_embedding_different_inputs() {
    let server = TestServer::builder(TestModels::SMOLLM_135M_CUDA)
        .start()
        .await
        .expect("server should start");

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

    let emb1 = &resp1.data[0].embedding;
    let emb2 = &resp2.data[0].embedding;
    let cosine: f32 = emb1.iter().zip(emb2.iter()).map(|(a, b)| a * b).sum();
    assert!(
        cosine < 0.999,
        "Different inputs should have cosine similarity < 1.0, got {cosine}"
    );
}

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_embedding_dimensions() {
    let server = TestServer::builder(TestModels::SMOLLM_135M_CUDA)
        .start()
        .await
        .expect("server should start");

    let client = Client::new(server.base_url());

    let mut request = embed_request(EmbeddingInput::Single("Test input".to_string()));
    request.dimensions = Some(32);
    let response = client.embedding(&request).await.unwrap();

    assert_eq!(response.data[0].embedding.len(), 32);
}

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_embedding_mean_pooling() {
    let server = TestServer::builder(TestModels::SMOLLM_135M_CUDA)
        .with_pooling_strategy("mean")
        .start()
        .await
        .expect("server should start");

    let client = Client::new(server.base_url());

    let request = embed_request(EmbeddingInput::Single(
        "The quick brown fox jumps over the lazy dog".to_string(),
    ));
    let response = client.embedding(&request).await.unwrap();

    assert_eq!(response.data.len(), 1);
    let emb = &response.data[0].embedding;
    assert!(!emb.is_empty());
    let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        (norm - 1.0).abs() < 0.01,
        "L2 norm should be ~1.0, got {norm}"
    );
}

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_embedding_cls_pooling() {
    let server = TestServer::builder(TestModels::SMOLLM_135M_CUDA)
        .with_pooling_strategy("cls")
        .start()
        .await
        .expect("server should start");

    let client = Client::new(server.base_url());

    let request = embed_request(EmbeddingInput::Single(
        "The quick brown fox jumps over the lazy dog".to_string(),
    ));
    let response = client.embedding(&request).await.unwrap();

    assert_eq!(response.data.len(), 1);
    let emb = &response.data[0].embedding;
    assert!(!emb.is_empty());
    let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        (norm - 1.0).abs() < 0.01,
        "L2 norm should be ~1.0, got {norm}"
    );
}

// ===========================================================================
// Additional model coverage
// ===========================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_embedding_qwen2() {
    let server = TestServer::builder(TestModels::QWEN2)
        .start()
        .await
        .expect("server should start");

    let client = Client::new(server.base_url());
    let request = embed_request(EmbeddingInput::Single("Test embedding".to_string()));
    let response = client.embedding(&request).await.unwrap();

    assert_eq!(response.data.len(), 1);
    assert!(!response.data[0].embedding.is_empty());

    let norm: f32 = response.data[0]
        .embedding
        .iter()
        .map(|x| x * x)
        .sum::<f32>()
        .sqrt();
    assert!(
        (norm - 1.0).abs() < 0.01,
        "L2 norm should be ~1.0, got {norm}"
    );
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_embedding_llama3() {
    let server = TestServer::builder(TestModels::LLAMA_3_2)
        .start()
        .await
        .expect("server should start");

    let client = Client::new(server.base_url());
    let request = embed_request(EmbeddingInput::Single("Test embedding".to_string()));
    let response = client.embedding(&request).await.unwrap();

    assert_eq!(response.data.len(), 1);
    assert!(!response.data[0].embedding.is_empty());

    let norm: f32 = response.data[0]
        .embedding
        .iter()
        .map(|x| x * x)
        .sum::<f32>()
        .sqrt();
    assert!(
        (norm - 1.0).abs() < 0.01,
        "L2 norm should be ~1.0, got {norm}"
    );
}

// ===========================================================================
// CUDA — mean vs last differ (confirms strategies are actually different)
// ===========================================================================

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_embedding_mean_vs_last_differ() {
    let input = "Hello world, this is a test sentence.".to_string();

    // Run mean server first, get embedding, then drop it to free GPU memory.
    let emb_mean = {
        let server = TestServer::builder(TestModels::SMOLLM_135M_CUDA)
            .with_pooling_strategy("mean")
            .start()
            .await
            .expect("server should start");
        let resp = Client::new(server.base_url())
            .embedding(&embed_request(EmbeddingInput::Single(input.clone())))
            .await
            .unwrap();
        resp.data[0].embedding.clone()
    };

    // Now run last server (previous one freed GPU memory on drop).
    let emb_last = {
        let server = TestServer::builder(TestModels::SMOLLM_135M_CUDA)
            .with_pooling_strategy("last")
            .start()
            .await
            .expect("server should start");
        let resp = Client::new(server.base_url())
            .embedding(&embed_request(EmbeddingInput::Single(input)))
            .await
            .unwrap();
        resp.data[0].embedding.clone()
    };

    assert_eq!(emb_mean.len(), emb_last.len());
    let cosine: f32 = emb_mean
        .iter()
        .zip(emb_last.iter())
        .map(|(a, b)| a * b)
        .sum();
    assert!(
        cosine < 0.999,
        "Mean and last pooling should produce different embeddings, cosine={cosine}"
    );
}
