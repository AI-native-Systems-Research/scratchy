// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Pooling runner E2E tests.
//!
//! Validates that `--runner pooling` routes embedding requests through the
//! scheduler, returns valid embeddings, and rejects generation endpoints.
//!
//! Run with: `cargo test -p vllm-e2e --features e2e,metal --test e_pooling -- --ignored --test-threads=1`

#![cfg(feature = "e2e")]

use scratchy_e2e::assertions::embedding_vector;
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
// Pooling mode — embedding through scheduler
// ===========================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_pooling_server_starts() {
    let server = TestServer::builder(TestModels::SMOLLM)
        .with_runner("pooling")
        .start()
        .await
        .expect("pooling server should start");

    let client = Client::new(server.base_url());

    // Health should work.
    assert!(client.health().await.unwrap());

    // Models should work.
    let models = client.list_models().await.unwrap();
    assert_eq!(models.data.len(), 1);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_pooling_single_embedding() {
    let server = TestServer::builder(TestModels::SMOLLM)
        .with_runner("pooling")
        .start()
        .await
        .expect("pooling server should start");

    let client = Client::new(server.base_url());

    let request = embed_request(EmbeddingInput::Single("Hello world".to_string()));
    let response = client.embedding(&request).await.unwrap();

    assert_eq!(response.object, "list");
    assert_eq!(response.data.len(), 1);
    assert_eq!(response.data[0].index, 0);
    assert_eq!(response.data[0].object, "embedding");
    assert!(!embedding_vector(&response.data[0]).is_empty());
    assert!(response.usage.prompt_tokens > 0);
    assert_eq!(response.usage.total_tokens, response.usage.prompt_tokens);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_pooling_multiple_embeddings() {
    let server = TestServer::builder(TestModels::SMOLLM)
        .with_runner("pooling")
        .start()
        .await
        .expect("pooling server should start");

    let client = Client::new(server.base_url());

    let request = embed_request(EmbeddingInput::Multiple(vec![
        scratchy_serving_api::protocol::EmbeddingInputItem::Text("Hello".to_string()),
        scratchy_serving_api::protocol::EmbeddingInputItem::Text("World".to_string()),
        scratchy_serving_api::protocol::EmbeddingInputItem::Text("Foo bar".to_string()),
    ]));
    let response = client.embedding(&request).await.unwrap();

    assert_eq!(response.data.len(), 3);
    // All embeddings should have the same dimension.
    let dim = embedding_vector(&response.data[0]).len();
    assert!(dim > 0);
    assert_eq!(embedding_vector(&response.data[1]).len(), dim);
    assert_eq!(embedding_vector(&response.data[2]).len(), dim);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_pooling_embedding_normalized() {
    let server = TestServer::builder(TestModels::SMOLLM)
        .with_runner("pooling")
        .start()
        .await
        .expect("pooling server should start");

    let client = Client::new(server.base_url());

    let request = embed_request(EmbeddingInput::Single(
        "The quick brown fox jumps over the lazy dog".to_string(),
    ));
    let response = client.embedding(&request).await.unwrap();

    let emb = embedding_vector(&response.data[0]);
    let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        (norm - 1.0).abs() < 0.01,
        "L2 norm should be ~1.0, got {norm}"
    );
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_pooling_embedding_dimensions() {
    let server = TestServer::builder(TestModels::SMOLLM)
        .with_runner("pooling")
        .start()
        .await
        .expect("pooling server should start");

    let client = Client::new(server.base_url());

    let mut request = embed_request(EmbeddingInput::Single("Test input".to_string()));
    request.dimensions = Some(32);
    let response = client.embedding(&request).await.unwrap();

    assert_eq!(embedding_vector(&response.data[0]).len(), 32);

    // Should still be normalized after truncation.
    let emb = embedding_vector(&response.data[0]);
    let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        (norm - 1.0).abs() < 0.01,
        "L2 norm should be ~1.0 after truncation, got {norm}"
    );
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_pooling_different_inputs_differ() {
    let server = TestServer::builder(TestModels::SMOLLM)
        .with_runner("pooling")
        .start()
        .await
        .expect("pooling server should start");

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

    let emb1 = embedding_vector(&resp1.data[0]);
    let emb2 = embedding_vector(&resp2.data[0]);
    let cosine: f32 = emb1.iter().zip(emb2.iter()).map(|(a, b)| a * b).sum();
    assert!(
        cosine < 0.999,
        "Different inputs should have cosine similarity < 1.0, got {cosine}"
    );
}

// ===========================================================================
// Pooling mode — generation endpoint rejection
// ===========================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_pooling_rejects_chat_completions() {
    let server = TestServer::builder(TestModels::SMOLLM)
        .with_runner("pooling")
        .start()
        .await
        .expect("pooling server should start");

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

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_pooling_rejects_completions() {
    let server = TestServer::builder(TestModels::SMOLLM)
        .with_runner("pooling")
        .start()
        .await
        .expect("pooling server should start");

    let client = Client::new(server.base_url());

    let body = serde_json::json!({"prompt": "Hello"});
    let resp = client.completion_raw(&body).await.unwrap();
    assert_eq!(
        resp.status().as_u16(),
        400,
        "Completions should return 400 in pooling mode, got {}",
        resp.status()
    );
}

// ===========================================================================
// Pooling mode with mean pooling strategy
// ===========================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_pooling_mean_strategy() {
    let server = TestServer::builder(TestModels::SMOLLM)
        .with_runner("pooling")
        .with_pooling_strategy("mean")
        .start()
        .await
        .expect("pooling server should start");

    let client = Client::new(server.base_url());

    let request = embed_request(EmbeddingInput::Single("The quick brown fox".to_string()));
    let response = client.embedding(&request).await.unwrap();

    assert_eq!(response.data.len(), 1);
    assert!(!embedding_vector(&response.data[0]).is_empty());

    let emb = embedding_vector(&response.data[0]);
    let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        (norm - 1.0).abs() < 0.01,
        "L2 norm should be ~1.0, got {norm}"
    );
}
