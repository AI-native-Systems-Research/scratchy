// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! E2E tests for `LLM::embed()` — the synchronous in-process embedding API.
//!
//! Run with: `cargo test -p vllm-e2e --features e2e --test e_llm_embed -- --ignored`

#![cfg(feature = "e2e")]

use scratchy_e2e::TestModels;
use scratchy_serving_api::llm::LLMBuilder;

/// Single string embedding should return a non-empty vector.
#[test]
#[ignore]
fn test_llm_embed_single() {
    scratchy_core_common::telemetry::init_tracing("off");

    let mut llm = LLMBuilder::new(TestModels::SMOLLM)
        .build()
        .expect("LLM should initialize");

    let embeddings = llm.embed(&["Hello world"]).expect("embed should succeed");

    assert_eq!(embeddings.len(), 1, "should return one embedding");
    assert!(
        !embeddings[0].is_empty(),
        "embedding vector should not be empty"
    );
    // Verify it's a real vector, not all zeros
    assert!(
        embeddings[0].iter().any(|&v| v != 0.0),
        "embedding should contain non-zero values"
    );
}

/// Multiple prompts should return one embedding per prompt.
#[test]
#[ignore]
fn test_llm_embed_multiple() {
    scratchy_core_common::telemetry::init_tracing("off");

    let mut llm = LLMBuilder::new(TestModels::SMOLLM)
        .build()
        .expect("LLM should initialize");

    let embeddings = llm
        .embed(&["Hello", "World", "Foo bar baz"])
        .expect("embed should succeed");

    assert_eq!(embeddings.len(), 3, "should return three embeddings");
    for (i, emb) in embeddings.iter().enumerate() {
        assert!(!emb.is_empty(), "embedding {i} should not be empty");
    }

    // All embeddings should have the same dimensionality
    let dim = embeddings[0].len();
    for (i, emb) in embeddings.iter().enumerate() {
        assert_eq!(
            emb.len(),
            dim,
            "embedding {i} has different dimension: {} vs {dim}",
            emb.len()
        );
    }
}

/// Different inputs should produce different embeddings.
#[test]
#[ignore]
fn test_llm_embed_different_inputs_differ() {
    scratchy_core_common::telemetry::init_tracing("off");

    let mut llm = LLMBuilder::new(TestModels::SMOLLM)
        .build()
        .expect("LLM should initialize");

    let embeddings = llm
        .embed(&["The cat sat on the mat", "Quantum mechanics is fascinating"])
        .expect("embed should succeed");

    assert_eq!(embeddings.len(), 2);
    // Compute cosine similarity — semantically different inputs should not be identical
    let dot: f32 = embeddings[0]
        .iter()
        .zip(&embeddings[1])
        .map(|(a, b)| a * b)
        .sum();
    let norm0: f32 = embeddings[0].iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm1: f32 = embeddings[1].iter().map(|x| x * x).sum::<f32>().sqrt();
    let cosine = dot / (norm0 * norm1);

    assert!(
        cosine < 0.99,
        "different inputs should produce different embeddings (cosine={cosine:.4})"
    );
}

/// Embedding the same input twice should produce identical results.
#[test]
#[ignore]
fn test_llm_embed_deterministic() {
    scratchy_core_common::telemetry::init_tracing("off");

    let mut llm = LLMBuilder::new(TestModels::SMOLLM)
        .build()
        .expect("LLM should initialize");

    let emb1 = llm.embed(&["Hello world"]).expect("first embed");
    let emb2 = llm.embed(&["Hello world"]).expect("second embed");

    assert_eq!(emb1.len(), 1);
    assert_eq!(emb2.len(), 1);
    assert_eq!(
        emb1[0], emb2[0],
        "same input should produce identical embeddings"
    );
}
