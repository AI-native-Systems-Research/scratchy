// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! E2E tests for RAG augmentation — indexing, retrieval, and sidecar spawning.
//!
//! Run with:
//!   cargo test -p vllm-e2e --features e2e,rag --test e_augment -- --ignored

#![cfg(all(feature = "e2e", feature = "rag"))]

use scratchy_e2e::TestModels;
use scratchy_serving_api::llm::LLMBuilder;
use spnl_core::ir::{Augment, Document, Generate, GenerateMetadata, Message, Query as SpnlQuery};

/// A small model for sidecar embedding tests — must differ from the generate
/// model so the sidecar path triggers. Any decoder model works with
/// `--runner pooling`; we use the F16 SmolLM2 variant which differs from the
/// 4-bit SmolLM used for generation.
const EMBEDDING_MODEL: &str = TestModels::SMOLLM_135M_F16;

/// Build an Augment query JSON that wraps a Generate around the augmented body.
fn make_augment_query(model: &str, embedding_model: &str, doc_text: &str) -> String {
    let query = SpnlQuery::Generate(Generate {
        metadata: GenerateMetadata {
            model: model.to_string(),
            max_tokens: Some(32),
            temperature: Some(0.0),
        },
        input: Box::new(SpnlQuery::Plus(vec![
            SpnlQuery::Augment(Augment {
                embedding_model: embedding_model.to_string(),
                body: Box::new(SpnlQuery::Message(Message::User(
                    "What is the capital of France?".to_string(),
                ))),
                doc: (
                    "test_doc.txt".to_string(),
                    Document::Text(doc_text.to_string()),
                ),
            }),
            SpnlQuery::Message(Message::User(
                "Based on the above context, answer: What is the capital of France?".to_string(),
            )),
        ])),
    });
    serde_json::to_string(&query).expect("query should serialize")
}

const DOC_TEXT: &str = "Paris is the capital and largest city of France. \
                        It is situated on the River Seine. \
                        The city has a population of over 2 million people. \
                        France is a country in Western Europe.";

/// In-process augment: same model for generate and embed. The Augment node
/// is indexed and retrieved using the engine's own embedding capability,
/// then rewritten to Plus(Message::User(...)) fragments for generation.
#[test]
#[ignore]
fn test_augment_inprocess() {
    scratchy_core_common::telemetry::init_tracing("info");

    let model = TestModels::SMOLLM;
    let mut llm = LLMBuilder::new(model).build().expect("LLM should init");

    let query = make_augment_query(model, model, DOC_TEXT);

    let output = llm
        .execute_query(&query, None, false, false)
        .expect("augment query should succeed");

    assert!(
        !output.steps.is_empty(),
        "should produce at least one output step"
    );
    let text = &output.steps[0].output.outputs[0].text;
    assert!(
        !text.is_empty(),
        "generated text should not be empty: {text:?}"
    );
    eprintln!("[test_augment_inprocess] generated: {text}");

    // Clean up index files
    let _ = std::fs::remove_dir_all("data/spnl");
}

/// Sidecar augment: the Augment node references a dedicated embedding model
/// (bge-small-en) that differs from the generate model (SmolLM). The sidecar
/// manager automatically spawns a vllm-rs child process serving the embedding
/// model with `--runner pooling`, waits for it to become healthy, and routes
/// embedding requests through it.
#[test]
#[ignore]
fn test_augment_sidecar() {
    scratchy_core_common::telemetry::init_tracing("info");

    let model = TestModels::SMOLLM;
    let mut llm = LLMBuilder::new(model).build().expect("LLM should init");

    let query = make_augment_query(model, EMBEDDING_MODEL, DOC_TEXT);

    let output = llm
        .execute_query(&query, None, false, false)
        .expect("sidecar augment query should succeed");

    assert!(
        !output.steps.is_empty(),
        "should produce at least one output step"
    );
    let text = &output.steps[0].output.outputs[0].text;
    assert!(!text.is_empty(), "generated text should not be empty");
    eprintln!("[test_augment_sidecar] generated: {text}");

    // Clean up
    let _ = std::fs::remove_dir_all("data/spnl");
}
