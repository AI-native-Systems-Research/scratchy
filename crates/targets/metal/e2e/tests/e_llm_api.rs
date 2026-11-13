// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! E2E tests for the offline batch `LLM` API.
//!
//! These tests exercise `LLM::new()`, `LLM::generate()`, and `LLM::chat()`
//! directly — no HTTP server involved.
//!
//! Run with: `cargo test -p vllm-e2e --features e2e --test e_llm_api -- --ignored`

#![cfg(feature = "e2e")]

use scratchy_e2e::TestModels;
use scratchy_serving_api::llm::{ChatMessage, LLM, SamplingParams};

// ---------------------------------------------------------------------------
// generate()
// ---------------------------------------------------------------------------

#[test]
#[ignore]
fn test_llm_generate_basic() {
    scratchy_core_common::telemetry::init_tracing("off");

    let mut llm = LLM::new(TestModels::SMOLLM).expect("LLM should initialize");
    assert!(llm.model_name().contains("SmolLM"));

    let outputs = llm
        .generate(&["The capital of France is"], None)
        .expect("generate should succeed");

    assert_eq!(outputs.len(), 1);
    let output = &outputs[0];
    assert!(output.finished);
    assert_eq!(output.outputs.len(), 1);
    assert!(
        !output.outputs[0].text.is_empty(),
        "generated text should not be empty"
    );
}

#[test]
#[ignore]
fn test_llm_generate_multiple_prompts() {
    scratchy_core_common::telemetry::init_tracing("off");

    let mut llm = LLM::new(TestModels::SMOLLM).expect("LLM should initialize");

    let prompts = &["Hello, world!", "The meaning of life is"];
    let params = SamplingParams {
        max_tokens: Some(20),
        temperature: 0.0,
        ..SamplingParams::default()
    };

    let outputs = llm
        .generate(prompts, Some(params))
        .expect("generate should succeed");

    assert_eq!(outputs.len(), 2);
    for (i, output) in outputs.iter().enumerate() {
        assert!(output.finished, "output {i} should be finished");
        assert!(
            !output.outputs[0].text.is_empty(),
            "output {i} should have text"
        );
        assert_eq!(
            output.prompt.as_deref(),
            Some(prompts[i]),
            "prompt should be preserved"
        );
    }
}

#[test]
#[ignore]
fn test_llm_generate_max_tokens() {
    scratchy_core_common::telemetry::init_tracing("off");

    let mut llm = LLM::new(TestModels::SMOLLM).expect("LLM should initialize");

    let params = SamplingParams {
        max_tokens: Some(3),
        temperature: 0.0,
        ..SamplingParams::default()
    };

    let outputs = llm
        .generate(&["Write a very long story about a dragon."], Some(params))
        .expect("generate should succeed");

    assert_eq!(outputs.len(), 1);
    let output = &outputs[0];
    assert!(output.finished);
    assert!(
        output.outputs[0].finish_reason.as_deref() == Some("length"),
        "should stop due to length, got: {:?}",
        output.outputs[0].finish_reason
    );
}

// ---------------------------------------------------------------------------
// chat()
// ---------------------------------------------------------------------------

#[test]
#[ignore]
fn test_llm_chat_basic() {
    scratchy_core_common::telemetry::init_tracing("off");

    let mut llm = LLM::new(TestModels::SMOLLM).expect("LLM should initialize");

    let messages = vec![ChatMessage::user("Say hello in one sentence.")];
    let params = SamplingParams {
        max_tokens: Some(50),
        temperature: 0.0,
        ..SamplingParams::default()
    };

    let output = llm
        .chat(&messages, Some(params))
        .expect("chat should succeed");

    assert!(output.finished);
    assert_eq!(output.outputs.len(), 1);
    let text = &output.outputs[0].text;
    assert!(!text.is_empty(), "chat response should not be empty");
    assert!(text.len() >= 2, "chat response should be at least 2 chars");
}

#[test]
#[ignore]
fn test_llm_chat_with_system_message() {
    scratchy_core_common::telemetry::init_tracing("off");

    let mut llm = LLM::new(TestModels::SMOLLM).expect("LLM should initialize");

    let messages = vec![
        ChatMessage::system("You are a helpful assistant."),
        ChatMessage::user("What is 2+2?"),
    ];
    let params = SamplingParams {
        max_tokens: Some(30),
        temperature: 0.0,
        ..SamplingParams::default()
    };

    let output = llm
        .chat(&messages, Some(params))
        .expect("chat should succeed");

    assert!(output.finished);
    assert!(!output.outputs[0].text.is_empty());
}

// ---------------------------------------------------------------------------
// chat_stream()
// ---------------------------------------------------------------------------

#[test]
#[ignore]
fn test_llm_chat_stream_deltas_not_cumulative() {
    scratchy_core_common::telemetry::init_tracing("off");

    let mut llm = LLM::new(TestModels::SMOLLM).expect("LLM should initialize");

    let messages = vec![ChatMessage::user("Count from 1 to 5.")];
    let params = SamplingParams {
        max_tokens: Some(30),
        temperature: 0.0,
        ..SamplingParams::default()
    };

    let mut chunks: Vec<String> = Vec::new();
    let output = llm
        .chat_stream(&messages, Some(params), |delta| {
            chunks.push(delta.to_string());
        })
        .expect("chat_stream should succeed");

    assert!(output.finished);
    assert!(!chunks.is_empty(), "should have received streaming chunks");

    // The concatenation of all deltas must equal the final output text.
    let concatenated: String = chunks.iter().map(|s| s.as_str()).collect();
    assert_eq!(
        concatenated, output.outputs[0].text,
        "concatenated deltas must equal final text"
    );

    // No individual chunk should contain a previous chunk's text (i.e., no
    // cumulative re-emission). Check that no chunk is a prefix of a later
    // chunk — the bug was that each "delta" was actually the full text so far.
    if chunks.len() >= 2 {
        for i in 1..chunks.len() {
            assert!(
                !chunks[i].starts_with(&chunks[0]),
                "chunk {} ({:?}) looks like cumulative text (starts with first chunk {:?})",
                i,
                chunks[i],
                chunks[0],
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

#[test]
#[ignore]
fn test_llm_builder() {
    scratchy_core_common::telemetry::init_tracing("off");

    let mut llm = LLM::builder(TestModels::SMOLLM)
        .max_model_len(512)
        .build()
        .expect("LLM builder should succeed");

    assert_eq!(llm.max_model_len(), 512);

    let outputs = llm
        .generate(&["Hello"], None)
        .expect("generate should succeed");
    assert_eq!(outputs.len(), 1);
    assert!(outputs[0].finished);
}
