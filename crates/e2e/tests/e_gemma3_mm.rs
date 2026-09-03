// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Gemma3-MM (Gemma3ForConditionalGeneration) image-inference E2E.
//!
//! Validates the SigLIP-vision + Gemma3-decoder stack end-to-end through
//! the multimodal request pipeline:
//!
//! - The chat template renders `<start_of_image>` for each image content
//!   part (after the universal `image_url` → `image` part-type
//!   normalization in `scratchy-serving-api`).
//! - Engine dispatches generically through
//!   `scratchy_vision::MmMetadata` declared by the `gemma3-mm` arch
//!   (placeholder = boi_token_index 255999, fixed 896² resize, symmetric
//!   ±1 normalization, 256 fixed tokens per image).
//! - Vision encoder runs (DSL body) and 256 projected vectors are
//!   spliced into the placeholder positions.
//!
//! Mirrors `e_qwen2_vl.rs` but with the SigLIP-class fixed-token policy
//! instead of Qwen's per-image grid. Only one bug-class reproducer is
//! kept (image-coherence): the cached-prefix and per-image hash bugs
//! that qwen2-vl's bug{2,3} cover are arch-independent and exercising
//! them on Gemma3-MM is duplicate coverage.
//!
//! Model: google/gemma-3-4b-it (~8 GB BF16)
//!
//! Run with:
//! ```bash
//! cargo test --release -p vllm-e2e --features e2e,cuda \
//!     --test e_gemma3_mm -- --ignored --test-threads=1
//! ```

#![cfg(feature = "e2e")]

use std::path::{Path, PathBuf};

use scratchy_e2e::assertions::{assert_coherent_text, assert_valid_chat_response};
use scratchy_e2e::{Client, TestModels, TestServer};
use scratchy_serving_api::protocol::{ChatCompletionMessageParam, ChatCompletionRequest};

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn load_fixture_b64(name: &str) -> String {
    use base64::Engine;
    let path = fixture_dir().join(name);
    let bytes = std::fs::read(&path)
        .unwrap_or_else(|e| panic!("fixture {} must exist: {e}", path.display()));
    base64::engine::general_purpose::STANDARD.encode(&bytes)
}

fn default_chat_request() -> ChatCompletionRequest {
    serde_json::from_str(r#"{"messages": []}"#).unwrap()
}

fn user_msg(content: &str) -> ChatCompletionMessageParam {
    ChatCompletionMessageParam {
        role: "user".to_string(),
        content: Some(serde_json::Value::String(content.to_string())),
        name: None,
        tool_calls: None,
        tool_call_id: None,
    }
}

fn user_msg_with_image(text: &str, image_base64: &str) -> ChatCompletionMessageParam {
    let content = serde_json::json!([
        {
            "type": "image_url",
            "image_url": { "url": format!("data:image/png;base64,{}", image_base64) }
        },
        { "type": "text", "text": text }
    ]);
    ChatCompletionMessageParam {
        role: "user".to_string(),
        content: Some(content),
        name: None,
        tool_calls: None,
        tool_call_id: None,
    }
}

fn assert_mentions_any(text: &str, keywords: &[&str], ctx: &str) {
    let lower = text.to_lowercase();
    let hit = keywords.iter().any(|k| lower.contains(&k.to_lowercase()));
    assert!(
        hit,
        "{ctx}: response does not mention any of {keywords:?} — \
         vision encoder likely did not engage. text: {text:?}",
    );
}

async fn start_gemma3_mm() -> (TestServer, Client) {
    let server = TestServer::builder(TestModels::GEMMA3_4B_IT)
        .start()
        .await
        .expect("Gemma3-MM server should start");
    let client = Client::new(server.base_url());
    (server, client)
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_gemma3_mm_server_starts() {
    let (server, client) = start_gemma3_mm().await;
    assert!(client.health().await.unwrap(), "server should be healthy");
    let models = client.list_models().await.unwrap();
    assert_eq!(models.data.len(), 1);
    drop(server);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_gemma3_mm_text_only_chat() {
    let (_server, client) = start_gemma3_mm().await;
    let request = ChatCompletionRequest {
        messages: vec![user_msg("What is 2 + 2?")],
        max_tokens: Some(32),
        temperature: Some(0.0),
        ..default_chat_request()
    };
    let resp = client.chat_completion(&request).await.unwrap();
    assert_valid_chat_response(&resp);
    let text = resp.choices[0].message.content.as_deref().unwrap_or("");
    assert_coherent_text(text, 1);
}

/// Image-coherence: the model describes the actual image content. Pre-
/// G.7(d): the chat template never renders `<start_of_image>` for an
/// `image_url` part, so no placeholder reaches the engine, the vision
/// encoder never runs, and the response describes "an image you have
/// not yet provided". Post: the universal `image_url` → `image` chat-
/// template normalization plus the inventory-driven SigLIP preprocess
/// land 256 spliced patches and the response mentions red / circle.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_gemma3_mm_image_red_circle() {
    let (_server, client) = start_gemma3_mm().await;
    let red = load_fixture_b64("red_circle_224.png");
    let request = ChatCompletionRequest {
        messages: vec![user_msg_with_image(
            "Describe this image. What color and shape do you see?",
            &red,
        )],
        max_tokens: Some(64),
        temperature: Some(0.0),
        ..default_chat_request()
    };
    let resp = client.chat_completion(&request).await.unwrap();
    assert_valid_chat_response(&resp);
    let text = resp.choices[0].message.content.as_deref().unwrap_or("");
    assert_coherent_text(text, 4);
    assert_mentions_any(
        text,
        &["red", "circle", "round"],
        "Gemma3-MM image-coherence",
    );
}

/// Two distinguishable images at the same dimensions. Same arch-
/// agnostic spirit as `test_qwen2_vl_bug2_two_distinguishable_images`:
/// guards against false KV reuse across requests with different image
/// bytes that hash-collide on token IDs.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_gemma3_mm_two_distinguishable_images() {
    let (_server, client) = start_gemma3_mm().await;

    let red = load_fixture_b64("red_circle_224.png");
    let blue = load_fixture_b64("blue_square_224.png");

    let mk = |b64: &str| ChatCompletionRequest {
        messages: vec![user_msg_with_image(
            "Describe this image. What color and shape do you see?",
            b64,
        )],
        max_tokens: Some(48),
        temperature: Some(0.0),
        ..default_chat_request()
    };

    let resp_a = client.chat_completion(&mk(&red)).await.unwrap();
    let text_a = resp_a.choices[0].message.content.as_deref().unwrap_or("");
    assert_coherent_text(text_a, 4);
    assert_mentions_any(text_a, &["red", "circle", "round"], "image A (red circle)");

    let resp_b = client.chat_completion(&mk(&blue)).await.unwrap();
    let text_b = resp_b.choices[0].message.content.as_deref().unwrap_or("");
    assert_coherent_text(text_b, 4);
    assert_mentions_any(
        text_b,
        &["blue", "square", "rectangle"],
        "image B (blue square)",
    );

    let lower_b = text_b.to_lowercase();
    assert!(
        !(lower_b.contains("red") && lower_b.contains("circle")),
        "image B response describes the red circle ({text_b:?}) — \
         prefix cache likely reused image A's vision KV.",
    );
}
