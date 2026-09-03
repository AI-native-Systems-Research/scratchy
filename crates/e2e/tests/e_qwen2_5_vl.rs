// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Qwen2.5-VL multimodal E2E tests.
//!
//! Mirrors `e_qwen2_vl.rs` for the pure-DSL `qwen2-5-vl` arch
//! crate. Vision tower diffs from Qwen2-VL: RMSNorm vision blocks,
//! SwiGLU MLP, per-layer attention switching (full at indices
//! `[7, 15, 23, 31]`, windowed otherwise), entry-side window
//! permutation gather + reverse permutation post-merger.
//!
//! Run with:
//! ```bash
//! cargo test --release -p vllm-e2e --features e2e,cuda \
//!     --test e_qwen2_5_vl -- --ignored --test-threads=1
//! ```

#![cfg(feature = "e2e")]

use std::path::{Path, PathBuf};

use scratchy_e2e::assertions::{
    assert_coherent_text, assert_valid_chat_response, assert_valid_stream,
};
use scratchy_e2e::{Client, TestModels, TestServer};
use scratchy_serving_api::protocol::{ChatCompletionMessageParam, ChatCompletionRequest};

const TINY_RED_PNG_BASE64: &str = "iVBORw0KGgoAAAANSUhEUgAAAAQAAAAECAIAAAAmkwkpAAAAEElEQVR4nGP4z8AARwzEcQCukw/x0F8jngAAAABJRU5ErkJggg==";

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn load_fixture_b64(name: &str) -> String {
    use base64::Engine;
    let path = fixture_dir().join(name);
    let bytes = std::fs::read(&path)
        .unwrap_or_else(|e| panic!("failed to read fixture {}: {e}", path.display()));
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
         likely the vision encoder did not run or a different image's KV \
         was reused. text: {text:?}",
    );
}

async fn start_qwen2_5_vl() -> (TestServer, Client) {
    let server = TestServer::builder(TestModels::QWEN2_5_VL_3B_INSTRUCT)
        .start()
        .await
        .expect("Qwen2.5-VL server should start");
    let client = Client::new(server.base_url());
    (server, client)
}

// Smoke ---------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_qwen2_5_vl_server_starts() {
    let (server, client) = start_qwen2_5_vl().await;
    assert!(client.health().await.unwrap(), "server should be healthy");
    let models = client.list_models().await.unwrap();
    assert_eq!(models.data.len(), 1);
    assert!(
        models.data[0].id.contains("Qwen2.5-VL"),
        "model name should contain 'Qwen2.5-VL', got: {}",
        models.data[0].id,
    );
    drop(server);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_qwen2_5_vl_text_only_chat() {
    let (_server, client) = start_qwen2_5_vl().await;
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

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_qwen2_5_vl_image_stream() {
    let (_server, client) = start_qwen2_5_vl().await;
    let request = ChatCompletionRequest {
        messages: vec![user_msg_with_image(
            "Describe this image briefly.",
            TINY_RED_PNG_BASE64,
        )],
        max_tokens: Some(32),
        temperature: Some(0.0),
        stream: true,
        ..default_chat_request()
    };
    let chunks = client.chat_completion_stream(&request).await.unwrap();
    assert_valid_stream(&chunks);
    let full_text: String = chunks
        .iter()
        .filter_map(|c| c.choices.first().and_then(|ch| ch.delta.content.as_deref()))
        .collect();
    assert_coherent_text(&full_text, 1);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_qwen2_5_vl_image_max_tokens() {
    let (_server, client) = start_qwen2_5_vl().await;
    let request = ChatCompletionRequest {
        messages: vec![user_msg_with_image("What do you see?", TINY_RED_PNG_BASE64)],
        max_tokens: Some(5),
        temperature: Some(0.0),
        ..default_chat_request()
    };
    let resp = client.chat_completion(&request).await.unwrap();
    assert_valid_chat_response(&resp);
    let usage = resp.usage;
    assert!(
        usage.completion_tokens.unwrap_or(0) <= 5,
        "completion_tokens > 5: {usage:?}",
    );
}

// Bug reproducers -----------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_qwen2_5_vl_bug1_prefill_graph_skips_encoder() {
    let (_server, client) = start_qwen2_5_vl().await;
    let red = load_fixture_b64("red_circle_224.png");
    let request = ChatCompletionRequest {
        messages: vec![user_msg_with_image(
            "Describe this image. What color and shape do you see?",
            &red,
        )],
        max_tokens: Some(48),
        temperature: Some(0.0),
        ..default_chat_request()
    };
    let resp = client.chat_completion(&request).await.unwrap();
    assert_valid_chat_response(&resp);
    let text = resp.choices[0].message.content.as_deref().unwrap_or("");
    eprintln!("[qwen2.5-vl bug1] response: {text:?}");
    assert_coherent_text(text, 4);
    assert_mentions_any(text, &["red", "circle", "round"], "Bug 1");
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_qwen2_5_vl_bug2_two_distinguishable_images() {
    let (_server, client) = start_qwen2_5_vl().await;
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
    assert_mentions_any(text_a, &["red", "circle", "round"], "Bug 2 / image A");

    let resp_b = client.chat_completion(&mk(&blue)).await.unwrap();
    let text_b = resp_b.choices[0].message.content.as_deref().unwrap_or("");
    assert_coherent_text(text_b, 4);
    assert_mentions_any(
        text_b,
        &["blue", "square", "rectangle"],
        "Bug 2 / image B (would describe red circle if KV was reused)",
    );

    let lower_b = text_b.to_lowercase();
    assert!(
        !(lower_b.contains("red") && lower_b.contains("circle")),
        "Bug 2: image B's response describes the red circle ({text_b:?}) — \
         prefix cache likely reused image A's vision KV.",
    );
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_qwen2_5_vl_bug3_same_image_twice_cached_prefix() {
    let (_server, client) = start_qwen2_5_vl().await;

    let png_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../docs/assets/deployment/streamlit-chat.png");
    let png_bytes =
        std::fs::read(&png_path).expect("streamlit-chat.png must exist for Bug 3 reproducer");
    use base64::Engine;
    let png_b64 = base64::engine::general_purpose::STANDARD.encode(&png_bytes);

    let make = || ChatCompletionRequest {
        messages: vec![user_msg_with_image(
            "What text appears on the page in this screenshot?",
            &png_b64,
        )],
        max_tokens: Some(64),
        temperature: Some(0.0),
        ..default_chat_request()
    };

    let resp_first = client.chat_completion(&make()).await.unwrap();
    assert_valid_chat_response(&resp_first);
    let first = resp_first.choices[0]
        .message
        .content
        .as_deref()
        .unwrap_or("");
    eprintln!("[qwen2.5-vl bug3 first] response: {first:?}");
    assert_coherent_text(first, 8);

    let resp_second = client.chat_completion(&make()).await.unwrap();
    assert_valid_chat_response(&resp_second);
    let second = resp_second.choices[0]
        .message
        .content
        .as_deref()
        .unwrap_or("");
    let completion_second = resp_second.usage.completion_tokens.unwrap_or(0);
    assert!(
        completion_second >= 8,
        "Bug 3: repeat-image collapsed to {completion_second} completion \
         tokens — cached prefix path skipped vision encoder and MRoPE \
         positions for the trailing new token fell back to 1D. text: {second:?}",
    );
    assert_coherent_text(second, 8);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_qwen2_5_vl_text_after_image() {
    let (_server, client) = start_qwen2_5_vl().await;

    let image_request = ChatCompletionRequest {
        messages: vec![user_msg_with_image("What do you see?", TINY_RED_PNG_BASE64)],
        max_tokens: Some(32),
        temperature: Some(0.0),
        ..default_chat_request()
    };
    let _ = client.chat_completion(&image_request).await.unwrap();

    let text_request = ChatCompletionRequest {
        messages: vec![user_msg("What is the capital of France?")],
        max_tokens: Some(32),
        temperature: Some(0.0),
        ..default_chat_request()
    };
    let resp = client.chat_completion(&text_request).await.unwrap();
    assert_valid_chat_response(&resp);
    let text = resp.choices[0].message.content.as_deref().unwrap_or("");
    assert_coherent_text(text, 8);
    assert!(
        text.to_lowercase().contains("paris"),
        "text-only after MM should still answer 'Paris', got: {text:?}",
    );
}
