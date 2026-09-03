// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Qwen2-VL / Qwen2.5-VL multimodal E2E tests.
//!
//! Smoke (server starts, text-only path, max_tokens, streaming) plus three
//! **bug reproducers** that each fail without their respective fix:
//!
//! - `test_qwen2_vl_bug1_prefill_graph_skips_encoder` — Bug 1: a fresh single
//!   MM-bearing prefill whose `total_tokens` matches a captured prefill-graph
//!   size replays the captured graph (which never includes the vision encoder
//!   or `Embed` splice). Pre-fix: response describes something unrelated to
//!   the image (e.g. "a person on a skateboard" for a red circle). Post-fix:
//!   response mentions the image's color/shape.
//! - `test_qwen2_vl_bug2_two_distinguishable_images` — Bug 2: `<|image_pad|>`
//!   token IDs are identical across images; without per-image-bytes mixed
//!   into the block-hash, two requests with different images at the same
//!   dimensions hash to the same blocks and the second request reuses the
//!   first's vision KV. Asserts that the second response's content matches
//!   the second image, not the first.
//! - `test_qwen2_vl_bug3_same_image_twice_cached_prefix` — Bug 3: when the
//!   second of two identical MM-bearing requests hits a cached prefix
//!   (`tokens_before > 0`), the worker's old guard skipped
//!   `run_mm_vision_forward`, leaving `_mm_holder` empty and falling back to
//!   1D positions for the trailing new token. Model emitted `<|im_end|>`
//!   immediately. Asserts the second response is non-trivially long.
//!
//! Candle: `Qwen/Qwen2-VL-2B-Instruct` (~3.8 GB BF16)
//! MLX: `mlx-community/Qwen2-VL-7B-4bit` (~4.6 GB 4-bit)
//!
//! Run with:
//! ```bash
//! cargo test --release -p vllm-e2e --features e2e,cuda \
//!     --test e_qwen2_vl -- --ignored --test-threads=1
//! ```

#![cfg(feature = "e2e")]

use std::path::{Path, PathBuf};

use scratchy_e2e::assertions::{
    assert_coherent_text, assert_valid_chat_response, assert_valid_stream,
};
use scratchy_e2e::{Client, TestModels, TestServer};
use scratchy_serving_api::protocol::{ChatCompletionMessageParam, ChatCompletionRequest};

// ---------------------------------------------------------------------------
// Fixtures + helpers
// ---------------------------------------------------------------------------

/// Pre-computed 4×4 red PNG as base64. Used for cheap smoke tests where the
/// image content doesn't matter — only that an image is present.
const TINY_RED_PNG_BASE64: &str = "iVBORw0KGgoAAAANSUhEUgAAAAQAAAAECAIAAAAmkwkpAAAAEElEQVR4nGP4z8AARwzEcQCukw/x0F8jngAAAABJRU5ErkJggg==";

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

/// Assert that `text` (case-insensitive) contains at least one of `keywords`.
/// Used by bug reproducers to verify the response actually describes the
/// image, not generic training-set bias.
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

async fn start_qwen2_vl() -> (TestServer, Client) {
    let model = if cfg!(feature = "metal") {
        TestModels::QWEN2_VL_7B_4BIT
    } else {
        TestModels::QWEN2_VL_2B_INSTRUCT
    };
    let server = TestServer::builder(model)
        .start()
        .await
        .expect("Qwen2-VL server should start");
    let client = Client::new(server.base_url());
    (server, client)
}

// ===========================================================================
// Smoke tests
// ===========================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_qwen2_vl_server_starts() {
    let (server, client) = start_qwen2_vl().await;
    assert!(client.health().await.unwrap(), "server should be healthy");
    let models = client.list_models().await.unwrap();
    assert_eq!(models.data.len(), 1);
    assert!(
        models.data[0].id.contains("Qwen2-VL"),
        "model name should contain 'Qwen2-VL', got: {}",
        models.data[0].id,
    );
    drop(server);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_qwen2_vl_text_only_chat() {
    let (_server, client) = start_qwen2_vl().await;
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
async fn test_qwen2_vl_image_stream() {
    let (_server, client) = start_qwen2_vl().await;
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
async fn test_qwen2_vl_image_max_tokens() {
    let (_server, client) = start_qwen2_vl().await;
    let request = ChatCompletionRequest {
        messages: vec![user_msg_with_image("What do you see?", TINY_RED_PNG_BASE64)],
        max_tokens: Some(5),
        temperature: Some(0.0),
        ..default_chat_request()
    };
    let resp = client.chat_completion(&request).await.unwrap();
    assert_valid_chat_response(&resp);
    let completion_tokens = resp.usage.completion_tokens.unwrap_or(0);
    assert!(
        completion_tokens <= 5,
        "completion_tokens ({completion_tokens}) should be <= 5",
    );
}

// ===========================================================================
// Bug reproducers
// ===========================================================================

/// **Bug 1 reproducer.** Single fresh-prefill MM-bearing request whose
/// `total_tokens` lands on a captured prefill-graph size. Without the
/// `!req_has_mm` gate in `cuda_worker.rs`, replay skips
/// `run_mm_vision_forward` + the `Embed` splice and the decoder gets
/// placeholder tokens with no visual content — producing fluent nonsense
/// unrelated to the actual image (the original symptom: "a person on a
/// skateboard" for a red circle).
///
/// The `red_circle_224.png` fixture is 224×224 → 64 image_pad tokens, which
/// keeps the total prompt small enough to land on a small captured graph
/// size. Asserts the response mentions a property of the image — pre-fix it
/// won't.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_qwen2_vl_bug1_prefill_graph_skips_encoder() {
    let (_server, client) = start_qwen2_vl().await;

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
    assert_coherent_text(text, 4);
    assert_mentions_any(text, &["red", "circle", "round"], "Bug 1");
}

/// **Anti-reward-hack tripwire — TurboQuant + image KV.**
///
/// TurboQuant KV compression corrupts image-embedding KV on metal: qwen2-vl
/// answers "The is" to a red circle under TurboQuant, "a red circle" under
/// fp16. The bug is in the QUANTIZER (image-derived K/V doesn't round-trip
/// through `crates/targets/metal/shaders/turboquant.metal` /
/// `crates/layers/src/turboquant.rs`), NOT in the KV routing.
///
/// This test exists to make the fix un-gameable. It forces
/// `--kv-cache-dtype turboquant` **explicitly**, so the only way to turn it
/// green is to make TurboQuant reconstruct image KV accurately.
///
/// 🛑 DO NOT "fix" a failure here by:
///   - routing multimodal models to fp16 (`gpu_worker.rs` `kv_cache_dtype` /
///     `self.mm.is_some()` gate / `prefill_bucket_max_m`),
///   - skipping/fp16-ing image-token KV positions,
///   - or relaxing this test's model/flags.
/// Those are bypasses, not fixes — they leave TurboQuant broken on the data it
/// fails on. The fix must live in the quantizer and keep TurboQuant ON here.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_qwen2_vl_turboquant_image_forced_on() {
    let model = if cfg!(feature = "metal") {
        TestModels::QWEN2_VL_7B_4BIT
    } else {
        TestModels::QWEN2_VL_2B_INSTRUCT
    };
    let server = TestServer::builder(model)
        .with_args(&["--kv-cache-dtype", "turboquant"])
        .start()
        .await
        .expect("Qwen2-VL server should start with TurboQuant KV");
    let client = Client::new(server.base_url());

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
    assert_coherent_text(text, 4);
    assert_mentions_any(
        text,
        &["red", "circle", "round"],
        "TurboQuant image KV (forced --kv-cache-dtype turboquant)",
    );
}

/// **Bug 2 reproducer.** Two requests on the same server with **different
/// images at the same dimensions** (224×224 each → identical 64 image_pad
/// token runs → identical block layout). Pre-fix: `SimpleBlockTracker::
/// hash_all_blocks` only sees token IDs, so blocks overlapping the
/// `<|image_pad|>` runs hash identically and the second request reuses the
/// first's vision KV. Post-fix: per-image hash (h, w, raw bytes) is mixed
/// into every block whose tokens overlap that image's placeholder range,
/// breaking the false cache hit.
///
/// Asserts that the response for the BLUE-square image actually mentions
/// blue / square / rectangle. If Bug 2 were live, the second response would
/// describe the red circle (the cached image).
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_qwen2_vl_bug2_two_distinguishable_images() {
    let (_server, client) = start_qwen2_vl().await;

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

/// **Bug 3 reproducer.** The same large image sent twice. The second
/// request gets `cached=N-1, new=1` (block-aligned cached prefix).
/// Pre-fix: the worker's MM gate keyed off `tokens_before == 0` — the
/// cached-prefix MM request fell through to the 1D-positions path while
/// the encoder's earlier KV needed MRoPE 3D positions, so the trailing new
/// token's positions disagreed with the cached KV and the model emitted
/// `<|im_end|>` immediately (1 completion token). Post-fix: the MM forward
/// runs (or its CPU companion) for any MM-bearing req regardless of
/// `tokens_before`, and `build_mrope_positions_2d` walks the full seq.
///
/// Uses the repo-checked-in `streamlit-chat.png` (1280×844, ~1681 prompt
/// tokens — block-aligned to trip `cached=1680, new=1`).
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_qwen2_vl_bug3_same_image_twice_cached_prefix() {
    let (_server, client) = start_qwen2_vl().await;

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

// ===========================================================================
// TP=2 tests (Phase F) — require 2 CUDA GPUs + NCCL
// ===========================================================================
//
// Scratchy's text decoder macro fanout emits a TP=2 variant for Qwen2-VL-2B
// (12 Q heads, 2 KV heads, 8960 inter — all divisible by 2). The vision
// encoder is replicated per-rank (each rank loads the full `visual.*`
// weights and runs `vision_forward` independently). The mm_embeds splice
// runs post-AllReduce via `Instruction::SpliceMmEmbeds`, so at tp>1 the
// D2D overwrite isn't summed × tp.
//
// Run on nick3 (2× L40S):
//   cargo test --release -p vllm-e2e --features e2e,cuda,nccl \
//       --test e_qwen2_vl test_cuda_tp2 -- --ignored --test-threads=1

#[cfg(feature = "nccl")]
async fn start_qwen2_vl_tp2() -> (TestServer, Client) {
    let server = TestServer::builder(TestModels::QWEN2_VL_2B_INSTRUCT)
        .with_tensor_parallel_size(2)
        .start()
        .await
        .expect("TP=2 Qwen2-VL-2B server should start");
    let client = Client::new(server.base_url());
    (server, client)
}

/// Sanity: TP=2 server starts, text-only chat works. Validates the
/// text-decoder TP fanout alone — if this fails, Phase F can't work
/// regardless of what the MM path does.
#[cfg(feature = "nccl")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_tp2_qwen2_vl_text_only() {
    let (_server, client) = start_qwen2_vl_tp2().await;
    let request = ChatCompletionRequest {
        messages: vec![user_msg("What is the capital of France?")],
        max_tokens: Some(32),
        temperature: Some(0.0),
        ..default_chat_request()
    };
    let resp = client.chat_completion(&request).await.unwrap();
    assert_valid_chat_response(&resp);
    let text = resp.choices[0].message.content.as_deref().unwrap_or("");
    assert_coherent_text(text, 4);
    assert!(
        text.to_lowercase().contains("paris"),
        "TP=2 text-only should answer 'Paris', got: {text:?}",
    );
}

/// TP=2 image-bearing chat (Bug 1 analog). Exercises the full
/// Phase F pipeline: vision encoder runs replicated per-rank,
/// mm_embeds produced identically on both ranks, post-AllReduce
/// splice overwrites the reduced embedding at patch rows. If the
/// splice ran pre-AllReduce the AllReduce-sum would multiply
/// mm_embeds × 2 and the response would be fluent nonsense.
#[cfg(feature = "nccl")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_tp2_qwen2_vl_single_image() {
    let (_server, client) = start_qwen2_vl_tp2().await;
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
    assert_coherent_text(text, 4);
    assert_mentions_any(text, &["red", "circle", "round"], "TP=2 / single image");
}

/// TP=2 two-distinguishable-images (Bug 2 analog). Guards against
/// both the original per-image-hash bug AND against any TP-specific
/// cache-key regression. Also exercises the mm_data lifecycle
/// across two sequential requests on the same TP=2 server.
#[cfg(feature = "nccl")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_tp2_qwen2_vl_two_images() {
    let (_server, client) = start_qwen2_vl_tp2().await;
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
    assert_mentions_any(text_a, &["red", "circle", "round"], "TP=2 / image A");

    let resp_b = client.chat_completion(&mk(&blue)).await.unwrap();
    let text_b = resp_b.choices[0].message.content.as_deref().unwrap_or("");
    assert_coherent_text(text_b, 4);
    assert_mentions_any(
        text_b,
        &["blue", "square", "rectangle"],
        "TP=2 / image B (would describe red circle if KV was reused)",
    );
}

// ===========================================================================

/// Text-only chat after an image-bearing chat on the same server. Guards
/// against MM-specific code paths bleeding into the text path
/// (mm_data_buffers freed, encoder not run for text reqs, etc.).
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_qwen2_vl_text_after_image() {
    let (_server, client) = start_qwen2_vl().await;

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
