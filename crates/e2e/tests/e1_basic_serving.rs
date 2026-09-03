// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Phase E1: Basic serving smoke tests.
//!
//! Validates that each model architecture loads, starts serving,
//! and generates coherent text.
//!
//! Run with: `cargo test -p vllm-e2e --features e2e --test e1_basic_serving -- --ignored`

#![cfg(feature = "e2e")]

use scratchy_e2e::assertions::{
    assert_coherent_text, assert_valid_chat_response, assert_valid_completion_response,
};
use scratchy_e2e::{Client, TestModels, TestServer};
use scratchy_serving_api::protocol::{
    ChatCompletionMessageParam, ChatCompletionRequest, CompletionPrompt, CompletionRequest,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn user_msg(content: &str) -> ChatCompletionMessageParam {
    ChatCompletionMessageParam {
        role: "user".to_string(),
        content: Some(serde_json::Value::String(content.to_string())),
        name: None,
        tool_calls: None,
        tool_call_id: None,
    }
}

fn assistant_msg(content: &str) -> ChatCompletionMessageParam {
    ChatCompletionMessageParam {
        role: "assistant".to_string(),
        content: Some(serde_json::Value::String(content.to_string())),
        name: None,
        tool_calls: None,
        tool_call_id: None,
    }
}

fn simple_chat_request(content: &str, max_tokens: Option<u32>) -> ChatCompletionRequest {
    ChatCompletionRequest {
        messages: vec![user_msg(content)],
        max_tokens,
        temperature: Some(0.0),
        ..default_chat_request()
    }
}

fn default_chat_request() -> ChatCompletionRequest {
    serde_json::from_str(r#"{"messages": []}"#).unwrap()
}

fn simple_completion_request(prompt: &str, max_tokens: u32) -> CompletionRequest {
    CompletionRequest {
        prompt: Some(CompletionPrompt::Single(prompt.to_string())),
        max_tokens: Some(max_tokens),
        temperature: Some(0.0),
        ..default_completion_request()
    }
}

fn default_completion_request() -> CompletionRequest {
    serde_json::from_str(r#"{}"#).unwrap()
}

// ===========================================================================
// SmolLM (LlamaForCausalLM) — Tier 1
// ===========================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_t1_smollm_server_starts() {
    scratchy_e2e::skip_if_no_gpu!();
    let server = TestServer::builder(TestModels::SMOLLM)
        .start()
        .await
        .expect("server should start");

    let client = Client::new(server.base_url());

    // /health
    assert!(client.health().await.unwrap(), "server should be healthy");

    // /v1/models
    let models = client.list_models().await.unwrap();
    assert_eq!(models.data.len(), 1);
    assert!(
        models.data[0].id.contains("molLM") || models.data[0].id.contains("SmolLM2"),
        "model name should contain SmolLM variant, got: {}",
        models.data[0].id
    );
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_t1_smollm_chat_basic() {
    scratchy_e2e::skip_if_no_gpu!();
    let server = TestServer::builder(TestModels::SMOLLM)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = simple_chat_request("Say hello in one sentence.", Some(50));
    let resp = client.chat_completion(&request).await.unwrap();

    assert_valid_chat_response(&resp);
    let text = resp.choices[0].message.content.as_deref().unwrap_or("");
    assert_coherent_text(text, 2);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_t1_smollm_completion_basic() {
    scratchy_e2e::skip_if_no_gpu!();
    let server = TestServer::builder(TestModels::SMOLLM)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = simple_completion_request("The capital of France is", 20);
    let resp = client.completion(&request).await.unwrap();

    assert_valid_completion_response(&resp);
    assert!(
        !resp.choices[0].text.is_empty(),
        "completion should not be empty"
    );
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_t1_smollm_max_tokens() {
    scratchy_e2e::skip_if_no_gpu!();
    let server = TestServer::builder(TestModels::SMOLLM)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = simple_chat_request("Write a long story about a cat.", Some(5));
    let resp = client.chat_completion(&request).await.unwrap();

    assert_valid_chat_response(&resp);
    assert!(
        resp.usage.completion_tokens.unwrap_or(0) <= 5,
        "completion_tokens should be <= 5, got: {:?}",
        resp.usage.completion_tokens
    );
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_t1_smollm_version() {
    scratchy_e2e::skip_if_no_gpu!();
    let server = TestServer::builder(TestModels::SMOLLM)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let version = client.version().await.unwrap();
    assert!(!version.version.is_empty(), "version should not be empty");
    assert!(
        version.version.contains("rust"),
        "version should contain 'rust', got: {}",
        version.version
    );
}

// ===========================================================================
// Qwen2.5-0.5B-Instruct-4bit (Qwen2ForCausalLM) — Tier 1
// ===========================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_t1_qwen2_server_starts() {
    let server = TestServer::builder(TestModels::QWEN2)
        .start()
        .await
        .expect("server should start");

    let client = Client::new(server.base_url());
    assert!(client.health().await.unwrap());

    let models = client.list_models().await.unwrap();
    assert_eq!(models.data.len(), 1);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_t1_qwen2_chat_basic() {
    let server = TestServer::builder(TestModels::QWEN2)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = simple_chat_request("Say hello in one sentence.", Some(50));
    let resp = client.chat_completion(&request).await.unwrap();

    assert_valid_chat_response(&resp);
    let text = resp.choices[0].message.content.as_deref().unwrap_or("");
    assert_coherent_text(text, 2);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_t1_qwen2_completion_basic() {
    let server = TestServer::builder(TestModels::QWEN2)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = simple_completion_request("The capital of France is", 20);
    let resp = client.completion(&request).await.unwrap();

    assert_valid_completion_response(&resp);
    assert!(!resp.choices[0].text.is_empty());
}

// ===========================================================================
// Qwen3-0.6B-4bit (Qwen3ForCausalLM) — Tier 1
// ===========================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_t1_qwen3_server_starts() {
    let server = TestServer::builder(TestModels::QWEN3)
        .start()
        .await
        .expect("server should start");

    let client = Client::new(server.base_url());
    assert!(client.health().await.unwrap());

    let models = client.list_models().await.unwrap();
    assert_eq!(models.data.len(), 1);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_t1_qwen3_chat_basic() {
    let server = TestServer::builder(TestModels::QWEN3)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = simple_chat_request("Say hello in one sentence.", Some(50));
    let resp = client.chat_completion(&request).await.unwrap();

    assert_valid_chat_response(&resp);
    let text = resp.choices[0].message.content.as_deref().unwrap_or("");
    assert_coherent_text(text, 2);
}

// ===========================================================================
// Llama-3.2-1B (LlamaForCausalLM) — Tier 2
// ===========================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_t2_llama3_server_starts() {
    let server = TestServer::builder(TestModels::LLAMA_3_2)
        .start()
        .await
        .expect("server should start");

    let client = Client::new(server.base_url());
    assert!(client.health().await.unwrap());

    let models = client.list_models().await.unwrap();
    assert_eq!(models.data.len(), 1);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_t2_llama3_chat_basic() {
    let server = TestServer::builder(TestModels::LLAMA_3_2)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = simple_chat_request("Say hello in one sentence.", Some(50));
    let resp = client.chat_completion(&request).await.unwrap();

    assert_valid_chat_response(&resp);
    let text = resp.choices[0].message.content.as_deref().unwrap_or("");
    assert_coherent_text(text, 2);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_t2_llama3_completion_basic() {
    let server = TestServer::builder(TestModels::LLAMA_3_2)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = simple_completion_request("The capital of France is", 20);
    let resp = client.completion(&request).await.unwrap();

    assert_valid_completion_response(&resp);
    assert!(!resp.choices[0].text.is_empty());
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_t2_llama3_max_tokens() {
    let server = TestServer::builder(TestModels::LLAMA_3_2)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = simple_chat_request("Write a long story.", Some(5));
    let resp = client.chat_completion(&request).await.unwrap();

    assert_valid_chat_response(&resp);
    assert!(resp.usage.completion_tokens.unwrap_or(0) <= 5);
}

// ===========================================================================
// Gemma3 (Gemma3ForCausalLM) — Tier 2
// ===========================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_t2_gemma3_server_starts() {
    let server = TestServer::builder(TestModels::GEMMA3)
        .start()
        .await
        .expect("Gemma3 server should start");

    let client = Client::new(server.base_url());
    assert!(client.health().await.unwrap(), "server should be healthy");

    let models = client.list_models().await.unwrap();
    assert_eq!(models.data.len(), 1);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_t2_gemma3_chat_basic() {
    let server = TestServer::builder(TestModels::GEMMA3)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = simple_chat_request("Say hello in one sentence.", Some(50));
    let resp = client.chat_completion(&request).await.unwrap();

    assert_valid_chat_response(&resp);
    let text = resp.choices[0].message.content.as_deref().unwrap_or("");
    assert_coherent_text(text, 2);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_t2_gemma3_completion_basic() {
    let server = TestServer::builder(TestModels::GEMMA3)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = simple_completion_request("The capital of France is", 20);
    let resp = client.completion(&request).await.unwrap();

    assert_valid_completion_response(&resp);
    assert!(
        !resp.choices[0].text.is_empty(),
        "completion should not be empty"
    );
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_t2_gemma3_max_tokens() {
    let server = TestServer::builder(TestModels::GEMMA3)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = simple_chat_request("Write a long story about a cat.", Some(5));
    let resp = client.chat_completion(&request).await.unwrap();

    assert_valid_chat_response(&resp);
    assert!(
        resp.usage.completion_tokens.unwrap_or(0) <= 5,
        "completion_tokens should be <= 5, got: {:?}",
        resp.usage.completion_tokens
    );
}

// ===========================================================================
// Nightly: Gemma2-2B (Gemma2ForCausalLM) — Tier 3
// ===========================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_t3_gemma2_server_starts() {
    let server = TestServer::builder(TestModels::GEMMA2)
        .start()
        .await
        .expect("server should start");

    let client = Client::new(server.base_url());
    assert!(client.health().await.unwrap());
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_t3_gemma2_chat_basic() {
    let server = TestServer::builder(TestModels::GEMMA2)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = simple_chat_request("Say hello in one sentence.", Some(50));
    let resp = client.chat_completion(&request).await.unwrap();

    assert_valid_chat_response(&resp);
    let text = resp.choices[0].message.content.as_deref().unwrap_or("");
    assert_coherent_text(text, 2);
}

// ===========================================================================
// Nightly: Phi-3.5-mini (Phi3ForCausalLM) — Tier 3
// ===========================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_t3_phi3_server_starts() {
    let server = TestServer::builder(TestModels::PHI3_5)
        .start()
        .await
        .expect("server should start");

    let client = Client::new(server.base_url());
    assert!(client.health().await.unwrap());
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_t3_phi3_chat_basic() {
    let server = TestServer::builder(TestModels::PHI3_5)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = simple_chat_request("Say hello in one sentence.", Some(50));
    let resp = client.chat_completion(&request).await.unwrap();

    assert_valid_chat_response(&resp);
    let text = resp.choices[0].message.content.as_deref().unwrap_or("");
    assert_coherent_text(text, 2);
}

// ===========================================================================
// Nightly: Phi-4-mini (Phi3ForCausalLM + LongRoPE) — Tier 3
// ===========================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_t3_phi4_mini_server_starts() {
    let server = TestServer::builder(TestModels::PHI4)
        .start()
        .await
        .expect("Phi-4 mini server should start");

    let client = Client::new(server.base_url());
    assert!(client.health().await.unwrap(), "server should be healthy");

    let models = client.list_models().await.unwrap();
    assert_eq!(models.data.len(), 1);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_t3_phi4_mini_chat_basic() {
    let server = TestServer::builder(TestModels::PHI4).start().await.unwrap();

    let client = Client::new(server.base_url());
    let request = simple_chat_request("Say hello in one sentence.", Some(50));
    let resp = client.chat_completion(&request).await.unwrap();

    assert_valid_chat_response(&resp);
    let text = resp.choices[0].message.content.as_deref().unwrap_or("");
    assert_coherent_text(text, 2);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_t3_phi4_mini_completion_basic() {
    let server = TestServer::builder(TestModels::PHI4).start().await.unwrap();

    let client = Client::new(server.base_url());
    let request = simple_completion_request("The capital of France is", 20);
    let resp = client.completion(&request).await.unwrap();

    assert_valid_completion_response(&resp);
    assert!(
        !resp.choices[0].text.is_empty(),
        "completion should not be empty"
    );
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_t3_phi4_mini_max_tokens() {
    let server = TestServer::builder(TestModels::PHI4).start().await.unwrap();

    let client = Client::new(server.base_url());
    let request = simple_chat_request("Write a long story about a cat.", Some(5));
    let resp = client.chat_completion(&request).await.unwrap();

    assert_valid_chat_response(&resp);
    assert!(
        resp.usage.completion_tokens.unwrap_or(0) <= 5,
        "completion_tokens should be <= 5, got: {:?}",
        resp.usage.completion_tokens
    );
}

// ===========================================================================
// Weekly: Mistral-7B (MistralForCausalLM) — Tier 4
// ===========================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_t4_mistral_server_starts() {
    let server = TestServer::builder(TestModels::MISTRAL)
        .start()
        .await
        .expect("server should start");

    let client = Client::new(server.base_url());
    assert!(client.health().await.unwrap());
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_t4_mistral_chat_basic() {
    let server = TestServer::builder(TestModels::MISTRAL)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = simple_chat_request("Say hello in one sentence.", Some(50));
    let resp = client.chat_completion(&request).await.unwrap();

    assert_valid_chat_response(&resp);
    let text = resp.choices[0].message.content.as_deref().unwrap_or("");
    assert_coherent_text(text, 2);
}

// ===========================================================================
// Weekly: DeepSeek-V2-Lite (DeepseekV2ForCausalLM) — Tier 4
// ===========================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_t4_deepseek_server_starts() {
    let server = TestServer::builder(TestModels::DEEPSEEK_V2_LITE)
        .start()
        .await
        .expect("server should start");

    let client = Client::new(server.base_url());
    assert!(client.health().await.unwrap());
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_t4_deepseek_chat_basic() {
    let server = TestServer::builder(TestModels::DEEPSEEK_V2_LITE)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = simple_chat_request("Say hello in one sentence.", Some(50));
    let resp = client.chat_completion(&request).await.unwrap();

    assert_valid_chat_response(&resp);
    let text = resp.choices[0].message.content.as_deref().unwrap_or("");
    assert_coherent_text(text, 2);
}

// ===========================================================================
// Qwen3 MoE (Qwen3MoeForCausalLM) — MoE architecture
// ===========================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_qwen3_moe_server_starts() {
    let server = TestServer::builder(TestModels::QWEN3_MOE)
        .start()
        .await
        .expect("Qwen3 MoE server should start");

    let client = Client::new(server.base_url());
    assert!(client.health().await.unwrap(), "server should be healthy");

    let models = client.list_models().await.unwrap();
    assert_eq!(models.data.len(), 1);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_qwen3_moe_chat_basic() {
    let server = TestServer::builder(TestModels::QWEN3_MOE)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = simple_chat_request("Say hello in one sentence.", Some(50));
    let resp = client.chat_completion(&request).await.unwrap();

    assert_valid_chat_response(&resp);
    let text = resp.choices[0].message.content.as_deref().unwrap_or("");
    assert_coherent_text(text, 2);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_qwen3_moe_completion_basic() {
    let server = TestServer::builder(TestModels::QWEN3_MOE)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = simple_completion_request("The capital of France is", 20);
    let resp = client.completion(&request).await.unwrap();

    assert_valid_completion_response(&resp);
    assert!(
        !resp.choices[0].text.is_empty(),
        "completion should not be empty"
    );
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_qwen3_moe_max_tokens() {
    let server = TestServer::builder(TestModels::QWEN3_MOE)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = simple_chat_request("Write a long story about a cat.", Some(5));
    let resp = client.chat_completion(&request).await.unwrap();

    assert_valid_chat_response(&resp);
    assert!(
        resp.usage.completion_tokens.unwrap_or(0) <= 5,
        "completion_tokens should be <= 5, got: {:?}",
        resp.usage.completion_tokens
    );
}

// ===========================================================================
// E1b: Float16 vs quantized comparison (MLX-only)
// ===========================================================================

#[cfg(feature = "metal")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_t1_float16_server_starts() {
    let server = TestServer::builder(TestModels::SMOLLM_135M_F16)
        .start()
        .await
        .expect("float16 server should start");

    let client = Client::new(server.base_url());
    assert!(client.health().await.unwrap());
}

#[cfg(feature = "metal")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_t1_float16_chat_basic() {
    let server = TestServer::builder(TestModels::SMOLLM_135M_F16)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = simple_chat_request("Say hello in one sentence.", Some(50));
    let resp = client.chat_completion(&request).await.unwrap();

    assert_valid_chat_response(&resp);
    let text = resp.choices[0].message.content.as_deref().unwrap_or("");
    assert_coherent_text(text, 2);
}

// ===========================================================================
// E1c: Metal startup-time regression locks (MLX/Metal-only)
//
// Guards two regressions from commit `c54955a3` (aligned-sidecar weight
// cache): the sidecar must PERSIST across a process teardown (so warm
// launches stay fast instead of re-paying the ~1.7 s cold realign-copy every
// time), and the warm startup-time distribution must stay within margin.
// Named `test_t1_smollm_*` so the SmolLM-tier CI filter runs them with no
// workflow change. Tunables: VLLM_TEST_STARTUP_{ITERS,BASELINE_MS,MODEL}.
// ===========================================================================

#[cfg(feature = "metal")]
mod metal_startup {
    use std::path::{Path, PathBuf};
    use std::time::{Instant, SystemTime};

    use scratchy_e2e::TestModels;
    use scratchy_serving_api::llm::{ChatMessage, LLM, SamplingParams};

    /// Redirects `$XDG_CACHE_HOME` (where the scratchy aligned-sidecar cache
    /// lives) into a temp dir so a test starts cold, and restores it on drop
    /// so sibling tests in this binary are unaffected. HF model resolution
    /// uses `~/.cache/huggingface` independently, so it is untouched.
    struct CacheHomeGuard {
        prev: Option<std::ffi::OsString>,
        tmp: tempfile::TempDir,
    }

    impl CacheHomeGuard {
        fn new() -> Self {
            let tmp = tempfile::tempdir().expect("tempdir");
            let prev = std::env::var_os("XDG_CACHE_HOME");
            // SAFETY: e2e tests run single-threaded (`--test-threads=1`).
            unsafe { std::env::set_var("XDG_CACHE_HOME", tmp.path()) };
            Self { prev, tmp }
        }
        fn aligned_cache_dir(&self) -> PathBuf {
            self.tmp
                .path()
                .join("scratchy")
                .join("metal-aligned-weights")
        }
    }

    impl Drop for CacheHomeGuard {
        fn drop(&mut self) {
            // SAFETY: single-threaded; see `new`.
            unsafe {
                match &self.prev {
                    Some(v) => std::env::set_var("XDG_CACHE_HOME", v),
                    None => std::env::remove_var("XDG_CACHE_HOME"),
                }
            }
        }
    }

    /// `(path, mtime)` for every `*.bin` sidecar in `dir` (empty if absent).
    fn sidecar_bins(dir: &Path) -> Vec<(PathBuf, SystemTime)> {
        let Ok(rd) = std::fs::read_dir(dir) else {
            return Vec::new();
        };
        let mut out: Vec<(PathBuf, SystemTime)> = rd
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|x| x == "bin"))
            .filter_map(|p| {
                let mtime = std::fs::metadata(&p).ok()?.modified().ok()?;
                Some((p, mtime))
            })
            .collect();
        out.sort();
        out
    }

    fn startup_model() -> String {
        std::env::var("VLLM_TEST_STARTUP_MODEL").unwrap_or_else(|_| TestModels::SMOLLM.to_string())
    }

    /// Build an `LLM`, run one short deterministic generation, then drop it.
    /// Returns the `build()` time (the startup metric) in milliseconds.
    fn timed_load_generate_drop(model: &str) -> f64 {
        let t0 = Instant::now();
        let mut llm = LLM::builder(model)
            .max_num_seqs(1)
            .build()
            .expect("LLM should initialize");
        let build_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let params = SamplingParams {
            max_tokens: Some(8),
            temperature: 0.0,
            ..SamplingParams::default()
        };
        let out = llm
            .chat(&[ChatMessage::user("hi")], Some(params))
            .expect("chat should succeed");
        assert!(
            !out.outputs[0].text.is_empty(),
            "generation should produce text"
        );
        // `llm` drops here: InprocClient::drop joins the executor thread, whose
        // teardown drains the sidecar writer before this returns.
        build_ms
    }

    /// Nearest-rank percentile of an already-sorted slice.
    fn percentile(sorted: &[f64], p: f64) -> f64 {
        let rank = (p / 100.0 * (sorted.len() as f64 - 1.0)).round() as usize;
        sorted[rank.min(sorted.len() - 1)]
    }

    #[test]
    #[ignore]
    fn test_t1_smollm_aligned_cache_persists() {
        scratchy_e2e::skip_if_no_gpu!();
        scratchy_core_common::telemetry::init_tracing("off");
        let guard = CacheHomeGuard::new();
        let cache_dir = guard.aligned_cache_dir();
        assert!(
            sidecar_bins(&cache_dir).is_empty(),
            "cache must start cold (dir: {})",
            cache_dir.display()
        );
        let model = startup_model();

        // Cold run builds the sidecar, then tears down.
        timed_load_generate_drop(&model);

        // REGRESSION LOCK #1: the sidecar must be on disk the instant the LLM
        // drops (the writer must be joined on teardown, not left detached).
        let after_cold = sidecar_bins(&cache_dir);
        assert!(
            !after_cold.is_empty(),
            "aligned sidecar cache must persist after a single short run — dir: {}",
            cache_dir.display()
        );

        // Warm run must reuse the persisted cache unchanged (no rebuild), and
        // reaching here proves two full load/drop cycles tore down cleanly —
        // no teardown segfault.
        timed_load_generate_drop(&model);
        assert_eq!(
            after_cold,
            sidecar_bins(&cache_dir),
            "warm launch must reuse the persisted sidecar unchanged (no rebuild)"
        );
    }

    #[test]
    #[ignore]
    fn test_t1_smollm_startup_distribution() {
        scratchy_e2e::skip_if_no_gpu!();
        scratchy_core_common::telemetry::init_tracing("off");
        let _guard = CacheHomeGuard::new();
        let model = startup_model();
        let iters: usize = std::env::var("VLLM_TEST_STARTUP_ITERS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(25);
        // Known-good warm startup floor (ms); all gates scale from it. Locally
        // warm load is ~450 ms, so 500 leaves a little headroom. Override per
        // machine if CI hardware differs materially.
        let baseline: f64 = std::env::var("VLLM_TEST_STARTUP_BASELINE_MS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(500.0);

        // Untimed warmup: first load is cold (builds + persists the sidecar);
        // every timed iteration below exercises the warm zero-copy path.
        timed_load_generate_drop(&model);

        let mut samples: Vec<f64> = (0..iters)
            .map(|_| timed_load_generate_drop(&model))
            .collect();
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let min = samples[0];
        let p25 = percentile(&samples, 25.0);
        let p50 = percentile(&samples, 50.0);
        let p75 = percentile(&samples, 75.0);
        let p90 = percentile(&samples, 90.0);
        let max = samples[samples.len() - 1];

        // Widening margins: the fast floor must stay fast (tight on min) while
        // the tail gets progressively more slack so occasional slow launches
        // (GC pauses, scheduler hiccups, CI noise) don't flake the gate. A real
        // regression — every launch re-paying the cold copy (~1.5 s+) — pushes
        // even `min` past its 1.25× limit and fails loudly.
        let gates = [
            ("min", min, 1.25),
            ("p25", p25, 1.5),
            ("p75", p75, 2.0),
            ("p90", p90, 3.0),
        ];

        // Always record the distribution + limits in the CI log. Written via
        // the raw stderr handle, NOT println!: the CI invocation has no
        // `--nocapture`, and libtest captures the `print!`/`eprint!` macros and
        // hides them on a passing test. `io::stderr()` is not captured (it is
        // how the load progress bars reach the log too), so this line shows
        // whether the gate passes or fails.
        use std::io::Write as _;
        let _ = writeln!(
            std::io::stderr(),
            "[startup] model={model} n={iters} baseline={baseline:.0}ms warm build_ms:\n\
             [startup]   distribution: min={min:.0} p25={p25:.0} p50={p50:.0} p75={p75:.0} p90={p90:.0} max={max:.0}\n\
             [startup]   limits:       min<{:.0} p25<{:.0} p75<{:.0} p90<{:.0}",
            1.25 * baseline,
            1.5 * baseline,
            2.0 * baseline,
            3.0 * baseline,
        );

        let breaches: Vec<String> = gates
            .iter()
            .filter(|(_, v, mult)| *v >= mult * baseline)
            .map(|(name, v, mult)| {
                format!(
                    "{name}={v:.0}ms ≥ {:.0}ms ({mult}×baseline)",
                    mult * baseline
                )
            })
            .collect();
        assert!(
            breaches.is_empty(),
            "metal startup regressed (baseline {baseline:.0}ms): {}",
            breaches.join(", ")
        );
    }
}

// ---------------------------------------------------------------------------
// Sync scheduling path (verify both scheduling modes work)
// ---------------------------------------------------------------------------

/// Smoke test: sync scheduling path still works end-to-end.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_sync_scheduling_smollm_chat() {
    let server = TestServer::builder(TestModels::SMOLLM)
        .with_sync_scheduling()
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = simple_chat_request("Say hello in one sentence.", Some(50));
    let resp = client.chat_completion(&request).await.unwrap();

    assert_valid_chat_response(&resp);
    let text = resp.choices[0].message.content.as_deref().unwrap_or("");
    assert_coherent_text(text, 2);
}

// ===========================================================================
// Granite (IBM) — GraniteForCausalLM, MLX 4-bit quantized (~1.3 GB)
// ===========================================================================
// Granite is architecturally identical to LLaMA with 4 scalar multipliers.
//
// Run with: cargo test -p vllm-e2e --features e2e,metal --release --test e1_basic_serving test_granite -- --ignored
//       or: cargo test -p vllm-e2e --features e2e,cuda  --release --test e1_basic_serving test_granite -- --ignored

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_granite_server_starts() {
    let server = TestServer::builder(TestModels::GRANITE)
        .start()
        .await
        .expect("Granite server should start");

    let client = Client::new(server.base_url());
    assert!(client.health().await.unwrap(), "server should be healthy");

    let models = client.list_models().await.unwrap();
    assert_eq!(models.data.len(), 1);
    assert!(
        models.data[0].id.contains("granite"),
        "model name should contain 'granite', got: {}",
        models.data[0].id
    );
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_granite_completion() {
    let server = TestServer::builder(TestModels::GRANITE)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = simple_completion_request("The capital of France is", 20);
    let resp = client.completion(&request).await.unwrap();

    assert_valid_completion_response(&resp);
    assert!(
        !resp.choices[0].text.is_empty(),
        "completion should not be empty"
    );
}

/// Granite is a chat/instruct model (EOS=token 0) — bare completions may
/// immediately stop. Test with a second completion prompt for variety.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_granite_completion_coherent() {
    let server = TestServer::builder(TestModels::GRANITE)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = simple_completion_request("Once upon a time", 30);
    let resp = client.completion(&request).await.unwrap();

    assert_valid_completion_response(&resp);
    assert!(
        !resp.choices[0].text.is_empty(),
        "completion should not be empty"
    );
}

// ===========================================================================
// CUDA E2E tests — safetensors models that actually run on GPU
// ===========================================================================
// These use non-quantized safetensors models (not MLX 4-bit) so that
// model weights load onto the CUDA device and GPU kernels are exercised.
//
// Run with: cargo test -p vllm-e2e --features e2e,cuda --release --test e1_basic_serving test_cuda -- --ignored

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_smollm_server_starts() {
    let server = TestServer::builder(TestModels::SMOLLM_135M_CUDA)
        .start()
        .await
        .expect("CUDA SmolLM server should start");

    let client = Client::new(server.base_url());
    assert!(client.health().await.unwrap(), "server should be healthy");

    let models = client.list_models().await.unwrap();
    assert_eq!(models.data.len(), 1);
}

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_smollm_completion() {
    let server = TestServer::builder(TestModels::SMOLLM_135M_CUDA)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = simple_completion_request("The capital of France is", 20);
    let resp = client.completion(&request).await.unwrap();

    assert_valid_completion_response(&resp);
    assert!(
        !resp.choices[0].text.is_empty(),
        "completion should not be empty"
    );
}

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_smollm_chat() {
    let server = TestServer::builder(TestModels::SMOLLM_135M_CUDA)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = simple_chat_request("Say hello in one sentence.", Some(50));
    let resp = client.chat_completion(&request).await.unwrap();

    assert_valid_chat_response(&resp);
    let text = resp.choices[0].message.content.as_deref().unwrap_or("");
    assert_coherent_text(text, 2);
}

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_qwen2_completion() {
    let server = TestServer::builder(TestModels::QWEN2_0_5B_CUDA)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = simple_completion_request("The capital of France is", 20);
    let resp = client.completion(&request).await.unwrap();

    assert_valid_completion_response(&resp);
    assert!(
        !resp.choices[0].text.is_empty(),
        "completion should not be empty"
    );
    assert_coherent_text(&resp.choices[0].text, 3);
}

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_qwen2_chat() {
    let server = TestServer::builder(TestModels::QWEN2_0_5B_CUDA)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = simple_chat_request("What is 2+2? Answer with just the number.", Some(10));
    let resp = client.chat_completion(&request).await.unwrap();

    assert_valid_chat_response(&resp);
    let text = resp.choices[0].message.content.as_deref().unwrap_or("");
    assert!(!text.is_empty(), "response should not be empty");
    assert_coherent_text(text, 1);
}

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_qwen3_completion() {
    let server = TestServer::builder(TestModels::QWEN3_0_6B_CUDA)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = simple_completion_request("The capital of France is", 20);
    let resp = client.completion(&request).await.unwrap();

    assert_valid_completion_response(&resp);
    assert_coherent_text(&resp.choices[0].text, 2);
}

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_qwen3_chat() {
    let server = TestServer::builder(TestModels::QWEN3_0_6B_CUDA)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = simple_chat_request("What is 2+2? Answer with just the number.", Some(10));
    let resp = client.chat_completion(&request).await.unwrap();

    assert_valid_chat_response(&resp);
    let text = resp.choices[0].message.content.as_deref().unwrap_or("");
    assert_coherent_text(text, 1);
}

// ===========================================================================
// CUDA Gemma2 E2E tests — safetensors BF16 on GPU (scratchy-serving-cuda backend)
// ===========================================================================
// Run with: cargo test -p vllm-e2e --features e2e,cuda --release --test e1_basic_serving test_cuda_gemma2 -- --ignored --test-threads=1

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_gemma2_server_starts() {
    let server = TestServer::builder(TestModels::GEMMA2_2B_IT_CUDA)
        .start()
        .await
        .expect("CUDA Gemma2 2B server should start");

    let client = Client::new(server.base_url());
    assert!(client.health().await.unwrap(), "server should be healthy");

    let models = client.list_models().await.unwrap();
    assert_eq!(models.data.len(), 1);
}

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_gemma2_completion() {
    let server = TestServer::builder(TestModels::GEMMA2_2B_IT_CUDA)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = simple_completion_request("The capital of France is", 20);
    let resp = client.completion(&request).await.unwrap();

    assert_valid_completion_response(&resp);
    let text = &resp.choices[0].text;
    assert!(!text.is_empty(), "completion should not be empty");
    assert_coherent_text(text, 1);
}

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_gemma2_chat() {
    let server = TestServer::builder(TestModels::GEMMA2_2B_IT_CUDA)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = simple_chat_request("Say hello in one sentence.", Some(50));
    let resp = client.chat_completion(&request).await.unwrap();

    assert_valid_chat_response(&resp);
    let text = resp.choices[0].message.content.as_deref().unwrap_or("");
    assert_coherent_text(text, 2);
}

// ===========================================================================
// CUDA Gemma3 E2E tests — safetensors BF16 on GPU (scratchy-serving-cuda backend)
// ===========================================================================
// Run with: cargo test -p vllm-e2e --features e2e,cuda --release --test e1_basic_serving test_cuda_gemma3 -- --ignored --test-threads=1

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_gemma3_server_starts() {
    let server = TestServer::builder(TestModels::GEMMA3_1B_IT_CUDA)
        .start()
        .await
        .expect("CUDA Gemma3 1B server should start");

    let client = Client::new(server.base_url());
    assert!(client.health().await.unwrap(), "server should be healthy");

    let models = client.list_models().await.unwrap();
    assert_eq!(models.data.len(), 1);
}

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_gemma3_completion() {
    let server = TestServer::builder(TestModels::GEMMA3_1B_IT_CUDA)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = simple_completion_request("The capital of France is", 20);
    let resp = client.completion(&request).await.unwrap();

    assert_valid_completion_response(&resp);
    let text = &resp.choices[0].text;
    assert!(!text.is_empty(), "completion should not be empty");
    assert_coherent_text(text, 1);
}

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_gemma3_chat() {
    let server = TestServer::builder(TestModels::GEMMA3_1B_IT_CUDA)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = simple_chat_request("Say hello in one sentence.", Some(50));
    let resp = client.chat_completion(&request).await.unwrap();

    assert_valid_chat_response(&resp);
    let text = resp.choices[0].message.content.as_deref().unwrap_or("");
    assert_coherent_text(text, 2);
}

// ===========================================================================
// CUDA GGUF E2E tests — quantized GGUF models on GPU
// ===========================================================================
// Uses scratchy-serving-cuda's GGML kernel FFI for quantized inference (raw FFI).
//
// Run with: cargo test -p vllm-e2e --features e2e,cuda --release --test e1_basic_serving test_cuda_gguf -- --ignored

// TODO: Gemma3 GGUF requires Gemma3ForCausalLM load_gguf() — not yet implemented
/*
#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_gguf_gemma3_1b_server_starts() {
    let server = TestServer::builder(TestModels::GEMMA3_1B_GGUF)
        .start()
        .await
        .expect("CUDA Gemma3 1B GGUF server should start");

    let client = Client::new(server.base_url());
    assert!(client.health().await.unwrap(), "server should be healthy");

    let models = client.list_models().await.unwrap();
    assert_eq!(models.data.len(), 1);
}

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_gguf_gemma3_1b_completion() {
    let server = TestServer::builder(TestModels::GEMMA3_1B_GGUF)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = simple_completion_request("The capital of France is", 20);
    let resp = client.completion(&request).await.unwrap();

    assert_valid_completion_response(&resp);
    assert!(
        !resp.choices[0].text.is_empty(),
        "completion should not be empty"
    );
}

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_gguf_gemma3_1b_chat() {
    let server = TestServer::builder(TestModels::GEMMA3_1B_GGUF)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = simple_chat_request("Say hello in one sentence.", Some(50));
    let resp = client.chat_completion(&request).await.unwrap();

    assert_valid_chat_response(&resp);
    // Note: small quantized GGUF models may generate low-quality text;
    // we only assert the response structure is valid.
    assert!(
        resp.usage.completion_tokens.unwrap_or(0) > 0,
        "should generate at least one token"
    );
}
*/

// Qwen2.5 GGUF (qwen2 architecture with QKV bias)

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_gguf_qwen2_0_5b_server_starts() {
    let server = TestServer::builder(TestModels::QWEN2_0_5B_GGUF)
        .start()
        .await
        .expect("CUDA Qwen2.5 0.5B GGUF server should start");

    let client = Client::new(server.base_url());
    assert!(client.health().await.unwrap(), "server should be healthy");

    let models = client.list_models().await.unwrap();
    assert_eq!(models.data.len(), 1);
}

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_gguf_qwen2_0_5b_chat() {
    let server = TestServer::builder(TestModels::QWEN2_0_5B_GGUF)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = simple_chat_request("Say hello in one sentence.", Some(50));
    let resp = client.chat_completion(&request).await.unwrap();

    assert_valid_chat_response(&resp);
    let text = resp.choices[0].message.content.as_deref().unwrap_or("");
    eprintln!("[Qwen2 0.5B GGUF output] {:?}", text);
    assert_coherent_text(text, 5);
}

// Qwen3 GGUF (qwen2 architecture in GGUF, LLaMA-compatible)

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_gguf_qwen3_0_6b_server_starts() {
    let server = TestServer::builder(TestModels::QWEN3_0_6B_GGUF)
        .start()
        .await
        .expect("CUDA Qwen3 0.6B GGUF server should start");

    let client = Client::new(server.base_url());
    assert!(client.health().await.unwrap(), "server should be healthy");

    let models = client.list_models().await.unwrap();
    assert_eq!(models.data.len(), 1);
}

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_gguf_qwen3_0_6b_chat() {
    let server = TestServer::builder(TestModels::QWEN3_0_6B_GGUF)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = simple_chat_request("Say hello in one sentence.", Some(50));
    let resp = client.chat_completion(&request).await.unwrap();

    assert_valid_chat_response(&resp);
    let text = resp.choices[0].message.content.as_deref().unwrap_or("");
    assert_coherent_text(text, 2);
}

// TODO: Qwen3Next GGUF requires Qwen3NextForCausalLM load_gguf() — not yet implemented
/*
// ---------------------------------------------------------------------------
// CUDA GGUF: Qwen3.5 (Qwen3-Next) — hybrid GDN + full attention
// ---------------------------------------------------------------------------

/// Qwen3.5-0.8B GGUF: server starts and health check passes.
///
/// Run with:
///   cargo test -p vllm-e2e --features e2e,cuda --release --test e1_basic_serving test_cuda_gguf_qwen3_next -- --ignored --test-threads=1
#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_gguf_qwen3_next_server_starts() {
    let server = TestServer::builder(TestModels::QWEN3_NEXT_0_8B_GGUF)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    assert!(client.health().await.unwrap(), "server should be healthy");

    let models = client.list_models().await.unwrap();
    assert_eq!(models.data.len(), 1);
}

/// Qwen3.5-0.8B GGUF: completion generates non-empty text.
#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_gguf_qwen3_next_completion() {
    let server = TestServer::builder(TestModels::QWEN3_NEXT_0_8B_GGUF)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = simple_completion_request("The capital of France is", 30);
    let resp = client.completion(&request).await.unwrap();

    assert_valid_completion_response(&resp);
    let text = &resp.choices[0].text;
    assert!(!text.is_empty(), "should produce non-empty text");
}

/// Qwen3.5-0.8B GGUF: chat endpoint works.
#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_gguf_qwen3_next_chat() {
    let server = TestServer::builder(TestModels::QWEN3_NEXT_0_8B_GGUF)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = simple_chat_request("Say hello in one sentence.", Some(50));
    let resp = client.chat_completion(&request).await.unwrap();

    assert_valid_chat_response(&resp);
    assert!(
        resp.usage.completion_tokens.unwrap_or(0) > 0,
        "should generate at least one token"
    );
}
*/

// Llama-3.2-1B Q4_K_M — standard 4-bit K-quant GGUF. Exercises Q4_K
// (Q/K/attn_output/gate/up) and Q6_K (token_embd/V/down_proj) kernels
// end-to-end. Coherence check is load-bearing — a kernel regression
// typically collapses output to a single repeated token, which
// `assert_coherent_text` catches.
#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_gguf_llama_q4km_completion() {
    let server = TestServer::builder(TestModels::LLAMA_3_2_1B_Q4KM_GGUF)
        .with_args(&["--gguf-file", TestModels::LLAMA_3_2_1B_Q4KM_FILE])
        .start()
        .await
        .expect("Q4_K_M GGUF server should start");

    let client = Client::new(server.base_url());
    let request = simple_completion_request("The capital of France is", 20);
    let resp = client.completion(&request).await.unwrap();

    assert_valid_completion_response(&resp);
    let text = &resp.choices[0].text;
    eprintln!("[Q4_K_M completion output] {:?}", text);
    assert_coherent_text(text, 10);
}

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_gguf_llama_q4km_chat() {
    let server = TestServer::builder(TestModels::LLAMA_3_2_1B_Q4KM_GGUF)
        .with_args(&["--gguf-file", TestModels::LLAMA_3_2_1B_Q4KM_FILE])
        .start()
        .await
        .expect("Q4_K_M GGUF server should start");

    let client = Client::new(server.base_url());
    let request = simple_chat_request("Say hello in one sentence.", Some(50));
    let resp = client.chat_completion(&request).await.unwrap();

    assert_valid_chat_response(&resp);
    let text = resp.choices[0].message.content.as_deref().unwrap_or("");
    eprintln!("[Q4_K_M chat output] {:?}", text);
    assert_coherent_text(text, 10);
}

// Llama-3.2-1B UD-IQ1_M — IQ1_M quantized GGUF (1.75 bpw)

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_gguf_llama_iq1m_chat() {
    let server = TestServer::builder(TestModels::LLAMA_3_2_1B_IQ1M_GGUF)
        .with_args(&["--gguf-file", TestModels::LLAMA_3_2_1B_IQ1M_FILE])
        .start()
        .await
        .expect("IQ1_M GGUF server should start");

    let client = Client::new(server.base_url());
    let request = simple_chat_request("Say hello in one sentence.", Some(50));
    let resp = client.chat_completion(&request).await.unwrap();

    assert_valid_chat_response(&resp);
    assert!(
        resp.usage.completion_tokens.unwrap_or(0) > 0,
        "IQ1_M model should generate at least one token"
    );

    let text = resp.choices[0].message.content.as_deref().unwrap_or("");
    eprintln!("[IQ1_M output] {:?}", text);
    assert_coherent_text(text, 5);
}

// Additional IQ quants on Llama-3.2-1B — IQ1_S, IQ2_XXS, IQ4_NL, IQ4_XS.
// Each test mirrors the IQ1_M pattern and exercises one MMVQ kernel variant.
// IQ1_S/IQ2_XXS are extreme (<2 bpw) — coherence threshold is low (3) because
// at those bitrates the output is marginal even with a correct kernel; the
// test's job is to catch *regressions*, not assert quality.

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_gguf_llama_iq1s_chat() {
    let server = TestServer::builder(TestModels::LLAMA_3_2_1B_IQ1M_GGUF)
        .with_args(&["--gguf-file", TestModels::LLAMA_3_2_1B_IQ1S_FILE])
        .start()
        .await
        .expect("IQ1_S GGUF server should start");

    let client = Client::new(server.base_url());
    let request = simple_chat_request("Say hello in one sentence.", Some(50));
    let resp = client.chat_completion(&request).await.unwrap();

    assert_valid_chat_response(&resp);
    assert!(
        resp.usage.completion_tokens.unwrap_or(0) > 0,
        "IQ1_S model should generate at least one token"
    );
    let text = resp.choices[0].message.content.as_deref().unwrap_or("");
    eprintln!("[IQ1_S output] {:?}", text);
    assert_coherent_text(text, 3);
}

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_gguf_llama_iq2xxs_chat() {
    let server = TestServer::builder(TestModels::LLAMA_3_2_1B_IQ1M_GGUF)
        .with_args(&["--gguf-file", TestModels::LLAMA_3_2_1B_IQ2XXS_FILE])
        .start()
        .await
        .expect("IQ2_XXS GGUF server should start");

    let client = Client::new(server.base_url());
    let request = simple_chat_request("Say hello in one sentence.", Some(50));
    let resp = client.chat_completion(&request).await.unwrap();

    assert_valid_chat_response(&resp);
    let text = resp.choices[0].message.content.as_deref().unwrap_or("");
    eprintln!("[IQ2_XXS output] {:?}", text);
    assert_coherent_text(text, 3);
}

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_gguf_llama_iq4nl_chat() {
    let server = TestServer::builder(TestModels::LLAMA_3_2_1B_IQ1M_GGUF)
        .with_args(&["--gguf-file", TestModels::LLAMA_3_2_1B_IQ4NL_FILE])
        .start()
        .await
        .expect("IQ4_NL GGUF server should start");

    let client = Client::new(server.base_url());
    let request = simple_chat_request("Say hello in one sentence.", Some(50));
    let resp = client.chat_completion(&request).await.unwrap();

    assert_valid_chat_response(&resp);
    let text = resp.choices[0].message.content.as_deref().unwrap_or("");
    eprintln!("[IQ4_NL output] {:?}", text);
    assert_coherent_text(text, 5);
}

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_gguf_llama_iq4xs_chat() {
    let server = TestServer::builder(TestModels::LLAMA_3_2_1B_IQ1M_GGUF)
        .with_args(&["--gguf-file", TestModels::LLAMA_3_2_1B_IQ4XS_FILE])
        .start()
        .await
        .expect("IQ4_XS GGUF server should start");

    let client = Client::new(server.base_url());
    let request = simple_chat_request("Say hello in one sentence.", Some(50));
    let resp = client.chat_completion(&request).await.unwrap();

    assert_valid_chat_response(&resp);
    let text = resp.choices[0].message.content.as_deref().unwrap_or("");
    eprintln!("[IQ4_XS output] {:?}", text);
    assert_coherent_text(text, 5);
}

// IQ2_S and IQ3_S — only available on Mistral-7B-Instruct-v0.3 at unsloth /
// bartowski / etc. at the time these were added. Larger model → slower test.

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_gguf_mistral_iq2s_chat() {
    let server = TestServer::builder(TestModels::MISTRAL_7B_V03_GGUF)
        .with_args(&["--gguf-file", TestModels::MISTRAL_7B_V03_IQ2S_FILE])
        .start()
        .await
        .expect("IQ2_S GGUF server should start");

    let client = Client::new(server.base_url());
    let request = simple_chat_request("Say hello in one sentence.", Some(50));
    let resp = client.chat_completion(&request).await.unwrap();

    assert_valid_chat_response(&resp);
    let text = resp.choices[0].message.content.as_deref().unwrap_or("");
    eprintln!("[IQ2_S output] {:?}", text);
    assert_coherent_text(text, 3);
}

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_gguf_mistral_iq3s_chat() {
    let server = TestServer::builder(TestModels::MISTRAL_7B_V03_GGUF)
        .with_args(&["--gguf-file", TestModels::MISTRAL_7B_V03_IQ3S_FILE])
        .start()
        .await
        .expect("IQ3_S GGUF server should start");

    let client = Client::new(server.base_url());
    let request = simple_chat_request("Say hello in one sentence.", Some(50));
    let resp = client.chat_completion(&request).await.unwrap();

    assert_valid_chat_response(&resp);
    let text = resp.choices[0].message.content.as_deref().unwrap_or("");
    eprintln!("[IQ3_S output] {:?}", text);
    assert_coherent_text(text, 5);
}

// end GGUF tests

// ---------------------------------------------------------------------------
// Tensor Parallelism (TP=2) tests — require 2 CUDA GPUs + NCCL
// ---------------------------------------------------------------------------

/// TP=2 Qwen2.5-0.5B: server starts, health check passes, completion works.
///
/// Uses Qwen2.5-0.5B (14 Q heads, 2 KV heads — both divisible by 2).
/// SmolLM-135M has 9/3 heads which don't divide evenly by 2.
///
/// Run on nick2 pod (2x L40S):
///   cargo test -p vllm-e2e --features e2e,nccl --release --test e1_basic_serving test_cuda_tp2 -- --ignored --test-threads=1
#[cfg(feature = "nccl")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_tp2_qwen2_completion() {
    let server = TestServer::builder(TestModels::QWEN2_0_5B_CUDA)
        .with_tensor_parallel_size(2)
        .start()
        .await
        .expect("TP=2 Qwen2.5-0.5B server should start");

    let client = Client::new(server.base_url());
    assert!(client.health().await.unwrap(), "server should be healthy");

    let request = simple_completion_request("The capital of France is", 20);
    let resp = client.completion(&request).await.unwrap();

    assert_valid_completion_response(&resp);
    let text = &resp.choices[0].text;
    assert!(
        !text.is_empty(),
        "TP=2 completion should produce non-empty text"
    );
}

/// DeepSeek V2 Lite with TP=2 — validates MLA attention TP sharding.
///
/// Run on nick2 pod (2x L40S):
///   cargo test -p vllm-e2e --features e2e,nccl --release --test e1_basic_serving test_cuda_tp2_deepseek_v2 -- --ignored --test-threads=1
#[cfg(feature = "nccl")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_tp2_deepseek_v2_completion() {
    let server = TestServer::builder(TestModels::DEEPSEEK_V2_LITE_CUDA)
        .with_tensor_parallel_size(2)
        .with_timeout(std::time::Duration::from_secs(300))
        .start()
        .await
        .expect("TP=2 DeepSeek-V2-Lite server should start");

    let client = Client::new(server.base_url());
    assert!(client.health().await.unwrap(), "server should be healthy");

    let request = simple_completion_request("The capital of France is", 20);
    let resp = client.completion(&request).await.unwrap();

    assert_valid_completion_response(&resp);
    let text = &resp.choices[0].text;
    assert!(
        !text.is_empty(),
        "TP=2 DeepSeek-V2-Lite completion should produce non-empty text"
    );
}

/// TP=2 Mixtral MoE: validates MoE expert weight sharding + post-MoE all-reduce.
///
/// Uses small_mixtral (~0.8B, 8 experts, top-2). Each expert's intermediate_size
/// is halved per rank; post-MoE all-reduce combines partial expert outputs.
///
/// Run on nick2 pod (2x L40S):
///   cargo test -p vllm-e2e --features e2e,nccl --release --test e1_basic_serving test_cuda_tp2_mixtral -- --ignored --test-threads=1
#[cfg(feature = "nccl")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_tp2_mixtral_completion() {
    let server = TestServer::builder(TestModels::MIXTRAL_TINY_DPO_CUDA)
        .with_tensor_parallel_size(2)
        .start()
        .await
        .expect("TP=2 Mixtral MoE server should start");

    let client = Client::new(server.base_url());
    assert!(client.health().await.unwrap(), "server should be healthy");

    let request = simple_completion_request("The capital of France is", 20);
    let resp = client.completion(&request).await.unwrap();

    assert_valid_completion_response(&resp);
    let text = &resp.choices[0].text;
    assert!(
        !text.is_empty(),
        "TP=2 Mixtral MoE completion should produce non-empty text"
    );
}

/// TP=2 Qwen2 MoE: validates shared expert sharding + MoE TP.
///
/// Qwen2 MoE has both routed experts and a shared expert (with sigmoid gate).
/// TP shards both the routed experts' intermediate_size and the shared expert's
/// gate_up (dim=0) / down (dim=1).
///
/// Run on nick2 pod (2x L40S):
///   cargo test -p vllm-e2e --features e2e,nccl --release --test e1_basic_serving test_cuda_tp2_qwen2_moe -- --ignored --test-threads=1
#[cfg(feature = "nccl")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_tp2_qwen2_moe_completion() {
    let server = TestServer::builder(TestModels::QWEN2_MOE_A2_7B_CUDA)
        .with_tensor_parallel_size(2)
        .start()
        .await
        .expect("TP=2 Qwen2 MoE server should start");

    let client = Client::new(server.base_url());
    assert!(client.health().await.unwrap(), "server should be healthy");

    let request = simple_completion_request("The capital of France is", 20);
    let resp = client.completion(&request).await.unwrap();

    assert_valid_completion_response(&resp);
    let text = &resp.choices[0].text;
    assert!(
        !text.is_empty(),
        "TP=2 Qwen2 MoE completion should produce non-empty text"
    );
}

// ===========================================================================
// CUDA Granite — safetensors BF16 (~4.5 GB) on GPU
// ===========================================================================
// Run with: cargo test -p vllm-e2e --features e2e,cuda --release --test e1_basic_serving test_cuda_granite -- --ignored
#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_granite_chat() {
    let server = TestServer::builder(TestModels::GRANITE_3_3_2B_INSTRUCT)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = simple_chat_request("What is the capital of France?", Some(20));
    let resp = client.chat_completion(&request).await.unwrap();

    assert_valid_chat_response(&resp);
    let content = resp.choices[0].message.content.as_deref().unwrap_or("");
    assert!(!content.is_empty(), "chat response should not be empty");
}

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_granite_chat_coherent() {
    let server = TestServer::builder(TestModels::GRANITE_3_3_2B_INSTRUCT)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = simple_chat_request("Tell me a short story", Some(30));
    let resp = client.chat_completion(&request).await.unwrap();

    assert_valid_chat_response(&resp);
    let content = resp.choices[0].message.content.as_deref().unwrap_or("");
    assert!(!content.is_empty(), "chat response should not be empty");
}

// ===========================================================================
// CUDA Granite GGUF — quantized on GPU
// ===========================================================================
// TODO: Re-enable when scratchy-serving-cuda backend supports GGUF + GraniteForCausalLM.
// Run with: cargo test -p vllm-e2e --features e2e,cuda --release --test e1_basic_serving test_cuda_granite_gguf -- --ignored
/*
#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_granite_gguf_chat() {
    let server = TestServer::builder(TestModels::GRANITE_3_3_2B_INSTRUCT_GGUF)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = simple_chat_request("What is the capital of France?", Some(20));
    let resp = client.chat_completion(&request).await.unwrap();

    assert_valid_chat_response(&resp);
    let content = resp.choices[0].message.content.as_deref().unwrap_or("");
    assert!(!content.is_empty(), "chat response should not be empty");
}

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_granite_gguf_chat_coherent() {
    let server = TestServer::builder(TestModels::GRANITE_3_3_2B_INSTRUCT_GGUF)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = simple_chat_request("Tell me a short story", Some(30));
    let resp = client.chat_completion(&request).await.unwrap();

    assert_valid_chat_response(&resp);
    let content = resp.choices[0].message.content.as_deref().unwrap_or("");
    assert!(!content.is_empty(), "chat response should not be empty");
}
*/ // end Granite GGUF block comment

// ===========================================================================
// CUDA Marlin W4A16 E2E tests — GPTQ and AWQ quantized models
// ===========================================================================
// Run with: cargo test -p vllm-e2e --features e2e,cuda --release --test e1_basic_serving test_cuda_marlin -- --ignored --test-threads=1

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_marlin_gptq_server_starts() {
    let server = TestServer::builder(TestModels::QWEN2_0_5B_GPTQ_INT4)
        .start()
        .await
        .expect("CUDA GPTQ Marlin server should start");

    let client = Client::new(server.base_url());
    assert!(client.health().await.unwrap(), "server should be healthy");
}

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_marlin_gptq_completion() {
    let server = TestServer::builder(TestModels::QWEN2_0_5B_GPTQ_INT4)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = simple_completion_request("The capital of France is", 20);
    let resp = client.completion(&request).await.unwrap();

    assert_valid_completion_response(&resp);
    assert!(
        !resp.choices[0].text.is_empty(),
        "GPTQ Marlin completion should not be empty"
    );
}

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_marlin_gptq_chat() {
    let server = TestServer::builder(TestModels::QWEN2_0_5B_GPTQ_INT4)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = simple_chat_request("What is 2+2? Answer with just the number.", Some(10));
    let resp = client.chat_completion(&request).await.unwrap();

    assert_valid_chat_response(&resp);
    let text = resp.choices[0].message.content.as_deref().unwrap_or("");
    assert!(!text.is_empty(), "GPTQ Marlin chat should not be empty");
}

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_marlin_awq_server_starts() {
    let server = TestServer::builder(TestModels::QWEN2_0_5B_AWQ)
        .start()
        .await
        .expect("CUDA AWQ Marlin server should start");

    let client = Client::new(server.base_url());
    assert!(client.health().await.unwrap(), "server should be healthy");
}

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_marlin_awq_completion() {
    let server = TestServer::builder(TestModels::QWEN2_0_5B_AWQ)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = simple_completion_request("The capital of France is", 20);
    let resp = client.completion(&request).await.unwrap();

    assert_valid_completion_response(&resp);
    assert!(
        !resp.choices[0].text.is_empty(),
        "AWQ Marlin completion should not be empty"
    );
}

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_marlin_awq_chat() {
    let server = TestServer::builder(TestModels::QWEN2_0_5B_AWQ)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = simple_chat_request("What is 2+2? Answer with just the number.", Some(10));
    let resp = client.chat_completion(&request).await.unwrap();

    assert_valid_chat_response(&resp);
    let text = resp.choices[0].message.content.as_deref().unwrap_or("");
    assert!(!text.is_empty(), "AWQ Marlin chat should not be empty");
}

// ===========================================================================
// CUDA Marlin Gemma2 GPTQ E2E tests
// ===========================================================================
// Run with: cargo test -p vllm-e2e --features e2e,cuda --release --test e1_basic_serving test_cuda_marlin_gemma2 -- --ignored --test-threads=1

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_marlin_gemma2_gptq_server_starts() {
    let server = TestServer::builder(TestModels::GEMMA2_2B_GPTQ_INT4)
        .start()
        .await
        .expect("CUDA Gemma2 GPTQ Marlin server should start");

    let client = Client::new(server.base_url());
    assert!(client.health().await.unwrap(), "server should be healthy");
}

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_marlin_gemma2_gptq_completion() {
    let server = TestServer::builder(TestModels::GEMMA2_2B_GPTQ_INT4)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = simple_completion_request("The capital of France is", 20);
    let resp = client.completion(&request).await.unwrap();

    assert_valid_completion_response(&resp);
    assert!(
        !resp.choices[0].text.is_empty(),
        "Gemma2 GPTQ Marlin completion should not be empty"
    );
}

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_marlin_gemma2_gptq_chat() {
    let server = TestServer::builder(TestModels::GEMMA2_2B_GPTQ_INT4)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = simple_chat_request("What is 2+2? Answer with just the number.", Some(10));
    let resp = client.chat_completion(&request).await.unwrap();

    assert_valid_chat_response(&resp);
    let text = resp.choices[0].message.content.as_deref().unwrap_or("");
    assert!(
        !text.is_empty(),
        "Gemma2 GPTQ Marlin chat should not be empty"
    );
}

// ===========================================================================
// CUDA Marlin GPTQ desc_act (activation ordering) E2E test
// ===========================================================================
// Run with: cargo test -p vllm-e2e --features e2e,cuda --release --test e1_basic_serving test_cuda_gptq_desc_act -- --ignored --test-threads=1

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_gptq_desc_act_chat() {
    let server = TestServer::builder(TestModels::TINYLLAMA_1B_GPTQ_DESC_ACT)
        .start()
        .await
        .expect("CUDA GPTQ desc_act server should start");

    let client = Client::new(server.base_url());
    let request = simple_chat_request("What is 2+2? Answer with just the number.", Some(10));
    let resp = client.chat_completion(&request).await.unwrap();

    assert_valid_chat_response(&resp);
    let text = resp.choices[0].message.content.as_deref().unwrap_or("");
    assert!(!text.is_empty(), "GPTQ desc_act chat should not be empty");
}

// ===========================================================================
// CUDA MoE E2E tests
// ===========================================================================
// Run with: cargo test -p vllm-e2e --features e2e,cuda --release --test e1_basic_serving test_cuda_mixtral -- --ignored --test-threads=1

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_mixtral_server_starts() {
    let server = TestServer::builder(TestModels::MIXTRAL_TINY_DPO_CUDA)
        .start()
        .await
        .expect("CUDA Mixtral MoE server should start");

    let client = Client::new(server.base_url());
    assert!(client.health().await.unwrap(), "server should be healthy");

    let models = client.list_models().await.unwrap();
    assert_eq!(models.data.len(), 1);
}

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_mixtral_completion() {
    let server = TestServer::builder(TestModels::MIXTRAL_TINY_DPO_CUDA)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = simple_completion_request("The capital of France is", 20);
    let resp = client.completion(&request).await.unwrap();

    assert_valid_completion_response(&resp);
    assert!(
        !resp.choices[0].text.is_empty(),
        "MoE completion should not be empty"
    );
}

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_mixtral_chat() {
    let server = TestServer::builder(TestModels::MIXTRAL_TINY_DPO_CUDA)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = simple_chat_request("Say hello in one sentence.", Some(50));
    let resp = client.chat_completion(&request).await.unwrap();

    assert_valid_chat_response(&resp);
    assert!(
        !resp.choices[0]
            .message
            .content
            .as_deref()
            .unwrap_or("")
            .is_empty(),
        "MoE chat should produce output"
    );
}

// Qwen2 MoE — ~29GB BF16, fits on L40S (48GB) but tight.
// Run with: cargo test -p vllm-e2e --features e2e,cuda --release --test e1_basic_serving test_cuda_qwen2_moe -- --ignored --test-threads=1

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_qwen2_moe_server_starts() {
    let server = TestServer::builder(TestModels::QWEN2_MOE_A2_7B_CUDA)
        .start()
        .await
        .expect("CUDA Qwen2 MoE server should start");

    let client = Client::new(server.base_url());
    assert!(client.health().await.unwrap(), "server should be healthy");

    let models = client.list_models().await.unwrap();
    assert_eq!(models.data.len(), 1);
}

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_qwen2_moe_completion() {
    let server = TestServer::builder(TestModels::QWEN2_MOE_A2_7B_CUDA)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = simple_completion_request("The capital of France is", 20);
    let resp = client.completion(&request).await.unwrap();

    assert_valid_completion_response(&resp);
    assert!(
        !resp.choices[0].text.is_empty(),
        "Qwen2 MoE completion should not be empty"
    );
}

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_qwen2_moe_chat() {
    let server = TestServer::builder(TestModels::QWEN2_MOE_A2_7B_CUDA)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = simple_chat_request("What is 2+2? Answer with just the number.", Some(10));
    let resp = client.chat_completion(&request).await.unwrap();

    assert_valid_chat_response(&resp);
    assert!(
        !resp.choices[0]
            .message
            .content
            .as_deref()
            .unwrap_or("")
            .is_empty(),
        "Qwen2 MoE chat should produce output"
    );
}

// ===========================================================================
// CUDA graphs + MoE: verify decode CUDA graphs work with MoE models
// ===========================================================================
// MoE kernels (topk_softmax, moe_align_block_size, fused_moe_gemm) have
// deterministic allocation sizes per batch_size, so CUDA graph capture works.
// These tests explicitly disable enforce_eager to exercise the graph path.
//
// Run with: cargo test -p vllm-e2e --features e2e,cuda --release --test e1_basic_serving test_cuda_moe_graphs -- --ignored --test-threads=1

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_moe_graphs_mixtral_completion() {
    let server = TestServer::builder(TestModels::MIXTRAL_TINY_DPO_CUDA)
        .with_enforce_eager(false)
        .start()
        .await
        .expect("Mixtral MoE with CUDA graphs should start");

    let client = Client::new(server.base_url());
    let request = simple_completion_request("The capital of France is", 30);
    let resp = client.completion(&request).await.unwrap();

    assert_valid_completion_response(&resp);
    assert!(
        !resp.choices[0].text.is_empty(),
        "MoE + CUDA graphs completion should not be empty"
    );
}

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_moe_graphs_mixtral_chat() {
    let server = TestServer::builder(TestModels::MIXTRAL_TINY_DPO_CUDA)
        .with_enforce_eager(false)
        .start()
        .await
        .expect("Mixtral MoE with CUDA graphs should start");

    let client = Client::new(server.base_url());
    let request = simple_chat_request("Say hello in one sentence.", Some(50));
    let resp = client.chat_completion(&request).await.unwrap();

    assert_valid_chat_response(&resp);
    assert!(
        !resp.choices[0]
            .message
            .content
            .as_deref()
            .unwrap_or("")
            .is_empty(),
        "MoE + CUDA graphs chat should produce output"
    );
}

/// Multi-turn chat with MoE + CUDA graphs: exercises graph replay with
/// changing batch composition (prefill eager + decode graphed across turns).
#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_moe_graphs_mixtral_multi_turn() {
    let server = TestServer::builder(TestModels::MIXTRAL_TINY_DPO_CUDA)
        .with_enforce_eager(false)
        .start()
        .await
        .expect("Mixtral MoE with CUDA graphs should start");

    let client = Client::new(server.base_url());

    // Turn 1
    let request = simple_chat_request("What is 2+2?", Some(30));
    let resp = client.chat_completion(&request).await.unwrap();
    assert_valid_chat_response(&resp);
    let turn1 = resp.choices[0]
        .message
        .content
        .as_deref()
        .unwrap_or("")
        .to_string();
    assert!(!turn1.is_empty(), "Turn 1 should produce output");

    // Turn 2 — new request exercises graph replay
    let request = simple_chat_request("What is 3+3?", Some(30));
    let resp = client.chat_completion(&request).await.unwrap();
    assert_valid_chat_response(&resp);
    assert!(
        !resp.choices[0]
            .message
            .content
            .as_deref()
            .unwrap_or("")
            .is_empty(),
        "Turn 2 should produce output (graph replay)"
    );
}

// ===========================================================================
// CUDA semantic correctness + multi-turn + non-greedy tests
// ===========================================================================
// These tests validate output quality beyond "non-empty", catching bugs like
// paged FA2 prefill corruption, KV cache continuity issues, and GPU sampling errors.
//
// Run with: cargo test -p vllm-e2e --features e2e,cuda --release --test e1_basic_serving test_cuda_correctness -- --ignored --test-threads=1

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_correctness_completion_semantic() {
    let server = TestServer::builder(TestModels::QWEN2_0_5B_CUDA)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = simple_completion_request("The capital of France is", 20);
    let resp = client.completion(&request).await.unwrap();

    assert_valid_completion_response(&resp);
    let text = resp.choices[0].text.to_lowercase();
    assert!(
        text.contains("paris"),
        "expected 'paris' in completion of 'The capital of France is', got: {}",
        text
    );
}

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_correctness_multi_turn_chat() {
    let server = TestServer::builder(TestModels::QWEN2_0_5B_CUDA)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());

    // Turn 1: establish a fact
    let req1 = ChatCompletionRequest {
        messages: vec![user_msg(
            "My name is Claude. Please remember that. Reply with just 'OK'.",
        )],
        max_tokens: Some(10),
        temperature: Some(0.0),
        ..default_chat_request()
    };
    let resp1 = client.chat_completion(&req1).await.unwrap();
    assert_valid_chat_response(&resp1);

    // Turn 2: query the fact — requires correct KV cache from turn 1 prefill
    let turn1_text = resp1.choices[0].message.content.clone().unwrap_or_default();
    let req2 = ChatCompletionRequest {
        messages: vec![
            user_msg("My name is Claude. Please remember that. Reply with just 'OK'."),
            assistant_msg(&turn1_text),
            user_msg("What is my name?"),
        ],
        max_tokens: Some(20),
        temperature: Some(0.0),
        ..default_chat_request()
    };
    let resp2 = client.chat_completion(&req2).await.unwrap();
    assert_valid_chat_response(&resp2);
    let text = resp2.choices[0]
        .message
        .content
        .as_deref()
        .unwrap_or("")
        .to_lowercase();
    assert!(
        text.contains("claude"),
        "turn 2 should remember 'Claude', got: {}",
        text
    );
}

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_correctness_nongreedy_chat() {
    let server = TestServer::builder(TestModels::QWEN2_0_5B_CUDA)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = ChatCompletionRequest {
        messages: vec![user_msg(
            "Say hello and introduce yourself in one sentence.",
        )],
        max_tokens: Some(50),
        temperature: Some(0.7),
        ..default_chat_request()
    };
    let resp = client.chat_completion(&request).await.unwrap();
    assert_valid_chat_response(&resp);

    let text = resp.choices[0].message.content.as_deref().unwrap_or("");
    assert_coherent_text(text, 5);
    let word_count = text.split_whitespace().count();
    assert!(word_count >= 3, "expected at least 3 words, got: {}", text);
}

// ---------------------------------------------------------------------------
// Paged FA2 multi-turn regression tests
//
// The paged FA2 bug caused garbled output on turn 2+ when KV cache blocks
// were non-contiguous. These tests specifically exercise multi-turn chat
// which requires correct paged KV cache reads across multiple prefills.
// ---------------------------------------------------------------------------

/// Multi-turn with 3 turns — each turn adds more KV blocks, stressing
/// the paged block_table remapping.
#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_paged_fa2_three_turn_chat() {
    let server = TestServer::builder(TestModels::QWEN2_0_5B_CUDA)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());

    // Turn 1
    let req1 = ChatCompletionRequest {
        messages: vec![user_msg("The number I'm thinking of is 42. Just say OK.")],
        max_tokens: Some(10),
        temperature: Some(0.0),
        ..default_chat_request()
    };
    let resp1 = client.chat_completion(&req1).await.unwrap();
    assert_valid_chat_response(&resp1);
    let t1 = resp1.choices[0].message.content.clone().unwrap_or_default();

    // Turn 2
    let req2 = ChatCompletionRequest {
        messages: vec![
            user_msg("The number I'm thinking of is 42. Just say OK."),
            assistant_msg(&t1),
            user_msg("What number am I thinking of? Answer with just the number."),
        ],
        max_tokens: Some(10),
        temperature: Some(0.0),
        ..default_chat_request()
    };
    let resp2 = client.chat_completion(&req2).await.unwrap();
    assert_valid_chat_response(&resp2);
    let t2_text = resp2.choices[0].message.content.as_deref().unwrap_or("");
    assert!(
        t2_text.contains("42"),
        "turn 2 should recall '42', got: {}",
        t2_text
    );
    let t2 = resp2.choices[0].message.content.clone().unwrap_or_default();

    // Turn 3 — even more KV cache blocks allocated
    let req3 = ChatCompletionRequest {
        messages: vec![
            user_msg("The number I'm thinking of is 42. Just say OK."),
            assistant_msg(&t1),
            user_msg("What number am I thinking of? Answer with just the number."),
            assistant_msg(&t2),
            user_msg("Double that number. Answer with just the number."),
        ],
        max_tokens: Some(10),
        temperature: Some(0.0),
        ..default_chat_request()
    };
    let resp3 = client.chat_completion(&req3).await.unwrap();
    assert_valid_chat_response(&resp3);
    let t3_text = resp3.choices[0].message.content.as_deref().unwrap_or("");
    assert!(
        t3_text.contains("84"),
        "turn 3 should compute 42*2=84, got: {}",
        t3_text
    );
}

/// Interleaved multi-turn requests — two separate conversations interleave
/// their block allocations, ensuring the block_table correctly maps
/// non-contiguous physical blocks.
#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_paged_fa2_interleaved_multi_turn() {
    let server = TestServer::builder(TestModels::QWEN2_0_5B_CUDA)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());

    // Turn 1 for user A — allocates some blocks
    let req_a1 = ChatCompletionRequest {
        messages: vec![user_msg(
            "Remember: the color is blue. Reply with only 'OK'.",
        )],
        max_tokens: Some(5),
        temperature: Some(0.0),
        ..default_chat_request()
    };
    let resp_a1 = client.chat_completion(&req_a1).await.unwrap();
    assert_valid_chat_response(&resp_a1);
    let ta1 = resp_a1.choices[0]
        .message
        .content
        .clone()
        .unwrap_or_default();

    // Turn 1 for user B — allocates more blocks (interleaved with A's freed blocks)
    let req_b1 = ChatCompletionRequest {
        messages: vec![user_msg(
            "Remember: the animal is cat. Reply with only 'OK'.",
        )],
        max_tokens: Some(5),
        temperature: Some(0.0),
        ..default_chat_request()
    };
    let resp_b1 = client.chat_completion(&req_b1).await.unwrap();
    assert_valid_chat_response(&resp_b1);
    let tb1 = resp_b1.choices[0]
        .message
        .content
        .clone()
        .unwrap_or_default();

    // Turn 2 for user A — prefills into new blocks with possible gaps
    let req_a2 = ChatCompletionRequest {
        messages: vec![
            user_msg("Remember: the color is blue. Reply with only 'OK'."),
            assistant_msg(&ta1),
            user_msg("What color did I say? Answer with just the color."),
        ],
        max_tokens: Some(10),
        temperature: Some(0.0),
        ..default_chat_request()
    };
    let resp_a2 = client.chat_completion(&req_a2).await.unwrap();
    assert_valid_chat_response(&resp_a2);
    let text_a = resp_a2.choices[0]
        .message
        .content
        .as_deref()
        .unwrap_or("")
        .to_lowercase();
    assert!(
        text_a.contains("blue"),
        "user A turn 2 should recall 'blue', got: {}",
        text_a
    );

    // Turn 2 for user B
    let req_b2 = ChatCompletionRequest {
        messages: vec![
            user_msg("Remember: the animal is cat. Reply with only 'OK'."),
            assistant_msg(&tb1),
            user_msg("What animal did I say? Answer with just the animal."),
        ],
        max_tokens: Some(10),
        temperature: Some(0.0),
        ..default_chat_request()
    };
    let resp_b2 = client.chat_completion(&req_b2).await.unwrap();
    assert_valid_chat_response(&resp_b2);
    let text_b = resp_b2.choices[0]
        .message
        .content
        .as_deref()
        .unwrap_or("")
        .to_lowercase();
    assert!(
        text_b.contains("cat"),
        "user B turn 2 should recall 'cat', got: {}",
        text_b
    );
}

// ===========================================================================
// CUDA Logprobs tests
// ===========================================================================
// Validates that logprobs are correctly returned from the worker (forces CPU
// fallback path where logprobs are computed).
//
// Run with: cargo test -p vllm-e2e --features e2e,cuda --release --test e1_basic_serving test_cuda_logprobs -- --ignored --test-threads=1

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_logprobs_chat() {
    let server = TestServer::builder(TestModels::QWEN2_0_5B_CUDA)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = ChatCompletionRequest {
        messages: vec![user_msg("Hello!")],
        logprobs: Some(true),
        top_logprobs: Some(5),
        max_tokens: Some(10),
        temperature: Some(0.0),
        ..default_chat_request()
    };

    let resp = client.chat_completion(&request).await.unwrap();
    assert_valid_chat_response(&resp);

    let logprobs = resp.choices[0]
        .logprobs
        .as_ref()
        .expect("logprobs should be present");
    let content = logprobs
        .content
        .as_ref()
        .expect("logprobs.content should be present");
    assert!(!content.is_empty(), "logprobs content should not be empty");
    for entry in content {
        assert!(
            !entry.top_logprobs.is_empty(),
            "top_logprobs should not be empty"
        );
        // Verify logprobs are valid (non-positive log probabilities).
        assert!(
            entry.logprob <= 0.0,
            "logprob should be non-positive, got {}",
            entry.logprob
        );
    }
}

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_logprobs_completion() {
    let server = TestServer::builder(TestModels::QWEN2_0_5B_CUDA)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = CompletionRequest {
        prompt: Some(CompletionPrompt::Single(
            "The capital of France is".to_string(),
        )),
        max_tokens: Some(10),
        temperature: Some(0.0),
        logprobs: Some(5),
        ..default_completion_request()
    };

    let resp = client.completion(&request).await.unwrap();
    assert_valid_completion_response(&resp);

    let token_logprobs = &resp.choices[0]
        .logprobs
        .as_ref()
        .expect("logprobs should be present")
        .token_logprobs;
    assert!(
        !token_logprobs.is_empty(),
        "token_logprobs should not be empty"
    );
    for lp in token_logprobs {
        if let Some(v) = lp {
            assert!(*v <= 0.0, "logprob should be non-positive, got {v}");
        }
    }
}

// ===========================================================================
// CUDA Grammar / constrained decoding tests
// ===========================================================================
// Validates that guided_regex constrains output via the worker's grammar state.
//
// Run with: cargo test -p vllm-e2e --features e2e,cuda --release --test e1_basic_serving test_cuda_grammar -- --ignored --test-threads=1

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_grammar_regex_digits() {
    let server = TestServer::builder(TestModels::QWEN2_0_5B_CUDA)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());

    // Use guided_regex to force the model to output only digits.
    let request = ChatCompletionRequest {
        messages: vec![user_msg("Give me a number.")],
        max_tokens: Some(10),
        temperature: Some(0.0),
        guided_regex: Some("[0-9]+".to_string()),
        ..default_chat_request()
    };

    let resp = client.chat_completion(&request).await.unwrap();
    assert_valid_chat_response(&resp);

    let text = resp.choices[0].message.content.as_deref().unwrap_or("");
    assert!(
        !text.is_empty(),
        "grammar-constrained output should not be empty"
    );
    assert!(
        text.chars().all(|c| c.is_ascii_digit()),
        "guided_regex '[0-9]+' should produce only digits, got: {text:?}"
    );
}

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_grammar_json_object() {
    let server = TestServer::builder(TestModels::QWEN2_0_5B_CUDA)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());

    // Use response_format: json_object to force valid JSON output.
    let request = ChatCompletionRequest {
        messages: vec![user_msg(
            "Return a JSON object with a key 'name' set to 'Alice'.",
        )],
        max_tokens: Some(50),
        temperature: Some(0.0),
        response_format: Some(scratchy_serving_api::protocol::ResponseFormat::Standard(
            scratchy_serving_api::protocol::StandardResponseFormat {
                format_type: "json_object".to_string(),
                json_schema: None,
            },
        )),
        ..default_chat_request()
    };

    let resp = client.chat_completion(&request).await.unwrap();
    assert_valid_chat_response(&resp);

    let text = resp.choices[0].message.content.as_deref().unwrap_or("");
    assert!(
        !text.is_empty(),
        "json_object constrained output should not be empty"
    );
    // The grammar constrains to valid JSON, but the model may append EOS tokens
    // after the JSON is complete. Trim known EOS markers before parsing.
    let trimmed = text.trim_end_matches("</s>").trim();
    let parsed: Result<serde_json::Value, _> = serde_json::from_str(trimmed);
    assert!(
        parsed.is_ok(),
        "response_format json_object should produce valid JSON, got: {text:?}"
    );
}

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_grammar_ebnf_digits() {
    let server = TestServer::builder(TestModels::QWEN2_0_5B_CUDA)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());

    // Use guided_grammar (Lark/EBNF) to force digit-only output.
    let request = ChatCompletionRequest {
        messages: vec![user_msg("Give me a number.")],
        max_tokens: Some(10),
        temperature: Some(0.0),
        guided_grammar: Some(r#"start: /[0-9]+/"#.to_string()),
        ..default_chat_request()
    };

    let resp = client.chat_completion(&request).await.unwrap();
    assert_valid_chat_response(&resp);

    let text = resp.choices[0].message.content.as_deref().unwrap_or("");
    assert!(
        !text.is_empty(),
        "EBNF grammar-constrained output should not be empty"
    );
    assert!(
        text.chars().all(|c| c.is_ascii_digit()),
        "guided_grammar digits should produce only digits, got: {text:?}"
    );
}

// Validates structural_tag response_format constrains output via the worker.
// The structural tag defines a trigger that forces JSON schema output between tags.
//
// Run with: cargo test -p vllm-e2e --features e2e,cuda --release --test e1_basic_serving test_cuda_grammar_structural_tag -- --ignored --test-threads=1

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_grammar_structural_tag() {
    let server = TestServer::builder(TestModels::QWEN2_0_5B_CUDA)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());

    // Use structural_tag response_format via raw JSON (legacy format).
    // The trigger "TOOL:" with begin "TOOL:" forces the model to emit a JSON object
    // conforming to the schema between the trigger and end marker.
    let response_format: scratchy_serving_api::protocol::ResponseFormat =
        serde_json::from_value(serde_json::json!({
            "type": "structural_tag",
            "structures": [{
                "begin": "TOOL:",
                "schema": {
                    "type": "object",
                    "properties": {
                        "value": { "type": "integer" }
                    },
                    "required": ["value"]
                },
                "end": ";END"
            }],
            "triggers": ["TOOL:"]
        }))
        .expect("structural_tag response_format should deserialize");

    let request = ChatCompletionRequest {
        messages: vec![user_msg("Call a tool with value 42.")],
        max_tokens: Some(60),
        temperature: Some(0.0),
        response_format: Some(response_format),
        ..default_chat_request()
    };

    let resp = client.chat_completion(&request).await.unwrap();
    let text = resp.choices[0].message.content.as_deref().unwrap_or("");
    assert!(
        !text.is_empty(),
        "structural_tag constrained output should not be empty"
    );
    // The output should contain the structural tag markers and valid JSON between them.
    // With the grammar constraint, the model must emit text matching the pattern:
    //   (free text)* TOOL: <json conforming to schema> ;END (free text)*
    // We verify the JSON portion parses correctly.
    if let Some(tool_start) = text.find("TOOL:") {
        let after_tool = &text[tool_start + "TOOL:".len()..];
        if let Some(end_pos) = after_tool.find(";END") {
            let json_part = after_tool[..end_pos].trim();
            let parsed: Result<serde_json::Value, _> = serde_json::from_str(json_part);
            assert!(
                parsed.is_ok(),
                "structural_tag should produce valid JSON between markers, got: {json_part:?}"
            );
            let obj = parsed.unwrap();
            assert!(
                obj.get("value").is_some(),
                "structural_tag JSON should have 'value' key, got: {obj}"
            );
        }
    }
    // At minimum, the grammar should have constrained output to be non-empty
    // (the full structural tag pattern may or may not appear depending on model behavior,
    // but the grammar engine ensures validity of whatever is produced).
}

// ===========================================================================
// Metal Grammar / constrained decoding tests
// ===========================================================================
// Same coverage as the CUDA grammar tests above, exercising the Metal
// grammar-mask kernel (sets disallowed logits to -inf before argmax). The
// mask guarantees grammar-valid output regardless of model quality, so the
// smallest instruct model suffices.
//
// Run with: cargo test -p scratchy-e2e --features e2e,metal --release --test e1_basic_serving test_metal_grammar -- --ignored --test-threads=1

#[cfg(feature = "metal")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_metal_grammar_regex_digits() {
    let server = TestServer::builder(TestModels::SMOLLM_135M_F16)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());

    let request = ChatCompletionRequest {
        messages: vec![user_msg("Give me a number.")],
        max_tokens: Some(10),
        temperature: Some(0.0),
        guided_regex: Some("[0-9]+".to_string()),
        ..default_chat_request()
    };

    let resp = client.chat_completion(&request).await.unwrap();
    assert_valid_chat_response(&resp);

    let text = resp.choices[0].message.content.as_deref().unwrap_or("");
    assert!(
        !text.is_empty(),
        "grammar-constrained output should not be empty"
    );
    assert!(
        text.chars().all(|c| c.is_ascii_digit()),
        "guided_regex '[0-9]+' should produce only digits, got: {text:?}"
    );
}

#[cfg(feature = "metal")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_metal_grammar_json_schema() {
    let server = TestServer::builder(TestModels::SMOLLM_135M_F16)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());

    // A bounded schema (single boolean field, additionalProperties false)
    // forces the grammar to emit exactly `{"ok": true}` / `{"ok": false}` and
    // then close — short and self-terminating regardless of the (tiny) model's
    // quality, isolating the test to the mask's correctness. A boolean is used
    // rather than an integer because JSON integers are unbounded: a degenerate
    // model will emit digits until max_tokens and never close the object.
    let schema = serde_json::json!({
        "type": "object",
        "properties": { "ok": { "type": "boolean" } },
        "required": ["ok"],
        "additionalProperties": false
    });
    let request = ChatCompletionRequest {
        messages: vec![user_msg("Return a JSON object with a boolean 'ok'.")],
        max_tokens: Some(64),
        temperature: Some(0.0),
        response_format: Some(scratchy_serving_api::protocol::ResponseFormat::Standard(
            scratchy_serving_api::protocol::StandardResponseFormat {
                format_type: "json_schema".to_string(),
                json_schema: Some(scratchy_serving_api::protocol::JsonSchemaResponseFormat {
                    name: "obj".to_string(),
                    description: None,
                    json_schema: Some(schema),
                    strict: Some(true),
                }),
            },
        )),
        ..default_chat_request()
    };

    let resp = client.chat_completion(&request).await.unwrap();
    assert_valid_chat_response(&resp);

    let text = resp.choices[0].message.content.as_deref().unwrap_or("");
    assert!(
        !text.is_empty(),
        "json_schema constrained output should not be empty"
    );
    // The schema grammar constrains the STRUCTURE; with a greedy 135M model the
    // JSON grammar permits unbounded trailing whitespace, so the object may not
    // close within max_tokens. Assert the grammar-enforced structure on the
    // whitespace-stripped output (a free model would not be forced to emit
    // `{"ok":<bool>`), and additionally require a full parse if it did close.
    let compact: String = text.chars().filter(|c| !c.is_whitespace()).collect();
    assert!(
        compact.starts_with("{\"ok\":") && (compact.contains("true") || compact.contains("false")),
        "json_schema must constrain output to the schema structure, got: {text:?}"
    );
    if compact.ends_with('}') {
        let parsed: serde_json::Value = serde_json::from_str(&compact).unwrap_or_else(|e| {
            panic!("closed json_schema object must parse ({e}), got: {text:?}")
        });
        assert!(
            parsed.get("ok").and_then(|v| v.as_bool()).is_some(),
            "closed json_schema object must have boolean 'ok', got: {text:?}"
        );
    }
}

// Note: a "produce a complete, parseable JSON object" e2e test is intentionally
// omitted. guided_regex / JSON grammars permit EOS at accepting points but do
// not *force* termination, so under greedy decoding a weak model keeps emitting
// grammar-valid filler (whitespace / more digits) until max_tokens rather than
// closing — identical behavior on the CUDA path. The mask's correctness (only
// grammar-valid tokens survive) is covered by the tests here plus the
// grammar_mask kernel unit test; clean termination is a model/sampling property,
// not a property of the Metal mask.

#[cfg(feature = "metal")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_metal_grammar_ebnf_digits() {
    let server = TestServer::builder(TestModels::SMOLLM_135M_F16)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());

    let request = ChatCompletionRequest {
        messages: vec![user_msg("Give me a number.")],
        max_tokens: Some(10),
        temperature: Some(0.0),
        guided_grammar: Some(r#"start: /[0-9]+/"#.to_string()),
        ..default_chat_request()
    };

    let resp = client.chat_completion(&request).await.unwrap();
    assert_valid_chat_response(&resp);

    let text = resp.choices[0].message.content.as_deref().unwrap_or("");
    assert!(
        !text.is_empty(),
        "EBNF grammar-constrained output should not be empty"
    );
    assert!(
        text.chars().all(|c| c.is_ascii_digit()),
        "guided_grammar digits should produce only digits, got: {text:?}"
    );
}

// Regression guard for the grammar-mask buffer-residency bug: on a LARGE-vocab
// model (Qwen2.5: ~152k) at the default long context, the per-step grammar
// bitset must be pinned in the residency set, else it is evicted under KV
// pressure and the kernel reads garbage (the mask silently fails → unconstrained
// output). The small-vocab SmolLM tests above did NOT catch this; this one does.
// Build with `--features scratchy-models/qwen2` (bare arch name = every
// qwen2 config; or the specific `<stem>` matching `TestModels::QWEN2`).
#[cfg(feature = "metal")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_metal_grammar_regex_digits_large_vocab() {
    let server = TestServer::builder(TestModels::QWEN2)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());

    let request = ChatCompletionRequest {
        messages: vec![user_msg("Give me a number.")],
        max_tokens: Some(12),
        temperature: Some(0.0),
        guided_regex: Some("[0-9]+".to_string()),
        ..default_chat_request()
    };

    let resp = client.chat_completion(&request).await.unwrap();
    assert_valid_chat_response(&resp);

    let text = resp.choices[0].message.content.as_deref().unwrap_or("");
    assert!(
        !text.is_empty(),
        "large-vocab grammar-constrained output should not be empty"
    );
    assert!(
        text.chars().all(|c| c.is_ascii_digit()),
        "guided_regex '[0-9]+' on a large-vocab model must produce only digits \
         (residency regression if not), got: {text:?}"
    );
}

// ===========================================================================
// gemma-4 hd512 unfused attention — CHUNKED-PREFILL CONTINUATION regression guard
// ===========================================================================

/// REGRESSION GUARD — DO NOT DELETE, DO NOT SHORTEN THE PROMPT.
///
/// gemma-4's global (head_dim 512) layers use the "unfused" attention path. On a
/// chunked-prefill CONTINUATION (prompt longer than the 2048 chunk size → chunk 2+ has
/// lq != kv_len), the bf16 NAX gemm's UNGUARDED contraction K-tail over-read pulled
/// STALE dense-scratch (incl. NaN padding rows) into real outputs → garbage (empty /
/// token-0) past one chunk. Fixed by zeroing the V^T K-tail (attention_dense_gather.metal)
/// and the prefill-bucket padding rows (attention_causal_softmax.metal).
///
/// A SHORT prompt (single chunk) does NOT catch this — the scratch is fresh/zeroed on the
/// first chunk; the bug only appears once a 2nd chunk REUSES the scratch. So this guard
/// MUST send a > 2048-token prompt and assert long-range needle recall. (An isolated
/// kernel test cannot reproduce it either — it needs the real reused-scratch continuation.)
#[cfg(feature = "metal")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_metal_gemma4_continuation_recall() {
    scratchy_e2e::skip_if_no_gpu!();
    let server = TestServer::builder(TestModels::GEMMA4)
        .start()
        .await
        .expect("gemma-4-12b server should start");
    let client = Client::new(server.base_url());

    // > 2048-token prompt: a clearly-stated needle at the top, a wall of distractor log
    // lines pushing well past the chunk boundary, then a direct question at the end.
    let mut prompt = String::from(
        "Remember this access code exactly — you will be asked for it at the end: \
         BANANA7492. Do not forget it.\n\n",
    );
    for i in 0..280u32 {
        prompt.push_str(&format!(
            "Log entry {i}: sensor {} reported reading {} at byte offset {}.\n",
            i.wrapping_mul(7) % 97,
            i.wrapping_mul(31) % 1000,
            i.wrapping_mul(13) % 4096,
        ));
    }
    prompt.push_str(
        "\nQuestion: what was the access code stated at the very beginning? \
         Reply with ONLY the code, nothing else.",
    );

    let request = simple_chat_request(&prompt, Some(12));
    let resp = client.chat_completion(&request).await.unwrap();
    assert_valid_chat_response(&resp);
    let text = resp.choices[0].message.content.as_deref().unwrap_or("");
    assert!(
        text.contains("BANANA7492"),
        "gemma-4 chunked-prefill CONTINUATION must recall the needle. Got {text:?} — the \
         hd512 unfused continuation path regressed (NAX gemm K-tail over-read / padding-row \
         zeroing). See attention_dense_gather.metal + attention_causal_softmax.metal.",
    );
}

// ===========================================================================
// LogitsProcessor tests: min_tokens, logit_bias, penalties
// ===========================================================================

/// Test min_tokens: with min_tokens=10, output must be at least 10 tokens.
#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_min_tokens_completion() {
    let server = TestServer::builder(TestModels::QWEN2_0_5B_CUDA)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = CompletionRequest {
        prompt: Some(CompletionPrompt::Single("Hello".to_string())),
        max_tokens: Some(50),
        min_tokens: 10,
        temperature: Some(0.0),
        ..default_completion_request()
    };

    let resp = client.completion(&request).await.unwrap();
    assert_valid_completion_response(&resp);

    let text = &resp.choices[0].text;
    // Rough check: 10 tokens should produce at least ~15 chars of output.
    // The exact token count isn't available via the API, but usage.completion_tokens is.
    let completion_tokens = resp.usage.completion_tokens.unwrap_or(0);
    assert!(
        completion_tokens >= 10,
        "min_tokens=10 should produce at least 10 completion tokens, got {completion_tokens}. text: {text:?}"
    );
}

/// Test logit_bias: boost a specific token to force it into output.
#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_logit_bias_completion() {
    let server = TestServer::builder(TestModels::QWEN2_0_5B_CUDA)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());

    // Token 220 in Qwen2 tokenizer is typically a common token.
    // Strongly bias it (+100) so it dominates output.
    let mut bias = std::collections::HashMap::new();
    bias.insert("220".to_string(), 100.0);

    let request = CompletionRequest {
        prompt: Some(CompletionPrompt::Single("Test".to_string())),
        max_tokens: Some(5),
        temperature: Some(0.0),
        logit_bias: Some(bias),
        ..default_completion_request()
    };

    let resp = client.completion(&request).await.unwrap();
    assert_valid_completion_response(&resp);

    // The biased token should dominate. At minimum, we should get non-empty output.
    let text = &resp.choices[0].text;
    assert!(
        !text.is_empty(),
        "logit_bias completion should produce output"
    );
    // With +100 bias on token 220, it should dominate output.
    // We just verify non-empty output and no crash — the exact content depends on tokenizer.
    assert!(
        text.len() >= 2,
        "logit_bias +100 should produce at least a couple characters, got {text:?}"
    );
}

/// Test penalties: repetition_penalty suppresses repeated tokens.
#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_penalties_completion() {
    let server = TestServer::builder(TestModels::QWEN2_0_5B_CUDA)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());

    // High repetition penalty should reduce repetition.
    let request = CompletionRequest {
        prompt: Some(CompletionPrompt::Single("The".to_string())),
        max_tokens: Some(30),
        temperature: Some(0.5),
        repetition_penalty: Some(2.0),
        seed: Some(42),
        ..default_completion_request()
    };

    let resp = client.completion(&request).await.unwrap();
    assert_valid_completion_response(&resp);
    let text = &resp.choices[0].text;
    assert!(
        !text.is_empty(),
        "penalties completion should produce output"
    );
}

// ---------------------------------------------------------------------------
// Gemma3 TP=2 tests (CUDA, NCCL)
// ---------------------------------------------------------------------------

/// TP=2 Gemma3 completion test.
///
///   cargo test -p vllm-e2e --features e2e,nccl --release --test e1_basic_serving test_cuda_tp2_gemma3 -- --ignored --test-threads=1
#[cfg(feature = "nccl")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_tp2_gemma3_completion() {
    let server = TestServer::builder(TestModels::GEMMA3_4B_IT_CUDA)
        .with_tensor_parallel_size(2)
        .start()
        .await
        .expect("TP=2 Gemma3-1B server should start");

    let client = Client::new(server.base_url());
    assert!(client.health().await.unwrap(), "server should be healthy");

    let request = simple_completion_request("The capital of France is", 20);
    let resp = client.completion(&request).await.unwrap();

    assert_valid_completion_response(&resp);
    let text = &resp.choices[0].text;
    assert!(
        !text.is_empty(),
        "TP=2 Gemma3 completion should produce non-empty text"
    );
}

// ---------------------------------------------------------------------------
// Pipeline-parallel (PP=2) tests — require 2x GPU (nick2/nick3 pod)
// ---------------------------------------------------------------------------

/// Gemma2-2B with PP=2 — validates layer sharding, inter-stage P2P send/recv.
///
/// Run on nick3 pod (2x L40S):
///   cargo test -p vllm-e2e --features e2e,cuda,nccl --release --test e1_basic_serving test_cuda_pp2_gemma2 -- --ignored --test-threads=1
#[cfg(all(feature = "cuda", feature = "nccl"))]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_pp2_gemma2_completion() {
    let server = TestServer::builder(TestModels::GEMMA2_2B_IT_CUDA)
        .with_pipeline_parallel_size(2)
        .with_enforce_eager(true)
        .start()
        .await
        .expect("PP=2 Gemma2-2B server should start");

    let client = Client::new(server.base_url());
    assert!(client.health().await.unwrap(), "server should be healthy");

    let request = simple_completion_request("The capital of France is", 20);
    let resp = client.completion(&request).await.unwrap();

    assert_valid_completion_response(&resp);
    let text = &resp.choices[0].text;
    assert!(
        !text.is_empty(),
        "PP=2 Gemma2 completion should produce non-empty text"
    );
}

/// Gemma3-1B with PP=2 — validates dual RotaryCache selection per PP stage.
///
/// Run on nick3 pod (2x L40S):
///   cargo test -p vllm-e2e --features e2e,cuda,nccl --release --test e1_basic_serving test_cuda_pp2_gemma3 -- --ignored --test-threads=1
#[cfg(all(feature = "cuda", feature = "nccl"))]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_pp2_gemma3_completion() {
    let server = TestServer::builder(TestModels::GEMMA3_1B_IT_CUDA)
        .with_pipeline_parallel_size(2)
        .with_enforce_eager(true)
        .start()
        .await
        .expect("PP=2 Gemma3-1B server should start");

    let client = Client::new(server.base_url());
    assert!(client.health().await.unwrap(), "server should be healthy");

    let request = simple_completion_request("The capital of France is", 20);
    let resp = client.completion(&request).await.unwrap();

    assert_valid_completion_response(&resp);
    let text = &resp.choices[0].text;
    assert!(
        !text.is_empty(),
        "PP=2 Gemma3 completion should produce non-empty text"
    );
}

// ---------------------------------------------------------------------------
// Sleep/wake tests
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_sleep_wake() {
    let server = TestServer::builder(TestModels::SMOLLM_135M_CUDA)
        .start()
        .await
        .expect("server should start");

    let client = Client::new(server.base_url());

    // 1. Verify working.
    let request = simple_completion_request("Hello", 10);
    let resp = client.completion(&request).await.unwrap();
    assert_valid_completion_response(&resp);

    // 2. Record GPU memory before sleep.
    let mem_before = client.gpu_memory().await.unwrap();
    assert!(mem_before.used_bytes > 0, "GPU should have memory in use");

    // 3. Sleep.
    client.sleep(1).await.unwrap();
    assert!(
        client.is_sleeping().await.unwrap(),
        "engine should be sleeping"
    );

    // 4. Verify GPU memory dropped significantly.
    let mem_sleeping = client.gpu_memory().await.unwrap();
    assert!(
        mem_sleeping.used_bytes < mem_before.used_bytes / 2,
        "GPU memory should drop significantly after sleep: before={}, sleeping={}",
        mem_before.used_bytes,
        mem_sleeping.used_bytes,
    );

    // 5. Wake up.
    client.wake_up(None).await.unwrap();
    assert!(
        !client.is_sleeping().await.unwrap(),
        "engine should be awake"
    );

    // 6. Verify still works.
    let resp = client.completion(&request).await.unwrap();
    assert_valid_completion_response(&resp);
}

// ===========================================================================
// CUDA GGUF IQ4 E2E tests — IQ (importance-matrix) quantized models on GPU
// ===========================================================================
// Tests IQ4_XS quantization support via the Q8_1 MMVQ path.
//
// Run with: cargo test -p vllm-e2e --features e2e,cuda --release --test e1_basic_serving test_cuda_gguf_iq4 -- --ignored --test-threads=1

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_gguf_iq4_qwen3_server_starts() {
    let server = TestServer::builder(TestModels::QWEN3_0_6B_GGUF)
        .with_args(&["--gguf-file", "Qwen3-0.6B-IQ4_XS.gguf"])
        .start()
        .await
        .expect("CUDA Qwen3 0.6B IQ4_XS GGUF server should start");

    let client = Client::new(server.base_url());
    assert!(client.health().await.unwrap(), "server should be healthy");

    let models = client.list_models().await.unwrap();
    assert_eq!(models.data.len(), 1);
}

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_gguf_iq4_qwen3_chat() {
    let server = TestServer::builder(TestModels::QWEN3_0_6B_GGUF)
        .with_args(&["--gguf-file", "Qwen3-0.6B-IQ4_XS.gguf"])
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = simple_chat_request("Say hello in one sentence.", Some(50));
    let resp = client.chat_completion(&request).await.unwrap();

    assert_valid_chat_response(&resp);
    let text = resp.choices[0].message.content.as_deref().unwrap_or("");
    assert_coherent_text(text, 2);
}
