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
        reasoning: None,
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
// Metal Grammar / constrained decoding tests
// ===========================================================================
// Exercises the Metal grammar-mask kernel (sets disallowed logits to -inf
// before argmax). The mask guarantees grammar-valid output regardless of
// model quality, so the smallest instruct model suffices.
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
// closing. The mask's correctness (only
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

// ---------------------------------------------------------------------------
// Pipeline-parallel (PP=2) tests — require 2x GPU (nick2/nick3 pod)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Sleep/wake tests
// ---------------------------------------------------------------------------
