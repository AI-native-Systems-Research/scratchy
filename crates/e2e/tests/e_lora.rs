// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! E2E tests for LoRA adapter support.
//!
//! Self-contained tests create a synthetic LoRA adapter with random weights
//! targeting SmolLM-135M's q_proj and v_proj. This proves the full LoRA
//! injection pipeline works end-to-end without needing external adapters.
//!
//! Run with: `cargo test -p vllm-e2e --features e2e --test e_lora -- --ignored --test-threads=1`

#![cfg(feature = "e2e")]

use scratchy_e2e::assertions::{assert_valid_chat_response, assert_valid_completion_response};
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

fn default_completion_request() -> CompletionRequest {
    serde_json::from_str(r#"{}"#).unwrap()
}

/// Create a synthetic LoRA adapter in a temp directory.
///
/// SmolLM-135M has hidden_size=576, num_attention_heads=9, num_kv_heads=3,
/// head_dim=64. So q_proj is [576, 576], v_proj is [192, 576].
/// LoRA A shape: [rank, in_features], B shape: [out_features, rank].
///
/// Returns the temp dir (adapter is at dir.path()).
fn create_synthetic_adapter(rank: usize) -> tempfile::TempDir {
    use safetensors::tensor::TensorView;

    let dir = tempfile::tempdir().unwrap();

    // adapter_config.json
    let alpha = rank * 2;
    let config = format!(
        r#"{{"r": {rank}, "lora_alpha": {alpha}, "target_modules": ["q_proj", "v_proj"]}}"#,
    );
    std::fs::write(dir.path().join("adapter_config.json"), config).unwrap();

    // SmolLM-135M dimensions.
    let hidden_size = 576;
    let num_attention_heads = 9;
    let num_kv_heads = 3;
    let head_dim = 64;
    let num_layers = 30;

    // Projection output sizes: q_proj = num_attention_heads * head_dim, v_proj = num_kv_heads * head_dim.
    let proj_out_sizes: Vec<(&str, usize)> = vec![
        ("q_proj", num_attention_heads * head_dim), // 576
        ("v_proj", num_kv_heads * head_dim),        // 192
    ];

    let mut tensor_data: Vec<Vec<u8>> = Vec::new();
    let mut tensor_specs: Vec<(String, Vec<usize>)> = Vec::new();

    for layer_idx in 0..num_layers {
        for &(proj, out_size) in &proj_out_sizes {
            // lora_A: [rank, in_features=hidden_size]
            let a_name = format!(
                "base_model.model.model.layers.{}.self_attn.{}.lora_A.weight",
                layer_idx, proj
            );
            let a_data: Vec<u8> = (0..rank * hidden_size)
                .map(|i| ((i as f32 * 0.1) % 1.0) - 0.5)
                .flat_map(|f| f.to_le_bytes())
                .collect();
            tensor_specs.push((a_name, vec![rank, hidden_size]));
            tensor_data.push(a_data);

            // lora_B: [out_features, rank]
            let b_name = format!(
                "base_model.model.model.layers.{}.self_attn.{}.lora_B.weight",
                layer_idx, proj
            );
            let b_data: Vec<u8> = (0..out_size * rank)
                .map(|i| ((i as f32 * 0.2) % 1.0) - 0.5)
                .flat_map(|f| f.to_le_bytes())
                .collect();
            tensor_specs.push((b_name, vec![out_size, rank]));
            tensor_data.push(b_data);
        }
    }

    let views: Vec<(&str, TensorView<'_>)> = tensor_specs
        .iter()
        .zip(tensor_data.iter())
        .map(|((name, shape), data)| {
            (
                name.as_str(),
                TensorView::new(safetensors::Dtype::F32, shape.clone(), data).unwrap(),
            )
        })
        .collect();

    let bytes = safetensors::tensor::serialize(views, None).unwrap();
    std::fs::write(dir.path().join("adapter_model.safetensors"), bytes).unwrap();

    dir
}

// ===========================================================================
// Synthetic LoRA adapter tests (self-contained — no external deps)
// ===========================================================================

/// Test that a server with a synthetic LoRA adapter starts and serves.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_lora_synthetic_server_starts() {
    let adapter_dir = create_synthetic_adapter(4);

    let server = TestServer::builder(TestModels::SMOLLM)
        .with_lora_adapter(adapter_dir.path().to_str().unwrap())
        .start()
        .await
        .expect("server with synthetic LoRA should start");

    let client = Client::new(server.base_url());
    assert!(client.health().await.unwrap(), "server should be healthy");

    let models = client.list_models().await.unwrap();
    assert_eq!(models.data.len(), 1);
}

/// Test that chat completion works with synthetic LoRA adapter.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_lora_synthetic_chat() {
    let adapter_dir = create_synthetic_adapter(4);

    let server = TestServer::builder(TestModels::SMOLLM)
        .with_lora_adapter(adapter_dir.path().to_str().unwrap())
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = simple_chat_request("Say hello.", Some(20));
    let resp = client.chat_completion(&request).await.unwrap();

    assert_valid_chat_response(&resp);
    let text = resp.choices[0].message.content.as_deref().unwrap_or("");
    assert!(!text.is_empty(), "response should not be empty");
}

/// Test that text completion works with synthetic LoRA adapter.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_lora_synthetic_completion() {
    let adapter_dir = create_synthetic_adapter(4);

    let server = TestServer::builder(TestModels::SMOLLM)
        .with_lora_adapter(adapter_dir.path().to_str().unwrap())
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = CompletionRequest {
        prompt: Some(CompletionPrompt::Single("Once upon a time".to_string())),
        max_tokens: Some(20),
        temperature: Some(0.0),
        ..default_completion_request()
    };
    let resp = client.completion(&request).await.unwrap();

    assert_valid_completion_response(&resp);
    assert!(
        !resp.choices[0].text.is_empty(),
        "completion should not be empty"
    );
}

/// Test that LoRA adapter changes the output compared to base model.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_lora_synthetic_output_differs() {
    let adapter_dir = create_synthetic_adapter(4);
    let prompt = "The meaning of life is";

    // Base model (no LoRA).
    let base_server = TestServer::builder(TestModels::SMOLLM)
        .start()
        .await
        .unwrap();
    let base_client = Client::new(base_server.base_url());
    let request = CompletionRequest {
        prompt: Some(CompletionPrompt::Single(prompt.to_string())),
        max_tokens: Some(30),
        temperature: Some(0.0),
        ..default_completion_request()
    };
    let base_resp = base_client.completion(&request).await.unwrap();
    let base_text = base_resp.choices[0].text.clone();
    drop(base_server);

    // LoRA model (synthetic adapter).
    let lora_server = TestServer::builder(TestModels::SMOLLM)
        .with_lora_adapter(adapter_dir.path().to_str().unwrap())
        .start()
        .await
        .unwrap();
    let lora_client = Client::new(lora_server.base_url());
    let lora_resp = lora_client.completion(&request).await.unwrap();
    let lora_text = lora_resp.choices[0].text.clone();

    assert_ne!(
        base_text, lora_text,
        "LoRA output should differ from base model output.\nBase: {:?}\nLoRA: {:?}",
        base_text, lora_text
    );
}

// ===========================================================================
// CUDA LoRA tests (CudaWorker backend)
// ===========================================================================

/// Test that a CUDA server with a synthetic LoRA adapter starts and serves.
#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_lora_synthetic_server_starts() {
    let adapter_dir = create_synthetic_adapter(4);

    let server = TestServer::builder(TestModels::SMOLLM_135M_CUDA)
        .with_lora_adapter(adapter_dir.path().to_str().unwrap())
        .start()
        .await
        .expect("CUDA server with synthetic LoRA should start");

    let client = Client::new(server.base_url());
    assert!(client.health().await.unwrap(), "server should be healthy");
}

/// Test that CUDA chat completion works with synthetic LoRA adapter.
#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_lora_synthetic_chat() {
    let adapter_dir = create_synthetic_adapter(4);

    let server = TestServer::builder(TestModels::SMOLLM_135M_CUDA)
        .with_lora_adapter(adapter_dir.path().to_str().unwrap())
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = simple_chat_request("Say hello.", Some(20));
    let resp = client.chat_completion(&request).await.unwrap();

    assert_valid_chat_response(&resp);
    let text = resp.choices[0].message.content.as_deref().unwrap_or("");
    assert!(!text.is_empty(), "response should not be empty");
}

/// Test that CUDA LoRA adapter changes the output compared to base model.
#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_lora_synthetic_output_differs() {
    let adapter_dir = create_synthetic_adapter(4);
    let prompt = "The meaning of life is";

    // Base model (no LoRA).
    let base_server = TestServer::builder(TestModels::SMOLLM_135M_CUDA)
        .start()
        .await
        .unwrap();
    let base_client = Client::new(base_server.base_url());
    let request = CompletionRequest {
        prompt: Some(CompletionPrompt::Single(prompt.to_string())),
        max_tokens: Some(30),
        temperature: Some(0.0),
        ..default_completion_request()
    };
    let base_resp = base_client.completion(&request).await.unwrap();
    let base_text = base_resp.choices[0].text.clone();
    drop(base_server);

    // LoRA model (synthetic adapter).
    let lora_server = TestServer::builder(TestModels::SMOLLM_135M_CUDA)
        .with_lora_adapter(adapter_dir.path().to_str().unwrap())
        .start()
        .await
        .unwrap();
    let lora_client = Client::new(lora_server.base_url());
    let lora_resp = lora_client.completion(&request).await.unwrap();
    let lora_text = lora_resp.choices[0].text.clone();

    assert_ne!(
        base_text, lora_text,
        "CUDA LoRA output should differ from base model output.\nBase: {:?}\nLoRA: {:?}",
        base_text, lora_text
    );
}
