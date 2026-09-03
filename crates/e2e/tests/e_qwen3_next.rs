// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Qwen3-Next E2E tests with synthetic tiny model.
//!
//! Since no small public Qwen3-Next model is available, these tests generate a
//! synthetic model directory with random weights, config.json, and tokenizer.json.
//! This validates the full config→load→forward→sample pipeline.
//!
//! Run with:
//!   cargo test -p vllm-e2e --features e2e,cuda --release --test e_qwen3_next -- --ignored --test-threads=1

#![cfg(all(feature = "e2e", feature = "cuda"))]

use std::collections::HashMap;
use std::path::Path;

use scratchy_e2e::assertions::{assert_valid_chat_response, assert_valid_completion_response};
use scratchy_e2e::{Client, TestServer};
use scratchy_serving_api::protocol::{
    ChatCompletionMessageParam, ChatCompletionRequest, CompletionPrompt, CompletionRequest,
};

// ---------------------------------------------------------------------------
// Synthetic model generation
// ---------------------------------------------------------------------------

/// Create a tiny synthetic Qwen3-Next model directory.
///
/// Generates config.json, tokenizer.json, tokenizer_config.json, and
/// model.safetensors with random f32 weights matching the architecture.
fn create_synthetic_model(dir: &Path) {
    // Config: tiny Qwen3-Next with 4 layers (3 GDN + 1 full_attention),
    // small dims, 4 experts with 2 active, dense MLP on layer 0.
    let config = serde_json::json!({
        "architectures": ["Qwen3NextForCausalLM"],
        "model_type": "qwen3_next",
        "hidden_size": 64,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "num_hidden_layers": 4,
        "intermediate_size": 128,
        "vocab_size": 256,
        "max_position_embeddings": 512,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "head_dim": 16,
        "tie_word_embeddings": false,
        "partial_rotary_factor": 0.25,
        "torch_dtype": "float16",
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 8,
        "linear_value_head_dim": 8,
        "linear_num_key_heads": 4,
        "linear_num_value_heads": 4,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 32,
        "shared_expert_intermediate_size": 0,
        "norm_topk_prob": true,
        "decoder_sparse_step": 1,
        "mlp_only_layers": [0],
        "layer_types": [
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention"
        ]
    });
    std::fs::write(
        dir.join("config.json"),
        serde_json::to_string_pretty(&config).unwrap(),
    )
    .unwrap();

    // Minimal byte-level BPE tokenizer (256 single-byte tokens).
    let tokenizer = make_minimal_tokenizer();
    std::fs::write(
        dir.join("tokenizer.json"),
        serde_json::to_string_pretty(&tokenizer).unwrap(),
    )
    .unwrap();

    // Tokenizer config with chat template.
    let tokenizer_config = serde_json::json!({
        "chat_template": "{% for message in messages %}{{ message.content }}{% endfor %}",
        "bos_token": "<s>",
        "eos_token": "</s>",
        "pad_token": "<pad>"
    });
    std::fs::write(
        dir.join("tokenizer_config.json"),
        serde_json::to_string_pretty(&tokenizer_config).unwrap(),
    )
    .unwrap();

    // Generate random safetensors weights.
    let tensors = generate_weights(&config);
    let filename = dir.join("model.safetensors");
    std::fs::write(&filename, safetensors::serialize(tensors, None).unwrap()).unwrap();
}

/// Generate a minimal byte-level BPE tokenizer JSON.
fn make_minimal_tokenizer() -> serde_json::Value {
    // Build vocab: 256 byte tokens + 3 special tokens.
    let mut vocab = serde_json::Map::new();
    for i in 0u32..256 {
        let token = if i < 33 || i == 127 {
            format!("<0x{i:02X}>")
        } else {
            String::from(i as u8 as char)
        };
        vocab.insert(token, serde_json::Value::from(i));
    }
    vocab.insert("<s>".to_string(), 256.into());
    vocab.insert("</s>".to_string(), 257.into());
    vocab.insert("<pad>".to_string(), 258.into());

    serde_json::json!({
        "version": "1.0",
        "model": {
            "type": "BPE",
            "vocab": vocab,
            "merges": []
        },
        "added_tokens": [
            {"id": 256, "content": "<s>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
            {"id": 257, "content": "</s>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
            {"id": 258, "content": "<pad>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
        ]
    })
}

/// Generate random f32 weight tensors for the synthetic model.
fn generate_weights(
    config: &serde_json::Value,
) -> HashMap<String, safetensors::tensor::TensorView<'static>> {
    // We'll collect the raw data in Vec<u8> owned by a leaked Box to keep
    // the borrow checker happy with TensorView lifetimes.
    let hidden = config["hidden_size"].as_u64().unwrap() as usize;
    let vocab = config["vocab_size"].as_u64().unwrap() as usize;
    let inter = config["intermediate_size"].as_u64().unwrap() as usize;
    let n_layers = config["num_hidden_layers"].as_u64().unwrap() as usize;
    let n_heads = config["num_attention_heads"].as_u64().unwrap() as usize;
    let n_kv = config["num_key_value_heads"].as_u64().unwrap() as usize;
    let head_dim = config["head_dim"].as_u64().unwrap() as usize;
    let hk = config["linear_key_head_dim"].as_u64().unwrap() as usize;
    let hv = config["linear_value_head_dim"].as_u64().unwrap() as usize;
    let n_k_heads = config["linear_num_key_heads"].as_u64().unwrap() as usize;
    let n_v_heads = config["linear_num_value_heads"].as_u64().unwrap() as usize;
    let conv_kernel = config["linear_conv_kernel_dim"].as_u64().unwrap() as usize;
    let n_experts = config["num_experts"].as_u64().unwrap() as usize;
    let moe_inter = config["moe_intermediate_size"].as_u64().unwrap() as usize;

    let key_dim = n_k_heads * hk;
    let value_dim = n_v_heads * hv;
    let conv_dim = 2 * key_dim + value_dim;

    let layer_types: Vec<String> = config["layer_types"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();
    let mlp_only: Vec<usize> = config["mlp_only_layers"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();

    let mut tensors = HashMap::new();

    // Helper: create a random f32 tensor and leak its data for TensorView.
    fn make_tensor(shape: &[usize]) -> safetensors::tensor::TensorView<'static> {
        let numel: usize = shape.iter().product();
        // Use small random values in f16 to avoid numerical issues.
        let data: Vec<u16> = (0..numel)
            .map(|i| half::f16::from_f32((i as f32 * 0.001).sin() * 0.01).to_bits())
            .collect();
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        let leaked: &'static [u8] = Box::leak(bytes.into_boxed_slice());
        safetensors::tensor::TensorView::new(safetensors::Dtype::F16, shape.to_vec(), leaked)
            .unwrap()
    }

    // Embeddings.
    tensors.insert(
        "model.embed_tokens.weight".into(),
        make_tensor(&[vocab, hidden]),
    );
    tensors.insert("lm_head.weight".into(), make_tensor(&[vocab, hidden]));

    // Final norm.
    tensors.insert("model.norm.weight".into(), make_tensor(&[hidden]));

    for i in 0..n_layers {
        let prefix = format!("model.layers.{i}");
        let is_full_attn = layer_types[i] == "full_attention";
        let is_moe = !mlp_only.contains(&i) && n_experts > 0;

        // Layer norms.
        tensors.insert(
            format!("{prefix}.input_layernorm.weight"),
            make_tensor(&[hidden]),
        );
        tensors.insert(
            format!("{prefix}.post_attention_layernorm.weight"),
            make_tensor(&[hidden]),
        );

        if is_full_attn {
            // Full attention.
            let q_size = n_heads * head_dim * 2; // doubled for gate
            let kv_size = n_kv * head_dim;
            tensors.insert(
                format!("{prefix}.self_attn.q_proj.weight"),
                make_tensor(&[q_size, hidden]),
            );
            tensors.insert(
                format!("{prefix}.self_attn.k_proj.weight"),
                make_tensor(&[kv_size, hidden]),
            );
            tensors.insert(
                format!("{prefix}.self_attn.v_proj.weight"),
                make_tensor(&[kv_size, hidden]),
            );
            tensors.insert(
                format!("{prefix}.self_attn.o_proj.weight"),
                make_tensor(&[hidden, n_heads * head_dim]),
            );
            tensors.insert(
                format!("{prefix}.self_attn.q_norm.weight"),
                make_tensor(&[head_dim]),
            );
            tensors.insert(
                format!("{prefix}.self_attn.k_norm.weight"),
                make_tensor(&[head_dim]),
            );
        } else {
            // GDN linear attention.
            let qkvz_size = 2 * key_dim + 2 * value_dim;
            let ba_size = 2 * n_v_heads;
            tensors.insert(
                format!("{prefix}.linear_attn.in_proj_qkvz.weight"),
                make_tensor(&[qkvz_size, hidden]),
            );
            tensors.insert(
                format!("{prefix}.linear_attn.in_proj_ba.weight"),
                make_tensor(&[ba_size, hidden]),
            );
            tensors.insert(
                format!("{prefix}.linear_attn.conv1d.weight"),
                make_tensor(&[conv_dim, 1, conv_kernel]),
            );
            tensors.insert(
                format!("{prefix}.linear_attn.A_log"),
                make_tensor(&[n_v_heads]),
            );
            tensors.insert(
                format!("{prefix}.linear_attn.dt_bias"),
                make_tensor(&[n_v_heads]),
            );
            tensors.insert(
                format!("{prefix}.linear_attn.norm.weight"),
                make_tensor(&[hv]),
            );
            tensors.insert(
                format!("{prefix}.linear_attn.out_proj.weight"),
                make_tensor(&[hidden, value_dim]),
            );
        }

        if is_moe {
            // MoE MLP.
            tensors.insert(
                format!("{prefix}.mlp.gate.weight"),
                make_tensor(&[n_experts, hidden]),
            );
            for e in 0..n_experts {
                tensors.insert(
                    format!("{prefix}.mlp.experts.{e}.gate_proj.weight"),
                    make_tensor(&[moe_inter, hidden]),
                );
                tensors.insert(
                    format!("{prefix}.mlp.experts.{e}.up_proj.weight"),
                    make_tensor(&[moe_inter, hidden]),
                );
                tensors.insert(
                    format!("{prefix}.mlp.experts.{e}.down_proj.weight"),
                    make_tensor(&[hidden, moe_inter]),
                );
            }
        } else {
            // Dense MLP.
            tensors.insert(
                format!("{prefix}.mlp.gate_proj.weight"),
                make_tensor(&[inter, hidden]),
            );
            tensors.insert(
                format!("{prefix}.mlp.up_proj.weight"),
                make_tensor(&[inter, hidden]),
            );
            tensors.insert(
                format!("{prefix}.mlp.down_proj.weight"),
                make_tensor(&[hidden, inter]),
            );
        }
    }

    tensors
}

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
        ..serde_json::from_str(r#"{"messages": []}"#).unwrap()
    }
}

fn simple_completion_request(prompt: &str, max_tokens: u32) -> CompletionRequest {
    CompletionRequest {
        prompt: Some(CompletionPrompt::Single(prompt.to_string())),
        max_tokens: Some(max_tokens),
        temperature: Some(0.0),
        ..serde_json::from_str(r#"{}"#).unwrap()
    }
}

// ===========================================================================
// Tests
// ===========================================================================

/// Verify the synthetic model loads, the server starts, and responds to health/models.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_qwen3_next_server_starts() {
    let tmp = tempfile::tempdir().unwrap();
    create_synthetic_model(tmp.path());

    let server = TestServer::builder(tmp.path().to_str().unwrap())
        .with_device("cuda")
        .with_dtype("f16")
        .start()
        .await
        .expect("Qwen3-Next synthetic server should start");

    let client = Client::new(server.base_url());
    assert!(client.health().await.unwrap(), "server should be healthy");

    let models = client.list_models().await.unwrap();
    assert_eq!(models.data.len(), 1);
}

/// Verify a basic chat completion request succeeds.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_qwen3_next_chat_basic() {
    let tmp = tempfile::tempdir().unwrap();
    create_synthetic_model(tmp.path());

    let server = TestServer::builder(tmp.path().to_str().unwrap())
        .with_device("cuda")
        .with_dtype("f16")
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = simple_chat_request("Hello", Some(10));
    let resp = client.chat_completion(&request).await.unwrap();

    assert_valid_chat_response(&resp);
    // With random weights, we just verify the pipeline completes.
    assert!(
        resp.choices[0].message.content.is_some(),
        "response should have content"
    );
}

/// Verify a basic completion request succeeds.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_qwen3_next_completion_basic() {
    let tmp = tempfile::tempdir().unwrap();
    create_synthetic_model(tmp.path());

    let server = TestServer::builder(tmp.path().to_str().unwrap())
        .with_device("cuda")
        .with_dtype("f16")
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = simple_completion_request("Hello world", 10);
    let resp = client.completion(&request).await.unwrap();

    assert_valid_completion_response(&resp);
    assert!(
        !resp.choices[0].text.is_empty(),
        "completion should not be empty"
    );
}
