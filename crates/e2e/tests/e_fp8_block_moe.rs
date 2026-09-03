// SPDX-License-Identifier: Apache-2.0

//! E2E tests for FP8 block-quantized MoE (Qwen3MoeForCausalLM with weight_block_size).
//!
//! Generates a synthetic tiny Qwen3 MoE model with FP8 E4M3 weights and 2D
//! block scales (`weight_scale_inv`). This tests the full pipeline:
//! config parsing → load_fp8_block → block dequant MoE experts → forward.
//!
//! Run with:
//!   cargo test -p vllm-e2e --features e2e,cuda --release --test e_fp8_block_moe -- --ignored --test-threads=1

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

/// Block size for FP8 quantization. All dimensions must be divisible by this.
const BLOCK_SIZE: usize = 128;

/// Create a tiny synthetic Qwen3 MoE model with FP8 block quantization.
///
/// Attention projections and MoE expert weights are stored as FP8 E4M3 with
/// 2D `weight_scale_inv` tensors. Layernorms, gate (router), and lm_head
/// stay BF16/F16 (matching the real checkpoint's `modules_to_not_convert`).
fn create_synthetic_model(dir: &Path) {
    let config = serde_json::json!({
        "architectures": ["Qwen3MoeForCausalLM"],
        "model_type": "qwen3_moe",
        "hidden_size": 256,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "num_hidden_layers": 2,
        "intermediate_size": 256,
        "vocab_size": 256,
        "max_position_embeddings": 512,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "head_dim": 64,
        "tie_word_embeddings": false,
        "torch_dtype": "bfloat16",
        "attention_bias": false,
        "attention_dropout": 0.0,
        // MoE config
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 128,
        "shared_expert_intermediate_size": 0,
        "norm_topk_prob": true,
        "decoder_sparse_step": 1,
        "mlp_only_layers": [],
        // FP8 block quantization config
        "quantization_config": {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "fmt": "e4m3",
            "weight_block_size": [BLOCK_SIZE, BLOCK_SIZE],
            "modules_to_not_convert": []
        }
    });
    std::fs::write(
        dir.join("config.json"),
        serde_json::to_string_pretty(&config).unwrap(),
    )
    .unwrap();

    // Minimal tokenizer.
    let tokenizer = make_minimal_tokenizer();
    std::fs::write(
        dir.join("tokenizer.json"),
        serde_json::to_string_pretty(&tokenizer).unwrap(),
    )
    .unwrap();

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

    let tensors = generate_weights(&config);
    let filename = dir.join("model.safetensors");
    std::fs::write(&filename, safetensors::serialize(tensors, None).unwrap()).unwrap();
}

fn make_minimal_tokenizer() -> serde_json::Value {
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

/// Create a BF16 tensor with small deterministic values.
fn make_bf16_tensor(shape: &[usize]) -> safetensors::tensor::TensorView<'static> {
    let numel: usize = shape.iter().product();
    let data: Vec<u16> = (0..numel)
        .map(|i| half::bf16::from_f32((i as f32 * 0.001).sin() * 0.01).to_bits())
        .collect();
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    let leaked: &'static [u8] = Box::leak(bytes.into_boxed_slice());
    safetensors::tensor::TensorView::new(safetensors::Dtype::BF16, shape.to_vec(), leaked).unwrap()
}

/// Create an FP8 E4M3 tensor with small deterministic values.
fn make_fp8_tensor(shape: &[usize]) -> safetensors::tensor::TensorView<'static> {
    let numel: usize = shape.iter().product();
    // Fill with small FP8 values (just use low byte values; the exact
    // numerical content doesn't matter for pipeline testing).
    let data: Vec<u8> = (0..numel).map(|i| ((i % 127) + 1) as u8).collect();
    let leaked: &'static [u8] = Box::leak(data.into_boxed_slice());
    safetensors::tensor::TensorView::new(safetensors::Dtype::F8_E4M3, shape.to_vec(), leaked)
        .unwrap()
}

/// Create an f32 block-scale tensor (all 1.0 so dequant is identity-ish).
fn make_block_scale(n: usize, k: usize) -> safetensors::tensor::TensorView<'static> {
    let scale_rows = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    let scale_cols = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;
    let numel = scale_rows * scale_cols;
    let data: Vec<f32> = vec![1.0f32; numel];
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    let leaked: &'static [u8] = Box::leak(bytes.into_boxed_slice());
    safetensors::tensor::TensorView::new(
        safetensors::Dtype::F32,
        vec![scale_rows, scale_cols],
        leaked,
    )
    .unwrap()
}

/// Insert an FP8 weight + its block scale_inv into the tensor map.
fn insert_fp8_weight(
    tensors: &mut HashMap<String, safetensors::tensor::TensorView<'static>>,
    name: &str,
    shape: &[usize],
) {
    let n = shape[0];
    let k = shape[1];
    tensors.insert(format!("{name}.weight"), make_fp8_tensor(shape));
    tensors.insert(format!("{name}.weight_scale_inv"), make_block_scale(n, k));
}

fn generate_weights(
    config: &serde_json::Value,
) -> HashMap<String, safetensors::tensor::TensorView<'static>> {
    let hidden = config["hidden_size"].as_u64().unwrap() as usize;
    let vocab = config["vocab_size"].as_u64().unwrap() as usize;
    let n_layers = config["num_hidden_layers"].as_u64().unwrap() as usize;
    let n_heads = config["num_attention_heads"].as_u64().unwrap() as usize;
    let n_kv = config["num_key_value_heads"].as_u64().unwrap() as usize;
    let head_dim = config["head_dim"].as_u64().unwrap() as usize;
    let n_experts = config["num_experts"].as_u64().unwrap() as usize;
    let moe_inter = config["moe_intermediate_size"].as_u64().unwrap() as usize;

    let q_size = n_heads * head_dim;
    let kv_size = n_kv * head_dim;

    let mut tensors = HashMap::new();

    // Embeddings + lm_head stay BF16.
    tensors.insert(
        "model.embed_tokens.weight".into(),
        make_bf16_tensor(&[vocab, hidden]),
    );
    tensors.insert("lm_head.weight".into(), make_bf16_tensor(&[vocab, hidden]));
    tensors.insert("model.norm.weight".into(), make_bf16_tensor(&[hidden]));

    for i in 0..n_layers {
        let prefix = format!("model.layers.{i}");

        // Layernorms stay BF16.
        tensors.insert(
            format!("{prefix}.input_layernorm.weight"),
            make_bf16_tensor(&[hidden]),
        );
        tensors.insert(
            format!("{prefix}.post_attention_layernorm.weight"),
            make_bf16_tensor(&[hidden]),
        );

        // QK-norm weights (BF16).
        tensors.insert(
            format!("{prefix}.self_attn.q_norm.weight"),
            make_bf16_tensor(&[head_dim]),
        );
        tensors.insert(
            format!("{prefix}.self_attn.k_norm.weight"),
            make_bf16_tensor(&[head_dim]),
        );

        // Attention projections: FP8 block-quantized.
        insert_fp8_weight(
            &mut tensors,
            &format!("{prefix}.self_attn.q_proj"),
            &[q_size, hidden],
        );
        insert_fp8_weight(
            &mut tensors,
            &format!("{prefix}.self_attn.k_proj"),
            &[kv_size, hidden],
        );
        insert_fp8_weight(
            &mut tensors,
            &format!("{prefix}.self_attn.v_proj"),
            &[kv_size, hidden],
        );
        insert_fp8_weight(
            &mut tensors,
            &format!("{prefix}.self_attn.o_proj"),
            &[hidden, q_size],
        );

        // MoE gate (router) stays BF16.
        tensors.insert(
            format!("{prefix}.mlp.gate.weight"),
            make_bf16_tensor(&[n_experts, hidden]),
        );

        // Expert weights: FP8 block-quantized.
        for e in 0..n_experts {
            let ep = format!("{prefix}.mlp.experts.{e}");
            insert_fp8_weight(
                &mut tensors,
                &format!("{ep}.gate_proj"),
                &[moe_inter, hidden],
            );
            insert_fp8_weight(&mut tensors, &format!("{ep}.up_proj"), &[moe_inter, hidden]);
            insert_fp8_weight(
                &mut tensors,
                &format!("{ep}.down_proj"),
                &[hidden, moe_inter],
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

/// FP8 block-quantized MoE server starts and passes health check.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_fp8_block_moe_server_starts() {
    let tmp = tempfile::tempdir().unwrap();
    create_synthetic_model(tmp.path());

    let server = TestServer::builder(tmp.path().to_str().unwrap())
        .with_device("cuda")
        .with_args(&["--enforce-eager"])
        .start()
        .await
        .expect("FP8 block MoE server should start");

    let client = Client::new(server.base_url());
    assert!(client.health().await.unwrap(), "server should be healthy");

    let models = client.list_models().await.unwrap();
    assert_eq!(models.data.len(), 1);
}

/// FP8 block-quantized MoE completion produces non-empty output.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_fp8_block_moe_completion() {
    let tmp = tempfile::tempdir().unwrap();
    create_synthetic_model(tmp.path());

    let server = TestServer::builder(tmp.path().to_str().unwrap())
        .with_device("cuda")
        .with_args(&["--enforce-eager"])
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = simple_completion_request("Hello world", 10);
    let resp = client.completion(&request).await.unwrap();

    assert_valid_completion_response(&resp);
    assert!(
        !resp.choices[0].text.is_empty(),
        "FP8 block MoE completion should produce non-empty output"
    );
}

/// FP8 block-quantized MoE chat produces non-empty response.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_fp8_block_moe_chat() {
    let tmp = tempfile::tempdir().unwrap();
    create_synthetic_model(tmp.path());

    let server = TestServer::builder(tmp.path().to_str().unwrap())
        .with_device("cuda")
        .with_args(&["--enforce-eager"])
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = simple_chat_request("Hello", Some(10));
    let resp = client.chat_completion(&request).await.unwrap();

    assert_valid_chat_response(&resp);
    assert!(
        resp.choices[0].message.content.is_some(),
        "response should have content"
    );
}
