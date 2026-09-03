// SPDX-License-Identifier: Apache-2.0
//! E2E tests for FP8 KV cache quantization.
//!
//! Run with:
//!   cargo test -p vllm-e2e --features e2e,cuda --release --test e_fp8_kv -- --ignored --test-threads=1

#![cfg(all(feature = "e2e", feature = "cuda"))]

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

fn default_chat_request() -> ChatCompletionRequest {
    serde_json::from_str(r#"{"messages": []}"#).unwrap()
}

fn simple_chat_request(content: &str, max_tokens: Option<u32>) -> ChatCompletionRequest {
    ChatCompletionRequest {
        messages: vec![user_msg(content)],
        max_tokens,
        temperature: Some(0.0),
        ..default_chat_request()
    }
}

fn default_completion_request() -> CompletionRequest {
    serde_json::from_str(r#"{}"#).unwrap()
}

fn simple_completion_request(prompt: &str, max_tokens: u32) -> CompletionRequest {
    CompletionRequest {
        prompt: Some(CompletionPrompt::Single(prompt.to_string())),
        max_tokens: Some(max_tokens),
        temperature: Some(0.0),
        ..default_completion_request()
    }
}

/// Fetch num_gpu_blocks from the server_info endpoint.
async fn get_num_gpu_blocks(client: &Client) -> u64 {
    let info = client.server_info(Some("json")).await.expect("server_info");
    info.scratchy_core_config["num_gpu_blocks"]
        .as_u64()
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// FP8 KV cache server starts and passes health check.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_fp8_kv_server_starts() {
    let server = TestServer::builder(TestModels::QWEN2_0_5B_CUDA)
        .with_args(&["--kv-cache-dtype", "fp8_e4m3", "--enforce-eager"])
        .start()
        .await
        .expect("FP8 KV cache server should start");

    let client = Client::new(server.base_url());
    assert!(client.health().await.unwrap(), "server should be healthy");
}

/// FP8 KV cache generates a non-empty completion.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_fp8_kv_completion() {
    let server = TestServer::builder(TestModels::QWEN2_0_5B_CUDA)
        .with_args(&["--kv-cache-dtype", "fp8_e4m3", "--enforce-eager"])
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = simple_completion_request("The meaning of life is", 20);
    let resp = client.completion(&request).await.unwrap();

    assert_valid_completion_response(&resp);
    assert!(
        !resp.choices[0].text.is_empty(),
        "FP8 KV completion should not be empty"
    );
}

/// FP8 KV cache with dynamic scale computation doesn't crash.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_fp8_kv_dynamic_scales() {
    let server = TestServer::builder(TestModels::QWEN2_0_5B_CUDA)
        .with_args(&[
            "--kv-cache-dtype",
            "fp8_e4m3",
            "--calculate-kv-scales",
            "--enforce-eager",
        ])
        .start()
        .await
        .expect("FP8 KV with dynamic scales should start");

    let client = Client::new(server.base_url());
    let request = simple_completion_request("Hello world", 10);
    let resp = client.completion(&request).await.unwrap();

    assert_valid_completion_response(&resp);
    assert!(
        !resp.choices[0].text.is_empty(),
        "dynamic scales completion should not be empty"
    );
}

/// Multi-turn chat exercises the decode dequant path.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_fp8_kv_multi_turn_chat() {
    let server = TestServer::builder(TestModels::QWEN2_0_5B_CUDA)
        .with_args(&["--kv-cache-dtype", "fp8_e4m3", "--enforce-eager"])
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());

    // Turn 1
    let req1 = simple_chat_request("What is 2+2?", Some(30));
    let resp1 = client.chat_completion(&req1).await.unwrap();
    assert_valid_chat_response(&resp1);
    let text1 = resp1.choices[0].message.content.as_deref().unwrap_or("");
    assert!(!text1.is_empty(), "turn 1 should not be empty");

    // Turn 2 (new request — exercises decode path with cached KV)
    let req2 = simple_chat_request("What is 3+3?", Some(30));
    let resp2 = client.chat_completion(&req2).await.unwrap();
    assert_valid_chat_response(&resp2);
    let text2 = resp2.choices[0].message.content.as_deref().unwrap_or("");
    assert!(!text2.is_empty(), "turn 2 should not be empty");

    // Turn 3
    let req3 = simple_chat_request("What is 4+4?", Some(30));
    let resp3 = client.chat_completion(&req3).await.unwrap();
    assert_valid_chat_response(&resp3);
    let text3 = resp3.choices[0].message.content.as_deref().unwrap_or("");
    assert!(!text3.is_empty(), "turn 3 should not be empty");
}

// ---------------------------------------------------------------------------
// CUDA graph tests (no --enforce-eager)
// ---------------------------------------------------------------------------

/// FP8 KV cache with CUDA graphs: server starts and captures graphs.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_fp8_kv_graph_server_starts() {
    let server = TestServer::builder(TestModels::QWEN2_0_5B_CUDA)
        .with_args(&["--kv-cache-dtype", "fp8_e4m3"])
        .start()
        .await
        .expect("FP8 KV + CUDA graph server should start");

    let client = Client::new(server.base_url());
    assert!(client.health().await.unwrap(), "server should be healthy");
}

/// FP8 KV cache with CUDA graphs: single completion.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_fp8_kv_graph_completion() {
    let server = TestServer::builder(TestModels::QWEN2_0_5B_CUDA)
        .with_args(&["--kv-cache-dtype", "fp8_e4m3"])
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = simple_completion_request("The meaning of life is", 20);
    let resp = client.completion(&request).await.unwrap();

    assert_valid_completion_response(&resp);
    assert!(
        !resp.choices[0].text.is_empty(),
        "FP8 KV graph completion should not be empty"
    );
}

/// FP8 KV cache with CUDA graphs: multi-turn chat exercises decode dequant
/// through the graph replay path (both full replay and fast replay).
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_fp8_kv_graph_multi_turn() {
    let server = TestServer::builder(TestModels::QWEN2_0_5B_CUDA)
        .with_args(&["--kv-cache-dtype", "fp8_e4m3"])
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());

    // Turn 1
    let req1 = simple_chat_request("What is 2+2?", Some(30));
    let resp1 = client.chat_completion(&req1).await.unwrap();
    assert_valid_chat_response(&resp1);
    let text1 = resp1.choices[0].message.content.as_deref().unwrap_or("");
    assert!(!text1.is_empty(), "turn 1 should not be empty");

    // Turn 2 (exercises steady-state graph decode with growing seqused_k)
    let req2 = simple_chat_request("What is 3+3?", Some(30));
    let resp2 = client.chat_completion(&req2).await.unwrap();
    assert_valid_chat_response(&resp2);
    let text2 = resp2.choices[0].message.content.as_deref().unwrap_or("");
    assert!(!text2.is_empty(), "turn 2 should not be empty");

    // Turn 3
    let req3 = simple_chat_request("What is 4+4?", Some(30));
    let resp3 = client.chat_completion(&req3).await.unwrap();
    assert_valid_chat_response(&resp3);
    let text3 = resp3.choices[0].message.content.as_deref().unwrap_or("");
    assert!(!text3.is_empty(), "turn 3 should not be empty");
}

/// FP8 KV cache with CUDA graphs: concurrent requests exercises batch > 1
/// decode through the graph, triggering batch-padded replay.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_fp8_kv_graph_concurrent() {
    let server = TestServer::builder(TestModels::QWEN2_0_5B_CUDA)
        .with_args(&["--kv-cache-dtype", "fp8_e4m3"])
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());

    // Launch 3 concurrent completions to exercise batch > 1 decode graphs.
    let futs: Vec<_> = (0..3)
        .map(|i| {
            let c = Client::new(server.base_url());
            let prompt = format!("Count from {i} to ten:");
            tokio::spawn(async move {
                let req = CompletionRequest {
                    prompt: Some(CompletionPrompt::Single(prompt)),
                    max_tokens: Some(20),
                    temperature: Some(0.0),
                    ..serde_json::from_str(r#"{}"#).unwrap()
                };
                c.completion(&req).await
            })
        })
        .collect();

    for (i, fut) in futs.into_iter().enumerate() {
        let resp = fut.await.unwrap().unwrap();
        assert_valid_completion_response(&resp);
        assert!(
            !resp.choices[0].text.is_empty(),
            "concurrent request {i} should not be empty"
        );
    }
}

// ---------------------------------------------------------------------------
// Block count comparison
// ---------------------------------------------------------------------------

/// FP8 KV cache should allocate approximately 2x the blocks of BF16.
/// We compare the server_info endpoint's num_gpu_blocks between the two.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_fp8_kv_more_blocks() {
    // Start BF16 server
    let bf16_server = TestServer::builder(TestModels::QWEN2_0_5B_CUDA)
        .with_args(&["--enforce-eager"])
        .start()
        .await
        .expect("BF16 server should start");

    let bf16_client = Client::new(bf16_server.base_url());
    let bf16_blocks = get_num_gpu_blocks(&bf16_client).await;

    // Drop BF16 server to free GPU memory
    drop(bf16_server);
    tokio::time::sleep(std::time::Duration::from_secs(2)).await;

    // Start FP8 server
    let fp8_server = TestServer::builder(TestModels::QWEN2_0_5B_CUDA)
        .with_args(&["--kv-cache-dtype", "fp8_e4m3", "--enforce-eager"])
        .start()
        .await
        .expect("FP8 server should start");

    let fp8_client = Client::new(fp8_server.base_url());
    let fp8_blocks = get_num_gpu_blocks(&fp8_client).await;

    // FP8 should have roughly 2x the blocks (allow 1.4x+ range
    // due to activation memory and other overheads).
    if bf16_blocks > 0 && fp8_blocks > 0 {
        let ratio = fp8_blocks as f64 / bf16_blocks as f64;
        assert!(
            ratio > 1.4,
            "FP8 should have more blocks than BF16: fp8={fp8_blocks}, bf16={bf16_blocks}, ratio={ratio:.2}"
        );
    }
}
