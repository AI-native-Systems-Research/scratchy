// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! E2E tests for GET /server_info endpoint.
//!
//! Run with: `cargo test -p vllm-e2e --features e2e --test e_server_info -- --ignored`

#![cfg(feature = "e2e")]

use scratchy_e2e::{Client, TestModels, TestServer};

// ---------------------------------------------------------------------------
// GET /server_info — default (text) format
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_server_info_default() {
    let server = TestServer::builder(TestModels::SMOLLM)
        .start()
        .await
        .expect("server should start");

    let client = Client::new(server.base_url());
    let resp = client.server_info(None).await.unwrap();

    // vllm_env is an object (may be empty if no VLLM_ vars are set).
    assert!(resp.vllm_env.is_object(), "vllm_env should be an object");

    // system_env should contain os and vllm_version.
    let sys = resp
        .system_env
        .as_object()
        .expect("system_env should be an object");
    assert!(sys.contains_key("os"), "system_env should contain 'os'");
    assert!(
        sys.contains_key("vllm_version"),
        "system_env should contain 'vllm_version'"
    );
}

// ---------------------------------------------------------------------------
// GET /server_info?config_format=json
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_server_info_json_format() {
    let server = TestServer::builder(TestModels::SMOLLM)
        .start()
        .await
        .expect("server should start");

    let client = Client::new(server.base_url());
    let resp = client.server_info(Some("json")).await.unwrap();

    // In E2E the test server doesn't pass VllmConfig, so scratchy_core_config is null.
    // But the response shape should still be valid.
    assert!(resp.vllm_env.is_object());
    assert!(resp.system_env.is_object());

    let sys = resp.system_env.as_object().unwrap();
    let version = sys["vllm_version"].as_str().unwrap();
    assert!(!version.is_empty(), "vllm_version should not be empty");
}
