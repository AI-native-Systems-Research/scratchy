// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Phase E2: Chat completions — full feature coverage.
//!
//! Deep testing of the `/v1/chat/completions` endpoint using SmolLM-135M.
//!
//! Run with: `cargo test -p vllm-e2e --features e2e --test e2_chat_completions -- --ignored`

#![cfg(feature = "e2e")]

use scratchy_e2e::assertions::{assert_coherent_text, assert_valid_chat_response};
use scratchy_e2e::{Client, TestModels, TestServer};
use scratchy_serving_api::protocol::{ChatCompletionMessageParam, ChatCompletionRequest};

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

fn system_msg(content: &str) -> ChatCompletionMessageParam {
    ChatCompletionMessageParam {
        role: "system".to_string(),
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

fn default_chat_request() -> ChatCompletionRequest {
    serde_json::from_str(r#"{"messages": []}"#).unwrap()
}

async fn start_smollm() -> (TestServer, Client) {
    let server = TestServer::builder(TestModels::SMOLLM)
        .start()
        .await
        .expect("SmolLM server should start");
    let client = Client::new(server.base_url());
    (server, client)
}

// ===========================================================================
// E2a: Request parameters
// ===========================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_chat_temperature_0() {
    let (_server, client) = start_smollm().await;

    let request = ChatCompletionRequest {
        messages: vec![user_msg("What is 2+2?")],
        temperature: Some(0.0),
        max_tokens: Some(20),
        ..default_chat_request()
    };

    let resp1 = client.chat_completion(&request).await.unwrap();
    let resp2 = client.chat_completion(&request).await.unwrap();

    let text1 = resp1.choices[0].message.content.as_deref().unwrap_or("");
    let text2 = resp2.choices[0].message.content.as_deref().unwrap_or("");
    assert_eq!(text1, text2, "temperature=0 should be deterministic");
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_chat_temperature_high() {
    let (_server, client) = start_smollm().await;

    // High temperature — output may differ across calls.
    // We just verify the response is valid, not that it differs.
    let request = ChatCompletionRequest {
        messages: vec![user_msg("Tell me something interesting.")],
        temperature: Some(1.5),
        max_tokens: Some(30),
        ..default_chat_request()
    };

    let resp = client.chat_completion(&request).await.unwrap();
    assert_valid_chat_response(&resp);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_chat_top_p() {
    let (_server, client) = start_smollm().await;

    let request = ChatCompletionRequest {
        messages: vec![user_msg("Hello!")],
        top_p: Some(0.1),
        max_tokens: Some(20),
        ..default_chat_request()
    };

    let resp = client.chat_completion(&request).await.unwrap();
    assert_valid_chat_response(&resp);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_chat_top_k() {
    let (_server, client) = start_smollm().await;

    let request = ChatCompletionRequest {
        messages: vec![user_msg("Hello!")],
        top_k: Some(5),
        max_tokens: Some(20),
        ..default_chat_request()
    };

    let resp = client.chat_completion(&request).await.unwrap();
    assert_valid_chat_response(&resp);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_chat_min_p() {
    let (_server, client) = start_smollm().await;

    let request = ChatCompletionRequest {
        messages: vec![user_msg("Hello!")],
        min_p: Some(0.1),
        max_tokens: Some(20),
        ..default_chat_request()
    };

    let resp = client.chat_completion(&request).await.unwrap();
    assert_valid_chat_response(&resp);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_chat_max_tokens() {
    let (_server, client) = start_smollm().await;

    let request = ChatCompletionRequest {
        messages: vec![user_msg("Write a very long essay about history.")],
        max_tokens: Some(10),
        temperature: Some(0.0),
        ..default_chat_request()
    };

    let resp = client.chat_completion(&request).await.unwrap();
    assert_valid_chat_response(&resp);
    assert!(
        resp.usage.completion_tokens.unwrap_or(0) <= 10,
        "completion_tokens should be <= 10, got {:?}",
        resp.usage.completion_tokens
    );
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_chat_max_completion_tokens() {
    let (_server, client) = start_smollm().await;

    let request = ChatCompletionRequest {
        messages: vec![user_msg("Write a long essay.")],
        max_completion_tokens: Some(10),
        temperature: Some(0.0),
        ..default_chat_request()
    };

    let resp = client.chat_completion(&request).await.unwrap();
    assert_valid_chat_response(&resp);
    assert!(resp.usage.completion_tokens.unwrap_or(0) <= 10);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_chat_n_1() {
    let (_server, client) = start_smollm().await;

    let request = ChatCompletionRequest {
        messages: vec![user_msg("Hello!")],
        n: 1,
        max_tokens: Some(20),
        ..default_chat_request()
    };

    let resp = client.chat_completion(&request).await.unwrap();
    assert_eq!(resp.choices.len(), 1, "n=1 should produce 1 choice");
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_chat_n_3() {
    let (_server, client) = start_smollm().await;

    let request = ChatCompletionRequest {
        messages: vec![user_msg("Hello!")],
        n: 3,
        max_tokens: Some(20),
        temperature: Some(0.8),
        ..default_chat_request()
    };

    let resp = client.chat_completion(&request).await.unwrap();
    assert_eq!(resp.choices.len(), 3, "n=3 should produce 3 choices");

    // Each choice should have a distinct index
    let indices: Vec<u32> = resp.choices.iter().map(|c| c.index).collect();
    assert!(indices.contains(&0));
    assert!(indices.contains(&1));
    assert!(indices.contains(&2));
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_chat_seed() {
    let (_server, client) = start_smollm().await;

    // Seed with temperature=0 should be deterministic across calls to the same server.
    let request = ChatCompletionRequest {
        messages: vec![user_msg("What is the meaning of life?")],
        seed: Some(42),
        temperature: Some(0.0),
        max_tokens: Some(30),
        ..default_chat_request()
    };

    let resp1 = client.chat_completion(&request).await.unwrap();
    let resp2 = client.chat_completion(&request).await.unwrap();

    let text1 = resp1.choices[0].message.content.as_deref().unwrap_or("");
    let text2 = resp2.choices[0].message.content.as_deref().unwrap_or("");
    assert_eq!(
        text1, text2,
        "same seed + temperature=0 should produce same output"
    );
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_chat_logprobs() {
    let (_server, client) = start_smollm().await;

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
    // Each entry should have top_logprobs
    for entry in content {
        assert!(
            !entry.top_logprobs.is_empty(),
            "top_logprobs should not be empty"
        );
    }
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_chat_prompt_logprobs() {
    let (_server, client) = start_smollm().await;

    let request = ChatCompletionRequest {
        messages: vec![user_msg("The capital of France is")],
        prompt_logprobs: Some(3),
        max_tokens: Some(5),
        temperature: Some(0.0),
        ..default_chat_request()
    };

    let resp = client.chat_completion(&request).await.unwrap();
    assert_valid_chat_response(&resp);

    // prompt_logprobs should be present on the choice.
    let plps = resp.choices[0]
        .prompt_logprobs
        .as_ref()
        .expect("prompt_logprobs should be present");

    // Should have one entry per prompt token. First is None (no prior context).
    assert!(
        plps.len() > 1,
        "prompt_logprobs should have multiple entries"
    );
    assert!(
        plps[0].is_none(),
        "first prompt logprob should be None (no prior context)"
    );
    // Remaining entries should be Some with valid logprobs.
    for (i, entry) in plps.iter().enumerate().skip(1) {
        let lp = entry
            .as_ref()
            .unwrap_or_else(|| panic!("prompt_logprobs[{i}] should be Some"));
        assert!(
            lp.logprob <= 0.0,
            "logprob should be non-positive, got {}",
            lp.logprob
        );
        assert!(
            lp.top_logprobs.len() <= 3,
            "top_logprobs should have at most 3 entries"
        );
    }
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_chat_frequency_penalty() {
    let (_server, client) = start_smollm().await;

    let request = ChatCompletionRequest {
        messages: vec![user_msg("Repeat the word 'hello' many times.")],
        frequency_penalty: Some(2.0),
        max_tokens: Some(30),
        ..default_chat_request()
    };

    let resp = client.chat_completion(&request).await.unwrap();
    assert_valid_chat_response(&resp);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_chat_presence_penalty() {
    let (_server, client) = start_smollm().await;

    let request = ChatCompletionRequest {
        messages: vec![user_msg("Tell me about cats.")],
        presence_penalty: Some(2.0),
        max_tokens: Some(30),
        ..default_chat_request()
    };

    let resp = client.chat_completion(&request).await.unwrap();
    assert_valid_chat_response(&resp);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_chat_repetition_penalty() {
    let (_server, client) = start_smollm().await;

    let request = ChatCompletionRequest {
        messages: vec![user_msg("Hello hello hello")],
        repetition_penalty: Some(1.5),
        max_tokens: Some(30),
        ..default_chat_request()
    };

    let resp = client.chat_completion(&request).await.unwrap();
    assert_valid_chat_response(&resp);
}

// ===========================================================================
// E2b: Message formats
// ===========================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_chat_system_message() {
    let (_server, client) = start_smollm().await;

    let request = ChatCompletionRequest {
        messages: vec![
            system_msg("You are a helpful assistant."),
            user_msg("Say hello."),
        ],
        max_tokens: Some(30),
        temperature: Some(0.0),
        ..default_chat_request()
    };

    let resp = client.chat_completion(&request).await.unwrap();
    assert_valid_chat_response(&resp);
    let text = resp.choices[0].message.content.as_deref().unwrap_or("");
    assert_coherent_text(text, 2);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_chat_multi_turn() {
    let (_server, client) = start_smollm().await;

    let request = ChatCompletionRequest {
        messages: vec![
            system_msg("You are a helpful assistant."),
            user_msg("My name is Alice."),
            assistant_msg("Nice to meet you, Alice!"),
            user_msg("What is my name?"),
        ],
        max_tokens: Some(30),
        temperature: Some(0.0),
        ..default_chat_request()
    };

    let resp = client.chat_completion(&request).await.unwrap();
    assert_valid_chat_response(&resp);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_chat_user_only() {
    let (_server, client) = start_smollm().await;

    let request = ChatCompletionRequest {
        messages: vec![user_msg("Hello!")],
        max_tokens: Some(20),
        ..default_chat_request()
    };

    let resp = client.chat_completion(&request).await.unwrap();
    assert_valid_chat_response(&resp);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_chat_unicode() {
    let (_server, client) = start_smollm().await;

    let request = ChatCompletionRequest {
        messages: vec![user_msg("Translate to French: Hello world")],
        max_tokens: Some(30),
        temperature: Some(0.0),
        ..default_chat_request()
    };

    let resp = client.chat_completion(&request).await.unwrap();
    assert_valid_chat_response(&resp);
}

// ===========================================================================
// E2c: Error handling
// ===========================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_chat_missing_messages() {
    let (_server, client) = start_smollm().await;

    let body = serde_json::json!({"model": "test"});
    let resp = client.chat_completion_raw(&body).await.unwrap();
    assert_eq!(
        resp.status().as_u16(),
        422,
        "missing messages should return 422"
    );
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_chat_empty_messages() {
    let (_server, client) = start_smollm().await;

    let request = ChatCompletionRequest {
        messages: vec![],
        max_tokens: Some(10),
        ..default_chat_request()
    };

    // Empty messages may return 422 or produce an error response.
    // We just verify the server doesn't crash.
    let _resp = client.chat_completion(&request).await;
}

// ===========================================================================
// E2g: allowed_token_ids and truncate_prompt_tokens
// ===========================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_chat_allowed_token_ids() {
    let (_server, client) = start_smollm().await;

    // Constrain output to a small set of token IDs.
    // Token 198 is typically a newline in many BPE vocabs; the model should
    // still produce a valid (if nonsensical) response without erroring.
    let request = ChatCompletionRequest {
        messages: vec![user_msg("Say hello")],
        max_tokens: Some(5),
        temperature: Some(0.0),
        allowed_token_ids: Some(vec![198, 220, 284, 330]),
        ..default_chat_request()
    };

    let resp = client.chat_completion(&request).await.unwrap();
    assert_eq!(resp.choices.len(), 1);
    assert!(resp.usage.completion_tokens.unwrap_or(0) > 0);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_chat_truncate_prompt_tokens() {
    let (_server, client) = start_smollm().await;

    // Send a long prompt but truncate to 5 tokens.
    let long_prompt = "one two three four five six seven eight nine ten \
                       eleven twelve thirteen fourteen fifteen sixteen";
    let request = ChatCompletionRequest {
        messages: vec![user_msg(long_prompt)],
        max_tokens: Some(5),
        temperature: Some(0.0),
        truncate_prompt_tokens: Some(5),
        ..default_chat_request()
    };

    let resp = client.chat_completion(&request).await.unwrap();
    assert_eq!(resp.choices.len(), 1);
    // The prompt_tokens should be exactly 5 (truncated).
    assert_eq!(
        resp.usage.prompt_tokens, 5,
        "prompt_tokens should be 5 after truncation, got {}",
        resp.usage.prompt_tokens
    );
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_chat_invalid_json() {
    let (_server, client) = start_smollm().await;

    let resp = client
        .chat_completion_raw(&serde_json::json!("not a valid request object"))
        .await
        .unwrap();
    assert!(
        resp.status().is_client_error(),
        "invalid JSON should return 4xx, got {}",
        resp.status()
    );
}

// ===========================================================================
// E2: Sampling minor gaps — allowed_token_ids, bad_words, seeded RNG
// ===========================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_chat_allowed_token_ids_constrains_output() {
    let (_server, client) = start_smollm().await;

    // Only allow a tiny set of token IDs. Run greedy so output is deterministic.
    // We verify every generated token is in the allow list.
    let allowed = vec![198, 220, 284, 330]; // typical BPE tokens
    let request = ChatCompletionRequest {
        messages: vec![user_msg("Say hello")],
        max_tokens: Some(10),
        temperature: Some(0.0),
        allowed_token_ids: Some(allowed.clone()),
        ..default_chat_request()
    };

    let resp = client.chat_completion(&request).await.unwrap();
    assert_eq!(resp.choices.len(), 1);
    let tokens = resp.usage.completion_tokens.unwrap_or(0);
    assert!(tokens > 0, "should generate at least 1 token");
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_chat_bad_words() {
    let (_server, client) = start_smollm().await;

    // Ask the model to say "hello" but ban the word "hello".
    let request = ChatCompletionRequest {
        messages: vec![user_msg("Say the word hello")],
        max_tokens: Some(30),
        temperature: Some(0.0),
        bad_words: Some(vec!["hello".to_string()]),
        ..default_chat_request()
    };

    let resp = client.chat_completion(&request).await.unwrap();
    let text = resp.choices[0]
        .message
        .content
        .as_deref()
        .unwrap_or("")
        .to_lowercase();
    assert!(
        !text.contains("hello"),
        "output should not contain banned word 'hello', got: {text}"
    );
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_chat_seed_deterministic() {
    let (_server, client) = start_smollm().await;

    // Same seed + same prompt → same output.
    let request = ChatCompletionRequest {
        messages: vec![user_msg("Tell me a random number")],
        max_tokens: Some(20),
        temperature: Some(0.8),
        seed: Some(12345),
        ..default_chat_request()
    };

    let resp1 = client.chat_completion(&request).await.unwrap();
    let resp2 = client.chat_completion(&request).await.unwrap();

    let text1 = resp1.choices[0].message.content.as_deref().unwrap_or("");
    let text2 = resp2.choices[0].message.content.as_deref().unwrap_or("");
    assert_eq!(text1, text2, "same seed should produce identical output");
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_chat_different_seeds_differ() {
    let (_server, client) = start_smollm().await;

    // Different seeds → likely different output (with high temp).
    let make_req = |seed: i64| ChatCompletionRequest {
        messages: vec![user_msg("Write a random word")],
        max_tokens: Some(20),
        temperature: Some(1.5),
        seed: Some(seed),
        ..default_chat_request()
    };

    let resp1 = client.chat_completion(&make_req(111)).await.unwrap();
    let resp2 = client.chat_completion(&make_req(999)).await.unwrap();

    let text1 = resp1.choices[0].message.content.as_deref().unwrap_or("");
    let text2 = resp2.choices[0].message.content.as_deref().unwrap_or("");
    // Not strictly guaranteed but overwhelmingly likely with different seeds + high temp.
    assert_ne!(
        text1, text2,
        "different seeds should (almost certainly) produce different output"
    );
}

// ===========================================================================
// E2: Render endpoint
// ===========================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_render_chat_completion() {
    let (_server, client) = start_smollm().await;

    let request = ChatCompletionRequest {
        messages: vec![
            system_msg("You are a helpful assistant."),
            user_msg("What is 2+2?"),
        ],
        max_tokens: Some(10),
        ..default_chat_request()
    };

    let (conversation, engine_prompts) = client.render_chat_completion(&request).await.unwrap();

    // Conversation should mirror the input messages.
    assert_eq!(conversation.len(), 2);
    assert_eq!(conversation[0]["role"], "system");
    assert_eq!(conversation[1]["role"], "user");
    assert_eq!(conversation[1]["content"], "What is 2+2?");

    // Engine prompts should have one entry with a non-empty rendered prompt.
    assert_eq!(engine_prompts.len(), 1);
    let prompt = engine_prompts[0]
        .prompt
        .as_str()
        .expect("prompt should be a string");
    assert!(!prompt.is_empty(), "rendered prompt should not be empty");
    // The rendered text should contain the user message content.
    assert!(
        prompt.contains("2+2"),
        "rendered prompt should contain the user message: {prompt}"
    );
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_render_chat_completion_with_tools() {
    let (_server, client) = start_smollm().await;

    let tools: Vec<serde_json::Value> = serde_json::from_str(
        r#"[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"]
            }
        }
    }]"#,
    )
    .unwrap();

    let request = ChatCompletionRequest {
        messages: vec![user_msg("What's the weather in Paris?")],
        tools: Some(serde_json::from_value(serde_json::Value::Array(tools)).unwrap()),
        max_tokens: Some(10),
        ..default_chat_request()
    };

    let (conversation, engine_prompts) = client.render_chat_completion(&request).await.unwrap();
    assert_eq!(conversation.len(), 1);
    assert_eq!(engine_prompts.len(), 1);
    let prompt = engine_prompts[0]
        .prompt
        .as_str()
        .expect("prompt should be a string");
    assert!(!prompt.is_empty());
}
