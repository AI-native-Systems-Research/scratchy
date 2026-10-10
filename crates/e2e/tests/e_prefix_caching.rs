// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! E2E tests for prefix caching (KV cache reuse).
//!
//! Verifies that sending the same prompt twice produces correct, identical
//! output — exercising the full path from HTTP → scheduler (with cached
//! prefix lookup) → worker (token slicing) → model → response.
//!
//! MLX tests use `device=metal` with the worker-level prefix cache pool.
//! CUDA tests use `device=cuda` with paged KV cache.
//!
//! Run with: `cargo test -p vllm-e2e --features e2e --test e_prefix_caching -- --ignored`

#![cfg(feature = "e2e")]

use scratchy_e2e::assertions::assert_valid_completion_response;
use scratchy_e2e::{Client, TestServer};
use scratchy_serving_api::protocol::{CompletionPrompt, CompletionRequest};

fn greedy_completion(prompt: &str, max_tokens: u32) -> CompletionRequest {
    CompletionRequest {
        prompt: Some(CompletionPrompt::Single(prompt.to_string())),
        max_tokens: Some(max_tokens),
        temperature: Some(0.0),
        ..serde_json::from_str(r#"{}"#).unwrap()
    }
}

async fn start_cpu_server() -> (TestServer, Client) {
    let server = TestServer::builder("HuggingFaceTB/SmolLM2-135M-Instruct")
        .with_device("cpu")
        .with_dtype("f32")
        .start()
        .await
        .expect("server should start");
    let client = Client::new(server.base_url());
    (server, client)
}

/// Send the same prompt twice — both should produce identical, valid output.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_prefix_caching_same_prompt_twice() {
    let (_server, client) = start_cpu_server().await;

    // Prompt long enough that at least 1 full block (16 tokens) is cached.
    let prompt = "The history of artificial intelligence began in the 1950s when researchers first explored the concept of machines that could think and reason about";
    let request = greedy_completion(prompt, 5);

    let resp1 = client.completion(&request).await.unwrap();
    assert_valid_completion_response(&resp1);
    let text1 = &resp1.choices[0].text;
    assert!(!text1.is_empty(), "first response should have text");

    let resp2 = client.completion(&request).await.unwrap();
    assert_valid_completion_response(&resp2);
    let text2 = &resp2.choices[0].text;

    assert_eq!(
        text1, text2,
        "greedy completions with same prompt should match:\n  first:  {text1:?}\n  second: {text2:?}"
    );
}

/// Three repeats of the same prompt — no degradation over time.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_prefix_caching_many_repeats() {
    let (_server, client) = start_cpu_server().await;

    let prompt = "The capital of France is";
    let request = greedy_completion(prompt, 5);

    let mut first_text = String::new();
    for i in 0..3 {
        let resp = client.completion(&request).await.unwrap();
        assert_valid_completion_response(&resp);
        let text = &resp.choices[0].text;
        assert!(!text.is_empty(), "response {i} should have text");

        if i == 0 {
            first_text = text.clone();
        } else {
            assert_eq!(text, &first_text, "greedy response {i} should match first");
        }
    }
}

// ---------------------------------------------------------------------------
// MLX (Metal) prefix caching
// ---------------------------------------------------------------------------

#[cfg(target_os = "macos")]
async fn start_metal_server() -> (TestServer, Client) {
    let server = TestServer::builder(scratchy_e2e::TestModels::SMOLLM)
        .with_device("metal")
        .start()
        .await
        .expect("server should start");
    let client = Client::new(server.base_url());
    (server, client)
}

/// Same-prompt-twice on MLX — exercises the worker-level KV cache pool.
#[cfg(target_os = "macos")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_prefix_caching_mlx_same_prompt_twice() {
    let (_server, client) = start_metal_server().await;

    let prompt = "The history of artificial intelligence began in the 1950s when researchers first explored the concept of machines that could think and reason about";
    let request = greedy_completion(prompt, 5);

    let resp1 = client.completion(&request).await.unwrap();
    assert_valid_completion_response(&resp1);
    let text1 = &resp1.choices[0].text;
    assert!(!text1.is_empty(), "first response should have text");

    let resp2 = client.completion(&request).await.unwrap();
    assert_valid_completion_response(&resp2);
    let text2 = &resp2.choices[0].text;

    assert_eq!(
        text1, text2,
        "greedy MLX completions with same prompt should match:\n  first:  {text1:?}\n  second: {text2:?}"
    );
}

// ---------------------------------------------------------------------------
// Cache-hit REPORTING (not just correctness)
//
// The tests above assert that a cached prefix produces the same text. They say
// nothing about whether the engine tells anyone how much it reused — and until
// the `engine_core.rs` fix it could not: `EngineCoreOutput.num_cached_tokens`
// was hardcoded `0` at every production site, so `prompt_tokens_details` was
// always `None` on every backend and the scheduler's correctly-computed hit was
// dropped on the floor.
//
// Epic #158 measures `reprefill_ratio` on an agent workload, so the reported
// number is the deliverable, not a diagnostic. This is the regression guard.
// ---------------------------------------------------------------------------

/// A chat request whose prompt is comfortably longer than one KV block.
///
/// The reportable hit is floored to the block size (16 by default), so a prompt
/// of <= 16 rendered tokens can never report a hit even when fully cached. This
/// prompt is a few hundred tokens, well clear of that edge. The text has to
/// carry the length itself: SmolLM's chat template does not reference `tools`,
/// so tool schemas on the request would render to nothing at all.
fn long_chat_request() -> scratchy_serving_api::protocol::ChatCompletionRequest {
    use scratchy_serving_api::protocol::{ChatCompletionMessageParam, ChatCompletionRequest};

    let mut prompt = String::new();
    for i in 0..40 {
        prompt.push_str(&format!(
            "Paragraph {i}: the scheduler hashes full blocks of prompt tokens and \
             reuses any contiguous run it has already computed. "
        ));
    }

    ChatCompletionRequest {
        messages: vec![ChatCompletionMessageParam {
            role: "user".to_string(),
            content: Some(serde_json::Value::String(prompt)),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }],
        max_tokens: Some(4),
        temperature: Some(0.0),
        // `ChatCompletionRequest.messages` has no serde default, so seed the
        // rest of the struct from a body that supplies it (same idiom as
        // `default_chat_request` in e1_basic_serving.rs).
        ..serde_json::from_str(r#"{"messages": []}"#).unwrap()
    }
}

/// Send the same long prompt twice and assert the SECOND request reports a
/// prefix-cache hit through `usage.prompt_tokens_details.cached_tokens`.
async fn assert_reports_cached_tokens(client: &Client, device: &str) {
    let request = long_chat_request();

    // A fresh TestServer means a fresh block pool, so the first request is the
    // cold arm by construction — no cache reset needed.
    let cold = client.chat_completion(&request).await.unwrap();
    let cold_cached = cold
        .usage
        .prompt_tokens_details
        .as_ref()
        .and_then(|d| d.cached_tokens)
        .unwrap_or(0);
    assert_eq!(
        cold_cached, 0,
        "first request to a fresh server must report no cache hit, got {cold_cached}"
    );

    let warm = client.chat_completion(&request).await.unwrap();
    let warm_cached = warm
        .usage
        .prompt_tokens_details
        .as_ref()
        .and_then(|d| d.cached_tokens)
        .unwrap_or(0);

    assert!(
        warm_cached > 0,
        "repeat of an identical {}-token prompt must report a cache hit, got 0 \
         (the exact failure mode the engine_core fix addresses: the scheduler \
         computes the hit and the output used to drop it)",
        cold.usage.prompt_tokens
    );
    assert!(
        warm_cached <= warm.usage.prompt_tokens,
        "cached {warm_cached} cannot exceed prompt {}",
        warm.usage.prompt_tokens
    );
    assert_eq!(
        cold.usage.prompt_tokens, warm.usage.prompt_tokens,
        "identical bodies must tokenize to the same prompt length"
    );
    eprintln!(
        "[prefix-cache/{device}] prompt={} cold_cached={} warm_cached={} reuse={:.1}%",
        warm.usage.prompt_tokens,
        cold_cached,
        warm_cached,
        100.0 * warm_cached as f64 / warm.usage.prompt_tokens as f64,
    );
}

/// Metal arm. There is no CPU arm: `device = "cpu"` has no backend in any
/// build ("No backend available for device 'cpu'"), which is also why this
/// file's older `start_cpu_server` tests cannot run.
///
/// On the hosted `macos-26` CI runner this SKIPS — paravirtual GPU, no Metal 4.
/// What CI gets from the matching workflow step is compile coverage of this
/// file; the assertions execute on a real GPU locally and via
/// `scripts/act-local.sh`.
///
/// Uses the DENSE checkpoint, not `TestModels::SMOLLM`: that constant is MLX
/// 4-bit on metal and will not load without a `quant/<preset>` feature. This
/// stem matches `model/smollm2-135m`, the feature set the metal CI job already
/// builds.
#[cfg(target_os = "macos")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_prefix_cache_reports_cached_tokens_metal() {
    scratchy_e2e::skip_if_no_gpu!();
    let server = TestServer::builder("HuggingFaceTB/SmolLM2-135M-Instruct")
        .with_device("metal")
        .start()
        .await
        .expect("server should start");
    let client = Client::new(server.base_url());
    assert_reports_cached_tokens(&client, "metal").await;
}

// ---------------------------------------------------------------------------
// Recurrent-hybrid (Gated-DeltaNet) prefix caching — issue #261
//
// Qwen3.6's linear-attention layers carry a per-sequence scan state that a cached KV prefix
// cannot rebuild, so a hit resumes only at a SNAPSHOT of that state, taken where an earlier
// request's prefill ended a forward: `floor((N - 8) / 16) * 16` of its N-token prompt
// (`GENERATION_PROMPT_ALLOWANCE` before the end, on a 16-token block), and only when it saves at
// least 256 tokens (`MIN_SNAPSHOT_GAIN`). The tests assert that EXACT resume point.
//
// Text is compared only between runs whose last prefill forward is identical: the same tokens,
// the same KV bits for the shared prefix, the same recurrent state (one live, one restored from a
// copy of it). Those must agree token for token. Runs that split a prefill differently (caching
// off vs on, a warm turn vs a cold one) agree only to f32 rounding, which a 256-expert MoE can
// turn into a different token — so no such comparison is made here.
//
// Needs a Metal 4 GPU and `mlx-community/Qwen3.6-35B-A3B-4bit` in the HF cache. Run against a
// prebuilt server so its flags apply, one test at a time:
//
//   cargo build --release -p scratchy-cli --features metal,serve,model/qwen3.6-35b-a3b,quant/mlx-affine-b4-g64-qembed
//   VLLM_TEST_SPAWN=1 VLLM_TEST_BINARY=$PWD/target/release/scr cargo test -p scratchy-e2e \
//     -p scratchy-models --features e2e,metal,scratchy-models/smollm2-135m \
//     --test e_prefix_caching gdn_ -- --ignored --test-threads=1 --nocapture
// ---------------------------------------------------------------------------

#[cfg(target_os = "macos")]
mod gdn {
    use std::time::Duration;

    use scratchy_e2e::{Client, TestServer};
    use scratchy_serving_api::protocol::{ChatCompletionRequest, ChatCompletionResponse};

    const MODEL: &str = "mlx-community/Qwen3.6-35B-A3B-4bit";

    /// Where an `n`-token prompt snapshots its recurrent state.
    fn snapshot_point(n: u32) -> u32 {
        (n - 8) / 16 * 16
    }

    async fn start() -> (TestServer, Client) {
        let server = TestServer::builder(MODEL)
            .with_device("metal")
            .with_max_num_seqs(4)
            .with_timeout(Duration::from_secs(600))
            .start()
            .await
            .expect("server should start");
        let client = Client::new(server.base_url());
        (server, client)
    }

    /// Deterministic filler text about `topic`, `sentences` sentences long (~16 tokens each).
    fn passage(topic: &str, sentences: usize) -> String {
        (0..sentences)
            .map(|i| {
                format!(
                    "Note {i} on {topic}: depot {} ships crate class {} on route {} every {} days. ",
                    (i * 7) % 13,
                    (i * 3) % 11,
                    (i * 5) % 17,
                    1 + i % 6,
                )
            })
            .collect()
    }

    /// Greedy, thinking off (a short, deterministic reply), at most `max_tokens`.
    fn chat(messages: &[(&str, &str)], max_tokens: u32) -> ChatCompletionRequest {
        let messages: Vec<_> = messages
            .iter()
            .map(|(role, content)| serde_json::json!({ "role": role, "content": content }))
            .collect();
        serde_json::from_value(serde_json::json!({
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "chat_template_kwargs": { "enable_thinking": false },
        }))
        .unwrap()
    }

    fn cached(resp: &ChatCompletionResponse) -> u32 {
        resp.usage
            .prompt_tokens_details
            .as_ref()
            .and_then(|d| d.cached_tokens)
            .unwrap_or(0)
    }

    fn text(resp: &ChatCompletionResponse) -> String {
        resp.choices[0].message.content.clone().unwrap_or_default()
    }

    /// The same request twice: the second resumes at the first's snapshot, and — its last
    /// forward being the first's last forward over a copy of the same state — answers alike.
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn gdn_resend_resumes_at_the_snapshot_and_answers_alike() {
        scratchy_e2e::skip_if_no_gpu!();
        let (_server, client) = start().await;
        let system = passage("freight", 40);
        let request = chat(
            &[
                ("system", &system),
                (
                    "user",
                    "Which depot ships crate class 4? Answer in one sentence.",
                ),
            ],
            24,
        );

        let cold = client.chat_completion(&request).await.unwrap();
        let n = cold.usage.prompt_tokens;
        assert_eq!(cached(&cold), 0, "a fresh server has nothing cached");
        assert!(
            snapshot_point(n) >= 256,
            "the prompt must be long enough to snapshot ({n})"
        );
        for resend in 1..=2 {
            let warm = client.chat_completion(&request).await.unwrap();
            assert_eq!(warm.usage.prompt_tokens, n);
            assert_eq!(cached(&warm), snapshot_point(n), "resend {resend}");
            assert_eq!(text(&warm), text(&cold), "resend {resend}");
        }
    }

    /// Two prompts that share everything up to their snapshot point and differ after it: the
    /// second, resuming from a snapshot the FIRST request wrote, answers exactly as it does cold.
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn gdn_a_snapshot_written_by_one_request_resumes_another() {
        scratchy_e2e::skip_if_no_gpu!();
        let (_server, client) = start().await;
        let system = passage("orchards", 40);
        let ask = |pad: usize, fruit: &str| {
            format!(
                "{}Name one crate class shipped on route 3, then say the word {fruit}.",
                "Please be brief. ".repeat(pad)
            )
        };
        // Differ only in the last word, and land the snapshot point at least 4 tokens before
        // the earliest place the two prompts can differ (the word, then 9 template tokens).
        let mut chosen = None;
        for pad in 0..16 {
            let (a, b) = (ask(pad, "apple"), ask(pad, "lemon"));
            let na = client
                .chat_completion(&chat(&[("system", &system), ("user", &a)], 1))
                .await
                .unwrap()
                .usage
                .prompt_tokens;
            let nb = client
                .chat_completion(&chat(&[("system", &system), ("user", &b)], 1))
                .await
                .unwrap()
                .usage
                .prompt_tokens;
            if na == nb && (na - 8) % 16 >= 4 {
                chosen = Some((a, b, na));
                break;
            }
        }
        let (a, b, n) = chosen.expect("some padding lands the snapshot point clear of the word");

        client.reset_prefix_cache().await.unwrap();
        let b_cold = client
            .chat_completion(&chat(&[("system", &system), ("user", &b)], 24))
            .await
            .unwrap();
        assert_eq!(cached(&b_cold), 0, "the reset must have emptied the cache");

        client.reset_prefix_cache().await.unwrap();
        // An unrelated prompt takes the first snapshot slot, so a's snapshot lands in one b
        // never wrote: b can then match its cold run only by restoring what a saved.
        let unrelated = passage("unrelated", 30);
        client
            .chat_completion(&chat(&[("user", &unrelated)], 1))
            .await
            .unwrap();
        let a_cold = client
            .chat_completion(&chat(&[("system", &system), ("user", &a)], 24))
            .await
            .unwrap();
        assert_eq!(cached(&a_cold), 0);
        let b_warm = client
            .chat_completion(&chat(&[("system", &system), ("user", &b)], 24))
            .await
            .unwrap();
        assert_eq!(
            cached(&b_warm),
            snapshot_point(n),
            "b resumes at a's snapshot"
        );
        assert_eq!(text(&b_warm), text(&b_cold));
    }

    /// Three turns of one conversation: each turn resumes at the previous prompt's snapshot,
    /// and a resend of each turn — resuming at that turn's own snapshot, in a slot of its own —
    /// answers exactly as the turn did.
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn gdn_multi_turn_resumes_each_turn_at_the_previous_prompts_snapshot() {
        scratchy_e2e::skip_if_no_gpu!();
        let (_server, client) = start().await;
        let system = passage("harbors", 40);
        // Each turn adds well over MIN_SNAPSHOT_GAIN tokens, so every prompt is snapshotted.
        let users: Vec<String> = (0..3)
            .map(|t| {
                format!(
                    "{}Which route is used most? One sentence.",
                    passage(&format!("week {t}"), 22)
                )
            })
            .collect();
        let mut history: Vec<(String, String)> = vec![("system".into(), system)];
        let mut previous_prompt: Option<u32> = None;
        for (turn, user) in users.iter().enumerate() {
            history.push(("user".into(), user.clone()));
            let messages: Vec<(&str, &str)> = history
                .iter()
                .map(|(r, c)| (r.as_str(), c.as_str()))
                .collect();
            let resp = client.chat_completion(&chat(&messages, 24)).await.unwrap();
            let n = resp.usage.prompt_tokens;
            let expected = previous_prompt.map_or(0, snapshot_point);
            eprintln!(
                "[gdn multi-turn] turn {} prompt={n} cached={}",
                turn + 1,
                cached(&resp)
            );
            assert_eq!(cached(&resp), expected, "turn {}", turn + 1);
            let resend = client.chat_completion(&chat(&messages, 24)).await.unwrap();
            assert_eq!(
                cached(&resend),
                snapshot_point(n),
                "turn {} resend",
                turn + 1
            );
            assert_eq!(text(&resend), text(&resp), "turn {} resend", turn + 1);
            history.push(("assistant".into(), text(&resp)));
            previous_prompt = Some(n);
        }
    }

    /// Four conversations interleaved on one server: each turn 2 resumes at its own turn 1's
    /// snapshot. Turn 2 adds under MIN_SNAPSHOT_GAIN tokens, so it takes no snapshot and the
    /// test needs only one slot per conversation (a 32 GB Mac has a handful).
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn gdn_concurrent_conversations_each_resume_at_their_turn_1_snapshot() {
        scratchy_e2e::skip_if_no_gpu!();
        let (server, _client) = start().await;
        let conversation = |i: usize| {
            let client = Client::new(server.base_url());
            async move {
                let system = format!("Conversation {i}. {}", passage(&format!("fleet {i}"), 30));
                let user1 = "Which depot ships crate class 2? One sentence.".to_string();
                let first = client
                    .chat_completion(&chat(&[("system", &system), ("user", &user1)], 16))
                    .await
                    .unwrap();
                let reply = text(&first);
                let user2 = format!("{}And class 5?", passage(&format!("addendum {i}"), 4));
                let second = client
                    .chat_completion(&chat(
                        &[
                            ("system", &system),
                            ("user", &user1),
                            ("assistant", &reply),
                            ("user", &user2),
                        ],
                        16,
                    ))
                    .await
                    .unwrap();
                (first.usage.prompt_tokens, cached(&first), cached(&second))
            }
        };
        let handles: Vec<_> = (0..4).map(|i| tokio::spawn(conversation(i))).collect();
        let mut results = Vec::new();
        for handle in handles {
            results.push(handle.await.expect("conversation task"));
        }
        for (i, (n1, cached1, cached2)) in results.into_iter().enumerate() {
            assert_eq!(cached1, 0, "conversation {i} turn 1");
            assert_eq!(cached2, snapshot_point(n1), "conversation {i} turn 2");
        }
    }
}
