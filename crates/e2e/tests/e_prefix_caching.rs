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
// Recurrent-hybrid prefix caching — issue #261
//
// The linear-attention layers of Qwen3.6 (Gated-DeltaNet) and Nemotron-3 Nano (Mamba-2) carry a
// per-sequence scan state that a cached KV prefix cannot rebuild, so a hit resumes only at a
// SNAPSHOT of that state, which the scan stores as it passes an earlier request's save point:
// `floor((N - 8) / 16) * 16` of its N-token prompt (`GENERATION_PROMPT_ALLOWANCE` before the end,
// on a 16-token block). The tests assert that EXACT resume point.
//
// Text is compared only between runs whose last prefill forward is the same: the same tokens, the
// same KV bits (a recompute over a cached block leaves it as written), the same recurrent state
// (one live, one a snapshot of it). Those must agree token for token. A run whose last forward has
// another shape agrees only to f32 rounding, which a near tie turns into another token — so each
// compared prompt is sized to snapshot exactly at a multiple of the server's prefill budget
// (`BUDGET`): its cold prefill ends a forward right where a warm one starts.
//
// Needs a Metal 4 GPU and the model in the HF cache: `mlx-community/Qwen3.6-35B-A3B-4bit` for
// `gdn::qwen3_6`, `mlx-community/NVIDIA-Nemotron-3-Nano-30B-A3B-4bit` for `gdn::nemotron_3_nano`.
// Run against a prebuilt server so its flags apply, one test at a time:
//
//   cargo build --release -p scratchy-cli --features metal,serve,model/qwen3.6-35b-a3b,quant/mlx-affine-b4-g64-qembed
//   VLLM_TEST_SPAWN=1 VLLM_TEST_BINARY=$PWD/target/release/scr cargo test -p scratchy-e2e \
//     -p scratchy-models --features e2e,metal,scratchy-models/smollm2-135m \
//     --test e_prefix_caching gdn::qwen3_6 -- --ignored --test-threads=1 --nocapture
//
// (`model/nemotron-3-nano-30b-a3b` and `gdn::nemotron_3_nano` for the other model.)
// ---------------------------------------------------------------------------

#[cfg(target_os = "macos")]
mod gdn {
    use std::time::Duration;

    use scratchy_e2e::{Client, TestServer};
    use scratchy_serving_api::protocol::{
        ChatCompletionRequest, ChatCompletionResponse, TokenizeRequest,
    };

    /// The prefill budget every server here runs with, and the snapshot point (or a multiple of
    /// it) every compared prompt is sized to.
    const BUDGET: u32 = 1024;

    /// Where an `n`-token prompt snapshots its recurrent state.
    fn snapshot_point(n: u32) -> u32 {
        (n - 8) / 16 * 16
    }

    async fn start(model: &str) -> (TestServer, Client) {
        let server = TestServer::builder(model)
            .with_device("metal")
            .with_max_num_seqs(4)
            .with_args(&["--max-num-batched-tokens", &BUDGET.to_string()])
            .with_timeout(Duration::from_secs(600))
            .start()
            .await
            .expect("server should start");
        let client = Client::new(server.base_url());
        (server, client)
    }

    /// Deterministic filler text about `topic`, `sentences` sentences long (~25 tokens each).
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

    /// `pad` sentences that grow a prompt a few tokens at a time.
    fn filler(pad: usize) -> String {
        "Please be brief. ".repeat(pad)
    }

    fn json_messages(messages: &[(&str, &str)]) -> Vec<serde_json::Value> {
        messages
            .iter()
            .map(|(role, content)| serde_json::json!({ "role": role, "content": content }))
            .collect()
    }

    /// Greedy, thinking off (a short, deterministic reply), at most `max_tokens`.
    fn chat(messages: &[(&str, &str)], max_tokens: u32) -> ChatCompletionRequest {
        serde_json::from_value(serde_json::json!({
            "messages": json_messages(messages),
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

    /// Measures the prompts `chat` sends without running them, so sizing one leaves no KV and no
    /// snapshot behind. `/tokenize` renders the template's default generation prompt, not the
    /// thinking-off one `chat` asks for; `offset` is the difference, measured once.
    struct Measure<'a> {
        client: &'a Client,
        offset: i64,
    }

    impl<'a> Measure<'a> {
        /// One request measures the offset; the cache is reset after it.
        async fn new(client: &'a Client) -> Self {
            let probe = [("user", "Say hi.")];
            let sent = client.chat_completion(&chat(&probe, 1)).await.unwrap();
            client.reset_prefix_cache().await.unwrap();
            let m = Self { client, offset: 0 };
            let rendered = m.tokens(&probe).await.len() as i64;
            Self {
                offset: i64::from(sent.usage.prompt_tokens) - rendered,
                ..m
            }
        }

        /// The tokens `/tokenize` renders for `messages`, generation prompt included.
        async fn tokens(&self, messages: &[(&str, &str)]) -> Vec<u32> {
            let request: TokenizeRequest = serde_json::from_value(serde_json::json!({
                "messages": json_messages(messages),
                "add_generation_prompt": true,
            }))
            .unwrap();
            self.client.tokenize(&request).await.unwrap().tokens
        }

        /// The prompt length `chat` sends for `messages`.
        async fn len(&self, messages: &[(&str, &str)]) -> u32 {
            (self.tokens(messages).await.len() as i64 + self.offset) as u32
        }

        /// The least padding at which `build`'s prompt snapshots at `at`, and its length: a
        /// binary search, as the prompt only grows with its padding — and a padding sentence is
        /// shorter than a block, so the first prompt to snapshot at or past `at` snapshots at it.
        async fn sized(
            &self,
            at: u32,
            build: impl Fn(usize) -> Vec<(String, String)>,
        ) -> (usize, u32) {
            let (mut lo, mut hi) = (0, 1024);
            while lo < hi {
                let mid = (lo + hi) / 2;
                match snapshot_point(self.len(&refs(&build(mid))).await) >= at {
                    true => hi = mid,
                    false => lo = mid + 1,
                }
            }
            let n = self.len(&refs(&build(lo))).await;
            assert_eq!(snapshot_point(n), at, "{lo} padding sentences: {n} tokens");
            (lo, n)
        }
    }

    fn refs(messages: &[(String, String)]) -> Vec<(&str, &str)> {
        messages
            .iter()
            .map(|(r, c)| (r.as_str(), c.as_str()))
            .collect()
    }

    /// One `#[ignore]`d test per case below for each model, in a module of its own.
    macro_rules! on_models {
        ($($module:ident => $model:literal),* $(,)?) => {$(
            mod $module {
                #[tokio::test(flavor = "multi_thread")]
                #[ignore]
                async fn gdn_resend_resumes_at_the_snapshot_and_answers_alike() {
                    scratchy_e2e::skip_if_no_gpu!();
                    super::resend_resumes_at_the_snapshot_and_answers_alike($model).await;
                }

                #[tokio::test(flavor = "multi_thread")]
                #[ignore]
                async fn gdn_a_snapshot_written_by_one_request_resumes_another() {
                    scratchy_e2e::skip_if_no_gpu!();
                    super::a_snapshot_written_by_one_request_resumes_another($model).await;
                }

                #[tokio::test(flavor = "multi_thread")]
                #[ignore]
                async fn gdn_multi_turn_resumes_each_turn_at_the_previous_prompts_snapshot() {
                    scratchy_e2e::skip_if_no_gpu!();
                    super::multi_turn_resumes_each_turn_at_the_previous_prompts_snapshot($model)
                        .await;
                }

                #[tokio::test(flavor = "multi_thread")]
                #[ignore]
                async fn gdn_concurrent_conversations_each_resume_at_their_turn_1_snapshot() {
                    scratchy_e2e::skip_if_no_gpu!();
                    super::concurrent_conversations_each_resume_at_their_turn_1_snapshot($model)
                        .await;
                }
            }
        )*};
    }

    on_models! {
        qwen3_6 => "mlx-community/Qwen3.6-35B-A3B-4bit",
        nemotron_3_nano => "mlx-community/NVIDIA-Nemotron-3-Nano-30B-A3B-4bit",
    }

    /// The same request three times. The first prefills `[0, BUDGET)`, snapshots, and runs the
    /// rest as its last forward; each resend restores the snapshot and runs that same forward —
    /// so it answers exactly alike.
    async fn resend_resumes_at_the_snapshot_and_answers_alike(model: &str) {
        let (_server, client) = start(model).await;
        let measure = Measure::new(&client).await;
        let system = passage("freight", 20);
        let build = |pad: usize| {
            vec![
                ("system".to_string(), system.clone()),
                (
                    "user".to_string(),
                    format!(
                        "{}Which depot ships crate class 4? Answer in one sentence.",
                        filler(pad)
                    ),
                ),
            ]
        };
        let (pad, n) = measure.sized(BUDGET, build).await;
        let request = chat(&refs(&build(pad)), 24);

        let cold = client.chat_completion(&request).await.unwrap();
        assert_eq!(
            cold.usage.prompt_tokens, n,
            "the prompt is the length measured"
        );
        assert_eq!(cached(&cold), 0, "a fresh server has nothing cached");
        for resend in 1..=2 {
            let warm = client.chat_completion(&request).await.unwrap();
            assert_eq!(cached(&warm), BUDGET, "resend {resend}");
            assert_eq!(text(&warm), text(&cold), "resend {resend}");
        }
    }

    /// Two prompts that share everything up to their snapshot point and differ after it: the
    /// second, resuming from a snapshot the FIRST request wrote, answers exactly as it does cold.
    async fn a_snapshot_written_by_one_request_resumes_another(model: &str) {
        let (_server, client) = start(model).await;
        let measure = Measure::new(&client).await;
        let system = passage("orchards", 20);
        let build = |pad: usize, fruit: &str| {
            vec![
                ("system".to_string(), system.clone()),
                (
                    "user".to_string(),
                    format!(
                        "{}Name one crate class shipped on route 3, then say the word {fruit}.",
                        filler(pad)
                    ),
                ),
            ]
        };
        // Both snapshot at BUDGET, past which they first differ, and are as long: what the two
        // prefill after the snapshot then runs as forwards of one shape, so the KV b reads from
        // a's past the snapshot is bit for bit what b computes cold.
        let (first, _) = measure.sized(BUDGET, |pad| build(pad, "apple")).await;
        let mut chosen = None;
        for pad in first.. {
            let (a, b) = (build(pad, "apple"), build(pad, "lemon"));
            if snapshot_point(measure.len(&refs(&a)).await) != BUDGET {
                break;
            }
            let (ta, tb) = (
                measure.tokens(&refs(&a)).await,
                measure.tokens(&refs(&b)).await,
            );
            let shared = ta.iter().zip(&tb).take_while(|(x, y)| x == y).count() as u32;
            if ta.len() == tb.len() && shared >= BUDGET {
                chosen = Some((a, b));
                break;
            }
        }
        let (a, b) = chosen.expect("some padding snapshots both at BUDGET, before they differ");

        let b_cold = client.chat_completion(&chat(&refs(&b), 24)).await.unwrap();
        assert_eq!(cached(&b_cold), 0, "a fresh server has nothing cached");

        client.reset_prefix_cache().await.unwrap();
        // An unrelated prompt takes the first snapshot slot, so a's snapshot lands in one b
        // never wrote: b can then match its cold run only by restoring what a saved.
        let unrelated = passage("unrelated", 30);
        client
            .chat_completion(&chat(&[("user", &unrelated)], 1))
            .await
            .unwrap();
        let a_cold = client.chat_completion(&chat(&refs(&a), 24)).await.unwrap();
        assert_eq!(cached(&a_cold), 0);
        let b_warm = client.chat_completion(&chat(&refs(&b), 24)).await.unwrap();
        assert_eq!(cached(&b_warm), BUDGET, "b resumes at a's snapshot");
        assert_eq!(text(&b_warm), text(&b_cold));
    }

    /// Three turns of one conversation, turn `k` sized to snapshot at `k * BUDGET`: each turn
    /// resumes at the previous turn's snapshot and prefills up to its own, then runs the rest as
    /// its last forward — which a resend, restoring that turn's snapshot, runs again exactly.
    async fn multi_turn_resumes_each_turn_at_the_previous_prompts_snapshot(model: &str) {
        let (_server, client) = start(model).await;
        let measure = Measure::new(&client).await;
        let mut history: Vec<(String, String)> = vec![("system".into(), passage("harbors", 20))];
        for turn in 1..=3u32 {
            let user = |pad: usize| {
                format!(
                    "{}{}Which route is used most? One sentence.",
                    passage(&format!("week {turn}"), 8),
                    filler(pad)
                )
            };
            let build = |pad: usize| {
                let mut messages = history.clone();
                messages.push(("user".into(), user(pad)));
                messages
            };
            let (pad, n) = measure.sized(turn * BUDGET, build).await;
            history.push(("user".into(), user(pad)));
            let messages = refs(&history);
            let resp = client.chat_completion(&chat(&messages, 24)).await.unwrap();
            eprintln!(
                "[gdn multi-turn] turn {turn} prompt={n} cached={}",
                cached(&resp)
            );
            assert_eq!(
                resp.usage.prompt_tokens, n,
                "turn {turn} is the length measured"
            );
            assert_eq!(cached(&resp), (turn - 1) * BUDGET, "turn {turn}");
            let resend = client.chat_completion(&chat(&messages, 24)).await.unwrap();
            assert_eq!(cached(&resend), turn * BUDGET, "turn {turn} resend");
            assert_eq!(text(&resend), text(&resp), "turn {turn} resend");
            history.push(("assistant".into(), text(&resp)));
        }
    }
    /// Four conversations interleaved on one server: each turn 2 resumes at its own turn 1's
    /// snapshot, which takes one slot per conversation while the turn 2s run — at least four
    /// snapshot slots. A 64 GB Mac has dozens beside Qwen3.6-35B-A3B; a 32 GB one, whose KV budget
    /// is about 1 GiB there, has one (the server's `state_snapshots=` line), and this test cannot
    /// pass on it.
    async fn concurrent_conversations_each_resume_at_their_turn_1_snapshot(model: &str) {
        let (server, _client) = start(model).await;
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
