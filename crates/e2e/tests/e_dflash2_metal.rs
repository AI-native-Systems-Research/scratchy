// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Speculative decoding with Qwen3.6-35B-A3B's DFlash2 drafter on Metal emits the target's own
//! output. At temperature 0, each request's text with the drafter — one request at a time, and 4
//! and 8 at once — matches the target's alone (`--num-speculative-tokens 0`, one at a time) up to a
//! near tie: where the two runs' tokens first differ, each is among the other's top 5 logprobs
//! (vLLM's `check_logprobs_close`, here with no late-position leniency) and the reference scores
//! the other's token within [`MAX_MARGIN`] of its own. The target alone, 4 and 8 at once, matches
//! itself one at a time the same way: a verify step's rows, like a batch's, run the target's
//! multi-row kernels.
//!
//! The drafter drafts at every width (asserted, from the server's draft-token count): a step verifies
//! up to four requests' blocks, as Splash's lanes.
//!
//! Needs a Metal 4 GPU, and `mlx-community/Qwen3.6-35B-A3B-4bit` with
//! `mlx-community/Qwen3.6-35B-A3B-DFlash2-4bit` in the HF cache. Run against a prebuilt server,
//! one test at a time:
//!
//!   cargo build --release -p scratchy-cli --features metal,serve,metrics,model/qwen3.6-35b-a3b,model/qwen3.6-35b-a3b-dflash2,quant/mlx-affine-b4-g64-qembed,quant/mlx-affine-b4-g64
//!   VLLM_TEST_SPAWN=1 VLLM_TEST_BINARY=$PWD/target/release/scr cargo test -p scratchy-e2e \
//!     -p scratchy-models --features e2e,metal,scratchy-models/smollm2-135m \
//!     --test e_dflash2_metal -- --ignored --test-threads=1 --nocapture

#![cfg(feature = "e2e")]
#![cfg(target_os = "macos")]

use scratchy_e2e::assertions::{
    EngineOutput, GoldenResult, check_logprobs_close_with_threshold, extract_engine_output,
};
use scratchy_e2e::{Client, TestServer};
use scratchy_serving_api::protocol::{CompletionPrompt, CompletionRequest};

const MODEL: &str = "mlx-community/Qwen3.6-35B-A3B-4bit";

/// Natural prompts, each answered at length: a run's divergence, if any, lands in prose.
const PROMPTS: [&str; 8] = [
    "Question: How does a refrigerator keep food cold? Explain it step by step.\nAnswer:",
    "Write a short story about a lighthouse keeper who finds a message in a bottle.\n\nStory:",
    "Question: What are the main differences between TCP and UDP?\nAnswer:",
    "Summarize the causes of the French Revolution in a few paragraphs.\n\nSummary:",
    "Write a Python function that checks whether a string is a palindrome, and explain it.\n\n",
    "Give me a recipe for a simple vegetable soup, with quantities.\n\nRecipe:",
    "Question: Why is the sky blue? Explain it to a ten-year-old.\nAnswer:",
    "Compare the economic policies of Keynes and Hayek.\n\nComparison:",
];

/// Tokens each request generates.
const TOKENS: u32 = 192;
/// The top logprobs each token carries: the near-tie window.
const TOP: u32 = 5;
/// How far below the reference's token a diverging token may score in the reference's own row:
/// a near tie. Mutual top-5 alone is no test of a drafter, whose wrong proposals are plausible
/// tokens: an acceptance that keeps one draft past the target's rejection passes it, at 1.75-3.88
/// nats. On Qwen3.6-35B-A3B (base M5), the target's own numerics move its tokens less: 0.375 nats
/// between one request and a batch, and up to 0.875 when which requests happen to share a step
/// changes the batch's kernels (its logits are bf16: 0.125-nat steps); a correct drafter, no more.
const MAX_MARGIN: f64 = 1.0;

fn request(prompt: &str) -> CompletionRequest {
    CompletionRequest {
        prompt: Some(CompletionPrompt::Single(prompt.to_string())),
        max_tokens: Some(TOKENS),
        temperature: Some(0.0),
        logprobs: Some(TOP),
        ..serde_json::from_str("{}").unwrap()
    }
}

/// Every prompt's output, `width` requests at once, in prompt order.
async fn outputs(base_url: &str, width: usize) -> Vec<EngineOutput> {
    let mut out = Vec::with_capacity(PROMPTS.len());
    for chunk in PROMPTS.chunks(width) {
        let mut set = tokio::task::JoinSet::new();
        for (i, &prompt) in chunk.iter().enumerate() {
            let client = Client::new(base_url);
            set.spawn(async move { (i, client.completion(&request(prompt)).await) });
        }
        let mut done = set.join_all().await;
        done.sort_by_key(|&(i, _)| i);
        for (_, response) in done {
            let response = response.expect("a completion");
            out.push(extract_engine_output(&response.choices[0]));
        }
    }
    out
}

/// Each run of `got` against the same prompt's run in `reference`, by `label`.
fn assert_close(reference: &[EngineOutput], got: &[EngineOutput], label: &str) {
    for (i, (want, got)) in reference.iter().zip(got).enumerate() {
        // A run that diverges may end its text sooner: they compare up to the shorter.
        assert!(
            !got.output_tokens.is_empty(),
            "{label}, prompt {i}: tokens generated"
        );
        let golden = GoldenResult {
            prompt: PROMPTS[i].to_string(),
            output_tokens: want.output_tokens.clone(),
            output_text: want.output_text.clone(),
            logprobs: want.logprobs.clone(),
        };
        eprintln!("{label}, prompt {i}:");
        check_logprobs_close_with_threshold(&golden, got, i, usize::MAX);
        let diverged =
            (want.output_tokens.iter().zip(&got.output_tokens)).position(|(a, b)| a != b);
        if let Some(at) = diverged {
            let (mine, theirs) = (&want.output_tokens[at], &got.output_tokens[at]);
            let margin = want.logprobs[at][mine] - want.logprobs[at][theirs];
            eprintln!(
                "MARGIN {label} prompt {i} position {at}: {margin:.4} nats ({mine:?} vs {theirs:?})"
            );
            assert!(
                margin <= MAX_MARGIN,
                "{label}, prompt {i}, position {at}: {theirs:?} is {margin} nats below the target's \
                 {mine:?}, past a near tie ({MAX_MARGIN})"
            );
        }
    }
}

async fn server(args: &[&str]) -> TestServer {
    TestServer::builder(MODEL)
        .with_device("metal")
        .with_args(args)
        .start()
        .await
        .expect("server should start")
}

/// The draft tokens the server at `base_url` has proposed (`vllm_spec_decode_num_draft_tokens`).
async fn draft_tokens(base_url: &str) -> u64 {
    counter(base_url, "vllm_spec_decode_num_draft_tokens").await
}

/// The server's counter `name`.
async fn counter(base_url: &str, name: &str) -> u64 {
    let metrics = Client::new(base_url).metrics().await.expect("metrics");
    (metrics.lines())
        .find_map(|line| line.strip_prefix(name)?.strip_prefix(' '))
        .map_or(0, |n| n.trim().parse::<f64>().expect("a count") as u64)
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn drafted_output_is_the_targets() {
    scratchy_e2e::skip_if_no_gpu!();
    let (alone, alone_4, alone_8) = {
        let server = server(&["--num-speculative-tokens", "0"]).await;
        let url = server.base_url();
        (
            outputs(&url, 1).await,
            outputs(&url, 4).await,
            outputs(&url, 8).await,
        )
    };
    assert_close(&alone, &alone_4, "target alone, 4 at once");
    assert_close(&alone, &alone_8, "target alone, 8 at once");
    let server = server(&["--enable-metrics"]).await;
    let url = server.base_url();
    for width in [1, 4, 8] {
        let before = draft_tokens(url).await;
        let drafted = outputs(url, width).await;
        let proposed = draft_tokens(url).await - before;
        eprintln!("drafting, {width} at once: {proposed} draft tokens proposed");
        assert!(proposed > 0, "{width} at once drafts");
        assert_close(&alone, &drafted, &format!("drafting, {width} at once"));
    }
}

// ── Sampled output (temperature 1) ───────────────────────────────────────────────────────────

/// A prompt whose continuations spread over many tokens past the first: the drafter drafts the
/// second animal.
const SAMPLED_PROMPT: &str = "My three favorite animals, in no particular order, are the";
/// Continuations a run samples; tokens each.
const SAMPLES: usize = 400;
const SAMPLED_TOKENS: u32 = 4;
/// A continuation seen this often in both runs together is its own category; rarer ones pool.
const CATEGORY_COUNT: usize = 10;
/// The chi-square test's p-value below which two runs' distributions differ.
const P_FLOOR: f64 = 1e-3;

/// `SAMPLES` continuations of [`SAMPLED_PROMPT`] at temperature 1, four requests at once.
async fn samples(base_url: &str) -> Vec<String> {
    let request = CompletionRequest {
        prompt: Some(CompletionPrompt::Single(SAMPLED_PROMPT.to_string())),
        max_tokens: Some(SAMPLED_TOKENS),
        temperature: Some(1.0),
        ..serde_json::from_str("{}").unwrap()
    };
    let mut out = Vec::with_capacity(SAMPLES);
    while out.len() < SAMPLES {
        let mut set = tokio::task::JoinSet::new();
        for _ in 0..4 {
            let (client, request) = (Client::new(base_url), request.clone());
            set.spawn(async move { client.completion(&request).await });
        }
        for response in set.join_all().await {
            out.push(response.expect("a completion").choices[0].text.clone());
        }
    }
    out
}

/// The two-sample chi-square test's statistic and degrees of freedom over `a` and `b`: a category
/// each continuation seen at least [`CATEGORY_COUNT`] times in both together, the rest pooled.
fn chi_square(a: &[String], b: &[String]) -> (f64, usize) {
    use std::collections::HashMap;
    let mut counts: HashMap<&str, (f64, f64)> = HashMap::new();
    for s in a {
        counts.entry(s).or_default().0 += 1.0;
    }
    for s in b {
        counts.entry(s).or_default().1 += 1.0;
    }
    let (mut cells, mut other) = (Vec::new(), (0.0, 0.0));
    for &(x, y) in counts.values() {
        if x + y >= CATEGORY_COUNT as f64 {
            cells.push((x, y));
        } else {
            other = (other.0 + x, other.1 + y);
        }
    }
    cells.push(other);
    let mut named: Vec<_> = (counts.iter())
        .filter(|&(_, &(x, y))| x + y >= CATEGORY_COUNT as f64)
        .collect();
    named.sort_by(|p, q| ((q.1.0 - q.1.1).abs()).total_cmp(&(p.1.0 - p.1.1).abs()));
    for (s, (x, y)) in named.iter().take(6) {
        eprintln!("SAMPLED   {x:4} vs {y:4} {s:?}");
    }
    eprintln!("SAMPLED   {:4} vs {:4} (rarer, pooled)", other.0, other.1);
    let (na, nb) = (a.len() as f64, b.len() as f64);
    let chi = (cells.iter())
        .filter(|&&(x, y)| x + y > 0.0)
        .map(|&(x, y)| {
            let (ea, eb) = ((x + y) * na / (na + nb), (x + y) * nb / (na + nb));
            (x - ea).powi(2) / ea + (y - eb).powi(2) / eb
        })
        .sum();
    (chi, cells.len() - 1)
}

/// The upper-tail p-value of `chi` at `df` degrees of freedom: Wilson-Hilferty's normal
/// approximation, then the normal tail by Abramowitz-Stegun 7.1.26.
fn chi_square_p(chi: f64, df: usize) -> f64 {
    let k = df as f64;
    let z = ((chi / k).cbrt() - (1.0 - 2.0 / (9.0 * k))) / (2.0 / (9.0 * k)).sqrt();
    let x = z.abs() / std::f64::consts::SQRT_2;
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let poly = t
        * (0.254_829_592
            + t * (-0.284_496_736
                + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    let erfc = poly * (-x * x).exp();
    match z >= 0.0 {
        true => erfc / 2.0,
        false => 1.0 - erfc / 2.0,
    }
}

/// At temperature 1 the drafter's output is distributed as the target's alone: its continuations,
/// sampled with drafting, are a sample of the target's own distribution (two-sample chi-square,
/// p at least [`P_FLOOR`]). An acceptance that ignores the target's probability fails this.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn drafted_sampling_is_the_targets() {
    scratchy_e2e::skip_if_no_gpu!();
    let alone = {
        let server = server(&["--num-speculative-tokens", "0"]).await;
        samples(server.base_url()).await
    };
    let server = server(&["--enable-metrics"]).await;
    let before = draft_tokens(server.base_url()).await;
    let drafted = samples(server.base_url()).await;
    let proposed = draft_tokens(server.base_url()).await - before;
    let (chi, df) = chi_square(&alone, &drafted);
    let p = chi_square_p(chi, df);
    eprintln!(
        "SAMPLED {proposed} draft tokens proposed; chi-square {chi:.2} at {df} df, p = {p:.4}"
    );
    assert!(proposed > 0, "the sampled run drafts");
    assert!(
        p >= P_FLOOR,
        "the drafter's sampled continuations are not the target's: chi-square {chi:.2} at {df} \
         df, p = {p:.2e}"
    );
}

// ── Prefix-cache hits ─────────────────────────────────────────────────────────────────────────

/// The characters of a document a hit's prompt takes: some 1,500 tokens.
const DOCUMENT_CHARS: usize = 6000;

/// A request for a summary of the head of `document`.
fn summary_prompt(document: &str) -> String {
    let head: String = document.chars().take(DOCUMENT_CHARS).collect();
    format!("{head}\n\nSummarize the document above in one paragraph.")
}

/// One request's text, its prompt tokens the prefix cache served, and the draft tokens the server
/// proposed and accepted while it ran.
#[derive(Debug)]
struct Run {
    text: String,
    cached: u32,
    proposed: u64,
    accepted: u64,
}

async fn run(base_url: &str, prompt: &str) -> Run {
    let accepted = "vllm_spec_decode_num_accepted_tokens";
    let (proposed_before, accepted_before) = (
        draft_tokens(base_url).await,
        counter(base_url, accepted).await,
    );
    // The chat endpoint reports the prompt tokens the prefix cache served.
    let body = serde_json::json!({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": TOKENS,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": false},
    });
    let response = (Client::new(base_url).chat_completion_raw(&body).await).expect("a reply");
    let response: serde_json::Value = response.json().await.expect("a JSON reply");
    let cached = &response["usage"]["prompt_tokens_details"]["cached_tokens"];
    Run {
        text: (response["choices"][0]["message"]["content"].as_str())
            .expect("its text")
            .to_string(),
        cached: cached.as_u64().unwrap_or(0) as u32,
        proposed: draft_tokens(base_url).await - proposed_before,
        accepted: counter(base_url, accepted).await - accepted_before,
    }
}

/// A prefix-cache hit drafts from the drafter's context of its cached prefix, which the snapshot it
/// resumes from kept (Splash's composite state: the window the hit rebuilds its rings from), not
/// from whatever its ring slot last held. A prompt hit right after its own prefill, and hit again
/// after another long prompt took its ring slot, says the same and keeps as many drafts. Without
/// the restore the second hit drafts from the other prompt's rows: it kept 71 drafts to the first
/// hit's 109, and its text differed. (The drafts each run proposes are left out: one run proposed
/// one draft more than the other, 220 to 219, with the same text and the same drafts kept.)
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn a_hit_drafts_from_the_context_its_snapshot_kept() {
    scratchy_e2e::skip_if_no_gpu!();
    let server = server(&["--enable-metrics"]).await;
    let url = server.base_url();
    let prompt = summary_prompt(include_str!("../../../README.md"));
    let other = summary_prompt(include_str!("../../../docs/BUILD.md"));
    let cold = run(&url, &prompt).await;
    let hit = run(&url, &prompt).await;
    run(&url, &other).await;
    let hit_after_other = run(&url, &prompt).await;
    eprintln!("HIT cold {cold:?}\nHIT hit {hit:?}\nHIT after another prompt {hit_after_other:?}");
    assert_eq!(cold.cached, 0, "the first run is cold");
    assert!(hit.cached > 0, "the second run hits");
    assert!(hit.proposed > 0, "the hit drafts");
    assert_eq!(hit.text, hit_after_other.text, "a hit says the same");
    assert_eq!(
        hit.accepted, hit_after_other.accepted,
        "a hit keeps as many drafts, from its snapshot's context"
    );
}
