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
//! The drafter drafts one request at a time (asserted, from the server's draft-token count): its
//! verify budget's 8 rows hold one request's block, so a step of 4 or 8 requests runs none, and
//! those runs check the target's batched rows. Splash drafts up to 4 lanes at once.
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
/// nats. On Qwen3.6-35B-A3B (base M5), a batch's numerics move the target's tokens by at most
/// 0.375 nats (its logits are bf16: 0.125-nat steps), and so does a correct drafter.
const MAX_MARGIN: f64 = 0.5;

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
    let metrics = Client::new(base_url).metrics().await.expect("metrics");
    (metrics.lines())
        .find_map(|line| line.strip_prefix("vllm_spec_decode_num_draft_tokens "))
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
        if width == 1 {
            assert!(proposed > 0, "one request at a time drafts");
        }
        assert_close(&alone, &drafted, &format!("drafting, {width} at once"));
    }
}
