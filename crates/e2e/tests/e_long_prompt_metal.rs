// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! E2E coverage for long-prompt / chunked-prefill on Metal.
//!
//! The metal Llama bucket spec is `workloads = [1, 8, 64, 512, 4096]`
//! (`crates/models/arch/dsl/llama.rs.in`). A prompt that fits in
//! the largest bucket runs as ONE contiguous prefill call; a prompt
//! larger than 4096 tokens forces the engine to chunk, and chunks
//! 2+ must read the prior chunk's K from the paged KV cache.
//!
//! The contiguous prefill kernel cannot read prior cached K — its
//! K-axis is `cu_seqlens_q` (new tokens only). Phase A landed the
//! `attention_prefill_sdpa_v2_paged_*` kernel + plumbing
//! (`60ef837f4`); Phase B (worker bucket plumbing + lowering arm +
//! engine `seqused_k` for prefill) is required to actually route
//! chunked prefill through the paged kernel.
//!
//! Phase B landed: the metal macro adapter now emits
//! `Instruction::AttentionPrefillPaged` for prefill on every Llama-arch
//! model (`crates/scratchy-forward-compiler-macro/src/metal/attention.rs`), and
//! the lowering arm routes that to `KernelId::AttentionPrefillSdpaPaged`.
//! Engine-side `seqused_k` / `block_table` were already populated by
//! `MetalWorker::execute_model` (`scratchy-serving-worker/src/gpu_worker.rs`).
//! Both chunked-prefill and multi-turn tests below are now gating
//! regressions — drop the `#[ignore]` and the suite must stay green.
//!
//! Run with:
//!   `cargo test -p vllm-e2e --features e2e --test e_long_prompt_metal`

#![cfg(feature = "e2e")]
#![cfg(target_os = "macos")]

use scratchy_e2e::assertions::assert_valid_completion_response;
use scratchy_e2e::{Client, TestServer};
use scratchy_serving_api::protocol::{CompletionPrompt, CompletionRequest};

/// Greedy completion request with a fixed `max_tokens` so the test
/// terminates predictably even if the model fixates on a pattern.
fn greedy_completion(prompt: &str, max_tokens: u32) -> CompletionRequest {
    CompletionRequest {
        prompt: Some(CompletionPrompt::Single(prompt.to_string())),
        max_tokens: Some(max_tokens),
        temperature: Some(0.0),
        ..serde_json::from_str(r#"{}"#).unwrap()
    }
}

/// Repeat a sentence until the rough token estimate hits the target.
/// SmolLM-135M's tokenizer averages ~4 chars/token on this kind of
/// boilerplate prose, so we count chars and divide.
fn build_prompt(target_tokens: usize, suffix: &str) -> String {
    let sentence = "The grass is green and the sky is blue. ";
    let chars_per_token = 4;
    let target_chars = target_tokens * chars_per_token;
    let reps = (target_chars / sentence.len()).max(1);
    let mut s = String::with_capacity(target_chars + suffix.len());
    for _ in 0..reps {
        s.push_str(sentence);
    }
    s.push_str(suffix);
    s
}

async fn start_metal_server() -> (TestServer, Client) {
    // SmolLM-135M-Instruct uses the same llama-arch bucket spec
    // (`workloads = [1, 8, 64, 512, 4096]`) as Llama-3.2 — the
    // 4096-token bucket cap is what triggers chunking.
    //
    // SmolLM-135M's default `max_model_len` is 2048; bump it to 8192
    // so a >4096-token prompt fits in the model context (otherwise
    // the engine bails before chunking matters — finish_reason="length"
    // with completion_tokens=0).
    let server = TestServer::builder(scratchy_e2e::TestModels::SMOLLM)
        .with_device("metal")
        .with_args(&["--max-model-len", "8192"])
        .start()
        .await
        .expect("server should start");
    let client = Client::new(server.base_url());
    (server, client)
}

/// Like [`start_metal_server`] but pins `--max-num-batched-tokens`, which
/// is the knob that forces chunked prefill: a prompt longer than this many
/// tokens is split across steps even when it would fit the 4096 bucket. Two
/// servers at different caps let a test diff single-prefill vs chunked-
/// prefill on the SAME prompt. Takes an explicit model so the caller can
/// pick one strong enough for a real needle assertion.
async fn start_metal_server_max_batched(model: &str, max_batched: &str) -> (TestServer, Client) {
    let server = TestServer::builder(model)
        .with_device("metal")
        .with_args(&[
            "--max-model-len",
            "8192",
            "--max-num-batched-tokens",
            max_batched,
        ])
        .start()
        .await
        .expect("server should start");
    let client = Client::new(server.base_url());
    (server, client)
}

/// Sanity baseline: ~1k-token prompt fits in the M=4096 bucket as a
/// single (paged) prefill call. Catches regressions in the prefill
/// path independent of the chunked / multi-turn cases below.
#[tokio::test(flavor = "multi_thread")]
async fn long_prompt_metal_within_bucket_completes() {
    scratchy_e2e::skip_if_no_gpu!();
    let (_server, client) = start_metal_server().await;

    let prompt = build_prompt(1000, "Q: What color is the grass? A:");
    let request = greedy_completion(&prompt, 8);

    let resp = client
        .completion(&request)
        .await
        .expect("completion request should succeed (single-bucket prefill)");
    assert_valid_completion_response(&resp);
    let text = &resp.choices[0].text;
    assert!(
        !text.is_empty(),
        "single-bucket prefill must produce a non-empty completion (~1k-token prompt)"
    );
}

/// Chunked-prefill regression: ~5000-token prompt forces the engine
/// to split the prefill across two steps (chunk 1 = 4096 tokens,
/// chunk 2 ≈ 900 tokens). Chunk 2 must attend over chunk 1's prior
/// cached K — only the paged prefill kernel can do this. Phase B
/// (commits landing on top of `4224bc2e4`) wired this through the
/// macro adapter + lowering arm; before that, this case panicked or
/// produced repetitive garbage. Validated end-to-end on
/// Llama-3.2-3B at 5000 tokens (`"The grass is green."`).
#[tokio::test(flavor = "multi_thread")]
async fn long_prompt_metal_chunked_prefill_completes() {
    scratchy_e2e::skip_if_no_gpu!();
    let (_server, client) = start_metal_server().await;

    // ~5000 tokens — needs chunked prefill (one bucket=4096 chunk +
    // one smaller chunk). Suffix is a short instruction so the model
    // has something concrete to answer (post-fix coherence check).
    let prompt = build_prompt(5000, "Q: What color is the grass? A:");
    let request = greedy_completion(&prompt, 16);

    let resp = client
        .completion(&request)
        .await
        .expect("chunked-prefill completion must not panic");
    assert_valid_completion_response(&resp);
    let text = &resp.choices[0].text;
    assert!(
        !text.is_empty(),
        "chunked-prefill must produce a non-empty completion"
    );

    // NOTE: this is a non-panic / non-empty smoke test for the
    // bucket-cap (>4096) chunking path. It deliberately does NOT assert
    // `contains("green")` — "green" is in the repeated boilerplate, so a
    // broken chunked-prefill that merely echoes the prompt passes it too
    // (this false-pass is why metal chunked-prefill shipped broken). The
    // real correctness gate is `long_prompt_metal_chunked_matches_single_prefill`
    // below, which diffs chunked output against single-prefill output.
}

/// Chunked-prefill CORRECTNESS gate: the same prompt must produce the
/// SAME greedy output whether prefilled in one step or split into chunks,
/// AND that output must answer a needle question whose answer is NOT in
/// the boilerplate.
///
/// Chunking is forced by lowering `--max-num-batched-tokens` below the
/// prompt length (independent of the 4096 bucket cap), so a ~3000-token
/// prompt runs as ONE prefill at `--max-num-batched-tokens 4096` and as
/// TWO chunks at `2048` — an apples-to-apples diff on the same model.
///
/// Uses Llama-3.2-1B (not the 135M SmolLM the other cases use): it's
/// strong enough that the needle assertion (`contains "42"`) is reliable,
/// so the test isn't a vacuous empty==empty pass.
///
/// Gates two chunked-prefill bugs that the old `contains("green")`
/// assertion silently passed ("green" is in the boilerplate):
///   1. Dropping the final chunk (missing re-arm) → prompt echo, no "42".
///   2. Sampling an INTERMEDIATE chunk → a spurious leading token
///      prepended before the real first generated token → differs from
///      the single-prefill output.
#[tokio::test(flavor = "multi_thread")]
async fn long_prompt_metal_chunked_matches_single_prefill() {
    scratchy_e2e::skip_if_no_gpu!();
    const MODEL: &str = scratchy_e2e::TestModels::LLAMA_3_2;
    // Needle at the front, ~2900 tokens of boilerplate, question at the
    // end. Total > 2048 (→ 2 chunks at cap 2048) but < 4096 (→ single
    // prefill at cap 4096). The answer ("42") is NOT in the boilerplate,
    // so a broken chunked-prefill cannot pass by echoing the prompt.
    let prompt = format!(
        "The magic number is 42. {}",
        build_prompt(2900, " Q: What is the magic number? A:")
    );

    // Single prefill first; drop its server before starting the next so
    // only one model is resident at a time.
    let single = {
        let (_server, client) = start_metal_server_max_batched(MODEL, "4096").await;
        let resp = client
            .completion(&greedy_completion(&prompt, 16))
            .await
            .expect("single-prefill completion must succeed");
        assert_valid_completion_response(&resp);
        resp.choices[0].text.clone()
    };

    // Single-prefill is the reference; it must answer the needle. (If this
    // fails the model/prompt is wrong, not the chunking — fix the test.)
    assert!(
        single.contains("42"),
        "single-prefill reference must answer the needle question with '42' — got {single:?}"
    );

    let chunked = {
        let (_server, client) = start_metal_server_max_batched(MODEL, "2048").await;
        let resp = client
            .completion(&greedy_completion(&prompt, 16))
            .await
            .expect("chunked-prefill completion must succeed");
        assert_valid_completion_response(&resp);
        resp.choices[0].text.clone()
    };

    assert_eq!(
        chunked, single,
        "chunked prefill (--max-num-batched-tokens 2048) must produce identical \
         greedy output to single prefill (4096) for the same prompt — a mismatch \
         means chunking changed the result (dropped the final chunk, or emitted a \
         spurious token for an intermediate chunk).\n  single:  {single:?}\n  chunked: {chunked:?}"
    );
}

/// Multi-turn regression: build a conversation whose accumulated
/// context exceeds the 4096-token bucket cap on turn 2+. Each turn
/// after the first must attend over the prior turns' cached K — the
/// same primitive as chunked prefill. Gated by the same Phase B
/// landing as `long_prompt_metal_chunked_prefill_completes`.
#[tokio::test(flavor = "multi_thread")]
async fn long_prompt_metal_multi_turn_continuation_completes() {
    scratchy_e2e::skip_if_no_gpu!();
    let (_server, client) = start_metal_server().await;

    // Turn 1: prompt ≈ 3000 tokens (single bucket, fine).
    let turn1 = build_prompt(3000, "Q: What is your favorite color? A:");
    let resp1 = client
        .completion(&greedy_completion(&turn1, 32))
        .await
        .expect("turn 1 must succeed (single-bucket prefill)");
    assert_valid_completion_response(&resp1);

    // Turn 2: append the model's reply + another long block, totaling
    // ~5500 tokens of context. Without paged prefill the worker has
    // to re-prefill from scratch (wasteful) AND chunk past the bucket
    // cap (broken).
    let turn2 = format!(
        "{}{}\n{}",
        turn1,
        resp1.choices[0].text,
        build_prompt(
            2500,
            "Q: Reflecting on the above, what one word answers the question? A:"
        ),
    );
    let resp2 = client
        .completion(&greedy_completion(&turn2, 16))
        .await
        .expect("turn 2 (chunked context) must not panic");
    assert_valid_completion_response(&resp2);
    assert!(
        !resp2.choices[0].text.is_empty(),
        "multi-turn continuation must produce a non-empty completion"
    );
}
