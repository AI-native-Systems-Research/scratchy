// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! E2E coverage for long-prompt / chunked-prefill on CUDA.
//!
//! Mirrors `e_long_prompt_metal.rs`: forces chunking by lowering
//! `--max-num-batched-tokens` below the prompt length, and asserts the
//! greedy completion still answers a needle question whose answer is
//! NOT in the boilerplate (so a broken chunked-prefill that echoes the
//! prompt cannot pass).
//!
//! On CUDA, prefill chunk 2+ has `max_seqlen_q < max_seqlen_k`: this
//! chunk's q_len rows must attend over chunk 1's K/V from the paged KV
//! cache. The pre-fix CUDA path always dispatched
//! `Instruction::AttentionPrefillContiguous` to `flash_attn_contiguous`
//! (no block_table, no cache read), so chunk 2 only saw its own
//! contiguous K — the model never observed the needle and produced a
//! generic answer. The fix routes the chunked case through
//! `flashinfer_attention` (paged), with a `flash_attn_paged_ext`
//! fallback when no FlashInfer plan is compiled.
//!
//! Run with:
//!   `cargo test -p vllm-e2e --features e2e,cuda --test e_long_prompt_cuda \
//!        -- --ignored --test-threads=1`

#![cfg(feature = "e2e")]
#![cfg(feature = "cuda")]

use scratchy_e2e::assertions::assert_valid_completion_response;
use scratchy_e2e::{Client, TestModels, TestServer};
use scratchy_serving_api::protocol::{CompletionPrompt, CompletionRequest};

fn greedy_completion(prompt: &str, max_tokens: u32) -> CompletionRequest {
    CompletionRequest {
        prompt: Some(CompletionPrompt::Single(prompt.to_string())),
        max_tokens: Some(max_tokens),
        temperature: Some(0.0),
        ..serde_json::from_str(r#"{}"#).unwrap()
    }
}

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

async fn start_cuda_server_max_batched(model: &str, max_batched: &str) -> (TestServer, Client) {
    let server = TestServer::builder(model)
        .with_device("cuda:0")
        .with_args(&[
            "--max-model-len",
            "8192",
            "--max-num-batched-tokens",
            max_batched,
            "--enforce-eager",
        ])
        .start()
        .await
        .expect("server should start");
    let client = Client::new(server.base_url());
    (server, client)
}

/// Chunked-prefill correctness gate on CUDA. The prompt is large enough
/// to force two chunks at `--max-num-batched-tokens 2048` and one
/// contiguous prefill at `4096`. The needle ("42") sits at the front, so
/// only chunk 2 attending over chunk 1's cached K can recover it; a
/// broken chunked path that misses chunk 1's K answers something else.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn long_prompt_cuda_chunked_answers_needle() {
    const MODEL: &str = TestModels::QWEN2_0_5B_CUDA;
    let prompt = format!(
        "The magic number is 42. {}",
        build_prompt(2900, " Q: What is the magic number? A:")
    );

    let single = {
        let (_server, client) = start_cuda_server_max_batched(MODEL, "4096").await;
        let resp = client
            .completion(&greedy_completion(&prompt, 16))
            .await
            .expect("single-prefill completion must succeed");
        assert_valid_completion_response(&resp);
        resp.choices[0].text.clone()
    };
    assert!(
        single.contains("42"),
        "single-prefill reference must answer the needle question with '42' — got {single:?}"
    );

    let chunked = {
        let (_server, client) = start_cuda_server_max_batched(MODEL, "2048").await;
        let resp = client
            .completion(&greedy_completion(&prompt, 16))
            .await
            .expect("chunked-prefill completion must succeed");
        assert_valid_completion_response(&resp);
        resp.choices[0].text.clone()
    };

    // Note: byte-equality between single and chunked is NOT asserted on
    // CUDA. Single (q_len == k_len) still routes through
    // `flash_attn_contiguous`; chunked (q_len < k_len) routes through
    // `flashinfer_attention` (paged). The two kernels are numerically
    // close but not bit-identical, and downstream greedy decoding can
    // diverge on a tied or near-tied logit. Functional correctness is
    // gated by the needle assertion below.
    assert!(
        chunked.contains("42"),
        "chunked-prefill must answer the needle question with '42' — \
         a missing '42' means chunk 2 didn't attend over chunk 1's cached K. \
         got {chunked:?}"
    );
}
