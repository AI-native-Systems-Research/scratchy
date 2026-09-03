// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! E2E tests for relocatable spans (BlockAnnotations).
//!
//! Verifies that:
//! 1. Relocatable annotations produce identical output to the normal path
//!    (the rotate-attend-unrotate cycle doesn't corrupt attention).
//! 2. Relocatable blocks produce cache hits regardless of document order.
//!
//! Run with: `cargo test -p vllm-e2e --features e2e --test e_spans -- --ignored`

#![cfg(feature = "e2e")]

use scratchy_core_common::BlockKind;
use scratchy_e2e::TestModels;
use scratchy_serving_api::llm::{LLMBuilder, Prompt, SamplingParams};
use std::collections::BTreeMap;

fn greedy_params(max_tokens: u32) -> SamplingParams {
    SamplingParams {
        max_tokens: Some(max_tokens),
        temperature: 0.0,
        detokenize: true,
        ..SamplingParams::default()
    }
}

/// Build the LLM, applying an optional `SPANS_E2E_GPU_UTIL` fraction override.
/// Default (env unset) leaves the builder's 0.9 — byte-identical to before. The
/// override is for running these validations when GPU memory is constrained by
/// a concurrent process (the KV pool otherwise reserves 0.9× of *system*
/// memory, which OOMs the Metal command buffer when another process already
/// holds most of the GPU).
fn build_llm(model: &str) -> scratchy_serving_api::llm::LLM {
    // Force fp16 KV: these tests assert the *fp16 rope-on-read* transparency
    // invariant (annotated == normal, bit-for-bit). TurboQuant is on by default
    // on metal and stores lossy KV codes, which makes the rope-on-read reuse
    // path diverge from rotate-on-write by quant noise — failing the assert by
    // construction. Spans-under-TurboQuant is covered separately by the bench
    // reuse/coherence checks, not this bit-identity gate.
    // Default fp16 KV: the bit-identity coherence tests need a lossless cache.
    // The PERF bench (which does not assert bit-identity) overrides via
    // SPANS_KV_DTYPE=auto to run the production TurboQuant path (what real
    // `serve`/`launch claude` use), so gemma's hd512 global KV is compressed
    // instead of ballooning fp16 and OOMing at long context on 32GB.
    let kv_dtype = std::env::var("SPANS_KV_DTYPE")
        .ok()
        .unwrap_or_else(|| "fp16".into());
    let mut b = LLMBuilder::new(model)
        .enable_prefix_caching(true)
        .kv_cache_dtype(kv_dtype);
    if let Some(frac) = std::env::var("SPANS_E2E_GPU_UTIL")
        .ok()
        .and_then(|s| s.trim().parse::<f64>().ok())
    {
        b = b.gpu_memory_utilization(frac);
    }
    // SPANS_MAX_LEN raises max_model_len + max_num_batched_tokens so a single
    // large prompt gets enough KV blocks AND runs in one (un-chunked) prefill.
    // The default caps the seq at the kernel block-table stride (~128 blocks =
    // 2048 tok), faulting the GPU command buffer past that.
    if let Some(len) = std::env::var("SPANS_MAX_LEN")
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
    {
        b = b.max_model_len(len).max_num_batched_tokens(len);
    }
    b.build().expect("LLM should initialize")
}

/// Body of the transparency test, parameterized by model id + prompt so the
/// SmolLM and gemma4 entry points share one implementation. gemma4's global
/// layers (head_dim 512, proportional rope) run the gqa_shared prefill kernel;
/// with spans they take the rope-once-to-scratch path (K roped once into a
/// dense scratch, attention reads pre-roped K). Annotation block indices are in
/// 16-token blocks regardless of layer class (the GLOBAL_BLOCK_SIZE 32 page
/// stride is irrelevant to annotation keying), so the pad/annotate is the same.
fn run_transparency(model: &str, prompt_text: &str) {
    scratchy_core_common::telemetry::init_tracing("off");

    let mut llm = build_llm(model);

    let block_size = 16;
    let tokenizer = llm.tokenizer().expect("tokenizer should be available");
    let token_ids = tokenizer.encode(prompt_text, false).expect("encode");

    // Pad to block boundary so annotations align.
    let mut padded = token_ids.clone();
    let remainder = padded.len() % block_size;
    if remainder > 0 {
        padded.resize(padded.len() + (block_size - remainder), 0);
    }

    // Run without annotations (normal path).
    let output_normal = llm
        .generate(&[Prompt::TokenIds(padded.clone())], Some(greedy_params(10)))
        .expect("generate normal");

    // Reset cache to avoid prefix hits confounding the test.
    llm.reset_prefix_cache().expect("reset");

    // Run with all blocks annotated as Relocatable.
    let num_blocks = padded.len() / block_size;
    let mut annotations = BTreeMap::new();
    for i in 0..num_blocks {
        annotations.insert(
            i,
            BlockKind::Relocatable {
                first_token: (i * block_size) as u32,
            },
        );
    }
    let output_annotated = llm
        .generate(
            &[Prompt::TokenIdsWithAnnotations(padded, annotations)],
            Some(greedy_params(10)),
        )
        .expect("generate annotated");

    let text_normal = &output_normal[0].outputs[0].text;
    let text_annotated = &output_annotated[0].outputs[0].text;

    assert_eq!(
        text_normal, text_annotated,
        "Relocatable annotations should not change output.\n  normal:    {text_normal:?}\n  annotated: {text_annotated:?}"
    );
}

/// Same prompt with and without Relocatable annotations should produce
/// identical output — proving the rotate-attend-unrotate cycle is transparent.
#[test]
#[ignore]
fn test_relocatable_annotations_produce_same_output() {
    run_transparency(TestModels::SMOLLM, "The capital of France is");
}

/// gemma4 transparency (the launch-claude target). Exercises the gqa_shared
/// global-prefill rope-once-to-scratch path: annotated (rope-on-read) output
/// must equal the normal (rotate-on-write) output. Loads the 12B dense variant
/// (smaller; set GEMMA4_E2E_MOE=1 to use the 26B-a4b MoE checkpoint instead).
#[cfg(feature = "metal")]
#[test]
#[ignore]
fn test_relocatable_annotations_produce_same_output_gemma4() {
    let model = if std::env::var_os("GEMMA4_E2E_MOE").is_some() {
        TestModels::GEMMA4_MOE
    } else {
        TestModels::GEMMA4
    };
    run_transparency(model, "The capital of France is");
}

/// Running the same ordering twice with Relocatable annotations should
/// produce identical output — proving that cache reuse via the
/// rotate-attend-unrotate cycle doesn't corrupt results.
///
/// NOTE: This test currently fails on MLX because the MLX worker does not
/// have annotation-aware block hashing or per-block KV cache reuse.
/// It should pass on CUDA once the paged rotation path is exercised.
#[test]
#[ignore]
fn test_relocatable_cache_hit_produces_same_output() {
    scratchy_core_common::telemetry::init_tracing("off");

    // fp16 KV: same bit-identity invariant as the transparency tests (see
    // build_llm) — TurboQuant's lossy codes would break reuse==fresh parity.
    let mut llm = LLMBuilder::new(TestModels::SMOLLM)
        .enable_prefix_caching(true)
        .kv_cache_dtype("fp16")
        .build()
        .expect("LLM should initialize");

    let block_size = 16;
    let tokenizer = llm.tokenizer().expect("tokenizer should be available");

    let doc_text = "Document about the history of computing and Charles Babbage";
    let query_text = "Summarize:";

    let doc_ids = tokenizer.encode(doc_text, false).expect("encode doc");
    let query_ids = tokenizer.encode(query_text, false).expect("encode q");

    // Pad each to block boundary.
    let pad_to = |ids: &[u32]| -> Vec<u32> {
        let mut v = ids.to_vec();
        let rem = v.len() % block_size;
        if rem > 0 {
            v.resize(v.len() + (block_size - rem), 0);
        }
        v
    };

    let doc = pad_to(&doc_ids);
    let query = pad_to(&query_ids);

    let doc_blocks = doc.len() / block_size;
    let query_blocks = query.len() / block_size;

    let mut tokens = Vec::new();
    tokens.extend_from_slice(&doc);
    tokens.extend_from_slice(&query);

    let mut ann = BTreeMap::new();
    for i in 0..doc_blocks {
        ann.insert(
            i,
            BlockKind::Relocatable {
                first_token: (i * block_size) as u32,
            },
        );
    }
    for i in doc_blocks..doc_blocks + query_blocks {
        ann.insert(
            i,
            BlockKind::Prefixed {
                first_token: (i * block_size) as u32,
            },
        );
    }

    // First run — populates cache.
    let output1 = llm
        .generate(
            &[Prompt::TokenIdsWithAnnotations(tokens.clone(), ann.clone())],
            Some(greedy_params(10)),
        )
        .expect("generate first");

    // Second run — should hit cached Relocatable blocks.
    let output2 = llm
        .generate(
            &[Prompt::TokenIdsWithAnnotations(tokens, ann)],
            Some(greedy_params(10)),
        )
        .expect("generate second");

    let text1 = &output1[0].outputs[0].text;
    let text2 = &output2[0].outputs[0].text;

    assert!(!text1.is_empty(), "first output should not be empty");
    assert_eq!(
        text1, text2,
        "Cache hit with Relocatable blocks should produce identical output.\n  first:  {text1:?}\n  second: {text2:?}"
    );
}

/// Bug gate (RECOMBINATION transparency): a Relocatable span reused in a
/// DIFFERENT document order than it was first prefilled must produce the same
/// output as computing that order fresh. Today spans prefill FULL-CAUSAL but are
/// reused CONTENT-ADDRESSED (parent hash reset to NONE), so a reordered reuse
/// returns each doc's K/V computed for the WRONG neighbors → silent corruption.
/// Self-only span isolation makes the
/// reuse transparent. EXPECTED: this FAILS today (it captures the bug) and
/// PASSES once Relocatable spans are prefilled block-diagonally (self-only).
///
/// Distinct from `test_relocatable_cache_hit_produces_same_output`, which only
/// reuses the SAME order (where full-causal reuse happens to be correct).
#[test]
#[ignore]
fn test_relocatable_recombination_reuse_matches_fresh() {
    scratchy_core_common::telemetry::init_tracing("off");
    let mut llm = build_llm(TestModels::SMOLLM);

    let block_size = 16;
    let tokenizer = llm.tokenizer().expect("tokenizer should be available");
    let pad_to = |s: &str| -> Vec<u32> {
        let mut v = tokenizer.encode(s, false).expect("encode");
        let rem = v.len() % block_size;
        if rem > 0 {
            v.resize(v.len() + (block_size - rem), 0);
        }
        v
    };

    let d0 = pad_to("Fact one: the river Nile flows north through Egypt.");
    let d1 = pad_to("Fact two: copper is an excellent conductor of electricity.");
    let d2 = pad_to("Fact three: the violin has four strings tuned in fifths.");
    let query = pad_to("Question: restate fact three exactly.");

    // Assemble an annotated prompt for a given document order: each doc's first
    // block starts a Relocatable span; the query tail is Prefixed (attends all).
    let assemble = |order: &[&Vec<u32>]| -> (Vec<u32>, BTreeMap<usize, BlockKind>) {
        let mut toks: Vec<u32> = Vec::new();
        let mut ann = BTreeMap::new();
        for d in order {
            let start_block = toks.len() / block_size;
            toks.extend_from_slice(d);
            ann.insert(
                start_block,
                BlockKind::Relocatable {
                    first_token: (start_block * block_size) as u32,
                },
            );
        }
        let q_start = toks.len() / block_size;
        toks.extend_from_slice(&query);
        for b in q_start..(toks.len() / block_size) {
            ann.insert(
                b,
                BlockKind::Prefixed {
                    first_token: (b * block_size) as u32,
                },
            );
        }
        (toks, ann)
    };

    let (toks_a, ann_a) = assemble(&[&d0, &d1, &d2]);
    let (toks_b, ann_b) = assemble(&[&d2, &d1, &d0]);

    // 1. Populate the cache with order A (docs' K/V computed in A-order context).
    let _ = llm
        .generate(
            &[Prompt::TokenIdsWithAnnotations(toks_a, ann_a)],
            Some(greedy_params(12)),
        )
        .expect("generate order A");

    // 2. Order B, reusing the now-cached (content-addressed) doc blocks.
    let out_b_cached = llm
        .generate(
            &[Prompt::TokenIdsWithAnnotations(
                toks_b.clone(),
                ann_b.clone(),
            )],
            Some(greedy_params(12)),
        )
        .expect("generate order B (cached)");
    let text_b_cached = out_b_cached[0].outputs[0].text.clone();

    // 3. Order B fresh on a cold cache — the ground truth for B under the same
    //    annotation semantics. Transparent reuse ⇒ (2) must equal (3).
    llm.reset_prefix_cache().expect("reset");
    let out_b_fresh = llm
        .generate(
            &[Prompt::TokenIdsWithAnnotations(toks_b, ann_b)],
            Some(greedy_params(12)),
        )
        .expect("generate order B (fresh)");
    let text_b_fresh = out_b_fresh[0].outputs[0].text.clone();

    assert_eq!(
        text_b_cached, text_b_fresh,
        "Reordered span reuse must match a fresh compute of the same order.\n  \
         cached (after order A): {text_b_cached:?}\n  fresh (cold cache):     {text_b_fresh:?}\n  \
         Divergence = the full-causal-prefill + content-addressed-reuse bug; self-only isolation fixes it."
    );
}

/// First (cold) TTFT: WITHOUT span annotations (full-causal) vs WITH
/// (block-diagonal) — SAME model, SAME `padded` token sequence, cache reset
/// before each so neither reuses. This is the clean, apples-to-apples
/// sparse-attention A/B: the ONLY difference between the two runs is the
/// per-block `Relocatable` annotation, i.e. `TokenIds` vs
/// `TokenIdsWithAnnotations` on identical tokens. `ratio = with/without`, so
/// `< 1.0` is a speedup (e.g. 0.55 = 1.83×).
///
/// Block-diagonal attention lets a token in span s attend only within its own
/// span (+ shared prefix): O(T²) → O(T·S). The saving GROWS with context and
/// with how attention-bound the model is.
///
/// 🔑 REQUIRED: `SPANS_BENCH_SPAN_BLOCKS` must be ≥2 (default is 1). With span
/// size 1 (16 tokens) a steel/NAX Q-tile (BQ≥32 = ≥2 blocks) STRADDLES two
/// spans, so the kb-bound can't engage and the ratio stays ~1.0 (measures only
/// rope-once overhead). Set it to 32 (= 512-token spans, tool-sized) so a
/// Q-tile is span-uniform and the block-diagonal skip fires. This exactly
/// mirrors launch-claude, whose tools are hundreds of tokens each.
///
/// REPRODUCE (Metal). `SPANS_BENCH_MODEL` overrides the model (SmolLM caps at
/// 2048 tokens, so long-context runs need a 128k model). Build with the
/// matching `scratchy-models/<stem>` feature (a mismatched
/// build panics with a GPU commit error) — llama-3.2-1b is
/// `llama-3.2-1b`, on by default:
///
///   SPANS_BENCH_MODEL=mlx-community/llama-3.2-1b-instruct-4bit \
///   SPANS_TTFT_LENS=8192,16384,24576 SPANS_BENCH_SPAN_BLOCKS=32 \
///   cargo test -p scratchy-e2e --features e2e,metal \
///     --release --test e_spans bench_first_ttft_spans_overhead -- --ignored --nocapture
///
/// Reference numbers (2026-07-01, M-series, span_blocks=32, ratio = speedup):
///   llama-3.2-1b (hd64, uniform full-attn): 8k 1.23×  16k 1.51×  24k 1.83×
///   gemma-4-12b  (hd512, hybrid SWA):       8k 1.07×  16k 1.15×  24k 1.23×
/// gemma is lower because 25/30 layers are sliding-window (already local) and
/// the dense 12B FFN dominates prefill; both climb with context. Set
/// SPANS_TTFT_LENS to override the lengths.
fn run_first_ttft_bench(model: &str, default_lens: &[usize]) {
    scratchy_core_common::telemetry::init_tracing("off");
    // SPANS_BENCH_MODEL overrides the model id — SmolLM-135M caps at a 2048-tok
    // context (128 blocks), so measuring the asymptotic span win at 8k/16k needs
    // a large-context model (e.g. Llama-3.2-1B = 128k, `llama-3.2-1b`,
    // on by default).
    let model_override = std::env::var("SPANS_BENCH_MODEL").ok();
    let model = model_override.as_deref().unwrap_or(model);
    let mut llm = build_llm(model);

    let block_size = 16;
    let tokenizer = llm.tokenizer().expect("tokenizer");
    let base = tokenizer
        .encode(
            "The quick brown fox jumps over the lazy dog near the riverbank. ",
            false,
        )
        .expect("encode");

    let lens: Vec<usize> = std::env::var("SPANS_TTFT_LENS")
        .ok()
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_else(|| default_lens.to_vec());

    for target_len in lens {
        let mut padded = Vec::new();
        while padded.len() < target_len {
            padded.extend_from_slice(&base);
        }
        padded.truncate(target_len);
        let rem = padded.len() % block_size;
        if rem > 0 {
            padded.resize(padded.len() + (block_size - rem), 0);
        }
        let num_blocks = padded.len() / block_size;
        // SPANS_BENCH_SPAN_BLOCKS = blocks per span. Default 1 (every block its
        // own 16-tok span — the extreme; only the sdpa per-query loop-bound can
        // exploit it). Larger (e.g. 32 = 512-tok spans, close to real tools)
        // makes steel Q-tiles (BQ=32 tok = 2 blocks) span-uniform so the steel
        // kb-bound restriction engages.
        let span_blocks = std::env::var("SPANS_BENCH_SPAN_BLOCKS")
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .filter(|&n| n > 0)
            .unwrap_or(1);
        let mut annotations = std::collections::BTreeMap::new();
        for i in (0..num_blocks).step_by(span_blocks) {
            annotations.insert(
                i,
                BlockKind::Relocatable {
                    first_token: (i * block_size) as u32,
                },
            );
        }

        // Warm up — build pipelines so the timed runs are warm + cold-cache.
        let _ = llm.generate(&[Prompt::TokenIds(padded.clone())], Some(greedy_params(1)));

        let iters = 5;
        let mut without = std::time::Duration::ZERO;
        let mut with = std::time::Duration::ZERO;
        for _ in 0..iters {
            llm.reset_prefix_cache().expect("reset");
            let t = std::time::Instant::now();
            llm.generate(&[Prompt::TokenIds(padded.clone())], Some(greedy_params(1)))
                .expect("gen no-spans");
            without += t.elapsed();

            llm.reset_prefix_cache().expect("reset");
            let t = std::time::Instant::now();
            llm.generate(
                &[Prompt::TokenIdsWithAnnotations(
                    padded.clone(),
                    annotations.clone(),
                )],
                Some(greedy_params(1)),
            )
            .expect("gen spans");
            with += t.elapsed();
        }
        let a = without.as_secs_f64() / iters as f64 * 1000.0;
        let b = with.as_secs_f64() / iters as f64 * 1000.0;
        eprintln!(
            "FIRST-TTFT (cold, {} tok prompt): without spans = {a:.1} ms  with spans = {b:.1} ms  \
             ratio = {:.3}x",
            padded.len(),
            b / a
        );
    }
}

#[test]
#[ignore]
fn bench_first_ttft_spans_overhead() {
    run_first_ttft_bench(TestModels::SMOLLM, &[2048, 8192, 16384]);
}

/// gemma4 first-TTFT overhead (the launch-claude target). The span K is roped
/// ONCE (an O(K) pre-pass) by `rope_once_gqa_shared` instead of per-q-tile
/// inside the gqa_shared attention, so the overhead AMORTIZES at long context:
/// the ratio should approach ~1.0× at 8k/16k (the one-time O(K) rope is
/// negligible vs the O(NQ·K) attention it replaces the per-tile rope of). Loads
/// the 12B dense variant (GEMMA4_E2E_MOE=1 → 26B-a4b MoE). Override lengths via
/// SPANS_TTFT_LENS.
#[cfg(feature = "metal")]
#[test]
#[ignore]
fn bench_first_ttft_spans_overhead_gemma4() {
    let model = if std::env::var_os("GEMMA4_E2E_MOE").is_some() {
        TestModels::GEMMA4_MOE
    } else {
        TestModels::GEMMA4
    };
    run_first_ttft_bench(model, &[2048, 8192]);
}
