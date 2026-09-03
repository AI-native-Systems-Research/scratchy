// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! `scr bench spans` — benchmark relocatable KV cache blocks (spans).
//!
//! Tests all permutations of document ordering to prove that span-enabled
//! prefix caching allows KV cache reuse regardless of order.

use std::time::Instant;

use anyhow::Result;
use indicatif::{ProgressBar, ProgressStyle};
use scratchy_core_config::{CudaGraphConfig, CudaGraphMode};
use scratchy_serving_api::llm::{LLM, LLMBuilder, Prompt, SamplingParams};

use crate::args::BenchSpansArgs;

// ---------------------------------------------------------------------------
// Colors — ANSI 256-color palette for up to 12 distinct documents
// ---------------------------------------------------------------------------

const DOC_COLORS: &[&str] = &[
    "\x1b[38;5;196m", // red
    "\x1b[38;5;46m",  // green
    "\x1b[38;5;33m",  // blue
    "\x1b[38;5;226m", // yellow
    "\x1b[38;5;201m", // magenta
    "\x1b[38;5;51m",  // cyan
    "\x1b[38;5;208m", // orange
    "\x1b[38;5;129m", // purple
    "\x1b[38;5;82m",  // lime
    "\x1b[38;5;197m", // pink
    "\x1b[38;5;39m",  // sky blue
    "\x1b[38;5;214m", // gold
];
const RST: &str = "\x1b[0m";
const DIM: &str = "\x1b[2m";
const BOLD: &str = "\x1b[1m";
const BLOCK_CHAR: char = '\u{2588}'; // full block: █

/// Render a permutation as color-coded block characters.
fn render_perm(perm: &[usize], _doc_blocks: usize) -> String {
    let mut s = String::new();
    for &doc_idx in perm {
        let color = DOC_COLORS[doc_idx % DOC_COLORS.len()];
        s.push_str(color);
        s.push(BLOCK_CHAR);
        s.push(BLOCK_CHAR);
    }
    s.push_str(RST);
    s
}

/// Render the document legend.
fn render_legend(num_docs: usize, _doc_blocks: usize) -> String {
    let mut s = String::new();
    for i in 0..num_docs {
        if i > 0 {
            s.push_str("  ");
        }
        let color = DOC_COLORS[i % DOC_COLORS.len()];
        s.push_str(&format!("Doc {i}="));
        s.push_str(color);
        s.push(BLOCK_CHAR);
        s.push_str(RST);
    }
    s
}

// ---------------------------------------------------------------------------
// Permutation generation
// ---------------------------------------------------------------------------

fn permutations(n: usize, max_perms: usize) -> Vec<Vec<usize>> {
    if n <= 1 {
        return vec![(0..n).collect()];
    }
    let total: usize = (1..=n).product();
    if total <= max_perms {
        // Heap's algorithm — all permutations.
        let mut result = Vec::with_capacity(total);
        let mut a: Vec<usize> = (0..n).collect();
        let mut c = vec![0usize; n];
        result.push(a.clone());
        let mut i = 0;
        while i < n {
            if c[i] < i {
                if i % 2 == 0 {
                    a.swap(0, i);
                } else {
                    a.swap(c[i], i);
                }
                result.push(a.clone());
                c[i] += 1;
                i = 0;
            } else {
                c[i] = 0;
                i += 1;
            }
        }
        result
    } else {
        // Sample random permutations.
        use rand::seq::SliceRandom;
        let mut rng = rand::rng();
        let mut result = Vec::with_capacity(max_perms);
        result.push((0..n).collect());
        result.push((0..n).rev().collect());
        while result.len() < max_perms {
            let mut perm: Vec<usize> = (0..n).collect();
            perm.shuffle(&mut rng);
            if !result.contains(&perm) {
                result.push(perm);
            }
        }
        result
    }
}

// ---------------------------------------------------------------------------
// Prompt construction
// ---------------------------------------------------------------------------

fn pad_to_block(tokens: &[u32], block_size: usize, pad_token: u32) -> Vec<u32> {
    let remainder = tokens.len() % block_size;
    if remainder == 0 {
        return tokens.to_vec();
    }
    let pad_count = block_size - remainder;
    let mut padded = tokens.to_vec();
    padded.extend(std::iter::repeat_n(pad_token, pad_count));
    padded
}

fn make_document(doc_id: u32, block_size: usize, doc_blocks: usize, pad_token: u32) -> Vec<u32> {
    let mut tokens = Vec::with_capacity(block_size * doc_blocks);
    for b in 0..doc_blocks {
        for j in 0..block_size {
            tokens.push(1000 + doc_id * 1000 + b as u32 * 100 + j as u32);
        }
    }
    pad_to_block(&tokens, block_size, pad_token)
}

/// Fresh, in-vocab, seed-unique documents — every call with a new `seed` yields
/// tokens that have never appeared before, so a prefill of them is a guaranteed
/// cache MISS (a genuine COLD prefill) even on a prefix-caching-ON model. This is
/// how the sparse-attention section forces cold prefills on the ONE already-loaded
/// model instead of loading a second (caching-off) model.
fn fresh_docs(
    seed: u32,
    num_docs: usize,
    doc_blocks: usize,
    block_size: usize,
    pad: u32,
) -> Vec<Vec<u32>> {
    (0..num_docs)
        .map(|d| {
            let mut t = Vec::with_capacity(block_size * doc_blocks);
            for b in 0..doc_blocks {
                for j in 0..block_size {
                    let h = seed.wrapping_mul(2_654_435_761)
                        ^ (d as u32).wrapping_mul(40_503)
                        ^ (b as u32).wrapping_mul(2_246_822_519)
                        ^ (j as u32).wrapping_mul(3_266_489_917);
                    t.push(1000 + (h % 100_000)); // safely inside any model's vocab
                }
            }
            pad_to_block(&t, block_size, pad)
        })
        .collect()
}

/// Build a prompt from ordered documents + query.
/// When `with_annotations` is true, each document's blocks are annotated as
/// `Relocatable` for span-aware caching.
fn build_prompt(
    documents: &[Vec<u32>],
    order: &[usize],
    query_base: u32,
    query_len: usize,
    block_size: usize,
    with_annotations: bool,
) -> Prompt {
    let mut tokens = Vec::new();
    let mut annotations = std::collections::BTreeMap::new();

    for &doc_idx in order {
        let doc = &documents[doc_idx];
        let start_block = tokens.len() / block_size;
        tokens.extend_from_slice(doc);
        if with_annotations {
            // Each DOCUMENT is ONE span: all of its blocks share the doc's first
            // token, so only the doc's FIRST block resets the parent chain
            // (hash_block_with_parent) and the whole doc attends block-diagonally
            // as a unit. (Bug fixed: previously each block used `b*block_size`,
            // making every 16-token block its own span.)
            let doc_first_token = (start_block * block_size) as u32;
            let end_block = tokens.len() / block_size;
            for b in start_block..end_block {
                annotations.insert(
                    b,
                    scratchy_core_common::BlockKind::Relocatable {
                        first_token: doc_first_token,
                    },
                );
            }
        }
    }
    for j in 0..query_len {
        tokens.push(query_base + j as u32);
    }

    if with_annotations && !annotations.is_empty() {
        Prompt::TokenIdsWithAnnotations(tokens, annotations)
    } else {
        Prompt::TokenIds(tokens)
    }
}

// ---------------------------------------------------------------------------
// LLM construction
// ---------------------------------------------------------------------------

fn build_llm(args: &BenchSpansArgs, prefix_caching: bool) -> Result<LLM> {
    let model = args.resolved_model().map_err(|e| anyhow::anyhow!(e))?;
    let mut builder = LLMBuilder::new(&model)
        .device(&args.device)
        .dtype(&args.dtype)
        .gpu_memory_utilization(args.gpu_memory_utilization)
        .max_num_seqs(args.max_num_seqs)
        .block_size(args.block_size)
        .enforce_eager(args.enforce_eager)
        .enable_prefix_caching(prefix_caching);

    let total_doc_tokens = args.num_docs * args.doc_blocks * args.block_size;
    let max_batched = total_doc_tokens + args.query_len + 512;
    builder = builder.max_num_batched_tokens(max_batched.max(8192));

    if let Some(len) = args.max_model_len {
        builder = builder.max_model_len(len);
    }
    if let Some(ref token) = args.hf_token {
        builder = builder.hf_token(token);
    }
    if let Some(ref gguf) = args.gguf_file {
        builder = builder.gguf_file(gguf);
    }

    if !args.enforce_eager {
        let sizes = CudaGraphConfig::parse_sizes("auto");
        if !sizes.is_empty() {
            builder = builder.cuda_graph_config(CudaGraphConfig {
                enabled: true,
                mode: CudaGraphMode::default(),
                capture_sizes: sizes,
                num_warmups: 3,
            });
        }
    }

    builder.build()
}

// ---------------------------------------------------------------------------
// Main benchmark
// ---------------------------------------------------------------------------

pub(crate) fn run_bench_spans(args: BenchSpansArgs) -> Result<()> {
    scratchy_core_common::telemetry::init_tracing(&args.log_level);

    if let Some(n) = args.nested {
        return run_bench_nested(&args, n);
    }

    let block_size = args.block_size;
    let num_docs = args.num_docs;
    let doc_blocks = args.doc_blocks;
    let query_len = args.query_len;
    let pad_token = args.pad_token;
    let doc_tokens = doc_blocks * block_size;
    let total_cached = num_docs * doc_tokens;
    let max_perms = args.max_perms;

    // All documents use the same token data — the difference is whether
    // block annotations are provided (relocatable caching) or not.
    let docs: Vec<Vec<u32>> = (0..num_docs as u32)
        .map(|i| make_document(i, block_size, doc_blocks, pad_token))
        .collect();

    let perms = permutations(num_docs, max_perms);
    let num_perms = perms.len();
    let total_factorial: usize = (1..=num_docs).product();

    // Output length: 1 = TTFT-only (legacy). >1 generates real decode so the
    // engine reports avg_itl_s (TPOT). Set SPANS_OUTPUT_LEN to measure decode.
    let output_len: u32 = std::env::var("SPANS_OUTPUT_LEN")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let sampling = SamplingParams {
        max_tokens: Some(output_len),
        temperature: 0.0,
        ignore_eos: true,
        detokenize: false,
        ..SamplingParams::default()
    };

    let canonical: Vec<usize> = (0..num_docs).collect();

    let spinner_style = ProgressStyle::with_template("  {spinner:.cyan} {msg}")
        .unwrap()
        .tick_strings(&[
            "\u{28fb}", "\u{28fd}", "\u{28fe}", "\u{28f7}", "\u{28ef}", "\u{28df}", "\u{287f}",
            "\u{28bf}", "\u{2847}", "\u{280b}", "\u{281b}", "\u{2839}", "\u{2838}",
        ]);

    // -- Header --
    eprintln!();
    eprintln!("{BOLD}vLLM Rust \u{2014} spans benchmark{RST}");
    eprintln!(
        "  {num_docs} docs x {doc_tokens} tok/doc ({total_cached} cached) + {query_len} query"
    );
    eprintln!("  {}", render_legend(num_docs, doc_blocks));
    if num_perms < total_factorial {
        eprintln!("  Testing {num_perms}/{total_factorial} permutations (sampled)");
    } else {
        eprintln!("  Testing all {num_perms} permutations");
    }

    // -----------------------------------------------------------------------
    // Helper: run a sequence of permutations and return per-perm latencies.
    // First request (canonical) populates cache; remaining are measured.
    // -----------------------------------------------------------------------
    let run_perms = |llm: &mut LLM,
                     docs: &[Vec<u32>],
                     perms: &[Vec<usize>],
                     query_base_start: u32,
                     label: &str,
                     with_annotations: bool|
     -> Result<(f64, Vec<f64>)> {
        // Populate cache with canonical order.
        llm.reset_prefix_cache()?;
        let perm_str = render_perm(&canonical, doc_blocks);
        let pb = ProgressBar::new_spinner()
            .with_style(spinner_style.clone())
            .with_message(format!("{perm_str}  {DIM}populate ({label})...{RST}"));
        pb.enable_steady_tick(std::time::Duration::from_millis(80));

        let populate_prompt = build_prompt(
            docs,
            &canonical,
            query_base_start,
            query_len,
            block_size,
            with_annotations,
        );
        let pop_start = Instant::now();
        llm.generate(&[populate_prompt], Some(sampling.clone()))?;
        let populate_ms = pop_start.elapsed().as_secs_f64() * 1000.0;

        pb.finish_and_clear();
        eprintln!("    {perm_str}  {BOLD}{populate_ms:>8.1}ms{RST}  {DIM}populate ({label}){RST}");

        // Run each permutation and measure.
        let mut latencies = Vec::with_capacity(perms.len());
        let mut ttfts = Vec::with_capacity(perms.len());
        let mut tpots = Vec::with_capacity(perms.len());
        for (pi, perm) in perms.iter().enumerate() {
            let perm_str = render_perm(perm, doc_blocks);
            let query_base = query_base_start + 1000 + (pi as u32) * 1000;

            let pb = ProgressBar::new_spinner()
                .with_style(spinner_style.clone())
                .with_message(format!(
                    "{perm_str}  {DIM}{}/{} ({label})...{RST}",
                    pi + 1,
                    perms.len()
                ));
            pb.enable_steady_tick(std::time::Duration::from_millis(80));

            let prompt = build_prompt(
                docs,
                perm,
                query_base,
                query_len,
                block_size,
                with_annotations,
            );
            let start = Instant::now();
            let out = llm.generate(&[prompt], Some(sampling.clone()))?;
            let ms = start.elapsed().as_secs_f64() * 1000.0;
            // Engine-measured TTFT and avg inter-token latency (TPOT). TPOT is
            // None when only 1 token is generated (set SPANS_OUTPUT_LEN > 1).
            let ttft = out[0].ttft_s.unwrap_or(0.0) * 1000.0;
            let tpot = out[0].avg_itl_s.unwrap_or(0.0) * 1000.0;

            pb.finish_and_clear();
            if output_len > 1 {
                eprintln!(
                    "    {perm_str}  {BOLD}TTFT {ttft:>7.1}ms  TPOT {tpot:>6.2}ms/tok{RST}  {DIM}{label}{RST}"
                );
            } else {
                eprintln!("    {perm_str}  {BOLD}{ms:>8.1}ms{RST}  {DIM}{label}{RST}");
            }
            latencies.push(ms);
            ttfts.push(ttft);
            tpots.push(tpot);
        }
        if output_len > 1 {
            let avg_ttft = ttfts.iter().sum::<f64>() / ttfts.len().max(1) as f64;
            let avg_tpot = tpots.iter().sum::<f64>() / tpots.len().max(1) as f64;
            eprintln!(
                "    {BOLD}avg TTFT {avg_ttft:.1}ms  avg TPOT {avg_tpot:.2}ms/tok{RST}  \
                 {DIM}({label}, {output_len} out tok){RST}"
            );
        }
        Ok((populate_ms, latencies))
    };

    // -----------------------------------------------------------------------
    // Without spans: no block annotations.
    // Normal prefix caching — block hashes chain by position, so reordering
    // documents breaks cache hits.
    // -----------------------------------------------------------------------
    eprintln!();
    eprintln!("{BOLD}Without spans{RST} {DIM}(prefix caching, order-dependent){RST}");

    let mut llm = build_llm(&args, true)?;
    // Skip perms[0] (canonical order) — it's the populate step and would
    // always cache-hit even without spans, biasing results.
    let test_perms: Vec<Vec<usize>> = perms.iter().filter(|p| *p != &canonical).cloned().collect();
    let (no_spans_populate, no_spans_latencies) =
        run_perms(&mut llm, &docs, &test_perms, 50000, "no spans", false)?;

    // -----------------------------------------------------------------------
    // With spans: documents have Relocatable block annotations.
    // Span-aware hashing resets parent chain — blocks cache independently
    // of position, so reordering still gets full cache hits.
    // -----------------------------------------------------------------------
    eprintln!();
    eprintln!("{BOLD}With spans{RST} {DIM}(prefix caching, order-independent){RST}");

    let (spans_populate, spans_latencies) =
        run_perms(&mut llm, &docs, &test_perms, 60000, "spans", true)?;

    // -----------------------------------------------------------------------
    // Sparse attention (COLD prefill): block-diagonal vs dense causal.
    // -----------------------------------------------------------------------
    // With Relocatable annotations each span attends ONLY its own tokens
    // (block-diagonal mask), so a prefill costs O(sum span_i^2) instead of dense
    // O(total^2). This reuses the ONE already-loaded model (no second load): every
    // prefill uses FRESH unique tokens (`fresh_docs`) ⇒ a guaranteed cache MISS ⇒
    // a genuine cold prefill even with prefix caching on. Dense vs block-diagonal
    // differ only in the annotations. Warm up + discard trial 0 (cold-start /
    // graph-capture). Uses the engine's TTFT (prefill time). The speedup grows
    // with context (attention O(T^2)) and with more/larger spans.
    eprintln!();
    eprintln!(
        "{BOLD}Sparse attention{RST} {DIM}(cold prefill, {total_cached}-tok context: block-diagonal vs dense causal){RST}"
    );

    // Free the reuse measurement's cached blocks so the pool has room for the
    // cold prefills below (the fresh tokens miss the cache regardless).
    llm.reset_prefix_cache()?;

    let mut cold_speedups: Vec<f64> = Vec::new();
    {
        let trials: usize = std::env::var("SPANS_SPARSE_TRIALS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(5);
        for t in 0..=trials {
            // Fresh, unique docs per prefill ⇒ every prefill is a cold cache miss.
            let dense_docs = fresh_docs(
                2 * t as u32 + 1,
                num_docs,
                doc_blocks,
                block_size,
                pad_token,
            );
            let dense = llm.generate(
                &[build_prompt(
                    &dense_docs,
                    &canonical,
                    80000,
                    query_len,
                    block_size,
                    false,
                )],
                Some(sampling.clone()),
            )?;
            let dense_ttft = dense[0].ttft_s.unwrap_or(0.0) * 1000.0;
            let sparse_docs = fresh_docs(
                2 * t as u32 + 2,
                num_docs,
                doc_blocks,
                block_size,
                pad_token,
            );
            let sparse = llm.generate(
                &[build_prompt(
                    &sparse_docs,
                    &canonical,
                    80000,
                    query_len,
                    block_size,
                    true,
                )],
                Some(sampling.clone()),
            )?;
            let sparse_ttft = sparse[0].ttft_s.unwrap_or(0.0) * 1000.0;
            let sp = if sparse_ttft > 0.0 {
                dense_ttft / sparse_ttft
            } else {
                0.0
            };
            if t == 0 {
                eprintln!(
                    "    {DIM}warmup    dense {dense_ttft:>8.1}ms  block-diag {sparse_ttft:>8.1}ms  {sp:.2}x (discarded){RST}"
                );
                continue;
            }
            eprintln!(
                "    trial {t}/{trials}  dense {BOLD}{dense_ttft:>8.1}ms{RST}  block-diag {BOLD}{sparse_ttft:>8.1}ms{RST}  {BOLD}{sp:.2}x{RST}"
            );
            cold_speedups.push(sp);
        }
    }
    drop(llm);
    cold_speedups.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // -----------------------------------------------------------------------
    // Summary
    // -----------------------------------------------------------------------
    // Compute per-permutation speedups: no_spans[i] / spans[i]
    let mut speedups: Vec<f64> = no_spans_latencies
        .iter()
        .zip(spans_latencies.iter())
        .map(|(&ns, &sp)| ns / sp)
        .collect();
    speedups.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let p50 = |v: &[f64]| {
        let mid = v.len() / 2;
        if v.len().is_multiple_of(2) && v.len() > 1 {
            (v[mid - 1] + v[mid]) / 2.0
        } else {
            v[mid]
        }
    };

    let no_spans_avg = no_spans_latencies.iter().sum::<f64>() / no_spans_latencies.len() as f64;
    let spans_avg = spans_latencies.iter().sum::<f64>() / spans_latencies.len() as f64;

    eprintln!();
    println!("{BOLD}=== Results ({} perms) ==={RST}", test_perms.len());
    println!();
    println!(
        "  Without spans:  avg {no_spans_avg:>7.1}ms  {DIM}({:.1}x vs populate){RST}",
        no_spans_populate / no_spans_avg
    );
    println!(
        "  With spans:     avg {spans_avg:>7.1}ms  {DIM}({:.1}x vs populate){RST}",
        spans_populate / spans_avg
    );
    println!();
    println!(
        "  {BOLD}Reuse speedup{RST}  {DIM}(reorder cache hit){RST}  min {BOLD}{:.1}x{RST}  p50 {BOLD}{:.1}x{RST}  max {BOLD}{:.1}x{RST}",
        speedups.first().unwrap_or(&0.0),
        p50(&speedups),
        speedups.last().unwrap_or(&0.0),
    );
    if !cold_speedups.is_empty() {
        println!(
            "  {BOLD}Sparse-attn speedup{RST}  {DIM}(cold prefill){RST}  min {BOLD}{:.2}x{RST}  p50 {BOLD}{:.2}x{RST}  max {BOLD}{:.2}x{RST}",
            cold_speedups.first().unwrap_or(&0.0),
            p50(&cold_speedups),
            cold_speedups.last().unwrap_or(&0.0),
        );
    }
    println!();

    Ok(())
}

// ---------------------------------------------------------------------------
// Nested generate benchmark
// ---------------------------------------------------------------------------

fn run_bench_nested(args: &BenchSpansArgs, num_inner: usize) -> Result<()> {
    let inner_tokens = args.inner_tokens;
    let block_size = args.block_size;

    let mut llm = build_llm(args, true)?;
    let model = llm.model_name().to_string();

    let inner_sampling = SamplingParams {
        max_tokens: Some(inner_tokens),
        temperature: 0.0,
        detokenize: true,
        ..SamplingParams::default()
    };

    let outer_sampling = SamplingParams {
        max_tokens: Some(1),
        temperature: 0.0,
        ignore_eos: true,
        detokenize: false,
        ..SamplingParams::default()
    };

    // Inner prompt texts (shared between baseline and spans paths).
    let prompts: Vec<String> = (0..num_inner)
        .map(|i| {
            format!(
                "Document {i}: This is a unique synthetic document number {i} \
                 with enough content to fill multiple KV cache blocks for the \
                 nested spans benchmark test. Content seed: {seed}.",
                seed = i * 12345 + 67890
            )
        })
        .collect();

    eprintln!();
    eprintln!("{BOLD}vLLM Rust \u{2014} nested spans benchmark{RST}");
    eprintln!("  {num_inner} inner generates x {inner_tokens} tokens each");

    // =====================================================================
    // Baseline: normal generate() — no spans, no SPNL, no seal
    // =====================================================================
    eprintln!();
    eprintln!("{BOLD}--- Baseline (normal generate) ---{RST}");

    use scratchy_serving_api::llm::ChatMessage;

    let baseline_inner_start = Instant::now();
    let mut baseline_inner_tokens: Vec<Vec<u32>> = Vec::with_capacity(num_inner);
    for (i, prompt) in prompts.iter().enumerate() {
        let start = Instant::now();
        let result = llm.chat(&[ChatMessage::user(prompt)], Some(inner_sampling.clone()))?;
        let ms = start.elapsed().as_secs_f64() * 1000.0;
        eprintln!(
            "    inner[{i}]  {BOLD}{ms:>8.1}ms{RST}  {DIM}{} output tokens{RST}",
            result.outputs[0].token_ids.len()
        );
        let mut all_toks = result.prompt_token_ids.clone();
        all_toks.extend_from_slice(&result.outputs[0].token_ids);
        baseline_inner_tokens.push(all_toks);
    }
    let baseline_inner_ms = baseline_inner_start.elapsed().as_secs_f64() * 1000.0;

    // Build outer prompt from raw token IDs (same as spans path for fair comparison).
    let tokenizer = llm
        .tokenizer()
        .ok_or_else(|| anyhow::anyhow!("nested bench requires a tokenizer"))?;
    let mut baseline_outer_tokens: Vec<u32> = Vec::new();
    for inner_toks in &baseline_inner_tokens {
        baseline_outer_tokens.extend_from_slice(inner_toks);
    }
    let query_ids = tokenizer.encode("Summarize all documents.", false)?;
    baseline_outer_tokens.extend_from_slice(&query_ids);

    let start = Instant::now();
    llm.generate(
        &[Prompt::TokenIds(baseline_outer_tokens)],
        Some(outer_sampling.clone()),
    )?;
    let baseline_outer_ms = start.elapsed().as_secs_f64() * 1000.0;
    eprintln!("    outer     {BOLD}{baseline_outer_ms:>8.1}ms{RST}  {DIM}full prefill{RST}");

    // =====================================================================
    // Reset cache — clear everything before the spans path
    // =====================================================================
    llm.reset_prefix_cache()?;

    // =====================================================================
    // Spans: execute_query with seal + relocatable annotations
    // =====================================================================
    // Spans: single nested execute_query call.
    // execute_spnl_query_sync runs inners with seal+volatile, builds the
    // Relocatable-annotated outer prompt, runs the outer, and returns
    // QueryOutput with per-step timing for inner and outer separately.
    // =====================================================================
    eprintln!();
    eprintln!("{BOLD}--- Spans (execute_query, nested) ---{RST}");

    let mut seq_children: Vec<serde_json::Value> = prompts
        .iter()
        .map(|p| {
            serde_json::json!({
                "g": {
                    "model": model,
                    "max_tokens": inner_tokens,
                    "temperature": 0.0,
                    "input": { "plus": [{ "user": p }] }
                }
            })
        })
        .collect();
    seq_children.push(serde_json::json!({ "user": "Summarize all documents." }));
    let nested_query = serde_json::json!({
        "g": {
            "model": model,
            "max_tokens": 1,
            "temperature": 0.0,
            "input": { "seq": seq_children }
        }
    });

    let result = llm.execute_query(
        &nested_query.to_string(),
        Some(outer_sampling.clone()),
        false, // seal
        false, // volatile
    )?;

    let mut spans_inner_ms = 0.0f64;
    let mut expected_cached_tokens: usize = 0;
    for step in result.inner_steps() {
        let prompt_toks = step.output.prompt_token_ids.len();
        let output_toks = step.output.outputs.first().map_or(0, |o| o.token_ids.len());
        let total = prompt_toks + output_toks;
        let sealed = total / block_size * block_size;
        expected_cached_tokens += sealed;
        spans_inner_ms += step.elapsed_ms;
        eprintln!(
            "    {}  {BOLD}{:>8.1}ms{RST}  {DIM}{prompt_toks} prompt + {output_toks} output = {total} ({sealed} sealed){RST}",
            step.label, step.elapsed_ms,
        );
    }
    eprintln!("    expected cached in outer: {BOLD}{expected_cached_tokens}{RST} tokens");

    let outer_step = result.outer_step();
    let outer_prompt_toks = outer_step.output.prompt_token_ids.len();
    let spans_outer_ms = outer_step.elapsed_ms;
    eprintln!(
        "    outer     {BOLD}{spans_outer_ms:>8.1}ms{RST}  {DIM}outer prompt: {outer_prompt_toks} tokens, {} blocks — span cache hit{RST}",
        outer_prompt_toks.div_ceil(block_size),
    );

    drop(llm);

    // =====================================================================
    // Summary
    // =====================================================================
    let baseline_total = baseline_inner_ms + baseline_outer_ms;
    let spans_total_ms = spans_inner_ms + spans_outer_ms;

    eprintln!();
    println!("{BOLD}=== Nested Spans Results ==={RST}");
    println!();
    println!(
        "  Baseline:  inner {BOLD}{baseline_inner_ms:>8.1}ms{RST}  outer {BOLD}{baseline_outer_ms:>8.1}ms{RST}  total {BOLD}{baseline_total:>8.1}ms{RST}"
    );
    println!(
        "  Spans:     inner {BOLD}{spans_inner_ms:>8.1}ms{RST}  outer {BOLD}{spans_outer_ms:>8.1}ms{RST}  total {BOLD}{spans_total_ms:>8.1}ms{RST}"
    );
    println!();
    println!(
        "  Outer speedup: {BOLD}{:.1}x{RST}",
        baseline_outer_ms / spans_outer_ms
    );
    println!(
        "  Total speedup: {BOLD}{:.1}x{RST}",
        baseline_total / spans_total_ms
    );
    println!();

    Ok(())
}
