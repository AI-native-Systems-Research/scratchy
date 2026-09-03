// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! `scr bench longbench` — LongBench v2 long-context accuracy benchmark.
//!
//! Downloads the LongBench v2 dataset from HuggingFace and evaluates
//! long-context understanding across diverse tasks: single/multi-document QA,
//! code repository understanding, in-context learning, structured data, and
//! dialogue history.
//!
//! LongBench v2 uses multiple-choice questions (A/B/C/D) with very long
//! contexts (median ~100K tokens, up to ~4M tokens). Each question has a
//! single unique context — this benchmark tests long-context correctness
//! rather than cross-query cache reuse.
//!
//! Five variants:
//! 1. **Plain** — `llm.chat()`, monolithic context, full prefill every query.
//! 2. **Chunked** — `execute_query()` with `cross`, context split into
//!    paragraph chunks but no `plus` — proves chunking doesn't hurt accuracy.
//! 3. **Spans** — `execute_query()` with `cross` + `plus`, monolithic context
//!    as a single relocatable block.
//! 4. **Spans-chunked** — `execute_query()` with `cross` + `plus`, context
//!    split into chunks as separate relocatable blocks.
//! 5. **Spans-chunked-permuted** — same as spans-chunked but chunks in random
//!    order. Tests position-independence and cross-query chunk reuse.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use indicatif::{ProgressBar, ProgressStyle};
use rand::SeedableRng;
use rand::seq::SliceRandom;
use scratchy_core_config::{CudaGraphConfig, CudaGraphMode};
use scratchy_serving_api::llm::{ChatMessage, LLM, LLMBuilder, SamplingParams};
use scratchy_serving_api::tokenizer::Tokenizer;

use crate::args::BenchLongbenchArgs;

// ---------------------------------------------------------------------------
// Dataset types and loading
// ---------------------------------------------------------------------------

/// A single query in the benchmark.
struct Query {
    question: String,
    choices: [String; 4],
    /// Correct answer: "A", "B", "C", or "D".
    answer: String,
    context: String,
    domain: String,
    difficulty: String,
    length: String,
}

/// The full benchmark dataset.
struct Dataset {
    queries: Vec<Query>,
}

/// Download and parse the LongBench v2 dataset (JSON from HuggingFace).
fn fetch_dataset() -> Result<Dataset> {
    let cache_dir = dirs::cache_dir()
        .ok_or_else(|| anyhow::anyhow!("Could not determine cache directory"))?
        .join("scr-bench")
        .join("longbench-v2");
    std::fs::create_dir_all(&cache_dir)?;
    let cache_file = cache_dir.join("data.json");

    if !cache_file.exists() {
        eprintln!("Downloading LongBench v2 dataset...");
        let url = "https://huggingface.co/datasets/zai-org/LongBench-v2/resolve/main/data.json";
        let agent = crate::http::agent(false);
        let bytes = crate::http::get_bytes(&agent, url)?;
        std::fs::write(&cache_file, &bytes)?;
        eprintln!("Cached {} bytes.", bytes.len());
    }

    let data = std::fs::read_to_string(&cache_file)?;
    let raw: Vec<serde_json::Value> = serde_json::from_str(&data)?;

    let mut queries: Vec<Query> = Vec::new();
    for record in &raw {
        let question = record["question"].as_str().unwrap_or("").to_string();
        let answer = record["answer"].as_str().unwrap_or("").to_string();
        let context = record["context"].as_str().unwrap_or("").to_string();
        let domain = record["domain"].as_str().unwrap_or("").to_string();
        let difficulty = record["difficulty"].as_str().unwrap_or("").to_string();
        let length = record["length"].as_str().unwrap_or("").to_string();

        let choices = [
            record["choice_A"].as_str().unwrap_or("").to_string(),
            record["choice_B"].as_str().unwrap_or("").to_string(),
            record["choice_C"].as_str().unwrap_or("").to_string(),
            record["choice_D"].as_str().unwrap_or("").to_string(),
        ];

        if question.is_empty() || answer.is_empty() || context.is_empty() {
            continue;
        }

        queries.push(Query {
            question,
            choices,
            answer,
            context,
            domain,
            difficulty,
            length,
        });
    }

    Ok(Dataset { queries })
}

// ---------------------------------------------------------------------------
// Tokenizer helpers
// ---------------------------------------------------------------------------

fn token_len(text: &str, tokenizer: &Tokenizer) -> usize {
    tokenizer
        .encode(text, false)
        .map(|ids| ids.len())
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Evaluation
// ---------------------------------------------------------------------------

/// Extract the answer letter from a model response.
///
/// Matches the official LongBench v2 extraction logic:
/// 1. "The correct answer is (X)" — primary pattern
/// 2. "The correct answer is X" — fallback without parens
///
/// Additionally handles common model outputs not covered by the official code:
/// 3. Starts with "(X)" or "X." or "X)" — direct answer
/// 4. Single letter response
fn extract_answer(response: &str) -> Option<String> {
    let cleaned = response.replace('*', "");
    let trimmed = cleaned.trim();
    let upper = trimmed.to_uppercase();

    // Official LongBench v2 pattern: "The correct answer is (X)"
    for letter in &["A", "B", "C", "D"] {
        let pat = format!("THE CORRECT ANSWER IS ({letter})");
        if upper.contains(&pat) {
            return Some(letter.to_string());
        }
    }

    // Official fallback: "The correct answer is X"
    for letter in &["A", "B", "C", "D"] {
        let pat = format!("THE CORRECT ANSWER IS {letter}");
        if upper.contains(&pat) {
            return Some(letter.to_string());
        }
    }

    // Additional: starts with "(X)", "X.", "X)"
    for letter in &["A", "B", "C", "D"] {
        if upper.starts_with(&format!("({letter})"))
            || upper.starts_with(&format!("{letter}."))
            || upper.starts_with(&format!("{letter})"))
        {
            return Some(letter.to_string());
        }
    }

    // Additional: single letter response
    if let Some(first_char) = upper.chars().next().filter(|c| "ABCD".contains(*c)) {
        let second_is_alpha = upper
            .chars()
            .nth(1)
            .is_some_and(|c| c.is_ascii_alphabetic());
        if !second_is_alpha {
            return Some(first_char.to_string());
        }
    }

    None
}

// ---------------------------------------------------------------------------
// Context chunking
// ---------------------------------------------------------------------------

/// Target number of tokens per chunk.
const CHUNK_TARGET_TOKENS: usize = 512;

/// Split a context into paragraph-level chunks of ~CHUNK_TARGET_TOKENS each.
fn split_into_chunks(context: &str, tokenizer: &Tokenizer) -> Vec<String> {
    // Split on double newlines (paragraph boundaries) first.
    let paragraphs: Vec<&str> = context.split("\n\n").collect();

    let mut chunks: Vec<String> = Vec::new();
    let mut current = String::new();

    for para in paragraphs {
        if current.is_empty() {
            current = para.to_string();
        } else {
            let combined = format!("{}\n\n{}", current, para);
            let combined_tokens = token_len(&combined, tokenizer);
            if combined_tokens > CHUNK_TARGET_TOKENS && !current.is_empty() {
                chunks.push(current);
                current = para.to_string();
            } else {
                current = combined;
            }
        }
    }
    if !current.is_empty() {
        chunks.push(current);
    }

    // If we only got 1 chunk (no paragraph breaks), split on single newlines.
    if chunks.len() <= 1 {
        chunks.clear();
        current = String::new();
        for line in context.split('\n') {
            if current.is_empty() {
                current = line.to_string();
            } else {
                let combined = format!("{}\n{}", current, line);
                if token_len(&combined, tokenizer) > CHUNK_TARGET_TOKENS && !current.is_empty() {
                    chunks.push(current);
                    current = line.to_string();
                } else {
                    current = combined;
                }
            }
        }
        if !current.is_empty() {
            chunks.push(current);
        }
    }

    chunks
}

// ---------------------------------------------------------------------------
// Query builders
// ---------------------------------------------------------------------------

/// Build a SPNL query with chunks as sequential messages inside `cross`
/// (no relocatable annotations — proves chunking is neutral).
fn build_chunked_query(
    model: &str,
    preamble: &str,
    chunks: &[String],
    question: &str,
    max_tokens: usize,
) -> serde_json::Value {
    let mut cross_children: Vec<serde_json::Value> = Vec::new();
    cross_children.push(serde_json::json!({ "user": preamble }));
    for chunk in chunks {
        cross_children.push(serde_json::json!({ "user": chunk }));
    }
    cross_children.push(serde_json::json!({ "user": question }));

    serde_json::json!({
        "g": {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": 0.1,
            "input": { "cross": cross_children }
        }
    })
}

/// Build a SPNL query with a single monolithic context in `plus`.
fn build_spans_query(
    model: &str,
    preamble: &str,
    context: &str,
    question: &str,
    max_tokens: usize,
) -> serde_json::Value {
    serde_json::json!({
        "g": {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": 0.1,
            "input": {
                "cross": [
                    { "user": preamble },
                    { "plus": [{ "user": context }] },
                    { "user": question }
                ]
            }
        }
    })
}

/// Build a SPNL query with chunks as separate relocatable blocks in `plus`.
fn build_spans_chunked_query(
    model: &str,
    preamble: &str,
    chunks: &[String],
    question: &str,
    max_tokens: usize,
) -> serde_json::Value {
    let plus_children: Vec<serde_json::Value> = chunks
        .iter()
        .map(|c| serde_json::json!({ "user": c }))
        .collect();

    serde_json::json!({
        "g": {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": 0.1,
            "input": {
                "cross": [
                    { "user": preamble },
                    { "plus": plus_children },
                    { "user": question }
                ]
            }
        }
    })
}

/// Evaluate response: 1.0 if extracted answer matches, 0.0 otherwise.
fn evaluate(response: &str, expected: &str, debug: bool) -> f64 {
    let extracted = extract_answer(response);
    if debug {
        eprintln!("  expected: {expected}");
        eprintln!("  response: {response}");
        eprintln!("  extracted: {:?}", extracted);
    }
    match extracted {
        Some(ref a) if a == expected => 1.0,
        _ => 0.0,
    }
}

// ---------------------------------------------------------------------------
// LLM construction
// ---------------------------------------------------------------------------

fn build_llm(args: &BenchLongbenchArgs) -> Result<LLM> {
    let model = args.resolved_model().map_err(|e| anyhow::anyhow!(e))?;
    let mut builder = LLMBuilder::new(&model)
        .device(&args.device)
        .dtype(&args.dtype)
        .gpu_memory_utilization(args.gpu_memory_utilization)
        .max_num_seqs(args.max_num_seqs)
        .block_size(args.block_size)
        .tensor_parallel_size(args.tensor_parallel_size)
        .enforce_eager(args.enforce_eager)
        .enable_prefix_caching(!args.no_prefix_caching);

    builder = builder.max_num_batched_tokens(8192);

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

/// Per-query results.
#[derive(Clone)]
struct QueryResult {
    acc: f64,
    ttft_ms: f64,
    itl_ms: f64,
    total_ms: f64,
}

// ---------------------------------------------------------------------------
// Per-variant results
// ---------------------------------------------------------------------------

struct VariantResults {
    results: Vec<QueryResult>,
}

impl VariantResults {
    fn new() -> Self {
        Self {
            results: Vec::new(),
        }
    }

    fn avg_acc(&self) -> f64 {
        let n = self.results.len() as f64;
        if n == 0.0 {
            return 0.0;
        }
        self.results.iter().map(|r| r.acc).sum::<f64>() / n
    }

    fn avg_ttft(&self) -> f64 {
        let n = self.results.len() as f64;
        if n == 0.0 {
            return 0.0;
        }
        self.results.iter().map(|r| r.ttft_ms).sum::<f64>() / n
    }
}

// ---------------------------------------------------------------------------
// Stats printing
// ---------------------------------------------------------------------------

fn print_results(
    label: &str,
    plain: &VariantResults,
    chunked: &VariantResults,
    spans: &VariantResults,
    spans_chunked: &VariantResults,
    spans_permuted: &VariantResults,
) {
    let n = plain.results.len();
    if n == 0 {
        return;
    }
    let w = format!("{n}").len();
    let plain_acc = plain.avg_acc();
    let plain_ttft = plain.avg_ttft();

    let fmt_line = |name: &str, r: &VariantResults, suffix: &str| {
        let acc = r.avg_acc();
        let perfect = r.results.iter().filter(|r| r.acc >= 1.0).count();
        let avg_ttft = r.avg_ttft();
        let avg_itl = r.results.iter().map(|r| r.itl_ms).sum::<f64>() / n as f64;
        let avg_total = r.results.iter().map(|r| r.total_ms).sum::<f64>() / n as f64;
        eprintln!(
            "    {name:>16}: acc={:>5.1}%  correct={:>w$}/{}  ttft={:>6.0}ms  itl={:>4.1}ms  total={:>6.0}ms{suffix}",
            acc * 100.0,
            perfect,
            n,
            avg_ttft,
            avg_itl,
            avg_total,
            w = w,
        );
    };

    let vs_plain = |r: &VariantResults| -> String {
        let ttft = r.avg_ttft();
        let acc = r.avg_acc();
        format!(
            "  ({:.2}x, {:+.1}pp)",
            plain_ttft / ttft.max(0.001),
            (acc - plain_acc) * 100.0,
        )
    };

    eprintln!("  {label}");
    fmt_line("plain", plain, "");
    fmt_line("chunked", chunked, &vs_plain(chunked));
    fmt_line("spans", spans, &vs_plain(spans));
    fmt_line("spans-chunked", spans_chunked, &vs_plain(spans_chunked));
    fmt_line(
        "spans-chunk-perm",
        spans_permuted,
        &vs_plain(spans_permuted),
    );
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

pub(crate) fn run_bench_longbench(args: BenchLongbenchArgs) -> Result<()> {
    scratchy_core_common::telemetry::init_tracing(&args.log_level);

    let mut dataset = fetch_dataset()?;
    let total_loaded = dataset.queries.len();

    if total_loaded == 0 {
        eprintln!("No queries to process.");
        return Ok(());
    }

    // Apply --max-context-chars filter if specified.
    if let Some(max_chars) = args.max_context_chars {
        let before = dataset.queries.len();
        dataset.queries.retain(|q| q.context.len() <= max_chars);
        let skipped = before - dataset.queries.len();
        if skipped > 0 {
            eprintln!(
                "\x1b[33mWarning: skipped {}/{} queries exceeding --max-context-chars={}\x1b[0m",
                skipped, before, max_chars
            );
        }
    }

    let mut llm = build_llm(&args)?;
    let model_name = llm.model_name().to_string();
    let max_model_len = llm.max_model_len();

    eprintln!("\n=== LongBench v2 Benchmark ===");
    eprintln!("Model:        {}", model_name);
    eprintln!("Max seq len:  {} tokens", max_model_len);
    let tokenizer: Arc<Tokenizer> = llm
        .tokenizer()
        .ok_or_else(|| anyhow::anyhow!("LongBench bench requires a tokenizer"))?
        .clone();

    // Auto-filter queries that exceed the model's max context length.
    // First pass: cheap character-based estimate (chars/4 ≈ tokens) to discard
    // obviously too-long contexts without tokenizing them.
    let overhead_tokens = 200;
    let max_context_tokens = max_model_len.saturating_sub(overhead_tokens);
    let before_filter = dataset.queries.len();
    // Cheap pre-filter: skip anything whose character count alone guarantees it won't fit.
    // Use chars/3 as a conservative lower bound on token count (tokens >= chars/4 typically).
    let cheap_max_chars = max_context_tokens * 3;
    dataset
        .queries
        .retain(|q| q.context.len() <= cheap_max_chars);
    let cheap_skipped = before_filter - dataset.queries.len();

    // Second pass: precise tokenization for borderline cases (parallelized).
    let before_precise = dataset.queries.len();

    let filter_pb = ProgressBar::new(dataset.queries.len() as u64).with_style(
        ProgressStyle::default_bar()
            .template("  filtering {bar:40.white/white} {pos:>4}/{len} (pre-filtered {msg})")
            .unwrap(),
    );
    filter_pb.set_message(format!("{} by char count", cheap_skipped));

    // Tokenize all candidates in parallel using std::thread::scope.
    let token_counts: Vec<usize> = {
        let queries = &dataset.queries;
        let tok = &tokenizer;
        let pb = &filter_pb;
        let n = queries.len();
        if n == 0 {
            filter_pb.finish_and_clear();
            eprintln!("No queries remain after character pre-filter.");
            return Ok(());
        }
        let n_threads = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(4)
            .min(n);
        let chunk_size = n.div_ceil(n_threads);
        let mut results = vec![0usize; n];
        std::thread::scope(|s| {
            let chunks: Vec<_> = results.chunks_mut(chunk_size).enumerate().collect();
            let handles: Vec<_> = chunks
                .into_iter()
                .map(|(chunk_idx, result_chunk)| {
                    let offset = chunk_idx * chunk_size;
                    s.spawn(move || {
                        for (i, slot) in result_chunk.iter_mut().enumerate() {
                            *slot = token_len(&queries[offset + i].context, tok);
                            pb.inc(1);
                        }
                    })
                })
                .collect();
            for h in handles {
                h.join().unwrap();
            }
        });
        results
    };
    filter_pb.finish_and_clear();

    // Filter using the computed token counts.
    let mut skipped_token_counts: Vec<usize> = Vec::new();
    let keep_indices: Vec<usize> = token_counts
        .iter()
        .enumerate()
        .filter(|(_, tc)| {
            let tc = **tc;
            if tc > max_context_tokens {
                skipped_token_counts.push(tc);
                false
            } else {
                true
            }
        })
        .map(|(i, _)| i)
        .collect();

    // Keep only the queries (and their token counts) that passed the filter.
    let kept_token_counts: Vec<usize> = keep_indices.iter().map(|&i| token_counts[i]).collect();
    let mut kept_queries: Vec<Query> = Vec::with_capacity(keep_indices.len());
    let mut old_queries = std::mem::take(&mut dataset.queries);
    for &i in keep_indices.iter().rev() {
        kept_queries.push(old_queries.swap_remove(i));
    }
    kept_queries.reverse();
    dataset.queries = kept_queries;

    let precise_skipped = before_precise - dataset.queries.len();
    let total_skipped = cheap_skipped + precise_skipped;
    if total_skipped > 0 {
        eprintln!(
            "\x1b[33mWarning: skipped {}/{} queries exceeding model max_seq_len={}\x1b[0m",
            total_skipped, before_filter, max_model_len,
        );
    }

    if dataset.queries.is_empty() {
        eprintln!("No queries fit within the model's max sequence length.");
        return Ok(());
    }

    // Apply -n limit AFTER filtering (so -n 1 gives the first query that fits).
    let mut context_tokens: Vec<usize> = kept_token_counts;
    if let Some(n) = args.num_queries {
        dataset.queries.truncate(n);
        context_tokens.truncate(n);
    }

    let n_queries = dataset.queries.len();
    let total_tokens: usize = context_tokens.iter().sum();
    let mut sorted_tokens = context_tokens.clone();
    sorted_tokens.sort();
    let median_tokens = sorted_tokens[sorted_tokens.len() / 2];

    // Domain distribution.
    let mut domain_counts: HashMap<&str, usize> = HashMap::new();
    for q in &dataset.queries {
        *domain_counts.entry(q.domain.as_str()).or_insert(0) += 1;
    }

    // Difficulty distribution.
    let mut diff_counts: HashMap<&str, usize> = HashMap::new();
    for q in &dataset.queries {
        *diff_counts.entry(q.difficulty.as_str()).or_insert(0) += 1;
    }

    // Length distribution.
    let mut len_counts: HashMap<&str, usize> = HashMap::new();
    for q in &dataset.queries {
        *len_counts.entry(q.length.as_str()).or_insert(0) += 1;
    }

    eprintln!("Queries:      {} (of {} total)", n_queries, total_loaded);
    eprintln!(
        "Context:      {} total tokens, median={}, min={}, max={}",
        total_tokens,
        median_tokens,
        sorted_tokens.first().unwrap_or(&0),
        sorted_tokens.last().unwrap_or(&0),
    );
    eprintln!(
        "Difficulty:   {}",
        ["easy", "hard"]
            .iter()
            .filter_map(|d| diff_counts.get(d).map(|c| format!("{d}={c}")))
            .collect::<Vec<_>>()
            .join(", ")
    );
    eprintln!(
        "Length:       {}",
        ["short", "medium", "long"]
            .iter()
            .filter_map(|l| len_counts.get(l).map(|c| format!("{l}={c}")))
            .collect::<Vec<_>>()
            .join(", ")
    );
    let mut domain_sorted: Vec<(&&str, &usize)> = domain_counts.iter().collect();
    domain_sorted.sort_by(|a, b| b.1.cmp(a.1));
    eprintln!(
        "Domains:      {}",
        domain_sorted
            .iter()
            .map(|(d, c)| format!("{d}={c}"))
            .collect::<Vec<_>>()
            .join(", ")
    );

    let sampling = SamplingParams {
        max_tokens: Some(args.max_tokens as u32),
        temperature: 0.1, // matches official LongBench v2
        ..SamplingParams::default()
    };

    // No system prompt — matches official LongBench v2 (single user message).
    // The prompt template matches prompts/0shot.txt from the official repo.

    // Pre-chunk all contexts (parallelized — tokenization is the bottleneck).
    let chunk_pb = ProgressBar::new(n_queries as u64).with_style(
        ProgressStyle::default_bar()
            .template("  chunking {bar:40.white/white} {pos:>4}/{len}")
            .unwrap(),
    );
    let all_chunks: Vec<Vec<String>> = {
        let queries = &dataset.queries;
        let tok = &tokenizer;
        let pb = &chunk_pb;
        let n = queries.len();
        let n_threads = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(4)
            .min(n.max(1));
        let chunk_size = n.div_ceil(n_threads.max(1));
        let mut results: Vec<Vec<String>> = (0..n).map(|_| Vec::new()).collect();
        std::thread::scope(|s| {
            let chunks: Vec<_> = results.chunks_mut(chunk_size).enumerate().collect();
            let handles: Vec<_> = chunks
                .into_iter()
                .map(|(chunk_idx, result_chunk)| {
                    let offset = chunk_idx * chunk_size;
                    s.spawn(move || {
                        for (i, slot) in result_chunk.iter_mut().enumerate() {
                            *slot = split_into_chunks(&queries[offset + i].context, tok);
                            pb.inc(1);
                        }
                    })
                })
                .collect();
            for h in handles {
                h.join().unwrap();
            }
        });
        results
    };
    chunk_pb.finish_and_clear();
    let total_chunks: usize = all_chunks.iter().map(|c| c.len()).sum();
    let avg_chunks = total_chunks as f64 / n_queries as f64;
    eprintln!(
        "Chunks:       {} total, {:.1} avg/query",
        total_chunks, avg_chunks,
    );

    let max_tokens = args.max_tokens;
    let debug = args.debug;

    // Build the full prompt matching official LongBench v2 0shot template.
    // The entire prompt is a single user message (no system prompt).
    // Prompt template matching official LongBench v2 prompts/0shot.txt exactly.
    // NOTE: no leading whitespace — Rust `\` continuations preserve indentation.
    let make_prompt = |q: &Query| -> String {
        format!(
            "Please read the following text and answer the question below.\n\
\n\
<text>\n\
{}\n\
</text>\n\
\n\
What is the correct answer to this question: {}\n\
Choices:\n\
(A) {}\n\
(B) {}\n\
(C) {}\n\
(D) {}\n\
\n\
Format your response as follows: \"The correct answer is (insert answer here)\".",
            q.context.trim(),
            q.question.trim(),
            q.choices[0].trim(),
            q.choices[1].trim(),
            q.choices[2].trim(),
            q.choices[3].trim(),
        )
    };

    // Build just the question portion (for chunked variants where context is separate).
    let make_question_only = |q: &Query| -> String {
        format!(
            "What is the correct answer to this question: {}\n\
Choices:\n\
(A) {}\n\
(B) {}\n\
(C) {}\n\
(D) {}\n\
\n\
Format your response as follows: \"The correct answer is (insert answer here)\".",
            q.question.trim(),
            q.choices[0].trim(),
            q.choices[1].trim(),
            q.choices[2].trim(),
            q.choices[3].trim(),
        )
    };

    // Helper: run one variant over all queries.
    let run_variant = |llm: &mut LLM,
                       label: &str,
                       style: ProgressStyle,
                       reset_each: bool,
                       build_fn: &dyn Fn(usize, &Query) -> Result<Option<serde_json::Value>>|
     -> Result<VariantResults> {
        let mut vr = VariantResults::new();
        let pb = ProgressBar::new(n_queries as u64)
            .with_style(style)
            .with_message(label.to_string());

        for (i, query) in dataset.queries.iter().enumerate() {
            if reset_each {
                llm.reset_prefix_cache()?;
            }

            let (response_text, ttft_ms, itl_ms, total_ms) = if let Some(spnl) = build_fn(i, query)?
            {
                let start = Instant::now();
                let results =
                    llm.execute_query(&spnl.to_string(), Some(sampling.clone()), false, false)?;
                let total_ms = start.elapsed().as_secs_f64() * 1000.0;
                let ttft = results[0].ttft_s.map(|s| s * 1000.0).unwrap_or(0.0);
                let itl = results[0].avg_itl_s.map(|s| s * 1000.0).unwrap_or(0.0);
                (results[0].outputs[0].text.clone(), ttft, itl, total_ms)
            } else {
                // Plain chat path — single user message, no system prompt.
                let prompt = make_prompt(query);
                let msgs = vec![ChatMessage::user(prompt)];
                let start = Instant::now();
                let result = llm.chat(&msgs, Some(sampling.clone()))?;
                let total_ms = start.elapsed().as_secs_f64() * 1000.0;
                let ttft = result.ttft_s.map(|s| s * 1000.0).unwrap_or(0.0);
                let itl = result.avg_itl_s.map(|s| s * 1000.0).unwrap_or(0.0);
                (result.outputs[0].text.clone(), ttft, itl, total_ms)
            };

            let acc = evaluate(&response_text, &query.answer, false);

            if debug {
                let extracted = extract_answer(&response_text);
                let mark = if acc >= 1.0 { "OK" } else { "MISS" };
                pb.suspend(|| {
                    eprintln!(
                        "    [{mark}] q={i} expected={} extracted={:?} response={:?}",
                        query.answer,
                        extracted,
                        response_text.chars().take(120).collect::<String>(),
                    );
                });
            }
            vr.results.push(QueryResult {
                acc,
                ttft_ms,
                itl_ms,
                total_ms,
            });

            let k = vr.results.len() as f64;
            pb.set_message(format!(
                "acc={:.0}%  ttft={:.0}ms",
                vr.avg_acc() * 100.0,
                vr.results.iter().map(|r| r.ttft_ms).sum::<f64>() / k,
            ));
            pb.inc(1);
        }
        pb.finish();
        Ok(vr)
    };

    let mk_style = |name: &str, color: &str| -> ProgressStyle {
        ProgressStyle::default_bar()
            .template(&format!(
                "  {name:>16} {{bar:40.{color}/{color}}} {{pos:>4}}/{{len}} {{msg}}"
            ))
            .unwrap()
    };

    // === 1. Plain ===
    let plain = run_variant(
        &mut llm,
        "plain",
        mk_style("plain", "yellow"),
        true,
        &|_i, _q| Ok(None),
    )?;

    let preamble = "Please read the following text and answer the question below.";

    // === 2. Chunked (cross, no plus) ===
    llm.reset_prefix_cache()?;
    let chunked = run_variant(
        &mut llm,
        "chunked",
        mk_style("chunked", "magenta"),
        true,
        &|i, q| {
            let question = make_question_only(q);
            Ok(Some(build_chunked_query(
                &model_name,
                preamble,
                &all_chunks[i],
                &question,
                max_tokens,
            )))
        },
    )?;

    // === 3. Spans (monolithic context in plus) ===
    llm.reset_prefix_cache()?;
    let spans = run_variant(
        &mut llm,
        "spans",
        mk_style("spans", "cyan"),
        false,
        &|_i, q| {
            let question = make_question_only(q);
            Ok(Some(build_spans_query(
                &model_name,
                preamble,
                &q.context,
                &question,
                max_tokens,
            )))
        },
    )?;

    // === 4. Spans-chunked (chunks in plus) ===
    llm.reset_prefix_cache()?;
    let spans_chunked = run_variant(
        &mut llm,
        "spans-chunked",
        mk_style("spans-chunked", "blue"),
        false,
        &|i, q| {
            let question = make_question_only(q);
            Ok(Some(build_spans_chunked_query(
                &model_name,
                preamble,
                &all_chunks[i],
                &question,
                max_tokens,
            )))
        },
    )?;

    // === 5. Spans-chunked-permuted (chunks in plus, random order) ===
    llm.reset_prefix_cache()?;
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let permuted_chunks: Vec<Vec<String>> = all_chunks
        .iter()
        .map(|chunks| {
            let mut perm = chunks.clone();
            perm.shuffle(&mut rng);
            perm
        })
        .collect();

    let spans_permuted = run_variant(
        &mut llm,
        "spans-chunk-perm",
        mk_style("spans-chunk-perm", "green"),
        false,
        &|i, q| {
            let question = make_question_only(q);
            Ok(Some(build_spans_chunked_query(
                &model_name,
                preamble,
                &permuted_chunks[i],
                &question,
                max_tokens,
            )))
        },
    )?;

    // === Results ===
    eprintln!("\n--- Results (n={n_queries}, model={model_name}) ---");
    print_results(
        "overall",
        &plain,
        &chunked,
        &spans,
        &spans_chunked,
        &spans_permuted,
    );

    // Per-difficulty breakdown.
    eprintln!("\n--- By difficulty ---");
    for diff in &["easy", "hard"] {
        let indices: Vec<usize> = dataset
            .queries
            .iter()
            .enumerate()
            .filter(|(_, q)| q.difficulty.as_str() == *diff)
            .map(|(i, _)| i)
            .collect();
        if indices.is_empty() {
            continue;
        }
        let filter = |vr: &VariantResults| -> VariantResults {
            VariantResults {
                results: indices.iter().map(|&i| vr.results[i].clone()).collect(),
            }
        };
        print_results(
            &format!("{diff} (n={})", indices.len()),
            &filter(&plain),
            &filter(&chunked),
            &filter(&spans),
            &filter(&spans_chunked),
            &filter(&spans_permuted),
        );
    }

    // Per-length breakdown.
    eprintln!("\n--- By context length ---");
    for len_cat in &["short", "medium", "long"] {
        let indices: Vec<usize> = dataset
            .queries
            .iter()
            .enumerate()
            .filter(|(_, q)| q.length.as_str() == *len_cat)
            .map(|(i, _)| i)
            .collect();
        if indices.is_empty() {
            continue;
        }
        let filter = |vr: &VariantResults| -> VariantResults {
            VariantResults {
                results: indices.iter().map(|&i| vr.results[i].clone()).collect(),
            }
        };
        print_results(
            &format!("{len_cat} (n={})", indices.len()),
            &filter(&plain),
            &filter(&chunked),
            &filter(&spans),
            &filter(&spans_chunked),
            &filter(&spans_permuted),
        );
    }

    eprintln!("\n=== LongBench v2 Benchmark Complete ===\n");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_single_letter() {
        assert_eq!(extract_answer("A"), Some("A".to_string()));
        assert_eq!(extract_answer("D"), Some("D".to_string()));
    }

    #[test]
    fn extract_parenthesized() {
        assert_eq!(extract_answer("(B)"), Some("B".to_string()));
        assert_eq!(extract_answer("(C) is correct"), Some("C".to_string()));
    }

    #[test]
    fn extract_official_format() {
        assert_eq!(
            extract_answer("The correct answer is (C)"),
            Some("C".to_string())
        );
        assert_eq!(
            extract_answer("The correct answer is B"),
            Some("B".to_string())
        );
    }

    #[test]
    fn extract_with_reasoning() {
        assert_eq!(
            extract_answer("Based on the context, the correct answer is (D)"),
            Some("D".to_string())
        );
    }

    #[test]
    fn evaluate_correct() {
        assert!((evaluate("A", "A", false) - 1.0).abs() < f64::EPSILON);
        assert!((evaluate("The correct answer is (B)", "B", false) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn evaluate_incorrect() {
        assert!(evaluate("A", "B", false).abs() < f64::EPSILON);
    }

    #[test]
    fn extract_multibyte_no_panic() {
        // Should not panic on multi-byte UTF-8 characters.
        assert_eq!(extract_answer("× × GUEST\n0"), None);
    }
}
