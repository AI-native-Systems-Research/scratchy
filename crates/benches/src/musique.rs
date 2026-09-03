// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! `scr bench musique` — MuSiQue multi-hop RAG accuracy benchmark.
//!
//! Downloads the MuSiQue validation set from HuggingFace and evaluates
//! multi-hop question answering (2-4 hops) over a corpus of Wikipedia
//! paragraphs.
//!
//! MuSiQue is harder than 2WikiMultihopQA: questions require 2-4 reasoning
//! hops, each query has 20 paragraphs (mix of supporting and distractors),
//! and the dataset includes explicit decomposition labels.
//!
//! Two variants:
//! 1. **Plain** — `llm.chat()`, documents inline, full prefill every query.
//! 2. **Spans** — `llm.execute_query()` with `cross` + `plus`, documents as
//!    relocatable blocks. Shared documents hit the prefix cache automatically.

use std::collections::{HashMap, HashSet};
use std::io::BufRead;
use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use indicatif::{ProgressBar, ProgressStyle};
use scratchy_core_config::{CudaGraphConfig, CudaGraphMode};
use scratchy_serving_api::llm::{ChatMessage, LLM, LLMBuilder, SamplingParams};
use scratchy_serving_api::tokenizer::Tokenizer;

use crate::args::BenchMusiqueArgs;

// ---------------------------------------------------------------------------
// Dataset types and loading
// ---------------------------------------------------------------------------

/// A single paragraph in the dataset.
struct Paragraph {
    title: String,
    text: String,
    is_supporting: bool,
}

/// A single query in the benchmark.
struct Query {
    question: String,
    answer: String,
    /// Alternative acceptable answers.
    answer_aliases: Vec<String>,
    /// Paragraphs for this query (20 per row: supporting + distractors).
    paragraphs: Vec<Paragraph>,
    /// Number of reasoning hops (2, 3, or 4).
    num_hops: usize,
}

/// The full benchmark dataset: queries + corpus stats.
struct Dataset {
    queries: Vec<Query>,
    /// Number of unique paragraph titles across all queries.
    corpus_size: usize,
    /// Unique paragraph texts for corpus token counting.
    corpus_texts: Vec<String>,
}

/// Download and parse the MuSiQue validation set (JSONL from HuggingFace).
fn fetch_dataset(num_queries: usize) -> Result<Dataset> {
    let cache_dir = dirs::cache_dir()
        .ok_or_else(|| anyhow::anyhow!("Could not determine cache directory"))?
        .join("scr-bench")
        .join("musique");
    std::fs::create_dir_all(&cache_dir)?;
    let cache_file = cache_dir.join("validation.jsonl");

    // Download if not cached.
    if !cache_file.exists() {
        eprintln!("Downloading MuSiQue validation set...");
        let url = "https://huggingface.co/datasets/dgslibisey/MuSiQue/resolve/main/musique_ans_v1.0_dev.jsonl";
        let agent = crate::http::agent(false);
        let bytes = crate::http::get_bytes(&agent, url)?;
        std::fs::write(&cache_file, &bytes)?;
        eprintln!("Cached {} bytes.", bytes.len());
    }

    // Parse JSONL.
    let file = std::fs::File::open(&cache_file)?;
    let reader = std::io::BufReader::new(file);
    let mut queries: Vec<Query> = Vec::new();
    let mut unique_titles: HashSet<String> = HashSet::new();
    let mut corpus_texts: Vec<String> = Vec::new();
    let mut seen_texts: HashSet<String> = HashSet::new();

    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let record: serde_json::Value = serde_json::from_str(&line)?;

        let question = record["question"].as_str().unwrap_or("").to_string();
        let answer = record["answer"].as_str().unwrap_or("").to_string();

        // Answer aliases for fuzzy matching.
        let answer_aliases: Vec<String> = record["answer_aliases"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();

        // Count hops from question_decomposition.
        let num_hops = record["question_decomposition"]
            .as_array()
            .map(|arr| arr.len())
            .unwrap_or(0);

        // Parse paragraphs.
        let paragraphs_raw = record["paragraphs"].as_array().cloned().unwrap_or_default();

        if question.is_empty() || answer.is_empty() || paragraphs_raw.is_empty() {
            continue;
        }

        let paragraphs: Vec<Paragraph> = paragraphs_raw
            .into_iter()
            .map(|p| {
                let title = p["title"].as_str().unwrap_or("").to_string();
                let text = p["paragraph_text"].as_str().unwrap_or("").to_string();
                let is_supporting = p["is_supporting"].as_bool().unwrap_or(false);

                unique_titles.insert(title.clone());
                let formatted = format!("{}: {}", title, text);
                if seen_texts.insert(formatted.clone()) {
                    corpus_texts.push(formatted);
                }

                Paragraph {
                    title,
                    text,
                    is_supporting,
                }
            })
            .collect();

        queries.push(Query {
            question,
            answer,
            answer_aliases,
            paragraphs,
            num_hops,
        });

        if queries.len() >= num_queries {
            break;
        }
    }

    Ok(Dataset {
        queries,
        corpus_size: unique_titles.len(),
        corpus_texts,
    })
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

/// Evaluate response against expected answer and aliases.
///
/// Returns 1.0 if any of the following match:
/// - The expected answer (or any alias) is a substring of the response
/// - Token F1 >= 0.5 against any acceptable answer
fn evaluate(response: &str, expected: &str, aliases: &[String], debug: bool) -> f64 {
    if debug {
        eprintln!("  expected: {expected}");
        if !aliases.is_empty() {
            eprintln!("  aliases:  {}", aliases.join(", "));
        }
        eprintln!("  response: {response}");
    }

    let resp_lower = response.to_lowercase();

    // Check all acceptable answers (primary + aliases).
    let all_answers: Vec<&str> = std::iter::once(expected)
        .chain(aliases.iter().map(|a| a.as_str()))
        .collect();

    for ans in &all_answers {
        let ans_lower = ans.to_lowercase();

        // Substring match.
        if resp_lower.contains(&ans_lower) {
            return 1.0;
        }

        // Token F1 match.
        let f1 = compute_token_f1(&ans_lower, &resp_lower);
        if f1 >= 0.5 {
            return 1.0;
        }
    }

    0.0
}

fn normalize_tokens(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect()
}

fn compute_token_f1(expected: &str, actual: &str) -> f64 {
    let et = normalize_tokens(expected);
    let at = normalize_tokens(actual);
    if et.is_empty() && at.is_empty() {
        return 1.0;
    }
    if et.is_empty() || at.is_empty() {
        return 0.0;
    }
    let common: usize = et.iter().filter(|t| at.contains(t)).count();
    if common == 0 {
        return 0.0;
    }
    let p = common as f64 / at.len() as f64;
    let r = common as f64 / et.len() as f64;
    2.0 * p * r / (p + r)
}

/// Compute raw token F1 (0.0-1.0) — best F1 across all acceptable answers.
fn token_f1(expected: &str, aliases: &[String], actual: &str) -> f64 {
    let all_answers: Vec<&str> = std::iter::once(expected)
        .chain(aliases.iter().map(|a| a.as_str()))
        .collect();

    all_answers
        .iter()
        .map(|ans| compute_token_f1(&ans.to_lowercase(), &actual.to_lowercase()))
        .fold(0.0_f64, f64::max)
}

// ---------------------------------------------------------------------------
// LLM construction
// ---------------------------------------------------------------------------

fn build_llm(args: &BenchMusiqueArgs) -> Result<LLM> {
    let model = args.resolved_model().map_err(|e| anyhow::anyhow!(e))?;
    let mut builder = LLMBuilder::new(&model)
        .device(&args.device)
        .dtype(&args.dtype)
        .gpu_memory_utilization(args.gpu_memory_utilization)
        .max_num_seqs(args.max_num_seqs)
        .block_size(args.block_size)
        .tensor_parallel_size(args.tensor_parallel_size)
        .enforce_eager(args.enforce_eager)
        .enable_prefix_caching(true);

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
    f1: f64,
    ttft_ms: f64,
    itl_ms: f64,
    total_ms: f64,
}

// ---------------------------------------------------------------------------
// Stats printing
// ---------------------------------------------------------------------------

fn print_comparison(label: &str, plain: &[QueryResult], spans: &[QueryResult]) {
    let n = plain.len();
    let pa = plain.iter().map(|r| r.acc).sum::<f64>() / n as f64;
    let sa = spans.iter().map(|r| r.acc).sum::<f64>() / n as f64;
    let pf = plain.iter().map(|r| r.f1).sum::<f64>() / n as f64;
    let sf = spans.iter().map(|r| r.f1).sum::<f64>() / n as f64;
    let pt = plain.iter().map(|r| r.ttft_ms).sum::<f64>() / n as f64;
    let st = spans.iter().map(|r| r.ttft_ms).sum::<f64>() / n as f64;
    let pi = plain.iter().map(|r| r.itl_ms).sum::<f64>() / n as f64;
    let si = spans.iter().map(|r| r.itl_ms).sum::<f64>() / n as f64;
    let pm = plain.iter().map(|r| r.total_ms).sum::<f64>() / n as f64;
    let sm = spans.iter().map(|r| r.total_ms).sum::<f64>() / n as f64;
    let pp = plain.iter().filter(|r| r.acc >= 1.0).count();
    let sp = spans.iter().filter(|r| r.acc >= 1.0).count();
    let w = format!("{n}").len();
    eprintln!("  {label}");
    eprintln!(
        "    plain: acc={:>5.1}%  F1={:.3}  perfect={:>w$}/{}  ttft={:>6.0}ms  itl={:>4.1}ms  total={:>6.0}ms",
        pa * 100.0,
        pf,
        pp,
        n,
        pt,
        pi,
        pm,
        w = w,
    );
    eprintln!(
        "    spans: acc={:>5.1}%  F1={:.3}  perfect={:>w$}/{}  ttft={:>6.0}ms  itl={:>4.1}ms  total={:>6.0}ms  ({:.2}x, {:+.1}pp)",
        sa * 100.0,
        sf,
        sp,
        n,
        st,
        si,
        sm,
        pt / st.max(0.001),
        (sa - pa) * 100.0,
        w = w,
    );
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

pub(crate) fn run_bench_musique(args: BenchMusiqueArgs) -> Result<()> {
    scratchy_core_common::telemetry::init_tracing(&args.log_level);

    let dataset = fetch_dataset(args.num_queries.unwrap_or(usize::MAX))?;
    let n_queries = dataset.queries.len();
    let n_corpus = dataset.corpus_size;

    if n_queries == 0 {
        eprintln!("No queries to process.");
        return Ok(());
    }

    let mut llm = build_llm(&args)?;
    let model_name = llm.model_name().to_string();

    eprintln!("\n=== MuSiQue Benchmark ===");
    eprintln!("Model:        {}", model_name);
    let tokenizer: Arc<Tokenizer> = llm
        .tokenizer()
        .ok_or_else(|| anyhow::anyhow!("MuSiQue bench requires a tokenizer"))?
        .clone();

    // --- Corpus stats ---
    let corpus_tokens: usize = dataset
        .corpus_texts
        .iter()
        .map(|t| token_len(t, &tokenizer))
        .sum();

    // Paragraphs per query (should be 20 for MuSiQue).
    let paras_per_query: Vec<usize> = dataset.queries.iter().map(|q| q.paragraphs.len()).collect();
    let avg_paras_per_query = paras_per_query.iter().sum::<usize>() as f64 / n_queries as f64;

    // Supporting paragraphs per query.
    let avg_supporting: f64 = dataset
        .queries
        .iter()
        .map(|q| q.paragraphs.iter().filter(|p| p.is_supporting).count() as f64)
        .sum::<f64>()
        / n_queries as f64;

    // Document reuse: how many queries reference each paragraph title.
    let mut doc_freq: HashMap<&str, usize> = HashMap::new();
    for q in &dataset.queries {
        for p in &q.paragraphs {
            *doc_freq.entry(p.title.as_str()).or_insert(0) += 1;
        }
    }
    let shared_docs = doc_freq.values().filter(|&&f| f > 1).count();

    // Hop distribution.
    let mut hop_counts: HashMap<usize, usize> = HashMap::new();
    for q in &dataset.queries {
        *hop_counts.entry(q.num_hops).or_insert(0) += 1;
    }
    let mut hops_sorted: Vec<(usize, usize)> = hop_counts.into_iter().collect();
    hops_sorted.sort();

    eprintln!(
        "Corpus:       {} paragraphs ({} tokens)",
        n_corpus, corpus_tokens
    );
    eprintln!("Queries:      {}", n_queries);
    eprintln!(
        "Paras/query:  {:.0} ({:.1} supporting)",
        avg_paras_per_query, avg_supporting
    );
    eprintln!(
        "Hops:         {}",
        hops_sorted
            .iter()
            .map(|(h, c)| format!("{h}-hop={c}"))
            .collect::<Vec<_>>()
            .join(", ")
    );
    eprintln!(
        "Shared docs:  {}/{} ({:.0}% of corpus appear in >1 query)",
        shared_docs,
        n_corpus,
        shared_docs as f64 / n_corpus as f64 * 100.0,
    );

    let sampling = SamplingParams {
        max_tokens: Some(args.max_tokens as u32),
        temperature: 0.0,
        ..SamplingParams::default()
    };

    let system_prompt = "You are a helpful assistant. Answer the question based only on the provided documents. Be concise — answer with just the entity, date, or fact requested.";

    let plain_style = ProgressStyle::default_bar()
        .template("  plain {bar:40.yellow/yellow} {pos:>4}/{len} {msg}")
        .unwrap();
    let spans_style = ProgressStyle::default_bar()
        .template("  spans {bar:40.cyan/blue} {pos:>4}/{len} {msg}")
        .unwrap();

    let mut plain_results: Vec<QueryResult> = Vec::with_capacity(n_queries);
    let mut span_results: Vec<QueryResult> = Vec::with_capacity(n_queries);

    // === 1. Plain: llm.chat(), all paragraphs inline, reset cache each time ===
    let pb = ProgressBar::new(n_queries as u64)
        .with_style(plain_style)
        .with_message("musique");

    for (i, query) in dataset.queries.iter().enumerate() {
        let debug = args.debug && i == 0;
        llm.reset_prefix_cache()?;

        let mut msgs = vec![ChatMessage::system(system_prompt)];
        for (j, para) in query.paragraphs.iter().enumerate() {
            msgs.push(ChatMessage::user(format!(
                "Document {j} ({title}): {text}",
                title = para.title,
                text = para.text,
            )));
        }
        msgs.push(ChatMessage::user(&query.question));

        let start = Instant::now();
        let result = llm.chat(&msgs, Some(sampling.clone()))?;
        let total_ms = start.elapsed().as_secs_f64() * 1000.0;
        let ttft_ms = result.ttft_s.map(|s| s * 1000.0).unwrap_or(0.0);
        let itl_ms = result.avg_itl_s.map(|s| s * 1000.0).unwrap_or(0.0);

        let response = &result.outputs[0].text;
        let acc = evaluate(response, &query.answer, &query.answer_aliases, debug);
        let f1 = token_f1(&query.answer, &query.answer_aliases, response);
        plain_results.push(QueryResult {
            acc,
            f1,
            ttft_ms,
            itl_ms,
            total_ms,
        });

        let k = plain_results.len() as f64;
        let avg_acc = plain_results.iter().map(|r| r.acc).sum::<f64>() / k;
        let avg_ttft = plain_results.iter().map(|r| r.ttft_ms).sum::<f64>() / k;
        pb.set_message(format!(
            "acc={:.0}%  ttft={:.0}ms",
            avg_acc * 100.0,
            avg_ttft
        ));
        pb.inc(1);
    }
    pb.finish();

    // === 2. Spans: execute_query with plus, corpus cached across queries ===
    llm.reset_prefix_cache()?;

    let pb = ProgressBar::new(n_queries as u64)
        .with_style(spans_style)
        .with_message("musique");

    // Track cumulative cache hit rate.
    let mut docs_seen: HashSet<String> = HashSet::new();
    let mut total_doc_refs: usize = 0;
    let mut cache_hits: usize = 0;

    for (i, query) in dataset.queries.iter().enumerate() {
        let debug = args.debug && i == 0;

        let doc_json: Vec<serde_json::Value> = query
            .paragraphs
            .iter()
            .enumerate()
            .map(|(j, para)| {
                serde_json::json!({ "user": format!(
                    "Document {j} ({title}): {text}",
                    title = para.title,
                    text = para.text,
                )})
            })
            .collect();

        // Track cache hit stats.
        for para in &query.paragraphs {
            total_doc_refs += 1;
            if docs_seen.contains(&para.title) {
                cache_hits += 1;
            } else {
                docs_seen.insert(para.title.clone());
            }
        }

        let span_query = serde_json::json!({
            "g": {
                "model": model_name,
                "max_tokens": args.max_tokens,
                "temperature": 0.0,
                "input": {
                    "cross": [
                        { "system": system_prompt },
                        { "plus": doc_json },
                        { "user": &query.question }
                    ]
                }
            }
        });

        let start = Instant::now();
        let results = llm.execute_query(
            &span_query.to_string(),
            Some(sampling.clone()),
            false,
            false,
        )?;
        let total_ms = start.elapsed().as_secs_f64() * 1000.0;
        let ttft_ms = results[0].ttft_s.map(|s| s * 1000.0).unwrap_or(0.0);
        let itl_ms = results[0].avg_itl_s.map(|s| s * 1000.0).unwrap_or(0.0);

        let response = &results[0].outputs[0].text;
        let acc = evaluate(response, &query.answer, &query.answer_aliases, debug);
        let f1 = token_f1(&query.answer, &query.answer_aliases, response);
        span_results.push(QueryResult {
            acc,
            f1,
            ttft_ms,
            itl_ms,
            total_ms,
        });

        let k = span_results.len() as f64;
        let avg_acc = span_results.iter().map(|r| r.acc).sum::<f64>() / k;
        let avg_ttft = span_results.iter().map(|r| r.ttft_ms).sum::<f64>() / k;
        let hit_rate = cache_hits as f64 / total_doc_refs as f64 * 100.0;
        pb.set_message(format!(
            "acc={:.0}%  ttft={:.0}ms  KVHitRate={:.0}%",
            avg_acc * 100.0,
            avg_ttft,
            hit_rate,
        ));
        pb.inc(1);
    }
    pb.finish();

    eprintln!(
        "\n  KV cache hit rate: {}/{} doc refs ({:.1}%), {} unique docs seen",
        cache_hits,
        total_doc_refs,
        cache_hits as f64 / total_doc_refs as f64 * 100.0,
        docs_seen.len(),
    );

    eprintln!("\n--- Results (n={n_queries}, model={model_name}) ---");
    print_comparison("overall", &plain_results, &span_results);

    // Per-hop breakdown.
    if hops_sorted.len() > 1 {
        eprintln!("\n--- By number of hops ---");
        for (nhops, _) in &hops_sorted {
            let indices: Vec<usize> = dataset
                .queries
                .iter()
                .enumerate()
                .filter(|(_, q)| q.num_hops == *nhops)
                .map(|(i, _)| i)
                .collect();
            if indices.is_empty() {
                continue;
            }
            let pr: Vec<QueryResult> = indices.iter().map(|&i| plain_results[i].clone()).collect();
            let sr: Vec<QueryResult> = indices.iter().map(|&i| span_results[i].clone()).collect();
            print_comparison(&format!("{nhops}-hop (n={})", indices.len()), &pr, &sr);
        }
    }

    eprintln!("\n=== MuSiQue Benchmark Complete ===\n");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evaluate_exact_substring() {
        assert!(
            (evaluate(
                "The answer is Miquette Giraudy.",
                "Miquette Giraudy",
                &[],
                false
            ) - 1.0)
                .abs()
                < f64::EPSILON
        );
    }

    #[test]
    fn evaluate_alias_match() {
        assert!(
            (evaluate(
                "The birthplace is Denver, Colorado.",
                "Denver",
                &["Denver, Colorado".to_string()],
                false
            ) - 1.0)
                .abs()
                < f64::EPSILON
        );
    }

    #[test]
    fn evaluate_case_insensitive() {
        assert!((evaluate("south park", "South Park", &[], false) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn evaluate_no_match() {
        assert!(
            evaluate(
                "Something completely different",
                "Denver",
                &["Denver, Colorado".to_string()],
                false
            )
            .abs()
                < f64::EPSILON
        );
    }

    #[test]
    fn evaluate_f1_threshold() {
        assert!(
            (evaluate(
                "Denver Colorado is the city",
                "Denver, Colorado",
                &[],
                false
            ) - 1.0)
                .abs()
                < f64::EPSILON
        );
    }

    #[test]
    fn token_f1_with_aliases() {
        let f1 = token_f1(
            "Denver",
            &["Denver, Colorado".to_string()],
            "Denver, Colorado is great",
        );
        assert!(f1 > 0.5);
    }

    #[test]
    fn normalize_tokens_basic() {
        assert_eq!(normalize_tokens("hello, world!"), vec!["hello", "world"]);
    }
}
