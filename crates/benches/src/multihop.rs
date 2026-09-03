// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! `scr bench multihop` — 2WikiMultihopQA RAG accuracy benchmark.
//!
//! Downloads the 2WikiMultihopQA dev set from HuggingFace and evaluates
//! multi-hop question answering over a corpus of Wikipedia documents.
//!
//! The dataset provides a natural RAG workload: documents are shared across
//! queries, so the span cache accumulates hits as more queries are processed.
//!
//! Two variants:
//! 1. **Plain** — `llm.chat()`, documents inline, full prefill every query.
//! 2. **Spans** — `llm.execute_query()` with `cross` + `plus`, documents as
//!    relocatable blocks. Shared documents hit the prefix cache automatically.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use indicatif::{ProgressBar, ProgressStyle};
use scratchy_core_config::{CudaGraphConfig, CudaGraphMode};
use scratchy_serving_api::llm::{ChatMessage, LLM, LLMBuilder, SamplingParams};
use scratchy_serving_api::tokenizer::Tokenizer;

use crate::args::BenchMultihopArgs;

// ---------------------------------------------------------------------------
// Dataset types and loading
// ---------------------------------------------------------------------------

/// A single query in the benchmark.
struct Query {
    question: String,
    answer: String,
    /// Documents for this query, as (title, text) pairs.
    /// Stored per-query because the same title may have different sentence
    /// selections in different rows.
    documents: Vec<(String, String)>,
    question_type: String,
}

/// The full benchmark dataset: queries + corpus stats.
struct Dataset {
    queries: Vec<Query>,
    /// Number of unique documents across all queries (by title).
    corpus_size: usize,
    /// Unique document texts keyed by "title: text" (for corpus stats).
    corpus_texts: Vec<String>,
}

/// Download and parse the 2WikiMultihopQA dev set (parquet from HuggingFace).
fn fetch_dataset(num_queries: usize) -> Result<Dataset> {
    let cache_dir = dirs::cache_dir()
        .ok_or_else(|| anyhow::anyhow!("Could not determine cache directory"))?
        .join("scr-bench")
        .join("multihop");
    std::fs::create_dir_all(&cache_dir)?;
    let cache_file = cache_dir.join("dev.json");

    // If we have a cached JSON version, use it.
    let raw: Vec<serde_json::Value> = if cache_file.exists() {
        let data = std::fs::read_to_string(&cache_file)?;
        serde_json::from_str(&data)?
    } else {
        eprintln!("Downloading 2WikiMultihopQA dev set...");
        let url = "https://huggingface.co/datasets/xanhho/2WikiMultihopQA/resolve/main/dev.parquet";
        let agent = crate::http::agent(false);

        let parquet_path = cache_dir.join("dev.parquet");
        let bytes = crate::http::get_bytes(&agent, url)?;
        std::fs::write(&parquet_path, &bytes)?;

        let records = parquet_to_json_records(&parquet_path)?;

        eprintln!("Caching {} records as JSON...", records.len());
        let json_str = serde_json::to_string(&records)?;
        std::fs::write(&cache_file, json_str.as_bytes())?;
        records
    };

    // Build queries and corpus stats.
    let mut queries: Vec<Query> = Vec::new();
    let mut unique_titles: HashSet<String> = HashSet::new();
    let mut corpus_texts: Vec<String> = Vec::new();
    let mut seen_texts: HashSet<String> = HashSet::new();

    let max_rows = num_queries.min(raw.len());
    for record in raw.iter().take(max_rows) {
        let question = record["question"].as_str().unwrap_or("").to_string();
        let answer = record["answer"].as_str().unwrap_or("").to_string();
        let question_type = record["type"].as_str().unwrap_or("").to_string();

        let context_str = record["context"].as_str().unwrap_or("[]");
        let context: Vec<(String, Vec<String>)> =
            serde_json::from_str(context_str).unwrap_or_default();

        if question.is_empty() || answer.is_empty() || context.is_empty() {
            continue;
        }

        let documents: Vec<(String, String)> = context
            .into_iter()
            .map(|(title, sents)| {
                let text = sents.join(" ");
                unique_titles.insert(title.clone());
                let formatted = format!("{}: {}", title, text);
                if seen_texts.insert(formatted.clone()) {
                    corpus_texts.push(formatted);
                }
                (title, text)
            })
            .collect();

        queries.push(Query {
            question,
            answer,
            documents,
            question_type,
        });
    }

    Ok(Dataset {
        queries,
        corpus_size: unique_titles.len(),
        corpus_texts,
    })
}

/// Read a parquet file and return rows as JSON values.
fn parquet_to_json_records(parquet_path: &std::path::Path) -> Result<Vec<serde_json::Value>> {
    use arrow::json::writer::{JsonArray, Writer};
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

    eprintln!("Reading parquet...");
    let file = std::fs::File::open(parquet_path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let reader = builder.build()?;

    let batches: Vec<_> = reader.collect::<std::result::Result<Vec<_>, _>>()?;
    let batch_refs: Vec<&_> = batches.iter().collect();

    let mut buf = Vec::new();
    let mut writer = Writer::<_, JsonArray>::new(&mut buf);
    writer.write_batches(&batch_refs)?;
    writer.finish()?;
    drop(writer);

    let records: Vec<serde_json::Value> = serde_json::from_slice(&buf)?;
    eprintln!("Read {} records.", records.len());
    Ok(records)
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

/// Evaluate response against expected answer.
///
/// Returns 1.0 if the expected answer is found as a substring or if
/// token F1 >= 0.5 (many answers are short entities and the model may
/// add surrounding context). Returns 0.0 otherwise.
fn evaluate(response: &str, expected: &str, debug: bool) -> f64 {
    if debug {
        eprintln!("  expected: {expected}");
        eprintln!("  response: {response}");
    }

    let resp_lower = response.to_lowercase();
    let exp_lower = expected.to_lowercase();

    if resp_lower.contains(&exp_lower) {
        return 1.0;
    }

    let exp_tokens = normalize_tokens(&exp_lower);
    let resp_tokens = normalize_tokens(&resp_lower);

    if exp_tokens.is_empty() {
        return 0.0;
    }

    let common: usize = exp_tokens
        .iter()
        .filter(|t| resp_tokens.contains(t))
        .count();

    if common == 0 {
        return 0.0;
    }

    let precision = common as f64 / resp_tokens.len().max(1) as f64;
    let recall = common as f64 / exp_tokens.len() as f64;
    let f1 = 2.0 * precision * recall / (precision + recall);

    if f1 >= 0.5 { 1.0 } else { 0.0 }
}

fn normalize_tokens(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect()
}

/// Compute raw token F1 (0.0-1.0) for detailed reporting.
fn token_f1(expected: &str, actual: &str) -> f64 {
    let et = normalize_tokens(&expected.to_lowercase());
    let at = normalize_tokens(&actual.to_lowercase());
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

// ---------------------------------------------------------------------------
// LLM construction
// ---------------------------------------------------------------------------

fn build_llm(args: &BenchMultihopArgs) -> Result<LLM> {
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

pub(crate) fn run_bench_multihop(args: BenchMultihopArgs) -> Result<()> {
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

    eprintln!("\n=== 2WikiMultihopQA Benchmark ===");
    eprintln!("Model:        {}", model_name);
    let tokenizer: Arc<Tokenizer> = llm
        .tokenizer()
        .ok_or_else(|| anyhow::anyhow!("Multihop bench requires a tokenizer"))?
        .clone();

    // --- Corpus stats ---
    let corpus_tokens: usize = dataset
        .corpus_texts
        .iter()
        .map(|t| token_len(t, &tokenizer))
        .sum();

    // Docs per query (should be 10 for 2WikiMultihopQA).
    let docs_per_query: Vec<usize> = dataset.queries.iter().map(|q| q.documents.len()).collect();
    let avg_docs_per_query = docs_per_query.iter().sum::<usize>() as f64 / n_queries as f64;

    // Document reuse: how many queries reference each doc title.
    let mut doc_freq: HashMap<&str, usize> = HashMap::new();
    for q in &dataset.queries {
        for (title, _) in &q.documents {
            *doc_freq.entry(title.as_str()).or_insert(0) += 1;
        }
    }
    let shared_docs = doc_freq.values().filter(|&&f| f > 1).count();

    eprintln!(
        "Corpus:       {} documents ({} tokens)",
        n_corpus, corpus_tokens
    );
    eprintln!("Queries:      {}", n_queries);
    eprintln!("Docs/query:   {:.0}", avg_docs_per_query);
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

    // === 1. Plain: llm.chat(), all docs inline, reset cache each time ===
    let pb = ProgressBar::new(n_queries as u64)
        .with_style(plain_style)
        .with_message("multihop");

    for (i, query) in dataset.queries.iter().enumerate() {
        let debug = args.debug && i == 0;
        llm.reset_prefix_cache()?;

        let mut msgs = vec![ChatMessage::system(system_prompt)];
        for (j, (title, text)) in query.documents.iter().enumerate() {
            msgs.push(ChatMessage::user(format!("Document {j} ({title}): {text}")));
        }
        msgs.push(ChatMessage::user(&query.question));

        let start = Instant::now();
        let result = llm.chat(&msgs, Some(sampling.clone()))?;
        let total_ms = start.elapsed().as_secs_f64() * 1000.0;
        let ttft_ms = result.ttft_s.map(|s| s * 1000.0).unwrap_or(0.0);
        let itl_ms = result.avg_itl_s.map(|s| s * 1000.0).unwrap_or(0.0);

        let response = &result.outputs[0].text;
        let acc = evaluate(response, &query.answer, debug);
        let f1 = token_f1(&query.answer, response);
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
    // Each query's `plus` block contains its specific documents. The prefix
    // cache means documents seen in prior queries are already cached — only
    // new documents require prefill.
    //
    // No cache reset between queries — this is the whole point.
    llm.reset_prefix_cache()?;

    let pb = ProgressBar::new(n_queries as u64)
        .with_style(spans_style)
        .with_message("multihop");

    // Track cumulative cache hit rate.
    let mut docs_seen: HashSet<String> = HashSet::new();
    let mut total_doc_refs: usize = 0;
    let mut cache_hits: usize = 0;

    for (i, query) in dataset.queries.iter().enumerate() {
        let debug = args.debug && i == 0;

        let doc_json: Vec<serde_json::Value> = query
            .documents
            .iter()
            .enumerate()
            .map(|(j, (title, text))| {
                serde_json::json!({ "user": format!("Document {j} ({title}): {text}") })
            })
            .collect();

        // Track cache hit stats.
        for (title, _) in &query.documents {
            total_doc_refs += 1;
            if docs_seen.contains(title) {
                cache_hits += 1;
            } else {
                docs_seen.insert(title.clone());
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
        let acc = evaluate(response, &query.answer, debug);
        let f1 = token_f1(&query.answer, response);
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

    // Per-type breakdown.
    let types: Vec<&str> = dataset
        .queries
        .iter()
        .map(|q| q.question_type.as_str())
        .collect();
    let unique_types: HashSet<&str> = types.iter().copied().collect();
    if unique_types.len() > 1 {
        eprintln!("\n--- By question type ---");
        for qt in &[
            "compositional",
            "comparison",
            "bridge_comparison",
            "inference",
        ] {
            let indices: Vec<usize> = types
                .iter()
                .enumerate()
                .filter(|(_, t)| *t == qt)
                .map(|(i, _)| i)
                .collect();
            if indices.is_empty() {
                continue;
            }
            let pr: Vec<QueryResult> = indices.iter().map(|&i| plain_results[i].clone()).collect();
            let sr: Vec<QueryResult> = indices.iter().map(|&i| span_results[i].clone()).collect();
            print_comparison(&format!("{qt} (n={})", indices.len()), &pr, &sr);
        }
    }

    eprintln!("\n=== 2WikiMultihopQA Benchmark Complete ===\n");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evaluate_exact_substring() {
        assert!(
            (evaluate(
                "The answer is Małgorzata Braunek.",
                "Małgorzata Braunek",
                false
            ) - 1.0)
                .abs()
                < f64::EPSILON
        );
    }

    #[test]
    fn evaluate_case_insensitive() {
        assert!(
            (evaluate("the mask of fu manchu", "The Mask Of Fu Manchu", false) - 1.0).abs()
                < f64::EPSILON
        );
    }

    #[test]
    fn evaluate_no_match() {
        assert!(
            evaluate(
                "Something completely different",
                "Małgorzata Braunek",
                false
            )
            .abs()
                < f64::EPSILON
        );
    }

    #[test]
    fn evaluate_f1_threshold() {
        assert!(
            (evaluate("12 June 1516 was the date", "12 June 1516", false) - 1.0).abs()
                < f64::EPSILON
        );
    }

    #[test]
    fn token_f1_identical() {
        assert!((token_f1("the cat", "the cat") - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn token_f1_no_overlap() {
        assert!(token_f1("alpha beta", "gamma delta").abs() < f64::EPSILON);
    }

    #[test]
    fn normalize_tokens_basic() {
        assert_eq!(normalize_tokens("hello, world!"), vec!["hello", "world"]);
    }
}
