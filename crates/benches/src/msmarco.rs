// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! `scr bench msmarco` — MS MARCO passage QA accuracy benchmark.
//!
//! Downloads the MS MARCO v2.1 validation set from HuggingFace and evaluates
//! single-hop question answering over Bing search result passages.
//!
//! MS MARCO is the industry-standard passage QA benchmark. Each query has ~10
//! passages retrieved from Bing, with `is_selected` labels marking the answer
//! source. 55K answerable queries in the validation set.
//!
//! Two variants:
//! 1. **Plain** — `llm.chat()`, passages inline, full prefill every query.
//! 2. **Spans** — `llm.execute_query()` with `cross` + `plus`, passages as
//!    relocatable blocks.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use indicatif::{ProgressBar, ProgressStyle};
use scratchy_core_config::{CudaGraphConfig, CudaGraphMode};
use scratchy_serving_api::llm::{ChatMessage, LLM, LLMBuilder, SamplingParams};
use scratchy_serving_api::tokenizer::Tokenizer;

use crate::args::BenchMsmarcoArgs;

// ---------------------------------------------------------------------------
// Dataset types and loading
// ---------------------------------------------------------------------------

/// A single query in the benchmark.
struct Query {
    question: String,
    /// All acceptable answers (from `answers` + `wellFormedAnswers`).
    answers: Vec<String>,
    /// Passages for this query, as (url, text) pairs.
    passages: Vec<(String, String)>,
    query_type: String,
}

/// The full benchmark dataset: queries + corpus stats.
struct Dataset {
    queries: Vec<Query>,
    /// Number of unique passages across all queries (by text content).
    corpus_size: usize,
    /// Unique passage texts for corpus token counting.
    corpus_texts: Vec<String>,
}

/// Download and parse the MS MARCO v2.1 validation set (parquet from HF).
fn fetch_dataset(num_queries: usize) -> Result<Dataset> {
    let cache_dir = dirs::cache_dir()
        .ok_or_else(|| anyhow::anyhow!("Could not determine cache directory"))?
        .join("scr-bench")
        .join("msmarco");
    std::fs::create_dir_all(&cache_dir)?;
    let cache_file = cache_dir.join("validation.json");

    let raw: Vec<serde_json::Value> = if cache_file.exists() {
        let data = std::fs::read_to_string(&cache_file)?;
        serde_json::from_str(&data)?
    } else {
        eprintln!("Downloading MS MARCO v2.1 validation set...");
        let url = "https://huggingface.co/datasets/microsoft/ms_marco/resolve/main/v2.1/validation-00000-of-00001.parquet";
        let agent = crate::http::agent(false);

        let parquet_path = cache_dir.join("validation.parquet");
        let bytes = crate::http::get_bytes(&agent, url)?;
        std::fs::write(&parquet_path, &bytes)?;

        let records = parquet_to_json_records(&parquet_path)?;

        eprintln!("Caching {} records as JSON...", records.len());
        let json_str = serde_json::to_string(&records)?;
        std::fs::write(&cache_file, json_str.as_bytes())?;
        records
    };

    // Build queries and corpus stats — only include answerable queries.
    let mut queries: Vec<Query> = Vec::new();
    let mut corpus_texts: Vec<String> = Vec::new();
    let mut seen_texts: HashSet<String> = HashSet::new();

    for record in &raw {
        let question = record["query"].as_str().unwrap_or("").to_string();
        let query_type = record["query_type"].as_str().unwrap_or("").to_string();

        // Collect answers: primary answers + well-formed answers.
        let mut answers: Vec<String> = Vec::new();
        if let Some(arr) = record["answers"].as_array() {
            for a in arr {
                if let Some(s) = a
                    .as_str()
                    .filter(|s| !s.is_empty() && *s != "No Answer Present.")
                {
                    answers.push(s.to_string());
                }
            }
        }
        if let Some(arr) = record["wellFormedAnswers"].as_array() {
            for a in arr {
                if let Some(s) = a.as_str().filter(|s| !s.is_empty() && *s != "[]") {
                    answers.push(s.to_string());
                }
            }
        }

        if question.is_empty() || answers.is_empty() {
            continue;
        }

        // Parse passages.
        let passages_obj = &record["passages"];
        let texts = passages_obj["passage_text"]
            .as_array()
            .cloned()
            .unwrap_or_default();
        let urls = passages_obj["url"].as_array().cloned().unwrap_or_default();

        if texts.is_empty() {
            continue;
        }

        let passages: Vec<(String, String)> = urls
            .iter()
            .zip(texts.iter())
            .map(|(url_val, text_val)| {
                let url = url_val.as_str().unwrap_or("").to_string();
                let text = text_val.as_str().unwrap_or("").to_string();

                if seen_texts.insert(text.clone()) {
                    corpus_texts.push(text.clone());
                }
                (url, text)
            })
            .collect();

        queries.push(Query {
            question,
            answers,
            passages,
            query_type,
        });

        if queries.len() >= num_queries {
            break;
        }
    }

    Ok(Dataset {
        queries,
        corpus_size: corpus_texts.len(),
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

/// Evaluate response against any acceptable answer.
///
/// Returns 1.0 if any answer is a substring match or token F1 >= 0.5.
fn evaluate(response: &str, answers: &[String], debug: bool) -> f64 {
    if debug {
        eprintln!("  answers: {:?}", answers);
        eprintln!("  response: {response}");
    }

    let resp_lower = response.to_lowercase();

    for ans in answers {
        let ans_lower = ans.to_lowercase();

        if resp_lower.contains(&ans_lower) {
            return 1.0;
        }

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

/// Compute best token F1 across all acceptable answers.
fn token_f1(answers: &[String], actual: &str) -> f64 {
    answers
        .iter()
        .map(|ans| compute_token_f1(&ans.to_lowercase(), &actual.to_lowercase()))
        .fold(0.0_f64, f64::max)
}

// ---------------------------------------------------------------------------
// LLM construction
// ---------------------------------------------------------------------------

fn build_llm(args: &BenchMsmarcoArgs) -> Result<LLM> {
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

pub(crate) fn run_bench_msmarco(args: BenchMsmarcoArgs) -> Result<()> {
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

    eprintln!("\n=== MS MARCO Benchmark ===");
    eprintln!("Model:        {}", model_name);
    let tokenizer: Arc<Tokenizer> = llm
        .tokenizer()
        .ok_or_else(|| anyhow::anyhow!("MS MARCO bench requires a tokenizer"))?
        .clone();

    // --- Corpus stats ---
    let corpus_tokens: usize = dataset
        .corpus_texts
        .iter()
        .map(|t| token_len(t, &tokenizer))
        .sum();

    let passages_per_query: Vec<usize> = dataset.queries.iter().map(|q| q.passages.len()).collect();
    let avg_passages_per_query = passages_per_query.iter().sum::<usize>() as f64 / n_queries as f64;

    // Passage text reuse across queries.
    let mut text_freq: HashMap<&str, usize> = HashMap::new();
    for q in &dataset.queries {
        for (_, text) in &q.passages {
            *text_freq.entry(text.as_str()).or_insert(0) += 1;
        }
    }
    let shared_passages = text_freq.values().filter(|&&f| f > 1).count();

    // Type distribution.
    let mut type_counts: HashMap<&str, usize> = HashMap::new();
    for q in &dataset.queries {
        *type_counts.entry(q.query_type.as_str()).or_insert(0) += 1;
    }
    let mut type_sorted: Vec<(&&str, &usize)> = type_counts.iter().collect();
    type_sorted.sort_by(|a, b| b.1.cmp(a.1));

    eprintln!(
        "Corpus:       {} unique passages ({} tokens)",
        n_corpus, corpus_tokens
    );
    eprintln!("Queries:      {} (answerable only)", n_queries);
    eprintln!("Passages/q:   {:.0}", avg_passages_per_query);
    eprintln!(
        "Types:        {}",
        type_sorted
            .iter()
            .map(|(t, c)| format!("{}={}", t, c))
            .collect::<Vec<_>>()
            .join(", ")
    );
    eprintln!(
        "Shared text:  {}/{} ({:.1}% of passages appear in >1 query)",
        shared_passages,
        n_corpus,
        shared_passages as f64 / n_corpus as f64 * 100.0,
    );

    let sampling = SamplingParams {
        max_tokens: Some(args.max_tokens as u32),
        temperature: 0.0,
        ..SamplingParams::default()
    };

    let system_prompt = "You are a helpful assistant. Answer the question based only on the provided passages. Be concise — answer with just the entity, date, number, or fact requested.";

    let plain_style = ProgressStyle::default_bar()
        .template("  plain {bar:40.yellow/yellow} {pos:>4}/{len} {msg}")
        .unwrap();
    let spans_style = ProgressStyle::default_bar()
        .template("  spans {bar:40.cyan/blue} {pos:>4}/{len} {msg}")
        .unwrap();

    let mut plain_results: Vec<QueryResult> = Vec::with_capacity(n_queries);
    let mut span_results: Vec<QueryResult> = Vec::with_capacity(n_queries);

    // === 1. Plain: llm.chat(), all passages inline, reset cache each time ===
    let pb = ProgressBar::new(n_queries as u64)
        .with_style(plain_style)
        .with_message("msmarco");

    for (i, query) in dataset.queries.iter().enumerate() {
        let debug = args.debug && i == 0;
        llm.reset_prefix_cache()?;

        let mut msgs = vec![ChatMessage::system(system_prompt)];
        for (j, (url, text)) in query.passages.iter().enumerate() {
            msgs.push(ChatMessage::user(format!("Passage {j} ({url}): {text}")));
        }
        msgs.push(ChatMessage::user(&query.question));

        let start = Instant::now();
        let result = llm.chat(&msgs, Some(sampling.clone()))?;
        let total_ms = start.elapsed().as_secs_f64() * 1000.0;
        let ttft_ms = result.ttft_s.map(|s| s * 1000.0).unwrap_or(0.0);
        let itl_ms = result.avg_itl_s.map(|s| s * 1000.0).unwrap_or(0.0);

        let response = &result.outputs[0].text;
        let acc = evaluate(response, &query.answers, debug);
        let f1 = token_f1(&query.answers, response);
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
        .with_message("msmarco");

    let mut texts_seen: HashSet<String> = HashSet::new();
    let mut total_passage_refs: usize = 0;
    let mut cache_hits: usize = 0;

    for (i, query) in dataset.queries.iter().enumerate() {
        let debug = args.debug && i == 0;

        let passage_json: Vec<serde_json::Value> = query
            .passages
            .iter()
            .enumerate()
            .map(|(j, (url, text))| {
                serde_json::json!({ "user": format!("Passage {j} ({url}): {text}") })
            })
            .collect();

        // Track cache hit stats by passage text content.
        for (_, text) in &query.passages {
            total_passage_refs += 1;
            if texts_seen.contains(text) {
                cache_hits += 1;
            } else {
                texts_seen.insert(text.clone());
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
                        { "plus": passage_json },
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
        let acc = evaluate(response, &query.answers, debug);
        let f1 = token_f1(&query.answers, response);
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
        let hit_rate = if total_passage_refs > 0 {
            cache_hits as f64 / total_passage_refs as f64 * 100.0
        } else {
            0.0
        };
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
        "\n  KV cache hit rate: {}/{} passage refs ({:.1}%), {} unique passages seen",
        cache_hits,
        total_passage_refs,
        if total_passage_refs > 0 {
            cache_hits as f64 / total_passage_refs as f64 * 100.0
        } else {
            0.0
        },
        texts_seen.len(),
    );

    eprintln!("\n--- Results (n={n_queries}, model={model_name}) ---");
    print_comparison("overall", &plain_results, &span_results);

    // Per-type breakdown.
    let types: Vec<&str> = dataset
        .queries
        .iter()
        .map(|q| q.query_type.as_str())
        .collect();
    let unique_types: HashSet<&str> = types.iter().copied().collect();
    if unique_types.len() > 1 {
        eprintln!("\n--- By query type ---");
        for qt in &["DESCRIPTION", "NUMERIC", "ENTITY", "PERSON", "LOCATION"] {
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

    eprintln!("\n=== MS MARCO Benchmark Complete ===\n");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evaluate_exact_substring() {
        assert!(
            (evaluate(
                "A corporation is a company authorized to act as a single entity.",
                &["A corporation is a company or group of people authorized to act as a single entity".to_string()],
                false
            ) - 1.0)
                .abs()
                < f64::EPSILON
        );
    }

    #[test]
    fn evaluate_multiple_answers() {
        assert!(
            (evaluate(
                "Denver is the answer",
                &["wrong answer".to_string(), "Denver".to_string()],
                false
            ) - 1.0)
                .abs()
                < f64::EPSILON
        );
    }

    #[test]
    fn evaluate_no_match() {
        assert!(
            evaluate(
                "Something completely different",
                &["corporation".to_string()],
                false
            )
            .abs()
                < f64::EPSILON
        );
    }

    #[test]
    fn token_f1_best_of_multiple() {
        let f1 = token_f1(
            &["wrong".to_string(), "the cat sat".to_string()],
            "the cat sat on the mat",
        );
        assert!(f1 > 0.5);
    }

    #[test]
    fn normalize_tokens_basic() {
        assert_eq!(normalize_tokens("hello, world!"), vec!["hello", "world"]);
    }
}
