// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! `scr bench ragcsv` — RAG CSV evaluation benchmark.
//!
//! Reads a CSV dataset with questions, document fragments, and expected answers,
//! runs them through the model via both plain (chat) and span (SPNL query with
//! relocatable document blocks) modes, and grades responses using LLM-judge
//! metrics (accuracy, faithfulness, relevancy) plus string metrics (token F1,
//! exact match, BLEU-1).

use std::collections::{HashMap, HashSet};
use std::time::Instant;

use anyhow::Result;
use indicatif::{ProgressBar, ProgressStyle};
use scratchy_core_config::{CudaGraphConfig, CudaGraphMode};
use scratchy_serving_api::llm::{ChatMessage, LLM, LLMBuilder, SamplingParams};

use crate::args::BenchRagcsvArgs;

// ---------------------------------------------------------------------------
// Metric flags
// ---------------------------------------------------------------------------

struct MetricFlags {
    accuracy: bool,
    faithfulness: bool,
    relevancy: bool,
}

impl MetricFlags {
    fn from_arg(arg: &str) -> Self {
        let tokens: HashSet<&str> = arg.split(',').map(|s| s.trim()).collect();
        if tokens.contains("all") {
            return Self {
                accuracy: true,
                faithfulness: true,
                relevancy: true,
            };
        }
        Self {
            accuracy: tokens.contains("accuracy"),
            faithfulness: tokens.contains("faithfulness"),
            relevancy: tokens.contains("relevancy"),
        }
    }
}

// ---------------------------------------------------------------------------
// CSV types
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct EvalRow {
    index: usize,
    expected: String,
    fragments: Vec<Fragment>,
    question: String,
}

#[derive(Debug, serde::Deserialize)]
#[allow(dead_code)]
struct Fragment {
    page_content: String,
    metadata: FragmentMetadata,
}

#[derive(Debug, serde::Deserialize)]
#[allow(dead_code)]
struct FragmentMetadata {
    #[serde(default)]
    title: String,
}

// ---------------------------------------------------------------------------
// Python repr → JSON conversion
// ---------------------------------------------------------------------------

fn python_repr_to_json(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let bytes = input.as_bytes();
    let len = bytes.len();
    let mut i = 0;
    while i < len {
        let c = bytes[i] as char;
        match c {
            '\'' => {
                out.push('"');
                i += 1;
                while i < len {
                    let sc = bytes[i] as char;
                    match sc {
                        '\\' if i + 1 < len => {
                            let next = bytes[i + 1] as char;
                            if next == '\'' {
                                out.push('\'');
                                i += 2;
                            } else {
                                out.push('\\');
                                out.push(next);
                                i += 2;
                            }
                        }
                        '\'' => {
                            out.push('"');
                            i += 1;
                            break;
                        }
                        '"' => {
                            out.push('\\');
                            out.push('"');
                            i += 1;
                        }
                        _ => {
                            out.push(sc);
                            i += 1;
                        }
                    }
                }
            }
            '"' => {
                out.push('"');
                i += 1;
                while i < len {
                    let sc = bytes[i] as char;
                    match sc {
                        '\\' if i + 1 < len => {
                            out.push('\\');
                            out.push(bytes[i + 1] as char);
                            i += 2;
                        }
                        '"' => {
                            out.push('"');
                            i += 1;
                            break;
                        }
                        _ => {
                            out.push(sc);
                            i += 1;
                        }
                    }
                }
            }
            'N' if input[i..].starts_with("None") => {
                out.push_str("null");
                i += 4;
            }
            'T' if input[i..].starts_with("True") => {
                out.push_str("true");
                i += 4;
            }
            'F' if input[i..].starts_with("False") => {
                out.push_str("false");
                i += 5;
            }
            _ => {
                out.push(c);
                i += 1;
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// CSV loading
// ---------------------------------------------------------------------------

fn load_csv(path: &str, limit: Option<usize>) -> Result<Vec<EvalRow>> {
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_path(path)
        .map_err(|e| anyhow::anyhow!("Failed to open CSV at {path}: {e}"))?;

    let mut rows = Vec::new();
    for (idx, result) in rdr.records().enumerate() {
        if let Some(limit) = limit
            && idx >= limit
        {
            break;
        }
        let record = result.map_err(|e| anyhow::anyhow!("CSV parse error at row {idx}: {e}"))?;
        let expected = record.get(0).unwrap_or("").to_string();
        let fragments_raw = record.get(1).unwrap_or("[]").to_string();
        let question = record.get(4).unwrap_or("").to_string();
        let fragments_json = python_repr_to_json(&fragments_raw);
        let fragments: Vec<Fragment> = serde_json::from_str(&fragments_json).unwrap_or_else(|e| {
            if idx == 0 {
                eprintln!("Warning: failed to parse fragments for row {idx}: {e}");
            }
            vec![]
        });
        rows.push(EvalRow {
            index: idx,
            expected,
            fragments,
            question,
        });
    }
    Ok(rows)
}

// ---------------------------------------------------------------------------
// String metrics
// ---------------------------------------------------------------------------

fn normalize_tokens(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect()
}

fn token_f1(expected: &str, actual: &str) -> f64 {
    let et = normalize_tokens(expected);
    let at = normalize_tokens(actual);
    if et.is_empty() && at.is_empty() {
        return 100.0;
    }
    if et.is_empty() || at.is_empty() {
        return 0.0;
    }
    let ec: HashMap<&str, usize> = et.iter().fold(HashMap::new(), |mut m, t| {
        *m.entry(t.as_str()).or_insert(0) += 1;
        m
    });
    let ac: HashMap<&str, usize> = at.iter().fold(HashMap::new(), |mut m, t| {
        *m.entry(t.as_str()).or_insert(0) += 1;
        m
    });
    let common: usize = ac
        .iter()
        .map(|(tok, &c)| c.min(*ec.get(tok).unwrap_or(&0)))
        .sum();
    if common == 0 {
        return 0.0;
    }
    let p = common as f64 / at.len() as f64;
    let r = common as f64 / et.len() as f64;
    2.0 * p * r / (p + r) * 100.0
}

fn exact_match(expected: &str, actual: &str) -> f64 {
    if normalize_tokens(expected).join(" ") == normalize_tokens(actual).join(" ") {
        100.0
    } else {
        0.0
    }
}

fn bleu_1(expected: &str, actual: &str) -> f64 {
    let rt = normalize_tokens(expected);
    let ht = normalize_tokens(actual);
    if rt.is_empty() || ht.is_empty() {
        return 0.0;
    }
    let rc: HashMap<&str, usize> = rt.iter().fold(HashMap::new(), |mut m, t| {
        *m.entry(t.as_str()).or_insert(0) += 1;
        m
    });
    let hc: HashMap<&str, usize> = ht.iter().fold(HashMap::new(), |mut m, t| {
        *m.entry(t.as_str()).or_insert(0) += 1;
        m
    });
    let clipped: usize = hc
        .iter()
        .map(|(tok, &c)| c.min(*rc.get(tok).unwrap_or(&0)))
        .sum();
    let p = clipped as f64 / ht.len() as f64;
    let bp = if ht.len() >= rt.len() {
        1.0
    } else {
        (1.0 - rt.len() as f64 / ht.len() as f64).exp()
    };
    bp * p * 100.0
}

fn parse_accuracy(response: &str) -> f64 {
    response
        .trim()
        .split(|c: char| !c.is_ascii_digit())
        .find(|s| !s.is_empty())
        .and_then(|s| s.parse::<f64>().ok())
        .map(|v| v.clamp(0.0, 100.0))
        .unwrap_or(0.0)
}

// ---------------------------------------------------------------------------
// LLM construction
// ---------------------------------------------------------------------------

fn build_llm(args: &BenchRagcsvArgs) -> Result<LLM> {
    let model = args.resolved_model().map_err(|e| anyhow::anyhow!(e))?;
    let mut builder = LLMBuilder::new(&model)
        .device(&args.device)
        .dtype(&args.dtype)
        .gpu_memory_utilization(args.gpu_memory_utilization)
        .max_num_seqs(args.max_num_seqs)
        .block_size(args.block_size)
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

// ---------------------------------------------------------------------------
// LLM-judge helpers
// ---------------------------------------------------------------------------

fn grade_accuracy(llm: &mut LLM, expected: &str, actual: &str) -> f64 {
    let system = "You are an accuracy evaluator. Compare the expected answer to the actual answer and return ONLY a single integer 0-100 representing accuracy percentage. 100 means perfectly correct, 0 means completely wrong.";
    let user = format!(
        "Expected answer: {expected}\n\nActual answer: {actual}\n\nAccuracy score (0-100):"
    );
    let sampling = SamplingParams {
        max_tokens: Some(16),
        temperature: 0.0,
        ..SamplingParams::default()
    };
    match llm.chat(
        &[ChatMessage::system(system), ChatMessage::user(&user)],
        Some(sampling),
    ) {
        Ok(r) => parse_accuracy(&r.outputs[0].text),
        Err(e) => {
            eprintln!("  accuracy grading error: {e}");
            0.0
        }
    }
}

fn grade_faithfulness(llm: &mut LLM, answer: &str, fragments: &[Fragment]) -> f64 {
    let system = "You are a faithfulness evaluator. Determine whether the answer is grounded in the provided documents. Return ONLY a single integer 0-100. 100 means fully grounded, 0 means completely fabricated.";
    let docs: String = fragments
        .iter()
        .enumerate()
        .map(|(i, f)| format!("Document {i}: {}", f.page_content))
        .collect::<Vec<_>>()
        .join("\n\n");
    let user = format!("Documents:\n{docs}\n\nAnswer: {answer}\n\nFaithfulness score (0-100):");
    let sampling = SamplingParams {
        max_tokens: Some(16),
        temperature: 0.0,
        ..SamplingParams::default()
    };
    match llm.chat(
        &[ChatMessage::system(system), ChatMessage::user(&user)],
        Some(sampling),
    ) {
        Ok(r) => parse_accuracy(&r.outputs[0].text),
        Err(e) => {
            eprintln!("  faithfulness grading error: {e}");
            0.0
        }
    }
}

fn grade_relevancy(llm: &mut LLM, question: &str, answer: &str) -> f64 {
    let system = "You are a relevancy evaluator. Determine whether the answer addresses the question. Return ONLY a single integer 0-100. 100 means the answer fully addresses the question, 0 means completely off-topic.";
    let user = format!("Question: {question}\n\nAnswer: {answer}\n\nRelevancy score (0-100):");
    let sampling = SamplingParams {
        max_tokens: Some(16),
        temperature: 0.0,
        ..SamplingParams::default()
    };
    match llm.chat(
        &[ChatMessage::system(system), ChatMessage::user(&user)],
        Some(sampling),
    ) {
        Ok(r) => parse_accuracy(&r.outputs[0].text),
        Err(e) => {
            eprintln!("  relevancy grading error: {e}");
            0.0
        }
    }
}

// ---------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------

struct ModeResult {
    #[allow(dead_code)]
    text: String,
    f1: f64,
    em: f64,
    bleu: f64,
    ms: f64,
    acc: f64,
    faith: f64,
    rel: f64,
}

fn avg_pair(
    plain: &[ModeResult],
    span: &[ModeResult],
    f: impl Fn(&ModeResult) -> f64,
) -> (f64, f64) {
    let n = plain.len() as f64;
    let p = plain.iter().map(&f).sum::<f64>() / n;
    let s = span.iter().map(&f).sum::<f64>() / n;
    (p, s)
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

pub(crate) fn run_bench_ragcsv(args: BenchRagcsvArgs) -> Result<()> {
    scratchy_core_common::telemetry::init_tracing(&args.log_level);

    let flags = MetricFlags::from_arg(&args.metrics);
    let rows = load_csv(&args.file, args.limit)?;
    let total = rows.len();

    eprintln!("\n=== RAGCSV Benchmark ===");
    eprintln!("Loaded {} rows from {}", total, args.file);
    eprintln!("Metrics: {}", args.metrics);
    eprintln!("Max tokens: {}", args.max_tokens);

    if total == 0 {
        eprintln!("No rows to process.");
        return Ok(());
    }

    let mut llm = build_llm(&args)?;
    let model_name = llm.model_name().to_string();

    let sampling = SamplingParams {
        max_tokens: Some(args.max_tokens as u32),
        temperature: 0.0,
        ..SamplingParams::default()
    };

    let system_prompt =
        "You are a helpful assistant. Answer the question based only on the provided Documents.";

    let plain_style = ProgressStyle::default_bar()
        .template("  plain {bar:40.yellow/yellow} {pos:>4}/{len} {msg}")
        .unwrap();
    let spans_style = ProgressStyle::default_bar()
        .template("  spans {bar:40.cyan/blue} {pos:>4}/{len} {msg}")
        .unwrap();

    // --- Plain pass ---
    let pb = ProgressBar::new(total as u64)
        .with_style(plain_style)
        .with_message("RAGCSV");

    let mut plain_results: Vec<ModeResult> = Vec::with_capacity(total);
    for row in &rows {
        let debug = args.debug && row.index == 0;

        let doc_messages: Vec<ChatMessage> = row
            .fragments
            .iter()
            .enumerate()
            .map(|(i, f)| ChatMessage::user(format!("Document {i}: {}", f.page_content)))
            .collect();

        let mut msgs = vec![ChatMessage::system(system_prompt)];
        msgs.extend(doc_messages);
        msgs.push(ChatMessage::user(&row.question));

        let start = Instant::now();
        let result = llm.chat(&msgs, Some(sampling.clone()))?;
        let ms = start.elapsed().as_secs_f64() * 1000.0;
        let text = result.outputs[0].text.clone();

        if debug {
            eprintln!("  plain response: {text}");
        }

        let f1 = token_f1(&row.expected, &text);
        let em = exact_match(&row.expected, &text);
        let bl = bleu_1(&row.expected, &text);
        let acc = if flags.accuracy {
            grade_accuracy(&mut llm, &row.expected, &text)
        } else {
            -1.0
        };
        let faith = if flags.faithfulness {
            grade_faithfulness(&mut llm, &text, &row.fragments)
        } else {
            -1.0
        };
        let rel = if flags.relevancy {
            grade_relevancy(&mut llm, &row.question, &text)
        } else {
            -1.0
        };

        plain_results.push(ModeResult {
            text,
            f1,
            em,
            bleu: bl,
            ms,
            acc,
            faith,
            rel,
        });
        pb.inc(1);
    }
    pb.finish();

    // --- Span pass ---
    let pb = ProgressBar::new(total as u64)
        .with_style(spans_style)
        .with_message("RAGCSV");

    let mut span_results: Vec<ModeResult> = Vec::with_capacity(total);
    for row in &rows {
        let debug = args.debug && row.index == 0;

        let doc_json: Vec<serde_json::Value> = row
            .fragments
            .iter()
            .enumerate()
            .map(
                |(i, f)| serde_json::json!({ "user": format!("Document {i}: {}", f.page_content) }),
            )
            .collect();

        let query = serde_json::json!({
            "g": {
                "model": model_name,
                "max_tokens": args.max_tokens,
                "temperature": 0.0,
                "input": {
                    "cross": [
                        { "system": system_prompt },
                        { "plus": doc_json },
                        { "user": &row.question }
                    ]
                }
            }
        });

        let start = Instant::now();
        let results =
            llm.execute_query(&query.to_string(), Some(sampling.clone()), false, false)?;
        let ms = start.elapsed().as_secs_f64() * 1000.0;
        let text = results[0].outputs[0].text.clone();

        if debug {
            eprintln!("  span response: {text}");
        }

        let f1 = token_f1(&row.expected, &text);
        let em = exact_match(&row.expected, &text);
        let bl = bleu_1(&row.expected, &text);
        let acc = if flags.accuracy {
            grade_accuracy(&mut llm, &row.expected, &text)
        } else {
            -1.0
        };
        let faith = if flags.faithfulness {
            grade_faithfulness(&mut llm, &text, &row.fragments)
        } else {
            -1.0
        };
        let rel = if flags.relevancy {
            grade_relevancy(&mut llm, &row.question, &text)
        } else {
            -1.0
        };

        span_results.push(ModeResult {
            text,
            f1,
            em,
            bleu: bl,
            ms,
            acc,
            faith,
            rel,
        });
        pb.inc(1);
    }
    pb.finish();

    // --- Summary ---
    let n = total as f64;

    eprintln!("\n=== RAGCSV Results (n={total}) ===");

    eprintln!("\n  String metrics (plain / span):");
    let (pf1, sf1) = avg_pair(&plain_results, &span_results, |r| r.f1);
    eprintln!("    Token F1:    {pf1:.1}% / {sf1:.1}%");
    let (pem, sem) = avg_pair(&plain_results, &span_results, |r| r.em);
    eprintln!("    Exact Match: {pem:.1}% / {sem:.1}%");
    let (pbl, sbl) = avg_pair(&plain_results, &span_results, |r| r.bleu);
    eprintln!("    BLEU-1:      {pbl:.1}% / {sbl:.1}%");

    let pa: Vec<f64> = plain_results
        .iter()
        .map(|r| r.acc)
        .filter(|&v| v >= 0.0)
        .collect();
    let sa: Vec<f64> = span_results
        .iter()
        .map(|r| r.acc)
        .filter(|&v| v >= 0.0)
        .collect();
    if !pa.is_empty() {
        eprintln!("\n  LLM-judge metrics (plain / span):");
        let pa_avg = pa.iter().sum::<f64>() / pa.len() as f64;
        let sa_avg = sa.iter().sum::<f64>() / sa.len() as f64;
        eprintln!("    Accuracy:      {pa_avg:.1}% / {sa_avg:.1}%");
    }
    let pf: Vec<f64> = plain_results
        .iter()
        .map(|r| r.faith)
        .filter(|&v| v >= 0.0)
        .collect();
    let sf: Vec<f64> = span_results
        .iter()
        .map(|r| r.faith)
        .filter(|&v| v >= 0.0)
        .collect();
    if !pf.is_empty() {
        let pf_avg = pf.iter().sum::<f64>() / pf.len() as f64;
        let sf_avg = sf.iter().sum::<f64>() / sf.len() as f64;
        eprintln!("    Faithfulness:  {pf_avg:.1}% / {sf_avg:.1}%");
    }
    let pr: Vec<f64> = plain_results
        .iter()
        .map(|r| r.rel)
        .filter(|&v| v >= 0.0)
        .collect();
    let sr: Vec<f64> = span_results
        .iter()
        .map(|r| r.rel)
        .filter(|&v| v >= 0.0)
        .collect();
    if !pr.is_empty() {
        let pr_avg = pr.iter().sum::<f64>() / pr.len() as f64;
        let sr_avg = sr.iter().sum::<f64>() / sr.len() as f64;
        eprintln!("    Relevancy:     {pr_avg:.1}% / {sr_avg:.1}%");
    }

    let plain_avg_ms = plain_results.iter().map(|r| r.ms).sum::<f64>() / n;
    let span_avg_ms = span_results.iter().map(|r| r.ms).sum::<f64>() / n;
    eprintln!(
        "\n  spans vs plain: {:.1}ms vs {:.1}ms ({:.2}x faster)",
        span_avg_ms,
        plain_avg_ms,
        plain_avg_ms / span_avg_ms
    );

    eprintln!("\n=== RAGCSV Benchmark Complete ===\n");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn python_repr_single_quoted_strings() {
        assert_eq!(python_repr_to_json("{'a': 'b'}"), r#"{"a": "b"}"#);
    }

    #[test]
    fn python_repr_none_true_false() {
        assert_eq!(
            python_repr_to_json("{'x': None, 'y': True, 'z': False}"),
            r#"{"x": null, "y": true, "z": false}"#
        );
    }

    #[test]
    fn metric_flags_all() {
        let f = MetricFlags::from_arg("all");
        assert!(f.accuracy && f.faithfulness && f.relevancy);
    }

    #[test]
    fn token_f1_identical() {
        assert!((token_f1("the cat sat", "the cat sat") - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn exact_match_after_normalization() {
        assert!((exact_match("Hello, World!", "hello world") - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn bleu_1_identical() {
        assert!((bleu_1("the cat sat", "the cat sat") - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn parse_accuracy_numeric() {
        assert!((parse_accuracy("85") - 85.0).abs() < f64::EPSILON);
    }

    #[test]
    fn parse_accuracy_clamps_above_100() {
        assert!((parse_accuracy("150") - 100.0).abs() < f64::EPSILON);
    }
}
