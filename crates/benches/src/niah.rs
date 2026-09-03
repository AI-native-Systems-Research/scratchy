// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! `scr bench niah` — Needle-in-a-Haystack accuracy benchmark.
//!
//! Inserts a known "needle" fact into a long context of Paul Graham essays
//! and measures whether the model can retrieve it at various context lengths
//! and insertion depths.
//!
//! Three variants:
//! 1. **Plain** — `llm.chat()`, one big context string, full prefill every time.
//! 2. **Chunked** — `llm.execute_query()` with `cross`, essays split into
//!    chunks but no `plus` annotations — proves chunking doesn't hurt accuracy.
//! 3. **Spans** — `llm.execute_query()` with `cross` + `plus` wrapping each
//!    chunk as a relocatable block. Cache populated once, then reused.

use std::io::Write;
use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use indicatif::{ProgressBar, ProgressStyle};
use scratchy_core_config::{CudaGraphConfig, CudaGraphMode};
use scratchy_serving_api::llm::{ChatMessage, LLM, LLMBuilder, SamplingParams};
use scratchy_serving_api::tokenizer::Tokenizer;

use crate::args::BenchNiahArgs;

// ---------------------------------------------------------------------------
// Paul Graham essay fetching (cached to disk)
// ---------------------------------------------------------------------------

const PG_ESSAYS_API_URL: &str = "https://api.github.com/repos/gkamradt/LLMTest_NeedleInAHaystack/contents/needlehaystack/PaulGrahamEssays";

const FALLBACK_ESSAYS: &str = r#"The way to get startup ideas is not to try to think of startup ideas. It's to look for problems, preferably problems you have yourself. The very best startup ideas tend to have three things in common: they're something the founders themselves want, that they themselves can build, and that few others realize are worth doing. Microsoft, Apple, Yahoo, Google, and Facebook all began this way.

One of the biggest things holding people back from doing great work is the fear of making something lame. And this fear is not an irrational one. Many things that are new are bad. But the way to get good ideas is to get lots of ideas. The way to get lots of ideas is to lower your standards. If you don't lower your standards, you won't get any ideas at all.

The most important quality in a startup founder is determination. Not intelligence—determination. This is a little depressing. It would be nice if intelligence were the most important quality, since that's what we're usually judged by. But determination is more important, because intelligence without determination is like a car without an engine."#;

#[derive(serde::Deserialize)]
struct GitHubFile {
    name: String,
    download_url: String,
}

fn fetch_pg_essays() -> Result<String> {
    let cache_dir = dirs::cache_dir()
        .ok_or_else(|| anyhow::anyhow!("Could not determine cache directory"))?
        .join("scr-bench")
        .join("niah");
    std::fs::create_dir_all(&cache_dir)?;
    let cache_file = cache_dir.join("paul_graham_essays_combined.txt");

    if cache_file.exists() {
        return Ok(std::fs::read_to_string(&cache_file)?);
    }

    eprintln!("Downloading Paul Graham essays from GitHub...");
    let agent = crate::http::agent(false);
    let files: Vec<GitHubFile> = crate::http::get_json(&agent, PG_ESSAYS_API_URL)?;
    let mut combined = String::new();
    let txt_files: Vec<_> = files
        .into_iter()
        .filter(|f| f.name.ends_with(".txt"))
        .collect();

    for (i, file) in txt_files.iter().enumerate() {
        eprint!("\rDownloading essay {}/{}...", i + 1, txt_files.len());
        let text = crate::http::get_text(&agent, &file.download_url)?;
        combined.push_str(&text);
        combined.push('\n');
    }
    eprintln!("\nDownload complete!");

    if combined.is_empty() {
        combined = FALLBACK_ESSAYS.to_string();
    }
    std::fs::File::create(&cache_file)?.write_all(combined.as_bytes())?;
    Ok(combined)
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

fn encode_and_trim(text: &str, max_tokens: usize, tokenizer: &Tokenizer) -> Result<String> {
    let ids = tokenizer
        .encode(text, false)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    if ids.len() > max_tokens {
        tokenizer
            .inner()
            .decode(&ids[..max_tokens], false)
            .map_err(|e| anyhow::anyhow!("{e}"))
    } else {
        Ok(text.to_string())
    }
}

// ---------------------------------------------------------------------------
// Context generation — returns chunks with needle at the right position
// ---------------------------------------------------------------------------

const NEEDLE: &str = "The special magic number mentioned in the context is 73.";
const QUESTION: &str = "What is the special magic number mentioned in the context?";
const EXPECTED_ANSWER: &str = "73";

/// Target number of tokens per essay chunk.
const CHUNK_TARGET_TOKENS: usize = 256;

/// Generate the context as a list of chunks (essay segments + needle).
///
/// The needle is placed at a position determined by `depth_percent` — it
/// becomes its own chunk inserted between essay chunks.
///
/// Returns: (chunks, flat_context)
///   - `chunks`: ordered list of text chunks; each essay segment and the
///     needle are separate entries
///   - `flat_context`: the same content as one concatenated string (for plain mode)
fn generate_chunked_context(
    essays: &str,
    context_length: usize,
    depth_percent: usize,
    buffer: usize,
    tokenizer: &Tokenizer,
) -> Result<(Vec<String>, String)> {
    let adjusted = context_length.saturating_sub(buffer);

    // Build essay text long enough to fill the context.
    let mut raw = String::new();
    while token_len(&raw, tokenizer) < adjusted {
        raw.push_str(essays);
        raw.push(' ');
    }
    let raw = encode_and_trim(&raw, adjusted, tokenizer)?;

    // Split into sentence-boundary chunks of ~CHUNK_TARGET_TOKENS tokens each.
    let sentences: Vec<&str> = raw.split_inclusive('.').collect();
    let mut essay_chunks: Vec<String> = Vec::new();
    let mut current = String::new();
    for sentence in &sentences {
        current.push_str(sentence);
        if token_len(&current, tokenizer) >= CHUNK_TARGET_TOKENS {
            essay_chunks.push(current.clone());
            current.clear();
        }
    }
    if !current.is_empty() {
        essay_chunks.push(current);
    }

    // Trim total essay tokens to leave room for the needle.
    let needle_tokens = token_len(NEEDLE, tokenizer);
    while essay_chunks.len() > 1 {
        let total: usize = essay_chunks.iter().map(|c| token_len(c, tokenizer)).sum();
        if total + needle_tokens <= adjusted {
            break;
        }
        essay_chunks.pop();
    }

    // Insert needle chunk at depth_percent position.
    let insert_idx = if depth_percent >= 100 {
        essay_chunks.len()
    } else {
        (essay_chunks.len() * depth_percent) / 100
    };
    essay_chunks.insert(insert_idx, NEEDLE.to_string());

    // Build flat context for plain mode.
    let flat = essay_chunks.join("\n");

    Ok((essay_chunks, flat))
}

// ---------------------------------------------------------------------------
// Evaluation
// ---------------------------------------------------------------------------

fn evaluate(response: &str, debug: bool) -> f64 {
    let response_lower = response.to_lowercase();
    let expected_lower = EXPECTED_ANSWER.to_lowercase();

    if debug {
        eprintln!("  expected: {EXPECTED_ANSWER}");
        eprintln!("  response: {response}");
    }

    if response_lower.contains(&expected_lower) {
        return 1.0;
    }
    if let Ok(expected_num) = EXPECTED_ANSWER.parse::<i32>() {
        for word in response.split_whitespace() {
            let cleaned = word.trim_matches(|c: char| !c.is_numeric());
            if let Ok(num) = cleaned.parse::<i32>()
                && num == expected_num
            {
                return 1.0;
            }
        }
    }
    0.0
}

// ---------------------------------------------------------------------------
// LLM construction
// ---------------------------------------------------------------------------

fn build_llm(args: &BenchNiahArgs) -> Result<LLM> {
    let model = args.resolved_model().map_err(|e| anyhow::anyhow!(e))?;
    let mut builder = LLMBuilder::new(&model)
        .device(&args.device)
        .dtype(&args.dtype)
        .gpu_memory_utilization(args.gpu_memory_utilization)
        .max_num_seqs(args.max_num_seqs)
        .block_size(args.block_size)
        .enforce_eager(args.enforce_eager)
        .enable_prefix_caching(true);

    let max_ctx = *args.context_lengths.iter().max().unwrap_or(&8000);
    builder = builder.max_num_batched_tokens((max_ctx + 512).max(8192));

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
// Query builders
// ---------------------------------------------------------------------------

/// Build a SPNL query with chunks as sequential messages inside `cross`
/// (no relocatable annotations — proves chunking is neutral).
fn build_chunked_query(
    model: &str,
    system: &str,
    chunks: &[String],
    question: &str,
) -> serde_json::Value {
    let mut cross_children: Vec<serde_json::Value> = Vec::new();
    cross_children.push(serde_json::json!({ "system": system }));
    for chunk in chunks {
        cross_children.push(serde_json::json!({ "user": chunk }));
    }
    cross_children.push(serde_json::json!({ "user": question }));

    serde_json::json!({
        "g": {
            "model": model,
            "max_tokens": 300,
            "temperature": 0.0,
            "input": { "cross": cross_children }
        }
    })
}

/// Build a SPNL query with chunks wrapped in `plus` (each chunk is a
/// relocatable block) inside `cross`.
fn build_spans_query(
    model: &str,
    system: &str,
    chunks: &[String],
    question: &str,
) -> serde_json::Value {
    let plus_children: Vec<serde_json::Value> = chunks
        .iter()
        .map(|c| serde_json::json!({ "user": c }))
        .collect();

    serde_json::json!({
        "g": {
            "model": model,
            "max_tokens": 300,
            "temperature": 0.0,
            "input": {
                "cross": [
                    { "system": system },
                    { "plus": plus_children },
                    { "user": question }
                ]
            }
        }
    })
}

// ---------------------------------------------------------------------------
// Per-variant runner
// ---------------------------------------------------------------------------

struct VariantResult {
    accs: Vec<f64>,
    ms: Vec<f64>,
}

fn print_results(
    plain: &VariantResult,
    padded: &VariantResult,
    chunked: &VariantResult,
    spans: &VariantResult,
) {
    let n = plain.accs.len();
    let w = format!("{n}").len();
    let plain_ms = plain.ms.iter().sum::<f64>() / n as f64;
    let plain_acc = plain.accs.iter().sum::<f64>() / n as f64;

    let fmt_line = |name: &str, r: &VariantResult, suffix: &str| {
        let acc = r.accs.iter().sum::<f64>() / n as f64;
        let perfect = r.accs.iter().filter(|&&a| a >= 1.0).count();
        let avg_ms = r.ms.iter().sum::<f64>() / n as f64;
        eprintln!(
            "  {name:>9}: acc={:>5.1}%  perfect={:>w$}/{}  {:>6.0}ms avg{suffix}",
            acc * 100.0,
            perfect,
            n,
            avg_ms,
            w = w,
        );
    };

    let vs_plain = |r: &VariantResult| -> String {
        let ms = r.ms.iter().sum::<f64>() / n as f64;
        let acc = r.accs.iter().sum::<f64>() / n as f64;
        format!(
            "  ({:.2}x, {:+.1}pp vs plain)",
            plain_ms / ms,
            (acc - plain_acc) * 100.0,
        )
    };

    fmt_line("plain", plain, "");
    fmt_line("padded", padded, &vs_plain(padded));
    fmt_line("chunked", chunked, &vs_plain(chunked));
    fmt_line("spans", spans, &vs_plain(spans));
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

pub(crate) fn run_bench_niah(args: BenchNiahArgs) -> Result<()> {
    scratchy_core_common::telemetry::init_tracing(&args.log_level);

    eprintln!("\n=== Needle In A Haystack Benchmark ===");
    eprintln!("Context lengths: {:?}", args.context_lengths);
    eprintln!("Depth %%:        {:?}", args.depth_percentages);
    eprintln!("Samples/config:  {}", args.num_samples);

    let essays = fetch_pg_essays()?;
    let mut llm = build_llm(&args)?;
    let model_name = llm.model_name().to_string();

    let tokenizer: Arc<Tokenizer> = llm
        .tokenizer()
        .ok_or_else(|| anyhow::anyhow!("NIAH bench requires a tokenizer"))?
        .clone();

    let sampling = SamplingParams {
        max_tokens: Some(300),
        temperature: 0.0,
        ..SamplingParams::default()
    };

    let system_prompt = "You are a helpful AI assistant. Answer the question based only on the information provided in the context. Be concise and direct.";

    let plain_style = ProgressStyle::default_bar()
        .template("    plain {bar:40.yellow/yellow} {pos:>4}/{len} {msg}")
        .unwrap();
    let chunked_style = ProgressStyle::default_bar()
        .template("  chunked {bar:40.magenta/magenta} {pos:>4}/{len} {msg}")
        .unwrap();
    let spans_style = ProgressStyle::default_bar()
        .template("    spans {bar:40.cyan/blue} {pos:>4}/{len} {msg}")
        .unwrap();

    for &context_length in &args.context_lengths {
        for &depth_percent in &args.depth_percentages {
            let label = format!("len={} depth={}%", context_length, depth_percent);
            eprintln!("\n--- {label} ---");

            let n = args.num_samples;
            let (chunks, flat_context) = generate_chunked_context(
                &essays,
                context_length,
                depth_percent,
                args.context_length_buffer,
                &tokenizer,
            )?;

            // Build a padded version: each chunk padded with spaces to block boundary.
            let block_size = args.block_size;
            let padded_context = {
                let mut parts = Vec::new();
                for chunk in &chunks {
                    let toks = token_len(chunk, &tokenizer);
                    let remainder = toks % block_size;
                    if remainder == 0 {
                        parts.push(chunk.clone());
                    } else {
                        let pad_count = block_size - remainder;
                        parts.push(format!("{}{}", chunk, " ".repeat(pad_count)));
                    }
                }
                parts.join("\n")
            };

            eprintln!(
                "  {} chunks, {} tokens (plain), {} tokens (padded)",
                chunks.len(),
                token_len(&flat_context, &tokenizer),
                token_len(&padded_context, &tokenizer),
            );

            // === 1. Plain: llm.chat(), one big string, reset cache each time ===
            let mut plain = VariantResult {
                accs: Vec::with_capacity(n),
                ms: Vec::with_capacity(n),
            };
            let pb = ProgressBar::new(n as u64)
                .with_style(plain_style.clone())
                .with_message(label.clone());

            for i in 0..n {
                let debug = args.debug && i == 0;
                llm.reset_prefix_cache()?;
                let messages = vec![
                    ChatMessage::system(system_prompt),
                    ChatMessage::user(&flat_context),
                    ChatMessage::user(QUESTION),
                ];
                let start = Instant::now();
                let result = llm.chat(&messages, Some(sampling.clone()))?;
                let ms = start.elapsed().as_secs_f64() * 1000.0;
                let acc = evaluate(&result.outputs[0].text, debug);
                plain.accs.push(acc);
                plain.ms.push(ms);
                let avg = plain.accs.iter().sum::<f64>() / plain.accs.len() as f64;
                pb.set_message(format!("{label}  acc={:.0}%", avg * 100.0));
                pb.inc(1);
            }
            pb.finish();

            // === 2. Padded: llm.chat() with padded context, reset cache each time ===
            let mut padded = VariantResult {
                accs: Vec::with_capacity(n),
                ms: Vec::with_capacity(n),
            };
            let padded_style = ProgressStyle::default_bar()
                .template("   padded {bar:40.red/red} {pos:>4}/{len} {msg}")
                .unwrap();
            let pb = ProgressBar::new(n as u64)
                .with_style(padded_style)
                .with_message(label.clone());

            for i in 0..n {
                let debug = args.debug && i == 0;
                llm.reset_prefix_cache()?;
                let messages = vec![
                    ChatMessage::system(system_prompt),
                    ChatMessage::user(&padded_context),
                    ChatMessage::user(QUESTION),
                ];
                let start = Instant::now();
                let result = llm.chat(&messages, Some(sampling.clone()))?;
                let ms = start.elapsed().as_secs_f64() * 1000.0;
                let acc = evaluate(&result.outputs[0].text, debug);
                padded.accs.push(acc);
                padded.ms.push(ms);
                let avg = padded.accs.iter().sum::<f64>() / padded.accs.len() as f64;
                pb.set_message(format!("{label}  acc={:.0}%", avg * 100.0));
                pb.inc(1);
            }
            pb.finish();

            // === 3. Chunked: execute_query with cross (no plus), reset each time ===
            let mut chunked = VariantResult {
                accs: Vec::with_capacity(n),
                ms: Vec::with_capacity(n),
            };
            let pb = ProgressBar::new(n as u64)
                .with_style(chunked_style.clone())
                .with_message(label.clone());

            for i in 0..n {
                let debug = args.debug && i == 0;
                llm.reset_prefix_cache()?;
                let query = build_chunked_query(&model_name, system_prompt, &chunks, QUESTION);
                let start = Instant::now();
                let results =
                    llm.execute_query(&query.to_string(), Some(sampling.clone()), false, false)?;
                let ms = start.elapsed().as_secs_f64() * 1000.0;
                let acc = evaluate(&results[0].outputs[0].text, debug);
                chunked.accs.push(acc);
                chunked.ms.push(ms);
                let avg = chunked.accs.iter().sum::<f64>() / chunked.accs.len() as f64;
                pb.set_message(format!("{label}  acc={:.0}%", avg * 100.0));
                pb.inc(1);
            }
            pb.finish();

            // === 4. Spans: execute_query with plus, populate once then reuse ===
            let mut spans = VariantResult {
                accs: Vec::with_capacity(n),
                ms: Vec::with_capacity(n),
            };

            // Populate cache.
            llm.reset_prefix_cache()?;
            let populate = build_spans_query(&model_name, system_prompt, &chunks, QUESTION);
            llm.execute_query(&populate.to_string(), Some(sampling.clone()), false, false)?;

            // Measured runs — chunk blocks should be cached.
            let pb = ProgressBar::new(n as u64)
                .with_style(spans_style.clone())
                .with_message(label.clone());

            for i in 0..n {
                let debug = args.debug && i == 0;
                let query = build_spans_query(&model_name, system_prompt, &chunks, QUESTION);
                let start = Instant::now();
                let results =
                    llm.execute_query(&query.to_string(), Some(sampling.clone()), false, false)?;
                let ms = start.elapsed().as_secs_f64() * 1000.0;
                let acc = evaluate(&results[0].outputs[0].text, debug);
                spans.accs.push(acc);
                spans.ms.push(ms);
                let avg = spans.accs.iter().sum::<f64>() / spans.accs.len() as f64;
                pb.set_message(format!("{label}  acc={:.0}%", avg * 100.0));
                pb.inc(1);
            }
            pb.finish();

            print_results(&plain, &padded, &chunked, &spans);
        }
    }

    eprintln!("\n=== NIAH Benchmark Complete ===\n");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evaluate_exact_substring_match() {
        assert!((evaluate("The answer is 73.", false) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn evaluate_case_insensitive_match() {
        assert!((evaluate("the number is 73", false) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn evaluate_numeric_match_in_tokens() {
        assert!((evaluate("The number is (73).", false) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn evaluate_no_match_returns_zero() {
        assert!((evaluate("Nothing relevant here", false)).abs() < f64::EPSILON);
    }
}
