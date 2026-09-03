// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! `scr bench ruler` — RULER benchmark (multi-needle NIAH + variable tracking).
//!
//! Tests long-context retrieval and reasoning. Runs both plain (chat) and span
//! (SPNL query) modes, comparing accuracy and latency.

use std::io::Write;
use std::time::Instant;

use std::sync::Arc;

use anyhow::Result;
use indicatif::{ProgressBar, ProgressStyle};
use rand::Rng;
use scratchy_core_config::{CudaGraphConfig, CudaGraphMode};
use scratchy_serving_api::llm::{ChatMessage, LLM, LLMBuilder, SamplingParams};
use scratchy_serving_api::tokenizer::Tokenizer;

use crate::args::BenchRulerArgs;

// ---------------------------------------------------------------------------
// Essay fetching (same as niah — inlined to avoid invented shared modules)
// ---------------------------------------------------------------------------

const PG_ESSAYS_API_URL: &str = "https://api.github.com/repos/gkamradt/LLMTest_NeedleInAHaystack/contents/needlehaystack/PaulGrahamEssays";

const FALLBACK_ESSAYS: &str = r#"The way to get startup ideas is not to try to think of startup ideas. It's to look for problems, preferably problems you have yourself. The very best startup ideas tend to have three things in common: they're something the founders themselves want, that they themselves can build, and that few others realize are worth doing."#;

#[derive(serde::Deserialize)]
struct GitHubFile {
    name: String,
    download_url: String,
}

fn fetch_pg_essays() -> Result<String> {
    let cache_dir = dirs::cache_dir()
        .ok_or_else(|| anyhow::anyhow!("Could not determine cache directory"))?
        .join("scr-bench")
        .join("ruler");
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
// Evaluation
// ---------------------------------------------------------------------------

fn string_match_all(prediction: &str, references: &[String]) -> f64 {
    let pred_lower = prediction.to_lowercase();
    let matches: usize = references
        .iter()
        .filter(|r| pred_lower.contains(&r.to_lowercase()))
        .count();
    (matches as f64) / (references.len() as f64)
}

// ---------------------------------------------------------------------------
// NIAH task
// ---------------------------------------------------------------------------

fn generate_random_number() -> String {
    let mut rng = rand::rng();
    rng.random_range(1000000..10000000).to_string()
}

struct NIAHConfig {
    context_length: usize,
    depth_percent: usize,
    num_needle_k: usize,
    num_needle_v: usize,
    num_needle_q: usize,
    buffer: usize,
}

fn generate_niah_context(
    cfg: &NIAHConfig,
    tokenizer: &Tokenizer,
    essays: &str,
) -> Result<(String, Vec<String>)> {
    let NIAHConfig {
        context_length,
        depth_percent,
        num_needle_k,
        num_needle_v,
        num_needle_q,
        buffer,
    } = *cfg;
    let mut keys = Vec::new();
    let mut all_values = Vec::new();
    let mut needles = Vec::new();

    for _ in 0..num_needle_k {
        let key = generate_random_number();
        keys.push(key.clone());
        for _ in 0..num_needle_v {
            let value = generate_random_number();
            all_values.push(value.clone());
            needles.push(format!(
                "One of the special magic numbers for {} is: {}.",
                key, value
            ));
        }
    }

    let haystack_words: Vec<&str> = essays.split_whitespace().collect();
    let adjusted = context_length.saturating_sub(buffer);

    // Binary search for optimal haystack size in words.
    let mut lower = 100usize;
    let mut upper = haystack_words.len();
    let mut optimal = lower;
    while lower <= upper {
        let mid = (lower + upper) / 2;
        let test = haystack_words[..mid.min(haystack_words.len())].join(" ");
        if token_len(&test, tokenizer) + needles.len() * 20 <= adjusted {
            optimal = mid;
            lower = mid + 1;
        } else {
            if mid == 0 {
                break;
            }
            upper = mid - 1;
        }
    }

    let context_text = haystack_words[..optimal.min(haystack_words.len())].join(" ");
    let sentences: Vec<&str> = context_text.split('.').collect();
    let insertion_point = if depth_percent == 100 {
        sentences.len()
    } else {
        (sentences.len() * depth_percent) / 100
    };

    let mut result = sentences[..insertion_point].to_vec();
    for needle in &needles {
        result.push(needle.as_str());
    }
    result.extend_from_slice(&sentences[insertion_point..]);
    let context_text = encode_and_trim(&result.join("."), adjusted, tokenizer)?;

    let query_indices: Vec<usize> = (0..num_needle_q.min(num_needle_k)).collect();
    let query_keys: Vec<String> = query_indices.iter().map(|&i| keys[i].clone()).collect();
    let query_str = if query_keys.len() > 1 {
        format!(
            "{}, and {}",
            query_keys[..query_keys.len() - 1].join(", "),
            query_keys.last().unwrap()
        )
    } else {
        query_keys[0].clone()
    };

    let type_v = if num_needle_q * num_needle_v == 1 {
        "number"
    } else {
        "numbers"
    };
    let prompt = format!(
        "Some special magic {} are hidden within the following text. Make sure to memorize it. I will quiz you about the {} afterwards.\n{}\nWhat are all the special magic {} for {} mentioned in the provided text?",
        type_v, type_v, context_text, type_v, query_str
    );

    let expected: Vec<String> = query_indices
        .iter()
        .flat_map(|&i| all_values[i * num_needle_v..(i + 1) * num_needle_v].to_vec())
        .collect();

    Ok((prompt, expected))
}

// ---------------------------------------------------------------------------
// Variable Tracking task
// ---------------------------------------------------------------------------

fn generate_var_name() -> String {
    let mut rng = rand::rng();
    (0..5)
        .map(|_| rng.random_range(b'A'..=b'Z') as char)
        .collect()
}

fn generate_vt_context(
    context_length: usize,
    num_chains: usize,
    num_hops: usize,
    buffer: usize,
    tokenizer: &Tokenizer,
) -> Result<(String, Vec<String>)> {
    let mut rng = rand::rng();
    let mut all_vars = Vec::new();
    let mut chains = Vec::new();

    for _ in 0..num_chains {
        let initial_value = rng.random_range(10000..100000).to_string();
        let mut chain_vars = Vec::new();
        let mut chain_stmts = Vec::new();
        let first = generate_var_name();
        chain_vars.push(first.clone());
        chain_stmts.push(format!("VAR {} = {}", first, initial_value));
        for _ in 0..num_hops {
            let next = generate_var_name();
            chain_vars.push(next.clone());
            chain_stmts.push(format!(
                "VAR {} = VAR {}",
                next,
                chain_vars[chain_vars.len() - 2]
            ));
        }
        all_vars.push(chain_vars);
        chains.push(chain_stmts);
    }

    let noise = "The grass is green. The sky is blue.";
    let adjusted = context_length.saturating_sub(buffer);

    let mut lower = 10usize;
    let mut upper = 1000usize;
    let mut optimal = lower;
    while lower <= upper {
        let mid = (lower + upper) / 2;
        let mut test: Vec<&str> = vec![noise; mid];
        for chain in &chains {
            for s in chain {
                test.push(s.as_str());
            }
        }
        if token_len(&test.join("\n"), tokenizer) <= adjusted {
            optimal = mid;
            lower = mid + 1;
        } else {
            if mid == 0 {
                break;
            }
            upper = mid - 1;
        }
    }

    let mut sentences: Vec<&str> = vec![noise; optimal];
    for chain in &chains {
        for stmt in chain {
            let pos = rng.random_range(0..sentences.len());
            sentences.insert(pos, stmt.as_str());
        }
    }

    let context = encode_and_trim(&sentences.join("\n"), adjusted, tokenizer)?;
    let initial_value = chains[0][0].split('=').nth(1).unwrap().trim();
    let prompt = format!(
        "Memorize and track the chain(s) of variable assignment hidden in the following text.\n\n{}\nQuestion: Find all variables that are assigned the value {} in the text above.",
        context, initial_value
    );
    Ok((prompt, all_vars[0].clone()))
}

// ---------------------------------------------------------------------------
// LLM construction
// ---------------------------------------------------------------------------

fn build_llm(args: &BenchRulerArgs) -> Result<LLM> {
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
// Stats
// ---------------------------------------------------------------------------

fn print_comparison(plain_accs: &[f64], plain_ms: &[f64], span_accs: &[f64], span_ms: &[f64]) {
    let n = plain_accs.len();
    let pa = plain_accs.iter().sum::<f64>() / n as f64;
    let sa = span_accs.iter().sum::<f64>() / n as f64;
    let pm = plain_ms.iter().sum::<f64>() / n as f64;
    let sm = span_ms.iter().sum::<f64>() / n as f64;
    let pp = plain_accs.iter().filter(|&&a| a >= 1.0).count();
    let sp = span_accs.iter().filter(|&&a| a >= 1.0).count();
    let w = format!("{n}").len();
    eprintln!(
        "  plain: acc={:>5.1}%  perfect={:>w$}/{}  {:>6.0}ms avg",
        pa * 100.0,
        pp,
        n,
        pm,
        w = w,
    );
    eprintln!(
        "  spans: acc={:>5.1}%  perfect={:>w$}/{}  {:>6.0}ms avg  ({:.2}x, {:+.1}pp)",
        sa * 100.0,
        sp,
        n,
        sm,
        pm / sm,
        (sa - pa) * 100.0,
        w = w,
    );
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

pub(crate) fn run_bench_ruler(args: BenchRulerArgs) -> Result<()> {
    scratchy_core_common::telemetry::init_tracing(&args.log_level);

    let tasks: Vec<&str> = args.tasks.split(',').map(|s| s.trim()).collect();

    eprintln!("\n=== RULER Benchmark ===");
    eprintln!("Context lengths: {:?}", args.context_lengths);
    eprintln!("Tasks:           {:?}", tasks);
    eprintln!("Samples/config:  {}", args.num_samples);

    let essays = fetch_pg_essays()?;
    let mut llm = build_llm(&args)?;
    let model_name = llm.model_name().to_string();
    let tokenizer: Arc<Tokenizer> = llm
        .tokenizer()
        .ok_or_else(|| anyhow::anyhow!("RULER bench requires a tokenizer"))?
        .clone();

    let plain_style = ProgressStyle::default_bar()
        .template("  plain {bar:40.yellow/yellow} {pos:>4}/{len} {msg}")
        .unwrap();
    let spans_style = ProgressStyle::default_bar()
        .template("  spans {bar:40.cyan/blue} {pos:>4}/{len} {msg}")
        .unwrap();

    // NIAH task
    if tasks.contains(&"niah") {
        let sampling = SamplingParams {
            max_tokens: Some(128),
            temperature: 0.0,
            ..SamplingParams::default()
        };
        let system = "You are a helpful AI assistant. Answer based only on the provided context.";

        for &ctx_len in &args.context_lengths {
            for &depth in &args.niah_depth_percentages {
                let label = format!("NIAH len={} depth={}%", ctx_len, depth);
                eprintln!("\n--- {label} ---");

                let n = args.num_samples;
                let mut plain_accs = Vec::with_capacity(n);
                let mut plain_ms = Vec::with_capacity(n);
                let mut span_accs = Vec::with_capacity(n);
                let mut span_ms = Vec::with_capacity(n);

                // Generate one context per config (same data for both modes).
                let niah_cfg = NIAHConfig {
                    context_length: ctx_len,
                    depth_percent: depth,
                    num_needle_k: args.niah_num_needle_k,
                    num_needle_v: args.niah_num_needle_v,
                    num_needle_q: args.niah_num_needle_q,
                    buffer: args.context_length_buffer,
                };
                let (prompt, expected) = generate_niah_context(&niah_cfg, &tokenizer, &essays)?;

                // Plain: no cache reuse, full prefill every time.
                let pb = ProgressBar::new(n as u64)
                    .with_style(plain_style.clone())
                    .with_message(label.clone());
                for i in 0..n {
                    let debug = args.debug && i == 0;
                    llm.reset_prefix_cache()?;
                    let msgs = vec![ChatMessage::system(system), ChatMessage::user(&prompt)];
                    let start = Instant::now();
                    let result = llm.chat(&msgs, Some(sampling.clone()))?;
                    let ms = start.elapsed().as_secs_f64() * 1000.0;
                    let acc = string_match_all(&result.outputs[0].text, &expected);
                    if debug {
                        eprintln!(
                            "  expected: {:?}\n  response: {}",
                            expected, result.outputs[0].text
                        );
                    }
                    plain_accs.push(acc);
                    plain_ms.push(ms);
                    pb.inc(1);
                }
                pb.finish();

                // Spans: populate cache once, then reuse.
                llm.reset_prefix_cache()?;
                let build_query = |q: &str| {
                    serde_json::json!({
                        "g": { "model": model_name, "max_tokens": 128, "temperature": 0.0,
                            "input": { "cross": [
                                { "system": system },
                                { "plus": [{ "user": prompt.clone() }] },
                                { "user": q },
                            ] }
                        }
                    })
                };

                // Populate (not timed).
                let populate = build_query("Warm up.");
                llm.execute_query(&populate.to_string(), Some(sampling.clone()), false, false)?;

                let query_str = if expected.len() == 1 {
                    "What is the special magic number mentioned in the text?".to_string()
                } else {
                    "What are all the special magic numbers mentioned in the text?".to_string()
                };

                let pb = ProgressBar::new(n as u64)
                    .with_style(spans_style.clone())
                    .with_message(label.clone());
                for _i in 0..n {
                    let query = build_query(&query_str);
                    let start = Instant::now();
                    let results = llm.execute_query(
                        &query.to_string(),
                        Some(sampling.clone()),
                        false,
                        false,
                    )?;
                    let ms = start.elapsed().as_secs_f64() * 1000.0;
                    let acc = string_match_all(&results[0].outputs[0].text, &expected);
                    span_accs.push(acc);
                    span_ms.push(ms);
                    pb.inc(1);
                }
                pb.finish();

                print_comparison(&plain_accs, &plain_ms, &span_accs, &span_ms);
            }
        }
    }

    // Variable Tracking task
    if tasks.contains(&"variable_tracking") {
        let sampling = SamplingParams {
            max_tokens: Some(30),
            temperature: 0.0,
            ..SamplingParams::default()
        };
        let system = "You are a helpful AI assistant.";

        for &ctx_len in &args.context_lengths {
            let label = format!("VT len={}", ctx_len);
            eprintln!("\n--- {label} ---");

            let n = args.num_samples;
            let mut plain_accs = Vec::with_capacity(n);
            let mut plain_ms = Vec::with_capacity(n);
            let mut span_accs = Vec::with_capacity(n);
            let mut span_ms = Vec::with_capacity(n);

            let (prompt, expected) = generate_vt_context(
                ctx_len,
                args.vt_num_chains,
                args.vt_num_hops,
                args.context_length_buffer,
                &tokenizer,
            )?;

            // Plain: no cache reuse, full prefill every time.
            let pb = ProgressBar::new(n as u64)
                .with_style(plain_style.clone())
                .with_message(label.clone());
            for i in 0..n {
                let debug = args.debug && i == 0;
                llm.reset_prefix_cache()?;
                let msgs = vec![ChatMessage::system(system), ChatMessage::user(&prompt)];
                let start = Instant::now();
                let result = llm.chat(&msgs, Some(sampling.clone()))?;
                let ms = start.elapsed().as_secs_f64() * 1000.0;
                let acc = string_match_all(&result.outputs[0].text, &expected);
                if debug {
                    eprintln!(
                        "  expected: {:?}\n  response: {}",
                        expected, result.outputs[0].text
                    );
                }
                plain_accs.push(acc);
                plain_ms.push(ms);
                pb.inc(1);
            }
            pb.finish();

            // Spans: populate cache once, then reuse.
            llm.reset_prefix_cache()?;
            let build_query = |q: &str| {
                serde_json::json!({
                    "g": { "model": model_name, "max_tokens": 30, "temperature": 0.0,
                        "input": { "cross": [
                            { "system": system },
                            { "plus": [{ "user": prompt.clone() }] },
                            { "user": q },
                        ] }
                    }
                })
            };

            // Populate (not timed).
            let initial_value = prompt.rsplit("the value ").next().unwrap_or("?");
            let initial_value = initial_value.split_whitespace().next().unwrap_or("?");
            let question = format!(
                "Find all variables that are assigned the value {initial_value} in the text above."
            );
            let populate = build_query("Warm up.");
            llm.execute_query(&populate.to_string(), Some(sampling.clone()), false, false)?;

            let pb = ProgressBar::new(n as u64)
                .with_style(spans_style.clone())
                .with_message(label.clone());
            for _i in 0..n {
                let query = build_query(&question);
                let start = Instant::now();
                let results =
                    llm.execute_query(&query.to_string(), Some(sampling.clone()), false, false)?;
                let ms = start.elapsed().as_secs_f64() * 1000.0;
                let acc = string_match_all(&results[0].outputs[0].text, &expected);
                span_accs.push(acc);
                span_ms.push(ms);
                pb.inc(1);
            }
            pb.finish();

            print_comparison(&plain_accs, &plain_ms, &span_accs, &span_ms);
        }
    }

    eprintln!("\n=== RULER Benchmark Complete ===\n");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn string_match_all_all_found() {
        let refs = vec!["alpha".into(), "beta".into()];
        assert!((string_match_all("alpha and beta are here", &refs) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn string_match_all_partial() {
        let refs = vec!["alpha".into(), "beta".into(), "gamma".into()];
        assert!((string_match_all("only alpha and gamma", &refs) - 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn string_match_all_none_found() {
        let refs = vec!["alpha".into(), "beta".into()];
        assert!(string_match_all("nothing here", &refs).abs() < f64::EPSILON);
    }
}
