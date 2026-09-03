// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! `scr bench latency` — end-to-end latency benchmarking.

use std::time::Instant;

use anyhow::Result;
use indicatif::{ProgressBar, ProgressStyle};
use scratchy_core_common::telemetry;
use scratchy_core_config::{CudaGraphConfig, CudaGraphMode};
use scratchy_serving_api::llm::{LLM, LLMBuilder, Prompt, SamplingParams};

use crate::args::BenchLatencyArgs;

// ---------------------------------------------------------------------------
// LLM construction
// ---------------------------------------------------------------------------

/// Build an [`LLM`] from bench args.
fn create_llm(args: &BenchLatencyArgs, model: &str) -> Result<LLM> {
    let mut builder = LLMBuilder::new(model)
        .device(&args.device)
        .dtype(&args.dtype)
        .gpu_memory_utilization(args.gpu_memory_utilization)
        .max_num_seqs(args.max_num_seqs)
        .block_size(args.block_size)
        .tensor_parallel_size(args.tensor_parallel_size)
        .enable_prefix_caching(!args.no_prefix_caching)
        .enforce_eager(args.enforce_eager);

    // Honor the user's --max-num-batched-tokens, otherwise leave the
    // SchedulerConfig default (2048) and let chunked prefill amortize
    // the prefill across iterations. Python's `vllm bench latency` does
    // the same — this is the apples-to-apples comparison. The previous
    // auto-bump to `max(bs*il, 8192)` claimed to mirror Python but does
    // not (verified against vllm 0.19.1 vllm/benchmarks/latency.py: it
    // never touches max_num_batched_tokens) and forced a single
    // un-chunked forward at M=bs*il, OOMing at 14B bs=32 il=2048
    // (3.6 GiB MLP gate_up output past the budget) and regressing 14B
    // bs=32 il=1024 by ~3x because the scheduler couldn't chunk.
    if let Some(mnbt) = args.max_num_batched_tokens {
        builder = builder.max_num_batched_tokens(mnbt);
    }
    if let Some(len) = args.max_model_len {
        builder = builder.max_model_len(len);
    }
    if let Some(ref token) = args.hf_token {
        builder = builder.hf_token(token);
    }
    if let Some(ref gguf) = args.gguf_file {
        builder = builder.gguf_file(gguf);
    }

    // Wire CUDA graph config unless --enforce-eager is set.
    if !args.enforce_eager {
        let sizes = CudaGraphConfig::parse_sizes(&args.cuda_graph_sizes);
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
// Result type
// ---------------------------------------------------------------------------

struct BenchResult {
    model: String,
    batch_size: usize,
    latencies: Vec<f64>,
}

impl BenchResult {
    fn avg_latency(&self) -> f64 {
        self.latencies.iter().sum::<f64>() / self.latencies.len() as f64
    }

    fn percentile_value(&self, p: f64) -> f64 {
        percentile(&self.latencies, p)
    }
}

// ---------------------------------------------------------------------------
// Bench runner
// ---------------------------------------------------------------------------

pub(crate) fn run_bench_latency(args: BenchLatencyArgs) -> Result<()> {
    telemetry::init_tracing(&args.log_level);

    let models = args.resolved_models().map_err(|e| anyhow::anyhow!(e))?;
    let batch_sizes = &args.batch_sizes;
    let pcts = &args.percentiles;

    eprintln!("vLLM Rust — latency benchmark");
    eprintln!(
        "Models: {:?}, batch_sizes: {:?}, percentiles: {:?}",
        models, batch_sizes, pcts
    );
    eprintln!(
        "Iters: {}, input_len: {}, output_len: {}, warmup: {}",
        args.num_iters, args.input_len, args.output_len, args.num_iters_warmup
    );

    // tqdm-style progress key helpers.
    fn fmt_duration(d: std::time::Duration, w: &mut dyn std::fmt::Write) {
        let secs = d.as_secs();
        if secs >= 3600 {
            write!(
                w,
                "{}:{:02}:{:02}",
                secs / 3600,
                (secs % 3600) / 60,
                secs % 60
            )
            .unwrap();
        } else {
            write!(w, "{:02}:{:02}", secs / 60, secs % 60).unwrap();
        }
    }
    fn tqdm_keys(style: ProgressStyle) -> ProgressStyle {
        style
            .with_key(
                "my_elapsed",
                |state: &indicatif::ProgressState, w: &mut dyn std::fmt::Write| {
                    fmt_duration(state.elapsed(), w);
                },
            )
            .with_key(
                "my_eta",
                |state: &indicatif::ProgressState, w: &mut dyn std::fmt::Write| {
                    fmt_duration(state.eta(), w);
                },
            )
            .with_key("per_sec", crate::fmt_tqdm_rate)
    }

    let warmup_style = tqdm_keys(
        ProgressStyle::with_template(
            "{msg} {wide_bar:.yellow/yellow} {pos}/{len} [{my_elapsed}<{my_eta}, {per_sec}]",
        )
        .unwrap(),
    );
    let bench_style = tqdm_keys(
        ProgressStyle::with_template(
            "{msg} {wide_bar:.cyan/blue} {pos}/{len} [{my_elapsed}<{my_eta}, {per_sec}]",
        )
        .unwrap(),
    );

    // Pre-compute max label width so all progress bars align.
    let max_msg_len = models
        .iter()
        .flat_map(|m| {
            let name = short_model_name(m);
            batch_sizes.iter().flat_map(move |&bs| {
                let bs_label = if batch_sizes.len() > 1 {
                    format!(" bs={bs}")
                } else {
                    String::new()
                };
                [
                    format!("{name}{bs_label} warmup"),
                    format!("{name}{bs_label} bench"),
                ]
            })
        })
        .map(|s| s.len())
        .max()
        .unwrap_or(0);

    let sampling_params = SamplingParams {
        temperature: args.temperature,
        top_p: args.top_p,
        top_k: args.top_k,
        ignore_eos: true,
        max_tokens: Some(args.output_len as u32),
        detokenize: !args.disable_detokenize,
        ..SamplingParams::default()
    };

    let mut results: Vec<BenchResult> = Vec::new();

    for model in &models {
        eprintln!("\nLoading model: {model}");
        let load_start = Instant::now();
        let mut llm = create_llm(&args, model)?;
        eprintln!("Model loaded in {:.2}s", load_start.elapsed().as_secs_f64());

        let required_len = args.input_len + args.output_len;
        anyhow::ensure!(
            llm.max_model_len() >= required_len,
            "max_model_len ({}) must be >= input_len + output_len ({} + {} = {})",
            llm.max_model_len(),
            args.input_len,
            args.output_len,
            required_len,
        );

        let short_name = short_model_name(model);

        for &bs in batch_sizes {
            let dummy_prompts: Vec<Prompt> = (0..bs)
                .map(|i| {
                    Prompt::TokenIds(
                        (0..args.input_len)
                            .map(|j| ((i * 997 + j * 31 + 42) % 10000) as u32)
                            .collect(),
                    )
                })
                .collect();

            let bs_label = if batch_sizes.len() > 1 {
                format!(" bs={bs}")
            } else {
                String::new()
            };

            // Warmup.
            if args.num_iters_warmup > 0 {
                let msg = format!(
                    "{:<width$}",
                    format!("{short_name}{bs_label} warmup"),
                    width = max_msg_len
                );
                let pb = ProgressBar::new(args.num_iters_warmup as u64)
                    .with_style(warmup_style.clone())
                    .with_message(msg);
                for _ in 0..args.num_iters_warmup {
                    llm.generate(&dummy_prompts, Some(sampling_params.clone()))?;
                    pb.inc(1);
                }
                pb.finish();
            }

            // Timed runs.
            let msg = format!(
                "{:<width$}",
                format!("{short_name}{bs_label} bench"),
                width = max_msg_len
            );
            let pb = ProgressBar::new(args.num_iters as u64)
                .with_style(bench_style.clone())
                .with_message(msg);
            let mut latencies = Vec::with_capacity(args.num_iters);
            for _ in 0..args.num_iters {
                let start = Instant::now();
                llm.generate(&dummy_prompts, Some(sampling_params.clone()))?;
                latencies.push(start.elapsed().as_secs_f64());
                pb.inc(1);
            }
            pb.finish();

            results.push(BenchResult {
                model: model.clone(),
                batch_size: bs,
                latencies,
            });
        }
    }

    // Print results.
    println!();
    if models.len() == 1 && batch_sizes.len() == 1 {
        print_single_result(&results[0], &args);
    } else {
        print_table(&results, &models, batch_sizes, pcts);
    }

    // JSON output.
    if let Some(ref path) = args.output_json {
        let output = if results.len() == 1 {
            // Single model + single batch size: flat format matching Python's output.
            let r = &results[0];
            let pct_map: serde_json::Map<String, serde_json::Value> =
                [10.0, 25.0, 50.0, 75.0, 90.0, 99.0]
                    .iter()
                    .chain(pcts.iter())
                    .map(|&p| (format!("{p:.0}"), serde_json::json!(r.percentile_value(p))))
                    .collect();
            serde_json::json!({
                "avg_latency": r.avg_latency(),
                "latencies": r.latencies,
                "percentiles": pct_map,
            })
        } else {
            // Multi-model or multi-batch: wrapped format (Rust extension).
            let json_results: Vec<serde_json::Value> = results
                .iter()
                .map(|r| {
                    let pct_map: serde_json::Map<String, serde_json::Value> =
                        [10.0, 25.0, 50.0, 75.0, 90.0, 99.0]
                            .iter()
                            .chain(pcts.iter())
                            .map(|&p| (format!("{p:.0}"), serde_json::json!(r.percentile_value(p))))
                            .collect();
                    serde_json::json!({
                        "model": r.model,
                        "batch_size": r.batch_size,
                        "avg_latency": r.avg_latency(),
                        "percentiles": pct_map,
                        "latencies": r.latencies,
                    })
                })
                .collect();
            serde_json::json!({ "results": json_results })
        };
        std::fs::write(path, serde_json::to_string_pretty(&output)?)?;
        eprintln!("Results written to {path}");
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Output formatting
// ---------------------------------------------------------------------------

/// Single-result output (backward compatible with original format).
fn print_single_result(r: &BenchResult, args: &BenchLatencyArgs) {
    let total_tokens = args.num_iters as u64 * r.batch_size as u64 * args.output_len as u64;
    let total_elapsed: f64 = r.latencies.iter().sum();
    let throughput = total_tokens as f64 / total_elapsed;

    println!("=== Benchmark Results ===");
    println!("Iterations:      {}", args.num_iters);
    println!("Batch size:      {}", r.batch_size);
    println!("Input len:       {} tokens", args.input_len);
    println!("Output len:      {}", args.output_len);
    println!("Total tokens:    {total_tokens}");
    println!("Throughput:      {throughput:.1} tokens/s");
    println!();
    println!("Avg latency: {} seconds", r.avg_latency());
    for &p in &[10.0, 25.0, 50.0, 75.0, 90.0, 99.0] {
        println!(
            "{p:.0}% percentile latency: {} seconds",
            r.percentile_value(p)
        );
    }
}

/// Standard percentile set used when only one dimension varies.
const ALL_PERCENTILES: &[f64] = &[10.0, 25.0, 50.0, 75.0, 90.0, 99.0];

/// Table output for multi-model and/or multi-batch-size runs.
fn print_table(results: &[BenchResult], models: &[String], batch_sizes: &[usize], pcts: &[f64]) {
    // When only one dimension is multi-valued, show all standard percentiles
    // as separate columns. Only limit to user-specified percentiles when both
    // models and batch sizes are multi-dimensional.
    let multi_model = models.len() > 1;
    let multi_bs = batch_sizes.len() > 1;
    let full_pcts = if multi_model && multi_bs {
        pcts
    } else {
        ALL_PERCENTILES
    };

    // Format a cell: comma-separated percentile values (2D matrix only).
    let fmt_cell = |r: &BenchResult| -> String {
        pcts.iter()
            .map(|&p| format!("{:.4}", r.percentile_value(p)))
            .collect::<Vec<_>>()
            .join(", ")
    };

    let pct_label = pcts
        .iter()
        .map(|p| format!("p{p:.0}"))
        .collect::<Vec<_>>()
        .join("/");

    if models.len() == 1 && batch_sizes.len() > 1 {
        // Rows = batch sizes, columns = percentiles.
        let col_width = 10_usize;
        let row_label_width = 12_usize;

        print!("{:<width$}", "Batch size", width = row_label_width);
        for p in full_pcts {
            print!("  {:>width$}", format!("p{p:.0}"), width = col_width);
        }
        println!();
        print!("{:<width$}", "----------", width = row_label_width);
        for _ in full_pcts {
            print!("  {:>width$}", "----------", width = col_width);
        }
        println!();

        for r in results {
            print!("{:<width$}", r.batch_size, width = row_label_width);
            for &p in full_pcts {
                print!("  {:>width$.4}", r.percentile_value(p), width = col_width);
            }
            println!();
        }
    } else if models.len() > 1 && batch_sizes.len() == 1 {
        // Rows = models, columns = percentiles.
        let col_width = 10_usize;

        print!("{:<30}", "Model");
        for p in full_pcts {
            print!("  {:>width$}", format!("p{p:.0}"), width = col_width);
        }
        println!();
        print!("{:<30}", "-----");
        for _ in full_pcts {
            print!("  {:>width$}", "----------", width = col_width);
        }
        println!();

        for r in results {
            print!("{:<30}", short_model_name(&r.model));
            for &p in full_pcts {
                print!("  {:>width$.4}", r.percentile_value(p), width = col_width);
            }
            println!();
        }
    } else {
        // N models x M batch sizes. Columns = batch sizes, cells = percentiles.
        let header_cells: Vec<String> = batch_sizes.iter().map(|b| format!("bs={b}")).collect();

        // Compute column widths from data.
        let data_cells: Vec<Vec<String>> = models
            .iter()
            .map(|m| {
                batch_sizes
                    .iter()
                    .map(|&bs| {
                        results
                            .iter()
                            .find(|r| r.model == *m && r.batch_size == bs)
                            .map(&fmt_cell)
                            .unwrap_or_default()
                    })
                    .collect()
            })
            .collect();

        let col_widths: Vec<usize> = (0..batch_sizes.len())
            .map(|ci| {
                let max_data = data_cells
                    .iter()
                    .map(|row| row[ci].len())
                    .max()
                    .unwrap_or(0);
                max_data.max(header_cells[ci].len()).max(8)
            })
            .collect();

        print!("{:<30}", format!("Model ({pct_label})"));
        for (i, h) in header_cells.iter().enumerate() {
            print!("  {:>width$}", h, width = col_widths[i]);
        }
        println!();
        print!("{:<30}", "-----");
        for w in &col_widths {
            print!("  {:>width$}", "----------", width = *w);
        }
        println!();

        for (mi, m) in models.iter().enumerate() {
            print!("{:<30}", short_model_name(m));
            for (ci, cell) in data_cells[mi].iter().enumerate() {
                print!("  {:>width$}", cell, width = col_widths[ci]);
            }
            println!();
        }
    }
}

/// Extract a short display name from a model path/ID.
fn short_model_name(model: &str) -> String {
    model.rsplit('/').next().unwrap_or(model).to_string()
}

/// Compute the p-th percentile using linear interpolation, matching
/// `numpy.percentile` (method='linear', the default).
fn percentile(data: &[f64], p: f64) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    // numpy linear interpolation: virtual index into [0, n-1]
    let idx = (p / 100.0) * (n - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = lo + 1;
    if hi >= n {
        sorted[n - 1]
    } else {
        let frac = idx - lo as f64;
        sorted[lo] + frac * (sorted[hi] - sorted[lo])
    }
}
