// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! `scr bench startup` — cold and warm startup time benchmarking.
//!
//! Measures total startup time (model loading + KV cache allocation) for both
//! cold (no HF cache) and warm (cached weights) scenarios by repeatedly
//! constructing an [`LLM`] instance.
//!
//! SCOPE, because it is easy to reach for the wrong one: what is timed here is
//! `LLMBuilder::build()` *in-process* (see the `Instant::now()` below). No
//! process is exec'd, no cache is wiped, and no request is ever sent — so this
//! cannot produce a TTFT, and its "cold" is a fresh engine object rather than a
//! cold machine. For "how long from `exec` until the user sees a word", and for
//! any cross-framework comparison, use `scripts/startup_probe.py` instead; the
//! two are not comparable. See `docs/BENCHMARKING.md`.

use std::time::Instant;

use anyhow::Result;
use indicatif::{ProgressBar, ProgressStyle};
use scratchy_core_common::telemetry;
use scratchy_core_config::{CudaGraphConfig, CudaGraphMode};
use scratchy_serving_api::llm::{LLM, LLMBuilder};

use crate::args::BenchStartupArgs;

/// Build an [`LLM`] from startup bench args.
fn create_llm(args: &BenchStartupArgs) -> Result<LLM> {
    let model = args.resolved_model().map_err(|e| anyhow::anyhow!(e))?;
    let mut builder = LLMBuilder::new(&model)
        .device(&args.device)
        .dtype(&args.dtype)
        .gpu_memory_utilization(args.gpu_memory_utilization)
        .max_num_seqs(args.max_num_seqs)
        .block_size(args.block_size)
        .tensor_parallel_size(args.tensor_parallel_size)
        .enable_prefix_caching(!args.no_prefix_caching)
        .enforce_eager(args.enforce_eager);

    if let Some(n) = args.max_num_batched_tokens {
        builder = builder.max_num_batched_tokens(n);
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

/// Compute the p-th percentile of a slice. Returns 0 for empty input.
fn percentile(data: &[f64], p: f64) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((p / 100.0) * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

const PERCENTAGES: &[f64] = &[10.0, 25.0, 50.0, 75.0, 90.0, 99.0];

pub(crate) fn run_bench_startup(args: BenchStartupArgs) -> Result<()> {
    telemetry::init_tracing(&args.log_level);

    let model = args.resolved_model().map_err(|e| anyhow::anyhow!(e))?;
    eprintln!("vLLM Rust — startup benchmark");
    eprintln!("Model: {model}");
    eprintln!(
        "Cold iters: {}, Warmup iters: {}, Warm iters: {}",
        args.num_iters_cold, args.num_iters_warmup, args.num_iters_warm
    );

    let bar_style =
        ProgressStyle::with_template("{msg} {wide_bar:.cyan/blue} {pos}/{len} [{elapsed}<{eta}]")
            .unwrap();

    // --- Cold startup ---
    // In the Rust engine there is no torch.compile cache. "Cold" means we
    // simply time creation from scratch each iteration. The HF cache is
    // shared (we don't wipe it), but model weight loading + KV pool
    // allocation still runs each time.
    eprintln!("\nMeasuring cold startup time...");
    let mut cold_startup_times = Vec::with_capacity(args.num_iters_cold);

    let pb = ProgressBar::new(args.num_iters_cold as u64)
        .with_style(bar_style.clone())
        .with_message("Cold startup");
    for _ in 0..args.num_iters_cold {
        let start = Instant::now();
        let _llm = create_llm(&args)?;
        cold_startup_times.push(start.elapsed().as_secs_f64());
        pb.inc(1);
    }
    pb.finish();

    // --- Warmup for warm startup ---
    if args.num_iters_warmup > 0 {
        eprintln!("\nWarming up...");
        let pb = ProgressBar::new(args.num_iters_warmup as u64)
            .with_style(bar_style.clone())
            .with_message("Warmup");
        for _ in 0..args.num_iters_warmup {
            let _llm = create_llm(&args)?;
            pb.inc(1);
        }
        pb.finish();
    }

    // --- Warm startup ---
    eprintln!("\nMeasuring warm startup time...");
    let mut warm_startup_times = Vec::with_capacity(args.num_iters_warm);

    let pb = ProgressBar::new(args.num_iters_warm as u64)
        .with_style(bar_style)
        .with_message("Warm startup");
    for _ in 0..args.num_iters_warm {
        let start = Instant::now();
        let _llm = create_llm(&args)?;
        warm_startup_times.push(start.elapsed().as_secs_f64());
        pb.inc(1);
    }
    pb.finish();

    // --- Statistics ---
    let avg_cold = cold_startup_times.iter().sum::<f64>() / cold_startup_times.len() as f64;
    let avg_warm = warm_startup_times.iter().sum::<f64>() / warm_startup_times.len() as f64;

    println!();
    println!("============================================================");
    println!("STARTUP TIME BENCHMARK RESULTS");
    println!("============================================================");

    println!("\nCOLD STARTUP:");
    println!("Avg total startup time: {avg_cold:.2} seconds");
    println!("Startup time percentiles:");
    for &p in PERCENTAGES {
        println!(
            "  {p:.0}%: {:.2} seconds",
            percentile(&cold_startup_times, p)
        );
    }

    println!("\nWARM STARTUP:");
    println!("Avg total startup time: {avg_warm:.2} seconds");
    println!("Startup time percentiles:");
    for &p in PERCENTAGES {
        println!(
            "  {p:.0}%: {:.2} seconds",
            percentile(&warm_startup_times, p)
        );
    }

    println!("============================================================");

    // --- JSON output ---
    if let Some(ref path) = args.output_json {
        let cold_pcts: serde_json::Map<String, serde_json::Value> = PERCENTAGES
            .iter()
            .map(|&p| {
                (
                    format!("{p:.0}"),
                    serde_json::json!(percentile(&cold_startup_times, p)),
                )
            })
            .collect();
        let warm_pcts: serde_json::Map<String, serde_json::Value> = PERCENTAGES
            .iter()
            .map(|&p| {
                (
                    format!("{p:.0}"),
                    serde_json::json!(percentile(&warm_startup_times, p)),
                )
            })
            .collect();

        let results = serde_json::json!({
            "avg_cold_startup_time": avg_cold,
            "cold_startup_times": cold_startup_times,
            "cold_startup_percentiles": cold_pcts,
            "avg_warm_startup_time": avg_warm,
            "warm_startup_times": warm_startup_times,
            "warm_startup_percentiles": warm_pcts,
        });
        std::fs::write(path, serde_json::to_string_pretty(&results)?)?;
        eprintln!("Results written to {path}");
    }

    Ok(())
}
