// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! `scr bench throughput` — offline throughput benchmarking.
//!
//! Generates a batch of random-token prompts, runs them all through the LLM
//! engine, and reports requests/s and tokens/s.

use std::path::Path;
use std::time::Instant;

use anyhow::Result;
use scratchy_core_common::telemetry;
use scratchy_core_config::{CudaGraphConfig, CudaGraphMode};
use scratchy_serving_api::llm::{LLM, LLMBuilder, Prompt, SamplingParams};

use crate::args::BenchThroughputArgs;
use crate::datasets;

/// Build an [`LLM`] from throughput bench args.
fn create_llm(args: &BenchThroughputArgs, model: &str) -> Result<LLM> {
    let mut builder = LLMBuilder::new(model)
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

pub(crate) fn run_bench_throughput(args: BenchThroughputArgs) -> Result<()> {
    telemetry::init_tracing(&args.log_level);

    let model = args.resolved_model().map_err(|e| anyhow::anyhow!(e))?;

    eprintln!("vLLM Rust — throughput benchmark");
    eprintln!("Model: {model}");
    eprintln!(
        "num_prompts: {}, input_len: {}, output_len: {}",
        args.num_prompts, args.input_len, args.output_len
    );

    eprintln!("\nLoading model: {model}");
    let load_start = Instant::now();
    let mut llm = create_llm(&args, &model)?;
    eprintln!("Model loaded in {:.2}s", load_start.elapsed().as_secs_f64());

    let tokenizer = llm
        .tokenizer()
        .ok_or_else(|| anyhow::anyhow!("tokenizer required for throughput benchmark"))?;

    // Generate or load prompts + per-request output lengths.
    struct PromptEntry {
        prompt: Prompt,
        output_len: usize,
    }

    let entries: Vec<PromptEntry> = match args.dataset_name.as_str() {
        "sharegpt" => {
            let dataset_path = args.dataset_path.as_ref().ok_or_else(|| {
                anyhow::anyhow!("--dataset-path is required for sharegpt dataset")
            })?;
            eprintln!("Loading ShareGPT dataset from {dataset_path}...");
            let samples = datasets::load_sharegpt(
                Path::new(dataset_path),
                tokenizer.inner(),
                args.num_prompts,
                Some(llm.max_model_len()),
                args.seed,
            )?;
            eprintln!("Loaded {} samples from ShareGPT dataset", samples.len());
            samples
                .into_iter()
                .map(|s| PromptEntry {
                    prompt: Prompt::Text(s.prompt),
                    output_len: s.expected_output_len,
                })
                .collect()
        }
        _ => {
            let required_len = args.input_len + args.output_len;
            anyhow::ensure!(
                llm.max_model_len() >= required_len,
                "max_model_len ({}) must be >= input_len + output_len ({} + {} = {})",
                llm.max_model_len(),
                args.input_len,
                args.output_len,
                required_len,
            );

            eprintln!(
                "Generating {} random prompts (matching Python RandomDataset)...",
                args.num_prompts
            );
            let samples = datasets::generate_random(
                tokenizer.inner(),
                args.num_prompts,
                args.input_len,
                args.output_len,
                args.random_range_ratio,
                args.random_prefix_len,
                args.seed,
            )?;

            samples
                .into_iter()
                .map(|s| PromptEntry {
                    prompt: Prompt::Text(s.prompt),
                    output_len: s.expected_output_len,
                })
                .collect()
        }
    };

    // For sharegpt, each request may have a different output length — use per-request params.
    // For random, all share the same output_len.
    let uniform_output_len = entries
        .iter()
        .all(|e| e.output_len == entries[0].output_len);
    let prompts: Vec<Prompt> = entries.iter().map(|e| e.prompt.clone()).collect();
    let default_output_len = entries[0].output_len;

    let sampling_params = SamplingParams {
        temperature: 1.0,
        top_p: 1.0,
        ignore_eos: true,
        max_tokens: if uniform_output_len {
            Some(default_output_len as u32)
        } else {
            Some(args.output_len as u32)
        },
        detokenize: !args.disable_detokenize,
        ..SamplingParams::default()
    };

    let start = Instant::now();
    // Feed all prompts at once — the engine's scheduler handles batching.
    let outputs = llm.generate_with_tqdm(&prompts, Some(sampling_params))?;
    let elapsed = start.elapsed().as_secs_f64();

    // Count actual tokens from outputs.
    let total_prompt_tokens: usize = outputs.iter().map(|o| o.prompt_token_ids.len()).sum();
    let total_output_tokens: usize = outputs
        .iter()
        .flat_map(|o| &o.outputs)
        .map(|c| c.token_ids.len())
        .sum();
    let total_tokens = total_prompt_tokens + total_output_tokens;

    println!();
    println!(
        "Throughput: {:.2} requests/s, {:.2} total tokens/s, {:.2} output tokens/s",
        outputs.len() as f64 / elapsed,
        total_tokens as f64 / elapsed,
        total_output_tokens as f64 / elapsed,
    );
    println!("Total num prompt tokens:  {total_prompt_tokens}");
    println!("Total num output tokens:  {total_output_tokens}");
    println!("Elapsed time: {elapsed:.2}s");

    if let Some(ref path) = args.output_json {
        let results = serde_json::json!({
            "elapsed_time": elapsed,
            "num_requests": outputs.len(),
            "total_num_tokens": total_tokens,
            "total_prompt_tokens": total_prompt_tokens,
            "total_output_tokens": total_output_tokens,
            "requests_per_second": outputs.len() as f64 / elapsed,
            "tokens_per_second": total_tokens as f64 / elapsed,
            "output_tokens_per_second": total_output_tokens as f64 / elapsed,
        });
        std::fs::write(path, serde_json::to_string_pretty(&results)?)?;
        eprintln!("Results written to {path}");
    }

    Ok(())
}
