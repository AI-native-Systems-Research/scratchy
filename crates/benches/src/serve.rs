// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! `scr bench serve` — online serving benchmark.
//!
//! Sends concurrent HTTP requests to a running vLLM server (OpenAI-compatible
//! API) and measures per-request latency metrics: TTFT, TPOT, ITL, and
//! end-to-end latency (E2EL).

use std::io::Read;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use anyhow::Result;
use indicatif::{ProgressBar, ProgressStyle};
use rand::Rng;
use rand_distr::Gamma;

use crate::args::BenchServeArgs;
use crate::datasets;
use crate::http::{Agent, Semaphore};

/// Per-request result.
struct RequestResult {
    /// Time to first token (seconds).
    ttft: f64,
    /// Inter-token latencies (seconds) — one per token after the first.
    itl: Vec<f64>,
    /// Total end-to-end latency (seconds).
    e2el: f64,
    /// Number of output tokens generated.
    output_tokens: usize,
    /// Timestamp (seconds since benchmark start) when request was sent.
    start_time: f64,
    success: bool,
}

/// Fetch the first model name from the server's /v1/models endpoint.
fn get_model_from_server(agent: &Agent, base_url: &str) -> Result<String> {
    let url = format!("{base_url}/v1/models");
    let body = agent.get(&url).call()?.into_body().read_to_string()?;
    let resp: serde_json::Value = serde_json::from_str(&body)?;
    let model = resp["data"][0]["id"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("No models found on server at {base_url}"))?;
    Ok(model.to_string())
}

/// Send a single streaming completions request and measure timing.
#[allow(clippy::too_many_arguments)]
fn send_request(
    agent: &Agent,
    api_url: &str,
    model: &str,
    prompt: &str,
    output_len: usize,
    ignore_eos: bool,
    temperature: Option<f64>,
    top_p: Option<f64>,
    top_k: Option<i32>,
    api_key: Option<&str>,
    request_id: &str,
    bench_start: Instant,
) -> RequestResult {
    // NOTE: `logprobs` omitted (rather than sent as null) so this bench
    // works against servers with strict OpenAI-schema validation
    // (mlx_lm.server, in particular, rejects `logprobs: null` because
    // it only accepts a bool). vLLM tolerates either.
    let mut body = serde_json::json!({
        "model": model,
        "prompt": prompt,
        "max_tokens": output_len,
        "stream": true,
        "stream_options": {"include_usage": true},
        "repetition_penalty": 1.0,
    });

    if ignore_eos {
        body["ignore_eos"] = serde_json::json!(true);
    }
    if let Some(t) = temperature {
        body["temperature"] = serde_json::json!(t);
    }
    if let Some(p) = top_p {
        body["top_p"] = serde_json::json!(p);
    }
    if let Some(k) = top_k {
        body["top_k"] = serde_json::json!(k);
    }

    let request_start = Instant::now();
    let start_time = request_start.duration_since(bench_start).as_secs_f64();
    let mut first_token_time: Option<Instant> = None;
    let mut last_token_time = request_start;
    let mut itl = Vec::new();
    let mut output_tokens = 0usize;

    let mut req = agent
        .post(api_url)
        .header("content-type", "application/json")
        .header("x-request-id", request_id);
    if let Some(key) = api_key {
        req = req.header("authorization", format!("Bearer {key}"));
    }

    let resp = match req.send(body.to_string().as_str()) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Request failed: {e}");
            return RequestResult {
                ttft: 0.0,
                itl: vec![],
                e2el: request_start.elapsed().as_secs_f64(),
                output_tokens: 0,
                start_time,
                success: false,
            };
        }
    };

    // The agent is built with `http_status_as_error(false)`, so a 4xx/5xx
    // arrives here as a normal response and the body is still readable.
    if !resp.status().is_success() {
        let status = resp.status();
        let body_text = resp.into_body().read_to_string().unwrap_or_default();
        eprintln!("Request failed with status {status}: {body_text}");
        return RequestResult {
            ttft: 0.0,
            itl: vec![],
            e2el: request_start.elapsed().as_secs_f64(),
            output_tokens: 0,
            start_time,
            success: false,
        };
    }

    // Parse SSE stream. Reading straight off the body reader keeps the
    // timestamps below as close to arrival as we can get them.
    let mut reader = resp.into_body().into_reader();
    let mut chunk = [0u8; 8192];
    let mut buf = String::new();
    let mut usage_completion_tokens: Option<usize> = None;

    loop {
        let n = match reader.read(&mut chunk) {
            Ok(0) => break,
            Ok(n) => n,
            Err(_) => break,
        };
        buf.push_str(&String::from_utf8_lossy(&chunk[..n]));

        // Process complete SSE lines.
        while let Some(pos) = buf.find("\n\n") {
            let event = buf[..pos].to_string();
            buf = buf[pos + 2..].to_string();

            for line in event.lines() {
                let line = line.trim();
                if line == "data: [DONE]" {
                    continue;
                }
                if let Some(data) = line.strip_prefix("data: ")
                    && let Ok(parsed) = serde_json::from_str::<serde_json::Value>(data)
                {
                    // Match Python's TTFT/ITL logic exactly:
                    // - First chunk with `choices` → TTFT
                    // - Every subsequent chunk with `choices` → ITL
                    // - `most_recent_timestamp` updated on every choices chunk
                    if parsed
                        .get("choices")
                        .is_some_and(|c| c.as_array().is_some_and(|a| !a.is_empty()))
                    {
                        let now = Instant::now();
                        if first_token_time.is_none() {
                            first_token_time = Some(now);
                        } else {
                            itl.push(now.duration_since(last_token_time).as_secs_f64());
                        }
                        last_token_time = now;
                    } else if let Some(ct) = parsed["usage"]["completion_tokens"].as_u64() {
                        usage_completion_tokens = Some(ct as usize);
                    }
                }
            }
        }
    }

    // Prefer server-reported token count.
    if let Some(ct) = usage_completion_tokens {
        output_tokens = ct;
    }

    // Python: output.latency = most_recent_timestamp - st (last choices chunk time).
    let e2el = last_token_time.duration_since(request_start).as_secs_f64();
    let ttft = first_token_time
        .map(|t| t.duration_since(request_start).as_secs_f64())
        .unwrap_or(e2el);

    RequestResult {
        ttft,
        itl,
        e2el,
        output_tokens,
        start_time,
        success: first_token_time.is_some(),
    }
}

/// Compute percentile of a sorted slice using linear interpolation
/// matching numpy.percentile(method='linear').
fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
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

/// Compute standard deviation.
fn std_dev(data: &[f64]) -> f64 {
    if data.len() < 2 {
        return 0.0;
    }
    let mean = data.iter().sum::<f64>() / data.len() as f64;
    let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;
    variance.sqrt()
}

/// Generate inter-request delays using a gamma distribution, matching Python's
/// `get_request_func()` methodology.
///
/// - `burstiness == 1.0`: Gamma(shape=1, scale=1/rate) = exponential (Poisson process)
/// - `burstiness < 1.0`: more bursty (clustered arrivals)
/// - `burstiness > 1.0`: more uniform (evenly spaced)
/// - `burstiness == inf`: constant delay = 1/rate
///
/// After generating raw delays, accumulates them cumulatively and normalizes
/// so that the total time span matches `num_prompts / rate`.
fn generate_request_delays(num_prompts: usize, rate: f64, burstiness: f64, seed: u64) -> Vec<f64> {
    if rate.is_infinite() || num_prompts <= 1 {
        return vec![0.0; num_prompts];
    }

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    if burstiness.is_infinite() {
        // Constant delay.
        let delay = 1.0 / rate;
        let mut cumulative = vec![0.0];
        for i in 1..num_prompts {
            cumulative.push(delay * i as f64);
        }
        return cumulative;
    }

    // Gamma distribution: shape = burstiness, scale = 1 / (rate * burstiness)
    // When burstiness=1, this reduces to Exponential(rate).
    let shape = burstiness;
    let scale = 1.0 / (rate * burstiness);
    let gamma = Gamma::new(shape, scale).expect("Invalid gamma distribution parameters");

    // Generate raw delays and accumulate.
    let mut cumulative = Vec::with_capacity(num_prompts);
    cumulative.push(0.0);
    for _ in 1..num_prompts {
        let delay: f64 = rng.sample(gamma);
        cumulative.push(cumulative.last().unwrap() + delay);
    }

    // Normalize cumulative delays to match target total time.
    // Python: intervals *= target / sum(intervals), then cumsum.
    let actual_total = *cumulative.last().unwrap();
    let target_total = (num_prompts - 1) as f64 / rate;
    if actual_total > 0.0 {
        let scale_factor = target_total / actual_total;
        for t in &mut cumulative {
            *t *= scale_factor;
        }
    }

    cumulative
}

use rand::SeedableRng;

/// The benchmark is blocking end to end (see [`crate::http`]), so it runs on a
/// blocking thread rather than holding a runtime worker for its whole duration.
pub(crate) async fn run_bench_serve(args: BenchServeArgs) -> Result<()> {
    tokio::task::spawn_blocking(move || run_bench_serve_blocking(args)).await?
}

fn run_bench_serve_blocking(args: BenchServeArgs) -> Result<()> {
    let agent = crate::http::agent(args.insecure);

    // Resolve model name.
    let model = match args.model_tag.as_ref().or(args.model.as_ref()) {
        Some(m) => m.clone(),
        None => {
            eprintln!("No --model specified, fetching from server...");
            get_model_from_server(&agent, &args.base_url)?
        }
    };

    let api_url = format!("{}{}", args.base_url, args.endpoint);
    eprintln!("vLLM Rust — serving benchmark");
    eprintln!("Model: {model}");
    eprintln!("API URL: {api_url}");
    eprintln!(
        "num_prompts: {}, input_len: {}, output_len: {}, request_rate: {}, burstiness: {}",
        args.num_prompts,
        args.input_len,
        args.output_len,
        if args.request_rate.is_infinite() {
            "inf".to_string()
        } else {
            format!("{:.1}", args.request_rate)
        },
        if args.burstiness.is_infinite() {
            "inf".to_string()
        } else {
            format!("{:.2}", args.burstiness)
        }
    );

    // Load tokenizer from HuggingFace hub (matches Python's get_tokenizer).
    // --tokenizer overrides the model name for tokenizer resolution, useful
    // when the model name isn't a valid HF repo (e.g. Ollama "llama3.2:3b").
    let tokenizer_id = args.tokenizer.as_deref().unwrap_or(&model);
    eprintln!("Loading tokenizer for {tokenizer_id}...");
    let tokenizer = {
        use scratchy_serving_engine::worker_factory::resolve_model_path;

        // Use the same model resolution logic as scr serve
        let model_dir = resolve_model_path(tokenizer_id, None, None, None)
            .map_err(|e| anyhow::anyhow!("Failed to resolve model path: {e}"))?;

        let tokenizer_path = model_dir.join("tokenizer.json");
        tokenizers::Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {e}"))?
    };

    // Generate or load prompts.
    struct PromptEntry {
        text: String,
        output_len: usize,
    }

    let prompt_entries: Vec<PromptEntry> = match args.dataset_name.as_str() {
        "sharegpt" => {
            let dataset_path = args.dataset_path.as_ref().ok_or_else(|| {
                anyhow::anyhow!("--dataset-path is required for sharegpt dataset")
            })?;
            eprintln!("Loading ShareGPT dataset from {dataset_path}...");
            let samples = datasets::load_sharegpt(
                Path::new(dataset_path),
                &tokenizer,
                args.num_prompts,
                None,
                args.seed,
            )?;
            eprintln!("Loaded {} samples from ShareGPT dataset", samples.len());
            samples
                .into_iter()
                .map(|s| PromptEntry {
                    text: s.prompt,
                    output_len: s.expected_output_len,
                })
                .collect()
        }
        _ => {
            eprintln!(
                "Generating {} random prompts (matching Python RandomDataset)...",
                args.num_prompts
            );
            let samples = datasets::generate_random(
                &tokenizer,
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
                    text: s.prompt,
                    output_len: s.expected_output_len,
                })
                .collect()
        }
    };

    let num_prompts = prompt_entries.len();

    // Pre-flight check: send one request to verify connectivity.
    eprintln!("Sending pre-flight request to verify connectivity...");
    {
        let result = send_request(
            &agent,
            &api_url,
            &model,
            &prompt_entries[0].text,
            prompt_entries[0].output_len.min(args.output_len),
            args.ignore_eos,
            args.temperature,
            args.top_p,
            args.top_k,
            args.api_key.as_deref(),
            "preflight",
            Instant::now(),
        );
        if !result.success {
            anyhow::bail!("Pre-flight request failed. Check server connectivity and model name.");
        }
        eprintln!("Pre-flight request succeeded.");
    }

    // Warmup requests.
    if args.num_warmups > 0 {
        eprintln!("Sending {} warmup request(s)...", args.num_warmups);
        let warmup_pb = ProgressBar::new(args.num_warmups as u64);
        warmup_pb.set_style(
            ProgressStyle::with_template(
                "Warmup {wide_bar:.yellow/blue} {pos}/{len} [{elapsed}<{eta}]",
            )
            .unwrap(),
        );
        for i in 0..args.num_warmups {
            let idx = i % num_prompts;
            send_request(
                &agent,
                &api_url,
                &model,
                &prompt_entries[idx].text,
                prompt_entries[idx].output_len.min(args.output_len),
                args.ignore_eos,
                args.temperature,
                args.top_p,
                args.top_k,
                args.api_key.as_deref(),
                &format!("warmup-{i}"),
                Instant::now(),
            );
            warmup_pb.inc(1);
        }
        warmup_pb.finish_and_clear();
        eprintln!("Warmup complete.");
    }

    // Progress bar.
    let pb = if !args.disable_tqdm {
        let pb = ProgressBar::new(num_prompts as u64);
        pb.set_style(
            ProgressStyle::with_template(
                "Benchmarking {wide_bar:.cyan/blue} {pos}/{len} [{elapsed}<{eta}, {per_sec}]",
            )
            .unwrap()
            .with_key("per_sec", crate::fmt_tqdm_rate),
        );
        Some(pb)
    } else {
        None
    };

    let completed = Arc::new(AtomicUsize::new(0));
    let semaphore = args.max_concurrency.map(|n| Arc::new(Semaphore::new(n)));

    // Generate request schedule using gamma distribution delays.
    let cumulative_delays = generate_request_delays(
        num_prompts,
        args.request_rate,
        args.burstiness,
        args.seed.wrapping_add(42),
    );

    let benchmark_start = Instant::now();
    let mut handles = Vec::with_capacity(num_prompts);

    for (i, entry) in prompt_entries.into_iter().enumerate() {
        // Wait until the scheduled time for this request.
        let target_time = Duration::from_secs_f64(cumulative_delays[i]);
        let elapsed = benchmark_start.elapsed();
        if target_time > elapsed {
            std::thread::sleep(target_time - elapsed);
        }

        let agent = agent.clone();
        let api_url = api_url.clone();
        let model = model.clone();
        let completed = completed.clone();
        let pb = pb.clone();
        let sem = semaphore.clone();
        let request_id = format!("bench-{i}");
        let ignore_eos = args.ignore_eos;
        let temperature = args.temperature;
        let top_p = args.top_p;
        let top_k = args.top_k;
        let api_key = args.api_key.clone();
        let output_len = entry.output_len;
        let bench_start = benchmark_start;

        // A thread per in-flight request. `--max-concurrency` bounds how many
        // exist at once, and the permit is acquired HERE rather than inside
        // the thread so the pacing loop blocks instead of running ahead and
        // spawning threads that would only queue — the same back-pressure the
        // async version got from awaiting the permit before spawning.
        let permit = sem.as_ref().map(|s| s.acquire_owned());
        handles.push(std::thread::spawn(move || {
            // Held for the request's lifetime; returned on drop, including on
            // unwind, so a failed request cannot shrink the concurrency bound.
            let _permit = permit;
            let result = send_request(
                &agent,
                &api_url,
                &model,
                &entry.text,
                output_len,
                ignore_eos,
                temperature,
                top_p,
                top_k,
                api_key.as_deref(),
                &request_id,
                bench_start,
            );
            completed.fetch_add(1, Ordering::Relaxed);
            if let Some(ref pb) = pb {
                pb.inc(1);
            }
            result
        }));
    }

    // Collect results.
    let mut results = Vec::with_capacity(handles.len());
    for handle in handles {
        results.push(
            handle
                .join()
                .map_err(|_| anyhow::anyhow!("benchmark request thread panicked"))?,
        );
    }

    let total_time = benchmark_start.elapsed().as_secs_f64();
    if let Some(pb) = pb {
        pb.finish_and_clear();
    }

    // Compute metrics.
    let successful: Vec<&RequestResult> = results.iter().filter(|r| r.success).collect();
    let num_success = successful.len();
    let num_fail = results.len() - num_success;

    if num_success == 0 {
        anyhow::bail!("All {num_fail} requests failed. Check server connectivity and model name.");
    }

    let total_output_tokens: usize = successful.iter().map(|r| r.output_tokens).sum();
    let total_input_tokens: usize = num_success * args.input_len;

    let mut ttfts: Vec<f64> = successful.iter().map(|r| r.ttft).collect();
    ttfts.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // TPOT = (e2el - ttft) / (output_tokens - 1) for requests with >1 token.
    let mut tpots: Vec<f64> = successful
        .iter()
        .filter(|r| r.output_tokens > 1)
        .map(|r| (r.e2el - r.ttft) / (r.output_tokens - 1) as f64)
        .collect();
    tpots.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mut itls: Vec<f64> = successful
        .iter()
        .flat_map(|r| r.itl.iter().copied())
        .collect();
    itls.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mut e2els: Vec<f64> = successful.iter().map(|r| r.e2el).collect();
    e2els.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Compute peak metrics matching Python's methodology.
    // Peak output tokens/s: bucket tokens into 1-second intervals.
    let peak_output_tps = compute_peak_output_tps(&successful, total_time);
    // Peak concurrent requests.
    let peak_concurrent = compute_peak_concurrent(&successful);

    let selected_metrics: Vec<&str> = args.percentile_metrics.split(',').collect();
    let selected_pcts: Vec<f64> = args
        .metric_percentiles
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    // Print summary (matches Python's benchmark_serving.py format).
    println!();
    println!("{:=^50}", " Serving Benchmark Result ");
    println!("{:<40} {:<10}", "Successful requests:", num_success);
    println!("{:<40} {:<10}", "Failed requests:", num_fail);
    if let Some(mc) = args.max_concurrency {
        println!("{:<40} {:<10}", "Maximum request concurrency:", mc);
    }
    if !args.request_rate.is_infinite() {
        println!(
            "{:<40} {:<10.2}",
            "Request rate configured (RPS):", args.request_rate
        );
    }
    println!("{:<40} {:<10.2}", "Benchmark duration (s):", total_time);
    println!("{:<40} {:<10}", "Total input tokens:", total_input_tokens);
    println!(
        "{:<40} {:<10}",
        "Total generated tokens:", total_output_tokens
    );
    println!(
        "{:<40} {:<10.2}",
        "Request throughput (req/s):",
        num_success as f64 / total_time
    );
    println!(
        "{:<40} {:<10.2}",
        "Output token throughput (tok/s):",
        total_output_tokens as f64 / total_time
    );
    println!(
        "{:<40} {:<10.2}",
        "Total token throughput (tok/s):",
        (total_input_tokens + total_output_tokens) as f64 / total_time
    );
    println!(
        "{:<40} {:<10.2}",
        "Peak output token throughput (tok/s):", peak_output_tps
    );
    println!(
        "{:<40} {:<10}",
        "Peak concurrent requests:", peak_concurrent
    );

    // Print per-metric stats (matches Python's process_one_metric format).
    let print_metric = |name: &str, header: &str, data: &[f64]| {
        if data.is_empty() {
            return;
        }
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let median = percentile(data, 50.0);
        let sd = std_dev(data);
        println!("{:-^50}", header);
        println!(
            "{:<40} {:<10.2}",
            format!("Mean {name} (ms):"),
            mean * 1000.0
        );
        println!(
            "{:<40} {:<10.2}",
            format!("Median {name} (ms):"),
            median * 1000.0
        );
        println!("{:<40} {:<10.2}", format!("Std {name} (ms):"), sd * 1000.0);
        for &p in &selected_pcts {
            let p_word = if p == p.floor() {
                format!("{}", p as i64)
            } else {
                format!("{p}")
            };
            println!(
                "{:<40} {:<10.2}",
                format!("P{p_word} {name} (ms):"),
                percentile(data, p) * 1000.0
            );
        }
    };

    for metric in &selected_metrics {
        match *metric {
            "ttft" => print_metric("TTFT", "Time to First Token", &ttfts),
            "tpot" => print_metric("TPOT", "Time per Output Token (excl. 1st token)", &tpots),
            "itl" => print_metric("ITL", "Inter-token Latency", &itls),
            "e2el" => print_metric("E2EL", "End-to-end Latency", &e2els),
            _ => eprintln!("Unknown metric: {metric}"),
        }
    }
    println!("{:=^50}", "");

    // Build JSON result object.
    let mean = |d: &[f64]| {
        if d.is_empty() {
            0.0
        } else {
            d.iter().sum::<f64>() / d.len() as f64
        }
    };

    let mut json = serde_json::json!({
        "duration": total_time,
        "completed": num_success,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "request_throughput": num_success as f64 / total_time,
        "output_throughput": total_output_tokens as f64 / total_time,
        "total_token_throughput": (total_input_tokens + total_output_tokens) as f64 / total_time,
        "peak_output_throughput": peak_output_tps,
        "peak_concurrent_requests": peak_concurrent,
    });
    let obj = json.as_object_mut().unwrap();

    let add_metric_json =
        |obj: &mut serde_json::Map<String, serde_json::Value>, attr: &str, data: &[f64]| {
            obj.insert(
                format!("mean_{attr}_ms"),
                serde_json::json!(mean(data) * 1000.0),
            );
            obj.insert(
                format!("median_{attr}_ms"),
                serde_json::json!(percentile(data, 50.0) * 1000.0),
            );
            obj.insert(
                format!("std_{attr}_ms"),
                serde_json::json!(std_dev(data) * 1000.0),
            );
            for &p in &selected_pcts {
                let p_word = if p == p.floor() {
                    format!("{}", p as i64)
                } else {
                    format!("{p}")
                };
                obj.insert(
                    format!("p{p_word}_{attr}_ms"),
                    serde_json::json!(percentile(data, p) * 1000.0),
                );
            }
        };

    for metric in &selected_metrics {
        match *metric {
            "ttft" => add_metric_json(obj, "ttft", &ttfts),
            "tpot" => add_metric_json(obj, "tpot", &tpots),
            "itl" => add_metric_json(obj, "itl", &itls),
            "e2el" => add_metric_json(obj, "e2el", &e2els),
            _ => {}
        }
    }

    // --output-json: explicit path.
    if let Some(ref path) = args.output_json {
        std::fs::write(path, serde_json::to_string_pretty(&json)?)?;
        eprintln!("Results written to {path}");
    }

    // --save-result: auto-generated filename.
    if args.save_result {
        let label = args.label.as_deref().unwrap_or("openai");
        let model_basename = model.rsplit('/').next().unwrap_or(&model);
        let datetime = crate::timestamp_filename_tag("-");

        let filename = if let Some(ref name) = args.result_filename {
            name.clone()
        } else {
            let rate_str = if args.request_rate.is_infinite() {
                "inf".to_string()
            } else {
                format!("{:.0}", args.request_rate)
            };
            format!("{label}-{rate_str}qps-{model_basename}-{datetime}.json")
        };

        let dir = args.result_dir.as_deref().unwrap_or(".");
        std::fs::create_dir_all(dir)?;
        let path = std::path::PathBuf::from(dir).join(&filename);
        std::fs::write(&path, serde_json::to_string_pretty(&json)?)?;
        eprintln!("Results saved to {}", path.display());
    }

    Ok(())
}

/// Compute peak output tokens/s by bucketing tokens into 1-second intervals.
fn compute_peak_output_tps(results: &[&RequestResult], total_time: f64) -> f64 {
    if results.is_empty() || total_time <= 0.0 {
        return 0.0;
    }

    let num_buckets = total_time.ceil() as usize + 1;
    let mut buckets = vec![0usize; num_buckets];

    for r in results {
        // Estimate when each token was generated:
        // First token at start_time + ttft, subsequent tokens spread via ITL.
        let first_token_time = r.start_time + r.ttft;

        // First token.
        let bucket = first_token_time.floor() as usize;
        if bucket < num_buckets {
            buckets[bucket] += 1;
        }

        // Subsequent tokens.
        let mut t = first_token_time;
        for &itl in &r.itl {
            t += itl;
            let bucket = t.floor() as usize;
            if bucket < num_buckets {
                buckets[bucket] += 1;
            }
        }
    }

    buckets.into_iter().max().unwrap_or(0) as f64
}

/// Compute peak concurrent requests.
fn compute_peak_concurrent(results: &[&RequestResult]) -> usize {
    if results.is_empty() {
        return 0;
    }

    // Build events: +1 at start_time, -1 at start_time + e2el.
    let mut events: Vec<(f64, i32)> = Vec::with_capacity(results.len() * 2);
    for r in results {
        events.push((r.start_time, 1));
        events.push((r.start_time + r.e2el, -1));
    }
    events.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(a.1.cmp(&b.1)));

    let mut current = 0i32;
    let mut peak = 0i32;
    for (_, delta) in events {
        current += delta;
        peak = peak.max(current);
    }

    peak as usize
}
