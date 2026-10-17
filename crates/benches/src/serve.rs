// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! `scr bench serve` — online serving benchmark.
//!
//! Sends concurrent HTTP requests to a running vLLM server (OpenAI-compatible
//! API) and measures per-request latency metrics: TTFT, TPOT, ITL, and
//! end-to-end latency (E2EL). `--dataset-name multi-turn` instead runs
//! closed-loop conversations against the chat endpoint and reports latency and
//! prefix-cache reuse per turn.

use std::io::Read;
use std::path::Path;
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use anyhow::Result;
use indicatif::{ProgressBar, ProgressStyle};
use rand::Rng;
use rand_distr::Gamma;
use scratchy_serving_api::Tokenizer;

use crate::args::{BenchServeArgs, ServeDataset};
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
    /// SSE chunks that carried text (what the timing clocks saw); 1 for a
    /// request whose only chunk was an empty final one (see `is_unstreamed`).
    chunks: usize,
    /// Timestamp (seconds since benchmark start) when request was sent.
    start_time: f64,
    success: bool,
}

impl RequestResult {
    fn failed(start_time: f64, e2el: f64) -> Self {
        RequestResult {
            ttft: 0.0,
            itl: vec![],
            e2el,
            output_tokens: 0,
            chunks: 0,
            start_time,
            success: false,
        }
    }
}

/// What a run measured.
struct Run {
    /// Every request sent.
    results: Vec<RequestResult>,
    /// Prompt tokens over the successful requests.
    total_input_tokens: usize,
    /// Wall time of the measured requests (seconds).
    total_time: f64,
}

/// What every request of a run shares: where it goes and how it samples.
struct Client {
    agent: Agent,
    api_url: String,
    model: String,
    api_key: Option<String>,
    ignore_eos: bool,
    temperature: Option<f64>,
    top_p: Option<f64>,
    top_k: Option<i32>,
}

impl Client {
    /// Add the sampling knobs to a request body. Unset ones are left out
    /// rather than sent as null, so the server applies its own default.
    fn sampling(&self, body: &mut serde_json::Value) {
        if self.ignore_eos {
            body["ignore_eos"] = serde_json::json!(true);
        }
        if let Some(t) = self.temperature {
            body["temperature"] = serde_json::json!(t);
        }
        if let Some(p) = self.top_p {
            body["top_p"] = serde_json::json!(p);
        }
        if let Some(k) = self.top_k {
            body["top_k"] = serde_json::json!(k);
        }
    }

    fn post(
        &self,
        body: &serde_json::Value,
        request_id: &str,
    ) -> Result<ureq::http::Response<ureq::Body>, ureq::Error> {
        let mut req = self
            .agent
            .post(&self.api_url)
            .header("content-type", "application/json")
            .header("x-request-id", request_id);
        if let Some(key) = &self.api_key {
            req = req.header("authorization", format!("Bearer {key}"));
        }
        req.send(body.to_string().as_str())
    }
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
fn send_request(
    client: &Client,
    prompt: &str,
    output_len: usize,
    request_id: &str,
    bench_start: Instant,
) -> RequestResult {
    // NOTE: `logprobs` omitted (rather than sent as null) so this bench
    // works against servers with strict OpenAI-schema validation
    // (mlx_lm.server, in particular, rejects `logprobs: null` because
    // it only accepts a bool). vLLM tolerates either.
    let mut body = serde_json::json!({
        "model": client.model,
        "prompt": prompt,
        "max_tokens": output_len,
        "stream": true,
        "stream_options": {"include_usage": true},
        "repetition_penalty": 1.0,
    });
    client.sampling(&mut body);

    let request_start = Instant::now();
    let start_time = request_start.duration_since(bench_start).as_secs_f64();
    let mut first_token_time: Option<Instant> = None;
    let mut last_token_time = request_start;
    let mut itl = Vec::new();
    let mut output_tokens = 0usize;

    let resp = match client.post(&body, request_id) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Request failed: {e}");
            return RequestResult::failed(start_time, request_start.elapsed().as_secs_f64());
        }
    };

    // The agent is built with `http_status_as_error(false)`, so a 4xx/5xx
    // arrives here as a normal response and the body is still readable.
    if !resp.status().is_success() {
        let status = resp.status();
        let body_text = resp.into_body().read_to_string().unwrap_or_default();
        eprintln!("Request failed with status {status}: {body_text}");
        return RequestResult::failed(start_time, request_start.elapsed().as_secs_f64());
    }

    // Parse SSE stream. Reading straight off the body reader keeps the
    // timestamps below as close to arrival as we can get them.
    let mut reader = resp.into_body().into_reader();
    let mut chunk = [0u8; 8192];
    let mut buf = String::new();
    let mut usage_completion_tokens: Option<usize> = None;
    let mut chunks = 0usize;
    let mut last_choices_time: Option<Instant> = None;
    // Why the stream says the request failed, if it does: the first one wins.
    let mut failure: Option<String> = None;
    // Whether the server closed the answer (`data: [DONE]`): a read error after that ends a
    // finished request, not a failed one.
    let mut done = false;

    loop {
        let n = match reader.read(&mut chunk) {
            Ok(0) => break,
            Ok(n) => n,
            Err(e) if !done => {
                failure.get_or_insert_with(|| format!("stream read failed: {e}"));
                break;
            }
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
                    done = true;
                    continue;
                }
                if let Some(data) = line.strip_prefix("data: ")
                    && let Ok(parsed) = serde_json::from_str::<serde_json::Value>(data)
                {
                    // Python's TTFT/ITL logic, counting only chunks that carry text:
                    // - First chunk with text → TTFT
                    // - Every subsequent chunk with text → ITL
                    // - `most_recent_timestamp` updated on every chunk with text
                    // An empty chunk is not a token: oMLX streams empty keepalive
                    // chunks while it prefills, and llama.cpp closes with one, so
                    // timing any chunk put TTFT before the prefill was done.
                    if has_choices(&parsed) {
                        last_choices_time = Some(Instant::now());
                    }
                    if chunk_has_text(&parsed) {
                        let now = Instant::now();
                        chunks += 1;
                        if first_token_time.is_none() {
                            first_token_time = Some(now);
                        } else {
                            itl.push(now.duration_since(last_token_time).as_secs_f64());
                        }
                        last_token_time = now;
                    }
                    usage_completion_tokens = merge_usage(usage_completion_tokens, &parsed);
                    if let Some(why) = chunk_failure(&parsed) {
                        failure.get_or_insert(why);
                    }
                }
            }
        }
    }

    // A stream that ends the request as aborted or errored failed, whatever it streamed before:
    // a server whose engine stopped answers every request at once with an empty `abort`.
    if let Some(why) = failure {
        eprintln!("Request failed: {why}");
        return RequestResult::failed(start_time, request_start.elapsed().as_secs_f64());
    }

    // No chunk carried text, but the server did answer: mlx_lm.server holds
    // back undecodable bytes and flushes one empty final chunk. Time that one
    // chunk, as before, so the request reads as unstreamed, not failed.
    if first_token_time.is_none()
        && let Some(t) = last_choices_time
    {
        first_token_time = Some(t);
        last_token_time = t;
        chunks = 1;
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
        chunks,
        start_time,
        success: first_token_time.is_some(),
    }
}

/// The completion-token count so far, updated from one chunk. Usage can ride
/// on any chunk: llama.cpp sends it on its last one, which also has `choices`.
/// The count only grows within a request, so keep the largest: an interim
/// `completion_tokens: 0` must not undo the final one.
fn merge_usage(so_far: Option<usize>, chunk: &serde_json::Value) -> Option<usize> {
    match chunk["usage"]["completion_tokens"].as_u64() {
        Some(ct) => Some(so_far.unwrap_or(0).max(ct as usize)),
        None => so_far,
    }
}

fn has_choices(chunk: &serde_json::Value) -> bool {
    chunk
        .get("choices")
        .is_some_and(|c| c.as_array().is_some_and(|a| !a.is_empty()))
}

/// Why a streamed chunk says its request failed: an `error` object (an SSE error event; a null
/// one is no error), or a choice finished `abort` or `error`. `None` for any other chunk, `stop`
/// and `length` included: a request may stop early on its own (oMLX stops at EOS despite
/// `ignore_eos`), so the token count alone says nothing.
fn chunk_failure(chunk: &serde_json::Value) -> Option<String> {
    if let Some(err) = chunk.get("error").filter(|e| !e.is_null()) {
        let message = err["message"]
            .as_str()
            .map_or_else(|| err.to_string(), str::to_string);
        return Some(format!("error event: {message}"));
    }
    let choices = chunk["choices"].as_array()?;
    (choices.iter())
        .filter_map(|c| c["finish_reason"].as_str())
        .find(|r| matches!(*r, "abort" | "error"))
        .map(|r| format!("finish_reason {r}"))
}

/// Whether a streamed chunk carries generated text: `text` on the completions
/// API, `delta.content` on chat. Keepalive and closing chunks carry none.
fn chunk_has_text(chunk: &serde_json::Value) -> bool {
    let Some(first) = chunk["choices"].as_array().and_then(|a| a.first()) else {
        return false;
    };
    [&first["text"], &first["delta"]["content"]]
        .iter()
        .any(|t| t.as_str().is_some_and(|s| !s.is_empty()))
}

/// A request whose whole multi-token output arrived in one `choices` chunk
/// was never observed streaming: its first-chunk time is its end time, so
/// TTFT == E2EL and TPOT == 0 by construction, not by measurement.
/// mlx_lm.server does this whenever the tokens decode to no printable text
/// (random-token prompts often make models emit undecodable byte
/// fragments), holding them back and flushing one empty final chunk.
fn is_unstreamed(r: &RequestResult) -> bool {
    r.output_tokens > 1 && r.chunks <= 1
}

/// `50` for 50.0, `99.9` for 99.9: the percentile's name in labels and keys.
fn p_word(p: f64) -> String {
    if p == p.floor() {
        format!("{}", p as i64)
    } else {
        format!("{p}")
    }
}

/// Compute percentile of a sorted slice using linear interpolation
/// matching numpy.percentile(method='linear').
pub(crate) fn percentile(sorted: &[f64], p: f64) -> f64 {
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

/// Arithmetic mean (0 for no samples).
fn mean(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    data.iter().sum::<f64>() / data.len() as f64
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

    let api_url = format!("{}{}", args.base_url, args.endpoint());
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
        scratchy_serving_api::Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {e}"))?
    };

    if args.is_multi_turn() {
        return run_multi_turn(&args, agent, api_url, &model, &tokenizer);
    }

    let client = Arc::new(Client {
        agent,
        api_url,
        model: model.clone(),
        api_key: args.api_key.clone(),
        ignore_eos: args.ignore_eos,
        temperature: args.temperature,
        top_p: args.top_p,
        top_k: args.top_k,
    });
    let run = run_prompts(&args, &client, &tokenizer)?;
    let total_time = run.total_time;

    // Compute metrics.
    let successful: Vec<&RequestResult> = run.results.iter().filter(|r| r.success).collect();
    let num_success = successful.len();
    let num_fail = run.results.len() - num_success;

    if num_success == 0 {
        anyhow::bail!("All {num_fail} requests failed. Check server connectivity and model name.");
    }

    let total_output_tokens: usize = successful.iter().map(|r| r.output_tokens).sum();
    let total_input_tokens = run.total_input_tokens;

    // TTFT/TPOT/ITL come only from requests the stream actually timed.
    let timed: Vec<&RequestResult> = successful
        .iter()
        .copied()
        .filter(|r| !is_unstreamed(r))
        .collect();
    let num_unstreamed = num_success - timed.len();

    let mut ttfts: Vec<f64> = timed.iter().map(|r| r.ttft).collect();
    ttfts.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // TPOT = (e2el - ttft) / (output_tokens - 1) for requests with >1 token.
    let mut tpots: Vec<f64> = timed
        .iter()
        .filter(|r| r.output_tokens > 1)
        .map(|r| (r.e2el - r.ttft) / (r.output_tokens - 1) as f64)
        .collect();
    tpots.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mut itls: Vec<f64> = timed.iter().flat_map(|r| r.itl.iter().copied()).collect();
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
    if num_unstreamed > 0 {
        println!(
            "{:<40} {:<10}",
            "Unstreamed requests (untimed):", num_unstreamed
        );
        eprintln!(
            "warning: {num_unstreamed}/{num_success} requests delivered all their tokens in a \
             single chunk, so their TTFT/TPOT/ITL are unobservable and were excluded; \
             E2EL and output throughput still count them"
        );
    }

    // Print per-metric stats (matches Python's process_one_metric format).
    let print_metric = |name: &str, header: &str, data: &[f64]| {
        if data.is_empty() {
            return;
        }
        let mean = mean(data);
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
            println!(
                "{:<40} {:<10.2}",
                format!("P{} {name} (ms):", p_word(p)),
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
    let mut json = serde_json::json!({
        "duration": total_time,
        "completed": num_success,
        // Requests sent that errored or timed out. Every metric here covers
        // only the completed ones, so a reader must see when that is a subset.
        "failed": num_fail,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "request_throughput": num_success as f64 / total_time,
        "output_throughput": total_output_tokens as f64 / total_time,
        "total_token_throughput": (total_input_tokens + total_output_tokens) as f64 / total_time,
        "peak_output_throughput": peak_output_tps,
        "peak_concurrent_requests": peak_concurrent,
        "unstreamed_requests": num_unstreamed,
    });
    let obj = json.as_object_mut().unwrap();

    // No samples is "not measured" (null), never 0 ms.
    let add_metric_json =
        |obj: &mut serde_json::Map<String, serde_json::Value>, attr: &str, data: &[f64]| {
            let ms = |stat: fn(&[f64]) -> f64| (!data.is_empty()).then(|| stat(data) * 1000.0);
            obj.insert(format!("mean_{attr}_ms"), serde_json::json!(ms(mean)));
            obj.insert(
                format!("median_{attr}_ms"),
                serde_json::json!(ms(|d| percentile(d, 50.0))),
            );
            obj.insert(format!("std_{attr}_ms"), serde_json::json!(ms(std_dev)));
            for &p in &selected_pcts {
                let v = (!data.is_empty()).then(|| percentile(data, p) * 1000.0);
                obj.insert(format!("p{}_{attr}_ms", p_word(p)), serde_json::json!(v));
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

    write_results(&args, &model, &json)
}

/// Write the results JSON where `--output-json` and `--save-result` say.
fn write_results(args: &BenchServeArgs, model: &str, json: &serde_json::Value) -> Result<()> {
    // --output-json: explicit path.
    if let Some(ref path) = args.output_json {
        std::fs::write(path, serde_json::to_string_pretty(json)?)?;
        eprintln!("Results written to {path}");
    }

    // --save-result: auto-generated filename.
    if args.save_result {
        let label = args.label.as_deref().unwrap_or("openai");
        let model_basename = model.rsplit('/').next().unwrap_or(model);
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
        std::fs::write(&path, serde_json::to_string_pretty(json)?)?;
        eprintln!("Results saved to {}", path.display());
    }

    Ok(())
}

/// One `multi-turn` request: how long it took and what the server's `usage`
/// said about it.
struct TurnResult {
    /// 0-based position in its conversation.
    turn: usize,
    success: bool,
    /// Request sent to response body read (seconds). Nothing streams, so this
    /// is the TTFT only at `--output-len 1`.
    latency: f64,
    /// `usage.completion_tokens` (0 when the server sent none).
    output_tokens: usize,
    /// `usage.prompt_tokens`, when the server sent it.
    prompt_tokens: Option<usize>,
    /// `usage.prompt_tokens_details.cached_tokens`, when the server sent it.
    cached_tokens: Option<usize>,
}

/// What every `multi-turn` request shares: where it goes and how it samples.
struct ChatClient {
    agent: Agent,
    api_url: String,
    api_key: Option<String>,
    /// Every field of a request body but its `messages`.
    body: serde_json::Value,
}

impl ChatClient {
    fn new(args: &BenchServeArgs, agent: Agent, api_url: String, model: &str) -> Self {
        let mut body = serde_json::json!({
            "model": model,
            "max_tokens": args.output_len,
            "stream": false,
        });
        // As in `send_request`: unset knobs are left out rather than sent as
        // null, so the server applies its own default.
        if args.ignore_eos {
            body["ignore_eos"] = serde_json::json!(true);
        }
        if let Some(t) = args.temperature {
            body["temperature"] = serde_json::json!(t);
        }
        if let Some(p) = args.top_p {
            body["top_p"] = serde_json::json!(p);
        }
        if let Some(k) = args.top_k {
            body["top_k"] = serde_json::json!(k);
        }
        if let Some(kwargs) = &args.chat_template_kwargs {
            body["chat_template_kwargs"] = serde_json::Value::Object(kwargs.clone());
        }
        ChatClient {
            agent,
            api_url,
            api_key: args.api_key.clone(),
            body,
        }
    }

    /// Send one non-streaming chat request, timed to the end of its body.
    ///
    /// The server renders the chat template; the client never does, so every
    /// backend applies its own to the same `messages`.
    fn send(&self, messages: serde_json::Value, turn: usize, request_id: &str) -> TurnResult {
        let mut body = self.body.clone();
        body["messages"] = messages;
        let mut result = TurnResult {
            turn,
            success: false,
            latency: 0.0,
            output_tokens: 0,
            prompt_tokens: None,
            cached_tokens: None,
        };

        let request_start = Instant::now();
        let mut req = self
            .agent
            .post(&self.api_url)
            .header("content-type", "application/json")
            .header("x-request-id", request_id);
        if let Some(key) = &self.api_key {
            req = req.header("authorization", format!("Bearer {key}"));
        }
        let resp = match req.send(body.to_string().as_str()) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("Request failed: {e}");
                return result;
            }
        };
        let status = resp.status();
        let text = match resp.into_body().read_to_string() {
            Ok(text) => text,
            Err(e) => {
                eprintln!("Reading the response failed: {e}");
                return result;
            }
        };
        result.latency = request_start.elapsed().as_secs_f64();
        if !status.is_success() {
            eprintln!("Request failed with status {status}: {text}");
            return result;
        }
        let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&text) else {
            eprintln!("Request returned a body that is not JSON: {text}");
            return result;
        };

        let count = |v: &serde_json::Value| v.as_u64().map(|n| n as usize);
        let usage = &parsed["usage"];
        result.output_tokens = count(&usage["completion_tokens"]).unwrap_or(0);
        result.prompt_tokens = count(&usage["prompt_tokens"]);
        result.cached_tokens = count(&usage["prompt_tokens_details"]["cached_tokens"]);
        result.success = parsed["choices"].as_array().is_some_and(|c| !c.is_empty());
        result
    }
}

/// What a `multi-turn` run measured.
struct MultiTurnRun {
    /// Every request sent, conversation by conversation.
    results: Vec<TurnResult>,
    /// Requests per conversation.
    turns: usize,
    /// Requests the run meant to send: more than `results` when a failed
    /// turn ended its conversation early.
    planned: usize,
    /// Prompt tokens over the successful requests.
    total_input_tokens: usize,
    /// Wall time of the measured requests (seconds).
    total_time: f64,
}

/// `--dataset-name multi-turn`: conversations sharing one system prompt, run
/// closed loop — turn k+1 is sent once turn k has answered — with
/// `--max-concurrency` bounding the conversations in flight.
fn run_multi_turn(
    args: &BenchServeArgs,
    agent: Agent,
    api_url: String,
    model: &str,
    tokenizer: &Tokenizer,
) -> Result<()> {
    eprintln!(
        "multi-turn: {} turns per conversation, system_len: {}, user_len: {}, reply_len: {}",
        args.multi_turn_turns,
        args.multi_turn_system_len,
        args.multi_turn_user_len,
        args.multi_turn_reply_len
    );
    let count_tokens = |text: &str| datasets::token_count(tokenizer, text);
    let spec = |conversations| datasets::MultiTurnSpec {
        conversations,
        turns: args.multi_turn_turns,
        system_len: args.multi_turn_system_len,
        user_len: args.multi_turn_user_len,
        reply_len: args.multi_turn_reply_len,
    };
    eprintln!(
        "Generating {} conversations sharing one ~{}-token system prompt...",
        args.num_prompts, args.multi_turn_system_len
    );
    let dataset = Arc::new(datasets::generate_multi_turn(
        &count_tokens,
        &spec(args.num_prompts),
        args.seed,
    )?);
    let turns = dataset.turns();
    if args.output_len > 1 {
        eprintln!(
            "note: multi-turn requests are not streamed, so at --output-len {} the per-turn \
             latency includes decode; --output-len 1 makes it the TTFT",
            args.output_len
        );
    }
    let client = Arc::new(ChatClient::new(args, agent, api_url, model));

    // Throwaway conversations come from a disjoint seed, so even their system
    // prompt is not the measured one. The pre-flight is conversation 0's
    // first turn; the warmups walk the rest in order, a conversation's turns
    // back to back as in the run.
    let throwaway = datasets::generate_multi_turn(
        &count_tokens,
        &spec(1 + args.num_warmups.div_ceil(turns)),
        datasets::throwaway_seed(args.seed),
    )?;
    preflight(|| {
        client
            .send(throwaway.messages(0, 0), 0, "preflight")
            .success
    })?;
    warm_up(args.num_warmups, |i| {
        let (conv, turn) = (1 + i / turns, i % turns);
        client.send(throwaway.messages(conv, turn), turn, &format!("warmup-{i}"));
    });

    let num_conversations = dataset.num_conversations();
    let pb = progress_bar(args, num_conversations * turns);
    let semaphore = args.max_concurrency.map(|n| Arc::new(Semaphore::new(n)));
    // `--request-rate` paces conversation starts.
    let cumulative_delays = generate_request_delays(
        num_conversations,
        args.request_rate,
        args.burstiness,
        args.seed.wrapping_add(42),
    );

    let benchmark_start = Instant::now();
    let mut handles = Vec::with_capacity(num_conversations);
    for (conv, &start_at) in cumulative_delays.iter().enumerate() {
        wait_until(benchmark_start, start_at);

        let client = client.clone();
        let dataset = dataset.clone();
        let pb = pb.clone();
        // As for single prompts, but the permit is held for the whole
        // conversation.
        let permit = semaphore.as_ref().map(|s| s.acquire_owned());
        handles.push(std::thread::spawn(move || {
            let _permit = permit;
            let mut sent = Vec::with_capacity(turns);
            for turn in 0..turns {
                let request_id = format!("bench-{conv}-{turn}");
                let result = client.send(dataset.messages(conv, turn), turn, &request_id);
                if let Some(ref pb) = pb {
                    pb.inc(1);
                }
                let ok = result.success;
                sent.push(result);
                // The next turn would be timed against a cache this one may
                // never have filled, so a failure ends the conversation.
                if !ok {
                    break;
                }
            }
            sent
        }));
    }

    let conversations = join_all(handles)?;
    let total_time = benchmark_start.elapsed().as_secs_f64();
    if let Some(pb) = pb {
        pb.finish_and_clear();
    }

    // The server's prompt count includes the chat template; the content count
    // is the floor to fall back on when it sends none.
    let mut total_input_tokens = 0;
    for (conv, sent) in conversations.iter().enumerate() {
        for r in sent.iter().filter(|r| r.success) {
            total_input_tokens += r
                .prompt_tokens
                .unwrap_or_else(|| dataset.content_tokens(conv, r.turn));
        }
    }
    let run = MultiTurnRun {
        results: conversations.into_iter().flatten().collect(),
        turns,
        planned: num_conversations * turns,
        total_input_tokens,
        total_time,
    };
    report_multi_turn(args, model, &run)
}

/// Print a `multi-turn` run's summary and per-turn table, and write its JSON.
fn report_multi_turn(args: &BenchServeArgs, model: &str, run: &MultiTurnRun) -> Result<()> {
    let num_success = run.results.iter().filter(|r| r.success).count();
    let num_fail = run.planned - num_success;
    if num_success == 0 {
        anyhow::bail!("All {num_fail} requests failed. Check server connectivity and model name.");
    }
    let total_output_tokens: usize = run
        .results
        .iter()
        .filter(|r| r.success)
        .map(|r| r.output_tokens)
        .sum();
    let total_time = run.total_time;
    let total_input_tokens = run.total_input_tokens;
    let selected_pcts: Vec<f64> = args
        .metric_percentiles
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let per_turn = turn_stats(&run.results, run.turns);

    println!();
    println!("{:=^50}", " Serving Benchmark Result ");
    println!("{:<40} {:<10}", "Successful requests:", num_success);
    println!("{:<40} {:<10}", "Failed requests:", num_fail);
    if let Some(mc) = args.max_concurrency {
        println!("{:<40} {:<10}", "Maximum conversation concurrency:", mc);
    }
    if !args.request_rate.is_infinite() {
        println!(
            "{:<40} {:<10.2}",
            "Conversation rate configured (conv/s):", args.request_rate
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
    print_turn_table(&per_turn, &selected_pcts);
    println!("{:=^50}", "");

    let mut json = serde_json::json!({
        "duration": total_time,
        "completed": num_success,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "request_throughput": num_success as f64 / total_time,
        "output_throughput": total_output_tokens as f64 / total_time,
        "total_token_throughput": (total_input_tokens + total_output_tokens) as f64 / total_time,
    });
    let obj = json.as_object_mut().unwrap();
    // Per turn, 1-based. "ttft" is the request latency: see `ChatClient::send`.
    for (k, turn) in per_turn.iter().enumerate().map(|(i, t)| (i + 1, t)) {
        let ms = |p| (!turn.latencies.is_empty()).then(|| percentile(&turn.latencies, p) * 1000.0);
        obj.insert(
            format!("completed_turn{k}"),
            serde_json::json!(turn.completed),
        );
        obj.insert(
            format!("median_ttft_ms_turn{k}"),
            serde_json::json!(ms(50.0)),
        );
        for &p in &selected_pcts {
            obj.insert(
                format!("p{}_ttft_ms_turn{k}", p_word(p)),
                serde_json::json!(ms(p)),
            );
        }
        obj.insert(
            format!("mean_prompt_tokens_turn{k}"),
            serde_json::json!(turn.mean_prompt_tokens),
        );
        obj.insert(
            format!("mean_cached_tokens_turn{k}"),
            serde_json::json!(turn.mean_cached_tokens),
        );
    }

    write_results(args, model, &json)
}

/// One turn's slice of a `multi-turn` run.
struct TurnStats {
    /// Successful requests at this turn.
    completed: usize,
    /// Their latencies (seconds), sorted.
    latencies: Vec<f64>,
    /// `None` when no request at this turn reported a prompt size.
    mean_prompt_tokens: Option<f64>,
    /// `None` when the server reported no cached tokens anywhere in the run.
    mean_cached_tokens: Option<f64>,
}

/// Break a `multi-turn` run's successful requests out per turn.
///
/// scratchy leaves `prompt_tokens_details` out when nothing was reused, so a
/// missing cached count reads as 0 once the server has reported one anywhere
/// in the run. A server that never reports one gets null, not a made-up 0.
fn turn_stats(results: &[TurnResult], turns: usize) -> Vec<TurnStats> {
    let ok = || results.iter().filter(|r| r.success);
    let reports_cached = ok().any(|r| r.cached_tokens.is_some());
    (0..turns)
        .map(|turn| {
            let at: Vec<&TurnResult> = ok().filter(|r| r.turn == turn).collect();
            let mut latencies: Vec<f64> = at.iter().map(|r| r.latency).collect();
            latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let prompt: Vec<f64> = at
                .iter()
                .filter_map(|r| r.prompt_tokens)
                .map(|n| n as f64)
                .collect();
            let cached: Vec<f64> = at
                .iter()
                .map(|r| r.cached_tokens.unwrap_or(0) as f64)
                .collect();
            TurnStats {
                completed: at.len(),
                latencies,
                mean_prompt_tokens: (!prompt.is_empty()).then(|| mean(&prompt)),
                mean_cached_tokens: (reports_cached && !cached.is_empty()).then(|| mean(&cached)),
            }
        })
        .collect()
}

fn print_turn_table(stats: &[TurnStats], pcts: &[f64]) {
    let tokens = |v: Option<f64>| v.map_or_else(|| "-".to_string(), |v| format!("{v:.0}"));
    println!("{:-^50}", "Per-turn Latency (TTFT at --output-len 1)");
    let mut header = format!("{:<6}{:>6}{:>12}", "Turn", "Done", "Median ms");
    for &p in pcts {
        header += &format!("{:>10}", format!("P{} ms", p_word(p)));
    }
    println!("{header}{:>12}{:>12}", "Prompt tok", "Cached tok");
    for (k, turn) in stats.iter().enumerate() {
        let ms = |p| {
            if turn.latencies.is_empty() {
                "-".to_string()
            } else {
                format!("{:.2}", percentile(&turn.latencies, p) * 1000.0)
            }
        };
        let mut row = format!("{:<6}{:>6}{:>12}", k + 1, turn.completed, ms(50.0));
        for &p in pcts {
            row += &format!("{:>10}", ms(p));
        }
        println!(
            "{row}{:>12}{:>12}",
            tokens(turn.mean_prompt_tokens),
            tokens(turn.mean_cached_tokens)
        );
    }
}

/// `random` / `sharegpt`: one streamed completion per prompt, open loop.
fn run_prompts(args: &BenchServeArgs, client: &Arc<Client>, tokenizer: &Tokenizer) -> Result<Run> {
    let prompt_entries = if args.dataset_name == ServeDataset::Sharegpt {
        let dataset_path = args
            .dataset_path
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("--dataset-path is required for sharegpt dataset"))?;
        eprintln!("Loading ShareGPT dataset from {dataset_path}...");
        let samples = datasets::load_sharegpt(
            Path::new(dataset_path),
            tokenizer,
            args.num_prompts,
            None,
            args.seed,
        )?;
        eprintln!("Loaded {} samples from ShareGPT dataset", samples.len());
        samples
    } else {
        eprintln!(
            "Generating {} random prompts (matching Python RandomDataset)...",
            args.num_prompts
        );
        datasets::generate_random(
            tokenizer,
            args.num_prompts,
            args.input_len,
            args.output_len,
            args.random_range_ratio,
            args.random_prefix_len,
            args.seed,
        )?
    };

    // Random-token throwaways for ShareGPT too: any other sample of the file
    // could be one of the measured ones.
    let throwaway = datasets::generate_random(
        tokenizer,
        1 + args.num_warmups,
        args.input_len,
        args.output_len,
        args.random_range_ratio,
        args.random_prefix_len,
        datasets::throwaway_seed(args.seed),
    )?;
    let send_throwaway = |i: usize, request_id: &str| {
        let p = &throwaway[i];
        let output_len = p.expected_output_len.min(args.output_len);
        send_request(client, &p.prompt, output_len, request_id, Instant::now())
    };
    preflight(|| send_throwaway(0, "preflight").success)?;
    warm_up(args.num_warmups, |i| {
        send_throwaway(1 + i, &format!("warmup-{i}"));
    });

    let num_prompts = prompt_entries.len();
    let prompt_lens: Vec<usize> = prompt_entries.iter().map(|p| p.prompt_len).collect();
    let pb = progress_bar(args, num_prompts);
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
        wait_until(benchmark_start, cumulative_delays[i]);

        let client = client.clone();
        let pb = pb.clone();
        let request_id = format!("bench-{i}");

        // A thread per in-flight request. `--max-concurrency` bounds how many
        // exist at once, and the permit is acquired HERE rather than inside
        // the thread so the pacing loop blocks instead of running ahead and
        // spawning threads that would only queue — the same back-pressure the
        // async version got from awaiting the permit before spawning.
        let permit = semaphore.as_ref().map(|s| s.acquire_owned());
        handles.push(std::thread::spawn(move || {
            // Held for the request's lifetime; returned on drop, including on
            // unwind, so a failed request cannot shrink the concurrency bound.
            let _permit = permit;
            let result = send_request(
                &client,
                &entry.prompt,
                entry.expected_output_len,
                &request_id,
                benchmark_start,
            );
            if let Some(ref pb) = pb {
                pb.inc(1);
            }
            result
        }));
    }

    let results = join_all(handles)?;
    let total_time = benchmark_start.elapsed().as_secs_f64();
    if let Some(pb) = pb {
        pb.finish_and_clear();
    }

    let total_input_tokens = results
        .iter()
        .zip(&prompt_lens)
        .filter(|(r, _)| r.success)
        .map(|(_, &len)| len)
        .sum();
    Ok(Run {
        results,
        total_input_tokens,
        total_time,
    })
}

/// Send one request to verify connectivity, and fail fast if it does not land.
fn preflight(send: impl FnOnce() -> bool) -> Result<()> {
    eprintln!("Sending pre-flight request to verify connectivity...");
    if !send() {
        anyhow::bail!("Pre-flight request failed. Check server connectivity and model name.");
    }
    eprintln!("Pre-flight request succeeded.");
    Ok(())
}

/// Send `n` untimed warmup requests, `send(i)` sending the i-th.
fn warm_up(n: usize, mut send: impl FnMut(usize)) {
    if n == 0 {
        return;
    }
    eprintln!("Sending {n} warmup request(s)...");
    let warmup_pb = ProgressBar::new(n as u64);
    warmup_pb.set_style(
        ProgressStyle::with_template(
            "Warmup {wide_bar:.yellow/blue} {pos}/{len} [{elapsed}<{eta}]",
        )
        .unwrap(),
    );
    for i in 0..n {
        send(i);
        warmup_pb.inc(1);
    }
    warmup_pb.finish_and_clear();
    eprintln!("Warmup complete.");
}

fn progress_bar(args: &BenchServeArgs, len: usize) -> Option<ProgressBar> {
    if args.disable_tqdm {
        return None;
    }
    let pb = ProgressBar::new(len as u64);
    pb.set_style(
        ProgressStyle::with_template(
            "Benchmarking {wide_bar:.cyan/blue} {pos}/{len} [{elapsed}<{eta}, {per_sec}]",
        )
        .unwrap()
        .with_key("per_sec", crate::fmt_tqdm_rate),
    );
    Some(pb)
}

/// Sleep until `at` seconds after `start`: the schedule's next send time.
fn wait_until(start: Instant, at: f64) {
    let target = Duration::from_secs_f64(at);
    let elapsed = start.elapsed();
    if target > elapsed {
        std::thread::sleep(target - elapsed);
    }
}

fn join_all<T>(handles: Vec<JoinHandle<T>>) -> Result<Vec<T>> {
    handles
        .into_iter()
        .map(|h| {
            h.join()
                .map_err(|_| anyhow::anyhow!("benchmark request thread panicked"))
        })
        .collect()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn usage_keeps_the_largest_count_from_any_chunk() {
        let c = |s: &str| serde_json::from_str::<serde_json::Value>(s).unwrap();
        // llama.cpp: usage on the last chunk, which also has choices.
        let last = c(r#"{"choices":[{"text":""}],"usage":{"completion_tokens":5}}"#);
        assert_eq!(merge_usage(None, &last), Some(5));
        // A later interim zero does not undo it; a chunk without usage keeps it.
        let zero = c(r#"{"choices":[],"usage":{"completion_tokens":0}}"#);
        assert_eq!(merge_usage(Some(5), &zero), Some(5));
        assert_eq!(
            merge_usage(Some(5), &c(r#"{"choices":[{"text":"x"}]}"#)),
            Some(5)
        );
        assert_eq!(merge_usage(None, &c(r#"{"choices":[{"text":"x"}]}"#)), None);
    }

    #[test]
    fn only_chunks_with_text_are_tokens() {
        let c = |s: &str| serde_json::from_str::<serde_json::Value>(s).unwrap();
        // oMLX's keepalive while it prefills, and llama.cpp's closing chunk.
        assert!(!chunk_has_text(&c(
            r#"{"model":"keepalive","choices":[{"index":0,"text":""}]}"#
        )));
        assert!(!chunk_has_text(&c(
            r#"{"choices":[{"text":"","finish_reason":"length"}],"usage":{"completion_tokens":5}}"#
        )));
        assert!(!chunk_has_text(&c(
            r#"{"choices":[],"usage":{"completion_tokens":4}}"#
        )));
        assert!(chunk_has_text(&c(
            r#"{"choices":[{"index":0,"text":" four"}]}"#
        )));
        assert!(chunk_has_text(&c(
            r#"{"choices":[{"delta":{"content":"hi"}}]}"#
        )));
        assert!(has_choices(&c(r#"{"choices":[{"text":""}]}"#)));
    }

    #[test]
    fn a_stream_that_ends_aborted_or_errored_failed() {
        let c = |s: &str| serde_json::from_str::<serde_json::Value>(s).unwrap();
        // A scratchy server whose engine stopped: an empty `abort` delta, at once or after the
        // no-progress watchdog.
        assert_eq!(
            chunk_failure(&c(
                r#"{"choices":[{"index":0,"text":"","finish_reason":"abort"}]}"#
            )),
            Some("finish_reason abort".into())
        );
        assert!(
            chunk_failure(&c(r#"{"choices":[{"delta":{},"finish_reason":"error"}]}"#)).is_some()
        );
        assert_eq!(
            chunk_failure(&c(r#"{"error":{"message":"out of memory","code":500}}"#)),
            Some("error event: out of memory".into())
        );
        // A request that stops early on its own, or runs to its length, did not fail.
        assert_eq!(
            chunk_failure(&c(r#"{"choices":[{"text":"","finish_reason":"stop"}]}"#)),
            None
        );
        assert_eq!(
            chunk_failure(&c(r#"{"choices":[{"text":"x","finish_reason":"length"}]}"#)),
            None
        );
        assert_eq!(chunk_failure(&c(r#"{"choices":[{"text":"x"}]}"#)), None);
        // A null error is no error.
        assert_eq!(
            chunk_failure(&c(r#"{"error":null,"choices":[{"text":"x"}]}"#)),
            None
        );
        assert_eq!(
            chunk_failure(&c(r#"{"choices":[],"usage":{"completion_tokens":4}}"#)),
            None
        );
    }

    /// A stream that breaks off mid-answer failed; one that breaks off only after `[DONE]` finished.
    /// The server here streams one token in a chunked body it closes without the final empty
    /// chunk, so the client's read ends in an error rather than EOF.
    #[test]
    fn a_read_error_fails_a_request_only_before_done() {
        use std::io::{Read as _, Write as _};
        let served = |done: bool| {
            let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            let port = listener.local_addr().unwrap().port();
            let server = std::thread::spawn(move || {
                let (mut s, _) = listener.accept().unwrap();
                let mut req = [0u8; 4096];
                let _ = s.read(&mut req);
                let mut events = String::from(
                    "data: {\"choices\":[{\"text\":\"x\",\"finish_reason\":null}]}\n\n",
                );
                if done {
                    events += "data: [DONE]\n\n";
                }
                let head = "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\n\
                            transfer-encoding: chunked\r\n\r\n";
                let body = format!("{:x}\r\n{events}\r\n", events.len());
                s.write_all(head.as_bytes()).unwrap();
                s.write_all(body.as_bytes()).unwrap();
                s.flush().unwrap();
                // Dropped here with no `0\r\n\r\n`: the body ends unterminated.
            });
            let client = Client {
                agent: crate::http::agent(false),
                api_url: format!("http://127.0.0.1:{port}/v1/completions"),
                model: "m".into(),
                api_key: None,
                ignore_eos: false,
                temperature: None,
                top_p: None,
                top_k: None,
            };
            let result = send_request(&client, "p", 4, "r", Instant::now());
            server.join().unwrap();
            result.success
        };
        assert!(!served(false), "a stream broken off before [DONE] failed");
        assert!(served(true), "a stream broken off after [DONE] finished");
    }

    fn req(output_tokens: usize, chunks: usize, ttft: f64, e2el: f64) -> RequestResult {
        RequestResult {
            ttft,
            itl: vec![],
            e2el,
            output_tokens,
            chunks,
            start_time: 0.0,
            success: true,
        }
    }

    #[test]
    fn one_chunk_carrying_many_tokens_is_unstreamed() {
        // mlx_lm.server holding back 128 undecodable tokens: one empty final
        // chunk, TTFT == E2EL — would read as TPOT 0 ms.
        assert!(is_unstreamed(&req(128, 1, 6.1, 6.1)));
    }

    #[test]
    fn streamed_and_single_token_requests_are_timed() {
        assert!(!is_unstreamed(&req(128, 129, 3.8, 6.1)));
        // Two chunks for 128 tokens: 126 were held back and flushed at the
        // end, so its one ITL means nothing. TPOT spans first to last chunk,
        // so it holds as long as the first chunk came on time, which chunk
        // counts can't show. Nor can they tell this from spec decode sending
        // several tokens per chunk on purpose, so it stays timed.
        assert!(!is_unstreamed(&req(128, 2, 8.0, 10.3)));
        // max_tokens 1: one chunk is all there is to see.
        assert!(!is_unstreamed(&req(1, 1, 0.3, 0.3)));
    }

    fn turn(turn: usize, latency: f64, cached_tokens: Option<usize>) -> TurnResult {
        TurnResult {
            turn,
            success: true,
            latency,
            output_tokens: 1,
            prompt_tokens: Some(2100 + 400 * turn),
            cached_tokens,
        }
    }

    #[test]
    fn a_missing_cached_count_is_zero_only_from_a_server_that_reports_one() {
        // scratchy on a cold turn 1: no `prompt_tokens_details` at all.
        let stats = turn_stats(&[turn(0, 0.8, None), turn(1, 0.1, Some(2096))], 2);
        assert_eq!(stats[0].mean_cached_tokens, Some(0.0));
        assert_eq!(stats[1].mean_cached_tokens, Some(2096.0));
        assert_eq!(stats[1].mean_prompt_tokens, Some(2500.0));
        assert_eq!(stats[1].latencies, vec![0.1]);

        let never = turn_stats(&[turn(0, 0.8, None), turn(1, 0.1, None)], 2);
        assert_eq!(never[1].mean_cached_tokens, None);
    }

    #[test]
    fn failed_turns_count_toward_no_turn_stats() {
        let failed = TurnResult {
            success: false,
            ..turn(1, 9.0, Some(2096))
        };
        let stats = turn_stats(&[turn(0, 0.8, None), failed], 2);
        assert_eq!(stats[1].completed, 0);
        assert!(stats[1].latencies.is_empty());
        assert_eq!(stats[1].mean_prompt_tokens, None);
        // Its cached count came from a request that failed: not a report.
        assert_eq!(stats[0].mean_cached_tokens, None);
    }
}
