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
use tokenizers::Tokenizer;

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
    /// SSE chunks that carried `choices` (what the timing clocks saw).
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

/// What a `multi-turn` request reports beyond its [`RequestResult`].
struct TurnUsage {
    /// 0-based position in its conversation.
    turn: usize,
    /// `usage.prompt_tokens`, when the server sent it.
    prompt_tokens: Option<usize>,
    /// `usage.prompt_tokens_details.cached_tokens`, when the server sent it.
    cached_tokens: Option<usize>,
}

/// What a run measured, whichever dataset drove it.
struct Run {
    /// Every request sent.
    results: Vec<RequestResult>,
    /// Index-aligned with `results`; empty unless `multi-turn`.
    turn_usage: Vec<TurnUsage>,
    /// Requests per conversation; 0 unless `multi-turn`.
    turns: usize,
    /// Requests the run meant to send: more than `results` when a failed
    /// turn ended its conversation early.
    planned: usize,
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
    chat_template_kwargs: Option<serde_json::Map<String, serde_json::Value>>,
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
                        chunks += 1;
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
        chunks,
        start_time,
        success: first_token_time.is_some(),
    }
}

/// Send one non-streaming chat request, timed to the end of its body.
///
/// Nothing streams, so the round trip stands in for TTFT and E2EL alike: it
/// is the TTFT at `max_tokens` 1, and past that [`is_unstreamed`] keeps it
/// out of the TTFT stats. The server renders the chat template; the client
/// never does, so every backend applies its own to the same `messages`.
fn send_chat_request(
    client: &Client,
    messages: &serde_json::Value,
    turn: usize,
    output_len: usize,
    request_id: &str,
    bench_start: Instant,
) -> (RequestResult, TurnUsage) {
    let mut body = serde_json::json!({
        "model": client.model,
        "messages": messages,
        "max_tokens": output_len,
        "stream": false,
    });
    client.sampling(&mut body);
    if let Some(kwargs) = &client.chat_template_kwargs {
        body["chat_template_kwargs"] = serde_json::Value::Object(kwargs.clone());
    }

    let request_start = Instant::now();
    let start_time = request_start.duration_since(bench_start).as_secs_f64();
    let mut usage = TurnUsage {
        turn,
        prompt_tokens: None,
        cached_tokens: None,
    };
    let failed = || RequestResult::failed(start_time, request_start.elapsed().as_secs_f64());

    let resp = match client.post(&body, request_id) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Request failed: {e}");
            return (failed(), usage);
        }
    };
    let status = resp.status();
    let text = match resp.into_body().read_to_string() {
        Ok(text) => text,
        Err(e) => {
            eprintln!("Reading the response failed: {e}");
            return (failed(), usage);
        }
    };
    let latency = request_start.elapsed().as_secs_f64();
    if !status.is_success() {
        eprintln!("Request failed with status {status}: {text}");
        return (failed(), usage);
    }
    let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&text) else {
        eprintln!("Request returned a body that is not JSON: {text}");
        return (failed(), usage);
    };

    let count = |v: &serde_json::Value| v.as_u64().map(|n| n as usize);
    usage.prompt_tokens = count(&parsed["usage"]["prompt_tokens"]);
    usage.cached_tokens = count(&parsed["usage"]["prompt_tokens_details"]["cached_tokens"]);
    let result = RequestResult {
        ttft: latency,
        itl: vec![],
        e2el: latency,
        output_tokens: count(&parsed["usage"]["completion_tokens"]).unwrap_or(0),
        chunks: 1,
        start_time,
        success: parsed["choices"].as_array().is_some_and(|c| !c.is_empty()),
    };
    (result, usage)
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
    if args.dataset_name == ServeDataset::MultiTurn {
        eprintln!(
            "multi-turn: {} turns per conversation, system_len: {}, user_len: {}, reply_len: {}",
            args.multi_turn_turns,
            args.multi_turn_system_len,
            args.multi_turn_user_len,
            args.multi_turn_reply_len
        );
    }

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

    let client = Arc::new(Client {
        agent,
        api_url,
        model: model.clone(),
        api_key: args.api_key.clone(),
        ignore_eos: args.ignore_eos,
        temperature: args.temperature,
        top_p: args.top_p,
        top_k: args.top_k,
        chat_template_kwargs: args.chat_template_kwargs.clone(),
    });
    let run = match args.dataset_name {
        ServeDataset::Random | ServeDataset::Sharegpt => run_prompts(&args, &client, &tokenizer)?,
        ServeDataset::MultiTurn => run_conversations(&args, &client, &tokenizer)?,
    };
    let total_time = run.total_time;

    // Compute metrics.
    let successful: Vec<&RequestResult> = run.results.iter().filter(|r| r.success).collect();
    let num_success = successful.len();
    let num_fail = run.planned - num_success;

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
    let per_turn = turn_stats(&run);
    if !per_turn.is_empty() {
        print_turn_table(&per_turn, &selected_pcts);
    }
    println!("{:=^50}", "");

    // Build JSON result object.
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

    // Per turn, 1-based. "ttft" is the request latency: see send_chat_request.
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
    preflight(|| send_throwaway(0, "preflight"))?;
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
        planned: results.len(),
        results,
        turn_usage: Vec::new(),
        turns: 0,
        total_input_tokens,
        total_time,
    })
}

/// `multi-turn`: conversations run closed loop — turn k+1 is sent once turn
/// k has answered — and `--max-concurrency` bounds conversations in flight.
fn run_conversations(
    args: &BenchServeArgs,
    client: &Arc<Client>,
    tokenizer: &Tokenizer,
) -> Result<Run> {
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

    // Throwaway conversations come from a disjoint seed, so even their system
    // prompt is not the measured one. The pre-flight is conversation 0's
    // first turn; the warmups walk the rest in order, a conversation's turns
    // back to back as in the run.
    let throwaway = datasets::generate_multi_turn(
        &count_tokens,
        &spec(1 + args.num_warmups.div_ceil(turns)),
        datasets::throwaway_seed(args.seed),
    )?;
    let send_throwaway = |conv: usize, turn: usize, request_id: &str| {
        let messages = throwaway.messages(conv, turn);
        send_chat_request(
            client,
            &messages,
            turn,
            args.output_len,
            request_id,
            Instant::now(),
        )
        .0
    };
    preflight(|| send_throwaway(0, 0, "preflight"))?;
    warm_up(args.num_warmups, |i| {
        send_throwaway(1 + i / turns, i % turns, &format!("warmup-{i}"));
    });

    let num_conversations = dataset.num_conversations();
    let pb = progress_bar(args, num_conversations * turns);
    let semaphore = args.max_concurrency.map(|n| Arc::new(Semaphore::new(n)));
    // Request-rate pacing applies to conversation starts.
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
        let output_len = args.output_len;
        // As in `run_prompts`, but the permit is held for the whole
        // conversation.
        let permit = semaphore.as_ref().map(|s| s.acquire_owned());
        handles.push(std::thread::spawn(move || {
            let _permit = permit;
            let mut sent = Vec::with_capacity(turns);
            for turn in 0..turns {
                let (result, usage) = send_chat_request(
                    &client,
                    &dataset.messages(conv, turn),
                    turn,
                    output_len,
                    &format!("bench-{conv}-{turn}"),
                    benchmark_start,
                );
                if let Some(ref pb) = pb {
                    pb.inc(1);
                }
                let ok = result.success;
                sent.push((result, usage));
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

    let mut run = Run {
        results: Vec::new(),
        turn_usage: Vec::new(),
        turns,
        planned: num_conversations * turns,
        total_input_tokens: 0,
        total_time,
    };
    for (conv, sent) in conversations.into_iter().enumerate() {
        for (result, usage) in sent {
            // The server's count includes the chat template; the content
            // count is the floor to fall back on when it sends none.
            if result.success {
                run.total_input_tokens += usage
                    .prompt_tokens
                    .unwrap_or_else(|| dataset.content_tokens(conv, usage.turn));
            }
            run.results.push(result);
            run.turn_usage.push(usage);
        }
    }
    Ok(run)
}

/// Send one request to verify connectivity, and fail fast if it does not land.
fn preflight(send: impl FnOnce() -> RequestResult) -> Result<()> {
    eprintln!("Sending pre-flight request to verify connectivity...");
    if !send().success {
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

/// Break a `multi-turn` run out per turn; empty for any other dataset.
///
/// scratchy leaves `prompt_tokens_details` out when nothing was reused, so a
/// missing cached count reads as 0 once the server has reported one anywhere
/// in the run. A server that never reports one gets null, not a made-up 0.
fn turn_stats(run: &Run) -> Vec<TurnStats> {
    let ok = || {
        run.results
            .iter()
            .zip(&run.turn_usage)
            .filter(|(r, _)| r.success)
    };
    let reports_cached = ok().any(|(_, u)| u.cached_tokens.is_some());
    (0..run.turns)
        .map(|turn| {
            let at: Vec<_> = ok().filter(|(_, u)| u.turn == turn).collect();
            let mut latencies: Vec<f64> = at.iter().map(|(r, _)| r.e2el).collect();
            latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let prompt: Vec<f64> = at
                .iter()
                .filter_map(|(_, u)| u.prompt_tokens)
                .map(|n| n as f64)
                .collect();
            let cached: Vec<f64> = at
                .iter()
                .map(|(_, u)| u.cached_tokens.unwrap_or(0) as f64)
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

    fn usage(turn: usize, cached_tokens: Option<usize>) -> TurnUsage {
        TurnUsage {
            turn,
            prompt_tokens: Some(2100 + 400 * turn),
            cached_tokens,
        }
    }

    #[test]
    fn a_missing_cached_count_is_zero_only_from_a_server_that_reports_one() {
        // scratchy on a cold turn 1: no `prompt_tokens_details` at all.
        let run = Run {
            results: vec![req(1, 1, 0.8, 0.8), req(1, 1, 0.1, 0.1)],
            turn_usage: vec![usage(0, None), usage(1, Some(2096))],
            turns: 2,
            planned: 2,
            total_input_tokens: 0,
            total_time: 1.0,
        };
        let stats = turn_stats(&run);
        assert_eq!(stats[0].mean_cached_tokens, Some(0.0));
        assert_eq!(stats[1].mean_cached_tokens, Some(2096.0));
        assert_eq!(stats[1].mean_prompt_tokens, Some(2500.0));
        assert_eq!(stats[1].latencies, vec![0.1]);

        let never = Run {
            turn_usage: vec![usage(0, None), usage(1, None)],
            ..run
        };
        assert_eq!(turn_stats(&never)[1].mean_cached_tokens, None);
    }
}
