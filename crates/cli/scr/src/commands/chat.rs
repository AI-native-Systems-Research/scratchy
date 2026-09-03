// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Interactive `chat` and `complete` CLI subcommands.
//!
//! `scr chat` supports two modes:
//! - **In-process** (with `--model`): loads the model locally using the `LLM`
//!   API and runs inference directly — no server needed.
//! - **Remote** (without `--model`): connects to a running vLLM server's
//!   OpenAI-compatible API for streaming chat completions.
//!
//! `scr complete` always uses remote mode (connects to a running server).

use std::io::{self, BufRead, Write};

use anyhow::{Context, Result};
use serde_json::Value;

use crate::args::ChatArgs;
use crate::args::CompleteArgs;
use crate::http::RemoteClient;

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

fn read_line(prompt: &str) -> Option<String> {
    print!("{prompt}");
    io::stdout().flush().ok();
    let stdin = io::stdin();
    let mut line = String::new();
    match stdin.lock().read_line(&mut line) {
        Ok(0) => None,
        Ok(_) => Some(line.trim_end().to_string()),
        Err(_) => None,
    }
}

// ---------------------------------------------------------------------------
// In-process chat (LLM API)
// ---------------------------------------------------------------------------

/// Collect per-token timestamps for benchmarking.
// Only the in-process (`chat`) path benchmarks; tests construct it too.
#[cfg(any(feature = "chat", test))]
struct BenchStats {
    startup_ms: f64,
    first_token: Option<std::time::Instant>,
    token_times: Vec<std::time::Instant>,
    gen_start: std::time::Instant,
}

#[cfg(any(feature = "chat", test))]
impl BenchStats {
    fn new(startup_ms: f64) -> Self {
        Self {
            startup_ms,
            first_token: None,
            token_times: Vec::new(),
            gen_start: std::time::Instant::now(),
        }
    }

    fn record_token(&mut self) {
        let now = std::time::Instant::now();
        if self.first_token.is_none() {
            self.first_token = Some(now);
        }
        self.token_times.push(now);
    }

    fn print(&self, num_output_tokens: usize) {
        eprintln!();
        eprintln!("--- bench ---");
        eprintln!("startup     : {:.1} ms", self.startup_ms);

        if let Some(first) = self.first_token {
            let ttft = first.duration_since(self.gen_start).as_secs_f64() * 1000.0;
            eprintln!("TTFT        : {ttft:.1} ms");
        }

        if self.token_times.len() >= 2 {
            let itls: Vec<f64> = self
                .token_times
                .windows(2)
                .map(|w| w[1].duration_since(w[0]).as_secs_f64() * 1000.0)
                .collect();
            let mean_itl = itls.iter().sum::<f64>() / itls.len() as f64;
            eprintln!("mean ITL    : {mean_itl:.1} ms");
            // ⛔⛔⛔ THE MEAN HID A BIMODAL DISTRIBUTION FOR A WEEK. Repeat runs of the same prompt land in one
            // of two clusters — measured 25.39 ms and 26.37 ms with a 0.99 ms gap and NOTHING between them, and
            // only ~2 runs in 10 reach the fast one. A mean averages the two together, so "26.3" looked like
            // noise around one number instead of one draw from two states.
            //
            // ⭐ AND THE PER-TOKEN INTERVALS WERE ALREADY HERE AND THROWN AWAY: `token_times` holds every
            // timestamp. Printing the SHAPE costs nothing and is the difference between "a run is slow" and
            // "some tokens are slow", which have different causes and different fixes.
            let mut sorted = itls.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).expect("no NaN in a duration"));
            let pct = |q: f64| sorted[((sorted.len() - 1) as f64 * q).round() as usize];
            eprintln!(
                "  ITL p10/p50/p90 : {:.2} / {:.2} / {:.2} ms   min {:.2}  max {:.2}",
                pct(0.10),
                pct(0.50),
                pct(0.90),
                sorted[0],
                sorted[sorted.len() - 1]
            );
            // A histogram over the observed range, so two clusters are visible as two humps rather than
            // inferred from three quantiles.
            let (lo, hi) = (sorted[0], sorted[sorted.len() - 1]);
            if hi > lo {
                const BINS: usize = 12;
                let mut h = [0usize; BINS];
                for v in &itls {
                    let i = (((v - lo) / (hi - lo)) * (BINS - 1) as f64).round() as usize;
                    h[i.min(BINS - 1)] += 1;
                }
                let peak = h.iter().copied().max().unwrap_or(1).max(1);
                for (i, c) in h.iter().enumerate() {
                    if *c == 0 {
                        continue;
                    }
                    let edge = lo + (hi - lo) * i as f64 / (BINS - 1) as f64;
                    eprintln!(
                        "  {edge:6.2} ms |{} {c}",
                        "#".repeat((*c * 24 / peak).max(1))
                    );
                }
            }
        }

        // Decode throughput: exclude TTFT, measure from first to last token.
        if let (Some(first), Some(last)) = (self.first_token, self.token_times.last()) {
            let decode_secs = last.duration_since(first).as_secs_f64();
            // num_output_tokens includes the first token, but decode_secs
            // starts after the first token, so we measure (N-1) intervals.
            if decode_secs > 0.0 && num_output_tokens > 1 {
                let tps = (num_output_tokens - 1) as f64 / decode_secs;
                eprintln!("tok/sec     : {tps:.1}");
            }
        }

        eprintln!("output toks : {num_output_tokens}");
        eprintln!("-------------");
    }
}

#[cfg(feature = "chat")]
fn run_chat_inproc(args: &ChatArgs, model: &str) -> Result<()> {
    use scratchy_serving_api::llm::{ChatMessage, LLM};

    // `init_tracing` honors `RUST_LOG` when set; "info" is just the default level so the
    // startup milestones (session ready, ladder built, KV pool sized, ...) still print without
    // it, matching `serve`/`batch`/`convert`. Set `RUST_LOG=debug` for the per-rung detail.
    scratchy_core_common::telemetry::init_tracing("info");

    let t0 = std::time::Instant::now();

    let mut builder = LLM::builder(model).device(&args.device).dtype(&args.dtype);
    // Chat is strictly sequential — one blocking chat_stream call at a
    // time (REPL, -q, multi-prompt, and --bench alike), so exactly one
    // sequence is ever running. Declare that instead of inheriting the
    // server default (256): hybrid GDN arches reserve a recurrent-state
    // slot per max_num_seqs up-front (~61 MiB/slot on Qwen3.5-35B —
    // 15.7 GiB at the default, which is the difference between the 35B
    // fitting on a 32 GiB box or failing the budget guard).
    builder = builder.max_num_seqs(1);
    if let Some(ref token) = args.hf_token {
        builder = builder.hf_token(token);
    }
    if let Some(ref gguf) = args.gguf_file {
        builder = builder.gguf_file(gguf);
    }
    if let Some(len) = args.max_model_len {
        builder = builder.max_model_len(len);
    }
    // VLLM_GPU_MEMORY_UTILIZATION env override: lets perf-debugging
    // workflows shrink the KV cache without touching the API. Defaults
    // to 0.9 (LLM builder default) when unset.
    if let Ok(s) = std::env::var("VLLM_GPU_MEMORY_UTILIZATION")
        && let Ok(f) = s.parse::<f64>()
    {
        builder = builder.gpu_memory_utilization(f);
    }
    builder = builder.tensor_parallel_size(args.tensor_parallel_size);
    builder = builder.enforce_eager(args.enforce_eager);
    if let Some(ref tpl) = args.chat_template {
        builder = builder.chat_template(tpl.clone());
    }
    let mut llm = builder.build()?;
    let startup_ms = t0.elapsed().as_secs_f64() * 1000.0;

    println!("Using model: {}", llm.model_name());

    let mut conversation: Vec<ChatMessage> = Vec::new();
    if let Some(ref system_prompt) = args.system_prompt {
        conversation.push(ChatMessage::system(system_prompt));
    }

    // Match Python's `vllm chat`: omit max_tokens so the server resolves it
    // to the full remaining context window, letting the model emit EOS
    // naturally. Base = the model's generation_config.json defaults
    // (temperature/top_p/top_k/min_p/repetition_penalty — e.g. Qwen3.5
    // thinkers ship 1.0/0.95/20 and loop endlessly without them);
    // explicit CLI flags override.
    let params = {
        let mut p = llm.default_sampling_params();
        // None => resolve to the full remaining window. Deliberately NOT
        // `.or(p.max_tokens)`: the struct default is the OpenAI 16,
        // which would silently cap chat at 16 tokens. Only an explicit
        // model recommendation (generation_config max_new_tokens) wins.
        p.max_tokens = args.max_tokens.or(llm.generation_max_new_tokens());
        if let Some(t) = args.temperature {
            p.temperature = t;
        }
        Some(p)
    };

    // Non-interactive mode: --prompt (multi-turn) or --quick (single turn).
    let prompts: Vec<String> = if !args.prompt.is_empty() {
        args.prompt.clone()
    } else if let Some(ref q) = args.quick {
        vec![q.clone()]
    } else {
        vec![]
    };
    if !prompts.is_empty() {
        // In --bench mode, do an untimed warmup first.
        if args.bench {
            conversation.push(ChatMessage::user(&prompts[0]));
            eprint!("(warmup) ");
            let warmup_params = scratchy_serving_api::llm::SamplingParams {
                max_tokens: Some(1),
                ..Default::default()
            };
            // ⛔ NOT `let _ =`. The warmup runs the SAME forward the timed prompt will, so an error
            // here is the model failing, not a benchmark nicety that can be skipped. Discarding it
            // left the failed request RUNNING and the next prompt unschedulable, which reached the
            // user as a HANG instead of the message the executor had already written.
            llm.chat_stream(&conversation, Some(warmup_params), |_| {})
                .context("bench warmup forward failed")?;
            eprintln!("done");
            conversation.pop();
        }

        for (i, message) in prompts.iter().enumerate() {
            conversation.push(ChatMessage::user(message));

            let mut stats = args.bench.then(|| BenchStats::new(startup_ms));

            if prompts.len() > 1 {
                eprintln!("[turn {}] {}", i + 1, message);
            }
            let output = llm.chat_stream(&conversation, params.clone(), |token| {
                print!("{token}");
                io::stdout().flush().ok();
                if let Some(ref mut s) = stats {
                    s.record_token();
                }
            })?;
            println!();

            // DIAGNOSTIC: print token IDs + finish reason so we can
            // see what the model actually produced (vs garbage tokens
            // or EOS).
            if std::env::var_os("VLLM_PRINT_TOKEN_IDS").is_some() {
                eprintln!(
                    "[diag] token_ids={:?} text={:?} finish_reason={:?}",
                    output.outputs[0].token_ids,
                    output.outputs[0].text,
                    output.outputs[0].finish_reason,
                );
            }

            if let Some(s) = stats {
                s.print(output.outputs[0].token_ids.len());
            }

            conversation.push(ChatMessage::assistant(&output.outputs[0].text));
        }
        return Ok(());
    }

    println!("Please enter a message for the chat model:");
    while let Some(input) = read_line("> ") {
        if input.is_empty() {
            continue;
        }
        conversation.push(ChatMessage::user(&input));

        let mut stats = args.bench.then(|| BenchStats::new(startup_ms));

        let output = llm.chat_stream(&conversation, params.clone(), |token| {
            print!("{token}");
            io::stdout().flush().ok();
            if let Some(ref mut s) = stats {
                s.record_token();
            }
        })?;
        println!();

        if let Some(s) = stats {
            s.print(output.outputs[0].token_ids.len());
        }

        conversation.push(ChatMessage::assistant(&output.outputs[0].text));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Remote chat (OpenAI-compatible API)
// ---------------------------------------------------------------------------

fn build_client(api_key: &str) -> RemoteClient {
    RemoteClient::new(api_key)
}

fn resolve_model_remote(
    client: &RemoteClient,
    base_url: &str,
    explicit: Option<&str>,
) -> Result<String> {
    if let Some(name) = explicit {
        return Ok(name.to_string());
    }
    let body = client
        .get_json(&format!("{base_url}/models"))
        .context("failed to list models from server")?;
    body["data"][0]["id"]
        .as_str()
        .map(|s| s.to_string())
        .context("no models available on the server")
}

/// Stream SSE using chunked transfer — reads line-by-line from the response
/// body for true streaming output.
fn stream_sse_chunked(
    mut reader: impl std::io::Read,
    extract_content: fn(&Value) -> Option<&str>,
) -> Result<String> {
    let mut output = String::new();
    let mut buf = String::new();
    let mut chunk = [0u8; 8192];

    loop {
        let n = reader
            .read(&mut chunk)
            .context("error reading SSE stream")?;
        if n == 0 {
            break;
        }
        buf.push_str(&String::from_utf8_lossy(&chunk[..n]));

        while let Some(newline_pos) = buf.find('\n') {
            let line = buf[..newline_pos].trim().to_string();
            buf = buf[newline_pos + 1..].to_string();

            if !line.starts_with("data: ") {
                continue;
            }
            let data = &line[6..];
            if data == "[DONE]" {
                println!();
                return Ok(output);
            }
            if let Ok(chunk) = serde_json::from_str::<Value>(data)
                && let Some(content) = extract_content(&chunk)
            {
                output.push_str(content);
                print!("{content}");
                io::stdout().flush().ok();
            }
        }
    }
    println!();
    Ok(output)
}

fn extract_chat_content(chunk: &Value) -> Option<&str> {
    chunk["choices"][0]["delta"]["content"].as_str()
}

fn extract_completion_text(chunk: &Value) -> Option<&str> {
    chunk["choices"][0]["text"].as_str()
}

fn run_chat_remote(args: &ChatArgs) -> Result<()> {
    let api_key = args
        .api_key
        .as_deref()
        .or(std::env::var("OPENAI_API_KEY").ok().as_deref())
        .unwrap_or("EMPTY")
        .to_string();

    let client = build_client(&api_key);
    let model = resolve_model_remote(&client, &args.url, args.model_name.as_deref())?;
    println!("Using model: {model}");

    let mut conversation: Vec<Value> = Vec::new();
    if let Some(ref system_prompt) = args.system_prompt {
        conversation.push(serde_json::json!({
            "role": "system",
            "content": system_prompt,
        }));
    }

    let mut chat_body = serde_json::json!({
        "model": model,
        "stream": true,
    });
    if let Some(mt) = args.max_tokens {
        chat_body["max_tokens"] = serde_json::json!(mt);
    }

    if let Some(ref message) = args.quick {
        conversation.push(serde_json::json!({
            "role": "user",
            "content": message,
        }));
        chat_body["messages"] = serde_json::json!(conversation);
        let resp = client
            .post_stream(&format!("{}/chat/completions", args.url), &chat_body)
            .context("failed to send chat completion request")?;
        stream_sse_chunked(resp, extract_chat_content)?;
        return Ok(());
    }

    println!("Please enter a message for the chat model:");
    while let Some(input) = read_line("> ") {
        if input.is_empty() {
            continue;
        }
        conversation.push(serde_json::json!({
            "role": "user",
            "content": input,
        }));
        chat_body["messages"] = serde_json::json!(conversation);
        let resp = client
            .post_stream(&format!("{}/chat/completions", args.url), &chat_body)
            .context("failed to send chat completion request")?;
        let output = stream_sse_chunked(resp, extract_chat_content)?;
        conversation.push(serde_json::json!({
            "role": "assistant",
            "content": output,
        }));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

pub async fn run_chat(args: ChatArgs) -> Result<()> {
    // Local in-process mode needs the engine (`chat` feature). When a model is
    // resolved, run it locally; otherwise fall through to the remote client.
    #[cfg(feature = "chat")]
    if let Some(ref model) = args.resolved_model() {
        let model = model.clone();
        return tokio::task::spawn_blocking(move || run_chat_inproc(&args, &model))
            .await
            .context("chat task panicked")?;
    }
    // No local model (or no `chat` engine): talk to a running server.
    // Blocking end to end (see crate::http) — and this REPL already blocked on
    // `read_line`, so it belongs off the runtime either way.
    tokio::task::spawn_blocking(move || run_chat_remote(&args))
        .await
        .context("chat task panicked")?
}

pub async fn run_complete(args: CompleteArgs) -> Result<()> {
    tokio::task::spawn_blocking(move || run_complete_blocking(args))
        .await
        .context("complete task panicked")?
}

fn run_complete_blocking(args: CompleteArgs) -> Result<()> {
    let api_key = args
        .api_key
        .as_deref()
        .or(std::env::var("OPENAI_API_KEY").ok().as_deref())
        .unwrap_or("EMPTY")
        .to_string();

    let client = build_client(&api_key);
    let model = resolve_model_remote(&client, &args.url, args.model_name.as_deref())?;
    println!("Using model: {model}");

    let mut body = serde_json::json!({
        "model": model,
        "stream": true,
    });
    if let Some(max_tokens) = args.max_tokens {
        body["max_tokens"] = serde_json::json!(max_tokens);
    }

    if let Some(ref prompt) = args.quick {
        body["prompt"] = serde_json::json!(prompt);
        let resp = client
            .post_stream(&format!("{}/completions", args.url), &body)
            .context("failed to send completion request")?;
        stream_sse_chunked(resp, extract_completion_text)?;
        return Ok(());
    }

    println!("Please enter prompt to complete:");
    while let Some(input) = read_line("> ") {
        if input.is_empty() {
            continue;
        }
        body["prompt"] = serde_json::json!(input);
        let resp = client
            .post_stream(&format!("{}/completions", args.url), &body)
            .context("failed to send completion request")?;
        stream_sse_chunked(resp, extract_completion_text)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_chat_content() {
        let chunk = serde_json::json!({
            "choices": [{"delta": {"content": "Hello"}}]
        });
        assert_eq!(extract_chat_content(&chunk), Some("Hello"));

        let empty = serde_json::json!({"choices": [{"delta": {}}]});
        assert_eq!(extract_chat_content(&empty), None);
    }

    #[test]
    fn test_extract_completion_text() {
        let chunk = serde_json::json!({
            "choices": [{"text": "world"}]
        });
        assert_eq!(extract_completion_text(&chunk), Some("world"));

        let null_text = serde_json::json!({"choices": [{"text": null}]});
        assert_eq!(extract_completion_text(&null_text), None);
    }

    #[test]
    fn test_bench_stats_no_tokens() {
        let stats = BenchStats::new(100.0);
        assert!(stats.first_token.is_none());
        assert!(stats.token_times.is_empty());
        assert!((stats.startup_ms - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_bench_stats_records_tokens() {
        let mut stats = BenchStats::new(50.0);
        stats.record_token();
        assert!(stats.first_token.is_some());
        assert_eq!(stats.token_times.len(), 1);

        stats.record_token();
        stats.record_token();
        assert_eq!(stats.token_times.len(), 3);
        // First token should not change after initial set.
        let first = stats.first_token.unwrap();
        assert!(stats.token_times[0] == first);
    }

    #[test]
    fn test_bench_stats_print_does_not_panic() {
        // With 0 tokens.
        let stats = BenchStats::new(10.0);
        stats.print(0);

        // With some tokens.
        let mut stats = BenchStats::new(10.0);
        stats.record_token();
        std::thread::sleep(std::time::Duration::from_millis(1));
        stats.record_token();
        stats.print(2);
    }
}
