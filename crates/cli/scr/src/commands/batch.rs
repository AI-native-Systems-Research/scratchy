// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! `scr batch` subcommand — offline batch processing of OpenAI-compatible requests.
//!
//! Reads a JSONL input file, processes all requests through the engine, and
//! writes results to a JSONL output file. Matches the Python vLLM batch API.

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use scratchy_core_common::telemetry;
use scratchy_serving_api::engine::AsyncEngine;
use scratchy_serving_api::init::{VllmConfig, initialize_stack};
use scratchy_serving_api::protocol::{
    BatchRequestInput, BatchRequestOutput, BatchResponseData, ChatCompletionRequest,
    CompletionRequest, EmbeddingRequest,
};
use tracing::info;
use uuid::Uuid;

use crate::args::BatchArgs;

/// Run the batch subcommand from CLI args.
pub async fn run_batch(args: BatchArgs) -> Result<()> {
    telemetry::init_tracing(&args.log_level);

    let model = args.resolved_model().map_err(|e| anyhow::anyhow!(e))?;

    info!("vLLM Rust — batch mode");
    info!(
        "Model: {}, device: {}, dtype: {}",
        model, args.device, args.dtype
    );
    info!("Input: {}, Output: {}", args.input, args.output);

    let config = VllmConfig {
        model,
        device: args.device,
        dtype: args.dtype,
        gpu_memory_utilization: args.gpu_memory_utilization,
        max_num_seqs: args.max_num_seqs,
        max_num_batched_tokens: args.max_num_batched_tokens,
        disable_async_scheduling: args.disable_async_scheduling,
        enable_prefix_caching: !args.no_prefix_caching,
        hf_token: args.hf_token,
        gguf_file: args.gguf_file,
        ..VllmConfig::default()
    };

    run_batch_from_config(
        config,
        &args.input,
        &args.output,
        args.tool_call_parser.as_deref(),
        args.reasoning_parser.as_deref(),
        args.default_chat_template_kwargs,
    )
    .await
}

/// Run batch processing from a config — callable from both CLI and E2E tests.
pub async fn run_batch_from_config(
    config: VllmConfig,
    input_path: &str,
    output_path: &str,
    tool_call_parser: Option<&str>,
    reasoning_parser: Option<&str>,
    default_chat_template_kwargs: Option<std::collections::HashMap<String, serde_json::Value>>,
) -> Result<()> {
    let start = Instant::now();

    // 1. Read and parse input JSONL.
    let input_content = std::fs::read_to_string(input_path)
        .with_context(|| format!("failed to read input file: {input_path}"))?;
    let requests = parse_input_jsonl(&input_content)?;
    let num_requests = requests.len();
    info!("Parsed {} batch request(s)", num_requests);

    if requests.is_empty() {
        std::fs::write(output_path, "")
            .with_context(|| format!("failed to write output file: {output_path}"))?;
        println!("Batch complete: 0 requests processed");
        return Ok(());
    }

    // 2. Initialize the engine stack.
    let mut stack = tokio::task::spawn_blocking(move || initialize_stack(&config, None))
        .await
        .expect("initialize_stack panicked")?;

    // 2b. Configure tool call parser if specified.
    if let Some(parser_name) = tool_call_parser {
        let parser = scratchy_serving_api::tool_parser::get_tool_parser(parser_name)
            .map_err(|e| anyhow::anyhow!(e))?;
        Arc::get_mut(&mut stack.engine)
            .expect("engine should not be shared yet")
            .set_tool_parser(parser);
    }

    // 2c. Configure reasoning parser if specified.
    if let Some(parser_name) = reasoning_parser {
        let vocab = stack
            .engine
            .tokenizer()
            .ok_or_else(|| anyhow::anyhow!("reasoning parser requires a tokenizer"))?
            .get_vocab();
        let parser =
            scratchy_serving_api::reasoning_parser::get_reasoning_parser(parser_name, &vocab)
                .map_err(|e| anyhow::anyhow!(e))?;
        Arc::get_mut(&mut stack.engine)
            .expect("engine should not be shared yet")
            .set_reasoning_parser(parser);
    }

    // 2d. Configure default chat template kwargs if specified.
    if let Some(kwargs) = default_chat_template_kwargs {
        Arc::get_mut(&mut stack.engine)
            .expect("engine should not be shared yet")
            .set_default_chat_template_kwargs(kwargs);
    }

    // 3. Spawn the engine step loop.
    let _step_handle = stack.engine.spawn_step_loop();

    // 4. Process all requests concurrently.
    let engine = stack.engine.clone();
    let mut handles = Vec::with_capacity(num_requests);

    for req_input in requests {
        let engine = engine.clone();
        handles.push(tokio::spawn(async move {
            process_single_request(&engine, req_input).await
        }));
    }

    // 5. Collect results.
    let mut outputs = Vec::with_capacity(num_requests);
    let mut succeeded = 0usize;
    let mut failed = 0usize;

    for handle in handles {
        let output = handle.await.expect("task panicked");
        if output.error.is_some() {
            failed += 1;
        } else {
            succeeded += 1;
        }
        outputs.push(output);
    }

    // 6. Write output JSONL.
    write_output_jsonl(output_path, &outputs)?;

    let elapsed = start.elapsed();
    println!(
        "Batch complete: {} requests ({} succeeded, {} failed) in {:.2}s",
        num_requests,
        succeeded,
        failed,
        elapsed.as_secs_f64()
    );

    Ok(())
}

/// Parse a JSONL string into a list of batch requests.
fn parse_input_jsonl(content: &str) -> Result<Vec<BatchRequestInput>> {
    let mut requests = Vec::new();
    for (i, line) in content.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let req: BatchRequestInput = serde_json::from_str(line)
            .with_context(|| format!("failed to parse line {} as BatchRequestInput", i + 1))?;
        requests.push(req);
    }
    Ok(requests)
}

/// Write batch outputs as JSONL to a file.
fn write_output_jsonl(path: &str, outputs: &[BatchRequestOutput]) -> Result<()> {
    // Ensure parent directory exists.
    if let Some(parent) = Path::new(path).parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create output directory: {}", parent.display()))?;
    }

    let mut content = String::new();
    for output in outputs {
        let line =
            serde_json::to_string(output).context("failed to serialize BatchRequestOutput")?;
        content.push_str(&line);
        content.push('\n');
    }
    std::fs::write(path, &content)
        .with_context(|| format!("failed to write output file: {path}"))?;
    Ok(())
}

/// Process a single batch request, dispatching to the appropriate engine method.
async fn process_single_request(
    engine: &AsyncEngine,
    input: BatchRequestInput,
) -> BatchRequestOutput {
    let batch_id = format!("batch-{}", Uuid::new_v4());
    let request_id = Uuid::new_v4().to_string();

    match dispatch_request(engine, &input, &request_id).await {
        Ok(body) => BatchRequestOutput {
            id: batch_id,
            custom_id: input.custom_id,
            response: Some(BatchResponseData {
                status_code: 200,
                request_id,
                body: Some(body),
            }),
            error: None,
        },
        Err(e) => BatchRequestOutput {
            id: batch_id,
            custom_id: input.custom_id,
            response: None,
            error: Some(serde_json::json!({
                "message": e.to_string(),
                "type": "batch_processing_error",
                "code": 400,
            })),
        },
    }
}

/// Dispatch a request to the appropriate engine method based on the URL.
async fn dispatch_request(
    engine: &AsyncEngine,
    input: &BatchRequestInput,
    _request_id: &str,
) -> Result<serde_json::Value> {
    let url = input.url.as_str();

    match url {
        "/v1/chat/completions" => {
            let mut req: ChatCompletionRequest = serde_json::from_value(input.body.clone())
                .context("failed to parse chat completion request body")?;
            // Disable streaming for batch.
            req.stream = false;
            let resp = engine
                .chat_completion(req)
                .await
                .map_err(|e| anyhow::anyhow!("chat completion failed: {e}"))?;
            serde_json::to_value(&resp).context("failed to serialize chat completion response")
        }
        "/v1/completions" => {
            let mut req: CompletionRequest = serde_json::from_value(input.body.clone())
                .context("failed to parse completion request body")?;
            req.stream = false;
            let resp = engine
                .completion(req)
                .await
                .map_err(|e| anyhow::anyhow!("completion failed: {e}"))?;
            serde_json::to_value(&resp).context("failed to serialize completion response")
        }
        "/v1/embeddings" => {
            let req: EmbeddingRequest = serde_json::from_value(input.body.clone())
                .context("failed to parse embedding request body")?;
            let resp = engine
                .embeddings(req)
                .await
                .map_err(|e| anyhow::anyhow!("embeddings failed: {e}"))?;
            serde_json::to_value(&resp).context("failed to serialize embedding response")
        }
        _ => anyhow::bail!("unsupported batch URL: {url}"),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_input_jsonl_empty() {
        let requests = parse_input_jsonl("").unwrap();
        assert!(requests.is_empty());
    }

    #[test]
    fn test_parse_input_jsonl_with_blanks() {
        let content = r#"
{"custom_id":"r1","method":"POST","url":"/v1/chat/completions","body":{"messages":[]}}

{"custom_id":"r2","method":"POST","url":"/v1/completions","body":{"prompt":"Hello"}}
"#;
        let requests = parse_input_jsonl(content).unwrap();
        assert_eq!(requests.len(), 2);
        assert_eq!(requests[0].custom_id, "r1");
        assert_eq!(requests[1].custom_id, "r2");
    }

    #[test]
    fn test_parse_input_jsonl_invalid_line() {
        let content = "not valid json\n";
        assert!(parse_input_jsonl(content).is_err());
    }

    #[test]
    fn test_write_output_jsonl() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("out.jsonl");
        let path_str = path.to_str().unwrap();

        let outputs = vec![
            BatchRequestOutput {
                id: "b1".to_string(),
                custom_id: "r1".to_string(),
                response: Some(BatchResponseData {
                    status_code: 200,
                    request_id: "req-1".to_string(),
                    body: Some(serde_json::json!({"result": true})),
                }),
                error: None,
            },
            BatchRequestOutput {
                id: "b2".to_string(),
                custom_id: "r2".to_string(),
                response: None,
                error: Some(serde_json::json!({"message": "failed"})),
            },
        ];

        write_output_jsonl(path_str, &outputs).unwrap();

        let content = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = content.lines().collect();
        assert_eq!(lines.len(), 2);

        let parsed: BatchRequestOutput = serde_json::from_str(lines[0]).unwrap();
        assert_eq!(parsed.custom_id, "r1");
        assert!(parsed.response.is_some());

        let parsed: BatchRequestOutput = serde_json::from_str(lines[1]).unwrap();
        assert_eq!(parsed.custom_id, "r2");
        assert!(parsed.error.is_some());
    }
}
