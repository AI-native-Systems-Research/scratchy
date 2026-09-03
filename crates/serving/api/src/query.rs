// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! HTTP handlers for span-query execution — `POST /v1/query/execute` and the
//! Anthropic `/v1/messages` spans path. The pure span execute/optimize logic
//! lives in [`crate::spans`]; this module only adapts it to axum request/
//! response types (streaming SSE, JSON, error responses). Gated behind `serve`.

use std::collections::BTreeMap;
use std::convert::Infallible;
use std::sync::Arc;

use scratchy_core_common::BlockKind;

use axum::Json;
use axum::extract::{Query, State};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use tokio_stream::StreamExt;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tracing::{error, info};

use spnl_core::ir::{Generate, Query as SpnlQuery};
use spnl_core::optimizer::llo::llir::{Bulk, Repeat, SingleGenerate, SingleGenerateQuery};

use crate::engine::StreamDelta;
use crate::error::{ServeError, ServeResult};
use crate::protocol;
use crate::server::AppState;
#[cfg(feature = "rag")]
use crate::spans::optimize_augments;
use crate::spans::{
    SpanConfig, collect_generates, has_nested_generates, non_generate_input_has_messages,
    outer_generate_to_single, query_variant_name, strip_generates, tokenize_map_input,
    tokenize_span_query,
};
use crate::tokenizer::Tokenizer;

#[derive(Debug, serde::Deserialize)]
pub(crate) struct ExecuteQueryParams {
    #[serde(default)]
    stream: Option<bool>,
}

fn stream_response(
    request_id: String,
    model: String,
    rx: tokio::sync::mpsc::UnboundedReceiver<StreamDelta>,
) -> Sse<impl tokio_stream::Stream<Item = Result<Event, Infallible>>> {
    let stream = UnboundedReceiverStream::new(rx).map(move |delta| {
        let finish_reason_str = delta.finish_reason.map(|r| r.to_string());

        let text = delta.text.unwrap_or_else(|| {
            use std::fmt::Write;
            let mut s = String::new();
            for id in &delta.new_token_ids {
                let _ = write!(s, "<token_{id}>");
            }
            s
        });

        let chunk = protocol::CompletionStreamResponse::new(
            format!("cmpl-{}", request_id),
            model.clone(),
            vec![protocol::CompletionResponseStreamChoice {
                index: delta.index,
                text,
                logprobs: None,
                finish_reason: finish_reason_str,
                stop_reason: delta.stop_reason.map(|sr| match sr {
                    scratchy_core_common::StopReason::Token(id) => {
                        serde_json::Value::Number(id.into())
                    }
                    scratchy_core_common::StopReason::String(s) => serde_json::Value::String(s),
                }),
            }],
        );

        let data = serde_json::to_string(&chunk).unwrap_or_default();
        Ok(Event::default().data(data))
    });

    let done_stream = tokio_stream::once(Ok(Event::default().data("[DONE]")));
    let full_stream = stream.chain(done_stream);
    Sse::new(full_stream).keep_alive(KeepAlive::default())
}

/// POST /v1/query/execute
///
/// Accepts a JSON span query (SPNL format), tokenizes it with the model's
/// tokenizer, and executes it as a completion request.
///
/// Query params: `?stream=true` for SSE streaming.
pub(crate) async fn execute_query(
    State(state): State<Arc<AppState>>,
    Query(params): Query<ExecuteQueryParams>,
    body: String,
) -> Response {
    let stream = params.stream.unwrap_or(false);
    info!("POST /v1/query/execute (stream={})", stream);

    match execute_query_inner(&state, &body, stream).await {
        Ok(response) => response,
        Err(e) => {
            error!("/v1/query/execute failed: {e}");
            e.into_response()
        }
    }
}

async fn execute_query_inner(state: &AppState, body: &str, stream: bool) -> ServeResult<Response> {
    let tokenizer = state
        .engine
        .tokenizer()
        .ok_or_else(|| ServeError::Validation("Tokenizer not available".to_string()))?;

    let template = state
        .engine
        .chat_template()
        .ok_or_else(|| ServeError::Validation("Chat template not available".to_string()))?;

    let block_size = state
        .scratchy_core_config
        .as_ref()
        .map(|c| c.block_size)
        .unwrap_or(16);

    let cfg = SpanConfig::from_tokenizer(block_size, tokenizer);

    // Try to parse as a full SpnlQuery first (supports nested generates).
    // Fall back to SingleGenerateQuery for backward compatibility.
    if let Ok(query) = serde_json::from_str::<SpnlQuery>(body) {
        // RAG: index any Augment nodes, then rewrite them to retrieved fragments.
        #[cfg(feature = "rag")]
        let query = {
            let aug_options = crate::augment::AugmentOptions {
                current_model: Some(state.engine.model_name().to_string()),
                embedder: Some(std::sync::Arc::new(
                    crate::augment::embed::AsyncEngineEmbedder::new(state.engine.clone()),
                )),
                tokenizer: state.engine.tokenizer().cloned(),
                sidecar_manager: Some(std::sync::Arc::new(crate::augment::SidecarManager::new())),
                ..Default::default()
            };
            crate::augment::index(&query, &aug_options)
                .await
                .map_err(|e| ServeError::Validation(format!("RAG indexing failed: {e}")))?;
            optimize_augments(&query, &aug_options)
                .await
                .map_err(|e| ServeError::Validation(format!("RAG retrieval failed: {e}")))?
        };
        return dispatch_spnl_query(state, &query, stream, tokenizer, template, &cfg, block_size)
            .await;
    }

    let query: SingleGenerateQuery = serde_json::from_str(body)
        .map_err(|e| ServeError::Validation(format!("Invalid span query: {e}")))?;

    match query {
        SingleGenerateQuery::SingleGenerate(spec) => {
            execute_single(state, &spec, 1, stream, tokenizer, template, &cfg).await
        }
        SingleGenerateQuery::Bulk(Bulk::Repeat(Repeat { n, generate: spec })) => {
            execute_single(state, &spec, n, stream, tokenizer, template, &cfg).await
        }
        SingleGenerateQuery::Bulk(Bulk::Map(map)) => {
            execute_map(state, &map, stream, tokenizer, template, &cfg).await
        }
    }
}

/// Dispatch a full `SpnlQuery` to the appropriate execution path.
async fn dispatch_spnl_query(
    state: &AppState,
    query: &SpnlQuery,
    stream: bool,
    tokenizer: &Arc<Tokenizer>,
    template: &crate::chat_template::ChatTemplate,
    cfg: &SpanConfig,
    block_size: usize,
) -> ServeResult<Response> {
    match query {
        // Nested generate: outer Generate whose input contains inner Generate nodes.
        SpnlQuery::Generate(g) if has_nested_generates(&g.input) => {
            execute_nested_generate(state, g, stream, tokenizer, template, cfg, block_size).await
        }

        // Flat generate: no nested generates — convert to SingleGenerate and use existing path.
        SpnlQuery::Generate(g) => {
            let spec = outer_generate_to_single(g);
            execute_single(state, &spec, 1, stream, tokenizer, template, cfg).await
        }

        // Seq: execute each generate in sequence, collect all results.
        SpnlQuery::Seq(children) => {
            execute_seq_query(
                state, children, stream, tokenizer, template, cfg, block_size,
            )
            .await
        }

        // Bulk variants at the top level.
        SpnlQuery::Bulk(spnl_core::ir::Bulk::Repeat(r)) => {
            let spec = outer_generate_to_single(&r.generate);
            execute_single(state, &spec, r.n, stream, tokenizer, template, cfg).await
        }
        SpnlQuery::Bulk(spnl_core::ir::Bulk::Map(map)) => {
            execute_map(state, map, stream, tokenizer, template, cfg).await
        }

        other => Err(ServeError::Validation(format!(
            "Unsupported top-level query variant: {}",
            query_variant_name(other)
        ))),
    }
}

/// Execute a `Seq` of queries, collecting all generate results.
async fn execute_seq_query(
    state: &AppState,
    children: &[SpnlQuery],
    stream: bool,
    tokenizer: &Arc<Tokenizer>,
    template: &crate::chat_template::ChatTemplate,
    cfg: &SpanConfig,
    block_size: usize,
) -> ServeResult<Response> {
    let mut steps: Vec<QueryStep> = Vec::new();
    for (i, child) in children.iter().enumerate() {
        match child {
            SpnlQuery::Generate(g) if has_nested_generates(&g.input) => {
                // Nested generate inside a Seq: execute it fully.
                // We can't stream intermediate results, so always use non-streaming here.
                let resp =
                    execute_nested_generate(state, g, false, tokenizer, template, cfg, block_size)
                        .await?;
                // Extract nested steps from the response body.
                let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
                    .await
                    .map_err(|e| ServeError::Internal(format!("body read: {e}")))?;
                let nested: NestedQueryResponse = serde_json::from_slice(&body)
                    .map_err(|e| ServeError::Internal(format!("nested parse: {e}")))?;
                steps.extend(nested.steps);
            }
            SpnlQuery::Generate(g) => {
                let spec = outer_generate_to_single(g);
                let resp = execute_single(state, &spec, 1, false, tokenizer, template, cfg).await?;
                let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
                    .await
                    .map_err(|e| ServeError::Internal(format!("body read: {e}")))?;
                let completion: protocol::CompletionResponse = serde_json::from_slice(&body)
                    .map_err(|e| ServeError::Internal(format!("completion parse: {e}")))?;
                steps.push(QueryStep {
                    label: format!("step[{i}]"),
                    response: completion,
                });
            }
            _ => {
                return Err(ServeError::Validation(format!(
                    "Seq child at index {i} is not a Generate — only Generate nodes are \
                     supported inside a top-level Seq query"
                )));
            }
        }
    }

    if stream {
        // Return steps as a single SSE event (streaming not meaningful for Seq).
        let json = serde_json::to_string(&NestedQueryResponse { steps })
            .map_err(|e| ServeError::Internal(format!("serialize: {e}")))?;
        let event_stream = tokio_stream::once(Ok::<Event, Infallible>(Event::default().data(json)));
        let done_stream = tokio_stream::once(Ok(Event::default().data("[DONE]")));
        Ok(Sse::new(event_stream.chain(done_stream))
            .keep_alive(KeepAlive::default())
            .into_response())
    } else {
        Ok(Json(NestedQueryResponse { steps }).into_response())
    }
}

/// Execute a single (possibly n>1) generation from a tokenized span query.
async fn execute_single(
    state: &AppState,
    spec: &SingleGenerate,
    n: u8,
    stream: bool,
    tokenizer: &Arc<Tokenizer>,
    template: &crate::chat_template::ChatTemplate,
    cfg: &SpanConfig,
) -> ServeResult<Response> {
    let span_tok = tokenize_span_query(spec, tokenizer, template, cfg)?;

    let max_tokens = spec
        .metadata
        .max_tokens
        .filter(|&t| t > 0)
        .map(|t| t as u32)
        .unwrap_or(2048);
    let temperature = spec.metadata.temperature.unwrap_or(0.0);

    let request = build_completion_request(
        &spec.metadata.model,
        protocol::CompletionPrompt::TokenIds(span_tok.tokens),
        span_tok.annotations,
        n.max(1) as u32,
        max_tokens,
        temperature,
        stream,
    );

    if stream {
        let (request_id, model, rx) = state.engine.completion_stream(request).await?;
        Ok(stream_response(request_id, model, rx).into_response())
    } else {
        let response = state.engine.completion(request).await?;
        Ok(Json(response).into_response())
    }
}

/// Run an Anthropic `/v1/messages` span query (built by `anthropic::build_spnl_query`)
/// and return the raw completion response for Anthropic-format conversion.
///
/// Mirrors `execute_query_inner` + `execute_single`, but returns the
/// `CompletionResponse` (non-streaming) rather than an HTTP response so the
/// Anthropic handler can wrap it in `MessagesResponse`. Each tool is a `Plus`
/// child → `BlockKind::Relocatable`, so the prefill attention takes the
/// block-diagonal `[span_lo, q_pos]` bound and the per-tool spans are
/// independently cacheable across launches.
pub(crate) async fn anthropic_spans_completion(
    state: &AppState,
    spnl_json: &str,
) -> ServeResult<protocol::CompletionResponse> {
    let tokenizer = state
        .engine
        .tokenizer()
        .ok_or_else(|| ServeError::Validation("Tokenizer not available".to_string()))?;
    let template = state
        .engine
        .chat_template()
        .ok_or_else(|| ServeError::Validation("Chat template not available".to_string()))?;
    let block_size = state
        .scratchy_core_config
        .as_ref()
        .map(|c| c.block_size)
        .unwrap_or(16);
    let cfg = SpanConfig::from_tokenizer(block_size, tokenizer);

    let query: SpnlQuery = serde_json::from_str(spnl_json)
        .map_err(|e| ServeError::Validation(format!("anthropic spans query: {e}")))?;
    let spec = match query {
        SpnlQuery::Generate(g) => outer_generate_to_single(&g),
        _ => {
            return Err(ServeError::Validation(
                "anthropic spans: expected a Generate at top level".to_string(),
            ));
        }
    };

    let span_tok = tokenize_span_query(&spec, tokenizer, template, &cfg)?;
    let max_tokens = spec
        .metadata
        .max_tokens
        .filter(|&t| t > 0)
        .map(|t| t as u32)
        .unwrap_or(2048);
    let temperature = spec.metadata.temperature.unwrap_or(0.0);
    let request = build_completion_request(
        &spec.metadata.model,
        protocol::CompletionPrompt::TokenIds(span_tok.tokens),
        span_tok.annotations,
        1,
        max_tokens,
        temperature,
        false,
    );
    state.engine.completion(request).await
}

/// One generate step in a nested query result.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub(crate) struct QueryStep {
    /// Human-readable label, e.g. "inner[0]", "outer".
    pub label: String,
    /// The completion response for this step.
    #[serde(flatten)]
    pub response: protocol::CompletionResponse,
}

/// Response from `/v1/query/execute` for a nested (multi-generate) query.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub(crate) struct NestedQueryResponse {
    pub steps: Vec<QueryStep>,
}

/// Execute a `SpnlQuery::Generate` that contains nested inner generates.
///
/// Algorithm (mirrors `bench spans --nested`):
/// 1. Collect all inner `Generate` nodes from `outer_g.input` (DFS order).
/// 2. Execute each inner generate with seal=true, volatile=true.
/// 3. Build the outer prompt:
///    - For each inner: [inner_prompt_tokens] + [re-tokenized output] + [EOS if stop]
///      as a Relocatable block.
///    - If the outer input has non-generate messages: tokenize them as Prefixed.
/// 4. Execute the outer generate.
/// 5. Return a `NestedQueryResponse` with all steps.
async fn execute_nested_generate(
    state: &AppState,
    outer_g: &Generate,
    stream: bool,
    tokenizer: &Arc<Tokenizer>,
    template: &crate::chat_template::ChatTemplate,
    cfg: &SpanConfig,
    block_size: usize,
) -> ServeResult<Response> {
    // 1. Collect inner generates.
    let mut inner_gens: Vec<&Generate> = Vec::new();
    collect_generates(&outer_g.input, &mut inner_gens);

    let mut steps: Vec<QueryStep> = Vec::new();
    let mut outer_tokens: Vec<u32> = Vec::new();
    let mut outer_annotations: BTreeMap<usize, BlockKind> = BTreeMap::new();

    let eos_token_id = tokenizer.eos_token_id();

    // 2. Execute each inner generate.
    for (i, inner_g) in inner_gens.iter().enumerate() {
        // Convert inner Generate's input (Box<SpnlQuery>) to NonGenerateInput.
        // Inner generates must not themselves contain nested generates.
        let inner_input = strip_generates(&inner_g.input);
        let inner_spec = SingleGenerate {
            metadata: inner_g.metadata.clone(),
            input: inner_input,
        };

        // Tokenize the inner generate's prompt (includes generation prompt prefix).
        let inner_tok = tokenize_span_query(&inner_spec, tokenizer, template, cfg)?;
        let inner_prompt_tokens = inner_tok.tokens.clone();

        let max_tokens = inner_g
            .metadata
            .max_tokens
            .filter(|&t| t > 0)
            .map(|t| t as u32)
            .unwrap_or(2048);
        let temperature = inner_g.metadata.temperature.unwrap_or(0.0);

        // Execute inner generate: seal=true so output is block-aligned in KV cache.
        let mut request = build_completion_request(
            &inner_g.metadata.model,
            protocol::CompletionPrompt::TokenIds(inner_tok.tokens),
            inner_tok.annotations,
            1,
            max_tokens,
            temperature,
            false, // never stream inner generates
        );
        request.seal = true;
        request.volatile = true;
        // skip_special_tokens=true (default) so choices[0].text has no EOS/pads.

        let inner_response = state.engine.completion(request).await?;

        // 3a. Reconstruct inner token sequence for the outer prompt.
        //     inner_prompt_tokens + re_tokenize(output_text) + [EOS if stop]
        //     Seal guarantees this is block-aligned.
        let block_idx = outer_tokens.len() / block_size;
        outer_annotations.insert(
            block_idx,
            BlockKind::Relocatable {
                first_token: (block_idx * block_size) as u32,
            },
        );
        outer_tokens.extend_from_slice(&inner_prompt_tokens);

        if let Some(choice) = inner_response.choices.first() {
            // Re-tokenize the decoded output (round-trips correctly for BPE).
            let content_ids = tokenizer.encode(&choice.text, false)?;
            outer_tokens.extend_from_slice(&content_ids);

            // Append EOS if generation stopped naturally (not max_tokens).
            if choice.finish_reason.as_deref() == Some("stop")
                && let Some(eos) = eos_token_id
            {
                outer_tokens.push(eos);
            }
        }

        steps.push(QueryStep {
            label: format!("inner[{i}]"),
            response: inner_response,
        });
    }

    // 3b. Tokenize non-generate messages from the outer input (Prefixed).
    let outer_input_stripped = outer_generate_to_single(outer_g);
    if non_generate_input_has_messages(&outer_input_stripped.input) {
        let msg_start_block = outer_tokens.len() / block_size;
        outer_annotations.insert(
            msg_start_block,
            BlockKind::Prefixed {
                first_token: (msg_start_block * block_size) as u32,
            },
        );

        let msg_tok = tokenize_span_query(&outer_input_stripped, tokenizer, template, cfg)?;
        outer_tokens.extend_from_slice(&msg_tok.tokens);
        // Merge any annotations from msg_tok (offset by msg_start_block).
        if let Some(ann) = msg_tok.annotations {
            for (k, v) in ann {
                outer_annotations.insert(msg_start_block + k, v);
            }
        }
    }

    // 4. Execute the outer generate.
    let outer_max_tokens = outer_g
        .metadata
        .max_tokens
        .filter(|&t| t > 0)
        .map(|t| t as u32)
        .unwrap_or(2048);
    let outer_temperature = outer_g.metadata.temperature.unwrap_or(0.0);

    let outer_request = build_completion_request(
        &outer_g.metadata.model,
        protocol::CompletionPrompt::TokenIds(outer_tokens),
        Some(outer_annotations),
        1,
        outer_max_tokens,
        outer_temperature,
        stream,
    );

    // Nested queries always return a NestedQueryResponse (no streaming support for inner steps).
    let outer_response = state.engine.completion(outer_request).await?;
    steps.push(QueryStep {
        label: "outer".to_string(),
        response: outer_response,
    });
    Ok(Json(NestedQueryResponse { steps }).into_response())
}

/// Execute a map (bulk) completion: one output per input string.
async fn execute_map(
    state: &AppState,
    map: &spnl_core::ir::Map,
    stream: bool,
    tokenizer: &Arc<Tokenizer>,
    template: &crate::chat_template::ChatTemplate,
    cfg: &SpanConfig,
) -> ServeResult<Response> {
    let mut all_ids: Vec<Vec<u32>> = Vec::with_capacity(map.inputs.len());
    // All map inputs get the same annotation structure (single relocatable block 0).
    let mut combined_annotations: Option<BTreeMap<usize, BlockKind>> = None;
    for input_text in &map.inputs {
        let span_tok = tokenize_map_input(input_text, tokenizer, template, cfg)?;
        all_ids.push(span_tok.tokens);
        if combined_annotations.is_none() {
            combined_annotations = span_tok.annotations;
        }
    }

    let max_tokens = map
        .metadata
        .max_tokens
        .filter(|&t| t > 0)
        .map(|t| t as u32)
        .unwrap_or(2048);
    let temperature = map.metadata.temperature.unwrap_or(0.0);

    let request = build_completion_request(
        &map.metadata.model,
        protocol::CompletionPrompt::MultipleTokenIds(all_ids),
        combined_annotations,
        1,
        max_tokens,
        temperature,
        stream,
    );

    if stream {
        let (request_id, model, rx) = state.engine.completion_stream(request).await?;
        Ok(stream_response(request_id, model, rx).into_response())
    } else {
        let response = state.engine.completion(request).await?;
        Ok(Json(response).into_response())
    }
}

/// Build a CompletionRequest from tokenized prompt IDs and generation metadata.
fn build_completion_request(
    model: &str,
    prompt: protocol::CompletionPrompt,
    annotations: Option<BTreeMap<usize, BlockKind>>,
    n: u32,
    max_tokens: u32,
    temperature: f32,
    stream: bool,
) -> protocol::CompletionRequest {
    protocol::CompletionRequest {
        model: Some(model.to_string()),
        prompt: Some(prompt),
        echo: false,
        temperature: Some(temperature as f64),
        top_p: None,
        n,
        max_tokens: Some(max_tokens),
        stream,
        stream_options: None,
        stop: None,
        frequency_penalty: None,
        presence_penalty: None,
        logit_bias: None,
        logprobs: None,
        prompt_logprobs: None,
        suffix: None,
        seed: None,
        user: None,
        top_k: None,
        min_p: None,
        repetition_penalty: None,
        min_tokens: 0,
        stop_token_ids: vec![],
        include_stop_str_in_output: false,
        ignore_eos: false,
        skip_special_tokens: true,
        add_special_tokens: true,
        priority: 0,
        cache_salt: None,
        request_id: None,
        guided_regex: None,
        guided_grammar: None,
        allowed_token_ids: None,
        bad_words: None,
        truncate_prompt_tokens: None,
        block_annotations: annotations,
        seal: false,
        volatile: false,
    }
}
