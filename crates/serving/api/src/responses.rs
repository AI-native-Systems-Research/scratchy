// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! OpenAI Responses API (`/v1/responses`) — translation layer over the
//! internal chat completion engine.

use std::collections::HashMap;
use std::convert::Infallible;
use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio_stream::StreamExt as _;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tracing::info;
use uuid::Uuid;

use scratchy_core_common::FinishReason;

use crate::engine::StreamDelta;
use crate::protocol;
use crate::server::AppState;

// ---------------------------------------------------------------------------
// Request types
// ---------------------------------------------------------------------------

/// Input to the Responses API — either a plain string or array of items.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ResponsesInput {
    Text(String),
    Items(Vec<InputItem>),
}

/// A single input item — tagged by `type`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum InputItem {
    /// A chat message.
    #[serde(rename = "message")]
    Message { role: String, content: InputContent },
    /// An assistant function/tool call.
    #[serde(rename = "function_call")]
    FunctionCall {
        #[serde(default)]
        id: Option<String>,
        call_id: String,
        name: String,
        arguments: String,
    },
    /// The result of a function/tool call.
    #[serde(rename = "function_call_output")]
    FunctionCallOutput { call_id: String, output: String },
}

/// Content of a message input item — string or array of content parts.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum InputContent {
    Text(String),
    Parts(Vec<InputContentPart>),
}

/// A content part within a message.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum InputContentPart {
    #[serde(rename = "input_text")]
    InputText { text: String },
    #[serde(rename = "input_image")]
    InputImage { image_url: Option<String> },
}

/// A function tool definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionToolDef {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub parameters: Option<Value>,
    #[serde(default)]
    pub strict: Option<bool>,
}

/// OpenAI Responses API request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponsesRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub input: ResponsesInput,
    #[serde(default)]
    pub instructions: Option<String>,
    #[serde(default)]
    pub max_output_tokens: Option<u32>,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub top_p: Option<f64>,
    #[serde(default)]
    pub top_k: Option<i32>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub tools: Vec<FunctionToolDef>,
    #[serde(default)]
    pub tool_choice: Option<Value>,
    #[serde(default)]
    pub metadata: Option<Value>,
    #[serde(default)]
    pub stop: Option<StopParam>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub repetition_penalty: Option<f64>,
    #[serde(default)]
    pub logit_bias: Option<HashMap<String, f64>>,
    #[serde(default)]
    pub top_logprobs: Option<u32>,
    #[serde(default)]
    pub text: Option<Value>,
}

/// Stop condition — single string or array.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum StopParam {
    Single(String),
    Multiple(Vec<String>),
}

// ---------------------------------------------------------------------------
// Response types
// ---------------------------------------------------------------------------

/// Top-level Responses API response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponsesResponse {
    pub id: String,
    pub object: String,
    pub created_at: i64,
    pub model: String,
    pub status: String,
    pub output: Vec<OutputItem>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<ResponseUsage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<Value>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<Value>,
}

/// An output item — either a message or a function call.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum OutputItem {
    #[serde(rename = "message")]
    Message {
        id: String,
        role: String,
        content: Vec<OutputContent>,
        status: String,
    },
    #[serde(rename = "function_call")]
    FunctionCall {
        id: String,
        call_id: String,
        name: String,
        arguments: String,
        status: String,
    },
}

/// Content within an output message.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum OutputContent {
    #[serde(rename = "output_text")]
    OutputText {
        text: String,
        #[serde(default)]
        annotations: Vec<Value>,
    },
}

/// Usage information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub total_tokens: u32,
}

fn new_item_id() -> String {
    format!("item_{}", Uuid::new_v4().simple())
}

fn new_resp_id() -> String {
    format!("resp_{}", Uuid::new_v4().simple())
}

// ---------------------------------------------------------------------------
// Conversion: ResponsesRequest → ChatCompletionRequest
// ---------------------------------------------------------------------------

fn convert_request(req: &ResponsesRequest) -> protocol::ChatCompletionRequest {
    let mut messages = Vec::new();

    // instructions → system message
    if let Some(ref instructions) = req.instructions {
        messages.push(protocol::ChatCompletionMessageParam {
            role: "system".to_string(),
            content: Some(Value::String(instructions.clone())),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        });
    }

    // Convert input
    match &req.input {
        ResponsesInput::Text(text) => {
            messages.push(protocol::ChatCompletionMessageParam {
                role: "user".to_string(),
                content: Some(Value::String(text.clone())),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            });
        }
        ResponsesInput::Items(items) => {
            for item in items {
                match item {
                    InputItem::Message { role, content } => {
                        let text = match content {
                            InputContent::Text(t) => t.clone(),
                            InputContent::Parts(parts) => parts
                                .iter()
                                .filter_map(|p| match p {
                                    InputContentPart::InputText { text } => Some(text.as_str()),
                                    _ => None,
                                })
                                .collect::<Vec<_>>()
                                .join("\n"),
                        };
                        messages.push(protocol::ChatCompletionMessageParam {
                            role: role.clone(),
                            content: Some(Value::String(text)),
                            name: None,
                            tool_calls: None,
                            tool_call_id: None,
                        });
                    }
                    InputItem::FunctionCall {
                        call_id,
                        name,
                        arguments,
                        ..
                    } => {
                        messages.push(protocol::ChatCompletionMessageParam {
                            role: "assistant".to_string(),
                            content: None,
                            name: None,
                            tool_calls: Some(vec![protocol::ToolCall {
                                id: call_id.clone(),
                                call_type: "function".to_string(),
                                function: protocol::FunctionCall {
                                    name: name.clone(),
                                    arguments: arguments.clone(),
                                },
                            }]),
                            tool_call_id: None,
                        });
                    }
                    InputItem::FunctionCallOutput { call_id, output } => {
                        messages.push(protocol::ChatCompletionMessageParam {
                            role: "tool".to_string(),
                            content: Some(Value::String(output.clone())),
                            name: None,
                            tool_calls: None,
                            tool_call_id: Some(call_id.clone()),
                        });
                    }
                }
            }
        }
    }

    // Convert tools
    let tools = if req.tools.is_empty() {
        None
    } else {
        Some(
            req.tools
                .iter()
                .filter(|t| t.tool_type == "function")
                .map(|t| protocol::ChatCompletionToolsParam {
                    tool_type: "function".to_string(),
                    function: protocol::FunctionDefinition {
                        name: t.name.clone(),
                        description: t.description.clone(),
                        parameters: t.parameters.clone(),
                    },
                })
                .collect(),
        )
    };

    // Convert tool_choice
    let tool_choice = req.tool_choice.clone().map(|tc| {
        if let Some(s) = tc.as_str() {
            match s {
                "auto" | "none" | "required" => tc,
                name => serde_json::json!({
                    "type": "function",
                    "function": { "name": name }
                }),
            }
        } else {
            tc
        }
    });

    // Convert stop
    let stop = req.stop.as_ref().map(|s| match s {
        StopParam::Single(s) => protocol::StopCondition::Single(s.clone()),
        StopParam::Multiple(v) => protocol::StopCondition::Multiple(v.clone()),
    });

    // Extract json_schema from text.format if present
    let response_format = req.text.as_ref().and_then(|text| {
        let format = text.get("format")?;
        let fmt_type = format.get("type")?.as_str()?;
        match fmt_type {
            "json_schema" => {
                let schema = format.get("json_schema")?;
                let name = schema
                    .get("name")
                    .and_then(|n| n.as_str())
                    .unwrap_or("response")
                    .to_string();
                Some(protocol::ResponseFormat::Standard(
                    protocol::StandardResponseFormat {
                        format_type: "json_schema".to_string(),
                        json_schema: Some(protocol::JsonSchemaResponseFormat {
                            name,
                            description: None,
                            json_schema: Some(schema.clone()),
                            strict: schema.get("strict").and_then(|s| s.as_bool()),
                        }),
                    },
                ))
            }
            "json_object" => Some(protocol::ResponseFormat::Standard(
                protocol::StandardResponseFormat {
                    format_type: "json_object".to_string(),
                    json_schema: None,
                },
            )),
            _ => None,
        }
    });

    protocol::ChatCompletionRequest {
        model: req.model.clone(),
        messages,
        temperature: req.temperature,
        top_p: req.top_p,
        n: 1,
        max_tokens: req.max_output_tokens,
        max_completion_tokens: None,
        stream: false,
        stream_options: None,
        stop,
        frequency_penalty: None,
        presence_penalty: None,
        logit_bias: req.logit_bias.clone(),
        logprobs: None,
        top_logprobs: req.top_logprobs,
        prompt_logprobs: None,
        seed: req.seed.map(|s| s as i64),
        response_format,
        tools,
        tool_choice,
        user: None,
        top_k: req.top_k,
        min_p: None,
        repetition_penalty: req.repetition_penalty,
        min_tokens: 0,
        stop_token_ids: Vec::new(),
        include_stop_str_in_output: false,
        ignore_eos: false,
        skip_special_tokens: true,
        priority: 0,
        cache_salt: None,
        request_id: None,
        guided_regex: None,
        guided_grammar: None,
        allowed_token_ids: None,
        bad_words: None,
        truncate_prompt_tokens: None,
        include_reasoning: true,
        chat_template_kwargs: None,
    }
}

// ---------------------------------------------------------------------------
// Conversion: ChatCompletionResponse → ResponsesResponse
// ---------------------------------------------------------------------------

fn map_status(finish_reason: Option<&str>) -> &'static str {
    match finish_reason {
        Some("length") => "incomplete",
        _ => "completed",
    }
}

fn convert_response(
    resp: protocol::ChatCompletionResponse,
    req: &ResponsesRequest,
) -> ResponsesResponse {
    let choice = resp.choices.into_iter().next();
    let status;
    let mut output = Vec::new();

    match choice {
        Some(c) => {
            status = map_status(c.finish_reason.as_deref());

            // Text content → OutputItem::Message
            let has_text = c.message.content.as_ref().is_some_and(|t| !t.is_empty());
            if has_text {
                output.push(OutputItem::Message {
                    id: new_item_id(),
                    role: "assistant".to_string(),
                    content: vec![OutputContent::OutputText {
                        text: c.message.content.unwrap_or_default(),
                        annotations: vec![],
                    }],
                    status: status.to_string(),
                });
            }

            // Tool calls → OutputItem::FunctionCall
            if let Some(tool_calls) = c.message.tool_calls {
                for tc in tool_calls {
                    output.push(OutputItem::FunctionCall {
                        id: new_item_id(),
                        call_id: tc.id,
                        name: tc.function.name,
                        arguments: tc.function.arguments,
                        status: "completed".to_string(),
                    });
                }
            }
        }
        None => {
            status = "completed";
        }
    }

    let usage = Some(ResponseUsage {
        input_tokens: resp.usage.prompt_tokens,
        output_tokens: resp.usage.completion_tokens.unwrap_or(0),
        total_tokens: resp.usage.total_tokens,
    });

    ResponsesResponse {
        id: new_resp_id(),
        object: "response".to_string(),
        created_at: resp.created,
        model: resp.model,
        status: status.to_string(),
        output,
        usage,
        temperature: req.temperature,
        top_p: req.top_p,
        max_output_tokens: req.max_output_tokens,
        tool_choice: req.tool_choice.clone(),
        tools: req
            .tools
            .iter()
            .map(|t| serde_json::to_value(t).unwrap_or_default())
            .collect(),
    }
}

// ---------------------------------------------------------------------------
// Handler: POST /v1/responses
// ---------------------------------------------------------------------------

/// POST /v1/responses — OpenAI Responses API.
pub async fn responses(
    State(state): State<Arc<AppState>>,
    Json(request): Json<ResponsesRequest>,
) -> Response {
    info!(
        "POST /v1/responses: model={:?}, stream={}",
        request.model, request.stream
    );

    let is_stream = request.stream;
    let mut chat_request = convert_request(&request);

    if is_stream {
        chat_request.stream = true;
        match state.engine.chat_completion_stream(chat_request).await {
            Ok((request_id, model, rx)) => {
                stream_responses(request_id, model, request, rx).into_response()
            }
            Err(e) => e.into_response(),
        }
    } else {
        match state.engine.chat_completion(chat_request).await {
            Ok(chat_resp) => {
                let resp = convert_response(chat_resp, &request);
                Json(resp).into_response()
            }
            Err(e) => e.into_response(),
        }
    }
}

// ---------------------------------------------------------------------------
// Streaming SSE
// ---------------------------------------------------------------------------

fn stream_responses(
    _request_id: String,
    model: String,
    request: ResponsesRequest,
    rx: tokio::sync::mpsc::UnboundedReceiver<StreamDelta>,
) -> Sse<impl tokio_stream::Stream<Item = Result<Event, Infallible>>> {
    let (tx, out_rx) = tokio::sync::mpsc::unbounded_channel::<Result<Event, Infallible>>();

    let resp_id = new_resp_id();
    let msg_item_id = new_item_id();
    let temperature = request.temperature;
    let top_p = request.top_p;
    let max_output_tokens = request.max_output_tokens;
    let tool_choice = request.tool_choice.clone();
    let tools: Vec<Value> = request
        .tools
        .iter()
        .map(|t| serde_json::to_value(t).unwrap_or_default())
        .collect();
    let model = model.clone();

    tokio::spawn(async move {
        let mut seq: u32 = 0;
        let mut started = false;
        let mut text_started = false;
        let mut accumulated_text = String::new();
        let mut output_tokens: u32 = 0;
        let mut func_call_items: Vec<Value> = Vec::new();
        let mut stream = UnboundedReceiverStream::new(rx);

        let mut next_seq = || {
            let s = seq;
            seq += 1;
            s
        };

        let send = |tx: &tokio::sync::mpsc::UnboundedSender<Result<Event, Infallible>>,
                    event_type: &str,
                    data: Value| {
            let _ = tx.send(Ok(Event::default()
                .event(event_type)
                .json_data(data)
                .unwrap()));
        };

        while let Some(delta) = stream.next().await {
            output_tokens += delta.new_token_ids.len() as u32;

            let has_tool_calls = delta.tool_call_deltas.is_some();
            let text = if has_tool_calls {
                None
            } else {
                delta.text.filter(|t| !t.is_empty())
            };

            if !started {
                started = true;

                // response.created
                let base_resp = serde_json::json!({
                    "id": resp_id,
                    "object": "response",
                    "created_at": protocol::unix_timestamp(),
                    "model": model,
                    "status": "queued",
                    "output": [],
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_output_tokens": max_output_tokens,
                    "tool_choice": tool_choice,
                    "tools": tools,
                });

                send(
                    &tx,
                    "response.created",
                    serde_json::json!({
                        "type": "response.created",
                        "sequence_number": next_seq(),
                        "response": base_resp,
                    }),
                );

                // response.in_progress
                let mut in_progress = base_resp.clone();
                in_progress["status"] = Value::String("in_progress".to_string());
                send(
                    &tx,
                    "response.in_progress",
                    serde_json::json!({
                        "type": "response.in_progress",
                        "sequence_number": next_seq(),
                        "response": in_progress,
                    }),
                );

                // output_item.added for the message
                if !has_tool_calls {
                    send(
                        &tx,
                        "response.output_item.added",
                        serde_json::json!({
                            "type": "response.output_item.added",
                            "sequence_number": next_seq(),
                            "output_index": 0,
                            "item": {
                                "id": msg_item_id,
                                "type": "message",
                                "role": "assistant",
                                "content": [],
                                "status": "in_progress"
                            }
                        }),
                    );

                    // content_part.added
                    send(
                        &tx,
                        "response.content_part.added",
                        serde_json::json!({
                            "type": "response.content_part.added",
                            "sequence_number": next_seq(),
                            "output_index": 0,
                            "content_index": 0,
                            "part": {
                                "type": "output_text",
                                "text": "",
                                "annotations": []
                            }
                        }),
                    );
                    text_started = true;
                }
            }

            // Text deltas
            if let Some(ref t) = text {
                if !text_started {
                    // Late start — first delta had no text, now we get text
                    send(
                        &tx,
                        "response.output_item.added",
                        serde_json::json!({
                            "type": "response.output_item.added",
                            "sequence_number": next_seq(),
                            "output_index": 0,
                            "item": {
                                "id": msg_item_id,
                                "type": "message",
                                "role": "assistant",
                                "content": [],
                                "status": "in_progress"
                            }
                        }),
                    );
                    send(
                        &tx,
                        "response.content_part.added",
                        serde_json::json!({
                            "type": "response.content_part.added",
                            "sequence_number": next_seq(),
                            "output_index": 0,
                            "content_index": 0,
                            "part": {
                                "type": "output_text",
                                "text": "",
                                "annotations": []
                            }
                        }),
                    );
                    text_started = true;
                }

                accumulated_text.push_str(t);
                send(
                    &tx,
                    "response.output_text.delta",
                    serde_json::json!({
                        "type": "response.output_text.delta",
                        "sequence_number": next_seq(),
                        "output_index": 0,
                        "content_index": 0,
                        "delta": t,
                    }),
                );
            }

            // Tool call deltas
            if let Some(ref tc_deltas) = delta.tool_call_deltas {
                for tc in tc_deltas {
                    // Track function call arguments accumulation
                    let output_idx = func_call_items.len();
                    if tc.index as usize >= func_call_items.len() {
                        // New function call item
                        let fc_item_id = new_item_id();
                        let call_id = tc.id.clone().unwrap_or_default();
                        let name = tc.function_name.clone().unwrap_or_default();
                        func_call_items.push(serde_json::json!({
                            "id": fc_item_id,
                            "call_id": call_id,
                            "name": name,
                            "arguments": "",
                        }));
                        let base_output_idx = if text_started {
                            output_idx + 1
                        } else {
                            output_idx
                        };
                        send(
                            &tx,
                            "response.output_item.added",
                            serde_json::json!({
                                "type": "response.output_item.added",
                                "sequence_number": next_seq(),
                                "output_index": base_output_idx,
                                "item": {
                                    "id": fc_item_id,
                                    "type": "function_call",
                                    "call_id": call_id,
                                    "name": name,
                                    "arguments": "",
                                    "status": "in_progress"
                                }
                            }),
                        );
                    }

                    if let Some(args) = tc.function_arguments.as_deref().filter(|a| !a.is_empty()) {
                        // Accumulate arguments
                        if let Some(item) = func_call_items.get_mut(tc.index as usize) {
                            let prev = item["arguments"].as_str().unwrap_or_default().to_string();
                            item["arguments"] = Value::String(format!("{}{}", prev, args));
                        }
                        let base_output_idx = if text_started {
                            tc.index as usize + 1
                        } else {
                            tc.index as usize
                        };
                        send(
                            &tx,
                            "response.function_call_arguments.delta",
                            serde_json::json!({
                                "type": "response.function_call_arguments.delta",
                                "sequence_number": next_seq(),
                                "output_index": base_output_idx,
                                "item_id": func_call_items[tc.index as usize]["id"],
                                "delta": args,
                            }),
                        );
                    }
                }
            }

            // Finish
            if delta.finish_reason.is_some() {
                let status = if delta.finish_reason == Some(FinishReason::Length) {
                    "incomplete"
                } else {
                    "completed"
                };

                // Close text content part + message item
                if text_started {
                    send(
                        &tx,
                        "response.output_text.done",
                        serde_json::json!({
                            "type": "response.output_text.done",
                            "sequence_number": next_seq(),
                            "output_index": 0,
                            "content_index": 0,
                            "text": accumulated_text,
                        }),
                    );
                    send(
                        &tx,
                        "response.content_part.done",
                        serde_json::json!({
                            "type": "response.content_part.done",
                            "sequence_number": next_seq(),
                            "output_index": 0,
                            "content_index": 0,
                            "part": {
                                "type": "output_text",
                                "text": accumulated_text,
                                "annotations": []
                            }
                        }),
                    );
                    send(
                        &tx,
                        "response.output_item.done",
                        serde_json::json!({
                            "type": "response.output_item.done",
                            "sequence_number": next_seq(),
                            "output_index": 0,
                            "item": {
                                "id": msg_item_id,
                                "type": "message",
                                "role": "assistant",
                                "content": [{
                                    "type": "output_text",
                                    "text": accumulated_text,
                                    "annotations": []
                                }],
                                "status": status
                            }
                        }),
                    );
                }

                // Close function call items
                for (i, fc_item) in func_call_items.iter().enumerate() {
                    let base_output_idx = if text_started { i + 1 } else { i };
                    send(
                        &tx,
                        "response.function_call_arguments.done",
                        serde_json::json!({
                            "type": "response.function_call_arguments.done",
                            "sequence_number": next_seq(),
                            "output_index": base_output_idx,
                            "item_id": fc_item["id"],
                            "arguments": fc_item["arguments"],
                        }),
                    );
                    send(
                        &tx,
                        "response.output_item.done",
                        serde_json::json!({
                            "type": "response.output_item.done",
                            "sequence_number": next_seq(),
                            "output_index": base_output_idx,
                            "item": {
                                "id": fc_item["id"],
                                "type": "function_call",
                                "call_id": fc_item["call_id"],
                                "name": fc_item["name"],
                                "arguments": fc_item["arguments"],
                                "status": "completed"
                            }
                        }),
                    );
                }

                // Build final output for response.completed
                let mut final_output = Vec::new();
                if text_started {
                    final_output.push(serde_json::json!({
                        "id": msg_item_id,
                        "type": "message",
                        "role": "assistant",
                        "content": [{
                            "type": "output_text",
                            "text": accumulated_text,
                            "annotations": []
                        }],
                        "status": status
                    }));
                }
                for fc_item in &func_call_items {
                    final_output.push(serde_json::json!({
                        "id": fc_item["id"],
                        "type": "function_call",
                        "call_id": fc_item["call_id"],
                        "name": fc_item["name"],
                        "arguments": fc_item["arguments"],
                        "status": "completed"
                    }));
                }

                send(
                    &tx,
                    "response.completed",
                    serde_json::json!({
                        "type": "response.completed",
                        "sequence_number": next_seq(),
                        "response": {
                            "id": resp_id,
                            "object": "response",
                            "created_at": protocol::unix_timestamp(),
                            "model": model,
                            "status": status,
                            "output": final_output,
                            "usage": {
                                "input_tokens": 0,
                                "output_tokens": output_tokens,
                                "total_tokens": output_tokens,
                            },
                            "temperature": temperature,
                            "top_p": top_p,
                            "max_output_tokens": max_output_tokens,
                            "tool_choice": tool_choice,
                            "tools": tools,
                        }
                    }),
                );
            }
        }
    });

    Sse::new(UnboundedReceiverStream::new(out_rx)).keep_alive(KeepAlive::default())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deserialize_simple_text_input() {
        let json = r#"{"input": "Hello", "max_output_tokens": 100}"#;
        let req: ResponsesRequest = serde_json::from_str(json).unwrap();
        match req.input {
            ResponsesInput::Text(t) => assert_eq!(t, "Hello"),
            _ => panic!("expected text input"),
        }
        assert!(!req.stream);
    }

    #[test]
    fn test_deserialize_items_input() {
        let json = r#"{
            "input": [
                {"type": "message", "role": "user", "content": "Hello"},
                {"type": "message", "role": "assistant", "content": "Hi there"},
                {"type": "message", "role": "user", "content": "How are you?"}
            ]
        }"#;
        let req: ResponsesRequest = serde_json::from_str(json).unwrap();
        match req.input {
            ResponsesInput::Items(items) => assert_eq!(items.len(), 3),
            _ => panic!("expected items input"),
        }
    }

    #[test]
    fn test_deserialize_function_call_items() {
        let json = r#"{
            "input": [
                {"type": "message", "role": "user", "content": "What's the weather?"},
                {"type": "function_call", "call_id": "call_1", "name": "get_weather", "arguments": "{\"city\":\"NYC\"}"},
                {"type": "function_call_output", "call_id": "call_1", "output": "72°F"}
            ]
        }"#;
        let req: ResponsesRequest = serde_json::from_str(json).unwrap();
        match req.input {
            ResponsesInput::Items(items) => {
                assert_eq!(items.len(), 3);
                match &items[1] {
                    InputItem::FunctionCall { name, .. } => assert_eq!(name, "get_weather"),
                    _ => panic!("expected function_call"),
                }
                match &items[2] {
                    InputItem::FunctionCallOutput { call_id, output } => {
                        assert_eq!(call_id, "call_1");
                        assert_eq!(output, "72°F");
                    }
                    _ => panic!("expected function_call_output"),
                }
            }
            _ => panic!("expected items input"),
        }
    }

    #[test]
    fn test_deserialize_tools() {
        let json = r#"{
            "input": "Hello",
            "tools": [{
                "type": "function",
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
            }]
        }"#;
        let req: ResponsesRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.tools.len(), 1);
        assert_eq!(req.tools[0].name, "get_weather");
    }

    #[test]
    fn test_convert_simple_text() {
        let req = ResponsesRequest {
            model: Some("test-model".into()),
            input: ResponsesInput::Text("Hello".into()),
            instructions: None,
            max_output_tokens: Some(100),
            temperature: Some(0.7),
            top_p: None,
            top_k: None,
            stream: false,
            tools: vec![],
            tool_choice: None,
            metadata: None,
            stop: None,
            seed: None,
            repetition_penalty: None,
            logit_bias: None,
            top_logprobs: None,
            text: None,
        };
        let chat = convert_request(&req);
        assert_eq!(chat.messages.len(), 1);
        assert_eq!(chat.messages[0].role, "user");
        assert_eq!(chat.max_tokens, Some(100));
        assert_eq!(chat.temperature, Some(0.7));
    }

    #[test]
    fn test_convert_instructions() {
        let req = ResponsesRequest {
            model: None,
            input: ResponsesInput::Text("Hi".into()),
            instructions: Some("Be helpful.".into()),
            max_output_tokens: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stream: false,
            tools: vec![],
            tool_choice: None,
            metadata: None,
            stop: None,
            seed: None,
            repetition_penalty: None,
            logit_bias: None,
            top_logprobs: None,
            text: None,
        };
        let chat = convert_request(&req);
        assert_eq!(chat.messages.len(), 2);
        assert_eq!(chat.messages[0].role, "system");
        assert_eq!(
            chat.messages[0].content,
            Some(Value::String("Be helpful.".into()))
        );
    }

    #[test]
    fn test_convert_function_call_items() {
        let req = ResponsesRequest {
            model: None,
            input: ResponsesInput::Items(vec![
                InputItem::Message {
                    role: "user".into(),
                    content: InputContent::Text("Weather?".into()),
                },
                InputItem::FunctionCall {
                    id: None,
                    call_id: "call_1".into(),
                    name: "get_weather".into(),
                    arguments: r#"{"city":"NYC"}"#.into(),
                },
                InputItem::FunctionCallOutput {
                    call_id: "call_1".into(),
                    output: "72°F".into(),
                },
            ]),
            instructions: None,
            max_output_tokens: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stream: false,
            tools: vec![],
            tool_choice: None,
            metadata: None,
            stop: None,
            seed: None,
            repetition_penalty: None,
            logit_bias: None,
            top_logprobs: None,
            text: None,
        };
        let chat = convert_request(&req);
        assert_eq!(chat.messages.len(), 3);
        assert_eq!(chat.messages[0].role, "user");
        assert_eq!(chat.messages[1].role, "assistant");
        assert!(chat.messages[1].tool_calls.is_some());
        assert_eq!(chat.messages[2].role, "tool");
        assert_eq!(chat.messages[2].tool_call_id.as_deref(), Some("call_1"));
    }

    #[test]
    fn test_convert_tools() {
        let req = ResponsesRequest {
            model: None,
            input: ResponsesInput::Text("Hi".into()),
            instructions: None,
            max_output_tokens: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stream: false,
            tools: vec![FunctionToolDef {
                tool_type: "function".into(),
                name: "search".into(),
                description: Some("Search".into()),
                parameters: Some(serde_json::json!({"type": "object"})),
                strict: None,
            }],
            tool_choice: None,
            metadata: None,
            stop: None,
            seed: None,
            repetition_penalty: None,
            logit_bias: None,
            top_logprobs: None,
            text: None,
        };
        let chat = convert_request(&req);
        let tools = chat.tools.unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].function.name, "search");
    }

    #[test]
    fn test_convert_response_text() {
        let req = ResponsesRequest {
            model: Some("test".into()),
            input: ResponsesInput::Text("Hi".into()),
            instructions: None,
            max_output_tokens: Some(100),
            temperature: Some(1.0),
            top_p: Some(1.0),
            top_k: None,
            stream: false,
            tools: vec![],
            tool_choice: None,
            metadata: None,
            stop: None,
            seed: None,
            repetition_penalty: None,
            logit_bias: None,
            top_logprobs: None,
            text: None,
        };
        let chat_resp = protocol::ChatCompletionResponse {
            id: "chatcmpl-123".into(),
            object: "chat.completion".into(),
            created: 1000,
            model: "test".into(),
            choices: vec![protocol::ChatCompletionResponseChoice {
                index: 0,
                message: protocol::ChatMessage {
                    role: "assistant".into(),
                    content: Some("Hello!".into()),
                    refusal: None,
                    tool_calls: None,
                    reasoning: None,
                },
                logprobs: None,
                finish_reason: Some("stop".into()),
                stop_reason: None,
                prompt_logprobs: None,
            }],
            system_fingerprint: None,
            usage: protocol::UsageInfo {
                prompt_tokens: 10,
                total_tokens: 15,
                completion_tokens: Some(5),
                prompt_tokens_details: None,
            },
        };
        let resp = convert_response(chat_resp, &req);
        assert_eq!(resp.object, "response");
        assert_eq!(resp.status, "completed");
        assert_eq!(resp.output.len(), 1);
        match &resp.output[0] {
            OutputItem::Message { content, .. } => match &content[0] {
                OutputContent::OutputText { text, .. } => assert_eq!(text, "Hello!"),
            },
            _ => panic!("expected message"),
        }
        let usage = resp.usage.unwrap();
        assert_eq!(usage.input_tokens, 10);
        assert_eq!(usage.output_tokens, 5);
    }

    #[test]
    fn test_convert_response_tool_calls() {
        let req = ResponsesRequest {
            model: None,
            input: ResponsesInput::Text("Hi".into()),
            instructions: None,
            max_output_tokens: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stream: false,
            tools: vec![],
            tool_choice: None,
            metadata: None,
            stop: None,
            seed: None,
            repetition_penalty: None,
            logit_bias: None,
            top_logprobs: None,
            text: None,
        };
        let chat_resp = protocol::ChatCompletionResponse {
            id: "chatcmpl-456".into(),
            object: "chat.completion".into(),
            created: 0,
            model: "test".into(),
            choices: vec![protocol::ChatCompletionResponseChoice {
                index: 0,
                message: protocol::ChatMessage {
                    role: "assistant".into(),
                    content: None,
                    refusal: None,
                    tool_calls: Some(vec![protocol::ToolCall {
                        id: "call_1".into(),
                        call_type: "function".into(),
                        function: protocol::FunctionCall {
                            name: "get_weather".into(),
                            arguments: r#"{"city":"NYC"}"#.into(),
                        },
                    }]),
                    reasoning: None,
                },
                logprobs: None,
                finish_reason: Some("tool_calls".into()),
                stop_reason: None,
                prompt_logprobs: None,
            }],
            system_fingerprint: None,
            usage: protocol::UsageInfo {
                prompt_tokens: 20,
                total_tokens: 30,
                completion_tokens: Some(10),
                prompt_tokens_details: None,
            },
        };
        let resp = convert_response(chat_resp, &req);
        assert_eq!(resp.status, "completed");
        assert_eq!(resp.output.len(), 1);
        match &resp.output[0] {
            OutputItem::FunctionCall {
                call_id,
                name,
                arguments,
                ..
            } => {
                assert_eq!(call_id, "call_1");
                assert_eq!(name, "get_weather");
                assert_eq!(arguments, r#"{"city":"NYC"}"#);
            }
            _ => panic!("expected function_call"),
        }
    }

    #[test]
    fn test_convert_response_length_status() {
        let req = ResponsesRequest {
            model: None,
            input: ResponsesInput::Text("Hi".into()),
            instructions: None,
            max_output_tokens: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stream: false,
            tools: vec![],
            tool_choice: None,
            metadata: None,
            stop: None,
            seed: None,
            repetition_penalty: None,
            logit_bias: None,
            top_logprobs: None,
            text: None,
        };
        let chat_resp = protocol::ChatCompletionResponse {
            id: "chatcmpl-789".into(),
            object: "chat.completion".into(),
            created: 0,
            model: "test".into(),
            choices: vec![protocol::ChatCompletionResponseChoice {
                index: 0,
                message: protocol::ChatMessage {
                    role: "assistant".into(),
                    content: Some("truncated...".into()),
                    refusal: None,
                    tool_calls: None,
                    reasoning: None,
                },
                logprobs: None,
                finish_reason: Some("length".into()),
                stop_reason: None,
                prompt_logprobs: None,
            }],
            system_fingerprint: None,
            usage: protocol::UsageInfo {
                prompt_tokens: 5,
                total_tokens: 105,
                completion_tokens: Some(100),
                prompt_tokens_details: None,
            },
        };
        let resp = convert_response(chat_resp, &req);
        assert_eq!(resp.status, "incomplete");
    }

    #[test]
    fn test_response_serialization() {
        let resp = ResponsesResponse {
            id: "resp_123".into(),
            object: "response".into(),
            created_at: 1000,
            model: "test".into(),
            status: "completed".into(),
            output: vec![OutputItem::Message {
                id: "item_1".into(),
                role: "assistant".into(),
                content: vec![OutputContent::OutputText {
                    text: "Hi".into(),
                    annotations: vec![],
                }],
                status: "completed".into(),
            }],
            usage: Some(ResponseUsage {
                input_tokens: 1,
                output_tokens: 1,
                total_tokens: 2,
            }),
            temperature: Some(1.0),
            top_p: Some(1.0),
            max_output_tokens: Some(100),
            tool_choice: None,
            tools: vec![],
        };
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["object"], "response");
        assert_eq!(json["status"], "completed");
        assert_eq!(json["output"][0]["type"], "message");
        assert_eq!(json["output"][0]["content"][0]["type"], "output_text");
        assert_eq!(json["output"][0]["content"][0]["text"], "Hi");
        assert_eq!(json["usage"]["total_tokens"], 2);
    }

    #[test]
    fn test_stop_param_deserialization() {
        let single: StopParam = serde_json::from_str(r#""END""#).unwrap();
        match single {
            StopParam::Single(s) => assert_eq!(s, "END"),
            _ => panic!("expected single"),
        }

        let multi: StopParam = serde_json::from_str(r#"["END", "STOP"]"#).unwrap();
        match multi {
            StopParam::Multiple(v) => assert_eq!(v, vec!["END", "STOP"]),
            _ => panic!("expected multiple"),
        }
    }
}
