// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! OpenAI-compatible API protocol types.
//!
//! These types map to the OpenAI API request/response schemas and are used
//! for JSON serialization/deserialization of HTTP request and response bodies.
//!
//! Port of:
//! - `vllm/entrypoints/openai/engine/protocol.py`
//! - `vllm/entrypoints/openai/chat_completion/protocol.py`
//! - `vllm/entrypoints/openai/completion/protocol.py`
//! - `vllm/entrypoints/openai/models/protocol.py`

use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use uuid::Uuid;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn random_uuid() -> String {
    Uuid::new_v4().to_string()
}

pub fn unix_timestamp() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

/// OpenAI-style error information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorInfo {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub param: Option<String>,
    pub code: u16,
}

/// OpenAI-style error response wrapper.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorResponse {
    pub error: ErrorInfo,
}

impl ErrorResponse {
    /// Create a new error response.
    pub fn new(message: impl Into<String>, error_type: impl Into<String>, code: u16) -> Self {
        Self {
            error: ErrorInfo {
                message: message.into(),
                error_type: error_type.into(),
                param: None,
                code,
            },
        }
    }
}

// ---------------------------------------------------------------------------
// Usage
// ---------------------------------------------------------------------------

/// Token usage information for prompt tokens.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PromptTokenUsageInfo {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cached_tokens: Option<u32>,
}

/// Token usage information.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UsageInfo {
    pub prompt_tokens: u32,
    pub total_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tokens_details: Option<PromptTokenUsageInfo>,
}

// ---------------------------------------------------------------------------
// Stream options
// ---------------------------------------------------------------------------

/// Options for streaming responses.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StreamOptions {
    #[serde(default)]
    pub include_usage: Option<bool>,
    #[serde(default)]
    pub continuous_usage_stats: Option<bool>,
}

// ---------------------------------------------------------------------------
// Response format
// ---------------------------------------------------------------------------

/// JSON schema definition for structured output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonSchemaResponseFormat {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub json_schema: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

/// A single structural tag definition (legacy format).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuralTag {
    pub begin: String,
    #[serde(alias = "schema")]
    pub structural_tag_schema: Option<serde_json::Value>,
    pub end: String,
}

/// Legacy structural tag response format with explicit structures + triggers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LegacyStructuralTagResponseFormat {
    #[serde(rename = "type")]
    pub format_type: String,
    pub structures: Vec<StructuralTag>,
    pub triggers: Vec<String>,
}

/// New structural tag response format with opaque `format` field.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewStructuralTagResponseFormat {
    #[serde(rename = "type")]
    pub format_type: String,
    pub format: serde_json::Value,
}

/// Response format specification.
///
/// Supports standard OpenAI types (`text`, `json_object`, `json_schema`)
/// and vLLM's `structural_tag` type.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ResponseFormat {
    /// Legacy structural tag: `{ type: "structural_tag", structures: [...], triggers: [...] }`
    LegacyStructuralTag(LegacyStructuralTagResponseFormat),
    /// New structural tag: `{ type: "structural_tag", format: ... }`
    NewStructuralTag(NewStructuralTagResponseFormat),
    /// Standard format: `{ type: "text" | "json_object" | "json_schema", json_schema?: ... }`
    Standard(StandardResponseFormat),
}

/// Standard response format (text, json_object, json_schema).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StandardResponseFormat {
    /// Must be "text", "json_object", or "json_schema".
    #[serde(rename = "type")]
    pub format_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub json_schema: Option<JsonSchemaResponseFormat>,
}

// ---------------------------------------------------------------------------
// Function / Tool types
// ---------------------------------------------------------------------------

/// A function definition for tool use.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDefinition {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<serde_json::Value>,
}

/// A function call in a response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

/// A tool call in a response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: FunctionCall,
}

/// A tool parameter in a request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionToolsParam {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: FunctionDefinition,
}

// ---------------------------------------------------------------------------
// Chat completion request
// ---------------------------------------------------------------------------

/// A single message in a chat conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionMessageParam {
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

/// Chat completion request.
///
/// Follows the OpenAI API specification:
/// https://platform.openai.com/docs/api-reference/chat/create
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    /// Model identifier.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,

    /// The messages to generate chat completions for.
    pub messages: Vec<ChatCompletionMessageParam>,

    /// Sampling temperature (0.0 to 2.0).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,

    /// Nucleus sampling parameter.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,

    /// Number of completions to generate.
    #[serde(default = "default_n")]
    pub n: u32,

    /// Maximum number of tokens to generate (deprecated in favor of `max_completion_tokens`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,

    /// Maximum number of tokens to generate (preferred over `max_tokens`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<u32>,

    /// Whether to stream the response.
    #[serde(default)]
    pub stream: bool,

    /// Options for streaming.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<StreamOptions>,

    /// Stop sequences.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<StopCondition>,

    /// Frequency penalty (-2.0 to 2.0).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f64>,

    /// Presence penalty (-2.0 to 2.0).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f64>,

    /// Modify the likelihood of specified tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logit_bias: Option<std::collections::HashMap<String, f64>>,

    /// Whether to return log probabilities.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<bool>,

    /// Number of top log probabilities to return.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<u32>,

    /// Number of per-prompt-token log-probabilities to return (vLLM extension).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_logprobs: Option<u32>,

    /// Random seed for deterministic generation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,

    /// Response format specification.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,

    /// Tools available for the model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ChatCompletionToolsParam>>,

    /// Controls which (if any) tool is called by the model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<serde_json::Value>,

    /// User identifier.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,

    // -- vLLM-specific extensions --
    /// Top-k sampling parameter.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<i32>,

    /// Minimum probability threshold.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_p: Option<f64>,

    /// Repetition penalty.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repetition_penalty: Option<f64>,

    /// Minimum number of tokens to generate.
    #[serde(default)]
    pub min_tokens: u32,

    /// Stop token IDs.
    #[serde(default)]
    pub stop_token_ids: Vec<u32>,

    /// Whether to include the stop string in output.
    #[serde(default)]
    pub include_stop_str_in_output: bool,

    /// Whether to ignore end-of-sequence tokens.
    #[serde(default)]
    pub ignore_eos: bool,

    /// Skip special tokens in output.
    #[serde(default = "default_true")]
    pub skip_special_tokens: bool,

    /// Request priority (lower = higher priority).
    #[serde(default)]
    pub priority: i32,

    /// Cache salt for prefix cache isolation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_salt: Option<String>,

    /// Custom request ID.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,

    /// Regex pattern for constrained decoding (mutually exclusive with `response_format` and `guided_grammar`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub guided_regex: Option<String>,

    /// Lark/EBNF grammar for constrained decoding (mutually exclusive with `response_format` and `guided_regex`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub guided_grammar: Option<String>,

    /// Only allow sampling from these token IDs.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allowed_token_ids: Option<Vec<u32>>,

    /// Words that should never appear in the output.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bad_words: Option<Vec<String>>,

    /// Truncate prompt to at most this many tokens (keeps the last N).
    /// Must be >= 1 or -1 (meaning use model's max input length).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub truncate_prompt_tokens: Option<i64>,

    /// Whether to include reasoning content in the response (default true).
    #[serde(default = "default_true")]
    pub include_reasoning: bool,

    /// Additional keyword args to pass to the chat template renderer.
    /// Will be accessible by the chat template (e.g. `{"enable_thinking": false}`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chat_template_kwargs: Option<std::collections::HashMap<String, serde_json::Value>>,
}

// ---------------------------------------------------------------------------
// Completion request
// ---------------------------------------------------------------------------

/// A prompt that can be a string, list of strings, or token IDs.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum CompletionPrompt {
    /// A single string prompt.
    Single(String),
    /// Multiple string prompts.
    Multiple(Vec<String>),
    /// Token IDs.
    TokenIds(Vec<u32>),
    /// Multiple token ID sequences.
    MultipleTokenIds(Vec<Vec<u32>>),
}

/// Completion request.
///
/// Follows the OpenAI API specification:
/// https://platform.openai.com/docs/api-reference/completions/create
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionRequest {
    /// Model identifier.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,

    /// The prompt to generate completions for.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt: Option<CompletionPrompt>,

    /// Echo back the prompt.
    #[serde(default)]
    pub echo: bool,

    /// Sampling temperature.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,

    /// Nucleus sampling parameter.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,

    /// Number of completions to generate.
    #[serde(default = "default_n")]
    pub n: u32,

    /// Maximum number of tokens to generate (default: 16, per OpenAI spec).
    #[serde(default = "default_completion_max_tokens")]
    pub max_tokens: Option<u32>,

    /// Whether to stream the response.
    #[serde(default)]
    pub stream: bool,

    /// Options for streaming.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<StreamOptions>,

    /// Stop sequences.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<StopCondition>,

    /// Frequency penalty.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f64>,

    /// Presence penalty.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f64>,

    /// Logit bias.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logit_bias: Option<std::collections::HashMap<String, f64>>,

    /// Number of log probabilities to return.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<u32>,

    /// Number of per-prompt-token log-probabilities to return (vLLM extension).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_logprobs: Option<u32>,

    /// Text to append after the completion.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub suffix: Option<String>,

    /// Random seed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,

    /// User identifier.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,

    // -- vLLM-specific extensions --
    /// Top-k sampling.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<i32>,

    /// Minimum probability threshold.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_p: Option<f64>,

    /// Repetition penalty.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repetition_penalty: Option<f64>,

    /// Minimum tokens to generate.
    #[serde(default)]
    pub min_tokens: u32,

    /// Stop token IDs.
    #[serde(default)]
    pub stop_token_ids: Vec<u32>,

    /// Include stop string in output.
    #[serde(default)]
    pub include_stop_str_in_output: bool,

    /// Ignore EOS token.
    #[serde(default)]
    pub ignore_eos: bool,

    /// Skip special tokens.
    #[serde(default = "default_true")]
    pub skip_special_tokens: bool,

    /// Prepend tokenizer-defined special tokens (e.g. BOS) when tokenizing
    /// the prompt string. Mirrors Python vLLM's `add_special_tokens` field
    /// on its completion request; default `true`, same as Python vLLM.
    /// Ignored for `TokenIds` / `MultipleTokenIds` prompts.
    #[serde(default = "default_true")]
    pub add_special_tokens: bool,

    /// Request priority.
    #[serde(default)]
    pub priority: i32,

    /// Cache salt for prefix cache isolation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_salt: Option<String>,

    /// Custom request ID.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,

    /// Regex pattern for constrained decoding (mutually exclusive with `guided_grammar`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub guided_regex: Option<String>,

    /// Lark/EBNF grammar for constrained decoding (mutually exclusive with `guided_regex`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub guided_grammar: Option<String>,

    /// Only allow sampling from these token IDs.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allowed_token_ids: Option<Vec<u32>>,

    /// Words that should never appear in the output.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bad_words: Option<Vec<String>>,

    /// Truncate prompt to at most this many tokens (keeps the last N).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub truncate_prompt_tokens: Option<i64>,

    /// Block annotations for span-aware caching (internal, not part of API).
    /// Produced by `/v1/query/execute` and threaded through to the scheduler.
    #[serde(skip)]
    pub block_annotations: Option<scratchy_core_common::BlockAnnotations>,

    /// 🦭 Pad + hash the final partial block on completion (internal).
    #[serde(skip)]
    pub seal: bool,

    /// Deprioritize cached blocks for eviction after completion (internal).
    #[serde(skip)]
    pub volatile: bool,
}

// ---------------------------------------------------------------------------
// Stop condition
// ---------------------------------------------------------------------------

/// Stop condition: either a single string or a list of strings.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum StopCondition {
    Single(String),
    Multiple(Vec<String>),
}

impl StopCondition {
    /// Convert to a list of stop strings.
    pub fn to_strings(&self) -> Vec<String> {
        match self {
            StopCondition::Single(s) => vec![s.clone()],
            StopCondition::Multiple(v) => v.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// Chat completion response
// ---------------------------------------------------------------------------

/// A message in a chat completion response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub refusal: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,
}

/// Log probability information for a single token.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionLogProb {
    pub token: String,
    pub logprob: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bytes: Option<Vec<u8>>,
}

/// Log probability information for a token with top alternatives.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionLogProbsContent {
    pub token: String,
    pub logprob: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bytes: Option<Vec<u8>>,
    #[serde(default)]
    pub top_logprobs: Vec<ChatCompletionLogProb>,
}

/// Log probabilities for a chat completion response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionLogProbs {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<Vec<ChatCompletionLogProbsContent>>,
}

/// A choice in a chat completion response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionResponseChoice {
    pub index: u32,
    pub message: ChatMessage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<ChatCompletionLogProbs>,
    pub finish_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<serde_json::Value>,
    /// Per-prompt-token log-probabilities (vLLM extension).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_logprobs: Option<Vec<Option<ChatCompletionLogProbsContent>>>,
}

/// Chat completion response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChatCompletionResponseChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,
    pub usage: UsageInfo,
}

impl ChatCompletionResponse {
    /// Create a new response with default id and timestamp.
    pub fn new(
        model: String,
        choices: Vec<ChatCompletionResponseChoice>,
        usage: UsageInfo,
    ) -> Self {
        Self {
            id: format!("chatcmpl-{}", random_uuid()),
            object: "chat.completion".to_string(),
            created: unix_timestamp(),
            model,
            choices,
            system_fingerprint: None,
            usage,
        }
    }
}

// ---------------------------------------------------------------------------
// Chat completion streaming response
// ---------------------------------------------------------------------------

/// A delta message in a streaming chat completion.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DeltaMessage {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<serde_json::Value>>,
}

/// A choice in a streaming chat completion response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionResponseStreamChoice {
    pub index: u32,
    pub delta: DeltaMessage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<ChatCompletionLogProbs>,
    pub finish_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<serde_json::Value>,
}

/// A streaming chat completion response chunk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionStreamResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChatCompletionResponseStreamChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<UsageInfo>,
}

impl ChatCompletionStreamResponse {
    /// Create a new streaming chunk.
    pub fn new(
        id: String,
        model: String,
        choices: Vec<ChatCompletionResponseStreamChoice>,
    ) -> Self {
        Self {
            id,
            object: "chat.completion.chunk".to_string(),
            created: unix_timestamp(),
            model,
            choices,
            usage: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Completion response
// ---------------------------------------------------------------------------

/// Log probabilities for a completion response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionLogProbs {
    pub text_offset: Vec<u32>,
    pub token_logprobs: Vec<Option<f64>>,
    pub tokens: Vec<String>,
    pub top_logprobs: Vec<Option<std::collections::HashMap<String, f64>>>,
}

/// A choice in a completion response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionResponseChoice {
    pub index: u32,
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<CompletionLogProbs>,
    pub finish_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<serde_json::Value>,
    /// Per-prompt-token log-probabilities (vLLM extension).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_logprobs: Option<CompletionLogProbs>,
}

/// Completion response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<CompletionResponseChoice>,
    pub usage: UsageInfo,
}

impl CompletionResponse {
    /// Create a new response with default id and timestamp.
    pub fn new(model: String, choices: Vec<CompletionResponseChoice>, usage: UsageInfo) -> Self {
        Self {
            id: format!("cmpl-{}", random_uuid()),
            object: "text_completion".to_string(),
            created: unix_timestamp(),
            model,
            choices,
            usage,
        }
    }
}

// ---------------------------------------------------------------------------
// Completion streaming response
// ---------------------------------------------------------------------------

/// A choice in a streaming completion response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionResponseStreamChoice {
    pub index: u32,
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<CompletionLogProbs>,
    pub finish_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<serde_json::Value>,
}

/// A streaming completion response chunk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionStreamResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<CompletionResponseStreamChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<UsageInfo>,
}

impl CompletionStreamResponse {
    /// Create a new streaming chunk.
    pub fn new(id: String, model: String, choices: Vec<CompletionResponseStreamChoice>) -> Self {
        Self {
            id,
            object: "text_completion".to_string(),
            created: unix_timestamp(),
            model,
            choices,
            usage: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Model types
// ---------------------------------------------------------------------------

/// Model permission information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPermission {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub allow_create_engine: bool,
    pub allow_sampling: bool,
    pub allow_logprobs: bool,
    pub allow_search_indices: bool,
    pub allow_view: bool,
    pub allow_fine_tuning: bool,
    pub organization: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub group: Option<String>,
    pub is_blocking: bool,
}

impl Default for ModelPermission {
    fn default() -> Self {
        Self {
            id: format!("modelperm-{}", random_uuid()),
            object: "model_permission".to_string(),
            created: unix_timestamp(),
            allow_create_engine: false,
            allow_sampling: true,
            allow_logprobs: true,
            allow_search_indices: false,
            allow_view: true,
            allow_fine_tuning: false,
            organization: "*".to_string(),
            group: None,
            is_blocking: false,
        }
    }
}

/// A model card.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCard {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub owned_by: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub root: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_model_len: Option<usize>,
    pub permission: Vec<ModelPermission>,
}

impl ModelCard {
    /// Create a new model card.
    pub fn new(id: String) -> Self {
        Self {
            id,
            object: "model".to_string(),
            created: unix_timestamp(),
            owned_by: "vllm".to_string(),
            root: None,
            parent: None,
            max_model_len: None,
            permission: vec![ModelPermission::default()],
        }
    }
}

/// A list of models.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelList {
    pub object: String,
    pub data: Vec<ModelCard>,
}

impl ModelList {
    pub fn new(data: Vec<ModelCard>) -> Self {
        Self {
            object: "list".to_string(),
            data,
        }
    }
}

// ---------------------------------------------------------------------------
// Health / version
// ---------------------------------------------------------------------------

/// Health check response.
#[derive(Debug, Clone, Serialize)]
pub struct HealthResponse {
    pub status: &'static str,
}

/// Version information response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionResponse {
    pub version: String,
}

/// A single engine prompt returned by `/v1/chat/completions/render`.
/// Mirrors Python's `ProcessorInputs` (text-only subset).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RenderEnginePrompt {
    /// The rendered prompt text (or token IDs when multimodal).
    pub prompt: serde_json::Value,
}

/// Response for `/v1/chat/completions/render` — `[conversation, engine_prompts]`.
/// Matches Python vLLM's return format exactly.
pub type ChatCompletionRenderResponse = (Vec<serde_json::Value>, Vec<RenderEnginePrompt>);

/// Query parameters for `GET /server_info`.
#[derive(Debug, Clone, Deserialize)]
pub struct ServerInfoParams {
    /// `"text"` (default) returns `scratchy_core_config` as a human-readable string.
    /// `"json"` returns it as a structured JSON object.
    pub config_format: Option<String>,
}

/// Response for `GET /server_info`.
///
/// Mirrors Python vLLM's `/server_info` endpoint with three top-level keys.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerInfoResponse {
    /// The VllmConfig used to initialize the stack.
    /// String when `config_format=text`, structured JSON when `config_format=json`.
    pub scratchy_core_config: serde_json::Value,
    /// VLLM_* environment variables (secrets filtered out).
    pub vllm_env: serde_json::Value,
    /// Basic system information (OS, version, etc.).
    pub system_env: serde_json::Value,
}

// ---------------------------------------------------------------------------
// Embedding types
// ---------------------------------------------------------------------------

/// Embedding request input: a string, array of strings, or array of token IDs.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingInput {
    /// A single string.
    Single(String),
    /// Multiple inputs (strings or token IDs).
    Multiple(Vec<EmbeddingInputItem>),
}

/// A single item in an embedding request input array.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingInputItem {
    /// A text string.
    Text(String),
    /// Pre-tokenized token IDs.
    TokenIds(Vec<u32>),
}

/// Embedding request.
///
/// Follows the OpenAI API specification:
/// https://platform.openai.com/docs/api-reference/embeddings/create
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingRequest {
    /// Input text(s) to embed.
    pub input: EmbeddingInput,

    /// Model identifier.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,

    /// Encoding format: "float" (default). "base64" is deferred.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub encoding_format: Option<String>,

    /// Matryoshka dimension truncation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dimensions: Option<usize>,

    /// User identifier.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
}

/// A single embedding object in the response.
///
/// `embedding` is `serde_json::Value` to support both single-vector (`[f32; D]`)
/// and multi-vector ColBERT output (`[[f32; D]; T]`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingObject {
    pub index: usize,
    pub object: String,
    pub embedding: serde_json::Value,
}

/// Usage information for embedding requests.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingUsage {
    pub prompt_tokens: u32,
    pub total_tokens: u32,
}

/// Embedding response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingResponse {
    pub object: String,
    pub data: Vec<EmbeddingObject>,
    pub model: String,
    pub usage: EmbeddingUsage,
}

impl EmbeddingResponse {
    /// Create a new embedding response.
    pub fn new(model: String, data: Vec<EmbeddingObject>, usage: EmbeddingUsage) -> Self {
        Self {
            object: "list".to_string(),
            data,
            model,
            usage,
        }
    }
}

// ---------------------------------------------------------------------------
// Tokenize / Detokenize types
// ---------------------------------------------------------------------------

/// Request body for `POST /tokenize`.
///
/// Accepts either a raw `prompt` string (completion-style) or a `messages`
/// array (chat-style). When `messages` is provided the server applies the
/// chat template before tokenizing.
///
/// Mirrors Python vLLM `TokenizeCompletionRequest` / `TokenizeChatRequest`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizeRequest {
    /// Model identifier (optional, ignored — single-model server).
    #[serde(default)]
    pub model: Option<String>,

    /// Raw prompt string to tokenize (completion-style).
    #[serde(default)]
    pub prompt: Option<String>,

    /// Chat messages to tokenize (chat-style). Requires a chat template.
    #[serde(default)]
    pub messages: Option<Vec<ChatCompletionMessageParam>>,

    /// Whether to add special tokens (e.g. BOS). Default `true` for prompt
    /// mode, `false` for chat mode (chat template usually adds them).
    #[serde(default)]
    pub add_special_tokens: Option<bool>,

    /// If true, also return the string representation of each token.
    #[serde(default)]
    pub return_token_strs: Option<bool>,

    /// Whether to add the generation prompt when using chat messages.
    #[serde(default = "default_true")]
    pub add_generation_prompt: bool,
}

/// Response body for `POST /tokenize`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizeResponse {
    /// Number of tokens.
    pub count: usize,
    /// Maximum context length of the model.
    pub max_model_len: usize,
    /// The token IDs.
    pub tokens: Vec<u32>,
    /// Optional string representations of each token.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token_strs: Option<Vec<String>>,
}

/// Request body for `POST /detokenize`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetokenizeRequest {
    /// Model identifier (optional, ignored).
    #[serde(default)]
    pub model: Option<String>,
    /// Token IDs to decode.
    pub tokens: Vec<u32>,
}

/// Response body for `POST /detokenize`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetokenizeResponse {
    /// The decoded text.
    pub prompt: String,
}

// ---------------------------------------------------------------------------
// Batch processing types
// ---------------------------------------------------------------------------

/// A single request in a batch input JSONL file.
///
/// Matches the OpenAI Batch API input format:
/// ```json
/// {"custom_id": "req-1", "method": "POST", "url": "/v1/chat/completions", "body": {...}}
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchRequestInput {
    /// User-specified identifier for correlating input to output.
    pub custom_id: String,
    /// HTTP method (always "POST" for supported endpoints).
    pub method: String,
    /// API endpoint URL (e.g. "/v1/chat/completions").
    pub url: String,
    /// Request body — deserialized based on the `url` field.
    pub body: serde_json::Value,
}

/// Response data for a single batch request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchResponseData {
    /// HTTP status code.
    pub status_code: u16,
    /// Unique request identifier.
    pub request_id: String,
    /// Response body (serialized response object), absent on error.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub body: Option<serde_json::Value>,
}

/// A single result in the batch output JSONL file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchRequestOutput {
    /// Unique batch output identifier.
    pub id: String,
    /// The custom_id from the corresponding input request.
    pub custom_id: String,
    /// Successful response data (None if error).
    pub response: Option<BatchResponseData>,
    /// Error information (None if success).
    pub error: Option<serde_json::Value>,
}

// ---------------------------------------------------------------------------
// Default helpers
// ---------------------------------------------------------------------------

fn default_n() -> u32 {
    1
}

fn default_true() -> bool {
    true
}

fn default_completion_max_tokens() -> Option<u32> {
    Some(16)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Error types --

    #[test]
    fn test_error_response_new() {
        let resp = ErrorResponse::new("not found", "NotFoundError", 404);
        assert_eq!(resp.error.message, "not found");
        assert_eq!(resp.error.error_type, "NotFoundError");
        assert_eq!(resp.error.code, 404);
        assert!(resp.error.param.is_none());
    }

    #[test]
    fn test_error_response_serde() {
        let resp = ErrorResponse::new("bad request", "BadRequestError", 400);
        let json = serde_json::to_string(&resp).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["error"]["message"], "bad request");
        assert_eq!(parsed["error"]["type"], "BadRequestError");
        assert_eq!(parsed["error"]["code"], 400);
    }

    // -- Usage --

    #[test]
    fn test_usage_info_default() {
        let usage = UsageInfo::default();
        assert_eq!(usage.prompt_tokens, 0);
        assert_eq!(usage.total_tokens, 0);
        assert!(usage.completion_tokens.is_none());
    }

    #[test]
    fn test_usage_info_serde() {
        let usage = UsageInfo {
            prompt_tokens: 10,
            total_tokens: 30,
            completion_tokens: Some(20),
            prompt_tokens_details: Some(PromptTokenUsageInfo {
                cached_tokens: Some(5),
            }),
        };
        let json = serde_json::to_string(&usage).unwrap();
        let parsed: UsageInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.prompt_tokens, 10);
        assert_eq!(parsed.completion_tokens, Some(20));
        assert_eq!(parsed.prompt_tokens_details.unwrap().cached_tokens, Some(5));
    }

    // -- Chat completion request --

    #[test]
    fn test_chat_completion_request_minimal() {
        let json = r#"{
            "messages": [
                {"role": "user", "content": "Hello"}
            ]
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.messages.len(), 1);
        assert_eq!(req.messages[0].role, "user");
        assert!(!req.stream);
        assert_eq!(req.n, 1);
    }

    #[test]
    fn test_chat_completion_request_full() {
        let json = r#"{
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hello"}
            ],
            "temperature": 0.7,
            "top_p": 0.9,
            "n": 2,
            "max_tokens": 100,
            "stream": true,
            "stream_options": {"include_usage": true},
            "stop": ["\n", "END"],
            "frequency_penalty": 0.5,
            "presence_penalty": 0.3,
            "seed": 42,
            "top_k": 50,
            "min_p": 0.1,
            "priority": 5
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.model.as_deref(), Some("gpt-4"));
        assert_eq!(req.messages.len(), 2);
        assert_eq!(req.temperature, Some(0.7));
        assert_eq!(req.top_p, Some(0.9));
        assert_eq!(req.n, 2);
        assert_eq!(req.max_tokens, Some(100));
        assert!(req.stream);
        assert!(req.stream_options.is_some());
        assert_eq!(req.frequency_penalty, Some(0.5));
        assert_eq!(req.seed, Some(42));
        assert_eq!(req.top_k, Some(50));
        assert_eq!(req.priority, 5);
    }

    #[test]
    fn test_chat_completion_request_stop_single() {
        let json = r#"{
            "messages": [{"role": "user", "content": "Hi"}],
            "stop": "\n"
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        let stop = req.stop.unwrap();
        assert_eq!(stop.to_strings(), vec!["\n"]);
    }

    // -- Chat completion response --

    #[test]
    fn test_chat_completion_response_serde() {
        let resp = ChatCompletionResponse::new(
            "gpt-4".to_string(),
            vec![ChatCompletionResponseChoice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".to_string(),
                    content: Some("Hello!".to_string()),
                    refusal: None,
                    tool_calls: None,
                    reasoning: None,
                },
                logprobs: None,
                finish_reason: Some("stop".to_string()),
                stop_reason: None,
                prompt_logprobs: None,
            }],
            UsageInfo {
                prompt_tokens: 5,
                total_tokens: 10,
                completion_tokens: Some(5),
                prompt_tokens_details: None,
            },
        );
        let json = serde_json::to_string(&resp).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["object"], "chat.completion");
        assert_eq!(parsed["model"], "gpt-4");
        assert_eq!(parsed["choices"][0]["message"]["content"], "Hello!");
        assert_eq!(parsed["choices"][0]["finish_reason"], "stop");
        assert_eq!(parsed["usage"]["prompt_tokens"], 5);
    }

    // -- Streaming response --

    #[test]
    fn test_chat_stream_response_serde() {
        let chunk = ChatCompletionStreamResponse::new(
            "chatcmpl-123".to_string(),
            "gpt-4".to_string(),
            vec![ChatCompletionResponseStreamChoice {
                index: 0,
                delta: DeltaMessage {
                    role: Some("assistant".to_string()),
                    content: Some("He".to_string()),
                    reasoning: None,
                    tool_calls: None,
                },
                logprobs: None,
                finish_reason: None,
                stop_reason: None,
            }],
        );
        let json = serde_json::to_string(&chunk).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["object"], "chat.completion.chunk");
        assert_eq!(parsed["choices"][0]["delta"]["content"], "He");
    }

    // -- Completion request --

    #[test]
    fn test_completion_request_string_prompt() {
        let json = r#"{
            "prompt": "Hello world",
            "max_tokens": 50
        }"#;
        let req: CompletionRequest = serde_json::from_str(json).unwrap();
        assert!(matches!(req.prompt, Some(CompletionPrompt::Single(ref s)) if s == "Hello world"));
        assert_eq!(req.max_tokens, Some(50));
    }

    #[test]
    fn test_completion_request_token_ids_prompt() {
        let json = r#"{
            "prompt": [1, 2, 3, 4],
            "max_tokens": 10
        }"#;
        let req: CompletionRequest = serde_json::from_str(json).unwrap();
        assert!(
            matches!(req.prompt, Some(CompletionPrompt::TokenIds(ref ids)) if ids == &[1, 2, 3, 4])
        );
    }

    // -- Completion response --

    #[test]
    fn test_completion_response_serde() {
        let resp = CompletionResponse::new(
            "gpt-3.5".to_string(),
            vec![CompletionResponseChoice {
                index: 0,
                text: "Hello!".to_string(),
                logprobs: None,
                finish_reason: Some("stop".to_string()),
                stop_reason: None,
                prompt_logprobs: None,
            }],
            UsageInfo {
                prompt_tokens: 3,
                total_tokens: 6,
                completion_tokens: Some(3),
                prompt_tokens_details: None,
            },
        );
        let json = serde_json::to_string(&resp).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["object"], "text_completion");
        assert_eq!(parsed["choices"][0]["text"], "Hello!");
    }

    // -- Model types --

    #[test]
    fn test_model_card() {
        let card = ModelCard::new("llama-3-8b".to_string());
        assert_eq!(card.id, "llama-3-8b");
        assert_eq!(card.object, "model");
        assert_eq!(card.owned_by, "vllm");
        assert_eq!(card.permission.len(), 1);
    }

    #[test]
    fn test_model_list() {
        let list = ModelList::new(vec![
            ModelCard::new("llama-3-8b".to_string()),
            ModelCard::new("mistral-7b".to_string()),
        ]);
        assert_eq!(list.object, "list");
        assert_eq!(list.data.len(), 2);

        let json = serde_json::to_string(&list).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["data"][0]["id"], "llama-3-8b");
        assert_eq!(parsed["data"][1]["id"], "mistral-7b");
    }

    // -- StopCondition --

    #[test]
    fn test_stop_condition_single() {
        let json = r#""stop""#;
        let stop: StopCondition = serde_json::from_str(json).unwrap();
        assert_eq!(stop.to_strings(), vec!["stop"]);
    }

    #[test]
    fn test_stop_condition_multiple() {
        let json = r#"["stop", "end"]"#;
        let stop: StopCondition = serde_json::from_str(json).unwrap();
        assert_eq!(stop.to_strings(), vec!["stop", "end"]);
    }

    // -- CompletionPrompt --

    #[test]
    fn test_completion_prompt_multiple_strings() {
        let json = r#"["Hello", "World"]"#;
        let prompt: CompletionPrompt = serde_json::from_str(json).unwrap();
        assert!(matches!(prompt, CompletionPrompt::Multiple(ref v) if v.len() == 2));
    }

    // -- DeltaMessage --

    #[test]
    fn test_delta_message_empty() {
        let delta = DeltaMessage::default();
        let json = serde_json::to_string(&delta).unwrap();
        assert_eq!(json, "{}");
    }

    #[test]
    fn test_delta_message_with_content() {
        let delta = DeltaMessage {
            role: Some("assistant".to_string()),
            content: Some("hi".to_string()),
            ..Default::default()
        };
        let json = serde_json::to_string(&delta).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["role"], "assistant");
        assert_eq!(parsed["content"], "hi");
    }

    // -- guided_regex deserialization --

    #[test]
    fn test_chat_completion_request_guided_regex() {
        let json = r#"{
            "messages": [{"role": "user", "content": "Hi"}],
            "guided_regex": "[0-9]+"
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.guided_regex.as_deref(), Some("[0-9]+"));
    }

    // -- Embedding types --

    #[test]
    fn test_embedding_request_single_string() {
        let json = r#"{"input": "Hello world", "model": "test"}"#;
        let req: EmbeddingRequest = serde_json::from_str(json).unwrap();
        assert!(matches!(req.input, EmbeddingInput::Single(ref s) if s == "Hello world"));
        assert_eq!(req.model.as_deref(), Some("test"));
    }

    #[test]
    fn test_embedding_request_multiple_strings() {
        let json = r#"{"input": ["Hello", "World"]}"#;
        let req: EmbeddingRequest = serde_json::from_str(json).unwrap();
        assert!(matches!(req.input, EmbeddingInput::Multiple(ref v) if v.len() == 2));
    }

    #[test]
    fn test_embedding_request_token_ids() {
        let json = r#"{"input": [[1, 2, 3], [4, 5]]}"#;
        let req: EmbeddingRequest = serde_json::from_str(json).unwrap();
        assert!(matches!(req.input, EmbeddingInput::Multiple(ref v) if v.len() == 2));
    }

    #[test]
    fn test_embedding_request_dimensions() {
        let json = r#"{"input": "test", "dimensions": 32}"#;
        let req: EmbeddingRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.dimensions, Some(32));
    }

    #[test]
    fn test_embedding_response_serde() {
        let resp = EmbeddingResponse::new(
            "test-model".to_string(),
            vec![EmbeddingObject {
                index: 0,
                object: "embedding".to_string(),
                embedding: serde_json::json!([0.1, 0.2, 0.3]),
            }],
            EmbeddingUsage {
                prompt_tokens: 3,
                total_tokens: 3,
            },
        );
        let json = serde_json::to_string(&resp).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["object"], "list");
        assert_eq!(
            parsed["data"][0]["embedding"],
            serde_json::json!([0.1, 0.2, 0.3])
        );
        assert_eq!(parsed["usage"]["prompt_tokens"], 3);
    }

    #[test]
    fn test_embedding_response_multi_vector_serde() {
        // ColBERT multi-vector: embedding is a 2D array [[f32; D]; T].
        let resp = EmbeddingResponse::new(
            "colbert-model".to_string(),
            vec![EmbeddingObject {
                index: 0,
                object: "embedding".to_string(),
                embedding: serde_json::json!([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),
            }],
            EmbeddingUsage {
                prompt_tokens: 3,
                total_tokens: 3,
            },
        );
        let json = serde_json::to_string(&resp).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["object"], "list");
        let emb = &parsed["data"][0]["embedding"];
        assert!(emb.is_array());
        assert_eq!(emb.as_array().unwrap().len(), 3);
        // Each element is itself an array (2D).
        assert!(emb[0].is_array());
        assert_eq!(emb[0], serde_json::json!([0.1, 0.2]));
        assert_eq!(emb[2], serde_json::json!([0.5, 0.6]));
    }

    #[test]
    fn test_completion_request_guided_regex() {
        let json = r#"{
            "prompt": "Give me a number",
            "guided_regex": "[0-9]+"
        }"#;
        let req: CompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.guided_regex.as_deref(), Some("[0-9]+"));
    }

    #[test]
    fn test_chat_completion_request_no_guided_regex() {
        let json = r#"{
            "messages": [{"role": "user", "content": "Hi"}]
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert!(req.guided_regex.is_none());
    }

    // -- Batch types --

    #[test]
    fn test_batch_request_input_serde() {
        let input = BatchRequestInput {
            custom_id: "req-1".to_string(),
            method: "POST".to_string(),
            url: "/v1/chat/completions".to_string(),
            body: serde_json::json!({"messages": [{"role": "user", "content": "Hello"}]}),
        };
        let json = serde_json::to_string(&input).unwrap();
        let parsed: BatchRequestInput = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.custom_id, "req-1");
        assert_eq!(parsed.url, "/v1/chat/completions");
        assert!(parsed.body["messages"].is_array());
    }

    #[test]
    fn test_batch_request_output_serde() {
        let output = BatchRequestOutput {
            id: "batch-abc123".to_string(),
            custom_id: "req-1".to_string(),
            response: Some(BatchResponseData {
                status_code: 200,
                request_id: "req-id-1".to_string(),
                body: Some(serde_json::json!({"choices": []})),
            }),
            error: None,
        };
        let json = serde_json::to_string(&output).unwrap();
        let parsed: BatchRequestOutput = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.id, "batch-abc123");
        assert_eq!(parsed.custom_id, "req-1");
        let resp = parsed.response.unwrap();
        assert_eq!(resp.status_code, 200);
        assert!(parsed.error.is_none());
    }

    #[test]
    fn test_batch_request_input_chat() {
        let json = r#"{
            "custom_id": "chat-1",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 50
            }
        }"#;
        let input: BatchRequestInput = serde_json::from_str(json).unwrap();
        assert_eq!(input.custom_id, "chat-1");
        // Body should be deserializable as a ChatCompletionRequest.
        let req: ChatCompletionRequest = serde_json::from_value(input.body).unwrap();
        assert_eq!(req.messages.len(), 1);
        assert_eq!(req.max_tokens, Some(50));
    }

    #[test]
    fn test_batch_request_input_embedding() {
        let json = r#"{
            "custom_id": "emb-1",
            "method": "POST",
            "url": "/v1/embeddings",
            "body": {
                "input": "Hello world",
                "model": "test"
            }
        }"#;
        let input: BatchRequestInput = serde_json::from_str(json).unwrap();
        assert_eq!(input.custom_id, "emb-1");
        let req: EmbeddingRequest = serde_json::from_value(input.body).unwrap();
        assert_eq!(req.model.as_deref(), Some("test"));
    }
}
