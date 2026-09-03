// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! HTTP client wrapper for E2E testing.

use anyhow::{Context, Result, bail};
use scratchy_serving_api::protocol::{
    ChatCompletionRenderResponse, ChatCompletionRequest, ChatCompletionResponse,
    ChatCompletionStreamResponse, CompletionRequest, CompletionResponse, DetokenizeRequest,
    DetokenizeResponse, ModelList, ServerInfoResponse, TokenizeRequest, TokenizeResponse,
    VersionResponse,
};

/// GPU memory usage info returned by `/gpu_memory`.
#[derive(Debug, serde::Deserialize)]
pub struct GpuMemoryInfo {
    pub used_bytes: u64,
    pub free_bytes: u64,
    pub total_bytes: u64,
}

/// A thin HTTP client for talking to a running vLLM server.
pub struct Client {
    inner: reqwest::Client,
    base_url: String,
}

impl Client {
    /// Create a new client pointing at the given base URL.
    pub fn new(base_url: &str) -> Self {
        // reqwest is built with `rustls-no-provider` (see the workspace
        // manifest); without this it panics in `build()`. The binary does the
        // same in main() — tests do not run main().
        let _ = rustls::crypto::ring::default_provider().install_default();
        Self {
            inner: reqwest::Client::new(),
            base_url: base_url.trim_end_matches('/').to_string(),
        }
    }

    /// GET /health — returns true if the server is healthy.
    pub async fn health(&self) -> Result<bool> {
        let resp = self
            .inner
            .get(format!("{}/health", self.base_url))
            .send()
            .await?;
        Ok(resp.status().is_success())
    }

    /// GET /version — returns the version string.
    pub async fn version(&self) -> Result<VersionResponse> {
        let resp = self
            .inner
            .get(format!("{}/version", self.base_url))
            .send()
            .await?;
        resp.json()
            .await
            .context("failed to parse version response")
    }

    /// GET /server_info — returns server configuration, env vars, and system info.
    pub async fn server_info(&self, config_format: Option<&str>) -> Result<ServerInfoResponse> {
        let mut url = format!("{}/server_info", self.base_url);
        if let Some(fmt) = config_format {
            url.push_str(&format!("?config_format={fmt}"));
        }
        let resp = self.inner.get(&url).send().await?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            bail!("server_info failed with status {status}: {body}");
        }
        resp.json()
            .await
            .context("failed to parse server_info response")
    }

    /// GET /v1/models — list available models.
    pub async fn list_models(&self) -> Result<ModelList> {
        let resp = self
            .inner
            .get(format!("{}/v1/models", self.base_url))
            .send()
            .await?;
        resp.json()
            .await
            .context("failed to parse model list response")
    }

    /// POST /v1/chat/completions — non-streaming chat completion.
    pub async fn chat_completion(
        &self,
        request: &ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse> {
        assert!(!request.stream, "use chat_completion_stream for streaming");

        let resp = self
            .inner
            .post(format!("{}/v1/chat/completions", self.base_url))
            .json(request)
            .send()
            .await?;

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            bail!("chat completion failed with status {status}: {body}");
        }

        resp.json()
            .await
            .context("failed to parse chat completion response")
    }

    /// POST /v1/chat/completions/render — render chat template without generating.
    pub async fn render_chat_completion(
        &self,
        request: &ChatCompletionRequest,
    ) -> Result<ChatCompletionRenderResponse> {
        let resp = self
            .inner
            .post(format!("{}/v1/chat/completions/render", self.base_url))
            .json(request)
            .send()
            .await?;

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            bail!("render chat completion failed with status {status}: {body}");
        }

        resp.json()
            .await
            .context("failed to parse render chat completion response")
    }

    /// POST /v1/chat/completions with stream=true — returns collected stream chunks.
    pub async fn chat_completion_stream(
        &self,
        request: &ChatCompletionRequest,
    ) -> Result<Vec<ChatCompletionStreamResponse>> {
        assert!(request.stream, "use chat_completion for non-streaming");

        let resp = self
            .inner
            .post(format!("{}/v1/chat/completions", self.base_url))
            .json(request)
            .send()
            .await?;

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            bail!("streaming chat completion failed with status {status}: {body}");
        }

        let body = resp.text().await?;
        parse_sse_chunks(&body)
    }

    /// POST /v1/completions — non-streaming text completion.
    pub async fn completion(&self, request: &CompletionRequest) -> Result<CompletionResponse> {
        let resp = self
            .inner
            .post(format!("{}/v1/completions", self.base_url))
            .json(request)
            .send()
            .await?;

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            bail!("completion failed with status {status}: {body}");
        }

        resp.json()
            .await
            .context("failed to parse completion response")
    }

    /// POST /v1/chat/completions — returns raw response for status code checking.
    pub async fn chat_completion_raw(&self, body: &serde_json::Value) -> Result<reqwest::Response> {
        let resp = self
            .inner
            .post(format!("{}/v1/chat/completions", self.base_url))
            .json(body)
            .send()
            .await?;
        Ok(resp)
    }

    /// POST /v1/completions — returns raw response for status code checking.
    pub async fn completion_raw(&self, body: &serde_json::Value) -> Result<reqwest::Response> {
        let resp = self
            .inner
            .post(format!("{}/v1/completions", self.base_url))
            .json(body)
            .send()
            .await?;
        Ok(resp)
    }

    /// POST /v1/embeddings — compute embeddings.
    pub async fn embedding(
        &self,
        request: &scratchy_serving_api::protocol::EmbeddingRequest,
    ) -> Result<scratchy_serving_api::protocol::EmbeddingResponse> {
        let resp = self
            .inner
            .post(format!("{}/v1/embeddings", self.base_url))
            .json(request)
            .send()
            .await?;

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            bail!("embedding failed with status {status}: {body}");
        }

        resp.json()
            .await
            .context("failed to parse embedding response")
    }

    /// POST /tokenize — tokenize text.
    pub async fn tokenize(&self, request: &TokenizeRequest) -> Result<TokenizeResponse> {
        let resp = self
            .inner
            .post(format!("{}/tokenize", self.base_url))
            .json(request)
            .send()
            .await?;

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            bail!("tokenize failed with status {status}: {body}");
        }

        resp.json()
            .await
            .context("failed to parse tokenize response")
    }

    /// POST /detokenize — decode token IDs back to text.
    pub async fn detokenize(&self, request: &DetokenizeRequest) -> Result<DetokenizeResponse> {
        let resp = self
            .inner
            .post(format!("{}/detokenize", self.base_url))
            .json(request)
            .send()
            .await?;

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            bail!("detokenize failed with status {status}: {body}");
        }

        resp.json()
            .await
            .context("failed to parse detokenize response")
    }

    /// POST /sleep — put the engine to sleep.
    pub async fn sleep(&self, level: u32) -> Result<()> {
        let resp = self
            .inner
            .post(format!("{}/sleep", self.base_url))
            .json(&serde_json::json!({ "level": level }))
            .send()
            .await?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            bail!("sleep failed with status {status}: {body}");
        }
        Ok(())
    }

    /// POST /wake_up — wake the engine from sleep.
    pub async fn wake_up(&self, tags: Option<Vec<String>>) -> Result<()> {
        let resp = self
            .inner
            .post(format!("{}/wake_up", self.base_url))
            .json(&serde_json::json!({ "tags": tags }))
            .send()
            .await?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            bail!("wake_up failed with status {status}: {body}");
        }
        Ok(())
    }

    /// GET /is_sleeping — query sleep state.
    pub async fn is_sleeping(&self) -> Result<bool> {
        let resp = self
            .inner
            .get(format!("{}/is_sleeping", self.base_url))
            .send()
            .await?;
        let v: serde_json::Value = resp.json().await?;
        Ok(v["is_sleeping"].as_bool().unwrap_or(false))
    }

    /// GET /gpu_memory — query GPU memory usage.
    pub async fn gpu_memory(&self) -> Result<GpuMemoryInfo> {
        let resp = self
            .inner
            .get(format!("{}/gpu_memory", self.base_url))
            .send()
            .await?;
        resp.json()
            .await
            .context("failed to parse gpu_memory response")
    }

    /// POST /v1/messages — Anthropic Messages API (non-streaming).
    pub async fn anthropic_messages(
        &self,
        request: &scratchy_serving_api::anthropic::MessagesRequest,
    ) -> Result<scratchy_serving_api::anthropic::MessagesResponse> {
        let resp = self
            .inner
            .post(format!("{}/v1/messages", self.base_url))
            .json(request)
            .send()
            .await?;

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            bail!("anthropic messages failed with status {status}: {body}");
        }

        resp.json()
            .await
            .context("failed to parse Anthropic messages response")
    }

    /// POST /v1/messages — Anthropic Messages API (raw JSON, for flexible testing).
    pub async fn anthropic_messages_raw(
        &self,
        body: &serde_json::Value,
    ) -> Result<reqwest::Response> {
        let resp = self
            .inner
            .post(format!("{}/v1/messages", self.base_url))
            .json(body)
            .send()
            .await?;
        Ok(resp)
    }

    /// POST /v1/messages with stream=true — returns SSE event bodies as JSON values.
    pub async fn anthropic_messages_stream(
        &self,
        body: &serde_json::Value,
    ) -> Result<Vec<serde_json::Value>> {
        let resp = self
            .inner
            .post(format!("{}/v1/messages", self.base_url))
            .json(body)
            .send()
            .await?;

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            bail!("anthropic streaming messages failed with status {status}: {body}");
        }

        let body = resp.text().await?;
        parse_anthropic_sse_events(&body)
    }

    /// POST /v1/responses — raw response.
    pub async fn responses_raw(&self, body: &serde_json::Value) -> Result<reqwest::Response> {
        let resp = self
            .inner
            .post(format!("{}/v1/responses", self.base_url))
            .json(body)
            .send()
            .await?;
        Ok(resp)
    }

    /// POST /v1/responses with stream=true — returns SSE event bodies as JSON values.
    pub async fn responses_stream(
        &self,
        body: &serde_json::Value,
    ) -> Result<Vec<serde_json::Value>> {
        let resp = self
            .inner
            .post(format!("{}/v1/responses", self.base_url))
            .json(body)
            .send()
            .await?;

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            bail!("responses streaming failed with status {status}: {body}");
        }

        let body = resp.text().await?;
        parse_anthropic_sse_events(&body)
    }

    /// GET /metrics — returns raw Prometheus text.
    pub async fn metrics(&self) -> Result<String> {
        let resp = self
            .inner
            .get(format!("{}/metrics", self.base_url))
            .send()
            .await?;
        resp.text().await.context("failed to read metrics")
    }
}

/// Parse Anthropic SSE response body into JSON event payloads.
fn parse_anthropic_sse_events(body: &str) -> Result<Vec<serde_json::Value>> {
    let mut events = Vec::new();
    for line in body.lines() {
        let line = line.trim();
        if let Some(data) = line
            .strip_prefix("data: ")
            .and_then(|d| serde_json::from_str::<serde_json::Value>(d).ok())
        {
            events.push(data);
        }
    }
    Ok(events)
}

/// Parse SSE response body into individual JSON chunks.
fn parse_sse_chunks(body: &str) -> Result<Vec<ChatCompletionStreamResponse>> {
    let mut chunks = Vec::new();

    for line in body.lines() {
        let line = line.trim();
        if let Some(data) = line.strip_prefix("data: ") {
            if data == "[DONE]" {
                break;
            }
            let chunk: ChatCompletionStreamResponse =
                serde_json::from_str(data).context(format!("failed to parse SSE chunk: {data}"))?;
            chunks.push(chunk);
        }
    }

    Ok(chunks)
}
