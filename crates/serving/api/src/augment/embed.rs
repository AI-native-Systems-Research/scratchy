use std::sync::Arc;

use anyhow::Result;
use ndarray::Array2;
use spnl_core::ir::{Message, Query};

use crate::tokenizer::Tokenizer;

/// Trait for computing embeddings from token IDs — abstracts over
/// pipeline-based (sync LLM) and channel-based (async server) paths.
pub trait TokenEmbedder: Send + Sync {
    fn embed_tokens(&self, token_id_seqs: Vec<Vec<u32>>) -> Result<Vec<Vec<f32>>>;
}

/// Pipeline-based embedder for the sync LLM path.
impl TokenEmbedder for scratchy_serving_engine::core_client::EmbedSender {
    fn embed_tokens(&self, token_id_seqs: Vec<Vec<u32>>) -> Result<Vec<Vec<f32>>> {
        self.embed(token_id_seqs)
            .map_err(|e| anyhow::anyhow!("{e}"))
    }
}

/// Async engine embedder for the server path.
pub struct AsyncEngineEmbedder {
    engine: Arc<crate::engine::AsyncEngine>,
}

impl AsyncEngineEmbedder {
    pub fn new(engine: Arc<crate::engine::AsyncEngine>) -> Self {
        Self { engine }
    }
}

impl TokenEmbedder for AsyncEngineEmbedder {
    fn embed_tokens(&self, token_id_seqs: Vec<Vec<u32>>) -> Result<Vec<Vec<f32>>> {
        self.engine.embed_sync(token_id_seqs)
    }
}

/// Extract text content from a Query tree for embedding.
pub fn contentify(input: &Query) -> Vec<String> {
    match input {
        Query::Seq(v) | Query::Plus(v) | Query::Cross(v) => v.iter().flat_map(contentify).collect(),
        Query::Message(Message::Assistant(s) | Message::System(s) | Message::User(s)) => {
            if s.is_empty() {
                vec![]
            } else {
                vec![s.clone()]
            }
        }
        _ => vec![],
    }
}

/// An embedding provider that calls an OpenAI-compatible `/v1/embeddings` endpoint.
///
/// Configure via:
/// - `VLLM_EMBEDDING_BASE_URL` — base URL (default: `http://localhost:11434/v1`)
/// - `VLLM_EMBEDDING_API_KEY` — optional API key
pub struct HttpEmbeddingProvider {
    pub model: String,
    pub dimensions: usize,
    base_url: String,
    api_key: Option<String>,
    agent: ureq::Agent,
}

impl HttpEmbeddingProvider {
    #[allow(dead_code)]
    pub fn new(model: &str, dimensions: usize) -> Self {
        let base_url = std::env::var("VLLM_EMBEDDING_BASE_URL")
            .unwrap_or_else(|_| "http://localhost:11434/v1".to_string());
        Self::with_base_url(model, dimensions, base_url)
    }

    /// Create a provider targeting an explicit base URL (e.g. a sidecar).
    pub fn with_base_url(model: &str, dimensions: usize, base_url: String) -> Self {
        let api_key = std::env::var("VLLM_EMBEDDING_API_KEY").ok();
        Self {
            model: model.to_string(),
            dimensions,
            base_url,
            api_key,
            agent: ureq::Agent::config_builder()
                .http_status_as_error(false)
                .build()
                .into(),
        }
    }

    /// Probe the embedding endpoint to detect dimensions.
    #[allow(dead_code)]
    pub fn probe_dimensions(model: &str) -> Result<usize> {
        let provider = Self::new(model, 0);
        let resp = provider.call_api(&["probe".to_string()])?;
        Ok(resp[0].len())
    }

    pub(crate) fn call_api(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>> {
        let url = format!("{}/embeddings", self.base_url);
        let body = serde_json::json!({
            "model": self.model,
            "input": inputs,
        });
        let mut req = self
            .agent
            .post(&url)
            .header("content-type", "application/json");
        if let Some(key) = &self.api_key {
            req = req.header("authorization", format!("Bearer {key}"));
        }
        let http = req.send(body.to_string().as_str())?;
        let status = http.status();
        let text = http.into_body().read_to_string()?;
        if !status.is_success() {
            anyhow::bail!("embedding endpoint {url} returned {status}: {text}");
        }
        let resp: serde_json::Value = serde_json::from_str(&text)?;
        let data = resp["data"]
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("missing 'data' in embedding response"))?;
        data.iter()
            .map(|item| {
                item["embedding"]
                    .as_array()
                    .ok_or_else(|| anyhow::anyhow!("missing 'embedding' in response item"))
                    .map(|arr: &Vec<serde_json::Value>| {
                        arr.iter()
                            .filter_map(|v: &serde_json::Value| v.as_f64().map(|f| f as f32))
                            .collect::<Vec<f32>>()
                    })
            })
            .collect()
    }
}

/// An embedding provider that routes through the in-process vllm-rs engine.
/// Used when the current engine serves the embedding model.
pub struct InProcessEmbeddingProvider {
    pub model: String,
    pub dimensions: usize,
    embedder: Arc<dyn TokenEmbedder>,
    tokenizer: Arc<Tokenizer>,
}

impl InProcessEmbeddingProvider {
    pub fn new(
        model: String,
        dimensions: usize,
        embedder: Arc<dyn TokenEmbedder>,
        tokenizer: Arc<Tokenizer>,
    ) -> Self {
        Self {
            model,
            dimensions,
            embedder,
            tokenizer,
        }
    }

    /// Probe the in-process engine to detect embedding dimensions.
    pub fn probe_dimensions(embedder: &dyn TokenEmbedder, tokenizer: &Tokenizer) -> Result<usize> {
        let token_ids = tokenizer.encode("probe", false)?;
        let vecs = embedder.embed_tokens(vec![token_ids])?;
        vecs.first()
            .map(|v| v.len())
            .ok_or_else(|| anyhow::anyhow!("probe returned no embeddings"))
    }
}

impl leann_core::embedding::EmbeddingProvider for InProcessEmbeddingProvider {
    fn compute_embeddings(
        &self,
        chunks: &[String],
        // leann-core grew a progress sink on this trait; neither provider has
        // anything incremental to report, so both ignore it.
        _progress: Option<&dyn leann_core::hnsw::IndexProgress>,
    ) -> Result<Array2<f32>> {
        // Tokenize each chunk
        let token_id_seqs: Vec<Vec<u32>> = chunks
            .iter()
            .map(|text| {
                self.tokenizer
                    .encode(text, false)
                    .map_err(|e| anyhow::anyhow!("tokenization failed: {e}"))
            })
            .collect::<Result<Vec<_>>>()?;

        // Embed via the in-process engine
        let vecs = self.embedder.embed_tokens(token_id_seqs)?;

        let nrows = vecs.len();
        let ncols = self.dimensions;
        let mut data = Vec::with_capacity(nrows * ncols);
        for v in &vecs {
            if v.len() < ncols {
                data.extend_from_slice(v);
                data.resize(data.len() + ncols - v.len(), 0.0);
            } else {
                data.extend_from_slice(&v[..ncols]);
            }
        }
        Ok(Array2::from_shape_vec((nrows, ncols), data)?)
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }

    fn name(&self) -> &str {
        &self.model
    }
}

impl leann_core::embedding::EmbeddingProvider for HttpEmbeddingProvider {
    fn compute_embeddings(
        &self,
        chunks: &[String],
        // leann-core grew a progress sink on this trait; neither provider has
        // anything incremental to report, so both ignore it.
        _progress: Option<&dyn leann_core::hnsw::IndexProgress>,
    ) -> Result<Array2<f32>> {
        let vecs = self.call_api(chunks)?;
        let nrows = vecs.len();
        let ncols = self.dimensions;
        let mut data = Vec::with_capacity(nrows * ncols);
        for v in &vecs {
            if v.len() < ncols {
                data.extend_from_slice(v);
                data.resize(data.len() + ncols - v.len(), 0.0);
            } else {
                data.extend_from_slice(&v[..ncols]);
            }
        }
        Ok(Array2::from_shape_vec((nrows, ncols), data)?)
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }

    fn name(&self) -> &str {
        &self.model
    }
}
