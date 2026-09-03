use std::sync::Arc;

use crate::tokenizer::Tokenizer;

use super::embed::TokenEmbedder;
use super::sidecar::SidecarManager;

/// Options controlling RAG augmentation behavior.
#[derive(Clone)]
pub struct AugmentOptions {
    /// Max fragments to retrieve per Augment node.
    pub max_aug: usize,
    /// Directory where LEANN indexes are stored.
    pub index_dir: String,
    /// Chunk size for sentence-based chunking (characters).
    pub chunk_size: usize,
    /// Chunk overlap for sentence-based chunking (characters).
    pub chunk_overlap: usize,
    /// Model name served by the current engine (for in-process embedding).
    pub current_model: Option<String>,
    /// In-process token embedder (set when the engine can compute embeddings).
    pub embedder: Option<Arc<dyn TokenEmbedder>>,
    /// Tokenizer for in-process embedding.
    pub tokenizer: Option<Arc<Tokenizer>>,
    /// Sidecar manager — spawns vllm-rs embedding processes on demand when
    /// the current model cannot serve embeddings in-process.
    pub sidecar_manager: Option<Arc<SidecarManager>>,
}

impl Default for AugmentOptions {
    fn default() -> Self {
        Self {
            max_aug: 10,
            index_dir: "data/spnl".to_string(),
            chunk_size: 512,
            chunk_overlap: 50,
            current_model: None,
            embedder: None,
            tokenizer: None,
            sidecar_manager: None,
        }
    }
}

impl AugmentOptions {
    /// Whether the current engine can serve embeddings for the given model
    /// in-process (model matches and embedder is available).
    pub fn can_embed_in_process(&self, embedding_model: &str) -> bool {
        self.current_model.as_deref() == Some(embedding_model)
            && self.embedder.is_some()
            && self.tokenizer.is_some()
    }
}
