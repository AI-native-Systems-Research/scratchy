use std::path::PathBuf;

use anyhow::Result;
use itertools::Itertools;
use spnl_core::ir::{Document, Query};

use leann_core::hnsw::{
    graph::VectorStorage,
    io::read_hnsw_index,
    search::{SearchParams, search_hnsw},
};
use leann_core::index::IndexPaths;
use leann_core::passages::{PassageManager, load_id_map};

use super::embed::{HttpEmbeddingProvider, contentify};
use super::options::AugmentOptions;

/// Sanitize a name for use as a filesystem path component.
fn sanitize_name(name: &str) -> String {
    name.replace(['/', ':'], "_")
}

/// Retrieve relevant document fragments using HNSW vector search.
///
/// Returns formatted strings like `"Relevant Document @base-doc.txt-3: <text>"`.
pub async fn retrieve(
    embedding_model: &str,
    body: &Query,
    (filename, _content): &(String, Document),
    options: &AugmentOptions,
) -> Result<Vec<String>> {
    let max_matches = options.max_aug;

    // Derive the index path (must match index.rs naming)
    let index_name = sanitize_name(&format!(
        "default.{embedding_model}.{filename}.SimpleEmbedRetrieve",
    ));
    let index_dir = PathBuf::from(&options.index_dir);
    let index_path = index_dir.join(format!("{index_name}.leann"));
    let paths = IndexPaths::new(&index_path);

    // Load HNSW graph from disk
    let mut index_file = std::fs::File::open(paths.index_file_path())?;
    let graph = read_hnsw_index(&mut index_file)?;

    // Extract stored vectors
    let stored_vectors: Vec<f32> = match &graph.vector_storage {
        VectorStorage::Raw { data, .. } => data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect(),
        VectorStorage::Null => {
            anyhow::bail!("HNSW index has no stored vectors; was it built with is_recompute=false?")
        }
    };

    // Load passage manager and ID map
    let passage_source = leann_core::index::PassageSource {
        source_type: "jsonl".to_string(),
        path: paths.passages_path().to_string_lossy().to_string(),
        index_path: paths.offset_path().to_string_lossy().to_string(),
        path_relative: None,
        index_path_relative: None,
    };
    let passages = PassageManager::load(&[passage_source], None)?;
    let id_map = load_id_map(&paths.id_map_path())?;

    // Embed the query body — in-process if model matches, otherwise HTTP
    let body_texts = contentify(body);
    let body_vectors: Vec<Vec<f32>> = if options.can_embed_in_process(embedding_model) {
        let embedder = options.embedder.clone().unwrap();
        let tokenizer = options.tokenizer.clone().unwrap();
        tokio::task::spawn_blocking(move || {
            let token_id_seqs: Vec<Vec<u32>> = body_texts
                .iter()
                .map(|text| tokenizer.encode(text, false))
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(|e| anyhow::anyhow!("tokenization failed: {e}"))?;
            embedder.embed_tokens(token_id_seqs)
        })
        .await??
    } else {
        let base_url = resolve_embedding_base_url(embedding_model, options)?;
        let embedding_model_owned = embedding_model.to_string();
        tokio::task::spawn_blocking(move || {
            let provider =
                HttpEmbeddingProvider::with_base_url(&embedding_model_owned, 0, base_url);
            provider.call_api(&body_texts)
        })
        .await??
    };

    // Search for each query vector
    let params = SearchParams::default();
    let matching_labels: Vec<usize> = body_vectors
        .into_iter()
        .flat_map(|query_vec| {
            let (labels, _distances) =
                search_hnsw(&graph, &query_vec, max_matches, &stored_vectors, &params);
            labels.into_iter()
        })
        .unique()
        .collect();

    // Resolve passage text — reversed so most relevant is closest to query
    let fragments: Vec<String> = matching_labels
        .into_iter()
        .rev()
        .filter_map(|label| {
            let id = id_map.get(label).map(|s| s.as_str()).unwrap_or("?");
            passages
                .get_passage_by_index(label)
                .ok()
                .map(|p| format!("Relevant Document {id}: {}", p.text))
        })
        .collect();

    Ok(fragments)
}

/// Resolve the base URL for an embedding model: use a sidecar if available,
/// otherwise fall back to the env-var default.
fn resolve_embedding_base_url(model: &str, options: &AugmentOptions) -> Result<String> {
    if let Some(mgr) = &options.sidecar_manager {
        let sidecar = mgr.get_or_spawn(model)?;
        Ok(sidecar.base_url.clone())
    } else {
        Ok(std::env::var("VLLM_EMBEDDING_BASE_URL")
            .unwrap_or_else(|_| "http://localhost:11434/v1".to_string()))
    }
}
