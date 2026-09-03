// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! `scr model pull` — download a model from HuggingFace Hub.

use std::path::{Path, PathBuf};

use hf_hub_downloader as hfdl;
use tracing::info;

use crate::args::PullArgs;

pub async fn run_pull(args: PullArgs) -> anyhow::Result<()> {
    let model = &args.model;
    let path = Path::new(model);

    // Local .gguf file — nothing to download.
    if path.is_file() && path.extension().is_some_and(|e| e == "gguf") {
        println!("Model already available locally: {}", path.display());
        return Ok(());
    }

    // Local directory — nothing to download.
    if path.is_dir() {
        println!("Model already available locally: {}", path.display());
        return Ok(());
    }

    // Download from HuggingFace Hub.
    let model = model.clone();
    let hf_token = args.hf_token.clone();
    let gguf_file = args.gguf_file.clone();
    let quantization = args.quantization.clone();

    let result_path = tokio::task::spawn_blocking(move || {
        download_model(
            &model,
            hf_token.as_deref(),
            gguf_file.as_deref(),
            quantization.as_deref(),
        )
    })
    .await??;

    println!("Model downloaded to: {}", result_path.display());
    Ok(())
}

fn download_model(
    model_id: &str,
    hf_token: Option<&str>,
    gguf_file: Option<&str>,
    quantization: Option<&str>,
) -> anyhow::Result<PathBuf> {
    info!("Downloading model from HuggingFace Hub: {model_id}");

    // Five attempts per file, shared across the ranges each file splits
    // into. A retry restarts from the offset the resume log recorded rather
    // than from zero, so a transient timeout costs the bytes since the last
    // checkpoint instead of the whole download.
    let mut builder = hfdl::Client::builder().retries(5);
    if let Some(token) = hf_token {
        builder = builder.token(Some(token.to_string()));
    }
    let client = builder.build();
    let repo = client.model(model_id);

    // GGUF download: explicit filename, --quantization match, or auto-detect.
    let gguf_filename = gguf_file.map(String::from).or_else(|| {
        // If --quantization is set, always try to find a matching GGUF file.
        // Otherwise, only auto-detect for repos with "GGUF" in the name.
        if quantization.is_none() && !model_id.to_ascii_uppercase().contains("GGUF") {
            return None;
        }
        let mut gguf_files: Vec<String> = repo
            .files()
            .ok()?
            .into_iter()
            .filter(|f| f.ends_with(".gguf"))
            .collect();
        if gguf_files.is_empty() {
            return None;
        }
        // If user specified a quantization, match case-insensitively.
        if let Some(quant) = quantization {
            let quant_upper = quant.to_ascii_uppercase();
            if let Some(f) = gguf_files
                .iter()
                .find(|s| s.to_ascii_uppercase().contains(&quant_upper))
            {
                return Some(f.clone());
            }
            // No match — warn but fall through to default selection.
            eprintln!("Warning: no GGUF file matching '{quant}' found, using default selection");
        }
        for pattern in &["Q4_K_M", "Q4_K_S", "Q4_K", "Q4_0", "Q8_0"] {
            if let Some(f) = gguf_files.iter().find(|s| s.contains(pattern)) {
                return Some(f.clone());
            }
        }
        gguf_files.sort();
        Some(gguf_files.swap_remove(0))
    });

    if let Some(ref gguf_file) = gguf_filename {
        info!("Downloading GGUF file: {gguf_file}");
        let gguf_path = repo.get(gguf_file)?;
        // Best-effort tokenizer download.
        let _ = repo.get("tokenizer.json");
        let _ = repo.get("tokenizer_config.json");
        // Gemma-4 ships the chat template as a standalone sibling file.
        let _ = repo.get("chat_template.jinja");
        return Ok(gguf_path);
    }

    // Safetensors model.
    let config_path = repo.get("config.json")?;
    let model_dir = config_path.parent().unwrap().to_path_buf();

    // Best-effort tokenizer download.
    let _ = repo.get("tokenizer.json");
    let _ = repo.get("tokenizer_config.json");
    // Gemma-4 ships the chat template as a standalone sibling file.
    let _ = repo.get("chat_template.jinja");

    // Single file model. "Absent" is the sharded case; any other failure is
    // real and must not be reported as "no safetensors weights found".
    match repo.get("model.safetensors") {
        Ok(_) => return Ok(model_dir),
        Err(e) if e.is_not_found() => {}
        Err(e) => anyhow::bail!("failed to download model.safetensors for {model_id}: {e}"),
    }

    // Sharded model.
    if let Ok(index_path) = repo.get("model.safetensors.index.json") {
        let index = scratchy_core_model::weight::SafeTensorsIndex::from_file(&index_path)?;
        let sorted_shards = index.shard_files();
        let total = sorted_shards.len();

        let needed: Vec<&String> = sorted_shards
            .iter()
            .filter(|s| !model_dir.join(s).exists())
            .collect();

        // Fail fast on a full disk rather than starting a doomed multi-GB
        // download (and burning the downloader's retry budget on ENOSPC).
        index
            .ensure_disk_space(&model_dir)
            .map_err(|e| anyhow::anyhow!("cannot download {model_id}: {e}"))?;

        if needed.is_empty() {
            info!("All {total} shard files already cached");
        } else {
            info!(
                "Downloading {} of {total} shard files (up to 8 in parallel)",
                needed.len()
            );

            let multi = indicatif::MultiProgress::new();
            const MAX_PARALLEL: usize = 8;
            let repo = &repo;
            let multi = &multi;

            for chunk in needed.chunks(MAX_PARALLEL) {
                let results: Vec<anyhow::Result<()>> = std::thread::scope(|s| {
                    let handles: Vec<_> = chunk
                        .iter()
                        .map(|shard| {
                            let mut bar = Bar(multi.add(indicatif::ProgressBar::new(0)));
                            s.spawn(move || {
                                repo.download(shard, &mut bar)
                                    .map(|_| ())
                                    .map_err(|e| anyhow::anyhow!("failed to download {shard}: {e}"))
                            })
                        })
                        .collect();
                    handles.into_iter().map(|h| h.join().unwrap()).collect()
                });
                for result in results {
                    result?;
                }
            }
        }
        return Ok(model_dir);
    }

    anyhow::bail!(
        "no safetensors weights found for {model_id} \
         (neither model.safetensors nor model.safetensors.index.json)"
    );
}

/// Drives an `indicatif` bar from download progress.
///
/// The downloader has no `indicatif` dependency by design — it defines the
/// trait and leaves rendering to whoever owns the terminal — so each caller
/// supplies its own few lines of glue.
struct Bar(indicatif::ProgressBar);

impl hfdl::Progress for Bar {
    fn init(&mut self, total: u64, filename: &str) {
        self.0.set_length(total);
        self.0.set_message(filename.to_string());
    }
    fn update(&mut self, delta: u64) {
        self.0.inc(delta);
    }
    fn finish(&mut self) {
        self.0.finish();
    }
}
