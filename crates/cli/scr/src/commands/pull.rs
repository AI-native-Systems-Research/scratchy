// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! `scr model pull` — download a model from HuggingFace Hub.

use std::path::Path;

use scratchy_core_model::hub::{self, Gguf};

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
        let gguf = match (gguf_file.as_deref(), quantization.as_deref()) {
            (Some(file), _) => Gguf::File(file),
            (None, Some(quant)) => Gguf::Matching(quant),
            (None, None) => Gguf::Auto,
        };
        hub::download(&model, hf_token.as_deref(), gguf, None)
    })
    .await??;

    println!("Model downloaded to: {}", result_path.display());
    Ok(())
}
