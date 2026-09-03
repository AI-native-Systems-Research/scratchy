// SPDX-License-Identifier: Apache-2.0
//! Spyre weight-fill validation (loading half): the disk names the `#[forward]`
//! macro wrote into the KTIR recipe load to sane f32 through scratchy's own
//! `GpuWeights` loader under the host `SpyreAllocator` — no GPU, no name
//! reconstruction. Proves the recipe's `disk` keys are real + the loader runs
//! under spyre. (The `[out,in]→[in,out]` transpose for gemm weights, the
//! runtime sources, and the full forward are the worker's job.)
#![cfg(feature = "spyre")]

use scratchy_target_cuda::{GpuWeights, SpyreAllocator};
use scratchy_target_spyre::manifest::Manifest;
use std::collections::HashSet;
use std::path::PathBuf;

fn home() -> PathBuf {
    PathBuf::from(std::env::var("HOME").unwrap())
}

/// First snapshot dir of a cached HF repo, or None.
fn hf_snapshot(repo: &str) -> Option<PathBuf> {
    let snaps = home().join(format!(".cache/huggingface/hub/{repo}/snapshots"));
    std::fs::read_dir(snaps)
        .ok()?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .find(|p| p.is_dir())
}

#[test]
fn weight_recipe_loads_to_f32_under_spyre() {
    let bundle = home().join(".cache/cudaforge/ktir/smollm2-135m");
    if !bundle.join("manifest.json").exists() {
        eprintln!("skip: no bundle — build -p scratchy-models --features spyre,arch-llama");
        return;
    }
    let Some(model_dir) = hf_snapshot("models--HuggingFaceTB--SmolLM2-135M") else {
        eprintln!("skip: SmolLM2-135M not in HF cache");
        return;
    };

    let manifest = Manifest::load(&bundle).expect("load manifest");
    let mut weights = GpuWeights::from_dir(&model_dir, SpyreAllocator::new())
        .expect("GpuWeights::from_dir under SpyreAllocator");

    let mut loaded = 0usize;
    let mut seen = HashSet::new();
    for s in &manifest.sources {
        if s.role != "weight" {
            continue;
        }
        let disk = s.disk.as_ref().expect("weight source carries a disk key");
        // Tied lm_head: no `lm_head.weight` on disk — it's the embedding.
        let name = if disk == "lm_head" {
            "model.embed_tokens.weight".to_string()
        } else {
            format!("{disk}.weight")
        };
        // `take_to_cpu_f32` removes the tensor, so take each unique key once.
        if !seen.insert(name.clone()) {
            continue;
        }
        let data = weights
            .take_to_cpu_f32(&name)
            .unwrap_or_else(|e| panic!("load {name}: {e}"));
        assert!(!data.is_empty(), "{name} empty");
        assert!(data.iter().all(|x| x.is_finite()), "{name} has non-finite");
        let max_abs = data.iter().fold(0f32, |m, x| m.max(x.abs()));
        assert!(
            max_abs < 100.0,
            "{name} max|w|={max_abs} — implausible for an LLM weight"
        );
        loaded += 1;
    }

    eprintln!("loaded {loaded} unique weight tensors to f32 under SpyreAllocator");
    // 272 weight sources, minus the lm_head/embed_tokens tie collapse → ~271.
    assert!(loaded >= 250, "expected ~271 weights, loaded {loaded}");
}
