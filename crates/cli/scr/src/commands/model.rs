// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! `scr model ls` — list cached models.

use std::path::{Path, PathBuf};

use crate::args::{ListArgs, ListSort, RmArgs};

pub async fn run_model_list(args: ListArgs) -> anyhow::Result<()> {
    let cache_dir = hf_cache_dir();
    if !cache_dir.is_dir() {
        println!(
            "No cached models found (cache dir: {})",
            cache_dir.display()
        );
        return Ok(());
    }

    let mut models: Vec<CachedModel> = Vec::new();

    for (model_id, dir) in cached_models(&cache_dir) {
        // Get total size of blobs.
        let blobs_dir = dir.join("blobs");
        let size = if blobs_dir.is_dir() {
            dir_size(&blobs_dir)
        } else {
            0
        };

        // List files in the latest snapshot.
        let snapshots_dir = dir.join("snapshots");
        let mut weight_files = Vec::new();
        if let Some(snap) = latest_snapshot(&dir, &snapshots_dir) {
            for e in std::fs::read_dir(&snap).into_iter().flatten().flatten() {
                let fname = e.file_name().to_string_lossy().to_string();
                if fname.ends_with(".safetensors")
                    || fname.ends_with(".gguf")
                    || fname.ends_with(".bin")
                {
                    weight_files.push(fname);
                }
            }
        }
        weight_files.sort();

        models.push(CachedModel {
            model_id,
            size,
            weight_files,
        });
    }

    match args.sort {
        ListSort::Name => models.sort_by(|a, b| a.model_id.cmp(&b.model_id)),
        ListSort::Size => models.sort_by_key(|b| std::cmp::Reverse(b.size)),
    }

    if models.is_empty() {
        println!("No cached models found.");
        return Ok(());
    }

    println!("{:<50} {:>10}  FILES", "MODEL", "SIZE");
    for m in &models {
        let files_summary = if m.weight_files.len() <= 3 {
            m.weight_files.join(", ")
        } else {
            format!(
                "{}, ... ({} total)",
                m.weight_files[..2].join(", "),
                m.weight_files.len()
            )
        };
        println!(
            "{:<50} {:>10}  {}",
            m.model_id,
            format_size(m.size),
            files_summary,
        );
    }
    println!(
        "\nTotal: {} model(s), {}",
        models.len(),
        format_size(models.iter().map(|m| m.size).sum())
    );

    Ok(())
}

struct CachedModel {
    model_id: String,
    size: u64,
    weight_files: Vec<String>,
}

/// Every model in the hf-hub cache as `("org/name", <its cache dir>)`.
///
/// The one place that knows hub cache directories are `models--org--name`
/// (`run_model_rm` below mangles the same mapping in reverse). Shared by
/// `scr model ls` and by completion's `--source cached`, so the two can never
/// disagree about what is cached — `scr model rm <TAB>` completes exactly the
/// ids `scr model ls` prints, which is what `rm`'s own error message promises.
/// Silent on an unreadable cache dir: an absent cache is "nothing cached", not
/// an error.
pub(crate) fn cached_models(cache_dir: &Path) -> Vec<(String, PathBuf)> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(cache_dir).into_iter().flatten().flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        let Some(rest) = name.strip_prefix("models--") else {
            continue;
        };
        let dir = entry.path();
        if dir.is_dir() {
            out.push((rest.replace("--", "/"), dir));
        }
    }
    out
}

pub async fn run_model_rm(args: RmArgs) -> anyhow::Result<()> {
    let cache_dir = hf_cache_dir();
    // "org/name" → "models--org--name"
    let dir_name = format!("models--{}", args.model.replace('/', "--"));
    let model_dir = cache_dir.join(&dir_name);

    if !model_dir.is_dir() {
        anyhow::bail!(
            "Model '{}' not found in cache. Run `scr model ls` to see cached models.",
            args.model
        );
    }

    let size = dir_size(&model_dir);
    println!("Removing {} ({})...", args.model, format_size(size));
    std::fs::remove_dir_all(&model_dir)?;
    println!("Done.");
    Ok(())
}

/// `scr model names [--source <src>] <prefix>` — one candidate per line, for
/// shell tab completion. Hidden; invoked by the scripts `scr completions`
/// generates, not meant to be typed.
///
/// Completion is advisory, not validation: `-m` accepts any string (a local
/// path or any Hub id — see `resolve_model_path`), and nothing maps a model NAME
/// to a compiled backbone, so an empty list here must never read as "invalid".
///
/// Each source is exactly one candidate set — no union. An earlier version had
/// the default source union the registry with the local hf-hub cache, so that
/// completion stayed useful when no registry was resolved. That was a mistake:
/// the cache is not scoped to the build, so a `-Fmodel/llama-3.2-3b,quant/mlx`
/// binary offered cached granite checkpoints, non-MLX checkpoints, and
/// `modernbert-embed-base` (not even a decoder) — and when the registry was
/// empty it returned the *same* list for every `-Fmodel` scope, which reads
/// exactly like a broken scope filter. Offering nothing is strictly better than
/// offering plausible-looking ids this binary cannot load.
///
/// Filtering the cache by compiled arch/shape/quant — which needs no network,
/// since every cached repo has a local `config.json` — would make it a valid
/// source and restore useful completion for air-gapped builds. Tracked in #26.
#[cfg(feature = "model")]
pub async fn run_model_names(args: crate::args::NamesArgs) -> anyhow::Result<()> {
    use crate::args::NamesSource;
    use std::collections::BTreeSet;

    // BTreeSet: sorted and deduped.
    let names: BTreeSet<String> = match args.source {
        // Baked in at BUILD time by `crates/models/arch/hf_registry_build.rs`,
        // scoped to exactly what `-Fmodel/<stem>` / `-Fquant/<preset>` compiled.
        // Empty when this build resolved no registry. No runtime network.
        NamesSource::Compiled => model::compiled_hf_registry()
            .iter()
            .map(|s| s.to_string())
            .collect(),
        NamesSource::Cached => cached_models(&hf_cache_dir())
            .into_iter()
            .map(|(id, _)| id)
            .collect(),
        NamesSource::Stems => compiled_variant_stems().into_iter().collect(),
    };

    for name in names.iter().filter(|n| n.starts_with(&args.prefix)) {
        println!("{name}");
    }
    Ok(())
}

/// Compiled config stems (`llama-3.2-1b`, `smollm2-135m`) — what `scr model
/// info` filters on, from the same `BackboneDumpRegistration` inventory that
/// command walks.
///
/// NOT offered for `-m`: a stem is a config file name, not a model argument, and
/// suggesting one there would send the user down a path that cannot resolve.
#[cfg(all(feature = "model", any(feature = "cuda", feature = "metal")))]
fn compiled_variant_stems() -> Vec<String> {
    scratchy_forward_compiler::inventory::iter::<
        scratchy_forward_compiler::BackboneDumpRegistration,
    >
    .into_iter()
    .flat_map(|reg| (reg.dump_all)())
    .map(|variant| variant.variant_stem.to_string())
    .collect()
}

/// No backend compiled in means no backbones were emitted, so there are no
/// stems to report (`scr model info` does not exist in such a build either).
#[cfg(all(feature = "model", not(any(feature = "cuda", feature = "metal"))))]
fn compiled_variant_stems() -> Vec<String> {
    Vec::new()
}

pub(crate) fn hf_cache_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("HF_HOME") {
        return PathBuf::from(dir).join("hub");
    }
    if let Ok(dir) = std::env::var("HUGGINGFACE_HUB_CACHE") {
        return PathBuf::from(dir);
    }
    PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| ".".to_string()))
        .join(".cache/huggingface/hub")
}

fn latest_snapshot(model_dir: &Path, snapshots_dir: &Path) -> Option<PathBuf> {
    // Try to resolve the "main" ref first.
    let refs_main = model_dir.join("refs/main");
    if let Ok(hash) = std::fs::read_to_string(&refs_main) {
        let snap = snapshots_dir.join(hash.trim());
        if snap.is_dir() {
            return Some(snap);
        }
    }
    // Fallback: first snapshot directory.
    std::fs::read_dir(snapshots_dir)
        .ok()?
        .flatten()
        .find(|e| e.path().is_dir())
        .map(|e| e.path())
}

fn dir_size(path: &Path) -> u64 {
    let mut total = 0u64;
    if let Ok(entries) = std::fs::read_dir(path) {
        for entry in entries.flatten() {
            let ft = entry.file_type();
            if let Ok(ft) = ft {
                if ft.is_file() || ft.is_symlink() {
                    total += entry.metadata().map(|m| m.len()).unwrap_or(0);
                } else if ft.is_dir() {
                    total += dir_size(&entry.path());
                }
            }
        }
    }
    total
}

pub(crate) fn format_size(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;
    if bytes >= GB {
        format!("{:.1} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.0} KB", bytes as f64 / KB as f64)
    } else {
        format!("{bytes} B")
    }
}
