// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! `scr model cache clean` — reclaim disk space from on-disk model caches.
//!
//! Scans two caches:
//!   1. The HuggingFace Hub cache (`~/.cache/huggingface/hub`, downloaded
//!      checkpoints) — one entry per `models--org--name` directory.
//!   2. The scratchy weight cache (`~/.cache/scratchy/metal-aligned-weights`,
//!      realigned weight blobs scratchy writes so it can mmap them zero-copy
//!      on relaunch) — one entry per `<hash>-<stem>` sidecar group.
//!
//! By default it removes entries not used in the last 30 days (`--days N`
//! overrides). `--nuke` ignores the age filter and targets everything;
//! `--force` skips the interactive y/N confirmation. This is pure
//! filesystem work, so the command is available on every build (cuda,
//! metal, cpu) regardless of which backend produced the caches.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use super::model::{format_size, hf_cache_dir};
use crate::args::{CacheCleanArgs, CacheInspectArgs, ListSort};

/// Seconds in a day.
const DAY_SECS: u64 = 86_400;

/// One removable cache item: a HF model directory, or one weight-cache
/// sidecar group (`.bin` + `.meta.json` + `.lock` + any orphaned `.tmp`).
struct CacheEntry {
    /// Human-readable identity, e.g. `meta-llama/Llama-3.2-1B` or the
    /// sidecar's source filename (`model-00001-of-00002.safetensors`).
    label: String,
    /// Every path to delete for this entry.
    paths: Vec<PathBuf>,
    /// Total bytes the entry occupies on disk.
    size: u64,
    /// Most recent access-or-modify time across the entry's files — our
    /// "last used" proxy. `UNIX_EPOCH` means unknown (treated as ancient).
    last_used: SystemTime,
}

pub async fn run_cache_clean(args: CacheCleanArgs) -> anyhow::Result<()> {
    let now = SystemTime::now();

    let hf_root = hf_cache_dir();
    let weight_root = scratchy_core_common::cache::metal_aligned_weights_dir();

    let key_to_model = build_weight_key_to_model(&hf_root);
    let hf = select_stale(scan_hf_cache(&hf_root), args.nuke, args.days, now);
    let weights = select_stale(
        scan_weight_cache(&weight_root, &key_to_model),
        args.nuke,
        args.days,
        now,
    );

    let count = hf.len() + weights.len();
    if count == 0 {
        if args.nuke {
            println!("No caches found. Nothing to remove.");
        } else {
            println!(
                "Nothing to clean — no cached items unused for {}+ days. \
                 Use --nuke to remove everything.",
                args.days
            );
        }
        return Ok(());
    }

    print_section("HuggingFace Hub cache", &hf_root, &hf, now);
    print_section("Scratchy weight cache", &weight_root, &weights, now);

    let total: u64 = hf.iter().chain(&weights).map(|e| e.size).sum();
    println!("\n{count} item(s) to remove, {} total.", format_size(total));

    if !args.force {
        let prompt = if args.nuke {
            format!(
                "Remove ALL {count} cached item(s) and free {}?",
                format_size(total)
            )
        } else {
            format!(
                "Remove these {count} item(s) and free {}?",
                format_size(total)
            )
        };
        if !confirm(&prompt) {
            println!("Aborted. Nothing was removed.");
            return Ok(());
        }
    }

    let mut freed = 0u64;
    let mut removed = 0usize;
    for entry in hf.iter().chain(&weights) {
        match remove_entry(entry) {
            Ok(()) => {
                freed += entry.size;
                removed += 1;
            }
            Err(e) => eprintln!("warning: failed to remove {}: {e}", entry.label),
        }
    }

    println!("Freed {} across {removed} item(s).", format_size(freed));
    Ok(())
}

/// `scr model cache inspect` — read-only disk-consumption breakdown of both
/// caches. Reuses the scan + formatting logic from `clean`, but never
/// filters by age, prompts, or removes anything.
pub async fn run_cache_inspect(args: CacheInspectArgs) -> anyhow::Result<()> {
    let now = SystemTime::now();

    let hf_root = hf_cache_dir();
    let weight_root = scratchy_core_common::cache::metal_aligned_weights_dir();

    let key_to_model = build_weight_key_to_model(&hf_root);
    let mut hf = scan_hf_cache(&hf_root);
    let mut weights = scan_weight_cache(&weight_root, &key_to_model);
    sort_entries(&mut hf, &args.sort);
    sort_entries(&mut weights, &args.sort);

    if hf.is_empty() && weights.is_empty() {
        println!("No caches found.");
        return Ok(());
    }

    print_section_breakdown("HuggingFace Hub cache", &hf_root, &hf, now);
    print_section_breakdown("Scratchy weight cache", &weight_root, &weights, now);

    let total: u64 = hf.iter().chain(&weights).map(|e| e.size).sum();
    let count = hf.len() + weights.len();
    println!("\nTotal: {count} item(s), {}", format_size(total));
    Ok(())
}

/// Delete every path backing an entry (directory or file).
fn remove_entry(entry: &CacheEntry) -> std::io::Result<()> {
    for p in &entry.paths {
        if p.is_dir() {
            std::fs::remove_dir_all(p)?;
        } else {
            std::fs::remove_file(p)?;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Scanning
// ---------------------------------------------------------------------------

/// One entry per `models--org--name` directory under the HF hub cache.
fn scan_hf_cache(root: &Path) -> Vec<CacheEntry> {
    let mut out = Vec::new();
    let Ok(rd) = std::fs::read_dir(root) else {
        return out;
    };
    for e in rd.flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        let Some(stem) = name.strip_prefix("models--") else {
            continue;
        };
        let dir = e.path();
        if !dir.is_dir() {
            continue;
        }
        // "models--org--name" → "org/name".
        let model_id = stem.replace("--", "/");
        let mut size = 0u64;
        let mut last_used = SystemTime::UNIX_EPOCH;
        // Real bytes live in `blobs/`; `snapshots/` holds symlinks into
        // them. Walk the whole model dir but count regular files only and
        // never follow symlinks, so snapshot links don't double-count.
        walk_real_files(&dir, &mut size, &mut last_used);
        out.push(CacheEntry {
            label: model_id,
            paths: vec![dir],
            size,
            last_used,
        });
    }
    out
}

/// Scan the scratchy weight cache. Files are first grouped by their shared
/// `<hash>-<stem>` sidecar key (so the `.bin`, `.meta.json`, `.lock`, and
/// any orphaned `.tmp.<pid>` stay together), then folded into one entry per
/// **model** when `key_to_model` resolves the source — a sharded model's
/// sidecars collapse into a single labelled row. Sidecars whose source is
/// gone (deleted model, local-path load) stay as standalone rows labelled
/// by their safetensors filename.
fn scan_weight_cache(root: &Path, key_to_model: &BTreeMap<String, String>) -> Vec<CacheEntry> {
    // Pass 1: raw files → sidecar groups keyed by `<hash>-<stem>`.
    struct Group {
        paths: Vec<PathBuf>,
        size: u64,
        last_used: SystemTime,
        meta_json: Option<PathBuf>,
    }
    let mut groups: BTreeMap<String, Group> = BTreeMap::new();
    let Ok(rd) = std::fs::read_dir(root) else {
        return Vec::new();
    };
    for e in rd.flatten() {
        let Ok(md) = e.metadata() else {
            continue;
        };
        if !md.is_file() {
            continue;
        }
        let name = e.file_name().to_string_lossy().into_owned();
        let key = weight_group_key(&name).to_string();
        let g = groups.entry(key).or_insert_with(|| Group {
            paths: Vec::new(),
            size: 0,
            last_used: SystemTime::UNIX_EPOCH,
            meta_json: None,
        });
        if name.ends_with(".meta.json") {
            g.meta_json = Some(e.path());
        }
        g.paths.push(e.path());
        g.size += md.len();
        let lu = file_last_used(&md);
        if lu > g.last_used {
            g.last_used = lu;
        }
    }

    // Pass 2: fold sidecar groups into per-model entries when resolvable.
    // Resolution order: the sidecar's own `meta.json` src_path (works for
    // local-path loads too), then a reverse-hash of the HF cache (for
    // sidecars built before src_path was recorded), then the bare
    // filename. Resolved groups share a `model:<id>` fold key so shards
    // merge; unresolved groups keep their unique `sidecar:<key>` so
    // unrelated caches with the same filename never collide.
    let mut entries: BTreeMap<String, CacheEntry> = BTreeMap::new();
    for (key, g) in groups {
        let resolved = g
            .meta_json
            .as_deref()
            .and_then(model_from_meta_json)
            .or_else(|| key_to_model.get(&key).cloned());
        let (fold_key, label) = match resolved {
            Some(model) => (format!("model:{model}"), model),
            None => (format!("sidecar:{key}"), weight_label(&key)),
        };
        let entry = entries.entry(fold_key).or_insert_with(|| CacheEntry {
            label,
            paths: Vec::new(),
            size: 0,
            last_used: SystemTime::UNIX_EPOCH,
        });
        entry.paths.extend(g.paths);
        entry.size += g.size;
        if g.last_used > entry.last_used {
            entry.last_used = g.last_used;
        }
    }
    entries.into_values().collect()
}

/// Build a `<hash>-<stem>` → `org/model` map by replaying the metal
/// allocator's sidecar-naming hash over every safetensors file in the HF
/// cache. The hash (`DefaultHasher` over the canonicalized source path,
/// stem = the shard filename) is deterministic across runs, so the
/// sidecar names are reproducible and reversible to a model id here. All
/// snapshot revisions are scanned (not just `refs/main`) since a sidecar
/// may have been built from a model loaded at a pinned revision.
fn build_weight_key_to_model(hf_root: &Path) -> BTreeMap<String, String> {
    let mut map = BTreeMap::new();
    let Ok(rd) = std::fs::read_dir(hf_root) else {
        return map;
    };
    for e in rd.flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        let Some(stem_id) = name.strip_prefix("models--") else {
            continue;
        };
        let model_dir = e.path();
        if !model_dir.is_dir() {
            continue;
        }
        let model_id = stem_id.replace("--", "/");
        let snapshots = model_dir.join("snapshots");
        for rev in std::fs::read_dir(&snapshots)
            .into_iter()
            .flatten()
            .flatten()
        {
            let rev_dir = rev.path();
            if !rev_dir.is_dir() {
                continue;
            }
            for f in std::fs::read_dir(&rev_dir).into_iter().flatten().flatten() {
                let fname = f.file_name().to_string_lossy().into_owned();
                if !fname.ends_with(".safetensors") {
                    continue;
                }
                map.insert(sidecar_key_for_source(&f.path(), &fname), model_id.clone());
            }
        }
    }
    map
}

/// The sidecar group key (`<hash>-<stem>`) a given source safetensors file
/// would produce — mirrors `scratchy_target_metal::metal_allocator`'s
/// `aligned_cache_meta`.
fn sidecar_key_for_source(path: &Path, stem: &str) -> String {
    use std::hash::{Hash, Hasher};
    let canon = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    let mut h = std::collections::hash_map::DefaultHasher::new();
    canon.hash(&mut h);
    format!("{:016x}-{stem}", h.finish())
}

/// Read a sidecar's `meta.json` and derive its model label from the
/// recorded `src_path` (present on sidecars built after the field was
/// added). `None` when the file is unreadable or predates the field.
fn model_from_meta_json(meta_path: &Path) -> Option<String> {
    let text = std::fs::read_to_string(meta_path).ok()?;
    let v: serde_json::Value = serde_json::from_str(&text).ok()?;
    label_from_source_path(v.get("src_path")?.as_str()?)
}

/// Turn a source weight path into a model label: `org/name` from an HF
/// cache `models--org--name/...` path, otherwise the containing
/// directory's name (a local model dir). `None` for empty/degenerate
/// paths.
fn label_from_source_path(p: &str) -> Option<String> {
    let path = Path::new(p);
    for comp in path.components() {
        if let std::path::Component::Normal(os) = comp
            && let Some(rest) = os.to_string_lossy().strip_prefix("models--")
        {
            return Some(rest.replace("--", "/"));
        }
    }
    path.parent()
        .and_then(|d| d.file_name())
        .map(|n| n.to_string_lossy().into_owned())
        .filter(|s| !s.is_empty())
}

/// Recursively accumulate regular-file sizes and the newest last-used
/// time under `path`. Symlinks are skipped (HF snapshot links point into
/// `blobs/`, already counted).
fn walk_real_files(path: &Path, size: &mut u64, newest: &mut SystemTime) {
    let Ok(rd) = std::fs::read_dir(path) else {
        return;
    };
    for e in rd.flatten() {
        let Ok(ft) = e.file_type() else {
            continue;
        };
        if ft.is_symlink() {
            continue;
        }
        if ft.is_dir() {
            walk_real_files(&e.path(), size, newest);
        } else if ft.is_file()
            && let Ok(md) = e.metadata()
        {
            *size += md.len();
            let lu = file_last_used(&md);
            if lu > *newest {
                *newest = lu;
            }
        }
    }
}

/// Strip the recognized companion suffix from a weight-cache filename to
/// recover its `<hash>-<stem>` group key.
fn weight_group_key(name: &str) -> &str {
    for suffix in [".meta.json", ".bin", ".lock"] {
        if let Some(key) = name.strip_suffix(suffix) {
            return key;
        }
    }
    // Atomic-write temp left by a killed builder: "<hash>-<stem>.tmp.<pid>".
    if let Some(pos) = name.find(".tmp") {
        return &name[..pos];
    }
    name
}

/// `<16-hex-hash>-<stem>` → `<stem>` for display (the sidecar's source
/// filename). Falls back to the whole key if the hash prefix is absent.
fn weight_label(key: &str) -> String {
    key.split_once('-')
        .map(|(_, stem)| stem.to_string())
        .unwrap_or_else(|| key.to_string())
}

/// Most recent of a file's access and modify times.
fn file_last_used(md: &std::fs::Metadata) -> SystemTime {
    match (md.accessed().ok(), md.modified().ok()) {
        (Some(a), Some(m)) => a.max(m),
        (Some(t), None) | (None, Some(t)) => t,
        (None, None) => SystemTime::UNIX_EPOCH,
    }
}

// ---------------------------------------------------------------------------
// Selection
// ---------------------------------------------------------------------------

/// Keep entries eligible for removal, sorted largest-first. With `nuke`,
/// every entry qualifies; otherwise only those unused for `> days` days.
/// Unknown last-used (`UNIX_EPOCH`) counts as ancient → eligible; a
/// future timestamp (clock skew) is treated as fresh → kept.
fn select_stale(
    mut entries: Vec<CacheEntry>,
    nuke: bool,
    days: u64,
    now: SystemTime,
) -> Vec<CacheEntry> {
    if !nuke {
        let cutoff = Duration::from_secs(days.saturating_mul(DAY_SECS));
        entries.retain(|e| {
            now.duration_since(e.last_used)
                .map(|age| age > cutoff)
                .unwrap_or(false)
        });
    }
    entries.sort_by_key(|e| std::cmp::Reverse(e.size));
    entries
}

// ---------------------------------------------------------------------------
// Display
// ---------------------------------------------------------------------------

/// One aligned row: `<label>  <size>  <age>`.
fn format_entry_row(e: &CacheEntry, now: SystemTime) -> String {
    format!(
        "  {:<44}  {:>9}   {}",
        truncate(&e.label, 44),
        format_size(e.size),
        humanize_age(now, e.last_used),
    )
}

/// `clean`'s removal listing: header + rows, skipped entirely when empty.
fn print_section(title: &str, root: &Path, entries: &[CacheEntry], now: SystemTime) {
    if entries.is_empty() {
        return;
    }
    println!("\n{title}  ({})", root.display());
    for e in entries {
        println!("{}", format_entry_row(e, now));
    }
}

/// `inspect`'s breakdown: always prints the header (showing `(empty)`
/// when there's nothing) and a subtotal under the size column.
fn print_section_breakdown(title: &str, root: &Path, entries: &[CacheEntry], now: SystemTime) {
    println!("\n{title}  ({})", root.display());
    if entries.is_empty() {
        println!("  (empty)");
        return;
    }
    for e in entries {
        println!("{}", format_entry_row(e, now));
    }
    let subtotal: u64 = entries.iter().map(|e| e.size).sum();
    println!(
        "  {:<44}  {:>9}",
        format!("subtotal ({} item(s))", entries.len()),
        format_size(subtotal),
    );
}

/// Order entries for display: largest-first (`Size`) or alphabetical
/// (`Name`).
fn sort_entries(entries: &mut [CacheEntry], sort: &ListSort) {
    match sort {
        ListSort::Name => entries.sort_by(|a, b| a.label.cmp(&b.label)),
        ListSort::Size => entries.sort_by_key(|e| std::cmp::Reverse(e.size)),
    }
}

/// Truncate `s` to `max` columns with a trailing ellipsis when clipped.
fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let keep = max.saturating_sub(1);
        format!("{}…", s.chars().take(keep).collect::<String>())
    }
}

/// "today" / "1 day ago" / "N days ago", or "unknown" for `UNIX_EPOCH`.
fn humanize_age(now: SystemTime, t: SystemTime) -> String {
    if t == SystemTime::UNIX_EPOCH {
        return "unknown".to_string();
    }
    let secs = now.duration_since(t).map(|d| d.as_secs()).unwrap_or(0);
    match secs / DAY_SECS {
        0 => "today".to_string(),
        1 => "1 day ago".to_string(),
        n => format!("{n} days ago"),
    }
}

/// Print a prompt and read a y/N answer. Non-interactive input (EOF /
/// closed stdin) defaults to "no" so a piped invocation never deletes
/// without `--force`.
fn confirm(prompt: &str) -> bool {
    use std::io::Write;
    print!("{prompt} [y/N] ");
    let _ = std::io::stdout().flush();
    let mut line = String::new();
    match std::io::stdin().read_line(&mut line) {
        Ok(0) | Err(_) => false,
        Ok(_) => matches!(line.trim().to_ascii_lowercase().as_str(), "y" | "yes"),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn entry(label: &str, size: u64, last_used: SystemTime) -> CacheEntry {
        CacheEntry {
            label: label.to_string(),
            paths: Vec::new(),
            size,
            last_used,
        }
    }

    #[test]
    fn select_stale_filters_by_age_and_sorts_by_size() {
        let now = SystemTime::now();
        let fresh = now - Duration::from_secs(5 * DAY_SECS);
        let stale = now - Duration::from_secs(45 * DAY_SECS);
        let entries = vec![
            entry("small-stale", 100, stale),
            entry("fresh", 999, fresh),
            entry("big-stale", 5000, stale),
        ];
        let kept = select_stale(entries, false, 30, now);
        // Only stale entries survive, largest first.
        let labels: Vec<_> = kept.iter().map(|e| e.label.as_str()).collect();
        assert_eq!(labels, vec!["big-stale", "small-stale"]);
    }

    #[test]
    fn nuke_keeps_everything_sorted_by_size() {
        let now = SystemTime::now();
        let fresh = now - Duration::from_secs(1 * DAY_SECS);
        let entries = vec![entry("a", 10, fresh), entry("b", 20, fresh)];
        let kept = select_stale(entries, true, 30, now);
        assert_eq!(kept.len(), 2);
        assert_eq!(kept[0].label, "b"); // larger first
    }

    #[test]
    fn unknown_last_used_is_eligible_future_is_kept() {
        let now = SystemTime::now();
        let unknown = entry("unknown", 1, SystemTime::UNIX_EPOCH);
        let future = entry("future", 1, now + Duration::from_secs(10 * DAY_SECS));
        let kept = select_stale(vec![unknown, future], false, 30, now);
        let labels: Vec<_> = kept.iter().map(|e| e.label.as_str()).collect();
        assert_eq!(labels, vec!["unknown"]);
    }

    #[test]
    fn weight_group_key_strips_all_companions() {
        assert_eq!(
            weight_group_key("abc123-model.safetensors.bin"),
            "abc123-model.safetensors"
        );
        assert_eq!(
            weight_group_key("abc123-model.safetensors.meta.json"),
            "abc123-model.safetensors"
        );
        assert_eq!(
            weight_group_key("abc123-model.safetensors.lock"),
            "abc123-model.safetensors"
        );
        assert_eq!(
            weight_group_key("abc123-model.safetensors.tmp.4242"),
            "abc123-model.safetensors"
        );
    }

    #[test]
    fn weight_label_drops_hash_prefix() {
        assert_eq!(
            weight_label("0123456789abcdef-model-00001-of-00002.safetensors"),
            "model-00001-of-00002.safetensors"
        );
        assert_eq!(weight_label("nohash"), "nohash");
    }

    #[test]
    fn scan_weight_cache_groups_sidecar_files() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        // Two sidecar groups; group A has bin+meta+lock, group B has bin only.
        std::fs::write(root.join("aaaa-m1.safetensors.bin"), vec![0u8; 1000]).unwrap();
        std::fs::write(root.join("aaaa-m1.safetensors.meta.json"), b"{}").unwrap();
        std::fs::write(root.join("aaaa-m1.safetensors.lock"), b"").unwrap();
        std::fs::write(root.join("bbbb-m2.safetensors.bin"), vec![0u8; 500]).unwrap();
        // A subdir must be ignored (only files are sidecars).
        std::fs::create_dir(root.join("subdir")).unwrap();

        // Empty resolver → unresolved sidecars labelled by filename.
        let mut entries = scan_weight_cache(root, &BTreeMap::new());
        entries.sort_by(|a, b| a.label.cmp(&b.label));
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].label, "m1.safetensors");
        assert_eq!(entries[0].paths.len(), 3); // bin + meta + lock
        assert_eq!(entries[0].size, 1002);
        assert_eq!(entries[1].label, "m2.safetensors");
        assert_eq!(entries[1].paths.len(), 1);
        assert_eq!(entries[1].size, 500);
    }

    #[test]
    fn scan_weight_cache_resolves_and_aggregates_shards_by_model() {
        let dir = tempfile::tempdir().unwrap();
        let hf_root = dir.path().join("hub");
        let weight_root = dir.path().join("weights");
        std::fs::create_dir_all(&weight_root).unwrap();

        // Fake HF model with a 2-shard snapshot, refs/main → rev.
        let model = hf_root.join("models--acme--Big-Model");
        let snap = model.join("snapshots").join("rev0");
        std::fs::create_dir_all(&snap).unwrap();
        std::fs::create_dir_all(model.join("refs")).unwrap();
        std::fs::write(model.join("refs").join("main"), "rev0").unwrap();
        for shard in [
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        ] {
            std::fs::write(snap.join(shard), b"weights").unwrap();
        }

        // Build the reverse map from the HF cache, then synthesize the two
        // sidecars the allocator would have written for those shards.
        let map = build_weight_key_to_model(&hf_root);
        assert_eq!(map.len(), 2, "both shards should map to the model");
        assert!(map.values().all(|m| m == "acme/Big-Model"));
        let mut total = 0u64;
        for (key, _model) in &map {
            std::fs::write(weight_root.join(format!("{key}.bin")), vec![0u8; 1000]).unwrap();
            std::fs::write(weight_root.join(format!("{key}.meta.json")), b"{}").unwrap();
            total += 1000 + 2;
        }
        // Plus one unrelated, unresolvable sidecar.
        std::fs::write(
            weight_root.join("ffff-orphan.safetensors.bin"),
            vec![0u8; 7],
        )
        .unwrap();

        let mut entries = scan_weight_cache(&weight_root, &map);
        entries.sort_by(|a, b| a.label.cmp(&b.label));
        assert_eq!(entries.len(), 2);
        // Both shards folded into ONE model row.
        assert_eq!(entries[0].label, "acme/Big-Model");
        assert_eq!(entries[0].size, total);
        assert_eq!(entries[0].paths.len(), 4); // 2 shards × (bin + meta)
        // The orphan stays standalone, labelled by filename.
        assert_eq!(entries[1].label, "orphan.safetensors");
        assert_eq!(entries[1].size, 7);
    }

    #[test]
    fn scan_hf_cache_counts_blobs_not_snapshot_symlinks() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        let model = root.join("models--meta-llama--Llama-3.2-1B");
        let blobs = model.join("blobs");
        let snap = model.join("snapshots").join("deadbeef");
        std::fs::create_dir_all(&blobs).unwrap();
        std::fs::create_dir_all(&snap).unwrap();
        let blob = blobs.join("abc123");
        std::fs::write(&blob, vec![0u8; 2048]).unwrap();
        // Snapshot symlink into the blob must not double-count.
        #[cfg(unix)]
        std::os::unix::fs::symlink(&blob, snap.join("model.safetensors")).unwrap();
        // A non-model dir in the cache root is ignored.
        std::fs::create_dir(root.join("datasets--foo--bar")).unwrap();

        let entries = scan_hf_cache(root);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].label, "meta-llama/Llama-3.2-1B");
        assert_eq!(entries[0].size, 2048); // blob counted once
        assert_eq!(entries[0].paths, vec![model]);
    }

    #[test]
    fn label_from_source_path_hf_and_local() {
        // HF cache snapshot path → org/name.
        assert_eq!(
            label_from_source_path(
                "/Users/me/.cache/huggingface/hub/models--acme--Big-Model/snapshots/rev0/model.safetensors"
            ),
            Some("acme/Big-Model".to_string())
        );
        // Local path → containing directory name.
        assert_eq!(
            label_from_source_path("/Users/me/models/my-llama-8b/model-00001-of-00004.safetensors"),
            Some("my-llama-8b".to_string())
        );
    }

    #[test]
    fn scan_weight_cache_resolves_local_model_via_meta_json() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        // Two shards of one local model, each with a meta.json recording a
        // src_path under the same local model directory.
        for (key, shard) in [
            (
                "aaaa-model-00001-of-00002.safetensors",
                "model-00001-of-00002.safetensors",
            ),
            (
                "bbbb-model-00002-of-00002.safetensors",
                "model-00002-of-00002.safetensors",
            ),
        ] {
            std::fs::write(root.join(format!("{key}.bin")), vec![0u8; 1000]).unwrap();
            let meta = format!("{{\"src_path\":\"/data/local-llama-8b/{shard}\"}}");
            std::fs::write(root.join(format!("{key}.meta.json")), meta).unwrap();
        }

        // Empty reverse-hash map: resolution comes purely from meta.json.
        let entries = scan_weight_cache(root, &BTreeMap::new());
        assert_eq!(entries.len(), 1, "both shards fold into one local model");
        assert_eq!(entries[0].label, "local-llama-8b");
        assert_eq!(entries[0].paths.len(), 4); // 2 × (bin + meta)
        // Size = both 1000-byte bins + both meta.json files.
        let meta_bytes: u64 = [
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        ]
        .iter()
        .map(|s| format!("{{\"src_path\":\"/data/local-llama-8b/{s}\"}}").len() as u64)
        .sum();
        assert_eq!(entries[0].size, 2000 + meta_bytes);
    }

    #[test]
    fn humanize_age_buckets() {
        let now = SystemTime::now();
        assert_eq!(humanize_age(now, SystemTime::UNIX_EPOCH), "unknown");
        assert_eq!(humanize_age(now, now), "today");
        assert_eq!(
            humanize_age(now, now - Duration::from_secs(DAY_SECS)),
            "1 day ago"
        );
        assert_eq!(
            humanize_age(now, now - Duration::from_secs(45 * DAY_SECS)),
            "45 days ago"
        );
    }

    #[test]
    fn sort_entries_by_size_then_name() {
        let now = SystemTime::now();
        let mk = |label, size| entry(label, size, now);
        let mut e = vec![mk("bbb", 100), mk("aaa", 300), mk("ccc", 200)];

        sort_entries(&mut e, &ListSort::Size);
        assert_eq!(
            e.iter().map(|x| x.label.as_str()).collect::<Vec<_>>(),
            vec!["aaa", "ccc", "bbb"]
        );

        sort_entries(&mut e, &ListSort::Name);
        assert_eq!(
            e.iter().map(|x| x.label.as_str()).collect::<Vec<_>>(),
            vec!["aaa", "bbb", "ccc"]
        );
    }

    #[test]
    fn truncate_clips_long_labels() {
        assert_eq!(truncate("short", 44), "short");
        let long = "a".repeat(60);
        let out = truncate(&long, 44);
        assert_eq!(out.chars().count(), 44);
        assert!(out.ends_with('…'));
    }
}
