// SPDX-License-Identifier: Apache-2.0
//! Fetching a model from the HuggingFace Hub into the local cache.
//!
//! [`download`] is the one routine for it — `scr model pull` and the engine's
//! `resolve_model_path` both call it — so which files a model needs, the
//! order they are fetched in, and how their progress is drawn are decided
//! once.

mod bar;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use hf_hub_downloader::{self as hfdl, Progress};
use tracing::{info, warn};

use crate::error::ModelError;
use crate::weight::{DiskSpaceError, SafeTensorsIndex};

use bar::{Bar, Bars};

/// Attempts per file before giving up, including the first.
///
/// The budget is per FILE, shared across every range that file is split
/// into, so this is five for the whole download rather than five each.
/// Retries are cheap: a range restarts from the offset its resume log
/// recorded, so a reset costs the bytes since the last checkpoint and never
/// re-fetches completed ones.
const RETRIES: usize = 5;

/// Shards downloading at once.
///
/// These MULTIPLY with the ranges each download splits itself into. The
/// client's own permit cap (`max_concurrency`, 8) bounds the product and
/// shares it round-robin between the shards asking, so eight shards move
/// together at a connection each rather than one at a time at eight — while
/// a single-file model spends the whole budget on ranges within that file.
const SHARDS_AT_ONCE: usize = 8;

const CONFIG: &str = "config.json";
const SINGLE_FILE: &str = "model.safetensors";
const SHARD_INDEX: &str = "model.safetensors.index.json";
/// Gemma-4 (HF transformers#45205) ships its chat template as a standalone
/// sibling file rather than inside `tokenizer_config.json`; most repos have
/// none.
const CHAT_TEMPLATE: &str = "chat_template.jinja";

/// Quantisations picked, in this order, when a GGUF repo is fetched without
/// saying which of its files to take.
const DEFAULT_GGUF: [&str; 5] = ["Q4_K_M", "Q4_K_S", "Q4_K", "Q4_0", "Q8_0"];

/// Which `.gguf`, if any, is fetched in place of safetensors weights.
#[derive(Debug, Clone, Copy)]
pub enum Gguf<'a> {
    /// This file of the repo, by name.
    File(&'a str),
    /// The repo's `.gguf` whose name contains this quantisation
    /// (case-insensitive), or the default pick if none does.
    Matching(&'a str),
    /// The default pick, but only from a repo whose id says it is GGUF; any
    /// other repo is fetched as safetensors.
    Auto,
}

/// Sink for byte-level model-download progress, so a caller that owns the
/// terminal (the ratatui TUI) can render its own progress instead of letting
/// the default `indicatif` bars draw to stderr over its alt-screen.
///
/// Methods are called from the parallel download threads (up to
/// [`SHARDS_AT_ONCE`] files at a time), keyed by filename — implementors must
/// be internally synchronized. The `Debug` supertrait lets an
/// `Option<Arc<dyn DownloadObserver>>` sit inside the `#[derive(Debug)]`
/// `VllmConfig`.
pub trait DownloadObserver: Send + Sync + std::fmt::Debug {
    /// A shard's download started; `total_bytes` is its full size.
    fn on_start(&self, shard: &str, total_bytes: usize);
    /// `delta_bytes` more bytes of `shard` have landed on disk.
    fn on_advance(&self, shard: &str, delta_bytes: usize);
    /// `shard` finished downloading.
    fn on_finish(&self, shard: &str);
}

/// Why a model could not be fetched.
#[derive(Debug, thiserror::Error)]
pub enum DownloadError {
    #[error("failed to download {file} for {repo}: {source}")]
    Fetch {
        repo: String,
        file: String,
        #[source]
        source: Box<hfdl::Error>,
    },

    #[error("failed to parse the shard index of {repo}: {source}")]
    Index {
        repo: String,
        #[source]
        source: ModelError,
    },

    #[error("cannot download {repo}: {source}")]
    DiskSpace {
        repo: String,
        #[source]
        source: DiskSpaceError,
    },

    /// The Hub reported neither weights layout present.
    #[error(
        "no safetensors weights found for {repo} \
         (neither model.safetensors nor model.safetensors.index.json)"
    )]
    NoWeights { repo: String },
}

/// Fetch `repo_id` into the Hub cache, and return where it landed: the
/// snapshot directory for safetensors weights, or the `.gguf` itself.
///
/// Cache-first throughout. A model whose every weight file is already on
/// disk is answered without touching the network, and an interrupted one
/// resumes. Weights report their progress — to `observer` when a caller owns
/// the terminal, as stderr bars otherwise; config and tokenizer files are
/// too small to report at all.
pub fn download(
    repo_id: &str,
    token: Option<&str>,
    gguf: Gguf<'_>,
    observer: Option<Arc<dyn DownloadObserver>>,
) -> Result<PathBuf, DownloadError> {
    let client = hfdl::Client::builder()
        .retries(RETRIES)
        .token(token.map(str::to_string))
        .build();
    let repo = client.model(repo_id);
    let display = Display::new(observer);

    if let Gguf::Auto = gguf
        && let Some(dir) = fully_cached(&repo)
    {
        // Backfills the template for caches pulled before it was part of the
        // download set: one network fetch, and only while it is absent.
        if !dir.join(CHAT_TEMPLATE).exists() {
            let _ = repo.get(CHAT_TEMPLATE);
        }
        return Ok(dir);
    }

    info!("Downloading model from HuggingFace Hub: {repo_id}");

    if let Some(file) = gguf_to_fetch(&repo, repo_id, gguf) {
        info!("Downloading GGUF file: {file}");
        let path = repo
            .download(&file, &mut display.sink(&file))
            .map_err(failed(repo_id, &file))?;
        fetch_tokenizer(&repo);
        return Ok(path);
    }

    let config = repo.get(CONFIG).map_err(failed(repo_id, CONFIG))?;
    let dir = config
        .parent()
        .expect("a snapshot file sits in its snapshot directory")
        .to_path_buf();
    fetch_tokenizer(&repo);

    // "This repo has no single-file weights" and "the single-file download
    // failed" are different facts, and only the first means we should go
    // looking for shards. Branching on the typed `NotFound` keeps a network
    // error, a full disk or a bad digest from being reported as "no
    // safetensors weights found" — the one cause they are not.
    match repo.download(SINGLE_FILE, &mut display.sink(SINGLE_FILE)) {
        Ok(_) => return Ok(dir),
        Err(e) if e.is_not_found() => {}
        Err(e) => return Err(failed(repo_id, SINGLE_FILE)(e)),
    }
    let index = match repo.get(SHARD_INDEX) {
        Ok(index) => index,
        Err(e) if e.is_not_found() => {
            return Err(DownloadError::NoWeights {
                repo: repo_id.to_string(),
            });
        }
        Err(e) => return Err(failed(repo_id, SHARD_INDEX)(e)),
    };
    download_shards(&repo, repo_id, &index, &dir, &display)?;
    Ok(dir)
}

fn failed<'a>(repo: &'a str, file: &'a str) -> impl FnOnce(hfdl::Error) -> DownloadError + 'a {
    move |source| DownloadError::Fetch {
        repo: repo.to_string(),
        file: file.to_string(),
        source: Box::new(source),
    }
}

/// The snapshot directory, when the model is already wholly on disk — found
/// without any network I/O, which would otherwise spend ~100ms on a TLS
/// handshake and a probe for a model already here.
fn fully_cached(repo: &hfdl::Repo<'_>) -> Option<PathBuf> {
    let dir = repo.cached(CONFIG)?.parent()?.to_path_buf();
    // `config.json` being present is NOT proof the model is: an interrupted
    // pull commonly leaves config and index on disk but only some shards.
    // Handed that directory, the weight loader dies deep in `mmap` with a
    // bare "No such file or directory" — so check, and resume if short.
    if !weights_present(&dir) {
        info!(
            "Cached model {} is incomplete (missing weight shards); resuming download",
            dir.display()
        );
        return None;
    }
    info!(
        "Using cached model: {} (HF cache, no network)",
        dir.display()
    );
    Some(dir)
}

/// Returns true iff every safetensors weight file this model references
/// is present on disk in `dir`.
///
/// Safetensors-only by design — and that is the safe default for every
/// other format:
///   * The partial-cache crash is *structural* to multi-file safetensors
///     (config.json + index + N shards, where config can exist while some
///     shards are still missing). It is the only weights layout
///     `GpuWeights::from_dir` loads from a directory.
///   * GGUF is a single file that downloads to `.incomplete` and renames
///     atomically — it is wholly present or absent, never partial — and is
///     resolved as a file path, not via this dir fast path. Anything that
///     is not complete safetensors returns false here and falls through to
///     the cache-first, retrying download path. The only unsafe direction
///     would be a false *positive* (claiming complete when it is not), which
///     this never does for any format.
fn weights_present(dir: &Path) -> bool {
    if dir.join(SINGLE_FILE).exists() {
        return true;
    }
    match SafeTensorsIndex::from_file(dir.join(SHARD_INDEX)) {
        Ok(index) => index
            .shard_files()
            .iter()
            .all(|shard| dir.join(shard).exists()),
        Err(_) => false,
    }
}

fn gguf_to_fetch(repo: &hfdl::Repo<'_>, repo_id: &str, gguf: Gguf<'_>) -> Option<String> {
    let wanted = match gguf {
        Gguf::File(file) => return Some(file.to_string()),
        Gguf::Matching(quant) => Some(quant),
        Gguf::Auto if repo_id.to_ascii_uppercase().contains("GGUF") => None,
        Gguf::Auto => return None,
    };
    let files = repo.files().ok()?;
    choose_gguf(files, wanted)
}

/// The `.gguf` among `files` to fetch: the one matching `wanted` if any
/// does, else the first of [`DEFAULT_GGUF`] present, else the first by name.
fn choose_gguf(files: Vec<String>, wanted: Option<&str>) -> Option<String> {
    let mut ggufs: Vec<String> = files.into_iter().filter(|f| f.ends_with(".gguf")).collect();
    if let Some(quant) = wanted {
        let upper = quant.to_ascii_uppercase();
        if let Some(file) = ggufs
            .iter()
            .find(|f| f.to_ascii_uppercase().contains(&upper))
        {
            return Some(file.clone());
        }
        if !ggufs.is_empty() {
            warn!("no GGUF file matching '{quant}' found, using default selection");
        }
    }
    for pattern in DEFAULT_GGUF {
        if let Some(file) = ggufs.iter().find(|f| f.contains(pattern)) {
            return Some(file.clone());
        }
    }
    ggufs.sort();
    (!ggufs.is_empty()).then(|| ggufs.swap_remove(0))
}

/// Best effort: a model still loads without them.
fn fetch_tokenizer(repo: &hfdl::Repo<'_>) {
    let _ = repo.get("tokenizer.json");
    let _ = repo.get("tokenizer_config.json");
    let _ = repo.get(CHAT_TEMPLATE);
}

fn download_shards(
    repo: &hfdl::Repo<'_>,
    repo_id: &str,
    index: &Path,
    dir: &Path,
    display: &Display,
) -> Result<(), DownloadError> {
    let index = SafeTensorsIndex::from_file(index).map_err(|source| DownloadError::Index {
        repo: repo_id.to_string(),
        source,
    })?;
    let shards = index.shard_files();
    let needed: Vec<&String> = shards
        .iter()
        .filter(|shard| !dir.join(shard).exists())
        .collect();

    // Fail fast on a full disk rather than starting a doomed multi-GB
    // download (and burning the downloader's retry budget on ENOSPC, which
    // never recovers). A disk-full download is what leaves the partial cache
    // `fully_cached` detects and resumes.
    index
        .ensure_disk_space(dir)
        .map_err(|source| DownloadError::DiskSpace {
            repo: repo_id.to_string(),
            source,
        })?;

    if needed.is_empty() {
        info!("All {} shard files already cached", shards.len());
        return Ok(());
    }
    info!(
        "Downloading {} of {} shard files (up to {SHARDS_AT_ONCE} in parallel)",
        needed.len(),
        shards.len()
    );

    // A window, not batches: the next shard starts as soon as any one
    // finishes, rather than when the slowest of the last eight does.
    let next = AtomicUsize::new(0);
    let stop = AtomicBool::new(false);
    let worker = || {
        while !stop.load(Ordering::Relaxed) {
            let Some(shard) = needed.get(next.fetch_add(1, Ordering::Relaxed)) else {
                break;
            };
            if let Err(e) = repo.download(shard, &mut display.sink(shard)) {
                stop.store(true, Ordering::Relaxed);
                return Err(failed(repo_id, shard)(e));
            }
        }
        Ok(())
    };
    std::thread::scope(|s| {
        let running: Vec<_> = (0..SHARDS_AT_ONCE.min(needed.len()))
            .map(|_| s.spawn(worker))
            .collect();
        running.into_iter().try_for_each(|worker| {
            worker
                .join()
                .unwrap_or_else(|panic| std::panic::resume_unwind(panic))
        })
    })
}

/// Where download progress is drawn.
enum Display {
    /// A caller that owns the terminal — the TUI — draws it from here, and
    /// nothing is written to stderr under its alt-screen.
    Observer(Arc<dyn DownloadObserver>),
    Bars(Bars),
}

impl Display {
    fn new(observer: Option<Arc<dyn DownloadObserver>>) -> Self {
        observer.map_or_else(|| Display::Bars(Bars::default()), Display::Observer)
    }

    fn sink(&self, file: &str) -> Sink {
        match self {
            Display::Observer(observer) => Sink::Observed {
                observer: observer.clone(),
                file: file.to_string(),
            },
            Display::Bars(bars) => Sink::Bar(bars.bar()),
        }
    }
}

/// One file's progress, wherever the [`Display`] sends it.
enum Sink {
    Bar(Bar),
    Observed {
        observer: Arc<dyn DownloadObserver>,
        file: String,
    },
}

impl Progress for Sink {
    fn init(&mut self, total: u64, filename: &str) {
        match self {
            Sink::Bar(bar) => bar.init(total, filename),
            Sink::Observed { observer, file } => observer.on_start(file, total as usize),
        }
    }

    fn resumed(&mut self, bytes: u64) {
        match self {
            Sink::Bar(bar) => bar.resumed(bytes),
            Sink::Observed { observer, file } => observer.on_advance(file, bytes as usize),
        }
    }

    fn update(&mut self, delta: u64) {
        match self {
            Sink::Bar(bar) => bar.update(delta),
            Sink::Observed { observer, file } => observer.on_advance(file, delta as usize),
        }
    }

    fn finish(&mut self) {
        match self {
            Sink::Bar(bar) => bar.finish(),
            Sink::Observed { observer, file } => observer.on_finish(file),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn files(names: &[&str]) -> Vec<String> {
        names.iter().map(|name| name.to_string()).collect()
    }

    /// `scr model pull --quantization` matched case-insensitively and the
    /// engine could not ask at all; both now come through here.
    #[test]
    fn a_named_quantisation_wins_over_the_default_pick() {
        let repo = files(&["m-Q4_K_M.gguf", "m-q8_0.gguf", "README.md"]);
        assert_eq!(
            choose_gguf(repo, Some("Q8_0")).as_deref(),
            Some("m-q8_0.gguf")
        );
    }

    #[test]
    fn an_unmatched_quantisation_falls_back_to_the_default_pick() {
        let repo = files(&["m-Q8_0.gguf", "m-Q4_K_S.gguf"]);
        assert_eq!(
            choose_gguf(repo, Some("IQ2_XS")).as_deref(),
            Some("m-Q4_K_S.gguf")
        );
    }

    #[test]
    fn the_default_pick_prefers_q4_k_m_then_the_first_by_name() {
        let preferred = files(&["m-Q8_0.gguf", "m-Q4_K_M.gguf"]);
        assert_eq!(
            choose_gguf(preferred, None).as_deref(),
            Some("m-Q4_K_M.gguf")
        );
        let unranked = files(&["m-F16.gguf", "m-BF16.gguf"]);
        assert_eq!(choose_gguf(unranked, None).as_deref(), Some("m-BF16.gguf"));
        assert_eq!(choose_gguf(files(&["config.json"]), Some("Q4_K_M")), None);
    }

    /// The fast path must never hand the loader a directory an interrupted
    /// pull left short of shards.
    #[test]
    fn a_cache_missing_a_shard_is_not_complete() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path();
        assert!(!weights_present(path), "no weights at all");

        std::fs::write(
            path.join(SHARD_INDEX),
            r#"{"weight_map": {"a": "model-00001-of-00002.safetensors",
                               "b": "model-00002-of-00002.safetensors"}}"#,
        )
        .unwrap();
        std::fs::write(path.join("model-00001-of-00002.safetensors"), b"").unwrap();
        assert!(!weights_present(path), "one of two shards");

        std::fs::write(path.join("model-00002-of-00002.safetensors"), b"").unwrap();
        assert!(weights_present(path), "both shards");
    }

    #[test]
    fn a_single_file_model_is_complete_on_its_own() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join(SINGLE_FILE), b"").unwrap();
        assert!(weights_present(dir.path()));
    }
}
