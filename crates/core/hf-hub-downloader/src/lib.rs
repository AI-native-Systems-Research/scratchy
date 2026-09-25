// SPDX-License-Identifier: Apache-2.0
//! Downloading model files from the HuggingFace Hub into the standard Hub
//! cache.
//!
//! Scope is deliberately the eight calls scratchy makes: build a client,
//! address a repo, fetch a file (with or without progress), list the repo's
//! files, and look the cache up without touching the network. Uploads,
//! commits, spaces, buckets and users are out of scope, and that is the
//! whole point — [`hf-hub`] 1.0 needs nineteen mandatory dependencies to
//! serve all of those, including a content-addressed-storage client and a C
//! cryptography toolchain, none of which a model download needs.
//!
//! What the Hub actually promises, established by probing it rather than
//! from documentation:
//!
//! * `GET /<repo>/resolve/<rev>/<file>` answers **302** for LFS-tracked
//!   content, with `x-linked-etag` (SHA-256), `x-linked-size`, and an
//!   absolute off-host CDN `location`.
//! * The same endpoint answers **307** for git-tracked content, with
//!   `x-linked-etag` (a git blob SHA-1) and a *relative* `location`.
//! * Both carry `x-repo-commit`.
//!
//! That contract is held stable by the entire ecosystem — Python
//! `huggingface_hub`, `transformers`, llama.cpp, ollama and every `curl` in
//! every tutorial resolve through it. Xet is the proof: the Hub shipped a
//! whole new transfer protocol and still serves `x-linked-*` on the same
//! response, so clients that ignore `x-xet-hash` keep working.
//!
//! [`hf-hub`]: https://crates.io/crates/hf-hub

pub mod cache;
mod error;
mod fetch;
mod limit;
#[cfg(test)]
mod local_hub;

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

pub use error::{Error, Result};

/// Default revision when none is given.
pub const MAIN: &str = "main";

/// Receives download progress. Implementors drive whatever UI they like;
/// this crate deliberately has no opinion and no `indicatif` dependency.
pub trait Progress: Send {
    /// Called once before any bytes move. `total` is 0 when the size is not
    /// known ahead of time, which is the case for small git-tracked files.
    fn init(&mut self, total: u64, filename: &str);
    /// Called at most once, right after [`init`](Self::init) and before any
    /// [`update`](Self::update), with the bytes a previous, interrupted run
    /// already placed.
    ///
    /// Counted like an update by default. A UI that shows a transfer rate
    /// overrides it, or gigabytes found on disk read as gigabytes a second.
    fn resumed(&mut self, bytes: u64) {
        self.update(bytes);
    }
    /// Called with a byte delta, not a running total.
    fn update(&mut self, delta: u64);
    /// Called once after the file is verified and placed.
    fn finish(&mut self);

    /// Polled while bytes are moving; returning `true` stops the download
    /// with [`Error::Cancelled`].
    ///
    /// Partial state is deliberately left on disk, so the next attempt
    /// resumes rather than restarting — which is what makes cancelling a
    /// multi-gigabyte download a reasonable thing for a UI to offer.
    fn cancelled(&self) -> bool {
        false
    }
}

/// No-op sink, for callers that do not want progress.
impl Progress for () {
    fn init(&mut self, _total: u64, _filename: &str) {}
    fn update(&mut self, _delta: u64) {}
    fn finish(&mut self) {}
}

/// Builds a [`Client`].
pub struct ClientBuilder {
    endpoint: String,
    token: Option<String>,
    root: PathBuf,
    chunks: u64,
    parallel_threshold: u64,
    max_concurrency: usize,
    retries: usize,
}

impl ClientBuilder {
    /// Reads `HF_ENDPOINT`, `HF_TOKEN`, and the cache-location variables,
    /// matching Python `huggingface_hub`.
    pub fn from_env() -> Self {
        Self {
            endpoint: std::env::var("HF_ENDPOINT")
                .unwrap_or_else(|_| "https://huggingface.co".to_string()),
            token: std::env::var("HF_TOKEN").ok().filter(|t| !t.is_empty()),
            root: cache::default_root(),
            chunks: 8,
            parallel_threshold: 64 * 1024 * 1024,
            max_concurrency: 8,
            retries: 5,
        }
    }

    /// How many ranges of one large file may be in flight at once.
    ///
    /// Composes with whatever concurrency the caller runs *across* files, so
    /// the two multiply. That product is bounded by
    /// [`max_concurrency`](Self::max_concurrency) rather than by this, and
    /// those permits are shared round-robin between the files asking, which
    /// means a large value here is safe: it widens the single-file case
    /// without letting a sharded download open a connection per chunk per
    /// shard, or letting one shard's ranges crowd out the others'.
    pub fn chunks(mut self, chunks: u64) -> Self {
        self.chunks = chunks.max(1);
        self
    }

    /// Hard ceiling on in-flight requests from this client, across every
    /// file and every chunk.
    ///
    /// This is the knob that actually bounds load on the Hub. Eight shards
    /// at eight ranges each is sixty-four connections to one host; the Hub
    /// throttles that, and it loses to fewer, fatter streams regardless.
    /// Files downloading at once split it evenly: eight shards get one
    /// connection each, a single file gets all of them.
    pub fn max_concurrency(mut self, permits: usize) -> Self {
        self.max_concurrency = permits.max(1);
        self
    }

    /// Attempts per request before giving up, including the first.
    ///
    /// Retries are cheap because a range resumes from its recorded offset
    /// rather than from zero, so a reset costs the bytes since the last
    /// checkpoint, not the whole file.
    pub fn retries(mut self, attempts: usize) -> Self {
        self.retries = attempts.max(1);
        self
    }

    /// Minimum size before a file is split at all. Below it the extra
    /// requests and the pre-sized sparse file cost more than they save.
    pub fn parallel_threshold(mut self, bytes: u64) -> Self {
        self.parallel_threshold = bytes;
        self
    }

    pub fn token(mut self, token: Option<String>) -> Self {
        if let Some(token) = token {
            self.token = Some(token);
        }
        self
    }

    pub fn endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = endpoint.into();
        self
    }

    pub fn cache_root(mut self, root: impl Into<PathBuf>) -> Self {
        self.root = root.into();
        self
    }

    pub fn build(self) -> Client {
        // Two agents on purpose. The probe must NOT follow redirects, because
        // every header worth having is on the first hop and following would
        // discard them. Byte fetches must follow, because the git-tracked
        // path redirects within huggingface.co.
        //
        // Both keep one idle connection per permit. A large file is fetched
        // as many short pieces from one CDN host, and ureq's default of three
        // idle connections per host would close most of them between pieces
        // and pay a fresh TLS handshake for the next.
        let pooled = self.max_concurrency;
        let probing: ureq::Agent = ureq::Agent::config_builder()
            .max_redirects(0)
            .max_idle_connections(pooled)
            .max_idle_connections_per_host(pooled)
            .build()
            .into();
        let following: ureq::Agent = ureq::Agent::config_builder()
            .max_redirects(8)
            .max_idle_connections(pooled)
            .max_idle_connections_per_host(pooled)
            .build()
            .into();
        Client {
            probing,
            following,
            endpoint: self.endpoint.trim_end_matches('/').to_string(),
            token: self.token,
            root: self.root,
            chunks: self.chunks,
            parallel_threshold: self.parallel_threshold,
            permits: limit::Semaphore::new(self.max_concurrency),
            flows: AtomicU64::new(0),
            retries: self.retries,
        }
    }
}

/// A Hub client.
pub struct Client {
    probing: ureq::Agent,
    following: ureq::Agent,
    endpoint: String,
    token: Option<String>,
    root: PathBuf,
    chunks: u64,
    parallel_threshold: u64,
    permits: limit::Semaphore,
    /// Numbers each download's [`limit::Flow`].
    flows: AtomicU64,
    retries: usize,
}

impl Client {
    /// Convenience for [`ClientBuilder::from_env`] + [`ClientBuilder::build`].
    pub fn from_env() -> Self {
        ClientBuilder::from_env().build()
    }

    pub fn builder() -> ClientBuilder {
        ClientBuilder::from_env()
    }

    /// Address a model repo at the default revision.
    pub fn model(&self, repo_id: impl Into<String>) -> Repo<'_> {
        Repo {
            client: self,
            repo_id: repo_id.into(),
            revision: MAIN.to_string(),
        }
    }

    /// The cache root this client reads and writes.
    pub fn cache_root(&self) -> &Path {
        &self.root
    }
}

/// A model repo at a revision.
pub struct Repo<'a> {
    client: &'a Client,
    repo_id: String,
    revision: String,
}

impl Repo<'_> {
    /// Pin to a specific revision (branch, tag, or commit).
    pub fn revision(mut self, revision: impl Into<String>) -> Self {
        self.revision = revision.into();
        self
    }

    /// Cache-first fetch with no progress reporting.
    pub fn get(&self, filename: &str) -> Result<PathBuf> {
        self.download(filename, &mut ())
    }

    /// Path to `filename` if it is already cached, without any network I/O.
    pub fn cached(&self, filename: &str) -> Option<PathBuf> {
        cache::cached_path(&self.client.root, &self.repo_id, &self.revision, filename)
    }

    /// Every file in the repo, for callers that must choose among them (the
    /// GGUF quantisation picker).
    pub fn files(&self) -> Result<Vec<String>> {
        #[derive(serde::Deserialize)]
        struct Sibling {
            rfilename: String,
        }
        #[derive(serde::Deserialize)]
        struct Info {
            siblings: Vec<Sibling>,
        }

        let url = format!(
            "{}/api/models/{}/revision/{}",
            self.client.endpoint, self.repo_id, self.revision
        );
        let mut req = self.client.following.get(&url);
        if let Some(token) = &self.client.token {
            req = req.header("Authorization", format!("Bearer {token}"));
        }
        let res = req.call().map_err(|e| Error::Transport {
            url: url.clone(),
            attempts: 1,
            source: Box::new(e),
        })?;
        let body = res
            .into_body()
            .read_to_string()
            .map_err(|e| Error::io(&url, std::io::Error::other(e)))?;
        let info: Info = serde_json::from_str(&body).map_err(|e| Error::Json {
            what: url,
            source: e,
        })?;
        Ok(info.siblings.into_iter().map(|s| s.rfilename).collect())
    }

    /// Cache-first fetch, reporting progress.
    ///
    /// Order matters and is the same order Python uses: resolve the cache
    /// without touching the network, then probe, then check whether the blob
    /// is already present under another revision, and only then move bytes.
    pub fn download(&self, filename: &str, progress: &mut dyn Progress) -> Result<PathBuf> {
        if let Some(hit) = self.cached(filename) {
            return Ok(hit);
        }

        let opts = fetch::Opts {
            ranged: &self.client.probing,
            following: &self.client.following,
            token: self.client.token.as_deref(),
            chunks: self.client.chunks,
            parallel_threshold: self.client.parallel_threshold,
            retries: self.client.retries,
            // One budget per file, spanning its probe and all its ranges.
            budget: fetch::RetryBudget::new(self.client.retries),
            permits: &self.client.permits,
            flow: limit::Flow(self.client.flows.fetch_add(1, Ordering::Relaxed)),
        };

        let meta = fetch::probe(
            &opts,
            &self.client.endpoint,
            &self.repo_id,
            &self.revision,
            filename,
        )?;

        let dir = cache::repo_dir(&self.client.root, &self.repo_id);
        let blob = cache::blob_path(&dir, &meta.etag);

        // Content already on disk under a different revision: link, do not
        // re-fetch. Blobs are named by digest precisely so this works.
        if blob.exists() {
            return cache::link_into_snapshot(
                &dir,
                &meta.commit,
                filename,
                &meta.etag,
                &self.revision,
            );
        }

        let blobs = dir.join("blobs");
        std::fs::create_dir_all(&blobs).map_err(|e| Error::io(&blobs, e))?;
        let partial = blobs.join(format!("{}.incomplete", meta.etag));

        // Pinned to the commit the first probe found, so a lapsed CDN URL is
        // re-signed for these bytes and never swapped for a newer revision's.
        let resolve = || {
            fetch::probe(
                &opts,
                &self.client.endpoint,
                &self.repo_id,
                &meta.commit,
                filename,
            )
        };
        fetch::fetch(&opts, &meta, filename, &partial, progress, &resolve)?;

        if let Err(e) = fetch::verify(&partial, &meta, filename) {
            // Never leave content that failed verification where a later run
            // could mistake it for complete.
            let _ = std::fs::remove_file(&partial);
            return Err(e);
        }

        std::fs::rename(&partial, &blob).map_err(|e| Error::io(&blob, e))?;
        let path =
            cache::link_into_snapshot(&dir, &meta.commit, filename, &meta.etag, &self.revision)?;
        progress.finish();
        Ok(path)
    }
}
