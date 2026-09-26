// SPDX-License-Identifier: Apache-2.0
//! Metadata probe and the byte-moving half of the downloader.

use std::fs::{File, OpenOptions};
use std::io::Read;
use std::path::Path;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

// `sha1` and `sha2` both re-export the same `digest::Digest`, so one import
// covers `Sha1` and `Sha256` alike.
use sha2::Digest as _;

use crate::Progress;
use crate::error::{Error, Result};
use crate::limit::{Flow, Semaphore};

/// Read buffer, and the granularity at which progress is reported.
const BUF: usize = 256 * 1024;

/// The most one ranged request asks for.
///
/// A range that spans a whole eighth of the file holds its permit until that
/// eighth has landed, so whichever file claimed the permits first keeps them
/// and the rest wait. Short pieces hand the permit back often enough for the
/// round-robin in [`Semaphore`] to share it: that is what lets eight shards
/// move together. Large enough that the extra request per piece — on a
/// kept-alive connection, one round trip — is a rounding error.
const PIECE: u64 = 32 * 1024 * 1024;

/// How often a piece records its progress for resume. Small enough that a
/// kill costs little, large enough that the sidecar write is noise.
const RESUME_STRIDE: u64 = 8 * 1024 * 1024;

/// Backoff base; retry N waits roughly `BACKOFF * 2^N`, staggered.
const BACKOFF: Duration = Duration::from_millis(500);

/// Retries shared by every request for one file.
///
/// Budgeting per request instead would multiply: a file split into eight
/// ranges, each allowed five retries, can spend forty. `retries(5)` should
/// mean five for the file, which is what a caller asking for five expects
/// and what hf-hub's own setting means.
pub(crate) struct RetryBudget {
    remaining: std::sync::atomic::AtomicUsize,
}

impl RetryBudget {
    pub(crate) fn new(retries: usize) -> Self {
        Self {
            // The first attempt is not a retry, so N attempts is N-1 retries.
            remaining: std::sync::atomic::AtomicUsize::new(retries.saturating_sub(1)),
        }
    }

    /// Claim one retry, or `None` when the file's budget is spent.
    fn claim(&self) -> Option<usize> {
        self.remaining
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |left| {
                (left > 0).then(|| left - 1)
            })
            .ok()
    }
}

/// Backoff for retry `attempt` on chunk `slot` of `slots`.
///
/// Half fixed, half staggered by slot. Pure exponential would have all eight
/// ranges fail on one blip and retry in lockstep — a thundering herd aimed at
/// the Hub. Staggering by slot decorrelates them deterministically, which
/// beats random jitter here: it guarantees the spread rather than hoping for
/// it, and it keeps the test suite reproducible.
fn backoff(attempt: usize, slot: usize, slots: usize) -> Duration {
    let window = BACKOFF * (1u32 << attempt.min(5));
    let slots = slots.max(1) as u32;
    window / 2 + (window / 2 / slots) * (slot as u32 % slots)
}

/// Everything the byte-moving path needs that is not per-file.
pub(crate) struct Opts<'a> {
    pub ranged: &'a ureq::Agent,
    pub following: &'a ureq::Agent,
    pub token: Option<&'a str>,
    pub chunks: u64,
    pub parallel_threshold: u64,
    /// Configured attempt count, carried only for error messages.
    pub retries: usize,
    /// Retries left for this file, shared across its probe and every range.
    pub budget: RetryBudget,
    pub permits: &'a Semaphore,
    /// This file's place in the round-robin for [`permits`](Self::permits).
    pub flow: Flow,
}

/// What the Hub says about one file, gathered from a single redirect-less
/// probe of `/resolve/`.
#[derive(Debug, Clone)]
pub(crate) struct FileMeta {
    /// Where the bytes actually live.
    pub url: String,
    /// The Hub's digest. Doubles as the blob's filename, which is what makes
    /// this cache readable by other Hub clients.
    pub etag: String,
    /// Commit the revision resolved to.
    pub commit: String,
    /// Known only for LFS files, whose `x-linked-size` the 302 carries.
    /// `None` means "stream it and find out" — true of the small JSON files.
    pub size: Option<u64>,
    /// LFS content is digested with SHA-256; git-tracked content with a git
    /// blob SHA-1. Getting this backwards makes every verification fail.
    pub lfs: bool,
    /// Whether our Hub token may be sent to [`url`](Self::url). False once we
    /// have been redirected off huggingface.co: the LFS CDN URL is presigned
    /// and needs no credential, and forwarding a bearer token to a third
    /// party host leaks it.
    pub send_token: bool,
}

/// Run `attempt` until it succeeds, it fails in a way retrying cannot fix, or
/// the file's shared budget runs out.
///
/// `slot`/`slots` only stagger the backoff; they do not partition the budget.
fn with_retry<T>(
    budget: &RetryBudget,
    slot: usize,
    slots: usize,
    mut attempt: impl FnMut() -> Result<T>,
) -> Result<T> {
    let mut round = 0usize;
    loop {
        match attempt() {
            Ok(value) => return Ok(value),
            // A 404, a cancellation or a hash mismatch will not improve.
            Err(e) if !e.is_transient() => return Err(e),
            Err(e) => match budget.claim() {
                Some(_) => {
                    std::thread::sleep(backoff(round, slot, slots));
                    round += 1;
                }
                None => return Err(e),
            },
        }
    }
}

/// Probe `/resolve/` without following redirects.
///
/// Everything needed comes off the first hop, which is what keeps this
/// simple: the LFS 302 carries `x-linked-etag`/`x-linked-size`, and the
/// plain-file 307 carries `x-linked-etag` too. Following the 307 would mean
/// resolving its **relative** `Location` by hand — a step whose omission is
/// an open bug upstream (huggingface/hf-hub#163).
pub(crate) fn probe(
    opts: &Opts<'_>,
    endpoint: &str,
    repo_id: &str,
    revision: &str,
    filename: &str,
) -> Result<FileMeta> {
    let url = format!("{endpoint}/{repo_id}/resolve/{revision}/{filename}");
    with_retry(&opts.budget, 0, 1, || {
        probe_once(opts, &url, repo_id, filename)
    })
}

/// Classify a ureq failure.
///
/// ureq reports a non-2xx as `Error::StatusCode` rather than handing back the
/// response, so mapping every failure to `Transport` gets two things wrong at
/// once. `Transport` is retryable, so a 404 sleeps through the full backoff
/// ladder; and the error that finally surfaces is `Transport`, not
/// `NotFound`, so callers probing for an optional file — `chat_template.jinja`
/// on the cached path, `model.safetensors` before falling back to a shard
/// index — see a failure where they should see an absence.
fn classify(e: ureq::Error, url: &str, repo_id: &str, filename: &str, attempts: usize) -> Error {
    match e {
        ureq::Error::StatusCode(404) => Error::NotFound {
            repo: repo_id.to_string(),
            filename: filename.to_string(),
        },
        ureq::Error::StatusCode(status) => Error::UnexpectedStatus {
            url: url.to_string(),
            status,
        },
        other => Error::Transport {
            url: url.to_string(),
            attempts,
            source: Box::new(other),
        },
    }
}

/// Read one buffer of a response body.
///
/// A read that fails is the connection failing mid-transfer — a reset, a
/// stall, a server that closed early — so it is transport, and retried from
/// the last recorded offset. Only the write side is the disk. Reporting both
/// as [`Error::Io`], which is not transient, made every reset fatal to the
/// whole file.
fn read_body(reader: &mut impl Read, buf: &mut [u8], url: &str, attempts: usize) -> Result<usize> {
    reader.read(buf).map_err(|e| Error::Transport {
        url: url.to_string(),
        attempts,
        source: Box::new(ureq::Error::Io(e)),
    })
}

fn probe_once(opts: &Opts<'_>, url: &str, repo_id: &str, filename: &str) -> Result<FileMeta> {
    let _permit = opts.permits.acquire(opts.flow);

    let mut req = opts.ranged.head(url);
    if let Some(token) = opts.token {
        req = req.header("Authorization", format!("Bearer {token}"));
    }
    let res = req
        .call()
        .map_err(|e| classify(e, url, repo_id, filename, opts.retries))?;

    let status = res.status().as_u16();
    let header = |name: &str| {
        res.headers()
            .get(name)
            .and_then(|v| v.to_str().ok())
            .map(|v| v.trim_matches('"').to_string())
    };

    if status == 404 {
        return Err(Error::NotFound {
            repo: repo_id.to_string(),
            filename: filename.to_string(),
        });
    }

    let missing = |header: &'static str| Error::MissingHeader {
        url: url.to_string(),
        header,
    };
    let commit = match (header("x-repo-commit"), header("location")) {
        (Some(commit), _) => commit,
        (None, Some(location)) => {
            return Err(Error::RepoRedirect {
                repo: repo_id.to_string(),
                filename: filename.to_string(),
                location,
            });
        }
        (None, None) => return Err(missing("x-repo-commit")),
    };

    match status {
        // LFS: redirected off-host to a presigned CDN URL.
        302 => Ok(FileMeta {
            url: header("location").ok_or_else(|| missing("location"))?,
            etag: header("x-linked-etag").ok_or_else(|| missing("x-linked-etag"))?,
            commit,
            size: Some(
                header("x-linked-size")
                    .and_then(|v| v.parse().ok())
                    .ok_or_else(|| missing("x-linked-size"))?,
            ),
            lfs: true,
            send_token: false,
        }),
        // Git-tracked: the redirect stays on huggingface.co, so keep the
        // original URL and let a redirect-following agent walk it.
        307 => Ok(FileMeta {
            url: url.to_string(),
            etag: header("x-linked-etag")
                .or_else(|| header("etag"))
                .ok_or_else(|| missing("x-linked-etag"))?,
            commit,
            size: None,
            lfs: false,
            send_token: true,
        }),
        200 => Ok(FileMeta {
            url: url.to_string(),
            etag: header("etag").ok_or_else(|| missing("etag"))?,
            commit,
            size: header("content-length").and_then(|v| v.parse().ok()),
            lfs: false,
            send_token: true,
        }),
        _ => Err(Error::UnexpectedStatus {
            url: url.to_string(),
            status,
        }),
    }
}

/// Re-resolves a file: another probe of `/resolve/`, pinned to the commit the
/// first one found, so the answer can only be a fresh signature for the same
/// content.
pub(crate) type Resolve<'a> = dyn Fn() -> Result<FileMeta> + Sync + 'a;

/// Where a file's bytes currently come from.
///
/// The Hub's LFS redirect is presigned for an hour, and a download can
/// outlive that: a large file on a slow link, or any shard sharing the
/// connection budget with seven others. Pieces are requested for as long as
/// the file is downloading, so once the URL lapses every one of them is
/// refused. A 403 from a URL that has already served bytes is that expiry,
/// and is answered by resolving the file again; a 403 from a URL that never
/// served anything is a real refusal, and fails as it always did.
struct Source<'a> {
    signed: Mutex<Signed>,
    resolve: &'a Resolve<'a>,
    etag: &'a str,
    filename: &'a str,
}

struct Signed {
    url: String,
    /// Bumped on every re-resolve, so a refused worker can tell a URL that
    /// lapsed from one another worker has already replaced.
    generation: u64,
    served: bool,
}

/// The URL one request was made against.
struct Lease {
    url: String,
    generation: u64,
}

impl<'a> Source<'a> {
    fn new(meta: &'a FileMeta, filename: &'a str, resolve: &'a Resolve<'a>) -> Self {
        Self {
            signed: Mutex::new(Signed {
                url: meta.url.clone(),
                generation: 0,
                served: false,
            }),
            resolve,
            etag: &meta.etag,
            filename,
        }
    }

    /// Run `request` against the current URL, re-resolving it when it lapses.
    fn fetch<T>(&self, mut request: impl FnMut(&Lease) -> Result<T>) -> Result<T> {
        loop {
            let lease = {
                let signed = self.signed.lock().expect("source poisoned");
                Lease {
                    url: signed.url.clone(),
                    generation: signed.generation,
                }
            };
            match request(&lease) {
                Err(Error::UnexpectedStatus { status: 403, .. }) if self.renew(&lease)? => {}
                result => return result,
            }
        }
    }

    /// The URL `lease` names has answered with bytes.
    fn served(&self, lease: &Lease) {
        let mut signed = self.signed.lock().expect("source poisoned");
        if signed.generation == lease.generation {
            signed.served = true;
        }
    }

    /// Replace the URL that refused `lease`. `false` means the refusal was
    /// real — that URL never served a byte — and must be returned as is.
    fn renew(&self, lease: &Lease) -> Result<bool> {
        {
            let signed = self.signed.lock().expect("source poisoned");
            if signed.generation != lease.generation {
                return Ok(true);
            }
            if !signed.served {
                return Ok(false);
            }
        }
        // Not under the lock: resolving waits for a permit, and a worker
        // holding one takes this lock to report that its URL served.
        let fresh = (self.resolve)()?;
        if fresh.etag != self.etag {
            return Err(Error::HashMismatch {
                filename: self.filename.to_string(),
                expected: self.etag.to_string(),
                actual: fresh.etag,
            });
        }
        let mut signed = self.signed.lock().expect("source poisoned");
        if signed.generation == lease.generation {
            *signed = Signed {
                url: fresh.url,
                generation: lease.generation + 1,
                served: false,
            };
        }
        Ok(true)
    }
}

/// Fetch `meta` into `dest`, in pieces when the size is known and large
/// enough, resuming whatever a previous run completed.
pub(crate) fn fetch(
    opts: &Opts<'_>,
    meta: &FileMeta,
    filename: &str,
    dest: &Path,
    progress: &mut dyn Progress,
    resolve: &Resolve<'_>,
) -> Result<()> {
    let source = Source::new(meta, filename, resolve);
    progress.init(meta.size.unwrap_or(0), filename);
    match meta.size {
        Some(size) if size >= opts.parallel_threshold && opts.chunks > 1 => {
            fetch_pieces(opts, &source, meta, size, dest, progress)
        }
        _ => fetch_stream(opts, &source, meta, dest, progress),
    }
}

/// Single sequential stream, for small and unknown-size files where
/// splitting costs more in requests than it saves.
fn fetch_stream(
    opts: &Opts<'_>,
    source: &Source<'_>,
    meta: &FileMeta,
    dest: &Path,
    progress: &mut dyn Progress,
) -> Result<()> {
    let filename = source.filename;
    let sink = Mutex::new(progress);
    with_retry(&opts.budget, 0, 1, || {
        source.fetch(|lease| {
            let _permit = opts.permits.acquire(opts.flow);

            let mut req = opts.following.get(&lease.url);
            if let Some(token) = meta.send_token.then_some(opts.token).flatten() {
                req = req.header("Authorization", format!("Bearer {token}"));
            }
            let res = req
                .call()
                .map_err(|e| classify(e, &lease.url, "", filename, opts.retries))?;
            source.served(lease);

            // Restart from zero: without a length there is nothing to resume
            // against, and these files are small by construction.
            let mut reader = res.into_body().into_reader();
            let mut file = File::create(dest).map_err(|e| Error::io(dest, e))?;
            let mut buf = vec![0u8; BUF];
            loop {
                let n = read_body(&mut reader, &mut buf, &lease.url, opts.retries)?;
                if n == 0 {
                    break;
                }
                std::io::Write::write_all(&mut file, &buf[..n]).map_err(|e| Error::io(dest, e))?;
                let mut sink = sink.lock().expect("progress sink poisoned");
                if sink.cancelled() {
                    return Err(Error::Cancelled {
                        filename: filename.to_string(),
                    });
                }
                sink.update(n as u64);
            }
            Ok(())
        })
    })
}

/// Split into pieces and fetch them concurrently with positional writes.
///
/// Positional (`pwrite`) rather than seek-then-write under a lock: the lock
/// would serialise every write and hand back the cost of splitting with none
/// of the benefit. The only shared mutable state is the progress sink and
/// the resume log, neither of which touches the payload.
fn fetch_pieces(
    opts: &Opts<'_>,
    source: &Source<'_>,
    meta: &FileMeta,
    size: u64,
    dest: &Path,
    progress: &mut dyn Progress,
) -> Result<()> {
    let file = OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .truncate(false)
        .open(dest)
        .map_err(|e| Error::io(dest, e))?;
    file.set_len(size).map_err(|e| Error::io(dest, e))?;

    // Capped at `PIECE`, and small enough that a file just over the
    // threshold still spreads across every range it is allowed.
    let piece = PIECE.min(size.div_ceil(opts.chunks)).max(1);
    let count = size.div_ceil(piece);
    let resume = ResumeLog::open(dest, piece, count)?;

    // Bytes a previous run already placed. Reported up front so a resumed
    // download does not look like it restarted.
    let done: u64 = (0..count).map(|idx| resume.get(idx)).sum();
    if done > 0 {
        progress.resumed(done);
    }

    let pieces = Pieces {
        opts,
        source,
        send_token: meta.send_token,
        file,
        dest,
        size,
        piece,
        count,
        resume,
        next: AtomicU64::new(0),
        stop: AtomicBool::new(false),
        unreported: AtomicU64::new(0),
        sink: Mutex::new(progress),
    };

    let workers = opts.chunks.min(count) as usize;
    std::thread::scope(|s| {
        let running: Vec<_> = (0..workers)
            .map(|slot| {
                let pieces = &pieces;
                s.spawn(move || pieces.work(slot, workers))
            })
            .collect();
        running.into_iter().try_for_each(|worker| {
            worker
                .join()
                .unwrap_or_else(|panic| std::panic::resume_unwind(panic))
        })
    })?;

    // The `?` above keeps the log on failure or cancellation — it is what
    // makes the next attempt resume instead of restart.
    let rest = pieces.unreported.swap(0, Ordering::Relaxed);
    if rest > 0 {
        pieces
            .sink
            .lock()
            .expect("progress sink poisoned")
            .update(rest);
    }
    pieces.resume.discard();
    Ok(())
}

/// One file mid-fetch, shared by the workers claiming its pieces.
struct Pieces<'a> {
    opts: &'a Opts<'a>,
    source: &'a Source<'a>,
    send_token: bool,
    file: File,
    dest: &'a Path,
    size: u64,
    piece: u64,
    count: u64,
    resume: ResumeLog,
    /// The next piece nobody has claimed.
    next: AtomicU64,
    /// Set by the first worker to fail, so the rest stop claiming pieces
    /// for a download that is already lost.
    stop: AtomicBool,
    /// Bytes landed but not yet reported, batched to a report per [`BUF`].
    unreported: AtomicU64,
    sink: Mutex<&'a mut dyn Progress>,
}

impl Pieces<'_> {
    /// Claim and fetch pieces until none are left or one fails.
    fn work(&self, slot: usize, workers: usize) -> Result<()> {
        while !self.stop.load(Ordering::Relaxed) {
            let idx = self.next.fetch_add(1, Ordering::Relaxed);
            if idx >= self.count {
                break;
            }
            let fetched = with_retry(&self.opts.budget, slot, workers, || {
                self.source.fetch(|lease| self.fetch_piece(idx, lease))
            });
            if let Err(e) = fetched {
                self.stop.store(true, Ordering::Relaxed);
                return Err(e);
            }
        }
        Ok(())
    }

    fn fetch_piece(&self, idx: u64, lease: &Lease) -> Result<()> {
        let filename = self.source.filename;
        let start = idx * self.piece;
        let end = (start + self.piece).min(self.size) - 1;
        // Re-read on every attempt: a retry must pick up from wherever the
        // failed attempt actually got to, not from where this piece began.
        let already = self.resume.get(idx);
        if already > end - start {
            return Ok(());
        }
        let from = start + already;

        let _permit = self.opts.permits.acquire(self.opts.flow);

        let mut req = self
            .opts
            .ranged
            .get(&lease.url)
            .header("Range", format!("bytes={from}-{end}"));
        if let Some(token) = self.send_token.then_some(self.opts.token).flatten() {
            req = req.header("Authorization", format!("Bearer {token}"));
        }
        let res = req
            .call()
            .map_err(|e| classify(e, &lease.url, "", filename, self.opts.retries))?;

        // A server that ignored `Range` answers 200 and would stream the whole
        // file into this piece's slot, silently corrupting it. Refuse instead.
        let status = res.status().as_u16();
        if status != 206 {
            return Err(Error::UnexpectedStatus {
                url: lease.url.clone(),
                status,
            });
        }
        self.source.served(lease);

        let mut reader = res.into_body().into_reader();
        let mut buf = vec![0u8; BUF];
        let mut at = from;
        let mut since_log = 0u64;

        loop {
            let n = read_body(&mut reader, &mut buf, &lease.url, self.opts.retries)?;
            if n == 0 {
                break;
            }
            write_at(&self.file, &buf[..n], at).map_err(|e| Error::io(self.dest, e))?;
            at += n as u64;
            since_log += n as u64;

            // Payload first, then the counter: the log may lag reality, which
            // costs a re-fetch, but it can never claim bytes that are not there.
            if since_log >= RESUME_STRIDE {
                self.resume.set(idx, at - start);
                since_log = 0;
            }

            if self.unreported.fetch_add(n as u64, Ordering::Relaxed) + n as u64 >= BUF as u64 {
                let carried = self.unreported.swap(0, Ordering::Relaxed);
                let mut sink = self.sink.lock().expect("progress sink poisoned");
                if sink.cancelled() {
                    self.resume.set(idx, at - start);
                    return Err(Error::Cancelled {
                        filename: filename.to_string(),
                    });
                }
                sink.update(carried);
            }
        }
        self.resume.set(idx, at - start);
        Ok(())
    }
}

#[cfg(unix)]
fn write_at(file: &File, buf: &[u8], offset: u64) -> std::io::Result<()> {
    std::os::unix::fs::FileExt::write_all_at(file, buf, offset)
}

#[cfg(windows)]
fn write_at(file: &File, buf: &[u8], mut offset: u64) -> std::io::Result<()> {
    let mut written = 0;
    while written < buf.len() {
        let n = std::os::windows::fs::FileExt::seek_write(file, &buf[written..], offset)?;
        written += n;
        offset += n as u64;
    }
    Ok(())
}

/// Per-piece byte counts, so an interrupted download resumes instead of
/// restarting.
///
/// A fixed array of `u64`s beside the target, one slot per piece, updated in
/// place. The alternative some implementations reach for — appending a second
/// copy of every byte to a `.part` file — doubles both write bandwidth and
/// peak disk to track a number that fits in eight bytes.
///
/// Laid out as `[MAGIC, piece, done_0, done_1, …]`. The piece size is recorded
/// because a count means nothing apart from the bounds it counts into: a log
/// written for other bounds — another `chunks` setting, or the one-range-per-
/// chunk layout this crate used to write — would read as progress it is not,
/// and "resume" into a file that fails its hash. A log whose header does not
/// match is started over.
pub(crate) struct ResumeLog {
    file: File,
    path: std::path::PathBuf,
    pieces: u64,
}

/// First slot of every log this layout writes.
const MAGIC: u64 = u64::from_le_bytes(*b"hfdl-pc1");
/// Slots before the first count: the magic, then the piece size.
const HEADER: u64 = 2;

impl ResumeLog {
    pub(crate) fn open(dest: &Path, piece: u64, pieces: u64) -> Result<Self> {
        let path = dest.with_extension("resume");
        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(&path)
            .map_err(|e| Error::io(&path, e))?;
        let len = (HEADER + pieces) * 8;
        let log = Self { file, path, pieces };

        let fits = log.file.metadata().is_ok_and(|m| m.len() == len)
            && log.slot(0) == Some(MAGIC)
            && log.slot(1) == Some(piece);
        if !fits {
            let fresh = || {
                log.file.set_len(0)?;
                log.file.set_len(len)?;
                write_at(&log.file, &MAGIC.to_le_bytes(), 0)?;
                write_at(&log.file, &piece.to_le_bytes(), 8)
            };
            fresh().map_err(|e| Error::io(&log.path, e))?;
        }
        Ok(log)
    }

    fn slot(&self, slot: u64) -> Option<u64> {
        let mut buf = [0u8; 8];
        read_at(&self.file, &mut buf, slot * 8)
            .ok()
            .map(|()| u64::from_le_bytes(buf))
    }

    pub(crate) fn get(&self, idx: u64) -> u64 {
        debug_assert!(idx < self.pieces);
        self.slot(HEADER + idx).unwrap_or(0)
    }

    pub(crate) fn set(&self, idx: u64, value: u64) {
        debug_assert!(idx < self.pieces);
        // Best effort: losing an update costs a re-fetch on the next run,
        // never correctness, so a failure here must not abort the download.
        let _ = write_at(&self.file, &value.to_le_bytes(), (HEADER + idx) * 8);
    }

    fn discard(&self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

#[cfg(unix)]
fn read_at(file: &File, buf: &mut [u8], offset: u64) -> std::io::Result<()> {
    std::os::unix::fs::FileExt::read_exact_at(file, buf, offset)
}

#[cfg(windows)]
fn read_at(file: &File, buf: &mut [u8], offset: u64) -> std::io::Result<()> {
    let n = std::os::windows::fs::FileExt::seek_read(file, buf, offset)?;
    if n == buf.len() {
        Ok(())
    } else {
        Err(std::io::ErrorKind::UnexpectedEof.into())
    }
}

/// Digest `path` the way the Hub digested it, and compare against the etag.
///
/// LFS objects are SHA-256 of the content. Git-tracked files are a git blob
/// SHA-1, which is `sha1("blob <len>\0" + content)` — the header is what
/// makes it a *git* hash rather than a plain SHA-1, and omitting it makes
/// every small file fail to verify.
pub(crate) fn verify(path: &Path, meta: &FileMeta, filename: &str) -> Result<()> {
    let mut file = File::open(path).map_err(|e| Error::io(path, e))?;
    let len = file.metadata().map_err(|e| Error::io(path, e))?.len();
    let mut buf = vec![0u8; 1024 * 1024];

    let actual = if meta.lfs {
        let mut hasher = sha2::Sha256::new();
        loop {
            let n = file.read(&mut buf).map_err(|e| Error::io(path, e))?;
            if n == 0 {
                break;
            }
            hasher.update(&buf[..n]);
        }
        hex::encode(hasher.finalize())
    } else {
        let mut hasher = sha1::Sha1::new();
        hasher.update(format!("blob {len}\0").as_bytes());
        loop {
            let n = file.read(&mut buf).map_err(|e| Error::io(path, e))?;
            if n == 0 {
                break;
            }
            hasher.update(&buf[..n]);
        }
        hex::encode(hasher.finalize())
    };

    if actual != meta.etag {
        return Err(Error::HashMismatch {
            filename: filename.to_string(),
            expected: meta.etag.clone(),
            actual,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_files_retries_are_shared_not_multiplied() {
        // retries(5) == 5 attempts == 4 retries, for the whole file.
        let budget = RetryBudget::new(5);
        let claimed = std::iter::from_fn(|| budget.claim()).count();
        assert_eq!(claimed, 4);
        assert!(
            budget.claim().is_none(),
            "budget must not refill; eight ranges at five each would be 40"
        );
    }

    #[test]
    fn one_attempt_means_no_retries() {
        assert!(RetryBudget::new(1).claim().is_none());
    }

    #[test]
    fn backoff_staggers_ranges_so_they_do_not_retry_in_lockstep() {
        const SLOTS: usize = 8;
        let delays: Vec<Duration> = (0..SLOTS).map(|s| backoff(2, s, SLOTS)).collect();

        assert!(
            delays.windows(2).all(|w| w[0] < w[1]),
            "slots must not share a wake-up time: {delays:?}"
        );
        // Staggering must not blow past the exponential window it splits.
        let window = BACKOFF * (1u32 << 2);
        assert!(
            delays.iter().all(|d| *d <= window),
            "stagger exceeded its window {window:?}: {delays:?}"
        );
    }

    #[test]
    fn backoff_grows_with_the_attempt() {
        let first = backoff(0, 0, 4);
        let later = backoff(3, 0, 4);
        assert!(later > first, "{later:?} should exceed {first:?}");
    }
}
