// SPDX-License-Identifier: Apache-2.0
//! Metadata probe and the byte-moving half of the downloader.

use std::fs::{File, OpenOptions};
use std::io::Read;
use std::path::Path;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use rayon::prelude::*;
// `sha1` and `sha2` both re-export the same `digest::Digest`, so one import
// covers `Sha1` and `Sha256` alike.
use sha2::Digest as _;

use crate::Progress;
use crate::error::{Error, Result};
use crate::limit::Semaphore;

/// Read buffer, and the granularity at which progress is reported.
const BUF: usize = 256 * 1024;

/// How often a chunk records its progress for resume. Small enough that a
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

fn probe_once(opts: &Opts<'_>, url: &str, repo_id: &str, filename: &str) -> Result<FileMeta> {
    let _permit = opts.permits.acquire();

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

/// Fetch `meta` into `dest`, in parallel ranges when the size is known and
/// large enough, resuming whatever a previous run completed.
pub(crate) fn fetch(
    opts: &Opts<'_>,
    meta: &FileMeta,
    filename: &str,
    dest: &Path,
    progress: &mut dyn Progress,
) -> Result<()> {
    match meta.size {
        Some(size) if size >= opts.parallel_threshold && opts.chunks > 1 => {
            progress.init(size, filename);
            fetch_ranges(opts, meta, filename, size, dest, progress)
        }
        size => {
            progress.init(size.unwrap_or(0), filename);
            fetch_stream(opts, meta, filename, dest, progress)
        }
    }
}

/// Single sequential stream, for small and unknown-size files where
/// splitting costs more in requests than it saves.
fn fetch_stream(
    opts: &Opts<'_>,
    meta: &FileMeta,
    filename: &str,
    dest: &Path,
    progress: &mut dyn Progress,
) -> Result<()> {
    let sink = Mutex::new(progress);
    with_retry(&opts.budget, 0, 1, || {
        let _permit = opts.permits.acquire();

        let mut req = opts.following.get(&meta.url);
        if let Some(token) = meta.send_token.then_some(opts.token).flatten() {
            req = req.header("Authorization", format!("Bearer {token}"));
        }
        let res = req
            .call()
            .map_err(|e| classify(e, &meta.url, "", filename, opts.retries))?;

        // Restart from zero: without a length there is nothing to resume
        // against, and these files are small by construction.
        let mut reader = res.into_body().into_reader();
        let mut file = File::create(dest).map_err(|e| Error::io(dest, e))?;
        let mut buf = vec![0u8; BUF];
        loop {
            let n = read_chunk(&mut reader, &mut buf, dest, opts, &meta.url)?;
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
}

/// Read one buffer, mapping a mid-body failure to a transient error so the
/// retry layer can re-request rather than aborting the file.
fn read_chunk(
    reader: &mut impl Read,
    buf: &mut [u8],
    dest: &Path,
    _opts: &Opts<'_>,
    _url: &str,
) -> Result<usize> {
    reader.read(buf).map_err(|e| Error::io(dest, e))
}

/// Split into ranges and fetch them concurrently with positional writes.
///
/// Positional (`pwrite`) rather than seek-then-write under a lock: the lock
/// would serialise every write and hand back the cost of chunking with none
/// of the benefit. The only shared mutable state is the progress sink and
/// the resume log, neither of which touches the payload.
fn fetch_ranges(
    opts: &Opts<'_>,
    meta: &FileMeta,
    filename: &str,
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

    let resume = ResumeLog::open(dest, opts.chunks as usize)?;
    let span = size.div_ceil(opts.chunks);
    let ranges: Vec<(usize, u64, u64)> = (0..opts.chunks)
        .map(|i| (i as usize, i * span, ((i + 1) * span).min(size) - 1))
        .filter(|(_, start, end)| start <= end)
        .collect();

    // Bytes a previous run already placed. Reported up front so a resumed
    // download does not look like it restarted.
    let done: u64 = (0..ranges.len()).map(|i| resume.get(i)).sum();
    progress.update(done);

    let advanced = AtomicU64::new(0);
    let sink = Mutex::new(progress);

    let slots = ranges.len();
    let outcome = ranges.par_iter().try_for_each(|&(idx, start, end)| {
        with_retry(&opts.budget, idx, slots, || {
            one_range(
                opts, meta, filename, &file, dest, idx, start, end, &resume, &advanced, &sink,
            )
        })
    });

    // Keep the log on failure or cancellation — it is what makes the next
    // attempt resume instead of restart.
    outcome?;
    resume.discard();
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn one_range(
    opts: &Opts<'_>,
    meta: &FileMeta,
    filename: &str,
    file: &File,
    dest: &Path,
    idx: usize,
    start: u64,
    end: u64,
    resume: &ResumeLog,
    advanced: &AtomicU64,
    sink: &Mutex<&mut dyn Progress>,
) -> Result<()> {
    // Re-read on every attempt: a retry must pick up from wherever the
    // failed attempt actually got to, not from where this range began.
    let already = resume.get(idx);
    if already > end - start {
        return Ok(());
    }
    let from = start + already;

    let _permit = opts.permits.acquire();

    let mut req = opts
        .ranged
        .get(&meta.url)
        .header("Range", format!("bytes={from}-{end}"));
    if let Some(token) = meta.send_token.then_some(opts.token).flatten() {
        req = req.header("Authorization", format!("Bearer {token}"));
    }
    let res = req
        .call()
        .map_err(|e| classify(e, &meta.url, "", filename, opts.retries))?;

    // A server that ignored `Range` answers 200 and would stream the whole
    // file into this chunk's slot, silently corrupting it. Refuse instead.
    let status = res.status().as_u16();
    if status != 206 {
        return Err(Error::UnexpectedStatus {
            url: meta.url.clone(),
            status,
        });
    }

    let mut reader = res.into_body().into_reader();
    let mut buf = vec![0u8; BUF];
    let mut at = from;
    let mut since_log = 0u64;

    loop {
        let n = read_chunk(&mut reader, &mut buf, dest, opts, &meta.url)?;
        if n == 0 {
            break;
        }
        write_at(file, &buf[..n], at).map_err(|e| Error::io(dest, e))?;
        at += n as u64;
        since_log += n as u64;

        // Payload first, then the counter: the log may lag reality, which
        // costs a re-fetch, but it can never claim bytes that are not there.
        if since_log >= RESUME_STRIDE {
            resume.set(idx, at - start);
            since_log = 0;
        }

        if advanced.fetch_add(n as u64, Ordering::Relaxed) + n as u64 >= BUF as u64 {
            let carried = advanced.swap(0, Ordering::Relaxed);
            let mut sink = sink.lock().expect("progress sink poisoned");
            if sink.cancelled() {
                resume.set(idx, at - start);
                return Err(Error::Cancelled {
                    filename: filename.to_string(),
                });
            }
            sink.update(carried);
        }
    }
    resume.set(idx, at - start);
    Ok(())
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

/// Per-chunk byte counts, so an interrupted download resumes instead of
/// restarting.
///
/// A fixed array of `u64`s beside the target, one slot per chunk, updated in
/// place. The alternative some implementations reach for — appending a second
/// copy of every byte to a `.part` file — doubles both write bandwidth and
/// peak disk to track a number that fits in eight bytes.
pub(crate) struct ResumeLog {
    file: File,
    path: std::path::PathBuf,
    slots: usize,
}

impl ResumeLog {
    pub(crate) fn open(dest: &Path, slots: usize) -> Result<Self> {
        let path = dest.with_extension("resume");
        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(&path)
            .map_err(|e| Error::io(&path, e))?;
        file.set_len((slots * 8) as u64)
            .map_err(|e| Error::io(&path, e))?;
        Ok(Self { file, path, slots })
    }

    pub(crate) fn get(&self, idx: usize) -> u64 {
        debug_assert!(idx < self.slots);
        let mut buf = [0u8; 8];
        match read_at(&self.file, &mut buf, (idx * 8) as u64) {
            Ok(()) => u64::from_le_bytes(buf),
            Err(_) => 0,
        }
    }

    pub(crate) fn set(&self, idx: usize, value: u64) {
        debug_assert!(idx < self.slots);
        // Best effort: losing an update costs a re-fetch on the next run,
        // never correctness, so a failure here must not abort the download.
        let _ = write_at(&self.file, &value.to_le_bytes(), (idx * 8) as u64);
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
