// SPDX-License-Identifier: Apache-2.0
//! End-to-end tests against the real Hub.
//!
//! `#[ignore]` by default: these need network, and a test that goes red on a
//! plane is a test people learn to ignore. Run them deliberately:
//!
//! ```text
//! cargo test -p hf-hub-downloader --test live_hub -- --ignored --nocapture
//! ```
//!
//! They are the only thing that can catch drift in the Hub's redirect and
//! digest behaviour, which is not documented anywhere and which every
//! from-scratch implementation of this gets wrong at least once.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use hf_hub_downloader::{Client, Error, Progress};

/// Small, public, stable, and — importantly — has both a git-tracked file
/// (307 + git blob SHA-1) and an LFS file (302 + SHA-256).
const REPO: &str = "Qwen/Qwen3-0.6B";
const GIT_FILE: &str = "config.json";
const LFS_FILE: &str = "tokenizer.json";
const LFS_BYTES: u64 = 11_422_654;

#[derive(Default)]
struct Counting {
    total: AtomicU64,
    seen: AtomicU64,
    finished: AtomicU64,
}

impl Progress for &Counting {
    fn init(&mut self, total: u64, _filename: &str) {
        self.total.store(total, Ordering::Relaxed);
    }
    fn update(&mut self, delta: u64) {
        self.seen.fetch_add(delta, Ordering::Relaxed);
    }
    fn finish(&mut self) {
        self.finished.fetch_add(1, Ordering::Relaxed);
    }
}

/// Cancels once `stop_after` bytes have been seen, and records the very
/// first delta it is given. That first delta is the credit for work a
/// previous run left behind, so it is how a resumed download proves it
/// resumed rather than started over.
#[derive(Default)]
struct Interrupting {
    stop_after: u64,
    seen: AtomicU64,
    first_delta: AtomicU64,
    stop: AtomicBool,
}

impl Progress for &Interrupting {
    fn init(&mut self, _total: u64, _filename: &str) {}
    fn update(&mut self, delta: u64) {
        let _ = self
            .first_delta
            .compare_exchange(0, delta, Ordering::SeqCst, Ordering::SeqCst);
        let seen = self.seen.fetch_add(delta, Ordering::SeqCst) + delta;
        if self.stop_after > 0 && seen >= self.stop_after {
            self.stop.store(true, Ordering::SeqCst);
        }
    }
    fn finish(&mut self) {}
    fn cancelled(&self) -> bool {
        self.stop.load(Ordering::SeqCst)
    }
}

/// Sum of the per-chunk counters in the resume sidecar next to the blob.
fn resume_total(blobs: &Path) -> u64 {
    let log = std::fs::read_dir(blobs)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.extension().is_some_and(|e| e == "resume"))
        .expect("no .resume sidecar — the interrupted run recorded nothing");
    let bytes = std::fs::read(log).unwrap();
    let (slots, _partial) = bytes.as_chunks::<8>();
    slots.iter().copied().map(u64::from_le_bytes).sum()
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("hfdl-live-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// The git-tracked path: 307, relative Location we never follow by hand, and
/// a git blob SHA-1 rather than a plain SHA-1 of the content.
#[test]
#[ignore = "requires network"]
fn git_tracked_file_downloads_and_verifies() {
    let root = scratch("git");
    let client = Client::builder().cache_root(&root).build();
    let path = client.model(REPO).get(GIT_FILE).expect("download failed");

    let body = std::fs::read_to_string(&path).unwrap();
    assert!(
        body.contains("\"model_type\""),
        "downloaded something that is not a config: {body:.120}"
    );
    // Verification is inside `get`; reaching here means the git blob SHA-1
    // matched, which is the check that fails if the `blob <len>\0` header is
    // omitted or a SHA-256 is used instead.
}

/// The LFS path: 302 to an off-host CDN, SHA-256, and — with the threshold
/// lowered — genuinely concurrent ranges.
#[test]
#[ignore = "requires network"]
fn lfs_file_downloads_in_parallel_ranges_and_verifies() {
    let root = scratch("lfs");
    let client = Client::builder()
        .cache_root(&root)
        .chunks(8)
        .parallel_threshold(1024 * 1024)
        .build();

    let counter = Counting::default();
    let path = client
        .model(REPO)
        .download(LFS_FILE, &mut &counter)
        .expect("download failed");

    assert_eq!(
        std::fs::metadata(&path).unwrap().len(),
        LFS_BYTES,
        "wrong length — a chunk boundary is off, or a range response was \
         written into the wrong slot"
    );
    assert_eq!(counter.total.load(Ordering::Relaxed), LFS_BYTES);
    assert_eq!(counter.finished.load(Ordering::Relaxed), 1);

    // The resume sidecar must not survive a completed download.
    let leftovers: Vec<_> = std::fs::read_dir(root.join("models--Qwen--Qwen3-0.6B").join("blobs"))
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.ends_with(".resume") || n.ends_with(".incomplete"))
        .collect();
    assert!(
        leftovers.is_empty(),
        "left temp files behind: {leftovers:?}"
    );
}

/// Kill a download partway, then finish it: the second run must pick up the
/// bytes the first one placed, not start over, and must still verify.
///
/// This is the path worth distrusting. Resume arithmetic is per-chunk — each
/// range restarts at `start + recorded`, and an off-by-one there produces a
/// file that is the right length, passes casually, and fails its hash.
#[test]
#[ignore = "requires network"]
fn an_interrupted_download_resumes_where_it_stopped() {
    let root = scratch("resume");
    let blobs = root.join("models--Qwen--Qwen3-0.6B").join("blobs");
    let build = || {
        Client::builder()
            .cache_root(&root)
            .chunks(4)
            .parallel_threshold(1024 * 1024)
            .build()
    };

    // Stop early enough that plenty is left, late enough that real bytes
    // landed.
    let interrupted = Interrupting {
        stop_after: 2 * 1024 * 1024,
        ..Default::default()
    };
    let err = build()
        .model(REPO)
        .download(LFS_FILE, &mut &interrupted)
        .expect_err("should have been cancelled");
    assert!(err.is_cancelled(), "wrong error: {err}");

    let carried = resume_total(&blobs);
    assert!(carried > 0, "cancelled run recorded no progress");
    assert!(
        carried < LFS_BYTES,
        "cancelled run claims the whole file ({carried} of {LFS_BYTES})"
    );

    // Second run: same client settings, no cancellation.
    let finishing = Interrupting::default();
    let path = build()
        .model(REPO)
        .download(LFS_FILE, &mut &finishing)
        .expect("resumed download failed");

    assert_eq!(
        finishing.first_delta.load(Ordering::SeqCst),
        carried,
        "second run did not credit the first run's bytes — it restarted"
    );
    assert_eq!(
        std::fs::metadata(&path).unwrap().len(),
        LFS_BYTES,
        "resumed file is the wrong length"
    );
    // Reaching here means the SHA-256 matched: the resumed ranges were
    // written at the right offsets, not merely the right number of bytes.
    assert!(
        !blobs.join("").exists() || resume_sidecar_gone(&blobs),
        "resume sidecar outlived a completed download"
    );
}

fn resume_sidecar_gone(blobs: &Path) -> bool {
    std::fs::read_dir(blobs)
        .map(|d| {
            !d.filter_map(|e| e.ok())
                .any(|e| e.path().extension().is_some_and(|x| x == "resume"))
        })
        .unwrap_or(true)
}

/// An absent file must report absence, immediately.
///
/// Probing for an optional file is routine — `chat_template.jinja` on every
/// cached load, `model.safetensors` before falling back to a shard index —
/// so this path runs constantly. Getting it wrong cost two things at once:
/// the caller saw a transport failure instead of a `NotFound` and so could
/// not fall through to the sharded layout, and the 404 slept through the
/// whole retry ladder first.
#[test]
#[ignore = "requires network"]
fn an_absent_file_is_not_found_and_does_not_burn_the_retry_ladder() {
    let root = scratch("absent");
    let client = Client::builder().cache_root(&root).retries(5).build();

    let started = std::time::Instant::now();
    let err = client
        .model(REPO)
        .get("definitely-not-a-real-file.json")
        .expect_err("a missing file must not succeed");
    let elapsed = started.elapsed();

    assert!(
        err.is_not_found(),
        "absence must be typed, or callers cannot distinguish it from failure: {err}"
    );
    // Five attempts would sleep 250+500+1000+2000ms before answering.
    assert!(
        elapsed < std::time::Duration::from_millis(2500),
        "took {elapsed:?} — a 404 was retried instead of returned"
    );
}

/// A wrongly-cased repo id is the commonest way to hold this crate wrong, and
/// the Hub answers it with a redirect with no `x-repo-commit`. The redirect
/// already carries the answer, so the error has to hand it back.
#[test]
#[ignore = "requires network"]
fn a_wrongly_cased_repo_id_reports_what_the_hub_wants_instead() {
    let root = scratch("wrong-case");
    let client = Client::builder().cache_root(&root).build();

    let err = client
        .model("Qwen/qwen3-0.6b")
        .get(GIT_FILE)
        .expect_err("a repo the Hub redirects but will not serve must not succeed");

    assert!(
        matches!(&err, Error::RepoRedirect { location, .. } if location.contains(REPO)),
        "the error must name the id that works, and it is right there in \
         `location`: {err:?}"
    );
}

/// The interop contract: a cache we wrote must be resolvable by every other
/// Hub client — `huggingface-cli`, Python `transformers`, `hf-hub`, and our
/// own `scr model ls`. If this fails we have built a private cache nothing
/// else can see.
///
/// The steps below are the foreign resolver, spelled out in literal path
/// components on purpose: it must not call our own `cache` helpers, or it
/// would prove only that we agree with ourselves.
#[test]
#[ignore = "requires network"]
fn the_cache_we_write_resolves_the_way_every_other_client_reads_it() {
    let root = scratch("interop");
    let client = Client::builder().cache_root(&root).build();
    client.model(REPO).get(GIT_FILE).expect("download failed");

    // 1. `owner/name` -> `models--owner--name`.
    let repo = root.join("models--Qwen--Qwen3-0.6B");
    assert!(
        repo.is_dir(),
        "no repo folder at {} — the `models--<owner>--<name>` naming is wrong",
        repo.display()
    );

    // 2. `refs/<revision>` holds the commit. Nobody else trims it: hf-hub
    //    joins the file's contents verbatim, so a trailing newline here
    //    yields a snapshot directory no other client can open.
    let raw = std::fs::read_to_string(repo.join("refs").join("main"))
        .expect("no refs/main — the revision -> commit mapping was never written");
    assert_eq!(
        raw.trim(),
        raw,
        "refs/main is padded with whitespace ({raw:?}); readers that do not trim \
         will look for a snapshot directory that does not exist"
    );
    assert!(
        raw.len() == 40 && raw.bytes().all(|b| b.is_ascii_hexdigit()),
        "refs/main is {raw:?}, not a 40-char commit hash — cache scanners key on the hash"
    );

    // 3. `snapshots/<commit>/<file>` is the path handed back to callers.
    let entry = repo.join("snapshots").join(&raw).join(GIT_FILE);
    assert!(
        entry.exists(),
        "nothing resolves at {}: the layout is wrong \
         (refs/<rev> -> commit -> snapshots/<commit>/<file>)",
        entry.display()
    );

    // 4. The entry is a relative symlink into `blobs/<etag>`. That indirection
    //    is not decoration — it is how the cache stays movable and how
    //    `huggingface-cli scan-cache`/`delete-cache` account for the bytes.
    #[cfg(unix)]
    {
        let target = std::fs::read_link(&entry)
            .expect("snapshot entry is a plain file, not a symlink into blobs/");
        assert!(
            target.is_relative(),
            "symlink target {} is absolute — the cache cannot be moved or copied",
            target.display()
        );
        let blob = repo.join("blobs").join(
            target
                .file_name()
                .expect("a link target always has a file name"),
        );
        assert!(
            blob.is_file(),
            "link points at {}, which is not a blob under blobs/",
            target.display()
        );
    }
}

/// A second fetch must not touch the network. Proven by pointing the client
/// at an endpoint that cannot resolve: if it tries, it fails.
#[test]
#[ignore = "requires network"]
fn second_fetch_is_served_from_cache_without_network() {
    let root = scratch("offline");
    Client::builder()
        .cache_root(&root)
        .build()
        .model(REPO)
        .get(GIT_FILE)
        .expect("first download failed");

    let offline = Client::builder()
        .cache_root(&root)
        .endpoint("https://invalid.invalid")
        .build();
    offline
        .model(REPO)
        .get(GIT_FILE)
        .expect("cached fetch must not touch the network");
}
