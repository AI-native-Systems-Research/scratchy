// SPDX-License-Identifier: Apache-2.0
//! The downloader against a Hub served in-process, over loopback.
//!
//! The live tests prove the real Hub's contract, but they need a network and
//! cannot make the Hub misbehave. These serve the same contract — a 302 from
//! `/resolve/` carrying `x-linked-*`, ranged GETs against a signed URL — and
//! control what the live tests cannot: how long each piece takes, the order
//! requests arrive in, and when a signature lapses.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use sha2::Digest as _;

use crate::{Client, ClientBuilder, Error, Progress};

const REPO: &str = "owner/model";
const COMMIT: &str = "0123456789abcdef0123456789abcdef01234567";

/// What the fake Hub serves.
struct Hub {
    files: HashMap<&'static str, Vec<u8>>,
    /// Ranged GETs one signature answers before it lapses into 403s; `None`
    /// never lapses, `Some(0)` refuses from the first.
    lifetime: Option<usize>,
    /// How long each ranged GET is held, so requests overlap.
    latency: Duration,
    /// This many ranged GETs, the first to arrive, send half the body they
    /// promise and hang up — a connection reset mid-transfer.
    cut_short: usize,
    /// The first probes wait until this many have arrived, so files the test
    /// starts together really do start together.
    rendezvous: usize,
    seen: Mutex<Seen>,
    probed: Condvar,
}

/// What the fake Hub was asked.
#[derive(Default)]
struct Seen {
    /// The file each ranged GET was for, in arrival order.
    gets: Vec<&'static str>,
    /// Probes of `/resolve/`; the latest also names the current signature.
    probes: u64,
    /// GETs each signature has answered.
    served: HashMap<u64, usize>,
    /// GETs hung up on partway.
    cut: usize,
    in_flight: usize,
    peak: usize,
}

impl Hub {
    fn new(files: &[(&'static str, usize)]) -> Self {
        Self {
            files: files
                .iter()
                .map(|&(name, len)| (name, content(name, len)))
                .collect(),
            lifetime: None,
            latency: Duration::ZERO,
            cut_short: 0,
            rendezvous: 0,
            seen: Mutex::default(),
            probed: Condvar::new(),
        }
    }

    /// Serve on an ephemeral loopback port; returns the endpoint to point a
    /// client at.
    fn serve(self) -> (Arc<Self>, String) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind loopback");
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let hub = Arc::new(self);
        let (serving, cdn) = (hub.clone(), endpoint.clone());
        std::thread::spawn(move || {
            for stream in listener.incoming().flatten() {
                let (hub, cdn) = (serving.clone(), cdn.clone());
                std::thread::spawn(move || hub.handle(&cdn, stream));
            }
        });
        (hub, endpoint)
    }

    fn handle(&self, cdn: &str, stream: TcpStream) {
        let mut reader = BufReader::new(stream);
        let mut request = String::new();
        if reader.read_line(&mut request).is_err() {
            return;
        }
        let mut range = None;
        loop {
            let mut header = String::new();
            if reader.read_line(&mut header).unwrap_or(0) == 0 || header == "\r\n" {
                break;
            }
            if let Some((name, value)) = header.split_once(':')
                && name.eq_ignore_ascii_case("range")
            {
                range = value
                    .trim()
                    .strip_prefix("bytes=")
                    .and_then(|r| r.split_once('-'))
                    .map(|(a, b)| (a.parse().unwrap(), b.parse().unwrap()));
            }
        }
        let mut parts = request.split_whitespace();
        let response = match (parts.next(), parts.next()) {
            (Some("HEAD"), Some(target)) => self.probe(cdn, target),
            (Some("GET"), Some(target)) => self.get(target, range),
            _ => status(400),
        };
        let _ = reader.into_inner().write_all(&response);
    }

    fn probe(&self, cdn: &str, target: &str) -> Vec<u8> {
        let Some((name, body)) = target
            .strip_prefix(&format!("/{REPO}/resolve/"))
            .and_then(|t| t.split_once('/'))
            .and_then(|(_revision, file)| self.files.get_key_value(file))
        else {
            return status(404);
        };
        let signature = {
            let mut seen = self.seen.lock().unwrap();
            seen.probes += 1;
            let signature = seen.probes;
            self.probed.notify_all();
            while seen.probes < self.rendezvous as u64 {
                seen = self.probed.wait(seen).unwrap();
            }
            signature
        };
        format!(
            "HTTP/1.1 302 Found\r\nlocation: {cdn}/cdn/{name}?sig={signature}\r\n\
             x-repo-commit: {COMMIT}\r\nx-linked-etag: \"{}\"\r\nx-linked-size: {}\r\n\
             content-length: 0\r\nconnection: close\r\n\r\n",
            hex::encode(sha2::Sha256::digest(body)),
            body.len(),
        )
        .into_bytes()
    }

    fn get(&self, target: &str, range: Option<(usize, usize)>) -> Vec<u8> {
        let Some(((name, body), signature)) = target
            .strip_prefix("/cdn/")
            .and_then(|t| t.split_once("?sig="))
            .and_then(|(file, sig)| Some((self.files.get_key_value(file)?, sig.parse().ok()?)))
        else {
            return status(404);
        };
        let (from, to) = range.expect("the downloader asks for a range on this path");
        let cut = {
            let mut seen = self.seen.lock().unwrap();
            let served = seen.served.entry(signature).or_default();
            if self.lifetime.is_some_and(|lifetime| *served >= lifetime) {
                return status(403);
            }
            *served += 1;
            seen.gets.push(*name);
            seen.in_flight += 1;
            seen.peak = seen.peak.max(seen.in_flight);
            let cut = seen.cut < self.cut_short;
            seen.cut += usize::from(cut);
            cut
        };
        std::thread::sleep(self.latency);
        self.seen.lock().unwrap().in_flight -= 1;

        let mut response = format!(
            "HTTP/1.1 206 Partial Content\r\ncontent-length: {}\r\n\
             content-range: bytes {from}-{to}/{}\r\nconnection: close\r\n\r\n",
            to + 1 - from,
            body.len(),
        )
        .into_bytes();
        let end = if cut { from + (to - from) / 2 } else { to };
        response.extend_from_slice(&body[from..=end]);
        response
    }
}

fn status(code: u16) -> Vec<u8> {
    format!("HTTP/1.1 {code} Status\r\ncontent-length: 0\r\nconnection: close\r\n\r\n").into_bytes()
}

/// Different at every offset and in every file, so a piece written to the
/// wrong place — or into the wrong file — fails the hash.
fn content(name: &str, len: usize) -> Vec<u8> {
    let seed = name
        .bytes()
        .fold(0u64, |h, b| h.rotate_left(5) ^ u64::from(b));
    (0..len as u64)
        .map(|i| ((i ^ seed).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 56) as u8)
        .collect()
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("hfdl-local-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn client(endpoint: &str, root: &std::path::Path) -> ClientBuilder {
    Client::builder()
        .endpoint(endpoint)
        .cache_root(root)
        .parallel_threshold(1)
}

/// Records what the downloader reported, and cancels once it has seen
/// `stop_after` bytes (never, when zero).
#[derive(Default)]
struct Watch {
    stop_after: u64,
    resumed: AtomicU64,
    seen: AtomicU64,
}

impl Progress for &Watch {
    fn init(&mut self, _total: u64, _filename: &str) {}
    fn resumed(&mut self, bytes: u64) {
        self.resumed.store(bytes, Ordering::SeqCst);
    }
    fn update(&mut self, delta: u64) {
        self.seen.fetch_add(delta, Ordering::SeqCst);
    }
    fn finish(&mut self) {}
    fn cancelled(&self) -> bool {
        self.stop_after > 0 && self.seen.load(Ordering::SeqCst) >= self.stop_after
    }
}

/// The complaint this exists for: with eight shards downloading, the first
/// took every connection and finished before the second moved.
#[test]
fn shards_downloading_together_share_the_connections() {
    const PIECES: usize = 8;
    let (hub, endpoint) = Hub {
        latency: Duration::from_millis(20),
        rendezvous: 2,
        ..Hub::new(&[
            ("a.safetensors", PIECES * 4096),
            ("b.safetensors", PIECES * 4096),
        ])
    }
    .serve();
    let root = scratch("share");
    let client = client(&endpoint, &root)
        .chunks(PIECES as u64)
        .max_concurrency(2)
        .build();

    std::thread::scope(|s| {
        for name in ["a.safetensors", "b.safetensors"] {
            let repo = client.model(REPO);
            s.spawn(move || repo.get(name).expect("download failed"));
        }
    });

    let gets = hub.seen.lock().unwrap().gets.clone();
    for (this, other) in [("a", "b"), ("b", "a")] {
        let last = gets
            .iter()
            .rposition(|f| f.starts_with(this))
            .expect("every file was requested");
        let meanwhile = gets[..last].iter().filter(|f| f.starts_with(other)).count();
        assert!(
            meanwhile >= PIECES / 2,
            "{other} got {meanwhile} of {PIECES} pieces in before {this} finished — \
             one file held the connections while the other waited: {gets:?}"
        );
    }
    let _ = std::fs::remove_dir_all(root);
}

/// Sharing must not cost the single-file case: alone, a file gets every
/// connection there is.
#[test]
fn a_file_downloading_alone_uses_every_connection() {
    const PERMITS: usize = 4;
    let (hub, endpoint) = Hub {
        latency: Duration::from_millis(20),
        ..Hub::new(&[("model.safetensors", 16 * 4096)])
    }
    .serve();
    let root = scratch("alone");
    client(&endpoint, &root)
        .chunks(PERMITS as u64)
        .max_concurrency(PERMITS)
        .build()
        .model(REPO)
        .get("model.safetensors")
        .expect("download failed");

    assert_eq!(hub.seen.lock().unwrap().peak, PERMITS);
    let _ = std::fs::remove_dir_all(root);
}

/// A signed URL that lapses mid-download is re-signed, and that is not spent
/// from the retry budget: with no retries at all, a download that outlives
/// three signatures still completes and verifies.
#[test]
fn a_url_that_lapses_mid_download_is_re_signed() {
    const PIECES: usize = 8;
    let (hub, endpoint) = Hub {
        lifetime: Some(3),
        ..Hub::new(&[("model.safetensors", PIECES * 4096)])
    }
    .serve();
    let root = scratch("lapse");
    let path = client(&endpoint, &root)
        .chunks(PIECES as u64)
        .max_concurrency(2)
        .retries(1)
        .build()
        .model(REPO)
        .get("model.safetensors")
        .expect("a lapsed signature failed the download");

    assert_eq!(std::fs::read(path).unwrap(), hub.files["model.safetensors"]);
    let probes = hub.seen.lock().unwrap().probes;
    assert!(
        probes >= PIECES.div_ceil(3) as u64,
        "{PIECES} pieces at 3 per signature need at least 3 signatures, got {probes}"
    );
    let _ = std::fs::remove_dir_all(root);
}

/// A connection that drops partway through a piece is a retry, not the end
/// of the file.
#[test]
fn a_connection_dropped_mid_piece_is_retried() {
    let (hub, endpoint) = Hub {
        cut_short: 1,
        ..Hub::new(&[("model.safetensors", 8 * 4096)])
    }
    .serve();
    let root = scratch("reset");
    let path = client(&endpoint, &root)
        .chunks(8)
        .retries(2)
        .build()
        .model(REPO)
        .get("model.safetensors")
        .expect("one reset mid-piece failed the whole download");

    assert_eq!(hub.seen.lock().unwrap().cut, 1);
    assert_eq!(std::fs::read(path).unwrap(), hub.files["model.safetensors"]);
    let _ = std::fs::remove_dir_all(root);
}

/// The other side of re-signing: a URL refused before it served anything is
/// a real refusal, and must fail rather than re-sign forever.
#[test]
fn a_url_refused_from_the_start_fails_instead_of_re_signing() {
    let (hub, endpoint) = Hub {
        lifetime: Some(0),
        ..Hub::new(&[("model.safetensors", 8 * 4096)])
    }
    .serve();
    let root = scratch("refused");
    let err = client(&endpoint, &root)
        .chunks(8)
        .build()
        .model(REPO)
        .get("model.safetensors")
        .expect_err("every range was refused");

    assert!(
        matches!(err, Error::UnexpectedStatus { status: 403, .. }),
        "wrong error: {err}"
    );
    assert_eq!(
        hub.seen.lock().unwrap().probes,
        1,
        "re-signed a real refusal"
    );
    let _ = std::fs::remove_dir_all(root);
}

/// Stop partway, then finish: the second run credits what the first left and
/// fetches only the rest.
#[test]
fn an_interrupted_download_resumes_where_it_stopped() {
    const LEN: usize = 4 << 20;
    let (hub, endpoint) = Hub::new(&[("model.safetensors", LEN)]).serve();
    let root = scratch("resume");
    let build = || {
        client(&endpoint, &root)
            .chunks(4)
            .max_concurrency(1)
            .build()
    };

    let stopping = Watch {
        stop_after: LEN as u64 / 4,
        ..Watch::default()
    };
    let err = build()
        .model(REPO)
        .download("model.safetensors", &mut &stopping)
        .expect_err("should have been cancelled");
    assert!(err.is_cancelled(), "wrong error: {err}");

    let finishing = Watch::default();
    let path = build()
        .model(REPO)
        .download("model.safetensors", &mut &finishing)
        .expect("resumed download failed");

    let carried = finishing.resumed.load(Ordering::SeqCst);
    assert!(
        carried > 0 && carried < LEN as u64,
        "credited {carried} of {LEN} bytes — it restarted, or claims it never stopped"
    );
    assert_eq!(
        carried + finishing.seen.load(Ordering::SeqCst),
        LEN as u64,
        "the credit and the new bytes must add up to exactly the file"
    );
    assert_eq!(std::fs::read(path).unwrap(), hub.files["model.safetensors"]);
    let _ = std::fs::remove_dir_all(root);
}

/// A resume log records counts against piece bounds; read against other
/// bounds they claim bytes that were never written. A log from another
/// `chunks` setting — or from this crate's older layout — must be started
/// over, not trusted into a file that fails its hash.
#[test]
fn a_resume_log_for_other_piece_bounds_is_not_trusted() {
    const LEN: usize = 4 << 20;
    let (hub, endpoint) = Hub::new(&[("model.safetensors", LEN)]).serve();
    let root = scratch("geometry");

    let stopping = Watch {
        stop_after: LEN as u64 / 3,
        ..Watch::default()
    };
    let err = client(&endpoint, &root)
        .chunks(2)
        .max_concurrency(1)
        .build()
        .model(REPO)
        .download("model.safetensors", &mut &stopping)
        .expect_err("should have been cancelled");
    assert!(err.is_cancelled(), "wrong error: {err}");

    let finishing = Watch::default();
    let path = client(&endpoint, &root)
        .chunks(8)
        .build()
        .model(REPO)
        .download("model.safetensors", &mut &finishing)
        .expect("a stale log was trusted into a file that failed its hash");

    assert_eq!(
        finishing.resumed.load(Ordering::SeqCst),
        0,
        "credited bytes counted against other piece bounds"
    );
    assert_eq!(std::fs::read(path).unwrap(), hub.files["model.safetensors"]);
    let _ = std::fs::remove_dir_all(root);
}
