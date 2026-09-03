// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! The bench crate's HTTP client and its concurrency bound.
//!
//! `ureq` rather than `reqwest`: the benchmarks are the only reason the CLI
//! ever compiled reqwest, and reqwest costs 54 crates here — its own hyper /
//! rustls-platform-verifier / security-framework stack, plus `url` and the
//! 24-crate ICU subtree `url` drags in for IDN. ureq is already in the tree
//! for the Hub downloader, already on ring, and gives us the one thing
//! `bench serve` actually needs from an HTTP client: a `Read` over the
//! response body so SSE events can be timestamped as they arrive.
//!
//! The cost is that ureq is blocking, so `bench serve` spends a thread per
//! in-flight request instead of a task. At the concurrency benchmarks
//! actually run that is a fair trade, and it arguably measures better: a
//! thread parked in `read()` is woken by the kernel, with no async scheduler
//! between the socket becoming readable and the clock being read.

use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

pub(crate) use ureq::Agent;

/// Build the agent every benchmark uses.
///
/// `http_status_as_error(false)` because a 4xx/5xx is data here, not a
/// transport failure — the callers print the status and the body. The one
/// hour timeout matches the previous reqwest client and exists because a
/// cold-start server can take minutes to answer the first request.
pub(crate) fn agent(insecure: bool) -> Agent {
    let config = Agent::config_builder().timeout_global(Some(Duration::from_secs(3600)));

    let config = if insecure {
        config.tls_config(
            ureq::tls::TlsConfig::builder()
                .disable_verification(true)
                .build(),
        )
    } else {
        config
    };

    config.http_status_as_error(false).build().into()
}

/// GET a URL and return the raw body.
///
/// Streams through `into_reader()` rather than ureq's `read_to_vec()`, which
/// caps the body at a default size — the dataset parquet files are tens of
/// megabytes and would trip it.
pub(crate) fn get_bytes(agent: &Agent, url: &str) -> anyhow::Result<Vec<u8>> {
    let res = agent.get(url).header("User-Agent", "scr-bench").call()?;
    let mut out = Vec::new();
    std::io::Read::read_to_end(&mut res.into_body().into_reader(), &mut out)?;
    Ok(out)
}

/// GET a URL and return the body as text.
pub(crate) fn get_text(agent: &Agent, url: &str) -> anyhow::Result<String> {
    let res = agent.get(url).header("User-Agent", "scr-bench").call()?;
    let mut out = String::new();
    std::io::Read::read_to_string(&mut res.into_body().into_reader(), &mut out)?;
    Ok(out)
}

/// GET a URL and deserialize the body as JSON.
pub(crate) fn get_json<T: serde::de::DeserializeOwned>(
    agent: &Agent,
    url: &str,
) -> anyhow::Result<T> {
    Ok(serde_json::from_str(&get_text(agent, url)?)?)
}

/// A counting semaphore bounding in-flight requests.
///
/// `bench serve` needs `--max-concurrency` to mean the same thing it meant
/// under `tokio::sync::Semaphore`: at most N requests in flight, with the
/// pacing loop blocking rather than running ahead. The permit is returned on
/// drop, including on unwind, so a panicking request thread cannot leak a
/// slot and wedge the run.
pub(crate) struct Semaphore {
    free: Mutex<usize>,
    released: Condvar,
}

impl Semaphore {
    pub(crate) fn new(permits: usize) -> Self {
        Self {
            free: Mutex::new(permits.max(1)),
            released: Condvar::new(),
        }
    }

    /// Block until a permit is free, returning a guard that owns its share of
    /// the semaphore rather than borrowing it — so it can be moved into the
    /// request thread and released when that thread ends. This is the
    /// blocking counterpart of `tokio::sync::Semaphore::acquire_owned`, which
    /// is what the async version of this benchmark used.
    pub(crate) fn acquire_owned(self: &Arc<Self>) -> OwnedPermit {
        self.take();
        OwnedPermit {
            sem: Arc::clone(self),
        }
    }

    fn take(&self) {
        let mut free = self.free.lock().expect("semaphore poisoned");
        while *free == 0 {
            free = self.released.wait(free).expect("semaphore poisoned");
        }
        *free -= 1;
    }

    fn give_back(&self) {
        let mut free = self.free.lock().expect("semaphore poisoned");
        *free += 1;
        self.released.notify_one();
    }
}

pub(crate) struct OwnedPermit {
    sem: Arc<Semaphore>,
}

impl Drop for OwnedPermit {
    fn drop(&mut self) {
        self.sem.give_back();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// The bound is the whole point: with N permits, no more than N threads
    /// may be inside the critical section at once.
    #[test]
    fn semaphore_never_exceeds_its_permits() {
        let sem = Arc::new(Semaphore::new(3));
        let in_flight = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));

        let threads: Vec<_> = (0..32)
            .map(|_| {
                let sem = sem.clone();
                let in_flight = in_flight.clone();
                let peak = peak.clone();
                std::thread::spawn(move || {
                    let _permit = sem.acquire_owned();
                    let now = in_flight.fetch_add(1, Ordering::SeqCst) + 1;
                    peak.fetch_max(now, Ordering::SeqCst);
                    std::thread::sleep(Duration::from_millis(2));
                    in_flight.fetch_sub(1, Ordering::SeqCst);
                })
            })
            .collect();

        for t in threads {
            t.join().unwrap();
        }
        assert!(
            peak.load(Ordering::SeqCst) <= 3,
            "semaphore let too many in"
        );
        assert_eq!(in_flight.load(Ordering::SeqCst), 0);
    }

    /// A panicking holder must still return its permit, or one bad request
    /// shrinks the concurrency bound for the rest of the run.
    #[test]
    fn permit_is_returned_on_unwind() {
        let sem = Arc::new(Semaphore::new(1));
        let s = sem.clone();
        let _ = std::thread::spawn(move || {
            let _permit = s.acquire_owned();
            panic!("request thread died");
        })
        .join();
        // Would block forever if the permit leaked.
        let _permit = sem.acquire_owned();
    }
}
