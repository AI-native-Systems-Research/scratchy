// SPDX-License-Identifier: Apache-2.0
//! A counting semaphore bounding in-flight requests.
//!
//! The cap has to live here rather than in either parallelism axis, because
//! the two multiply: a caller fetching eight shards at once, each split into
//! eight ranges, opens sixty-four connections to one host. Capping chunks
//! alone does not bound that, and capping files alone gives up the
//! single-file case this crate exists for. One budget, shared by every
//! request a [`Client`](crate::Client) makes, bounds the product.

use std::sync::{Condvar, Mutex};

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

    /// Block until a permit is free. The guard returns it on drop, including
    /// on the error paths and on unwind.
    pub(crate) fn acquire(&self) -> Permit<'_> {
        let mut free = self.free.lock().expect("semaphore poisoned");
        while *free == 0 {
            free = self.released.wait(free).expect("semaphore poisoned");
        }
        *free -= 1;
        Permit { sem: self }
    }
}

pub(crate) struct Permit<'a> {
    sem: &'a Semaphore,
}

impl Drop for Permit<'_> {
    fn drop(&mut self) {
        let mut free = self.sem.free.lock().expect("semaphore poisoned");
        *free += 1;
        self.sem.released.notify_one();
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    #[test]
    fn never_exceeds_its_permits() {
        const PERMITS: usize = 3;
        const THREADS: usize = 32;

        let sem = Arc::new(Semaphore::new(PERMITS));
        let live = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));

        std::thread::scope(|s| {
            for _ in 0..THREADS {
                let (sem, live, peak) = (sem.clone(), live.clone(), peak.clone());
                s.spawn(move || {
                    let _permit = sem.acquire();
                    let now = live.fetch_add(1, Ordering::SeqCst) + 1;
                    peak.fetch_max(now, Ordering::SeqCst);
                    std::thread::yield_now();
                    live.fetch_sub(1, Ordering::SeqCst);
                });
            }
        });

        assert!(
            peak.load(Ordering::SeqCst) <= PERMITS,
            "{} concurrent holders with {PERMITS} permits",
            peak.load(Ordering::SeqCst)
        );
    }

    #[test]
    fn a_permit_is_returned_when_its_holder_panics() {
        let sem = Semaphore::new(1);
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _permit = sem.acquire();
            panic!("holder blew up mid-download");
        }));
        // Would block forever if the permit had leaked.
        let _permit = sem.acquire();
    }
}
