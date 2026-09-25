// SPDX-License-Identifier: Apache-2.0
//! A counting semaphore bounding in-flight requests, shared fairly between
//! the files contending for it.
//!
//! The cap has to live here rather than in either parallelism axis, because
//! the two multiply: a caller fetching eight shards at once, each split into
//! eight ranges, opens sixty-four connections to one host. Capping chunks
//! alone does not bound that, and capping files alone gives up the
//! single-file case this crate exists for. One budget, shared by every
//! request a [`Client`](crate::Client) makes, bounds the product.
//!
//! Bounding is not enough on its own; *who* gets a freed permit decides what
//! the user sees. Handed to whichever waiter wakes first — or first in line —
//! the file that asked first keeps every permit, because its ranges re-queue
//! as a block, and eight shards download one after another behind eight idle
//! progress bars. So permits go round-robin between [`Flow`]s, one per file:
//! eight files get one connection each, one file gets all eight.

use std::collections::VecDeque;
use std::sync::{Condvar, Mutex};

/// The requests of one file, which share that file's turn at the permits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Flow(pub(crate) u64);

pub(crate) struct Semaphore {
    queue: Mutex<Queue>,
    granted: Condvar,
}

struct Queue {
    /// Permits nobody holds or is owed. Nonzero only while nobody waits:
    /// a release with a waiter hands its permit on instead.
    free: usize,
    /// Flows with a waiter, in the order they are next owed a permit, each
    /// with how many of its requests are waiting. A flow appears at most once.
    waiting: VecDeque<(Flow, usize)>,
    /// Permits handed to a flow that none of its waiters has collected yet.
    owed: Vec<(Flow, usize)>,
}

impl Semaphore {
    pub(crate) fn new(permits: usize) -> Self {
        Self {
            queue: Mutex::new(Queue {
                free: permits.max(1),
                waiting: VecDeque::new(),
                owed: Vec::new(),
            }),
            granted: Condvar::new(),
        }
    }

    /// Block until `flow` is handed a permit. The guard returns it on drop,
    /// including on the error paths and on unwind.
    pub(crate) fn acquire(&self, flow: Flow) -> Permit<'_> {
        let mut queue = self.queue.lock().expect("semaphore poisoned");
        if queue.free > 0 {
            queue.free -= 1;
            return Permit { sem: self };
        }
        match queue.waiting.iter_mut().find(|(f, _)| *f == flow) {
            Some((_, waiters)) => *waiters += 1,
            None => queue.waiting.push_back((flow, 1)),
        }
        loop {
            if let Some(at) = queue.owed.iter().position(|(f, _)| *f == flow) {
                queue.owed[at].1 -= 1;
                if queue.owed[at].1 == 0 {
                    queue.owed.swap_remove(at);
                }
                return Permit { sem: self };
            }
            queue = self.granted.wait(queue).expect("semaphore poisoned");
        }
    }

    fn release(&self) {
        let mut queue = self.queue.lock().expect("semaphore poisoned");
        let Some((flow, waiters)) = queue.waiting.pop_front() else {
            queue.free += 1;
            return;
        };
        // Back of the line: the flow has had its turn.
        if waiters > 1 {
            queue.waiting.push_back((flow, waiters - 1));
        }
        match queue.owed.iter_mut().find(|(f, _)| *f == flow) {
            Some((_, owed)) => *owed += 1,
            None => queue.owed.push((flow, 1)),
        }
        drop(queue);
        self.granted.notify_all();
    }
}

pub(crate) struct Permit<'a> {
    sem: &'a Semaphore,
}

impl Drop for Permit<'_> {
    fn drop(&mut self) {
        self.sem.release();
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
            for t in 0..THREADS {
                let (sem, live, peak) = (sem.clone(), live.clone(), peak.clone());
                s.spawn(move || {
                    let _permit = sem.acquire(Flow(t as u64 % 4));
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
            let _permit = sem.acquire(Flow(0));
            panic!("holder blew up mid-download");
        }));
        // Would block forever if the permit had leaked.
        let _permit = sem.acquire(Flow(0));
    }

    /// The property the progress bars show: a flow that queued all its
    /// requests first does not get served all of them first.
    #[test]
    fn freed_permits_alternate_between_flows_not_arrival_order() {
        const EACH: usize = 3;
        let sem = Semaphore::new(1);
        let order = Mutex::new(Vec::new());

        let held = sem.acquire(Flow(99));
        std::thread::scope(|s| {
            // Every request of flow 0 queues before any request of flow 1,
            // which is exactly how one file's ranges arrive: as a block.
            for flow in [0, 1] {
                for queued in 1..=EACH {
                    let (sem, order) = (&sem, &order);
                    s.spawn(move || {
                        let _permit = sem.acquire(Flow(flow));
                        order.lock().unwrap().push(flow);
                    });
                    wait_until_queued(sem, Flow(flow), queued);
                }
            }
            drop(held);
        });

        assert_eq!(
            order.into_inner().unwrap(),
            [0, 1].repeat(EACH),
            "the first flow to queue kept the permit — its file would finish \
             before the other's bar moved"
        );
    }

    /// Spin until `flow` has `count` requests parked, so the test controls
    /// arrival order instead of the thread scheduler.
    fn wait_until_queued(sem: &Semaphore, flow: Flow, count: usize) {
        let waiting = || {
            let queue = sem.queue.lock().unwrap();
            queue
                .waiting
                .iter()
                .find(|(f, _)| *f == flow)
                .map_or(0, |(_, n)| *n)
        };
        while waiting() < count {
            std::thread::yield_now();
        }
    }
}
