// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Request queue implementations, ported from
//! `vllm/v1/core/sched/request_queue.py`.
//!
//! Provides FCFS (first-come-first-served) and priority-based queues for
//! managing waiting requests in the scheduler.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, VecDeque};

use scratchy_core_common::Request;

// ---------------------------------------------------------------------------
// SchedulingPolicy
// ---------------------------------------------------------------------------

/// Scheduling policy for request ordering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulingPolicy {
    /// First come, first served -- requests are handled in arrival order.
    Fcfs,
    /// Priority-based -- requests are handled by priority value (lower =
    /// higher scheduling priority), with ties broken by arrival time.
    Priority,
}

// ---------------------------------------------------------------------------
// RequestQueue trait
// ---------------------------------------------------------------------------

/// Abstract interface for request queues.
///
/// Ported from the Python `RequestQueue(ABC)`.
pub trait RequestQueue: Send {
    /// Add a request to the queue according to the policy.
    fn add_request(&mut self, request: Request);

    /// Pop a request from the queue according to the policy.
    /// Returns `None` if the queue is empty.
    fn pop_request(&mut self) -> Option<Request>;

    /// Peek at the next request without removing it.
    /// Returns `None` if the queue is empty.
    fn peek_request(&self) -> Option<&Request>;

    /// Prepend a request to the front of the queue.
    /// For priority queues, this is equivalent to `add_request` since
    /// ordering is determined by the priority comparator.
    fn prepend_request(&mut self, request: Request);

    /// Remove a specific request by its ID.
    /// Returns `true` if the request was found and removed.
    fn remove_request(&mut self, request_id: &str) -> bool;

    /// Number of requests in the queue.
    fn len(&self) -> usize;

    /// Whether the queue is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Iterate over the queue in policy order.
    /// The iterator yields shared references; the queue is not modified.
    fn iter(&self) -> Box<dyn Iterator<Item = &Request> + '_>;

    /// Drain all requests from the queue, returning them in policy order.
    fn drain_all(&mut self) -> Vec<Request>;
}

// ---------------------------------------------------------------------------
// FCFSRequestQueue
// ---------------------------------------------------------------------------

/// A first-come-first-served request queue backed by a `VecDeque`.
///
/// Requests are appended to the back and popped from the front.
#[derive(Debug, Clone)]
pub struct FCFSRequestQueue {
    inner: VecDeque<Request>,
}

impl FCFSRequestQueue {
    /// Create a new empty FCFS queue.
    pub fn new() -> Self {
        Self {
            inner: VecDeque::new(),
        }
    }
}

impl Default for FCFSRequestQueue {
    fn default() -> Self {
        Self::new()
    }
}

impl RequestQueue for FCFSRequestQueue {
    fn add_request(&mut self, request: Request) {
        self.inner.push_back(request);
    }

    fn pop_request(&mut self) -> Option<Request> {
        self.inner.pop_front()
    }

    fn peek_request(&self) -> Option<&Request> {
        self.inner.front()
    }

    fn prepend_request(&mut self, request: Request) {
        self.inner.push_front(request);
    }

    fn remove_request(&mut self, request_id: &str) -> bool {
        if let Some(pos) = self.inner.iter().position(|r| r.request_id == request_id) {
            self.inner.remove(pos);
            true
        } else {
            false
        }
    }

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn iter(&self) -> Box<dyn Iterator<Item = &Request> + '_> {
        Box::new(self.inner.iter())
    }

    fn drain_all(&mut self) -> Vec<Request> {
        self.inner.drain(..).collect()
    }
}

// ---------------------------------------------------------------------------
// PriorityRequestQueue
// ---------------------------------------------------------------------------

/// A priority-based request queue backed by a `BinaryHeap`.
///
/// Uses `Reverse<Request>` so that lower priority values (higher scheduling
/// priority) are popped first. The `Request` type implements `Ord` by
/// `(priority, arrival_time, request_id)`.
#[derive(Debug, Clone)]
pub struct PriorityRequestQueue {
    heap: BinaryHeap<Reverse<Request>>,
}

impl PriorityRequestQueue {
    /// Create a new empty priority queue.
    pub fn new() -> Self {
        Self {
            heap: BinaryHeap::new(),
        }
    }
}

impl Default for PriorityRequestQueue {
    fn default() -> Self {
        Self::new()
    }
}

impl RequestQueue for PriorityRequestQueue {
    fn add_request(&mut self, request: Request) {
        self.heap.push(Reverse(request));
    }

    fn pop_request(&mut self) -> Option<Request> {
        self.heap.pop().map(|Reverse(r)| r)
    }

    fn peek_request(&self) -> Option<&Request> {
        self.heap.peek().map(|Reverse(r)| r)
    }

    fn prepend_request(&mut self, request: Request) {
        // In a priority queue, prepend is the same as add -- ordering is
        // determined by the comparator.
        self.heap.push(Reverse(request));
    }

    fn remove_request(&mut self, request_id: &str) -> bool {
        let original_len = self.heap.len();
        let items: Vec<_> = self
            .heap
            .drain()
            .filter(|Reverse(r)| r.request_id != request_id)
            .collect();
        let removed = items.len() < original_len;
        self.heap = BinaryHeap::from(items);
        removed
    }

    fn len(&self) -> usize {
        self.heap.len()
    }

    fn iter(&self) -> Box<dyn Iterator<Item = &Request> + '_> {
        Box::new(self.heap.iter().map(|Reverse(r)| r))
    }

    fn drain_all(&mut self) -> Vec<Request> {
        let mut result = Vec::with_capacity(self.heap.len());
        while let Some(Reverse(r)) = self.heap.pop() {
            result.push(r);
        }
        result
    }
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

/// Create a request queue for the given scheduling policy.
pub fn create_request_queue(policy: SchedulingPolicy) -> Box<dyn RequestQueue> {
    match policy {
        SchedulingPolicy::Fcfs => Box::new(FCFSRequestQueue::new()),
        SchedulingPolicy::Priority => Box::new(PriorityRequestQueue::new()),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scratchy_core_common::SamplingParams;

    fn make_request(id: &str, priority: i32, arrival: f64) -> Request {
        Request::new(
            id.into(),
            vec![1, 2, 3],
            SamplingParams::default(),
            arrival,
            0,
            priority,
            None,
        )
    }

    // -- FCFS queue tests --

    #[test]
    fn test_fcfs_add_and_pop() {
        let mut q = FCFSRequestQueue::new();
        assert!(q.is_empty());
        assert_eq!(q.len(), 0);

        q.add_request(make_request("r1", 0, 1.0));
        q.add_request(make_request("r2", 0, 2.0));
        q.add_request(make_request("r3", 0, 3.0));

        assert_eq!(q.len(), 3);
        assert!(!q.is_empty());

        let r = q.pop_request().unwrap();
        assert_eq!(r.request_id, "r1");
        let r = q.pop_request().unwrap();
        assert_eq!(r.request_id, "r2");
        let r = q.pop_request().unwrap();
        assert_eq!(r.request_id, "r3");
        assert!(q.pop_request().is_none());
    }

    #[test]
    fn test_fcfs_peek() {
        let mut q = FCFSRequestQueue::new();
        assert!(q.peek_request().is_none());

        q.add_request(make_request("r1", 0, 1.0));
        q.add_request(make_request("r2", 0, 2.0));

        assert_eq!(q.peek_request().unwrap().request_id, "r1");
        // Peek does not remove.
        assert_eq!(q.len(), 2);
    }

    #[test]
    fn test_fcfs_prepend() {
        let mut q = FCFSRequestQueue::new();
        q.add_request(make_request("r1", 0, 1.0));
        q.prepend_request(make_request("r0", 0, 0.5));

        let r = q.pop_request().unwrap();
        assert_eq!(r.request_id, "r0");
    }

    #[test]
    fn test_fcfs_remove() {
        let mut q = FCFSRequestQueue::new();
        q.add_request(make_request("r1", 0, 1.0));
        q.add_request(make_request("r2", 0, 2.0));
        q.add_request(make_request("r3", 0, 3.0));

        assert!(q.remove_request("r2"));
        assert_eq!(q.len(), 2);
        assert!(!q.remove_request("r2")); // Already removed.

        let r = q.pop_request().unwrap();
        assert_eq!(r.request_id, "r1");
        let r = q.pop_request().unwrap();
        assert_eq!(r.request_id, "r3");
    }

    #[test]
    fn test_fcfs_iter() {
        let mut q = FCFSRequestQueue::new();
        q.add_request(make_request("r1", 0, 1.0));
        q.add_request(make_request("r2", 0, 2.0));

        let ids: Vec<&str> = q.iter().map(|r| r.request_id.as_str()).collect();
        assert_eq!(ids, vec!["r1", "r2"]);
    }

    #[test]
    fn test_fcfs_drain_all() {
        let mut q = FCFSRequestQueue::new();
        q.add_request(make_request("r1", 0, 1.0));
        q.add_request(make_request("r2", 0, 2.0));

        let drained = q.drain_all();
        assert_eq!(drained.len(), 2);
        assert_eq!(drained[0].request_id, "r1");
        assert_eq!(drained[1].request_id, "r2");
        assert!(q.is_empty());
    }

    // -- Priority queue tests --

    #[test]
    fn test_priority_add_and_pop() {
        let mut q = PriorityRequestQueue::new();
        assert!(q.is_empty());

        // Add in non-priority order.
        q.add_request(make_request("r_low", 10, 1.0));
        q.add_request(make_request("r_high", -1, 1.0));
        q.add_request(make_request("r_mid", 5, 1.0));

        assert_eq!(q.len(), 3);

        // Should pop in priority order: r_high (-1), r_mid (5), r_low (10).
        let r = q.pop_request().unwrap();
        assert_eq!(r.request_id, "r_high");
        let r = q.pop_request().unwrap();
        assert_eq!(r.request_id, "r_mid");
        let r = q.pop_request().unwrap();
        assert_eq!(r.request_id, "r_low");
        assert!(q.pop_request().is_none());
    }

    #[test]
    fn test_priority_tiebreak_by_arrival() {
        let mut q = PriorityRequestQueue::new();
        q.add_request(make_request("r_late", 0, 2.0));
        q.add_request(make_request("r_early", 0, 1.0));

        let r = q.pop_request().unwrap();
        assert_eq!(r.request_id, "r_early");
        let r = q.pop_request().unwrap();
        assert_eq!(r.request_id, "r_late");
    }

    #[test]
    fn test_priority_peek() {
        let mut q = PriorityRequestQueue::new();
        q.add_request(make_request("r_low", 10, 1.0));
        q.add_request(make_request("r_high", -1, 1.0));

        assert_eq!(q.peek_request().unwrap().request_id, "r_high");
        assert_eq!(q.len(), 2); // Peek does not remove.
    }

    #[test]
    fn test_priority_prepend_is_add() {
        let mut q = PriorityRequestQueue::new();
        q.add_request(make_request("r_high", -1, 1.0));
        q.prepend_request(make_request("r_low", 10, 1.0));

        // Should still pop r_high first since priority determines order.
        let r = q.pop_request().unwrap();
        assert_eq!(r.request_id, "r_high");
    }

    #[test]
    fn test_priority_remove() {
        let mut q = PriorityRequestQueue::new();
        q.add_request(make_request("r1", 0, 1.0));
        q.add_request(make_request("r2", 0, 2.0));
        q.add_request(make_request("r3", 0, 3.0));

        assert!(q.remove_request("r2"));
        assert_eq!(q.len(), 2);
        assert!(!q.remove_request("r2"));

        let r = q.pop_request().unwrap();
        assert_eq!(r.request_id, "r1");
        let r = q.pop_request().unwrap();
        assert_eq!(r.request_id, "r3");
    }

    #[test]
    fn test_priority_drain_all() {
        let mut q = PriorityRequestQueue::new();
        q.add_request(make_request("r_low", 10, 1.0));
        q.add_request(make_request("r_high", -1, 1.0));

        let drained = q.drain_all();
        assert_eq!(drained.len(), 2);
        // drain_all returns in priority order.
        assert_eq!(drained[0].request_id, "r_high");
        assert_eq!(drained[1].request_id, "r_low");
        assert!(q.is_empty());
    }

    // -- Factory tests --

    #[test]
    fn test_create_fcfs_queue() {
        let mut q = create_request_queue(SchedulingPolicy::Fcfs);
        q.add_request(make_request("r2", 0, 2.0));
        q.add_request(make_request("r1", 0, 1.0));

        // FCFS: pop in insertion order.
        assert_eq!(q.pop_request().unwrap().request_id, "r2");
        assert_eq!(q.pop_request().unwrap().request_id, "r1");
    }

    #[test]
    fn test_create_priority_queue() {
        let mut q = create_request_queue(SchedulingPolicy::Priority);
        q.add_request(make_request("r_low", 10, 1.0));
        q.add_request(make_request("r_high", -1, 1.0));

        // Priority: pop by priority.
        assert_eq!(q.pop_request().unwrap().request_id, "r_high");
        assert_eq!(q.pop_request().unwrap().request_id, "r_low");
    }
}
