// SPDX-License-Identifier: Apache-2.0
//! Per-request GDN recurrent-state slot allocator.
//!
//! A GDN layer keeps one recurrent-state slot per concurrently-resident
//! sequence (see [`crate::gdn_state_layout::GdnStateLayout`]). Unlike the
//! paged KV cache there is no block paging: each active sequence owns exactly
//! one slot for its whole lifetime, and the slot is recycled when the sequence
//! finishes.
//!
//! This is the host-side bookkeeping that the worker drives each step to build
//! the `state_indices` tensor the `gdn_*` kernels consume. It carries no GPU
//! state, so it is unconditional (not feature-gated) and unit-tested on CPU.
//!
//! ## The "degeneration after N requests" hazard
//!
//! Git history (`scratchy: Qwen3-Next GDN state pool slot allocator — fix !!!
//! degeneration after N requests`) records the failure mode this type exists
//! to prevent: once every slot has been used, a *recycled* slot still holds a
//! prior (now-finished) sequence's recurrent state. If the new owner continues
//! from that stale state, output degenerates into garbage. The fix is the
//! `is_fresh` flag below: the first forward of any request — including one
//! that claims a recycled slot — is flagged fresh, and the GDN op MUST
//! zero-initialize the slot's state on a fresh step rather than read it.

use std::collections::HashMap;

/// Maps live request/sequence ids to GDN state slots, recycling on release.
#[derive(Debug)]
pub struct GdnSlotAllocator {
    num_slots: usize,
    /// Free slot ids (LIFO stack).
    free: Vec<u32>,
    /// request_id → owned slot, for the request's lifetime.
    assigned: HashMap<u64, u32>,
}

impl GdnSlotAllocator {
    /// Create an allocator with `num_slots` slots (= `max_num_seqs`).
    pub fn new(num_slots: usize) -> Self {
        // Hand out low ids first (descending stack so pop() yields 0,1,2,…).
        let free = (0..num_slots as u32).rev().collect();
        Self {
            num_slots,
            free,
            assigned: HashMap::new(),
        }
    }

    /// Resolve the state slot for `request_id`.
    ///
    /// Returns `Some((slot, is_fresh))`:
    /// * `is_fresh == true`  → this is the request's **first** forward (or its
    ///   first prefill chunk). The GDN op MUST zero-init the slot's conv +
    ///   recurrent state and ignore whatever stale data a prior, now-released
    ///   owner left there.
    /// * `is_fresh == false` → a continuing forward (decode step / later
    ///   prefill chunk); read and update the slot's existing state.
    ///
    /// Returns `None` when the pool is exhausted — the scheduler must never
    /// admit more GDN sequences than `num_slots`.
    pub fn slot_for(&mut self, request_id: u64) -> Option<(u32, bool)> {
        if let Some(&slot) = self.assigned.get(&request_id) {
            return Some((slot, false));
        }
        let slot = self.free.pop()?;
        self.assigned.insert(request_id, slot);
        Some((slot, true))
    }

    /// Release a finished request's slot back to the free list. The state in
    /// that slot is now stale; the next request to claim it is flagged fresh.
    /// No-op if the request held no slot.
    pub fn release(&mut self, request_id: u64) {
        if let Some(slot) = self.assigned.remove(&request_id) {
            self.free.push(slot);
        }
    }

    /// Total slot capacity (`max_num_seqs`).
    pub fn capacity(&self) -> usize {
        self.num_slots
    }

    /// Number of currently-assigned (live) sequences.
    pub fn num_active(&self) -> usize {
        self.assigned.len()
    }

    /// Whether `request_id` currently owns a slot.
    pub fn is_assigned(&self, request_id: u64) -> bool {
        self.assigned.contains_key(&request_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fresh_then_continue() {
        let mut a = GdnSlotAllocator::new(4);
        // First forward of req 7 → fresh.
        let (s, fresh) = a.slot_for(7).unwrap();
        assert!(fresh, "first forward must be fresh");
        // Subsequent decode steps reuse the same slot, not fresh.
        let (s2, fresh2) = a.slot_for(7).unwrap();
        assert_eq!(s, s2);
        assert!(!fresh2, "continuing forward must not be fresh");
        assert_eq!(a.num_active(), 1);
    }

    #[test]
    fn test_distinct_requests_get_distinct_slots() {
        let mut a = GdnSlotAllocator::new(4);
        let (s0, _) = a.slot_for(100).unwrap();
        let (s1, _) = a.slot_for(200).unwrap();
        let (s2, _) = a.slot_for(300).unwrap();
        assert_ne!(s0, s1);
        assert_ne!(s1, s2);
        assert_ne!(s0, s2);
        assert_eq!(a.num_active(), 3);
    }

    /// Regression for the "degeneration after N requests" bug: a recycled slot
    /// must be flagged fresh for its new owner so stale recurrent state can't
    /// bleed across sequences.
    #[test]
    fn test_recycled_slot_is_fresh_no_degeneration() {
        let mut a = GdnSlotAllocator::new(2);
        let (s_a, fa) = a.slot_for(1).unwrap();
        let (_s_b, fb) = a.slot_for(2).unwrap();
        assert!(fa && fb);
        assert_eq!(a.num_active(), 2);
        // Pool full. Finish req 1, freeing its slot.
        a.release(1);
        assert_eq!(a.num_active(), 1);
        // New req 3 must reuse the freed slot AND be flagged fresh.
        let (s_c, fc) = a.slot_for(3).unwrap();
        assert_eq!(s_c, s_a, "freed slot should be recycled");
        assert!(
            fc,
            "recycled slot MUST be fresh — else stale state degenerates output"
        );
    }

    #[test]
    fn test_exhaustion_returns_none() {
        let mut a = GdnSlotAllocator::new(2);
        assert!(a.slot_for(1).is_some());
        assert!(a.slot_for(2).is_some());
        // Third distinct request with no release → exhausted.
        assert!(a.slot_for(3).is_none());
        // But an already-assigned request still resolves.
        assert!(a.slot_for(1).is_some());
    }

    /// Long churn well past capacity never exhausts as long as active ≤ cap,
    /// and every first-touch is flagged fresh.
    #[test]
    fn test_long_churn_recycles_cleanly() {
        let cap = 3usize;
        let mut a = GdnSlotAllocator::new(cap);
        for round in 0..1000u64 {
            let rid = round; // each round a brand-new request id
            let (_slot, fresh) = a.slot_for(rid).expect("never exhausts at active=1");
            assert!(fresh, "each brand-new request must be fresh");
            // one decode step (not fresh), then it finishes
            let (_s2, fresh2) = a.slot_for(rid).unwrap();
            assert!(!fresh2);
            a.release(rid);
            assert_eq!(a.num_active(), 0);
        }
        // No leak: all slots back in the free list.
        assert_eq!(a.free.len(), cap);
    }

    #[test]
    fn test_release_unknown_is_noop() {
        let mut a = GdnSlotAllocator::new(2);
        a.release(999); // never assigned
        assert_eq!(a.num_active(), 0);
        assert_eq!(a.free.len(), 2);
    }
}
