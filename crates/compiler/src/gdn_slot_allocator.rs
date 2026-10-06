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
//!
//! ## Speculative decoding
//!
//! A verify step runs a sequence's drafts through its recurrent state, so a rejected draft would
//! stay in it. The verify step leaves the state it started from in the slot and records its rows
//! ([`scratchy_layers::gdn_state`]); [`GdnSlotAllocator::verified`] records how many drafts were
//! kept, and the request's next step replays its rows up to the last kept one.

use std::collections::HashMap;

use scratchy_layers::gdn_state::{CheckpointRows, GdnStart, GdnStep, RecordArea};

/// Why a step's GDN inputs could not be built.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GdnStepError {
    /// Every slot is owned: the scheduler admitted more concurrent sequences than `max_num_seqs`.
    Exhausted { capacity: usize },
    /// A verify step carries more drafts than a slot keeps checkpoints for.
    Drafts {
        drafts: usize,
        checkpoint_rows: CheckpointRows,
    },
}

impl std::fmt::Display for GdnStepError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Exhausted { capacity } => write!(
                f,
                "GDN state-slot pool exhausted (capacity {capacity}): scheduler admitted more \
                 concurrent sequences than max_num_seqs"
            ),
            Self::Drafts {
                drafts,
                checkpoint_rows,
            } => write!(
                f,
                "verify step carries {drafts} drafts but each GDN slot keeps {} checkpoints",
                checkpoint_rows.0
            ),
        }
    }
}

/// Maps live request/sequence ids to GDN state slots, recycling on release.
#[derive(Debug)]
pub struct GdnSlotAllocator {
    num_slots: usize,
    /// The state pool's: the most drafts a slot keeps (its layout).
    checkpoint_rows: CheckpointRows,
    /// Free slot ids (LIFO stack).
    free: Vec<u32>,
    /// request_id → owned slot and its state, for the request's lifetime.
    assigned: HashMap<u64, (u32, SlotState)>,
}

/// Where a slot's next step starts, and which of its record areas is current.
#[derive(Clone, Copy, Debug)]
struct SlotState {
    start: GdnStart,
    records: RecordArea,
}

impl GdnSlotAllocator {
    /// Create an allocator with `num_slots` slots (= `max_num_seqs`) over a state pool keeping
    /// `checkpoint_rows` checkpoints per slot.
    pub fn new(num_slots: usize, checkpoint_rows: CheckpointRows) -> Self {
        // Hand out low ids first (descending stack so pop() yields 0,1,2,…).
        let free = (0..num_slots as u32).rev().collect();
        Self {
            num_slots,
            checkpoint_rows,
            free,
            assigned: HashMap::new(),
        }
    }

    /// One step's GDN inputs, for its sequences in `cu_seqlens` order as `(request_id, drafts)`:
    /// each sequence's state slot (`state_indices`) and its [`GdnStep`] code. A verify step
    /// (`drafts > 0`) keeps what a rejected draft needs undone (`gdn_state` module docs).
    pub fn step(
        &mut self,
        seqs: impl IntoIterator<Item = (u64, usize)>,
    ) -> Result<(Vec<i32>, Vec<u32>), GdnStepError> {
        let (mut entries, mut steps) = (Vec::new(), Vec::new());
        let (capacity, slot_rows) = (self.num_slots, self.checkpoint_rows);
        for (request_id, drafts) in seqs {
            let checkpoint_rows = u8::try_from(drafts)
                .map(CheckpointRows)
                .ok()
                .filter(|&rows| rows <= slot_rows)
                .ok_or(GdnStepError::Drafts {
                    drafts,
                    checkpoint_rows: slot_rows,
                })?;
            let (slot, state) =
                (self.slot_for(request_id)).ok_or(GdnStepError::Exhausted { capacity })?;
            entries.push(i32::try_from(slot).expect("GDN state slot fits the kernels' i32"));
            let step = GdnStep {
                start: std::mem::replace(&mut state.start, GdnStart::Slot),
                checkpoint_rows,
                records: state.records,
            };
            state.records = step.after();
            steps.push(step.encode());
        }
        Ok((entries, steps))
    }

    /// A verify step kept `accepted` of `request_id`'s `drafts`. The slot still holds the state
    /// the step started from, so the next step replays its rows through the last kept one —
    /// row `accepted` (row 0 is the sequence's own token), every draft kept or not.
    pub fn verified(&mut self, request_id: u64, accepted: usize, drafts: usize) {
        debug_assert!(accepted <= drafts, "{accepted} of {drafts} drafts kept");
        if let Some((_, state)) = self.assigned.get_mut(&request_id) {
            let row = u8::try_from(accepted).expect("accepted ≤ drafts ≤ CheckpointRows (u8)");
            state.start = GdnStart::Checkpoint(row);
        }
    }

    /// Resolve the state slot for `request_id` and where its step starts:
    /// * [`GdnStart::Fresh`] → the request's **first** forward (or its first prefill chunk). The
    ///   GDN op MUST zero-init the slot's conv + recurrent state and ignore whatever stale data a
    ///   prior, now-released owner left there.
    /// * [`GdnStart::Slot`] → a continuing forward (decode step / later prefill chunk); read and
    ///   update the slot's existing state.
    /// * [`GdnStart::Checkpoint`] → the step after a verify step ([`Self::verified`]).
    ///
    /// Returns `None` when the pool is exhausted — the scheduler must never
    /// admit more GDN sequences than `num_slots`.
    fn slot_for(&mut self, request_id: u64) -> Option<(u32, &mut SlotState)> {
        if !self.assigned.contains_key(&request_id) {
            let slot = self.free.pop()?;
            let fresh = SlotState {
                start: GdnStart::Fresh,
                records: RecordArea::First,
            };
            self.assigned.insert(request_id, (slot, fresh));
        }
        let (slot, state) = self.assigned.get_mut(&request_id)?;
        Some((*slot, state))
    }

    /// Release a finished request's slot back to the free list. The state in
    /// that slot is now stale; the next request to claim it is flagged fresh.
    /// No-op if the request held no slot.
    pub fn release(&mut self, request_id: u64) {
        if let Some((slot, _)) = self.assigned.remove(&request_id) {
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

    /// The slot `request_id` steps in and where it starts, as `step` takes them.
    fn take_start(a: &mut GdnSlotAllocator, request_id: u64) -> Option<(u32, GdnStart)> {
        let (slot, state) = a.slot_for(request_id)?;
        Some((slot, std::mem::replace(&mut state.start, GdnStart::Slot)))
    }

    #[test]
    fn test_fresh_then_continue() {
        let mut a = GdnSlotAllocator::new(4, CheckpointRows::NONE);
        // First forward of req 7 → fresh.
        let (s, fresh) = take_start(&mut a, 7).unwrap();
        assert_eq!(fresh, GdnStart::Fresh, "first forward must be fresh");
        // Subsequent decode steps reuse the same slot, not fresh.
        let (s2, fresh2) = take_start(&mut a, 7).unwrap();
        assert_eq!(s, s2);
        assert_eq!(
            fresh2,
            GdnStart::Slot,
            "continuing forward must not be fresh"
        );
        assert_eq!(a.num_active(), 1);
    }

    #[test]
    fn test_distinct_requests_get_distinct_slots() {
        let mut a = GdnSlotAllocator::new(4, CheckpointRows::NONE);
        let (s0, _) = take_start(&mut a, 100).unwrap();
        let (s1, _) = take_start(&mut a, 200).unwrap();
        let (s2, _) = take_start(&mut a, 300).unwrap();
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
        let mut a = GdnSlotAllocator::new(2, CheckpointRows::NONE);
        let (s_a, fa) = take_start(&mut a, 1).unwrap();
        let (_s_b, fb) = take_start(&mut a, 2).unwrap();
        assert_eq!((fa, fb), (GdnStart::Fresh, GdnStart::Fresh));
        assert_eq!(a.num_active(), 2);
        // Pool full. Finish req 1, freeing its slot.
        a.release(1);
        assert_eq!(a.num_active(), 1);
        // New req 3 must reuse the freed slot AND be flagged fresh.
        let (s_c, fc) = take_start(&mut a, 3).unwrap();
        assert_eq!(s_c, s_a, "freed slot should be recycled");
        assert_eq!(
            fc,
            GdnStart::Fresh,
            "recycled slot MUST be fresh — else stale state degenerates output"
        );
    }

    #[test]
    fn test_exhaustion_returns_none() {
        let mut a = GdnSlotAllocator::new(2, CheckpointRows::NONE);
        assert!(take_start(&mut a, 1).is_some());
        assert!(take_start(&mut a, 2).is_some());
        // Third distinct request with no release → exhausted.
        assert!(take_start(&mut a, 3).is_none());
        // But an already-assigned request still resolves.
        assert!(take_start(&mut a, 1).is_some());
    }

    /// Long churn well past capacity never exhausts as long as active ≤ cap,
    /// and every first-touch is flagged fresh.
    #[test]
    fn test_long_churn_recycles_cleanly() {
        let cap = 3usize;
        let mut a = GdnSlotAllocator::new(cap, CheckpointRows::NONE);
        for round in 0..1000u64 {
            let rid = round; // each round a brand-new request id
            let (_slot, fresh) = take_start(&mut a, rid).expect("never exhausts at active=1");
            assert_eq!(
                fresh,
                GdnStart::Fresh,
                "each brand-new request must be fresh"
            );
            // one decode step (not fresh), then it finishes
            let (_s2, fresh2) = take_start(&mut a, rid).unwrap();
            assert_eq!(fresh2, GdnStart::Slot);
            a.release(rid);
            assert_eq!(a.num_active(), 0);
        }
        // No leak: all slots back in the free list.
        assert_eq!(a.free.len(), cap);
    }

    /// A verify step makes the request's next step — and only that one — replay its rows through
    /// the last kept one, whether it kept every draft or none: the slot holds the state the verify
    /// step started from. A verify step moves the records to the other area; a step that verifies
    /// nothing leaves them. Slots, not entries, index the state.
    #[test]
    fn test_verify_resumes_from_the_kept_row() {
        use RecordArea::{First as R0, Second as R1};
        let rows = CheckpointRows(3);
        let mut a = GdnSlotAllocator::new(2, rows);
        let code = |start, checkpoint_rows, records| {
            GdnStep {
                start,
                checkpoint_rows,
                records,
            }
            .encode()
        };
        let none = CheckpointRows::NONE;
        let fresh = code(GdnStart::Fresh, none, R0);
        assert_eq!(a.step([(5, 0), (6, 0)]), Ok((vec![0, 1], vec![fresh; 2])));
        let ok = |start, records| Ok((vec![0], vec![code(start, rows, records)]));
        assert_eq!(a.step([(5, 3)]), ok(GdnStart::Slot, R0));
        a.verified(5, 1, 3);
        // Replay R1's rows 0..=1, record to R0.
        assert_eq!(a.step([(5, 3)]), ok(GdnStart::Checkpoint(1), R1));
        a.verified(5, 3, 3);
        // Every draft kept: still a replay, of all four rows.
        assert_eq!(a.step([(5, 3)]), ok(GdnStart::Checkpoint(3), R0));
        a.verified(5, 0, 3);
        // A decode after a verify step replays and updates the slot in place.
        let decode = code(GdnStart::Checkpoint(0), none, R1);
        assert_eq!(a.step([(5, 0)]), Ok((vec![0], vec![decode])));
        let slot = code(GdnStart::Slot, none, R1);
        assert_eq!(a.step([(5, 0)]), Ok((vec![0], vec![slot])));

        assert_eq!(
            a.step([(6, 4)]),
            Err(GdnStepError::Drafts {
                drafts: 4,
                checkpoint_rows: rows
            })
        );
    }

    #[test]
    fn test_release_unknown_is_noop() {
        let mut a = GdnSlotAllocator::new(2, CheckpointRows::NONE);
        a.release(999); // never assigned
        assert_eq!(a.num_active(), 0);
        assert_eq!(a.free.len(), 2);
    }
}
