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
//! in the sequence's draft lane ([`scratchy_layers::gdn_state`]); [`GdnSlotAllocator::verified`]
//! records how many drafts were kept, and the request's next step replays its rows up to the last
//! kept one.
//!
//! A sequence holds a lane from the step it drafts in ([`GdnSlotAllocator::hold_lane`], or a
//! verify step that finds it without one) through the next step it runs that does not verify:
//! that step replays the lane's records, so the lane goes back to the free list once the step's
//! inputs are built — a verify step built beside it cannot take the lane it reads, and a lane is
//! first written by a step after the one it is taken in.

use std::collections::HashMap;
use std::collections::hash_map::Entry;

use scratchy_layers::gdn_state::{
    CheckpointRows, DraftLane, DraftLanes, GdnStart, GdnStep, RecordArea,
};

/// Why a step's GDN inputs could not be built.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GdnStepError {
    /// Every slot is owned: the scheduler admitted more concurrent sequences than `max_num_seqs`.
    Exhausted { capacity: usize },
    /// A verify step carries more drafts than a lane keeps checkpoints for.
    Drafts {
        drafts: usize,
        checkpoint_rows: CheckpointRows,
    },
    /// Every draft lane is held: more sequences verify than the pool keeps draft state for.
    Lanes { capacity: u16 },
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
                "verify step carries {drafts} drafts but each GDN draft lane keeps {} checkpoints",
                checkpoint_rows.0
            ),
            Self::Lanes { capacity } => write!(
                f,
                "GDN draft lanes exhausted (capacity {capacity}): more sequences verify drafts \
                 than the pool keeps draft state for"
            ),
        }
    }
}

/// Maps live request/sequence ids to GDN state slots, recycling on release.
#[derive(Debug)]
pub struct GdnSlotAllocator {
    num_slots: usize,
    /// The state pool's draft lanes; `None` when the model drafts nothing.
    lanes: Option<DraftLanes>,
    /// Free slot ids (LIFO stack).
    free: Vec<u32>,
    /// Free draft lanes (LIFO stack).
    free_lanes: Vec<DraftLane>,
    /// request_id → owned slot and its state, for the request's lifetime.
    assigned: HashMap<u64, (u32, SlotState)>,
}

/// Where a slot's next step starts, and the sequence's draft lane with its current record area.
#[derive(Clone, Copy, Debug)]
struct SlotState {
    start: GdnStart,
    records: RecordArea,
    lane: Option<DraftLane>,
}

impl GdnSlotAllocator {
    /// Create an allocator with `num_slots` slots (= `max_num_seqs`) over a state pool keeping
    /// `lanes` draft lanes.
    pub fn new(num_slots: usize, lanes: Option<DraftLanes>) -> Self {
        // Hand out low ids first (descending stack so pop() yields 0,1,2,…).
        let free = (0..num_slots as u32).rev().collect();
        let free_lanes = lanes.into_iter().flat_map(|l| l.lanes().rev()).collect();
        Self {
            num_slots,
            lanes,
            free,
            free_lanes,
            assigned: HashMap::new(),
        }
    }

    /// One step's GDN inputs, for its sequences in `cu_seqlens` order as `(request_id, drafts)`:
    /// each sequence's state slot (`state_indices`) and its [`GdnStep`] code. A verify step
    /// (`drafts > 0`) keeps what a rejected draft needs undone (`gdn_state` module docs) in the
    /// sequence's draft lane; a sequence whose step does not verify gives its lane back, once every
    /// sequence's inputs are built (module docs).
    pub fn step(
        &mut self,
        seqs: impl IntoIterator<Item = (u64, usize)>,
    ) -> Result<(Vec<i32>, Vec<u32>), GdnStepError> {
        let mut released = Vec::new();
        let built = self.build_step(seqs, &mut released);
        self.free_lanes.extend(released);
        built
    }

    /// [`Self::step`]'s inputs, the lanes its sequences give back in `released`.
    fn build_step(
        &mut self,
        seqs: impl IntoIterator<Item = (u64, usize)>,
        released: &mut Vec<DraftLane>,
    ) -> Result<(Vec<i32>, Vec<u32>), GdnStepError> {
        let (mut entries, mut steps) = (Vec::new(), Vec::new());
        let capacity = self.num_slots;
        let lane_rows = self.lanes.map_or(CheckpointRows::NONE, DraftLanes::rows);
        let lane_capacity = self.lanes.map_or(0, DraftLanes::count);
        for (request_id, drafts) in seqs {
            let checkpoint_rows = u8::try_from(drafts)
                .map(CheckpointRows)
                .ok()
                .filter(|&rows| rows <= lane_rows)
                .ok_or(GdnStepError::Drafts {
                    drafts,
                    checkpoint_rows: lane_rows,
                })?;
            let free_lanes = &mut self.free_lanes;
            let (slot, state) = (Self::slot_for(&mut self.free, &mut self.assigned, request_id))
                .ok_or(GdnStepError::Exhausted { capacity })?;
            entries.push(i32::try_from(slot).expect("GDN state slot fits the kernels' i32"));
            let start = std::mem::replace(&mut state.start, GdnStart::Slot);
            if drafts > 0 && state.lane.is_none() {
                state.lane = Some(Self::take_lane(free_lanes, &mut state.records).ok_or(
                    GdnStepError::Lanes {
                        capacity: lane_capacity,
                    },
                )?);
            }
            let replays = matches!(start, GdnStart::Checkpoint(_));
            let lane = match (drafts > 0 || replays, state.lane) {
                (false, _) => None,
                (true, Some(lane)) => Some(lane),
                // `verified` starts a sequence from a checkpoint only after a verify step, which
                // held its lane through this step.
                (true, None) => unreachable!("request {request_id} replays without its lane"),
            };
            let step = GdnStep {
                start,
                checkpoint_rows,
                records: state.records,
                lane,
            };
            state.records = step.after();
            if drafts == 0 {
                released.extend(state.lane.take());
            }
            steps.push(step.encode());
        }
        Ok((entries, steps))
    }

    /// Hold a draft lane for `request_id`, which drafts in the step being built: `false` when it
    /// owns no slot or every lane is held — it does not draft. A sequence that holds one keeps it.
    pub fn hold_lane(&mut self, request_id: u64) -> bool {
        let Some((_, state)) = self.assigned.get_mut(&request_id) else {
            return false;
        };
        if state.lane.is_none() {
            state.lane = Self::take_lane(&mut self.free_lanes, &mut state.records);
        }
        state.lane.is_some()
    }

    /// A free lane, its records starting from the first area (no verify step wrote it for this
    /// sequence yet).
    fn take_lane(free: &mut Vec<DraftLane>, records: &mut RecordArea) -> Option<DraftLane> {
        let lane = free.pop()?;
        *records = RecordArea::First;
        Some(lane)
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
    fn slot_for<'a>(
        free: &mut Vec<u32>,
        assigned: &'a mut HashMap<u64, (u32, SlotState)>,
        request_id: u64,
    ) -> Option<(u32, &'a mut SlotState)> {
        let (slot, state) = match assigned.entry(request_id) {
            Entry::Occupied(owned) => owned.into_mut(),
            Entry::Vacant(new) => {
                let fresh = SlotState {
                    start: GdnStart::Fresh,
                    records: RecordArea::First,
                    lane: None,
                };
                new.insert((free.pop()?, fresh))
            }
        };
        Some((*slot, state))
    }

    /// Release a finished request's slot, and any draft lane it holds, back to the free lists. The
    /// state in that slot is now stale; the next request to claim it is flagged fresh. No-op if
    /// the request held no slot.
    pub fn release(&mut self, request_id: u64) {
        if let Some((slot, state)) = self.assigned.remove(&request_id) {
            self.free.push(slot);
            self.free_lanes.extend(state.lane);
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
        let (slot, state) = GdnSlotAllocator::slot_for(&mut a.free, &mut a.assigned, request_id)?;
        Some((slot, std::mem::replace(&mut state.start, GdnStart::Slot)))
    }

    #[test]
    fn test_fresh_then_continue() {
        let mut a = GdnSlotAllocator::new(4, None);
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
        let mut a = GdnSlotAllocator::new(4, None);
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
        let mut a = GdnSlotAllocator::new(2, None);
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
        let mut a = GdnSlotAllocator::new(2, None);
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
        let mut a = GdnSlotAllocator::new(cap, None);
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

    /// The code of a step.
    fn code(
        start: GdnStart,
        checkpoint_rows: CheckpointRows,
        records: RecordArea,
        lane: Option<DraftLane>,
    ) -> u32 {
        GdnStep {
            start,
            checkpoint_rows,
            records,
            lane,
        }
        .encode()
    }

    /// A verify step makes the request's next step — and only that one — replay its rows through
    /// the last kept one, whether it kept every draft or none: the slot holds the state the verify
    /// step started from. A verify step moves the records to the other area of its lane; a step
    /// that verifies nothing leaves them, and gives the lane back.
    #[test]
    fn test_verify_resumes_from_the_kept_row() {
        use RecordArea::{First as R0, Second as R1};
        let rows = CheckpointRows(3);
        let lanes = DraftLanes::new(rows, 2);
        let lane = lanes.and_then(|l| l.lanes().next());
        let mut a = GdnSlotAllocator::new(2, lanes);
        let none = CheckpointRows::NONE;
        let fresh = code(GdnStart::Fresh, none, R0, None);
        assert_eq!(a.step([(5, 0), (6, 0)]), Ok((vec![0, 1], vec![fresh; 2])));
        let ok = |start, records| Ok((vec![0], vec![code(start, rows, records, lane)]));
        assert_eq!(a.step([(5, 3)]), ok(GdnStart::Slot, R0));
        a.verified(5, 1, 3);
        // Replay R1's rows 0..=1, record to R0.
        assert_eq!(a.step([(5, 3)]), ok(GdnStart::Checkpoint(1), R1));
        a.verified(5, 3, 3);
        // Every draft kept: still a replay, of all four rows.
        assert_eq!(a.step([(5, 3)]), ok(GdnStart::Checkpoint(3), R0));
        a.verified(5, 0, 3);
        // A decode after a verify step replays and updates the slot in place.
        let decode = code(GdnStart::Checkpoint(0), none, R1, lane);
        assert_eq!(a.step([(5, 0)]), Ok((vec![0], vec![decode])));
        let slot = code(GdnStart::Slot, none, R1, None);
        assert_eq!(a.step([(5, 0)]), Ok((vec![0], vec![slot])));
        assert_eq!(a.free_lanes.len(), 2, "the replay gave the lane back");

        assert_eq!(
            a.step([(6, 4)]),
            Err(GdnStepError::Drafts {
                drafts: 4,
                checkpoint_rows: rows
            })
        );
    }

    /// A lane a step's replay reads is free only for the steps after it: a verify step built beside
    /// that replay takes another lane, or none — the step then cannot be built.
    #[test]
    fn a_replayed_lane_is_free_after_its_step_not_in_it() {
        let rows = CheckpointRows(2);
        let lanes = DraftLanes::new(rows, 1);
        let lane = lanes.and_then(|l| l.lanes().next());
        let mut a = GdnSlotAllocator::new(3, lanes);
        assert!(a.step([(1, 0), (2, 0)]).is_ok());
        assert!(a.step([(1, 2)]).is_ok());
        a.verified(1, 2, 2);
        // 1 replays its one lane while 2 would verify: no lane is left for 2.
        assert_eq!(
            a.step([(1, 0), (2, 2)]),
            Err(GdnStepError::Lanes { capacity: 1 })
        );
        let mut a = GdnSlotAllocator::new(3, lanes);
        assert!(a.step([(1, 0), (2, 0)]).is_ok());
        assert!(a.step([(1, 2)]).is_ok());
        a.verified(1, 2, 2);
        let replay = code(
            GdnStart::Checkpoint(2),
            CheckpointRows::NONE,
            RecordArea::Second,
            lane,
        );
        let plain = code(
            GdnStart::Slot,
            CheckpointRows::NONE,
            RecordArea::First,
            None,
        );
        assert_eq!(
            a.step([(1, 0), (2, 0)]),
            Ok((vec![0, 1], vec![replay, plain]))
        );
        // The step after it: 2 verifies in the lane 1 replayed.
        let verify = code(GdnStart::Slot, rows, RecordArea::First, lane);
        assert_eq!(a.step([(2, 2)]), Ok((vec![1], vec![verify])));
    }

    /// A sequence holds a lane from the step it drafts in, across steps it does not run, until a
    /// step of its own that verifies nothing; drafting finds no lane when every one is held, and a
    /// released request gives its lane back.
    #[test]
    fn a_drafting_sequence_holds_its_lane_until_a_step_without_drafts() {
        let rows = CheckpointRows(2);
        let lanes = DraftLanes::new(rows, 2);
        let mut a = GdnSlotAllocator::new(4, lanes);
        assert!(a.step([(1, 0), (2, 0), (3, 0)]).is_ok());
        assert!(!a.hold_lane(9), "a request without a slot drafts nothing");
        assert!(a.hold_lane(1) && a.hold_lane(1), "a held lane stays held");
        assert!(a.hold_lane(2));
        assert!(!a.hold_lane(3), "both lanes are held");
        // 1 is skipped for a step; 2 runs one without its drafts and gives its lane back.
        assert!(a.step([(2, 0)]).is_ok());
        assert!(a.hold_lane(3));
        let first = lanes.and_then(|l| l.lanes().next());
        let verify = code(GdnStart::Slot, rows, RecordArea::First, first);
        assert_eq!(a.step([(1, 2)]), Ok((vec![0], vec![verify])));
        a.release(1);
        assert_eq!(a.free_lanes, Vec::from_iter(first));
    }

    #[test]
    fn test_release_unknown_is_noop() {
        let mut a = GdnSlotAllocator::new(2, None);
        a.release(999); // never assigned
        assert_eq!(a.num_active(), 0);
        assert_eq!(a.free.len(), 2);
    }
}
