// SPDX-License-Identifier: Apache-2.0

//! The prefix cache's recurrent-state snapshots (Gated-DeltaNet and Mamba-2 hybrids: Qwen3.5 /
//! Qwen3.6, Nemotron-H).
//!
//! A recurrent layer's state after `p` tokens is the scan over all `p` of them, so a cached KV
//! prefix alone cannot resume one: the request would start its first forward at `p` with a zeroed
//! state. A hit is real only where a SNAPSHOT of that state exists — the state a request's scan
//! held as it passed `p`, which the recurrent kernels store into a snapshot slot in the forward
//! that passes it. A restore starts the scan from exactly those numbers, with no recompute drift.
//!
//! This index is the host-side half: which snapshot slot holds which prefix, whether its copy
//! is known to have happened, and which slot to reuse next. It carries no device state.
//!
//! ⛔ A SNAPSHOT IS READY ONLY ONCE ITS STEP'S OUTPUT HAS COME BACK. It is reserved when the
//! step that passes its position is scheduled, but the copy happens when that step EXECUTES —
//! and under async scheduling the next step is already being scheduled by then. A step that fails
//! never reports its output, so a snapshot that went hittable at reservation would hand every
//! later request with that prefix whatever the slot held before: fluent, wrong, and silent.

use std::cmp::Reverse;
use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::ops::Range;

/// ⭐ HOW A RECURRENT MODEL'S SNAPSHOTS SHARE THE KV BUDGET, as its worker lays the memory out
/// (#495): KV is mapped and released [`Self::kv_granule_blocks`] blocks at a time and snapshots
/// [`Self::unit_slots`] slots at a time — each the least run whose bytes are whole sparse tiles in
/// every store it spans, so a release frees whole tiles and nothing else.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SnapshotBudget {
    /// Bytes one KV block takes across every store a block lives in.
    pub kv_block_bytes: u64,
    /// Blocks per KV granule.
    pub kv_granule_blocks: NonZeroUsize,
    /// Bytes one snapshot slot takes across every recurrent-state store.
    pub slot_bytes: u64,
    /// Slots per snapshot unit.
    pub unit_slots: NonZeroUsize,
    /// Slots the worker can address: as many as the whole budget holds, a whole number of units.
    pub max_slots: usize,
}

/// One snapshot slot's contents.
#[derive(Debug, Clone)]
struct Snapshot {
    /// Strict-prefix chain hash of the block ending at the snapshot's position.
    key: u64,
    /// The token count it holds the state after.
    position: u32,
    /// `true` once the step that writes it reported back; only ready snapshots are hit.
    ready: bool,
    /// The request whose step writes it, until it is ready. A reservation whose request is
    /// freed first is dropped: its step either failed or will write a slot nobody reads.
    writer: Option<String>,
    /// Scheduling step of the last hit, reservation, or admission whose prefix it lies on (LRU).
    last_used: u64,
    /// Scheduling step in which a request reads it (a restore) or writes it (a reservation):
    /// the slot is not reused within that step.
    ///
    /// ⛔ RECENCY ALONE DOES NOT PIN. An older snapshot on a hit's prefix is neither read nor
    /// written; pinning it too let one long conversation fill the pool with its own ancestors,
    /// after which every turn's save found no slot to take.
    pinned: u64,
    /// The one request that read it in step `pinned` (its restore), `None` when the pin is a
    /// reservation's or more than one request read it. That request's own save may take the
    /// slot over in the same step: every recurrent kernel reads a restored row at the start of
    /// its row loop and stores a saved one later, in the same thread, so the read sees the old
    /// state — and the conversation's newer snapshot replaces its older one.
    reader: Option<String>,
}

/// Snapshot slot index: prefix hash → slot, readiness, and LRU over a fixed slot count.
#[derive(Debug, Clone)]
pub(crate) struct StateSnapshots {
    slots: Vec<Option<Snapshot>>,
    by_key: HashMap<u64, usize>,
    /// Scheduling step counter, advanced once per `schedule()`.
    step: u64,
}

impl StateSnapshots {
    pub(crate) fn new(num_slots: usize) -> Self {
        Self {
            slots: vec![None; num_slots],
            by_key: HashMap::new(),
            step: 0,
        }
    }

    pub(crate) fn num_slots(&self) -> usize {
        self.slots.len()
    }

    /// Start a new scheduling step.
    pub(crate) fn tick(&mut self) {
        self.step += 1;
    }

    /// The slot holding a READY snapshot for `key`.
    pub(crate) fn ready_slot(&self, key: u64) -> Option<usize> {
        let &slot = self.by_key.get(&key)?;
        self.slots[slot].as_ref().filter(|c| c.ready).map(|_| slot)
    }

    /// Whether any snapshot for `key` exists, ready or not — one prefix is snapshotted once.
    pub(crate) fn contains(&self, key: u64) -> bool {
        self.by_key.contains_key(&key)
    }

    /// Mark a slot used this step: an older snapshot on a hit's prefix. Recency only.
    pub(crate) fn touch(&mut self, slot: usize) {
        if let Some(c) = self.slots[slot].as_mut() {
            c.last_used = self.step;
        }
    }

    /// Mark a slot read this step by `reader` (a restore): used, and not reused before the step
    /// runs — but by `reader`'s own save, when no other request read it too.
    pub(crate) fn pin(&mut self, slot: usize, reader: &str) {
        let step = self.step;
        if let Some(c) = self.slots[slot].as_mut() {
            c.reader = match (c.pinned == step, c.reader.take()) {
                (false, _) => Some(reader.to_owned()),
                (true, Some(r)) if r == reader => Some(r),
                (true, _) => None,
            };
            c.last_used = step;
            c.pinned = step;
        }
    }

    /// The scheduling step now: the clock every `last_used` reads.
    pub(crate) fn now(&self) -> u64 {
        self.step
    }

    /// The first empty slot whose memory `mapped` says is there.
    pub(crate) fn empty_slot(&self, mapped: impl Fn(usize) -> bool) -> Option<usize> {
        (0..self.slots.len()).find(|&s| self.slots[s].is_none() && mapped(s))
    }

    /// The snapshot a save by `writer` would evict, and when it was last used: never one read or
    /// written this step — but the one `writer` alone restores this step, taken over when nothing
    /// else can be. `None` when every snapshot is read or written this step.
    pub(crate) fn victim(&self, writer: &str) -> Option<(usize, u64)> {
        let step = self.step;
        let held = |c: &Snapshot| c.pinned < step;
        let own = |c: &Snapshot| c.pinned == step && c.reader.as_deref() == Some(writer);
        (self.slots.iter().enumerate())
            .filter_map(|(i, c)| c.as_ref().map(|c| (i, c)))
            .filter(|(_, c)| held(c) || own(c))
            // Unpinned first, oldest first; among equals a pending snapshot before a ready one
            // (the only kind a request can use), then the DEEPEST: an agent loop resumes from the
            // oldest snapshot of its chain once the next user message re-renders the loop's turns,
            // and the newest one is pinned by the restore that read it.
            .min_by_key(|(_, c)| (own(c), c.last_used, c.ready, Reverse(c.position)))
            .map(|(i, c)| (i, c.last_used))
    }

    /// When the snapshots in `slots` were last used (0 with none there), or `None` while one of
    /// them is read or written this step: whether, and how cheaply, their memory can be released.
    pub(crate) fn age_of(&self, slots: Range<usize>) -> Option<u64> {
        let step = self.step;
        let mut age = 0;
        for c in self.slots[slots].iter().flatten() {
            if c.pinned == step {
                return None;
            }
            age = age.max(c.last_used);
        }
        Some(age)
    }

    /// Drop `key`'s snapshot, unless a step reads or writes it now: the KV it resumes over is gone
    /// (a block a step reads is held, so never evicted under it).
    pub(crate) fn evict_key(&mut self, key: u64) {
        let step = self.step;
        if let Some(&slot) = self.by_key.get(&key)
            && self.slots[slot].as_ref().is_some_and(|c| c.pinned < step)
        {
            self.evict(slot);
        }
    }

    /// Drop every snapshot in `slots`: their memory is being released.
    pub(crate) fn evict_all(&mut self, slots: Range<usize>) {
        for slot in slots {
            self.evict(slot);
        }
    }

    /// Hold `key`'s snapshot at `position`, written by `writer`'s step, in `slot`, evicting what
    /// it held.
    pub(crate) fn place(&mut self, slot: usize, key: u64, position: u32, writer: &str) {
        self.evict(slot);
        let step = self.step;
        self.slots[slot] = Some(Snapshot {
            key,
            position,
            ready: false,
            writer: Some(writer.to_owned()),
            last_used: step,
            pinned: step,
            reader: None,
        });
        self.by_key.insert(key, slot);
    }

    /// `writer`'s step that writes `slot` reported back: its snapshot is usable. A no-op unless
    /// the slot still holds THAT writer's reservation of `key` — one dropped and re-reserved for
    /// the same prefix by another request is that request's to commit, after its own step ran.
    pub(crate) fn commit(&mut self, slot: usize, key: u64, writer: &str) {
        if let Some(c) = self.slots.get_mut(slot).and_then(Option::as_mut)
            && c.key == key
            && c.writer.as_deref() == Some(writer)
        {
            c.ready = true;
            c.writer = None;
        }
    }

    /// Drop the not-yet-ready snapshots `writer`'s steps were going to write.
    pub(crate) fn drop_pending_of(&mut self, writer: &str) {
        let pending: Vec<usize> = self
            .slots
            .iter()
            .enumerate()
            .filter(|(_, c)| {
                c.as_ref()
                    .is_some_and(|c| c.writer.as_deref() == Some(writer))
            })
            .map(|(i, _)| i)
            .collect();
        for slot in pending {
            self.evict(slot);
        }
    }

    pub(crate) fn clear(&mut self) {
        self.slots.iter_mut().for_each(|c| *c = None);
        self.by_key.clear();
    }

    /// Number of snapshots a request can hit.
    pub(crate) fn num_ready(&self) -> usize {
        self.slots.iter().flatten().filter(|c| c.ready).count()
    }

    fn evict(&mut self, slot: usize) {
        if let Some(c) = self.slots[slot].take()
            && self.by_key.get(&c.key) == Some(&slot)
        {
            self.by_key.remove(&c.key);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A save into a pool whose every slot is mapped: an empty slot first, else the victim's.
    fn reserve(snaps: &mut StateSnapshots, key: u64, position: u32, writer: &str) -> Option<usize> {
        if snaps.contains(key) {
            return None;
        }
        let slot = (snaps.empty_slot(|_| true)).or_else(|| snaps.victim(writer).map(|(s, _)| s))?;
        snaps.place(slot, key, position, writer);
        Some(slot)
    }

    #[test]
    fn a_reserved_snapshot_is_hit_only_after_commit() {
        let mut snaps = StateSnapshots::new(2);
        let slot = reserve(&mut snaps, 7, 112, "a").unwrap();
        assert_eq!(
            snaps.ready_slot(7),
            None,
            "not hittable before its step reports back"
        );
        assert!(snaps.contains(7));
        snaps.commit(slot, 7, "a");
        assert_eq!(snaps.ready_slot(7), Some(slot));
    }

    #[test]
    fn one_prefix_is_snapshotted_once() {
        let mut snaps = StateSnapshots::new(4);
        assert!(reserve(&mut snaps, 7, 112, "a").is_some());
        assert_eq!(reserve(&mut snaps, 7, 112, "b"), None, "pending duplicate");
        let slot = snaps.by_key[&7];
        snaps.commit(slot, 7, "a");
        assert_eq!(reserve(&mut snaps, 7, 112, "c"), None, "ready duplicate");
    }

    #[test]
    fn a_freed_writer_drops_only_its_pending_snapshots() {
        let mut snaps = StateSnapshots::new(3);
        let done = reserve(&mut snaps, 1, 16, "a").unwrap();
        snaps.commit(done, 1, "a");
        reserve(&mut snaps, 2, 32, "a").unwrap();
        reserve(&mut snaps, 3, 48, "b").unwrap();
        snaps.drop_pending_of("a");
        assert_eq!(
            snaps.ready_slot(1),
            Some(done),
            "a committed snapshot outlives its writer"
        );
        assert!(!snaps.contains(2));
        assert!(snaps.contains(3));
    }

    #[test]
    fn a_late_commit_for_a_replaced_reservation_is_ignored() {
        let mut snaps = StateSnapshots::new(1);
        let slot = reserve(&mut snaps, 1, 16, "a").unwrap();
        snaps.drop_pending_of("a");
        snaps.tick();
        assert_eq!(reserve(&mut snaps, 2, 32, "b"), Some(slot));
        snaps.commit(slot, 1, "a");
        assert_eq!(snaps.ready_slot(1), None);
        assert_eq!(
            snaps.ready_slot(2),
            None,
            "still pending: its own step has not reported"
        );
    }

    #[test]
    fn a_late_commit_does_not_ready_another_writers_reservation_of_the_same_prefix() {
        // a's step failed after it reserved; a was freed, b re-reserved the same prefix in the
        // same slot. a's step reporting back (it ran, on a state the failure left) must not make
        // the slot hittable before b's own step writes it.
        let mut snaps = StateSnapshots::new(1);
        let slot = reserve(&mut snaps, 9, 144, "a").unwrap();
        snaps.drop_pending_of("a");
        snaps.tick();
        assert_eq!(reserve(&mut snaps, 9, 144, "b"), Some(slot));
        snaps.commit(slot, 9, "a");
        assert_eq!(snaps.ready_slot(9), None);
        snaps.commit(slot, 9, "b");
        assert_eq!(snaps.ready_slot(9), Some(slot));
    }

    #[test]
    fn eviction_takes_the_least_recently_used_slot_not_used_this_step() {
        let mut snaps = StateSnapshots::new(2);
        let a = reserve(&mut snaps, 1, 16, "a").unwrap();
        snaps.commit(a, 1, "a");
        let b = reserve(&mut snaps, 2, 32, "b").unwrap();
        snaps.commit(b, 2, "b");
        assert_eq!(
            reserve(&mut snaps, 3, 48, "c"),
            None,
            "both slots written this step"
        );
        snaps.tick();
        snaps.touch(a);
        assert_eq!(
            reserve(&mut snaps, 3, 48, "c"),
            Some(b),
            "b is older than a's hit"
        );
        assert_eq!(snaps.ready_slot(2), None);
        assert_eq!(snaps.ready_slot(1), Some(a));
    }

    #[test]
    fn a_touched_ancestor_stays_evictable_a_restored_slot_does_not() {
        // A conversation whose own snapshots fill the pool: the turn restores its newest one
        // (pinned) and touches the older one (recency only), and its save still finds a slot.
        let mut snaps = StateSnapshots::new(2);
        let old = reserve(&mut snaps, 1, 592, "t1").unwrap();
        snaps.commit(old, 1, "t1");
        snaps.tick();
        let new = reserve(&mut snaps, 2, 976, "t2").unwrap();
        snaps.commit(new, 2, "t2");
        snaps.tick();
        snaps.touch(old);
        snaps.pin(new, "t3");
        assert_eq!(reserve(&mut snaps, 3, 1376, "t3"), Some(old));
        assert_eq!(
            snaps.ready_slot(2),
            Some(new),
            "the restored slot survives its step"
        );
    }

    /// One slot — a 32 GB Mac's lot beside Qwen3.6-35B-A3B: each turn restores the snapshot the
    /// turn before saved and saves its own into the same slot in the same step, so a conversation
    /// keeps resuming at its latest turn.
    #[test]
    fn a_turn_takes_over_the_slot_it_alone_restores() {
        let mut snaps = StateSnapshots::new(1);
        let slot = reserve(&mut snaps, 1, 1024, "t1").unwrap();
        snaps.commit(slot, 1, "t1");
        snaps.tick();
        snaps.pin(slot, "t2");
        assert_eq!(reserve(&mut snaps, 2, 2048, "t2"), Some(slot));
        assert!(!snaps.contains(1), "the older snapshot is replaced");
        snaps.commit(slot, 2, "t2");
        assert_eq!(snaps.ready_slot(2), Some(slot));
    }

    /// A slot two requests restore in one step, or another request restores, is not taken over:
    /// the other reader could read it after the save wrote it.
    #[test]
    fn a_slot_another_request_restores_is_not_taken_over() {
        let mut snaps = StateSnapshots::new(1);
        let slot = reserve(&mut snaps, 1, 1024, "t1").unwrap();
        snaps.commit(slot, 1, "t1");
        snaps.tick();
        snaps.pin(slot, "a");
        assert_eq!(reserve(&mut snaps, 2, 2048, "b"), None, "read by another");
        snaps.pin(slot, "b");
        assert_eq!(reserve(&mut snaps, 2, 2048, "a"), None, "read by two");
        assert_eq!(reserve(&mut snaps, 2, 2048, "b"), None, "read by two");
        assert_eq!(snaps.ready_slot(1), Some(slot));
    }

    #[test]
    fn among_equally_old_snapshots_the_deepest_goes_first() {
        let mut snaps = StateSnapshots::new(3);
        for (key, position) in [(1, 512), (2, 1024), (3, 1536)] {
            let slot = reserve(&mut snaps, key, position, "w").unwrap();
            snaps.commit(slot, key, "w");
        }
        snaps.tick();
        assert_eq!(
            reserve(&mut snaps, 4, 2048, "x"),
            Some(2),
            "the 1536 snapshot"
        );
        assert!(
            snaps.ready_slot(1).is_some(),
            "the oldest of the chain survives"
        );
    }

    #[test]
    fn clear_forgets_everything() {
        let mut snaps = StateSnapshots::new(2);
        let a = reserve(&mut snaps, 1, 16, "a").unwrap();
        snaps.commit(a, 1, "a");
        snaps.clear();
        assert_eq!(snaps.num_ready(), 0);
        assert!(!snaps.contains(1));
        assert_eq!(reserve(&mut snaps, 1, 16, "b"), Some(0));
    }
}
