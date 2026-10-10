// SPDX-License-Identifier: Apache-2.0

//! The prefix cache's recurrent-state snapshots (Gated-DeltaNet hybrids: Qwen3.5 / Qwen3.6).
//!
//! A GDN layer's state after `p` tokens is the scan over all `p` of them, so a cached KV prefix
//! alone cannot resume one: the request would start its first forward at `p` with a zeroed state.
//! A hit is real only where a SNAPSHOT of that state exists — a copy of the live state a request
//! held when one of its forwards ENDED exactly at `p`. The worker writes the state once, after a
//! forward's last token, so the state at a forward end is exactly what the scan left there: a
//! restore replays the same numbers, with no recompute drift.
//!
//! This index is the host-side half: which snapshot slot holds which prefix, whether its copy
//! is known to have happened, and which slot to reuse next. It carries no device state.
//!
//! ⛔ A SNAPSHOT IS READY ONLY ONCE ITS STEP'S OUTPUT HAS COME BACK. It is reserved when the
//! step that ends at its position is scheduled, but the copy happens when that step EXECUTES —
//! and under async scheduling the next step is already being scheduled by then. A step that fails
//! never reports its output, so a snapshot that went hittable at reservation would hand every
//! later request with that prefix whatever the slot held before: fluent, wrong, and silent.

use std::cmp::Reverse;
use std::collections::HashMap;

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

    /// Mark a slot read this step (a restore): used, and not reused before the step runs.
    pub(crate) fn pin(&mut self, slot: usize) {
        if let Some(c) = self.slots[slot].as_mut() {
            c.last_used = self.step;
            c.pinned = self.step;
        }
    }

    /// Reserve a slot for `key` at `position`, written by `writer`'s step. `None` when `key`
    /// already has a snapshot, or when every slot is read or written this step.
    pub(crate) fn reserve(&mut self, key: u64, position: u32, writer: &str) -> Option<usize> {
        if self.contains(key) {
            return None;
        }
        let slot = match self.slots.iter().position(Option::is_none) {
            Some(free) => free,
            None => {
                let (victim, _) = self
                    .slots
                    .iter()
                    .enumerate()
                    .filter_map(|(i, c)| c.as_ref().map(|c| (i, c)))
                    .filter(|(_, c)| c.pinned < self.step)
                    // Oldest first; among equals a pending snapshot before a ready one (the only
                    // kind a request can use), then the DEEPEST: an agent loop resumes from the
                    // oldest snapshot of its chain once the next user message re-renders the
                    // loop's turns, and the newest one is pinned by the restore that read it.
                    .min_by_key(|(_, c)| (c.last_used, c.ready, Reverse(c.position)))?;
                self.evict(victim);
                victim
            }
        };
        self.slots[slot] = Some(Snapshot {
            key,
            position,
            ready: false,
            writer: Some(writer.to_owned()),
            last_used: self.step,
            pinned: self.step,
        });
        self.by_key.insert(key, slot);
        Some(slot)
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

    #[test]
    fn a_reserved_snapshot_is_hit_only_after_commit() {
        let mut snaps = StateSnapshots::new(2);
        let slot = snaps.reserve(7, 112, "a").unwrap();
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
        assert!(snaps.reserve(7, 112, "a").is_some());
        assert_eq!(snaps.reserve(7, 112, "b"), None, "pending duplicate");
        let slot = snaps.by_key[&7];
        snaps.commit(slot, 7, "a");
        assert_eq!(snaps.reserve(7, 112, "c"), None, "ready duplicate");
    }

    #[test]
    fn a_freed_writer_drops_only_its_pending_snapshots() {
        let mut snaps = StateSnapshots::new(3);
        let done = snaps.reserve(1, 16, "a").unwrap();
        snaps.commit(done, 1, "a");
        snaps.reserve(2, 32, "a").unwrap();
        snaps.reserve(3, 48, "b").unwrap();
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
        let slot = snaps.reserve(1, 16, "a").unwrap();
        snaps.drop_pending_of("a");
        snaps.tick();
        assert_eq!(snaps.reserve(2, 32, "b"), Some(slot));
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
        let slot = snaps.reserve(9, 144, "a").unwrap();
        snaps.drop_pending_of("a");
        snaps.tick();
        assert_eq!(snaps.reserve(9, 144, "b"), Some(slot));
        snaps.commit(slot, 9, "a");
        assert_eq!(snaps.ready_slot(9), None);
        snaps.commit(slot, 9, "b");
        assert_eq!(snaps.ready_slot(9), Some(slot));
    }

    #[test]
    fn eviction_takes_the_least_recently_used_slot_not_used_this_step() {
        let mut snaps = StateSnapshots::new(2);
        let a = snaps.reserve(1, 16, "a").unwrap();
        snaps.commit(a, 1, "a");
        let b = snaps.reserve(2, 32, "b").unwrap();
        snaps.commit(b, 2, "b");
        assert_eq!(
            snaps.reserve(3, 48, "c"),
            None,
            "both slots written this step"
        );
        snaps.tick();
        snaps.touch(a);
        assert_eq!(
            snaps.reserve(3, 48, "c"),
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
        let old = snaps.reserve(1, 592, "t1").unwrap();
        snaps.commit(old, 1, "t1");
        snaps.tick();
        let new = snaps.reserve(2, 976, "t2").unwrap();
        snaps.commit(new, 2, "t2");
        snaps.tick();
        snaps.touch(old);
        snaps.pin(new);
        assert_eq!(snaps.reserve(3, 1376, "t3"), Some(old));
        assert_eq!(
            snaps.ready_slot(2),
            Some(new),
            "the restored slot survives its step"
        );
    }

    #[test]
    fn among_equally_old_snapshots_the_deepest_goes_first() {
        let mut snaps = StateSnapshots::new(3);
        for (key, position) in [(1, 512), (2, 1024), (3, 1536)] {
            let slot = snaps.reserve(key, position, "w").unwrap();
            snaps.commit(slot, key, "w");
        }
        snaps.tick();
        assert_eq!(snaps.reserve(4, 2048, "x"), Some(2), "the 1536 snapshot");
        assert!(
            snaps.ready_slot(1).is_some(),
            "the oldest of the chain survives"
        );
    }

    #[test]
    fn clear_forgets_everything() {
        let mut snaps = StateSnapshots::new(2);
        let a = snaps.reserve(1, 16, "a").unwrap();
        snaps.commit(a, 1, "a");
        snaps.clear();
        assert_eq!(snaps.num_ready(), 0);
        assert!(!snaps.contains(1));
        assert_eq!(snaps.reserve(1, 16, "b"), Some(0));
    }
}
