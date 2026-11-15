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
//! is known to have happened, and which slot to reuse next. It carries no device state. With a
//! disk tier, it also indexes which disk slot holds which prefix: a ready snapshot a slot is
//! taken from goes there, and a hit on it is read back (`SnapshotDiskOps`).
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

use super::output::{SnapshotDiskOps, SnapshotFill, SnapshotMove};

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
    /// Snapshots the worker's disk tier holds ([`SnapshotDiskOps`]), outside the budget: a
    /// snapshot evicted from a slot is written out to one, and a hit on it reads it back. 0
    /// without a disk tier.
    ///
    /// [`SnapshotDiskOps`]: crate::scheduler::SnapshotDiskOps
    pub disk_slots: usize,
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
    /// Where a ready snapshot goes when its slot is taken.
    disk: DiskTier,
}

/// A ready snapshot in the disk tier.
#[derive(Debug, Clone, Copy)]
struct Spilled {
    key: u64,
    position: u32,
    /// As [`Snapshot::last_used`], carried over from its slot and renewed by every fill.
    last_used: u64,
}

/// ⭐ THE DISK TIER: ready snapshots a slot was taken from, which a hit reads back into a slot
/// (`SnapshotDiskOps`). Every one is the exact state a scan stored, so a restore through it
/// resumes as one from a slot does. A disk slot is reused least recently used first, never in a
/// step that reads it: the worker writes a step's spills before it reads its fills.
#[derive(Debug, Clone)]
struct DiskTier {
    slots: Vec<Option<Spilled>>,
    by_key: HashMap<u64, usize>,
    /// Per disk slot: the step in which a fill reads it.
    read_at: Vec<u64>,
    ops: SnapshotDiskOps,
}

impl DiskTier {
    fn new(num_slots: usize) -> Self {
        Self {
            slots: vec![None; num_slots],
            by_key: HashMap::new(),
            read_at: vec![0; num_slots],
            ops: SnapshotDiskOps::default(),
        }
    }

    /// The disk slot a spill in `step` writes: an empty one, else the least recently used. Never
    /// one a fill reads in `step`.
    fn take(&mut self, step: u64) -> Option<usize> {
        let free = |d: &usize| self.read_at[*d] < step;
        let empty = (0..self.slots.len()).find(|d| free(d) && self.slots[*d].is_none());
        let d = empty.or_else(|| {
            (0..self.slots.len())
                .filter(free)
                .min_by_key(|&d| self.slots[d].map(|s| s.last_used))
        })?;
        self.forget(d);
        Some(d)
    }

    fn forget(&mut self, d: usize) {
        if let Some(s) = self.slots[d].take() {
            self.by_key.remove(&s.key);
        }
    }
}

impl StateSnapshots {
    pub(crate) fn new(num_slots: usize, disk_slots: usize) -> Self {
        Self {
            slots: vec![None; num_slots],
            by_key: HashMap::new(),
            step: 0,
            disk: DiskTier::new(disk_slots),
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

    /// Whether the disk tier holds `key`'s snapshot, which a hit reads back into a slot
    /// ([`Self::fill`]).
    pub(crate) fn on_disk(&self, key: u64) -> bool {
        self.disk.by_key.contains_key(&key)
    }

    /// Whether a hit on `key` reads its snapshot back from disk: it is there, and in no slot — a
    /// fill whose step has not come back is not read back twice.
    pub(crate) fn fillable(&self, key: u64) -> bool {
        self.on_disk(key) && !self.by_key.contains_key(&key)
    }

    /// Whether any snapshot for `key` exists, ready or not, in a slot or on disk — one prefix is
    /// snapshotted once.
    pub(crate) fn contains(&self, key: u64) -> bool {
        self.by_key.contains_key(&key) || self.on_disk(key)
    }

    /// Whether a ready snapshot evicted from its slot goes to a disk tier rather than away.
    pub(crate) fn spills(&self) -> bool {
        !self.disk.slots.is_empty()
    }

    /// Number of snapshots the disk tier holds.
    pub(crate) fn num_on_disk(&self) -> usize {
        self.disk.by_key.len()
    }

    /// ⭐ READ `key`'s SNAPSHOT BACK FROM THE DISK TIER INTO `slot`, for `reader` to restore this
    /// step, evicting what the slot held. `reader`'s restore reads it in this step, after the
    /// worker's fill. Every other request waits for the step to come back
    /// ([`Self::commit`]), as for a save. A fill that fails is forgotten ([`Self::forget_fill`]).
    pub(crate) fn fill(&mut self, slot: usize, key: u64, reader: &str) {
        let step = self.step;
        let Some(&d) = self.disk.by_key.get(&key) else {
            return;
        };
        // Read this step: no spill of the eviction below may take its disk slot.
        self.disk.read_at[d] = step;
        let spilled = self.disk.slots[d].as_mut().expect("indexed");
        spilled.last_used = step;
        let position = spilled.position;
        self.evict(slot);
        self.slots[slot] = Some(Snapshot {
            key,
            position,
            ready: false,
            writer: Some(reader.to_owned()),
            last_used: step,
            pinned: step,
            reader: Some(reader.to_owned()),
        });
        self.by_key.insert(key, slot);
        self.disk.ops.fills.push(SnapshotFill {
            req_id: reader.to_owned(),
            slot,
            disk: d,
            key,
        });
    }

    /// ⛔ `fill` NEVER CAME BACK, OR CAME BACK FAILED: forget its snapshot, in its slot and on disk.
    /// The next hit on its prefix resumes cold, as it would with no disk tier.
    pub(crate) fn forget_fill(&mut self, fill: &SnapshotFill) {
        let held = |c: &Snapshot| c.key == fill.key && c.writer.as_deref() == Some(&fill.req_id);
        if self.slots[fill.slot].as_ref().is_some_and(held) {
            self.slots[fill.slot] = None;
            self.by_key.remove(&fill.key);
        }
        if let Some(&d) = self.disk.by_key.get(&fill.key) {
            self.disk.forget(d);
        }
    }

    /// The snapshots this step moved between device memory and the disk tier.
    pub(crate) fn take_disk_ops(&mut self) -> SnapshotDiskOps {
        std::mem::take(&mut self.disk.ops)
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
    /// (a block a step reads is held, so never evicted under it). Its disk copy goes too.
    pub(crate) fn evict_key(&mut self, key: u64) {
        let step = self.step;
        if let Some(&slot) = self.by_key.get(&key)
            && self.slots[slot].as_ref().is_some_and(|c| c.pinned < step)
        {
            self.slots[slot] = None;
            self.by_key.remove(&key);
        }
        if let Some(&d) = self.disk.by_key.get(&key) {
            self.disk.forget(d);
        }
    }

    /// Drop every snapshot in `slots`, the ready ones to the disk tier: their memory is being
    /// released.
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
        self.disk.slots.iter_mut().for_each(|s| *s = None);
        self.disk.by_key.clear();
    }

    /// Number of snapshots a request can hit.
    pub(crate) fn num_ready(&self) -> usize {
        self.slots.iter().flatten().filter(|c| c.ready).count()
    }

    /// Empty `slot`. A ready snapshot not on disk yet is written out to the disk tier, where a hit
    /// still finds it.
    fn evict(&mut self, slot: usize) {
        let Some(c) = self.slots[slot].take() else {
            return;
        };
        if self.by_key.get(&c.key) != Some(&slot) {
            return;
        }
        self.by_key.remove(&c.key);
        if !c.ready || self.on_disk(c.key) {
            return;
        }
        if let Some(d) = self.disk.take(self.step) {
            self.disk.slots[d] = Some(Spilled {
                key: c.key,
                position: c.position,
                last_used: c.last_used,
            });
            self.disk.by_key.insert(c.key, d);
            (self.disk.ops.spills).push(SnapshotMove {
                slot,
                disk: d,
                key: c.key,
            });
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
        let mut snaps = StateSnapshots::new(2, 0);
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
        let mut snaps = StateSnapshots::new(4, 0);
        assert!(reserve(&mut snaps, 7, 112, "a").is_some());
        assert_eq!(reserve(&mut snaps, 7, 112, "b"), None, "pending duplicate");
        let slot = snaps.by_key[&7];
        snaps.commit(slot, 7, "a");
        assert_eq!(reserve(&mut snaps, 7, 112, "c"), None, "ready duplicate");
    }

    #[test]
    fn a_freed_writer_drops_only_its_pending_snapshots() {
        let mut snaps = StateSnapshots::new(3, 0);
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
        let mut snaps = StateSnapshots::new(1, 0);
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
        let mut snaps = StateSnapshots::new(1, 0);
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
        let mut snaps = StateSnapshots::new(2, 0);
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
        let mut snaps = StateSnapshots::new(2, 0);
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
        let mut snaps = StateSnapshots::new(1, 0);
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
        let mut snaps = StateSnapshots::new(1, 0);
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
        let mut snaps = StateSnapshots::new(3, 0);
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
        let mut snaps = StateSnapshots::new(2, 0);
        let a = reserve(&mut snaps, 1, 16, "a").unwrap();
        snaps.commit(a, 1, "a");
        snaps.clear();
        assert_eq!(snaps.num_ready(), 0);
        assert!(!snaps.contains(1));
        assert_eq!(reserve(&mut snaps, 1, 16, "b"), Some(0));
    }

    fn fill(req_id: &str, slot: usize, disk: usize, key: u64) -> SnapshotFill {
        SnapshotFill {
            req_id: req_id.into(),
            slot,
            disk,
            key,
        }
    }

    /// `key`'s snapshot at `position`, saved by `writer` and reported back.
    fn saved(snaps: &mut StateSnapshots, key: u64, position: u32, writer: &str) -> usize {
        let slot = reserve(snaps, key, position, writer).unwrap();
        snaps.commit(slot, key, writer);
        snaps.tick();
        slot
    }

    #[test]
    fn a_ready_snapshot_a_slot_is_taken_from_goes_to_disk_and_comes_back() {
        let mut snaps = StateSnapshots::new(1, 2);
        let slot = saved(&mut snaps, 7, 112, "a");
        saved(&mut snaps, 8, 144, "b");
        assert_eq!(
            snaps.take_disk_ops().spills,
            vec![SnapshotMove {
                slot,
                disk: 0,
                key: 7
            }]
        );
        assert_eq!(snaps.ready_slot(7), None);
        assert!(
            snaps.on_disk(7) && snaps.contains(7),
            "one prefix is snapshotted once"
        );
        assert_eq!(reserve(&mut snaps, 7, 112, "c"), None);

        snaps.fill(slot, 7, "c");
        assert_eq!(
            snaps.ready_slot(7),
            None,
            "a hit for no other request before its step comes back"
        );
        assert!(!snaps.fillable(7), "nor read back twice");
        let ops = snaps.take_disk_ops();
        assert_eq!(ops.fills, vec![fill("c", slot, 0, 7)]);
        assert_eq!(
            ops.spills,
            vec![SnapshotMove {
                slot,
                disk: 1,
                key: 8
            }],
            "the snapshot the fill takes the slot of, past the disk slot it reads"
        );
        assert_eq!(snaps.victim("d"), None, "a fill is read this step");
        snaps.commit(slot, 7, "c");
        assert_eq!(
            snaps.ready_slot(7),
            Some(slot),
            "a hit once its step came back"
        );
        snaps.tick();
        snaps.place(slot, 9, 160, "d");
        assert!(
            snaps.take_disk_ops().spills.is_empty(),
            "a snapshot read back is on disk already"
        );
        assert!(snaps.on_disk(7));
    }

    #[test]
    fn a_pending_snapshot_never_goes_to_disk() {
        let mut snaps = StateSnapshots::new(1, 2);
        reserve(&mut snaps, 7, 112, "a").unwrap();
        snaps.tick();
        reserve(&mut snaps, 8, 144, "b").unwrap();
        assert!(snaps.take_disk_ops().is_empty());
        assert!(!snaps.contains(7));
    }

    /// The worker writes a step's spills before it reads its fills, so no spill may write the disk
    /// slot a fill reads in that step: the snapshot it would have spilled is dropped instead.
    #[test]
    fn a_disk_slot_a_fill_reads_is_not_rewritten_in_its_step() {
        let mut snaps = StateSnapshots::new(1, 1);
        let slot = saved(&mut snaps, 7, 112, "a");
        saved(&mut snaps, 8, 144, "b");
        snaps.take_disk_ops();
        snaps.fill(slot, 7, "c");
        let ops = snaps.take_disk_ops();
        assert_eq!(ops.fills, vec![fill("c", slot, 0, 7)]);
        assert!(ops.spills.is_empty());
        assert!(!snaps.contains(8));
        assert!(snaps.on_disk(7));
    }

    #[test]
    fn the_least_recently_used_snapshot_on_disk_makes_room() {
        let mut snaps = StateSnapshots::new(1, 2);
        for (key, writer) in [(1, "a"), (2, "b"), (3, "c"), (4, "d")] {
            saved(&mut snaps, key, 16 * key as u32, writer);
        }
        assert!(!snaps.on_disk(1), "the oldest gone");
        assert!(snaps.on_disk(2) && snaps.on_disk(3));
        let spills = snaps.take_disk_ops().spills;
        assert_eq!(
            spills.iter().map(|m| (m.key, m.disk)).collect::<Vec<_>>(),
            vec![(1, 0), (2, 1), (3, 0)]
        );
    }

    #[test]
    fn a_snapshot_whose_kv_is_gone_leaves_disk_too() {
        let mut snaps = StateSnapshots::new(1, 2);
        saved(&mut snaps, 7, 112, "a");
        saved(&mut snaps, 8, 144, "b");
        snaps.evict_key(7);
        assert!(!snaps.contains(7));
        snaps.evict_key(8);
        assert!(
            !snaps.contains(8),
            "nor is it spilled on its way out of its slot"
        );
        assert_eq!(snaps.take_disk_ops().spills.len(), 1, "7's alone");
    }

    /// A fill the worker could not read back, or whose step never came back, is forgotten in its
    /// slot and on disk: the next request for its prefix finds nothing to resume from.
    #[test]
    fn a_failed_fill_is_forgotten_in_its_slot_and_on_disk() {
        let mut snaps = StateSnapshots::new(1, 2);
        let slot = saved(&mut snaps, 7, 112, "a");
        saved(&mut snaps, 8, 144, "b");
        snaps.fill(slot, 7, "c");
        let ops = snaps.take_disk_ops();
        snaps.forget_fill(&ops.fills[0]);
        snaps.commit(slot, 7, "c");
        assert_eq!(snaps.ready_slot(7), None, "a late commit readies nothing");
        assert!(!snaps.contains(7) && !snaps.fillable(7));
        assert!(snaps.on_disk(8), "the snapshot the fill spilled stays");
        snaps.tick();
        assert_eq!(
            reserve(&mut snaps, 7, 112, "d"),
            Some(slot),
            "the prefix is saved anew"
        );
    }
}
