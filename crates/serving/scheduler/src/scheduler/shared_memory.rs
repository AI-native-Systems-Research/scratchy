// SPDX-License-Identifier: Apache-2.0

//! ⭐ ONE BUDGET FOR KV AND THE RECURRENT-STATE SNAPSHOTS (#495).
//!
//! A Gated-DeltaNet / Mamba-2 hybrid's prefix cache keeps two kinds of state, KV blocks and
//! snapshots of the recurrent state, and a hit needs both. They draw on ONE budget — what the KV
//! pool alone maps with caching off — and hand memory to each other as they need it: KV a granule
//! of blocks at a time, the snapshots a unit of slots at a time ([`SnapshotBudget`]).
//!
//! This is the ledger: what is mapped, what it costs, which granules hold no block (so may be
//! released), and the map and release ops this step sends the worker. Which side gives way is the
//! block tracker's to decide; it owns both.

use std::ops::Range;

use super::output::{KvGranule, SharedMemoryOps, SnapshotUnit};
use super::recurrent_state::SnapshotBudget;

/// One KV granule's state.
#[derive(Debug, Clone, Copy, Default)]
struct Granule {
    mapped: bool,
    /// Its blocks a request holds.
    held: u32,
    /// The last step one of its blocks was taken, touched or freed: how recently its cached
    /// contents were used.
    age: u64,
}

/// What KV and the snapshots have mapped of their shared budget.
#[derive(Debug, Clone)]
pub(crate) struct SharedMemory {
    budget: SnapshotBudget,
    total_blocks: usize,
    granules: Vec<Granule>,
    units: Vec<bool>,
    used: u64,
    ops: SharedMemoryOps,
}

impl SharedMemory {
    /// Nothing mapped yet, over `total_blocks` KV blocks: the budget is what they take.
    pub(crate) fn new(budget: SnapshotBudget, total_blocks: usize) -> Self {
        let g = budget.kv_granule_blocks.get();
        let u = budget.unit_slots.get();
        assert!(
            budget.max_slots.is_multiple_of(u),
            "{} snapshot slots are not a whole number of {u}-slot units",
            budget.max_slots
        );
        Self {
            budget,
            total_blocks,
            granules: vec![Granule::default(); total_blocks.div_ceil(g)],
            units: vec![false; budget.max_slots / u],
            used: 0,
            ops: SharedMemoryOps::default(),
        }
    }

    /// The whole budget: every block mapped and no snapshot.
    pub(crate) fn limit(&self) -> u64 {
        self.total_blocks as u64 * self.budget.kv_block_bytes
    }

    /// What is not mapped yet.
    pub(crate) fn headroom(&self) -> u64 {
        self.limit() - self.used
    }

    pub(crate) fn granule_of(&self, block: usize) -> usize {
        block / self.budget.kv_granule_blocks.get()
    }

    /// Granule `g`'s blocks: the last granule's run ends with the pool.
    pub(crate) fn blocks_of(&self, g: usize) -> Range<usize> {
        let n = self.budget.kv_granule_blocks.get();
        g * n..((g + 1) * n).min(self.total_blocks)
    }

    pub(crate) fn granule_bytes(&self, g: usize) -> u64 {
        self.blocks_of(g).len() as u64 * self.budget.kv_block_bytes
    }

    pub(crate) fn block_bytes(&self) -> u64 {
        self.budget.kv_block_bytes
    }

    pub(crate) fn unit_bytes(&self) -> u64 {
        self.budget.unit_slots.get() as u64 * self.budget.slot_bytes
    }

    pub(crate) fn slots_of(&self, u: usize) -> Range<usize> {
        let n = self.budget.unit_slots.get();
        u * n..(u + 1) * n
    }

    pub(crate) fn block_mapped(&self, block: usize) -> bool {
        self.granules[self.granule_of(block)].mapped
    }

    pub(crate) fn slot_mapped(&self, slot: usize) -> bool {
        self.units[slot / self.budget.unit_slots.get()]
    }

    /// The granules not mapped, lowest first: the order KV maps them in.
    pub(crate) fn unmapped_granules(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.granules.len()).filter(|&g| !self.granules[g].mapped)
    }

    pub(crate) fn lowest_unmapped_unit(&self) -> Option<usize> {
        self.units.iter().position(|&m| !m)
    }

    pub(crate) fn mapped_units(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.units.len()).filter(|&u| self.units[u])
    }

    /// Mapped granules no request holds a block of, oldest first, with their ages.
    pub(crate) fn free_granules(&self) -> Vec<(usize, u64)> {
        let mut free: Vec<(usize, u64)> = (self.granules.iter().enumerate())
            .filter(|(_, g)| g.mapped && g.held == 0)
            .map(|(i, g)| (i, g.age))
            .collect();
        free.sort_by_key(|&(i, age)| (age, i));
        free
    }

    pub(crate) fn map_granule(&mut self, g: usize) {
        let bytes = self.granule_bytes(g);
        assert!(!self.granules[g].mapped, "granule {g} mapped twice");
        assert!(bytes <= self.headroom(), "granule {g} past the budget");
        self.granules[g] = Granule {
            mapped: true,
            ..Granule::default()
        };
        self.used += bytes;
        self.ops.map_kv.push(KvGranule(g as u32));
    }

    pub(crate) fn release_granule(&mut self, g: usize) {
        let granule = self.granules[g];
        assert!(granule.mapped, "granule {g} released unmapped");
        assert_eq!(granule.held, 0, "granule {g} released while held");
        self.granules[g].mapped = false;
        self.used -= self.granule_bytes(g);
        // Mapped this step, so not mapped on the worker yet: the map is dropped, not undone.
        let id = KvGranule(g as u32);
        match self.ops.map_kv.iter().position(|&m| m == id) {
            Some(i) => {
                self.ops.map_kv.remove(i);
            }
            None => self.ops.release_kv.push(id),
        }
    }

    pub(crate) fn map_unit(&mut self, u: usize) {
        assert!(!self.units[u], "snapshot unit {u} mapped twice");
        assert!(
            self.unit_bytes() <= self.headroom(),
            "unit {u} past the budget"
        );
        self.units[u] = true;
        self.used += self.unit_bytes();
        self.ops.map_snapshots.push(SnapshotUnit(u as u32));
    }

    pub(crate) fn release_unit(&mut self, u: usize) {
        assert!(self.units[u], "snapshot unit {u} released unmapped");
        self.units[u] = false;
        self.used -= self.unit_bytes();
        let id = SnapshotUnit(u as u32);
        match self.ops.map_snapshots.iter().position(|&m| m == id) {
            Some(i) => {
                self.ops.map_snapshots.remove(i);
            }
            None => self.ops.release_snapshots.push(id),
        }
    }

    /// A request took `block` (fresh, or a cached block it hits) at step `now`.
    pub(crate) fn take(&mut self, block: usize, now: u64) {
        let g = &mut self.granules[block / self.budget.kv_granule_blocks.get()];
        g.held += 1;
        g.age = g.age.max(now);
    }

    /// The last request holding `block` let it go at step `now`.
    pub(crate) fn give_back(&mut self, block: usize, now: u64) {
        let g = &mut self.granules[block / self.budget.kv_granule_blocks.get()];
        g.held -= 1;
        g.age = g.age.max(now);
    }

    /// This step's map and release ops, for its output.
    pub(crate) fn take_ops(&mut self) -> SharedMemoryOps {
        std::mem::take(&mut self.ops)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::num::NonZeroUsize;

    fn ledger() -> SharedMemory {
        // 10 blocks of 4 bytes in granules of 4 (the last one 2 blocks), 2-slot units of 6 bytes.
        let budget = SnapshotBudget {
            kv_block_bytes: 4,
            kv_granule_blocks: NonZeroUsize::new(4).unwrap(),
            slot_bytes: 3,
            unit_slots: NonZeroUsize::new(2).unwrap(),
            max_slots: 6,
        };
        SharedMemory::new(budget, 10)
    }

    #[test]
    fn the_budget_is_what_every_block_takes_and_the_last_granule_is_short() {
        let mut m = ledger();
        assert_eq!((m.limit(), m.headroom()), (40, 40));
        assert_eq!(m.blocks_of(2), 8..10);
        assert_eq!(m.granule_bytes(2), 8);
        for g in 0..3 {
            m.map_granule(g);
        }
        assert_eq!(m.headroom(), 0, "every block mapped is the whole budget");
    }

    #[test]
    fn a_map_released_in_its_own_step_is_dropped_not_undone() {
        let mut m = ledger();
        m.map_granule(0);
        m.map_unit(1);
        m.release_granule(0);
        m.release_unit(1);
        assert!(m.take_ops().is_empty());
        m.map_granule(0);
        m.take_ops();
        m.release_granule(0);
        m.map_granule(0);
        let ops = m.take_ops();
        assert_eq!(
            (ops.release_kv, ops.map_kv),
            (vec![KvGranule(0)], vec![KvGranule(0)]),
            "released then mapped again: the worker releases first, then maps fresh pages"
        );
    }

    #[test]
    fn free_granules_are_the_unheld_mapped_ones_oldest_first() {
        let mut m = ledger();
        for g in 0..3 {
            m.map_granule(g);
        }
        m.take(0, 5);
        m.take(4, 2);
        m.give_back(4, 3);
        m.take(9, 1);
        m.give_back(9, 1);
        assert_eq!(m.free_granules(), vec![(2, 1), (1, 3)], "granule 0 is held");
    }
}
