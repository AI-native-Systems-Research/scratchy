// SPDX-License-Identifier: Apache-2.0
//! TurboQuant eviction layer — the LRU map between logical KV blocks (backed by
//! the packed 3-bit store, full capacity) and a BOUNDED fp16 "window" of
//! physical blocks that attention actually reads.
//!
//! This is the brain of the pool-capacity 4.6x design: the durable cache lives
//! packed (4.6x smaller, so the KV budget buys 4.6x more logical blocks); only
//! the blocks an active forward touches are dequant'd into the bounded fp16
//! window. A single long sequence whose whole context is active still needs all
//! its blocks resident (no win there — known + accepted); the win is that
//! inactive/preempted sequences' blocks stay packed-only.
//!
//! `WindowMap` is pure index logic (no GPU) so the eviction policy is unit
//! tested in isolation; the GPU fill (`tq_dequant_paged`) + forward wiring sit
//! on top of it.

use std::collections::{HashMap, VecDeque};

/// Result of resolving one logical block against the window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Resolved {
    /// The physical window slot the logical block now occupies.
    pub slot: usize,
    /// True if the block was NOT resident and must be dequant-filled from the
    /// packed store into `slot` before attention reads it.
    pub miss: bool,
    /// The logical block evicted from `slot` to make room (its fp16 is now
    /// stale; it survives in the packed store). `None` if `slot` was free.
    pub evicted: Option<u32>,
}

/// LRU map: logical KV block index -> bounded fp16 window slot. Fixed capacity;
/// a miss on a full window evicts the least-recently-used block.
pub struct WindowMap {
    capacity: usize,
    slot_of: HashMap<u32, usize>,
    block_at: Vec<Option<u32>>,
    /// Logical blocks in use-order; front = least recently used.
    lru: VecDeque<u32>,
}

impl WindowMap {
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "window capacity must be > 0");
        Self {
            capacity,
            slot_of: HashMap::with_capacity(capacity),
            block_at: vec![None; capacity],
            lru: VecDeque::with_capacity(capacity),
        }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn resident(&self) -> usize {
        self.slot_of.len()
    }

    fn bump_lru(&mut self, block: u32) {
        if let Some(pos) = self.lru.iter().position(|&b| b == block) {
            self.lru.remove(pos);
        }
        self.lru.push_back(block);
    }

    /// Ensure `block` is resident, returning its slot + whether it was a miss
    /// (needs dequant) + any evicted block. Marks `block` most-recently-used.
    pub fn resolve(&mut self, block: u32) -> Resolved {
        if let Some(&slot) = self.slot_of.get(&block) {
            self.bump_lru(block);
            return Resolved {
                slot,
                miss: false,
                evicted: None,
            };
        }
        // Miss: find a free slot, else evict the LRU block.
        let (slot, evicted) = if let Some(free) = self.block_at.iter().position(|b| b.is_none()) {
            (free, None)
        } else {
            let victim = self.lru.pop_front().expect("non-empty LRU on full window");
            let slot = self.slot_of.remove(&victim).expect("LRU block has a slot");
            (slot, Some(victim))
        };
        self.slot_of.insert(block, slot);
        self.block_at[slot] = Some(block);
        self.bump_lru(block);
        Resolved {
            slot,
            miss: true,
            evicted,
        }
    }

    pub fn slot_of(&self, block: u32) -> Option<usize> {
        self.slot_of.get(&block).copied()
    }

    /// Assign a FRESH window slot for `block` — its content is being written
    /// anew this forward (first token of the block, offset 0), so any stale
    /// mapping (a reused freed logical block) must be dropped and NO dequant
    /// should follow. Evicts LRU if the window is full. Self-contained reuse
    /// handling: detecting an offset-0 write needs no scheduler block-free hook.
    pub fn assign_fresh(&mut self, block: u32) -> usize {
        // Drop any stale mapping for this logical block (reuse after free).
        if let Some(old) = self.slot_of.remove(&block) {
            self.block_at[old] = None;
            if let Some(pos) = self.lru.iter().position(|&b| b == block) {
                self.lru.remove(pos);
            }
        }
        let slot = if let Some(free) = self.block_at.iter().position(|b| b.is_none()) {
            free
        } else {
            let victim = self.lru.pop_front().expect("non-empty LRU on full window");
            let s = self.slot_of.remove(&victim).expect("victim has a slot");
            self.block_at[s] = None;
            s
        };
        self.slot_of.insert(block, slot);
        self.block_at[slot] = Some(block);
        self.bump_lru(block);
        slot
    }

    /// Resolve a whole sequence's logical block_table to window slots in one
    /// pass: returns the remapped block_table (window slots, same order) and the
    /// list of `(logical_block, slot)` MISSES that must be dequant-filled before
    /// attention. All requested blocks must fit the window (caller guarantees
    /// the active working set <= capacity), else a needed block could be evicted
    /// by a later block in the SAME request — which we detect and panic on.
    pub fn resolve_table(&mut self, logical: &[u32]) -> (Vec<u32>, Vec<(u32, usize)>) {
        assert!(
            logical.len() <= self.capacity,
            "active block_table ({}) exceeds window capacity ({}) — would self-evict",
            logical.len(),
            self.capacity
        );
        let mut slots = Vec::with_capacity(logical.len());
        let mut misses = Vec::new();
        for &b in logical {
            let r = self.resolve(b);
            // A request must not evict one of its own just-resolved blocks.
            if let Some(ev) = r.evicted {
                debug_assert!(
                    !logical[..slots.len()].contains(&ev),
                    "in-request self-eviction of block {ev}"
                );
            }
            if r.miss {
                misses.push((b, r.slot));
            }
            slots.push(r.slot as u32);
        }
        (slots, misses)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hit_keeps_slot_and_no_miss() {
        let mut w = WindowMap::new(4);
        let a = w.resolve(10);
        assert!(a.miss && a.evicted.is_none());
        let a2 = w.resolve(10);
        assert_eq!(a2.slot, a.slot);
        assert!(!a2.miss && a2.evicted.is_none());
    }

    #[test]
    fn lru_evicts_least_recently_used() {
        let mut w = WindowMap::new(2);
        let sa = w.resolve(1).slot; // [1]
        let _sb = w.resolve(2).slot; // [1,2]
        w.resolve(1); // touch 1 -> LRU front is now 2
        let c = w.resolve(3); // miss, evicts 2 (LRU)
        assert!(c.miss);
        assert_eq!(c.evicted, Some(2));
        // 1 still resident in its slot (no re-dequant)
        let a = w.resolve(1);
        assert!(!a.miss);
        assert_eq!(a.slot, sa);
        // 2 was evicted -> re-request is a miss again
        let b = w.resolve(2);
        assert!(b.miss);
    }

    #[test]
    fn resolve_table_remaps_and_lists_misses() {
        let mut w = WindowMap::new(8);
        // First request: all 3 blocks miss, get slots 0,1,2.
        let (slots, misses) = w.resolve_table(&[100, 101, 102]);
        assert_eq!(slots, vec![0, 1, 2]);
        assert_eq!(misses, vec![(100, 0), (101, 1), (102, 2)]);
        // Second request reuses 101,102 (hits) + a new 103 (miss).
        let (slots2, misses2) = w.resolve_table(&[101, 102, 103]);
        assert_eq!(slots2, vec![1, 2, 3]);
        assert_eq!(misses2, vec![(103, 3)]);
    }

    #[test]
    fn assign_fresh_invalidates_stale_reused_block() {
        let mut w = WindowMap::new(4);
        // Block 7 first lands at some slot with prior data.
        let s = w.resolve(7).slot;
        assert_eq!(w.slot_of(7), Some(s));
        // The logical block 7 is freed + reused by a new sequence -> fresh.
        let s2 = w.assign_fresh(7);
        // It's resident again (single mapping, no stale duplicate) and a later
        // hit returns the fresh slot without a miss.
        assert_eq!(w.slot_of(7), Some(s2));
        let hit = w.resolve(7);
        assert!(!hit.miss);
        assert_eq!(hit.slot, s2);
        assert_eq!(
            w.resident(),
            1,
            "no stale duplicate entry for the reused block"
        );
    }

    #[test]
    fn window_holds_fewer_than_logical_capacity() {
        // The whole point: 4 physical slots back many more logical blocks over
        // time (each miss dequant-fills from the packed store).
        let mut w = WindowMap::new(4);
        for b in 0..100u32 {
            let r = w.resolve(b);
            assert!(r.miss, "fresh block is always a miss");
            assert!(r.slot < 4, "slot stays within the bounded window");
        }
        assert_eq!(w.resident(), 4);
    }
}
