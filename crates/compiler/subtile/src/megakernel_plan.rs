// SPDX-License-Identifier: Apache-2.0
//! THE MEGAKERNEL PLAN — what one persistent kernel needs to know to play a tape.
//!
//! A megakernel runs every step of a tape inside ONE dispatch: threadgroups claim work in tape
//! order and wait, per work item, only on the work that produced what it reads. Everything that
//! decides how — which waits exist, which steps share a threadgroup, which memory must be
//! device-coherent — is a fact about the tape's DATAFLOW, so it is computed here, once, at
//! expansion time. The runtime decides nothing.
//!
//! The target states each step's reads and writes in its own location vocabulary (`L`) and how
//! many work items the step splits into; this pass is generic over `L` and knows nothing else.
//!
//! # Units
//!
//! A [`Unit`] is what a threadgroup claims. A multi-item step is a unit of its own. A run of
//! single-item steps where each depends on the run is ONE unit: one threadgroup plays it back to
//! back, so its intermediates never leave the threadgroup — a dependency inside a unit is a
//! threadgroup barrier, not a device-wide handoff. (Measured on M5: 1.64 µs per dependent step
//! in one threadgroup vs 2.95 µs handed across threadgroups vs 3.06 µs as separate dispatches.)
//!
//! # Coherence
//!
//! Device memory is only threadgroup-coherent by default (MSL §4.8). A location needs
//! `coherent(device)` exactly when one unit writes it and a DIFFERENT unit touches it inside the
//! same megakernel; everything else keeps the cheaper default.

use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use std::num::NonZeroU32;
use std::ops::Range;

/// A tape step by position.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct StepIx(pub u32);

/// A work unit by position; units are in tape order.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct UnitIx(pub u32);

/// How many work items a step splits into — one threadgroup claims one item.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Items(pub NonZeroU32);

impl Items {
    pub const ONE: Items = Items(NonZeroU32::MIN);
}

/// One step's effect on memory, in the target's location vocabulary.
#[derive(Clone, Debug)]
pub struct StepFlow<L> {
    pub reads: Vec<L>,
    pub writes: Vec<L>,
    pub items: Items,
}

/// A run of consecutive steps one threadgroup claims item by item.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Unit {
    pub steps: Range<u32>,
    pub items: Items,
    /// Units every item of this unit waits for (all of their items), transitively reduced.
    pub waits: Vec<UnitIx>,
}

#[derive(Clone, Debug)]
pub struct MegakernelPlan<L> {
    pub units: Vec<Unit>,
    /// Locations one unit writes and another touches: these need `coherent(device)`.
    pub shared: HashSet<L>,
}

/// Every step a step must wait for: read-after-write, write-after-write and write-after-read.
fn step_deps<L: Copy + Eq + Hash>(steps: &[StepFlow<L>]) -> Vec<HashSet<u32>> {
    let mut last_writer: HashMap<L, u32> = HashMap::new();
    let mut readers: HashMap<L, Vec<u32>> = HashMap::new();
    let mut deps = Vec::with_capacity(steps.len());
    for (s, flow) in (0u32..).zip(steps) {
        let mut d = HashSet::new();
        for l in &flow.reads {
            d.extend(last_writer.get(l).copied());
        }
        for l in &flow.writes {
            d.extend(last_writer.get(l).copied());
            d.extend(readers.get(l).into_iter().flatten().copied());
        }
        d.remove(&s);
        for l in &flow.reads {
            readers.entry(*l).or_default().push(s);
        }
        for l in &flow.writes {
            last_writer.insert(*l, s);
            readers.remove(l);
        }
        deps.push(d);
    }
    deps
}

/// Plan a tape (or one contiguous segment of it) for a megakernel.
pub fn plan<L: Copy + Eq + Hash>(steps: &[StepFlow<L>]) -> MegakernelPlan<L> {
    let deps = step_deps(steps);

    // Units: a single-item step that depends on the open single-item unit joins it.
    let mut units: Vec<Unit> = Vec::new();
    let mut unit_of = Vec::with_capacity(steps.len());
    for (s, flow) in (0u32..).zip(steps) {
        let joins = flow.items == Items::ONE
            && units.last().is_some_and(|u| {
                u.items == Items::ONE && u.steps.end == s && deps[s as usize].iter().any(|p| u.steps.contains(p))
            });
        if joins {
            units.last_mut().expect("joins implies a unit").steps.end = s + 1;
        } else {
            units.push(Unit { steps: s..s + 1, items: flow.items, waits: Vec::new() });
        }
        unit_of.push(UnitIx(units.len() as u32 - 1));
    }

    // Unit waits, transitively reduced: drop a wait already implied through another wait.
    let mut ancestors: Vec<HashSet<UnitIx>> = Vec::with_capacity(units.len());
    for (u, unit) in units.iter_mut().enumerate() {
        let mut direct: Vec<UnitIx> = unit
            .steps
            .clone()
            .flat_map(|s| deps[s as usize].iter().map(|p| unit_of[*p as usize]))
            .filter(|p| p.0 as usize != u)
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        direct.sort();
        let implied: HashSet<UnitIx> =
            direct.iter().flat_map(|w| ancestors[w.0 as usize].iter().copied()).collect();
        unit.waits = direct.iter().copied().filter(|w| !implied.contains(w)).collect();
        let mut anc = implied;
        anc.extend(direct);
        ancestors.push(anc);
    }

    // Shared locations: written by one unit, touched by another.
    let mut writers: HashMap<L, HashSet<UnitIx>> = HashMap::new();
    let mut touchers: HashMap<L, HashSet<UnitIx>> = HashMap::new();
    for (s, flow) in steps.iter().enumerate() {
        for l in &flow.writes {
            writers.entry(*l).or_default().insert(unit_of[s]);
            touchers.entry(*l).or_default().insert(unit_of[s]);
        }
        for l in &flow.reads {
            touchers.entry(*l).or_default().insert(unit_of[s]);
        }
    }
    let shared = writers
        .iter()
        .filter(|(l, w)| touchers[*l].iter().any(|t| !w.contains(t)) || w.len() > 1)
        .map(|(l, _)| *l)
        .collect();

    MegakernelPlan { units, shared }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn items(n: u32) -> Items {
        Items(NonZeroU32::new(n).expect("nonzero"))
    }
    fn step(reads: &[u8], writes: &[u8], n: u32) -> StepFlow<u8> {
        StepFlow { reads: reads.to_vec(), writes: writes.to_vec(), items: items(n) }
    }

    #[test]
    fn a_dependent_single_item_chain_is_one_unit_with_private_intermediates() {
        // x0 -> x1 -> x2 -> x3, all single-item: one threadgroup, nothing crosses it.
        let p = plan(&[step(&[0], &[1], 1), step(&[1], &[2], 1), step(&[2], &[3], 1)]);
        assert_eq!(p.units, vec![Unit { steps: 0..3, items: Items::ONE, waits: vec![] }]);
        assert!(p.shared.is_empty());
    }

    #[test]
    fn independent_single_item_steps_stay_separate_so_they_run_in_parallel() {
        let p = plan(&[step(&[0], &[1], 1), step(&[0], &[2], 1)]);
        assert_eq!(p.units.len(), 2);
        assert!(p.units.iter().all(|u| u.waits.is_empty()));
    }

    #[test]
    fn a_multi_item_producer_is_its_own_unit_and_its_output_is_shared() {
        // norm (1 item) -> qmv (4 items) -> norm (1 item)
        let p = plan(&[step(&[0], &[1], 1), step(&[1], &[2], 4), step(&[2], &[3], 1)]);
        assert_eq!(p.units.len(), 3);
        assert_eq!(p.units[1].waits, vec![UnitIx(0)]);
        assert_eq!(p.units[2].waits, vec![UnitIx(1)]);
        assert_eq!(p.shared, HashSet::from([1, 2]));
    }

    #[test]
    fn write_after_read_is_a_wait() {
        // unit 0 reads 5; unit 1 (independent of 0 by RAW) overwrites 5.
        let p = plan(&[step(&[5], &[1], 2), step(&[0], &[5], 2)]);
        assert_eq!(p.units[1].waits, vec![UnitIx(0)]);
        assert_eq!(p.shared, HashSet::from([5]));
    }

    #[test]
    fn a_wait_implied_through_another_wait_is_dropped() {
        // 0 -> 1 -> 2 and 0 -> 2 directly: unit 2 waits only on unit 1.
        let p = plan(&[step(&[], &[1], 2), step(&[1], &[2], 2), step(&[1, 2], &[3], 2)]);
        assert_eq!(p.units[2].waits, vec![UnitIx(1)]);
    }

    #[test]
    fn a_chain_absorbs_its_members_external_waits() {
        // multi-item producer of 1, then a single-item chain whose SECOND step also reads 1.
        let p = plan(&[step(&[], &[1], 3), step(&[0], &[2], 1), step(&[1, 2], &[4], 1)]);
        assert_eq!(p.units.len(), 2);
        assert_eq!(p.units[1].steps, 1..3);
        assert_eq!(p.units[1].waits, vec![UnitIx(0)]);
    }
}
