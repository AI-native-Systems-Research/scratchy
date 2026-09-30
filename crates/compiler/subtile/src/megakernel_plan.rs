// SPDX-License-Identifier: Apache-2.0
//! THE MEGAKERNEL PLAN — what one persistent kernel needs to know to play a tape.
//!
//! A megakernel runs a whole tape inside ONE dispatch of `P` persistent threadgroups.
//! Everything that decides how — which steps wait on which, which steps share a threadgroup,
//! how the work splits over the threadgroups, where a grid barrier stands, which memory must be
//! device-coherent — is a fact about the tape's DATAFLOW, so it is computed here, once, at
//! expansion time ([`plan`], then [`schedule`]). The runtime decides nothing.
//!
//! The target states each step's reads and writes in its own location vocabulary (`L`) and how
//! many work items the step splits into; this pass is generic over `L` and knows nothing else.
//!
//! # Units
//!
//! A [`Unit`] is what the schedule places. A multi-item step is a unit of its own. A run of
//! single-item steps where each depends on the run is ONE unit: one threadgroup plays it back to
//! back, so its intermediates never leave the threadgroup — a dependency inside a unit is a
//! threadgroup barrier, not a device-wide handoff. (Measured on M5: 1.64 µs per dependent step
//! in one threadgroup vs 2.95 µs handed across threadgroups vs 3.06 µs as separate dispatches.)
//!
//! # Schedule
//!
//! The units, in tape order, fall into PHASES separated by grid barriers. A multi-item unit
//! spreads its items round-robin over every threadgroup; a single-item unit is pinned to one.
//! A unit joins the open phase unless it depends on that phase in a way one threadgroup cannot
//! order: a pinned unit whose in-phase waits all sit on one lane follows them there. Nothing
//! is claimed and nothing is counted at run time: each threadgroup's program is fixed.
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

/// How many work items a step splits into: one PHYSICAL threadgroup of the persistent kernel
/// plays one item, whatever threadgroup size the step was dispatched with (the target packs
/// several of the step's threadgroups into one item).
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
    /// A grid barrier stands before this step whatever the dataflow: the first step of an
    /// iteration of a loop the target keeps rolled (every iteration must schedule alike), or the
    /// first after the loop. No unit spans it.
    pub phase_break: bool,
    /// One threadgroup may play every item of this step (the target sized its grid small at
    /// expansion): where its waits in the open phase are one threadgroup's, it follows them there
    /// instead of spreading after a grid barrier.
    pub whole: bool,
}

/// A run of consecutive steps one threadgroup plays item by item.
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

    // Units: a single-item step that READS what the open single-item unit wrote joins it. An order
    // alone (write-after-read, write-after-write) passes no data, so it chains nothing: joining
    // would only hold the unit's earlier steps back until the step's other waits are met.
    let mut units: Vec<Unit> = Vec::new();
    let mut unit_of = Vec::with_capacity(steps.len());
    for (s, flow) in (0u32..).zip(steps) {
        let joins = flow.items == Items::ONE
            && !flow.phase_break
            && units.last().is_some_and(|u| {
                u.items == Items::ONE
                    && u.steps.end == s
                    && deps[s as usize].iter().any(|&p| {
                        u.steps.contains(&p)
                            && steps[p as usize]
                                .writes
                                .iter()
                                .any(|l| flow.reads.contains(l))
                    })
            });
        if joins {
            units.last_mut().expect("joins implies a unit").steps.end = s + 1;
        } else {
            units.push(Unit {
                steps: s..s + 1,
                items: flow.items,
                waits: Vec::new(),
            });
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
        let implied: HashSet<UnitIx> = direct
            .iter()
            .flat_map(|w| ancestors[w.0 as usize].iter().copied())
            .collect();
        unit.waits = direct
            .iter()
            .copied()
            .filter(|w| !implied.contains(w))
            .collect();
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

impl<L: Copy + Eq + Hash> MegakernelPlan<L> {
    /// Two units that both touch `loc` (read or write) with neither waiting — directly or
    /// transitively — on the other, if any. A buffer several of the target's logical locations
    /// alias must be totally ordered (`None`), reads included: it holds different contents over
    /// the tape, and which reads see the same contents is not a fact of the plan. `steps` are
    /// the flows the plan was made from.
    pub fn unordered_on(&self, steps: &[StepFlow<L>], loc: L) -> Option<(UnitIx, UnitIx)> {
        let mut ancestors: Vec<HashSet<UnitIx>> = Vec::with_capacity(self.units.len());
        for unit in &self.units {
            let mut anc = HashSet::new();
            for w in &unit.waits {
                anc.insert(*w);
                anc.extend(ancestors[w.0 as usize].iter().copied());
            }
            ancestors.push(anc);
        }
        let touchers: Vec<UnitIx> = (0u32..)
            .zip(&self.units)
            .filter(|(_, u)| {
                let flows = &steps[u.steps.start as usize..u.steps.end as usize];
                flows
                    .iter()
                    .any(|f| f.reads.contains(&loc) || f.writes.contains(&loc))
            })
            .map(|(u, _)| UnitIx(u))
            .collect();
        touchers.iter().enumerate().find_map(|(i, &b)| {
            let unordered = touchers[..i]
                .iter()
                .find(|a| !ancestors[b.0 as usize].contains(a));
            unordered.map(|&a| (a, b))
        })
    }
}

/// A persistent threadgroup by compile-time index: a launch of `P` threadgroups runs lane `l` on
/// threadgroup `l mod P`.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct Lane(pub u32);

/// Where a unit runs in its phase.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Placement {
    /// Every threadgroup takes items round-robin: item `i` runs on threadgroup
    /// `(cursor + i) mod P`, `cursor` counting the items spread earlier in the phase — so
    /// consecutive spread units continue where the last one stopped.
    Spread { cursor: u32 },
    /// One threadgroup plays the unit, every item of it.
    Pinned(Lane),
    /// EVERY threadgroup plays the (single-item) unit, each for itself: a unit that reads nothing it
    /// writes before writing it, so its copies write the same values wherever they run. What reads
    /// only its outputs runs beside it in its phase — each threadgroup reads its own copy's.
    Everywhere,
}

/// A plan's static schedule: its units in phases (consecutive ranges of `plan.units`, a grid
/// barrier between two phases), and where each unit runs in its phase.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Schedule {
    pub phases: Vec<Range<u32>>,
    pub placements: Vec<Placement>,
}

/// Schedule `plan` (made from `steps`) for a launch of `lanes` threadgroups (the balance assumes
/// `lanes`; any `P` runs the schedule correctly). A unit joins the open phase unless it starts at
/// a phase break, or one of its waits is in that phase and cannot be ordered by one threadgroup:
/// a single-item unit — or a [`StepFlow::whole`] one — whose in-phase waits all run on one lane
/// follows them there (and plays every item there); any other multi-item unit, or waits on several
/// lanes, closes the phase — unless every such wait is a single-item unit that only feeds it and
/// that every lane can play for itself: those become [`Placement::Everywhere`] and it joins. A free
/// single-item unit takes the lane with the least work in the phase (items spread there plus steps
/// and items pinned there).
pub fn schedule<L: PartialEq>(
    plan: &MegakernelPlan<L>,
    steps: &[StepFlow<L>],
    lanes: u32,
) -> Schedule {
    let lanes = lanes.max(1);
    // The first unit of every phase; the open phase is the last.
    let mut starts = vec![0u32];
    let mut placements: Vec<Placement> = Vec::with_capacity(plan.units.len());
    let mut phase_of: Vec<usize> = Vec::with_capacity(plan.units.len());
    let mut load = vec![0u64; lanes as usize];
    let mut cursor = 0u32;
    for (u, unit) in (0u32..).zip(&plan.units) {
        let open = starts.len() - 1;
        let waits: Vec<UnitIx> = unit
            .waits
            .iter()
            .filter(|w| phase_of[w.0 as usize] == open)
            .copied()
            .collect();
        // Every in-phase wait a copy on each lane satisfies — a single-item producer this unit
        // only reads from, whose own in-phase producers are copies too, and that nothing placed
        // after it in the phase writes over — lets the unit join the phase, those producers copied.
        let start = starts[open] as usize;
        let local = |w: &UnitIx| {
            let p = &plan.units[w.0 as usize];
            let fed = p.waits.iter().all(|x| {
                phase_of[x.0 as usize] != open
                    || (placements[x.0 as usize] == Placement::Everywhere
                        && only_reads(p, &plan.units[x.0 as usize], steps))
            });
            let kept = plan.units[start..u as usize]
                .iter()
                .skip_while(|q| q.steps.start <= p.steps.start)
                .all(|q| only_reads(q, p, steps));
            let can = placements[w.0 as usize] == Placement::Everywhere
                || (p.items == Items::ONE && fed && kept && replicable(p, steps));
            can && only_reads(unit, p, steps)
        };
        let in_phase: Vec<Placement> = waits.iter().map(|w| placements[w.0 as usize]).collect();
        let single = unit.items == Items::ONE;
        let whole = steps[unit.steps.start as usize].whole;
        let broken = u > 0 && steps[unit.steps.start as usize].phase_break;
        let follows = match in_phase.as_slice() {
            _ if broken => None,
            [] => Some(None),
            [Placement::Pinned(l), rest @ ..]
                if (single || whole) && rest.iter().all(|p| *p == Placement::Pinned(*l)) =>
            {
                Some(Some(*l))
            }
            _ if waits.iter().all(local) => {
                waits
                    .iter()
                    .for_each(|w| placements[w.0 as usize] = Placement::Everywhere);
                Some(None)
            }
            _ => None,
        };
        let follows = follows.unwrap_or_else(|| {
            starts.push(u);
            load.iter_mut().for_each(|w| *w = 0);
            cursor = 0;
            None
        });
        let pinned = single || (whole && follows.is_some());
        let placement = if pinned {
            let lane = follows.unwrap_or_else(|| {
                let least = (0..lanes).min_by_key(|&l| load[l as usize]);
                Lane(least.expect("at least one lane"))
            });
            let work = match single {
                true => unit.steps.end - unit.steps.start,
                false => unit.items.0.get(),
            };
            load[lane.0 as usize] += u64::from(work);
            Placement::Pinned(lane)
        } else {
            let items = unit.items.0.get();
            for (l, w) in (0u32..).zip(load.iter_mut()) {
                let first = (l + lanes - cursor % lanes) % lanes;
                *w += u64::from(items.saturating_sub(first).div_ceil(lanes));
            }
            let at = cursor;
            cursor += items;
            Placement::Spread { cursor: at }
        };
        phase_of.push(starts.len() - 1);
        placements.push(placement);
    }
    let ends = starts
        .iter()
        .skip(1)
        .copied()
        .chain([plan.units.len() as u32]);
    let phases = starts.iter().zip(ends).map(|(&s, e)| s..e).collect();
    Schedule { phases, placements }
}

/// A unit whose copies may run concurrently: none of its steps reads, before the unit wrote it, a
/// location the unit writes (an in-place update read by one copy after another copy updated it).
fn replicable<L: PartialEq>(unit: &Unit, steps: &[StepFlow<L>]) -> bool {
    let flows = &steps[unit.steps.start as usize..unit.steps.end as usize];
    let mut written: Vec<&L> = Vec::new();
    for f in flows {
        let early = |r: &L| !written.contains(&r) && flows.iter().any(|g| g.writes.contains(r));
        if f.reads.iter().any(early) {
            return false;
        }
        written.extend(&f.writes);
    }
    true
}

/// `consumer` only reads from `producer`: it writes nothing `producer` reads or writes (which a
/// copy of the producer on another lane could still be touching).
fn only_reads<L: PartialEq>(consumer: &Unit, producer: &Unit, steps: &[StepFlow<L>]) -> bool {
    let of = |u: &Unit| &steps[u.steps.start as usize..u.steps.end as usize];
    let touched = |l: &L| {
        of(producer)
            .iter()
            .any(|f| f.reads.contains(l) || f.writes.contains(l))
    };
    of(consumer).iter().all(|f| !f.writes.iter().any(touched))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn items(n: u32) -> Items {
        Items(NonZeroU32::new(n).expect("nonzero"))
    }
    fn step(reads: &[u8], writes: &[u8], n: u32) -> StepFlow<u8> {
        StepFlow {
            reads: reads.to_vec(),
            writes: writes.to_vec(),
            items: items(n),
            phase_break: false,
            whole: false,
        }
    }

    #[test]
    fn independent_units_share_a_phase_and_spread_items_continue_round_robin() {
        // two independent multi-item steps, then a single-item one: one phase, no barrier.
        let steps = [
            step(&[0], &[1], 3),
            step(&[0], &[2], 4),
            step(&[0], &[3], 1),
        ];
        let s = schedule(&plan(&steps), &steps, 4);
        assert_eq!(s.phases, vec![0..3]);
        assert_eq!(s.placements[0], Placement::Spread { cursor: 0 });
        assert_eq!(s.placements[1], Placement::Spread { cursor: 3 });
        // lanes 0..3 carry 2, 2, 2, 1 items: the pinned unit takes lane 3.
        assert_eq!(s.placements[2], Placement::Pinned(Lane(3)));
    }

    #[test]
    fn a_wait_on_the_open_phase_is_a_grid_barrier_unless_one_lane_orders_it() {
        // in-place norm (1) -> qmv (4 items) -> in-place norm (1) -> side chain step (1) reading
        // the first norm. (In place, no norm can be copied onto every lane.)
        let steps = [
            step(&[0, 1], &[1], 1),
            step(&[1], &[2], 4),
            step(&[2, 3], &[3], 1),
            step(&[3], &[4], 4),
            step(&[1, 5], &[6], 1),
        ];
        let s = schedule(&plan(&steps), &steps, 4);
        assert_eq!(s.phases, vec![0..1, 1..2, 2..3, 3..5]);
        // The last unit waits only on unit 0 (phase 0): it runs beside the qmv of phase 3.
        assert_eq!(s.placements[3], Placement::Spread { cursor: 0 });
        assert!(matches!(s.placements[4], Placement::Pinned(_)));
        // A single-item unit waiting on a pinned unit of the open phase follows it.
        let steps = [
            step(&[0], &[1], 1),
            step(&[9], &[8], 2),
            step(&[1], &[2], 1),
        ];
        let q = plan(&steps);
        assert_eq!(q.units.len(), 3);
        let t = schedule(&q, &steps, 4);
        assert_eq!(t.phases, vec![0..3]);
        assert_eq!(t.placements[2], t.placements[0]);
    }

    #[test]
    fn a_dependent_single_item_chain_is_one_unit_with_private_intermediates() {
        // x0 -> x1 -> x2 -> x3, all single-item: one threadgroup, nothing crosses it.
        let p = plan(&[
            step(&[0], &[1], 1),
            step(&[1], &[2], 1),
            step(&[2], &[3], 1),
        ]);
        assert_eq!(
            p.units,
            vec![Unit {
                steps: 0..3,
                items: Items::ONE,
                waits: vec![]
            }]
        );
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
        let p = plan(&[
            step(&[0], &[1], 1),
            step(&[1], &[2], 4),
            step(&[2], &[3], 1),
        ]);
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
        let p = plan(&[
            step(&[], &[1], 2),
            step(&[1], &[2], 2),
            step(&[1, 2], &[3], 2),
        ]);
        assert_eq!(p.units[2].waits, vec![UnitIx(1)]);
    }

    #[test]
    fn a_chain_absorbs_its_members_external_waits() {
        // multi-item producer of 1, then a single-item chain whose SECOND step also reads 1.
        let p = plan(&[
            step(&[], &[1], 3),
            step(&[0], &[2], 1),
            step(&[1, 2], &[4], 1),
        ]);
        assert_eq!(p.units.len(), 2);
        assert_eq!(p.units[1].steps, 1..3);
        assert_eq!(p.units[1].waits, vec![UnitIx(0)]);
    }

    #[test]
    fn an_order_without_data_does_not_chain() {
        // A single-item step, then one that overwrites what it read (write-after-read) and reads
        // a multi-item producer's output: the second waits on the first but does not join it, so
        // the first is not held back until the producer is done.
        let steps = [step(&[], &[2], 3), step(&[0], &[1], 1), step(&[2], &[0], 1)];
        let p = plan(&steps);
        assert_eq!(p.units.len(), 3);
        assert_eq!(p.units[1].waits, vec![]);
        assert_eq!(p.units[2].waits, vec![UnitIx(0), UnitIx(1)]);
        // The first runs beside the producer; the second after the grid barrier the producer's
        // spread items need.
        let s = schedule(&p, &steps, 4);
        assert_eq!(s.phases, vec![0..2, 2..3]);
    }

    #[test]
    fn a_phase_break_splits_a_chain_and_opens_a_phase_even_without_a_wait() {
        // a single-item chain across a break, then an independent step after a second break.
        let mut steps = [
            step(&[0], &[1], 1),
            step(&[1], &[2], 1),
            step(&[9], &[8], 3),
        ];
        steps[1].phase_break = true;
        steps[2].phase_break = true;
        let p = plan(&steps);
        assert_eq!(p.units.len(), 3);
        assert_eq!(p.units[1].waits, vec![UnitIx(0)]);
        let s = schedule(&p, &steps, 4);
        assert_eq!(s.phases, vec![0..1, 1..2, 2..3]);
        // Without the breaks: one chain, and the independent step beside it.
        let free = [steps[0].clone(), step(&[1], &[2], 1), step(&[9], &[8], 3)];
        let q = plan(&free);
        assert_eq!(q.units.len(), 2);
        assert_eq!(schedule(&q, &free, 4).phases, vec![0..2]);
    }

    #[test]
    fn a_whole_step_follows_the_lane_it_waits_on_and_otherwise_spreads() {
        // norm (1) -> small elementwise (3 items, whole) -> its reader (1): one lane, one phase.
        let mut steps = [
            step(&[0], &[1], 1),
            step(&[1], &[2], 3),
            step(&[2], &[3], 1),
        ];
        steps[1].whole = true;
        let s = schedule(&plan(&steps), &steps, 4);
        assert_eq!(s.phases, vec![0..3]);
        assert!(s.placements.iter().all(|p| *p == s.placements[0]));
        // Not whole: the elementwise step spreads — beside a copy of the norm on every lane — and
        // its reader follows after a grid barrier.
        steps[1].whole = false;
        let t = schedule(&plan(&steps), &steps, 4);
        assert_eq!(t.phases, vec![0..2, 2..3]);
        assert_eq!(t.placements[0], Placement::Everywhere);
        // Whole but waiting on a spread step: it spreads after the barrier as any multi-item step.
        let mut spread = [step(&[0], &[1], 4), step(&[1], &[2], 3)];
        spread[1].whole = true;
        let w = schedule(&plan(&spread), &spread, 4);
        assert_eq!(w.phases, vec![0..1, 1..2]);
        assert_eq!(w.placements[1], Placement::Spread { cursor: 0 });
    }

    #[test]
    fn a_producer_every_lane_can_copy_saves_the_barrier_its_spread_readers_need() {
        // norm (1, out of place) -> qmv (4 items): the norm runs on every lane, one phase.
        let steps = [step(&[0], &[1], 1), step(&[1], &[2], 4)];
        let s = schedule(&plan(&steps), &steps, 4);
        assert_eq!(s.phases, vec![0..2]);
        assert_eq!(s.placements[0], Placement::Everywhere);
        // In place (it reads what it writes): a copy could read another copy's result.
        let in_place = [step(&[0, 1], &[1], 1), step(&[1], &[2], 4)];
        let t = schedule(&plan(&in_place), &in_place, 4);
        assert_eq!(t.phases, vec![0..1, 1..2]);
        assert!(matches!(t.placements[0], Placement::Pinned(_)));
        // The reader also overwrites the norm's input: another lane's copy could still read it.
        let clobber = [step(&[0], &[1], 1), step(&[1], &[2, 0], 4)];
        let w = schedule(&plan(&clobber), &clobber, 4);
        assert_eq!(w.phases, vec![0..1, 1..2]);
        // A single-item chain that writes its own intermediate and reads it back is copied whole.
        let chain = [
            step(&[0], &[1], 1),
            step(&[1], &[2], 1),
            step(&[2], &[3], 4),
        ];
        let c = plan(&chain);
        assert_eq!(c.units.len(), 2);
        let x = schedule(&c, &chain, 4);
        assert_eq!(x.phases, vec![0..2]);
        assert_eq!(x.placements[0], Placement::Everywhere);
    }

    #[test]
    fn a_location_is_totally_ordered_only_when_every_two_touchers_are() {
        // 9 is one physical buffer: written, read, written again — each access waits on the last.
        let serial = [
            step(&[], &[9], 2),
            step(&[9], &[1], 2),
            step(&[0], &[2], 2),
            step(&[1], &[9], 2),
        ];
        assert_eq!(plan(&serial).unordered_on(&serial, 9), None);
        // Two readers of what one writer left wait on the writer, not on each other.
        let fan = [step(&[], &[9], 2), step(&[9], &[1], 2), step(&[9], &[2], 2)];
        let p = plan(&fan);
        assert_eq!(p.unordered_on(&fan, 9), Some((UnitIx(1), UnitIx(2))));
        // A location nothing touches is trivially ordered.
        assert_eq!(p.unordered_on(&fan, 7), None);
    }

    #[test]
    fn the_commands_of_one_row_are_ordered_by_the_rows_writes() {
        // One row realized as two commands (e.g. a split matmul and its reduce), both carrying
        // the row's whole dataflow, then a reader of the row's output.
        let row = || step(&[0], &[1, 2], 3);
        let units = plan(&[row(), row(), step(&[2], &[3], 3)]).units;
        assert_eq!(units.len(), 3);
        assert_eq!(units[1].waits, vec![UnitIx(0)]);
        // The reader waits on the row's LAST command; the first is implied through it.
        assert_eq!(units[2].waits, vec![UnitIx(1)]);
    }
}
