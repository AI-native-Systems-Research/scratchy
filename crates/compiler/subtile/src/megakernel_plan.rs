// SPDX-License-Identifier: Apache-2.0
//! THE FUSION PLAN — how a tape's steps fuse into generated kernels, one launch each.
//!
//! A tape's steps are played as SEGMENTS: runs of steps one generated kernel plays in ONE launch
//! of `P` threadgroups, a launch boundary between two segments. Everything that decides how —
//! which steps wait on which, which steps share a threadgroup, how the work splits over the
//! threadgroups, where a segment ends — is a fact about the tape's DATAFLOW, so it is computed
//! here, once, at expansion time ([`plan`], then [`segment`]). The runtime decides nothing.
//!
//! The target states each step's reads and writes in its own location vocabulary (`L`) and how
//! many work items the step splits into; this pass is generic over `L` and knows nothing else.
//!
//! # Units
//!
//! A [`Unit`] is what a segment places. A multi-item step is a unit of its own. A run of
//! single-item steps where each depends on the run is ONE unit: one threadgroup plays it back to
//! back, so its intermediates never leave the threadgroup — a dependency inside a unit is a
//! threadgroup barrier, not a launch boundary. (Measured on M5: 1.64 µs per dependent step in one
//! threadgroup vs 3.06 µs as separate dispatches.)
//!
//! # Segments
//!
//! The units, in tape order, fall into [`Segments`]. A multi-item unit spreads its items
//! round-robin over the launch's threadgroups; a single-item unit is pinned to one. A unit joins
//! the open segment only where ONE threadgroup orders each of its waits on it
//! ([`LocalWait`]): it follows what it waits on onto that unit's lane, or reads a producer every
//! lane plays for itself. Anything else ends the segment. So no generated kernel ever waits on
//! another threadgroup — nothing spins, nothing needs every threadgroup of a launch running at
//! once, and the GPU may preempt between any two launches. [`Segments`] is built only by
//! [`segment`], which rejects a segment holding any other wait.

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

/// How many work items a step splits into: one threadgroup of a segment's launch plays one item,
/// whatever threadgroup size the step was dispatched with (the target packs several of the step's
/// threadgroups into one item).
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
    /// A segment boundary stands before this step whatever the dataflow: the first step of an
    /// iteration of a loop the target keeps rolled (every iteration must segment alike), or the
    /// first after the loop. No unit spans it.
    pub segment_break: bool,
    /// One threadgroup may play every item of this step (the target sized its grid small at
    /// expansion): where its waits in the open segment are one threadgroup's, it follows them there
    /// instead of spreading in a segment of its own.
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

/// A tape's units, in tape order, each with what it waits on.
#[derive(Clone, Debug)]
pub struct FusionPlan {
    pub units: Vec<Unit>,
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

/// Plan a tape (or one contiguous run of it) for fusion.
pub fn plan<L: Copy + Eq + Hash>(steps: &[StepFlow<L>]) -> FusionPlan {
    let deps = step_deps(steps);

    // Units: a single-item step that READS what the open single-item unit wrote joins it. An order
    // alone (write-after-read, write-after-write) passes no data, so it chains nothing: joining
    // would only hold the unit's earlier steps back until the step's other waits are met.
    let mut units: Vec<Unit> = Vec::new();
    let mut unit_of = Vec::with_capacity(steps.len());
    for (s, flow) in (0u32..).zip(steps) {
        let joins = flow.items == Items::ONE
            && !flow.segment_break
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

    FusionPlan { units }
}

impl FusionPlan {
    /// Two units that both touch `loc` (read or write) with neither waiting — directly or
    /// transitively — on the other, if any. A buffer several of the target's logical locations
    /// alias must be totally ordered (`None`), reads included: it holds different contents over
    /// the tape, and which reads see the same contents is not a fact of the plan. `steps` are
    /// the flows the plan was made from.
    pub fn unordered_on<L: PartialEq>(
        &self,
        steps: &[StepFlow<L>],
        loc: L,
    ) -> Option<(UnitIx, UnitIx)> {
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

/// A threadgroup of a segment's launch by compile-time index: a launch of `P` threadgroups runs
/// lane `l` on threadgroup `l mod P`.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct Lane(pub u32);

/// Where a unit runs in its segment.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Placement {
    /// Every threadgroup takes items round-robin: item `i` runs on threadgroup
    /// `(cursor + i) mod P`, `cursor` counting the items spread earlier in the segment — so
    /// consecutive spread units continue where the last one stopped.
    Spread { cursor: u32 },
    /// One threadgroup plays the unit, every item of it.
    Pinned(Lane),
    /// EVERY threadgroup plays the (single-item) unit, each for itself: a unit that reads nothing it
    /// writes before writing it, so its copies write the same values wherever they run. What reads
    /// only its outputs runs beside it in its segment — each threadgroup reads its own copy's.
    Everywhere,
}

/// How a unit's wait on a unit of its OWN segment is met: by one threadgroup — the only order a
/// segment has. A wait of any other kind cannot stand inside a segment ([`Segments`]).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LocalWait {
    /// Both run on this lane, the waiter after the unit it waits on.
    SameLane(Lane),
    /// The unit waited on runs on every lane: each lane's waiter reads its own lane's copy.
    OwnCopy,
}

/// Why a segment starts where it does.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Cut {
    /// The tape's first segment.
    Start,
    /// The target asked for a boundary before its first step ([`StepFlow::segment_break`]).
    Break,
    /// Its first unit waits on these units of the segment before it, and one threadgroup cannot
    /// order that: a result spread over threadgroups, results on several lanes, or one no copy
    /// can stand in for.
    Waits(Vec<UnitIx>),
}

/// One launch's units: a consecutive range of the plan's units, and why it starts there.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Segment {
    pub units: Range<u32>,
    pub cut: Cut,
}

/// A plan cut into segments, and where each unit runs in its segment. Built only by [`segment`],
/// which proves every wait inside a segment is a [`LocalWait`]: there is no way to hold a
/// segment in which a unit waits on another threadgroup.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Segments {
    segments: Vec<Segment>,
    placements: Vec<Placement>,
    /// Per unit: its waits on units of its own segment, each met by one threadgroup.
    local: Vec<Vec<(UnitIx, LocalWait)>>,
}

/// Why units cannot form the segments asked for.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SegmentError {
    /// `unit` waits on `on` inside one segment, and no one threadgroup plays both.
    CrossThreadgroupWait { unit: UnitIx, on: UnitIx },
    /// `unit` is played on every threadgroup, but its copies would not write the same values (it
    /// reads what it writes before writing it — an in-place update), or it has several items.
    NotCopyable { unit: UnitIx },
    /// The segments do not cover the plan's units in order, each once.
    NotATiling,
}

impl std::fmt::Display for SegmentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CrossThreadgroupWait { unit, on } => write!(
                f,
                "unit {} waits on unit {} inside one segment, and no one threadgroup plays both",
                unit.0, on.0
            ),
            Self::NotCopyable { unit } => {
                write!(
                    f,
                    "unit {} is played on every threadgroup but cannot be",
                    unit.0
                )
            }
            Self::NotATiling => write!(f, "segments do not cover the units in order"),
        }
    }
}

impl std::error::Error for SegmentError {}

impl Segments {
    /// `segments` over `plan`'s units (made from `steps`) with `placements`, every unit played on
    /// every threadgroup copyable and every wait inside a segment typed — or the first that is
    /// not.
    fn new<L: PartialEq>(
        plan: &FusionPlan,
        steps: &[StepFlow<L>],
        segments: Vec<Segment>,
        placements: Vec<Placement>,
    ) -> Result<Self, SegmentError> {
        let n = plan.units.len() as u32;
        let tiles = segments
            .iter()
            .map(|s| s.units.clone())
            .try_fold(0, |at, r| {
                (r.start == at && r.end > r.start).then_some(r.end)
            });
        if tiles != Some(n) || placements.len() != plan.units.len() {
            return Err(SegmentError::NotATiling);
        }
        let copied = (0u32..).zip(&placements).zip(&plan.units);
        if let Some(((u, _), _)) = copied
            .filter(|((_, p), _)| **p == Placement::Everywhere)
            .find(|(_, unit)| !copyable(unit, steps))
        {
            return Err(SegmentError::NotCopyable { unit: UnitIx(u) });
        }
        let mut local = vec![Vec::new(); plan.units.len()];
        for s in &segments {
            for u in s.units.clone() {
                let inside = plan.units[u as usize].waits.iter();
                for &w in inside.filter(|w| s.units.contains(&w.0)) {
                    let wait = match (placements[u as usize], placements[w.0 as usize]) {
                        (_, Placement::Everywhere) => LocalWait::OwnCopy,
                        (Placement::Pinned(a), Placement::Pinned(b)) if a == b => {
                            LocalWait::SameLane(a)
                        }
                        _ => {
                            return Err(SegmentError::CrossThreadgroupWait {
                                unit: UnitIx(u),
                                on: w,
                            });
                        }
                    };
                    local[u as usize].push((w, wait));
                }
            }
        }
        Ok(Self {
            segments,
            placements,
            local,
        })
    }

    /// The segments, in tape order: consecutive ranges of the plan's units.
    pub fn segments(&self) -> &[Segment] {
        &self.segments
    }

    /// Per unit: where it runs in its segment.
    pub fn placements(&self) -> &[Placement] {
        &self.placements
    }

    /// `u`'s waits on units of its own segment, each with the one threadgroup that meets it.
    pub fn local_waits(&self, u: UnitIx) -> &[(UnitIx, LocalWait)] {
        &self.local[u.0 as usize]
    }
}

/// Segment `plan` (made from `steps`) for launches of `lanes` threadgroups (the balance assumes
/// `lanes`; any `P` plays the segments correctly). A unit joins the open segment unless it starts
/// at a segment break, or one of its waits is in that segment and cannot be ordered by one
/// threadgroup: a single-item unit — or a [`StepFlow::whole`] one — whose in-segment waits all run
/// on one lane follows them there (and plays every item there); any other multi-item unit, or
/// waits on several lanes, closes the segment — unless every such wait is a single-item unit that
/// only feeds it and that every lane can play for itself: those become [`Placement::Everywhere`]
/// and it joins. A free single-item unit takes the lane with the least work in the segment (items
/// spread there plus steps and items pinned there).
pub fn segment<L: PartialEq>(
    plan: &FusionPlan,
    steps: &[StepFlow<L>],
    lanes: u32,
) -> Result<Segments, SegmentError> {
    let lanes = lanes.max(1);
    // The first unit of every segment and why it starts there; the open segment is the last.
    let mut starts = vec![(0u32, Cut::Start)];
    let mut placements: Vec<Placement> = Vec::with_capacity(plan.units.len());
    let mut segment_of: Vec<usize> = Vec::with_capacity(plan.units.len());
    let mut load = vec![0u64; lanes as usize];
    let mut cursor = 0u32;
    for (u, unit) in (0u32..).zip(&plan.units) {
        let open = starts.len() - 1;
        let waits: Vec<UnitIx> = unit
            .waits
            .iter()
            .filter(|w| segment_of[w.0 as usize] == open)
            .copied()
            .collect();
        // Every in-segment wait a copy on each lane satisfies — a single-item producer this unit
        // only reads from, whose own in-segment producers are copies too, and that nothing placed
        // after it in the segment writes over — lets the unit join the segment, those producers
        // copied.
        let start = starts[open].0 as usize;
        let local = |w: &UnitIx| {
            let p = &plan.units[w.0 as usize];
            let fed = p.waits.iter().all(|x| {
                segment_of[x.0 as usize] != open
                    || (placements[x.0 as usize] == Placement::Everywhere
                        && only_reads(p, &plan.units[x.0 as usize], steps))
            });
            let kept = plan.units[start..u as usize]
                .iter()
                .skip_while(|q| q.steps.start <= p.steps.start)
                .all(|q| only_reads(q, p, steps));
            let can = placements[w.0 as usize] == Placement::Everywhere
                || (fed && kept && copyable(p, steps));
            can && only_reads(unit, p, steps)
        };
        let in_segment: Vec<Placement> = waits.iter().map(|w| placements[w.0 as usize]).collect();
        let single = unit.items == Items::ONE;
        let whole = steps[unit.steps.start as usize].whole;
        let broken = u > 0 && steps[unit.steps.start as usize].segment_break;
        let follows = match in_segment.as_slice() {
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
            let cut = match broken {
                true => Cut::Break,
                false => Cut::Waits(waits.clone()),
            };
            starts.push((u, cut));
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
        segment_of.push(starts.len() - 1);
        placements.push(placement);
    }
    let ends = starts
        .iter()
        .skip(1)
        .map(|(s, _)| *s)
        .chain([plan.units.len() as u32]);
    let segments = starts
        .iter()
        .zip(ends)
        .map(|((s, cut), e)| Segment {
            units: *s..e,
            cut: cut.clone(),
        })
        .collect();
    Segments::new(plan, steps, segments, placements)
}

/// A unit every threadgroup may play for itself: a single item, whose copies write the same
/// values ([`replicable`]).
fn copyable<L: PartialEq>(unit: &Unit, steps: &[StepFlow<L>]) -> bool {
    unit.items == Items::ONE && replicable(unit, steps)
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
            segment_break: false,
            whole: false,
        }
    }
    fn cut(steps: &[StepFlow<u8>], lanes: u32) -> Segments {
        segment(&plan(steps), steps, lanes).expect("every wait inside a segment is local")
    }
    fn ranges(s: &Segments) -> Vec<Range<u32>> {
        s.segments().iter().map(|s| s.units.clone()).collect()
    }

    #[test]
    fn independent_units_share_a_segment_and_spread_items_continue_round_robin() {
        // two independent multi-item steps, then a single-item one: one segment, one launch.
        let steps = [
            step(&[0], &[1], 3),
            step(&[0], &[2], 4),
            step(&[0], &[3], 1),
        ];
        let s = cut(&steps, 4);
        assert_eq!(ranges(&s), vec![0..3]);
        assert_eq!(s.segments()[0].cut, Cut::Start);
        assert_eq!(s.placements()[0], Placement::Spread { cursor: 0 });
        assert_eq!(s.placements()[1], Placement::Spread { cursor: 3 });
        // lanes 0..3 carry 2, 2, 2, 1 items: the pinned unit takes lane 3.
        assert_eq!(s.placements()[2], Placement::Pinned(Lane(3)));
    }

    #[test]
    fn a_wait_on_the_open_segment_ends_it_unless_one_lane_orders_it() {
        // in-place norm (1) -> qmv (4 items) -> in-place norm (1) -> side chain step (1) reading
        // the first norm. (In place, no norm can be copied onto every lane.)
        let steps = [
            step(&[0, 1], &[1], 1),
            step(&[1], &[2], 4),
            step(&[2, 3], &[3], 1),
            step(&[3], &[4], 4),
            step(&[1, 5], &[6], 1),
        ];
        let s = cut(&steps, 4);
        assert_eq!(ranges(&s), vec![0..1, 1..2, 2..3, 3..5]);
        let cuts: Vec<&Cut> = s.segments().iter().map(|s| &s.cut).collect();
        assert_eq!(
            cuts,
            [
                &Cut::Start,
                &Cut::Waits(vec![UnitIx(0)]),
                &Cut::Waits(vec![UnitIx(1)]),
                &Cut::Waits(vec![UnitIx(2)])
            ]
        );
        // The last unit waits only on unit 0 (segment 0): it runs beside the qmv of segment 3.
        assert_eq!(s.placements()[3], Placement::Spread { cursor: 0 });
        assert!(matches!(s.placements()[4], Placement::Pinned(_)));
        assert!(s.local_waits(UnitIx(4)).is_empty());
        // A single-item unit waiting on a pinned unit of the open segment follows it.
        let steps = [
            step(&[0], &[1], 1),
            step(&[9], &[8], 2),
            step(&[1], &[2], 1),
        ];
        let q = plan(&steps);
        assert_eq!(q.units.len(), 3);
        let t = segment(&q, &steps, 4).expect("local");
        assert_eq!(ranges(&t), vec![0..3]);
        assert_eq!(t.placements()[2], t.placements()[0]);
        let Placement::Pinned(lane) = t.placements()[0] else {
            panic!("pinned");
        };
        assert_eq!(
            t.local_waits(UnitIx(2)),
            &[(UnitIx(0), LocalWait::SameLane(lane))]
        );
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
    }

    #[test]
    fn independent_single_item_steps_stay_separate_so_they_run_in_parallel() {
        let p = plan(&[step(&[0], &[1], 1), step(&[0], &[2], 1)]);
        assert_eq!(p.units.len(), 2);
        assert!(p.units.iter().all(|u| u.waits.is_empty()));
    }

    #[test]
    fn a_multi_item_producer_is_its_own_unit() {
        // norm (1 item) -> qmv (4 items) -> norm (1 item)
        let p = plan(&[
            step(&[0], &[1], 1),
            step(&[1], &[2], 4),
            step(&[2], &[3], 1),
        ]);
        assert_eq!(p.units.len(), 3);
        assert_eq!(p.units[1].waits, vec![UnitIx(0)]);
        assert_eq!(p.units[2].waits, vec![UnitIx(1)]);
    }

    #[test]
    fn write_after_read_is_a_wait() {
        // unit 0 reads 5; unit 1 (independent of 0 by RAW) overwrites 5.
        let p = plan(&[step(&[5], &[1], 2), step(&[0], &[5], 2)]);
        assert_eq!(p.units[1].waits, vec![UnitIx(0)]);
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
        // The first runs beside the producer; the second in the next segment, after the launch
        // the producer's spread items need.
        let s = segment(&p, &steps, 4).expect("local");
        assert_eq!(ranges(&s), vec![0..2, 2..3]);
        assert_eq!(s.segments()[1].cut, Cut::Waits(vec![UnitIx(0), UnitIx(1)]));
    }

    #[test]
    fn a_segment_break_splits_a_chain_and_opens_a_segment_even_without_a_wait() {
        // a single-item chain across a break, then an independent step after a second break.
        let mut steps = [
            step(&[0], &[1], 1),
            step(&[1], &[2], 1),
            step(&[9], &[8], 3),
        ];
        steps[1].segment_break = true;
        steps[2].segment_break = true;
        let p = plan(&steps);
        assert_eq!(p.units.len(), 3);
        assert_eq!(p.units[1].waits, vec![UnitIx(0)]);
        let s = segment(&p, &steps, 4).expect("local");
        assert_eq!(ranges(&s), vec![0..1, 1..2, 2..3]);
        assert!(s.segments()[1..].iter().all(|s| s.cut == Cut::Break));
        // Without the breaks: one chain, and the independent step beside it.
        let free = [steps[0].clone(), step(&[1], &[2], 1), step(&[9], &[8], 3)];
        let q = plan(&free);
        assert_eq!(q.units.len(), 2);
        assert_eq!(ranges(&segment(&q, &free, 4).expect("local")), vec![0..2]);
    }

    #[test]
    fn a_whole_step_follows_the_lane_it_waits_on_and_otherwise_spreads() {
        // norm (1) -> small elementwise (3 items, whole) -> its reader (1): one lane, one segment.
        let mut steps = [
            step(&[0], &[1], 1),
            step(&[1], &[2], 3),
            step(&[2], &[3], 1),
        ];
        steps[1].whole = true;
        let s = cut(&steps, 4);
        assert_eq!(ranges(&s), vec![0..3]);
        assert!(s.placements().iter().all(|p| *p == s.placements()[0]));
        // Not whole: the elementwise step spreads — beside a copy of the norm on every lane — and
        // its reader follows in the next segment.
        steps[1].whole = false;
        let t = cut(&steps, 4);
        assert_eq!(ranges(&t), vec![0..2, 2..3]);
        assert_eq!(t.placements()[0], Placement::Everywhere);
        assert_eq!(t.local_waits(UnitIx(1)), &[(UnitIx(0), LocalWait::OwnCopy)]);
        // Whole but waiting on a spread step: it spreads in a segment of its own, as any
        // multi-item step.
        let mut spread = [step(&[0], &[1], 4), step(&[1], &[2], 3)];
        spread[1].whole = true;
        let w = cut(&spread, 4);
        assert_eq!(ranges(&w), vec![0..1, 1..2]);
        assert_eq!(w.placements()[1], Placement::Spread { cursor: 0 });
    }

    #[test]
    fn a_producer_every_lane_can_copy_saves_the_launch_its_spread_readers_need() {
        // norm (1, out of place) -> qmv (4 items): the norm runs on every lane, one segment.
        let steps = [step(&[0], &[1], 1), step(&[1], &[2], 4)];
        let s = cut(&steps, 4);
        assert_eq!(ranges(&s), vec![0..2]);
        assert_eq!(s.placements()[0], Placement::Everywhere);
        // In place (it reads what it writes): a copy could read another copy's result.
        let in_place = [step(&[0, 1], &[1], 1), step(&[1], &[2], 4)];
        let t = cut(&in_place, 4);
        assert_eq!(ranges(&t), vec![0..1, 1..2]);
        assert!(matches!(t.placements()[0], Placement::Pinned(_)));
        // The reader also overwrites the norm's input: another lane's copy could still read it.
        let clobber = [step(&[0], &[1], 1), step(&[1], &[2, 0], 4)];
        let w = cut(&clobber, 4);
        assert_eq!(ranges(&w), vec![0..1, 1..2]);
        // A single-item chain that writes its own intermediate and reads it back is copied whole.
        let chain = [
            step(&[0], &[1], 1),
            step(&[1], &[2], 1),
            step(&[2], &[3], 4),
        ];
        let c = plan(&chain);
        assert_eq!(c.units.len(), 2);
        let x = segment(&c, &chain, 4).expect("local");
        assert_eq!(ranges(&x), vec![0..2]);
        assert_eq!(x.placements()[0], Placement::Everywhere);
    }

    /// No segment holds a wait one threadgroup does not order: a spread producer read inside its
    /// own segment, a pinned producer read on another lane, or a copied unit reading a pinned one
    /// — each is refused, naming the wait.
    #[test]
    fn a_wait_no_one_threadgroup_orders_cannot_stand_inside_a_segment() {
        let steps = [step(&[0], &[1], 4), step(&[1], &[2], 1)];
        let p = plan(&steps);
        let one = || {
            vec![Segment {
                units: 0..2,
                cut: Cut::Start,
            }]
        };
        let refused = Err(SegmentError::CrossThreadgroupWait {
            unit: UnitIx(1),
            on: UnitIx(0),
        });
        let spread = vec![
            Placement::Spread { cursor: 0 },
            Placement::Spread { cursor: 4 },
        ];
        assert_eq!(Segments::new(&p, &steps, one(), spread), refused);
        let lanes = vec![Placement::Pinned(Lane(0)), Placement::Pinned(Lane(1))];
        assert_eq!(Segments::new(&p, &steps, one(), lanes), refused);
        let copied = vec![Placement::Pinned(Lane(0)), Placement::Everywhere];
        assert_eq!(Segments::new(&p, &steps, one(), copied), refused);
        // The same waits across a launch boundary stand.
        let two = vec![
            Segment {
                units: 0..1,
                cut: Cut::Start,
            },
            Segment {
                units: 1..2,
                cut: Cut::Waits(vec![UnitIx(0)]),
            },
        ];
        let apart = vec![
            Placement::Spread { cursor: 0 },
            Placement::Spread { cursor: 0 },
        ];
        let s = Segments::new(&p, &steps, two, apart).expect("the wait crosses a launch boundary");
        assert!(s.local_waits(UnitIx(1)).is_empty());
        // Segments that skip or repeat a unit are no tiling.
        let gap = vec![Segment {
            units: 1..2,
            cut: Cut::Start,
        }];
        let any = vec![Placement::Everywhere; 2];
        assert_eq!(
            Segments::new(&p, &steps, gap, any),
            Err(SegmentError::NotATiling)
        );
    }

    /// THE IN-PLACE HAZARD: a unit played on every threadgroup must write what every other copy
    /// writes. One that updates in place (a residual add, `fused_add_rmsnorm`) would let a copy
    /// that starts late read another copy's update and apply it twice — refused (there is no
    /// construct that writes such a unit back once), as is a copy of several items.
    #[test]
    fn a_unit_whose_copies_would_differ_is_never_played_on_every_threadgroup() {
        let mut steps = [
            step(&[0], &[1], 4),
            step(&[1, 5], &[5, 2], 1),
            step(&[2], &[3], 4),
        ];
        let p = plan(&steps);
        let one = || {
            vec![
                Segment {
                    units: 0..1,
                    cut: Cut::Start,
                },
                Segment {
                    units: 1..3,
                    cut: Cut::Waits(vec![UnitIx(0)]),
                },
            ]
        };
        let spread = Placement::Spread { cursor: 0 };
        let in_place = vec![spread, Placement::Everywhere, spread];
        assert_eq!(
            Segments::new(&p, &steps, one(), in_place.clone()),
            Err(SegmentError::NotCopyable { unit: UnitIx(1) })
        );
        // The same segments with the update out of place stand: each lane reads its own copy.
        steps[1] = step(&[1, 5], &[6, 2], 1);
        let q = plan(&steps);
        let s = Segments::new(&q, &steps, one(), in_place).expect("an out-of-place copy");
        assert_eq!(s.local_waits(UnitIx(2)), &[(UnitIx(1), LocalWait::OwnCopy)]);
        // A copied step of several items.
        let multi = [
            step(&[0], &[1], 4),
            step(&[1], &[2], 3),
            step(&[2], &[3], 4),
        ];
        let m = plan(&multi);
        let copied = vec![spread, Placement::Everywhere, spread];
        assert_eq!(
            Segments::new(&m, &multi, one(), copied),
            Err(SegmentError::NotCopyable { unit: UnitIx(1) })
        );
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
