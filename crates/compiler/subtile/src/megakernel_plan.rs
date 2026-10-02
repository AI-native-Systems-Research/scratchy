// SPDX-License-Identifier: Apache-2.0
//! THE FUSION PLAN — how a tape's steps fuse into generated kernels, one launch each.
//!
//! A tape's steps are played as LAUNCHES: a run of steps one generated kernel plays in one launch
//! of `P` threadgroups, or a step its own dispatch kernel plays. Which steps CAN share a launch —
//! which steps wait on which, which share a threadgroup, how a launch's work splits over its
//! threadgroups — is a fact of the tape's DATAFLOW, computed here, once, at expansion time
//! ([`plan`], then [`runs`]). Which launches the tape IS played as is an optimization over those
//! candidates ([`cheapest_tiling`]) under the target's costs.
//!
//! The target states each step's reads and writes in its own location vocabulary (`L`) and how
//! many work items the step splits into; this pass is generic over `L` and knows nothing else.
//!
//! # Units
//!
//! A [`Unit`] is what a run places. A multi-item step is a unit of its own. A run of single-item
//! steps where each depends on the run is ONE unit: one threadgroup plays it back to back, so its
//! intermediates never leave the threadgroup — a dependency inside a unit is a threadgroup
//! barrier, not a launch boundary.
//!
//! # Runs
//!
//! A [`Run`] is a candidate launch: consecutive units, a multi-item unit spreading its items
//! round-robin over the launch's threadgroups, a single-item unit pinned to one. A unit joins a
//! run only where ONE threadgroup orders each of its waits on it ([`LocalWait`]): it follows what
//! it waits on onto that unit's lane, or reads a producer every lane plays for itself. So no
//! generated kernel ever waits on another threadgroup — nothing spins, nothing needs every
//! threadgroup of a launch running at once, and the GPU may preempt between any two launches.
//! [`Run`] is built only by [`runs`], which rejects a run holding any other wait.
//!
//! # Tiling
//!
//! Every unit opens the run of itself, so the runs tile the plan however they are picked. A longer
//! run saves launch boundaries; a shorter one avoids what the longer one costs — a producer copied
//! onto every lane, a whole step's items serialized on one lane — or leaves a step to its own
//! dispatch kernel. [`cheapest_tiling`] picks exactly, under the costs the target gives each
//! candidate.

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
    /// A launch boundary stands before this step whatever the dataflow: the first step of an
    /// iteration of a loop the target keeps rolled (every iteration must be played alike), or the
    /// first after the loop. No unit or run spans it.
    pub segment_break: bool,
    /// One threadgroup may play every item of this step (an elementwise step): where its waits
    /// inside a run are one lane group's, it follows them there — beside the run in which it
    /// spreads after a launch boundary of its own.
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

/// A lane group of a run: pinned units that share one threadgroup of the run's launch — a unit and
/// the units of the run it follows. Which threadgroup a group gets is the launch's work split.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct Lane(pub u32);

/// Where a unit runs in its run.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Placement {
    /// Every threadgroup takes items round-robin, `cursor` counting the items spread earlier in
    /// the run — so consecutive spread units continue where the last one stopped.
    Spread { cursor: u32 },
    /// One threadgroup — the lane group's — plays the unit, every item of it.
    Pinned(Lane),
    /// EVERY threadgroup plays the (single-item) unit, each for itself: a unit that reads nothing it
    /// writes before writing it, so its copies write the same values wherever they run. What reads
    /// only its outputs runs beside it in its run — each threadgroup reads its own copy's.
    Everywhere,
}

/// How a unit's wait on a unit of its OWN run is met: by one threadgroup — the only order a
/// launch has. A wait of any other kind cannot stand inside a run ([`Run`]).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LocalWait {
    /// Both run in this lane group, the waiter after the unit it waits on.
    SameLane(Lane),
    /// The unit waited on runs on every lane: each lane's waiter reads its own lane's copy.
    OwnCopy,
}

/// A LAUNCH a generated kernel may play: consecutive units of the plan with no
/// [`StepFlow::segment_break`] inside, where each runs, and how each wait inside it is met — by
/// one threadgroup ([`LocalWait`]). Built only by [`runs`], which proves every wait inside is
/// local: no generated kernel waits on another threadgroup, so nothing spins, nothing needs every
/// threadgroup of a launch running at once, and the GPU may preempt between any two launches.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Run {
    units: Range<u32>,
    placements: Vec<Placement>,
    /// Per unit of the run: its waits on units of the run, each met by one threadgroup.
    local: Vec<Vec<(UnitIx, LocalWait)>>,
    groups: u32,
}

/// Why units cannot form the run asked for.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RunError {
    /// `unit` waits on `on` inside one run, and no one threadgroup plays both.
    CrossThreadgroupWait { unit: UnitIx, on: UnitIx },
    /// `unit` is played on every threadgroup, but its copies would not write the same values (it
    /// reads what it writes before writing it — an in-place update), or it has several items.
    NotCopyable { unit: UnitIx },
}

impl std::fmt::Display for RunError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CrossThreadgroupWait { unit, on } => write!(
                f,
                "unit {} waits on unit {} inside one run, and no one threadgroup plays both",
                unit.0, on.0
            ),
            Self::NotCopyable { unit } => {
                write!(
                    f,
                    "unit {} is played on every threadgroup but cannot be",
                    unit.0
                )
            }
        }
    }
}

impl std::error::Error for RunError {}

impl Run {
    /// `units` of `plan` (made from `steps`) placed at `placements` (one per unit, lane groups
    /// numbered densely), every unit played on every threadgroup copyable and every wait inside
    /// typed — or the first that is not.
    fn new<L: PartialEq>(
        plan: &FusionPlan,
        steps: &[StepFlow<L>],
        units: Range<u32>,
        placements: Vec<Placement>,
    ) -> Result<Self, RunError> {
        let at = |u: u32| (u - units.start) as usize;
        let copied = units.clone().zip(&placements);
        if let Some((u, _)) = copied
            .filter(|(_, p)| **p == Placement::Everywhere)
            .find(|(u, _)| !copyable(&plan.units[*u as usize], steps))
        {
            return Err(RunError::NotCopyable { unit: UnitIx(u) });
        }
        let mut local = vec![Vec::new(); placements.len()];
        for u in units.clone() {
            let inside = plan.units[u as usize].waits.iter();
            for &w in inside.filter(|w| units.contains(&w.0)) {
                let wait = match (placements[at(u)], placements[at(w.0)]) {
                    (_, Placement::Everywhere) => LocalWait::OwnCopy,
                    (Placement::Pinned(a), Placement::Pinned(b)) if a == b => {
                        LocalWait::SameLane(a)
                    }
                    _ => {
                        return Err(RunError::CrossThreadgroupWait {
                            unit: UnitIx(u),
                            on: w,
                        });
                    }
                };
                local[at(u)].push((w, wait));
            }
        }
        let groups = placements
            .iter()
            .filter_map(|p| match p {
                Placement::Pinned(l) => Some(l.0 + 1),
                _ => None,
            })
            .max()
            .unwrap_or(0);
        Ok(Self {
            units,
            placements,
            local,
            groups,
        })
    }

    /// The plan's units the run plays, in order.
    pub fn units(&self) -> Range<u32> {
        self.units.clone()
    }

    /// Per unit of the run, in order: where it runs.
    pub fn placements(&self) -> &[Placement] {
        &self.placements
    }

    /// Unit `u`'s (of the run) waits on units of the run, each with the lane group that meets it.
    pub fn local_waits(&self, u: UnitIx) -> &[(UnitIx, LocalWait)] {
        &self.local[(u.0 - self.units.start) as usize]
    }

    /// Lane groups of the run: its pinned units' groups are `Lane(0)..Lane(groups)`.
    pub fn groups(&self) -> u32 {
        self.groups
    }
}

/// A run being extended unit by unit: what it admitted and where each runs.
struct Open<'a, L> {
    plan: &'a FusionPlan,
    steps: &'a [StepFlow<L>],
    start: u32,
    placements: Vec<Placement>,
    groups: u32,
    cursor: u32,
}

impl<'a, L: PartialEq> Open<'a, L> {
    fn new(plan: &'a FusionPlan, steps: &'a [StepFlow<L>], start: u32) -> Self {
        Self {
            plan,
            steps,
            start,
            placements: Vec::new(),
            groups: 0,
            cursor: 0,
        }
    }

    /// Admit the run's next unit `u`, placed, or `false`: a segment break stands before it, or
    /// one threadgroup cannot order one of its waits inside the run.
    fn admit(&mut self, u: u32) -> bool {
        let (plan, steps, start) = (self.plan, self.steps, self.start);
        let unit = &plan.units[u as usize];
        let first = &steps[unit.steps.start as usize];
        if u > start && first.segment_break {
            return false;
        }
        let at = |w: &UnitIx| (w.0 - start) as usize;
        let waits: Vec<UnitIx> = (unit.waits.iter())
            .filter(|w| w.0 >= start)
            .copied()
            .collect();
        // Every wait inside the run a copy on each lane satisfies — a single-item producer this
        // unit only reads from, whose own producers inside the run are copies too, and that
        // nothing placed after it in the run writes over — lets the unit join, those producers
        // copied.
        let placements = &self.placements;
        let local = |w: &UnitIx| {
            let p = &plan.units[w.0 as usize];
            let fed = p.waits.iter().all(|x| {
                x.0 < start
                    || (placements[at(x)] == Placement::Everywhere
                        && only_reads(p, &plan.units[x.0 as usize], steps))
            });
            let kept = plan.units[start as usize..u as usize]
                .iter()
                .skip_while(|q| q.steps.start <= p.steps.start)
                .all(|q| only_reads(q, p, steps));
            let can =
                placements[at(w)] == Placement::Everywhere || (fed && kept && copyable(p, steps));
            can && only_reads(unit, p, steps)
        };
        let single = unit.items == Items::ONE;
        let inside: Vec<Placement> = waits.iter().map(|w| placements[at(w)]).collect();
        // A single-item unit — or a whole one — whose waits inside all run in one lane group
        // follows them there.
        let follows = match inside.as_slice() {
            [Placement::Pinned(l), rest @ ..]
                if (single || first.whole) && rest.iter().all(|p| *p == Placement::Pinned(*l)) =>
            {
                Some(*l)
            }
            _ if waits.iter().all(local) => {
                for w in &waits {
                    self.placements[at(w)] = Placement::Everywhere;
                }
                None
            }
            _ => return false,
        };
        let placement = match (follows, single) {
            (Some(l), _) => Placement::Pinned(l),
            (None, true) => {
                self.groups += 1;
                Placement::Pinned(Lane(self.groups - 1))
            }
            (None, false) => {
                self.cursor += unit.items.0.get();
                Placement::Spread {
                    cursor: self.cursor - unit.items.0.get(),
                }
            }
        };
        self.placements.push(placement);
        true
    }

    /// The run admitted so far, its lane groups numbered densely (a copied unit leaves its group).
    fn run(&self) -> Run {
        let mut dense: Vec<Option<u32>> = vec![None; self.groups as usize];
        let mut next = 0;
        let placements = (self.placements.iter())
            .map(|p| match *p {
                Placement::Pinned(Lane(g)) => {
                    let d = *dense[g as usize].get_or_insert_with(|| {
                        next += 1;
                        next - 1
                    });
                    Placement::Pinned(Lane(d))
                }
                other => other,
            })
            .collect();
        let end = self.start + self.placements.len() as u32;
        Run::new(self.plan, self.steps, self.start..end, placements)
            .expect("an admitted run holds only local waits")
    }
}

/// Every run of `plan` (made from `steps`) a generated kernel may play, by first unit, shortest
/// first: from each unit, the run of itself and each run it opens by admitting the next unit while
/// ONE threadgroup orders each of that unit's waits inside the run. A single-item unit — or a
/// [`StepFlow::whole`] one — whose waits inside all run in one lane group follows them there (and
/// plays every item there); a unit whose waits inside are single-item producers that only feed it
/// and that every lane can play for itself joins, those producers copied
/// ([`Placement::Everywhere`]); any other wait inside — a spread result, results in several lane
/// groups, one no copy can stand in for — ends the run, as does a segment break. A free
/// single-item unit opens a lane group of its own; a free multi-item unit spreads. Every unit
/// opens the run of itself, so the runs tile the plan however a tiling picks them.
pub fn runs<L: PartialEq>(plan: &FusionPlan, steps: &[StepFlow<L>]) -> Vec<Run> {
    let n = plan.units.len() as u32;
    let mut out = Vec::new();
    for start in 0..n {
        let mut open = Open::new(plan, steps, start);
        let mut u = start;
        while u < n && open.admit(u) {
            out.push(open.run());
            u += 1;
        }
    }
    out
}

/// The cheapest tiling of units `0..n` by `candidates` — each a range of units and what playing
/// it costs — minimizing the total by dynamic programming over the split points: exact. A tie
/// keeps the candidate listed first. The candidates' indices in order, or `None` when they tile
/// nothing.
pub fn cheapest_tiling<C>(n: u32, candidates: &[(Range<u32>, C)]) -> Option<Vec<usize>>
where
    C: Copy + PartialOrd + std::ops::Add<Output = C> + Default,
{
    let n = n as usize;
    let mut ending: Vec<Vec<usize>> = vec![Vec::new(); n + 1];
    for (k, (r, _)) in candidates.iter().enumerate() {
        if r.start < r.end && r.end as usize <= n {
            ending[r.end as usize].push(k);
        }
    }
    // best[j]: the cheapest tiling of `0..j` — its cost and the candidate ending it.
    let mut best: Vec<Option<(C, usize)>> = vec![None; n + 1];
    best[0] = Some((C::default(), usize::MAX));
    for j in 1..=n {
        for &k in &ending[j] {
            let (r, c) = &candidates[k];
            let Some((before, _)) = best[r.start as usize] else {
                continue;
            };
            let total = before + *c;
            if best[j].is_none_or(|(b, _)| total < b) {
                best[j] = Some((total, k));
            }
        }
    }
    let mut tiling = Vec::new();
    let mut j = n;
    while j > 0 {
        let (_, k) = best[j]?;
        tiling.push(k);
        j = candidates[k].0.start as usize;
    }
    tiling.reverse();
    Some(tiling)
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
    /// The longest run each unit opens, from the first unit on: the fewest launches the runs
    /// allow.
    fn longest(steps: &[StepFlow<u8>]) -> Vec<Run> {
        let all = runs(&plan(steps), steps);
        let mut out: Vec<Run> = Vec::new();
        let mut at = 0;
        while let Some(r) = all.iter().filter(|r| r.units.start == at).next_back() {
            at = r.units.end;
            out.push(r.clone());
        }
        out
    }
    fn ranges(runs: &[Run]) -> Vec<Range<u32>> {
        runs.iter().map(Run::units).collect()
    }

    #[test]
    fn independent_units_share_a_run_and_spread_items_continue_round_robin() {
        // two independent multi-item steps, then a single-item one: one run, one launch.
        let steps = [
            step(&[0], &[1], 3),
            step(&[0], &[2], 4),
            step(&[0], &[3], 1),
        ];
        let s = longest(&steps);
        assert_eq!(ranges(&s), vec![0..3]);
        assert_eq!(
            s[0].placements(),
            [
                Placement::Spread { cursor: 0 },
                Placement::Spread { cursor: 3 },
                Placement::Pinned(Lane(0))
            ]
        );
        assert_eq!(s[0].groups(), 1);
    }

    #[test]
    fn every_unit_opens_the_run_of_itself_and_each_run_it_extends_to() {
        let steps = [
            step(&[0], &[1], 3),
            step(&[0], &[2], 4),
            step(&[0], &[3], 1),
        ];
        let all = runs(&plan(&steps), &steps);
        assert_eq!(ranges(&all), vec![0..1, 0..2, 0..3, 1..2, 1..3, 2..3]);
        // A run's placements are its own: the same unit first in a run spreads from cursor 0.
        assert_eq!(all[3].placements(), [Placement::Spread { cursor: 0 }]);
    }

    #[test]
    fn a_wait_inside_a_run_ends_it_unless_one_lane_orders_it() {
        // in-place norm (1) -> qmv (4 items) -> in-place norm (1) -> side chain step (1) reading
        // the first norm. (In place, no norm can be copied onto every lane.)
        let steps = [
            step(&[0, 1], &[1], 1),
            step(&[1], &[2], 4),
            step(&[2, 3], &[3], 1),
            step(&[3], &[4], 4),
            step(&[1, 5], &[6], 1),
        ];
        let s = longest(&steps);
        assert_eq!(ranges(&s), vec![0..1, 1..2, 2..3, 3..5]);
        // The last unit waits only on unit 0 (an earlier launch): it runs beside the qmv.
        assert_eq!(s[3].placements()[0], Placement::Spread { cursor: 0 });
        assert_eq!(s[3].placements()[1], Placement::Pinned(Lane(0)));
        assert!(s[3].local_waits(UnitIx(4)).is_empty());
        // A single-item unit waiting on a pinned unit of its run follows it.
        let steps = [
            step(&[0], &[1], 1),
            step(&[9], &[8], 2),
            step(&[1], &[2], 1),
        ];
        let q = plan(&steps);
        assert_eq!(q.units.len(), 3);
        let t = longest(&steps);
        assert_eq!(ranges(&t), vec![0..3]);
        let p = t[0].placements();
        assert_eq!(p[2], p[0]);
        assert_eq!(p[0], Placement::Pinned(Lane(0)));
        assert_eq!(
            t[0].local_waits(UnitIx(2)),
            &[(UnitIx(0), LocalWait::SameLane(Lane(0)))]
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
        // The first runs beside the producer; the second in the next launch, after the boundary
        // the producer's spread items need.
        assert_eq!(ranges(&longest(&steps)), vec![0..2, 2..3]);
    }

    #[test]
    fn a_segment_break_splits_a_chain_and_no_run_spans_it() {
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
        assert_eq!(ranges(&runs(&p, &steps)), vec![0..1, 1..2, 2..3]);
        // Without the breaks: one chain, and the independent step beside it.
        let free = [steps[0].clone(), step(&[1], &[2], 1), step(&[9], &[8], 3)];
        let q = plan(&free);
        assert_eq!(q.units.len(), 2);
        assert_eq!(ranges(&longest(&free)), vec![0..2]);
    }

    #[test]
    fn a_whole_step_follows_the_lane_it_waits_on_and_otherwise_spreads() {
        // norm (1) -> small elementwise (3 items, whole) -> its reader (1): one lane, one run.
        let mut steps = [
            step(&[0], &[1], 1),
            step(&[1], &[2], 3),
            step(&[2], &[3], 1),
        ];
        steps[1].whole = true;
        let s = longest(&steps);
        assert_eq!(ranges(&s), vec![0..3]);
        assert!(
            s[0].placements()
                .iter()
                .all(|p| *p == Placement::Pinned(Lane(0)))
        );
        // The alternative stands beside it: the whole step opening a run of its own spreads there.
        let all = runs(&plan(&steps), &steps);
        let alone = all
            .iter()
            .find(|r| r.units() == (1..2))
            .expect("its own run");
        assert_eq!(alone.placements(), [Placement::Spread { cursor: 0 }]);
        // Not whole: the elementwise step spreads — beside a copy of the norm on every lane — and
        // its reader follows in the next launch.
        steps[1].whole = false;
        let t = longest(&steps);
        assert_eq!(ranges(&t), vec![0..2, 2..3]);
        assert_eq!(t[0].placements()[0], Placement::Everywhere);
        assert_eq!(
            t[0].local_waits(UnitIx(1)),
            &[(UnitIx(0), LocalWait::OwnCopy)]
        );
        // Whole but waiting on a spread step: it spreads in a run of its own, as any multi-item
        // step.
        let mut spread = [step(&[0], &[1], 4), step(&[1], &[2], 3)];
        spread[1].whole = true;
        let w = longest(&spread);
        assert_eq!(ranges(&w), vec![0..1, 1..2]);
        assert_eq!(w[1].placements()[0], Placement::Spread { cursor: 0 });
    }

    #[test]
    fn a_producer_every_lane_can_copy_saves_the_launch_its_spread_readers_need() {
        // norm (1, out of place) -> qmv (4 items): the norm runs on every lane, one run.
        let steps = [step(&[0], &[1], 1), step(&[1], &[2], 4)];
        let s = longest(&steps);
        assert_eq!(ranges(&s), vec![0..2]);
        assert_eq!(s[0].placements()[0], Placement::Everywhere);
        // The norm's run of itself keeps it on one lane: a copy only where a reader needs it.
        let all = runs(&plan(&steps), &steps);
        assert_eq!(all[0].placements(), [Placement::Pinned(Lane(0))]);
        // In place (it reads what it writes): a copy could read another copy's result.
        let in_place = [step(&[0, 1], &[1], 1), step(&[1], &[2], 4)];
        let t = longest(&in_place);
        assert_eq!(ranges(&t), vec![0..1, 1..2]);
        assert!(matches!(t[0].placements()[0], Placement::Pinned(_)));
        // The reader also overwrites the norm's input: another lane's copy could still read it.
        let clobber = [step(&[0], &[1], 1), step(&[1], &[2, 0], 4)];
        assert_eq!(ranges(&longest(&clobber)), vec![0..1, 1..2]);
        // A single-item chain that writes its own intermediate and reads it back is copied whole.
        let chain = [
            step(&[0], &[1], 1),
            step(&[1], &[2], 1),
            step(&[2], &[3], 4),
        ];
        assert_eq!(plan(&chain).units.len(), 2);
        let x = longest(&chain);
        assert_eq!(ranges(&x), vec![0..2]);
        assert_eq!(x[0].placements()[0], Placement::Everywhere);
    }

    /// No run holds a wait one threadgroup does not order: a spread producer read inside its own
    /// run, a pinned producer read in another lane group, or a copied unit reading a pinned one —
    /// each is refused, naming the wait.
    #[test]
    fn a_wait_no_one_threadgroup_orders_cannot_stand_inside_a_run() {
        let steps = [step(&[0], &[1], 4), step(&[1], &[2], 1)];
        let p = plan(&steps);
        let refused = Err(RunError::CrossThreadgroupWait {
            unit: UnitIx(1),
            on: UnitIx(0),
        });
        let spread = vec![
            Placement::Spread { cursor: 0 },
            Placement::Spread { cursor: 4 },
        ];
        assert_eq!(Run::new(&p, &steps, 0..2, spread), refused);
        let lanes = vec![Placement::Pinned(Lane(0)), Placement::Pinned(Lane(1))];
        assert_eq!(Run::new(&p, &steps, 0..2, lanes), refused);
        let copied = vec![Placement::Pinned(Lane(0)), Placement::Everywhere];
        assert_eq!(Run::new(&p, &steps, 0..2, copied), refused);
        // The same wait across a launch boundary stands: two runs.
        let a = Run::new(&p, &steps, 0..1, vec![Placement::Spread { cursor: 0 }]);
        let b = Run::new(&p, &steps, 1..2, vec![Placement::Spread { cursor: 0 }]);
        assert!(a.is_ok());
        assert!(
            b.expect("the wait crosses a launch boundary")
                .local_waits(UnitIx(1))
                .is_empty()
        );
        // And `runs` offers only those.
        assert_eq!(ranges(&runs(&p, &steps)), vec![0..1, 1..2]);
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
        let spread = Placement::Spread { cursor: 0 };
        let copied = vec![Placement::Everywhere, spread];
        assert_eq!(
            Run::new(&p, &steps, 1..3, copied.clone()),
            Err(RunError::NotCopyable { unit: UnitIx(1) })
        );
        // The same run with the update out of place stands: each lane reads its own copy.
        steps[1] = step(&[1, 5], &[6, 2], 1);
        let q = plan(&steps);
        let s = Run::new(&q, &steps, 1..3, copied).expect("an out-of-place copy");
        assert_eq!(s.local_waits(UnitIx(2)), &[(UnitIx(1), LocalWait::OwnCopy)]);
        // A copied step of several items.
        let multi = [
            step(&[0], &[1], 4),
            step(&[1], &[2], 3),
            step(&[2], &[3], 4),
        ];
        let m = plan(&multi);
        assert_eq!(
            Run::new(&m, &multi, 1..3, vec![Placement::Everywhere, spread]),
            Err(RunError::NotCopyable { unit: UnitIx(1) })
        );
    }

    #[test]
    fn the_cheapest_tiling_is_exact_and_a_tie_keeps_the_candidate_listed_first() {
        // Taking the longest candidate first (0..2) forces 2..3 at 10: 13. The optimum is 1 + 2 + 1.
        let candidates = [(0..2, 3u32), (0..1, 1), (1..3, 2), (2..3, 10), (1..2, 9)];
        assert_eq!(cheapest_tiling(3, &candidates), Some(vec![1, 2]));
        // Equal totals: the candidate listed first ends the tiling.
        let tie = [(0..2, 2u32), (0..1, 1), (1..2, 1)];
        assert_eq!(cheapest_tiling(2, &tie), Some(vec![0]));
        // A gap tiles nothing.
        assert_eq!(cheapest_tiling(3, &[(0..1, 1u32), (2..3, 1)]), None);
    }

    /// The tiling weighs a launch boundary against what the longer run costs: a norm copied onto
    /// every lane saves the launch its spread reader needs when a launch costs more than the copy,
    /// and not otherwise.
    #[test]
    fn a_launch_boundary_is_weighed_against_what_the_longer_run_costs() {
        let steps = [step(&[0], &[1], 1), step(&[1], &[2], 4)];
        let all = runs(&plan(&steps), &steps);
        assert_eq!(ranges(&all), vec![0..1, 0..2, 1..2]);
        let solve = |launch: u32, copy: u32| {
            let cost = |r: &Run| {
                let work = r.placements().iter().map(|p| match p {
                    Placement::Everywhere => copy,
                    Placement::Pinned(_) => 1,
                    Placement::Spread { .. } => 4,
                });
                launch + work.sum::<u32>()
            };
            let candidates: Vec<(Range<u32>, u32)> =
                all.iter().map(|r| (r.units(), cost(r))).collect();
            let tiling = cheapest_tiling(2, &candidates).expect("the runs tile");
            tiling.iter().map(|&k| all[k].units()).collect::<Vec<_>>()
        };
        assert_eq!(solve(10, 1), vec![0..2]);
        assert_eq!(solve(1, 50), vec![0..1, 1..2]);
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
