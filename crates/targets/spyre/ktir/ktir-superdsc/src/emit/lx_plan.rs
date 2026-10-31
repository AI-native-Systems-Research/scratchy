//! ⭐⭐ THE SCRATCHPAD PLAN — torch-spyre's LX planning for one lowered function: which of the
//! intermediates it mints stay in each core's LX instead of making an HBM round trip between the op
//! that writes them and the ops that read them.
//!
//! torch-spyre plans LX after work division (`torch_spyre/_inductor/passes.py`: `_distribute_work`,
//! then `_maybe_scratchpad_planning`). Its allocator (`scratchpad/allocator.py`) keeps a buffer out
//! of LX for the reasons `_BufferResidencyChecks` names — among them a tensor read as indices or
//! through them (`indirect`), one an op walks a trip at a time (`advancing`), and one whose producer
//! and consumers hand each core a different slice (`core_division`: "a resident producer must match
//! every consumer", `exhaustive_search.py`) — and places the rest to minimise the HBM bytes left
//! unpinned. Its compiled paged attention is the result: every trip intermediate and the carried
//! running max, sum and output sit in LX, and only the pages, the page table and the mask come from
//! HBM.
//!
//! Here the same plan runs over the descriptors themselves. The function is lowered once on a copy
//! of the layout with the probe on ([`BundleLayout::lx_probe`]), which logs how every op slices
//! every tensor it names (the signature the core-local read law compares, `emit::core_slices`).
//! A tensor this function minted is a candidate when every op touching it hands each core the same
//! bytes and none reads it through an index or a trip at a time, and the candidates are placed as
//! torch-spyre's greedy solver places them (`scratchpad/greedy_solver.py`): by lifetime, so two
//! tensors live at disjoint times share bytes, in the window the pins leave
//! ([`crate::placement::lx_window_fits`]). The plan reaches the real lowering through one point,
//! [`BundleLayout::synth_df`], and the read law re-checks every op against it there.

use std::collections::{BTreeMap, BTreeSet};

use super::EmittedOp;
use super::lower_ktir_to_superdsc::{Error, err};
use crate::ktir_node::KtirNode;
use crate::place::PlaceId;
use crate::placement::{BundleLayout, LX_PIN_BASE, LX_PIN_END, LxShare, LxTouch, WalkDivision};

/// The 128-byte pin granularity [`BundleLayout::synth_lx_core_local`] reserves at.
const PIN_ALIGN: u64 = 128;

/// A node's mint record ([`KtirNode::minted`]), as one candidate's probe left it.
type MintRecord = std::collections::HashMap<crate::ktir_node::MintKey, u32>;

/// One tensor the plan keeps in LX: its id, one core's slice, and where that slice sits in the
/// plan's window (an offset into the bytes [`BundleLayout::reserve_lx_plan`] reserves).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Candidate {
    tid: u32,
    per_core: u64,
    at: u64,
}

/// A tensor's lifetime over the function's ops in program order: from its first use to one past its
/// last (torch-spyre's `LifetimeBoundBuffer::start_time` / `end_time`, `plan_solver.py`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Lifetime {
    start: usize,
    end: usize,
}

/// One op as the lifetimes read it: whether it is in the walk's body, and each tensor it names with
/// whether it reads it.
type OpUses<'a> = (bool, Vec<(&'a str, bool)>);

/// Plan `node`'s function into `layout` ([`BundleLayout::lx_planned`], [`BundleLayout::lx_division`]).
/// `probe` lowers the function into the layout it is handed; it runs on copies of `layout`, once per
/// candidate division, and its descriptors are thrown away. Run before the real lowering, which
/// [`check`] then holds to the plan.
///
/// ⭐ THE DIVISION IS PLANNED WITH THE RESIDENCY, as torch-spyre co-optimises them
/// (`co_optimizing_lx_planning`): each candidate — the ops' own divisions, one row group per batch of
/// the program's batched contractions (torch-spyre's decode trip, one kv head a core), and the
/// program's per-head rows held query-row-major and split into chunks (its prompt trip,
/// [`crate::placement::WalkDivision::HeadsInner`]) — is probed, and the one that leaves the fewest of
/// its minted bytes in HBM wins (torch-spyre's `score_leaf`, `exhaustive_search.py`: the whole
/// footprint of every buffer the placement does not pin); a tie keeps the ops' own.
///
/// ⭐ ONCE PER PROGRAM. A program already planned into this layout ([`KtirNode::lx_plan`], its
/// pinned tensors still planned here) lowers under that plan again, with the tensors it minted: a card
/// bake lowers each program twice into one layout (the pass that declares the bundle's synthetics,
/// then the real one), and planning the second time would plan around the first time's own pins.
pub(crate) fn plan(
    node: &KtirNode,
    layout: &BundleLayout,
    probe: impl Fn(&BundleLayout) -> Result<Vec<EmittedOp>, Error>,
) -> Result<(), Error> {
    let done = node
        .lx_plan
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .clone();
    if let Some(done) = done
        && done
            .pinned
            .iter()
            .all(|id| layout.lx_planned.borrow().contains_key(id))
    {
        layout.lx_division.set(done.division);
        return Ok(());
    }
    let minted = || -> BTreeSet<u32> {
        node.minted
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .values()
            .copied()
            .collect()
    };
    let free = (LX_PIN_END - LX_PIN_BASE).saturating_sub(layout.lx_window_bytes());
    let record = || {
        node.minted
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clone()
    };
    let mut best: Option<(u64, Option<WalkDivision>, Vec<Candidate>, MintRecord)> = None;
    // A minted tensor's whole footprint, wherever the probe declared it.
    let footprint = |l: &BundleLayout, name: &str| -> u64 {
        l.synth
            .borrow()
            .sizes
            .get(name)
            .or(l.lx.borrow().full_sizes.get(name))
            .copied()
            .unwrap_or(0)
    };
    let candidates = batch_groups(node)
        .into_iter()
        .map(WalkDivision::Batches)
        .chain(heads_inner(node));
    // Every probe mints from the record the plan started with: a division that mints in another
    // order would otherwise find a key an earlier probe recorded under a tid this probe has since
    // handed to a different value, and address the two as one tensor.
    let initial = record();
    for groups in std::iter::once(None).chain(candidates.map(Some)) {
        *node.minted.lock().unwrap_or_else(|p| p.into_inner()) = initial.clone();
        let copy = layout.clone();
        copy.lx_division.set(groups);
        *copy.lx_probe.borrow_mut() = Some(Vec::new());
        // A division the program cannot take — a form that refuses its shape — is no candidate, as
        // torch-spyre's search drops an illegal split; the ops' own division must lower.
        let ops = match probe(&copy) {
            Ok(ops) => ops,
            Err(_) if groups.is_some() => continue,
            Err(e) => return Err(e),
        };
        let touches = copy.lx_probe.borrow_mut().take().unwrap_or_default();
        let life = lifetimes(ops.iter().map(|e| {
            (
                e.walk.is_some(),
                e.arg_bindings
                    .iter()
                    .map(|a| (a.buffer.as_str(), a.is_input))
                    .collect(),
            )
        }));
        let chosen = choose(&touches, &minted(), &life, free);
        // A form that needs a tensor core-local takes this division only if the plan keeps it there.
        let kept_names: BTreeSet<String> = chosen
            .iter()
            .map(|c| crate::place::act_name(c.tid))
            .collect();
        if touches
            .iter()
            .any(|t| t.share == LxShare::Required && !kept_names.contains(&t.tensor))
        {
            continue;
        }
        let in_hbm: u64 = minted()
            .into_iter()
            .map(crate::place::act_name)
            .filter(|name| !kept_names.contains(name))
            .map(|name| footprint(&copy, &name))
            .sum();
        if best.as_ref().is_none_or(|(b, ..)| in_hbm < *b) {
            best = Some((in_hbm, groups, chosen, record()));
        }
    }
    let (_, groups, chosen, minted_under) = best.unwrap_or_default();
    // The real lowering mints as the chosen division's probe did, so the record it replays is that
    // probe's, whichever candidate ran last.
    *node.minted.lock().unwrap_or_else(|p| p.into_inner()) = minted_under;
    layout.lx_division.set(groups);
    let pinned: Vec<PlaceId> = chosen.iter().map(|c| PlaceId::Act(c.tid)).collect();
    let reach = chosen
        .iter()
        .map(|c| c.at + pin_bytes(c.per_core))
        .max()
        .unwrap_or(0);
    let base = layout.reserve_lx_plan(reach);
    layout
        .lx_planned
        .borrow_mut()
        .extend(chosen.into_iter().map(|c| {
            (
                PlaceId::Act(c.tid),
                crate::placement::LxPin {
                    per_core: c.per_core,
                    at: base + c.at,
                },
            )
        }));
    *node.lx_plan.lock().unwrap_or_else(|p| p.into_inner()) = Some(crate::ktir_node::WalkPlan {
        division: groups,
        pinned,
    });
    Ok(())
}

/// The batches of `node`'s batched contractions — `linalg.batch_matmul`'s `[b, m, n]` leading extent,
/// distinct and in order: the row-group counts the plan probes.
fn batch_groups(node: &KtirNode) -> Vec<u32> {
    let mut groups: Vec<u32> = Vec::new();
    for op in node.func.ops_deep() {
        if op.op_type != ktir_core::opkind::OpKind::LinalgBatchMatmul {
            continue;
        }
        if let Some([b, _, _]) = op
            .result_type
            .and_then(|t| t.dims())
            .map(<[i64]>::to_vec)
            .as_deref()
            && let Ok(b) = u32::try_from(*b)
            && b > 1
            && !groups.contains(&b)
        {
            groups.push(b);
        }
    }
    groups
}

/// The query-row-major division of a program that lays per-head rows head-major — a
/// `linalg.transpose` `[1, 0, 2]` of `[rows, heads, width]`, spyre-inference's
/// `q_rows.reshape(P, num_heads, D).transpose(0, 1)` — with more than one row a head: its rows split
/// into as many whole chunks as the cores take ([`crate::work::core_split`]). `None` for a program
/// with no such relayout or one row a head (a decode step: its rows are its heads).
fn heads_inner(node: &KtirNode) -> Option<WalkDivision> {
    node.func.ops_deep().into_iter().find_map(|op| {
        let is_102 = op.op_type == ktir_core::opkind::OpKind::LinalgTranspose
            && op.attributes.iter().any(|(k, v)| {
                matches!(
                    (k, v),
                    (ktir_core::attrkey::AttrKey::Permutation, ktir_core::ir::Attr::IntList(p))
                        if *p == [1, 0, 2]
                )
            });
        let [_, rows, _] = op.result_type?.dims()? else {
            return None;
        };
        let rows = u32::try_from(*rows).ok()?;
        (is_102 && rows > 1).then(|| WalkDivision::HeadsInner {
            chunks: crate::work::core_split(rows, crate::superdsc_opspec::MAX_CORES),
        })
    })
}

/// After the real lowering: every tensor the plan kept in LX was minted and declared there. A plan
/// naming a tid the lowering never minted means the two lowerings diverged, and nothing the plan
/// promised would hold.
pub(crate) fn check(layout: &BundleLayout, what: &str) -> Result<(), Error> {
    for id in layout.lx_planned.borrow().keys() {
        let name = id.to_string();
        if !layout.lx_core_local(&name) {
            return err(format!(
                "{what}: the scratchpad plan keeps {name} in LX, but the lowering never declared it \
                 — the probe and the real lowering minted different tensors"
            ));
        }
    }
    Ok(())
}

/// One core's slice of a tensor as a pin holds it: whole 128-byte units.
fn pin_bytes(per_core: u64) -> u64 {
    per_core.max(1).div_ceil(PIN_ALIGN) * PIN_ALIGN
}

/// torch-spyre's liveness (`calculate_liveness`) over the probe's ops in program order, widened as
/// `counted_loop_lifetime_overrides` widens it (`scratchpad/utils.py:139`): the walk's body is ONE
/// textual copy of every trip, so a tensor born outside the body (or never written) and read inside
/// it is live for the body's whole span — the next trip reads it again after the body's last op —
/// while a tensor the body itself writes keeps its own lifetime in each trip.
fn lifetimes<'a>(ops: impl IntoIterator<Item = OpUses<'a>>) -> BTreeMap<&'a str, Lifetime> {
    let ops: Vec<OpUses<'a>> = ops.into_iter().collect();
    let mut life: BTreeMap<&str, Lifetime> = BTreeMap::new();
    let mut born_in_body: BTreeMap<&str, bool> = BTreeMap::new();
    for (i, (in_body, uses)) in ops.iter().enumerate() {
        // Where a tensor is born: the first op that writes it, an op's writes before its reads; a
        // tensor read before anything writes it is born outside every loop.
        for &(name, _) in uses.iter().filter(|&&(_, read)| !read) {
            born_in_body.entry(name).or_insert(*in_body);
        }
        for &(name, _) in uses.iter().filter(|&&(_, read)| read) {
            born_in_body.entry(name).or_insert(false);
        }
        for &(name, _) in uses {
            life.entry(name)
                .and_modify(|l| l.end = i + 1)
                .or_insert(Lifetime {
                    start: i,
                    end: i + 1,
                });
        }
    }
    let body: Vec<usize> = (0..ops.len()).filter(|&i| ops[i].0).collect();
    if let (Some(&first), Some(&last)) = (body.first(), body.last()) {
        for &i in &body {
            for &(name, read) in &ops[i].1 {
                if read && !born_in_body.get(name).copied().unwrap_or(false) {
                    let l = life.get_mut(name).expect("a named tensor has a lifetime");
                    l.start = l.start.min(first);
                    l.end = l.end.max(last + 1);
                }
            }
        }
    }
    life
}

/// torch-spyre's residency verdict and placement over the probe's log: a minted tensor every op
/// hands out identically is a candidate, its slice as far as any op reads it; and the candidates are
/// placed by [`place`] in `free` bytes of the pin window.
fn choose(
    touches: &[LxTouch],
    minted: &BTreeSet<u32>,
    life: &BTreeMap<&str, Lifetime>,
    free: u64,
) -> Vec<Candidate> {
    let mut by_tensor: BTreeMap<&str, Vec<&LxShare>> = BTreeMap::new();
    for t in touches.iter().filter(|t| t.share != LxShare::Required) {
        by_tensor
            .entry(t.tensor.as_str())
            .or_default()
            .push(&t.share);
    }
    let candidates: Vec<(u32, u64, Lifetime)> = minted
        .iter()
        .filter_map(|&tid| {
            let name = crate::place::act_name(tid);
            let shares = by_tensor.get(name.as_str())?;
            let lifetime = *life.get(name.as_str())?;
            let LxShare::Slices { corners, .. } = shares.first()? else {
                return None;
            };
            // Every op starts each core's slice alike; the slice reaches as far as any op reads it
            // (a stick-plane window reaches its plane, `emit::stick_plane`).
            let mut per_core = 0u64;
            shares
                .iter()
                .all(|s| match s {
                    LxShare::Slices { bytes, corners: c } if c == corners => {
                        per_core = per_core.max(*bytes);
                        true
                    }
                    _ => false,
                })
                .then_some((tid, per_core, lifetime))
        })
        .collect();
    place(&candidates, free)
}

/// torch-spyre's greedy placement (`GreedyLayoutSolver.plan_layout`, `greedy_solver.py`) of each
/// candidate's slice in `limit` bytes. Time walks the candidates' starts and ends in order; at each
/// tick the tensors whose lifetime ends there free their bytes first, then each one whose lifetime
/// starts there takes the lowest address that holds it ([`free_block`]). One that finds none stays in
/// HBM.
fn place(candidates: &[(u32, u64, Lifetime)], limit: u64) -> Vec<Candidate> {
    let ticks: BTreeSet<usize> = candidates
        .iter()
        .flat_map(|&(_, _, l)| [l.start, l.end])
        .collect();
    // (address, pin bytes, end) of each tensor placed and still live.
    let mut live: Vec<(u64, u64, usize)> = Vec::new();
    let mut placed = Vec::new();
    for t in ticks {
        live.retain(|&(_, _, end)| end != t);
        for &(tid, per_core, _) in candidates.iter().filter(|&&(_, _, l)| l.start == t) {
            let size = pin_bytes(per_core);
            if let Some(at) = free_block(&live, size, limit) {
                live.push((at, size, candidates_end(candidates, tid)));
                placed.push(Candidate { tid, per_core, at });
            }
        }
    }
    placed
}

/// The end of candidate `tid`'s lifetime.
fn candidates_end(candidates: &[(u32, u64, Lifetime)], tid: u32) -> usize {
    candidates
        .iter()
        .find(|&&(t, _, _)| t == tid)
        .map_or(0, |&(_, _, l)| l.end)
}

/// torch-spyre's `_find_free_block`: the lowest address `size` bytes fit at among the `live`
/// placements in `limit` bytes — address 0 when nothing is live or the lowest live tensor starts at
/// least `size` in, else the 128-byte boundary above every live tensor, else the first gap between
/// them that holds it.
fn free_block(live: &[(u64, u64, usize)], size: u64, limit: u64) -> Option<u64> {
    if limit < size {
        return None;
    }
    let lowest = live.iter().map(|&(at, _, _)| at).min();
    if lowest.is_none_or(|lo| lo >= size) {
        return Some(0);
    }
    let highest = live
        .iter()
        .map(|&(at, bytes, _)| at + bytes)
        .max()
        .unwrap_or(0);
    let above = highest.div_ceil(PIN_ALIGN) * PIN_ALIGN;
    if above + size <= limit {
        return Some(above);
    }
    let mut spans: Vec<(u64, u64)> = live
        .iter()
        .map(|&(at, bytes, _)| (at, at + bytes))
        .collect();
    spans.sort_unstable();
    let mut merged: Vec<(u64, u64)> = Vec::new();
    for (lo, hi) in spans {
        match merged.last_mut() {
            Some(last) if lo <= last.1 => last.1 = last.1.max(hi),
            _ => merged.push((lo, hi)),
        }
    }
    merged.windows(2).find_map(|w| {
        let from = w[0].1.div_ceil(PIN_ALIGN) * PIN_ALIGN;
        (w[1].0 >= from + size).then_some(from)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn touch(op: &str, tensor: &str, share: LxShare) -> LxTouch {
        LxTouch {
            op: op.to_string(),
            tensor: tensor.to_string(),
            share,
        }
    }

    fn slices(bytes: u64, corners: &[u64]) -> LxShare {
        LxShare::Slices {
            bytes,
            corners: corners.to_vec(),
        }
    }

    /// Every tensor live over the whole log.
    fn all_live<'a>(names: &[&'a str]) -> BTreeMap<&'a str, Lifetime> {
        names
            .iter()
            .map(|&n| (n, Lifetime { start: 0, end: 1 }))
            .collect()
    }

    /// ⭐ torch-spyre's verdicts, one tensor each: kept when every op starts each core's slice alike
    /// (one op reading only a stick plane of it reaches less far, and the slice is sized by the
    /// furthest reach); spilled on a division mismatch, an indirect read, a trip-at-a-time walk, or
    /// when this function did not mint it.
    #[test]
    fn a_tensor_stays_in_lx_only_when_every_op_slices_it_alike() {
        let even = slices(128, &[0, 64, 128, 192]);
        let log = [
            touch("mul", "t10", even.clone()),
            touch("exp", "t10", slices(64, &[0, 64, 128, 192])),
            touch("mul", "t11", even.clone()),
            touch("bmm0", "t11", slices(128, &[0, 0, 0, 0])),
            touch("gather", "t12", even.clone()),
            touch("index", "t12", LxShare::Indirect),
            touch("add", "t13", even.clone()),
            touch("add_tiled", "t13", LxShare::TimeTiled),
            touch("add", "t14", even.clone()),
        ];
        let minted: BTreeSet<u32> = [10, 11, 12, 13].into();
        let life = all_live(&["t10", "t11", "t12", "t13", "t14"]);
        let kept = choose(&log, &minted, &life, 1 << 20);
        assert_eq!(
            kept.iter().map(|c| (c.tid, c.per_core)).collect::<Vec<_>>(),
            [(10, 128)],
            "t10 at its furthest reach; t11 mismatched, t12 indirect, t13 tiled, t14 not minted here"
        );
    }

    /// ⭐ torch-spyre's greedy placement: tensors live at disjoint times share bytes — t3 arrives as t1
    /// leaves and takes its slot — and one that arrives with the window full stays in HBM.
    #[test]
    fn tensors_live_at_disjoint_times_share_the_window() {
        let at = |start, end| Lifetime { start, end };
        let placed = place(
            &[
                (1, 512, at(0, 2)),
                (2, 512, at(1, 3)),
                (3, 512, at(2, 4)),
                (4, 512, at(2, 3)),
            ],
            1024,
        );
        assert_eq!(
            placed.iter().map(|c| (c.tid, c.at)).collect::<Vec<_>>(),
            [(1, 0), (2, 512), (3, 0)],
            "t1 at 0 and t2 above it; at tick 2 t1 has left, so t3 takes 0 and t4 finds no room"
        );
    }

    /// The first gap that holds a tensor takes it, at a 128-byte boundary, once the top is full.
    #[test]
    fn a_tensor_takes_the_first_gap_that_holds_it() {
        let live = [(0, 256, 9), (512, 128, 9), (768, 256, 9)];
        assert_eq!(free_block(&live, 256, 1024), Some(256), "the gap 256..512");
        assert_eq!(free_block(&live, 384, 1024), None, "no gap holds 384");
        assert_eq!(free_block(&[], 2048, 1024), None, "past the limit");
    }

    /// ⭐ torch-spyre's counted-loop widening: a tensor born before the walk and read inside it lives
    /// through the whole body (the next trip reads it again), a tensor the body writes keeps its own
    /// lifetime, and one read only after the walk is not widened.
    #[test]
    fn a_tensor_read_by_the_walk_lives_through_the_whole_body() {
        let ops: Vec<OpUses<'_>> = vec![
            (false, vec![("acc", false)]),
            (false, vec![("q", false)]),
            (true, vec![("q", true), ("s", false)]),
            (true, vec![("s", true), ("p", false)]),
            (true, vec![("p", true), ("acc", true), ("acc", false)]),
            (false, vec![("acc", true), ("out", false)]),
        ];
        let life = lifetimes(ops);
        let span = |n: &str| {
            let l = life[n];
            (l.start, l.end)
        };
        assert_eq!(
            span("q"),
            (1, 5),
            "q: born before the body, read in it — through its end"
        );
        assert_eq!(span("s"), (2, 4), "s: the body's own");
        assert_eq!(span("p"), (3, 5), "p: the body's own");
        assert_eq!(
            span("acc"),
            (0, 6),
            "acc: seeded before the body, carried through it"
        );
        assert_eq!(span("out"), (5, 6), "out: after the walk");
    }
}
