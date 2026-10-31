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
//! A tensor this function minted is kept in LX when every op touching it hands each core the same
//! bytes and none reads it through an index or a trip at a time, largest first while the pin window
//! holds ([`crate::placement::lx_window_fits`]). The plan reaches the real lowering through one
//! point, [`BundleLayout::synth_df`], and the read law re-checks every op against it there.

use std::collections::{BTreeMap, BTreeSet};

use super::EmittedOp;
use super::lower_ktir_to_superdsc::{Error, err};
use crate::ktir_node::KtirNode;
use crate::place::PlaceId;
use crate::placement::{BundleLayout, LX_PIN_BASE, LX_PIN_END, LxShare, LxTouch};

/// The 128-byte pin granularity [`BundleLayout::synth_lx_core_local`] reserves at.
const PIN_ALIGN: u64 = 128;

/// A node's mint record ([`KtirNode::minted`]), as one candidate's probe left it.
type MintRecord = std::collections::HashMap<crate::ktir_node::MintKey, u32>;

/// One tensor the plan may keep in LX: its id, one core's slice, and the HBM bytes it saves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Candidate {
    tid: u32,
    per_core: u64,
    total: u64,
}

/// Plan `node`'s function into `layout` ([`BundleLayout::lx_planned`], [`BundleLayout::lx_row_groups`]).
/// `probe` lowers the function into the layout it is handed; it runs on copies of `layout`, once per
/// candidate division, and its descriptors are thrown away. Run before the real lowering, which
/// [`check`] then holds to the plan.
///
/// ⭐ THE DIVISION IS PLANNED WITH THE RESIDENCY, as torch-spyre co-optimises them
/// (`co_optimizing_lx_planning`): each candidate — the ops' own divisions, and one row group per batch
/// of the program's batched contractions ([`crate::work::EwDivision::Groups`], torch-spyre's decode
/// trip, one kv head a core) — is probed, and the one that keeps the most bytes in LX wins; a tie keeps
/// the ops' own.
pub(crate) fn plan(
    node: &KtirNode,
    layout: &BundleLayout,
    probe: impl Fn(&BundleLayout) -> Result<Vec<EmittedOp>, Error>,
) -> Result<(), Error> {
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
    let mut best: Option<(u64, Option<u32>, Vec<Candidate>, MintRecord)> = None;
    for groups in std::iter::once(None).chain(batch_groups(node).into_iter().map(Some)) {
        let copy = layout.clone();
        copy.lx_row_groups.set(groups);
        *copy.lx_probe.borrow_mut() = Some(Vec::new());
        // A division the program cannot take — a form that refuses its shape — is no candidate, as
        // torch-spyre's search drops an illegal split; the ops' own division must lower.
        match probe(&copy) {
            Ok(_) => {}
            Err(_) if groups.is_some() => continue,
            Err(e) => return Err(e),
        }
        let touches = copy.lx_probe.borrow_mut().take().unwrap_or_default();
        let chosen = choose(&touches, &minted(), free);
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
        let kept: u64 = chosen.iter().map(|c| c.total).sum();
        if best.as_ref().is_none_or(|(b, ..)| kept > *b) {
            best = Some((kept, groups, chosen, record()));
        }
    }
    let (_, groups, chosen, minted_under) = best.unwrap_or_default();
    // The real lowering mints as the chosen division's probe did, so the record it replays is that
    // probe's, whichever candidate ran last.
    *node.minted.lock().unwrap_or_else(|p| p.into_inner()) = minted_under;
    layout.lx_row_groups.set(groups);
    layout.lx_planned.borrow_mut().extend(
        chosen
            .into_iter()
            .map(|c| (PlaceId::Act(c.tid), c.per_core)),
    );
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

/// torch-spyre's residency verdict and placement over the probe's log: a minted tensor every op
/// hands out identically is a candidate, and candidates are pinned largest first (the HBM bytes they
/// save) while their slices fit `free` bytes of the pin window.
fn choose(touches: &[LxTouch], minted: &BTreeSet<u32>, free: u64) -> Vec<Candidate> {
    let mut by_tensor: BTreeMap<&str, Vec<&LxShare>> = BTreeMap::new();
    for t in touches.iter().filter(|t| t.share != LxShare::Required) {
        by_tensor
            .entry(t.tensor.as_str())
            .or_default()
            .push(&t.share);
    }
    let mut candidates: Vec<Candidate> = minted
        .iter()
        .filter_map(|&tid| {
            let name = crate::place::act_name(tid);
            let shares = by_tensor.get(name.as_str())?;
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
                .then(|| Candidate {
                    tid,
                    per_core,
                    total: per_core * corners.len() as u64,
                })
        })
        .collect();
    candidates.sort_by(|a, b| b.total.cmp(&a.total).then(a.tid.cmp(&b.tid)));
    let mut left = free;
    candidates
        .into_iter()
        .filter(|c| {
            let fp = c.per_core.max(1).div_ceil(PIN_ALIGN) * PIN_ALIGN;
            let fits = fp <= left;
            if fits {
                left -= fp;
            }
            fits
        })
        .collect()
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
        let kept = choose(&log, &minted, 1 << 20);
        assert_eq!(
            kept.iter().map(|c| (c.tid, c.per_core)).collect::<Vec<_>>(),
            [(10, 128)],
            "t10 at its furthest reach; t11 mismatched, t12 indirect, t13 tiled, t14 not minted here"
        );
    }

    /// The window is spent on the largest tensors first, and a slice that no longer fits spills.
    #[test]
    fn the_largest_tensors_take_the_window_first() {
        let log = [
            touch("a", "t1", slices(256, &[0, 128])),
            touch("b", "t2", slices(1024, &[0, 512])),
            touch("c", "t3", slices(512, &[0, 256])),
        ];
        let minted: BTreeSet<u32> = [1, 2, 3].into();
        let kept: Vec<u32> = choose(&log, &minted, 1024 + 256)
            .iter()
            .map(|c| c.tid)
            .collect();
        assert_eq!(
            kept,
            [2, 1],
            "t2 first; t3's 512 no longer fits, t1's 256 does"
        );
    }
}
