// SPDX-License-Identifier: Apache-2.0
//! ⭐⭐⭐ TORCH-SPYRE'S WORK DIVISION, PORTED — how ONE op's iteration space is cut
//! across the device's cores, as ONE type every op lowering consumes.
//!
//! Ported from `torch_spyre/_inductor/work_division.py`
//! (`multi_dim_iteration_space_split`, `_largest_legal_split_from`,
//! `prioritize_dimensions`, `_default_split`), statement for statement:
//!
//! 1. **Committed splits first.** The splits a fused group's anchor contraction
//!    already committed (ownership is keyed by split count, so every op in the
//!    group must agree on the shared dims) are mandatory and consume their
//!    cores before anything is chosen.
//! 2. **Output dims by decreasing extent** (ties keep the iteration-space
//!    order). Each takes the largest domain-legal divisor of its extent that
//!    fits the remaining core budget.
//! 3. **At most ONE reduction dim**, the most splittable, if cores remain —
//!    the device accumulates a K-split through PSUM, which one dim at a time
//!    costs and two never do.
//!
//! The product of all splits never exceeds the core budget. A dim's legal
//! divisors come from its [`SplitDomain`], which the op lowering states from
//! the op's OWN facts — never from a model, a kernel or a name:
//!
//! * [`SplitDomain::Any`] — a free dim of a dense operand: every divisor.
//! * [`SplitDomain::Whole`] — a dim that must stay whole: an indirectly
//!   accessed (gathered/scattered) operand's interior dims, a coarse-tile-local
//!   dim, and a contraction's K until a combine op exists.
//! * [`SplitDomain::StageSlabs`] — a contraction's row dim: the per-core row
//!   block must be a whole number of stage slabs (the LHS stage the device
//!   streams the PT array from — `dataStageParam.ss_.mb_` in torch-spyre's own
//!   descriptors, 64 rows at gemma's sliding shapes).
//!
//! The design targets, stated as Rust fixtures in
//! `tests/zz_the_work_division_is_torch_spyres_law.rs`, are torch-spyre's own
//! gemma sliding prefill attention descriptors (`13_batchmatmul`
//! `{y:8, x:2, mb:2}` and `14_add` `{mb:8, x:2, y:2}`, both on 32 cores) — no
//! JSON golden comparer.
//!
//! ⛔ THIS TYPE REPLACES THE SCRATCHPAD-DIVISION PROBE. The device FORM an op
//! lowers to is the op's own, at its stated dims; nothing scores candidate
//! divisions by footprint. LX residency — which buffer stays in LX — is a
//! separate law (the scratchpad allocator), never a division probe.

use std::collections::BTreeMap;

/// One dimension of an op's iteration space: its name, its extent, and whether
/// it is an output dim (the result spans it) or a reduction dim (it is
/// contracted away).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IterDim {
    pub name: &'static str,
    pub extent: u32,
    pub role: DimRole,
    pub domain: SplitDomain,
}

/// Whether an iteration dim survives into the op's result or is contracted away.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DimRole {
    Output,
    Reduction,
}

/// The legal core-partition counts of one iteration dim, stated from the op's
/// own operand facts. Values are split COUNTS (divisors of the extent), not
/// extents: a split of 1 means one partition walks the dim's full extent
/// serially inside the op's own loop.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SplitDomain {
    /// A free dim of a dense operand: every divisor of the extent.
    Any,
    /// The dim stays whole. Stated for an indirectly accessed operand's
    /// interior dims, a coarse-tile-local dim, and a contraction's K.
    Whole,
    /// A contraction's row dim: the per-core row block must be a whole number
    /// of `slab_rows`-row stage slabs.
    StageSlabs { slab_rows: u32 },
}

impl SplitDomain {
    /// The largest legal divisor of `extent` that is at most `budget`.
    fn largest_within(&self, extent: u32, budget: u32) -> u32 {
        match self {
            SplitDomain::Whole => 1,
            SplitDomain::Any => largest_divisor_within(extent, budget),
            SplitDomain::StageSlabs { slab_rows } => {
                let mut best = 1;
                let mut split = 1;
                while split <= budget && split <= extent {
                    let per_core = extent / split;
                    if extent.is_multiple_of(split)
                        && per_core.is_multiple_of(*slab_rows)
                        && split > best
                    {
                        best = split;
                    }
                    split += 1;
                }
                best
            }
        }
    }
}

/// The largest divisor of `extent` that is at most `budget` (1 if none).
fn largest_divisor_within(extent: u32, budget: u32) -> u32 {
    let mut best = 1;
    let mut split = 1;
    while split <= budget && split <= extent {
        if extent.is_multiple_of(split) {
            best = split;
        }
        split += 1;
    }
    best
}

/// An op's iteration space distributed over the device's cores: one split count
/// per iteration dim, minted once by [`WorkDivision::law`]. Every op lowering
/// consumes this type; none re-derives a split.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WorkDivision {
    splits: BTreeMap<&'static str, u32>,
    cores: u32,
}

impl WorkDivision {
    /// The ported three-pass law. `committed` carries the splits the fused
    /// group's anchor contraction already committed to the shared dims —
    /// mandatory, and their cores are spent before the greedy passes.
    ///
    /// Refuses (as a `String`, the bake's refusal surface) when a committed
    /// split is not a divisor of its dim's extent, is not in the dim's domain,
    /// or when the committed splits alone exceed the core budget.
    pub fn law(
        dims: &[IterDim],
        committed: &[(&'static str, u32)],
        max_cores: u32,
    ) -> Result<WorkDivision, String> {
        let mut splits: BTreeMap<&'static str, u32> =
            dims.iter().map(|d| (d.name, 1)).collect();
        let mut cores_used = 1u32;

        // ── Pass 1: the committed splits are mandatory. ─────────────────────
        for (name, split) in committed {
            let dim = dims.iter().find(|d| d.name == *name).ok_or_else(|| {
                format!("work division: committed split of unknown dim {name}")
            })?;
            if *split == 1 {
                continue;
            }
            if dim.extent % split != 0 {
                return Err(format!(
                    "work division: committed split {split} of {name} does not divide its \
                     extent {}",
                    dim.extent
                ));
            }
            if dim.domain.largest_within(dim.extent, u32::MAX) < *split {
                return Err(format!(
                    "work division: committed split {split} of {name} is outside its domain"
                ));
            }
            let prev = splits[name];
            cores_used = cores_used / prev * *split;
            splits.insert(name, *split);
        }
        if cores_used > max_cores {
            return Err(format!(
                "work division: committed splits need {cores_used} cores, over the {max_cores}\
                 -core budget"
            ));
        }

        // ── Pass 2: output dims by decreasing extent, ties in iteration-space
        //           order, each the largest domain-legal divisor that fits. A
        //           dim the group's anchor committed is the GROUP's: this op
        //           takes its split and does not grow it — ownership is keyed
        //           by split count, so the group's ops cannot disagree. ─────
        let committed_of = |name: &str| {
            committed
                .iter()
                .find(|(n, _)| *n == name)
                .map(|&(_, s)| s)
        };
        let mut output_order: Vec<&IterDim> = dims
            .iter()
            .filter(|d| d.role == DimRole::Output)
            .collect();
        output_order.sort_by_key(|d| std::cmp::Reverse(d.extent));
        for dim in output_order {
            if cores_used >= max_cores || committed_of(dim.name).is_some() {
                continue;
            }
            let budget = max_cores / cores_used;
            let current = splits[dim.name];
            let split = dim.domain.largest_within(dim.extent, budget * current);
            if split > current {
                cores_used = cores_used / current * split;
                splits.insert(dim.name, split);
            }
        }

        // ── Pass 3: at most ONE reduction dim, the most splittable. ────────
        if cores_used < max_cores {
            let mut best: Option<(&IterDim, u32)> = None;
            for dim in dims
                .iter()
                .filter(|d| d.role == DimRole::Reduction && committed_of(d.name).is_none())
            {
                let budget = max_cores / cores_used;
                let current = splits[dim.name];
                let split = dim.domain.largest_within(dim.extent, budget * current);
                if split > current && best.is_none_or(|(_, b)| split > b) {
                    best = Some((dim, split));
                }
            }
            if let Some((dim, split)) = best {
                cores_used = cores_used / splits[dim.name] * split;
                splits.insert(dim.name, split);
            }        }

        Ok(WorkDivision { splits, cores: cores_used })
    }

    /// The split count of one dim (1 = whole).
    pub fn split_of(&self, name: &str) -> u32 {
        self.splits.get(name).copied().unwrap_or(1)
    }

    /// The cores this division uses: the product of the splits.
    pub fn cores(&self) -> u32 {
        self.cores
    }
}
