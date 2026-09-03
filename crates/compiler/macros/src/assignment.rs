// SPDX-License-Identifier: Apache-2.0
//! Workload ASSIGNMENT types — the map keys and per-workload payload that
//! downstream codegen indexes by.
//!
//! These describe WHICH workload point (num_tokens, sk_bucket) a result
//! belongs to; they say nothing about how that result was produced. The
//! tape-scheduled path builds them directly with empty payloads (see
//! `lib.rs`'s `tape_pilot` branch) and never runs the solver, so codegen's
//! dependence on these types is not a dependence on instruction selection.
//!
//! They lived in `solver` because the solver was their only producer. Keeping
//! them there means a build that performs no instruction selection still has
//! to compile the solver to name its own map keys.

use std::collections::{BTreeMap, HashMap};

use crate::fuf::TileId;

/// Stable identifier for one [`Implementation`] in the
/// [`ImplementationLibrary`]. Indices are dense in the library's
/// `entries` vector.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ImplId(pub u32);

/// Stable identifier for one claimed subgraph within an SFUF. Each
/// commit of "these tiles are claimed by this impl" allocates a
/// fresh `SubgraphId`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SubgraphId(pub u32);
/// An SFUF: the FUF with every tile bound to a subgraph and every
/// subgraph bound to one Impl.
#[derive(Clone, Debug, Default)]
pub struct Assignment {
    /// Tile → which subgraph claims it.
    pub cover: HashMap<TileId, SubgraphId>,
    /// Subgraph → which Impl realizes it.
    pub impls: HashMap<SubgraphId, ImplId>,
    /// Sum of per-subgraph cost estimates in microseconds at this
    /// workload point.
    pub predicted_us: f64,
}
impl Assignment {
    // ⛔ SOLVER ONLY. Read by `solver.rs`/`cost.rs`, which a tape-scheduled target does
    // not run — so metal/spyre should not compile it rather than allow it to be dead.
    #[cfg(not(any(feature = "metal", feature = "spyre")))]
    // Used by the solver's tests only.
    #[cfg(test)]
    pub fn is_cover_complete(&self, num_tiles: usize) -> bool {
        self.cover.len() == num_tiles
    }

    pub fn subgraph_of(&self, tile: TileId) -> Option<SubgraphId> {
        self.cover.get(&tile).copied()
    }

    pub fn impl_of(&self, sg: SubgraphId) -> Option<ImplId> {
        self.impls.get(&sg).copied()
    }

    pub fn subgraphs(&self) -> impl Iterator<Item = SubgraphId> + '_ {
        self.impls.keys().copied()
    }

    pub fn tiles_in_subgraph(&self, sg: SubgraphId) -> Vec<TileId> {
        let mut v: Vec<TileId> = self
            .cover
            .iter()
            .filter_map(|(t, s)| if *s == sg { Some(*t) } else { None })
            .collect();
        v.sort();
        v
    }

    pub fn num_subgraphs(&self) -> usize {
        self.impls.len()
    }
}
/// A single point in the solver's workload sweep grid.
///
/// Historically the sweep was 1-D (only `num_tokens`). Attention
/// impls whose cost depends on the KV-cache span — notably
/// FlashInfer's persistent runner, which widens its decode advantage
/// as `sk` grows — add a second axis. Non-attention impls (GEMM,
/// RoPE, RMSNorm) declare `WorkloadConstraint::Any` or
/// `::NumTokensRange` and are `sk_bucket`-insensitive; the codegen
/// coalesces identical-assignment `(num_tokens, sk_bucket)` pairs so
/// only attention tiles actually force distinct compiled forwards.
///
/// `sk_bucket == 0` is the sentinel "sk axis unused" point used by
/// legacy 1-D callers (unit tests and any model file that doesn't
/// declare `sk_buckets = [..]`). Impls that require a concrete sk
/// range (`WorkloadConstraint::NumTokensAndSkRange`) simply never
/// match at `sk_bucket == 0`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct WorkloadPoint {
    pub num_tokens: u64,
    pub sk_bucket: u64,
}
impl WorkloadPoint {
    // ⛔ SOLVER ONLY. Read by `solver.rs`/`cost.rs`, which a tape-scheduled target does
    // not run — so metal/spyre should not compile it rather than allow it to be dead.
    #[cfg(not(any(feature = "metal", feature = "spyre")))]
    /// Legacy num-tokens-only point, with `sk_bucket = 0`.
    // Used by the solver's tests only.
    #[cfg(test)]
    pub fn num_tokens_only(num_tokens: u64) -> Self {
        Self {
            num_tokens,
            sk_bucket: 0,
        }
    }
}
/// Result of a full workload sweep: one SFUF per `(num_tokens,
/// sk_bucket)` point. For 1-D callers (legacy tests, models that
/// don't declare `sk_buckets`) every entry has `sk_bucket = 0` and
/// the map is effectively keyed on `num_tokens`.
#[derive(Clone, Debug, Default)]
pub struct WorkloadAssignments {
    pub per_workload: BTreeMap<WorkloadPoint, Assignment>,
}
impl WorkloadAssignments {
    /// Lookup by num_tokens only, picking the first matching
    /// sk_bucket in sorted order. Convenience for 1-D callers and
    /// invariants that care only about coverage, not sk dispatch.
    pub fn get_nt(&self, num_tokens: u64) -> Option<&Assignment> {
        self.per_workload
            .iter()
            .find_map(|(wp, a)| (wp.num_tokens == num_tokens).then_some(a))
    }

    /// Distinct num_tokens values present in the sweep, sorted.
    pub fn num_tokens_points(&self) -> Vec<u64> {
        let mut v: Vec<u64> = self
            .per_workload
            .keys()
            .map(|wp| wp.num_tokens)
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        v.sort();
        v
    }
}
