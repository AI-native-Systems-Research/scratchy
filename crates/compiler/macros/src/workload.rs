// SPDX-License-Identifier: Apache-2.0
//! Workload points — the `(num_tokens, sk_bucket)` grid every model is
//! compiled at. Each point is one bucket of the emitted forward; codegen
//! keys its per-bucket tapes, statics and dispatch table by it.

use std::collections::BTreeSet;

/// A single point in a model's workload grid.
///
/// `num_tokens` is the bucket's row count (1 = decode). `sk_bucket` is the
/// KV-cache span axis for models that declare `sk_buckets = [..]`;
/// `sk_bucket == 0` is the sentinel "sk axis unused" point every other
/// model compiles at.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct WorkloadPoint {
    pub num_tokens: u64,
    pub sk_bucket: u64,
}

/// Every workload point one model is compiled at, in sorted
/// `(num_tokens, sk_bucket)` order.
#[derive(Clone, Debug, Default)]
pub struct WorkloadPoints {
    points: BTreeSet<WorkloadPoint>,
}

impl WorkloadPoints {
    /// The cross product of a model's `num_tokens` rungs and its
    /// `sk_buckets` (just `[0]` when it declares none).
    pub fn grid(num_tokens: &[u64], sk_buckets: &[u64]) -> Self {
        let sk_eff: &[u64] = if sk_buckets.is_empty() {
            &[0]
        } else {
            sk_buckets
        };
        let points = num_tokens
            .iter()
            .flat_map(|&m| {
                sk_eff.iter().map(move |&sk| WorkloadPoint {
                    num_tokens: m,
                    sk_bucket: sk,
                })
            })
            .collect();
        Self { points }
    }

    /// The points, sorted.
    pub fn iter(&self) -> impl Iterator<Item = &WorkloadPoint> + '_ {
        self.points.iter()
    }

    /// Distinct num_tokens values present in the grid, sorted.
    pub fn num_tokens_points(&self) -> Vec<u64> {
        let v: BTreeSet<u64> = self.points.iter().map(|wp| wp.num_tokens).collect();
        v.into_iter().collect()
    }
}
