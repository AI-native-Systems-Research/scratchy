// SPDX-License-Identifier: Apache-2.0
//! Backend-neutral GPU memory-budget + prefill-bucket helpers.
//!
//! Pure index/arithmetic logic shared by the CUDA and Metal `Worker`
//! implementations. Lives in `scratchy-serving-engine` (which neither the CUDA
//! runtime nor the worker crate cycle through) so both backends can reach it
//! without a dependency cycle.

/// Compute KV cache budget matching Python vLLM's formula exactly:
///   requested = total_memory * gpu_memory_utilization
///   non_kv_cache = weights_and_overhead + peak_activations + 150 MiB
///   available_kv_bytes = requested - non_kv_cache
///
/// This is the exact logic used in `CudaWorker::determine_available_memory`.
pub fn compute_available_kv_bytes(
    total_memory: usize,
    weights_and_overhead: usize,
    peak_activation_bytes: usize,
    gpu_memory_utilization: f64,
) -> usize {
    let redundancy_buffer: usize = 150 * 1024 * 1024; // 150 MiB
    let non_kv_cache = weights_and_overhead + peak_activation_bytes + redundancy_buffer;
    let requested = (total_memory as f64 * gpu_memory_utilization) as usize;
    requested.saturating_sub(non_kv_cache)
}

/// Outcome of target-reactive prefill-bucket selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefillBucketSelection {
    /// Largest prefill `bucket_m` to keep resident. The backend prunes its
    /// per-bucket tapes/shapes above this, and the scheduler clamps
    /// `max_num_batched_tokens` to it so a single forward never exceeds it.
    pub max_bucket_m: u32,
    /// Activation footprint of the selected bucket (the cost actually paid):
    /// on Metal the colored arena bytes, on CUDA the capture working set.
    pub arena_bytes: u64,
    /// Bytes left for the KV cache after the fixed allocations and the
    /// selected activation footprint.
    pub kv_bytes: u64,
}

/// Pick the largest prefill bucket the device can afford — the backend-neutral
/// half of "target-reactive buckets". The forward macro emits the SAME
/// candidate ladder for every backend; each backend supplies its own
/// per-bucket activation cost (`bucket_costs`, `(bucket_m, bytes)`) and applies
/// the result to its own mechanism (Metal: the pre-reserved colored arena that
/// trades off against KV; CUDA: which graph shapes to capture). The policy:
/// the activation footprint may use up to `arena_fraction` of the headroom left
/// after the fixed (weights + recurrent-state reserve + redundancy)
/// allocations; the KV cache gets the rest. The smallest bucket is always kept
/// so a memory-starved device still runs (it just chunks prefill harder).
///
/// This is why nothing here is Apple-specific: a 24 GB 4090 vs an 80 GB H100
/// reacts exactly like a 32 GB M-series vs a 192 GB Ultra.
pub fn select_prefill_bucket(
    budget_bytes: u64,
    fixed_bytes: u64,
    bucket_costs: &[(u32, u64)],
    arena_fraction: f64,
) -> PrefillBucketSelection {
    let headroom = budget_bytes.saturating_sub(fixed_bytes);
    let arena_budget = (headroom as f64 * arena_fraction.clamp(0.0, 1.0)) as u64;
    // Always-available fallback: the smallest-`m` bucket. Used when even it
    // exceeds `arena_budget` on a severely memory-starved device.
    let fallback = bucket_costs
        .iter()
        .copied()
        .min_by_key(|&(m, _)| m)
        .unwrap_or((1, 0));
    // Largest `m` whose activation cost fits the arena budget. Costs are
    // monotonic in `m`, but `max_by_key` over the fitting set is robust to
    // any ordering of `bucket_costs`.
    let chosen = bucket_costs
        .iter()
        .copied()
        .filter(|&(_, cost)| cost <= arena_budget)
        .max_by_key(|&(m, _)| m)
        .unwrap_or(fallback);
    PrefillBucketSelection {
        max_bucket_m: chosen.0,
        arena_bytes: chosen.1,
        kv_bytes: headroom.saturating_sub(chosen.1),
    }
}

/// Stable per-process u64 key for a request's GDN state slot.
///
/// `GdnSlotAllocator` keys on `u64`, but the engine identifies requests
/// by `String`. A `DefaultHasher` (fixed SipHash keys `(0, 0)`) is
/// deterministic across calls within a run, so the same `req_id` always
/// maps to the same slot for its whole lifetime; collisions among the
/// few hundred concurrently-live ids are astronomically unlikely.
pub fn gdn_slot_key(req_id: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    req_id.hash(&mut h);
    h.finish()
}

#[cfg(test)]
mod bucket_selection_tests {
    use super::{PrefillBucketSelection, select_prefill_bucket};

    const GIB: u64 = 1024 * 1024 * 1024;
    const MIB: u64 = 1024 * 1024;

    // The Qwen3.5-35B-A3B ladder costs measured on Metal (arena bytes per
    // bucket): tiny for the decode buckets, then the prefill ramp.
    fn ladder() -> Vec<(u32, u64)> {
        vec![
            (1, 4 * MIB),
            (8, 8 * MIB),
            (64, 36 * MIB),
            (512, 288 * MIB),
            (2048, 1150 * MIB),
            (4096, 2300 * MIB),
        ]
    }

    #[test]
    fn starved_box_picks_small_bucket_keeps_kv() {
        // 32 GB M-series running a 35B: budget 22.5 GiB, fixed (weights+gdn)
        // ~21.3 GiB -> ~1.2 GiB headroom. Only the 512 bucket fits 0.6x of it.
        let sel = select_prefill_bucket(22_500 * MIB, 21_300 * MIB, &ladder(), 0.6);
        assert_eq!(
            sel.max_bucket_m, 512,
            "32GB box should pick 512, got {sel:?}"
        );
        assert!(
            sel.kv_bytes > 700 * MIB,
            "must leave healthy KV, got {sel:?}"
        );
    }

    #[test]
    fn roomy_box_picks_top_bucket() {
        // 64 GB part: budget ~43 GiB, same fixed -> ~21.7 GiB headroom ->
        // the 4096 bucket (2.3 GiB) easily fits 0.6x of it.
        let sel = select_prefill_bucket(43 * GIB, 21_300 * MIB, &ladder(), 0.6);
        assert_eq!(
            sel.max_bucket_m, 4096,
            "64GB box should pick 4096, got {sel:?}"
        );
        assert!(
            sel.kv_bytes > 18 * GIB,
            "should leave lots of KV, got {sel:?}"
        );
    }

    #[test]
    fn severely_starved_falls_back_to_smallest() {
        // ⛔ THE NUMBERS DID NOT MATCH THE PREMISE, AND THE TEST WAS FAILING BECAUSE OF IT.
        // It said "no headroom at all" while passing `21 * GIB` (21504 MiB) against `21_300 * MIB` of
        // fixed allocations — 204 MiB of headroom, 122 MiB of arena budget at 0.6, which the 64 bucket
        // (36 MiB) fits. Returning 64 there is the documented policy working correctly ("the largest
        // prefill bucket the device can afford"), so the function was right and the assertion was wrong.
        // Budget BELOW the fixed cost is what "severely starved" means.
        let sel = select_prefill_bucket(21_000 * MIB, 21_300 * MIB, &ladder(), 0.6);
        assert_eq!(
            sel.max_bucket_m, 1,
            "starved box falls back to smallest, got {sel:?}"
        );
        assert_eq!(sel.kv_bytes, 0, "nothing left for KV either, got {sel:?}");
    }

    /// The boundary the broken assertion was really reaching for: headroom that is nonzero but smaller
    /// than the cheapest bucket still falls back, and one MiB more of arena budget than that bucket
    /// costs does not.
    #[test]
    fn the_fallback_boundary_is_the_cheapest_bucket_not_zero_headroom() {
        // 4 MiB cheapest bucket, 0.5 fraction => needs 8 MiB of headroom to afford it.
        let just_short = select_prefill_bucket(21_307 * MIB, 21_300 * MIB, &ladder(), 0.5);
        assert_eq!(just_short.max_bucket_m, 1, "3.5 MiB arena affords nothing");
        let just_enough = select_prefill_bucket(21_308 * MIB, 21_300 * MIB, &ladder(), 0.5);
        assert_eq!(
            just_enough.max_bucket_m, 1,
            "4 MiB arena affords the m=1 bucket and only it"
        );
        assert_eq!(just_enough.arena_bytes, 4 * MIB);
    }

    #[test]
    fn monotonic_kv_decreases_as_bucket_grows() {
        // Sanity: a bigger arena_fraction selects a bigger bucket and leaves
        // less KV — the explicit prefill-vs-KV tradeoff.
        let lean = select_prefill_bucket(43 * GIB, 21_300 * MIB, &ladder(), 0.1);
        let rich = select_prefill_bucket(43 * GIB, 21_300 * MIB, &ladder(), 0.9);
        assert!(lean.max_bucket_m <= rich.max_bucket_m);
        assert!(lean.kv_bytes >= rich.kv_bytes);
    }

    #[test]
    fn empty_ladder_is_safe() {
        let sel = select_prefill_bucket(43 * GIB, 1 * GIB, &[], 0.6);
        assert_eq!(
            sel,
            PrefillBucketSelection {
                max_bucket_m: 1,
                arena_bytes: 0,
                kv_bytes: 42 * GIB
            }
        );
    }
}
