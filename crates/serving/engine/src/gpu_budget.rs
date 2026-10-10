// SPDX-License-Identifier: Apache-2.0
//! Backend-neutral GPU memory-budget + prefill-bucket helpers.
//!
//! Pure index/arithmetic logic shared by the CUDA and Metal `Worker`
//! implementations. Lives in `scratchy-serving-engine` (which neither the CUDA
//! runtime nor the worker crate cycle through) so both backends can reach it
//! without a dependency cycle.

use scratchy_core_config::SchedulerConfig;

/// The default `max_num_seqs` when no caller, backend, or device tier named one.
///
/// Mirrors Python vLLM `DEFAULT_MAX_NUM_SEQS`. A bare constant is only the
/// LAST resort: a backend that knows its own batched-decode width answers
/// first ([`Worker::max_num_seqs_override`]), and the device-tier table
/// ([`SchedulerConfig::batch_defaults`]) answers before this on backends
/// whose ladders track `max_num_seqs` (cuda/metal).
///
/// [`Worker::max_num_seqs_override`]: crate::worker::Worker::max_num_seqs_override
pub const BASE_MAX_NUM_SEQS: usize = SchedulerConfig::DEFAULT_MAX_NUM_SEQS;

/// The KV floor the OOM guard's flag hint reserves: at least this much of the
/// device budget must stay spendable on KV, not per-sequence state. ONE home
/// — both the load-time default resolver and the guard's refusal hint
/// subtract it, so the default and the refusal can never disagree about
/// what "affordable" means.
pub const KV_FLOOR_BYTES: usize = 1 << 30;

/// Everything the shared unset-`--max-num-seqs` resolver needs. A backend
/// builds this at the END of `load_model` (weights resident, the model's
/// GDN config answerable) and nothing else — the resolver is backend-neutral
/// arithmetic, so metal and cuda can never drift into near-copies of each
/// other again.
#[derive(Debug, Clone)]
pub struct MaxNumSeqsFacts {
    /// Device total memory + name, when the backend can query them — feeds
    /// the device-tier table (`SchedulerConfig::batch_defaults`). `None`
    /// means "no tier answer" and resolves to the base constant: metal
    /// reports nothing (its factory has no `device_total_bytes_and_name`),
    /// so its default is the SAME 128 as before the memory-aware resolver
    /// existed. Any tier bump is a separate, measured decision.
    pub device_total_bytes: Option<u64>,
    /// See [`Self::device_total_bytes`].
    pub device_name: Option<String>,
    /// The utilization-governed budget (`total × gpu_memory_utilization`).
    /// `None` = "no budget to clamp against" (a caller that only wants the
    /// TIER half of the resolution) — the tier answer then survives as-is.
    pub device_budget_bytes: Option<usize>,
    /// What this process already holds at the end of `load_model` (metal:
    /// `currentAllocatedSize`; cuda: `total − free` from `mem_get_info`).
    pub allocated_bytes: usize,
    /// Peak activation estimate the OOM guard's flag-hint path uses — the
    /// prefill-bucket arena (or full-ladder peak), rung scratch, and the
    /// 64 MiB runtime/staging pad. Does NOT include the 150 MiB redundancy
    /// buffer or the per-row terms below; the resolver adds those.
    pub peak_activation_bytes: usize,
    /// The GDN state pool's cost of ONE slot (`reserve_bytes(_, 1, ..)` —
    /// the reservation is linear in `num_slots`). `None` = non-hybrid arch.
    pub gdn_per_slot_bytes: Option<usize>,
    /// The sampler arena's cost per row — the resolver sizes the arena it
    /// itself triggers (`n·(vocab + 2·max_hist)·4` plus the sliced buffers'
    /// terms). 0 when the backend keeps no per-row sampler arena.
    pub sampler_bytes_per_row: usize,
}

impl MaxNumSeqsFacts {
    /// The 150 MiB redundancy buffer `compute_available_kv_bytes` subtracts
    /// alongside weights and activations — the resolver budgets the same
    /// term so an unset default leaves the same KV the formula would.
    const REDUNDANCY_BYTES: usize = 150 * 1024 * 1024;

    /// The largest width whose per-sequence + fixed costs still leave the KV
    /// floor: `budget − allocated − activations − 150 MiB pad −
    /// n·(gdn_per_slot + sampler_row) ≥ KV_FLOOR_BYTES`. Floored at 1 — a
    /// slot count of 0 cannot be built (`GdnStatePool::new` requires
    /// `num_slots >= 1`) and a model always serves at least one sequence.
    fn affordable_width(&self) -> usize {
        // No budget = nothing to clamp against — clamping a tier on absent
        // facts would be a silent width cut, not a memory decision.
        let Some(headroom) = self.kv_headroom_bytes(0) else {
            return usize::MAX;
        };
        let slot_budget = headroom.saturating_sub(KV_FLOOR_BYTES);
        (slot_budget / self.per_row_bytes().max(1)).max(1)
    }

    /// Per-sequence bytes: one GDN state slot plus one sampler-arena row.
    fn per_row_bytes(&self) -> usize {
        self.gdn_per_slot_bytes
            .unwrap_or(0)
            .saturating_add(self.sampler_bytes_per_row)
    }

    /// The bytes left for KV at `width` resident sequences: the budget less what is allocated,
    /// the activation peak, the 150 MiB pad and `width` per-sequence rows. `None` without a
    /// budget.
    pub fn kv_headroom_bytes(&self, width: usize) -> Option<usize> {
        let fixed = self
            .allocated_bytes
            .saturating_add(self.peak_activation_bytes)
            .saturating_add(Self::REDUNDANCY_BYTES)
            .saturating_add(width.saturating_mul(self.per_row_bytes()));
        Some(self.device_budget_bytes?.saturating_sub(fixed))
    }
}

/// The share of the KV headroom recurrent-state snapshots may take (1/8).
///
/// A snapshot is only useful beside the KV of the prefix it resumes, so KV keeps the larger part;
/// an eighth still buys ~47 snapshots of Qwen3.6-35B-A3B's 62.8 MiB state on a 64 GB Mac, enough
/// for a long agent loop plus its side requests, and a handful on a 32 GB one.
pub const SNAPSHOT_HEADROOM_SHARE: usize = 8;

/// At most this many recurrent-state snapshots, however roomy the device: past it they are
/// wired memory the prefix cache of a single box does not use.
pub const MAX_RECURRENT_SNAPSHOTS: usize = 64;

/// Below this many snapshots prefix caching on a recurrent-hybrid model is not worth its extra
/// forward per prompt: any second request evicts the first one's snapshot.
pub const MIN_RECURRENT_SNAPSHOTS: usize = 2;

/// ⭐ HOW MANY RECURRENT-STATE SNAPSHOTS (prefix-cache resume points of a Gated-DeltaNet hybrid's
/// per-sequence state) to keep, given `headroom_bytes` of KV headroom before any are reserved and
/// `snapshot_bytes` per snapshot (one state slot).
///
/// [`SNAPSHOT_HEADROOM_SHARE`] of the headroom, never past [`KV_FLOOR_BYTES`] of KV left, at most
/// [`MAX_RECURRENT_SNAPSHOTS`]; 0 — prefix caching off for the model — below
/// [`MIN_RECURRENT_SNAPSHOTS`]. Derived from the same facts as the width, so neither a flag nor the
/// OOM guard's arithmetic has to know about it.
pub fn recurrent_snapshot_slots(headroom_bytes: usize, snapshot_bytes: usize) -> usize {
    if snapshot_bytes == 0 {
        return 0;
    }
    let share = headroom_bytes / SNAPSHOT_HEADROOM_SHARE / snapshot_bytes;
    let above_floor = headroom_bytes.saturating_sub(KV_FLOOR_BYTES) / snapshot_bytes;
    let n = share.min(above_floor).min(MAX_RECURRENT_SNAPSHOTS);
    if n < MIN_RECURRENT_SNAPSHOTS { 0 } else { n }
}

/// ⭐ THE WIDTH AND THE RECURRENT-STATE SNAPSHOT COUNT, resolved together from one set of facts:
/// `(max_num_seqs, snapshots)`.
///
/// `asked` is an explicit `--max-num-seqs`, honoured as is (snapshots then share what that width
/// leaves). `snapshot_bytes` is one snapshot — `None` when the model has no recurrent state or
/// the worker keeps no snapshots, which leaves the width exactly what
/// [`resolve_default_max_num_seqs`] answers.
///
/// Unset: snapshots take [`recurrent_snapshot_slots`] of what the default width leaves for KV —
/// or, on a box where that width already spends the headroom down to the KV floor, of what ONE
/// sequence leaves, paid for in width (the width is then resolved with the snapshots counted as
/// allocated). There resuming a long prompt is worth more than the last few resident
/// sequences, and taking it from KV would leave none. One, not zero: the width never resolves
/// below 1, so sizing against zero rows could put KV under the floor.
pub fn resolve_width_and_snapshots(
    facts: &MaxNumSeqsFacts,
    asked: Option<usize>,
    snapshot_bytes: Option<usize>,
    is_offline: bool,
) -> (usize, usize) {
    let snapshots_at = |facts: &MaxNumSeqsFacts, width: usize| match (
        facts.kv_headroom_bytes(width),
        snapshot_bytes,
    ) {
        (Some(headroom), Some(bytes)) => recurrent_snapshot_slots(headroom, bytes),
        _ => 0,
    };
    if let Some(width) = asked {
        return (width, snapshots_at(facts, width));
    }
    let width = resolve_default_max_num_seqs(facts, is_offline);
    let snapshots = match snapshots_at(facts, width) {
        0 => snapshots_at(facts, 1),
        n => n,
    };
    let mut charged = facts.clone();
    charged.allocated_bytes += snapshots * snapshot_bytes.unwrap_or(0);
    (
        resolve_default_max_num_seqs(&charged, is_offline),
        snapshots,
    )
}

/// Resolve an UNSET `--max-num-seqs` from device memory + per-sequence state
/// needs. The tier answer (`SchedulerConfig::batch_defaults` when the
/// backend reports device facts, else the base constant) clamped to
/// [`MaxNumSeqsFacts::affordable_width`] — the same terms the OOM guard's
/// flag hint uses, so the default a bare `scr serve` runs at never trips the
/// guard it is sized against.
///
/// `is_offline` selects the LLM/throughput vs online-server tier context.
/// An explicit ask NEVER comes through here: the worker honours it verbatim
/// and the guard refuses an unaffordable one (capping-is-not-validating).
pub fn resolve_default_max_num_seqs(facts: &MaxNumSeqsFacts, is_offline: bool) -> usize {
    let tier = match facts.device_total_bytes {
        // The name only downgrades a ≥70 GiB A100 back to the small tier
        // (large batched-token budgets regress it — Python vLLM PR #17885);
        // a backend that reports no name (metal) can never be an A100, so
        // `unwrap_or("")` keeps the tier decision intact.
        Some(total) => {
            SchedulerConfig::batch_defaults(
                total,
                facts.device_name.as_deref().unwrap_or(""),
                is_offline,
            )
            .1
        }
        None => BASE_MAX_NUM_SEQS,
    };
    let affordable = facts.affordable_width();
    if affordable < tier { affordable } else { tier }
}

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
mod max_num_seqs_default_tests {
    use super::{BASE_MAX_NUM_SEQS, KV_FLOOR_BYTES, MaxNumSeqsFacts, resolve_default_max_num_seqs};

    const GIB_U64: u64 = 1024 * 1024 * 1024;
    const GIB: usize = 1024 * 1024 * 1024;
    const MIB: usize = 1024 * 1024;

    /// The roomy baseline: every per-sequence and fixed term comfortably
    /// covered, so the tier answer survives untouched. Metal-shaped (no
    /// device facts → the base constant, SAME 128 as before the resolver).
    fn roomy(over: impl FnOnce(&mut MaxNumSeqsFacts)) -> MaxNumSeqsFacts {
        let mut f = MaxNumSeqsFacts {
            device_total_bytes: None,
            device_name: None,
            device_budget_bytes: Some(24 * GIB),
            allocated_bytes: 4 * GIB,
            peak_activation_bytes: 512 * MIB,
            gdn_per_slot_bytes: Some(19 * MIB),
            sampler_bytes_per_row: 8 * MIB,
        };
        over(&mut f);
        f
    }

    /// The criterion the resolver must satisfy at a clamped width, checked
    /// against the RESOLVED number rather than a recomputation of the
    /// division: every fixed + per-row cost the facts name, PLUS the KV
    /// floor, fits inside the budget — i.e. a default never trips the OOM
    /// guard it is sized against.
    fn kv_left(f: &MaxNumSeqsFacts, width: usize) -> i64 {
        let per_row = f.gdn_per_slot_bytes.unwrap_or(0) + f.sampler_bytes_per_row;
        let fixed = f.allocated_bytes + f.peak_activation_bytes + MaxNumSeqsFacts::REDUNDANCY_BYTES;
        f.device_budget_bytes.unwrap_or(0) as i64 - fixed as i64 - (per_row * width) as i64
    }

    /// A starved box (budget below every fixed term) still serves one
    /// sequence — the resolver floors at 1, never 0.
    #[test]
    fn a_starved_box_still_serves_one_sequence() {
        let starved = roomy(|f| {
            f.device_budget_bytes = Some(512 * MIB);
            f.allocated_bytes = 0;
        });
        assert_eq!(resolve_default_max_num_seqs(&starved, false), 1);
    }

    #[test]
    fn unqueryable_device_falls_back_to_the_base_constant() {
        assert_eq!(
            resolve_default_max_num_seqs(&roomy(|_| {}), false),
            BASE_MAX_NUM_SEQS
        );
        assert_eq!(
            resolve_default_max_num_seqs(&roomy(|_| {}), true),
            BASE_MAX_NUM_SEQS
        );
    }

    #[test]
    fn small_gpu_tier_defaults_to_256_and_a_large_one_to_1024() {
        assert_eq!(
            resolve_default_max_num_seqs(
                &roomy(|f| {
                    f.device_total_bytes = Some(24 * GIB_U64);
                    f.device_name = Some("NVIDIA L4".into());
                }),
                true
            ),
            256
        );
        assert_eq!(
            resolve_default_max_num_seqs(
                &roomy(|f| {
                    f.device_total_bytes = Some(80 * GIB_U64);
                    f.device_name = Some("NVIDIA H100 80GB HBM3".into());
                    // An 80 GiB card's facts: a 72 GiB budget with the same
                    // per-row and overhead terms comfortably covers 1024
                    // rows, so the tier answer survives.
                    f.device_budget_bytes = Some(72 * GIB);
                }),
                true
            ),
            1024
        );
    }

    /// The Qwen3.5-MoE-35B "!!!!" incident as an arithmetic test: ~61 MiB per
    /// slot on a box whose headroom after weights + activations + pads is
    /// ~2 GiB must NOT answer the base 128 (7.9 GiB of f32 state) — it
    /// answers what fits above the 1 GiB KV floor.
    #[test]
    fn the_incident_box_answers_the_affordable_count_not_the_default() {
        let f = roomy(|f| {
            f.device_budget_bytes = Some(5 * GIB);
            f.allocated_bytes = 2 * GIB;
            f.peak_activation_bytes = 300 * MIB;
            f.gdn_per_slot_bytes = Some(61 * MIB);
            f.sampler_bytes_per_row = 0;
        });
        let w = resolve_default_max_num_seqs(&f, false);
        assert!(w < BASE_MAX_NUM_SEQS, "128 was the incident, got {w}");
        assert!(w >= 16, "2 GiB minus floors buys ~17 slots, got {w}");
        // The criterion, not the recomputation: the KV left at the resolved
        // width is ≥ the floor; one more row would breach it.
        assert!(kv_left(&f, w) >= KV_FLOOR_BYTES as i64);
        assert!(kv_left(&f, w + 1) < KV_FLOOR_BYTES as i64);
    }

    /// The clamp must count the SAMPLER ARENA row too — the width sizes the
    /// very arena allocated right after resolution (audit point 2: the old
    /// clamp "leaves out the sampler arena it sizes itself").
    #[test]
    fn the_sampler_row_counts_against_the_width() {
        let base = |sampler_row: usize| {
            roomy(|f| {
                f.device_budget_bytes = Some(6 * GIB);
                f.allocated_bytes = 2 * GIB;
                f.peak_activation_bytes = 300 * MIB;
                f.gdn_per_slot_bytes = None;
                f.sampler_bytes_per_row = sampler_row;
            })
        };
        let bare = resolve_default_max_num_seqs(&base(0), false);
        let with_arena = resolve_default_max_num_seqs(&base(32 * MIB), false);
        assert!(
            with_arena < bare,
            "a per-row arena cost must shrink the width: {with_arena} vs {bare}"
        );
        let f = base(32 * MIB);
        assert!(kv_left(&f, with_arena) >= KV_FLOOR_BYTES as i64);
    }

    #[test]
    fn a_roomy_box_is_not_clamped() {
        // 60 GiB of headroom at 61 MiB/slot affords ~970 slots — more than
        // any tier default, so the caller keeps the default number.
        let w = resolve_default_max_num_seqs(
            &roomy(|f| {
                f.device_budget_bytes = Some(64 * GIB);
                f.allocated_bytes = 4 * GIB;
                f.gdn_per_slot_bytes = Some(61 * MIB);
            }),
            false,
        );
        assert_eq!(w, BASE_MAX_NUM_SEQS);
    }
}

#[cfg(test)]
mod recurrent_snapshot_tests {
    use super::{
        BASE_MAX_NUM_SEQS, KV_FLOOR_BYTES, MAX_RECURRENT_SNAPSHOTS, MaxNumSeqsFacts,
        recurrent_snapshot_slots, resolve_default_max_num_seqs, resolve_width_and_snapshots,
    };

    const GIB: usize = 1024 * 1024 * 1024;
    const MIB: usize = 1024 * 1024;
    /// Qwen3.6-35B-A3B: 30 GDN layers × (8192×3 + 32×128×128) f32.
    const SNAP: usize = 65_863_680;

    #[test]
    fn an_eighth_of_the_headroom_on_a_roomy_box() {
        // The 64 GB M5 Max at width 1: ~23.5 GiB of KV headroom.
        let n = recurrent_snapshot_slots(23 * GIB + GIB / 2, SNAP);
        assert_eq!(n, 47);
        assert!(n * SNAP <= (23 * GIB + GIB / 2) / 8);
    }

    #[test]
    fn capped_on_a_huge_box_and_off_on_a_starved_one() {
        assert_eq!(
            recurrent_snapshot_slots(400 * GIB, SNAP),
            MAX_RECURRENT_SNAPSHOTS
        );
        // An eighth of 1 GiB is 2 snapshots, but nothing may come out of the KV floor.
        assert_eq!(recurrent_snapshot_slots(GIB, SNAP), 0);
        // One that fits is not worth an extra forward per prompt.
        assert_eq!(
            recurrent_snapshot_slots(KV_FLOOR_BYTES + SNAP + MIB, SNAP),
            0
        );
        assert_eq!(recurrent_snapshot_slots(10 * GIB, 0), 0);
    }

    /// Qwen3.6-35B-A3B on a box shaped by `budget` GiB, with weights and the arena allocated.
    fn box_with(budget: usize, allocated: usize) -> MaxNumSeqsFacts {
        MaxNumSeqsFacts {
            device_total_bytes: None,
            device_name: None,
            device_budget_bytes: Some(budget),
            allocated_bytes: allocated,
            peak_activation_bytes: 64 * MIB,
            gdn_per_slot_bytes: Some(SNAP),
            sampler_bytes_per_row: 0,
        }
    }

    /// KV left once `width` sequences and `snapshots` snapshots are paid for.
    fn kv_after(f: &MaxNumSeqsFacts, width: usize, snapshots: usize) -> usize {
        f.kv_headroom_bytes(width).unwrap() - snapshots * SNAP
    }

    #[test]
    fn a_roomy_box_keeps_its_default_width_and_shares_the_kv_left_beside_it() {
        // The 64 GB M5 Max: 46.7 GiB budget, ~22.3 GiB of weights and arena.
        let f = box_with(46 * GIB + 700 * MIB, 22 * GIB + 300 * MIB);
        let (width, snapshots) = resolve_width_and_snapshots(&f, None, Some(SNAP), false);
        assert_eq!(width, BASE_MAX_NUM_SEQS);
        let left = f.kv_headroom_bytes(width).unwrap();
        assert_eq!(snapshots, recurrent_snapshot_slots(left, SNAP));
        assert!(snapshots * SNAP <= left / 8);
        assert!(kv_after(&f, width, snapshots) >= KV_FLOOR_BYTES);
    }

    #[test]
    fn a_squeezed_box_pays_for_snapshots_in_width_and_keeps_the_kv_floor() {
        // The 32 GB incident shape: the default width spends the headroom to the floor.
        let f = box_with(22 * GIB + GIB / 2, 19 * GIB + 600 * MIB);
        let bare = resolve_default_max_num_seqs(&f, false);
        assert_eq!(
            recurrent_snapshot_slots(f.kv_headroom_bytes(bare).unwrap(), SNAP),
            0,
            "the shape this test is about"
        );
        let (width, snapshots) = resolve_width_and_snapshots(&f, None, Some(SNAP), false);
        assert!(
            snapshots >= 2,
            "a squeezed box still gets a few, got {snapshots}"
        );
        assert!(width < bare && width >= 1, "{width} vs {bare}");
        assert!(kv_after(&f, width, snapshots) >= KV_FLOOR_BYTES);
    }

    #[test]
    fn the_last_band_above_the_floor_never_dips_below_it() {
        // Headroom for the floor, two snapshots and less than one more row: the width floors at
        // 1, so the snapshots must be sized against one row, not zero.
        for extra in [0, MIB, 30 * MIB, 62 * MIB] {
            let budget = 20 * GIB;
            let headroom = KV_FLOOR_BYTES + 2 * SNAP + extra;
            let f = box_with(budget, budget - headroom - 64 * MIB - 150 * MIB);
            let (width, snapshots) = resolve_width_and_snapshots(&f, None, Some(SNAP), false);
            assert!(width >= 1);
            assert!(
                kv_after(&f, width, snapshots) >= KV_FLOOR_BYTES,
                "extra {extra}: width {width}, {snapshots} snapshots"
            );
        }
    }

    #[test]
    fn an_explicit_width_is_honoured_and_snapshots_share_what_it_leaves() {
        let f = box_with(46 * GIB + 700 * MIB, 22 * GIB + 300 * MIB);
        let (width, snapshots) = resolve_width_and_snapshots(&f, Some(1), Some(SNAP), false);
        assert_eq!(width, 1);
        assert_eq!(
            snapshots,
            recurrent_snapshot_slots(f.kv_headroom_bytes(1).unwrap(), SNAP)
        );
        let (width, snapshots) = resolve_width_and_snapshots(&f, Some(400), Some(SNAP), false);
        assert_eq!(
            (width, snapshots),
            (400, 0),
            "an ask that spends the headroom gets none"
        );
    }

    #[test]
    fn without_snapshots_the_width_is_the_resolvers() {
        let f = box_with(22 * GIB + GIB / 2, 19 * GIB + 600 * MIB);
        assert_eq!(
            resolve_width_and_snapshots(&f, None, None, false),
            (resolve_default_max_num_seqs(&f, false), 0)
        );
    }
}

#[cfg(test)]
mod bucket_selection_tests {
    use super::{KV_FLOOR_BYTES, PrefillBucketSelection, select_prefill_bucket};

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

    /// The KV-floor interaction `determine_available_memory` runs: a model
    /// whose resident set fits Apple's recommended working set but not
    /// `util × working_set` (the 48 GiB gpt-oss-120b-3bit on a 64 GiB M5
    /// Max: raw budget 46.8 GiB < weights 48.2 GiB) must not collapse the
    /// prefill ladder to the fallback — the floor lifts the budget past the
    /// resident set BEFORE the selector runs, and the selector, handed the
    /// floored budget, recovers a real bucket.
    #[test]
    fn the_kv_floor_reaches_the_ladder_instead_of_collapsing_it() {
        let working_set = 52 * GIB;
        let utilization_budget = 46_800 * MIB; // 0.9 × working_set
        let weights = 48_200 * MIB;
        let pad = 214 * MIB; // (64 + 150) MiB — the worker's non-arena terms
        let fixed = weights + pad;

        // The RAW budget: the resident set overflows it entirely — the same
        // fallback `severely_starved_falls_back_to_smallest` pins, except
        // here the device holds 3.6 GiB of working set the budget never
        // saw. THE DEFECT: m=1 prefill steps at ~15× decode cost (TTFT
        // linear in prompt length) while the box sits on free headroom.
        let raw = select_prefill_bucket(utilization_budget, fixed, &ladder(), 0.6);
        assert_eq!(raw.max_bucket_m, 1);

        // One pass of the worker's fixed point: floor the budget at
        // resident + the raw-budget estimate (the collapsed arena plus its
        // pads/rung), capped at the working set.
        let est_raw = raw.arena_bytes + 64 * MIB + 90 * MIB;
        let floored = utilization_budget
            .max(weights + est_raw + KV_FLOOR_BYTES as u64)
            .min(working_set);

        let sel = select_prefill_bucket(floored, fixed, &ladder(), 0.6);
        assert!(
            sel.max_bucket_m >= 512,
            "the floor must recover a real bucket, got {sel:?}"
        );

        // Monotone — the loop's convergence law: re-estimating under the
        // floored budget (the bigger arena the recovered ladder actually
        // runs) and re-flooring only LIFTS the budget, never un-floors it,
        // so iterating the two reaches a fixed point instead of oscillating.
        let est_floored = sel.arena_bytes + 64 * MIB + 90 * MIB;
        let refloored = utilization_budget
            .max(weights + est_floored + KV_FLOOR_BYTES as u64)
            .min(working_set);
        assert!(refloored >= floored);
    }
}
