// SPDX-License-Identifier: Apache-2.0
//! Solver: match kernels to FUF tiles, produce the SFUF.
//!
//! Real dynamic-programming over impl choice. State is
//! `(position i, claim_state)` where `claim_state` is a K-bit
//! bitmask encoding which of positions `[i, i+K)` are pre-claimed
//! by a multi-tile impl committed at an earlier position.
//!
//! Recurrence:
//! ```text
//! dp[i][cs] =
//!     if cs bit 0 set:                      // i already claimed
//!         dp[i+1][cs >> 1]
//!     else:
//!         min over candidates c at position i (c.mask & cs == 0):
//!             per_impl_cost(c) + dp[i+1][(cs | c.mask) >> 1]
//! ```
//!
//! `c.mask` is `c`'s claim pattern as a bitmask with bit 0 = seed
//! position. After moving past position i, the bitmask shifts left
//! (bit 0 drops off).
//!
//! This is optimal over impl choice given per-impl costs. Contention
//! (the wave-level 0.5× shadowing of memory-bound impls behind
//! compute-bound impls) is applied later in `cost::loop_cost_us`
//! after `schedule::schedule()` builds waves — the solver doesn't
//! need to know about waves.
//!
//! Complexity: O(n × 2^K × C) where n = tiles, K = max claim spread
//! in the library, C = candidates per position. For today's library
//! max spread is ~4 and K=8 is plenty of headroom.
//!
//! NOT ported from the old DP (llama-specific hacks):
//!
//! - The `matches!(node.kind, TileKind::ResidualAdd)` auto-claim
//!   fallback for tiles no impl matched. An unmatched tile is a
//!   library bug, not something to paper over.
//! - The `TileKind::GemmQ/K/V/...` taxonomy and any code that
//!   matched on it. Our OpKind is uniform.
//! - `weight_name: String` and weight-name substring checks.
//!
//! The workload sweep stays: for each `num_tokens` the caller asks
//! about, we produce one SFUF. Codegen coalesces adjacent-equal
//! SFUFs into match arms.
//!
//! Output: a [`WorkloadAssignments`] keyed by `num_tokens`. Each
//! [`Assignment`] (== SFUF) has `cover: tile → SubgraphId`, `impls:
//! SubgraphId → ImplId`, and `predicted_us`.

#![allow(dead_code)]

use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap};

use crate::assignment::ImplId;
use crate::assignment::{Assignment, SubgraphId, WorkloadAssignments, WorkloadPoint};
use crate::classified::{OpKind, Program};
use crate::config::ModelParams;
use crate::fuf::{Fuf, FufInput, FufNode, TileId};
use crate::impl_lib::{CostCtx, ImplementationLibrary, MatchContext, MatchInfo};
use crate::shape::{Inferred, Shape, extern_shape};
use crate::target::TargetProfile;

#[derive(Debug)]
pub enum SolveError {
    /// No Impl in the library matched the given tile at this
    /// workload point. Library is incomplete, or a workload/target
    /// filter excluded the only viable Impl. Never papered over by
    /// an auto-claim fallback.
    UnclaimedTile {
        tile: TileId,
        op: OpKind,
        point: WorkloadPoint,
        /// Guard G3: tiles covered by NO candidate claim from any seed
        /// — the genuinely-unclaimable ops. Suppresses the lone-`Mul`
        /// red herring (a fusion-only tile IS covered, so it's absent).
        orphan_ops: Vec<(TileId, OpKind)>,
        /// Guard G3: per-registered-Impl reason it declined the first
        /// reported orphan (target_compatible / matches /
        /// workload_constraint). `applies_to` is NOT re-checked here.
        declines: Vec<String>,
    },
    /// An Impl's cost function returned None despite having a
    /// match and passing applicability filters. Means either (a)
    /// shape inference left a Var, or (b) a bug in the cost fn.
    /// Silently skipping the Impl would mask the bug — hard error.
    UnreachableCost {
        tile: TileId,
        op: OpKind,
        impl_id: ImplId,
        point: WorkloadPoint,
    },
    /// Guard G4: the DP PICKED an impl for a tile, but the impl's
    /// `validate_pick` precondition (mirroring its lowering assert)
    /// rejected the model's bounds. Catches at compile time what would
    /// otherwise be a worker-thread runtime panic at lowering.
    PickPreconditionFailed {
        impl_name: &'static str,
        tile: TileId,
        op: OpKind,
        point: WorkloadPoint,
        why: String,
    },
}

impl std::fmt::Display for SolveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnclaimedTile {
                tile,
                op,
                point,
                orphan_ops,
                declines,
            } => {
                write!(
                    f,
                    "no Impl in the library matched tile {} op {:?} at \
                     num_tokens={} sk_bucket={}",
                    tile.0, op, point.num_tokens, point.sk_bucket
                )?;
                if orphan_ops.is_empty() {
                    write!(
                        f,
                        "\nall tiles have candidates but no feasible cover — likely K too \
                         small or a claim reaches backward (not a missing-Impl case)"
                    )?;
                } else {
                    write!(
                        f,
                        "\ngenuinely-unclaimable tile(s) (covered by NO candidate claim from \
                         any seed):"
                    )?;
                    for (t, o) in orphan_ops {
                        write!(f, "\n  tile {} op {:?}", t.0, o)?;
                    }
                }
                if !declines.is_empty() {
                    write!(
                        f,
                        "\nwhy each registered Impl declined tile {} (op {:?}) \
                         [applies_to NOT checked here]:",
                        tile.0, op
                    )?;
                    for d in declines {
                        write!(f, "\n{d}")?;
                    }
                }
                write!(
                    f,
                    "\nAdd or fix an Impl that claims op {op:?} (its matches() / \
                     target_compatible() must return true for this tile), in impl_lib.rs / \
                     crates/compiler/macros/src/metal/."
                )
            }
            Self::UnreachableCost {
                tile,
                op,
                impl_id,
                point,
            } => write!(
                f,
                "impl {} could not cost tile {} op {:?} at \
                 num_tokens={} sk_bucket={} (cost_fn returned None \
                 despite invariant that shapes are closed)",
                impl_id.0, tile.0, op, point.num_tokens, point.sk_bucket
            ),
            Self::PickPreconditionFailed {
                impl_name,
                tile,
                op,
                point,
                why,
            } => write!(
                f,
                "impl {impl_name} picked for tile {} op {op:?} at num_tokens={} \
                 sk_bucket={}, but precondition failed: {why}",
                tile.0, point.num_tokens, point.sk_bucket
            ),
        }
    }
}

impl std::error::Error for SolveError {}

/// Solve each `(num_tokens, sk_bucket)` point independently.
/// `bounds` supplies every symbolic bound except `num_tokens` and
/// `sk_bucket`; the solver overwrites both per point. Empty
/// `sk_points` defaults to `&[0]` — the 1-D legacy sweep.
pub fn solve(
    fuf: &Fuf,
    lib: &ImplementationLibrary,
    target: &TargetProfile,
    inferred: &Inferred,
    bounds: &BTreeMap<String, u64>,
    num_tokens_points: &[u64],
    sk_points: &[u64],
) -> Result<WorkloadAssignments, SolveError> {
    solve_with_arch_filter(
        fuf,
        lib,
        target,
        None,
        inferred,
        bounds,
        num_tokens_points,
        sk_points,
        1,
    )
}

/// Variant of [`solve`] that wires a per-canonical
/// `(Program, ModelParams)` context into [`Implementation::arch_filter`].
/// Production drive uses this path so per-arch matchers (today: the
/// three MoE Impls — `DeepSeekMoeRefImpl` /  `FusedMoeRefImpl` /
/// `SharedFusedMoeRefImpl`) can disambiguate by config keys before
/// claiming. Tests that don't exercise per-arch gating stay on
/// `solve` (no churn), which forwards `arch_ctx = None`; with no
/// context the filter is treated as default-true.
#[allow(clippy::too_many_arguments)]
pub fn solve_with_arch_filter(
    fuf: &Fuf,
    lib: &ImplementationLibrary,
    target: &TargetProfile,
    arch_ctx: Option<(&Program, &ModelParams)>,
    inferred: &Inferred,
    bounds: &BTreeMap<String, u64>,
    num_tokens_points: &[u64],
    sk_points: &[u64],
    tp_world_size: u8,
) -> Result<WorkloadAssignments, SolveError> {
    use rayon::prelude::*;

    // Empty sk_points = legacy 1-D sweep. `sk_bucket = 0` is the
    // sentinel "sk axis unused"; FI impls with a real sk range
    // simply won't match at sk_bucket = 0.
    let sk_effective: Vec<u64> = if sk_points.is_empty() {
        vec![0]
    } else {
        sk_points.to_vec()
    };

    let points: Vec<WorkloadPoint> = num_tokens_points
        .iter()
        .flat_map(|&m| {
            sk_effective.iter().map(move |&sk| WorkloadPoint {
                num_tokens: m,
                sk_bucket: sk,
            })
        })
        .collect();

    // ── Precompute workload-invariant match info per (tile, impl). ──
    //
    // `target_compatible` and `matches` don't depend on `num_tokens`
    // or `sk_bucket` — only on target + FUF structure. Before the
    // 2-D sweep landed, solving ran 5 points per model and the
    // redundant enumeration didn't matter; at 20 points it burns
    // ~N_points× extra work through `matches()` (which walks tile
    // inputs). Cache once per model, reuse across every point.
    //
    // `match_cache[i]` = list of `(impl_id, match_info)` for tile
    // `fuf.nodes[i]`, filtered by target-compat and whose
    // `matches()` returned Some. Per-point work then only re-checks
    // `workload_constraint` + computes `cost_us`, avoiding the
    // expensive structural match.
    // Per-canonical Impl gate: evaluate `applies_to` once per Impl
    // (sequentially — `Program`/`ModelParams` carry proc_macro2 Spans
    // that aren't `Sync`) and cache the disabled set as a Vec<bool>
    // keyed by ImplId. The parallel match loop below then only reads
    // the bool, sidestepping the Sync issue while keeping the
    // tile-level loop fully parallel.
    let disabled_impls: Vec<bool> = if let Some((program, model)) = arch_ctx {
        let ctx = MatchContext {
            program,
            model,
            profile: target,
            tp_world_size,
        };
        lib.iter_enumerated()
            .map(|(_, imp)| !imp.applies_to(&ctx))
            .collect()
    } else {
        vec![false; lib.len()]
    };

    let t_cache = std::time::Instant::now();
    let match_cache: Vec<Vec<(ImplId, MatchInfo)>> = fuf
        .nodes
        .par_iter()
        .map(|node| {
            let mut out = Vec::new();
            for (imp_id, imp) in lib.iter_enumerated() {
                if disabled_impls[imp_id.0 as usize] {
                    continue;
                }
                if !imp.target_compatible(target) {
                    continue;
                }
                if let Some(info) = imp.matches(fuf, node.id, target) {
                    out.push((imp_id, info));
                }
            }
            out
        })
        .collect();
    let d_cache = t_cache.elapsed();

    // Each workload point is an independent solve: the DP table and
    // matches_at vector are rebuilt from scratch per point (workload
    // constraints and per-impl costs vary with both axes). Running
    // them in parallel drops the total from `Σ per-workload` to
    // `max(per-workload)` on well-parallel hardware.
    //
    // Per-phase timing is summed across workers via atomic counters
    // so the macro driver (lib.rs) can print a breakdown per model
    // and we can stop guessing where time is going.
    use std::sync::atomic::{AtomicU64, Ordering};
    let ns_phase1 = AtomicU64::new(0);
    let ns_phase2 = AtomicU64::new(0);
    let ns_phase3 = AtomicU64::new(0);
    let ns_phase4 = AtomicU64::new(0);

    let solved: Vec<Result<(WorkloadPoint, Assignment), SolveError>> = points
        .par_iter()
        .map(|&wp| {
            let mut scratch = bounds.clone();
            scratch.insert("num_tokens".into(), wp.num_tokens);
            scratch.insert("sk_bucket".into(), wp.sk_bucket);
            solve_one(
                fuf,
                lib,
                target,
                inferred,
                &scratch,
                wp,
                &match_cache,
                &ns_phase1,
                &ns_phase2,
                &ns_phase3,
                &ns_phase4,
            )
            .map(|a| (wp, a))
        })
        .collect();

    let mut per_workload: BTreeMap<WorkloadPoint, Assignment> = BTreeMap::new();
    for result in solved {
        let (wp, a) = result?;
        per_workload.insert(wp, a);
    }

    if crate::scratchy_debug() {
        eprintln!(
            "    solve-profile: tiles={} impls={} points={} | cache={}ms p1(cost+filter)={}ms p2(mask)={}ms p3(dp)={}ms p4(reconstruct)={}ms",
            fuf.len(),
            lib.len(),
            points.len(),
            d_cache.as_millis(),
            ns_phase1.load(Ordering::Relaxed) / 1_000_000,
            ns_phase2.load(Ordering::Relaxed) / 1_000_000,
            ns_phase3.load(Ordering::Relaxed) / 1_000_000,
            ns_phase4.load(Ordering::Relaxed) / 1_000_000,
        );
    }

    Ok(WorkloadAssignments { per_workload })
}

/// One pass of the DP over the whole FUF at a single workload
/// point. Returns the SFUF.
#[allow(clippy::too_many_arguments)]
fn solve_one(
    fuf: &Fuf,
    lib: &ImplementationLibrary,
    target: &TargetProfile,
    inferred: &Inferred,
    bounds: &BTreeMap<String, u64>,
    point: WorkloadPoint,
    match_cache: &[Vec<(ImplId, MatchInfo)>],
    ns_phase1: &std::sync::atomic::AtomicU64,
    ns_phase2: &std::sync::atomic::AtomicU64,
    ns_phase3: &std::sync::atomic::AtomicU64,
    ns_phase4: &std::sync::atomic::AtomicU64,
) -> Result<Assignment, SolveError> {
    use std::sync::atomic::Ordering as AtomicOrdering;
    let num_tokens = point.num_tokens;
    let sk_bucket = point.sk_bucket;
    let n = fuf.len();
    if n == 0 {
        return Ok(Assignment {
            cover: HashMap::new(),
            impls: HashMap::new(),
            predicted_us: 0.0,
        });
    }

    let t_p1 = std::time::Instant::now();
    // ── Phase 1: per-point filter + cost evaluation ──
    let mut matches_at: Vec<Vec<(ImplId, MatchInfo, f64)>> = vec![Vec::new(); n];

    let ctx = CostCtx {
        fuf,
        profile: target,
        bounds,
        expected_calls_per_load: CostCtx::default_expected_calls_per_load(target.backend),
    };
    let amortization_n = ctx.expected_calls_per_load as f64;

    for (i, node) in fuf.nodes.iter().enumerate() {
        for (imp_id, info) in &match_cache[i] {
            let imp = lib.get(*imp_id);
            if !imp
                .workload_constraint()
                .accepts(num_tokens as u32, sk_bucket)
            {
                continue;
            }
            // Amortized cost = startup + N × per_call. Defaults to
            // pure per-call (startup_us = 0.0 trait default), so any
            // impl that doesn't override `startup_us` ranks identically
            // to before. Impls with packed weight accessors (e.g.
            // `MetalFusedGateUpSiluMulImpl`) override `startup_us`
            // with the load-time CPU memcpy cost; the solver picks
            // the unpacked alternative when N is small enough that
            // the per-call savings can't pay for the pack.
            let per_call = imp.cost_us(info, &ctx);
            let startup = imp.startup_us(info, &ctx);
            let cost = startup + amortization_n * per_call;
            if !cost.is_finite() {
                return Err(SolveError::UnreachableCost {
                    tile: node.id,
                    op: node.op,
                    impl_id: *imp_id,
                    point,
                });
            }
            matches_at[i].push((*imp_id, info.clone(), cost));
        }

        // Sort: claim size DESCENDING, then cost ASCENDING.
        //
        // Multi-tile claims represent fusion — by construction they
        // are preferable to covering the same tiles with separate
        // singletons (that is literally what fusion means: one
        // launch + shared memory traffic instead of N launches).
        // Picking the smaller claim at the seed can *orphan*
        // downstream tiles that only a multi-tile claim can cover
        // (e.g. Silu/Mul have no singleton impl — `silu_and_mul_fused`
        // is the only kernel). The greedy has to prefer the larger
        // claim to stay correct, not just optimal.
        //
        // Singleton-vs-singleton at the same seed falls through to
        // the cost tiebreak.
        matches_at[i].sort_by(|a, b| {
            b.1.claimed_tiles
                .len()
                .cmp(&a.1.claimed_tiles.len())
                .then_with(|| a.2.partial_cmp(&b.2).unwrap_or(Ordering::Equal))
        });
    }
    let _ = inferred; // no longer used here — shapes come via ctx.fuf
    ns_phase1.fetch_add(t_p1.elapsed().as_nanos() as u64, AtomicOrdering::Relaxed);

    let t_p2 = std::time::Instant::now();
    // ── Phase 2: convert candidate claim_tiles to bitmasks ──
    //
    // Each candidate's claim_tiles becomes a K-bit mask relative to
    // its seed position. Bit 0 = seed; bit j = seed + j. Candidates
    // whose claim exceeds K bits or reaches backward in topo order
    // are dropped (with the matches_at[i] entry removed) — they'd
    // be invariant violations for this DP. FusedQkvQkNormRopeCacheImpl
    // spans up to 12 tiles per layer (Gemma3 QK-norm chain), so K=16.
    const K: usize = 16;
    type ClaimMask = u16;

    let candidates: Vec<Vec<Candidate>> = matches_at
        .iter()
        .enumerate()
        .map(|(i, v)| {
            v.iter()
                .filter_map(|(imp_id, info, cost)| {
                    let mut mask: ClaimMask = 0;
                    for t in &info.claimed_tiles {
                        let pos = t.0 as usize;
                        if pos < i {
                            return None; // claims backward — reject
                        }
                        let off = pos - i;
                        if off >= K {
                            return None; // spread exceeds K bits — reject
                        }
                        mask |= 1 << off;
                    }
                    if mask & 1 == 0 {
                        return None; // seed must be in the claim
                    }
                    Some(Candidate {
                        imp_id: *imp_id,
                        mask,
                        cost: *cost,
                    })
                })
                .collect()
        })
        .collect();
    ns_phase2.fetch_add(t_p2.elapsed().as_nanos() as u64, AtomicOrdering::Relaxed);

    let t_p3 = std::time::Instant::now();
    // ── Phase 3: sparse FORWARD DP ──
    //
    // Old formulation: dense backward-DP with `dp[i][cs]` a
    // `Vec<Vec<DpEntry>>` of size `(n+1) × 2^K`. At K=16 (bumped
    // for Gemma3's 12-tile QK-norm fusion) that's 65 536 states
    // per position — a 972-tile FUF meant iterating 64M cells
    // per solve × 20 workload points × 33 models = tens of
    // billions of iterations plus ~1.5 GB of Vec allocation per
    // solve. Measured: 174 seconds of Phase 3 alone for a single
    // gemma2-27b solve set. That's the compile-time pain.
    //
    // New formulation: forward DP with sparse state. Start from
    // `dp[0]={0: 0.0}` and propagate only REACHABLE states.
    // Each reachable cell records a back-pointer `(prev_cs,
    // choice)` so we can reconstruct the picked sequence. Most
    // positions have only a handful of reachable claim_states
    // (bounded by the product of candidate-count × claim-spread
    // up to that position), so memory + time collapse from
    // `2^K × n` to roughly `candidates × n`.
    //
    // Correctness: identical optimal-plan guarantee as backward
    // DP — we cover the same transition graph, just visit only
    // reachable nodes. Tie-breaking (min cost) is preserved.
    #[derive(Clone, Copy)]
    struct SparseCell {
        cost: f64,
        /// claim_state at position `i-1` that produced this cell
        /// (via the transition at position i-1).
        prev_cs: ClaimMask,
        /// Candidate chosen AT position `i-1` to arrive here.
        /// `None` means position i-1 was a pass-through (its
        /// tile was pre-claimed by an earlier multi-tile impl).
        choice: Option<(ImplId, ClaimMask)>,
        /// Total number of impl picks taken on the path to this cell
        /// (`choice = Some(_)` increments; pass-through does not).
        /// **Tiebreaker on equal cost (lower wins).** Required
        /// because alternatives at uncalibrated shapes (lm_head
        /// vocab-N) often tie to fp precision between
        /// `(fused N-tile)` and `(fused (N-1)-tile + 1 singleton)` —
        /// both cover the same tiles for the same total cost, but
        /// fewer picks = fewer kernel launches and matches the
        /// design intent at solver.rs `// Multi-tile claims represent
        /// fusion …`. Without this tiebreak, HashMap iteration
        /// order made the DP non-deterministic — STATUS Step 1b(a)
        /// 0-pick anomaly traced here.
        picks_count: u32,
    }

    let mut dp: Vec<HashMap<ClaimMask, SparseCell>> = vec![HashMap::new(); n + 1];
    dp[0].insert(
        0,
        SparseCell {
            cost: 0.0,
            prev_cs: 0,
            choice: None,
            picks_count: 0,
        },
    );

    let update = |cell_map: &mut HashMap<ClaimMask, SparseCell>,
                  key: ClaimMask,
                  cand: SparseCell| {
        let better = match cell_map.get(&key) {
            Some(ex) => {
                cand.cost < ex.cost || (cand.cost == ex.cost && cand.picks_count < ex.picks_count)
            }
            None => true,
        };
        if better {
            cell_map.insert(key, cand);
        }
    };

    for i in 0..n {
        // Snapshot `dp[i]` to avoid borrow conflicts while writing
        // `dp[i+1]` in the same iteration. `dp[i]` is small
        // (sparse), so cloning its (cost, picks_count) pairs is cheap.
        let at_i: Vec<(ClaimMask, f64, u32)> = dp[i]
            .iter()
            .map(|(k, c)| (*k, c.cost, c.picks_count))
            .collect();
        for (cs, cost, picks_count) in at_i {
            if cs & 1 != 0 {
                // Tile i is pre-claimed — pass through.
                let new_cs = cs >> 1;
                update(
                    &mut dp[i + 1],
                    new_cs,
                    SparseCell {
                        cost,
                        prev_cs: cs,
                        choice: None,
                        picks_count,
                    },
                );
            } else {
                for cand in &candidates[i] {
                    if cand.mask & cs != 0 {
                        continue; // conflict with pending claims
                    }
                    let new_cs = (cs | cand.mask) >> 1;
                    let new_cost = cost + cand.cost;
                    let new_picks_count = picks_count + 1;
                    update(
                        &mut dp[i + 1],
                        new_cs,
                        SparseCell {
                            cost: new_cost,
                            prev_cs: cs,
                            choice: Some((cand.imp_id, cand.mask)),
                            picks_count: new_picks_count,
                        },
                    );
                }
            }
        }
    }

    let Some(terminal) = dp[n].get(&0).copied() else {
        // ── Guard G3: explain the unclaimable tile ──
        // No feasible plan for this workload. The old code reported the
        // first tile with empty `candidates[t]`, which is a RED HERRING:
        // a fusion-only tile (lone Mul/Silu/Gelu) has empty candidates
        // but IS claimed by an earlier multi-tile fusion mask — so it's
        // not the real problem. Compute the set of tile positions
        // coverable by ANY candidate claim from ANY seed; a tile covered
        // by no claim is the genuinely-unclaimable op.
        let mut coverable = vec![false; n];
        for (seed, cands) in candidates.iter().enumerate() {
            for c in cands {
                let mut m = c.mask;
                while m != 0 {
                    let j = m.trailing_zeros() as usize;
                    if seed + j < n {
                        coverable[seed + j] = true;
                    }
                    m &= m - 1;
                }
            }
        }
        let orphan_ops: Vec<(TileId, OpKind)> = (0..n)
            .filter(|&t| !coverable[t])
            .map(|t| (fuf.nodes[t].id, fuf.nodes[t].op))
            .collect();

        // Report the first genuinely-unclaimable tile (topo order), or
        // tile 0 if every tile IS coverable but no terminal exists (K
        // too small / backward-reaching claim).
        let report = (0..n).find(|&t| !coverable[t]).unwrap_or(0);
        let node = &fuf.nodes[report];

        // FAILURE-PATH-ONLY: re-run every registered Impl against the
        // reported orphan and record why it declined. Skips `applies_to`
        // (no MatchContext threaded into solve_one) — target_compatible
        // / matches / workload_constraint cover the real cases.
        let mut declines: Vec<String> = Vec::new();
        for (id, imp) in lib.iter_enumerated() {
            let why = if !imp.target_compatible(target) {
                "target_compatible() = false".to_string()
            } else if imp.matches(fuf, node.id, target).is_none() {
                "matches() returned None".to_string()
            } else if !imp
                .workload_constraint()
                .accepts(num_tokens as u32, sk_bucket)
            {
                "workload_constraint rejected this num_tokens/sk_bucket".to_string()
            } else {
                // It DID match + accept — must be applies_to (not checked
                // here) or a cost/mask reject; surface it so the report
                // isn't silently empty.
                "matched but was filtered (applies_to / cost / claim-mask)".to_string()
            };
            declines.push(format!("    {} ({}): {}", imp.name(), id.0, why));
        }

        return Err(SolveError::UnclaimedTile {
            tile: node.id,
            op: node.op,
            point,
            orphan_ops,
            declines,
        });
    };
    ns_phase3.fetch_add(t_p3.elapsed().as_nanos() as u64, AtomicOrdering::Relaxed);

    let t_p4 = std::time::Instant::now();
    // ── Phase 4: reconstruct from forward DP back-pointers. ──
    //
    // Walk dp[n] → dp[n-1] → … → dp[0], recovering the choice
    // made at each position. `terminal` is the cell at dp[n][0];
    // `terminal.prev_cs` is the claim-state at dp[n-1] that led
    // here, and `terminal.choice` is the candidate picked at
    // position n-1 (or None for pass-through).
    let mut choices_at: Vec<Option<(ImplId, ClaimMask)>> = vec![None; n];
    let mut cell = terminal;
    for i in (0..n).rev() {
        choices_at[i] = cell.choice;
        if i > 0 {
            cell = dp[i][&cell.prev_cs];
        }
    }

    // Walk forward, allocating subgraph ids in visitation order.
    let mut assignment = Assignment::default();
    let mut next_sg: u32 = 0;
    let mut total = 0.0_f64;

    for i in 0..n {
        let (imp_id, mask) = match choices_at[i] {
            Some(c) => c,
            None => continue, // pass-through
        };

        let sg = SubgraphId(next_sg);
        next_sg += 1;

        let cand = candidates[i]
            .iter()
            .find(|c| c.imp_id == imp_id && c.mask == mask)
            .expect("DP stored a candidate that exists in the candidate list");

        // ── Guard G4: solve-time precondition for the PICKED impl ──
        // Recover the MatchInfo for this pick from matches_at[i] (still
        // in scope) by imp_id, then run the impl's validate_pick against
        // the model bounds. Default Ok(()) → no effect on impls that
        // don't opt in. Mirrors a lowering assert so a violation is a
        // compile-time macro error, not a worker-thread runtime panic.
        let info = &matches_at[i]
            .iter()
            .find(|(id, _info, _cost)| *id == imp_id)
            .expect("picked imp_id must exist in matches_at")
            .1;
        if let Err(why) = lib.get(imp_id).validate_pick(info, bounds) {
            return Err(SolveError::PickPreconditionFailed {
                impl_name: lib.get(imp_id).name(),
                tile: fuf.nodes[i].id,
                op: fuf.nodes[i].op,
                point,
                why,
            });
        }

        for j in 0..K {
            if mask & (1 << j) != 0 {
                let tile_idx = i + j;
                if tile_idx < n {
                    assignment.cover.insert(fuf.nodes[tile_idx].id, sg);
                }
            }
        }
        assignment.impls.insert(sg, imp_id);
        total += cand.cost;
    }

    // Invariant: every tile is claimed.
    debug_assert!(
        assignment.cover.len() == n,
        "DP left {} tiles uncovered",
        n - assignment.cover.len(),
    );

    assignment.predicted_us = total;
    ns_phase4.fetch_add(t_p4.elapsed().as_nanos() as u64, AtomicOrdering::Relaxed);
    Ok(assignment)
}

/// One candidate impl choice at a given tile position.
#[derive(Debug, Clone, Copy)]
struct Candidate {
    imp_id: ImplId,
    /// Bit `j` set ⇒ position `seed + j` is claimed. Bit 0 (seed)
    /// is always set by construction.
    mask: u16,
    cost: f64,
}

/// Resolve each tile input's shape for the cost function.
///
/// - `Tile` inputs read the producing tile's output slot.
/// - `Weight` inputs read from [`Inferred::weights`].
/// - `Extern` inputs read from [`extern_shape`].
fn resolve_input_shapes(fuf: &Fuf, node: &FufNode, inferred: &Inferred) -> Vec<Shape> {
    node.inputs
        .iter()
        .map(|inp| match inp {
            FufInput::Tile { id, slot } => {
                let upstream = fuf.get(*id);
                upstream
                    .outputs
                    .get(*slot as usize)
                    .cloned()
                    .unwrap_or_default()
            }
            FufInput::Weight { id, .. } => inferred.weights.get(id).cloned().unwrap_or_default(),
            FufInput::Extern { kind, .. } => extern_shape(*kind),
            FufInput::Scalar(_) => Shape::new(),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg::build_cfg;
    use crate::classify::classify;
    use crate::config::{self, ModelParams};
    use crate::fuf::unroll;
    use crate::impl_lib::{
        CostCtx, Handoff, Implementation, ImplementationLibrary, LaunchKind, Layout, MatchInfo,
        Resources, WorkloadConstraint, starter_library,
    };
    use crate::parse::parse_block;
    use crate::shape::infer;
    use crate::target::from_profile_def;
    use std::path::PathBuf;

    fn llama_params(stem: &str) -> ModelParams {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../models/arch")
            .join("configs")
            .join("llama")
            .join(format!("{stem}.json"));
        config::load_file(&path).unwrap()
    }

    fn l4_target() -> TargetProfile {
        from_profile_def(&scratchy_target_cuda::targets::L4_SM89)
    }

    fn build(src: &str, params: &ModelParams) -> (Fuf, Inferred) {
        let file: syn::File = syn::parse_str(&format!("fn _c() {{ {src} }}")).expect("parse");
        let block = match &file.items[0] {
            syn::Item::Fn(f) => &*f.block,
            _ => unreachable!(),
        };
        let ast = parse_block(block).unwrap();
        let program = classify(&ast).unwrap();
        let inferred = infer(
            &program,
            &crate::weights_manifest::WeightsManifest::llama_test_conventions(),
            &std::collections::BTreeMap::new(),
        )
        .unwrap();
        let cfg = build_cfg(&program, params).unwrap();
        let fuf = unroll(&cfg, &inferred).unwrap();
        (fuf, inferred)
    }

    /// Realistic Llama body with the full SwiGLU MLP (gate+silu+up+mul
    /// → down). The SwiGLU quadruple is the exemplar multi-tile claim
    /// exercised by `FusedGateUpSiluMulImpl`; if you strip silu/mul
    /// from this body the solver falls back to all-singleton coverage
    /// and the fusion tests below go dead.
    const LLAMA_BODY: &str = r#"
        hidden_states = embed(input_ids, embed_tokens);
        for layer in 0..num_hidden_layers {
            normed = rmsnorm(hidden_states, input_layernorm[layer]);
            q = gemm(normed, self_attn.q_proj[layer]);
            k = gemm(normed, self_attn.k_proj[layer]);
            v = gemm(normed, self_attn.v_proj[layer]);
            (q, k, v) = rope_append(q, k, v, positions, rotary, kv_cache[layer]);
            attn = attention(q, k, v, kv_cache[layer], block_table);
            oproj = gemm(attn, self_attn.o_proj[layer]);
            hidden_states = add(oproj, hidden_states);

            normed2 = rmsnorm(hidden_states, post_attention_layernorm[layer]);
            gate = silu(gemm(normed2, mlp.gate_proj[layer]));
            up = gemm(normed2, mlp.up_proj[layer]);
            down = gemm(gate * up, mlp.down_proj[layer]);
            hidden_states = add(down, hidden_states);
        }
        normed = rmsnorm(hidden_states, norm);
        logits = gemm(normed, lm_head);
    "#;

    const WORKLOAD_POINTS: [u64; 5] = [1, 8, 64, 512, 4096];

    #[test]
    fn dp_assigns_every_tile_at_every_workload_point() {
        let params = llama_params("llama-3.1-8b");
        let (fuf, inferred) = build(LLAMA_BODY, &params);
        let lib = starter_library();
        let target = l4_target();
        let bounds = params.bounds.clone();

        let t0 = std::time::Instant::now();
        let workloads = solve(
            &fuf,
            &lib,
            &target,
            &inferred,
            &bounds,
            &WORKLOAD_POINTS,
            &[],
        )
        .expect("solve");
        let elapsed_ms = t0.elapsed().as_millis();

        assert_eq!(workloads.per_workload.len(), WORKLOAD_POINTS.len());
        for &m in &WORKLOAD_POINTS {
            let sfuf = workloads
                .get_nt(m)
                .unwrap_or_else(|| panic!("no sfuf for m={m}"));
            assert!(
                sfuf.is_cover_complete(fuf.len()),
                "cover not complete at m={m}"
            );
            // Fusion savings per layer:
            //   SwiGLU (gate, up, silu, mul) → 1 subgraph (saves 3)
            //   QKV + rope (q, k, v gemms + rope_append) → 1 (saves 3)
            //   attn-residual Add + post_attn_layernorm → 1 (saves 1)
            //   MLP-residual Add + next-layer input_layernorm (or
            //     final norm, on the last layer) → 1 (saves 1)
            // Total: 8 fewer subgraphs per layer — these per-layer
            // fusions are structural and must always land.
            //
            // Plus an optional fleet-wide save at the lm_head boundary
            // when CutlassFusedAddRmsNormGemm wins on cost: the
            // (final-residual Add, final RmsNorm, lm_head Gemm) triple
            // collapses into one subgraph instead of (Add+RmsNorm
            // fused) + (lm_head Gemm) = 2. The DP solver picks this
            // 3-tile fusion only at workload points where the CUTLASS
            // tile zoo's calibrated cost beats the cuBLAS singleton at
            // (M, vocab_size, hidden_size); at uncalibrated or
            // unfavorable M it falls back to the 2-tile (Add+RmsNorm)
            // pair plus a singleton lm_head Gemm.
            let nl = params.bounds["num_hidden_layers"] as usize;
            let actual_fewer = fuf.len() - sfuf.num_subgraphs();
            let min_fewer = 8 * nl; // structural per-layer fusions
            let max_fewer = 8 * nl + 1; // + optional lm_head triple
            assert!(
                actual_fewer == min_fewer || actual_fewer == max_fewer,
                "at m={m}: expected fuf.len()-num_subgraphs() in \
                 {{{min_fewer}, {max_fewer}}} (per-layer fusions, \
                 plus optional lm_head NormGemm fusion), got {actual_fewer}",
            );
            assert!(
                sfuf.predicted_us > 0.0 && sfuf.predicted_us.is_finite(),
                "predicted_us at m={m} = {}",
                sfuf.predicted_us,
            );
        }
        // Budget chosen for debug-mode test runs. The cutlass tile zoo
        // (~17 variants) multiplies candidate evaluation per Gemm
        // tile, but the polynomial DP still solves Llama-3.1-8B's
        // 483-tile graph across all 5 workload points well under
        // a second in release — debug mode adds a ~100× constant.
        // If this trips, something non-linear slipped into the
        // candidate / DP cost scan.
        assert!(
            elapsed_ms < 30_000,
            "solve took {elapsed_ms} ms, budget 30000 (debug mode)"
        );
    }

    #[test]
    fn swiglu_mlp_claimed_as_fused_subgraph_per_layer() {
        // For every unrolled MLP in a real Llama body, the (gate_gemm,
        // up_gemm, silu, mul) quadruple must collapse into one
        // FusedGateUpSiluMulImpl subgraph. No phase tags, no weight
        // names — this is the structural-matcher acid test.
        let params = llama_params("llama-3.1-8b");
        let (fuf, inferred) = build(LLAMA_BODY, &params);
        let lib = starter_library();
        let target = l4_target();

        let workloads = solve(
            &fuf,
            &lib,
            &target,
            &inferred,
            &params.bounds,
            &[1, 512, 4096],
            &[],
        )
        .unwrap();

        let nl = params.bounds["num_hidden_layers"] as usize;

        for (wp, sfuf) in workloads.per_workload.iter() {
            let m = wp.num_tokens;
            // Count subgraphs that claim 4 tiles of kinds {Gemm,
            // Gemm, Silu, Mul} with shared activation on the Gemms.
            let mut fused_count = 0;
            for sg in sfuf.subgraphs() {
                let tiles = sfuf.tiles_in_subgraph(sg);
                if tiles.len() != 4 {
                    continue;
                }
                let ops: Vec<OpKind> = tiles.iter().map(|t| fuf.get(*t).op).collect();
                let n_gemm = ops.iter().filter(|o| **o == OpKind::Gemm).count();
                let n_silu = ops.iter().filter(|o| **o == OpKind::Silu).count();
                let n_mul = ops.iter().filter(|o| **o == OpKind::Mul).count();
                if n_gemm == 2 && n_silu == 1 && n_mul == 1 {
                    fused_count += 1;
                    // And the bound Impl must be the fused one —
                    // lookup by name to avoid coupling to ImplId ordering.
                    let imp_id = sfuf.impl_of(sg).unwrap();
                    assert_eq!(
                        lib.get(imp_id).name(),
                        "fused_gate_up_silu_mul",
                        "subgraph at m={m} has fused topology but wrong impl",
                    );
                }
            }
            assert_eq!(
                fused_count, nl,
                "expected one fused SwiGLU subgraph per layer at m={m}",
            );
        }
    }

    #[test]
    fn qkv_rope_claimed_as_fused_subgraph_per_layer() {
        // Every attention block's (q_gemm, k_gemm, v_gemm, rope_append)
        // quadruple must collapse into one FusedQkvRopeCacheImpl
        // subgraph. Structural — three Gemms share an activation and
        // feed the RopeAppend's first three Tile slots.
        let params = llama_params("llama-3.1-8b");
        let (fuf, inferred) = build(LLAMA_BODY, &params);
        let lib = starter_library();
        let target = l4_target();

        let workloads = solve(
            &fuf,
            &lib,
            &target,
            &inferred,
            &params.bounds,
            &[1, 512, 4096],
            &[],
        )
        .unwrap();

        let nl = params.bounds["num_hidden_layers"] as usize;

        for (wp, sfuf) in workloads.per_workload.iter() {
            let m = wp.num_tokens;
            let mut fused_count = 0;
            for sg in sfuf.subgraphs() {
                let tiles = sfuf.tiles_in_subgraph(sg);
                if tiles.len() != 4 {
                    continue;
                }
                let ops: Vec<OpKind> = tiles.iter().map(|t| fuf.get(*t).op).collect();
                let n_gemm = ops.iter().filter(|o| **o == OpKind::Gemm).count();
                let n_rope = ops.iter().filter(|o| **o == OpKind::RopeAppend).count();
                if n_gemm == 3 && n_rope == 1 {
                    fused_count += 1;
                    let imp_id = sfuf.impl_of(sg).unwrap();
                    let name = lib.get(imp_id).name();
                    // WorkloadConstraint-driven dispatch: decode (M=1)
                    // picks a cache-fused variant, prefill (M>=2) picks
                    // a split variant that keeps K/V contiguous for the
                    // prefill attention path. Both the cuBLAS-backed
                    // peer (`fused_qkv_rope_{cache,prefill}`) and the
                    // CUTLASS-backed peers (`cutlass_*` — name reflects
                    // the underlying GEMM tile shape) are valid; the
                    // DP solver picks whichever is cheaper at the
                    // calibrated workload point. Either is structurally
                    // a fused-QKV-rope subgraph; what we forbid is a
                    // stray Gemm singleton claiming part of the quad.
                    let cublas_expected = if m == 1 {
                        "fused_qkv_rope_cache"
                    } else {
                        "fused_qkv_rope_prefill"
                    };
                    let is_cutlass_peer = name.starts_with("cutlass_");
                    assert!(
                        name == cublas_expected || is_cutlass_peer,
                        "subgraph at m={m} has QKV-rope topology but \
                         wrong impl: got {name}, expected {cublas_expected} \
                         or any cutlass_* peer",
                    );
                }
            }
            assert_eq!(
                fused_count, nl,
                "expected one fused QKV+rope subgraph per layer at m={m}",
            );
        }
    }

    #[test]
    fn fused_qkv_accessor_covers_three_source_weights() {
        // FusedQkvRopeCacheImpl declares one packed LinearLayer
        // accessor whose source_weights lists q_proj, k_proj, v_proj
        // for the matched layer.
        let params = llama_params("llama-3.2-1b");
        let (fuf, inferred) = build(LLAMA_BODY, &params);
        let lib = starter_library();
        let target = l4_target();
        let workloads = solve(&fuf, &lib, &target, &inferred, &params.bounds, &[1], &[]).unwrap();
        let sfuf = workloads.get_nt(1).unwrap();
        // First QKV-fused subgraph.
        let qkv_sg = sfuf
            .subgraphs()
            .find(|sg| {
                let tiles = sfuf.tiles_in_subgraph(*sg);
                tiles.len() == 4 && tiles.iter().any(|t| fuf.get(*t).op == OpKind::RopeAppend)
            })
            .expect("at least one fused QKV subgraph");
        let claim = sfuf.tiles_in_subgraph(qkv_sg);
        let imp = lib.get(sfuf.impl_of(qkv_sg).unwrap());
        let decls = imp.required_weights(&claim, &fuf, &classify_program(LLAMA_BODY));
        assert_eq!(decls.len(), 1, "one fused accessor per claim");
        assert_eq!(
            decls[0].source_weights.len(),
            3,
            "fused QKV accessor covers q_proj, k_proj, v_proj (3 sources)"
        );
    }

    #[test]
    fn add_rmsnorm_pairs_claimed_as_fused_subgraph() {
        // Every Add in a realistic Llama body has an immediate
        // RmsNorm consumer. FusedAddRmsNormImpl claims each pair as
        // one 2-tile subgraph. The first layer's input_layernorm is
        // the exception — its upstream is `embed`, not an Add — so
        // it stays a singleton RmsNormRefImpl claim.
        let params = llama_params("llama-3.1-8b");
        let (fuf, inferred) = build(LLAMA_BODY, &params);
        let lib = starter_library();
        let target = l4_target();

        let workloads = solve(
            &fuf,
            &lib,
            &target,
            &inferred,
            &params.bounds,
            &[1, 512, 4096],
            &[],
        )
        .unwrap();

        let nl = params.bounds["num_hidden_layers"] as usize;

        for (wp, sfuf) in workloads.per_workload.iter() {
            let m = wp.num_tokens;
            let mut fused_add_rms_norm_count = 0;
            let mut singleton_rmsnorm_count = 0;
            let mut singleton_add_count = 0;
            let mut gemm_add_count = 0;
            // 3-tile (Add, RmsNorm, Gemm) — `CutlassFusedAddRmsNormGemm`.
            // Lands at lm_head where the (Add, RmsNorm) pair feeds a
            // single-consumer Gemm. Counts as both an Add+RmsNorm
            // absorption AND a Gemm absorption (one less downstream
            // singleton-RmsNorm pinned by a CutlassGemmAdd).
            let mut fused_3tile_arn_g_count = 0;
            for sg in sfuf.subgraphs() {
                let tiles = sfuf.tiles_in_subgraph(sg);
                let ops: Vec<OpKind> = tiles.iter().map(|t| fuf.get(*t).op).collect();
                let imp_id = sfuf.impl_of(sg).unwrap();
                let name = lib.get(imp_id).name();
                if tiles.len() == 2 && ops.contains(&OpKind::Add) && ops.contains(&OpKind::RmsNorm)
                {
                    fused_add_rms_norm_count += 1;
                    assert_eq!(
                        name, "fused_add_rms_norm",
                        "subgraph at m={m} has Add+RmsNorm topology but wrong impl",
                    );
                } else if tiles.len() == 1 && ops[0] == OpKind::RmsNorm {
                    singleton_rmsnorm_count += 1;
                    assert_eq!(
                        name, "rmsnorm_ref",
                        "singleton RmsNorm at m={m} bound to wrong impl",
                    );
                } else if tiles.len() == 1 && ops[0] == OpKind::Add {
                    singleton_add_count += 1;
                } else if tiles.len() == 2
                    && ops.contains(&OpKind::Gemm)
                    && ops.contains(&OpKind::Add)
                {
                    gemm_add_count += 1;
                } else if tiles.len() == 3
                    && ops.contains(&OpKind::Add)
                    && ops.contains(&OpKind::RmsNorm)
                    && ops.contains(&OpKind::Gemm)
                {
                    fused_3tile_arn_g_count += 1;
                }
            }
            // Every Add fuses into something — `fused_add_rms_norm`
            // (always wins at decode m=1, where the upstream is a
            // GEMV) or `cutlass_gemm_add` (wins at prefill m≥512
            // via the beta=1.0 epilogue that amortizes the aux-read)
            // or `CutlassFusedAddRmsNormGemm` 3-tile (lm_head).
            // No Add stays unfused.
            assert_eq!(
                singleton_add_count, 0,
                "expected zero unfused Add subgraphs at m={m}",
            );
            // 2 Adds per layer, each absorbed by one of the three
            // fusion families. Sum equals 2*NL regardless of which
            // fusion the solver chose.
            assert_eq!(
                fused_add_rms_norm_count + gemm_add_count + fused_3tile_arn_g_count,
                2 * nl,
                "expected fused_add_rms_norm + gemm_add + 3tile-arng to equal 2*NL at m={m}",
            );
            // Each `cutlass_gemm_add` fusion strands the downstream
            // RmsNorm as a singleton. Plus the first layer's
            // input_layernorm always escapes (upstream is Embed, not
            // Add). The 3-tile (Add, RmsNorm, Gemm) absorbs both the
            // RmsNorm and the Gemm so it does NOT contribute a
            // singleton RmsNorm. So singleton RmsNorm count =
            // 1 + gemm_add_count (independent of 3-tile count).
            assert_eq!(
                singleton_rmsnorm_count,
                1 + gemm_add_count,
                "expected 1 + gemm_add_count singleton RmsNorm at m={m}",
            );
        }
    }

    #[test]
    fn fused_weight_accessor_collides_to_single_declaration_across_buckets() {
        // A fused impl declares one accessor per claim. Across many
        // buckets that pick the same Impl, WeightBundle-trait
        // aggregation must dedupe — otherwise user gets N copies of
        // the same accessor. Regression-guards the SFUF-walking
        // emission in codegen::emit_weight_bundle_trait.
        use crate::impl_lib::default_required_weights;
        let params = llama_params("llama-3.2-1b");
        let (fuf, inferred) = build(LLAMA_BODY, &params);
        let lib = starter_library();
        let target = l4_target();
        let workloads = solve(
            &fuf,
            &lib,
            &target,
            &inferred,
            &params.bounds,
            &[1, 512, 4096],
            &[],
        )
        .unwrap();

        // Pick a SwiGLU-fused subgraph in the first bucket, record
        // its declared accessor name, then verify the same name
        // shows up exactly once in every other bucket's SFUF too.
        let first_sfuf = workloads.per_workload.values().next().unwrap();
        let first_sg = first_sfuf
            .subgraphs()
            .find(|sg| {
                let tiles = first_sfuf.tiles_in_subgraph(*sg);
                tiles.len() == 4
                    && tiles.iter().any(|t| fuf.get(*t).op == OpKind::Silu)
                    && tiles.iter().any(|t| fuf.get(*t).op == OpKind::Mul)
            })
            .expect("at least one fused SwiGLU claim exists");
        let first_claim = first_sfuf.tiles_in_subgraph(first_sg);
        let first_imp = lib.get(first_sfuf.impl_of(first_sg).unwrap());
        let decls = first_imp.required_weights(&first_claim, &fuf, &classify_program(LLAMA_BODY));
        assert_eq!(
            decls.len(),
            1,
            "fused impl declares exactly one accessor per claim"
        );
        assert_eq!(
            decls[0].source_weights.len(),
            2,
            "fused accessor covers two source weights (gate_proj + up_proj)"
        );
        // And the default still works for Embed (singleton).
        let embed_sg = first_sfuf
            .subgraphs()
            .find(|sg| {
                let tiles = first_sfuf.tiles_in_subgraph(*sg);
                tiles.len() == 1 && fuf.get(tiles[0]).op == OpKind::Embed
            })
            .unwrap();
        let embed_claim = first_sfuf.tiles_in_subgraph(embed_sg);
        let default_decls =
            default_required_weights(&embed_claim, &fuf, &classify_program(LLAMA_BODY));
        assert_eq!(default_decls.len(), 1);
        assert_eq!(default_decls[0].source_weights.len(), 1);
    }

    fn classify_program(src: &str) -> crate::classified::Program {
        let file: syn::File =
            syn::parse_str(&format!("fn _c() {{ {src} }}")).expect("parse carrier");
        let block = match &file.items[0] {
            syn::Item::Fn(f) => &*f.block,
            _ => unreachable!(),
        };
        let ast = crate::parse::parse_block(block).unwrap();
        crate::classify::classify(&ast).unwrap()
    }

    /// Prefill (m=4096) predicted cost dwarfs decode (m=1) by at
    /// least 10× — order-of-magnitude smoke test that catches
    /// regressions where gemm cost silently collapses (the prior
    /// greedy bug) or the cost model otherwise loses prefill's
    /// FLOP count. Used to live in `phase7_end_to_end.rs` reading
    /// emitted per-bucket `m_<N>::PREDICTED_US` constants — moved
    /// here so the assertion runs against solver output directly,
    /// without baking ~9k lines of workspace-wide stub modules
    /// just to expose one f64 per bucket.
    #[test]
    fn prefill_cost_dwarfs_decode_cost() {
        let params = llama_params("llama-3.2-1b");
        let (fuf, inferred) = build(LLAMA_BODY, &params);
        let lib = starter_library();
        let target = l4_target();
        let workloads = solve(
            &fuf,
            &lib,
            &target,
            &inferred,
            &params.bounds,
            &[1, 4096],
            &[],
        )
        .unwrap();

        let decode = workloads
            .per_workload
            .iter()
            .find(|(wp, _)| wp.num_tokens == 1)
            .map(|(_, sfuf)| sfuf.predicted_us)
            .expect("solver returned m=1 workload");
        let prefill = workloads
            .per_workload
            .iter()
            .find(|(wp, _)| wp.num_tokens == 4096)
            .map(|(_, sfuf)| sfuf.predicted_us)
            .expect("solver returned m=4096 workload");
        assert!(
            decode > 0.0 && decode.is_finite(),
            "bogus decode predicted_us: {decode}"
        );
        assert!(
            prefill > 0.0 && prefill.is_finite(),
            "bogus prefill predicted_us: {prefill}"
        );
        assert!(
            prefill > decode * 10.0,
            "prefill ({prefill} µs) should dwarf decode ({decode} µs) by 10×",
        );
    }

    #[test]
    fn gemm_cost_is_actually_counted() {
        // Regression: the prior greedy's gemm cost silently dropped
        // to zero because weight shapes weren't threaded. If this
        // regresses, prefill at M=4096 will collapse toward decode.
        let params = llama_params("llama-3.1-8b");
        let (fuf, inferred) = build(LLAMA_BODY, &params);
        let lib = starter_library();
        let target = l4_target();
        let bounds = params.bounds.clone();

        let workloads = solve(&fuf, &lib, &target, &inferred, &bounds, &[1, 4096], &[]).unwrap();
        let decode_us = workloads.get_nt(1).unwrap().predicted_us;
        let prefill_us = workloads.get_nt(4096).unwrap().predicted_us;
        assert!(
            prefill_us > decode_us * 10.0,
            "prefill ({prefill_us}) should dwarf decode ({decode_us})",
        );
    }

    #[test]
    fn missing_impl_is_unclaimed_tile_error() {
        // Empty library — no Impl can match anything. First tile
        // should fail with UnclaimedTile, NOT be auto-claimed by
        // a fallback (the old scratchy DP had such a fallback for
        // ResidualAdd; we explicitly did not port it).
        let params = llama_params("llama-3.2-1b");
        let (fuf, inferred) = build("hidden_states = embed(input_ids, embed_tokens);", &params);
        let lib = ImplementationLibrary::new();
        let target = l4_target();
        let bounds = params.bounds.clone();

        let err = solve(&fuf, &lib, &target, &inferred, &bounds, &[1], &[]).unwrap_err();
        assert!(
            matches!(
                err,
                SolveError::UnclaimedTile {
                    op: OpKind::Embed,
                    ..
                }
            ),
            "expected UnclaimedTile on embed, got {err:?}",
        );
    }

    // Minimal trait impl used by tests that exercise solver
    // semantics at the trait boundary. Configurable cost + workload
    // constraint + optional multi-tile matcher.
    #[derive(Debug)]
    struct TestImpl {
        name: &'static str,
        op: OpKind,
        cost: f64,
        workload: WorkloadConstraint,
        multi_tile: Option<fn(&Fuf, TileId) -> Option<MatchInfo>>,
    }
    impl Implementation for TestImpl {
        fn name(&self) -> &'static str {
            self.name
        }
        fn target_compatible(&self, _p: &TargetProfile) -> bool {
            true
        }
        fn workload_constraint(&self) -> WorkloadConstraint {
            self.workload
        }
        fn matches(&self, fuf: &Fuf, seed: TileId, _p: &TargetProfile) -> Option<MatchInfo> {
            if let Some(f) = self.multi_tile {
                f(fuf, seed)
            } else if fuf.get(seed).op == self.op {
                Some(MatchInfo {
                    claimed_tiles: vec![seed],
                    boundary_inputs: vec![],
                    boundary_outputs: vec![seed],
                })
            } else {
                None
            }
        }
        fn cost_us(&self, _m: &MatchInfo, _ctx: &CostCtx) -> f64 {
            self.cost
        }
        fn resources(&self, _m: &MatchInfo) -> Resources {
            Resources::ZERO
        }
        fn launch_kind(&self) -> LaunchKind {
            LaunchKind::HostCallback
        }
        fn supported_input_handoffs(&self) -> &[Handoff] {
            &[Handoff::StreamOrder]
        }
        fn supported_output_handoffs(&self) -> &[Handoff] {
            &[Handoff::StreamOrder]
        }
        fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
            vec![Layout::Any; m.boundary_inputs.len()]
        }
        fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
            vec![Layout::Any; m.boundary_outputs.len()]
        }
    }

    #[test]
    fn infinite_cost_is_fatal() {
        // The solver's cost aggregator treats non-finite costs as
        // UnreachableCost. A bug in an Impl's cost_us (returning
        // INFINITY or NaN) surfaces loudly rather than silently
        // propagating.
        let params = llama_params("llama-3.2-1b");
        let (fuf, inferred) = build("hidden_states = embed(input_ids, embed_tokens);", &params);
        let target = l4_target();

        let mut lib = ImplementationLibrary::new();
        lib.push(Box::new(TestImpl {
            name: "embed_buggy",
            op: OpKind::Embed,
            cost: f64::INFINITY,
            workload: WorkloadConstraint::Any,
            multi_tile: None,
        }));

        let err = solve(&fuf, &lib, &target, &inferred, &params.bounds, &[1], &[]).unwrap_err();
        assert!(
            matches!(err, SolveError::UnreachableCost { .. }),
            "expected UnreachableCost, got {err:?}",
        );
    }

    #[test]
    fn workload_constraint_excludes_impls_outside_range() {
        let params = llama_params("llama-3.2-1b");
        let (fuf, inferred) = build("hidden_states = embed(input_ids, embed_tokens);", &params);
        let target = l4_target();

        let mut lib = ImplementationLibrary::new();
        lib.push(Box::new(TestImpl {
            name: "embed_decode_only",
            op: OpKind::Embed,
            cost: 1.0,
            workload: WorkloadConstraint::NumTokensRange { min: 1, max: 8 },
            multi_tile: None,
        }));

        // At M=1 it's fine.
        let ok = solve(&fuf, &lib, &target, &inferred, &params.bounds, &[1], &[]).unwrap();
        assert!(ok.get_nt(1).unwrap().is_cover_complete(fuf.len()));

        // At M=4096 it's excluded; no other impl; UnclaimedTile.
        let err = solve(&fuf, &lib, &target, &inferred, &params.bounds, &[4096], &[]).unwrap_err();
        assert!(
            matches!(
                err,
                SolveError::UnclaimedTile {
                    op: OpKind::Embed,
                    ..
                }
            ),
            "expected UnclaimedTile, got {err:?}",
        );
    }

    #[test]
    fn multi_tile_matcher_collapses_to_one_subgraph() {
        fn matches_double_add(fuf: &Fuf, seed: TileId) -> Option<MatchInfo> {
            if fuf.get(seed).op != OpKind::Add {
                return None;
            }
            let after = (seed.0 as usize + 1..fuf.len())
                .map(|i| TileId(i as u32))
                .find(|t| fuf.get(*t).op == OpKind::Add)?;
            Some(MatchInfo {
                claimed_tiles: vec![seed, after],
                boundary_inputs: vec![],
                boundary_outputs: vec![seed, after],
            })
        }

        let params = llama_params("llama-3.2-1b");
        let (fuf, inferred) = build(
            r#"
            hidden_states = embed(input_ids, embed_tokens);
            for layer in 0..2 {
                hidden_states = add(hidden_states, hidden_states);
                hidden_states = add(hidden_states, hidden_states);
            }
            "#,
            &params,
        );
        let target = l4_target();

        let mut lib = starter_library();
        lib.push(Box::new(TestImpl {
            name: "double_add_ref",
            op: OpKind::Add,
            cost: 0.001, // cheaper than two single adds
            workload: WorkloadConstraint::Any,
            multi_tile: Some(matches_double_add),
        }));

        let workloads = solve(&fuf, &lib, &target, &inferred, &params.bounds, &[1], &[]).unwrap();
        let sfuf = workloads.get_nt(1).unwrap();

        // 1 embed + 4 adds; fused double-add claims pairs.
        // Expected: 1 embed + 2 fused = 3 subgraphs.
        assert_eq!(sfuf.num_subgraphs(), 3, "adds should pair into subgraphs");
        assert!(sfuf.is_cover_complete(fuf.len()));

        // Find the subgraphs that cover Add tiles (structural —
        // ask the FUF what op each subgraph's tiles have).
        let add_subgraphs: Vec<_> = sfuf
            .subgraphs()
            .filter(|sg| {
                sfuf.tiles_in_subgraph(*sg)
                    .iter()
                    .all(|t| fuf.get(*t).op == OpKind::Add)
            })
            .collect();
        assert_eq!(add_subgraphs.len(), 2);
        for sg in add_subgraphs {
            assert_eq!(sfuf.tiles_in_subgraph(sg).len(), 2);
        }
    }

    /// Gemma-style body: GELU MLP (`down(gelu(gate) * up)`),
    /// sliding attention on odd layers, final logit softcap.
    /// Exercises [`FusedGateUpGeluMulImpl`], both
    /// [`SlidingAttentionViaCacheImpl`] variants (decode + prefill),
    /// and [`TanhSoftCapImpl`] through the real solver.
    const GEMMA_LIKE_BODY: &str = r#"
        hidden_states = embed(input_ids, embed_tokens);
        for layer in 0..num_hidden_layers {
            normed = rmsnorm(hidden_states, input_layernorm[layer]);
            q = gemm(normed, self_attn.q_proj[layer]);
            k = gemm(normed, self_attn.k_proj[layer]);
            v = gemm(normed, self_attn.v_proj[layer]);
            (q, k, v) = rope_append(q, k, v, positions, rotary, kv_cache[layer]);
            if layer % 2 == 0 {
                attn = attention(q, k, v, kv_cache[layer], block_table);
            } else {
                attn = sliding_attention(q, k, v, kv_cache[layer], block_table);
            }
            oproj = gemm(attn, self_attn.o_proj[layer]);
            hidden_states = add(oproj, hidden_states);

            normed2 = rmsnorm(hidden_states, post_attention_layernorm[layer]);
            gate = gelu(gemm(normed2, mlp.gate_proj[layer]));
            up = gemm(normed2, mlp.up_proj[layer]);
            down = gemm(gate * up, mlp.down_proj[layer]);
            hidden_states = add(down, hidden_states);
        }
        normed = rmsnorm(hidden_states, norm);
        logits = gemm(normed, lm_head);
        capped = tanh_softcap(logits);
    "#;

    /// Llama-3.2-1B's numeric bounds + the Gemma-convention fields a
    /// body using `sliding_attention` / `tanh_softcap` requires. No
    /// real Gemma2 config lives in the per-arch crate's `configs/` at this
    /// point; this synthetic `ModelParams` lets the tests exercise
    /// the new Impls on real unrolled FUF sizes without committing
    /// the full arch.
    fn gemma_like_params() -> ModelParams {
        let mut p = llama_params("llama-3.2-1b");
        p.bounds.insert("sliding_window".into(), 4096);
        p.scalars.insert("query_pre_attn_scalar".into(), 256.0);
        p.scalars.insert("attn_logit_softcapping".into(), 50.0);
        p.scalars.insert("final_logit_softcapping".into(), 30.0);
        p
    }

    #[test]
    fn gelu_mlp_fusion_claims_four_tiles_per_layer() {
        // Structural: `(Gemm, Gemm, Gelu, Mul)` collapses into one
        // FusedGateUpGeluMulImpl subgraph per layer — same shape as
        // SwiGLU, different activation.
        let params = gemma_like_params();
        let (fuf, inferred) = build(GEMMA_LIKE_BODY, &params);
        let lib = starter_library();
        let target = l4_target();

        let workloads = solve(
            &fuf,
            &lib,
            &target,
            &inferred,
            &params.bounds,
            &[1, 512],
            &[],
        )
        .unwrap();
        let nl = params.bounds["num_hidden_layers"] as usize;

        for (wp, sfuf) in workloads.per_workload.iter() {
            let m = wp.num_tokens;
            let mut fused = 0;
            for sg in sfuf.subgraphs() {
                let tiles = sfuf.tiles_in_subgraph(sg);
                if tiles.len() != 4 {
                    continue;
                }
                let ops: Vec<OpKind> = tiles.iter().map(|t| fuf.get(*t).op).collect();
                let n_gemm = ops.iter().filter(|o| **o == OpKind::Gemm).count();
                let n_gelu = ops.iter().filter(|o| **o == OpKind::Gelu).count();
                let n_mul = ops.iter().filter(|o| **o == OpKind::Mul).count();
                if n_gemm == 2 && n_gelu == 1 && n_mul == 1 {
                    fused += 1;
                    let imp = lib.get(sfuf.impl_of(sg).unwrap()).name();
                    assert_eq!(imp, "fused_gate_up_gelu_mul", "m={m}");
                }
            }
            assert_eq!(fused, nl, "one Gelu-fused MLP per layer at m={m}");
        }
    }

    #[test]
    fn sliding_and_dense_attention_each_pick_their_matching_impl() {
        // With the `if layer % 2 == 0` branch in GEMMA_LIKE_BODY, the
        // FUF alternates Attention / SlidingAttention tiles. The
        // solver must bind each to its OpKind-matching Impl — no
        // OpKind inference, no arch-name filter: each Impl's
        // `matches` singleton-claims its own kind.
        let params = gemma_like_params();
        let (fuf, inferred) = build(GEMMA_LIKE_BODY, &params);
        let lib = starter_library();
        let target = l4_target();

        // Decode (m=1) and prefill (m=512) should both split the
        // picks across the two Attention kinds correctly.
        let workloads = solve(
            &fuf,
            &lib,
            &target,
            &inferred,
            &params.bounds,
            &[1, 512],
            &[],
        )
        .unwrap();

        for (wp, sfuf) in workloads.per_workload.iter() {
            let m = wp.num_tokens;
            let expected_impl_for_decode = ("attention_via_cache", "sliding_attention_via_cache");
            let expected_impl_for_prefill = (
                "attention_prefill_contiguous",
                "sliding_attention_prefill_contiguous",
            );
            let (dense_name, sliding_name) = if m == 1 {
                expected_impl_for_decode
            } else {
                expected_impl_for_prefill
            };

            for sg in sfuf.subgraphs() {
                let tiles = sfuf.tiles_in_subgraph(sg);
                if tiles.len() != 1 {
                    continue;
                }
                let op = fuf.get(tiles[0]).op;
                let name = lib.get(sfuf.impl_of(sg).unwrap()).name();
                match op {
                    OpKind::Attention => assert_eq!(name, dense_name, "dense at m={m}"),
                    OpKind::SlidingAttention => assert_eq!(name, sliding_name, "sliding at m={m}"),
                    _ => {}
                }
            }
        }
    }

    #[test]
    fn tanh_softcap_singleton_tile_is_claimed_by_tanh_softcap_impl() {
        let params = gemma_like_params();
        let (fuf, inferred) = build(GEMMA_LIKE_BODY, &params);
        let lib = starter_library();
        let target = l4_target();
        let workloads = solve(
            &fuf,
            &lib,
            &target,
            &inferred,
            &params.bounds,
            &[1, 512],
            &[],
        )
        .unwrap();

        for (wp, sfuf) in workloads.per_workload.iter() {
            let m = wp.num_tokens;
            let softcap_subgraphs: Vec<_> = sfuf
                .subgraphs()
                .filter(|sg| {
                    let tiles = sfuf.tiles_in_subgraph(*sg);
                    tiles.len() == 1 && fuf.get(tiles[0]).op == OpKind::TanhSoftCap
                })
                .collect();
            assert_eq!(
                softcap_subgraphs.len(),
                1,
                "exactly one TanhSoftCap tile per bucket at m={m}"
            );
            let sg = softcap_subgraphs[0];
            let imp = lib.get(sfuf.impl_of(sg).unwrap()).name();
            assert_eq!(imp, "tanh_softcap_inplace", "m={m}");
        }
    }

    #[test]
    fn empty_workload_points_is_empty_result() {
        let params = llama_params("llama-3.2-1b");
        let (fuf, inferred) = build(LLAMA_BODY, &params);
        let lib = starter_library();
        let target = l4_target();
        let bounds = params.bounds.clone();

        let workloads = solve(&fuf, &lib, &target, &inferred, &bounds, &[], &[]).unwrap();
        assert!(workloads.per_workload.is_empty());
    }

    // ── sk axis regression tests ──────────────────────────────────────
    //
    // These verify the 2-D workload grid works end-to-end: product
    // sweep, sk_bucket threaded into bounds and CostCtx, and coverage
    // holds for every (num_tokens, sk_bucket) pair.

    #[test]
    fn sk_axis_product_sweep_covers_every_pair() {
        let params = llama_params("llama-3.2-1b");
        let (fuf, inferred) = build(LLAMA_BODY, &params);
        let lib = starter_library();
        let target = l4_target();

        let num_tokens = [1u64, 512];
        let sk_buckets = [128u64, 2048, 8192];
        let workloads = solve(
            &fuf,
            &lib,
            &target,
            &inferred,
            &params.bounds,
            &num_tokens,
            &sk_buckets,
        )
        .unwrap();

        assert_eq!(
            workloads.per_workload.len(),
            num_tokens.len() * sk_buckets.len(),
            "product sweep should produce one entry per (m, sk) pair"
        );

        // Every (m, sk) pair must have full coverage, same as the 1-D
        // sweep invariant.
        for &m in &num_tokens {
            for &sk in &sk_buckets {
                let wp = WorkloadPoint {
                    num_tokens: m,
                    sk_bucket: sk,
                };
                let sfuf = workloads
                    .per_workload
                    .get(&wp)
                    .unwrap_or_else(|| panic!("no sfuf for {:?}", wp));
                assert!(
                    sfuf.is_cover_complete(fuf.len()),
                    "cover incomplete at {:?}",
                    wp
                );
            }
        }
    }

    #[test]
    #[ignore = "FlashInfer Impls are not registered in starter_library() \
                while FI is disabled fleet-wide pending a fix for the \
                persistent-kernel CUDA_ERROR_ILLEGAL_ADDRESS at tp>1 with \
                num_kv_heads=1. Re-enable when the FI registration loop \
                in impl_lib.rs is uncommented."]
    fn flashinfer_impls_picked_for_llama_3_2_1b_when_csv_has_rows() {
        // End-to-end solver regression: with the calibrated L4 CSV
        // (which has `flashinfer_attn_bf16_h64_nosoftcap_q32_k8` rows
        // for every calibrated (M, sk) cell), the solver must choose
        // `flashinfer_attention_{decode,prefill}` over
        // `attention_via_cache` / `attention_prefill_contiguous` at
        // the workload points where FI's calibrated cost beats FA2.
        // At (M=1, sk=2048) the FI decode row is ~4.6× faster than
        // FA2 on L4; at (M=512, sk=2048) FI prefill is within a
        // margin of FA2 and the solver may pick either — we assert
        // only on the decode case where FI is unambiguously cheaper.
        let params = llama_params("llama-3.2-1b");
        let (fuf, inferred) = build(LLAMA_BODY, &params);
        let lib = starter_library();
        let target = l4_target();

        let workloads = solve(
            &fuf,
            &lib,
            &target,
            &inferred,
            &params.bounds,
            &[1, 512],
            &[128, 2048],
        )
        .unwrap();

        let mut decode_fi_count = 0;
        let mut prefill_fi_count = 0;
        let mut attention_tiles_at_m1_sk2048 = 0;
        for (wp, sfuf) in workloads.per_workload.iter() {
            for sg in sfuf.subgraphs() {
                let tiles = sfuf.tiles_in_subgraph(sg);
                if tiles.len() != 1 {
                    continue;
                }
                if fuf.get(tiles[0]).op != OpKind::Attention {
                    continue;
                }
                let name = lib.get(sfuf.impl_of(sg).unwrap()).name();
                if wp.num_tokens == 1 && wp.sk_bucket == 2048 {
                    attention_tiles_at_m1_sk2048 += 1;
                    assert_eq!(
                        name, "flashinfer_attention_decode",
                        "FI decode must win over FA2 at (M=1, sk=2048) on calibrated L4",
                    );
                    decode_fi_count += 1;
                } else if wp.num_tokens == 512 && name == "flashinfer_attention_prefill" {
                    prefill_fi_count += 1;
                }
            }
        }
        assert!(
            attention_tiles_at_m1_sk2048 >= 16,
            "llama-3.2-1b has 16 attention layers — every one must be reachable at (M=1, sk=2048); got {attention_tiles_at_m1_sk2048}",
        );
        assert!(
            decode_fi_count >= 16,
            "all 16 attention layers must bind FI decode at (M=1, sk=2048); got {decode_fi_count}",
        );
        // Prefill outcome is workload-dependent and the CSV margin is
        // tight — not asserting a specific count, just recording it
        // so the test is a useful observability signal if this ever
        // changes. (At the time of writing, prefill_fi_count > 0
        // at (M=512, sk=2048) on L4.)
        let _ = prefill_fi_count;
    }

    #[test]
    fn sk_axis_unused_is_backward_compatible() {
        // Passing `&[]` for sk_points must produce exactly the
        // pre-sk-axis behavior: one Assignment per num_tokens,
        // keyed on WorkloadPoint { num_tokens, sk_bucket: 0 }.
        let params = llama_params("llama-3.2-1b");
        let (fuf, inferred) = build(LLAMA_BODY, &params);
        let lib = starter_library();
        let target = l4_target();

        let m_points = [1u64, 64, 512];
        let workloads = solve(
            &fuf,
            &lib,
            &target,
            &inferred,
            &params.bounds,
            &m_points,
            &[],
        )
        .unwrap();

        assert_eq!(workloads.per_workload.len(), m_points.len());
        for &m in &m_points {
            let wp = WorkloadPoint::num_tokens_only(m);
            assert!(workloads.per_workload.contains_key(&wp));
            assert_eq!(wp.sk_bucket, 0);
        }
    }
}
