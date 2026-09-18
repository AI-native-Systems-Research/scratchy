// SPDX-License-Identifier: Apache-2.0
//! ⭐⭐⭐ ONE NODE PER K TRIP — a contraction too deep for LX becomes SEVERAL whole matmuls plus
//! explicit adds, never a PSUM the hardware carries across trips.
//!
//! # THE SHAPE THAT FORCED IT
//!
//! `swiglu_mlp_granite_flat`'s down projection is `m=64 n=4096 k=12800`. Its per-core LX residency
//! ([`matmul_lx_resident_generic`](crate::ir::island::tile_op::matmul_lx_resident_generic)) is
//! `mb_per_core·k·2 + k·out_per_time·2 + mb_per_core·out_per_time·2`, and the WEIGHT term does not
//! shrink with the `out` time-tile past one stick: `12800·64·2 = 1_638_400` B is already 97.7 % of
//! [`USABLE_LX_BYTES`](crate::superdsc_opspec::USABLE_LX_BYTES) on its own. `MAX_CORES` is 32 and
//! `m` is 64, so `mb_per_core` is 2 and the activation slab adds 51_200 B — 1_689_856 B against a
//! 1_677_721 B budget. `WorkPlan::time_tile_for_lx`'s only lever is `out`, so it refuses, and
//! `divide_and_time_tile_for_lx`'s repair then buys the fit with `in: 2` — a REDUCTION-CORE split,
//! and MEASURED on dxp image `dev-2026_09_11-150524` THAT descriptor does not map: the swiglu bundle
//! carrying `{in: 2, mb: 4, out: 4}` failed with `Scheduler failed to find a suitable op mapping`,
//! and at `{in: 1, mb: 8, out: 4}` the same bundle exits 0.
//!
//! ⚠️ THAT IS ONE DESCRIPTOR, NOT A LAW, AND NOTHING HERE TREATS IT AS ONE. `matmul_s3`
//! (`2x512x2048`) is emitted at `in: 4` on this same image, bakes, runs, and computes the CORRECT
//! answer in the shipping 2b fp8 configuration — so an `in > 1` split is NOT refused anywhere in this
//! tree, and it must not be. What this planner does is narrower and purely local: when the whole
//! contraction does not fit LX, it PREFERS a trip count whose plan does not need the reduction-core
//! split, because for THIS shape that split is the only thing the LX repair can offer and it did not
//! map. A shape that fits at `in: 1` reports one trip and is emitted byte-identically.
//!
//! # WHY TRIPS AND NOT A K-TIME TILE
//!
//! A K-TIME tile is the thing this crate calls out of scope, and the reasons are structural, not
//! effort: [`crate::emit::rewrite_op_for_time_tile`] requires the tiled dim to BE the operand's
//! stick (`out` for the kernel and the output, never `in`); the OUTPUT layout omits `in` entirely,
//! so an `in`-tiled op would leave the output un-advanced and every trip would OVERWRITE the last;
//! and [`crate::superdsc_opspec::TimeTile`] carries no accumulate flag. All three are about asking
//! the hardware to CARRY a partial across trips.
//!
//! So nothing here asks for that. Trip `t` is a WHOLE, ordinary matmul over `k/T` of the reduction
//! axis writing its OWN buffer, which is a disjoint write like any other — the tiling property
//! `time_tile_sticklayout_stride_tiles_disjoint` is satisfied rather than dodged — and the
//! accumulation is an explicit [`crate::emit::pw2`] `add` BETWEEN nodes. That is the same answer
//! IBM's C++ gives for a multi-trip sweep: unrolled or re-launched, not folded.
//!
//! ⭐ THE PARTIAL/ACCUMULATOR VOCABULARY WAS ALREADY VENDORED FOR THIS.
//! [`SynthRole::Blk`](crate::place::SynthRole::Blk) and [`Acc`](crate::place::SynthRole::Acc) are
//! declared in `place.rs` under the comment "K-split / down-projection blocking (the only INDEXED
//! roles)" and had NO producer in this tree. This is that producer.

use super::dims::{matmul_split_map_for, matmul_dims};
use super::walk::{InAxis, MatmulWrapperSite, OutAxis, SharedKernelBmmForm, WalkAxis};
use crate::emit::{EmittedOp, In, pw2, rb};
use crate::ir::island::tile_op::{TileOp, TileOpKind};
use crate::place::{PlaceId, SynthRole as R};
use crate::placement::BundleLayout;
use crate::sdsc_abstract::{BlockCols, KernelTag, RowBlockedTag, RowCount, Stk};
use crate::superdsc_opspec::{
    DataFormat, Df, Fp16, MAX_CORES, MaxCores, SdscFoldSet, StickExtent,
};

/// How ONE matmul's reduction axis is cut into trips: `trips` whole matmuls of `k_per_trip` each.
///
/// `trips == 1` is the ORDINARY case and means "emit exactly what was emitted before" — every op
/// that bakes today reports it, which is what makes this decomposition unreachable for them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KTripPlan {
    trips: u32,
    k_per_trip: u32,
}

impl KTripPlan {
    pub fn trips(&self) -> u32 {
        self.trips
    }
    pub fn k_per_trip(&self) -> u32 {
        self.k_per_trip
    }
    /// The element offset, along the flattened reduction axis, of trip `t`'s first K element.
    pub fn k_start(&self, t: u32) -> u32 {
        t * self.k_per_trip
    }
}

/// ⭐⭐⭐ THE TRIGGER IS A MEASURED LX FIT, NOT A SHAPE HEURISTIC — and it is measured by running the
/// SAME tiler the emission runs, on the SAME dims, with the SAME splitter MINUS ITS REDUCTION SPLIT
/// (see the body: a reduction split the cost model merely PREFERRED is not evidence about the fit).
///
/// `T = 1` is tried FIRST, so a matmul whose own work-division already admits an `out` time-tile
/// without a reduction split takes the single-node path and is byte-identical. `swiglu_mlp_flat`
/// (`m=64 n=512 k=128`: weight slab `128·64·2 = 16_384` B) and every rmsnorm/rope operand are
/// nowhere near the budget and provably answer 1 — not incidentally, but because the predicate is
/// the fit itself.
///
/// The ladder walks divisors of the reduction axis's STICK COUNT ascending, so `k/T` is always a
/// whole DF-stick (a sub-stick `in` is `L3DlOpsScheduler.cpp:1040`, refused by `WorkPlan::divide`
/// anyway) and the answer is the FEWEST trips that fit. A trip count that would need MORE cores
/// than the card has is not considered: the cap is the stick count, and every candidate is validated
/// by the real tiler, so an illegal one self-eliminates instead of reaching dxp.
pub fn plan_k_trips<DF: DataFormat>(
    m: u32,
    n: u32,
    k: u32,
    batch: u32,
    operand_df: Df,
    form: SharedKernelBmmForm,
) -> Result<KTripPlan, String> {
    // ⭐⭐⭐ THE FIT THIS PLANNER MEASURES IS THE FIT WITH THE REDUCTION UNSPLIT, AND THAT IS THE WHOLE
    // POINT OF THE MEASUREMENT. The live cost model ([`crate::work::matmul_split_plan`]) k-splits most
    // granite projections when it is free to, for weight-traffic reasons of its own — so a plan it
    // CHOSE at `in > 1` says nothing about whether one node fits LX. Dropping the reduction entry from
    // the division measured here asks the one question a trip count answers: does ONE node fit with the
    // WHOLE reduction resident on every core? An `in > 1` in the answer can then only have come from
    // `divide_and_time_tile_for_lx`'s LX repair, which is the thing this decomposition replaces.
    //
    // ⚠️ THE EMISSION IS NOT PINNED AND MUST NOT BE — that is the mistake this shape of code invites.
    // `matmul_opspec` keeps the live splitter and the cost model's own reduction split; only the NODE
    // COUNT is decided here. (Pinning the model globally instead moved shipping fp8 arithmetic.)
    let live = matmul_split_map_for(form);
    let splitter = move |dims: &[crate::superdsc_opspec::ItDim], cores: u32| {
        let mut splits = live(dims, cores);
        splits.remove(InAxis::NAME);
        splits
    };
    let stk = operand_df.elems_per_stick();
    if stk == 0 || !k.is_multiple_of(stk) {
        return Err(format!(
            "plan_k_trips: K={k} is not a whole multiple of the {stk}-elem {} stick",
            operand_df.dataformat()
        ));
    }
    let k_sticks = k / stk;
    let mut tried: Vec<u32> = Vec::new();
    for t in 1..=k_sticks {
        if !k_sticks.is_multiple_of(t) {
            continue;
        }
        let k_t = k / t;
        // The candidate's OWN dims, built exactly as `matmul_opspec_split` builds them — including
        // the per-axis operand stick basis, so an fp8 activation's 128-stick reduction is measured
        // in fp8 sticks here too and not in fp16 ones.
        let n_ext = StickExtent::<DF>::new(n)?;
        let k_ext = StickExtent::<DF>::new(k_t)?;
        let mut dims = matmul_dims::<DF>(m, &n_ext, &k_ext, batch);
        if let Some(d) = dims.iter_mut().find(|d| d.name == InAxis::NAME) {
            d.df = operand_df;
        }
        let tile_op = TileOp {
            kind: TileOpKind::Matmul,
            dims,
            df: operand_df,
        };
        // The tiler is the AUTHORITY on the fit, and it is asked with the reduction UNSPLIT (above):
        // `Err` means not even `out`-tiled to one stick plus the whole reduction-core ladder placed it,
        // and an `in > 1` plan means only `divide_and_time_tile_for_lx`'s LX REPAIR placed it — it had
        // to spread the reduction across cores to buy the fit. Either way this trip count is not the
        // answer. (An `in > 1` split is REFUSED NOWHERE in this tree; this door declines to NEED one
        // for a shape it can cut into trips instead — see the module docs.)
        match tile_op.tile(MaxCores::<MAX_CORES>, splitter, OutAxis::NAME) {
            Ok(tiled) if tiled.plan.split_of(InAxis::NAME) <= 1 => {
                return Ok(KTripPlan {
                    trips: t,
                    k_per_trip: k_t,
                });
            }
            Ok(tiled) => tried.push(tiled.plan.split_of(InAxis::NAME)),
            Err(_) => tried.push(0),
        }
        // A trip per stick is the floor; past that there is no divisor left to try.
    }
    Err(format!(
        "matmul {m}x{n}x{k}: NO K-trip count fits LX without splitting the reduction axis across \
         cores. Every divisor of the {k_sticks}-stick reduction was tried and each one either \
         refused outright (the `out` time-tile bottoms out at one stick) or was placed only by the \
         LX repair's `in > 1` reduction-core split, which this door declines to NEED (the `in: 2` \
         descriptor for this very shape did not map on dxp image `dev-2026_09_11-150524`; see the \
         module docs, and note that an `in > 1` split is refused NOWHERE in this tree). The per-trip \
         `in` splits the ladder reported, in trip-count \
         order, were {tried:?} (0 = the tiler refused the shape outright). A shape here needs a \
         SMALLER per-core output tile or a smaller `m`, not more trips: the weight term \
         `k_per_trip·one_stick·2` is what does not fit, and it is already at its floor."
    ))
}

/// ⭐ THE K-TRIP EMISSION: `trips` matmul descriptors into per-trip partial buffers, then the adds
/// that sum them into `o`.
///
/// # THE ACCUMULATION ORDER IS A DELIBERATE CHOICE AND IT IS STATED HERE
///
/// The partials are summed as a LEFT FOLD IN ASCENDING K:
/// `((P₀ + P₁) + P₂) + …`, so trip 0's slab (the lowest K elements) is the innermost term. That is
/// the order a single monolithic contraction streams the reduction axis in, and the order scratchy's
/// own emulator K-loop accumulates in (`ktir-optimizer/src/matmul_tile.rs`'s `ScfFor` over
/// `lb=0, ub=k, step=kb` with an `ArithAddf` per trip). It is NOT associatively identical to a dxp
/// PSUM accumulation, and every partial and every accumulator here is fp16, so the difference is
/// VISIBLE — this crate validates no arithmetic (`dxp_standalone` executes none), so a future
/// numeric check needs to know which order it is checking. This is that order.
///
/// # THE BUFFERS
///
/// Trip `t` writes `Blk(t)` `[m, n]`; add `i` writes `Acc(i)`, except the LAST add, which writes `o`
/// directly — so `trips = 2` costs two partials and one add and no accumulator at all. Each is
/// declared to [`BundleLayout::synth`] at its full `[m, n]` footprint, so `resolve_seg_base` places
/// it in the Intermediate segment rather than bump-allocating it lazily (which is a build error).
///
/// # THE OPERAND WINDOWS
///
/// Both contracted operands are addressed by the ONE rank-2 stick-block law
/// `(j/stk)·(dims[0]·stk) + i·stk + (j%stk)`:
///
/// * ACTIVATION `[mb, in]`, stick `in`. Its stick-GROUP stride is `mb·stk`, which the trip does not
///   change (`mb` is whole), so a K-window is CONTIGUOUS and needs only the base offset
///   `k_start·m`. No device extent.
/// * KERNEL `[in, out]`, stick `out`. Its stick-group stride is `in·stk` — the FULL weight depth —
///   so a trip must DECLARE `in = k` (`kernel_phys_in`) while sweeping `k/T`, or every `out` core
///   but the first reads another trip's slab. Its base offset is `k_start·stk`.
///
/// `n` arrives ALREADY RESOLVED by the caller (`DeviceWidth::for_output` at the FULL `k`, narrowed
/// by `out_width_the_weight_holds`) and is used unchanged for every trip: the device width is a
/// property of the whole matmul and of the weight's placement, not of one trip's depth.
#[allow(clippy::too_many_arguments)]
pub fn try_assemble_matmul_k_trips(
    op_name: &str,
    m: u32,
    n: u32,
    k: u32,
    batch: u32,
    a: &Stk<RowBlockedTag>,
    w: &Stk<KernelTag>,
    o: &Stk<RowBlockedTag>,
    // The output's IDENTITY, not just its spelling — the partials and accumulators are synthetics
    // DERIVED from it (`PlaceId::synth`), which is the join key the layout is keyed by.
    out_id: PlaceId,
    plan: KTripPlan,
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
) -> Result<Vec<EmittedOp>, String> {
    let trips = plan.trips();
    if trips <= 1 {
        return Err(format!(
            "try_assemble_matmul_k_trips {op_name}: called with {trips} trip(s). One trip is the \
             ORDINARY matmul and must go through `try_assemble_matmul_seeded` so that path stays \
             byte-identical; this entry point exists only for a decomposition."
        ));
    }
    let stk = Fp16::ELEMS_PER_STICK;
    let k_t = plan.k_per_trip();

    // Every buffer this decomposition invents, declared at its FULL `[m, n]` footprint before any
    // op references it. Lazy allocation is a build error by design (`resolve_seg_base`).
    let syn = |r: R| crate::placement::syn(layout, out_id.synth(r));
    if let Some(l) = layout {
        for t in 0..trips {
            l.synth(out_id.synth(R::Blk(t)), &[m, n]);
        }
        // Adds 1..trips-1 land in accumulators; the last one writes `o`.
        for i in 1..trips.saturating_sub(1) {
            l.synth(out_id.synth(R::Acc(i)), &[m, n]);
        }
    }
    let blk_names: Vec<String> = (0..trips).map(|t| syn(R::Blk(t))).collect();
    let acc_names: Vec<String> = (0..trips).map(|i| syn(R::Acc(i))).collect();

    let mut ops: Vec<EmittedOp> = Vec::with_capacity((2 * trips - 1) as usize);

    // ── THE TRIPS ────────────────────────────────────────────────────────────────────────────────
    for t in 0..trips {
        let k0 = plan.k_start(t);
        let partial = rb(&blk_names[t as usize], m, n);
        let op = super::opspec::matmul_opspec_split::<Fp16, _>(
            m,
            n,
            k_t,
            batch,
            a.name(),
            w.name(),
            partial.name(),
            // ACTIVATION: a K-window of `[mb, in]` is whole stick GROUPS, each `m·stk` elements.
            k0 * m,
            // KERNEL: `k` indexes the NON-stick axis of `[in, out]`, whose unit step is one stick.
            k0 * stk,
            0,
            <Fp16 as DataFormat>::DF,
            None,
            // The weight is `k` deep however little of it this trip contracts — see the fn doc.
            Some(k),
            SharedKernelBmmForm::batch_inner_proven(MatmulWrapperSite::witness()),
            None,
            matmul_split_map_for(SharedKernelBmmForm::batch_inner_proven(
                MatmulWrapperSite::witness(),
            )),
        )?;
        let folds = SdscFoldSet::new(op.iter.cores_used());
        ops.push(
            crate::emit::emit_sdsc_tiled(
                &format!("{op_name}_k{t}"),
                &op,
                &folds,
                sym_id_base,
                layout,
            )
            .map_err(|e| e.0)?,
        );
    }

    // ── THE ACCUMULATION (left fold, ascending K — see the fn doc) ───────────────────────────────
    let rows = RowCount::of_token_rows(m);
    let cols = BlockCols::of_feature_cols(n);
    for i in 1..trips {
        let lhs = if i == 1 {
            rb(&blk_names[0], m, n)
        } else {
            rb(&acc_names[(i - 1) as usize], m, n)
        };
        let rhs = rb(&blk_names[i as usize], m, n);
        let last = i + 1 == trips;
        let dst = if last {
            rb(o.name(), m, n)
        } else {
            rb(&acc_names[i as usize], m, n)
        };
        ops.push(pw2(
            &format!("{op_name}_kacc{i}"),
            "add",
            rows,
            cols,
            In::full(&lhs),
            In::full(&rhs),
            &dst,
            sym_id_base,
            layout,
        ));
    }
    Ok(ops)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn proven() -> SharedKernelBmmForm {
        SharedKernelBmmForm::batch_inner_proven(MatmulWrapperSite::witness())
    }

    /// ⭐ THE SHAPE THIS EXISTS FOR. `swiglu_mlp_granite_flat`'s down projection needs exactly TWO
    /// trips, and the CONTROL is the first assertion: at one trip the tiler places it only with an
    /// `in: 2` reduction-core split — the descriptor that did not map for this shape.
    #[test]
    fn granite_down_proj_at_64_rows_needs_two_k_trips() {
        // CONTROL — the whole contraction is placed only by an `in > 1` split.
        let n_ext = StickExtent::<Fp16>::new(4096).unwrap();
        let k_ext = StickExtent::<Fp16>::new(12800).unwrap();
        let dims = matmul_dims::<Fp16>(64, &n_ext, &k_ext, 1);
        let whole = TileOp {
            kind: TileOpKind::Matmul,
            dims,
            df: Df::Fp16,
        }
        .tile(
            MaxCores::<MAX_CORES>,
            matmul_split_map_for(proven()),
            OutAxis::NAME,
        )
        .expect("the LX repair places it, just only with a reduction-core split");
        assert!(
            whole.plan.split_of(InAxis::NAME) > 1,
            "the control must be the reduction-core-split case, got in={}",
            whole.plan.split_of(InAxis::NAME)
        );

        let p = plan_k_trips::<Fp16>(64, 4096, 12800, 1, Df::Fp16, proven()).unwrap();
        assert_eq!(p.trips(), 2, "the FEWEST trips that fit, not more");
        assert_eq!(p.k_per_trip(), 6400);
        assert_eq!(p.k_start(0), 0);
        assert_eq!(p.k_start(1), 6400);
    }

    /// The TOY width of the same kernel (`swiglu_mlp_flat`) and its two wide siblings are ONE trip,
    /// so the single-node path — and every byte it emits — is unreachable from this decomposition.
    #[test]
    fn the_shapes_that_already_bake_are_one_trip() {
        for (m, n, k) in [
            (64, 512, 128),    // swiglu_mlp_flat's down proj (n padded to 512)
            (64, 256, 128),    // and unpadded
            (64, 12800, 4096), // granite gate/up proj — wide N, shallow K
            (1, 4096, 12800),  // granite-8b decode down proj at ONE row
        ] {
            let p = plan_k_trips::<Fp16>(m, n, k, 1, Df::Fp16, proven())
                .unwrap_or_else(|e| panic!("{m}x{n}x{k}: {e}"));
            assert_eq!(p.trips(), 1, "{m}x{n}x{k} must stay a single node");
            assert_eq!(p.k_per_trip(), k);
        }
    }

    /// The trip count is the LX FIT, so it MOVES with the row count on one fixed `n`/`k` — which is
    /// what makes it a measurement rather than a table. granite-8b's down projection is one node at
    /// one decode row and needs trips once `m` puts two rows on a core.
    #[test]
    fn the_trip_count_tracks_the_row_count() {
        let t = |m: u32| plan_k_trips::<Fp16>(m, 4096, 12800, 1, Df::Fp16, proven()).unwrap();
        assert_eq!(t(1).trips(), 1, "one row per core fits");
        assert!(
            t(64).trips() > 1,
            "two rows per core does not, and that is the whole gap"
        );
    }

    /// A one-trip plan must NOT reach the decomposing emitter — that path has to stay the ordinary
    /// assembler's, or `swiglu_mlp_flat` moves.
    #[test]
    fn one_trip_is_refused_by_the_decomposing_emitter() {
        let mut sym = 0i64;
        let e = try_assemble_matmul_k_trips(
            "x",
            64,
            512,
            128,
            1,
            &rb("a", 64, 128),
            &Stk::<KernelTag>::kernel(128, 512, "w"),
            &rb("o", 64, 512),
            PlaceId::Act(7),
            KTripPlan {
                trips: 1,
                k_per_trip: 128,
            },
            &mut sym,
            None,
        );
        let e = match e {
            Err(m) => m,
            Ok(ops) => panic!("one trip must be refused, got {} op(s)", ops.len()),
        };
        assert!(e.contains("byte-identical"), "got: {e}");
    }
}
