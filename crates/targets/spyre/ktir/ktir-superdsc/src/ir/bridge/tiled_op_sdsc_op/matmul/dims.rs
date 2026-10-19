//! Matmul family: the TileOp dim-vocabulary (`matmul_dims`) and the cost-model-splitter adapter
//! (`matmul_split_map`) that `opspec`'s builders feed into `TileOp::tile`. See `super`'s module doc.

use super::walk::{InAxis, MbAxis, OutAxis, WalkAxis, YAxis};
use crate::superdsc_opspec::{DataFormat, Df};

/// Build the iteration `ItDim`s for a matmul. `mb`=M (out), `out`=N (out,stick),
/// `in`=K (reduction,stick), `x`=batch (out), `y`=1 (out).
///
/// `DF` is the COMPILE-TIME device format of the matmul OPERANDS on the stick axes
/// (N, K): [`Fp16`] for the dense path (64-lane sticks), [`Fp8`] for the W8A8 path
/// (128-lane sticks). The stick basis carried onto the `out`/`in` dims is
/// `DF::ELEMS_PER_STICK` — a TYPE-LEVEL const, not a hand-picked value — so the
/// work-division splits them by the operand format's stick and a sub-stick per-core
/// slice (the `L3DlOpsScheduler:1070 multiple-of-stick` DtException) is impossible to
/// express: an fp8 matmul can only be built from `StickExtent<Fp8>` extents, and its
/// dims carry `Df::Fp8` ⇒ 128, never fp16's 64.
pub fn matmul_dims<DF: DataFormat>(
    m: u32,
    n_ext: &crate::superdsc_opspec::StickExtent<DF>,
    k_ext: &crate::superdsc_opspec::StickExtent<DF>,
    batch: u32,
) -> Vec<crate::superdsc_opspec::ItDim> {
    // Plain 2-D matmul (batch==1): dims {mb, out, in} — NO `y`, matching torch-spyre's own
    // generate_sdsc for a 2-D mm (REF primaryDsInfo A=[mb,in]/C=[mb,out], N_={mb,out,in}). A `y`=1
    // phantom made the tensors 3-D and forced the batchmatmul path, which mis-iterated mb at mq>1
    // (the prefill collapse). A TRUE batched matmul (batch>1, attention bmm) keeps `y`=batch.
    let mut dims = vec![
        crate::superdsc_opspec::ItDim {
            name: MbAxis::NAME,
            size: m,
            is_reduction: false,
            is_stick: false,
            df: Df::Fp16, // M (token) axis is not a stick axis — basis unused.
        },
        crate::superdsc_opspec::ItDim {
            name: OutAxis::NAME,
            size: n_ext.elems(),
            is_reduction: false,
            is_stick: true,
            df: DF::DF, // stick basis = DF::ELEMS_PER_STICK (fp16=64, fp8=128), type-sourced.
        },
        crate::superdsc_opspec::ItDim {
            name: InAxis::NAME,
            size: k_ext.elems(),
            is_reduction: true,
            is_stick: true,
            df: DF::DF,
        },
    ];
    if batch > 1 {
        dims.push(crate::superdsc_opspec::ItDim {
            name: YAxis::NAME,
            size: batch,
            is_reduction: false,
            is_stick: false,
            df: Df::Fp16, // batch axis is not a stick axis — basis unused.
        });
    }
    dims
}

// The matmul cost-model splitter as a [`crate::superdsc_opspec::WorkPlan`] splitter: maps the
// `(b,m,n,k)` cost split onto the named dims (m→mb, n→out, k→in, b→x). Reuses the existing
// `matmul_split_plan`/`matmul_cost_split`/`core_split` (the live emitter's cost-model search)
// verbatim.
// ⛔⛔⛔ THE THREAD-LOCAL `ROWS_ARE_REQUESTS` FLAG WAS DELETED HERE, AND IT WAS THE ROOT OF THE RAGGED
// BATCH-DECODE BUG. It held the single most semantically important fact about a bundle — whether a query
// row is a POSITION of one sequence or an independent SEQUENCE — in ambient mutable state, set once per
// bundle. Its own comment named the hazard and then chose the state anyway:
//
//   "a batched decode's `mb=8` is INDISTINGUISHABLE from a prompt chunk's"
//   "Set once at the top of the lowering ... rather than threaded through four signatures"
//
// Threading it through those four signatures is precisely the work that was skipped, and the cost was
// measured by counting: 101 sites in the emitter branch on the row count `mq`, and FOUR consulted this
// flag. The other ~97 emit a batched decode AS A PREFILL CHUNK — a wrong KIND, not a wrong value. That is
// invisible until the batch is RAGGED, because a prefill chunk's defining property is that its rows share
// ONE valid extent, which a uniform decode batch satisfies by coincidence and a ragged one never does.
//
// The kind is now `sdsc_abstract::QueryRows<K>` with `K` in {`Chunk`, `Requests`} — carried in the type,
// so no site can read it without having been given it, and none can infer it from a count. Nothing here
// may reintroduce an ambient copy: if the splitter needs the kind, it takes it as a parameter.
//
// ⛔ AND THE LAST AMBIENT SURVIVOR, THE DECODE BATCH'S `mb`-SPLIT BAN, IS GONE TOO — BECAUSE IT WAS
// MEASURED, NOT BECAUSE IT WAS AMBIENT. It was a thread-local set for the span of
// `lower_subtile_tape_to_ktir`, and every descriptor is built later, in `emit_bundle`, so it never
// reached one: every shipped decode batch already ran the cost model's own split. Making it reach the
// emitter (the weight-stationary form: no `mb` split, and the cost model's `in`/`mb` cores handed to
// `out`) was built and measured on the card against that split, ABAB, conc 1/2/4/8 at ~1k tokens:
//
//   | granite-3.1  | conc-1 | conc-2 | conc-4 | conc-8 |
//   |--------------|--------|--------|--------|--------|
//   | 2b fp16      | flat   | flat   | flat   | flat   |
//   | 8b fp16      | flat   | −21%   | −15%   | −6.6%  |
//
// So the premise — that an `mb` split reloads the stationary weight per split and cancels batching's
// amortization — does not survive the card: a decode batch keeps the cost model's split, the same one
// prefill runs, and the splitter needs no row kind at all.

/// The splitter every matmul takes: the joint cost-model search, for prefill and decode batches alike.
pub fn matmul_split_map(
    dims: &[crate::superdsc_opspec::ItDim],
    max_cores: u32,
) -> std::collections::BTreeMap<&'static str, u32> {
    matmul_split_map_inner(dims, max_cores, BatchedSplit::JointCostModel)
}

/// The REQUEST-BATCH splitter: identical to [`matmul_split_map`] on every dim set without a real
/// batch axis, and batch-divides-first on the y>1 arm. Handed out only by the request regime's
/// [`super::walk::RungRegime::split_map`].
pub(crate) fn matmul_split_map_batch_requests(
    dims: &[crate::superdsc_opspec::ItDim],
    max_cores: u32,
) -> std::collections::BTreeMap<&'static str, u32> {
    matmul_split_map_inner(dims, max_cores, BatchedSplit::BatchDividesFirst)
}

/// ⭐ THE CONTRACTION-FIRST SPLITTER — every core a whole-stick run of `in`, rows and output whole.
///
/// Reached only through [`crate::sdsc_abstract::ContractionSplit::AcrossCores`] (the split-KV chunk's
/// one-request value leg), never as a regime's policy: that op's output is ONE stick of a few rows, so
/// the planner's single-stick arm gives it its rows alone (`core_split(mb)` — 4 cores at granite-2b, 1
/// at granite-8b's per-head row), each contracting the whole chunk. Here the cores divide the chunk
/// instead: the largest divisor of the contraction's stick count within the core budget, so every
/// core's run is whole sticks (the same law the reduction split obeys on the planner's arm). dxp
/// PSUM-accumulates the partials into the shared output — the emitter admits a shared output only for a
/// split `in` that the output does not carry (`emit_sdsc`'s reduction-split rule).
pub(crate) fn matmul_split_map_contraction(
    dims: &[crate::superdsc_opspec::ItDim],
    max_cores: u32,
) -> std::collections::BTreeMap<&'static str, u32> {
    use crate::work::FP16_ELEMS_PER_STICK;
    let (k, stick) = dims
        .iter()
        .find(|d| d.name == InAxis::NAME)
        .map(|d| (d.size, d.df.elems_per_stick()))
        .unwrap_or((1, FP16_ELEMS_PER_STICK));
    let sticks = k.div_ceil(stick).max(1);
    let split = (1..=max_cores.min(sticks))
        .rev()
        .find(|s| sticks.is_multiple_of(*s))
        .unwrap_or(1);
    let mut map = std::collections::BTreeMap::new();
    if split > 1 {
        map.insert(InAxis::NAME, split);
    }
    map
}

/// ⭐ THE COLUMNS-FIRST SPLITTER — every core a whole-stick run of `out`, rows and contraction whole.
///
/// Reached only through [`crate::sdsc_abstract::OutputSplit::ColumnsAcrossCores`] (a part of the
/// split-KV chunk's one-request score, which writes its columns of a wider score block). The cores
/// divide the columns alone: the largest divisor of the output's stick count within the core budget,
/// so every core's run is whole sticks of the operand's own format.
pub(crate) fn matmul_split_map_columns(
    dims: &[crate::superdsc_opspec::ItDim],
    max_cores: u32,
) -> std::collections::BTreeMap<&'static str, u32> {
    use crate::work::FP16_ELEMS_PER_STICK;
    let (n, stick) = dims
        .iter()
        .find(|d| d.name == OutAxis::NAME)
        .map(|d| (d.size, d.df.elems_per_stick()))
        .unwrap_or((1, FP16_ELEMS_PER_STICK));
    let sticks = n.div_ceil(stick).max(1);
    let split = crate::work::core_split(sticks, max_cores);
    let mut map = std::collections::BTreeMap::new();
    if split > 1 {
        map.insert(OutAxis::NAME, split);
    }
    map
}

/// ⭐ THE ROWS-FIRST SPLITTER — every core a run of `mb`, columns and contraction whole.
///
/// Reached only through [`crate::sdsc_abstract::OutputSplit::RowsAcrossCores`] (the chunk-formed causal
/// block's score, whose output stays core-local in LX for consumers that each divide its rows alone).
/// The cores divide the rows exactly as the pointwise door's rows-first law does
/// ([`crate::work::distribute_cores`]: `core_split(mb)`), so core `c`'s output slice IS the slice the
/// row max, `esub`, `exp`, row sum and PV on core `c` read. Whether those slices agree is not assumed
/// here — `emit`'s core-local read law compares every op's per-core slice at build.
pub(crate) fn matmul_split_map_rows(
    dims: &[crate::superdsc_opspec::ItDim],
    max_cores: u32,
) -> std::collections::BTreeMap<&'static str, u32> {
    let mb = dims
        .iter()
        .find(|d| d.name == MbAxis::NAME)
        .map(|d| d.size)
        .unwrap_or(1);
    let split = crate::work::core_split(mb, max_cores);
    let mut map = std::collections::BTreeMap::new();
    if split > 1 {
        map.insert(MbAxis::NAME, split);
    }
    map
}

/// The work-division splitter signature `TileOp::tile` consumes — the type a rung regime's
/// [`super::walk::RungRegime::split_map`] answers with.
pub(crate) type SplitMapFn =
    fn(&[crate::superdsc_opspec::ItDim], u32) -> std::collections::BTreeMap<&'static str, u32>;

/// The live splitter FOR a shared-kernel bmm form. Each regime pairs its splitter with its walk in
/// ONE [`super::walk::RungRegime`] impl, so the split policy and the declared order cannot be
/// paired across regimes; this only opens the form.
pub(crate) fn matmul_split_map_for(form: super::walk::SharedKernelBmmForm) -> SplitMapFn {
    use super::walk::{RungRegime, SharedKernelBmmRegime};
    match form.regime() {
        SharedKernelBmmRegime::Proven(regime) => regime.split_map(),
        SharedKernelBmmRegime::Requests(regime) => regime.split_map(),
    }
}

/// HOW the y>1 (batched attention bmm) arm divides its cores — the split half of a rung regime,
/// each value reachable only through its regime's [`super::walk::RungRegime::split_map`].
#[derive(Clone, Copy)]
enum BatchedSplit {
    /// The joint most-cores search over `(y, mb, out, in)` — the split every shipped bundle
    /// carries, on-hardware-proven at mq==1 (`{mb:1, y:4}` on 4 cores).
    JointCostModel,
    /// The BATCH divides FIRST, at its full `core_split`; only the remaining cores go to
    /// `mb`/`out`. `y` is the outermost axis of the head-outermost walk (one head/request plane
    /// per step), so a y-slice is the cheapest disjoint unit of work — where the joint search
    /// saturates `mb` whenever `mb` has the larger divisor (measured at mq=23: `{mb:23, y:1}` on
    /// 23 cores), and a y-split of 1 at a prime row count is a planner artifact, not a work-shape
    /// fact. At the ladder widths (y=4, mb ∈ {2,4,8}, one-stick `out`) y-first picks the SAME
    /// split the joint search does; the two differ only where the joint search saturates rows.
    BatchDividesFirst,
}

fn matmul_split_map_inner(
    dims: &[crate::superdsc_opspec::ItDim],
    max_cores: u32,
    batched_split: BatchedSplit,
) -> std::collections::BTreeMap<&'static str, u32> {
    use crate::work::{FP16_ELEMS_PER_STICK, core_split, matmul_cost_split, matmul_split_plan};

    let ext = |name: &str| {
        dims.iter()
            .find(|d| d.name == name)
            .map(|d| d.size)
            .unwrap_or(1)
    };
    // Stick basis for the OUTPUT (N) axis: 64 for fp16, 128 for fp8 — so the `out`
    // split is in whole sticks of the OPERAND'S format. Splitting an fp8 `out` at the
    // fp16 ÷64 basis is exactly the sub-stick per-core slice dxp rejects (DtException
    // L3DlOpsScheduler:1070); using `d.df` keeps every core ≥1 fp8 (128-lane) stick.
    let out_stick_elems = dims
        .iter()
        .find(|d| d.name == OutAxis::NAME)
        .map(|d| d.df.elems_per_stick())
        .unwrap_or(FP16_ELEMS_PER_STICK);
    let out_sticks = |elems: u32| elems.div_ceil(out_stick_elems);
    let (y, mb, out, in_) = (
        ext(YAxis::NAME),
        ext(MbAxis::NAME),
        ext(OutAxis::NAME),
        ext(InAxis::NAME),
    );
    let mut map = std::collections::BTreeMap::new();
    // WHICH DIMS A MATMUL'S WORK IS SPLIT ON (`SCRATCHY_SDSC_SPLIT_TRACE=1`). This is the one thing
    // no counter here reports and the one that decides weight TRAFFIC: an `mb` split hands every
    // core the SAME stationary weight (see the HW CAP note below), so `mb=c` reads the weight c
    // times, while an `out` split gives each core its own N-slice and reads it once. A batched
    // decode's cost is `11.2 + 3.79*B` ms — linear in requests — which is the shape a per-core
    // weight re-read would have, and this says whether that is what is happening.
    let trace = std::env::var_os("SCRATCHY_SDSC_SPLIT_TRACE").is_some();
    // A decode batch takes the cost model's split like prefill does: the weight-stationary form that
    // would have forbidden an `mb` split here measured slower on the card (see the note at the top of
    // this file).
    if y <= 1 {
        // ⭐ 2026-07-08 ROOT-CAUSE FIX (IBM-reference-confirmed): SPLIT `mb` (M) across cores, not just
        // `out`. The OLD "split ONLY out, keep mb WHOLE (the PT streams M≤8 on one core)" was the mq=8
        // prefill INF root cause. Evidence: (1) IBM's work_division.py `multi_dim_iteration_space_split`
        // splits the OUTPUT dims (M AND N) across cores, reduction last (~/git/torch-spyre; test_it_space_
        // splits.py::test_matmul_2d splits M). (2) Our OWN `distribute_cores` (the pointwise/reduce/silu
        // splitter) splits mb — which is why pointwise mb=8 tensors are FINITE on-card while matmul outputs
        // (this path) were INF. (3) The historical "mb-split collapse" was the `y`-phantom forcing the
        // batchmatmul path (matmul_dims ~2348) — GONE now (plain 2-D matmul), so mb-split on the clean 2-D
        // matmul uses the SAME proven per_core_addr/coordInfo mechanism pointwise already uses. Split mb
        // FIRST (so EVERY M>1 projection gets it, incl wide-N q/o/down_proj), then `out` (whole STICK count,
        // stick atomicity) with the remaining cores. `in`(K)/reduction stays UNSPLIT (a K-split needs a
        // cross-core PSUM merge — see distribute_cores; mb×out already fills the cores). At M=1 (decode)
        // core_split(1,·)=1 ⇒ mb unsplit ⇒ byte-identical to the proven-coherent decode path.
        // ⛔ HW CAP (PT array): the weight is STATIONARY in the PT (systolic) array and M streams THROUGH it
        // in passes of PT_ROWS=8. Cap the M-split at mb/8 so each core keeps >= 8 rows: a 1-row-per-core
        // split (prime m=31 → core_split=31) underfills the array AND — since all mb-split cores read the
        // SAME stationary weight — broadcasts the full weight to >8 cores (the fragile edge where the
        // batched V-proj read 0 while K survived). The remaining cores fill via `out`, where each core reads
        // an N-SLICE of the weight (no full-weight broadcast — the decode geometry that works). torch-spyre
        // work_division does the same (targets ~40 rows/core, penalizes tiny M-tiles). m<8 (decode m=1) ⇒
        // cap=1 ⇒ mb unsplit ⇒ byte-identical to the proven decode path.
        if out_sticks(out) > 1 {
            // REAL projection (q/k/v/o/mlp, multi-stick output) — split EXACTLY per torch-spyre's
            // analytic cost model (`matmul_split_plan`, the faithful `_cost_model_matmul_planner` port),
            // NOT the old hand-written "cap M at mb/8 then fill with out" heuristic that only matched
            // torch-spyre by accident. `shared_weight=true`: a projection's weight is loaded once
            // (broadcast across M), the torch-spyre `rhs_loaded_once` path.
            if mb > 1 {
                // PREFILL (m>1): split EXACTLY per torch-spyre's cost model, which K-splits most granite
                // projections. K-split: `k` cores each contract a K-slice into a PARTIAL product that dxp
                // PSUM-accumulates into the SHARED output tile (torch-spyre's reduction-dim split); the
                // #50 disjoint-output guard is relaxed for a reduction (`in`) split (see emit_sdsc).
                let mut s = matmul_split_plan(mb, out, in_, 1, out_stick_elems, max_cores, true);
                // ⛔⛔⛔ THE REDUCTION SPLIT MUST LEAVE EVERY CORE WHOLE STICKS OF `in`. Ported from
                // `a1cde93f` on the gemma-4 branch, whose one-op-per-dxp-compile table measured it:
                //
                //   | k     | n     | in | out | k/in | /128 | n/out | /128 | dxp     |
                //   | 3840  | 15360 |  1 |  30 | 3840 | 30   |  512  |  4   | ok      |
                //   | 15360 |  3840 |  5 |   6 | 3072 | 24   |  640  |  5   | ok      |
                //   | 3840  | 15360 |  4 |   8 |  960 | 7.5  | 1920  | 15   | refused |
                //
                // Checking only `out` got gemma-4's 40-layer body compiling and left FOUR ops refused:
                // `k=3840 n=15360` split `in=4 out=8` leaves a per-core `in` of 960 = SEVEN AND A HALF
                // fp8 sticks. dxp schedules the reduction slice in the same unit as the output slice, so
                // it obeys the same law as `out` — `DtException: There must be at least one valid
                // candidate.` (`L3DlOpsScheduler.cpp:1375`).
                //
                // ⭐ THE STICK BASIS IS THE REDUCTION OPERAND'S, which is why this reads `in`'s own `df`
                // rather than reusing `out_stick_elems`: an fp8 matmul contracts in 128-lane sticks while
                // its output stick basis is fp16's 64 (`operand_df` on the fp8 path). Reusing the output
                // basis would pass a 7.5-stick `in` as "60 whole sticks" of the wrong unit.
                //
                // Declining is safe by construction: `s.k` only ever DROPS toward 1, and `in=1` is
                // trivially whole sticks, so the fallback is the unsplit reduction the decode branch runs.
                let in_stick_elems = dims
                    .iter()
                    .find(|d| d.name == InAxis::NAME)
                    .map(|d| d.df.elems_per_stick())
                    .unwrap_or(FP16_ELEMS_PER_STICK);
                while s.k > 1
                    && !(in_.is_multiple_of(s.k) && (in_ / s.k).is_multiple_of(in_stick_elems))
                {
                    s.k -= 1;
                }
                // ⛔⛔ NO K-SPLIT, PERIOD — MEASURED ON CARD (2026-09-12 and 2026-09-27, granite
                // swiglu_mlp tiled_k, the first ladder case ever to split `in`; and the small
                // tiled_k twin, which runs `in:2` splits CARD-CORRECT at 0.9229).
                //
                // The ORIGINAL 2026-09-12 reading blamed non-accumulating partials for the 0.50-slope
                // signature. The 2026-09-27 re-measurement pinned the real defect one layer down: the
                // WALK, not the split — with `N_.in_` = the SWEPT window k, a kernel core striding
                // >1 out-stick walked them at `k·64` where the staged row-blocked weight packs them
                // `in_phys·64` apart (gate 2×, down 25×), so every out-stick past the first gathered
                // WRONG weight sticks and the output's first stick of each 512-block carried
                // ~0.5-corr signal with the rest noise (within2pct 0.0070, max|err| 6.93). Fixed at
                // the walk — `assemble_matmul_windowed` rewrites `N_.in_` to the kernel's PHYSICAL
                // extent post-emission (the scratchy row-window law: `N_` is the RESIDENCY the card
                // reconstructs from; the swept k is WORK, carried by the splits/starts). With that
                // fix in place the twin's `in:2` splits and the full granite case both run correct,
                // and clamping `s.k = 1` here was re-measured as NEUTRAL (identical score to 4
                // digits) — so the clamp is REMOVED to keep every card-validated non-windowed bundle
                // byte-identical, and the `in` split stays legal exactly as the twin proves it.
                if s.m > 1 {
                    map.insert(MbAxis::NAME, s.m);
                }
                if s.n > 1 {
                    map.insert(OutAxis::NAME, s.n);
                }
                if s.k > 1 {
                    map.insert(InAxis::NAME, s.k);
                }
            } else {
                // ONE ROW: not the cost model (it prices M-tile shapes a single row does not have), but
                // [`crate::work::one_row_core_fill`] — the out-only split, filled to every core with a
                // whole-stick `in` factor where the output's stick count leaves cores idle. A one-row
                // matmul is a weight stream that reaches the card's read ceiling only on all 32 cores;
                // the measured table is on the fill. Where `out` alone already fills the cores (q/o/
                // down/lm_head at granite-8b) the fill returns `(out, 1)` and the split is unchanged.
                //
                // The reduction's stick basis is the `in` dim's OWN format (128 for an fp8 operand, 64
                // for fp16) — the same rule the prefill branch's whole-stick clamp states — so an fp8
                // `in` slice is whole fp8 sticks by construction.
                let in_stick_elems = dims
                    .iter()
                    .find(|d| d.name == InAxis::NAME)
                    .map(|d| d.df.elems_per_stick())
                    .unwrap_or(FP16_ELEMS_PER_STICK);
                let (out_split, in_split) = crate::work::one_row_core_fill(
                    out_sticks(out),
                    in_.div_ceil(in_stick_elems),
                    max_cores,
                );
                if out_split > 1 {
                    map.insert(OutAxis::NAME, out_split);
                }
                if in_split > 1 {
                    map.insert(InAxis::NAME, in_split);
                }
            }
        } else {
            // Single-stick output = the matmul-by-ones REDUCE (rmmsum / fp8 amax) — a SCRATCHY construct,
            // not a torch-spyre matmul, so no fidelity reference. It can't N-split (one stick) and must
            // NOT drop to ONE core (that mangles the coordInfo → dxp l3_lx_input2 / out_of_range), so it
            // keeps a plain `mb` core_split.
            let mb_split = core_split(mb, max_cores);
            if mb_split > 1 {
                map.insert(MbAxis::NAME, mb_split);
            }
        }
    } else {
        // batched (attention bmm, y>1): WHICH axes divide is the policy's — see [`BatchedSplit`].
        let s = match batched_split {
            // `CoreSplit` is 2-D; the joint cost-model split (it splits the batch).
            BatchedSplit::JointCostModel => matmul_cost_split(y, mb, out, in_, max_cores),
            BatchedSplit::BatchDividesFirst => {
                let ys = core_split(y, max_cores);
                let mut s = matmul_cost_split(1, mb, out, in_, (max_cores / ys).max(1));
                s.b = ys;
                s
            }
        };
        for (name, v) in [
            (YAxis::NAME, s.b),
            (MbAxis::NAME, s.m),
            (OutAxis::NAME, s.n),
            (InAxis::NAME, s.k),
        ] {
            if v > 1 {
                map.insert(name, v);
            }
        }
    }
    if trace {
        eprintln!(
            "[sdsc-split] mb={mb:<4} out={out:<6} in={in_:<6} y={y:<3} -> {}",
            if map.is_empty() {
                "UNSPLIT (1 core)".to_string()
            } else {
                map.iter()
                    .map(|(k, v)| format!("{k}={v}"))
                    .collect::<Vec<_>>()
                    .join(" ")
            }
        );
    }
    map
}
