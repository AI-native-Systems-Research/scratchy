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
// ⚠️ SCOPED SURVIVOR, AND IT IS A PERFORMANCE GATE — NOT A SEMANTIC AUTHORITY.
//
// The row KIND now lives in a type (`sdsc_abstract::QueryRows<ROWS_ARE_REQUESTS>`) and is threaded
// through the signatures that make SEMANTIC decisions (`lower_rope_node`, `lower_attn_node`). What
// remains here is one PERF/COMPATIBILITY choice: "a decode batch must not split `mb`", whose purpose is
// to leave prefill's hardware-proven bundles byte-identical while stopping a decode batch from reloading
// the stationary weight per split. It is not a correctness fact about what a row means.
//
// ⛔ IT IS STILL AMBIENT, AND THAT IS A DEBT, NOT A DESIGN. `matmul_split_map` is consumed as an
// `impl FnOnce(&[ItDim], u32)` by `TileOp::tile`, and its two call sites in `matmul/opspec.rs` do not
// receive the kind — threading it there touches every matmul signature in the model, which is the
// "four signatures" the original comment declined to thread and is a separate, larger change.
//
// ⏭ THE FIX, and it is a CONST GENERIC one: this crate is a PROC-MACRO compiler. `codegen.rs` reads
// `head_dim`, the head counts and the rung widths from the model config and interpolates them as
// LITERALS, so every one of them — and the row kind with them — is a compile-time constant. The splitter
// should be `matmul_split_map::<const ROWS_ARE_REQUESTS: bool>(dims, cores)` and be passed as
// `matmul_split_map::<RAR>` from a call chain generic over the same const. Then the two split policies
// are two instantiations rather than one runtime branch, and no ambient value can exist to be stale.
thread_local! {
    static SPLIT_MB_FORBIDDEN: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Whether the bundle being lowered forbids an `mb` split (a decode batch does). PERF GATE ONLY — see
/// the note above; do not reintroduce semantic decisions on top of this.
pub(crate) fn split_mb_forbidden() -> bool {
    SPLIT_MB_FORBIDDEN.with(|c| c.get())
}

/// Set for one bundle's lowering; returns the previous value so the caller restores it.
pub fn set_split_mb_forbidden(v: bool) -> bool {
    SPLIT_MB_FORBIDDEN.with(|c| c.replace(v))
}

/// ⭐ THE SPLIT POLICY AS A CONST GENERIC — two instantiations, not one runtime branch.
///
/// `FORBID_MB_SPLIT` is a compile-time constant of the bundle (a decode batch forbids it; a prefill chunk
/// does not), so the two policies are now separate specialisations that the compiler can see. Only the
/// SELECTION between them is still ambient (see [`split_mb_forbidden`]), and that is the remaining debt:
/// the selection belongs in the call chain as `matmul_split_map::<RAR>`, which needs the kind threaded
/// through `matmul/opspec.rs`'s two call sites and their 28 callers.
pub(crate) fn matmul_split_map_gated<const FORBID_MB_SPLIT: bool>(
    dims: &[crate::superdsc_opspec::ItDim],
    max_cores: u32,
) -> std::collections::BTreeMap<&'static str, u32> {
    matmul_split_map_inner(
        dims,
        max_cores,
        FORBID_MB_SPLIT,
        BatchedSplit::JointCostModel,
    )
}

/// ⛔ THE ADAPTER, AND THE ONLY REMAINING AMBIENT READ IN THE SPLIT PATH. It exists because
/// `TileOp::tile` consumes the splitter as a bare `impl FnOnce(&[ItDim], u32)` from call sites that were
/// never given the row kind. One line, named, and greppable — not a fact spread across the file.
pub fn matmul_split_map(
    dims: &[crate::superdsc_opspec::ItDim],
    max_cores: u32,
) -> std::collections::BTreeMap<&'static str, u32> {
    if split_mb_forbidden() {
        matmul_split_map_gated::<true>(dims, max_cores)
    } else {
        matmul_split_map_gated::<false>(dims, max_cores)
    }
}

/// The REQUEST-BATCH splitter: identical to [`matmul_split_map`] on every dim set without a real
/// batch axis (including the ambient mb gate), and batch-divides-first on the y>1 arm. Handed out
/// only by the request regime's [`super::walk::RungRegime::split_map`].
pub(crate) fn matmul_split_map_batch_requests(
    dims: &[crate::superdsc_opspec::ItDim],
    max_cores: u32,
) -> std::collections::BTreeMap<&'static str, u32> {
    matmul_split_map_inner(
        dims,
        max_cores,
        split_mb_forbidden(),
        BatchedSplit::BatchDividesFirst,
    )
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
    forbid_mb_split: bool,
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
    // A DECODE BATCH MUST NOT SPLIT `mb`. The weight is STATIONARY in the PT array and M streams
    // through it, so the weight load is a FIXED cost amortized over the rows — which is the whole
    // reason to batch. An `mb` split hands every one of those cores the SAME weight (see the HW CAP
    // note below), so it reloads once per split and the amortization is cancelled.
    //
    // MEASURED: prefill reaches ~6 TFLOP/s on these matmuls at mb=71; a decode batch gets ~1.4 at
    // mb=8, and per-row cost stays FLAT from 8 to 32 rows instead of falling — because the planner
    // starts splitting `mb` at 16 (`in=2 mb=2 out=8`). Batching stops paying exactly there.
    // `shared_weight=true` is why the cost model does not see it: it says the weight is loaded once
    // and broadcast across M, so an `mb` split looks free.
    //
    // Only the decode batch, because prefill already runs at 4x this efficiency and its bundles are
    // proven on hardware; this must leave every one of them byte-identical.
    let batched_decode = forbid_mb_split;
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
                let s = matmul_split_plan(mb, out, in_, 1, out_stick_elems, max_cores, true);
                let mut s = s;
                // NO K-SPLIT FOR A DECODE BATCH. Splitting the reduction dim makes `k` cores each
                // contract a slice of K into a PARTIAL product that dxp PSUM-accumulates into the
                // SHARED output tile — cross-core accumulation and the traffic that carries it, on
                // every matmul in the layer.
                //
                // It is the one structural difference between the configuration that is fast per row
                // and the one that is not: at one request the planner picks `out=32` and no K-split
                // (25.6 ms, essentially the weight stream alone); from two requests up it picks
                // `in=4 out=8` (47 ms at eight, and the extra scales with rows while being invisible
                // to op count, launch count, the mb-split and barriers — all measured).
                //
                // `out` alone fills the cores here: the projections are 2048 and 8192 wide, which is
                // 16 and 64 fp8 sticks against 32 cores. So hand the reduction's cores to `out` and
                // run the geometry one request already runs.
                // ⛔ HANDING CORES TO `out` MUST KEEP `out`'s SPLIT A DIVISOR OF ITS STICK COUNT.
                //
                // `free = sticks / s.n` only bounds how many MORE cores `out` could absorb; it does NOT
                // make the product legal. `s.n *= give` then produced splits that leave every core a
                // FRACTION of a stick, which dxp refuses to schedule for the whole group
                // (`DtException: There must be at least one valid candidate`, L3DlOpsScheduler:1375).
                //
                // MEASURED — this is what stopped granite-3.1-8b-FP8 from baking: its MLP `out` is
                // 12800 = 200 sticks, the cost model picked `n=8 k=4`, `give=4` made `n=32`, and
                // `200 % 32 == 8`. granite-2b's 8192 = 128 sticks with `128 % 32 == 0`, so the same code
                // was legal there BY ARITHMETIC LUCK and the fault was invisible on the model every test
                // used. Any `out` that is not a multiple of `32 * stick` trips it.
                //
                // So `give` is chosen as the LARGEST donation that still divides the stick count. The
                // batched-decode intent is unchanged (cores still move from the reduction/rows to `out`,
                // which is the whole point — one N-slice per core, weight read once); only illegal
                // donations are declined, and declining leaves the cost model's own legal split in place.
                let legal_give = |n: u32, want: u32| -> u32 {
                    let sticks = out_sticks(out);
                    let mut give = want.min((sticks / n.max(1)).max(1));
                    while give > 1 && sticks % n.saturating_mul(give) != 0 {
                        give -= 1;
                    }
                    give
                };
                if batched_decode && s.k > 1 {
                    let give = legal_give(s.n, s.k);
                    if give > 1 {
                        s.n = s.n.saturating_mul(give);
                        s.k /= give;
                    }
                }
                if batched_decode && s.m > 1 {
                    // Hand those cores to `out`, where each takes its own N-slice and the weight is
                    // read once — the geometry the 2..8 rungs already run and the one bs=1 runs.
                    let give = legal_give(s.n, s.m);
                    if give > 1 {
                        s.n = s.n.saturating_mul(give);
                        s.m /= give;
                    }
                }
                // ⛔⛔⛔ AND THE SAME LAW BINDS THE REDUCTION AXIS — `in`, not just `out`. Ported from
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
                // it obeys the same law — `DtException: There must be at least one valid candidate.`
                // (`L3DlOpsScheduler.cpp:1375`), the same fault the `out` donation above declines.
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
                // DECODE (m=1): keep the proven OUT-split (byte-identical to the coherent decode path). The
                // cost model must NOT touch decode — a K-split here would regress the working lm_head.
                let out_split = core_split(out_sticks(out), max_cores);
                if out_split > 1 {
                    map.insert(OutAxis::NAME, out_split);
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
