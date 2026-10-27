//! `TileOp` — the ONE declaration every SDSC-emitting op builds, so its resident-LX-fit and
//! core-split proof (already `m`-agnostic in [`WorkPlan`]/[`time_tile_for_lx`](crate::superdsc_opspec))
//! is reached through a single dispatch, not a hand-picked closure per call site.
//!
//! What already existed and is CORRECT: [`WorkPlan::divide`] (core-split, proof (b)) and
//! [`WorkPlan::time_tile_for_lx`] (LX-fit time-tiling) do not branch on `m` — decode (m=1) and a
//! wide prefill run the SAME code, differing only in the `extent` fed in. The bugs that ate four
//! weeks were NOT in that core; they were in the layer above it, where each `assemble_*` call site
//! hand-wrote its OWN residency formula (`matmul_lx_resident`, `pointwise_lx_resident`) and its OWN
//! `Vec<ItDim>`, and — critically — a SEPARATE op-decomposition fork per `m` (a whole different set
//! of ops emitted for `rows>1`, e.g. `assemble_rmsnorm_mq` vs `assemble_rmsnorm`).
//!
//! `TileOp` does not replace `WorkPlan`; it is the thing that DECLARES what `WorkPlan` tiles, in one
//! typed value derived from each operand's [`StickLayout`] (the layout SSOT), so:
//!   1. the residency formula is picked BY THE OP KIND, not re-derived per call site (killing the
//!      "two hand-rolled residency formulas that could silently diverge" class), and
//!   2. an op that declares a `TileOp` cannot ALSO fork its own decomposition on `m` inside the same
//!      function without that fork being visible as a second `TileOp` — it becomes a decision that
//!      has to be made explicit, not smuggled into a `rows > 1` branch mid-emit.
//!
//! `TileOp::tile()`'s actual orchestration (core-split then LX-fit) lives in
//! `crate::ir::bridge::tile_op_tiled_op` (the tiler, Bridge 2) — this method is a thin, call-site-
//! preserving wrapper so every existing `some_tile_op.tile(...)` call in the emitter is unaffected
//! by the module split.

use super::tiled_op::TiledOp;
use crate::sdsc_abstract::StickLayout;
use crate::superdsc_opspec::{Df, FP16_BYTES, ItDim, MaxCores, WorkPlan};

/// The three op-kinds whose LX residency this crate hand-rolls today. Every new op kind must be added
/// here (exhaustive match in [`TileOp::resident_bytes`]) — there is no `_ => `, so a fourth residency
/// formula cannot be silently bolted on without going through the same proof discipline as these
/// three.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TileOpKind {
    /// A `[mb,in]·[in,out]→[mb,out]` matmul (optionally per-batch `y`, optionally K-split).
    Matmul,
    /// A pointwise or reduce over `[mb,out,y]` with `n_operands` live tensors (inputs + the output;
    /// a reduce whose accumulator is ON-CHIP — the fused unseeded form — is one of them, matching the
    /// existing `pointwise_lx_resident`).
    PointwiseOrReduce { n_operands: u32 },
    /// A SEEDED reduce over `[mb,out,y]`: `data` active at the tiled width, plus an `out`-axis
    /// RedStick accumulator that is ONE STICK PER ROW whatever the trip — the DDL's seeded
    /// convention (`sum`/`max`/`mean`, `ddl_input_arity() = None`), which binds data + accumulator
    /// as `inputLabeledDs[0..1]`, both HBM-resident. The accumulator's residency is its true
    /// one-stick-per-row size, never the data's per-trip width — see
    /// [`seeded_reduce_lx_resident_generic`].
    SeededReduce,
    /// ⭐⭐⭐ WALL 12 — the NONSTICK seeded reduce over `[out,y]`: the REDUCED axis is the
    /// non-stick `out` dim (this crate's reduce vocabulary names the reduced dim `out`, the
    /// DDL's own `%reduce_dim:4` "OUT dimension") and the SURVIVORS are the stick `y` dim (the
    /// DDL's `%outer_dim#0` lane axis). The accumulator is the survivor vector — the lanes the
    /// PE accumulators hold — not one stick per row. The template's OWN loop is the time-tiling:
    /// `summeanmaxexx2.ddl` chunks the reduce dims and reads the HBM accumulator only on
    /// second-and-later reduce-dim chunk trips (`:159-196`, `:582-614`), so the tiled dim is the
    /// REDUCED `out`, exactly as the stick form tiles its reduced stick. See
    /// [`nonstick_reduce_lx_resident_generic`].
    NonstickReduce,
}

/// One op's full tiling declaration: its iteration domain (the `WorkPlan::divide` input) plus enough
/// of its shape (`df`, `kind`) to pick the RIGHT residency formula — no per-call-site closure.
/// `dims`/`df` are handed straight to [`WorkPlan::divide`]/`time_tile_for_lx`, so a `TileOp` is a
/// thin, checked FRONT END to machinery that already carries the core-split and LX-fit proofs.
pub struct TileOp {
    pub kind: TileOpKind,
    pub dims: Vec<ItDim>,
    /// The stick (device format) of the OUTPUT/tiled dim — drives the `time_tile_for_lx` stick-count
    /// search. Matches `StickLayout::df` of the tensor whose `out`/N dim this op tiles.
    pub df: Df,
}

impl TileOp {
    /// Build a `TileOp` from an operand's [`StickLayout`] plus the extra dims a matmul/reduce needs
    /// beyond the layout's own `(rows, cols)` — e.g. a matmul's `in`(K)/`out`(N) or a batched
    /// attention's `y`(heads). `sl` supplies `df` (the stick basis) directly — one less place for the
    /// stick width to be hand-copied and drift from the tensor's real device format.
    pub fn from_layout(kind: TileOpKind, sl: &StickLayout, dims: Vec<ItDim>) -> Self {
        TileOp {
            kind,
            dims,
            df: sl.df,
        }
    }

    /// The per-core LX residency for THIS op's kind, at a given `out_per_time` (the same shape
    /// `time_tile_for_lx` passes to `resident_bytes_fn`). Dispatches to the existing, unchanged
    /// `matmul_lx_resident`/`pointwise_lx_resident` math — see their Kani-proven equivalence below;
    /// this function is the SOLE place an op kind is mapped to its formula, replacing the per-call-
    /// site choice of which closure to write.
    pub fn resident_bytes(&self, plan: &WorkPlan, out_per_time: u32) -> u64 {
        match self.kind {
            TileOpKind::Matmul => matmul_lx_resident_generic(plan, out_per_time, self.df),
            TileOpKind::PointwiseOrReduce { n_operands } => {
                pointwise_lx_resident_generic(plan, n_operands as u64, out_per_time, self.df)
            }
            TileOpKind::SeededReduce => {
                seeded_reduce_lx_resident_generic(plan, out_per_time, self.df)
            }
            TileOpKind::NonstickReduce => {
                nonstick_reduce_lx_resident_generic(plan, out_per_time, self.df)
            }
        }
    }

    /// Run the full tiler (Bridge 2: `crate::ir::bridge::tile_op_tiled_op`) over THIS declaration's
    /// dims/kind. Thin wrapper — kept here, not inlined at call sites, so every existing
    /// `some_tile_op.tile(...)` call in the emitter is unaffected by where the orchestration lives.
    pub fn tile<const N: u32>(
        &self,
        budget: MaxCores<N>,
        splitter: impl Fn(&[ItDim], u32) -> std::collections::BTreeMap<&'static str, u32>,
        tiled_dim: &'static str,
    ) -> Result<TiledOp, crate::superdsc_error::SuperDscError> {
        crate::ir::bridge::tile_op_tiled_op::run_tiler(self, budget, splitter, tiled_dim)
    }
}

/// THE matmul per-core LX residency formula — `lower_subtile_tape_to_superdsc::matmul_lx_resident` is
/// now a thin wrapper calling this (one implementation, not two proven-equal copies). `pub` so
/// that wrapper (and any future direct caller) can reach it without going through a `TileOp` value.
/// (It read `pub(crate)` while the wrapper was in the same crate; the wrapper is now across the
/// `ktir-superdsc` boundary, so the SAME intent spells `pub`. Nothing new is exposed that was
/// deliberately sealed — this doc already stated reachability was the point.)
/// `per_core_extent`/`extent` are already `m`-agnostic (no `rows>1` branch) — this is the SAME formula
/// that existed before `TileOp`, just no longer duplicated at the two call sites that needed it.
/// `operand_df` is the RESIDENCY format of A and W — deliberately NOT the stick basis carried by the
/// dims. Those are different facts, and an fp8 W8A8 matmul is where they part: its geometry is built
/// `::<Fp16>` on purpose (the OUTPUT is fp16 and the KERNEL's N-stick is 64; 128 is only the fp8
/// activation's K-stick, the reduction axis, never split) while A and W are physically 1 byte.
/// Reading the width off the stick dim charged every fp8 matmul DOUBLE, and granite-3.1-8b's
/// down_proj (k=12800, per-core mb=21) came out at 21*12800*2 + 12800*64*2 + 21*64*2 = 2,178,688 B
/// against a 1,677,721 B budget even tiled to one out-stick — a refusal that reads as "needs K-time
/// PSUM accumulation" but is pure over-counting: the true fp8 residency is 1,090,688 B, which fits
/// with a third of LX spare. The OUTPUT stays FP16_BYTES; an fp8 matmul accumulates into fp16.
pub fn matmul_lx_resident_generic(plan: &WorkPlan, out_per_time: u32, operand_df: Df) -> u64 {
    let per_core_mb = plan.per_core_extent("mb") as u64;
    let k = plan.per_core_extent("in") as u64;
    // ⭐⭐⭐ THE BATCH IS BILLED AT THE PER-CORE EXTENT — the same law
    // [`pointwise_lx_resident_generic`] states for its own sprays ("a core that took the sprays
    // as its share of the cores holds one"): the work-division MAY split the batch (`y`) across
    // cores — the joint cost model's `s.b`, on-hardware-proven at mq==1 (`{mb:1, y:4}` on 4
    // cores) — and a core holding `y/split` batch planes residents exactly that many, in A, W and
    // OUT alike (the attention bmm's per-batch KERNEL is `[y,in,out]`, one plane per batch
    // entry). Billing the FULL extent charged every core the whole batch's planes — MEASURED at
    // the spliced golden's rows-are-requests score dot (mb=4, out=64, in=64, y=256 split 8):
    // `(1·64 + 64·64 + 1·64)·256·2 = 2,162,688 B` against the 1,677,721-B budget, a refusal
    // reading "needs K-time PSUM accumulation (Stage 2)" for a shape whose true per-core
    // residency is 270,336 B — a sixth of the budget. The same false-refusal class as the fp8
    // word-length over-charge this formula's own doc records.
    //
    // ⭐ MONOTONE IN THE SAFE DIRECTION: `per_core_extent ≤ extent` ALWAYS (it is
    // `extent / split.max(1)`, floored at 1), so the per-core batch read is ≤ the full-extent
    // read it replaces — the charge can only DROP, never rise. No plan that fit before can
    // newly fail; the fix can only un-refuse a shape or shrink a time factor, never mint one.
    // Where the batch is UNSPLIT (`per_core_extent == extent`, every shipped bmm — the known
    // 576-KiB fixture's y=16 under an mb/out split), the charge is byte-identical.
    let batch = plan
        .per_core_extent("x")
        .max(plan.per_core_extent("y"))
        .max(1) as u64;
    let opt = out_per_time as u64;
    let operand_bytes = operand_df.word_length() as u64;
    let a = per_core_mb * k * batch * operand_bytes;
    let w = k * opt * batch * operand_bytes;
    let o = per_core_mb * opt * batch * FP16_BYTES;
    a + w + o
}

/// THE pointwise/reduce per-core LX residency formula — `lower_subtile_tape_to_superdsc::
/// pointwise_lx_resident` is now a thin wrapper calling this. `pub` for the same reason as
/// [`matmul_lx_resident_generic`].
pub fn pointwise_lx_resident_generic(
    plan: &WorkPlan,
    n_operands: u64,
    out_per_time: u32,
    df: Df,
) -> u64 {
    let per_core_mb = plan.per_core_extent("mb") as u64;
    let opt = out_per_time as u64;
    // ⭐ THE ROW-SPLIT (Y) MODE'S SPRAY CHARGE — the spray dim is a real iteration extent (the
    // TileOp's `y`), so a core whose slice leaves it unsplit holds ALL its sprays' rows: the
    // same batch charge the matmul formula makes for its own batch. Billed at the PER-CORE
    // extent — a core that took the sprays as its share of the cores holds one — and a no-op at
    // `y == 1` (every non-row-split op: `per_core_extent` floors at 1), so no existing charge
    // moves. The y-BROADCAST operand itself does not range over `y` (its scale is `RedNonStick`
    // there), so this over-charges it by its spray count — the safe direction, the same
    // conservatism the matmul formula applies to its own batch operands.
    let sprays = plan.per_core_extent("y") as u64;
    n_operands * per_core_mb * opt * sprays * df.word_length() as u64
}

/// THE seeded-reduce per-core LX residency formula — the DATA operand at its per-trip width,
/// plus the ACCUMULATOR at its true ONE-STICK-PER-ROW size. The old model billed the accum at
/// the data's width (the pointwise formula with `n_operands = 2`), which over-charges exactly
/// when the row is wide: the seeded accum is `[mb, RedStick out, y]` — ONE reduced stick per
/// row, `elems_per_stick` elements — never `out_per_time` of them. Measured at the width that
/// broke: gemma-4's composite argmax max32 (m=1, vocab 262144, fp32) billed
/// `2·262144·4 = 2,097,152 B` against the 1,677,721-B usable LX, so the tiler split the reduce
/// into time=2 — and both trips wrote the SAME address-stable accum stick (a RedStick scale is
/// stride-exempt in `rewrite_op_for_time_tile`), the alias the bake's `tiled_trips_alias`
/// guard refuses. The true charge is `262144·4 + 32·4 = 1,048,704 B`: ONE trip, the form that
/// ships. The fused unseeded reduce is NOT this kind — its accumulator is an on-chip
/// `ddl.internal_tensor`, correctly billed by `PointwiseOrReduce { n_operands: 1 }`.
///
/// SAFETY BOUND (proven below): the tiler splits the STICK dim, so every candidate
/// `out_per_time` is a whole stick of the op's own format — hence `elems_per_stick ≤
/// out_per_time`, and the true charge never EXCEEDS the old two-operand charge. A seeded
/// reduce that fit before still fits, byte-identically; only a reduce the old model
/// over-billed past the budget gains back a trip count it never truly needed.
pub fn seeded_reduce_lx_resident_generic(plan: &WorkPlan, out_per_time: u32, df: Df) -> u64 {
    let per_core_mb = plan.per_core_extent("mb") as u64;
    let opt = out_per_time as u64;
    let word = df.word_length() as u64;
    per_core_mb * opt * word + per_core_mb * df.elems_per_stick() as u64 * word
}

/// ⭐⭐⭐ WALL 12 — THE NONSTICK seeded reduce's per-core LX residency, the survivor-lane mirror
/// of [`seeded_reduce_lx_resident_generic`]. The stick form charges `data(rows × per-trip
/// reduced width) + accum(rows × ONE stick)`; the nonstick form swaps the roles — the SURVIVORS
/// are the lanes, so a core holds `data(reduced-chunk × its survivor share) + accum(its survivor
/// share)` — the accumulator IS the survivor vector the PE accumulators write, one element per
/// lane, whole sticks because [`nonstick_geometry`] only fires on a whole-stick survivor width
/// and `distribute_cores` splits a stick dim by stick count.
///
/// The per-trip reduced chunk (`out_per_time`, the tiled dim being the REDUCED `out`) is the
/// template's own reduce-dim chunking — `summeanmaxexx2.ddl` reads the HBM accumulator only on
/// second-and-later reduce-dim chunk trips (`:159-196`, `:582-614`), which is exactly a
/// time-tiled reduce carrying its accum in HBM between trips.
pub fn nonstick_reduce_lx_resident_generic(plan: &WorkPlan, out_per_time: u32, df: Df) -> u64 {
    // The core's SURVIVOR share (the reduced `out` axis is never core-split — it is the
    // accumulation axis; splitting it would need the psum ring the nonstick arm does not bind —
    // and its per-trip extent IS `out_per_time`, the tiled dim).
    let per_core_y = plan.per_core_extent("y") as u64;
    let word = df.word_length() as u64;
    // The data this trip walks: every survivor lane of the core reads its `out_per_time` run of
    // the reduced axis (row-major: a reduced-axis chunk is `out_per_time` CONTIGUOUS survivor
    // vectors). The accum: the core's survivor lanes, once.
    out_per_time as u64 * per_core_y.max(1) * word + per_core_y.max(1) * word
}

#[cfg(test)]
mod matmul_residency_tests {
    use super::matmul_lx_resident_generic;
    use crate::superdsc_opspec::{Df, Fp16, MAX_CORES, MaxCores, StickExtent, WorkPlan};

    /// The LIVE plan for a batched matmul — the production dim vocabulary and the joint cost
    /// model's own splitter, so the split under test is the one the emitter derives, never a
    /// hand-picked map.
    fn batched_plan(m: u32, n: u32, k: u32, batch: u32) -> WorkPlan {
        let n_ext = StickExtent::<Fp16>::new(n).expect("whole-stick N");
        let k_ext = StickExtent::<Fp16>::new(k).expect("whole-stick K");
        let dims = crate::ir::bridge::tiled_op_sdsc_op::matmul::dims::matmul_dims::<Fp16>(
            m, &n_ext, &k_ext, batch,
        );
        WorkPlan::divide(
            &dims,
            MaxCores::<MAX_CORES>,
            crate::ir::bridge::tiled_op_sdsc_op::matmul::dims::matmul_split_map,
        )
        .expect("the cost model divides every batched shape")
    }

    /// ⭐⭐⭐ A BATCH SPLIT ACROSS CORES BILLS EACH CORE ITS OWN PLANES — the golden's own
    /// numbers, verbatim from the refusal that motivated the law. The rows-are-requests score
    /// dot states `mb=4, out=64, in=64, y=256`; the joint cost model's split is `y=8 × mb=4`
    /// (32 cores), and a core holding 32 of the 256 batch planes residents exactly those:
    /// `(1·64 + 64·64 + 1·64) · 32 · 2 = 270,336 B` — a sixth of the budget, one trip. The
    /// full-extent read this replaces charged `(…) · 256 · 2 = 2,162,688 B` and refused with
    /// "needs K-time PSUM accumulation (Stage 2)" — the same false-refusal class as the fp8
    /// word-length over-charge the formula's own doc records.
    #[test]
    fn a_batch_split_across_cores_bills_each_core_its_own_planes() {
        let plan = batched_plan(4, 64, 64, 256);
        assert_eq!(
            plan.split_of("y"),
            8,
            "the joint cost model splits the batch"
        );
        assert_eq!(plan.split_of("mb"), 4, "and the rows — 32 cores in all");
        assert_eq!(
            plan.per_core_extent("y"),
            32,
            "each core holds 32 of the 256 planes"
        );
        assert_eq!(
            matmul_lx_resident_generic(&plan, 64, Df::Fp16),
            270_336,
            "the per-core charge: (1·64 + 64·64 + 1·64) · 32 planes · 2 B"
        );
        let tt = plan
            .time_tile_for_lx(|p, o| matmul_lx_resident_generic(p, o, Df::Fp16), "out")
            .expect("a sixth of the budget is not a Stage-2 shape");
        assert!(
            tt.is_none(),
            "one trip — no time-tile, no reduction split, no Stage-2"
        );
    }

    /// ⭐ THE CONTROL — an UNSPLIT batch bills its full extent, byte-identical to every shipped
    /// bmm: the known-good fixture's shape (M384 N384 K64 batch16 under an mb16×out2 split)
    /// keeps its 589,824-B charge, `per_core_extent == extent` where no split divides the
    /// batch.
    #[test]
    fn an_unsplit_batch_bills_its_full_extent() {
        let plan = batched_plan(384, 384, 64, 16);
        assert_eq!(
            plan.split_of("y"),
            1,
            "the fixture's split divides mb and out only"
        );
        assert_eq!(plan.per_core_extent("mb"), 24);
        assert_eq!(
            matmul_lx_resident_generic(&plan, 192, Df::Fp16),
            589_824,
            "the known bmm's charge, unchanged: (24·64 + 64·192 + 24·192) · 16 · 2"
        );
    }

    /// A batched matmul plan with a STATED y split (the kani proofs' own construction): the
    /// split map is declared, not searched — the law under test is the plan's READ of a
    /// stated split, not the cost model's choice of one.
    fn stated_split_plan(mb: u32, out: u32, k: u32, y: u32, y_split: u32) -> WorkPlan {
        let dim = |name: &'static str, size: u32, is_reduction: bool, is_stick: bool| {
            crate::superdsc_opspec::ItDim {
                name,
                size,
                is_reduction,
                is_stick,
                df: Df::Fp16,
            }
        };
        let dims = vec![
            dim("mb", mb, false, false),
            dim("out", out, false, true),
            dim("in", k, true, true),
            dim("y", y, false, false),
        ];
        WorkPlan::divide(&dims, MaxCores::<MAX_CORES>, move |_, _| {
            let mut m: std::collections::BTreeMap<&'static str, u32> = Default::default();
            m.insert("y", y_split);
            m
        })
        .expect("a stated split within the core budget divides")
    }

    /// ⭐⭐⭐ THE SPLIT WIRING, ENUMERATED — the wiring half of the per-core batch law (the
    /// kani harness `matmul_resident_bills_the_batch_the_plan_states` carries the arithmetic
    /// half at every shape): for every split the matmul splitters can rule — the loop runs
    /// to the splitters' OWN bound, `MAX_CORES` (the joint arm's `divisors_upto(y,
    /// max_cores)` and the batch-first arm's `core_split(y, max_cores)` both rule up to 32:
    /// y=256 → 32) — and every batch it can divide up to 256, a REAL `WorkPlan::divide`
    /// plan reads
    /// `per_core_extent("y") == y / y_split` — and the formula charges exactly the per-core
    /// reference through it. A divided plan cannot cross CBMC's symex (the split map's
    /// BTreeMap rebalancing does not concretize), so the split's own read is pinned HERE,
    /// over the whole ruled vocabulary — the 270,336-B pin being the golden's own point in
    /// this grid. This test is the kani-red-on-parent for the 7ce19a3f9 fix: on the
    /// pre-fix formula (the batch read at its FULL extent) every row fails by exactly the
    /// split factor.
    #[test]
    fn the_split_wiring_bills_the_per_core_batch_at_every_ruled_split() {
        for y_split in 2..=MAX_CORES {
            for y in (y_split..=256u32).step_by(y_split as usize) {
                let plan = stated_split_plan(4, 64, 64, y, y_split);
                assert_eq!(
                    plan.per_core_extent("y"),
                    y / y_split,
                    "y={y} split {y_split} ways: each core holds y/y_split planes"
                );
                assert_eq!(plan.per_core_extent("mb"), 4, "the rows are unsplit");
                assert_eq!(plan.per_core_extent("in"), 64, "the contraction is unsplit");
                let charge = matmul_lx_resident_generic(&plan, 64, Df::Fp16);
                let reference = (4 * 64 + 64 * 64 + 4 * 64) as u64 * (y / y_split) as u64 * 2;
                assert_eq!(
                    charge, reference,
                    "y={y} split {y_split}: the charge bills the per-core batch, not the full extent"
                );
            }
        }
    }
}

#[cfg(kani)]
mod proofs {
    use super::*;

    fn small_plan(mb: u32, out: u32, y: u32) -> WorkPlan {
        let dims = vec![
            ItDim {
                name: "mb",
                size: mb,
                is_reduction: false,
                is_stick: false,
                df: Df::Fp16,
            },
            ItDim {
                name: "out",
                size: out,
                is_reduction: false,
                is_stick: true,
                df: Df::Fp16,
            },
            ItDim {
                name: "y",
                size: y,
                is_reduction: false,
                is_stick: false,
                df: Df::Fp16,
            },
        ];
        WorkPlan::divide(&dims, MaxCores::<32>, |_, _| Default::default()).unwrap()
    }

    /// `TileOp::resident_bytes` for `PointwiseOrReduce` matches the ORIGINAL hand-rolled formula at
    /// every symbolic `(mb, out, y, n_operands, out_per_time)` — proving the dispatch move changed no
    /// emitted byte for the pointwise/reduce family, decode (`mb=1`) included.
    #[kani::proof]
    fn pointwise_resident_matches_original() {
        let mb: u32 = kani::any();
        kani::assume(mb >= 1 && mb <= 64);
        let out: u32 = kani::any();
        kani::assume(out >= 64 && out <= 8192 && out % 64 == 0);
        let n_operands: u32 = kani::any();
        kani::assume(n_operands >= 1 && n_operands <= 8);
        let out_per_time: u32 = kani::any();
        kani::assume(out_per_time >= 1 && out_per_time <= out);

        let plan = small_plan(mb, out, 1);
        let op = TileOp {
            kind: TileOpKind::PointwiseOrReduce { n_operands },
            dims: plan.dims().to_vec(),
            df: Df::Fp16,
        };
        let via_tileop = op.resident_bytes(&plan, out_per_time);
        let original =
            pointwise_lx_resident_generic(&plan, n_operands as u64, out_per_time, Df::Fp16);
        assert_eq!(via_tileop, original);
    }

    /// Same equivalence for the `Matmul` kind, including the batched-attention `y` factor.
    #[kani::proof]
    fn matmul_resident_matches_original() {
        let mb: u32 = kani::any();
        kani::assume(mb >= 1 && mb <= 64);
        let out: u32 = kani::any();
        kani::assume(out >= 64 && out <= 8192 && out % 64 == 0);
        let y: u32 = kani::any();
        kani::assume(y >= 1 && y <= 32);
        let out_per_time: u32 = kani::any();
        kani::assume(out_per_time >= 1 && out_per_time <= out);

        let plan = small_plan(mb, out, y);
        let op = TileOp {
            kind: TileOpKind::Matmul,
            dims: plan.dims().to_vec(),
            df: Df::Fp16,
        };
        let via_tileop = op.resident_bytes(&plan, out_per_time);
        // `Df::Fp16` because that is the `TileOp`'s own `df` above — the proof is that the dispatch
        // agrees with the generic for the SAME operand residency, not across formats.
        let original = matmul_lx_resident_generic(&plan, out_per_time, Df::Fp16);
        assert_eq!(via_tileop, original);
    }

    /// `TileOp::resident_bytes` for `SeededReduce` matches the new formula at every symbolic
    /// `(mb, out, out_per_time)` — and NEVER EXCEEDS the old two-operand pointwise charge
    /// (`2 · mb · out_per_time · word`) at the same shape, whenever `out_per_time` is a whole
    /// stick of the op's own format (the only candidate the tiler's stick-dim search produces).
    /// The charge fix can only make a seeded reduce fit MORE easily: a bundle that baked
    /// before bakes byte-identically after, and only an over-billed reduce gains back trips.
    /// Both formats asserted — fp16 (the seeded sum/mean/max family) and fp32 (the composite
    /// argmax's max32, the width that broke).
    #[kani::proof]
    fn seeded_reduce_resident_matches_formula_and_never_exceeds_the_old_charge() {
        let mb: u32 = kani::any();
        kani::assume(mb >= 1 && mb <= 64);
        let out: u32 = kani::any();
        kani::assume(out >= 64 && out <= 8192 && out % 64 == 0);
        let out_per_time: u32 = kani::any();
        kani::assume(out_per_time >= 1 && out_per_time <= out);

        let plan = small_plan(mb, out, 1);
        for df in [Df::Fp16, Df::Fp32] {
            // A whole stick of this format — the tiler's own candidate shape, and the
            // precondition of the bound (`elems_per_stick ≤ out_per_time`).
            kani::assume(out_per_time % df.elems_per_stick() == 0);
            let op = TileOp {
                kind: TileOpKind::SeededReduce,
                dims: plan.dims().to_vec(),
                df,
            };
            let via_tileop = op.resident_bytes(&plan, out_per_time);
            assert_eq!(
                via_tileop,
                seeded_reduce_lx_resident_generic(&plan, out_per_time, df)
            );
            let old = pointwise_lx_resident_generic(&plan, 2, out_per_time, df);
            assert!(
                via_tileop <= old,
                "the true charge must never exceed the old over-charge"
            );
        }
    }

    /// decode (`mb=1`) residency, for EITHER kind, is reachable through the same dispatch as any
    /// other `mb` — no special-cased "decode path" inside `resident_bytes`.
    #[kani::proof]
    fn decode_mb1_uses_same_dispatch() {
        let out: u32 = kani::any();
        kani::assume(out >= 64 && out <= 8192 && out % 64 == 0);
        let plan = small_plan(1, out, 1);
        let pw = TileOp {
            kind: TileOpKind::PointwiseOrReduce { n_operands: 2 },
            dims: plan.dims().to_vec(),
            df: Df::Fp16,
        };
        let mm = TileOp {
            kind: TileOpKind::Matmul,
            dims: plan.dims().to_vec(),
            df: Df::Fp16,
        };
        assert_eq!(
            pw.resident_bytes(&plan, out),
            pointwise_lx_resident_generic(&plan, 2, out, Df::Fp16)
        );
        assert_eq!(
            mm.resident_bytes(&plan, out),
            matmul_lx_resident_generic(&plan, out, Df::Fp16)
        );
    }

    /// A BATCHED matmul plan — the dims [`crate::ir::bridge::tiled_op_sdsc_op::matmul::dims::
    /// matmul_dims`] states (mb / out-stick / in-reduction-stick / y-batch), with the batch split
    /// `y_split` ways when `y_split > 1`. The split map is stated, not searched: the law under
    /// proof is the FORMULA's read of the plan (per-core vs full extent), not the cost model's
    /// choice of split — any split the plan really carries exercises it.
    fn batched_matmul_plan(mb: u32, out: u32, k: u32, y: u32, y_split: u32) -> WorkPlan {
        let dim = |name: &'static str, size: u32, is_reduction: bool, is_stick: bool| ItDim {
            name,
            size,
            is_reduction,
            is_stick,
            df: Df::Fp16,
        };
        let dims = vec![
            dim("mb", mb, false, false),
            dim("out", out, false, true),
            dim("in", k, true, true),
            dim("y", y, false, false),
        ];
        WorkPlan::divide(&dims, MaxCores::<32>, move |_, _| {
            let mut m: std::collections::BTreeMap<&'static str, u32> = Default::default();
            if y_split > 1 {
                m.insert("y", y_split);
            }
            m
        })
        .unwrap()
    }

    /// ⭐⭐⭐ THE PER-CORE BATCH LAW, ARITHMETIC HALF (the 7ce19a3f9 fix's own bound,
    /// PROVEN at the typed vocabulary): at every symbolic shape, the charge is exactly the
    /// per-core reference — `(per-core mb·k + k·out_per_time + per-core mb·out_per_time) ·
    /// THE BATCH THE PLAN STATES · word` — read through the REAL production formula. The
    /// plan states the batch a core HOLDS as its own `y` dim: under a y-split plan that dim
    /// is the per-core extent `y / y_split` (the wiring half, pinned at runtime — see
    /// `the_split_wiring_bills_the_per_core_batch_at_every_ruled_split`), and the formula
    /// must bill exactly it, never the full-extent batch the pre-fix formula read (the
    /// over-charge that refused the spliced golden's score dot 2,162,688 B against a true
    /// 270,336 B).
    ///
    /// WHY A REPLICATE PLAN, NOT A DIVIDED ONE — a divided plan cannot cross CBMC's symex:
    /// the split map's `BTreeMap` removal/rebalancing machinery (reached through
    /// `WorkPlan::divide`'s `retain`) does not concretize — 900+ unwind events in
    /// `correct_childrens_parent_links` alone, no verdict inside the 15-minute
    /// no-convergence rule. The same wall is why this harness carries the ARITHMETIC half
    /// and the runtime grid carries the WIRING half: where nothing splits, per-core extent
    /// IS the stated dim (so this harness exercises the formula's real read), and the
    /// split's own `per_core_extent == y / y_split` read is enumerated at runtime over the
    /// whole ruled vocabulary. The kani-red-on-parent for the fix is correspondingly the
    /// RUNTIME pin's, not this harness's: the old and new formulas agree wherever nothing
    /// splits — the defect only manifests under a divided plan, which symex cannot cross.
    /// (`a_batch_split_across_cores_bills_each_core_its_own_planes` fails at the parent
    /// with 2,162,688 B against the pinned 270,336 B.)
    ///
    /// DIVISION OF LABOR — the runtime pins carry the shipped numbers and this harness
    /// carries the structural law: the 270,336-B pin at the golden's own
    /// mb=4/out=64/in=64/y=256 under the cost model's REAL split, the 589,824-B unsplit
    /// bmm pin, and the split-wiring grid over every (y, y_split) the matmul splitters
    /// can rule (bounded by their own max_cores, `MAX_CORES`); this harness
    /// proves the formula at every shape between and around them. The vocabulary is TYPED
    /// at what the emitter really states, which is also the convergence discipline (no
    /// two-sided symbolic product survives): `mb` 1..=8 (the real rows-are-requests
    /// widths), `out` and `k` each picked from {64, 128} (the real whole-stick widths at
    /// these shapes — a CONCRETE side for every product they touch), the stated batch
    /// 1..=256 (covering both pins' batches, 16 and 256), and `out_per_time` a whole
    /// number of fp16 sticks (`sticks · 64`, the only candidate the tiler's stick-dim
    /// search produces — the same vocabulary the seeded-reduce harness assumes for its own
    /// bound). `#[kani::unwind(16)]` bounds the `memcmp` loops the `&str`-keyed dim
    /// lookups lower to (the bound's sufficiency is itself a verified unwinding
    /// assertion).
    #[kani::proof]
    #[kani::unwind(16)]
    fn matmul_resident_bills_the_batch_the_plan_states() {
        let mb: u32 = kani::any();
        kani::assume(mb >= 1 && mb <= 8);
        let out_wide: bool = kani::any();
        let out: u32 = if out_wide { 128 } else { 64 };
        let k_wide: bool = kani::any();
        let k: u32 = if k_wide { 128 } else { 64 };
        // The batch a core HOLDS — under a y-split plan this is the per-core extent
        // `y / y_split`; the harness states it as the plan's own y dim, which is the value
        // the formula must bill.
        let pc_batch: u32 = kani::any();
        kani::assume(pc_batch >= 1 && pc_batch <= 256);
        let opt_sticks: u32 = kani::any();
        kani::assume(opt_sticks >= 1 && opt_sticks <= out / 64);
        let out_per_time = opt_sticks * 64;

        let dim = |name: &'static str, size: u32, is_reduction: bool, is_stick: bool| ItDim {
            name,
            size,
            is_reduction,
            is_stick,
            df: Df::Fp16,
        };
        let dims = vec![
            dim("mb", mb, false, false),
            dim("out", out, false, true),
            dim("in", k, true, true),
            dim("y", pc_batch, false, false),
        ];
        // REPLICATE — no splits: every per-core extent IS its dim's stated size, so the
        // stated y dim is exactly the per-core batch the split wiring hands the formula.
        let plan = WorkPlan::replicate(&dims);
        assert_eq!(plan.per_core_extent("y"), pc_batch);
        assert_eq!(plan.per_core_extent("mb"), mb);
        assert_eq!(plan.per_core_extent("in"), k);

        let charge = matmul_lx_resident_generic(&plan, out_per_time, Df::Fp16);
        let (pc_mb, pc_k, opt) = (mb as u64, k as u64, out_per_time as u64);
        let reference = (pc_mb * pc_k + pc_k * opt + pc_mb * opt) * pc_batch as u64 * 2;
        assert_eq!(charge, reference);
    }

    /// ⭐ THE UNSPLIT EQUALITY — the fix's byte-identity half: where NO split divides the batch
    /// (every shipped bmm — the known 589,824-B fixture's y=16 under an mb/out split), the new
    /// per-core batch read and the OLD full-extent read charge the SAME bytes at every symbolic
    /// shape. The old formula is spelled inline (the full-extent batch read it was) so the
    /// equality is between two formulas, not a formula and itself. Same typed vocabulary and
    /// division of labor as the split harness above — the pins carry 589,824, this carries the
    /// equality at every unsplit shape around it.
    #[kani::proof]
    #[kani::unwind(16)]
    fn matmul_resident_is_byte_identical_where_the_batch_is_unsplit() {
        let mb: u32 = kani::any();
        kani::assume(mb >= 1 && mb <= 8);
        let out_wide: bool = kani::any();
        let out: u32 = if out_wide { 128 } else { 64 };
        let k_wide: bool = kani::any();
        let k: u32 = if k_wide { 128 } else { 64 };
        let y: u32 = kani::any();
        kani::assume(y >= 1 && y <= 256);
        let opt_sticks: u32 = kani::any();
        kani::assume(opt_sticks >= 1 && opt_sticks <= out / 64);
        let out_per_time = opt_sticks * 64;

        // No split at all — the plan every unbatched-scope bmm runs under.
        let plan = batched_matmul_plan(mb, out, k, y, 1);
        assert_eq!(plan.per_core_extent("y"), y, "the batch is unsplit");

        let new = matmul_lx_resident_generic(&plan, out_per_time, Df::Fp16);
        // The pre-7ce19a3f9 formula: the same per-core mb/k/opt reads, the batch at its FULL
        // extent. Where nothing splits, per-core == extent for every dim, so the two must agree.
        let (pc_mb, pc_k, opt) = (mb as u64, k as u64, out_per_time as u64);
        let old = (pc_mb * pc_k + pc_k * opt + pc_mb * opt) * y as u64 * 2;
        assert_eq!(new, old);
    }
}
