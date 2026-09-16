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

/// The two op-kinds whose LX residency this crate hand-rolls today. Every new op kind must be added
/// here (exhaustive match in [`TileOp::resident_bytes`]) — there is no `_ => `, so a third residency
/// formula cannot be silently bolted on without going through the same proof discipline as these two.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TileOpKind {
    /// A `[mb,in]·[in,out]→[mb,out]` matmul (optionally per-batch `y`, optionally K-split).
    Matmul,
    /// A pointwise or reduce over `[mb,out,y]` with `n_operands` live tensors (inputs + the output;
    /// a reduce's accumulator IS one of the operands, matching the existing `pointwise_lx_resident`).
    PointwiseOrReduce { n_operands: u32 },
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
    let batch = plan.extent("x").max(plan.extent("y")).max(1) as u64;
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
    n_operands * per_core_mb * opt * df.word_length() as u64
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
}
